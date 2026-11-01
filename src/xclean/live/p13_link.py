#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.3 ENGINE SIDE: the frame process's link to the loop-closure WORKER.  The main builder's half of the P1.3 build: transport,
packets, publication.  The worker's logic (retrieval, verification, admission, head filter, merges) is NOT here: it lives
in xclean/slam/ (the other builder) and plugs in through the WORKER INTERFACE below; xclean/live/p13_stub_worker.py is a
stub that speaks the same interface.

API (the design's `link` module, section 9; the engine calls only these, and only when a link is attached):
    kf_record(eng, s, left, right, kind)     H2: a new keyframe s (kind make_kf | init | reanchor): header + the
                                             CREATION-TIME odometry edge to the segment's previous keyframe (from the
                                             latest window solve's covariance, H1) + references to the rectified pair
    edge_final(eng, s)                       H3: keyframe s leaves the window: its FINAL edge to the next keyframe
    segment_start(eng, link, s)              H4: a new segment's keyframe s: the link (kind, Z, Sigma or None)
    compose(idx, t, T_odom) -> (T_map, extra)    H5: LIVE-MAP = T_applied T_odom (+ flags); LIVE-ODOM untouched
    emit(eng)                                H6: after both publishes: images into the KfRing, messages onto the ctl
                                             queue (never blocks: ring full -> images dropped, header kept, counted)
    close(); summary()                       H7
Every function only READS engine state and COPIES what it keeps (LIVE-ODOM bit identity; the estimator never reads
anything written here).  The engine holds the module as `eng._p13` (None = loop off: no call at all).

EDGES:
    body:   Z_ab = T_wb,a^-1 T_wb,b,  Sigma = J C J^T over the window's joint covariance C (rotation right, translation
            additive in frame a; CONVENTIONS.md);
    camera: Z_c = T_bc^-1 Z_b T_bc,  Sigma_c = J Sigma_b J^T, J = [[R_bc^T, 0], [-R_bc^T R_Zb [p_bc]x, R_bc^T]].
    Messages carry the CAMERA form (the worker's chain), plus the body form for audit.

WORKER INTERFACE (for xclean/slam/worker.py; the stub implements it):
    entry point  worker_main(handles, wcfg)   (handles from make_transport(); wcfg: tag, calib, mode, out_dir, seed)
    side = WorkerSide(handles): side.ready() once warmed up (the frame process waits for it before the clock starts);
        m = side.get(timeout)   the next ctl message (dict, FIFO, None on timeout); types:
            "kf"            n (1, 2, ...), sid, fidx, t, segment, vi, g_w, kind, T_odom_cam (16, the PUBLISHED LIVE-ODOM),
                            T_wb (16), edge_prev (None | {sid_a, fidx_a, t_a, R, p, S (camera), R_body, p_body, S_body}),
                            slot (KfRing slot or None = images dropped), t_emit_frame (the stamp of the frame whose
                            publish emitted it), wall_emit (perf_counter at emission)
            "edge_final"    sid_a, fidx_a, sid_b, fidx_b, R, p, S (camera) + body form, t_emit_frame
            "segment_start" kind (unknown | coast | coast_vi | demote), segment, kf0 {sid, fidx, t}, prev {sid, fidx, t}
                            or None, R, p (prev -> kf0, camera; None without prev), S (camera relative) or None,
                            t_emit_frame
            "eos"           n_kf
        left, right = side.images(slot) (copies); side.free(slot)   (free EVERY slot it receives, after reading)
        side.publish(epoch, T_target (4x4), kind, n_loops, merge, finish_t)  a NEW correction target (head-anchored
            T_map_odom).  kind: 1 loop, 2 merge, 3 merge_revert; merge: 0 none, 1 provisional, 2 confirmed;
            finish_t = the modelled finish stamp (lock-step, frozen cost model) or the wall-clock stamp (paced).
            Lock-step: also onto the result queue (the frame side reads it); paced / live: the seqlock only (handles
            "mode"; an unread queue would fill its pipe and hold the worker at exit)
        side.progress(n_done)   after the job of packet n_done is COMPLETE (and its results published): lock-step
            determinism - the frame process, at each frame, waits until every emitted packet is done and
            applies exactly the results whose finish_t <= the frame stamp.
    The worker never writes anything else; it may keep its own files in wcfg["out_dir"].

SWITCHES (one config for all rows):
    XCLEAN_P13_OFF   '+'-list, token `loop`: no worker, no hook call, LIVE-MAP = LIVE-ODOM exactly (the P1.2 behaviour).
                     Default: loop ON.  Any value labels a run a diagnostic run (harness).
    XCLEAN_P13_WORKER  slam (default: xclean.slam.worker) | stub (this package's stub; a diagnostic run)
    XCLEAN_P13_APPLY   slew (default) | step (A/B arm, a diagnostic run);  XCLEAN_P13_SLEW_F (default 0.5, a diagnostic run when set)

SLEW: per frame the applied transform moves toward
the target by at most b_t = f max(0.10 m, 0.10 path_2s) dt / 2 s and b_r = f max(2 deg, 0.10 rot_2s) dt / 2 s, measured
at the head (the current LIVE-ODOM pose); path_2s / rot_2s = the LIVE-ODOM NET displacement |p(t) - p(t_first)| / NET
rotation angle(R(t_first)^T R(t)) over the last 2 s (t_first = the oldest published frame inside the window), i.e. the
B4 bound as score_live reads it (HOOK_SPEC S3; the summed path of the first integration was >= the net one and let the
slew outrun the B4 relative bound).  `slew_step` is a pure function (the offline replay must import this one).
"""
from __future__ import annotations

import math
import os
import time
from collections import deque
from multiprocessing import shared_memory

import numpy as np

NSLOTS = 8                       # class B: 2 waiting + 1 in work + 1 being written, x 2 margin
B4_WINDOW_S = 2.0                # the B4 relative window
B4_T_MIN_M = 0.10                # B4 translation floor
B4_T_FRAC = 0.10                 # B4 translation fraction of the window's path
B4_R_MIN_DEG = 2.0               # B4 rotation floor
B4_R_FRAC = 0.10                 # B4 rotation fraction
SYNC_TIMEOUT_S = 300.0           # lock-step: the longest wait for the worker before a frame gives up (logged)
SEQ_SPINS = 2000                 # seqlock reader: bounded retries, then the previous value

_P13_RAW = (os.environ.get("XCLEAN_P13_OFF", "") or "").strip()
if "," in _P13_RAW:
    raise ValueError("XCLEAN_P13_OFF is a '+'-separated list (loop), got %r" % _P13_RAW)
P13_OFF = set(x.strip() for x in _P13_RAW.split("+") if x.strip())
if not P13_OFF <= {"loop"}:
    raise ValueError("XCLEAN_P13_OFF: unknown component in %r" % sorted(P13_OFF))
APPLY = (os.environ.get("XCLEAN_P13_APPLY", "slew") or "slew").strip()
if APPLY not in ("slew", "step"):
    raise ValueError("XCLEAN_P13_APPLY must be slew or step, got %r" % APPLY)
SLEW_F = float(os.environ.get("XCLEAN_P13_SLEW_F", "0.5") or 0.5)
WORKER = (os.environ.get("XCLEAN_P13_WORKER", "slam") or "slam").strip()
if WORKER not in ("slam", "stub"):
    raise ValueError("XCLEAN_P13_WORKER must be slam or stub, got %r" % WORKER)
MERGE_NAMES = {0: "none", 1: "provisional", 2: "confirmed"}
KIND_NAMES = {0: "none", 1: "loop", 2: "merge", 3: "merge_revert"}


def loop_on():
    return "loop" not in P13_OFF


# =============================================================================================== small SE(3) math
def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _so3_exp(w):
    th = float(np.linalg.norm(w))
    K = _skew(w)
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + math.sin(th) / th * K + (1.0 - math.cos(th)) / th ** 2 * (K @ K)


def _so3_log(R):
    c = max(-1.0, min(1.0, 0.5 * (float(np.trace(R)) - 1.0)))
    th = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * w
    if math.pi - th < 1e-6:
        A = 0.5 * (R + np.eye(3))
        ax = np.sqrt(np.maximum(np.diag(A), 0.0))
        k = int(np.argmax(ax))
        ax = A[:, k] / max(ax[k], 1e-12)
        return th * ax / np.linalg.norm(ax)
    return th / (2.0 * math.sin(th)) * w


def rel_cov(Ra, pa, Rb, pb, Caa, Cab, Cbb):
    """Relative BODY pose a -> b and its covariance from the window's joint covariance."""
    Rab = Ra.T @ Rb
    pab = Ra.T @ (pb - pa)
    Ja = np.zeros((6, 6))
    Ja[:3, :3] = -Rab.T
    Ja[3:, :3] = _skew(pab)
    Ja[3:, 3:] = -Ra.T
    Jb = np.zeros((6, 6))
    Jb[:3, :3] = np.eye(3)
    Jb[3:, 3:] = Ra.T
    S = Ja @ Caa @ Ja.T + Ja @ Cab @ Jb.T + Jb @ Cab.T @ Ja.T + Jb @ Cbb @ Jb.T
    return Rab, pab, 0.5 * (S + S.T)


def body_to_cam(ZR, Zp, S, Rbc, pbc):
    """Z_c = T_bc^-1 Z_b T_bc and Sigma_c = J Sigma_b J^T (S may be None)."""
    ZcR = Rbc.T @ ZR @ Rbc
    Zcp = Rbc.T @ (ZR @ pbc + Zp - pbc)
    if S is None:
        return ZcR, Zcp, None
    J = np.zeros((6, 6))
    J[:3, :3] = Rbc.T
    J[3:, :3] = -Rbc.T @ ZR @ _skew(pbc)
    J[3:, 3:] = Rbc.T
    Sc = J @ S @ J.T
    return ZcR, Zcp, 0.5 * (Sc + Sc.T)


def slew_step(T_applied, T_target, T_odom, dt, path_2s, rot_2s_deg, f=SLEW_F, apply=APPLY):
    """One frame of the correction slew, PURE.  Returns (T_applied_new, pending_m, pending_deg, moved).
    The head is X = T_applied T_odom; the applied transform moves so that the head moves toward T_target T_odom by at
    most b_t (m) and b_r (rad) along the geodesic (rotation and translation by the same fraction)."""
    Xa = T_applied @ T_odom
    Xt = T_target @ T_odom
    dpos = Xt[:3, 3] - Xa[:3, 3]
    d = float(np.linalg.norm(dpos))
    w = _so3_log(Xa[:3, :3].T @ Xt[:3, :3])
    th = float(np.linalg.norm(w))
    if d == 0.0 and th == 0.0:
        return T_applied, 0.0, 0.0, False
    if apply == "step":
        frac = 1.0
    else:
        if not (dt > 0.0):
            return T_applied, d, math.degrees(th), False
        b_t = f * max(B4_T_MIN_M, B4_T_FRAC * float(path_2s)) * dt / B4_WINDOW_S
        b_r = math.radians(f * max(B4_R_MIN_DEG, B4_R_FRAC * float(rot_2s_deg)) * dt / B4_WINDOW_S)
        frac = min(1.0, (b_t / d) if d > 0.0 else 1.0, (b_r / th) if th > 0.0 else 1.0)
    if frac >= 1.0:
        return np.array(T_target, float), 0.0, 0.0, True
    Xn = np.eye(4)
    Xn[:3, :3] = Xa[:3, :3] @ _so3_exp(frac * w)
    Xn[:3, 3] = Xa[:3, 3] + frac * dpos
    Tn = Xn @ np.linalg.inv(T_odom)
    return Tn, (1.0 - frac) * d, math.degrees((1.0 - frac) * th), True


# =============================================================================================== transport
class KfRing:
    """NSLOTS keyframe image pairs (rectified left + right, uint8 H x W) in shared memory.  state[i]: 0 FREE (the frame
    process may write), 1 WRITTEN (the worker may read, then frees).  One writer per transition: no lock."""

    def __init__(self, name, W, H, nslots=NSLOTS, create=False):
        self.W, self.H, self.n = int(W), int(H), int(nslots)
        img = 2 * self.W * self.H
        size = 8 * (2 * self.n) + img * self.n
        self.shm = shared_memory.SharedMemory(name=name, create=create, size=size if create else 0)
        self.state = np.ndarray((self.n,), np.int64, self.shm.buf, 0)
        self.seq = np.ndarray((self.n,), np.int64, self.shm.buf, 8 * self.n)
        self.img = np.ndarray((self.n, 2, self.H, self.W), np.uint8, self.shm.buf, 16 * self.n)
        if create:
            self.state[:] = 0
            self.seq[:] = 0

    def write(self, left, right, n):
        free = np.nonzero(self.state == 0)[0]
        if len(free) == 0 or left is None or right is None:
            return None
        i = int(free[0])
        hh, ww = min(self.H, left.shape[0]), min(self.W, left.shape[1])
        self.img[i, 0, :hh, :ww] = left[:hh, :ww]
        self.img[i, 1, :hh, :ww] = right[:hh, :ww]
        self.seq[i] = int(n)
        self.state[i] = 1
        return i

    def read(self, i):
        return self.img[i, 0].copy(), self.img[i, 1].copy()

    def free(self, i):
        self.state[int(i)] = 0

    def close(self, unlink=False):
        try:
            del self.state, self.seq, self.img
        except AttributeError:
            pass
        self.shm.close()
        if unlink:
            self.shm.unlink()


I_SEQ, I_EPOCH, I_KIND, I_NLOOPS, I_MERGE, I_NDONE, I_NRES, I_ALIVE = range(8)
V_T, V_FINISH = 0, 16


class Seqlock:
    """The worker's publication block: one writer (the worker), seq odd while writing.  ints: seq, epoch,
    kind, n_loops, merge_state, n_done (packets complete), n_res (targets published), alive; vals: T_target (16),
    finish_t."""

    def __init__(self, name, create=False):
        size = 8 * 16 + 8 * 24
        self.shm = shared_memory.SharedMemory(name=name, create=create, size=size if create else 0)
        self.ints = np.ndarray((16,), np.int64, self.shm.buf, 0)
        self.vals = np.ndarray((24,), np.float64, self.shm.buf, 128)
        if create:
            self.ints[:] = 0
            self.vals[:] = 0.0
            self.vals[V_T:V_T + 16] = np.eye(4).ravel()
            self.vals[V_FINISH] = -np.inf

    def write(self, **kw):
        self.ints[I_SEQ] += 1                       # odd: writing
        for k, v in kw.items():
            if k == "T":
                self.vals[V_T:V_T + 16] = np.asarray(v, float).reshape(16)
            elif k == "finish":
                self.vals[V_FINISH] = float(v)
            else:
                self.ints[{"epoch": I_EPOCH, "kind": I_KIND, "n_loops": I_NLOOPS, "merge": I_MERGE, "n_done": I_NDONE,
                           "n_res": I_NRES, "alive": I_ALIVE}[k]] = int(v)
        self.ints[I_SEQ] += 1                       # even: consistent

    def read(self, spins=SEQ_SPINS):
        for _ in range(spins):
            s1 = int(self.ints[I_SEQ])
            if s1 & 1:
                continue
            ints = self.ints.copy()
            vals = self.vals.copy()
            if int(self.ints[I_SEQ]) == s1:
                return {"seq": s1, "epoch": int(ints[I_EPOCH]), "kind": int(ints[I_KIND]),
                        "n_loops": int(ints[I_NLOOPS]), "merge": int(ints[I_MERGE]), "n_done": int(ints[I_NDONE]),
                        "n_res": int(ints[I_NRES]), "alive": int(ints[I_ALIVE]),
                        "T": vals[V_T:V_T + 16].reshape(4, 4).copy(), "finish": float(vals[V_FINISH])}
        return None

    def close(self, unlink=False):
        try:
            del self.ints, self.vals
        except AttributeError:
            pass
        self.shm.close()
        if unlink:
            self.shm.unlink()


def make_transport(ctx, W, H, tag="p13", mode="lockstep"):
    """In the PARENT (harness / oak_demo): the shared memory and the queues.  -> (handles, owners) where handles is
    picklable into the spawned engine and worker (pass it in their args) and owners must be closed by the parent at the
    end (close_transport).  mode = the run's mode: the result queue exists for LOCK-STEP only (its sole reader); a paced
    / live run publishes through the seqlock alone."""
    pid = os.getpid()
    ring_name = "xclean_p13ring_%d" % pid
    seq_name = "xclean_p13seq_%d" % pid
    ring = KfRing(ring_name, W, H, NSLOTS, create=True)
    seq = Seqlock(seq_name, create=True)
    handles = {"ring": ring_name, "seq": seq_name, "W": int(W), "H": int(H), "nslots": NSLOTS,
               "ctl_q": ctx.Queue(), "res_q": ctx.Queue(), "ready": ctx.Event(), "tag": tag, "mode": str(mode)}
    return handles, (ring, seq)


def close_transport(owners):
    for o in owners:
        try:
            o.close(unlink=True)
        except Exception:
            pass


def worker_available(which=None):
    """(ok, why): can the configured worker be started?"""
    which = which or WORKER
    if which == "stub":
        return True, "stub"
    try:
        import importlib
        m = importlib.import_module("xclean.slam.worker")
        if not hasattr(m, "worker_main"):
            return False, "xclean.slam.worker has no worker_main"
        return True, "slam"
    except Exception as e:                                  # noqa: BLE001
        return False, "xclean.slam.worker not importable: %r" % (e,)


def worker_entry(handles, wcfg):
    """The worker process's target (spawned by the parent): the configured implementation's worker_main."""
    os.environ.update(wcfg.get("env", {}))
    if (wcfg.get("worker") or WORKER) == "stub":
        from xclean.live.p13_stub_worker import worker_main
    else:
        from xclean.slam.worker import worker_main
    return worker_main(handles, wcfg)


def start_worker(ctx, handles, wcfg):
    p = ctx.Process(target=worker_entry, args=(handles, wcfg), name="p13worker")
    p.start()
    return p


class WorkerSide:
    """The worker's view of the transport (see WORKER INTERFACE in the module docstring)."""

    def __init__(self, handles):
        self.h = handles
        self.ring = KfRing(handles["ring"], handles["W"], handles["H"], handles["nslots"])
        self.seq = Seqlock(handles["seq"])
        self.ctl_q = handles["ctl_q"]
        self.res_q = handles["res_q"]
        self.lockstep = handles.get("mode", "lockstep") == "lockstep"
        self.n_res = 0
        self.epoch = 0

    def ready(self):
        self.seq.write(alive=1)
        self.h["ready"].set()

    def get(self, timeout=1.0):
        import queue
        try:
            return self.ctl_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def images(self, slot):
        return self.ring.read(int(slot))

    def free(self, slot):
        self.ring.free(int(slot))

    def publish(self, epoch, T_target, kind, n_loops, merge, finish_t):
        self.n_res += 1
        self.epoch = int(epoch)
        if self.lockstep:                                     # lock-step delivery (ordered, with its finish stamp);
            rec = {"epoch": int(epoch), "T": np.asarray(T_target, float).reshape(16).tolist(), "kind": int(kind),
                   "n_loops": int(n_loops), "merge": int(merge), "finish": float(finish_t), "n": self.n_res}
            self.res_q.put(rec)                               # paced: nobody reads the queue (seqlock only)
        self.seq.write(epoch=epoch, T=T_target, kind=kind, n_loops=n_loops, merge=merge, finish=finish_t,
                       n_res=self.n_res)

    def progress(self, n_done):
        self.seq.write(n_done=int(n_done), n_res=self.n_res)

    def close(self):
        try:
            self.seq.write(alive=0)
        except Exception:
            pass
        # results the frame side never reads (published after its last lock-step sync) must not hold this process at
        # exit: multiprocessing joins a Queue's feeder thread, which blocks while the pipe is full and unread
        try:
            self.res_q.cancel_join_thread()
        except Exception:
            pass
        self.ring.close()
        self.seq.close()


# =============================================================================================== frame side (state)
class _Link:
    def __init__(self, handles, mode, Rbc, pbc, cam_period, log_path=None, log_writer=None):
        self.h = handles
        # the loop-closure design: the packet log.  With a log_writer (live/logwriter.LineWriter, run_live) its lines are written
        # by the writer's thread, off the frame path; without one, by this process as before
        self.log_path = log_path
        self.logw = log_writer if log_path else None
        if self.logw is not None:
            self.logw.open(log_path, "a", encoding="utf-8")
            self.log = None
        else:
            self.log = open(log_path, "a", encoding="utf-8") if log_path else None
        self.mode = mode
        self.ring = KfRing(handles["ring"], handles["W"], handles["H"], handles["nslots"])
        self.seq = Seqlock(handles["seq"])
        self.ctl_q = handles["ctl_q"]
        self.res_q = handles["res_q"]
        self.Rbc = np.array(Rbc, float)
        self.pbc = np.array(pbc, float)
        self.cam_period = cam_period
        self.pending = []                     # messages generated during the current frame, in order
        self.n_emitted = 0                    # KF packets emitted
        self.n_res_rx = 0                     # lock-step: results received from res_q
        self.res_wait = []                    # lock-step: received, not yet applicable (finish > t)
        self.target = None                    # the newest applicable result
        self.T_applied = None                 # None = no correction ever: LIVE-MAP is LIVE-ODOM itself
        self.hist = deque()                   # (t, position, R) of the published LIVE-ODOM, last B4 window
        self.path_2s = 0.0
        self.rot_2s = 0.0
        self.t_prev = None
        self.last_extra = None
        self.st = {"kf_packets": 0, "images_dropped": 0, "edge_final": 0, "edge_final_missing": 0,
                   "segment_start": 0, "edge_prev_missing": 0, "results_applied": 0, "frames_corrected": 0,
                   "seq_fallback": 0, "sync_waits": 0, "sync_timeouts": 0, "kinds": {}}
        self.t_emit = []
        self.t_compose = []
        self.t_sync = []


_L = None


def init(handles, mode, Rbc, pbc, cam_period=None, log_path=None, log_writer=None):
    """Attach the frame side (run_live, before warm-up ends).  log_path: the packet log (every message as emitted,
    without images, one JSON line each; the offline replay's input, the loop-closure design); log_writer: write it off the frame
    path (live/logwriter.LineWriter)."""
    global _L
    _L = _Link(handles, mode, Rbc, pbc, cam_period, log_path, log_writer)
    return _L


def wait_ready(timeout=600.0):
    if _L is None:
        return True
    return bool(_L.h["ready"].wait(timeout))


def active():
    return _L is not None


# ---- H1 helpers: the latest window solve, as the engine stored it (references only) -----------------------------------
def _edge(eng, sid_a, sid_b):
    """The relative edge a -> b from the latest window solve: dict with camera and body forms, or None."""
    c = getattr(eng, "_p13_cov", None)
    if c is None:
        return None
    ia, ib = c["sid2i"].get(int(sid_a)), c["sid2i"].get(int(sid_b))
    if ia is None or ib is None:
        return None
    C, off = c["C"], c["off"]
    a, b = int(off[ia]), int(off[ib])
    Ra, pa, Rb, pb = c["Rwb"][ia], c["pwb"][ia], c["Rwb"][ib], c["pwb"][ib]
    ZR, Zp, S = rel_cov(Ra, pa, Rb, pb, C[a:a + 6, a:a + 6], C[a:a + 6, b:b + 6], C[b:b + 6, b:b + 6])
    cR, cp, cS = body_to_cam(ZR, Zp, S, _L.Rbc, _L.pbc)
    return {"R": cR.ravel().tolist(), "p": cp.tolist(), "S": cS.ravel().tolist(), "R_body": ZR.ravel().tolist(),
            "p_body": Zp.tolist(), "S_body": S.ravel().tolist(), "solve": int(c["n_solve"])}


def _prev_kf_in_solve(eng, sid):
    """The newest KEYFRAME of the latest solve's window other than sid (sid itself may still have been a frame)."""
    c = getattr(eng, "_p13_cov", None)
    if c is None:
        return None
    best = None
    for i, (sd, kd, t) in enumerate(zip(c["sids"], c["kinds"], c["ts"])):
        if kd == "kf" and int(sd) != int(sid) and (best is None or t > best[2]):
            best = (int(sd), int(c["fidx"][i]), float(t), i)
    return best


def kf_record(eng, s, left, right, kind):
    """H2: a new keyframe (make_kf / init / reanchor)."""
    if _L is None:
        return
    prev = None if kind == "init" else _prev_kf_in_solve(eng, s["sid"])
    edge = None
    if prev is not None:
        e = _edge(eng, prev[0], s["sid"])
        if e is not None:
            edge = dict(e, sid_a=prev[0], fidx_a=prev[1], t_a=prev[2])
    if kind != "init" and edge is None:
        _L.st["edge_prev_missing"] += 1
    T_wb = np.eye(4)
    T_wb[:3, :3] = s["R"]
    T_wb[:3, 3] = s["p"]
    _L.pending.append({"type": "kf", "sid": int(s["sid"]), "fidx": int(s["fidx"]), "t": float(s["t"]),
                       "segment": int(eng.segment), "vi": bool(eng.vi),
                       "g_w": None if eng.g_w is None else [float(x) for x in np.asarray(eng.g_w).ravel()],
                       "kind": kind, "T_wb": T_wb.ravel().tolist(), "edge_prev": edge,
                       "_img": (left, right)})
    _L.st["kinds"][kind] = _L.st["kinds"].get(kind, 0) + 1


def edge_final(eng, s):
    """H3: keyframe s leaves the window: its FINAL edge to the next keyframe of the window."""
    if _L is None:
        return
    kfs = sorted([x for x in eng.slots if x["kind"] == "kf"], key=lambda x: x["t"])
    nxt = None
    for x in kfs:
        if x["t"] > s["t"] and x is not s:
            nxt = x
            break
    e = None if nxt is None else _edge(eng, s["sid"], nxt["sid"])
    if e is None:
        _L.st["edge_final_missing"] += 1
        return
    _L.pending.append(dict(e, type="edge_final", sid_a=int(s["sid"]), fidx_a=int(s["fidx"]), sid_b=int(nxt["sid"]),
                           fidx_b=int(nxt["fidx"])))
    _L.st["edge_final"] += 1


def segment_start(eng, link, s, left, right):
    """H4: a new segment's first keyframe s.  link = (kind, payload) set by the caller's engine line:
    ("unknown", None) | ("coast", Sig_pred or None) | ("coast_vi", Sig) | ("demote", sid of the demoted slot)."""
    if _L is None:
        return
    kind, payload = link if link is not None else ("unknown", None)
    c = getattr(eng, "_p13_cov", None)
    prev = None
    if c is not None:
        best = None
        for i, (sd, kd, t) in enumerate(zip(c["sids"], c["kinds"], c["ts"])):
            if kd == "kf" and (best is None or t > best[2]):
                best = (int(sd), int(c["fidx"][i]), float(t), i)
        prev = best
    msg = {"type": "segment_start", "kind": kind, "segment": int(eng.segment),
           "kf0": {"sid": int(s["sid"]), "fidx": int(s["fidx"]), "t": float(s["t"])},
           "prev": None, "R": None, "p": None, "S": None, "R_body": None, "p_body": None, "S_body": None}
    if prev is not None:
        Ra, pa = c["Rwb"][prev[3]], c["pwb"][prev[3]]
        ZR = Ra.T @ np.asarray(s["R"], float)
        Zp = Ra.T @ (np.asarray(s["p"], float) - pa)
        S = None
        if kind in ("coast", "coast_vi") and payload is not None:
            Sg = np.asarray(payload, float)[:6, :6]
            Jb = np.zeros((6, 6))
            Jb[:3, :3] = np.eye(3)
            Jb[3:, 3:] = Ra.T
            S = Jb @ Sg @ Jb.T                              # absolute covariance of T_start -> relative to frame a
            S = 0.5 * (S + S.T)
        elif kind == "demote" and payload is not None:
            e = _edge(eng, prev[0], payload)                # last KF -> the demoted slot
            if e is not None:
                S = np.asarray(e["S_body"], float).reshape(6, 6)
        cR, cp, cS = body_to_cam(ZR, Zp, S, _L.Rbc, _L.pbc)
        msg.update(prev={"sid": prev[0], "fidx": prev[1], "t": prev[2]}, R=cR.ravel().tolist(), p=cp.tolist(),
                   S=None if cS is None else cS.ravel().tolist(), R_body=ZR.ravel().tolist(), p_body=Zp.tolist(),
                   S_body=None if S is None else S.ravel().tolist())
    _L.pending.append(msg)
    _L.st["segment_start"] += 1
    kf_record(eng, s, left, right, "init")


# ---- H5: publication ---------------------------------------------------------------------------------------------
def _sync_lockstep(t):
    """Lock-step determinism: wait until the worker has completed every emitted packet, then take every
    result it published (ordered); the caller applies those with finish <= t."""
    import queue
    L = _L
    if L.st["sync_timeouts"] > 0:            # the worker stopped answering once: never wait again (flagged)
        return
    t0 = time.perf_counter()
    waited = False
    while True:
        r = L.seq.read()
        if r is not None and r["n_done"] >= L.n_emitted:
            break
        if time.perf_counter() - t0 > SYNC_TIMEOUT_S:
            L.st["sync_timeouts"] += 1
            r = None
            break
        waited = True
        time.sleep(0.0005)
    n_res = r["n_res"] if r is not None else L.n_res_rx
    while L.n_res_rx < n_res:
        try:
            rec = L.res_q.get(timeout=max(1.0, SYNC_TIMEOUT_S - (time.perf_counter() - t0)))
        except queue.Empty:
            L.st["sync_timeouts"] += 1
            break
        L.n_res_rx += 1
        L.res_wait.append(rec)
    if waited:
        L.st["sync_waits"] += 1
    L.t_sync.append(1e3 * (time.perf_counter() - t0))


def compose(idx, t, T_odom):
    """H5: LIVE-MAP for this frame.  -> (T_map, extra flags) ; with no link: (T_odom, None) (P1.2 behaviour)."""
    if _L is None:
        return T_odom, None
    L = _L
    c0 = time.perf_counter()
    T_odom = np.asarray(T_odom, float)
    # the odometry's own 2-s NET displacement and NET rotation (the B4 window, read as B4 reads it: score_live / the live benchmark rules
    # |dg| and the net relative rotation; HOOK_SPEC S3) for the slew bound
    dt = 0.0 if L.t_prev is None else float(t) - L.t_prev
    L.hist.append((float(t), T_odom[:3, 3].copy(), T_odom[:3, :3].copy()))
    while L.hist and L.hist[0][0] < float(t) - B4_WINDOW_S:
        L.hist.popleft()
    h0 = L.hist[0]
    path_2s = float(np.linalg.norm(T_odom[:3, 3] - h0[1]))
    rot_2s = math.degrees(float(np.linalg.norm(_so3_log(h0[2].T @ T_odom[:3, :3]))))
    L.t_prev = float(t)
    # the newest applicable target
    if L.mode == "lockstep":
        _sync_lockstep(float(t))
        keep = []
        for rec in L.res_wait:
            if rec["finish"] <= float(t):
                L.target = rec
                L.st["results_applied"] += 1
            else:
                keep.append(rec)
        L.res_wait = keep
    else:
        r = L.seq.read()
        if r is None:
            L.st["seq_fallback"] += 1
        elif r["n_res"] > 0 and (L.target is None or r["epoch"] != L.target["epoch"]):
            L.target = {"epoch": r["epoch"], "T": r["T"].ravel().tolist(), "kind": r["kind"], "n_loops": r["n_loops"],
                        "merge": r["merge"], "finish": r["finish"]}
            L.st["results_applied"] += 1
    extra = {"map_epoch": 0, "map_corr": False, "map_pending_m": 0.0, "map_pending_deg": 0.0, "map_merged": "none"}
    if L.target is None:
        L.t_compose.append(1e3 * (time.perf_counter() - c0))
        L.last_extra = extra
        return T_odom, extra
    Tt = np.asarray(L.target["T"], float).reshape(4, 4)
    Ta = np.eye(4) if L.T_applied is None else L.T_applied
    Tn, pm, pd, moved = slew_step(Ta, Tt, T_odom, dt, path_2s, rot_2s)
    if moved or L.T_applied is not None:
        L.T_applied = Tn
    extra.update(map_epoch=int(L.target["epoch"]), map_corr=bool(moved), map_pending_m=float(pm),
                 map_pending_deg=float(pd), map_merged=MERGE_NAMES.get(int(L.target["merge"]), "none"),
                 map_kind=KIND_NAMES.get(int(L.target["kind"]), "none"))
    if moved:
        L.st["frames_corrected"] += 1
    L.t_compose.append(1e3 * (time.perf_counter() - c0))
    L.last_extra = extra
    if L.T_applied is None:
        return T_odom, extra
    return L.T_applied @ T_odom, extra


# ---- H6: emission ------------------------------------------------------------------------------------------------
def emit(eng, t_frame=None, T_published=None, T_map=None):
    """H6: after both publishes of the frame.  Never blocks (ring full -> images dropped, header kept).
    T_published = the LIVE-ODOM pose published for the frame, T_map = the LIVE-MAP pose published for it (the ANCHOR of
    a keyframe, the loop-closure design)."""
    if _L is None or not _L.pending:
        return
    L = _L
    c0 = time.perf_counter()
    # HOOK_SPEC S2: the frame's kf messages LAST (stable): _make_kf (H2) runs before _upkeep -> _marginalise_kf (H3), so
    # the raw batch is [kf(j), edge_final(j - 7)]; the worker must see the final edge (candidacy of j - 7) before job j
    L.pending.sort(key=lambda m: m["type"] == "kf")
    for m in L.pending:
        m["t_emit_frame"] = None if t_frame is None else float(t_frame)
        m["wall_emit"] = time.perf_counter()
        if m["type"] == "kf":
            L.n_emitted += 1
            m["n"] = L.n_emitted
            left, right = m.pop("_img")
            slot = L.ring.write(left, right, L.n_emitted)
            if slot is None:
                L.st["images_dropped"] += 1
            m["slot"] = slot
            m["T_odom_cam"] = None if T_published is None else np.asarray(T_published, float).ravel().tolist()
            m["T_map_cam"] = None if T_map is None else np.asarray(T_map, float).ravel().tolist()
            L.st["kf_packets"] += 1
        L.ctl_q.put_nowait(m)
        if L.logw is not None:
            import json
            L.logw.write(L.log_path, json.dumps(m, default=float) + "\n")
        elif L.log is not None:
            import json
            L.log.write(json.dumps(m, default=float) + "\n")
    L.pending = []
    L.t_emit.append(1e3 * (time.perf_counter() - c0))


def close():
    """H7: end of stream: tell the worker (it exits by itself); the parent joins it."""
    if _L is None:
        return
    L = _L
    try:
        L.ctl_q.put_nowait({"type": "eos", "n_kf": L.n_emitted})
    except Exception:
        pass
    if L.log is not None:
        L.log.close()
        L.log = None
    if L.logw is not None:                   # the writer closes the file after its queued lines (run_live closes it)
        L.logw.close_file(L.log_path)
        L.logw = None
    # the frame process must not be held at exit by a worker that stopped reading (multiprocessing joins the ctl
    # queue's feeder thread, which blocks while the pipe is full): wait (bounded) until the worker has completed every
    # emitted packet; if it never became ready, or does not finish in SYNC_TIMEOUT_S, its unread messages are moot
    t0 = time.perf_counter()
    ok = bool(L.h["ready"].is_set())
    while ok:
        r = L.seq.read()
        if r is not None and r["n_done"] >= L.n_emitted:
            break
        if time.perf_counter() - t0 > SYNC_TIMEOUT_S:
            ok = False
            break
        time.sleep(0.01)
    L.st["close_wait_s"] = time.perf_counter() - t0
    if not ok:
        L.st["close_unread"] = True
        try:
            L.ctl_q.cancel_join_thread()
        except Exception:
            pass


def _pct(v):
    v = np.asarray(v, float)
    if not len(v):
        return None
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
            "max": float(v.max()), "n": int(len(v))}


def summary():
    if _L is None:
        return {"active": False}
    L = _L
    return dict(L.st, active=True, mode=L.mode, worker=WORKER, apply=APPLY, slew_f=SLEW_F, n_emitted=L.n_emitted,
                results_received=L.n_res_rx, last_epoch=None if L.target is None else int(L.target["epoch"]),
                emit_ms=_pct(L.t_emit), compose_ms=_pct(L.t_compose), sync_ms=_pct(L.t_sync),
                deterministic=(L.mode != "lockstep" or L.st["sync_timeouts"] == 0))
