# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""ENGINE (live mode): the xclean tracker behind a DATA FENCE, fed only by the FEEDER.

It receives the calibration (a dict) and the feeder stream - no dataset path, no GT path, no timestamp list.  Every data
access goes through the fence:
  * LiveSource.images(i) returns the rectified pair of the CURRENT released frame only; any other index raises;
  * LiveImu.delta_rotation_body(t0, t1) integrates released samples only (beyond the latest released stamp the
    last released rate is held: zero-order extrapolation, see the method);
  * frames() / imu() (whole-sequence access) raise.
Published per processed frame: one LIVE-ODOM message (never revised) and one LIVE-MAP message (T_map_odom x T_odom; no
online correction exists yet, so T_map_odom = I and the two coincide - stated, not hidden); a LAG message for every
frame of a keyframe interval when the NEXT keyframe's local BA has run (frame pose re-derived from its anchor
keyframe's updated pose; reported only).
"""
from __future__ import annotations

import os
import queue
import time

import numpy as np

from xclean.live.protocol import FrameSlot, make_msg, now_ns


class FenceError(RuntimeError):
    pass


class LiveSource:
    """The tracker's dataset object in live mode: rig + calibration + the current frame, nothing else."""

    def __init__(self, calib, tag):
        import cv2
        from xclean.core.derive import Rig
        self.calib = calib
        self.tag = tag
        self.path = None
        self.rig = Rig.from_calib_dict(calib, name=tag)
        c = calib
        M1 = np.array(c["M1"], float)
        D1 = np.array(c["D1"], float)
        M2 = np.array(c["M2"], float)
        D2 = np.array(c["D2"], float)
        R = np.array(c["R_stereo"], float)
        T = np.array(c["T_stereo"], float).reshape(3, 1)
        size = (int(c["camera_width"]), int(c["camera_height"]))
        R1, R2, P1, P2, _Q, _a, _b = cv2.stereoRectify(M1, D1, M2, D2, size, R, T, alpha=0)
        self._m1 = cv2.initUndistortRectifyMap(M1, D1, R1, P1, size, cv2.CV_16SC2)
        self._m2 = cv2.initUndistortRectifyMap(M2, D2, R2, P2, size, cv2.CV_16SC2)
        self._cur = None
        self._cur_index = -1
        self.horizon = -np.inf
        self.max_read = -np.inf

    def set_current(self, index, t, left_raw, right_raw):
        self._cur = (int(index), float(t), left_raw, right_raw)
        self.horizon = max(self.horizon, float(t))

    def images(self, index, rectified=True):
        import cv2
        if self._cur is None or int(index) != self._cur[0]:
            raise FenceError("images(%s): only the current released frame (%s) is readable"
                             % (index, None if self._cur is None else self._cur[0]))
        self.max_read = max(self.max_read, self._cur[1])
        _i, _t, l, r = self._cur
        if not rectified:
            return l, r
        return (cv2.remap(l, self._m1[0], self._m1[1], cv2.INTER_LINEAR),
                cv2.remap(r, self._m2[0], self._m2[1], cv2.INTER_LINEAR))

    def frames(self, *a, **k):
        raise FenceError("frames(): no timestamp list in live mode")

    def imu(self):
        raise FenceError("imu(): no whole-sequence IMU in live mode")

    def rect_digest(self):
        return "live"


def make_live_imu(source, verbose):
    """An ImuSource that starts EMPTY and grows with the released samples (fenced)."""
    from xclean.backend import imu as I

    class LiveImu(I.ImuSource):
        # IMU-intake fix: the released samples live in
        # capacity-doubling buffers, so append is amortised O(1) (np.concatenate on the whole history per append was
        # O(N): 30-138 ms per frame on 4Seasons' 2000 Hz).  t / g / a are views of the filled prefix (contiguous),
        # read exactly as the arrays they replace.
        t = property(lambda self: self._tb[:self._n], lambda self, v: self._set_all(t=v))
        g = property(lambda self: self._gb[:self._n], lambda self, v: self._set_all(g=v))
        a = property(lambda self: self._ab[:self._n], lambda self, v: self._set_all(a=v))

        def _set_all(self, t=None, g=None, a=None):
            if t is not None:
                t = np.asarray(t, np.float64).reshape(-1)
                self._n = len(t)
                cap = max(self._n, 1)
                self._tb = np.zeros(cap)
                self._tb[:self._n] = t
                if getattr(self, "_gb", None) is None or len(self._gb) < cap:
                    self._gb = np.zeros((cap, 3))
                    self._ab = np.zeros((cap, 3))
            if g is not None:
                self._gb[:self._n] = np.asarray(g, np.float64).reshape(-1, 3)[:self._n]
            if a is not None:
                self._ab[:self._n] = np.asarray(a, np.float64).reshape(-1, 3)[:self._n]

        def __init__(self, src):
            self.bias_uncertainty = "spread"
            self.ds = src
            self._n = 0
            self._tb = np.zeros(1024)
            self._gb = np.zeros((1024, 3))
            self._ab = np.zeros((1024, 3))
            cal = src.calib
            self.R_cb = np.array(cal.get("R_cam0_imu", np.eye(3)), dtype=np.float64)
            n = cal.get("imu_noise") or {}
            self.sigma_g = float(n.get("gyroscope_noise_density", 0.0) or 0.0)
            self.rw_g = float(n.get("gyroscope_random_walk", 0.0) or 0.0)
            self.sigma_a = float(n.get("accelerometer_noise_density", 0.0) or 0.0)
            self.rw_a = float(n.get("accelerometer_random_walk", 0.0) or 0.0)
            self.declared = bool(n)
            T_bc = np.array(cal.get("T_body_cam0", np.eye(4)), dtype=np.float64)
            if T_bc.shape != (4, 4):
                T_bc = np.eye(4)
                T_bc[:3, :3] = self.R_cb.T
            self.T_bc = T_bc
            self.T_cb = np.linalg.inv(T_bc)
            self.rate = float(cal.get("imu_rate_hz") or 0.0)      # Class A; measured from the stream if absent
            # B3: no whole-sequence frame interval; the T1-c gap arm is not supported live
            if I.gapspan_on():
                raise FenceError("XTEN_IF_GAPSPAN needs the whole-sequence frame interval: not supported live")
            self.frame_interval_s = None
            self.gaps = None
            self.gap_refused = {s: {"gap": 0, "sparse": 0} for s in I.GAP_SITES}
            self.gba_preint_intervals = None
            self.gba_preint_none = None
            self.gba_kf_unobservable = None
            self.bias = np.zeros(3)
            self.bias_sigma = None
            self.bias_sigma_spread = None
            self.bias_sigma_tight = None
            self.bias_clamp = False
            self._clamp_armed = False
            self._P = None
            self._dt_accum = 0.0
            self.n_clamped = 0
            self.static_window_s = 0.0
            self._resid = []
            self.bias_updates = 0
            self.bias_source = "none"
            self.verbose = verbose
            self.horizon = -np.inf

        def append(self, t, g, a):
            if len(t) == 0:
                return
            k = len(t)
            if self._n + k > len(self._tb):
                cap = max(2 * len(self._tb), self._n + k)
                for name, shape in (("_tb", (cap,)), ("_gb", (cap, 3)), ("_ab", (cap, 3))):
                    old = getattr(self, name)
                    new = np.zeros(shape)
                    new[:self._n] = old[:self._n]
                    setattr(self, name, new)
            self._tb[self._n:self._n + k] = t
            self._gb[self._n:self._n + k] = g
            self._ab[self._n:self._n + k] = a
            self._n += k
            self.horizon = float(self._tb[self._n - 1])
            if not self.rate and len(self.t) > 10:
                self.rate = float(1.0 / np.median(np.diff(self.t)))

        def available(self):
            return len(self.t) > 10 and self.rate > 0

        def delta_rotation_body(self, t0, t1):
            """Causal gyro integration over [t0, t1].  The offline rule (ImuSource.delta_rotation_body) advances the
            last partial sub-interval with the FIRST sample at or after t1 - on a rig whose IMU is not synchronised
            to the camera (Rosario, 4Seasons, ...) that sample lies up to one IMU period AFTER the frame, i.e. in the
            future at frame time (the fence used to raise here: 2332 of 2540 rosario_04 frames in the first paced
            test series).  Live, only released samples exist: integrate to the last released sample (the right-endpoint
            rule ends exactly on it), then HOLD that sample's rate over (horizon, t1] - zero-order extrapolation,
            no future data, no constant.  A request inside the released span is unchanged."""
            from xclean.core.lie import so3_exp, so3_project
            if t1 <= self.horizon + 1e-9:
                self.max_read = max(getattr(self, "max_read", -np.inf), float(t1))
                return super().delta_rotation_body(t0, t1)
            if not self.available() or t1 <= t0:
                return None
            th = float(self.horizon)
            if th > t0:
                R = super().delta_rotation_body(t0, th)
                if R is None:
                    return None
                ta = th
            else:
                R = np.eye(3)
                ta = t0
            R = R @ so3_exp((self.g[-1] - self.bias) * (t1 - ta))
            self.n_zoh = getattr(self, "n_zoh", 0) + 1
            self.max_read = max(getattr(self, "max_read", -np.inf), th)
            return so3_project(R)

    return LiveImu(source)


LOCKSTEP_IMU_WAIT_S = 2.0      # carried (the pre-fix cap): lock-step semantics unchanged (every sample <= the frame)


def _pct(v):
    v = np.asarray(v, float)
    if not len(v):
        return None
    return {"p50": float(np.median(v)), "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
            "max": float(v.max()), "n": int(len(v))}


def imu_wait_bound(cfg):
    """The engine's IMU wait per frame.  PACED: one camera period (class B: waiting longer than the time to the next
    frame cannot help this frame; the period from the calib's camera_fps, class A); past it the C40 zero-order hold
    carries the frame and the frame is FLAGGED (imu_short).  LOCKSTEP: the carried 2.0 s (semantics unchanged)."""
    if cfg.get("mode") == "paced":
        fps = float(cfg["calib"].get("camera_fps") or 0.0)
        return 1.0 / fps if fps > 0 else 0.0
    return LOCKSTEP_IMU_WAIT_S


def drain_imu(imu_q, imu, need, wait_s, fresh):
    """IMU-intake fix: every queued packet is drained in ONE batch and appended once (amortised
    O(1), LiveImu buffers); if this frame's IMU (horizon >= need, the last sample the feeder released before it) has
    not all arrived, block at most wait_s in total for it.  -> (n_samples, waited_s, short): short = the horizon is
    still below need (the C40 hold carries the frame)."""
    ts, gs, as_ = [], [], []
    t0 = time.perf_counter()
    hz = imu.horizon if imu is not None else np.inf

    def take(item):
        nonlocal hz
        ti, gi, ai = item
        if len(ti):
            ts.append(ti)
            gs.append(gi)
            as_.append(ai)
            hz = max(hz, float(ti[-1]))

    while True:
        try:
            while True:
                take(imu_q.get_nowait())
        except queue.Empty:
            pass
        if not fresh or imu is None or hz >= need - 1e-9:
            break
        rem = wait_s - (time.perf_counter() - t0)
        if rem <= 0:
            break
        try:
            take(imu_q.get(timeout=rem))
        except queue.Empty:
            break
    n = 0
    if ts and imu is not None:
        imu.append(np.concatenate(ts), np.concatenate(gs), np.concatenate(as_))
        n = sum(len(x) for x in ts)
    short = bool(fresh and imu is not None and imu.horizon < need - 1e-9)
    return n, time.perf_counter() - t0, short


def engine_main(cfg, slot_name, imu_q, msg_q, ctl, out_dir):
    """cfg: tag, calib (dict), env (dict), final (bool: run the end-of-run map stages), seed."""
    os.environ.update(cfg.get("env", {}))
    if (os.environ.get("XCLEAN_ENGINE", "") or "").strip().lower() == "p1":
        from xclean.vio.engine_live import run_live            # P1 (DESIGN_P1): CPU-only frame process
        return run_live(cfg, slot_name, imu_q, msg_q, ctl, out_dir)
    import cv2
    import cupy as cp
    from xclean.obs.latency import Ledger
    from xclean.stages.depth import SgmDepth
    from xclean.stages.features import Frontend
    from xclean.track.tracker import Tracker
    from xclean.io.dataset import FrameRef
    np.random.seed(int(cfg.get("seed", 0)))
    cv2.setRNGSeed(int(cfg.get("seed", 0)))
    if os.environ.get("XCLEAN_CV_THREADS"):
        cv2.setNumThreads(int(os.environ["XCLEAN_CV_THREADS"]))
    tag = cfg["tag"]
    src = LiveSource(cfg["calib"], tag)
    led = Ledger(run=tag, tier=1, sync=cp.cuda.runtime.deviceSynchronize,
                 gpu_stages=("depth", "extract", "match", "track"))
    imu_wait_s = imu_wait_bound(cfg)
    imu_wait_ms = []
    imu_short = []
    depth = SgmDepth(verbose=False)

    class _LiveDepth:
        def __init__(self, inner):
            self.inner = inner
            self.n_hit = 0

        def compute(self, left, right):
            return self.inner.compute(left, right)

        def stats(self):
            return self.inner.stats()

    cdepth = _LiveDepth(depth)
    fe = Frontend(src.rig, cap=2048, verbose=False)
    imu = make_live_imu(src, verbose=False) if cfg["calib"].get("imu_rate_hz") or cfg.get("has_imu") else None
    tr = Tracker(src, fe, cdepth, led, None, verbose=False, backend=True, imu=imu, localmap=True, far_band=True,
                 inverse_depth=True, frozen_bearing=True, inertial=True, age_consensus=True,
                 structure_welds=True)
    from xclean.loop.place import PlaceDescriptor
    from xclean.loop.closer import LoopCloser
    tr.loop = LoopCloser(src.rig, fe, PlaceDescriptor(verbose=False), tr, min_gap=tr.state_window,
                         seed=int(cfg.get("seed", 0)), verbose=False, apply_online=False, prox_loops=True,
                         weld_ratio_gate=True)
    # ---- warm-up (before the feeder's clock starts): compile every kernel on synthetic data of the rig's size
    from xclean.backend.ba import warmup
    warmup(src.rig)
    W, H = src.rig.width, src.rig.height
    rng = np.random.default_rng(0)
    syn = (cv2.GaussianBlur((rng.random((H, W)) * 255).astype(np.uint8), (0, 0), 1.5))
    depth.compute(syn, np.roll(syn, -8, axis=1))
    from xclean.stereo.epipolar import EpipolarMatcher
    EpipolarMatcher().match(syn, np.roll(syn, -8, axis=1), np.array([[W / 2.0, H / 2.0]]), np.array([8.0]))
    cp.cuda.runtime.deviceSynchronize()
    slot = FrameSlot(slot_name, int(cfg["calib"]["camera_width"]), int(cfg["calib"]["camera_height"]))
    ctl["ready"].set()

    T_map_odom = np.eye(4)
    lag_buf = []            # (frame_index, t, T_wc at publish, anchor kf id, anchor pose at publish)
    n_pub = 0
    fence_errors = []
    last_seq = 0
    t_first = None
    kf_wall = []
    while True:
        ctl["new_frame"].wait(0.5)
        ctl["new_frame"].clear()
        with ctl["lock"]:
            seq = int(slot.hdr[0])
            eos = bool(slot.hdr[4])
            fresh = seq != last_seq and slot.hdr[3] == 0
            if fresh:
                idx = int(slot.hdr[1])
                t = float(slot.tstamp[0])
                imu_need = float(slot.tstamp[1])
                l = slot.buf[0].copy()
                r = slot.buf[1].copy()
                slot.hdr[3] = 1
                last_seq = seq
        # every released IMU sample, in order, in one batch; bounded wait for this frame's samples (intake fix)
        _n_imu, _w_imu, _short = drain_imu(imu_q, imu, imu_need if fresh else -np.inf, imu_wait_s, fresh)
        if fresh:
            imu_wait_ms.append(1e3 * _w_imu)
            if _short:
                imu_short.append(int(idx))
        if not fresh:
            if eos:
                break
            continue
        if t_first is None:
            t_first = t
        src.set_current(idx, t, l, r)
        src._cur_index = idx
        fr = FrameRef(index=idx, t=t, left=None, right=None)
        try:
            if imu is not None:
                imu.causal_bootstrap(t)
            state = tr.step(fr)
        except FenceError as e:
            fence_errors.append([idx, str(e)])
            state = "fence-error"
        led.frame_done(idx)
        T = tr.T_wc
        # the newest data stamp this frame's processing actually READ (frame + IMU through the fence)
        horizon = max(src.max_read, getattr(imu, "max_read", -np.inf) if imu is not None else -np.inf)
        inl = getattr(tr, "_frame_inliers", None)
        degr = state not in ("tracking", "relocalized", "init")
        common = dict(state=state, n_inliers=inl, degraded=degr, gravity_aligned=False, segment=0, horizon=horizon)
        msg_q.put(make_msg(idx, t, "LIVE-ODOM", T, **common))
        msg_q.put(make_msg(idx, t, "LIVE-MAP", T_map_odom @ T, **common))
        n_pub += 1
        # LAG: when a keyframe was made on this frame, the frames of the closed interval are re-derived from their
        # anchor keyframe's pose as it stands after this keyframe's local BA
        made_kf = len(tr.all_kf) and int(tr.all_kf[-1][1]) == idx
        if made_kf:
            kf_wall.append((time.time(), len(tr.all_kf)))          # B6 growth per keyframe (harness regression)
        anc = (tr.all_kf[-1][0], np.array(tr.all_kf[-1][3])) if len(tr.all_kf) else (None, np.eye(4))
        if made_kf and lag_buf:
            pos = {int(rec[0]): np.array(rec[3]) for rec in tr.all_kf[-tr.state_window:]}
            for (j, tj, Tj, akid, Ta) in lag_buf:
                Tn = pos.get(akid)
                Tl = Tj if Tn is None else Tn @ np.linalg.inv(Ta) @ Tj
                msg_q.put(make_msg(j, tj, "LAG", Tl, state="lag", n_inliers=None, degraded=False,
                                   gravity_aligned=False, segment=0, horizon=horizon))
            lag_buf = []
        lag_buf.append((idx, t, np.array(T), anc[0], anc[1]))
        ctl["done"].value = idx
        ctl["done_evt"].set()
    # ---- end of stream: the end-of-run stages exist for MAPS only (not a development metric)
    summary = {"frames_processed": n_pub, "fence_errors": fence_errors[:20], "n_fence_errors": len(fence_errors),
               "imu_zoh_extrapolations": int(getattr(imu, "n_zoh", 0)) if imu is not None else None,
               "kf_wall": kf_wall, "imu_wait_bound_s": imu_wait_s, "imu_wait_ms": _pct(imu_wait_ms),
               "imu_short_frames": len(imu_short), "imu_short_first": imu_short[:20]}
    if tr.loop is not None:
        if cfg.get("final", False):
            d = tr.loop.finalise(tr, run_pgo=True)
        tr.loop.close()
    if cfg.get("final", False):
        tr.global_refine(verbose=False)
    tr.flush()
    try:
        summary.update(tr.summary())
    except Exception as e:
        summary["summary_error"] = str(e)
    import json
    from xclean.run import _engine_md5
    summary["engine_md5"] = _engine_md5()
    summary["env_flags"] = {k: os.environ[k] for k in sorted(os.environ) if k.startswith(("XTEN_", "XONE_", "XCLEAN_"))}
    json.dump(summary, open(os.path.join(out_dir, "engine_summary.json"), "w"), indent=1, default=str)
    led.save(os.path.join(out_dir, "latency.json"))
    msg_q.put("EOS")
    slot.close()
