#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1 ENGINE (Phase 1: VO core). One frame in, one published pose out, from a fixed-lag window that contains it.

    eng = P1Engine(source, led)        # source: .rig, .images(i) -> rectified (left, right)  (LiveSource or Dataset)
    state = eng.step(fr)               # fr: .index, .t ; publishes eng.T_wc (T_world_cam0_rectified)

Per frame (DESIGN_P1 §3; Phase 1 has no IMU, so prediction is constant velocity IN TIME):
  rectify -> detect + describe (XC-03 cell floor) -> predict pose + covariance -> map matching (projection + gated
  Hamming; prediction-free descriptor matching until the constant-velocity covariance is measured) -> PnP-RANSAC +
  whole-correspondence verify (track/pose.py, always in VO) -> window solve (5 LM it) -> outlier removal + robust scales
  -> publish -> KF decision (connected ratio < 0.7 or count < 3 x MIN_PNP_INLIERS) -> KF landmark creation (1-D stereo)
  -> window upkeep (3 frames: drop oldest; 7 KFs: marginalise oldest KF + its hosted landmarks into the sqrt prior).
Coast: PnP fails -> publish the prediction (state "coasting"), covariance grows with dt^3. Map out of view while
coasting -> new segment at the first frame with >= 3 x MIN_PNP_INLIERS stereo points, starting at the propagated pose
(LIVE-ODOM never jumps).

P1.2: W1 (vio/imu_watch.py)
watches the newest IMU factor every frame and demotes only when the camera's stage A certifies itself (it replaces D3's
acting path; D3 still logs).  W3 item 1: the window solve keeps POSE-FREE landmarks (only valid observation = the host's
left one) out of the Schur system in the exact form (window.solve's `pf` block).  W2 (time offset) and A4 were
DROPPED and REMOVED (vio/td_est.py is an unused design module).  XCLEAN_P12_OFF = '+'-list of w1 /
w3 (attribution arms, a diagnostic run).

Constants: every one is listed in the change log (P1 block) with its class.  Window 7 + 3, 5 iterations, KF ratio
0.7: class D (DESIGN_P1 §13).  Gates: chi2 quantiles (statistics).  Scales: Class C running robust estimates.
"""
from __future__ import annotations

import math
import os
import zlib

import numpy as np

from xclean.core.derive import MIN_PNP_INLIERS, FEATURE_SIGMA_PX, HUBER_K
from xclean.vio import window as Wn
from xclean.vio.marg import Prior, combine, gauge_prior, schur, add_block_info

N_KF = 7                       # class D (Basalt), ablation {5, 7, 10}
N_FR = 3                       # class D
LM_ITERS = 5                   # class D, ablation {3, 5, 7}
KF_RATIO = 0.7                 # class D (Basalt connected-landmark ratio)
KF_COUNT = 3 * MIN_PNP_INLIERS  # carried (3 x the conditioning floor)
CHI2_2_999 = 13.815510557964274
CHI2_1_999 = 10.827566170662733
RAYLEIGH_MED = math.sqrt(2.0 * math.log(2.0))     # median of a unit 2-D Gaussian norm
MAD_K = 1.482602218505602                         # 1 / Phi^-1(3/4)
SCALE_WINDOW = N_KF + N_FR                        # running-scale memory = the window length (class B)
STEREO_IMPL = (os.environ.get("XCLEAN_P1_STEREO_IMPL", "v1") or "v1").strip()   # ST2 A/B flag: "v1" = front/stereo1d
                               # (P1-03c gap rule, the default), "v2" = front/stereo1d_v2
ADMIT_MODE = (os.environ.get("XCLEAN_P1_ADMIT", "gate") or "gate").strip()   # "gate" (default, vi_init admission)
                               # or "force_diag": ORACLE DIAGNOSTIC ONLY, never a
                               # shipped configuration; the harness labels such runs a diagnostic run
if ADMIT_MODE not in ("gate", "force_diag"):
    raise ValueError("XCLEAN_P1_ADMIT must be gate or force_diag, got %r" % ADMIT_MODE)
MAX_INIT_KF = 20               # D carried: the VI-init window is the segment's newest <= 20 KFs = the n range U6
                               # measured the admission statistic on.
                               # P2-fix-1: unbounded, the fit grew with every deferral and stalled the engine.
MAX_DESC = 3                                      # OKVIS keeps 3 descriptors per landmark (class D)
K_BAND = 3.2905267314918945    # P1.1 D4: sqrt(chi2_1(0.999)), the row band's sigma multiple (class-D level P_GATE)
P11_OFF = set(x.strip() for x in (os.environ.get("XCLEAN_P11_OFF", "") or "").replace("+", ",").split(",") if x.strip())
# ATTRIBUTION ONLY (never a shipped config): P1.1 components switched off, any of d1, d2, d3, d4
if not P11_OFF <= {"d1", "d2", "d3", "d4"}:
    raise ValueError("XCLEAN_P11_OFF: unknown component in %r" % sorted(P11_OFF))
# P1.2: ATTRIBUTION ARMS ONLY.  A '+'-separated token list (no commas: the harness --env value
# is itself comma-separated): w1 = W1 computes and logs but never acts (no demotion, no block of re-admission); w3 = the
# window solve keeps pose-free landmarks in its Schur system (the pre-W3 full solve; W3 item 1 is bit-exact, so this
# arm is its identity check).  Default: every P1.2 component ON.  Any value labels the run a diagnostic run (harness).
# (The w2 / a4 tokens were removed with W2 and A4.)
_P12_RAW = (os.environ.get("XCLEAN_P12_OFF", "") or "").strip()
if "," in _P12_RAW:
    raise ValueError("XCLEAN_P12_OFF is a '+'-separated list (w1+w3), got %r" % _P12_RAW)
P12_OFF = set(x.strip() for x in _P12_RAW.split("+") if x.strip())
if not P12_OFF <= {"w1", "w3"}:
    raise ValueError("XCLEAN_P12_OFF: unknown component in %r" % sorted(P12_OFF))


def _rectified_extrinsic(calib):
    """T_bc (camera -> body) for the RECTIFIED cam0 (CONVENTIONS.md): calib T_body_cam0 x [[R1^T, 0], [0, 1]], R1 from
    cv2.stereoRectify with the calib's own intrinsics / extrinsics, alpha = 0 (as io/dataset.py rectify_maps)."""
    import cv2
    c = calib
    T_bc = np.array(c.get("T_body_cam0", np.eye(4)), float).reshape(4, 4)
    if "M1" not in c:
        return T_bc
    R1 = cv2.stereoRectify(np.array(c["M1"], float), np.array(c["D1"], float), np.array(c["M2"], float),
                           np.array(c["D2"], float), (int(c["camera_width"]), int(c["camera_height"])),
                           np.array(c["R_stereo"], float), np.array(c["T_stereo"], float).reshape(3, 1),
                           alpha=0)[0]
    Tr = np.eye(4)
    Tr[:3, :3] = np.asarray(R1, float).T
    return T_bc @ Tr


class OfflineImu:
    """The whole IMU stream of a dataset with a CAUSAL horizon (advance(t): the last sample with stamp <= t), the
    same view LiveImu gives the live engine.  imu_factor reads no sample past the horizon (C40)."""

    def __init__(self, path):
        import json
        t, g, a = [], [], []
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    d = json.loads(ln)
                    t.append(d["t"])
                    g.append((d["gx"], d["gy"], d["gz"]))
                    a.append((d["ax"], d["ay"], d["az"]))
        self.t = np.asarray(t, float)
        self.g = np.asarray(g, float)
        self.a = np.asarray(a, float)
        self.horizon = -np.inf

    def advance(self, t_frame):
        k = int(np.searchsorted(self.t, float(t_frame), side="right"))
        self.horizon = float(self.t[k - 1]) if k > 0 else -np.inf


def _stereo_module():
    """The 1-D stereo matcher selected by XCLEAN_P1_STEREO_IMPL (same API in both modules); XCLEAN_LAT gf: the device
    twin of stereo1d (bit-identical, D4 band included)."""
    from xclean.front import fastpaths as FP
    if FP.on("gf"):
        if STEREO_IMPL != "v1":
            raise ValueError("XCLEAN_LAT gf reproduces front/stereo1d only: refuse XCLEAN_P1_STEREO_IMPL=%r" % STEREO_IMPL)
        from xclean.front import gpu_front as m
        return m
    if STEREO_IMPL == "v2":
        from xclean.front import stereo1d_v2 as m
    elif STEREO_IMPL == "v1":
        from xclean.front import stereo1d as m
    else:
        raise ValueError("XCLEAN_P1_STEREO_IMPL must be v1 or v2, got %r" % STEREO_IMPL)
    return m


def _se3(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


class P1Engine:
    def __init__(self, source, led=None, verbose=False):
        from xclean.front import fastpaths as FP
        if FP.on("gf"):                              # device front
            if STEREO_IMPL != "v1":
                raise ValueError("XCLEAN_LAT gf refuses XCLEAN_P1_STEREO_IMPL=%r" % STEREO_IMPL)
            from xclean.front.gpu_front import Detector
        else:
            from xclean.front.detect import Detector
        if FP.on("mf") or FP.on("gf"):
            from xclean.front.mapmatch_fast import MapMatcher
        else:
            from xclean.front.mapmatch import MapMatcher
        self.src = source
        rig = source.rig
        self.rig = rig
        self.fx = float(rig.fx_rect)
        self.fy = float(rig.fx_rect)
        self.cx = float(rig.cx)
        self.cy = float(rig.cy)
        self.W = int(rig.width)
        self.H = int(rig.height)
        self.fB = float(rig.fB_m_px)
        self.B = self.fB / self.fx
        self.Rbc = np.eye(3)            # VO: body = rectified camera (CONVENTIONS.md); VI: the calib extrinsic below
        self.pbc = np.zeros(3)
        # ---- Phase 2: IMU (DESIGN_P1 §4, §6).  VI only when the rig declares its IMU noise (Class A) and an IMU
        # stream is attached; XCLEAN_P1_VI=0 forces VO (the IMU-off A/B arm).
        self.imu = getattr(source, "imu_stream", None)
        self.noise = None
        self.vi = False                 # True after admission (the window then carries 15-dim states)
        self.vi_refused = False
        self.g_w = None
        self.kappa_imu = 1.0
        self._kimu_buf = []             # raw (kappa = 1) chi2 of the last marginalised IMU factors (DESIGN_P1 6.3)
        self._vi_b0 = None              # (bg0, ba0, cov_bg0, cov_ba0, t0) at admission: the bias bound's reference
        self.imu_factors = []           # {"i": sid, "j": sid, "pre": Preint, "W": W, "kappa": k}
        self._vi_prev = None            # the previous admission Candidate's init (vi_init agreement test)
        self._kf_hist = []              # (sid, t, T_wb, horizon) of the segment's keyframes, for the VI init
        self.Sig_full = None            # the newest state's 15 x 15 covariance (VI)
        # ---- P1.1: D1 IMU-first init, D2 inertial continuity, D3 watchdog
        self._frame_t = []              # the last SCALE_WINDOW + 1 frame stamps (D1's sub-windows = frame intervals)
        self._d1_used = False           # D1 admits at most once per run (a demotion is evidence against the IMU)
        self._demoted = False
        self.vi_path = None             # "d1" (IMU-first) or "dyn" (vi_init dynamic, gated)
        self._wd_cons = None            # watchdog consecutive-failure counts
        self._last_imu_chi2 = []        # the last window solve's per-factor whitened IMU chi2
        self.dy_hat = 0.0               # D4: class-C vertical rectification residual [px] and its row band
        self.dy_sigma = None
        self.dy_band = 0
        self._dy_pool = []
        calib = getattr(source, "calib", None)
        # ---- P1.2 W1: the fault-attributing IMU watchdog, every frame on
        # every path; it replaces D3's acting path (D3 still logs).  Camera period: calib camera_fps (class A), else
        # imu_watch's class-C running median of the frame intervals.
        from xclean.vio import imu_watch as IWATCH
        fps = (calib or {}).get("camera_fps")
        self.cam_period = (1.0 / float(fps)) if fps else None
        self.w1 = IWATCH.ImuWatch(period=self.cam_period)
        self._w1_last = None            # the frame's W1 decision dict
        self._w1_done = False
        self._stage_a = (0, 0, False, None)   # (n_matches_A, n_inliers_A, ok_A, s_A) of this frame's stage A
        self._n_solves0 = 0
        # ---- P1.2 W3 item 1: pose-free landmarks leave the window solve's Schur system (exact form)
        self.w3 = "w3" not in P12_OFF
        # ---- P1.3 ENGINE HOOK: the loop-closure link (xclean/live/p13_link.py), attached by
        # run_live when the loop is on; None = loop off: no hook call and no `_p13_` write at all.  The hooks only READ
        # engine state; `_p13_cov` / `_p13_link` are the hook's own slots, never read by the estimator.
        self._p13 = None
        self._p13_cov = None
        self._p13_link = None
        if (os.environ.get("XCLEAN_P1_VI", "1") or "1").strip() != "0" and self.imu is not None and calib \
                and calib.get("imu_noise"):
            from xclean.vio import imu_factor as F
            try:
                self.noise = F.ImuNoise.from_calib(calib)
            except ValueError:
                self.noise = None
            if self.noise is not None:
                Tbc = _rectified_extrinsic(calib)
                self.Rbc = np.ascontiguousarray(Tbc[:3, :3])
                self.pbc = np.ascontiguousarray(Tbc[:3, 3])
        self.led = led
        self.verbose = verbose
        # start-up self-check policy (front/gpu_front.py): the GPU Detector builds and checks the device front; if that
        # fails, the whole run uses the CPU front (_front_fallback); XCLEAN_STRICT_SELFCHECK=1 raises as before
        self.front_fallback = None
        if FP.on("gf"):
            from xclean.front.gpu_front import strict_selfcheck
            if strict_selfcheck():
                self.det = Detector(rig)
            else:
                try:
                    self.det = Detector(rig)
                except Exception as e:
                    Detector, MapMatcher = self._front_fallback(e)
                    self.det = Detector(rig)
        else:
            self.det = Detector(rig)
        self.mm = MapMatcher(self.W, self.H)
        self.stereo_mode = (os.environ.get("XCLEAN_P1_STEREO", "prior") or "prior").strip().lower()
        if self.stereo_mode not in ("prior", "ur"):
            raise RuntimeError("XCLEAN_P1_STEREO=%r: prior (default, judge A1) or ur" % self.stereo_mode)
        self.dump_dir = os.environ.get("XCLEAN_P1_DUMP") or None
        # the whole Problem (VI included) pickled before the solve, every k-th solve, at most 500
        self.wd_log = os.environ.get("XCLEAN_P1_WD_LOG") or None       # D3 statistics per evaluation (diagnostic)
        # P1.2: per-frame log (visual inliers, reprojection chi2, IMU chi2, innovation); run_live writes it into the run
        # dir by default (XCLEAN_P1_FRAME_LOG=0 disables), the offline driver only when a path is given
        fl = os.environ.get("XCLEAN_P1_FRAME_LOG") or None
        self.frame_log = None if fl in (None, "0") else fl
        self._fl = {}
        # run_live sets a live/logwriter.LineWriter: the frame log and the watchdog log are then written by its thread,
        # off the frame path (same bytes, same order); None = the synchronous append below (offline drivers)
        self.log_writer = None
        self.pkl_dir = os.environ.get("XCLEAN_P1_DUMP_PKL") or None
        self.pkl_every = int(os.environ.get("XCLEAN_P1_DUMP_PKL_EVERY", "1") or 1)
        self.n_pkl = 0
        self.n_solves = 0
        self.dump_every = int(os.environ.get("XCLEAN_P1_DUMP_EVERY", "1") or 1)
        self.n_dumped = 0
        # window
        self.slots = []                 # dicts; KFs first (oldest -> newest), then frames (oldest -> newest)
        self.prior = None
        self._next_sid = 0
        cap = 4096
        self.n_lm = 0
        self.lm_p = np.zeros((cap, 3))
        self.lm_host = np.full(cap, -1, np.int64)
        self.lm_alive = np.zeros(cap, bool)
        self.lm_rho_meas = np.zeros(cap)
        self.lm_sig_lk = np.full(cap, np.inf)
        self.lm_span = np.zeros(cap)
        self.lm_desc = np.zeros((cap, MAX_DESC, 32), np.uint8)
        self.lm_ndesc = np.zeros(cap, np.int64)
        self.lm_Sl = np.zeros((cap, 3, 3))
        # scales (Class C, running)
        self.sig_px = FEATURE_SIGMA_PX
        self.kappa_d = 1.0
        self._sig_hist = []
        self._kd_hist = []
        # motion model (constant velocity in time) + its measured residual covariance
        self.T_last = np.eye(4)         # body pose of the last published frame
        self.t_last = None
        self.D_body = None
        self.dt_vel = None
        self.Sig_last = None
        self._q_hist = []
        self._carry = (np.zeros(3), np.zeros(3), np.zeros(3))
        self.T_wc = np.eye(4)
        self.debug = []
        self.state = "init"
        self.segment = 0
        self.coast_streak = 0
        self.n_last_kf_obs = 0
        self.stats = {"frames": 0, "tracked": 0, "coasted": 0, "init_frames": 0, "segments": 0, "kfs": 0,
                      "landmarks_created": 0, "matched_gated": 0, "matched_bf": 0, "bf_fallback": 0,
                      "re_search": 0, "pnp_fail": 0, "pnp_inliers": 0, "outliers_removed": 0, "marg_kf": 0,
                      "drop_frame": 0, "window_P": [], "window_L": [], "window_M": [], "cost_ratio": [],
                      "vi_admit_frame": None, "vi_refused": 0, "vi_defer": 0, "marg_full": 0, "repreint": 0,
                      "imu_coast": 0, "vi_init_ms": [], "vi_defer_reasons": {}, "bias_chi2_max_bg": None,
                      "bias_chi2_max_ba": None, "bias_exceed_frame": None, "kappa_imu_updates": 0,
                      "d1_tests": 0, "d1_admit_frame": None, "d1_F_max": None, "d1_kappa0": None, "reanchor": 0,
                      "vi_resets": 0, "vi_coast_max": 0, "demote_frame": None, "demote_reason": None,
                      "wd_active": {}, "wd_kfs": 0, "dy_updates": 0, "dy_band_max": 0, "events": [],
                      "vi_frames": 0, "w1_demotes": 0, "window_L_posefree": []}

    def _front_fallback(self, err):
        """Start-up self-check policy (front/gpu_front.py): the GPU front failed to build or failed a self-check, so
        this whole run uses the CPU front (front/detect.Detector, front/stereo1d and the map matcher the remaining
        XCLEAN_LAT flags select).  The GPU front is dropped and CuPy's cached GPU memory freed first; a CUDA context
        that already exists is kept and recorded (gpu_front.release_device).  Called before the first frame only.
        -> (Detector class, MapMatcher class)."""
        import sys
        from xclean.front import fastpaths as FP
        from xclean.front import gpu_front
        FP.LAT.discard("gf")
        self.det = None                   # a GPU Detector holds the device front: drop it before its memory is freed
        rel = gpu_front.release_device()
        from xclean.front.detect import Detector
        if FP.on("mf"):
            from xclean.front.mapmatch_fast import MapMatcher
        else:
            from xclean.front.mapmatch import MapMatcher
        reason = ("%s: %s" % (type(err).__name__, err))[:500]
        self.front_fallback = {"reason": reason, "cuda_context_released": bool(rel["cuda_context_released"]),
                               "cuda_context_note": rel["note"], "gpu_memory_freed": rel["gpu_memory_freed"],
                               "release_errors": rel["errors"]}
        print("xclean: warning: the GPU front end failed its start-up self-check (%s). This run uses the CPU front end "
              "instead; per-frame latency will be higher. Set XCLEAN_STRICT_SELFCHECK=1 to stop instead.%s"
              % (reason[:300], "" if rel["cuda_context_released"] else " " + rel["note"]),
              file=sys.stderr, flush=True)
        if getattr(self, "mm", None) is not None:      # after construction (a warm-up step failed): swap both now
            self.det = Detector(self.rig)
            self.mm = MapMatcher(self.W, self.H)
        return Detector, MapMatcher

    # ------------------------------------------------------------------------------------------ landmarks
    def _grow(self, need):
        cap = len(self.lm_alive)
        if self.n_lm + need <= cap:
            return
        new = max(2 * cap, self.n_lm + need)

        def g(a, fill=0):
            out = np.full((new,) + a.shape[1:], fill, a.dtype)
            out[:cap] = a
            return out
        self.lm_p = g(self.lm_p)
        self.lm_host = g(self.lm_host, -1)
        self.lm_alive = g(self.lm_alive, False)
        self.lm_rho_meas = g(self.lm_rho_meas)
        self.lm_sig_lk = g(self.lm_sig_lk, np.inf)
        self.lm_span = g(self.lm_span)
        self.lm_desc = g(self.lm_desc)
        self.lm_ndesc = g(self.lm_ndesc)
        self.lm_Sl = g(self.lm_Sl)

    def _new_slot(self, kind, R, p, fidx, t):
        s = {"sid": self._next_sid, "kind": kind, "R": np.array(R, float), "p": np.array(p, float),
             "fidx": int(fidx), "t": float(t), "lid": np.zeros(0, np.int64), "uv": np.zeros((0, 2)),
             "cam": np.zeros(0, np.int64), "valid": np.zeros(0, bool), "sig": np.zeros(0),
             "full": False, "v": np.zeros(3), "bg": np.zeros(3), "ba": np.zeros(3)}
        self._next_sid += 1
        return s

    def _add_obs(self, s, lid, uv, cam, sig):
        s["lid"] = np.concatenate([s["lid"], np.asarray(lid, np.int64)])
        s["uv"] = np.concatenate([s["uv"], np.asarray(uv, float).reshape(-1, 2)])
        s["cam"] = np.concatenate([s["cam"], np.asarray(cam, np.int64)])
        s["valid"] = np.concatenate([s["valid"], np.ones(len(lid), bool)])
        s["sig"] = np.concatenate([s["sig"], np.asarray(sig, float)])

    def _create_landmarks(self, s, kp, d, sig_lk, desc, span=None):
        """Landmarks hosted in keyframe slot s from stereo-measured keypoints (d px > 0)."""
        n = len(kp)
        if n == 0:
            return np.zeros(0, np.int64)
        self._grow(n)
        ids = np.arange(self.n_lm, self.n_lm + n, dtype=np.int64)
        self.n_lm += n
        a = (kp[:, 0] - self.cx) / self.fx
        b = (kp[:, 1] - self.cy) / self.fy
        rho = d / self.fB
        self.lm_p[ids] = np.stack([a, b, rho], axis=1)
        self.lm_host[ids] = s["sid"]
        self.lm_alive[ids] = True
        self.lm_rho_meas[ids] = rho
        self.lm_sig_lk[ids] = sig_lk
        self.lm_span[ids] = 0.0 if span is None else span
        self.lm_desc[ids, 0] = desc
        self.lm_ndesc[ids] = 1
        self.lm_Sl[ids] = 0.0
        self._add_obs(s, ids, kp, np.zeros(n, np.int64), np.full(n, self.sig_px))
        if self.stereo_mode == "ur":
            ur = np.stack([kp[:, 0] - d, kp[:, 1]], axis=1)
            self._add_obs(s, ids, ur, np.ones(n, np.int64), sig_lk)      # raw LK sigma; x kappa_d at assembly
        self.stats["landmarks_created"] += n
        return ids

    # ------------------------------------------------------------------------------------------ problem
    def _problem(self, lm_select=None, imu_sel=None, with_prior=True, posefree=False):
        """imu_sel: None = every IMU factor of the window, else the indices into self.imu_factors to include.
        posefree (P1.2 W3 item 1; the window solve only, never a marginalisation): landmarks whose ONLY valid observation
        is their host's left one (stereo form "prior") leave the Schur arrays and ride in the Problem's `pf` block
        (window.PoseFree; window.solve treats them exactly).  The kept landmarks keep their order."""
        slots = self.slots
        sid2i = {s["sid"]: i for i, s in enumerate(slots)}
        alive = np.nonzero(self.lm_alive[:self.n_lm])[0]
        if lm_select is not None:
            alive = alive[lm_select(alive)]
        L = len(alive)
        loc = np.full(self.n_lm, -1, np.int64)
        loc[alive] = np.arange(L)
        lids, uvs, cams, vals, sigs, poses, srcs = [], [], [], [], [], [], []
        for i, s in enumerate(slots):
            n = len(s["lid"])
            if n == 0:
                continue
            lids.append(s["lid"])
            uvs.append(s["uv"])
            cams.append(s["cam"])
            vals.append(s["valid"])
            sigs.append(s["sig"])
            poses.append(np.full(n, i, np.int64))
            srcs.append(np.stack([np.full(n, i, np.int64), np.arange(n, dtype=np.int64)], axis=1))
        if lids:
            lid = np.concatenate(lids)
            uv = np.concatenate(uvs)
            cam = np.concatenate(cams)
            val = np.concatenate(vals)
            sig = np.concatenate(sigs)
            pose = np.concatenate(poses)
            src = np.concatenate(srcs)
        else:
            lid = np.zeros(0, np.int64)
            uv = np.zeros((0, 2))
            cam = np.zeros(0, np.int64)
            val = np.zeros(0, bool)
            sig = np.zeros(0)
            pose = np.zeros(0, np.int64)
            src = np.zeros((0, 2), np.int64)
        local = loc[lid] if len(lid) else np.zeros(0, np.int64)
        keep = local >= 0
        local, uv, cam, val, sig, pose, src = local[keep], uv[keep], cam[keep], val[keep], sig[keep], pose[keep], \
            src[keep]
        order = np.argsort(local, kind="stable")
        local, uv, cam, val, sig, pose, src = local[order], uv[order], cam[order], val[order], sig[order], \
            pose[order], src[order]
        ob_start = np.zeros(L + 1, np.int64)
        if L:
            np.cumsum(np.bincount(local, minlength=L), out=ob_start[1:])
        host = np.array([sid2i[int(h)] for h in self.lm_host[alive]], np.int64) if L else np.zeros(0, np.int64)
        if self.stereo_mode == "prior":
            rm = self.lm_rho_meas[alive]
            # P1-03d: an AMBIGUOUS disparity carries its own span (|best - next basin|) as a sigma floor (Class C)
            sg = np.maximum(self.kappa_d * self.lm_sig_lk[alive], self.lm_span[alive]) / self.fB
            ris = np.where(np.isfinite(sg) & (sg > 0) & (rm > 0), 1.0 / np.where(sg > 0, sg, 1.0), 0.0)
        else:
            rm = np.zeros(L)
            ris = np.zeros(L)
        pf_blk = None
        pf_lids = alive[:0]
        if posefree and self.stereo_mode == "prior" and L:
            # P1.2 W3 item 1: POSE-FREE = exactly one valid observation, the host's LEFT one, and none
            # in another slot (its pose Jacobians are zero at t == h: it adds nothing to the Schur system)
            hl = host[local]
            n_valid = np.bincount(local[val], minlength=L)
            n_nonhost = np.bincount(local[val & (pose != hl)], minlength=L)
            lh = val & (pose == hl) & (cam == 0)
            n_lh = np.bincount(local[lh], minlength=L)
            pfm = (n_nonhost == 0) & (n_valid == 1) & (n_lh == 1)
            if pfm.any():
                z = np.zeros((L, 2))
                z[local[lh]] = uv[lh]
                pf_lids = alive[pfm]
                pf_blk = Wn.PoseFree(host[pfm], z[pfm], 1.0 / self.sig_px, rm[pfm], ris[pfm], self.lm_p[pf_lids])
                kl = ~pfm
                om = kl[local]
                newloc = np.cumsum(kl) - 1
                local, uv, cam, val, sig, pose, src = newloc[local[om]], uv[om], cam[om], val[om], sig[om], pose[om], \
                    src[om]
                alive, host, rm, ris = alive[kl], host[kl], rm[kl], ris[kl]
                L = len(alive)
                ob_start = np.zeros(L + 1, np.int64)
                if L:
                    np.cumsum(np.bincount(local, minlength=L), out=ob_start[1:])
        dims = np.array([15 if (self.vi and x["full"]) else 6 for x in slots], np.int64)
        off = np.zeros(len(slots) + 1, np.int64)
        np.cumsum(dims, out=off[1:])
        prior = self.prior if with_prior else None
        if prior is not None:
            if all(sd in sid2i for sd in prior.sids):
                prior.bind(sid2i, off, dims)
            else:
                raise RuntimeError("prior refers to a slot that left the window")
        vel = bgs = bas = None
        imu = []
        if self.vi:
            vel = np.ascontiguousarray(np.stack([x["v"] for x in slots]))
            bgs = np.ascontiguousarray(np.stack([x["bg"] for x in slots]))
            bas = np.ascontiguousarray(np.stack([x["ba"] for x in slots]))
            for k, f in enumerate(self.imu_factors):
                if imu_sel is not None and k not in imu_sel:
                    continue
                imu.append((sid2i[f["i"]], sid2i[f["j"]], f["pre"], f["W"]))
        pb = Wn.Problem(Rwb=np.ascontiguousarray(np.stack([s["R"] for s in slots])),
                        pwb=np.ascontiguousarray(np.stack([s["p"] for s in slots])),
                        Rbc=self.Rbc, pbc=self.pbc, lmp=np.ascontiguousarray(self.lm_p[alive]),
                        lm_host=host, lm_mask=np.ones(L, np.bool_), rho_meas=np.ascontiguousarray(rm),
                        rho_isig=np.ascontiguousarray(ris), ob_start=ob_start, ob_pose=np.ascontiguousarray(pose),
                        ob_cam=np.ascontiguousarray(cam), ob_z=np.ascontiguousarray(uv),
                        # the CURRENT Class-C scales apply to every observation: sigma_px (left), kappa_d x the
                        # observation's own LK sigma (u_R); both held fixed inside one LM pass
                        ob_isig=np.ascontiguousarray(np.where(cam == 0, 1.0 / self.sig_px,
                                                              1.0 / np.maximum(self.kappa_d * sig, 1e-9))),
                        ob_valid=np.ascontiguousarray(val), fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, B=self.B,
                        huber_k=HUBER_K, prior=prior, dims=dims if self.vi else None, vel=vel, bgs=bgs, bas=bas,
                        imu=imu, g_w=self.g_w, pf=pf_blk)
        return pb, {"alive": alive, "src": src, "sid2i": sid2i, "local": local, "off": off, "dims": dims,
                    "pf_lids": pf_lids}

    def _writeback(self, pb, meta):
        for i, s in enumerate(self.slots):
            s["R"] = pb.Rwb[i].copy()
            s["p"] = pb.pwb[i].copy()
            if pb.vel is not None and s["full"]:
                s["v"] = pb.vel[i].copy()
                s["bg"] = pb.bgs[i].copy()
                s["ba"] = pb.bas[i].copy()
        self.lm_p[meta["alive"]] = pb.lmp
        if getattr(pb, "pf", None) is not None:          # P1.2 W3 item 1: the pose-free landmarks' solved states
            self.lm_p[meta["pf_lids"]] = pb.pf.lmp

    # ------------------------------------------------------------------------------------------ window solve
    def _solve(self):
        pb, meta = self._problem(posefree=self.w3)
        if self.dump_dir and self.n_dumped < 500 and self.stats["tracked"] % self.dump_every == 0:
            self._dump(pb)
        self.n_solves += 1
        if self.pkl_dir and self.n_pkl < 500 and self.n_solves % self.pkl_every == 0:
            import pickle
            os.makedirs(self.pkl_dir, exist_ok=True)
            with open(os.path.join(self.pkl_dir, "pb_%05d.pkl" % self.n_pkl), "wb") as f:
                pickle.dump(pb, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.n_pkl += 1
        P, L, M = pb.Rwb.shape[0], pb.lmp.shape[0], len(pb.ob_pose)
        self.stats["window_P"].append(P)
        self.stats["window_L"].append(L)                  # the SOLVED (Schur) landmarks
        self.stats["window_L_posefree"].append(len(meta["pf_lids"]))   # P1.2 W3 item 1: the excluded ones
        self.stats["window_M"].append(M)
        with self._stage("window"):
            res = Wn.solve(pb, iters=LM_ITERS)
        with self._stage("window_final"):
            Hf, _bf, chi2, lm_ok, Hll_inv, imu_chi2, off = Wn.final_linearisation(pb)
        self._last_imu_chi2 = [float(x) for x in imu_chi2]
        self._writeback(pb, meta)
        if res["cost0"]:
            self.stats["cost_ratio"].append(float(res["cost"] / max(res["cost0"], 1e-12)))
        # outlier removal (chi2 0.999 of the observation's dof) and the Class-C robust scales
        src = meta["src"]
        cam = pb.ob_cam
        thr = np.where(cam == 0, CHI2_2_999, CHI2_1_999)
        local = meta["local"]
        is_host = pb.ob_pose == pb.lm_host[local] if len(local) else np.zeros(0, bool)
        bad = pb.ob_valid & np.isfinite(chi2) & (chi2 > thr) & ~(is_host & (cam == 0))
        for (i, k) in src[bad]:
            self.slots[i]["valid"][k] = False
        self.stats["outliers_removed"] += int(bad.sum())
        # ---- Class-C noise scales (P1-05b).  Post-fit residuals are biased low by the parameters they fitted, so:
        # sigma_px: the DOF-CORRECTED pooled variance over landmarks with >= 3 valid left observations,
        #           s^2 = sum chi2 / sum (2 n_l - 3)  (3 landmark parameters absorbed per landmark; the shared pose
        #           parameters are ignored, a slight under-estimate);
        # kappa_d (prior form): the INNOVATION of the stereo prior against the multi-view inverse depth,
        #           z = r0 sqrt(I_tot / I_mv), I_tot = 1 / Hll_inv[2,2] (prior included), I_p = isig^2, I_mv = I_tot - I_p,
        #           over landmarks whose multi-view information exceeds the prior's (z ~ N(0,1) if sigma_rho is right);
        # kappa_d (u_R form): pooled s^2 = sum chi2_R / sum (n_R - 1) over landmarks with >= 3 right observations.
        # Each new estimate enters a running median over the window length; scales stay fixed inside an LM pass.
        frozen = getattr(self, "freeze_scales", False)          # tests only (NEES with the true noise)
        valid = pb.ob_valid & ~bad & np.isfinite(chi2)
        if self.frame_log:
            # P1.2 diagnostics (W1 / W2 inputs): the newest frame's own left-observation chi2 and the window's
            i_new = meta["sid2i"][self._newest()["sid"]]
            vl = valid & (cam == 0)
            mine = vl & (src[:, 0] == i_new) if len(src) else vl
            self._fl["vis_frame_n"] = int(mine.sum())
            self._fl["vis_frame_chi2_sum"] = float(chi2[mine].sum()) if mine.any() else 0.0
            self._fl["vis_frame_chi2_med"] = float(np.median(chi2[mine])) if mine.any() else None
            self._fl["vis_win_n"] = int(vl.sum())
            self._fl["vis_win_chi2_sum"] = float(chi2[vl].sum()) if vl.any() else 0.0
            self._fl["outliers_removed_now"] = int(bad.sum())
        alive = meta["alive"]
        Lw = len(alive)
        if not frozen and Lw:
            v0 = valid & (cam == 0)
            n0 = np.bincount(local[v0], minlength=Lw)
            c0 = np.bincount(local[v0], weights=chi2[v0], minlength=Lw)
            m = n0 >= 3
            dof = float((2 * n0[m] - 3).sum())
            if dof >= 2 * KF_COUNT:
                s2 = float(c0[m].sum()) / dof
                self._sig_hist.append(self.sig_px * float(np.sqrt(s2)))
                self._sig_hist = self._sig_hist[-SCALE_WINDOW:]
                self.sig_px = float(np.median(self._sig_hist))
            if self.stereo_mode == "ur":
                v1 = valid & (cam == 1)
                n1 = np.bincount(local[v1], minlength=Lw)
                c1 = np.bincount(local[v1], weights=chi2[v1], minlength=Lw)
                m1 = n1 >= 3
                dof1 = float((n1[m1] - 1).sum())
                if dof1 >= 2 * KF_COUNT:
                    self._kd_hist.append(self.kappa_d * float(np.sqrt(float(c1[m1].sum()) / dof1)))
                    self._kd_hist = self._kd_hist[-SCALE_WINDOW:]
                    self.kappa_d = max(1.0, float(np.median(self._kd_hist)))   # sigma_LK is the CRB of the 1-D fit: a floor (P1-05e)
            else:
                Ip = pb.rho_isig ** 2
                with np.errstate(divide="ignore", invalid="ignore"):
                    Itot = np.where(lm_ok, 1.0 / Hll_inv[:, 2, 2], 0.0)
                Imv = Itot - Ip
                sel = lm_ok & (Ip > 0) & np.isfinite(Itot) & (Imv > Ip)
                if sel.sum() >= 2 * KF_COUNT:
                    r0 = (pb.lmp[sel, 2] - pb.rho_meas[sel]) * pb.rho_isig[sel]
                    z = r0 * np.sqrt(Itot[sel] / Imv[sel])
                    sz = float(MAD_K * np.median(np.abs(z)))
                    if sz > 0 and np.isfinite(sz):
                        self._kd_hist.append(self.kappa_d * sz)
                        self._kd_hist = self._kd_hist[-SCALE_WINDOW:]
                        self.kappa_d = max(1.0, float(np.median(self._kd_hist)))   # sigma_LK is the CRB of the 1-D fit: a floor (P1-05e)
        # kappa_imu (Class C, DESIGN_P1 §6.3; vi_init.kappa_imu: the median normalised chi2 of the window's IMU
        # factors, whitened at the kappa in use; floored at 1 inside the estimator), running median over the window
        # P2-fix-2: kappa_imu is estimated in _marginalise_full from the last SCALE_WINDOW MARGINALISED factors, pooled
        # at kappa = 1 (DESIGN_P1 6.3).  The former per-solve update multiplied the running value by the FLOORED
        # estimate (>= 1), so it could only rise: a ratchet (madmax_E1 15 -> 844).
        # landmark marginals for gating (given the poses)
        self.lm_Sl[meta["alive"][lm_ok]] = Hll_inv[lm_ok]
        if getattr(pb, "pf", None) is not None:          # P1.2 W3 item 1: the pose-free landmarks' marginals
            self.lm_Sl[meta["pf_lids"][pb.pf.ok]] = pb.pf.Hinv[pb.pf.ok]
        # the newest pose's covariance RELATIVE TO THE MAP the gate projects: conditioned on the oldest window keyframe
        # (Sigma_n|0 = Sigma_nn - Sigma_n0 Sigma_00^-1 Sigma_0n).  The absolute marginal also carries the whole
        # trajectory's drift, which the window's landmarks share; gating with it would widen every gate with distance.
        n = Hf.shape[0]
        try:
            C = np.linalg.inv(Hf + 1e-9 * np.eye(n))
        except np.linalg.LinAlgError:
            C = np.linalg.pinv(Hf)
        if self._p13 is not None:      # P1.3 H1: the latest window solve, by reference (read by the link only)
            self._p13_cov = {"C": C, "off": off, "sid2i": meta["sid2i"], "Rwb": pb.Rwb, "pwb": pb.pwb,
                             "sids": [x["sid"] for x in self.slots], "kinds": [x["kind"] for x in self.slots],
                             "fidx": [x["fidx"] for x in self.slots], "ts": [x["t"] for x in self.slots],
                             "n_solve": self.n_solves}
        inew = meta["sid2i"][self._newest()["sid"]]
        j = int(off[inew])
        dj = int(off[inew + 1] - off[inew])
        Snn = C[j:j + dj, j:j + dj]
        if inew > 0:
            S00 = C[0:6, 0:6]
            Sn0 = C[j:j + dj, 0:6]
            Sig = Snn - Sn0 @ np.linalg.solve(S00 + 1e-15 * np.eye(6), Sn0.T)
        else:
            Sig = Snn
        Sig = 0.5 * (Sig + Sig.T)
        self.Sig_full = Sig if dj == 15 else None
        return Sig[:6, :6].copy(), res

    # ------------------------------------------------------------------------------------------ window upkeep
    def _newest(self):
        return max(self.slots, key=lambda x: x["t"])

    def _fulls(self):
        return sorted([x for x in self.slots if x["full"]], key=lambda x: x["t"])

    def _frames(self):
        return [s for s in self.slots if s["kind"] == "fr"]

    def _kfs(self):
        return [s for s in self.slots if s["kind"] == "kf"]

    def _reorder(self):
        self.slots = self._kfs() + self._frames()

    def _drop_frame(self, s):
        if self.prior is not None and s["sid"] in self.prior.sids:
            self.prior = self.prior.drop_slot(s["sid"])
        self.slots = [x for x in self.slots if x is not s]
        self.stats["drop_frame"] += 1

    def _marg_into_prior(self, pb, meta, H, b, marg_cols, drop_sid=None):
        """Schur the columns marg_cols out of (H, b) (linearised at the current estimate) and make the new prior over
        every block still connected (nonzero diagonal), FEJ points kept (marg.combine)."""
        off = meta["off"]
        dims = meta["dims"]
        P = len(self.slots)
        mset = set(int(c) for c in marg_cols)
        blocks = []
        for i in range(P):
            cols = [c for c in range(int(off[i]), int(off[i + 1])) if c not in mset]
            if not cols:
                continue
            if np.abs(np.diag(H)[cols]).sum() > 0:
                blocks.append((i, cols))
        if not blocks:
            self.prior = None
            return
        keep = np.concatenate([np.array(c, np.int64) for _i, c in blocks])
        H2, b2 = schur(H, b, keep, np.asarray(sorted(mset), np.int64))
        sids, bdims, states = [], [], []
        for i, cols in blocks:
            x = self.slots[i]
            sids.append(x["sid"])
            bdims.append(len(cols))
            states.append((x["R"], x["p"], x["v"], x["bg"], x["ba"]))
        self.prior = combine(self.prior, sids, bdims, states, H2, b2)

    def _marginalise_kf(self, s):
        """Oldest keyframe + its hosted landmarks -> the square-root prior (Schur), FEJ points kept.  Only the visual
        factors of its hosted landmarks and the prior enter (its other observations are dropped, OKVIS / Basalt)."""
        if self._p13 is not None:      # P1.3 H3: its FINAL odometry edge to the next keyframe (read only)
            self._p13.edge_final(self, s)
        sid = s["sid"]
        hosted = lambda alive: self.lm_host[alive] == sid      # noqa: E731
        pb, meta = self._problem(lm_select=hosted, imu_sel=[])
        H, b, _off = Wn.marg_system(pb, np.ones(pb.lmp.shape[0], np.bool_))
        i_m = meta["sid2i"][sid]
        off = meta["off"]
        self._marg_into_prior(pb, meta, H, b, range(int(off[i_m]), int(off[i_m + 1])))
        # remove the KF, its hosted landmarks and every observation of them
        dead = np.nonzero(self.lm_alive[:self.n_lm] & (self.lm_host[:self.n_lm] == sid))[0]
        self.lm_alive[dead] = False
        deadset = np.zeros(self.n_lm, bool)
        deadset[dead] = True
        self.slots = [x for x in self.slots if x is not s]
        for x in self.slots:
            if len(x["lid"]):
                k = ~deadset[x["lid"]]
                for key in ("lid", "uv", "cam", "valid", "sig"):
                    x[key] = x[key][k]
        self.stats["marg_kf"] += 1

    def _marginalise_full(self, o):
        """VI: the oldest full state leaves the newest-N_FR buffer.  Its IMU factor to the next full state and the
        prior enter the marginalisation (FEJ).  A keyframe keeps its pose (v, bg, ba are marginalised); a frame is
        marginalised whole and its visual rows are dropped (OKVIS / Basalt)."""
        ks = [k for k, f in enumerate(self.imu_factors) if f["i"] == o["sid"] or f["j"] == o["sid"]]
        pb, meta = self._problem(lm_select=lambda alive: np.zeros(len(alive), bool), imu_sel=ks)
        if not getattr(self, "freeze_scales", False) and ks:
            c2 = Wn._imu_terms(pb, Wn._state(pb), Wn._offsets(pb)[1], 0, want_system=False)[3]
            for k, c in zip(ks, c2):
                if np.isfinite(c):
                    self._kimu_buf.append(float(c) * float(self.imu_factors[k]["kappa"]))
            self._kimu_buf = self._kimu_buf[-SCALE_WINDOW:]
            if len(self._kimu_buf) >= SCALE_WINDOW:
                from xclean.vio import vi_init as V
                kk = V.kappa_imu(np.asarray(self._kimu_buf, float), 15.0)
                if kk is not None and np.isfinite(kk.kappa):
                    self.kappa_imu = float(kk.kappa)
                    self.stats["kappa_imu_updates"] += 1
        H, b, _off = Wn.marg_system(pb, np.zeros(0, np.bool_))
        i_m = meta["sid2i"][o["sid"]]
        off = meta["off"]
        if o["kind"] == "kf":
            cols = range(int(off[i_m]) + 6, int(off[i_m + 1]))
        else:
            cols = range(int(off[i_m]), int(off[i_m + 1]))
        self._marg_into_prior(pb, meta, H, b, cols)
        self.imu_factors = [f for k, f in enumerate(self.imu_factors) if k not in ks]
        o["full"] = False
        if o["kind"] != "kf":
            self.slots = [x for x in self.slots if x is not o]
        self.stats["marg_full"] += 1

    def _upkeep(self):
        if self.vi:
            while len(self._fulls()) > N_FR:
                self._marginalise_full(self._fulls()[0])
        while len(self._frames()) > N_FR:
            fr0 = self._frames()[0]
            if self.vi and fr0["full"]:
                break
            self._drop_frame(fr0)
        while len(self._kfs()) > N_KF:
            k0 = self._kfs()[0]
            if k0["full"]:
                break
            self._marginalise_kf(k0)

    # ------------------------------------------------------------------------------------------ helpers
    class _Null:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _stage(self, name):
        if self.led is None:
            return P1Engine._Null()
        return self.led.stage(name, frame=self._fidx)

    def _dump(self, pb):
        os.makedirs(self.dump_dir, exist_ok=True)
        np.savez_compressed(os.path.join(self.dump_dir, "win_%05d.npz" % self.n_dumped),
                            Rwb=pb.Rwb, pwb=pb.pwb, Rbc=pb.Rbc, pbc=pb.pbc, lmp=pb.lmp, lm_host=pb.lm_host,
                            rho_meas=pb.rho_meas, rho_isig=pb.rho_isig, ob_start=pb.ob_start, ob_pose=pb.ob_pose,
                            ob_cam=pb.ob_cam, ob_z=pb.ob_z, ob_isig=pb.ob_isig, ob_valid=pb.ob_valid,
                            cam=np.array([pb.fx, pb.fy, pb.cx, pb.cy, pb.B]),
                            prior_S=(pb.prior.S if pb.prior is not None else np.zeros((0, 0))),
                            prior_r=(pb.prior.r if pb.prior is not None else np.zeros(0)),
                            prior_idx=(np.asarray(pb.prior._idx, np.int64) if pb.prior is not None
                                       else np.zeros(0, np.int64)),
                            prior_dims=(np.asarray(pb.prior.dims, np.int64) if pb.prior is not None
                                        else np.zeros(0, np.int64)),
                            prior_R0=(np.stack([x[0] for x in pb.prior.x0]) if pb.prior is not None
                                      else np.zeros((0, 3, 3))),
                            prior_p0=(np.stack([x[1] for x in pb.prior.x0]) if pb.prior is not None
                                      else np.zeros((0, 3))))
        self.n_dumped += 1

    def _host_cams(self, lids):
        """Camera rotation / centre of each landmark's host keyframe (vectorised over unique hosts)."""
        hosts = self.lm_host[lids]
        uh, inv = np.unique(hosts, return_inverse=True)
        sid2s = {s["sid"]: s for s in self.slots}
        Ru = np.stack([sid2s[int(h)]["R"] @ self.Rbc for h in uh])
        pu = np.stack([sid2s[int(h)]["p"] + sid2s[int(h)]["R"] @ self.pbc for h in uh])
        return Ru[inv], pu[inv]

    def _world_points(self, lids):
        """World positions of landmarks (for PnP)."""
        Rwc, pwc = self._host_cams(lids)
        a, b, rho = self.lm_p[lids, 0], self.lm_p[lids, 1], self.lm_p[lids, 2]
        m = np.stack([a / rho, b / rho, 1.0 / rho], axis=1)
        return pwc + np.einsum("nij,nj->ni", Rwc, m)

    def _stereo(self, left, right, kp):
        with self._stage("stereo"):
            m = _stereo_module()
            if STEREO_IMPL != "v1":
                return m.disparities(left, right, kp)
            out = m.disparities(left, right, kp, dy=self.dy_hat, band=self.dy_band)
            self._update_dy(m.disparities.last_dy, out[2])
            return out

    def _stereo_ex(self, left, right, kp):
        with self._stage("stereo"):
            m = _stereo_module()
            if STEREO_IMPL != "v1":
                return m.disparities_ex(left, right, kp)
            out = m.disparities_ex(left, right, kp, dy=self.dy_hat, band=self.dy_band)
            self._update_dy(m.disparities.last_dy, out[2])
            return out

    def _update_dy(self, dys, ok):
        """P1.1 D4 (OPEN-1): the class-C vertical rectification residual.  Each stereo call with >= KF_COUNT accepted
        points (the count a stereo init needs, carried) adds its points' own vertical offsets (stereo1d column scan);
        over the last SCALE_WINDOW such calls dy_hat = the pooled median, sigma = 1.4826 MAD, and the 1-D search covers
        the rows dy_hat +- K_BAND sigma: band = round(K_BAND sigma) integer rows around the shifted row, at most HALF
        (the column scan's own range).  K_BAND = sqrt(chi2_1(P_GATE)) (class-D level)."""
        from xclean.stereo.epipolar import HALF as _H
        if "d4" in P11_OFF:
            return
        v = np.asarray(dys, float)[np.asarray(ok, bool)]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return
        # every call's accepted points enter (a starved init still measures its few points: the estimate is what lifts
        # the starvation); the estimate needs >= KF_COUNT pooled points over the last SCALE_WINDOW calls
        self._dy_pool.append(v)
        del self._dy_pool[:-SCALE_WINDOW]
        pool = np.concatenate(self._dy_pool)
        if len(pool) < KF_COUNT:
            return
        self.dy_hat = float(np.median(pool))
        self.dy_sigma = float(MAD_K * np.median(np.abs(pool - self.dy_hat)))
        self.dy_band = int(min(round(K_BAND * self.dy_sigma), _H))
        self.stats["dy_updates"] += 1
        self.stats["dy_band_max"] = max(self.stats["dy_band_max"], self.dy_band)

    def _predict(self, t):
        from xclean.core.lie import se3_exp, se3_log
        if self.t_last is None:
            return self.T_last.copy(), None
        dt = float(t) - self.t_last
        T = self.T_last.copy()
        if self.D_body is not None and self.dt_vel and self.dt_vel > 0 and dt > 0:
            s = dt / self.dt_vel
            T = self.T_last @ (self.D_body if s == 1.0 else se3_exp(s * se3_log(self.D_body)))
        if self.noise is not None and self.imu is not None and dt > 0:
            # VO with an IMU not (yet) admitted: the ROTATION comes from the gyro (causal preintegration over the real
            # interval, at the bootstrap bias), the translation from constant velocity in time (P1-08b)
            from xclean.vio import imu_factor as F
            bg0 = np.asarray(getattr(self.imu, "bias", np.zeros(3)), float).reshape(3)
            try:
                pre = F.preintegrate(self.imu.t, self.imu.g, self.imu.a, self.t_last, float(t), self._imu_horizon(),
                                     bg0, np.zeros(3), None)
                if pre.ok:
                    T = T.copy()
                    T[:3, :3] = self.T_last[:3, :3] @ pre.dR
            except ValueError:
                pass
        Sig = None
        if self.Sig_last is not None and len(self._q_hist) >= 3 and dt > 0:
            Q = np.mean(np.stack(self._q_hist), axis=0)
            Sig = self.Sig_last + Q * dt ** 3
        return T, Sig

    # ------------------------------------------------------------------------------------------ init / segments
    def _try_init(self, fr, left, right, kp, desc, T_start):
        d, sig, ok, amb, span, _zn = self._stereo_ex(left, right, kp)
        if int(ok.sum()) < KF_COUNT:
            return False
        use = ok
        self.slots = []
        self.prior = None
        self.lm_alive[:self.n_lm] = False
        s = self._new_slot("kf", T_start[:3, :3], T_start[:3, 3], fr.index, fr.t)
        self.slots.append(s)
        self._create_landmarks(s, kp[use].astype(float), d[use], sig[use], desc[use], span[use])
        self.prior = gauge_prior(s["sid"], s["R"], s["p"])
        self.n_last_kf_obs = 0
        self._kf_hist = [(s["sid"], float(fr.t), _se3(s["R"], s["p"]), self._imu_horizon())]
        if not self.vi:
            self.vi_refused = False          # a refusal holds for its segment only (DESIGN_P1 §6.2)
            self._vi_prev = None
        self.imu_factors = []
        if self.vi:
            # VI continues across a new segment (DESIGN_P1 §7): the propagated state seeds KF0 as a full state
            s["full"] = True
            s["v"], s["bg"], s["ba"] = self._carry
            if self.Sig_full is not None:
                Hb = np.linalg.inv(self.Sig_full + 1e-12 * np.eye(15))
                Hb[:6, :] = 0.0
                Hb[:, :6] = 0.0
                self.prior = add_block_info(self.prior, s["sid"], 15, (s["R"], s["p"], s["v"], s["bg"], s["ba"]),
                                            Hb, np.zeros(15))
        self.segment += 1
        self.stats["segments"] += 1
        self.stats["kfs"] += 1
        self.D_body = None
        self.dt_vel = None
        self.Sig_last = None
        self._q_hist = []
        # P1.1 D1 also at a segment's first keyframe (the platform may still be standing when vision first initialises)
        if self.noise is not None and not self.vi and not self._d1_used and not self._demoted and "d1" not in P11_OFF:
            self._try_d1(s)
        if self._p13 is not None:      # P1.3 H4: the segment link (set by the caller's line) + the KF0 packet
            self._p13.segment_start(self, self._p13_link, s, left, right)
            self._p13_link = None
        return True

    # ------------------------------------------------------------------------------------------ the frame
    def step(self, fr):
        """One frame (see _step); with a frame log, one JSON line per frame (P1.2 W1 diagnostics).
        P1.2: W1 runs once per frame on every path (right after the window solve where one ran, so a demotion acts
        before the keyframe decision; at the end of the frame otherwise)."""
        self._fl = {}
        n_ev = len(self.stats["events"])
        self._w1_done = False
        self._w1_last = None
        self._stage_a = (0, 0, False, None)
        self._n_solves0 = self.n_solves
        st = self._step(fr)
        if not self._w1_done:                    # init / coast paths: no window solve this frame (c_new None)
            if self._w1_update(fr) and self.vi:
                self._demote_nopose("w1")
                st = self.state
        if self.vi:
            self.stats["vi_frames"] += 1
        if self.frame_log:
            self._log_frame(fr, st, n_ev)
        return st

    def _log_frame(self, fr, st, n_ev):
        import json
        fl = self._fl
        rec = {"frame": int(fr.index), "t": float(fr.t), "state": st, "vi": bool(self.vi), "vi_path": self.vi_path,
               "segment": int(self.segment), "kappa_imu": float(self.kappa_imu), "sig_px": float(self.sig_px),
               "kappa_d": float(self.kappa_d), "imu_chi2": [float(x) for x in self._last_imu_chi2],
               "solved": bool(self.n_solves > self._n_solves0), "stage_a": list(self._stage_a),
               "events": self.stats["events"][n_ev:]}
        rec.update(fl)
        line = json.dumps(rec, default=float) + "\n"
        if self.log_writer is not None:
            self.log_writer.write(self.frame_log, line)
        else:
            with open(self.frame_log, "a", encoding="utf-8") as f:
                f.write(line)

    def _step(self, fr):
        self._fidx = int(fr.index)
        self.stats["frames"] += 1
        with self._stage("rectify"):
            left, right = self.src.images(fr.index)
        with self._stage("extract"):
            kp, desc = self.det(left)
        self._fl["n_kp"] = int(len(kp))
        if self.imu is not None and hasattr(self.imu, "advance"):
            self.imu.advance(fr.t)
        self._frame_t.append(float(fr.t))
        del self._frame_t[:-(SCALE_WINDOW + 1)]
        if self.vi:
            self._repreintegrate()
        if not self.slots:
            if self._p13 is not None:  # P1.3 H4d: no covariance links this segment to the previous one
                self._p13_link = ("unknown", None)
            ok = self._try_init(fr, left, right, kp, desc, self.T_last)
            self.state = "tracking" if ok else "init"
            if not ok:
                self.stats["init_frames"] += 1
            self.t_last = float(fr.t)
            self.T_wc = self.T_last @ _se3(self.Rbc, self.pbc)
            return self.state
        pred = None
        if self.vi:
            pred = self._predict_vi(fr.t)
        if pred is not None and pred.state is not None:
            T_pred = _se3(pred.state.R, pred.state.p)
            Sig_pred = None if pred.P is None else 0.5 * (pred.P[:6, :6] + pred.P[:6, :6].T)
        else:
            T_pred, Sig_pred = self._predict(fr.t)
        # ---- association + pose (P1-07c): (A) prediction-free descriptor matching against the window's landmarks
        # + PnP-RANSAC gives the pose hypothesis, robust to a wrong motion prediction; (B) every window landmark is
        # projected with THAT pose and matched inside its covariance gate (ORB-SLAM's TrackLocalMap mechanism,
        # our own code); (C) PnP-RANSAC + whole-correspondence verify on the union.  Only when (A) fails does the
        # constant-velocity prediction carry the gate (the measured mh_04 take-off: a CV-gated first stage locked
        # onto in-gate wrong neighbours and walked 0.7 m off in 5 frames; prediction-free (A) tracked it).
        alive = np.nonzero(self.lm_alive[:self.n_lm])[0]
        with self._stage("match"):
            T_new, m_lid, m_kp, n_in_view, mode = self._track(fr, kp, desc, T_pred, Sig_pred, alive)
        if self.frame_log:
            self._fl.update(mode=mode, n_match=int(len(m_lid)), n_in_view=int(n_in_view),
                            pred_src="imu" if (pred is not None and pred.state is not None) else "cv")
            if T_new is not None:
                # the visual (PnP) pose against the prediction, before the window solve: the innovation W1 / W2 read
                dR = T_pred[:3, :3].T @ T_new[:3, :3]
                self._fl["innov_rot_deg"] = float(np.degrees(np.linalg.norm(Wn.so3_log_nb(np.ascontiguousarray(dR)))))
                self._fl["innov_rotvec_body"] = [float(x) for x in Wn.so3_log_nb(np.ascontiguousarray(dR))]
                self._fl["innov_trans_m"] = float(np.linalg.norm(T_new[:3, 3] - T_pred[:3, 3]))
        if T_new is not None:
            self.stats["pnp_inliers"] += int(len(m_lid))
            self.stats["mode_" + mode] = self.stats.get("mode_" + mode, 0) + 1
        if os.environ.get("XCLEAN_P1_DEBUG"):
            dT = None if T_new is None else np.linalg.inv(T_pred) @ T_new
            self.debug.append({"f": int(fr.index), "mode": mode, "n_match": int(len(m_lid)),
                               "sig_tr": None if Sig_pred is None else float(np.trace(Sig_pred[3:, 3:])),
                               "pred_err_m": None if dT is None else float(np.linalg.norm(dT[:3, 3])),
                               "n_kp": int(len(kp)), "n_view": int(n_in_view)})
        if T_new is None:
            self.stats["pnp_fail"] += 1
            if pred is not None and pred.state is not None:
                if "d2" not in P11_OFF:
                    # P1.1 D2: while the IMU is admitted a visual dropout never resets the segment by itself
                    return self._coast_vi(fr, pred, left, right, kp, desc, n_in_view)
                if self.coast_streak <= SCALE_WINDOW:            # pre-P1.1 behaviour (attribution arm only)
                    return self._coast_vi(fr, pred, left, right, kp, desc, n_in_view, d2=False)
                self._carry = (pred.state.v.copy(), pred.state.bg.copy(), pred.state.ba.copy())
            elif self.vi and "d2" not in P11_OFF:
                # the IMU could not carry this frame (propagation failed): demote rather than propagate from an ever
                # older state on later frames
                self._demote_nopose("propagate")
            return self._coast(fr, left, right, kp, desc, T_pred, Sig_pred, n_in_view)
        # ---- the frame enters the window
        s = self._new_slot("fr", T_new[:3, :3], T_new[:3, 3], fr.index, fr.t)
        if pred is not None and pred.state is not None:
            self._attach_imu(s, pred)
        self._add_obs(s, m_lid, kp[m_kp], np.zeros(len(m_lid), np.int64), np.full(len(m_lid), self.sig_px))
        if self.stereo_mode == "ur":
            d, sig, ok, _ = self._stereo(left, right, kp[m_kp].astype(float))
            if ok.any():
                ur = np.stack([kp[m_kp][ok, 0] - d[ok], kp[m_kp][ok, 1]], axis=1)
                self._add_obs(s, m_lid[ok], ur, np.ones(int(ok.sum()), np.int64), sig[ok])
        self.slots.append(s)
        self._reorder()
        Sig, _res = self._solve()
        T_sol = _se3(s["R"], s["p"])
        # ---- motion model: constant velocity in time + its measured residual covariance (Class C; VO)
        if self.t_last is not None and not self.vi:
            dt = float(fr.t) - self.t_last
            if dt > 0:
                if self.D_body is not None:
                    e = np.concatenate([Wn.so3_log_nb(np.ascontiguousarray(T_pred[:3, :3].T @ T_sol[:3, :3])),
                                        T_sol[:3, 3] - T_pred[:3, 3]])
                    self._q_hist.append(np.outer(e, e) / dt ** 3)
                    self._q_hist = self._q_hist[-SCALE_WINDOW:]
                self.D_body = np.linalg.inv(self.T_last) @ T_sol
                self.dt_vel = dt
        self.T_last = T_sol
        self.t_last = float(fr.t)
        self.Sig_last = Sig
        self.coast_streak = 0
        n_obs = int(s["valid"][s["cam"] == 0].sum())
        self.det.report_yield(n_obs, len(kp))
        self.stats["tracked"] += 1
        self.state = "tracking"
        self.T_wc = T_sol @ _se3(self.Rbc, self.pbc)
        self._frame_inliers = n_obs
        # ---- P1.2 W1, right after the solve and before the keyframe decision: a demotion starts a
        # new VO segment at this frame's pose (D3's path; LIVE-ODOM does not jump)
        if self._w1_update(fr):
            self._demote(["w1"], s, left, right, kp, desc)
            return self.state
        # ---- P1.1 D1: IMU-first admission when the IMU says the platform is stationary (provisional; D3 demotes)
        if self.noise is not None and not self.vi and not self._d1_used and not self._demoted and "d1" not in P11_OFF:
            self._try_d1(s)
        # ---- keyframe decision (post-publish work in the live split; serial here)
        # P1-07b/d: the connected count is the frame's matches to landmarks hosted in the NEWEST keyframe (ORB-SLAM's
        # reference-keyframe form; Basalt's ratio 0.7, class D).  Its reference is that count on the FIRST frame after
        # the keyframe (per-frame re-detection never re-finds all of a keyframe's own detections).  Matches to older
        # keyframes' landmarks do not count: they persist and would suppress keyframes while the view moves on.
        kfs = self._kfs()
        newest = kfs[-1]["sid"] if kfs else -1
        vm = (s["cam"] == 0) & s["valid"]
        n_conn = int((self.lm_host[s["lid"][vm]] == newest).sum()) if vm.any() else 0
        if self.n_last_kf_obs <= 0:
            self.n_last_kf_obs = max(n_conn, 1)
        ratio = n_conn / max(self.n_last_kf_obs, 1)
        if n_obs < KF_COUNT or ratio < KF_RATIO:
            with self._stage("kf"):
                self._make_kf(s, left, right, kp, desc, m_kp)
        with self._stage("marg"):
            self._upkeep()
        return self.state

    # ------------------------------------------------------------------------------------------ VI (Phase 2)
    def _imu_horizon(self):
        return float(self.imu.horizon) if self.imu is not None else float("nan")

    def _W(self, pre):
        from xclean.vio import imu_factor as F
        return F.sqrt_information(pre, self.kappa_imu)

    def _repreintegrate(self):
        """A factor whose interval ended in a C40 hold is re-preintegrated once the samples past it are released;
        every factor is re-whitened when kappa_imu moved (judge A3 / Track B's note on rosario_04)."""
        from xclean.vio import imu_factor as F
        hz = self._imu_horizon()
        for f in self.imu_factors:
            pre = f["pre"]
            if pre.extrapolated and hz >= pre.t1:
                f["pre"] = F.preintegrate(self.imu.t, self.imu.g, self.imu.a, pre.t0, pre.t1, hz, pre.bg_lin,
                                          pre.ba_lin, self.noise)
                f["W"] = self._W(f["pre"])
                f["kappa"] = self.kappa_imu
                self.stats["repreint"] += 1
            elif f["kappa"] != self.kappa_imu:
                f["W"] = self._W(pre)
                f["kappa"] = self.kappa_imu

    def _predict_vi(self, t):
        """IMU propagation of the newest full state to t (imu_factor.propagate, causal), with its covariance."""
        from xclean.vio import imu_factor as F
        fulls = self._fulls()
        if not fulls:
            return None
        n = fulls[-1]
        st = F.State(n["R"], n["p"], n["v"], n["bg"], n["ba"])
        P = self.Sig_full if self.Sig_full is not None else None
        try:
            return F.propagate(st, self.imu.t, self.imu.g, self.imu.a, n["t"], float(t), self._imu_horizon(), self.g_w,
                               noise=self.noise, P=P, kappa=self.kappa_imu)
        except ValueError:
            return None

    def _attach_imu(self, s, pred):
        """The new frame becomes a full state, joined to the previous newest full state by the propagated record."""
        prev = self._fulls()[-1]
        s["full"] = True
        s["v"] = pred.state.v.copy()
        s["bg"] = pred.state.bg.copy()
        s["ba"] = pred.state.ba.copy()
        self.imu_factors.append({"i": prev["sid"], "j": s["sid"], "pre": pred.pre, "W": self._W(pred.pre),
                                 "kappa": self.kappa_imu})

    def _coast_vi(self, fr, pred, left, right, kp, desc, n_in_view, d2=True):
        """No visual pose: the frame still enters the window as a full state carried by its IMU factor (DESIGN_P1 §7:
        the window solves every frame, the gate widens with the propagated covariance).  Published as coasting
        (degraded).  P1.1 D2: no segment reset while the IMU is admitted, except when the propagated position sigma
        exceeds the map-matching search radius (_exceeds_search, class B); when the map is out of view, or the coast
        has outlived the window (SCALE_WINDOW frames, P1-07f's class-B rule), the IMU-carried frame becomes a keyframe
        with fresh stereo landmarks INSIDE the same window (re-anchor), linked to it by its IMU factor."""
        s = self._new_slot("fr", pred.state.R, pred.state.p, fr.index, fr.t)
        self._attach_imu(s, pred)
        self.slots.append(s)
        self._reorder()
        Sig, _res = self._solve()
        T_sol = _se3(s["R"], s["p"])
        self.T_last = T_sol
        self.t_last = float(fr.t)
        self.Sig_last = Sig
        self.coast_streak += 1
        self.stats["coasted"] += 1
        self.stats["imu_coast"] += 1
        self.stats["vi_coast_max"] = max(self.stats["vi_coast_max"], self.coast_streak)
        self.state = "coasting"
        self.T_wc = T_sol @ _se3(self.Rbc, self.pbc)
        self._frame_inliers = 0
        # P1.2 W1 on the IMU-only frame (its newest factor was solved): a demotion here takes D3's no-pose path
        if self._w1_update(fr):
            self._demote_nopose("w1")
            return self.state
        if not d2:
            pass
        elif self._exceeds_search(Sig):
            self.stats["vi_resets"] += 1
            self._carry = (s["v"].copy(), s["bg"].copy(), s["ba"].copy())
            if self._p13 is not None:  # P1.3 H4b: the VI window's newest-state covariance links the segments
                self._p13_link = ("coast_vi", Sig)
            if self._try_init(fr, left, right, kp, desc, T_sol):
                self.T_last = T_sol
                self.t_last = float(fr.t)
                self.state = "coasting"
                return self.state
        elif n_in_view < MIN_PNP_INLIERS or self.coast_streak > SCALE_WINDOW:
            self._reanchor(s, left, right, kp, desc)
        with self._stage("marg"):
            self._upkeep()
        return self.state

    def _exceeds_search(self, Sig):
        """D2 (b): the newest position sigma (relative to the window's map, _solve's conditional covariance) against the
        map-matching search radius: the metric offset at the map's median depth z whose chi2_2(0.999) gate spans half
        the image, r = min(W / fx, H / fy) / 2 * z / sqrt(chi2_2(0.999)) (class B: image size and focal A, depth C)."""
        if Sig is None:
            return False
        alive = np.nonzero(self.lm_alive[:self.n_lm])[0]
        rho = self.lm_p[alive, 2] if len(alive) else np.zeros(0)
        rho = rho[rho > 0]
        if len(rho) == 0:
            return False
        z = float(np.median(1.0 / rho))
        r = 0.5 * min(self.W / self.fx, self.H / self.fy) * z / math.sqrt(CHI2_2_999)
        try:
            sp = math.sqrt(max(float(np.linalg.eigvalsh(0.5 * (Sig[3:6, 3:6] + Sig[3:6, 3:6].T)).max()), 0.0))
        except np.linalg.LinAlgError:
            return False
        return sp > r

    def _reanchor(self, s, left, right, kp, desc):
        """D2: the IMU-carried frame s becomes a keyframe hosting fresh stereo landmarks (no segment reset)."""
        if len(kp) == 0:
            return False
        d, sig, ok, amb, span, _ = self._stereo_ex(left, right, kp.astype(float))
        if int(ok.sum()) < KF_COUNT:
            return False
        s["kind"] = "kf"
        self._reorder()
        self.stats["kfs"] += 1
        self.stats["reanchor"] += 1
        self._create_landmarks(s, kp[ok].astype(float), d[ok], sig[ok], desc[ok], span[ok])
        self.n_last_kf_obs = 0
        self.coast_streak = 0
        self._kf_hist.append((s["sid"], s["t"], _se3(s["R"], s["p"]), self._imu_horizon()))
        self._kf_hist = self._kf_hist[-MAX_INIT_KF:]
        if self._p13 is not None:      # P1.3 H2 (re-anchor keyframe): its packet
            self._p13.kf_record(self, s, left, right, "reanchor")
        return True

    # ------------------------------------------------------------------------------------------ P1.1 D1 / D3
    def _try_d1(self, s):
        """D1: stationarity from the IMU alone over the last SCALE_WINDOW frame
        intervals (one-way ANOVA per axis, imu_init_watch.stationarity); if stationary, the IMU is admitted
        PROVISIONALLY at this frame with the IMU's own state (imu_init_watch.imu_first_state).  kappa_imu starts at
        the window's MEASURED frame-interval noise (the ANOVA mean square between sub-windows, the white-equivalent
        variance that reaches a preintegrated factor) over the declared density^2 x rate, median over the 6 axes,
        floored at 1 (class C); the marginalised factors take over after SCALE_WINDOW of them (P2-fix-2)."""
        from xclean.vio import imu_init_watch as IW
        if len(self._frame_t) < SCALE_WINDOW + 1 or not hasattr(self.imu, "t"):
            return False
        st = IW.stationarity(self.imu.t, self.imu.g, self.imu.a, self._imu_horizon(), self._frame_t)
        self.stats["d1_tests"] += 1
        if not st.get("ok"):
            return False
        ini = IW.imu_first_state(s["R"], st)
        f = float(st["rate"])
        dens = np.concatenate([np.full(3, float(self.noise.sigma_g) ** 2), np.full(3, float(self.noise.sigma_a) ** 2)])
        k0 = float(np.median(np.concatenate([st["s2b_g"], st["s2b_a"]]) / (dens * f)))
        self.kappa_imu = max(1.0, k0)
        self.g_w = np.asarray(ini["g_w"], float)
        s["full"] = True
        s["v"] = ini["v"].copy()
        s["bg"] = ini["bg"].copy()
        s["ba"] = ini["ba"].copy()
        self.prior = add_block_info(self.prior, s["sid"], 15, (s["R"], s["p"], s["v"], s["bg"], s["ba"]), ini["Hb"],
                                    np.zeros(15))
        self.vi = True
        self.vi_path = "d1"
        self._d1_used = True
        self.imu_factors = []
        self._kimu_buf = []
        self.Sig_full = None
        self._wd_cons = None
        self._vi_b0 = (s["bg"].copy(), s["ba"].copy(), np.diag(ini["sd_bg"] ** 2), np.diag(ini["sd_ba"] ** 2),
                       float(s["t"]))
        self.stats["events"].append(["admit_d1", int(s["fidx"]), float(s["t"]), float(self.kappa_imu)])
        self.stats["d1_admit_frame"] = int(s["fidx"])
        self.stats["d1_F_max"] = float(np.max(st["F"]))
        self.stats["d1_F_thr"] = float(st["thr"])
        self.stats["d1_kappa0"] = self.kappa_imu
        self.stats["vi_admit_frame"] = int(s["fidx"])
        self.stats["vi_admit_kf"] = int(self.stats["kfs"])
        self.stats["vi_admit_segment"] = int(self.segment)
        self.stats["vi_admit_kappa"] = self.kappa_imu
        return True

    def _watchdog(self, s, left, right, kp, desc, act=True):
        """D3 at each keyframe while the IMU is admitted: vi_init.watchdog_stat on the window's whitened IMU chi2 (the
        class-C kappa_imu applied; imu_hi at P_GATE); failing on N_CONSEC consecutive KF intervals demotes the IMU.
        (The bias-RATE clause was DROPPED and REMOVED under the D3 rule: in arm A it demoted mh_04 and v1_02,
        rows where VI beats VO)  act=False: statistics only - always so since P1.2 (W1 acts).
        XCLEAN_P1_WD_LOG=<file>: one JSON line per evaluation (diagnostic for the D3-replacement track)."""
        from xclean.vio import vi_init as V
        self.stats["wd_kfs"] += 1
        summ = {"imu_chi2": self._last_imu_chi2, "imu_dof": 15.0} if self._last_imu_chi2 else {}
        wd = V.watchdog_stat(summ, state=self._wd_cons)
        cons = dict(wd.consec)
        trig = list(wd.triggers)
        active = list(wd.active)
        self._wd_cons = cons
        for k in active:
            self.stats["wd_active"][k] = self.stats["wd_active"].get(k, 0) + 1
        if self.wd_log:
            import json
            line = json.dumps({"frame": int(s["fidx"]), "t": float(s["t"]), "kappa_imu": float(self.kappa_imu),
                               "imu_chi2": [float(x) for x in self._last_imu_chi2],
                               "stats": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                         for k, v in wd.stats.items()},
                               "active": active, "consec": cons, "trig": trig, "act": bool(act),
                               "vi_path": self.vi_path}) + "\n"
            if self.log_writer is not None:
                self.log_writer.write(self.wd_log, line)
            else:
                with open(self.wd_log, "a", encoding="utf-8") as f:
                    f.write(line)
        if trig and act:
            self._demote(trig, s, left, right, kp, desc)

    def _demote_nopose(self, why):
        """Demotion on a frame without a visual pose: the IMU leaves; the frame coasts in VO (no reset here)."""
        self.stats["events"].append(["demote", int(self._fidx), None, [why]])
        self.stats["demote_frame"] = int(self._fidx)
        self.stats["demote_reason"] = [why]
        self.stats["demote_kappa_imu"] = self.kappa_imu
        self.vi = False
        self._demoted = True
        self.imu_factors = []
        self.Sig_full = None
        # IMU-only frames have no information without their IMU factors: the window restarts (the next frame
        # initialises a VO segment at the published pose, so LIVE-ODOM does not jump)
        self.slots = []
        self.prior = None
        self.lm_alive[:self.n_lm] = False

    def _demote(self, trig, s, left, right, kp, desc):
        """A demotion (P1.2: W1, trig ["w1"]; formerly D3): the IMU leaves (VO), a new segment starts at this frame's
        pose (LIVE-ODOM does not jump)."""
        from types import SimpleNamespace
        self.stats["events"].append(["demote", int(s["fidx"]), float(s["t"]), list(trig)])
        self.stats["demote_frame"] = int(s["fidx"])
        self.stats["demote_reason"] = list(trig)
        self.stats["demote_kappa_imu"] = self.kappa_imu
        T = _se3(s["R"], s["p"])
        self.vi = False
        self._demoted = True
        self.imu_factors = []
        self.Sig_full = None
        self.T_last = T
        if self._p13 is not None:      # P1.3 H4c: the old window's last KF -> the demoted slot is an ordinary edge
            self._p13_link = ("demote", int(s["sid"]))
        if not self._try_init(SimpleNamespace(index=s["fidx"], t=s["t"]), left, right, kp, desc, T):
            self.slots = []
            self.prior = None
            self.lm_alive[:self.n_lm] = False

    def _try_vi_admission(self, s):
        """At each keyframe until admitted: vi_init.init_candidate on the segment's keyframes (causal; each record's
        own released horizon); admission is a pluggable call."""
        from xclean.vio import vi_init as V
        if len(self._kf_hist) < V.MIN_INIT_KF + 1:
            return
        hist = self._kf_hist[-MAX_INIT_KF:]
        T_wb = [h[2] for h in hist]
        kf_t = np.array([h[1] for h in hist])
        horizons = np.array([h[3] for h in hist[1:]])
        bg0 = np.asarray(getattr(self.imu, "bias", np.zeros(3)), float).reshape(3)
        try:
            cand = V.init_candidate(T_wb, kf_t, self.imu.t, self.imu.g, self.imu.a, horizons, self.noise, bg0,
                                    prev=self._vi_prev)
        except Exception as e:                            # a failed fit defers; it never admits
            self.stats["vi_error"] = repr(e)[:200]
            return
        self._vi_prev = cand.init
        self.stats["vi_init_ms"].append(float((getattr(cand, "timing", None) or {}).get("total", float("nan"))))
        dec = cand.admission.decision
        if ADMIT_MODE == "force_diag":
            # ORACLE DIAGNOSTIC: admit at the first KF whose stereo_vi_init fit is non-degenerate, whatever the gate
            # says (IGT refusal, |g| test, agreement, VCE convergence are all overridden); never a shipped config
            ini_ = cand.init
            # "non-degenerate": ok, full rank, not degenerate (incl. the
            # bias-infeasible cause 'bias'), variance components converged, |g| constraint not rejected
            bad_fit = (ini_ is None or not getattr(ini_, "ok", False) or getattr(ini_, "rank_deficient", False)
                       or getattr(ini_, "degenerate", False) or not getattr(ini_, "vce_converged", True)
                       or not getattr(ini_, "mag_ok", True))
            if bad_fit:
                self.stats["vi_defer"] += 1
                key = "force_diag: no non-degenerate fit (%s)" % (
                    "no fit" if ini_ is None or not getattr(ini_, "ok", False) else
                    "rank" if getattr(ini_, "rank_deficient", False) else
                    ("degenerate: " + str(getattr(ini_, "degenerate_cause", ""))) if getattr(ini_, "degenerate", False)
                    else "vce not converged" if not getattr(ini_, "vce_converged", True) else "|g| rejected")
                self.stats["vi_defer_reasons"][key] = self.stats["vi_defer_reasons"].get(key, 0) + 1
                return
            self.stats["vi_force_diag_gate_said"] = [dec] + [str(r) for r in cand.admission.reasons][:4]
            dec = "admit"
        if dec == "refuse":
            self.vi_refused = True
            self.stats["vi_refused"] += 1
            self.stats["vi_refuse_reasons"] = list(cand.admission.reasons)[:3]
            return
        if dec != "admit":
            self.stats["vi_defer"] += 1
            import re
            for r in cand.admission.reasons:
                key = re.sub(r"[-+]?\d+(\.\d+)?(e[-+]?\d+)?", "#", str(r))[:90]
                self.stats["vi_defer_reasons"][key] = self.stats["vi_defer_reasons"].get(key, 0) + 1
            return
        ini = cand.init
        vw = np.asarray(ini.v_w, float).reshape(-1, 3)
        # P1.2-NANGUARD: vi_init can ADMIT a candidate whose newest keyframe no usable record touches
        # (data snooping removed its interval: v_w[-1] = NaN, cov_v[-1] = inf; madmax_E1 frame 5287 crash).  The engine
        # admits the NEWEST keyframe's state, so a non-finite admitted state is a DEFERRAL, never an admission (the
        # next keyframe adds a record and re-tests); nothing of the candidate enters the engine.
        g_c = np.asarray(ini.g_w, float).reshape(3)
        bg_c = np.asarray(ini.bg, float).reshape(3)
        ba_c = np.asarray(ini.ba, float).reshape(3)
        if not (np.all(np.isfinite(vw[-1])) and np.all(np.isfinite(g_c)) and np.all(np.isfinite(bg_c))
                and np.all(np.isfinite(ba_c))):
            self.stats["vi_defer"] += 1
            key = "non-finite admitted state"
            self.stats["vi_defer_reasons"][key] = self.stats["vi_defer_reasons"].get(key, 0) + 1
            self.stats["events"].append(["admit_nonfinite_deferred", int(s["fidx"]), float(s["t"])])
            return
        self.g_w = g_c
        s["full"] = True
        s["v"] = vw[-1].copy()
        s["bg"] = bg_c.copy()
        s["ba"] = ba_c.copy()
        kk = getattr(ini, "kappa", None)
        kv = getattr(kk, "kappa", kk)
        if kv is not None and np.isfinite(float(kv)):
            self.kappa_imu = max(1.0, float(kv))
        Hb = np.zeros((15, 15))
        # P1.2-COVV: ViInit.cov_v is (n_kf, 3, 3), one block per KF; the admitted state is the NEWEST KF, so the read is
        # cov_v[-1] (the former 2-D read fell through to 1e6 I).  P1.2-VPRIOR-DROPPED: it is NOT applied as a
        # velocity prior (applying it harmed both rows it acts on: 4s_173 +19.4 %, madmax_E1 +17.3 %); the velocity
        # block of Hb stays 0.  The read is kept as a diagnostic.  The bias priors are unchanged; a block that is not
        # finite (a KF no usable record touches: inf diagonal) carries no information.
        cv = np.asarray(ini.cov_v, float)
        cv = cv[-1] if cv.ndim == 3 else cv
        self.stats["vi_admit_cov_v_diag"] = [float(x) for x in np.diag(np.asarray(cv, float).reshape(3, 3))]
        for sl, C in ((slice(9, 12), np.asarray(ini.cov_bg, float).reshape(3, 3)),
                      (slice(12, 15), np.asarray(ini.cov_ba, float).reshape(3, 3))):
            C = np.asarray(C, float).reshape(3, 3)
            if np.all(np.isfinite(C)):
                Hb[sl, sl] = np.linalg.inv(C + 1e-15 * np.eye(3))
        self.prior = add_block_info(self.prior, s["sid"], 15, (s["R"], s["p"], s["v"], s["bg"], s["ba"]), Hb,
                                    np.zeros(15))
        self.vi = True
        self.vi_path = "dyn"
        self.stats["events"].append(["admit_dyn", int(s["fidx"]), float(s["t"]), float(self.kappa_imu)])
        self._wd_cons = None
        self.imu_factors = []
        self._vi_b0 = (s["bg"].copy(), s["ba"].copy(), np.asarray(ini.cov_bg, float).reshape(3, 3),
                       np.asarray(ini.cov_ba, float).reshape(3, 3), float(s["t"]))
        self.stats["vi_admit_frame"] = int(s["fidx"])
        self.stats["vi_admit_kf"] = int(self.stats["kfs"])
        self.stats["vi_admit_segment"] = int(self.segment)
        self.stats["vi_admit_kappa"] = self.kappa_imu

    # ------------------------------------------------------------------------------------------ P1.2 W1
    def _w1_blocks(self):
        """W1 design rule: dynamic re-admission is blocked after a W1 demotion, for the rest of the run.  With
        XCLEAN_P12_OFF w1 W1 never blocks."""
        return "w1" not in P12_OFF and self.w1.blocked

    def _w1_update(self, fr):
        """P1.2 W1, once per frame.  c_new = the newest window IMU factor's whitened chi2 (kappa_imu
        applied, as the solve weighs it) when the IMU is admitted AND a window solve ran THIS frame, else None; stage
        A's (vis_ok, q_A, n_A, s_A) from this frame's _track via imu_watch.stage_a_obs (as the replay).  The camera
        frame-drop flag of the interval (imu_watch's own rule, read before its update) is logged (diagnostic).
        -> True when W1 demotes AND acts (XCLEAN_P12_OFF w1: W1 computes and logs, never acts)."""
        from xclean.vio import imu_watch as IWATCH
        self._w1_done = True
        c = None
        if self.vi and self.n_solves > self._n_solves0 and self._last_imu_chi2:
            c = float(self._last_imu_chi2[-1])
        nm, ni, ok, sA = self._stage_a
        vis_ok, q, n, s_a = IWATCH.stage_a_obs([("p1pnpA", nm, ni, ok)], sA)
        w = self.w1
        t = float(fr.t)
        per = w.period if w.period else (float(np.median(w._dts)) if len(w._dts) else None)
        drop = bool(w.prev_t is not None and per is not None and t - w.prev_t > 1.5 * per)
        d = w.update(int(fr.index), bool(self.vi), c, vis_ok, q, n, s_a, t=t)
        d["drop"] = drop
        self._w1_last = d
        act = bool(d["demote"]) and "w1" not in P12_OFF
        if self.frame_log:
            self._fl["w1"] = {"in": [bool(self.vi), c, bool(vis_ok), q, n, s_a, t], "fail": bool(d["fail"]),
                              "episode": d["episode"], "attribution": d["attribution"],
                              "n_imu_sustained": int(d["n_imu_sustained"]), "blocked": bool(d["blocked"]),
                              "demote": bool(d["demote"]), "drop": drop, "act": act}
        if act:
            self.stats["w1_demotes"] += 1
        return act

    def _pnp(self, fr, lids, kpi, kp, T_init_wb, tag, scale_px=None):
        import cv2
        from xclean.track.pose import solve_pose
        if len(lids) < MIN_PNP_INLIERS:
            return None, lids[:0], kpi[:0]
        Pw = self._world_points(lids)
        cv2.setRNGSeed((zlib.crc32(tag) + int(fr.index)) % (2 ** 31 - 1))
        # P1-07e: with a measured PREDICTIVE residual scale of this frame (stage A's inliers around their own pose,
        # Class C) the inlier radius is its chi2_2(0.999) radius; without one, the carried 3 x FEATURE_SIGMA_PX.
        # (The window's post-fit sigma_px is not predictive: it under-states a new frame's residuals.)
        Twc, mask, _dec = solve_pose(Pw, kp[kpi].astype(np.float64), self.rig,
                                     T_init=T_init_wb @ _se3(self.Rbc, self.pbc),
                                     reproj_px=None if scale_px is None else math.sqrt(CHI2_2_999) * scale_px)
        if Twc is None:
            return None, lids[:0], kpi[:0]
        return Twc @ np.linalg.inv(_se3(self.Rbc, self.pbc)), lids[mask], kpi[mask]

    def _project(self, T_wb, Spose, alive, sig_px):
        Rwc_h, pwc_h = self._host_cams(alive)
        L = len(alive)
        uv = np.full((L, 2), np.nan)
        S = np.zeros((L, 2, 2))
        okp = np.zeros(L, np.bool_)
        depth = np.zeros(L)
        Wn.predict_projection(np.ascontiguousarray(T_wb[:3, :3]), np.ascontiguousarray(T_wb[:3, 3]), self.Rbc,
                              self.pbc, np.ascontiguousarray(Rwc_h), np.ascontiguousarray(pwc_h),
                              np.ascontiguousarray(self.lm_p[alive]), np.ascontiguousarray(Spose),
                              np.ascontiguousarray(self.lm_Sl[alive]), float(sig_px), self.fx, self.fy, self.cx,
                              self.cy, float(self.W), float(self.H), uv, S, okp, depth)
        return uv, S, okp

    def _track(self, fr, kp, desc, T_pred, Sig_pred, alive):
        """-> (T_wb or None, landmark ids, keypoint indices, landmarks predicted in view, mode)."""
        if len(alive) == 0 or len(kp) == 0:
            return None, np.zeros(0, np.int64), np.zeros(0, np.int64), 0, "empty"
        _uv, _S, okp = self._project(T_pred, np.zeros((6, 6)), alive, self.sig_px)
        n_view = int(okp.sum())
        # (A) prediction-free hypothesis
        last = np.ascontiguousarray(self.lm_desc[alive, np.maximum(self.lm_ndesc[alive] - 1, 0)])
        i0, i1 = self._match_desc(last, desc)
        self.stats["matched_bf"] += int(len(i0))
        T_A, lA, kA = self._pnp(fr, alive[i0], i1.astype(np.int64), kp, T_pred, b"p1pnpA")
        # P1.2 W1: stage A = the only visual estimate the IMU cannot steer
        self._stage_a = (int(len(i0)), int(len(lA)), T_A is not None, None)
        mode = "A"
        if T_A is None:
            # (A) failed: the constant-velocity prediction carries the gate
            self.stats["bf_fallback"] += 1
            lids, kpi, _nv, gated = self._associate(T_pred, Sig_pred, alive, kp, desc)
            T_new, lids, kpi = self._pnp(fr, lids, kpi, kp, T_pred, b"p1pnpG")
            return T_new, lids, kpi, n_view, ("G" if gated else "none")
        # (B) project every window landmark with the hypothesis; gate = the hypothesis' own residual scale (Class C:
        # the robust rms of (A)'s inliers around T_A) on top of the landmark marginals
        Pw = self._world_points(lA)
        Rwc = T_A[:3, :3] @ self.Rbc
        pwc = T_A[:3, 3] + T_A[:3, :3] @ self.pbc
        Xc = (Pw - pwc) @ Rwc
        pr = np.stack([self.fx * Xc[:, 0] / Xc[:, 2] + self.cx, self.fy * Xc[:, 1] / Xc[:, 2] + self.cy], axis=1)
        res = np.linalg.norm(pr - kp[kA], axis=1)
        s_A = float(np.median(res)) / RAYLEIGH_MED
        self._stage_a = (int(len(i0)), int(len(lA)), True, s_A)
        uv, S, okp = self._project(T_A, np.zeros((6, 6)), alive, s_A)
        sel = np.nonzero(okp)[0]
        k, dist, d2 = self.mm.match(kp, desc, uv[sel], S[sel], self.lm_desc[alive[sel]], self.lm_ndesc[alive[sel]])
        got = k >= 0
        lB, kB = alive[sel[got]], k[got]
        self.stats["matched_gated"] += int(got.sum())
        # union: (B) wins per landmark and per keypoint; (A) inliers fill the rest
        usedk = np.zeros(len(kp), bool)
        usedk[kB] = True
        usedl = np.zeros(self.n_lm, bool)
        usedl[lB] = True
        keepA = ~usedk[kA] & ~usedl[lA]
        lids = np.concatenate([lB, lA[keepA]])
        kpi = np.concatenate([kB, kA[keepA]])
        T_new, lids, kpi = self._pnp(fr, lids, kpi, kp, T_A, b"p1pnpC", scale_px=s_A)
        if T_new is None:
            return T_A, lA, kA, n_view, mode
        return T_new, lids, kpi, n_view, "AB"

    def _match_desc(self, last, desc):
        """Stage-A prediction-free descriptor matching (Frontend.match_desc; XCLEAN_LAT=m1: the numba one-pass kernel,
        identical matches, fastpaths.py)."""
        from xclean.front import fastpaths as FP
        if FP.on("gf"):
            # the device Hamming the GPU Detector installed as fe.match_desc (desc = the array det() returned, so the
            # descriptors already on the device are reused); checked FIRST, or m1 would bypass it
            return self.det.fe.match_desc({"descriptors": last}, {"descriptors": desc})
        if FP.on("m1"):
            return FP.match_desc_fast(last, desc)
        return self.det.fe.match_desc({"descriptors": last}, {"descriptors": desc})

    def _associate(self, T_pred, Sig_pred, alive, kp, desc):
        """-> (landmark ids, keypoint indices, landmarks predicted in view, gated?)"""
        if len(alive) == 0 or len(kp) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), 0, False
        Rwc_h, pwc_h = self._host_cams(alive)
        L = len(alive)
        uv = np.full((L, 2), np.nan)
        S = np.zeros((L, 2, 2))
        okp = np.zeros(L, np.bool_)
        depth = np.zeros(L)
        Sp = Sig_pred if Sig_pred is not None else np.zeros((6, 6))
        Wn.predict_projection(np.ascontiguousarray(T_pred[:3, :3]), np.ascontiguousarray(T_pred[:3, 3]), self.Rbc,
                              self.pbc, np.ascontiguousarray(Rwc_h), np.ascontiguousarray(pwc_h),
                              np.ascontiguousarray(self.lm_p[alive]), np.ascontiguousarray(Sp),
                              np.ascontiguousarray(self.lm_Sl[alive]), self.sig_px, self.fx, self.fy, self.cx,
                              self.cy, float(self.W), float(self.H), uv, S, okp, depth)
        n_view = int(okp.sum())
        if Sig_pred is not None and not os.environ.get("XCLEAN_P1_DEBUG_NOGATE"):
            sel = np.nonzero(okp)[0]
            k, dist, d2 = self.mm.match(kp, desc, uv[sel], S[sel], self.lm_desc[alive[sel]],
                                        self.lm_ndesc[alive[sel]])
            got = k >= 0
            if got.sum() < KF_COUNT and got.sum() > 0:
                # one re-search with the covariance inflated by the observed / predicted innovation ratio (Class C)
                infl = max(float(np.median(d2[got])) / RAYLEIGH_MED ** 2, 1.0)
                if infl > 1.0:
                    self.stats["re_search"] += 1
                    k2, dist2, d22 = self.mm.match(kp, desc, uv[sel], S[sel] * infl, self.lm_desc[alive[sel]],
                                                   self.lm_ndesc[alive[sel]])
                    if (k2 >= 0).sum() > got.sum():
                        k, got = k2, k2 >= 0
            if got.sum() >= MIN_PNP_INLIERS:
                self.stats["matched_gated"] += int(got.sum())
                return alive[sel[got]], k[got], n_view, True
        # prediction-free fallback: descriptor matching against every window landmark (mutual + ratio + chance)
        self.stats["bf_fallback"] += 1
        last = np.ascontiguousarray(self.lm_desc[alive, np.maximum(self.lm_ndesc[alive] - 1, 0)])
        i0, i1 = self._match_desc(last, desc)
        self.stats["matched_bf"] += int(len(i0))
        return alive[i0], i1.astype(np.int64), n_view, False

    def _make_kf(self, s, left, right, kp, desc, used):
        self._fl["kf"] = True
        s["kind"] = "kf"
        self._reorder()
        self.stats["kfs"] += 1
        # descriptors of the matched landmarks from this KF (keep the last MAX_DESC)
        m = (s["cam"] == 0) & s["valid"]
        lids = s["lid"][m]
        uvs = s["uv"][m]
        if len(lids) and len(kp):
            # the keypoint each observation came from: exact coordinate match
            from scipy.spatial import cKDTree
            dd, ii = cKDTree(kp.astype(np.float64)).query(uvs, k=1)
            for l, j, dist in zip(lids, ii, dd):
                if dist > 1e-6:
                    continue
                n = self.lm_ndesc[l]
                if n < MAX_DESC:
                    self.lm_desc[l, n] = desc[j]
                    self.lm_ndesc[l] = n + 1
                else:
                    self.lm_desc[l, :-1] = self.lm_desc[l, 1:]
                    self.lm_desc[l, -1] = desc[j]
        # kappa_d (prior form, Class C; P1-05c): at each keyframe the matched landmarks' disparity is RE-MEASURED and
        # compared with the window's prediction d_pred = fB rho / q2 in this camera; kappa_d = the running median of the
        # MAD scale of (d_meas - d_pred) / (kappa_d sigma_LK).  The prior's own post-fit residual cannot estimate it:
        # an over-confident prior never lets the multi-view information exceed it.
        if self.stereo_mode == "prior" and not getattr(self, "freeze_scales", False) and len(lids) >= 2 * KF_COUNT:
            ok_l = np.asarray([dist <= 1e-6 for dist in dd]) if len(lids) else np.zeros(0, bool)
            lk = lids[ok_l]
            kj = ii[ok_l]
            dm, sm, okm, _ = self._stereo(left, right, kp[kj].astype(float))
            if okm.sum() >= 2 * KF_COUNT:
                Rwc_h, pwc_h = self._host_cams(lk[okm])
                Rwc_t = s["R"] @ self.Rbc
                pwc_t = s["p"] + s["R"] @ self.pbc
                ab = self.lm_p[lk[okm]]
                m3 = np.stack([ab[:, 0], ab[:, 1], np.ones(len(ab))], axis=1)
                Y = np.einsum("nij,nj->ni", Rwc_h, m3) + ab[:, 2:3] * (pwc_h - pwc_t)
                q = Y @ Rwc_t
                good = q[:, 2] > 1e-9
                d_pred = self.fB * ab[good, 2] / q[good, 2]
                z = (dm[okm][good] - d_pred) / np.maximum(self.kappa_d * sm[okm][good], 1e-12)
                if good.sum() >= 2 * KF_COUNT:
                    sz = float(MAD_K * np.median(np.abs(z)))
                    if sz > 0 and np.isfinite(sz):
                        self._kd_hist.append(self.kappa_d * sz)
                        self._kd_hist = self._kd_hist[-SCALE_WINDOW:]
                        self.kappa_d = max(1.0, float(np.median(self._kd_hist)))   # sigma_LK is the CRB of the 1-D fit: a floor (P1-05e)
        free = np.ones(len(kp), bool)
        free[used] = False
        idx = np.nonzero(free)[0]
        if len(idx):
            d, sig, ok, amb, span, _ = self._stereo_ex(left, right, kp[idx].astype(float))
            use = ok                     # P1-03d (ambiguous peaks as landmarks with a span sigma) was DROPPED
            sel = idx[use]
            self._create_landmarks(s, kp[sel].astype(float), d[use], sig[use], desc[sel], span[use])
        self.n_last_kf_obs = 0                      # re-set by the first frame after this keyframe
        self._kf_hist.append((s["sid"], s["t"], _se3(s["R"], s["p"]), self._imu_horizon()))
        self._kf_hist = self._kf_hist[-MAX_INIT_KF:]
        if self.noise is not None and not self.vi and not self.vi_refused:
            if not self._w1_blocks():        # P1.2 W1: no re-admission after a W1 demotion
                self._try_vi_admission(s)
        elif self.vi and self._fulls():
            if self._vi_b0 is not None:
                self._bias_check(s)          # the M2 declared-density bound, RECORDED only (diagnostic)
            # P1.2: D3's ACTING path is off - W1 replaces it; D3's statistics are still computed (stats
            # wd_active, and XCLEAN_P1_WD_LOG); XCLEAN_P11_OFF d3 now only stops that logging
            if "d3" not in P11_OFF or self.wd_log:
                self._watchdog(s, left, right, kp, desc, act=False)
        if self._p13 is not None:      # P1.3 H2: the keyframe packet (header, creation-time edge, the rectified pair)
            self._p13.kf_record(self, s, left, right, "make_kf")

    def _bias_check(self, s):
        """KP3 diagnostic: the newest full state's bias drift since admission against vi_init.bias_bound (P_GATE)."""
        from xclean.vio import vi_init as V
        bg0, ba0, Cg, Ca, t0 = self._vi_b0
        n = self._fulls()[-1]
        bb = V.bias_bound(self.noise, float(n["t"]) - t0, Cg, Ca, self.kappa_imu)
        cg, eg = V.bias_excursion(n["bg"], bg0, bb.P_bg)
        ca, ea = V.bias_excursion(n["ba"], ba0, bb.P_ba)
        for key, c in (("bias_chi2_max_bg", cg), ("bias_chi2_max_ba", ca)):
            if self.stats[key] is None or c > self.stats[key]:
                self.stats[key] = float(c)
        if (eg or ea) and self.stats["bias_exceed_frame"] is None:
            self.stats["bias_exceed_frame"] = int(s["fidx"])

    def _coast(self, fr, left, right, kp, desc, T_pred, Sig_pred, n_in_view):
        self.coast_streak += 1
        self.stats["coasted"] += 1
        # map out of view (fewer predicted-visible landmarks than the conditioning floor), OR the coast has outlived
        # the window (SCALE_WINDOW = N_KF + N_FR frames: every window frame then predates it, the map is stale by
        # construction - class B; P1-07f, v2_03 measured a 1668-frame dead-lock with its map "in view"): new segment
        # from here, starting at the propagated pose (LIVE-ODOM continues without a jump)
        if n_in_view < MIN_PNP_INLIERS or self.coast_streak > SCALE_WINDOW:
            if self._p13 is not None:  # P1.3 H4a: the constant-velocity prediction's covariance (or None)
                self._p13_link = ("coast", Sig_pred)
            if self._try_init(fr, left, right, kp, desc, T_pred):
                self.T_last = T_pred
                self.t_last = float(fr.t)
                self.T_wc = T_pred @ _se3(self.Rbc, self.pbc)
                self.state = "coasting"
                return self.state
        self.T_last = T_pred
        self.t_last = float(fr.t)
        if Sig_pred is not None:
            self.Sig_last = Sig_pred
        self.T_wc = T_pred @ _se3(self.Rbc, self.pbc)
        self.state = "coasting"
        self._frame_inliers = 0
        return self.state

    def summary(self):
        st = dict(self.stats)
        for k in ("window_P", "window_L", "window_L_posefree", "window_M", "cost_ratio", "vi_init_ms"):
            v = np.asarray(st.pop(k), float)
            st[k] = ({"p50": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max()),
                      "n": int(len(v))} if len(v) else None)
        st.update(vi=self.vi, vi_refused=self.vi_refused, kappa_imu=self.kappa_imu,
                  g_w=None if self.g_w is None else [float(x) for x in self.g_w],
                  bg=None if not self.slots else [float(x) for x in self._newest()["bg"]],
                  ba=None if not self.slots else [float(x) for x in self._newest()["ba"]])
        from xclean.front import fastpaths as FP
        st.update(lat=sorted(FP.LAT))
        st.update(vi_path=self.vi_path, demoted=self._demoted, dy_hat=self.dy_hat, dy_sigma=self.dy_sigma,
                  dy_band=self.dy_band, p11_off=sorted(P11_OFF))
        st.update(stereo_impl=STEREO_IMPL, kappa_d_hist=[float(x) for x in self._kd_hist], admit_mode=ADMIT_MODE,
                  not_for_bars=(ADMIT_MODE != "gate" or bool(P11_OFF) or bool(P12_OFF)))
        # P1.2: the effective switch set, W1's own summary, whether W3 item 1 ran (window_L_posefree above)
        st.update(p12_off=sorted(P12_OFF), w1=self.w1.summary(), w3_posefree=bool(self.w3))
        st.update(sig_px=self.sig_px, kappa_d=self.kappa_d, stereo_mode=self.stereo_mode, segment=self.segment,
                  detector=dict(self.det.stats, budget=self.det.budget), mapmatch=self.mm.stats)
        if self.front_fallback is not None:          # start-up self-check policy: the CPU front replaced the GPU one
            st.update(front_fallback=self.front_fallback)
        return st


# ================================================================================================ live harness entry
def run_live(cfg, slot_name, imu_q, msg_q, ctl, out_dir):
    """The P1 engine behind the live harness's FEEDER / fence / RECORDER (selected by XCLEAN_ENGINE=p1).  Mirrors
    xclean/live/engine.py engine_main; one CUDA context only for the GPU front (XCLEAN_LAT gf), otherwise none (checked
    before the feeder's clock starts).  A GPU front that fails its start-up self-check is replaced by the CPU front
    (P1Engine._front_fallback, _gf_guarded) unless XCLEAN_STRICT_SELFCHECK=1."""
    import json
    import queue
    import sys
    import time
    import cv2
    from xclean.live.engine import LiveSource, make_live_imu, FenceError, drain_imu, imu_wait_bound, _pct
    from xclean.live.protocol import FrameSlot, make_msg
    from xclean.obs.latency import Ledger
    from xclean.io.dataset import FrameRef
    np.random.seed(int(cfg.get("seed", 0)))
    cv2.setRNGSeed(int(cfg.get("seed", 0)))
    if os.environ.get("XCLEAN_CV_THREADS"):
        cv2.setNumThreads(int(os.environ["XCLEAN_CV_THREADS"]))
    tag = cfg["tag"]
    src = LiveSource(cfg["calib"], tag)
    led = Ledger(run=tag, tier=1, sync=None, gpu_stages=())
    imu = make_live_imu(src, verbose=False) if cfg["calib"].get("imu_rate_hz") or cfg.get("has_imu") else None
    src.imu_stream = imu
    eng = P1Engine(src, led)
    # P1.3 ENGINE HOOK: attach the loop-closure link when the loop is on and the parent started a worker
    # (ctl["p13"] = the transport handles); otherwise LIVE-MAP = LIVE-ODOM exactly (the P1.2 behaviour)
    from xclean.live import p13_link as P13L
    from xclean.live.logwriter import LineWriter
    # the run logs (p1_frames.jsonl, p13_packets.jsonl, the watchdog log) are written by one background thread, off the
    # frame path: a slow output disk then cannot stall the frame loop (same bytes, same order as before)
    logw = LineWriter(name="xclean-engine-logs")
    eng.log_writer = logw
    p13h = ctl.get("p13") if isinstance(ctl, dict) else None
    if P13L.loop_on() and p13h is not None:
        P13L.init(p13h, cfg.get("mode", "paced"), eng.Rbc, eng.pbc, eng.cam_period,
                  log_path=None if (os.environ.get("XCLEAN_P13_LOG") or "") == "0"
                  else os.path.join(out_dir, "p13_packets.jsonl"), log_writer=logw)
        eng._p13 = P13L
    if (os.environ.get("XCLEAN_P1_FRAME_LOG") or "") != "0" and eng.frame_log is None:
        eng.frame_log = os.path.join(out_dir, "p1_frames.jsonl")
    _gf_guarded(eng, warmup, src.rig)       # start-up self-check policy: a GPU-front failure here -> the CPU front
    from xclean.front import fastpaths as FP
    FP.warm()
    if FP.on("mf") or FP.on("gf"):
        from xclean.front import mapmatch_fast
        mapmatch_fast.warmup()
    gpu_calib = None
    if FP.on("gf"):                         # K-01b-RELAX: ONE CUDA context, ONE stream, the pool sized once, all before
        from xclean.front import gpu_front  # ctl["ready"] (the feeder's clock)
        gpu_calib = _gf_guarded(eng, gpu_front.warmup, src.rig)
    if FP.on("g1"):                         # latency fix G1: no GC pauses on the frame path after warm-up
        import gc
        gc.collect()
        gc.freeze()
    slot = FrameSlot(slot_name, int(cfg["calib"]["camera_width"]), int(cfg["calib"]["camera_height"]))
    # K-01b (DESIGN_P1 section 2), relaxed: a CUDA context is allowed in the frame
    # process only through gf (one context, one stream, pool sized once at warm-up, created before ready); without gf
    # the frame process stays CPU-only
    # (start-up self-check policy) cupy may be imported here without a context: only to look for a GPU (default fast
    # paths, no usable GPU) or for a GPU front that failed its self-check (P1Engine._front_fallback).  Without a
    # fallback no CUDA context may be held (the driver is asked directly); after a fallback the context the failed
    # front created is kept on purpose (gpu_front.release_device) and recorded as cuda_context_at_start.
    if not FP.on("gf") and "cupy" in sys.modules:
        from xclean.front.gpu_front import cuda_context_active
        held = cuda_context_active()
        if eng.front_fallback is not None:
            eng.front_fallback["cuda_context_at_start"] = held
            if held and eng.front_fallback["cuda_context_released"]:
                print("xclean: warning: a CUDA context exists again after the failed GPU front end released its own; "
                      "the run continues on the CPU front end.", file=sys.stderr, flush=True)
        else:
            assert not held, "P1 frame process must stay CPU-only without XCLEAN_LAT gf"
    p13_ready = None
    if eng._p13 is not None:                # the worker warms up before the feeder's clock starts
        p13_ready = P13L.wait_ready(600.0)
    ctl["ready"].set()
    n_pub = 0
    fence_errors = []
    last_seq = 0
    kf_wall = []
    n_kf_prev = 0
    imu_wait_s = imu_wait_bound(cfg)
    imu_wait_ms = []
    imu_short = []
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
        # IMU intake: one batch per frame, bounded wait (live/engine.drain_imu)
        _n_imu, _w_imu, _short = drain_imu(imu_q, imu, imu_need if fresh else -np.inf, imu_wait_s, fresh)
        if fresh:
            imu_wait_ms.append(1e3 * _w_imu)
            if _short:
                imu_short.append(int(idx))
        if not fresh:
            if eos:
                break
            continue
        src.set_current(idx, t, l, r)
        src._cur_index = idx
        fr = FrameRef(index=idx, t=t, left=None, right=None)
        try:
            if imu is not None:
                imu.causal_bootstrap(t)              # B3: the stationary-window gyro bias, causal (backend/imu.py)
            state = eng.step(fr)
        except FenceError as e:
            fence_errors.append([idx, str(e)])
            state = "fence-error"
        led.frame_done(idx)
        T = eng.T_wc
        horizon = max(src.max_read, getattr(imu, "max_read", -np.inf) if imu is not None else -np.inf)
        degr = state not in ("tracking", "relocalized")
        common = dict(state=state, n_inliers=getattr(eng, "_frame_inliers", None), degraded=degr,
                      gravity_aligned=False, segment=int(eng.segment), horizon=horizon)
        msg_q.put(make_msg(idx, t, "LIVE-ODOM", T, **common))
        if eng._p13 is not None:            # P1.3 H5: LIVE-MAP = T_applied LIVE-ODOM + flags (LIVE-ODOM untouched)
            with led.stage("p13_compose", frame=idx):
                T_map, p13x = P13L.compose(idx, t, T)
            msg_q.put(make_msg(idx, t, "LIVE-MAP", T_map, extra=p13x, **common))
        else:
            msg_q.put(make_msg(idx, t, "LIVE-MAP", T, **common))
        n_pub += 1
        if eng._p13 is not None:            # P1.3 H6: after both publishes (never blocks); T_map = the KF anchor
            with led.stage("p13_emit", frame=idx):
                P13L.emit(eng, t, T, T_map)
        if eng.stats["kfs"] != n_kf_prev:
            n_kf_prev = eng.stats["kfs"]
            kf_wall.append((time.time(), n_kf_prev))
        ctl["done"].value = idx
        ctl["done_evt"].set()
    gpu_info = None
    if FP.on("gf"):
        gpu_info = {"n_ham_fallback": int(getattr(eng.det.gpu, "n_ham_fallback", 0)), "calib": gpu_calib}
    elif eng.front_fallback is not None:    # start-up self-check policy: the run used the CPU front
        gpu_info = dict(eng.front_fallback, fallback_to_cpu_front=True)
    if eng._p13 is not None:                # P1.3 H7: end of stream for the worker (the parent joins it)
        P13L.close()
        p13_sum = dict(P13L.summary(), worker_ready=p13_ready)
    else:
        p13_sum = {"active": False, "loop_on": P13L.loop_on(),
                   "why": cfg.get("p13_inactive_why") or ("XCLEAN_P13_OFF loop" if not P13L.loop_on()
                                                          else "no worker transport from the parent")}
    summary = {"engine": "p1", "frames_processed": n_pub, "fence_errors": fence_errors[:20], "gpu_front": gpu_info,
               "n_fence_errors": len(fence_errors), "kf_wall": kf_wall, "p1": eng.summary(),
               "imu_wait_bound_s": imu_wait_s, "imu_wait_ms": _pct(imu_wait_ms), "imu_short_frames": len(imu_short),
               "imu_short_first": imu_short[:20], "p13": p13_sum, "run_logs": _close_run_logs(logw)}
    from xclean.run import _engine_md5
    summary["engine_md5"] = _engine_md5()
    summary["env_flags"] = {k: os.environ[k] for k in sorted(os.environ) if k.startswith(("XTEN_", "XONE_", "XCLEAN_"))}
    json.dump(summary, open(os.path.join(out_dir, "engine_summary.json"), "w"), indent=1, default=str)
    led.save(os.path.join(out_dir, "latency.json"))
    msg_q.put("EOS")
    slot.close()


def _close_run_logs(logw):
    """End of run: write every queued run-log line, close the files; a problem is printed once and recorded."""
    import sys
    info = logw.close()
    if info["n_errors"] or info["lost_lines"] or not info["finished"]:
        print("xclean: warning: the run logs were not written completely (%d lines lost; %s). The poses and scores do "
              "not depend on them." % (info["lost_lines"], "; ".join(info["errors"][:3]) or "the writer did not "
              "finish"), file=sys.stderr, flush=True)
    return info


def _gf_guarded(eng, fn, *args):
    """Run one start-up warm-up step.  With the GPU front on (XCLEAN_LAT gf) and XCLEAN_STRICT_SELFCHECK unset, a
    failure switches `eng` to the CPU front (P1Engine._front_fallback); the CPU warm-up is then run again on the CPU
    modules, the GPU front's own warm-up is skipped (-> None).  Without gf, or strict: fn(*args) unchanged."""
    from xclean.front import fastpaths as FP
    if not FP.on("gf"):
        return fn(*args)
    from xclean.front.gpu_front import strict_selfcheck
    if strict_selfcheck():
        return fn(*args)
    try:
        return fn(*args)
    except Exception as e:
        eng._front_fallback(e)
        return fn(*args) if fn is warmup else None


def warmup(rig):
    """Compile / load every numba kernel on synthetic data of the rig's size (before the feeder's clock starts)."""
    disparities = _stereo_module().disparities
    from xclean.front.mapmatch import MapMatcher
    W, H = int(rig.width), int(rig.height)
    rng = np.random.default_rng(0)
    img = (rng.random((H, W)) * 255).astype(np.uint8)
    disparities(img, np.roll(img, -8, axis=1), np.array([[W / 2.0, H / 2.0]]))
    # the D4 row band with a vertical shift (dy != 0 resamples the right image to float32: kernels of their own types)
    disparities(img, np.roll(img, -8, axis=1), np.array([[W / 2.0, H / 2.0]]), dy=0.25, band=1)
    Wn.so3_log_nb(np.eye(3))               # the frame log's innovation and marg.py's prior residual
    mm = MapMatcher(W, H)
    mm.match(np.array([[10.0, 10.0], [20.0, 20.0]]), np.zeros((2, 32), np.uint8), np.array([[10.0, 10.0]]),
             np.eye(2)[None] * 4.0, np.zeros((1, 3, 32), np.uint8), np.array([1]))
    # a 2-pose, 1-landmark window
    pb = Wn.Problem(Rwb=np.stack([np.eye(3)] * 2), pwb=np.array([[0.0, 0, 0], [0.1, 0, 0]]), Rbc=np.eye(3),
                    pbc=np.zeros(3), lmp=np.array([[0.0, 0.0, 0.2]]), lm_host=np.array([0], np.int64),
                    lm_mask=np.ones(1, np.bool_), rho_meas=np.array([0.2]), rho_isig=np.array([10.0]),
                    ob_start=np.array([0, 2], np.int64), ob_pose=np.array([0, 1], np.int64),
                    ob_cam=np.array([0, 0], np.int64), ob_z=np.array([[300.0, 200.0], [290.0, 200.0]]),
                    ob_isig=np.ones(2), ob_valid=np.ones(2, np.bool_), fx=400.0, fy=400.0, cx=300.0, cy=200.0,
                    B=0.1, huber_k=HUBER_K, prior=None)
    Wn.solve(pb, iters=1)
    Wn.final_linearisation(pb)
    try:                                   # VI: compile imu_factor / vi_init before the clock starts (its first call
        from xclean.vio import vi_init as V    # otherwise JIT-compiles inside a keyframe, ~3.7 s in M2b)
        V.warmup()
    except Exception:
        pass
    # P1.2 W3 item 1: the pose-free kernels (posefree_cost / posefree_step) on a 1-landmark pf block
    pb2 = Wn.Problem(Rwb=np.stack([np.eye(3)] * 2), pwb=np.array([[0.0, 0, 0], [0.1, 0, 0]]), Rbc=np.eye(3),
                     pbc=np.zeros(3), lmp=np.array([[0.0, 0.0, 0.2]]), lm_host=np.array([0], np.int64),
                     lm_mask=np.ones(1, np.bool_), rho_meas=np.array([0.2]), rho_isig=np.array([10.0]),
                     ob_start=np.array([0, 2], np.int64), ob_pose=np.array([0, 1], np.int64),
                     ob_cam=np.array([0, 0], np.int64), ob_z=np.array([[300.0, 200.0], [290.0, 200.0]]),
                     ob_isig=np.ones(2), ob_valid=np.ones(2, np.bool_), fx=400.0, fy=400.0, cx=300.0, cy=200.0,
                     B=0.1, huber_k=HUBER_K, prior=None,
                     pf=Wn.PoseFree(np.array([0], np.int64), np.array([[310.0, 205.0]]), 1.0, np.array([0.25]),
                                    np.array([10.0]), np.array([[0.02, 0.01, 0.25]])))
    Wn.solve(pb2, iters=1)
    Wn.final_linearisation(pb2)
    uv = np.zeros((1, 2))
    S = np.zeros((1, 2, 2))
    Wn.predict_projection(np.eye(3), np.zeros(3), np.eye(3), np.zeros(3), np.eye(3)[None], np.zeros((1, 3)),
                          np.array([[0.0, 0.0, 0.2]]), np.zeros((6, 6)), np.zeros((1, 3, 3)), 1.0, 400.0, 400.0,
                          300.0, 200.0, 600.0, 400.0, uv, S, np.zeros(1, np.bool_), np.zeros(1))
