#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.2 W1: the FAULT-ATTRIBUTING IMU WATCHDOG (an internal note W1; design + evidence:
an internal note).  Pure python / numpy / scipy, CPU, no engine state: the engine feeds one
observation per frame and acts on the returned decision.  Our own code; design references are cited, nothing copied.

WHAT IT REPLACES.  P1.1's D3 demoted the IMU when the window's whitened IMU chi2 failed its P_GATE test at two
consecutive keyframes.  That test cannot tell a bad IMU from a struggling camera (v2_03 demoted on a camera fault,
inliers 173 -> 61; MadMax cycling 8 / 23 times).  W1 keeps the IMU-side detection (a chi2 test of the IMU residual at
the class-D level) and adds the missing half: an ATTRIBUTION by the camera's own self-consistency, measured on the
part of the visual pipeline that does not depend on the IMU.

THE OBSERVATION PER FRAME (all measured at run time, class C):
  c_new   the NEWEST window IMU factor's whitened chi2 after the frame's window solve (kappa_imu applied, exactly as the
          solve weighs it; 15 dof) - the factor that ends at this frame; None when the IMU is not admitted / no factor
  q_A     stage-A consensus fraction: PnP-RANSAC inliers / prediction-free descriptor matches (engine_live._track
          stage A: descriptor matching against the window's landmarks + PnP-RANSAC, no motion prediction used)
  n_A     stage-A inlier count (the consensus size)
  s_A     stage-A predictive residual scale [px] (median reprojection residual of the stage-A inliers at the stage-A
          pose / the Rayleigh median, as _track computes it for its stage-B gate)
  vis_ok  False when stage A produced no consensus (too few matches, RANSAC failure, coasting frame, empty map)
Why stage A: it is the only visual estimate the IMU cannot steer.  The window's own visual statistics (post-solve
inlier counts, outlier counts, the window sigma, the joint pose's visual cost) are CONTAMINATED by a faulty IMU
through the joint solve; measured on rosario_04 (D3 off, frame 45): the vision-only PnP kept 121 of 143 consensus
inliers (88 %) while the joint solve rejected 67 of them and the recorder's n_inliers fell 155 -> 54.

THE RULE (constants below; every level is P_GATE):
  1. IMU disagreement on the newest interval: fail_k = c_new > chi2_15(P_GATE).  An EPISODE is a maximal run of
     consecutive frames with fail_k.
  2. ATTRIBUTION, decided once at the episode's onset frame k0 and held for the episode (sticky): the camera is
     CERTIFIED self-consistent when, on both endpoints of the failing factor (frames k0 - 1 and k0), stage A produced a
     consensus and none of q_A (low), log n_A (low), log s_A (high) is anomalous against its own robust baseline =
     the last SCALE_WINDOW stage-A values before frame k0 - 1 (median, 1.4826 MAD; one-sided robust z against
     Z_VIS = Phi^-1(1 - (1 - P_GATE) / 3), Bonferroni over the three statistics), AND no failing interval of the
     episode up to the frame where it becomes sustained is a camera FRAME DROP (round(dt / period) >= 2, period =
     1 / calib camera_fps; timestamps are not steered by the IMU, so this check may look past the onset; POST HOC,
     declared in our design notes).  Certified -> the episode is attributed to the IMU; otherwise to the camera (also
     when the baseline holds fewer than SCALE_WINDOW values: a camera that cannot be certified is never used to blame
     the IMU).
     Why at the onset only: once the joint solve has followed a faulty IMU, the window's map is distorted and stage A
     degrades too (rosario_03 D3 off: s_A robust z +1.1 at the onset frame 191, +9.0 at frame 200).  The pre-onset
     baseline is the last evidence the IMU has not yet touched.
  3. An episode is SUSTAINED when it reaches N_CONSEC consecutive failing frame intervals.  A single failing interval
     is what one bad frame produces (the factors around it), which the window's robust kernel and 0.999 outlier gate
     already absorb (v2_03 frame 633: 250 of the 311 PnP inliers rejected by the joint solve; the next factor passes).
  4. DEMOTE at the N_CONSEC-th SUSTAINED episode attributed to the IMU.  The class-D rule "N_CONSEC consecutive
     failures" (DESIGN_P1 6.4) asks for independent repetitions; consecutive frames inside one episode are one motion
     event, not independent evidence, so the repetition is counted in episodes.
     (POST HOC, declared: this counting level was chosen after seeing v2_03's single camera-certified sustained
     episode at frames 1117-1120 in the D3-off run; declared in our design notes.)
  5. After a W1 demotion the IMU is BLOCKED for the run: no D1 / dynamic re-admission.  A demotion is a P_GATE
     rejection of this IMU's model with a certified camera; the admission tests check initialisation quality, not the
     fault, so re-admitting re-tests a rejected model (the measured MadMax cycling).  An external model change (W2's
     clock-offset estimate converging) may lift it - the integration's decision, not W1's.
  A camera-attributed episode keeps the IMU and takes NO new action: the camera's bad observations are distrusted by
  the window's existing robust kernel and outlier gate.

CONSTANTS
  P_GATE = 0.999          D  carried (vi_init.P_GATE, DESIGN_P1 13)
  N_CONSEC = 2            D  carried (vi_init.N_CONSEC, DESIGN_P1 6.4) - used twice: sustained = 2 intervals,
                             demotion = 2 sustained IMU episodes (4. above, POST HOC level)
  SCALE_WINDOW = 10       B  carried (engine_live SCALE_WINDOW = N_KF + N_FR, the window's memory)
  IMU_DOF = 15            A  the IMU factor's dimension (imu_factor residual)
  N_VIS_TESTS = 3         D  the Bonferroni count of the visual tests (q_A, n_A, s_A)
  C_IMU = chi2_15^-1(P_GATE) = 37.70;  Z_VIS = Phi^-1(1 - (1 - P_GATE)/3) = 3.403;  MAD_K = 1 / Phi^-1(3/4)
  frame drop: dt > 1.5 camera periods   A + B  camera_fps (A); the rounding midpoint to the nearest period count (B)
  REALISED SIZE (stated, not tuned): on calm frames the camera is not certified 7-44 % of the time (nominal 0.1 %;
  data_w1_calibration.json) - the robust z on a 10-value baseline is heavy-tailed; the error runs toward keeping
  the IMU.
Nothing is carried from any Rosario measurement.

References (design, cited; no code taken): Baarda 1968 (data snooping) and Teunissen 1990 ("An integrity and quality
control procedure for use in multi sensor integration": detection - identification - adaptation); Hampel 1974 (the
median / MAD robust scale); Wald 1945 (sequential evidence, the counting of independent events).

    from xclean.vio.imu_watch import ImuWatch
    w = ImuWatch()
    d = w.update(frame=k, imu_admitted=eng.vi, c_new=..., vis_ok=..., q_A=..., n_A=..., s_A=...)   # every frame
    if d["demote"]: eng demotes (new segment at this frame's pose) and keeps d["blocked"] = True for the run
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.stats import chi2 as _chi2, norm as _norm

P_GATE = 0.999
N_CONSEC = 2
SCALE_WINDOW = 10
IMU_DOF = 15
N_VIS_TESTS = 3
MAD_K = 1.482602218505602
C_IMU = float(_chi2.ppf(P_GATE, IMU_DOF))
Z_VIS = float(_norm.ppf(1.0 - (1.0 - P_GATE) / N_VIS_TESTS))

__all__ = ["P_GATE", "N_CONSEC", "SCALE_WINDOW", "IMU_DOF", "N_VIS_TESTS", "C_IMU", "Z_VIS", "robust_z",
           "VisualBaseline", "ImuWatch", "stage_a_obs", "engine_hook_factory"]


def robust_z(base, x):
    """(x - median(base)) / (1.4826 MAD(base)); +-inf when the MAD is 0 and x differs from the median; None when base
    is empty or x is None."""
    if x is None or base is None or len(base) == 0:
        return None
    b = np.asarray(base, float)
    m = float(np.median(b))
    s = float(MAD_K * np.median(np.abs(b - m)))
    if s > 0.0:
        return (float(x) - m) / s
    if float(x) == m:
        return 0.0
    return math.copysign(math.inf, float(x) - m)


class VisualBaseline:
    """The camera's own recent history: the last SCALE_WINDOW stage-A values (frames with a consensus), each stored
    with its frame index so a baseline can be taken strictly BEFORE a given frame."""

    def __init__(self, n=SCALE_WINDOW):
        self.n = int(n)
        self.buf = deque(maxlen=self.n + 2)          # + the two frames of a failing factor's span
        self.last = {}                               # frame -> its stage-A record (the two newest frames)

    def push(self, frame, vis_ok, q_A, n_A, s_A):
        rec = None
        if vis_ok and q_A is not None and n_A is not None and n_A > 0 and s_A is not None and s_A > 0:
            rec = (int(frame), float(q_A), math.log(float(n_A)), math.log(float(s_A)))
            self.buf.append(rec)
        self.last = {k: v for k, v in self.last.items() if k >= int(frame) - 1}
        self.last[int(frame)] = rec

    def certify(self, frames):
        """Certify the camera on `frames` (the failing factor's endpoints).  -> (certified, detail dict)."""
        f0 = min(frames)
        base = [r for r in self.buf if r[0] < f0][-self.n:]
        det = {"n_base": len(base), "z": {}, "why": None}
        if len(base) < self.n:
            det["why"] = "baseline holds %d < %d stage-A values" % (len(base), self.n)
            return False, det
        cols = {"q_A": [r[1] for r in base], "n_A": [r[2] for r in base], "s_A": [r[3] for r in base]}
        worst = {"q_A": math.inf, "n_A": math.inf, "s_A": -math.inf}
        for f in frames:
            rec = self.last.get(int(f))
            if rec is None:
                det["why"] = "no stage-A consensus at frame %d" % f
                return False, det
            zq = robust_z(cols["q_A"], rec[1])
            zn = robust_z(cols["n_A"], rec[2])
            zs = robust_z(cols["s_A"], rec[3])
            worst["q_A"] = min(worst["q_A"], zq)
            worst["n_A"] = min(worst["n_A"], zn)
            worst["s_A"] = max(worst["s_A"], zs)
        det["z"] = worst
        bad = [k for k in ("q_A", "n_A") if worst[k] < -Z_VIS] + (["s_A"] if worst["s_A"] > Z_VIS else [])
        if bad:
            det["why"] = "visual anomaly: " + ",".join(bad)
            return False, det
        return True, det


class ImuWatch:
    """W1 state machine (one per run).  update() once per frame, in frame order, from the very first frame (the
    visual baseline is built whether or not the IMU is admitted).  period: the camera period [s] (class A, calib
    camera_fps); None -> the running median of the observed frame intervals (class C)."""

    def __init__(self, period=None):
        self.vis = VisualBaseline()
        self.period = None if period is None else float(period)
        self._dts = deque(maxlen=SCALE_WINDOW)
        self.prev_t = None
        self.ep = None                 # the open episode: dict(f0, len, attr, sustained, cert)
        self.n_imu_sustained = 0
        self.n_cam_sustained = 0
        self.blocked = False
        self.demote_frame = None
        self.prev_frame = None
        self.events = []               # [frame, kind, detail]
        self.stats = {"frames": 0, "admitted_frames": 0, "fail_frames": 0, "episodes": 0, "episodes_imu": 0,
                      "episodes_cam": 0, "sustained_imu": 0, "sustained_cam": 0}

    def _gap(self, t):
        """A camera FRAME DROP on the interval ending at this frame: round(dt / period) >= 2 (at least one camera
        period without a frame; the rounding midpoint 1.5 periods is class B)."""
        if t is None or self.prev_t is None:
            return False
        dt = float(t) - self.prev_t
        per = self.period if self.period else (float(np.median(self._dts)) if len(self._dts) else None)
        if dt > 0:
            self._dts.append(dt)
        return bool(per is not None and dt > 1.5 * per)

    def update(self, frame, imu_admitted, c_new, vis_ok, q_A=None, n_A=None, s_A=None, t=None):
        """-> dict(demote, blocked, fail, episode (or None), attribution (or None), n_imu_sustained, detail).
        t: the frame stamp [s] (the frame-drop test; None disables it)."""
        frame = int(frame)
        self.stats["frames"] += 1
        self.vis.push(frame, vis_ok, q_A, n_A, s_A)
        gap = self._gap(t)
        if t is not None:
            self.prev_t = float(t)
        out = {"frame": frame, "demote": False, "blocked": self.blocked, "fail": False, "episode": None,
               "attribution": None, "n_imu_sustained": self.n_imu_sustained, "detail": None}
        contiguous = self.prev_frame is not None and frame == self.prev_frame + 1
        self.prev_frame = frame
        if self.blocked or not imu_admitted or c_new is None or not np.isfinite(c_new):
            self.ep = None
            return out
        self.stats["admitted_frames"] += 1
        fail = float(c_new) > C_IMU
        out["fail"] = fail
        if not fail:
            self.ep = None
            return out
        self.stats["fail_frames"] += 1
        if self.ep is None or not contiguous:
            cert, det = self.vis.certify([frame - 1, frame])
            if cert and gap:
                cert, det = False, dict(det, why="camera frame drop on the failing interval")
            self.ep = {"f0": frame, "len": 0, "attr": "imu" if cert else "camera", "sustained": False, "cert": det}
            self.stats["episodes"] += 1
            self.stats["episodes_" + ("imu" if cert else "cam")] += 1
            self.events.append([frame, "onset_" + self.ep["attr"], det])
        ep = self.ep
        if gap and ep["attr"] == "imu" and not ep["sustained"]:
            # a frame drop inside the episode before it is sustained: the camera's timing failed on a failing
            # interval -> not certified (timestamps are not steered by the IMU, so this check may look past the onset)
            ep["attr"] = "camera"
            ep["cert"] = dict(ep["cert"], why="camera frame drop on interval %d of the episode" % (ep["len"] + 1))
            self.stats["episodes_imu"] -= 1
            self.stats["episodes_cam"] += 1
            self.events.append([frame, "reattributed_camera_drop", ep["cert"]])
        ep["len"] += 1
        out["episode"] = {"f0": ep["f0"], "len": ep["len"]}
        out["attribution"] = ep["attr"]
        out["detail"] = ep["cert"]
        if ep["len"] >= N_CONSEC and not ep["sustained"]:
            ep["sustained"] = True
            if ep["attr"] == "imu":
                self.n_imu_sustained += 1
                self.stats["sustained_imu"] += 1
                self.events.append([frame, "sustained_imu", self.n_imu_sustained])
                if self.n_imu_sustained >= N_CONSEC:
                    self.blocked = True
                    self.demote_frame = frame
                    out["demote"] = True
                    self.events.append([frame, "demote", self.n_imu_sustained])
            else:
                self.n_cam_sustained += 1
                self.stats["sustained_cam"] += 1
                self.events.append([frame, "sustained_camera", self.n_cam_sustained])
        out["blocked"] = self.blocked
        out["n_imu_sustained"] = self.n_imu_sustained
        return out

    def summary(self):
        return dict(self.stats, n_imu_sustained=self.n_imu_sustained, n_cam_sustained=self.n_cam_sustained,
                    blocked=self.blocked, demote_frame=self.demote_frame, events=self.events[:200])


def stage_a_obs(pnp_calls, s_A):
    """The stage-A observation from the engine's per-frame PnP call list [(tag, n_matches, n_inliers, ok), ...]
    (tag 'p1pnpA' = stage A).  -> (vis_ok, q_A, n_A, s_A)."""
    for tag, n_in, n_out, ok in pnp_calls:
        if tag in ("p1pnpA", b"p1pnpA"):
            if ok and n_in > 0 and n_out > 0 and s_A is not None and s_A > 0:
                return True, n_out / float(n_in), int(n_out), float(s_A)
            return False, None, None, None
    return False, None, None, None


def engine_hook_factory():
    """For the design track's trace driver: a per-frame
    LOG-ONLY W1 evaluation from the trace hook's frame record.  Never acts on the engine."""
    w = None

    def on_frame(eng, rec):
        nonlocal w
        if w is None:
            fps = (getattr(eng.src, "calib", None) or {}).get("camera_fps")
            w = ImuWatch(period=(1.0 / float(fps)) if fps else None)
            on_frame.watch = w
        tr = rec.get("track", {})
        vis_ok, q, n, s = stage_a_obs(tr.get("pnp", []), tr.get("s_A"))
        jt = rec.get("joint") or {}
        ic = jt.get("imu_chi2") or []
        c = ic[-1] if (rec.get("vi") and ic) else None
        return w.update(rec["f"], bool(rec.get("vi")), c, vis_ok, q, n, s, t=rec["t"])
    on_frame.watch = None
    return on_frame
