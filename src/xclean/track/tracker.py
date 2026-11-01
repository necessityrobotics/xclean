#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""THE FRAME LOOP - features to a pose, with every stage on the ledger.

    tr = Tracker(dataset, frontend, depth, ledger, kit)
    for f in dataset.frames():
        tr.step(f)

WHAT THIS IS AND IS NOT
    This is P2: tracking to a pose, ODOMETRY ONLY.  There is no bundle
    adjustment, no loop closure and no map here - P2's exit is precisely "the
    odometry-only ATE per row, recorded as the CEILING the backend must improve
    on".  Every number it produces is a floor to be beaten, not a result.

★ THE KEYFRAME POLICY IS DERIVED, NOT A FRAME COUNT (plan 5.2)
    "A frame count at 20 fps means something different at 27.8 fps."  Three
    triggers, each a function of calibration or of what the run measures:

      A. TRANSLATION >= the STEREO BASELINE.  This is the honest information
         criterion: once the camera has moved by one baseline, the temporal
         pair carries as much triangulation angle as the stereo pair does, so
         a new keyframe adds genuinely new geometry.  Class B - it is
         `rig.baseline_m`, and it is 11 cm on EuRoC and 54 cm on KITTI, which
         is exactly the scaling a frame count cannot express.
      B. ROTATION >= a quarter of the image width in pixel shift, i.e.
         `theta = 0.25 * w / fx`.  Class B: rotation destroys overlap at a rate
         set by the focal length, not by the scene.
      C. TRACKING HEALTH: inliers below `3 x min_pnp_inliers`.  Class C/D -
         the floor is conditioning, the headroom is measured.

    Every trigger emits a `Decision` carrying its dials, so a keyframe rate can
    be replayed and argued with instead of guessed at.

★ DEPTH RUNS WHERE ITS OUTPUT IS CONSUMED
    Landmarks are created at keyframes, so dense RAFT runs at keyframes - about
    11 % of frames.  This is the bounded-work invariant applied to the front
    end, and it is REPORTED rather than hidden: the per-frame depth cost is
    still on the ledger, it simply is not paid on frames that would discard it.
    Non-keyframe views still get a pose, which is what the map build kit wants
    them for.
"""
from __future__ import annotations

import os
from collections import namedtuple

import numpy as np

from xclean.core.derive import (MIN_PNP_INLIERS, DISPARITY_NOISE_PX,
                                  HUBER_K)
from xclean.core.lie import (se3_inv, so3_log, so3_project, rot_angle,
                               se3_from_Rt, se3_exp, se3_log)
from xclean.obs.decision import Decision
from xclean.backend.ba import BAState, LandmarkStore, LocalBA
from xclean.track.pose import (solve_pose, project, FEATURE_SIGMA_PX, pnp_verify_mode,
                               pnp_verify_summary)

# ARM XTEN_GBA_DEPTH_TRUST=1 (2026-09-26). Class D: the upper QUARTILE of the map's own disparity distribution - the
# nearest quarter of landmarks, whose stereo depth carries the smallest relative error (sigma_d / d) - keeps the metric
# gauge in the final global BA; see depth_trust_prior().
GBA_DEPTH_TRUST_KEEP = 0.25


def depth_trust_prior(prior_id, fB, keep=GBA_DEPTH_TRUST_KEEP):
    """Final-GBA prior with the inverse-depth term kept only for the nearest `keep` fraction of landmarks.

    ★ WHY. A disparity error that is harmless at 20 px is a percent of depth at 3 px, so the far band sets the map's
    SCALE. Measured on saved GBA problems (an internal replay study): the near band is the good scale anchor
    (dropping it: KITTI +19..+72 % ATE), the mid/far priors pull scale wrong, and the effect grows as disparity
    shrinks - the signature of a small CONSTANT disparity offset that RAFT and our own sparse matcher share (they
    agree to 0.1 px), i.e. not the matcher. ORB-SLAM3 sidesteps it by using stereo depth only for close points.
    Here the cut is the map's OWN distribution, not a fixed range. Landmarks outside it keep the BEARING prior; only
    the inverse-depth weight is zeroed, so their depth comes from multi-view parallax.
    Returns (new_prior, stats).
    """
    m0P, rho0P, wbP, invsigP, validP = prior_id
    ok = validP > 0.5
    if ok.sum() < 4:
        return prior_id, {"applied": False}
    disp = rho0P * fB
    thr = float(np.quantile(disp[ok], 1.0 - keep))
    drop = ok & (disp < thr)
    inv2 = np.where(drop, 0.0, invsigP)
    return (m0P, rho0P, wbP, inv2, validP), {"applied": True, "keep": keep, "thr_disp_px": thr,
                                            "thr_depth_m": float(fB / max(thr, 1e-9)),
                                            "kept": int((ok & ~drop).sum()), "dropped": int(drop.sum())}


KF_INLIER_HEADROOM = 3.0        # Class D: keyframe before conditioning is at
                                # risk, not after

# ---- T1-b (tier-1 programme, 2026-09-29): the health trigger once the feature budget is EXHAUSTED ----------
# ARM XTEN_KF_HEALTH=lookahead.  Default (unset / off): the old rule, and every Decision is byte-identical to
# the pre-T1-b inline code (proved by tests/test_t1_kfhealth.py, live tree vs staged tree).
#
# THE DEFECT.  The health branch fires when inliers < KF_INLIER_HEADROOM x MIN_PNP_INLIERS = 75.  Its premise
# is that a keyframe restores headroom (fresh landmarks -> more inliers).  The feature-budget controller
# (stages/features.py, HEADROOM = 3.0, the same 75) grows the budget on the same premise.  Once the budget is
# AT ITS CAP and the inlier count still sits below 75 (~65 on Rosario), neither lever has anything left: the
# premise is false and the trigger fires on nearly every frame (rosario_06: a keyframe on 79 % of frames,
# 77 % of them health-triggered; rosario_04: 71 %).
#
# THE RULE - only while fe.budget >= fe.cap; otherwise the old rule, exactly:
#     fire when the ONE-FRAME-AHEAD prediction of the inlier decay since the reference keyframe falls below
#     MIN_PNP_INLIERS = 25:      pred = inl * (inl / n_ref) ** (1 / m)
#     n_ref = the reference keyframe's creating-frame inliers (LOG's kf_evidence; when it is missing the old
#     rule decides and the fallback is COUNTED), m = frames since that keyframe (>= 1).
#     Never extrapolate upward: inl >= n_ref gives pred = inl (pred = min(pred, inl)).
#     No new constant: 25 is the PnP floor the old rule already scales by 3.
#
# PRE-REGISTERED HIDDEN CHOICES (stated, not tuned; also in summary.kf_health.pre_registered):
#     (1) geometric decay, AVERAGED since the reference keyframe (one rate over the m frames, not the last
#         step - a single-step rate would react to the frame-to-frame inlier noise);
#     (2) a ONE-FRAME horizon (the next frame is the first one a keyframe inserted now can help);
#     (3) the conditioning margin is given up at cap: keyframes then land at ~25-30 inliers instead of 75.
KF_HEALTH_ENV = "XTEN_KF_HEALTH"
KF_HEALTH_MODES = ("off", "lookahead")
KF_HEALTH_REASON = "tracking health: inliers approaching the conditioning floor"
KF_HEALTH_REASON_LOOKAHEAD = ("tracking health (lookahead): the inlier decay since the reference "
                              "keyframe predicts fewer than the PnP floor one frame ahead, with the "
                              "feature budget at its cap")
KF_HEALTH_CHOICES = ("geometric decay averaged since the reference keyframe",
                     "one-frame horizon",
                     "conditioning margin given up at cap: keyframes land at ~25-30 inliers")
# fire/margin/threshold/reason are what the tracker's Decision is built from; rule names which rule decided
# ("old" | "lookahead" | "fallback" = lookahead requested at cap but no reference evidence); pred is the
# one-frame-ahead prediction (None unless the lookahead rule decided)
HealthVerdict = namedtuple("HealthVerdict", "fire margin threshold reason rule pred")


def kf_health_mode(env=None):
    """The run's health-rule mode from XTEN_KF_HEALTH: unset / "" / "0" / "off" -> "off";
    "lookahead" -> "lookahead"; anything else is an error (a misspelt queue flag must not
    silently measure the default)."""
    v = (os.environ if env is None else env).get(KF_HEALTH_ENV, "")
    v = (v or "").strip().lower()
    if v in ("", "0", "off"):
        return "off"
    if v == "lookahead":
        return "lookahead"
    raise RuntimeError("%s=%r is not a mode: unset/off or lookahead" % (KF_HEALTH_ENV, v))


def health_trigger(inl, thr_i, n_ref, m, at_cap, mode):
    """The keyframe health rule, pure.  -> HealthVerdict(fire, margin, threshold, reason, rule, pred).

    inl    the current frame's PnP inlier count (int, mask.sum())
    thr_i  the old rule's threshold, KF_INLIER_HEADROOM x MIN_PNP_INLIERS
    n_ref  the reference keyframe's creating-frame inliers (None when unknown)
    m      frames since the reference keyframe (>= 1)
    at_cap fe.budget >= fe.cap at this decision
    mode   kf_health_mode(): "off" or "lookahead"

    mode "off", or the budget below its cap, or (mode lookahead at cap) without
    reference evidence: the OLD rule - (inl < thr_i, inl - thr_i, thr_i, the old
    reason) - exactly the pre-T1-b inline code, so the Decision is unchanged.
    Otherwise the lookahead rule described at KF_HEALTH_ENV above.
    """
    if mode != "lookahead" or not at_cap:
        return HealthVerdict(inl < thr_i, inl - thr_i, thr_i, KF_HEALTH_REASON, "old", None)
    if n_ref is None or n_ref <= 0:
        return HealthVerdict(inl < thr_i, inl - thr_i, thr_i, KF_HEALTH_REASON, "fallback", None)
    floor = float(MIN_PNP_INLIERS)
    m = max(int(m), 1)
    if inl >= n_ref:
        pred = float(inl)                       # never extrapolate upward
    else:
        pred = min(float(inl) * (float(inl) / float(n_ref)) ** (1.0 / m), float(inl))
    return HealthVerdict(pred < floor, pred - floor, floor, KF_HEALTH_REASON_LOOKAHEAD,
                         "lookahead", pred)


# ---- FRONT GATE (kitti_01 root cause, 2026-10-02; an internal note; ported to the engine as P2 on
# 2026-10-09 with the verifier's fixes, an internal note) ---------------------------------------------------------
# THE DEFECT.  The matching window hands every LightGlue match to PnP with NO use of the pose prediction.  On a
# highway with a periodic crash barrier 66 % of the correspondences are wrong (measured tracker-independently by
# the verifier: 72 % of near matches on the hot frames) and RANSAC takes the alias: ~160 stalled steps per run,
# 70 m of path lost, SE3 ATE 55-82 m against ORB-SLAM3's 11-13 m.  ORB-SLAM3 never matches without its motion model:
# a projection window around the prediction, doubled when fewer than 20 matches survive (Tracking.cc:2882-2894).
#
# ARM XTEN_FRONT_GATE=recount  (default off: every statement below is behind `if self._fg_active`).
#   THE WINDOW picks the pose HYPOTHESIS.  A correspondence enters the PnP solve when its Mahalanobis distance under
#   the PREDICTED pose is below chi2(2 dof, 99.9 %).  The covariance is the tracker's OWN measured prediction-error
#   history (the last INNOV_SPAN frames of the final pose's innovation against the prediction, per axis: robust spread
#   1.4826 MAD plus the squared median offset), propagated through the projection Jacobian to a per-correspondence
#   pixel covariance, plus the detector's 1 px.  If fewer than KF_INLIER_HEADROOM x MIN_PNP_INLIERS survive the
#   window is DOUBLED (radius) until they do or until the window holds every point in front of the camera (ORB3's
#   "search again with a wider window", without a cap: the limit of the doubling is the ungated set).  If even that
#   leaves < MIN_PNP_INLIERS the frame is left ungated.  Nothing is gated before INNOV_SPAN frames of history exist.
#   THE RECOUNT then re-counts EVERY correspondence (the full ungated set) under the pose the gated solve returned,
#   with the plain solver's own 3-sigma reprojection gate: the inlier mask that credits observations to landmarks and
#   feeds keyframing and the feature budget covers the whole set (ORB-SLAM3's TrackLocalMap re-projects with the
#   refined pose), so the window chooses the pose but never starves the map.
#   If the gated solve FAILS (no pose) the frame is re-solved on the UNGATED correspondences exactly as the baseline
#   would (ORB3's prior-free fallback, TrackReferenceKeyFrame): the gate can only ever change a frame it solved,
#   never turn a solvable frame into a coast.
# NOT PORTED to the engine (both RAISE at Tracker construction, so a stale queue line cannot silently run the default):
#   XTEN_FRONT_GATE=innov - the gated subset kept as the frame's inlier set: REFUTED, it starves the map (kitti_00
#     coasted 388 frames in the builder's replay; an internal note section 0);
#   XTEN_FRONT_VEL=innov - the innovation-tested velocity guard (handoff D3: "UNSAFE" alone, 32.6 -> 40.5 m).
# NO TYPED PIXEL RADIUS.  Constants: the chi2 quantile (statistics), 75 and the doubling (both borrowed, stated),
# INNOV_SPAN = the span step() already uses for the measured parallax history (parallax_hist[-20:]).
FRONT_GATE_ENV = "XTEN_FRONT_GATE"
FRONT_VEL_ENV = "XTEN_FRONT_VEL"            # NOT PORTED: read only to refuse it
INNOV_SPAN = 20
CHI2_2DOF_999 = 13.815510557964274      # -2 ln(0.001): exact for 2 dof
FRONT_GATE_CHOICES = ("history span = the 20 frames of the tracker's measured parallax history",
                      "gate only once the history is full",
                      "robust per-axis spread: 1.4826 MAD about the median plus the squared median offset",
                      "window doubling (radius) until 75 survive or the ungated set (no cap)",
                      "recount: every correspondence re-counted under the final pose with the solver's own 3-sigma gate",
                      "failed gated solve -> ungated re-solve (baseline behaviour)")


def front_gate_mode(env=None):
    """XTEN_FRONT_GATE: unset / "" / "0" / "off" -> "off"; "recount" -> "recount" (the gate chooses the HYPOTHESIS,
    then every correspondence is re-counted under the final pose).  "innov" (REFUTED, not ported) and anything else
    raise RuntimeError: a misspelt or refuted queue flag must not silently measure the default."""
    v = ((os.environ if env is None else env).get(FRONT_GATE_ENV, "") or "").strip().lower()
    if v in ("", "0", "off"):
        return "off"
    if v == "recount":
        return "recount"
    if v == "innov":
        raise RuntimeError("%s=innov is not part of this engine (it starves the map); the mode is recount"
                           % FRONT_GATE_ENV)
    raise RuntimeError("%s=%r is not a mode: unset/off or recount" % (FRONT_GATE_ENV, v))


def front_vel_refuse(env=None):
    """XTEN_FRONT_VEL is NOT ported to the engine: unset / "" / "0" / "off" -> None; any other value raises RuntimeError
    (a queue line that asks for the velocity guard must fail, not run without it)."""
    v = ((os.environ if env is None else env).get(FRONT_VEL_ENV, "") or "").strip().lower()
    if v in ("", "0", "off"):
        return None
    raise RuntimeError("%s=%r: the velocity guard is not part of this engine; unset it"
                       % (FRONT_VEL_ENV, v))


def innov_variance(hist):
    """Per-axis second moment of the prediction innovation about ZERO from a (n,6) history
    [translation in the predicted camera axes (m), rotation vector (rad)]: the robust spread squared plus the
    squared median offset (a systematic prediction bias is error too).  -> (6,)"""
    H = np.asarray(hist, float)
    med = np.median(H, axis=0)
    sd = 1.4826 * np.median(np.abs(H - med), axis=0)
    return sd ** 2 + med ** 2


def innov_error(T_pred, T_fin):
    """The 6-vector innovation of the final pose against the prediction, in the predicted camera's axes."""
    Rp = T_pred[:3, :3]
    return np.concatenate([Rp.T @ (T_fin[:3, 3] - T_pred[:3, 3]),
                           so3_log(so3_project(Rp.T @ T_fin[:3, :3]))])


def front_gate_keep(P, uv, T_pred, var, fx, cx, cy, enough):
    """The innovation window on the data association, pure.

    P (n,3) world landmarks, uv (n,2) matched keypoints, T_pred the predicted T_wc, var (6,) = innov_variance(),
    enough = the survivor count that stops the widening.  -> (keep (n,) bool or None, scale, n_in_front):
    keep None = leave the set ungated (fewer than MIN_PNP_INLIERS in front of the camera, or fewer than
    MIN_PNP_INLIERS survive even the widest window)."""
    from xclean.track.pose import _jacobian
    n = len(P)
    if n == 0:
        return None, 1.0, 0
    T_cw = se3_inv(T_pred)
    Pc = (T_cw[:3, :3] @ P.T).T + T_cw[:3, 3]
    z = np.maximum(Pc[:, 2], 1e-6)
    pu = fx * Pc[:, 0] / z + cx
    pv = fx * Pc[:, 1] / z + cy
    J = _jacobian(Pc, fx, fx)                       # (n,2,6) [rho, phi]
    su2 = FEATURE_SIGMA_PX ** 2 + np.einsum("nj,j->n", J[:, 0, :] ** 2, var)
    sv2 = FEATURE_SIGMA_PX ** 2 + np.einsum("nj,j->n", J[:, 1, :] ** 2, var)
    with np.errstate(all="ignore"):
        d2 = ((pu - uv[:, 0]) ** 2) / su2 + ((pv - uv[:, 1]) ** 2) / sv2
    ok = (Pc[:, 2] > 1e-3) & np.isfinite(d2)
    n_ok = int(ok.sum())
    s_w = 1.0
    keep = ok & (d2 < CHI2_2DOF_999)
    while int(keep.sum()) < enough and int(keep.sum()) < n_ok and s_w < 1e12:
        s_w *= 2.0
        keep = ok & (d2 < CHI2_2DOF_999 * s_w * s_w)
    if int(keep.sum()) < MIN_PNP_INLIERS:
        return None, s_w, n_ok
    return keep, s_w, n_ok


# ---- TRACK PACKAGE (a port, 2026-10-11; design and diagnosis in internal notes, ranks 4-5 of a survey of lost
# runs, with its later corrections).  Two parts, each behind its OWN default-off
# flag; with both unset not one statement below changes what the tracker computes (identity proof:
# tests/test_track_identity.py, base tree vs this tree).  A misspelt value raises at Tracker construction.
#
#   F0  XTEN_DECISION_LOG=1   provenance only: one JSON line per frame (kit/decisions.jsonl, or memory without a
#       kit) with every Decision the frame made (PnP and its guess-free retry with the failed RANSAC counts, local
#       map, staged chain, relocalisation with each candidate tried, keyframe), the referee's route (anchored / weak
#       / union / trusted motion / exemption / quarantine / rebase), mat_ev, history trust in and out, the mature-
#       visible count, budget, streaks.  Estimates byte-identical.
#   F1  XTEN_COAST_FEED=1     a coasted frame is a measurement of zero usable inliers: it is fed to the feature-budget
#       controller (today report_inliers runs only on tracked / relocalised frames, so the controller is starved of
#       exactly the frames it exists to answer; our study of lost runs, section 5 item 1).  Bit-identical on every frame
#       sequence without a coasted frame.
#   The package's parts P (XTEN_MOTION_PRED: gyro / dt-aware prediction) and R (XTEN_RELOC_GROW: relocalisation
#   match growth) were KILLED on their pre-registered CPU bars (an internal note; their code is kept
#   only in a port/DROPPED_P_R_full4.diff).  Setting either flag raises: a queue that still carries one
#   must not silently measure the default.
DECISION_LOG_ENV = "XTEN_DECISION_LOG"
COAST_FEED_ENV = "XTEN_COAST_FEED"
DROPPED_TRACK_ENVS = ("XTEN_MOTION_PRED", "XTEN_RELOC_GROW")   # killed parts: unset / "" / "0" / "off" only


def _track_flag(name, env=None):
    """unset / "" / "0" / "off" -> False; "1" / "on" -> True; anything else raises (a misspelt queue flag must not
    silently measure the default)."""
    v = ((os.environ if env is None else env).get(name, "") or "").strip().lower()
    if v in ("", "0", "off"):
        return False
    if v in ("1", "on"):
        return True
    raise RuntimeError("%s=%r is not a value: unset/0/off or 1/on" % (name, v))


def decision_log_on(env=None):
    return _track_flag(DECISION_LOG_ENV, env)


def coast_feed_on(env=None):
    return _track_flag(COAST_FEED_ENV, env)


def dropped_track_flags_unset(env=None):
    """The killed parts' flags must be unset / "" / "0" / "off"; anything else raises (their code is not here)."""
    e = os.environ if env is None else env
    for name in DROPPED_TRACK_ENVS:
        v = (e.get(name, "") or "").strip().lower()
        if v not in ("", "0", "off"):
            raise RuntimeError("%s=%r: this part is not in this engine; unset it" % (name, v))
    return True


def cost_stats(ms):
    """p50 / p95 / max / mean / n of a list of per-call costs in ms (None when empty)."""
    if not ms:
        return None
    a = np.asarray(ms, float)
    return {"n": int(len(a)), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "max_ms": float(a.max()), "mean_ms": float(a.mean()), "total_s": float(a.sum() / 1000.0)}


def _jsonable(o):
    """A Decision dial value made JSON-safe (numpy scalars and arrays, nested containers)."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else "array%s" % (o.shape,)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (bool, int, float, str)) or o is None:
        return o
    return str(o)[:80]


def decision_record(d):
    """One Decision as a compact JSON-safe dict (gate, accepted, reason, dials, margin)."""
    return {"g": d.gate, "a": bool(d.accepted), "r": str(d.reason)[:160], "d": _jsonable(d.dials or {}),
            "m": None if d.margin is None else float(d.margin)}


class _Cols:
    """A tiny struct-of-arrays that grows by doubling.

    The point is that the backend can take a VIEW of each column with no copy
    and no per-keyframe conversion.
    """

    def __init__(self, *cols):
        self._names = [c[0] for c in cols]
        self._a = {n: np.zeros(4096, dtype=d) for n, d in cols}
        self.n = 0

    def _grow(self, need):
        if need <= len(self._a[self._names[0]]):
            return
        cap = max(len(self._a[self._names[0]]) * 2, need + 4096)
        for k, v in self._a.items():
            g = np.zeros(cap, dtype=v.dtype)
            g[:self.n] = v[:self.n]
            self._a[k] = g

    def add(self, **cols):
        m = len(next(iter(cols.values())))
        self._grow(self.n + m)
        for k, v in cols.items():
            self._a[k][self.n:self.n + m] = v
        base = self.n
        self.n += m
        return base

    def col(self, name):
        return self._a[name][:self.n]

    def __getattr__(self, name):
        a = self.__dict__.get("_a")
        if a is not None and name in a:
            return a[name][:self.__dict__["n"]]
        raise AttributeError(name)


class Keyframe:
    __slots__ = ("id", "frame", "t", "T_wc", "feats", "lm_id", "kp",
                 "left", "right")

    def __init__(self, kid, frame, t, T_wc, feats, kp, lm_id, left, right):
        self.id, self.frame, self.t, self.T_wc = kid, frame, t, T_wc
        self.feats, self.kp, self.lm_id = feats, kp, lm_id
        self.left, self.right = left, right


# ---- T1-h (XTEN_GBA_ROTF=nodup, 2026-09-30): THE GYRO ENTERS THE INERTIAL GLOBAL BA TWICE ----------
# On a coupled IMU row the same gyro samples reach the run-end global BA as TWO factor classes:
#   (1) the online rotation factors `rotf` (R integrated at the bias of the moment, sigma from the
#       online rule; global_ba._rot_blocks) and
#   (2) the preintegration rotation block (the same samples re-integrated at the final bias, with
#       bg a STATE of the solve; preint.inertial_system).
# Two factors from one measurement is a double count: the rotation information is claimed twice and
# the second copy sits at a stale bias.  The RULE (tier 1, no constant): when inertial coupling is
# active in the solve, omit the rotf entry for every keyframe interval that a non-None
# preintegration factor already covers; keep rotf where coupling is refused (pres[k] is None:
# the GAPSPAN / SAMPLES refusals) and everywhere the coupling is off (no IMU, the |g| gate, the
# 0.8 gap rule, nk < 8).  "Covered" is exact: the factor's two keyframes are CONSECUTIVE
# positions (a, a+1) in the solve's keyframe list and pres[a] is not None.  A non-consecutive
# pair (none exist today: the online factor is always (prev, kid)) is not covered and is counted
# on its own.  Default (flag unset): every factor is handed over exactly as landed, and the
# provenance below is emitted regardless (LOG discipline: default-on, behaviour bit-identical).
GBA_ROTF_ENV = "XTEN_GBA_ROTF"
GBA_ROTF_MODES = ("default", "nodup")


def gba_rotf_mode(env=None):
    """The run-end global BA's rotation-factor policy from XTEN_GBA_ROTF: unset / "" / "0" /
    "off" / "default" -> "default" (every online factor is handed to the solver, as landed);
    "nodup" -> the T1-h rule (gba_rotf_dedup); anything else is an error (a misspelt queue flag
    must not silently measure the default)."""
    v = (os.environ if env is None else env).get(GBA_ROTF_ENV, "")
    v = (v or "").strip().lower()
    if v in ("", "0", "off", "default"):
        return "default"
    if v == "nodup":
        return "nodup"
    raise RuntimeError("%s=%r is not a mode: unset/off/default or nodup" % (GBA_ROTF_ENV, v))


def gba_rotf_assemble(rot_factors, kf_pos, kf_t, imu, tight):
    """The run-end global BA's rotation-factor list, pure (the loop global_refine carried inline
    until 2026-09-30, byte for byte).  `rot_factors` is keyed by keyframe id (i, j) -> (R, sigma);
    a factor is handed over as (pos_i, pos_j, R, sigma) when BOTH ends are in the solve (`kf_pos`:
    keyframe id -> position in the solve's keyframe list), in the dict's own order.  `tight` (the
    --gba-tight-gyro / --gba-robust-gyro arms) recomputes sigma from the IMU's tight rule over
    the keyframe times `kf_t` at the final bias quality; the stored sigma is whatever the online
    rule was at creation.  No T1-h logic here: see gba_rotf_dedup."""
    rotf = []
    for (i, j), (R, sg) in rot_factors.items():
        if int(i) in kf_pos and int(j) in kf_pos:
            if tight and imu is not None:
                dt_f = kf_t.get(int(j), 0.0) - kf_t.get(int(i), 0.0)
                if dt_f > 0:
                    sg = imu.prior_sigma_tight(dt_f)
            rotf.append((kf_pos[int(i)], kf_pos[int(j)], R, float(sg)))
    return rotf


def gba_rotf_dedup(rotf, pres, mode):
    """T1-h, pure.  -> (the rotf list handed to the solver, the provenance dict).

    rotf   [(a, b, R, sigma)] with a, b POSITIONS in the solve's keyframe list (gba_rotf_assemble)
    pres   the inertial term's preintegration list (pres[k] covers positions (k, k+1); None entries
           are refused intervals), or None when inertial coupling is NOT active in this solve
    mode   gba_rotf_mode(): "default" hands `rotf` over untouched (the SAME list object);
           "nodup" drops every COVERED entry and keeps the rest in their original order (the
           same list object again when nothing is covered).

    covered(a, b)  <=>  b == a + 1  and  pres is not None  and  0 <= a < len(pres)  and
                        pres[a] is not None
    Every other entry is kept under both modes: a consecutive interval with no preintegration
    factor (rotf_kept_uncovered: the refused-coupling cases the rule must preserve) and a pair that
    is not consecutive in the solve (rotf_kept_nonconsecutive, counted separately by design).
    The provenance is the same under both modes except rotf_mode, rotf_dropped_covered (0 by
    default) and rotf_handed, so a default run reports how many factors the rule WOULD drop
    (rotf_covered)."""
    if mode not in GBA_ROTF_MODES:
        raise RuntimeError("gba_rotf_dedup: mode %r is not one of %s" % (mode, GBA_ROTF_MODES))
    coupled = pres is not None
    n_pres = len(pres) if coupled else 0
    covered = []
    n_unc = n_non = 0
    for f in rotf:
        a, b = int(f[0]), int(f[1])
        if b != a + 1:
            n_non += 1
            covered.append(False)
        elif coupled and 0 <= a < n_pres and pres[a] is not None:
            covered.append(True)
        else:
            n_unc += 1
            covered.append(False)
    n_cov = int(sum(covered))
    if mode == "nodup" and n_cov:
        out = [f for f, c in zip(rotf, covered) if not c]
    else:
        out = rotf              # nothing to drop: the SAME list object, both modes
    prov = {"rotf_mode": mode, "rotf_coupled": bool(coupled),
            "rotf_total": int(len(rotf)), "rotf_covered": n_cov,
            "rotf_dropped_covered": int(len(rotf) - len(out)),
            "rotf_kept_uncovered": int(n_unc), "rotf_kept_nonconsecutive": int(n_non),
            "rotf_handed": int(len(out))}
    return out, prov


class Tracker:
    def __init__(self, ds, frontend, depth, ledger, kit=None, max_kf_map=3,
                 verbose=True, backend=True, ba_iters=5, state_window=30,
                 loop=None, pgo_every=10, imu=None, localmap=True,
                 far_band=True, inverse_depth=True, frozen_bearing=True,
                 inertial=False, inertial_inflate=3.0, ba_span=None,
                 stereo_obs=False, age_consensus=False, scale_bias=False,
                 gba_tight_gyro=False, gba_robust_gyro=False,
                 aniso_parallax=True, structure_welds=False,
                 preint_cal=False, kp_refine=False, vision_trust=False):
        self.ds = ds
        self.rig = ds.rig
        self.fe = frontend
        self.depth = depth
        self.led = ledger
        self.kit = kit
        self.max_kf_map = int(max_kf_map)
        self.verbose = verbose

        # ★ TWO DIFFERENT WINDOWS, AND CONFLATING THEM IS EXPENSIVE.
        #   `kfs`         the MATCHING window - every keyframe in it costs one
        #                 LightGlue call PER FRAME, so it stays at 3.
        #   `state`       the ESTIMATOR window - keyframes the backend may
        #                 optimise.  It costs nothing per frame and everything
        #                 per keyframe, so it holds `state_window` (30, the
        #                 plan 5.1 span ceiling) and the covisibility rule
        #                 picks the actual solve window out of it.
        # Setting the matching window to 8 to give the backend more structure
        # would have multiplied the front end by 2.7x for no extra constraint.
        self.kfs = []                       # the MATCHING window
        self.state_kfs = []                 # the ESTIMATOR window (ids)
        self.state_window = int(state_window)
        self.all_kf = []                    # (kf_id, frame, t, T_wc)
        self.store = LandmarkStore()
        self.state = BAState(self.store)
        # plan 5.4: landmarks are one inverse depth in a host keyframe.  The
        # Euclidean path stays reachable as `--xyz-landmarks`, because the
        # far-band claim is only meaningful against it.
        self.inverse_depth = bool(inverse_depth)
        self.frozen_bearing = bool(frozen_bearing)
        # ARM XTEN_STEREO_OVERLAP=1 (2026-09-24): a keypoint whose stereo correspondence u - d falls LEFT of the
        # right image has no stereo measurement - the dense network can only extrapolate there. Default OFF;
        # with it off no statement below executes differently (every use is inside `if self.stereo_overlap`).
        import os as _os
        self.stereo_overlap = _os.environ.get("XTEN_STEREO_OVERLAP") == "1"
        # ARM XTEN_FAR_BEARING_ONLY=1 (2026-09-24): landmarks beyond max_depth_m = 0.2*f*B (where the stereo depth
        # error passes 10 %) are created WITHOUT the stereo inverse-depth prior - direction-only, depth from
        # multi-view parallax. ORB-SLAM3 does exactly this beyond ThDepth (7.1 m on rosario_06; ours 8.3 m).
        # Rosario's yaw drift accrues on STRAIGHTS at the full rate (0.21 deg/s), the signature of a depth
        # error that differs across the image; ORB-SLAM3, which does not trust far stereo depth, has none.
        # --near-band (DROP far points) made it worse: this keeps them, as directions. Default OFF.
        self.far_bearing_only = _os.environ.get("XTEN_FAR_BEARING_ONLY") == "1"
        self.n_lm_far_bearing_only = 0
        # ARM XTEN_FAR_DEPTH. The KITTI long scale comes
        # from a drive-level stereo DISPARITY OFFSET (~0.1-0.35 px, in the images, every matcher) whose relative depth
        # error b/d is largest on FAR points; ORB-SLAM3 trusts single-frame stereo only below Stereo.ThDepth.
        #   prior : landmarks beyond z_far are created WITHOUT the stereo inverse-depth prior (direction only, depth
        #           from multi-view parallax) - the far_bearing_only mechanism at ORB-SLAM3's threshold
        #   pnp   : the tracker's pose solve down-weights far points smoothly: sigma x max(1, z / z_far)
        #   both  : prior + pnp
        # z_far = XTEN_FAR_DEPTH_TH x baseline; default 60 = the Stereo.ThDepth ORB-SLAM3 ran with on our board
        # (Class B: the competitor's own rule, adopted for the trial). Unset = OFF = bit-identical.
        self.far_depth = _os.environ.get("XTEN_FAR_DEPTH", "")
        self.far_depth_z = (float(_os.environ.get("XTEN_FAR_DEPTH_TH", "60")) * self.rig.fB / self.rig.fx_rect
                            if self.far_depth in ("prior", "pnp", "both") else None)
        self.n_lm_far_depth = 0
        self.n_kp_disp = 0
        self.n_kp_no_overlap = 0
        # ★ THE COVARIANCE INFLATION IS DECLARED, NOT TUNED.
        # Plan 8.4's NEES check measured the preintegration
        # covariance as ~3x OPTIMISTIC, so the factor is
        # de-weighted by exactly that measured figure rather
        # than by a fudge chosen to make a row pass.
        self.stereo_obs = bool(stereo_obs)
        # ★ THE AGE-CONSENSUS ARM (kitti_07's crossing truck, 2026-08-27).
        # The PnP consensus may only be DEFINED by landmarks that have
        # survived beyond the matching window; younger ones EXTEND it.  See
        # `solve_pose_staged`.  Dials: staged_fallback / staged_union counts.
        self.age_consensus = bool(age_consensus)
        # ★ THE STEREO-SCALE-BIAS ARM (the V-room ~1 % under-scale).  A
        # global nuisance parameter in the global BA that lets the trajectory
        # take the IMU's metric scale while the stereo priors stay satisfied.
        # See backend/global_ba.py.  Free only when inertial is in the solve.
        self.scale_bias = bool(scale_bias)
        # ★ --gba-tight-gyro: tight gyro sigma in the OFFLINE
        # global BA only.  See global_refine's rotf loop.
        self.gba_tight_gyro = bool(gba_tight_gyro)
        # ★ --gba-robust-gyro: tight sigma AND a Huber kernel on
        # every rot factor in the OFFLINE global BA - the per-factor
        # resolution of the row-class conflict the bias-sigma saga mapped
        # (scale rows want tight factors, the rotation-drift trio rejects
        # every tight global form).  Implies the tight recompute.
        self.gba_robust_gyro = bool(gba_robust_gyro)
        # T1-h (XTEN_GBA_ROTF): the run-end global BA's rotation-factor policy, read once here
        # so a misspelt queue flag fails at construction rather than after the run
        self.gba_rotf_mode = gba_rotf_mode()
        # T1-g (XTEN_GBA_ACCEPT): validated once here for the same reason
        # (global_ba_id reads it per call; a misspelt value raises there too)
        from xclean.backend.global_ba import gba_accept_mode
        self.gba_accept_mode = gba_accept_mode()
        # P6 (XTEN_GBA_SOLVER, XTEN_GBA_ITERBUDGET): validated once here for the
        # same reason (global_ba_id reads both per call; a misspelt value raises
        # there too)
        from xclean.backend.gba_linsolve import gba_solver_mode
        from xclean.backend.global_ba import gba_iterbudget_mode
        self.gba_solver_mode = gba_solver_mode()
        self.gba_iterbudget = gba_iterbudget_mode()
        # XTEN_GBA_SCREEN: validated once here for the same reason
        from xclean.backend.global_ba import gba_screen_mode
        self.gba_screen_mode = gba_screen_mode()
        # XTEN_GBA_SCALECAL: the stereo prior's
        # disparity offset + gain self-calibrated against the inertial term in the run-end global BA; validated here
        # (a misspelt queue flag fails at construction, not after a run), and refused together with XTEN_GBA_DOF (one
        # model of the prior's common mode at a time).  XTEN_GBA_DOF is read only here and only when XTEN_GBA_SCALECAL
        # is set (its offset-only arm is not ported: backend/global_ba.py, the block above gba_dof_mode)
        from xclean.backend.global_ba import gba_scalecal_mode
        self.gba_scalecal_mode = gba_scalecal_mode()
        if self.gba_scalecal_mode is not None:
            from xclean.backend.global_ba import gba_dof_mode
            if gba_dof_mode() is not None:
                raise RuntimeError("XTEN_GBA_SCALECAL and XTEN_GBA_DOF are exclusive (one model of the stereo "
                                   "prior's common mode at a time)")
        # XTEN_GBA_DUMP: the run-end GBA entry pickled for
        # paired CPU replays; a relative / illegal / uncreatable directory raises here (backend/gba_dump.py)
        from xclean.backend.gba_dump import gba_dump_root
        self.gba_dump_root = gba_dump_root()
        # ★ --no-parallax: the
        # dial feeding the anisotropic gate is the keypoint-x SPREAD
        # (~100-200 px), not the measured FLOW (~5-20 px) the derivation
        # specifies - sigma_par comes out 5-10x too large and the along-ray
        # whitening over-deweights the radial (depth-carrying) residuals on
        # every tracked frame.  This arm disables the gate outright; a
        # measured gain anywhere justifies the true-flow fix.
        self.aniso_parallax = bool(aniso_parallax)
        # ★ --structure-welds: verified loop-weld inliers
        # become reprojection observations of the OLD landmarks from the NEW
        # keyframe in the final GBA record.  See loop/closer.py _verify and
        # add_structure_obs below.
        self.structure_welds = bool(structure_welds)
        self.n_structure_obs = 0
        self.n_quarantined = 0
        self.n_rebase = 0
        self._quar_streak = 0
        self._quarantine = False
        self._hist_trust = True
        # ★ RELOCALIZATION-GRADE WINDOW RECOVERY (the coast-deadlock fix,
        # Counters are DIALS; `_coast_streak` is the trigger
        # state.  Default ON for every row because it engages only when the
        # correspondence pool has been dead past a full window turnover -
        # a healthy row never reaches the trigger and stays bit-identical.
        self.n_reloc = 0
        self.n_reloc_attempts = 0
        self._coast_streak = 0
        self._relocalized = False
        self.n_motion_resets = 0
        self.n_staged_weak = 0
        self.n_staged_anchored = 0
        self.n_staged_fallback = 0
        self.n_staged_union = 0
        self.inertial = bool(inertial)
        self.inertial_inflate = float(inertial_inflate)
        # arm --preint-cal: pass-2 accel-block covariance calibration in the
        # offline GBA (see global_ba.py; measured basis diag/preint_calib.py)
        self.preint_cal = bool(preint_cal)
        # arm --vision-trust: per-keyframe vision down-weighting in the
        # offline GBA from the run's own gyro-vs-vision disagreement
        # (log-first receipt diag/trust_signal.py; S7 report §7)
        self.vision_trust = bool(vision_trust)
        self.kf_disagree = {}
        # arm --kp-refine: per-keypoint ZNCC disparity refinement at landmark
        # creation (measured basis diag/kp_disparity.py).  kp_refine is
        # "all" (every band - the kpr arm: kitti_05 -21%, v1_02 -9%, but
        # mh_01 +16% scale-only harm) or "far" (only landmarks beyond the
        # rig's derived max_depth_m, where the measured dense bias lives
        # (+0.94% KITTI far band) and EuRoC rooms have no landmarks - the
        # default candidate) or False.
        self.kp_refine = kp_refine if kp_refine in ("all", "far") \
            else ("all" if kp_refine else False)
        self.inertial_diag = None
        # ---- XTEN_GBA_INERT_GATE: read once here so a misspelt
        # queue value fails at construction; `inertial_gate` is the gate's provenance (None until evaluated)
        from xclean.backend.preint import inert_gate_on as _inert_gate_on
        self.inert_gate = _inert_gate_on()
        self.inertial_gate = None
        # XCLEAN_DETERMINISTIC=1: the local BA runs on its CPU (numpy) path - the GPU path accumulates with float
        # atomicAdd, whose summation order (hence the last bits, hence the trajectory) differs run to run
        self.deterministic = (_os.environ.get("XCLEAN_DETERMINISTIC", "") or "").strip() in ("1", "on")
        self.ba = (LocalBA(self.rig, iters=ba_iters, budget_ms=10.0,
                           inverse_depth=self.inverse_depth, use_gpu=not self.deterministic)
                   if backend else None)
        # ★ THE PINNED-SPAN EXPERIMENT ARM (the control loop is the default,
        # exactly as `--features N` is an arm over the feature controller).
        # It exists to answer one question with a measurement instead of an
        # assumption: DOES THE SPAN CONTROLLER'S LEVER CONTROL THE COST IT
        # REGULATES?  The v5 test series says no, and says it backwards - on
        # V1_01 span 6 cost 9.175 ms and span 4 cost 10.268 ms.
        if self.ba is not None and ba_span:
            self.ba._span = int(ba_span)
            self.ba._pinned = True        # pinned means pinned
        if self.ba is not None and self.frozen_bearing:
            self.ba.bearing_scale = 1e-3
        # ★ GROWABLE ARRAYS, NOT LISTS.  These were Python lists converted to
        # numpy on EVERY keyframe so the backend could read them - O(N) per
        # keyframe over a growing N, which is O(N^2) over a run and measured
        # as `landmark_creation` at **12.9 ms against a 5 ms bar** by the end
        # of MH_01.  A per-keyframe process whose cost grows with map size is
        # exactly what plan 5.1 forbids, and it was hiding in bookkeeping
        # rather than in the solver.
        # ★ TWO INVERSE DEPTHS, AND THEY ARE DIFFERENT OBJECTS.
        # `invd0` is what the STEREO PAIR MEASURED in the host keyframe and it
        # never changes - it is the prior, and a prior that drifts with the
        # state it constrains is not a prior.  `invd` is the ESTIMATOR STATE,
        # which the bundle adjustment refines.  The first version had one
        # column doing both jobs, which meant the record retired the raw
        # triangulation and threw the backend's refinement away.
        # `mx,my,mz` is the NORMALISED-PLANE bearing `m = (x_n, y_n, 1)`, so
        # `m / invd` is the host-frame position exactly - no range/depth
        # conversion anywhere, which is the units defect made unwritable.
        # ★ AND TWO INFORMATIONS, FOR THE SAME REASON.
        # `invsig` is on inverse DEPTH and is CONSTANT (`f*B/sigma_d`);
        # `invsig_r` is on inverse RANGE and carries the `k = range/depth`
        # factor the Euclidean prior needs.  Sharing ONE column between the
        # two parameterisations is a measured mistake, not a hypothetical:
        # it silently weakened the Euclidean ablation's stereo prior by k
        # (median 1.14, max 1.43) and turned the arm that was supposed to
        # isolate defect 13 into a two-variable experiment - visible only
        # because the LIVE frames trajectory moved 18 %, and the fix that
        # arm was testing cannot reach the frames file.
        self.lm = _Cols(("host", np.int64), ("invd", np.float64),
                        ("invd0", np.float64), ("invrange", np.float64),
                        ("invsig", np.float64), ("invsig_r", np.float64),
                        ("obs", np.int64),
                        ("mx0", np.float64), ("my0", np.float64))
        self.lm_desc = None                 # (N,32) uint8 ORB, host (xclean)
        self.lm_desc_n = 0
        # ---- xclean front-end / stereo provenance
        self._klt_prev = None               # (landmark id per previous keypoint, previous frame time)
        self.klt_pred_src = {}              # frames predicted from the gyro / the velocity model
        self.fe_join_hist = []              # track-id-join correspondences per frame
        self.n_desc_rescue = 0              # frames whose window match needed the ORB rescue
        self.n_desc_rescue_pairs = 0
        self.epi = None                     # the per-keypoint epipolar matcher (built lazily at the first keyframe)
        self.kp_disp_mode = (_os.environ.get("XCLEAN_KP_DISP", "epipolar") or "epipolar").strip().lower()
        if self.kp_disp_mode not in ("epipolar", "sgm", "hybrid"):
            raise RuntimeError("XCLEAN_KP_DISP=%r is not a mode: epipolar (default), sgm or hybrid"
                               % self.kp_disp_mode)
        self.lm_band = {}                   # landmarks created per depth band (see _create)
        self.reanchor_on = (_os.environ.get("XCLEAN_REANCHOR", "") or "").strip() in ("1", "on")
        # XC-01 (ARM XCLEAN_REACQ=1): per-track landmark BINDING + re-acquisition by projection (see _reacquire)
        self.reacq_on = (_os.environ.get("XCLEAN_REACQ", "") or "").strip() in ("1", "on")
        self.bind = {}                      # track id -> landmark id (verified inlier, carried along the KLT track)
        self.bind_fail = {}                 # track id -> consecutive frames its bound correspondence failed
        self.lm_last_obs = {}               # landmark id -> (keyframe id, u, v) of its last keyframe observation
        self.kf_pyr = {}                    # keyframe id -> image pyramid (GPU), bounded by the estimator window
        self.reacq_stats = {"frames": 0, "candidates": 0, "accepted": 0, "bound_existing": 0, "new_points": 0,
                            "coast_tries": 0, "coast_rescued": 0}
        self.disp_sigma_mode = (_os.environ.get("XCLEAN_DISP_SIGMA", "declared") or "declared").strip().lower()
        if self.disp_sigma_mode not in ("declared", "measured"):
            raise RuntimeError("XCLEAN_DISP_SIGMA=%r: declared (default) or measured" % self.disp_sigma_mode)
        self.n_lm_measured_sigma = 0
        self.n_lm_bearing_only = 0
        self.sigma_samples = []
        self.n_kp_quad_invalid = 0          # keypoints whose bilinear SGM quad touched an invalid pixel
        self.n_kp_sgm_valid = 0
        self.n_kp_epi_ok = 0
        self.n_localmap = []                # extra correspondences per frame
        self.ba_decisions = []
        # the FULL observation record.  The estimator working set is bounded
        # by the window (plan 5.1); this is the RECORD, bounded by the
        # trajectory - the same two-objects distinction that makes landmark
        # retirement lossless (plan 1b).  ~130 k rows on a 650-keyframe run.
        self.obs_kf = []
        self.obs_lm = []
        self.obs_uv = []
        self.obs_d = []           # measured disparity per observation, px
        self.imu = imu
        self.localmap = bool(localmap)
        self.far_band = bool(far_band)
        self.n_gyro_factors = 0
        self.loop = loop
        self.pgo_every = int(pgo_every)
        self.loop_decisions = []
        self.n_applied = 0
        self.T_wc = np.eye(4)
        self.prev_T = None
        self.vel = np.eye(4)                # constant-velocity model
        # TEST ARM XCLEAN_DT_CV=1 (default OFF = bit-identical): the constant-velocity model is constant in TIME -
        # the last measured relative motion (spanning _vel_dt seconds) is scaled on SE(3) to the actual interval
        # since the last pose.  Off: the per-FRAME increment is replayed whatever the interval (a 0.49 s dataset
        # gap on madmax_E1 frame 8552 is predicted as one 70 ms step).
        self.dt_cv = (_os.environ.get("XCLEAN_DT_CV", "") or "").strip() in ("1", "on")
        self._t_last = None
        self._vel_dt = None
        self._vel_body = None
        self.n_dt_scaled = 0
        self.n_frames = 0
        self.n_lost = 0
        self.decisions = []
        self.inlier_hist = []
        self.parallax_hist = []
        self.depth_frames = 0
        self._pending_desc = None
        self._verified_sigma_px = FEATURE_SIGMA_PX
        self._kf_pose_snapshot = {}
        # ---- LOG (tier-1 programme, 2026-09-29): provenance, default-on.
        # Nothing below feeds an estimate: counters, the creating frame's
        # inliers and the median depth per keyframe (T1-b reads the
        # reference keyframe's inliers; T1-e reads inliers + z_med per
        # keyframe for the pose graph), and the per-frame budget-at-cap
        # record that decides where T1-b's gate is a structural no-op.
        self.kf_evidence = {}               # kid -> {inliers, z_med}
        self.kf_trigger_counts = {}         # accepted keyframe decisions
        self.kf_decision_counts = {}        # every keyframe decision
        self.n_frames_at_cap = 0
        self.n_kf_evidence_err = 0
        self._frame_inliers = None
        # ---- T1-b (XTEN_KF_HEALTH=lookahead, see health_trigger above): the
        # mode is read once per run; with it off only the counters below run.
        self.kf_health_mode = kf_health_mode()
        self.kf_health_evals = {"old": 0, "lookahead": 0, "fallback": 0}
        self.kf_health_fires = {"old": 0, "lookahead": 0, "fallback": 0}
        self._frames_since_kf = 0           # m: +1 per keyframe decision,
                                            # 0 at every keyframe creation
        # ---- FRONT GATE (see FRONT_GATE_ENV above): the mode is read once per run (a misspelt, refuted or unported
        # queue flag fails at construction); with it off nothing below executes beyond `if self._fg_active`.
        front_vel_refuse()
        self.front_gate = front_gate_mode() == "recount"
        self.front_recount = self.front_gate
        self._fg_active = self.front_gate
        self.pnp_verify = pnp_verify_mode() == "all"       # XTEN_PNP_VERIFY (track/pose.py), validated here
        self._fg_hist = []                  # (<=INNOV_SPAN, 6) innovation history
        self._fg_pred = None                # this frame's prediction, None when it is not a measured one
        self._fg_pre = None                 # the ungated (P, uv, src, kpi) when the gate removed something
        self.fg_stats = {"frames": 0, "gated": 0, "warmup": 0, "ungated_floor": 0, "widened": 0,
                         "max_scale": 1.0, "n_in": 0, "n_kept": 0, "fallback_ungated": 0,
                         "fallback_rescued": 0}
        self.fg_log = []                    # per frame (index, n_in, n_kept, scale, None) - replay analysis
        # ---- TRACK PACKAGE (see DECISION_LOG_ENV above): the two modes are read once per run (a misspelt value
        # fails at construction, as does a killed part's flag); with both off nothing below executes beyond the
        # `if self.<flag>` tests.
        dropped_track_flags_unset()
        self.decision_log = decision_log_on()
        self.coast_feed = coast_feed_on()
        self._dl = None                     # F0: this frame's record (None when the log is off)
        self._dl_fh = None
        self.dlog_mem = [] if self.decision_log else None
        self.dlog_stats = {"frames": 0, "bytes": 0, "path": None, "cost_ms": []}
        self.n_coast_feed = 0               # F1: coasted frames fed to the budget controller

    @property
    def lm_xyz(self):
        return self.store.xyz

    # ------------------------------------------------------------------ ---
    def step(self, fr):
        i = fr.index
        if self.decision_log:                        # F0: this frame's record
            self._dl = {"i": int(i), "t": float(fr.t), "k0": len(self.decisions),
                        "trust_in": bool(self._hist_trust), "q0": int(self.n_quarantined),
                        "rb0": int(self.n_rebase)}
        with self.led.stage("rectify", frame=i):
            left, right = self.ds.images(i)
        with self.led.stage("extract", frame=i):
            # xclean: KLT carries the previous frame's points into this one; the initial flow is PREDICTED (gyro
            # rotation where an IMU exists, else the constant-velocity model; landmarks projected, the rest at the
            # map's median depth) - see _klt_prediction.  The front end retries any failed track from zero motion.
            pred_uv = self._klt_prediction(fr) if self.kfs else None
            feats = self.fe.extract(left, pred_uv=pred_uv)
            kp = feats["keypoints"]

        if not self.kfs:
            self._bootstrap(fr, left, right, feats, kp)
            if self._dl is not None:
                self._dl_emit(i, "init")
            if self.dt_cv:
                self._t_last = float(fr.t)
            return "init"

        # ---- match against the local keyframe window --------------------
        with self.led.stage("match", frame=i):
            P, uv, src, kpi = self._collect(feats, kp)
        if self._dl is not None:
            self._dl["n_win"] = int(len(P))

        T_pred = self._predict(self.T_wc, fr.t) if self.prev_T is not None else self.T_wc
        if not np.isfinite(T_pred).all() \
                or np.abs(T_pred[:3, 3] - self.T_wc[:3, 3]).max() \
                > self.rig.far_bearing_m:
            # ★ REPRESENTABILITY GUARD (defects "overflow" AND 25).  A long
            # coast once compounded the motion model into float overflow
            # (kitti_07, NaN trajectory), and a finite-only check missed the
            # SAME runaway when it stayed under float max: v2_03's landing
            # tumble grew the coast super-exponentially (8 m -> 1e154 in six
            # frames, `isfinite` never fired, frames ATE = inf; the benchmark's
            # own r1r2 draws 1/3 carried NaN frame rows the same way).  The
            # bound is Class B: a SINGLE-FRAME motion larger than the deepest
            # representable landmark (far_bearing_m) is not a motion the
            # vision system could have measured - real per-frame motion sits
            # 200-6000x below it, so healthy rows are bit-identical.
            T_pred = self.T_wc
            self.n_motion_resets += 1
        self._quarantine = False
        if self._fg_active:
            P, uv, src, kpi = self._fg_gate(i, P, uv, src, kpi, T_pred)
        with self.led.stage("track", frame=i):
            par = float(np.median(self.parallax_hist[-20:])) \
                if self.parallax_hist else None
            mz = self._median_depth()
            far_scale = self._far_point_scale(P, T_pred)      # None unless XTEN_FAR_DEPTH=pnp/both
            T_wc, mask, dec = solve_pose(P, uv, self.rig, T_init=T_pred,
                                         parallax_px=par, median_z=mz,
                                         point_scale=far_scale)
            # ★ THE GUESS-FREE RETRY.  solve_pose with a guess runs RANSAC
            # over SOLVEPNP_ITERATIVE, which cannot converge from a bad
            # init - and after a few coasted frames the constant-velocity
            # prediction IS a bad init, so one failure became a permanent
            # coast (measured: 458 frames, the rest of a kitti_07 run,
            # while the map was intact and the view had returned).  PnP
            # against landmarks is ABSOLUTE: a guess-free EPnP attempt
            # recovers from the map no matter how far the coast drifted.
            if T_wc is None and len(P) >= MIN_PNP_INLIERS:
                T_wc, mask, dec2 = solve_pose(P, uv, self.rig, T_init=None,
                                              parallax_px=par, median_z=mz,
                                              point_scale=far_scale)
                if T_wc is not None:
                    dec = dec2
            if T_wc is None and self._fg_pre is not None:
                # FRONT GATE: the gated set gave no pose - solve the UNGATED set exactly as the baseline would
                self.decisions.append(dec)
                P, uv, src, kpi = self._fg_pre
                far_scale = self._far_point_scale(P, T_pred)
                T_wc, mask, dec = solve_pose(P, uv, self.rig, T_init=T_pred,
                                             parallax_px=par, median_z=mz,
                                             point_scale=far_scale)
                if T_wc is None and len(P) >= MIN_PNP_INLIERS:
                    T_wc, mask, dec2 = solve_pose(P, uv, self.rig, T_init=None,
                                                  parallax_px=par, median_z=mz,
                                                  point_scale=far_scale)
                    if T_wc is not None:
                        dec = dec2
                self.fg_stats["fallback_ungated"] += 1
                if T_wc is not None:
                    self.fg_stats["fallback_rescued"] += 1
            elif self.front_recount and T_wc is not None and self._fg_pre is not None:
                # FRONT GATE (recount): the window chose the pose; EVERY correspondence is now re-counted under it
                # (ORB-SLAM3's TrackLocalMap re-projects with the refined pose: a true match the prediction window
                # missed is an inlier again, an aliased one stays out because it disagrees with THIS pose)
                P, uv, src, kpi = self._fg_pre
                mask = self._fg_recount(T_wc, P, uv)
        self.decisions.append(dec)
        state = "tracking"
        self._relocalized = False
        assoc = np.full(len(kp), -1, np.int64)
        if T_wc is None:
            # no pose: coast on the motion model and say so.  A coasted frame
            # is a LOGGED state, never a silent gap.
            T_wc = T_pred
            state = "coasting"
            self._coast_streak += 1
            # ★ RELOCALIZATION-GRADE WINDOW RECOVERY.  COASTING WAS ABSORBING:
            # coasted frames make no keyframes ("not from a coasted pose"),
            # so the 3-keyframe matching window goes permanently stale and
            # can never re-match once the scene moves on - measured as a
            # 458-frame terminal coast (42 % of a kitti_07 run) after the
            # event-quarantine thinned the window, while the mature map still
            # claimed the view.  The guess-free PnP retry cannot help there
            # because MATCHING starves, not PnP.  Once the pool has been dead
            # for more than one full window turnover at the run's own
            # keyframe cadence, the window is stale BY CONSTRUCTION (every
            # keyframe in it predates the whole streak) - so match beyond it:
            # retrieval over every keyframe the loop store holds, then
            # absolute guess-free PnP at the reloc floor.  The loop machinery
            # is the donor; nothing new is computed per healthy frame, and a
            # row that never coasts past a turnover is bit-identical.
            if self.loop is not None and self.loop.D is not None \
                    and self._coast_streak > self._window_turnover_frames():
                with self.led.stage("reloc", frame=i):
                    T_r, m_r, src_r, kpi_r, dec_r = \
                        self._relocalize(fr, left, feats, kp)
                self.n_reloc_attempts += 1
                self.decisions.append(dec_r)
                if T_r is not None:
                    # a reloc pose is history-verified BY CONSTRUCTION: it
                    # stands on >= MIN_RELOC_INLIERS mature landmarks, a
                    # stronger anchoring than the referee's own bar - so it
                    # restores history trust and clears both streaks.  The
                    # local-map stage and the referee are skipped this frame
                    # (their evidence pools are the stale window's); the
                    # forced keyframe below re-arms the window and the next
                    # frame runs the normal pipeline.
                    T_wc = T_r
                    state = "relocalized"
                    self._relocalized = True
                    self.n_reloc += 1
                    self._coast_streak = 0
                    self._quar_streak = 0
                    self._hist_trust = True
                    self.fe.report_inliers(int(m_r.sum()))
                    self.inlier_hist.append(int(m_r.sum()))
                    sel_r = np.nonzero(m_r)[0]
                    assoc[kpi_r[sel_r]] = src_r[sel_r]
                    for k in sel_r:
                        self.lm.col("obs")[src_r[k]] += 1
                    mask = m_r
            if state == "coasting" and self.reacq_on:
                # XC-01(b): re-acquisition from the PREDICTED pose on every coasted frame, the search radius inflated
                # by the coast length (the prediction's error grows with it)
                self.reacq_stats["coast_tries"] += 1
                T_r, kp = self._reacquire(i, left, T_pred, feats, assoc, inflate=1.0 + self._coast_streak,
                                          solve_only_new=False)
                if self._reacq_solved:
                    T_wc = T_r
                    assoc = self._reacq_assoc
                    kpi = np.nonzero(assoc >= 0)[0]
                    src = assoc[kpi]
                    P, uv = self.lm_xyz[src], kp[kpi]
                    mask = np.ones(len(kpi), bool)
                    state = "tracking"
                    self._coast_streak = 0
                    self.reacq_stats["coast_rescued"] += 1
                    self.fe.report_inliers(int(mask.sum()))
                    self.inlier_hist.append(int(mask.sum()))
            if state == "coasting":
                self.n_lost += 1
                if self.coast_feed:
                    # F1 (XTEN_COAST_FEED): a coasted frame measured ZERO usable inliers - the controller's own
                    # signal, which it otherwise never sees during a coast (the budget froze for 367-619 frames
                    # on every lost run of our evaluation, our study of lost runs, section 5)
                    self.fe.report_inliers(0)
                    self.n_coast_feed += 1
        else:
            self._coast_streak = 0
            self.fe.report_inliers(int(mask.sum()))
            self.inlier_hist.append(int(mask.sum()))
            sel = np.nonzero(mask)[0]
            if len(sel) >= 8:
                pr, _pc = project(se3_inv(T_wc), P[sel], self.rig.fx_rect,
                                  self.rig.fx_rect, self.rig.cx, self.rig.cy)
                self._verified_sigma_px = float(np.median(
                    np.linalg.norm(pr - uv[sel], axis=1))) or FEATURE_SIGMA_PX
            # ★ THE ASSOCIATION.  Without this a new keyframe creates entirely
            # NEW landmarks and no landmark is ever shared, so the covisibility
            # graph is empty and local BA has nothing to optimise.  Only PnP
            # INLIERS are associated: an outlier match that became a landmark
            # observation would be a wrong constraint with full weight.
            assoc[kpi[sel]] = src[sel]
            for k in sel:
                self.lm.col("obs")[src[k]] += 1

        # ---- LOCAL-MAP TRACKING (plan 10, P2) -------------------------
        # The window match sees only the 3 keyframes whose descriptors are
        # still resident, so a landmark is observed 3-4 times and then never
        # again: track lengths that short give bundle adjustment almost
        # nothing to work with (measured: local BA improved MH_01 by only
        # 10 %).  Projecting the WHOLE estimator window into the frame and
        # matching by descriptor inside a geometric gate is what turns a
        # 4-observation landmark into a 15-observation one.
        lm2_mask = lm2_ids = None
        if state == "tracking" and self.localmap:
            with self.led.stage("track", frame=i):
                T_wc, mask2, extra, kpi2, src2 = self._track_local_map(
                    T_wc, feats, kp, src, P, uv, mask, kpi)
            if extra:
                self.n_localmap.append(extra)
                sel2 = np.nonzero(mask2)[0]
                assoc[kpi2[sel2]] = src2[sel2]
                for k in sel2:
                    self.lm.col("obs")[src2[k]] += 1
            if mask2 is not None:
                lm2_mask, lm2_ids = mask2, src2
        if self.reacq_on and state == "tracking":
            T_wc, kp = self._reacquire(i, left, T_wc, feats, assoc, inflate=1.0)
            assoc = self._reacq_assoc

        # ---- AGE-CONSENSUS REFEREE, judged AFTER local-map tracking -------
        # ★ v8, and the two previous verdicts explain its shape.  v6 judged
        # anchoring on the 3-keyframe MATCHING-WINDOW pool alone; during fast
        # rotation the mature entries churn out of that pool by inheritance
        # while the LOCAL-MAP stage geometrically re-acquires them every
        # frame - so ~350 healthy V1_01 frames were treated as un-anchored
        # and the control moved 0.0369 -> 0.0401 (n=3, falsifier fired).
        # The confidence-floor variant then broke kitti_07 the other way
        # (1.54/0.82/1.40).  So the referee now reads the mature evidence of
        # the FULL baseline solve (window + local map):
        #   >= MIN_PNP_INLIERS  -> anchored, the baseline result stands;
        #   >= HYPOTHESIS floor -> the pose TOUCHES history: a mover-
        #      consistent pose cannot geometrically keep mature static
        #      inliers, so it is accepted;
        #   below the floor     -> historyless: the hypothesis chain runs,
        #      and QUARANTINE applies where the mature map claims the view.
        # ★ SENSOR-CONDITIONAL, NOT PER-DATASET (v13, final form).  Twelve
        # variants established a hard trade: every form robust against the
        # kitti_07 capture (v5, v10: four seeds each at ~0.5) costs the
        # IMU-equipped V1_01 control +8..12 % on its benign fast-rotation
        # bursts, and every softening that spares those bursts reopens the
        # capture (v9 seed 2: 2.54; v11: misses the flip; v12: pool
        # collapse, 457 coasts).  The referee exists to give a vision-only
        # rig the independent cross-check an IMU-equipped rig already has -
        # so it engages exactly where that substitute is needed: rigs
        # WITHOUT an IMU.  Same class of conditionality as "gyro factors
        # exist only when an IMU exists"; IMU rows are bit-for-bit baseline.
        if self.age_consensus and self.imu is None \
                and state == "tracking" and len(src):
            host = self.lm.col("host")
            cur = len(self.all_kf) - 1
            span = self.max_kf_map

            def _mature_inliers(ids_arr, m):
                if ids_arr is None or m is None:
                    return 0
                sel_ = np.nonzero(m)[0]
                if not len(sel_):
                    return 0
                ids_ = np.asarray(ids_arr)[sel_]
                ids_ = ids_[(ids_ >= 0) & (ids_ < len(host))]
                return int(((cur - host[ids_]) > span).sum())

            if lm2_ids is not None:
                mat_ev = _mature_inliers(lm2_ids, lm2_mask)
            else:
                mat_ev = _mature_inliers(src, mask)
            if self._dl is not None:
                self._dl["ref"] = {"mat_ev": int(mat_ev)}
            # ★ THE OCCLUDER TEST (v11).  v10's chain still engaged ~170
            # times per V1_01 run on benign fast-rotation bursts and cost
            # the control +12 %.  The physical discriminator: AN OCCLUDER IS
            # CLOSER THAN THE STRUCTURE IT OCCLUDES, BY CONSTRUCTION.  At
            # the kitti_07 flip the consensus median depth collapsed 23 m ->
            # 7 m - under the 10th percentile of what the mature map
            # predicts visible in that view - while V1_01's historyless
            # bursts stay AT scene depth.  So the chain engages only when
            # the historyless consensus is DEPTH-ANOMALOUS: median inlier
            # depth below the mature prediction's p10 (a population
            # quantile of the map's own geometry, not a typed constant).
            # engage on EVERY historyless frame (v10's rule, restored): the
            # jurisdiction test lives at quarantine time under the FINAL
            # pose - gating the CHAIN on the plain pose's view let a
            # captured phantom evade it by facing away from the map (seed 2
            # cascaded on exactly that)
            if mat_ev >= MIN_PNP_INLIERS:
                self.n_staged_anchored += 1
                self._hist_trust = True
            elif mat_ev >= 6:
                self.n_staged_weak += 1
                self._hist_trust = True
            elif T_wc is None:
                pass
            else:
                from xclean.track.pose import solve_pose_staged
                age = cur - host[src]
                T_c, mask_c, dec_c = solve_pose_staged(
                    P, uv, age, span, self.rig, T_init=T_pred,
                    parallax_px=par, median_z=mz,
                    plain=(T_wc, mask, dec), T_prev=self.T_wc)
                self.decisions.append(dec_c)
                s = dec_c.dials.get("staged", "")
                if self._dl is not None:
                    self._dl["ref"]["staged"] = str(s)
                if s.startswith("fallback"):
                    self.n_staged_fallback += 1
                else:
                    self.n_staged_union += 1
                if T_c is not None and s in ("union", "motion-union",
                                             "motion-union-decel",
                                             "motion-union-stop",
                                             "reject-arbitration"):
                    # the chain overrode the baseline pose: adopt it and
                    # re-derive the associations from ITS inlier set
                    T_wc, mask = T_c, mask_c
                    assoc[:] = -1
                    sel_c = np.nonzero(mask_c)[0]
                    assoc[kpi[sel_c]] = src[sel_c]
                # ★ QUARANTINE - HISTORY'S JURISDICTION.  A pose that is
                # historyless in a view the mature map itself claims (>= the
                # conditioning floor of its landmarks predicted visible) may
                # not EXTEND the map.  Occlusion by a mover freezes the map
                # and the mover never becomes landmarks; a genuine scene
                # change (the map claims nothing here) rebases.
                # CORROBORATION LIFTS IT - BUT ONLY WITH PROVENANCE (v10).
                # v9 lifted quarantine on "motion-agree" unconditionally and
                # kitti_07 seed 2 blew up to 2.54: once ONE truck step
                # slips, the prediction, the plain consensus and the motion
                # hypothesis all agree WITH EACH OTHER - corroboration is
                # exploitable by exactly the self-perpetuation it was meant
                # to break.  So motion-based acceptance is honoured only
                # while `_hist_trust` holds (the prediction descends from a
                # history-verified frame), and a motion-verified frame does
                # NOT refresh that trust: the SECOND consecutive
                # motion-only frame quarantines.  v1_01's benign historyless
                # bursts are 1-3 frames interleaved with anchored ones, so
                # they ride on trust; the truck's cascade cannot.
                # jurisdiction is RE-CHECKED under the FINAL pose (v10's
                # form, regressed in v13 and restored): folding it into the
                # engage test alone left long quarantine streaks with no
                # release valve, and two kitti_07 seeds starved the matching
                # window into a 368/458-frame coast cascade
                if s in ("union", "union-agree"):
                    self._hist_trust = True
                else:
                    trusted_motion = s in ("motion-agree", "motion-union",
                                           "motion-union-decel",
                                           "motion-union-stop",
                                           "reject-arbitration") \
                        and self._hist_trust
                    if not trusted_motion and T_wc is not None \
                            and self._mature_visible(T_wc)[0] \
                            >= MIN_PNP_INLIERS:
                        # ★ THE ESCAPE VALVE.  A quarantine streak may not
                        # outlive TWO full matching-window turnovers at the
                        # run's own keyframe cadence: occlusion by a
                        # crossing mover is transient by construction, and
                        # a longer streak means the freeze has failed its
                        # purpose and is now starving the window (measured:
                        # an 18-frame streak at kitti_07's pull-away from
                        # the stop turned into a permanent 348-frame coast).
                        gaps = np.diff([r[1] for r in self.all_kf[-12:]])
                        cadence = float(np.median(gaps)) if len(gaps) else 1.0
                        cap = max(2.0 * self.max_kf_map * cadence,
                                  2.0 * self.max_kf_map)
                        if self._quar_streak < cap:
                            self._quarantine = True
                            self.n_quarantined += 1
                        else:
                            self.n_rebase += 1
                    self._hist_trust = False
        if self._dl is not None and "staged" in self._dl.get("ref", {}) and T_wc is not None:
            # F0: the mature-visible count under the final pose (pure read; the referee itself reads it only
            # off the trusted-motion route)
            import time as _time
            _c0 = _time.perf_counter()
            self._dl["ref"]["mature_visible"] = int(self._mature_visible(T_wc)[0])
            self._dl["_cost_ms"] = (_time.perf_counter() - _c0) * 1e3
        if self._quarantine:
            state = "quarantined"
            self._quar_streak += 1
        elif state == "tracking":
            self._quar_streak = 0

        self.prev_T = self.T_wc
        if self.dt_cv:
            self._vel_dt = (float(fr.t) - self._t_last) if self._t_last is not None else None
            self._t_last = float(fr.t)
            self._vel_body = se3_inv(self.prev_T) @ T_wc
        if self._relocalized:
            # the pose jumped from a drifted coast back onto the map: the
            # transform across that jump is not a velocity.  Unknown motion
            # is identity, exactly as at bootstrap.
            self.vel = np.eye(4)
        else:
            self.vel = T_wc @ se3_inv(self.prev_T)
            if not np.isfinite(self.vel).all() \
                    or np.abs(self.vel[:3, 3]).max() > self.rig.far_bearing_m:
                # the same representability guard as T_pred (defect 25): a
                # non-finite OR unrepresentably large velocity model poisons
                # every later frame - the v2_03 runaway compounded through
                # exactly this update (vel = T @ inv(prev) accumulates error
                # proportional to |t| once the pose is huge)
                self.vel = np.eye(4)
                self.n_motion_resets += 1
            elif self._coast_streak > self._window_turnover_frames():
                # ★ THE VELOCITY MODEL HAS A CREDIBILITY HORIZON (defect 25,
                # part 2).  Constant velocity extrapolates the last MEASURED
                # motion; past one window turnover of consecutive coasts
                # every measurement supporting it has expired - the same
                # horizon at which the matching window is stale by
                # construction, and the same derivation as the reloc
                # trigger.  Walking it further is manufacture, measured on
                # v2_03's landing: 150 coasted frames x a perfectly
                # reasonable last velocity = 1.8-1.9 m of frames-file error
                # the keyframe trajectory never sees.  Past the horizon the
                # coast HOLDS POSE.  Only >turnover coast streaks reach
                # this branch, so healthy rows are bit-identical.
                self.vel = np.eye(4)
        if self.dt_cv and np.array_equal(self.vel, np.eye(4)):
            self._vel_body = None                        # any velocity reset above: unknown motion is identity
        if self._fg_active:
            self._fg_update(T_wc, state)
        self.T_wc = T_wc
        self.n_frames += 1
        ev = self._log_frame_evidence(mask)          # LOG: inliers, budget
        with self.led.stage("map_kit_append", frame=i):
            if self.kit:
                # the anchor is the CURRENT keyframe and its pose AS IT
                # STANDS NOW; the relative transform that comes out is
                # invariant to every correction that lands later
                anc = ((self.all_kf[-1][0], self.all_kf[-1][3])
                       if self.all_kf else None)
                self.kit.frame(i, fr.t, T_wc, state, anchor=anc,
                               evidence=ev)

        d = self._need_keyframe(mask, state)
        self.decisions.append(d)
        self._log_kf_decision(d)                     # LOG: kf_triggers
        if d.accepted:
            self._make_keyframe(fr, left, right, feats, kp, assoc=assoc)
        if self.reacq_on:
            self._update_bindings(feats, assoc, state)
        # xclean: what the next frame's KLT prediction reads - the landmark of each of THIS frame's keypoints (the
        # new keyframe's lm_id when one was made here: it also carries the landmarks just created) and the time
        made = d.accepted and self.kfs and self.kfs[-1].frame == i
        self._klt_prev = (np.array(self.kfs[-1].lm_id if made else assoc, np.int64), float(fr.t))
        if self._dl is not None:
            self._dl_emit(i, state, mask)
        return state

    # ---- FRONT GATE (FRONT_GATE_ENV above; the velocity guard is not ported) ------------------------------------
    def _fg_gate(self, i, P, uv, src, kpi, T_pred):
        """Apply the innovation window to this frame's correspondences (or leave them alone).  Always records the
        prediction for _fg_update; -> (P, uv, src, kpi)."""
        self._fg_pre = None
        self._fg_pred = T_pred if self.prev_T is not None else None
        if not self.front_gate or self._fg_pred is None:
            return P, uv, src, kpi
        st = self.fg_stats
        st["frames"] += 1
        if len(self._fg_hist) < INNOV_SPAN or len(P) == 0:
            st["warmup"] += 1
            return P, uv, src, kpi
        var = innov_variance(self._fg_hist)
        keep, s_w, _n_ok = front_gate_keep(P, uv, T_pred, var, self.rig.fx_rect, self.rig.cx, self.rig.cy,
                                           KF_INLIER_HEADROOM * MIN_PNP_INLIERS)
        if keep is None:
            st["ungated_floor"] += 1
            self.fg_log.append((i, len(P), len(P), s_w, None))
            return P, uv, src, kpi
        st["gated"] += 1
        st["n_in"] += len(P)
        st["n_kept"] += int(keep.sum())
        if s_w > 1.0:
            st["widened"] += 1
            st["max_scale"] = max(st["max_scale"], s_w)
        self.fg_log.append((i, len(P), int(keep.sum()), s_w, None))
        if keep.all():
            return P, uv, src, kpi
        self._fg_pre = (P, uv, src, kpi)
        return P[keep], uv[keep], src[keep], kpi[keep]

    def _fg_recount(self, T_wc, P, uv):
        """Inlier mask of ALL correspondences under the final pose, with the plain solver's own reprojection gate."""
        pr, pc = project(se3_inv(T_wc), P, self.rig.fx_rect, self.rig.fx_rect, self.rig.cx, self.rig.cy)
        thr = max(3.0 * FEATURE_SIGMA_PX, 2.0 * FEATURE_SIGMA_PX)
        ok = (pc[:, 2] > 1e-3) & (np.linalg.norm(pr - uv, axis=1) <= thr)
        self.fg_stats["recounted"] = self.fg_stats.get("recounted", 0) + 1
        self.fg_stats["recount_added"] = self.fg_stats.get("recount_added", 0) + int(ok.sum())
        return ok

    def _fg_update(self, T_wc, state):
        """Enter this frame's innovation into the history (a tracked frame with a measured prediction only)."""
        if self._fg_pred is None or state != "tracking":
            return
        self._fg_hist.append(innov_error(self._fg_pred, T_wc))
        if len(self._fg_hist) > INNOV_SPAN:
            self._fg_hist.pop(0)

    def front_gate_summary(self):
        st = self.fg_stats
        return {"gate": self.front_gate, "recount": self.front_recount, "span": INNOV_SPAN,
                "recounted": int(st.get("recounted", 0)), "recount_inliers": int(st.get("recount_added", 0)),
                "frames": int(st["frames"]), "gated": int(st["gated"]), "warmup": int(st["warmup"]),
                "ungated_floor": int(st["ungated_floor"]), "widened": int(st["widened"]),
                "max_scale": float(st["max_scale"]),
                "kept_frac": (st["n_kept"] / st["n_in"]) if st["n_in"] else None,
                "fallback_ungated": int(st["fallback_ungated"]),
                "fallback_rescued": int(st["fallback_rescued"]),
                "pre_registered": list(FRONT_GATE_CHOICES)}

    # ---- TRACK PACKAGE (DECISION_LOG_ENV above) -----------------------------------------------------------------
    def _dl_emit(self, i, state, mask=None):
        """F0 (XTEN_DECISION_LOG): write this frame's record (one JSON line).  Reads only: nothing here touches an
        estimate, the RNG or a value the tracker reads later.  Its own cost is measured (dlog_stats.cost_ms)."""
        import json
        import time as _time
        c0 = _time.perf_counter()
        r, self._dl = self._dl, None
        k0, q0, rb0 = r.pop("k0"), r.pop("q0"), r.pop("rb0")
        extra_ms = r.pop("_cost_ms", 0.0)
        decs = self.decisions[k0:]
        r["state"] = state
        r["inliers"] = None if mask is None else int(np.asarray(mask).sum())
        bud = getattr(self.fe, "budget", None)
        r["budget"] = None if bud is None else int(bud)
        r["coast_streak"] = int(self._coast_streak)
        r["quar_streak"] = int(self._quar_streak)
        r["trust_out"] = bool(self._hist_trust)
        r["quarantined"] = bool(self.n_quarantined > q0)
        r["rebase"] = bool(self.n_rebase > rb0)
        r["window"] = [int(k.frame) for k in self.kfs]
        ref = r.get("ref")
        if ref is not None:
            # the referee's route (our study of lost runs, 2.5): 1 anchored / weak, 2 union, 3 trusted motion, 4 exemption,
            # or the quarantine / its escape valve
            me, lab = ref["mat_ev"], ref.get("staged")
            if me >= MIN_PNP_INLIERS:
                route = "anchored"
            elif me >= 6:
                route = "weak"
            elif lab is None:
                route = "no-pose"
            elif lab in ("union", "union-agree"):
                route = "union"
            elif r["quarantined"]:
                route = "quarantine"
            elif r["rebase"]:
                route = "rebase"
            elif lab in ("motion-agree", "motion-union", "motion-union-decel", "motion-union-stop",
                         "reject-arbitration") and r["trust_in"]:
                route = "trusted-motion"
            else:
                route = "exemption"
            ref["route"] = route
        r["decs"] = [decision_record(d) for d in decs]
        if self.coast_feed:
            r["coast_feed"] = int(self.n_coast_feed)
        line = json.dumps(r, separators=(",", ":"), default=str)
        if self.kit is not None and getattr(self.kit, "enabled", False) and getattr(self.kit, "dir", None):
            if self._dl_fh is None:
                import os as _os
                self.dlog_stats["path"] = _os.path.join(self.kit.dir, "decisions.jsonl")
                self._dl_fh = open(self.dlog_stats["path"], "w", encoding="utf-8", buffering=1)
            self._dl_fh.write(line + "\n")
        else:
            self.dlog_mem.append(line)
        self.dlog_stats["frames"] += 1
        self.dlog_stats["bytes"] += len(line) + 1
        self.dlog_stats["cost_ms"].append((_time.perf_counter() - c0) * 1e3 + extra_ms)

    def _dl_close(self):
        if self._dl_fh is not None:
            self._dl_fh.close()
            self._dl_fh = None

    def track_package_summary(self):
        """Provenance of the two TRACK PACKAGE parts (only the parts whose flag is on)."""
        out = {}
        if self.decision_log:
            ds_ = self.dlog_stats
            out["decision_log"] = {"frames": int(ds_["frames"]), "bytes": int(ds_["bytes"]), "path": ds_["path"],
                                   "cost": cost_stats(ds_["cost_ms"])}
        if self.coast_feed:
            out["coast_feed"] = {"fed": int(self.n_coast_feed)}
        return out

    # ------------------------------------------------------------------ ---
    def _landmark_yield(self, left, right, kp):
        """(keypoints that WOULD become landmarks, keypoints) at this frame - a dry run of _create's measurement and
        band (no state touched)."""
        if len(kp) == 0:
            return 0, 0
        disp = self.depth.compute(left, right)
        d, on_edge = self._kp_disparity(kp, disp, left, right, count=False)
        z = np.where(d > 1e-6, self.rig.fB / np.maximum(d, 1e-6), np.inf)
        hi = self.rig.far_bearing_m if self.far_band else self.rig.max_depth_m
        good = (d > 0) & (z >= self.rig.near_clip_m) & (z <= hi) & (~on_edge)
        return int(good.sum()), int(len(kp))

    def _bootstrap(self, fr, left, right, feats, kp):
        self.T_wc = np.eye(4)
        self._frame_inliers = None                   # LOG: no PnP yet
        # ★ xclean: THE FIRST KEYFRAME MUST BE ABLE TO CARRY THE NEXT SOLVE.  A coasted frame makes no keyframe, so a
        # bootstrap map with fewer landmarks than a PnP needs is terminal (measured on mh_04: 102 keypoints spread on
        # the grid, 63 % of the first view nearer than the 128 px disparity range, 21 landmarks, 299 of 300 frames
        # coasted).  The budget's own conditioning floor is MIN / (match_rate x inlier_rate) at COLD-START rates; here
        # the rate that matters - keypoints that become landmarks - is MEASURED on the bootstrap frame itself and
        # the budget is raised to KF_INLIER_HEADROOM x MIN_PNP_INLIERS / yield (the count the front gate and the
        # health trigger already call "enough"), then the frame is re-extracted.  The budget only grows and is
        # capped, so the loop ends; a frame that cannot reach it even at the cap bootstraps as before.
        import math as _m
        need = KF_INLIER_HEADROOM * MIN_PNP_INLIERS
        self.boot_log = []
        while True:
            n_ok, n_kp = self._landmark_yield(left, right, kp)
            self.boot_log.append([int(self.fe.budget), n_kp, n_ok])
            if n_ok >= need or self.fe.budget >= self.fe.cap or n_kp == 0:
                break
            # at least the controller's own growth step (geometric, so the loop is O(log(cap / budget)) even where
            # the yield saturates - a view whose depth lies mostly beyond the disparity range)
            y = max(n_ok, 1) / float(n_kp)
            self.fe.budget = int(min(self.fe.cap, max(_m.ceil(self.fe.budget * self.fe.GROW_FACTOR),
                                                      _m.ceil(need / y))))
            self.fe.reset()
            feats = self.fe.extract(left)
            kp = feats["keypoints"]
        self._make_keyframe(fr, left, right, feats, kp, first=True)
        self._klt_prev = (np.array(self.kfs[-1].lm_id, np.int64), float(fr.t))
        self.kf_trigger_counts["bootstrap"] = \
            self.kf_trigger_counts.get("bootstrap", 0) + 1
        self.n_frames += 1
        ev = self._log_frame_evidence(None)
        if self.kit:
            self.kit.frame(fr.index, fr.t, self.T_wc, "init", evidence=ev)

    def _update_bindings(self, feats, assoc, state):
        """XC-01(c): bind every keypoint whose correspondence was a verified inlier this frame; a bound track whose
        correspondence failed on 2 consecutive TRACKED frames is unbound; bindings of dead tracks are dropped."""
        tid = np.asarray(feats["track_id"], np.int64)
        live = set(tid.tolist())
        self.bind = {t: l for t, l in self.bind.items() if t in live}
        self.bind_fail = {t: c for t, c in self.bind_fail.items() if t in live}
        if state not in ("tracking", "relocalized"):
            return
        a = np.asarray(assoc, np.int64)
        for j, t in enumerate(tid.tolist()):
            if j < len(a) and a[j] >= 0:
                self.bind[t] = int(a[j])
                self.bind_fail[t] = 0
            elif t in self.bind:
                self.bind_fail[t] = self.bind_fail.get(t, 0) + 1
                if self.bind_fail[t] >= 2:
                    del self.bind[t]
                    del self.bind_fail[t]

    def _reacquire(self, i, left, T_wc, feats, assoc, inflate=1.0, solve_only_new=True):
        """XC-01: search every estimator-window landmark that no live keypoint currently observes AT ITS PROJECTION,
        with the affine aligner (stages/align_cuda.py) from its last keyframe observation's patch.  The predicted
        affine is the depth ratio z_ref / z_cur (scale); the search radius is 3 x the verified reprojection sigma
        (the local map's own radius), times `inflate` on coasted frames.  An accepted point that coincides (within
        FEATURE_SIGMA_PX) with an existing unassociated keypoint binds that keypoint; otherwise it is ADDED to the
        frame as a new track.  The union of the frame's associations and the re-acquired ones is re-solved.
        -> (T_wc, kp); self._reacq_assoc (extended assoc) and self._reacq_solved are set."""
        from scipy.spatial import cKDTree
        self._reacq_assoc = np.asarray(assoc, np.int64)
        self._reacq_solved = False
        kp = feats["keypoints"]
        if getattr(self, "aligner", None) is None or not self.kf_pyr or T_wc is None or not np.isfinite(T_wc).all():
            return T_wc, kp
        self.reacq_stats["frames"] += 1
        used = set(int(x) for x in self._reacq_assoc[self._reacq_assoc >= 0])
        cand = [l for l in self.state.lm_kfs.keys() if l not in used and l in self.lm_last_obs
                and self.lm_last_obs[l][0] in self.kf_pyr]
        if not cand:
            return T_wc, kp
        cand = np.asarray(cand, np.int64)
        T_cw = se3_inv(T_wc)
        P = self.lm_xyz[cand]
        Pc = (T_cw[:3, :3] @ P.T).T + T_cw[:3, 3]
        z = Pc[:, 2]
        fx, cx, cy = self.rig.fx_rect, self.rig.cx, self.rig.cy
        m = self.fe.cv2 and (z > self.rig.near_clip_m) & (z < self.rig.far_bearing_m)
        u = fx * Pc[:, 0] / np.maximum(z, 1e-6) + cx
        v = fx * Pc[:, 1] / np.maximum(z, 1e-6) + cy
        h = 12
        m &= (u > h) & (u < self.rig.width - h) & (v > h) & (v < self.rig.height - h)
        if not m.any():
            return T_wc, kp
        cand, u, v, z = cand[m], u[m], v[m], z[m]
        cap = int(self.fe.budget)
        if len(cand) > cap:
            o = np.argsort(z)[:cap]
            cand, u, v, z = cand[o], u[o], v[o], z[o]
        self.reacq_stats["candidates"] += len(cand)
        sigma = max(self._verified_sigma_px, FEATURE_SIGMA_PX)
        radius = 3.0 * sigma * float(inflate)
        cur_pyr = self.aligner.pyramid(left)
        new_l, new_uv = [], []
        by_kf = {}
        for j, l in enumerate(cand.tolist()):
            by_kf.setdefault(self.lm_last_obs[l][0], []).append(j)
        for kf_id, js in by_kf.items():
            js = np.asarray(js)
            ref = np.array([[self.lm_last_obs[int(cand[j])][1], self.lm_last_obs[int(cand[j])][2]] for j in js],
                           np.float32)
            Tk = self.state.kf_pose.get(kf_id)
            if Tk is None:
                continue
            Pk = (se3_inv(Tk)[:3, :3] @ self.lm_xyz[cand[js]].T).T + se3_inv(Tk)[:3, 3]
            sc = np.clip(Pk[:, 2] / np.maximum(z[js], 1e-6), 0.25, 4.0).astype(np.float32)
            A0 = np.zeros((len(js), 2, 2), np.float32)
            A0[:, 0, 0] = sc
            A0[:, 1, 1] = sc
            init = np.stack([u[js], v[js]], 1).astype(np.float32)
            c, A, ok, zn = self.aligner.align(None, ref, None, init, A0, radius=radius,
                                              ref_pyr=self.kf_pyr[kf_id], cur_pyr=cur_pyr)
            for q in np.nonzero(ok)[0]:
                new_l.append(int(cand[js[q]]))
                new_uv.append(c[q])
        if not new_l:
            return T_wc, kp
        new_l = np.asarray(new_l, np.int64)
        new_uv = np.asarray(new_uv, np.float32)
        self.reacq_stats["accepted"] += len(new_l)
        # an existing UNASSOCIATED keypoint at the aligned position is that point: bind it instead of adding one
        idx = np.full(len(new_l), -1, np.int64)
        if len(kp):
            free = np.nonzero(self._reacq_assoc < 0)[0]
            if len(free):
                dd, ii = cKDTree(kp[free]).query(new_uv.astype(np.float64), k=1,
                                                 distance_upper_bound=FEATURE_SIGMA_PX)
                hit = np.isfinite(dd)
                idx[hit] = free[ii[hit]]
                _u, first = np.unique(idx[hit], return_index=True)
                keep_hit = np.zeros(len(new_l), bool)
                keep_hit[np.nonzero(hit)[0][first]] = True
                idx[hit & ~keep_hit] = -2                      # a second landmark on the same keypoint: dropped
        self.reacq_stats["bound_existing"] += int((idx >= 0).sum())
        add = idx == -1
        if add.any():
            nidx = self.fe.add_points(feats, new_uv[add])
            idx[add] = nidx
            self.reacq_stats["new_points"] += int(add.sum())
            kp = feats["keypoints"]
            self._reacq_assoc = np.concatenate([self._reacq_assoc, np.full(int(add.sum()), -1, np.int64)])
        okx = idx >= 0
        new_l, idx = new_l[okx], idx[okx]
        # the union re-solve: the frame's associations + the re-acquired ones
        have = np.nonzero(self._reacq_assoc >= 0)[0]
        Lall = np.concatenate([self._reacq_assoc[have], new_l])
        Kall = np.concatenate([have, idx])
        T2, m2, dec = solve_pose(self.lm_xyz[Lall], kp[Kall], self.rig, T_init=T_wc)
        self.decisions.append(dec)
        if T2 is None:
            return T_wc, kp
        a = np.full(len(kp), -1, np.int64)
        a[Kall[m2]] = Lall[m2]
        self._reacq_assoc = a
        self._reacq_solved = True
        for l_ in Lall[m2]:
            self.lm.col("obs")[l_] += 1
        return T2, kp

    def _predict(self, T0, t_now):
        """The constant-velocity prediction from pose T0 at t_now.  ARM XCLEAN_DT_CV off: self.vel @ T0 (the
        per-frame increment, the default).  On: T0 @ exp(s log D), D = the last measured BODY-frame increment
        (stored at the velocity update, None after any velocity reset), s = (t_now - t_last) / (t_last - t_prev).
        T0 is right-multiplied by a rigid D_s, so no inverse of the current pose is taken (a conjugation
        T0 D_s T0^-1 amplified rounding on held coasts: orthonormality error x2.4 per frame, v2_03 crash).
        No constant."""
        if not self.dt_cv or self._vel_body is None or self._vel_dt is None or self._t_last is None:
            return self.vel @ T0
        dt = float(t_now) - self._t_last
        if not (self._vel_dt > 0 and dt > 0):
            return self.vel @ T0
        sc = dt / self._vel_dt
        D = self._vel_body
        Ds = D if sc == 1.0 else se3_exp(sc * se3_log(D))
        if not np.isfinite(Ds).all():
            return self.vel @ T0
        if sc != 1.0:
            self.n_dt_scaled += 1
        return T0 @ Ds

    def _klt_prediction(self, fr):
        """xclean: where each of the PREVIOUS frame's keypoints should appear in this frame (aligned to
        fe.prev_keypoints()), or None.

        The relative motion is the tracker's own constant-velocity model (the same `vel` the PnP prediction uses);
        where an IMU exists its ROTATION is replaced by the gyro's (imu.delta_rotation_camera over the frame
        interval, at the run's own bias estimate) and the camera's displacement is kept from the velocity model.
        A keypoint with a landmark is predicted by projecting the landmark; one without is placed on its ray at the
        map's median depth (`_median_depth`, the same dial the PnP gate reads); with no depth at all, the rotation
        alone (the infinite homography).  The front end treats the prediction as an initial flow only and re-tracks
        any track it fails from zero motion, so a bad prediction cannot remove a track plain KLT would keep."""
        prev = getattr(self, "_klt_prev", None)
        kp0 = self.fe.prev_keypoints()
        if prev is None or kp0 is None or len(kp0) == 0:
            return None
        lm0, t0 = prev
        if len(lm0) != len(kp0) or not np.isfinite(self.T_wc).all():
            return None
        T0 = self.T_wc
        T1 = self._predict(T0, fr.t) if self.prev_T is not None else T0
        if not np.isfinite(T1).all() or np.abs(T1[:3, 3] - T0[:3, 3]).max() > self.rig.far_bearing_m:
            T1 = T0
        T10 = se3_inv(T1) @ T0
        R10, t10 = T10[:3, :3], T10[:3, 3]
        src = "cv"
        if self.imu is not None and self.imu.available() and fr.t > t0:
            Rg = self.imu.delta_rotation_camera(t0, fr.t)          # R_c0c1
            if Rg is not None and np.isfinite(Rg).all():
                c1_in_0 = -(R10.T @ t10)
                R10 = Rg.T
                t10 = -R10 @ c1_in_0
                src = "gyro"
        self.klt_pred_src[src] = self.klt_pred_src.get(src, 0) + 1
        fx, cx, cy = self.rig.fx_rect, self.rig.cx, self.rig.cy
        ray = np.stack([(kp0[:, 0] - cx) / fx, (kp0[:, 1] - cy) / fx, np.ones(len(kp0))], axis=1)
        X0 = np.full((len(kp0), 3), np.nan)
        ok = (lm0 >= 0) & (lm0 < self.store.n)
        if ok.any():
            T0inv = se3_inv(T0)
            X0[ok] = (T0inv[:3, :3] @ self.lm_xyz[lm0[ok]].T).T + T0inv[:3, 3]
        zmed = self._median_depth()
        need = ~(np.isfinite(X0).all(axis=1) & (X0[:, 2] > self.rig.near_clip_m))
        if zmed is not None and np.isfinite(zmed) and zmed > 0:
            X0[need] = ray[need] * zmed
            X1 = (R10 @ X0.T).T + t10
        else:
            X1 = (R10 @ X0.T).T + t10
            X1[need] = (R10 @ ray[need].T).T                   # rotation only: the infinite homography
        z1 = X1[:, 2]
        with np.errstate(all="ignore"):
            uv = np.stack([fx * X1[:, 0] / z1 + cx, fx * X1[:, 1] / z1 + cy], axis=1)
        uv[~(z1 > 1e-3)] = np.nan
        return uv

    def _collect(self, feats, kp):
        """3D-2D correspondences from every keyframe in the local window.

        Returns (P_world, uv, landmark_ids, current-frame keypoint indices) -
        the last of these is what lets a new keyframe INHERIT existing
        landmarks instead of duplicating them.

        xclean: a window keyframe's keypoint and this frame's keypoint correspond when they are the SAME KLT TRACK
        (fe.match, the track-id join - the replacement of the per-keyframe LightGlue call).  When the joins give fewer
        than KF_INLIER_HEADROOM x MIN_PNP_INLIERS correspondences (the count the front gate already treats as
        "enough") the PREDICTION-FREE RESCUE runs: ORB matching of each window keyframe's landmark-carrying
        keypoints against this frame's still-unused keypoints (fe.match_desc).  That is what LightGlue gave for free
        (wide-baseline, prediction-free association) and what KLT alone cannot do after blur or a coast.
        """
        ids_all, kpi_all = [], []
        if self.reacq_on and self.bind:
            tid = np.asarray(feats["track_id"], np.int64)
            b = np.array([self.bind.get(int(t), -1) for t in tid], np.int64)
            sel = np.nonzero(b >= 0)[0]
            if len(sel):
                ids_all.append(b[sel])
                kpi_all.append(sel)
        for k in self.kfs:
            i0, i1 = self.fe.match(k.feats, feats)
            if len(i0) == 0:
                continue
            lid = k.lm_id[i0]
            ok = lid >= 0
            if not ok.any():
                continue
            ids_all.append(lid[ok])
            kpi_all.append(i1[ok])
        n_join = len(np.unique(np.concatenate(ids_all))) if ids_all else 0
        self.fe_join_hist.append(n_join)
        # ★ W0 K4: KLT's join supply is below
        # LightGlue's PnP inliers at equal budget, so the prediction-free ORB match is a PER-FRAME path, not a rescue:
        # every window keyframe's landmark-carrying keypoints whose track did NOT join are matched against this
        # frame's keypoints that carry no correspondence yet (mutual NN, ratio, chance bound).  The front gate and
        # PnP RANSAC judge them exactly as they judged LightGlue's matches.
        # ARM XCLEAN_DESC_PERFRAME=0 (diagnostic): back to the rescue-only form (ORB only when the joins give < 75)
        perframe = (os.environ.get("XCLEAN_DESC_PERFRAME", "1") or "1").strip() != "0"
        if self.kfs and (perframe or n_join < KF_INLIER_HEADROOM * MIN_PNP_INLIERS):
            used = np.zeros(len(kp), bool)
            for a in kpi_all:
                used[a] = True
            n_add = 0
            for k in reversed(self.kfs):                      # newest keyframe first
                joined0 = np.zeros(len(k.lm_id), bool)
                j00, _j11 = self.fe.match(k.feats, feats)
                joined0[j00] = True
                m0 = (k.lm_id >= 0) & ~joined0
                if not m0.any() or used.all():
                    continue
                j0, j1 = self.fe.match_desc(k.feats, feats, mask0=m0, mask1=~used)
                if len(j0) == 0:
                    continue
                ids_all.append(k.lm_id[j0])
                kpi_all.append(j1)
                used[j1] = True
                n_add += len(j0)
            if n_add:
                self.n_desc_rescue += 1
            self.n_desc_rescue_pairs += n_add
        if not ids_all:
            z = np.zeros(0, np.int64)
            return np.zeros((0, 3)), np.zeros((0, 2)), z, z
        s = np.concatenate(ids_all)
        kpi = np.concatenate(kpi_all).astype(np.int64)
        # a landmark seen through two window keyframes appears twice; keep one
        _u, first = np.unique(s, return_index=True)
        s, kpi = s[first], kpi[first]
        return self.lm_xyz[s], kp[kpi], s, kpi

    def _track_local_map(self, T_wc, feats, kp, already, P_win, uv_win,
                         mask_win, kpi_win):
        """Project the estimator window into this frame and match what lands.

        Gates, all derived:
          * in front of the camera and inside the image;
          * depth inside the rig's DERIVED band (near clip .. max_depth);
          * within a search radius set by the reprojection uncertainty, not a
            typed pixel count;
          * descriptor similarity above the population's own median + scaled
            MAD, so the threshold moves with the scene instead of being a
            constant that is wrong on half the datasets.
        """
        from xclean.stages.features import hamming, HAMMING_CHANCE
        if self.lm_desc is None or self.lm_desc_n == 0:
            return T_wc, None, 0, None, None
        ids = np.fromiter(self.state.lm_kfs.keys(), dtype=np.int64)
        if len(ids) < 32:
            return T_wc, None, 0, None, None
        ids = ids[ids < self.lm_desc_n]
        seen = set(already.tolist()) if already is not None and len(already) \
            else set()
        if seen:
            ids = ids[~np.isin(ids, list(seen))]
        if len(ids) < 16:
            return T_wc, None, 0, None, None

        T_cw = se3_inv(T_wc)
        P = self.lm_xyz[ids]
        Pc = (T_cw[:3, :3] @ P.T).T + T_cw[:3, 3]
        z = Pc[:, 2]
        ok = (z > self.rig.near_clip_m) & (z < self.rig.max_depth_m)
        if ok.sum() < 16:
            return T_wc, None, 0, None, None
        ids, Pc, z = ids[ok], Pc[ok], z[ok]
        u = self.rig.fx_rect * Pc[:, 0] / z + self.rig.cx
        v = self.rig.fx_rect * Pc[:, 1] / z + self.rig.cy
        inside = (u > 0) & (u < self.rig.width) & (v > 0) & (v < self.rig.height)
        if inside.sum() < 16:
            return T_wc, None, 0, None, None
        ids, u, v = ids[inside], u[inside], v[inside]

        # xclean: PROJECTION-GUIDED ORB matching.  Each projected landmark is compared (Hamming) with the keypoints
        # inside its search radius only - the radius is the engine's own (3 x the VERIFIED reprojection spread), it was
        # applied AFTER a global descriptor arg-max there; a binary descriptor's global arg-max on repetitive texture
        # is often elsewhere, so the radius now selects the candidates and the descriptor picks among them.
        from scipy.spatial import cKDTree
        sigma = max(self._verified_sigma_px, FEATURE_SIGMA_PX)
        radius = 3.0 * sigma
        Q = feats["descriptors"]
        D = self.lm_desc[ids]
        tree = cKDTree(kp)
        nbrs = tree.query_ball_point(np.stack([u, v], axis=1), r=radius)
        cnt = np.fromiter((len(x) for x in nbrs), dtype=np.int64, count=len(nbrs))
        if cnt.sum() == 0:
            return T_wc, None, 0, None, None
        li = np.repeat(np.arange(len(ids)), cnt)
        ki = np.fromiter((j for x in nbrs for j in x), dtype=np.int64, count=int(cnt.sum()))
        dist = hamming(D[li], Q[ki])
        # per landmark: best and second-best candidate inside the radius
        order = np.lexsort((ki, dist, li))
        li, ki, dist = li[order], ki[order], dist[order]
        start = np.concatenate([[0], np.cumsum(cnt[cnt > 0])[:-1]])
        lm_of = li[start]
        bi_ = ki[start]
        bd_ = dist[start]
        n_c = cnt[cnt > 0]
        sd_ = np.where(n_c > 1, dist[np.minimum(start + 1, len(dist) - 1)], 1 << 20)

        # ★ THE THRESHOLD IS WHAT WE ALREADY BELIEVE, NOT A CONSTANT: a
        # candidate must be at least as good as the weakest quarter of the descriptor distances of the
        # correspondences the pose was just solved from (the 75 % quantile of the verified Hamming distances).
        ref = self._verified_distance(feats, already, kpi_win, mask_win)
        if ref is None:
            return T_wc, None, 0, None, None
        thr = float(np.quantile(ref, 0.75))
        # and a ratio test, because a landmark that matches two keypoints almost equally well has told us nothing
        # about which one it is
        ratio_ok = bd_ < 0.95 * sd_
        good = (bd_ <= thr) & (bd_ <= HAMMING_CHANCE) & ratio_ok
        if good.sum() < 8:
            return T_wc, None, 0, None, None
        ids = ids[lm_of]
        bi = bi_
        bs = -bd_.astype(np.float64)                  # higher = better, for the per-keypoint dedupe below
        ids_g, kpi_g = ids[good], bi[good]
        # a keypoint may win twice; keep the strongest
        order = np.argsort(-bs[good])
        seen_kp, keep = set(), []
        for o in order:
            k = int(kpi_g[o])
            if k in seen_kp:
                continue
            seen_kp.add(k)
            keep.append(o)
        keep = np.array(sorted(keep), dtype=np.int64)
        ids_g, kpi_g = ids_g[keep], kpi_g[keep]

        # ★ THE UNION, NOT THE REPLACEMENT.  The first version re-solved the
        # pose from the local-map correspondences ALONE and threw away the
        # ~150 window inliers the pose had just been built on, keeping ~35.
        # Fewer, newer correspondences are not better correspondences: the
        # odometry arm went from 0.0649 to 0.0897 on MH_01 with "local-map
        # tracking" that was really local-map REPLACEMENT.  Local-map matches
        # EXTEND the evidence; they never substitute for it.
        win = np.nonzero(mask_win)[0]
        Pw = np.concatenate([P_win[win], self.lm_xyz[ids_g]], axis=0)
        uv = np.concatenate([uv_win[win], kp[kpi_g]], axis=0)
        ids_all = np.concatenate([np.asarray(already)[win], ids_g])
        kpi_all = np.concatenate([np.asarray(kpi_win)[win], kpi_g])
        T2, m2, dec = solve_pose(Pw, uv, self.rig, T_init=T_wc,
                                 point_scale=self._far_point_scale(Pw, T_wc))
        self.decisions.append(dec)
        if T2 is None:
            return T_wc, None, 0, None, None
        n_extra = int(m2[len(win):].sum())
        return T2, m2, n_extra, kpi_all, ids_all

    def _verified_distance(self, feats, already, kpi, mask):
        """Hamming distance of the correspondences the pose was solved from (PnP inliers: landmark descriptor vs the
        keypoint it was matched to).  They are the frame's own definition of "a match we believe", so the local-map
        threshold is self-calibrating."""
        from xclean.stages.features import hamming
        if already is None or kpi is None or mask is None or self.lm_desc is None:
            return None
        ids = np.asarray(already, dtype=np.int64)
        kq = np.asarray(kpi, dtype=np.int64)
        m = np.asarray(mask, bool) & (ids < self.lm_desc_n) & (ids >= 0)
        if m.sum() < 8:
            return None
        return hamming(self.lm_desc[ids[m]], feats["descriptors"][kq[m]])

    def _mature_visible(self, T_wc):
        """(count, z_p10) of the MATURE landmarks the map itself predicts
        visible in this view — history's jurisdiction, and the depth an
        occluder must undercut (see the referee in step()).

        Same geometric gates as the local-map projection: in front of the
        camera, inside the image, inside the derived depth band.
        """
        ids = np.fromiter(self.state.lm_kfs.keys(), dtype=np.int64)
        if not len(ids):
            return 0, None
        cur = len(self.all_kf) - 1
        host = self.lm.col("host")
        ids = ids[ids < len(host)]
        age = cur - host[ids]
        ids = ids[age > self.max_kf_map]
        if not len(ids):
            return 0, None
        T_cw = se3_inv(T_wc)
        P = self.lm_xyz[ids]
        Pc = (T_cw[:3, :3] @ P.T).T + T_cw[:3, 3]
        z = Pc[:, 2]
        ok = (z > self.rig.near_clip_m) & (z < self.rig.max_depth_m)
        if not ok.any():
            return 0, None
        Pc = Pc[ok]; z = z[ok]
        u = self.rig.fx_rect * Pc[:, 0] / z + self.rig.cx
        v = self.rig.fx_rect * Pc[:, 1] / z + self.rig.cy
        inside = (u > 0) & (u < self.rig.width) & (v > 0) \
            & (v < self.rig.height)
        if not inside.any():
            return 0, None
        return int(inside.sum()), float(np.quantile(z[inside], 0.10))

    def add_structure_obs(self, kf_id, lm_ids, uv):
        """Cross-visit reprojection rows for the final global BA record.

        Called by the loop closer at weld verification (finalise path, main
        thread, before global_refine reads the record).  The rows carry no
        stereo disparity (obs_d = 0: re-observation, not re-measurement) and
        enter the same obs_* record every keyframe observation uses.
        """
        lm_ids = np.asarray(lm_ids, np.int64)
        if not len(lm_ids):
            return
        self.obs_kf.append(np.full(len(lm_ids), int(kf_id), np.int32))
        self.obs_lm.append(lm_ids)
        self.obs_uv.append(np.asarray(uv, np.float64))
        self.obs_d.append(np.zeros(len(lm_ids)))
        self.n_structure_obs += len(lm_ids)

    def _window_turnover_frames(self):
        """One full matching-window turnover at the run's own keyframe
        cadence, in frames.

        Past this horizon every keyframe in the matching window predates the
        entire coast streak, so the window is stale BY CONSTRUCTION and
        matching inside it can only fail.  Same derivation as the quarantine
        escape valve's cap (which allows TWO turnovers, because a transient
        occluder must first pass through); Class C - the cadence is measured
        from the run's own keyframes, the span is the window's own size.
        """
        gaps = np.diff([r[1] for r in self.all_kf[-12:]])
        cadence = float(np.median(gaps)) if len(gaps) else 1.0
        return max(self.max_kf_map * cadence, float(self.max_kf_map))

    def _relocalize(self, fr, left, feats, kp):
        """Match beyond the stale matching window (the coast-deadlock fix).

        Retrieval runs over EVERY keyframe the loop store holds - including
        the most recent ones the loop's `min_gap` exists to exclude, because
        the best recovery candidate is precisely the last healthy keyframe.
        Retrieval only SURFACES candidates (the loop stack's own division of
        labour); the decider is absolute, guess-free PnP at
        `MIN_RELOC_INLIERS` - twice the loop-weld floor, because a weld is
        additionally vetted by a soft gate, a robust kernel and the bundle,
        while a reloc replaces the live pose outright on this evidence alone.
        The candidate count is the matching window's own size: this IS the
        matching window, relocated.

        Reads the loop worker's stores (`D`, `order`, `kf`); the worker only
        appends, so a stale prefix is a smaller candidate set, never a wrong
        one.  Returns (T_wc, mask, landmark_ids, keypoint_ids, Decision).
        """
        import cv2
        import zlib
        from xclean.core.derive import MIN_RELOC_INLIERS
        lp = self.loop
        fail = (None, None, None, None)
        D, order = lp.D, lp.order
        n = min(len(D), len(order))
        dials = {"coast_streak": int(self._coast_streak), "store": int(n)}
        if n < 1:
            return (*fail, Decision("reloc", False, "empty keyframe store",
                                    dials))
        # xclean: the RICH sets (loop/closer.py _rich) - the current frame's own retrieval ORB set (1000 oriented
        # multi-scale) is matched prediction-free against each candidate's rich keypoints that carry stereo 3D, and
        # the candidate's 3D, placed by its CURRENT pose, decides by absolute PnP at MIN_RELOC_INLIERS.  On mh_04
        # revisit pairs the rich set gives a median 60 geometric inliers vs 16 for the tracker's keypoints (c1).
        # Inliers whose candidate keypoint is a tracker landmark are associated back to this frame's tracker
        # keypoint within FEATURE_SIGMA_PX, so the forced keyframe inherits them.
        from scipy.spatial import cKDTree
        desc, r_uv, r_desc = lp.place.describe_full(left)
        sims = D[:n] @ desc
        k = min(self.max_kf_map, n)
        top = np.argsort(-sims)[:k]
        dials["best_sim"] = float(sims[top[0]])
        tried = []
        cur = {"descriptors": r_desc}
        for j in top:
            kid = int(order[j])
            rec = lp.kf.get(kid)
            ra = None if rec is None else rec.get("rich")
            if ra is None or not len(r_desc):
                continue
            has3d = np.isfinite(ra["X"][:, 0])
            i0, i1 = self.fe.match_desc({"descriptors": ra["desc"]}, cur, mask0=has3d)
            nm = int(len(i0))
            if nm < MIN_RELOC_INLIERS:
                tried.append([kid, nm, 0])
                continue
            T_a = lp._current_pose(kid)
            Xa = ra["X"][i0].astype(np.float64)
            P = (T_a[:3, :3] @ Xa.T).T + T_a[:3, 3]
            uv = r_uv[i1].astype(np.float64)
            # this pathway's own DETERMINISTIC seed.  crc32, not hash():
            # str hash is salted per process, so hash-derived seeds are a
            # different RANSAC sequence every process.
            cv2.setRNGSeed((zlib.crc32(b"reloc") + int(fr.index))
                           % (2 ** 31 - 1))
            T, m, _d = solve_pose(P, uv, self.rig, T_init=None,
                                  min_inliers=MIN_RELOC_INLIERS)
            inl = 0 if m is None else int(m.sum())
            tried.append([kid, nm, inl])
            if self.decision_log:
                dials.setdefault("ransac", []).append([kid, int((_d.dials or {}).get("ransac_inliers", inl))])
            if T is not None:
                lid = ra["lm"][i0]
                sel = np.nonzero(m & (lid >= 0))[0]
                src_r = np.zeros(0, np.int64)
                kpi_r = np.zeros(0, np.int64)
                if len(sel) and len(kp):
                    dd, ii = cKDTree(np.asarray(kp, np.float64)).query(uv[sel], k=1,
                                                                       distance_upper_bound=FEATURE_SIGMA_PX)
                    hit = np.isfinite(dd)
                    src_r = lid[sel][hit].astype(np.int64)
                    kpi_r = ii[hit].astype(np.int64)
                    _u, first = np.unique(kpi_r, return_index=True)
                    src_r, kpi_r = src_r[first], kpi_r[first]
                m_r = np.ones(len(src_r), bool)
                dials.update(candidate=kid, sim=float(sims[j]), matches=nm,
                             inliers=inl, associated=int(len(src_r)), tried=tried)
                self.n_reloc_rich_inliers = getattr(self, "n_reloc_rich_inliers", 0) + inl
                return (T, m_r, src_r, kpi_r, Decision(
                            "reloc", True, "relocalized against kf %d: %d "
                            "inliers (floor %d)"
                            % (kid, inl, MIN_RELOC_INLIERS), dials,
                            margin=inl - MIN_RELOC_INLIERS,
                            threshold=MIN_RELOC_INLIERS))
        dials["tried"] = tried
        return (*fail, Decision("reloc", False,
                                "no candidate verified at the reloc floor",
                                dials, threshold=MIN_RELOC_INLIERS))

    def _far_point_scale(self, P_world, T_wc):
        """ARM XTEN_FAR_DEPTH in (pnp, both): per-correspondence sigma factor max(1, z / z_far), z = the landmark's
        depth in the (predicted) current camera - a far point's stereo depth carries the drive-level disparity
        offset at relative size b/d, so its weight in the translation falls as (z_far / z)^2 beyond z_far.
        None when the arm is off (solve_pose then runs its original, bit-identical path)."""
        if self.far_depth not in ("pnp", "both") or T_wc is None or len(P_world) == 0 \
                or not np.isfinite(T_wc).all():
            return None
        T_cw = se3_inv(T_wc)
        z = (T_cw[:3, :3] @ np.asarray(P_world, float).T).T[:, 2] + T_cw[2, 3]
        z = np.where(np.isfinite(z) & (z > 0), z, 0.0)
        return np.maximum(1.0, z / self.far_depth_z)

    def _median_depth(self):
        if not self.kfs:
            return None
        k = self.kfs[-1]
        ids = k.lm_id[k.lm_id >= 0]
        if len(ids) == 0:
            return None
        P = self.lm_xyz[ids[:400]]
        T_cw = se3_inv(self.T_wc)
        z = (T_cw[:3, :3] @ P.T).T[:, 2] + T_cw[2, 3]
        z = z[np.isfinite(z) & (z > 0)]
        return float(np.median(z)) if len(z) else None

    # ---- LOG (tier-1 programme, 2026-09-29): provenance, default-on -----
    # Counters and records only.  Nothing here touches an estimate, the
    # RNG or a floating-point path the estimator reads: `mask.sum()` is
    # the integer _need_keyframe already forms, and _median_depth is the
    # pure read step() already performs once per frame.
    KF_TRIGGER_LABELS = (("relocalized", "relocalized"),
                         ("translation", "translation"),
                         ("rotation", "rotation"),
                         ("tracking health (lookahead)", "health_lookahead"),   # T1-b
                         ("tracking health", "health"),
                         ("no trigger", "no_trigger"),
                         ("not from a coasted", "coasting"),
                         ("quarantined", "quarantined"))

    @classmethod
    def kf_trigger_label(cls, reason):
        """The short label of a keyframe Decision reason (summary key)."""
        for pre, lab in cls.KF_TRIGGER_LABELS:
            if reason.startswith(pre):
                return lab
        return reason[:32]

    def _log_kf_decision(self, d):
        lab = self.kf_trigger_label(d.reason)
        self.kf_decision_counts[lab] = self.kf_decision_counts.get(lab, 0) + 1
        if d.accepted:
            self.kf_trigger_counts[lab] = \
                self.kf_trigger_counts.get(lab, 0) + 1

    def _log_frame_evidence(self, mask):
        """(inliers, budget_at_cap, budget) for this frame, as the keyframe
        decision sees them: the PnP inlier count and the feature budget
        AFTER this frame's controller update.  Also keeps the count of
        frames at cap for the summary."""
        inl = None if mask is None else int(mask.sum())
        self._frame_inliers = inl
        bud = getattr(self.fe, "budget", None)
        cap = getattr(self.fe, "cap", None)
        at_cap = None
        if bud is not None and cap is not None:
            at_cap = int(int(bud) >= int(cap))
            self.n_frames_at_cap += at_cap
        return (inl, at_cap, None if bud is None else int(bud))

    def _log_kf_evidence(self, kid):
        """Record the creating frame's inliers and _median_depth() for the
        keyframe just appended (self.kfs[-1]).  A logging failure must
        never take a run down: it is counted instead."""
        try:
            z = self._median_depth()
        except Exception:
            z = None
            self.n_kf_evidence_err += 1
        self.kf_evidence[int(kid)] = {"inliers": self._frame_inliers,
                                      "z_med": z}

    # ---- T1-b: the health rule's inputs and counters --------------------
    def _budget_at_cap(self):
        """fe.budget >= fe.cap (the same read as the LOG's budget_at_cap
        column).  A frontend without a budget cannot host the lookahead
        rule: hard error under the flag, False otherwise."""
        bud = getattr(self.fe, "budget", None)
        cap = getattr(self.fe, "cap", None)
        if bud is None or cap is None:
            if self.kf_health_mode == "lookahead":
                raise RuntimeError("%s=lookahead needs a frontend with a "
                                   "feature budget and cap" % KF_HEALTH_ENV)
            return False
        return int(bud) >= int(cap)

    def _health_verdict(self, ref, inl, thr_i):
        """Evaluate health_trigger() for this frame against the reference
        keyframe `ref` and count which rule decided (summary.kf_health)."""
        ev = self.kf_evidence.get(int(ref.id))
        n_ref = None if ev is None else ev.get("inliers")
        hv = health_trigger(inl, thr_i, n_ref, self._frames_since_kf,
                            self._budget_at_cap(), self.kf_health_mode)
        self.kf_health_evals[hv.rule] += 1
        if hv.fire:
            self.kf_health_fires[hv.rule] += 1
        return hv

    def kf_health_summary(self):
        ev, fi = self.kf_health_evals, self.kf_health_fires
        return {"mode": self.kf_health_mode,
                "lookahead_fires": int(fi["lookahead"]),
                "old_rule_fires": int(fi["old"] + fi["fallback"]),
                "fallbacks_no_evidence": int(ev["fallback"]),
                "frames_at_cap": int(self.n_frames_at_cap),
                "evaluations": {k: int(v) for k, v in ev.items()},
                "fallback_fires": int(fi["fallback"]),
                "pre_registered": list(KF_HEALTH_CHOICES)}

    # ---- the keyframe policy --------------------------------------------
    def _need_keyframe(self, mask, state):
        self._frames_since_kf += 1          # T1-b: m for the health rule
        ref = self.kfs[-1]
        dT = se3_inv(ref.T_wc) @ self.T_wc
        trans = float(np.linalg.norm(dT[:3, 3]))
        rot = float(rot_angle(np.eye(3), dT[:3, :3]))
        inl = int(mask.sum())
        thr_t = self.rig.baseline_m
        thr_r = 0.25 * self.rig.width / self.rig.fx_rect
        thr_i = KF_INLIER_HEADROOM * MIN_PNP_INLIERS
        dials = {"translation_m": trans, "rotation_rad": rot, "inliers": inl,
                 "thr_translation_m": thr_t, "thr_rotation_rad": thr_r,
                 "thr_inliers": thr_i, "state": state}
        if state == "relocalized":
            # ★ THE RE-ARM.  The matching window is stale by construction
            # (that is what triggered the reloc) and this pose stands on
            # mature history at the reloc floor - a keyframe here is what
            # puts a matchable keyframe back into the window.  Without it the
            # tracker relocalizes every frame until a motion trigger happens
            # to fire, which is the deadlock wearing a different mask.
            return Decision("keyframe", True, "relocalized: a fresh keyframe "
                            "re-arms the stale matching window", dials)
        if state == "coasting":
            return Decision("keyframe", False, "not from a coasted pose", dials)
        if state == "quarantined":
            return Decision("keyframe", False, "quarantined: the pose is not "
                            "verified against mature history while the map "
                            "claims this view (see age-consensus)", dials)
        if trans >= thr_t:
            return Decision("keyframe", True, "translation >= one stereo "
                            "baseline (the temporal pair now carries as much "
                            "parallax as the stereo pair)", dials,
                            margin=trans - thr_t, threshold=thr_t)
        if rot >= thr_r:
            return Decision("keyframe", True, "rotation >= quarter-image shift",
                            dials, margin=rot - thr_r, threshold=thr_r)
        # T1-b: the health rule is health_trigger() (module level).  With
        # XTEN_KF_HEALTH unset, or the budget below its cap, its verdict is
        # exactly the old inline code's (fire, margin, threshold, reason).
        hv = self._health_verdict(ref, inl, thr_i)
        if hv.rule == "lookahead":          # under the flag only: the dials
            dials.update(pred_inliers=hv.pred, ref_inliers=self.kf_evidence[
                int(ref.id)]["inliers"], frames_since_kf=self._frames_since_kf,
                budget_at_cap=True)
        if hv.fire:
            return Decision("keyframe", True, hv.reason, dials,
                            margin=hv.margin, threshold=hv.threshold)
        return Decision("keyframe", False, "no trigger", dials)

    def _make_keyframe(self, fr, left, right, feats, kp, first=False,
                       assoc=None, no_create=False):
        i = fr.index
        kid = len(self.all_kf)
        if no_create:
            # quarantined: the keyframe carries only INHERITED associations
            # and measures no depth - nothing new enters the map from an
            # unverified pose
            lm_id = np.array(assoc if assoc is not None
                             else np.full(len(kp), -1, np.int64), np.int64)
            self._last_disp = None
            self._kf_disp = None
            self._pending_desc = feats["descriptors"]
        else:
            with self.led.stage("depth", frame=i):
                disp = self.depth.compute(left, right)
                self.depth_frames += 1
            self._kf_disp = disp                      # xclean: handed to the loop worker (rich keypoint depth)
            if self.reanchor_on:
                # XC-07: the keyframe observation is re-measured against each track's birth-keyframe patch; a track
                # whose chain drifted beyond 3 sigma from its anchor ends its association (new id, no inherited
                # landmark: it becomes a new landmark candidate)
                kp, ended, tid = self.fe.reanchor(left, kp, feats["track_id"])
                feats["keypoints"] = kp
                feats["track_id"] = tid
                if ended.any():
                    if assoc is not None:
                        assoc = np.array(assoc, np.int64)
                        assoc[ended] = -1
            self._pending_desc = feats["descriptors"]
            with self.led.stage("landmark_creation", kf=kid, frame=i):
                lm_id = self._create(kp, disp, kid,
                                     assoc if assoc is not None
                                     else np.full(len(kp), -1, np.int64),
                                     left=left, right=right)
        k = Keyframe(kid, i, fr.t, self.T_wc.copy(), feats, kp, lm_id,
                     fr.left, fr.right)
        self.kfs.append(k)
        self.all_kf.append([kid, i, fr.t, self.T_wc.copy()])
        # LOG: AFTER the append - _median_depth reads self.kfs[-1]
        self._log_kf_evidence(kid)
        self._frames_since_kf = 0           # T1-b: this is the new reference
        good = lm_id >= 0
        self.state.add_keyframe(kid, self.T_wc, lm_id[good], kp[good])
        if self.reacq_on:
            if getattr(self, "aligner", None) is None:
                from xclean.stages.align_cuda import CudaPatchAligner
                self.aligner = CudaPatchAligner()
            self.kf_pyr[kid] = self.aligner.pyramid(left)
            for l_, (u_, v_) in zip(lm_id[good], kp[good]):
                self.lm_last_obs[int(l_)] = (kid, float(u_), float(v_))
            alive = set(self.state_kfs[-self.state_window:]) | {kid}
            for k_ in [k_ for k_ in self.kf_pyr if k_ not in alive]:
                del self.kf_pyr[k_]
        self.obs_kf.append(np.full(int(good.sum()), kid, np.int32))
        self.obs_lm.append(lm_id[good].astype(np.int64))
        self.obs_uv.append(kp[good].astype(np.float64))
        # the STEREO measurement at this observation.  0 means "no usable
        # disparity here" (behind the near clip, or on an occlusion edge) and
        # the solver drops those rows rather than inventing a depth.
        dsp = getattr(self, "_last_disp", None)
        self.obs_d.append(np.zeros(int(good.sum())) if dsp is None
                          else np.asarray(dsp, np.float64)[good])
        # ---- the gyro relative-rotation factor to the previous keyframe ----
        # `all_kf[-1]` is the keyframe just appended, so the PREVIOUS one is
        # [-2].  Using [-1] made dt exactly zero and silently produced zero
        # factors for a whole run - which the summary caught only because
        # `gyro_factors` is a reported dial rather than an internal counter.
        if self.imu is not None and self.imu.available() and len(self.all_kf) > 1:
            prev = self.all_kf[-2]
            dt = fr.t - prev[2]
            R = self.imu.delta_rotation_camera(prev[2], fr.t)
            if R is not None and dt > 0:
                # the visual relative rotation, which is what the bias is
                # measured against (T1-c: the interval travels with it, so
                # a gap-overlapping interval is refused inside observe too)
                dR_vis = prev[3][:3, :3].T @ self.T_wc[:3, :3]
                self.imu.observe(R, dR_vis, dt, t0=prev[2], t1=fr.t)
                # the interval's gyro-vs-vision disagreement, attributed to
                # the NEW keyframe: the --vision-trust dial (gyro noise
                # 6.6e-5 rad is negligible against 2e-4..2e-3 measured)
                e_tr = so3_log(so3_project(R.T @ dR_vis))
                if np.all(np.isfinite(e_tr)):
                    self.kf_disagree[kid] = float(np.linalg.norm(e_tr))
                if self.imu.has_bias():
                    self.state.rot_factors[(prev[0], kid)] = (
                        R, self.imu.prior_sigma(dt))
                    self.n_gyro_factors += 1
            elif R is None and dt > 0:
                # T1-c provenance: a refused interval withholds its bias
                # observation together with its rotation factor (no-op
                # unless XTEN_IF_GAPSPAN=1)
                self.imu.note_observe_withheld(prev[2], fr.t)
        with self.led.stage("kf_insert", kf=kid, frame=i):
            if self.kit:
                self.kit.keyframe(kid, i, fr.t, self.T_wc, k.left, k.right,
                                  "")
        # ---- THE BACKEND.  One bounded solve per keyframe (plan 5.1) ------
        if self.ba is not None and len(self.state.kf_pose) >= 3:
            with self.led.stage("local_ba", kf=kid, frame=i):
                dec = self.ba.solve(self.state, kid)
            self.ba_decisions.append(dec)
            if dec.accepted:
                self._absorb()

        # ---- LOOP CLOSURE.  The SOLVE is asynchronous by construction; what
        # happens here is enqueue-and-poll, and the apply is its own budgeted
        # per-event process (plan 3.4: async is a CORRECTNESS property).
        if self.loop is not None:
            with self.led.stage("loop_handoff", frame=i):
                self.loop.enqueue(kid, kp, feats, lm_id, self.T_wc, left, right=right,
                                  disp=getattr(self, "_kf_disp", None))
            got = self.loop.poll()
            if got is not None:
                with self.led.stage("loop_apply", kf=kid, frame=i):
                    d = self.loop.apply(got, self)
                if d is not None:
                    self.loop_decisions.append(d)
                    self.n_applied += 1
        # bound the working set: keyframes leave the window, landmarks retire
        with self.led.stage("kf_culling", kf=kid, frame=i):
            # the matching window sheds keyframes fast; their DESCRIPTORS are
            # what costs, and those are not needed once the frame has moved on
            while len(self.kfs) > self.max_kf_map:
                old = self.kfs.pop(0)
                old.feats = None
            # the estimator window sheds them slowly, and THAT is where a
            # landmark leaves the working set and is retired to the record
            self.state_kfs.append(kid)
            while len(self.state_kfs) > self.state_window:
                dead = self.state_kfs.pop(0)
                orphans = self.state.drop_keyframe(dead)
                self._retire_landmarks(orphans, reason=1)
        # the parallax dial the anisotropic gate reads (measured, not assumed)
        if len(self.all_kf) > 1 and self.inlier_hist:
            self.parallax_hist.append(float(np.median(
                np.abs(kp[:, 0] - np.median(kp[:, 0])))))

    # RE-DERIVED 2026-09-23 FROM 15 CLEAN SOLVE TIMINGS.
    # The old rule was LINEAR in observations. Measured cost is SUPERLINEAR and carries a large inertial term
    # the old rule ignored entirely:
    #
    #     solve_s = exp(-16.887590) * n_obs ** 1.681538 * 3.723154 ** inertial     R2 = 0.859
    #
    # Fitting log(solve_s) over 15 rows: n_obs alone R2 0.678, n_kf alone 0.561, n_obs+inertial 0.859, and
    # adding n_kf to that gains nothing (0.860, coefficient 0.106). So an earlier evaluation s10.4 is REFUTED on its own
    # terms: it says the budget uses the wrong run statistic (observations, when the cost is in keyframes),
    # but observations predict BETTER than keyframes. The defect is the FUNCTIONAL FORM (linear against a
    # measured exponent of 1.681538) plus the missing inertial multiplier - not the choice of statistic.
    #
    # It reproduces the observed truncation without being given any truncation data. Under the OLD rule
    # rosario_06 (408,051 obs, inertial) needs a predicted 695 s against a 528 s budget = 1.32x -> truncates,
    # while all 15 measured rows sit at 0.07-0.62x and fit. An earlier evaluation independently reports Rosario truncating
    # on 3-5 of 5 draws and every other row 0 of 175. Under the new rule rosario_06 gets 1173 s and fits, and
    # kitti_09 - which is NOT inertial - correctly drops 356 s -> 162 s instead of being over-provisioned.
    #
    # HEADROOM is DERIVED, not typed: exp(2.326 * sigma) with sigma = 0.3940 the log-residual scatter, a
    # 99 pct one-sided prediction bound. It covers the measured worst case (kitti_05 at 2.21x of prediction).
    # The 120 s floor is kept so rows that already fit are unaffected.
    #
    # ---- THE EARLIER EVALUATION DISCIPLINE (2026-09-24): this model is an ARM, not the default. It was verified on 3 rows x
    # 1 draw, and the earlier evaluation's rule is that no mechanism becomes a default without a test series. Default = the earlier evaluation's
    # linear rule, byte-for-byte. XTEN_GBA_COST_MODEL=1 selects the model below.
    # Every run records which rule it used in summary.json global_ba.budget_rule.
    GBA_CM_LOG_C = -16.887590
    GBA_CM_ALPHA = 1.681538
    GBA_CM_INERTIAL_MULT = 3.723154
    GBA_CM_HEADROOM = 2.5002
    # an earlier evaluation (defect 21): 181 runs on disk give a p95 global BA cost of 6.47e-4 s per observation; 2x that is
    # the bound, floored at 120 s.
    GBA_S_PER_OBS_P95 = 6.47e-4
    GBA_HEADROOM = 2.0
    GBA_FLOOR_S = 120.0

    def _gba_budget_rule(self):
        import os
        return "cost-model" if os.environ.get("XTEN_GBA_COST_MODEL") == "1" else "linear"

    def _gba_budget(self, n_obs):
        if self._gba_budget_rule() == "cost-model":
            import math
            cost = (math.exp(self.GBA_CM_LOG_C) * float(max(n_obs, 1)) ** self.GBA_CM_ALPHA
                    * (self.GBA_CM_INERTIAL_MULT if self.inertial else 1.0))
            return max(self.GBA_FLOOR_S, self.GBA_CM_HEADROOM * cost)
        return max(self.GBA_FLOOR_S,
                   self.GBA_HEADROOM * self.GBA_S_PER_OBS_P95 * float(n_obs))

    def global_refine(self, iters=10, time_budget_s=None, verbose=True):
        """One global bundle adjustment over the whole trajectory.

        Per EVENT, off the frame path, and never inside the per-keyframe
        budget - see `backend/global_ba.py` for why the two solvers are
        separate objects rather than one solver with a bigger window.
        """
        from xclean.backend.global_ba import global_ba
        if not self.obs_kf or len(self.all_kf) < 5:
            return None
        okf_raw = np.concatenate(self.obs_kf)
        olm_raw = np.concatenate(self.obs_lm)
        ouv = np.concatenate(self.obs_uv)
        odisp = (np.concatenate(self.obs_d) if self.obs_d
                 else np.zeros(len(ouv)))
        kf_ids = np.array([r[0] for r in self.all_kf], np.int64)
        kf_pos = {int(k): i for i, k in enumerate(kf_ids)}
        keep = np.array([int(k) in kf_pos for k in okf_raw])
        okf_raw, olm_raw, ouv = okf_raw[keep], olm_raw[keep], ouv[keep]
        odisp = odisp[keep]
        # a landmark needs two observations to be an unknown at all
        uniq, inv, cnt = np.unique(olm_raw, return_inverse=True,
                                   return_counts=True)
        m = cnt[inv] >= 2
        okf_raw, olm_raw, ouv = okf_raw[m], olm_raw[m], ouv[m]
        odisp = odisp[m]
        uniq, inv = np.unique(olm_raw, return_inverse=True)
        if len(uniq) < 50:
            return None
        okf = np.array([kf_pos[int(k)] for k in okf_raw], np.int64)
        poses = np.stack([r[3] for r in self.all_kf])
        pts = self.store.xyz[uniq].copy()

        host = self.lm.col("host")
        rho = self.lm.col("invrange")
        sig = self.lm.col("invsig_r")      # RANGE information, see _gather_prior
        c = np.zeros((len(uniq), 3))
        r0 = np.zeros(len(uniq))
        isg = np.zeros(len(uniq))
        val = np.zeros(len(uniq))
        for i, l in enumerate(uniq):
            l = int(l)
            if l >= len(host):
                continue
            h = int(host[l])
            if h in kf_pos:
                c[i] = poses[kf_pos[h]][:3, 3]
                r0[i] = rho[l]
                isg[i] = sig[l]
                val[i] = 1.0 if sig[l] > 0 else 0.0
        if self.inverse_depth:
            # ★ THE WHOLE TRAJECTORY IN THE SAME PARAMETERISATION AS THE
            # WINDOW.  Running the global solve on Euclidean points while the
            # record is host-relative would leave the two disagreeing: the
            # trajectory would carry the global BA's corrections and the
            # landmarks would carry the local BA's, and the kit would be a
            # mixture of the two with nothing saying so.
            hidx = np.full(len(uniq), -1, np.int64)
            alive = np.zeros(len(uniq), bool)
            for i, l in enumerate(uniq):
                l = int(l)
                if l < len(host) and int(host[l]) in kf_pos:
                    hidx[i] = kf_pos[int(host[l])]
                    alive[i] = True
            if alive.sum() < 50:
                return None
            if not alive.all():
                keep_o = alive[inv]
                okf, inv2, ouv2 = okf[keep_o], inv[keep_o], ouv[keep_o]
                odisp = odisp[keep_o]
                remap = np.full(len(uniq), -1, np.int64)
                remap[alive] = np.arange(int(alive.sum()))
                inv = remap[inv2]
                ouv = ouv2
                uniq, hidx = uniq[alive], hidx[alive]
            m = self.store.m[uniq].copy()
            rho_state = self.lm.col("invd")[uniq].copy()
            prior_id = (np.stack([self.lm.col("mx0")[uniq],
                                  self.lm.col("my0")[uniq]], axis=1),
                        self.lm.col("invd0")[uniq].copy(),
                        np.full(len(uniq), 1.0 / (self.rig.bearing_sigma
                                * (1e-3 if self.frozen_bearing else 1.0))),
                        self.lm.col("invsig")[uniq].copy(),
                        (self.lm.col("invsig")[uniq] > 0).astype(float))
        # ★ REFUTED, 2026-08-25: re-preintegrating EVERY interval with the
        # final bias (so the first ~20 keyframes, which get no factor while
        # the online bias converges, receive one retroactively) measured
        # NEUTRAL on both rows over three draws each - V1_01 0.0948 -> 0.0969
        # with a worse spread, MH_02 0.0392 -> 0.0400/0.0451.  The early error
        # is real (first sixth 0.130 vs 0.06) but it is not a missing rotation
        # constraint.  Left as a comment rather than as code that does nothing.
        # ★ --gba-tight-gyro: the ONLINE factors keep
        # the loop-stable spread sigma; the run-end global BA has no
        # feedback loop, so its copies of the same factors may carry
        # the externally-validated TIGHT sigma instead.  Recomputed
        # here, at the final bias quality, rather than stored - the
        # stored sg is whatever the online rule was at creation time.
        # (T1-h, 2026-09-30: the loop is now the pure gba_rotf_assemble,
        # byte for byte, so the covered rule below is testable where it
        # is applied.)
        kf_t = {int(r[0]): float(r[2]) for r in self.all_kf}
        rotf = gba_rotf_assemble(self.state.rot_factors, kf_pos, kf_t, self.imu,
                                 self.gba_tight_gyro or self.gba_robust_gyro)
        # the verified loop welds, so the global solve cannot relax what the
        # pose graph just closed
        welds = []
        if self.loop is not None:
            for wd in self.loop.welds:
                a, b = int(wd["from"]), int(wd["to"])
                if a in kf_pos and b in kf_pos:
                    welds.append((kf_pos[a], kf_pos[b], wd["Z"],
                                  wd.get("info", float(wd["w"]))))

        if time_budget_s is None:
            time_budget_s = self._gba_budget(len(okf))
        # P6 (XTEN_GBA_ITERBUDGET=1): the solver gets NO wall-clock budget - it
        # stops on its iteration budget (iters LM iterations per pass, 2 passes)
        # or on convergence, so machine load cannot change what it returns.  The
        # rule's budget is still computed and recorded (budget_s), never applied.
        _itb = getattr(self, "gba_iterbudget", None)
        if _itb is None:
            from xclean.backend.global_ba import gba_iterbudget_mode
            _itb = gba_iterbudget_mode()
        tb_solver = None if _itb else time_budget_s
        # heartbeat (the madmax_B0 lesson): announce the long silent
        # stages with their sizes and budget, so a big-row close is
        # observable without a debugger
        if verbose or len(kf_ids) > 2000:
            print(" [gba] entering: %d keyframes, %d observations, "
                  "budget %.0f s" % (len(kf_ids), len(okf),
                                     float(time_budget_s))
                  + ("" if not _itb else " NOT APPLIED (XTEN_GBA_ITERBUDGET=1: "
                     "%d LM iterations x 2 passes, or convergence)" % iters),
                  flush=True)
        # ★ --vision-trust: per-keyframe vision weights from the run's own
        # gyro-vs-vision disagreement population (min(1, median/dis) -
        # down-weight only, floored at the population median; keyframes
        # with no gyro measurement keep weight 1)
        vt_w = None
        if self.vision_trust and self.kf_disagree:
            vt_w = np.ones(len(kf_ids))
            dis = np.array([self.kf_disagree.get(int(k), np.nan)
                            for k in kf_ids])
            fin = np.isfinite(dis)
            if fin.sum() >= 20:
                med = float(np.median(dis[fin]))
                if med > 1e-12:
                    vt_w[fin] = np.minimum(1.0, med / np.maximum(dis[fin],
                                                                 1e-12))
        dt_stats = None
        if self.inverse_depth and os.environ.get("XTEN_GBA_DEPTH_TRUST") == "1" and prior_id is not None:
            prior_id, dt_stats = depth_trust_prior(prior_id, self.rig.fB)
        gba_dump = None
        if self.inverse_depth:
            from xclean.backend.global_ba import global_ba_id
            inert = self._build_inertial(poses) if self.inertial else None
            # T1-h (XTEN_GBA_ROTF=nodup): when the inertial term is in the
            # solve, the online rotation factor of every interval that a
            # preintegration factor covers is the same gyro counted twice;
            # `pres` is None whenever coupling is off or refused, and the
            # default mode hands the very same list over (see gba_rotf_dedup)
            rotf, rotf_prov = gba_rotf_dedup(
                rotf, None if inert is None else inert["pres"],
                getattr(self, "gba_rotf_mode", None) or gba_rotf_mode())
            # ONE argument set feeds the dump and the call, so a later kwarg
            # cannot be missing from the dump; with every flag unset the solver receives exactly the landed arguments
            _gargs = (kf_ids, poses, rho_state, m, hidx, okf, inv, ouv, self.rig)
            _gkw = dict(odisp=(odisp if self.stereo_obs else None),
                        prior=prior_id, rotf=rotf, welds=welds, iters=iters,
                        rotf_huber=(HUBER_K if self.gba_robust_gyro else None),
                        want_cov=200, rho_floor=self.rig.rho_floor,
                        inertial=inert, scale_bias=self.scale_bias,
                        vt_weights=vt_w)
            # XTEN_GBA_SCALECAL: only with the inertial term in the solve (one global offset state carried jointly
            # with beta, beta free); otherwise (None, None) and the call is the landed one
            from xclean.backend.global_ba import scalecal_build
            _sc_dof, _sc_sb = scalecal_build(len(hidx), inert is not None)
            if _sc_dof is not None:
                _gkw["dof"] = _sc_dof
                _gkw["scale_bias"] = _sc_sb
            _droot = getattr(self, "gba_dump_root", None)
            if _droot:
                # XTEN_GBA_DUMP=<dir>: pickle EXACTLY what the solver below is handed (post-rotf-dedup, post-depth-
                # trust), one file per DRAW named after the run's own out dir; never raises (an error is recorded)
                from xclean.backend.gba_dump import dump_gba_entry
                _kd = getattr(self.kit, "dir", None) if self.kit else None
                _tag = (os.path.basename(os.path.dirname(os.path.abspath(_kd)))
                        if _kd else "run")
                gba_dump = dump_gba_entry(
                    _droot, _tag, _gargs, _gkw, np.array([r[2] for r in self.all_kf]),
                    time_budget_s, ("iterbudget" if _itb else "wall"),
                    extra={"depth_trust": dt_stats,
                           "scalecal": getattr(self, "gba_scalecal_mode", None)})
            out_poses, out_rho, out_m, st = global_ba_id(
                *_gargs, **_gkw, time_budget_s=tb_solver, verbose=False)
        else:
            # T1-h: the Euclidean path carries no inertial term (nothing is
            # covered); the same provenance is emitted for the record
            rotf, rotf_prov = gba_rotf_dedup(
                rotf, None, getattr(self, "gba_rotf_mode", None) or gba_rotf_mode())
            out_poses, out_pts, st = global_ba(
                kf_ids, poses, pts, okf, inv, ouv, self.rig,
                prior=(c, r0, isg, val), rotf=rotf, welds=welds, iters=iters,
                rotf_huber=(HUBER_K if self.gba_robust_gyro else None),
                want_cov=200,
                time_budget_s=tb_solver, verbose=False)
        if isinstance(st, dict):
            if dt_stats is not None:
                st["depth_trust"] = dt_stats
            st["budget_s"] = float(time_budget_s)
            st["budget_rule"] = self._gba_budget_rule()
            st["truncated"] = bool(st.get("seconds", 0.0)
                                   >= 0.98 * float(time_budget_s))
            if _itb:
                # P6: no wall budget reached the solver, so nothing was cut by
                # one; whether each pass converged or ran out of iterations is
                # in st["iterbudget"] (written by global_ba_id)
                st["truncated"] = False
                st["budget_mode"] = "iterbudget"
            if gba_dump is not None:
                st["gba_dump"] = gba_dump          # XTEN_GBA_DUMP provenance (path, bytes, seconds | error)
            # T1-h provenance (rotf_mode, rotf_coupled, rotf_total,
            # rotf_covered, rotf_dropped_covered, rotf_kept_uncovered,
            # rotf_kept_nonconsecutive, rotf_handed): default-on, so a
            # default run reports how many factors the rule would drop
            st.update(rotf_prov)
        for i, rec in enumerate(self.all_kf):
            rec[3] = out_poses[i]
        if self.inverse_depth:
            self.lm.col("invd")[uniq] = out_rho
            self.store.m[uniq] = out_m
            # the host cache and the world cache both follow the trajectory
            # this solve just wrote - neither is a state, both are derived
            self._sync_host_cache()
            self._refresh_world(uniq)
            for k in self.kfs:
                T = self.state.kf_pose.get(k.id)
                if k.id < len(self.all_kf):
                    k.T_wc = self.all_kf[k.id][3]
            for kid in list(self.state.kf_pose):
                if kid < len(self.all_kf):
                    self.state.kf_pose[kid] = self.all_kf[kid][3]
        else:
            self.store.xyz[uniq] = out_pts
        if verbose:
            print(" [global BA] %d KF, %d landmarks, %d obs, %d welds, %d gyro:"
                  " %.3f -> %.3f px in %.1f s"
                  % (st["keyframes"], st["landmarks"], st["observations"],
                     len(welds), len(rotf), st["rms_px_before"],
                     st["rms_px_after"], st["seconds"]))
            if st.get("measured_disparity_sigma_px"):
                print("             disparity sigma: DECLARED %.2f px, "
                      "MEASURED %.2f px; signed BIAS %+.3f px"
                      % (st["declared_disparity_sigma_px"],
                         st["measured_disparity_sigma_px"],
                         st.get("measured_disparity_bias_px") or 0.0))
            if st.get("rejected"):
                print("             rejected %d observations (%.1f %%) above "
                      "%.2f px, then re-solved"
                      % (st["rejected"], st["rejected_pct"],
                         st.get("reject_threshold_px", float("nan"))))
        # the estimator's OWN covariance, exported so the consistency check
        # of plan 8.4 is a measurement rather than an assertion
        cov = st.pop("cov", {}) or {}
        if cov and self.kit is not None:
            import json as _json
            import os as _os
            blocks = {str(int(kf_ids[i])): cov[i].tolist() for i in cov
                      if i < len(kf_ids)}
            _os.makedirs(self.kit.dir, exist_ok=True)
            _json.dump({"meta": {"source": "global BA reduced camera system, "
                                 "structure marginalised out",
                                 "dof_order": "[rho(3), phi(3)], left "
                                 "perturbation",
                                 "sampled": len(blocks), "of": len(kf_ids)},
                        "blocks": blocks},
                       open(_os.path.join(self.kit.dir,
                                          "pose_covariance.json"), "w"))
        st["cov_exported"] = len(cov)
        self.gba_stats = st
        return st

    def _build_inertial(self, poses):
        """Preintegrate every keyframe interval and initialise the inertial
        states, so the global solve can be tightly coupled.

        Returns None rather than guessing when the IMU is absent, when the
        bias has never been estimated, or when the linear initialisation does
        not produce a credible gravity - a wrong gravity is worse than none,
        and this project has already measured what that costs (ORB-SLAM3's MAP
        init rejected 24 of 27 attempts with gravity ~17 degrees off).
        """
        if self.imu is None or not self.imu.available():
            return None
        from xclean.backend.preint import (preintegrate, inertial_init,
                                             gravity_from_accel_mean,
                                             GRAVITY_MAG, inert_gate_on,
                                             rotation_consistency, INERT_GATE_K)
        T_cb = self.imu.T_cb
        nk = len(poses)
        if nk < 8:
            return None
        T_wb = np.stack([poses[i] @ T_cb for i in range(nk)])
        bg0 = np.array(self.imu.bias, float)
        pres = []
        for i in range(nk - 1):
            pres.append(preintegrate(self.imu, self.all_kf[i][2],
                                     self.all_kf[i + 1][2], bg0,
                                     np.zeros(3)))
        if hasattr(self.imu, "note_gba_preint"):
            # T1-c provenance (read-only): intervals preintegrated / None,
            # and keyframes with no inertial row on either side
            self.imu.note_gba_preint(
                nk - 1, sum(p is None for p in pres),
                sum(1 for i in range(nk)
                    if (i == 0 or pres[i - 1] is None)
                    and (i == nk - 1 or pres[i] is None)))
        if sum(p is not None for p in pres) < 0.8 * (nk - 1):
            if self.verbose:
                print(" ! [imu] too many gaps to preintegrate: inertial "
                      "coupling stays OFF for this run")
            return None
        # ★ XTEN_GBA_INERT_GATE=1: refuse the coupling when the gyro is grossly
        # inconsistent with the entering ROTATIONS (k_R, preint.rotation_consistency,
        # > INERT_GATE_K), before the gravity / linear init so a refused run pays
        # nothing more.  The rank-deficient (n < 2) factors are already None here
        # (the gate implies the SAMPLES rule in `preintegrate`).
        if inert_gate_on():
            rc = rotation_consistency(pres, T_wb[:, :3, :3], bg0,
                                      self.inertial_inflate)
            if rc is None:
                self.inertial_gate = {"K": INERT_GATE_K, "refused": False,
                                      "evaluated": False, "samples_implied": True}
            else:
                self.inertial_gate = dict(rc, K=INERT_GATE_K,
                                          refused=bool(rc["k_R"] > INERT_GATE_K),
                                          evaluated=True, samples_implied=True)
            if self.inertial_gate["refused"]:
                if self.verbose:
                    print(" ! [imu] gyro inconsistent with the trajectory (k_R %.1f "
                          "> %.0f): inertial coupling stays OFF for the global BA"
                          % (rc["k_R"], INERT_GATE_K))
                self.inertial_diag = {"refused": True,
                                      "refused_reason": "rotation consistency",
                                      "rot_consistency": self.inertial_gate}
                return None
        # ★ GRAVITY FIRST, FROM THE MEAN SPECIFIC FORCE.
        # It uses pose ROTATIONS only, so keyframe position noise never enters
        # it: measured 0.35 % on V1_01 against 7-27 % for the least-squares
        # form, and stable across thirds of the run. With gravity supplied,
        # the linear solve below has three fewer unknowns and the two that are
        # left - velocity and accel bias - are the well-conditioned ones.
        gm = gravity_from_accel_mean(self.imu, T_wb[:, :3, :3],
                                     [r[2] for r in self.all_kf[:nk]])
        if gm is None:
            return None
        g_w, gmag = gm
        if len(self.all_kf) > 2000:
            print(" [imu-init] solving %d keyframes (%s route)"
                  % (len(self.all_kf),
                     "sparse-lsqr" if len(self.all_kf) > 5000 else "dense"),
                  flush=True)
        out = inertial_init(pres, T_wb[:, :3, :3], T_wb[:, :3, 3],
                            fix_magnitude=False, verbose=self.verbose,
                            g_fixed=g_w)
        if out is None:
            return None
        _g, vel, ba, diag = out
        diag["g_source"] = "mean specific force"
        diag["g_magnitude"] = gmag
        # ★ THE GATE IS THE MAGNITUDE, AND IT IS A REAL TEST.
        # |g| is not a free parameter of the universe: if the linear solve
        # returns something far from 9.81 then the accelerometer, the
        # extrinsics or the visual scale disagree, and coupling them tightly
        # would propagate that disagreement into the trajectory.
        if not (0.85 * GRAVITY_MAG < gmag < 1.15 * GRAVITY_MAG):
            if self.verbose:
                print(" ! [imu] inertial init returned |g| = %.3f m/s^2 "
                      "(%.1f %% off): REFUSED, coupling stays OFF"
                      % (gmag, 100 * (gmag / GRAVITY_MAG - 1)))
            self.inertial_diag = dict(diag, refused=True)
            return None
        self.inertial_diag = dict(diag, refused=False)
        g_w = g_w / gmag * GRAVITY_MAG
        return {"pres": pres, "vel": vel, "g_w": g_w, "bg": bg0, "ba": ba,
                "n_refused": int(sum(1 for q in pres if q is None)),
                "T_cb": T_cb, "inflate": self.inertial_inflate,
                "chi_cal": self.preint_cal}

    def flush(self):
        """End of run: everything still in the working set retires now, so the
        record is complete even for landmarks the run never culled."""
        alive = sorted(self.state.lm_kfs.keys())
        self._retire_landmarks(alive, reason=0)
        if self.decision_log:
            self._dl_close()                # F0: the per-frame decision log is complete

    def rehost(self, corrected):
        """A loop correction moved keyframes; the landmarks follow their HOST.

        This is the payoff of the host-relative record (plan 1b): a landmark is
        a bearing and an inverse depth in ITS OWN keyframe, so moving the
        keyframe moves the landmark exactly, with no re-triangulation and no
        staleness.  A world-position store would have had to be rebuilt.
        """
        if not corrected:
            return 0
        # ★ VECTORISED.  The first version looped over corrected keyframes
        # and did `nonzero(host == k)` inside, which is O(keyframes x
        # landmarks): it measured **22.4 ms p95** against a 5 ms bar on a
        # 650-keyframe run.  The same work as a gather over a per-keyframe
        # delta array is one pass.
        host = self.lm.col("host")
        n = min(len(host), self.store.n)
        if n == 0:
            return 0
        kmax = int(host[:n].max()) + 1
        R = np.tile(np.eye(3), (kmax, 1, 1))
        t = np.zeros((kmax, 3))
        touched = np.zeros(kmax, bool)
        # the per-keyframe delta, built with ONE batched inverse and one
        # batched product instead of 650 four-by-four solves in Python
        snap = self._kf_pose_snapshot
        ids = np.fromiter((k for k in corrected
                           if k < kmax and k in snap), np.int64)
        if len(ids):
            Tn = np.stack([corrected[int(k)] for k in ids])
            To = np.stack([snap[int(k)] for k in ids])
            delta = Tn @ se3_inv(To)
            R[ids] = delta[:, :3, :3]
            t[ids] = delta[:, :3, 3]
            touched[ids] = True
        h = host[:n]
        sel = np.nonzero(touched[h])[0]
        for rec in self.all_kf:
            T = corrected.get(rec[0])
            if T is not None:
                rec[3] = T
        # ★ THE HOST CACHE IS PART OF THE STATE, SO IT MOVES WITH IT.
        # A keyframe that has left the estimator window still hosts
        # landmarks, and its pose is read from `host_pose_cache` when they are
        # re-derived.  Correcting the trajectory without correcting that cache
        # would leave every landmark hosted before the loop closure anchored
        # to the pre-closure trajectory - the exact staleness the host-relative
        # record exists to prevent.
        self._sync_host_cache(corrected)
        if len(sel):
            if self.inverse_depth:
                # ★ EXACT, not incremental.  A host-relative landmark IS a
                # function of its host pose, so the corrected position is a
                # re-derivation, not a delta applied to a stale world point.
                # The delta form below is only correct while the world point
                # still lies on its own bearing ray, which a Euclidean bundle
                # adjustment does not guarantee.
                self._refresh_world(sel)
            else:
                hs = h[sel]
                P = self.store.xyz[sel]
                self.store.xyz[sel] = np.einsum("nij,nj->ni", R[hs], P) \
                    + t[hs]
        moved = int(len(sel))
        # (the loop above is over keyframe RECORDS, not landmarks - it is
        # O(keyframes) and stays; the landmark pass above is the one that was
        # O(keyframes x landmarks))
        for k in self.kfs:
            T = corrected.get(k.id)
            if T is not None:
                k.T_wc = T
        newest = corrected.get(self.all_kf[-1][0])
        if newest is not None:
            self.T_wc = newest.copy()
        return moved

    def _absorb(self):
        """Read the optimised poses back out of the estimator state.

        The BA owns the poses while it runs; the tracker reads them afterwards.
        This is the transaction boundary of plan 5.6 - "which subsystem moved
        this pose" stays an answerable question because there is exactly one
        place where the answer changes.
        """
        for k in self.kfs:
            T = self.state.kf_pose.get(k.id)
            if T is not None:
                k.T_wc = T
        for rec in self.all_kf[-self.state_window:]:
            T = self.state.kf_pose.get(rec[0])
            if T is not None:
                rec[3] = T
        newest = self.state.kf_pose.get(self.all_kf[-1][0])
        if newest is not None:
            self.T_wc = newest.copy()

    def _kp_refine_disparity(self, left, right, kp, d):
        """Per-keypoint ZNCC subpixel refinement of the dense disparity.

        ★ THE DENSE FAMILY HAS A CORNER BIAS AND IT IS MEASURED, NOT ASSUMED.
        diag/kp_disparity.py: dense-vs-correlation at corners is +0.40 %
        (proportional) on v1_02's near field - the RIGHT SIGN and ~half the
        magnitude of its -1 % under-scale - and +0.94 % in KITTI's far band.
        ORB-SLAM2's estimator (whose published rows carry no -1 %) measures
        disparity by patch correlation at the keypoint itself; this method
        is that mechanism, seeded by the dense value: ZNCC over +-3 px
        around the dense disparity, 11x11 patch, parabola subpixel fit.
        The refined value replaces the dense sample only where the peak is
        confident (>= 0.5), interior, and within 1.5 px of the seed - the
        dense value is the fallback, never the reverse.

        Vectorised: one (N,11,11) template gather + 7 candidate gathers.
        Runs inside the `landmark_creation` ledger stage, whose T1 bar
        (5.0 ms) gates it per test series rule - a budget violation fails the
        test series, which is the honest form of "is it cheap enough".
        """
        H, W = left.shape[:2]
        if left.ndim == 3:
            left = left.mean(axis=2)
        if right.ndim == 3:
            right = right.mean(axis=2)
        half, search = 5, 3
        n = len(kp)
        out = d.copy()
        ui = np.round(kp[:, 0]).astype(np.int64)
        vi = np.round(kp[:, 1]).astype(np.int64)
        d_int = np.round(d).astype(np.int64)
        ok = ((d > 0) & (vi >= half) & (vi < H - half)
              & (ui >= half) & (ui < W - half)
              & (ui - d_int - search >= half)
              & (ui - d_int + search < W - half))
        if not ok.any():
            return out
        idx = np.nonzero(ok)[0]
        uu, vv, dd0 = ui[idx], vi[idx], d_int[idx]
        win = np.arange(-half, half + 1)
        rows = vv[:, None, None] + win[None, :, None]
        cols = uu[:, None, None] + win[None, None, :]
        tpl = left[rows, cols].astype(np.float32)
        tm = tpl.mean(axis=(1, 2), keepdims=True)
        ts = tpl.std(axis=(1, 2))
        tpl = tpl - tm
        scores = np.full((len(idx), 2 * search + 1), -2.0, np.float32)
        for k in range(2 * search + 1):
            x = uu - (dd0 - search + k)
            colsr = x[:, None, None] + win[None, None, :]
            pat = right[rows, colsr].astype(np.float32)
            pm = pat.mean(axis=(1, 2), keepdims=True)
            ps = pat.std(axis=(1, 2))
            num = ((tpl * (pat - pm)).mean(axis=(1, 2)))
            den = np.maximum(ts * ps, 1e-6)
            scores[:, k] = num / den
        good_std = ts >= 1.0
        k_best = np.argmax(scores, axis=1)
        interior = (k_best > 0) & (k_best < 2 * search) & good_std
        r_ = np.arange(len(idx))
        y1 = scores[r_, k_best]
        y0 = scores[r_, np.maximum(k_best - 1, 0)]
        y2 = scores[r_, np.minimum(k_best + 1, 2 * search)]
        denom = y0 - 2 * y1 + y2
        with np.errstate(divide="ignore", invalid="ignore"):
            off = np.where(denom < -1e-9, 0.5 * (y0 - y2) / denom, 0.0)
        d_ref = (dd0 - search + k_best) + off
        accept = interior & (y1 >= 0.5) & (np.abs(d_ref - d[idx]) <= 1.5)
        out[idx[accept]] = d_ref[accept]
        self._kp_refined = int(accept.sum())
        self._kp_refine_total = getattr(self, "_kp_refine_total", 0) \
            + int(accept.sum())
        return out

    def _kp_disparity(self, kp, disp, left=None, right=None, count=True, sel=None):
        """xclean: the disparity MEASUREMENT at each keypoint -> (d (N,), on_edge (N,) bool); d = 0 = no measurement.
        Pure apart from the provenance counters (count=False for the bootstrap's dry run).  The measurement itself is
        stereo/epipolar.keypoint_disparity, shared with the loop worker's rich keypoints.

        ★ BILINEAR, AND WITH AN OCCLUSION CHECK: a bilinear sample across a depth discontinuity is a blend
        of two surfaces, so the four neighbours must agree within 6 x DISPARITY_NOISE_PX.
        ★ xclean DEFECT (d): THE DENSE MAP HAS HOLES.  sgm32 returns 0 where the LR check, the uniqueness test or the
        search range rejected a pixel; a bilinear sample next to one is a blend with ZERO - the disparity under-read,
        the depth OVER-read (~3x for d < 3 px, where the spread test cannot fire).  Any invalid neighbour = no
        measurement.
        ★ xclean: THE KEYPOINT'S DISPARITY IS MEASURED AT THE KEYPOINT (stereo/epipolar.py): a full epipolar row
        search (ZNCC) + affine 1-D LK sub-pixel, accepted where it agrees with SGM within the SGM's own LR threshold
        or, where SGM has no value, passes its own right-to-left check (D2).  XCLEAN_KP_DISP=sgm keeps the SGM sample.
        """
        from xclean.stereo.epipolar import keypoint_disparity, EpipolarMatcher
        if self.kp_disp_mode in ("epipolar", "hybrid") and self.epi is None and left is not None:
            self.epi = EpipolarMatcher()
        from xclean.core.derive import DEPTH_ERR_FRACTION
        d, on_edge, quad_valid, n_ok = keypoint_disparity(kp, disp, left, right, self.epi,
                                                          6.0 * DISPARITY_NOISE_PX, mode=self.kp_disp_mode, sel=sel,
                                                          sgm_min_disp=DISPARITY_NOISE_PX / DEPTH_ERR_FRACTION)
        self._kp_sigma = keypoint_disparity.last_sigma          # per keypoint LK sigma, or None (sgm mode)
        if count:
            self.n_kp_quad_invalid += int((~quad_valid).sum())
            self.n_kp_sgm_valid += int((quad_valid & ~on_edge).sum())
            self.n_kp_epi_ok += n_ok
        return d, on_edge

    def _create(self, kp, disp, kf_id, assoc, left=None, right=None):
        """Landmarks from disparity at the keypoints, inside the DERIVED band.

        A keypoint that already matched an existing landmark INHERITS it;
        only the unmatched ones create new structure.  Duplicating a landmark
        per keyframe would leave the covisibility graph empty and the backend
        with nothing to solve.
        """
        d, on_edge = self._kp_disparity(kp, disp, left, right, sel=(np.asarray(assoc) < 0))
        # arm --kp-refine: replace the dense sample with the per-keypoint
        # correlation where confident (see _kp_refine_disparity).  Runs
        # before the band gate so the refined depth decides the band, and
        # before _last_disp so re-observation rows carry the refined value.
        if self.kp_refine and left is not None and right is not None:
            d_in = np.where(on_edge, 0.0, d)
            if self.kp_refine == "far":
                # only the far band: z > max_depth_m  <=>  d < fB/max_depth
                d_far_thr = self.rig.fB / max(self.rig.max_depth_m, 1e-6)
                d_in = np.where(d_in < d_far_thr, d_in, 0.0)
                d_out = self._kp_refine_disparity(left, right, kp, d_in)
                d = np.where((d_in > 0), d_out, d)
            else:
                d = self._kp_refine_disparity(left, right, kp, d_in)
        # a disparity on an occlusion boundary is a blend of two surfaces and
        # is not a measurement of either - the same test that gates landmark
        # CREATION gates the re-observation constraint
        if self.stereo_overlap:
            # Class B (geometry): the match must lie inside the right image. Rosario's near ground (1-2 m at
            # f*B 41.7) puts 20-40 px of the LEFT image outside the right one - always the same side, which is
            # the shape a one-signed yaw drift needs. Zeroing d removes the point from creation AND from the
            # re-observation constraint, exactly as an occlusion edge is removed.
            no_ov = (d > 1e-6) & ((kp[:, 0] - d) < 0.0)
            self.n_kp_disp += int((d > 1e-6).sum())
            self.n_kp_no_overlap += int(no_ov.sum())
            d = np.where(no_ov, 0.0, d)
        self._last_disp = np.where(on_edge | (d <= 1e-6), 0.0, d)
        z = np.where(d > 1e-6, self.rig.fB / np.maximum(d, 1e-6), np.inf)
        # ★ THE BAND IS REPRESENTABILITY, NOT QUALITY (plan 5.4).
        # `near_clip` is where disparity SATURATES - below it no depth is
        # representable at all.  `far_bearing` is where disparity falls under
        # a pixel - above it no depth is MEASURABLE.  In between, depth is
        # measurable, and the honest statement of how well is the constant
        # `sigma_rho = sigma_d / (f*B)`, because `rho = d/(f*B)` is LINEAR in
        # disparity: a far landmark is not a worse measurement in inverse
        # depth, it is the same measurement.
        #
        # `max_depth_m = 0.2*f*B` is where the DEPTH error passes 10 %, which
        # is a quality threshold and the right one for a depth image - but
        # using it as the landmark band throws away every distant point, and
        # distant points are what constrain ROTATION.  On KITTI it would
        # discard everything past 76 m of a scene that runs to 380.
        # This is the "far points as bearings" clause: OKVIS2 gets it for free
        # because d(u)/d(t) ~ 1/Z does the job with no heuristic, and the
        # inverse-depth prior does the same job here.
        if self.far_band:
            good = (d > 0) & (z >= self.rig.near_clip_m) \
                & (z <= self.rig.far_bearing_m) & (~on_edge)
        else:
            good = (d > 0) & (z >= self.rig.near_clip_m) \
                & (z <= self.rig.max_depth_m) & (~on_edge)
        lm_id = np.array(assoc, dtype=np.int64)
        good &= (lm_id < 0)                     # inherited keypoints keep theirs
        if not good.any():
            return lm_id
        idx = np.nonzero(good)[0]
        x = (kp[idx, 0] - self.rig.cx) * z[idx] / self.rig.fx_rect
        y = (kp[idx, 1] - self.rig.cy) * z[idx] / self.rig.fx_rect
        Pc = np.stack([x, y, z[idx]], axis=1)
        Pw = (self.T_wc[:3, :3] @ Pc.T).T + self.T_wc[:3, 3]
        rng_ = np.linalg.norm(Pc, axis=1)
        zc = np.maximum(z[idx], 1e-9)
        # ★ THE BEARING IS ON THE NORMALISED PLANE, NOT THE UNIT SPHERE.
        # `m = Pc/z = (x_n, y_n, 1)` pairs with an inverse DEPTH, which is
        # what the disparity measured (`rho = d/(f*B)`).  A UNIT bearing pairs
        # with an inverse RANGE, and mixing the two places every landmark at
        # `range * b_z` instead of `range` - measured on the V1_01 kit as a
        # median 14.3 %, p90 29.8 %, max 42.9 % pull toward the host camera.
        # Storing what the consumer reads is the whole repair.
        m = Pc / zc[:, None]
        base = self.store.add(Pw, m)
        self._store_descriptors(base, idx)
        # xclean provenance: landmarks created per depth band - the stereo band where a 0.5 px
        # disparity error is <= 10 % of depth (z <= max_depth_m = 0.2 f*B) and the far band beyond it (to f*B / 1 px)
        nb = int((zc <= self.rig.max_depth_m).sum())
        self.lm_band["stereo_band"] = self.lm_band.get("stereo_band", 0) + nb
        self.lm_band["far_band"] = self.lm_band.get("far_band", 0) + int(len(zc) - nb)
        rho_d = 1.0 / zc
        # the direction-only threshold: XTEN_FAR_DEPTH=prior/both (z_far) takes precedence over XTEN_FAR_BEARING_ONLY
        # (max_depth_m); neither set -> None -> the original expressions, bit-identical
        zthr = self.far_depth_z if self.far_depth in ("prior", "both") else \
            (self.rig.max_depth_m if self.far_bearing_only else None)
        inv_s = np.full(len(idx), 1.0 / self.rig.rho_sigma)
        if zthr is not None:
            inv_s = np.where(zc > zthr, 0.0, inv_s)
        sig_kp = getattr(self, "_kp_sigma", None)
        if self.disp_sigma_mode == "measured" and sig_kp is not None:
            # ARM XCLEAN_DISP_SIGMA=measured: each landmark's inverse-depth information comes from
            # ITS OWN measured disparity sigma (the keypoint LK covariance, stereo/epipolar.py), not the declared
            # 0.5 px; and the stereo prior is kept only where that sigma is <= DEPTH_ERR_FRACTION (10 %) of the
            # disparity - the system's own definition of a usable depth - otherwise the landmark is a BEARING (no depth
            # prior; multi-view triangulation), the ORB-SLAM3 ThDepth idea with the threshold measured per point.
            from xclean.core.derive import DEPTH_ERR_FRACTION
            sk = np.asarray(sig_kp, float)[idx]
            dk = self.rig.fB / zc
            usable = np.isfinite(sk) & (sk > 0) & (sk <= DEPTH_ERR_FRACTION * dk)
            inv_s = np.where(usable, self.rig.fB / np.maximum(sk, 1e-6), 0.0)
            self.n_lm_measured_sigma += int(usable.sum())
            self.n_lm_bearing_only += int((~usable).sum())
            self.sigma_samples.extend(sk[np.isfinite(sk)][:64].tolist())
        self.lm.add(host=np.full(len(idx), kf_id, np.int64),
                    invd=rho_d, invd0=rho_d,
                    invrange=1.0 / rng_,
                    invsig=inv_s,
                    invsig_r=inv_s * (rng_ / zc),
                    obs=np.ones(len(idx), np.int64),
                    mx0=m[:, 0], my0=m[:, 1])
        if self.far_bearing_only:
            self.n_lm_far_bearing_only += int((zc > self.rig.max_depth_m).sum())
        if self.far_depth in ("prior", "both"):
            self.n_lm_far_depth += int((zc > zthr).sum())
        lm_id[idx] = base + np.arange(len(idx))
        self._publish_landmark_views()
        return lm_id

    def _publish_landmark_views(self):
        """VIEWS, not conversions - the whole point of `_Cols`.

        Re-taken after every `add` because the columns grow by doubling and a
        stale view points at the old buffer.
        """
        st = self.state
        st.lm_host = self.lm.col("host")
        st.lm_invsig = self.lm.col("invsig")
        st.lm_rho_state = self.lm.col("invd")     # the ESTIMATOR's variable
        st.lm_rho_meas = self.lm.col("invd0")     # the frozen MEASUREMENT
        st.lm_m = self.store.m[:self.store.n]      # a WRITABLE view
        st.lm_m0 = np.stack([self.lm.col("mx0"), self.lm.col("my0")], axis=1)
        # the Euclidean ablation reads the inverse-RANGE pair, and it must be
        # BOTH columns: `invrange` with `invsig` (an inverse-DEPTH sigma) is
        # the same units mismatch one level up
        st.lm_rho = self.lm.col("invrange")
        st.lm_invsig_r = self.lm.col("invsig_r")

    def _refresh_world(self, ids):
        """`store.xyz` is a CACHE of a DERIVED quantity, and this is where it
        is re-derived.

        Under inverse depth a world position is not a variable - it is
        `T_wc_host * (m / rho)`.  Everything outside the estimator (PnP, the
        local-map projection, the loop weld, the median-depth dial) still
        wants a world point, so the cache exists; but it is recomputed from
        the state rather than nudged, which is what makes a later loop
        correction of the host EXACT instead of approximately right.
        """
        ids = np.asarray(ids, np.int64)
        if not len(ids):
            return
        host = self.lm.col("host")[ids]
        rho = np.maximum(self.lm.col("invd")[ids], 1e-12)
        m = self.store.m[ids]
        T = self._host_poses(host)
        Ph = m / rho[:, None]
        self.store.xyz[ids] = np.einsum("nij,nj->ni", T[:, :3, :3], Ph) \
            + T[:, :3, 3]

    def _host_poses(self, host_ids):
        """T_wc of each host keyframe, from the CURRENT trajectory.

        `all_kf` is the authority: it is updated by the local BA absorb, by
        the loop correction and by the global BA, so a landmark re-derived
        from it can never be stale in the way a world-position store can.
        """
        out = np.empty((len(host_ids), 4, 4))
        out[:] = np.eye(4)
        n = len(self.all_kf)
        h = np.clip(host_ids, 0, max(n - 1, 0))
        if n:
            out[:] = np.stack([self.all_kf[int(k)][3] for k in h])
        return out

    def _sync_host_cache(self, corrected=None):
        """Push corrected poses into `state.host_pose_cache`.

        `corrected` is a {kf_id: T_wc} mapping; with None, every keyframe the
        cache already holds is refreshed from `all_kf`, which is the authority.
        """
        cache = self.state.host_pose_cache
        if corrected is not None:
            for k, T in corrected.items():
                if int(k) in cache:
                    cache[int(k)] = T
            return
        n = len(self.all_kf)
        for k in list(cache):
            if 0 <= k < n:
                cache[k] = self.all_kf[k][3]

    def _landmarks_hosted_by(self, kf_ids):
        """Landmark ids hosted by these keyframes - O(log N) per keyframe.

        `host` is NON-DECREASING because `_create` stamps the current keyframe
        id and keyframe ids increase monotonically, so each host owns a
        CONTIGUOUS slice.  That is what keeps the refresh bounded by the
        window rather than by the map, which plan 5.1 requires.
        """
        host = self.lm.col("host")
        if not len(host):
            return np.zeros(0, np.int64)
        out = []
        for k in kf_ids:
            a, b = np.searchsorted(host, [k, k + 1])
            if b > a:
                out.append(np.arange(a, b, dtype=np.int64))
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def _store_descriptors(self, base, idx):
        """One descriptor per landmark, taken from its HOST observation.

        xclean: a 32-byte ORB descriptor per landmark in a growable host array (8x smaller than the 256-float
        SuperPoint vector it replaces; the local-map match compares only the candidates inside each projected
        landmark's search radius, so there is nothing to gain from keeping it on the GPU).
        """
        d = self._pending_desc
        if d is None:
            return
        block = np.asarray(d)[np.asarray(idx, dtype=np.int64)]
        need = base + len(idx)
        if self.lm_desc is None:
            self.lm_desc = np.zeros((max(8192, need * 2), 32), np.uint8)
        elif need > self.lm_desc.shape[0]:
            g = np.zeros((max(self.lm_desc.shape[0] * 2, need + 4096), 32), np.uint8)
            g[:self.lm_desc_n] = self.lm_desc[:self.lm_desc_n]
            self.lm_desc = g
        self.lm_desc[base:need] = block
        self.lm_desc_n = need

    def _retire_landmarks(self, ids, reason=1):
        """A landmark leaves the estimator working set: RETIRED, not deleted.

        Appended to `landmarks.bin` in HOST-RELATIVE inverse-depth form, so a
        later loop correction of its host keyframe cannot stale it - the
        post-process re-derives world positions from the FINAL trajectory.
        The working set is bounded by the window; the RECORD is bounded by the
        trajectory; they are two different objects (plan 1b).
        """
        if not self.kit or not len(ids):
            return
        # ★ THE RECORD GETS THE OPTIMISED INVERSE DEPTH, NOT THE RAW ONE.
        # `invd` is the state the bundle adjustment refined; `invd0` is the
        # triangulation it started from.  Retiring `invd0` - which the first
        # version did, because one column was doing both jobs - threw away
        # every correction the backend made and handed the 3DGS build the
        # pre-backend cloud.
        # The variance is on INVERSE DEPTH, in (1/m)^2, because that is the
        # unit of the number beside it.  It used to be a DEPTH variance in
        # m^2 in a field named `inv_depth_var`.
        rows = []
        var = float(self.rig.rho_sigma) ** 2
        for j in ids:
            j = int(j)
            if j >= self.lm.n:
                continue
            m = tuple(float(x) for x in self.store.m[j])
            rows.append((int(self.lm.col("host")[j]), m,
                         float(self.lm.col("invd")[j]), var,
                         int(self.lm.col("obs")[j]), reason))
        self.kit.retire_many(rows)
        self.n_retired = getattr(self, "n_retired", 0) + len(rows)

    # ---- reporting -------------------------------------------------------
    def xclean_summary(self):
        jh = np.asarray(self.fe_join_hist) if self.fe_join_hist else np.zeros(1)
        out = {"klt_pred_src": dict(self.klt_pred_src),
               "join_median": float(np.median(jh)), "join_p10": float(np.percentile(jh, 10)),
               "desc_rescue_frames": int(self.n_desc_rescue), "desc_rescue_pairs": int(self.n_desc_rescue_pairs),
               "kp_disp_mode": self.kp_disp_mode, "kp_quad_invalid": int(self.n_kp_quad_invalid),
               "kp_sgm_valid": int(self.n_kp_sgm_valid), "kp_epi_ok": int(self.n_kp_epi_ok),
               "epipolar": (dict(self.epi.stats) if self.epi is not None else None),
               "bootstrap": getattr(self, "boot_log", None), "lm_band": dict(self.lm_band),
               "reacq": dict(self.reacq_stats) if self.reacq_on else None, "bindings": len(self.bind),
               "disp_sigma": {"mode": self.disp_sigma_mode, "with_prior": int(self.n_lm_measured_sigma),
                              "bearing_only": int(self.n_lm_bearing_only),
                              "sigma_px_p10_50_90": ([float(x) for x in np.percentile(self.sigma_samples,
                                                                                    [10, 50, 90])]
                                                     if self.sigma_samples else None)}}
        st = getattr(self.depth, "stats", None)
        if callable(st):
            out["sgm"] = st()
        return out

    def summary(self):
        inl = np.array(self.inlier_hist) if self.inlier_hist else np.zeros(1)
        return {"frames": self.n_frames, "keyframes": len(self.all_kf),
                "kf_rate": len(self.all_kf) / max(self.n_frames, 1),
                "depth_frames": self.depth_frames,
                "coasted": self.n_lost,
                "coasted_pct": 100.0 * self.n_lost / max(self.n_frames, 1),
                "median_inliers": float(np.median(inl)),
                "landmarks": int(self.store.n),
                "ba_solves": self.ba.n_solves if self.ba else 0,
                "ba_over_budget": self.ba.truncated if self.ba else 0,
                # the span controller decides the whole latency/accuracy
                # trade, so where it settled is a DIAL, not an internal
                "ba_span": self.ba._span if self.ba else None,
                "ba_fixed_cap": self.ba._fixed_cap if self.ba else None,
                "far_band": self.far_band,
                "inverse_depth": self.inverse_depth,
                "frozen_bearing": self.frozen_bearing,
                "stereo_overlap_arm": self.stereo_overlap,
                "far_bearing_only_arm": self.far_bearing_only,
                "lm_far_bearing_only": self.n_lm_far_bearing_only,
                "far_depth_arm": self.far_depth or None,
                "far_depth_z_m": self.far_depth_z,
                "lm_far_depth": self.n_lm_far_depth,
                "kp_with_disparity": self.n_kp_disp,
                "kp_no_right_overlap": self.n_kp_no_overlap,
                "kp_refined": getattr(self, "_kp_refine_total", 0),
                "inertial": self.inertial,
                "inertial_init": self.inertial_diag,
                "localmap_median": float(np.median(self.n_localmap))
                if self.n_localmap else 0.0,
                "loop": self.loop.summary() if self.loop else None,
                "loop_applied": self.n_applied,
                "age_consensus": self.age_consensus,
                "staged_anchored": self.n_staged_anchored,
                "staged_fallback": self.n_staged_fallback,
                "staged_union": self.n_staged_union,
                "staged_weak": self.n_staged_weak,
                "quarantined": self.n_quarantined,
                "rebase": self.n_rebase,
                "reloc": self.n_reloc,
                "reloc_attempts": self.n_reloc_attempts,
                "structure_obs": self.n_structure_obs,
                "motion_resets": self.n_motion_resets,
                "dt_cv": {"on": bool(self.dt_cv), "scaled_calls": int(self.n_dt_scaled)},
                "gyro_factors": self.n_gyro_factors,
                "gyro_factors_final": getattr(self, "n_gyro_final", 0),
                "global_ba": getattr(self, "gba_stats", None),
                "imu": self.imu.summary() if self.imu else None,
                "feature_budget": self.fe.dial_summary(),
                # ---- xclean: front end, stereo and their latency provenance
                "xclean": self.xclean_summary(),
                # ---- LOG (tier-1 programme): provenance, default-on ----
                "kf_triggers": dict(self.kf_trigger_counts),
                "kf_decisions": dict(self.kf_decision_counts),
                "frames_at_cap": int(self.n_frames_at_cap),
                "kf_evidence": [[int(k), v["inliers"], v["z_med"]]
                                for k, v in sorted(self.kf_evidence.items())],
                "kf_evidence_errors": int(self.n_kf_evidence_err),
                # ---- T1-b (XTEN_KF_HEALTH): which health rule decided ----
                "kf_health": self.kf_health_summary(),
                **({"front_gate": self.front_gate_summary()} if self._fg_active else {}),
                **({"pnp_verify": pnp_verify_summary()} if self.pnp_verify else {}),
                # ---- XTEN_GBA_INERT_GATE provenance: absent when the flag is unset
                **({"inertial_gate": getattr(self, "inertial_gate", None)}
                   if (getattr(self, "inert_gate", False)
                       or getattr(self, "inertial_gate", None) is not None) else {}),
                # ---- TRACK PACKAGE (XTEN_DECISION_LOG / XTEN_COAST_FEED): only when on ----
                **({"track_pkg": self.track_package_summary()}
                   if (self.decision_log or self.coast_feed) else {})}
