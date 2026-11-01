#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""LIVE SCORER: reads ONLY the RECORDER log (and the FEEDER release log), never engine files.

    python -m xclean.live.score_live DIR [--tag <sequence dir | row tag>] [--rows REGISTRY.json] [--gt GT.csv]
        [--family oak]                                             -> DIR/live_score.json

* B5 accuracy: LIVE-MAP stream, frame level, SE(3) Umeyama in cam0 over the row's FULL GT span (frames without a GT
  sample within 20 ms are excluded identically for every system), every frame of the sequence counted: a dropped or
  unpublished frame takes a ZERO-ORDER HOLD of the last LIVE-MAP pose published for an earlier frame.  No hygiene.
* B1 latency: L(k) = RECORDER receipt of LIVE-MAP(k) - FEEDER release(k), over ALL frames, +inf when dropped;
  frame-loop gap = the largest gap between consecutive LIVE-MAP receipts, in camera periods.
* B2 drops: count, %, longest consecutive run, worst 1 s window, latency drift (p95 of the last 10 % / first 10 %).
* B4 unsupported frames and LIVE-ODOM jumps; the 2-s relative test carries the bar's "3 x GT jitter" term (DT1,
  `gt_jitter`).
* GT frame (GTF1, `GT_FRAMES`): every row's GT is put into SCORED_FRAME = the rectified cam0 per family, from the
  producer's source (4Seasons GT is already cam0; MadMax GT position is the right GNSS antenna; EuRoC adds the
  rectification R1; Rosario v1 is position-only -> B4 rotation clause NOT APPLICABLE, translation at its lower bound,
  flagged).  `gt_frame="legacy"` reproduces the pre-GTF1 transform for audits only.
`score_frames(t, T, ...)` scores any per-frame trajectory the same way (the x1 causal baseline freeze).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------------------------------------------------
# GT FRAME.  The scorer compares every system in
# ONE frame: SCORED_FRAME, the RECTIFIED cam0 (xclean LIVE-MAP T_world_cam0 and the x1 kit, the engine
# FRAME_CONVENTION, both track on cv2.stereoRectify(calib.json) images).  A gt.csv states neither its frame nor where
# its position point sits (memory rule FRAME / CONVENTION), so each family's facts below were read from the PRODUCER'S
# source (our converter + the dataset's own GT-producing code / tools) and the dataset documentation.  Every gt.csv is
# t_ns,x,y,z,qw,qx,qy,qz (W FIRST) by construction of our converters (each one reorders explicitly; checked per
# family below), read by xclean.eval.score.load_gt / quat_to_R(qw, qx, qy, qz).
# Keys: calib.json "source" (the converter's own stamp).
#   rot   "body"      the GT rotation is R_world_body; cam0_rect = body o T_body_cam0 (calib.json) o R1^T, where R1 is
#                     cv2.stereoRectify's left rectification rotation (identity on rigs shipped rectified);
#         "cam0_rect" the GT rotation already IS the rectified cam0 (no extrinsic, no R1);
#         None        the GT has NO orientation (position-only): the 2-s rotation clause is NOT APPLICABLE and the
#                     GT position cannot be moved to cam0 (the lever needs the orientation) - both flagged.
#   point_in_body     where the GT POSITION sits, in the body frame (the body origin unless stated); the cam0 lever is
#                     T_body_cam0.t - point_in_body.
SCORED_FRAME = "T_world_cam0_rectified"
GT_FRAME_DEF = "gtf1"          # "legacy" = the pre-GTF1 transform (body o T_body_cam0 for every row), audit only
GT_FRAMES = {
    "EuRoC MAV": dict(
        rot="body", point_in_body=(0.0, 0.0, 0.0),
        gt="body = IMU pose T_RS (mav0/state_groundtruth_estimate0: q_RS_w first; its sensor.yaml T_BS = I); "
           "T_body_cam0 = cam0 sensor.yaml T_BS (euroc_to_oak.py); the engine tracks in R1-rectified cam0",
        source="EuRoC data.csv header + sensor.yaml; euroc_to_oak.py:179-188,273"),
    "KITTI odometry": dict(
        rot="cam0_rect", point_in_body=(0.0, 0.0, 0.0),
        gt="rectified left gray camera (P0) pose, devkit poses/XX.txt 3x4 row-major T_w_cam0; kitti_to_oak.py "
           "rot_to_quat_wxyz -> w first; T_body_cam0 = I, R_stereo = I",
        source="KITTI odometry devkit readme (poses); kitti_to_oak.py:1-16,92-102"),
    "4Seasons": dict(
        rot="cam0_rect", point_in_body=(0.0, 0.0, 0.0),
        gt="GNSSPoses.txt = the stereo VIO KEYFRAME camera pose fused with RTK-GNSS (camToWorld of the left camera "
           "on the shipped undistorted = rectified images), t,x,y,z,qx,qy,qz,qw,scale: fourseasons_to_oak.write_gt "
           "reorders to w first and multiplies xyz by scale (S-GTSCALE). NOT the IMU: calib T_body_cam0 must not be "
           "applied (pre-GTF1 it was: a constant 179.6 deg rotation and a 0.185 m rotating lever)",
        source="libartipy (Artisense, the 4Seasons producer): dataset_slam.set_keyframe_poses_to_gps_poses assigns "
               "GPSPose.pose directly as the keyframe camera pose (keyframe.py 'camToWorld'), gps_pose.py column "
               "order xyz + quat xyzw; fourseasons_to_oak.py:45-79"),
    "MADMAX": dict(
        rot="body", point_in_body=(0.07, -0.6425, 0.13),
        gt="gt_6DoF_gnss_and_imu.csv: ORIENTATION = body B (x fwd, y left, z up; = IMU axes, tf__imu_to_B rotation "
           "= I), Hamilton w first, R_ENU_B; POSITION = the RIGHT GNSS ANTENNA (posEnuRight, no lever applied), "
           "a forward constant-velocity KF at 100 Hz on 1 Hz RTK fixes (M-GTSAW). Right antenna in the IMU/body "
           "frame = tf__imu_to_B (0.07, 0, 0.13) + tf__B_to_R (0, -0.6425, 0), identical in the A-G calibration "
           "zips; T_body_cam0 = tf__imu_to_camera_left (camera_rect_left_info R = I, so R1 = I)",
        source="00_gnss_gt_computation_matlab.zip RUN_GroundTruthEstimation.m (6DoF position filter observes "
               "posEnuRight; baselineBodyIMU2Right is defined but never used; euler_2_quat ZYX Hamilton w first, "
               "DCM_2_euler of RIB = [eB1 eB2 eB3] = R_ENU_B); <loc>_calibration.zip tf__*.csv; madmax_to_oak.py",
        gt_faults=["M-GTSAW: 100 Hz positions are a forward CV-KF between 1 Hz RTK fixes (steps 0.09-0.20 m median); "
                   "B4 failures rise with the GT's age since the last fix",
                   "GT attitude inconsistent with the gyro over 2 s (> 1 deg in 34-82 % of the 2-s windows on the "
                   "sequences checked), leaking into cam0 positions through the 0.70 m lever"]),
    "Rosario v1": dict(
        rot=None, point_in_body=None,
        gt="sequenceNN_gt.txt (TUM t x y z qx qy qz qw) = RTK GPS ANTENNA POSITIONS, quaternion identity "
           "(no orientation); rosario_to_oak.py reorders to w first. Antenna sits at position_gps_baselink "
           "(1.80, -0.03, 1.59) vs the IMU at (1.96, -0.017, 1.02) in base_link (calibrationNN.yaml): a ~0.6 m, "
           "mostly vertical lever that cannot be applied without orientation",
        source="rosario_to_oak.py:1-22,197-206; calibrationNN.yaml gps/imu blocks"),
    "Rosario v2": dict(
        rot="body", point_in_body=(0.0, 0.0, 0.0),
        gt="pgt bag (MINS: PPK GNSS + RealSense D435i IMU, ENU) - our converter took the most-populated pose topic; "
           "the converted GT is 200 Hz with optical-convention axes (start roll ~109 deg = a z-forward frame pitched "
           "~19 deg down), i.e. the RealSense IMU / infra1 optical frame, not the paper's base_link. calib "
           "T_body_cam0 = I: the IMU-to-infra1 lever (cm) and rotation (< 1 deg by RealSense design) are not in our "
           "calib; the pgt bag is no longer on disk, so this is not re-verifiable from source",
        source="rosariov2_to_oak.py:131-152; arXiv 2508.21635 (ground truth = MINS, IMU frame -> base_link)"),
}


def rect_R1(cal):
    """cv2.stereoRectify's LEFT rectification rotation for a calib.json (x_rect = R1 x_raw), exactly the call the
    engines make (xclean/live/engine.py, xclean/io/dataset.py, the engine; R1 depends on R_stereo and
    T_stereo only)."""
    import cv2
    M1, D1, M2, D2 = (np.array(cal[k], float) for k in ("M1", "D1", "M2", "D2"))
    R1, *_ = cv2.stereoRectify(M1, D1, M2, D2, (int(cal["camera_width"]), int(cal["camera_height"])),
                               np.array(cal["R_stereo"], float), np.array(cal["T_stereo"], float).reshape(3, 1),
                               alpha=0)
    return np.asarray(R1, float)


def gt_frame_model(cal):
    """The GT-frame facts for a calib.json (GT_FRAMES entry + family key), or the pre-GTF1 assumption flagged
    UNVERIFIED for a source not in GT_FRAMES (malaga, dream and OAK rows: not on the dev set or the benchmark)."""
    src = cal.get("source")
    if src in GT_FRAMES:
        return dict(GT_FRAMES[src], family=src, verified=True)
    return dict(rot="legacy", point_in_body=(0.0, 0.0, 0.0), family=src, verified=False,
                gt="UNVERIFIED: pre-GTF1 assumption kept (GT = body pose, cam0 = body o T_body_cam0, no R1)",
                source="none read")


def gt_to_cam0(p, Rg, cal, model):
    """GT (positions p (N,3), rotations Rg (N,3,3) as read from gt.csv) -> (pc, Rc) of SCORED_FRAME per `model`.
    Rc is all-NaN when the GT has no orientation."""
    p = np.asarray(p, float)
    Rg = np.asarray(Rg, float)
    rot = model["rot"]
    if rot is None:
        return p.copy(), np.full((len(p), 3, 3), np.nan)
    if rot == "cam0_rect":
        return p.copy(), Rg.copy()
    T_bc = np.array(cal["T_body_cam0"], float)
    if rot == "legacy":
        return p + np.einsum("nij,j->ni", Rg, T_bc[:3, 3]), Rg @ T_bc[:3, :3]
    lever = T_bc[:3, 3] - np.asarray(model["point_in_body"], float)
    R_b_rect = T_bc[:3, :3] @ rect_R1(cal).T
    return p + np.einsum("nij,j->ni", Rg, lever), Rg @ R_b_rect


def _gt_load(tag, frame=GT_FRAME_DEF):
    sys.path.insert(0, REPO)
    from xclean.io import rows as R
    from xclean.eval.score import load_gt, quat_to_R
    r = R.get(tag)
    cal = json.load(open(os.path.join(r.dataset, "calib.json"), encoding="utf-8"))
    t_ns, p, q = load_gt(r.gt_path)
    tg = (t_ns - int(cal["t0_ns"])) / 1e9
    Rg = quat_to_R(q[:, 0], q[:, 1], q[:, 2], q[:, 3])           # gt.csv is w,x,y,z (every converter, GT_FRAMES)
    if frame == "legacy":
        model = dict(rot="legacy", point_in_body=(0.0, 0.0, 0.0), family=cal.get("source"), verified=False,
                     gt="pre-GTF1 transform (audit only)", source="-")
    else:
        model = gt_frame_model(cal)
    pc, Rc = gt_to_cam0(p, Rg, cal, model)
    return tg, pc, Rc, model, cal


def _gt_cam(tag, frame=GT_FRAME_DEF):
    """(tg, pc, Rc): GT stamps (s, calib t0), positions and rotations of SCORED_FRAME.  Rc is all-NaN for a
    position-only GT.  frame="legacy" reproduces the pre-GTF1 transform for the before/after audit ONLY."""
    tg, pc, Rc, _m, _c = _gt_load(tag, frame)
    return tg, pc, Rc


def gt_frame_meta(tag, frame=GT_FRAME_DEF):
    """What the scorer did with this row's GT (reported in every score); None for an unregistered tag."""
    try:
        sys.path.insert(0, REPO)
        from xclean.io import rows as R
        cal = json.load(open(os.path.join(R.get(tag).dataset, "calib.json"), encoding="utf-8"))
    except Exception:
        return None
    m = gt_frame_model(cal) if frame != "legacy" else dict(rot="legacy", point_in_body=(0, 0, 0), verified=False)
    out = {"def": frame, "scored_frame": SCORED_FRAME, "family": cal.get("source"), "gt_rot": m["rot"],
           "verified": bool(m.get("verified")), "gt_orientation": m["rot"] is not None,
           "gt_faults": list(m.get("gt_faults", []))}
    if m["rot"] in ("body", "legacy"):
        T_bc = np.array(cal["T_body_cam0"], float)
        lever = T_bc[:3, 3] - np.asarray(m["point_in_body"], float)
        out["lever_body_m"] = [float(x) for x in lever]
        out["R1_deg"] = (0.0 if m["rot"] == "legacy" else
                         float(_angle(rect_R1(cal)[None])[0]))
    return out


def _assoc(t_est, tg, max_dt=0.02):
    j = np.searchsorted(tg, t_est)
    j = np.clip(j, 1, len(tg) - 1)
    j = np.where(np.abs(tg[j - 1] - t_est) <= np.abs(tg[j] - t_est), j - 1, j)
    ok = np.abs(tg[j] - t_est) <= max_dt
    return j, ok


def _angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


# DT1: the "3 x GT jitter" term of the live benchmark rules B4.
# MAD_TO_SIGMA: class D, the Gaussian consistency factor of the median absolute deviation (1 / Phi^-1(3/4)).
# JITTER_STENCIL: class D, 5 consecutive GT samples = the smallest stencil whose residual is blind to motion up to
#   constant JERK (a cubic in time) for ANY sample spacing; the residual is then the 4th divided difference.
# B4_JITTER_MULT = 3 and B4 floors 0.10 m / 10 % / 2 deg: the live benchmark rules v1 B4, verbatim (the bar's own numbers).
MAD_TO_SIGMA = 1.482602218505602
JITTER_STENCIL = 5
B4_JITTER_MULT = 3.0
GT_JITTER_DEF = "dd4_native_mad"   # 4th divided difference over 5 native GT samples, unit-norm weights, 1.4826 x MAD


def gt_jitter(tg, pg, stride=1):
    """DT1: the GT's own per-sample position JITTER (class C: a statistic of the GT samples, identical for every
    system scored on the row).

    For each run of JITTER_STENCIL consecutive GT samples (t_k, p_k) the 4th DIVIDED DIFFERENCE weights
    w_k = 1 / prod_{l != k} (t_k - t_l) are formed and normalised to unit length; z = sum_k w_k p_k (per axis).
      * Motion: w annihilates every polynomial of degree <= 3 in t EXACTLY, for any (non-uniform) spacing, so smooth
        motion enters z only through its 4th derivative times ~dt^4 (checked per row with the stride diagnostic
        below, an internal note).
    It is computed on the cam0 GT positions the test compares against, so GT orientation jitter seen through the
    body-cam0 lever arm is part of it.
      * Jitter: for white per-sample noise of standard deviation s per axis, z has standard deviation EXACTLY s
        (|w| = 1).  So z is the GT's high-frequency residual, scaled so that its spread IS the per-sample jitter.
    Per axis: s_axis = MAD_TO_SIGMA x median|z - median z| (robust: stencils that straddle a GT gap or a GT glitch
    are a minority and do not move it; 50 % breakdown).  GT jitter = sigma_3d = sqrt(sum_axis s_axis^2), the per-
    sample 3-D RMS position jitter.  Why 3 x sigma_3d: IF the jitter is WHITE and Gaussian, a 2-s relative
    displacement read from the GT carries the jitter of BOTH ends, per axis s*sqrt(2); 3 x sigma_3d = 3*sqrt(3)*s =
    3.67 x (s*sqrt(2)), and P(chi_3 > 3.67) = 0.4 %, so white GT jitter alone flags ~0.4 % of perfect frames.
    WHAT IT DOES NOT SEE (stated, not hidden): GT error that is CORRELATED over more than a few GT samples, e.g.
    MadMax's sub-2-s GNSS-like wiggle, and the measurement jitter of a GT
    that was upsampled / smoothed from a lower rate.  From the GT alone such error cannot be separated from real
    motion at the same frequencies; it is a GT-fault flag, not this term.
    WHAT IT CAN OVER-READ (stated): on a LOW-RATE GT the 5-sample stencil spans 4 periods (0.8 s at 5 Hz), so rough
    motion near Nyquist leaks in (adversarial verifier: a zero-jitter 5 Hz rough rover reads 35.8 mm, 3x = 0.107 m >
    the floor; real Rosario v1 residuals rise with GT acceleration).  On today's rows 3 x sigma_3d <= 0.058 m, so it
    never binds.  No guard is coded: if it ever binds, check the row's stride
    ratio (the `stride` diagnostic) before trusting the loosening.
    DEFINITION CHOICE: an earlier design note proposed J4 =
    residual to a local quadratic over +-1 s.  This scorer does NOT use J4; it uses the 5-sample 4th divided
    difference above.  Why that measures white jitter and not motion: its weights sum every cubic in t to EXACTLY
    zero, and the stencil spans only 4 GT periods (20 ms at 200 Hz, 0.4 s at 10 Hz); on uniform spacing z =
    Delta^4 p / sqrt(70) ~ p''''(t) x dt^4 / sqrt(70), so motion leaks only through the 4th derivative (1 m/s^4 gives
    7.5e-11 m at 200 Hz, 1.2e-5 m at 10 Hz), while white noise passes with gain exactly 1.
    J4's quadratic over 2 s leaves the cubic and higher terms of real motion over a 2-s span in the residual (mh_04,
    Vicon with sub-mm jitter: J4 18.3 mm vs the native 2nd difference 0.0 mm, gt_jitter_scale_probe /
    gt_jitter_rel2s_probe), so J4 would let motion loosen B4.  J4 is still REPORTED alongside, as a
    diagnostic, by an internal script (J4_m per dev row and the B4 result with
    J4 substituted, an internal note); it never enters a bar.
    `stride` > 1 uses every stride-th sample (a diagnostic: white jitter is stride-invariant, motion leakage grows
    as stride^4, so equality across strides shows the number is jitter, not motion).
    Returns dict(sigma_3d, sigma_axis, n_stencils, median_dt_s); sigma_3d is None with fewer than 5 samples."""
    t = np.asarray(tg, float).reshape(-1)
    p = np.asarray(pg, float).reshape(len(t), -1)
    keep = np.isfinite(t) & np.isfinite(p).all(1)
    t, p = t[keep], p[keep]
    o = np.argsort(t, kind="stable")
    t, p = t[o], p[o]
    inc = np.concatenate([[True], np.diff(t) > 0])          # a repeated stamp is not a second sample
    t, p = t[inc][::int(stride)], p[inc][::int(stride)]
    n, S = len(t), JITTER_STENCIL
    out = {"sigma_3d": None, "sigma_axis": None, "n_stencils": int(max(n - S + 1, 0)), "stride": int(stride),
           "median_dt_s": float(np.median(np.diff(t))) if n > 1 else None}
    if n < S:
        return out
    m = n - S + 1
    Ts = np.stack([t[k:k + m] for k in range(S)], axis=1)          # (m, S)
    w = np.ones((m, S))
    for k in range(S):
        for q in range(S):
            if q != k:
                w[:, k] /= (Ts[:, k] - Ts[:, q])
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    z = np.einsum("mk,mkc->mc", w, np.stack([p[k:k + m] for k in range(S)], axis=1))
    s_ax = MAD_TO_SIGMA * np.median(np.abs(z - np.median(z, axis=0)), axis=0)
    out["sigma_axis"] = [float(x) for x in s_ax]
    out["sigma_3d"] = float(np.sqrt(np.sum(s_ax ** 2)))
    return out


def no_gt_reason(tag):
    """OAK-SCORE: the reason a registered OWN-RECORDING row (family "oak": OAK-D-LITE
    recordings, no ground truth exists) has no GT file, else None.  Only family "oak" rows are affected: for every
    other row a missing GT file still raises (a missing GT drive must never pass as "no GT")."""
    sys.path.insert(0, REPO)
    from xclean.io import rows as R
    r = R.get(tag)
    if r.family == "oak" and not os.path.isfile(r.gt_path):
        return "row %s (family oak, own recording) has no GT (%s absent)" % (tag, r.gt_path)
    return None


def score_frames(t, T, tag, has=None, dt1=True, gt_frame=GT_FRAME_DEF):
    """t (N,) frame stamps of EVERY frame, T (N,4,4) the pose held for each (NaN where none yet), has (N,) bool =
    a pose message existed for that very frame.  -> dict (ATE over the full GT span, coverage, B4 accuracy part).
    dt1=False reproduces the pre-DT1 B4 test (no 3 x GT-jitter term) for the before/after audit ONLY.
    gt_frame="legacy" reproduces the pre-GTF1 GT transform (GT_FRAMES) for the before/after audit ONLY.
    T is T_world_cam0_rectified (SCORED_FRAME) for every system scored.
    A no-GT row (no_gt_reason) returns ate None, no_gt True and a NO_GT flag; no accuracy test is applied."""
    why = no_gt_reason(tag)
    if why:
        return {"frames": int(len(t)), "frames_with_gt": 0, "frames_scored": 0, "coverage": None, "ate": None,
                "no_gt": True, "gt_frame": None, "rel2s_trans_clause": "NOT_APPLICABLE (no GT)",
                "rel2s_rot_clause": "NOT_APPLICABLE (no GT)", "flags": ["NO_GT: " + why],
                "_rel_bad": np.zeros(len(t), bool)}
    from xclean.eval.score import umeyama
    tg, pg, Rg = _gt_cam(tag) if gt_frame == GT_FRAME_DEF else _gt_cam(tag, frame=gt_frame)
    j, ok = _assoc(t, tg)
    have = np.isfinite(T[:, 0, 3])
    use = ok & have
    out = {"frames": int(len(t)), "frames_with_gt": int(ok.sum()), "frames_scored": int(use.sum()),
           "coverage": float(use.sum() / max(ok.sum(), 1))}
    if use.sum() < 3:
        out["ate"] = None
        return out
    pe = T[use, :3, 3]
    R, tr, _ = umeyama(pe, pg[j[use]], False)
    pa = (R @ pe.T).T + tr
    e = np.linalg.norm(pa - pg[j[use]], axis=1)
    out["ate"] = float(np.sqrt(np.mean(e ** 2)))
    out["ate_unheld_only"] = (float(np.sqrt(np.mean(e[has[use]] ** 2))) if has is not None and has[use].any()
                              else None)
    # B4 accuracy part: 2 s relative motion error vs GT (window starting at each frame)
    idx = np.nonzero(use)[0]
    ts = t[idx]
    k = np.searchsorted(ts, ts + 2.0)
    bad = np.zeros(len(t), bool)
    v = k < len(idx)
    a_, b_ = idx[v], idx[k[v]]
    Ra, Rb = T[a_, :3, :3], T[b_, :3, :3]
    de = np.einsum("nji,nj->ni", Ra, T[b_, :3, 3] - T[a_, :3, 3])
    ga, gb = j[a_], j[b_]
    # GTF1: a position-only GT (Rosario v1) has no orientation (Rg all-NaN).  The bar's translation error |de - dg|
    # needs dg in the GT's own frame at the window start, which does not exist; pre-GTF1 the scorer used the file's
    # constant identity quaternion, i.e. compared de (camera frame) with dg in a FIXED world frame, and failed almost
    # every window.  Here the translation clause is evaluated at its LOWER BOUND | |de| - |dg| | (= min over every
    # possible GT orientation of |de - R dg|, triangle inequality): a window it flags fails the bar under ANY GT
    # orientation; windows the bar would flag only through direction are not seen.  The rotation clause is NOT
    # APPLICABLE.  Both are stated in the output (flags); how the bar reads on such rows is the user's decision.
    orient = bool(np.isfinite(Rg).all())
    if orient:
        dg = np.einsum("nji,nj->ni", Rg[ga], pg[gb] - pg[ga])
        path = np.linalg.norm(dg, axis=1)
        terr = np.linalg.norm(de - dg, axis=1)
        rerr = _angle(np.einsum("nji,njk->nik", np.einsum("nji,njk->nik", Ra, Rb),
                                np.einsum("nji,njk->nik", Rg[ga], Rg[gb])))
        rot_gt = _angle(np.einsum("nji,njk->nik", Rg[ga], Rg[gb]))
    else:
        path = np.linalg.norm(pg[gb] - pg[ga], axis=1)
        terr = np.abs(np.linalg.norm(de, axis=1) - path)
    # the live benchmark rules B4: > max(0.10 m, 3 x GT jitter, 10 % of the window's path) or > max(2 deg, 10 % of its rotation).
    # "path" is read as the window's NET GT displacement |dg| (pre-DT1 reading, unchanged; stricter than a path-length
    # reading, which is a separate decision, the change log DT1); the rotation is the net GT rotation likewise.
    jit = gt_jitter(tg, pg)
    sj = jit["sigma_3d"] if (dt1 and jit["sigma_3d"] is not None) else 0.0
    t_floor = max(0.10, B4_JITTER_MULT * sj)
    f_t = terr > np.maximum(t_floor, 0.10 * path)
    f_r = (rerr > np.maximum(2.0, 0.10 * rot_gt)) if orient else np.zeros(len(a_), bool)
    fail = f_t | f_r
    bad[a_[fail]] = True
    out["rel2s_fail_frames"] = int(bad.sum())
    out["rel2s_windows"] = int(len(a_))
    out["rel2s_fail_trans_windows"] = int(f_t.sum())
    out["rel2s_fail_rot_windows"] = int(f_r.sum())
    out["gt_jitter_m"] = jit["sigma_3d"]
    out["gt_jitter_axis_m"] = jit["sigma_axis"]
    out["gt_jitter_def"] = GT_JITTER_DEF
    out["b4_trans_floor_m"] = float(t_floor)
    out["dt1"] = bool(dt1)
    out["gt_frame"] = gt_frame_meta(tag, gt_frame)
    out["rel2s_trans_clause"] = ("local_frame |de - dg|" if orient else
                                 "LOWER_BOUND | |de| - |dg| | (GT has no orientation)")
    out["rel2s_rot_clause"] = "applied" if orient else "NOT_APPLICABLE (GT has no orientation)"
    flags = []
    if not orient:
        flags.append("GT_POSITION_ONLY: B4 rotation clause NOT APPLICABLE; translation clause at its lower bound; "
                     "GT position is the GNSS antenna (cam0 lever not applicable), so the relative test is weaker on "
                     "this sequence")
    gm = out["gt_frame"]
    if gm is not None and not gm.get("verified", True) and gt_frame == GT_FRAME_DEF:
        flags.append("GT_FRAME_UNVERIFIED: source %r not in GT_FRAMES (pre-GTF1 assumption kept)" % gm.get("family"))
    if gm is not None and gm.get("gt_faults"):
        flags.append("GT_FAULTS (B4 relative test reads them as failures): " + " | ".join(gm["gt_faults"]))
    out["flags"] = flags
    out["_rel_bad"] = bad
    return out


def score_dir(d, tag=None, write=True, dt1=True, gt_frame=GT_FRAME_DEF):
    """write=False returns the score without writing DIR/live_score.json (re-scoring someone else's run);
    dt1=False: the pre-DT1 B4 test, gt_frame="legacy": the pre-GTF1 GT transform, both for the before/after audit
    only (see score_frames)."""
    rec = [json.loads(x) for x in open(os.path.join(d, "recorder.jsonl"), encoding="utf-8") if x.strip()]
    feeder = json.load(open(os.path.join(d, "feeder.json"), encoding="utf-8"))
    hj = json.load(open(os.path.join(d, "harness.json"), encoding="utf-8")) \
        if os.path.isfile(os.path.join(d, "harness.json")) else {}
    tag = tag or hj.get("tag")
    invalid = [m for m in rec if "invalid" in m]
    msgs = {}
    for m in rec:
        if "stream" in m and "frame_index" in m:
            msgs.setdefault(m["stream"], {})[m["frame_index"]] = m
    fr = feeder["frames"]                                    # [idx, t, planned_ns, actual_ns] for EVERY released frame
    idx = np.array([f[0] for f in fr], np.int64)
    t = np.array([f[1] for f in fr], float)
    rel = np.array([f[3] for f in fr], float)
    period = float(np.median(np.diff(t))) if len(t) > 1 else None
    lm = msgs.get("LIVE-MAP", {})
    N = len(idx)
    T = np.full((N, 4, 4), np.nan)
    has = np.zeros(N, bool)
    state_ok = np.zeros(N, bool)
    last = None
    lat = np.full(N, np.inf)
    rx = np.full(N, np.nan)
    for n in range(N):
        m = lm.get(int(idx[n]))
        if m is not None:
            last = np.array(m["T"], float).reshape(4, 4)
            has[n] = True
            state_ok[n] = (m["state"] in ("tracking", "relocalized", "init")) and not m["degraded"]
            lat[n] = (m["rx_ns"] - rel[n]) / 1e6
            rx[n] = m["rx_ns"]
        if last is not None:
            T[n] = last                                       # zero-order hold
    out = {"tag": tag, "dir": d, "mode": hj.get("mode"), "valid_run": hj.get("valid"), "duplicates": len(invalid),
           "camera_period_ms": None if period is None else 1e3 * period}
    acc = score_frames(t, T, tag, has, dt1=dt1, gt_frame=gt_frame) if tag else {}
    bad_rel = acc.pop("_rel_bad", np.zeros(N, bool))
    out["B5"] = acc
    # B1
    finite = np.isfinite(lat)
    pct = lambda q: float(np.percentile(np.where(finite, lat, 1e12), q))
    rxs = np.sort(rx[np.isfinite(rx)])
    gap = float(np.max(np.diff(rxs)) / 1e6) if len(rxs) > 1 else None
    out["B1"] = {"latency_ms_p50": pct(50), "latency_ms_p95": pct(95), "latency_ms_p99": pct(99),
                 "latency_ms_max": float(np.max(np.where(finite, lat, np.inf))),
                 "p95_periods": (pct(95) / (1e3 * period)) if period else None,
                 "p99_periods": (pct(99) / (1e3 * period)) if period else None,
                 "max_loop_gap_periods": (gap / (1e3 * period)) if (gap and period) else None,
                 "kind": "receipt(LIVE-MAP k) - release(k)" if hj.get("mode") == "paced" else
                         "lockstep: not a latency (unpaced)"}
    # B2
    dropped = ~has
    runs, cur = [], 0
    for x in dropped:
        cur = cur + 1 if x else 0
        runs.append(cur)
    win = 0.0
    if period and N:
        w = max(int(round(1.0 / period)), 1)
        c = np.convolve(dropped.astype(float), np.ones(w), "valid") / w if N >= w else np.array([dropped.mean()])
        win = float(c.max())
    q10 = max(N // 10, 1)
    l1, l2 = lat[:q10][np.isfinite(lat[:q10])], lat[-q10:][np.isfinite(lat[-q10:])]
    out["B2"] = {"dropped": int(dropped.sum()), "dropped_pct": 100.0 * dropped.mean() if N else None,
                 "max_consecutive": int(max(runs) if runs else 0), "worst_1s_window_pct": 100.0 * win,
                 "latency_drift": (float(np.percentile(l2, 95) / np.percentile(l1, 95))
                                   if len(l1) and len(l2) else None)}
    # B4
    unsup = (~has) | (~state_ok) | bad_rel
    odo = msgs.get("LIVE-ODOM", {})
    P = np.array([np.array(odo[k]["T"]).reshape(4, 4)[:3, 3] for k in sorted(odo)]) if odo else np.zeros((0, 3))
    jumps = 0
    odo_jump_frames = set()
    if len(P) > 12:
        st = np.linalg.norm(np.diff(P, axis=0), axis=1)
        odk = sorted(odo)
        for i in range(len(st)):
            nb = np.concatenate([st[max(0, i - 5):i], st[i + 1:i + 6]])
            if len(nb) and st[i] > 4 * np.median(nb) and st[i] > 0.05:
                jumps += 1
                odo_jump_frames.add(int(odk[i + 1]))
    # P1.3: a LIVE-MAP jump (the LIVE-ODOM jump rule applied to LIVE-MAP) must coincide
    # with a FLAGGED correction (the message's map_corr); reported beside the LIVE-ODOM count, no existing field changes
    lmk = sorted(lm)
    PM = np.array([np.array(lm[k]["T"]).reshape(4, 4)[:3, 3] for k in lmk]) if lm else np.zeros((0, 3))
    map_jumps, map_unflagged = 0, []
    if len(PM) > 12:
        st = np.linalg.norm(np.diff(PM, axis=0), axis=1)
        for i in range(len(st)):
            nb = np.concatenate([st[max(0, i - 5):i], st[i + 1:i + 6]])
            if len(nb) and st[i] > 4 * np.median(nb) and st[i] > 0.05:
                map_jumps += 1
                if not lm[lmk[i + 1]].get("map_corr", False):
                    map_unflagged.append(int(lmk[i + 1]))
    map_corr_frames = int(sum(1 for k in lmk if lm[k].get("map_corr", False)))
    # an unflagged LIVE-MAP jump on a frame where LIVE-ODOM itself jumps is the odometry's jump (live_odom_jumps), carried
    # by LIVE-MAP = T_applied LIVE-ODOM; the P1.3 clause concerns the rest (added column, 2026-10-26)
    map_unflagged_not_odom = [k for k in map_unflagged if k not in odo_jump_frames]
    first_pose = int(np.argmax(has)) if has.any() else None
    longest = 0
    cur = 0
    for x in unsup:
        cur = cur + 1 if x else 0
        longest = max(longest, cur)
    out["B4"] = {"unsupported_pct": 100.0 * unsup.mean() if N else None, "no_message": int((~has).sum()),
                 "state_not_ok": int((has & ~state_ok).sum()), "rel2s_fail": int(bad_rel.sum()),
                 "longest_unsupported_frames": int(longest), "first_pose_frame": first_pose,
                 "live_odom_jumps": int(jumps), "unrecovered_loss": bool(unsup[-max(q10, 1):].all()) if N else None,
                 "live_map_jumps": int(map_jumps), "live_map_jumps_unflagged": len(map_unflagged),
                 "live_map_jumps_unflagged_first": map_unflagged[:20], "live_map_corr_frames": map_corr_frames,
                 "live_map_jumps_unflagged_not_odom": len(map_unflagged_not_odom),
                 "live_map_jumps_unflagged_not_odom_first": map_unflagged_not_odom[:20],
                 "rel2s_trans_clause": acc.get("rel2s_trans_clause"), "rel2s_rot_clause": acc.get("rel2s_rot_clause"),
                 "gt_flags": acc.get("flags", [])}
    if write:
        json.dump(out, open(os.path.join(d, "live_score.json"), "w"), indent=1, default=str)
    return out


def score_error_message(e, row=None):
    """A plain reason why a run could not be scored (no traceback)."""
    gt = getattr(row, "gt_path", None)
    if isinstance(e, FileNotFoundError) and gt and not os.path.isfile(gt):
        return ("there is no ground-truth file %s. Give one with --gt, or use --family oak for an own recording "
                "without ground truth" % gt)
    if isinstance(e, KeyError):
        return "the sequence is not known here (%s); give the sequence folder with --tag" % (e.args[0] if e.args else e)
    return "%s: %s" % (type(e).__name__, e)


def load_kit_frames(path):
    """A per-frame trajectory file (kit or stripped TUM, t,x,y,z,qx,qy,qz,qw first) -> (t, T)."""
    from scipy.spatial.transform import Rotation as Rot
    A = np.loadtxt(path, delimiter=",", comments="#", usecols=range(8), ndmin=2)
    T = np.tile(np.eye(4), (len(A), 1, 1))
    T[:, :3, :3] = Rot.from_quat(A[:, 4:8]).as_matrix()
    T[:, :3, 3] = A[:, 1:4]
    return A[:, 0], T


def main():
    ap = argparse.ArgumentParser(description="Score a replay or live run folder against ground truth (writes "
                                             "live_score.json into it). The same as: xclean score <run folder>",
                                 allow_abbrev=False)
    ap.add_argument("dir", metavar="run_folder", help="the run folder (it holds recorder.jsonl and feeder.json)")
    ap.add_argument("--tag", default=None,
                    help="the sequence folder of the run (default: the one the run was made from)")
    ap.add_argument("--rows", default=None,
                    help="a JSON file that gives names to sequence folders (docs/SEQUENCE_FORMAT.md)")
    ap.add_argument("--gt", default=None, help="the ground-truth file, when it is not gt.csv in the sequence folder")
    ap.add_argument("--family", default=None,
                    help="'oak': an own recording without ground truth (reports no ATE instead of an error)")
    a = ap.parse_args()
    from xclean.io import rows as R
    R.set_registry(a.rows)
    if a.gt or a.family:
        t = a.tag or json.load(open(os.path.join(a.dir, "harness.json"), encoding="utf-8")).get("tag")
        R.get(t, gt=a.gt, family=a.family)
    print(json.dumps(score_dir(a.dir, a.tag), indent=1, default=str))


if __name__ == "__main__":
    main()
