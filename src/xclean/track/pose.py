#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""POSE FROM 3D-2D - RANSAC, then a Huber LM refine we own.

    T_wc, inl, dec = solve_pose(P_world, uv, rig, T_init)

WHY THE REFINE IS OURS AND THE RANSAC IS NOT
    RANSAC over minimal solvers is a library problem and OpenCV's is fine.  The
    REFINE is not: it is where the Jacobian lives, where the robust weight
    lives, and where the reprojection gate lives - three things this project
    has already been burned by (`slam-dlt-refit-bug`: a coplanar DLT refit
    replaced a WINNING pose with garbage and read as "fast-motion failure" for
    weeks; 16.4 m -> 3.3 m when fixed).  So the refine is written against
    `core/lie`, whose Jacobians are pinned by identity tests.

★ THE ANISOTROPIC REPROJECTION GATE (plan 2.11, 6)
    A single pixel radius is not rig-independent, and it is not even
    ISOTROPIC.  Depth uncertainty projects ALONG the ray; feature localisation
    noise is perpendicular to it.  So the gate is an ellipse:

        sigma_perp = feature localisation noise      (Class D, pixels)
        sigma_par  = sigma_z * parallax / z          (Class C, MEASURED)

    with the parallax measured from the run (`median(flow_px)*median(z)/fx`),
    never assumed.  A circular gate either rejects good far points or admits
    bad near ones, depending on which radius you pick - and picking one is the
    hand-set constant the whole doctrine forbids.

    The gate is applied as a RETRY, not as a first pass: solve, then re-gate
    with the measured parallax, then re-solve.  A gate that runs before the
    pose exists has to guess the pose.
"""
from __future__ import annotations

import numpy as np

from xclean.core.lie import se3_exp, se3_inv, so3_project, se3_from_Rt
from xclean.core.derive import HUBER_K, MIN_PNP_INLIERS
from xclean.obs.decision import Decision

FEATURE_SIGMA_PX = 1.0        # Class D: SuperPoint localises to ~1 px at any
                              # focal length.  A DETECTOR property.

# ARM XTEN_PNP_VERIFY=all  (tier-1 CORRECTNESS FIX, 2026-10-02, kitti_01 root cause; default off = byte-identical).
# THE DEFECT.  solve_pose drops post-refine outliers from the RANSAC mask only when at least `min_inliers` of them
# survive; when fewer survive the mask is left UNCHANGED and the pose is returned with the stale count.  A refine that
# diverged (kitti_01 frame 233: "114 inliers", rms 8.4e7 px, 4 of 730 correspondences within 3 px of the returned
# pose, a 5.6 deg / 8.25 m step) is therefore ACCEPTED: the returned mask contradicts the returned pose.
# THE FIX.  An accepted pose must explain >= min_inliers of ALL the correspondences within the same reprojection
# threshold UNDER THE FINAL POSE (in front of the camera), not merely have survived the RANSAC list.  Healthy calls
# are untouched (their mask members are within the threshold by construction); the caller's existing guess-free
# retry / coast path takes over a refused pose.  Counters: PNP_VERIFY_STATS (process-wide).
# SCOPE: the verify lives in
# solve_pose, so it acts in EVERY caller - Tracker.step's tracking solve and its guess-free retry, the local-map re-solve
# (Tracker._track_local_map), relocalisation (Tracker._relocalize), the loop closer's verification PnP
# (LoopCloser._verify, MIN_LOOP_INLIERS) and solve_pose_staged / solve_pose_refereed.  PNP_VERIFY_STATS is process-wide
# (the closer's verification runs in the same process); PNP_VERIFY_BY_CALLER splits the same counts by the FIRST engine
# frame above solve_pose ("<module>.<function>", e.g. "track.tracker.step", "loop.closer._verify"), so a summary says
# which caller refused what.  Both are provenance only: no decision reads them.
PNP_VERIFY_ENV = "XTEN_PNP_VERIFY"
PNP_VERIFY_STATS = {"calls": 0, "refused": 0}
PNP_VERIFY_BY_CALLER = {}
PNP_VERIFY_LOCK = __import__("threading").Lock()    # the closer's thread and the tracker update the same counters


def pnp_verify_caller():
    """'<module>.<function>' of the first engine frame above solve_pose (harness wrappers outside the package are
    skipped); 'unknown' when there is none.  Called only with XTEN_PNP_VERIFY=all."""
    import sys
    f = sys._getframe(2)
    while f is not None:
        mod = f.f_globals.get("__name__", "")
        if mod.startswith("xclean.") and not (mod == "xclean.track.pose" and f.f_code.co_name == "solve_pose"):
            return "%s.%s" % (mod[len("xclean."):], f.f_code.co_name)
        f = f.f_back
    return "unknown"


def pnp_verify_summary():
    """The provenance record a run summary carries when XTEN_PNP_VERIFY=all: totals + the per-caller split."""
    with PNP_VERIFY_LOCK:
        return {"calls": int(PNP_VERIFY_STATS["calls"]), "refused": int(PNP_VERIFY_STATS["refused"]),
                "by_caller": {k: dict(v) for k, v in sorted(PNP_VERIFY_BY_CALLER.items())}}


def pnp_verify_mode(env=None):
    """XTEN_PNP_VERIFY: unset / "" / "0" / "off" -> "off"; "all" -> "all"; anything else is an error."""
    import os
    v = (os.environ if env is None else env).get(PNP_VERIFY_ENV, "")
    v = (v or "").strip().lower()
    if v in ("", "0", "off"):
        return "off"
    if v == "all":
        return "all"
    raise RuntimeError("%s=%r is not a mode: unset/off or all" % (PNP_VERIFY_ENV, v))


def project(T_cw, P_world, fx, fy, cx, cy):
    """World points -> pixels, plus the camera-frame points."""
    Pc = (T_cw[:3, :3] @ P_world.T).T + T_cw[:3, 3]
    z = np.maximum(Pc[:, 2], 1e-6)
    u = fx * Pc[:, 0] / z + cx
    v = fy * Pc[:, 1] / z + cy
    return np.stack([u, v], axis=1), Pc


def _jacobian(Pc, fx, fy):
    """d(u,v)/d(xi) for a LEFT perturbation of T_cw, ordering [rho, phi]."""
    x, y, z = Pc[:, 0], Pc[:, 1], np.maximum(Pc[:, 2], 1e-6)
    iz = 1.0 / z
    iz2 = iz * iz
    n = Pc.shape[0]
    J = np.zeros((n, 2, 6))
    J[:, 0, 0] = fx * iz
    J[:, 0, 2] = -fx * x * iz2
    J[:, 1, 1] = fy * iz
    J[:, 1, 2] = -fy * y * iz2
    # rotation block: d(Pc)/d(phi) = -hat(Pc)
    J[:, 0, 3] = -fx * x * y * iz2
    J[:, 0, 4] = fx * (1.0 + x * x * iz2)
    J[:, 0, 5] = -fx * y * iz
    J[:, 1, 3] = -fy * (1.0 + y * y * iz2)
    J[:, 1, 4] = fy * x * y * iz2
    J[:, 1, 5] = fy * x * iz
    return J


def _lat_m2():
    import os
    return "m2" in (os.environ.get("XCLEAN_LAT", "") or "").lower().replace("+", ",").split(",")


def refine(T_cw, P_world, uv, rig, sigma_perp=FEATURE_SIGMA_PX,
           sigma_par=None, iters=10, point_scale=None):
    """Huber-robust LM on SE(3).  Returns (T_cw, residual_px, weights).

    `point_scale` (ARM XTEN_FAR_DEPTH=pnp, default None = bit-identical): a per-point factor >= 1 dividing the
    whitening, i.e. multiplying that observation's sigma - far points whose stereo depth carries the drive-level
    disparity bias are down-weighted smoothly (tracker._far_point_scale)."""
    fx, fy, cx, cy = rig.fx_rect, rig.fx_rect, rig.cx, rig.cy
    if sigma_par is None and point_scale is None and _lat_m2():
        # XCLEAN_LAT=m2 (latency track M2): the same LM in numba; rounding-level, not bit-identical
        from xclean.track.pose_fast import refine_nb
        T_cw = refine_nb(np.ascontiguousarray(T_cw, np.float64), np.ascontiguousarray(P_world, np.float64),
                         np.ascontiguousarray(uv, np.float64), float(fx), float(fy), float(cx), float(cy),
                         float(sigma_perp), int(iters), float(HUBER_K))
        pred, Pc = project(T_cw, P_world, fx, fy, cx, cy)
        return T_cw, np.linalg.norm(pred - uv, axis=1), Pc
    lam = 1e-3
    for _ in range(iters):
        pred, Pc = project(T_cw, P_world, fx, fy, cx, cy)
        r = pred - uv
        # anisotropic whitening: along-ray vs perpendicular
        W = _whiten(r, Pc, sigma_perp, sigma_par, fx)
        if point_scale is not None:
            W = W / point_scale[:, None]
        rw = r * W
        e = np.linalg.norm(rw, axis=1)
        k = HUBER_K
        w = np.where(e <= k, 1.0, k / np.maximum(e, 1e-9))
        J = _jacobian(Pc, fx, fy) * W[:, :, None]
        Jw = J * np.sqrt(w)[:, None, None]
        rww = rw * np.sqrt(w)[:, None]
        H = np.einsum("nij,nik->jk", Jw, Jw)
        b = -np.einsum("nij,ni->j", Jw, rww)
        H_lm = H + lam * np.diag(np.maximum(np.diag(H), 1e-9))
        try:
            dx = np.linalg.solve(H_lm, b)
        except np.linalg.LinAlgError:
            break
        T_new = se3_exp(dx) @ T_cw
        T_new[:3, :3] = so3_project(T_new[:3, :3])
        pred2, _ = project(T_new, P_world, fx, fy, cx, cy)
        if np.mean(np.linalg.norm(pred2 - uv, axis=1)) <= \
                np.mean(np.linalg.norm(r, axis=1)):
            T_cw = T_new
            lam = max(lam * 0.5, 1e-6)
        else:
            lam = min(lam * 4.0, 1e3)
        if np.linalg.norm(dx) < 1e-9:
            break
    pred, Pc = project(T_cw, P_world, fx, fy, cx, cy)
    return T_cw, np.linalg.norm(pred - uv, axis=1), Pc


def _whiten(r, Pc, sigma_perp, sigma_par, fx):
    """Per-observation 1/sigma in (u,v), from the anisotropic ellipse.

    The covariance of a reprojection is

        C = s_par^2 * d d^T  +  s_perp^2 * (I - d d^T)

    where `d` is the unit RADIAL direction in the image (the projection of the
    ray direction).  Depth uncertainty stretches the ellipse along `d`;
    feature-localisation noise is isotropic and fills the rest.  The diagonal
    is used - the off-diagonal correlation would need a full 2x2 whitening per
    observation for a second-order effect on a robust cost that already
    down-weights the tail.
    """
    if sigma_par is None:
        return np.full(r.shape, 1.0 / sigma_perp)
    d = np.stack([Pc[:, 0], Pc[:, 1]], axis=1)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = np.divide(d, n, out=np.zeros_like(d), where=n > 1e-9)
    sp = np.asarray(sigma_par, dtype=float).reshape(-1)
    sp = np.maximum(sp, sigma_perp)
    d2 = d * d
    var = (sp[:, None] ** 2) * d2 + (sigma_perp ** 2) * (1.0 - d2)
    return 1.0 / np.maximum(np.sqrt(var), 1e-6)


def solve_pose_refereed(P_world, uv, age_kf, window_span, rig, T_init=None,
                        min_inliers=MIN_PNP_INLIERS, parallax_px=None,
                        median_z=None, T_prev=None):
    """The plain solver first; the hypothesis chain ONLY when its consensus
    is not anchored by history.

    ★ WHY THE REFEREE EXISTS (2026-08-27).  The always-on staged chain was
    REFUTED by its own control at n=2: kitti_00 moved 0.858-0.908 ->
    0.9818/0.9839, far outside its band, because reordering trust on every
    frame perturbs ordinary tracking.  The repair is the project's superset
    doctrine one level down: the plain solver runs EXACTLY as before, and its
    answer stands whenever its inlier set contains at least `min_inliers`
    MATURE landmarks - the existing Class-D conditioning floor, now applied
    to the anchored subset.  A consensus with that much history behind it
    cannot be a crossing mover.  Only when anchoring fails (the kitti_07
    flip frame: 45 inliers, 4 mature) does the chain engage.
    """
    T, mask, dec = solve_pose(P_world, uv, rig, T_init=T_init,
                              min_inliers=min_inliers,
                              parallax_px=parallax_px, median_z=median_z)
    age_kf = np.asarray(age_kf)
    mature = age_kf > int(window_span)
    if T is not None and int((mask & mature).sum()) >= min_inliers:
        dec.dials["staged"] = "anchored"
        return T, mask, dec
    return solve_pose_staged(P_world, uv, age_kf, window_span, rig,
                             T_init=T_init, min_inliers=min_inliers,
                             parallax_px=parallax_px, median_z=median_z,
                             plain=(T, mask, dec), T_prev=T_prev)


def solve_pose_staged(P_world, uv, age_kf, window_span, rig, T_init=None,
                      min_inliers=MIN_PNP_INLIERS, parallax_px=None,
                      median_z=None, plain=None, T_prev=None):
    """PnP where the CONSENSUS is defined by mature landmarks only.

    ★ WHY (kitti_07, 2026-08-27).  For nine frames at t=66.2-67.0 a crossing
    truck owned the RANSAC vote: the winning consensus was landmarks of age
    0-2 keyframes at ~7 m moving 11-15 m/s, while the STATIC world sat in the
    reject set as a coherent second model with 37-66 inliers at exactly the
    ground-truth speed.  8-9 m of phantom motion entered the trajectory in one
    second, the displaced map then manufactured duplicate landmarks, and loop
    verification died against it (321 "PnP under the loop floor" rejects).
    An inlier COUNT cannot tell a rigid mover from the world; a track that has
    SURVIVED CAMERA MOTION can.

    THE MECHANISM, following the local-map union rule ("matches EXTEND the
    evidence; they never substitute"):
      1. mature = age > window_span.  A landmark whose host has left the
         matching window has been re-associated across a window turnover -
         it holds a track record over parallax.  `window_span` is the
         matching-window size (Class B: an architecture quantity, not a tune).
      2. RANSAC + refine on the mature subset ONLY - the consensus.
      3. every correspondence (young included) is admitted against that pose
         under the same residual rule the plain solver uses, and the final
         refine runs on the union - so agreeing young landmarks lose nothing.
      4. if the mature subset is under the conditioning floor (cold start, or
         the scene is genuinely all-new after a turn) the plain solver runs
         and the fallback is a LOGGED dial, never silence.
    """
    # ★ THE HYPOTHESIS FLOOR IS NOT THE POSE FLOOR.  The mature stage does
    # not claim a pose - the UNION does, and the union must reach the full
    # `min_inliers` conditioning floor or the staged result is abandoned.
    # So the mature stage needs only enough points for the hypothesis to be
    # determined: 6, the PnP uniqueness minimum (Class D, geometry).  This
    # matters because the measured failure was exactly a floor collision:
    # at the kitti_07 flip frame the truck OCCLUDED the background down to
    # 23 mature correspondences - two under the old floor of 25 - and the
    # fallback handed the vote back to the truck.
    HYPOTHESIS_FLOOR = 6
    age_kf = np.asarray(age_kf)
    mature = age_kf > int(window_span)
    n_mat = int(mature.sum())
    thr = 3.0 * FEATURE_SIGMA_PX          # the plain solver's own gate

    def _union_under(T_hyp):
        """Admit the WHOLE population against a hypothesis, one refine, and
        the FULL conditioning floor - the single verification rule every
        hypothesis in the chain must pass."""
        T_cw = se3_inv(T_hyp)
        pred, Pc = project(T_cw, P_world, rig.fx_rect, rig.fx_rect,
                           rig.cx, rig.cy)
        resid = np.linalg.norm(pred - uv, axis=1)
        admit = (resid <= thr) & (Pc[:, 2] > 0)
        if admit.sum() < min_inliers:
            return None
        T_cw2, resid2, _ = refine(se3_inv(T_hyp), P_world[admit], uv[admit],
                                  rig)
        keep = resid2 <= thr
        if keep.sum() < min_inliers:
            return None
        mk = np.zeros(len(P_world), bool)
        mk[np.nonzero(admit)[0][keep]] = True
        return se3_inv(T_cw2), mk

    # ---- hypothesis 1: the mature consensus ------------------------------
    if n_mat >= HYPOTHESIS_FLOOR:
        T_m, m_mask, _dec = solve_pose(P_world[mature], uv[mature], rig,
                                       T_init=T_init,
                                       min_inliers=HYPOTHESIS_FLOOR,
                                       parallax_px=parallax_px,
                                       median_z=median_z)
        if T_m is not None:
            got = _union_under(T_m)
            if got is not None:
                T, mask = got
                # ★ AGREEMENT PREFERS THE PLAIN CONSENSUS.  When the plain
                # solve's own inliers are consistent with the mature-union
                # pose (>= the floor of them reproject inside the same gate),
                # the two describe the SAME motion and the plain result -
                # which is bit-for-bit the baseline solver's - stands.  The
                # chain then overrides ONLY on genuine disagreement (the
                # capture frames).  Without this, the union pose replaced a
                # healthy plain consensus on 297 ordinary V1_01 frames and
                # moved a control row.
                if plain is not None and plain[0] is not None:
                    Tp, mp, dp = plain
                    if mp is not None and mp.sum() >= min_inliers:
                        pr, Pc_ = project(se3_inv(T), P_world[mp],
                                          rig.fx_rect, rig.fx_rect,
                                          rig.cx, rig.cy)
                        agree = (np.linalg.norm(pr - uv[mp], axis=1) <= thr) \
                            & (Pc_[:, 2] > 0)
                        if int(agree.sum()) >= min_inliers:
                            dp.dials["staged"] = "union-agree"
                            return Tp, mp, dp
                n_young = int(mask.sum() - (mask & mature).sum())
                return T, mask, Decision(
                    "pnp", True, "%d inliers (%d mature-consensus, %d young "
                    "admitted)" % (mask.sum(), int((mask & mature).sum()),
                                   n_young),
                    {"staged": "union", "n_mature": n_mat,
                     "n_young_admitted": n_young, "inliers": int(mask.sum()),
                     "n_corr": len(P_world)},
                    margin=int(mask.sum()) - min_inliers,
                    threshold=min_inliers)

    # ---- hypothesis 2: the motion model ----------------------------------
    # The constant-velocity prediction extrapolates the last mature-verified
    # motion, so it is a legitimate consensus surrogate when occlusion has
    # starved the mature supply (kitti_07: the truck occluded the background
    # down to 0-9 mature correspondences for ~0.6 s).  A mover cannot pass
    # this union: at 7 m a crossing truck displaces ~50 px/frame against a
    # 3 px gate.  A genuinely new scene fails the floor here and falls
    # through to the plain solver, so cold starts are untouched.
    if T_init is not None:
        cands = [("motion-union", T_init)]
        if T_prev is not None:
            # deceleration: the truth during braking lies between constant
            # velocity and full stop, and at 20 m the constant-velocity
            # prediction error sits right at the 3 px admission gate - the
            # measured draw-to-draw bimodality of the kitti_07 event.  The
            # translation midpoint covers it; rotation stays the prediction's.
            T_half = np.array(T_init, float).copy()
            T_half[:3, 3] = 0.5 * (T_init[:3, 3] + T_prev[:3, 3])
            cands.append(("motion-union-decel", T_half))
            T_stop = np.array(T_init, float).copy()
            T_stop[:3, 3] = T_prev[:3, 3]
            cands.append(("motion-union-stop", T_stop))
        for label, T_hyp in cands:
            got = _union_under(T_hyp)
            if got is not None:
                T, mask = got
                # the same agreement rule the mature-union carries: if the
                # plain consensus is consistent with this verified pose, the
                # frame is CORROBORATED and the plain result stands
                if plain is not None and plain[0] is not None:
                    Tp, mp, dp = plain
                    if mp is not None and mp.sum() >= min_inliers:
                        pr, Pc_ = project(se3_inv(T), P_world[mp],
                                          rig.fx_rect, rig.fx_rect,
                                          rig.cx, rig.cy)
                        agree = (np.linalg.norm(pr - uv[mp], axis=1) <= thr) \
                            & (Pc_[:, 2] > 0)
                        if int(agree.sum()) >= min_inliers:
                            dp.dials["staged"] = "motion-agree"
                            return Tp, mp, dp
                return T, mask, Decision(
                    "pnp", True, "%d inliers under the %s hypothesis "
                    "(mature supply %d)" % (mask.sum(), label, n_mat),
                    {"staged": label, "n_mature": n_mat,
                     "inliers": int(mask.sum()), "n_corr": len(P_world)},
                    margin=int(mask.sum()) - min_inliers,
                    threshold=min_inliers)

    # ---- hypothesis 3: the plain consensus, ARBITRATED against its own
    # rejects.  At the deepest occlusion the mature supply is 0 and the
    # braking motion prediction misses the union floor by a couple of pixels,
    # so both hypotheses above fail - and the plain consensus is the mover.
    # Its REJECT set still holds the static world as a coherent second model
    # (measured: 37-66 inliers at exactly ground-truth speed).  Solve it,
    # verify it through the SAME union rule, and prefer whichever model
    # deviates LESS from the motion prediction - a comparison between two
    # verified models, not a threshold on either.
    if plain is not None:
        T, mask, dec = plain
    else:
        T, mask, dec = solve_pose(P_world, uv, rig, T_init=T_init,
                                  min_inliers=min_inliers,
                                  parallax_px=parallax_px, median_z=median_z)
    if (T is not None and T_init is not None
            and int((~mask).sum()) >= min_inliers):
        T2, m2, _d2 = solve_pose(P_world[~mask], uv[~mask], rig,
                                 T_init=T_init, min_inliers=min_inliers)
        if T2 is not None:
            got = _union_under(T2)
            if got is not None:
                Tc, mc = got
                dev_plain = float(np.linalg.norm(
                    (se3_inv(T_init) @ T)[:3, 3]))
                dev_chal = float(np.linalg.norm(
                    (se3_inv(T_init) @ Tc)[:3, 3]))
                if dev_chal < dev_plain:
                    return Tc, mc, Decision(
                        "pnp", True, "%d inliers by reject-arbitration "
                        "(challenger dev %.2f m vs plain %.2f m)"
                        % (mc.sum(), dev_chal, dev_plain),
                        {"staged": "reject-arbitration",
                         "n_mature": n_mat, "inliers": int(mc.sum()),
                         "n_corr": len(P_world)},
                        margin=int(mc.sum()) - min_inliers,
                        threshold=min_inliers)
    dec.dials["staged"] = "fallback: mature supply %d, no hypothesis " \
        "passed the union floor" % n_mat
    return T, mask, dec


def solve_pose(P_world, uv, rig, T_init=None, reproj_px=None,
               min_inliers=MIN_PNP_INLIERS, parallax_px=None,
               median_z=None, point_scale=None):
    """RANSAC PnP, then the anisotropic-gate retry, then a robust refine.

    Returns (T_wc or None, inlier mask, Decision).
    `point_scale` (per correspondence, ARM XTEN_FAR_DEPTH=pnp): passed to both refines; RANSAC is unweighted.
    None (the default) is bit-identical to the pre-arm code.
    """
    import cv2
    n = len(P_world)
    dials = {"n_corr": n, "min_inliers": min_inliers}
    if n < min_inliers:
        return None, np.zeros(n, bool), Decision(
            "pnp", False, "too few correspondences", dials, margin=n - min_inliers)

    K = np.array([[rig.fx_rect, 0, rig.cx],
                  [0, rig.fx_rect, rig.cy], [0, 0, 1]], float)
    thr = float(reproj_px if reproj_px is not None else 3.0 * FEATURE_SIGMA_PX)
    rvec = tvec = None
    use_guess = False
    if T_init is not None:
        T_cw0 = se3_inv(T_init)
        rvec, _ = cv2.Rodrigues(np.ascontiguousarray(T_cw0[:3, :3]))
        tvec = T_cw0[:3, 3].reshape(3, 1).copy()
        use_guess = True
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        P_world.astype(np.float64), uv.astype(np.float64), K, None,
        rvec=rvec, tvec=tvec, useExtrinsicGuess=use_guess,
        iterationsCount=200, reprojectionError=thr, confidence=0.9999,
        flags=cv2.SOLVEPNP_ITERATIVE if use_guess else cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < min_inliers:
        k = 0 if inl is None else len(inl)
        dials["ransac_inliers"] = k
        return None, np.zeros(n, bool), Decision(
            "pnp", False, "RANSAC failed or under the conditioning floor",
            dials, margin=k - min_inliers, threshold=min_inliers)

    mask = np.zeros(n, bool)
    mask[inl.reshape(-1)] = True
    R, _ = cv2.Rodrigues(rvec)
    T_cw = se3_from_Rt(so3_project(R), tvec.reshape(3))

    # ---- the anisotropic gate, as a RETRY on the measured parallax --------
    sigma_par = None
    if parallax_px is not None and median_z is not None:
        sz = rig.depth_sigma_m(max(median_z, 1e-3))
        sigma_par = np.full(mask.sum(), max(sz * parallax_px /
                                            max(median_z, 1e-3),
                                            FEATURE_SIGMA_PX))
        dials["sigma_par_px"] = float(sigma_par[0])
    T_cw, resid, _Pc = refine(T_cw, P_world[mask], uv[mask], rig,
                              sigma_par=sigma_par,
                              point_scale=None if point_scale is None else point_scale[mask])
    keep = resid <= max(thr, 2.0 * FEATURE_SIGMA_PX)
    if keep.sum() >= min_inliers and keep.sum() < mask.sum():
        idx = np.nonzero(mask)[0][~keep]
        mask[idx] = False
        T_cw, resid, _Pc = refine(T_cw, P_world[mask], uv[mask], rig,
                                  sigma_par=None,
                                  point_scale=None if point_scale is None else point_scale[mask])

    dials.update(inliers=int(mask.sum()),
                 rms_px=float(np.sqrt(np.mean(resid ** 2))) if len(resid) else
                 float("nan"))
    if pnp_verify_mode() == "all":
        _pr, _pc = project(T_cw, P_world, rig.fx_rect, rig.fx_rect, rig.cx, rig.cy)
        _ok = (_pc[:, 2] > 1e-3) & (np.linalg.norm(_pr - uv, axis=1) < thr)
        dials["verify_n"] = int(_ok.sum())
        _refuse = int(_ok.sum()) < min_inliers
        _caller = pnp_verify_caller()
        with PNP_VERIFY_LOCK:
            PNP_VERIFY_STATS["calls"] += 1
            _vc = PNP_VERIFY_BY_CALLER.setdefault(_caller, {"calls": 0, "refused": 0})
            _vc["calls"] += 1
            if _refuse:
                PNP_VERIFY_STATS["refused"] += 1
                _vc["refused"] += 1
        if _refuse:
            return None, mask, Decision("pnp", False, "final pose explains %d < %d of all %d correspondences"
                                        % (int(_ok.sum()), min_inliers, n), dials,
                                        margin=int(_ok.sum()) - min_inliers, threshold=min_inliers)
    if mask.sum() < min_inliers:
        return None, mask, Decision("pnp", False, "refine dropped below the "
                                    "conditioning floor", dials,
                                    margin=int(mask.sum()) - min_inliers)
    return se3_inv(T_cw), mask, Decision(
        "pnp", True, "%d inliers, %.2f px rms" % (mask.sum(), dials["rms_px"]),
        dials, margin=int(mask.sum()) - min_inliers, threshold=min_inliers)
