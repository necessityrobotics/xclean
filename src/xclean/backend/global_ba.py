#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""GLOBAL BUNDLE ADJUSTMENT - the whole trajectory, once, off the frame path.

    poses, pts, stats = global_ba(kf_poses, lm_xyz, observations, rig, prior)

WHY THIS EXISTS ALONGSIDE THE BOUNDED LOCAL BA
    They answer different questions and the plan's latency budget is about only
    one of them.  `local_ba` runs ONCE PER KEYFRAME inside a hard 10 ms bar and
    is bounded by construction: no solve's cost may depend on map size
    (plan 5.1).  A global BA is not a per-keyframe process at all - it runs at
    most a handful of times in a run, on a worker or after the last frame, and
    it is budgeted the way `loop_apply` is: per EVENT, never in the frame path.

    Keeping them separate is what lets both statements be true at once:
    "the backend costs <=20 ms per keyframe" and "the trajectory is globally
    consistent".  Merging them would make the first claim false and the second
    unmeasurable.

★ WHY IT IS SPARSE AND NOT THE SAME KERNEL
    The local solver's reduced camera system is 60-180 unknowns and its
    single-block Cholesky is the right shape for that (P0: cuSOLVER was 91 % of
    the floor at those sizes).  A 650-keyframe global solve is 3,900 unknowns,
    and a dense O(n^3) factorisation of that is 6e10 operations - the same
    reasoning that made a single block right at 180 makes it wrong at 3,900.
    So the reduced system is assembled sparse and factorised by SuiteSparse.
    The Schur complement, the robust weights and the stereo scale prior are
    identical to the local solver; only the linear algebra changes.
"""
from __future__ import annotations

import os
import time

import numpy as np

from xclean.core.derive import HUBER_K, DISPARITY_NOISE_PX
from xclean.core.lie import se3_exp, se3_inv, so3_project, so3_log, so3_jr_inv
from xclean.backend.preint import GRAVITY_MAG

D = 6


def global_ba(kf_ids, poses, pts, okf, olm, ouv, rig, prior=None, rotf=None,
              welds=None, iters=8, fixed=(0,), lam0=1e-4, verbose=False,
              time_budget_s=None, passes=2, reject_sigma=3.0,
              want_cov=0, rotf_huber=None):
    """Levenberg-Marquardt over every keyframe and every landmark.

    `okf`/`olm` index into `poses`/`pts`; `ouv` is the measured pixel.  Returns
    the optimised poses and points plus a stats dict.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    nk, nl, N = len(poses), len(pts), len(okf)
    n = D * nk
    fx, cx, cy = rig.fx_rect, rig.cx, rig.cy
    T_cw = np.stack([se3_inv(T) for T in poses])
    pts = np.array(pts, dtype=np.float64)
    free = np.ones(nk, bool)
    for f in fixed:
        if 0 <= f < nk:
            free[f] = False

    order = np.argsort(olm, kind="stable")
    okf_s, olm_s, ouv_s = okf[order], olm[order], ouv[order]
    counts = np.bincount(olm_s, minlength=nl)
    offs = np.zeros(nl + 1, np.int64)
    offs[1:] = np.cumsum(counts)

    def residuals(T_cw, pts):
        R = T_cw[okf_s, :3, :3]
        t = T_cw[okf_s, :3, 3]
        Pc = np.einsum("nij,nj->ni", R, pts[olm_s]) + t
        z = np.maximum(Pc[:, 2], 1e-6)
        pred = np.stack([fx * Pc[:, 0] / z + cx, fx * Pc[:, 1] / z + cy], 1)
        return pred - ouv_s, Pc

    r0, _ = residuals(T_cw, pts)
    rms0 = float(np.sqrt(np.mean(np.sum(r0 ** 2, 1))))
    lam = lam0
    t_start = time.perf_counter()
    best = (T_cw.copy(), pts.copy(), float(np.sum(r0 ** 2)))
    n_rejected = 0
    keep_mask = np.ones(len(okf_s), bool)
    # declared BEFORE the loop that assigns it - the first version put this
    # after, so the loop is trailing H was overwritten with None and the
    # covariance export silently produced zero blocks every run
    last_H, last_free = None, free

    for it in range(iters):
        if time_budget_s and time.perf_counter() - t_start > time_budget_s:
            break
        r, Pc = residuals(T_cw, pts)
        chi = float(np.sum(r ** 2))
        e = np.linalg.norm(r, axis=1)
        w = np.where(e <= HUBER_K, 1.0, HUBER_K / np.maximum(e, 1e-9))
        sw = np.sqrt(w)[:, None]
        x, y, zz = Pc[:, 0], Pc[:, 1], np.maximum(Pc[:, 2], 1e-6)
        iz = 1.0 / zz
        iz2 = iz * iz
        A = np.zeros((N, 2, 3))
        A[:, 0, 0] = fx * iz
        A[:, 0, 2] = -fx * x * iz2
        A[:, 1, 1] = fx * iz
        A[:, 1, 2] = -fx * y * iz2
        dP = np.zeros((N, 3, 6))
        dP[:, 0, 0] = dP[:, 1, 1] = dP[:, 2, 2] = 1.0
        dP[:, 0, 4] = zz;  dP[:, 0, 5] = -y
        dP[:, 1, 3] = -zz; dP[:, 1, 5] = x
        dP[:, 2, 3] = y;   dP[:, 2, 4] = -x
        Jp = np.einsum("nij,njk->nik", A, dP) * sw[:, :, None]
        Jl = np.einsum("nij,njk->nik", A, T_cw[okf_s, :3, :3]) * sw[:, :, None]
        rw = r * sw

        Hll = np.zeros((nl, 3, 3))
        bl = np.zeros((nl, 3))
        np.add.at(Hll, olm_s, np.einsum("nri,nrj->nij", Jl, Jl))
        np.add.at(bl, olm_s, -np.einsum("nri,nr->ni", Jl, rw))
        Hpl = np.einsum("nri,nrj->nij", Jp, Jl)

        if prior is not None:
            c, rho0, invsig, valid = prior
            dPv = pts - c
            dist = np.linalg.norm(dPv, axis=1)
            ok = (valid > 0.5) & (dist > 1e-6)
            if ok.any():
                rho = np.zeros_like(dist)
                rho[ok] = 1.0 / dist[ok]
                rp = (rho - rho0) * invsig
                wp = np.where(np.abs(rp) <= HUBER_K, 1.0,
                              HUBER_K / np.maximum(np.abs(rp), 1e-9))
                swp = np.sqrt(wp)
                gg = np.zeros_like(dist)
                gg[ok] = -rho[ok] ** 2 / dist[ok] * invsig[ok]
                Jpr = dPv * (gg * swp)[:, None]
                Jpr[~ok] = 0.0
                rpw = rp * swp
                rpw[~ok] = 0.0
                Hll += np.einsum("ni,nj->nij", Jpr, Jpr)
                bl -= Jpr * rpw[:, None]

        Hll = Hll + lam * np.eye(3) * np.maximum(
            np.trace(Hll, axis1=1, axis2=2)[:, None, None] / 3.0, 1e-9)
        Hlli = np.linalg.inv(Hll)

        rows, cols, vals = [], [], []
        bp = np.zeros(n)
        blk = np.einsum("nri,nrj->nij", Jp, Jp)
        gp = -np.einsum("nri,nr->ni", Jp, rw)
        ii = (okf_s[:, None] * D + np.arange(D)[None, :])
        for a in range(D):
            np.add.at(bp, ii[:, a], gp[:, a])
        rows.append(np.repeat(ii, D, axis=1).reshape(-1))
        cols.append(np.tile(ii, (1, D)).reshape(-1))
        vals.append(blk.reshape(N, -1).reshape(-1))

        # Schur: every ordered observation pair of the same landmark
        T3 = np.einsum("nij,njk->nik", Hpl, Hlli[olm_s])
        sq = counts * counts
        total = int(sq.sum())
        rep = np.repeat(np.arange(nl, dtype=np.int64), sq)
        base = np.zeros(nl + 1, np.int64)
        base[1:] = np.cumsum(sq)
        local = np.arange(total, dtype=np.int64) - base[rep]
        cc = counts[rep]
        pa = offs[rep] + local // np.maximum(cc, 1)
        pb = offs[rep] + local % np.maximum(cc, 1)
        blocks = -np.einsum("pij,pkj->pik", T3[pa], Hpl[pb])
        ra = (okf_s[pa][:, None] * D + np.arange(D)[None, :])
        rb = (okf_s[pb][:, None] * D + np.arange(D)[None, :])
        rows.append(np.repeat(ra, D, axis=1).reshape(-1))
        cols.append(np.tile(rb, (1, D)).reshape(-1))
        vals.append(blocks.reshape(total, -1).reshape(-1))
        gsch = -np.einsum("nij,nj->ni", T3, bl[olm_s])
        for a in range(D):
            np.add.at(bp, ii[:, a], gsch[:, a])

        H = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n)).tocsr()
        if rotf:
            Hr, gr, _c = _rot_blocks(T_cw, rotf, D, n, huber=rotf_huber)
            H = H + sp.csr_matrix(Hr)
            bp = bp + gr
        if welds:
            Hw, gw, _c, _hpb, _hbb, _gb = _weld_blocks(T_cw, welds, D, n)
            H = H + sp.csr_matrix(Hw)
            bp = bp + gw
        H = H.tolil()
        for i in np.nonzero(~free)[0]:
            f0 = i * D
            H[f0:f0 + D, :] = 0
            H[:, f0:f0 + D] = 0
            for q in range(D):
                H[f0 + q, f0 + q] = 1.0
            bp[f0:f0 + D] = 0.0
        H = H.tocsr()
        dgl = H.diagonal()
        H = H + sp.diags(lam * np.maximum(dgl, 1e-9))
        last_H = H
        try:
            dxp = spl.spsolve(H.tocsc(), bp)
        except Exception:
            break
        if not np.isfinite(dxp).all():
            lam = min(lam * 10, 1e3)
            continue

        rhs = bl.copy()
        np.add.at(rhs, olm_s, -np.einsum("nij,ni->nj", Hpl,
                                         dxp.reshape(nk, D)[okf_s]))
        dxl = np.einsum("nij,nj->ni", Hlli, rhs)

        T_new = T_cw.copy()
        upd = free
        if upd.any():
            T_new[upd] = se3_exp(dxp.reshape(nk, D)[upd]) @ T_cw[upd]
        pts_new = pts + dxl
        r2, _ = residuals(T_new, pts_new)
        chi2 = float(np.sum(r2 ** 2))
        if chi2 <= chi:
            T_cw, pts = T_new, pts_new
            lam = max(lam * 0.4, 1e-7)
            if chi2 < best[2]:
                best = (T_cw.copy(), pts.copy(), chi2)
        else:
            lam = min(lam * 5.0, 1e3)
        if verbose:
            print("   [gba] it %d  chi2 %.4e -> %.4e  lam %.1e"
                  % (it, chi, chi2, lam))

    T_cw, pts = best[0], best[1]

    # ★ ONE OUTLIER PASS, THEN RE-SOLVE.
    # Huber DOWN-WEIGHTS a bad observation; it never removes it, so a
    # correspondence that is simply wrong keeps a small vote forever.  After
    # the first solve the geometry is good enough to tell "noisy" from
    # "wrong", which it was not at the start - so rejection happens here and
    # not in the front end, where it would have been a guess.
    #
    # The threshold is derived from the run's own residual distribution
    # (median + reject_sigma * scaled MAD), not typed: `max()` is banned and a
    # fixed pixel count would be a different statement on every rig.
    if passes > 1 and (not time_budget_s
                       or time.perf_counter() - t_start < 0.6 * time_budget_s):
        rr, _ = residuals(T_cw, pts)
        e = np.linalg.norm(rr, axis=1)
        med = float(np.median(e))
        mad = float(np.median(np.abs(e - med))) * 1.4826
        thr = med + reject_sigma * max(mad, 1e-6)
        keep_mask = e <= thr
        n_rejected = int((~keep_mask).sum())

        # ★ THE STEREO PRIOR IS RE-WEIGHTED BY WHAT WAS MEASURED, NOT BY WHAT
        # WAS DECLARED.  `sigma_d = 0.5 px` is the figure the depth band is
        # defined at, and it is the right Class-D starting point - but what
        # RAFT-Stereo actually delivers on a given rig is a Class-C quantity,
        # and the prior residuals measure it directly because they are already
        # expressed in units of their own sigma.
        #
        # Measured on V1_01: **0.14 px against the declared 0.50** - RAFT is
        # 3.6x better than the prior was giving it credit for, so the stereo
        # measurement was being down-weighted by that factor in every solve.
        # An under-confident prior is not the safe direction; it is information
        # thrown away.
        prior_rw = prior
        if prior is not None:
            cP, rho0P, invsigP, validP = prior
            dv = pts - cP
            dd = np.linalg.norm(dv, axis=1)
            okp = (validP > 0.5) & (dd > 1e-6)
            if okp.sum() > 50:
                rp = ((1.0 / np.maximum(dd, 1e-9) - rho0P) * invsigP)[okp]
                spread = float(np.median(np.abs(rp)) * 1.4826)
                if 0.05 < spread < 20.0:
                    prior_rw = (cP, rho0P, invsigP / spread, validP)

        if n_rejected and keep_mask.sum() > 0.5 * len(e):
            sub = global_ba(kf_ids, np.stack([se3_inv(T) for T in T_cw]), pts,
                            okf_s[keep_mask], olm_s[keep_mask],
                            ouv_s[keep_mask], rig, prior=prior_rw, rotf=rotf,
                            rotf_huber=rotf_huber,
                            welds=welds, iters=iters, fixed=fixed, lam0=lam0,
                            verbose=verbose, passes=1, want_cov=want_cov,
                            time_budget_s=(None if not time_budget_s else
                                           time_budget_s
                                           - (time.perf_counter() - t_start)))
            out_poses, out_pts, st = sub
            st.update(rms_px_before=rms0, observations=N,
                      rejected=n_rejected,
                      rejected_pct=100.0 * n_rejected / max(N, 1),
                      reject_threshold_px=thr,
                      seconds=time.perf_counter() - t_start)
            return out_poses, out_pts, st

    T_cw[:, :3, :3] = so3_project(T_cw[:, :3, :3])
    rn, _ = residuals(T_cw, pts)
    cov = _marginal_cov(last_H, nk, D, want_cov, last_free) \
        if (want_cov and last_H is not None) else {}
    # ★ SELF-CALIBRATE sigma_d.  The prior residual is expressed in units
    # of its own sigma, so its spread says directly whether the DECLARED
    # 0.5 px disparity noise is the right weight for what RAFT-Stereo
    # actually delivers on this rig.  A declared sigma is a CALIBRATED
    # SYSTEM WEIGHT - the Rosario receipt is correct calibration on the
    # wrong rig costing +522 % - so it is measured, not believed.
    meas_sigma_px = None
    if prior is not None:
        cP, rho0P, invsigP, validP = prior
        dv = pts - cP
        dd = np.linalg.norm(dv, axis=1)
        okp = (validP > 0.5) & (dd > 1e-6)
        if okp.any():
            rp = (1.0 / np.maximum(dd, 1e-9) - rho0P) * invsigP
            rp = rp[okp]
            meas_sigma_px = float(np.median(np.abs(rp)) * 1.4826
                                  * DISPARITY_NOISE_PX)
            # the SIGNED residual: a nonzero median says the depth
            # measurement disagrees with the geometry in one direction, which
            # is a BIAS and not noise.  Sim(3) scale sits at 1.005-1.014
            # across rows, so the trajectory is systematically short and this
            # is where that would show up.
            meas_bias_px = float(np.median(rp) * DISPARITY_NOISE_PX)
    return (np.stack([se3_inv(T) for T in T_cw]), pts,
            {"rms_px_before": rms0, "cov": cov,
             "measured_disparity_sigma_px": meas_sigma_px,
             "measured_disparity_bias_px": locals().get("meas_bias_px"),
             "declared_disparity_sigma_px": DISPARITY_NOISE_PX,
             "rms_px_after": float(np.sqrt(np.mean(np.sum(rn ** 2, 1)))),
             "keyframes": nk, "landmarks": nl, "observations": N,
             "rejected": n_rejected,
             "rejected_pct": 100.0 * n_rejected / max(N, 1),
             "seconds": time.perf_counter() - t_start})


def _stereo_chi(Pc_z, d_meas, fB, inv_sig):
    """chi2 of the per-observation stereo rows, Huber'd the same way.

    The measurement is a DISPARITY, so the residual is written in the inverse
    depth the disparity actually is - `rho = d / fB` - and reuses the declared
    `rho_sigma`.  Writing it in metres would give a depth-dependent weight the
    measurement does not have.
    """
    if d_meas is None:
        return 0.0
    ok = d_meas > 1e-6
    if not ok.any():
        return 0.0
    r = (1.0 / np.maximum(Pc_z[ok], 1e-6) - d_meas[ok] / fB) * inv_sig
    w = np.where(np.abs(r) <= HUBER_K, 1.0, HUBER_K / np.maximum(np.abs(r), 1e-9))
    return float(np.sum(w * r * r))


# ---- T1-g (XTEN_GBA_ACCEPT=full, 2026-10-01): SCORE WHAT IS SOLVED ----------
# global_ba_id's normal equations carry SEVEN terms: the reprojection rows, the
# per-observation stereo rows, the inertial factors, the loop welds
# (_weld_blocks), the gyro rotation factors (_rot_blocks), the landmark prior
# (bearing x, bearing y, inverse depth) and, under scale_bias, beta's own
# prior.  The landed LM accept test and the `best` bookkeeping score only the
# first three, so a step that buys a lower vision + stereo + inertial chi with
# a larger weld / rotation-factor / prior chi is ACCEPTED although it worsens
# the objective the step was solved for, and `best` keeps it.  The rule (tier
# 1, no constant): with XTEN_GBA_ACCEPT=full the seed AND every candidate are
# scored with all seven terms, each from the function that assembles it.
#
# PRE-REGISTERED (2026-10-01, before any number was looked at):
#   * the VISION term stays the UNROBUSTIFIED sum r^2 the landed accept test
#     uses (vt_weights not applied either).  The defect is the ABSENT terms;
#     re-weighting vision as well would make one flag change two things and
#     the A/B could not say which one moved the benchmark.  The Huber-solve /
#     plain-score mismatch of the vision term is shared by both arms and is
#     recorded as an open question, not fixed here.
#   * every other term is scored exactly as its block assembles it: stereo
#     (_stereo_chi) and the inertial chi (inertial_system) as landed; welds
#     e^T W e; rotation factors |w e|^2 with the --gba-robust-gyro Huber
#     weight when it is on (the IRLS "sum w r^2" convention _stereo_chi and
#     inertial_system already use); the prior sum (rp * swp)^2 over valid
#     landmarks (bearing rows plain, rho row Huber - kernels.landmark_prior_id);
#     beta (beta / BETA_SIGMA)^2 when beta is free.
#   * ONE summation order everywhere: ((((((vision + stereo) + inertial)
#     + welds) + rotf) + prior) + beta), so the seed, every candidate, `best`
#     and the provenance agree bit for bit.
#   * ACCEPT=full IMPLIES XTEN_GBA_BESTFIX (its seed - the full objective at
#     the entering state - is a superset of the BESTFIX seed); with ACCEPT
#     unset the solve is byte-identical to the landed tree whatever BESTFIX
#     says.  The provenance (lm_pass*: accept_objective, chi_seed_components,
#     chi_best_components, chi_seed_total, chi_best_total) is written in BOTH
#     modes (LOG discipline: default-on, estimates bit-identical), so a
#     default run shows what its returned state did to the full objective.
GBA_ACCEPT_ENV = "XTEN_GBA_ACCEPT"
GBA_ACCEPT_MODES = ("partial", "full")
_ACCEPT_TERMS = ("vision", "stereo", "inertial", "welds", "rotf", "prior",
                 "beta")


def gba_accept_mode(env=None):
    """T1-g: global_ba_id's LM accept objective from XTEN_GBA_ACCEPT.  unset /
    "" / "0" / "off" / "default" / "partial" -> "partial" (the landed test:
    vision + stereo + inertial); "full" -> every term the normal equations
    carry; anything else is an error (a misspelt queue flag must not silently
    measure the default).  Read per call by global_ba_id and once by
    Tracker.__init__, so a misspelt flag fails before the run, not at its end."""
    v = (os.environ if env is None else env).get(GBA_ACCEPT_ENV, "")
    v = (v or "").strip().lower()
    if v in ("", "0", "off", "default", "partial"):
        return "partial"
    if v == "full":
        return "full"
    raise RuntimeError("%s=%r is not a mode: unset/partial or full"
                       % (GBA_ACCEPT_ENV, v))


# ---- P6: AN ITERATION BUDGET, NOT A WALL CLOCK -------------
# The landed solve stops when time.perf_counter() passes a budget (the LM loop, the pass-2 gate at 0.6 of
# the budget, the recursion's remainder), so WHAT IT RETURNS depends on machine load: 17 % of runs were
# cut at 3 concurrent runs and 40 % at 5, and 4s_131 / 4s_141 / rosario_02 / rosario_06 on 5 of 5 draws.
# With XTEN_GBA_ITERBUDGET=1 global_ba_id IGNORES time_budget_s
# (no clock read decides anything) and stops each pass on
#   (a) the iteration budget it already has: `iters` LM iterations per pass, `passes` passes, or
#   (b) convergence: an ACCEPTED step, solved at a damping no larger than the pass's starting damping
#       lam0 (P6r2: a small change bought by a heavily damped step after rejections is damping-limited,
#       not converged), that changed EVERY term of the accept objective (vision, stereo, inertial, and
#       the total that was compared) by at most GBA_ITER_FTOL of its own value.  Per term,
#       not on the sum, because one term can dwarf the others by orders of magnitude (the inertial chi at
#       a 4Seasons GBA entry is 1.45e10 against a vision chi near 1e6, in an earlier internal study), and a relative
#       test on the sum would then stop while the vision term is still moving.
# GBA_ITER_FTOL is Ceres Solver's default function_tolerance (Solver::Options, 1e-6): a published solver
# default, not tuned on any row.  unset / "" / "0" / "off" -> False (landed path, byte for byte); "1" ->
# True; anything else raises (read per call here and once at Tracker construction).
GBA_ITERBUDGET_ENV = "XTEN_GBA_ITERBUDGET"
GBA_ITER_FTOL = 1e-6


def gba_iterbudget_mode(env=None):
    """P6: XTEN_GBA_ITERBUDGET -> bool (see the block above); a misspelt value is a RuntimeError."""
    v = (os.environ if env is None else env).get(GBA_ITERBUDGET_ENV, "")
    v = (v or "").strip().lower()
    if v in ("", "0", "off"):
        return False
    if v == "1":
        return True
    raise RuntimeError("%s=%r is not a mode: unset/0/off or 1"
                       % (GBA_ITERBUDGET_ENV, v))


def _iter_converged(cur, new, ftol=None):
    """P6: True when every term moved by at most ftol (default: the module's GBA_ITER_FTOL, read at call
    time) of its value at the current state (a term that is 0 must stay exactly 0)."""
    if ftol is None:
        ftol = GBA_ITER_FTOL
    for a, b in zip(cur, new):
        if abs(float(a) - float(b)) > ftol * abs(float(a)):
            return False
    return True


# ======================================================================
# THE PER-RUN STEREO DISPARITY OFFSET MACHINERY of the run-end global BA (global_ba_id's `dof=` argument; attack malaga01,
# 2026-10-02), ported into the engine 2026-10-15 as the CARRIER of XTEN_GBA_SCALECAL's joint offset state
# (the block below).
#
# Every landmark's inverse-depth prior is the stereo disparity the front end read, d_meas/fB.  A disparity offset b (px,
# `d_true = d_meas - b`, the convention of an internal note) is a CORRELATED measurement error: it shifts
# every prior of a stretch of road the same way and the trajectory takes it up as a metric-scale error that grows with the
# depth the tracker leans on.  The solver already carried a multiplicative nuisance (beta) that is free only when an IMU
# metric reference is in the solve; b is the ADDITIVE counterpart.  `dof` = {"seg": segment id per landmark, "K",
# "sigma_px" (the offset's prior sigma), "b0" (start values or None)[, "joint": True]}.  dof=None (every engine call
# unless XTEN_GBA_SCALECAL acts) = the landed solver byte for byte.
#
# NOT PORTED: the offset-only ARM XTEN_GBA_DOF itself - its segment builder (dof_segments / dof_build) and
# the tracker's wiring.  Its replays project kitti_08 tie -> ORB3 (internal notes); its reviewed port is not
# part of this package.  The offset-only branch of global_ba_id (a
# `dof` without "joint") is therefore reachable only by direct callers (tests, CPU replays).  gba_dof_mode() is kept
# only so that XTEN_GBA_DOF set together with XTEN_GBA_SCALECAL is refused at Tracker construction (one model of the
# prior's common mode at a time); XTEN_GBA_DOF alone stays what it is in this engine: a variable nothing reads.
# ======================================================================
GBA_DOF_ENV = "XTEN_GBA_DOF"


def gba_dof_mode(env=None):
    """XTEN_GBA_DOF: unset / 0 / off -> None; 'seg' -> 'seg'.  A misspelt value raises.  (the engine: read only by the
    Tracker's XTEN_GBA_SCALECAL exclusivity check - see the block above.)"""
    v = ((os.environ if env is None else env).get(GBA_DOF_ENV, "") or "").strip().lower()
    if v in ("", "0", "off", "default"):
        return None
    if v == "seg":
        return "seg"
    raise RuntimeError("%s=%r is not a mode: unset/off or seg" % (GBA_DOF_ENV, v))


# ======================================================================
# XTEN_GBA_SCALECAL (stage-5 scale-prior research, 2026-10-03): the stereo prior's COMMON-MODE error, self-calibrated
# against the inertial metric reference in the run-end global BA.
#
# Every landmark's inverse-depth prior is the stereo disparity the front end read, rho0 = d_meas / fB, and its dominant
# error is common to all of them (shared by ~1e4-1e5 landmarks) while the prior treats every landmark as independent -
# along the common direction its information is overstated ~N-fold, and it owns 73-100 % of the stiffness along the
# metric scale mode (replays of 22 rows; 100 % without an IMU).  The common mode has two physical components: an
# ADDITIVE disparity offset b (px; d_true = d_meas - b) and a MULTIPLICATIVE gain (fB, rectification focal: the
# existing nuisance beta).  They have different range dependence, so vision separates b from the trajectory scale only
# up to the gain, and the gain itself is metric: it needs a metric reference.  'imu' adds ONE global offset state b
# (prior N(0, DISPARITY_NOISE_PX^2)) carried JOINTLY with beta (free only when the inertial term is in the solve); each
# prior row is whitened in MEASUREMENT space (the predicted disparity against the measured one, whose noise is constant;
# _prior_rows_id(meas=True)):
#     r = (rho / (1 + beta) - (rho0 - b / fB)) * invsig
# and the state exists ONLY when the inertial term is in the solve: an offset corrected without its gain moves the
# metric scale by an amount nothing metric references (an internal note: kitti_08, the offset the
# LiDAR and the GT-motion parallax both confirm, corrected alone, overshoots the GT scale by 0.2 % and flips the row).
# Vision-only rows and the flag unset: the landed solver byte for byte.  Accept policy: the base's (XTEN_GBA_ACCEPT), as
# beta has always run - the joint mode does not force the full objective the offset-only XTEN_GBA_DOF forces.  The
# research evidence was measured with the calibrated preintegration covariance on: run the arm with --preint-cal.
# ======================================================================
GBA_SCALECAL_ENV = "XTEN_GBA_SCALECAL"


def gba_scalecal_mode(env=None):
    """XTEN_GBA_SCALECAL: unset / '' / 0 / off / default -> None (the landed solver); 'imu' -> the joint offset + gain
    self-calibration against the inertial term.  Anything else raises - a misspelt queue flag must not silently
    measure the default (validated at Tracker construction)."""
    v = ((os.environ if env is None else env).get(GBA_SCALECAL_ENV, "") or "").strip().lower()
    if v in ("", "0", "off", "default"):
        return None
    if v == "imu":
        return "imu"
    raise RuntimeError("%s=%r is not a mode: unset/off or imu" % (GBA_SCALECAL_ENV, v))


def scalecal_build(nl, inertial_present, env=None):
    """(dof, scale_bias) for global_ba_id from XTEN_GBA_SCALECAL: (None, None) when the flag is off OR no inertial
    term is in the solve (the caller keeps its own arguments - the landed solver); otherwise one global offset state
    (K = 1) carried jointly with beta, and beta free."""
    if gba_scalecal_mode(env) is None or not inertial_present:
        return None, None
    return ({"seg": np.zeros(int(nl), np.int64), "K": 1, "sigma_px": float(DISPARITY_NOISE_PX), "b0": None,
             "joint": True}, True)


def _prior_rows_id(prior, rho, mbear, beta, meas=False):
    """The landmark prior's rows at one state, EXACTLY as global_ba_id's
    normal equations assemble them (T1-g moved the landed inline code here
    unchanged, so the accept objective scores the very rows that are
    solved): the whitened residual `rp` (bearing x, bearing y, rho), the unit
    Jacobian `J0`, the sqrt IRLS weight `swp` (Huber on the rho row only) and
    the valid mask `ok`."""
    m0P, rho0, wbear, invsig, valid = prior
    ok = valid > 0.5
    rp = np.stack([(mbear[:, 0] - m0P[:, 0]) * wbear,
                   (mbear[:, 1] - m0P[:, 1]) * wbear,
                   (rho - rho0 * (1.0 + beta)) * invsig], 1)
    J0 = np.stack([wbear, wbear, invsig], 1)
    if meas:
        # XTEN_GBA_SCALECAL: the gain whitened in MEASUREMENT space - the predicted disparity rho / (1 + beta) against
        # the measured rho0, whose noise is constant: r = (rho / (1 + beta) - rho0) * invsig.  Exactly invariant along
        # the joint scale mode (rho / a, (1 + beta) / a); the landed form (rho - rho0 (1 + beta)) * invsig scales by
        # 1 / a there and so pulls the joint solve toward larger worlds by ~2 chi_prior / (inertial stiffness)
        q_ = 1.0 + beta
        rp[:, 2] = (rho / q_ - rho0) * invsig
        J0[:, 2] = invsig / q_
    swp = np.sqrt(np.where(np.abs(rp) <= HUBER_K, 1.0,
                           HUBER_K / np.maximum(np.abs(rp), 1e-9)))
    swp[:, 0] = 1.0        # the bearing prior is NOT robustified -
    swp[:, 1] = 1.0        # see kernels.landmark_prior_id
    return rp, J0, swp, ok


def _prior_chi_id(prior, rho, mbear, beta, meas=False):
    """T1-g: chi of the prior rows the solve assembles - the sum over VALID
    landmarks of (rp * swp)^2 (bearing rows plain, rho row Huber-IRLS)."""
    rp, _J0, swp, ok = _prior_rows_id(prior, rho, mbear, beta, meas)
    rw = rp * swp
    return float(np.sum((rw * rw)[ok]))


def _accept_extras(T_cw, rho, mbear, beta, rotf, welds, rotf_huber, prior,
                   SB, beta_sigma, d, n, beta_extra=None, meas=False):
    """T1-g: the four terms the landed accept test leaves out, at one state,
    each from the function that assembles it (chi_only: residuals and
    weights, no Jacobians).  -> {welds, rotf, prior, beta} (floats, 0.0 where
    the term is absent)."""
    return {"welds": (float(_weld_blocks(T_cw, welds, d, n, beta=beta,
                                         chi_only=True)[2])
                      if welds else 0.0),
            "rotf": (float(_rot_blocks(T_cw, rotf, d, n, huber=rotf_huber,
                                       chi_only=True)[2])
                     if rotf else 0.0),
            "prior": (_prior_chi_id(prior, rho, mbear, beta, meas)
                      if prior is not None else 0.0),
            "beta": ((float((beta / beta_sigma) ** 2) if SB else 0.0)
                     if beta_extra is None else float(beta_extra))}


def _with_extras(partial, x):
    """T1-g: the full objective from the landed partial chi ((vision + stereo)
    + inertial) and the extras, in THE one summation order."""
    return (((partial + x["welds"]) + x["rotf"]) + x["prior"]) + x["beta"]


def _accept_terms(vision, stereo, inertial, x):
    """T1-g provenance: one state's chi, all seven terms."""
    t = {"vision": float(vision), "stereo": float(stereo),
         "inertial": float(inertial)}
    for k in ("welds", "rotf", "prior", "beta"):
        t[k] = float(x[k])
    return t


def _accept_total(t):
    """T1-g: the full objective from a terms dict, in THE one summation order
    (equal bit for bit to what the accept test scored for that state)."""
    return _with_extras((t["vision"] + t["stereo"]) + t["inertial"], t)

# ---- XTEN_GBA_SCREEN:
# NO GEOMETRICALLY IMPOSSIBLE ROW ENTERS PASS 1.
# MEASURED: on v2_03 two draws (stk2, gbf3) re-acquired
# wrongly after a coast and the run-end global BA entered pass 1 at 4.3e7 /
# 9.5e5 px rms with landmarks BEHIND the cameras that observe them.  `geom`
# clamps such a depth to 1e-6 m, so the row's residual is fx * x / 1e-6 -
# ~1e8 px - and its Jacobian is evaluated at a depth that does not exist.  The
# solve weights rows by Huber, but the LM accept test and the best-state
# bookkeeping score the PLAIN sum r^2 (see T1-g above), so a vision chi of
# ~1e20 made of a few impossible rows decides every accept of pass 1.
#
# THE RULE (mode 'cheir': a correctness gate, zero constants).  Before pass 1
# (the top-level call only; the recursed pass 2 is NOT screened) every
# observation row is tested at the entering state:
#   non-finite   any of the observing-camera point, the residual or the
#                measured pixel is NaN / inf;
#   cheirality   the landmark's depth in the OBSERVING camera is <= 0 (the
#                host depth 1/rho is > 0 by construction: rho >= rho_floor).
# Screened rows are held out of PASS 1 only.  Before pass 2's rejection they
# are RESTORED and re-tested by the landed rule at pass 1's returned state,
# so a row that was impossible only because its keyframe entered wrong comes
# back once pass 1 has repaired that keyframe.  A NON-FINITE row is the one
# exception: no state repairs it, so it stays out of the whole solve.  A
# screen that removes nothing changes nothing: the solve is then
# byte-identical to the unscreened one.
# NOT PORTED: the builder's 'gross' mode (rows with a residual above
# FEATURE_SIGMA_PX * sqrt(2N)).  Its verifier REJECTED it as tier 1: at GBA
# entry that bound removes state-error rows of valid measurements, i.e. it
# is an outlier-gating policy (a tier-2 modelling choice), not a correctness
# fix.  It therefore raises like any other unknown value.
# Modes: unset / '' / '0' / 'off' -> off (landed, byte-identical); 'cheir' ->
# non-finite + cheirality.  Anything else raises (a misspelt queue flag must
# not silently measure the default).
GBA_SCREEN_ENV = "XTEN_GBA_SCREEN"
GBA_SCREEN_MODES = ("off", "cheir")


def gba_screen_mode(env=None):
    """XTEN_GBA_SCREEN -> 'off' | 'cheir'; anything else (including the
    rejected residual-bound mode) is an error.  Read per call by global_ba_id
    and once by Tracker.__init__."""
    v = (os.environ if env is None else env).get(GBA_SCREEN_ENV, "")
    v = (v or "").strip().lower()
    if v in ("", "0", "off"):
        return "off"
    if v == "cheir":
        return v
    raise RuntimeError("%s=%r is not a mode: unset/off or cheir"
                       % (GBA_SCREEN_ENV, v))


def gba_screen(r, Pc, ouv, mode="cheir"):
    """The pass-1 entry screen.  r (N,2) residuals [px] and Pc (N,3) points in
    the OBSERVING cameras, both as `geom` returns them (Pc unclamped), ouv
    (N,2) the measured pixels.  -> (keep (N,) bool, finite (N,) bool, info).
    `keep` = the rows pass 1 solves; `finite` = the rows pass 2 may test
    again (a non-finite row is out of the whole solve: no state repairs it,
    and one NaN residual would make pass 2's median - and with it every
    rejection - NaN)."""
    r = np.asarray(r, np.float64)
    Pc = np.asarray(Pc, np.float64)
    ouv = np.asarray(ouv, np.float64)
    n = int(len(r))
    finite = (np.isfinite(Pc).all(axis=1) & np.isfinite(r).all(axis=1)
              & np.isfinite(ouv).all(axis=1))
    with np.errstate(invalid="ignore"):
        behind = finite & (Pc[:, 2] <= 0.0)
        e = np.where(finite, np.sqrt(np.sum(np.where(finite[:, None], r, 0.0)
                                            ** 2, axis=1)), np.inf)
    keep = finite & ~behind
    ek = e[keep]
    info = {"mode": str(mode), "rows": n,
            "nonfinite": int((~finite).sum()), "behind": int(behind.sum()),
            "screened": int(n - int(keep.sum())),
            "max_residual_px_kept": (float(ek.max()) if len(ek) else None),
            "rms_px_kept": (float(np.sqrt(np.mean(ek ** 2))) if len(ek)
                            else None)}
    return keep, finite, info


def global_ba_id(kf_ids, poses, rho, mbear, host_idx, okf, olm, ouv, rig,
                 odisp=None,
                 prior=None, rotf=None, welds=None, iters=8, fixed=(0,),
                 lam0=1e-4, verbose=False, time_budget_s=None, passes=2,
                 reject_sigma=3.0, want_cov=0, rho_floor=1e-4,
                 inertial=None, scale_bias=False, beta0=0.0,
                 rotf_huber=None, vt_weights=None, dof=None, _screen=True):
    """The whole trajectory, in inverse depth (plan 5.4).

    ★ THE SCALAR LANDMARK BLOCK MAKES THE SCHUR COMPLEMENT A MATRIX PRODUCT.
      With a 3x3 block the reduction has to be assembled pair by pair, which
      is why the Euclidean path above builds a COO list of every ordered
      observation pair.  With a SCALAR block it is exactly a rank-`nl` update

          Hred = Hpp - U diag(1/Hll) U^T ,   U = Jpose^T Jrho

      and scipy's sparse product does it.  That is not a micro-optimisation:
      each observation now touches TWO pose blocks (observer and host), so the
      pair form would have grown 4x in entries, while `U` grows only from `c`
      to `c+1` columns per landmark because THE HOST IS THE SAME KEYFRAME FOR
      EVERY OBSERVATION OF THAT LANDMARK.  A 4x memory blow-up and a 1.3x one
      are the same algebra written two ways.

    ★ AND THE PRIOR IS LINEAR.  `r = (rho - rho_meas) * invsig` with a
      CONSTANT invsig: no range/depth conversion, no state-dependent Jacobian,
      and a strictly positive contribution to every `Hll` - so the scalar
      inverse below cannot divide by something near zero the way the 3x3
      cofactor inverse could on a far landmark.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    nk, nl, N = len(poses), len(rho), len(okf)
    n = D * nk
    fx, cx, cy = rig.fx_rect, rig.cx, rig.cy
    T_cw = np.stack([se3_inv(T) for T in poses])
    rho = np.array(rho, dtype=np.float64)
    mbear = np.asarray(mbear, np.float64)
    host_idx = np.asarray(host_idx, np.int64)
    free = np.ones(nk, bool)
    for f in fixed:
        if 0 <= f < nk:
            free[f] = False

    okf = np.asarray(okf, np.int64)
    olm = np.asarray(olm, np.int64)
    hk = host_idx[olm]                       # host pose index per observation
    keep0 = hk != okf                        # the structurally zero host rows
    okf, olm, ouv, hk = okf[keep0], olm[keep0], ouv[keep0], hk[keep0]
    N = len(okf)
    # ★ THE STEREO MEASUREMENT AT EVERY OBSERVATION, NOT ONLY AT THE HOST.
    # The host keyframe's disparity enters as the prior on rho.  Every OTHER
    # view of the same landmark also measured one, and it was computed and
    # thrown away: on V1_01, 9,471 stereo constraints used where 56,491 were
    # available.  ORB-SLAM3 and OKVIS2 both form this residual at every
    # observation of a close point.
    #
    # It is the right instrument for V1_01 specifically.  Its per-keyframe
    # rotation error sits 5.4x above its own information bound where MH_02
    # sits at 1.6x, and its landmark depths span 2.9x where MH_02's span
    # 15.8x.  A narrow depth range ENTANGLES rotation with translation: every
    # point moves the same way, so the solver cannot tell a small rotation
    # from a small sideways translation.  A tighter GYRO factor does not help
    # - measured at n=3, arm v8, rotation error 0.146 -> 0.149 deg - because
    # neither channel is short of information; the geometry is ill-conditioned.
    # A depth measurement breaks the entanglement directly: a rotation moves a
    # point laterally WITHOUT changing its depth, a translation changes it.
    d_meas = None if odisp is None else np.asarray(odisp, np.float64)[keep0]
    # only the stereo path needs these, and a synthetic rig in a unit test
    # need not carry them
    fB = float(getattr(rig, "fB", 0.0) or 0.0) if d_meas is not None else 0.0
    inv_sig_s = (1.0 / max(float(getattr(rig, "rho_sigma", 0.0) or 1e-9), 1e-12)
                 if d_meas is not None else 0.0)
    if d_meas is not None and fB <= 0.0:
        d_meas = None                    # no metric scale to write it against
    two = np.arange(2 * N)

    def geom(T_cw, rho, mbear):
        Th = se3_inv(T_cw)[host_idx]
        Ph = mbear / np.maximum(rho, 1e-12)[:, None]
        Pw = np.einsum("nij,nj->ni", Th[:, :3, :3], Ph) + Th[:, :3, 3]
        Pc = np.einsum("nij,nj->ni", T_cw[okf, :3, :3], Pw[olm]) \
            + T_cw[okf, :3, 3]
        z = np.maximum(Pc[:, 2], 1e-6)
        r = np.stack([fx * Pc[:, 0] / z + cx, fx * Pc[:, 1] / z + cy], 1) - ouv
        return r, Pc, Ph, Th

    # ---- the INERTIAL states, if this solve is tightly coupled ----------
    IN = inertial
    n_pose = n
    if IN is not None:
        from xclean.backend.preint import inertial_system
        vel = np.array(IN["vel"], float)
        g_w = np.array(IN["g_w"], float)
        bg = np.array(IN["bg"], float)
        ba = np.array(IN["ba"], float)
        T_cb = np.array(IN["T_cb"], float)
        pres = IN["pres"]
        inert_w = IN.get("inflate", 1.0)      # scalar, or [iR, iV, iP]
        inert_w = (float(inert_w) if np.ndim(inert_w) == 0
                   else np.asarray(inert_w, float))
        nx = n_pose + 3 * nk + 8
    else:
        vel = g_w = bg = ba = T_cb = pres = None
        nx = n_pose

    # ★ THE STEREO-SCALE BIAS beta (2026-08-27).  The V-room rows carry a
    # ~1 % metric under-scale that the inertial coupling provably does not
    # remove (v4 scale 1.01193 vs iner 1.01188 on identical draws): thousands
    # of tight rho priors grip the scale and the slow global scale mode never
    # moves.  beta gives that mode its own coordinate: every prior reads
    # `rho - rho_meas*(1+beta)`, so the trajectory can take the METRIC scale
    # the IMU measures while the priors stay satisfied through beta.  It is a
    # nuisance parameter with a declared prior (a stereo rig's scale is right
    # to 10 % by construction) and it is FREE ONLY WHEN AN INERTIAL METRIC
    # REFERENCE IS IN THE SOLVE - without one it would be pure gauge and is
    # locked at beta0 (KITTI: no IMU, beta stays 0).  Verified on a synthetic
    # before it was written here: a +1 % injected measurement bias returns
    # beta = -0.0100 with trajectory scale 0.9999, and the no-reference
    # control leaves beta at +0.002 with the map still shrunken.
    BETA_SIGMA = 0.1
    beta = float(beta0)
    SB = bool(scale_bias) and (prior is not None) and (IN is not None)
    # XTEN_GBA_DOF: per-segment stereo disparity offset states (see the block above gba_dof_mode)
    DOF = (dof is not None) and (prior is not None)
    # XTEN_GBA_SCALECAL: the offset state carried JOINTLY with beta (see the block above gba_scalecal_mode)
    DSB = DOF and bool(dof.get("joint"))
    if DOF:
        if DSB and not SB:
            raise RuntimeError("the joint offset + gain self-calibration needs beta free (scale_bias and the inertial term)")
        if SB and not DSB:
            raise RuntimeError("dof and scale_bias are exclusive (the multiplicative and the additive stereo nuisance)")
        dof_seg = np.asarray(dof["seg"], np.int64)
        dof_K = int(dof["K"])
        dof_sig = float(dof["sigma_px"])
        dof_fB = float(rig.fB)
        dof_b = (np.zeros(dof_K) if dof.get("b0") is None
                 else np.array(dof["b0"], np.float64))

        def _prior_at(bvec):
            m0_, rho0_, wb_, isg_, val_ = prior
            return (m0_, rho0_ - bvec[dof_seg] / dof_fB, wb_, isg_, val_)

        def _dof_pen(bvec):
            return float(np.sum((bvec / dof_sig) ** 2))

        def _nuis_pen(bvec, bet):
            """XTEN_GBA_SCALECAL: the offset shrinkage plus beta's own prior - the nuisance terms of the joint model."""
            return _dof_pen(bvec) + float((bet / BETA_SIGMA) ** 2)

    r0, Pc0, _, _ = geom(T_cw, rho, mbear)
    rms0 = float(np.sqrt(np.mean(np.sum(r0 ** 2, 1))))
    # ---- XTEN_GBA_SCREEN: hold impossible rows out of PASS 1 (see
    # gba_screen).  rms0 stays the all-row figure; the seed chi and every
    # candidate are scored on the rows the pass solves.  Off -> untouched.
    _screen_mode = gba_screen_mode() if _screen else "off"
    screen = None
    _obs_all = None
    if _screen_mode != "off":
        _keep_s, _fin_s, screen = gba_screen(r0, Pc0, ouv, _screen_mode)
        screen["applied"] = bool(0 < int(_keep_s.sum()) < N)
        if screen["applied"]:
            _obs_all = (okf[_fin_s], olm[_fin_s], ouv[_fin_s], hk[_fin_s],
                        None if d_meas is None else d_meas[_fin_s],
                        int(_fin_s.sum()))
            okf, olm, ouv, hk = (okf[_keep_s], olm[_keep_s], ouv[_keep_s],
                                 hk[_keep_s])
            if d_meas is not None:
                d_meas = d_meas[_keep_s]
            N = len(okf)
            two = np.arange(2 * N)
            r0, Pc0, _, _ = geom(T_cw, rho, mbear)
            screen["rms_px_all"] = rms0
    lam = lam0
    t_start = time.perf_counter()
    # ---- THE BEST-STATE SEED MUST BE SCORED THE WAY THE CANDIDATES ARE ----
    # MEASURED 2026-09-23: on 25 saved runs -- rosario_01/02/03/04/06, every
    # draw, and NOTHING else in 1381 runs carrying a global_ba block -- the
    # end-of-run global BA is a literal NO-OP: rms_px_after equals
    # rms_px_before to 1e-15 after 111-794 s of solving.
    #
    # Cause: `best` was seeded with the VISION-ONLY chi (sum(r0**2)) while
    # every candidate below is scored with vision + stereo + inertial.  On an
    # inertial row `best` can therefore only update when the vision chi falls
    # by more than the ENTIRE inertial chi, and where it never does, the
    # restore after the loop puts the INITIAL state back -- the whole solve is
    # discarded.  EuRoC escapes because its vision chi drops ~6x (mh_03
    # 2.48 -> 0.98 px); Rosario starts at 1.4-1.7 px and has no such margin.
    # The LM accept test itself was always consistent (`chi` gets chi_in added
    # before the comparison), so ONLY the bookkeeping was wrong.
    #
    # Seeding with the full chi can only make `best` update more often, never
    # less, because the extra terms are non-negative.  Gated anyway, so the
    # A/B runs on one unchanged tree: XTEN_GBA_BESTFIX=1.
    from xclean.backend.preint import _inertial_fix
    _ifh = HUBER_K if _inertial_fix("HUBER") else None
    chi0_vision = float(np.sum(r0 ** 2))
    chi0_stereo = float(_stereo_chi(Pc0[:, 2], d_meas, fB, inv_sig_s))
    chi0_inert = 0.0
    inert_diag = None
    if IN is not None:
        _T_wb0 = np.stack([se3_inv(T_cw[i]) @ T_cb for i in range(nk)])
        _o0 = inertial_system(pres, _T_wb0, vel, g_w, bg, ba, T_cb, nk,
                              n_pose, inflate=inert_w, huber=_ifh)
        chi0_inert = float(_o0[2]) if _o0 is not None else 0.0
        # ---- WHERE DOES THE INERTIAL CHI LIVE?  (2026-09-24)
        # Measured: inertial chi PER KEYFRAME is 11.4 on mh_03 and 13.3 on
        # v1_02 -- consistent -- but 1e6 on rosario_02/03/04, 2e7 on
        # rosario_01 and 1.4e8 on rosario_06, while rosariov2_01 sits at 160.
        # Same rig family, same declared imu_noise, and rosario_01 produced
        # BOTH 104 and 2.0e7 on two draws, so it is a run-time divergence and
        # not a calibration constant.  A sum says nothing about whether every
        # factor is slightly wrong or a handful are catastrophically wrong,
        # and those two have opposite fixes.  block_chi is already the
        # measurement half of --preint-cal; this reads it at the ENTERING
        # state and reports the shape of the distribution, per block.
        try:
            from xclean.backend.preint import block_chi
            bc = block_chi(pres, _T_wb0, vel, g_w, bg, ba)
        except Exception:
            bc = None
        if bc is not None and len(bc):
            tot = bc.sum(axis=1)
            order = np.argsort(-tot)
            k1 = max(1, len(tot) // 100)
            inert_diag = {
                "n_factors": int(len(tot)),
                "chi_per_factor_sum": float(tot.sum()),
                "block_median": [float(x) for x in np.median(bc, axis=0)],
                "block_p90": [float(x) for x in np.percentile(bc, 90, axis=0)],
                "block_p99": [float(x) for x in np.percentile(bc, 99, axis=0)],
                "block_max": [float(x) for x in bc.max(axis=0)],
                "top1pct_share": float(tot[order[:k1]].sum()
                                       / max(tot.sum(), 1e-30)),
                "top10": [{"factor": int(i), "chi": float(tot[i]),
                           "rR": float(bc[i, 0]), "rv": float(bc[i, 1]),
                           "rp": float(bc[i, 2])} for i in order[:10]],
            }
    chi0_full = chi0_vision + chi0_stereo + chi0_inert
    _bestfix = (os.environ.get("XTEN_GBA_BESTFIX") == "1") or _inertial_fix("BESTFIX")
    # ---- T1-g (XTEN_GBA_ACCEPT=full): score the seed AND every candidate
    # with all seven terms (see the block above gba_accept_mode).  The seed
    # is then the full objective at the entering state, which subsumes the
    # BESTFIX seed: ACCEPT=full implies BESTFIX.  ACCEPT unset leaves
    # `_bestfix` and the seed exactly as landed.
    _accept_full = gba_accept_mode() == "full"
    if DOF and not DSB:
        # the offset states live ONLY in the landmark prior rows, which the partial accept test (vision + stereo) does
        # not see: a step that moves b could never be judged by it.  DOF therefore scores the FULL objective.
        # (XTEN_GBA_SCALECAL keeps the base policy, as beta always has.)
        _accept_full = True
    # P6.  SOLVER: above the SuperLU size the
    # step comes from a CONVERGED solve of the same damped system (backend/gba_linsolve.py) instead of the
    # 2000-iteration Jacobi CG whose result was used converged or not.  ITERBUDGET: no wall-clock budget
    # reaches any decision of this call (see gba_iterbudget_mode).  Both read per call (a misspelt value
    # raises here and at Tracker construction); unset = the landed path, and nothing is added to the stats.
    from xclean.backend.gba_linsolve import gba_solver_mode
    _gsolve = gba_solver_mode()
    _gs_log = []
    _itb = gba_iterbudget_mode()
    if _itb:
        time_budget_s = None
    _itb_stop = False
    _itb_cur = None
    if _accept_full:
        _bestfix = True
    _xargs = (rotf, welds, rotf_huber, prior, SB, BETA_SIGMA, D, n)

    def _xargs_at(bvec):
        return (rotf, welds, rotf_huber, _prior_at(bvec), SB, BETA_SIGMA, D, n)
    # VERIFY_T2 L1 (2026-10-01): on the DEFAULT path the four extra terms are
    # provenance only, so they are NOT evaluated here, inside the wall-clock
    # budget window: a budget-limited solve must stop at the same iteration as
    # the landed tree.  The entering state is snapshotted (array copies, no
    # clock read) and `_prov_fill` evaluates seed and best terms just before
    # this call returns, after every budget-dependent decision.
    _seed_state = None
    if _accept_full:
        seed_terms = _accept_terms(chi0_vision, chi0_stereo, chi0_inert,
                                   (_accept_extras(T_cw, rho, mbear, beta,
                                                   *_xargs_at(dof_b),
                                                   beta_extra=(_nuis_pen(dof_b, beta) if DSB
                                                               else _dof_pen(dof_b)),
                                                   **({"meas": True} if DSB else {}))
                                    if DOF else
                                    _accept_extras(T_cw, rho, mbear, beta,
                                                   *_xargs)))
        chi_seed = _with_extras(chi0_full, seed_terms)
    else:
        seed_terms = None
        _seed_state = (T_cw.copy(), rho.copy(), mbear.copy(), beta)
        chi_seed = (chi0_full if _bestfix else chi0_vision)
    best = (T_cw.copy(), rho.copy(), mbear.copy(), chi_seed,
            None if vel is None else (vel.copy(), g_w.copy(), bg.copy(),
                                      ba.copy()), beta)
    best_dof = dof_b.copy() if DOF else None
    # T1-g provenance: the terms of the state held in `best` - a dict for
    # the seed, (vision, stereo, inertial, extras-or-None) after an update
    best_terms = seed_terms
    n_accepted = n_best_updates = 0
    n_rejected, thr = 0, None
    keep_mask = np.ones(N, bool)
    last_H, last_free = None, free
    n_iter_run = 0                       # LOG: iterations actually entered

    for it in range(iters):
        if time_budget_s and time.perf_counter() - t_start > time_budget_s:
            break
        n_iter_run += 1
        r, Pc, Ph, Th = geom(T_cw, rho, mbear)
        chi = float(np.sum(r ** 2)) + _stereo_chi(Pc[:, 2], d_meas, fB,
                                                  inv_sig_s)
        if _itb:                         # P6: the terms of the current state, and the damping
            _itb_cur = (float(np.sum(r ** 2)),     # this iteration's step is solved at (P6r2)
                        _stereo_chi(Pc[:, 2], d_meas, fB, inv_sig_s), lam)
        e = np.linalg.norm(r, axis=1)
        w = np.where(e <= HUBER_K, 1.0, HUBER_K / np.maximum(e, 1e-9))
        # ★ --vision-trust: per-keyframe VISION
        # down-weighting where the run's own gyro measured the vision
        # rotation wrong (w_k = min(1, median/disagreement), population =
        # this run's per-keyframe gyro-vs-vision residuals).  Log-first
        # receipt: the top-decile disagreement intervals carry 24-46% of
        # the squared rotation error (diag/trust_signal.py, S7 report Â§7).
        # Applied like the Huber weight - rows of a distrusted keyframe
        # count less; the chi convention is unchanged (Huber's own).
        if vt_weights is not None:
            w = w * np.asarray(vt_weights, float)[okf]
        sw = np.sqrt(w)[:, None]
        x, y, zz = Pc[:, 0], Pc[:, 1], np.maximum(Pc[:, 2], 1e-6)
        iz = 1.0 / zz
        A = np.zeros((N, 2, 3))
        A[:, 0, 0] = fx * iz
        A[:, 0, 2] = -fx * x * iz * iz
        A[:, 1, 1] = fx * iz
        A[:, 1, 2] = -fx * y * iz * iz
        dPo = np.zeros((N, 3, 6))
        dPo[:, 0, 0] = dPo[:, 1, 1] = dPo[:, 2, 2] = 1.0
        dPo[:, 0, 4] = zz;  dPo[:, 0, 5] = -y
        dPo[:, 1, 3] = -zz; dPo[:, 1, 5] = x
        dPo[:, 2, 3] = y;   dPo[:, 2, 4] = -x
        Jobs = np.einsum("nij,njk->nik", A, dPo) * sw[:, :, None]
        Rch = np.einsum("nij,njk->nik", T_cw[okf, :3, :3], Th[olm, :3, :3])
        ARch = np.einsum("nij,njk->nik", A, Rch)
        Pho = Ph[olm]
        dPh = np.zeros((N, 3, 6))
        dPh[:, 0, 0] = dPh[:, 1, 1] = dPh[:, 2, 2] = -1.0
        dPh[:, 0, 4] = -Pho[:, 2]; dPh[:, 0, 5] = Pho[:, 1]
        dPh[:, 1, 3] = Pho[:, 2];  dPh[:, 1, 5] = -Pho[:, 0]
        dPh[:, 2, 3] = -Pho[:, 1]; dPh[:, 2, 4] = Pho[:, 0]
        Jhost = np.einsum("nij,njk->nik", ARch, dPh) * sw[:, :, None]
        irho = 1.0 / np.maximum(rho[olm], 1e-12)
        Jlm = np.empty((N, 2, 3))
        Jlm[:, :, 0] = ARch[:, :, 0] * irho[:, None]
        Jlm[:, :, 1] = ARch[:, :, 1] * irho[:, None]
        Jlm[:, :, 2] = -np.einsum("nqk,nk->nq", ARch, Pho) * irho[:, None]
        Jlm *= sw[:, :, None]
        rw = (r * sw).reshape(-1)

        # the pose Jacobian as ONE sparse matrix: 12 nonzeros per residual row
        rowi = np.repeat(two, 6)
        colo = (okf[:, None] * D + np.arange(D)[None, :])
        colh = (hk[:, None] * D + np.arange(D)[None, :])
        Jp_sp = sp.csr_matrix(
            (np.concatenate([Jobs.reshape(-1), Jhost.reshape(-1)]),
             (np.concatenate([rowi, rowi]),
              np.concatenate([np.repeat(colo, 2, axis=0).reshape(-1),
                              np.repeat(colh, 2, axis=0).reshape(-1)]))),
            shape=(2 * N, n))
        col3 = (olm[:, None] * 3 + np.arange(3)[None, :])
        Jr_sp = sp.csr_matrix(
            (Jlm.reshape(-1),
             (np.repeat(two, 3), np.repeat(col3, 2, axis=0).reshape(-1))),
            shape=(2 * N, 3 * nl))

        # nine bincounts rather than one np.add.at over (nl,3,3): the scatter
        # form is ~1.2 M indexed adds on a real row and profiles as the
        # slowest line in the assembly
        outer = np.einsum("nqa,nqb->nab", Jlm, Jlm)
        gl = np.einsum("nqa,nq->na", Jlm, (r * sw))
        Hll = np.empty((nl, 3, 3))
        bl = np.empty((nl, 3))
        for a in range(3):
            bl[:, a] = -np.bincount(olm, weights=gl[:, a], minlength=nl)
            for b in range(3):
                Hll[:, a, b] = np.bincount(olm, weights=outer[:, a, b],
                                           minlength=nl)

        # ---- the per-observation STEREO rows, accumulated into the SAME
        # system.  `odisp=None` reproduces the visual-only answer exactly,
        # which is the ablation this is judged against - the same discipline
        # the inertial block follows.  It lands HERE, before the prior and
        # the damping, because `Hlli` is inverted a few lines below and the
        # landmark half of the Schur complement has to carry these terms.
        n_stereo = 0
        if d_meas is not None:
            okd = d_meas > 1e-6
            n_stereo = int(okd.sum())
        if n_stereo:
            zs = zz[okd]
            rs = (1.0 / zs - d_meas[okd] / fB) * inv_sig_s
            # a disparity genuinely can be an outlier - the same Huber the rho
            # prior keeps, for the same reason
            ws = np.where(np.abs(rs) <= HUBER_K,
                          1.0, HUBER_K / np.maximum(np.abs(rs), 1e-9))
            sws = np.sqrt(ws)
            cz = (-1.0 / (zs * zs)) * inv_sig_s * sws     # d(1/z)/dz, whitened
            Rch_z = Rch[okd][:, 2, :]
            Js_obs = dPo[okd][:, 2, :] * cz[:, None]
            Js_host = np.einsum("nij,njk->nik",
                                Rch[okd], dPh[okd])[:, 2, :] * cz[:, None]
            irho_s, Pho_s = irho[okd], Pho[okd]
            Js_lm = np.empty((n_stereo, 3))
            Js_lm[:, 0] = Rch_z[:, 0] * irho_s
            Js_lm[:, 1] = Rch_z[:, 1] * irho_s
            Js_lm[:, 2] = -np.einsum("nk,nk->n", Rch_z, Pho_s) * irho_s
            Js_lm *= cz[:, None]
            rsw = rs * sws
            rows = np.arange(n_stereo)
            oks, hks, lms = okf[okd], hk[okd], olm[okd]
            Jsp = sp.csr_matrix(
                (np.concatenate([Js_obs.reshape(-1), Js_host.reshape(-1)]),
                 (np.concatenate([np.repeat(rows, D), np.repeat(rows, D)]),
                  np.concatenate([
                      (oks[:, None] * D + np.arange(D)[None, :]).reshape(-1),
                      (hks[:, None] * D + np.arange(D)[None, :]).reshape(-1)]))),
                shape=(n_stereo, n))
            Jsr = sp.csr_matrix(
                (Js_lm.reshape(-1),
                 (np.repeat(rows, 3),
                  (lms[:, None] * 3 + np.arange(3)[None, :]).reshape(-1))),
                shape=(n_stereo, 3 * nl))
            outer_s = np.einsum("na,nb->nab", Js_lm, Js_lm)
            gl_s = Js_lm * rsw[:, None]
            for a in range(3):
                bl[:, a] -= np.bincount(lms, weights=gl_s[:, a], minlength=nl)
                for b in range(3):
                    Hll[:, a, b] += np.bincount(lms, weights=outer_s[:, a, b],
                                                minlength=nl)
            Jp_sp = sp.vstack([Jp_sp, Jsp]).tocsr()
            Jr_sp = sp.vstack([Jr_sp, Jsr]).tocsr()
            rw = np.concatenate([rw, rsw])

        ulb = None
        Hbb = bbeta = 0.0
        if prior is not None:
            m0P, rho0, wbear, invsig, valid = prior
            # T1-g: the rows come from _prior_rows_id (the landed inline
            # code, moved unchanged), so the full accept objective scores
            # exactly the rows assembled here
            rp, J0, swp, ok = _prior_rows_id(_prior_at(dof_b) if DOF else prior,
                                             rho, mbear, beta, **({"meas": True} if DSB else {}))
            Jpr = J0 * swp
            for c in range(3):
                Hll[ok, c, c] += (Jpr[:, c] ** 2)[ok]
            bl[ok] -= (Jpr * rp * swp)[ok]
            if SB and not DSB:
                # the beta column: each prior row is  swp2*(rho - rho0(1+b))
                # *invsig  with  d/drho = swp2*invsig,  d/dbeta =
                # -swp2*rho0*invsig.  It couples rho to beta only; poses feel
                # it through the Schur complement below.
                Jr2 = np.where(ok, invsig * swp[:, 2], 0.0)
                Jb = np.where(ok, -(rho0 * invsig) * swp[:, 2], 0.0)
                rrow = np.where(ok, rp[:, 2] * swp[:, 2], 0.0)
                ulb = np.zeros((nl, 3))
                ulb[:, 2] = Jr2 * Jb
                Hbb = float(np.sum(Jb ** 2)) + (1.0 / BETA_SIGMA) ** 2
                bbeta = float(-np.sum(Jb * rrow)) - beta / (BETA_SIGMA ** 2)
            if DOF and not DSB:
                # r_rho = (rho - (rho0 - b_s/fB)) * invsig  ->  d/db_s = +invsig/fB; the rows are the same IRLS rows
                Jr2 = np.where(ok, invsig * swp[:, 2], 0.0)
                Jb = np.where(ok, (invsig / dof_fB) * swp[:, 2], 0.0)
                rrow = np.where(ok, rp[:, 2] * swp[:, 2], 0.0)
                ulb = np.zeros((nl, 3))
                ulb[:, 2] = Jr2 * Jb
                Hbb_v = (np.bincount(dof_seg, weights=Jb ** 2, minlength=dof_K)
                         + 1.0 / dof_sig ** 2)
                bbeta_v = (-np.bincount(dof_seg, weights=Jb * rrow, minlength=dof_K)
                           - dof_b / dof_sig ** 2)
            if DSB:
                # XTEN_GBA_SCALECAL, measurement space: r = (rho / (1+beta) - (rho0 - b_s/fB)) * invsig  ->
                # d/drho = invsig / (1+beta), d/dbeta = -rho / (1+beta)^2 invsig, d/db_s = invsig / fB.  ONE nuisance
                # block [beta, b_1..b_K] (K = 1 here), reduced through the landmarks below; both columns touch only
                # the rho component of a landmark.
                q_ = 1.0 + beta
                Jr2 = np.where(ok, (invsig / q_) * swp[:, 2], 0.0)
                Jbt = np.where(ok, -(rho / (q_ * q_) * invsig) * swp[:, 2], 0.0)
                Jdf = np.where(ok, (invsig / dof_fB) * swp[:, 2], 0.0)
                rrow = np.where(ok, rp[:, 2] * swp[:, 2], 0.0)
                dsb_u_bt = Jr2 * Jbt
                dsb_u_df = Jr2 * Jdf
                dsb_H = np.zeros((dof_K + 1, dof_K + 1))
                dsb_H[0, 0] = float(np.sum(Jbt ** 2)) + (1.0 / BETA_SIGMA) ** 2
                _x = np.bincount(dof_seg, weights=Jbt * Jdf, minlength=dof_K)
                dsb_H[0, 1:] = _x
                dsb_H[1:, 0] = _x
                dsb_H[1:, 1:] += np.diag(np.bincount(dof_seg, weights=Jdf ** 2, minlength=dof_K)
                                         + 1.0 / dof_sig ** 2)
                dsb_b = np.concatenate([[float(-np.sum(Jbt * rrow)) - beta / (BETA_SIGMA ** 2)],
                                        -np.bincount(dof_seg, weights=Jdf * rrow, minlength=dof_K)
                                        - dof_b / dof_sig ** 2])
                ulb = np.zeros((nl, 3))       # non-None: the joint reduction below runs
        Hd = Hll + lam * np.maximum(
            np.trace(Hll, axis1=1, axis2=2)[:, None, None] / 3.0,
            1e-9) * np.eye(3)
        Hlli = np.linalg.inv(Hd)
        base3 = np.repeat(np.arange(nl) * 3, 9).reshape(nl, 9)
        bi = base3 + np.tile(np.repeat(np.arange(3), 3), nl).reshape(nl, 9)
        bj = base3 + np.tile(np.tile(np.arange(3), 3), nl).reshape(nl, 9)
        Hlli_sp = sp.csr_matrix(
            (Hlli.reshape(-1), (bi.reshape(-1), bj.reshape(-1))),
            shape=(3 * nl, 3 * nl))

        Hpp = (Jp_sp.T @ Jp_sp).tocsr()
        grad = -np.asarray(Jp_sp.T @ rw).ravel()
        U = (Jp_sp.T @ Jr_sp).tocsr()
        H = Hpp - (U @ Hlli_sp @ U.T).tocsr()
        bp = grad - U @ np.einsum("lcd,ld->lc", Hlli, bl).reshape(-1)

        if rotf:
            Hr, gr, _c = _rot_blocks(T_cw, rotf, D, n, huber=rotf_huber)
            H = H + sp.csr_matrix(Hr)
            bp = bp + gr
        if welds:
            Hw, gw, _c, w_Hpb, w_hbb, w_gb = _weld_blocks(
                T_cw, welds, D, n, beta=beta, want_beta=SB)
            H = H + sp.csr_matrix(Hw)
            bp = bp + gw
        else:
            w_Hpb, w_hbb, w_gb = None, 0.0, 0.0

        chi_in = 0.0
        basis = None
        if IN is not None:
            # ★ THE VISUAL SYSTEM IS EMBEDDED, NOT REWRITTEN.  The landmark
            # Schur complement above already reduced structure out onto the
            # POSE block; the inertial factors simply add their own blocks in
            # a longer vector.  Keeping it additive is what lets `inertial=
            # None` reproduce the visual-only answer byte for byte, which is
            # the ablation this build is judged against.
            # NOT `csr_matrix((data, indices, indptr), shape=(nx,nx))` - that
            # reuses an indptr of length n_pose+1 for a matrix that needs
            # nx+1, and scipy rejects it. Re-emit through COO, where the row
            # index is explicit and the shape is free.
            Hc = H.tocoo()
            Hbig = sp.coo_matrix((Hc.data, (Hc.row, Hc.col)),
                                 shape=(nx, nx)).tocsr()
            bbig = np.zeros(nx)
            bbig[:n_pose] = bp
            T_wb = np.stack([se3_inv(T_cw[i]) @ T_cb for i in range(nk)])
            out = inertial_system(pres, T_wb, vel, g_w, bg, ba, T_cb, nk,
                                  n_pose, inflate=inert_w, huber=_ifh)
            if out is not None:
                Hi, gi, chi_in, _used, basis = out
                Hbig = Hbig + Hi
                bbig = bbig + gi
            H, bp = Hbig, bbig

        # ---- the beta row/column, through the landmark Schur complement --
        wlb = None
        if SB and not DSB and ulb is not None:
            wlb = np.einsum("lcd,ld->lc", Hlli, ulb)
            Hpb = -np.asarray(U @ wlb.reshape(-1)).ravel()
            Hbb_red = Hbb - float(np.einsum("lc,lc->", ulb, wlb))
            bb_red = bbeta - float(np.einsum("lc,lc->", wlb, bl))
            if w_Hpb is not None:
                Hpb = Hpb.copy()
                Hpb[:len(w_Hpb)] += w_Hpb
                Hbb_red += w_hbb
                bb_red += w_gb
            nb = H.shape[0] + 1
            Hc = H.tocoo()
            idx = np.nonzero(Hpb)[0]
            data = np.concatenate([Hc.data, Hpb[idx], Hpb[idx],
                                   [max(Hbb_red, 1e-9)]])
            rows_ = np.concatenate([Hc.row, idx,
                                    np.full(len(idx), nb - 1), [nb - 1]])
            cols_ = np.concatenate([Hc.col, np.full(len(idx), nb - 1),
                                    idx, [nb - 1]])
            H = sp.coo_matrix((data, (rows_, cols_)), shape=(nb, nb)).tocsr()
            bp = np.concatenate([bp, [bb_red]])
        if DOF and not DSB and ulb is not None:
            wlb = np.einsum("lcd,ld->lc", Hlli, ulb)
            Wb = sp.csr_matrix((wlb.reshape(-1),
                                (np.arange(3 * nl), np.repeat(dof_seg, 3))),
                               shape=(3 * nl, dof_K))
            Hpb_m = (-(U @ Wb)).tocsr()                       # (n_pose, K)
            if H.shape[0] > Hpb_m.shape[0]:
                Hpb_m = sp.vstack([Hpb_m, sp.csr_matrix(
                    (H.shape[0] - Hpb_m.shape[0], dof_K))]).tocsr()
            Hbb_red = Hbb_v - np.bincount(
                dof_seg, weights=np.einsum("lc,lc->l", ulb, wlb), minlength=dof_K)
            bb_red = bbeta_v - np.bincount(
                dof_seg, weights=np.einsum("lc,lc->l", wlb, bl), minlength=dof_K)
            H = sp.bmat([[H, Hpb_m],
                         [Hpb_m.T, sp.diags(np.maximum(Hbb_red, 1e-9))]],
                        format="csr")
            bp = np.concatenate([bp, bb_red])
        dsb_w_bt = dsb_w_df = None
        if DSB and ulb is not None:
            # XTEN_GBA_SCALECAL: the [beta, b_1..b_K] block through the landmark Schur complement.  Each column touches
            # only the rho component (u e_rho), so Hlli u = Hlli[:, :, 2] u; the joint block is dense (K+1)^2 because
            # beta couples to every segment through the shared landmarks.
            h2 = Hlli[:, :, 2]
            dsb_w_bt = h2 * dsb_u_bt[:, None]
            dsb_w_df = h2 * dsb_u_df[:, None]
            h22 = Hlli[:, 2, 2]
            hlb = np.einsum("lj,lj->l", Hlli[:, 2, :], bl)
            Hn = dsb_H.copy()
            Hn[0, 0] -= float(np.sum(dsb_u_bt * h22 * dsb_u_bt))
            _x = np.bincount(dof_seg, weights=dsb_u_bt * h22 * dsb_u_df, minlength=dof_K)
            Hn[0, 1:] -= _x
            Hn[1:, 0] -= _x
            Hn[1:, 1:] -= np.diag(np.bincount(dof_seg, weights=dsb_u_df * h22 * dsb_u_df, minlength=dof_K))
            bn = dsb_b.copy()
            bn[0] -= float(np.sum(dsb_u_bt * hlb))
            bn[1:] -= np.bincount(dof_seg, weights=dsb_u_df * hlb, minlength=dof_K)
            Hp_bt = -np.asarray(U @ dsb_w_bt.reshape(-1)).ravel()
            if w_Hpb is not None:
                # the loop welds read beta (their measurements carry the stereo scale)
                Hp_bt = Hp_bt.copy()
                Hp_bt[:len(w_Hpb)] += w_Hpb
                Hn[0, 0] += w_hbb
                bn[0] += w_gb
            Wd = sp.csr_matrix((dsb_w_df.reshape(-1), (np.arange(3 * nl), np.repeat(dof_seg, 3))),
                               shape=(3 * nl, dof_K))
            Hp_n = sp.hstack([sp.csr_matrix(Hp_bt[:, None]), -(U @ Wd)]).tocsr()      # (n_pose, K+1)
            if H.shape[0] > Hp_n.shape[0]:
                Hp_n = sp.vstack([Hp_n, sp.csr_matrix((H.shape[0] - Hp_n.shape[0], dof_K + 1))]).tocsr()
            Hn[np.diag_indices(dof_K + 1)] = np.maximum(np.diag(Hn), 1e-9)
            H = sp.bmat([[H, Hp_n], [Hp_n.T, sp.csr_matrix(Hn)]], format="csr")
            bp = np.concatenate([bp, bn])

        H = H.tolil()
        for i in np.nonzero(~free)[0]:
            f0 = i * D
            H[f0:f0 + D, :] = 0
            H[:, f0:f0 + D] = 0
            for q in range(D):
                H[f0 + q, f0 + q] = 1.0
            bp[f0:f0 + D] = 0.0
        H = H.tocsr()
        H = H + sp.diags(lam * np.maximum(H.diagonal(), 1e-9))
        last_H = H[:n_pose, :n_pose] if (IN is not None or DOF) else H
        # ★ SuperLU DIES AT R4 SCALE, AND IT DIES NATIVELY.  madmax_B0
        # (13.2k keyframes, nx=118,862) crashed spsolve with an ACCESS
        # VIOLATION after "Can't expand MemType 1" - fill-in exhausted
        # SuperLU's memory, and a segfault cannot be caught.  The damped
        # Schur system is SPD, so above the largest size spsolve has ever
        # been validated on (60k unknowns covers every benchmark sequence
        # bit-identically, kitti_00's 40k included) the step comes from
        # Jacobi-preconditioned CG instead: matvec-only, no fill-in.  An
        # approximate step is safe here - the chi accept test and the
        # damping already guard bad steps.
        try:
            if H.shape[0] > 30_000 and _gsolve is not None:
                # P6 XTEN_GBA_SOLVER: two-level deflated PCG (exact segment blocks + the segments' rigid /
                # scale / velocity modes) to ||bp - H dx|| <= 1e-8 ||bp|| on the TRUE residual.  A step that
                # did not get there (safety cap, wall deadline when a budget applies, breakdown, setup) is
                # REPORTED in stats['gba_solver'] and NOT used: the damping is raised as for a non-finite step.
                from xclean.backend import gba_linsolve as _ls
                _gs_dl = (t_start + time_budget_s) if time_budget_s else None
                try:
                    dxall, _gs_info = _ls.solve(H, bp, T_cw, vel, free, nk, n_pose,
                                                mode=_gsolve, deadline=_gs_dl)
                except Exception as _gs_e:      # recorded; the landed handler then stops the LM
                    _gs_log.append({"it": int(it), "lam": float(lam), "converged": False,
                                    "reason": "exception %s: %s" % (type(_gs_e).__name__, _gs_e)})
                    raise
                _gs_info.update(it=int(it), lam=float(lam))
                _gs_log.append(_gs_info)
                if not _gs_info["converged"]:
                    lam = min(lam * 10, 1e3)
                    continue
            elif H.shape[0] > 30_000:
                Mpre = sp.diags(1.0 / np.maximum(H.diagonal(), 1e-12))
                try:
                    dxall, cg_info = spl.cg(H, bp, M=Mpre, maxiter=2000,
                                            rtol=1e-8, atol=0.0)
                except TypeError:       # older scipy: tol= not rtol=
                    dxall, cg_info = spl.cg(H, bp, M=Mpre, maxiter=2000,
                                            tol=1e-8, atol=0.0)
                if cg_info != 0 and not np.isfinite(dxall).all():
                    lam = min(lam * 10, 1e3)
                    continue
            else:
                dxall = spl.spsolve(H.tocsc(), bp)
        except Exception:
            break
        if not np.isfinite(dxall).all():
            lam = min(lam * 10, 1e3)
            continue
        dbeta = float(dxall[-1]) if (SB and wlb is not None) else 0.0
        ddof = dxall[-dof_K:] if (DOF and wlb is not None) else None
        if DSB and dsb_w_bt is not None:
            # XTEN_GBA_SCALECAL: [..., beta, b_1..b_K] at the tail of the step
            dbeta = float(dxall[-(dof_K + 1)])
            ddof = dxall[-dof_K:]
        dxp = dxall[:n_pose]
        dxl = np.einsum("lcd,ld->lc", Hlli,
                        bl - np.asarray(U.T @ dxp).reshape(nl, 3))
        if DSB and dsb_w_bt is not None:
            # XTEN_GBA_SCALECAL: the landmark back-substitution of both nuisances
            dxl = dxl - dbeta * dsb_w_bt - dsb_w_df * ddof[dof_seg][:, None]
        else:
            if wlb is not None and dbeta != 0.0:
                dxl = dxl - dbeta * wlb
            if ddof is not None:
                dxl = dxl - wlb * ddof[dof_seg][:, None]

        T_new = T_cw.copy()
        if free.any():
            T_new[free] = se3_exp(dxp.reshape(nk, D)[free]) @ T_cw[free]
        m_new = mbear.copy()
        m_new[:, :2] += dxl[:, :2]
        rho_new = np.maximum(rho + dxl[:, 2], rho_floor)
        vel_n = g_n = bg_n = ba_n = None
        if IN is not None:
            vel_n = vel + dxall[n_pose:n_pose + 3 * nk].reshape(nk, 3)
            o = n_pose + 3 * nk
            bg_n = bg + dxall[o:o + 3]
            ba_n = ba + dxall[o + 3:o + 6]
            if basis is not None:
                gg = g_w + basis @ dxall[o + 6:o + 8]
                g_n = gg / max(np.linalg.norm(gg), 1e-12) * GRAVITY_MAG
            else:
                g_n = g_w
        r2, Pc2, _, _ = geom(T_new, rho_new, m_new)
        # (T1-g: the two terms are kept apart for the provenance; the sum is
        # the landed expression, same operands, same order)
        cv2 = float(np.sum(r2 ** 2))
        cs2 = _stereo_chi(Pc2[:, 2], d_meas, fB, inv_sig_s)
        chi2 = cv2 + cs2
        ci2 = 0.0
        if IN is not None:
            T_wb_n = np.stack([se3_inv(T_new[i]) @ T_cb for i in range(nk)])
            o2 = inertial_system(pres, T_wb_n, vel_n, g_n, bg_n, ba_n, T_cb,
                                 nk, n_pose, inflate=inert_w, huber=_ifh)
            ci2 = (o2[2] if o2 is not None else 0.0)
            chi2 = chi2 + ci2
            chi = chi + chi_in
        xt2 = None
        if _accept_full:
            # T1-g: the current state and the candidate on the FULL
            # objective - the four missing terms from the functions that
            # assemble them (chi_only), at each state (the candidate's beta
            # is beta + dbeta: the welds and the prior read it)
            if DOF:
                _bn = dof_b + (ddof if ddof is not None else 0.0)
                chi = _with_extras(chi, _accept_extras(
                    T_cw, rho, mbear, beta, *_xargs_at(dof_b),
                    beta_extra=(_nuis_pen(dof_b, beta) if DSB else _dof_pen(dof_b)),
                    **({"meas": True} if DSB else {})))
                xt2 = _accept_extras(T_new, rho_new, m_new, beta + dbeta,
                                     *_xargs_at(_bn), beta_extra=(_nuis_pen(_bn, beta + dbeta) if DSB
                                                                  else _dof_pen(_bn)),
                                     **({"meas": True} if DSB else {}))
            else:
                chi = _with_extras(chi, _accept_extras(T_cw, rho, mbear, beta,
                                                       *_xargs))
                xt2 = _accept_extras(T_new, rho_new, m_new, beta + dbeta, *_xargs)
            chi2 = _with_extras(chi2, xt2)
        if chi2 <= chi:
            n_accepted += 1
            T_cw, rho, mbear = T_new, rho_new, m_new
            beta = beta + dbeta
            if DOF and ddof is not None:
                dof_b = dof_b + ddof
            if IN is not None:
                vel, g_w, bg, ba = vel_n, g_n, bg_n, ba_n
            lam = max(lam * 0.4, 1e-7)
            if chi2 < best[3]:
                n_best_updates += 1
                best = (T_cw.copy(), rho.copy(), m_new.copy(), chi2,
                        None if vel is None else (vel.copy(), g_w.copy(),
                                                  bg.copy(), ba.copy()), beta)
                best_terms = (cv2, cs2, ci2, xt2)       # T1-g provenance
                if DOF:
                    best_dof = dof_b.copy()
            if _itb and _itb_cur is not None:
                # P6: every term of the accept objective, and the total that was compared
                _itb_stop = (_itb_cur[2] <= lam0 and _iter_converged(
                    (_itb_cur[0], _itb_cur[1], chi_in, chi), (cv2, cs2, ci2, chi2)))
        else:
            lam = min(lam * 5.0, 1e3)
        if verbose:
            print("   [gba/id] it %d  chi2 %.4e -> %.4e  lam %.1e"
                  % (it, chi, chi2, lam))
        if SB and os.environ.get("XONE_BETA_DEBUG"):
            print("   [beta] it %d  dbeta %+.6f  beta %+.6f  chi_in %.3e  "
                  "accept %s" % (it, dbeta, beta, chi_in, chi2 <= chi))
        if _itb_stop:
            break
    # P6: this pass's iteration-budget record (flag only)
    _itb_rec = (None if not _itb else
                {"iters_max": int(iters), "iters_run": int(n_iter_run),
                 "stop": ("ftol" if _itb_stop else
                          ("iters" if n_iter_run >= int(iters) else "lm_break"))})

    T_cw, rho, mbear = best[0], best[1], best[2]
    beta = best[5]
    if DOF:
        dof_b = best_dof
    if IN is not None and best[4] is not None:
        vel, g_w, bg, ba = best[4]
    # ---- LOG (tier-1 programme, 2026-09-29): THIS pass's bookkeeping.
    # The solver recurses once for the outlier pass and the returned `lm`
    # block describes only the LAST pass; this record is kept per pass
    # (lm_pass1 / lm_pass2 / lm_passes) so T1-f can see what each pass
    # kept.  `lm` itself is unchanged.
    # T1-g provenance: the terms of the returned (best) state.  Its vision /
    # stereo / inertial were scored when it was accepted; its four extra
    # terms too under ACCEPT=full, and are evaluated here otherwise (the same
    # functions, at the restored state) - so a default pass shows what its
    # returned state did to the objective the normal equations minimised.
    _best_raw, _best_state = best_terms, None
    if _accept_full:
        if not isinstance(best_terms, dict):
            _v, _s, _i, _x = best_terms
            best_terms = _accept_terms(_v, _s, _i, _x)
    else:
        # VERIFY_T2 L1: snapshot the returned state (later code projects
        # T_cw in place); its terms are evaluated by _prov_fill at return
        if best_terms is not None and not isinstance(best_terms, dict):
            _best_state = (T_cw.copy(), rho.copy(), mbear.copy(), beta)
        best_terms = None
    lm_pass = _lm_pass_record(1, chi0_vision, chi0_stereo, chi0_inert,
                              chi0_full, _bestfix, float(best[3]),
                              n_accepted, n_best_updates, iters, n_iter_run,
                              accept_objective=("full" if _accept_full
                                                else "partial"),
                              seed_terms=seed_terms, best_terms=best_terms,
                              chi_seed=chi_seed)
    if screen is not None:
        lm_pass["screen"] = screen        # XTEN_GBA_SCREEN provenance

    def _prov_fill():
        """VERIFY_T2 L1: the default path's T1-g provenance, evaluated after
        every budget-dependent decision of this call (no clock read)."""
        if _accept_full:
            return
        sterms = _accept_terms(chi0_vision, chi0_stereo, chi0_inert,
                               _accept_extras(_seed_state[0], _seed_state[1],
                                              _seed_state[2], _seed_state[3],
                                              *_xargs))
        if _best_state is None:          # no best update: returned the seed
            bterms = sterms
        else:
            _v, _s, _i, _x = _best_raw
            bterms = _accept_terms(_v, _s, _i, _accept_extras(
                _best_state[0], _best_state[1], _best_state[2],
                _best_state[3], *_xargs))
        lm_pass["chi_seed_components"] = dict(sterms)
        lm_pass["chi_best_components"] = dict(bterms)
        lm_pass["chi_seed_total"] = float(_accept_total(sterms))
        lm_pass["chi_best_total"] = float(_accept_total(bterms))

    # ---- EXACT 1-D SCALE RELAXATION along the reprojection-invariant mode.
    # The LM debug showed why beta alone cannot get there: the global scale
    # mode is reprojection-invariant, so the damping term OWNS it and the
    # per-iteration step grows only as fast as lam anneals (dbeta rose 2.5x
    # per iteration - exactly 1/0.4 - and 10 iterations moved beta 8 % of the
    # way).  Along that mode everything is closed-form: scaling the world by
    # alpha scales t_cw, 1/rho, the velocities and the weld measurements
    # together, leaving reprojection, priors (through beta' = (1+beta)/alpha
    # - 1) and welds EXACTLY invariant - only the inertial cost and beta's
    # own prior change, so the 1-D objective is theirs alone.
    if SB and pres is not None:
        from xclean.backend.preint import inertial_system

        def _inert_chi(alpha):
            T_s = T_cw.copy()
            T_s[:, :3, 3] *= alpha
            T_wb_s = np.stack([se3_inv(T_s[i]) @ T_cb for i in range(nk)])
            o = inertial_system(pres, T_wb_s, vel * alpha, g_w, bg, ba, T_cb,
                                nk, n_pose, inflate=inert_w, huber=_ifh)
            c = o[2] if o is not None else 0.0
            bnew = (1.0 + beta) / alpha - 1.0
            return c + (bnew / BETA_SIGMA) ** 2
        alpha = 1.0
        if os.environ.get("XONE_BETA_DEBUG"):
            for a_ in (0.98, 0.99, 0.995, 1.0, 1.005, 1.01, 1.015, 1.02,
                       1.03):
                print("   [alpha-grid] %.3f -> %.6e" % (a_, _inert_chi(a_)))
        for h in (0.01, 0.003, 0.001):
            c0, cm, cp = (_inert_chi(alpha), _inert_chi(alpha - h),
                          _inert_chi(alpha + h))
            denom = cp - 2.0 * c0 + cm
            if denom <= 0:
                continue
            step = 0.5 * h * (cm - cp) / denom
            alpha = float(np.clip(alpha + step, 0.9, 1.1))
        if abs(alpha - 1.0) > 1e-6:
            T_cw[:, :3, 3] *= alpha
            rho = rho / alpha
            vel = vel * alpha
            beta = (1.0 + beta) / alpha - 1.0
            # welds are beta-scaled inside _weld_blocks, so the alpha move
            # keeps them consistent through beta' - rescaling Z here as well
            # would double-count
        scale_relax_alpha = alpha
    else:
        scale_relax_alpha = None

    # XTEN_GBA_SCREEN: the screened rows come back here and face pass 2's
    # rejection rule at the state pass 1 returned, like every other row
    if _obs_all is not None:
        okf, olm, ouv, hk, d_meas, N = _obs_all
        two = np.arange(2 * N)
    # one outlier pass, then re-solve - identical policy to the Euclidean
    # path, and the threshold is still the run's own residual distribution
    if passes > 1 and (not time_budget_s
                       or time.perf_counter() - t_start < 0.6 * time_budget_s):
        rr, _, _, _ = geom(T_cw, rho, mbear)
        e = np.linalg.norm(rr, axis=1)
        med = float(np.median(e))
        mad = float(np.median(np.abs(e - med))) * 1.4826
        thr = med + reject_sigma * max(mad, 1e-6)
        keep_mask = e <= thr
        n_rejected = int((~keep_mask).sum())
        # ★ RE-WEIGHTED ONCE, FROM THE FIRST PASS - AND THE "DEFECT 14"
        #   DIAGNOSIS ABOVE WAS AN OVER-CORRECTION, CORRECTED BY MEASUREMENT.
        #
        # The observation that started it was real: the inverse-depth rows
        # reported a measured sigma of EXACTLY 0.500 on every run where the
        # Euclidean rows reported 0.42-0.45. But that is a REPORTING artifact,
        # not evidence the re-weighting is wrong - the quoted number comes from
        # the RECURSED pass, whose prior has already been re-weighted, so its
        # spread is 1.0 by construction. This project had already written that
        # down once ("report the FIRST-pass figure") and it did not stop me
        # from reading the artifact as the defect.
        #
        # Removing it cost real accuracy: V1_01 went from 0.0424 to 0.0651 with
        # everything else held, because the declared 0.5 px is genuinely
        # under-confident against RAFT's measured 0.24 and the re-weighting is
        # what recovers that. So it is restored, applied ONCE from the
        # first-pass residual, and the FIRST-pass figure is what gets reported.
        first_pass_sigma = None
        prior_rw = prior
        if prior is not None:
            m0P, rho0P, wbP, invsigP, validP = prior
            okp = validP > 0.5
            if os.environ.get("XTEN_GBA_DEPTH_TRUST") == "1":
                # measure the spread only over landmarks that still carry a depth prior: zero-weight rows would bias
                # the median low and silently strengthen every remaining prior (depth_trust_prior in tracker.py)
                okp = okp & (invsigP > 0)
            if okp.sum() > 50:
                rho0Pe = rho0P if not DOF else rho0P - dof_b[dof_seg] / dof_fB
                rp = (((rho / (1.0 + beta) - rho0Pe) if DSB else (rho - rho0Pe * (1.0 + beta)))
                      * invsigP)[okp]
                spread = float(np.median(np.abs(rp)) * 1.4826)
                first_pass_sigma = spread * DISPARITY_NOISE_PX
                if 0.05 < spread < 20.0:
                    prior_rw = (m0P, rho0P, wbP, invsigP / spread, validP)
        if n_rejected and keep_mask.sum() > 0.5 * N:
            IN2 = None
            preint_cal = None
            if IN is not None:
                IN2 = dict(IN)
                IN2.update(vel=vel, g_w=g_w, bg=bg, ba=ba)
                # ★ PREINT-COVARIANCE CALIBRATION (arm --preint-cal).
                # diag/preint_calib.py measured the ACCEL blocks of the
                # declared-density covariance 6-9x optimistic against GT on
                # every EuRoC row (stationary datasheet densities under
                # in-flight vibration) while the rotation block is
                # consistent.  The run-conditioned form (doctrine A1):
                # measure each block's median factor chi at the pass-1
                # converged state against the DECLARED covariance and
                # inflate the accel blocks so their factors become
                # consistent in pass 2.  Rotation stays at the standing
                # global inflation - the gyro-weight thread is CLOSED (S6
                # report section 12) and this arm must not reopen it.  The
                # accel inflation is FLOORED at the standing value (pure
                # softening, monotone) and capped at 30.
                if IN.get("chi_cal"):
                    from xclean.backend.preint import block_chi
                    T_wb_c = np.stack([se3_inv(T_cw[i]) @ T_cb
                                       for i in range(nk)])
                    bc = block_chi(pres, T_wb_c, vel, g_w, bg, ba)
                    if bc is not None and len(bc) >= 20:
                        med = np.median(bc, axis=0)
                        CHI3_MED = 2.366        # median of a chi^2_3
                        base = float(IN.get("inflate", 1.0)) \
                            if np.ndim(IN.get("inflate", 1.0)) == 0 else 1.0
                        iv = float(np.clip(np.sqrt(med[1] / CHI3_MED),
                                           base, 30.0))
                        ip = float(np.clip(np.sqrt(med[2] / CHI3_MED),
                                           base, 30.0))
                        IN2["inflate"] = [base, iv, ip]
                        preint_cal = {"median_chi": [float(x) for x in med],
                                      "inflate": [base, iv, ip],
                                      "factors": int(len(bc))}
                        if verbose:
                            print("   [preint-cal] median block chi "
                                  "%.2f/%.2f/%.2f (consistent=2.37) -> "
                                  "inflate [%.2f, %.2f, %.2f]"
                                  % (med[0], med[1], med[2], base, iv, ip))
            out_poses, out_rho, out_m, st = global_ba_id(
                kf_ids, se3_inv(T_cw), rho, mbear,
                host_idx, okf[keep_mask], olm[keep_mask], ouv[keep_mask], rig,
                prior=prior_rw, rotf=rotf, rotf_huber=rotf_huber,
                welds=welds, iters=iters, vt_weights=vt_weights,
                fixed=fixed, lam0=lam0, verbose=verbose, passes=1,
                want_cov=want_cov, rho_floor=rho_floor, inertial=IN2,
                scale_bias=scale_bias, beta0=beta,
                dof=(None if not DOF else dict({"seg": dof_seg, "K": dof_K,
                                                "sigma_px": dof_sig, "b0": dof_b},
                                               **({"joint": True} if DSB else {}))),
                time_budget_s=(None if not time_budget_s else time_budget_s
                               - (time.perf_counter() - t_start)),
                _screen=False)
            st.update(stereo_rows=int(n_stereo),
                      stereo_available=int(0 if d_meas is None
                                           else len(d_meas)),
                      rms_px_before=rms0, observations=N,
                      rejected=n_rejected,
                      rejected_pct=100.0 * n_rejected / max(N, 1),
                      reject_threshold_px=thr,
                      seconds=time.perf_counter() - t_start)
            # ★ THE FIRST-PASS FIGURE IS THE ONE THAT MEANS ANYTHING.
            # After re-weighting the spread is 1.0 by construction, so the
            # recursed call's number measures the feedback loop rather than
            # RAFT. Overwrite it with what the first pass actually saw.
            if first_pass_sigma is not None:
                st["measured_disparity_sigma_px"] = first_pass_sigma
                st["measured_sigma_is_first_pass"] = True
            if preint_cal is not None:
                st["preint_cal"] = preint_cal
            # LOG: pass 1's all-row rms (pass 2's rms_px_after is on the
            # post-rejection rows and cannot show it), then the per-pass
            # blocks - the recursed call's own record becomes pass 2
            lm_pass["rms_px_after"] = float(np.sqrt(np.mean(e ** 2)))
            _prov_fill()                 # VERIFY_T2 L1: after the recursion
            _lm_pass_attach(st, lm_pass)
            if _gsolve is not None or _itb_rec is not None:
                _p6_merge(st, _gsolve, _gs_log, _itb_rec)
            return out_poses, out_rho, out_m, st

    T_cw[:, :3, :3] = so3_project(T_cw[:, :3, :3])
    rn, _, _, _ = geom(T_cw, rho, mbear)
    lm_pass["rms_px_after"] = float(np.sqrt(np.mean(np.sum(rn ** 2, 1))))
    cov = _marginal_cov(last_H, nk, D, want_cov, last_free) \
        if (want_cov and last_H is not None) else {}
    meas_sigma_px = meas_bias_px = None
    if prior is not None:
        m0P, rho0P, wbP, invsigP, validP = prior
        okp = validP > 0.5
        if okp.any():
            rho0Pf = rho0P if not DOF else rho0P - dof_b[dof_seg] / dof_fB
            rp = (((rho / (1.0 + beta) - rho0Pf) if DSB else (rho - rho0Pf * (1.0 + beta)))
                  * invsigP)[okp]
            meas_sigma_px = float(np.median(np.abs(rp)) * 1.4826
                                  * DISPARITY_NOISE_PX)
            meas_bias_px = float(np.median(rp) * DISPARITY_NOISE_PX)
    inert_out = None
    if IN is not None:
        inert_out = {"vel_median": float(np.median(np.linalg.norm(vel, axis=1))),
                     "g_w": [float(x) for x in g_w],
                     "g_magnitude": float(np.linalg.norm(g_w)),
                     "gyro_bias": [float(x) for x in bg],
                     "accel_bias": [float(x) for x in ba]}
    _prov_fill()                         # VERIFY_T2 L1: after every budget decision, no clock read
    _st_out = (
            {"rms_px_before": rms0, "cov": cov, "inertial": inert_out,
             "stereo_scale_bias": float(beta), "scale_bias_free": bool(SB),
             "scale_relax_alpha": scale_relax_alpha,
             "measured_disparity_sigma_px": meas_sigma_px,
             "measured_disparity_bias_px": meas_bias_px,
             "declared_disparity_sigma_px": DISPARITY_NOISE_PX,
             "rms_px_after": float(np.sqrt(np.mean(np.sum(rn ** 2, 1)))),
             "keyframes": nk, "landmarks": nl, "observations": N,
             "rejected": n_rejected,
             "rejected_pct": 100.0 * n_rejected / max(N, 1),
             "lm": {"chi0_vision": chi0_vision, "chi0_stereo": chi0_stereo,
                    "chi0_inert": chi0_inert, "chi0_full": chi0_full,
                    "chi_best": float(best[3]), "seed_is_full": bool(_bestfix),
                    "accepted": int(n_accepted),
                    "best_updates": int(n_best_updates),
                    "iters_allowed": int(iters)},
             "inert_diag": inert_diag,
             "inertial_fix": {"samples": _inertial_fix("SAMPLES"), "huber": _ifh is not None,
                              "biascov": _inertial_fix("BIASCOV"), "bestfix": bool(_bestfix),
                              "refused_factors": (None if IN is None else IN.get("n_refused"))},
             "seconds": time.perf_counter() - t_start,
             # LOG: per-pass blocks (a single-pass solve has one)
             "lm_pass1": lm_pass, "lm_passes": [lm_pass]})
    if DOF:
        _st_out["dof"] = {"mode": ("joint" if DSB else "seg"), "K": int(dof_K), "sigma_px": dof_sig,
                          "b_px": [float(x) for x in dof_b],
                          "n_per_segment": [int(x) for x in np.bincount(dof_seg, minlength=dof_K)]}
    return _p6_attach((se3_inv(T_cw), rho, mbear, _st_out), _gsolve, _gs_log, _itb_rec)


def _lm_pass_record(pass_index, chi0_vision, chi0_stereo, chi0_inert,
                    chi0_full, seed_is_full, chi_best, n_accepted,
                    n_best_updates, iters, iters_run,
                    accept_objective=None, seed_terms=None, best_terms=None,
                    chi_seed=None):
    """LOG (tier-1 programme, 2026-09-29): one LM pass's bookkeeping.

    `seed_is_full` says whether the best-state seed was the full
    vision + stereo + inertial chi (XTEN_GBA_BESTFIX=1) or the vision-only
    chi (default); `restored_initial` is the T1-f symptom - no best update,
    so the pass returned the state it entered with.

    T1-g (2026-10-01) appends five keys (the LOG keys above keep their
    meaning; `chi_seed` / `chi_best` stay on the scale the pass SCORED, which
    under XTEN_GBA_ACCEPT=full is the full objective, and `seed_is_full` is
    then True - ACCEPT=full implies BESTFIX):
      accept_objective     'partial' (landed: vision + stereo + inertial) or
                           'full' (all seven terms)
      chi_seed_components  {vision, stereo, inertial, welds, rotf, prior,
                           beta} at the pass's entering state
      chi_best_components  the same at the state the pass returned
      chi_seed_total       their sums in THE one order = the full objective
      chi_best_total       (NOT chi0_full, which is vision + stereo +
                           inertial and is never redefined).  A 'partial'
                           pass with chi_best_total > chi_seed_total
                           returned a state WORSE on the objective it solved."""
    rec = {"pass": int(pass_index),
           "chi0_vision": float(chi0_vision),
           "chi0_stereo": float(chi0_stereo),
           "chi0_inert": float(chi0_inert),
           "chi0_full": float(chi0_full),
           "chi_seed": float(chi0_full if seed_is_full else chi0_vision),
           "chi_best": float(chi_best),
           "seed_is_full": bool(seed_is_full),
           "accepted": int(n_accepted),
           "best_updates": int(n_best_updates),
           "restored_initial": int(n_best_updates) == 0,
           "iters_allowed": int(iters),
           "iters_run": int(iters_run)}
    if chi_seed is not None:
        rec["chi_seed"] = float(chi_seed)
    if accept_objective is not None:
        rec["accept_objective"] = str(accept_objective)
        rec["chi_seed_components"] = (None if seed_terms is None
                                      else dict(seed_terms))
        rec["chi_best_components"] = (None if best_terms is None
                                      else dict(best_terms))
        rec["chi_seed_total"] = (None if seed_terms is None
                                 else float(_accept_total(seed_terms)))
        rec["chi_best_total"] = (None if best_terms is None
                                 else float(_accept_total(best_terms)))
    return rec


def _lm_pass_attach(st, outer):
    """Attach the per-pass blocks to the stats of a recursed (two-pass)
    solve: `outer` is pass 1, the recursed call's own record is pass 2.
    `lm` is left exactly as it was (it describes the last pass)."""
    inner = st.get("lm_pass1")
    passes = [outer]
    if inner is not None:
        inner = dict(inner)
        inner["pass"] = int(outer["pass"]) + 1
        passes.append(inner)
    st["lm_pass1"] = passes[0]
    if len(passes) > 1:
        st["lm_pass2"] = passes[1]
    st["lm_passes"] = passes


def _gs_summary(mode, passes):
    """P6 XTEN_GBA_SOLVER: the per-pass solve records and their totals."""
    recs = [r for p in passes for r in p]
    return {"mode": mode, "passes": passes, "solves": len(recs),
            "converged": int(sum(1 for r in recs if r.get("converged"))),
            "unconverged": int(sum(1 for r in recs if not r.get("converged"))),
            "iters_max": int(max([r.get("iters", 0) for r in recs] or [0])),
            "iters_total": int(sum(r.get("iters", 0) for r in recs)),
            "seconds": float(sum(r.get("t_setup", 0.0) + r.get("t_iter", 0.0)
                                  for r in recs))}


def _itb_summary(passes):
    """P6 XTEN_GBA_ITERBUDGET: the per-pass records (iters_max, iters_run, stop) and their totals."""
    return {"ftol": GBA_ITER_FTOL, "passes": passes,
            "lm_iters_run": int(sum(p["iters_run"] for p in passes)),
            "converged_passes": int(sum(1 for p in passes if p["stop"] == "ftol"))}


def _p6_attach(out, gs_mode, gs_log, itb_rec):
    """P6: a single-pass return.  `out` is returned untouched unless a flag is set: the solver record only
    if the new solve ran (a system above 30k unknowns; below that size the flag is a structural no-op),
    the iteration-budget record whenever XTEN_GBA_ITERBUDGET=1."""
    if gs_mode is not None and gs_log:
        out[3]["gba_solver"] = _gs_summary(gs_mode, [gs_log])
    if itb_rec is not None:
        out[3]["iterbudget"] = _itb_summary([itb_rec])
    return out


def _p6_merge(st, gs_mode, gs_log, itb_rec):
    """P6: a two-pass return - pass 1's records in front of the recursed pass's."""
    if gs_mode is not None:
        inner = (st.get("gba_solver") or {}).get("passes") or []
        if gs_log or inner:
            st["gba_solver"] = _gs_summary(gs_mode, [gs_log] + list(inner))
    if itb_rec is not None:
        inner = (st.get("iterbudget") or {}).get("passes") or []
        st["iterbudget"] = _itb_summary([itb_rec] + list(inner))


def _scale_weld(Z, alpha):
    """A weld is measured against the map, so it carries the map's scale."""
    Z2 = np.array(Z, dtype=np.float64).copy()
    Z2[:3, 3] *= alpha
    return Z2


def _rot_blocks(T_cw, factors, d, n, huber=None, chi_only=False):
    # T1-g (2026-10-01): the third return value is now the factors' chi,
    # sum |w e|^2 with w the (Huber-reduced, under --gba-robust-gyro) weight
    # actually assembled - it was a constant 0.0 and every caller discards
    # it, so H and g are unchanged bit for bit.  `chi_only=True` returns
    # (None, None, chi) without forming a Jacobian: the full accept
    # objective's scorer, the same loop, the same residual and weight.
    # ★ SPARSE, NOT DENSE (2026-08-30): a dense (n,n) here is 36.4 GiB at
    # rosariov2_01's 11.7k keyframes (MemoryError) and 50 GiB at madmax_B0's
    # 13.2k.  The factors touch only consecutive-pair 3x3 blocks; the COO
    # assembly below sums duplicate entries exactly like the dense += did.
    import scipy.sparse as sp
    tr_, tc_, tv_ = [], [], []
    idx3 = np.arange(3)

    def _acc(r0, c0, blk):
        tr_.append((r0 + idx3)[:, None] + np.zeros((1, 3), np.int64))
        tc_.append(c0 + idx3[None, :] + np.zeros((3, 1), np.int64))
        tv_.append(blk)
    g = np.zeros(n)
    chi = 0.0
    for (a, b, M, sig) in factors:
        Ri, Rj = T_cw[a][:3, :3], T_cw[b][:3, :3]
        A = Ri @ Rj.T
        e = so3_log(M.T @ A)
        w = 1.0 / max(sig, 1e-9)
        # ★ --gba-robust-gyro: the bias-sigma saga mapped a hard
        # ROW-CLASS CONFLICT - the scale rows (v1_02/v1_03/mh_05) want tight
        # gyro factors, the rotation-drift trio (v2_01/v2_02/mh_04) rejects
        # every tight global form (stderr +14.9 %, bkal +14..30 %).  No
        # global sigma serves both; the standard resolution is the robust
        # kernel PER FACTOR: full weight while the whitened residual is
        # inside HUBER_K, down-weighted where the factor fights the visual
        # consensus.  These are MEASUREMENTS (relative rotations), not
        # priors - the S3 "Huber on a prior releases what it holds" lesson
        # does not apply here.
        if huber is not None:
            en = w * float(np.linalg.norm(e))
            if en > huber:
                w = w * float(np.sqrt(huber / en))
        r = w * e
        chi += float(np.dot(r, r))          # T1-g
        if chi_only:
            continue
        J = so3_jr_inv(e)
        J = J[0] if J.ndim == 3 else J
        Ja, Jb = w * (J @ A.T), -w * J
        ra, rb = a * d + 3, b * d + 3
        _acc(ra, ra, Ja.T @ Ja)
        _acc(ra, rb, Ja.T @ Jb)
        _acc(rb, ra, Jb.T @ Ja)
        _acc(rb, rb, Jb.T @ Jb)
        g[ra:ra + 3] -= Ja.T @ r
        g[rb:rb + 3] -= Jb.T @ r
    if chi_only:
        return None, None, chi
    if not tv_:
        return sp.csr_matrix((n, n)), g, chi
    H = sp.coo_matrix((np.concatenate([v.reshape(-1) for v in tv_]),
                       (np.concatenate([r.reshape(-1) for r in tr_]),
                        np.concatenate([c.reshape(-1) for c in tc_]))),
                      shape=(n, n)).tocsr()
    return H, g, chi


def _weld_blocks(T_cw, welds, d, n, beta=0.0, want_beta=False,
                 chi_only=False):
    """★ THE LOOP WELDS, CARRIED INTO THE GLOBAL SOLVE.

    Without this, the pose graph and the bundle adjustment disagree about the
    same run.  Loop closure ties two distant keyframes together in the PGO, but
    the global BA sees only reprojection terms - and our loop closure does not
    MERGE landmarks across the loop, so nothing in the bundle knows the two
    ends are the same place.  The global solve is then free to relax the
    correction the loop just made, which is exactly what the measurement
    showed: loop alone 0.0620, loop + global BA 0.0666.

    A weld is a full SE(3) relative-pose measurement `Z` between keyframes i
    and j, carrying **the weight its evidence earned** (plan 5.5: weights, not
    booleans).

        e = log( Z^-1  (T_wc_i^-1 T_wc_j) )

    ★ THE CHART IS NOT THE POSE GRAPH'S, AND COPYING ITS JACOBIAN WAS A
      SIGN ERROR - DEFECT 13, found by finite difference on 2026-08-26.
      `loop/closer.py:pose_graph` stores **T_wc** and perturbs on the RIGHT
      (`T <- T exp(d)`), where the Jacobians are `J_i = -Ad(T_j^-1 T_i)`,
      `J_j = +I`.  This solver stores **T_cw** and perturbs on the LEFT
      (`T_cw <- exp(d) T_cw`), which flips both:

        T_cw_i <- exp(d) T_cw_i   =>   T_wc_i^-1 <- exp(d) T_wc_i^-1
        =>  J_i = +Ad(T_wc_j^-1 T_wc_i),   J_j = -I

      Both being flipped left `H = J^T W J` IDENTICAL - so the defect was
      invisible in the Hessian - while negating `g = -J^T W e`.  Every weld
      was therefore pushing the global solve AWAY from the closure the pose
      graph had just made, in proportion to the weight the weld had earned.
      The file's own recorded measurement is the symptom: "loop alone 0.0620,
      loop + global BA 0.0666".  `_rot_blocks` above is native to this chart
      and finite-differences clean, which is why only this one was wrong.
      Pinned by `tests/test_ba_id.py::test_weld_gradient_chart`.

    The `Jr^-1(e)` factor of the exact form is still omitted, as in the pose
    graph: it is second order in a residual that is small by construction.
    """
    from xclean.core.lie import se3_log as _log, adjoint as _ad
    import scipy.sparse as sp
    # sparse for the same reason as _rot_blocks (dense (n,n) is tens of GiB
    # at R4 scale); COO duplicate-sum == the dense += it replaces
    wr_, wc_, wv_ = [], [], []
    idx6 = np.arange(6)
    g = np.zeros(n)
    eye6 = np.eye(6)
    # ★ A WELD IS A MAP-SCALE MEASUREMENT (2026-08-27).  Its translation was
    # measured against the landmark map, so it carries the same fractional
    # scale bias the stereo priors do: the effective measurement is
    # `Z_t / (1 + beta)`.  Holding weld translations FIXED while beta frees
    # the priors left the welds as the last thing gripping the wrong scale -
    # measured: the joint solve equilibrated at beta = -0.0038 of a true
    # -0.009 on V1_02, with the weld stiffness comparable to the inertial
    # pull.  The beta column pieces are returned so the caller can extend the
    # reduced system.
    Hpb = np.zeros(n)
    hbb = 0.0
    gb = 0.0
    h = 1e-6
    # T1-g (2026-10-01): the third return value is now the welds' chi,
    # sum e^T W e (no robust kernel on a weld) - it was a constant 0.0 and
    # every caller discards it, so H, g and the beta pieces are unchanged bit
    # for bit.  `chi_only=True` returns (None, None, chi, None, 0.0, 0.0)
    # without forming a Jacobian (the full accept objective's scorer).
    chi = 0.0
    for (a, b, Z, w) in welds:
        Zb = np.array(Z, dtype=np.float64).copy()
        Zb[:3, 3] /= (1.0 + beta)
        Ti = se3_inv(T_cw[a])          # T_wc for keyframe a
        Tj = se3_inv(T_cw[b])
        rel = se3_inv(Ti) @ Tj
        e = _log(se3_inv(Zb) @ rel)
        # `w` is either a scalar or the derived 6-vector of information
        W = np.diag(np.asarray(w, dtype=float)) if np.ndim(w) else \
            np.eye(6) * float(w)
        chi += float(e @ (W @ e))           # T1-g
        if chi_only:
            continue
        Ja = _ad(se3_inv(Tj) @ Ti)          # +Ad, see the chart note above
        Jb = -eye6
        Jbeta = None
        if want_beta:
            Z2 = np.array(Z, dtype=np.float64).copy()
            Z2[:3, 3] /= (1.0 + beta + h)
            Jbeta = (_log(se3_inv(Z2) @ rel) - e) / h
        for (bi, J1) in ((a, Ja), (b, Jb)):
            g[bi * d:bi * d + 6] -= J1.T @ (W @ e)
            for (bj, J2) in ((a, Ja), (b, Jb)):
                wr_.append((bi * d + idx6)[:, None]
                           + np.zeros((1, 6), np.int64))
                wc_.append(bj * d + idx6[None, :]
                           + np.zeros((6, 1), np.int64))
                wv_.append(J1.T @ W @ J2)
            if Jbeta is not None:
                Hpb[bi * d:bi * d + 6] += J1.T @ (W @ Jbeta)
        if Jbeta is not None:
            hbb += float(Jbeta @ (W @ Jbeta))
            gb -= float(Jbeta @ (W @ e))
    if chi_only:
        return None, None, chi, None, 0.0, 0.0
    if not wv_:
        return sp.csr_matrix((n, n)), g, chi, Hpb, hbb, gb
    H = sp.coo_matrix((np.concatenate([v.reshape(-1) for v in wv_]),
                       (np.concatenate([r.reshape(-1) for r in wr_]),
                        np.concatenate([c.reshape(-1) for c in wc_]))),
                      shape=(n, n)).tocsr()
    return H, g, chi, Hpb, hbb, gb


def _marginal_cov(H, nk, d, want, free):
    """Marginal 6x6 covariance for a SAMPLE of keyframes, from the reduced
    camera system.

    `H` is already the Schur complement, so its inverse is the joint covariance
    of the poses with the STRUCTURE MARGINALISED OUT - which is the right
    object: a pose uncertainty that ignored the landmarks it was solved from
    would be a statement about arithmetic, not about the estimate.

    A sample rather than all of them, because each block costs `d` sparse
    solves and the point is a consistency STATISTIC, not a per-keyframe
    product.  The gauge keyframe is excluded: its covariance is zero by
    construction and including it would flatter the mean.
    """
    import scipy.sparse.linalg as spl
    ids = [i for i in range(nk) if free[i]]
    if not ids:
        return {}
    # SuperLU segfaults (natively, uncatchably) on R4-scale systems -
    # the madmax_B0 receipt.  The covariance export is a consistency
    # STATISTIC, not a deliverable; above the validated size it is
    # skipped rather than risked.
    if H.shape[0] > 30_000:
        return {}
    step = max(1, len(ids) // max(want, 1))
    ids = ids[::step][:want]
    try:
        lu = spl.splu(H.tocsc())
    except Exception:
        return {}
    out = {}
    n = H.shape[0]
    for i in ids:
        C = np.zeros((d, d))
        ok = True
        for c in range(d):
            e = np.zeros(n)
            e[i * d + c] = 1.0
            try:
                col = lu.solve(e)
            except Exception:
                ok = False
                break
            C[:, c] = col[i * d:i * d + d]
        if ok and np.isfinite(C).all():
            out[i] = 0.5 * (C + C.T)
    return out
