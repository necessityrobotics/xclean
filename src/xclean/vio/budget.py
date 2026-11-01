#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.2 W3 WINDOW BUDGET: which landmarks the per-frame window solve carries, so that the solve meets B1b
 without losing accuracy.  Also the D4 row-band rule and the 1-D stereo cost law.

Design:

1. **Pose-free landmarks leave the Schur system (exact).** A landmark whose only valid observation is its host's
   left observation has a zero pose Jacobian (t == h, CONVENTIONS.md), so its Schur block is zero: it adds nothing to
   the reduced pose system H, b.  It still costs time (build_system zeroes its Hlp row and runs the full Jacobian,
   back_substitute and apply_update loop it, visual_cost evaluates it): 70-75 % of a window's landmarks are pose-free.
   Its own 3 x 3 Gauss-Newton step and its cost are closed form.  window.solve accepts a step iff c_new < c_tot over
   ALL landmarks, so the pose-free terms take part in that decision (their step never raises their cost): the EXACT
   form (PosefreeLM, 2026-10-26) keeps their cost in c_tot and c_new of every iteration and moves their state with the
   solve's own accept decisions; final_linearisation's marginals follow at lam = 0.  The 10-24 form (posefree_replay
   after the solve, on the accept pattern of the coupled-only solve) is exact only when the pose-free terms never
   decide an accept (tests: test_caseA_replay_vs_inloop builds windows where it departs).  In floating point the
   in-loop form differs from the full solve only in the summation order of the total cost.
2. **Cost model (class B: measured on THIS machine).** One solve at LM_ITERS iterations costs
       t = c0 + cL L + cM M + cS sum_l k_l^2 + cN (n/10)^2 + cI n_imu      [ms]
   L landmarks, M valid observations, k_l the slots landmark l couples in the Schur step (host + observing slots), n the
   reduced state dimension, n_imu the IMU factors.  fit_cost: non-negative least squares on the relative error over
   6000 timed sub-windows (the current engine's windows at fractions 1, 3/4, 1/2, 1/4 of their pose-coupled landmarks;
   tools/w3_costfit.py), pinned to one logical CPU of a quiet machine.
3. **The cap (class B: the bar and the measured costs).** A window is admitted at predicted cost t_cap with
       t_cap = min(BAR_P95 / q95, BAR_P99 / q99),
   q95 / q99 the 95th / 99th percentile of measured / predicted on the fit sample: a window at the cap stays under the
   p95 bar at its own 95 % timing noise and under the p99 bar at its 99 % noise.  Over the cap, pose-coupled landmarks
   are dropped (from THIS solve only: their observations stay in the window and they compete again next frame) in
   reverse priority:
     (i)  landmarks observed (left camera, valid) in the NEWEST state first: they are the visual constraints of the pose
          being published (the KP1 condition, >= 75 newest-frame observations, is this set);
     (ii) then by information rank, the number of distinct non-host observing slots (k_l - 1; with the stereo prior a
          landmark with j such slots contributes rank 2 j to the pose information);
     (iii) ties: the older landmark first (creation order).
   Deterministic (no wall clock in the engine: the trajectory never depends on load).
4. **3 vs 5 iterations** is the ladder's next rung, used only if the cap fails.
   A later design rule binds items 3-4: they may change the solve only where their B5 cost is measured on the design
   rows and stays within 10 % per row; item 1 is exact.  The decision is in our design notes.

D4 (P1.1, row band): `band_from_dy` (mixture rule), `stereo_cost_ms` (the CPU module's measured cost law) and
`STEREO_GPU_MS` (the integrated GPU front's); the derivation is in our design notes.

Constants: BAR_* are the bar.  COST_* / Q* are class B (machine measurements, re-measured with
tools/w3_costfit.py on a new machine; they are stored with the CPU they were measured on).  No dataset constant.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from xclean.core.derive import HUBER_K
from xclean.vio.window import _cam_poses, _huber, _mtm, obs_residual

BAR_P95_MS = 10.0             # the live benchmark rules B1b
BAR_P99_MS = 12.0

# Class B, measured 2026-10-26 15:30-15:31 on logical CPU 4 (Ryzen 9 9950X; the protocol CPU, outside the
# engine cores 8-15), QUIET machine (no python process above 5 % of a core, pinned CPU 8-14 %, total 2-3 %; W3
# timing session, TIMING_ACTIVE held), NUMBA_NUM_THREADS=1, window.py md5 43e5f697 (p12covv snapshot); 6000 timed
# sub-windows (fractions 1, 3/4, 1/2, 1/4 of the pose-coupled landmarks) from the current engine's pickles
# p12d_w3_pkl_{mh_04, v2_03_d3off, v2_03} (tools/w3_costfit.py -> data/p12_w3_costfit5_q.json, p12_w3_costfit3_q.json;
# fit = fit_cost).  One sample per fit is a first-use outlier (mh_04 window 87, the first VI window: about 2 s of
# one-time compilation in a fresh process); its relative residual is bounded by 1 in the fit.  ms per unit.  The 10-24
# fit (CPU 28, others' load elsewhere) gave the same cap within 0.02 %.
COST = {
    5: {"c0": 0.240994, "cL": 0.0009617, "cM": 0.0012140, "cS": 0.00019900, "cN": 0.010666, "cI": 0.195399,
        "q95": 1.0452, "q99": 1.0769, "cpu": 4},
    3: {"c0": 0.145718, "cL": 0.0005735, "cM": 0.0009291, "cS": 0.00010937, "cN": 0.006361, "cI": 0.123755,
        "q95": 1.0553, "q99": 1.1015, "cpu": 4},                  # ladder rung 2
}


def t_cap(iters=5):
    c = COST[iters]
    return min(BAR_P95_MS / c["q95"], BAR_P99_MS / c["q99"])


COST_FEATURES = ("c0", "cL", "cM", "cS", "cN", "cI")


def cost_design(L, M, S2, n, nimu):
    """Feature row(s) of the cost model, in COST_FEATURES order."""
    L = np.asarray(L, float)
    return np.stack([np.ones_like(L), L, np.asarray(M, float), np.asarray(S2, float),
                     (np.asarray(n, float) / 10.0) ** 2, np.asarray(nimu, float)], axis=-1)


def fit_cost(samples):
    """The class-B cost model from timed sub-windows (tools/w3_costfit.py samples: dicts with L, M, S2, n, nimu, ms).
    Non-negative least squares on the RELATIVE error (each row divided by its measured time: the cap is a ratio rule,
    and a cost cannot be negative), then q95 / q99 = the 95th / 99th percentile of measured / predicted.
    -> {c0, cL, cM, cS, cN, cI, q95, q99, n, rel_rmse}."""
    from scipy.optimize import nnls
    X = cost_design([s["L"] for s in samples], [s["M"] for s in samples], [s["S2"] for s in samples],
                    [s["n"] for s in samples], [s["nimu"] for s in samples])
    y = np.array([s["ms"] for s in samples], float)
    w = 1.0 / y
    c, _ = nnls(X * w[:, None], y * w)
    r = y / (X @ c)
    out = {k: float(v) for k, v in zip(COST_FEATURES, c)}
    out.update(q95=float(np.percentile(r, 95)), q99=float(np.percentile(r, 99)), n=int(len(y)),
               rel_rmse=float(np.sqrt(np.mean((r - 1.0) ** 2))))
    return out


# ------------------------------------------------------------------------------------- Problem-level helpers
def obs_landmark(pb):
    """Local landmark index of every observation of a window Problem."""
    return np.repeat(np.arange(pb.lmp.shape[0], dtype=np.int64), np.diff(pb.ob_start))


def pose_coupled(pb):
    """(L,) bool: the landmark has >= 1 valid observation in a slot other than its host."""
    L = pb.lmp.shape[0]
    if L == 0:
        return np.zeros(0, bool)
    loc = obs_landmark(pb)
    nh = pb.ob_valid & (pb.ob_pose != pb.lm_host[loc])
    return np.bincount(loc[nh], minlength=L) > 0


def touched_slots(pb):
    """(L,) number of distinct window slots a landmark couples in the Schur step: its host plus every other slot with a
    valid observation (0 for a pose-free landmark)."""
    L = pb.lmp.shape[0]
    if L == 0:
        return np.zeros(0, np.int64)
    loc = obs_landmark(pb)
    nh = pb.ob_valid & (pb.ob_pose != pb.lm_host[loc])
    P = pb.Rwb.shape[0]
    key = np.unique(loc[nh] * P + pb.ob_pose[nh])
    k = np.bincount(key // P, minlength=L)
    return np.where(k > 0, k + 1, 0).astype(np.int64)


def state_dim(pb):
    P = pb.Rwb.shape[0]
    return int(6 * P if pb.dims is None else np.sum(pb.dims))


def features(pb):
    """Size features of a window Problem for the cost model."""
    k = touched_slots(pb)
    return {"P": int(pb.Rwb.shape[0]), "L": int(pb.lmp.shape[0]), "M": int(np.sum(pb.ob_valid)),
            "Mtot": int(len(pb.ob_pose)), "S2": int(np.sum(k * k)), "n": state_dim(pb),
            "nimu": int(len(pb.imu) if pb.imu else 0)}


def subproblem(pb, keep):
    """A new Problem holding only the landmarks keep (bool (L,)) and their observations, in the same order the engine's
    `_problem(lm_select=...)` builds them; poses, prior and IMU shared by reference (copy before solving)."""
    from xclean.vio import window as Wn
    keep = np.asarray(keep, bool)
    idx = np.nonzero(keep)[0]
    cnt = np.diff(pb.ob_start)
    om = keep[obs_landmark(pb)]
    ob_start = np.zeros(len(idx) + 1, np.int64)
    np.cumsum(cnt[idx], out=ob_start[1:])
    kw = {k: getattr(pb, k) for k in Wn.Problem.__slots__}
    for k in ("lmp", "lm_host", "lm_mask", "rho_meas", "rho_isig"):
        kw[k] = np.ascontiguousarray(getattr(pb, k)[idx])
    for k in ("ob_pose", "ob_cam", "ob_z", "ob_isig", "ob_valid"):
        kw[k] = np.ascontiguousarray(getattr(pb, k)[om])
    kw["ob_start"] = ob_start
    return Wn.Problem(**kw)


# --------------------------------------------------------------------------------------------- the cost model
def predict_ms(L, M, S2, n, nimu, iters=5):
    c = COST[iters]
    return c["c0"] + c["cL"] * L + c["cM"] * M + c["cS"] * S2 + c["cN"] * (n / 10.0) ** 2 + c["cI"] * nimu


def select(coupled, in_newest, k, m, n, nimu, iters=5, cap_ms=None):
    """The budget rule on per-landmark arrays (one entry per alive landmark, in creation order).
    coupled: pose-coupled (bool); in_newest: valid left observation in the newest state (bool); k: touched slots
    (host + observing slots; 0 if pose-free); m: valid observations; n: state dimension; nimu: IMU factors.
    Returns (keep bool, info dict).  keep = coupled, minus the lowest-priority landmarks while the predicted cost
    exceeds the cap."""
    c = COST[iters]
    cap = t_cap(iters) if cap_ms is None else float(cap_ms)
    coupled = np.asarray(coupled, bool)
    in_newest = np.asarray(in_newest, bool)
    k = np.asarray(k, np.int64)
    m = np.asarray(m, np.int64)
    base = c["c0"] + c["cN"] * (n / 10.0) ** 2 + c["cI"] * nimu
    cl = c["cL"] + c["cM"] * m + c["cS"] * (k * k)
    keep = coupled.copy()
    pred_all = base + float(cl[coupled].sum())
    info = {"L": int(len(coupled)), "L_posefree": int((~coupled).sum()), "L_capped": 0, "pred_ms": pred_all,
            "pred_ms_uncapped": pred_all, "cap_ms": cap, "bound": False}
    if pred_all <= cap:
        return keep, info
    idx = np.nonzero(coupled)[0]
    order = idx[np.lexsort((idx, -k[idx], ~in_newest[idx]))]      # newest first, then k desc, then older first
    cum = base + np.cumsum(cl[order])
    n_keep = int(np.searchsorted(cum, cap, side="right"))
    keep[:] = False
    keep[order[:n_keep]] = True
    info.update(L_capped=int(len(order) - n_keep), pred_ms=float(cum[n_keep - 1]) if n_keep else base, bound=True,
                newest_cut=int(in_newest[order[n_keep:]].sum()))
    return keep, info


def select_problem(pb, iters=5, cap_ms=None):
    """`select` on a window Problem (offline re-timing / tests).  The newest state is the last slot (the engine's slot
    order: keyframes, then frames, oldest -> newest)."""
    L = pb.lmp.shape[0]
    P = pb.Rwb.shape[0]
    loc = obs_landmark(pb)
    k = touched_slots(pb)
    m = np.bincount(loc[pb.ob_valid], minlength=L)
    nw = pb.ob_valid & (pb.ob_pose == P - 1) & (pb.ob_cam == 0)
    in_newest = np.bincount(loc[nw], minlength=L) > 0
    return select(k > 0, in_newest, k, m, state_dim(pb), len(pb.imu) if pb.imu else 0, iters, cap_ms)


# ------------------------------------------------------------------------------------------ engine-side helpers
def engine_arrays(eng, alive):
    """Per alive landmark (engine P1Engine state, before the window Problem is built): coupled, in_newest, k, m, and the
    host observation's pixel (NaN if none).  Same definitions as the Problem-level helpers."""
    L = len(alive)
    loc = np.full(eng.n_lm, -1, np.int64)
    loc[alive] = np.arange(L)
    m = np.zeros(L, np.int64)
    kk = np.zeros(L, np.int64)
    in_newest = np.zeros(L, bool)
    host_uv = np.full((L, 2), np.nan)
    newest_sid = eng._newest()["sid"] if eng.slots else None
    for s in eng.slots:
        n = len(s["lid"])
        if n == 0:
            continue
        v = s["valid"]
        li = loc[s["lid"]]
        ok = v & (li >= 0)
        if not ok.any():
            continue
        lo = li[ok]
        m += np.bincount(lo, minlength=L)
        hst = eng.lm_host[s["lid"][ok]] == s["sid"]
        u = np.unique(lo[~hst])
        kk[u] += 1
        hl = ok & (li >= 0) & (eng.lm_host[np.maximum(s["lid"], 0)] == s["sid"]) & (s["cam"] == 0)
        host_uv[li[hl]] = s["uv"][hl]
        if s["sid"] == newest_sid:
            in_newest[np.unique(li[ok & (s["cam"] == 0)])] = True
    coupled = kk > 0
    k = np.where(coupled, kk + 1, 0)
    return coupled, in_newest, k, m, host_uv


def state_dims_engine(eng):
    n = sum(15 if (eng.vi and s["full"]) else 6 for s in eng.slots)
    nimu = len(eng.imu_factors) if eng.vi else 0
    return int(n), int(nimu)


# ------------------------------------------------------------------------------- pose-free landmarks, bit-exact
# The window solve's only effect on a pose-free landmark is its own 3x3 Gauss-Newton step in every ACCEPTED LM iteration
# (its pose blocks are zero, so the step cannot depend on the poses' step), plus its marginal from final_linearisation.
# `posefree_replay` reproduces exactly that, with window.py's own kernels (obs_residual, obs_jacobians_ws, _huber,
# _cam_poses) and build_system / back_substitute / apply_update's operation order, along the LM path the solve took
# (LMPath records lam and the poses at every linearisation).  Result: lm_p and lm_Sl bit-identical to the full solve
# (tests/test_p12_budget.py), so a budgeted run equals the unbudgeted one bit for bit while the cap does not bind.
@njit(cache=True, nogil=True)
def _pf_block_fast(Rwc, pwc, RcR, h, a, b, rho, z0, z1, isg, ris, rmeas, fx, fy, cx, cy, huber_k, lam, Hll, bl, Hi):
    """_pf_block with obs_residual / obs_jacobians_ws (t == h, left camera) written out: the same operations in the same
    order (so the same bits), without the second Y / q evaluation, the per-landmark R^T R (RcR = _mtm(Rwc[h], Rwc[h]),
    precomputed per host by the caller) and the unused target / host pose Jacobians."""
    d0 = pwc[h, 0] - pwc[h, 0]
    d1 = pwc[h, 1] - pwc[h, 1]
    d2 = pwc[h, 2] - pwc[h, 2]
    Y0 = Rwc[h, 0, 0] * a + Rwc[h, 0, 1] * b + Rwc[h, 0, 2] + rho * d0
    Y1 = Rwc[h, 1, 0] * a + Rwc[h, 1, 1] * b + Rwc[h, 1, 2] + rho * d1
    Y2 = Rwc[h, 2, 0] * a + Rwc[h, 2, 1] * b + Rwc[h, 2, 2] + rho * d2
    q0 = Rwc[h, 0, 0] * Y0 + Rwc[h, 1, 0] * Y1 + Rwc[h, 2, 0] * Y2
    q1 = Rwc[h, 0, 1] * Y0 + Rwc[h, 1, 1] * Y1 + Rwc[h, 2, 1] * Y2
    q2 = Rwc[h, 0, 2] * Y0 + Rwc[h, 1, 2] * Y1 + Rwc[h, 2, 2] * Y2
    for i in range(3):
        bl[i] = 0.0
        for j in range(3):
            Hll[i, j] = 0.0
    nobs = 0
    if q2 > 1e-9:
        e0 = fx * q0 / q2 + cx - z0
        e1 = fy * q1 / q2 + cy - z1
        en2 = 0.0
        e0 *= isg
        en2 += e0 * e0
        e1 *= isg
        en2 += e1 * e1
        en = math.sqrt(en2)
        if en <= huber_k:
            w = 1.0
        else:
            w = huber_k / en
        iq = 1.0 / q2
        D00 = fx * iq
        D01 = 0.0
        D02 = -fx * q0 * iq * iq
        D10 = 0.0
        D11 = fy * iq
        D12 = -fy * q1 * iq * iq
        r0 = Rwc[h, 0, 0] * d0 + Rwc[h, 1, 0] * d1 + Rwc[h, 2, 0] * d2
        r1 = Rwc[h, 0, 1] * d0 + Rwc[h, 1, 1] * d1 + Rwc[h, 2, 1] * d2
        r2 = Rwc[h, 0, 2] * d0 + Rwc[h, 1, 2] * d1 + Rwc[h, 2, 2] * d2
        J00 = D00 * RcR[0, 0] + D01 * RcR[1, 0] + D02 * RcR[2, 0]
        J01 = D00 * RcR[0, 1] + D01 * RcR[1, 1] + D02 * RcR[2, 1]
        J02 = D00 * r0 + D01 * r1 + D02 * r2
        J10 = D10 * RcR[0, 0] + D11 * RcR[1, 0] + D12 * RcR[2, 0]
        J11 = D10 * RcR[0, 1] + D11 * RcR[1, 1] + D12 * RcR[2, 1]
        J12 = D10 * r0 + D11 * r1 + D12 * r2
        J00 *= isg
        J01 *= isg
        J02 *= isg
        J10 *= isg
        J11 *= isg
        J12 *= isg
        nobs += 1
        acc = 0.0
        acc += J00 * J00
        acc += J10 * J10
        Hll[0, 0] += w * acc
        acc = 0.0
        acc += J00 * J01
        acc += J10 * J11
        Hll[0, 1] += w * acc
        acc = 0.0
        acc += J00 * J02
        acc += J10 * J12
        Hll[0, 2] += w * acc
        acc = 0.0
        acc += J00 * e0
        acc += J10 * e1
        bl[0] += w * acc
        acc = 0.0
        acc += J01 * J00
        acc += J11 * J10
        Hll[1, 0] += w * acc
        acc = 0.0
        acc += J01 * J01
        acc += J11 * J11
        Hll[1, 1] += w * acc
        acc = 0.0
        acc += J01 * J02
        acc += J11 * J12
        Hll[1, 2] += w * acc
        acc = 0.0
        acc += J01 * e0
        acc += J11 * e1
        bl[1] += w * acc
        acc = 0.0
        acc += J02 * J00
        acc += J12 * J10
        Hll[2, 0] += w * acc
        acc = 0.0
        acc += J02 * J01
        acc += J12 * J11
        Hll[2, 1] += w * acc
        acc = 0.0
        acc += J02 * J02
        acc += J12 * J12
        Hll[2, 2] += w * acc
        acc = 0.0
        acc += J02 * e0
        acc += J12 * e1
        bl[2] += w * acc
    if ris > 0.0:
        rr = (rho - rmeas) * ris
        ar = abs(rr)
        if ar <= huber_k:
            wp = 1.0
        else:
            wp = huber_k / ar
        Hll[2, 2] += wp * ris * ris
        bl[2] += wp * ris * rr
    for i in range(3):
        Hll[i, i] += lam * Hll[i, i] + 1e-12
    det = (Hll[0, 0] * (Hll[1, 1] * Hll[2, 2] - Hll[1, 2] * Hll[2, 1])
           - Hll[0, 1] * (Hll[1, 0] * Hll[2, 2] - Hll[1, 2] * Hll[2, 0])
           + Hll[0, 2] * (Hll[1, 0] * Hll[2, 1] - Hll[1, 1] * Hll[2, 0]))
    tr = Hll[0, 0] + Hll[1, 1] + Hll[2, 2]
    if nobs == 0 or not (det > 1e-18 * tr * tr * tr) or not math.isfinite(det):
        return False
    idet = 1.0 / det
    Hi[0, 0] = (Hll[1, 1] * Hll[2, 2] - Hll[1, 2] * Hll[2, 1]) * idet
    Hi[0, 1] = (Hll[0, 2] * Hll[2, 1] - Hll[0, 1] * Hll[2, 2]) * idet
    Hi[0, 2] = (Hll[0, 1] * Hll[1, 2] - Hll[0, 2] * Hll[1, 1]) * idet
    Hi[1, 0] = (Hll[1, 2] * Hll[2, 0] - Hll[1, 0] * Hll[2, 2]) * idet
    Hi[1, 1] = (Hll[0, 0] * Hll[2, 2] - Hll[0, 2] * Hll[2, 0]) * idet
    Hi[1, 2] = (Hll[0, 2] * Hll[1, 0] - Hll[0, 0] * Hll[1, 2]) * idet
    Hi[2, 0] = (Hll[1, 0] * Hll[2, 1] - Hll[1, 1] * Hll[2, 0]) * idet
    Hi[2, 1] = (Hll[0, 1] * Hll[2, 0] - Hll[0, 0] * Hll[2, 1]) * idet
    Hi[2, 2] = (Hll[0, 0] * Hll[1, 1] - Hll[0, 1] * Hll[1, 0]) * idet
    return True


@njit(cache=True, nogil=True)
def posefree_replay(Rwb_it, pwb_it, lam_it, acc_it, Rwb_f, pwb_f, Rbc, pbc, lmp, host, rmeas, ris, z, isg, fx, fy, cx,
                    cy, B, huber_k, Hinv_out, ok_out):
    """Replay the solve on pose-free landmarks: lmp (n,3) updated in place exactly as window.solve updates them;
    Hinv_out / ok_out = final_linearisation's Hll_inv / lm_ok for them.  Rwb_it / pwb_it / lam_it: the poses and damping
    at each of the solve's linearisations; acc_it: whether that iteration's step was applied; Rwb_f / pwb_f: the poses
    of the final linearisation.  (B unused: the left camera only; kept for the call signature.)"""
    P = Rwb_f.shape[0]
    n = lmp.shape[0]
    Rwc = np.zeros((P, 3, 3))
    pwc = np.zeros((P, 3))
    RcR = np.zeros((P, 3, 3))
    Hll = np.zeros((3, 3))
    bl = np.zeros(3)
    Hi = np.zeros((3, 3))
    v = np.zeros(3)
    for it in range(lam_it.shape[0] + 1):
        final = it == lam_it.shape[0]
        if not final and not acc_it[it]:
            continue                                     # a rejected (or failed) iteration leaves the landmark as is
        if final:
            _cam_poses(Rwb_f, pwb_f, Rbc, pbc, Rwc, pwc)
            lam = 0.0
        else:
            _cam_poses(Rwb_it[it], pwb_it[it], Rbc, pbc, Rwc, pwc)
            lam = lam_it[it]
        for s in range(P):
            _mtm(Rwc[s], Rwc[s], RcR[s])
        for l in range(n):
            h = host[l]
            ok = _pf_block_fast(Rwc, pwc, RcR[h], h, lmp[l, 0], lmp[l, 1], lmp[l, 2], z[l, 0], z[l, 1], isg, ris[l],
                                rmeas[l], fx, fy, cx, cy, huber_k, lam, Hll, bl, Hi)
            if final:
                ok_out[l] = ok
                if ok:
                    for i in range(3):
                        for j in range(3):
                            Hinv_out[l, i, j] = Hi[i, j]
                continue
            if not ok:
                continue
            for i in range(3):
                v[i] = bl[i] + 0.0                       # back_substitute adds the (zero) pose coupling
            acc = 0.0
            acc += Hi[0, 0] * v[0]
            acc += Hi[0, 1] * v[1]
            acc += Hi[0, 2] * v[2]
            d0 = -acc
            acc = 0.0
            acc += Hi[1, 0] * v[0]
            acc += Hi[1, 1] * v[1]
            acc += Hi[1, 2] * v[2]
            d1 = -acc
            acc = 0.0
            acc += Hi[2, 0] * v[0]
            acc += Hi[2, 1] * v[1]
            acc += Hi[2, 2] * v[2]
            d2 = -acc
            r = lmp[l, 2] + d2
            if r > 1e-9 and math.isfinite(r):
                lmp[l, 0] = lmp[l, 0] + d0
                lmp[l, 1] = lmp[l, 1] + d1
                lmp[l, 2] = r
    return 0


class LMPath:
    """Context manager: records window.build_system's (lam, Rwb, pwb) at every call while active (window.solve's
    linearisations have lam > 0, final_linearisation's lam == 0).  Implemented by swapping the module global that
    window.solve / final_linearisation call; restored on exit."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        from xclean.vio import window as Wn
        self._Wn = Wn
        self._orig = Wn.build_system
        orig, calls = self._orig, self.calls

        def rec(*a):
            calls.append((float(a[21]), a[0].copy(), a[1].copy()))
            return orig(*a)
        Wn.build_system = rec
        return self

    def __exit__(self, *exc):
        self._Wn.build_system = self._orig
        return False

    def path(self, lam_final):
        """-> (Rwb_it, pwb_it, lam_it, acc_it, Rwb_f, pwb_f).  Iteration i's step was applied iff the next damping is
        smaller (window.solve: accept -> lam / 10, reject or Cholesky failure -> lam x 10; the 1e-12 floor cannot be
        reached in <= 8 iterations from 1e-4)."""
        sol = [c for c in self.calls if c[0] > 0.0]
        fin = [c for c in self.calls if c[0] == 0.0]
        lam = np.array([c[0] for c in sol], float)
        nxt = np.append(lam[1:], float(lam_final))
        acc = nxt < lam
        P = fin[-1][1].shape[0]
        Rit = np.ascontiguousarray(np.stack([c[1] for c in sol])) if sol else np.zeros((0, P, 3, 3))
        pit = np.ascontiguousarray(np.stack([c[2] for c in sol])) if sol else np.zeros((0, P, 3))
        return Rit, pit, lam, acc, np.ascontiguousarray(fin[-1][1]), np.ascontiguousarray(fin[-1][2])


# --------------------------------------------------------- pose-free landmarks INSIDE the LM loop (exact decisions)
# Why (2026-10-26): window.solve accepts an LM step iff c_new < c_tot, and both costs include EVERY landmark.  A
# pose-free landmark's own Gauss-Newton step never raises its cost (its residuals are linear in (a, b, rho) and the
# step minimises their damped Huber-weighted model), so in the FULL problem the pose-free terms can only make a step
# MORE acceptable.  The post-solve replay (posefree_replay) follows the accept pattern of the solve WITHOUT them, so it
# differs from the full solve whenever a step the pose-coupled system rejects is carried by the pose-free landmarks'
# cost decrease (a "stale" pose-free landmark, off its host ray, in an iteration the coupled system rejects).  The
# classes below put the pose-free terms back into both costs of every iteration and move their state with the solve's
# own accept decisions: the decision rule is then the full problem's.  In floating point the only remaining difference
# is the summation order of the total cost (the pose-free terms are summed as one block instead of interleaved), which
# matters only when a step's cost change is within rounding of zero.
@njit(cache=True, nogil=True)
def posefree_cost(Rwb, pwb, Rbc, pbc, lmp, host, rmeas, ris, z, isg, fx, fy, cx, cy, B, huber_k, behind):
    """Robust cost of pose-free landmarks (the host left observation + the stereo inverse-depth prior), each term
    evaluated exactly as window.build_system (behind=False: a point with q2 <= 1e-9 adds nothing) or window.visual_cost
    (behind=True: it adds 1e6) evaluates it; summed in landmark order."""
    P = Rwb.shape[0]
    Rwc = np.zeros((P, 3, 3))
    pwc = np.zeros((P, 3))
    _cam_poses(Rwb, pwb, Rbc, pbc, Rwc, pwc)
    e = np.zeros(2)
    cost = 0.0
    for l in range(lmp.shape[0]):
        h = host[l]
        _nd, q2 = obs_residual(Rwc, pwc, lmp[l], h, h, 0, z[l], fx, fy, cx, cy, B, e)
        if q2 <= 1e-9:
            if behind:
                cost += 1e6
        else:
            en2 = 0.0
            e0 = e[0] * isg
            en2 += e0 * e0
            e1 = e[1] * isg
            en2 += e1 * e1
            cost += _huber(math.sqrt(en2), huber_k)[1]
        if ris[l] > 0.0:
            r0 = (lmp[l, 2] - rmeas[l]) * ris[l]
            cost += _huber(abs(r0), huber_k)[1]
    return cost


@njit(cache=True, nogil=True)
def posefree_step(Rwb, pwb, Rbc, pbc, lmp_in, host, rmeas, ris, z, isg, fx, fy, cx, cy, huber_k, lam, lmp_out,
                  Hinv_out, ok_out):
    """One linearisation of the pose-free landmarks at damping lam (window.build_system's block + back_substitute's
    step + apply_update's rule, the same operations in the same order as posefree_replay): lmp_out = the trial state;
    Hinv_out / ok_out = the (damped) 3x3 inverse and lm_ok (with lam = 0: final_linearisation's marginals)."""
    P = Rwb.shape[0]
    n = lmp_in.shape[0]
    Rwc = np.zeros((P, 3, 3))
    pwc = np.zeros((P, 3))
    RcR = np.zeros((P, 3, 3))
    Hll = np.zeros((3, 3))
    bl = np.zeros(3)
    Hi = np.zeros((3, 3))
    v = np.zeros(3)
    _cam_poses(Rwb, pwb, Rbc, pbc, Rwc, pwc)
    for s in range(P):
        _mtm(Rwc[s], Rwc[s], RcR[s])
    for l in range(n):
        h = host[l]
        for i in range(3):
            lmp_out[l, i] = lmp_in[l, i]
        ok = _pf_block_fast(Rwc, pwc, RcR[h], h, lmp_in[l, 0], lmp_in[l, 1], lmp_in[l, 2], z[l, 0], z[l, 1], isg, ris[l],
                            rmeas[l], fx, fy, cx, cy, huber_k, lam, Hll, bl, Hi)
        ok_out[l] = ok
        if not ok:
            continue
        for i in range(3):
            for j in range(3):
                Hinv_out[l, i, j] = Hi[i, j]
        for i in range(3):
            v[i] = bl[i] + 0.0                           # back_substitute adds the (zero) pose coupling
        acc = 0.0
        acc += Hi[0, 0] * v[0]
        acc += Hi[0, 1] * v[1]
        acc += Hi[0, 2] * v[2]
        d0 = -acc
        acc = 0.0
        acc += Hi[1, 0] * v[0]
        acc += Hi[1, 1] * v[1]
        acc += Hi[1, 2] * v[2]
        d1 = -acc
        acc = 0.0
        acc += Hi[2, 0] * v[0]
        acc += Hi[2, 1] * v[1]
        acc += Hi[2, 2] * v[2]
        d2 = -acc
        r = lmp_in[l, 2] + d2
        if r > 1e-9 and math.isfinite(r):
            lmp_out[l, 0] = lmp_in[l, 0] + d0
            lmp_out[l, 1] = lmp_in[l, 1] + d1
            lmp_out[l, 2] = r
    return 0


class PosefreeLM:
    """Context manager around ONE window.solve + its final_linearisation on a Problem WITHOUT its pose-free landmarks
    (arrays from problem_posefree_arrays / EngineBudget): while active, window.build_system's cost (c_tot) and
    window.visual_cost's cost (c_new) each get the pose-free landmarks' cost added, their trial state follows every
    linearisation, and a trial is committed exactly when window.solve accepts the step (the next damping is smaller;
    window.solve's lam / 10 on accept, x 10 on reject or Cholesky failure; the 1e-12 floor is out of reach in <= 8
    iterations from 1e-4).  window.solve is wrapped too, so the last iteration's decision is read from its returned lam
    before final_linearisation (lam == 0) computes the pose-free marginals.
        with PosefreeLM(pb_full.Rbc, ..., arrays) as pl:
            res = Wn.solve(sub, iters); out = Wn.final_linearisation(sub)
        pl.lmp, pl.Hinv, pl.ok     # = the full solve's landmark states, Hll_inv and lm_ok for these landmarks
    Not re-entrant; marg_system must not run inside it."""

    def __init__(self, Rbc, pbc, lmp, host, rmeas, ris, z, isg, fx, fy, cx, cy, B, huber_k):
        self.args_c = (np.ascontiguousarray(Rbc, float), np.ascontiguousarray(pbc, float))
        n = len(host)
        self.n = n
        self.lmp = np.ascontiguousarray(lmp, float).copy()
        self.trial = np.zeros((n, 3))
        self.host = np.ascontiguousarray(host, np.int64)
        self.rmeas = np.ascontiguousarray(rmeas, float)
        self.ris = np.ascontiguousarray(ris, float)
        self.z = np.ascontiguousarray(z, float)
        self.isg = float(isg)
        self.cam = (float(fx), float(fy), float(cx), float(cy), float(B), float(huber_k))
        self.Hinv = np.zeros((n, 3, 3))
        self.ok = np.zeros(n, np.bool_)
        self._Hs = np.zeros((n, 3, 3))
        self._oks = np.zeros(n, np.bool_)
        self.pending = False
        self.lam_it = None
        self.solved = False
        self.final = False
        self.n_commit = 0

    def _cost(self, Rwb, pwb, lmp, behind):
        fx, fy, cx, cy, B, hk = self.cam
        return posefree_cost(Rwb, pwb, self.args_c[0], self.args_c[1], lmp, self.host, self.rmeas, self.ris, self.z,
                             self.isg, fx, fy, cx, cy, B, hk, behind)

    def _step(self, Rwb, pwb, lam, out, Hinv, ok):
        fx, fy, cx, cy, _B, hk = self.cam
        posefree_step(Rwb, pwb, self.args_c[0], self.args_c[1], self.lmp, self.host, self.rmeas, self.ris, self.z,
                      self.isg, fx, fy, cx, cy, hk, lam, out, Hinv, ok)

    def _commit_if(self, lam_next):
        if self.pending and lam_next < self.lam_it:
            self.lmp[:] = self.trial
            self.n_commit += 1
        self.pending = False

    def __enter__(self):
        from xclean.vio import window as Wn
        self._Wn = Wn
        self._o = (Wn.build_system, Wn.visual_cost, Wn.solve)
        o_bs, o_vc, o_sv = self._o

        def bs(*a):
            c = o_bs(*a)
            if self.n == 0:
                return c
            lam = float(a[21])
            if lam > 0.0:
                if self.solved:
                    raise RuntimeError("PosefreeLM: a second solve inside one context")
                self._commit_if(lam)
                self.lam_it = lam
                cf = self._cost(a[0], a[1], self.lmp, False)
                self._step(a[0], a[1], lam, self.trial, self._Hs, self._oks)
                return c + cf
            if not self.solved:
                raise RuntimeError("PosefreeLM: an undamped linearisation before the solve finished")
            self._step(a[0], a[1], 0.0, self.trial, self.Hinv, self.ok)    # final_linearisation: marginals only
            self.final = True
            return c + self._cost(a[0], a[1], self.lmp, False)

        def vc(*a):
            c = o_vc(*a)
            if self.n == 0:
                return c
            self.pending = True
            return c + self._cost(a[0], a[1], self.trial, True)

        def sv(pb, iters=5, lam0=1e-4):
            res = o_sv(pb, iters=iters, lam0=lam0)
            self._commit_if(float(res["lam"]))
            self.solved = True
            return res
        Wn.build_system, Wn.visual_cost, Wn.solve = bs, vc, sv
        return self

    def __exit__(self, *exc):
        self._Wn.build_system, self._Wn.visual_cost, self._Wn.solve = self._o
        return False


def problem_posefree_arrays(pb, keep):
    """For a Problem: the arrays posefree_replay needs for the landmarks NOT in keep that are pose-free with exactly one
    valid observation (the left host one).  -> (idx, host, rmeas, ris, z, isg)."""
    loc = obs_landmark(pb)
    L = pb.lmp.shape[0]
    nval = np.bincount(loc[pb.ob_valid], minlength=L)
    hm = pb.ob_valid & (pb.ob_pose == pb.lm_host[loc]) & (pb.ob_cam == 0)
    z = np.full((L, 2), np.nan)
    z[loc[hm]] = pb.ob_z[hm]
    isg = pb.ob_isig[hm]
    idx = np.nonzero(~np.asarray(keep, bool) & (nval == 1) & np.isfinite(z[:, 0]))[0]
    isg0 = float(isg[0]) if len(isg) else 1.0
    if len(isg) and not np.all(isg == isg0):
        raise ValueError("left observations carry different sigmas")
    return idx, np.ascontiguousarray(pb.lm_host[idx]), np.ascontiguousarray(pb.rho_meas[idx]), \
        np.ascontiguousarray(pb.rho_isig[idx]), np.ascontiguousarray(z[idx]), isg0


class EngineBudget:
    """The budget as the engine applies it.  Per window solve:
      exact="inloop" (default, 2026-10-26):
        sel = eb.selector(eng)        # passed to eng._problem(lm_select=sel); installs PosefreeLM for this solve
        try: <window.solve + final_linearisation>  finally: eb.close(eng)    # uninstall + write the pose-free back
      exact="replay" (the 10-24 form, kept for comparison):
        sel = eb.selector(eng); with BG.LMPath() as rec: <solve + final>; eb.after_solve(eng, rec, res["lam"])
    mode: "posefree" (item 1 only) or "cap" (items 1-3).  Only the default stereo form ("prior") excludes pose-free
    landmarks (with u_R observations a pose-free landmark has two host observations; it then stays in the solve)."""

    def __init__(self, iters=5, mode="cap", exact="inloop"):
        if mode not in ("posefree", "cap"):
            raise ValueError(mode)
        if exact not in ("inloop", "replay"):
            raise ValueError(exact)
        self.iters = iters
        self.mode = mode
        self.exact = exact
        self._pending = None
        self._pl = None
        self.last = None

    def selector(self, eng):
        def sel(alive):
            coupled, in_newest, k, m, host_uv = engine_arrays(eng, alive)
            n, nimu = state_dims_engine(eng)
            excl = ~coupled & (m == 1) & np.isfinite(host_uv[:, 0]) & (eng.stereo_mode == "prior")
            if self.mode == "posefree":
                keep = ~excl
                info = {"L": int(len(alive)), "L_posefree": int(excl.sum()), "L_capped": 0, "bound": False,
                        "pred_ms": None, "pred_ms_uncapped": None}
            else:
                keep, info = select(~excl, in_newest, k, m, n, nimu, self.iters)
            info["newest_obs_kept"] = int(np.sum(keep & in_newest))
            lids = alive[excl]
            sid2i = {s["sid"]: i for i, s in enumerate(eng.slots)}
            host = np.array([sid2i[int(h)] for h in eng.lm_host[lids]], np.int64)
            rm = eng.lm_rho_meas[lids]
            sg = np.maximum(eng.kappa_d * eng.lm_sig_lk[lids], eng.lm_span[lids]) / eng.fB
            with np.errstate(divide="ignore", invalid="ignore"):
                ris = np.where(np.isfinite(sg) & (sg > 0) & (rm > 0), 1.0 / np.where(sg > 0, sg, 1.0), 0.0)
            self._pending = (lids, host, np.ascontiguousarray(rm), np.ascontiguousarray(ris),
                             np.ascontiguousarray(host_uv[excl]), 1.0 / eng.sig_px)
            self.last = info
            if self.exact == "inloop":
                if self._pl is not None:
                    raise RuntimeError("EngineBudget: previous solve not closed")
                self._pl = PosefreeLM(eng.Rbc, eng.pbc, eng.lm_p[lids], host, rm, ris, host_uv[excl], 1.0 / eng.sig_px,
                                      eng.fx, eng.fy, eng.cx, eng.cy, eng.B, HUBER_K)
                self._pl.__enter__()
            return keep
        return sel

    def close(self, eng):
        """exact="inloop": after the engine's _solve (call it in a finally): uninstall PosefreeLM and write the pose-free
        landmarks' state and marginals back (only when the solve and its final linearisation completed)."""
        pl, self._pl = self._pl, None
        if pl is None:
            return 0
        pl.__exit__(None, None, None)
        lids = self._pending[0]
        self._pending = None
        if pl.n and pl.final:
            eng.lm_p[lids] = pl.lmp
            eng.lm_Sl[lids[pl.ok]] = pl.Hinv[pl.ok]
        return int(pl.n)

    def after_solve(self, eng, rec, lam_final):
        if self._pending is None:
            return 0
        lids, host, rm, ris, z, isg = self._pending
        self._pending = None
        if len(lids) == 0:
            return 0
        Rit, pit, lam, acc, Rf, pf = rec.path(lam_final)
        lmp = np.ascontiguousarray(eng.lm_p[lids])
        Hinv = np.zeros((len(lids), 3, 3))
        ok = np.zeros(len(lids), np.bool_)
        posefree_replay(Rit, pit, lam, acc, Rf, pf, eng.Rbc, eng.pbc, lmp, host, rm, ris, z, isg, eng.fx, eng.fy,
                        eng.cx, eng.cy, eng.B, HUBER_K, Hinv, ok)
        eng.lm_p[lids] = lmp
        eng.lm_Sl[lids[ok]] = Hinv[ok]
        return int(len(lids))


# ---------------------------------------------------------------------------------------- D4 row band (P1.1)
def band_from_dy(v, half, k_band, center=None, iters=100):
    """D4's row band from the pooled per-point vertical offsets v [px, raw right-image frame].  The samples are a
    MIXTURE: points whose patch has vertical structure localise their row (a peak N(mu, s^2)); points without it give a
    flat column scan whose maximum lands anywhere in the scan range, i.e. uniform on [c - half - 1/2, c + half + 1/2]
    around the shift c the scan was run at (class A: stereo1d's column scan j in [-HALF, HALF] + its sub-pixel
    clamp).  Only the first kind needs the band (the second matches on any row).  c = the pooled median (the engine's
    dy_hat, the shift it applies; an approximation when the shift moved inside the pool).  EM for (eps, mu, s) with the
    uniform's support fixed; mu, s start at the median / 1.4826 MAD, eps at 1/2.
    -> (band, mu, s, eps), band = min(round(|mu - c| + k_band s), half)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    c = float(np.median(v)) if center is None and len(v) else (0.0 if center is None else float(center))
    if len(v) < 3:
        return 0, c, None, None
    v = v[np.abs(v - c) <= half + 0.5]
    mu = c
    s = max(float(1.482602218505602 * np.median(np.abs(v - mu))), 1e-3)
    eps = 0.5
    width = 2.0 * half + 1.0
    for _ in range(iters):
        pg = (1.0 - eps) * np.exp(-0.5 * ((v - mu) / s) ** 2) / (math.sqrt(2.0 * math.pi) * s)
        pu = eps / width
        w = pg / np.maximum(pg + pu, 1e-300)
        sw = float(w.sum())
        if sw <= 1.0:
            break
        mu_n = float((w * v).sum() / sw)
        s_n = max(math.sqrt(float((w * (v - mu_n) ** 2).sum() / sw)), 1e-3)
        eps_n = 1.0 - sw / len(v)
        done = abs(mu_n - mu) < 1e-6 and abs(s_n - s) < 1e-6
        mu, s, eps = mu_n, s_n, eps_n
        if done:
            break
    band = int(min(round(abs(mu - c) + k_band * s), half))
    return band, mu, s, eps


# The CPU module's 1-D stereo cost law (front/stereo1d; class B, measured 2026-10-24, one pinned logical CPU (30) of a
# shared machine, v2_03 752x480, full disparity range 128, 30 frames x 1835 keypoints; tools/w3_stereo_probe.py ->
# data/p12_w3_k11era/p12_w3_stereo_v2_03.json): per point 19.6 us + 16.6 us per band row-pair.  Used only when no GPU is
# visible (the lock-step test series); the live config runs the GPU front (STEREO_GPU_MS below).
STEREO_US = {"c0": 19.6, "c_band": 16.6, "cpu": 30, "width": 752, "ndisp": 128}


def stereo_cost_ms(n_points, band):
    return 1e-3 * n_points * (STEREO_US["c0"] + STEREO_US["c_band"] * band)


# The INTEGRATED GPU front's 1-D stereo (front/gpu_front.disparities, the live config's stereo whenever a GPU is
# visible; bit-identical to stereo1d at every band, 6/6 frames x 6 bands): class B, measured 2026-10-26 ~15:50 on the
# RTX 5090 (host thread pinned to logical CPU 30, machine shared with three CPU lock-step runs), v2_03 752 x 480,
# 30 frames x ~1835 keypoints (the whole-frame call: an upper bound on a KF call), each call end to end (upload,
# kernels, download), min of 3 repetitions; non-negative LS of t = g0 + g1 n + g2 n b; q95 = the 95th percentile of
# measured / law (tools/w3_stereo_gpu.py -> data/p12_w3_stereo_gpu_v2_03.json).  At the caps (2048 points, band HALF = 5)
# the law x q95 gives 5.6 ms per call, measured worst 5.9 ms: D4's band costs <= ~6 ms per KF stereo call against the
# 50 ms frame period of the band-cap row (CPU module: 357-397 ms p50 at band 5).
STEREO_GPU_MS = {"g0": 0.5213, "g1": 0.000292, "g2": 0.000314, "q95": 1.281, "gpu": "RTX 5090", "width": 752}


def stereo_cost_gpu_ms(n_points, band):
    g = STEREO_GPU_MS
    return (g["g0"] + g["g1"] * n_points + g["g2"] * n_points * band) * g["q95"]
