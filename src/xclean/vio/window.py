#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1 FIXED-LAG WINDOW (VO core): keyframe + frame poses, keyframe-hosted inverse-depth landmarks, Schur, LM (numba).

Conventions: xclean/vio/CONVENTIONS.md (binding).
- Pose block [dphi, dp]: R_wb <- R_wb Exp(dphi) (right), p_wb <- p_wb + dp (world). The camera is T_wc = T_wb T_bc.
- Landmark (a, b, rho) hosted in keyframe h: m = (a, b, 1) is the rectified ray, rho the inverse depth.
- Observation in camera t: q = R_wc,t^T (R_wc,h m + rho (p_wc,h - p_wc,t)).
  - left (u, v) = (fx q0/q2 + cx, fy q1/q2 + cy);
  - right u_R = fx (q0 - rho B)/q2 + cx.
- Host inverse-depth prior (judge A1, the default stereo form): (rho - rho_meas) / sigma_rho.
- Whitened residuals, Huber k = 1.345 (class D) by IRLS weights; normal equations H dx = -b.

numba kernels (single-threaded, deterministic, float64):
- `build_system`: per-landmark 3x3 blocks, Schur into the dense 6P x 6P pose system (P <= MAXP);
- `back_substitute`: the landmark updates;
- `apply_update`: on-manifold pose and landmark updates;
- `visual_cost`: the robust cost of the second LM pass.
The marginal prior (square root, FEJ) lives in vio/marg.py; the solver below adds it to the reduced system.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

MAXP = 16                     # max pose slots in one system (7 KF + 3 frames + margin); a data-structure bound only


# ---------------------------------------------------------------------------------------------------- small math
@njit(cache=True, nogil=True, inline="always")
def _skew_mul(a, M, out):
    """out = [a]x @ M for M (3,k)."""
    k = M.shape[1]
    for j in range(k):
        out[0, j] = -a[2] * M[1, j] + a[1] * M[2, j]
        out[1, j] = a[2] * M[0, j] - a[0] * M[2, j]
        out[2, j] = -a[1] * M[0, j] + a[0] * M[1, j]


@njit(cache=True, nogil=True)
def so3_exp_nb(w):
    th2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
    th = math.sqrt(th2)
    K = np.zeros((3, 3))
    K[0, 1] = -w[2]
    K[0, 2] = w[1]
    K[1, 0] = w[2]
    K[1, 2] = -w[0]
    K[2, 0] = -w[1]
    K[2, 1] = w[0]
    if th < 1e-8:
        a = 1.0 - th2 / 6.0
        b = 0.5 - th2 / 24.0
    else:
        a = math.sin(th) / th
        b = (1.0 - math.cos(th)) / th2
    R = np.eye(3) + a * K + b * (K @ K)
    return R


@njit(cache=True, nogil=True)
def so3_log_nb(R):
    c = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) * 0.5
    if c > 1.0:
        c = 1.0
    if c < -1.0:
        c = -1.0
    th = math.acos(c)
    w = np.zeros(3)
    v0 = R[2, 1] - R[1, 2]
    v1 = R[0, 2] - R[2, 0]
    v2 = R[1, 0] - R[0, 1]
    if th < 1e-6:
        w[0] = 0.5 * v0
        w[1] = 0.5 * v1
        w[2] = 0.5 * v2
        return w
    if th > math.pi - 1e-4:
        # near pi: from the symmetric part (memory P1: so3_log wrong near pi)
        B = (R + R.T) * 0.5
        i = 0
        if B[1, 1] > B[i, i]:
            i = 1
        if B[2, 2] > B[i, i]:
            i = 2
        den = math.sqrt(max((B[i, i] - c) / (1.0 - c), 0.0))
        ax = np.zeros(3)
        for j in range(3):
            ax[j] = (B[j, i] - (c if j == i else 0.0)) / max(den * (1.0 - c), 1e-12)
        ax[i] = den
        n = math.sqrt(ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2)
        s = 1.0
        if ax[0] * v0 + ax[1] * v1 + ax[2] * v2 < 0:
            s = -1.0
        for j in range(3):
            w[j] = s * th * ax[j] / n
        return w
    f = th / (2.0 * math.sin(th))
    w[0] = f * v0
    w[1] = f * v1
    w[2] = f * v2
    return w


# ---------------------------------------------------------------------------------------------- the visual factor
@njit(cache=True, nogil=True)
def _cam_poses(Rwb, pwb, Rbc, pbc, Rwc, pwc):
    P = Rwb.shape[0]
    for s in range(P):
        for i in range(3):
            for j in range(3):
                acc = 0.0
                for k in range(3):
                    acc += Rwb[s, i, k] * Rbc[k, j]
                Rwc[s, i, j] = acc
            acc = pwb[s, i]
            for k in range(3):
                acc += Rwb[s, i, k] * pbc[k]
            pwc[s, i] = acc


@njit(cache=True, nogil=True)
def obs_residual(Rwc, pwc, lmp, h, t, cam, z, fx, fy, cx, cy, B, e):
    """Whitening-free residual of one observation into e (2 or 1 used).  Returns (ndim, q2)."""
    a = lmp[0]
    b = lmp[1]
    rho = lmp[2]
    Y0 = Rwc[h, 0, 0] * a + Rwc[h, 0, 1] * b + Rwc[h, 0, 2] + rho * (pwc[h, 0] - pwc[t, 0])
    Y1 = Rwc[h, 1, 0] * a + Rwc[h, 1, 1] * b + Rwc[h, 1, 2] + rho * (pwc[h, 1] - pwc[t, 1])
    Y2 = Rwc[h, 2, 0] * a + Rwc[h, 2, 1] * b + Rwc[h, 2, 2] + rho * (pwc[h, 2] - pwc[t, 2])
    q0 = Rwc[t, 0, 0] * Y0 + Rwc[t, 1, 0] * Y1 + Rwc[t, 2, 0] * Y2
    q1 = Rwc[t, 0, 1] * Y0 + Rwc[t, 1, 1] * Y1 + Rwc[t, 2, 1] * Y2
    q2 = Rwc[t, 0, 2] * Y0 + Rwc[t, 1, 2] * Y1 + Rwc[t, 2, 2] * Y2
    if cam == 0:
        if q2 <= 1e-9:
            return 2, q2
        e[0] = fx * q0 / q2 + cx - z[0]
        e[1] = fy * q1 / q2 + cy - z[1]
        return 2, q2
    if q2 <= 1e-9:
        return 1, q2
    e[0] = fx * (q0 - rho * B) / q2 + cx - z[0]
    return 1, q2


@njit(cache=True, nogil=True, inline="always")
def _mm(A, B, out):
    for i in range(3):
        for j in range(3):
            out[i, j] = A[i, 0] * B[0, j] + A[i, 1] * B[1, j] + A[i, 2] * B[2, j]


@njit(cache=True, nogil=True, inline="always")
def _mtm(A, B, out):
    """out = A^T B."""
    for i in range(3):
        for j in range(3):
            out[i, j] = A[0, i] * B[0, j] + A[1, i] * B[1, j] + A[2, i] * B[2, j]


@njit(cache=True, nogil=True)
def obs_jacobians_ws(Rwb, Rwc, pwc, Rcb, Px, lmp, h, t, cam, fx, fy, B, Jl, Jt, Jh, ws, Dp):
    """Allocation-free form of obs_jacobians (ws: (10,3,3) scratch, Dp: (2,3) scratch; Rcb = R_bc^T and
    Px = [p_bc]x precomputed by the caller)."""
    a = lmp[0]
    b = lmp[1]
    rho = lmp[2]
    d0 = pwc[h, 0] - pwc[t, 0]
    d1 = pwc[h, 1] - pwc[t, 1]
    d2 = pwc[h, 2] - pwc[t, 2]
    Y0 = Rwc[h, 0, 0] * a + Rwc[h, 0, 1] * b + Rwc[h, 0, 2] + rho * d0
    Y1 = Rwc[h, 1, 0] * a + Rwc[h, 1, 1] * b + Rwc[h, 1, 2] + rho * d1
    Y2 = Rwc[h, 2, 0] * a + Rwc[h, 2, 1] * b + Rwc[h, 2, 2] + rho * d2
    q0 = Rwc[t, 0, 0] * Y0 + Rwc[t, 1, 0] * Y1 + Rwc[t, 2, 0] * Y2
    q1 = Rwc[t, 0, 1] * Y0 + Rwc[t, 1, 1] * Y1 + Rwc[t, 2, 1] * Y2
    q2 = Rwc[t, 0, 2] * Y0 + Rwc[t, 1, 2] * Y1 + Rwc[t, 2, 2] * Y2
    nd = 2 if cam == 0 else 1
    iq = 1.0 / q2
    if cam == 0:
        Dp[0, 0] = fx * iq
        Dp[0, 1] = 0.0
        Dp[0, 2] = -fx * q0 * iq * iq
        Dp[1, 0] = 0.0
        Dp[1, 1] = fy * iq
        Dp[1, 2] = -fy * q1 * iq * iq
    else:
        Dp[0, 0] = fx * iq
        Dp[0, 1] = 0.0
        Dp[0, 2] = -fx * (q0 - rho * B) * iq * iq
    RcR = ws[0]
    _mtm(Rwc[t], Rwc[h], RcR)                 # R_ct R_wc,h
    # landmark: dq/da = RcR[:,0], dq/db = RcR[:,1], dq/drho = R_ct (p_h - p_t) (- B e_x for the right camera)
    r0 = Rwc[t, 0, 0] * d0 + Rwc[t, 1, 0] * d1 + Rwc[t, 2, 0] * d2
    r1 = Rwc[t, 0, 1] * d0 + Rwc[t, 1, 1] * d1 + Rwc[t, 2, 1] * d2
    r2 = Rwc[t, 0, 2] * d0 + Rwc[t, 1, 2] * d1 + Rwc[t, 2, 2] * d2
    if cam == 1:
        r0 -= B
    for r in range(nd):
        Jl[r, 0] = Dp[r, 0] * RcR[0, 0] + Dp[r, 1] * RcR[1, 0] + Dp[r, 2] * RcR[2, 0]
        Jl[r, 1] = Dp[r, 0] * RcR[0, 1] + Dp[r, 1] * RcR[1, 1] + Dp[r, 2] * RcR[2, 1]
        Jl[r, 2] = Dp[r, 0] * r0 + Dp[r, 1] * r1 + Dp[r, 2] * r2
    if t == h:
        for r in range(nd):
            for j in range(6):
                Jt[r, j] = 0.0
                Jh[r, j] = 0.0
        return nd
    # camera frame: target dq/dphi = [q]x, dq/dp = -rho R_ct; host dq/dphi = -RcR [m]x, dq/dp = rho R_ct
    Qx = ws[1]
    Qx[0, 0] = 0.0
    Qx[0, 1] = -q2
    Qx[0, 2] = q1
    Qx[1, 0] = q2
    Qx[1, 1] = 0.0
    Qx[1, 2] = -q0
    Qx[2, 0] = -q1
    Qx[2, 1] = q0
    Qx[2, 2] = 0.0
    Mx = ws[2]
    Mx[0, 0] = 0.0
    Mx[0, 1] = -1.0
    Mx[0, 2] = b
    Mx[1, 0] = 1.0
    Mx[1, 1] = 0.0
    Mx[1, 2] = -a
    Mx[2, 0] = -b
    Mx[2, 1] = a
    Mx[2, 2] = 0.0
    Ah = ws[3]
    _mm(RcR, Mx, Ah)                          # dq/dphi_c,h = -Ah
    Rct = ws[4]
    for i in range(3):
        for j in range(3):
            Rct[i, j] = Rwc[t, j, i]
    # chain to the body: dq/dphi_b = dq/dphi_c R_cb - dq/dp_c R_wb [p_bc]x ; dq/dp_b = dq/dp_c
    Wt = ws[5]
    Wh = ws[6]
    _mm(Rwb[t], Px, Wt)
    _mm(Rwb[h], Px, Wh)
    T1 = ws[7]
    T2 = ws[8]
    T3 = ws[9]
    _mm(Qx, Rcb, T1)                          # [q]x R_cb
    _mm(Rct, Wt, T2)                          # R_ct W_t
    for i in range(3):
        for j in range(3):
            T1[i, j] = T1[i, j] + rho * T2[i, j]  # target dphi: [q]x Rcb - (-rho Rct) Wt
    _mm(Ah, Rcb, T2)                          # Ah R_cb
    _mm(Rct, Wh, T3)                          # R_ct W_h
    for i in range(3):
        for j in range(3):
            T2[i, j] = -T2[i, j] - rho * T3[i, j]  # host dphi: -Ah Rcb - (rho Rct) Wh
    for r in range(nd):
        for j in range(3):
            Jt[r, j] = Dp[r, 0] * T1[0, j] + Dp[r, 1] * T1[1, j] + Dp[r, 2] * T1[2, j]
            Jh[r, j] = Dp[r, 0] * T2[0, j] + Dp[r, 1] * T2[1, j] + Dp[r, 2] * T2[2, j]
            v = Dp[r, 0] * Rct[0, j] + Dp[r, 1] * Rct[1, j] + Dp[r, 2] * Rct[2, j]
            Jt[r, 3 + j] = -rho * v
            Jh[r, 3 + j] = rho * v
    return nd


@njit(cache=True, nogil=True)
def obs_jacobians(Rwb, Rwc, pwc, Rbc, pbc, lmp, h, t, cam, fx, fy, B, Jl, Jt, Jh):
    """Jacobians of the (unwhitened) residual of one observation: Jl (nd,3) wrt (a, b, rho); Jt / Jh (nd,6) wrt the
    target / host BODY pose blocks [dphi, dp].  Jt = Jh = 0 when t == h (the host observation is pose-free).
    (Wrapper of the allocation-free obs_jacobians_ws; tests/test_p1_window.py pins both by finite differences.)"""
    ws = np.zeros((10, 3, 3))
    Dp = np.zeros((2, 3))
    Rcb = Rbc.T.copy()
    Px = np.zeros((3, 3))
    Px[0, 1] = -pbc[2]
    Px[0, 2] = pbc[1]
    Px[1, 0] = pbc[2]
    Px[1, 2] = -pbc[0]
    Px[2, 0] = -pbc[1]
    Px[2, 1] = pbc[0]
    return obs_jacobians_ws(Rwb, Rwc, pwc, Rcb, Px, lmp, h, t, cam, fx, fy, B, Jl, Jt, Jh, ws, Dp)


@njit(cache=True, nogil=True)
def _huber(en, k):
    if en <= k:
        return 1.0, 0.5 * en * en
    return k / en, k * (en - 0.5 * k)


@njit(cache=True, nogil=True)
def build_system(Rwb, pwb, Rbc, pbc, lmp, lm_host, lm_mask, rho_meas, rho_isig, ob_start, ob_pose, ob_cam, ob_z,
                 ob_isig, ob_valid, fx, fy, cx, cy, B, huber_k, lam, H, bvec, Hll_inv, bl_out, Hlp, lm_ok, ob_chi2):
    """Linearise every valid observation of every masked landmark; Schur the landmarks into H (6P x 6P), bvec.
    lam > 0 damps the landmark blocks (Marquardt).  Outputs per landmark: Hll_inv, bl_out, Hlp (P,3,6), lm_ok.
    ob_chi2[k] = the whitened squared residual of observation k (inf when not evaluable).  Returns the robust cost."""
    P = Rwb.shape[0]
    L = lmp.shape[0]
    Rwc = np.zeros((P, 3, 3))
    pwc = np.zeros((P, 3))
    _cam_poses(Rwb, pwb, Rbc, pbc, Rwc, pwc)
    H[:, :] = 0.0
    bvec[:] = 0.0
    Jl = np.zeros((2, 3))
    Jt = np.zeros((2, 6))
    Jh = np.zeros((2, 6))
    e = np.zeros(2)
    Hll = np.zeros((3, 3))
    bl = np.zeros(3)
    touched = np.zeros(P, np.bool_)
    ws = np.zeros((10, 3, 3))
    Dpw = np.zeros((2, 3))
    Rcb = Rbc.T.copy()
    Px = np.zeros((3, 3))
    Px[0, 1] = -pbc[2]
    Px[0, 2] = pbc[1]
    Px[1, 0] = pbc[2]
    Px[1, 2] = -pbc[0]
    Px[2, 0] = -pbc[1]
    Px[2, 1] = pbc[0]
    HiB = np.zeros(3)
    T1 = np.zeros((3, 6))
    cost = 0.0
    for l in range(L):
        lm_ok[l] = False
        if not lm_mask[l]:
            continue
        h = lm_host[l]
        Hll[:, :] = 0.0
        bl[:] = 0.0
        for s in range(P):
            touched[s] = False
            for i in range(3):
                for j in range(6):
                    Hlp[l, s, i, j] = 0.0
        nobs = 0
        for k in range(ob_start[l], ob_start[l + 1]):
            ob_chi2[k] = np.inf
            if not ob_valid[k]:
                continue
            t = ob_pose[k]
            cam = ob_cam[k]
            nd, q2 = obs_residual(Rwc, pwc, lmp[l], h, t, cam, ob_z[k], fx, fy, cx, cy, B, e)
            if q2 <= 1e-9:
                continue
            isg = ob_isig[k]
            en2 = 0.0
            for i in range(nd):
                e[i] *= isg
                en2 += e[i] * e[i]
            ob_chi2[k] = en2
            en = math.sqrt(en2)
            w, c = _huber(en, huber_k)
            cost += c
            obs_jacobians_ws(Rwb, Rwc, pwc, Rcb, Px, lmp[l], h, t, cam, fx, fy, B, Jl, Jt, Jh, ws, Dpw)
            for i in range(nd):
                for j in range(3):
                    Jl[i, j] *= isg
                for j in range(6):
                    Jt[i, j] *= isg
                    Jh[i, j] *= isg
            nobs += 1
            for i in range(3):
                for j in range(3):
                    acc = 0.0
                    for r in range(nd):
                        acc += Jl[r, i] * Jl[r, j]
                    Hll[i, j] += w * acc
                acc = 0.0
                for r in range(nd):
                    acc += Jl[r, i] * e[r]
                bl[i] += w * acc
            if t == h:
                continue
            touched[t] = True
            touched[h] = True
            # Hlp and the pose-pose blocks
            for i in range(3):
                for j in range(6):
                    at = 0.0
                    ah = 0.0
                    for r in range(nd):
                        at += Jl[r, i] * Jt[r, j]
                        ah += Jl[r, i] * Jh[r, j]
                    Hlp[l, t, i, j] += w * at
                    Hlp[l, h, i, j] += w * ah
            for i in range(6):
                bt = 0.0
                bh = 0.0
                for r in range(nd):
                    bt += Jt[r, i] * e[r]
                    bh += Jh[r, i] * e[r]
                bvec[6 * t + i] += w * bt
                bvec[6 * h + i] += w * bh
                for j in range(6):
                    tt = 0.0
                    hh = 0.0
                    th_ = 0.0
                    for r in range(nd):
                        tt += Jt[r, i] * Jt[r, j]
                        hh += Jh[r, i] * Jh[r, j]
                        th_ += Jt[r, i] * Jh[r, j]
                    H[6 * t + i, 6 * t + j] += w * tt
                    H[6 * h + i, 6 * h + j] += w * hh
                    H[6 * t + i, 6 * h + j] += w * th_
                    H[6 * h + j, 6 * t + i] += w * th_
        # host inverse-depth prior (judge A1 default)
        if rho_isig[l] > 0.0:
            r0 = (lmp[l, 2] - rho_meas[l]) * rho_isig[l]
            w, c = _huber(abs(r0), huber_k)
            cost += c
            Hll[2, 2] += w * rho_isig[l] * rho_isig[l]
            bl[2] += w * rho_isig[l] * r0
        # damping + conditioning guard
        for i in range(3):
            Hll[i, i] += lam * Hll[i, i] + 1e-12
        det = (Hll[0, 0] * (Hll[1, 1] * Hll[2, 2] - Hll[1, 2] * Hll[2, 1])
               - Hll[0, 1] * (Hll[1, 0] * Hll[2, 2] - Hll[1, 2] * Hll[2, 0])
               + Hll[0, 2] * (Hll[1, 0] * Hll[2, 1] - Hll[1, 1] * Hll[2, 0]))
        tr = Hll[0, 0] + Hll[1, 1] + Hll[2, 2]
        if nobs == 0 or not (det > 1e-18 * tr * tr * tr) or not math.isfinite(det):
            continue
        # closed-form 3x3 inverse (no allocation)
        idet = 1.0 / det
        Hi = Hll_inv[l]
        Hi[0, 0] = (Hll[1, 1] * Hll[2, 2] - Hll[1, 2] * Hll[2, 1]) * idet
        Hi[0, 1] = (Hll[0, 2] * Hll[2, 1] - Hll[0, 1] * Hll[2, 2]) * idet
        Hi[0, 2] = (Hll[0, 1] * Hll[1, 2] - Hll[0, 2] * Hll[1, 1]) * idet
        Hi[1, 0] = (Hll[1, 2] * Hll[2, 0] - Hll[1, 0] * Hll[2, 2]) * idet
        Hi[1, 1] = (Hll[0, 0] * Hll[2, 2] - Hll[0, 2] * Hll[2, 0]) * idet
        Hi[1, 2] = (Hll[0, 2] * Hll[1, 0] - Hll[0, 0] * Hll[1, 2]) * idet
        Hi[2, 0] = (Hll[1, 0] * Hll[2, 1] - Hll[1, 1] * Hll[2, 0]) * idet
        Hi[2, 1] = (Hll[0, 1] * Hll[2, 0] - Hll[0, 0] * Hll[2, 1]) * idet
        Hi[2, 2] = (Hll[0, 0] * Hll[1, 1] - Hll[0, 1] * Hll[1, 0]) * idet
        lm_ok[l] = True
        for i in range(3):
            bl_out[l, i] = bl[i]
        # Schur: H -= Hpl Hll^-1 Hlp ; b -= Hpl Hll^-1 bl
        for i in range(3):
            HiB[i] = Hi[i, 0] * bl[0] + Hi[i, 1] * bl[1] + Hi[i, 2] * bl[2]
        for s1 in range(P):
            if not touched[s1]:
                continue
            A1 = Hlp[l, s1]                              # (3,6)
            for i in range(3):
                for j in range(6):
                    T1[i, j] = Hi[i, 0] * A1[0, j] + Hi[i, 1] * A1[1, j] + Hi[i, 2] * A1[2, j]
            for i in range(6):
                acc = 0.0
                for r in range(3):
                    acc += A1[r, i] * HiB[r]
                bvec[6 * s1 + i] -= acc
            for s2 in range(P):
                if not touched[s2]:
                    continue
                A2 = Hlp[l, s2]
                for i in range(6):
                    for j in range(6):
                        acc = 0.0
                        for r in range(3):
                            acc += A2[r, i] * T1[r, j]
                        H[6 * s2 + i, 6 * s1 + j] -= acc
    return cost


@njit(cache=True, nogil=True)
def visual_cost(Rwb, pwb, Rbc, pbc, lmp, lm_host, lm_mask, rho_meas, rho_isig, ob_start, ob_pose, ob_cam, ob_z,
                ob_isig, ob_valid, fx, fy, cx, cy, B, huber_k):
    P = Rwb.shape[0]
    Rwc = np.zeros((P, 3, 3))
    pwc = np.zeros((P, 3))
    _cam_poses(Rwb, pwb, Rbc, pbc, Rwc, pwc)
    e = np.zeros(2)
    cost = 0.0
    for l in range(lmp.shape[0]):
        if not lm_mask[l]:
            continue
        h = lm_host[l]
        for k in range(ob_start[l], ob_start[l + 1]):
            if not ob_valid[k]:
                continue
            nd, q2 = obs_residual(Rwc, pwc, lmp[l], h, ob_pose[k], ob_cam[k], ob_z[k], fx, fy, cx, cy, B, e)
            if q2 <= 1e-9:
                cost += 1e6                              # a point behind the camera: reject the step
                continue
            en2 = 0.0
            for i in range(nd):
                en2 += (e[i] * ob_isig[k]) ** 2
            cost += _huber(math.sqrt(en2), huber_k)[1]
        if rho_isig[l] > 0.0:
            r0 = (lmp[l, 2] - rho_meas[l]) * rho_isig[l]
            cost += _huber(abs(r0), huber_k)[1]
    return cost


@njit(cache=True, nogil=True)
def back_substitute(dp, Hll_inv, bl, Hlp, lm_ok, dl):
    """dl[l] = -Hll^-1 (bl + sum_s Hlp[l,s] dp_s) for every ok landmark (0 otherwise)."""
    L = Hll_inv.shape[0]
    P = Hlp.shape[1]
    v = np.zeros(3)
    for l in range(L):
        for i in range(3):
            dl[l, i] = 0.0
        if not lm_ok[l]:
            continue
        for i in range(3):
            v[i] = bl[l, i]
        for s in range(P):
            for i in range(3):
                acc = 0.0
                for j in range(6):
                    acc += Hlp[l, s, i, j] * dp[6 * s + j]
                v[i] += acc
        for i in range(3):
            acc = 0.0
            for j in range(3):
                acc += Hll_inv[l, i, j] * v[j]
            dl[l, i] = -acc


@njit(cache=True, nogil=True)
def apply_update(Rwb, pwb, lmp, dp, dl, lm_ok, Rwb_out, pwb_out, lmp_out):
    P = Rwb.shape[0]
    w = np.zeros(3)
    for s in range(P):
        for i in range(3):
            w[i] = dp[6 * s + i]
        Rwb_out[s] = Rwb[s] @ so3_exp_nb(w)
        for i in range(3):
            pwb_out[s, i] = pwb[s, i] + dp[6 * s + 3 + i]
    for l in range(lmp.shape[0]):
        for i in range(3):
            lmp_out[l, i] = lmp[l, i]
        if lm_ok[l]:
            r = lmp[l, 2] + dl[l, 2]
            if r > 1e-9 and math.isfinite(r):
                lmp_out[l, 0] = lmp[l, 0] + dl[l, 0]
                lmp_out[l, 1] = lmp[l, 1] + dl[l, 1]
                lmp_out[l, 2] = r


# ------------------------------------------------------------------------------------------------ the solver
# ------------------------------------------------------------------ P1.2 W3 item 1: pose-free landmarks (exact)
# Moved VERBATIM from xclean/vio/budget.py: their arithmetic is this module's own
# (obs_residual, _huber, _cam_poses, _mtm, build_system's landmark block order), so the window solve with a `pf` block
# equals the full solve bit for bit.  A landmark is POSE-FREE when its only valid observation is its host's LEFT one
# (stereo form "prior"): at t == h its pose Jacobians are zero, so it adds nothing to the Schur system; it keeps its own
# 3x3 step inside the LM loop and its cost inside the accept test (solve below).
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


class Problem:
    """One window's arrays in solver form (assembled by vio/engine_live.P1Engine._problem()).

    VO: every slot is a 6-dim pose block.  VI (Phase 2): `dims` (P,) gives 6 (pose) or 15 (full state [dphi, dp, dv,
    dbg, dba]) per slot, `vel` / `bgs` / `bas` (P,3) hold the full states' extra variables (unused rows for pose
    slots), `imu` is a list of (i, j, Preint, W) between full slots (W = imu_factor.sqrt_information), `g_w` the
    window's gravity (held fixed in the solve).  `pf` (P1.2 W3 item 1): None, or a PoseFree block of landmarks kept
    OUT of the Schur system (solve / final_linearisation treat them exactly; marg_system never sees one)."""

    __slots__ = ("Rwb", "pwb", "Rbc", "pbc", "lmp", "lm_host", "lm_mask", "rho_meas", "rho_isig", "ob_start",
                 "ob_pose", "ob_cam", "ob_z", "ob_isig", "ob_valid", "fx", "fy", "cx", "cy", "B", "huber_k",
                 "prior", "dims", "vel", "bgs", "bas", "imu", "g_w", "pf")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __setstate__(self, state):
        """Unpickling: a Problem pickled before a slot existed (e.g. `pf`, P1.2 W3) gets None for it."""
        d, s = state if isinstance(state, tuple) else (state, None)
        for k in self.__slots__:
            setattr(self, k, None)
        for src in (d, s):
            for k, v in (src or {}).items():
                setattr(self, k, v)


class PoseFree:
    """P1.2 W3 item 1: the pose-free landmarks of one window solve.  host (n,) slot indices, z (n,2)
    the host LEFT pixels, isg the left observations' common whitening (1 / sigma_px), rmeas / ris (n,) the inverse-depth
    prior (the same formula as the kept landmarks), lmp (n,3) the state (IN / OUT: updated on every accepted LM
    iteration), Hinv (n,3,3) / ok (n,) OUT: final_linearisation's marginals and lm_ok."""

    def __init__(self, host, z, isg, rmeas, ris, lmp):
        self.host = np.ascontiguousarray(host, np.int64)
        self.z = np.ascontiguousarray(np.asarray(z, np.float64).reshape(-1, 2))
        self.isg = float(isg)
        self.rmeas = np.ascontiguousarray(rmeas, np.float64)
        self.ris = np.ascontiguousarray(ris, np.float64)
        self.lmp = np.ascontiguousarray(np.asarray(lmp, np.float64).reshape(-1, 3)).copy()
        n = len(self.host)
        self.Hinv = np.zeros((n, 3, 3))
        self.ok = np.zeros(n, np.bool_)


def _pf_of(pb):
    pf = getattr(pb, "pf", None)
    return pf if (pf is not None and len(pf.host) > 0) else None


def _offsets(pb):
    P = pb.Rwb.shape[0]
    dims = np.full(P, 6, np.int64) if pb.dims is None else np.asarray(pb.dims, np.int64)
    off = np.zeros(P + 1, np.int64)
    np.cumsum(dims, out=off[1:])
    return dims, off


def _state(pb, R=None, p=None, v=None, bg=None, ba=None):
    return (pb.Rwb if R is None else R, pb.pwb if p is None else p, pb.vel if v is None else v,
            pb.bgs if bg is None else bg, pb.bas if ba is None else ba)


def _imu_terms(pb, st, off, n, want_system=True):
    """The IMU factors' (H, b, robust cost, per-factor whitened chi2) at state st (Huber on the whitened norm with
    the dof-equivalent threshold, imu_factor.robust_weight rule 'dof')."""
    from xclean.vio import imu_factor as F
    H = np.zeros((n, n)) if want_system else None
    b = np.zeros(n) if want_system else None
    cost = 0.0
    chi2 = []
    if not pb.imu:
        return H, b, cost, chi2
    kd = F.huber_k_dof(15)
    Rwb, pwb, vel, bgs, bas = st
    for (i, j, pre, W) in pb.imu:
        si = (Rwb[i], pwb[i], vel[i], bgs[i], bas[i])
        sj = (Rwb[j], pwb[j], vel[j], bgs[j], bas[j])
        r, Ji, Jj = F.imu_residual_jacobians(si, sj, pre, pb.g_w, W=W)
        e2 = float(r @ r)
        chi2.append(e2)
        e = np.sqrt(e2)
        w = 1.0 if e <= kd else kd / e
        cost += 0.5 * e2 if e <= kd else kd * (e - 0.5 * kd)
        if want_system:
            ci = np.arange(off[i], off[i] + 15)
            cj = np.arange(off[j], off[j] + 15)
            H[np.ix_(ci, ci)] += w * (Ji.T @ Ji)
            H[np.ix_(cj, cj)] += w * (Jj.T @ Jj)
            Hij = w * (Ji.T @ Jj)
            H[np.ix_(ci, cj)] += Hij
            H[np.ix_(cj, ci)] += Hij.T
            b[ci] += w * (Ji.T @ r)
            b[cj] += w * (Jj.T @ r)
    return H, b, cost, chi2


def _full_system(pb, lam, H6, b6, dims, off):
    """Scatter the Schur-reduced visual pose system (6P) into the full (n) system and add prior + IMU terms."""
    P = pb.Rwb.shape[0]
    n = int(off[P])
    H = np.zeros((n, n))
    b = np.zeros(n)
    pc = np.concatenate([np.arange(off[s], off[s] + 6) for s in range(P)])
    H[np.ix_(pc, pc)] = H6
    b[pc] = b6
    st = _state(pb)
    cost = 0.0
    if pb.prior is not None:
        Hp, bp, c = pb.prior.terms(st, n)
        H += Hp
        b += bp
        cost += c
    Hi, bi, ci, _chi = _imu_terms(pb, st, off, n)
    if Hi is not None and pb.imu:
        H += Hi
        b += bi
        cost += ci
    return H, b, cost, pc


def solve(pb, iters=5, lam0=1e-4):
    """Fixed-iteration LM (class D: 5).  Each iteration: linearise + Schur (numba), add the prior and the IMU factors,
    damp, Cholesky, back-substitute, second cost pass, accept / reject.  Returns dict(cost0, cost, accepted, iters)."""
    P = pb.Rwb.shape[0]
    L = pb.lmp.shape[0]
    dims, off = _offsets(pb)
    vi = pb.vel is not None
    H6 = np.zeros((6 * P, 6 * P))
    b6 = np.zeros(6 * P)
    Hll_inv = np.zeros((L, 3, 3))
    bl = np.zeros((L, 3))
    Hlp = np.zeros((L, P, 3, 6))
    lm_ok = np.zeros(L, np.bool_)
    ob_chi2 = np.zeros(len(pb.ob_pose))
    dl = np.zeros((L, 3))
    R2 = np.empty_like(pb.Rwb)
    p2 = np.empty_like(pb.pwb)
    l2 = np.empty_like(pb.lmp)
    lam = lam0
    cost0 = None
    accepted = 0
    cur_cost = None
    pf = _pf_of(pb)                    # P1.2 W3 item 1: pose-free landmarks, exact
    if pf is not None:
        pf_trial = np.zeros_like(pf.lmp)
        pf_Hs = np.zeros((len(pf.host), 3, 3))
        pf_oks = np.zeros(len(pf.host), np.bool_)
    for it in range(iters):
        c_vis = build_system(pb.Rwb, pb.pwb, pb.Rbc, pb.pbc, pb.lmp, pb.lm_host, pb.lm_mask, pb.rho_meas,
                             pb.rho_isig, pb.ob_start, pb.ob_pose, pb.ob_cam, pb.ob_z, pb.ob_isig, pb.ob_valid,
                             pb.fx, pb.fy, pb.cx, pb.cy, pb.B, pb.huber_k, lam, H6, b6, Hll_inv, bl, Hlp, lm_ok,
                             ob_chi2)
        if pf is not None:
            # their cost joins c_tot; their own damped 3x3 step at this lam is the trial (zero pose coupling)
            c_vis += posefree_cost(pb.Rwb, pb.pwb, pb.Rbc, pb.pbc, pf.lmp, pf.host, pf.rmeas, pf.ris, pf.z, pf.isg,
                                   pb.fx, pb.fy, pb.cx, pb.cy, pb.B, pb.huber_k, False)
            posefree_step(pb.Rwb, pb.pwb, pb.Rbc, pb.pbc, pf.lmp, pf.host, pf.rmeas, pf.ris, pf.z, pf.isg, pb.fx, pb.fy,
                          pb.cx, pb.cy, pb.huber_k, lam, pf_trial, pf_Hs, pf_oks)
        H, bvec, c_rest, pc = _full_system(pb, lam, H6, b6, dims, off)
        c_tot = c_vis + c_rest
        if cost0 is None:
            cost0 = c_tot
        cur_cost = c_tot
        n = H.shape[0]
        Hd = H.copy()
        dg = np.diag(Hd).copy()
        Hd[np.diag_indices(n)] += lam * dg + 1e-9
        try:
            Lc = np.linalg.cholesky(Hd)
            dx = -np.linalg.solve(Lc.T, np.linalg.solve(Lc, bvec))
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue
        dp6 = np.ascontiguousarray(dx[pc])
        back_substitute(dp6, Hll_inv, bl, Hlp, lm_ok, dl)
        apply_update(pb.Rwb, pb.pwb, pb.lmp, dp6, dl, lm_ok, R2, p2, l2)
        v2 = b2g = b2a = None
        if vi:
            v2 = pb.vel.copy()
            b2g = pb.bgs.copy()
            b2a = pb.bas.copy()
            for s in range(P):
                if dims[s] == 15:
                    o = off[s]
                    v2[s] += dx[o + 6:o + 9]
                    b2g[s] += dx[o + 9:o + 12]
                    b2a[s] += dx[o + 12:o + 15]
        c_new = visual_cost(R2, p2, pb.Rbc, pb.pbc, l2, pb.lm_host, pb.lm_mask, pb.rho_meas, pb.rho_isig,
                            pb.ob_start, pb.ob_pose, pb.ob_cam, pb.ob_z, pb.ob_isig, pb.ob_valid, pb.fx, pb.fy, pb.cx,
                            pb.cy, pb.B, pb.huber_k)
        if pf is not None:
            c_new += posefree_cost(R2, p2, pb.Rbc, pb.pbc, pf_trial, pf.host, pf.rmeas, pf.ris, pf.z, pf.isg, pb.fx,
                                   pb.fy, pb.cx, pb.cy, pb.B, pb.huber_k, True)
        st2 = (R2, p2, v2, b2g, b2a) if vi else (R2, p2, None, None, None)
        if pb.prior is not None:
            c_new += pb.prior.cost(st2)
        if vi and pb.imu:
            c_new += _imu_terms(pb, st2, off, 0, want_system=False)[2]
        if c_new < c_tot:
            pb.Rwb[:] = R2
            pb.pwb[:] = p2
            pb.lmp[:] = l2
            if pf is not None:
                pf.lmp[:] = pf_trial
            if vi:
                pb.vel[:] = v2
                pb.bgs[:] = b2g
                pb.bas[:] = b2a
            lam = max(lam / 10.0, 1e-12)
            accepted += 1
            cur_cost = c_new
        else:
            lam *= 10.0
    return {"cost0": cost0, "cost": cur_cost, "accepted": accepted, "iters": iters, "lam": lam}


def final_linearisation(pb):
    """One undamped linearisation at the solution: the full reduced information (prior and IMU included) for the
    newest state's covariance, the whitened chi2 of every observation (outlier removal, robust scale), and the IMU
    factors' whitened chi2 (kappa_imu, the watchdog)."""
    P = pb.Rwb.shape[0]
    L = pb.lmp.shape[0]
    dims, off = _offsets(pb)
    H6 = np.zeros((6 * P, 6 * P))
    b6 = np.zeros(6 * P)
    Hll_inv = np.zeros((L, 3, 3))
    bl = np.zeros((L, 3))
    Hlp = np.zeros((L, P, 3, 6))
    lm_ok = np.zeros(L, np.bool_)
    ob_chi2 = np.zeros(len(pb.ob_pose))
    build_system(pb.Rwb, pb.pwb, pb.Rbc, pb.pbc, pb.lmp, pb.lm_host, pb.lm_mask, pb.rho_meas, pb.rho_isig,
                 pb.ob_start, pb.ob_pose, pb.ob_cam, pb.ob_z, pb.ob_isig, pb.ob_valid, pb.fx, pb.fy, pb.cx, pb.cy,
                 pb.B, pb.huber_k, 0.0, H6, b6, Hll_inv, bl, Hlp, lm_ok, ob_chi2)
    pf = _pf_of(pb)
    if pf is not None:                 # P1.2 W3 item 1: the pose-free marginals at lam = 0 (into pf.Hinv / pf.ok)
        posefree_step(pb.Rwb, pb.pwb, pb.Rbc, pb.pbc, pf.lmp, pf.host, pf.rmeas, pf.ris, pf.z, pf.isg, pb.fx, pb.fy,
                      pb.cx, pb.cy, pb.huber_k, 0.0, np.zeros_like(pf.lmp), pf.Hinv, pf.ok)
    H, bvec, _c, _pc = _full_system(pb, 0.0, H6, b6, dims, off)
    imu_chi2 = _imu_terms(pb, _state(pb), off, 0, want_system=False)[3] if (pb.imu and pb.vel is not None) else []
    return H, bvec, ob_chi2, lm_ok, Hll_inv, imu_chi2, off


def marg_system(pb, lm_mask):
    """The linear system (H, b at the current estimate) of the factors that touch the variables being marginalised:
    the visual observations of the masked landmarks (Schur'd out), the prior and the IMU factors.  -> (H, b, off)."""
    P = pb.Rwb.shape[0]
    L = pb.lmp.shape[0]
    dims, off = _offsets(pb)
    H6 = np.zeros((6 * P, 6 * P))
    b6 = np.zeros(6 * P)
    Hll_inv = np.zeros((L, 3, 3))
    bl = np.zeros((L, 3))
    Hlp = np.zeros((L, P, 3, 6))
    lm_ok = np.zeros(L, np.bool_)
    chi2 = np.zeros(len(pb.ob_pose))
    build_system(pb.Rwb, pb.pwb, pb.Rbc, pb.pbc, pb.lmp, pb.lm_host, np.ascontiguousarray(lm_mask), pb.rho_meas,
                 pb.rho_isig, pb.ob_start, pb.ob_pose, pb.ob_cam, pb.ob_z, pb.ob_isig, pb.ob_valid, pb.fx, pb.fy,
                 pb.cx, pb.cy, pb.B, pb.huber_k, 0.0, H6, b6, Hll_inv, bl, Hlp, lm_ok, chi2)
    H, b, _c, _pc = _full_system(pb, 0.0, H6, b6, dims, off)
    return H, b, off


# ------------------------------------------------------------------------------------------- gating prediction
@njit(cache=True, nogil=True)
def predict_projection(Rwb_t, pwb_t, Rbc, pbc, Rwc_h, pwc_h, lmp, Spose, Sl, sig_px, fx, fy, cx, cy, W, H,
                       uv, S, ok, depth):
    """Project every landmark into the PREDICTED pose (body Rwb_t, pwb_t) and form its innovation covariance
    S = Jp Spose Jp^T + Jl Sl Jl^T + sig_px^2 I (Spose: 6x6 in [dphi, dp]; Sl: (L,3,3) landmark marginals or 0).
    Allocation-free inner loop."""
    Rwc_t = np.zeros((3, 3))
    _mm(Rwb_t, Rbc, Rwc_t)
    pwc_t = np.zeros(3)
    for i in range(3):
        pwc_t[i] = pwb_t[i] + Rwb_t[i, 0] * pbc[0] + Rwb_t[i, 1] * pbc[1] + Rwb_t[i, 2] * pbc[2]
    Rcb = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Rcb[i, j] = Rbc[j, i]
    Px = np.zeros((3, 3))
    Px[0, 1] = -pbc[2]
    Px[0, 2] = pbc[1]
    Px[1, 0] = pbc[2]
    Px[1, 2] = -pbc[0]
    Px[2, 0] = -pbc[1]
    Px[2, 1] = pbc[0]
    Wt = np.zeros((3, 3))
    _mm(Rwb_t, Px, Wt)
    Rct = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Rct[i, j] = Rwc_t[j, i]
    RctWt = np.zeros((3, 3))
    _mm(Rct, Wt, RctWt)
    Qx = np.zeros((3, 3))
    QR = np.zeros((3, 3))
    Jp = np.zeros((2, 6))
    Jl = np.zeros((2, 3))
    RcR = np.zeros((3, 3))
    Dp = np.zeros((2, 3))
    tmp6 = np.zeros((2, 6))
    tmp3 = np.zeros((2, 3))
    L = lmp.shape[0]
    for l in range(L):
        ok[l] = False
        a = lmp[l, 0]
        b = lmp[l, 1]
        rho = lmp[l, 2]
        d0 = pwc_h[l, 0] - pwc_t[0]
        d1 = pwc_h[l, 1] - pwc_t[1]
        d2 = pwc_h[l, 2] - pwc_t[2]
        Y0 = Rwc_h[l, 0, 0] * a + Rwc_h[l, 0, 1] * b + Rwc_h[l, 0, 2] + rho * d0
        Y1 = Rwc_h[l, 1, 0] * a + Rwc_h[l, 1, 1] * b + Rwc_h[l, 1, 2] + rho * d1
        Y2 = Rwc_h[l, 2, 0] * a + Rwc_h[l, 2, 1] * b + Rwc_h[l, 2, 2] + rho * d2
        q0 = Rct[0, 0] * Y0 + Rct[0, 1] * Y1 + Rct[0, 2] * Y2
        q1 = Rct[1, 0] * Y0 + Rct[1, 1] * Y1 + Rct[1, 2] * Y2
        q2 = Rct[2, 0] * Y0 + Rct[2, 1] * Y1 + Rct[2, 2] * Y2
        if q2 <= 1e-9 or rho <= 0.0:
            continue
        depth[l] = q2 / rho
        iq = 1.0 / q2
        u = fx * q0 * iq + cx
        v = fy * q1 * iq + cy
        uv[l, 0] = u
        uv[l, 1] = v
        if u < 0.0 or u >= W or v < 0.0 or v >= H:
            continue
        Dp[0, 0] = fx * iq
        Dp[0, 1] = 0.0
        Dp[0, 2] = -fx * q0 * iq * iq
        Dp[1, 0] = 0.0
        Dp[1, 1] = fy * iq
        Dp[1, 2] = -fy * q1 * iq * iq
        Qx[0, 1] = -q2
        Qx[0, 2] = q1
        Qx[1, 0] = q2
        Qx[1, 2] = -q0
        Qx[2, 0] = -q1
        Qx[2, 1] = q0
        _mm(Qx, Rcb, QR)
        for r in range(2):
            for j in range(3):
                Jp[r, j] = (Dp[r, 0] * (QR[0, j] + rho * RctWt[0, j]) + Dp[r, 1] * (QR[1, j] + rho * RctWt[1, j])
                            + Dp[r, 2] * (QR[2, j] + rho * RctWt[2, j]))
                Jp[r, 3 + j] = -rho * (Dp[r, 0] * Rct[0, j] + Dp[r, 1] * Rct[1, j] + Dp[r, 2] * Rct[2, j])
        for i in range(3):
            for j in range(3):
                RcR[i, j] = Rct[i, 0] * Rwc_h[l, 0, j] + Rct[i, 1] * Rwc_h[l, 1, j] + Rct[i, 2] * Rwc_h[l, 2, j]
        e0 = Rct[0, 0] * d0 + Rct[0, 1] * d1 + Rct[0, 2] * d2
        e1 = Rct[1, 0] * d0 + Rct[1, 1] * d1 + Rct[1, 2] * d2
        e2 = Rct[2, 0] * d0 + Rct[2, 1] * d1 + Rct[2, 2] * d2
        for r in range(2):
            Jl[r, 0] = Dp[r, 0] * RcR[0, 0] + Dp[r, 1] * RcR[1, 0] + Dp[r, 2] * RcR[2, 0]
            Jl[r, 1] = Dp[r, 0] * RcR[0, 1] + Dp[r, 1] * RcR[1, 1] + Dp[r, 2] * RcR[2, 1]
            Jl[r, 2] = Dp[r, 0] * e0 + Dp[r, 1] * e1 + Dp[r, 2] * e2
        # C = Jp Spose Jp^T + Jl Sl Jl^T
        for r in range(2):
            for j in range(6):
                acc = 0.0
                for k in range(6):
                    acc += Jp[r, k] * Spose[k, j]
                tmp6[r, j] = acc
            for j in range(3):
                acc = 0.0
                for k in range(3):
                    acc += Jl[r, k] * Sl[l, k, j]
                tmp3[r, j] = acc
        for r in range(2):
            for c in range(2):
                acc = 0.0
                for k in range(6):
                    acc += tmp6[r, k] * Jp[c, k]
                for k in range(3):
                    acc += tmp3[r, k] * Jl[c, k]
                S[l, r, c] = acc + (sig_px * sig_px if r == c else 0.0)
        ok[l] = True
    return 0
