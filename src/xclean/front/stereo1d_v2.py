#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1 1-D STEREO, VERSION 2 (ST2): the disparity of each requested left keypoint on the rectified pair, CPU (numba).

Same API as `stereo1d.py` (an engine flag can switch between them):

    d, sig, ok, zn = disparities(left, right, kp, dlo=None, dhi=None)
    d, sig, ok, amb, span, zn = disparities_ex(left, right, kp, dlo=None, dhi=None)
    stereo_points(L, R, kp, dlo, dhi, out_d, out_sig, out_ok, out_zn, gap_alpha, out_amb, out_span)

`amb` is always False and `span` 0 (the P1-03d ambiguous-peak arm was dropped); `gap_alpha` is accepted and ignored.

WHY A VERSION 2
  - The P1-03c distinctiveness rule (Gumbel spacing of the row's ZNCC scores) is nearly non-discriminative on real
    images. It removes 55 % of correct-peak and 68 % of wrong-peak keypoints, cuts the correct yield from 62 % to 28 %,
    and alone fails tests/test_p1_front.py::test_stereo_subpixel (0.765 < 0.90). Its null assumes independent chance
    scores; neighbouring ZNCC scores of a smooth texture are strongly correlated.
  - The fronto-parallel ZNCC >= 0.5 floor rejects 37 % of correct-peak keypoints on KITTI 01's slanted road.
  - 82 % of the gross errors are depth-edge errors that take the NEARER surface (window "foreground fattening").
  - The Gauss-Newton sigma under-states the error about 4.7x; part of the inlier error is the bilinear
    interpolation of the left template at the sub-pixel keypoint (error MAD 0.20 px at half-pixel phase vs 0.15).

THE CHAIN
  1. Border and flat-patch skips as stereo1d (11x11 patch at the INTEGER pixel nearest the keypoint).
  2. TEXTURE SUFFICIENCY along x: the Cramer-Rao bound of a 1-D shift of this window under the frame's noise,
     sigma_CRB = sigma_n sqrt(2 / sum g_x^2) (sigma_n: Immerkaer 1996 on the left image, every NOISE_ROW_STEP-th row,
     class C; g_x central difference; the factor 2 = noise in both images). Rejected when LOWTEX_NSIG sigma_CRB >
     GROSS_PX: an ideal matcher could not place the match inside the 1-px gross bar (class D: 3 sigma, 1 px).
  3. INTEGER SEARCH: ZNCC along the right row for d in [dlo, dhi] of the full 11x11 window AND of its four
     HALF-WINDOWS (left / right columns -5..0 / 0..5, top / bottom rows -5..0 / 0..5), all from one pass of 3 x 3
     block sums (no extra multiply per pixel).
  4. UNIQUENESS (carried, sgm32): the second best (> 1 px away) dissimilarity exceeds the best x (1 + UNIQUENESS).
  5. LEFT-RIGHT CHECK (carried, sgm32's 1 px): the right patch at the peak, searched back along the full left row.
  6. SUB-PIXEL: LK on a PLANAR disparity patch, d(x, y) = d + gu x + gv y (the disparity of any 3-D plane is affine in
     the pixel coordinates: exact for planes, class A), with gain and offset, at the INTEGER centre: the left template
     is the raw pixel grid (no interpolation), the right image is sampled on its own integer rows, linear in x only.
     LK_ITERS Gauss-Newton steps (carried). The keypoint's disparity is the plane at the sub-pixel keypoint.
     Converged: finite, |d - seed| <= 1 px and gain in (0.5, 2) (stereo1d's rule, carried), |gu| < 1. The upper bound
     is class A: a surface seen by both cameras keeps x - d(x) increasing, i.e. gu < 1 (gu -> 1 = seen edge-on by the
     right camera). The lower bound -1 is NOT physically required (gu -> -inf = edge-on for the left camera); it is a
     class-D symmetric choice (a right-image width at most twice the left's). On the bench it decided 1 keypoint of
     247,826 (ST2-02).
  7. MULTIPLE-WINDOW CONSENSUS (replaces the ZNCC floor and the P1-03c rule): each half-window's own integer peak must
     lie within CONS_TOL_PX of the fitted plane's disparity at that half's centroid (+-2.5 px from the centre).
     - one surface, enough texture: all four halves find the same surface where the plane says (slant included);
     - a window straddling a depth edge: the halves on the two surfaces disagree;
     - a chance peak (no true match in the row): the four halves' own peaks scatter over the row.
     Tolerance: the left-right threshold LR_THRESHOLD_PX (carried) + 0.5 px, the rounding of an integer peak (class A).
     The windows are the patch split at its centre (parameter-free; Fusiello et al. 1997 / Hirschmuller et al. 2002
     use multiple supporting windows to choose, here they must agree).
  8. d >= MIN_DISP_PX (carried: a smaller disparity is a bearing, not a depth).

SIGMA (per point): the sandwich (heteroskedasticity- and autocorrelation-consistent, HAC) variance of the fitted
plane's disparity at the keypoint, Var = c^T A^-1 (sum_kl w_kl J_k J_l^T r_k r_l) A^-1 c, c = (1, 0, 0, du, dv),
A = J^T J, Bartlett product weights of bandwidth HAC_BW px (Newey & West 1987; 2-D: Conley 1999). The Gauss-Newton
form assumes independent pixel residuals; the residuals of an interpolated, optically blurred image are not.

Single-threaded and deterministic. `stereo_points_diag` runs the same chain recording every check (bench only).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from xclean.stereo.sgm32 import MAX_DISP, UNIQUENESS, LR_THRESHOLD_PX, MIN_DISP_PX
from xclean.stereo.epipolar import HALF, LK_ITERS

PW = 2 * HALF + 1
NPIX = PW * PW
NPAR = 5                          # d, gain, offset, gu, gv
CONS_TOL_PX = LR_THRESHOLD_PX + 0.5
LOWTEX_NSIG = 3.0
GROSS_PX = 1.0
GAIN_LO = 0.5
GAIN_HI = 2.0
SLANT_MAX = 1.0
HAC_BW = 1
NOISE_ROW_STEP = 4
GAP_ALPHA = 0.0                   # API compatibility only: version 2 has no gap rule

# failed-check bits (diag); production stops at the first failed check, the diag path evaluates all it can
F_BORDER = 1
F_EMPTY = 2
F_FLAT = 4
F_LOWTEX = 8
F_UNIQ = 16
F_FLATR = 32
F_LR = 64
F_LK = 128
F_CONS = 256
F_MIND = 512
NDIAG = 26


@njit(cache=True, nogil=True)
def noise_sigma(L):
    """Immerkaer (1996) noise sigma of an 8-bit image, on every NOISE_ROW_STEP-th interior row."""
    H, W = L.shape
    s = 0.0
    cnt = 0
    for y in range(1, H - 1, NOISE_ROW_STEP):
        for x in range(1, W - 1):
            v = (int(L[y - 1, x - 1]) - 2 * int(L[y - 1, x]) + int(L[y - 1, x + 1])
                 - 2 * int(L[y, x - 1]) + 4 * int(L[y, x]) - 2 * int(L[y, x + 1])
                 + int(L[y + 1, x - 1]) - 2 * int(L[y + 1, x]) + int(L[y + 1, x + 1]))
            s += abs(v)
            cnt += 1
    if cnt == 0:
        return 0.0
    return math.sqrt(math.pi / 2.0) / 6.0 * s / cnt


@njit(cache=True, nogil=True)
def _zn(sr, sr2, srt, n, st, st2):
    cov = srt - sr * st / n
    vr = sr2 - sr * sr / n
    vt = st2 - st * st / n
    den = vr * vt
    if den > 1e-12:
        return cov / math.sqrt(den)
    return -2.0


@njit(cache=True, nogil=True)
def _band(i):
    if i < HALF:
        return 0
    if i == HALF:
        return 1
    return 2


@njit(cache=True, nogil=True)
def _chol_solve(A, b, n, x):
    """In-place Cholesky of the n x n SPD block of A (lower), solve A x = b. False if not positive definite."""
    for j in range(n):
        s = A[j, j]
        for k in range(j):
            s -= A[j, k] * A[j, k]
        if not (s > 0.0):
            return False
        A[j, j] = math.sqrt(s)
        for i in range(j + 1, n):
            s = A[i, j]
            for k in range(j):
                s -= A[i, k] * A[j, k]
            A[i, j] = s / A[j, j]
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= A[i, k] * x[k]
        x[i] = s / A[i, i]
    for i in range(n - 1, -1, -1):
        s = x[i]
        for k in range(i + 1, n):
            s -= A[k, i] * x[k]
        x[i] = s / A[i, i]
    return True


@njit(cache=True, nogil=True)
def _lk_plane(L, R, ui, vi, seed, th, Tt, A, g, J, xs):
    """Planar-disparity affine LK at the integer centre. th <- (d, gain, offset, gu, gv). -> finite."""
    W = R.shape[1]
    th[0] = seed
    th[1] = 1.0
    th[2] = 0.0
    th[3] = 0.0
    th[4] = 0.0
    for _it in range(LK_ITERS):
        for i in range(NPAR):
            g[i] = 0.0
            for j in range(NPAR):
                A[i, j] = 0.0
        k = 0
        for r in range(PW):
            dy = r - HALF
            y = vi + dy
            for c in range(PW):
                dx = c - HALF
                x = ui + dx - (th[0] + th[3] * dx + th[4] * dy)
                x0 = int(math.floor(x))
                if x0 < 0:
                    x0 = 0
                if x0 > W - 2:
                    x0 = W - 2
                f = x - x0
                a0 = float(R[y, x0])
                gr = float(R[y, x0 + 1]) - a0
                rv = a0 + f * gr
                res = Tt[k] - (th[1] * rv + th[2])
                J[0] = -th[1] * gr
                J[1] = rv
                J[2] = 1.0
                J[3] = J[0] * dx
                J[4] = J[0] * dy
                for i in range(NPAR):
                    g[i] += J[i] * res
                    for j in range(i + 1):
                        A[i, j] += J[i] * J[j]
                k += 1
        for i in range(NPAR):
            A[i, i] += 1e-9
        if not _chol_solve(A, g, NPAR, xs):
            return False
        for i in range(NPAR):
            th[i] += xs[i]
        if not (math.isfinite(th[0]) and math.isfinite(th[1]) and math.isfinite(th[3]) and math.isfinite(th[4])):
            return False
    return True


@njit(cache=True, nogil=True)
def _lk_sigma(L, R, ui, vi, th, Tt, du, dv, A, J, xs, Q, out4):
    """At the LK solution: out4 <- (GN variance, HAC variance at HAC_BW, HAC b=0, HAC b=2) of the plane's disparity
    at the keypoint (du, dv) = keypoint - integer centre. False if the normal matrix is singular."""
    W = R.shape[1]
    Jall = np.empty((NPIX, NPAR))
    rs = np.empty(NPIX)
    for i in range(NPAR):
        for j in range(NPAR):
            A[i, j] = 0.0
    rss = 0.0
    k = 0
    for r in range(PW):
        dy = r - HALF
        y = vi + dy
        for c in range(PW):
            dx = c - HALF
            x = ui + dx - (th[0] + th[3] * dx + th[4] * dy)
            x0 = int(math.floor(x))
            if x0 < 0:
                x0 = 0
            if x0 > W - 2:
                x0 = W - 2
            f = x - x0
            a0 = float(R[y, x0])
            gr = float(R[y, x0 + 1]) - a0
            rv = a0 + f * gr
            res = Tt[k] - (th[1] * rv + th[2])
            rs[k] = res
            rss += res * res
            Jall[k, 0] = -th[1] * gr
            Jall[k, 1] = rv
            Jall[k, 2] = 1.0
            Jall[k, 3] = Jall[k, 0] * dx
            Jall[k, 4] = Jall[k, 0] * dy
            for i in range(NPAR):
                for j in range(i + 1):
                    A[i, j] += Jall[k, i] * Jall[k, j]
            k += 1
    for i in range(NPAR):
        A[i, i] += 1e-9
    # gvec = A^-1 c, c = (1, 0, 0, du, dv)
    cv = np.zeros(NPAR)
    cv[0] = 1.0
    cv[3] = du
    cv[4] = dv
    if not _chol_solve(A, cv, NPAR, xs):
        return False
    s2 = rss / (NPIX - NPAR)
    gn = 0.0
    gn += xs[0] + du * xs[3] + dv * xs[4]           # c^T A^-1 c
    out4[0] = s2 * gn
    k = 0
    for r in range(PW):
        for c in range(PW):
            q = 0.0
            for i in range(NPAR):
                q += xs[i] * Jall[k, i]
            Q[r, c] = q * rs[k]
            k += 1
    for slot, b in ((1, HAC_BW), (2, 0), (3, 2)):
        h = 0.0
        for ly in range(-b, b + 1):
            wy = 1.0 - abs(ly) / (b + 1.0)
            for lx in range(-b, b + 1):
                wx = 1.0 - abs(lx) / (b + 1.0)
                acc = 0.0
                for r in range(PW):
                    r2 = r + ly
                    if r2 < 0 or r2 >= PW:
                        continue
                    for c in range(PW):
                        c2 = c + lx
                        if c2 < 0 or c2 >= PW:
                            continue
                        acc += Q[r, c] * Q[r2, c2]
                h += wx * wy * acc
        out4[slot] = h
    return True


@njit(cache=True, nogil=True)
def _core(L, R, kp, dlo, dhi, out_d, out_sig, out_ok, out_zn, diag, strict):
    H, W = L.shape
    N = kp.shape[0]
    sn = noise_sigma(L)
    rec = diag.shape[0] == N
    T = np.empty((PW, PW))
    Tt = np.empty(NPIX)
    T2 = np.empty(NPIX)
    sc = np.empty(MAX_DISP)
    nb = MAX_DISP + PW + 2
    bR = np.empty((3, nb))
    bR2 = np.empty((3, nb))
    cL = np.empty(nb + MAX_DISP)
    cL2 = np.empty(nb + MAX_DISP)
    C = np.empty((3, 3))
    SR = np.empty((3, 3))
    SR2 = np.empty((3, 3))
    sT = np.zeros((3, 3))
    sT2 = np.zeros((3, 3))
    nB = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            nB[i, j] = (HALF if i != 1 else 1) * (HALF if j != 1 else 1)
    wS = np.empty(5)
    wS2 = np.empty(5)
    wn = np.empty(5)
    hb = np.empty(5, np.int64)
    hs = np.empty(5)
    th = np.empty(NPAR)
    A = np.empty((NPAR, NPAR))
    gg = np.empty(NPAR)
    J = np.empty(NPAR)
    xs = np.empty(NPAR)
    Q = np.empty((PW, PW))
    o4 = np.empty(4)
    for n in range(N):
        out_ok[n] = False
        out_d[n] = 0.0
        out_sig[n] = np.inf
        out_zn[n] = -2.0
        if rec:
            for j in range(NDIAG):
                diag[n, j] = np.nan
            diag[n, 25] = sn
        fail = 0
        u = kp[n, 0]
        v = kp[n, 1]
        ui = int(math.floor(u + 0.5))
        vi = int(math.floor(v + 0.5))
        if ui - HALF - 2 < 0 or ui + HALF + 2 >= W or vi - HALF - 2 < 0 or vi + HALF + 2 >= H:
            if rec:
                diag[n, 0] = F_BORDER
            continue
        lo = dlo[n]
        hi = dhi[n]
        if lo < 0:
            lo = 0
        if hi > MAX_DISP - 1:
            hi = MAX_DISP - 1
        if hi > ui - HALF - 2:
            hi = ui - HALF - 2
        if hi < lo:
            if rec:
                diag[n, 0] = F_EMPTY
            continue
        # template (integer grid), block sums, x-gradient energy
        for i in range(3):
            for j in range(3):
                sT[i, j] = 0.0
                sT2[i, j] = 0.0
        g2 = 0.0
        k = 0
        for r in range(PW):
            br = _band(r)
            for c in range(PW):
                t = float(L[vi + r - HALF, ui + c - HALF])
                T[r, c] = t
                Tt[k] = t
                k += 1
                bc = _band(c)
                sT[br, bc] += t
                sT2[br, bc] += t * t
                gx = 0.5 * (float(L[vi + r - HALF, ui + c - HALF + 1]) - float(L[vi + r - HALF, ui + c - HALF - 1]))
                g2 += gx * gx
        tsum = 0.0
        tsum2 = 0.0
        for i in range(3):
            for j in range(3):
                tsum += sT[i, j]
                tsum2 += sT2[i, j]
        if tsum2 / NPIX - (tsum / NPIX) ** 2 <= 1e-12:
            if rec:
                diag[n, 0] = F_FLAT
            continue
        crb = sn * math.sqrt(2.0 / g2) if g2 > 0.0 else np.inf
        if rec:
            diag[n, 11] = crb
        if LOWTEX_NSIG * crb > GROSS_PX:
            fail |= F_LOWTEX
            if strict:
                continue
        # window template sums: 0 full, 1 L6 (col bands 0,1), 2 R6 (1,2), 3 T6 (row bands 0,1), 4 B6 (1,2)
        for w in range(5):
            wS[w] = 0.0
            wS2[w] = 0.0
            wn[w] = 0.0
        for i in range(3):
            for j in range(3):
                for w in range(5):
                    if (w == 0 or (w == 1 and j <= 1) or (w == 2 and j >= 1) or (w == 3 and i <= 1)
                            or (w == 4 and i >= 1)):
                        wS[w] += sT[i, j]
                        wS2[w] += sT2[i, j]
                        wn[w] += nB[i, j]
        # right-image band column sums for x in [xa, xb]
        xa = ui - hi - HALF
        xb = ui - lo + HALF
        for x in range(xa, xb + 1):
            for b in range(3):
                bR[b, x - xa] = 0.0
                bR2[b, x - xa] = 0.0
            for r in range(PW):
                p = float(R[vi + r - HALF, x])
                b = _band(r)
                bR[b, x - xa] += p
                bR2[b, x - xa] += p * p
        for w in range(5):
            hb[w] = -1
            hs[w] = -3.0
        for d in range(lo, hi + 1):
            x0 = ui - d - HALF                   # leftmost column of the right window
            for i in range(3):
                for j in range(3):
                    C[i, j] = 0.0
                    SR[i, j] = 0.0
                    SR2[i, j] = 0.0
            for r in range(PW):
                br = _band(r)
                row = vi + r - HALF
                a0 = 0.0
                a1 = 0.0
                a2 = 0.0
                for c in range(HALF):
                    a0 += float(R[row, x0 + c]) * T[r, c]
                a1 = float(R[row, x0 + HALF]) * T[r, HALF]
                for c in range(HALF + 1, PW):
                    a2 += float(R[row, x0 + c]) * T[r, c]
                C[br, 0] += a0
                C[br, 1] += a1
                C[br, 2] += a2
            for b in range(3):
                s0 = 0.0
                q0 = 0.0
                for c in range(HALF):
                    s0 += bR[b, x0 + c - xa]
                    q0 += bR2[b, x0 + c - xa]
                SR[b, 0] = s0
                SR2[b, 0] = q0
                SR[b, 1] = bR[b, x0 + HALF - xa]
                SR2[b, 1] = bR2[b, x0 + HALF - xa]
                s0 = 0.0
                q0 = 0.0
                for c in range(HALF + 1, PW):
                    s0 += bR[b, x0 + c - xa]
                    q0 += bR2[b, x0 + c - xa]
                SR[b, 2] = s0
                SR2[b, 2] = q0
            for w in range(5):
                a = 0.0
                q = 0.0
                cc = 0.0
                for i in range(3):
                    for j in range(3):
                        if (w == 0 or (w == 1 and j <= 1) or (w == 2 and j >= 1) or (w == 3 and i <= 1)
                                or (w == 4 and i >= 1)):
                            a += SR[i, j]
                            q += SR2[i, j]
                            cc += C[i, j]
                s = _zn(a, q, cc, wn[w], wS[w], wS2[w])
                if w == 0:
                    sc[d] = s
                if s > hs[w]:
                    hs[w] = s
                    hb[w] = d
        best = hb[0]
        bs = hs[0]
        out_zn[n] = bs
        if rec:
            diag[n, 1] = best
            diag[n, 2] = bs
            for w in range(4):
                diag[n, 12 + w] = hb[w + 1]
        if best < 0 or bs <= -2.0:
            if rec:
                diag[n, 0] = F_FLAT
            continue
        # uniqueness (sgm32's rule on 1 - zncc)
        sec = -2.0
        for d in range(lo, hi + 1):
            if abs(d - best) > 1 and sc[d] > sec:
                sec = sc[d]
        if not ((1.0 - sec) > (1.0 - bs) * (1.0 + UNIQUENESS)):
            fail |= F_UNIQ
            if strict:
                continue
        # left-right check: the right patch at the peak, searched back along the full LEFT row
        xr = ui - best
        k = 0
        rsum = 0.0
        rsum2 = 0.0
        for r in range(PW):
            for c in range(PW):
                p = float(R[vi + r - HALF, xr + c - HALF])
                T2[k] = p
                rsum += p
                rsum2 += p * p
                k += 1
        if rsum2 / NPIX - (rsum / NPIX) ** 2 <= 1e-12:
            fail |= F_FLATR
            if strict:
                continue
        else:
            dmax_rl = MAX_DISP - 1
            if xr + dmax_rl + HALF >= W:
                dmax_rl = W - 1 - HALF - xr
            xa2 = xr - HALF
            for x in range(xa2, xr + dmax_rl + HALF + 1):
                s0 = 0.0
                q0 = 0.0
                for r in range(PW):
                    p = float(L[vi + r - HALF, x])
                    s0 += p
                    q0 += p * p
                cL[x - xa2] = s0
                cL2[x - xa2] = q0
            bb = -1
            bbs = -3.0
            a = 0.0
            q = 0.0
            for c in range(PW):
                a += cL[c]
                q += cL2[c]
            for d in range(0, dmax_rl + 1):
                if d > 0:
                    a += cL[d + PW - 1] - cL[d - 1]
                    q += cL2[d + PW - 1] - cL2[d - 1]
                cc = 0.0
                k = 0
                for r in range(PW):
                    row = vi + r - HALF
                    for c in range(PW):
                        cc += float(L[row, xr + d + c - HALF]) * T2[k]
                        k += 1
                s = _zn(a, q, cc, float(NPIX), rsum, rsum2)
                if s > bbs:
                    bbs = s
                    bb = d
            if rec:
                diag[n, 21] = bb
            if bb < 0 or abs(bb - best) > LR_THRESHOLD_PX:
                fail |= F_LR
                if strict:
                    continue
        # sub-pixel: parabola seed at the integer centre, then the planar LK
        off = 0.0
        if best > lo and best < hi:
            y0 = sc[best - 1]
            y2 = sc[best + 1]
            den = y0 - 2.0 * bs + y2
            if den < -1e-9:
                off = 0.5 * (y0 - y2) / den
                if off > 0.5:
                    off = 0.5
                if off < -0.5:
                    off = -0.5
        seed = best + off
        fin = _lk_plane(L, R, ui, vi, seed, th, Tt, A, gg, J, xs)
        conv = fin and abs(th[0] - seed) <= 1.0 and th[1] > GAIN_LO and th[1] < GAIN_HI and abs(th[3]) < SLANT_MAX
        du = u - ui
        dv = v - vi
        dk = th[0] + th[3] * du + th[4] * dv
        if rec:
            diag[n, 8] = th[3]
            diag[n, 9] = th[4]
            diag[n, 10] = th[1]
            diag[n, 20] = th[0]
            diag[n, 22] = dk
        if not conv:
            fail |= F_LK
            if strict or not fin:
                if rec:
                    diag[n, 0] = fail
                continue
        # multiple-window consensus against the fitted plane
        cons = True
        for w in range(4):
            cx = -0.5 * HALF if w == 0 else (0.5 * HALF if w == 1 else 0.0)
            cy = -0.5 * HALF if w == 2 else (0.5 * HALF if w == 3 else 0.0)
            pred = th[0] + th[3] * cx + th[4] * cy
            if rec:
                diag[n, 16 + w] = pred
            if abs(hb[w + 1] - pred) > CONS_TOL_PX:
                cons = False
        if not cons:
            fail |= F_CONS
            if strict:
                continue
        if dk < MIN_DISP_PX:
            fail |= F_MIND
            if strict:
                continue
        okv = _lk_sigma(L, R, ui, vi, th, Tt, du, dv, A, J, xs, Q, o4)
        if rec:
            diag[n, 0] = fail
            if okv:
                diag[n, 3] = math.sqrt(max(o4[0], 0.0))
                diag[n, 4] = math.sqrt(max(o4[1], 0.0))
                diag[n, 5] = math.sqrt(max(o4[2], 0.0))
                diag[n, 6] = math.sqrt(max(o4[3], 0.0))
        if fail != 0 or not okv:
            continue
        out_d[n] = dk
        out_sig[n] = math.sqrt(max(o4[1], 0.0))
        out_ok[n] = True
    return 0


@njit(cache=True, nogil=True)
def stereo_points(L, R, kp, dlo, dhi, out_d, out_sig, out_ok, out_zn, gap_alpha, out_amb, out_span):
    """Same signature as stereo1d.stereo_points. gap_alpha is ignored; out_amb = False, out_span = 0."""
    for n in range(kp.shape[0]):
        out_amb[n] = False
        out_span[n] = 0.0
    dummy = np.zeros((0, NDIAG))
    return _core(L, R, kp, dlo, dhi, out_d, out_sig, out_ok, out_zn, dummy, True)


@njit(cache=True, nogil=True)
def stereo_points_diag(L, R, kp, dlo, dhi, out_d, out_sig, out_ok, out_zn, diag):
    """Bench path: every check evaluated where possible; diag (N, NDIAG):
    0 failed-check bits (0 = accepted), 1 integer peak, 2 peak ZNCC, 3 sigma GN, 4 sigma HAC (HAC_BW), 5 HAC b=0,
    6 HAC b=2, 8 gu, 9 gv, 10 gain, 11 sigma_CRB, 12-15 half peaks (L, R, T, B), 16-19 the plane's prediction at
    each half, 20 d at the integer centre, 21 left-right peak, 22 d at the keypoint, 25 sigma_n."""
    return _core(L, R, kp, dlo, dhi, out_d, out_sig, out_ok, out_zn, diag, False)


def _prep(left, right, kp, dlo, dhi):
    kp = np.ascontiguousarray(kp, np.float64).reshape(-1, 2)
    n = len(kp)
    lo = np.zeros(n, np.int64) if dlo is None else np.ascontiguousarray(dlo, np.int64)
    hi = np.full(n, MAX_DISP - 1, np.int64) if dhi is None else np.ascontiguousarray(dhi, np.int64)
    return (np.ascontiguousarray(left, np.uint8), np.ascontiguousarray(right, np.uint8), kp, lo, hi)


def disparities(left, right, kp, dlo=None, dhi=None):
    """kp (N,2) -> (d, sigma, ok, zncc). dlo/dhi default to the full range (as stereo1d.disparities)."""
    L, R, kp, lo, hi = _prep(left, right, kp, dlo, dhi)
    n = len(kp)
    d = np.zeros(n)
    s = np.full(n, np.inf)
    ok = np.zeros(n, np.bool_)
    zn = np.zeros(n)
    amb = np.zeros(n, np.bool_)
    span = np.zeros(n)
    if n:
        stereo_points(L, R, kp, lo, hi, d, s, ok, zn, GAP_ALPHA, amb, span)
    disparities.last_amb = amb
    disparities.last_span = span
    return d, s, ok, zn


def disparities_ex(left, right, kp, dlo=None, dhi=None):
    """-> (d, sigma, ok, amb, span, zncc); amb is all False and span 0 in version 2."""
    d, s, ok, zn = disparities(left, right, kp, dlo, dhi)
    return d, s, ok, disparities.last_amb, disparities.last_span, zn


def disparities_diag(left, right, kp, dlo=None, dhi=None):
    """Bench path: -> (d, sigma, ok, zncc, diag)."""
    L, R, kp, lo, hi = _prep(left, right, kp, dlo, dhi)
    n = len(kp)
    d = np.zeros(n)
    s = np.full(n, np.inf)
    ok = np.zeros(n, np.bool_)
    zn = np.zeros(n)
    dg = np.zeros((n, NDIAG))
    if n:
        stereo_points_diag(L, R, kp, lo, hi, d, s, ok, zn, dg)
    return d, s, ok, zn, dg
