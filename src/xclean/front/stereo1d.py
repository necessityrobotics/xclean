#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1 1-D STEREO on the CPU (numba): the disparity of each requested left keypoint on the rectified pair.

    d, sig, ok, zn = stereo_points(left, right, kp, dlo, dhi)

The algorithm is `stereo/epipolar.py`'s, ported to numba for P1's CPU-only
frame process. P1's frame path holds no dense SGM, so the SGM cross-check becomes the classical left-right check.
  1. ZNCC of the 11x11 left patch along the right row for d in [dlo, dhi]. The full search is dlo = 0,
     dhi = MAX_DISP - 1; a matched landmark's search is seeded from its predicted disparity +- the gate.
  2. Accept the integer peak only if ZNCC >= ZNCC_MIN and it is unique. Uniqueness is SGM's rule on the
     dissimilarity 1 - zncc: the second best, more than 1 px away, exceeds the best x (1 + UNIQUENESS).
  3. LEFT-RIGHT CHECK: the right patch at the peak, searched back along the left row over the full range, must land
     within LR_THRESHOLD_PX of the keypoint.
  4. Sub-pixel: a parabola seed, then affine 1-D Lucas-Kanade on (d, gain, offset), 6 Gauss-Newton steps, bilinear right
     samples. It must converge within +-1 px of the seed, with gain in (0.5, 2).
     - The per-point sigma is the Gauss-Newton covariance: residual variance x (J^T J)^-1[0, 0].
     - XC-06 scales it by the Class-C kappa_d in the window.
  5. d >= 0.5 px, as the epipolar matcher (a smaller disparity is a bearing, not a depth).

Every constant is carried: HALF 5, ZNCC_MIN 0.5, LK_ITERS 6 (epipolar.py); MAX_DISP 128, UNIQUENESS 0.05,
LR_THRESHOLD_PX 1.0 (sgm32.py). Single-threaded and deterministic.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from xclean.stereo.sgm32 import MAX_DISP, UNIQUENESS, LR_THRESHOLD_PX
from xclean.stereo.epipolar import HALF, ZNCC_MIN, LK_ITERS

PW = 2 * HALF + 1
NPIX = PW * PW


@njit(cache=True, nogil=True)
def _patch_stats(img, u, v):
    s = 0.0
    s2 = 0.0
    for dy in range(-HALF, HALF + 1):
        for dx in range(-HALF, HALF + 1):
            x = float(img[v + dy, u + dx])
            s += x
            s2 += x * x
    m = s / NPIX
    var = s2 / NPIX - m * m
    return m, math.sqrt(var if var > 0.0 else 0.0)


@njit(cache=True, nogil=True)
def _zncc(T, tm, ts, img, x, v):
    """ZNCC of the template T (11x11 float, mean tm, std ts) against img's patch centred at (x, v)."""
    s = 0.0
    s2 = 0.0
    sx = 0.0
    k = 0
    for dy in range(-HALF, HALF + 1):
        for dx in range(-HALF, HALF + 1):
            r = float(img[v + dy, x + dx])
            s += r
            s2 += r * r
            sx += r * (T[k] - tm)
            k += 1
    m = s / NPIX
    var = s2 / NPIX - m * m
    sd = math.sqrt(var if var > 0.0 else 0.0)
    den = ts * sd * NPIX
    if den > 1e-6:
        return sx / den
    return -2.0


@njit(cache=True, nogil=True)
def _bil(img, x, y):
    H, W = img.shape
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    ax = x - x0
    ay = y - y0
    if x0 < 0:
        x0 = 0
    if x0 > W - 2:
        x0 = W - 2
    if y0 < 0:
        y0 = 0
    if y0 > H - 2:
        y0 = H - 2
    a = float(img[y0, x0])
    b = float(img[y0, x0 + 1])
    c = float(img[y0 + 1, x0])
    d = float(img[y0 + 1, x0 + 1])
    top = a + ax * (b - a)
    bot = c + ax * (d - c)
    val = top + ay * (bot - top)
    gx = (b - a) * (1.0 - ay) + (d - c) * ay
    return val, gx


@njit(cache=True, nogil=True)
def _lk(L, R, u, v, d0):
    """Affine 1-D LK: min sum (L(p) - (a R(p.x - d, p.y) + b))^2 over (d, a, b) from d0. -> (d, a, sigma, ok)."""
    T = np.empty(NPIX)
    px = np.empty(NPIX)
    py = np.empty(NPIX)
    k = 0
    for gy in range(-HALF, HALF + 1):
        for gx in range(-HALF, HALF + 1):
            px[k] = u + gx
            py[k] = v + gy
            T[k] = _bil(L, px[k], py[k])[0]
            k += 1
    d = d0
    a = 1.0
    b = 0.0
    A = np.zeros((3, 3))
    g = np.zeros(3)
    J = np.zeros(3)
    for _it in range(LK_ITERS):
        A[:, :] = 0.0
        g[:] = 0.0
        for k in range(NPIX):
            rv, rg = _bil(R, px[k] - d, py[k])
            r = T[k] - (a * rv + b)
            J[0] = -a * rg
            J[1] = rv
            J[2] = 1.0
            for i in range(3):
                g[i] += J[i] * r
                for j in range(3):
                    A[i, j] += J[i] * J[j]
        for i in range(3):
            A[i, i] += 1e-9
        delta = np.linalg.solve(A, g)
        d += delta[0]
        a += delta[1]
        b += delta[2]
        if not (math.isfinite(d) and math.isfinite(a)):
            return d, a, np.inf, False
    # Gauss-Newton covariance at the solution
    A[:, :] = 0.0
    s2 = 0.0
    for k in range(NPIX):
        rv, rg = _bil(R, px[k] - d, py[k])
        r = T[k] - (a * rv + b)
        s2 += r * r
        J[0] = -a * rg
        J[1] = rv
        J[2] = 1.0
        for i in range(3):
            for j in range(3):
                A[i, j] += J[i] * J[j]
    for i in range(3):
        A[i, i] += 1e-9
    s2 /= (NPIX - 3)
    Ai = np.linalg.inv(A)
    var = s2 * Ai[0, 0]
    sig = math.sqrt(var) if var > 0.0 else 0.0
    ok = math.isfinite(d) and abs(d - d0) <= 1.0 and a > 0.5 and a < 2.0
    return d, a, sig, ok


@njit(cache=True, nogil=True)
def stereo_points(L, R, kp, dlo, dhi, out_d, out_sig, out_ok, out_zn, gap_alpha, out_amb, out_span):
    """L, R (H,W) uint8 rectified; kp (N,2) float64; dlo/dhi (N,) int64 search bounds (inclusive, clipped here).
    Fills out_d (px, 0 where not ok), out_sig (px, inf where not ok), out_ok, out_zn (peak ZNCC)."""
    H, W = L.shape
    N = kp.shape[0]
    T = np.empty(NPIX)
    T2 = np.empty(NPIX)
    sc = np.empty(MAX_DISP)
    for n in range(N):
        out_ok[n] = False
        out_amb[n] = False
        out_span[n] = 0.0
        out_d[n] = 0.0
        out_sig[n] = np.inf
        out_zn[n] = -2.0
        u = kp[n, 0]
        v = kp[n, 1]
        ui = int(math.floor(u + 0.5))
        vi = int(math.floor(v + 0.5))
        if ui - HALF - 2 < 0 or ui + HALF + 2 >= W or vi - HALF - 2 < 0 or vi + HALF + 2 >= H:
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
            continue
        k = 0
        for dy in range(-HALF, HALF + 1):
            for dx in range(-HALF, HALF + 1):
                T[k] = float(L[vi + dy, ui + dx])
                k += 1
        tm, ts = _patch_stats(L, ui, vi)
        if ts <= 1e-6:
            continue
        best = -1
        bs = -2.0
        for d in range(lo, hi + 1):
            s = _zncc(T, tm, ts, R, ui - d, vi)
            sc[d] = s
            if s > bs:
                bs = s
                best = d
        out_zn[n] = bs
        if best < 0 or bs < ZNCC_MIN:
            continue
        sec = -2.0
        for d in range(lo, hi + 1):
            if abs(d - best) > 1 and sc[d] > sec:
                sec = sc[d]
        if not ((1.0 - sec) > (1.0 - bs) * (1.0 + UNIQUENESS)):
            continue
        # (P1-03b, a peak-significance test, was DROPPED: on real frames it cut acceptance to 4-22 % without lowering
        # the gross-error rate vs RAFT)
        # (P1-03c, the extreme-value-spacing distinctiveness rule, was DROPPED and REMOVED (a later design change): it caused
        # the 4s_120 VO regression 13.8 -> 114.9 m, blocked v2_03's init until frame 252 and was non-discriminative on
        # the held-out GT bench  gap_alpha stays in the signature, ignored.)
        # left-right check: the right patch at the peak, searched back along the LEFT row (full range)
        xr = ui - best
        k = 0
        for dy in range(-HALF, HALF + 1):
            for dx in range(-HALF, HALF + 1):
                T2[k] = float(R[vi + dy, xr + dx])
                k += 1
        rm, rs = _patch_stats(R, xr, vi)
        if rs <= 1e-6:
            continue
        bb = -1
        bbs = -2.0
        dmax_rl = MAX_DISP - 1
        if xr + dmax_rl + HALF >= W:
            dmax_rl = W - 1 - HALF - xr
        for d in range(0, dmax_rl + 1):
            s = _zncc(T2, rm, rs, L, xr + d, vi)
            if s > bbs:
                bbs = s
                bb = d
        if bb < 0 or abs(bb - best) > LR_THRESHOLD_PX:
            continue
        # parabola seed, then LK from the sub-pixel keypoint position
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
        seed = best + off + (u - ui)
        dl, ga, sig, conv = _lk(L, R, u, v, seed)
        if not conv or dl < 0.5:
            continue
        out_d[n] = dl
        out_sig[n] = sig
        out_ok[n] = True
    return 0


@njit(cache=True, nogil=True)
def _lk_v(L, R, u, v, d0, voff):
    """_lk with the right image sampled at row offset voff (P1.1 D4: the right row of a left row v is v + voff).
    voff = 0 gives _lk's numbers exactly."""
    T = np.empty(NPIX)
    px = np.empty(NPIX)
    py = np.empty(NPIX)
    k = 0
    for gy in range(-HALF, HALF + 1):
        for gx in range(-HALF, HALF + 1):
            px[k] = u + gx
            py[k] = v + gy
            T[k] = _bil(L, px[k], py[k])[0]
            k += 1
    d = d0
    a = 1.0
    b = 0.0
    A = np.zeros((3, 3))
    g = np.zeros(3)
    J = np.zeros(3)
    for _it in range(LK_ITERS):
        A[:, :] = 0.0
        g[:] = 0.0
        for k in range(NPIX):
            rv, rg = _bil(R, px[k] - d, py[k] + voff)
            r = T[k] - (a * rv + b)
            J[0] = -a * rg
            J[1] = rv
            J[2] = 1.0
            for i in range(3):
                g[i] += J[i] * r
                for j in range(3):
                    A[i, j] += J[i] * J[j]
        for i in range(3):
            A[i, i] += 1e-9
        delta = np.linalg.solve(A, g)
        d += delta[0]
        a += delta[1]
        b += delta[2]
        if not (math.isfinite(d) and math.isfinite(a)):
            return d, a, np.inf, False
    A[:, :] = 0.0
    s2 = 0.0
    for k in range(NPIX):
        rv, rg = _bil(R, px[k] - d, py[k] + voff)
        r = T[k] - (a * rv + b)
        s2 += r * r
        J[0] = -a * rg
        J[1] = rv
        J[2] = 1.0
        for i in range(3):
            for j in range(3):
                A[i, j] += J[i] * J[j]
    for i in range(3):
        A[i, i] += 1e-9
    s2 /= (NPIX - 3)
    Ai = np.linalg.inv(A)
    var = s2 * Ai[0, 0]
    sig = math.sqrt(var) if var > 0.0 else 0.0
    ok = math.isfinite(d) and abs(d - d0) <= 1.0 and a > 0.5 and a < 2.0
    return d, a, sig, ok


@njit(cache=True, nogil=True)
def stereo_points_band(L, R, kp, dlo, dhi, out_d, out_sig, out_ok, out_zn, gap_alpha, out_amb, out_span, band,
                       out_dy):
    """stereo_points with P1.1 D4: the ZNCC search covers the right rows v + j, j in [-band, band] (R is the right image
    already shifted by the class-C vertical residual, so j = 0 is the estimated row), every disparity scoring its best
    row; the left-right check and the LK use the winning row.  out_dy (NaN unless ok): the point's own vertical offset
    in R, the sub-pixel peak of the ZNCC column scan j in [-HALF, HALF] at the matched disparity (the patch's own
    half-size: beyond it the patches no longer overlap; class A/B), for the class-C estimator.  band = 0 reproduces
    stereo_points' d / sigma / ok exactly.  out_amb / out_span stay False / 0 (P1-03c, P1-03d removed)."""
    H, W = L.shape
    N = kp.shape[0]
    T = np.empty(NPIX)
    T2 = np.empty(NPIX)
    sc = np.empty(MAX_DISP)
    jb = np.zeros(MAX_DISP, np.int64)
    col = np.empty(2 * HALF + 1)
    for n in range(N):
        out_ok[n] = False
        out_amb[n] = False
        out_span[n] = 0.0
        out_d[n] = 0.0
        out_sig[n] = np.inf
        out_zn[n] = -2.0
        out_dy[n] = np.nan
        u = kp[n, 0]
        v = kp[n, 1]
        ui = int(math.floor(u + 0.5))
        vi = int(math.floor(v + 0.5))
        if ui - HALF - 2 < 0 or ui + HALF + 2 >= W or vi - HALF - 2 - band < 0 or vi + HALF + 2 + band >= H:
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
            continue
        k = 0
        for dy in range(-HALF, HALF + 1):
            for dx in range(-HALF, HALF + 1):
                T[k] = float(L[vi + dy, ui + dx])
                k += 1
        tm, ts = _patch_stats(L, ui, vi)
        if ts <= 1e-6:
            continue
        best = -1
        bs = -2.0
        for d in range(lo, hi + 1):
            s = _zncc(T, tm, ts, R, ui - d, vi)
            jj = 0
            for j in range(1, band + 1):
                s1 = _zncc(T, tm, ts, R, ui - d, vi - j)
                if s1 > s:
                    s = s1
                    jj = -j
                s1 = _zncc(T, tm, ts, R, ui - d, vi + j)
                if s1 > s:
                    s = s1
                    jj = j
            sc[d] = s
            jb[d] = jj
            if s > bs:
                bs = s
                best = d
        out_zn[n] = bs
        if best < 0 or bs < ZNCC_MIN:
            continue
        sec = -2.0
        for d in range(lo, hi + 1):
            if abs(d - best) > 1 and sc[d] > sec:
                sec = sc[d]
        if not ((1.0 - sec) > (1.0 - bs) * (1.0 + UNIQUENESS)):
            continue
        # (P1-03c removed by a later design change; gap_alpha ignored)
        js = jb[best]
        xr = ui - best
        k = 0
        for dy in range(-HALF, HALF + 1):
            for dx in range(-HALF, HALF + 1):
                T2[k] = float(R[vi + js + dy, xr + dx])
                k += 1
        rm, rs = _patch_stats(R, xr, vi + js)
        if rs <= 1e-6:
            continue
        bb = -1
        bbs = -2.0
        dmax_rl = MAX_DISP - 1
        if xr + dmax_rl + HALF >= W:
            dmax_rl = W - 1 - HALF - xr
        for d in range(0, dmax_rl + 1):
            s = _zncc(T2, rm, rs, L, xr + d, vi)
            if s > bbs:
                bbs = s
                bb = d
        if bb < 0 or abs(bb - best) > LR_THRESHOLD_PX:
            continue
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
        seed = best + off + (u - ui)
        dl, ga, sig, conv = _lk_v(L, R, u, v, seed, float(js))
        if not conv or dl < 0.5:
            continue
        out_d[n] = dl
        out_sig[n] = sig
        out_ok[n] = True
        # the point's own vertical offset (column scan at the matched integer disparity)
        if vi - 2 * HALF - 1 >= 0 and vi + 2 * HALF + 1 < H:
            jp = 0
            cs = -2.0
            for j in range(-HALF, HALF + 1):
                col[j + HALF] = _zncc(T, tm, ts, R, xr, vi + j)
                if col[j + HALF] > cs:
                    cs = col[j + HALF]
                    jp = j
            offy = 0.0
            if jp > -HALF and jp < HALF:
                y0 = col[jp - 1 + HALF]
                y2 = col[jp + 1 + HALF]
                den = y0 - 2.0 * cs + y2
                if den < -1e-9:
                    offy = 0.5 * (y0 - y2) / den
                    if offy > 0.5:
                        offy = 0.5
                    if offy < -0.5:
                        offy = -0.5
            out_dy[n] = jp + offy
    return 0


GAP_ALPHA = 0.0                # P1-03c removed (a later design change): kept as a name for API compatibility


def shift_rows(right, dy):
    """The right image resampled at rows y + dy (linear in y, float32; rows clamped at the borders): the D4 frame in
    which the estimated vertical rectification residual is 0.  dy = 0 returns the image unchanged."""
    if not dy:
        return right
    H = right.shape[0]
    y = np.arange(H, dtype=np.float64) + float(dy)
    y0 = np.floor(y).astype(np.int64)
    f = (y - y0).astype(np.float32)[:, None]
    a = right[np.clip(y0, 0, H - 1)].astype(np.float32)
    b = right[np.clip(y0 + 1, 0, H - 1)].astype(np.float32)
    return np.ascontiguousarray(a + f * (b - a))


def disparities(left, right, kp, dlo=None, dhi=None, dy=0.0, band=0):
    """Convenience wrapper: kp (N,2) -> (d, sigma, ok, zncc).  dlo/dhi default to the full range.  dy / band (P1.1 D4):
    the class-C vertical residual and the row band; dy = 0 and band = 0 give the pre-P1.1 matcher's numbers.  Also sets
    disparities.last_dy (per-point vertical offset in the ORIGINAL right image, NaN unless ok)."""
    kp = np.ascontiguousarray(kp, np.float64).reshape(-1, 2)
    n = len(kp)
    lo = np.zeros(n, np.int64) if dlo is None else np.ascontiguousarray(dlo, np.int64)
    hi = np.full(n, MAX_DISP - 1, np.int64) if dhi is None else np.ascontiguousarray(dhi, np.int64)
    d = np.zeros(n)
    s = np.full(n, np.inf)
    ok = np.zeros(n, np.bool_)
    zn = np.zeros(n)
    amb = np.zeros(n, np.bool_)
    span = np.zeros(n)
    ody = np.full(n, np.nan)
    if n:
        # non-finite or off-image coordinates get the border-reject result without reaching the kernel's int
        # conversion (undefined for inf / huge values; the GPU twin gpu_front.disparities applies the same rule)
        H, W = left.shape
        bad = ~(np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1]) & (kp[:, 0] >= 0) & (kp[:, 0] < W)
                & (kp[:, 1] >= 0) & (kp[:, 1] < H))
        if bad.any():
            kp = kp.copy()
            kp[bad] = -float(HALF + 3)
        Rs = shift_rows(np.ascontiguousarray(right, np.uint8), dy)
        stereo_points_band(np.ascontiguousarray(left, np.uint8), Rs, kp, lo, hi, d, s, ok, zn, GAP_ALPHA, amb, span,
                           int(band), ody)
    disparities.last_amb = amb
    disparities.last_span = span
    disparities.last_dy = ody + float(dy)
    return d, s, ok, zn


def disparities_ex(left, right, kp, dlo=None, dhi=None, dy=0.0, band=0):
    """-> (d, sigma, ok, amb, span, zncc).  amb is all False and span 0 since P1-03c / P1-03d were removed (kept for
    the API)."""
    d, s, ok, zn = disparities(left, right, kp, dlo, dhi, dy, band)
    return d, s, ok, disparities.last_amb, disparities.last_span, zn
