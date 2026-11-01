#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""PER-KEYPOINT EPIPOLAR STEREO - the keypoint's disparity MEASUREMENT, cross-checked against the dense SGM.

    em = EpipolarMatcher()
    d, ok, info = em.match(left, right, kp, d_sgm)     # d float64 (0 where not ok)

WHY A SECOND MATCHER
    The dense SGM is PIXEL-LOCKED (tests/test_xc_sgm32.py: -0.21 px at a true 7.25, -0.31 px at 20.4) - the known
    behaviour of a parabola on aggregated costs - and that is a 7-10 % depth error at 3 px of disparity, i.e. on
    exactly the far points that carry rotation and, on KITTI, the scale.  the engine read depth only at keypoints
    (tracker._create), so the keypoint is where the measurement must be good.  This matcher measures it there:

      1. FULL EPIPOLAR ROW SEARCH (rectified pair: the epipolar line is the row): ZNCC of an 11x11 patch at every
         disparity d in [0, min(127, u)] - on the GPU, one block per keypoint, one thread per disparity.  It does
         NOT start from the SGM value, so it is an independent second opinion, not a refinement of the first.
      2. SUB-PIXEL by affine 1-D Lucas-Kanade on the raw intensities (unknowns: disparity, gain, offset; bilinear
         right image; Gauss-Newton from the integer peak), at the keypoint's own sub-pixel position.
      3. THE CROSS-CHECK: accepted only where the ZNCC peak is >= ZNCC_MIN, the peak is unique by the SAME rule
         SGM uses (second-best non-adjacent dissimilarity > best x (1 + 5 %)), LK converged inside +-1 px of the
         integer peak, the SGM disparity at the keypoint is VALID, and the two agree within the SGM's own LR
         threshold (1 px).  Two independent matchers that agree is the measurement; anything else is "no
         measurement" (0), never a guess.

    Constants: ZNCC_MIN = 0.5 and the 11x11 patch are the engine's own (_kp_refine_disparity, Class D carried);
    the uniqueness ratio and the agreement threshold are sgm32's (UNIQUENESS, LR_THRESHOLD_PX).  Nothing new.
"""
from __future__ import annotations

import numpy as np

from xclean.stereo.sgm32 import MAX_DISP, UNIQUENESS, LR_THRESHOLD_PX

HALF = 5                     # 11x11 patch
# D2 (the keypoint's own right-to-left check where SGM has no value) was DROPPED: against the RAFT cache
# its accepted points are poor (median |err| 0.55-1.71 px, only 43-70 % within 1 px on mh_04 / kitti_00 / kitti_01:
# SGM's holes sit on occlusion edges, where a two-way ZNCC agreement is not a depth).
import os as _os
D2_ON = False
if (_os.environ.get("XCLEAN_D2", "") or "").strip() not in ("", "0", "off"):
    raise RuntimeError("XCLEAN_D2 was removed from the engine; unset it")
ZNCC_MIN = 0.5               # the engine _kp_refine_disparity's confidence floor
LK_ITERS = 6

_KERNEL = r"""
#define MAX_DISP 128
#define HALF 5
#define PW 11
extern "C" __global__ void zncc_row(const unsigned char* __restrict__ L, const unsigned char* __restrict__ R,
                                    const int* __restrict__ ku, const int* __restrict__ kv,
                                    float* __restrict__ out, int n, int H, int W) {
    int k = blockIdx.x, d = threadIdx.x;
    if (k >= n) return;
    __shared__ float tp[PW * PW];
    __shared__ float stats[2];
    int u = ku[k], v = kv[k];
    for (int i = d; i < PW * PW; i += blockDim.x) {
        int dy = i / PW - HALF, dx = i % PW - HALF;
        tp[i] = (float)L[(v + dy) * W + (u + dx)];
    }
    __syncthreads();
    if (d == 0) {
        float s = 0.f, s2 = 0.f;
        for (int i = 0; i < PW * PW; i++) { s += tp[i]; s2 += tp[i] * tp[i]; }
        float m = s / (PW * PW);
        stats[0] = m; stats[1] = sqrtf(fmaxf(s2 / (PW * PW) - m * m, 0.f));
    }
    __syncthreads();
    float res = -2.0f;
    int x = u - d;
    if (d < MAX_DISP && x - HALF >= 0) {
        float s = 0.f, s2 = 0.f, sx = 0.f;
        for (int i = 0; i < PW * PW; i++) {
            int dy = i / PW - HALF, dx = i % PW - HALF;
            float r = (float)R[(v + dy) * W + (x + dx)];
            s += r; s2 += r * r; sx += r * (tp[i] - stats[0]);
        }
        float m = s / (PW * PW);
        float sd = sqrtf(fmaxf(s2 / (PW * PW) - m * m, 0.f));
        float den = stats[1] * sd * (PW * PW);
        res = (den > 1e-6f) ? sx / den : -2.0f;
    }
    out[k * MAX_DISP + d] = res;
}

/* the right-to-left search: template = the RIGHT patch at (u, v), candidates in the LEFT row at u + d */
extern "C" __global__ void zncc_row_rl(const unsigned char* __restrict__ L, const unsigned char* __restrict__ R,
                                       const int* __restrict__ ku, const int* __restrict__ kv,
                                       float* __restrict__ out, int n, int H, int W) {
    int k = blockIdx.x, d = threadIdx.x;
    if (k >= n) return;
    __shared__ float tp[PW * PW];
    __shared__ float stats[2];
    int u = ku[k], v = kv[k];
    for (int i = d; i < PW * PW; i += blockDim.x) {
        int dy = i / PW - HALF, dx = i % PW - HALF;
        tp[i] = (float)R[(v + dy) * W + (u + dx)];
    }
    __syncthreads();
    if (d == 0) {
        float s = 0.f, s2 = 0.f;
        for (int i = 0; i < PW * PW; i++) { s += tp[i]; s2 += tp[i] * tp[i]; }
        float m = s / (PW * PW);
        stats[0] = m; stats[1] = sqrtf(fmaxf(s2 / (PW * PW) - m * m, 0.f));
    }
    __syncthreads();
    float res = -2.0f;
    int x = u + d;
    if (d < MAX_DISP && x + HALF < W) {
        float s = 0.f, s2 = 0.f, sx = 0.f;
        for (int i = 0; i < PW * PW; i++) {
            int dy = i / PW - HALF, dx = i % PW - HALF;
            float r = (float)L[(v + dy) * W + (x + dx)];
            s += r; s2 += r * r; sx += r * (tp[i] - stats[0]);
        }
        float m = s / (PW * PW);
        float sd = sqrtf(fmaxf(s2 / (PW * PW) - m * m, 0.f));
        float den = stats[1] * sd * (PW * PW);
        res = (den > 1e-6f) ? sx / den : -2.0f;
    }
    out[k * MAX_DISP + d] = res;
}
"""


def _bilinear(img, x, y):
    """img (H,W) float32, x/y float arrays (same shape) -> samples and x-gradient (central, on the bilinear)."""
    H, W = img.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    ax = x - x0
    ay = y - y0
    x0c = np.clip(x0, 0, W - 2)
    y0c = np.clip(y0, 0, H - 2)
    a = img[y0c, x0c]
    b = img[y0c, x0c + 1]
    c = img[y0c + 1, x0c]
    d = img[y0c + 1, x0c + 1]
    top = a + ax * (b - a)
    bot = c + ax * (d - c)
    val = top + ay * (bot - top)
    gx = (b - a) * (1 - ay) + (d - c) * ay
    return val, gx


class EpipolarMatcher:
    def __init__(self):
        import cupy as cp
        self.cp = cp
        mod = cp.RawModule(code=_KERNEL)
        self._k = mod.get_function("zncc_row")
        self._krl = mod.get_function("zncc_row_rl")
        self.n_calls = 0
        self.stats = {"kp": 0, "searched": 0, "zncc_ok": 0, "unique": 0, "lk_ok": 0, "sgm_valid": 0,
                      "agree": 0, "accepted": 0}

    def scores(self, left, right, ui, vi, rl=False):
        """(N, 128) ZNCC scores for integer keypoints (ui, vi) (caller guarantees the patch is inside).
        rl=False: left template, right candidates at u - d.  rl=True: right template at (ui, vi), left candidates at
        u + d (the right-to-left consistency search)."""
        cp = self.cp
        H, W = left.shape
        n = len(ui)
        if n == 0:
            return np.zeros((0, MAX_DISP), np.float32)
        Lg = cp.asarray(np.ascontiguousarray(left, np.uint8))
        Rg = cp.asarray(np.ascontiguousarray(right, np.uint8))
        out = cp.empty(n * MAX_DISP, cp.float32)
        kern = self._krl if rl else self._k
        kern((n,), (MAX_DISP,), (Lg, Rg, cp.asarray(ui.astype(np.int32)), cp.asarray(vi.astype(np.int32)),
                                  out, np.int32(n), np.int32(H), np.int32(W)))
        return cp.asnumpy(out).reshape(n, MAX_DISP)

    def lk_refine(self, left, right, u, v, d0):
        """Affine 1-D Lucas-Kanade: minimise sum (L(p) - (a R(p.x - d, p.y) + b))^2 over (d, a, b), from d0.
        -> (d, converged bool)."""
        Lf = left.astype(np.float32)
        Rf = right.astype(np.float32)
        n = len(u)
        g = np.arange(-HALF, HALF + 1, dtype=np.float64)
        gy, gx = np.meshgrid(g, g, indexing="ij")
        px = u[:, None] + gx.ravel()[None, :]                  # (n, 121) sub-pixel left sample positions
        py = v[:, None] + gy.ravel()[None, :]
        T, _ = _bilinear(Lf, px, py)
        d = d0.astype(np.float64).copy()
        a = np.ones(n)
        b = np.zeros(n)
        for _ in range(LK_ITERS):
            Rv, Rg = _bilinear(Rf, px - d[:, None], py)
            r = T - (a[:, None] * Rv + b[:, None])
            # d r / d(d) = +a * dR/dx ; d r / da = -R ; d r / db = -1   (Gauss-Newton on -J)
            J = np.stack([-a[:, None] * Rg, Rv, np.ones_like(Rv)], axis=2)     # model derivative (n,121,3)
            A = np.einsum("nki,nkj->nij", J, J)
            g_ = np.einsum("nki,nk->ni", J, r)
            A[:, [0, 1, 2], [0, 1, 2]] += 1e-9
            try:
                delta = np.linalg.solve(A, g_[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                delta = np.zeros((n, 3))
            # model m = a R(x - d) + b ; dm/dd = -a Rg -> r = T - m ; step: J delta = r
            d = d + delta[:, 0]
            a = a + delta[:, 1]
            b = b + delta[:, 2]
        conv = np.isfinite(d) & (np.abs(d - d0) <= 1.0) & (a > 0.5) & (a < 2.0)
        # the MEASURED uncertainty of this disparity: the Gauss-Newton covariance at the solution, residual variance
        # x (J^T J)^-1 (121 samples, 3 unknowns) - per keypoint, from its own texture and its own fit residual
        Rv, Rg = _bilinear(Rf, px - d[:, None], py)
        r = T - (a[:, None] * Rv + b[:, None])
        J = np.stack([-a[:, None] * Rg, Rv, np.ones_like(Rv)], axis=2)
        A = np.einsum("nki,nkj->nij", J, J)
        A[:, [0, 1, 2], [0, 1, 2]] += 1e-9
        s2 = (r * r).sum(axis=1) / max(r.shape[1] - 3, 1)
        try:
            Ainv = np.linalg.inv(A)
            sig = np.sqrt(np.maximum(s2 * Ainv[:, 0, 0], 0.0))
        except np.linalg.LinAlgError:
            sig = np.full(n, np.inf)
        self.last_sigma = np.where(np.isfinite(sig), sig, np.inf)
        return d, conv

    def match(self, left, right, kp, d_sgm):
        """kp (N,2) sub-pixel keypoints, d_sgm (N,) the dense SGM sample at each (0 = invalid).
        -> (d (N,) float64 with 0 where not accepted, ok (N,) bool, info dict)."""
        H, W = left.shape[:2]
        N = len(kp)
        out = np.zeros(N)
        ok = np.zeros(N, bool)
        self.out_sigma = np.full(N, np.inf)
        st = self.stats
        st["kp"] += N
        self.n_calls += 1
        if N == 0:
            return out, ok, {}
        u = kp[:, 0].astype(np.float64)
        v = kp[:, 1].astype(np.float64)
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        # the LK samples reach HALF + 1 beyond the patch centre; keep the whole support inside
        inside = (ui - HALF - 2 >= 0) & (ui + HALF + 2 < W) & (vi - HALF - 2 >= 0) & (vi + HALF + 2 < H)
        idx = np.nonzero(inside)[0]
        st["searched"] += len(idx)
        if not len(idx):
            return out, ok, {}
        S = self.scores(left, right, ui[idx], vi[idx])
        best = S.argmax(axis=1)
        r_ = np.arange(len(idx))
        bs = S[r_, best]
        dgrid = np.arange(MAX_DISP)[None, :]
        far = np.abs(dgrid - best[:, None]) > 1
        sec = np.where(far, S, -2.0).max(axis=1)
        zok = bs >= ZNCC_MIN
        # uniqueness, SGM's rule on the dissimilarity 1 - zncc: second > best x (1 + UNIQUENESS)
        uniq = (1.0 - sec) > (1.0 - bs) * (1.0 + UNIQUENESS)
        st["zncc_ok"] += int(zok.sum())
        st["unique"] += int((zok & uniq).sum())
        # parabola seed on the ZNCC curve, then LK
        y0 = S[r_, np.maximum(best - 1, 0)]
        y2 = S[r_, np.minimum(best + 1, MAX_DISP - 1)]
        den = y0 - 2 * bs + y2
        with np.errstate(divide="ignore", invalid="ignore"):
            off = np.where((den < -1e-9) & (best > 0) & (best < MAX_DISP - 1), 0.5 * (y0 - y2) / den, 0.0)
        seed = best + np.clip(off, -0.5, 0.5) + (u[idx] - ui[idx])     # the sub-pixel keypoint shifts d too
        dl, conv = self.lk_refine(left, right, u[idx], v[idx], seed)
        sig_lk = self.last_sigma
        st["lk_ok"] += int((zok & uniq & conv).sum())
        ds = np.asarray(d_sgm, np.float64)[idx]
        sv = ds > 0
        st["sgm_valid"] += int((zok & uniq & conv & sv).sum())
        agree = np.abs(dl - ds) <= LR_THRESHOLD_PX
        base = zok & uniq & conv & (dl >= 0.5)
        acc_sgm = base & sv & agree
        st["agree"] += int((zok & uniq & conv & sv & agree).sum())
        acc_rl = np.zeros_like(base)          # D2 (keypoint right-to-left where SGM has no value) DROPPED: C24
        acc = acc_sgm | acc_rl
        st["accepted"] += int(acc.sum())
        out[idx[acc]] = dl[acc]
        ok[idx[acc]] = True
        self.out_sigma[idx[acc]] = sig_lk[acc]
        return out, ok, {"zncc": bs, "idx": idx, "d_lk": dl, "conv": conv, "unique": uniq, "via_rl": acc_rl}


def sgm_quad_sample(kp, disp, edge_px):
    """Bilinear SGM sample at sub-pixel keypoints with xclean's invalid rule (any invalid neighbour = 0) and the engine's
    occlusion-edge test (quad spread > edge_px).  -> (d, quad_valid, on_edge)."""
    h, w = disp.shape
    uf = np.clip(kp[:, 0], 0, w - 1.001)
    vf = np.clip(kp[:, 1], 0, h - 1.001)
    u0 = uf.astype(np.int32)
    v0 = vf.astype(np.int32)
    u1 = np.minimum(u0 + 1, w - 1)
    v1 = np.minimum(v0 + 1, h - 1)
    au = uf - u0
    av = vf - v0
    d00, d01, d10, d11 = disp[v0, u0], disp[v0, u1], disp[v1, u0], disp[v1, u1]
    d = (1 - av) * ((1 - au) * d00 + au * d01) + av * ((1 - au) * d10 + au * d11)
    quad = np.stack([d00, d01, d10, d11], axis=1)
    quad_valid = (quad > 0).all(axis=1)
    d = np.where(quad_valid, d, 0.0)
    on_edge = quad_valid & ((quad.max(axis=1) - quad.min(axis=1)) > edge_px)
    return d, quad_valid, on_edge


def keypoint_disparity(kp, disp, left, right, matcher, edge_px, mode="epipolar", sel=None, sgm_min_disp=None):
    """THE keypoint disparity measurement (tracker._create and the loop worker's rich keypoints use this one function)
    -> (d (N,), on_edge (N,), quad_valid (N,), n_epi_ok); d = 0 = no measurement.

    mode "epipolar": the keypoint ZNCC+LK value where it agrees with SGM (else nothing).
    mode "sgm":      the SGM sample (quad rule + edge test).
    mode "hybrid": keypoint ZNCC+LK where it agrees with SGM; where the keypoint matcher has NO
                     valid result (no confident / unique peak, LK not converged) the SGM sample, but only at
                     d >= sgm_min_disp (the band where SGM's declared noise is <= 10 % of the disparity - SGM is
                     pixel-locked, so never in the small-disparity band); a keypoint where both are valid but disagree
                     gets nothing (measured: then each is right only ~half the time)."""
    d, quad_valid, on_edge = sgm_quad_sample(kp, disp, edge_px)
    n_ok = 0
    sigma = None
    if mode in ("epipolar", "hybrid") and left is not None and right is not None and len(kp):
        lg = left if left.ndim == 2 else left.mean(axis=2).astype(np.uint8)
        rg = right if right.ndim == 2 else right.mean(axis=2).astype(np.uint8)
        idx = np.arange(len(kp)) if sel is None else np.nonzero(sel)[0]
        d_epi, ok_epi, _inf = matcher.match(lg, rg, kp[idx], np.where(on_edge, 0.0, d)[idx])
        n_ok = int(ok_epi.sum())
        d_sgm = np.where(on_edge, 0.0, d)
        d = d.copy()
        d[idx] = np.where(ok_epi, d_epi, 0.0)
        if mode == "hybrid":
            # keypoints the matcher could not measure at all (not the ones where it disagreed with SGM)
            no_result = np.ones(len(idx), bool)
            inf_idx = _inf.get("idx") if isinstance(_inf, dict) else None
            if inf_idx is not None and len(inf_idx):
                valid = (_inf["zncc"] >= ZNCC_MIN) & _inf["unique"] & _inf["conv"]
                no_result[inf_idx[valid]] = False
            dmin = 0.0 if sgm_min_disp is None else float(sgm_min_disp)
            fill = no_result & ~ok_epi & (d_sgm[idx] >= max(dmin, 0.5))
            d[idx[fill]] = d_sgm[idx][fill]
            matcher.stats["hybrid_sgm_fill"] = matcher.stats.get("hybrid_sgm_fill", 0) + int(fill.sum())
        sigma = np.full(len(kp), np.inf)
        sigma[idx] = matcher.out_sigma
    keypoint_disparity.last_sigma = sigma        # per keypoint LK sigma (px), inf where not measured; None in sgm mode
    return d, on_edge, quad_valid, n_ok
