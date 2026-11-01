#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1 GPU-RESIDENT FRONT END (CuPy RawKernels): the frame's left image goes up ONCE, detection + ORB description run on
the device and only keypoints + descriptors come back; at keyframes the right image goes up once and the 1-D stereo
returns only per-point disparities; stage-A Hamming matching reuses the frame's descriptors already on the device.

SAME APIs AS THE CPU MODULES (the engine swaps them by a flag; nothing in the engine's logic changes):

    det = Detector(rig)                          # = front/detect.Detector: __call__, report_yield, budget, stats, fe
    kp, desc = det(gray)                         # (N,2) float32, (N,32) uint8: BIT-IDENTICAL to the CPU Detector
    d, sig, ok, zn = disparities(left, right, kp, dlo=None, dhi=None, dy=0.0, band=0)     # = front/stereo1d
    d, sig, ok, amb, span, zn = disparities_ex(...);  disparities.last_dy / last_amb / last_span
    i0, i1 = det.fe.match_desc(f0, f1)           # = Frontend.match_desc (stage A), identical arrays
    warmup(rig)                                  # compile/load kernels, calibrate + self-check against cv2/numpy

START-UP SELF-CHECK POLICY (a user never gets a refusal to start; XCLEAN_STRICT_SELFCHECK=1 = every failed check raises)
  * Only the 3x3 solver check differs (check_lk_solver; e.g. SciPy built on Intel MKL instead of OpenBLAS): the device
    solver is a correct LU solve whose results differ from this machine's CPU LAPACK at rounding level only.  The device
    front stays on, calib records lk_solver_equal False + lk_solver_mismatch, and one note is printed.
  * Any other failed check, or a failure to build the front at all: the check raises, and the P1 engine
    (vio/engine_live.py) falls back to the CPU front for the whole run (release_device() first drops the device front
    and frees CuPy's cached GPU memory; a CUDA context that already exists is kept, see release_device), prints one
    warning (higher latency) and records the reason in its summary.
  * Every check passes: nothing of the above runs; the code path and the results are unchanged.

WHAT IS REPRODUCED, AND HOW (every step is checked against the CPU code at warmup(); a failed check raises, see above)
  * Shi-Tomasi response = cv2.cornerMinEigenVal(gray, 3, 3) BIT FOR BIT: the Sobel pair as OpenCV's separable filter
    computes it (row pass, column pass; fused multiply-add exactly where OpenCV's SIMD body uses it, plain multiply-add
    in its scalar tail: the boundary is CALIBRATED at warmup against cv2 on a random image of the frame size), the 3x3
    box sum in float64 with OpenCV's running column sum (the same rounding sequence), the min-eigenvalue in float32.
    Subnormals are kept (the kernels are compiled WITHOUT CuPy's forced -ftz=true).
  * The cell scope (XC-03): 3x3 dilation, the per-cell max of the zero-padded response, the 0.01 quality in float32,
    the Immerkaer noise floor with numpy's own float32 PAIRWISE summation order (the recursion tree is built on the
    host once per image size) - so the candidate mask equals Frontend._response's.
  * Greedy selection (strongest first, one per cell, min distance r, the strict '<'): an exact parallel form. Each cell
    holds its strongest unskipped candidate; a cell commits when no neighbouring cell's current candidate outranks it,
    and skips a candidate within r of a committed higher-ranked pick. This yields exactly the sequential greedy set
    (the globally highest current candidate is always ready, decisions never change); the first n_want picks in rank
    order are the sequential loop's picks. Rank = (response desc, y asc, x asc) = np.lexsort((xs, ys, -resp)).
  * Sub-pixel parabola in float64 (Frontend._subpix's expression order).
  * ORB (no orientation) = cv2 ORB.compute on the reflect-101 padded image BIT FOR BIT: the Gaussian 7x7 sigma 2 blur as
    OpenCV's float separable path computes it for a sub-matrix (calibrated like the response), and the 256 test pairs
    RECOVERED AT RUNTIME by probing cv2's ORB with line / point images (no table is copied into this source; the pattern
    is a property of the OpenCV build we call, Apache-2.0). Verified at warmup on random keypoints.
  * 1-D stereo = stereo1d.disparities (P1.1 D4 band + dy shift included): ZNCC, uniqueness, left-right check, column
    scan in float64 with the CPU's summation order (FMA contraction off) -> identical integer peaks and ZNCC values;
    the affine LK sums its 121 terms sequentially in the CPU's pixel order and solves / inverts its 3x3 systems with
    the arithmetic of the LAPACK numba calls on this build (OpenBLAS getf2 / getrs / trti2 / dgetri, fused
    multiply-adds where its kernels use them; check_lk_solver verifies it at warmup) -> d, sigma and the accept flags
    BIT-IDENTICAL.  (A first version reduced the sums as a tree and used a textbook LU: d / sigma differed at 1e-12,
    which flipped 4 exact-tie accept flags in 1.8 M points and, through the landmark depths, made lock-step
    trajectories diverge from the CPU's.)
  * Hamming (stage A): integer, exact; ties as cv2.BFMatcher (first minimum, second = second smallest).

DATA PATH: one CUDA stream, one context; every device and
pinned buffer is allocated once in GpuFront.__init__ from a private memory pool (no allocation per call; the tests
assert the pool does not grow). Per frame: H2D left (pinned), ~14 kernel launches, ONE D2H of a packed result
(count, floor, keypoints, descriptors), one stream sync. Stereo: H2D right once per frame (only on frames that call
stereo), H2D of the query points, one D2H. Hamming: H2D of the landmark descriptors, one D2H.

Constants: none new. Kernel block sizes are class B (MAX_DISP = 128 threads per stereo block, warp multiples
elsewhere); buffer capacities are data-structure sizes (H x W candidates, cap keypoints) that change no result.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading

import numpy as np

from xclean.stereo.sgm32 import MAX_DISP, UNIQUENESS, LR_THRESHOLD_PX
from xclean.stereo.epipolar import HALF, ZNCC_MIN, LK_ITERS

from xclean.stages.features import ORB_PAD, ORB_PATCH   # carried: the engine's ORB geometry (pad 32, patch 31)
POOL_LIMIT_BYTES = 1 << 30           # class B: private pool cap under the live benchmark rules B6 (VRAM <= 2 GB incl. the context);
                                     # actual use ~0.1 GB; sized once at construction (K-01b-RELAX)
STEREO_CAP = 4096                    # data-structure size: points per stereo launch (larger calls are chunked)
LM_CAP = 16384                       # data-structure size: landmark descriptors per Hamming launch (a larger call, or
                                     # more query keypoints than the descriptor buffer holds, runs on the numba matcher:
                                     # identical arrays)

_SRC = r"""
typedef unsigned char u8;
typedef long long i64;
typedef unsigned long long u64;
#define CUDART_INF_F __int_as_float(0x7f800000)
#define CUDART_INF __longlong_as_double(0x7ff0000000000000LL)
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000LL)
#define REFL(i, n) ((i) < 0 ? -(i) : ((i) >= (n) ? 2 * (n) - 2 - (i) : (i)))
#define HALF 5
#define PW 11
#define NPIX 121
#define MAXD 128

/* ======================================= 1. Shi-Tomasi response ======================================= */
extern "C" __global__ void k_sobel(const u8* I, int H, int W, float ks, float k2s, int dy_vend, int dx_vend,
                                   float* DX, float* DY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x >= W) return;
    int xm = REFL(x - 1, W), xp = REFL(x + 1, W);
    float rx[3], ry[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        int yy = REFL(y - 1 + j, H);
        const u8* row = I + (size_t)yy * W;
        float a = (float)row[xm], b = (float)row[x], c = (float)row[xp];
        rx[j] = __fsub_rn(c, a);
        if (x < dy_vend) {            /* OpenCV RowVec_8u32f body: v_muladd chain from zero */
            float s = __fmul_rn(a, ks); s = __fmaf_rn(b, k2s, s); s = __fmaf_rn(c, ks, s); ry[j] = s;
        } else {                       /* RowFilter scalar tail */
            float s = __fmul_rn(ks, a); s = __fadd_rn(s, __fmul_rn(k2s, b)); s = __fadd_rn(s, __fmul_rn(ks, c));
            ry[j] = s;
        }
    }
    float dx;
    if (x < dx_vend) dx = __fmaf_rn(__fadd_rn(rx[0], rx[2]), ks, __fmul_rn(rx[1], k2s));
    else dx = __fadd_rn(__fmul_rn(__fadd_rn(rx[0], rx[2]), ks), __fmul_rn(rx[1], k2s));
    DX[(size_t)y * W + x] = dx;
    DY[(size_t)y * W + x] = __fsub_rn(ry[2], ry[0]);
}

__device__ __forceinline__ double rs_at(const float* DX, const float* DY, int W, int y, int xm, int x, int xp, int ch)
{
    const float* dxr = DX + (size_t)y * W;
    const float* dyr = DY + (size_t)y * W;
    float a0, a1, a2;
    if (ch == 0) {
        float u0 = dxr[xm], u1 = dxr[x], u2 = dxr[xp];
        a0 = __fmul_rn(u0, u0); a1 = __fmul_rn(u1, u1); a2 = __fmul_rn(u2, u2);
    } else if (ch == 1) {
        a0 = __fmul_rn(dxr[xm], dyr[xm]); a1 = __fmul_rn(dxr[x], dyr[x]); a2 = __fmul_rn(dxr[xp], dyr[xp]);
    } else {
        float u0 = dyr[xm], u1 = dyr[x], u2 = dyr[xp];
        a0 = __fmul_rn(u0, u0); a1 = __fmul_rn(u1, u1); a2 = __fmul_rn(u2, u2);
    }
    return __dadd_rn(__dadd_rn((double)a0, (double)a1), (double)a2);
}

/* OpenCV boxFilter(CV_32FC3, 3x3, normalize=false): RowSum in double (k_boxrows, parallel), then ColumnSum's
   running double sum per (column, channel) (k_boxcols: the same rounding sequence, rows prefetched 8 at a time) */
extern "C" __global__ void k_boxrows(const float* __restrict__ DX, const float* __restrict__ DY, int H, int W,
                                     double* __restrict__ RS)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x >= W) return;
    int xm = REFL(x - 1, W), xp = REFL(x + 1, W);
    size_t o = (size_t)y * W + x, n = (size_t)H * W;
#pragma unroll
    for (int ch = 0; ch < 3; ++ch) RS[ch * n + o] = rs_at(DX, DY, W, y, xm, x, xp, ch);
}

extern "C" __global__ void k_boxcols(const double* __restrict__ RS, int H, int W, float* __restrict__ COV)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 3 * W) return;
    int x = t % W, ch = t / W;
    const double* col = RS + (size_t)ch * H * W + x;
    double a = col[(size_t)REFL(-1, H) * W];
    double b = col[0];
    double S = __dadd_rn(__dadd_rn(0.0, a), b);
    int y = 0;
    for (; y + 8 <= H; y += 8) {
        double c[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) c[j] = col[(size_t)REFL(y + 1 + j, H) * W];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            double s0 = __dadd_rn(S, c[j]);
            COV[((size_t)(y + j) * W + x) * 3 + ch] = __double2float_rn(s0);
            S = __dsub_rn(s0, a);
            a = b; b = c[j];
        }
    }
    for (; y < H; ++y) {
        double c = col[(size_t)REFL(y + 1, H) * W];
        double s0 = __dadd_rn(S, c);
        COV[((size_t)y * W + x) * 3 + ch] = __double2float_rn(s0);
        S = __dsub_rn(s0, a);
        a = b; b = c;
    }
}

extern "C" __global__ void k_eig(const float* COV, int n, float* R, int* flags)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = 0;
    if (i < n) {
        float a = __fmul_rn(COV[3 * (size_t)i], 0.5f), b = COV[3 * (size_t)i + 1], c = __fmul_rn(COV[3 * (size_t)i + 2], 0.5f);
        float t = __fsub_rn(a, c);
        float v = __fadd_rn(__fmul_rn(t, t), __fmul_rn(b, b));
        float r = __fsub_rn(__fadd_rn(a, c), __fsqrt_rn(v));
        R[i] = r;
        pos = r > 0.0f;
    }
    if (__any_sync(0xffffffffu, pos) && (threadIdx.x & 31) == 0) atomicOr(flags, 1);
}

/* ============================ 2. noise floor: numpy float32 pairwise sum of |Immerkaer| ============================ */
__device__ __forceinline__ float lap_abs(const u8* I, int W, i64 k)
{
    int Wi = W - 2;
    int y = 1 + (int)(k / Wi), x = 1 + (int)(k % Wi);
    const u8* r0 = I + (size_t)(y - 1) * W + x;
    const u8* r1 = r0 + W;
    const u8* r2 = r1 + W;
    int v = (int)r0[-1] - 2 * (int)r0[0] + (int)r0[1] - 2 * (int)r1[-1] + 4 * (int)r1[0] - 2 * (int)r1[1]
          + (int)r2[-1] - 2 * (int)r2[0] + (int)r2[1];
    return (float)(v < 0 ? -v : v);
}

extern "C" __global__ void k_noise_leaves(const u8* I, int W, const i64* lstart, const int* llen, int nleaf,
                                          float* nodeval)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nleaf) return;
    i64 st = lstart[l];
    int n = llen[l];
    float res;
    if (n < 8) {
        res = -0.0f;
        for (int i = 0; i < n; ++i) res = __fadd_rn(res, lap_abs(I, W, st + i));
    } else {
        float r[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) r[j] = lap_abs(I, W, st + j);
        int i = 8;
        int lim = n - (n % 8);
        for (; i < lim; i += 8) {
#pragma unroll
            for (int j = 0; j < 8; ++j) r[j] = __fadd_rn(r[j], lap_abs(I, W, st + i + j));
        }
        res = __fadd_rn(__fadd_rn(__fadd_rn(r[0], r[1]), __fadd_rn(r[2], r[3])),
                        __fadd_rn(__fadd_rn(r[4], r[5]), __fadd_rn(r[6], r[7])));
        for (; i < n; ++i) res = __fadd_rn(res, lap_abs(I, W, st + i));
    }
    nodeval[l] = res;
}

extern "C" __global__ void k_noise_tree(float* nodeval, const int* left, const int* right, const int* lvl_start,
                                        int nlvl, int root, double K, double Q, double* floor_d, float* floor_f)
{
    for (int h = 0; h < nlvl; ++h) {
        for (int i = lvl_start[h] + threadIdx.x; i < lvl_start[h + 1]; i += blockDim.x)
            nodeval[i] = __fadd_rn(nodeval[left[i]], nodeval[right[i]]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float s = nodeval[root];
        double sn = __dmul_rn(K, (double)s);
        double fl = __ddiv_rn(__dmul_rn(__dmul_rn(Q, sn), sn), 65025.0);
        *floor_d = fl;
        *floor_f = __double2float_rn(fl);
    }
}

/* ============================ 3. candidates (cell scope) and the exact greedy selection ============================ */
extern "C" __global__ void k_cellmax(const float* R, int H, int W, int c, int gw, float q, float* thr)
{
    int cid = blockIdx.x;
    int gy = cid / gw, gx = cid % gw;
    int y0 = gy * c, x0 = gx * c;
    float m = -CUDART_INF_F;
    int pad = 0;
    for (int p = threadIdx.x; p < c * c; p += blockDim.x) {
        int y = y0 + p / c, x = x0 + p % c;
        if (y < H && x < W) m = fmaxf(m, R[(size_t)y * W + x]); else pad = 1;
    }
    __shared__ float sm[128];
    __shared__ int sp[128];
    sm[threadIdx.x] = m; sp[threadIdx.x] = pad;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) { sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + o]); sp[threadIdx.x] |= sp[threadIdx.x + o]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mm = sm[0];
        if (sp[0]) mm = fmaxf(mm, 0.0f);
        thr[cid] = __fmul_rn(q, mm);
    }
}

__device__ __forceinline__ double npy_floordiv(double a, double b)
{
    double mod = fmod(a, b);
    double div = __ddiv_rn(__dsub_rn(a, mod), b);
    if (mod != 0.0) {
        if ((b < 0) != (mod < 0)) { mod = __dadd_rn(mod, b); div = __dsub_rn(div, 1.0); }
    }
    double fd;
    if (div != 0.0) { fd = floor(div); if (__dsub_rn(div, fd) > 0.5) fd = __dadd_rn(fd, 1.0); }
    else fd = copysign(0.0, __ddiv_rn(a, b));
    return fd;
}

__device__ __forceinline__ unsigned int ord_f(float v)
{
    unsigned int u = __float_as_uint(v);
    if (u == 0x80000000u) u = 0u;                         /* -0 ranks as +0 (numpy compares them equal) */
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

extern "C" __global__ void k_cand(const float* R, int H, int W, int b, int c, int gwc, const float* thr,
                                  const float* floor_f, const int* flags, double s, int ncx, int* ncand,
                                  int* cx, int* cy, u64* ckey, int* ccell, int* cellcount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x < b || x >= W - b || y < b || y >= H - b) return;
    if (!flags[0]) return;                                 /* rmax <= 0: no candidates (Frontend._response) */
    const float* r = R + (size_t)y * W + x;
    float v = r[0];
    float m = v;
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy)
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) m = fmaxf(m, r[dy * W + dx]);
    if (!(v >= m)) return;
    if (!(v >= thr[(y / c) * gwc + x / c])) return;
    if (!(v >= floor_f[0])) return;
    int cyc = (int)npy_floordiv((double)y, s), cxc = (int)npy_floordiv((double)x, s);
    int cell = cyc * ncx + cxc;
    int k = atomicAdd(ncand, 1);
    cx[k] = x; cy[k] = y; ccell[k] = cell;
    ckey[k] = ((u64)ord_f(v) << 32) | ((u64)(0xFFFF - y) << 16) | (u64)(0xFFFF - x);
    atomicAdd(&cellcount[cell], 1);
}

/* single block: exclusive scan of the per-cell counts */
extern "C" __global__ void k_scan(const int* cnt, int n, int* start, int* cursor)
{
    __shared__ int part[1024];
    int t = threadIdx.x, T = blockDim.x;
    int per = (n + T - 1) / T;
    int a = t * per, e = min(n, a + per);
    int s = 0;
    for (int i = a; i < e; ++i) s += cnt[i];
    part[t] = s;
    __syncthreads();
    for (int o = 1; o < T; o <<= 1) {
        int v = t >= o ? part[t - o] : 0;
        __syncthreads();
        part[t] += v;
        __syncthreads();
    }
    int off = t > 0 ? part[t - 1] : 0;
    for (int i = a; i < e; ++i) { start[i] = off; off += cnt[i]; cursor[i] = 0; }
    if (t == T - 1) start[n] = part[T - 1];
}

/* candidates regrouped by cell, in list order: key, packed (y << 16 | x), skip flag (contiguous per cell) */
extern "C" __global__ void k_fill(const int* ncand, const int* ccell, const int* cx, const int* cy, const u64* ckey,
                                  const int* start, int* cursor, u64* lkey, int* lxy, u8* lskip)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= ncand[0]) return;
    int c = ccell[k];
    int q = start[c] + atomicAdd(&cursor[c], 1);
    lkey[q] = ckey[k];
    lxy[q] = (cy[k] << 16) | cx[k];
    lskip[q] = 0;
}

/* the exact greedy: cell state in shared memory (dynamic: 21 bytes per cell); candidates addressed by list position */
extern "C" __global__ void k_greedy(int ncell, int gw, int gh, const int* __restrict__ start,
                                    const u64* __restrict__ lkey, const int* __restrict__ lxy, u8* lskip, double r2,
                                    int* plist, int* npick, int* rounds)
{
    extern __shared__ unsigned char smem[];
    volatile u64* ckk = (volatile u64*)smem;                         /* key of the current candidate / of the pick */
    volatile int* ccur = (volatile int*)(smem + 8 * (size_t)ncell);  /* current position (-1 = recompute) */
    volatile int* cpick = ccur + ncell;                              /* pick position (-1 none) */
    volatile int* cpxy = cpick + ncell;                              /* pick (y << 16 | x) */
    volatile u8* cdone = (volatile u8*)(cpxy + ncell);
    __shared__ int unresolved;
    __shared__ int np;
    int t = threadIdx.x, T = blockDim.x;
    for (int c = t; c < ncell; c += T) {
        ccur[c] = -1; cpick[c] = -1; ckk[c] = 0; cpxy[c] = 0; cdone[c] = start[c + 1] == start[c];
    }
    if (t == 0) np = 0;
    __syncthreads();
    int it = 0;
    while (true) {
        if (t == 0) unresolved = 0;
        __syncthreads();
        for (int c = t; c < ncell; c += T) {
            if (cdone[c]) continue;
            while (true) {
                int q = ccur[c];
                u64 kk;
                if (q < 0) {
                    u64 bk = 0;
                    for (int j = start[c]; j < start[c + 1]; ++j) {
                        if (!lskip[j]) { u64 kj = lkey[j]; if (q < 0 || kj > bk) { bk = kj; q = j; } }
                    }
                    if (q < 0) { cpick[c] = -1; ckk[c] = 0; __threadfence_block(); cdone[c] = 1; break; }
                    ckk[c] = bk;
                    __threadfence_block();
                    ccur[c] = q;
                    kk = bk;
                } else kk = ckk[c];
                int xy = lxy[q];
                int qx = xy & 0xFFFF, qy = xy >> 16;
                int gx = c % gw, gy = c / gw;
                bool wait = false, blocked = false;
                for (int ny = gy - 1; ny <= gy + 1 && !blocked; ++ny) {
                    if (ny < 0 || ny >= gh) continue;
                    for (int nx = gx - 1; nx <= gx + 1; ++nx) {
                        if (nx < 0 || nx >= gw) continue;
                        int nb = ny * gw + nx;
                        if (nb == c) continue;
                        if (cdone[nb]) {
                            if (cpick[nb] >= 0 && ckk[nb] > kk) {
                                int pxy = cpxy[nb];
                                long long ddx = (long long)(pxy & 0xFFFF) - qx, ddy = (long long)(pxy >> 16) - qy;
                                if ((double)(ddx * ddx + ddy * ddy) < r2) { blocked = true; break; }
                            }
                        } else if (ccur[nb] < 0 || ckk[nb] > kk) wait = true;
                    }
                }
                if (blocked) { lskip[q] = 1; ccur[c] = -1; continue; }
                if (wait) { atomicAdd(&unresolved, 1); break; }
                cpick[c] = q;
                cpxy[c] = xy;
                ckk[c] = kk;
                __threadfence_block();
                cdone[c] = 1;
                break;
            }
        }
        __syncthreads();
        ++it;
        if (unresolved == 0) break;
        __syncthreads();
    }
    for (int c = t; c < ncell; c += T) {
        int p = cpick[c];
        if (p >= 0) plist[atomicAdd(&np, 1)] = p;
    }
    __syncthreads();
    if (t == 0) { npick[0] = np; rounds[0] = it; }
}

/* rank of each pick (descending key) -> output slot; count = min(npick, n_want) */
extern "C" __global__ void k_rank(const int* plist, const int* npick, const u64* lkey, const int* lxy, int n_want,
                                  int* ox, int* oy, int* count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int P = npick[0];
    if (i == 0) count[0] = P < n_want ? P : n_want;
    if (i >= P) return;
    u64 k = lkey[plist[i]];
    int r = 0;
    for (int j = 0; j < P; ++j) r += lkey[plist[j]] > k;
    if (r < n_want) { int xy = lxy[plist[i]]; ox[r] = xy & 0xFFFF; oy[r] = xy >> 16; }
}

extern "C" __global__ void k_subpix(const float* R, int H, int W, const int* count, const int* ox, const int* oy,
                                    float* kp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count[0]) return;
    int x = ox[i], y = oy[i];
    double fx = (double)x, fy = (double)y;
    if (x > 0 && x < W - 1 && y > 0 && y < H - 1) {
        const float* r = R + (size_t)y * W + x;
        double c = r[0], l = r[-1], rr = r[1], u = r[-W], d = r[W];
        double c2 = __dmul_rn(2.0, c);
        double denx = __dadd_rn(__dsub_rn(l, c2), rr);
        double deny = __dadd_rn(__dsub_rn(u, c2), d);
        double dx = denx < 0.0 ? __ddiv_rn(__dmul_rn(0.5, __dsub_rn(l, rr)), denx) : 0.0;
        double dy = deny < 0.0 ? __ddiv_rn(__dmul_rn(0.5, __dsub_rn(u, d)), deny) : 0.0;
        dx = fmin(fmax(dx, -0.5), 0.5);
        dy = fmin(fmax(dy, -0.5), 0.5);
        fx = __dadd_rn(fx, dx);
        fy = __dadd_rn(fy, dy);
    }
    kp[2 * i] = __double2float_rn(fx);
    kp[2 * i + 1] = __double2float_rn(fy);
}

/* ======================================= 4. ORB (no orientation) ======================================= */
__device__ __forceinline__ int padsrc(int i, int n)   /* padded index -> source index (reflect-101 pad 32) */
{
    return REFL(i - 32, n);
}

extern "C" __global__ void k_blur_rows(const u8* I, int H, int W, int Hp, int Wp, const float* kx, int vend,
                                       float* RB)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x >= Wp) return;
    const u8* row = I + (size_t)padsrc(y, H) * W;
    float p[7];
#pragma unroll
    for (int k = 0; k < 7; ++k) p[k] = (float)row[padsrc(REFL(x - 3 + k, Wp), W)];
    float s;
    if (x < vend) {
        s = __fmul_rn(p[0], kx[0]);
#pragma unroll
        for (int k = 1; k < 7; ++k) s = __fmaf_rn(p[k], kx[k], s);
    } else {
        s = __fmul_rn(kx[0], p[0]);
#pragma unroll
        for (int k = 1; k < 7; ++k) s = __fadd_rn(s, __fmul_rn(kx[k], p[k]));
    }
    RB[(size_t)y * Wp + x] = s;
}

extern "C" __global__ void k_blur_cols(const float* RB, int Hp, int Wp, const float* ky, int vend, u8* B)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x >= Wp) return;
    float s;
    float S0 = RB[(size_t)y * Wp + x];
    if (x < vend) {
        s = __fmul_rn(ky[3], S0);
#pragma unroll
        for (int k = 1; k <= 3; ++k) {
            float t = __fadd_rn(RB[(size_t)REFL(y + k, Hp) * Wp + x], RB[(size_t)REFL(y - k, Hp) * Wp + x]);
            s = __fmaf_rn(ky[3 + k], t, s);
        }
    } else {
        s = __fadd_rn(__fmul_rn(ky[3], S0), 0.0f);
#pragma unroll
        for (int k = 1; k <= 3; ++k) {
            float t = __fadd_rn(RB[(size_t)REFL(y + k, Hp) * Wp + x], RB[(size_t)REFL(y - k, Hp) * Wp + x]);
            s = __fadd_rn(s, __fmul_rn(ky[3 + k], t));
        }
    }
    int v = __float2int_rn(s);
    B[(size_t)y * Wp + x] = (u8)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

extern "C" __global__ void k_orb(const u8* B, int Wp, const float* kp, const int* count, const signed char* pat,
                                 u8* desc)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int i = t >> 5, by = t & 31;
    if (i >= count[0]) return;
    int cx = __float2int_rn(__fadd_rn(kp[2 * i], 32.0f));
    int cy = __float2int_rn(__fadd_rn(kp[2 * i + 1], 32.0f));
    const u8* c = B + (size_t)cy * Wp + cx;
    int val = 0;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const signed char* q = pat + 4 * (8 * by + b);
        int t0 = c[(int)q[1] * Wp + (int)q[0]], t1 = c[(int)q[3] * Wp + (int)q[2]];
        val |= (t0 < t1) << b;
    }
    desc[32 * i + by] = (u8)val;
}

/* ======================================= 5. 1-D stereo (stereo1d.stereo_points_band) ======================================= */
__device__ __forceinline__ double zncc_ru8(const double* Td, double ts, const u8* img, int W, int x, int v)
{
    int s = 0, s2 = 0;
    double sx = 0.0;
    int k = 0;
    for (int dy = -HALF; dy <= HALF; ++dy) {
        const u8* row = img + (size_t)(v + dy) * W + x;
        for (int dx = -HALF; dx <= HALF; ++dx) {
            int r = row[dx];
            s += r; s2 += r * r;
            sx = __dadd_rn(sx, __dmul_rn((double)r, Td[k]));
            ++k;
        }
    }
    double m = __ddiv_rn((double)s, (double)NPIX);
    double var = __dsub_rn(__ddiv_rn((double)s2, (double)NPIX), __dmul_rn(m, m));
    double sd = __dsqrt_rn(var > 0.0 ? var : 0.0);
    double den = __dmul_rn(__dmul_rn(ts, sd), (double)NPIX);
    return den > 1e-6 ? __ddiv_rn(sx, den) : -2.0;
}

/* float32 image: numba's float(float32) stays float32, so the CPU's r * r is a float32 product (stereo1d with dy != 0) */
__device__ __forceinline__ double zncc_rf(const double* Td, double ts, const float* img, int W, int x, int v)
{
    double s = 0.0, s2 = 0.0, sx = 0.0;
    int k = 0;
    for (int dy = -HALF; dy <= HALF; ++dy) {
        const float* row = img + (size_t)(v + dy) * W + x;
        for (int dx = -HALF; dx <= HALF; ++dx) {
            float rf = row[dx];
            double r = (double)rf;
            s = __dadd_rn(s, r); s2 = __dadd_rn(s2, (double)__fmul_rn(rf, rf));
            sx = __dadd_rn(sx, __dmul_rn(r, Td[k]));
            ++k;
        }
    }
    double m = __ddiv_rn(s, (double)NPIX);
    double var = __dsub_rn(__ddiv_rn(s2, (double)NPIX), __dmul_rn(m, m));
    double sd = __dsqrt_rn(var > 0.0 ? var : 0.0);
    double den = __dmul_rn(__dmul_rn(ts, sd), (double)NPIX);
    return den > 1e-6 ? __ddiv_rn(sx, den) : -2.0;
}

__device__ __forceinline__ double sub_px(u8 b, u8 a) { return (double)b - (double)a; }
__device__ __forceinline__ double sub_px(float b, float a) { return (double)__fsub_rn(b, a); }   /* numba float32 */

template <typename TI>
__device__ __forceinline__ void bil(const TI* img, int H, int W, double x, double y, double* val, double* gx)
{
    int x0 = (int)floor(x), y0 = (int)floor(y);
    double ax = __dsub_rn(x, (double)x0), ay = __dsub_rn(y, (double)y0);
    if (x0 < 0) x0 = 0;
    if (x0 > W - 2) x0 = W - 2;
    if (y0 < 0) y0 = 0;
    if (y0 > H - 2) y0 = H - 2;
    TI ia = img[(size_t)y0 * W + x0], ib = img[(size_t)y0 * W + x0 + 1];
    TI ic = img[(size_t)(y0 + 1) * W + x0], id = img[(size_t)(y0 + 1) * W + x0 + 1];
    double a = (double)ia, c = (double)ic;
    double ba = sub_px(ib, ia), dc = sub_px(id, ic);
    double top = __dadd_rn(a, __dmul_rn(ax, ba));
    double bot = __dadd_rn(c, __dmul_rn(ax, dc));
    *val = __dadd_rn(top, __dmul_rn(ay, __dsub_rn(bot, top)));
    *gx = __dadd_rn(__dmul_rn(ba, __dsub_rn(1.0, ay)), __dmul_rn(dc, ay));
}

/* The 3x3 solve and inverse EXACTLY as the CPU's numba np.linalg.solve / np.linalg.inv compute them on this build
   (scipy cython_lapack -> OpenBLAS): OpenBLAS getf2 (left-looking LU: per column apply the previous interchanges,
   dot for the upper part, gemv for the lower part with the temp accumulated from 0 by fused multiply-adds and then
   subtracted, first-max pivot, swap of the columns done so far, scale by the reciprocal), getrs (trsv lower-unit and
   upper-non-unit, axpy with fused multiply-add, division by the diagonal), trti2 (trmv + scal) and reference dgetri
   (gemv per column, column interchanges).  Verified bit for bit against numba on 7000 LK-like systems incl. degenerate
   ones (warmup re-checks it on random systems: gpu_front.check_lk_solver).  a[i][j]: row i, column j. */
__device__ __forceinline__ void ob_getf2(double a[3][3], int piv[3])
{
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < j; ++i) { int ip = piv[i]; if (ip != i) { double tt = a[i][j]; a[i][j] = a[ip][j]; a[ip][j] = tt; } }
        for (int i = 1; i < j; ++i) {
            double s = 0.0;
            for (int k = 0; k < i; ++k) s = __fma_rn(a[i][k], a[k][j], s);
            a[i][j] = __dsub_rn(a[i][j], s);
        }
        if (j > 0) {
            for (int i = j; i < 3; ++i) {
                double s = 0.0;
                for (int k = 0; k < j; ++k) s = __fma_rn(a[i][k], a[k][j], s);
                a[i][j] = __dsub_rn(a[i][j], s);
            }
        }
        int jp = j; double mx = fabs(a[j][j]);
        for (int i = j + 1; i < 3; ++i) if (fabs(a[i][j]) > mx) { mx = fabs(a[i][j]); jp = i; }
        piv[j] = jp;
        double t = a[jp][j];
        if (t != 0.0) {
            t = __ddiv_rn(1.0, t);
            if (jp != j) for (int k = 0; k <= j; ++k) { double tt = a[j][k]; a[j][k] = a[jp][k]; a[jp][k] = tt; }
            for (int i = j + 1; i < 3; ++i) a[i][j] = __dmul_rn(a[i][j], t);
        }
    }
}

__device__ __forceinline__ void ob_getrs(const double a[3][3], const int piv[3], double y[3])
{
    for (int j = 0; j < 3; ++j) { int p = piv[j]; if (p != j) { double tt = y[j]; y[j] = y[p]; y[p] = tt; } }
    for (int j = 0; j < 3; ++j)
        for (int i = j + 1; i < 3; ++i) y[i] = __fma_rn(-y[j], a[i][j], y[i]);
    for (int j = 2; j >= 0; --j) {
        y[j] = __ddiv_rn(y[j], a[j][j]);
        for (int i = 0; i < j; ++i) y[i] = __fma_rn(-y[j], a[i][j], y[i]);
    }
}

/* inv(A) from the getf2 factors (overwrites a with the inverse) */
__device__ __forceinline__ void ob_getri(double a[3][3], const int piv[3])
{
    for (int j = 0; j < 3; ++j) {                               /* trti2, upper non-unit */
        double ajj = __ddiv_rn(1.0, a[j][j]);
        a[j][j] = ajj;
        double x[3];
        for (int i = 0; i < j; ++i) x[i] = a[i][j];
        for (int i = 0; i < j; ++i) {                           /* trmv NUN: axpy then scale by the diagonal */
            for (int q = 0; q < i; ++q) x[q] = __fma_rn(x[i], a[q][i], x[q]);
            x[i] = __dmul_rn(x[i], a[i][i]);
        }
        for (int i = 0; i < j; ++i) a[i][j] = __dmul_rn(x[i], -ajj);
    }
    double work[3] = {0.0, 0.0, 0.0};
    for (int j = 2; j >= 0; --j) {                              /* reference dgetri, unblocked */
        for (int i = j + 1; i < 3; ++i) { work[i] = a[i][j]; a[i][j] = 0.0; }
        if (j < 2) {
            for (int r = 0; r < 3; ++r) {
                double s = 0.0;
                for (int k = j + 1; k < 3; ++k) s = __fma_rn(a[r][k], work[k], s);
                a[r][j] = __dsub_rn(a[r][j], s);
            }
        }
    }
    for (int j = 1; j >= 0; --j) {
        int jp = piv[j];
        if (jp != j) for (int r = 0; r < 3; ++r) { double tt = a[r][j]; a[r][j] = a[r][jp]; a[r][jp] = tt; }
    }
}

/* self-check kernel: solve + inverse of n systems (row-major 9 doubles + rhs 3) -> x (3) and inv (9) */
extern "C" __global__ void k_lkcheck(const double* As, const double* gs, int n, double* X, double* Inv)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    double a[3][3]; int piv[3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) a[i][j] = As[9 * k + 3 * i + j];
    ob_getf2(a, piv);
    double y[3] = {gs[3 * k], gs[3 * k + 1], gs[3 * k + 2]};
    ob_getrs(a, piv, y);
    for (int i = 0; i < 3; ++i) X[3 * k + i] = y[i];
    ob_getri(a, piv);
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) Inv[9 * k + 3 * i + j] = a[i][j];
}

extern "C" __global__ void k_stereo(const u8* L, const float* R, int H, int W, const double* kp, const i64* dlo,
                                    const i64* dhi, int N, int band, double zmin, double uniq1, double lrt,
                                    double* out_d, double* out_sig, u8* out_ok, double* out_zn, double* out_dy)
{
    int n = blockIdx.x, t = threadIdx.x;
    if (n >= N) return;
    __shared__ double Td[NPIX], T2d[NPIX], sc[MAXD], sc2[MAXD], col[PW];
    __shared__ int jb[MAXD];
    __shared__ double sh_ts, sh_rs, sh_seed, sh_d, sh_a, sh_b;
    __shared__ double terms[10][NPIX], ssum[10];
    __shared__ int sh_go, sh_ui, sh_vi, sh_lo, sh_hi, sh_best, sh_js, sh_dmax;
    double u = kp[2 * n], v = kp[2 * n + 1];
    if (t == 0) {
        out_ok[n] = 0; out_d[n] = 0.0; out_sig[n] = CUDART_INF; out_zn[n] = -2.0; out_dy[n] = CUDART_NAN;
        int ui = (int)floor(__dadd_rn(u, 0.5)), vi = (int)floor(__dadd_rn(v, 0.5));
        int go = 1;
        if (ui - HALF - 2 < 0 || ui + HALF + 2 >= W || vi - HALF - 2 - band < 0 || vi + HALF + 2 + band >= H) go = 0;
        i64 lo = dlo[n], hi = dhi[n];
        if (lo < 0) lo = 0;
        if (hi > MAXD - 1) hi = MAXD - 1;
        if (hi > ui - HALF - 2) hi = ui - HALF - 2;
        if (hi < lo) go = 0;
        sh_go = go; sh_ui = ui; sh_vi = vi; sh_lo = (int)lo; sh_hi = (int)hi;
    }
    __syncthreads();
    if (!sh_go) return;
    int ui = sh_ui, vi = sh_vi, lo = sh_lo, hi = sh_hi;
    if (t == 0) {          /* _patch_stats(L, ui, vi) and the template, in the CPU order */
        double s = 0.0, s2 = 0.0;
        for (int dy = -HALF; dy <= HALF; ++dy)
            for (int dx = -HALF; dx <= HALF; ++dx) {
                double x = (double)L[(size_t)(vi + dy) * W + ui + dx];
                s = __dadd_rn(s, x); s2 = __dadd_rn(s2, __dmul_rn(x, x));
            }
        double m = __ddiv_rn(s, (double)NPIX);
        double var = __dsub_rn(__ddiv_rn(s2, (double)NPIX), __dmul_rn(m, m));
        sh_ts = __dsqrt_rn(var > 0.0 ? var : 0.0);
        int k = 0;
        for (int dy = -HALF; dy <= HALF; ++dy)
            for (int dx = -HALF; dx <= HALF; ++dx) { Td[k] = __dsub_rn((double)L[(size_t)(vi + dy) * W + ui + dx], m); ++k; }
    }
    __syncthreads();
    double ts = sh_ts;
    if (ts <= 1e-6) return;
    {
        int d = t;
        double s = -2.0;
        int jj = 0;
        if (d >= lo && d <= hi) {
            s = zncc_rf(Td, ts, R, W, ui - d, vi);
            for (int j = 1; j <= band; ++j) {
                double s1 = zncc_rf(Td, ts, R, W, ui - d, vi - j);
                if (s1 > s) { s = s1; jj = -j; }
                s1 = zncc_rf(Td, ts, R, W, ui - d, vi + j);
                if (s1 > s) { s = s1; jj = j; }
            }
        }
        sc[d] = s; jb[d] = jj;
    }
    __syncthreads();
    if (t == 0) {
        int best = -1; double bs = -2.0;
        for (int d = lo; d <= hi; ++d) if (sc[d] > bs) { bs = sc[d]; best = d; }
        out_zn[n] = bs;
        int go = 1;
        if (best < 0 || bs < zmin) go = 0;
        else {
            double sec = -2.0;
            for (int d = lo; d <= hi; ++d) if (abs(d - best) > 1 && sc[d] > sec) sec = sc[d];
            if (!(__dsub_rn(1.0, sec) > __dmul_rn(__dsub_rn(1.0, bs), uniq1))) go = 0;
        }
        sh_go = go; sh_best = best;
        if (go) {
            int js = jb[best];
            sh_js = js;
            int xr = ui - best;
            double s = 0.0, s2 = 0.0;
            for (int dy = -HALF; dy <= HALF; ++dy)
                for (int dx = -HALF; dx <= HALF; ++dx) {
                    float xf = R[(size_t)(vi + js + dy) * W + xr + dx];
                    s = __dadd_rn(s, (double)xf); s2 = __dadd_rn(s2, (double)__fmul_rn(xf, xf));
                }
            double m = __ddiv_rn(s, (double)NPIX);
            double var = __dsub_rn(__ddiv_rn(s2, (double)NPIX), __dmul_rn(m, m));
            sh_rs = __dsqrt_rn(var > 0.0 ? var : 0.0);
            int k = 0;
            for (int dy = -HALF; dy <= HALF; ++dy)
                for (int dx = -HALF; dx <= HALF; ++dx) { T2d[k] = __dsub_rn((double)R[(size_t)(vi + js + dy) * W + xr + dx], m); ++k; }
            int dmax = MAXD - 1;
            if (xr + dmax + HALF >= W) dmax = W - 1 - HALF - xr;
            sh_dmax = dmax;
        }
    }
    __syncthreads();
    if (!sh_go) return;
    if (sh_rs <= 1e-6) return;
    int best = sh_best, js = sh_js, xr = ui - best;
    {
        int d = t;
        sc2[d] = d <= sh_dmax ? zncc_ru8(T2d, sh_rs, L, W, xr + d, vi) : -2.0;
    }
    __syncthreads();
    if (t == 0) {
        int bb = -1; double bbs = -2.0;
        for (int d = 0; d <= sh_dmax; ++d) if (sc2[d] > bbs) { bbs = sc2[d]; bb = d; }
        int go = 0;
        if (!(bb < 0 || fabs((double)(bb - best)) > lrt)) {
            double off = 0.0;
            if (best > lo && best < hi) {
                double y0 = sc[best - 1], y2 = sc[best + 1], bs = sc[best];
                double den = __dadd_rn(__dsub_rn(y0, __dmul_rn(2.0, bs)), y2);
                if (den < -1e-9) {
                    off = __ddiv_rn(__dmul_rn(0.5, __dsub_rn(y0, y2)), den);
                    if (off > 0.5) off = 0.5;
                    if (off < -0.5) off = -0.5;
                }
            }
            sh_seed = __dadd_rn(__dadd_rn((double)best, off), __dsub_rn(u, (double)ui));
            sh_d = sh_seed; sh_a = 1.0; sh_b = 0.0;
            go = 1;
        }
        sh_go = go;
    }
    __syncthreads();
    if (!sh_go) return;
    /* affine 1-D LK (_lk_v), bit for bit: the 121 per-pixel terms in parallel; each of the 10 sums is then taken by
       one thread SEQUENTIALLY in the CPU's pixel order (k = 0..120, from 0.0, no fused multiply-add); the 3x3 solve and
       the covariance inverse are OpenBLAS's arithmetic (ob_getf2 / ob_getrs / ob_getri). */
    {
        double pxk = 0.0, pyk = 0.0, Tk = 0.0;
        double voff = (double)js;
        if (t < NPIX) {
            pxk = __dadd_rn(u, (double)(t % PW - HALF));
            pyk = __dadd_rn(v, (double)(t / PW - HALF));
            double g_;
            bil<u8>(L, H, W, pxk, pyk, &Tk, &g_);
        }
        for (int it = 0; it <= LK_ITERS_; ++it) {
            double d = sh_d, a = sh_a, b = sh_b;
            if (t < NPIX) {
                double rv, rg;
                bil<float>(R, H, W, __dsub_rn(pxk, d), __dadd_rn(pyk, voff), &rv, &rg);
                double r = __dsub_rn(Tk, __dadd_rn(__dmul_rn(a, rv), b));
                double J0 = __dmul_rn(-a, rg), J1 = rv;
                terms[0][t] = __dmul_rn(J0, r); terms[1][t] = __dmul_rn(J1, r); terms[2][t] = r;
                terms[3][t] = __dmul_rn(J0, J0); terms[4][t] = __dmul_rn(J0, J1); terms[5][t] = J0;
                terms[6][t] = __dmul_rn(J1, J1); terms[7][t] = J1; terms[8][t] = 1.0; terms[9][t] = __dmul_rn(r, r);
            }
            __syncthreads();
            if (t < 10) {
                double s = 0.0;
                for (int k = 0; k < NPIX; ++k) s = __dadd_rn(s, terms[t][k]);
                ssum[t] = s;
            }
            __syncthreads();
            if (t == 0) {
                double S[10];
                for (int q = 0; q < 10; ++q) S[q] = ssum[q];
                double A[3][3] = {{S[3], S[4], S[5]}, {S[4], S[6], S[7]}, {S[5], S[7], S[8]}};
                for (int i = 0; i < 3; ++i) A[i][i] = __dadd_rn(A[i][i], 1e-9);
                int piv[3];
                ob_getf2(A, piv);
                if (it < LK_ITERS_) {
                    double dl[3] = {S[0], S[1], S[2]};
                    ob_getrs(A, piv, dl);
                    double dn = __dadd_rn(d, dl[0]), an = __dadd_rn(a, dl[1]), bn = __dadd_rn(b, dl[2]);
                    sh_d = dn; sh_a = an; sh_b = bn;
                    if (!(isfinite(dn) && isfinite(an))) sh_go = 0;
                } else {
                    double s2 = __ddiv_rn(S[9], (double)(NPIX - 3));
                    ob_getri(A, piv);
                    double var = __dmul_rn(s2, A[0][0]);
                    double sig = var > 0.0 ? __dsqrt_rn(var) : 0.0;
                    int conv = isfinite(d) && fabs(__dsub_rn(d, sh_seed)) <= 1.0 && a > 0.5 && a < 2.0;
                    if (conv && !(d < 0.5)) { out_d[n] = d; out_sig[n] = sig; out_ok[n] = 1; }
                    else sh_go = 0;
                }
            }
            __syncthreads();
            if (!sh_go) return;
        }
    }
    if (vi - 2 * HALF - 1 >= 0 && vi + 2 * HALF + 1 < H) {
        if (t < PW) col[t] = zncc_rf(Td, ts, R, W, xr, vi + t - HALF);
        __syncthreads();
        if (t == 0) {
            int jp = 0; double cs = -2.0;
            for (int j = -HALF; j <= HALF; ++j) if (col[j + HALF] > cs) { cs = col[j + HALF]; jp = j; }
            double offy = 0.0;
            if (jp > -HALF && jp < HALF) {
                double y0 = col[jp - 1 + HALF], y2 = col[jp + 1 + HALF];
                double den = __dadd_rn(__dsub_rn(y0, __dmul_rn(2.0, cs)), y2);
                if (den < -1e-9) {
                    offy = __ddiv_rn(__dmul_rn(0.5, __dsub_rn(y0, y2)), den);
                    if (offy > 0.5) offy = 0.5;
                    if (offy < -0.5) offy = -0.5;
                }
            }
            out_dy[n] = __dadd_rn((double)jp, offy);
        }
    }
}

/* shift_rows: the right image resampled at rows y + dy (float32, as numpy computes it) */
extern "C" __global__ void k_shift(const u8* Rr, int H, int W, double dy, float* Rs)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x >= W) return;
    double yy = __dadd_rn((double)y, dy);
    double y0d = floor(yy);
    float f = __double2float_rn(__dsub_rn(yy, y0d));
    long long y0 = (long long)y0d;
    long long ya = y0 < 0 ? 0 : (y0 > H - 1 ? H - 1 : y0);
    long long yb = y0 + 1 < 0 ? 0 : (y0 + 1 > H - 1 ? H - 1 : y0 + 1);
    float a = (float)Rr[(size_t)ya * W + x], b = (float)Rr[(size_t)yb * W + x];
    Rs[(size_t)y * W + x] = __fadd_rn(a, __fmul_rn(f, __fsub_rn(b, a)));
}

extern "C" __global__ void k_tof(const u8* Rr, int n, float* Rs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Rs[i] = (float)Rr[i];
}

/* ======================================= 6. Hamming (stage A) ======================================= */
extern "C" __global__ void k_ham_rows(const u64* A, int L, const u64* B, int N, int* o1, int* o2, int* oi)
{
    int i = blockIdx.x;
    if (i >= L) return;
    __shared__ int s1[128], s2[128], si[128];
    u64 a0 = A[4 * i], a1 = A[4 * i + 1], a2 = A[4 * i + 2], a3 = A[4 * i + 3];
    int b1 = 1 << 30, b2 = 1 << 30, bi = -1;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        int d = __popcll(a0 ^ B[4 * j]) + __popcll(a1 ^ B[4 * j + 1]) + __popcll(a2 ^ B[4 * j + 2])
              + __popcll(a3 ^ B[4 * j + 3]);
        if (d < b1) { b2 = b1; b1 = d; bi = j; } else if (d < b2) b2 = d;
    }
    int t = threadIdx.x;
    s1[t] = b1; s2[t] = b2; si[t] = bi;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (t < o) {
            int xb1 = s1[t], xb2 = s2[t], xi = si[t];
            int yb1 = s1[t + o], yb2 = s2[t + o], yi = si[t + o];
            bool ywin = (yb1 < xb1) || (yb1 == xb1 && yi >= 0 && (xi < 0 || yi < xi));
            if (ywin) { s1[t] = yb1; si[t] = yi; s2[t] = min(yb2, xb1); }
            else { s2[t] = min(xb2, yb1); }
        }
        __syncthreads();
    }
    if (t == 0) { o1[i] = s1[0]; o2[i] = s2[0]; oi[i] = si[0]; }
}

extern "C" __global__ void k_ham_cols(const u64* A, int L, const u64* B, int N, int* ci)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    u64 b0 = B[4 * j], b1 = B[4 * j + 1], b2 = B[4 * j + 2], b3 = B[4 * j + 3];
    int best = 1 << 30, idx = -1;
    for (int i = 0; i < L; ++i) {
        int d = __popcll(A[4 * i] ^ b0) + __popcll(A[4 * i + 1] ^ b1) + __popcll(A[4 * i + 2] ^ b2)
              + __popcll(A[4 * i + 3] ^ b3);
        if (d < best) { best = d; idx = i; }
    }
    ci[j] = idx;
}
""".replace("LK_ITERS_", str(int(LK_ITERS)))

_OPTS = ("--std=c++14", "--fmad=false", "-ftz=false", "-prec-sqrt=true", "-prec-div=true")


def _module():
    """Compile _SRC with NVRTC WITHOUT CuPy's forced -ftz=true (subnormal responses must survive), cached on disk."""
    import cupy as cp
    from cupy.cuda import compiler, function
    arch = compiler._get_arch()
    key = hashlib.sha1((_SRC + repr(_OPTS) + str(arch) + str(compiler._get_nvrtc_version())
                        + cp.__version__).encode()).hexdigest()
    cache = os.path.join(os.environ.get("CUPY_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cupy"),
                         "xclean_gpu_front")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, key + ".bin")
    binary = None
    if os.path.exists(path):
        with open(path, "rb") as fh:
            binary = fh.read()
    if not binary:
        binary, _map = compiler.compile_using_nvrtc(_SRC, _OPTS, arch, "xclean_gpu_front.cu")
        tmp = path + ".%d.tmp" % os.getpid()
        with open(tmp, "wb") as fh:
            fh.write(binary)
        os.replace(tmp, path)
    m = function.Module()
    m.load(binary)
    return m


def _pairwise_tree(n):
    """numpy's float32 pairwise-sum recursion over n elements (PW_BLOCKSIZE 128, unroll 8) as leaves + a combine tree
    evaluated level by level.  -> (leaf_start, leaf_len, left, right, level_start, root)."""
    leaves = []
    inner = []                     # (node_id placeholder) built recursively

    def rec(st, m):
        if m <= 128:
            leaves.append((st, m))
            return ("L", len(leaves) - 1, 0)
        n2 = m // 2
        n2 -= n2 % 8
        a = rec(st, n2)
        b = rec(st + n2, m - n2)
        h = max(a[2], b[2]) + 1
        inner.append((a, b, h))
        return ("I", len(inner) - 1, h)

    import sys
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    root = rec(0, n)
    nl = len(leaves)
    # order internal nodes by height; node ids: leaves 0..nl-1, then internal by (height, creation)
    order = sorted(range(len(inner)), key=lambda i: (inner[i][2], i))
    nid = {}
    for pos, i in enumerate(order):
        nid[i] = nl + pos

    def ref(x):
        return x[1] if x[0] == "L" else nid[x[1]]

    left = np.zeros(nl + len(inner), np.int32)
    right = np.zeros(nl + len(inner), np.int32)
    heights = np.zeros(nl + len(inner), np.int32)
    for i, (a, b, h) in enumerate(inner):
        left[nid[i]] = ref(a)
        right[nid[i]] = ref(b)
        heights[nid[i]] = h
    maxh = int(heights.max()) if len(inner) else 0
    lvl = np.zeros(maxh + 1, np.int32)        # level h (1..maxh) occupies [lvl[h-1], lvl[h]) after the shift below
    starts = [nl]
    for h in range(1, maxh + 1):
        starts.append(starts[-1] + int((heights[nl:] == h).sum()))
    lvl = np.asarray(starts, np.int32)
    rid = ref(root)
    ls = np.asarray([s for s, _ in leaves], np.int64)
    ll = np.asarray([m for _, m in leaves], np.int32)
    return ls, ll, left, right, lvl, int(rid)


def _ndarray_ptr(a):
    return a.__array_interface__["data"][0]


class GpuFront:
    """Device-resident front end for one image size.  All buffers are allocated here, once."""

    def __init__(self, W, H, cap=2048, verbose=False):
        import cupy as cp
        self.cp = cp
        self.W, self.H, self.cap = int(W), int(H), int(cap)
        W, H = self.W, self.H
        self.pool = cp.cuda.MemoryPool()
        self.pool.set_limit(size=POOL_LIMIT_BYTES)
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.mod = _module()
        f = self.mod.get_function
        self.k = {n: f(n) for n in ("k_sobel", "k_boxrows", "k_boxcols", "k_eig", "k_noise_leaves", "k_noise_tree", "k_cellmax",
                                     "k_cand", "k_scan", "k_fill", "k_greedy", "k_rank", "k_subpix", "k_blur_rows",
                                     "k_blur_cols", "k_orb", "k_stereo", "k_shift", "k_tof", "k_ham_rows",
                                     "k_ham_cols", "k_lkcheck")}
        Hp, Wp = H + 2 * ORB_PAD, W + 2 * ORB_PAD
        self.Hp, self.Wp = Hp, Wp
        # the greedy kernel keeps 17 bytes of state per cell in shared memory: opt in to the device maximum
        dev = cp.cuda.Device()
        optin = int(dev.attributes.get("MaxSharedMemoryPerBlockOptin", 49152))
        self.greedy_shm_max = optin - 64
        from cupy.cuda import driver as _drv
        _drv.funcSetAttribute(self.k["k_greedy"].ptr, 8, self.greedy_shm_max)   # MAX_DYNAMIC_SHARED_SIZE_BYTES
        ncell_max = int(math.ceil(W / 4.0)) * (int(math.ceil(H / 4.0)) + 2) + 1
        self.ncell_max = ncell_max
        tree = _pairwise_tree((H - 2) * (W - 2))
        with cp.cuda.using_allocator(self.pool.malloc):
            e = cp.empty
            self.d_left = e((H, W), cp.uint8)
            self.d_right = e((H, W), cp.uint8)
            self.d_rs = e((H, W), cp.float32)
            self.d_dx = e((H, W), cp.float32)
            self.d_dy = e((H, W), cp.float32)
            self.d_cov = e((H, W, 3), cp.float32)
            self.d_rsum = e((3, H, W), cp.float64)
            self.d_R = e((H, W), cp.float32)
            self.d_flags = e(4, cp.int32)
            ls, ll, left, right, lvl, root = tree
            self.n_leaf = len(ls)
            self.n_lvl = len(lvl) - 1
            self.root = root
            self.d_lstart = cp.asarray(ls)
            self.d_llen = cp.asarray(ll)
            self.d_left_n = cp.asarray(left)
            self.d_right_n = cp.asarray(right)
            self.d_lvl = cp.asarray(lvl)
            self.d_nodeval = e(len(left), cp.float32)
            self.d_floor_f = e(1, cp.float32)
            self.d_thr = e(ncell_max, cp.float32)
            self.d_ncand = e(1, cp.int32)
            nc = H * W
            self.d_cx = e(nc, cp.int32)
            self.d_cy = e(nc, cp.int32)
            self.d_ckey = e(nc, cp.uint64)
            self.d_ccell = e(nc, cp.int32)
            self.d_lkey = e(nc, cp.uint64)
            self.d_lxy = e(nc, cp.int32)
            self.d_lskip = e(nc, cp.uint8)
            self.d_ccount = e(ncell_max, cp.int32)
            self.d_cstart = e(ncell_max + 1, cp.int32)
            self.d_cursor = e(ncell_max, cp.int32)
            self.d_plist = e(ncell_max, cp.int32)
            self.d_npick = e(1, cp.int32)
            self.d_rounds = e(1, cp.int32)
            self.d_ox = e(self.cap, cp.int32)
            self.d_oy = e(self.cap, cp.int32)
            # packed detection result: [count i32, pad, floor f64, kp f32 cap x 2, desc u8 cap x 32]
            self.res_bytes = 16 + 8 * self.cap + 32 * self.cap
            self.d_res = e(self.res_bytes, cp.uint8)
            self.d_count = self.d_res[0:4].view(cp.int32)
            self.d_floor_d = self.d_res[8:16].view(cp.float64)
            self.d_kp = self.d_res[16:16 + 8 * self.cap].view(cp.float32)
            self.d_desc = self.d_res[16 + 8 * self.cap:].view(cp.uint8)
            self.d_rb = e((Hp, Wp), cp.float32)
            self.d_blur = e((Hp, Wp), cp.uint8)
            self.d_kx = e(7, cp.float32)
            self.d_pat = e(256 * 4, cp.int8)
            # stereo: per call [kp f64 2m | lo i64 m | hi i64 m] in, [d | sig | zn | dy f64 m each | ok u8 m] out
            S = STEREO_CAP
            self.st_in_bytes = 32 * S
            self.d_st_in = e(self.st_in_bytes, cp.uint8)
            self.st_out_bytes = 33 * S
            self.d_st_out = e(self.st_out_bytes, cp.uint8)
            # hamming: per call [b1 | b2 | bi i32 L each | ci i32 N] out
            self.d_ham_a = e(LM_CAP * 32, cp.uint8)
            self.d_ham_b = e(max(self.cap, 4096) * 32, cp.uint8)
            self.ham_out_bytes = 4 * (3 * LM_CAP + max(self.cap, 4096))
            self.d_ham_out = e(self.ham_out_bytes, cp.uint8)
        # pinned host buffers
        pm = cp.cuda.alloc_pinned_memory
        self._pins = []

        def pin(nbytes, dtype, shape):
            m = pm(int(nbytes))
            self._pins.append(m)
            return np.frombuffer(m, dtype, int(np.prod(shape))).reshape(shape)

        self.h_left = pin(H * W, np.uint8, (H, W))
        self.h_right = pin(H * W, np.uint8, (H, W))
        self.h_res = pin(self.res_bytes, np.uint8, (self.res_bytes,))
        self.h_st_in = pin(self.st_in_bytes, np.uint8, (self.st_in_bytes,))
        self.h_st_out = pin(self.st_out_bytes, np.uint8, (self.st_out_bytes,))
        self.h_ham_a = pin(LM_CAP * 32, np.uint8, (LM_CAP * 32,))
        self.h_ham_b = pin(max(self.cap, 4096) * 32, np.uint8, (max(self.cap, 4096) * 32,))
        self.h_ham_out = pin(self.ham_out_bytes, np.uint8, (self.ham_out_bytes,))
        self.h_small = pin(64, np.uint8, (64,))
        # calibration (set by calibrate())
        self.dy_vend = (W // 32) * 32
        self.dx_vend = W
        self.blur_vend_r = (Wp // 32) * 32
        self.blur_vend_c = (Wp // 32) * 32
        self.ky = None
        self.calibrated = False
        self.calib = {}
        # residency
        self._left_ref = None
        self._right_ref = None
        self._rs_key = None
        self._desc_ref = None
        self._desc_n = 0
        self.last = {}
        self.verbose = verbose
        self._owner = threading.get_ident()
        self.n_ham_fallback = 0

    # ---------------------------------------------------------------------------------------------------- helpers
    def _launch(self, name, grid, block, args, shared_mem=0):
        self.k[name](grid, block, args, stream=self.stream, shared_mem=shared_mem)

    def _own(self):
        """One stream, one set of buffers: every call must come from the thread that created this front."""
        if threading.get_ident() != self._owner:
            raise RuntimeError("gpu_front: called from a second thread (the GpuFront buffers are single-owner)")

    def _upload_left(self, gray, force=False):
        if gray is self._left_ref and not force:
            return False
        np.copyto(self.h_left, gray)
        self.d_left.set(self.h_left, stream=self.stream)
        self._left_ref = gray
        return True

    def _upload_right(self, right):
        if right is self._right_ref:
            return False
        np.copyto(self.h_right, right)
        self.d_right.set(self.h_right, stream=self.stream)
        self._right_ref = right
        self._rs_key = None
        return True

    # ---------------------------------------------------------------------------------------------------- response
    def _response(self, dy_vend=None, dx_vend=None):
        W, H = self.W, self.H
        s = 1.0 / (4 * 3 * 255.0)
        self._launch("k_sobel", ((W + 127) // 128, H), (128,),
                     (self.d_left, np.int32(H), np.int32(W), np.float32(s), np.float32(2.0 * s),
                      np.int32(self.dy_vend if dy_vend is None else dy_vend),
                      np.int32(self.dx_vend if dx_vend is None else dx_vend), self.d_dx, self.d_dy))
        self._launch("k_boxrows", ((W + 127) // 128, H), (128,), (self.d_dx, self.d_dy, np.int32(H), np.int32(W),
                                                                  self.d_rsum))
        self._launch("k_boxcols", ((3 * W + 63) // 64,), (64,), (self.d_rsum, np.int32(H), np.int32(W), self.d_cov))
        self.cp.cuda.runtime.memsetAsync(self.d_flags.data.ptr, 0, 16, self.stream.ptr)
        self._launch("k_eig", ((H * W + 255) // 256,), (256,), (self.d_cov, np.int32(H * W), self.d_R, self.d_flags))

    def _noise_floor(self, Q):
        W, H = self.W, self.H
        K = float(np.sqrt(np.pi / 2.0) / (6.0 * (W - 2) * (H - 2)))
        self._launch("k_noise_leaves", ((self.n_leaf + 127) // 128,), (128,),
                     (self.d_left, np.int32(W), self.d_lstart, self.d_llen, np.int32(self.n_leaf), self.d_nodeval))
        self._launch("k_noise_tree", (1,), (1024,), (self.d_nodeval, self.d_left_n, self.d_right_n, self.d_lvl,
                                                     np.int32(self.n_lvl), np.int32(self.root), np.float64(K),
                                                     np.float64(Q), self.d_floor_d, self.d_floor_f))

    # ---------------------------------------------------------------------------------------------------- detection
    def detect(self, gray, budget, fe):
        """Detector.__call__ (cell scope, budget n): -> (kp (N,2) float32, desc (N,32) uint8)."""
        cp = self.cp
        W, H = self.W, self.H
        n = int(budget)
        from xclean.stages import features as FE
        if FE._NOISE_Q99 is None:
            FE.noise_floor_8u(np.zeros((8, 8), np.uint8))
        Q = float(FE._NOISE_Q99)
        s = float(fe._cell(W, H, n))
        r = float(fe._min_dist(W, H, n))
        c = int(max(4, round(fe._cell(W, H, n))))
        ncx = int(math.ceil(W / s))
        gh = int(math.ceil(H / s)) + 1
        ncell = ncx * gh
        gwc = -(-W // c)
        ghc = -(-H // c)
        b = FE.KLT_WIN // 2 + 1
        st = self.stream
        self._own()
        # a new frame: the left image ALWAYS goes up (a caller may reuse one buffer for every frame, so object identity
        # cannot prove the device copy current), and the previous frame's right image is no longer valid
        self._upload_left(gray, force=True)
        self._right_ref = None
        self._rs_key = None
        self._response()
        self._noise_floor(Q)
        self._launch("k_cellmax", (gwc * ghc,), (128,), (self.d_R, np.int32(H), np.int32(W), np.int32(c),
                                                         np.int32(gwc), np.float32(FE.SHI_QUALITY), self.d_thr))
        rt = cp.cuda.runtime
        rt.memsetAsync(self.d_ncand.data.ptr, 0, 4, st.ptr)
        rt.memsetAsync(self.d_ccount.data.ptr, 0, 4 * ncell, st.ptr)
        self._launch("k_cand", ((W + 127) // 128, H), (128,),
                     (self.d_R, np.int32(H), np.int32(W), np.int32(b), np.int32(c), np.int32(gwc), self.d_thr,
                      self.d_floor_f, self.d_flags, np.float64(s), np.int32(ncx), self.d_ncand, self.d_cx, self.d_cy,
                      self.d_ckey, self.d_ccell, self.d_ccount))
        self._launch("k_scan", (1,), (1024,), (self.d_ccount, np.int32(ncell), self.d_cstart, self.d_cursor))
        self._launch("k_fill", ((H * W + 255) // 256,), (256,), (self.d_ncand, self.d_ccell, self.d_cx, self.d_cy,
                                                               self.d_ckey, self.d_cstart, self.d_cursor, self.d_lkey,
                                                               self.d_lxy, self.d_lskip))
        shm = 21 * ncell
        if shm > self.greedy_shm_max:
            raise RuntimeError("gpu_front: %d cells exceed the greedy kernel's shared memory" % ncell)
        self._launch("k_greedy", (1,), (1024,),
                     (np.int32(ncell), np.int32(ncx), np.int32(gh), self.d_cstart, self.d_lkey, self.d_lxy,
                      self.d_lskip, np.float64(r * r), self.d_plist, self.d_npick, self.d_rounds),
                     shared_mem=shm)
        self._launch("k_rank", ((ncell + 255) // 256,), (256,), (self.d_plist, self.d_npick, self.d_lkey, self.d_lxy,
                                                                 np.int32(n), self.d_ox, self.d_oy, self.d_count))
        self._launch("k_subpix", ((self.cap + 127) // 128,), (128,), (self.d_R, np.int32(H), np.int32(W),
                                                                      self.d_count, self.d_ox, self.d_oy, self.d_kp))
        self._describe_dev()
        # the "any response > 0" flag rides in the result's pad word (Frontend._response returns before the noise floor,
        # and leaves stats["noise_floor_last"] untouched, when the frame has no positive response)
        rt.memcpyAsync(self.d_res.data.ptr + 4, self.d_flags.data.ptr, 4, rt.memcpyDeviceToDevice, st.ptr)
        self.d_res.get(stream=st, out=self.h_res)       # blocking (default): the ONE stream sync of the frame
        cnt = int(self.h_res[0:4].view(np.int32)[0])
        anypos = int(self.h_res[4:8].view(np.int32)[0])
        floor = float(self.h_res[8:16].view(np.float64)[0])
        kp = self.h_res[16:16 + 8 * cnt].view(np.float32).reshape(cnt, 2).copy()
        o = 16 + 8 * self.cap
        desc = self.h_res[o:o + 32 * cnt].reshape(cnt, 32).copy()
        if anypos:
            fe.stats["noise_floor_last"] = floor
        self._desc_ref = desc
        self._desc_n = cnt
        self.last = {"n": n, "count": cnt, "floor": floor}
        return kp, desc

    def _describe_dev(self):
        Hp, Wp = self.Hp, self.Wp
        self._launch("k_blur_rows", ((Wp + 127) // 128, Hp), (128,),
                     (self.d_left, np.int32(self.H), np.int32(self.W), np.int32(Hp), np.int32(Wp), self.d_kx,
                      np.int32(self.blur_vend_r), self.d_rb))
        self._launch("k_blur_cols", ((Wp + 127) // 128, Hp), (128,),
                     (self.d_rb, np.int32(Hp), np.int32(Wp), self.d_ky, np.int32(self.blur_vend_c), self.d_blur))
        self._launch("k_orb", ((self.cap * 32 + 255) // 256,), (256,),
                     (self.d_blur, np.int32(Wp), self.d_kp, self.d_count, self.d_pat, self.d_desc))

    # ---------------------------------------------------------------------------------------------------- stereo
    def _rs(self, dy):
        key = (id(self._right_ref), float(dy))
        if self._rs_key == key:
            return
        W, H = self.W, self.H
        if dy:
            self._launch("k_shift", ((W + 127) // 128, H), (128,), (self.d_right, np.int32(H), np.int32(W),
                                                                    np.float64(dy), self.d_rs))
        else:
            self._launch("k_tof", ((H * W + 255) // 256,), (256,), (self.d_right, np.int32(H * W), self.d_rs))
        self._rs_key = key

    def stereo(self, left, right, kp, lo, hi, dy=0.0, band=0):
        """stereo1d.stereo_points_band on the device -> (d, sig, ok, zn, ody), host float64 / bool arrays."""
        N = len(kp)
        d = np.zeros(N)
        sg = np.full(N, np.inf)
        ok = np.zeros(N, np.bool_)
        zn = np.zeros(N)
        ody = np.full(N, np.nan)
        if N == 0:
            return d, sg, ok, zn, ody
        st = self.stream
        self._own()
        self._upload_left(left)
        self._upload_right(right)
        self._rs(dy)
        S = STEREO_CAP
        cp = self.cp
        for a in range(0, N, S):
            e = min(N, a + S)
            m = e - a
            hin = self.h_st_in
            hin[0:16 * m].view(np.float64)[:] = kp[a:e].ravel()
            hin[16 * m:24 * m].view(np.int64)[:] = lo[a:e]
            hin[24 * m:32 * m].view(np.int64)[:] = hi[a:e]
            din = self.d_st_in[:32 * m]
            din.set(hin[:32 * m], stream=st)
            dout = self.d_st_out[:33 * m]
            self._launch("k_stereo", (m,), (MAX_DISP,),
                         (self.d_left, self.d_rs, np.int32(self.H), np.int32(self.W), din[0:16 * m].view(cp.float64),
                          din[16 * m:24 * m].view(cp.int64), din[24 * m:32 * m].view(cp.int64), np.int32(m),
                          np.int32(band), np.float64(ZNCC_MIN), np.float64(1.0 + UNIQUENESS),
                          np.float64(LR_THRESHOLD_PX), dout[0:8 * m].view(cp.float64),
                          dout[8 * m:16 * m].view(cp.float64), dout[32 * m:33 * m],
                          dout[16 * m:24 * m].view(cp.float64), dout[24 * m:32 * m].view(cp.float64)))
            ho = self.h_st_out[:33 * m]
            dout.get(stream=st, out=ho)                 # blocking (default): waits for the stream
            d[a:e] = ho[0:8 * m].view(np.float64)
            sg[a:e] = ho[8 * m:16 * m].view(np.float64)
            zn[a:e] = ho[16 * m:24 * m].view(np.float64)
            ody[a:e] = ho[24 * m:32 * m].view(np.float64)
            ok[a:e] = ho[32 * m:33 * m] != 0
        return d, sg, ok, zn, ody

    # ---------------------------------------------------------------------------------------------------- hamming
    def hamming(self, A, B):
        """Row best / second / index (first minimum) of A (L,32) against B (N,32), and the column best index.
        B is the frame's descriptor array returned by detect() -> already on the device (no upload)."""
        cp = self.cp
        st = self.stream
        self._own()
        L, N = len(A), len(B)
        if B is self._desc_ref and N == self._desc_n:
            dB = self.d_desc[:32 * N]
        else:
            nb = 32 * N
            if N > self.h_ham_b.size // 32:
                raise ValueError("hamming: %d keypoint descriptors > capacity" % N)
            self.h_ham_b[:nb] = np.ascontiguousarray(B, np.uint8).ravel()
            self.d_ham_b[:nb].set(self.h_ham_b[:nb], stream=st)
            dB = self.d_ham_b[:nb]
        if L > LM_CAP:
            raise ValueError("hamming: %d landmark descriptors > LM_CAP %d" % (L, LM_CAP))
        Bu = dB.view(cp.uint64)
        self.h_ham_a[:32 * L] = np.ascontiguousarray(A, np.uint8).ravel()
        self.d_ham_a[:32 * L].set(self.h_ham_a[:32 * L], stream=st)
        Au = self.d_ham_a[:32 * L].view(cp.uint64)
        nb = 12 * L + 4 * N
        o = self.d_ham_out[:nb]
        self._launch("k_ham_rows", (L,), (128,), (Au, np.int32(L), Bu, np.int32(N), o[0:4 * L].view(cp.int32),
                                                   o[4 * L:8 * L].view(cp.int32), o[8 * L:12 * L].view(cp.int32)))
        self._launch("k_ham_cols", ((N + 127) // 128,), (128,), (Au, np.int32(L), Bu, np.int32(N),
                                                                 o[12 * L:nb].view(cp.int32)))
        h = self.h_ham_out[:nb]
        o.get(stream=st, out=h)                         # blocking (default): waits for the stream
        b1 = h[0:4 * L].view(np.int32).astype(np.int64)
        b2 = h[4 * L:8 * L].view(np.int32).astype(np.int64)
        bi = h[8 * L:12 * L].view(np.int32).astype(np.int64)
        ci = h[12 * L:nb].view(np.int32).astype(np.int64)
        return b1, b2, bi, ci

    # ---------------------------------------------------------------------------------------------------- calibration
    def calibrate(self, seed=0):
        """Find OpenCV's SIMD/scalar boundaries for this image size and verify every bit-exact claim against cv2 /
        numpy on random data.  Raises RuntimeError if any check fails."""
        import cv2
        cp = self.cp
        W, H = self.W, self.H
        rng = np.random.default_rng(seed)
        img = (rng.random((H, W)) * 255).astype(np.uint8)
        # FIXTURE (calibration image; no estimate depends on its values): blur 1.0 px for realistic gradients (some
        # ties, some flat), a flat corner (grey 77) for exact zeros / subnormal sums; seed 0
        img = cv2.GaussianBlur(img, (0, 0), 1.0)
        img[: H // 8, : W // 8] = 77
        R_cv = cv2.cornerMinEigenVal(img, 3, 3)
        # the SIMD vector-end candidates: multiples of OpenCV's possible vector widths (4..128 lanes, class A: the
        # instruction sets OpenCV dispatches); the calibration picks the one that reproduces cv2 bit for bit
        cands = sorted(set([W] + [(W // m) * m for m in (4, 8, 16, 32, 64, 128)] + [0]), reverse=True)
        found = None
        self._upload_left(img)
        for dxv in (W,) + tuple(c for c in cands if c != W):
            for dyv in cands:
                self._response(dyv, dxv)
                Rg = cp.asnumpy(self.d_R, stream=self.stream)      # blocking copy (waits for the stream)
                if np.array_equal(Rg.view(np.uint32), R_cv.view(np.uint32)):
                    found = (dyv, dxv)
                    break
            if found:
                break
        if not found:
            raise RuntimeError("gpu_front: no SIMD-boundary model reproduces cv2.cornerMinEigenVal on this build")
        self.dy_vend, self.dx_vend = found
        self.calib["response_vend"] = {"dy_row": found[0], "dx_col": found[1], "W": W}
        # noise floor sum vs numpy
        from xclean.stages import features as FE
        FE.noise_floor_8u(img)
        Q = float(FE._NOISE_Q99)
        self._noise_floor(Q)
        fl = cp.asnumpy(self.d_floor_d, stream=self.stream)
        ref = FE.noise_floor_8u(img)
        if float(fl[0]) != float(ref):
            raise RuntimeError("gpu_front: pairwise noise sum differs from numpy (%r vs %r)" % (fl[0], ref))
        self.calib["noise_floor_equal"] = True
        # blur kernels (float, as cv2.getGaussianKernel) and the ORB SIMD boundaries
        kf = cv2.getGaussianKernel(7, 2, ktype=cv2.CV_32F).ravel().astype(np.float32)
        self.d_kx.set(kf)
        self.ky = kf
        self.d_ky = self.d_kx
        pad = cv2.copyMakeBorder(img, ORB_PAD, ORB_PAD, ORB_PAD, ORB_PAD, cv2.BORDER_REFLECT_101)
        B_cv = cv2.sepFilter2D(pad, -1, kf, kf, borderType=cv2.BORDER_REFLECT_101)
        Wp = self.Wp
        bc = sorted(set([Wp] + [(Wp // m) * m for m in (4, 8, 16, 32, 64, 128)] + [0]), reverse=True)
        foundb = None
        for vr in bc:
            for vc in bc:
                self.blur_vend_r, self.blur_vend_c = vr, vc
                Hp = self.Hp
                self._launch("k_blur_rows", ((Wp + 127) // 128, Hp), (128,),
                             (self.d_left, np.int32(H), np.int32(W), np.int32(Hp), np.int32(Wp), self.d_kx,
                              np.int32(vr), self.d_rb))
                self._launch("k_blur_cols", ((Wp + 127) // 128, Hp), (128,),
                             (self.d_rb, np.int32(Hp), np.int32(Wp), self.d_ky, np.int32(vc), self.d_blur))
                Bg = cp.asnumpy(self.d_blur, stream=self.stream)
                if np.array_equal(Bg, B_cv):
                    foundb = (vr, vc)
                    break
            if foundb:
                break
        if not foundb:
            raise RuntimeError("gpu_front: no SIMD-boundary model reproduces cv2.sepFilter2D (ORB blur)")
        self.blur_vend_r, self.blur_vend_c = foundb
        self.calib["blur_vend"] = {"row": foundb[0], "col": foundb[1], "Wp": Wp}
        pat = probe_orb_pattern()
        self.d_pat.set(pat.astype(np.int8).ravel())
        self.pattern = pat
        self.calib["orb_pattern_points"] = int(len(set(map(tuple, pat[:, :2].tolist() + pat[:, 2:].tolist()))))
        # end-to-end ORB check on random sub-pixel keypoints (incl. exact .5 phases)
        nk = min(self.cap, 1500)
        kx = rng.uniform(11, W - 12, nk).astype(np.float32)
        ky = rng.uniform(11, H - 12, nk).astype(np.float32)
        kx[:50] = np.floor(kx[:50]) + np.float32(0.5)
        ky[50:100] = np.floor(ky[50:100]) + np.float32(0.5)
        kpr = np.stack([kx, ky], axis=1)
        dref = cpu_orb(img, kpr)
        dg = self.describe_points(img, kpr)
        if not np.array_equal(dg, dref):
            raise RuntimeError("gpu_front: ORB descriptors differ from cv2 (%d bits)"
                               % int(np.unpackbits(dg ^ dref).sum()))
        self.calib["orb_equal_random"] = nk
        self.calib["lk_solver_equal"] = self.check_lk_solver(rng)
        self.calibrated = True
        self._left_ref = None
        return self.calib

    def check_lk_solver(self, rng, n=1024):
        """The device 3x3 solve / inverse vs numba's np.linalg.solve / inv (the CPU LK's calls) on LK-shaped systems
        (sums of J J^T with J = (-a gx, I, 1) + 1e-9 I; a fifth with a zero gradient column, a fifth with a flat
        patch).  -> n when bit for bit.  Otherwise (e.g. a different LAPACK backend under numba): False, with the
        mismatch count in self.calib and one note printed; RuntimeError with XCLEAN_STRICT_SELFCHECK=1."""
        cp = self.cp
        As = np.empty((n, 3, 3))
        gs = rng.normal(0.0, 1.0, (n, 3)) * (10.0 ** rng.uniform(-6, 4, (n, 1)))
        for k in range(n):
            J0 = -rng.uniform(0.4, 2.2) * rng.normal(0.0, 10.0 ** rng.uniform(-2, 2.5), 121)
            J1 = np.clip(rng.uniform(0, 255) + rng.normal(0.0, 10.0 ** rng.uniform(-1, 1.8), 121), 0, 255)
            if k % 5 == 1:
                J0[:] = 0.0
            elif k % 5 == 2:
                J1[:] = round(J1[0])
            J = np.stack([J0, J1, np.ones(121)], axis=1)
            As[k] = J.T @ J + 1e-9 * np.eye(3)
        Xr = np.empty((n, 3))
        Ir = np.empty((n, 3, 3))
        _nb_solve_inv()(As, gs, Xr, Ir)
        with cp.cuda.using_allocator(self.pool.malloc):           # warmup only; never the CuPy default pool
            dA, dg = cp.asarray(As.reshape(-1)), cp.asarray(gs.reshape(-1))
            dX, dI = cp.empty(3 * n, cp.float64), cp.empty(9 * n, cp.float64)
            self.k["k_lkcheck"](((n + 127) // 128,), (128,), (dA, dg, np.int32(n), dX, dI))
            Xg, Ig = cp.asnumpy(dX).reshape(n, 3), cp.asnumpy(dI).reshape(n, 3, 3)
        del dA, dg, dX, dI
        if not (np.array_equal(Xg.view(np.uint64), Xr.view(np.uint64))
                and np.array_equal(Ig.view(np.uint64), Ir.view(np.uint64))):
            k = int((~(Xg == Xr).all(1) | ~(Ig == Ir).all((1, 2))).sum())
            if strict_selfcheck():
                raise RuntimeError("gpu_front: the device 3x3 solve/inverse differs from numba's LAPACK on %d of %d "
                                   "systems" % (k, n))
            # start-up self-check policy: the device solver is a correct LU solve (OpenBLAS's arithmetic); another
            # LAPACK build differs from it at rounding level only, so the device front stays on and the mismatch is
            # recorded (lk_solver_equal False) and reported once
            self.calib["lk_solver_mismatch"] = k
            self.calib["lk_solver_systems"] = int(n)
            warn_lk_mismatch(k, n)
            return False
        return n

    def describe_points(self, gray, kp):
        """ORB descriptors of given keypoints (test / calibration path; the detector path never leaves the device)."""
        n = len(kp)
        self._upload_left(gray, force=True)
        self._desc_ref = None                 # the device descriptor buffer is overwritten below
        h = self.h_res
        h[0:4].view(np.int32)[0] = n
        h[16:16 + 8 * n].view(np.float32)[:] = np.asarray(kp, np.float32).ravel()
        self.d_res.set(h, stream=self.stream)
        self._describe_dev()
        self.d_res.get(stream=self.stream, out=self.h_res)
        o = 16 + 8 * self.cap
        return self.h_res[o:o + 32 * n].reshape(n, 32).copy()

    def pool_bytes(self):
        return int(self.pool.total_bytes())


# ================================================================================ start-up self-check policy helpers
def strict_selfcheck():
    """XCLEAN_STRICT_SELFCHECK=1: every failed start-up self-check raises (the engine refuses to start), as before the
    fallback policy existed.  Default (unset or 0): see the module docstring, "START-UP SELF-CHECK POLICY"."""
    return (os.environ.get("XCLEAN_STRICT_SELFCHECK") or "").strip() not in ("", "0")


def lk_mismatch_cause(vendor, source, core):
    """Why the CPU LAPACK differs from the GPU front end's 3x3 solver, in a few words, from what xclean doctor detects:
    vendor / source = doctor.blas_origin("scipy")[:2], core = the OpenBLAS CPU kernel name (or None)."""
    from xclean.doctor import MATCHED_OPENBLAS_CORE
    if vendor == "openblas" and source == "pip":
        if core and core.lower() != MATCHED_OPENBLAS_CORE:
            return ("SciPy's OpenBLAS uses its %s kernel on this CPU; the GPU front end matches its kernel for CPUs "
                    "with AVX-512" % core)
        return "SciPy's own OpenBLAS%s; the cause was not identified" % (" (%s kernel)" % core if core else "")
    if vendor == "mkl":
        return "SciPy here uses Intel MKL; the GPU front end matches the OpenBLAS of SciPy's pip package"
    if vendor == "openblas":
        return ("SciPy here uses OpenBLAS from %s, not the build in SciPy's pip package that the GPU front end matches"
                % ("a conda package" if source == "conda" else "another package"))
    return ("SciPy here uses %s LAPACK, not the OpenBLAS of SciPy's pip package that the GPU front end matches"
            % (vendor or "an unidentified"))


def _lk_cause_here():
    """lk_mismatch_cause for this process (the same detection as xclean doctor's LAPACK line)."""
    try:
        from xclean import doctor
        vendor, source, _ev, path = doctor.blas_origin("scipy")
        core = doctor._openblas_corename(path) if (vendor == "openblas" and path) else None
        return lk_mismatch_cause(vendor, source, core)
    except Exception as e:
        return "the LAPACK build could not be identified (%s)" % type(e).__name__


def warn_lk_mismatch(k, n, cause=None):
    """The one message printed when only the 3x3 solver check differs (the GPU front end stays on)."""
    import sys
    print("xclean: note: the GPU front end's 3x3 solver does not match this machine's CPU LAPACK bit for bit (%d of %d "
          "test systems differ at rounding level; %s). The GPU front end stays on; results can differ from a CPU-only "
          "run in the last digits." % (k, n, cause or _lk_cause_here()), file=sys.stderr, flush=True)


def cuda_context_active():
    """True when this process holds a CUDA context: a current context on this thread or an active primary context on
    any visible device.  Asks the driver API directly and creates none; False when CuPy was never imported here."""
    import ctypes
    import sys
    if "cupy" not in sys.modules:
        return False
    try:
        lib = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False
    try:
        if lib.cuInit(0) != 0:
            return False
        cur = ctypes.c_void_p(0)
        if lib.cuCtxGetCurrent(ctypes.byref(cur)) == 0 and cur.value:
            return True
        n = ctypes.c_int(0)
        if lib.cuDeviceGetCount(ctypes.byref(n)) != 0:
            return False
        for i in range(n.value):
            dev = ctypes.c_int(0)
            flags, active = ctypes.c_uint(0), ctypes.c_int(0)
            if lib.cuDeviceGet(ctypes.byref(dev), i) == 0 and \
                    lib.cuDevicePrimaryCtxGetState(dev, ctypes.byref(flags), ctypes.byref(active)) == 0 and active.value:
                return True
    except Exception:
        return False
    return False


CONTEXT_KEPT_NOTE = ("Its CUDA context stays open for the rest of the run (its cached GPU memory was freed): closing it "
                     "while CuPy still holds handles in this process could crash the program later.")


def release_device(cp=None, context_active=None):
    """After a failed start-up self-check (the engine falls back to the CPU front): drop every front of this module and
    free CuPy's cached device and pinned memory.  A CUDA context that already exists is KEPT: CuPy keeps compiled
    modules, streams and library handles for the life of the process, and their destructors would call into a
    destroyed context (CuPy 13 has no supported way to reset the device under them).  Errors are recorded, not
    hidden.  cp / context_active: stand-ins for tests (default: the imported CuPy, cuda_context_active).
    -> {"cuda_context_released": no context left, "gpu_memory_freed": True / False / None (no CuPy),
        "note": plain text when a context is kept, "errors": [text, ...]}"""
    import gc
    import sys
    _FRONT.clear()
    gc.collect()
    if cp is None:
        cp = sys.modules.get("cupy")
    active = context_active or cuda_context_active
    out = {"cuda_context_released": True, "gpu_memory_freed": None, "note": None, "errors": []}
    if cp is None:
        return out                                   # CuPy was never imported here: there is no context
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        out["gpu_memory_freed"] = True
    except Exception as e:
        out["gpu_memory_freed"] = False
        out["errors"].append("freeing CuPy's memory pools: %s: %s" % (type(e).__name__, e))
    try:
        held = bool(active())
    except Exception as e:
        held = True
        out["errors"].append("asking the driver for a CUDA context: %s: %s" % (type(e).__name__, e))
    if held:
        out["cuda_context_released"] = False
        out["note"] = CONTEXT_KEPT_NOTE
    return out


_NB_SOLVE_INV = None


def _nb_solve_inv():
    """numba's np.linalg.solve / np.linalg.inv, the calls stereo1d._lk_v makes (same LAPACK binding)."""
    global _NB_SOLVE_INV
    if _NB_SOLVE_INV is None:
        from numba import njit

        @njit(cache=True)
        def f(As, gs, X, Inv):
            for k in range(As.shape[0]):
                A = As[k].copy()
                X[k] = np.linalg.solve(A, gs[k].copy())
                Inv[k] = np.linalg.inv(As[k].copy())
        _NB_SOLVE_INV = f
    return _NB_SOLVE_INV


# ======================================================================================== ORB pattern by probing cv2
_PATTERN = None


def _orb():
    """The engine's ORB extractor (carried: the same cv2.ORB_create call as stages/features.Frontend; nfeatures and
    scaleFactor are inert for compute() with nlevels = 1)."""
    import cv2
    return cv2.ORB_create(nfeatures=100000, scaleFactor=1.2, nlevels=1, edgeThreshold=ORB_PATCH, firstLevel=0,
                          WTA_K=2, patchSize=ORB_PATCH)


def cpu_orb(gray, kp):
    """The engine's describe (Frontend.describe): reflect-101 pad, cv2.KeyPoint per point, class_id map."""
    import cv2
    pad = cv2.copyMakeBorder(gray, ORB_PAD, ORB_PAD, ORB_PAD, ORB_PAD, cv2.BORDER_REFLECT_101)
    kps = [cv2.KeyPoint(float(x) + ORB_PAD, float(y) + ORB_PAD, ORB_PATCH, 0.0, 0.0, 0, int(i))
           for i, (x, y) in enumerate(kp)]
    out_k, desc = _orb().compute(pad, kps)
    res = np.zeros((len(kp), 32), np.uint8)
    if desc is not None and len(out_k):
        cid = np.array([k.class_id for k in out_k], np.int64)
        res[cid] = desc
    return res


def probe_orb_pattern():
    """Recover the 256 (x1, y1, x2, y2) test pairs of the cv2 ORB we call (no orientation, 31-px patch) from its own
    outputs: vertical / horizontal line probes (positive and negative) fix every coordinate that differs within a
    pair; point probes fix the shared one.  Bit i of byte b is pair 8 b + i.  -> int array (256, 4)."""
    global _PATTERN
    if _PATTERN is not None:
        return _PATTERN.copy()
    import cv2
    # FIXTURE geometry of the probe canvas (no descriptor depends on it beyond covering the pattern): pair coordinates
    # lie within +-ORB_PATCH // 2 = 15 < Rr = 17 (decode range); line / point offsets probed over +-S = 21; tile
    # T = 64 > 2 S + the 7x7 blur; margin M = 48 >= ORB_PAD so no probe touches the canvas border
    T, M, Rr, S = 64, 48, 17, 21
    c = T // 2
    tiles = []
    ss = np.arange(-S, S + 1)
    for kind in ("v+", "v-", "h+", "h-"):
        for s in ss:
            t = np.full((T, T), 255 if kind[1] == "-" else 0, np.uint8)
            if kind[0] == "v":
                t[:, c + s] = 0 if kind[1] == "-" else 255
            else:
                t[c + s, :] = 0 if kind[1] == "-" else 255
            tiles.append(t)
    for sy in ss:
        for sx in ss:
            t = np.zeros((T, T), np.uint8)
            t[c + sy, c + sx] = 255
            tiles.append(t)
    n = len(tiles)
    nc = int(np.ceil(np.sqrt(n)))
    nr = int(np.ceil(n / nc))
    canvas = np.zeros((2 * M + nr * T, 2 * M + nc * T), np.uint8)
    kps = []
    for i, t in enumerate(tiles):
        r_, c_ = divmod(i, nc)
        y0, x0 = M + r_ * T, M + c_ * T
        canvas[y0:y0 + T, x0:x0 + T] = t
        kps.append(cv2.KeyPoint(float(x0 + c), float(y0 + c), ORB_PATCH, 0.0, 0.0, 0, i))
    out_k, desc = _orb().compute(canvas, kps)
    cid = np.array([k.class_id for k in out_k])
    if len(cid) != n:
        raise RuntimeError("probe_orb_pattern: cv2 dropped probe keypoints")
    res = np.zeros((n, 32), np.uint8)
    res[cid] = desc
    bits = np.unpackbits(res, axis=1, bitorder="little").astype(np.int8)     # (n, 256)
    k = len(ss)
    vx, nx, hy, ny = bits[:k], bits[k:2 * k], bits[2 * k:3 * k], bits[3 * k:4 * k]
    pt = bits[4 * k:].reshape(k, k, 256)
    # predicted line-probe bits for every candidate pair (a, b): blur bump strictly decreasing in |offset| over 0..3
    cand = np.array([(a, b) for a in range(-Rr, Rr + 1) for b in range(-Rr, Rr + 1)])
    da = np.abs(cand[:, :1] - ss[None, :])
    db = np.abs(cand[:, 1:] - ss[None, :])
    pred = np.concatenate([(db <= 3) & (db < da), (da <= 3) & (da < db)], axis=1).astype(np.int8)   # (C, 2k)

    def decode(pos, neg):
        obs = np.concatenate([pos, neg], axis=0).T                            # (256, 2k)
        eq = (obs[:, None, :] == pred[None, :, :]).all(axis=2)                # (256, C)
        return eq

    EX = decode(vx, nx)
    EY = decode(hy, ny)
    pairs = np.zeros((256, 4), np.int64)
    for i in range(256):
        xs = cand[EX[i]]
        ys = cand[EY[i]]
        if len(ys) == 1:
            y1, y2 = ys[0]
        elif len(ys) and (ys[:, 0] == ys[:, 1]).all():
            y1 = y2 = None
        else:
            raise RuntimeError("probe_orb_pattern: pair %d y undecidable" % i)
        if len(xs) == 1:
            x1, x2 = xs[0]
        elif len(xs) and (xs[:, 0] == xs[:, 1]).all():
            x1 = x2 = None
        else:
            raise RuntimeError("probe_orb_pattern: pair %d x undecidable" % i)
        if x1 is None and y1 is None:
            raise RuntimeError("probe_orb_pattern: pair %d degenerate" % i)
        if x1 is None:
            on = ss[pt[list(ss).index(int(y2)), :, i] == 1]
            x1 = x2 = int(round((on.min() + on.max()) / 2.0))
        if y1 is None:
            on = ss[pt[:, list(ss).index(int(x2)), i] == 1]
            y1 = y2 = int(round((on.min() + on.max()) / 2.0))
        pairs[i] = (x1, y1, x2, y2)
    _PATTERN = pairs
    return pairs.copy()


# ======================================================================================== engine-facing API
_FRONT = {}                           # (W, H) -> GpuFront: ONE front (one buffer set, one stream) per image size


def front_for(W, H, cap=2048):
    """The front for this image size; a larger keypoint cap than the cached front's replaces it (the cap is a buffer
    size, never a result), so the detector, the stereo and stage A always share one front."""
    key = (int(W), int(H))
    f = _FRONT.get(key)
    if f is None or f.cap < int(cap):
        f = GpuFront(W, H, max(int(cap), 2048 if f is None else f.cap))
        f.calibrate()
        _FRONT[key] = f
    return f


def _front_any():
    if not _FRONT:
        raise RuntimeError("gpu_front: no GpuFront yet (create the Detector or call warmup(rig) first)")
    return next(iter(_FRONT.values()))


from xclean.front.detect import Detector as _CpuDetector        # noqa: E402


class Detector(_CpuDetector):
    """front/detect.Detector with the extraction on the device.  Budget logic, stats and fe are the CPU class's.
    fe.match_desc is replaced by the device Hamming when hamming=True (identical arrays to cv2's)."""

    def __init__(self, rig, cap=2048, window=10, hamming=True):
        super().__init__(rig, cap=cap, window=window)
        self.gpu = front_for(rig.width, rig.height, max(cap, 2048))
        if hamming:
            self.fe.match_desc = self._match_desc

    def __call__(self, gray):
        n = self.budget
        kp, desc = self.gpu.detect(gray, n, self.fe)
        self.fe.stats["detected"] += int(len(kp))
        self.stats["frames"] += 1
        self.stats["detected"] += int(len(kp))
        return kp, desc

    def _match_desc(self, f0, f1, mask0=None, mask1=None):
        return match_desc_gpu(self.gpu, f0, f1, mask0, mask1)


def match_desc_gpu(gf, f0, f1, mask0=None, mask1=None):
    """Frontend.match_desc semantics (mutual NN + Lowe ratio + Hamming chance bound) with the device Hamming."""
    from xclean.stages.features import HAMMING_CHANCE, RATIO
    d0, d1 = f0["descriptors"], f1["descriptors"]
    i0 = np.arange(len(d0)) if mask0 is None else np.nonzero(mask0)[0]
    i1 = np.arange(len(d1)) if mask1 is None else np.nonzero(mask1)[0]
    z = np.zeros(0, np.int64)
    if len(i0) < 2 or len(i1) < 2:
        return z, z
    if len(i0) > LM_CAP or len(i1) > gf.h_ham_b.size // 32:
        from xclean.front.mapmatch_fast import match_desc as _nb_match_desc      # same arrays (tests), no capacity
        gf.n_ham_fallback += 1
        return _nb_match_desc(f0, f1, mask0, mask1)
    A = d0 if mask0 is None else np.ascontiguousarray(d0[i0])
    B = d1 if mask1 is None else np.ascontiguousarray(d1[i1])
    b1, b2, bi, ci = gf.hamming(np.ascontiguousarray(A), B)
    q = np.arange(len(A))
    ok = (bi >= 0) & (b1 <= HAMMING_CHANCE) & (b1 < RATIO * b2)
    ok &= ci[np.maximum(bi, 0)] == q
    r0 = q[ok]
    if not len(r0):
        return z, z
    return i0[r0].astype(np.int64), i1[bi[ok]].astype(np.int64)


def disparities(left, right, kp, dlo=None, dhi=None, dy=0.0, band=0):
    """front/stereo1d.disparities on the device (same signature, same outputs; sets last_dy / last_amb / last_span)."""
    kp = np.ascontiguousarray(kp, np.float64).reshape(-1, 2)
    n = len(kp)
    lo = np.zeros(n, np.int64) if dlo is None else np.ascontiguousarray(dlo, np.int64)
    hi = np.full(n, MAX_DISP - 1, np.int64) if dhi is None else np.ascontiguousarray(dhi, np.int64)
    H, W = left.shape
    gf = _FRONT.get((int(W), int(H))) or front_for(W, H)
    # non-finite or off-image coordinates never reach the kernel's int conversion (an overflowed index is an illegal
    # address that poisons the context): they get the border-reject result (d 0, sigma inf, not ok, zncc -2), which
    # is what stereo1d returns for any point off the image (verifier probes nf_inf / nf_1e300 / nf_3e9)
    bad = ~(np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1]) & (kp[:, 0] >= 0) & (kp[:, 0] < W) & (kp[:, 1] >= 0)
            & (kp[:, 1] < H))
    if bad.any():
        kp = kp.copy()
        kp[bad] = -float(HALF + 3)          # off the image: rejected by the kernel's own border test
    d, s, ok, zn, ody = gf.stereo(left, right, kp, lo, hi, float(dy), int(band))
    disparities.last_amb = np.zeros(n, np.bool_)
    disparities.last_span = np.zeros(n)
    disparities.last_dy = ody + float(dy)
    return d, s, ok, zn


def disparities_ex(left, right, kp, dlo=None, dhi=None, dy=0.0, band=0):
    """-> (d, sigma, ok, amb, span, zncc) as stereo1d.disparities_ex."""
    d, s, ok, zn = disparities(left, right, kp, dlo, dhi, dy, band)
    return d, s, ok, disparities.last_amb, disparities.last_span, zn


def warmup(rig):
    """Create the device front for the rig's size, calibrate + self-check against cv2/numpy, and run one detection,
    one stereo call and one Hamming call on synthetic data (before the feeder's clock starts).  -> calibration dict."""
    W, H = int(rig.width), int(rig.height)
    gf = front_for(W, H)
    rng = np.random.default_rng(0)
    import cv2
    img = cv2.GaussianBlur((rng.random((H, W)) * 255).astype(np.uint8), (0, 0), 1.0)
    from xclean.stages.features import Frontend
    fe = Frontend(rig, cap=2048, verbose=False)
    fe.detect_mode = "cell"
    kp, desc = gf.detect(img, 1500, fe)
    disparities(img, np.roll(img, -8, axis=1), kp[:64].astype(np.float64))
    disparities(img, np.roll(img, -8, axis=1), kp[:64].astype(np.float64), dy=0.25, band=1)
    if len(desc) >= 2:
        match_desc_gpu(gf, {"descriptors": desc[::-1].copy()}, {"descriptors": desc})
    gf._left_ref = gf._right_ref = gf._desc_ref = None
    return dict(gf.calib, pool_bytes=gf.pool_bytes())
