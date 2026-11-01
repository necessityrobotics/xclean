# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Fused CUDA backend of the affine patch aligner (xclean/stages/align.py is the numpy reference; same model, same
acceptance).  One launch per pyramid level; one block per point; every Gauss-Newton iteration inside the kernel; the
8x8 normal equations solved by Cholesky in thread 0.  Tested for equality of results against the reference (tests)."""
from __future__ import annotations

import numpy as np

from xclean.stages.align import ITERS, CONV_PX, _pyramid
from xclean.stages.features import KLT_WIN, KLT_LEVELS
from xclean.stereo.epipolar import ZNCC_MIN

_SRC = r"""
#define NP 8
#define NS 44
__device__ __forceinline__ float bil(const float* img, int H, int W, float x, float y, float* gx, float* gy, int* in) {
    *in = (x >= 0.f && x <= W - 1.f && y >= 0.f && y <= H - 1.f);
    float fx = floorf(x), fy = floorf(y);
    float ax = x - fx, ay = y - fy;
    int x0 = min(max((int)fx, 0), W - 2), y0 = min(max((int)fy, 0), H - 2);
    float a = img[y0 * W + x0], b = img[y0 * W + x0 + 1], c = img[(y0 + 1) * W + x0], d = img[(y0 + 1) * W + x0 + 1];
    float top = a + ax * (b - a), bot = c + ax * (d - c);
    *gx = (b - a) * (1.f - ay) + (d - c) * ay;
    *gy = (c - a) * (1.f - ax) + (d - b) * ax;
    return top + ay * (bot - top);
}

extern "C" __global__ void align_level(const float* ref, int HR, int WR, const float* cur, int HC, int WC,
                                       const float* rp, float* cp, float* A, float* gb, int* conv,
                                       int n, int h, float s, int iters, float conv_px) {
    int k = blockIdx.x;
    if (k >= n) return;
    int tid = threadIdx.x, nt = blockDim.x;
    int side = 2 * h + 1, np_ = side * side;
    extern __shared__ float sm[];
    float* T = sm;                      // np_
    float* red = sm + np_;              // NS * nt
    __shared__ float P[8];              // cx, cy, a11, a12, a21, a22, g, b
    __shared__ float Hs[64], gs[8], d[8];
    __shared__ int tin_all, done, failed;
    if (tid == 0) {
        P[0] = cp[2 * k] * s; P[1] = cp[2 * k + 1] * s;
        P[2] = A[4 * k]; P[3] = A[4 * k + 1]; P[4] = A[4 * k + 2]; P[5] = A[4 * k + 3];
        P[6] = gb[2 * k]; P[7] = gb[2 * k + 1];
        tin_all = 1; done = 0; failed = 0;
    }
    __syncthreads();
    float rx = rp[2 * k] * s, ry = rp[2 * k + 1] * s;
    for (int i = tid; i < np_; i += nt) {
        float gx_ = (float)(i % side - h), gy_ = (float)(i / side - h);
        float dx, dy; int in;
        T[i] = bil(ref, HR, WR, rx + gx_, ry + gy_, &dx, &dy, &in);
        if (!in) atomicAnd(&tin_all, 0);
    }
    __syncthreads();
    for (int it = 0; it < iters; it++) {
        float acc[NS];
        for (int q = 0; q < NS; q++) acc[q] = 0.f;
        for (int i = tid; i < np_; i += nt) {
            float gx_ = (float)(i % side - h), gy_ = (float)(i / side - h);
            float wx = P[0] + P[2] * gx_ + P[3] * gy_, wy = P[1] + P[4] * gx_ + P[5] * gy_;
            float Ix, Iy; int in;
            float I = bil(cur, HC, WC, wx, wy, &Ix, &Iy, &in);
            if (!in) continue;
            float t = T[i];
            float r = I - (P[6] * t + P[7]);
            float J[8] = {Ix, Iy, Ix * gx_, Ix * gy_, Iy * gx_, Iy * gy_, -t, -1.f};
            int q = 0;
            for (int a = 0; a < 8; a++) for (int b = a; b < 8; b++) acc[q++] += J[a] * J[b];
            for (int a = 0; a < 8; a++) acc[36 + a] += J[a] * r;
        }
        for (int q = 0; q < NS; q++) red[q * nt + tid] = acc[q];
        __syncthreads();
        for (int st = nt / 2; st > 0; st >>= 1) {
            if (tid < st) for (int q = 0; q < NS; q++) red[q * nt + tid] += red[q * nt + tid + st];
            __syncthreads();
        }
        if (tid == 0) {
            int q = 0;
            for (int a = 0; a < 8; a++) for (int b = a; b < 8; b++) { Hs[a * 8 + b] = red[q * nt]; Hs[b * 8 + a] = red[q * nt]; q++; }
            for (int a = 0; a < 8; a++) { gs[a] = red[(36 + a) * nt]; Hs[a * 8 + a] += 1e-6f; }
            // Cholesky solve H d = -g
            float L[64];
            int ok = 1;
            for (int i = 0; i < 8 && ok; i++) {
                for (int j = 0; j <= i; j++) {
                    float sum = Hs[i * 8 + j];
                    for (int m = 0; m < j; m++) sum -= L[i * 8 + m] * L[j * 8 + m];
                    if (i == j) { if (sum <= 0.f) { ok = 0; break; } L[i * 8 + i] = sqrtf(sum); }
                    else L[i * 8 + j] = sum / L[j * 8 + j];
                }
            }
            if (ok) {
                float y[8];
                for (int i = 0; i < 8; i++) { float sum = -gs[i]; for (int m = 0; m < i; m++) sum -= L[i * 8 + m] * y[m]; y[i] = sum / L[i * 8 + i]; }
                for (int i = 7; i >= 0; i--) { float sum = y[i]; for (int m = i + 1; m < 8; m++) sum -= L[m * 8 + i] * d[m]; d[i] = sum / L[i * 8 + i]; }
                for (int i = 0; i < 8; i++) P[i] += d[i];
                float step = sqrtf(d[0] * d[0] + d[1] * d[1]) / s;
                done = (step <= conv_px);
            } else { done = 1; failed = 1; }
        }
        __syncthreads();
        if (done) break;
    }
    if (tid == 0) {
        cp[2 * k] = P[0] / s; cp[2 * k + 1] = P[1] / s;
        A[4 * k] = P[2]; A[4 * k + 1] = P[3]; A[4 * k + 2] = P[4]; A[4 * k + 3] = P[5];
        gb[2 * k] = P[6]; gb[2 * k + 1] = P[7];
        conv[k] = done && tin_all && !failed;
    }
}

extern "C" __global__ void zncc_at(const float* ref, int HR, int WR, const float* cur, int HC, int WC,
                                   const float* rp, const float* cp, const float* A, float* zn, int* inside,
                                   int n, int h) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int side = 2 * h + 1, np_ = side * side;
    float st = 0, si = 0, stt = 0, sii = 0, sti = 0; int allin = 1;
    for (int i = 0; i < np_; i++) {
        float gx_ = (float)(i % side - h), gy_ = (float)(i / side - h);
        float dx, dy; int in1, in2;
        float t = bil(ref, HR, WR, rp[2 * k] + gx_, rp[2 * k + 1] + gy_, &dx, &dy, &in1);
        float wx = cp[2 * k] + A[4 * k] * gx_ + A[4 * k + 1] * gy_, wy = cp[2 * k + 1] + A[4 * k + 2] * gx_ + A[4 * k + 3] * gy_;
        float I = bil(cur, HC, WC, wx, wy, &dx, &dy, &in2);
        allin &= (in1 & in2);
        st += t; si += I; stt += t * t; sii += I * I; sti += t * I;
    }
    float N = (float)np_;
    float ct = stt - st * st / N, ci = sii - si * si / N, cti = sti - st * si / N;
    zn[k] = cti / fmaxf(sqrtf(fmaxf(ct, 0.f) * fmaxf(ci, 0.f)), 1e-6f);
    inside[k] = allin;
}
"""


class CudaPatchAligner:
    def __init__(self, win=KLT_WIN, levels=KLT_LEVELS, threads=128):
        import cupy as cp
        self.cp = cp
        mod = cp.RawModule(code=_SRC)
        self._k = mod.get_function("align_level")
        self._z = mod.get_function("zncc_at")
        self.h = win // 2
        self.levels = int(levels)
        self.nt = int(threads)

    def pyramid(self, img):
        return [self.cp.asarray(np.ascontiguousarray(a, np.float32)) for a in _pyramid(img, self.levels)]

    def align(self, ref_img, ref_pts, cur_img, init_pts, A0=None, radius=None, ref_pyr=None, cur_pyr=None):
        cp = self.cp
        N = len(ref_pts)
        if N == 0:
            return (np.zeros((0, 2), np.float32), np.zeros((0, 2, 2)), np.zeros(0, bool), np.zeros(0))
        rp = ref_pyr or self.pyramid(ref_img)
        cq = cur_pyr or self.pyramid(cur_img)
        r = cp.asarray(np.ascontiguousarray(ref_pts, np.float32))
        c = cp.asarray(np.ascontiguousarray(init_pts, np.float32)).copy()
        A = cp.asarray(np.ascontiguousarray(np.tile(np.eye(2, dtype=np.float32), (N, 1, 1)) if A0 is None
                                            else np.asarray(A0, np.float32))).reshape(N, 4).copy()
        gb = cp.asarray(np.tile(np.array([1.0, 0.0], np.float32), (N, 1)))
        conv = cp.zeros(N, np.int32)
        np_ = (2 * self.h + 1) ** 2
        shm = (np_ + 44 * self.nt) * 4
        for L in range(self.levels, -1, -1):
            s = np.float32(0.5 ** L)
            R, C = rp[L], cq[L]
            self._k((N,), (self.nt,), (R, np.int32(R.shape[0]), np.int32(R.shape[1]), C, np.int32(C.shape[0]),
                                       np.int32(C.shape[1]), r, c, A, gb, conv, np.int32(N), np.int32(self.h), s,
                                       np.int32(ITERS), np.float32(CONV_PX)), shared_mem=shm)
        zn = cp.empty(N, np.float32)
        ins = cp.empty(N, np.int32)
        self._z(((N + 127) // 128,), (128,), (rp[0], np.int32(rp[0].shape[0]), np.int32(rp[0].shape[1]), cq[0],
                                              np.int32(cq[0].shape[0]), np.int32(cq[0].shape[1]), r, c, A, zn, ins,
                                              np.int32(N), np.int32(self.h)))
        Ah = A.get().reshape(N, 2, 2)
        ch = c.get()
        znh = zn.get()
        det = Ah[:, 0, 0] * Ah[:, 1, 1] - Ah[:, 0, 1] * Ah[:, 1, 0]
        ok = (conv.get() > 0) & (ins.get() > 0) & (det > 0) & (znh >= ZNCC_MIN) & np.isfinite(ch).all(axis=1)
        if radius is not None:
            ok &= np.linalg.norm(ch - np.asarray(init_pts, np.float32), axis=1) <= np.asarray(radius, np.float32)
        return ch.astype(np.float32), Ah, ok, znh
