#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""32-DIRECTION SGM - our dense stereo, as a library (CuPy raw CUDA kernels).

    sgm = SGM32(width, height)
    disp, conf = sgm.compute(rect_left_u8, rect_right_u8)     # numpy float32, 0 = INVALID

WHAT IT IS, NAMED HONESTLY
    The source is an earlier in-house prototype ("32-path MGM", July 2026, our own CuPy kernels).  Its aggregation is
    Hirschmueller's semi-global matching recursion with ONE predecessor per path, run along 32 directed paths
    (axis, diagonal, knight (1,2)/(2,1), and the wider (1,3)/(3,1)/(2,3)/(3,2) steps).  Facciolo's MGM uses two
    predecessors per recursion; this is not that.  It is 32-direction SGM, and that is its name here.

    Pipeline (rectified uint8 in, nothing else):
        census 9x9 (80 comparisons, TWO 64-bit words)  ->  Hamming cost volume (uint8)
        ->  ONE 32-direction aggregation (uint16, exact)  ->  left WTA + uniqueness + parabola sub-pixel
        ->  right WTA read off the SAME volume's diagonal  ->  left-right check  ->  disparity, 0 where invalid.
    No hole filling.  An invalid pixel is 0 and stays 0: the consumer must treat it as "no measurement".
    (The prototype's WLS fill is the measured edge-fattening source, -5.2 % depth at 4-6 m on V1_02; not ported.)

THE FOUR VERIFIED DEFECTS OF THE PROTOTYPE, AND WHAT THIS MODULE DOES INSTEAD (each was checked by a test)
    (a) CENSUS OVERFLOW.  The prototype packed the 9x9 census (81 positions, centre skipped but still counted) into ONE
        64-bit word with `<< bit` for bit up to 80: shifts >= 64 are lost, so 17 of the 80 comparisons never
        reached the cost (63 effective bits, the bottom two rows of the window ignored: a vertically ASYMMETRIC
        support).  Here: bits 0..63 in word 0 and 64..79 in word 1, the centre skipped WITHOUT consuming a bit;
        cost = popc(w0 ^ w0') + popc(w1 ^ w1') in [0, 80].
    (b) TWO AGGREGATIONS PER FRAME.  The prototype aggregated a left volume and a separately built right volume (the
        right map only feeds the LR check): ~45 % of the time.  Here the right disparity of pixel (x_r, y) is the
        WTA of S(y, x_r + d, d) over d, read off the ONE left-referenced aggregated volume (the standard SGM
        form: C_R(x_r, d) = C_L(x_r + d, d), so the right image's matching costs are already in the volume).
        One aggregation per frame, counted by `self.n_aggregations`.
    (c) uint32 VOLUMES.  The census cost is <= 80, so the cost volume is uint8 (exact).  A path cost is
        L_r(p,d) = C(p,d) + min(...) - min_k L_r(p-r,k) <= C_max + P2_max (the min term exceeds min_k by at most
        P2), so the 32-path sum is <= 32 (80 + P2_max).  With the penalties below that is 20,064 < 65,535: the
        aggregated volume is uint16 and EXACT (asserted at construction from the actual parameters).
    (d) HOLES BLENDED INTO FAR POINTS.  the engine's `_create` bilinear-sampled the dense map; with LR holes = 0 a
        sample next to a hole is a blend with zero, the disparity under-read and the depth OVER-read (~3x for
        d < 3 px).  That fix lives in the CONSUMER (xclean/track/tracker.py `_create`: any invalid neighbour of
        the bilinear quad = no measurement); this module's part is the contract: 0 means invalid, never "far".

KEPT FROM THE PROTOTYPE, DELIBERATELY
    * the 32 directions and their penalty scaling (axis and diagonal unscaled; (1,2)-family x 9/4, (1,3)-family
      x 19/6, (2,3)-family x 18/5, integer arithmetic exactly as the prototype's SCALE_P);
    * MAX_DISP = 128 (= xclean.core.derive.NUM_DISPARITIES, so the derived near clip is unchanged);
    * uniqueness 5 % against the best non-adjacent disparity, parabola sub-pixel, d < 0.5 px invalid,
      LR threshold 1.0 px, confidence = 1 - min/second.
CHANGED, AND WHY (principled, not tuned)
    * P1, P2 are the prototype's (10, 120) RE-EXPRESSED ON THE CENSUS SCALE: its cost range was 0..63 (the
      effective bits); with all 80 comparisons the same smoothness-to-data ratio is P = P_prototype x 80 / 63
      -> P1 = 13, P2 = 152.  A penalty is only meaningful relative to the data term it competes with.
    * out-of-range cost (x - d < 0) = the census maximum (80, "maximally dissimilar"), not 255; the left WTA only
      considers d <= x (a disparity whose right pixel is outside the image is not a measurement).
    * ONE launch aggregates all 32 directions: every path is a warp (32 lanes x 4 disparities, warp shuffles,
      no shared memory, no __syncthreads); paths of different directions add into the same uint16 volume through
      32-bit atomicAdd on PAIRS of uint16 (no carry can cross the halves because each half is bounded by 20,064).
      Integer addition is associative and commutative, so the volume is bit-identical whatever the schedule.

PURE STAGE: it reads no pose, no IMU and no map, so it declares `key()` for the content-addressed cache.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

MAX_DISP = 128
CENSUS_HALF = 4                         # 9x9 window
CENSUS_BITS = (2 * CENSUS_HALF + 1) ** 2 - 1      # 80 comparisons
PROTOTYPE_EFFECTIVE_BITS = 63           # the prototype's census after the 64-bit overflow
PROTOTYPE_P1, PROTOTYPE_P2 = 10, 120
P1 = int(round(PROTOTYPE_P1 * CENSUS_BITS / PROTOTYPE_EFFECTIVE_BITS))     # 13
P2 = int(round(PROTOTYPE_P2 * CENSUS_BITS / PROTOTYPE_EFFECTIVE_BITS))     # 152
UNIQUENESS = 0.05
LR_THRESHOLD_PX = 1.0
MIN_DISP_PX = 0.5

# the prototype's penalty scale (numerator, denominator) per family; axis and diagonal paths are unscaled there
_FAMILIES = (((1, 1), (1, 1)), ((1, 2), (9, 4)), ((2, 1), (9, 4)), ((1, 3), (19, 6)), ((3, 1), (19, 6)),
             ((2, 3), (18, 5)), ((3, 2), (18, 5)))


def directions():
    """[(dx, dy, num, den)] - the 32 directed paths: the 4 axis paths, then every sign combination of the 7
    oblique families (4 + 7 x 4 = 32)."""
    out = [(1, 0, 1, 1), (-1, 0, 1, 1), (0, 1, 1, 1), (0, -1, 1, 1)]
    for (ax, ay), (n, d) in _FAMILIES:
        for sx in (1, -1):
            for sy in (1, -1):
                out.append((sx * ax, sy * ay, n, d))
    assert len(out) == 32 and len({(a, b) for a, b, _n, _d in out}) == 32
    return out


def path_starts(H, W, dx, dy):
    """Pixels whose predecessor p - (dx, dy) lies outside the image: each pixel lies on exactly one path of a
    direction, and its path starts at the unique such pixel reached by stepping back.  -> (n, 2) int32 (x, y)."""
    ys, xs = np.mgrid[0:H, 0:W]
    px, py = xs - dx, ys - dy
    m = (px < 0) | (px >= W) | (py < 0) | (py >= H)
    return np.stack([xs[m], ys[m]], axis=1).astype(np.int32)


KERNELS = r"""
#define MAX_DISP 128
#define CBITS 80

extern "C" __global__ void census9(const unsigned char* __restrict__ img,
                                   unsigned long long* __restrict__ w0,
                                   unsigned short* __restrict__ w1,
                                   int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    unsigned char c = img[y * W + x];
    unsigned long long a = 0ULL;
    unsigned int b = 0u;
    int bit = 0;
    for (int dy = -4; dy <= 4; dy++) {
        int ny = y + dy;
        for (int dx = -4; dx <= 4; dx++) {
            if (dx == 0 && dy == 0) continue;          /* the centre is not a comparison and takes no bit */
            int nx = x + dx;
            unsigned char v = (nx >= 0 && nx < W && ny >= 0 && ny < H) ? img[ny * W + nx] : c;
            unsigned int s = (v < c) ? 1u : 0u;
            if (bit < 64) a |= ((unsigned long long)s) << bit;
            else          b |= s << (bit - 64);
            bit++;
        }
    }
    w0[y * W + x] = a;
    w1[y * W + x] = (unsigned short)b;
}

extern "C" __global__ void cost_volume(const unsigned long long* __restrict__ L0,
                                       const unsigned short* __restrict__ L1,
                                       const unsigned long long* __restrict__ R0,
                                       const unsigned short* __restrict__ R1,
                                       unsigned char* __restrict__ C, int H, int W) {
    /* one thread per (pixel, disparity): C[(y*W+x)*128 + d] */
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)H * W * MAX_DISP;
    if (i >= n) return;
    int d = (int)(i & (MAX_DISP - 1));
    long p = i >> 7;
    int x = (int)(p % W);
    int rx = x - d;
    if (rx < 0) { C[i] = (unsigned char)CBITS; return; }
    long q = p - d;
    int c = __popcll(L0[p] ^ R0[q]) + __popc(((unsigned int)L1[p]) ^ ((unsigned int)R1[q]));
    C[i] = (unsigned char)c;
}

__device__ __forceinline__ int wmin(int v) {
    v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
    v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
    v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

/* All 32 directions in ONE launch.  Warp w walks path w: start (sx[w], sy[w]) of direction dir[w].  Lane l holds
   disparities 4l..4l+3.  S is uint16 (H,W,128) viewed as uint32 pairs for the atomic add. */
extern "C" __global__ void aggregate32(const unsigned char* __restrict__ C,
                                       unsigned int* __restrict__ S2,
                                       const int* __restrict__ sx, const int* __restrict__ sy,
                                       const int* __restrict__ sdir,
                                       const int* __restrict__ ddx, const int* __restrict__ ddy,
                                       const int* __restrict__ dp1, const int* __restrict__ dp2,
                                       int npaths, int H, int W) {
    int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= npaths) return;                  /* whole warps exit together: npaths is per-warp */
    int k = sdir[w];
    int dx = ddx[k], dy = ddy[k], p1 = dp1[k], p2 = dp2[k];
    int x = sx[w], y = sy[w];
    const int INF = 1 << 28;
    long base = ((long)y * W + x) * MAX_DISP + lane * 4;
    uchar4 c4 = *reinterpret_cast<const uchar4*>(C + base);
    int L0 = c4.x, L1 = c4.y, L2 = c4.z, L3 = c4.w;
    atomicAdd(S2 + (base >> 1),     (unsigned int)L0 | ((unsigned int)L1 << 16));
    atomicAdd(S2 + (base >> 1) + 1, (unsigned int)L2 | ((unsigned int)L3 << 16));
    int mprev = wmin(min(min(L0, L1), min(L2, L3)));
    x += dx; y += dy;
    while (x >= 0 && x < W && y >= 0 && y < H) {
        base = ((long)y * W + x) * MAX_DISP + lane * 4;
        c4 = *reinterpret_cast<const uchar4*>(C + base);
        int left  = __shfl_up_sync(0xFFFFFFFFu, L3, 1);     /* L[d-1] for d = 4l   (lane l-1's 4th) */
        int right = __shfl_down_sync(0xFFFFFFFFu, L0, 1);   /* L[d+1] for d = 4l+3 (lane l+1's 1st) */
        if (lane == 0) left = INF;
        if (lane == 31) right = INF;
        int mp2 = mprev + p2;
        int n0 = (int)c4.x + min(min(L0, mp2), min(left, L1) + p1) - mprev;
        int n1 = (int)c4.y + min(min(L1, mp2), min(L0, L2) + p1) - mprev;
        int n2 = (int)c4.z + min(min(L2, mp2), min(L1, L3) + p1) - mprev;
        int n3 = (int)c4.w + min(min(L3, mp2), min(L2, right) + p1) - mprev;
        L0 = n0; L1 = n1; L2 = n2; L3 = n3;
        atomicAdd(S2 + (base >> 1),     (unsigned int)L0 | ((unsigned int)L1 << 16));
        atomicAdd(S2 + (base >> 1) + 1, (unsigned int)L2 | ((unsigned int)L3 << 16));
        mprev = wmin(min(min(L0, L1), min(L2, L3)));
        x += dx; y += dy;
    }
}

/* Left WTA: d in [0, min(127, x)]; uniqueness vs the best non-adjacent; parabola; d < 0.5 -> invalid (0). */
extern "C" __global__ void select_left(const unsigned short* __restrict__ S, float* __restrict__ disp,
                                       float* __restrict__ conf, int H, int W, float uniq, float dmin) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    long b = ((long)y * W + x) * MAX_DISP;
    int nd = min(MAX_DISP, x + 1);
    unsigned int best = 0xFFFFFFFFu; int bd = 0;
    for (int d = 0; d < nd; d++) { unsigned int c = S[b + d]; if (c < best) { best = c; bd = d; } }
    unsigned int sec = 0xFFFFFFFFu;
    for (int d = 0; d < nd; d++) { if (abs(d - bd) > 1) { unsigned int c = S[b + d]; if (c < sec) sec = c; } }
    int p = y * W + x;
    if (sec != 0xFFFFFFFFu && (float)sec <= (float)best * (1.0f + uniq)) { disp[p] = 0.0f; conf[p] = 0.0f; return; }
    float sd = (float)bd;
    if (bd > 0 && bd < nd - 1) {
        int cv = (int)S[b + bd - 1], cc = (int)best, cn = (int)S[b + bd + 1];
        int den = cv - 2 * cc + cn;
        if (den > 0) { float off = (float)(cv - cn) / (float)(2 * den); sd = (float)bd + fmaxf(-1.0f, fminf(1.0f, off)); }
    }
    float cval = (sec != 0xFFFFFFFFu && sec > 0) ? 1.0f - (float)best / (float)sec : 0.5f;
    disp[p] = (sd >= dmin) ? sd : 0.0f;
    conf[p] = (sd >= dmin) ? fmaxf(0.0f, fminf(1.0f, cval)) : 0.0f;
}

/* Right WTA off the SAME volume: right pixel xr at disparity d is left pixel xr + d -> S(y, xr + d, d). */
extern "C" __global__ void select_right(const unsigned short* __restrict__ S, float* __restrict__ dispr,
                                        int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int nd = min(MAX_DISP, W - x);
    unsigned int best = 0xFFFFFFFFu; int bd = 0;
    for (int d = 0; d < nd; d++) {
        unsigned int c = S[((long)y * W + x + d) * MAX_DISP + d];
        if (c < best) { best = c; bd = d; }
    }
    float sd = (float)bd;
    if (bd > 0 && bd < nd - 1) {
        int cv = (int)S[((long)y * W + x + bd - 1) * MAX_DISP + bd - 1];
        int cn = (int)S[((long)y * W + x + bd + 1) * MAX_DISP + bd + 1];
        int den = cv - 2 * (int)best + cn;
        if (den > 0) { float off = (float)(cv - cn) / (float)(2 * den); sd = (float)bd + fmaxf(-1.0f, fminf(1.0f, off)); }
    }
    dispr[y * W + x] = sd;
}

extern "C" __global__ void lr_check(float* __restrict__ dl, const float* __restrict__ dr, float* __restrict__ conf,
                                    int H, int W, float thr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int i = y * W + x;
    float d = dl[i];
    if (d <= 0.0f) return;
    int xr = x - (int)(d + 0.5f);
    if (xr < 0 || xr >= W || fabsf(d - dr[y * W + xr]) > thr) { dl[i] = 0.0f; conf[i] = 0.0f; }
}
"""


def _kernel_sha1():
    return hashlib.sha1(KERNELS.encode("utf-8")).hexdigest()


class SGM32:
    """Dense 32-direction SGM on rectified uint8 pairs of one fixed size.  `compute` returns host numpy
    (disp float32 with 0 = invalid, conf float32); `compute_gpu` keeps them on the device."""

    NAME = "sgm32"

    def __init__(self, width, height, p1=P1, p2=P2, uniqueness=UNIQUENESS, lr_threshold=LR_THRESHOLD_PX,
                 verbose=False):
        import cupy as cp
        self.cp = cp
        self.W, self.H = int(width), int(height)
        self.p1, self.p2 = int(p1), int(p2)
        self.uniq = float(uniqueness)
        self.lr_thr = float(lr_threshold)
        dirs = directions()
        p1s = [self.p1 * n // d for (_x, _y, n, d) in dirs]
        p2s = [self.p2 * n // d for (_x, _y, n, d) in dirs]
        # (c): the uint16 volume is exact only while 32 x (C_max + P2_max) fits
        self.s_bound = len(dirs) * (CENSUS_BITS + max(p2s))
        if self.s_bound >= 65536:
            raise ValueError("uint16 aggregation not exact: bound %d" % self.s_bound)
        mod = cp.RawModule(code=KERNELS, options=("-std=c++14",))
        self._k = {n: mod.get_function(n) for n in
                   ("census9", "cost_volume", "aggregate32", "select_left", "select_right", "lr_check")}
        H, W = self.H, self.W
        sx, sy, sd = [], [], []
        for k, (dx, dy, _n, _d) in enumerate(dirs):
            st = path_starts(H, W, dx, dy)
            sx.append(st[:, 0]); sy.append(st[:, 1]); sd.append(np.full(len(st), k, np.int32))
        self.npaths = int(sum(len(a) for a in sx))
        self._sx = cp.asarray(np.concatenate(sx))
        self._sy = cp.asarray(np.concatenate(sy))
        self._sd = cp.asarray(np.concatenate(sd))
        self._ddx = cp.asarray(np.array([d[0] for d in dirs], np.int32))
        self._ddy = cp.asarray(np.array([d[1] for d in dirs], np.int32))
        self._dp1 = cp.asarray(np.array(p1s, np.int32))
        self._dp2 = cp.asarray(np.array(p2s, np.int32))
        self._L0 = cp.empty(H * W, cp.uint64); self._L1 = cp.empty(H * W, cp.uint16)
        self._R0 = cp.empty(H * W, cp.uint64); self._R1 = cp.empty(H * W, cp.uint16)
        self._C = cp.empty(H * W * MAX_DISP, cp.uint8)
        self._S = cp.empty(H * W * MAX_DISP, cp.uint16)
        self._disp = cp.empty(H * W, cp.float32)
        self._dispr = cp.empty(H * W, cp.float32)
        self._conf = cp.empty(H * W, cp.float32)
        self.n_aggregations = 0
        self.n_frames = 0
        self.ms = []
        if verbose:
            print(" [sgm32] %dx%d, 32 directions (%d paths), census 9x9 = %d bits in two words, P1 %d P2 %d "
                  "(an earlier prototype's 10/120, rescaled from its 63-step census cost to 80 bits), uint8 cost + "
                  "uint16 aggregation (bound %d), "
                  "one aggregation per frame, no hole fill" % (W, H, self.npaths, CENSUS_BITS, self.p1,
                                                               self.p2, self.s_bound))

    # ---- the cache key: inputs and algorithm, never the engine build ----------------------------------------
    def key(self, dataset="", extra=None):
        from xclean.stages.cache import build_key
        return build_key(stage=self.NAME, dataset=str(dataset), kernel_sha1=_kernel_sha1()[:16],
                         p1=self.p1, p2=self.p2, uniq=self.uniq, lr=self.lr_thr, census=CENSUS_BITS,
                         extra=dict(extra or {}))

    def compute_gpu(self, left, right):
        """-> (disp, conf) cupy (H, W) float32; disp 0 = invalid."""
        cp = self.cp
        H, W = self.H, self.W
        l = cp.asarray(np.ascontiguousarray(left, dtype=np.uint8)).reshape(H * W)
        r = cp.asarray(np.ascontiguousarray(right, dtype=np.uint8)).reshape(H * W)
        b2 = (16, 16)
        g2 = ((W + 15) // 16, (H + 15) // 16)
        self._k["census9"](g2, b2, (l, self._L0, self._L1, np.int32(H), np.int32(W)))
        self._k["census9"](g2, b2, (r, self._R0, self._R1, np.int32(H), np.int32(W)))
        n = H * W * MAX_DISP
        self._k["cost_volume"](((n + 255) // 256,), (256,),
                               (self._L0, self._L1, self._R0, self._R1, self._C, np.int32(H), np.int32(W)))
        self._S.fill(0)
        S2 = self._S.view(cp.uint32)
        threads = 128                                       # 4 warps = 4 paths per block
        blocks = (self.npaths * 32 + threads - 1) // threads
        self._k["aggregate32"]((blocks,), (threads,),
                               (self._C, S2, self._sx, self._sy, self._sd, self._ddx, self._ddy,
                                self._dp1, self._dp2, np.int32(self.npaths), np.int32(H), np.int32(W)))
        self.n_aggregations += 1
        self._k["select_left"](g2, b2, (self._S, self._disp, self._conf, np.int32(H), np.int32(W),
                                        np.float32(self.uniq), np.float32(MIN_DISP_PX)))
        self._k["select_right"](g2, b2, (self._S, self._dispr, np.int32(H), np.int32(W)))
        self._k["lr_check"](g2, b2, (self._disp, self._dispr, self._conf, np.int32(H), np.int32(W),
                                     np.float32(self.lr_thr)))
        self.n_frames += 1
        return self._disp.reshape(H, W), self._conf.reshape(H, W)

    def compute(self, left, right):
        """(H,W) uint8 rectified pair -> (disp, conf) host float32; disp 0 = invalid (no measurement)."""
        t0 = time.perf_counter()
        d, c = self.compute_gpu(left, right)
        out = (self.cp.asnumpy(d), self.cp.asnumpy(c))
        self.ms.append((time.perf_counter() - t0) * 1e3)
        return out

    def aggregated_volume(self):
        """The last uint16 aggregated volume (H, W, 128) on the host (tests only)."""
        return self.cp.asnumpy(self._S).reshape(self.H, self.W, MAX_DISP)

    def cost_volume(self):
        return self.cp.asnumpy(self._C).reshape(self.H, self.W, MAX_DISP)


class SgmDepth:
    """The depth STAGE: the engine contract `compute(left, right) -> (H,W) float32 positive disparity`, with 0 =
    invalid made explicit.  One SGM32 per image size, built lazily."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self._sgm = None
        self.last_conf = None

    def _get(self, W, H):
        if self._sgm is None or (self._sgm.W, self._sgm.H) != (W, H):
            self._sgm = SGM32(W, H, verbose=self.verbose)
        return self._sgm

    def key(self, dataset="", extra=None):
        from xclean.stages.cache import build_key
        return build_key(stage=SGM32.NAME, dataset=str(dataset), kernel_sha1=_kernel_sha1()[:16],
                         p1=P1, p2=P2, uniq=UNIQUENESS, lr=LR_THRESHOLD_PX, census=CENSUS_BITS,
                         extra=dict(extra or {}))

    def compute(self, left, right):
        H, W = left.shape[:2]
        d, c = self._get(W, H).compute(left, right)
        self.last_conf = c
        return d

    def stats(self):
        s = self._sgm
        if s is None or not s.ms:
            return None
        a = np.asarray(s.ms)
        return {"frames": int(s.n_frames), "aggregations": int(s.n_aggregations),
                "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
                "size": [s.W, s.H]}
