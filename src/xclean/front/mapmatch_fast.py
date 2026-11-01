#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1 MAP MATCHING, FAST PATH (numba, CPU only): the same API and the SAME OUTPUTS as front/mapmatch.py and as the
stage-A matcher Frontend.match_desc, with the per-call Python / numpy work fused into single compiled calls.

    mm = MapMatcher(width, height)                     # = mapmatch.MapMatcher
    lm_kp, dist, d2 = mm.match(kp, desc, uv_pred, S_uv, lm_desc, lm_ndesc)
    i0, i1 = match_desc(f0, f1, mask0=None, mask1=None) # = Frontend.match_desc (cv2.BFMatcher knnMatch x2 + loop)
    fe.match_desc = match_desc                          # how the engine's stage A picks it up (no engine edit)

WHAT CHANGES AND WHAT DOES NOT
  * MapMatcher.match: the inverse innovation covariances, the gate half-widths, the 16-px keypoint grid (a stable
    counting sort = np.argsort(cell, kind="stable")) and the gated Hamming loop run in ONE njit call; every expression
    keeps mapmatch.py's evaluation order (det, Sinv, half = sqrt(max(chi2 S_ii, 0)), the gated loop verbatim), so the
    outputs are identical (tests/test_lat_front.py checks every array on real frames).
  * match_desc: one pass over the L x N Hamming matrix (hardware popcount) keeping, per query row, the first minimum
    and the second smallest distance (cv2.BFMatcher.knnMatch k=2's order: a later equal distance never displaces an
    earlier one) and, per train column, the first minimum (knnMatch k=1 backward). The chance bound, the Lowe ratio
    (best < 0.8 second, in float64 as the Python comparison) and the mutual check are Frontend.match_desc's; the
    returned arrays are in query order, as the DMatch loop builds them.
No constant is added: GRID (16 px) and CHI2_2_999 are mapmatch.py's; HAMMING_CHANCE / RATIO are features.py's.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, types
from numba.extending import intrinsic

from xclean.front.mapmatch import CHI2_2_999, GRID, _gated
from xclean.stages.features import HAMMING_CHANCE, RATIO


@intrinsic
def _ctpop(typingctx, x):
    sig = types.uint64(types.uint64)

    def codegen(context, builder, signature, args):
        return builder.ctpop(args[0])
    return sig, codegen


@njit(cache=True, nogil=True)
def _bf_mutual(A, B, ham_max, ratio, o0, o1):
    """One-pass mutual nearest neighbour + ratio + chance bound on (L,4) / (N,4) uint64 descriptor rows.
    Writes the matches (query order) into o0 / o1; returns their count."""
    L = A.shape[0]
    N = B.shape[0]
    b1 = np.full(L, 1 << 30, np.int64)
    b2 = np.full(L, 1 << 30, np.int64)
    bi = np.full(L, -1, np.int64)
    cb = np.full(N, 1 << 30, np.int64)
    ci = np.full(N, -1, np.int64)
    for i in range(L):
        a0 = A[i, 0]
        a1 = A[i, 1]
        a2 = A[i, 2]
        a3 = A[i, 3]
        r1 = b1[i]
        r2 = b2[i]
        ri = bi[i]
        for j in range(N):
            dd = int(_ctpop(a0 ^ B[j, 0]) + _ctpop(a1 ^ B[j, 1]) + _ctpop(a2 ^ B[j, 2]) + _ctpop(a3 ^ B[j, 3]))
            if dd < r1:
                r2 = r1
                r1 = dd
                ri = j
            elif dd < r2:
                r2 = dd
            if dd < cb[j]:
                cb[j] = dd
                ci[j] = i
        b1[i] = r1
        b2[i] = r2
        bi[i] = ri
    m = 0
    for i in range(L):
        j = bi[i]
        if j < 0:
            continue
        if b1[i] > ham_max or not (b1[i] < ratio * b2[i]):
            continue
        if ci[j] != i:
            continue
        o0[m] = i
        o1[m] = j
        m += 1
    return m


def match_desc(f0, f1, mask0=None, mask1=None):
    """Frontend.match_desc (prediction-free ORB matching: mutual NN, Lowe ratio, Hamming chance bound) -> (i0, i1)."""
    d0, d1 = f0["descriptors"], f1["descriptors"]
    i0 = np.arange(len(d0)) if mask0 is None else np.nonzero(mask0)[0]
    i1 = np.arange(len(d1)) if mask1 is None else np.nonzero(mask1)[0]
    z = np.zeros(0, np.int64)
    if len(i0) < 2 or len(i1) < 2:
        return z, z
    A = np.ascontiguousarray(d0[i0] if mask0 is not None else d0, np.uint8)
    B = np.ascontiguousarray(d1[i1] if mask1 is not None else d1, np.uint8)
    Au = A.view(np.uint64).reshape(len(A), 4)
    Bu = B.view(np.uint64).reshape(len(B), 4)
    o0 = np.empty(len(A), np.int64)
    o1 = np.empty(len(A), np.int64)
    m = _bf_mutual(Au, Bu, int(HAMMING_CHANCE), float(RATIO), o0, o1)
    if m == 0:
        return z, z
    return i0[o0[:m]].astype(np.int64), i1[o1[:m]].astype(np.int64)


@njit(cache=True, nogil=True)
def _match_fused(kp, kd, uvp, S, ld, lnd, chi2, ratio, ham_max, gw, gh, out_kp, out_d, out_d2):
    """mapmatch.MapMatcher.match's numpy preamble (Sinv, half, nd, the stable grid) + _gated, in one call."""
    L = uvp.shape[0]
    Sinv = np.zeros((L, 2, 2))
    half = np.empty((L, 2))
    nd = np.zeros(L, np.int64)
    for l in range(L):
        det = S[l, 0, 0] * S[l, 1, 1] - S[l, 0, 1] * S[l, 1, 0]
        if det > 0:
            Sinv[l, 0, 0] = S[l, 1, 1] / det
            Sinv[l, 1, 1] = S[l, 0, 0] / det
            Sinv[l, 0, 1] = -S[l, 0, 1] / det
            Sinv[l, 1, 0] = -S[l, 1, 0] / det
            nd[l] = lnd[l]
        h0 = chi2 * S[l, 0, 0]                 # np.sqrt(np.maximum(chi2 S_ii, 0.0)): NaN propagates as in numpy
        h1 = chi2 * S[l, 1, 1]
        half[l, 0] = math.sqrt(h0 if (h0 >= 0.0 or math.isnan(h0)) else 0.0)
        half[l, 1] = math.sqrt(h1 if (h1 >= 0.0 or math.isnan(h1)) else 0.0)
    N = kp.shape[0]
    ncell = gw * gh
    cnt = np.zeros(ncell + 1, np.int64)
    cell = np.empty(N, np.int64)
    for k in range(N):
        cxk = int(math.floor(kp[k, 0] / GRID))
        cyk = int(math.floor(kp[k, 1] / GRID))
        if cxk < 0:
            cxk = 0
        if cxk > gw - 1:
            cxk = gw - 1
        if cyk < 0:
            cyk = 0
        if cyk > gh - 1:
            cyk = gh - 1
        c = cyk * gw + cxk
        cell[k] = c
        cnt[c + 1] += 1
    for c in range(ncell):
        cnt[c + 1] += cnt[c]
    cstart = cnt.copy()
    cur = cnt[:ncell].copy()
    cidx = np.empty(N, np.int64)
    for k in range(N):                      # stable counting sort (= np.argsort(cell, kind="stable"))
        c = cell[k]
        cidx[cur[c]] = k
        cur[c] += 1
    _gated(kp, kd, cstart, cidx, gw, gh, uvp, Sinv, half, ld, nd, chi2, ratio, ham_max, out_kp, out_d, out_d2)
    return 0


class MapMatcher:
    """mapmatch.MapMatcher with the per-call numpy work fused into one njit call (identical outputs)."""

    def __init__(self, width, height):
        self.W = int(width)
        self.H = int(height)
        self.gw = int(math.ceil(self.W / GRID))
        self.gh = int(math.ceil(self.H / GRID))
        self.stats = {"calls": 0, "landmarks": 0, "matched": 0}

    def match(self, kp, desc, uv_pred, S_uv, lm_desc, lm_ndesc, chi2=CHI2_2_999):
        L = len(uv_pred)
        out_kp = np.full(L, -1, np.int64)
        out_d = np.full(L, 9999, np.int64)
        out_d2 = np.full(L, np.inf)
        self.stats["calls"] += 1
        self.stats["landmarks"] += L
        if L == 0 or len(kp) == 0:
            return out_kp, out_d, out_d2
        kp64 = np.ascontiguousarray(kp, np.float64)
        kd = np.ascontiguousarray(desc, np.uint8).view(np.uint64).reshape(len(kp), 4)
        ld = np.ascontiguousarray(lm_desc, np.uint8).view(np.uint64).reshape(L, lm_desc.shape[1], 4)
        _match_fused(kp64, kd, np.ascontiguousarray(uv_pred, np.float64), np.ascontiguousarray(S_uv, np.float64),
                     ld, np.ascontiguousarray(lm_ndesc, np.int64), float(chi2), float(RATIO), int(HAMMING_CHANCE),
                     self.gw, self.gh, out_kp, out_d, out_d2)
        self.stats["matched"] += int((out_kp >= 0).sum())
        return out_kp, out_d, out_d2


def warmup():
    """Compile both kernels (cache=True: later processes load them from disk)."""
    mm = MapMatcher(64, 64)
    mm.match(np.array([[10.0, 10.0], [20.0, 20.0]]), np.zeros((2, 32), np.uint8), np.array([[10.0, 10.0]]),
             np.eye(2)[None] * 4.0, np.zeros((1, 3, 32), np.uint8), np.array([1]))
    d = np.arange(64, dtype=np.uint8).reshape(2, 32)
    match_desc({"descriptors": d}, {"descriptors": d[::-1].copy()})
