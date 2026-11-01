#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1 MAP MATCHING: project the window's landmarks with the PREDICTED pose, then gated Hamming matching (numba).

    mm = MapMatcher(width, height)
    lm_kp, dist, d2 = mm.match(kp, desc, uv_pred, S_uv, lm_desc, lm_ndesc)

- **Candidates of landmark l:** the keypoints inside its chi2_2(0.999) ellipse of the projected innovation covariance
  S_uv(l) = J Sigma_pose J^T + J_X Sigma_X J_X^T + sigma_px^2 I. The caller builds S_uv (Class C: it is the predicted
  covariance; the gate is reshaped by it, never widened by hand).
- **Score:** the minimum Hamming distance over the landmark's stored descriptors (up to 3; OKVIS keeps 3).
- **Acceptance** (carried constants, features.py): best <= HAMMING_CHANCE (104, the binomial chance bound of 256 bits);
  the Lowe ratio best < RATIO x second best (0.8) among that landmark's in-gate candidates; then MUTUAL best: a
  keypoint claimed by several landmarks keeps the one with the smallest distance (ties: smaller Mahalanobis d2, then
  lower landmark index, which is deterministic).
- Keypoints are bucketed in a uniform grid (cell = 16 px, a data-structure size only: it changes no result) so a
  landmark scans only the cells under its ellipse's bounding box.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from xclean.stages.features import HAMMING_CHANCE, RATIO

CHI2_2_999 = 13.815510557964274        # chi2 quantile, 2 dof, 0.999 (statistics)
GRID = 16


@njit(cache=True, nogil=True, inline="always")
def _pop64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True, nogil=True)
def _ham(a, i, b, l, j):
    s = 0
    for w in range(4):
        s += _pop64(a[i, w] ^ b[l, j, w])
    return s


@njit(cache=True, nogil=True)
def _gated(kp, kd, cstart, cidx, gw, gh, uvp, Sinv, half, ld, ln, chi2, ratio, ham_max,
           out_kp, out_d, out_d2):
    L = uvp.shape[0]
    for l in range(L):
        out_kp[l] = -1
        out_d[l] = 9999
        out_d2[l] = np.inf
        if ln[l] <= 0 or not (math.isfinite(uvp[l, 0]) and math.isfinite(uvp[l, 1])):
            continue
        x0 = int(math.floor((uvp[l, 0] - half[l, 0]) / GRID))
        x1 = int(math.floor((uvp[l, 0] + half[l, 0]) / GRID))
        y0 = int(math.floor((uvp[l, 1] - half[l, 1]) / GRID))
        y1 = int(math.floor((uvp[l, 1] + half[l, 1]) / GRID))
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > gw - 1:
            x1 = gw - 1
        if y1 > gh - 1:
            y1 = gh - 1
        best = 9999
        second = 9999
        bk = -1
        bd2 = np.inf
        for cy in range(y0, y1 + 1):
            for cx in range(x0, x1 + 1):
                c = cy * gw + cx
                for t in range(cstart[c], cstart[c + 1]):
                    k = cidx[t]
                    dx = kp[k, 0] - uvp[l, 0]
                    dy = kp[k, 1] - uvp[l, 1]
                    d2 = dx * (Sinv[l, 0, 0] * dx + Sinv[l, 0, 1] * dy) + dy * (Sinv[l, 1, 0] * dx + Sinv[l, 1, 1] * dy)
                    if d2 > chi2:
                        continue
                    h = 9999
                    for j in range(ln[l]):
                        hj = _ham(kd, k, ld, l, j)
                        if hj < h:
                            h = hj
                    if h < best or (h == best and d2 < bd2):
                        second = best
                        best = h
                        bk = k
                        bd2 = d2
                    elif h < second:
                        second = h
        if bk < 0 or best > ham_max:
            continue
        if second < 9999 and not (best < ratio * second):
            continue
        out_kp[l] = bk
        out_d[l] = best
        out_d2[l] = bd2
    # mutual best: one landmark per keypoint
    N = kp.shape[0]
    owner = np.full(N, -1, np.int64)
    for l in range(L):
        k = out_kp[l]
        if k < 0:
            continue
        o = owner[k]
        if o < 0:
            owner[k] = l
            continue
        if (out_d[l] < out_d[o]) or (out_d[l] == out_d[o] and out_d2[l] < out_d2[o]):
            owner[k] = l
            out_kp[o] = -1
        else:
            out_kp[l] = -1
    return 0


class MapMatcher:
    def __init__(self, width, height):
        self.W = int(width)
        self.H = int(height)
        self.gw = int(math.ceil(self.W / GRID))
        self.gh = int(math.ceil(self.H / GRID))
        self.stats = {"calls": 0, "landmarks": 0, "matched": 0}

    def _grid(self, kp):
        cx = np.clip((kp[:, 0] // GRID).astype(np.int64), 0, self.gw - 1)
        cy = np.clip((kp[:, 1] // GRID).astype(np.int64), 0, self.gh - 1)
        cell = cy * self.gw + cx
        order = np.argsort(cell, kind="stable")
        cnt = np.bincount(cell, minlength=self.gw * self.gh)
        start = np.zeros(self.gw * self.gh + 1, np.int64)
        np.cumsum(cnt, out=start[1:])
        return start, order.astype(np.int64)

    def match(self, kp, desc, uv_pred, S_uv, lm_desc, lm_ndesc, chi2=CHI2_2_999):
        """kp (N,2), desc (N,32) uint8; uv_pred (L,2); S_uv (L,2,2) innovation covariances; lm_desc (L,3,32) uint8,
        lm_ndesc (L,) number of valid descriptors.  -> (kp index per landmark or -1, Hamming distance, Mahalanobis d2)."""
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
        S = np.ascontiguousarray(S_uv, np.float64)
        det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
        good = det > 0
        Sinv = np.zeros_like(S)
        Sinv[good, 0, 0] = S[good, 1, 1] / det[good]
        Sinv[good, 1, 1] = S[good, 0, 0] / det[good]
        Sinv[good, 0, 1] = -S[good, 0, 1] / det[good]
        Sinv[good, 1, 0] = -S[good, 1, 0] / det[good]
        half = np.sqrt(np.maximum(chi2 * np.stack([S[:, 0, 0], S[:, 1, 1]], axis=1), 0.0))
        nd = np.where(good, np.asarray(lm_ndesc, np.int64), 0)
        cstart, cidx = self._grid(kp64)
        _gated(kp64, kd, cstart, cidx, self.gw, self.gh, np.ascontiguousarray(uv_pred, np.float64), Sinv, half,
               ld, nd, float(chi2), float(RATIO), int(HAMMING_CHANCE), out_kp, out_d, out_d2)
        self.stats["matched"] += int((out_kp >= 0).sum())
        return out_kp, out_d, out_d2
