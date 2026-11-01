#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""AFFINE PATCH ALIGNER (spec XC-01 / XC-07 / XC-14): find a reference patch in the current image under an affine warp.

    al = PatchAligner()
    pts, A, ok, zncc = al.align(ref_img, ref_pts, cur_img, init_pts, A0)

For each point i the model is   I_cur(c_i + A_i x) = g_i * I_ref(r_i + x) + b_i   for x on the KLT_WIN x KLT_WIN grid,
with unknowns the current centre c_i (2), the affine A_i (4, seeded by the PREDICTED warp A0_i - from the relative
pose and the landmark depth in the tracker) and the photometric gain / offset (2).  Forward-additive Gauss-Newton,
coarse to fine over an image pyramid, vectorised over all points (numpy, or CuPy when `xp=cupy`).

Acceptance: converged (last centre step <= CONV_PX), the warp stays orientation-preserving, the centre moved no more
than the search radius the caller gives, and the zero-mean normalised correlation of the aligned patch >= ZNCC_MIN.
Constants: the KLT window (the tracker's own localisation support), ZNCC_MIN, the pyramid depth
= KLT's, CONV_PX = 0.01 px (a numerical stopping rule, two orders below FEATURE_SIGMA_PX).
"""
from __future__ import annotations

import numpy as np

from xclean.stages.features import KLT_WIN, KLT_LEVELS
from xclean.stereo.epipolar import ZNCC_MIN

ITERS = 12
CONV_PX = 0.01


def _pyramid(img, levels):
    import cv2
    out = [np.asarray(img, np.float32)]
    for _ in range(levels):
        out.append(cv2.pyrDown(out[-1]))
    return out


def _bilinear(xp, img, x, y):
    H, W = img.shape
    x0 = xp.floor(x)
    y0 = xp.floor(y)
    ax = x - x0
    ay = y - y0
    x0 = xp.clip(x0, 0, W - 2).astype(np.int64)
    y0 = xp.clip(y0, 0, H - 2).astype(np.int64)
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    top = a + ax * (b - a)
    bot = c + ax * (d - c)
    v = top + ay * (bot - top)
    gx = (b - a) * (1 - ay) + (d - c) * ay
    gy = (c - a) * (1 - ax) + (d - b) * ax
    inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    return v, gx, gy, inside


class PatchAligner:
    def __init__(self, xp=np, win=KLT_WIN, levels=KLT_LEVELS):
        self.xp = xp
        self.h = win // 2
        self.levels = int(levels)
        g = np.arange(-self.h, self.h + 1, dtype=np.float32)
        gy, gx = np.meshgrid(g, g, indexing="ij")
        self.gx = xp.asarray(gx.ravel())
        self.gy = xp.asarray(gy.ravel())

    def align(self, ref_img, ref_pts, cur_img, init_pts, A0=None, radius=None, ref_pyr=None, cur_pyr=None):
        """ref_pts / init_pts (N,2) full-resolution pixels; A0 (N,2,2) predicted affine (identity when None);
        radius (N,) or scalar: the largest allowed centre displacement from init_pts (None = unlimited).
        -> (pts (N,2) float32, A (N,2,2), ok (N,) bool, zncc (N,)) as numpy arrays."""
        xp = self.xp
        N = len(ref_pts)
        if N == 0:
            return (np.zeros((0, 2), np.float32), np.zeros((0, 2, 2)), np.zeros(0, bool), np.zeros(0))
        rp = ref_pyr or _pyramid(ref_img, self.levels)
        cp_ = cur_pyr or _pyramid(cur_img, self.levels)
        rp = [xp.asarray(a) for a in rp]
        cp_ = [xp.asarray(a) for a in cp_]
        r = xp.asarray(np.asarray(ref_pts, np.float32))
        c = xp.asarray(np.asarray(init_pts, np.float32)).copy()
        A = xp.asarray(np.tile(np.eye(2, dtype=np.float32), (N, 1, 1)) if A0 is None
                       else np.asarray(A0, np.float32)).copy()
        g = xp.ones(N, np.float32)
        b = xp.zeros(N, np.float32)
        conv = xp.zeros(N, bool)
        gx, gy = self.gx[None, :], self.gy[None, :]
        for L in range(self.levels, -1, -1):
            s = 0.5 ** L
            # the template at this level: the reference image sampled on the (unwarped) grid around r
            T, _, _, tin = _bilinear(xp, rp[L], r[:, 0:1] * s + gx, r[:, 1:2] * s + gy)
            conv[:] = False
            for _it in range(ITERS):
                wx = c[:, 0:1] * s + A[:, 0, 0:1] * gx + A[:, 0, 1:2] * gy
                wy = c[:, 1:2] * s + A[:, 1, 0:1] * gx + A[:, 1, 1:2] * gy
                I, Ix, Iy, cin = _bilinear(xp, cp_[L], wx, wy)
                valid = (tin & cin).astype(np.float32)
                res = (I - (g[:, None] * T + b[:, None])) * valid
                # model derivatives wrt [cx, cy, a11, a12, a21, a22, g, b] (c in LEVEL pixels)
                J = xp.stack([Ix, Iy, Ix * gx, Ix * gy, Iy * gx, Iy * gy, -T, -xp.ones_like(T)], axis=2)
                J = J * valid[:, :, None]
                H_ = xp.einsum("nki,nkj->nij", J, J) + 1e-6 * xp.eye(8, dtype=np.float32)[None]
                gr = xp.einsum("nki,nk->ni", J, res)
                d = -xp.linalg.solve(H_, gr[:, :, None])[:, :, 0]
                c = c + d[:, 0:2] / s
                A = A + d[:, 2:6].reshape(N, 2, 2)
                g = g + d[:, 6]
                b = b + d[:, 7]
                step = xp.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) / s
                conv = step <= CONV_PX
                if bool(conv.all()):
                    break
        # acceptance at full resolution
        wx = c[:, 0:1] + A[:, 0, 0:1] * gx + A[:, 0, 1:2] * gy
        wy = c[:, 1:2] + A[:, 1, 0:1] * gx + A[:, 1, 1:2] * gy
        I, _, _, cin = _bilinear(xp, cp_[0], wx, wy)
        T, _, _, tin = _bilinear(xp, rp[0], r[:, 0:1] + gx, r[:, 1:2] + gy)
        Ic = I - I.mean(axis=1, keepdims=True)
        Tc = T - T.mean(axis=1, keepdims=True)
        zn = (Ic * Tc).sum(axis=1) / xp.maximum(xp.sqrt((Ic ** 2).sum(axis=1) * (Tc ** 2).sum(axis=1)), 1e-6)
        det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
        ok = conv & (det > 0) & cin.all(axis=1) & tin.all(axis=1) & (zn >= ZNCC_MIN) & xp.isfinite(c).all(axis=1)
        if radius is not None:
            disp = xp.sqrt(((c - xp.asarray(np.asarray(init_pts, np.float32))) ** 2).sum(axis=1))
            ok = ok & (disp <= xp.asarray(radius, dtype=np.float32))
        to = (lambda a: a.get()) if xp is not np else (lambda a: a)
        return to(c).astype(np.float32), to(A), to(ok), to(zn)
