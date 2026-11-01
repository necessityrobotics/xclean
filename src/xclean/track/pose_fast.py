#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""M2: pose.refine in numba for sigma_par None and point_scale None (the P1 call),
one fused pass per evaluation.  The profiler's microbench kernel, integrated behind XCLEAN_LAT=m2.  Rounding-level
equal to the numpy path (<= 2.3e-12 measured; the SO(3) projection is an SVD here), NOT bit-identical: inlier masks can
flip at the gate, so its effect is judged on trajectories, not by identity.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

_TAYLOR = float((7257600.0 * np.finfo(np.float64).eps) ** 0.1)


@njit(cache=True, nogil=True)
def _se3_exp_nb(xi):
    rho = xi[:3]
    phi = xi[3:]
    t2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]
    t = math.sqrt(t2)
    if t < _TAYLOR:
        a = 1.0 - t2 / 6.0 + t2 ** 2 / 120.0 - t2 ** 3 / 5040.0
        b = 0.5 - t2 / 24.0 + t2 ** 2 / 720.0 - t2 ** 3 / 40320.0
        c = 1.0 / 6.0 - t2 / 120.0 + t2 ** 2 / 5040.0 - t2 ** 3 / 362880.0
    else:
        a = math.sin(t) / t
        b = (1.0 - math.cos(t)) / t2
        c = (t - math.sin(t)) / (t2 * t)
    K = np.zeros((3, 3))
    K[0, 1] = -phi[2]
    K[0, 2] = phi[1]
    K[1, 0] = phi[2]
    K[1, 2] = -phi[0]
    K[2, 0] = -phi[1]
    K[2, 1] = phi[0]
    K2 = K @ K
    T = np.eye(4)
    V = np.eye(3) + b * K + c * K2
    T[:3, :3] = np.eye(3) + a * K + b * K2
    T[:3, 3] = V @ rho
    return T


@njit(cache=True, nogil=True)
def refine_nb(T_cw, P, uv, fx, fy, cx, cy, sigma_perp, iters, hk):
    """pose.refine (Huber LM on SE(3), isotropic whitening 1 / sigma_perp) -> T_cw."""
    T = T_cw.copy()
    n = P.shape[0]
    lam = 1e-3
    wi = 1.0 / sigma_perp
    J = np.zeros((2, 6))
    for _it in range(iters):
        H = np.zeros((6, 6))
        b = np.zeros(6)
        e0 = 0.0
        for i in range(n):
            x = T[0, 0] * P[i, 0] + T[0, 1] * P[i, 1] + T[0, 2] * P[i, 2] + T[0, 3]
            y = T[1, 0] * P[i, 0] + T[1, 1] * P[i, 1] + T[1, 2] * P[i, 2] + T[1, 3]
            z = T[2, 0] * P[i, 0] + T[2, 1] * P[i, 1] + T[2, 2] * P[i, 2] + T[2, 3]
            z = max(z, 1e-6)
            r0 = fx * x / z + cx - uv[i, 0]
            r1 = fy * y / z + cy - uv[i, 1]
            e0 += math.sqrt(r0 * r0 + r1 * r1)
            rw0 = r0 * wi
            rw1 = r1 * wi
            e = math.sqrt(rw0 * rw0 + rw1 * rw1)
            w = 1.0 if e <= hk else hk / max(e, 1e-9)
            iz = 1.0 / z
            iz2 = iz * iz
            J[0, 0] = fx * iz
            J[0, 1] = 0.0
            J[0, 2] = -fx * x * iz2
            J[1, 0] = 0.0
            J[1, 1] = fy * iz
            J[1, 2] = -fy * y * iz2
            J[0, 3] = -fx * x * y * iz2
            J[0, 4] = fx * (1.0 + x * x * iz2)
            J[0, 5] = -fx * y * iz
            J[1, 3] = -fy * (1.0 + y * y * iz2)
            J[1, 4] = fy * x * y * iz2
            J[1, 5] = fy * x * iz
            for r in range(2):
                rr = rw0 if r == 0 else rw1
                for p in range(6):
                    jp = J[r, p] * wi
                    b[p] -= w * jp * rr
                    for qq in range(6):
                        H[p, qq] += w * jp * J[r, qq] * wi
        Hl = H.copy()
        for p in range(6):
            Hl[p, p] += lam * max(H[p, p], 1e-9)
        dx = np.linalg.solve(Hl, b)
        Tn = _se3_exp_nb(dx) @ T
        U, _s, Vt = np.linalg.svd(Tn[:3, :3])
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(U @ Vt))
        Tn[:3, :3] = U @ D @ Vt
        e1 = 0.0
        for i in range(n):
            x = Tn[0, 0] * P[i, 0] + Tn[0, 1] * P[i, 1] + Tn[0, 2] * P[i, 2] + Tn[0, 3]
            y = Tn[1, 0] * P[i, 0] + Tn[1, 1] * P[i, 1] + Tn[1, 2] * P[i, 2] + Tn[1, 3]
            z = max(Tn[2, 0] * P[i, 0] + Tn[2, 1] * P[i, 1] + Tn[2, 2] * P[i, 2] + Tn[2, 3], 1e-6)
            r0 = fx * x / z + cx - uv[i, 0]
            r1 = fy * y / z + cy - uv[i, 1]
            e1 += math.sqrt(r0 * r0 + r1 * r1)
        if e1 / n <= e0 / n:
            T = Tn
            lam = max(lam * 0.5, 1e-6)
        else:
            lam = min(lam * 4.0, 1e3)
        if math.sqrt((dx * dx).sum()) < 1e-9:
            break
    return T
