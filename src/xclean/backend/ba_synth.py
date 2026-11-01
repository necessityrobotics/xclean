# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""A synthetic local bundle-adjustment problem with a known ground truth.

Moved verbatim from xclean/tests/test_ba.py (release packaging, 2026-10-30): `backend.ba.warmup()` builds its
throwaway JIT warm-up solve from `make_problem`, so it is runtime code, not test code.  The tests import it
from here.
"""
from __future__ import annotations

import numpy as np


class _Rig:
    fx_rect = 458.654
    cx, cy = 367.215, 248.375
    width, height = 752, 480


def make_problem(nk=15, nl=600, obs_per=4, seed=5, pose_noise=0.02,
                 pt_noise=0.05, px_noise=0.5):
    """A local window: nk keyframes on a smooth arc, nl points in front."""
    from xclean.core.lie import se3_exp, se3_inv
    rng = np.random.RandomState(seed)
    rig = _Rig()
    poses_true = []
    for i in range(nk):
        xi = np.array([0.09 * i, 0.01 * i, 0.0, 0.0, 0.02 * i, 0.0])
        poses_true.append(se3_exp(xi))
    poses_true = np.stack(poses_true)

    pts_true = np.stack([rng.uniform(-2.5, 2.5, nl),
                         rng.uniform(-1.5, 1.5, nl),
                         rng.uniform(1.5, 8.0, nl)], axis=1)
    pts_true[:, 0] += 0.6
    okf, olm, ouv = [], [], []
    for j in range(nl):
        ks = rng.choice(nk, size=min(obs_per, nk), replace=False)
        for k in ks:
            T_cw = se3_inv(poses_true[k])
            Pc = T_cw[:3, :3] @ pts_true[j] + T_cw[:3, 3]
            if Pc[2] < 0.5:
                continue
            u = rig.fx_rect * Pc[0] / Pc[2] + rig.cx
            v = rig.fx_rect * Pc[1] / Pc[2] + rig.cy
            if not (0 <= u < rig.width and 0 <= v < rig.height):
                continue
            okf.append(k)
            olm.append(j)
            ouv.append((u + rng.randn() * px_noise,
                        v + rng.randn() * px_noise))
    okf = np.array(okf, np.int32)
    olm = np.array(olm, np.int32)
    ouv = np.array(ouv, np.float64)
    uniq, olm = np.unique(olm, return_inverse=True)
    pts_true = pts_true[uniq]

    poses0 = poses_true.copy()
    for i in range(1, nk):
        poses0[i] = poses_true[i] @ se3_exp(rng.randn(6) * pose_noise)
    pts0 = pts_true + rng.randn(*pts_true.shape) * pt_noise
    return dict(rig=rig, poses_true=poses_true, pts_true=pts_true,
                poses0=poses0, pts0=pts0, okf=okf, olm=olm.astype(np.int32),
                ouv=ouv)
