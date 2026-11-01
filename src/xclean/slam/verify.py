# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.3 worker: loop VERIFICATION of a query keyframe against a candidate keyframe's rich set.

  matching   the candidate's keypoints WITH stereo 3D against all of the query's keypoints: mutual nearest neighbour,
             Lowe ratio 0.8, Hamming chance bound 104 (xclean/stages/features.Frontend.match_desc, IMPORTED: the same
             rule as the front end; the numba matcher of the design's build list is NOT built);
  pose       PnP-RANSAC + robust refine (xclean/track/pose.solve_pose, imported; min 10 correspondences to try,
             cv2's RNG seeded per attempt so a verification is a pure function of the two rich sets);
  covariance the PnP information at the solution with every inlier's pixel noise sigma_0 1.2^octave plus its stereo
             point covariance propagated into the query image; Sigma_loop = s^2 H^-1 is formed by
             admission (admit.loop_cov).
Z = T_c^-1 T_q (the query camera in the candidate camera frame).  PnP's own final check (XTEN_PNP_VERIFY) is pinned OFF
in the worker process (the design evidence's setting), independent of the frame process's environment.
"""
from __future__ import annotations

import os
import time

import numpy as np

PNP_SEED = 12345        # cv2 RNG seed per attempt (determinism; not a tuning constant)
MIN_TRY = 10            # correspondences before PnP is attempted (the design evidence's value; D, carried)


def pnp_information(Z, Xc, uv_q, oct_q, dsig_c, uv_c, fx, cx, cy, fB, sig0):
    """Information of the query pose Z = T_c^-1 T_q (perturbation [dphi, dt], rotation right, t additive in the
    candidate frame) from the inlier correspondences, each with pixel covariance sig0^2 1.2^(2 oct) I + the stereo point
    covariance of X_c propagated through the projection.  -> (H 6x6, chi2 whitened, n).  (Design p13_verify.py.)"""
    R, t = Z[:3, :3], Z[:3, 3]
    Xq = (Xc - t) @ R
    z = Xq[:, 2]
    ok = z > 1e-6
    Xq, Xc, uv_q, oct_q, dsig_c, uv_c, z = Xq[ok], Xc[ok], uv_q[ok], oct_q[ok], dsig_c[ok], uv_c[ok], z[ok]
    n = len(z)
    Dpi = np.zeros((n, 2, 3))
    Dpi[:, 0, 0] = fx / z
    Dpi[:, 0, 2] = -fx * Xq[:, 0] / z ** 2
    Dpi[:, 1, 1] = fx / z
    Dpi[:, 1, 2] = -fx * Xq[:, 1] / z ** 2
    pred = np.stack([fx * Xq[:, 0] / z + cx, fx * Xq[:, 1] / z + cy], 1)
    r = pred - uv_q
    dX = np.zeros((n, 3, 6))
    dX[:, 0, 1], dX[:, 0, 2] = -Xq[:, 2], Xq[:, 1]
    dX[:, 1, 0], dX[:, 1, 2] = Xq[:, 2], -Xq[:, 0]
    dX[:, 2, 0], dX[:, 2, 1] = -Xq[:, 1], Xq[:, 0]
    dX[:, :, 3:] = -R.T
    J = np.einsum("nij,njk->nik", Dpi, dX)
    zc = Xc[:, 2]
    d = fB / zc
    Jx = np.zeros((n, 3, 3))
    Jx[:, 0, 0] = zc / fx
    Jx[:, 1, 1] = zc / fx
    dz_dd = -fB / d ** 2
    Jx[:, 0, 2] = (uv_c[:, 0] - cx) / fx * dz_dd
    Jx[:, 1, 2] = (uv_c[:, 1] - cy) / fx * dz_dd
    Jx[:, 2, 2] = dz_dd
    s_c = sig0 * np.ones(n)
    Su = np.zeros((n, 3, 3))
    Su[:, 0, 0] = s_c ** 2
    Su[:, 1, 1] = s_c ** 2
    Su[:, 2, 2] = np.where(np.isfinite(dsig_c), dsig_c, sig0) ** 2
    SX = np.einsum("nij,njk,nlk->nil", Jx, Su, Jx)
    JXq = np.einsum("nij,jk->nik", Dpi, R.T)
    Sp = np.einsum("nij,njk,nlk->nil", JXq, SX, JXq)
    sq = sig0 * (1.2 ** oct_q.astype(float))
    Sp[:, 0, 0] += sq ** 2
    Sp[:, 1, 1] += sq ** 2
    W = np.linalg.inv(Sp)
    H = np.einsum("nji,njk,nkl->il", J, W, J)
    chi2 = float(np.einsum("ni,nij,nj->", r, W, r))
    return H, chi2, n


class Verifier:
    def __init__(self, rig):
        import cv2
        from types import SimpleNamespace
        from xclean.stages.features import Frontend
        from xclean.track.pose import solve_pose, PNP_VERIFY_ENV
        from xclean.core.derive import FEATURE_SIGMA_PX
        os.environ[PNP_VERIFY_ENV] = "off"       # worker process only (see the module docstring)
        self.cv2 = cv2
        self.rig = rig
        self._prig = rig.pose_rig()
        # Frontend.match_desc is called UNBOUND on a stand-in that holds only what it reads (its BFMatcher): the
        # front end's exact matching code without building a whole front end (which needs a full rig) in the worker
        self._match_desc = Frontend.match_desc
        self._mself = SimpleNamespace(_bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False))
        self.solve_pose = solve_pose
        self.sig0 = FEATURE_SIGMA_PX
        self.n_calls = 0

    def match(self, rc, rq):
        has3d = np.isfinite(rc.X[:, 0])
        return self._match_desc(self._mself, {"descriptors": rc.desc}, {"descriptors": rq.desc}, mask0=has3d)

    def verify(self, rc, rq):
        """candidate rich set rc, query rich set rq -> dict(matches, inliers, Z (4x4) or None, H, chi2, n_info, t_ms)."""
        self.n_calls += 1
        tt = time.thread_time
        t0 = tt()
        i0, i1 = self.match(rc, rq)
        t1 = tt()
        out = {"matches": int(len(i0)), "inliers": 0, "Z": None, "H": None, "chi2": None, "n_info": 0}
        if len(i0) >= MIN_TRY:
            Xc = rc.X.astype(np.float64)
            self.cv2.setRNGSeed(PNP_SEED)
            Tq, mask, _d = self.solve_pose(Xc[i0], rq.uv[i1].astype(np.float64), self._prig, min_inliers=MIN_TRY)
            if Tq is not None:
                out["Z"] = Tq
                out["inliers"] = int(mask.sum())
                sel = np.nonzero(mask)[0]
                r = self.rig
                H, chi2, n = pnp_information(Tq, Xc[i0][sel], rq.uv[i1][sel].astype(np.float64), rq.oct[i1][sel],
                                             rc.dsig[i0][sel].astype(np.float64), rc.uv[i0][sel].astype(np.float64),
                                             r.fx, r.cx, r.cy, r.fB, self.sig0)
                out["H"], out["chi2"], out["n_info"] = H, chi2, n
        t2 = tt()
        out["t_ms"] = {"match": 1e3 * (t1 - t0), "pnp": 1e3 * (t2 - t1)}
        return out

    def covisible_inliers(self, ra, rb):
        """Are two keyframes the same place by our own verification?  -> PnP inliers of b's query against a's 3D
        (the R1 'distinct places' test: distinct when < MIN_LOOP_INLIERS)."""
        v = self.verify(ra, rb)
        return v["inliers"]
