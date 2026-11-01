# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.3 geometry: SO(3) / SE(3) helpers, relative-pose composition with covariance, the PSD repair of window edges.

Own code, ported VERBATIM (same formulas, same operation order) from the P1.3 design prototype
 so the
build reproduces the design's measured numbers.  Conventions (xclean/vio/CONVENTIONS.md):
  * a pose T is 4x4 (R, p); LIVE-ODOM / LIVE-MAP poses are T_world_cam0_rectified;
  * a RELATIVE pose Z_ab = T_a^-1 T_b carries the perturbation used by P1's window covariance: rotation RIGHT
    (R <- R Exp(dphi)), translation ADDITIVE in frame a; its covariance is ordered [rotation, translation];
  * the head filter works on right-SE(3) perturbations ordered [rho, phi] (T <- T Exp([rho, phi]), rho = R^T dt);
    `to_right` converts a [phi, t] relative covariance to that form.
"""
from __future__ import annotations

import numpy as np


def so3_exp(w):
    w = np.asarray(w, float)
    th = np.linalg.norm(w, axis=-1, keepdims=True)
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -w[..., 2], w[..., 1]
    K[..., 1, 0], K[..., 1, 2] = w[..., 2], -w[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -w[..., 1], w[..., 0]
    th2 = th[..., None]
    small = th2 < 1e-10
    a = np.where(small, 1.0, np.sin(th2) / np.where(small, 1.0, th2))
    b = np.where(small, 0.5, (1 - np.cos(th2)) / np.where(small, 1.0, th2 ** 2))
    I = np.broadcast_to(np.eye(3), K.shape)
    return I + a * K + b * (K @ K)


def so3_log(R):
    R = np.asarray(R, float)
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(c)
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(th)
    f = np.where(th < 1e-8, 0.5, th / (2 * np.where(np.abs(s) < 1e-12, 1.0, s)))
    out = f[..., None] * w
    near = np.pi - th < 1e-6
    if np.any(near):
        for k in np.nonzero(np.atleast_1d(near))[0]:
            Rk = R[k] if R.ndim == 3 else R
            A = 0.5 * (Rk + np.eye(3))
            ax = np.sqrt(np.maximum(np.diag(A), 0))
            m = int(np.argmax(ax))
            ax = A[:, m] / max(ax[m], 1e-12)
            v = np.pi * ax / np.linalg.norm(ax)
            if R.ndim == 3:
                out[k] = v
            else:
                out = v
    return out


def skew(v):
    v = np.asarray(v, float)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -v[..., 2], v[..., 1]
    K[..., 1, 0], K[..., 1, 2] = v[..., 2], -v[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -v[..., 1], v[..., 0]
    return K


def huber_k_dim(D, k1=1.345):
    """Tail-matched Huber threshold for a D-dimensional whitened residual norm (closer.huber_k_dim's rule)."""
    from scipy.stats import chi2, norm
    return float(np.sqrt(chi2.isf(2.0 * float(norm.cdf(-k1)), int(D))))


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def ortho(T):
    U, _s, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = T.copy()
    T[:3, :3] = R
    return T


def se3(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def to_right(R, S):
    """(dphi right, dt additive in the left frame) covariance of a relative pose with rotation R -> right-SE(3)
    [rho, phi] covariance (rho = R^T dt)."""
    K = np.zeros((6, 6))
    K[:3, 3:] = R.T
    K[3:, :3] = np.eye(3)
    return K @ S @ K.T


def Ad(T):
    """SE(3) adjoint for [rho, phi] ordering."""
    R, t = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[:3, 3:] = skew(t) @ R
    A[3:, 3:] = R
    return A


def logr(T):
    """[rho, phi] first-order log of a near-identity SE(3) (rho = t)."""
    return np.concatenate([T[:3, 3], so3_log(T[:3, :3])])


def expr(x):
    T = np.eye(4)
    T[:3, :3] = so3_exp(x[3:])
    T[:3, 3] = x[:3]
    return T


def boxminus(Za, Zb):
    """e with Z_a = Z_b [+] e (rotation right, translation additive) -> [phi, t]."""
    return np.concatenate([so3_log(Zb[:3, :3].T @ Za[:3, :3]), Za[:3, 3] - Zb[:3, 3]])


def compose(A, B):
    """First-order composition of two relative poses with covariance (rotation right, translation additive in the left
    frame; associative).  A, B = (R, p, S[phi, t])."""
    R1, p1, S1 = A
    R2, p2, S2 = B
    J1 = np.zeros((6, 6))
    J1[:3, :3] = R2.T
    J1[3:, :3] = -R1 @ skew(p2)
    J1[3:, 3:] = np.eye(3)
    J2 = np.zeros((6, 6))
    J2[:3, :3] = np.eye(3)
    J2[3:, 3:] = R1
    return R1 @ R2, p1 + R1 @ p2, J1 @ S1 @ J1.T + J2 @ S2 @ J2.T


def psd_repair(S):
    """Window edge blocks that are not positive definite are replaced by the last
    valid block (numerical guard, counted)."""
    S = S.copy()
    n_rep, last = 0, None
    for k in range(len(S)):
        Sk = 0.5 * (S[k] + S[k].T)
        ok = np.isfinite(Sk).all() and np.linalg.eigvalsh(np.nan_to_num(Sk)).min() > 0
        if ok:
            last = Sk
            S[k] = Sk
        else:
            n_rep += 1
            S[k] = last if last is not None else np.diag(np.abs(np.nan_to_num(np.diag(Sk))) + 1e-12)
    return S, n_rep


def psd_ok(Sk):
    Sk = 0.5 * (Sk + Sk.T)
    return bool(np.isfinite(Sk).all() and np.linalg.eigvalsh(np.nan_to_num(Sk)).min() > 0)


def body_to_cam_rel(ZR, Zp, S, Rbc, pbc):
    """A body-frame relative pose (rotation right, translation additive) and its covariance -> the rectified-camera
    relative pose (Z_c = T_bc^-1 Z_b T_bc), as the design's p13_drift_cov.load_edges.  S may be None (unknown)."""
    ZcR = Rbc.T @ ZR @ Rbc
    Zcp = Rbc.T @ (ZR @ pbc + Zp - pbc)
    if S is None:
        return ZcR, Zcp, None
    J = np.zeros((6, 6))
    J[:3, :3] = Rbc.T
    J[3:, :3] = -Rbc.T @ ZR @ skew(pbc)
    J[3:, 3:] = Rbc.T
    Sc = J @ S @ J.T
    return ZcR, Zcp, 0.5 * (Sc + Sc.T)


def rel_cov(Ra, pa, Rb, pb, Caa, Cab, Cbb):
    """Relative body pose a -> b and its covariance from the window's joint covariance (P1 perturbation: R right,
    p additive in the world); the design hook's p13_hook.rel_cov."""
    Rab = Ra.T @ Rb
    pab = Ra.T @ (pb - pa)
    Ja = np.zeros((6, 6))
    Ja[:3, :3] = -Rab.T
    Ja[3:, :3] = skew(pab)
    Ja[3:, 3:] = -Ra.T
    Jb = np.zeros((6, 6))
    Jb[:3, :3] = np.eye(3)
    Jb[3:, 3:] = Ra.T
    S = Ja @ Caa @ Ja.T + Ja @ Cab @ Jb.T + Jb @ Cab.T @ Ja.T + Jb @ Cbb @ Jb.T
    return Rab, pab, 0.5 * (S + S.T)


class ChainPrefix:
    """Pose chain C_k = Z_0 ... Z_{k-1} and world-frame (left, about the chain origin) covariance prefix sums: rel(a, b)
    in O(1) (first order)."""

    def __init__(self, ZR, Zp, S):
        n = len(ZR) + 1
        self.R = np.zeros((n, 3, 3))
        self.p = np.zeros((n, 3))
        self.P = np.zeros((n, 6, 6))
        self.R[0] = np.eye(3)
        for k in range(n - 1):
            self._step(k, ZR[k], Zp[k], S[k])

    def _step(self, k, ZR, Zp, S):
        self.R[k + 1] = self.R[k] @ ZR
        self.p[k + 1] = self.p[k] + self.R[k] @ Zp
        M = np.zeros((6, 6))
        M[:3, :3] = self.R[k + 1]
        M[3:, :3] = skew(self.p[k + 1]) @ self.R[k + 1]
        M[3:, 3:] = self.R[k]
        self.P[k + 1] = self.P[k] + M @ S @ M.T

    def rel(self, a, b):
        Ra, pa, Rb, pb = self.R[a], self.p[a], self.R[b], self.p[b]
        ZR = Ra.T @ Rb
        Zp = Ra.T @ (pb - pa)
        N = np.zeros((6, 6))
        N[:3, :3] = Rb.T
        N[3:, :3] = -Ra.T @ skew(pb)
        N[3:, 3:] = Ra.T
        S = N @ (self.P[b] - self.P[a]) @ N.T
        return ZR, Zp, 0.5 * (S + S.T)


def slew_step(T_app, T_tgt, To, budget_t, budget_r):
    """Move the applied map transform toward the target by at most (budget_t m, budget_r rad), measured AT THE HEAD
    (T_app To -> T_tgt To).  -> (new applied transform, reached)."""
    H = T_app @ To
    E = inv(H) @ (T_tgt @ To)
    phi = so3_log(E[:3, :3])
    t = E[:3, 3]
    nt, nr = float(np.linalg.norm(t)), float(np.linalg.norm(phi))
    a = 1.0
    if nt > 1e-12:
        a = min(a, budget_t / nt)
    if nr > 1e-12:
        a = min(a, budget_r / nr)
    if a >= 1.0:
        return ortho(T_tgt), True
    Es = np.eye(4)
    Es[:3, :3] = so3_exp(a * phi)
    Es[:3, 3] = a * t
    return ortho(H @ Es @ inv(To)), False
