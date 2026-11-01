#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""SO(3) and SE(3) - the layer where two REAL bugs were found before.

    from xclean.core.lie import se3_exp, se3_log, so3_project

WHY THIS FILE IS WRITTEN RATHER THAN IMPORTED
    `slam-se3-core-math-bugs` records two genuine defects in this project's
    previous core math, both proven by numerics:

      1. a **spurious 1/theta^2** in `V_inv` - the closed form already carries
         its own 1/theta^2, and multiplying by another one is wrong everywhere
         except where theta happens to be 1;
      2. **theta -> 0 cancellation** - `(1 - cos t)/t^2` and `(t - sin t)/t^3`
         are catastrophic near zero in floating point, and a keyframe pair with
         almost no rotation is the common case, not the corner case.

    Neither produced a measurable ATE change, which is exactly why they lived:
    **the estimator absorbed them.**  So they are not tested through an ATE.
    They are tested as identities on the tensors themselves - `V @ V_inv == I`,
    `exp(log(T)) == T`, `exp(Ad_T x) == T exp(x) T^-1` - at magnitudes spanning
    1e-14 to pi.  That is the same doctrine as hashing the disparity tensor
    instead of scoring a trajectory.

THE SMALL-ANGLE THRESHOLD IS DERIVED, NOT TYPED (plan section 6)
    Every series here switches at the angle where its own truncation error
    drops below machine epsilon for the dtype in use.  For a series truncated
    after the theta^4 term that is `(k*eps)**0.25`, computed from
    `finfo(dtype).eps` at call time.  A hand-typed `1e-6` would be a Class-D
    claim about universality that nobody checked, and it would be wrong for
    float32 in one direction and wasteful for float64 in the other.

CONVENTIONS, STATED ONCE AND NEVER ASSUMED AGAIN
    * twist ordering is **[rho (translation), phi (rotation)]**, 6-vector;
    * `T` is 4x4 homogeneous, `T_world_cam` unless a name says otherwise;
    * `so3_exp(phi)` rotates by |phi| about phi/|phi|, right-handed;
    * everything is float64 here.  The float32 square-root backend (plan 5.3)
      is a BACKEND question; the geometry layer is not where precision is
      bought, and P0 measured that these shapes are launch-bound anyway.
"""
from __future__ import annotations

import numpy as np


def _series_threshold(dtype=np.float64):
    """The angle below which the SERIES is used, balanced against the CLOSED
    FORM's cancellation - derived, never typed.

    ★ The first version of this function got the derivation wrong and the
    regression harness caught it at exactly the switch point (2.4e-10 error at
    theta = 6.3e-4).  Recording why, because the mistake is the natural one:

    Balancing only the SERIES truncation error against eps gives
    `(720 eps)**0.25` = 6.3e-4 - and it is the wrong criterion, because just
    ABOVE that angle the closed form is the one that is inaccurate.
    `(1 - cos t)` at t = 6e-4 subtracts two numbers within 2e-7 of each other,
    so its relative error is ~2 eps / t^2 ~ 1e-9.  The threshold has to sit
    where the two error sources CROSS, not where one of them vanishes.

        closed-form cancellation   ~ 2 eps / t^2      (grows as t -> 0)
        4-term series truncation   ~ t^8 / 3628800    (grows as t -> inf)
        crossing                   t = (7257600 eps) ** 0.1

    which is 0.126 rad in float64 and puts both error sources at ~3e-14.  Each
    series below therefore carries FOUR terms, not three, and one threshold
    serves all of them because they cross within a factor of two of each other.
    """
    return float((7257600.0 * np.finfo(dtype).eps) ** 0.1)


def _zero_threshold(dtype=np.float64):
    """Where an angle is indistinguishable from zero: t^2/6 < eps."""
    return float(np.sqrt(6.0 * np.finfo(dtype).eps))


TAYLOR_ANGLE = _series_threshold(np.float64)    # 0.126 rad in float64
ANGLE_ZERO = _zero_threshold(np.float64)        # 3.65e-08 rad in float64


def hat(v):
    """so(3) generator.  Accepts (...,3) and returns (...,3,3)."""
    v = np.asarray(v, dtype=np.float64)
    z = np.zeros(v.shape[:-1] + (3, 3))
    z[..., 0, 1] = -v[..., 2]
    z[..., 0, 2] = v[..., 1]
    z[..., 1, 0] = v[..., 2]
    z[..., 1, 2] = -v[..., 0]
    z[..., 2, 0] = -v[..., 1]
    z[..., 2, 1] = v[..., 0]
    return z


def vee(M):
    M = np.asarray(M, dtype=np.float64)
    return np.stack([M[..., 2, 1] - M[..., 1, 2],
                     M[..., 0, 2] - M[..., 2, 0],
                     M[..., 1, 0] - M[..., 0, 1]], axis=-1) * 0.5


# --------------------------------------------------------------------------
# the three coefficient functions everything else is built from.
# Each returns the closed form away from zero and the SERIES near it, and the
# switch happens at the derived angle above.  Writing them once, here, is what
# stops the cancellation bug from being reintroduced in the fourth place.
# --------------------------------------------------------------------------
def _coeff_a(t2):
    """sin(t)/t  -> 1 - t^2/6 + t^4/120 - t^6/5040"""
    t = np.sqrt(t2)
    small = t < TAYLOR_ANGLE
    out = np.empty_like(t2)
    x = t2[small]
    out[small] = 1.0 - x / 6.0 + x ** 2 / 120.0 - x ** 3 / 5040.0
    ns = ~small
    out[ns] = np.sin(t[ns]) / t[ns]
    return out


def _coeff_b(t2):
    """(1 - cos t)/t^2 -> 1/2 - t^2/24 + t^4/720 - t^6/40320"""
    t = np.sqrt(t2)
    small = t < TAYLOR_ANGLE
    out = np.empty_like(t2)
    x = t2[small]
    out[small] = 0.5 - x / 24.0 + x ** 2 / 720.0 - x ** 3 / 40320.0
    ns = ~small
    out[ns] = (1.0 - np.cos(t[ns])) / t2[ns]
    return out


def _coeff_c(t2):
    """(t - sin t)/t^3 -> 1/6 - t^2/120 + t^4/5040 - t^6/362880"""
    t = np.sqrt(t2)
    small = t < TAYLOR_ANGLE
    out = np.empty_like(t2)
    x = t2[small]
    out[small] = 1.0 / 6.0 - x / 120.0 + x ** 2 / 5040.0 - x ** 3 / 362880.0
    ns = ~small
    out[ns] = (t[ns] - np.sin(t[ns])) / (t2[ns] * t[ns])
    return out


# ------------------------------------------------------------------ SO(3) ---
def so3_exp(phi):
    """Rodrigues.  (...,3) -> (...,3,3)."""
    phi = np.atleast_2d(np.asarray(phi, dtype=np.float64))
    squeeze = phi.ndim == 2 and phi.shape[0] == 1
    t2 = np.sum(phi * phi, axis=-1)
    K = hat(phi)
    a = _coeff_a(t2)[..., None, None]
    b = _coeff_b(t2)[..., None, None]
    R = np.eye(3) + a * K + b * (K @ K)
    return R[0] if squeeze else R


def _quat_from_R(R):
    """Rotation matrix -> unit quaternion [w, x, y, z], Shepperd's method.

    The branch is chosen by the LARGEST of {trace, R00, R11, R22}, which is
    what makes it stable everywhere including at theta = pi, where the trace
    form alone has no information left: `cos` is flat there, so recovering an
    angle from the trace loses every digit exactly on the rows that turn
    around - corridor loop closures and any rig that reverses.
    """
    R = np.asarray(R, dtype=np.float64)
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    q = np.zeros(R.shape[:-2] + (4,))

    cand = np.stack([tr, m00, m11, m22], axis=-1)
    which = np.argmax(cand, axis=-1)
    which = np.where(tr > 0, 0, np.maximum(which, 1))

    def sel(k):
        return which == k

    b = sel(0)
    if np.any(b):
        t = np.sqrt(np.maximum(tr[b] + 1.0, 0.0)) * 2.0
        q[b, 0] = 0.25 * t
        q[b, 1] = (R[b, 2, 1] - R[b, 1, 2]) / t
        q[b, 2] = (R[b, 0, 2] - R[b, 2, 0]) / t
        q[b, 3] = (R[b, 1, 0] - R[b, 0, 1]) / t
    for k, (a, c, d) in enumerate(((0, 1, 2), (1, 2, 0), (2, 0, 1)), start=1):
        b = sel(k)
        if not np.any(b):
            continue
        t = np.sqrt(np.maximum(1.0 + R[b, a, a] - R[b, c, c] - R[b, d, d],
                               0.0)) * 2.0
        q[b, 0] = (R[b, d, c] - R[b, c, d]) / t
        q[b, 1 + a] = 0.25 * t
        q[b, 1 + c] = (R[b, c, a] + R[b, a, c]) / t
        q[b, 1 + d] = (R[b, d, a] + R[b, a, d]) / t
    # canonical hemisphere: w >= 0 <=> theta in [0, pi]
    flip = q[..., 0] < 0
    q[flip] = -q[flip]
    return q


def so3_log(R):
    """(...,3,3) -> (...,3).  Branch-safe at theta = 0 AND at theta = pi.

    Routed through the quaternion rather than through `arccos((tr-1)/2)`:

      * at theta -> pi the trace form has no precision left (cos is flat), and
        the first version of this file returned errors of **6.2 rad** at
        theta = pi - 1e-6.  The regression harness caught it; no ATE would
        have, because the estimator would simply have taken a bad step;
      * `2 * atan2(|q_vec|, q_w)` is well conditioned at both ends, which is
        what a corridor loop closure and any reversing rig actually need.
    """
    R = np.asarray(R, dtype=np.float64)
    single = R.ndim == 2
    R = R[None] if single else R
    q = _quat_from_R(R)
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1)
    theta = 2.0 * np.arctan2(n, q[..., 0])
    scale = np.empty_like(theta)
    tiny = n < ANGLE_ZERO
    # theta/n -> 2/q_w as n -> 0, and q_w -> 1, so the limit is simply 2
    scale[tiny] = 2.0
    scale[~tiny] = theta[~tiny] / n[~tiny]
    out = v * scale[..., None]
    return out[0] if single else out


def so3_project(R):
    """Nearest rotation in Frobenius norm.  The repair for the doubling bug.

    An earlier engine of ours had a whole class of failure whose tell was
    `rot_angle(A, A) != 0`: repeated R-products drift off SO(3), and once a
    matrix is 1e-7 off orthonormal every angle derived from it is wrong in a
    way that looks like a tracking failure rather than like a numerical one.
    Projection is cheap and it is applied wherever rotations are accumulated.
    """
    R = np.asarray(R, dtype=np.float64)
    U, _s, Vt = np.linalg.svd(R)
    D = np.eye(3) * np.ones(R.shape[:-2] + (1, 1))
    D[..., 2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def rot_angle(A, B=None):
    """Geodesic angle between two rotations, radians - via atan2, not arccos.

    `rot_angle(A, A) != 0` was the TELL for an earlier engine's doubling bug, so this
    function has to be trustworthy precisely where the answer is near zero -
    and `arccos` is at its worst there.  `arccos(1 - d)` ~ sqrt(2d), so an
    orthonormality error of 2e-16 reads as **1.4e-8 rad** of rotation that does
    not exist.  The atan2 form reads the same matrix as ~1e-17.
    """
    A = np.asarray(A, dtype=np.float64)
    if B is None:
        B = np.eye(3)
    M = np.swapaxes(A, -1, -2) @ np.asarray(B, dtype=np.float64)
    c = (np.trace(M, axis1=-2, axis2=-1) - 1.0) * 0.5
    sv = np.stack([M[..., 2, 1] - M[..., 1, 2],
                   M[..., 0, 2] - M[..., 2, 0],
                   M[..., 1, 0] - M[..., 0, 1]], axis=-1) * 0.5
    return np.arctan2(np.linalg.norm(sv, axis=-1), c)


# ------------------------------------------------------------------ SE(3) ---
def _V(phi):
    """The left Jacobian of SO(3): the translation coupling in se3_exp."""
    phi = np.atleast_2d(phi)
    t2 = np.sum(phi * phi, axis=-1)
    K = hat(phi)
    b = _coeff_b(t2)[..., None, None]
    c = _coeff_c(t2)[..., None, None]
    return np.eye(3) + b * K + c * (K @ K)


def _V_inv(phi):
    """The inverse, in the form that does NOT carry a second 1/theta^2.

        V^-1 = I - 1/2 K + (1/t^2)(1 - (t sin t)/(2(1 - cos t))) K^2

    Bug 1 lived here: the parenthesised term is already the whole coefficient
    and multiplying it by another 1/t^2 is wrong at every angle except 1 rad.
    The series near zero is 1/12 + t^2/720, and it is used below the derived
    threshold because the closed form is 0/0 there.
    """
    phi = np.atleast_2d(phi)
    t2 = np.sum(phi * phi, axis=-1)
    t = np.sqrt(t2)
    K = hat(phi)
    coef = np.empty_like(t2)
    small = t < TAYLOR_ANGLE
    x = t2[small]
    coef[small] = (1.0 / 12.0 + x / 720.0 + x ** 2 / 30240.0
                   + x ** 3 / 1209600.0)
    ns = ~small
    coef[ns] = (1.0 - (t[ns] * np.sin(t[ns]))
                / (2.0 * (1.0 - np.cos(t[ns])))) / t2[ns]
    return np.eye(3) - 0.5 * K + coef[..., None, None] * (K @ K)


def se3_exp(xi):
    """(...,6) [rho, phi] -> (...,4,4)."""
    xi = np.atleast_2d(np.asarray(xi, dtype=np.float64))
    squeeze = xi.shape[0] == 1 and np.ndim(xi) == 2
    rho, phi = xi[..., :3], xi[..., 3:]
    R = so3_exp(phi)
    R = R[None] if R.ndim == 2 else R
    t = np.einsum("...ij,...j->...i", _V(phi), rho)
    T = np.zeros(xi.shape[:-1] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T[0] if squeeze else T


def se3_log(T):
    """(...,4,4) -> (...,6) [rho, phi]."""
    T = np.asarray(T, dtype=np.float64)
    single = T.ndim == 2
    T = T[None] if single else T
    phi = so3_log(T[..., :3, :3])
    phi = phi[None] if phi.ndim == 1 else phi
    rho = np.einsum("...ij,...j->...i", _V_inv(phi), T[..., :3, 3])
    out = np.concatenate([rho, phi], axis=-1)
    return out[0] if single else out


def se3_inv(T):
    T = np.asarray(T, dtype=np.float64)
    out = np.zeros_like(T)
    Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def adjoint(T):
    """Ad_T, 6x6, in the [rho, phi] ordering declared at the top."""
    T = np.asarray(T, dtype=np.float64)
    single = T.ndim == 2
    T = T[None] if single else T
    R, t = T[..., :3, :3], T[..., :3, 3]
    A = np.zeros(T.shape[:-2] + (6, 6))
    A[..., :3, :3] = R
    A[..., :3, 3:] = hat(t) @ R
    A[..., 3:, 3:] = R
    return A[0] if single else A


def so3_jr(phi):
    """Right Jacobian of SO(3).  Jr(phi) = Jl(-phi)."""
    phi = np.atleast_2d(np.asarray(phi, dtype=np.float64))
    return _V(-phi)


def so3_jr_inv(phi):
    phi = np.atleast_2d(np.asarray(phi, dtype=np.float64))
    return _V_inv(-phi)


def se3_compose(A, B):
    return np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64)


def se3_from_Rt(R, t):
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T
