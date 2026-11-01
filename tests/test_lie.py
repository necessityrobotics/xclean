# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Core SE(3)/SO(3) math (xclean/core/lie.py): identities on tensors, not ATEs.

A wrong Jacobian or a branch error near 0 or pi barely moves an ATE (the solver takes a worse step and still
converges), so these checks test identities directly, swept across magnitudes from 1e-14 to pi, where such bugs
live.  Adapted from the engine's internal regression harness (same checks, same tolerances).
"""
import numpy as np
import pytest

from xclean.core import lie
from xclean.core.lie import (TAYLOR_ANGLE, hat, vee, so3_exp, so3_log, so3_project, rot_angle, se3_exp, se3_log,
                             se3_inv, adjoint, so3_jr, so3_jr_inv, _V, _V_inv)

# magnitudes chosen to straddle the derived series threshold from both sides
MAGS = [0.0, 1e-14, 1e-10, 1e-8, 1e-6, TAYLOR_ANGLE * 0.5, TAYLOR_ANGLE, TAYLOR_ANGLE * 2, 1e-2, 0.1, 1.0, 2.0,
        3.0, np.pi - 1e-6, np.pi - 1e-9]


def _axes(seed=5, n=12):
    rng = np.random.RandomState(seed)
    a = rng.standard_normal((n, 3))
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_V_times_Vinv_is_identity():
    worst = max(float(np.abs(_V(ax * t)[0] @ _V_inv(ax * t)[0] - np.eye(3)).max()) for t in MAGS for ax in _axes())
    assert worst < 1e-12, worst


def test_Jr_times_Jrinv_is_identity():
    worst = max(float(np.abs(so3_jr(ax * t)[0] @ so3_jr_inv(ax * t)[0] - np.eye(3)).max())
                for t in MAGS for ax in _axes(7))
    assert worst < 1e-12, worst


def test_exp_log_roundtrip_so3_including_near_pi():
    worst = max(float(np.abs(so3_log(so3_exp(ax * t)) - ax * t).max()) for t in MAGS for ax in _axes(11))
    assert worst < 1e-8, worst


def test_exp_log_roundtrip_se3():
    rng = np.random.RandomState(3)
    worst = 0.0
    for t in MAGS:
        for ax in _axes(13, 8):
            xi = np.concatenate([rng.standard_normal(3) * 2.0, ax * t])
            worst = max(worst, float(np.abs(se3_log(se3_exp(xi)) - xi).max()))
    assert worst < 1e-8, worst


def test_no_cancellation_near_zero():
    worst = 0.0
    for t in (1e-14, 1e-12, 1e-10, 1e-9, 1e-8, 1e-7):
        phi = np.array([t, 0.0, 0.0])
        worst = max(worst, float(np.abs(_V(phi)[0] - (np.eye(3) + 0.5 * hat(phi))).max()) / t)
    assert worst < 1e-6, worst


def test_adjoint_identity():
    """exp(Ad_T xi) == T exp(xi) T^-1 catches a [rho, phi] vs [phi, rho] ordering mistake."""
    rng = np.random.RandomState(17)
    worst = 0.0
    for _ in range(50):
        T = se3_exp(rng.standard_normal(6))
        xi = rng.standard_normal(6) * 0.3
        worst = max(worst, float(np.abs(se3_exp(adjoint(T) @ xi) - T @ se3_exp(xi) @ se3_inv(T)).max()))
    assert worst < 1e-10, worst


@pytest.mark.parametrize("t", [1e-6, 1e-3, 0.1, 1.0, 2.5])
def test_right_jacobian_against_finite_differences(t):
    phi = _axes(31, 1)[0] * t
    R = so3_exp(phi)
    J = np.zeros((3, 3))
    h = 1e-7
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        J[:, k] = so3_log(R.T @ so3_exp(phi + d)) / h
    assert float(np.abs(J - so3_jr(phi)[0]).max()) < 1e-5


def test_rot_angle_of_a_matrix_with_itself_is_exactly_zero():
    rng = np.random.RandomState(29)
    for _ in range(200):
        A = so3_exp(rng.standard_normal(3) * 2)
        assert rot_angle(A, A) == 0.0


def test_projection_repairs_accumulated_drift():
    rng = np.random.RandomState(37)
    A = so3_exp(rng.standard_normal(3) * 0.4)
    R = np.eye(3)
    for _ in range(20000):
        R = R @ A
    P = so3_project(R)
    assert float(np.abs(P.T @ P - np.eye(3)).max()) < 1e-14


def test_hat_vee_roundtrip():
    v = np.random.RandomState(41).standard_normal((100, 3))
    assert np.abs(vee(hat(v)) - v).max() < 1e-15


def test_inverse_is_an_inverse():
    T = se3_exp(np.random.RandomState(43).standard_normal((50, 6)))
    assert np.abs(T @ se3_inv(T) - np.eye(4)).max() < 1e-12


def test_batch_matches_single_bit_for_bit():
    xi = np.random.RandomState(47).standard_normal((37, 6))
    B = se3_exp(xi)
    assert all(np.array_equal(B[i], se3_exp(xi[i])) for i in range(37))
    lg = se3_log(B)
    assert all(np.array_equal(lg[i], se3_log(B[i])) for i in range(37))


def test_threshold_is_derived_not_typed():
    f32 = lie._series_threshold(np.float32)
    f64 = lie._series_threshold(np.float64)
    assert f64 < f32 < np.pi and 0.05 < f64 < 0.5
