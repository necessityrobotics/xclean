#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1 IMU FACTOR: on-manifold preintegration, the 15-row inertial residual with analytic Jacobians and square-root
information, and the causal IMU prediction for the frame path.  numba, float64, CPU only.

    from xclean.vio.imu_factor import ImuNoise, State, preintegrate, imu_residual_jacobians, propagate
    noise = ImuNoise.from_calib(calib)                         # Class A; refused if not declared
    pre = preintegrate(t, gyro, acc, t_i, t_j, horizon, bg_lin, ba_lin, noise)
    r, Ji, Jj = imu_residual_jacobians(X_i, X_j, pre, g_w, kappa=kappa_imu)     # whitened, 15 rows
    pred = propagate(X_k, t, gyro, acc, t_k, t_new, horizon, g_w, noise=noise, P=P_k)

Design reference (cited, not copied): Forster, Carlone, Dellaert, Scaramuzza, "On-Manifold Preintegration for
Real-Time Visual-Inertial Odometry", T-RO 33(1), 2017.  Carrying the bias random walk inside the propagated
covariance (15 x 15, with its cross terms) follows the published OKVIS formulation (Leutenegger et al., IJRR 2015) as
a design reference.  This is our own code; nothing is taken from OKVIS2 / Basalt / ORB-SLAM3 / VINS source.
Place in the design: an internal note §3 steps 1-2, §4.2 "IMU", §6.3 (kappa_imu);
the causal rule of our design notes.  Pose conventions: xclean/vio/CONVENTIONS.md.

THE NATIVE PARAMETRISATION (= xclean/vio/CONVENTIONS.md "pose block" and "VI frame states")
    state X = (R_wb, p_wb, v_w, bg, ba); R_wb maps body to world (w = the segment's odom frame).
    error state dx = [dphi, dp, dv, dbg, dba] (15):
        R <- R Exp(dphi)  (RIGHT, body frame);  p <- p + dp;  v <- v + dv;  b <- b + db   (p, v additive in w)
    residual r = [rR, rv, rp, rbg, rba] (15), prediction minus measurement:
        rR  = Log( (dR Exp(JRg dbg))^T R_i^T R_j )
        rv  = R_i^T (v_j - v_i - g dt) - (dv + Jvg dbg + Jva dba)
        rp  = R_i^T (p_j - p_i - v_i dt - g dt^2 / 2) - (dp + Jpg dbg + Jpa dba)
        rbg = bg_j - bg_i,   rba = ba_j - ba_i,      with dbg = bg_i - bg_lin, dba = ba_i - ba_lin
    J_i = dr/d(dx_i), J_j = dr/d(dx_j): 15 x 15, rows in residual order, columns in error-state order.  Exact
    derivatives (Jr^-1(rR) and the Jr of the bias correction kept, not approximated by I).  J_g = dr/dg (15 x 3,
    additive) for the window's gravity chart.  Normal equations as CONVENTIONS.md: H = J^T J, b = J^T r.

THE T_cw LEFT ADAPTER (deliverable 4; also preint.py's global-BA chart)
    pose T_cw (camera <- world), T <- [Exp(dw) | dt] T, dx = [dt(0:3), dw(3:6)]: the update form of
    an internal script update_poses (V = I; to first order identical to
    Exp_se3([dt, dw]) T, so the Jacobians at dx = 0 are the same).  With T_bc = body <- cam0 (the RECTIFIED cam0,
    CONVENTIONS.md "Extrinsic"), T_wb = T_cw^-1 T_bc^-1 and the native pose perturbation is
        dphi = -R_bc dw
        dp   = -R_wb R_bc dt - R_wb [t_bc]x R_bc dw
    so J_chart[:, 0:6] = J_native[:, 0:6] @ chart_tcw_left(R_wb, T_bc); columns 6:15 (v, bg, ba) are unchanged.
    NOTE: the bench's own (Rw, tw) project a point as Rw^T (X - tw), i.e. they are T_wc (camera -> world), not T_cw.
    Its update_poses is therefore a LEFT perturbation of T_wc, which is chart_twc_left:
        dphi = R_wb^T dw,  dp = dt - [p_wb]x dw          (independent of T_bc)
    Both charts are provided and both are pinned by finite differences (tests/test_vio_imu_factor.py (a)).

THE INTEGRATION RULE (stated; identical to backend/preint.py inside the released span)
    Sub-interval edges: t0, every RELEASED sample strictly inside (t0, min(t1, t_end)), min(t1, t_end), and t1 when
    t1 > t_end (t_end = the stamp of the last released sample).  Both ends are clipped exactly; nothing is snapped
    to the sample grid.  On each sub-interval the gyro and accelerometer are the LINEAR INTERPOLATION of the two
    released samples bracketing its MIDPOINT (np.interp's formula), the rotation is advanced by Exp(w h) and the
    acceleration is rotated by the MIDPOINT attitude dR Exp(w h / 2): the midpoint rule, second order (preint.py's
    measured lesson: the right-endpoint rule lags by half a sample).  Zero-length sub-intervals are skipped exactly;
    there is NO edge-merge constant (preint.py merges edges closer than 1e-12 s; the difference is < 1e-11).
    The bias Jacobians are the EXACT derivatives of this discrete scheme (preint.py's recursion evaluates them at the
    start-of-step attitude, a first-order approximation of its own midpoint scheme).

THE CAUSAL RULE (C40 / A3)
    Only samples with t <= horizon are released.  The values (gyro, acc) of any later sample are never read; the
    stamp array is searched to find the horizon, so later stamps may hold any value > horizon (test (e) poisons them).
    Beyond t_end the last released sample is HELD (zero-order, no constant) over (t_end, t1]: `extrapolated` is set
    and `extrap_span` = t1 - max(t0, t_end).  An interval wholly past t_end gives dR = Exp((g_last - bg) (t1 - t0)),
    exactly LiveImu.delta_rotation_body's C40 hold.  The held sample's NOISE is in the covariance (sample-support rule
    below: a hold longer than T_nom is NOT over-confident); the held span's MODEL error (the signal moving during the
    hold) is NOT: the caller sees the flag and the span.
    n_inner counts the released samples strictly inside (t0, t1), the held one included when t0 < t_end < t1.

THE COVARIANCE (residual order [rR, rv, rp, rbg, rba], 15 x 15, joint)
    The declared densities are continuous-time (Class A, calib `imu_noise`): white sigma_g, sigma_a and bias random
    walks rw_g, rw_a.  Per sub-interval of length h the white noise is discretised EXACTLY as continuous white noise
    through the sub-interval: rotation block sigma_g^2 h I (rotation-invariant, exact), accelerometer-driven v/p
    blocks sigma_a^2 [h, h^2/2; h^2/2, h^3/3] (exact), and the gyro -> v/p cross terms to first order in the attitude
    change within the sub-interval.  The bias random walk is carried as two augmented 3-states (the bias deviation
    from its value at t0), so the preintegrated block, the bias-residual block (rw^2 dt) and their CROSS terms come
    out of one propagation (within-sub-interval drift included in closed form).  The transition is the exact
    derivative of the discrete scheme (the same matrices as the bias Jacobians).  The record stores the UNSCALED
    covariance; kappa_imu and BIASCOV enter at the whitening (below).

THE SAMPLE-SUPPORT RULE (the white noise of a SAMPLED sensor; fix of the hold / gap over-confidence, 2026-10-23)
    The densities are continuous, but the stream is sampled: in the declared (Kalibr) model every sample carries its
    own white noise of variance sigma^2 / T_nom, T_nom = 1 / imu_rate_hz.  On each sub-interval j the scheme uses a
    CONSTANT measurement (1 - u) x_a + u x_b (u = the interpolation weight at the midpoint; the hold: u = 1 on the
    held sample), so the exact first-order error is sum_j K_j m_j with K_j = int_j k(t) dt (k = the response kernel
    of the noise) and m_j = (1 - u_j) n_a + u_j n_b.  A sample enters at most TWO sub-intervals (once as the right
    end of a bracket, once as its left end or as the held sample).  Young's inequality (eps = 1) bounds the joint
    covariance by twice the per-sub-interval terms, and Jensen (int k)(int k)^T <= h int k k^T bounds each constant
    noise by continuous white noise of density sigma^2 h / T_nom.  So the continuous model of sub-interval j with its
    white densities scaled by
        s_j = max(1, nu_j h_j / T_nom),   nu_j = 2 ((1 - u_j)^2 + u_j^2)            (interpolated sub-interval)
                                          nu_j = 2 if t0 < t_end (the held sample also feeds the previous
                                                 sub-interval), else 1               (the C40 hold)
    dominates the exact error covariance of the sampled white noise, sub-interval by sub-interval, to first order.
    On a regular stream s_j = 1 exactly (full bracket: nu = 1, h = T_nom; a partial one: nu h / T_nom =
    (2 - 2x + x^2) x <= 1, x = h / T_nom), so nothing changes there.  It bites where the continuous model was
    over-confident: across a stream GAP (bracket L > T_nom: s = L / T_nom for a full bracket) and in a hold longer
    than T_nom / 2 (s = 2 h / T_nom; a wholly-held interval s = h / T_nom), where the old rule (s = 1) under-modelled
    the variance by f h.  T_nom is Class A (calib imu_rate_hz, ImuNoise.rate_hz) or, when not declared, Class C (the
    median spacing of the RELEASED stamps, measured per call - declare the rate on the frame path).  The bias random
    walk is a continuous process and is not scaled.  No constant.  Measured (test (g), exact linearised covariance):
    largest generalised eigenvalue <= 1.00005 on regular, causal, SAMPLES, hold and gap configurations, 1.0119 on a
    50 ms wholly-held interval (residual: the measured bias is frozen at the held sample's value while the walk model
    keeps walking); the old rule gave 3.2 to 10 where a hold or gap dominates the interval.

THE SAMPLES LESSON (backend/preint.py _inertial_fix): KEPT, with a full-rank conservative covariance; not refused
    preint.py's discrete form gave an interval with no interior sample an EXACTLY rank-6 9x9 covariance (one noise
    draw drives v and p, dp = dv dt/2) and 28 such Rosario factors owned 53-99.8 % of the inertial chi.  The exact
    first-order error of one interpolated sample IS rank 6, so no correct full-rank covariance exists in that model.
    Here the continuous-time discretisation is full rank for every h > 0, and for an interval no longer than the
    nominal sample period it DOMINATES the exact error covariance of the interpolated sample (accelerometer v/p block:
    det of the difference = sigma^4 T^4 (1 - x) / 12 >= 0 with x = T f c(lambda) <= 1; in general the Jensen bound of
    the sample-support rule; checked by Monte Carlo in test (c) and by the exact linearised covariance in test (g)):
    the factor is conservative there, never over-confident.  Why keep rather than refuse: in P1 this factor is the
    only link carrying v and the biases between consecutive frame states; refusing it leaves the newest state's
    velocity unconstrained for that frame.  A stream GAP (spacing > nominal) no longer breaks the domination (the
    sample-support rule scales it) but still HIDES unmeasured motion (a model error, like the hold's): `gap_max` (the
    longest released-sample spacing the interval touched) is returned so the caller applies backend/imu.py's T1-c
    rule (threshold = the measured median frame interval).

WHITENING (sqrt_information): C_used = kappa_imu * C + BIASCOV;  C_used = L L^T (lower Cholesky);  W = L^-1.
    kappa_imu (Class C, measured by the window, DESIGN_P1 §6.3) multiplies the whole 15 x 15 (a variance factor:
    sigma x sqrt(kappa)).  BIASCOV (preint.py's lesson) adds J_b diag(sigma_bg^2, sigma_ba^2) J_b^T to the 9 preint
    rows, J_b = the bias Jacobians: the uncertainty of a bias the interval was integrated at and that is NOT a free
    variable at this factor (reduces to preint.py's (sigma_bg span)^2 I on rR when JRg ~ -span I).  In the P1 window
    the biases ARE free per-state variables joined by the random-walk rows, so the default is 0 (adding it there
    would count the bias uncertainty twice); use it for a factor whose bias is held (motion-only pre-solve, IGT k_R).
    W is lower triangular, so the first 9 whitened rows are the preintegration residual whitened by its OWN marginal
    covariance: ||r_w[:9]||^2 is the 9-row chi, ||r_w||^2 the full 15-row chi.

ROBUST WEIGHT (returned separately, never baked in): robust_weight(r_w) = Huber IRLS weight on the whitened NORM at
    HUBER_K (CONVENTIONS.md "Huber on the whitened norm").  Measured caveat (test (c)): for a 15-row block the norm of
    a CONSISTENT factor is ~sqrt(15) > HUBER_K, so that rule down-weights every consistent IMU factor (median weight
    ~0.35).  rule="dof" gives the equal-tail threshold sqrt(chi2_d^-1(P(|z| <= HUBER_K))) instead (4.46 for d = 15;
    equal to HUBER_K for d = 1).  The choice is the window's; both are derived from the one Class-D constant.

CONSTANTS (no other number is set by hand in this file)
    sigma_g, sigma_a, rw_g, rw_a   A   calib imu_noise; a missing or non-positive density is REFUSED (ValueError),
                                       never substituted (preint.py's 1e-4 / 1e-3 fallbacks are not carried)
    T_nom = 1 / rate_hz            A   calib imu_rate_hz (the discrete noise model's sample period); if absent, C:
                                       the median spacing of the released stamps (backend/imu.py's rule); with fewer
                                       than two released stamps and no declared rate a covariance is REFUSED
    nu_j (1 or 2), Jensen factor   D   the sample-support rule's proof (multiplicity of a sample, Young eps = 1)
    kappa_imu                      C   input (vio/init.py measures it); default 1
    sigma_bg, sigma_ba (BIASCOV)   C   input (the run's own bias standard error); default 0 (see above)
    HUBER_K                        D   core/derive.py
    TAYLOR_ANGLE, ANGLE_ZERO       D   core/lie.py, derived from float64 eps (series switch of Exp / Jr / Jr^-1 / Log)
    huber_k_dof(d)                 D   derived from HUBER_K and the residual dimension (equal tail probability)
    no gap or edge-merge threshold (gap_max is reported; zero-length sub-intervals are skipped exactly)

DIFFERENCES FROM backend/preint.py (measured in test (d))
    dR, dv, dp: same rule inside the released span (agreement to rounding).  Covariance: continuous-time vs
    preint.py's piecewise-constant discrete noise (differs at O(1/n^2) in the p block and in O(h) cross terms) and
    preint.py adds the bias random walk to the diagonal rotation / velocity blocks only.  Bias Jacobians: exact
    derivatives vs preint.py's start-of-step first-order recursion.  Causal: preint.py interpolates the last
    sub-interval toward the next sample even when it lies past the frame (offline look-ahead); here it is held.

FOR THE WINDOW (numba to numba): preint_core, sqrt_info_core and factor_core are @njit kernels on preallocated
    arrays (see their docstrings); the Python functions below are thin wrappers.  warmup() compiles every kernel.
"""
from __future__ import annotations

import math
from collections import namedtuple

import numpy as np
from numba import njit

from xclean.core.derive import HUBER_K
from xclean.core.lie import ANGLE_ZERO, TAYLOR_ANGLE

__all__ = ["ImuNoise", "State", "Preint", "Prediction", "preintegrate", "imu_residual", "imu_residual_jacobians",
           "sqrt_information", "robust_weight", "huber_k_dof", "propagate", "chart_tcw_left", "chart_twc_left",
           "jacobian_to_chart", "retract_left", "state_from_tcw", "tcw_from_state", "state_from_twc",
           "twc_from_state", "box_minus", "box_plus", "preint_core", "sqrt_info_core", "factor_core", "warmup"]

_TA = float(TAYLOR_ANGLE)        # 0.126 rad: series below, closed form above (core/lie.py, Class D derived)
_AZ = float(ANGLE_ZERO)          # 3.65e-8 rad: Log's zero branch (core/lie.py)

# ---- status of a preintegration record ---------------------------------------------------------------------------
OK = 0
REFUSED_BAD_INTERVAL = 1         # t1 <= t0, or a non-finite end
REFUSED_NO_DATA = 2              # no sample released at all
REFUSED_BEFORE_FIRST_SAMPLE = 3  # t0 precedes the first sample: nothing measured the start
REFUSED_NONFINITE = 4            # a used sample is not finite
REFUSED_NO_RATE = 5              # a covariance was requested but T_nom is unknown (no declared rate, < 2 stamps)
STATUS_TEXT = {OK: "ok", REFUSED_BAD_INTERVAL: "t1 <= t0 or non-finite interval",
               REFUSED_NO_DATA: "no IMU sample released", REFUSED_BEFORE_FIRST_SAMPLE: "t0 before the first sample",
               REFUSED_NONFINITE: "non-finite IMU sample inside the interval",
               REFUSED_NO_RATE: "no IMU sample period (rate not declared, fewer than two released stamps)"}

# ---- preint_core info vector ---------------------------------------------------------------------------------------
I_STATUS, I_NSUB, I_NINNER, I_EXTRAP, I_TEND, I_GAPMAX, I_NREL, I_TNOM, I_SMAX = range(9)
N_INFO = 9


# =================================================================================================== numba kernels ==
@njit(cache=False, nogil=True)
def _abc(t2):
    """sin t / t, (1 - cos t) / t^2, (t - sin t) / t^3 with core/lie.py's derived series switch."""
    t = math.sqrt(t2)
    if t < _TA:
        x = t2
        a = 1.0 - x / 6.0 + x * x / 120.0 - x * x * x / 5040.0
        b = 0.5 - x / 24.0 + x * x / 720.0 - x * x * x / 40320.0
        c = 1.0 / 6.0 - x / 120.0 + x * x / 5040.0 - x * x * x / 362880.0
    else:
        s = math.sin(t)
        a = s / t
        b = (1.0 - math.cos(t)) / t2
        c = (t - s) / (t2 * t)
    return a, b, c


@njit(cache=False, nogil=True)
def _exp_jr(x, y, z, R, Jr):
    """R = Exp([x, y, z]) and Jr = its right Jacobian (core/lie.py so3_exp / so3_jr)."""
    t2 = x * x + y * y + z * z
    a, b, c = _abc(t2)
    xy = x * y
    xz = x * z
    yz = y * z
    k00 = -(y * y + z * z)
    k11 = -(x * x + z * z)
    k22 = -(x * x + y * y)
    R[0, 0] = 1.0 + b * k00
    R[0, 1] = -a * z + b * xy
    R[0, 2] = a * y + b * xz
    R[1, 0] = a * z + b * xy
    R[1, 1] = 1.0 + b * k11
    R[1, 2] = -a * x + b * yz
    R[2, 0] = -a * y + b * xz
    R[2, 1] = a * x + b * yz
    R[2, 2] = 1.0 + b * k22
    Jr[0, 0] = 1.0 + c * k00
    Jr[0, 1] = b * z + c * xy
    Jr[0, 2] = -b * y + c * xz
    Jr[1, 0] = -b * z + c * xy
    Jr[1, 1] = 1.0 + c * k11
    Jr[1, 2] = b * x + c * yz
    Jr[2, 0] = b * y + c * xz
    Jr[2, 1] = -b * x + c * yz
    Jr[2, 2] = 1.0 + c * k22


@njit(cache=False, nogil=True)
def _jr_inv(v, out):
    """Jr^-1(v) = I + K/2 + coef K^2 (core/lie.py so3_jr_inv)."""
    x = v[0]
    y = v[1]
    z = v[2]
    t2 = x * x + y * y + z * z
    t = math.sqrt(t2)
    if t < _TA:
        co = 1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0 + t2 * t2 * t2 / 1209600.0
    else:
        co = (1.0 - (t * math.sin(t)) / (2.0 * (1.0 - math.cos(t)))) / t2
    out[0, 0] = 1.0 - co * (y * y + z * z)
    out[0, 1] = -0.5 * z + co * x * y
    out[0, 2] = 0.5 * y + co * x * z
    out[1, 0] = 0.5 * z + co * x * y
    out[1, 1] = 1.0 - co * (x * x + z * z)
    out[1, 2] = -0.5 * x + co * y * z
    out[2, 0] = -0.5 * y + co * x * z
    out[2, 1] = 0.5 * x + co * y * z
    out[2, 2] = 1.0 - co * (x * x + y * y)


@njit(cache=False, nogil=True)
def _log(R, out):
    """Log of SO(3) through the quaternion (Shepperd's branch choice), core/lie.py so3_log."""
    m00 = R[0, 0]
    m11 = R[1, 1]
    m22 = R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        br = 0
    else:
        br = 0
        best = tr
        if m00 > best:
            br = 1
            best = m00
        if m11 > best:
            br = 2
            best = m11
        if m22 > best:
            br = 3
            best = m22
        if br == 0:
            br = 1
    if br == 0:
        s = math.sqrt(max(tr + 1.0, 0.0)) * 2.0
        qw = 0.25 * s
        out[0] = (R[2, 1] - R[1, 2]) / s
        out[1] = (R[0, 2] - R[2, 0]) / s
        out[2] = (R[1, 0] - R[0, 1]) / s
    else:
        if br == 1:
            a = 0
            c = 1
            d = 2
        elif br == 2:
            a = 1
            c = 2
            d = 0
        else:
            a = 2
            c = 0
            d = 1
        s = math.sqrt(max(1.0 + R[a, a] - R[c, c] - R[d, d], 0.0)) * 2.0
        qw = (R[d, c] - R[c, d]) / s
        out[a] = 0.25 * s
        out[c] = (R[c, a] + R[a, c]) / s
        out[d] = (R[d, a] + R[a, d]) / s
    if qw < 0.0:
        qw = -qw
        out[0] = -out[0]
        out[1] = -out[1]
        out[2] = -out[2]
    n = math.sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2])
    if n < _AZ:
        sc = 2.0
    else:
        sc = 2.0 * math.atan2(n, qw) / n
    out[0] *= sc
    out[1] *= sc
    out[2] *= sc


@njit(cache=False, nogil=True)
def _mm(A, B, C):
    """C = A B (3x3); C must not alias A or B."""
    for i in range(3):
        for j in range(3):
            C[i, j] = A[i, 0] * B[0, j] + A[i, 1] * B[1, j] + A[i, 2] * B[2, j]


@njit(cache=False, nogil=True)
def _mtm(A, B, C):
    """C = A^T B (3x3)."""
    for i in range(3):
        for j in range(3):
            C[i, j] = A[0, i] * B[0, j] + A[1, i] * B[1, j] + A[2, i] * B[2, j]


@njit(cache=False, nogil=True)
def _mmt(A, B, C):
    """C = A B^T (3x3)."""
    for i in range(3):
        for j in range(3):
            C[i, j] = A[i, 0] * B[j, 0] + A[i, 1] * B[j, 1] + A[i, 2] * B[j, 2]


@njit(cache=False, nogil=True)
def _mv(A, v, out):
    for i in range(3):
        out[i] = A[i, 0] * v[0] + A[i, 1] * v[1] + A[i, 2] * v[2]


@njit(cache=False, nogil=True)
def _mtv(A, v, out):
    for i in range(3):
        out[i] = A[0, i] * v[0] + A[1, i] * v[1] + A[2, i] * v[2]


@njit(cache=False, nogil=True)
def _F_left(Rt, F1, F2, h, Bg0, Bg1, Bg2, Ba1, Ba2, X, Y):
    """Y = F X for the 15 x 15 sub-interval transition (internal order [phi, v, p, eps_g, eps_a])
        F = [[Rt, 0, 0, Bg0, 0  ],
             [F1, I, 0, Bg1, Ba1],
             [F2, hI, I, Bg2, Ba2],
             [0,  0, 0, I,   0  ],
             [0,  0, 0, 0,   I  ]]"""
    m = X.shape[1]
    for c in range(m):
        for i in range(3):
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            for k in range(3):
                xph = X[k, c]
                xg = X[9 + k, c]
                xa = X[12 + k, c]
                s0 += Rt[i, k] * xph + Bg0[i, k] * xg
                s1 += F1[i, k] * xph + Bg1[i, k] * xg + Ba1[i, k] * xa
                s2 += F2[i, k] * xph + Bg2[i, k] * xg + Ba2[i, k] * xa
            Y[i, c] = s0
            Y[3 + i, c] = s1 + X[3 + i, c]
            Y[6 + i, c] = s2 + h * X[3 + i, c] + X[6 + i, c]
            Y[9 + i, c] = X[9 + i, c]
            Y[12 + i, c] = X[12 + i, c]


@njit(cache=False, nogil=True)
def _step(h, w, a, sg2, sa2, swg2, swa2, dR, dv, dp, J, P, S, Y, Z, Jn, am):
    """Advance the preintegration by one sub-interval of length h with bias-corrected measurements w, a.
    P is the internal covariance [phi, v, p, eps_g, eps_a] (errors = computed minus true; eps = bias deviation)."""
    dRk = S[0]
    Jrf = S[1]
    Rh = S[2]
    Jrh = S[3]
    dRm = S[4]
    M = S[5]
    N = S[6]
    F1 = S[7]
    F2 = S[8]
    Bg0 = S[9]
    Bg1 = S[10]
    Bg2 = S[11]
    Ba1 = S[12]
    Ba2 = S[13]
    T = S[14]
    MMt = S[15]
    Rt = S[16]
    Rn = S[17]
    wx = w[0] * h
    wy = w[1] * h
    wz = w[2] * h
    _exp_jr(wx, wy, wz, dRk, Jrf)
    _exp_jr(0.5 * wx, 0.5 * wy, 0.5 * wz, Rh, Jrh)
    _mm(dR, Rh, dRm)                 # midpoint attitude
    _mv(dRm, a, am)
    for i in range(3):               # M = dRm [a]x
        r0 = dRm[i, 0]
        r1 = dRm[i, 1]
        r2 = dRm[i, 2]
        M[i, 0] = r1 * a[2] - r2 * a[1]
        M[i, 1] = -r0 * a[2] + r2 * a[0]
        M[i, 2] = r0 * a[1] - r1 * a[0]
    _mmt(M, Rh, N)                   # N = dRm [a]x Rh^T = dR [Rh a]x
    _mm(M, Jrh, T)                   # T = M Jr(theta/2)
    h2 = h * h
    h3 = h2 * h
    h4 = h3 * h
    h5 = h4 * h
    h6 = h5 * h
    h7 = h6 * h
    for i in range(3):
        for j in range(3):
            Rt[i, j] = dRk[j, i]
            F1[i, j] = -N[i, j] * h
            F2[i, j] = -0.5 * N[i, j] * h2
            Bg0[i, j] = Jrf[i, j] * h
            Bg1[i, j] = -T[i, j] * (0.5 * h2)
            Bg2[i, j] = -T[i, j] * (0.25 * h3)
            Ba1[i, j] = dRm[i, j] * h
            Ba2[i, j] = 0.5 * dRm[i, j] * h2
    # ---- covariance: P <- F P F^T + Q --------------------------------------------------------------------------
    _F_left(Rt, F1, F2, h, Bg0, Bg1, Bg2, Ba1, Ba2, P, Y)          # Y = F P
    for i in range(15):
        for j in range(15):
            Z[i, j] = Y[j, i]                                       # Z = P F^T
    _F_left(Rt, F1, F2, h, Bg0, Bg1, Bg2, Ba1, Ba2, Z, Y)          # Y = F P F^T
    for i in range(15):
        P[i, i] = Y[i, i]
        for j in range(i + 1, 15):
            s = 0.5 * (Y[i, j] + Y[j, i])
            P[i, j] = s
            P[j, i] = s
    _mmt(M, M, MMt)
    # continuous white noise through the sub-interval (gyro -> v/p cross terms first order in the attitude change
    # inside it) + the bias random walk's drift inside it, both in closed form
    qpp = sg2 * h + swg2 * h3 / 3.0
    qvp1 = sg2 * h2 / 2.0 + swg2 * h4 / 8.0
    qpp1 = sg2 * h3 / 6.0 + swg2 * h5 / 30.0
    qvv_i = sa2 * h + swa2 * h3 / 3.0
    qvv_m = sg2 * h3 / 3.0 + swg2 * h5 / 20.0
    qpv_i = sa2 * h2 / 2.0 + swa2 * h4 / 8.0
    qpv_m = sg2 * h4 / 8.0 + swg2 * h6 / 72.0
    qpp_i = sa2 * h3 / 3.0 + swa2 * h5 / 20.0
    qpp_m = sg2 * h5 / 20.0 + swg2 * h7 / 252.0
    for i in range(3):
        for j in range(3):
            d = 1.0 if i == j else 0.0
            mm = MMt[i, j]
            mij = M[i, j]
            P[i, j] += qpp * d
            q = -mij * qvp1
            P[3 + i, j] += q
            P[j, 3 + i] += q
            q = -mij * qpp1
            P[6 + i, j] += q
            P[j, 6 + i] += q
            P[3 + i, 3 + j] += qvv_i * d + qvv_m * mm
            q = qpv_i * d + qpv_m * mm
            P[6 + i, 3 + j] += q
            P[3 + j, 6 + i] += q
            P[6 + i, 6 + j] += qpp_i * d + qpp_m * mm
            q = swg2 * h2 / 2.0 * d
            P[i, 9 + j] += q
            P[9 + j, i] += q
            q = -mij * swg2 * h3 / 6.0
            P[3 + i, 9 + j] += q
            P[9 + j, 3 + i] += q
            q = -mij * swg2 * h4 / 24.0
            P[6 + i, 9 + j] += q
            P[9 + j, 6 + i] += q
            q = dRm[i, j] * swa2 * h2 / 2.0
            P[3 + i, 12 + j] += q
            P[12 + j, 3 + i] += q
            q = dRm[i, j] * swa2 * h3 / 6.0
            P[6 + i, 12 + j] += q
            P[12 + j, 6 + i] += q
            P[9 + i, 9 + j] += swg2 * h * d
            P[12 + i, 12 + j] += swa2 * h * d
    # ---- bias Jacobians: J <- A9 J - B (the exact derivatives of the discrete scheme) ---------------------------
    for col in range(6):
        for i in range(3):
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            for k in range(3):
                jp = J[k, col]
                s0 += Rt[i, k] * jp
                s1 += F1[i, k] * jp
                s2 += F2[i, k] * jp
            Jn[i, col] = s0
            Jn[3 + i, col] = s1 + J[3 + i, col]
            Jn[6 + i, col] = s2 + h * J[3 + i, col] + J[6 + i, col]
    for i in range(3):
        for j in range(3):
            Jn[i, j] -= Bg0[i, j]
            Jn[3 + i, j] -= Bg1[i, j]
            Jn[6 + i, j] -= Bg2[i, j]
            Jn[3 + i, 3 + j] -= Ba1[i, j]
            Jn[6 + i, 3 + j] -= Ba2[i, j]
    for i in range(9):
        for j in range(6):
            J[i, j] = Jn[i, j]
    # ---- the mean (preint.py's order) -----------------------------------------------------------------------------
    for i in range(3):
        dp[i] = dp[i] + dv[i] * h + 0.5 * am[i] * h2
        dv[i] = dv[i] + am[i] * h
    _mm(dR, dRk, Rn)
    for i in range(3):
        for j in range(3):
            dR[i, j] = Rn[i, j]


@njit(cache=False, nogil=True)
def preint_core(t, gyr, acc, t0, t1, horizon, bg, ba, sg2, sa2, swg2, swa2, tnom, dR, dv, dp, J, C, info):
    """Preintegrate [t0, t1] from the samples with t <= horizon (module docstring: rule, causality, covariance).

    in : t (n,) non-decreasing stamps [s]; gyr, acc (n, 3) [rad/s, m/s^2] C-contiguous; bg, ba (3,) linearisation
         biases; sg2, sa2, swg2, swa2 = squared continuous densities (white gyro, white accel, gyro rw, accel rw);
         tnom = the nominal sample period T_nom [s] of the sample-support rule (required > 0 when sg2 or sa2 > 0).
    out: dR (3,3), dv (3,), dp (3,); J (9, 6) bias Jacobians, rows [phi, v, p], cols [bg, ba]
         (JRg = J[0:3, 0:3], Jvg = J[3:6, 0:3], Jva = J[3:6, 3:6], Jpg = J[6:9, 0:3], Jpa = J[6:9, 3:6]);
         C (15, 15) UNSCALED residual covariance, order [rR, rv, rp, rbg, rba]; info (N_INFO,) = [status, n_sub,
         n_inner, extrap_span, t_end, gap_max, n_released, t_nom, white_scale_max].  Nothing is valid unless
         info[I_STATUS] == OK."""
    for i in range(3):
        dv[i] = 0.0
        dp[i] = 0.0
        for j in range(3):
            dR[i, j] = 1.0 if i == j else 0.0
    for i in range(9):
        for j in range(6):
            J[i, j] = 0.0
    for i in range(15):
        for j in range(15):
            C[i, j] = 0.0
    for i in range(info.shape[0]):
        info[i] = 0.0
    n = t.shape[0]
    lo = 0
    hi = n
    while lo < hi:                            # released count: stamps only, never a value
        mid = (lo + hi) // 2
        if t[mid] <= horizon:
            lo = mid + 1
        else:
            hi = mid
    nrel = lo
    info[I_NREL] = nrel
    if not (t1 > t0) or not math.isfinite(t0) or not math.isfinite(t1):
        info[I_STATUS] = REFUSED_BAD_INTERVAL
        return
    if nrel == 0:
        info[I_STATUS] = REFUSED_NO_DATA
        return
    if t0 < t[0]:
        info[I_STATUS] = REFUSED_BEFORE_FIRST_SAMPLE
        return
    rate_ok = tnom > 0.0 and math.isfinite(tnom)
    if (sg2 > 0.0 or sa2 > 0.0) and not rate_ok:
        info[I_STATUS] = REFUSED_NO_RATE
        return
    info[I_TNOM] = tnom if rate_ok else 0.0
    smax = 1.0
    te = t[nrel - 1]
    info[I_TEND] = te
    P = np.zeros((15, 15))
    Y = np.empty((15, 15))
    Z = np.empty((15, 15))
    S = np.empty((18, 3, 3))
    Jn = np.empty((9, 6))
    w = np.empty(3)
    a = np.empty(3)
    am = np.empty(3)
    nsub = 0
    ninner = 0
    gapmax = 0.0
    if te > t0:
        tie = t1 if t1 < te else te
        lo = 0
        hi = nrel
        while lo < hi:                        # first released index with t > t0
            mid = (lo + hi) // 2
            if t[mid] <= t0:
                lo = mid + 1
            else:
                hi = mid
        k = lo
        left = k - 1                          # t[left] <= t0 < t[k]; left >= 0 since t0 >= t[0]
        e0 = t0
        while True:
            if k < nrel and t[k] < tie:
                e1 = t[k]
                samp = True
            else:
                e1 = tie
                samp = False
            h = e1 - e0
            if h > 0.0:
                m = 0.5 * (e0 + e1)
                # bracket t[left] <= m < t[left + 1] (both released). Bounded at the last released pair (fix C41c,
                # verifier an earlier design review: a one-ulp final sub-interval can round its midpoint UP to t_end, and the unbounded
                # walk then read t/gyr/acc[nrel], the first UNRELEASED sample - a B3 look-ahead / out-of-bounds read).
                # Clamped, u = (m - ta) / gap = 1 interpolates exactly the released sample at t_end.
                while left + 2 < nrel and t[left + 1] <= m:
                    left += 1
                ta = t[left]
                gap = t[left + 1] - ta
                if gap > gapmax:
                    gapmax = gap
                for c in range(3):
                    s = (gyr[left + 1, c] - gyr[left, c]) / gap
                    w[c] = s * (m - ta) + gyr[left, c] - bg[c]
                    s = (acc[left + 1, c] - acc[left, c]) / gap
                    a[c] = s * (m - ta) + acc[left, c] - ba[c]
                if not (math.isfinite(w[0]) and math.isfinite(w[1]) and math.isfinite(w[2])
                        and math.isfinite(a[0]) and math.isfinite(a[1]) and math.isfinite(a[2])):
                    info[I_STATUS] = REFUSED_NONFINITE
                    return
                sw = 1.0                      # sample-support rule (module docstring): s = max(1, nu h / T_nom)
                if rate_ok:
                    u = (m - ta) / gap
                    sw = 2.0 * ((1.0 - u) * (1.0 - u) + u * u) * h / tnom
                    if sw < 1.0:
                        sw = 1.0
                    if sw > smax:
                        smax = sw
                _step(h, w, a, sg2 * sw, sa2 * sw, swg2, swa2, dR, dv, dp, J, P, S, Y, Z, Jn, am)
                nsub += 1
            if not samp:
                break
            ninner += 1
            e0 = e1
            k += 1
    if t1 > te:                               # C40: hold the last released sample over (t_end, t1]
        ts = t0 if t0 > te else te
        h = t1 - ts
        for c in range(3):
            w[c] = gyr[nrel - 1, c] - bg[c]
            a[c] = acc[nrel - 1, c] - ba[c]
        if not (math.isfinite(w[0]) and math.isfinite(w[1]) and math.isfinite(w[2])
                and math.isfinite(a[0]) and math.isfinite(a[1]) and math.isfinite(a[2])):
            info[I_STATUS] = REFUSED_NONFINITE
            return
        sw = 1.0                              # the held sample: nu = 2 if it also fed the previous sub-interval
        if rate_ok:
            sw = (2.0 if te > t0 else 1.0) * h / tnom
            if sw < 1.0:
                sw = 1.0
            if sw > smax:
                smax = sw
        _step(h, w, a, sg2 * sw, sa2 * sw, swg2, swa2, dR, dv, dp, J, P, S, Y, Z, Jn, am)
        nsub += 1
        if te > t0:                           # t_end lies strictly inside (t0, t1): a released interior sample
            ninner += 1
        info[I_EXTRAP] = h
    # one Newton-Schulz polar step keeps dR on SO(3) (the product drifts by ~n eps)
    T = S[0]
    T2 = S[1]
    Rn = S[2]
    _mtm(dR, dR, T)
    for i in range(3):
        for j in range(3):
            T2[i, j] = (1.5 if i == j else 0.0) - 0.5 * T[i, j]
    _mm(dR, T2, Rn)
    for i in range(3):
        for j in range(3):
            dR[i, j] = Rn[i, j]
    # residual covariance: r9 = -(computed - true), r_b = +eps(t1)  ->  the cross block changes sign
    for i in range(15):
        for j in range(15):
            if (i < 9) != (j < 9):
                C[i, j] = -P[i, j]
            else:
                C[i, j] = P[i, j]
    info[I_NSUB] = nsub
    info[I_NINNER] = ninner
    info[I_GAPMAX] = gapmax
    info[I_SMAX] = smax
    info[I_STATUS] = OK


@njit(cache=False, nogil=True)
def sqrt_info_core(C, kappa, sbg2, sba2, J, W):
    """W = L^-1 (lower), L L^T = kappa C + J_bg sbg2 J_bg^T + J_ba sba2 J_ba^T (the BIASCOV term on the 9 preint
    rows).  Returns False (W undefined) if the matrix is not positive definite."""
    A = np.empty((15, 15))
    L = np.zeros((15, 15))
    for i in range(15):
        for j in range(15):
            A[i, j] = kappa * C[i, j]
    if sbg2 > 0.0 or sba2 > 0.0:
        for i in range(9):
            for j in range(9):
                s = 0.0
                for k in range(3):
                    s += sbg2 * J[i, k] * J[j, k] + sba2 * J[i, 3 + k] * J[j, 3 + k]
                A[i, j] += s
    for j in range(15):
        s = A[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        if not (s > 0.0):
            return False
        dj = math.sqrt(s)
        L[j, j] = dj
        for i in range(j + 1, 15):
            s = A[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            L[i, j] = s / dj
    for i in range(15):
        for j in range(15):
            W[i, j] = 0.0
    for j in range(15):
        W[j, j] = 1.0 / L[j, j]
        for i in range(j + 1, 15):
            s = 0.0
            for k in range(j, i):
                s += L[i, k] * W[k, j]
            W[i, j] = -s / L[i, i]
    return True


@njit(cache=False, nogil=True)
def _apply_lower(W, X):
    """X <- W X in place for lower-triangular W (bottom-up: row i reads rows k <= i, still unmodified)."""
    m = X.shape[1]
    for i in range(14, -1, -1):
        for c in range(m):
            s = 0.0
            for k in range(i + 1):
                s += W[i, k] * X[k, c]
            X[i, c] = s


@njit(cache=False, nogil=True)
def _apply_lower_vec(W, x):
    """x <- W x in place for lower-triangular W."""
    for i in range(14, -1, -1):
        s = 0.0
        for k in range(i + 1):
            s += W[i, k] * x[k]
        x[i] = s


@njit(cache=False, nogil=True)
def factor_core(Ri, pi, vi, bgi, bai, Rj, pj, vj, bgj, baj, g, dR, dv, dp, dt, J, bg_lin, ba_lin, W, whiten,
                r, Ji, Jj, Jg):
    """Residual r (15,), J_i, J_j (15, 15) and J_g (15, 3) of the native factor (module docstring); whitened by
    the lower-triangular W when `whiten`.  Outputs are preallocated and fully overwritten."""
    S = np.empty((10, 3, 3))
    V = np.empty((8, 3))
    dbg = V[0]
    dba = V[1]
    phib = V[2]
    rR = V[3]
    uv = V[4]
    up = V[5]
    Dv = V[6]
    Dp = V[7]
    Eb = S[0]
    Jrb = S[1]
    dRc = S[2]
    Mij = S[3]
    E = S[4]
    Jri = S[5]
    T1 = S[6]
    T2 = S[7]
    T3 = S[8]
    T4 = S[9]
    for c in range(3):
        dbg[c] = bgi[c] - bg_lin[c]
        dba[c] = bai[c] - ba_lin[c]
    for i in range(3):
        phib[i] = J[i, 0] * dbg[0] + J[i, 1] * dbg[1] + J[i, 2] * dbg[2]
    _exp_jr(phib[0], phib[1], phib[2], Eb, Jrb)
    _mm(dR, Eb, dRc)                      # first-order bias-corrected dR
    _mtm(Ri, Rj, Mij)                     # R_i^T R_j
    _mtm(dRc, Mij, E)
    _log(E, rR)
    dt2 = dt * dt
    for c in range(3):
        Dv[c] = vj[c] - vi[c] - g[c] * dt
        Dp[c] = pj[c] - pi[c] - vi[c] * dt - 0.5 * g[c] * dt2
    _mtv(Ri, Dv, uv)
    _mtv(Ri, Dp, up)
    for i in range(3):
        cv = dv[i]
        cp = dp[i]
        for k in range(3):
            cv += J[3 + i, k] * dbg[k] + J[3 + i, 3 + k] * dba[k]
            cp += J[6 + i, k] * dbg[k] + J[6 + i, 3 + k] * dba[k]
        r[i] = rR[i]
        r[3 + i] = uv[i] - cv
        r[6 + i] = up[i] - cp
        r[9 + i] = bgj[i] - bgi[i]
        r[12 + i] = baj[i] - bai[i]
    _jr_inv(rR, Jri)
    _mmt(Jri, Mij, T1)                    # Jr^-1(rR) R_j^T R_i
    _mmt(Jri, E, T2)                      # Jr^-1(rR) E^T
    _mm(T2, Jrb, T3)                      # ... Jr(JRg dbg)
    for i in range(3):                    # ... JRg
        for j in range(3):
            T4[i, j] = T3[i, 0] * J[0, j] + T3[i, 1] * J[1, j] + T3[i, 2] * J[2, j]
    for i in range(15):
        for j in range(15):
            Ji[i, j] = 0.0
            Jj[i, j] = 0.0
        for j in range(3):
            Jg[i, j] = 0.0
    for i in range(3):
        for j in range(3):
            RiT = Ri[j, i]
            Ji[i, j] = -T1[i, j]                       # rR / dphi_i
            Ji[i, 9 + j] = -T4[i, j]                   # rR / dbg_i
            Jj[i, j] = Jri[i, j]                       # rR / dphi_j
            Ji[3 + i, 6 + j] = -RiT                    # rv / dv_i
            Ji[3 + i, 9 + j] = -J[3 + i, j]            # rv / dbg_i
            Ji[3 + i, 12 + j] = -J[3 + i, 3 + j]       # rv / dba_i
            Ji[6 + i, 3 + j] = -RiT                    # rp / dp_i
            Ji[6 + i, 6 + j] = -RiT * dt               # rp / dv_i
            Ji[6 + i, 9 + j] = -J[6 + i, j]            # rp / dbg_i
            Ji[6 + i, 12 + j] = -J[6 + i, 3 + j]       # rp / dba_i
            Jj[3 + i, 6 + j] = RiT                     # rv / dv_j
            Jj[6 + i, 3 + j] = RiT                     # rp / dp_j
            Jg[3 + i, j] = -RiT * dt
            Jg[6 + i, j] = -0.5 * RiT * dt2
    # rv / dphi_i = [R_i^T Dv]x,  rp / dphi_i = [R_i^T Dp]x
    Ji[3, 1] = -uv[2]
    Ji[3, 2] = uv[1]
    Ji[4, 0] = uv[2]
    Ji[4, 2] = -uv[0]
    Ji[5, 0] = -uv[1]
    Ji[5, 1] = uv[0]
    Ji[6, 1] = -up[2]
    Ji[6, 2] = up[1]
    Ji[7, 0] = up[2]
    Ji[7, 2] = -up[0]
    Ji[8, 0] = -up[1]
    Ji[8, 1] = up[0]
    for i in range(3):
        Ji[9 + i, 9 + i] = -1.0
        Ji[12 + i, 12 + i] = -1.0
        Jj[9 + i, 9 + i] = 1.0
        Jj[12 + i, 12 + i] = 1.0
    if whiten:
        _apply_lower_vec(W, r)
        _apply_lower(W, Ji)
        _apply_lower(W, Jj)
        _apply_lower(W, Jg)


# ==================================================================================================== Python API ==
State = namedtuple("State", "R p v bg ba")
State.__doc__ = "Native state: R = R_wb (3,3), p = p_wb, v = v_w, bg, ba (3,).  Error state [dphi, dp, dv, dbg, dba]."
Prediction = namedtuple("Prediction", "state P pre")
Prediction.__doc__ = "propagate(): predicted State at t_to (None if refused), its 15x15 covariance (or None), the record."


def _f64(x, shape=None):
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if shape is not None and a.shape != shape:
        raise ValueError("expected shape %s, got %s" % (shape, a.shape))
    return a


def as_state(x):
    """A State from a State, or any (R, p, v, bg, ba) sequence."""
    R, p, v, bg, ba = x
    return State(_f64(R, (3, 3)), _f64(p, (3,)), _f64(v, (3,)), _f64(bg, (3,)), _f64(ba, (3,)))


class ImuNoise:
    """The declared continuous-time IMU densities (Class A).  Kalibr/EuRoC units:
    sigma_g rad/s/sqrt(Hz), sigma_a m/s^2/sqrt(Hz), rw_g rad/s^2/sqrt(Hz), rw_a m/s^3/sqrt(Hz).
    A missing or non-positive value is REFUSED: a declared sigma is a calibrated system weight, and substituting one
    is exactly the class of error that once weighted the IMU 40,000x too weakly.
    rate_hz: the sample rate of the discrete noise model (T_nom = 1 / rate_hz, the sample-support rule); Class A
    (calib imu_rate_hz) when declared, else None and preintegrate() measures the median released spacing (Class C)."""
    __slots__ = ("sigma_g", "sigma_a", "rw_g", "rw_a", "rate_hz")

    def __init__(self, sigma_g, sigma_a, rw_g, rw_a, rate_hz=None):
        vals = [float(sigma_g), float(sigma_a), float(rw_g), float(rw_a)]
        for name, v in zip(self.__slots__, vals):
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError("ImuNoise.%s = %r: a declared density must be finite and > 0 (Class A input; "
                                 "never substituted)" % (name, v))
        self.sigma_g, self.sigma_a, self.rw_g, self.rw_a = vals
        if rate_hz is not None:
            rate_hz = float(rate_hz)
            if not (math.isfinite(rate_hz) and rate_hz > 0.0):
                raise ValueError("ImuNoise.rate_hz = %r: must be finite and > 0, or None (measured)" % rate_hz)
        self.rate_hz = rate_hz

    @classmethod
    def from_calib(cls, calib, t=None):
        """The four densities from calib `imu_noise` (Class A, all required); the rate from calib `imu_rate_hz`
        (Class A) or, if absent and stamps `t` are given, 1 / their median spacing (Class C, backend/imu.py's rule);
        else None (measured per call from the released stamps)."""
        n = (calib or {}).get("imu_noise") or {}
        rate = float((calib or {}).get("imu_rate_hz") or 0.0)
        if not rate > 0.0 and t is not None:
            tt = np.asarray(t, dtype=np.float64)
            if tt.shape[0] >= 2:
                med = float(np.median(np.diff(tt)))
                rate = 1.0 / med if med > 0.0 else 0.0
        try:
            return cls(n["gyroscope_noise_density"], n["accelerometer_noise_density"],
                       n["gyroscope_random_walk"], n["accelerometer_random_walk"],
                       rate if rate > 0.0 else None)
        except KeyError as e:
            raise ValueError("calib imu_noise lacks %s: the IMU factor needs all four declared densities" % e)

    def squares(self):
        return self.sigma_g ** 2, self.sigma_a ** 2, self.rw_g ** 2, self.rw_a ** 2

    def __repr__(self):
        return "ImuNoise(sigma_g=%g, sigma_a=%g, rw_g=%g, rw_a=%g, rate_hz=%s)" % (
            self.sigma_g, self.sigma_a, self.rw_g, self.rw_a, self.rate_hz)


def _as_noise(noise):
    if noise is None or isinstance(noise, ImuNoise):
        return noise
    if isinstance(noise, dict):
        return ImuNoise.from_calib({"imu_noise": noise})
    return ImuNoise(*noise)


def _t_nom(noise, t, horizon):
    """T_nom: 1 / the declared rate (Class A), else the median spacing of the RELEASED stamps (Class C; only stamps
    with t <= horizon are looked at, so the result is causal); 0 when neither exists (a covariance is then refused)."""
    if noise is None:
        return 0.0
    if noise.rate_hz is not None:
        return 1.0 / noise.rate_hz
    nrel = int(np.searchsorted(t, horizon, side="right"))
    if nrel < 2:
        return 0.0
    med = float(np.median(np.diff(t[:nrel])))
    return med if (math.isfinite(med) and med > 0.0) else 0.0


class Preint:
    """One preintegrated interval [t0, t1], in the body frame at t0.  `ok` is False for a refused request (see
    `reason`); then no other field is meaningful."""
    __slots__ = ("t0", "t1", "dt", "dR", "dv", "dp", "J", "bg_lin", "ba_lin", "cov", "status", "n_sub", "n_inner",
                 "extrap_span", "t_data_end", "gap_max", "n_released", "t_nom", "white_scale_max")

    @property
    def ok(self):
        return self.status == OK

    @property
    def reason(self):
        return STATUS_TEXT.get(self.status, "status %d" % self.status)

    @property
    def extrapolated(self):
        """C40: the interval runs past the last released sample, whose value was held."""
        return self.extrap_span > 0.0

    @property
    def single_sample(self):
        """The SAMPLES case: no released sample strictly inside (t0, t1) (covariance conservative there).  The last
        released sample counts when it lies strictly inside, also when the interval then runs into a hold."""
        return self.n_inner == 0

    JRg = property(lambda s: s.J[0:3, 0:3])
    Jvg = property(lambda s: s.J[3:6, 0:3])
    Jva = property(lambda s: s.J[3:6, 3:6])
    Jpg = property(lambda s: s.J[6:9, 0:3])
    Jpa = property(lambda s: s.J[6:9, 3:6])
    cov9 = property(lambda s: s.cov[0:9, 0:9])

    def __repr__(self):
        return ("Preint(%s, [%.6f, %.6f], dt=%.6f, n_sub=%d, n_inner=%d, extrap=%.6f, gap_max=%.6f, "
                "white_scale_max=%.3f)"
                % (self.reason, self.t0, self.t1, self.dt, self.n_sub, self.n_inner, self.extrap_span,
                   self.gap_max, self.white_scale_max))


def preintegrate(t, gyro, acc, t0, t1, horizon, bg_lin, ba_lin, noise):
    """Preintegrate the IMU over [t0, t1] using only samples with t <= horizon (None = +inf, OFFLINE only).

    t (n,) non-decreasing [s]; gyro, acc (n, 3); bg_lin, ba_lin the linearisation biases; noise an ImuNoise (or the
    calib `imu_noise` dict; None = mean only, zero covariance).  Returns a Preint (module docstring for the rule,
    the hold, the covariance, the sample-support rule and the SAMPLES policy)."""
    t = _f64(t)
    gyro = _f64(gyro)
    acc = _f64(acc)
    if t.ndim != 1 or gyro.shape != (t.shape[0], 3) or acc.shape != (t.shape[0], 3):
        raise ValueError("t (n,), gyro (n,3), acc (n,3) expected; got %s %s %s" % (t.shape, gyro.shape, acc.shape))
    noise = _as_noise(noise)
    sg2, sa2, swg2, swa2 = (0.0, 0.0, 0.0, 0.0) if noise is None else noise.squares()
    bg = _f64(bg_lin, (3,))
    ba = _f64(ba_lin, (3,))
    hz = np.inf if horizon is None else float(horizon)
    tnom = _t_nom(noise, t, hz)
    pre = Preint()
    pre.dR = np.empty((3, 3))
    pre.dv = np.empty(3)
    pre.dp = np.empty(3)
    pre.J = np.empty((9, 6))
    pre.cov = np.empty((15, 15))
    info = np.empty(N_INFO)
    preint_core(t, gyro, acc, float(t0), float(t1), hz, bg, ba, sg2, sa2, swg2, swa2, tnom,
                pre.dR, pre.dv, pre.dp, pre.J, pre.cov, info)
    pre.t0 = float(t0)
    pre.t1 = float(t1)
    pre.dt = float(t1) - float(t0)
    pre.bg_lin = bg.copy()
    pre.ba_lin = ba.copy()
    pre.status = int(info[I_STATUS])
    pre.n_sub = int(info[I_NSUB])
    pre.n_inner = int(info[I_NINNER])
    pre.extrap_span = float(info[I_EXTRAP])
    pre.t_data_end = float(info[I_TEND])
    pre.gap_max = float(info[I_GAPMAX])
    pre.n_released = int(info[I_NREL])
    pre.t_nom = float(info[I_TNOM])
    pre.white_scale_max = float(info[I_SMAX])
    return pre


def sqrt_information(pre, kappa=1.0, bias_sigma_g=0.0, bias_sigma_a=0.0):
    """W (15 x 15, lower triangular) with W^T W = (kappa C + BIASCOV)^-1 (module docstring, WHITENING)."""
    if not pre.ok:
        raise ValueError("refused preintegration (%s) has no information" % pre.reason)
    kappa = float(kappa)
    if not (math.isfinite(kappa) and kappa > 0.0):
        raise ValueError("kappa_imu must be finite and > 0, got %r" % kappa)
    W = np.empty((15, 15))
    ok = sqrt_info_core(pre.cov, kappa, float(bias_sigma_g) ** 2, float(bias_sigma_a) ** 2, pre.J, W)
    if not ok:
        raise ValueError("IMU covariance not positive definite (%r)" % pre)
    return W


def _call_factor(state_i, state_j, pre, g_w, W, whiten):
    if not pre.ok:
        raise ValueError("refused preintegration (%s) is not a factor" % pre.reason)
    si = as_state(state_i)
    sj = as_state(state_j)
    r = np.empty(15)
    Ji = np.empty((15, 15))
    Jj = np.empty((15, 15))
    Jg = np.empty((15, 3))
    factor_core(si.R, si.p, si.v, si.bg, si.ba, sj.R, sj.p, sj.v, sj.bg, sj.ba, _f64(g_w, (3,)),
                pre.dR, pre.dv, pre.dp, pre.dt, pre.J, pre.bg_lin, pre.ba_lin,
                np.eye(15) if W is None else W, bool(whiten), r, Ji, Jj, Jg)
    return r, Ji, Jj, Jg


def imu_residual(state_i, state_j, pre, g_w):
    """The raw (unwhitened) 15-row residual."""
    return _call_factor(state_i, state_j, pre, g_w, None, False)[0]


def imu_residual_jacobians(state_i, state_j, pre, g_w, kappa=1.0, bias_sigma_g=0.0, bias_sigma_a=0.0,
                           whiten=True, W=None, with_gravity=False):
    """r (15: rR, rv, rp, rbg, rba), J_i (15x15), J_j (15x15) [, J_g (15x3)] of the native factor, whitened by
    W = sqrt_information(pre, kappa, bias_sigma_g, bias_sigma_a) unless `whiten` is False (pass W to reuse one).
    The robust weight is NOT applied: robust_weight(r) returns it."""
    if whiten and W is None:
        W = sqrt_information(pre, kappa, bias_sigma_g, bias_sigma_a)
    r, Ji, Jj, Jg = _call_factor(state_i, state_j, pre, g_w, W, whiten)
    return (r, Ji, Jj, Jg) if with_gravity else (r, Ji, Jj)


def huber_k_dof(dof):
    """The Huber threshold on the whitened NORM of a dof-dimensional residual at the same tail probability as
    HUBER_K for one standard normal: sqrt(chi2_dof^-1(P(|z| <= HUBER_K))).  Class D, derived (= HUBER_K at dof 1)."""
    from scipy.special import erf, gammaincinv
    q = float(erf(HUBER_K / math.sqrt(2.0)))
    return float(math.sqrt(2.0 * gammaincinv(0.5 * float(dof), q)))


def robust_weight(r_w, rows=15, rule="norm"):
    """(w, e): the Huber IRLS weight of a WHITENED factor residual and its norm e = ||r_w[:rows]||.
    rule "norm": threshold HUBER_K (CONVENTIONS.md); "dof": huber_k_dof(rows).  Scale r and J by sqrt(w)."""
    e = float(np.linalg.norm(np.asarray(r_w, dtype=np.float64)[:rows]))
    if rule == "norm":
        k = HUBER_K
    elif rule == "dof":
        k = huber_k_dof(rows)
    else:
        raise ValueError("rule must be 'norm' or 'dof'")
    return (1.0 if e <= k else k / e), e


def propagate(state, t, gyro, acc, t_from, t_to, horizon, g_w, noise=None, P=None, kappa=1.0):
    """The IMU-predicted state at t_to from `state` at t_from, causal (samples t <= horizon; the last released one
    is held beyond it, C40), preintegrated at the state's own biases.  With P (15x15, native error-state order) and
    `noise`, also the predicted covariance  P_j = X P X^T + Jj^-1 (kappa C) Jj^-T,  X = -Jj^-1 Ji, from the factor's
    own Jacobians at (state, prediction).  Returns Prediction(state, P, pre); state is None if the record is refused."""
    si = as_state(state)
    pre = preintegrate(t, gyro, acc, t_from, t_to, horizon, si.bg, si.ba, noise)
    if not pre.ok:
        return Prediction(None, None, pre)
    g = _f64(g_w, (3,))
    T = pre.dt
    Rj = si.R @ pre.dR
    vj = si.v + g * T + si.R @ pre.dv
    pj = si.p + si.v * T + 0.5 * g * T * T + si.R @ pre.dp
    sj = State(Rj, pj, vj, si.bg.copy(), si.ba.copy())
    Pj = None
    if P is not None:
        if noise is None:
            raise ValueError("a covariance prediction needs the declared noise")
        _r, Ji, Jj, _g = _call_factor(si, sj, pre, g, None, False)
        Jinv = np.linalg.inv(Jj)
        X = -Jinv @ Ji
        Pj = X @ _f64(P, (15, 15)) @ X.T + Jinv @ (float(kappa) * pre.cov) @ Jinv.T
        Pj = 0.5 * (Pj + Pj.T)
    return Prediction(sj, Pj, pre)


# ---- state algebra (native) -------------------------------------------------------------------------------------
def _so3_exp(v):
    R = np.empty((3, 3))
    Jr = np.empty((3, 3))
    _exp_jr(float(v[0]), float(v[1]), float(v[2]), R, Jr)
    return R


def _so3_log(R):
    out = np.empty(3)
    _log(_f64(R, (3, 3)), out)
    return out


def box_plus(state, dx):
    """X [+] dx in the native chart: R Exp(dphi), p + dp, v + dv, bg + dbg, ba + dba."""
    s = as_state(state)
    dx = _f64(dx, (15,))
    return State(s.R @ _so3_exp(dx[0:3]), s.p + dx[3:6], s.v + dx[6:9], s.bg + dx[9:12], s.ba + dx[12:15])


def box_minus(state, state0):
    """X [-] X0 = [Log(R0^T R), p - p0, v - v0, bg - bg0, ba - ba0] (CONVENTIONS.md box-minus)."""
    s = as_state(state)
    s0 = as_state(state0)
    return np.concatenate([_so3_log(s0.R.T @ s.R), s.p - s0.p, s.v - s0.v, s.bg - s0.bg, s.ba - s0.ba])


# ---- the chart adapters ------------------------------------------------------------------------------------------
def _hat(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def chart_tcw_left(R_wb, T_bc):
    """Mc (6x6): native [dphi, dp] = Mc @ [dt, dw] for T_cw <- [Exp(dw) | dt] T_cw (module docstring)."""
    R_wb = _f64(R_wb, (3, 3))
    T_bc = _f64(T_bc, (4, 4))
    R_bc = T_bc[:3, :3]
    t_bc = T_bc[:3, 3]
    Mc = np.zeros((6, 6))
    Mc[0:3, 3:6] = -R_bc
    Mc[3:6, 0:3] = -R_wb @ R_bc
    Mc[3:6, 3:6] = -R_wb @ _hat(t_bc) @ R_bc
    return Mc


def chart_twc_left(R_wb, p_wb):
    """Mc (6x6): native [dphi, dp] = Mc @ [dt, dw] for T_wc <- [Exp(dw) | dt] T_wc (the bench's storage)."""
    R_wb = _f64(R_wb, (3, 3))
    Mc = np.zeros((6, 6))
    Mc[0:3, 3:6] = R_wb.T
    Mc[3:6, 0:3] = np.eye(3)
    Mc[3:6, 3:6] = -_hat(_f64(p_wb, (3,)))
    return Mc


def jacobian_to_chart(J, Mc):
    """Replace the 6 native pose columns of J (rows x 15) by the chart's [dt, dw] columns; v, bg, ba unchanged."""
    J = np.asarray(J, dtype=np.float64)
    out = J.copy()
    out[:, 0:6] = J[:, 0:6] @ Mc
    return out


def retract_left(T, dx):
    """vi_window_bench.py update_poses: R <- Exp(dw) R, t <- Exp(dw) t + dt, dx = [dt(0:3), dw(3:6)]."""
    T = _f64(T, (4, 4))
    dx = _f64(dx, (6,))
    E = _so3_exp(dx[3:6])
    out = np.eye(4)
    out[:3, :3] = E @ T[:3, :3]
    out[:3, 3] = E @ T[:3, 3] + dx[0:3]
    return out


def _inv_T(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def state_from_tcw(T_cw, v, bg, ba, T_bc):
    """Native State from a T_cw pose: T_wb = T_cw^-1 T_bc^-1."""
    T_wb = _inv_T(_f64(T_cw, (4, 4))) @ _inv_T(_f64(T_bc, (4, 4)))
    return as_state((T_wb[:3, :3], T_wb[:3, 3], v, bg, ba))


def tcw_from_state(state, T_bc):
    s = as_state(state)
    T_wb = np.eye(4)
    T_wb[:3, :3] = s.R
    T_wb[:3, 3] = s.p
    return _inv_T(T_wb @ _f64(T_bc, (4, 4)))


def state_from_twc(T_wc, v, bg, ba, T_bc):
    """Native State from a T_wc pose: T_wb = T_wc T_bc^-1."""
    T_wb = _f64(T_wc, (4, 4)) @ _inv_T(_f64(T_bc, (4, 4)))
    return as_state((T_wb[:3, :3], T_wb[:3, 3], v, bg, ba))


def twc_from_state(state, T_bc):
    s = as_state(state)
    T_wb = np.eye(4)
    T_wb[:3, :3] = s.R
    T_wb[:3, 3] = s.p
    return T_wb @ _f64(T_bc, (4, 4))


def warmup():
    """Compile every kernel (call before the frame clock starts: R1/R2 import audit, no JIT in the frame path)."""
    t = np.arange(8, dtype=np.float64) * 0.005
    g = np.ascontiguousarray(np.tile([0.1, -0.2, 0.3], (8, 1)), dtype=np.float64)
    a = np.ascontiguousarray(np.tile([0.2, 0.1, 9.8], (8, 1)), dtype=np.float64)
    nz = ImuNoise(1e-4, 1e-3, 1e-5, 1e-4)
    pre = preintegrate(t, g, a, 0.0013, 0.0262, 0.025, np.zeros(3), np.zeros(3), nz)
    X = State(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    imu_residual_jacobians(X, X, pre, np.array([0.0, 0.0, -9.81]), with_gravity=True)
    imu_residual(X, X, pre, np.array([0.0, 0.0, -9.81]))
    box_minus(X, X)
    return True
