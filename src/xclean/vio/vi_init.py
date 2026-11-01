#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1 VI INITIALISATION AND IMU ADMISSION: pure functions the engine calls causally at each keyframe until the IMU is
admitted, plus the IMU-trust and watchdog statistics used afterwards.  numpy / scipy, float64, CPU only.

    from xclean.vio import vi_init as V
    cand = V.init_candidate(T_wb, kf_t, t, gyro, acc, horizons, noise, bg0, bg0_cov=None, prev=prev_init)
    if cand.admission.decision == "admit": ...   # cand.init: g_w, v_w, ba, bg (at the newest KF), covariances
    prev_init = cand.init                         # the next KF's candidate is tested against this one

    the parts (each usable alone):
    ga  = V.gyro_bias_align(R_wb, pres, bg_prior=None, bg_prior_cov=None, noise=noise)          # deliverable 1
    ini = V.stereo_vi_init(T_wb, pres_at_bg, g_mag=V.G_STD, bg_prior=ga.bg, bg_prior_cov=ga.cov,
                           pose_sigma=(None, ga.sigma_th), noise=noise, gyro=ga)                    # deliverable 2
    kr  = V.igt_k_R(R_wb, pres_at_bg0, bg0)                                        # the IGT statistic (causal variant)
    adm = V.admission(n_kf, kr, ga, ini, prev=prev_init, noise=noise)              # deliverable 3
    kap = V.kappa_imu(chi2_per_factor, dof_per_factor)                             # deliverable 4
    bb  = V.bias_bound(noise, t_since_init, cov_bg0, cov_ba0, kappa)               # deliverable 5
    wd  = V.watchdog_stat(summary, state)                                          # deliverable 5

Design place: an internal note section 6.2 (VI init + admission), 6.3 (kappa_imu), 6.4
(watchdog), 4.2 (bias plausibility), 13 (constant classes); the causal rule and two later rules of our design notes.
Conventions: xclean/vio/CONVENTIONS.md (T_wb, right rotation perturbation, p additive in w, error state
[dphi, dp, dv, dbg, dba]; g_w is a SEPARATE variable; w is NOT rotated at VI init).  IMU factor: xclean/vio/imu_factor.py
(read, not edited).  Design references (cited, not copied): Forster et al., T-RO 2017 (preintegration, first-order bias
correction); Qin & Shen, "Robust initialization of monocular visual-inertial estimation", IROS 2017, and Campos et al.,
"Inertial-only optimization for visual-inertial initialization", ICRA 2020 (the init as a least-squares problem in
v, g, b given the visual poses; here the stereo poses are metric, so there is no scale unknown); Foerstner, AVN 1979,
Rao 1971 / Koch 1986 (variance-component estimation, iterated MINQUE = REML Fisher scoring); Satterthwaite,
Biometrics 1946 (effective degrees of freedom of an estimated variance); Baarda 1968 (data snooping).  Our own code;
nothing is taken from OKVIS / Basalt / ORB-SLAM3 / VINS source.

CAUSALITY (C40; structural)
    Every function receives ONLY the keyframes created so far (k = 0..n) and the preintegration records between
    consecutive ones; nothing here reads an IMU array except `preintegrate_intervals` / `init_candidate`, which pass
    each interval's own released horizon to imu_factor.preintegrate (no sample with t > horizon is read; beyond the
    last released sample the last value is held).  `check_causal` asserts the structure (contiguous records, record k
    spans [t_k, t_k+1], every record's data end <= its horizon <= t_n).

THE PROBLEM (deliverable 2).  KF poses T_wb_k = (R_k, p_k), k = 0..n, from the VO window (gravity-free odom frame w,
metric because stereo).  Per interval k (i = k, j = k+1, dt), with the imu_factor residuals (prediction - measurement):
    e_v = R_i^T (v_j - v_i - g dt)              - (dv + Jvg (bg - bg_lin) + Jva (ba - ba_lin))
    e_p = R_i^T (p_j - p_i - v_i dt - g dt^2/2) - (dp + Jpg (bg - bg_lin) + Jpa (ba - ba_lin))
  With the poses known these are LINEAR in x = (v_0..v_n, g, ba, bg).  bg enters through the first-order correction and
  carries the gyro alignment's estimate as a prior.  Weighted least squares; then |g| = g_mag is imposed by a 2-DoF
  GLOBAL minimisation: at a fixed noise model the chi2 profiled over (v, ba, bg) is an exact quadratic in g, so the
  constrained problem is the trust-region subproblem on the sphere (_profile_g, _sphere_modes): every local minimum is
  found from the secular equation and the global one is returned, whatever the start (the earlier tangent-space
  Gauss-Newton from the free-LS direction returned whichever of two equal minima that direction picked).  The tilt
  coordinates d (radians) are g = Exp(B d) g*, B (3x2) orthonormal and perpendicular to g*.
  THE NOISE MODEL (errors in variables).  The residual is the IMU noise PLUS the VO pose noise, which dominates the
  position rows at short KF spacing (U7: position block 90-840x the declared sigma between fixed states).  So the
  residual covariance is modelled as
      C = kappa Q_imu + s_p Q_p + s_th Q_th
      Q_imu = blockdiag(Preint.cov[3:9, 3:9]) + the bias RANDOM WALK across intervals (Class A, pinned at the newest
              KF t_n: the returned ba, bg are the values AT t_n, which is where the window continues; without the
              walk, EuRoC's declared rw_a made the tilt NEES 3x over-confident, test a2)
      Q_p   = J_p J_p^T   (KF position jitter, white per KF, isotropic in w; consecutive intervals share a KF, so C is
                           block-tridiagonal)
      Q_th  = blockdiag(J_th J_th^T)   (KF attitude jitter, right perturbation; e_v / dphi_i = [R_i^T Dv]x,
                                        e_p / dphi_i = [R_i^T Dp]x, the factor's own Jacobians, at the current x)
  and the three scales are Class-C run-time statistics estimated by REML Fisher scoring (iterated MINQUE: F s = q,
  F_gh = tr(P Q_g P Q_h), q_g = e^T W Q_g W e).  kappa is held at its floor 1 when the step would go below (the
  kappa rule below); s_p, s_th >= 0 (active set).  The attitude jitter is identified far better by the rotation
  residuals, so init_candidate takes s_th from gyro_bias_align and holds it; an engine that knows its pose jitter
  passes pose_sigma = (sigma_p [m], sigma_th [rad]) (None entries are estimated).
  CALIBRATION: the plug-in covariance N^-1 at estimated components is optimistic when a component is weakly
  determined (measured: kappa 4, tilt / bg NEES 1.46 / 1.55 per dof); each block (tilt, ba, bg, v) is scaled by the
  Satterthwaite factor nu / (nu - 2), nu = 2 tr(Phi_B)^2 / Var(tr Phi_B); nu <= 2 = the noise scale is undetermined
  (tilt not certified, cause "scale").
  Outlier intervals: Baarda data snooping, the largest standardised interval residual against chi2_6 at the
  Bonferroni level (1 - P_GATE) / m; removed one at a time while it fails, at most half of the intervals (the 50 %
  breakdown point of the median, Class D); an unconverged fit is not snooped.

OBSERVABILITY (EXCITATION) CONDITION, stated.  With the poses metric, v is determined by the positions; g and ba
    enter the velocity rows only as R_k^T g dt - ba dt.  Their combination is confounded exactly when a direction
    dg perpendicular to g (|g| is fixed) satisfies R_k^T dg = const over the window, i.e. when dg lies along the
    WORLD-frame axis of every relative rotation R_k R_0^T.  Hence:
      * no rotation in the window (e.g. constant velocity, constant attitude)  -> the 2-D tilt is unobservable
        (the classical tilt / accelerometer-bias ambiguity, tilt error = ba_horizontal / g);
      * rotation about ONE horizontal axis only                                 -> 1-D degenerate (tilt about it);
      * rotation about the vertical ONLY (yaw: a ground vehicle on a flat road)  -> the tilt AXIS is observable
        (yaw spins every horizontal dg) but the gravity SIGN is not: with every R_k^T u equal to one body vector u_b
        (u the vertical), (g + s u, ba + s u_b) is an EXACT null direction of the free problem, and |g| = g_mag
        leaves two equal minima, g and -g with ba_vertical + 2 g_mag (verifier 2026-10-24; the version before
        stated "observable, the vertical ba fixed by |g|", which is false).  The DATA separate them only through
        roll / pitch excitation; on real rows a model misfit can prefer the mirror decisively (4s_20200612_101 x1,
        KF 37: dchi2 18.5 for -g with |ba| = 19.6 m/s^2, certified upside down by the global solution before the
        bound below).  What separates them is physics: THE ACCELEROMETER-BIAS BOUND |ba| <= G_STD (Class D below)
        excludes the mirror, whose bias is >= 2 G_STD - |ba|; a declared ba prior (ba_prior) would too;
      * strength ~ the size of the relative rotations: `excitation` = sqrt(lambda_min(B^T M B) / m),
        M = sum_k (R_k - R_0)(R_k - R_0)^T, a dimensionless chord length (0 = degenerate, 2 = maximal).
    The DEGENERACY FLAG = "the tilt is NOT certified": numerically rank-deficient (the Jacobi-scaled Schur information
    of (tilt, ba) or the normal matrix: lambda_min <= dim * eps * lambda_max, Class D, float64; scale-invariant), OR
    the certified P_GATE (0.999) radius of the newest KF's roll/pitch exceeds TILT_BAR_DEG (the live benchmark rules B4 NEXT: "never a
    wrong init, roll/pitch vs GT <= 2 deg"), OR the noise model is not valid (the REML iteration did not converge, or
    the data reject |g| = g_mag at P_GATE: a covariance under a rejected or unconverged model certifies nothing).
    THE CERTIFIED RADIUS of g_w = max(the Laplace radius sqrt(chi2_2(P_GATE) lambda_max(cov_tilt)), the radius of the
    EXACT posterior of the gravity direction on the sphere, _gravity_posterior: p(u) ~ exp(-chi2(G u) / (2 fs)), which
    sees a second minimum (yaw only: the sign; one horizontal axis: the two crossings of the weak line with the
    sphere) and a non-quadratic weak direction that the local radius cannot - verifier 2026-10-24: a roll-only window
    admitted at 3.1 deg with a 1.9 deg Laplace radius, exact radius 7.5 deg); the newest KF's attitude variance is
    added in quadrature.  The posterior's support is the part of the sphere where |ba(g)| <= G_STD (the bias bound).
    Causes: rank | bias (no local minimum inside the bias bound) | unconverged | model | scale (Satterthwaite nu <= 2) |
    ambiguous (a second
    local minimum carries > 1 - P_GATE) | weak (the Laplace radius passed, the exact one did not) | attitude (the
    newest KF's VO attitude alone fails the bar) | excitation (would be certified were ba known) | noise; vif = tilt
    variance with ba free / with ba known is reported.

THE IGT STATISTIC (admission, deliverable 3) is backend/preint.py rotation_consistency itself, called on adapters of
    the imu_factor records: per interval the rotation block of the residual, whitened by the declared rotation
    covariance x IGT_INFLATE^2, at bg0 and after a 3-parameter Gauss-Newton refit; k_R = sqrt(median chi / 2.366),
    min of the two; intervals with fewer than two integration steps are not factors.
    CAUSAL variant only: bg0 is the causal bootstrap bias held at t_n (U6 'boot'), the poses are the causal KF poses,
    the refit uses the window's own KFs, and the records hold the last released sample past the horizon (C40) where
    U6's backend preintegration interpolated one IMU period ahead.

CONSTANTS (no other number is set by hand in this file)
    G_STD = 9.80665 m/s^2          D   standard gravity (CGPM 1901).  A Class-C local value would need |g_free| to be
                                       measured more precisely than local gravity varies (0.3 %); the free-|g|
                                       sigma at init is 1-10 %, so it is not justified
    P_GATE = 0.999                 D   the design's probability level (DESIGN_P1 13)
    IGT_K = 10                     D   carried: backend/preint.py INERT_GATE_K
    IGT_INFLATE = 3.0              D   carried: part of the IGT definition (the tracker's inertial_inflate, at which
                                       K was chosen); changing it changes the statistic
    TILT_BAR_DEG = 2.0             -   external bar, the live benchmark rules B4 NEXT (not set here)
    MIN_INIT_KF = 8                D   carried: DESIGN_P1 6.2 "evaluated over the init window's KFs (>= 8)"
    N_CONSEC = 2                   D   carried: DESIGN_P1 6.4 / 13 watchdog "2 consecutive KF intervals"
    CHI2 quantiles                 D   from P_GATE and the dimension (Bonferroni (1 - P_GATE) / m for snooping)
    |ba| <= G_STD (the bias bound) D   an accelerometer's bias is below the 1 g it is calibrated against (every
                                       MEMS to navigation-grade spec is in mg; the saved biases of every dev rig are
                                       0.07-0.63 m/s^2).  It removes the yaw-only mirror -g (|ba| >= 2 G_STD - |ba|) and
                                       nothing a functioning sensor can produce; its exact value is immaterial anywhere
                                       between the largest real bias and 2 G_STD minus it.  A local minimum outside it is
                                       never returned while a feasible one exists; none feasible = cause 'bias'
    _POST_* , _LOG_TINY            D   numerical quadrature of the exact gravity posterior (each stated where defined:
                                       grid steps 0.1 deg / 0.05 Laplace sd / 3.5 % geometric, +-12 sd windows, the
                                       float64 underflow exponent 745); brute-force agreement 0.01-0.15 deg (test a15)
    50 % snooping limit            D   the breakdown point of the median
    numerical tolerances           D   derived from float64 eps (rank test, GN convergence, component floor); the
                                       REML stop _VCE_FRAC = 1 % of each component's own standard error
    MAX_GN = 20, MAX_VCE = 30      D   iteration caps (numerical; Gauss-Newton converges in 2-5 steps, the REML
                                       scoring in 5-15 on every well-posed window measured; a fit that hits a cap is
                                       reported `converged = False` (admission defers), never silently accepted)
    TD_HARM = 0.0677 s             E   EXTERNAL EFFECT SIZE, NOT one of the classes A-D (verifier 2026-10-24): an
                                       offline measurement
                                       on Rosario v1, the very rows the 'pigt' rule is judged on - the smallest
                                       confirmed camera-IMU clock offset (r04 / r06; r02 / r03 124.5 / 132.7 ms).  The
                                       constant doctrine is NOT met by it; it is used only by the opt-in, killed
                                       'pigt' rule, never by the default 'igt' path.  A next rule
                                       must derive its effect size (class B) before it touches these rows
    N_MC = 20000                   D   Monte-Carlo replays per probability (numerical; MC standard error at the
                                       decision level 2.2e-4, reported with every result)

THE POWERED IGT RULE:
    U6 fired its kill on both sides (Rosario v1 admitted at low rotation, good EuRoC IMUs refused under noisy VO
    rotations), so k_R > K alone is not an admission rule at live init.  'pigt' keeps the statistic and K and decides
    only when the decision is calibrated on the window: REFUSE if k_R > K and the false-refusal probability of a
    consistent IMU on this window (alpha) <= 1 - P_GATE, or if the clock offset estimated jointly with the gyro bias
    (offset_align) is resolved well enough to tell 0 from TD_HARM at P_GATE and is >= TD_HARM / 2; PASS if k_R <= K and
    either the probability that a TD_HARM offset would have given k_R > K (power) >= P_GATE or the resolved offset is
    < TD_HARM / 2; otherwise stay in VO.  First decision final; watchdog afterwards (watchdog_stat 'igt').
"""
from __future__ import annotations

import math
from collections import namedtuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import gammainc, gammaincinv

from xclean.core.lie import so3_exp, so3_jr, so3_jr_inv, so3_log
from xclean.vio import imu_factor as F

__all__ = ["G_STD", "P_GATE", "IGT_K", "IGT_INFLATE", "TILT_BAR_DEG", "MIN_INIT_KF", "N_CONSEC",
           "Kappa", "GyroAlign", "ViInit", "Admission", "BiasBound", "Watchdog", "Candidate",
           "chi2_ppf", "chi2_cdf", "kappa_imu", "kappa_from_component", "gyro_bias_align", "stereo_vi_init", "igt_k_R", "admission",
           "bias_bound", "bias_excursion", "watchdog_stat", "preintegrate_intervals", "check_causal", "att_chain",
           "init_candidate", "tilt_error_deg", "warmup",
           "TD_HARM", "N_MC", "ADMISSION_RULE", "OffsetAlign", "kf_rates", "offset_align", "igt_power", "igt_decide"]

G_STD = 9.80665
P_GATE = 0.999
IGT_K = 10.0
IGT_INFLATE = 3.0
TILT_BAR_DEG = 2.0
MIN_INIT_KF = 8
N_CONSEC = 2
MAX_GN = 20
MAX_VCE = 30
_EPS = float(np.finfo(np.float64).eps)
_SQEPS = math.sqrt(_EPS)
_GN_TOL = _EPS ** 0.75          # GN stop: step^T H step (the step in units of its own sigma, squared) < eps^0.75
_VCE_FRAC = 0.01                # VCE stop: every component moved by < 1 % of its own standard error s sqrt(2 / r)
                                # (numerical; the answer then differs from the fixed point by <= 0.01 sigma)
CHI2_3_MEDIAN = 2.365974  # median of chi2_3 (the IGT normaliser, backend/preint.py), for reference only


# ============================================================================================== small helpers ======
def chi2_ppf(p, dof):
    """chi2 quantile, dof may be non-integer."""
    return float(2.0 * gammaincinv(0.5 * float(dof), float(p)))


def chi2_cdf(x, dof):
    return float(gammainc(0.5 * float(dof), 0.5 * float(x)))


def _chi2_median(dof):
    return 2.0 * gammaincinv(0.5 * np.asarray(dof, float), 0.5)


def _hat(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _basis(g):
    """Two orthonormal columns perpendicular to g (deterministic: the world axis least aligned with g seeds it)."""
    u = np.asarray(g, float) / np.linalg.norm(g)
    a = np.eye(3)[int(np.argmin(np.abs(u)))]
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return np.stack([e1, e2], axis=1)


def _rotvec_between(a, b):
    """The rotation vector of the smallest rotation taking direction a to direction b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    c = np.cross(a, b)
    s = np.linalg.norm(c)
    ang = math.atan2(s, float(np.dot(a, b)))
    if s < _EPS:
        return np.zeros(3)
    return c / s * ang


def tilt_error_deg(g_a, g_b):
    """Angle between two gravity directions (degrees): the roll/pitch error when both are in the same frame."""
    return float(np.degrees(np.linalg.norm(_rotvec_between(g_a, g_b))))


def _rot_stack(T_wb):
    """(n+1, 3, 3) rotations and (n+1, 3) positions from a list/array of 4x4, or of (R, p) pairs."""
    R, p = [], []
    for T in T_wb:
        if isinstance(T, (tuple, list)) and len(T) == 2:
            R.append(np.asarray(T[0], float).reshape(3, 3))
            p.append(np.asarray(T[1], float).reshape(3))
        else:
            T = np.asarray(T, float)
            R.append(T[:3, :3])
            p.append(T[:3, 3])
    return np.array(R), np.array(p)


def _rot_only(R_wb):
    R_wb = np.asarray(R_wb, float)
    if R_wb.ndim == 3 and R_wb.shape[1:] == (4, 4):
        return R_wb[:, :3, :3]
    if R_wb.ndim == 3 and R_wb.shape[1:] == (3, 3):
        return R_wb
    raise ValueError("R_wb: expected (n, 3, 3) rotations or (n, 4, 4) poses, got %s" % (R_wb.shape,))


def _rank_deficient(M):
    """Numerical rank test, SCALE-INVARIANT: the Jacobi-scaled matrix D M D, D = diag(M_ii^-1/2), is singular to
    float64 precision (lambda_min <= dim eps lambda_max, Class D).  On the unscaled normal matrix the same test flags
    unit / magnitude disparity, not degeneracy (madmax_F3: a 34-s first interval next to a tight bias prior spans 16
    orders of magnitude and every window read 'rank' while driving)."""
    M = np.asarray(M, float)
    dg = np.diag(M)
    if not np.all(dg > 0.0) or not np.all(np.isfinite(M)):
        return True
    d = 1.0 / np.sqrt(dg)
    ev = np.linalg.eigvalsh(M * d[:, None] * d[None, :])
    return bool(ev.min() <= ev.size * _EPS * ev.max())


class _Chol:
    """Cholesky of a symmetric positive-definite matrix in JACOBI-EQUILIBRATED form: D N D = L L^T, D = diag(N_ii^-1/2);
    solve(B) = D (D N D)^-1 D B = N^-1 B.  The normal matrices here mix velocities, a tilt in radians and biases next
    to an 18-34 s stationary interval (madmax_F3, E1) and span ~16 orders of magnitude: the unscaled factorisation
    failed numerically on positive-definite matrices.  Raises LinAlgError when N is not positive definite."""
    __slots__ = ("d", "cf")

    def __init__(self, N):
        N = np.asarray(N, float)
        dg = np.diag(N)
        if not np.all(dg > 0.0):
            raise np.linalg.LinAlgError("non-positive diagonal")
        self.d = 1.0 / np.sqrt(dg)
        self.cf = cho_factor(N * self.d[:, None] * self.d[None, :], lower=True, check_finite=False)

    def solve(self, B):
        B = np.asarray(B, float)
        dd = self.d if B.ndim == 1 else self.d[:, None]
        return dd * cho_solve(self.cf, dd * B, check_finite=False)


def _chol_lower(C):
    """Lower Cholesky factor of a covariance through Jacobi equilibration: D C D = Ls Ls^T, L = D^-1 Ls (still lower
    triangular, L L^T = C).  Same reason as _Chol: the residual covariances span many orders of magnitude next to a
    long stationary interval.  Raises LinAlgError when C is not positive definite."""
    C = np.asarray(C, float)
    dg = np.diag(C)
    if not np.all(dg > 0.0):
        raise np.linalg.LinAlgError("non-positive diagonal")
    d = 1.0 / np.sqrt(dg)
    return np.linalg.cholesky(C * d[:, None] * d[None, :]) / d[:, None]


def _inv_lower(L):
    return solve_triangular(L, np.eye(L.shape[0]), lower=True)


# ============================================================================================== kappa_imu (4) =====
Kappa = namedtuple("Kappa", "kappa estimate lo hi lo68 hi68 n rule")
Kappa.__doc__ = ("kappa: the value to use (rule applied); estimate: the robust estimate; [lo, hi]: its distribution-free "
                 "P_GATE interval, [lo68, hi68]: the 68.27 % interval; n: factors used; rule: which branch of the rule")


def _median_ci(x, level):
    """Distribution-free order-statistic interval of the median of the sample x at `level` (binomial(n, 1/2)):
    the widest-tail indices j with P(Bin(n, 1/2) < j) <= (1 - level) / 2.  (lo, hi) = (x_(j), x_(n-j+1)); when n is
    too small for the level, the interval is the whole real half-line (0, inf)."""
    from scipy.stats import binom
    x = np.sort(np.asarray(x, float))
    n = len(x)
    a = 0.5 * (1.0 - level)
    j = int(binom.ppf(a, n, 0.5))           # P(Bin <= j) >= a ;  P(Bin <= j - 1) < a
    while j > 0 and binom.cdf(j - 1, n, 0.5) > a:
        j -= 1
    if j < 1:
        return 0.0, math.inf
    return float(x[j - 1]), float(x[n - j])


def kappa_imu(chi2, dof, p=P_GATE):
    """The Class-C IMU noise scale (a variance factor on the whole 15 x 15, imu_factor WHITENING) from per-factor
    whitened chi2 values and their degrees of freedom (the factor's dimension, or its redundancy for post-fit
    residuals; non-integer allowed).

    ESTIMATOR (U7's definition, made robust and dof-aware): kappa_k = chi2_k / median(chi2_{dof_k}) is each factor's
    normalised chi2 (consistent value 1); estimate = median_k kappa_k.  With equal dof this is exactly U7's
    `median_norm_chi2_9` (median chi2_9 / 8.343).  If the residuals are N(0, kappa C), chi2_k / kappa ~ chi2_dof_k, so
    every kappa_k has population median kappa and the sample median is consistent with a 50 % breakdown point
    (a minority of outlying factors cannot move it).
    UNCERTAINTY: the distribution-free order-statistic interval of the median (binomial(n, 1/2)) at P_GATE and at
    68.27 %; no distributional assumption beyond independence of the factors.
    THE RULE: kappa = max(1, estimate) - DESIGN_P1 6.3 "bounded below by 1": the declared density is the Class-A
    floor, and shrinking the IMU covariance below it makes the IMU over-confident, the failure the design guards
    against (4s noise 10-206x too tight).  When the data REJECT kappa >= 1 at P_GATE (the upper end of the P_GATE
    interval < 1) the declared density is conservative; that is REPORTED in `rule` and kappa stays 1.  (The version
    before 2026-10-24 returned that upper bound below 1, contradicting the design: verifier finding)
    Returns None when no finite factor is given."""
    c = np.atleast_1d(np.asarray(chi2, float))
    d = np.broadcast_to(np.asarray(dof, float), c.shape)
    ok = np.isfinite(c) & np.isfinite(d) & (d > 0.0) & (c >= 0.0)
    if not ok.any():
        return None
    kk = c[ok] / _chi2_median(d[ok])
    est = float(np.median(kk))
    lo, hi = _median_ci(kk, p)
    lo68, hi68 = _median_ci(kk, math.erf(1.0 / math.sqrt(2.0)))
    if est < 1.0:
        kap = 1.0
        rule = ("floor 1 (the data reject kappa >= 1 at P_GATE: the declared density is conservative; held at 1, "
                "DESIGN_P1 6.3)" if hi < 1.0 else "floor 1 (estimate below 1, not significant)")
    else:
        kap, rule = est, "estimate"
    return Kappa(float(kap), est, lo, hi, lo68, hi68, int(ok.sum()), rule)


def kappa_from_component(est, sd, p=P_GATE):
    """The same rule for the variance-component estimate of the IMU scale (stereo_vi_init's kappa): the REML
    estimate is approximately est * chi2_nu / nu with nu = 2 est^2 / sd^2 (Satterthwaite moment matching), so its
    P_GATE interval is [est nu / chi2_nu(1 - a/2), est nu / chi2_nu(a/2)], a = 1 - P_GATE; then kappa_imu's rule."""
    if not (np.isfinite(est) and np.isfinite(sd) and sd > 0 and est > 0):
        return Kappa(1.0, float(est), 0.0, math.inf, 0.0, math.inf, 0, "no estimate: 1")
    nu = 2.0 * est * est / (sd * sd)
    a = 1.0 - p
    a68 = 1.0 - math.erf(1.0 / math.sqrt(2.0))

    def _div(q):
        if nu > 1.0 / _EPS:                          # sd below float64 resolution of est: the estimate is exact
            return est
        c = chi2_ppf(q, nu)
        with np.errstate(over="ignore"):
            return float(est * nu / c) if c > 0.0 else math.inf
    lo, hi = _div(1.0 - a / 2), _div(a / 2)
    lo68, hi68 = _div(1.0 - a68 / 2), _div(a68 / 2)
    if est < 1.0:
        kap = 1.0
        rule = ("floor 1 (the data reject kappa >= 1 at P_GATE: the declared density is conservative; held at 1, "
                "DESIGN_P1 6.3)" if hi < 1.0 else "floor 1 (estimate below 1, not significant)")
    else:
        kap, rule = est, "estimate"
    return Kappa(float(kap), float(est), float(lo), float(hi), float(lo68), float(hi68), int(round(nu)), rule)


# ========================================================================================= gyro_bias_align (1) =====
GyroAlign = namedtuple("GyroAlign", "bg cov chi2 dof chi2_dof kappa sigma_th comp_sd n removed iters converged "
                                    "vce_iters vce_converged chi_k leverage prior_chi2 bg_data cov_data used satt")
GyroAlign.__doc__ = ("bg (3,) and cov (3x3): the posterior (data + prior when given); bg_data / cov_data: data only; "
                     "chi2 = the GLS chi2 of the rotation residuals at the fitted noise model, dof = 3 m - 3, chi2_dof; "
                     "kappa = Kappa of the gyro (IMU) component, sigma_th = the KF attitude jitter [rad] (REML), comp_sd "
                     "their standard errors (variances); n factors used, removed = snooped record indices; iters (GN), "
                     "converged; vce_iters, vce_converged; chi_k (m,) per-factor chi at the DECLARED rotation covariance "
                     "(the naive U7 / IGT form); leverage (m,); prior_chi2: the data-only estimate vs the prior, chi2_3 "
                     "under H0 (cov_data + cov_prior), None without a prior; used = indices of the records used")


def _rot_residuals(R, pres, used, bg):
    """Per used factor: r = Log((dR Exp(JRg (bg - bg_lin)))^T R_k^T R_k+1), J = dr/dbg (exact, as imu_factor),
    Jr^-1(r) and A = R_k^T R_k+1."""
    m = len(used)
    dR = np.stack([pres[k].dR for k in used])
    JRg = np.stack([pres[k].JRg for k in used])
    bgl = np.stack([pres[k].bg_lin for k in used])
    idx = np.asarray(used)
    A = np.einsum("nji,njk->nik", R[idx], R[idx + 1])                 # R_k^T R_k+1
    phi = np.einsum("nij,nj->ni", JRg, bg[None, :] - bgl)
    Eb = so3_exp(phi).reshape(m, 3, 3)
    Jrb = so3_jr(phi).reshape(m, 3, 3)
    dRc = np.einsum("nij,njk->nik", dR, Eb)
    E = np.einsum("nji,njk->nik", dRc, A)                             # dRc^T A
    r = so3_log(E).reshape(m, 3)
    Jri = so3_jr_inv(r).reshape(m, 3, 3)
    J = -np.einsum("nij,nkj,nkl,nlm->nim", Jri, E, Jrb, JRg)          # -Jr^-1(r) E^T Jr(phi) JRg
    return r, J, Jri, A


def _align_fit(R, pres, used, n_kf, bg, noise, th_fixed, max_iter):
    """GLS fit of bg with the two-component noise model (gyro x kappa, KF attitude jitter) by REML; data only."""
    m = len(used)
    # the gyro component's FACTOR directly (as stereo_vi_init's U_imu): per-interval symmetric square roots + the
    # low-rank factor of the gyro bias walk across intervals pinned at t_n
    Ub = np.zeros((3 * m, 3 * m))
    for r_, k in enumerate(used):
        ev, U = np.linalg.eigh(0.5 * (pres[k].cov[0:3, 0:3] + pres[k].cov[0:3, 0:3].T))
        Ub[3 * r_:3 * r_ + 3, 3 * r_:3 * r_ + 3] = U * np.sqrt(np.maximum(ev, 0.0))[None, :]
    parts = [Ub]
    if noise is not None:                                        # gyro bias walk across intervals, pinned at t_n
        tk = np.array([pres[k].t0 for k in used])
        K = float(pres[used[-1]].t1) - np.maximum.outer(tk, tk)
        evk, Uk = np.linalg.eigh(0.5 * (K + K.T))
        _r, Jb0, _a, _b = _rot_residuals(R, pres, used, bg)
        Jb = np.zeros((3 * m, 3 * m))
        for r_ in range(m):
            Jb[3 * r_:3 * r_ + 3, 3 * r_:3 * r_ + 3] = Jb0[r_]
        parts.append(noise.rw_g * (Jb @ np.kron(Uk * np.sqrt(np.maximum(evk, 0.0))[None, :], np.eye(3))))
    Ug = np.concatenate(parts, axis=1)
    Qg = Ug @ Ug.T
    s_init = np.array([1.0, float(np.median([np.trace(pres[k].cov[0:3, 0:3]) / 3.0 for k in used]))])
    fixed = np.array([False, th_fixed is not None])
    s = s_init.copy()
    if th_fixed is not None:
        s[1] = float(th_fixed) ** 2
    it_gn, conv_gn, conv_vce, it_vce = 0, False, False, 0
    sd = np.full(2, np.inf)
    for it_vce in range(1, MAX_VCE + 1):
        for it_gn in range(1, int(max_iter) + 1):
            r, J, Jri, A = _rot_residuals(R, pres, used, bg)
            Jth = np.zeros((3 * m, 3 * n_kf))
            for r_, k in enumerate(used):
                Jth[3 * r_:3 * r_ + 3, 3 * k:3 * k + 3] = -Jri[r_] @ A[r_].T
                Jth[3 * r_:3 * r_ + 3, 3 * k + 3:3 * k + 6] = Jri[r_]
            C = s[0] * Qg + s[1] * (Jth @ Jth.T)
            L = _chol_lower(C)
            Jw = solve_triangular(L, J.reshape(3 * m, 3), lower=True)
            rw = solve_triangular(L, r.reshape(3 * m), lower=True)
            N = Jw.T @ Jw
            step = -np.linalg.solve(N, Jw.T @ rw)
            bg = bg + step
            if float(step @ N @ step) < _GN_TOL:
                conv_gn = True
                break
        r, J, Jri, A = _rot_residuals(R, pres, used, bg)
        rw = solve_triangular(L, r.reshape(3 * m), lower=True)
        cf = _Chol(N)
        new, sd = _reml_step(L, [Ug, Jth], rw, Jw, cf, s, fixed, s_init)
        ch = np.abs(new - s)
        s = new
        if np.all(ch <= _VCE_FRAC * sd):
            conv_vce = True
            break
    # final linearisation at the converged components
    r, J, Jri, A = _rot_residuals(R, pres, used, bg)
    Jth = np.zeros((3 * m, 3 * n_kf))
    for r_, k in enumerate(used):
        Jth[3 * r_:3 * r_ + 3, 3 * k:3 * k + 3] = -Jri[r_] @ A[r_].T
        Jth[3 * r_:3 * r_ + 3, 3 * k + 3:3 * k + 6] = Jri[r_]
    C = s[0] * Qg + s[1] * (Jth @ Jth.T)
    L = _chol_lower(C)
    Jw = solve_triangular(L, J.reshape(3 * m, 3), lower=True)
    rw = solve_triangular(L, r.reshape(3 * m), lower=True)
    N = Jw.T @ Jw
    cf = _Chol(N)
    Ni = np.linalg.inv(N)                                                 # plug-in (snooping, leverage)
    satt = _satterthwaite(Ni, L, [Ug, Jth], Jw, cf, s, fixed, s_init, {"bg": np.arange(3)})
    raw = []
    for gi, U in enumerate([Ug, Jth]):
        num, rg = _vce_terms(L, U, rw, Jw, cf, s[gi])
        raw.append(s[gi] * num / rg if rg > _SQEPS else float("nan"))
    _s, sd = _reml_step(L, [Ug, Jth], rw, Jw, cf, s, fixed, s_init)
    # standardised per-factor residuals (snooping): Cov(e) = C - J N^-1 J^T, 3 x 3 blocks
    Jf = J.reshape(3 * m, 3)
    wst = np.zeros(m)
    lev = np.zeros(m)
    for r_ in range(m):
        sl = slice(3 * r_, 3 * r_ + 3)
        Ce = C[sl, sl] - Jf[sl] @ Ni @ Jf[sl].T
        ev, U = np.linalg.eigh(0.5 * (Ce + Ce.T))
        keep = ev > 3 * _EPS * max(ev.max(), _EPS)
        z = U[:, keep].T @ r[r_]
        wst[r_] = float(np.sum(z * z / ev[keep]))
        lev[r_] = float(np.trace(Jw[sl] @ Ni @ Jw[sl].T))
    Wd = np.stack([_inv_lower(np.linalg.cholesky(pres[k].cov[0:3, 0:3])) for k in used])
    rwd = np.einsum("nij,nj->ni", Wd, r)
    chi_k = np.einsum("ni,ni->n", rwd, rwd)
    return dict(bg=bg, N=N, cov=_scale_blocks(Ni, [(np.arange(3), satt["bg"][0])]), satt=satt, C=C, chi2=float(rw @ rw), s=s, raw=raw, sd=sd, wst=wst, lev=lev, chi_k=chi_k,
                it_gn=it_gn, conv_gn=conv_gn, it_vce=it_vce, conv_vce=conv_vce, Jw=Jw, L=L)


def gyro_bias_align(R_wb, preints, bg_prior=None, bg_prior_cov=None, noise=None, sigma_th=None, p=P_GATE,
                    max_iter=MAX_GN):
    """Deliverable 1: the gyro bias from the rotation consistency of the VO keyframe rotations.

    Least squares over the consecutive KF pairs of r_k(bg) = Log((dR_k Exp(JRg_k (bg - bg_lin_k)))^T R_k^T R_k+1)
    (imu_factor's rR, the first-order bias correction from the Preint), Gauss-Newton with the exact Jacobian.
    NOISE MODEL (the same errors-in-variables form as stereo_vi_init): C = kappa Q_gyro + s_th J_th J_th^T, with
    Q_gyro = blockdiag(Preint.cov[0:3, 0:3]) (+ the gyro bias walk across intervals, pinned at t_n, when `noise` is
    given) and J_th the residuals' Jacobian w.r.t. the KF attitudes (d r_k / d theta_k = -Jr^-1(r_k) (R_k^T R_k+1)^T,
    d r_k / d theta_k+1 = Jr^-1(r_k)): consecutive residuals share a KF, so the VO attitude jitter TELESCOPES along the
    window and is not independent per factor (a single variance factor over-states cov(bg) by about the number of
    factors; measured 10x in test a2 before this model).  kappa and s_th by REML (stereo_vi_init); s_th may be fixed
    (sigma_th [rad]).  ROBUSTNESS: Baarda data snooping, the largest standardised factor residual against chi2_3 at
    (1 - P_GATE) / m, removed while it fails, at most half of the factors.
    PRIOR (bg_prior, bg_prior_cov; e.g. the causal stationary bootstrap): the data-only fit is made FIRST (a variance
    factor estimated at a prior's start value would inflate and lock the solution onto the prior), then combined at
    the data's own noise model; prior_chi2 tests the prior against the data.
    Returns GyroAlign; None when fewer than three usable records exist."""
    R = _rot_only(R_wb)
    if len(preints) != len(R) - 1:
        raise ValueError("gyro_bias_align: %d records for %d keyframes (need n records for n + 1 KFs)"
                         % (len(preints), len(R)))
    used0 = [k for k, q in enumerate(preints) if q is not None and q.ok]
    if len(used0) < 3:
        return None
    if isinstance(noise, dict):
        noise = F.ImuNoise.from_calib({"imu_noise": noise})
    bg0 = np.asarray(preints[used0[0]].bg_lin, float).copy()
    used = list(used0)
    removed = []
    while True:
        try:
            res = _align_fit(R, preints, used, len(R), bg0.copy(), noise, sigma_th, max_iter)
        except np.linalg.LinAlgError:
            return None
        m = len(used)
        if len(removed) >= len(used0) // 2 or m <= 3:
            break
        kmax = int(np.argmax(res["wst"]))
        if not (res["wst"][kmax] > chi2_ppf(1.0 - (1.0 - p) / m, 3)):
            break
        removed.append(used[kmax])
        used = [k for k in used if k != used[kmax]]
    bg_d, cov_d = res["bg"], 0.5 * (res["cov"] + res["cov"].T)
    bg, cov, pchi = bg_d.copy(), cov_d, None
    if bg_prior is not None and bg_prior_cov is not None:
        bp = np.asarray(bg_prior, float).reshape(3)
        P0 = np.asarray(bg_prior_cov, float)
        dd = bg_d - bp
        pchi = float(dd @ np.linalg.solve(cov_d + P0, dd))
        P0i = np.linalg.inv(P0)
        Ndi = np.linalg.inv(cov_d)                              # the data's information at its own noise model
        cov = np.linalg.inv(Ndi + P0i)
        bg = cov @ (Ndi @ bg_d + P0i @ bp)                      # the problem is linear to first order around bg_d
        cov = 0.5 * (cov + cov.T)
    kap = kappa_from_component(res["raw"][0], res["sd"][0], p)
    m = len(used)
    dof = 3 * m - 3
    return GyroAlign(bg, cov, res["chi2"], dof, res["chi2"] / max(dof, 1), kap, float(math.sqrt(max(res["s"][1], 0.0))),
                     [float(x) for x in res["sd"]], m, removed, res["it_gn"], res["conv_gn"], res["it_vce"],
                     res["conv_vce"], res["chi_k"], res["lev"], pchi, bg_d, cov_d, used, res["satt"])


# ========================================================================================== stereo_vi_init (2) =====
ViInit = namedtuple("ViInit", "ok reason g_w v_w ba bg cov_x cov_tilt cov_g cov_ba cov_bg cov_v basis "
                              "tilt_sigma_deg tilt_r_deg tilt_r_known_ba_deg vif cond_gb rank_deficient degenerate "
                              "degenerate_cause excitation rot_span_deg g_free g_free_norm g_free_sigma dchi2_mag "
                              "mag_ok chi2 dof chi2_dof kappa_vce sigma_p sigma_th comp_sd vce_iters vce_converged "
                              "removed n_used kappa chi9 dof9 kappa_naive info_tb t_span satt t_anchor tilt_r_gw_deg att_sd_newest_deg "
                              "att_chain n_touched igt_state tilt_r_laplace_deg tilt_r_exact_deg gpost",
                    defaults=(None, None, None, None))
ViInit.__doc__ = ("stereo_vi_init result; field meanings in its docstring.  SHAPES: g_w, ba, bg (3,); v_w (n_kf, 3) "
                  "(NaN for a KF no usable record touches); cov_v (n_kf, 3, 3) PER-KF velocity covariances (inf "
                  "diagonal for an untouched KF) - the newest KF's is cov_v[-1]; cov_x the joint covariance in the "
                  "layout [v (touched KFs), tilt (2, basis), ba, bg].  igt_state (set by init_candidate under the "
                  "'pigt' rule): None | 'pass' - the powered IGT decision carried to the next KF's candidate (final).  "
                  "tilt_r_gw_deg = the CERTIFIED P_GATE radius of g_w = max(tilt_r_laplace_deg, tilt_r_exact_deg); "
                  "tilt_r_exact_deg: from the exact posterior on the sphere (None when not computed: the Laplace "
                  "radius already fails the bar); gpost: that posterior's summary (_gravity_posterior).")


class _Sys:
    """The linear system of one init: rows for the used intervals, unknowns x = [v_0..v_n, g | d, ba, bg]."""

    def __init__(self, R, p, pres, used, n_kf, noise=None):
        self.R, self.p, self.pres, self.used, self.n_kf = R, p, pres, list(used), n_kf
        # velocity unknowns only for the KFs a usable record touches (a refused record, e.g. KF 0 before the first IMU
        # sample on MadMax, leaves its KF's v in no row: a zero column, and the normal matrix is singular)
        self.touched = sorted(set(self.used) | set(k + 1 for k in self.used))
        self.vcol = {kf: 3 * c for c, kf in enumerate(self.touched)}
        self.nv = 3 * len(self.touched)
        self.m = len(self.used)
        m = self.m
        A = np.zeros((6 * m, self.nv + 9))
        b = np.zeros(6 * m)
        G0, B0, D0 = self.nv, self.nv + 3, self.nv + 6
        Qimu = np.zeros((6 * m, 6 * m))
        Jp = np.zeros((6 * m, 3 * n_kf))                        # KF position jitter: one 3-block per KF
        self.dts = np.zeros(m)
        for r, k in enumerate(self.used):
            q = pres[k]
            i, j = k, k + 1
            RiT = R[i].T
            dt = q.dt
            self.dts[r] = dt
            o = 6 * r
            # e_v rows
            ci, cj = self.vcol[i], self.vcol[j]
            A[o:o + 3, cj:cj + 3] = RiT
            A[o:o + 3, ci:ci + 3] = -RiT
            A[o:o + 3, G0:G0 + 3] = -RiT * dt
            A[o:o + 3, B0:B0 + 3] = -q.Jva
            A[o:o + 3, D0:D0 + 3] = -q.Jvg
            b[o:o + 3] = q.dv - q.Jva @ q.ba_lin - q.Jvg @ q.bg_lin
            # e_p rows
            A[o + 3:o + 6, ci:ci + 3] = -RiT * dt
            A[o + 3:o + 6, G0:G0 + 3] = -0.5 * RiT * dt * dt
            A[o + 3:o + 6, B0:B0 + 3] = -q.Jpa
            A[o + 3:o + 6, D0:D0 + 3] = -q.Jpg
            b[o + 3:o + 6] = q.dp - q.Jpa @ q.ba_lin - q.Jpg @ q.bg_lin - RiT @ (p[j] - p[i])
            Qimu[o:o + 6, o:o + 6] = q.cov[3:9, 3:9]
            Jp[o + 3:o + 6, 3 * i:3 * i + 3] = -RiT
            Jp[o + 3:o + 6, 3 * j:3 * j + 3] = RiT
        # the IMU component's FACTOR is built directly (U_imu U_imu^T = Q_imu), never by factoring the sum: per
        # interval the symmetric square root of its 6 x 6 block (eigen; rounding negatives clipped to 0), plus the
        # low-rank bias-walk factors.  (madmax_F3: a 34-s first interval has a nearly singular v/p block, and the
        # Cholesky of the summed Q_imu failed on every window while the vehicle was driving.)
        Ub = np.zeros((6 * m, 6 * m))
        for r in range(m):
            blk = 0.5 * (Qimu[6 * r:6 * r + 6, 6 * r:6 * r + 6] + Qimu[6 * r:6 * r + 6, 6 * r:6 * r + 6].T)
            ev, U = np.linalg.eigh(blk)
            Ub[6 * r:6 * r + 6, 6 * r:6 * r + 6] = U * np.sqrt(np.maximum(ev, 0.0))[None, :]
        Uparts = [Ub]
        # the bias RANDOM WALK across intervals (Class A densities, scaled with kappa like the whole 15 x 15): the
        # unknown ba, bg are the biases at the newest KF t_n; interval k starts at t_k with ba + w_a(t_k), where w is
        # a Brownian motion pinned at t_n: Cov(w(t_k), w(t_l)) = rw^2 (t_n - max(t_k, t_l)) I.  It enters row block k
        # through -[Jva; Jpa] (and -[Jvg; Jpg] for the gyro).  The within-interval drift is already in Preint.cov.
        if noise is not None:
            tk = np.array([pres[k].t0 for k in self.used])
            tn = float(pres[self.used[-1]].t1)
            K = tn - np.maximum.outer(tk, tk)
            evk, Uk = np.linalg.eigh(0.5 * (K + K.T))
            Ksq = np.kron(Uk * np.sqrt(np.maximum(evk, 0.0))[None, :], np.eye(3))
            Ja = np.zeros((6 * m, 3 * m))
            Jg = np.zeros((6 * m, 3 * m))
            for r, k in enumerate(self.used):
                q = pres[k]
                Ja[6 * r:6 * r + 3, 3 * r:3 * r + 3] = -q.Jva
                Ja[6 * r + 3:6 * r + 6, 3 * r:3 * r + 3] = -q.Jpa
                Jg[6 * r:6 * r + 3, 3 * r:3 * r + 3] = -q.Jvg
                Jg[6 * r + 3:6 * r + 6, 3 * r:3 * r + 3] = -q.Jpg
            Uparts += [noise.rw_a * (Ja @ Ksq), noise.rw_g * (Jg @ Ksq)]
            self.t_anchor = tn
        else:
            self.t_anchor = None
        self.Uimu = np.concatenate(Uparts, axis=1)
        Qimu = self.Uimu @ self.Uimu.T
        self.A, self.b, self.Qimu, self.Jp = A, b, Qimu, Jp
        self.G0, self.B0, self.D0 = G0, B0, D0

    def set_att_rw(self, scale):
        """The VO ATTITUDE RANDOM WALK (fixed, not estimated here): increments q_l = scale tr(C_RR,l) / 3 (isotropic)
        per KF interval l, anchored at KF 0 (the gauge of w); Qc[i] = sum_{l < i} q_l is KF i's attitude variance.
        K[r, s] = Qc[min(i_r, i_s)] for the used intervals' start KFs i_r (Brownian covariance)."""
        q = np.array([(scale * np.trace(self.pres[l].cov[0:3, 0:3]) / 3.0) if (self.pres[l] is not None and
                                                                                 self.pres[l].ok) else 0.0
                      for l in range(self.n_kf - 1)])
        self.att_q = q
        self.att_Qc = np.concatenate([[0.0], np.cumsum(q)])      # (n_kf,)
        iu = np.asarray(self.used)
        K = self.att_Qc[np.minimum.outer(iu, iu)]
        ev, U = np.linalg.eigh(0.5 * (K + K.T))
        self.att_Ksq = U * np.sqrt(np.maximum(ev, 0.0))[None, :]     # K = Ksq Ksq^T
        self.att_on = bool(scale > 0.0 and np.any(q > 0.0))

    def Ua(self, Jth):
        """The fixed attitude-random-walk component's factor: C_a = Ua Ua^T, Ua = Jth (Ksq kron I3)."""
        return Jth @ np.kron(self.att_Ksq, np.eye(3))

    def Jth(self, x):
        """Attitude-jitter Jacobian blocks (6m x 3m): e_v/dphi_i = [R_i^T Dv]x, e_p/dphi_i = [R_i^T Dp]x."""
        m = self.m
        J = np.zeros((6 * m, 3 * m))
        g = x[self.G0:self.G0 + 3]
        for r, k in enumerate(self.used):
            i, j = k, k + 1
            dt = self.dts[r]
            ci, cj = self.vcol[i], self.vcol[j]
            vi, vj = x[ci:ci + 3], x[cj:cj + 3]
            Dv = vj - vi - g * dt
            Dp = self.p[j] - self.p[i] - vi * dt - 0.5 * g * dt * dt
            J[6 * r:6 * r + 3, 3 * r:3 * r + 3] = _hat(self.R[i].T @ Dv)
            J[6 * r + 3:6 * r + 6, 3 * r:3 * r + 3] = _hat(self.R[i].T @ Dp)
        return J


# ============================================= the |g| = g_mag problem solved GLOBALLY; the exact gravity posterior ===
_POST_UNIF = 3600        # D numerical: theta grid over the whole great circle, 0.1 deg
_POST_DENSE = 481        # D numerical: theta points over +-_POST_K Laplace sd around each local minimum (0.05 sd step)
_POST_K = 12.0           # D numerical: a Gaussian beyond 12 sd carries e^-72 of its mass (below float64 relevance)
_POST_PSI = 721          # D numerical: psi quadrature over [-pi/2, pi/2] (0.25 deg) where Laplace in psi is not used
_POST_PSI_SD = 0.01      # D numerical: Laplace in psi when its conditional sd <= 0.01 rad (the sphere-curvature error of
                         # the Gaussian approximation is O(sd^2) = 1e-4 relative)
_POST_GEO = 400          # D numerical: geometric theta grid 1e-7 .. pi rad on each side of a mode (3.5 % steps: resolves a
                         # peak of any width >= 1e-6 rad, scale-free)
_LOG_TINY = 745.0        # D numerical: exp(-745) underflows float64 to 0 - a row more e-folds down contributes exactly 0
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _profile_g(S, L, prior_free):
    """The init's chi2 as a function of g ALONE at a FIXED noise model C = L L^T.  With the poses known the problem is
    linear in x = [v, g, ba, bg], so eliminating y = (v, ba, bg) (with their priors) is exact:
        chi2(g) = g^T Sg g - 2 sg^T g + c0,   Sg = N_gg - N_gy N_yy^-1 N_yg,   sg = h_g - N_gy N_yy^-1 h_y
    (the Schur complement of the free normal equations; free layout [v, g, ba, bg]).  Returns (Sg, sg, c0, aux); aux
    recovers y(g) = X[:, 3] - X[:, :3] g and holds the whitened free system (Aw, bw).  Raises LinAlgError when y is
    not determined given g."""
    Aw = solve_triangular(L, S.A, lower=True)
    bw = solve_triangular(L, S.b, lower=True)
    N = Aw.T @ Aw
    h = Aw.T @ bw
    c = float(bw @ bw)
    for (cols, Pi, mu) in prior_free:
        N[np.ix_(cols, cols)] += Pi
        h[cols] += Pi @ mu
        c += float(mu @ Pi @ mu)
    gi = np.arange(S.G0, S.G0 + 3)
    yi = np.concatenate([np.arange(0, S.G0), np.arange(S.G0 + 3, N.shape[0])])
    Nyg = N[np.ix_(yi, gi)]
    X = _Chol(N[np.ix_(yi, yi)]).solve(np.concatenate([Nyg, h[yi][:, None]], axis=1))
    Sg = N[np.ix_(gi, gi)] - Nyg.T @ X[:, :3]
    sg = h[gi] - Nyg.T @ X[:, 3]
    c0 = c - float(h[yi] @ X[:, 3])
    return 0.5 * (Sg + Sg.T), sg, c0, {"Aw": Aw, "bw": bw, "X": X}


def _mode_at(g, mu, Sg, sg, G):
    """One stationary point of q on the sphere as a dict {g, mu, q, B, Ht} (see _sphere_modes)."""
    g = np.asarray(g, float)
    ng = float(np.linalg.norm(g))
    if not (ng > 0.0 and np.isfinite(ng)):
        return None
    g = g * (G / ng)
    B = _basis(g)
    Gm = -_hat(g) @ B
    Ht = Gm.T @ (Sg - float(mu) * np.eye(3)) @ Gm
    return {"g": g, "mu": float(mu), "q": float(g @ Sg @ g - 2.0 * sg @ g), "B": B, "Ht": 0.5 * (Ht + Ht.T)}


def _lae(a, b):
    """log(e^a + e^b) for scalars (either may be -inf)."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = a if a > b else b
    return m + math.log1p(math.exp(-abs(a - b)))


def _sphere_modes(Sg, sg, G):
    """Every LOCAL MINIMUM of q(g) = g^T Sg g - 2 sg^T g on the sphere |g| = G (the profile chi2 of _profile_g).
    This is the trust-region subproblem with an equality constraint (Gay 1981; More & Sorensen 1983; Martinez, SIAM J.
    Optim. 1994): a stationary point satisfies (Sg - mu I) g = sg; the GLOBAL minimiser has mu <= lambda_1(Sg); at
    most ONE local non-global minimiser exists, with lambda_1 < mu < lambda_2.  In the eigenbasis Sg = E diag(lam) E^T,
    beta = E^T sg: g(mu) = E beta / (lam - mu) and the secular equation phi(mu) = sum beta_i^2 / (lam_i - mu)^2 = G^2
    is solved in log form on variables that keep lam_i - mu exact near the poles: below lambda_1, t = log(lam_1 - mu)
    (log phi decreasing in t; analytic bracket: t <= log(2|beta| / G) gives phi <= G^2 / 4, t >= log(|beta_1| / (2G))
    gives phi >= 4 G^2); inside (lambda_1, lambda_2), z = logit((mu - lam_1) / (lam_2 - lam_1)) (phi convex in mu:
    its minimum from the monotone phi', then one root on each side); scalar brentq to machine precision on the float64
    exponent range (no tolerance set by hand).  The HARD case (beta_1 = 0 to float64: phi stays below G^2 up to the
    pole; in exact arithmetic the yaw-only sign ambiguity) returns its two minimisers g = E (w +- tau e_1),
    w_i = beta_i / (lam_i - lam_1), tau = sqrt(G^2 - |w|^2).  A point inside (lambda_1, lambda_2) is kept when its
    tangent Hessian is positive semi-definite (to 64 eps of the matrix scale).
    Returns a list of dicts {g, mu, q, B, Ht} sorted by q (the global minimum first).  B = _basis(g) and
    Ht = Gm^T (Sg - mu I) Gm, Gm = -[g]x B: q(Exp(B d) g) = q + d^T Ht d + O(d^3) - the tangent Hessian INCLUDING the
    constraint's curvature -mu G^2 I that a tangent-space Gauss-Newton omits."""
    from scipy.optimize import brentq
    Sg = 0.5 * (np.asarray(Sg, float) + np.asarray(Sg, float).T)
    sg = np.asarray(sg, float)
    lam, E = np.linalg.eigh(Sg)
    beta = E.T @ sg
    lb = [math.log(abs(float(x))) if x != 0.0 else -math.inf for x in beta]
    dl = [max(float(lam[i] - lam[0]), 0.0) for i in range(3)]
    ldl = [math.log(x) if x > 0.0 else -math.inf for x in dl]
    lG2 = 2.0 * math.log(G)
    LO, HI = -745.0, 709.0                                     # the float64 exponent range (exp underflow / overflow)
    out = []
    nb = float(np.linalg.norm(beta))

    def Fg(t):                                                 # log phi - log G^2 below lambda_1 (decreasing in t)
        acc = -math.inf
        for i in range(3):
            if lb[i] > -math.inf:
                acc = _lae(acc, 2.0 * lb[i] - 2.0 * _lae(ldl[i], t))
        return acc - lG2
    if nb > 0.0:
        t_hi = min(math.log(2.0 * nb / G), HI)
        t_lo = max(lb[0] - math.log(2.0 * G), LO) if lb[0] > -math.inf else LO
        if Fg(t_lo) > 0.0 and Fg(t_hi) < 0.0:
            t_ = brentq(Fg, t_lo, t_hi, xtol=_EPS, rtol=4.0 * _EPS, maxiter=500)
            y = math.exp(t_)
            md = _mode_at(E @ (beta / (np.asarray(dl) + y)), lam[0] - y, Sg, sg, G)
            if md is not None:
                out.append(md)
        elif Fg(t_lo) <= 0.0:                                  # hard case
            w = np.array([beta[i] / dl[i] if dl[i] > 0.0 else 0.0 for i in range(3)])
            w[0] = 0.0
            tau = math.sqrt(max(G * G - float(w @ w), 0.0))
            for sgn in (1.0, -1.0):
                ww = w.copy()
                ww[0] = sgn * tau
                md = _mode_at(E @ ww, lam[0], Sg, sg, G)
                if md is not None:
                    out.append(md)
    scale = float(np.max(np.abs(lam))) if np.max(np.abs(lam)) > 0 else 1.0
    if dl[1] > 8.0 * _EPS * scale and lb[0] > -math.inf and (lb[1] > -math.inf or lb[2] > -math.inf):
        ld1 = ldl[1]
        ld21 = math.log(dl[2] - dl[1]) if dl[2] > dl[1] else -math.inf

        def logs(z):                                           # log|lam_i - mu|, i = 1..3, at logit z
            lx = ld1 - _lae(0.0, -z)
            l1 = ld1 - _lae(0.0, z)
            return lx, l1, _lae(ld21, l1)

        def Fi(z):
            ls_ = logs(z)
            acc = -math.inf
            for i in range(3):
                if lb[i] > -math.inf:
                    acc = _lae(acc, 2.0 * lb[i] - 2.0 * ls_[i])
            return acc - lG2

        def dphi_sign(z):                                      # sign of phi'(mu): beta_1^2 / x^3 vs the rest
            ls_ = logs(z)
            left = 2.0 * lb[0] - 3.0 * ls_[0]
            right = -math.inf
            for i in (1, 2):
                if lb[i] > -math.inf:
                    right = _lae(right, 2.0 * lb[i] - 3.0 * ls_[i])
            return right - left                                # > 0: phi increasing
        zl, zh = LO, -LO
        if dphi_sign(zl) < 0.0 < dphi_sign(zh):
            zm = brentq(dphi_sign, zl, zh, xtol=_EPS, rtol=4.0 * _EPS, maxiter=500)
            inner = []
            if Fi(zm) < 0.0:
                for a_, b_ in ((zl, zm), (zm, zh)):
                    if Fi(a_) * Fi(b_) < 0.0:
                        z_ = brentq(Fi, a_, b_, xtol=_EPS, rtol=4.0 * _EPS, maxiter=500)
                        x1 = dl[1] / (1.0 + math.exp(-z_)) if z_ > -700.0 else dl[1] * math.exp(z_)
                        r1 = dl[1] / (1.0 + math.exp(z_)) if z_ < 700.0 else dl[1] * math.exp(-z_)
                        den = np.array([-x1, r1, (dl[2] - dl[1]) + r1])
                        md = _mode_at(E @ (beta / den), lam[0] + x1, Sg, sg, G)
                        if md is not None:
                            inner.append(md)
            tol = 64.0 * _EPS * (scale + max((abs(md["mu"]) for md in inner), default=0.0)) * G * G
            for md in inner:
                if float(np.linalg.eigvalsh(md["Ht"])[0]) >= -tol:
                    out.append(md)
    out.sort(key=lambda md: md["q"])
    return out


def _circle_mins(a, b, c, f1, f2):
    """Per row: the LOCAL MINIMA of h(psi) = v^T M v - 2 f^T v on the unit circle v = (cos psi, sin psi), M = [[a, c],
    [c, b]], f = (f1, f2) (_gravity_posterior's conditional problem in units of G^2).  The trust-region theory of
    _sphere_modes in 2-D: the global minimum has multiplier mu < l1 (y = l1 - mu = the root of the secular equation
    B1^2 / y^2 + B2^2 / (D + y)^2 = 1, B = the eigen-coordinates of f, D = l2 - l1), the local non-global one (where it
    exists) l1 < mu < l2; the hard case (B1 = 0 to float64) has two symmetric minima at mu = l1.  Closed-form 2 x 2
    eigen-decomposition; roots by vectorised bisection on log / logit variables over the float64 exponent range (64
    halvings of the 1490-wide bracket: machine precision).  Returns (psi, h, h2, ok), each (n, 2) - slot 0 the global
    minimum (+ branch in the hard case), slot 1 the hard case's - branch or the local non-global minimum - with h2 =
    d^2 h / dpsi^2 = 2 (t^T M t - mu) (t = dv / dpsi), and hard (n,)."""
    a = np.asarray(a, float)
    b, c, f1, f2 = (np.broadcast_to(np.asarray(x, float), a.shape) for x in (b, c, f1, f2))
    n = a.size
    mh, dh = 0.5 * (a + b), 0.5 * (a - b)
    rr = np.hypot(dh, c)
    l1, Dl = mh - rr, 2.0 * rr
    ph = 0.5 * np.arctan2(2.0 * c, a - b)
    e2x, e2y = np.cos(ph), np.sin(ph)                            # eigenvector of l2 = l1 + Dl
    e1x, e1y = -e2y, e2x                                         # eigenvector of l1
    B1 = f1 * e1x + f2 * e1y
    B2 = f1 * e2x + f2 * e2y
    with np.errstate(divide="ignore"):
        lb1, lb2, lD = np.log(np.abs(B1)), np.log(np.abs(B2)), np.log(Dl)
    psi = np.full((n, 2), np.nan)
    hv = np.full((n, 2), np.inf)
    h2 = np.full((n, 2), np.nan)
    ok = np.zeros((n, 2), bool)

    def fill(slot, rows, vx, vy, mu):
        nv = np.hypot(vx, vy)
        vx, vy = vx / nv, vy / nv
        ar, br, cr = a[rows], b[rows], c[rows]
        psi[rows, slot] = np.arctan2(vy, vx)
        hv[rows, slot] = ar * vx * vx + br * vy * vy + 2.0 * cr * vx * vy - 2.0 * (f1[rows] * vx + f2[rows] * vy)
        tx, ty = -vy, vx
        h2[rows, slot] = 2.0 * (ar * tx * tx + br * ty * ty + 2.0 * cr * tx * ty - mu)
        ok[rows, slot] = np.isfinite(hv[rows, slot]) & np.isfinite(h2[rows, slot])

    def Fg(t, r):
        return np.logaddexp(2.0 * lb1[r] - 2.0 * t, 2.0 * lb2[r] - 2.0 * np.logaddexp(lD[r], t))
    allr = np.arange(n)
    hard = Fg(np.full(n, -745.0), allr) <= 0.0
    r = np.nonzero(~hard)[0]
    if r.size:
        lo, hi = np.full(r.size, -745.0), np.full(r.size, 745.0)
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            pos = Fg(mid, r) > 0.0
            lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
        y = np.exp(0.5 * (lo + hi))
        w1, w2 = B1[r] / y, B2[r] / (Dl[r] + y)
        fill(0, r, w1 * e1x[r] + w2 * e2x[r], w1 * e1y[r] + w2 * e2y[r], l1[r] - y)
    r = np.nonzero(hard)[0]
    if r.size:                                                   # two symmetric minima at mu = l1
        w2 = np.where(Dl[r] > 0.0, B2[r] / np.where(Dl[r] > 0.0, Dl[r], 1.0), 0.0)
        tau = np.sqrt(np.maximum(1.0 - w2 * w2, 0.0))
        for slot, sgn in ((0, 1.0), (1, -1.0)):
            fill(slot, r, sgn * tau * e1x[r] + w2 * e2x[r], sgn * tau * e1y[r] + w2 * e2y[r], l1[r])
    # the local non-global minimum: x = mu - l1 in (0, D), phi(x) = B1^2 / x^2 + B2^2 / (D - x)^2 convex, minimum at
    # logit(x / D) = (2 / 3) (log|B1| - log|B2|); two roots when phi there < 1, the one with h2 > 0 is the minimum
    zm = (2.0 / 3.0) * (lb1 - lb2)

    def Fi(z, rr_):
        lx = lD[rr_] - np.logaddexp(0.0, -z)
        l2_ = lD[rr_] - np.logaddexp(0.0, z)
        return np.logaddexp(2.0 * lb1[rr_] - 2.0 * lx, 2.0 * lb2[rr_] - 2.0 * l2_)
    cand = (~hard) & (Dl > 0.0) & np.isfinite(zm) & (np.abs(B1) > 0.0) & (np.abs(B2) > 0.0)
    rc = np.nonzero(cand)[0]
    if rc.size:
        ex = Fi(zm[rc], rc) < 0.0
        rc = rc[ex]
    if rc.size:
        best_h2 = np.full(rc.size, -np.inf)
        keep = None
        for side in (-1.0, 1.0):
            lo = np.where(side < 0, -745.0, zm[rc])
            hi = np.where(side < 0, zm[rc], 745.0)
            for _ in range(64):
                mid = 0.5 * (lo + hi)
                pos = Fi(mid, rc) > 0.0                          # outside the two roots
                if side < 0:
                    lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
                else:
                    lo, hi = np.where(pos, lo, mid), np.where(pos, mid, hi)
            z_ = 0.5 * (lo + hi)
            from scipy.special import expit
            x1 = Dl[rc] * expit(z_)
            r1 = Dl[rc] * expit(-z_)
            w1, w2 = B1[rc] / (-x1), B2[rc] / r1
            vx, vy = w1 * e1x[rc] + w2 * e2x[rc], w1 * e1y[rc] + w2 * e2y[rc]
            nv = np.hypot(vx, vy)
            vx, vy = vx / nv, vy / nv
            tx, ty = -vy, vx
            mu = l1[rc] + x1
            hh = 2.0 * (a[rc] * tx * tx + b[rc] * ty * ty + 2.0 * c[rc] * tx * ty - mu)
            better = hh > best_h2
            best_h2 = np.where(better, hh, best_h2)
            keep = (np.where(better, vx, keep[0]) if keep is not None else vx,
                    np.where(better, vy, keep[1]) if keep is not None else vy,
                    np.where(better, mu, keep[2]) if keep is not None else mu)
        mn = best_h2 > 0.0
        if mn.any():
            fill(1, rc[mn], keep[0][mn], keep[1][mn], keep[2][mn])
    return psi, hv, h2, ok, hard


def _gravity_posterior(Sg, sg, G, modes, fs, p=P_GATE, ba_map=None):
    """The posterior of the gravity DIRECTION at the fixed noise model, exact up to quadrature.  The nuisance
    y = (v, ba, bg) enters linearly and Gaussian, so marginalising it leaves a normaliser independent of g; with a
    uniform prior on the sphere
        p(u)  proportional to  exp(-q(G u) / (2 fs)),   q = _profile_g's chi2 (a Fisher-Bingham density on S^2),
    fs = the tilt block's Satterthwaite factor (the noise components are estimated).  This is what the local (Laplace)
    radius cannot see: a second local minimum (yaw-only motion: g and -g; one horizontal rotation axis: the two
    intersections of the weak line with the sphere) and a weak direction whose profile is not quadratic (the weak line
    crossing the sphere near its tangent point: quartic, much heavier tails than its Laplace curvature says).
    QUADRATURE: great-circle coordinates about the global mode m, u = cos psi (cos theta m + sin theta t) + sin psi n,
    t = the weakest tangent direction (displacement), n = m x t, theta in [-pi, pi], psi in [-pi/2, pi/2], area element
    cos psi.  theta on the union of a uniform grid (_POST_UNIF over the circle), _POST_DENSE points over +-_POST_K
    Laplace sd and a geometric grid (_POST_GEO points from 1e-7 rad to pi, scale-free) on both sides of every local
    minimum.  Per theta the conditional problem in psi is a 2-D trust-region problem on the circle (_circle_mins: its
    local minima exactly); the psi integral is Laplace at each minimum in the chart where its sd <= _POST_PSI_SD and it
    lies > _POST_K sd inside the chart, else quadrature over [-pi/2, pi/2] (_POST_PSI points); a row whose exact
    conditional minimum lies more than _LOG_TINY e-folds above the global minimum contributes exactly 0 in float64.
    ba_map = (ba0, Bba): the SUPPORT is restricted by the accelerometer-bias bound, |ba(G u)| = |ba0 - Bba G u| <= G
    (the prior is uniform on the feasible part of the sphere; checked per psi point in quadrature rows and at each
    conditional minimum in Laplace rows, where the psi-width (<= _POST_PSI_SD) moves ba by far less than the margin).
    `modes` are the FEASIBLE local minima, the one the posterior is centred on first.
    Returns dict(r = the P_GATE credible cap radius about m [rad] (inf when it exceeds pi/2), p_mode2 = the Laplace
    posterior probability of the best other local minimum, mode2_deg / mode2_dchi2 (its angle from m, its q - q_m),
    p_far = the exact posterior mass farther from m than half the angle to that mode, sd_weak_deg (Laplace), n_modes,
    n_theta, n_psi_quad, mu_G2 (the constraint curvature term at m))."""
    from scipy.special import expit, ndtr
    g1 = np.asarray(modes[0]["g"], float)
    m = g1 / np.linalg.norm(g1)
    Gm = -_hat(g1) @ modes[0]["B"]
    ev, U = np.linalg.eigh(modes[0]["Ht"])
    t = Gm @ U[:, 0]
    t /= np.linalg.norm(t)
    nn_ = np.cross(m, t)

    def sd_of(H):
        e0 = float(np.linalg.eigvalsh(H)[0])
        return min(math.sqrt(fs / e0), math.pi) if e0 > 0.0 else math.pi
    sdw = sd_of(modes[0]["Ht"])
    geo = np.geomspace(1e-7, math.pi, _POST_GEO)
    th = [np.linspace(-math.pi, math.pi, _POST_UNIF + 1), sdw * np.linspace(-_POST_K, _POST_K, _POST_DENSE),
          geo, -geo, np.zeros(1)]
    for md in modes[1:]:
        u2 = md["g"] / np.linalg.norm(md["g"])
        th2 = math.atan2(float(u2 @ t), float(u2 @ m))
        loc = np.concatenate([sd_of(md["Ht"]) * np.linspace(-_POST_K, _POST_K, _POST_DENSE), geo, -geo, [0.0]])
        th.append(np.mod(th2 + loc + math.pi, 2.0 * math.pi) - math.pi)
    th = np.unique(np.clip(np.concatenate(th), -math.pi, math.pi))
    Sm, St, Sn = Sg @ m, Sg @ t, Sg @ nn_
    c_, s_ = np.cos(th), np.sin(th)
    a = c_ * c_ * float(m @ Sm) + 2.0 * c_ * s_ * float(m @ St) + s_ * s_ * float(t @ St)
    cc = c_ * float(m @ Sn) + s_ * float(t @ Sn)
    b = float(nn_ @ Sn)
    d = c_ * float(sg @ m) + s_ * float(sg @ t)
    e = float(sg @ nn_)
    G2 = G * G
    # exact per-row LOWER bound on q over the whole circle: q >= G^2 lambda_min(M) - 2 G |f| (M = [[a, cc], [cc, b]],
    # f = (d, e)); a row more than _LOG_TINY e-folds above the solution's q (>= the feasible reference used below)
    # contributes exactly 0 in float64 and is not solved
    lmin = 0.5 * (a + b) - np.hypot(0.5 * (a - b), cc)
    lbnd = G2 * lmin - 2.0 * G * np.hypot(d, e)
    rows = np.nonzero((lbnd - float(modes[0]["q"])) / (2.0 * fs) <= _LOG_TINY)[0]
    psi = np.full((th.size, 2), np.nan)
    hv = np.full((th.size, 2), np.inf)
    h2 = np.full((th.size, 2), np.nan)
    ok = np.zeros((th.size, 2), bool)
    hard = np.zeros(th.size, bool)
    if rows.size:
        psi[rows], hv[rows], h2[rows], ok[rows], hard[rows] = _circle_mins(a[rows], b, cc[rows], d[rows] / G, e / G)
    qv, q2 = G2 * hv, G2 * h2                                   # conditional minima and curvatures in q units

    def feasible(th_, ps_):                                     # |ba(G u)| <= G at chart points (broadcast)
        if ba_map is None:
            return np.ones(np.broadcast(th_, ps_).shape, bool)
        cp, sp = np.cos(ps_), np.sin(ps_)
        u = (cp * np.cos(th_))[..., None] * m + (cp * np.sin(th_))[..., None] * t + sp[..., None] * nn_
        bb = ba_map[0] - G * (u @ ba_map[1].T)
        return np.einsum("...i,...i->...", bb, bb) <= G2
    lb = np.nanmin(np.where(ok, qv, np.inf), axis=1)            # the exact conditional minimum over the full circle
    inch = ok & (np.abs(psi) < 0.5 * math.pi)                   # minima inside this theta's half-circle chart
    if rows.size:                                               # ... and inside the bias bound (solved rows only)
        inch[rows] &= feasible(th[rows, None], np.where(ok[rows], psi[rows], 0.0))
    # the reference level: the lowest FEASIBLE value (an infeasible mirror below it must not underflow the support)
    qref = min(float(modes[0]["q"]), float(qv[inch].min()) if inch.any() else np.inf)
    sdc = np.sqrt(np.where(ok & (q2 > 0.0), 2.0 * fs / np.where(q2 > 0.0, q2, 1.0), np.inf))
    lapc = inch & (sdc <= _POST_PSI_SD) & (np.abs(psi) + _POST_K * sdc < 0.5 * math.pi)
    # a minimum OUTSIDE the chart but within _POST_K sd of its edge reaches into this chart (near the poles +-n)
    near = ok & ~inch & ((np.abs(psi) - 0.5 * math.pi) < _POST_K * np.where(np.isfinite(sdc), sdc, math.pi))
    live = np.isfinite(lb) & ((lb - qref) / (2.0 * fs) <= _LOG_TINY)
    lap_row = live & ~hard & inch.any(axis=1) & np.all(~inch | lapc, axis=1) & ~np.any(near, axis=1)
    iq = np.nonzero(live & ~lap_row)[0]
    pj = np.linspace(-0.5 * math.pi, 0.5 * math.pi, _POST_PSI)
    cumq, apj, Q, Fq = None, None, None, None
    if iq.size:                                                 # quadrature rows: their values first (qref)
        cp_, sp_ = np.cos(pj)[None, :], np.sin(pj)[None, :]
        Q = G2 * (a[iq, None] * cp_ * cp_ + b * sp_ * sp_ + 2.0 * cc[iq, None] * sp_ * cp_) \
            - 2.0 * G * (d[iq, None] * cp_ + e * sp_)
        Fq = feasible(th[iq, None], pj[None, :])
        if Fq.any():
            qref = min(qref, float(Q[Fq].min()))
    with np.errstate(over="ignore", under="ignore"):
        wc = np.where(lapc & lap_row[:, None],
                      np.exp(-(qv - qref) / (2.0 * fs)) * np.sqrt(2.0 * math.pi) * np.where(lapc, sdc, 0.0)
                      * np.cos(np.where(lapc, psi, 0.0)), 0.0)
    hden = wc.sum(axis=1)
    if iq.size:
        with np.errstate(over="ignore", under="ignore"):
            W = np.where(Fq, np.exp(-(Q - qref) / (2.0 * fs)), 0.0) * np.cos(pj)[None, :]
        hden[iq] = _trapz(W, pj, axis=1)
        c0 = (_POST_PSI - 1) // 2                               # the integrand folded onto psi >= 0 (f(0) = 2 W(0))
        dpj = pj[1] - pj[0]
        Wsym = np.concatenate([2.0 * W[:, c0:c0 + 1], W[:, c0 + 1:] + W[:, c0 - 1::-1]], axis=1)
        cumq = np.concatenate([np.zeros((iq.size, 1)), np.cumsum(0.5 * (Wsym[:, :-1] + Wsym[:, 1:]), axis=1) * dpj],
                              axis=1)                           # cumq[:, k] = the mass with |psi| <= apj[k]
        apj = pj[c0:]
    # the trapezoid rule on the (non-uniform) theta grid as weights tw_i = (th_i+1 - th_i-1) / 2 (half-steps at the
    # ends): Z = sum tw_i hden_i, and the cap mass sums only the rows of non-zero density (the others add exactly 0)
    tw = np.zeros_like(th)
    dth = np.diff(th)
    tw[:-1] += 0.5 * dth
    tw[1:] += 0.5 * dth
    Z = float(np.dot(tw, hden))
    il = np.nonzero(lap_row & (hden > 0.0))[0]
    qpos = {int(i): j for j, i in enumerate(iq)}
    iqn = np.array([i for i in iq if hden[i] > 0.0], dtype=int)
    iqr = np.array([qpos[int(i)] for i in iqn], dtype=int)
    thl, thq = th[il], th[iqn]
    psl, sdl = psi[il], np.where(lapc[il], sdc[il], 1.0)
    wcl, lcl = wc[il], lapc[il]

    def cap(r):
        cr = math.cos(r)
        tot = 0.0
        ins = np.abs(thl) < r
        if ins.any():
            pm = np.arccos(np.clip(cr / np.cos(thl[ins]), -1.0, 1.0))[:, None]
            ps, sd_ = psl[ins], sdl[ins]
            frac = ndtr((pm - ps) / sd_) - ndtr((-pm - ps) / sd_)
            tot += float(np.dot(tw[il][ins], np.sum(wcl[ins] * np.where(lcl[ins], frac, 0.0), axis=1)))
        if iqn.size:
            qi = np.abs(thq) < r
            if qi.any():
                pmq = np.arccos(np.clip(cr / np.cos(thq[qi]), -1.0, 1.0))
                rr = iqr[qi]
                k = np.clip(np.searchsorted(apj, pmq, side="right") - 1, 0, len(apj) - 2)
                fr = np.clip((pmq - apj[k]) / (apj[k + 1] - apj[k]), 0.0, 1.0)
                tot += float(np.dot(tw[iqn][qi], cumq[rr, k] + fr * (cumq[rr, k + 1] - cumq[rr, k])))
        return tot
    r = math.inf
    if Z > 0.0 and np.isfinite(Z) and cap(0.5 * math.pi) >= p * Z:
        lo, hi = 0.0, 0.5 * math.pi
        for _ in range(52):                                    # the float64 mantissa: the radius to ~1e-16 rad
            mid = 0.5 * (lo + hi)
            if cap(mid) >= p * Z:
                hi = mid
            else:
                lo = mid
        r = hi
    pm2, ang2, dq2, pfar = 0.0, None, None, 0.0
    if len(modes) > 1 and Z > 0.0:
        d1 = float(np.linalg.det(modes[0]["Ht"]))
        best = None
        for md in modes[1:]:
            d2 = float(np.linalg.det(md["Ht"]))
            lr = (-(md["q"] - modes[0]["q"]) / (2.0 * fs) + 0.5 * (math.log(d1) - math.log(d2))
                  if (d1 > 0.0 and d2 > 0.0) else math.inf)
            if best is None or lr > best[0]:
                best = (lr, md)
        lr, md = best
        pm2 = float(expit(lr))
        ang2 = tilt_error_deg(modes[0]["g"], md["g"])
        dq2 = float(md["q"] - modes[0]["q"])
        pfar = max(0.0, 1.0 - cap(min(0.5 * math.pi, 0.5 * math.radians(ang2))) / Z)
    return {"r": r, "p_mode2": float(pm2), "mode2_deg": ang2, "mode2_dchi2": dq2, "p_far": float(pfar),
            "sd_weak_deg": math.degrees(sdw), "n_modes": len(modes), "n_theta": int(th.size),
            "n_psi_quad": int(iq.size), "mu_G2": float(modes[0]["mu"]) * G2}


def _solve_constrained(S, L, prior_rows, g0, g_mag, max_iter=MAX_GN):
    """Whitened LS with |g| = g_mag, solved GLOBALLY at the fixed noise model: the profile chi2(g) (_profile_g) on the
    sphere has its global minimum found exactly (_sphere_modes; the start g0 plays no part - a local Gauss-Newton
    from the free-LS direction returned whichever of two equal minima that direction picked: verifier 2026-10-24,
    yaw-only gravity sign).  The solution is the lowest local minimum that satisfies the accelerometer-bias bound
    |ba(g)| <= g_mag (Class D; module docstring), or the global one when none does (glob['ba_ok'] False: the caller
    flags cause 'bias').  Returns (x_full, x_c, N_c, cf, Ac, bc, Aw, bw, B, iters, conv, glob): x_full has the
    ambient g, x_c = [v, d = 0, ba, bg]; N_c is the Gauss-Newton normal matrix of the tangent layout (the linear
    model: REML, snooping, leverage use it); glob = {Sg, sg, c0, modes (the feasible local minima, the solution
    first), modes_all, ba_ok, ba_map = (ba0, Bba) with ba(g) = ba0 - Bba g, mu, N_ex, cf_ex} with N_ex = N_c plus the
    constraint's curvature -mu g_mag^2 on the tangent block (the exact local Hessian: the parameter covariance), cf_ex
    None when N_ex is not positive definite.  Falls back to the local tangent Gauss-Newton (glob None) when the
    profile cannot be formed."""
    nv, G0 = S.nv, S.G0
    prior_free = [(c + 1, Pi, mu) for (c, Pi, mu) in prior_rows]
    modes = []
    try:
        Sg, sg, c0, aux = _profile_g(S, L, prior_free)
        if np.all(np.isfinite(Sg)) and np.all(np.isfinite(sg)):
            modes = _sphere_modes(Sg, sg, g_mag)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        modes = []
    if not modes:
        return _solve_constrained_gn(S, L, prior_rows, g0, g_mag, max_iter) + (None,)
    # THE ACCELEROMETER-BIAS BOUND |ba| <= g_mag (Class D; module docstring): ba(g) = y(g)[ba] is linear in g; a local
    # minimum that needs |ba| > g_mag is not a functioning accelerometer (the yaw-only mirror -g needs |ba| ~ 2 g_mag)
    ba0_, Bba_ = aux["X"][nv:nv + 3, 3].copy(), aux["X"][nv:nv + 3, :3].copy()
    for md in modes:
        md["ba"] = ba0_ - Bba_ @ md["g"]
        md["ba_ok"] = bool(np.linalg.norm(md["ba"]) <= g_mag)
    feas = [md for md in modes if md["ba_ok"]]
    sel = feas[0] if feas else modes[0]
    g = sel["g"]
    B = sel["B"]
    Gm = -_hat(g) @ B
    Aw_f, bw_f = aux["Aw"], aux["bw"]
    Ac = np.concatenate([S.A[:, :nv], S.A[:, G0:G0 + 3] @ Gm, S.A[:, G0 + 3:]], axis=1)
    bc = S.b - S.A[:, G0:G0 + 3] @ g
    Aw = np.concatenate([Aw_f[:, :nv], Aw_f[:, G0:G0 + 3] @ Gm, Aw_f[:, G0 + 3:]], axis=1)
    bw = bw_f - Aw_f[:, G0:G0 + 3] @ g
    N = Aw.T @ Aw
    for (cols, Pi, mu) in prior_rows:
        N[np.ix_(cols, cols)] += Pi
    cf = _Chol(N)
    y = aux["X"][:, 3] - aux["X"][:, :3] @ g                  # [v (nv), ba, bg] at the global g (exact)
    xc = np.concatenate([y[:nv], np.zeros(2), y[nv:]])
    x = np.concatenate([y[:nv], g, y[nv:]])
    Nex = N.copy()
    Nex[nv:nv + 2, nv:nv + 2] -= sel["mu"] * g_mag * g_mag * np.eye(2)
    try:
        cf_ex = _Chol(Nex)
    except np.linalg.LinAlgError:
        cf_ex = None
    glob = {"Sg": Sg, "sg": sg, "c0": c0, "modes": feas if feas else [sel], "modes_all": modes, "ba_ok": bool(feas),
            "ba_map": (ba0_, Bba_), "mu": sel["mu"], "N_ex": Nex, "cf_ex": cf_ex}
    return x, xc, N, cf, Ac, bc, Aw, bw, B, 1, True, glob


def _solve_constrained_gn(S, L, prior_rows, g0, g_mag, max_iter=MAX_GN):
    """FALLBACK (the previous solver, kept for a profile that cannot be formed): whitened LS with |g| = g_mag by
    Gauss-Newton on the 2-DoF tangent d (g = Exp(B d) g0), a LOCAL method.  Returns (x_full, x_c, N_c, cf, Ac, bc, Aw,
    bw, B, iters, conv)."""
    nv, G0 = S.nv, S.G0
    g = g0 / np.linalg.norm(g0) * g_mag
    conv = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        B = _basis(g)
        Gm = -_hat(g) @ B                                      # dg/dd at d = 0
        Ac = np.concatenate([S.A[:, :nv], S.A[:, G0:G0 + 3] @ Gm, S.A[:, G0 + 3:]], axis=1)
        bc = S.b - S.A[:, G0:G0 + 3] @ g
        Aw = solve_triangular(L, Ac, lower=True)
        bw = solve_triangular(L, bc, lower=True)
        N = Aw.T @ Aw
        h = Aw.T @ bw
        for (cols, Pi, mu) in prior_rows:                     # prior on absolute ba / bg (cols in the c-layout)
            N[np.ix_(cols, cols)] += Pi
            h[cols] += Pi @ mu
        cf = _Chol(N)
        xc = cf.solve(h)
        d = xc[nv:nv + 2]
        g = so3_exp(B @ d).reshape(3, 3) @ g
        if float(d @ d) < _EPS:                               # tangent step below sqrt(eps) rad
            conv = True
            break
    # final linearisation at the converged g (d = 0)
    B = _basis(g)
    Gm = -_hat(g) @ B
    Ac = np.concatenate([S.A[:, :nv], S.A[:, G0:G0 + 3] @ Gm, S.A[:, G0 + 3:]], axis=1)
    bc = S.b - S.A[:, G0:G0 + 3] @ g
    Aw = solve_triangular(L, Ac, lower=True)
    bw = solve_triangular(L, bc, lower=True)
    N = Aw.T @ Aw
    h = Aw.T @ bw
    for (cols, Pi, mu) in prior_rows:
        N[np.ix_(cols, cols)] += Pi
        h[cols] += Pi @ mu
    cf = _Chol(N)
    xc = cf.solve(h)
    xc[nv:nv + 2] = 0.0                                       # the tangent step is absorbed into g (converged)
    x = np.concatenate([xc[:nv], g, xc[nv + 2:]])
    return x, xc, N, cf, Ac, bc, Aw, bw, B, it, conv


def stereo_vi_init(T_wb, preints, g_mag=G_STD, bg_prior=None, bg_prior_cov=None, ba_prior=None, ba_prior_cov=None,
                   pose_sigma=None, p=P_GATE, tilt_bar_deg=TILT_BAR_DEG, snoop=True, gyro=None, noise=None,
                   att_rw=0.0):
    """Deliverable 2.  g_w (3, in w), per-KF v_w (n+1, 3), ba, bg and their covariance, from the metric VO keyframe
    poses T_wb (k = 0..n; 4x4 or (R, p)) and the records `preints` (n, between consecutive KFs; re-linearised at the
    gyro alignment's bias).  See the module docstring for the model, the noise model, the snooping, the observability
    condition and the degeneracy flag.

    bg_prior / bg_prior_cov: the gyro alignment's estimate (the rotation rows' information enters as this prior).
    ba_prior / ba_prior_cov: optional (none is declared on any dev rig; DESIGN_P1 4.2's turn-on bias is not in any calib).
    pose_sigma: optional (sigma_p [m], sigma_th [rad]) of the KF pose jitter; an entry that is not None is held fixed,
    a None entry is estimated (REML).  init_candidate passes (None, the gyro alignment's sigma_th): the rotation
    residuals identify the attitude jitter far better than the velocity rows do.
    gyro: the GyroAlign (optional): its per-factor rotation chi / leverage complete the U7 9-row chi2.
    noise: the ImuNoise (Class A) for the cross-interval bias random walk; with it ba / bg are the values at the newest
    KF; without it (None) they are window constants and the walk is NOT modelled (stated; on EuRoC's declared rw_a
    that makes the tilt over-confident, test a2).
    att_rw: the VO ATTITUDE RANDOM WALK scale (fixed): per KF interval l the VO relative rotation is allowed an extra
    isotropic increment of variance att_rw tr(C_RR,l) / 3, accumulated from KF 0.  init_candidate passes
    (kappa_g - 1)_+ from gyro_bias_align: the part of the measured VO-vs-gyro rotation disagreement beyond the gyro's
    declared noise, attributed to the VO (white per-KF jitter alone could not represent a drifting VO attitude:
    madmax_E1 x1, KF 7-11 drift 2.5-3.3 deg against GT within 3 s, admitted with a 4.2 deg error before this term).
    It also enters the CERTIFIED tilt: the bar is on the roll/pitch the engine carries forward, the gravity direction
    in the NEWEST KF's body frame, so its attitude variance (the walk's Qc[n] + sigma_th^2, isotropic) is added to
    the 2-D tilt covariance of g_w (tilt_r_deg; tilt_r_gw_deg is g_w alone).
    Returns ViInit (ok False with `reason` when the system cannot be formed)."""
    R, P = _rot_stack(T_wb)
    n_kf = len(R)
    if len(preints) != n_kf - 1:
        raise ValueError("stereo_vi_init: %d records for %d keyframes" % (len(preints), n_kf))
    used0 = [k for k, q in enumerate(preints) if q is not None and q.ok]
    if len(used0) < 3:
        return _empty_init("fewer than 3 usable intervals")
    # the excitation needs only the rotations and a gravity direction: the seed -mean(R_k dv_k / dt_k) (the mean
    # specific force in w; exact when the window does not accelerate on average) serves when the fit cannot be formed
    g_seed = -np.mean([R[k] @ preints[k].dv / preints[k].dt for k in used0], axis=0)

    def bad(why, cause="no fit"):
        return _empty_init(why, R, used0, g_seed, cause)
    removed = []
    used = list(used0)
    while True:
        res = _fit(R, P, preints, used, n_kf, g_mag, bg_prior, bg_prior_cov, ba_prior, ba_prior_cov, pose_sigma,
                   noise, att_rw)
        if res is None:
            return bad("normal matrix not positive definite (rank-deficient: gravity / ba not separable)", "rank")
        if not snoop or not res["conv_vce"] or len(removed) >= len(used0) // 2:
            break                                           # an unconverged fit is not snooped (it defers anyway)
        w = res["w_std"]
        m = len(used)
        thr = chi2_ppf(1.0 - (1.0 - p) / m, 6)
        kmax = int(np.argmax(w))
        if not (w[kmax] > thr) or res["dof"] - 3 <= 0:
            break
        removed.append(used[kmax])
        used = [k for k in used if k != used[kmax]]
        if len(used) < 3:
            return bad("snooping removed too many intervals")
    return _finish(res, R, P, preints, used, removed, n_kf, g_mag, p, tilt_bar_deg, gyro)


def att_chain(R_wb, preints, gyro, g_w, p=P_GATE):
    """The newest KF's VO attitude against the GYRO CHAIN.  The rotation residual of interval k (records at the
    aligned bias) r_k = Log(dR_k^T R_k^T R_k+1) is the VO-vs-gyro disagreement over that interval, expressed in KF
    k+1's body; carried to the newest KF n: D_n = sum_k (R_n^T R_k+1) r_k (first order) is how far the VO attitude of
    KF n has drifted from the gyro-propagated one.  Its covariance under the alignment's own noise model:
    sum_k M_k kappa_r C_RR,k M_k^T (kappa_r: the ROBUST per-interval disagreement scale, median of the alignment's
    per-factor chi) + J cov_bg J^T (the common
    bias, J = sum_k M_k (-JRg_k)) + 2 sigma_th^2 I (jitter at both ends).  Only the TILT part matters for the init
    (yaw does not change the gravity direction in the body): the projection on the plane perpendicular to the gravity
    direction in KF n's body, chi2_2.  A drift CONCENTRATED in a few intervals (madmax_E1 x1: 2.5-3.3 deg in 3 s) is
    what a homogeneous scale cannot represent; this test sees it directly.
    Returns dict(deg = |tilt part of D_n|, chi2, ok = chi2 <= chi2_2(P_GATE)) or None when the chain is broken (a
    refused record) or there is no gyro alignment."""
    R = _rot_only(R_wb)
    n = len(R) - 1
    if gyro is None or any(q is None or not q.ok for q in preints):
        return None
    # the per-interval disagreement scale must not be estimable by the drift under test: the REML kappa_g absorbs a
    # concentrated drift (test a10: sd 1.23 deg, a 1.7 deg drift passed); the MEDIAN of the per-factor chi (kappa_imu,
    # 50 % breakdown) is not moved by a few drifting intervals
    kr = kappa_imu(gyro.chi_k, np.maximum(3.0 - np.asarray(gyro.leverage, float), _SQEPS), p)
    kg = float(kr.kappa) if kr is not None else 1.0
    D = np.zeros(3)
    Cv = np.zeros((3, 3))
    Jb = np.zeros((3, 3))
    for k, q in enumerate(preints):
        dbg = gyro.bg - q.bg_lin
        dRc = q.dR @ np.asarray(so3_exp(q.JRg @ dbg)).reshape(3, 3)
        r = np.asarray(so3_log(dRc.T @ R[k].T @ R[k + 1])).reshape(3)
        M = R[n].T @ R[k + 1]
        D += M @ r
        Cv += M @ (kg * q.cov[0:3, 0:3]) @ M.T
        Jb += M @ (-q.JRg)
    Cv += Jb @ gyro.cov @ Jb.T + 2.0 * gyro.sigma_th ** 2 * np.eye(3)
    gb = R[n].T @ np.asarray(g_w, float)
    B = _basis(gb)
    z = B.T @ D
    c = float(z @ np.linalg.solve(B.T @ Cv @ B, z))
    return {"deg": float(np.degrees(np.linalg.norm(z))), "chi2": c, "ok": bool(c <= chi2_ppf(p, 2)),
            "sd_deg": float(np.degrees(math.sqrt(max(np.linalg.eigvalsh(B.T @ Cv @ B).max(), 0.0))))}


def _excitation(R, used, g):
    """sqrt(lambda_min(B^T M B) / m), M = sum_k (R_k - R_0)(R_k - R_0)^T over the KFs the used intervals touch, B the
    gravity tangent: 0 when the window holds no rotation that moves a direction perpendicular to g (module docstring);
    and the largest rotation from the first KF [deg]."""
    idx = sorted(set([k for k in used] + [k + 1 for k in used]))
    Rr = R[idx]
    Mx = np.einsum("nij,nkj->ik", Rr - Rr[0], Rr - Rr[0])
    Bg = _basis(g)
    exc = float(math.sqrt(max(np.linalg.eigvalsh(Bg.T @ Mx @ Bg).min(), 0.0) / max(len(idx) - 1, 1)))
    span = float(np.degrees(max(np.linalg.norm(so3_log(Rr[0].T @ Rk)) for Rk in Rr)))
    return exc, span


def _empty_init(reason, R=None, used=None, g_seed=None, cause="no fit"):
    vals = {f: None for f in ViInit._fields}
    vals.update(ok=False, reason=reason, degenerate=True, degenerate_cause=cause, rank_deficient=True)
    if R is not None and used and g_seed is not None and np.all(np.isfinite(g_seed)) and np.linalg.norm(g_seed) > 0:
        vals["excitation"], vals["rot_span_deg"] = _excitation(R, used, g_seed)
    return ViInit(**vals)


def _fit(R, P, pres, used, n_kf, g_mag, bg_prior, bg_prior_cov, ba_prior, ba_prior_cov, pose_sigma, noise=None,
         att_rw=0.0):
    S = _Sys(R, P, pres, used, n_kf, noise)
    S.set_att_rw(float(att_rw or 0.0))
    m = S.m
    nv = S.nv
    # priors in the constrained layout [v (nv), d (2), ba (3), bg (3)]
    prior_rows = []
    if bg_prior is not None and bg_prior_cov is not None:
        prior_rows.append((np.arange(nv + 5, nv + 8), np.linalg.inv(np.asarray(bg_prior_cov, float)),
                           np.asarray(bg_prior, float).reshape(3)))
    if ba_prior is not None and ba_prior_cov is not None:
        prior_rows.append((np.arange(nv + 2, nv + 5), np.linalg.inv(np.asarray(ba_prior_cov, float)),
                           np.asarray(ba_prior, float).reshape(3)))
    prior_free = [(c + 1, Pi, mu) for (c, Pi, mu) in prior_rows]   # free layout [v, g(3), ba, bg]: shift by 1
    # component start values (comparable to the IMU's own per-interval scale) or the engine's fixed jitter
    tr_pp = np.median([np.trace(pres[k].cov[6:9, 6:9]) / 3.0 for k in used])
    tr_rr = np.median([np.trace(pres[k].cov[0:3, 0:3]) / 3.0 for k in used])
    ps = (None, None) if pose_sigma is None else tuple(pose_sigma)
    fixed = np.array([False, ps[0] is not None, ps[1] is not None])
    s = np.array([1.0, tr_pp if ps[0] is None else float(ps[0]) ** 2, tr_rr if ps[1] is None else float(ps[1]) ** 2])
    s_init = s.copy()
    # starting gravity: free LS with the IMU covariance alone, then the components
    g0 = None
    conv_vce = False
    it_vce = 0
    x = None
    for it_vce in range(1, MAX_VCE + 1):
        Jth = S.Jth(x) if x is not None else None
        C = s[0] * S.Qimu + s[1] * (S.Jp @ S.Jp.T)
        Ul = [S.Uimu, S.Jp]
        U0 = None
        if Jth is not None:
            C = C + s[2] * (Jth @ Jth.T)
            Ul.append(Jth)
            if S.att_on:
                U0 = S.Ua(Jth)
                C = C + U0 @ U0.T
        try:
            L = _chol_lower(C)
        except np.linalg.LinAlgError:
            return None
        if g0 is None:
            # free solve for the starting direction
            Aw = solve_triangular(L, S.A, lower=True)
            bw = solve_triangular(L, S.b, lower=True)
            N = Aw.T @ Aw
            h = Aw.T @ bw
            for (cols, Pi, mu) in prior_free:
                N[np.ix_(cols, cols)] += Pi
                h[cols] += Pi @ mu
            try:
                xf = np.linalg.solve(N, h)
            except np.linalg.LinAlgError:
                xf = np.linalg.lstsq(N, h, rcond=None)[0]
            g0 = xf[S.G0:S.G0 + 3]
            if not np.all(np.isfinite(g0)) or np.linalg.norm(g0) < _SQEPS:
                # seed: minus the mean specific force in w (the rows' own dv / dt), exact when the window does not
                # accelerate on average; only the DIRECTION seeds the GN
                g0 = -np.mean([R[k] @ pres[k].dv / pres[k].dt for k in used], axis=0)
        try:
            x, xc, N, cf, Ac, bc, Aw, bw, B, it_gn, conv_gn, glob = _solve_constrained(
                S, L, prior_rows, g0, g_mag, max_iter=MAX_GN if Jth is None else 2)
        except np.linalg.LinAlgError:
            return None
        g0 = x[S.G0:S.G0 + 3]
        if Jth is None:
            if _rank_deficient(N):                              # the rank test first: a numerically singular normal
                return None                                     # matrix is not iterated on (Class D, float64)
            continue                                            # first pass only seeds x for the attitude Jacobian
        e = Ac @ xc - bc                                        # data residual (prediction - measurement)
        ew = solve_triangular(L, e, lower=True)
        new, sd = _reml_step(L, Ul, ew, Aw, cf, s, fixed, s_init, U0)
        tol = _VCE_FRAC * sd
        ch = np.abs(new - s)
        s = new
        if np.all(ch <= tol):
            conv_vce = True
            break
    # final pass at the converged components
    Jth = S.Jth(x)
    Ul = [S.Uimu, S.Jp, Jth]
    C = s[0] * S.Qimu + s[1] * (S.Jp @ S.Jp.T) + s[2] * (Jth @ Jth.T)
    U0 = S.Ua(Jth) if S.att_on else None
    if U0 is not None:
        C = C + U0 @ U0.T
    try:
        L = _chol_lower(C)
        x, xc, N, cf, Ac, bc, Aw, bw, B, it_gn, conv_gn, glob = _solve_constrained(S, L, prior_rows, g0, g_mag)
    except np.linalg.LinAlgError:
        return None
    e = Ac @ xc - bc
    ew = solve_triangular(L, e, lower=True)
    # raw (unfloored) single-component estimates s_g num_g / r_g at the converged point, and the REML standard errors
    comp_raw = []
    for gi, U in enumerate(Ul):
        num, r_g = _vce_terms(L, U, ew, Aw, cf, s[gi])
        comp_raw.append(s[gi] * num / r_g if r_g > _SQEPS else float("nan"))
    _s_unused, sdv = _reml_step(L, Ul, ew, Aw, cf, s, fixed, s_init, U0)
    comp_sd = [float(v) for v in sdv]
    chi2 = float(ew @ ew)
    for (cols, Pi, mu) in prior_rows:
        d = xc[cols] - mu
        chi2 += float(d @ Pi @ d)
    nrow = 6 * m + 3 * len(prior_rows)
    dof = nrow - (S.nv + 8)
    # the free-|g| solution with the same covariance (diagnostic + magnitude test)
    Awf = solve_triangular(L, S.A, lower=True)
    bwf = solve_triangular(L, S.b, lower=True)
    Nf = Awf.T @ Awf
    hf = Awf.T @ bwf
    for (cols, Pi, mu) in prior_free:
        Nf[np.ix_(cols, cols)] += Pi
        hf[cols] += Pi @ mu
    try:
        Nfi = np.linalg.inv(Nf)
        xf = Nfi @ hf
        rf = Awf @ xf - bwf
        chi2f = float(rf @ rf)
        for (cols, Pi, mu) in prior_free:
            d = xf[cols] - mu
            chi2f += float(d @ Pi @ d)
        gf = xf[S.G0:S.G0 + 3]
        u = gf / max(np.linalg.norm(gf), _EPS)
        gfs = float(math.sqrt(max(u @ Nfi[S.G0:S.G0 + 3, S.G0:S.G0 + 3] @ u, 0.0)))
    except np.linalg.LinAlgError:
        gf, chi2f, gfs = np.full(3, np.nan), float("nan"), float("nan")
    # standardised interval residuals (snooping): Cov(e) = C - A N^-1 A^T, per interval 6x6
    Ninv = cf.solve(np.eye(N.shape[0]))       # plug-in of the linear model (snooping, leverage)
    # the parameter covariance: the exact local Hessian (Gauss-Newton + the constraint's curvature on the tangent block)
    Nexi = glob["cf_ex"].solve(np.eye(N.shape[0])) if (glob is not None and glob.get("cf_ex") is not None) else Ninv
    blocks = {"v": np.arange(0, nv), "tilt": np.arange(nv, nv + 2), "ba": np.arange(nv + 2, nv + 5),
              "bg": np.arange(nv + 5, nv + 8)}
    satt = _satterthwaite(Nexi, L, Ul, Aw, cf, s, fixed, s_init, blocks, U0)
    Phi = _scale_blocks(Nexi, [(blocks[k], satt[k][0]) for k in blocks])
    w_std = np.zeros(m)
    for r in range(m):
        sl = slice(6 * r, 6 * r + 6)
        Ce = C[sl, sl] - Ac[sl] @ Ninv @ Ac[sl].T
        ev, evec = np.linalg.eigh(0.5 * (Ce + Ce.T))
        keep = ev > 6 * _EPS * max(ev.max(), _EPS)
        z = evec[:, keep].T @ e[sl]
        w_std[r] = float(np.sum(z * z / ev[keep]))
    # leverage of each interval's 6 rows (for the per-factor redundancy of the U7 chi2)
    lev6 = np.array([float(np.sum(Aw[6 * r:6 * r + 6] * cf.solve(Aw[6 * r:6 * r + 6].T).T)) for r in range(m)])
    return dict(S=S, x=x, xc=xc, N=N, Ninv=Phi, satt=satt, B=B, C=C, e=e, att_Qc=S.att_Qc, chi2=chi2, dof=dof, s=s, comp_raw=comp_raw,
                comp_sd=comp_sd, it_vce=it_vce, conv_vce=conv_vce, conv_gn=conv_gn, gf=gf, chi2f=chi2f, gfs=gfs,
                w_std=w_std, lev6=lev6, n_prior=len(prior_rows), glob=glob)


def _reml_step(L, Ul, ew, Aw, cf, s, fixed, s_init, U0=None):
    """One iterated-MINQUE / REML Fisher-scoring step for C = sum_g s_g U_g U_g^T (Rao 1971; Koch 1986):
    solve F s = q, F_gh = tr(P Q_g P Q_h) = ||U_g^T P U_h||_F^2, q_g = e^T W Q_g W e = ||Uw_g^T ew||^2, with
    P = W - W A N^-1 A^T W.  Converges in a few steps (quadratically near the solution).  kappa is held at its floor
    1 when the step would put it below (the rule), the jitter components at the numerical floor when non-positive (an
    active set, the remaining components re-solved); `fixed` (bool mask) holds components (engine-supplied sigma).
    U0: a FIXED component C_0 = U0 U0^T (known scale): the equations become F s = q - c, c_g = tr(P Q_g P C_0).
    Returns (s_new, sd) with sd = sqrt(diag(2 F^-1)) the components' standard errors (Gaussian REML)."""
    Uw = [solve_triangular(L, U, lower=True) for U in Ul]
    AtU = [Aw.T @ u for u in Uw]
    NiAtU = [cf.solve(a) for a in AtU]
    G = len(Ul)
    Fm = np.zeros((G, G))
    q = np.zeros(G)
    for g_ in range(G):
        q[g_] = float(np.sum((Uw[g_].T @ ew) ** 2))
        for h_ in range(g_, G):
            M = Uw[g_].T @ Uw[h_] - AtU[g_].T @ NiAtU[h_]
            Fm[g_, h_] = Fm[h_, g_] = float(np.sum(M * M))
    if U0 is not None:
        U0w = solve_triangular(L, U0, lower=True)
        Ni0 = cf.solve(Aw.T @ U0w)
        for g_ in range(G):
            M = Uw[g_].T @ U0w - AtU[g_].T @ Ni0
            q[g_] -= float(np.sum(M * M))
    floor = np.concatenate([[1.0], np.asarray(s_init[1:], float) * _SQEPS])
    free = ~np.asarray(fixed, bool)
    out = s.copy()
    for _ in range(G + 1):                                      # active-set passes
        fx = ~free
        rhs = q[free] - Fm[np.ix_(free, fx)] @ out[fx]
        try:
            sol = np.linalg.solve(Fm[np.ix_(free, free)], rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(Fm[np.ix_(free, free)], rhs, rcond=None)[0]
        out[free] = sol
        low = free & (out < floor)
        if not low.any():
            break
        out[low] = floor[low]
        free &= ~low
    try:
        sd = np.sqrt(np.maximum(np.diag(2.0 * np.linalg.inv(Fm)), 0.0))
    except np.linalg.LinAlgError:
        sd = np.full(G, np.inf)
    return out, sd


def _satterthwaite(Phi, L, Ul, Aw, cf, s, fixed, s_init, blocks, U0=None):
    """Small-sample calibration of the plug-in covariance Phi = N^-1 when variance components are ESTIMATED
    (Satterthwaite 1946, as used for F tests in mixed models): per parameter block B, the plug-in variance tr(Phi_B)
    is itself an estimate with Var(tr Phi_B) ~ t^T Cov(s) t, t_g = d tr(Phi_B) / d s_g = ||Uw_g^T Aw Phi E_B||_F^2
    (dPhi / ds_g = Phi A^T W Q_g W A Phi), Cov(s) = 2 F^-1 over the free components (REML).  Effective dof
    nu_B = 2 tr(Phi_B)^2 / Var(tr Phi_B); the standardised error then follows F(d, nu_B), whose mean is nu_B / (nu_B - 2):
    that factor scales block B so its NEES is calibrated; nu_B <= 2 means the noise scale itself is undetermined
    (factor inf).  Components held fixed or at their floor (kappa = 1 by the rule) do not vary: factor 1.
    Measured: test a2 kappa 4 (kappa estimated): tilt / bg NEES per dof 1.46 / 1.55 with the plug-in.
    Returns {block name: (factor, nu)}."""
    G = len(Ul)
    floor = np.concatenate([[1.0], np.asarray(s_init[1:], float) * _SQEPS])
    free = (~np.asarray(fixed, bool)) & (np.asarray(s) > floor * (1.0 + _SQEPS))
    out = {k: (1.0, math.inf) for k in blocks}
    if not free.any():
        return out
    Uw = [solve_triangular(L, U, lower=True) for U in Ul]
    AtU = [Aw.T @ u for u in Uw]
    NiAtU = [cf.solve(a) for a in AtU]
    Fm = np.zeros((G, G))
    for g_ in range(G):
        for h_ in range(g_, G):
            M = Uw[g_].T @ Uw[h_] - AtU[g_].T @ NiAtU[h_]
            Fm[g_, h_] = Fm[h_, g_] = float(np.sum(M * M))
    idx = np.nonzero(free)[0]
    try:
        Cs = 2.0 * np.linalg.inv(Fm[np.ix_(idx, idx)])
    except np.linalg.LinAlgError:
        return {k: (math.inf, 0.0) for k in blocks}
    del U0                                                     # a fixed component has no sampling variance
    for name, cols in blocks.items():
        PhiB = Phi[:, cols]                                     # Phi E_B
        t = np.array([float(np.sum((AtU[g_].T @ PhiB) ** 2)) for g_ in idx])
        var = float(t @ Cs @ t)
        tr = float(np.trace(Phi[np.ix_(cols, cols)]))
        nu = 2.0 * tr * tr / var if var > 0 else math.inf
        out[name] = ((nu / (nu - 2.0)) if nu > 2.0 else math.inf, nu)
    return out


def _scale_blocks(Phi, sc):
    """Congruence D Phi D, D = diag(sqrt(factor)) per block (finite factors only): keeps Phi PSD and the
    correlations; an infinite factor is reported, not applied (the caller flags the scale as undetermined)."""
    d = np.ones(Phi.shape[0])
    for cols, f in sc:
        if math.isfinite(f):
            d[cols] = math.sqrt(f)
    return Phi * d[:, None] * d[None, :]


def _vce_terms(L, U, ew, Aw, cf, s_g):
    """Foerstner's terms for one component C_g = s_g U U^T: (num, r) with num = e^T W C_g W e and
    r = tr(W C_g R) = s_g (||L^-1 U||_F^2 - tr(Uw^T Aw N^-1 Aw^T Uw)); s_g num / r is the single-component estimate
    reported as the raw (unfloored) value; at the REML fixed point it equals s_g."""
    Uw = solve_triangular(L, U, lower=True)
    num = s_g * float(np.sum((Uw.T @ ew) ** 2))
    AtU = Aw.T @ Uw
    T = cf.solve(AtU)
    r = s_g * float(np.sum(Uw * Uw) - np.sum(AtU * T))
    return num, r


def _finish(res, R, P, pres, used, removed, n_kf, g_mag, p, tilt_bar_deg, gyro):
    S, x, Ninv, B = res["S"], res["x"], res["Ninv"], res["B"]
    nv = S.nv
    g = x[S.G0:S.G0 + 3]
    v = np.full((n_kf, 3), np.nan)                             # NaN for a KF no usable record touches
    for kf, c in S.vcol.items():
        v[kf] = x[c:c + 3]
    ba = x[S.G0 + 3:S.G0 + 6]
    bg = x[S.G0 + 6:S.G0 + 9]
    cov_x = 0.5 * (Ninv + Ninv.T)                              # constrained layout [v, d, ba, bg]
    cov_tilt = cov_x[nv:nv + 2, nv:nv + 2]
    Gm = -_hat(g) @ B
    cov_g = Gm @ cov_tilt @ Gm.T
    cov_ba = cov_x[nv + 2:nv + 5, nv + 2:nv + 5]
    cov_bg = cov_x[nv + 5:nv + 8, nv + 5:nv + 8]
    cov_v = np.stack([cov_x[S.vcol[k]:S.vcol[k] + 3, S.vcol[k]:S.vcol[k] + 3] if k in S.vcol
                      else np.diag([np.inf] * 3) for k in range(n_kf)])
    # gravity / ba block: Schur information of (tilt, ba) after eliminating v and bg
    P5 = cov_x[nv:nv + 5, nv:nv + 5]
    k2 = chi2_ppf(p, 2)
    lam_t = float(np.linalg.eigvalsh(cov_tilt).max())
    try:
        I5 = np.linalg.inv(P5)
        Pt_known = np.linalg.inv(I5[0:2, 0:2])                 # tilt covariance were ba known
        lam_k = float(np.linalg.eigvalsh(Pt_known).max())
        D = np.diag([1.0 / g_mag, 1.0 / g_mag, 1.0, 1.0, 1.0])  # tilt in m/s^2 (g_mag d) next to ba
        ev = np.linalg.eigvalsh(D @ I5 @ D)
        cond = float(ev.max() / ev.min()) if ev.min() > 0 else float("inf")   # diagnostic (units: m/s^2)
        rank_def = _rank_deficient(I5)
    except np.linalg.LinAlgError:
        I5, lam_k, cond, rank_def = None, float("inf"), float("inf"), True
    rank_def = rank_def or _rank_deficient(res["N"])
    lam_gw = lam_t
    vif = lam_gw / lam_k if lam_k > 0 else float("inf")          # g_w's tilt: ba free / ba known
    # the attitude variance of the NEWEST KF (walk from KF 0 + white jitter), isotropic: adds to both tilt directions
    att_n = float(res["att_Qc"][n_kf - 1]) + float(res["s"][2])
    lam_t = lam_t + att_n
    lam_k = lam_k + att_n
    fs_t = res["satt"]["tilt"][0]
    if not math.isfinite(fs_t):                             # Satterthwaite nu <= 2: the noise scale is undetermined
        lam_t = math.inf
    tilt_sigma = math.degrees(math.sqrt(max(lam_t, 0.0)))
    tilt_r_lap = math.degrees(math.sqrt(max(k2 * lam_gw, 0.0)))   # the LOCAL (Laplace) P_GATE radius of g_w
    att_r = math.degrees(math.sqrt(max(k2 * att_n, 0.0)))         # ... of the newest KF's VO attitude
    att_sd_n = math.degrees(math.sqrt(max(att_n, 0.0)))
    tilt_r_l = math.degrees(math.sqrt(max(k2 * lam_t, 0.0)))     # Laplace, newest KF (= sqrt(r_lap^2 + att_r^2))
    tilt_rk = math.degrees(math.sqrt(max(k2 * lam_k, 0.0))) if math.isfinite(lam_k) else float("inf")
    # magnitude test: chi2 increase of the |g| constraint vs chi2_1
    dchi = res["chi2"] - res["chi2f"]
    mag_ok = bool(np.isfinite(dchi) and dchi <= chi2_ppf(p, 1))
    # THE EXACT POSTERIOR of the gravity direction (_gravity_posterior), whenever the Laplace radius would certify:
    # the certified g_w radius is the larger of the two (the local radius cannot see a second minimum or a
    # non-quadratic weak direction; the exact one is never allowed to shrink the local one)
    gl = res.get("glob")
    post, tilt_r_ex = None, None
    if gl is not None and gl.get("modes") and tilt_r_l <= tilt_bar_deg:
        try:
            post = _gravity_posterior(gl["Sg"], gl["sg"], g_mag, gl["modes"], fs_t, p, gl.get("ba_map"))
            tilt_r_ex = math.degrees(post["r"])
            excl = [md for md in gl.get("modes_all", []) if not md.get("ba_ok", True)]
            post.update(n_excluded_by_bias_bound=len(excl),
                        excluded_min_ba=min((float(np.linalg.norm(md["ba"])) for md in excl), default=None),
                        excluded_dchi2=min((float(md["q"] - gl["modes"][0]["q"]) for md in excl), default=None),
                        excluded_deg=min((tilt_error_deg(md["g"], g) for md in excl), default=None))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, ZeroDivisionError) as exc_:
            post, tilt_r_ex = {"error": repr(exc_)}, math.inf
    tilt_r_gw = max(tilt_r_lap, tilt_r_ex if tilt_r_ex is not None else 0.0)
    tilt_r = math.sqrt(tilt_r_gw ** 2 + att_r ** 2) if math.isfinite(lam_t) else math.inf
    not_cert = not (tilt_r <= tilt_bar_deg)
    # a certification needs a VALID noise model: the REML fixed point reached and |g| = g_mag not rejected; and a
    # solution that a functioning accelerometer can produce (the bias bound)
    ba_ok = bool(gl.get("ba_ok", True)) if gl is not None else bool(np.linalg.norm(ba) <= g_mag)
    degenerate = rank_def or not_cert or (not res["conv_vce"]) or (not mag_ok) or (not ba_ok)
    cause = None
    if rank_def:
        cause = "rank"
    elif not ba_ok:
        cause = "bias"                                         # every local minimum needs |ba| > g_mag
    elif not res["conv_vce"]:
        cause = "unconverged"                                  # no REML fixed point: the covariance certifies nothing
    elif not mag_ok:
        cause = "model"                                        # the data reject |g| = g_mag: model / poses wrong
    elif not_cert:
        if not math.isfinite(fs_t):
            cause = "scale"                                    # the noise scale itself is undetermined (nu <= 2)
        elif post is not None and post.get("p_mode2", 0.0) > 1.0 - p:
            cause = "ambiguous"                                # a second local minimum carries > 1 - P_GATE
        elif tilt_r_ex is not None:
            cause = "weak"                                     # Laplace passed, the exact posterior did not
        elif att_r > tilt_bar_deg:
            cause = "attitude"                                 # the newest KF's VO attitude alone fails the bar
        else:
            cause = "excitation" if tilt_rk <= tilt_bar_deg else "noise"
    # excitation: chord of the relative rotations projected on the gravity tangent
    exc, rot_span = _excitation(R, used, g)
    # U7 9-row chi2 per factor at the solution, declared covariance (kappa 1), dof = 9 - leverage
    chi9, dof9 = [], []
    rot_lev = {}
    if gyro is not None:
        for kk, k in enumerate(gyro.used):
            rot_lev[k] = float(gyro.leverage[kk])
    for r, k in enumerate(used):
        q = pres[k]
        Xi = F.State(R[k], P[k], v[k], bg, ba)
        Xj = F.State(R[k + 1], P[k + 1], v[k + 1], bg, ba)
        r15 = F.imu_residual(Xi, Xj, q, g)
        c9 = q.cov[0:9, 0:9]
        try:
            L9 = np.linalg.cholesky(c9)
            z = solve_triangular(L9, r15[:9], lower=True)
            chi9.append(float(z @ z))
            dof9.append(max(9.0 - float(res["lev6"][r]) - rot_lev.get(k, 0.0), _SQEPS))
        except np.linalg.LinAlgError:
            pass
    kap_naive = kappa_imu(chi9, dof9, p) if chi9 else None          # U7 at FIXED VO poses: pose jitter included
    kap = kappa_from_component(res["comp_raw"][0], res["comp_sd"][0], p)
    info_tb = None if I5 is None else I5
    t_span = float(pres[used[-1]].t1 - pres[used[0]].t0)
    return ViInit(True, "ok", g, v, ba, bg, cov_x, cov_tilt, cov_g, cov_ba, cov_bg, cov_v, B,
                  tilt_sigma, tilt_r, tilt_rk, vif, cond, rank_def, degenerate, cause, exc, rot_span,
                  res["gf"], float(np.linalg.norm(res["gf"])), res["gfs"], dchi, mag_ok,
                  res["chi2"], res["dof"], res["chi2"] / max(res["dof"], 1), float(res["comp_raw"][0]),
                  float(math.sqrt(max(res["s"][1], 0.0))), float(math.sqrt(max(res["s"][2], 0.0))),
                  res["comp_sd"], res["it_vce"], res["conv_vce"], removed, len(used),
                  kap, np.array(chi9), np.array(dof9), kap_naive, info_tb, t_span, res["satt"],
                  float(pres[used[-1]].t1), tilt_r_gw, att_sd_n, att_chain(R, pres, gyro, g, p), len(S.touched),
                  tilt_r_laplace_deg=tilt_r_lap, tilt_r_exact_deg=tilt_r_ex, gpost=post)


# ================================================================================================ igt_k_R (3) =====
class _PreAdapter:
    """imu_factor.Preint seen as backend/preint.py Preintegrated (the fields rotation_consistency reads)."""
    __slots__ = ("dR", "dv", "dp", "dt", "cov", "J_R_bg", "J_v_bg", "J_v_ba", "J_p_bg", "J_p_ba", "bg", "ba", "n")

    def __init__(self, q):
        self.dR, self.dv, self.dp, self.dt = q.dR, q.dv, q.dp, q.dt
        self.cov = q.cov[0:9, 0:9]                       # [rR, rv, rp] = preint.py's [dphi, dv, dp] order
        self.J_R_bg, self.J_v_bg, self.J_v_ba = q.JRg, q.Jvg, q.Jva
        self.J_p_bg, self.J_p_ba = q.Jpg, q.Jpa
        self.bg, self.ba, self.n = q.bg_lin, q.ba_lin, q.n_sub


def igt_k_R(R_wb, preints, bg0, inflate=IGT_INFLATE, K=IGT_K):
    """The IGT rotation-consistency statistic, CAUSAL variant: backend/preint.py
    rotation_consistency called unchanged on the records (linearised at bg0 = the causal bootstrap bias), the factor
    list = records with >= 2 integration steps.  Returns its dict + n_lt2, K, refused (k_R > K), or None when no
    factor exists."""
    from xclean.backend.preint import rotation_consistency
    R = _rot_only(R_wb)
    ad = []
    n_lt2 = 0
    for q in preints:
        if q is None or not q.ok:
            ad.append(None)
        elif q.n_sub < 2:
            n_lt2 += 1
            ad.append(None)
        else:
            ad.append(_PreAdapter(q))
    rc = rotation_consistency(ad, R, np.asarray(bg0, float).reshape(3), inflate)
    if rc is None:
        return None
    rc = dict(rc)
    rc.update(n_lt2=n_lt2, K=float(K), refused=bool(rc["k_R"] > K), inflate=float(inflate))
    return rc


# ============================================================================================== admission (3) =====
Admission = namedtuple("Admission", "decision reasons stats")
Admission.__doc__ = ("decision: 'admit' | 'defer' (not yet: wait for more KFs / excitation) | 'refuse' (IMU refused "
                     "for the segment: the IGT gate); reasons: list of strings; stats: dict of the tested values")


def _agreement(prev, cur, p, noise=None):
    """Two consecutive candidates agree within their sigma (DESIGN_P1 6.2): chi2 of the (tilt, ba) difference between
    the candidate at the previous KF and the current one with Cov(d) = P_prev + P_cur (+ kappa rw_a^2 dt I on ba: the
    biases are the values at each candidate's newest KF, dt apart).  For two estimates from overlapping windows the
    true Cov(d) = P_prev + P_cur - 2 Cov(prev, cur) <= P_prev + P_cur (positive correlation), so the test is
    conservative: it rejects a gross disagreement (a bad interval, a different minimum), not sampling scatter.
    (The sharper nested form P_prev - P_cur assumes the same noise model in both fits; here the variance components
    are re-estimated per fit, and test a9 measured 13.8 % rejections of consistent pairs with it.)
    Returns (chi2, dof, ok, method)."""
    nvp = prev.cov_x.shape[0] - 8                              # layout [v (touched KFs), tilt (2), ba, bg]
    nvc = cur.cov_x.shape[0] - 8
    P5p = prev.cov_x[nvp:nvp + 5, nvp:nvp + 5]
    P5c = cur.cov_x[nvc:nvc + 5, nvc:nvc + 5]
    T = np.eye(5)
    T[0:2, 0:2] = cur.basis.T @ prev.basis               # previous tangent coordinates -> the current basis
    D = T @ P5p @ T.T + P5c
    if noise is not None and prev.t_anchor is not None and cur.t_anchor is not None:
        kap = cur.kappa.kappa if cur.kappa is not None else 1.0
        D[2:5, 2:5] += kap * noise.rw_a ** 2 * abs(cur.t_anchor - prev.t_anchor) * np.eye(3)
    d = np.concatenate([cur.basis.T @ _rotvec_between(cur.g_w, prev.g_w), prev.ba - cur.ba])
    c = float(d @ np.linalg.solve(0.5 * (D + D.T), d))
    return c, 5, bool(c <= chi2_ppf(p, 5)), "sum"


def admission(n_kf, kR, gyro, init, prev=None, K=IGT_K, p=P_GATE, min_kf=MIN_INIT_KF, noise=None, power=None,
              rule=None):
    """Deliverable 3: admit / defer / refuse, with reasons.

    rule (default ADMISSION_RULE): 'igt' = the IGT gate as DESIGN_P1 6.2 / U6 define it (below); 'pigt' = the POWERED
    rule: on an init window of >= MIN_INIT_KF KFs, REFUSE when k_R > K and
    the test is sized (power['alpha'] <= 1 - P_GATE); the IGT part PASSES when k_R <= K and the test is powered against
    a TD_HARM clock offset (power['power'] >= P_GATE) - final for the segment, carried in prev.igt_state; otherwise
    the IGT part is UNDECIDED and the candidate defers (VO mode).  `power` = igt_power(...) on the same records.
    Every other condition below is the same under both rules.

    REFUSE ('igt'; the IMU is refused for the segment; DESIGN_P1 6.2): the IGT statistic k_R (causal variant, igt_k_R) > K
    on an init window of >= MIN_INIT_KF KFs (below that, k_R > K defers: a refusal is final for the segment).
    DEFER (not yet; the engine tries again at the next KF; DESIGN_P1 6.2 'accept only if ...'):
      * fewer than MIN_INIT_KF keyframes, or no statistic / no fit;
      * the gravity / ba block is degenerate (rank) or the tilt is not certified to the bar at P_GATE (excitation or
        noise; stereo_vi_init);
      * the |g| = g_mag constraint is rejected by the data at P_GATE (dchi2 > chi2_1(P_GATE));
      * the gyro alignment or the variance-component iteration did not converge;
      * the newest KF's VO tilt disagrees with the gyro chain at P_GATE (att_chain: a concentrated VO attitude drift
        that would be carried forward as a wrong roll/pitch);
      * no previous candidate, or the previous candidate disagrees at P_GATE (two consecutive candidates must agree
        within their sigma; DESIGN_P1 6.2).
    ADMIT otherwise.  The engine keeps the admitted candidate (the agreement test makes "best vs last" moot at the
    moment of admission; before it, nothing is used)."""
    reasons = []
    # the init window is the KFs a usable record touches (a refused record - KF 0 before the first IMU sample on
    # MadMax - leaves its KF out of every statistic): the >= MIN_INIT_KF rule counts those
    n_eff = int(n_kf)
    if init is not None and init.ok and init.n_touched is not None:
        n_eff = min(n_eff, int(init.n_touched))
    if kR is not None and kR.get("n"):
        n_eff = min(n_eff, int(kR["n"]) + 1)
    st = {"n_kf": int(n_kf), "n_kf_eff": n_eff}
    rule = ADMISSION_RULE if rule is None else rule
    if rule not in ("igt", "pigt"):
        raise ValueError("admission rule %r: expected 'igt' or 'pigt'" % (rule,))
    st["rule"] = rule
    if kR is not None:
        st.update(k_R=kR["k_R"], k_R_bg0=kR["k_R_bg0"], k_R_refit=kR["k_R_refit"], K=float(K))
    if rule == "pigt":
        state = getattr(prev, "igt_state", None)
        if power is not None:
            st.update(igt_alpha=power["alpha"], igt_power=power["power"], igt_k_R_signal=power["k_R_signal"],
                      td_hat_ms=1e3 * power["td_hat"], td_sd_ms=1e3 * power["td_sd"], igt_kappa_g=power["kappa_g"],
                      igt_sigma_th_mrad=1e3 * power["sigma_th"])
        if state == "pass":
            st["igt"] = "pass (final, decided at an earlier KF)"
        elif kR is None:
            reasons.append("no IGT statistic (no factor with >= 2 IMU steps)")
        elif n_eff < min_kf:
            reasons.append("%d KFs in the init window < %d: no IGT decision yet" % (n_eff, min_kf))
        else:
            dec, why = igt_decide(kR, power, K, p)
            st.update(igt=dec, igt_reason=why)
            if dec == "refuse":
                st["igt_state"] = "refuse"
                return Admission("refuse", [why], st)
            if dec == "pass":
                state = "pass"
            else:
                reasons.append(why)
        st["igt_state"] = state
    elif kR is not None:
        # the gate is evaluated over the init window (>= MIN_INIT_KF KFs, DESIGN_P1 6.2): a refusal is final for the
        # segment, so it is never taken on fewer KFs (below, k_R > K only defers)
        if kR["k_R"] > K and n_eff >= min_kf:
            return Admission("refuse", ["IGT k_R %.2f > K %.0f (gyro inconsistent with the VO rotations)"
                                        % (kR["k_R"], K)], st)
        if kR["k_R"] > K:
            reasons.append("IGT k_R %.2f > K before the init window is complete" % kR["k_R"])
    else:
        reasons.append("no IGT statistic (no factor with >= 2 IMU steps)")
    if n_eff < min_kf:
        reasons.append("%d KFs in the init window < %d" % (n_eff, min_kf))
    if gyro is None:
        reasons.append("no gyro alignment")
    elif not gyro.converged:
        reasons.append("gyro alignment not converged")
    if init is None or not init.ok:
        reasons.append("no init fit (%s)" % (None if init is None else init.reason))
        return Admission("defer", reasons, st)
    st.update(tilt_r_deg=init.tilt_r_deg, tilt_r_known_ba_deg=init.tilt_r_known_ba_deg, vif=init.vif,
              excitation=init.excitation, g_free_norm=init.g_free_norm, dchi2_mag=init.dchi2_mag,
              chi2_dof=init.chi2_dof, kappa_vce=init.kappa_vce)
    if init.rank_deficient:
        reasons.append("gravity/ba block rank-deficient")
    elif init.degenerate:
        reasons.append("tilt not certified (cause: %s): radius %.2f deg at %.3f, bar %.1f" % (
            init.degenerate_cause, init.tilt_r_deg, p, TILT_BAR_DEG))
    if not init.mag_ok:
        reasons.append("|g| constraint rejected: free |g| %.3f +- %.3f, dchi2 %.1f" % (
            init.g_free_norm, init.g_free_sigma, init.dchi2_mag))
    ac = init.att_chain
    if ac is not None:
        st.update(att_chain_deg=ac["deg"], att_chain_chi2=ac["chi2"])
        if not ac["ok"]:
            reasons.append("newest KF's VO tilt disagrees with the gyro chain: %.2f deg (sd %.2f), chi2 %.1f" % (
                ac["deg"], ac["sd_deg"], ac["chi2"]))
    if not init.vce_converged:
        reasons.append("variance components not converged")
    if prev is None or not getattr(prev, "ok", False):
        reasons.append("no previous candidate")
    else:
        c, dof, ok, meth = _agreement(prev, init, p, noise)
        st.update(agree_chi2=c, agree_dof=dof, agree_method=meth)
        if not ok:
            reasons.append("disagrees with the previous candidate (chi2 %.1f, dof %d, %s)" % (c, dof, meth))
    return Admission("defer" if reasons else "admit", reasons, st)


# ======================================================= the POWERED IGT rule =====
# U6 fired the pre-registered kill on both sides: k_R <= K on Rosario v1 at 2.7-8.8 deg of rotation (the
# test had no power there) and k_R > K on good EuRoC IMUs (v2_03 / v1_03: the VO attitude noise of the live window, not
# the gyro, pushes the statistic past K; the test was not sized there).  The replacement keeps the IGT statistic and K
# and decides only when the decision is CALIBRATED on this window (igt_decide; refusal first):
#   REFUSE  k_R > K  and  alpha = P(k_R > K | consistent IMU, this window's measured noise)        <= 1 - P_GATE  (sized)
#           or the OFFSET ROUTE is powered (td_sd <= TD_HARM / (2 z), z = sqrt(chi2_1(P_GATE))) and |td_hat| >= TD_HARM/2
#   PASS    k_R <= K and [ power = P(k_R > K | clock offset TD_HARM, both signs) >= P_GATE  or  the offset route is
#           powered and |td_hat| < TD_HARM / 2 ]
#   otherwise UNDECIDED: the segment stays in VO mode and the next KF's window is tested.  (The offset route exists
#   because the k_R test has almost no power on ground vehicles: a yaw-only synthetic window with a 67.7 ms offset reads
#   k_R 1.65, power 0, while the pooled offset estimate reads -66 +- 7 ms.)  The first decision is final
#   for the segment (a PASS is carried in ViInit.igt_state); after admission the watchdog re-tests rolling windows with
#   the sized-refusal half (watchdog_stat 'igt').  Both probabilities are Monte-Carlo replays of the IGT statistic itself
#   (median chi at bg0 and after the bias refit, min) under the window's own noise model: the gyro component kappa_g
#   Q_gyro and the KF attitude jitter s_th J_th J_th^T, estimated by REML jointly with the gyro bias AND the clock offset
#   (offset_align), so a real offset is not absorbed into the noise model the size test uses.
TD_HARM = 0.0677     # E: an EXTERNAL effect size, not a class A-D constant (module docstring): register an internal note
                     # (an offline survey), the SMALLEST confirmed Rosario v1 camera-IMU offset (r04 / r06
                     # -67.7 ms; r02 / r03 -124.5 / -132.7) measured offline on the rows 'pigt' is judged on; used only by
                     # the opt-in 'pigt' rule
N_MC = 20000         # D numerical: Monte-Carlo replays per probability; standard error at the decision level
                     # sqrt((1 - P_GATE) P_GATE / N_MC) = 2.2e-4 (stated with every result)
ADMISSION_RULE = "igt"   # init_candidate / admission default: "igt" (the U6 rule, kept until the engine adopts the
                         # replacement between test series) | "pigt" (the powered rule)

OffsetAlign = namedtuple("OffsetAlign", "bg td td_sd cov kappa sigma_th comp_sd n removed used converged chi2 dof "
                                        "with_td")
OffsetAlign.__doc__ = ("offset_align result: bg (3,) and td [s] (the gyro integral over stamps [t_k, t_k+1] is the "
                       "rotation over [t_k + td, t_k+1 + td]; Rosario v1 R-TD offsets are NEGATIVE in this sign), td_sd, "
                       "cov (4x4 [bg, td], plug-in), kappa (Kappa of the gyro component), sigma_th [rad] KF attitude jitter, "
                       "comp_sd, n factors, removed (snooped records), used, converged, chi2 / dof (GLS), with_td (False "
                       "when the window holds no rate change: td not estimable, fitted at 0)")


def kf_rates(t, gyro, kf_t, bg, span=TD_HARM, horizon=None):
    """The body angular rate at each KF stamp t_k from the gyro: the mean of the bias-corrected samples in the CAUSAL
    span [t_k - span, t_k] (a clock offset of `span` moves the gyro integral by exactly this mean rate x span), holding
    the last sample at or before t_k when the span holds none.  Only samples with t <= min(t_k, horizon) are read (C40;
    horizon = the newest released IMU horizon).  (n+1, 3); NaN where no sample exists."""
    t = np.asarray(t, float)
    kf_t = np.asarray(kf_t, float)
    bg = np.asarray(bg, float).reshape(3)
    hz = math.inf if horizon is None else float(horizon)
    out = np.full((len(kf_t), 3), np.nan)
    for k, tk in enumerate(kf_t):
        te = min(float(tk), hz)
        i1 = int(np.searchsorted(t, te, side="right"))
        i0 = int(np.searchsorted(t, te - float(span), side="left"))
        if i1 > i0:
            out[k] = np.asarray(gyro[i0:i1], float).mean(axis=0) - bg
        elif i1 > 0:
            out[k] = np.asarray(gyro[i1 - 1], float) - bg
    return out


def _rot_noise_parts(R, pres, used, n_kf, bg, noise):
    """The noise components of the rotation residuals of the factor list `used` at bias bg (the model of
    gyro_bias_align): Ug (Ug Ug^T = blockdiag(C_RR,k) + the gyro bias walk across intervals pinned at t_n when `noise`
    is given), Ub (the white part alone), Jth (3m x 3 n_kf, the KF attitude-jitter Jacobian), and r, J, A of
    _rot_residuals."""
    m = len(used)
    Ub = np.zeros((3 * m, 3 * m))
    for r_, k in enumerate(used):
        ev, U = np.linalg.eigh(0.5 * (pres[k].cov[0:3, 0:3] + pres[k].cov[0:3, 0:3].T))
        Ub[3 * r_:3 * r_ + 3, 3 * r_:3 * r_ + 3] = U * np.sqrt(np.maximum(ev, 0.0))[None, :]
    r, J, Jri, A = _rot_residuals(R, pres, used, bg)
    parts = [Ub]
    if noise is not None:
        tk = np.array([pres[k].t0 for k in used])
        K = float(pres[used[-1]].t1) - np.maximum.outer(tk, tk)
        evk, Uk = np.linalg.eigh(0.5 * (K + K.T))
        Jb = np.zeros((3 * m, 3 * m))
        for r_ in range(m):
            Jb[3 * r_:3 * r_ + 3, 3 * r_:3 * r_ + 3] = J[r_]
        parts.append(noise.rw_g * (Jb @ np.kron(Uk * np.sqrt(np.maximum(evk, 0.0))[None, :], np.eye(3))))
    Ug = np.concatenate(parts, axis=1)
    Jth = np.zeros((3 * m, 3 * n_kf))
    for r_, k in enumerate(used):
        Jth[3 * r_:3 * r_ + 3, 3 * k:3 * k + 3] = -Jri[r_] @ A[r_].T
        Jth[3 * r_:3 * r_ + 3, 3 * k + 3:3 * k + 6] = Jri[r_]
    return Ug, Ub, Jth, r, J, A


def _offset_fit(R, pres, used, n_kf, bg0, H, noise, max_iter, with_td):
    """GLS of e_k(bg, td) = r_k(bg) + td h_k (h_k (3,) per unit offset; first order: r_k = -td h_k under an offset)
    with the REML noise model of gyro_bias_align (kappa_g Q_gyro + s_th J_th J_th^T); data only."""
    m = len(used)
    Hf = np.asarray(H, float).reshape(3 * m)
    npar = 4 if with_td else 3
    x = np.concatenate([np.asarray(bg0, float).reshape(3), [0.0]])
    s_init = np.array([1.0, float(np.median([np.trace(pres[k].cov[0:3, 0:3]) / 3.0 for k in used]))])
    fixed = np.array([False, False])
    s = s_init.copy()

    def lin(x_, s_):
        Ug, _Ub, Jth, r, J, _A = _rot_noise_parts(R, pres, used, n_kf, x_[:3], noise)
        e = r.reshape(3 * m) + x_[3] * Hf
        Jx = J.reshape(3 * m, 3) if not with_td else np.concatenate([J.reshape(3 * m, 3), Hf[:, None]], axis=1)
        C = s_[0] * (Ug @ Ug.T) + s_[1] * (Jth @ Jth.T)
        L = _chol_lower(C)
        return Ug, Jth, e, Jx, C, L, solve_triangular(L, Jx, lower=True), solve_triangular(L, e, lower=True)

    it_gn = it_vce = 0
    conv_gn = conv_vce = False
    for it_vce in range(1, MAX_VCE + 1):
        for it_gn in range(1, int(max_iter) + 1):
            Ug, Jth, e, Jx, C, L, Jw, ew = lin(x, s)
            N = Jw.T @ Jw
            step = -_Chol(N).solve(Jw.T @ ew)
            x[:npar] = x[:npar] + step
            if float(step @ N @ step) < _GN_TOL:
                conv_gn = True
                break
        Ug, Jth, e, Jx, C, L, Jw, ew = lin(x, s)
        cf = _Chol(Jw.T @ Jw)
        new, sd = _reml_step(L, [Ug, Jth], ew, Jw, cf, s, fixed, s_init)
        ch = np.abs(new - s)
        s = new
        if np.all(ch <= _VCE_FRAC * sd):
            conv_vce = True
            break
    Ug, Jth, e, Jx, C, L, Jw, ew = lin(x, s)
    N = Jw.T @ Jw
    cf = _Chol(N)
    Ni = cf.solve(np.eye(npar))
    num, rg = _vce_terms(L, Ug, ew, Jw, cf, s[0])
    raw0 = s[0] * num / rg if rg > _SQEPS else float("nan")
    _s, sd = _reml_step(L, [Ug, Jth], ew, Jw, cf, s, fixed, s_init)
    # small-sample calibration of the td variance (estimated noise components): Satterthwaite, as stereo_vi_init
    satt_td = (_satterthwaite(Ni, L, [Ug, Jth], Jw, cf, s, fixed, s_init, {"td": np.array([3])})["td"]
               if with_td else (math.inf, 0.0))
    wst = np.zeros(m)
    for r_ in range(m):
        sl = slice(3 * r_, 3 * r_ + 3)
        Ce = C[sl, sl] - Jx[sl] @ Ni @ Jx[sl].T
        ev, U = np.linalg.eigh(0.5 * (Ce + Ce.T))
        keep = ev > 3 * _EPS * max(ev.max(), _EPS)
        z = U[:, keep].T @ e[sl]
        wst[r_] = float(np.sum(z * z / ev[keep]))
    cov = np.full((4, 4), np.nan)
    cov[:npar, :npar] = Ni
    if not with_td:
        cov[3, 3] = math.inf
    return dict(x=x, cov=cov, s=s, raw0=raw0, sd=sd, wst=wst, chi2=float(ew @ ew), dof=3 * m - npar,
                conv=bool(conv_gn and conv_vce), satt_td=satt_td)


def offset_align(R_wb, preints, omega_kf, bg0, noise=None, p=P_GATE, max_iter=MAX_GN):
    """The gyro alignment with a camera-IMU CLOCK OFFSET as a fourth unknown (the kind of Rosario v1's defect, R-TD):
    e_k = Log((dR_k Exp(JRg_k (bg - bg_lin_k)))^T R_k^T R_k+1) + td h_k,  h_k = w_k+1 - (R_k^T R_k+1)^T w_k
    (w_k = kf_rates at KF k; first order: an offset td moves the gyro integral to [t_k + td, t_k+1 + td], so the
    rotation residual is -td h_k).  Same REML noise model and Baarda snooping as gyro_bias_align, on the IGT factor list
    (records with >= 2 integration steps) with finite rates at both ends.  Design reference: the camera-IMU offset as a
    state (Li & Mourikis, IJRR 2014; Qin & Shen, IROS 2018) - here a batch estimate on the init window.
    Returns OffsetAlign, or None with fewer than three such factors."""
    R = _rot_only(R_wb)
    if len(preints) != len(R) - 1:
        raise ValueError("offset_align: %d records for %d keyframes" % (len(preints), len(R)))
    om = np.asarray(omega_kf, float).reshape(len(R), 3)
    used0 = [k for k, q in enumerate(preints) if q is not None and q.ok and q.n_sub >= 2
             and np.all(np.isfinite(om[k])) and np.all(np.isfinite(om[k + 1]))]
    if len(used0) < 3:
        return None
    if isinstance(noise, dict):
        noise = F.ImuNoise.from_calib({"imu_noise": noise})
    b0 = np.asarray(bg0, float).reshape(3)
    Hall = {k: om[k + 1] - (R[k].T @ R[k + 1]).T @ om[k] for k in used0}
    used, removed = list(used0), []
    while True:
        H = np.stack([Hall[k] for k in used])
        res, with_td = None, True
        try:
            res = _offset_fit(R, preints, used, len(R), b0.copy(), H, noise, max_iter, True)
        except np.linalg.LinAlgError:
            with_td = False                                  # no rate change in the window: td not estimable
        if res is None:
            try:
                res = _offset_fit(R, preints, used, len(R), b0.copy(), H, noise, max_iter, False)
            except np.linalg.LinAlgError:
                return None
        m = len(used)
        if len(removed) >= len(used0) // 2 or m <= 3:
            break
        kmax = int(np.argmax(res["wst"]))
        if not (res["wst"][kmax] > chi2_ppf(1.0 - (1.0 - p) / m, 3)):
            break
        removed.append(used[kmax])
        used = [k for k in used if k != used[kmax]]
    x = res["x"]
    kap = kappa_from_component(res["raw0"], res["sd"][0], p)
    # td standard error: plug-in x the Satterthwaite factor nu / (nu - 2) (inf when nu <= 2: noise scale undetermined)
    fs = res["satt_td"][0]
    td_sd = (float(math.sqrt(res["cov"][3, 3] * fs)) if with_td and res["cov"][3, 3] >= 0 and math.isfinite(fs)
             else math.inf)
    return OffsetAlign(x[:3].copy(), float(x[3]), td_sd, res["cov"], kap, float(math.sqrt(max(res["s"][1], 0.0))),
                       [float(v) for v in res["sd"]], len(used), removed, used, res["conv"], res["chi2"], res["dof"],
                       with_td)


def igt_power(R_wb, preints, bg0, omega_kf, noise=None, td_h=TD_HARM, K=IGT_K, inflate=IGT_INFLATE, p=P_GATE,
              n_mc=N_MC, seed=0, off=None, need=None):
    """SIZE and POWER of the IGT refusal 'k_R > K' on THIS window (the records `preints` linearised at bg0, exactly the
    ones igt_k_R reads).  The IGT statistic is replayed n_mc times (fixed seed: a pure function) on residuals
    r = mu + z over the statistic's own factor list (ok records with >= 2 integration steps), whitened as
    rotation_consistency (chol(C_RR + 1e-14 I)^-1 / inflate), median chi at bg0 and after the linearised bias refit
    (the GN's least-squares step on the chi sum), k_R = sqrt(min / CHI2_3_MEDIAN):
      noise z ~ N(0, C), C = kappa_g Q_gyro + s_th J_th J_th^T from offset_align (the window's measured gyro and VO
              attitude noise with a possible offset fitted out; kappa_g after kappa_imu's >= 1 rule); and, for the
              power only, also C_decl = Q_gyro (declared, no VO jitter): the power is the MIN of the two (conservative:
              noise can lower the median chi of a large offset signal)
      alpha  = P(k_R > K | mu = 0)                    (the false-refusal rate of a consistent IMU on this window)
      power  = min over sign and noise model of P(k_R > K | mu_k = -(+-td_h) h_k), h_k = w_k+1 - A_k^T w_k
    Factors whose rates are not finite carry no offset signal (mu = 0: conservative).
    The OFFSET ROUTE (analytic, from offset_align): td_powered = td_sd <= td_h / (2 z), z = sqrt(chi2_1(P_GATE));
    td_refuse = td_powered and |td_hat| >= td_h / 2; td_alpha / td_power = its two error rates at this td_sd.
    need: None = every probability (reporting); 'alpha' or 'power' = only what igt_decide needs for k_R > K or <= K;
    'none' = the offset route alone (it decides) (the probabilities not computed are returned as None).
    Returns a dict (alpha, power, power_fit, power_declared, k_R_signal = the noise-free k_R of the offset alone (no
    refit), td_hat / td_sd / kappa_g / sigma_th of offset_align, the offset route, n, mc_se, ...) or None (fewer than
    3 factors)."""
    from xclean.backend.preint import CHI2_3_MEDIAN as C3
    R = _rot_only(R_wb)
    n_kf = len(R)
    om = np.asarray(omega_kf, float).reshape(n_kf, 3)
    idx = [k for k, q in enumerate(preints) if q is not None and q.ok and q.n_sub >= 2]
    if len(idx) < 3:
        return None
    if isinstance(noise, dict):
        noise = F.ImuNoise.from_calib({"imu_noise": noise})
    b0 = np.asarray(bg0, float).reshape(3)
    if off is None:
        off = offset_align(R, preints, om, b0, noise, p)
    if off is None:
        return None
    m = len(idx)
    Ug, _Ub, Jth, _r0, J, A = _rot_noise_parts(R, preints, idx, n_kf, b0, noise)
    kap = float(off.kappa.kappa) if (off.kappa is not None and np.isfinite(off.kappa.kappa)) else 1.0
    Qg = Ug @ Ug.T
    Cfit = kap * Qg + off.sigma_th ** 2 * (Jth @ Jth.T)
    Cr = np.stack([preints[k].cov[0:3, 0:3] for k in idx]) + np.eye(3)[None] * 1e-14
    W = np.linalg.inv(np.linalg.cholesky(Cr)) / float(inflate)
    Hm = np.zeros((m, 3))
    nmu = 0
    for r_, k in enumerate(idx):
        if np.all(np.isfinite(om[k])) and np.all(np.isfinite(om[k + 1])):
            Hm[r_] = om[k + 1] - A[r_].T @ om[k]
            nmu += 1
    mu1 = (-Hm).reshape(3 * m)                                      # residual per second of offset
    Wb = np.zeros((3 * m, 3 * m))                                   # block-diagonal IGT whitening
    for r_ in range(m):
        Wb[3 * r_:3 * r_ + 3, 3 * r_:3 * r_ + 3] = W[r_]
    Jwb = Wb @ J.reshape(3 * m, 3)                                  # whitened bias Jacobian (3m x 3)
    P_ref = np.eye(3 * m) - Jwb @ np.linalg.solve(Jwb.T @ Jwb, Jwb.T)   # the refit's residual projector (linear GN)
    mu_w = Wb @ mu1
    # the records are linearised at bg0, not at the bias: every replay carries the fitted bias error as a mean,
    # r(bg0) = r(bg_hat) + J (bg0 - bg_hat), so min(k_R_bg0, k_R_refit) is replayed in the regime the window is in
    # (measured before this term: a zero bias error made the replayed median k_R 4 % low at sigma_th 2 mrad)
    mu_b = Jwb @ (b0 - np.asarray(off.bg, float).reshape(3))
    Mf, Md = Wb @ _chol_lower(Cfit), Wb @ _chol_lower(Qg)
    # which probabilities the decision needs (igt_decide): need=None -> all (reporting)
    if need == "none":
        names = []
    elif need == "alpha":
        names = ["alpha"]
    elif need == "power":
        names = ["fit+", "fit-", "decl+", "decl-"]
    else:
        names = ["alpha", "fit+", "fit-", "decl+", "decl-"]
    cases = {"alpha": (Mf, 0.0), "fit+": (Mf, 1.0), "fit-": (Mf, -1.0), "decl+": (Md, 1.0), "decl-": (Md, -1.0)}
    cnt = dict.fromkeys(names, 0)
    meds = {k: [] for k in names}
    rng = np.random.default_rng(seed)
    left = int(n_mc)
    thr = K * K * C3                                                # k_R > K  <=>  min(median chi) > K^2 C3
    while left > 0:
        nb = min(left, 5000)
        left -= nb
        xi = rng.standard_normal((nb, 3 * m))
        for name in names:
            M, sgn = cases[name]
            rw = xi @ M.T + mu_b[None, :]
            if sgn != 0.0:
                rw += sgn * float(td_h) * mu_w[None, :]
            med0 = np.median((rw * rw).reshape(nb, m, 3).sum(axis=2), axis=1)
            rw1 = rw @ P_ref                                        # P_ref symmetric
            med1 = np.median((rw1 * rw1).reshape(nb, m, 3).sum(axis=2), axis=1)
            mm = np.minimum(med0, med1)
            cnt[name] += int(np.count_nonzero(mm > thr))
            meds[name].append(np.median(mm))
    fr = {k: (cnt[k] / float(n_mc) if k in cnt else None) for k in cases}
    kmed = {k: (float(math.sqrt(np.mean(meds[k]) / C3)) if k in meds else None) for k in cases}
    sig = mu_w.reshape(m, 3) * float(td_h)
    q = 1.0 - p
    # the OFFSET route (the pooled estimate of the defect itself): the two-point test td = 0 vs |td| = td_h decided at
    # the midpoint td_h / 2; both error rates <= 1 - P_GATE when td_sd <= td_h / (2 z), z = sqrt(chi2_1(P_GATE)) (the
    # two-sided normal quantile; the one-sided miss rate is then smaller still)
    from scipy.stats import norm
    z = math.sqrt(chi2_ppf(p, 1))
    tsd = float(off.td_sd)
    td_pow = bool(np.isfinite(tsd) and tsd <= float(td_h) / (2.0 * z))
    a_td = float(2.0 * norm.sf(0.5 * float(td_h) / tsd)) if (np.isfinite(tsd) and tsd > 0) else 1.0
    pw_td = float(norm.sf(-0.5 * float(td_h) / tsd)) if (np.isfinite(tsd) and tsd > 0) else 0.0
    return {"td_powered": td_pow, "td_refuse": bool(td_pow and abs(off.td) >= 0.5 * float(td_h)),
            "td_alpha": a_td, "td_power": pw_td, "z": z,
            "alpha": fr["alpha"],
            "power": None if fr["fit+"] is None else min(fr["fit+"], fr["fit-"], fr["decl+"], fr["decl-"]),
            "power_fit": None if fr["fit+"] is None else min(fr["fit+"], fr["fit-"]),
            "power_declared": None if fr["decl+"] is None else min(fr["decl+"], fr["decl-"]),
            "k_R_signal": float(math.sqrt(np.median(np.einsum("ki,ki->k", sig, sig)) / C3)),
            "k_R_med_H0": kmed["alpha"], "k_R_med_H1": kmed["fit+"],
            "td_hat": off.td, "td_sd": off.td_sd, "td_with": off.with_td, "kappa_g": kap,
            "kappa_g_est": None if off.kappa is None else float(off.kappa.estimate), "sigma_th": off.sigma_th,
            "offset_removed": len(off.removed), "offset_converged": off.converged, "n": m, "n_mu": nmu,
            "n_mc": int(n_mc), "mc_se": math.sqrt(q * (1.0 - q) / float(n_mc)), "td_h": float(td_h), "K": float(K)}


def igt_decide(kR, pw, K=IGT_K, p=P_GATE):
    """The powered IGT decision on one window: ('refuse' | 'pass' | 'undecided', reason).  Two routes, refusal first:
      REFUSE  (a) k_R > K and the k_R test is SIZED on this window: alpha <= 1 - P_GATE, or
              (b) the offset route is powered (td_sd <= td_h / (2 z)) and |td_hat| >= td_h / 2;
      PASS    k_R <= K and (a') the k_R test is POWERED (power >= P_GATE) or (b') the offset route is powered and
              |td_hat| < td_h / 2;
      otherwise UNDECIDED (stay in VO, test the next KF's window)."""
    if kR is None or pw is None:
        return "undecided", "IGT undecided: no statistic / no power model (fewer than 3 factors)"
    k = float(kR["k_R"])
    tdm, tsm, thm = 1e3 * pw["td_hat"], 1e3 * pw["td_sd"], 1e3 * pw["td_h"]
    al = pw["alpha"] if pw.get("alpha") is not None else math.inf          # not computed = not sized
    pk = pw["power"] if pw.get("power") is not None else -math.inf         # not computed = not powered
    if k > K and al <= 1.0 - p:
        return "refuse", ("IGT k_R %.2f > K %.0f, sized: P(k_R > K | consistent IMU, this window's noise) = %.4f "
                          "(td_hat %.1f +- %.1f ms)" % (k, K, pw["alpha"], tdm, tsm))
    if pw.get("td_refuse"):
        return "refuse", ("clock offset td_hat %.1f +- %.1f ms, |td_hat| >= %.1f ms (half the %.1f ms harmful "
                          "offset), powered (k_R %.2f)" % (tdm, tsm, 0.5 * thm, thm, k))
    if k > K:
        return "undecided", ("IGT undecided: k_R %.2f > K but not sized (P(k_R > K | consistent IMU) = %.4f > %.4f); "
                             "offset route %s" % (k, al, 1.0 - p,
                                                  "powered" if pw.get("td_powered") else "not powered"))
    if pw.get("td_powered"):
        return "pass", ("IGT k_R %.2f <= K, offset route powered: td_hat %.1f +- %.1f ms, |td_hat| < %.1f ms"
                        % (k, tdm, tsm, 0.5 * thm))
    if pk >= p:
        return "pass", ("IGT k_R %.2f <= K, k_R test powered: P(k_R > K | clock offset %.1f ms) = %.4f"
                        % (k, thm, pk))
    return "undecided", ("IGT undecided: k_R %.2f <= K, not powered (P(k_R > K | offset %.1f ms) = %.4f; td_hat %.1f "
                         "+- %.1f ms needs sd <= %.1f ms)" % (k, thm, pk, tdm, tsm, thm / (2.0 * pw["z"])))


# ========================================================================================= bias bound (5) =========
BiasBound = namedtuple("BiasBound", "r_bg r_ba k P_bg P_ba t")
BiasBound.__doc__ = ("radii (P_GATE) on |bg - bg_init| [rad/s] and |ba - ba_init| [m/s^2] after t s; k = the "
                     "per-axis sigma multiple sqrt(chi2_3(P_GATE)); P_bg, P_ba the 3x3 covariances they come from")


def bias_bound(noise, t_since_init, cov_bg0=None, cov_ba0=None, kappa=1.0, p=P_GATE):
    """Deliverable 5a, Class B: how far the biases may drift from their init values in t seconds.
    A bias random walk of declared density rw (Class A) has per-axis variance kappa rw^2 t after t seconds (kappa: the
    IMU-trust factor, which imu_factor applies to the whole 15 x 15 including the random-walk rows); add the init
    covariance; the bound is the P_GATE radius of the 3-D Gaussian: |b - b_init| <= sqrt(chi2_3(P_GATE) lambda_max(P)).
    k = sqrt(chi2_3(0.999)) = 4.03 per axis (Class D).  No turn-on bias is declared on any dev rig (DESIGN_P1 4.2's
    'declared turn-on bias' does not exist in the calibs), so the bound is on the DRIFT since init, not on |b|."""
    if isinstance(noise, dict):
        noise = F.ImuNoise.from_calib({"imu_noise": noise})
    t = max(float(t_since_init), 0.0)
    k2 = chi2_ppf(p, 3)
    Pg = (np.zeros((3, 3)) if cov_bg0 is None else np.asarray(cov_bg0, float)) + float(kappa) * noise.rw_g ** 2 * t * \
        np.eye(3)
    Pa = (np.zeros((3, 3)) if cov_ba0 is None else np.asarray(cov_ba0, float)) + float(kappa) * noise.rw_a ** 2 * t * \
        np.eye(3)
    rg = math.sqrt(k2 * float(np.linalg.eigvalsh(Pg).max()))
    ra = math.sqrt(k2 * float(np.linalg.eigvalsh(Pa).max()))
    return BiasBound(rg, ra, math.sqrt(k2), Pg, Pa, t)


def bias_excursion(b, b0, P, p=P_GATE):
    """(chi2, exceeded): the Mahalanobis excursion of a bias from its init value under the bound's covariance P,
    tested against chi2_3(P_GATE)."""
    d = np.asarray(b, float) - np.asarray(b0, float)
    c = float(d @ np.linalg.solve(np.asarray(P, float), d))
    return c, bool(c > chi2_ppf(p, 3))


# =========================================================================================== watchdog (5) ==========
Watchdog = namedtuple("Watchdog", "to_vo triggers active consec stats")
Watchdog.__doc__ = ("to_vo: send the segment to VO now; triggers: the triggers that caused it; active: tests failing at "
                    "this KF interval; consec: the updated consecutive-failure counts (pass back as `state`); stats")


def watchdog_stat(summary, state=None, p=P_GATE, n_consec=N_CONSEC, K=IGT_K):
    """Deliverable 5b: the IMU-divergence watchdog (DESIGN_P1 6.4), evaluated once per KF interval on the window's
    residual summary.  Every test is a chi2 test at the Class-D level P_GATE; nothing else is set.

    summary (dict; each group optional):
      imu_chi2 (m,), imu_dof (m,) : the window's per-factor WHITENED IMU chi2 (kappa applied, as in the solve) and
                                    their dof; `imu_redundancy` True when imu_dof are post-fit redundancies
                                    (dof - leverage), which enables the lower-tail test
      innov (6,), innov_cov (6,6) : the vision-only relative pose over the last KF interval (step-7 PnP) minus the
                                    window's, [dphi, dp], and the covariance of that difference
      bg, ba, bg0, ba0, P_bg, P_ba: current biases, their init values and the bias_bound covariances
      igt = {k_R, power}          : the IGT statistic on the newest KF window (igt_k_R's k_R) and igt_power on that
                                    window - DESIGN_P1 6.4's third trigger, decided by igt_decide's refusal half
    TESTS
      imu_hi  : sum chi2 > chi2_{sum dof}(P_GATE)  - the IMU disagrees with the window (it is being out-voted);
      imu_lo  : sum chi2 < chi2_{sum dof}(1 - P_GATE), only with redundancies - the residuals are implausibly SMALL:
                the window follows the IMU and vision no longer constrains it.  This is the C1 lesson (ALT_basalt 7.4:
                a 79 deg heading divergence passed an inertial-chi monitor whose chi ratios were 1e-2..1e-4);
      innov   : innovation chi2 > chi2_6(P_GATE) (the design's primary trigger); its heading part is reported;
      bias    : either bias excursion chi2 > chi2_3(P_GATE)  -> to_vo IMMEDIATELY (DESIGN_P1 4.2 'crossing it sends the
                IMU to IMU-off');
      igt     : igt_decide says 'refuse' on the newest window (k_R > K with the test sized there, or a powered
                clock offset >= TD_HARM / 2): the powered rule's refusal half (an unsized k_R > K - large VO attitude
                noise - is reported, not acted on).
    to_vo when `bias` fires, or when any statistical test has failed on n_consec (= 2, Class D, DESIGN_P1 6.4)
    consecutive KF intervals.  Pure: pass the returned `consec` back as `state`."""
    st = {}
    active = []
    cons = dict(state or {})
    if summary.get("imu_chi2") is not None:
        c = np.asarray(summary["imu_chi2"], float)
        d = np.broadcast_to(np.asarray(summary.get("imu_dof", 15.0), float), c.shape)
        cs, ds = float(c.sum()), float(d.sum())
        st.update(imu_chi2_sum=cs, imu_dof_sum=ds, imu_ratio=cs / max(ds, _EPS),
                  imu_p_upper=1.0 - chi2_cdf(cs, ds), imu_p_lower=chi2_cdf(cs, ds))
        if cs > chi2_ppf(p, ds):
            active.append("imu_hi")
        if summary.get("imu_redundancy") and cs < chi2_ppf(1.0 - p, ds):
            active.append("imu_lo")
    if summary.get("innov") is not None:
        dv = np.asarray(summary["innov"], float).reshape(6)
        Pi = np.asarray(summary["innov_cov"], float)
        ci = float(dv @ np.linalg.solve(Pi, dv))
        st.update(innov_chi2=ci, innov_rot_deg=float(np.degrees(np.linalg.norm(dv[:3]))))
        if ci > chi2_ppf(p, 6):
            active.append("innov")
    bias_fire = False
    if summary.get("bg") is not None and summary.get("P_bg") is not None:
        cg, eg = bias_excursion(summary["bg"], summary["bg0"], summary["P_bg"], p)
        st["bg_chi2"] = cg
        bias_fire |= eg
    if summary.get("ba") is not None and summary.get("P_ba") is not None:
        ca, ea = bias_excursion(summary["ba"], summary["ba0"], summary["P_ba"], p)
        st["ba_chi2"] = ca
        bias_fire |= ea
    if summary.get("igt") is not None:
        g_ = summary["igt"]
        dec, why = igt_decide({"k_R": g_["k_R"]} if g_.get("k_R") is not None else None, g_.get("power"), K, p)
        st.update(igt_k_R=g_.get("k_R"), igt_decision=dec, igt_reason=why)
        if dec == "refuse":
            active.append("igt")
    if bias_fire:
        active.append("bias")
    for k in ("imu_hi", "imu_lo", "innov", "igt"):
        cons[k] = (cons.get(k, 0) + 1) if k in active else 0
    trig = [k for k in ("imu_hi", "imu_lo", "innov", "igt") if cons[k] >= n_consec]
    if bias_fire:
        trig.append("bias")
    return Watchdog(bool(trig), trig, active, cons, st)


# ============================================================================ causal plumbing + the pipeline ======
def preintegrate_intervals(t, gyro, acc, kf_t, horizons, bg, ba, noise):
    """The n records between consecutive KFs (stamps kf_t, k = 0..n), record k preintegrated with ITS OWN released
    horizon (horizons[k], scalar = one for all; the engine passes the horizon at KF k+1's creation), linearised at
    (bg, ba).  imu_factor.preintegrate reads no sample past the horizon (C40)."""
    kf_t = np.asarray(kf_t, float)
    hz = np.broadcast_to(np.asarray(horizons, float), (len(kf_t) - 1,))
    return [F.preintegrate(t, gyro, acc, kf_t[k], kf_t[k + 1], float(hz[k]), bg, ba, noise)
            for k in range(len(kf_t) - 1)]


def check_causal(kf_t, preints, t_n=None):
    """Structural causality audit: n records for n + 1 KFs; record k spans exactly [kf_t[k], kf_t[k+1]]; stamps
    strictly increasing; every record's last released sample <= its horizon <= t_n (the newest KF's stamp, or the
    given decision time).  Raises AssertionError on a violation; returns the max (t_data_end) seen."""
    kf_t = np.asarray(kf_t, float)
    tn = float(kf_t[-1]) if t_n is None else float(t_n)
    assert len(preints) == len(kf_t) - 1, "records / KFs mismatch"
    assert np.all(np.diff(kf_t) > 0), "KF stamps not strictly increasing"
    tmax = -math.inf
    for k, q in enumerate(preints):
        assert q.t0 == kf_t[k] and q.t1 == kf_t[k + 1], "record %d does not span its KF interval" % k
        if q.ok:
            assert q.t_data_end <= tn, "record %d read IMU data at %.6f > t_n %.6f" % (k, q.t_data_end, tn)
            tmax = max(tmax, q.t_data_end)
    return tmax


Candidate = namedtuple("Candidate", "n_kf t_n kR gyro init admission timing power", defaults=(None,))


def init_candidate(T_wb, kf_t, t, gyro, acc, horizons, noise, bg0, bg0_cov=None, prev=None, g_mag=G_STD,
                   pose_sigma=None, p=P_GATE, rule=None, td_h=TD_HARM, ba_prior=None, ba_prior_cov=None):
    """The whole causal VI-init step at one keyframe (what the engine calls until admission):
      1. records at the held bias bg0 (the causal bootstrap; the IGT statistic is defined at it);
      2. igt_k_R on them; under rule 'pigt' (default ADMISSION_RULE) also igt_power (offset_align + the Monte-Carlo
         size / power on the same records, the KF rates from kf_rates over the causal span TD_HARM), skipped once
         prev.igt_state == 'pass' (final);
      3. gyro_bias_align (prior: bg0 / bg0_cov when the bootstrap found a stationary window);
      4. records re-preintegrated at the aligned bias (same horizons);
      5. stereo_vi_init with the alignment as the bg prior;
      6. admission against the previous candidate (prev = the previous Candidate's init, or None).
    Inputs are only the KFs created so far and IMU arrays whose samples past each record's horizon are never read
    (the KF rates read samples <= min(t_k, the newest horizon)).  ba_prior / ba_prior_cov: an optional DECLARED
    accelerometer-bias prior (Class A: a turn-on bias spec; none is declared on any dev rig, so the engine passes
    none) - the only thing besides roll / pitch excitation that separates g from -g under yaw-only motion.  Under 'pigt' the returned init always exists (an
    empty ViInit when no fit) and carries igt_state, so an engine that keeps prev = cand.init carries the decision.
    Returns Candidate with per-stage wall times [ms] (+ `power`, the igt_power dict or None)."""
    import time
    rule = ADMISSION_RULE if rule is None else rule
    tm = {}
    t0 = time.perf_counter()
    R, P = _rot_stack(T_wb)
    bg0 = np.asarray(bg0, float).reshape(3)
    pres0 = preintegrate_intervals(t, gyro, acc, kf_t, horizons, bg0, np.zeros(3), noise)
    tm["preint0"] = 1e3 * (time.perf_counter() - t0)
    t1 = time.perf_counter()
    kR = igt_k_R(R, pres0, bg0)
    tm["igt"] = 1e3 * (time.perf_counter() - t1)
    pw = None
    if rule == "pigt" and getattr(prev, "igt_state", None) != "pass" and len(R) >= MIN_INIT_KF and kR is not None:
        t1 = time.perf_counter()
        hz = float(np.max(np.asarray(horizons, float))) if np.size(horizons) else None
        om = kf_rates(t, gyro, kf_t, bg0, td_h, hz)
        off = offset_align(R, pres0, om, bg0, noise, p)
        # only the probability igt_decide needs: alpha when k_R > K; the k_R power when k_R <= K and the offset route
        # does not decide (a powered offset route decides without the Monte Carlo)
        need = "alpha" if kR["k_R"] > IGT_K else "power"
        if need == "power" and off is not None and np.isfinite(off.td_sd) and \
                off.td_sd <= td_h / (2.0 * math.sqrt(chi2_ppf(p, 1))):
            need = "none"
        pw = igt_power(R, pres0, bg0, om, noise=noise, td_h=td_h, p=p, off=off, need=need) if off is not None else None
        tm["power"] = 1e3 * (time.perf_counter() - t1)
    t1 = time.perf_counter()
    ga = gyro_bias_align(R, pres0, bg_prior=bg0 if bg0_cov is not None else None, bg_prior_cov=bg0_cov, noise=noise,
                         p=p)
    tm["gyro"] = 1e3 * (time.perf_counter() - t1)
    ini = None
    if ga is not None:
        t1 = time.perf_counter()
        pres1 = preintegrate_intervals(t, gyro, acc, kf_t, horizons, ga.bg, np.zeros(3), noise)
        tm["preint1"] = 1e3 * (time.perf_counter() - t1)
        t1 = time.perf_counter()
        ini = stereo_vi_init(list(zip(R, P)), pres1, g_mag=g_mag, bg_prior=ga.bg, bg_prior_cov=ga.cov,
                             ba_prior=ba_prior, ba_prior_cov=ba_prior_cov,
                             pose_sigma=(pose_sigma if pose_sigma is not None else (None, ga.sigma_th)),
                             p=p, gyro=ga, noise=noise,
                             att_rw=max(ga.kappa.estimate - 1.0, 0.0) if np.isfinite(ga.kappa.estimate) else 0.0)
        tm["init"] = 1e3 * (time.perf_counter() - t1)
    adm = admission(len(R), kR, ga, ini, prev=prev, p=p, noise=noise, power=pw, rule=rule)
    if rule == "pigt":
        if ini is None:
            ini = _empty_init("no gyro alignment (fewer than 3 usable records)")
        ini = ini._replace(igt_state=adm.stats.get("igt_state"))
    tm["total"] = 1e3 * (time.perf_counter() - t0)
    return Candidate(len(R), float(np.asarray(kf_t, float)[-1]), kR, ga, ini, adm, tm, pw)


def warmup():
    """Compile imu_factor's kernels and touch every path once (before the frame clock starts)."""
    F.warmup()
    rng = np.random.default_rng(0)
    ts = np.arange(0.0, 2.0, 0.005)
    W = np.tile([0.3, -0.2, 0.4], (len(ts), 1)) + 0.01 * rng.standard_normal((len(ts), 3))
    A = np.tile([0.1, 0.2, 9.8], (len(ts), 1)) + 0.01 * rng.standard_normal((len(ts), 3))
    kt = np.linspace(0.1, 1.9, 9)
    T = []
    for k in range(9):
        T.append((so3_exp(np.array([0.3, -0.2, 0.4]) * (kt[k] - kt[0])).reshape(3, 3), np.array([0.1 * k, 0, 0])))
    nz = F.ImuNoise(1.7e-4, 2e-3, 2e-5, 3e-3, 200.0)
    init_candidate(T, kt, ts, W, A, kt[1:], nz, np.zeros(3))
    return True
