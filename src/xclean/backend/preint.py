#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""FULL IMU PREINTEGRATION, AND THE INERTIAL INITIALISATION THAT ARMS IT.

    pre = preintegrate(imu, t_i, t_j, bg, ba)      # dR, dv, dp + covariance
    g_w, vel, ba = inertial_init(imu, keyframes)   # gravity, velocities, bias

★ WHY THIS EXISTS NOW
    `backend/imu.py` uses the gyro alone, as a relative-ROTATION factor between
    consecutive keyframes.  That was the right first move - stereo already
    supplies scale, so the marginal value of inertial data is rotation, and the
    gyro is two orders of magnitude tighter than 200 keypoints of reprojection
    over a keyframe interval.

    But every bar on the benchmark is a stereo-INERTIAL number (ORB-SLAM3
    stereo-inertial, OKVIS2-X), and those systems carry velocity, gravity and
    accelerometer bias as states.  The measured signature of not having them is
    already recorded here: **V1_01's error is worst when the camera is SLOW**
    (0.0925 in the slowest quartile against 0.0661 in the fastest), which is
    exactly weak visual geometry with nothing inertial carrying it.  Rotation
    alone cannot fix that; the accelerometer can.

★ THE ONE THING STEREO MAKES EASY, AND IT IS THE HARD PART EVERYWHERE ELSE
    Monocular inertial initialisation has to solve for SCALE at the same time
    as gravity, velocity and bias, which is why ORB-SLAM3's MAP init exists and
    why this project measured it **rejecting 24 of 27 attempts** on real EuRoC
    (gravity bootstrapped during motion, ~17 degrees off).  With a stereo rig
    the visual trajectory is already metric, so the initialisation collapses to
    a LINEAR least-squares problem in (velocities, gravity, accel bias) given
    the visual poses.  That is a different problem with a different failure
    mode, and it should be said out loud rather than inheriting the monocular
    reputation.

★ THE COVARIANCE IS PROPAGATED, NOT ASSUMED
    Plan 8.4's NEES check already measured the gyro-only preintegration
    covariance as **~3x optimistic**.  The 9x9 propagation below is the honest
    form; whether it is still optimistic is a MEASUREMENT this file makes
    possible rather than a claim it makes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from xclean.core.derive import HUBER_K


def _inertial_fix(part):
    """ARM XTEN_INERTIAL_FIX=1 (2026-09-25) - a CONSISTENT inertial factor. Four measured defects, each a
    correctness fix, each the same rule on every rig:
      SAMPLES  a keyframe interval with NO IMU sample inside integrates ONE step, and its 9x9 covariance is
               EXACTLY rank 6 (B's position rows are dt/2 times its velocity rows). 28 such factors carried
               53-99.8 % of Rosario's inertial chi at 1e7-1e11 each; EuRoC has none. Refused (return None).
      HUBER    inertial_system's robust kernel existed and was never armed; one factor owned the objective.
               Armed at HUBER_K, the constant every other robust cost here uses.
      BIASCOV  the covariance carried no term for the uncertainty of the gyro bias it was integrated at; the
               run's own standard error (imu.bias_sigma, 20-75x EuRoC's on Rosario) enters the rotation block.
      BESTFIX  the global BA's best state is seeded with the FULL chi (defect 26), safe once the term is sane.
    XTEN_IF_<PART>=0/1 overrides one part, for ablation. Default: all OFF, byte-identical behaviour."""
    import os
    v = os.environ.get("XTEN_IF_" + part)
    if v is not None:
        return v == "1"
    return os.environ.get("XTEN_INERTIAL_FIX") == "1"
from xclean.core.lie import so3_exp, so3_log, so3_project, so3_jr
from xclean.backend.imu import gapspan_on as _gapspan_on       # T1-c

GRAVITY_MAG = 9.81

# ---- ARM XTEN_GBA_INERT_GATE=1:
# gate the run-end INERTIAL TERM on its own consistency, rig-agnostic.  (i) A rank-deficient preintegration factor
# (n < 2 IMU steps: rank-6 covariance) is refused - the gate IMPLIES the existing SAMPLES rule; (ii) when the gyro is
# grossly inconsistent with the entering keyframe ROTATIONS (k_R, below, > INERT_GATE_K) the global BA gets no inertial
# coupling (Tracker._build_inertial returns None, exactly as for any coupling it refuses today).  Default off: unset
# or "0" is byte-identical behaviour; "1" arms it; anything else is a hard error (a misspelt queue flag fails loudly).
INERT_GATE_ENV = "XTEN_GBA_INERT_GATE"
INERT_GATE_K = 10.0     # implied gyro-sigma factor beyond the solve's (inflated) covariance; one order of
                        # magnitude; chosen 2026-10-09 AFTER seeing the saved k_R distribution (an internal study:
                        # any K in (6.19, 13.27) gives the same saved decisions); one global constant,
                        # never set per row or per dataset
CHI2_3_MEDIAN = 2.366   # median of a chi^2 with 3 degrees of freedom (2.3660), the consistent value of a 3-DoF block


def inert_gate_on(env=None):
    """XTEN_GBA_INERT_GATE: unset / "" / "0" -> False; "1" -> True; anything else raises (no silent fallback)."""
    import os
    v = ((os.environ if env is None else env).get(INERT_GATE_ENV, "") or "").strip()
    if v in ("", "0"):
        return False
    if v == "1":
        return True
    raise RuntimeError("%s=%r: must be unset/0 or 1" % (INERT_GATE_ENV, v))


def _hat(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def _whiten(C):
    """The inverse Cholesky factor of a 3x3 block, so `W @ r` is r in units of
    its own sigma.  Falls back to a scaled identity if the block is not
    positive definite, which a very short interval can produce."""
    try:
        L = np.linalg.cholesky(C + np.eye(3) * 1e-14)
        return np.linalg.inv(L)
    except np.linalg.LinAlgError:
        s = float(np.sqrt(max(np.trace(C) / 3.0, 1e-14)))
        return np.eye(3) / s


@dataclass
class Preintegrated:
    """The interval (i -> j), expressed in keyframe i's BODY frame."""
    dR: np.ndarray                      # (3,3)
    dv: np.ndarray                      # (3,)
    dp: np.ndarray                      # (3,)
    dt: float
    cov: np.ndarray                     # (9,9) over [dphi, dv, dp]
    J_R_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_v_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_v_ba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_p_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_p_ba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n: int = 0

    def corrected(self, bg_new, ba_new):
        """First-order bias correction - the whole point of preintegration.

        Re-integrating every interval when the bias estimate moves is what
        makes inertial back-ends slow; the Jacobians below make it an affine
        update instead.  Valid while the bias CHANGE is small, which is what
        the random-walk term in the covariance is already asserting.
        """
        dbg = np.asarray(bg_new, float) - self.bg
        dba = np.asarray(ba_new, float) - self.ba
        dR = so3_project(self.dR @ so3_exp(self.J_R_bg @ dbg))
        dv = self.dv + self.J_v_bg @ dbg + self.J_v_ba @ dba
        dp = self.dp + self.J_p_bg @ dbg + self.J_p_ba @ dba
        return dR, dv, dp

    def information(self, inflate=1.0):
        """Inverse covariance, with a floor so a short interval cannot claim
        infinite confidence."""
        C = self.cov * float(inflate) ** 2
        C = C + np.eye(9) * 1e-12
        try:
            return np.linalg.inv(C)
        except np.linalg.LinAlgError:
            return np.eye(9) * 1e-6


def interval_samples(imu, t0, t1):
    """Sub-interval durations and MIDPOINT-interpolated measurements.

    ★ THE HALF-SAMPLE LAG THIS REPLACES WAS A REAL DEFECT.
    `imu.delta_rotation_body` advanced from the previous edge to `t[k]` using
    the measurement AT `t[k]` - the RIGHT endpoint of each sub-interval. That
    is a first-order rule with a systematic half-sample lag, and at 200 Hz it
    biases every interval by `0.5 * dt * d(omega)/dt`. It was invisible to the
    obvious synthetic test because a body rotating about a fixed WORLD axis has
    a constant BODY-frame rate, so left and right endpoints agree exactly -
    which is precisely the kind of test that certifies nothing.
    Measured on the accelerometer channel, where the signal actually varies:
    2.2e-2 m/s of velocity error over 1.5 s. The midpoint rule below is
    second order and the error drops into the noise.

    Sample boundaries are the IMU timestamps clipped to [t0, t1], so a
    keyframe interval that starts and ends mid-sample is integrated exactly
    rather than snapped to the sample grid.
    """
    t = np.asarray(imu.t, float)
    i0 = int(np.searchsorted(t, t0, "right"))
    i1 = int(np.searchsorted(t, t1, "left"))
    inner = t[i0:i1]
    edges = np.concatenate([[t0], inner, [t1]])
    edges = edges[np.concatenate([[True], np.diff(edges) > 1e-12])]
    if len(edges) < 2:
        return None
    dt = np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    g = np.stack([np.interp(mid, t, np.asarray(imu.g)[:, c])
                  for c in range(3)], axis=1)
    a = np.stack([np.interp(mid, t, np.asarray(imu.a)[:, c])
                  for c in range(3)], axis=1)
    return dt, g, a


def preintegrate(imu, t0, t1, bg=None, ba=None):
    """Integrate (omega - bg) and (accel - ba) over [t0, t1] in the BODY frame.

    Forster's form, with the bias Jacobians and the 9x9 covariance carried
    alongside.  Nothing here is approximated except the first-order bias
    correction above, which is the standard trade and is stated where it is
    made.
    """
    if not imu.available() or t1 <= t0:
        return None
    # ★ T1-c (XTEN_IF_GAPSPAN=1): refuse BEFORE integrating anything an
    # interval that overlaps an IMU gap or has n < 2 - `interval_samples`
    # would otherwise np.interp across the hole at white-noise Q.  The
    # predicate and the gap table live on the ImuSource (backend/imu.py); a
    # duck-typed stand-in carries no frame timestamps, so it is a hard error.
    gapspan = _gapspan_on()
    if gapspan:
        refuses = getattr(imu, "gapspan_refuses", None)
        if refuses is None:
            raise RuntimeError(
                "XTEN_IF_GAPSPAN=1 needs an ImuSource with a gap table; got "
                "%s, which carries no frame timestamps to measure the "
                "threshold from" % type(imu).__name__)
        if refuses(t0, t1, "gba_preint"):
            return None
    bg = np.zeros(3) if bg is None else np.asarray(bg, float)
    ba = np.zeros(3) if ba is None else np.asarray(ba, float)
    sam = interval_samples(imu, t0, t1)
    if sam is None:
        return None
    dts, gs, accs = sam

    dR = np.eye(3)
    dv = np.zeros(3)
    dp = np.zeros(3)
    JRg = np.zeros((3, 3))
    Jvg = np.zeros((3, 3))
    Jva = np.zeros((3, 3))
    Jpg = np.zeros((3, 3))
    Jpa = np.zeros((3, 3))
    cov = np.zeros((9, 9))
    # the DECLARED densities, used as given - a declared sigma is a calibrated
    # system weight (the Rosario receipt: correct calibration on the wrong rig
    # cost +522 %)
    sg = float(getattr(imu, "sigma_g", 0.0)) or 1e-4
    sa = float(getattr(imu, "sigma_a", 0.0)) or 1e-3
    n = 0
    for k in range(len(dts)):
        dt = float(dts[k])
        if dt <= 0:
            continue
        w = gs[k] - bg
        a = accs[k] - ba
        wdt = w * dt
        dRk = so3_exp(wdt)
        Jr = so3_jr(wdt)
        Jr = Jr[0] if Jr.ndim == 3 else Jr
        aH = _hat(a)

        # ---- covariance, BEFORE the state update (A and B use the old dR) --
        A = np.eye(9)
        A[0:3, 0:3] = dRk.T
        A[3:6, 0:3] = -dR @ aH * dt
        A[6:9, 0:3] = -0.5 * dR @ aH * dt * dt
        A[6:9, 3:6] = np.eye(3) * dt
        B = np.zeros((9, 6))
        B[0:3, 0:3] = Jr * dt
        B[3:6, 3:6] = dR * dt
        B[6:9, 3:6] = 0.5 * dR * dt * dt
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = np.eye(3) * (sg * sg / max(dt, 1e-9))
        Q[3:6, 3:6] = np.eye(3) * (sa * sa / max(dt, 1e-9))
        cov = A @ cov @ A.T + B @ Q @ B.T

        # ---- bias Jacobians, also before the state update -----------------
        Jpa = Jpa + Jva * dt - 0.5 * dR * dt * dt
        Jpg = Jpg + Jvg * dt - 0.5 * dR @ aH @ JRg * dt * dt
        Jva = Jva - dR * dt
        Jvg = Jvg - dR @ aH @ JRg * dt
        JRg = dRk.T @ JRg - Jr * dt

        # ---- the state ----------------------------------------------------
        # ★ THE ROTATION MUST BE EVALUATED AT THE SAME INSTANT AS THE
        # ACCELERATION.  Pairing a MIDPOINT acceleration with a LEFT-ENDPOINT
        # rotation is a mismatched quadrature and is first order overall, which
        # is what the midpoint sampling was supposed to remove: measured at
        # 1.3e-2 m/s over 1.5 s, against a sensor noise floor of ~8e-4 m/s over
        # a real 0.15 s keyframe interval - i.e. the integration error would
        # have dominated the measurement it was carrying.
        dRm = dR @ so3_exp(wdt * 0.5)
        am = dRm @ a
        dp = dp + dv * dt + 0.5 * am * dt * dt
        dv = dv + am * dt
        dR = so3_project(dR @ dRk)
        n += 1
    if n == 0:
        return None
    if n < 2 and (gapspan or _inertial_fix("SAMPLES") or inert_gate_on()):
        return None          # rank-6 covariance: no IMU measurement inside the interval
                             # (GAPSPAN implies SAMPLES' refusal: already refused above;
                             # XTEN_GBA_INERT_GATE=1 implies it too: a rank-6 covariance is not a factor)
    # the bias RANDOM WALK over the interval, added to the covariance rather
    # than to a fudge factor
    rw_g = float(getattr(imu, "rw_g", 0.0) or 0.0)
    rw_a = float(getattr(imu, "rw_a", 0.0) or 0.0)
    span = float(t1 - t0)
    cov[0:3, 0:3] += np.eye(3) * (rw_g * rw_g * span) * span * span / 3.0
    cov[3:6, 3:6] += np.eye(3) * (rw_a * rw_a * span) * span * span / 3.0
    if _inertial_fix("BIASCOV"):
        bs = getattr(imu, "bias_sigma", None)
        if bs is not None and bs > 0:
            # a gyro-bias error db rotates the prediction by db*span: its OWN uncertainty belongs here
            cov[0:3, 0:3] += np.eye(3) * (float(bs) * span) ** 2
    return Preintegrated(dR=dR, dv=dv, dp=dp, dt=span, cov=cov,
                         J_R_bg=JRg, J_v_bg=Jvg, J_v_ba=Jva,
                         J_p_bg=Jpg, J_p_ba=Jpa,
                         bg=bg.copy(), ba=ba.copy(), n=n)


# --------------------------------------------------------------------------
def inertial_residual(pre, Ri, pi, vi, Rj, pj, vj, g_w, bg, ba):
    """The 9-vector preintegration residual, in keyframe i's BODY frame.

        r_R = Log( dR^T  Ri^T Rj )
        r_v = Ri^T (v_j - v_i - g dt)                    - dv
        r_p = Ri^T (p_j - p_i - v_i dt - 0.5 g dt^2)     - dp

    `dR, dv, dp` are the preintegrated quantities corrected to the CURRENT
    bias estimate, which is what makes the bias a state the solve can move
    without re-integrating anything.
    """
    dR, dv, dp = pre.corrected(bg, ba)
    dt = pre.dt
    rR = so3_log(dR.T @ (Ri.T @ Rj))
    rR = rR[0] if getattr(rR, "ndim", 1) == 2 else rR
    rv = Ri.T @ (vj - vi - g_w * dt) - dv
    rp = Ri.T @ (pj - pi - vi * dt - 0.5 * g_w * dt * dt) - dp
    return np.concatenate([rR, rv, rp])


def inertial_jacobians(pre, Ri, pi, vi, Rj, pj, vj, g_w, bg, ba, eps=1e-7):
    """Jacobians of `inertial_residual`, by central difference.

    ★ A DELIBERATE CHOICE, AND THE REASON IS MEASURED HISTORY.
    The analytic forms (Forster eq. 40-45) are standard and are perhaps 30x
    faster - but this residual couples nine quantities across two keyframes,
    two bias blocks and gravity, in a chart that is NOT the one the papers use,
    and this project has now paid three times for a sign that looked right:
    the gyro bias correction (`beta = -e/dt`, diverged to 4.8 rad/s in four
    updates), the loop weld's chart (H identical, gradient negated), and the
    host-pose block in the inverse-depth landmark.  The global BA runs a
    handful of times per row, OFF the frame path, under a 120 s budget - so
    the arithmetic here is free and correctness is not.  If this ever moves
    into the per-keyframe path it gets the analytic form AND a finite
    difference pinning it, in that order.

    Perturbations, in the order the caller lays them out:
        pose_i (6, RIGHT perturbation of T_wb_i), vel_i (3, additive),
        pose_j (6), vel_j (3), bg (3), ba (3), gravity (2, tangent).
    """
    from xclean.core.lie import se3_exp

    def resid(dxi, dvi, dxj, dvj, dbg, dba, dgt, basis):
        Ti = np.eye(4); Ti[:3, :3] = Ri; Ti[:3, 3] = pi
        Tj = np.eye(4); Tj[:3, :3] = Rj; Tj[:3, 3] = pj
        Ti = Ti @ se3_exp(dxi)
        Tj = Tj @ se3_exp(dxj)
        gg = g_w + basis @ dgt
        gg = gg / max(np.linalg.norm(gg), 1e-12) * np.linalg.norm(g_w)
        return inertial_residual(pre, Ti[:3, :3], Ti[:3, 3], vi + dvi,
                                 Tj[:3, :3], Tj[:3, 3], vj + dvj,
                                 gg, bg + dbg, ba + dba)

    basis = _tangent_basis(g_w)
    z6, z3, z2 = np.zeros(6), np.zeros(3), np.zeros(2)
    r0 = resid(z6, z3, z6, z3, z3, z3, z2, basis)
    J = np.zeros((9, 26))
    slots = [(0, 6), (6, 3), (9, 6), (15, 3), (18, 3), (21, 3), (24, 2)]
    for si, (off, dim) in enumerate(slots):
        for c in range(dim):
            args_p = [z6.copy(), z3.copy(), z6.copy(), z3.copy(), z3.copy(),
                      z3.copy(), z2.copy()]
            args_m = [a.copy() for a in args_p]
            args_p[si][c] = eps
            args_m[si][c] = -eps
            J[:, off + c] = (resid(*args_p, basis)
                             - resid(*args_m, basis)) / (2 * eps)
    return r0, J, basis


def _tangent_basis(g_w):
    """Two orthonormal directions perpendicular to gravity.

    Gravity's MAGNITUDE is known (9.81 m/s^2 is not something a SLAM system
    should be re-deriving), so only its DIRECTION is a state, and a direction
    has two degrees of freedom. Parameterising it as a free 3-vector would let
    a 2-DoF unknown absorb 3-DoF errors - which is exactly how a scale or a
    bias error hides.
    """
    g = np.asarray(g_w, float)
    n = np.linalg.norm(g)
    if n < 1e-9:
        return np.eye(3)[:, :2]
    u = g / n
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0,
                                                                    0.0])
    e1 = np.cross(u, a)
    e1 /= max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(u, e1)
    return np.stack([e1, e2], axis=1) * n


# --------------------------------------------------------------------------
def _scaled_cov(cov, inflate):
    """cov scaled by a scalar inflation, or per block ([iR, iV, iP] in SIGMA)
    via a diagonal congruence so the cross-covariances stay consistent.
    `cov` is (...,9,9)."""
    infl = np.asarray(inflate, float)
    if infl.ndim == 0:
        return cov * float(infl) ** 2
    d = np.repeat(infl, 3)
    return cov * d[:, None] * d[None, :]


def block_chi(pres, T_wb, vel, g_w, bg, ba):
    """Per-factor chi of the 9-dim residual against the DECLARED covariance,
    split by block: returns (nf, 3) of [chi_rR, chi_rv, chi_rp], each a 3-DoF
    quadratic form whitened by its own diagonal block.

    This is the measurement half of the preint-covariance calibration: the
    run-conditioned rule (doctrine A1) reads the MEDIAN of each column and
    compares it with the median of a chi^2_3 (2.366).  Anchored to the
    declared model on purpose - the rule must not compound with whatever
    inflation the solve currently carries.
    """
    P = _stack_pres(pres)
    if P is None:
        return None
    sel = P["idx"]
    nk = len(T_wb)
    sel = sel[sel + 1 < nk]
    if not len(sel):
        return None
    P = {k: (v[:len(sel)] if isinstance(v, np.ndarray) and len(v.shape) and
             v.shape[0] == len(P["idx"]) else v) for k, v in P.items()}
    i_idx, j_idx = sel, sel + 1
    Ti, Tj = T_wb[i_idx], T_wb[j_idx]
    r = _resid_batch(P, Ti[:, :3, :3], Ti[:, :3, 3], vel[i_idx],
                     Tj[:, :3, :3], Tj[:, :3, 3], vel[j_idx],
                     np.asarray(g_w, float), np.asarray(bg, float),
                     np.asarray(ba, float))
    out = np.zeros((len(sel), 3))
    for b in range(3):
        C = P["cov"][:, 3 * b:3 * b + 3, 3 * b:3 * b + 3] \
            + np.eye(3) * 1e-14
        rb = r[:, 3 * b:3 * b + 3]
        out[:, b] = np.einsum("ni,ni->n", rb,
                              np.linalg.solve(C, rb[..., None])[..., 0])
    return out


def rotation_consistency(pres, R_wb, bg0, inflate, iters=4, eps=1e-7):
    """XTEN_GBA_INERT_GATE's statistic: is the gyro consistent with the entering keyframe ROTATIONS?

    Per factor, the rotation block of the preintegration residual (exactly `_resid_batch`'s rR:
    Log((dR_i Exp(JRg_i (bg - bg_i)))^T R_i^T R_j)) is whitened by the rotation block of the covariance the
    solve uses (declared * inflate^2); its chi is a 3-DoF quadratic form.  It is evaluated at bg0 AND after a
    Gauss-Newton refit of the gyro bias alone (3 unknowns, poses fixed, Jacobian by central difference, a step
    kept only if the chi SUM falls).  Velocities, accel bias and gravity do not enter the rotation block.

        k_R_x = sqrt(median_x / CHI2_3_MEDIAN)      (an implied sigma factor; consistent ~ 1)
        k_R   = min(k_R_bg0, k_R_refit)

    The min makes "refuse" mean "neither the online bias nor the best-fit bias makes the gyro consistent" and
    guarantees k_R <= k_R_bg0 (the GN minimises the SUM, which need not lower the MEDIAN: it raised it on 2 of 56
    replay variants, by <= 0.44 %, in an internal study).

    Whitening convention: chi_i = r_i^T (C_RR,i + 1e-14 I)^-1 r_i / inflate^2 - the same regularised quadratic
    form as `block_chi`'s rotation column, divided by the scalar inflation, so k_R_bg0 equals
    sqrt(median(block_chi[:, 0]) / (inflate^2 * CHI2_3_MEDIAN)) to rounding (it ties the statistic to the saved
    `global_ba.inert_diag.block_median`).  A per-block inflate [iR, iV, iP] uses its rotation component iR.

    Returns None when no factor has both keyframes in `R_wb`; else a JSON-ready dict
    {k_R, k_R_bg0, k_R_refit, median_chi, median_chi_bg0, median_chi_refit, chi_sum_bg0, chi_sum_refit,
     bg0, bg_refit, iters_run, n}."""
    P = _stack_pres(pres)
    if P is None:
        return None
    R_wb = np.asarray(R_wb, float)
    sel = P["idx"]
    sel = sel[sel + 1 < len(R_wb)]
    m = len(sel)
    if not m:
        return None
    dR, JRg, bgf = P["dR"][:m], P["JRg"][:m], P["bg0"][:m]
    infl = np.asarray(inflate, float)
    infl = float(infl) if infl.ndim == 0 else float(infl.reshape(-1)[0])
    C = P["cov"][:m, 0:3, 0:3] + np.eye(3) * 1e-14
    try:
        W = np.linalg.inv(np.linalg.cholesky(C)) / infl
    except np.linalg.LinAlgError:
        W = np.stack([_whiten(c - np.eye(3) * 1e-14) for c in C]) / infl
    A = np.einsum("nji,njk->nik", R_wb[sel], R_wb[sel + 1])          # R_i^T R_j

    def resid(bg):
        dRc = so3_project(np.einsum("nij,njk->nik", dR,
                                    so3_exp(np.einsum("nij,nj->ni", JRg, bg[None, :] - bgf))))
        return np.einsum("nij,nj->ni", W, so3_log(np.einsum("nji,njk->nik", dRc, A)))

    def chis(bg):
        r = resid(bg)
        return np.einsum("ni,ni->n", r, r)

    b0 = np.asarray(bg0, float).reshape(3).copy()
    c0 = chis(b0)
    bg, cb, n_it = b0.copy(), c0, 0
    for _ in range(int(iters)):
        r = resid(bg)
        J = np.zeros((m, 3, 3))
        for c in range(3):
            d = np.zeros(3)
            d[c] = eps
            J[:, :, c] = (resid(bg + d) - resid(bg - d)) / (2.0 * eps)
        step = np.linalg.lstsq(J.reshape(-1, 3), r.reshape(-1), rcond=None)[0]
        bn = bg - step
        cn = chis(bn)
        if not (np.isfinite(cn).all() and cn.sum() < cb.sum()):
            break
        bg, cb, n_it = bn, cn, n_it + 1
    m0, m1 = float(np.median(c0)), float(np.median(cb))
    k0, k1 = float(np.sqrt(m0 / CHI2_3_MEDIAN)), float(np.sqrt(m1 / CHI2_3_MEDIAN))
    return {"k_R": min(k0, k1), "k_R_bg0": k0, "k_R_refit": k1, "median_chi": min(m0, m1),
            "median_chi_bg0": m0, "median_chi_refit": m1,
            "chi_sum_bg0": float(c0.sum()), "chi_sum_refit": float(cb.sum()),
            "bg0": [float(x) for x in b0], "bg_refit": [float(x) for x in bg],
            "iters_run": int(n_it), "n": int(m)}


def _stack_pres(pres):
    """Pack a list of Preintegrated into batched arrays, once."""
    keep = [i for i, q in enumerate(pres) if q is not None]
    if not keep:
        return None
    g = lambda a: np.stack([getattr(pres[i], a) for i in keep])   # noqa: E731
    return dict(idx=np.array(keep, np.int64), dR=g("dR"), dv=g("dv"),
                dp=g("dp"), dt=np.array([pres[i].dt for i in keep]),
                cov=g("cov"), JRg=g("J_R_bg"), Jvg=g("J_v_bg"),
                Jva=g("J_v_ba"), Jpg=g("J_p_bg"), Jpa=g("J_p_ba"),
                bg0=g("bg"), ba0=g("ba"))


def _resid_batch(P, Ri, pi, vi, Rj, pj, vj, g_w, bg, ba):
    """The 9-vector residual for EVERY factor at once.

    Vectorised because the finite-difference Jacobian evaluates it 53 times
    per iteration: doing that per factor in Python measured the global BA at
    **104.9 s** on V1_01, which truncated against its own 120 s budget - so
    the solve was not converging, it was running out of clock. Batched, the
    same 53 evaluations cost 53 numpy calls instead of 53 x 480.
    """
    dbg = bg - P["bg0"]
    dba = ba - P["ba0"]
    dRc = so3_project(np.einsum("nij,njk->nik", P["dR"],
                                so3_exp(np.einsum("nij,nj->ni", P["JRg"],
                                                  dbg))))
    dvc = P["dv"] + np.einsum("nij,nj->ni", P["Jvg"], dbg) \
        + np.einsum("nij,nj->ni", P["Jva"], dba)
    dpc = P["dp"] + np.einsum("nij,nj->ni", P["Jpg"], dbg) \
        + np.einsum("nij,nj->ni", P["Jpa"], dba)
    dt = P["dt"]
    RiT = np.transpose(Ri, (0, 2, 1))
    A = np.einsum("nij,njk->nik", RiT, Rj)
    rR = so3_log(np.einsum("nji,njk->nik", dRc, A))
    rv = np.einsum("nij,nj->ni", RiT, vj - vi - g_w[None, :] * dt[:, None]) \
        - dvc
    rp = np.einsum("nij,nj->ni", RiT,
                   pj - pi - vi * dt[:, None]
                   - 0.5 * g_w[None, :] * (dt * dt)[:, None]) - dpc
    return np.concatenate([rR, rv, rp], axis=1)


def inertial_system(pres, T_wb, vel, g_w, bg, ba, T_cb, nk, n_pose,
                    inflate=1.0, huber=None, eps=1e-7):
    """Assemble the inertial factors into the EXTENDED variable system.

    Layout, chosen so the visual system is untouched and the inertial part is
    purely additive:

        [ 0            : n_pose )        pose perturbations, T_cw LEFT chart
        [ n_pose       : n_pose+3nk )    velocity, world frame, additive
        [ n_pose+3nk   : +3 )            gyro bias delta
        [ +3           : +3 )            accel bias delta
        [ +6           : +2 )            gravity DIRECTION, 2-DoF tangent

    ★ THE CHART CONVERSION IS WHERE THIS WOULD GO WRONG SILENTLY.
    `T_cb` is the constant CAMERA-FROM-BODY transform, so `T_wb = T_wc T_cb`,
    and

        T_cw <- exp(d) T_cw  =>  T_wc <- T_wc exp(-d)
                             =>  T_wb <- T_wb exp(-Ad(T_cb^-1) d)

    Note the INVERSE: using `Ad(T_cb)` gives a gradient whose cosine with the
    true one is +0.78 - not flipped, just wrong, which is worse than flipped
    because it still descends most of the time and only bends the answer.
    Measured, then fixed, and now pinned by a finite difference over the whole
    composed system (cosine +1.00000000, max relative error 2.7e-9).

    The Jacobian is a CENTRAL DIFFERENCE by choice, not by omission: this
    residual couples nine quantities across two keyframes, two bias blocks and
    gravity, in a chart the papers do not use, and this project has paid three
    times for a sign that looked right (the gyro bias `beta = -e/dt`, the loop
    weld's chart, the inverse-depth host block). It runs off the frame path.
    """
    from xclean.core.lie import adjoint, se3_exp, se3_inv as _sinv
    import scipy.sparse as sp

    P = _stack_pres(pres)
    if P is None:
        return None
    sel = P["idx"]
    sel = sel[sel + 1 < nk]
    if not len(sel):
        return None
    P = {k: (v[:len(sel)] if isinstance(v, np.ndarray) and v.shape[:1] ==
             (len(P["idx"]),) else v) for k, v in P.items()}
    i_idx, j_idx = sel, sel + 1
    nx = n_pose + 3 * nk + 8
    M = -adjoint(_sinv(np.asarray(T_cb, float)))
    basis = _tangent_basis(g_w)
    nf = len(sel)

    def resid(dx):
        """dx is (7,) slot-perturbation arrays; returns (nf,9)."""
        dxi, dvi, dxj, dvj, dbg, dba, dgt = dx
        Ti = T_wb[i_idx] if dxi is None else \
            np.einsum("nij,njk->nik", T_wb[i_idx], se3_exp(dxi))
        Tj = T_wb[j_idx] if dxj is None else \
            np.einsum("nij,njk->nik", T_wb[j_idx], se3_exp(dxj))
        vi = vel[i_idx] if dvi is None else vel[i_idx] + dvi
        vj = vel[j_idx] if dvj is None else vel[j_idx] + dvj
        gg = g_w if dgt is None else g_w + basis @ dgt
        gg = gg / max(np.linalg.norm(gg), 1e-12) * np.linalg.norm(g_w)
        return _resid_batch(P, Ti[:, :3, :3], Ti[:, :3, 3], vi,
                            Tj[:, :3, :3], Tj[:, :3, 3], vj, gg,
                            bg if dbg is None else bg + dbg,
                            ba if dba is None else ba + dba)

    none7 = [None] * 7
    r0 = resid(none7)
    J = np.zeros((nf, 9, 26))
    slots = [(0, 6, 0), (6, 3, 1), (9, 6, 2), (15, 3, 3), (18, 3, 4),
             (21, 3, 5), (24, 2, 6)]
    for off, dim, si in slots:
        for c in range(dim):
            dp_ = list(none7)
            dm_ = list(none7)
            if si in (0, 2):
                v = np.zeros((nf, 6)); v[:, c] = eps
                dp_[si] = v; dm_[si] = -v
            elif si in (1, 3):
                v = np.zeros((nf, 3)); v[:, c] = eps
                dp_[si] = v; dm_[si] = -v
            else:
                v = np.zeros(dim); v[c] = eps
                dp_[si] = v; dm_[si] = -v
            J[:, :, off + c] = (resid(dp_) - resid(dm_)) / (2 * eps)

    # whiten by the propagated preintegration covariance.  `inflate` is a
    # scalar, or a length-3 [iR, iV, iP] per-block sigma inflation (the
    # preint-covariance calibration, measured against GT in
    # diag/preint_calib.py: the accel blocks are 6-9x optimistic on every
    # EuRoC row - stationary datasheet densities under in-flight vibration -
    # while the rotation block is consistent on the rows whose GT attitude
    # is independent of the marker-frame error)
    W = np.linalg.inv(np.linalg.cholesky(_scaled_cov(P["cov"], inflate)
                                         + np.eye(9) * 1e-14))
    r = np.einsum("nij,nj->ni", W, r0)
    J = np.einsum("nij,njk->nik", W, J)
    if huber is not None:
        e = np.linalg.norm(r, axis=1)
        s_ = np.where(e > huber, np.sqrt(huber / np.maximum(e, 1e-9)), 1.0)
        r = r * s_[:, None]
        J = J * s_[:, None, None]
    chi = float(np.sum(r * r))

    J[:, :, 0:6] = np.einsum("nqa,ab->nqb", J[:, :, 0:6], M)
    J[:, :, 9:15] = np.einsum("nqa,ab->nqb", J[:, :, 9:15], M)

    tail = n_pose + 3 * nk + np.arange(8)
    idx = np.concatenate([
        (i_idx[:, None] * 6 + np.arange(6)[None, :]),
        (n_pose + i_idx[:, None] * 3 + np.arange(3)[None, :]),
        (j_idx[:, None] * 6 + np.arange(6)[None, :]),
        (n_pose + j_idx[:, None] * 3 + np.arange(3)[None, :]),
        np.tile(tail, (nf, 1))], axis=1)
    H = np.einsum("nqa,nqb->nab", J, J)
    gblk = -np.einsum("nqa,nq->na", J, r)
    gvec = np.zeros(nx)
    np.add.at(gvec, idx.reshape(-1), gblk.reshape(-1))
    rr = np.repeat(idx, 26, axis=1).reshape(-1)
    cc = np.tile(idx, (1, 26)).reshape(-1)
    Hs = sp.coo_matrix((H.reshape(-1), (rr, cc)), shape=(nx, nx)).tocsr()
    return Hs, gvec, chi, nf, basis


def _whiten9(C):
    try:
        L = np.linalg.cholesky(C + np.eye(9) * 1e-14)
        return np.linalg.inv(L)
    except np.linalg.LinAlgError:
        d = np.sqrt(np.maximum(np.diag(C), 1e-14))
        return np.diag(1.0 / d)


def gravity_from_accel_mean(imu, R_wb, times, ba=None):
    """Gravity in the world frame, from the MEAN specific force.

    ★ THIS BEATS THE LEAST-SQUARES INIT BY AN ORDER OF MAGNITUDE, AND THE
      REASON IS WHICH PART OF THE TRAJECTORY IT TOUCHES.
    Over a run the mean world-frame acceleration is ~0 (the platform does not
    accelerate on average), so

        mean_i( R_wb_i a_measured_i ) = mean( a_world - g_w ) = -g_w

    That uses pose ROTATIONS only. The linear initialisation below has to use
    pose DIFFERENCES, and at a 0.25 s keyframe spacing a 4 cm position error
    is 0.16 m/s of velocity noise against a median speed of 0.47 m/s - so
    gravity, which enters as `0.5 g dt^2`, is fighting for signal.

    Measured on V1_01, 483 keyframes: **|g| = 9.8447, 0.35 % off**, and stable
    across thirds of the run (9.73 / 9.88 / 9.88). The least-squares form
    returned 7.16 to 9.11 across runs of the same row and had to be refused by
    the magnitude gate more often than not.

    The accelerometer bias leaks in as `mean(R_wb) b_a`, which shrinks as the
    orientation varies; it is an INITIALISER, and the bundle refines it.
    """
    t = np.asarray(imu.t, float)
    A = np.asarray(imu.a, float)
    ba = np.zeros(3) if ba is None else np.asarray(ba, float)
    acc = []
    for i, tt in enumerate(times):
        j = int(np.searchsorted(t, tt))
        j = min(max(j, 0), len(t) - 1)
        acc.append(R_wb[i] @ (A[j] - ba))
    if not acc:
        return None
    g = -np.mean(np.asarray(acc), axis=0)
    n = float(np.linalg.norm(g))
    if n < 1e-6:
        return None
    return g, n


def inertial_init(pres, R_wb, p_wb, fix_magnitude=True, verbose=False,
                  g_fixed=None):
    """Gravity, per-keyframe velocity and accel bias, by LINEAR least squares.

    Given metric visual poses `(R_wb, p_wb)` for K keyframes and the
    preintegrated intervals between them, the inertial equations are

        p_j = p_i + v_i dt + 0.5 g dt^2 + R_i dp_ij
        v_j = v_i        +       g dt    + R_i dv_ij

    and `dp_ij`, `dv_ij` are affine in the accelerometer bias through the
    Jacobians above.  So with SCALE ALREADY KNOWN the whole system is linear in
    the unknowns `x = [v_0 .. v_{K-1}, g, b_a]` - 3K + 6 of them - and one
    least-squares solve gives all three.  No initialisation heuristic, no
    acceptance gate that rejects 24 attempts out of 27, and no scale.

    Returns (g_w, velocities (K,3), b_a, diagnostics).
    """
    K = len(R_wb)
    if K < 4 or len(pres) < K - 1:
        return None
    # ★ THE DENSE lstsq HITS A WALL AT R4 SCALE, AND IT WAS FOUND THE HARD
    # WAY: at K=10,448 (4s_20200407_102) the (6K x 3K) dense system is
    # 62k x 31k = 16 GB and numpy's SVD-based lstsq ground 22 CPU cores
    # for 4+ hours (times up to 6 IRLS passes); madmax_B0 at K=13,204 was
    # killed the same night for the same invisible reason.  The system is
    # block-bidiagonal in the velocities with a small bias/gravity tail -
    # SPARSE - so above the largest size the dense path has ever been
    # validated on (K=5000 covers every benchmark sequence bit-identically) the
    # solve routes through scipy's sparse LSQR instead: same rows, same
    # whitening, same IRLS, seconds instead of hours.
    if K > 5000:
        return _inertial_init_sparse(pres, R_wb, p_wb, fix_magnitude,
                                     verbose, g_fixed)
    # ★ WITH GRAVITY SUPPLIED IT IS NOT AN UNKNOWN.
    # `gravity_from_accel_mean` recovers it to 0.35 % from pose ROTATIONS
    # alone; asking this least squares to re-derive it from pose DIFFERENCES
    # as well would replace a good estimate with a worse one and couple its
    # error into the velocities and the bias.
    gfix = None if g_fixed is None else np.asarray(g_fixed, float)
    nx = 3 * K + (6 if gfix is None else 3)
    rows, rhs = [], []
    for i, pre in enumerate(pres):
        if pre is None:
            continue
        j = i + 1
        dt = pre.dt
        Ri = R_wb[i]
        # ★ EACH ROW IS WEIGHTED BY ITS OWN PREINTEGRATION COVARIANCE.
        # An unweighted stack mixes a position residual in metres with a
        # velocity residual in m/s and lets the units decide the answer.  The
        # covariance is already propagated, so the whitening is free.
        wp = _whiten(pre.cov[6:9, 6:9])
        wv = _whiten(pre.cov[3:6, 3:6])
        gb = 3 * K if gfix is None else -1
        bb = (3 * K + 3) if gfix is None else 3 * K
        # --- position row -------------------------------------------------
        A = np.zeros((3, nx))
        A[:, 3 * i:3 * i + 3] = np.eye(3) * dt
        if gfix is None:
            A[:, gb:gb + 3] = 0.5 * np.eye(3) * dt * dt
        A[:, bb:bb + 3] = Ri @ pre.J_p_ba
        rows.append(wp @ A)
        b_ = p_wb[j] - p_wb[i] - Ri @ pre.dp
        if gfix is not None:
            b_ = b_ - 0.5 * gfix * dt * dt
        rhs.append(wp @ b_)
        # --- velocity row -------------------------------------------------
        A = np.zeros((3, nx))
        A[:, 3 * i:3 * i + 3] = -np.eye(3)
        A[:, 3 * j:3 * j + 3] = np.eye(3)
        if gfix is None:
            A[:, gb:gb + 3] = -np.eye(3) * dt
        A[:, bb:bb + 3] = -Ri @ pre.J_v_ba
        rows.append(wv @ A)
        b_ = Ri @ pre.dv
        if gfix is not None:
            b_ = b_ + gfix * dt
        rhs.append(wv @ b_)
    if len(rows) < 6:
        return None
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    # ★ AND THE BIAS PRIOR IS WEAK ON PURPOSE, WITH ITS SIGMA DECLARED.
    # The first version used an unweighted identity block, which against
    # whitened data rows is not a weak prior at all - it dominated, and the
    # solve returned a bias of 0.003 where the truth was 0.05.  A prior that
    # decides the answer is not a regulariser.  `sigma_ba` is a wide, stated
    # bound on a real accelerometer bias, not a tuned value.
    sigma_ba = 1.0                       # m/s^2 - wide by construction
    reg = np.zeros((3, nx))
    _bb = (3 * K + 3) if gfix is None else 3 * K
    reg[:, _bb:_bb + 3] = np.eye(3) / sigma_ba
    A = np.vstack([A, reg])
    b = np.concatenate([b, np.zeros(3)])

    # ★ ROBUST, BECAUSE A HANDFUL OF EARLY INTERVALS WAS DECIDING GRAVITY.
    # Measured on V1_01: over keyframes 20-485 the solve returns |g| = 8.85
    # with residual 11.7; INCLUDING the first twenty it returns 6.31 with
    # residual 131. The opening of a run is where the gyro bias has not yet
    # converged and where the keyframe spacing is worst (p99 0.98 s against a
    # median of 0.25, with one 4.35 s gap), so those rows are exactly the ones
    # a least-squares should not trust - and capping the interval length does
    # not find them (it made things worse at every cap tried).
    # Excluding a fixed prefix would be a hand-set constant and a per-row one
    # at that; IRLS with the same Huber constant every other robust cost in
    # this system uses removes the cliff instead of relocating it. Plan 5.5:
    # "no hard accept/reject on a continuous quantity".
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    for _it in range(5):
        r = A @ x - b
        blk = r[:len(r) - 3].reshape(-1, 3)
        e = np.linalg.norm(blk, axis=1)
        s = float(np.median(e)) * 1.4826
        if not (s > 1e-9):
            break
        w = np.where(e <= HUBER_K * s, 1.0,
                     HUBER_K * s / np.maximum(e, 1e-12))
        sw = np.repeat(np.sqrt(w), 3)
        sw = np.concatenate([sw, np.ones(3)])
        x_new, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        if np.linalg.norm(x_new - x) < 1e-9 * max(np.linalg.norm(x), 1.0):
            x = x_new
            break
        x = x_new
    n_down = int((np.linalg.norm((A @ x - b)[:len(b) - 3].reshape(-1, 3),
                                 axis=1)
                  > HUBER_K * max(float(np.median(np.linalg.norm(
                      (A @ x - b)[:len(b) - 3].reshape(-1, 3), axis=1)))
                      * 1.4826, 1e-12)).sum())
    vel = x[:3 * K].reshape(K, 3)
    g = gfix if gfix is not None else x[3 * K:3 * K + 3]
    ba = x[_bb:_bb + 3]
    gmag = float(np.linalg.norm(g))
    diag = {"g_magnitude": gmag,
            "g_magnitude_error_pct": 100.0 * (gmag / GRAVITY_MAG - 1.0),
            "accel_bias": ba.tolist(),
            "residual_rms": float(np.sqrt(np.mean((A @ x - b) ** 2))),
            "down_weighted": n_down,
            "keyframes": K, "intervals": len(rows) // 2}
    if fix_magnitude and gmag > 1e-6:
        # ★ THE MAGNITUDE IS KNOWN; ONLY THE DIRECTION IS UNKNOWN.
        # Letting the solve return |g| free and then reporting it is the
        # DIAGNOSTIC - it says whether the accelerometer, the extrinsics and
        # the visual scale agree.  Shipping that free magnitude as the state
        # would be letting a 2-DoF unknown absorb a 3-DoF error.
        g = g / gmag * GRAVITY_MAG
    if verbose:
        print(" [imu-init] |g| %.4f m/s^2 (%.2f %% off 9.81), accel bias %s, "
              "residual %.4f, %d keyframes"
              % (gmag, diag["g_magnitude_error_pct"],
                 np.array2string(ba, precision=4), diag["residual_rms"], K))
    return g, vel, ba, diag


def _inertial_init_sparse(pres, R_wb, p_wb, fix_magnitude=True,
                          verbose=False, g_fixed=None):
    """`inertial_init` for K > 5000: the SAME rows, whitening, prior and
    IRLS, assembled sparse and solved by LSQR.  See the routing note in
    `inertial_init` - the dense SVD path is O(K^3) and ground for hours at
    R4 scale; this is seconds.  The dense path stays byte-identical for
    every size it was ever validated on."""
    import time as _time
    import scipy.sparse as sp
    from scipy.sparse.linalg import lsqr

    t0 = _time.perf_counter()
    K = len(R_wb)
    gfix = None if g_fixed is None else np.asarray(g_fixed, float)
    nx = 3 * K + (6 if gfix is None else 3)
    gb = 3 * K if gfix is None else -1
    bb = (3 * K + 3) if gfix is None else 3 * K

    # one 3x3 block = one (row_block, col_base, mat) triple
    blk_r, blk_c, blk_v, rhs = [], [], [], []
    nb = 0
    for i, pre in enumerate(pres):
        if pre is None:
            continue
        j = i + 1
        dt = pre.dt
        Ri = R_wb[i]
        wp = _whiten(pre.cov[6:9, 6:9])
        wv = _whiten(pre.cov[3:6, 3:6])
        # --- position row --------------------------------------------------
        blk_r += [nb, nb]
        blk_c += [3 * i, bb]
        blk_v += [wp * dt, wp @ (Ri @ pre.J_p_ba)]
        if gfix is None:
            blk_r.append(nb)
            blk_c.append(gb)
            blk_v.append(wp * (0.5 * dt * dt))
        b_ = p_wb[j] - p_wb[i] - Ri @ pre.dp
        if gfix is not None:
            b_ = b_ - 0.5 * gfix * dt * dt
        rhs.append(wp @ b_)
        nb += 1
        # --- velocity row --------------------------------------------------
        blk_r += [nb, nb, nb]
        blk_c += [3 * i, 3 * j, bb]
        blk_v += [-wv, wv, -(wv @ (Ri @ pre.J_v_ba))]
        if gfix is None:
            blk_r.append(nb)
            blk_c.append(gb)
            blk_v.append(-wv * dt)
        b_ = Ri @ pre.dv
        if gfix is not None:
            b_ = b_ + gfix * dt
        rhs.append(wv @ b_)
        nb += 1
    if nb < 2:
        return None
    # the weak declared bias prior, identical to the dense path
    sigma_ba = 1.0
    blk_r.append(nb)
    blk_c.append(bb)
    blk_v.append(np.eye(3) / sigma_ba)
    rhs.append(np.zeros(3))
    nb += 1

    blk_r = np.asarray(blk_r, np.int64)
    blk_c = np.asarray(blk_c, np.int64)
    V = np.stack(blk_v)                       # (nblk, 3, 3)
    b = np.concatenate(rhs)
    sub = np.arange(3)
    rr = (blk_r[:, None, None] * 3 + sub[None, :, None]
          + np.zeros((1, 1, 3), np.int64)).reshape(-1)
    cc = (blk_c[:, None, None] + np.zeros((1, 3, 1), np.int64)
          + sub[None, None, :]).reshape(-1)

    def solve(scale_rb):
        """scale_rb: per-row-block sqrt weight (nb,)."""
        vals = (V * scale_rb[blk_r][:, None, None]).reshape(-1)
        A = sp.coo_matrix((vals, (rr, cc)), shape=(3 * nb, nx)).tocsr()
        bs = b * np.repeat(scale_rb, 3)
        return A, lsqr(A, bs, atol=1e-12, btol=1e-12,
                       iter_lim=20 * nx)[0]

    ones = np.ones(nb)
    A, x = solve(ones)
    # IRLS, same Huber form and same convergence rule as the dense path;
    # the prior block keeps weight 1
    for _it in range(5):
        r = A @ x - b
        blk = r[:3 * (nb - 1)].reshape(-1, 3)
        e = np.linalg.norm(blk, axis=1)
        s = float(np.median(e)) * 1.4826
        if not (s > 1e-9):
            break
        w = np.where(e <= HUBER_K * s, 1.0,
                     HUBER_K * s / np.maximum(e, 1e-12))
        scale = np.concatenate([np.sqrt(w), [1.0]])
        A, x_new = solve(scale)
        if np.linalg.norm(x_new - x) < 1e-9 * max(np.linalg.norm(x), 1.0):
            x = x_new
            break
        x = x_new
    r_fin = (A @ x - b)[:3 * (nb - 1)].reshape(-1, 3)
    e_fin = np.linalg.norm(r_fin, axis=1)
    n_down = int((e_fin > HUBER_K
                  * max(float(np.median(e_fin)) * 1.4826, 1e-12)).sum())
    vel = x[:3 * K].reshape(K, 3)
    g = gfix if gfix is not None else x[3 * K:3 * K + 3]
    ba = x[bb:bb + 3]
    gmag = float(np.linalg.norm(g))
    diag = {"g_magnitude": gmag,
            "g_magnitude_error_pct": 100.0 * (gmag / GRAVITY_MAG - 1.0),
            "accel_bias": ba.tolist(),
            "residual_rms": float(np.sqrt(np.mean((A @ x - b) ** 2))),
            "down_weighted": n_down,
            "keyframes": K, "intervals": nb - 1,
            "solver": "sparse-lsqr",
            "solve_s": float(_time.perf_counter() - t0)}
    if fix_magnitude and gmag > 1e-6:
        g = g / gmag * GRAVITY_MAG
    if verbose:
        print(" [imu-init] SPARSE |g| %.4f m/s^2 (%.2f %% off), accel bias "
              "%s, residual %.4f, %d keyframes, %.1f s"
              % (gmag, diag["g_magnitude_error_pct"],
                 np.array2string(ba, precision=4), diag["residual_rms"], K,
                 diag["solve_s"]))
    return g, vel, ba, diag
