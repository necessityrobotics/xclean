#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.2 W2: ONLINE CAMERA-IMU TIME-OFFSET (t_d) ESTIMATION AND COMPENSATION.  numpy + numba, CPU only, causal.

    from xclean.vio import td_est as TD
    est = TD.TdProfile(R_bc, cam_period, imu_period)            # one per run: t_d is a property of the recording
    est.step(t_frame, R_wb_vis, valid, segment, imu.t, imu.g, imu.horizon)   # every frame, after publish
    e = est.estimate()                                          # TdEstimate (e.apply = the t_d to compensate with)
    a, b = TD.imu_interval(t_i, t_j, e.apply)                   # the IMU-clock span of camera interval [t_i, t_j]
    pre = imu_factor.preintegrate(t, g, acc, a, b, horizon, bg, ba, noise)   # C40: never reads past `horizon`

SIGN CONVENTION (the defect register R-TD / M-TD and diag/td_survey.py):  t_imu = t_cam + t_d.
    A physical instant stamped t_c by the camera is stamped t_c + t_d by the IMU, so the IMU samples that measure the
    motion between camera stamps t_i and t_j are those in [t_i + t_d, t_j + t_d].  Rosario v1's frames are stamped
    late: t_d < 0 there.  vi_init.offset_align uses the OPPOSITE sign.

DESIGN: option (c) = (b) TdProfile finds,
  decides and initialises t_d globally; (a) TdRefine refines it on the full IMU residual once (b) is resolved.
  INPUT RULE: R_wb_vis must be VISION-ONLY - the frame's PnP attitude while the IMU is admitted
  (the window-solved pose then carries the uncompensated IMU and pulls t_d to 0: rosario_03 -0.1 ms vs -126 ms).
  (b) TdProfile - a causal PROFILE-LIKELIHOOD estimator on a lag grid (no linearisation in t_d):
      per consecutive pair of visually tracked frames (i, j) the vision-only body rotation dR_vis = R_wb,i^T R_wb,j is
      compared with the gyro rotation over the SHIFTED interval [t_i + tau, t_j + tau] for every grid lag tau,
          r_k(tau, bg) = Log( dR_gyro(t_i + tau, t_j + tau; bg0)^T dR_vis ) + dt_k (bg - bg0)        (first order in bg)
      The gyro bias is a NUISANCE PER BLOCK of BLOCK consecutive intervals (profiled out in closed form), so it may drift
      freely between blocks: no bias-stability constant.  Each closed block contributes
          S_b(tau) = sum_k w_k |r_k(tau, bg0)|^2 - |sum_k w_k dt_k r_k(tau, bg0)|^2 / sum_k w_k dt_k^2
      and the run's profile is S(tau) = sum_b S_b(tau) (t_d is modelled as constant per recording, as the register
      measured it).  t_d_hat = argmin S (grid, then a parabola through the minimum and its two neighbours).
      Blocks share no pose: the interval after a full block is skipped (1 in BLOCK + 1), so blocks are independent
      under per-frame pose noise.
      Robust: an interval that NO lag explains is a visual fault, not offset signal; its Huber weight (dof-3 threshold,
      imu_factor.huber_k_dof(3), class D) is taken from min over tau of its whitened block residual, never from the
      residual at the current estimate (that would down-weight exactly the intervals that carry the offset signal).
      Scale (class C): s^2 = S(t_d_hat) / dof, dof = sum_b (3 n_b - 3) - 1.
      UNCERTAINTY: the delete-one-block JACKKNIFE of t_d_hat, sd = sd_jk (robust to the serial correlation of the
      interval residuals - consecutive intervals share a pose, and window-smoothed poses are correlated over the
      window = a block - and to heteroscedastic noise).  The iid-model curvature sd_c = s / sqrt(c) (S ~ S_min +
      c (tau - t)^2) is reported only: on synthetic per-frame attitude noise it was 2-4x conservative, on
      window-smoothed poses its error sign is unknown.  The apply test uses the Student-t quantile with B - 1 dof
      (the jackknife variance's own dof), so a few blocks can never trigger compensation.
      OBSERVABILITY: t_d enters only through CHANGES of the body rate: dr_k/dtau ~ w(t_j) - dR_k^T w(t_i).  A constant
      rate (or no rotation) is indistinguishable from the per-block bias, so the information is
      I = sum_k w_k |w_j - dR_k^T w_i|^2 / s^2 (rotation excitation = angular acceleration).  The estimator states
      `resolved` only when the profile is IDENTIFIED (its contrast, median S - S_min, exceeds z_id = Phi^-1(1 -
      (1 - P_GATE) / L) times its no-signal fluctuation 2 s^2 sqrt(dof max(1, F)): without rate changes S is flat up to
      chi-square noise), the minimum is interior to the grid, unique (no second basin within s^2 max(1, F)
      chi2_1(P_GATE) of the minimum, F = (sd_jk / sd_c)^2 the iid model's variance inflation) and at least two blocks
      exist; it is APPLIED (e.apply != 0) only when resolved AND t_d = 0 is rejected at P_GATE (|t_d_hat| > q sd,
      q = t_{B-1}^-1(1 - (1 - P_GATE) / 2)); otherwise e.apply = 0 (no compensation).
  (a) imu_td_jacobian - t_d as a window state: the derivative of the 15-row IMU residual wrt t_d (both interval ends
      move by +dtau), analytic, from the bias-corrected rates / specific forces at the shifted ends:
          dR <- dR Exp(dtau (w1 - dR^T w0)),  dv <- dv + dtau (dR a1 - a0 - [w0]x dv),  dp <- dp + dtau (dv - a0 dt - [w0]x dp)
      (the IMU-side form of Li & Mourikis, IJRR 2014 and of VINS-Mono's t_d, which shifts the features instead; our
      own derivation and code).  td_schur_step gives the window's Gauss-Newton t_d increment and its marginal
      information (the Schur complement of the window system).  Linearised: valid only near the truth (from 0 it reads
      -7.5 ms on rosario_04 (-65) and +3.8 ms on rosario_03 (-130)), hence (b) initialises / decides.
      TdRefine accumulates the per-keyframe steps as absolute estimates (td_lin + dtd) weighted by their information,
      with a block-jackknife sd (the steps' own information was 10-25x optimistic) and the same P_GATE test.

CONSTANTS
  BLOCK = 10        class B: the engine's window length SCALE_WINDOW = N_KF + N_FR (the span over which window-smoothed
                    pose errors are correlated; blocks are the jackknife's units and the bias nuisance's support).
  lag range         class B: +- (BLOCK / 2) camera periods (the shifted gyro span still overlaps more than half of the
                    block's own time support); an estimate at the grid edge is reported `boundary`, never applied.
  lag step          class B: max(IMU period (A), camera period / BLOCK): the profile's curvature scale is the motion
                    bandwidth (bounded by the camera rate); the parabola interpolates inside a step.  <= BLOCK^2 + 1 lags.
  gap rule          class B: an interval whose shifted gyro span (over all lags) holds an IMU sample gap longer than one
                    camera period is skipped (its rotation is not measured; Rosario R-GAP).
  P_GATE = 0.999    class D (= vi_init.P_GATE = imu_init_watch.P_GATE).  Huber: imu_factor.huber_k_dof(3) (class D).
Nothing is carried from a dataset measurement; the register offsets are used only to EVALUATE, never here.

CAUSALITY (C40): every gyro read goes through _rot_lags with n_rel = the released sample count at `horizon`; an
interval is processed only once t_j + max lag <= horizon (no hold is ever used by the estimator).  imu_interval()
shifts a camera interval; imu_factor.preintegrate then holds past the horizon exactly as today (t_d > 0 ends are held
and re-preintegrated by the engine's existing _repreintegrate when released).
"""
from __future__ import annotations

import math
from collections import deque, namedtuple

import numpy as np
from numba import njit

P_GATE = 0.999                       # class D (= vi_init.P_GATE = imu_init_watch.P_GATE; test asserts equality)
BLOCK = 10                           # class B (= engine_live.SCALE_WINDOW = N_KF + N_FR)
BREAK = None                         # marker in the pending queue: the visual chain is broken here

TdEstimate = namedtuple("TdEstimate", "td sd sd_curv sd_jk apply resolved boundary ambiguous n_blocks n_int dof s "
                                      "excitation info t_last q contrast z_id")
TdEstimate.__doc__ = ("td [s] (register sign, t_imu = t_cam + td; nan before any block), sd = sd_jk (delete-one-block "
                      "jackknife; sd_curv = the iid-model curvature sd, diagnostic), "
                      "apply = the td to compensate with (0 unless resolved and |td| > q sd), resolved, "
                      "boundary (minimum at the grid edge), ambiguous (a second basin within s^2 chi2_1(P_GATE)), "
                      "n_blocks, n_int (intervals used), dof, s (rad, residual scale per interval), excitation "
                      "(rms change of the visual mean body rate between consecutive intervals, rad/s; diagnostic), "
                      "info (= 1 / sd_curv^2, s^-2), t_last (newest "
                      "camera stamp used), q = the Student-t quantile t_{B-1}(1 - (1 - P_GATE) / 2) of the apply test, contrast = the "
                      "identification statistic (median S - S_min) / (2 s^2 sqrt(dof max(1, F))), z_id = its "
                      "Bonferroni threshold Phi^-1(1 - (1 - P_GATE) / L)")


def _chi2_ppf(p, dof):
    from scipy.stats import chi2
    return float(chi2.ppf(p, dof))


def _t_ppf(p, dof):
    from scipy.stats import t as tdist
    return float(tdist.ppf(p, dof))


# ============================================================================================== numba kernels ======
@njit(cache=True, nogil=True)
def _exp3(w, R):
    th2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
    th = math.sqrt(th2)
    if th < 1e-8:
        a = 1.0 - th2 / 6.0
        b = 0.5 - th2 / 24.0
    else:
        a = math.sin(th) / th
        b = (1.0 - math.cos(th)) / th2
    x, y, z = w[0], w[1], w[2]
    # R = I + a K + b K^2
    R[0, 0] = 1.0 + b * (-y * y - z * z)
    R[0, 1] = -a * z + b * x * y
    R[0, 2] = a * y + b * x * z
    R[1, 0] = a * z + b * x * y
    R[1, 1] = 1.0 + b * (-x * x - z * z)
    R[1, 2] = -a * x + b * y * z
    R[2, 0] = -a * y + b * x * z
    R[2, 1] = a * x + b * y * z
    R[2, 2] = 1.0 + b * (-x * x - y * y)


@njit(cache=True, nogil=True)
def _log3(R, out):
    c = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) * 0.5
    if c > 1.0:
        c = 1.0
    if c < -1.0:
        c = -1.0
    th = math.acos(c)
    v0 = R[2, 1] - R[1, 2]
    v1 = R[0, 2] - R[2, 0]
    v2 = R[1, 0] - R[0, 1]
    if th < 1e-6:
        f = 0.5
    elif th > math.pi - 1e-4:
        # near pi (relative rotations over one frame interval never get here; kept finite for safety)
        B00 = R[0, 0]
        B11 = R[1, 1]
        B22 = R[2, 2]
        i = 0
        if B11 > B00:
            i = 1
        if B22 > max(B00, B11):
            i = 2
        ax = np.zeros(3)
        den = math.sqrt(max((R[i, i] - c) / (1.0 - c), 0.0))
        for j in range(3):
            ax[j] = 0.5 * (R[j, i] + R[i, j]) / max(den * (1.0 - c), 1e-12)
        ax[i] = den
        n = math.sqrt(ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2)
        s = 1.0 if ax[0] * v0 + ax[1] * v1 + ax[2] * v2 >= 0 else -1.0
        for j in range(3):
            out[j] = s * th * ax[j] / n
        return
    else:
        f = th / (2.0 * math.sin(th))
    out[0] = f * v0
    out[1] = f * v1
    out[2] = f * v2


@njit(cache=True, nogil=True)
def _rot_over(t, g, n_rel, a, b, bg, R):
    """dR over [a, b] at bias bg with imu_factor.preintegrate's rule: sub-interval edges at a, every RELEASED sample
    strictly inside, min(b, t_end) and b; on each sub-interval the LINEAR INTERPOLATION of the two bracketing released
    samples at its MIDPOINT, R <- R Exp((w - bg) h); beyond t_end (= t[n_rel - 1]) the last released sample is held.
    Returns 0 ok, 1 bad interval, 2 a precedes the first sample (nothing measured the start), 3 no released sample."""
    for r in range(3):
        for c in range(3):
            R[r, c] = 1.0 if r == c else 0.0
    if n_rel < 1:
        return 3
    if not (b > a):
        return 1
    if a < t[0]:
        return 2
    # first released sample strictly after a
    lo = 0
    hi = n_rel
    while lo < hi:
        mid = (lo + hi) // 2
        if t[mid] <= a:
            lo = mid + 1
        else:
            hi = mid
    i = lo
    cur = a
    w = np.empty(3)
    E = np.empty((3, 3))
    T = np.empty((3, 3))
    while cur < b:
        if i < n_rel and t[i] < b:
            nxt = t[i]
        else:
            nxt = b
        h = nxt - cur
        if h > 0.0:
            m = 0.5 * (cur + nxt)
            if i >= n_rel:                       # past the last released sample: hold it (C40)
                for k in range(3):
                    w[k] = g[n_rel - 1, k] - bg[k]
            else:
                t0 = t[i - 1]
                t1 = t[i]
                u = (m - t0) / (t1 - t0) if t1 > t0 else 1.0
                for k in range(3):
                    w[k] = (1.0 - u) * g[i - 1, k] + u * g[i, k] - bg[k]
            for k in range(3):
                w[k] *= h
            _exp3(w, E)
            for r in range(3):
                for c in range(3):
                    T[r, c] = R[r, 0] * E[0, c] + R[r, 1] * E[1, c] + R[r, 2] * E[2, c]
            for r in range(3):
                for c in range(3):
                    R[r, c] = T[r, c]
        cur = nxt
        if i < n_rel and nxt == t[i]:
            i += 1
        elif nxt >= b:
            break
    return 0


@njit(cache=True, nogil=True)
def _rot_lags(t, g, n_rel, ta, tb, lags, bg, dRvis, out):
    """out[m] = Log(dR_gyro(ta + lags[m], tb + lags[m]; bg)^T dRvis) for every lag; returns the worst status."""
    R = np.empty((3, 3))
    M = np.empty((3, 3))
    v = np.empty(3)
    worst = 0
    for m in range(lags.shape[0]):
        st = _rot_over(t, g, n_rel, ta + lags[m], tb + lags[m], bg, R)
        if st != 0:
            worst = max(worst, st)
            for k in range(3):
                out[m, k] = np.nan
            continue
        for r in range(3):
            for c in range(3):
                M[r, c] = R[0, r] * dRvis[0, c] + R[1, r] * dRvis[1, c] + R[2, r] * dRvis[2, c]
        _log3(M, v)
        for k in range(3):
            out[m, k] = v[k]
    return worst


@njit(cache=True, nogil=True)
def _max_gap(t, n_rel, a, b):
    """The longest spacing between consecutive released samples whose span touches [a, b] (incl. the edges)."""
    lo = 0
    hi = n_rel
    while lo < hi:
        mid = (lo + hi) // 2
        if t[mid] <= a:
            lo = mid + 1
        else:
            hi = mid
    i = max(lo - 1, 0)
    gmax = 0.0
    while i + 1 < n_rel and t[i] < b:
        d = t[i + 1] - t[i]
        if d > gmax:
            gmax = d
        i += 1
    return gmax


def rot_over(t, gyro, a, b, horizon, bg=None):
    """Python wrapper: the gyro rotation matrix over [a, b] (C40: samples with t <= horizon only).  (R, status)."""
    t = np.ascontiguousarray(t, np.float64)
    g = np.ascontiguousarray(gyro, np.float64)
    n_rel = int(np.searchsorted(t, np.inf if horizon is None else float(horizon), side="right"))
    R = np.empty((3, 3))
    st = _rot_over(t, g, n_rel, float(a), float(b), np.zeros(3) if bg is None else np.asarray(bg, float), R)
    return R, int(st)


# ============================================================================================== compensation =======
def imu_interval(t_i, t_j, td):
    """The IMU-clock span of camera interval [t_i, t_j] under offset td (register sign): [t_i + td, t_j + td].
    Causality is imu_factor.preintegrate's: samples past the horizon are never read (a td > 0 end is HELD and the
    record flagged `extrapolated`; the engine's _repreintegrate redoes it once released).  A start before the first
    IMU sample is refused by preintegrate (REFUSED_BEFORE_FIRST_SAMPLE): apply != 0 only after >= 2 closed blocks, so
    t_i + td >= t_first holds in practice; the engine treats a refusal as a failed propagation (as today)."""
    return float(t_i) + float(td), float(t_j) + float(td)


def lag_grid(cam_period, imu_period, block=BLOCK):
    """The lag grid (class B, module docstring): step max(imu_period, cam_period / block), range +- block/2 cam periods."""
    cam_period = float(cam_period)
    imu_period = float(imu_period)
    if not (cam_period > 0 and imu_period > 0):
        raise ValueError("lag_grid needs positive camera and IMU periods")
    h = max(imu_period, cam_period / block)
    R = 0.5 * block * cam_period
    M = int(math.floor(R / h + 1e-9))
    return h * np.arange(-M, M + 1, dtype=np.float64)


# ============================================================================================== (b) the profile ====
class TdProfile:
    """Causal profile-likelihood estimator of t_d (module docstring (b)).  Feed every frame with step(); query
    estimate() whenever needed (cost B x L, B = closed blocks, L <= BLOCK^2 + 1 lags)."""

    def __init__(self, R_bc, cam_period, imu_period, block=BLOCK, bg0=None, p=P_GATE):
        self.R_bc = np.asarray(R_bc, float).reshape(3, 3)
        self.cam_period = float(cam_period)
        self.imu_period = float(imu_period)
        self.block = int(block)
        self.p = float(p)
        self.lags = lag_grid(cam_period, imu_period, block)
        self.h = float(self.lags[1] - self.lags[0]) if len(self.lags) > 1 else float("nan")
        self.L = len(self.lags)
        self.bg0 = np.zeros(3) if bg0 is None else np.asarray(bg0, float).reshape(3)
        from xclean.vio.imu_factor import huber_k_dof
        self.k_hub = huber_k_dof(3)
        self.chi2_1 = _chi2_ppf(p, 1)
        self.z = math.sqrt(self.chi2_1)
        self.med3 = _chi2_ppf(0.5, 3)
        from scipy.stats import norm
        self.z_id = float(norm.ppf(1.0 - (1.0 - self.p) / max(self.L, 1)))   # Bonferroni over the lags (class D)
        self._last = None                     # (t, R_wb, segment) of the previous valid frame
        self._pending = deque()               # intervals (ta, tb, dRvis) and BREAK markers, in stream order
        self._cur = []                        # released intervals of the open block: (dt, r0 (L,3), h_rate (3,))
        self.S = np.zeros(self.L)             # sum of closed blocks' profiles
        self._Sb = np.zeros((64, self.L))     # per-block profiles (jackknife)
        self.n_blocks = 0
        self.n_int = 0
        self.dof = 0.0
        self.skipped = {"gap": 0, "status": 0, "nonfinite": 0}
        self._exc = 0.0                       # sum w |h|^2 (excitation, rad^2/s^2)
        self._exc_n = 0.0
        self._s2_run = None                   # class-C scale used for the robust weights
        self._skip_next = False               # drop the interval that would share a full block's last pose
        self.t_last = float("nan")

    # ---------------------------------------------------------------------------------------- stream
    def step(self, t_frame, R_wb, valid, segment, imu_t, imu_g, horizon):
        """One frame.  R_wb: the frame's VISION-ONLY body attitude (PnP / VO window), valid: tracked (not coasting /
        degraded / init), segment: map segment id.  Then every pending interval whose shifted span is released
        (t_j + max lag <= horizon) is processed (C40)."""
        t_frame = float(t_frame)
        if valid and R_wb is not None:
            R_wb = np.asarray(R_wb, float).reshape(3, 3)
            if self._last is not None and self._last[2] == segment:
                Ri = self._last[1]
                self._pending.append((self._last[0], t_frame, np.ascontiguousarray(Ri.T @ R_wb)))
            elif self._last is not None:
                self._pending.append(BREAK)
            self._last = (t_frame, R_wb, segment)
        else:
            if self._last is not None:
                self._pending.append(BREAK)
            self._last = None
        self.release(imu_t, imu_g, horizon)

    def release(self, imu_t, imu_g, horizon):
        if not self._pending:
            return
        t = np.ascontiguousarray(imu_t, np.float64)
        g = np.ascontiguousarray(imu_g, np.float64)
        hz = float(horizon)
        n_rel = int(np.searchsorted(t, hz, side="right"))
        lmax = float(self.lags[-1])
        lmin = float(self.lags[0])
        while self._pending:
            item = self._pending[0]
            if item is BREAK:
                self._pending.popleft()
                self._close_block()
                self._skip_next = False
                continue
            ta, tb, dRvis = item
            if tb + lmax > hz:
                break                                        # not yet released for every lag (C40)
            self._pending.popleft()
            if self._skip_next:
                # the interval after a FULL block shares that block's last pose: skipped, so blocks share no pose and
                # are independent under per-frame pose noise (the jackknife's assumption)
                self._skip_next = False
                continue
            if _max_gap(t, n_rel, ta + lmin, tb + lmax) > self.cam_period:
                self.skipped["gap"] += 1
                self._close_block()
                continue
            r0 = np.empty((self.L, 3))
            st = _rot_lags(t, g, n_rel, ta, tb, self.lags, self.bg0, dRvis, r0)
            if st != 0:
                self.skipped["status"] += 1
                self._close_block()
                continue
            if not np.all(np.isfinite(r0)):
                self.skipped["nonfinite"] += 1
                self._close_block()
                continue
            # excitation diagnostic: the VISUAL mean body rate of the interval (its changes between consecutive
            # intervals are the angular acceleration that identifies t_d; gyro end-point rates would add vibration)
            lv = np.empty(3)
            _log3(dRvis, lv)
            self._cur.append((tb - ta, r0, lv / (tb - ta)))
            self.t_last = tb
            if len(self._cur) >= self.block:
                self._close_block()
                self._skip_next = True

    # ---------------------------------------------------------------------------------------- blocks
    def _block_fit(self, dt, r0, w):
        """Bias profiled per lag: e[k, m] = r0[k, m] + dt_k dbg_m, dbg_m = -sum w dt r0 / sum w dt^2."""
        A = float(np.sum(w * dt * dt))
        bm = np.einsum("k,kmc->mc", w * dt, r0)              # (L, 3)
        dbg = -bm / A
        e = r0 + dt[:, None, None] * dbg[None, :, :]
        return e, A, bm

    def _close_block(self):
        cur = self._cur
        self._cur = []
        n = len(cur)
        if n < 2:                                            # 3n - 3 = 0 dof: no information after the bias
            return
        dt = np.array([c[0] for c in cur])
        r0 = np.stack([c[1] for c in cur])                   # (n, L, 3)
        wv = np.stack([c[2] for c in cur])
        w = np.ones(n)
        # robust weights: an interval that no lag explains is a fault (IRLS with weights from min over lags)
        s2 = self._s2_run
        for _it in range(3):
            e, A, bm = self._block_fit(dt, r0, w)
            e2min = np.min(np.sum(e * e, axis=2), axis=1)        # (n,)
            if s2 is None:
                s2b = float(np.median(e2min)) / self.med3
                if not (s2b > 0 and np.isfinite(s2b)):
                    break
                s2 = s2b
            en = np.sqrt(e2min / s2)
            w_new = np.where(en <= self.k_hub, 1.0, self.k_hub / np.maximum(en, 1e-300))
            if np.allclose(w_new, w):
                break
            w = w_new
        e, A, bm = self._block_fit(dt, r0, w)
        Sb = np.einsum("k,kmc->m", w, e * e)
        if self.n_blocks >= self._Sb.shape[0]:
            self._Sb = np.concatenate([self._Sb, np.zeros_like(self._Sb)], axis=0)
        self._Sb[self.n_blocks] = Sb
        self.n_blocks += 1
        self.S += Sb
        self.n_int += n
        self.dof += 3.0 * float(np.sum(w)) - 3.0
        dw = np.diff(wv, axis=0)
        wpair = np.minimum(w[1:], w[:-1])
        self._exc += float(np.sum(wpair * np.sum(dw * dw, axis=1)))
        self._exc_n += float(np.sum(wpair))
        # class-C running scale for the next block's weights: the profile's own minimum
        m = int(np.argmin(self.S))
        if self.dof > 1:
            self._s2_run = float(self.S[m] / max(self.dof - 1.0, 1.0))

    def flush(self):
        """Close the open block (end of stream); pending unreleased intervals stay pending."""
        self._close_block()

    # ---------------------------------------------------------------------------------------- estimate
    @staticmethod
    def _parabola(S, m, h, lags):
        """(t_hat, S_min, curvature c) of a parabola through (m-1, m, m+1); edge -> grid point, c = nan."""
        if m <= 0 or m >= len(S) - 1:
            return float(lags[m]), float(S[m]), float("nan")
        a, b, c = float(S[m - 1]), float(S[m]), float(S[m + 1])
        den = a - 2.0 * b + c
        if not den > 0:
            return float(lags[m]), b, float("nan")
        d = 0.5 * (a - c) / den
        return float(lags[m] + d * h), b - 0.25 * (a - c) * d, den / (2.0 * h * h)

    def _argmin_rows(self, M):
        """Vectorised grid argmin + parabola per row of M (B, L) -> t_hat (B,)."""
        m = np.argmin(M, axis=1)
        B = M.shape[0]
        out = self.lags[m].astype(float).copy()
        inner = (m > 0) & (m < self.L - 1)
        if inner.any():
            r = np.nonzero(inner)[0]
            a = M[r, m[r] - 1]
            b = M[r, m[r]]
            c = M[r, m[r] + 1]
            den = a - 2.0 * b + c
            ok = den > 0
            d = np.where(ok, 0.5 * (a - c) / np.where(ok, den, 1.0), 0.0)
            out[r] = self.lags[m[r]] + d * self.h
        return out, m, B

    def estimate(self, with_profile=False):
        nan = float("nan")
        if self.n_blocks == 0 or self.dof <= 1:
            e = TdEstimate(nan, nan, nan, nan, 0.0, False, False, False, self.n_blocks, self.n_int, self.dof, nan,
                           nan, 0.0, self.t_last, math.inf, 0.0, self.z_id)
            return (e, None) if with_profile else e
        S = self.S
        m = int(np.argmin(S))
        td, Smin, c = self._parabola(S, m, self.h, self.lags)
        s2 = max(Smin, 0.0) / max(self.dof - 1.0, 1.0)
        s = math.sqrt(s2)
        boundary = m == 0 or m == self.L - 1
        sd_c = math.sqrt(s2 / c) if (c > 0 and math.isfinite(c)) else math.inf
        # delete-one-block jackknife (the reported sd; the curvature sd assumes independent interval residuals, which
        # per-frame attitude errors violate: consecutive intervals share a pose, so sd_c is NOT used for decisions)
        B = self.n_blocks
        sd_jk = math.inf
        if B >= 2:
            M = S[None, :] - self._Sb[:B]
            tj, mj, _ = self._argmin_rows(M)
            sd_jk = float(math.sqrt((B - 1) / B * np.sum((tj - tj.mean()) ** 2)))
        # variance inflation of the iid model (>= 1: never less strict than the iid model)
        F = (sd_jk / sd_c) ** 2 if (math.isfinite(sd_jk) and math.isfinite(sd_c) and sd_c > 0) else 1.0
        # a second basin: any local minimum outside the main basin within s^2 max(1, F) chi2_1(P_GATE) of the minimum
        thr = Smin + s2 * max(1.0, F) * self.chi2_1
        lo = m
        while lo > 0 and S[lo - 1] >= S[lo]:
            lo -= 1
        hi = m
        while hi < self.L - 1 and S[hi + 1] >= S[hi]:
            hi += 1
        ambiguous = False
        for k in list(range(0, lo)) + list(range(hi + 1, self.L)):
            left = S[k - 1] if k > 0 else math.inf
            right = S[k + 1] if k < self.L - 1 else math.inf
            if S[k] <= left and S[k] <= right and S[k] <= thr:
                ambiguous = True
                break
        sd = sd_jk
        q = _t_ppf(1.0 - 0.5 * (1.0 - self.p), B - 1) if B >= 2 else math.inf
        # IDENTIFICATION: the profile's contrast against its own no-signal fluctuation.  Without rate changes S(tau) is
        # flat up to noise: each S(tau) ~ s^2 chi2_dof, so S(tau) - S(tau') has sd <= 2 s^2 sqrt(dof) (x sqrt(F) for
        # positively correlated residuals).  Identified when (median S - S_min) exceeds z_id such sds (Bonferroni over
        # the L lags at P_GATE).  (Post hoc, declared: added after madmax_F3's stationary first 32 s gave a
        # persistent spurious minimum near -200 ms that the jackknife / ambiguity tests passed at one instant.)
        contrast = (float(np.median(S)) - Smin) / (2.0 * s2 * math.sqrt(max(self.dof, 1.0) * max(1.0, F)))             if s2 > 0 else 0.0
        identified = contrast > self.z_id
        resolved = (not boundary) and (not ambiguous) and identified and B >= 2 and math.isfinite(sd) and sd > 0
        apply = td if (resolved and abs(td) > q * sd) else 0.0
        exc = math.sqrt(self._exc / self._exc_n) if self._exc_n > 0 else nan
        info = (c / s2) if (s2 > 0 and c > 0 and math.isfinite(c)) else 0.0
        e = TdEstimate(td, sd, sd_c, sd_jk, apply, resolved, boundary, ambiguous, B, self.n_int, self.dof, s, exc,
                       info, self.t_last, q, contrast, self.z_id)
        if with_profile:
            return e, {"lags": self.lags.copy(), "S": S.copy(), "s2": s2}
        return e


# ============================================================================================== (a) window state ===
def _interp_rate(t, x, n_rel, arr):
    k = int(np.searchsorted(t[:n_rel], x, side="right"))
    if k <= 0:
        return np.asarray(arr[0], float)
    if k >= n_rel:
        return np.asarray(arr[n_rel - 1], float)
    u = (x - t[k - 1]) / (t[k] - t[k - 1]) if t[k] > t[k - 1] else 1.0
    return (1.0 - u) * np.asarray(arr[k - 1], float) + u * np.asarray(arr[k], float)


def _hat(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _jr_inv(phi):
    th = float(np.linalg.norm(phi))
    K = _hat(phi)
    if th < 1e-6:
        return np.eye(3) + 0.5 * K + K @ K / 12.0
    return np.eye(3) + 0.5 * K + (1.0 / th ** 2 - (1.0 + math.cos(th)) / (2.0 * th * math.sin(th))) * (K @ K)


def preint_td_derivative(pre, t, gyro, acc, horizon):
    """d(dR, dv, dp)/d td of a Preint whose span [pre.t0, pre.t1] is already in the IMU clock (both ends move by
    +dtau): returns (h_R (3,) with dR(td + dtau) = dR Exp(dtau h_R), d_dv (3,), d_dp (3,)).  Rates / specific forces
    at the ends: linear interpolation of RELEASED samples (held beyond the horizon, C40), corrected by the record's
    linearisation biases."""
    t = np.asarray(t, float)
    n_rel = int(np.searchsorted(t, np.inf if horizon is None else float(horizon), side="right"))
    w0 = _interp_rate(t, pre.t0, n_rel, gyro) - pre.bg_lin
    w1 = _interp_rate(t, pre.t1, n_rel, gyro) - pre.bg_lin
    a0 = _interp_rate(t, pre.t0, n_rel, acc) - pre.ba_lin
    a1 = _interp_rate(t, pre.t1, n_rel, acc) - pre.ba_lin
    dR = pre.dR
    hR = w1 - dR.T @ w0
    ddv = dR @ a1 - a0 - np.cross(w0, pre.dv)
    ddp = pre.dv - a0 * pre.dt - np.cross(w0, pre.dp)
    return hR, ddv, ddp


def imu_td_jacobian(state_i, state_j, pre, g_w, t, gyro, acc, horizon, W=None):
    """d r / d td (15,) of the native IMU residual (imu_factor docstring order rR, rv, rp, rbg, rba), whitened by W
    when given.  First order in td; the bias Jacobians' own td dependence is neglected (second order)."""
    from xclean.vio import imu_factor as F
    hR, ddv, ddp = preint_td_derivative(pre, t, gyro, acc, horizon)
    si = F.as_state(state_i)
    r = F.imu_residual(state_i, state_j, pre, g_w)
    dbg = si.bg - pre.bg_lin
    # rR = Log(M^T Ri^T Rj), M = dR Exp(JRg dbg); dR -> dR Exp(e hR)  =>  M -> M Exp(e Exp(-JRg dbg) hR)
    # Log(Exp(-e u) Exp(rR)) ~ rR - e Jl^-1(rR) u,  Jl^-1(x) = Jr^-1(-x)
    Rc = F._so3_exp(pre.JRg @ dbg)
    u = Rc.T @ hR
    J = np.zeros(15)
    J[0:3] = -_jr_inv(-r[0:3]) @ u
    J[3:6] = -ddv
    J[6:9] = -ddp
    if W is not None:
        J = np.asarray(W, float) @ J
    return J


def td_schur_step(H, b, Hxt, Htt, bt, damp=0.0):
    """One Gauss-Newton step on t_d with the window's other variables x eliminated (Schur complement):
    [[H, Hxt], [Hxt^T, Htt]] [dx; dtd] = -[b; bt].  Returns (dtd, I_td) with I_td = Htt - Hxt^T H^-1 Hxt the marginal
    information of t_d given this window (the quantity a marginal prior on t_d accumulates)."""
    n = H.shape[0]
    Hd = H + (damp + 1e-9) * np.eye(n)
    try:
        L = np.linalg.cholesky(Hd)
        Y = np.linalg.solve(L, np.column_stack([Hxt, b]))
        y_t, y_b = Y[:, 0], Y[:, 1]
    except np.linalg.LinAlgError:
        Hi = np.linalg.pinv(Hd)
        y_t = None
        I_td = float(Htt - Hxt @ Hi @ Hxt)
        g_td = float(bt - Hxt @ Hi @ b)
        return (-g_td / I_td if I_td > 0 else 0.0), I_td
    I_td = float(Htt - y_t @ y_t)
    g_td = float(bt - y_t @ y_b)
    return (-g_td / I_td if I_td > 0 else 0.0), I_td


TdRefined = namedtuple("TdRefined", "td sd n_windows n_blocks significant")
TdRefined.__doc__ = ("TdRefine.estimate(): td [s] = the information-weighted mean of the windows' absolute estimates "
                     "(linearisation td + Schur increment), sd = its delete-one-block jackknife sd (the windows' own "
                     "information is NOT used for the sd: it was 10-25x optimistic), significant = the estimate "
                     "differs from the given reference at P_GATE (Student-t, B - 1 dof)")


class TdRefine:
    """Option (c)'s refinement stage: t_d on the FULL 15-row IMU residual, one Gauss-Newton step per keyframe window
    computed outside the window solve (td_schur_step on the window's own final linearisation + imu_td_jacobian per
    factor), accumulated as absolute estimates td_lin + dtd with their marginal information as weights.  Fed only while
    the profile (b) is resolved, so the linearisation point is inside the basin (from 0 the step is attenuated to
    <= 0.12 of a 65 / 130 ms offset).  Blocks of BLOCK windows are the jackknife units."""

    def __init__(self, block=BLOCK, p=P_GATE):
        self.block = int(block)
        self.p = float(p)
        self._cur = []
        self._blocks = []                    # (sum I x, sum I) per closed block

    def add(self, td_lin, dtd, info):
        if not (info > 0 and math.isfinite(info) and math.isfinite(dtd)):
            return
        self._cur.append((float(td_lin) + float(dtd), float(info)))
        if len(self._cur) >= self.block:
            x = np.array(self._cur)
            self._blocks.append((float(np.sum(x[:, 0] * x[:, 1])), float(np.sum(x[:, 1]))))
            self._cur = []

    def reset(self):
        self._cur = []
        self._blocks = []

    def estimate(self, ref=0.0):
        B = len(self._blocks)
        n = B * self.block + len(self._cur)
        if B < 2:
            return TdRefined(float("nan"), math.inf, n, B, False)
        sx = np.array([b[0] for b in self._blocks])
        si = np.array([b[1] for b in self._blocks])
        td = float(sx.sum() / si.sum())
        jk = (sx.sum() - sx) / (si.sum() - si)
        sd = float(math.sqrt((B - 1) / B * np.sum((jk - jk.mean()) ** 2)))
        q = _t_ppf(1.0 - 0.5 * (1.0 - self.p), B - 1)
        return TdRefined(td, sd, n, B, bool(sd > 0 and abs(td - float(ref)) > q * sd))


def warmup():
    """Compile the numba kernels (call once before timing)."""
    t = np.linspace(0.0, 1.0, 201)
    g = np.zeros((201, 3))
    out = np.empty((3, 3))
    lags = np.array([-0.01, 0.0, 0.01])
    _rot_lags(t, g, 201, 0.3, 0.35, lags, np.zeros(3), np.eye(3), out)
    _max_gap(t, 201, 0.2, 0.5)
