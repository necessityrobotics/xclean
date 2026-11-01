#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""THE INERTIAL LAYER - gyro preintegration, and a bias that is MEASURED.

    imu = ImuSource(dataset)
    imu.calibrate_bias()                       # from the stationary bootstrap
    dR = imu.delta_rotation_camera(t0, t1)     # camera-frame relative rotation

★ WHY ROTATION FIRST, AND WHY IT IS NOT A COMPROMISE
    Stereo already supplies SCALE, so the marginal value of inertial data here
    is not metric - it is ROTATION.  Rotation drift is what this project has
    repeatedly found at the bottom of its hardest rows: MH_04 and MH_05 were
    root-caused to "rotation drift in FAST stretches at 500+ inliers, not
    dark/far".  And the gyro is extraordinarily good at exactly that over a
    keyframe interval:

        sigma_g   = 1.70e-4 rad/s/sqrt(Hz)   (EuRoC ADIS16448, DECLARED)
        dt        ~ 0.15 s between keyframes
        sigma_dR  = sigma_g * sqrt(dt)  =  6.6e-5 rad

    That is two orders of magnitude tighter than what 200 keypoints of stereo
    reprojection can say about relative rotation.

★ THE BIAS IS MEASURED, NEVER ASSUMED - AND NEVER HAND-LOOSENED
    `slam-imu-noise-model`: the IMU factor was once weighted **40,000x too
    weakly**, so every "tight coupling" experiment in that era measured
    nothing.  The declared noise densities are Class-A INPUTS (calib.json) and
    they are used as given.  What is estimated is the BIAS, from the
    stationary bootstrap window, together with its own uncertainty - and that
    uncertainty is what inflates the factor's sigma, rather than a fudge
    factor.

    Stationarity is DETECTED, not assumed: a window qualifies when the gyro
    standard deviation is inside the declared noise density's own prediction
    and the accelerometer magnitude sits within a few sigma of gravity.  A
    sequence that never holds still simply gets no bias estimate and says so,
    and the prior widens to match.
"""
from __future__ import annotations

import os

import numpy as np

from xclean.core.lie import so3_exp, so3_log, so3_project

GRAVITY = 9.81


# ---------------------------------------------------------------------------
# T1-c (tier-1 programme, staged 2026-09-29): IMU-GAP REFUSAL, XTEN_IF_GAPSPAN=1
#
# * WHY REFUSE, NOT MERGE.  A keyframe interval that overlaps a HOLE in the
#   IMU stream (Rosario v1 01/02/03/06 carry holes of 0.06-0.48 s that straddle
#   a frame interval) is integrated by `preint.interval_samples` with an
#   np.interp ACROSS the hole and a white-noise Q as if it had been measured:
#   0.2-0.95 s of unmeasured motion enters the factor at measurement noise.
#   Online, `delta_rotation_body` already returns None for an interval WHOLLY
#   inside a hole, so the two intervals straddling each hole are the affected
#   ones.  The existing SAMPLES refusal (`preint._inertial_fix("SAMPLES")`)
#   refuses only n < 2.
# * THE RULE.  With the flag on, the GBA preintegration, the online gyro
#   rotation factor and the bias observation are REFUSED (None / skipped,
#   exactly like the existing refusal paths) whenever
#
#       overlaps_gap(t0, t1)   OR   n < 2        (n = integration sub-intervals)
#
#   so SAMPLES stays an explicit subset of GAPSPAN.
# * THE THRESHOLD IS A MEASUREMENT, NOT A CONSTANT.  A gap is an IMU sample
#   interval LONGER than the MEDIAN frame interval measured from the dataset's
#   own frame timestamps - never the declared camera_fps (Rig.fps is 0.0 when
#   calib.json has none, core/derive.py).  If that measurement is unavailable
#   while the flag is on, it is a hard error.
# * The gap table is built once per stream; `overlaps` is one searchsorted.
#   Flag unset: the predicate is never consulted and every estimate is
#   bit-identical; the summary gains only the read-only `imu.gaps` block.
# * Armed by XTEN_IF_GAPSPAN=1 ONLY - deliberately NOT by the
#   XTEN_INERTIAL_FIX=1 umbrella, whose four parts were measured as a unit
#   on 2026-09-25 (a fifth part would silently redefine that arm).
# ---------------------------------------------------------------------------
GAPSPAN_ENV = "XTEN_IF_GAPSPAN"
GAP_SITES = ("gba_preint", "rot_factor", "observe")


def gapspan_on():
    """The T1-c flag, read at call time (like `preint._inertial_fix`).

    Decision 9 (2026-09-30): unset / "" / "0" / "off" -> off; "1" -> on; ANY other
    value is a hard error, so a misspelt queue line cannot silently measure the
    default (the same contract as XTEN_KF_HEALTH).  Byte-identical when unset.
    """
    v = os.environ.get(GAPSPAN_ENV)
    if v is None or v in ("", "0", "off"):
        return False
    if v == "1":
        return True
    raise RuntimeError("%s=%r is not a valid value (use 1 to arm the T1-c gap-span "
                       "refusal, unset/0/off for the default)" % (GAPSPAN_ENV, v))


def measured_frame_interval(ds):
    """The MEDIAN positive frame-to-frame interval of the dataset's own frame
    timestamps, in seconds - the gap threshold.  None when fewer than two
    timestamped frames exist (the callers decide whether that is fatal)."""
    try:
        t = np.asarray([float(f.t) for f in ds.frames()], float)
    except Exception:
        return None
    if len(t) < 2:
        return None
    d = np.diff(t)
    d = d[d > 0]
    if len(d) == 0:
        return None
    return float(np.median(d))


def sub_interval_count(t, t0, t1):
    """The number of integration sub-intervals `preint.interval_samples`
    builds for [t0, t1]: the samples strictly inside, clipped to the
    interval, edges closer than 1e-12 s merged - the SAME construction, so
    `< 2` here is exactly preintegrate's `n < 2` (test_t1_gapspan asserts
    the equality over random intervals)."""
    t = np.asarray(t, float)
    i0 = int(np.searchsorted(t, t0, "right"))
    i1 = int(np.searchsorted(t, t1, "left"))
    edges = np.concatenate([[t0], t[i0:i1], [t1]])
    edges = edges[np.concatenate([[True], np.diff(edges) > 1e-12])]
    return int(len(edges) - 1)


class ImuGapTable:
    """Every IMU sample interval LONGER than `threshold_s`, as sorted
    (start, end) pairs, built once; `overlaps` is O(log n)."""

    def __init__(self, t, threshold_s):
        t = np.asarray(t, float)
        self.threshold_s = float(threshold_s)
        self.n_samples = int(len(t))
        d = np.diff(t) if len(t) > 1 else np.zeros(0)
        self.n_nonmonotonic = int((d < 0).sum())
        self.imu_median_dt_s = (float(np.median(d[d > 0]))
                                if bool((d > 0).any()) else None)
        m = d > self.threshold_s
        start, end = t[:-1][m], t[1:][m]
        order = np.argsort(start, kind="stable")
        self.start = start[order]
        self.end = end[order]
        self.sizes = self.end - self.start
        # the running maximum of the ends keeps the query exact even if the
        # stream were not monotonic (the gaps then need not be disjoint)
        self._end_max = (np.maximum.accumulate(self.end) if len(self.end)
                         else self.end)

    def __len__(self):
        return int(len(self.start))

    def overlaps(self, t0, t1):
        """True iff [t0, t1] and some gap (s, e) share a positive-length
        intersection: s < t1 and e > t0.  Touching at an endpoint is not an
        overlap: the interval that ends on the last sample before a hole,
        or starts on the first sample after it, is fully measured."""
        if len(self.start) == 0 or t1 <= t0:
            return False
        i = int(np.searchsorted(self.start, t1, "left"))    # gaps with s < t1
        return bool(i > 0 and float(self._end_max[i - 1]) > t0)

    def summary(self):
        s = np.asarray(self.sizes, float)
        return {"threshold_s": self.threshold_s,
                "imu_median_dt_s": self.imu_median_dt_s,
                "n_samples": self.n_samples,
                "n_nonmonotonic": self.n_nonmonotonic,
                "n_gaps": int(len(s)),
                "gap_total_s": float(s.sum()) if len(s) else 0.0,
                "gap_max_s": float(s.max()) if len(s) else 0.0,
                "gap_quantiles_s": ([float(np.percentile(s, q))
                                     for q in (50, 90, 100)] if len(s) else None),
                "gap_sizes_s": ([round(float(x), 4) for x in s]
                                if 0 < len(s) <= 64 else None)}


class ImuSource:
    def __init__(self, ds, verbose=True, bias_uncertainty="spread",
                 bias_clamp=False):
        # ★ "spread" is the DEFAULT again.  "stderr" (defect 19) was measured
        # at n=3 on R0 on 2026-08-27 and REFUTED: it left the per-keyframe
        # rotation error unmoved (V1_01 0.146 -> 0.149 deg) while making the
        # bias estimate 3-27x WORSE against EuRoC's published b_w_RS_S_*
        # (MH_02 0.00059 -> 0.01586 rad/s).  The declared spread is doing a
        # second job the arithmetic missed: it keeps the factor loose enough
        # that the bias estimator - which measures itself against a rotation
        # the factor is pulling - stays observable.
        self.bias_uncertainty = str(bias_uncertainty)
        self.ds = ds
        b = ds.imu()
        self.t = np.asarray(b.t, dtype=np.float64)
        self.g = np.asarray(b.gyro, dtype=np.float64)
        self.a = np.asarray(b.accel, dtype=np.float64)
        cal = ds.calib
        # R_cam0_imu maps a vector from the BODY (IMU) frame into cam0
        self.R_cb = np.array(cal.get("R_cam0_imu", np.eye(3)), dtype=np.float64)
        n = cal.get("imu_noise") or {}
        self.sigma_g = float(n.get("gyroscope_noise_density", 0.0) or 0.0)
        self.rw_g = float(n.get("gyroscope_random_walk", 0.0) or 0.0)
        # the ACCELEROMETER half, declared and used as given.  It was loaded
        # but never read while the inertial layer was gyro-only; `preint.py`
        # is what consumes it.
        self.sigma_a = float(n.get("accelerometer_noise_density", 0.0) or 0.0)
        self.rw_a = float(n.get("accelerometer_random_walk", 0.0) or 0.0)
        self.declared = bool(n)
        # body-from-cam0 and its inverse: the LEVER ARM is 68.9 mm on EuRoC and
        # getting this frame wrong is worth 2.3-2.6x (mh02 0.0174 -> 0.0404)
        T_bc = np.array(cal.get("T_body_cam0", np.eye(4)), dtype=np.float64)
        if T_bc.shape != (4, 4):
            T_bc = np.eye(4)
            T_bc[:3, :3] = self.R_cb.T
        self.T_bc = T_bc
        self.T_cb = np.linalg.inv(T_bc)
        self.rate = float(cal.get("imu_rate_hz") or 0.0) or self._rate()
        # ---- T1-c: the gap table, built once (read-only unless the flag is
        # on); the threshold is the MEASURED median frame interval ----------
        self.frame_interval_s = measured_frame_interval(ds)
        self.gaps = (ImuGapTable(self.t, self.frame_interval_s)
                     if self.frame_interval_s is not None else None)
        self.gap_refused = {s: {"gap": 0, "sparse": 0} for s in GAP_SITES}
        self.gba_preint_intervals = None      # set by Tracker._build_inertial
        self.gba_preint_none = None
        self.gba_kf_unobservable = None
        if gapspan_on() and self.gaps is None and self.available():
            raise RuntimeError(
                "%s=1: the gap threshold is the MEASURED median frame "
                "interval, and dataset %r gives no measurable one (fewer "
                "than two timestamped frames); camera_fps is never used"
                % (GAPSPAN_ENV, getattr(ds, "tag", "?")))
        self.bias = np.zeros(3)
        self.bias_sigma = None
        self.bias_sigma_spread = None
        self.bias_sigma_tight = None
        # ★ THE MOBILITY CLAMP.  The bias-sigma
        # saga (defects 19/24, arms v8/bspread/tightgba) decomposed the trade:
        # the TIGHT sigma is externally accurate (1.08x the true error) but
        # the unclamped estimator RUNS AWAY under it (S4: 3-27x worse vs
        # EuRoC's published bias) because vision drags the estimate through
        # the factor's own feedback loop; the SPREAD sigma is loop-stable but
        # collapses the factor on blur rows (v1_02 +27 %, v1_03 +14 %).  The
        # clamp keeps the tight sigma and removes the runaway: once the
        # online estimate has CONVERGED (first update whose correction lies
        # inside the physics bound), the bias may only move at the DECLARED
        # random walk per update - the only rate a real bias can move at.
        # Wrong inits still converge freely (the clamp arms only after
        # convergence); blur can no longer drag a converged bias.
        self.bias_clamp = bool(bias_clamp)
        self._clamp_armed = False        # v1 relic, kept for the summary key
        self._P = None                   # v2: per-axis filter covariance
        self._dt_accum = 0.0
        self.n_clamped = 0
        self.static_window_s = 0.0
        self._resid = []          # online bias observations, see `observe`
        self.bias_updates = 0
        self.bias_source = "none"
        self.verbose = verbose
        if not self.declared and verbose:
            print(" ! [imu] no imu_noise block: the IMU is not used on this "
                  "row (vision only); add imu_noise to calib.json to use it.")

    def _rate(self):
        if len(self.t) < 3:
            return 200.0
        return 1.0 / max(float(np.median(np.diff(self.t))), 1e-6)

    def available(self):
        return len(self.t) > 10 and self.declared

    # ---------------------------------------------------- bias, MEASURED --
    def causal_bootstrap(self, t_now, max_window_s=2.0):
        """xclean, STRICT CAUSALITY: the stationary-window bias bootstrap, using only IMU samples
        with t <= t_now.  calibrate_bias() below scanned windows starting anywhere in the first 8 s and kept the best,
        i.e. it read up to ~10 s of FUTURE IMU before frame 0.  Here, at every frame, the windows of the same scan that
        have fully elapsed are tested in scan order with the SAME stationarity test, and the FIRST one that qualifies
        sets the bias; the bootstrap then freezes (the online observe() estimator takes over as before).  Before that
        the bias is 0 and has_bias() is False, exactly as on a rig whose bootstrap finds no window.  No new constant.
        Called by the runner before each frame; a no-op once decided."""
        if getattr(self, "_boot_done", False) or not self.available():
            return self.bias
        f = self.rate
        expect_sd = self.sigma_g * np.sqrt(f)
        n = max(int(max_window_s * f), 20)
        step = max(n // 4, 1)
        s = getattr(self, "_boot_next", 0)
        # the scan covers window STARTS within the first 8 s of IMU (calibrate_bias's range), by TIME, so the same code
        # serves a complete array (offline) and a growing one (live)
        while s + n <= len(self.t) and float(self.t[s + n - 1]) <= t_now:
            if float(self.t[s]) - float(self.t[0]) >= 8.0:
                break
            gw = self.g[s:s + n]
            aw = self.a[s:s + n]
            sd = float(np.max(np.std(gw, axis=0)))
            amag = float(np.mean(np.linalg.norm(aw, axis=1)))
            a_sd = float(np.std(np.linalg.norm(aw, axis=1)))
            if (sd < 5.0 * expect_sd) and abs(amag - GRAVITY) < 0.5 and a_sd < 0.3:
                self.bias = gw.mean(axis=0)
                self.bias_sigma = float(expect_sd / np.sqrt(n))
                self.bias_sigma_spread = self.bias_sigma
                self.bias_sigma_tight = self.bias_sigma
                self.static_window_s = n / f
                self.bias_source = "stationary bootstrap"
                self.boot_window_end_s = float(self.t[s + n - 1])
                self._boot_done = True
                if self.verbose:
                    print(" [imu] CAUSAL gyro bias %s rad/s from the stationary window ending at t=%.2f s (decided "
                          "at frame time %.2f s)" % (np.array2string(self.bias, precision=6),
                                                     self.boot_window_end_s, t_now))
                return self.bias
            s += step
        self._boot_next = s
        if s < len(self.t) and float(self.t[s]) - float(self.t[0]) >= 8.0:
            self._boot_done = True              # the scan range is exhausted: no stationary window, no bootstrap
            self.bias_sigma = None
        return self.bias

    def calibrate_bias(self, max_window_s=2.0):
        """Estimate the gyro bias over the longest qualifying stationary run.

        The test is against the DECLARED noise density, not against a typed
        threshold: over a window of N samples at rate f, the standard
        deviation of a bias-free gyro is `sigma_g * sqrt(f)`, so a window
        whose measured sd exceeds a few times that is not stationary.
        """
        if not self.available():
            return None
        f = self.rate
        expect_sd = self.sigma_g * np.sqrt(f)
        n = int(min(max_window_s, self.t[-1] - self.t[0]) * f)
        n = max(n, 20)
        best = None
        step = max(n // 4, 1)
        for s in range(0, min(len(self.t) - n, int(8 * f)), step):
            gw = self.g[s:s + n]
            aw = self.a[s:s + n]
            sd = float(np.max(np.std(gw, axis=0)))
            amag = float(np.mean(np.linalg.norm(aw, axis=1)))
            a_sd = float(np.std(np.linalg.norm(aw, axis=1)))
            ok = (sd < 5.0 * expect_sd) and abs(amag - GRAVITY) < 0.5 \
                and a_sd < 0.3
            score = sd
            if ok and (best is None or score < best[0]):
                best = (score, s, n, gw.mean(axis=0), sd)
        if best is None:
            if self.verbose:
                print(" ! [imu] no stationary window found: the bias is not "
                      "estimated and the rotation prior widens to match")
            self.bias_sigma = None
            return None
        _sc, s, n, b, sd = best
        self.bias = b
        # the uncertainty of the MEAN of n samples, from the declared density
        self.bias_sigma = float(expect_sd / np.sqrt(n))
        # the bootstrap sigma is EXTERNAL (declared density over a stationary
        # window) - both rules coincide here
        self.bias_sigma_spread = self.bias_sigma
        self.bias_sigma_tight = self.bias_sigma
        self.static_window_s = n / f
        self.bias_source = "stationary bootstrap"
        if self.verbose:
            print(" [imu] gyro bias %s rad/s from a %.2f s stationary window "
                  "(sd %.2e vs declared %.2e); bias sigma %.2e"
                  % (np.array2string(b, precision=6), self.static_window_s,
                     sd, expect_sd, self.bias_sigma))
        return self.bias

    # ------------------------------------------------- preintegration ----
    def delta_rotation_body(self, t0, t1):
        """Integrate (omega - bias) over [t0, t1] in the BODY frame.

        ★ THIS IS THE RIGHT-ENDPOINT RULE, AND IT IS KEPT DELIBERATELY.
        Each sub-interval is advanced using the measurement at its RIGHT edge,
        which carries a systematic half-sample lag and is first order.
        `preint.interval_samples` is the second-order midpoint replacement,
        verified against synthetic truth at exactly 4.00x error reduction per
        halving of dt, and `preintegrate` uses it - the accelerometer channel
        needs it, where the same lag measured 1.3e-2 m/s of error over 1.5 s.

        Swapping it in HERE was tried on 2026-08-26 and REVERTED. The two
        differ by only 3.8e-4 rad over a 0.25 s keyframe interval (0.02 deg),
        but on V1_01 the change took the ATE from 0.0413 to 0.0771 and the
        global BA's entry residual from 5.0 px to 27.7 - measured by bisect,
        with the midpoint rule as the only difference between the two arms.
        That is n=1 on a row with a KNOWN bimodal cliff (0.0480 / 0.0521 /
        0.6196 on one recorded configuration), so it is a diagnostic, not a
        verdict - but shipping an accuracy change that the only measurement
        available says is harmful would be exactly backwards. The honest
        state: the lag is real, the replacement is more accurate in
        isolation, and it owes an n>=3 arm of its own before it lands.
        """
        if t1 <= t0 or not self.available():
            return None
        if gapspan_on() and self.gapspan_refuses(t0, t1, "rot_factor"):
            return None           # T1-c: overlaps an IMU gap, or n < 2
        i0 = int(np.searchsorted(self.t, t0))
        i1 = int(np.searchsorted(self.t, t1))
        if i1 <= i0:
            return None
        R = np.eye(3)
        tp = t0
        for k in range(i0, min(i1 + 1, len(self.t))):
            tk = float(self.t[k])
            dt = min(tk, t1) - tp
            if dt <= 0:
                tp = max(tp, tk)
                continue
            R = R @ so3_exp((self.g[k] - self.bias) * dt)
            tp = min(tk, t1)
            if tp >= t1:
                break
        return so3_project(R)

    def delta_rotation_camera(self, t0, t1):
        """The same rotation expressed between the two CAMERA frames."""
        Rb = self.delta_rotation_body(t0, t1)
        if Rb is None:
            return None
        return so3_project(self.R_cb @ Rb @ self.R_cb.T)

    # ------------------------------------------------- T1-c: gap refusal --
    def gapspan_refuses(self, t0, t1, site=None):
        """THE T1-c PREDICATE (pure; the callers consult the flag):

            refuse  <=>  gaps.overlaps(t0, t1)  OR  sub_interval_count(t, t0, t1) < 2

        `site` in GAP_SITES counts the refusal by kind ("gap" / "sparse").
        Hard error when no gap table exists (no measurable frame interval).
        """
        if self.gaps is None:
            raise RuntimeError(
                "%s=1: no gap table on this IMU source - the measured median "
                "frame interval was unavailable at construction (dataset %r); "
                "the threshold is that measurement, never camera_fps"
                % (GAPSPAN_ENV, getattr(self.ds, "tag", "?")))
        if self.gaps.overlaps(t0, t1):
            kind = "gap"
        elif sub_interval_count(self.t, t0, t1) < 2:
            kind = "sparse"
        else:
            return False
        if site is not None:
            self.gap_refused[site][kind] += 1
        return True

    def note_observe_withheld(self, t0, t1):
        """Provenance for the tracker: an interval whose rotation factor was
        refused never reaches `observe`; it is counted under the observe
        site.  No-op unless the flag is on."""
        if gapspan_on():
            self.gapspan_refuses(t0, t1, "observe")

    def note_gba_preint(self, n_intervals, n_none, n_kf_unobservable=None):
        """Provenance from Tracker._build_inertial: how many keyframe
        intervals it preintegrated, how many came back None, and how many
        keyframes have BOTH adjacent intervals None (no inertial row at all:
        their velocity is unobservable in the init and the solve)."""
        self.gba_preint_intervals = int(n_intervals)
        self.gba_preint_none = int(n_none)
        self.gba_kf_unobservable = (None if n_kf_unobservable is None
                                    else int(n_kf_unobservable))

    def gaps_summary(self):
        out = {"flag": gapspan_on(),
               "threshold_source": "measured median frame interval",
               "frame_interval_s": self.frame_interval_s}
        if self.gaps is not None:
            out.update(self.gaps.summary())
        else:
            out.update(threshold_s=None, n_gaps=None)
        out["refused"] = {s: {"gap": v["gap"], "sparse": v["sparse"],
                              "total": v["gap"] + v["sparse"]}
                          for s, v in self.gap_refused.items()}
        ni, nn = self.gba_preint_intervals, self.gba_preint_none
        out["gba_preint"] = {"intervals": ni, "none": nn,
                             "none_frac": (None if not ni else nn / ni),
                             "kf_unobservable": getattr(self, "gba_kf_unobservable", None)}
        return out

    def has_bias(self):
        """A gyro factor is only worth adding once the bias is known.

        MH_01's true gyro bias is **0.078 rad/s** on one axis - 0.67 degrees of
        unmodelled rotation per keyframe interval.  Adding a tight rotation
        prior before that is measured does not add information, it adds a
        confident error.
        """
        return self.bias_updates > 0 or self.bias_source == "stationary bootstrap"

    def prior_sigma(self, dt):
        """The sigma of that relative rotation, in radians, per axis.

        Two independent terms, both from DECLARED quantities:
          * white noise integrated:   sigma_g * sqrt(dt)
          * bias error integrated:    sigma_bias * dt     (+ random walk)
        No fudge factor.  If the bias was never estimated, its uncertainty
        falls back to the random walk over the elapsed time, which is the
        honest statement of "we do not know it".
        """
        wn = self.sigma_g * np.sqrt(max(dt, 1e-6))
        sb = self.bias_sigma if self.bias_sigma is not None \
            else max(self.rw_g * 30.0, 1e-3)
        return float(np.sqrt(wn * wn + (sb * dt) ** 2))

    def prior_sigma_tight(self, dt):
        """`prior_sigma` with the TIGHT (standard-error) bias uncertainty.

        For the run-end global BA ONLY (--gba-tight-gyro): offline there is
        no feedback loop for the tight factor to destabilise, and the tight
        sigma is the externally-validated one (1.08x the true bias error vs
        the spread's 3.84x, measured against EuRoC's published b_w_RS_S).
        """
        wn = self.sigma_g * np.sqrt(max(dt, 1e-6))
        sb = self.bias_sigma_tight if self.bias_sigma_tight is not None \
            else (self.bias_sigma if self.bias_sigma is not None
                  else max(self.rw_g * 30.0, 1e-3))
        return float(np.sqrt(wn * wn + (sb * dt) ** 2))

    # ------------------------------------------------- the online estimate --
    # ★ MEASURED, NOT ASSUMED - AND MOST SEQUENCES ARE NEVER STILL.
    # EuRoC MH_01 has a gyro standard deviation of **0.23 rad/s from its very
    # first sample**: the drone is already flying, so there is no stationary
    # bootstrap window to calibrate against, and the detector above correctly
    # says so instead of returning a number from moving data.
    #
    # The bias is therefore estimated against the only other thing that knows
    # the rotation: the ESTIMATOR.  Over a keyframe interval,
    #
    #     e = log( dR_gyro^T dR_visual )  ~  R_cb ( db ) dt
    #
    # so each keyframe donates one observation of the bias error.  The update
    # is a MEDIAN over a sufficiency floor of samples - not a mean, because a
    # single bad visual rotation would otherwise move the bias, and not a
    # `max()`-style statistic, which this project has banned for exactly that
    # reason.
    SUFFICIENCY = 20
    UPDATE_EVERY = 20

    def observe(self, dR_gyro, dR_visual, dt, t0=None, t1=None):
        """One keyframe interval's worth of evidence about the gyro bias.

        `t0`, `t1` (the interval, optional) let T1-c refuse it: with
        XTEN_IF_GAPSPAN=1 an interval that overlaps an IMU gap or has n < 2
        contributes nothing.  Without them, or with the flag unset, the
        behaviour is exactly the pre-T1-c one."""
        if t0 is not None and t1 is not None and gapspan_on() \
                and self.gapspan_refuses(t0, t1, "observe"):
            return None
        if dR_gyro is None or dR_visual is None or dt <= 1e-6:
            return None
        e = so3_log(so3_project(dR_gyro.T @ dR_visual))
        if not np.isfinite(e).all() or np.linalg.norm(e) > 0.5:
            return None                      # a visual rotation that far off
                                             # is evidence about tracking, not
                                             # about the bias
        # ★ THE SIGN.  Preintegration with a bias that is too SMALL
        # over-rotates: dR_gyro(b) ~ dR_true * exp((beta - b) dt), so
        # e = log(dR_gyro^T dR_true) ~ -(beta - b) dt and the correction is
        # MINUS e/dt.  The first version added +e/dt and the bias ran away to
        # 4.8 rad/s in four updates.  Settled against EuRoC's own published
        # ground-truth bias columns (b_w_RS_S_*): MH_01 carries
        # [-0.0032, +0.0213, +0.0784] rad/s, and the measured residual was
        # exactly its negative.
        self._resid.append(-(self.R_cb.T @ (e / dt)))
        self._dt_accum += float(dt)
        if len(self._resid) < self.SUFFICIENCY:
            return None
        if len(self._resid) % self.UPDATE_EVERY:
            return None
        arr = np.asarray(self._resid[-200:])
        db = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - db), axis=0) * 1.4826
        if self.bias_clamp:
            # ★ v2: A PER-AXIS KALMAN FILTER ON THE BIAS, replacing the v1
            # arming latch.  v1's latch was a CLIFF measured in one test series:
            # v1_03 armed on a lucky small correction 1 draw in 3 and froze
            # a wrong bias under a tight sigma (0.0380 vs 0.0252) - a binary
            # accept on a continuous quantity, the exact family the loop
            # stack banned after mag1.  The filter is the continuous form:
            #   measurement: db with sigma = its own standard error (per
            #                axis, 1.2533*MAD/sqrt(n) - measured);
            #   process:     P += rw_g^2 * dt  (DECLARED random walk);
            #   prior:       the first batch's population spread (measured).
            # Early, P is large -> K ~ 1 -> full corrections (fast
            # convergence, no latch to mis-arm).  Converged, P ~ physics ->
            # vision can only drag the bias at the random-walk rate.  The
            # factor's bias uncertainty IS sqrt(P) - the spread/stderr
            # dichotomy (defects 19/24, arms v8/bspread/tightgba/bclamp-v1)
            # dissolves into the filter's own covariance.
            se_ax = 1.2533 * mad / np.sqrt(max(len(arr), 1))
            if self._P is None:
                self._P = np.maximum(mad, 1e-4) ** 2
            else:
                self._P = self._P + (self.rw_g ** 2) * max(self._dt_accum,
                                                           1e-6)
            K = self._P / (self._P + se_ax ** 2 + 1e-18)
            if float(K.max()) < 0.5:
                self.n_clamped += 1
            db = K * db
            self._P = (1.0 - K) * self._P
        self._dt_accum = 0.0
        self.bias = self.bias + db
        # ★ DEFECT 19 - THE FACTOR WAS WEIGHTED BY THE ERROR IT EXISTS TO FIX.
        #
        # Each residual `-(R_cb^T e/dt)` carries TWO error sources: the gyro's
        # own noise, and the VISUAL rotation error of the interval it was
        # differenced against.  On V1_01 the visual per-keyframe rotation error
        # is 0.148 deg, so the population spread of those residuals is a
        # measurement of the VISION, not of the bias.  Declaring it as the bias
        # uncertainty means the gyro factor is weighted by exactly the quantity
        # it is supposed to correct - a circularity that caps the gyro at
        # "no better than the vision it is arguing with".
        #
        # What the factor needs is the uncertainty of the ESTIMATE, and the
        # estimate is a MEDIAN of n samples, whose standard error is
        # `1.2533 * sigma / sqrt(n)` for a normal population.  The bias then
        # random-walks away from it at the DECLARED `rw_g` until the next
        # update, so the two add in quadrature.  Both terms are declared or
        # measured; neither is chosen.
        #
        # ★ VALIDATED AGAINST AN EXTERNAL REFERENCE, NOT AGAINST THE ATE.
        # EuRoC publishes the true gyro bias (`b_w_RS_S_*`).  Measured on the
        # four R0 rows after defect 18:
        #     declared SPREAD          overstates the true error by  **3.84x**
        #     standard error of median predicts it at a ratio of     **1.08**
        # (V1_01, the binding row: true error 0.00021 rad/s against a declared
        # 0.00483 - **23x** overstated.)
        #
        # The previous comment recorded that the standard error measured WORSE
        # on MH_01 (0.0548 vs 0.0504).  That measurement was taken while
        # defect 18 was live, when every visual rotation carried a systematic
        # 1.184 deg error and the bias estimate itself was 8x worse on V1_01
        # (0.00172 against 0.00021 now).  A tight prior on a corrupted bias is
        # genuinely harmful; the reason it failed has been removed.
        # `--bias-spread` keeps the old rule as the ablation.
        n_s = max(len(arr), 1)
        sd = float(np.max(mad))
        # ★ BOTH candidate sigmas are tracked ALWAYS: the spread
        # (loop-stable online, overstates the true error 3.84x) and the
        # standard error of the median (externally validated at 1.08x but
        # UNSTABLE inside the online feedback loop - defect 19's test series).
        # The GBA runs once, offline, with the final bias: it has NO feedback
        # loop, so it may carry the tight sigma safely (--gba-tight-gyro).
        se = 1.2533 * sd / np.sqrt(n_s)
        rw = self.rw_g * np.sqrt(max(self.UPDATE_EVERY * 0.25, 1e-6))
        self.bias_sigma_spread = float(max(sd, 1e-4))
        self.bias_sigma_tight = float(max(np.hypot(se, rw), 1e-4))
        if self.bias_clamp and self._P is not None:
            # the filter's own covariance IS the estimate's uncertainty
            self.bias_sigma = float(max(np.sqrt(float(self._P.max())), 1e-4))
        elif self.bias_uncertainty == "spread":
            self.bias_sigma = self.bias_sigma_spread
        else:
            self.bias_sigma = self.bias_sigma_tight
        self.bias_updates += 1
        self.bias_source = "online (visual-gyro residual)"
        self._resid = []
        if self.verbose and self.bias_updates <= 3:
            print(" [imu] bias update %d: %s rad/s  (sigma %.2e, n=%d)"
                  % (self.bias_updates, np.array2string(self.bias, precision=6),
                     self.bias_sigma, len(arr)))
        return self.bias

    def summary(self):
        return {"available": self.available(), "bias": self.bias.tolist(),
                "bias_sigma": self.bias_sigma,
                "bias_clamp": self.bias_clamp,
                "bias_filter_P": (None if self._P is None
                                  else [float(x) for x in np.sqrt(self._P)]),
                "bias_clamped_updates": self.n_clamped,
                "bias_uncertainty_rule": self.bias_uncertainty,
                "source": self.bias_source,
                "updates": self.bias_updates,
                "static_window_s": self.static_window_s,
                "sigma_g": self.sigma_g, "declared": self.declared,
                # T1-c (read-only unless XTEN_IF_GAPSPAN=1): the gap table
                # and the per-site refusal counts
                "gaps": self.gaps_summary()}
