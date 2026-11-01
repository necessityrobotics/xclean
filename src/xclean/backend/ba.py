#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""BOUNDED LOCAL BUNDLE ADJUSTMENT - the phase the kill criterion hangs on.

    ba = LocalBA(rig)
    dec = ba.solve(state, center_kf, ledger)

★ BOUNDED WORK IS A DESIGN INVARIANT, NOT A TUNING PARAMETER (plan 5.1)

    > **No solve's cost may depend on map size.**

    * the window is the **covisibility neighbourhood of the current keyframe**
      and nothing else.  The span is DERIVED: a keyframe joins the window only
      if it shares at least `MIN_PNP_INLIERS` landmarks with the centre - fewer
      than that and it cannot constrain its own pose against the centre, so it
      does not belong in the same solve.  Capped at the plan's 30;
    * **one solve per keyframe** is the design point.  an earlier engine does 2.41, over
      a 326-keyframe mean span, which is 53x ORB-SLAM3's work per keyframe and
      the entire origin of the 450-824 ms/KF gap;
    * **fixed iteration count.**  No convergence loop that can run long;
    * a solve that would exceed its budget is **TRUNCATED AND LOGGED**, never
      silently extended.  That is the difference between a latency guarantee
      and a latency hope.

★ WHAT P0 FORCED INTO THE IMPLEMENTATION
    Every stage is ONE fused kernel (`backend/kernels.py`), because the chain
    is launch-bound: 128x the work costs 1.33x the time and a cupy expression
    chain issues ~226 launches per iteration.  The linear solve is a
    single-block Cholesky, because cuSOLVER measured 91 % of the composed
    floor.  float32 throughout, because the square-root/conditioning study says
    it is usable and the FP64 path buys nothing on a launch-bound problem.

★ A DEVIATION FROM THE PLAN, RECORDED RATHER THAN TAKEN QUIETLY
    Plan 5.4 specifies landmarks in **inverse depth**, host-relative, with far
    points as bearings.  This implementation parameterises the ESTIMATOR's
    landmarks as Euclidean world points while the RECORD stays host-relative
    inverse depth (`kit/writer.py`), which is what makes retirement lossless.
    The reason is sequencing, not disagreement: inverse depth earns its keep on
    **far points**, and the row where that is measurable is `kitti_09` with its
    path-ratio gate of 1.00 +- 0.02 (P5).  On EuRoC, depths are bounded by the
    derived 9.6 m band and the two parameterisations differ negligibly.
    **The deviation is listed with the test that will settle it**, so it cannot
    become a silent capability gap of the kind that killed render-and-align.
"""
from __future__ import annotations

import numpy as np

from xclean.core.derive import (HUBER_K, MIN_PNP_INLIERS,
                                  DISPARITY_NOISE_PX as DISPARITY_SIGMA_PX)
from xclean.core.lie import (se3_exp, se3_inv, so3_project, so3_log,
                               so3_jr_inv)
from xclean.obs.decision import Decision

MAX_SPAN = 30            # plan 5.1: "target span 10-30 keyframes"
ITERS = 5                # fixed; no convergence loop
D_POSE = 6               # visual-only for now; inertial adds vel+bias at P4+


class BAState:
    """The one mutable object that owns poses and landmarks (plan 5.6).

    Loop closure PROPOSES a correction; it does not reach in here.  That is
    what makes rollback cheap and "which subsystem moved this pose" an
    answerable question.
    """

    def __init__(self, store=None):
        self.kf_pose = {}                 # kf_id -> T_wc (4x4 float64)
        self.obs = {}                     # kf_id -> (lm_ids, uv)
        self.lm_kfs = {}                  # lm_id -> set(kf_id)
        # the landmark positions live in ONE growable array, shared with the
        # tracker.  A dict of 3-vectors turns every window gather into a Python
        # loop, and the window gather runs on every frame.
        self.store = store if store is not None else LandmarkStore()
        # the stereo measurement that fixes SCALE: each landmark's host
        # keyframe and the inverse depth the disparity measured there
        self.lm_host = None          # (N,) int64, set by the tracker
        self.lm_rho = None           # (N,) float64 - inverse RANGE, see below
        self.lm_invsig = None        # (N,) float64 - 1/sigma on inverse DEPTH
        self.lm_invsig_r = None      # (N,) float64 - 1/sigma on inverse RANGE
        # ---- the inverse-depth parameterisation (plan 5.4) ---------------
        # `lm_rho_state` is the ESTIMATOR'S VARIABLE and `lm_rho_meas` is the
        # frozen stereo measurement that priors it.  They start equal and
        # must never be the same array: a prior that follows the state it
        # constrains constrains nothing.
        self.lm_rho_state = None     # (N,) float64 - inverse DEPTH, the state
        self.lm_rho_meas = None      # (N,) float64 - inverse DEPTH, measured
        self.lm_m = None             # (N,3) float64 - m = (x_n, y_n, 1), STATE
        self.lm_m0 = None            # (N,2) float64 - the host KEYPOINT
        # a host that has left the window keeps its last pose: the stereo
        # measurement was made once and does not expire with the window
        self.host_pose_cache = {}
        # relative-rotation measurements between consecutive keyframes, from
        # the gyro: {(i, j): (R_meas_cam, sigma_rad)}
        self.rot_factors = {}

    @property
    def lm_xyz(self):
        return self.store.xyz

    def add_keyframe(self, kf_id, T_wc, lm_ids, uv):
        self.kf_pose[kf_id] = np.array(T_wc, dtype=np.float64)
        lm_ids = np.asarray(lm_ids, dtype=np.int64)
        self.obs[kf_id] = (lm_ids, np.asarray(uv, dtype=np.float64))
        for l in lm_ids:
            self.lm_kfs.setdefault(int(l), set()).add(kf_id)

    def drop_keyframe(self, kf_id):
        T = self.kf_pose.get(kf_id)
        if T is not None:
            self.host_pose_cache[kf_id] = T
        """A keyframe leaves the working set.  Its observations go with it -
        that is what keeps the covisibility graph, and therefore the solve,
        bounded (plan 5.1).  The landmarks themselves are RETIRED to the
        record, not deleted (plan 1b)."""
        lids, _uv = self.obs.pop(kf_id, (np.zeros(0, np.int64), None))
        self.kf_pose.pop(kf_id, None)
        orphaned = []
        for l in lids:
            ks = self.lm_kfs.get(int(l))
            if ks is None:
                continue
            ks.discard(kf_id)
            if not ks:
                self.lm_kfs.pop(int(l), None)
                orphaned.append(int(l))
        return orphaned


    def host_pose(self, kf_id):
        T = self.kf_pose.get(int(kf_id))
        return T if T is not None else self.host_pose_cache.get(int(kf_id))

    def hosted_by(self, kf_ids):
        """Landmark ids hosted by these keyframes, in O(log N) per keyframe.

        `lm_host` is NON-DECREASING - the tracker stamps the current keyframe
        id at creation and keyframe ids only increase - so each host owns a
        CONTIGUOUS slice and the lookup is a pair of `searchsorted`s.  A
        boolean scan over the whole landmark array would work and would also
        make a per-keyframe process cost O(map size), which is exactly what
        plan 5.1 forbids and what `landmark_creation` was already caught doing
        at 12.9 ms.
        """
        if self.lm_host is None or not len(self.lm_host) or not len(kf_ids):
            return np.zeros(0, np.int64)
        out = []
        for k in kf_ids:
            a, b = np.searchsorted(self.lm_host, [int(k), int(k) + 1])
            if b > a:
                out.append(np.arange(a, b, dtype=np.int64))
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def refresh_world(self, lm_ids):
        """Re-derive `store.xyz` for these landmarks - the cache, not a state.

        `P_w = T_wc_host (m / rho)`.  Landmarks whose host pose is
        unrecoverable are left alone rather than silently moved to the origin.
        """
        lm_ids = np.asarray(lm_ids, np.int64)
        if not len(lm_ids) or self.lm_m is None:
            return 0
        lm_ids = lm_ids[lm_ids < len(self.lm_host)]
        if not len(lm_ids):
            return 0
        hosts = self.lm_host[lm_ids]
        uniq = np.unique(hosts)
        T = {int(h): self.host_pose(h) for h in uniq}
        ok = np.array([T[int(h)] is not None for h in hosts])
        if not ok.any():
            return 0
        lm_ids, hosts = lm_ids[ok], hosts[ok]
        M = np.stack([T[int(h)] for h in hosts])
        Ph = self.lm_m[lm_ids] / np.maximum(self.lm_rho_state[lm_ids],
                                            1e-12)[:, None]
        self.store.xyz[lm_ids] = np.einsum("nij,nj->ni", M[:, :3, :3], Ph) \
            + M[:, :3, 3]
        return int(len(lm_ids))

    def covisible(self, kf_id, minimum=MIN_PNP_INLIERS, cap=MAX_SPAN):
        """The window, ranked by shared-landmark count.

        Derived, never typed: a keyframe joins only if it shares at least
        `MIN_PNP_INLIERS` landmarks with the centre - fewer than that and it
        cannot constrain its own pose against the centre, so it does not belong
        in the same solve.
        """
        if kf_id not in self.obs:
            return [kf_id]
        mine = set(self.obs[kf_id][0].tolist())
        counts = {}
        for l in mine:
            for k in self.lm_kfs.get(int(l), ()):
                if k != kf_id:
                    counts[k] = counts.get(k, 0) + 1
        nb = [k for k, c in counts.items() if c >= minimum]
        nb.sort(key=lambda k: -counts[k])
        return [kf_id] + nb[:cap - 1]

    def fixed_neighbours(self, window, lm_ids, cap=8):
        """★ THE KEYFRAMES THAT CONSTRAIN THE WINDOW WITHOUT MOVING.

        A local BA that optimises a window and ignores every OTHER keyframe
        observing the same landmarks is not a local BA - it is a free-floating
        sub-problem that can slide relative to the rest of the map, and it will
        happily reduce its own reprojection error while making the trajectory
        worse.  Measured here: without this, MH_01's first 600 frames went from
        0.0177 m (odometry alone) to 0.0242 m WITH bundle adjustment.

        These keyframes contribute residuals and are never updated.  ORB-SLAM3
        calls them fixed keyframes; the reason they exist is the same.
        """
        inside = set(window)
        cand = {}
        for l in lm_ids:
            for k in self.lm_kfs.get(int(l), ()):
                if k not in inside:
                    cand[k] = cand.get(k, 0) + 1
        out = sorted(cand, key=lambda k: -cand[k])[:cap]
        return out


class LandmarkStore:
    """Growable (N,3) arrays, so the window gather is a fancy index instead of
    a Python loop.

    `xyz` is the DERIVED world cache; `m` is the host-frame normalised-plane
    bearing `(x_n, y_n, 1)` and is a STATE, not a constant - the bundle
    adjustment writes it back.  It lives here rather than as three `_Cols`
    columns because the backend needs a WRITABLE (N,3) view: stacking three
    columns produces a copy, and a copy silently swallows every correction
    the solver makes to the bearing.
    """

    def __init__(self, cap=4096):
        self.xyz = np.zeros((cap, 3), np.float64)
        self.m = np.zeros((cap, 3), np.float64)
        self.n = 0

    def add(self, P, m=None):
        P = np.asarray(P, np.float64)
        if self.n + len(P) > len(self.xyz):
            new = max(len(self.xyz) * 2, self.n + len(P) + 1024)
            for name in ("xyz", "m"):
                g = np.zeros((new, 3), np.float64)
                g[:self.n] = getattr(self, name)[:self.n]
                setattr(self, name, g)
        base = self.n
        self.xyz[base:base + len(P)] = P
        if m is not None:
            self.m[base:base + len(P)] = m
        self.n += len(P)
        return base

class LocalBA:
    def __init__(self, rig, iters=ITERS, budget_ms=None, use_gpu=True,
                 verbose=False, inverse_depth=True):
        self.rig = rig
        # plan 5.4.  The Euclidean path stays as the ABLATION - it is the only
        # internal reference the engine keeps, and the far-band claim is only
        # meaningful against it.
        self.inverse_depth = bool(inverse_depth)
        # the FROZEN-BEARING ablation: scaling the bearing sigma down by 1e-3
        # pins (x_n, y_n) at the host keypoint and recovers the one-parameter
        # form exactly, so the two are one dial rather than two code paths
        self.bearing_scale = 1.0
        self.iters = int(iters)
        self.budget_ms = budget_ms
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.n_solves = 0
        self.truncated = 0
        self._recent_ms = []
        self._fixed_cap = 8
        # start at the LOW end of the plan's 10-30 span and grow into the
        # budget.  Starting at 30 and shrinking spends the first ~100
        # keyframes over budget while the controller walks down, and those
        # keyframes are in the p95 that decides the test series.
        self._span = 10
        # the PINNED-SPAN experiment arm; the control loop is the default
        self._pinned = False

    # ------------------------------------------------------------------ --
    def solve(self, st: BAState, center, ledger=None):
        import time
        span = self._span_for_budget()
        kfs = st.covisible(center, cap=span)
        if len(kfs) < 2:
            return Decision("local_ba", False, "window too small",
                            {"window": len(kfs)})

        lm_seen = set()
        for k in kfs:
            lm_seen.update(st.obs[k][0].tolist())
        fixed_kfs = st.fixed_neighbours(kfs, lm_seen, cap=self._fixed_cap)
        allk = list(kfs) + list(fixed_kfs)
        kf_index = {k: i for i, k in enumerate(allk)}
        n_free = len(kfs)

        rows_kf, rows_lm, rows_uv = [], [], []
        for k in allk:
            lids, uv = st.obs[k]
            if k in fixed_kfs:
                m = np.isin(lids, list(lm_seen))
                lids, uv = lids[m], uv[m]
                if not len(lids):
                    continue
            rows_kf.append(np.full(len(lids), kf_index[k], np.int32))
            rows_lm.append(lids)
            rows_uv.append(uv)
        okf = np.concatenate(rows_kf)
        olm = np.concatenate(rows_lm)
        ouv = np.concatenate(rows_uv)

        # a landmark enters the solve only if it is seen twice - one
        # observation carries no information about a 3-DoF point
        uniq, inv, cnt = np.unique(olm, return_inverse=True, return_counts=True)
        keep = cnt[inv] >= 2
        okf, olm, ouv = okf[keep], olm[keep], ouv[keep]
        uniq2, inv2 = np.unique(olm, return_inverse=True)
        if len(uniq2) < 8 or len(okf) < 20:
            return Decision("local_ba", False, "too little structure",
                            {"landmarks": len(uniq2), "observations": len(okf)})

        poses = np.stack([st.kf_pose[k] for k in allk])
        # the gauge plus the fixed neighbours: everything from n_free on is
        # held, and if there are none the oldest window keyframe is held
        free = np.ones(len(allk), bool)
        free[n_free:] = False
        if not fixed_kfs:
            free[kf_index[min(kfs)]] = False

        rotf = _gather_rot_factors(st, allk, kf_index, free)

        if self.inverse_depth:
            packed = _pack_id(st, uniq2, kf_index, allk, okf, inv2, ouv,
                              bearing_sigma=self.rig.bearing_sigma
                              * self.bearing_scale)
            if packed is None:
                return Decision("local_ba", False, "no hosted structure",
                                {"landmarks": int(len(uniq2))})
            (uniq2, rho, mbear, hidx, hpose_ext, okf_i, olm_i, ouv_i,
             prior) = packed
            if len(okf_i) < 20:
                return Decision("local_ba", False,
                                "too little structure once the structurally "
                                "zero host rows are dropped",
                                {"observations": int(len(okf_i))})
            t0 = time.perf_counter()
            fn = _solve_gpu_id if self.use_gpu else _solve_np_id
            poses, rho, mbear, stats = fn(
                poses, rho, mbear, hidx, hpose_ext, okf_i, olm_i, ouv_i,
                self.rig, free, self.iters, prior=prior, rotf=rotf,
                rho_floor=self.rig.rho_floor)
            ms = (time.perf_counter() - t0) * 1e3
            n_obs_used = len(okf_i)
        else:
            pts = st.lm_xyz[uniq2].copy()
            prior = _gather_prior(st, uniq2)
            t0 = time.perf_counter()
            fn = _solve_gpu if self.use_gpu else _solve_np
            poses, pts, stats = fn(poses, pts, okf, inv2, ouv, self.rig, free,
                                   self.iters, prior=prior, rotf=rotf)
            ms = (time.perf_counter() - t0) * 1e3
            n_obs_used = len(okf)
        self.n_solves += 1
        self._observe_cost(ms, n_obs_used, len(allk))

        # ---- the transaction: state is mutated here and nowhere else -----
        for i, k in enumerate(allk):
            if not free[i]:
                continue
            T = poses[i].copy()
            T[:3, :3] = so3_project(T[:3, :3])
            st.kf_pose[k] = T
        if self.inverse_depth:
            st.lm_rho_state[uniq2] = rho
            st.lm_m[uniq2] = mbear
            # the world cache is DERIVED, and it is re-derived from the poses
            # this transaction just wrote - not from the ones it read.  Every
            # landmark hosted by a keyframe that MOVED is re-derived too, not
            # only the ones this solve saw: a host-relative landmark that is
            # not refreshed when its host moves is a stale cache entry, and
            # stale caches are what the next frame's PnP would read.
            moved = [k for i, k in enumerate(allk) if free[i]]
            st.refresh_world(np.union1d(uniq2, st.hosted_by(moved)))
            okf = okf_i
        else:
            st.lm_xyz[uniq2] = pts

        dials = {"window": len(kfs), "fixed": len(fixed_kfs),
                 "gyro_factors": len(rotf), "fixed_cap": self._fixed_cap,
                 "landmarks": int(len(uniq2)),
                 "observations": int(len(okf)), "iters": self.iters,
                 "ms": ms, "span_cap": span,
                 "chi2_before": stats["chi2_0"], "chi2_after": stats["chi2_n"],
                 "rms_px_before": stats["rms_0"],
                 "rms_px_after": stats["rms_n"]}
        if self.budget_ms is not None and ms > self.budget_ms:
            self.truncated += 1
        return Decision("local_ba", True,
                        "%d free + %d fixed KFs, %d lm, %d obs, %.2f -> %.2f px"
                        % (len(kfs), len(fixed_kfs), len(uniq2), len(okf),
                           stats["rms_0"], stats["rms_n"]),
                        dials, margin=(self.budget_ms - ms)
                        if self.budget_ms else None, threshold=self.budget_ms)

    # ---- the span is set BY THE BUDGET, closed-loop on the MEASURED time
    # ------------------------------------------------------------------
    # Section 3.4: "a solve that would exceed its budget is TRUNCATED and
    # LOGGED, never silently extended.  This is the difference between a
    # latency guarantee and a latency hope."
    #
    # The first version predicted cost as ms-per-keyframe and truncated on the
    # prediction.  It measured 13.34 ms p95 against a 10 ms bar, because the
    # cost is not linear in keyframes: it also carries the observation count
    # and an O(n^3) factorisation.  A controller on the PREDICTION is only as
    # good as the model; a controller on the MEASUREMENT needs no model, and
    # the measurement is the thing the budget is actually about.
    SHRINK_AT = 0.85          # of budget, on the rolling p95
    GROW_AT = 0.55
    SPAN_MIN = 4
    Q = 0.95                  # the quantile the TEST SERIES gates on
    # ★ A QUANTILE THE SAMPLE CANNOT RESOLVE IS NOT A QUANTILE.
    # By the plotting-position rule the k-th of n sorted samples estimates the
    # k/(n+1) quantile, so a sample can resolve the q-quantile only when
    # q(n+1) <= n, i.e. n >= q/(1-q).  For q = 0.95 that is 19.  Below it the
    # old `xs[ceil(0.95 n) - 1]` returns the MAXIMUM, which estimates the
    # 100 n/(n+1) percentile - 88.9 % at n = 8 - and the controller read it as
    # a p95.
    #
    # MEASURED, by replaying this very estimator over the recorded per-solve
    # times of the twelve shipped v4 runs: the controller read 3.6-12.5 %
    # BELOW the run p95 the gate applies (median -8.5 %), at a median sample
    # count of 8, not 24, because the buffer is cleared on every adjustment.
    # So it parked the true p95 at ~1.0x the bar while believing it sat at
    # 0.85x - which is exactly the 10.08-10.20 ms breach on 4 of 12 runs.
    # This is an ESTIMATOR DEFECT, not a tuning knob, and the repair is to
    # refuse to decide from a sample that cannot carry the decision.
    N_MIN = int(np.ceil(Q / (1.0 - Q)))

    def _span_for_budget(self):
        """Returns the free-window span; `self._fixed_cap` moves with it.

        The FIXED neighbours are work too - they contribute observations, and
        therefore Jacobians, residuals and Schur pairs - so a controller that
        governs only the free span is governing half the cost.  When the span
        has already bottomed out at its floor and the budget is still missed,
        the fixed cap is what gives.
        """
        if self._pinned:
            return self._span
        if self.budget_ms is None:
            return MAX_SPAN
        if len(self._recent_ms) >= self.N_MIN:
            # plotting-position quantile, matching the N_MIN derivation above
            p95 = float(np.quantile(self._recent_ms, self.Q,
                                    method="weibull"))
            if p95 > self.SHRINK_AT * self.budget_ms:
                if self._span > self.SPAN_MIN:
                    self._span -= 1
                elif self._fixed_cap > 2:
                    self._fixed_cap -= 1
                self._recent_ms.clear()
            elif p95 < self.GROW_AT * self.budget_ms:
                if self._fixed_cap < 8:
                    self._fixed_cap += 1
                elif self._span < MAX_SPAN:
                    self._span += 1
                self._recent_ms.clear()
        return self._span

    def _observe_cost(self, ms, n_obs, n_kf):
        self._recent_ms.append(ms)
        if len(self._recent_ms) > 24:
            self._recent_ms.pop(0)
        # a solve that exceeded its budget is a TRUNCATION EVENT, and the
        # response is immediate rather than statistical: the next window is
        # smaller.  Waiting for a rolling p95 to notice would let the same
        # over-budget solve happen eight more times first.
        if self._pinned:
            return
        if self.budget_ms is not None and ms > self.budget_ms:
            changed = False
            if self._span > self.SPAN_MIN:
                self._span -= 1
                changed = True
            elif self._fixed_cap > 2:
                self._fixed_cap -= 1
                changed = True
            # ★ CLEAR ONLY IF THE CONFIGURATION ACTUALLY CHANGED.
            # Clearing is right when the window moved, because the old samples
            # came from a different problem.  Clearing when NOTHING moved -
            # the controller already at span floor and fixed-cap floor - is
            # pure CENSORING: it deletes exactly the upper-tail samples the
            # p95 is trying to see, and every deletion is triggered BY a
            # sample being large.  The estimator ends up conditioned on "no
            # sample exceeded the budget", which is the one condition under
            # which it cannot measure the thing it is for.
            # Measured by replaying both rules over the twelve v4 runs' own
            # recorded solve times, identical harness: median bias against the
            # run p95 goes -3.2 % -> -2.3 %, with the decision count unchanged
            # at 6 (range 2-11), so the responsiveness is not traded away.
            if changed:
                self._recent_ms.clear()


def _pack_id(st, lm_ids, kf_index, allk, okf, olm_local, ouv,
             bearing_sigma=1e-3):
    """Pack the window into the inverse-depth problem the solver takes.

    Three things happen here that the Euclidean path does not need:

      * **the host index is resolved against THIS window.**  A host inside the
        window is a variable and gets a slot; a host outside it is a constant
        and gets slot -1 plus its pose in `hpose_ext`.  Most long-tracked
        landmarks are in the second class, because the window has slid past
        the keyframe that created them;
      * **landmarks with no recoverable host pose are dropped.**  A landmark
        whose host is gone from both the window and the cache has no frame to
        be expressed in, and guessing one would be inventing structure;
      * **the host's OWN observation is dropped.**  Its residual is
        identically zero - `m/rho` reprojects to the keypoint that defined
        `m` - so keeping it would cost threads and quietly deflate the
        reported RMS with exact zeros.
    """
    if st.lm_rho_state is None or st.lm_m is None:
        return None
    n = len(lm_ids)
    rho = np.asarray(st.lm_rho_state[lm_ids], np.float64).copy()
    mbear = np.asarray(st.lm_m[lm_ids], np.float64).copy()
    host = np.asarray(st.lm_host[lm_ids], np.int64)
    hidx = np.full(n, -1, np.int64)
    hpose = np.zeros((n, 12))
    alive = np.ones(n, bool)
    for i in range(n):
        h = int(host[i])
        if h in kf_index:
            hidx[i] = kf_index[h]
            continue
        T = st.kf_pose.get(h)
        if T is None:
            T = st.host_pose_cache.get(h)
        if T is None:
            alive[i] = False
            continue
        hpose[i] = T[:3, :].reshape(12)
    alive &= rho > 0
    if not alive.all():
        keep_obs = alive[olm_local]
        okf, olm_local, ouv = okf[keep_obs], olm_local[keep_obs], ouv[keep_obs]
        remap = np.full(n, -1, np.int64)
        remap[alive] = np.arange(int(alive.sum()))
        olm_local = remap[olm_local]
        rho, mbear, hidx, hpose = (rho[alive], mbear[alive], hidx[alive],
                                   hpose[alive])
        lm_ids = lm_ids[alive]
        n = len(rho)
    if n == 0:
        return None
    # drop the structurally zero host rows
    keep = okf != hidx[olm_local]
    okf, olm_local, ouv = okf[keep], olm_local[keep], ouv[keep]
    # the prior IS the host keyframe's own observation, carried explicitly so
    # it survives the host leaving the window (see kernels.landmark_prior_id)
    wb = np.full(n, 1.0 / max(bearing_sigma, 1e-12))
    prior = (np.asarray(st.lm_m0[lm_ids], np.float64),
             np.asarray(st.lm_rho_meas[lm_ids], np.float64),
             wb,
             np.asarray(st.lm_invsig[lm_ids], np.float64),
             (np.asarray(st.lm_invsig[lm_ids]) > 0).astype(np.float64))
    return (lm_ids, rho, mbear, hidx, hpose, okf.astype(np.int32),
            olm_local.astype(np.int32), ouv, prior)


def _gather_prior(st, lm_ids):
    """(host camera centre, measured inverse RANGE, 1/sigma, valid).

    ★ THE UNITS HAVE TO MATCH, AND THE FIRST VERSION'S DID NOT.
    Disparity measures the camera-frame DEPTH `z`; the prior below is written
    on the inverse RANGE `1/||P - c_host||`, and range >= depth for every point
    off the optical axis.  Comparing `1/range` against a measured `1/depth`
    therefore pulls every landmark CLOSER than it is, and the map shrinks:
    measured on MH_01, path ratio **0.8906** and ATE 0.4393 - worse than
    having no prior at all.

    The repair is to store what the prior actually reads.  Range and depth
    differ by a factor `k = ||Pc|| / z` that is fixed by the BEARING, and the
    bearing is fixed by the keypoint, so `k` is a constant per landmark:

        rho_range = rho_depth / k ,   sigma_rho_range = sigma_rho_depth / k

    and `sigma_rho_depth = sigma_disparity / (f*B)` is the Class-B constant
    that comes straight from `rho = d / (f*B)` being LINEAR in disparity.
    """
    if st.lm_host is None or st.lm_rho is None:
        return None
    host, rho_all = st.lm_host, st.lm_rho
    # ★ the RANGE information, to go with the RANGE measurement above it.
    # These two columns are a PAIR and reading one with the other's partner
    # weakens the prior by `k = range/depth` - measured at a median 1.14 and
    # a max 1.43, and it moved a whole test series arm.
    sig_all = st.lm_invsig_r if st.lm_invsig_r is not None else st.lm_invsig
    n = len(lm_ids)
    c = np.zeros((n, 3))
    rho = np.zeros(n)
    invsig = np.zeros(n)
    valid = np.zeros(n)
    for i, l in enumerate(lm_ids):
        l = int(l)
        if l >= len(host):
            continue
        h = int(host[l])
        T = st.kf_pose.get(h)
        if T is None:
            T = st.host_pose_cache.get(h)
        if T is None or rho_all[l] <= 0:
            continue
        c[i] = T[:3, 3]
        rho[i] = rho_all[l]
        invsig[i] = sig_all[l] if sig_all is not None else 0.0
        valid[i] = 1.0 if invsig[i] > 0 else 0.0
    return c, rho, invsig, valid


def _gather_rot_factors(st, allk, kf_index, free):
    """Gyro relative-rotation factors whose BOTH keyframes are in the window.

    A factor with one endpoint outside the window still constrains the inside
    one, but the outside pose is held anyway, so it enters as a unary prior -
    which the pairwise form already produces once the held row is zeroed.
    """
    out = []
    for (i, j), (R, sig) in st.rot_factors.items():
        if i in kf_index and j in kf_index:
            a, b = kf_index[i], kf_index[j]
            if free[a] or free[b]:
                out.append((a, b, R, float(sig)))
    return out


def _rot_factor_blocks(T_cw, factors, d, n):
    """Hessian and gradient of the gyro rotation factors - VECTORISED.

    Residual, with `A = R_cw_i R_cw_j^T` and `M` the measured camera-frame
    relative rotation:

        e = log( M^T A )

    Under the LEFT perturbation this solver uses (`T <- exp(dx) T`),

        e(phi) ~ log(E) + Jr_inv(log E) ( A^T phi_i - phi_j )

    so the Jacobians are `Jr_inv(e) A^T` and `-Jr_inv(e)`, exactly - the right
    Jacobian inverse is in `core/lie` and pinned by identity tests, so there is
    no reason to approximate it.

    The per-factor Python loop cost 2.5 ms per solve because it runs on every
    LM iteration; `core/lie` is batched, so this is too.
    """
    H = np.zeros((n, n))
    g = np.zeros(n)
    if not factors:
        return H, g, 0.0
    a = np.fromiter((f[0] for f in factors), np.int64, len(factors))
    b = np.fromiter((f[1] for f in factors), np.int64, len(factors))
    M = np.stack([f[2] for f in factors])
    w = 1.0 / np.maximum(np.fromiter((f[3] for f in factors), np.float64,
                                     len(factors)), 1e-9)
    Ri = T_cw[a][:, :3, :3]
    Rj = T_cw[b][:, :3, :3]
    A = np.einsum("nij,nkj->nik", Ri, Rj)          # Ri @ Rj^T
    E = np.einsum("nji,njk->nik", M, A)            # M^T A
    e = so3_log(E)
    Jr = so3_jr_inv(e)
    Ja = w[:, None, None] * np.einsum("nij,nkj->nik", Jr, A)   # Jr @ A^T
    Jb = -w[:, None, None] * Jr
    r = w[:, None] * e
    chi = float(np.sum(r * r))
    Haa = np.einsum("nji,njk->nik", Ja, Ja)
    Hab = np.einsum("nji,njk->nik", Ja, Jb)
    Hbb = np.einsum("nji,njk->nik", Jb, Jb)
    ga = -np.einsum("nji,nj->ni", Ja, r)
    gb = -np.einsum("nji,nj->ni", Jb, r)
    ra = a * d + 3
    rb = b * d + 3
    for k in range(len(factors)):
        ia, ib = ra[k], rb[k]
        H[ia:ia + 3, ia:ia + 3] += Haa[k]
        H[ia:ia + 3, ib:ib + 3] += Hab[k]
        H[ib:ib + 3, ia:ia + 3] += Hab[k].T
        H[ib:ib + 3, ib:ib + 3] += Hbb[k]
        g[ia:ia + 3] += ga[k]
        g[ib:ib + 3] += gb[k]
    return H, g, chi


def _free_mask(free, nk):
    """Accept either a boolean mask or a single fixed index.

    The single-index form is the gauge-only case and it is what the synthetic
    tests use; the mask form carries the fixed neighbours as well.
    """
    if np.ndim(free) == 0:
        m = np.ones(nk, bool)
        m[int(free)] = False
        return m
    return np.asarray(free, bool)


def _solve_np(poses, pts, okf, olm, ouv, rig, free, iters, lam0=1e-4,
              prior=None, rotf=None):
    fx = rig.fx_rect
    cx, cy = rig.cx, rig.cy
    nk, nl = len(poses), len(pts)
    d, n = D_POSE, D_POSE * len(poses)
    free = _free_mask(free, nk)
    lam = lam0
    T_cw = np.stack([se3_inv(T) for T in poses])

    def residuals(T_cw, pts):
        R = T_cw[okf, :3, :3]
        t = T_cw[okf, :3, 3]
        Pc = np.einsum("nij,nj->ni", R, pts[olm]) + t
        z = np.maximum(Pc[:, 2], 1e-6)
        pred = np.stack([fx * Pc[:, 0] / z + cx, fx * Pc[:, 1] / z + cy], 1)
        return pred - ouv, Pc

    r0, _ = residuals(T_cw, pts)
    rms0 = float(np.sqrt(np.mean(np.sum(r0 ** 2, 1))))
    chi0 = float(np.sum(r0 ** 2))

    for _ in range(iters):
        r, Pc = residuals(T_cw, pts)
        e = np.linalg.norm(r, axis=1)
        w = np.where(e <= HUBER_K, 1.0, HUBER_K / np.maximum(e, 1e-9))
        sw = np.sqrt(w)[:, None]
        x, y, zz = Pc[:, 0], Pc[:, 1], np.maximum(Pc[:, 2], 1e-6)
        iz = 1.0 / zz
        iz2 = iz * iz
        # d(u,v)/d(Pc)
        Jp3 = np.zeros((len(r), 2, 3))
        Jp3[:, 0, 0] = fx * iz
        Jp3[:, 0, 2] = -fx * x * iz2
        Jp3[:, 1, 1] = fx * iz
        Jp3[:, 1, 2] = -fx * y * iz2
        # pose block: d(Pc)/d(xi) = [I | -hat(Pc)]
        dP = np.zeros((len(r), 3, 6))
        dP[:, 0, 0] = dP[:, 1, 1] = dP[:, 2, 2] = 1.0
        dP[:, 0, 4] = zz;  dP[:, 0, 5] = -y
        dP[:, 1, 3] = -zz; dP[:, 1, 5] = x
        dP[:, 2, 3] = y;   dP[:, 2, 4] = -x
        Jpose = np.einsum("nij,njk->nik", Jp3, dP) * sw[:, :, None]
        Jlm = np.einsum("nij,njk->nik", Jp3, T_cw[okf, :3, :3]) * sw[:, :, None]
        rw = r * sw

        Hpp = np.zeros((n, n))
        Hll = np.zeros((nl, 3, 3))
        Hpl = np.zeros((len(r), d, 3))
        bp = np.zeros(n)
        bl = np.zeros((nl, 3))
        np.add.at(Hll, olm, np.einsum("nri,nrj->nij", Jlm, Jlm))
        np.add.at(bl, olm, -np.einsum("nri,nr->ni", Jlm, rw))
        Hpl[:] = np.einsum("nri,nrj->nij", Jpose, Jlm)
        blk = np.einsum("nri,nrj->nij", Jpose, Jpose)
        gp = -np.einsum("nri,nr->ni", Jpose, rw)
        for a in range(d):
            np.add.at(bp, okf * d + a, gp[:, a])
            for b in range(d):
                np.add.at(Hpp, (okf * d + a, okf * d + b), blk[:, a, b])

        if prior is not None:
            c, rho0, invsig, valid = prior
            dP = pts - c
            dist = np.linalg.norm(dP, axis=1)
            ok = (valid > 0.5) & (dist > 1e-6)
            if ok.any():
                inv_sigma = invsig
                rho = np.zeros_like(dist)
                rho[ok] = 1.0 / dist[ok]
                rp = (rho - rho0) * inv_sigma
                wp = np.where(np.abs(rp) <= HUBER_K, 1.0,
                              HUBER_K / np.maximum(np.abs(rp), 1e-9))
                swp = np.sqrt(wp)
                gg = np.zeros_like(dist)
                gg[ok] = -rho[ok] ** 2 / dist[ok] * inv_sigma
                Jp_ = dP * (gg * swp)[:, None]
                Jp_[~ok] = 0.0
                rpw = rp * swp
                rpw[~ok] = 0.0
                Hll += np.einsum("ni,nj->nij", Jp_, Jp_)
                bl -= Jp_ * rpw[:, None]
        for j in range(nl):
            Hll[j] += lam * np.eye(3) * max(np.trace(Hll[j]) / 3.0, 1e-9)
        Hlli = np.linalg.inv(Hll)

        # Schur complement
        T3 = np.einsum("nij,njk->nik", Hpl, Hlli[olm])
        Hred = Hpp.copy()
        contrib = np.einsum("nij,mkj->nimk", T3, Hpl) if False else None
        # pairwise per landmark
        order = np.argsort(olm, kind="stable")
        o_s, kf_s = olm[order], okf[order]
        T3_s, Hpl_s = T3[order], Hpl[order]
        bounds = np.searchsorted(o_s, np.arange(nl + 1))
        for j in range(nl):
            a0, a1 = bounds[j], bounds[j + 1]
            if a1 - a0 < 2:
                if a1 - a0 == 1:
                    ki = kf_s[a0] * d
                    Hred[ki:ki + d, ki:ki + d] -= T3_s[a0] @ Hpl_s[a0].T
                    bp[ki:ki + d] -= T3_s[a0] @ bl[j]
                continue
            Ts = T3_s[a0:a1]
            Hs = Hpl_s[a0:a1]
            ks = kf_s[a0:a1] * d
            blocks = np.einsum("aij,bkj->abik", Ts, Hs)
            for ai in range(a1 - a0):
                bp[ks[ai]:ks[ai] + d] -= Ts[ai] @ bl[j]
                for bi in range(a1 - a0):
                    Hred[ks[ai]:ks[ai] + d, ks[bi]:ks[bi] + d] -= blocks[ai, bi]

        if rotf:
            Hr, gr, _c = _rot_factor_blocks(T_cw, rotf, d, n)
            Hpp += Hr
            bp += gr
        # gauge + fixed neighbours: held keyframes contribute residuals but
        # are not updated, so their rows and columns leave the solve
        for i in np.nonzero(~free)[0]:
            f0 = i * d
            Hred[f0:f0 + d, :] = 0.0
            Hred[:, f0:f0 + d] = 0.0
            Hred[f0:f0 + d, f0:f0 + d] = np.eye(d)
            bp[f0:f0 + d] = 0.0
        Hred[np.arange(n), np.arange(n)] += lam * np.maximum(
            np.diag(Hred), 1e-9)
        try:
            dxp = np.linalg.solve(Hred, bp)
        except np.linalg.LinAlgError:
            break

        # landmark back-substitution
        rhs = bl.copy()
        np.add.at(rhs, olm, -np.einsum("nij,ni->nj", Hpl,
                                       dxp.reshape(nk, d)[okf]))
        dxl = np.einsum("nij,nj->ni", Hlli, rhs)

        T_new = T_cw.copy()
        for i in range(nk):
            if not free[i]:
                continue
            T_new[i] = se3_exp(dxp[i * d:(i + 1) * d]) @ T_cw[i]
            T_new[i][:3, :3] = so3_project(T_new[i][:3, :3])
        pts_new = pts + dxl

        r2, _ = residuals(T_new, pts_new)
        if np.sum(r2 ** 2) <= np.sum(r ** 2):
            T_cw, pts = T_new, pts_new
            lam = max(lam * 0.5, 1e-7)
        else:
            lam = min(lam * 4.0, 1e2)

    rn, _ = residuals(T_cw, pts)
    return (np.stack([se3_inv(T) for T in T_cw]), pts,
            {"chi2_0": chi0, "chi2_n": float(np.sum(rn ** 2)),
             "rms_0": rms0,
             "rms_n": float(np.sqrt(np.mean(np.sum(rn ** 2, 1))))})


# --------------------------------------------------------------------------
# THE GPU PATH - every stage one fused kernel, float32, single-block Cholesky.
# Tested against `_solve_np` before its speed is ever quoted.
# --------------------------------------------------------------------------

class _Workspace:
    """Device buffers reused across solves.

    At these sizes an allocation is not free relative to the kernel that uses
    it - the same lesson P0 measured on the launch side.  The workspace grows
    to the largest shape seen and is then reused; a bounded backend has a
    bounded working set by construction, so it stops growing almost at once.
    """

    def __init__(self):
        self.cap = {}
        self.buf = {}

    def get(self, cp, name, shape, dtype):
        need = int(np.prod(shape))
        if self.cap.get(name, 0) < need or self.buf.get(name) is None \
                or self.buf[name].dtype != dtype:
            self.buf[name] = cp.empty(need, dtype=dtype)
            self.cap[name] = need
        return self.buf[name][:need].reshape(shape)


_WS = _Workspace()


def build_pairs(olm_sorted, nl):
    """Every ordered observation pair inside a landmark segment - VECTORISED.

    The first version looped over landmarks in Python and measured **2.145 ms**
    for a 575-landmark window: a fifth of the entire Tier-1 budget, spent in
    the host language, on bookkeeping.  It is the same lesson P0 taught on the
    GPU side - the arithmetic is free and the overhead is everything - and it
    applies just as hard above the driver as below it.
    """
    counts = np.bincount(olm_sorted, minlength=nl).astype(np.int64)
    offs = np.zeros(nl + 1, np.int32)
    np.cumsum(counts, out=offs[1:].view(np.int64)[:0] if False else None) \
        if False else None
    offs[1:] = np.cumsum(counts).astype(np.int32)
    sq = counts * counts
    total = int(sq.sum())
    if total == 0:
        return offs, np.zeros(0, np.int32), np.zeros(0, np.int32)
    rep = np.repeat(np.arange(nl, dtype=np.int64), sq)
    base = np.zeros(nl + 1, np.int64)
    base[1:] = np.cumsum(sq)
    local = np.arange(total, dtype=np.int64) - base[rep]
    c = counts[rep]
    pa = (offs[rep].astype(np.int64) + local // c).astype(np.int32)
    pb = (offs[rep].astype(np.int64) + local % c).astype(np.int32)
    return offs, pa, pb


def _solve_gpu(poses, pts, okf, olm, ouv, rig, free, iters, lam0=1e-4,
               led=None, kf_id=None, prior=None, rotf=None):
    import cupy as cp
    from xclean.backend import kernels as K

    nk, nl, N = len(poses), len(pts), len(okf)
    d, n = D_POSE, D_POSE * nk
    order = np.argsort(olm, kind="stable")
    okf_s = okf[order].astype(np.int32)
    olm_s = olm[order].astype(np.int32)
    ouv_s = ouv[order].astype(np.float32)
    offs, pa, pb = build_pairs(olm_s, nl)
    npair = len(pa)

    g_okf = cp.asarray(okf_s)
    g_olm = cp.asarray(olm_s)
    g_ouv = cp.asarray(ouv_s)
    g_offs = cp.asarray(offs)
    g_pa, g_pb = cp.asarray(pa), cp.asarray(pb)
    g_pts = cp.asarray(pts, dtype=cp.float32)

    W = _WS
    Jpose = W.get(cp, "Jpose", (N, 2, d), cp.float32)
    Jlm = W.get(cp, "Jlm", (N, 2, 3), cp.float32)
    resw = W.get(cp, "resw", (N, 2), cp.float32)
    Hpl = W.get(cp, "Hpl", (N, d, 3), cp.float32)
    Hll = W.get(cp, "Hll", (nl, 9), cp.float32)
    Hlli = W.get(cp, "Hlli", (nl, 9), cp.float32)
    bl = W.get(cp, "bl", (nl, 3), cp.float32)
    Hred = W.get(cp, "Hred", (n, n), cp.float32)
    grad = W.get(cp, "grad", (n,), cp.float32)
    dxp = W.get(cp, "dxp", (n,), cp.float32)
    dxl = W.get(cp, "dxl", (nl, 3), cp.float32)
    g_pts = W.get(cp, "pts", (nl, 3), cp.float32)
    g_pts.set(np.ascontiguousarray(pts, dtype=np.float32))
    g_dia = W.get(cp, "dia", (n,), cp.int32)
    g_dia[:] = cp.arange(n, dtype=cp.int32)
    # ★ PERSISTENT BUFFERS.  `cp.asarray` allocates, copies and synchronises;
    # at 32 calls per solve that measured 1.8 ms of the budget on transfers of
    # a few kilobytes.  `.set()` writes into a buffer that already exists.
    g_T = W.get(cp, "Tcw", (nk, 12), cp.float32)
    g_Hr = W.get(cp, "Hr", (n, n), cp.float32)
    g_gr = W.get(cp, "gr", (n,), cp.float32)
    h_T = np.empty((nk, 12), np.float32)
    h_step = np.empty(n, np.float32)
    h_dl = np.empty((nl, 3), np.float32)
    # ★ THE HOLD, AS A MASK.  Zeroing each held keyframe's rows and columns by
    # slicing costs four cupy calls per held keyframe per ITERATION - with 9
    # held and 5 iterations that is 180 launches per solve, on a problem P0
    # already showed is launch-bound.  A precomputed mask does it in three.

    free_mask = _free_mask(free, nk)
    held = np.nonzero(~free_mask)[0].tolist()

    g_prior = None
    if prior is not None:
        c, rho0, invsig, valid = prior
        g_prior = (cp.asarray(np.ascontiguousarray(c), dtype=cp.float32),
                   cp.asarray(rho0, dtype=cp.float32),
                   cp.asarray(invsig, dtype=cp.float32),
                   cp.asarray(valid, dtype=cp.float32))

    keep = np.ones(n, np.float32)
    for i in held:
        keep[i * d:(i + 1) * d] = 0.0
    g_keep = cp.asarray(keep)
    g_keep2 = g_keep[:, None] * g_keep[None, :]
    hd = np.nonzero(keep == 0)[0].astype(np.int32)
    g_hdiag = cp.asarray(hd)

    T_cw = np.stack([se3_inv(T) for T in poses])
    lam = lam0
    chi_prev = None
    best = (T_cw.copy(), pts.copy(), None)
    rms0 = None

    for it in range(iters):
        np.copyto(h_T, T_cw[:, :3, :].reshape(nk, 12))
        g_T.set(h_T)
        Hll.fill(0); bl.fill(0); Hred.fill(0); grad.fill(0)
        K.build_blocks(g_T, g_pts, g_okf, g_olm, g_ouv, Jpose, Jlm, resw,
                       Hll, bl, Hpl, N, rig.fx_rect, rig.cx, rig.cy, HUBER_K)
        if g_prior is not None:
            K.depth_prior(g_pts, g_prior[0], g_prior[1], g_prior[2],
                          g_prior[3], Hll, bl, nl, HUBER_K)
        chi = float(cp.sum(resw * resw))
        if rms0 is None:
            rms0 = float(np.sqrt(chi / N))
        if chi_prev is not None and chi > chi_prev:
            # reject: restore and damp harder.  A fixed iteration count still
            # needs a monotone guard, or one bad step undoes the solve.
            T_cw, pts = best[0].copy(), best[1].copy()
            g_pts.set(np.ascontiguousarray(pts, dtype=np.float32))
            lam = min(lam * 4.0, 1e2)
            continue
        chi_prev = chi
        best = (T_cw.copy(), pts.copy(), chi)

        K.inv3_damped(Hll, Hlli, nl, lam)
        K.pose_blocks(Jpose, resw, g_okf, Hred, grad, N, 2, d, n)
        if npair:
            K.schur_pairs(Hpl, Hlli, g_okf, g_olm, g_pa, g_pb, Hred, npair,
                          d, n)
        K.schur_grad(Hpl, Hlli, bl, g_okf, g_olm, grad, N, d)

        # the gyro factors touch only the pose blocks, so they are assembled
        # on the host - there are ~10 of them, against thousands of
        # observations, and a kernel for that would cost more in launches
        # than it saves in arithmetic (P0's lesson, applied in reverse)
        if rotf:
            Hr, gr, _c = _rot_factor_blocks(T_cw, rotf, d, n)
            g_Hr.set(np.ascontiguousarray(Hr, dtype=np.float32))
            g_gr.set(np.ascontiguousarray(gr, dtype=np.float32))
            Hred += g_Hr
            grad += g_gr

        Hred *= g_keep2
        Hred[g_hdiag, g_hdiag] = 1.0
        grad *= g_keep
        Hred[g_dia, g_dia] += lam * cp.maximum(Hred[g_dia, g_dia], 1e-9)

        K.chol_solve(Hred, grad, dxp, n)
        K.backsub(Hpl, Hlli, bl, dxp, g_okf, g_offs, dxl, nl, d)

        dxp.get(out=h_step)
        dxl.get(out=h_dl)
        step = h_step.astype(np.float64).reshape(nk, d)
        dl = h_dl.astype(np.float64)
        if not (np.isfinite(step).all() and np.isfinite(dl).all()):
            lam = min(lam * 10.0, 1e3)
            continue
        # ★ BATCHED.  The first version called se3_exp and so3_project once
        # per pose per iteration: 560 calls and 560 SVDs for ten solves, which
        # profiled as 3.4 ms of pure host time - a third of the whole budget,
        # spent re-entering numpy.  `core/lie` is batched; use it batched.
        # Re-orthonormalisation moves to the END: `exp(dx) @ T` preserves
        # SO(3) to machine precision, so projecting every iteration buys
        # nothing and costs an SVD per pose per step.
        upd = free_mask
        if upd.any():
            T_cw[upd] = se3_exp(step[upd]) @ T_cw[upd]
        pts = pts + dl
        g_pts.set(np.ascontiguousarray(pts, dtype=np.float32))
        lam = max(lam * 0.5, 1e-7)

    # one re-orthonormalisation for the whole solve (see the note above)
    T_cw[:, :3, :3] = so3_project(T_cw[:, :3, :3])

    # final residual for the report
    np.copyto(h_T, T_cw[:, :3, :].reshape(nk, 12))
    g_T.set(h_T)
    g_pts.set(np.ascontiguousarray(pts, dtype=np.float32))
    Hll.fill(0); bl.fill(0)
    K.build_blocks(g_T, g_pts, g_okf, g_olm, g_ouv, Jpose, Jlm, resw,
                   Hll, bl, Hpl, N, rig.fx_rect, rig.cx, rig.cy, HUBER_K)
    chi_n = float(cp.sum(resw * resw))
    return (np.stack([se3_inv(T) for T in T_cw]), pts,
            {"chi2_0": best[2] if best[2] else chi_n, "chi2_n": chi_n,
             "rms_0": rms0, "rms_n": float(np.sqrt(chi_n / N)),
             "npair": int(npair), "n_obs": int(N)})


# ==========================================================================
# INVERSE-DEPTH LANDMARKS (plan 5.4)
#
# One scalar per landmark instead of three, anchored to a HOST keyframe by a
# fixed normalised-plane bearing `m = (x_n, y_n, 1)`:
#
#     P_host = m / rho ,  P_world = T_wc_host P_host ,  P_cam = T_cw_obs P_world
#
# ★ THE DEVIATION RECORDED AT THE TOP OF THIS FILE IS NOW CLOSED.
#   It said: "inverse depth earns its keep on far points, and the row where
#   that is measurable is kitti_09".  Two things changed that reading.
#   First, the stereo prior stops needing a units conversion at all - it is
#   `(rho - rho_meas) * invsig` with a CONSTANT invsig, so the class of defect
#   that produced path ratio 0.8906 is unwritable here.  Second, the landmark
#   block becomes a POSITIVE SCALAR with a constant prior term in it, which is
#   what the far band actually needed: a 3x3 block on a 48 m point against a
#   0.1 m window baseline has a conditioning ratio near 1e-6, and float32's
#   eps is 1.2e-7, so `inv3_damped`'s cofactor inverse was returning noise.
#
# The Schur complement also gets simpler, not harder: with a scalar `Hll` the
# reduction is exactly a rank-`nl` update `Hred = Hpp - U diag(1/Hll) U^T`.
# ==========================================================================


def _host_frames(poses, hidx, hpose_ext):
    """(R_wc_host, t_wc_host) per landmark, from the window or from the cache.

    A host inside the window is a VARIABLE and its pose is read from the live
    array every iteration; a host outside it is a constant carried in
    `hpose_ext`.  Two sources, one lookup, so nothing downstream has to know
    which case it is.
    """
    nl = len(hidx)
    R = np.empty((nl, 3, 3))
    t = np.empty((nl, 3))
    ext = hpose_ext.reshape(nl, 3, 4)
    inw = hidx >= 0
    if inw.any():
        Th = poses[hidx[inw]]
        R[inw] = Th[:, :3, :3]
        t[inw] = Th[:, :3, 3]
    if (~inw).any():
        R[~inw] = ext[~inw, :, :3]
        t[~inw] = ext[~inw, :, 3]
    return R, t


def id_world(poses, rho, mbear, hidx, hpose_ext):
    """World positions from the inverse-depth state - the DERIVED cache.

    Everything outside the estimator (PnP, local-map projection, the loop
    weld, the kit) wants a world point.  Under this parameterisation a world
    point is not a variable, it is a function of (host pose, bearing, rho), so
    it is recomputed rather than stored-and-hoped.  That is the same property
    that makes landmark retirement lossless under a later loop correction.
    """
    R, t = _host_frames(poses, hidx, hpose_ext)
    Ph = mbear / np.maximum(rho, 1e-12)[:, None]
    return np.einsum("nij,nj->ni", R, Ph) + t


def _analytic_blocks_id(poses, T_cw, rho, mbear, hidx, hpose_ext, okf, olm,
                        ouv, rig, sw=None):
    """The three analytic Jacobian blocks and the residual - UNWEIGHTED unless
    `sw` is given.

    Factored out of the reference solver so the finite-difference test in
    `tests/test_ba_id.py` checks THE BLOCKS THE SOLVER USES rather than a
    second copy of them.  A test against a re-derivation would pass on two
    identically wrong derivations.
    """
    fx, cx, cy = rig.fx_rect, rig.cx, rig.cy
    N = len(okf)
    hidx = np.asarray(hidx, np.int64)
    Rh, th = _host_frames(poses, hidx, hpose_ext)
    Ph = mbear / np.maximum(rho, 1e-12)[:, None]
    Pw = np.einsum("nij,nj->ni", Rh, Ph) + th
    Pc = np.einsum("nij,nj->ni", T_cw[okf, :3, :3], Pw[olm]) + T_cw[okf, :3, 3]
    zz = np.maximum(Pc[:, 2], 1e-6)
    r = np.stack([fx * Pc[:, 0] / zz + cx, fx * Pc[:, 1] / zz + cy], 1) - ouv
    x, y = Pc[:, 0], Pc[:, 1]
    iz = 1.0 / zz
    A = np.zeros((N, 2, 3))
    A[:, 0, 0] = fx * iz
    A[:, 0, 2] = -fx * x * iz * iz
    A[:, 1, 1] = fx * iz
    A[:, 1, 2] = -fx * y * iz * iz
    # observer: d(Pc)/d(xi) = [I | -hat(Pc)] under T_cw <- exp(xi) T_cw
    dPo = np.zeros((N, 3, 6))
    dPo[:, 0, 0] = dPo[:, 1, 1] = dPo[:, 2, 2] = 1.0
    dPo[:, 0, 4] = zz;  dPo[:, 0, 5] = -y
    dPo[:, 1, 3] = -zz; dPo[:, 1, 5] = x
    dPo[:, 2, 3] = y;   dPo[:, 2, 4] = -x
    Jobs = np.einsum("nij,njk->nik", A, dPo)
    # host: T_cw_h <- exp(xi) T_cw_h  =>  T_wc_h <- T_wc_h exp(-xi), so
    # d(Ph)/d(xi) = [-I | +hat(Ph)] expressed in the HOST camera frame
    Rch = np.einsum("nij,njk->nik", T_cw[okf, :3, :3], Rh[olm])
    ARch = np.einsum("nij,njk->nik", A, Rch)
    Pho = Ph[olm]
    dPh = np.zeros((N, 3, 6))
    dPh[:, 0, 0] = dPh[:, 1, 1] = dPh[:, 2, 2] = -1.0
    dPh[:, 0, 4] = -Pho[:, 2]; dPh[:, 0, 5] = Pho[:, 1]
    dPh[:, 1, 3] = Pho[:, 2];  dPh[:, 1, 5] = -Pho[:, 0]
    dPh[:, 2, 3] = -Pho[:, 1]; dPh[:, 2, 4] = Pho[:, 0]
    Jhost = np.einsum("nij,njk->nik", ARch, dPh)
    Jhost[hidx[olm] < 0] = 0.0
    # the LANDMARK block, three parameters (x_n, y_n, rho):
    #   d(Ph)/d(x_n) = e1/rho ,  d(Ph)/d(y_n) = e2/rho ,  d(Ph)/d(rho) = -Ph/rho
    irho = 1.0 / np.maximum(rho[olm], 1e-12)
    Jlm = np.empty((N, 2, 3))
    Jlm[:, :, 0] = ARch[:, :, 0] * irho[:, None]
    Jlm[:, :, 1] = ARch[:, :, 1] * irho[:, None]
    Jlm[:, :, 2] = -np.einsum("nqk,nk->nq", ARch, Pho) * irho[:, None]
    if sw is not None:
        Jobs = Jobs * sw[:, :, None]
        Jhost = Jhost * sw[:, :, None]
        Jlm = Jlm * sw[:, :, None]
    return Jobs, Jhost, Jlm, r


def _solve_np_id(poses, rho, mbear, hidx, hpose_ext, okf, olm, ouv, rig, free,
                 iters, lam0=1e-4, prior=None, rotf=None, rho_floor=1e-4):
    """The reference implementation.  Dense, obvious, slow, and the thing the
    fused GPU path is checked against before its speed is ever quoted."""
    fx, cx, cy = rig.fx_rect, rig.cx, rig.cy
    nk, nl, N = len(poses), len(rho), len(okf)
    d, n = D_POSE, D_POSE * nk
    free = _free_mask(free, nk)
    lam = lam0
    poses = np.array(poses, np.float64)
    rho = np.array(rho, np.float64)
    T_cw = np.stack([se3_inv(T) for T in poses])
    hidx = np.asarray(hidx, np.int64)
    kslot = np.stack([okf.astype(np.int64), hidx[olm]], 1)     # (N,2)

    def geom(poses, T_cw, rho, mbear):
        Rh, th = _host_frames(poses, hidx, hpose_ext)
        Ph = mbear / np.maximum(rho, 1e-12)[:, None]
        Pw = np.einsum("nij,nj->ni", Rh, Ph) + th
        Pc = np.einsum("nij,nj->ni", T_cw[okf, :3, :3], Pw[olm]) \
            + T_cw[okf, :3, 3]
        z = np.maximum(Pc[:, 2], 1e-6)
        r = np.stack([fx * Pc[:, 0] / z + cx, fx * Pc[:, 1] / z + cy], 1) - ouv
        return r, Pc, Ph, Rh

    mbear = np.array(mbear, np.float64)
    r0, _, _, _ = geom(poses, T_cw, rho, mbear)
    rms0 = float(np.sqrt(np.mean(np.sum(r0 ** 2, 1))))
    chi0 = float(np.sum(r0 ** 2))

    for _ in range(iters):
        r, Pc, Ph, Rh = geom(poses, T_cw, rho, mbear)
        e = np.linalg.norm(r, axis=1)
        w = np.where(e <= HUBER_K, 1.0, HUBER_K / np.maximum(e, 1e-9))
        sw = np.sqrt(w)[:, None]
        Jobs, Jhost, Jlm, _r = _analytic_blocks_id(
            poses, T_cw, rho, mbear, hidx, hpose_ext, okf, olm, ouv, rig,
            sw=sw)
        rw = r * sw

        Jp = np.stack([Jobs, Jhost], 1)              # (N,2,2,6)
        Hll = np.zeros((nl, 3, 3))
        bl = np.zeros((nl, 3))
        np.add.at(Hll, olm, np.einsum("nqa,nqb->nab", Jlm, Jlm))
        np.add.at(bl, olm, -np.einsum("nqa,nq->na", Jlm, rw))
        Hpl = np.einsum("nsqa,nqc->nsac", Jp, Jlm)   # (N,2,6,3)

        if prior is not None:
            m0, rho0, wbear, invsig, valid = prior
            ok = valid > 0.5
            rp = np.stack([(mbear[:, 0] - m0[:, 0]) * wbear,
                           (mbear[:, 1] - m0[:, 1]) * wbear,
                           (rho - rho0) * invsig], 1)
            J0 = np.stack([wbear, wbear, invsig], 1)
            wp = np.where(np.abs(rp) <= HUBER_K, 1.0,
                          HUBER_K / np.maximum(np.abs(rp), 1e-9))
            swp = np.sqrt(wp)
            swp[:, 0] = 1.0        # the bearing prior is NOT robustified -
            swp[:, 1] = 1.0        # see kernels.landmark_prior_id
            Jpr = J0 * swp
            for c in range(3):
                Hll[ok, c, c] += (Jpr[:, c] ** 2)[ok]
            bl[ok] -= (Jpr * rp * swp)[ok]

        Hpp = np.zeros((n, n))
        grad = np.zeros(n)
        U = np.zeros((n, nl, 3))
        for sa in range(2):
            ka = kslot[:, sa]
            ok_a = ka >= 0
            for a in range(d):
                np.add.at(grad, (ka[ok_a] * d + a),
                          -np.sum(Jp[ok_a, sa, :, a] * rw[ok_a], 1))
                for c in range(3):
                    np.add.at(U, (ka[ok_a] * d + a, olm[ok_a], c),
                              Hpl[ok_a, sa, a, c])
                for sb in range(2):
                    kb = kslot[:, sb]
                    m2 = ok_a & (kb >= 0)
                    for b in range(d):
                        np.add.at(Hpp, (ka[m2] * d + a, kb[m2] * d + b),
                                  np.sum(Jp[m2, sa, :, a] * Jp[m2, sb, :, b],
                                         1))
        if rotf:
            Hr, gr, _c = _rot_factor_blocks(T_cw, rotf, d, n)
            Hpp += Hr
            grad += gr

        Hd = Hll + lam * np.maximum(
            np.trace(Hll, axis1=1, axis2=2)[:, None, None] / 3.0,
            1e-9) * np.eye(3)
        Hlli = np.linalg.inv(Hd)
        T3 = np.einsum("alc,lcd->ald", U, Hlli)
        Hred = Hpp - np.einsum("ald,bld->ab", T3, U)
        bp = grad - np.einsum("ald,ld->a", T3, bl)
        for i in np.nonzero(~free)[0]:
            f0 = i * d
            Hred[f0:f0 + d, :] = 0.0
            Hred[:, f0:f0 + d] = 0.0
            Hred[f0:f0 + d, f0:f0 + d] = np.eye(d)
            bp[f0:f0 + d] = 0.0
        Hred[np.arange(n), np.arange(n)] += lam * np.maximum(np.diag(Hred),
                                                             1e-9)
        try:
            dxp = np.linalg.solve(Hred, bp)
        except np.linalg.LinAlgError:
            break
        dxl = np.einsum("lcd,ld->lc", Hlli,
                        bl - np.einsum("alc,a->lc", U, dxp))

        T_new = T_cw.copy()
        upd = free
        if upd.any():
            T_new[upd] = se3_exp(dxp.reshape(nk, d)[upd]) @ T_cw[upd]
            T_new[upd, :3, :3] = so3_project(T_new[upd, :3, :3])
        m_new = mbear.copy()
        m_new[:, 0] += dxl[:, 0]
        m_new[:, 1] += dxl[:, 1]
        rho_new = np.maximum(rho + dxl[:, 2], rho_floor)
        p_new = np.stack([se3_inv(T) for T in T_new])
        r2, _, _, _ = geom(p_new, T_new, rho_new, m_new)
        if np.sum(r2 ** 2) <= np.sum(r ** 2):
            T_cw, poses, rho, mbear = T_new, p_new, rho_new, m_new
            lam = max(lam * 0.5, 1e-7)
        else:
            lam = min(lam * 4.0, 1e2)

    rn, _, _, _ = geom(poses, T_cw, rho, mbear)
    return (poses, rho, mbear,
            {"chi2_0": chi0, "chi2_n": float(np.sum(rn ** 2)), "rms_0": rms0,
             "rms_n": float(np.sqrt(np.mean(np.sum(rn ** 2, 1))))})


def _solve_gpu_id(poses, rho, mbear, hidx, hpose_ext, okf, olm, ouv, rig, free,
                  iters, lam0=1e-4, prior=None, rotf=None, rho_floor=1e-4):
    import cupy as cp
    from xclean.backend import kernels as K

    nk, nl, N = len(poses), len(rho), len(okf)
    d, n = D_POSE, D_POSE * nk
    order = np.argsort(olm, kind="stable")
    okf_s = okf[order].astype(np.int32)
    olm_s = olm[order].astype(np.int32)
    ouv_s = ouv[order].astype(np.float32)
    offs, pa, pb = build_pairs(olm_s, nl)
    npair = len(pa)

    W = _WS
    g_okf = W.get(cp, "id_okf", (N,), cp.int32);  g_okf.set(okf_s)
    g_olm = W.get(cp, "id_olm", (N,), cp.int32);  g_olm.set(olm_s)
    g_ouv = W.get(cp, "id_ouv", (N, 2), cp.float32)
    g_ouv.set(np.ascontiguousarray(ouv_s))
    g_offs = W.get(cp, "id_offs", (nl + 1,), cp.int32); g_offs.set(offs)
    g_pa = W.get(cp, "id_pa", (max(npair, 1),), cp.int32)
    g_pb = W.get(cp, "id_pb", (max(npair, 1),), cp.int32)
    if npair:
        g_pa.set(pa); g_pb.set(pb)

    g_hidx = W.get(cp, "id_hidx", (nl,), cp.int32)
    g_hidx.set(np.ascontiguousarray(hidx, dtype=np.int32))
    g_m = W.get(cp, "id_m", (nl, 3), cp.float32)
    g_m.set(np.ascontiguousarray(mbear, dtype=np.float32))
    g_hp = W.get(cp, "id_hpose", (nl, 12), cp.float32)
    g_hp.set(np.ascontiguousarray(hpose_ext, dtype=np.float32))
    g_rho = W.get(cp, "id_rho", (nl,), cp.float32)

    Jp = W.get(cp, "id_Jp", (N, 2, 2, d), cp.float32)
    resw = W.get(cp, "id_resw", (N, 2), cp.float32)
    kslot = W.get(cp, "id_kslot", (N, 2), cp.int32)
    Hpl = W.get(cp, "id_Hpl", (N, 2, d, 3), cp.float32)
    Hll = W.get(cp, "id_Hll", (nl, 9), cp.float32)
    Hlli = W.get(cp, "id_Hlli", (nl, 9), cp.float32)
    bl = W.get(cp, "id_bl", (nl, 3), cp.float32)
    Hred = W.get(cp, "id_Hred", (n, n), cp.float32)
    grad = W.get(cp, "id_grad", (n,), cp.float32)
    dxp = W.get(cp, "id_dxp", (n,), cp.float32)
    dxl = W.get(cp, "id_dxl", (nl, 3), cp.float32)
    g_T = W.get(cp, "id_Tcw", (nk, 12), cp.float32)
    g_Hr = W.get(cp, "id_Hr", (n, n), cp.float32)
    g_gr = W.get(cp, "id_gr", (n,), cp.float32)
    g_dia = W.get(cp, "id_dia", (n,), cp.int32)
    g_dia[:] = cp.arange(n, dtype=cp.int32)
    h_T = np.empty((nk, 12), np.float32)
    h_step = np.empty(n, np.float32)
    h_dl = np.empty((nl, 3), np.float32)

    g_prior = None
    if prior is not None:
        m0, rho0, wbear, invsig, valid = prior
        g_prior = (cp.asarray(np.ascontiguousarray(m0), dtype=cp.float32),
                   cp.asarray(rho0, dtype=cp.float32),
                   cp.asarray(wbear, dtype=cp.float32),
                   cp.asarray(invsig, dtype=cp.float32),
                   cp.asarray(valid, dtype=cp.float32))

    free_mask = _free_mask(free, nk)
    keep = np.ones(n, np.float32)
    for i in np.nonzero(~free_mask)[0]:
        keep[i * d:(i + 1) * d] = 0.0
    g_keep = cp.asarray(keep)
    g_keep2 = g_keep[:, None] * g_keep[None, :]
    g_hdiag = cp.asarray(np.nonzero(keep == 0)[0].astype(np.int32))

    rho = np.array(rho, np.float64)
    mbear = np.array(mbear, np.float64)
    T_cw = se3_inv(np.asarray(poses, np.float64))
    lam = lam0
    chi_prev = None
    best = (T_cw.copy(), rho.copy(), mbear.copy(), None)
    rms0 = None

    for _it in range(iters):
        np.copyto(h_T, T_cw[:, :3, :].reshape(nk, 12))
        g_T.set(h_T)
        g_rho.set(np.ascontiguousarray(rho, dtype=np.float32))
        g_m.set(np.ascontiguousarray(mbear, dtype=np.float32))
        Hll.fill(0); bl.fill(0); Hred.fill(0); grad.fill(0)
        K.build_blocks_id(g_T, g_hp, g_hidx, g_m, g_rho, g_okf, g_olm,
                          g_ouv, Jp, resw, kslot, Hll, bl, Hpl, N,
                          rig.fx_rect, rig.cx, rig.cy, HUBER_K)
        if g_prior is not None:
            K.landmark_prior_id(g_m, g_rho, g_prior[0], g_prior[1],
                                g_prior[3], g_prior[2], g_prior[4],
                                Hll, bl, nl, HUBER_K)
        chi = float(cp.sum(resw * resw))
        if rms0 is None:
            rms0 = float(np.sqrt(chi / max(N, 1)))
        if chi_prev is not None and chi > chi_prev:
            T_cw, rho, mbear = (best[0].copy(), best[1].copy(),
                                best[2].copy())
            lam = min(lam * 4.0, 1e2)
            continue
        chi_prev = chi
        best = (T_cw.copy(), rho.copy(), mbear.copy(), chi)

        K.inv3_damped(Hll, Hlli, nl, lam)
        K.pose_blocks_id(Jp, resw, kslot, Hred, grad, N, d, n)
        if npair:
            K.schur_pairs_id(Hpl, Hlli, kslot, g_olm, g_pa, g_pb, Hred,
                             npair, d, n)
        K.schur_grad_id(Hpl, Hlli, bl, kslot, g_olm, grad, N, d)

        if rotf:
            Hr, gr, _c = _rot_factor_blocks(T_cw, rotf, d, n)
            g_Hr.set(np.ascontiguousarray(Hr, dtype=np.float32))
            g_gr.set(np.ascontiguousarray(gr, dtype=np.float32))
            Hred += g_Hr
            grad += g_gr

        Hred *= g_keep2
        Hred[g_hdiag, g_hdiag] = 1.0
        grad *= g_keep
        Hred[g_dia, g_dia] += lam * cp.maximum(Hred[g_dia, g_dia], 1e-9)

        K.chol_solve(Hred, grad, dxp, n)
        K.backsub_id(Hpl, Hlli, bl, dxp, kslot, g_offs, dxl, nl, d)

        dxp.get(out=h_step)
        dxl.get(out=h_dl)
        step = h_step.astype(np.float64).reshape(nk, d)
        dl = h_dl.astype(np.float64)
        if not (np.isfinite(step).all() and np.isfinite(dl).all()):
            lam = min(lam * 10.0, 1e3)
            continue
        if free_mask.any():
            T_cw[free_mask] = se3_exp(step[free_mask]) @ T_cw[free_mask]
        mbear[:, 0] += dl[:, 0]
        mbear[:, 1] += dl[:, 1]
        rho = np.maximum(rho + dl[:, 2], rho_floor)
        lam = max(lam * 0.5, 1e-7)

    T_cw[:, :3, :3] = so3_project(T_cw[:, :3, :3])

    np.copyto(h_T, T_cw[:, :3, :].reshape(nk, 12))
    g_T.set(h_T)
    g_rho.set(np.ascontiguousarray(rho, dtype=np.float32))
    g_m.set(np.ascontiguousarray(mbear, dtype=np.float32))
    Hll.fill(0); bl.fill(0)
    K.build_blocks_id(g_T, g_hp, g_hidx, g_m, g_rho, g_okf, g_olm,
                      g_ouv, Jp, resw, kslot, Hll, bl, Hpl, N,
                      rig.fx_rect, rig.cx, rig.cy, HUBER_K)
    chi_n = float(cp.sum(resw * resw))
    return (se3_inv(T_cw), rho, mbear,
            {"chi2_0": best[3] if best[3] else chi_n, "chi2_n": chi_n,
             "rms_0": rms0, "rms_n": float(np.sqrt(chi_n / max(N, 1))),
             "npair": int(npair), "n_obs": int(N)})


def warmup(rig=None):
    """Compile the kernels and run one throwaway solve BEFORE the row starts.

    The first solve of a cold process JIT-compiles the module and measured
    **948 ms** - which would land in the p95 of a 82-keyframe run and make the
    budget verdict a statement about nvrtc rather than about the backend.
    Compilation is a build cost, not a solve cost, so it is paid up front and
    said out loud rather than hidden inside the first keyframe.
    """
    from xclean.backend.ba_synth import make_problem
    pb = make_problem(nk=6, nl=60, seed=1)
    _solve_gpu(pb["poses0"], pb["pts0"], pb["okf"], pb["olm"], pb["ouv"],
               pb["rig"], 0, 2)
    return True
