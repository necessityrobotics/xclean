#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""LOOP CLOSURE - precision first, weights not booleans, never in the frame
path.

    lc = LoopCloser(rig, frontend, place, tracker)
    lc.enqueue(kf_id, kp, feats, lm_id, T_wc, image)   # frame path: a handoff
    res = lc.poll()                                     # non-blocking
    lc.apply(res, tracker)                              # the transaction

★ THE RECEIPT THIS FILE IS BUILT AROUND (plan 5.5)
    MH_04 accepted **522 welds** and got **3-8x worse**: its loops-off ablation
    scored 0.1846 against loops-on 1.4344.  The loop stack was doing net HARM,
    and no external system's ATE could have said so - only the engine's own
    ablation could.  So every rule here is about ADMISSION, not recall:

    * **every gate is an object** with a name, a printed threshold, a dial and
      a logged reason.  A rejection is never a bare `False`, and a DISABLED
      gate is a logged STATE - a structure-BA path once silently re-armed a
      cooldown and disabled loop closure for a whole run, and the log said
      nothing because "off" was silence;
    * **every pathway draws from its OWN RNG stream.**  Two builds diverged at
      closure #533 although one made ZERO welds, because three verification
      pathways shared one sequential generator.  The obvious repair is already
      refuted: content-derived per-call seeding gave mh02 0.0438 vs 0.0171;
    * **no hard accept/reject on a continuous quantity.**  A marginal weld
      contributes WEAKLY.  That removes the cliff rather than choosing a better
      statistic on it - mag1 spread 13.5x from a 0.4 % perturbation flipping a
      binary outcome;
    * **`max()` is banned.**  The retrieval threshold is a quantile of the
      CANDIDATE POPULATION with a sufficiency floor of 30, a declared cold
      start, and **scaled** MAD (x1.4826 - unscaled leaked 75 mag1 aliases,
      scaled leaked 0);
    * **sub-30-inlier poses are never welded raw** (mag2 doubled its error).
      They are counted and left to a pooled pathway, which when it exists may
      only touch candidates this build REJECTS - a strict superset.  Replacing
      top-k retrieval outright was refuted in one test series; the union form
      fixed it.

★ ASYNCHRONOUS IS A CORRECTNESS PROPERTY, NOT AN OPTIMISATION
    "Loop correction contributing anything to the frame path is a FAILURE, not
    a slow path" (plan 3.4).  Retrieval, verification and the PGO all run on a
    worker thread over a snapshot.  The frame path does two things: hand off,
    and poll.  Applying a finished correction is a bounded transaction and is
    budgeted as its own per-event process (`loop_apply`) rather than hidden.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from collections import namedtuple

import numpy as np

from xclean.core.derive import (MIN_LOOP_INLIERS, MIN_PNP_INLIERS,
                                  HUBER_K)
from xclean.core.lie import se3_exp, se3_log, se3_inv, so3_project, adjoint
from xclean.obs.decision import Decision, Gate

SCALED_MAD = 1.4826          # Class D: makes MAD an estimate of sigma
RETRIEVAL_K = 3.0            # Class D: median + K*sigma outlier rule
SUFFICIENCY = 30             # samples before a quantile means anything

# ---- FIX A (tier 1, staged 2026-10-01): XTEN_LOOP_FISHER=infeasible -----------------------------------------
# THE DEFECT.  The retrieval threshold median + K x scaled MAD is an outlier rule for an UNBOUNDED variable,
# applied here to a cosine similarity, which cannot exceed COSINE_BOUND = 1 (Cauchy-Schwarz on L2-normalised
# descriptors).  Where every keyframe looks alike (MadMax desert, Rosario fields) the threshold reaches >= 1
# and NO candidate can pass, however strong the true revisit: madmax_F3's end-to-start revisit at similarity
# 0.935-0.94 meets raw thresholds of 1.17-1.20.
#
# THE RULE.  ONLY for a query whose raw threshold is INFEASIBLE (thr >= COSINE_BOUND), evaluate the SAME rule
# (_outlier_rule: the same K, the same scaled-MAD estimator, the same 1e-6 floor) and the SAME soft-gate weight
# formula (Gate.soft with width = max(MAD, 1e-6)) on the Fisher transform atanh(s) of the SAME sample, s
# clipped to [-1 + FISHER_EPS, 1 - FISHER_EPS] in float64 so atanh stays finite at |s| = 1 (and at float32
# round-off just above 1).  Every query with thr < 1 is evaluated exactly as before, bit for bit.
# No new tuned constant: COSINE_BOUND is the support of the variable; FISHER_EPS is the float64 machine epsilon
# (a numerical guard, not a dial).  Unlike the global arm XTEN_RETRIEVAL_FISHER=1 ("fsh", rejected after
# kitti_00 +3.8 %), the transform is applied ONLY where the raw rule has no feasible region.
#
# STATED, NOT HIDDEN:
#   * the switch is a cliff at the bound: a query at raw thr 0.99 stays on the raw rule (it needs s > 0.99);
#     one at 1.00 drops to tanh(thr_F) ~ 0.87-0.95.  Report it as "applied only where the raw rule has no
#     feasible region", NOT as neutral everywhere;
#   * it is NOT bit-identical on ~25 benchmark sequences (the desert / Rosario ones, 4s 173/174/102/101/120,
#     mh_01/02/03, v1_03, v2_01/02/03, malaga_01, kitti_01): every one needs the n=3 no-harm gate.
# The flag is read ONCE, at LoopCloser construction: unset / "" / "0" / "off" -> off, "infeasible" -> on,
# anything else is a RuntimeError (a misspelt queue flag must not silently measure the default), and so is
# combining it with the global arm XTEN_RETRIEVAL_FISHER=1 (two different retrieval rules).
# Provenance, LoopCloser.summary(): fisher_mode, queries_total (queries that reached the threshold),
# queries_infeasible (raw thr >= 1; counted in BOTH modes, None under the global arm), fallback_accepts.
# A fallback query is logged under its own reason (FISHER_REASON), so its margin / dials never share a
# dial-reservoir key with raw-rule queries; its dials carry the raw rule as raw_median / raw_scaled_mad /
# raw_threshold beside the Fisher-scale median / scaled_mad / threshold and threshold_sim = tanh(threshold).
LOOP_FISHER_ENV = "XTEN_LOOP_FISHER"
LOOP_FISHER_MODES = ("off", "infeasible")
COSINE_BOUND = 1.0                          # the support of a cosine similarity, not a dial
FISHER_EPS = float(np.finfo(np.float64).eps)  # float64 machine epsilon: keeps atanh finite at |s| = 1
FISHER_REASON = "best candidate kf %d (Fisher fallback: raw thr >= 1)"

RetrievalRule = namedtuple("RetrievalRule", "score med mad thr fallback raw")


def loop_fisher_mode(env=None):
    """The run's Fix A mode from XTEN_LOOP_FISHER: unset / "" / "0" / "off" -> "off"; "infeasible" ->
    "infeasible"; anything else is a RuntimeError (the XTEN_KF_HEALTH contract)."""
    v = (os.environ if env is None else env).get(LOOP_FISHER_ENV, "")
    v = (v or "").strip().lower()
    if v in ("", "0", "off"):
        return "off"
    if v == "infeasible":
        return "infeasible"
    raise RuntimeError("%s=%r is not a mode: unset/off or infeasible" % (LOOP_FISHER_ENV, v))


def _outlier_rule(score):
    """THE retrieval rule on one sample -> (median, scaled MAD, threshold = median + K x max(MAD, 1e-6)).
    The raw rule and the Fix A fallback both call this, so the two cannot drift apart."""
    med = float(np.median(score))
    mad = float(np.median(np.abs(score - med))) * SCALED_MAD
    thr = med + RETRIEVAL_K * max(mad, 1e-6)
    return med, mad, thr


def fisher_score(sims):
    """atanh of the similarity sample, in float64, clipped to [-1 + FISHER_EPS, 1 - FISHER_EPS]."""
    s = np.asarray(sims, dtype=np.float64)
    return np.arctanh(np.clip(s, -1.0 + FISHER_EPS, 1.0 - FISHER_EPS))


def retrieval_rule(sims, fisher_global=False, loop_fisher="off"):
    """One query's retrieval threshold -> RetrievalRule(score, med, mad, thr, fallback, raw).

    score          the variable the soft gate reads: `sims` itself (the raw rule), atanh under the global
                   arm (XTEN_RETRIEVAL_FISHER=1, unchanged), or fisher_score(sims) on a Fix A fallback
    med, mad, thr  _outlier_rule(score)
    fallback       True only when loop_fisher == "infeasible", the global arm is off and the RAW
                   threshold is >= COSINE_BOUND
    raw            the raw rule's (med, mad, thr) on a fallback, else None
    """
    score = np.arctanh(np.clip(sims, -0.999999, 0.999999)) if fisher_global else sims
    med, mad, thr = _outlier_rule(score)
    if loop_fisher == "infeasible" and not fisher_global and thr >= COSINE_BOUND:
        raw = (med, mad, thr)
        score = fisher_score(sims)
        med, mad, thr = _outlier_rule(score)
        return RetrievalRule(score, med, mad, thr, True, raw)
    return RetrievalRule(score, med, mad, thr, False, None)


# ---- PROX SPIKE GATE (tier 1, staged 2026-10-03): XTEN_PROX_PATH=despike ----------------------------------------
# THE DEFECT.  The pi path/direct test (_consider_proximity, and _path_ratio for appearance candidates under the
# default --weld-ratio-gate) reads "went away and came back" off the keyframe chain as it stands.  ONE aliased
# keyframe that is placed far off and then returns (an out-and-back spike) satisfies it twice over: its two legs
# inflate the path, and its own position can sit next to keyframes the vehicle reaches later, so the candidate is
# the spike itself.  Measured on kitti_01: keyframe 381 (frame 389) placed
# 53.2 m out and 61.9 m back (GT steps 2.6 m; chord 11.2 m); keyframes 411-435 then saw it as their nearest
# gap-passing candidate at 4-57 m with raw ratios 3.2-26.6 > pi; two of the 25 PnP checks passed on the
# repetitive barrier (33 and 36 inliers) and welded 381 -> 425 / 428, which are 116.4 / 124.3 m apart in GT.
#
# THE RULE.  A keyframe is a SPIKE when (a) the 3-point vector median of (previous, itself, next) is NOT itself -
# its position is supported by neither neighbour (the chord between the neighbours is shorter than one of its
# legs) - and (b) BOTH legs are outliers of the query's own candidate step population by the closer's own
# outlier rule (_outlier_rule: median + RETRIEVAL_K x scaled MAD, the retrieval threshold's estimator).  Spikes
# are replaced by that vector median (a neighbour's position); every other keyframe is untouched.  The path
# from the candidate AND the candidate-to-query distance are re-measured on that chain (step_med and the
# gap-window term are unchanged), and the candidate passes only if BOTH the raw ratio and the despiked ratio
# clear pi.  So the flag can only REFUSE what the default admits, never admit anything new, and on a chain with
# no spike it is the default decision exactly.  No new constant: pi, RETRIEVAL_K, SCALED_MAD and the 3-point
# window (the smallest window that removes a single-sample excursion) are all already the closer's.
# NARROWED: the veto acts only on a
# pair ONE OF WHOSE OWN ENDPOINTS - the candidate or the query - IS a spike (the defect is "the candidate is the spike
# itself"); a small glitch elsewhere on the path no longer vetoes every borderline pair downstream of it (the
# verifier's vfix v2_03 chain: 311 appearance flips at 3-6 m -> 7; kitti_01: 25 proximity refusals -> 25, both false
# welds still refused).  On the proximity path the veto acts only INSIDE the soft radius (r_prox - distance > 0):
# outside it the soft gate rejects the pair anyway, so out-of-radius records stay the default's and
# spike_refused_prox counts only refusals that changed a decision.
# The refusal's dials carry spike_endpoint ("candidate" / "query" / "both").
# DISCLOSED: "only refuses" holds at the GATE level.  At the WELD level a refused hand-off draws no verification
# seed (_verify calls cv2.setRNGSeed(self.s_verify.cv_seed()) once per hand-off that reaches PnP), so every later
# PnP verification of the run sees a shifted stream and a later borderline verification can flip either way.
#
# STATED, NOT HIDDEN: a spike of TWO or more consecutive keyframes is not removed (window 3), and a forward
# displacement (kitti_01 frames 481-485: steps 51-257 m that do NOT come back) is not a spike by (a) - that is
# a front-end defect (XTEN_PNP_VERIFY / XTEN_FRONT_GATE), not this gate's.  Neighbours are taken in `order`
# (the query's successor is used when it exists, i.e. at finalise), and the endpoints of the fetched chain are
# never spikes.
# The flag is read ONCE, at LoopCloser construction: unset / "" / "0" / "off" -> off; "despike" -> on; anything
# else is a RuntimeError.  Provenance (LoopCloser.summary(), present ONLY when the flag is on): prox_path,
# spike_refused_prox, spike_refused_samepath.  Default path: byte-identical (tests/test_prox_spike.py).
PROX_PATH_ENV = "XTEN_PROX_PATH"
PROX_PATH_MODES = ("off", "despike")


def prox_path_mode(env=None):
    """The run's proximity-path mode from XTEN_PROX_PATH: unset / "" / "0" / "off" -> "off"; "despike" ->
    "despike"; anything else is a RuntimeError (the XTEN_LOOP_FISHER contract)."""
    v = (os.environ if env is None else env).get(PROX_PATH_ENV, "")
    v = (v or "").strip().lower()
    if v in ("", "0", "off"):
        return "off"
    if v == "despike":
        return "despike"
    raise RuntimeError("%s=%r is not a mode: unset/off or despike" % (PROX_PATH_ENV, v))


def despike_chain(P, step_thr):
    """(F, spike) for a keyframe position chain P (n x 3, in order): spike[k] is True for an interior keyframe
    whose 3-point vector median is a neighbour (ties keep the keyframe itself) AND whose two legs both exceed
    `step_thr`; F is P with every spike replaced by that neighbour.  Endpoints are never spikes."""
    P = np.asarray(P, dtype=np.float64)
    spike = np.zeros(len(P), dtype=bool)
    if len(P) < 3:
        return P.copy(), spike
    a, b, c = P[:-2], P[1:-1], P[2:]
    ab = np.linalg.norm(a - b, axis=1)
    bc = np.linalg.norm(b - c, axis=1)
    ac = np.linalg.norm(a - c, axis=1)
    sa, sb, sc = ab + ac, ab + bc, ac + bc
    keep = (sb <= sa) & (sb <= sc)
    use_a = (~keep) & (sa <= sc)
    spike[1:-1] = (~keep) & (np.minimum(ab, bc) > float(step_thr))
    F = P.copy()
    sel = spike[1:-1]
    F[1:-1][sel] = np.where(use_a[sel][:, None], a[sel], c[sel])
    return F, spike


class _Stream:
    """One RNG per pathway.  Not a nicety: g2 and g2c diverged at closure #533
    although g2c made ZERO welds, because three pathways shared one generator
    and every draw shifted the others."""

    def __init__(self, name, seed):
        self.rng = np.random.default_rng(
            abs(hash((name, int(seed)))) % (2 ** 32))

    def cv_seed(self):
        return int(self.rng.integers(0, 2 ** 31 - 1))


class LoopCloser:
    def __init__(self, rig, frontend, place, tracker, min_gap=30, seed=0,
                 enabled=True, pgo_every=8, verbose=True, apply_online=False,
                 prox_loops=False, weld_ratio_gate=False):
        self.rig = rig
        self.fe = frontend
        self.place = place
        self.tracker = tracker
        self.min_gap = int(min_gap)
        self.enabled = bool(enabled)
        self.pgo_every = int(pgo_every)
        self.apply_online = bool(apply_online)
        self.verbose = verbose

        self.kf = {}
        self.order = []
        self._Dbuf = None                   # xclean: growable (n, V) store; `D` is its live prefix (see the property)
        self._nD = 0
        self.n_rich_kf = 0                  # xclean: keyframes with a rich verification set, and its 3D points
        self.n_rich_3d = 0
        self._epi = None
        self.welds = []
        self.n_considered = 0
        self.n_since_pgo = 0
        self.worker_ms = []
        self.pgo_ms = []
        self.pgo_mode = "6dof"
        import os as _os
        self.retrieval_fisher = _os.environ.get("XTEN_RETRIEVAL_FISHER") == "1"
        # FIX A (XTEN_LOOP_FISHER=infeasible, see LOOP_FISHER_ENV): read ONCE, here, so a misspelt queue flag
        # fails at construction, before the first frame.  The counters are provenance only (summary()); no
        # decision reads them.
        self.loop_fisher = loop_fisher_mode()
        if self.loop_fisher != "off" and self.retrieval_fisher:
            raise RuntimeError("%s=%s cannot be combined with XTEN_RETRIEVAL_FISHER=1 (the global Fisher arm "
                               "already transforms every query)" % (LOOP_FISHER_ENV, self.loop_fisher))
        self.n_fisher_queries = 0
        self.n_fisher_infeasible = 0
        self.n_fisher_accepts = 0
        # PROX SPIKE GATE (XTEN_PROX_PATH=despike, see PROX_PATH_ENV): read ONCE, here; counters are provenance.
        self.prox_path = prox_path_mode()
        self.n_spike_prox = 0
        self.n_spike_samepath = 0
        # ARM XTEN_LOOP_COVIS=1 (2026-09-25): see _covis_pool. Counters are provenance, not tuning.
        self.loop_covis = _os.environ.get("XTEN_LOOP_COVIS") == "1"
        if self.loop_covis:
            # XC-28: the covisible pool indexes tracker keypoints and landmark ids while xclean verifies on the rich
            # sets (lm id -1 wraps, two index spaces mixed) - refused rather than silently wrong
            raise RuntimeError("XTEN_LOOP_COVIS is not supported by xclean (rich-set verification); unset it")
        self.n_covis_tried = 0
        self.n_covis_pooled = 0
        self.n_covis_rescued = 0
        self.covis_nbrs = []
        self.pgo_info = "isotropic"
        self.pgo_gravity = None
        # T1-e provenance (staged 2026-09-29): what the LAST pose graph used; None until a graph has run.
        #   pgo_huber = {"mode": "1d"|"dimk", "k": float, "D": int}
        #   pgo_odo   = {"mode": "scalar"|"median"|"edge", "edges": int, "evidence": int, "fallback": int}
        self.pgo_huber = None
        self.pgo_odo = None
        self.n_pgo = 0

        # arm --prox-loops: the geometrically
        # NEAREST gap-passing keyframe by CURRENT pose is verified in
        # addition to the appearance best.  Aimed at v2_01's weld
        # starvation (11-15 welds vs 126-294; retrieval AND verify 2-4x
        # worse there - the early yaw bend displaces stored landmarks, but
        # room-scale pose displacement is only cm, so proximity still
        # finds the revisit).  The strict verification is UNCHANGED - a
        # strict superset of candidates, the form the union-retrieval
        # refutation permits (header note above).  The plausibility radius
        # is DERIVED: median inter-candidate step x min_gap = the distance
        # one gap window covers (population: this run's own keyframes).
        self.prox_loops = bool(prox_loops)
        # arm --weld-ratio-gate: the pi path/direct test applied to
        # APPEARANCE candidates too (the kitti_01 false-weld fix)
        self.weld_ratio_gate = bool(weld_ratio_gate)
        self.n_prox_verified = 0
        self.g_retrieval = Gate("loop_retrieval")
        self.g_gap = Gate("loop_gap", threshold=self.min_gap)
        self.g_verify = Gate("loop_verify", threshold=MIN_LOOP_INLIERS)
        self.g_subbar = Gate("loop_subbar", threshold=MIN_LOOP_INLIERS)
        self.g_prox = Gate("loop_prox")
        self.g_samepath = Gate("loop_samepath")
        self.gates = [self.g_retrieval, self.g_gap, self.g_verify,
                      self.g_subbar] + ([self.g_prox] if prox_loops else []) \
            + ([self.g_samepath] if weld_ratio_gate else [])

        self.s_retrieval = _Stream("retrieval", seed)
        self.s_verify = _Stream("verify", seed)
        self.s_pool = _Stream("pool", seed)

        self._in = queue.Queue()
        self._out = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if verbose:
            print(" [loop] async worker up: gap %d KFs, retrieval = median + "
                  "%.1f x scaled MAD, verify floor %d inliers (soft)"
                  % (self.min_gap, RETRIEVAL_K, MIN_LOOP_INLIERS))
            if self.loop_fisher != "off":
                print(" [loop] %s=%s: a query whose raw threshold is >= %g is re-evaluated by the same rule "
                      "on atanh(sim)" % (LOOP_FISHER_ENV, self.loop_fisher, COSINE_BOUND))
            if self.prox_path != "off":
                print(" [loop] %s=%s: the pi path/direct gate must also clear pi on the despiked keyframe "
                      "chain" % (PROX_PATH_ENV, self.prox_path))

    # ---- xclean: the descriptor store grows by doubling (a 10^4-word BoW vector per keyframe made the engine's
    # per-keyframe np.vstack O(n^2) in copies) -----------------------------------------------------------------
    @property
    def D(self):
        return None if self._Dbuf is None else self._Dbuf[:self._nD]

    def _append_desc(self, desc):
        desc = np.asarray(desc, np.float32)
        if self._Dbuf is None:
            self._Dbuf = np.zeros((256, len(desc)), np.float32)
        elif self._nD >= len(self._Dbuf):
            g = np.zeros((2 * len(self._Dbuf), self._Dbuf.shape[1]), np.float32)
            g[:self._nD] = self._Dbuf[:self._nD]
            self._Dbuf = g
        self._Dbuf[self._nD] = desc
        self._nD += 1

    # ---------------------------------------------------- the frame path --
    def enqueue(self, kf_id, kp, feats, lm_id, T_wc, image, right=None, disp=None):
        """The ONLY thing the frame path does.  A queue put, and nothing else (xclean: the right image and the SGM map
        travel with it, for the rich keypoints' depth, computed on the worker)."""
        if not self.enabled:
            return False
        self._in.put((int(kf_id), kp, feats, lm_id, np.array(T_wc), image, right, disp))
        return True

    def poll(self):
        """Non-blocking.  Returns a finished correction or None."""
        try:
            return self._out.get_nowait()
        except queue.Empty:
            return None

    def apply(self, res, tracker):
        """The transaction.  The only place a loop correction touches state."""
        corrected, snapshot, ms, nw = res
        tracker._kf_pose_snapshot = snapshot
        moved = 0
        for k, T in corrected.items():
            if k in tracker.state.kf_pose:
                tracker.state.kf_pose[k] = T
                moved += 1
        n_lm = tracker.rehost(corrected)
        return Decision("loop_apply", True,
                        "PGO over %d keyframes and %d welds landed: %d poses, "
                        "%d landmarks rehosted (%.1f ms of solve, off the "
                        "frame path)" % (len(corrected), nw, moved, n_lm, ms),
                        {"keyframes": moved, "landmarks": n_lm, "welds": nw,
                         "solve_ms": ms})

    def close(self):
        self._stop.set()

    # -------------------------------------------------------- the worker --
    def _run(self):
        while not self._stop.is_set():
            try:
                job = self._in.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                t0 = time.perf_counter()
                self._handle(job)
                self.worker_ms.append((time.perf_counter() - t0) * 1e3)
            except Exception as e:      # a loop failure must never take the
                                        # run down: it is logged and skipped
                self.g_verify.reject("worker exception: %s: %s"
                                     % (type(e).__name__, str(e)[:80]), {})

    def _handle(self, job):
        """★ THE WORKER ONLY DESCRIBES.  Verification moved to `finalise`.

        Retrieval and verification read the LIVE landmark array while the main
        thread is still writing to it, so the weld set depended on thread
        timing: the same build produced 116, 121, 128, 130 and 135 welds on
        five runs, and the ATE followed (V1_01 0.0405 / 0.0420 / 0.0675 /
        0.0911 / 0.1095).  Every A/B decision taken on top of that was being
        taken on noise, which is precisely the failure this project has
        documented over and over - "a row with no measured floor produces a
        diagnostic, not a verdict".

        Computing the descriptor is a pure function of the image, so it stays
        on the worker and stays off the frame path.  Everything that reads the
        MAP happens once, at the end, against a map that is no longer moving.
        """
        kf_id, kp, feats, lm_id, T_wc, image, right, disp = job
        desc, r_uv, r_desc = self.place.describe_full(image)
        self.kf[kf_id] = {"kp": kp, "feats": feats, "lm_id": lm_id,
                          "T_wc": T_wc, "rich": self._rich(r_uv, r_desc, image, right, disp, kp, lm_id)}
        self.order.append(kf_id)
        self._append_desc(desc)
        if self.apply_online:
            self._consider(kf_id, desc)
        self.n_since_pgo += 1
        # ★ THE CORRECTION IS DEFERRED, AND THAT IS A CORRECTNESS DECISION.
        # Applying a pose-graph correction INCREMENTALLY, while the tracker
        # keeps running, tears the map: landmarks whose host keyframe is not in
        # the correction do not move, the live pose jumps away from the local
        # structure it was just solved against, and the next PGO then treats
        # the skewed trajectory as its own odometry measurement.  Measured on
        # V1_01, and it is not subtle - two runs of the same build:
        #
        #     loops off               ATE 0.0712, reprojection 0.998 -> 0.653 px
        #     loops on, applied live  ATE 0.4571 (one run) / 0.9725 (another),
        #                             44 % of frames COASTING, and a global BA
        #                             that started at 113,000,000 px
        #
        # Two runs differing by 2x on the same code is the signature of a
        # cliff, which this project has paid for before.  So detection and
        # verification stay online and asynchronous - they are cheap and they
        # are where the evidence is gathered - and the CORRECTION happens once,
        # at the end, through PGO and then the global bundle adjustment, which
        # carries the same welds.  The frame path is then trivially free of
        # loop correction rather than nearly free.
        if self.apply_online and self.welds \
                and self.n_since_pgo >= self.pgo_every:
            self.n_since_pgo = 0
            self._pgo()

    def _rich(self, uv, desc, left, right, disp, kp, lm_id):
        """xclean: the keyframe's RICH verification set - the retrieval ORB set (1000 oriented multi-scale features)
        with stereo 3D in the keyframe's camera frame, measured exactly as tracker._create measures a landmark
        (stereo/epipolar.keypoint_disparity: SGM quad rule + edge test + the keypoint epipolar check, inside the
        derived band near clip .. far bearing), and, where a rich keypoint lies within FEATURE_SIGMA_PX of a tracker
        keypoint that carries a landmark, that landmark's id (for the structure welds).
        Measured on mh_04 revisit pairs (c1): the tracker's ~200 keypoints gave a median 16 geometric inliers per
        pair (15 % of pairs reach the 30-inlier loop floor); this set gives 60 (80 %)."""
        from xclean.core.derive import DISPARITY_NOISE_PX, FEATURE_SIGMA_PX
        from xclean.stereo.epipolar import keypoint_disparity, EpipolarMatcher
        n = len(uv)
        X = np.full((n, 3), np.nan, np.float32)
        lm = np.full(n, -1, np.int64)
        if n and disp is not None and right is not None:
            if getattr(self, "_epi", None) is None:
                self._epi = EpipolarMatcher()
            d, on_edge, _qv, _n = keypoint_disparity(uv.astype(np.float64), disp, left, right, self._epi,
                                                     6.0 * DISPARITY_NOISE_PX)
            rig = self.rig
            z = np.where(d > 1e-6, rig.fB / np.maximum(d, 1e-6), np.inf)
            ok = (d > 0) & (z >= rig.near_clip_m) & (z <= rig.far_bearing_m) & ~on_edge
            X[ok, 0] = (uv[ok, 0] - rig.cx) * z[ok] / rig.fx_rect
            X[ok, 1] = (uv[ok, 1] - rig.cy) * z[ok] / rig.fx_rect
            X[ok, 2] = z[ok]
            if kp is not None and len(kp):
                from scipy.spatial import cKDTree
                dd, ii = cKDTree(np.asarray(kp, np.float64)).query(uv.astype(np.float64), k=1,
                                                                   distance_upper_bound=FEATURE_SIGMA_PX)
                hit = np.isfinite(dd)
                lm[hit] = np.asarray(lm_id)[ii[hit]]
        self.n_rich_kf += 1
        self.n_rich_3d += int(np.isfinite(X[:, 0]).sum())
        return {"uv": uv, "desc": desc, "X": X, "lm": lm}

    def _consider_upto(self, kf_id, j):
        """Consider `kf_id` against the keyframes that PRECEDE it.

        Identical to the online form except that the candidate set is the
        prefix of the order rather than "whatever the worker had seen", which
        is what makes the result reproducible.
        """
        self._n_upto = j
        try:
            self._consider(kf_id, self.D[j])
        finally:
            self._n_upto = None

    # ------------------------------------------------------- retrieval ----
    def _consider(self, kf_id, me):
        self.n_considered += 1
        n = (upto + 1) if (upto := getattr(self, "_n_upto", None)) is not None \
            else len(self.order)
        if n < SUFFICIENCY + self.min_gap:
            self.g_retrieval.reject(
                "cold start: %d keyframes, need %d before a candidate "
                "population exists" % (n, SUFFICIENCY + self.min_gap),
                {"keyframes": n})
            return
        upto = getattr(self, "_n_upto", None)
        ids = np.array(self.order[:upto if upto is not None else -1],
                       dtype=np.int64)
        m = (kf_id - ids) >= self.min_gap
        if m.sum() < SUFFICIENCY:
            self.g_gap.reject("too few candidates past the gap",
                              {"candidates": int(m.sum())})
            return
        sims = self.D[:upto if upto is not None else -1][m] @ me
        cand_ids = ids[m]

        # ARM XTEN_RETRIEVAL_FISHER=1 (2026-09-25): the median + K*MAD outlier rule assumes an UNBOUNDED variable,
        # but a cosine similarity cannot exceed 1. Where every keyframe looks alike (MadMax desert) the threshold
        # rises above 1 and NO candidate can ever pass - measured on madmax_F3: ~79 % of late queries, the
        # end-to-start revisit at similarity 0.935-0.94 against a threshold of 1.17-1.20. The Fisher z-transform
        # atanh(s) is the standard variance-stabilising map for correlation-like quantities: it is unbounded, so
        # the same rule (same K, same MAD) can always be met. No new constant. Default: the raw rule.
        # FIX A (XTEN_LOOP_FISHER=infeasible): retrieval_rule applies the same transform ONLY to a query whose
        # raw threshold is >= COSINE_BOUND (rr.fallback); every other query is the raw rule, bit for bit.
        fisher = self.retrieval_fisher
        rr = retrieval_rule(sims, fisher_global=fisher, loop_fisher=self.loop_fisher)
        score, med, mad, thr = rr.score, rr.med, rr.mad, rr.thr
        self.n_fisher_queries += 1
        if not fisher and (rr.fallback or thr >= COSINE_BOUND):
            self.n_fisher_infeasible += 1
        self.g_retrieval.threshold = thr
        best = int(np.argmax(sims))
        dials = {"best_sim": float(sims[best]), "median": med,
                 "scaled_mad": mad, "threshold": thr,
                 "candidates": int(len(cand_ids)),
                 "candidate_kf": int(cand_ids[best])}
        if fisher or rr.fallback:
            dials["threshold_sim"] = float(np.tanh(thr))
        if rr.fallback:
            dials["raw_median"], dials["raw_scaled_mad"], dials["raw_threshold"] = rr.raw
        d = self.g_retrieval.soft(float(score[best]) - thr,
                                  (FISHER_REASON if rr.fallback
                                   else "best candidate kf %d") % cand_ids[best],
                                  dials, width=max(mad, 1e-6))
        if rr.fallback and d.accepted:
            self.n_fisher_accepts += 1
        if d.accepted:
            # ★ REFUTED, 2026-08-25, and left recorded rather than deleted.
            # The hypothesis was that V1_01's "loops" are really covisibility -
            # a 30-keyframe gap is about 3 m in a 5 m room - and that rejecting
            # covisible candidates would recover its no-loop score.  Measured
            # over three draws each: V1_01 did NOT improve (0.0967 / 0.0924
            # against 0.0926 / 0.0983 without the gate) and MH_02 got slightly
            # worse (0.0423 median against 0.0392).  The gate is off; the
            # helper below stays because the question will come back at R4,
            # where the rig scales change by 12x.
            cand = int(cand_ids[best])
            # ★ arm --weld-ratio-gate: the
            # SAME pi path/direct test the proximity path carries, applied
            # to APPEARANCE candidates.  kitti_01 (highway, zero true
            # revisits) accepted 124 welds through retrieval AND strict
            # PnP - perceptual aliasing on repetitive structure - and the
            # PGO+GBA were handed wreckage (est path 3.09x GT).  A
            # candidate whose connecting path is barely longer than its
            # direct distance is a same-road pair, not a loop.
            if self.weld_ratio_gate:
                rr = self._path_ratio(kf_id, cand_ids, cand)
                if rr is not None and rr < np.pi:
                    self.g_samepath.reject(
                        "path/direct ratio %.2f < pi: same-road pair, "
                        "not a revisit" % rr, {"candidate": cand,
                                               "ratio": rr})
                    cand = None
                elif rr is not None and self.prox_path == "despike":
                    sp = self._despiked_ratio(kf_id, cand_ids, cand)
                    if sp is not None and sp[0] < np.pi and sp[4] is not None:
                        self.n_spike_samepath += 1
                        self.g_samepath.reject(
                            "despiked path/direct ratio %.2f < pi (raw %.2f): "
                            "a keyframe spike, not a revisit" % (sp[0], rr),
                            {"candidate": cand, "ratio": rr,
                             "ratio_despiked": sp[0], "path_despiked": sp[1],
                             "distance_despiked": sp[2], "spikes": sp[3],
                             "spike_endpoint": sp[4]})
                        cand = None
            if cand is not None:
                self._verify(kf_id, cand, d.weight)
        if self.prox_loops:
            self._consider_proximity(kf_id, cand_ids,
                                     int(cand_ids[best]) if d.accepted
                                     else -1)

    def _positions(self, cand_ids):
        """Candidate positions, memoised ONLY on the finalise path (_n_upto
        set), where every pose is final and static - measured 2x wall on
        kitti_00 (11 -> 22 min) without the memo, O(n^2) pose fetches.  The
        online path (apply_online) keeps live fetches: poses move there."""
        offline = getattr(self, "_n_upto", None) is not None
        if offline:
            cache = getattr(self, "_pos_cache", None)
            if cache is None:
                cache = self._pos_cache = {}
            def _pos(k):
                p = cache.get(k)
                if p is None:
                    p = cache[k] = self._current_pose(k)[:3, 3].copy()
                return p
        else:
            def _pos(k):
                return self._current_pose(k)[:3, 3]
        return np.stack([_pos(int(c)) for c in cand_ids])

    def _path_ratio(self, kf_id, cand_ids, cand):
        """Path-travelled/direct-distance from `cand` to the current
        keyframe, the pi-gate's quantity (see _consider_proximity).
        None when the population is degenerate."""
        pos = self._positions(cand_ids)
        if len(pos) < 3:
            return None
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        ok = steps > 1e-9
        if not ok.any():
            return None
        step_med = float(np.median(steps[ok]))
        me_t = self._current_pose(kf_id)[:3, 3]
        idx = int(np.nonzero(cand_ids == cand)[0][0]) \
            if (cand_ids == cand).any() else None
        if idx is None:
            return None
        cum = np.concatenate([[0.0], np.cumsum(steps)])
        path_back = float(cum[-1] - cum[idx]) + step_med * self.min_gap
        dist = float(np.linalg.norm(pos[idx] - me_t))
        return path_back / max(dist, 1e-6)

    def _despiked_ratio(self, kf_id, cand_ids, cand):
        """XTEN_PROX_PATH=despike: the pi-gate's quantity re-measured on the despiked chain (see PROX_PATH_ENV)
        -> (ratio, path_back, distance, spikes in the chain, spike_endpoint), or None when the population is
        degenerate.  spike_endpoint: "candidate" / "query" / "both" when an endpoint of the pair is itself a spike
        (the veto acts only then), else None.

        The chain is `order` up to the query's successor; the query sits at index L (= len of the candidate
        prefix the caller used), the candidates are the gap-passing members of order[:L]."""
        upto = getattr(self, "_n_upto", None)
        L = upto if upto is not None else len(self.order) - 1
        if L < 0 or L >= len(self.order) or int(self.order[L]) != int(kf_id):
            raise RuntimeError("despiked ratio: query kf %s is not order[%d]" % (kf_id, L))
        if upto is not None:
            # finalise: every pose is final and static (the _positions memo's premise), so the whole order and its
            # positions are fetched ONCE and sliced per query - O(1) Python work per query instead of O(n)
            memo = getattr(self, "_spike_memo", None)
            if memo is None or len(memo[0]) != len(self.order):
                ids_all = np.asarray(self.order, dtype=np.int64)
                memo = self._spike_memo = (ids_all, self._positions(ids_all))
            ids_all, pos_all = memo
        else:
            ids_all = np.asarray(self.order[:L + 2], dtype=np.int64)
            pos_all = self._positions(ids_all)
        prior = ids_all[:L]
        cidx = np.nonzero((int(kf_id) - prior) >= self.min_gap)[0]
        if not np.array_equal(prior[cidx], np.asarray(cand_ids, dtype=np.int64)):
            raise RuntimeError("despiked ratio: candidate set is not the gap-passing prefix of order")
        hit = np.nonzero(prior[cidx] == int(cand))[0]
        if len(cidx) < 3 or len(hit) == 0:
            return None
        j = int(hit[0])
        chain = pos_all[:L + 2]
        pc = chain[cidx]
        steps = np.linalg.norm(np.diff(pc, axis=0), axis=1)
        ok = steps > 1e-9
        if not ok.any():
            return None
        step_med = float(np.median(steps[ok]))
        _, _, thr = _outlier_rule(steps[ok])
        F, spike = despike_chain(chain, thr)
        fc = F[cidx]
        fcum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(fc, axis=0), axis=1))])
        path_back = float(fcum[-1] - fcum[j]) + step_med * self.min_gap
        dist = float(np.linalg.norm(fc[j] - F[L]))
        _sc, _sq = bool(spike[int(cidx[j])]), bool(spike[L])
        return (path_back / max(dist, 1e-6), path_back, dist, int(spike.sum()),
                ("both" if (_sc and _sq) else "candidate" if _sc else "query" if _sq else None))

    def _consider_proximity(self, kf_id, cand_ids, already_verified):
        """The nearest gap-passing keyframe by CURRENT pose, verified when it
        lies within one gap-window of travel (see __init__ note)."""
        me_t = self._current_pose(kf_id)[:3, 3]
        pos = self._positions(cand_ids)
        if len(pos) < 3:
            return
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        step_med = float(np.median(steps[steps > 1e-9])) \
            if (steps > 1e-9).any() else 0.0
        if step_med <= 0:
            self.g_prox.reject("degenerate step population (hover)",
                               {"step_median": step_med})
            return
        r_prox = step_med * self.min_gap
        dists = np.linalg.norm(pos - me_t[None, :], axis=1)
        j = int(np.argmin(dists))
        cand = int(cand_ids[j])
        if cand == already_verified:
            self.g_prox.reject("nearest == appearance best, already verified",
                               {"candidate": cand})
            return
        # ★ A REVISIT WENT AWAY AND CAME BACK; A FOLLOWER NEVER LEFT.
        # At driving scale the nearest gap-passing keyframe is barely past
        # min_gap on the same road - a weld there is quasi-odometry noise
        # (measured: kitti_07 +7% with welds 12->35, the prox test series's
        # F1).  Landmark-id covisibility CANNOT catch it (ids only survive
        # the continuous tracking chain, ~8-28 keyframes - measured by this
        # gate rejecting nothing in the prox2 test series).  The discriminator
        # that scales: the PATH LENGTH travelled from the candidate to now,
        # over the DIRECT distance.  A follower on a road scores ~1; the
        # tightest possible went-around-something return (a semicircle)
        # scores pi; every true revisit scores above it.  pi is geometry,
        # not tuning.
        cum = np.concatenate([[0.0], np.cumsum(steps)])
        path_back = float(cum[-1] - cum[j]) + step_med * self.min_gap
        ratio = path_back / max(float(dists[j]), 1e-6)
        if ratio < np.pi:
            self.g_prox.reject(
                "path/direct ratio %.2f < pi: same-road neighbour, not a "
                "revisit" % ratio,
                {"candidate": cand, "ratio": ratio,
                 "path_back": path_back, "distance": float(dists[j])})
            return
        if self.prox_path == "despike" and float(r_prox - dists[j]) > 0:
            # only a pair the soft radius gate below would admit (its weight > 0.5 iff the margin is > 0)
            sp = self._despiked_ratio(kf_id, cand_ids, cand)
            if sp is not None and sp[0] < np.pi and sp[4] is not None:
                self.n_spike_prox += 1
                self.g_prox.reject(
                    "despiked path/direct ratio %.2f < pi (raw %.2f): a "
                    "keyframe spike, not a revisit" % (sp[0], ratio),
                    {"candidate": cand, "ratio": ratio, "ratio_despiked": sp[0],
                     "path_back": path_back, "path_despiked": sp[1],
                     "distance": float(dists[j]), "distance_despiked": sp[2],
                     "spikes": sp[3], "spike_endpoint": sp[4]})
                return
        d = self.g_prox.soft(float(r_prox - dists[j]),
                             "nearest kf %d at %.2f m (radius %.2f)"
                             % (cand, float(dists[j]), r_prox),
                             {"candidate": cand, "distance": float(dists[j]),
                              "radius": r_prox, "step_median": step_med},
                             width=0.25 * r_prox)
        if d.accepted:
            self.n_prox_verified += 1
            self._verify(kf_id, cand, d.weight)

    def _shared(self, a, b):
        ra, rb = self.kf.get(a), self.kf.get(b)
        if ra is None or rb is None:
            return 0
        ia = set(int(x) for x in ra["lm_id"] if x >= 0)
        if not ia:
            return 0
        return sum(1 for x in rb["lm_id"] if int(x) in ia)

    def _is_covisible(self, a, b):
        return self._shared(a, b) >= MIN_PNP_INLIERS

    def _covis_pool(self, cand, b, lids0, q0):
        """ARM XTEN_LOOP_COVIS=1 - verify against the candidate's COVISIBLE NEIGHBOURHOOD, not the keyframe alone.

        Measured: on madmax_F3 retrieval nominates the true
        end-to-start revisit but 87 % of verifications die on "too few shared landmarks"; on all 24 starved GT
        revisit pairs the candidate keyframe alone puts a median 19 landmarks in the query's view, while the
        keyframes covisible with it put 57-946 (median 108). ORB-SLAM3 verifies a loop by projecting the map points
        of the candidate's covisible keyframes into the query; this is that supply, pooled by descriptor matching.

        Called only when the single-keyframe check has already failed, so every loop the default accepts is
        accepted identically (strict superset). No new constant: the neighbourhood is the keyframes within the
        loop gap of the candidate (min_gap - the span this closer already treats as one local map, the complement
        of what may be a loop) that are covisible with it by this closer's own rule (>= MIN_PNP_INLIERS shared
        landmarks, _is_covisible). The pooled correspondences then face the SAME PnP floor (MIN_LOOP_INLIERS) and
        the same soft gate as a single-keyframe verification.
        """
        self.n_covis_tried += 1
        ra = self.kf.get(cand)
        if ra is None:
            return None
        own = np.unique(ra["lm_id"][ra["lm_id"] >= 0])
        if len(own) == 0:
            return None
        nbrs = []
        for k in range(cand - self.min_gap, cand + self.min_gap + 1):
            rk = self.kf.get(k)
            if k == cand or rk is None or rk["feats"] is None:
                continue
            lk = rk["lm_id"]
            sh = int(np.isin(lk[lk >= 0], own).sum())
            if sh >= MIN_PNP_INLIERS:
                nbrs.append((sh, k))
        self.covis_nbrs.append(len(nbrs))
        if not nbrs:
            return None
        nbrs.sort(key=lambda x: (-x[0], x[1]))        # most covisible first, deterministic ties
        L = [int(x) for x in lids0]
        Q = [int(x) for x in q0]
        seen_l, seen_q = set(L), set(Q)
        for _sh, k in nbrs:
            rk = self.kf[k]
            j0, j1 = self.fe.match_desc(rk["feats"], b["feats"], mask0=(np.asarray(rk["lm_id"]) >= 0))
            if len(j0) == 0:
                continue
            for l, q in zip(rk["lm_id"][j0], j1):
                l, q = int(l), int(q)
                if l < 0 or l in seen_l or q in seen_q:
                    continue
                seen_l.add(l)
                seen_q.add(q)
                L.append(l)
                Q.append(q)
        if len(L) < MIN_PNP_INLIERS:
            return None
        self.n_covis_pooled += 1
        return np.asarray(L, np.int64), np.asarray(Q, np.int64)

    # ----------------------------------------------------- verification ---
    def _verify(self, kf_id, cand, retrieval_weight):
        import cv2
        from xclean.track.pose import solve_pose

        a, b = self.kf.get(cand), self.kf.get(kf_id)
        if a is None or b is None or a.get("rich") is None or b.get("rich") is None:
            self.g_verify.reject("candidate evicted", {"candidate": cand})
            return
        # xclean: the RICH sets (see _rich): prediction-free ORB matching (mutual NN + ratio + chance bound) of the
        # candidate's keypoints that carry stereo 3D against all of the query's; PnP RANSAC at MIN_LOOP_INLIERS on
        # the candidate's 3D placed by its CURRENT pose (the role lm_xyz played) still decides
        ra, rb = a["rich"], b["rich"]
        has3d = np.isfinite(ra["X"][:, 0])
        i0, i1 = self.fe.match_desc({"descriptors": ra["desc"]}, {"descriptors": rb["desc"]}, mask0=has3d)
        if len(i0) == 0:
            self.g_verify.reject("no matches", {"candidate": cand})
            return
        lid = ra["lm"][i0]
        ok = np.ones(len(i0), bool)
        pooled = None
        if self.loop_covis and ok.sum() < MIN_PNP_INLIERS:
            pooled = self._covis_pool(cand, b, lid[ok], i1[ok])
        if ok.sum() < MIN_PNP_INLIERS and pooled is None:
            # margin + supply are recorded so the REJECT DISTRIBUTION is readable: a pile-up just under
            # the floor is a THRESHOLD problem, a pile-up far below it is a SUPPLY problem, and those two
            # demand opposite fixes. An earlier evaluation could only report that this gate fired 1,845 times on
            # rosario_06, never by how much. Informational only - reject() refuses regardless of margin.
            self.g_verify.reject("too few shared landmarks",
                                 {"candidate": cand,
                                  "with_landmarks": int(ok.sum()),
                                  "matches": int(len(i0)),
                                  "kf_landmarks": int((a["lm_id"] >= 0).sum())},
                                 margin=int(ok.sum()) - MIN_PNP_INLIERS)
            return
        if pooled is not None:
            P = self.tracker.lm_xyz[pooled[0]]
            uv = b["kp"][pooled[1]]
        else:
            T_a = self._current_pose(cand)
            Xa = ra["X"][i0].astype(np.float64)
            P = (T_a[:3, :3] @ Xa.T).T + T_a[:3, 3]
            uv = rb["uv"][i1].astype(np.float64)

        cv2.setRNGSeed(self.s_verify.cv_seed())      # THIS pathway's stream
        T_wc, mask, _dec = solve_pose(P, uv, self.rig,
                                      min_inliers=MIN_LOOP_INLIERS)
        inl = int(mask.sum())
        dials = {"candidate": cand, "matches": int(ok.sum()), "inliers": inl,
                 "retrieval_weight": float(retrieval_weight)}
        if pooled is not None:
            dials["covis_matches"] = int(len(pooled[0]))
        if T_wc is None:
            # ★ sub-bar evidence is COUNTED, never welded raw
            self.g_subbar.reject("sub-bar evidence: left to the pooled "
                                 "pathway, which does not exist yet and must "
                                 "be a strict superset when it does", dials)
            self.g_verify.reject("PnP under the loop floor", dials,
                                 margin=inl - MIN_LOOP_INLIERS)
            return
        d = self.g_verify.soft(inl - MIN_LOOP_INLIERS,
                               "verified against kf %d" % cand, dials,
                               width=MIN_LOOP_INLIERS)
        if not d.accepted:
            return
        if pooled is not None:
            self.n_covis_rescued += 1
        # the candidate's CURRENT pose, for the same reason
        T_i = self._current_pose(cand)
        Z = se3_inv(T_i) @ T_wc
        gap = float(np.linalg.norm(Z[:3, 3]))

        # ★ A WELD CARRIES THE INFORMATION ITS EVIDENCE EARNED, IN THE UNITS
        # THE SOLVE USES.  A soft-gate weight in [0,1] is the right shape for
        # ADMISSION but the wrong quantity for a factor: 132 pose factors of
        # weight ~0.5 against 57,000 whitened reprojection terms is not a loop
        # closure, it is a rounding error, and the measurement said so (the
        # bundle barely moved: 0.0675 / 0.0759 with welds handed to it).
        #
        # The information is derived, not chosen.  A relative pose from N
        # inlier correspondences at feature noise sigma_px and focal fx:
        #
        #     sigma_rot   = (sigma_px / fx) / sqrt(N)
        #     sigma_trans = (sigma_px / fx) * z_median / sqrt(N)
        #
        # and the soft-gate weight then scales that information rather than
        # replacing it - so a marginal weld still contributes WEAKLY (plan
        # 5.5) instead of flipping, and a strong one contributes like the
        # fifty measurements it actually is.
        Pc_sel = P[mask]
        z_med = float(np.median(np.linalg.norm(Pc_sel - T_wc[:3, 3], axis=1))) \
            if len(Pc_sel) else 1.0
        # ★ THE WELD'S UNCERTAINTY IS THE LANDMARKS', NOT THE PIXELS'.
        # A weld is a pose solved against landmarks that are THEMSELVES
        # uncertain, so its translation error is bounded by
        # `sigma_z(z) / sqrt(N)` - the physics bound of plan 8.4 - and not by
        # `(sigma_px/fx) * z / sqrt(N)`, which is what a pose against PERFECT
        # structure would give.  The pixel form overstated the information by
        # ~1000x and the measurement caught it in one run: MH_02 went
        # 0.0364 -> 0.1337 and V1_01 0.062 -> 0.119.
        sN = np.sqrt(max(inl, 1))
        sigma_t = max(self.rig.depth_sigma_m(max(z_med, 1e-3)), 1e-4) / sN
        sigma_r = sigma_t / max(z_med, 1e-3)
        wgt = float(d.weight * retrieval_weight)
        info = np.array([wgt / sigma_t ** 2] * 3 + [wgt / sigma_r ** 2] * 3)
        self.welds.append({"from": cand, "to": kf_id, "Z": Z, "w": wgt,
                           "info": info, "inliers": inl, "gap_m": gap,
                           "z_med": z_med})
        # ★ STRUCTURE WELDS.  A pose weld
        # ties two keyframes; the bundle then drags structure along by the
        # host delta.  These observations instead bind the NEW keyframe to
        # the OLD visit's landmarks DIRECTLY - the verified PnP inliers
        # become reprojection rows in the final global BA, so the two visits
        # share structure and the early segment is pinned by the late one.
        # Budget 60 obs/weld.  Only weld
        # inliers enter - an outlier here would be a wrong constraint at
        # full weight.  Runs on the finalise path (main thread), before the
        # global BA reads the record.
        if getattr(self.tracker, "structure_welds", False):
            if pooled is not None:
                sel = np.nonzero(mask)[0][:60]
                self.tracker.add_structure_obs(kf_id, pooled[0][sel], uv[sel])
            else:
                # xclean: only weld inliers whose rich keypoint is a tracker landmark of the candidate (see _rich)
                sel = np.nonzero(mask & (lid >= 0))[0][:60]
                self.tracker.add_structure_obs(kf_id, lid[sel], uv[sel])

    # ------------------------------------------------------------- PGO ----
    def _current_pose(self, kf_id):
        """The pose the ESTIMATOR currently believes, not the one recorded at
        enqueue time."""
        live = self.tracker.state.kf_pose.get(kf_id)
        if live is not None:
            return np.array(live)
        for rec in self.tracker.all_kf:
            if int(rec[0]) == int(kf_id):
                return np.array(rec[3])
        return self.kf[kf_id]["T_wc"]

    def _pgo(self):
        """★ THE POSE GRAPH RUNS ON THE CURRENT TRAJECTORY, NOT ON SNAPSHOTS.

        `self.kf[k]["T_wc"]` is the pose a keyframe had when it was ENQUEUED -
        before local bundle adjustment refined it.  Building the graph from
        those and then applying the correction to the refined trajectory
        computes a delta between two different trajectories, and the result is
        distortion rather than closure: the global BA that followed started at
        **21.8 px** of reprojection error and the sim(3) scale drifted to
        1.033.  The tracker's own records are the trajectory; the graph uses
        them.
        """
        poses = {}
        for rec in self.tracker.all_kf:
            poses[int(rec[0])] = np.array(rec[3])
        live = self.tracker.state.kf_pose
        for k in list(poses):
            if k in live:
                poses[k] = np.array(live[k])
        if len(poses) < 3:
            return
        # ---- 4-DoF GRAVITY-ALIGNED ARM (XTEN_PGO_4DOF=1), 2026-09-24.
        # Measured on rosariov2_01: the online trajectory tilts 3.2 deg out of GT's ground plane, the global BA
        # alone 1.8 deg, the pose graph alone 26.1 deg - the 6-DoF graph, whose only gauge is the first
        # keyframe, lets 2,800 welds rotate the map about horizontal axes. ORB-SLAM3 (0.06 deg) optimises
        # x, y, z, yaw only on an inertial map. Gravity comes from the mean specific force, the estimator the
        # inertial init already uses, and must pass the same |g| gate or the graph stays 6-DoF.
        g_axis = None
        if os.environ.get("XTEN_PGO_4DOF") == "1":
            g_axis = self._gravity_axis(poses)
        self.pgo_mode = "4dof" if g_axis is not None else "6dof"
        self.pgo_gravity = None if g_axis is None else [float(x) for x in g_axis]
        t0 = time.perf_counter()
        self.pgo_info = "derived" if os.environ.get("XTEN_PGO_INFO") == "derived" else "isotropic"
        # ---- T1-e "pgi split" SUB-FLAGS (staged 2026-09-29). Read ONLY under the derived arm: with
        # XTEN_PGO_INFO unset the isotropic path never sees them, and with both sub-flags unset the derived
        # path is byte-identical to before.
        #   XTEN_PGO_HUBER=dimk  Huber threshold transported to the D-dim whitened residual (huber_k_dim)
        #   XTEN_PGO_ODO=edge    odometry edge (a->b) information from keyframe b's OWN evidence at w = 1
        huber = odo = None
        if self.pgo_info == "derived":
            huber = "dimk" if os.environ.get("XTEN_PGO_HUBER") == "dimk" else None
            odo = "edge" if os.environ.get("XTEN_PGO_ODO") == "edge" else None
        # WIRED 2026-09-29 (integrator, T1-e x LOG): XTEN_PGO_ODO=edge reads the tracker's per-keyframe
        # evidence that LOG records as `Tracker.kf_evidence[int(kid)] = {"inliers": N, "z_med": z}`
        # (track/tracker.py _log_kf_evidence, called right after the keyframe is appended so that
        # _median_depth() sees the new keyframe; the bootstrap keyframe carries inliers None and z_med is
        # None when no landmark lies in front of the camera).  `poses` is keyed by the same int(kf_id), so
        # odometry_edge_info(b, ...) finds keyframe b's record directly.  The mapping is handed over ONLY
        # under the sub-flag: with XTEN_PGO_ODO unset pose_graph never receives it and no float can change
        # (tests/test_t1_pgisplit.py, tests/test_t1_identity_all.py hold this against the LIVE tree).
        # `getattr` keeps a tracker without LOG (duck-typed test trackers) on the fallback path, which
        # summary()["pgo_odo"]["fallback"] counts; on a LOG draw it reads 0 apart from edges into a
        # keyframe whose z_med is None (test_t1_identity_all checks this on the real Tracker).
        kf_evidence = None
        if odo == "edge":
            kf_evidence = getattr(self.tracker, "kf_evidence", None)
        pg_stats = {}
        out = pose_graph(poses, list(self.welds), g_axis=g_axis,
                         info_mode=("derived" if self.pgo_info == "derived" else None),
                         huber=huber, odo=odo, kf_evidence=kf_evidence,
                         depth_sigma=getattr(self.rig, "depth_sigma_m", None), stats=pg_stats)
        self.pgo_huber = pg_stats.get("huber")
        self.pgo_odo = pg_stats.get("odo")
        ms = (time.perf_counter() - t0) * 1e3
        self.pgo_ms.append(ms)
        self.n_pgo += 1
        for k, T in out.items():
            if k in self.kf:
                self.kf[k]["T_wc"] = T
        self._out.put((out, poses, ms, len(self.welds)))

    def _gravity_axis(self, poses):
        """Unit gravity direction in the world frame of `poses`, or None when there is no credible one."""
        tr = self.tracker
        imu = getattr(tr, "imu", None)
        if imu is None or not imu.available():
            return None
        from xclean.backend.preint import gravity_from_accel_mean, GRAVITY_MAG
        t_of = {int(r[0]): float(r[2]) for r in tr.all_kf}
        ids = [k for k in sorted(poses) if k in t_of]
        if len(ids) < 8:
            return None
        R_wb = np.stack([(np.asarray(poses[k]) @ imu.T_cb)[:3, :3] for k in ids])
        gm = gravity_from_accel_mean(imu, R_wb, [t_of[k] for k in ids])
        if gm is None:
            return None
        g_w, gmag = gm
        if not (0.85 * GRAVITY_MAG < gmag < 1.15 * GRAVITY_MAG):
            return None
        return np.asarray(g_w, float) / max(float(np.linalg.norm(g_w)), 1e-12)

    # -------------------------------------------------------- reporting ---
    def finalise(self, tracker, run_pgo=False):
        """Run the pose graph ONCE, over everything, and apply it.

        Called after the last frame.  This is where a loop closure becomes a
        trajectory correction; before this point it is only evidence.
        """
        self._drain()
        if not self.apply_online:
            # the deterministic pass: every keyframe considered in order,
            # against a map that has stopped moving.
            # ★ HEARTBEAT: an un-instrumented long stage is
            # indistinguishable from a hang — that silence cost a 3-hour
            # row on the R4 sweep (madmax_B0, killed blind while its
            # scaled budget ground legitimately).  Every 2000 considers
            # prints progress; the sweep operator can tell alive from
            # stuck without a debugger.
            n_ord = len(self.order)
            t_hb = time.perf_counter()
            if self.verbose and n_ord > 2000:
                print(" [loop] finalise: %d keyframes to consider" % n_ord,
                      flush=True)
            for j, kf_id in enumerate(self.order):
                if j < 1:
                    continue
                self._consider_upto(kf_id, j)
                if self.verbose and n_ord > 2000 and j % 2000 == 0:
                    print(" [loop] finalise %d/%d  welds %d  %.0f s"
                          % (j, n_ord, len(self.welds),
                             time.perf_counter() - t_hb), flush=True)
        if not self.welds:
            return None
        if not run_pgo:
            # ★ THE WELDS GO TO THE BUNDLE, NOT TO A SEPARATE POSE GRAPH.
            # A pose graph moves POSES and then the landmarks are dragged
            # along by their host's delta.  That is exact arithmetic, but it
            # leaves the structure fitting a trajectory it was never solved
            # against, and the global bundle adjustment then has to repair it:
            # measured pre-BA reprojection over three runs of the same build
            # was 2.5, 5.2 and 23.6 px, and the ATE followed (0.0405, 0.0420,
            # 0.1095).  The global BA already carries every weld as a
            # relative-pose factor, so it can close the loop and move the
            # structure in ONE consistent solve.  The pose graph stays behind
            # `run_pgo=True` for the case where the loop error is too large
            # for a local optimiser to start from.
            return Decision("loop_finalise", True,
                            "%d welds handed to the global bundle adjustment "
                            "(no separate pose graph: it moves poses without "
                            "the structure and the bundle has to undo it)"
                            % len(self.welds),
                            {"welds": len(self.welds), "pgo": 0})
        self._pgo()
        got = self.poll()
        if got is None:
            return None
        return self.apply(got, tracker)

    def _drain(self, timeout_s=30.0):
        t0 = time.perf_counter()
        while not self._in.empty() and time.perf_counter() - t0 < timeout_s:
            time.sleep(0.05)

    def summary(self):
        s = {"considered": self.n_considered, "welds": len(self.welds),
                "prox_verified": self.n_prox_verified,
                "pgo_runs": self.n_pgo,
                "pgo_mode": self.pgo_mode,
                "pgo_info": self.pgo_info,
                "pgo_huber": self.pgo_huber,
                "pgo_odo": self.pgo_odo,
                "retrieval_fisher": self.retrieval_fisher,
                # FIX A provenance (staged 2026-10-01; see LOOP_FISHER_ENV)
                "fisher_mode": self.loop_fisher,
                "queries_total": self.n_fisher_queries,
                "queries_infeasible": (None if self.retrieval_fisher
                                       else self.n_fisher_infeasible),
                "fallback_accepts": self.n_fisher_accepts,
                "loop_covis": self.loop_covis,
                "rich_kf": self.n_rich_kf, "rich_3d_per_kf": (self.n_rich_3d / max(self.n_rich_kf, 1)),
                "covis_tried": self.n_covis_tried,
                "covis_pooled": self.n_covis_pooled,
                "covis_rescued": self.n_covis_rescued,
                "covis_nbrs_median": (float(np.median(self.covis_nbrs)) if self.covis_nbrs else None),
                "pgo_gravity": self.pgo_gravity,
                "pgo_ms_median": float(np.median(self.pgo_ms))
                if self.pgo_ms else None,
                "pgo_ms_max": float(np.max(self.pgo_ms))
                if self.pgo_ms else None,
                "worker_ms_p95": float(np.percentile(self.worker_ms, 95))
                if self.worker_ms else None,
                "weld_inliers_median": float(np.median(
                    [w["inliers"] for w in self.welds])) if self.welds else None,
                "weld_weight_median": float(np.median(
                    [w["w"] for w in self.welds])) if self.welds else None,
                "gates": [g.summary() for g in self.gates]}
        if self.prox_path != "off":
            # PROX SPIKE GATE provenance, present ONLY when the flag is on (the default summary is unchanged)
            s["prox_path"] = self.prox_path
            s["spike_refused_prox"] = self.n_spike_prox
            s["spike_refused_samepath"] = self.n_spike_samepath
        return s

    def dial_lines(self):
        return [g.summary_line() for g in self.gates]


# --------------------------------------------------------------------------
# ---- T1-e "pgi split" (staged 2026-09-29): the two heuristics inside the derived-information arm, each
# behind its own sub-flag and each a FUNCTION, so the unit test can hold it against the closer's own formula.
def huber_k_dim(D, k1=HUBER_K):
    """The 1-D Huber threshold transported to a D-dimensional WHITENED residual (XTEN_PGO_HUBER=dimk).

    PRE-REGISTERED CHOICE: the TAIL-MATCHED mapping, not the efficiency-matched one.  HUBER_K = 1.345 is
    defined by 95 % asymptotic efficiency at the 1-D Gaussian (core/derive.py).  Applying that same k to the
    NORM of a D-vector of unit-variance components is a different rule: |x| ~ chi_D, whose median alone is
    2.31 at D = 6, so nearly every consistent 6-DoF edge is down-weighted.  Tail matching keeps the
    PROBABILITY that a consistent residual is touched by the kernel what it is in 1-D:

        P(|x| > k_D) = P(|x_1| > k_1) = 2 Phi(-k_1)      =>     k_D = sqrt(chi2.isf(2 Phi(-k_1), D))

    i.e. 2.9852 for D = 6 and 2.5077 for D = 4 (scipy.stats, no hard-coded constant; the identity at D = 1).
    The efficiency-matched alternative (k_D chosen so the Huber M-estimator keeps 95 % efficiency on a
    D-dim Gaussian) was NOT chosen: it needs a numerical integral over the chi_D density and answers a
    different question (variance of the location estimate) from the one the graph asks (which edges are
    outliers).

    OPEN INCONSISTENCY, stated and not addressed here: the global BA's reprojection Huber
    (backend/global_ba.py ~562-563) applies the 1-D k = 1.345 to a 2-D pixel norm, which touches ~40 % of
    consistent observations (P(chi_2 > 1.345^2) = 0.40).  Changing that is a separate decision.
    """
    from scipy.stats import chi2, norm
    p_tail = 2.0 * float(norm.cdf(-float(k1)))
    return float(np.sqrt(chi2.isf(p_tail, int(D))))


def weld_information(depth_sigma_m, inliers, z_med, wgt=1.0):
    """The closer's own weld information (LoopCloser._verify) as a function of its evidence:

        sigma_t = max(depth_sigma_m(max(z_med, 1e-3)), 1e-4) / sqrt(max(N, 1)),   sigma_r = sigma_t / z_med,
        info    = [wgt / sigma_t^2] * 3 + [wgt / sigma_r^2] * 3

    Same operations in the same order as _verify (the test holds the two sources against each other), so the
    XTEN_PGO_ODO=edge odometry information is exactly a weld's at w = 1 from (N, z_med)."""
    sN = np.sqrt(max(inliers, 1))
    sigma_t = max(depth_sigma_m(max(z_med, 1e-3)), 1e-4) / sN
    sigma_r = sigma_t / max(z_med, 1e-3)
    return np.array([wgt / sigma_t ** 2] * 3 + [wgt / sigma_r ** 2] * 3)


def _evidence_of(rec):
    """(inliers, z_med) from one kf_evidence record, or None.  Accepts a (N, z) pair or a dict with 'inliers'
    and 'z_med' (LOG decides the stored shape; both are read).  Non-finite values count as missing."""
    if rec is None:
        return None
    try:
        if isinstance(rec, dict):
            n, z = rec["inliers"], rec["z_med"]
        else:
            n, z = rec[0], rec[1]
        n, z = float(n), float(z)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if not (np.isfinite(n) and np.isfinite(z)):
        return None
    return n, z


def odometry_edge_info(kid_to, odo_L, kf_evidence, depth_sigma):
    """XTEN_PGO_ODO=edge: the information of the odometry edge (a -> b) from keyframe b's OWN evidence with
    the weld formula at w = 1.  Returns (info 6-vector, used_evidence).  Missing evidence (no mapping, no
    record for b, no depth model) falls back to the MEDIAN weld information `odo_L` with used_evidence=False,
    so the caller can count it."""
    ev = None
    if kf_evidence is not None and depth_sigma is not None:
        try:
            rec = kf_evidence.get(kid_to) if hasattr(kf_evidence, "get") else kf_evidence[kid_to]
        except (KeyError, IndexError, TypeError):
            rec = None
        ev = _evidence_of(rec)
    if ev is None:
        return odo_L, False
    return weld_information(depth_sigma, ev[0], ev[1], 1.0), True


def pose_graph(poses, welds, iters=15, odom_info=1.0, g_axis=None, info_mode=None,
               huber=None, odo=None, kf_evidence=None, depth_sigma=None, stats=None):
    """Gauss-Newton pose-graph optimisation on SE(3), sparse.

    T1-e sub-flags (staged 2026-09-29), effective ONLY with info_mode="derived" and at least one valid weld
    (the isotropic path never reads them):
      huber="dimk"  Huber threshold k_D = huber_k_dim(D) on the whitened norm, D = 6 (or 4 in the 4-DoF arm,
                    whose projected residual has 3 translation + 1 yaw components);
      odo="edge"    odometry edge (a->b) information = weld_information(depth_sigma, N_b, z_med_b, 1) from
                    kf_evidence[b]; an edge without evidence falls back to the median weld information and
                    is counted.
    `stats` (a dict) receives the provenance: {"info", "huber": {mode,k,D}, "odo": {mode,edges,evidence,
    fallback}, "welds"}.

    Edges are the odometry chain (measured from the current estimate) and the
    verified welds, each carrying **the weight its evidence earned** rather
    than a boolean.  A marginal weld pulls weakly; it does not flip the graph.

    The Jacobian uses the standard `J_i = -Ad(T_j^-1 T_i)`, `J_j = I`
    simplification: the exact form carries an SE(3) right-Jacobian inverse
    whose effect is second order in the residual, and a pose graph starts from
    a good initial guess by construction.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    ids = sorted(poses)
    index = {k: i for i, k in enumerate(ids)}
    n = len(ids)
    T = [np.array(poses[k]) for k in ids]

    # ---- DERIVED INFORMATION ARM (XTEN_PGO_INFO=derived), 2026-09-25. The default graph weights every 6-DoF
    # residual with ONE scalar, so a metre of translation costs the same as a radian of rotation, and the
    # anisotropic per-weld information the closer derives from the depth model (w/sigma_t^2, w/sigma_r^2 with
    # sigma_t = depth_sigma(z_med)/sqrt(N), sigma_r = sigma_t/z_med) was computed and never used. Here: every weld
    # uses its own information; every odometry edge uses the MEDIAN weld information (the same physics at the
    # run's typical depth and inlier count - a run-time statistic, the same rule on every rig); Huber acts on the
    # WHITENED norm. Diagnosed offline on rosariov2_01 and MadMax. Default path unchanged.
    derived = info_mode == "derived"
    odo_L = None
    if derived:
        wl = [np.asarray(w["info"], float) for w in welds
              if w["from"] in index and w["to"] in index and np.ndim(w.get("info", 0)) == 1]
        odo_L = np.median(np.stack(wl), axis=0) if wl else None
        derived = odo_L is not None
    # ---- T1-e sub-flags: gated on `derived`, so the isotropic path never sees them and with both unset
    # k_rob IS HUBER_K and every odometry edge IS odo_L (byte-identical to the pre-T1-e derived path).
    dimk = bool(derived and huber == "dimk")
    edge = bool(derived and odo == "edge")
    four = g_axis is not None
    D_rob = 4 if four else 6
    k_rob = huber_k_dim(D_rob) if dimk else HUBER_K
    n_ev = n_fb = 0
    edges = []
    for a, b in zip(ids[:-1], ids[1:]):
        Z = se3_inv(poses[a]) @ poses[b]
        L = odo_L if derived else odom_info
        if edge:
            L, used = odometry_edge_info(b, odo_L, kf_evidence, depth_sigma)
            n_ev += int(used)
            n_fb += int(not used)
        edges.append((index[a], index[b], Z, L))
    for w in welds:
        if w["from"] in index and w["to"] in index:
            edges.append((index[w["from"]], index[w["to"]], w["Z"],
                          np.asarray(w["info"], float) if derived else
                          float(w["w"])))
    if stats is not None:
        stats["info"] = "derived" if derived else "isotropic"
        stats["huber"] = {"mode": "dimk" if dimk else "1d", "k": float(k_rob), "D": int(D_rob)}
        stats["odo"] = {"mode": "edge" if edge else ("median" if derived else "scalar"),
                        "edges": max(n - 1, 0), "evidence": n_ev, "fallback": n_fb}
        stats["welds"] = len(edges) - max(n - 1, 0)
    if len(edges) <= n - 1:
        return {k: poses[k] for k in ids}

    eye6 = np.eye(6)
    # 4-DoF mode: node k may move only in translation and in yaw about world gravity. With the right
    # perturbation T_k exp([rho, phi]) a world-gravity rotation is phi along a_k = R_k^T g, and
    # R_k exp([a_k] t) = exp([g] t) R_k, so roll and pitch relative to gravity are preserved EXACTLY.
    # Residuals keep translation plus the yaw component about the downstream node's gravity axis, as in
    # VINS-Mono / ORB-SLAM3's OptimizeEssentialGraph4DoF: a weld's roll/pitch disagreement cannot tilt the
    # map because it is not in the cost.
    if four:                                   # (`four` is set above, with the sub-flag gating)
        g_axis = np.asarray(g_axis, float) / max(float(np.linalg.norm(g_axis)), 1e-12)
    for _ in range(iters):
        if four:
            ax = [T[k][:3, :3].T @ g_axis for k in range(n)]
            Bk = []
            for k in range(n):
                B = np.zeros((6, 4))
                B[0, 0] = B[1, 1] = B[2, 2] = 1.0
                B[3:, 3] = ax[k]
                Bk.append(B)
        D = 4 if four else 6
        rows, cols, vals = [], [], []
        b = np.zeros(D * n)
        for (i, j, Z, wgt) in edges:
            e = se3_log(se3_inv(Z) @ se3_inv(T[i]) @ T[j])
            if four:
                Pj = np.eye(6)
                Pj[3:, 3:] = np.outer(ax[j], ax[j])
                e = Pj @ e
            # ★ ROBUST EDGES.  A pose graph without a robust kernel lets ONE
            # bad weld decide the trajectory - the exact cliff plan 5.5 bans
            # for admission, reappearing inside the solver.  Measured: V1_01's
            # three draws of the same configuration spread 2.55x
            # (0.0406 / 0.0948 / 0.1037) because the graph sometimes converged
            # around a bad edge and sometimes did not.  Huber down-weights the
            # edge instead of letting it win, and it uses the SAME k = 1.345
            # that every other robust cost in this system uses.
            if derived:
                en = float(np.sqrt(np.sum(wgt * e * e)))      # whitened norm
                # k_rob = HUBER_K (1-D, the pre-T1-e rule) or huber_k_dim(D) under XTEN_PGO_HUBER=dimk
                rob = 1.0 if en <= k_rob else (k_rob / max(en, 1e-9))
                wgt = wgt * rob                                   # 6-vector information
            else:
                en = float(np.linalg.norm(e))
                rob = 1.0 if en <= HUBER_K else (HUBER_K / max(en, 1e-9))
                wgt = wgt * rob
            Ji = -adjoint(se3_inv(T[j]) @ T[i])
            Jj = eye6
            if four:
                Ji = Pj @ Ji @ Bk[i]
                Jj = Pj @ Bk[j]
            for (bi, J1) in ((i, Ji), (j, Jj)):
                if derived:
                    b[bi * D:bi * D + D] -= J1.T @ (wgt * e)
                else:
                    b[bi * D:bi * D + D] -= wgt * (J1.T @ e)
                for (bj, J2) in ((i, Ji), (j, Jj)):
                    H = (J1.T @ (wgt[:, None] * J2)) if derived else wgt * (J1.T @ J2)
                    ii = np.repeat(np.arange(D), D) + bi * D
                    jj = np.tile(np.arange(D), D) + bj * D
                    rows.append(ii)
                    cols.append(jj)
                    vals.append(H.reshape(-1))
        H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                  np.concatenate(cols))),
                          shape=(D * n, D * n)).tocsr()
        H = H + sp.eye(D * n, format="csr") * 1e-6
        H = H.tolil()
        for r in range(D):                 # gauge: hold the first keyframe
            H[r, :] = 0
            H[:, r] = 0
            H[r, r] = 1.0
        H = H.tocsr()
        b[:D] = 0.0
        try:
            dx = spl.spsolve(H, b)
        except Exception:
            break
        if not np.isfinite(dx).all():
            break
        for k in range(n):
            step = dx[k * D:(k + 1) * D]
            if four:
                step = Bk[k] @ step
            T[k] = T[k] @ se3_exp(step)
            T[k][:3, :3] = so3_project(T[k][:3, :3])
        if np.linalg.norm(dx) < 1e-10:
            break
    return {k: T[index[k]] for k in ids}
