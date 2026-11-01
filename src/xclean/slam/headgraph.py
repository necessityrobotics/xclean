# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.3 GLOBAL CORRECTION: the ANCHORED HEAD FILTER, with PROVISIONAL weak-prior episodes (fix round
2026-10-26, verifier major 1 = judge major 1).

Every LIVE-MAP pose already published is a fixed anchor; the only free variable is the head keyframe's
map pose H (6-DoF, right [rho, phi]).  Per correction event (head keyframe j = the query keyframe of the deciding job):
  prior          H_0 = T_target_prev LIVE-ODOM(j); covariance = the previous posterior carried to j (adjoint)
                 + kappa Sigma_chain(q_last -> j); before the first event kappa Sigma_chain(0 -> j); across an unknown
                 link: no information (1e6 I, numerical);
  loop i         H_i = A(c_i) Z_i LIVE-ODOM(q_i)^-1 LIVE-ODOM(j), A(c) = the LIVE-MAP pose PUBLISHED at c's frame;
                 covariance Ad Sigma_loop Ad^T + kappa Sigma_chain(q_i -> j);
  robust kernel  Huber on the loop terms at the tail-matched k for 6 dimensions (closer.huber_k_dim's rule);
  solve          Gauss-Newton, <= 5 iterations (class D, DESIGN_P1 LM_ITERS);
  output         T_target = H LIVE-ODOM(j)^-1 (head-anchored T_map_odom) and the posterior carried to the next event.
O(1) per event, ~1 ms.  Own code; the arithmetic is the design prototype's (p13_core.fuse).

WHY EPISODES (verifier major 1): Huber bounds a loop only RELATIVE to other information.  Where the prior is weaker than
the new episode (the run's first event, the first after a long traverse) one consistent wrong episode is applied almost in
full, and its carried (over-confident, R11) posterior then LOCKS it in against later true loops, which Huber now treats
as outliers (build 1: mh_04 0.662 -> 2.547; madmax_E1 5.8 m for 186 s).  Admission is the only protection of such an
event, so the event is made PROVISIONAL, as R1 does for merges.  The arithmetic of every applied event is the design's
(sequential, posterior carried event to event); what changes is that a weak-prior episode keeps a CHECKPOINT and can be
taken back:
  weak prior     lambda_max(S0 I_L) > 1, I_L = the new loops' information at the head: in some direction the prior is
                 less informative than the episode (a wrong episode would move the head by more than half its error);
                 class-D break-even, the mirror of strict informativeness.  No episode open + a weak
                 prior -> the state before the event is CHECKPOINTED and the event applied (published: L3 latency kept);
                 its loops are the provisional set P
  join           while P is open, an admitted loop PCM-consistent (chi2_6(0.999), merge.pcm_pair, kappa-scaled) with >= 1
                 member of P is applied as in the design and joins P
  rival          a loop consistent with no member of P is NOT applied: it is held as a RIVAL; the largest pairwise-
                 consistent rival set R (merge.r1_set's greedy rule) replaces P when |R| > |P| (R1's no-rival rule):
                 the state returns to the checkpoint and R is fused onto it (SWAP: the wrong target is reverted, the
                 rival's applied; nothing of P is carried)
  confirm        P is committed (the checkpoint dropped) when its members' queries span more than N_KF keyframes: two
                 non-overlapping windows agree (R1's confirm: a consistent loop more than N_KF after the set)
  merge          an R1 merge event (its own provisional / confirm / revert) closes an open loop episode as it stands
Unplanted, with no rival, the published targets are the design's to the bit (only the bookkeeping differs).
`episodes` = "provisional" (DEFAULT) | "carry" (the design / build 1, kept for the replay-equivalence test).
`prior_mode` arms ("posterior" = the design, DEFAULT; "chain"; "lam") unchanged.
"""
from __future__ import annotations

import time

import numpy as np

from xclean.slam.geom import inv, ortho, Ad, logr, expr, to_right, huber_k_dim
from xclean.slam.admit import Kappa, N_KF, PCM_WIN

ITERS = 5
NO_INFO = 1e6
_EPS = 1e-15 * np.eye(6)


def fuse(H0, S0, His, Sis, kD, iters=ITERS):
    """Gauss-Newton on the head's map pose; Huber on the loop terms.  -> (H, posterior, prior whitened residual^2)."""
    H = H0.copy()
    W = [np.linalg.inv(S0 + 1e-15 * np.eye(6))] + [np.linalg.inv(S + 1e-15 * np.eye(6)) for S in Sis]
    T = [H0] + list(His)
    A = None
    for _ in range(iters):
        A = np.zeros((6, 6))
        b = np.zeros(6)
        for n, (Ti, Wi) in enumerate(zip(T, W)):
            r = logr(inv(H) @ Ti)
            w = 1.0
            if n > 0:
                m = float(np.sqrt(max(r @ Wi @ r, 0.0)))
                w = 1.0 if m <= kD else kD / m
            A += w * Wi
            b += w * Wi @ r
        dx = np.linalg.solve(A, b)
        H = ortho(H @ expr(dx))
        if np.abs(dx).max() < 1e-9:
            break
    r0 = logr(inv(H) @ H0)
    m0 = float(r0 @ W[0] @ r0)
    Sp = np.linalg.inv(A)
    return H, 0.5 * (Sp + Sp.T), m0


def weak_prior(S0, Sis):
    """lambda_max(S0 I_L) with I_L = the sum of the loops' information: > 1 = the prior is weaker than the episode in some
    direction.  -> (weak, lambda_max)."""
    I = np.zeros((6, 6))
    for S in Sis:
        I += np.linalg.inv(S + _EPS)
    try:
        L = np.linalg.cholesky(0.5 * (I + I.T) + _EPS)
        lam = float(np.linalg.eigvalsh(L.T @ S0 @ L).max())
    except np.linalg.LinAlgError:
        return True, float("inf")
    return lam > 1.0, lam


class HeadFilter:
    """The worker-side correction state.  T_target is the head-anchored T_map_odom the frame process slews toward."""

    def __init__(self, prior_mode="posterior", episodes="provisional"):
        self.kD = huber_k_dim(6)
        self.T_target = np.eye(4)
        self.Spost = None
        self.q_last = 0
        self.epoch = 0
        if prior_mode not in ("posterior", "chain", "lam"):
            raise ValueError("HeadFilter prior_mode must be posterior | chain | lam, got %r" % (prior_mode,))
        if episodes not in ("provisional", "carry"):
            raise ValueError("HeadFilter episodes must be provisional | carry, got %r" % (episodes,))
        self.prior_mode = prior_mode
        self.episodes = episodes
        self.prov = None
        self.walls = []
        self.m0_hist = []
        self.ep_log = []
        self.n_prov = self.n_confirm = self.n_swap = self.n_drop = self.n_rival_held = 0

    # ---- state ---------------------------------------------------------------------------------------------------------
    def _state(self):
        return (self.T_target.copy(), None if self.Spost is None else self.Spost.copy(), self.q_last)

    def _set_state(self, st):
        T, S, q = st
        self.T_target, self.Spost, self.q_last = T.copy(), (None if S is None else S.copy()), q

    def snapshot(self):
        P = None
        if self.prov is not None:
            P = dict(self.prov)
            P["loops"] = list(self.prov["loops"])
            P["rival"] = list(self.prov["rival"])
        return (self.T_target.copy(), None if self.Spost is None else self.Spost.copy(), self.q_last, self.epoch, P)

    def restore(self, snap):
        T, S, q, _e = snap[:4]
        self.T_target, self.Spost, self.q_last = T.copy(), (None if S is None else S.copy()), q
        self.prov = snap[4] if len(snap) > 4 else None
        self.epoch += 1

    # ---- pieces ------------------------------------------------------------------------------------------------------
    def _terms(self, j, Tj, loops, chain, T_odom, anchors):
        His, Sis, used = [], [], []
        for L in loops:
            qq, c = L["q"], L["c"]
            A = anchors(c)
            if A is None:
                continue
            B = inv(T_odom(qq)) @ Tj
            His.append(A @ L["Z"] @ B)
            Ab = Ad(inv(B))
            if qq < j:
                g = chain.rel(qq, j)
                Sc = Kappa.scale(g[2], L["kv"]) if g is not None else np.zeros((6, 6))
            else:
                Sc = np.zeros((6, 6))
            Sis.append(Ab @ to_right(L["Z"][:3, :3], L["Sl"]) @ Ab.T + to_right(B[:3, :3], Sc))
            used.append(L)
        return His, Sis, used

    def _prior(self, j, Tj, chain, T_odom, kv, no_prior):
        """The prior at head j from the CURRENT state (the design's): -> (H0, S0)."""
        H0 = self.T_target @ Tj
        q_last = self.q_last
        g = chain.rel(q_last, j) if q_last < j else None
        Bl = inv(T_odom(q_last)) @ Tj
        if no_prior:
            S0 = NO_INFO * np.eye(6)
        elif g is not None:
            S0 = to_right(Bl[:3, :3], Kappa.scale(g[2], kv))
        elif q_last < j:
            S0 = NO_INFO * np.eye(6)          # an unknown link between the last event and the head: no information
        else:
            S0 = np.zeros((6, 6))
        if self.Spost is not None and not no_prior and self.prior_mode in ("posterior", "lam"):
            Ab = Ad(inv(Bl))
            lam = 1.0
            if self.prior_mode == "lam" and len(self.m0_hist) >= 10:
                # ARM (not the design): the carried posterior inflated by the run's own causal calibration of the
                # event prior residual (median of the last 30 m0 / median chi2_6, >= 1; class C like kappa)
                lam = max(1.0, float(np.median(self.m0_hist[-30:])) / 5.348120627447122)
            S0 = S0 + lam * (Ab @ self.Spost @ Ab.T)
        return H0, S0

    def _apply(self, j, Tj, H0, S0, His, Sis):
        """The design's event arithmetic on the current state: fuse, carry the posterior."""
        H, Sp, m0 = fuse(H0, S0 + 1e-15 * np.eye(6), His, Sis, self.kD)
        self.m0_hist.append(m0)
        self.Spost = Sp
        self.q_last = j
        self.T_target = H @ inv(Tj)
        return m0

    @staticmethod
    def _best_set(pool, L, chain, kv):
        """The largest pairwise-consistent set containing L, greedy by inliers (merge.r1_set's rule, no place test)."""
        from xclean.slam.merge import pcm_pair
        nb = [x for x in pool if x is not L and pcm_pair(chain, x, L, kv)]
        C = [L]
        for x in sorted(nb, key=lambda y: (-int(y.get("inliers", 0)), -int(y["q"]))):
            if all(pcm_pair(chain, x, y, kv) for y in C[1:]):
                C.append(x)
        return C

    # ---- one event ---------------------------------------------------------------------------------------------------
    def event(self, j, T_odom_j, loops, chain, T_odom, anchors, kv, no_prior=False, t=None):
        """Apply one correction event at head keyframe j.
        loops: dicts with q, c, Z (4x4), Sl (6x6 [phi, t]), kv (and inliers); T_odom(k) -> LIVE-ODOM of keyframe k;
        anchors(c) -> the LIVE-MAP pose published at keyframe c's frame (None: not published yet -> loop skipped);
        no_prior: a MERGE across an unknown link (the prior carries no information).
        -> dict (T_new, used loops, m0, wall_s, epoch, ep, provisional) or None when nothing new is applied."""
        t0 = time.perf_counter()
        Tj = T_odom_j
        His, Sis, used = self._terms(j, Tj, loops, chain, T_odom, anchors)
        if not His:
            return None
        if self.episodes == "carry":
            H0, S0 = self._prior(j, Tj, chain, T_odom, kv, no_prior)
            m0 = self._apply(j, Tj, H0, S0, His, Sis)
            return self._done(t0, used, m0, "carry", None)
        from xclean.slam.merge import pcm_pair
        rec = {"j": j, "t": t, "n_new": len(used)}
        lam_w = None
        if no_prior:                                     # an R1 merge: its own provisional / confirm / revert
            if self.prov is not None:
                self.n_drop += 1
                rec["closed_prov"] = len(self.prov["loops"])
                self.prov = None
            H0, S0 = self._prior(j, Tj, chain, T_odom, kv, True)
            m0 = self._apply(j, Tj, H0, S0, His, Sis)
            ep, fused = "merge", used
        elif self.prov is None:
            H0, S0 = self._prior(j, Tj, chain, T_odom, kv, False)
            weak, lam_w = weak_prior(S0, Sis)
            ep = "commit"
            if weak:
                self.prov = {"loops": list(used), "rival": [], "ck": self._state(), "j0": j, "t0": t, "id": self.n_prov}
                self.n_prov += 1
                ep = "open"
            m0 = self._apply(j, Tj, H0, S0, His, Sis)
            fused = used
        else:
            P = self.prov
            joined, riv = [], []
            for L in used:
                ok = any(pcm_pair(chain, M, L, kv) for M in reversed(P["loops"][-PCM_WIN:]))
                (joined if ok else riv).append(L)
            P["rival"].extend(riv)
            P["rival"] = P["rival"][-PCM_WIN:]
            self.n_rival_held += len(riv)
            rec.update(joined=len(joined), rival=len(riv))
            R = self._best_set(P["rival"], riv[-1], chain, kv) if riv else []
            rec["rival_set"] = len(R)
            if len(R) > len(P["loops"]) + len(joined):
                # SWAP: back to the checkpoint, the rival set fused onto it; P's loops become the rival pool
                rest = [x for x in P["rival"] if not any(x is y for y in R)]
                old = P["loops"] + joined
                self._set_state(P["ck"])
                Hr, Sr, fused = self._terms(j, Tj, R, chain, T_odom, anchors)
                H0, S0 = self._prior(j, Tj, chain, T_odom, kv, False)
                m0 = self._apply(j, Tj, H0, S0, Hr, Sr)
                P["loops"] = list(R)
                P["rival"] = (rest + old)[-PCM_WIN:]
                self.n_swap += 1
                ep = "swap"
            elif joined:
                Hj, Sj = [His[used.index(L)] for L in joined], [Sis[used.index(L)] for L in joined]
                H0, S0 = self._prior(j, Tj, chain, T_odom, kv, False)
                m0 = self._apply(j, Tj, H0, S0, Hj, Sj)
                P["loops"].extend(joined)
                ep, fused = "join", joined
            else:                                        # rivals only, smaller than P: held, nothing applied
                rec.update(ep="rival_held", n_prov=len(P["loops"]))
                self.ep_log.append(rec)
                return None
            P["loops"] = P["loops"][-PCM_WIN:]
        if self.prov is not None:
            qs = [int(L["q"]) for L in self.prov["loops"]]
            rec.update(n_prov=len(qs), q_span=max(qs) - min(qs), prov_id=self.prov["id"])
            if max(qs) - min(qs) > N_KF:                 # two non-overlapping windows agree: commit
                self.n_confirm += 1
                rec["rival_dropped"] = len(self.prov["rival"])
                self.prov = None
                ep = ep + "+confirm"
        rec.update(ep=ep, lam_w=lam_w, m0=m0)
        self.ep_log.append(rec)
        return self._done(t0, fused, m0, ep, rec)

    def _done(self, t0, used, m0, ep, rec):
        wall = time.perf_counter() - t0
        self.walls.append(wall)
        self.epoch += 1
        return {"T_new": self.T_target.copy(), "used": used, "m0": m0, "wall_s": wall, "epoch": self.epoch, "ep": ep,
                "provisional": self.prov is not None}


# =====================================================================================================================
# F2 "MV-HEAD": the MINIMUM-VARIANCE (GLS)
# head correction under an explicit first-order error model.  Used only with worker cfg head = "mv"; HeadFilter above is
# p13b_fix1's, byte for byte.
#
# Error space ("W"): first-order LEFT twists of the LIVE-ODOM world, ordered [rho, phi] (geom.Ad's order).  A FORM is a
# linear combination of mutually independent SOURCES, one 6x6 coefficient block per source slot (Ledger).  Per event at
# head j (loops i = (q_i, c_i, Z_i, Sigma_loop,i), published anchors A(c_i), head-state gauge origin o):
#   Psi_c  = D(max(c, o) -> j) + sum_n (lambda^head_n - beta^(c)_n) Delta_n        the head prior's error RELATIVE TO
#            (in-window epochs; the out-of-window part is anchor-private)           anchor c (3.4)
#   nu_i   = e_Z,i + D(q_i -> j) + pi_i                                             the loop's own error
#   pi_i   = D(c_i -> o) (anchor-private provenance drift, c_i < o; REFUSED while a class in it is uncalibrated)
#            + the out-of-window lineage completion
#   rows   prior Psibar = sum_i omega_i Psi_ci (omega_i = (sum I)^-1 I_i, I_i = Var(nu_i)^-1); loop i: Psibar - Psi_ci + nu_i
#   y      y_0 = 0, y_i = log(H_0^-1 H_i), H_i = A(c_i) Z_i LIVE-ODOM(q_i)^-1 LIVE-ODOM(j)   (head right [rho, phi])
#   solve  GLS x = (1^T C^-1 1)^-1 1^T C^-1 y, Huber IRLS on the loops' marginal innovations (variance inflation),
#          Gauss-Newton on SE(3) around H_0 (<= 5 iterations); the increment's random part
#          Delta_new = sum_i W_i (nu_i - Psi_ci) is a new epoch of the ledger.
# Drift intervals carry P_kappa (admit.Chain per-class block prefix sums) at the event's kappa_F (one per class and block,
# 3.3); links are never scaled; an interval holding an unknown link carries NO_INFO per link (its offset is cancelled
# exactly by a merge epoch's lineage).
# =====================================================================================================================
import bisect                                                          # noqa: E402

from xclean.slam.admit import CLS_VI, CLS_VO, SUFF                    # noqa: E402

_K_IV, _K_Z, _K_PI, _K_FOLD = 0, 1, 2, 3
EMPTY_LIN = (-1, frozenset())
_I6 = np.eye(6)


def _Ad_many(T):
    """geom.Ad over a stack of poses (n, 4, 4) -> (n, 6, 6) ([rho, phi] order)."""
    R, t = T[:, :3, :3], T[:, :3, 3]
    K = np.zeros((len(T), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -t[:, 2], t[:, 1]
    K[:, 1, 0], K[:, 1, 2] = t[:, 2], -t[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -t[:, 1], t[:, 0]
    A = np.zeros((len(T), 6, 6))
    A[:, :3, :3] = R
    A[:, :3, 3:] = K @ R
    A[:, 3:, 3:] = R
    return A


def _Kr_many(R):
    """geom.to_right's K over a stack of relative rotations: [phi, t] -> [rho, phi] (rho = R^T t)."""
    K = np.zeros((len(R), 6, 6))
    K[:, :3, 3:] = np.transpose(R, (0, 2, 1))
    K[:, 3:, :3] = np.eye(3)
    return K


def _sym(S):
    return 0.5 * (S + np.swapaxes(S, -1, -2))


class Ledger:
    """F2_DESIGN 3.5: the bounded lineage ledger (exact inside the window).

    Source slots (mutually independent): IV elementary drift interval (a, b] of the chain axis, covariance re-evaluated
    per event (P_kappa at the event's kappa); Z one fused loop's noise; PI one fused loop's private term; FOLD one folded
    source per kept form.  Interval slots partition the referenced part of the axis (breakpoints = their endpoints);
    splitting an interval copies its coefficients to both parts (exact: drift is additive).
    Kept: the random parts Delta_n of the newest `win` epochs (orphans included), as dense rows E[row, slot, 6, 6];
    Var(Delta_n) of EVERY epoch (6x6, forever; prefix sums for the out-of-window completion).
    Window (win > 0): when an epoch leaves, the horizon k_H = the head keyframe of the oldest kept epoch; every kept form's
    coefficients on sources older than k_H (intervals ending at or before k_H, loop / private sources of epochs that left,
    and every fold source) are FOLDED into that form's new fold source (coef Cov coef^T at fold time; cross-form
    correlations through them dropped: the declared approximation, gate A4).  win = 0: unbounded (A4 reference only)."""

    def __init__(self, win=PCM_WIN, cap=128):
        self.win = int(win)
        self.cap = 0
        self.kind = np.zeros(0, np.int8)
        self.alive = np.zeros(0, bool)
        self.ia = np.zeros(0, np.int64)
        self.ib = np.zeros(0, np.int64)
        self.born = np.zeros(0, np.int64)
        self.cov = np.zeros((0, 6, 6))
        self.free = []
        self.iv_of = {}
        self.bpref = {}
        self.bps = []
        self.E = np.zeros((0, 0, 6, 6))
        self.rows_free = []
        self.row_of = {}
        self.kept = []
        self.ep_j = {}
        self.ep_nl = {}
        self.var = []
        self.cumV = [np.zeros((6, 6))]
        self.horizon = 0
        self.n_fold = self.n_split = self.n_drop = self.n_freed = 0
        self.fresh = []
        self._grow_slots(cap)
        self._grow_rows(max(8, (self.win + 2) if self.win > 0 else 64))

    # ---- storage ------------------------------------------------------------------------------------------------
    def _grow_slots(self, need):
        if need <= self.cap:
            return
        m = max(need, (3 * self.cap) // 2, 64)
        for name, shp, dt in (("kind", (), np.int8), ("alive", (), bool), ("ia", (), np.int64), ("ib", (), np.int64),
                              ("born", (), np.int64), ("cov", (6, 6), float)):
            a = getattr(self, name)
            b = np.zeros((m,) + shp, dt)
            b[:len(a)] = a
            setattr(self, name, b)
        E = np.zeros((self.E.shape[0], m, 6, 6))
        E[:, :self.cap] = self.E
        self.E = E
        self.free.extend(range(m - 1, self.cap - 1, -1))
        self.cap = m

    def _grow_rows(self, need):
        n0 = self.E.shape[0]
        if need <= n0:
            return
        m = max(need, (3 * n0) // 2)
        E = np.zeros((m, self.cap, 6, 6))
        E[:n0] = self.E
        self.E = E
        self.rows_free.extend(range(m - 1, n0 - 1, -1))

    def _alloc(self, kind, a=-1, b=-1, cov=None, born=-1):
        if not self.free:
            self._grow_slots(self.cap + 1)
        s = self.free.pop()
        self.kind[s], self.alive[s], self.ia[s], self.ib[s], self.born[s] = kind, True, a, b, born
        self.cov[s] = 0.0 if cov is None else cov
        self.E[:, s] = 0.0
        if kind == _K_IV:
            self.iv_of[(a, b)] = s
            for p in (a, b):
                if self.bpref.get(p, 0) == 0:
                    bisect.insort(self.bps, p)
                self.bpref[p] = self.bpref.get(p, 0) + 1
        self.fresh.append(s)
        return s

    def _free(self, s):
        if not self.alive[s]:
            return
        if self.kind[s] == _K_IV:
            a, b = int(self.ia[s]), int(self.ib[s])
            self.iv_of.pop((a, b), None)
            for p in (a, b):
                self.bpref[p] -= 1
                if self.bpref[p] == 0:
                    del self.bpref[p]
                    self.bps.pop(bisect.bisect_left(self.bps, p))
        self.alive[s] = False
        self.E[:, s] = 0.0
        self.free.append(s)
        self.n_freed += 1

    def kept_rows(self):
        return [self.row_of[e] for e in self.kept]

    # ---- the partition ------------------------------------------------------------------------------------------
    def ensure_bp(self, m):
        """make m a breakpoint: an interval slot (u, v) with u < m < v is split into (u, m] and (m, v] (exact)."""
        if m in self.bpref:
            return
        i = bisect.bisect_left(self.bps, m)
        if 0 < i < len(self.bps):
            u, v = self.bps[i - 1], self.bps[i]
            s = self.iv_of.get((u, v))
            if s is not None:
                s1 = self._alloc(_K_IV, u, m)
                s2 = self._alloc(_K_IV, m, v)
                self.E[:, s1] = self.E[:, s]
                self.E[:, s2] = self.E[:, s]
                self._free(s)
                self.n_split += 1

    def touch_drift(self, a, b):
        """create the interval slots of D(a -> b) = the elementary intervals of (a, b] (gaps get a slot)."""
        if b <= a:
            return
        self.ensure_bp(a)
        self.ensure_bp(b)
        i0 = bisect.bisect_right(self.bps, a)
        i1 = bisect.bisect_left(self.bps, b)
        pts = [a] + self.bps[i0:i1] + [b]
        for u, v in zip(pts[:-1], pts[1:]):
            if (u, v) not in self.iv_of:
                self._alloc(_K_IV, u, v)

    def drift_slots(self, a, b):
        """the slots of D(a -> b) on the CURRENT partition (after touch_drift)."""
        if b <= a:
            return []
        i0 = bisect.bisect_right(self.bps, a)
        i1 = bisect.bisect_left(self.bps, b)
        pts = [a] + self.bps[i0:i1] + [b]
        return [self.iv_of[(u, v)] for u, v in zip(pts[:-1], pts[1:])]

    # ---- epochs ---------------------------------------------------------------------------------------------------
    def add_epoch(self, eid, j, form, var, n_loops):
        assert eid == len(self.var)
        self.var.append(np.array(var, float))
        self.cumV.append(self.cumV[-1] + self.var[-1])
        if not self.rows_free:
            self._grow_rows(self.E.shape[0] + 1)
        row = self.rows_free.pop()
        self.E[row] = 0.0
        self.E[row, :form.shape[0]] = form
        self.row_of[eid] = row
        self.kept.append(eid)
        self.ep_j[eid] = int(j)
        self.ep_nl[eid] = int(n_loops)

    def clean_fresh(self):
        """slots created in this event that no kept form references (e.g. a refused loop's) are freed."""
        rows = self.kept_rows()
        for s in self.fresh:
            if self.alive[s] and (not rows or not np.any(self.E[rows, s] != 0.0)):
                self._free(s)
        self.fresh = []

    def slide(self, cov_iv):
        """window maintenance after an epoch was added (win > 0): drop the oldest epochs beyond `win`, fold every kept
        form's old coefficients into its new fold source.  cov_iv(slots) -> W covariances of interval slots now."""
        if self.win <= 0 or len(self.kept) <= self.win:
            return
        while len(self.kept) > self.win:
            e = self.kept.pop(0)
            row = self.row_of.pop(e)
            self.E[row] = 0.0
            self.rows_free.append(row)
            self.ep_j.pop(e, None)
            self.ep_nl.pop(e, None)
            self.n_drop += 1
        kH = self.ep_j[self.kept[0]]
        self.ensure_bp(kH)
        self.horizon = kH
        al = self.alive[:self.cap]
        kd = self.kind[:self.cap]
        old = al & (((kd == _K_IV) & (self.ib[:self.cap] <= kH)) |
                    (((kd == _K_Z) | (kd == _K_PI)) & (self.born[:self.cap] < self.kept[0])) | (kd == _K_FOLD))
        old = np.nonzero(old)[0]
        if len(old):
            Cv = self.cov[old].copy()
            iv = self.kind[old] == _K_IV
            if iv.any():
                Cv[iv] = cov_iv(old[iv])
            rows = self.kept_rows()
            Fo = self.E[np.ix_(rows, old)]
            add = np.einsum("rsab,sbc,rsdc->rad", Fo, Cv, Fo, optimize=True)
            self.E[np.ix_(rows, old)] = 0.0
            for e, r, A in zip(self.kept, rows, add):
                if np.abs(A).max() > 0.0:
                    f = self._alloc(_K_FOLD, cov=_sym(A), born=e)
                    self.E[r, f] = _I6
            for s in old:
                self._free(int(s))
            self.n_fold += 1
        self.fresh = []

    def n_sources(self):
        return int(self.alive[:self.cap].sum())

    def n_by_kind(self):
        al = self.alive[:self.cap]
        return {k_: int((al & (self.kind[:self.cap] == v)).sum()) for k_, v in
                (("iv", _K_IV), ("z", _K_Z), ("pi", _K_PI), ("fold", _K_FOLD))}


class HeadMV:
    """F2 MV-HEAD: the worker-side correction state for head = "mv" (same interface as HeadFilter: event / snapshot /
    restore / epoch / walls / ep_log / counters).  The provisional-episode logic of HeadFilter.event is kept verbatim;
    only the prior / loop arithmetic (the GLS above) and the weak-prior test are replaced (MV-5):
      weak  the largest eigenvalue modulus of W_L = I - W_0 (W_0 = the prior row's GLS weight) exceeds 1/2 (with
            uncorrelated errors this is lambda_max(S_0 I_L) > 1, the frozen weak_prior, exactly).
    Checkpoints carry T_target and lambda^head (the set of applied-and-kept epochs); the ledger's forms are facts (a
    reverted epoch stays as an ORPHAN until it leaves the window), published lineages are never reverted.
    ctx (set by the worker): kf (admit.KappaF), vi(k) -> bool, beta(c) -> lineage mixture ((w, (n_max, excl)), ...)."""

    def __init__(self, ledger_win=PCM_WIN, episodes="provisional", time_reps=1, kD=None):
        self.kD = huber_k_dim(6) if kD is None else float(kD)
        self.T_target = np.eye(4)
        self.epoch = 0
        if episodes not in ("provisional", "carry"):
            raise ValueError("HeadMV episodes must be provisional | carry, got %r" % (episodes,))
        self.episodes = episodes
        self.prov = None
        self.walls, self.m0_hist, self.ep_log = [], [], []
        self.n_prov = self.n_confirm = self.n_swap = self.n_drop = self.n_rival_held = 0
        self.led = Ledger(ledger_win)
        self.n_eps = 0
        self.reverted = set()
        self.j_last = 0
        self.ctx = None
        self.time_reps = int(time_reps)
        self.refusals = []
        self.f2_events = []
        self.n_l4_viol = 0
        self.last_info = None
        self._To = {}
        self.last_ev = None
        self.merge_links = {}                  # merge epoch id -> the bridged unknown link edges it aligned
        self._dbg = None

    # ---- lineage / state --------------------------------------------------------------------------------------------
    def lineage(self):
        """lambda of the target just published: (newest epoch id, reverted epochs)."""
        return (self.n_eps - 1, frozenset(self.reverted))

    def _state(self):
        return (self.T_target.copy(), self.n_eps, frozenset(self.reverted), self.j_last)

    def _set_state(self, st):
        T, n, rev, jl = st
        self.T_target = T.copy()
        self.reverted = set(rev) | set(range(n, self.n_eps))
        self.j_last = jl

    def snapshot(self):
        P = None
        if self.prov is not None:
            P = dict(self.prov)
            P["loops"] = list(self.prov["loops"])
            P["rival"] = list(self.prov["rival"])
        return (self.T_target.copy(), None, self.j_last, self.epoch, P, self.n_eps, frozenset(self.reverted))

    def restore(self, snap):
        T, _S, jl, _e, P, n, rev = snap
        self._set_state((T, n, rev, jl))
        self.prov = P
        self.epoch += 1

    def _T(self, k, T_odom):
        T = self._To.get(k)
        if T is None:
            T = np.asarray(T_odom(k), float)
            self._To[k] = T
        return T

    # ---- the event's kappa_F (3.3) ------------------------------------------------------------------------------------
    @staticmethod
    def _only(s):
        return ((1.0, 1.0), (0.0, 0.0)) if s == CLS_VI else ((0.0, 0.0), (1.0, 1.0))

    def _kappa_ev(self, j, cand, chain, t):
        kf = self.ctx.kf
        kv = [[1.0, 1.0], [1.0, 1.0]]
        st = [["absent", "absent"], ["absent", "absent"]]
        uu = [[None, None], [None, None]]
        cold = [kf.n_before(s, t) < SUFF for s in (CLS_VI, CLS_VO)]
        for s in (CLS_VI, CLS_VO):
            us = ([], [])
            for x in cand:
                if chain.class_count(x["c"], x["q"])[s] > 0:
                    S = chain.p_kappa_rel(x["c"], x["q"], self._only(s), 0.0)[2]
                    us[0].append(float(np.linalg.slogdet(S[:3, :3])[1]))
                    us[1].append(float(np.linalg.slogdet(S[3:, 3:])[1]))
            if not us[0] and self.j_last < j and chain.class_count(self.j_last, j)[s] > 0:
                S = chain.p_kappa_rel(self.j_last, j, self._only(s), 0.0)[2]
                us[0].append(float(np.linalg.slogdet(S[:3, :3])[1]))
                us[1].append(float(np.linalg.slogdet(S[3:, 3:])[1]))
            for b in (0, 1):
                v = [u for u in us[b] if np.isfinite(u)]
                if v:
                    uu[s][b] = float(np.median(v))
                    kv[s][b], st[s][b] = kf.value(s, b, uu[s][b], t)
                elif cold[s]:
                    st[s][b] = "cold"
        return (tuple(kv[0]), tuple(kv[1])), st, uu, cold

    # ---- lineage gap coefficients (3.4 / 3.5) -------------------------------------------------------------------------
    def _g_inwin(self, beta):
        kept = np.asarray(self.led.kept, np.int64)
        if not len(kept):
            return kept, np.zeros(0)
        g = np.array([0.0 if e in self.reverted else 1.0 for e in kept])
        for w, (nk, ex) in beta:
            m = (kept <= nk) & np.array([e not in ex for e in kept])
            g -= w * m
        return kept, g

    def _g_one(self, n, beta):
        g = 0.0 if n in self.reverted else 1.0
        for w, (nk, ex) in beta:
            if n <= nk and n not in ex:
                g -= w
        return g

    def _completion(self, beta):
        """out-of-window part of the lineage gap -> sum_n g_n^2 Var(Delta_n) (anchor-private, conservative)."""
        led = self.led
        e_lo = led.kept[0] if led.kept else self.n_eps
        if e_lo <= 0:
            return None
        cum = led.cumV
        nks = sorted(set(min(nk, e_lo - 1) for _w, (nk, _ex) in beta))
        out = np.zeros((6, 6))
        # pieces of constant c(n) = sum_{k: n_k < n} w_k (no exclusion), n in [0, e_lo)
        edges = [-1] + [nk for nk in nks if nk >= 0] + [e_lo - 1]
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            if hi_ <= lo_:
                continue
            n0, n1 = lo_ + 1, hi_                                  # epochs n0..n1 inclusive
            c = sum(w for w, (nk, _ex) in beta if nk < n0)
            if c != 0.0:
                out += c * c * (cum[n1 + 1] - cum[n0])
        special = set(x for x in self.reverted if x < e_lo)
        for _w, (nk, ex) in beta:
            special |= set(x for x in ex if x < e_lo)
        for n in special:
            c = sum(w for w, (nk, _ex) in beta if nk < n)
            g = self._g_one(n, beta)
            out += (g * g - c * c) * led.var[n]
        if np.abs(out).max() == 0.0:
            return None
        return _sym(out)

    # ---- loop terms (pure) --------------------------------------------------------------------------------------------
    def _terms(self, j, Tj, loops, chain, T_odom, anchors, t):
        """-> (used terms, event context).  A loop whose anchor-private provenance drift carries no information
        (a class in it uncalibrated, or an unknown link) is REFUSED (stage f2_vianchor), not fused."""
        ctx = self.ctx
        hc = CLS_VI if ctx.vi(j) else CLS_VO
        o = chain.origin(j, hc)
        cand = []
        for L in loops:
            q, c = int(L["q"]), int(L["c"])
            A = anchors(c)
            if A is None:
                continue
            Tq = self._T(q, T_odom)
            B = inv(Tq) @ Tj
            Aq = Ad(Tq)
            SlW = Aq @ to_right(L["Z"][:3, :3], L["Sl"]) @ Aq.T
            cand.append({"L": L, "q": q, "c": c, "H": A @ L["Z"] @ B, "SlW": _sym(SlW)})
        if not cand:
            return [], None
        kv, st, uu, cold = self._kappa_ev(j, cand, chain, t)
        used = []
        for x in cand:
            c, q = x["c"], x["q"]
            x["ce"] = max(c, o)
            priv = None
            reason = None
            if c < o:
                if chain.n_unbridged(c, o) > 0:
                    reason = "unknown_link"
                else:
                    cnt = chain.class_count(c, o)
                    bad = [s for s in (CLS_VI, CLS_VO) if cnt[s] > 0 and cold[s]]
                    if bad:
                        reason = "cold_class_%s" % ("vi" if bad[0] == CLS_VI else "vo")
                    else:
                        ZR, _p, S = chain.p_kappa_rel(c, o, kv)
                        J = Ad(self._T(o, T_odom)) @ _Kr_many(ZR[None])[0]
                        priv = _sym(J @ S @ J.T)
            beta = ctx.beta(c)
            if reason is None and self.merge_links:
                # an anchor on the head's side of an R1 merge whose epoch the anchor and the head do not share is in a
                # frame the merge re-aligned (its offset is the unknown link's): no information (declared, F2_BUILD)
                for m, ks in self.merge_links.items():
                    if any(k < c for k in ks) and abs(self._g_one(m, beta)) > 1e-12:
                        reason = "unaligned"
                        break
            if reason is not None:
                self.refusals.append({"j": int(j), "q": q, "c": c, "o": int(o), "why": reason, "t": t,
                                      "k": x["L"].get("k")})
                x["L"]["f2_refused"] = reason
                continue
            x["beta"] = beta
            comp = self._completion(beta)
            if comp is not None:
                priv = comp if priv is None else priv + comp
            x["priv"] = priv
            used.append(x)
        return used, {"kv": kv, "st": st, "u": uu, "o": int(o), "hc": int(hc), "cold": cold}

    # ---- the solve ------------------------------------------------------------------------------------------------
    def _alloc_event(self, j, terms):
        """mutating, exact: breakpoints and interval slots of every drift form of the event, loop / private slots."""
        led = self.led
        for x in terms:
            led.touch_drift(x["ce"], j)
            led.touch_drift(x["q"], j)
        for x in terms:
            x["z_slot"] = led._alloc(_K_Z, cov=x["SlW"], born=self.n_eps)
            x["pi_slot"] = None if x["priv"] is None else led._alloc(_K_PI, cov=x["priv"], born=self.n_eps)

    def _cov_iv(self, slots, chain, kv, T_odom):
        led = self.led
        slots = np.asarray(slots, np.int64)
        if not len(slots):
            return np.zeros((0, 6, 6))
        A, B = led.ia[slots], led.ib[slots]
        ZR, S = chain.p_kappa_rel_many(A, B, kv)
        Tb = np.stack([self._T(int(b), T_odom) for b in B])
        J = _Ad_many(Tb) @ _Kr_many(ZR)
        C = _sym(J @ S @ np.transpose(J, (0, 2, 1)))
        nu = np.array([chain.n_unbridged(int(a), int(b)) for a, b in zip(A, B)])
        if nu.any():
            C = C + (nu * NO_INFO)[:, None, None] * _I6[None]
        return C

    def _forms(self, j, Tj, terms, ev, no_prior, chain, T_odom, log=True):
        """pure: the forms, the covariance C of the stacked rows (head right), the refusal of unaligned loops (logged
        once: log=False on the timing re-executions)."""
        led = self.led
        kv = ev["kv"]
        while True:
            L = len(terms)
            cap = led.cap
            Psi = np.zeros((L, cap, 6, 6))
            Nu = np.zeros((L, cap, 6, 6))
            krows = led.kept_rows()
            for i, x in enumerate(terms):
                Psi[i, led.drift_slots(x["ce"], j)] += _I6
                if krows:
                    _k, g = self._g_inwin(x["beta"])
                    nz = np.nonzero(g)[0]
                    if len(nz):
                        gf = np.zeros(led.E.shape[0])                      # over the whole E view: no row copy
                        gf[[krows[n] for n in nz]] = g[nz]
                        Psi[i] += np.tensordot(gf, led.E, axes=1)[:cap]
                Nu[i, led.drift_slots(x["q"], j)] += _I6
                Nu[i, x["z_slot"]] += _I6
                if x["pi_slot"] is not None:
                    Nu[i, x["pi_slot"]] += _I6
            ref = np.nonzero(np.any(Psi != 0.0, axis=(0, 2, 3)) | np.any(Nu != 0.0, axis=(0, 2, 3)))[0]
            Cv = led.cov[ref].copy()
            ivm = led.kind[ref] == _K_IV
            if ivm.any():
                Cv[ivm] = self._cov_iv(ref[ivm], chain, kv, T_odom)
            Psr, Nur = Psi[:, ref], Nu[:, ref]
            VN = np.einsum("isab,sbc,isdc->iad", Nur, Cv, Nur, optimize=True)
            Ii = np.linalg.inv(VN + 1e-30 * _I6)
            om = np.linalg.solve(Ii.sum(0), Ii)                                       # omega_i (W)
            Pbar = np.einsum("iab,isbc->sac", om, Psr)
            # unaligned guard (declared, F2_BUILD): a loop whose row keeps an unknown-link interval (an anchor in a frame
            # the head's lineage has since re-aligned) carries no information.  The row tested is the one the solve
            # uses: the innovation nu_i - Psi_ci with a prior row; Psibar - Psi_ci + nu_i without (a merge: the common
            # link offset cancels there).
            unk = np.zeros(len(ref), bool)
            if ivm.any():
                unk[ivm] = np.array([chain.n_unbridged(int(led.ia[s]), int(led.ib[s])) > 0 for s in ref[ivm]])
            if unk.any():
                if no_prior:
                    tst = [(Pbar - Psr[i] + Nur[i])[unk] for i in range(L)]
                else:
                    tst = [(Nur[i] - Psr[i])[unk] for i in range(L)]
                bad = [i for i in range(L) if np.abs(tst[i]).max() > 1e-6]
                if bad:
                    for i in bad:
                        x = terms[i]
                        if log:
                            self.refusals.append({"j": int(j), "q": x["q"], "c": x["c"], "o": ev["o"],
                                                  "why": "unaligned", "t": ev.get("t"), "k": x["L"].get("k")})
                        x["L"]["f2_refused"] = "unaligned"
                    terms = [x for i, x in enumerate(terms) if i not in bad]
                    if not terms:
                        return None
                    continue
            G = Ad(inv(Tj))
            rows = [Pbar] if not no_prior else []
            rows += [Pbar - Psr[i] + Nur[i] for i in range(L)]
            F = G[None, None] @ np.stack(rows)                                         # (nr, nref, 6, 6) head right
            nr = len(rows)
            X = F @ Cv[None]
            C = np.einsum("msac,nsdc->mand", X, F, optimize=True).reshape(nr * 6, nr * 6)
            C = 0.5 * (C + C.T)
            return {"terms": terms, "ref": ref, "Cv": Cv, "F": F, "Fbar": G[None] @ Pbar, "C": C, "om": om, "G": G,
                    "no_prior": no_prior, "nr": nr}

    def _gls_weights(self, Cw, nr):
        Ci = np.linalg.inv(Cw)
        B4 = Ci.reshape(nr, 6, nr, 6)
        A = B4.sum(axis=(0, 2))
        R = B4.sum(axis=0)                                                            # (6, nr, 6)
        Ai = np.linalg.inv(A)
        return [Ai @ R[:, n, :] for n in range(nr)], Ai

    def _weak(self, fm):
        """MV-5: weak iff max |eig(W_L)| > 1/2, W_L = I - W_0 (no robust weights, as the frozen test)."""
        if fm["no_prior"]:
            return False, None
        W, _Ai = self._gls_weights(fm["C"], fm["nr"])
        lam = float(np.abs(np.linalg.eigvals(_I6 - W[0])).max())
        return lam > 0.5, lam

    def _solve(self, fm, H0, Tj):
        """pure: robust GLS + Gauss-Newton -> (H, W (final IRLS GLS weights per row), x, robust weights)."""
        nr, C = fm["nr"], fm["C"]
        terms = fm["terms"]
        prior = not fm["no_prior"]
        Ts = ([H0] if prior else []) + [x["H"] for x in terms]
        off = 1 if prior else 0
        L = len(terms)
        Cb = C.reshape(nr, 6, nr, 6)
        Vin = []
        for i in range(L):
            m = i + off
            V = Cb[m, :, m, :] + (Cb[0, :, 0, :] - Cb[m, :, 0, :] - Cb[0, :, m, :] if prior else 0.0)
            Vin.append(np.linalg.inv(0.5 * (V + V.T) + 1e-30 * _I6))
        H = H0.copy()
        W = None
        wts = np.ones(L)
        for _ in range(ITERS):
            r = np.stack([logr(inv(H) @ T) for T in Ts])
            for i in range(L):
                d = r[i + off] - (r[0] if prior else 0.0)
                mi = float(np.sqrt(max(d @ Vin[i] @ d, 0.0)))
                wts[i] = 1.0 if mi <= self.kD else self.kD / mi
            dv = np.ones(nr)
            dv[off:] = 1.0 / np.sqrt(wts)
            dd = np.repeat(dv, 6)
            W, _Ai = self._gls_weights(C * np.outer(dd, dd), nr)
            dx = sum(W[n] @ r[n] for n in range(nr))
            H = ortho(H @ expr(dx))
            if np.abs(dx).max() < 1e-9:
                break
        x = logr(inv(H0) @ H)
        return H, W, x, wts

    def _increment(self, fm, W, Tj):
        """Delta_new = sum_i W_i (nu_i - Psi_ci) (head right -> W): a dense form over the ledger's slots, its Var."""
        off = 0 if fm["no_prior"] else 1
        D = np.zeros((len(fm["ref"]), 6, 6))
        for i in range(len(fm["terms"])):
            D += W[i + off][None] @ (fm["F"][i + off] - fm["Fbar"])
        D = Ad(Tj)[None] @ D
        Var = _sym(np.einsum("sab,sbc,sdc->ad", D, fm["Cv"], D, optimize=True))
        full = np.zeros((self.led.cap, 6, 6))
        full[fm["ref"]] = D
        return full, Var

    def _fuse(self, j, Tj, terms, ev, chain, T_odom, no_prior, decide_weak=False):
        """allocate, build, (weak test), solve, commit one epoch.  -> (m0, weak, info) or None (nothing fusable)."""
        led = self.led
        tA = time.perf_counter()
        self._alloc_event(j, terms)
        tA = time.perf_counter() - tA
        H0 = self.T_target @ Tj
        best = None
        out = None
        for _rep in range(max(1, self.time_reps)):
            t1 = time.perf_counter()
            fm = self._forms(j, Tj, terms, ev, no_prior, chain, T_odom, log=(_rep == 0))
            if fm is None:
                led.clean_fresh()
                return None
            weak = self._weak(fm) if decide_weak else (False, None)
            H, W, x, wts = self._solve(fm, H0, Tj)
            form, Var = self._increment(fm, W, Tj)
            dt = time.perf_counter() - t1
            best = dt if best is None else min(best, dt)
            out = (fm, weak, H, W, x, wts, form, Var)
        fm, weak, H, W, x, wts, form, Var = out
        self._dbg = {"fm": fm, "W": W, "H0": H0, "H": H, "x": x, "weak": weak, "Var": Var}   # tests (A3) read it
        tC = time.perf_counter()
        eid = self.n_eps
        self.n_eps += 1
        if no_prior:                                     # an R1 merge across unknown links: remember which (alignment)
            c0 = min(x["c"] for x in fm["terms"])
            ks = [k for k in chain.unknown_links(c0, j) if chain.bridged(k)]
            if ks:
                self.merge_links[eid] = ks
        led.add_epoch(eid, j, form, Var, len(fm["terms"]))
        led.clean_fresh()
        self.T_target = H @ inv(Tj)
        self.j_last = j
        kv = ev["kv"]
        led.slide(lambda s: self._cov_iv(s, chain, kv, T_odom))
        tC = time.perf_counter() - tC
        nref = len(fm["ref"])
        L = len(fm["terms"])
        # ledger bound: <= win epochs, S <= M_L (L_max + 1) + 2 M_L L_max + M_L
        ML = led.win if led.win > 0 else max(len(led.kept), 1)
        Lmax = max(led.ep_nl.values()) if led.ep_nl else L
        S_led = led.n_sources()
        bound = ML * (Lmax + 1) + 2 * ML * Lmax + ML
        ok = (led.win <= 0 or len(led.kept) <= led.win) and S_led <= bound
        if not ok:
            self.n_l4_viol += 1
        m0 = 0.0
        if not no_prior:
            C00 = fm["C"][:6, :6]
            m0 = float(x @ np.linalg.solve(C00 + 1e-30 * _I6, x))
        G = fm["G"]
        Gi = np.linalg.inv(G)
        ybar = sum((G @ fm["om"][i] @ Gi) @ logr(inv(H0) @ y["H"]) for i, y in enumerate(fm["terms"]))
        gt = float(np.linalg.norm(x[:3]) / max(np.linalg.norm(ybar[:3]), 1e-15))
        gr = float(np.linalg.norm(x[3:]) / max(np.linalg.norm(ybar[3:]), 1e-15))
        self.last_info = np.linalg.inv(fm["C"][6 * (1 - int(no_prior)):6 * (2 - int(no_prior)),
                                       6 * (1 - int(no_prior)):6 * (2 - int(no_prior))] + 1e-30 * _I6)
        info = {"eid": eid, "L": L, "S": int(nref), "work": int((fm["nr"]) ** 2 * nref),
                "solve_ms": 1e3 * (tA + best + tC), "solve_pure_ms": 1e3 * best, "kv": [list(kv[0]), list(kv[1])],
                "kst": ev["st"], "u": ev["u"], "o": ev["o"], "hc": ev["hc"], "gain_t": gt, "gain_r": gr,
                "huber_w": [float(w) for w in wts], "kept": len(led.kept), "S_ledger": S_led, "S_bound": int(bound),
                "l4_ok": bool(ok), "folds": led.n_fold, "splits": led.n_split, "src": led.n_by_kind(),
                "weak_lam": weak[1], "no_prior": bool(no_prior), "m0": m0, "var_tr": float(np.trace(Var))}
        self.f2_events.append(info)
        return m0, weak, info, fm["terms"]

    @staticmethod
    def _best_set(pool, L, chain, kv):
        return HeadFilter._best_set(pool, L, chain, kv)

    # ---- one event (HeadFilter.event's episode logic, verbatim; the arithmetic replaced) ------------------------------
    def event(self, j, T_odom_j, loops, chain, T_odom, anchors, kv, no_prior=False, t=None):
        t0 = time.perf_counter()
        Tj = np.asarray(T_odom_j, float)
        self._To[j] = Tj
        terms, ev = self._terms(j, Tj, loops, chain, T_odom, anchors, t)
        if not terms:
            return None
        ev["t"] = t
        used = [x["L"] for x in terms]
        tix = {id(x["L"]): x for x in terms}
        if self.episodes == "carry":
            r = self._fuse(j, Tj, terms, ev, chain, T_odom, no_prior)
            if r is None:
                return None
            m0, _w, info, fused_t = r
            return self._done(t0, [x["L"] for x in fused_t], m0, "carry", None, info)
        from xclean.slam.merge import pcm_pair
        rec = {"j": j, "t": t, "n_new": len(used)}
        lam_w = None
        info = None
        if no_prior:
            if self.prov is not None:
                self.n_drop += 1
                rec["closed_prov"] = len(self.prov["loops"])
                self.prov = None
            r = self._fuse(j, Tj, terms, ev, chain, T_odom, True)
            if r is None:
                return None
            m0, _w, info, fused_t = r
            ep, fused = "merge", [x["L"] for x in fused_t]
        elif self.prov is None:
            ck = self._state()
            r = self._fuse(j, Tj, terms, ev, chain, T_odom, False, decide_weak=True)
            if r is None:
                return None
            m0, (weak, lam_w), info, fused_t = r
            fused = [x["L"] for x in fused_t]
            ep = "commit"
            if weak:
                self.prov = {"loops": list(fused), "rival": [], "ck": ck, "j0": j, "t0": t, "id": self.n_prov}
                self.n_prov += 1
                ep = "open"
        else:
            P = self.prov
            joined, riv = [], []
            for L in used:
                ok = any(pcm_pair(chain, M, L, kv) for M in reversed(P["loops"][-PCM_WIN:]))
                (joined if ok else riv).append(L)
            P["rival"].extend(riv)
            P["rival"] = P["rival"][-PCM_WIN:]
            self.n_rival_held += len(riv)
            rec.update(joined=len(joined), rival=len(riv))
            R = self._best_set(P["rival"], riv[-1], chain, kv) if riv else []
            rec["rival_set"] = len(R)
            if len(R) > len(P["loops"]) + len(joined):
                rest = [x for x in P["rival"] if not any(x is y for y in R)]
                old = P["loops"] + joined
                self._set_state(P["ck"])
                tr, evr = self._terms(j, Tj, R, chain, T_odom, anchors, t)
                r = None
                if tr:
                    evr["t"] = t
                    r = self._fuse(j, Tj, tr, evr, chain, T_odom, False)
                if r is None:
                    m0, fused = 0.0, []
                else:
                    m0, _w, info, fused_t = r
                    fused = [x["L"] for x in fused_t]
                P["loops"] = list(R)
                P["rival"] = (rest + old)[-PCM_WIN:]
                self.n_swap += 1
                ep = "swap"
            elif joined:
                r = self._fuse(j, Tj, [tix[id(L)] for L in joined], ev, chain, T_odom, False)
                if r is None:
                    rec.update(ep="join_refused", n_prov=len(P["loops"]))
                    self.ep_log.append(rec)
                    return None
                m0, _w, info, fused_t = r
                fused = [x["L"] for x in fused_t]
                P["loops"].extend(fused)
                ep = "join"
            else:
                rec.update(ep="rival_held", n_prov=len(P["loops"]))
                self.ep_log.append(rec)
                return None
            P["loops"] = P["loops"][-PCM_WIN:]
        if self.prov is not None and self.prov["loops"]:
            qs = [int(L["q"]) for L in self.prov["loops"]]
            rec.update(n_prov=len(qs), q_span=max(qs) - min(qs), prov_id=self.prov["id"])
            if max(qs) - min(qs) > N_KF:
                self.n_confirm += 1
                rec["rival_dropped"] = len(self.prov["rival"])
                self.prov = None
                ep = ep + "+confirm"
        rec.update(ep=ep, lam_w=lam_w, m0=m0)
        self.ep_log.append(rec)
        return self._done(t0, fused, m0, ep, rec, info)

    def _done(self, t0, used, m0, ep, rec, info=None):
        wall = time.perf_counter() - t0
        self.walls.append(wall)
        self.epoch += 1
        self.m0_hist.append(m0)
        out = {"T_new": self.T_target.copy(), "used": used, "m0": m0, "wall_s": wall, "epoch": self.epoch, "ep": ep,
               "provisional": self.prov is not None, "f2": info, "lineage": self.lineage()}
        self.last_ev = out
        return out


# ==================================================================================================== F3-CAL
# P1.3 fix round 3, designer "cal": CONSISTENCY FIRST.  The frozen anchored head filter
# (HeadFilter, p13b_fix1) with ONE change in the fusion: the prior covariance S0 of every non-merge event is scaled per
# block (head-right rho = translation, phi = rotation) by mu_b in [0, 1], estimated online from the head filter's OWN
# innovations by covariance matching (Mehra 1970; Myers and Tapley 1976): per fused loop i the innovation
# nu_i = log(H0^-1 H_i) has predicted covariance mu S0 + S_i; one sample per block solves
# nu_b^T (mu S0_bb + S_i,bb)^-1 nu_b = chi2_3 median (the rule of Kappa._solve); mu_b(t) = min(1, median of the last SUFF
# samples of the event's SPAN CLASS decided strictly before t); cold (< SUFF samples) -> 1 = the frozen prior.
# Span class (class D, the logdet break-even of kappa): "long" = the prior is drift-dominated (no carried posterior, or
# logdet kappa Sigma_chain(q_last -> j) > logdet of the carried posterior); "short" = posterior-dominated.
# Merges (no_prior) and unknown-link priors are not calibrated (frozen).  The episode logic (weak-prior test, rivals,
# swaps, confirm) runs on the FROZEN S0: its protection is unchanged; CAL changes only the weight of the fusion.
# mu <= 1 (class D): CAL can only lower a gain below the p13b_fix1 gain, never raise it; a wrong episode with large
# innovations cannot buy itself more weight.  Off (head = "anchored") the frozen class above runs, unchanged.
# =====================================================================================================================
_CHI2_3_MED = 2.3659738843753377
_RHO, _PHI = slice(0, 3), slice(3, 6)
from xclean.slam.admit import CHI2_6_999                              # noqa: E402  (carried: PCM's level)


def cal_solve(nu, A, L, med=_CHI2_3_MED):
    """mu >= 0 with nu^T (mu A + L)^-1 nu = med (the rule, bounds and 70 bisections of Kappa._solve), evaluated in the
    L-whitened eigenbasis of A (closed form per mu).  0 when mu -> 0 already explains nu.  None on a numerical failure."""
    try:
        C = np.linalg.cholesky(0.5 * (L + L.T) + 1e-15 * np.eye(len(nu)))
        Ci = np.linalg.inv(C)
        w = Ci @ nu
        Aw = Ci @ A @ Ci.T
        lam, U = np.linalg.eigh(0.5 * (Aw + Aw.T))
    except np.linalg.LinAlgError:
        return None
    lam = np.maximum(lam, 0.0)
    a2 = (U.T @ w) ** 2
    if not np.all(np.isfinite(a2)) or not np.all(np.isfinite(lam)):
        return None

    def f(m):
        return float(np.sum(a2 / (m * lam + 1.0)))
    if f(1e-12) <= med:
        return 0.0
    lo, hi = 1e-6, 1e12
    for _ in range(70):
        mid = np.sqrt(lo * hi)
        if f(mid) > med:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


class HeadCal(HeadFilter):
    """HeadFilter with the F3-CAL weighting.  Same state, same episodes, same publish.
    mode "rel" (THE DESIGN): every loop carrying its admission residual (cal_r) and raw chain covariance (cal_Sc) is
    pre-fused with the odometry over its own span at the calibrated minimum-variance weight (xclean.slam.cal.RelCal):
    Z_cal replaces Z in the fusion term only (the episode / PCM logic reads Z).  mode "prior" (development arm CAL-1,
    declared in F3_CAL_DESIGN 7): the head prior S0 scaled per block by the innovation-matched mu (the code above)."""

    CLASSES = ("long", "short")

    def __init__(self, prior_mode="posterior", episodes="provisional", suff=SUFF, time_reps=1, mode="rel", relcal=None,
                 gate="nis"):
        super().__init__(prior_mode=prior_mode, episodes=episodes)
        if mode not in ("rel", "prior"):
            raise ValueError("HeadCal mode must be rel | prior, got %r" % (mode,))
        if gate not in ("nis", "off"):
            raise ValueError("HeadCal gate must be nis | off, got %r" % (gate,))
        self.mode = mode
        self.relcal = relcal
        # F3-CAL gate: a loop whose NIS against the odometry over its own span, under the CALIBRATED
        # covariance (at least one block warm), exceeds chi2_6(0.999) (CHI2_6_999, carried: PCM's level) is not fused
        self.gate = gate
        self.gate_log = []                   # per refused loop: q, c, j, nis, nis_k, planted, states
        self.pre_log = []                    # mode rel: per pre-fused loop (q, c, class, state, lambda, gains)
        self.suff = int(suff)
        self.time_reps = int(time_reps)
        self.smp = {c: {"t": [], "mu": []} for c in self.CLASSES}
        self._pend = []
        self._t = None
        self._cls = None
        self.cal_log = []                    # per fused (non-merge) apply: class, mu used, state, n, samples, gain proxies
        self.n_sample_fail = 0
        # F3B arm X, X1 (F3B_DESIGN X1): the worker's running composition of the prior's chain (xclean.slam.cal.
        # RunningRel), attached by the worker when cfg cal_rc = "on"; None = the frozen chain.rel (CAL-b's path)
        self.rc = None

    # ---- calibration -------------------------------------------------------------------------------------------------
    def mu(self, cls, t):
        """-> ((mu_rho, mu_phi), state, n): min(1, median of the last SUFF samples of the class decided before t)."""
        S = self.smp[cls]
        n = len(S["t"]) if t is None else bisect.bisect_left(S["t"], t)
        if n < self.suff:
            return (1.0, 1.0), "cold", n
        P = np.asarray(S["mu"][n - self.suff:n], float)
        return (min(1.0, float(np.median(P[:, 0]))), min(1.0, float(np.median(P[:, 1])))), "warm", n

    def _commit(self, t):
        for cls, mr, mp in self._pend:
            self.smp[cls]["t"].append(-np.inf if t is None else t)
            self.smp[cls]["mu"].append((mr, mp))
        self._pend = []

    # ---- frozen pieces, instrumented ---------------------------------------------------------------------------------
    def _terms(self, j, Tj, loops, chain, T_odom, anchors):
        if self.mode != "rel":
            return super()._terms(j, Tj, loops, chain, T_odom, anchors)
        # the frozen arithmetic with Z_cal in place of Z where a loop was pre-fused (the same loop dicts are returned)
        sub = []
        for L in loops:
            if L.get("cal_refused"):
                continue                     # F3-CAL gate: inconsistent with the calibrated odometry over its span
            if "Zcal" in L:
                L2 = dict(L)
                L2["Z"] = L["Zcal"]
                L2["_orig"] = L
                sub.append(L2)
            else:
                sub.append(L)
        His, Sis, used = super()._terms(j, Tj, sub, chain, T_odom, anchors)
        return His, Sis, [L.get("_orig", L) for L in used]

    def _prior(self, j, Tj, chain, T_odom, kv, no_prior):
        if self.rc is not None:
            # X1: the frozen prior text, its chain.rel(q_last, j) served by the running composition (same bytes)
            from xclean.slam.cal import RCChain
            chain = RCChain(chain, self.rc)
        H0, S0 = super()._prior(j, Tj, chain, T_odom, kv, no_prior)
        if self.mode != "prior":
            return H0, S0
        cls = None
        if not no_prior and self.prior_mode == "posterior":
            q_last = self.q_last
            g = chain.rel(q_last, j) if q_last < j else None
            if q_last < j and g is None:
                cls = None                                   # unknown link: a NO_INFO prior, not calibrated
            elif self.Spost is None:
                cls = "long"
            else:
                Bl = inv(T_odom(q_last)) @ Tj
                Ab = Ad(inv(Bl))
                Sp = Ab @ self.Spost @ Ab.T
                if g is None:
                    cls = "short"
                else:
                    Sc = to_right(Bl[:3, :3], Kappa.scale(g[2], kv))
                    try:
                        cls = "long" if np.linalg.slogdet(Sc + 1e-300 * np.eye(6))[1] > \
                            np.linalg.slogdet(Sp + 1e-300 * np.eye(6))[1] else "short"
                    except np.linalg.LinAlgError:
                        cls = None
        self._cls = cls
        return H0, S0

    def _apply(self, j, Tj, H0, S0, His, Sis):
        cls = self._cls
        if self.mode != "prior" or cls is None:
            return super()._apply(j, Tj, H0, S0, His, Sis)
        (mr, mp), st, n = self.mu(cls, self._t)
        smp = []
        iH0 = inv(H0)
        for Hi, Si in zip(His, Sis):
            nu = logr(iH0 @ Hi)
            a = cal_solve(nu[_RHO], S0[_RHO, _RHO], Si[_RHO, _RHO])
            b = cal_solve(nu[_PHI], S0[_PHI, _PHI], Si[_PHI, _PHI])
            if a is None or b is None:
                self.n_sample_fail += 1
                continue
            smp.append((a, b))
            self._pend.append((cls, a, b))
        d = np.r_[[np.sqrt(mr)] * 3, [np.sqrt(mp)] * 3]
        S0s = S0 * np.outer(d, d)
        nL = max(len(His), 1)
        Sl_r = float(np.mean([np.trace(S[_RHO, _RHO]) for S in Sis]))
        Sl_p = float(np.mean([np.trace(S[_PHI, _PHI]) for S in Sis]))
        p_r, p_p = float(np.trace(S0s[_RHO, _RHO])), float(np.trace(S0s[_PHI, _PHI]))
        f_r, f_p = float(np.trace(S0[_RHO, _RHO])), float(np.trace(S0[_PHI, _PHI]))
        self.cal_log.append({"j": int(j), "t": self._t, "cls": cls, "mu": [mr, mp], "st": st, "n": n,
                             "n_loops": len(His), "samples": smp,
                             "gain_proxy": [p_r / (p_r + Sl_r / nL), p_p / (p_p + Sl_p / nL)],
                             "gain_proxy_frozen": [f_r / (f_r + Sl_r / nL), f_p / (f_p + Sl_p / nL)]})
        return super()._apply(j, Tj, H0, S0s, His, Sis)

    # ---- one event ---------------------------------------------------------------------------------------------------
    def _event1(self, j, T_odom_j, loops, chain, T_odom, anchors, kv, no_prior=False, t=None):
        self._t = t
        self._cls = None
        self._prefuse(j, loops, no_prior, t, chain)
        try:
            out = HeadFilter.event(self, j, T_odom_j, loops, chain, T_odom, anchors, kv, no_prior=no_prior, t=t)
        finally:
            self._commit(t)
        return out

    def _prefuse(self, j, loops, no_prior, t, chain):
        """mode rel: Z_cal for every loop marked by the worker (cal_r: it passed admission with a known chain), once per
        loop (idempotent).  The residual is re-formed HERE from the Z that enters the fusion (admit.residual on the
        same chain c -> q): bit-identical to the admission residual for an unplanted loop, and consistent for a loop
        whose Z was replaced after admission."""
        if self.mode == "rel" and self.relcal is not None and not no_prior:
            from xclean.slam.admit import residual as _residual
            for L in loops:
                if "cal_r" in L and "Zcal" not in L and "cal_refused" not in L:
                    # the admission-time chain c -> q when the worker carries it (attempt-2 cost fix; the same Chain
                    # state within one job), else formed here
                    g = L.get("cal_g")
                    if g is None:
                        g = chain.rel(L["c"], L["q"]) if L["c"] < L["q"] else None
                    if g is None:
                        continue
                    Zc, info = self.relcal.prefuse(L["Z"], _residual(g, L["Z"]), g[2], L["Sl"], t, L["kv"])
                    info.update(q=int(L["q"]), c=int(L["c"]), j=int(j), t=t, planted=bool(L.get("planted", False)))
                    self.pre_log.append(info)
                    if self.gate == "nis" and info["st"] in ("warm", "mixed") and info["nis"] > CHI2_6_999:
                        L["cal_refused"] = "nis"
                        self.gate_log.append({k_: info[k_] for k_ in ("q", "c", "j", "t", "nis", "nis_k", "planted",
                                                                      "sts", "lam")})
                        continue
                    L["Zcal"] = Zc

    def _save(self):
        P = None
        if self.prov is not None:
            P = dict(self.prov)
            P["loops"] = list(self.prov["loops"])
            P["rival"] = list(self.prov["rival"])
        return (self.T_target.copy(), None if self.Spost is None else self.Spost.copy(), self.q_last, self.epoch, P,
                len(self.walls), len(self.m0_hist), len(self.ep_log), len(self.cal_log), list(self._pend),
                {c: len(self.smp[c]["t"]) for c in self.CLASSES}, self.n_prov, self.n_confirm, self.n_swap,
                self.n_drop, self.n_rival_held, self.n_sample_fail)

    def _load(self, s):
        (T, S, q, e, P, nw, nm, ne, nc, pend, ns, self.n_prov, self.n_confirm, self.n_swap, self.n_drop,
         self.n_rival_held, self.n_sample_fail) = s
        self.T_target, self.Spost, self.q_last, self.epoch = T.copy(), (None if S is None else S.copy()), q, e
        if P is not None:
            P2 = dict(P)
            P2["loops"] = list(P["loops"])
            P2["rival"] = list(P["rival"])
            P = P2
        self.prov = P
        del self.walls[nw:]
        del self.m0_hist[nm:]
        del self.ep_log[ne:]
        del self.cal_log[nc:]
        self._pend = list(pend)
        for c in self.CLASSES:
            del self.smp[c]["t"][ns[c]:]
            del self.smp[c]["mu"][ns[c]:]

    def event(self, j, T_odom_j, loops, chain, T_odom, anchors, kv, no_prior=False, t=None):
        """HeadFilter.event with the calibrated prior.  time_reps > 1 (replay, model clock only; L1): the event is
        executed time_reps times from the same saved state; the minimum wall time is reported (outputs identical)."""
        if self.time_reps <= 1:
            return self._event1(j, T_odom_j, loops, chain, T_odom, anchors, kv, no_prior=no_prior, t=t)
        tp0 = time.perf_counter()
        self._prefuse(j, loops, no_prior, t, chain)          # timed once (it is idempotent)
        tp = time.perf_counter() - tp0
        st = self._save()
        best, out = np.inf, None
        for r in range(self.time_reps):
            if r:
                self._load(st)
            t0 = time.perf_counter()
            out = self._event1(j, T_odom_j, loops, chain, T_odom, anchors, kv, no_prior=no_prior, t=t)
            best = min(best, time.perf_counter() - t0)
        best += tp
        if out is not None:
            out["wall_s"] = best
            if self.walls:
                self.walls[-1] = best
        return out

    def summary(self):
        cl = self.cal_log

        def q(v):
            v = [x for x in v if x is not None and np.isfinite(x)]
            if not v:
                return None
            return {"n": len(v), "p10": float(np.percentile(v, 10)), "p50": float(np.percentile(v, 50)),
                    "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)), "min": float(min(v)),
                    "max": float(max(v))}
        out = {"mode": self.mode, "applies": len(cl), "samples": {c: len(self.smp[c]["t"]) for c in self.CLASSES},
               "sample_fail": self.n_sample_fail, "suff": self.suff, "time_reps": self.time_reps}
        if self.mode == "rel":
            pl = self.pre_log
            out["prefused"] = len(pl)
            out["prefuse_by_state"] = {s_: sum(1 for x in pl if x["st"] == s_)
                                       for s_ in ("cold", "warm", "mixed", "fail")}
            out["prefuse_block_states"] = {s_: sum(1 for x in pl for y in x.get("sts", []) if y == s_)
                                           for s_ in ("cold", "below", "near")}
            for st_ in ("all", "warm", "cold"):
                w = [x for x in pl if x["st"] != "fail" and (st_ == "all" or x["st"] == st_)]
                out["pre_" + st_] = {"n": len(w), "lam_phi": q([x["lam"][0] for x in w]),
                                     "lam_t": q([x["lam"][1] for x in w]), "gain_phi": q([x["gain"][0] for x in w]),
                                     "gain_t": q([x["gain"][1] for x in w]), "trK_phi": q([x["trK"][0] for x in w]),
                                     "trK_t": q([x["trK"][1] for x in w])}
            if self.relcal is not None:
                out["relcal"] = self.relcal.summary()
            gl = self.gate_log
            wl = [x for x in pl if x.get("st") in ("warm", "mixed") and "nis" in x]
            out["gate"] = {"mode": self.gate, "refused": len(gl), "refused_planted": sum(1 for x in gl if x["planted"]),
                           "refused_unplanted": sum(1 for x in gl if not x["planted"]),
                           "refused_kappa_test_also_fails": sum(1 for x in gl if x["nis_k"] > CHI2_6_999),
                           "nis_refused": q([x["nis"] for x in gl]),
                           "nis_warm": q([x["nis"] for x in wl]),
                           "nis_k_warm": q([x["nis_k"] for x in wl]),
                           "frac_warm_over_thr": (float(np.mean([x["nis"] > CHI2_6_999 for x in wl])) if wl else None),
                           "planted_seen": sum(1 for x in pl if x.get("planted")),
                           "planted_warm": sum(1 for x in wl if x.get("planted"))}
        for c in self.CLASSES:
            e = [x for x in cl if x["cls"] == c]
            w = [x for x in e if x["st"] == "warm"]
            out[c] = {"applies": len(e), "warm": len(w), "mu_rho": q([x["mu"][0] for x in w]),
                      "mu_phi": q([x["mu"][1] for x in w]),
                      "gain_rho": q([x["gain_proxy"][0] for x in e]), "gain_phi": q([x["gain_proxy"][1] for x in e]),
                      "gain_rho_frozen": q([x["gain_proxy_frozen"][0] for x in e]),
                      "gain_phi_frozen": q([x["gain_proxy_frozen"][1] for x in e]),
                      "smp_rho": q([s[0] for x in e for s in x["samples"]]),
                      "smp_phi": q([s[1] for x in e for s in x["samples"]])}
        out["solve_ms"] = q([1e3 * w for w in self.walls])
        return out
