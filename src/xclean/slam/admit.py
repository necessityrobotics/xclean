# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.3 ADMISSION: the worker's odometry chain, the causal self-calibration kappa(t), and the four
admission tests of a verified loop (floor, Mahalanobis at the calibrated drift, strict informativeness, PCM >= 1).

Own code; the arithmetic is the design prototype's (an internal script, Chain / Kappa /
WorkerCore.job) re-written for INCREMENTAL use: edges arrive with the keyframe packets (creation-time value) and with
`edge_final` messages (the value at the older keyframe's marginalisation), in keyframe order.  Every decision is a
logged record (closer.py doctrine).  Constants:
  FLOOR = MIN_LOOP_INLIERS 30 (D, carried); SUFFICIENCY 30 (statistic, carried); kappa = max(1, median of the last 30
  long-span samples), cold start = no admission before 30 samples (C); Mahalanobis level 6 F(6, 6n; 0.999) (D level);
  strict informativeness lambda_max <= 1 (D break-even); PCM chi2_6(0.999) against the newest 60 candidates that passed
  tests 1-3 (D level; 60 = declared D cost bound).
"""
from __future__ import annotations

import bisect

import numpy as np

from xclean.slam.geom import compose, so3_log, ChainPrefix, psd_ok, inv

N_KF = 7                       # engine_live N_KF (class D, carried): the active window's keyframes
SUFF = 30                      # SUFFICIENCY (closer.py, carried)
FLOOR = 30                     # MIN_LOOP_INLIERS (core/derive.py, carried)
RELOC = 60                     # MIN_RELOC_INLIERS (core/derive.py, carried)
CHI2_6_MED = 5.348120627447122
CHI2_3_MED = 2.3659738843753377
CHI2_6_999 = 22.457744484825323
ALPHA = 0.001                  # class-D probability level (P_GATE)
PCM_WIN = 60                   # declared D cost bound
_TINY = np.eye(6) * 1e-12
# ---- F2 (P1.3 fix round 2, F2_DESIGN 3.1 / 3.3 / 3.7): estimator-state classes of a window edge (class A: the
# estimator's own state, from the packets' vi flags).  VI > VO; a LINK (kind != "edge") has no class and is never scaled.
CLS_VI, CLS_VO, CLS_LINK = 0, 1, 2
_NBLK = 7                      # per-class prefix blocks: VI (rr, tt, x), VO (rr, tt, x), LINK (whole covariance)


class Chain:
    """The worker's odometry chain over keyframe indices 0..N-1: edge k = KF k -> k+1 (rectified camera frame,
    rotation right, translation additive in frame k, covariance [phi, t]).

    add_edge(k, Z, S, kind)  : the creation-time value (with packet k+1) or a segment LINK (kind != "edge"); S None =
                               UNKNOWN link (the chain splits there); a link value is final at once
    finalise(k, Z, S)        : the marginalisation-time value (edge_final); finals arrive in keyframe order
    rel(a, b)                : chain a -> b (a < b) as known NOW: final values via world-frame prefix sums, the newest
                               non-final edges composed from their creation-time values; None across an unknown link
    Non-positive-definite window blocks are replaced by the last valid block of the same array (the design's
    psd_repair, applied incrementally in the same order, so the repaired values are the design's)."""

    def __init__(self, n_max=1024):
        self.n = 0                                   # number of edges known (creation or link)
        self.ZR = np.zeros((n_max, 3, 3))
        self.Zp = np.zeros((n_max, 3))
        self.S1 = np.zeros((n_max, 6, 6))            # creation-time ("first") values, repaired in order
        self.Sf = np.zeros((n_max, 6, 6))            # final values (repaired), once final
        self.ZRf = np.zeros((n_max, 3, 3))
        self.Zpf = np.zeros((n_max, 3))
        self.final = np.zeros(n_max, bool)
        self.unknown = np.zeros(n_max, bool)
        self.kind = {}
        self.n_final = 0                             # edges 0..n_final-1 are final and in the prefix sums
        self._last_valid_final = None
        self._last_valid_first = None
        self.n_rep = 0
        self.pref = None
        self._pR = [np.eye(3)]
        self._pp = [np.zeros(3)]
        self._pP = [np.zeros((6, 6))]
        self._fin_vals = {}                          # k -> (ZR, Zp, S) final (repaired), waiting for contiguity
        self.ucount = [0]
        self.bridges = {}                            # component id -> (root comp, a in root, b in comp, R, p, S)
        # F2 (F2_DESIGN 3.1): per-class, per-block world prefix sums beside _pP (maintained exactly where _pP is), the
        # prefix poses as arrays (vectorised interval covariances), creation-time class counts (homogeneity) and the
        # head-state gauge-origin arrays (3.2).  Read only by the F2 fusion (head = "mv") and kappa_F.
        self.ecls = np.full(n_max, CLS_VO, np.int8)
        self._cP = np.zeros((n_max + 1, _NBLK, 6, 6))
        self._cR = np.zeros((n_max + 1, 3, 3))
        self._cR[0] = np.eye(3)
        self._cp = np.zeros((n_max + 1, 3))
        self._ccnt = np.zeros((n_max + 1, 3), np.int64)
        self._o_vi = np.zeros(n_max + 1, np.int64)
        self._o_vo = np.zeros(n_max + 1, np.int64)

    def _grow_f2(self, k):
        m = len(self.ecls)
        if k < m:
            return
        m2 = max(2 * m, k + 1)
        e = np.full(m2, CLS_VO, np.int8)
        e[:m] = self.ecls
        self.ecls = e
        for name in ("_cP", "_cR", "_cp", "_ccnt", "_o_vi", "_o_vo"):
            a = getattr(self, name)
            b = np.zeros((m2 + 1,) + a.shape[1:], a.dtype)
            b[:len(a)] = a
            setattr(self, name, b)

    def _grow(self, k):
        self._grow_f2(k)
        if k < len(self.final):
            return
        m = max(2 * len(self.final), k + 1)
        for name, shp in (("ZR", (3, 3)), ("Zp", (3,)), ("S1", (6, 6)), ("Sf", (6, 6)), ("ZRf", (3, 3)),
                          ("Zpf", (3,))):
            a = getattr(self, name)
            b = np.zeros((m,) + shp)
            b[:len(a)] = a
            setattr(self, name, b)
        for name in ("final", "unknown"):
            a = getattr(self, name)
            b = np.zeros(m, bool)
            b[:len(a)] = a
            setattr(self, name, b)

    def _repair(self, S, which):
        """psd_repair for one block, carrying the last valid block of the same array."""
        attr = "_last_valid_final" if which == "final" else "_last_valid_first"
        Sk = 0.5 * (S + S.T)
        if psd_ok(Sk):
            setattr(self, attr, Sk)
            return Sk
        self.n_rep += int(which == "final")
        last = getattr(self, attr)
        return last if last is not None else np.diag(np.abs(np.nan_to_num(np.diag(Sk))) + 1e-12)

    def add_edge(self, k, ZR, Zp, S, kind="edge", cls=None):
        """Edge or link k -> k+1.  Edges must be added in keyframe order (k = n); a missing k (no edge, no link) is
        filled as an UNKNOWN gap.  cls (F2): CLS_VI / CLS_VO of a window edge (None = VO, the weakest); a link is
        CLS_LINK whatever is passed."""
        while self.n < k:                            # a gap with no edge and no link = unknown
            self.add_edge(self.n, np.eye(3), np.zeros(3), None, kind="missing")
        if k != self.n:
            raise ValueError("chain edges must arrive in order: got %d, have %d" % (k, self.n))
        self._grow(k)
        self.ZR[k], self.Zp[k] = ZR, Zp
        # F2 bookkeeping (class, class counts, gauge-origin arrays); no frozen quantity reads it
        ec = CLS_LINK if kind != "edge" else (CLS_VO if cls is None else int(cls))
        self.ecls[k] = ec
        self._ccnt[k + 1] = self._ccnt[k]
        self._ccnt[k + 1, ec] += 1
        known_link = kind != "edge" and S is not None
        self._o_vi[k + 1] = self._o_vi[k] if (ec == CLS_VI or known_link) else k + 1
        self._o_vo[k + 1] = self._o_vo[k] if (kind == "edge" or known_link) else k + 1
        self.n += 1
        if kind != "edge":
            self.kind[k] = kind
        if S is None:
            self.unknown[k] = True
            self.S1[k] = self._repair(_TINY, "first")   # the design's quirk kept: 1e-12 I is "valid" for the carry
            self._fin_vals[k] = (ZR, Zp, _TINY)
            self.ucount.append(self.ucount[-1] + 1)
        else:
            self.ucount.append(self.ucount[-1])
            S = np.asarray(S, float)
            if kind == "edge":
                self.S1[k] = self._repair(S, "first")
            else:                                    # a link is final at once (its first value = its final value)
                Sv = S if psd_ok(S) else (self._last_valid_final if self._last_valid_final is not None else S)
                self.S1[k] = self._repair(Sv, "first")
                self._fin_vals[k] = (ZR, Zp, S)
        self._advance()

    def finalise(self, k, ZR, Zp, S):
        if k >= self.n or k in self._fin_vals or self.final[k]:
            return
        if self.unknown[k] or k in self.kind:
            return
        self._fin_vals[k] = (np.asarray(ZR, float), np.asarray(Zp, float), np.asarray(S, float))
        self._advance()

    def _advance(self):
        """Final values enter the prefix sums in keyframe order; the final array's PSD repair runs here, in that order
        (the design's psd_repair over the final array)."""
        while self.n_final in self._fin_vals:
            k = self.n_final
            ZR, Zp, S = self._fin_vals.pop(k)
            S = self._repair(S, "final")
            self.ZRf[k], self.Zpf[k], self.Sf[k] = ZR, Zp, S
            R0, p0, P0 = self._pR[-1], self._pp[-1], self._pP[-1]
            R1 = R0 @ ZR
            p1 = p0 + R0 @ Zp
            M = np.zeros((6, 6))
            M[:3, :3] = R1
            M[3:, :3] = _skew(p1) @ R1
            M[3:, 3:] = R0
            self._pR.append(R1)
            self._pp.append(p1)
            self._pP.append(P0 + M @ S @ M.T)
            # F2 (3.1): the same M, the same repaired S, split by class and block (S = S_rr + S_tt + S_x)
            self._cP[k + 1] = self._cP[k]
            self._cR[k + 1] = R1
            self._cp[k + 1] = p1
            ec = int(self.ecls[k])
            if ec == CLS_LINK:
                self._cP[k + 1, 6] += M @ S @ M.T
            else:
                Sb = np.zeros((3, 6, 6))
                Sb[0, :3, :3] = S[:3, :3]
                Sb[1, 3:, 3:] = S[3:, 3:]
                Sb[2, :3, 3:] = S[:3, 3:]
                Sb[2, 3:, :3] = S[3:, :3]
                self._cP[k + 1, 3 * ec:3 * ec + 3] += M[None] @ Sb @ M.T[None]
            self.final[k] = True
            self.n_final += 1

    # ---- components (maximal runs between UNKNOWN links) and merge BRIDGES (merge.py R1) ---------------------------
    def comp(self, k):
        """component id of keyframe k (number of unknown links before it)."""
        return self.ucount[min(k, self.n)]

    def root(self, k):
        c = self.comp(k)
        b = self.bridges.get(c)
        return c if b is None else b[0]

    def known(self, a, b):
        lo, hi = min(a, b), max(a, b)
        if self.ucount[hi] == self.ucount[lo]:
            return True
        return self.root(lo) == self.root(hi)

    def add_bridge(self, comp_new, a, b, ZR, Zp, S):
        """Merge: component comp_new (b in it) is placed relative to keyframe a of an OLDER component by the relative pose
        a -> b (ZR, Zp, S [phi, t]).  Stored relative to a's ROOT component (composed through a's own bridge), so every
        relative pose across merged components is one hop: root-internal chain + at most two bridges."""
        ca = self.comp(a)
        br = self.bridges.get(ca)
        if br is not None:                           # a's component is itself merged: compose through its bridge
            r0, a0, b0, R0, p0, S0 = br
            if b0 != a:
                g = self.rel_within(b0, a)
                R0, p0, S0 = compose((R0, p0, S0), g)
            R1, p1, S1 = compose((R0, p0, S0), (ZR, Zp, S))
            self.bridges[comp_new] = (r0, a0, b, R1, p1, S1)
        else:
            self.bridges[comp_new] = (ca, a, b, np.asarray(ZR, float), np.asarray(Zp, float), np.asarray(S, float))
        # components already bridged to comp_new now hang off the new root
        for c, brc in list(self.bridges.items()):
            if c != comp_new and brc[0] == comp_new:
                r0, a0, b0, R0, p0, S0 = self.bridges[comp_new]
                _r, ac, bc, Rc, pc, Sc = brc
                g = self.rel_within(b0, ac) if b0 != ac else (np.eye(3), np.zeros(3), np.zeros((6, 6)))
                R1, p1, S1 = compose(compose((R0, p0, S0), g), (Rc, pc, Sc))
                self.bridges[c] = (r0, a0, bc, R1, p1, S1)

    def remove_bridge(self, comp_new):
        self.bridges.pop(comp_new, None)

    def rel_within(self, u, v):
        """relative pose u -> v inside ONE component, any order (identity if equal; first-order inverse if u > v)."""
        if u == v:
            return (np.eye(3), np.zeros(3), np.zeros((6, 6)))
        if u < v:
            return self._rel_fwd(u, v)
        return rel_inverse(self._rel_fwd(v, u))

    def rel_any(self, u, v):
        """relative pose u -> v for any order, across merged components (None if not connected)."""
        if u == v:
            return (np.eye(3), np.zeros(3), np.zeros((6, 6)))
        if self.ucount[min(max(u, v), self.n)] == self.ucount[min(u, v)] and max(u, v) <= self.n:
            return self.rel_within(u, v)
        if max(u, v) > self.n or self.root(u) != self.root(v):
            return None
        # root-frame hop: u -> (its bridge b_u, back to a_u in the root) -> root chain -> a_v -> bridge -> v
        def to_root(x):                              # rel(anchor_in_root -> x) and the anchor
            br = self.bridges.get(self.comp(x))
            if br is None:
                return x, (np.eye(3), np.zeros(3), np.zeros((6, 6)))
            _r, a, b, R, p, S = br
            g = self.rel_within(b, x)
            return a, compose((R, p, S), g)
        au, gu = to_root(u)
        av, gv = to_root(v)
        mid = self.rel_within(au, av)
        return compose(compose(rel_inverse(gu), mid), gv)

    def set_link(self, k, ZR, Zp, S, kind):
        """A MERGE (merge.py) gives an unknown link k a value and a covariance (or a revert takes it back to unknown):
        the unknown counts and the prefix sums are rebuilt from k with the stored final values (same operation order)."""
        ZR = np.asarray(ZR, float)
        Zp = np.asarray(Zp, float)
        self.ZR[k], self.Zp[k] = ZR, Zp
        self.unknown[k] = S is None
        self.kind[k] = kind
        Sf = _TINY if S is None else 0.5 * (np.asarray(S, float) + np.asarray(S, float).T)
        self.S1[k] = Sf
        uc = [0]
        for i in range(self.n):
            uc.append(uc[-1] + int(self.unknown[i]))
        self.ucount = uc
        for i in range(k, self.n):                   # F2: the gauge-origin arrays from k (a link changed state)
            kl = self.ecls[i] == CLS_LINK and not self.unknown[i]
            self._o_vi[i + 1] = self._o_vi[i] if (self.ecls[i] == CLS_VI or kl) else i + 1
            self._o_vo[i + 1] = self._o_vo[i] if (self.ecls[i] != CLS_LINK or kl) else i + 1
        if k < self.n_final:
            stored = [(i, self.ZRf[i].copy(), self.Zpf[i].copy(), self.Sf[i].copy()) for i in range(k, self.n_final)]
            self._pR, self._pp, self._pP = self._pR[:k + 1], self._pp[:k + 1], self._pP[:k + 1]
            self.n_final = k
            for (i, R_, p_, S_) in stored:
                self.final[i] = False
                self._fin_vals[i] = (ZR, Zp, Sf) if i == k else (R_, p_, S_)
            self._advance()
        else:
            self._fin_vals[k] = (ZR, Zp, Sf)
            self._advance()

    def _prefix_rel(self, a, b):
        Ra, pa, Rb, pb = self._pR[a], self._pp[a], self._pR[b], self._pp[b]
        ZR = Ra.T @ Rb
        Zp = Ra.T @ (pb - pa)
        N = np.zeros((6, 6))
        N[:3, :3] = Rb.T
        N[3:, :3] = -Ra.T @ _skew(pb)
        N[3:, 3:] = Ra.T
        S = N @ (self._pP[b] - self._pP[a]) @ N.T
        return ZR, Zp, 0.5 * (S + S.T)

    def _rel_fwd(self, a, b):
        m = min(max(self.n_final, a), b)
        if m >= b:
            return self._prefix_rel(a, b)
        out = self._prefix_rel(a, m) if m > a else (np.eye(3), np.zeros(3), np.zeros((6, 6)))
        for k in range(m, b):
            out = compose(out, (self.ZR[k], self.Zp[k], self.S1[k]))
        return out

    def rel(self, a, b):
        """chain a -> b (a < b) as the worker knows it now: final values via world-frame prefix sums, the newest
        non-final edges composed from their creation-time values; across merged components through the bridge; None
        across an unknown (unmerged) link or past the newest known edge."""
        if a >= b:
            raise ValueError("rel needs a < b")
        if b > self.n:
            return None
        if self.ucount[b] == self.ucount[a]:
            return self._rel_fwd(a, b)
        if self.root(a) != self.root(b):
            return None
        return self.rel_any(a, b)

    # ---- F2 (F2_DESIGN 3.1 / 3.2): per-class block-scaled covariances, classes, the head-state gauge origin --------
    @staticmethod
    def kv_weights(kv, lw=1.0):
        """kv = ((kr_VI, kt_VI), (kr_VO, kt_VO)) -> the 7 prefix-block weights (block scaling of an edge's [phi, t]
        covariance is linear in (kr, kt, sqrt(kr kt)): exact)."""
        (a, b), (c, d) = kv
        return np.array([a, b, np.sqrt(a * b), c, d, np.sqrt(c * d), lw], float)

    def p_kappa_world(self, a, b, kv, lw=1.0):
        """World-frame (about the chain origin) covariance over the FINAL edges a..b-1 (b <= n_final) with per-class
        block scales: sum_s [kr Pi^s_rr + kt Pi^s_tt + sqrt(kr kt) Pi^s_x] + lw Pi_link.  kv all 1, lw 1 = the Chain's
        own _pP[b] - _pP[a]."""
        return np.tensordot(self.kv_weights(kv, lw), self._cP[b] - self._cP[a], axes=1)

    @staticmethod
    def _scale_edge(S, ec, kv, lw):
        if ec == CLS_LINK:
            return lw * S
        kr, kt = kv[ec]
        d = np.r_[[np.sqrt(kr)] * 3, [np.sqrt(kt)] * 3]
        return S * np.outer(d, d)

    def p_kappa_rel(self, a, b, kv, lw=1.0):
        """Chain a -> b (a < b) as rel() but with per-class block scales: the final part through the per-class prefix
        sums, the newest non-final edges composed from their creation-time values, each scaled by its own class (links
        by lw).  -> (ZR, Zp, S [phi, t]).  kv all 1, lw 1 = rel(a, b) inside one component (test A3-a)."""
        m = min(max(self.n_final, a), b)
        if m > a:
            Ra, pa, Rb, pb = self._cR[a], self._cp[a], self._cR[m], self._cp[m]
            N = np.zeros((6, 6))
            N[:3, :3] = Rb.T
            N[3:, :3] = -Ra.T @ _skew(pb)
            N[3:, 3:] = Ra.T
            S = N @ self.p_kappa_world(a, m, kv, lw) @ N.T
            out = (Ra.T @ Rb, Ra.T @ (pb - pa), 0.5 * (S + S.T))
        else:
            out = (np.eye(3), np.zeros(3), np.zeros((6, 6)))
        for k in range(m, b):
            out = compose(out, (self.ZR[k], self.Zp[k], self._scale_edge(self.S1[k], int(self.ecls[k]), kv, lw)))
        return out

    def p_kappa_rel_many(self, A, B, kv, lw=1.0):
        """p_kappa_rel over many intervals (A[i] < B[i]), vectorised over the final ones -> (ZR (n,3,3), S (n,6,6))."""
        A = np.asarray(A, np.int64)
        B = np.asarray(B, np.int64)
        n = len(A)
        ZR = np.zeros((n, 3, 3))
        S = np.zeros((n, 6, 6))
        fin = B <= self.n_final
        if fin.any():
            a, b = A[fin], B[fin]
            Pw = np.tensordot(self._cP[b] - self._cP[a], self.kv_weights(kv, lw), axes=([1], [0]))
            Ra, Rb, pb = self._cR[a], self._cR[b], self._cp[b]
            RaT = np.transpose(Ra, (0, 2, 1))
            N = np.zeros((len(a), 6, 6))
            N[:, :3, :3] = np.transpose(Rb, (0, 2, 1))
            N[:, 3:, :3] = -RaT @ _skew_n(pb)
            N[:, 3:, 3:] = RaT
            Sf = N @ Pw @ np.transpose(N, (0, 2, 1))
            S[fin] = 0.5 * (Sf + np.transpose(Sf, (0, 2, 1)))
            ZR[fin] = RaT @ Rb
        for i in np.nonzero(~fin)[0]:
            ZR[i], _p, S[i] = self.p_kappa_rel(int(A[i]), int(B[i]), kv, lw)
        return ZR, S

    def class_count(self, a, b):
        """creation-time class counts (VI, VO, LINK) of the edges a..b-1."""
        return self._ccnt[b] - self._ccnt[a]

    def homogeneous(self, a, b):
        """the class of the chain a -> b if every edge in it is a window edge of ONE class (no link), else None."""
        if not 0 <= a < b <= self.n:
            return None
        c = self._ccnt[b] - self._ccnt[a]
        if c[CLS_VI] == b - a:
            return CLS_VI
        if c[CLS_VO] == b - a:
            return CLS_VO
        return None

    def n_unknown(self, a, b):
        """unknown links among the edges a..b-1."""
        return self.ucount[min(b, self.n)] - self.ucount[min(a, self.n)]

    def bridged(self, k):
        """the unknown link k -> k+1 is bridged by an R1 merge (both sides share a root component)."""
        return bool(self.unknown[k]) and k + 1 <= self.n and self.root(k) == self.root(k + 1)

    def n_unbridged(self, a, b):
        """unknown links among the edges a..b-1 that no R1 merge has bridged."""
        if self.n_unknown(a, b) == 0:
            return 0
        ks = np.nonzero(self.unknown[a:min(b, self.n)])[0] + a
        return int(sum(1 for k in ks if not self.bridged(int(k))))

    def unknown_links(self, a, b):
        """the unknown link edges among a..b-1."""
        if self.n_unknown(a, b) == 0:
            return []
        return [int(k) for k in np.nonzero(self.unknown[a:min(b, self.n)])[0] + a]

    def origin(self, j, head_cls):
        """F2_DESIGN 3.2: the head-state gauge origin o(j), the earliest keyframe such that the chain o -> j contains no
        window edge of a class weaker than the head's and no unknown link (an R1-bridged unknown link counts as a known
        link: its offset is carried by the merge epoch's lineage)."""
        arr = self._o_vi if head_cls == CLS_VI else self._o_vo
        o = int(arr[min(j, self.n)])
        while o > 0 and self.bridged(o - 1):
            o = int(arr[o - 1])
        return o


def _skew_n(v):
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -v[..., 2], v[..., 1]
    K[..., 1, 0], K[..., 1, 2] = v[..., 2], -v[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -v[..., 1], v[..., 0]
    return K


def rel_inverse(g):
    """(R, p, S[phi, t]) of Z -> Z^-1 to first order (rotation right, translation additive in the left frame):
    phi' = -R phi, dt' = -[R^T p]x phi - R^T dt."""
    R, p, S = g
    Ri = R.T
    pi = -Ri @ p
    J = np.zeros((6, 6))
    J[:3, :3] = -R
    J[3:, :3] = -_skew(Ri @ p)
    J[3:, 3:] = -Ri
    return Ri, pi, J @ S @ J.T


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def lcb_rank(n, alpha=ALPHA):
    """Largest r with P(Bin(n, 1/2) <= r - 1) <= alpha (distribution-free lower confidence bound of the median)."""
    from scipy.stats import binom
    r = 0
    for rr in range(1, n + 1):
        if binom.cdf(rr - 1, n, 0.5) <= alpha:
            r = rr
        else:
            break
    return r


class Kappa:
    """Causal self-calibration of the chain covariance.  mode "scalar" (the design's decision) or
    "block2" (per rotation / translation block, the measured arm); cold start "median30" (decision) or "lcb" (arm)."""

    def __init__(self, mode="scalar", cold="median30"):
        self.mode, self.cold = mode, cold
        self.samples = []
        self.ts = []
        self._lcb = {}

    @staticmethod
    def _solve(r, Sc, Sl, med):
        f = lambda k: float(r @ np.linalg.solve(k * Sc + Sl, r))     # noqa: E731
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

    def sample(self, r, Sc, Sl):
        if self.mode == "scalar":
            k = self._solve(r, Sc, Sl, CHI2_6_MED)
            return (k, k)
        kr = self._solve(r[:3], Sc[:3, :3], Sl[:3, :3], CHI2_3_MED)
        kt = self._solve(r[3:], Sc[3:, 3:], Sl[3:, 3:], CHI2_3_MED)
        return (kr, kt)

    def add(self, t, ks):
        self.samples.append((t,) + tuple(ks))
        self.ts.append(t)

    def value(self, t):
        """-> (kr, kt, n) or None (cold).  Only samples decided strictly before t."""
        n = bisect.bisect_left(self.ts, t)
        if n >= SUFF:
            P_ = np.array(self.samples[n - SUFF:n])
            return max(1.0, float(np.median(P_[:, 1]))), max(1.0, float(np.median(P_[:, 2]))), SUFF
        if self.cold == "lcb":
            if n not in self._lcb:
                self._lcb[n] = lcb_rank(n)
            r = self._lcb[n]
            if r >= 1:
                P_ = np.array(self.samples[:n])
                kr = np.sort(P_[:, 1])[r - 1]
                kt = np.sort(P_[:, 2])[r - 1]
                return max(1.0, float(kr)), max(1.0, float(kt)), n
        return None

    @staticmethod
    def scale(S, kv):
        d = np.r_[[np.sqrt(kv[0])] * 3, [np.sqrt(kv[1])] * 3]
        return S * np.outer(d, d)


class KappaF:
    """F2 MV-2 (F2_DESIGN 3.3): kappa_F(class, block, span, t), the FUSION calibration of the chain (never admission's).

    Samples: made attempts of the GATED population (every one while kappa(t) is cold; once warm only those passing the
    frozen Mahalanobis test at kappa(t)) whose chain c -> q is class-homogeneous; per block the frozen per-sample rule
    Kappa._solve on the LIVE-ODOM residual (FO-3) at the median chi2_3; span coordinate u = logdet of the raw block
    covariance.  Committed with the job's finish time (only later jobs use them).
    Value: fewer than SUFF class samples decided before t -> COLD (shared drift: 1 = the raw chain; anchor-private
    drift: None = no information); else the SUFF samples nearest in u (ties to the newest): u below their smallest u
    -> 1 (below the calibrated domain); else max(1, the lcb_rank(SUFF, ALPHA)-th smallest of them).
    Constants: SUFF, ALPHA, lcb_rank, CHI2_3_MED, the floor 1 (all carried)."""

    def __init__(self):
        self.pending = []
        self.us = {(s, b): [] for s in (CLS_VI, CLS_VO) for b in (0, 1)}     # sorted u
        self.rec = {(s, b): [] for s in (CLS_VI, CLS_VO) for b in (0, 1)}    # parallel (u, t, order, kappa)
        self.ts = {CLS_VI: [], CLS_VO: []}                                    # commit times per class (sorted)
        self.n = 0
        self.r_lcb = lcb_rank(SUFF, ALPHA)

    def add(self, cls, kr, kt, ur, ut):
        self.pending.append((int(cls), float(kr), float(kt), float(ur), float(ut)))

    def commit(self, t):
        for (s, kr, kt, ur, ut) in self.pending:
            for b, (k, u) in enumerate(((kr, ur), (kt, ut))):
                i = bisect.bisect_right(self.us[(s, b)], u)
                self.us[(s, b)].insert(i, u)
                self.rec[(s, b)].insert(i, (u, float(t), self.n, k))
            bisect.insort(self.ts[s], float(t))
            self.n += 1
        self.pending = []

    def n_before(self, s, t):
        return bisect.bisect_left(self.ts[s], t)

    def value(self, s, b, u, t, private=False):
        """-> (kappa or None, state) with state cold | below | lcb."""
        if self.n_before(s, t) < SUFF:
            return (None if private else 1.0), "cold"
        us, rec = self.us[(s, b)], self.rec[(s, b)]
        pos = bisect.bisect_left(us, u)
        lo, hi = pos - 1, pos
        sel = []
        while len(sel) < SUFF and (lo >= 0 or hi < len(rec)):
            while lo >= 0 and rec[lo][1] >= t:
                lo -= 1
            while hi < len(rec) and rec[hi][1] >= t:
                hi += 1
            cl = rec[lo] if lo >= 0 else None
            ch = rec[hi] if hi < len(rec) else None
            if cl is None and ch is None:
                break
            if ch is None or (cl is not None and (u - cl[0] < ch[0] - u or (u - cl[0] == ch[0] - u and cl[2] > ch[2]))):
                sel.append(cl)
                lo -= 1
            else:
                sel.append(ch)
                hi += 1
        if len(sel) < SUFF:
            return (None if private else 1.0), "cold"
        if u < min(x[0] for x in sel):
            return 1.0, "below"
        ks = sorted(x[3] for x in sel)
        return max(1.0, float(ks[self.r_lcb - 1])), "lcb"


def loop_cov(H, chi2, n_info):
    """Sigma_loop = s^2 H^-1, s^2 = max(1, chi2 / dof), dof = 2 n - 6."""
    dof = max(2 * n_info - 6, 1)
    s2 = max(1.0, float(chi2) / dof)
    Sl = s2 * np.linalg.inv(H)
    return 0.5 * (Sl + Sl.T)


def residual(Zch, Zp):
    """r = Z_chain [-] Z_pnp ([phi, t])."""
    Rc, pc = Zch[0], Zch[1]
    return np.concatenate([so3_log(Zp[:3, :3].T @ Rc), pc - Zp[:3, 3]])


class CycleKappa:
    """R1's chain scale while kappa(t) is COLD (fix round 2026-10-26, verifier major 2 / risk R17).

    kappa(t) calibrates the chain covariance from cycles closed by ONE verified loop (chain c -> q vs the PnP pose); it
    needs c and q in one known chain, so a run fragmented by unknown links before 30 such samples exist never calibrates
    and R1's pairwise cycle test ran on the raw (19-1 000x over-confident) chain.  A PAIR of verified loops (q1, c1),
    (q2, c2) closes the cycle Z1 Zodo(q1 -> q2) = Zodo(c1 -> c2) Z2 with no unknown link in it whenever c1, c2 are
    connected and q1, q2 are connected (the cross loops of a new segment into an old one are such pairs).  Its residual e
    against Sl1 + Sl2 + k (Sc_q + Sc_c) gives a sample k by kappa's own rule (Kappa._solve: e whitened = the median
    chi2_6), taken only when the chain part dominates (logdet, as kappa's samples).  This is the SAME cycle statistic R1
    tests (merge.pcm_pair arithmetic), so the scale is calibrated on what it gates.
    One sample per verified attempt = the MEDIAN of its pair samples against the newest PCM_WIN verified attempts (a wrong
    partner cannot dominate it); kappa_cyc(t) = max(1, median of the last SUFF attempt samples decided before t), cold
    until SUFF (the same sufficiency as kappa).  Collected only while an unmerged link exists and kappa(t) is cold (the
    worker sets `on`), so a run without links is unchanged.  Constants: none new (SUFF, PCM_WIN, CHI2_6_MED carried)."""

    def __init__(self, chain, win=PCM_WIN):
        self.chain = chain
        self.win = win
        self.ver = []                          # verified attempts: dicts q, c, Z, Sl
        self.samples = []
        self.ts = []
        self.pending = []
        self.n_pairs = 0

    def observe(self, q, c, Z, Sl):
        from xclean.slam.geom import se3
        A = {"q": q, "c": c, "Z": Z, "Sl": Sl}
        ks = []
        for B in self.ver[-self.win:]:
            if B["q"] == q and B["c"] == c:
                continue
            g1 = self.chain.rel_any(B["q"], q)
            g2 = self.chain.rel_any(B["c"], c)
            if g1 is None or g2 is None:
                continue
            A_ = B["Z"] @ se3(g1[0], g1[1])
            B_ = se3(g2[0], g2[1]) @ Z
            e = np.concatenate([so3_log(A_[:3, :3].T @ B_[:3, :3]), B_[:3, 3] - A_[:3, 3]])
            Sc = g1[2] + g2[2]
            Sll = B["Sl"] + Sl
            try:
                if np.linalg.slogdet(Sc)[1] <= np.linalg.slogdet(Sll)[1]:
                    continue
                ks.append(Kappa._solve(e, Sc, Sll, CHI2_6_MED))
            except np.linalg.LinAlgError:
                continue
        self.n_pairs += len(ks)
        if ks:
            self.pending.append(float(np.median(ks)))
        self.ver.append(A)
        if len(self.ver) > 2 * self.win:
            self.ver = self.ver[-self.win:]

    def commit(self, t):
        for k in self.pending:
            self.samples.append(k)
            self.ts.append(t)
        self.pending = []

    def value(self, t):
        """-> (k, k, SUFF) or None (cold); only samples decided strictly before t."""
        n = bisect.bisect_left(self.ts, t)
        if n >= SUFF:
            k = max(1.0, float(np.median(self.samples[n - SUFF:n])))
            return k, k, SUFF
        return None


class Admission:
    """The four admission tests + kappa(t) + PCM state.  Pure logic on a Chain; the clock is injected (t = the deciding
    job's modelled / wall finish time; kappa, PCM and eligibility see only attempts decided before it).

    feed (fix round 2026-10-26, verifier major 1 (b) / risk R16): "gated" (DEFAULT) = once kappa(t) is calibrated, an
    attempt's kappa sample is queued only if the attempt PASSES the Mahalanobis test at the current kappa (a sustained,
    self-consistent wrong episode is rejected by that test and can no longer widen its own gate); during the cold start
    every long-span sample is queued (median-of-30 robustness, as designed).  "all" = the design / build 1 (every made
    long-span attempt feeds kappa, including the ones Mahalanobis rejects)."""

    def __init__(self, chain, kappa_mode="scalar", cold="median30", pcm_win=PCM_WIN, prefilter=True, feed="gated"):
        from scipy.stats import f as fdist
        if feed not in ("gated", "all"):
            raise ValueError("Admission feed must be gated | all, got %r" % (feed,))
        self.chain = chain
        self.kap = Kappa(kappa_mode, cold)
        self.feed = feed
        self.cyc = CycleKappa(chain)
        self.cyc_on = False                    # the worker turns it on while an unmerged link exists and kappa is cold
        self.n_feed_gated = 0
        self.pcm_win = pcm_win
        self.prefilter = prefilter
        self.pcm_list = []                     # made attempts passing floor + Mahalanobis + informativeness
        self.sl_hist = []                      # logdet Sigma_loop of made attempts with >= 30 inliers (eligibility)
        self._F = {}
        self._fdist = fdist
        self.pending_samples = []
        # F2 (F2_DESIGN 3.3 / 3.7): the fusion calibration kappa_F (fed from the SAME gated decision; read only by the
        # F2 head), the PCM pool's cold start (MV-4, behind pcm_cold), and the replay-only informativeness arm
        self.kf = KappaF()
        self.T_odom_fn = None                  # keyframe -> LIVE-ODOM pose (the worker's db.hdr): the FO-3 residual
        self.pcm_cold = "drop"                 # "drop" (frozen) | "rejudge" (MV-4)
        self.cold_pool = []
        self.n_cold_pooled = self.n_cold_rejudged = self.n_cold_joined = 0
        self.inf_raw = False                   # REPLAY-ONLY perturbation arm (--perturb infraw): test 3 at kappa = 1
        self.n_f2_err = 0

    def _f2_sample(self, c, q, Sc, Sl, Z):
        """kappa_F sample of one gated attempt (F2_DESIGN 3.3): class-homogeneous chain c -> q, per block the frozen
        per-sample rule on the LIVE-ODOM residual."""
        if self.T_odom_fn is None:
            return
        try:
            s = self.chain.homogeneous(c, q)
            if s is None:
                return
            To = inv(self.T_odom_fn(c)) @ self.T_odom_fn(q)
            ro = residual((To[:3, :3], To[:3, 3]), Z)
            kr = Kappa._solve(ro[:3], Sc[:3, :3], Sl[:3, :3], CHI2_3_MED)
            kt = Kappa._solve(ro[3:], Sc[3:, 3:], Sl[3:, 3:], CHI2_3_MED)
            ur = float(np.linalg.slogdet(Sc[:3, :3])[1])
            ut = float(np.linalg.slogdet(Sc[3:, 3:])[1])
            if np.isfinite(ur) and np.isfinite(ut):
                self.kf.add(s, kr, kt, ur, ut)
        except (np.linalg.LinAlgError, KeyError, ValueError, FloatingPointError):
            self.n_f2_err += 1

    def rejudge_cold(self, t_fin):
        """MV-4 (FO-6): at the first job whose kappa(t) is warm (and every later one, for anything still pooled), every
        attempt pooled while kappa was cold is judged ONCE at that job's kappa(t) with tests 2-3 (Mahalanobis, strict
        informativeness, as decide computes them); passing ones join pcm_list in their original decision order, as PCM
        PARTNERS ONLY (never admitted, never kappa samples again)."""
        if self.pcm_cold != "rejudge" or not self.cold_pool:
            return 0
        kv = self.kap.value(t_fin)
        if kv is None:
            return 0
        n0 = len(self.pcm_list)
        for d in self.cold_pool:
            self.n_cold_rejudged += 1
            try:
                Scs = Kappa.scale(d["Sc"], kv)
                m2 = float(d["r"] @ np.linalg.solve(Scs + d["Sl"], d["r"]))
                if m2 > self.thr6(kv[2]):
                    continue
                Lc = np.linalg.cholesky((d["Sc"] if self.inf_raw else Scs) + 1e-18 * np.eye(6))
                Li = np.linalg.inv(Lc)
                if float(np.linalg.eigvalsh(Li @ d["Sl"] @ Li.T).max()) > 1.0:
                    continue
            except np.linalg.LinAlgError:
                continue
            self.pcm_list.append({"k": d["k"], "q": d["q"], "c": d["c"], "Z": d["Z"], "Sl": d["Sl"], "kv": kv,
                                  "cold": True})
        self.cold_pool = []
        self.n_cold_joined += len(self.pcm_list) - n0
        return len(self.pcm_list) - n0

    def thr6(self, n):
        if n not in self._F:
            self._F[n] = 6.0 * float(self._fdist.ppf(1 - ALPHA, 6, 6 * n))
        return self._F[n]

    def eligible(self, c, j, t):
        """Eligibility before verification: a candidate is verified only if a loop there COULD be
        informative: logdet(kappa Sigma_chain(c -> j)) > median logdet(Sigma_loop) of the last 30 verified attempts.
        Cold start: every above-threshold candidate."""
        if not self.prefilter:
            return True
        kv = self.kap.value(t)
        if kv is None or len(self.sl_hist) < SUFF or not c < j:
            return True
        g = self.chain.rel(c, j)
        if g is None:
            return True
        ldc = float(np.linalg.slogdet(Kappa.scale(g[2], kv) + 1e-300 * np.eye(6))[1])
        return ldc > float(np.median(self.sl_hist[-SUFF:]))

    def decide(self, att, t_fin):
        """One MADE attempt (dict with q, c, inliers, Z (4x4) or None, H, chi2, n_info) decided at t_fin.  Updates the
        record in place: stage = floor | unknown_link | cold | mahal | informative | pcm | admitted, and the test values.
        kappa samples are queued and committed by commit(t_fin) after the job (only later jobs use them)."""
        rec = att
        rec["stage"] = "floor"
        j, c = att["q"], att["c"]
        if att["inliers"] < FLOOR or att.get("Z") is None or not np.isfinite(att["Z"][0, 0]):
            return rec
        try:
            Sl = loop_cov(att["H"], att["chi2"], att["n_info"])
        except np.linalg.LinAlgError:
            return rec
        rec["Sl"] = Sl
        self.sl_hist.append(float(np.linalg.slogdet(Sl)[1]))
        if self.cyc_on:
            self.cyc.observe(j, c, att["Z"], Sl)
        g = self.chain.rel(c, j) if c < j else None
        if g is None:
            rec["stage"] = "unknown_link"
            return rec
        r = residual(g, att["Z"])
        Sc = g[2]
        rec["r"] = r
        kv = self.kap.value(t_fin)
        smp = None
        if np.linalg.slogdet(Sc)[1] > np.linalg.slogdet(Sl)[1]:
            try:
                smp = self.kap.sample(r, Sc, Sl)
            except np.linalg.LinAlgError:
                smp = None
        if smp is not None and (kv is None or self.feed == "all"):
            self.pending_samples.append(smp)             # cold start (or the design's "all"): queued unconditionally
            smp = None
        rec["stage"] = "cold"
        if kv is None:
            self._f2_sample(c, j, Sc, Sl, att["Z"])      # F2: the cold phase feeds kappa_F unconditionally (as kappa)
            if self.pcm_cold == "rejudge":               # MV-4: pooled for the first warm job's re-judging
                self.cold_pool.append({"k": att.get("k"), "q": j, "c": c, "Z": att["Z"], "Sl": Sl, "r": r, "Sc": Sc,
                                       "t": t_fin})
                self.cold_pool = self.cold_pool[-self.pcm_win:] if self.pcm_win > 0 else self.cold_pool
                self.n_cold_pooled += 1
            return rec
        Scs = Kappa.scale(Sc, kv)
        try:
            m2 = float(r @ np.linalg.solve(Scs + Sl, r))
        except np.linalg.LinAlgError:
            return rec
        thr = self.thr6(kv[2])
        rec.update(m2=m2, thr=thr, kr=kv[0], kt=kv[1], kn=kv[2])
        if smp is not None:                              # "gated": only an attempt the current gate accepts calibrates it
            if m2 <= thr:
                self.pending_samples.append(smp)
            else:
                self.n_feed_gated += 1
                rec["kappa_sample_gated"] = True
        if m2 > thr:
            rec["stage"] = "mahal"
            return rec
        self._f2_sample(c, j, Sc, Sl, att["Z"])          # F2: warm phase, only attempts the Mahalanobis gate accepts
        try:
            Lc = np.linalg.cholesky((Sc if self.inf_raw else Scs) + 1e-18 * np.eye(6))
            Li = np.linalg.inv(Lc)
            lam = float(np.linalg.eigvalsh(Li @ Sl @ Li.T).max())
        except np.linalg.LinAlgError:
            return rec
        rec["lam"] = lam
        if lam > 1.0:
            rec["stage"] = "informative"
            return rec
        d = {"k": att.get("k"), "q": j, "c": c, "Z": att["Z"], "Sl": Sl, "kv": kv}
        win = self.pcm_list if self.pcm_win <= 0 else self.pcm_list[-self.pcm_win:]
        npcm = 0
        for d2 in win:
            if d2["q"] >= j:
                continue
            if npcm >= 2:
                break
            if self.pcm(d2, d, kv):
                npcm += 1
        self.pcm_list.append(d)
        rec["pcm"] = npcm
        if npcm < 1:
            rec["stage"] = "pcm"
            return rec
        rec["stage"] = "admitted"
        rec["kv"] = kv
        return rec

    def commit(self, t_fin):
        for s in self.pending_samples:
            self.kap.add(t_fin, s)
        self.pending_samples = []
        self.cyc.commit(t_fin)
        self.kf.commit(t_fin)

    def r1_scale(self, t):
        """R1's chain scale at t: kappa(t) when calibrated, else kappa_cyc(t) (CycleKappa), else None (no R1 decision:
        the cross loops are pooled until a scale exists).  -> (kv, source)."""
        kv = self.kap.value(t)
        if kv is not None:
            return kv, "kappa"
        kc = self.cyc.value(t)
        if kc is not None:
            return kc, "cycle"
        return None, None

    def pcm(self, d2, d, kv):
        """Cycle consistency of two loops (q', c') and (q, c): Z' Z_odo(q' -> q) vs Z_odo(c' -> c) Z at chi2_6(0.999)."""
        return pcm_test(self.chain, d2, d, kv)


def pcm_test(chain, d2, d, kv, level=CHI2_6_999):
    g1 = chain.rel(d2["q"], d["q"]) if d2["q"] < d["q"] else None
    if d2["c"] == d["c"]:
        g2 = (np.eye(3), np.zeros(3), np.zeros((6, 6)))
        inv2 = False
    elif d2["c"] < d["c"]:
        g2 = chain.rel(d2["c"], d["c"])
        inv2 = False
    else:
        g2 = chain.rel(d["c"], d2["c"])
        inv2 = True
    if g1 is None or g2 is None:
        return False
    Zq = np.eye(4)
    Zq[:3, :3], Zq[:3, 3] = g1[0], g1[1]
    Zc = np.eye(4)
    Zc[:3, :3], Zc[:3, 3] = g2[0], g2[1]
    if inv2:
        Zc = inv(Zc)
    A_ = d2["Z"] @ Zq
    B_ = Zc @ d["Z"]
    e = np.concatenate([so3_log(A_[:3, :3].T @ B_[:3, :3]), B_[:3, 3] - A_[:3, 3]])
    S = d2["Sl"] + d["Sl"] + Kappa.scale(g1[2] + g2[2], kv)
    try:
        return float(e @ np.linalg.solve(S, e)) <= level
    except np.linalg.LinAlgError:
        return False


__all__ = ["Chain", "Kappa", "CycleKappa", "Admission", "loop_cov", "residual", "pcm_test", "lcb_rank", "N_KF", "SUFF", "FLOOR",
           "RELOC", "CHI2_6_MED", "CHI2_6_999", "ChainPrefix", "KappaF", "CLS_VI", "CLS_VO", "CLS_LINK"]
