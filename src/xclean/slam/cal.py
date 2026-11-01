# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.3 fix round 3, designer "cal": CONSISTENCY FIRST.

The relative weight of a loop against the odometry over the loop's OWN span is set by covariance matching of the
loop-vs-odometry residuals (Mehra 1970 innovation-based adaptive estimation; Bar-Shalom NIS consistency), per component
(rotation phi / translation t) and per span, GT-free and causal:

  residual   r = chain(c -> q) [-] Z  ([phi, t], admit.residual; the admission's own residual)
  model      r ~ N(0, D + Sigma_loop),  D = lambda (.) Sigma_chain,raw(c -> q)   (the TRUE odometry error over the span)
  sample     per block b: lambda_b solves r_b^T (lambda_b Sc_bb + Sl_bb)^-1 r_b = chi2_3 median  (Kappa._solve's rule),
             span coordinate u_b = logdet Sc_raw,bb (the F2 kappa_F coordinate)
  population every made attempt that passes the Mahalanobis test at the current kappa (the GATED population that feeds
             kappa) whose chain c -> q lies inside ONE estimator state with both ends in it (CONFIRM 4 sample hygiene)
  value      lambda_b(u, t) = median of the SUFF samples nearest in u among those decided strictly before t (ties to the
             newest; the F2 KappaF selection); u below the selected samples' smallest u -> min(1, that median) (never
             extrapolated UP to a shorter span); fewer than SUFF samples -> COLD
  cold       D = kappa (.) Sc_raw: the drift model the head filter itself uses (the loop's admission kappa)
  gain       K = D (D + Sigma_loop)^-1 (6x6, the minimum-variance weight of the loop against the odometry over c -> q)
  pre-fusion Z_cal = Z [+] (I - K) r: the MV estimate of the relative pose c -> q from the odometry and the loop.  It
             replaces Z in the head filter's fusion term (covariance unchanged: Sigma_loop, so the odometry's part of
             Z_cal is not credited twice); K -> I (loop far better than the odometry) gives Z (p13b_fix1), K -> 0 gives
             the odometry (no correction).

Constants: SUFF (carried), CHI2_3 median (statistic), the solve's bounds and iteration count (carried from Kappa._solve),
the nearest-in-u selection and its tie rule (carried from F2 KappaF), the below-domain cap 1 (class D: the odometry's own
covariance, never a larger drift than measured at longer spans).  Nothing new is set, nothing per row.
Development record: v1 (two span classes by the logdet break-even, cold -> Z) is kept in an internal note.
"""
from __future__ import annotations

import bisect
import math
from collections import deque

import numpy as np

from xclean.slam.admit import SUFF, Kappa
from xclean.slam.geom import so3_exp, compose

_CHI2_3_MED = 2.3659738843753377
_BLK = (slice(0, 3), slice(3, 6))            # [phi, t]
_I3 = np.eye(3)


def cal_solve(nu, A, L, med=_CHI2_3_MED):
    """lambda >= 0 with nu^T (lambda A + L)^-1 nu = med (the rule, bounds and 70 bisections of Kappa._solve), evaluated
    in the L-whitened eigenbasis of A.  0 when lambda -> 0 already explains nu.  None on a numerical failure."""
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
    pairs = [(float(x), float(y)) for x, y in zip(a2, lam)]

    def f(m):
        return sum(x / (m * y + 1.0) for x, y in pairs)
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


def _u(S):
    s, ld = np.linalg.slogdet(S + 1e-300 * _I3)
    return float(ld) if s > 0 else -np.inf


# F4 arm R: H = the first output of a SplitMix64 generator whose
# state is x (Steele, Lea, Flood 2014; Java SplittableRandom / Vigna splitmix64.c constants), of ONE fixed packing of
# (b, k, m) into 64 bits: x = b * 2^63 + (k + 2^30) * 2^32 + m.  No seed.
_M64 = (1 << 64) - 1


def _splitmix64(x):
    z = (int(x) + 0x9E3779B97F4A7C15) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def _res_pack(b, k, m):
    kk = int(k) + (1 << 30)
    if b not in (0, 1) or not 0 <= kk < (1 << 31) or not 1 <= int(m) < (1 << 32):
        raise ValueError("F4 R packing out of its domain: b %r k %r m %r" % (b, k, m))
    return (int(b) << 63) | (kk << 32) | int(m)


class RelCal:
    """The causal per-block, per-span calibration of the odometry's error over a loop's span (see the module doc)."""

    def __init__(self, suff=SUFF, store="all"):
        if store not in ("all", "bin", "res"):
            raise ValueError("RelCal store must be all | bin | res, got %r" % (store,))
        self.suff = int(suff)
        self.store = store
        self.us = {0: [], 1: []}             # sorted u per block (0 = phi, 1 = t)
        self.rec = {0: [], 1: []}            # parallel (u, t, order, lambda)
        self.ts = []                         # commit times (sorted); store "bin": the SUFF smallest only (X2.2-5)
        self.pending = []
        self.n = 0
        self.n_obs = 0
        self.n_fail = 0
        self.n_xstate = 0
        # F3B arm X, X2: store = "bin" keeps per block the newest SUFF samples per unit bin floor(u);
        # "all" = CAL-b's store (every sample).
        self.bins = ({}, {})                 # store "bin": per block, bin floor(u) -> deque of (order n, u), oldest first
        self.n_evict = 0
        self.n_viol = 0                      # L4: a bin above SUFF, or a block above SUFF x occupied bins (must stay 0)
        self.n_nonmono = 0                   # commits whose time is below an earlier commit's (reported; expected 0)
        self.t_last = -np.inf
        self.size_max = [0, 0]
        self.bins_max = [0, 0]
        self.fill_max = 0
        self.ts_max = 0
        self.u_lo = [np.inf, np.inf]
        self.u_hi = [-np.inf, -np.inf]
        # F4 arm R: store = "res" keeps per block, per unit bin floor(u), Vitter's Algorithm R
        # reservoir of capacity SUFF: bin k -> [m_k (arrivals ever committed), slots (list of (order n, u))]
        self.res = ({}, {})
        self.n_skip = 0                      # store "res": arrivals not stored (j >= SUFF)
        self.m_max = 0                       # store "res": the largest per-bin counter m_k

    def observe(self, r, Sc, Sl, xstate=False):
        """One gated attempt: queue its per-block (lambda, u) sample (committed after the job)."""
        if xstate:
            self.n_xstate += 1
            return None
        out = []
        for b, sl in enumerate(_BLK):
            lam = cal_solve(r[sl], Sc[sl, sl], Sl[sl, sl])
            u = _u(Sc[sl, sl])
            if lam is None or not np.isfinite(u):
                self.n_fail += 1
                return None
            out.append((lam, u))
        self.pending.append(tuple(out))
        self.n_obs += 1
        return out

    def commit(self, t):
        if self.pending:
            if float(t) < self.t_last:
                self.n_nonmono += 1
            self.t_last = max(self.t_last, float(t))
        for smp in self.pending:
            for b, (lam, u) in enumerate(smp):
                if self.store == "res":
                    self._res_add(b, u, float(t), self.n, lam)     # F4 R: Algorithm R decides whether it is stored
                else:
                    i = bisect.bisect_right(self.us[b], u)
                    self.us[b].insert(i, u)
                    self.rec[b].insert(i, (u, float(t), self.n, lam))
                self.u_lo[b] = min(self.u_lo[b], u)
                self.u_hi[b] = max(self.u_hi[b], u)
                if self.store == "bin":
                    self._bin_add(b, u, self.n)
            bisect.insort(self.ts, float(t))
            if self.store in ("bin", "res"):
                del self.ts[self.suff:]      # count(ts < t) >= SUFF  <=>  the SUFF-th smallest commit time < t (X2.2-5)
            self.n += 1
        if self.pending:
            self._measure()
        self.pending = []

    def _bin_add(self, b, u, n):
        """X2 (F3B_DESIGN X2.2-1..3): the sample (n, u) joins its unit bin floor(u); while the bin holds more than SUFF,
        its oldest leaves the store (the sorted-u list and the records together)."""
        k = math.floor(u)
        dq = self.bins[b].get(k)
        if dq is None:
            dq = self.bins[b][k] = deque()
        dq.append((n, u))
        while len(dq) > self.suff:
            n0, u0 = dq.popleft()
            us, rec = self.us[b], self.rec[b]
            i = bisect.bisect_left(us, u0)
            while rec[i][2] != n0:
                i += 1
            del us[i]
            del rec[i]
            self.n_evict += 1

    def _res_add(self, b, u, t, n, lam):
        """F4 arm R: arrival (n, u) of block b into its unit bin k = floor(u); m_k += 1.
        m <= SUFF: stored in slot m - 1.  m > SUFF: j = H(b, k, m) mod m; j < SUFF: slot j's sample leaves the store (the
        sorted-u list and the records together) and the new sample takes slot j; otherwise it is not stored."""
        k = math.floor(u)
        st = self.res[b].get(k)
        if st is None:
            st = self.res[b][k] = [0, []]
        st[0] += 1
        m = st[0]
        if m > self.m_max:
            self.m_max = m
        us, rec = self.us[b], self.rec[b]
        if m <= self.suff:
            st[1].append((n, u))                                   # slot m - 1
        else:
            j = _splitmix64(_res_pack(b, k, m)) % m
            if j >= self.suff:
                self.n_skip += 1
                return
            n0, u0 = st[1][j]
            i = bisect.bisect_left(us, u0)
            while rec[i][2] != n0:
                i += 1
            del us[i]
            del rec[i]
            st[1][j] = (n, u)
            self.n_evict += 1
        i = bisect.bisect_right(us, u)
        us.insert(i, u)
        rec.insert(i, (u, t, n, lam))

    def _measure(self):
        """L4 bookkeeping after a commit (no output reads it)."""
        for b in (0, 1):
            sz = len(self.rec[b])
            self.size_max[b] = max(self.size_max[b], sz)
            if self.store == "bin":
                nb = len(self.bins[b])
                self.bins_max[b] = max(self.bins_max[b], nb)
                fill = max(len(d) for d in self.bins[b].values()) if nb else 0
                self.fill_max = max(self.fill_max, fill)
                if fill > self.suff or sz > self.suff * nb:
                    self.n_viol += 1
            elif self.store == "res":                              # F4 R: asserted at every commit
                bins = self.res[b]
                nb = len(bins)
                self.bins_max[b] = max(self.bins_max[b], nb)
                fill = max(len(s[1]) for s in bins.values()) if nb else 0
                self.fill_max = max(self.fill_max, fill)
                tot = sum(len(s[1]) for s in bins.values())
                bad = any(len(s[1]) != min(s[0], self.suff) for s in bins.values())
                if (bad or fill > self.suff or tot != sz or len(self.us[b]) != sz or sz > self.suff * nb or
                        len(self.ts) > self.suff):
                    self.n_viol += 1
        self.ts_max = max(self.ts_max, len(self.ts))

    def value(self, b, u, t):
        """-> (lambda or None, state): cold | below | near (F2 KappaF's nearest-in-u walk, ties to the newest)."""
        if bisect.bisect_left(self.ts, t) < self.suff:
            return None, "cold"
        us, rec = self.us[b], self.rec[b]
        pos = bisect.bisect_left(us, u)
        lo, hi = pos - 1, pos
        sel = []
        while len(sel) < self.suff and (lo >= 0 or hi < len(rec)):
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
        if len(sel) < self.suff:
            return None, "cold"
        med = float(np.median([x[3] for x in sel]))
        if u < min(x[0] for x in sel):
            return min(1.0, med), "below"
        return med, "near"

    def prefuse(self, Z, r, Sc, Sl, t, kv):
        """-> (Z_cal, info).  Z_cal = Z [+] (I - K) r, K = D (D + Sl)^-1, D = lambda (.) Sc; a cold block uses kappa."""
        lam, st = [], []
        for b, sl in enumerate(_BLK):
            v, s = self.value(b, _u(Sc[sl, sl]), t)
            if v is None:
                v = float(kv[b])
            lam.append(v)
            st.append(s)
        D = Kappa.scale(Sc, lam)
        try:
            K = D @ np.linalg.inv(D + Sl)
            # F3-CAL gate: the loop's NIS against the odometry over its OWN span under the calibrated
            # covariance (Bar-Shalom's normalised innovation squared); beside it the same statistic at the admission's
            # kappa
            nis = float(r @ np.linalg.solve(D + Sl, r))
            nis_k = float(r @ np.linalg.solve(Kappa.scale(Sc, kv) + Sl, r))
        except np.linalg.LinAlgError:
            return Z, {"st": "fail", "sts": st}
        d = (np.eye(6) - K) @ r
        Zc = np.array(Z, float, copy=True)
        Zc[:3, :3] = Z[:3, :3] @ so3_exp(d[:3])
        Zc[:3, 3] = Z[:3, 3] + d[3:]
        g = K @ r
        gp = float(np.linalg.norm(g[:3]) / max(np.linalg.norm(r[:3]), 1e-300))
        gt = float(np.linalg.norm(g[3:]) / max(np.linalg.norm(r[3:]), 1e-300))
        stt = "cold" if st == ["cold", "cold"] else ("warm" if "cold" not in st else "mixed")
        return Zc, {"st": stt, "sts": st, "lam": [lam[0], lam[1]], "gain": [gp, gt],
                    "trK": [float(np.trace(K[:3, :3]) / 3.0), float(np.trace(K[3:, 3:]) / 3.0)],
                    "nis": nis, "nis_k": nis_k}

    def summary(self):
        out = {"suff": self.suff, "observed": self.n_obs, "fail": self.n_fail, "xstate_not_sampled": self.n_xstate,
               "samples": self.n}
        for b, nm in ((0, "phi"), (1, "t")):
            L = [x[3] for x in self.rec[b]]
            out["lam_" + nm + "_p10_50_90"] = [float(x) for x in np.percentile(L, [10, 50, 90])] if L else None
        # F3B X2 / L4: the store and its stated bound (F3B_DESIGN X2.3)
        occ, fill = [], 0
        for b in (0, 1):
            cnt = {}
            for x in self.rec[b]:
                k = math.floor(x[0])
                cnt[k] = cnt.get(k, 0) + 1
            occ.append(len(cnt))
            fill = max([fill] + list(cnt.values()))
        ur = [[float(self.u_lo[b]), float(self.u_hi[b])] if np.isfinite(self.u_lo[b]) else None for b in (0, 1)]
        bnd = self.store in ("bin", "res")
        out.update({"store": self.store, "store_size": [len(self.rec[0]), len(self.rec[1])],
                    "store_size_max": list(self.size_max), "bins_occupied_end": occ,
                    "bins_occupied_max": (list(self.bins_max) if bnd else occ), "u_range": ur,
                    "bound": [None if u is None else self.suff * (math.floor(u[1]) - math.floor(u[0]) + 1) for u in ur],
                    "bin_fill_max": (self.fill_max if bnd else fill), "evictions": self.n_evict,
                    "bound_violations": (self.n_viol if bnd else None),
                    "nonmono_commits": self.n_nonmono, "ts_len": len(self.ts), "ts_len_max": self.ts_max})
        if self.store == "res":                                    # F4 R: the counters m_k
            out.update({"res_counters": [len(self.res[0]), len(self.res[1])],
                        "res_m_max": [max([s[0] for s in self.res[b].values()] or [0]) for b in (0, 1)],
                        "res_m_sum": [sum(s[0] for s in self.res[b].values()) for b in (0, 1)],
                        "res_replaced": self.n_evict, "res_not_stored": self.n_skip,
                        "res_bin_fill_end": [max([len(s[1]) for s in self.res[b].values()] or [0]) for b in (0, 1)]})
        return out


# =====================================================================================================================
# F3B arm X, X1: the OUTPUT-NEUTRAL running composition of the
# non-final chain edges in the frozen head's prior.  admit.Chain._rel_fwd(a, b) composes the non-final edges m .. b-1
# (m = min(max(n_final, a), b)) one at a time at the event; where n_final stops (a segment restart leaves an edge that is
# never finalised) the first-event prior chain.rel(0, j) composes ~j edges inside the event (madmax_E1: ~1 400, 17-28 ms).
# RunningRel performs the SAME geom.compose calls on the SAME operands in the SAME order, advanced at the start of every
# worker job (one compose per job while the key holds), so the event finds rel(a, j) already formed.  Reads never write
# (the replay's min-of-3 re-executions all do the same work); merge actions invalidate every state before they touch
# the chain (set_link / add_bridge / remove_bridge are the only mutators of past chain values).
# =====================================================================================================================
def _identity_triple():
    return (np.eye(3), np.zeros(3), np.zeros((6, 6)))     # the triple Chain._rel_fwd starts from when m == a


class RunningRel:
    """Per maintained anchor a: [m, k, out] with out == the triple Chain._rel_fwd(a, k) holds after composing edge
    k - 1 (start = chain._prefix_rel(a, m) if m > a else the identity triple).  rel(chain, a, b) == chain.rel(a, b)
    byte for byte (F3B_DESIGN X1.2)."""

    def __init__(self):
        self.st = {}
        self.n_adv = 0
        self.n_restart = 0
        self.n_comp = 0
        self.n_hit = 0
        self.n_miss = 0
        self.n_inval = 0
        self.n_hit_comp = 0                  # edges composed inside rel() on a hit (normally 0: k = b = j)
        self.max_anchors = 0

    def invalidate(self):
        """every state dropped (a merge action is about to change past chain values)."""
        self.st = {}
        self.n_inval += 1

    def advance(self, chain, anchors, upto=None):
        """Bring every anchor's state to keyframe n = min(upto, chain.n); drop states of other anchors.  -> composes."""
        n = chain.n if upto is None else min(int(upto), chain.n)
        new = {}
        nc = 0
        for a in anchors:
            a = int(a)
            if a in new or a >= n or chain.ucount[n] != chain.ucount[a]:
                continue
            m = min(max(chain.n_final, a), n)
            if m >= n:                                       # all final: _rel_fwd uses the prefix sums (O(1))
                continue
            s = self.st.get(a)
            if s is not None and s[0] == m and s[1] <= n:
                k, out = s[1], s[2]
            else:                                            # restart (first use, an anchor change, n_final moved)
                out = chain._prefix_rel(a, m) if m > a else _identity_triple()
                k = m
                self.n_restart += 1
            for i in range(k, n):
                out = compose(out, (chain.ZR[i], chain.Zp[i], chain.S1[i]))
            nc += n - k
            new[a] = [m, n, out]
        self.st = new
        self.n_comp += nc
        self.n_adv += 1
        self.max_anchors = max(self.max_anchors, len(new))
        return nc

    def rel(self, chain, a, b):
        """== chain.rel(a, b) (bit for bit); served from the running state when it applies, else chain.rel itself."""
        s = self.st.get(a)
        if s is not None and a < b <= chain.n and chain.ucount[b] == chain.ucount[a]:
            m = min(max(chain.n_final, a), b)
            if m < b and m == s[0] and s[1] <= b:
                out = s[2]
                for i in range(s[1], b):
                    out = compose(out, (chain.ZR[i], chain.Zp[i], chain.S1[i]))
                self.n_hit += 1
                self.n_hit_comp += b - s[1]
                return out
        self.n_miss += 1
        return chain.rel(a, b)

    def summary(self):
        return {"advances": self.n_adv, "restarts": self.n_restart, "composes_in_advance": self.n_comp,
                "prior_hits": self.n_hit, "prior_misses": self.n_miss, "hit_composes": self.n_hit_comp,
                "invalidations": self.n_inval, "max_anchors": self.max_anchors}


class RCChain:
    """The chain as the frozen HeadFilter._prior sees it under X1: rel() is RunningRel's (byte-identical), every other
    attribute is the chain's own."""
    __slots__ = ("chain", "rc")

    def __init__(self, chain, rc):
        self.chain = chain
        self.rc = rc

    def rel(self, a, b):
        return self.rc.rel(self.chain, a, b)

    def __getattr__(self, name):
        return getattr(self.chain, name)
