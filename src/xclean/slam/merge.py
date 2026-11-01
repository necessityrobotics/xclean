# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.3 RELOCALISATION / SEGMENT MERGING.

When P1 starts a new segment, the link between the previous segment's newest keyframe and the new KF0 is one of:
  demote            W1 demotion with a pose: the old window solved both poses jointly -> an ORDINARY edge (admit.py);
  coast / coast_vi  a coast with a propagated covariance -> the chain continues through the link (kappa-scaled), BUT the
                    first loop episode that crosses it is a MERGE: R1 below, every member also passing the Mahalanobis
                    and strict-informativeness tests through the link;
  unknown / missing no covariance -> the new segment is a separate COMPONENT; only R1 can merge it.  LIVE-MAP continues
                    with the previous target (no jump).
Merge rule R1 (decided per cross loop, inside the deciding job, causal):
  * the cross loops (query in the newer component, candidate in an older one) with >= MIN_RELOC_INLIERS (60, carried)
    form a set C that is PAIRWISE cycle-consistent at chi2_6(0.999) through both sides' own chains (kappa-scaled);
  * C spans >= 2 DISTINCT older places: two candidates are distinct when they are NOT co-visible by our own verification
    (rich-set PnP between them < MIN_LOOP_INLIERS 30: the verification floor, no new constant);
  * no rival: no inconsistent set of the same size among the recent cross loops (newest 60);
  * PROVISIONAL: published (slewed, flagged map_merged = provisional); CONFIRMED by a later loop (>= 30 inliers, query
    more than N_KF keyframes after C's newest) consistent with C; REVERTED by a later R1-qualifying set inconsistent with
    every member of C (the target returns to the pre-merge state, the decision restarts with all loops).
The merge itself: unknown links get a component BRIDGE (admit.Chain.add_bridge) fused from C; coast links get their
value re-estimated from C (Chain.set_link).  The LIVE-MAP correction is a head-filter event with C's loops (no prior
across an unknown link).
Own code; the R1 set rule is the design's planted-cut tool's (p13/tools/p13_merge.py r1_set), with a proper first-order
inverse for reversed chains (the tool used a same-magnitude approximation for its cycle tests).
"""
from __future__ import annotations

import numpy as np

from xclean.slam.admit import Kappa, RELOC, FLOOR, N_KF, CHI2_6_999
from xclean.slam.geom import so3_log, inv, se3, Ad, to_right, logr, expr, ortho

POOL_WIN = 60           # the PCM window


def _T(g):
    return se3(g[0], g[1])


def pcm_pair(chain, L1, L2, kv, level=CHI2_6_999):
    """cycle Z1 Zodo(q1 -> q2) vs Zodo(c1 -> c2) Z2 (both = T_c1^-1 T_q2) for loops in any order."""
    g1 = chain.rel_any(L1["q"], L2["q"])
    g2 = chain.rel_any(L1["c"], L2["c"])
    if g1 is None or g2 is None:
        return False
    A_ = L1["Z"] @ _T(g1)
    B_ = _T(g2) @ L2["Z"]
    e = np.concatenate([so3_log(A_[:3, :3].T @ B_[:3, :3]), B_[:3, 3] - A_[:3, 3]])
    S = L1["Sl"] + L2["Sl"] + Kappa.scale(g1[2] + g2[2], kv)
    try:
        return float(e @ np.linalg.solve(S, e)) <= level
    except np.linalg.LinAlgError:
        return False


def _to_phit(R, Sr):
    """inverse of geom.to_right: right-SE(3) [rho, phi] covariance -> [phi, t] (t additive in the left frame)."""
    Ki = np.zeros((6, 6))
    Ki[:3, 3:] = np.eye(3)
    Ki[3:, :3] = R
    return Ki @ Sr @ Ki.T


def fuse_rel(cands):
    """Weighted SE(3) fusion of relative-pose measurements [(Z 4x4, S [phi, t])] (Gauss-Newton, <= 5 it, right
    perturbation) -> (Z, S [phi, t])."""
    Ts = [Z for Z, _S in cands]
    Ws = [np.linalg.inv(to_right(Z[:3, :3], S) + 1e-12 * np.eye(6)) for Z, S in cands]
    M = Ts[-1].copy()
    A = None
    for _ in range(5):
        A = np.zeros((6, 6))
        b = np.zeros(6)
        for Ti, Wi in zip(Ts, Ws):
            r = logr(inv(M) @ Ti)
            A += Wi
            b += Wi @ r
        dx = np.linalg.solve(A, b)
        M = ortho(M @ expr(dx))
        if np.abs(dx).max() < 1e-9:
            break
    Sr = np.linalg.inv(A)
    return M, _to_phit(M[:3, :3], 0.5 * (Sr + Sr.T))


class MergeManager:
    def __init__(self, chain, covis_fn, enabled=True):
        """covis_fn(a, b) -> rich-set PnP inliers between keyframes a and b (cached here; each call is charged by the
        worker's cost model)."""
        self.chain = chain
        self.covis_fn = covis_fn
        self.enabled = enabled
        self.links = {}          # k -> dict(kind, state, Z, S)  for coast (needs_merge) and unknown links
        self.pools = {}          # merge key -> [cross loops], key = ("comp", comp_new) or ("link", k)
        self.prov = {}           # key -> provisional state
        self.done = {}           # key -> confirmed state
        self._covis = {}
        self.n_covis = 0
        self.n_pooled_cold = 0
        self.log = []

    # ---- links ---------------------------------------------------------------------------------------------------
    def register_link(self, k, kind, ZR, Zp, S):
        if kind == "demote" or kind == "edge":
            return
        if kind in ("coast", "coast_vi") and S is not None:
            self.links[k] = {"kind": kind, "state": "needs_merge", "Z": (np.array(ZR), np.array(Zp), np.array(S))}
        else:
            self.links[k] = {"kind": kind, "state": "unknown", "Z": None}

    def covis(self, a, b):
        key = (min(a, b), max(a, b))
        if key not in self._covis:
            self.n_covis += 1
            self._covis[key] = int(self.covis_fn(key[0], key[1]))
        return self._covis[key]

    def distinct(self, a, b):
        return a != b and self.covis(a, b) < FLOOR

    def key_of(self, c, q):
        """The merge relation a loop (c, q) crosses, or None: an unmerged / provisional coast link between them (the
        newest), or the newer component when c and q are in different components."""
        if c >= q:
            return None
        ch = self.chain
        if ch.comp(c) != ch.comp(q):
            return ("comp", ch.comp(q))
        ks = [k for k, L in self.links.items() if L["kind"] in ("coast", "coast_vi") and L["Z"] is not None
              and c <= k < q and L["state"] in ("needs_merge", "provisional")]
        if ks:
            return ("link", max(ks))
        return None

    # ---- R1 ------------------------------------------------------------------------------------------------------
    def r1_set(self, pool, L, kv):
        nb = [x for x in pool if pcm_pair(self.chain, x, L, kv)]
        C = [L]
        for x in sorted(nb, key=lambda y: -y["inliers"]):
            if all(pcm_pair(self.chain, x, y, kv) for y in C[1:]):
                C.append(x)
        if len(C) < 2:
            return None
        places = []
        for x in C:
            if all(self.distinct(x["c"], p_) for p_ in places):
                places.append(x["c"])
            if len(places) >= 2:
                break
        if len(places) < 2:
            return None
        rivals = [x for x in pool if all(x is not y for y in C) and not any(pcm_pair(self.chain, x, y, kv) for y in C)]
        if len(rivals) >= len(C):
            rv = rivals[0]
            rC = [rv] + [x for x in rivals[1:] if pcm_pair(self.chain, x, rv, kv)]
            if len(rC) >= len(C):
                return None
        return C

    def on_loop(self, L, kv, t, decide=True):
        """A verified cross loop (dict with q, c, inliers, Z, Sl, 'passed' = passed admission tests 1-3 through a known
        link, 'cold' = kappa(t) was cold so nothing could pass them) decided at t.  -> an action dict for the worker:
        {"merge": ...} / {"revert": ...} / None.
        decide=False (fix round, verifier major 2): no chain scale exists yet (kappa(t) and kappa_cyc(t) both cold): the
        loop is POOLED for the decision a later job takes once a scale exists (build 1 tested on the raw, 19-1 000x
        over-confident chain, which never holds)."""
        if not self.enabled:
            return None
        key = self.key_of(L["c"], L["q"])
        if key is None:
            return None
        if not decide:
            if key not in self.prov and key not in self.done and L["inliers"] >= RELOC and \
                    (key[0] == "comp" or L.get("passed", False) or L.get("cold", False)):
                self.pools.setdefault(key, []).append(L)
                self.n_pooled_cold += 1
            return None
        kv = kv if kv is not None else (1.0, 1.0, 0)
        if key in self.prov:                               # confirm / revert: any verified crossing loop counts
            return self._provisional(key, L, kv, t)
        if key in self.done:
            return None
        if key[0] == "link" and not (L.get("passed", False) or L.get("cold", False)):
            return None                                    # coast link: members must pass Mahalanobis + informativeness
            #                                                (while kappa is cold none can: R1 decides alone, as for an
            #                                                unknown link)
        if L["inliers"] < RELOC:
            return None
        pool = self.pools.setdefault(key, [])
        C = self.r1_set(pool[-POOL_WIN:], L, kv)
        pool.append(L)
        if C is None:
            return None
        return {"merge": key, "C": C, "t": t}

    def _provisional(self, key, L, kv, t):
        st = self.prov[key]
        if L["q"] <= st["qmax"] + N_KF or L["inliers"] < FLOOR:
            return None
        if any(pcm_pair(self.chain, y, L, kv) for y in st["C"]):
            st["state"] = "confirmed"
            st["t_confirm"] = t
            self.done[key] = self.prov.pop(key)
            if key[0] == "link":
                self.links[key[1]]["state"] = "confirmed"
            self.log.append({"t": t, "kind": "confirm", "key": list(key), "q": L["q"], "c": L["c"]})
            return {"confirm": key, "t": t}
        if L["inliers"] >= RELOC:
            C2 = self.r1_set(st["pool2"][-POOL_WIN:], L, kv)
            st["pool2"].append(L)
            if C2 is not None and not any(pcm_pair(self.chain, y, z, kv) for y in st["C"] for z in C2):
                return {"revert": key, "C2": C2, "t": t}
        return None

    # ---- applying a merge (called by the worker, which owns the head filter) ---------------------------------------
    def fuse_merge(self, key, C, kv):
        """-> (kind, payload) for the chain: ("bridge", (comp_new, a, b, ZR, Zp, S)) or ("link", (k, ZR, Zp, S))."""
        ch = self.chain
        kv = kv if kv is not None else (1.0, 1.0, 0)
        L0 = C[0]
        if key[0] == "comp":
            a, b = L0["c"], L0["q"]
            meas = []
            for Li in C:
                ga = ch.rel_any(a, Li["c"])            # old side: a -> c_i
                gb = ch.rel_any(Li["q"], b)            # new side: q_i -> b
                if ga is None or gb is None:
                    continue
                Zi = _T(ga) @ Li["Z"] @ _T(gb)
                Ab = Ad(inv(_T(gb)))
                Sr = Ab @ to_right(Li["Z"][:3, :3], Li["Sl"]) @ Ab.T + to_right(Zi[:3, :3], Kappa.scale(ga[2] + gb[2], kv))
                meas.append((Zi, _to_phit(Zi[:3, :3], Sr)))
            Z, S = fuse_rel(meas)
            return "bridge", (key[1], a, b, Z[:3, :3], Z[:3, 3], S)
        k = key[1]
        meas = []
        for Li in C:
            ga = ch.rel_any(k, Li["c"])                # k -> c_i (c_i <= k: an inverse chain)
            gb = ch.rel_any(Li["q"], k + 1)            # q_i -> k+1
            if ga is None or gb is None:
                continue
            Zi = _T(ga) @ Li["Z"] @ _T(gb)
            Ab = Ad(inv(_T(gb)))
            Sr = Ab @ to_right(Li["Z"][:3, :3], Li["Sl"]) @ Ab.T + to_right(Zi[:3, :3], Kappa.scale(ga[2] + gb[2], kv))
            meas.append((Zi, _to_phit(Zi[:3, :3], Sr)))
        Z, S = fuse_rel(meas)
        return "link", (k, Z[:3, :3], Z[:3, 3], S)

    def set_provisional(self, key, C, t, snapshot, chain_change):
        self.prov[key] = {"state": "provisional", "C": C, "qmax": max(x["q"] for x in C), "t": t, "pool2": [],
                          "snapshot": snapshot, "chain_change": chain_change}
        if key[0] == "link":
            self.links[key[1]]["state"] = "provisional"
        self.log.append({"t": t, "kind": "merge", "key": list(key), "n": len(C),
                         "loops": [(int(x["q"]), int(x["c"]), int(x["inliers"])) for x in C]})

    def undo(self, key, t):
        st = self.prov.pop(key)
        if key[0] == "link":
            self.links[key[1]]["state"] = "needs_merge"
        self.log.append({"t": t, "kind": "revert", "key": list(key)})
        pool = self.pools.setdefault(key, [])
        pool.extend(st["pool2"])
        return st

    def state_of(self, key):
        if key in self.done:
            return "confirmed"
        if key in self.prov:
            return "provisional"
        return "none"
