# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.3 WORKER: `WorkerCore`, the worker's whole decision logic with an INJECTED CLOCK and no
I/O; `MessageAdapter`, the live message protocol (xclean/live/p13_link.py WORKER INTERFACE) -> WorkerCore; and
`worker_main(handles, wcfg)`, the live process entry point the main builder's harness spawns (p13_link.worker_entry).

Per keyframe packet j, one JOB (FIFO; a job that STARTS with >= 2 packets waiting is SHED: it stores the rich set and
the BoW so the keyframe can still become a candidate, and makes no attempt; a packet whose images the ring dropped gives
a header-only job):
  rich set + BoW + stereo 3D -> DB -> retrieval (median + 3 scaled MAD; top 3 ABOVE the threshold) -> eligibility ->
  rich-set PnP verification -> [messages up to the job's finish] -> admission (floor 30, Mahalanobis at kappa(t)
  Sigma_chain, strict informativeness, PCM >= 1) -> R1 merge logic for cross loops -> the anchored head filter ->
  T_target published at finish + solve.
Causality: the worker sees a packet only after its emission; kappa, PCM and eligibility use only attempts
decided before the deciding job finished; anchors are LIVE-MAP poses already published; GT is never read.
Clock: "model" (lock-step / replay: the FROZEN cost model, per-stage p95 x the job's own work counts, so a replay is a
pure function of its inputs) or "wall" (paced / live: the measured thread CPU time of the job).
Switches: XCLEAN_P13_OFF (token `loop`), XCLEAN_P13_APPLY, XCLEAN_P13_SLEW_F, XCLEAN_P13_WORKER are the frame side's
(xclean/live/p13_link.py).  The worker adds none; its attribution arms are arguments of the offline replay only.
"""
from __future__ import annotations

import bisect
import hashlib as _hl
import json
import os
import time

import numpy as np

from xclean.slam.admit import Chain, Admission, FLOOR, N_KF, CLS_VI, CLS_VO
from xclean.slam.headgraph import HeadFilter, HeadMV, EMPTY_LIN, HeadCal
from xclean.slam.geom import inv as _inv, logr as _logr
from xclean.slam.kfdb import KfDb
from xclean.slam.merge import MergeManager
from xclean.slam.packets import KfPacket, EdgeFinal

# SNAPSHOT: DEFAULT flipped to head=cal, cal_cold=feed, cal_store=res (X1 cal_rc=on); off = head=anchored, cal_cold=none, cal_store=all (p13b_fix1)
DEFAULT_CFG = {"kappa": "scalar", "cold": "median30", "pcm_win": 60, "prefilter": True, "merge": True,
               "prior_mode": "posterior", "clock": "model", "cost": None, "shed_at": 2,
               # fix round 2026-10-26 (verifier majors 1 / 2): provisional weak-prior episodes, the gated kappa feed, and
               # R1's cycle-calibrated scale while kappa is cold.  The design / build-1 values: carry / all / raw.
               "episodes": "provisional", "kappa_feed": "gated", "r1_cold": "cycle",
               # P1.3 fix round 2 (F2_DESIGN 3.11): the head correction ("anchored" = HeadFilter, p13b_fix1; "mv" =
               # HeadMV, F2), the PCM pool's cold start ("drop" = frozen; "rejudge" = MV-4) and the ledger window M_L
               # (60 = PCM_WIN; 0 = unbounded, the A4 reference only).  Defaults = p13b_fix1 until the F2 verdict.
               "head": "cal", "pcm_cold": "drop", "ledger_win": 60,
               "f2_time_reps": 1,             # F2 solve timing: min-of-N re-executions (replay, model clock only)
               # P1.3 fix round 3, designer "cal": head = "cal" = HeadCal (the frozen head filter;
               # every admitted loop pre-fused with the odometry over its own span at the innovation-calibrated weight,
               # the calibrated NIS gate, the cross-state refusal); cal_time_reps = min-of-N timing (replay only);
               # cal_gate = "nis" (the design) | "off" (development arm only)
               # cal_cold = "none" (design a: RelCal fed only once kappa is warm) | "feed" (design b: fed unconditionally
               # while kappa(t) is cold, kappa's own cold-start rule; F3_CAL_DESIGN 3.2b)
               "cal_time_reps": 1, "cal_mode": "rel", "cal_gate": "nis", "cal_cold": "feed",
               # F3B arm X, read only under head = "cal": cal_rc = "on" (X1: the output-neutral
               # running composition of the head prior's chain, advanced per job) | "off" (CAL-b's path, timing control)
               "cal_rc": "on",
               # X2: cal_store = "all" (CAL-b: every calibration sample kept) | "bin" (the bounded store: the newest SUFF
               # samples per unit bin of u per block, F3_CAL_DESIGN 5; the arm's configuration)
               # | "res" (F4 arm R of the loop-closure design: per unit bin of u per block, Vitter's Algorithm R reservoir of
               # capacity SUFF, H = splitmix64 of the fixed (b, k, m) packing; a uniform sample of the bin's whole history)
               "cal_store": "res"}
F2_CFG = {"head": "mv", "pcm_cold": "rejudge", "ledger_win": 60}
BUILD1_CFG = {"episodes": "carry", "kappa_feed": "all", "r1_cold": "raw"}
MERGE_CODE = {"none": 0, "provisional": 1, "confirmed": 2}
KIND_CODE = {"loop": 1, "merge": 2, "merge_revert": 3}
COST_TABLE = "p13_worker_cost.json"     # kind "table" without XCLEAN_P13_COST: a table file in the repository root
C_SOLVE_S = 1.0e-3        # head-filter solve, modelled
# ROW-INDEPENDENT lock-step cost model (fix round 2026-10-26, verifier minor 8): per-stage p95 costs scaled by the job's own
# work counts, so a lock-step LIVE-MAP is a pure function of its inputs on EVERY row (held-out, OAK) with no per-row
# table.  Class C: least-squares fits over the 11 design rows' frozen measurements (p13/data/r1/p13_worker_cost.json,
# single-thread min-of-3, p95), made by an internal script (its output
# p13/data/p13f_cost_ri.json carries the per-row residuals).  Seconds; n_q / n_c = keypoints of the query / candidate rich
# set, n_db = candidates in the database.
COST_RI = {"kind": "ri", "b_img": 1.4468e-3, "a_img": 3.2938e-5, "a_ret": 5.9326e-6, "c_pnp": 7.02e-3,
           "a_match": 1.14061e-8, "c_hdr": 0.0,
           # F2: the modelled F2 solve (head = "mv" only),
           # c = b_mv + a_mv (L + 1)^2 S seconds; class-C least squares over the per-row p95s of the measured solve (min-of-3,
           # single thread) and of its work count, pass 1 on the 13 design rows (p13/f2b/data/f2_cost.json)
           "b_mv": 3.4357e-4, "a_mv": 2.45426e-6}


# ==================================================================================================== F2 lineage
class Lineage:
    """F2 MV-3 (F2_DESIGN 3.4): beta^(c), the share of every epoch's increment that the PUBLISHED anchor A(c) carries,
    keyframe-recursive from T_map_cam (exact under the frame side's geodesic slew, first order).
    T_app(c) = A(c) LIVE-ODOM(c)^-1.  Between keyframes c-1 and c the frame side slews from T_app(c-1) toward the newest
    target T_new published at or before c's frame:
        gamma_c = clip(<x, d> / <d, d>, 0, 1),  x = log(T_app(c-1)^-1 T_app(c)), d = log(T_app(c-1)^-1 T_new), both at c's
                  head (head right [rho, phi]) with the loop metric (the information of the newest fused loop; identity
                  before the first event),
        beta^(c) = (1 - gamma_c) beta^(c-1) + gamma_c lambda^(T_new).
    beta is stored sparsely as a mixture of target lineages ((w, (n_max, reverted)), ...), weights summing to 1;
    lambda^(T) = (newest epoch id, reverted epochs) of target T (HeadMV.lineage())."""

    def __init__(self):
        self.b = {}
        self.Tapp = {}
        self.pub_t, self.pub_lin, self.pub_T = [], [], []
        self.metric = None
        self.last_j = None
        self.n_mix = 0

    def on_publish(self, t_pub, lin, T):
        i = bisect.bisect_right(self.pub_t, float(t_pub))
        self.pub_t.insert(i, float(t_pub))
        self.pub_lin.insert(i, lin)
        self.pub_T.insert(i, np.array(T, float))

    def on_kf(self, j, t, T_map, T_odom):
        if T_map is None or T_odom is None:
            return
        To = np.asarray(T_odom, float)
        Ta = np.asarray(T_map, float) @ _inv(To)
        prev = self.last_j
        bp = self.b.get(prev, ((1.0, EMPTY_LIN),)) if prev is not None else ((1.0, EMPTY_LIN),)
        k = bisect.bisect_right(self.pub_t, float(t))
        if k == 0:
            beta = bp
        else:
            Tn, ln = self.pub_T[k - 1], self.pub_lin[k - 1]
            Tp = self.Tapp.get(prev, np.eye(4)) if prev is not None else np.eye(4)
            H0 = Tp @ To
            x = _logr(_inv(H0) @ (Ta @ To))
            d = _logr(_inv(H0) @ (Tn @ To))
            M = self.metric if self.metric is not None else np.eye(6)
            dd = float(d @ M @ d)
            g = 1.0 if dd <= 1e-24 else min(1.0, max(0.0, float(x @ M @ d) / dd))
            if g >= 1.0:
                beta = ((1.0, ln),)
            else:
                acc = {}
                for w, lin in bp:
                    acc[lin] = acc.get(lin, 0.0) + (1.0 - g) * w
                acc[ln] = acc.get(ln, 0.0) + g
                beta = tuple((w, lin) for lin, w in acc.items() if w > 1e-12)
                if len(beta) > 1:
                    self.n_mix += 1
        self.b[j] = beta
        self.Tapp[j] = Ta
        self.last_j = j

    def beta(self, c):
        return self.b.get(c, ((1.0, EMPTY_LIN),))


class _F2Ctx:
    """what HeadMV reads from the worker: kappa_F, the keyframes' vi flags, the anchors' lineages."""

    def __init__(self, core):
        self.core = core
        self.kf = core.adm.kf

    def vi(self, k):
        return bool(self.core.db.hdr[k]["vi"])

    def beta(self, c):
        return self.core.lin.beta(c)


# ==================================================================================================== fronts
class RealFront:
    """Rich sets from the packet's rectified pair; verification on the DB's rich sets (the live path)."""

    def __init__(self, rig):
        from xclean.slam.rich import RichExtractor
        from xclean.slam.verify import Verifier
        self.ex = RichExtractor(rig)
        self.ver = Verifier(rig)

    def extract(self, p):
        if p.left is None:
            return None
        return self.ex.extract(p.left, p.right)

    def verify(self, db, c, j, rq):
        return self.ver.verify(db.rich(c), rq)

    def covis(self, db, a, b):
        return self.ver.covisible_inliers(db.rich(a), db.rich(b))


class CachedFront:
    """The design caches (p13r_cache rich_<row>.npz, verify_<row>.npz) in place of the image pipeline: the SAME
    computations made earlier, looked up by keyframe index (fast arms, and the reproduction check of the build against
    the design evidence).  A (query, candidate) pair the cache lacks is verified on the cached rich sets."""

    def __init__(self, rich_npz, verify_npz, rig):
        from xclean.slam.rich import RichSet
        from xclean.slam.verify import Verifier
        self.RichSet = RichSet
        self.rc = dict(np.load(rich_npz))
        V = dict(np.load(verify_npz, allow_pickle=False))
        self.V = V
        self.idx = {}
        for k in range(len(V["q"])):
            self.idx.setdefault((int(V["q"][k]), int(V["c"][k])), k)
        self.ver = Verifier(rig) if rig is not None else None
        self.n_miss = 0

    def rich_at(self, j):
        rc, off, bo = self.rc, self.rc["off"], self.rc["bow_off"]
        a, b = int(off[j]), int(off[j + 1])
        return self.RichSet(uv=rc["uv"][a:b], desc=rc["desc"][a:b], oct=rc["oct"][a:b], X=rc["X"][a:b],
                            dsig=rc["dsig"][a:b], bow_i=rc["bow_i"][int(bo[j]):int(bo[j + 1])].astype(np.int32),
                            bow_v=rc["bow_v"][int(bo[j]):int(bo[j + 1])], t_ms={})

    def extract(self, p):
        if p.meta.get("ring_drop") or (p.left is None and not p.meta.get("cached_images", False)):
            return None
        return self.rich_at(p.j)

    def verify(self, db, c, j, rq):
        k = self.idx.get((j, c))
        V = self.V
        if k is None:
            self.n_miss += 1
            return self.ver.verify(self.rich_at(c), rq)
        Z = V["Z_pnp"][k] if "Z_pnp" in V and np.isfinite(V["Z_pnp"][k][0, 0]) else None
        out = {"matches": int(V["matches"][k]), "inliers": int(V["inliers"][k]), "Z": None, "H": None, "chi2": None,
               "n_info": 0, "t_ms": {}}
        if Z is not None:
            out.update(Z=Z, H=V["H"][k], chi2=float(V["chi2"][k]), n_info=int(V["n_info"][k]))
        return out

    def covis(self, db, a, b):
        return self.ver.covisible_inliers(self.rich_at(a), self.rich_at(b))


# ==================================================================================================== the core
class WorkerCore:
    def __init__(self, cfg=None, rig=None, front=None, store_path=None):
        self.cfg = dict(DEFAULT_CFG)
        self.cfg.update(cfg or {})
        c = self.cfg
        for k_, ok_ in (("episodes", ("provisional", "carry")), ("kappa_feed", ("gated", "all")),
                        ("r1_cold", ("cycle", "raw")), ("head", ("anchored", "mv", "cal")),
                        ("pcm_cold", ("drop", "rejudge"))):
            if c[k_] not in ok_:
                raise ValueError("worker cfg %s must be one of %s, got %r" % (k_, ok_, c[k_]))
        if int(c["ledger_win"]) < 0:
            raise ValueError("worker cfg ledger_win must be >= 0, got %r" % (c["ledger_win"],))
        if c.get("cal_rc", "on") not in ("on", "off"):
            raise ValueError("worker cfg cal_rc must be on | off, got %r" % (c.get("cal_rc"),))
        if c.get("cal_store", "all") not in ("all", "bin", "res"):
            raise ValueError("worker cfg cal_store must be all | bin | res, got %r" % (c.get("cal_store"),))
        self.rc = None                         # F3B X1 (head = "cal", cal_rc = "on" only): RunningRel
        self._l4_prov = [0, 0]                 # F3B L4 (head = "cal"): largest open-episode loops / rivals seen
        self.chain = Chain()
        self.adm = Admission(self.chain, kappa_mode=c["kappa"], cold=c["cold"], pcm_win=c["pcm_win"],
                             prefilter=c["prefilter"], feed=c["kappa_feed"])
        self.db = KfDb(store_path)
        # F2: kappa_F's FO-3 residual reads LIVE-ODOM; MV-4 behind pcm_cold
        self.adm.T_odom_fn = lambda k: self.db.hdr[k]["T_odom"]
        self.adm.pcm_cold = c["pcm_cold"]
        self.lin = Lineage()                   # F2 MV-3: per-keyframe anchor lineage (computed in both modes)
        if c["head"] == "mv":
            self.hf = HeadMV(ledger_win=int(c["ledger_win"]), episodes=c["episodes"],
                             time_reps=int(c.get("f2_time_reps", 1)))
            self.hf.ctx = _F2Ctx(self)
        elif c["head"] == "cal":
            from xclean.slam.cal import RelCal
            self.relcal = RelCal(store=c.get("cal_store", "all"))
            self.hf = HeadCal(prior_mode=c["prior_mode"], episodes=c["episodes"],
                              time_reps=int(c.get("cal_time_reps", 1)), mode=c.get("cal_mode", "rel"),
                              relcal=self.relcal, gate=c.get("cal_gate", "nis"))
            if c.get("cal_rc", "on") == "on":
                from xclean.slam.cal import RunningRel
                self.rc = RunningRel()         # F3B X1: advanced per job, read by HeadCal._prior (output-neutral)
                self.hf.rc = self.rc
        else:
            self.hf = HeadFilter(prior_mode=c["prior_mode"], episodes=c["episodes"])
        self.perturb = None                    # REPLAY-ONLY perturbation arms (F2_DESIGN 5.4): {"thin": (p, seed)}
        self.n_f2_err = 0
        self.front = front if front is not None else RealFront(rig)
        self._job_covis = 0
        self._job_covis_pairs = 0
        self.merge = MergeManager(self.chain, self._covis, enabled=c["merge"])
        self.attempts = []                     # every MADE attempt (decision records)
        self.events = []                       # every correction published
        self.jobs = []                         # per job: j, t_start, t_fin, shed, dropped, n_att, cpu ms, waiting
        self.n_shed = self.n_dropped = self.n_below_thr = self.n_inelig = self.n_cold_db = 0
        self.plant = c.get("plant")            # TEST / AUDIT ONLY: {"event": n, "dt_m":, "dyaw_deg":, "stage": post|pre}
        self.n_events_admitted = 0
        self.nkp = {}                          # keyframe -> keypoints of its rich set (the RI cost model's work count)
        self._jt = (0.0, 0.0)                  # (t_start, thread-time origin) of the running job (wall clock)

    # ---- the cost model (clock "model") ------------------------------------------------------------------------------
    def _cost(self):
        return self.cfg["cost"] if self.cfg["clock"] == "model" else None

    def _c_base(self, n_q, n_db):
        cm = self._cost()
        if cm.get("kind") == "ri":
            return cm["b_img"] + cm["a_img"] * n_q + cm["a_ret"] * n_db
        return cm["c_base"]

    def _c_att(self, n_q, n_c):
        cm = self._cost()
        if cm.get("kind") == "ri":
            return cm["c_pnp"] + cm["a_match"] * n_q * n_c
        return cm["c_att"]

    def _now(self):
        """wall clock: the job's start + the thread CPU time spent in it so far (the worker's own clock)."""
        t_start, c0 = self._jt
        return t_start + (time.thread_time() - c0)

    # ---- messages ------------------------------------------------------------------------------------------------
    def on_header(self, p):
        """The header part of KF packet p (always delivered, even when the ring dropped its images)."""
        self.db.add_header(p)
        if p.j > 0:
            kind = "edge" if p.edge_kind == "edge" else p.edge_kind
            # F2: the window edge's estimator-state class (VI iff both keyframes' packets say vi)
            prev = self.db.hdr.get(p.j - 1)
            cls = CLS_VI if (bool(p.vi) and prev is not None and bool(prev["vi"])) else CLS_VO
            self.chain.add_edge(p.j - 1, p.ZR, p.Zp, p.S, kind=kind, cls=cls)
            if kind != "edge":
                self.merge.register_link(p.j - 1, kind, p.ZR, p.Zp, p.S)
        self._f2_guard(self.lin.on_kf, p.j, p.t, p.T_map, p.T_odom)

    def _f2_guard(self, fn, *a):
        """F2 bookkeeping: errors propagate in head = "mv"; in "anchored" mode (where nothing reads it) they are counted
        and swallowed, so the frozen path cannot be disturbed."""
        if self.cfg["head"] == "mv":
            return fn(*a)
        try:
            return fn(*a)
        except Exception:                                                    # noqa: BLE001
            self.n_f2_err += 1
            return None

    def on_final(self, m, t_emit=-np.inf):
        """edge_final of keyframe m.a, emitted with the frame stamped t_emit (it left the window then)."""
        if m.ZR is not None:
            self.chain.finalise(m.a, m.ZR, m.Zp, m.S)
        self.db.make_candidate(m.a, t_emit)

    def _covis(self, a, b):
        self._job_covis += 1
        if self._cost() is not None:
            self._job_covis_cost += self._c_att(self.nkp.get(a, 0), self.nkp.get(b, 0))
        return self.front.covis(self.db, a, b)

    def open_links(self):
        """An unmerged relation exists: an unknown link whose component is not confirmed-merged, or a coast link not
        confirmed (R1 may be needed)."""
        mm = self.merge
        for k, L in mm.links.items():
            if L["kind"] in ("coast", "coast_vi") and L["Z"] is not None:
                if L["state"] != "confirmed":
                    return True
            elif mm.state_of(("comp", self.chain.comp(k + 1))) != "confirmed":
                return True
        return False

    # ---- one job ---------------------------------------------------------------------------------------------------
    def job(self, p, t_start, waiting, feed=None):
        """Run the job of packet p starting at t_start (worker clock) with `waiting` packets queued behind it.  feed(t)
        delivers every message emitted up to t.  -> dict(t_fin, events, ...).
        Clock "model": t_fin = t_start + the cost model of the job's own work (deterministic); the decision time t_dec is
        the verification's end, events are published at t_dec + the modelled solve.  Clock "wall": the same points are
        the job's measured thread CPU time at the moment they happen (the publish stamp is taken AFTER the admission, R1
        and head-filter work: verifier minor 5)."""
        cfg = self.cfg
        cost = self._cost()
        tt = time.thread_time
        c0 = tt()
        self._jt = (t_start, c0)
        j = p.j
        shed = waiting >= cfg["shed_at"]
        self._job_covis = 0
        self._job_covis_cost = 0.0
        out = {"j": j, "t_kf": p.t, "t_start": t_start, "shed": shed, "dropped": p.left is None and not
               p.meta.get("cached_images", False), "waiting": waiting, "n_att": 0, "events": [], "stage_ms": {}}
        if self.rc is not None:
            # F3B X1 (F3B_DESIGN X1.2): bring the running composition of the head prior's chain to this job's keyframe
            # for the anchors the frozen prior can be formed from: q_last and an open episode's checkpoint q_last
            t_rc = time.perf_counter()
            anc = [self.hf.q_last]
            if self.hf.prov is not None:
                anc.append(self.hf.prov["ck"][2])
            out["rc_nc"] = self.rc.advance(self.chain, anc, upto=j)
            out["rc_ms"] = 1e3 * (time.perf_counter() - t_rc)
        if cfg["head"] == "cal" and self.hf.prov is not None:     # F3B L4 bookkeeping (read by nothing else)
            self._l4_prov = [max(self._l4_prov[0], len(self.hf.prov["loops"])),
                             max(self._l4_prov[1], len(self.hf.prov["rival"]))]
        rq = self.front.extract(p)
        if rq is None:
            self.n_dropped += 1
            out["t_fin"] = t_start + (cost["c_hdr"] if cost else (tt() - c0))
            out["cpu_ms"] = 1e3 * (tt() - c0)
            self.jobs.append(out)
            return out
        self.db.add_rich(j, rq)
        n_q = int(len(rq.uv))
        self.nkp[j] = n_q
        out["stage_ms"].update(rq.t_ms or {})
        if shed:
            self.n_shed += 1
            out["t_fin"] = t_start + (self._c_base(n_q, len(self.db.cand)) if cost else (tt() - c0))
            out["t_dec"] = out["t_fin"]
            out["cpu_ms"] = 1e3 * (tt() - c0)
            self.jobs.append(out)
            return out
        c1 = tt()
        ret = self.db.retrieve(rq.bow_i, rq.bow_v, t_query=p.t)
        out["stage_ms"]["retrieval"] = 1e3 * (tt() - c1)
        atts = []
        n_db = len(self.db.cand)
        if ret is None:
            self.n_cold_db += 1
        else:
            n_db = ret["n_db"]
            out["n_db"] = ret["n_db"]
            out["postings"] = ret["postings"]
            above = {c for c, _s, _r in ret["above"]}
            for (c, sim, rank) in ret["ranked"]:
                if c not in above:
                    self.n_below_thr += 1
                    continue
                if not self.adm.eligible(c, j, t_start):
                    self.n_inelig += 1
                    continue
                v = self.front.verify(self.db, c, j, rq)
                v.update({"q": j, "c": c, "sim": sim, "thr_ret": ret["thr"], "rank": rank, "t_start": t_start})
                atts.append(v)
        out["n_att"] = len(atts)
        if cost and cost.get("kind") != "ri":            # the per-row constant table (build 1 arithmetic, bit for bit)
            t_fin = t_start + cost["c_base"] + len(atts) * cost["c_att"]
        elif cost:
            t_fin = t_start + self._c_base(n_q, n_db) + sum(self._c_att(n_q, self.nkp.get(v["c"], 0)) for v in atts)
        else:
            t_fin = t_start + (tt() - c0)
        if feed is not None:
            feed(t_fin)
        if self.plant and self.plant.get("stage") == "pre":
            self._plant_pre(atts)
        kv_job = self.adm.kap.value(t_fin)
        if cfg["r1_cold"] == "cycle":
            self.adm.cyc_on = kv_job is None and cfg["merge"] and self.open_links()
            kv_r1, r1_src = self.adm.r1_scale(t_fin)
        else:
            kv_r1, r1_src = kv_job, ("kappa" if kv_job is not None else None)
        out["r1_scale"] = r1_src
        if cfg["pcm_cold"] == "rejudge":                 # F2 MV-4: the cold pool joins pcm_list at the first warm job
            nj = self.adm.rejudge_cold(t_fin)
            if nj:
                out["pcm_cold_joined"] = nj
        admitted, merges = [], []
        for v in atts:
            self.adm.decide(v, t_fin)
            v["t_dec"] = t_fin
            # F3-CAL: the chain c -> q is formed ONCE per attempt here (Chain.rel
            # costs O(non-final edges): 3.6 ms mean, 9 ms late on mh_01) and carried to the pre-fusion with the loop
            g_cal = None
            # the calibration population: once kappa(t) is warm, the attempts its Mahalanobis gate
            # accepts; while kappa(t) is cold (cal_cold = "feed", design b, 3.2b), EVERY made attempt with a known chain,
            # unconditionally: kappa's own cold-start rule (admit.decide), so RelCal is warm from the first admission
            cold_feed = cfg.get("cal_cold", "none") == "feed" and v.get("m2") is None and v.get("stage") == "cold"
            if cfg["head"] == "cal" and v.get("r") is not None and (
                    cold_feed or (v.get("m2") is not None and v["m2"] <= v["thr"])):
                g_cal = self.chain.rel(v["c"], v["q"]) if v["c"] < v["q"] else None
                # F3-CAL: the gated population calibrates the odometry over a loop span; a sample
                # only from a chain inside ONE estimator state with both ends in it (the CONFIRM 4 sample hygiene)
                xs = bool(self.db.hdr[v["q"]]["vi"]) != bool(self.db.hdr[v["c"]]["vi"]) or \
                    self.chain.homogeneous(v["c"], v["q"]) is None
                if g_cal is not None:
                    self.relcal.observe(np.asarray(v["r"], float), g_cal[2], v["Sl"], xstate=xs)
            v["age_s"] = t_fin - p.t
            key = self.merge.key_of(v["c"], v["q"]) if cfg["merge"] else None
            v["merge_key"] = None if key is None else list(key)
            if key is not None and self.merge.state_of(key) != "confirmed":
                passed = v["stage"] in ("pcm", "admitted")
                prov = self.merge.state_of(key) == "provisional"
                # while kappa(t) is cold no member can pass the Mahalanobis test through a coast link: R1 decides alone
                # there, as across an unknown link (fix round, verifier major 2)
                cold = v["stage"] == "cold" and cfg["r1_cold"] == "cycle"
                cross = (v["stage"] == "unknown_link" or passed or prov or cold) and v["inliers"] >= FLOOR and \
                    v.get("Sl") is not None
                if cross:
                    L = {"q": v["q"], "c": v["c"], "inliers": v["inliers"], "Z": v["Z"], "Sl": v["Sl"],
                         "passed": passed, "cold": cold, "kv": v.get("kv") or kv_r1 or (1.0, 1.0, 0)}
                    if cfg["r1_cold"] == "cycle":
                        act = self.merge.on_loop(L, kv_r1, t_fin, decide=kv_r1 is not None)
                    else:
                        act = self.merge.on_loop(L, kv_job, t_fin)
                    if act is not None:
                        merges.append(act)
                if v["stage"] == "admitted":
                    v["stage"] = "merge_pool"          # a crossing loop is never applied before its merge
            if v["stage"] == "admitted" and self.perturb is not None and self.perturb.get("thin") is not None:
                # REPLAY-ONLY arm (CONFIRM's C_THIN, byte for byte): an ADMITTED loop is refused when
                # md5("q,c,seed") maps below p, between admission and the head correction
                p_, seed_ = self.perturb["thin"]
                hq = int(_hl.md5(("%d,%d,%s" % (int(v["q"]), int(v["c"]), seed_)).encode()).hexdigest()[:8], 16)
                if hq / 2.0 ** 32 < p_:
                    v["stage"] = "perturb_thin"
            if v["stage"] == "admitted" and cfg["head"] == "cal" and bool(self.db.hdr[v["q"]]["vi"]) and                     not bool(self.db.hdr[v["c"]]["vi"]):
                # F3-CAL: a loop whose anchor was published by another estimator state than the
                # head (a VI head, a pre-IMU-init VO anchor) belongs to a class with no calibration source: not fused
                v["stage"] = "cal_xstate"
            if v["stage"] == "admitted":
                admitted.append({"k": len(self.attempts), "q": v["q"], "c": v["c"], "Z": v["Z"], "Sl": v["Sl"],
                                 "kv": v["kv"], "inliers": v["inliers"]})
                if cfg["head"] == "cal" and cfg.get("cal_mode", "rel") == "rel" and v.get("r") is not None:
                    if g_cal is None and v["c"] < v["q"]:
                        g_cal = self.chain.rel(v["c"], v["q"])
                    if g_cal is not None:                # F3-CAL: the residual and raw chain the pre-fusion reads
                        admitted[-1]["cal_r"] = np.asarray(v["r"], float)
                        admitted[-1]["cal_Sc"] = g_cal[2]
                        admitted[-1]["cal_g"] = g_cal    # the admission-time chain c -> q (3.1), reused by _prefuse
            self.attempts.append(v)
        self.adm.commit(t_fin)
        if cfg["head"] == "cal":
            self.relcal.commit(t_fin)
        if self._job_covis and cost:                     # R1 co-visibility checks are charged (judge minor 4)
            t_fin += self._job_covis_cost if cost.get("kind") == "ri" else self._job_covis * cost["c_att"]
        out["n_covis"] = self._job_covis
        Tf = lambda k: self.db.hdr[k]["T_odom"]        # noqa: E731
        anchors = lambda c: self.db.hdr[c]["T_map"]    # noqa: E731
        # one merge decision per relation per job: several attempts of one job can each complete an R1 set for the
        # same relation (it becomes provisional only when applied); the LAST one carries the largest pool
        last = {}
        for i, act in enumerate(merges):
            k_ = act.get("merge") or act.get("revert") or act.get("confirm")
            last[tuple(k_)] = i
        merges = [act for i, act in enumerate(merges)
                  if last[tuple(act.get("merge") or act.get("revert") or act.get("confirm"))] == i]
        for act in merges:
            self._apply_merge_action(act, j, p.T_odom, Tf, anchors, kv_r1, t_fin, out)
        if admitted:
            if self.plant and self.plant.get("stage", "post") == "post" and \
                    self.n_events_admitted == int(self.plant.get("event", -1)):
                for L in admitted:
                    L["Z"] = _perturb(L["Z"], self.plant)
                    L["planted"] = True
                out["planted"] = len(admitted)
            ev = self.hf.event(j, p.T_odom, admitted, self.chain, Tf, anchors, admitted[-1]["kv"], t=t_fin)
            self.n_events_admitted += 1
            for L in admitted:                           # F2: a loop the MV head refused (f2_vianchor / unaligned)
                if L.get("f2_refused") and L.get("k") is not None and L["k"] < len(self.attempts):
                    a_ = self.attempts[L["k"]]
                    a_["stage"] = "f2_vianchor" if L["f2_refused"] != "unaligned" else "f2_unaligned"
                    a_["f2_why"] = L["f2_refused"]
                if L.get("cal_refused") and L.get("k") is not None and L["k"] < len(self.attempts):
                    self.attempts[L["k"]]["stage"] = "cal_nis"   # F3-CAL gate (head = "cal" only)
            if ev is not None:
                self._publish(ev, "loop", t_fin, out, [(L["q"], L["c"]) for L in ev["used"]])
        out["t_dec"] = t_fin
        out["t_fin"] = t_fin if cost else self._now()
        out["cpu_ms"] = 1e3 * (tt() - c0)
        self.jobs.append(out)
        return out

    def _plant_pre(self, atts):
        """AUDIT ONLY (judge major 1): perturb the PnP results of the first `n_queries` queries that reach a verified
        attempt with >= 30 inliers, starting at query index plant['from_q'] (pre-admission wrong episode)."""
        pl = self.plant
        for v in atts:
            if v["q"] < int(pl.get("from_q", 0)) or v.get("Z") is None or v["inliers"] < FLOOR:
                continue
            qs = pl.setdefault("_qs", [])
            if v["q"] not in qs:
                if len(qs) >= int(pl.get("n_queries", 3)):
                    continue
                qs.append(v["q"])
            v["Z"] = _perturb(v["Z"], pl)
            v["planted"] = True

    def _publish(self, ev, kind, t_fin, out, loops):
        cost = self._cost()
        # model: decision + the modelled solve; wall: the ACTUAL publish instant on the worker's clock (after admission,
        # R1 co-visibility checks, the merge and the head-filter solve: verifier minor 5)
        f2 = ev.get("f2")
        if cost and f2 is not None and "b_mv" in cost:
            # F2 (F2_DESIGN 3.9): head = "mv" only, the class-C fit of the measured F2 solve against (L + 1)^2 S, frozen
            # before gate B; before that fit exists the frozen 1 ms stands in
            t_pub = t_fin + cost["b_mv"] + cost["a_mv"] * float(f2["work"])
        else:
            t_pub = t_fin + C_SOLVE_S if cost else self._now()
        ms = "none"
        if self.merge.prov:
            ms = "provisional"
        elif self.merge.done:
            ms = "confirmed"
        rec = {"t_pub": t_pub, "t_dec": t_fin, "j": out["j"], "kind": kind, "T_target": ev["T_new"],
               "epoch": self.hf.epoch, "n_loops": len(loops), "loops": loops, "merge_state": ms, "m0": ev["m0"],
               "solve_ms": 1e3 * ev["wall_s"], "planted": out.get("planted", 0), "ep": ev.get("ep"),
               "provisional": bool(ev.get("provisional", False))}
        if f2 is not None:
            rec["f2"] = f2
        out["events"].append(rec)
        self.events.append(rec)
        lin = ev.get("lineage")
        if lin is None:                                  # a revert publish (mv), or anchored mode (computed, unread)
            lin = self.hf.lineage() if isinstance(self.hf, HeadMV) else (len(self.events) - 1, frozenset())
        self._f2_guard(self.lin.on_publish, t_pub, lin, ev["T_new"])
        if isinstance(self.hf, HeadMV) and self.hf.last_info is not None:
            self.lin.metric = self.hf.last_info

    def _apply_merge_action(self, act, j, Tj, Tf, anchors, kv, t_fin, out):
        mm = self.merge
        if "confirm" in act:
            return
        if self.rc is not None:
            self.rc.invalidate()                         # F3B X1: past chain values are about to change
        if "revert" in act:
            key = act["revert"]
            st = mm.undo(key, t_fin)
            self.hf.restore(st["snapshot"])
            self._undo_chain(st["chain_change"])
            self._publish({"T_new": self.hf.T_target.copy(), "m0": 0.0, "wall_s": 0.0}, "merge_revert", t_fin, out, [])
            act = {"merge": key, "C": act["C2"], "t": t_fin}
        key, C = act["merge"], act["C"]
        snap = self.hf.snapshot()
        kind, payload = mm.fuse_merge(key, C, kv)
        if kind == "bridge":
            comp_new, a, b, ZR, Zp, S = payload
            self.chain.add_bridge(comp_new, a, b, ZR, Zp, S)
            change = ("bridge", comp_new)
        else:
            k, ZR, Zp, S = payload
            orig = mm.links[k]["Z"]
            self.chain.set_link(k, ZR, Zp, S, kind="merged")
            change = ("link", k, orig)
        loops = [{"q": x["q"], "c": x["c"], "Z": x["Z"], "Sl": x["Sl"], "kv": x["kv"]} for x in C]
        ev = self.hf.event(j, Tj, loops, self.chain, Tf, anchors, kv or (1.0, 1.0, 0), no_prior=(key[0] == "comp"),
                           t=t_fin)
        mm.set_provisional(key, C, t_fin, snap, change)
        if ev is not None:
            self._publish(ev, "merge", t_fin, out, [(x["q"], x["c"]) for x in C])

    def _undo_chain(self, change):
        if self.rc is not None:
            self.rc.invalidate()                         # F3B X1: past chain values are about to change
        if change[0] == "bridge":
            self.chain.remove_bridge(change[1])
        else:
            _l, k, orig = change
            self.chain.set_link(k, orig[0], orig[1], orig[2], kind="coast")

    def stage_counts(self):
        st = {}
        for v in self.attempts:
            st[v["stage"]] = st.get(v["stage"], 0) + 1
        return st

    def x1_summary(self):
        """F3B X1 report: the running composition's per-job cost (ms, composes) and its counters (head = "cal")."""
        def q(vals):
            v = [x for x in vals if x is not None and np.isfinite(x)]
            if not v:
                return None
            return {"n": len(v), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                    "max": float(max(v)), "sum": float(sum(v))}
        if self.rc is None:
            return {"cal_rc": self.cfg.get("cal_rc", "on"), "active": False}
        return {"cal_rc": self.cfg.get("cal_rc", "on"), "active": True, "rc": self.rc.summary(),
                "advance_ms_per_job": q([jb.get("rc_ms") for jb in self.jobs]),
                "advance_composes_per_job": q([jb.get("rc_nc") for jb in self.jobs]),
                "chain_n_final": int(self.chain.n_final), "chain_n": int(self.chain.n)}

    def l4_summary(self):
        """F3B L4 inventory (F3B_DESIGN X2.3): every structure that can grow with the run, by class, at the end of the
        run (the calibration store's own bound and counters are in f2.cal.relcal).  Measurement only."""
        def ln(get):
            try:
                v = get()
                return int(v) if isinstance(v, (int, np.integer)) else len(v)
            except Exception:                                                # noqa: BLE001
                return None
        hf, a, db, ch, lin, mm = self.hf, self.adm, self.db, self.chain, self.lin, self.merge
        return {
            "head_state": {"prov_loops_max": self._l4_prov[0], "prov_rival_max": self._l4_prov[1]},
            "map": {"chain_edges": ln(lambda: ch.n), "chain_alloc": ln(lambda: ch.final), "chain_prefix": ln(lambda: ch._pR),
                    "chain_fin_waiting": ln(lambda: ch._fin_vals), "db_kf_headers": ln(lambda: db.hdr),
                    "db_bow": ln(lambda: db.bow), "db_candidates": ln(lambda: db.cand),
                    "db_postings": ln(lambda: db.inv.n_postings), "lineage_kf": ln(lambda: lin.b),
                    "lineage_pub": ln(lambda: lin.pub_t), "kf_keypoints": ln(lambda: self.nkp)},
            "fix1_state_logs": {"kappa_samples": ln(lambda: a.kap.samples),
                                "kappaF_samples": ln(lambda: sum(len(v) for v in a.kf.ts.values())),
                                "cycle_ver": ln(lambda: a.cyc.ver), "cycle_samples": ln(lambda: a.cyc.samples),
                                "pcm_list": ln(lambda: a.pcm_list), "sl_hist": ln(lambda: a.sl_hist),
                                "cold_pool": ln(lambda: a.cold_pool), "attempts": ln(lambda: self.attempts),
                                "events": ln(lambda: self.events), "jobs": ln(lambda: self.jobs),
                                "head_walls": ln(lambda: hf.walls), "head_m0_hist": ln(lambda: hf.m0_hist),
                                "head_ep_log": ln(lambda: hf.ep_log), "merge_log": ln(lambda: mm.log),
                                "merge_pools": ln(lambda: mm.pools)},
            "cal_logs": {"pre_log": ln(lambda: hf.pre_log), "gate_log": ln(lambda: hf.gate_log)}}

    def f2_summary(self):
        """F2 report (F2_DESIGN 5.3): refusals, kappa_F states, ledger occupancy and folds, solve time, gains."""
        a = self.adm
        out = {"head": self.cfg["head"], "pcm_cold": self.cfg["pcm_cold"], "ledger_win": int(self.cfg["ledger_win"]),
               "kappaF_samples": {"vi": len(a.kf.ts[CLS_VI]), "vo": len(a.kf.ts[CLS_VO])},
               "cold_pool": {"pooled": a.n_cold_pooled, "rejudged": a.n_cold_rejudged, "joined": a.n_cold_joined},
               "lineage_mixed_kfs": self.lin.n_mix, "f2_errors_swallowed": self.n_f2_err + a.n_f2_err}
        hf = self.hf
        if isinstance(hf, HeadCal):
            out["cal"] = hf.summary()
            out["x1"] = self.x1_summary()
            out["l4"] = self.l4_summary()
        if isinstance(hf, HeadMV):
            fe = hf.f2_events

            def q(vals):
                v = [x for x in vals if x is not None and np.isfinite(x)]
                if not v:
                    return None
                return {"n": len(v), "p10": float(np.percentile(v, 10)), "p50": float(np.percentile(v, 50)),
                        "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)), "max": float(max(v))}
            why = {}
            for r in hf.refusals:
                why[r["why"]] = why.get(r["why"], 0) + 1
            kst = {}
            for e in fe:
                for s, nm in ((0, "vi"), (1, "vo")):
                    for b, bn in ((0, "rot"), (1, "trans")):
                        key = "%s_%s" % (nm, bn)
                        d = kst.setdefault(key, {"cold": 0, "below": 0, "lcb": 0, "absent": 0, "values": []})
                        stt = e["kst"][s][b]
                        d[stt] = d.get(stt, 0) + 1
                        if stt == "lcb":
                            d["values"].append(e["kv"][s][b])
            for d in kst.values():
                v = d.pop("values")
                d["value_p50"] = float(np.median(v)) if v else None
                d["value_p10_p90"] = [float(np.percentile(v, 10)), float(np.percentile(v, 90))] if v else None
            out.update({"events_fused": len(fe), "refusals": len(hf.refusals), "refusals_by_why": why,
                        "refusal_list": hf.refusals[:60], "gain_t": q([e["gain_t"] for e in fe]),
                        "gain_r": q([e["gain_r"] for e in fe]), "solve_ms": q([e["solve_ms"] for e in fe]),
                        "solve_pure_ms": q([e["solve_pure_ms"] for e in fe]), "S_solve": q([e["S"] for e in fe]),
                        "work": q([e["work"] for e in fe]), "S_ledger": q([e["S_ledger"] for e in fe]),
                        "kept_max": max([e["kept"] for e in fe], default=0), "folds": hf.led.n_fold,
                        "splits": hf.led.n_split, "epochs": hf.n_eps, "reverted": len(hf.reverted),
                        "l4_violations": hf.n_l4_viol, "L_max": max([e["L"] for e in fe], default=0),
                        "S_bound_min": min([e["S_bound"] for e in fe], default=None), "kappaF_states": kst,
                        "huber_downweighted": int(sum(sum(1 for w in e["huber_w"] if w < 1.0) for e in fe)),
                        "no_prior_events": int(sum(1 for e in fe if e["no_prior"])),
                        "var_tr": q([e.get("var_tr") for e in fe]),
                        "var_tr_by_decile": [float(np.median([e.get("var_tr", np.nan) for e in fe[int(len(fe) * d / 10):max(int(len(fe) * (d + 1) / 10), int(len(fe) * d / 10) + 1)]])) for d in range(10)] if len(fe) >= 10 else None})
        return out


def _perturb(Z, pl):
    from xclean.slam.geom import so3_exp
    Zw = np.array(Z, float).copy()
    Zw[:3, 3] = Zw[:3, 3] + float(pl.get("dt_m", 1.0)) * np.array([1.0, 0.0, 0.0])
    Zw[:3, :3] = Zw[:3, :3] @ so3_exp(np.radians(float(pl.get("dyaw_deg", 5.0))) * np.array([0.0, 0.0, 1.0]))
    return Zw


def cost_model(tag_or_row, repo=None, scale=1.0, kind=None):
    """The lock-step cost model.
    kind "ri" (DEFAULT since the fix round): the ROW-INDEPENDENT model COST_RI (per-stage p95 scaled by the job's own
    keypoint / database / candidate counts), the same on every row, so lock-step is deterministic everywhere (held-out
    rows, OAK) with no table.
    kind "table" (build 1; used when XCLEAN_P13_COST names a table): the FROZEN per-row table (the loop-closure design: per-stage p95,
    single-thread min-of-3): c_base, c_att, c_hdr; a row the table lacks falls back to "ri"."""
    row = tag_or_row[5:] if tag_or_row.startswith("ver2_") else tag_or_row
    env = os.environ.get("XCLEAN_P13_COST")
    if kind is None:
        kind = "table" if env else "ri"
    if kind == "table":
        repo = repo or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        p = env or os.path.join(repo, COST_TABLE)
        cm = json.load(open(p)).get(row) if os.path.isfile(p) else None
        if cm is not None:
            m3 = cm["min3_ms"]
            return {"c_base": scale * 1e-3 * m3["base_total"]["p95"], "c_att": scale * 1e-3 * m3["per_attempt"]["p95"],
                    "c_hdr": 0.0, "source": p, "row": row}
    out = {k: (v * scale if isinstance(v, float) else v) for k, v in COST_RI.items()}
    out.update(source="COST_RI (row-independent, class C)", row=row)
    return out


# ==================================================================================================== live protocol
class MessageAdapter:
    """The frame side's messages (p13_link WORKER INTERFACE dicts) -> WorkerCore calls.

    kf             -> a KfPacket (worker sequence j = n - 1); its creation-time edge if edge_prev's keyframe is the
                      previous packet, the pending segment LINK for an "init" keyframe, else a MISSING (unknown) link
    edge_final     -> EdgeFinal(j_a) (value only when b is j_a's next packet; candidacy always)
    segment_start  -> the link for the next "init" packet (kind, Z, Sigma or None); the previous segment's keyframes
                      that never left the window by marginalisation become candidates now (the restart dropped them)
    The anchor (the LIVE-MAP pose published for the keyframe's frame) is the message's T_map_cam; a frame side that does
    not send it gets the worker's estimate (its newest target with finish <= the frame stamp, slew ignored), counted."""

    def __init__(self, core, period):
        self.core = core
        self.period = float(period)
        self.sid2j = {}
        self.j_sid = []
        self.link = None
        self.final_seen = set()
        self.n_anchor_fallback = 0
        self.n_edge_mismatch = 0

    def _anchor(self, m, T_odom):
        if m.get("T_map_cam") is not None:
            return np.asarray(m["T_map_cam"], float).reshape(4, 4)
        self.n_anchor_fallback += 1
        t = float(m.get("t_emit_frame") if m.get("t_emit_frame") is not None else m["t"])
        T = np.eye(4)
        for ev in self.core.events:
            if ev["t_pub"] <= t:
                T = ev["T_target"]
        return T @ T_odom

    def on_kf(self, m):
        j = len(self.j_sid)
        T_odom = np.asarray(m["T_odom_cam"], float).reshape(4, 4)
        p = KfPacket(j=j, fidx=int(m["fidx"]), t=float(m["t"]), segment=int(m["segment"]), vi=bool(m["vi"]),
                     T_odom=T_odom, sid=int(m["sid"]))
        p.T_map = self._anchor(m, T_odom)
        p.meta = {"n": int(m["n"]), "slot": m.get("slot"), "kind": m.get("kind"),
                  "t_emit_frame": m.get("t_emit_frame"), "wall_emit": m.get("wall_emit")}
        if j > 0:
            prev_sid = self.j_sid[-1]
            e = m.get("edge_prev")
            if m.get("kind") == "init" and self.link is not None:
                lk = self.link
                if lk.get("prev") is not None and int(lk["prev"]["sid"]) == prev_sid and lk.get("R") is not None:
                    p.edge_kind = lk["kind"]
                    p.ZR = np.asarray(lk["R"], float).reshape(3, 3)
                    p.Zp = np.asarray(lk["p"], float)
                    p.S = None if lk.get("S") is None else np.asarray(lk["S"], float).reshape(6, 6)
                else:
                    p.edge_kind, p.ZR, p.Zp, p.S = "missing", np.eye(3), np.zeros(3), None
            elif e is not None and int(e["sid_a"]) == prev_sid:
                p.edge_kind = "edge"
                p.ZR = np.asarray(e["R"], float).reshape(3, 3)
                p.Zp = np.asarray(e["p"], float)
                p.S = np.asarray(e["S"], float).reshape(6, 6)
            else:
                self.n_edge_mismatch += 1
                p.edge_kind, p.ZR, p.Zp, p.S = "missing", np.eye(3), np.zeros(3), None
        if m.get("kind") == "init":
            self.link = None
        self.sid2j[int(m["sid"])] = j
        self.j_sid.append(int(m["sid"]))
        self.core.on_header(p)
        return p

    def can_final(self, m):
        return int(m["sid_a"]) in self.sid2j and int(m.get("sid_b", -1)) in self.sid2j

    def on_edge_final(self, m):
        ja = self.sid2j.get(int(m["sid_a"]))
        jb = self.sid2j.get(int(m["sid_b"]))
        if ja is None:
            return
        self.final_seen.add(ja)
        if jb is not None and jb == ja + 1:
            self.core.on_final(EdgeFinal(ja, np.asarray(m["R"], float).reshape(3, 3), np.asarray(m["p"], float),
                                         np.asarray(m["S"], float).reshape(6, 6)), _t_emit(m))
        else:
            self.core.on_final(EdgeFinal(ja, None, None, None), _t_emit(m))

    def on_segment_start(self, m):
        self.link = m
        for j in range(len(self.j_sid)):                 # the restart dropped the old window: its KFs left it now
            if j not in self.final_seen:
                self.final_seen.add(j)
                self.core.on_final(EdgeFinal(j, None, None, None), _t_emit(m))


def _t_emit(m):
    """The stamp of the frame whose publish emitted message m (-inf when the frame side did not say)."""
    t = m.get("t_emit_frame")
    return -np.inf if t is None else float(t)


def worker_main(handles, wcfg):
    """The LIVE worker process (spawned by xclean/live/p13_link.start_worker).  CPU only, one thread, BELOW_NORMAL;
    warms up, signals ready, then: FIFO over the ctl queue; a "kf" message runs its job (with the images from the ring,
    freed at once); results are published through the seqlock + result queue; progress(n) after every job."""
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        import ctypes
        if sys.platform.startswith("win"):
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        else:
            os.nice(5)                            # below normal, as on Windows: the engine keeps priority
    except Exception:                                                        # noqa: BLE001
        pass
    import cv2
    cv2.setNumThreads(1)
    from xclean.live.p13_link import WorkerSide
    from xclean.core.derive import Rig
    from xclean.slam.rich import RigParams
    side = WorkerSide(handles)
    calib = wcfg["calib"]
    tag = wcfg.get("tag", "live")
    rig_full = Rig.from_calib_dict(calib, name=tag)
    W, H = int(calib["camera_width"]), int(calib["camera_height"])
    rig = RigParams.from_rig(rig_full, W, H)
    mode = wcfg.get("mode", "paced")
    cost = cost_model(tag) if mode == "lockstep" else None   # lock-step: row-independent model (or XCLEAN_P13_COST table)
    cfg = {"clock": "model" if cost else "wall", "cost": cost}
    out_dir = wcfg.get("out_dir")
    core = WorkerCore(cfg, rig=rig, store_path=os.path.join(out_dir, "p13_richsets.bin") if out_dir else None)
    period = 1.0 / float(getattr(rig_full, "fps", 0) or calib.get("fps") or 20.0)
    ad = MessageAdapter(core, period)
    t_free = -np.inf
    if hasattr(core.front, "ex"):                      # warm-up before ready: ORB, BoW and the numba 1-D stereo compile
        rng = np.random.default_rng(0)
        Lw = rng.integers(0, 255, (H, W), dtype=np.uint8)
        core.front.ex.extract(Lw, np.roll(Lw, -8, axis=1))
    side.ready()
    t_wall0 = time.perf_counter()
    buf = []
    log = open(os.path.join(out_dir, "p13_worker_jobs.jsonl"), "w", encoding="utf-8") if out_dir else None
    st = {"eos": False, "jobs": 0, "published": 0, "errors": []}

    def pull(block):
        try:
            m = side.get(timeout=0.5 if block else 0.0005)
        except Exception:                                                    # noqa: BLE001
            m = None
        while m is not None:
            buf.append(m)
            m = side.get(timeout=0.0) if not block else side.get(timeout=0.0005)

    def apply_until_kf():
        """Apply non-kf messages in order; return the next kf message (removed from the buffer) or None."""
        while buf:
            m = buf.pop(0)
            ty = m.get("type")
            if ty == "kf":
                return m
            if ty == "edge_final":
                ad.on_edge_final(m)
            elif ty == "segment_start":
                ad.on_segment_start(m)
            elif ty == "eos":
                st["eos"] = True
        return None

    while not st["eos"]:
        if not buf:
            pull(True)
        m = apply_until_kf()
        if m is None:
            continue
        p = ad.on_kf(m)
        slot = m.get("slot")
        if slot is not None:
            L_, R_ = side.images(slot)
            side.free(slot)
            p.left, p.right = L_, R_
        t_emit = float(m.get("t_emit_frame") if m.get("t_emit_frame") is not None else m["t"])

        def finals_ahead():
            """edge_final messages further down the queue for keyframes the worker already has: applied now (they
            have arrived; kf headers and segment messages stay in place).  The replay does the same (finals_ahead)."""
            keep = []
            for x in buf:
                if x.get("type") == "edge_final" and ad.can_final(x):
                    ad.on_edge_final(x)
                else:
                    keep.append(x)
            buf[:] = keep
        if cost:                                       # lock-step: the cost model on the frame-stamp clock
            t_start = max(t_emit + period, t_free)
            feed = None
        else:                                          # paced: the wall clock mapped onto the frame-stamp clock
            t_start = t_emit + (time.perf_counter() - float(m.get("wall_emit") or time.perf_counter()))
            pull(False)                                # drain the queue BEFORE counting the waiting packets (verifier
            finals_ahead()                             # minor 6: packets that arrived during the last job count)

            def feed(_t):
                pull(False)
                finals_ahead()
        waiting = sum(1 for x in buf if x.get("type") == "kf")
        try:
            out = core.job(p, t_start, waiting, feed=feed)
        except Exception as e:                                               # noqa: BLE001
            st["errors"].append([p.j, repr(e)[:300]])
            out = {"t_fin": t_start, "events": [], "j": p.j}
        t_free = out["t_fin"]
        for ev in out["events"]:
            side.publish(ev["epoch"], ev["T_target"], KIND_CODE[ev["kind"]], ev["n_loops"],
                         MERGE_CODE.get(ev["merge_state"], 0), ev["t_pub"])
            st["published"] += 1
        side.progress(int(m["n"]))
        st["jobs"] += 1
        if log is not None:
            rec = {k: out.get(k) for k in ("j", "t_kf", "t_start", "t_fin", "shed", "dropped", "waiting", "n_att",
                                           "n_covis", "cpu_ms", "stage_ms", "n_db", "postings")}
            rec["events"] = [{k: ev.get(k) for k in ("t_pub", "kind", "epoch", "n_loops", "merge_state", "ep",
                                                     "provisional")} for ev in out["events"]]
            rec["r1_scale"] = out.get("r1_scale")
            log.write(json.dumps(rec, default=float) + "\n")
    summ = {"jobs": st["jobs"], "published": st["published"], "errors": st["errors"][:20], "shed": core.n_shed,
            "dropped": core.n_dropped, "attempts": len(core.attempts), "stages": core.stage_counts(),
            "merges": core.merge.log, "anchor_fallback": ad.n_anchor_fallback, "edge_mismatch": ad.n_edge_mismatch,
            "clock": cfg["clock"], "cost_model": cost, "wall_s": time.perf_counter() - t_wall0,
            "db_bytes": core.db.store.bytes, "covis_checks": core.merge.n_covis,
            "config": {k: core.cfg[k] for k in ("episodes", "kappa_feed", "r1_cold", "prior_mode")},
            "episodes": {"opened": core.hf.n_prov, "confirmed": core.hf.n_confirm, "swaps": core.hf.n_swap,
                         "dropped_by_merge": core.hf.n_drop, "open_at_eos": core.hf.prov is not None},
            "kappa_feed_gated": core.adm.n_feed_gated, "kappa_cyc_samples": len(core.adm.cyc.samples),
            "f2": core.f2_summary()}
    if out_dir:
        json.dump(summ, open(os.path.join(out_dir, "p13_worker.json"), "w"), indent=1, default=str)
    if log is not None:
        log.close()
    core.db.store.close()
    if out_dir:                                        # the rich-set store is the worker's scratch (re-made per run)
        try:
            os.remove(os.path.join(out_dir, "p13_richsets.bin"))
        except OSError:
            pass
    assert "cupy" not in sys.modules, "the P1.3 worker must stay CPU-only"
    side.close()
    return 0


_worker_main = worker_main            # the name the tests used before the entry point was exposed
