# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.3 OFFLINE CAUSAL REPLAY: drives `WorkerCore` exactly as the live integration does, on a
recorded P1 run, on a virtual clock (the frame stamps).

    live (xclean/live/p13_link.py + xclean/slam/worker.py)        replay (this module)
    ---------------------------------------------------------      ------------------------------------------------------
    H6 emits a frame's messages after its LIVE-MAP publish:        the same messages, visible at t_frame + emit_periods
      edge_final / segment messages first, the kf header last        camera periods, in the same FIFO order
      (the loop-closure build's hook spec, item S2)
    KfRing, 8 slots; full -> images dropped, header kept           occupancy = emitted - started; >= 8 -> images dropped
    worker job: FIFO; >= 2 packets waiting at the start -> shed    same code (WorkerCore.job), waiting = packets visible at
                                                                     the job's start
    paced: messages arriving during the job are drained before     feed(t_fin): the non-kf messages after the job's own
      admission                                                      header with visibility <= t_fin (sync "paced")
    lock-step (the frame side waits for every emitted packet):     sync "lockstep": no mid-job messages, no backlog seen
      no mid-job messages, no shedding
    job duration: the frozen cost model (lock-step) / wall         the frozen cost model (clock "model") or the measured
                                                                     thread CPU time (clock "wall")
    compose(): results with finish <= t (lock-step), slew_step     the same rule and p13_link.slew_step (IMPORTED), the
      at the B4 rate, the anchor = the LIVE-MAP just published       NET 2-s displacement / rotation (p13_link.compose since
                                                                     HOOK_SPEC S3 landed: "net", DEFAULT) or build 1's
                                                                     summed steps ("sum", arm)
LIVE-MAP(k) = T_applied(k) LIVE-ODOM(k): causal, published once per frame, never revised.

Two inputs (fix round 2026-10-26, verifier minor 4):
  run_log(LiveLog)      THE PREDICTION PATH: the live frame side's own messages (p13_packets.jsonl of a harness run),
                        fed through MessageAdapter exactly as worker_main does; lock-step = the live lock-step protocol
                        (equal to the live LIVE-MAP byte for byte for an unchanged worker: test_p13_replay_equiv), paced =
                        the paced model below
  run(TraceStream)      the design trace (creation-time edges from one solve later than H2, finals by the window rule):
                        kept as an arm, its deviation from run_log is reported per row
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from xclean.slam.geom import so3_log


class FrameSide:
    """The frame process's compose() (xclean/live/p13_link.py) on the replay clock."""

    def __init__(self, f=None, apply=None, bound="net"):
        from xclean.live import p13_link as PL
        self.PL = PL
        self.f = PL.SLEW_F if f is None else float(f)
        self.apply = PL.APPLY if apply is None else apply
        self.bound = bound
        self.hist = deque()
        self.t_prev = None
        self.target = None
        self.T_applied = None
        self.n_corr = 0

    def step(self, t, T_odom):
        PL = self.PL
        dt = 0.0 if self.t_prev is None else float(t) - self.t_prev
        if self.hist:
            _tp, pp, Rp, _s, _r = self.hist[-1]
            step = float(np.linalg.norm(T_odom[:3, 3] - pp))
            rot = math.degrees(float(np.linalg.norm(PL._so3_log(Rp.T @ T_odom[:3, :3]))))
            self.hist.append((float(t), T_odom[:3, 3].copy(), T_odom[:3, :3].copy(), step, rot))
        else:
            self.hist.append((float(t), T_odom[:3, 3].copy(), T_odom[:3, :3].copy(), 0.0, 0.0))
        while self.hist and self.hist[0][0] < float(t) - PL.B4_WINDOW_S:
            self.hist.popleft()
        if self.bound == "net":                         # the design prototype's form (net displacement / rotation)
            h0, h1 = self.hist[0], self.hist[-1]
            path_2s = float(np.linalg.norm(h1[1] - h0[1]))
            rot_2s = math.degrees(float(np.linalg.norm(PL._so3_log(h0[2].T @ h1[2]))))   # p13_link.compose's own log
        else:                                           # p13_link.compose: summed steps
            hl = list(self.hist)[1:]
            path_2s = sum(h[3] for h in hl)
            rot_2s = sum(h[4] for h in hl)
        self.t_prev = float(t)
        extra = {"map_epoch": 0, "map_corr": False, "map_pending_m": 0.0, "map_pending_deg": 0.0, "map_merged": "none"}
        if self.target is None:
            return T_odom, extra, True
        Tt = self.target["T_target"]
        Ta = np.eye(4) if self.T_applied is None else self.T_applied
        Tn, pm, pd, moved = PL.slew_step(Ta, Tt, T_odom, dt, path_2s, rot_2s, f=self.f, apply=self.apply)
        if moved or self.T_applied is not None:
            self.T_applied = Tn
        if moved:
            self.n_corr += 1
        extra.update(map_epoch=int(self.target["epoch"]), map_corr=bool(moved), map_pending_m=float(pm),
                     map_pending_deg=float(pd), map_merged=self.target["merge_state"], map_kind=self.target["kind"])
        reached = (pm == 0.0 and pd == 0.0)
        if self.T_applied is None:
            return T_odom, extra, reached
        return self.T_applied @ T_odom, extra, reached


def run(stream, core, images_fn=None, emit_periods=1.0, ring_slots=8, sync="paced", frame_side=None, poison=None,
        progress=None):
    """Replay `stream` (packets.TraceStream or a synthetic equivalent) through `core` (WorkerCore).
    images_fn(fidx) -> (left, right) rectified, or None (a front that needs no images).
    poison: {"j": m, "fn": callable(packet)} mutates packet m before its emission (the causality test).
    -> dict(LM (F,4,4), extras, jobs, events, latencies, backlog, ...)."""
    fr_t, fr_T = stream.fr_t, stream.fr_T
    F = len(fr_t)
    period = float(np.median(np.diff(fr_t))) if F > 1 else 0.05
    emit_dt = emit_periods * period
    fs = frame_side or FrameSide()
    pos_kf = {int(f): j for j, f in enumerate(stream.kf_frame)}
    finals_by_j = {}
    for (j_emit, m) in stream.finals:
        finals_by_j.setdefault(int(j_emit), []).append(m)
    msgs = []                     # FIFO: (t_vis, kind, payload)
    mi = 0                        # next message to deliver
    emitted = []                  # packets in emission order (t_vis set)
    jn = 0
    t_free = -np.inf
    pend = []                     # published events not yet applicable
    LM = np.zeros_like(fr_T)
    extras = []
    lat_target, lat_done = {}, {}
    awaiting_done = []
    backlog = []
    ring_occ = []
    n_ring_drop = 0

    done_msg = []                 # delivered flags (parallel to msgs)
    cur_j = [-1]

    def deliver(i):
        m = msgs[i]
        done_msg[i] = True
        if m[1] == "final":
            core.on_final(m[2], m[3])                    # emitted with the frame stamped m[3]
        else:
            core.on_header(m[2])

    def finals_ahead(t_lim):
        """edge_final messages already visible (t_vis <= t_lim) further down the FIFO, for keyframes whose header the
        worker has (a + 1 <= the current job): delivered early, as the live worker drains them (kf headers stay in
        place: a header is applied when its job starts)."""
        for i in range(mi, len(msgs)):
            m = msgs[i]
            if m[0] > t_lim:
                break
            if not done_msg[i] and m[1] == "final" and m[2].a + 1 <= cur_j[0]:
                deliver(i)

    def feed_until_header(j, s):
        nonlocal mi
        cur_j[0] = j
        while mi < len(msgs):
            i = mi
            mi += 1
            if done_msg[i]:
                continue
            deliver(i)
            if msgs[i][1] == "kf" and msgs[i][2].j == j:
                break
        if sync == "paced":
            finals_ahead(s)

    def feed_mid(t_fin):
        if sync != "paced":
            return
        finals_ahead(t_fin)

    for k in range(F):
        t = float(fr_t[k])
        # 1. every job that starts at or before this frame's stamp
        while jn < len(emitted):
            p = emitted[jn]
            s = max(p.meta["t_vis"], t_free)
            if s > t:
                break
            feed_until_header(p.j, s)
            if sync == "paced":
                waiting = sum(1 for q in emitted[jn + 1:] if q.meta["t_vis"] <= s)
            else:
                waiting = 0
            backlog.append(waiting)
            if not p.meta.get("ring_drop") and images_fn is not None:
                im = images_fn(p.fidx)
                if im is not None:
                    p.left, p.right = im
            out = core.job(p, s, waiting, feed=feed_mid)
            p.left = p.right = None
            t_free = out["t_fin"]
            jn += 1
            pend.extend(out["events"])
            if progress is not None:
                progress(jn, len(stream.packets))
        # 2. results applicable at this frame (lock-step rule: finish <= t), in publication order
        keep = []
        newly = []
        for ev in pend:
            if ev["t_pub"] <= t:
                fs.target = ev
                newly.append(ev)
            else:
                keep.append(ev)
        pend = keep
        for ev in newly:
            for (q, _c) in ev["loops"]:
                lat_target.setdefault(q, t - float(stream.kf_t[q]))
            awaiting_done.append(ev)
        # 3. LIVE-MAP
        T_map, ex, reached = fs.step(t, fr_T[k])
        LM[k] = T_map
        extras.append(ex)
        if reached and awaiting_done:
            for ev in awaiting_done:
                for (q, _c) in ev["loops"]:
                    lat_done.setdefault(q, t - float(stream.kf_t[q]))
            awaiting_done = []
        # 4. emission (after the publish): finals of this frame first, the kf header last
        j = pos_kf.get(int(stream.fr_idx[k]))
        if j is not None:
            p = stream.packets[j]
            if poison is not None and poison["j"] == j:
                poison["fn"](p)
            p.T_map = T_map.copy()
            p.meta["t_vis"] = t + emit_dt
            occ = sum(1 for q_ in emitted[jn:] if not q_.meta.get("ring_drop"))     # slots held (a dropped packet holds none)
            ring_occ.append(occ)
            if occ >= ring_slots:
                p.meta["ring_drop"] = True
                n_ring_drop += 1
            for m in finals_by_j.get(j, []):
                msgs.append((t + emit_dt, "final", m, t))
                done_msg.append(False)
            msgs.append((t + emit_dt, "kf", p, t))
            done_msg.append(False)
            emitted.append(p)
    # jobs still queued at the end of the stream: run them (their results apply to no frame; counted)
    n_tail = len(emitted) - jn
    return {"LM": LM, "extras": extras, "lat_target": lat_target, "lat_done": lat_done, "backlog": backlog,
            "ring_occ": ring_occ, "ring_drops": n_ring_drop, "jobs_unrun_at_eos": n_tail, "period": period,
            "emit_dt": emit_dt, "frames_corrected": fs.n_corr}


def run_log(log, core, images_fn=None, emit_periods=1.0, ring_slots=8, sync="paced", frame_side=None, poison=None,
            progress=None, period=None, cached_images=False, job_errors=None):
    """Replay a LIVE run's message log (packets.LiveLog) through `core` exactly as worker_main consumes it.
    job_errors (F2 build, replay-only; None = DEFAULT, an exception ends the replay): a list -> a job that raises is
    handled as worker_main handles it live (the error is recorded, the job publishes nothing, the worker is free at the
    job's start), so a run whose worker fails mid-way still yields the LIVE-MAP the live system would publish.

    lockstep  the live lock-step protocol (p13_link._sync_lockstep): before frame k is composed, every message emitted by
              frames < k has been processed in FIFO order; a kf job starts at max(t_emit + period, t_free) with nothing
              waiting and no mid-job message (worker_main's lock-step branch, with period = 1 / fps as there)
    paced     the paced model: a message is visible emit_periods camera periods after its frame; the worker is FIFO; a kf
              job starts at max(visible, worker free) and, as worker_main (since the minor-6 fix), first drains the
              queue (edge_final messages for keyframes it has are applied at once), THEN counts the kf packets waiting
              (shedding at >= 2); finals arriving during the job are applied before its admission (feed); the 8-slot
              ring: a kf message emitted while 8 image packets are held loses its images (header-only job, counted)
    The anchor of every kf message is replaced by this replay's own LIVE-MAP of its emitting frame (S1); everything else
    (LIVE-ODOM, edges, finals, links) is the live frame side's.  poison: {"j": n, "fn": callable(message dict)} mutates
    the n-th kf message before its emission (causality test).
    -> dict as run()."""
    from xclean.slam.worker import MessageAdapter
    fr_t, fr_T = log.fr_t, log.fr_T
    F = len(fr_t)
    if period is None:
        period = float(np.median(np.diff(fr_t))) if F > 1 else 0.05
    emit_dt = emit_periods * period
    fs = frame_side or FrameSide()
    ad = MessageAdapter(core, period)
    by_frame = {}
    for i, k in enumerate(log.msg_frame):
        by_frame.setdefault(0 if k is None else int(k), []).append(i)
    Q = deque()                       # emitted, not yet consumed: (t_vis, message)
    st = {"t_free": -np.inf, "held": 0, "n_kf": 0, "jobs": 0}
    pend, LM, extras = [], np.zeros_like(fr_T), []
    lat_target, lat_done = {}, {}
    awaiting_done, backlog, ring_occ = [], [], []
    n_ring_drop = 0

    def apply_msg(m):
        ty = m.get("type")
        if ty == "edge_final":
            ad.on_edge_final(m)
        elif ty == "segment_start":
            ad.on_segment_start(m)

    def finals_ahead(t_lim):
        keep = deque()
        for (tv, m) in Q:
            if tv <= t_lim and m.get("type") == "edge_final" and ad.can_final(m):
                ad.on_edge_final(m)
            else:
                keep.append((tv, m))
        Q.clear()
        Q.extend(keep)

    def start_job(m, s, paced):
        p = ad.on_kf(m)
        drop = m.get("slot") is None
        p.meta["ring_drop"] = bool(drop)
        if cached_images and not drop:
            p.meta["cached_images"] = True
        if not drop:
            st["held"] -= 1                               # the slot is freed when the job copies the images
            if images_fn is not None:
                im = images_fn(p.fidx)
                if im is not None:
                    p.left, p.right = im
        if paced:
            finals_ahead(s)
            waiting = sum(1 for (tv, x) in Q if tv <= s and x.get("type") == "kf")
            feed = finals_ahead
        else:
            waiting, feed = 0, None
        backlog.append(waiting)
        if job_errors is None:
            out = core.job(p, s, waiting, feed=feed)
        else:
            try:
                out = core.job(p, s, waiting, feed=feed)
            except Exception as e:                                           # noqa: BLE001  (worker_main's rule)
                job_errors.append([p.j, repr(e)[:300]])
                out = {"t_fin": s, "events": [], "j": p.j}
        p.left = p.right = None
        st["t_free"] = out["t_fin"]
        st["jobs"] += 1
        pend.extend(out["events"])
        if progress is not None:
            progress(st["jobs"], len(log.kf_t))

    for k in range(F):
        t = float(fr_t[k])
        # 1. the worker
        if sync == "lockstep":
            while Q:
                _tv, m = Q.popleft()
                if m.get("type") == "kf":
                    start_job(m, max(float(m["t_emit_frame"]) + period, st["t_free"]), False)
                else:
                    apply_msg(m)
        else:
            while True:
                idx = next((i for i, (_tv, m) in enumerate(Q) if m.get("type") == "kf"), None)
                if idx is None:
                    break
                s = max(Q[idx][0], st["t_free"])
                if s > t:
                    break
                for _ in range(idx):
                    apply_msg(Q.popleft()[1])
                _tv, mk = Q.popleft()
                start_job(mk, s, True)
        # 2. results applicable at this frame (finish <= t), in publication order (the newest wins)
        keep, newly = [], []
        for ev in pend:
            if ev["t_pub"] <= t:
                fs.target = ev
                newly.append(ev)
            else:
                keep.append(ev)
        pend = keep
        for ev in newly:
            for (q, _c) in ev["loops"]:
                lat_target.setdefault(q, t - float(log.kf_t[q]))
            awaiting_done.append(ev)
        # 3. LIVE-MAP
        T_map, ex, reached = fs.step(t, fr_T[k])
        LM[k] = T_map
        extras.append(ex)
        if reached and awaiting_done:
            for ev in awaiting_done:
                for (q, _c) in ev["loops"]:
                    lat_done.setdefault(q, t - float(log.kf_t[q]))
            awaiting_done = []
        # 4. emission of this frame's messages (after its publish; the kf message last, as the live emit sorts them)
        for i in by_frame.get(k, []):
            m = dict(log.msgs[i])
            if m.get("type") == "kf":
                jn = st["n_kf"]
                st["n_kf"] += 1
                if poison is not None and poison["j"] == jn:
                    poison["fn"](m)
                m["T_map_cam"] = T_map.ravel().tolist()
                if sync != "lockstep":
                    ring_occ.append(st["held"])
                    if st["held"] >= ring_slots:
                        m["slot"] = None
                        n_ring_drop += 1
                    else:
                        m["slot"] = st["held"]
                        st["held"] += 1
                elif m.get("slot") is not None:
                    st["held"] += 1
            Q.append(((t + emit_dt) if sync != "lockstep" else t, m))
    n_tail = sum(1 for (_tv, m) in Q if m.get("type") == "kf")
    return {"LM": LM, "extras": extras, "lat_target": lat_target, "lat_done": lat_done, "backlog": backlog,
            "ring_occ": ring_occ, "ring_drops": n_ring_drop, "jobs_unrun_at_eos": n_tail, "period": period,
            "emit_dt": emit_dt, "frames_corrected": fs.n_corr}
