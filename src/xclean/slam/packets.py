# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.3 messages between the frame process and the worker, and the packet stream of a RECORDED P1 run
(the offline causal replay's input).

Live (frame process -> worker, emitted by link.emit() after the frame's LIVE-MAP publish; never blocking):
  KfPacket    one per keyframe: header (worker sequence j, engine slot id, frame index, stamp, segment, VI flag), the
              LIVE-ODOM pose published for the keyframe's frame, the LIVE-MAP pose published for it (the ANCHOR), the
              CREATION-TIME odometry edge j-1 -> j (or the segment LINK when j starts a segment), the rectified pair
              (None when the ring was full: header kept, images dropped, counted)
  EdgeFinal   the marginalisation-time value of edge a -> a+1 (keyframe a left P1's window: it becomes a loop candidate)
All poses T_world_cam0_rectified; relative poses in the rectified camera frame (rotation right, translation additive in
the left frame, covariance [phi, t]; CONVENTIONS.md, the loop-closure design).

Replay: `stream_from_trace(run_dir)` builds the same messages from a P1 lock-step run that carries the design track's
read-only edge trace.  The
keyframe table, the edge conversion and the link rules are the design's (p13_kfstreams.kf_table, p13_drift_cov.
load_edges) so the replay sees exactly the design evidence's chain.  The marginalisation time of a keyframe is not in
the trace: the replay emits edge_final(a) with packet a + N_KF (the window rule; live, hook H3 gives the exact time).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from xclean.slam.geom import body_to_cam_rel

N_KF = 7


@dataclass
class KfPacket:
    j: int                                   # worker sequence number (0, 1, 2, ... over the run)
    fidx: int                                # frame index
    t: float                                 # frame stamp (s)
    segment: int
    vi: bool
    T_odom: np.ndarray                       # LIVE-ODOM published for this frame (4x4)
    T_map: Optional[np.ndarray] = None       # LIVE-MAP published for this frame = the anchor (set at emit)
    edge_kind: str = "first"                 # "first" (no previous KF), "edge", or a link kind (demote, coast,
    #                                          coast_vi, unknown, missing)
    ZR: Optional[np.ndarray] = None          # edge / link j-1 -> j (camera frame); None for "first"
    Zp: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None           # its covariance [phi, t]; None = UNKNOWN link
    left: Optional[np.ndarray] = None
    right: Optional[np.ndarray] = None
    sid: int = -1                            # engine slot id (live)
    meta: dict = field(default_factory=dict)


@dataclass
class EdgeFinal:
    a: int                                   # edge a -> a+1, keyframe a becomes a candidate
    ZR: Optional[np.ndarray]
    Zp: Optional[np.ndarray]
    S: Optional[np.ndarray]


def _kf_table(odo, fr):
    """The design's keyframe rule (p13_kfstreams.kf_table): a frame with the "kf" key, or the first frame of a segment
    (its KF0, which carries no "kf" key), that has a LIVE-ODOM pose."""
    kf_f, kf_t, kf_seg, kf_vi = [], [], [], []
    last_seg = None
    for r in fr:
        seg = r["segment"]
        is_kf = bool(r.get("kf")) or (seg != last_seg)
        last_seg = seg
        if is_kf and r["frame"] in odo:
            kf_f.append(r["frame"])
            kf_t.append(r["t"])
            kf_seg.append(seg)
            kf_vi.append(bool(r["vi"]))
    return np.array(kf_f, np.int64), np.array(kf_t), np.array(kf_seg, np.int64), np.array(kf_vi, bool)


def read_recorder(path, stream="LIVE-ODOM"):
    out = {}
    for ln in open(path, encoding="utf-8"):
        if not ln.strip():
            continue
        m = json.loads(ln)
        if m.get("stream") == stream:
            out[m["frame_index"]] = m
    return out


def _edges_from_trace(d):
    """p13_edges.json -> {(fa, fb): (first (R, p, S), final (R, p, S), vi)} and links {(fa, fb): (kind, R, p, S|None)}
    in the rectified camera frame (the design's p13_drift_cov.load_edges, same arithmetic)."""
    Rbc = np.array(d["meta"]["Rbc"], float)
    pbc = np.array(d["meta"]["pbc"], float)
    E = {}
    for e in d["edges"]:
        fin = body_to_cam_rel(np.array(e["Z_R"]), np.array(e["Z_p"]), np.array(e["S"]), Rbc, pbc)
        if "Z_R_first" in e:
            fst = body_to_cam_rel(np.array(e["Z_R_first"]), np.array(e["Z_p_first"]), np.array(e["S_first"]), Rbc,
                                  pbc)
        else:
            fst = fin
        E[(int(e["fa"]), int(e["fb"]))] = (fst, fin, bool(e["vi"]))
    L = {}
    for lk in d.get("links", []):
        if lk.get("prev_kf") is None or lk.get("kf0") is None:
            continue
        Ra, pa = np.array(lk["prev_kf"]["R"]), np.array(lk["prev_kf"]["p"])
        Rb, pb = np.array(lk["kf0"]["R"]), np.array(lk["kf0"]["p"])
        ZR = Ra.T @ Rb
        Zp = Ra.T @ (pb - pa)
        S = None
        if lk.get("sig_caller") is not None:
            Sg = np.array(lk["sig_caller"], float)
            if lk["caller"] == "_demote":
                S = Sg                                         # already relative (rel_cov, frame a)
            else:                                              # absolute covariance of T_start -> relative to frame a
                Jb = np.zeros((6, 6))
                Jb[:3, :3] = np.eye(3)
                Jb[3:, 3:] = Ra.T
                S = Jb @ Sg @ Jb.T
        cR, cp, cS = body_to_cam_rel(ZR, Zp, S, Rbc, pbc)
        kind = {"_demote": "demote", "_coast": "coast", "_coast_vi": "coast_vi", "_step": "unknown"}.get(lk["caller"],
                                                                                                         lk["caller"])
        L[(int(lk["prev_kf"]["fidx"]), int(lk["kf0"]["fidx"]))] = (kind, cR, cp, cS)
    return E, L, Rbc, pbc


class LiveLog:
    """The messages a LIVE run's frame side emitted (p13_packets.jsonl of a main-builder harness run with the loop ON:
    segment_start / kf / edge_final / eos dicts in emission order, each stamped with its frame, t_emit_frame) and the
    run's LIVE-ODOM frames (recorder.jsonl).  The replay input of the fix round (verifier minor 4): every prediction
    is made from what the live frame side really sends (H2's creation-time edges from the keyframe's own solve, H3's
    finals at the real marginalisation, the real segment links), not from the design trace.  LIVE-ODOM, the edges, the
    finals and the links do not depend on the worker; the ANCHOR (T_map_cam) does, so the replay replaces it with
    its own LIVE-MAP of the emitting frame, and the ring slot with its own ring model (paced).

    fr_idx, fr_t, fr_T, fr_msgs : LIVE-ODOM frames (as TraceStream);  msgs : the messages;  kf_frame / kf_t / kf_seg /
    kf_vi / kf_T : per keyframe in worker sequence (the order of the kf messages);  counts : link kinds."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        rec = read_recorder(os.path.join(run_dir, "recorder.jsonl"))
        ks = sorted(rec)
        self.fr_idx = np.array(ks, np.int64)
        self.fr_t = np.array([rec[k]["t_pose"] for k in ks], float)
        self.fr_T = np.stack([np.array(rec[k]["T"], float).reshape(4, 4) for k in ks])
        self.fr_state = [rec[k]["state"] for k in ks]
        self.fr_msgs = rec
        self.msgs = [json.loads(x) for x in open(os.path.join(run_dir, "p13_packets.jsonl"), encoding="utf-8")
                     if x.strip()]
        kfs = [m for m in self.msgs if m.get("type") == "kf"]
        self.kf_frame = np.array([int(m["fidx"]) for m in kfs], np.int64)
        self.kf_t = np.array([float(m["t"]) for m in kfs])
        self.kf_seg = np.array([int(m["segment"]) for m in kfs], np.int64)
        self.kf_vi = np.array([bool(m["vi"]) for m in kfs], bool)
        self.kf_T = np.stack([np.array(m["T_odom_cam"], float).reshape(4, 4) for m in kfs]) if kfs else np.zeros((0, 4, 4))
        # the emitting frame of every message (its t_emit_frame is that frame's LIVE-ODOM stamp)
        pos = {float(t): i for i, t in enumerate(self.fr_t)}
        self.msg_frame = []
        n_bad = 0
        for m in self.msgs:
            te = m.get("t_emit_frame")
            k = pos.get(float(te)) if te is not None else None
            if k is None and te is not None:
                k = int(np.searchsorted(self.fr_t, float(te), side="left"))
                n_bad += 1
            self.msg_frame.append(k)
        segs = [m for m in self.msgs if m.get("type") == "segment_start"]
        self.counts = {"kfs": len(kfs), "segment_starts": len(segs), "emit_frame_inexact": n_bad,
                       "link_kinds": sorted({str(m.get("kind")) for m in segs[1:]}),
                       "finals": sum(1 for m in self.msgs if m.get("type") == "edge_final")}
        self.pos_f = {int(f): i for i, f in enumerate(self.fr_idx)}


class TraceStream:
    """The packets, edge_final messages and frames of a recorded P1 run (lock-step) with the design's edge trace.

    frames      : idx (F,), t (F,), T_odom (F,4,4) (LIVE-ODOM), state (F,) str, has_odom
    packets     : list of KfPacket (no images, no anchor: the replay fills both)
    finals      : list of (j_emit, EdgeFinal): the message is emitted with packet j_emit (window rule)
    """

    def __init__(self, run_dir, edges_json=None):
        self.run_dir = run_dir
        rec = read_recorder(os.path.join(run_dir, "recorder.jsonl"))
        fr = [json.loads(x) for x in open(os.path.join(run_dir, "p1_frames.jsonl"), encoding="utf-8") if x.strip()]
        odo = {k: np.array(m["T"], float).reshape(4, 4) for k, m in rec.items()}
        ks = sorted(rec)
        self.fr_idx = np.array(ks, np.int64)
        self.fr_t = np.array([rec[k]["t_pose"] for k in ks], float)
        self.fr_T = np.stack([odo[k] for k in ks])
        self.fr_state = [rec[k]["state"] for k in ks]
        self.fr_msgs = rec
        kf_f, kf_t, kf_seg, kf_vi = _kf_table(odo, fr)
        ep = edges_json or os.path.join(run_dir, "p13_edges.json")
        d = json.load(open(ep, encoding="utf-8"))
        E, L, self.Rbc, self.pbc = _edges_from_trace(d)
        # RE-ANCHOR keyframes (P1.1 D2: an IMU-carried frame becomes a keyframe without _make_kf, so p1_frames has no
        # "kf" key for it) are window keyframes: the live hook H2 emits them (kind "reanchor").  They are recovered as
        # the trace's edge endpoints that the frame-log table lacks (the design's table lacked them too: on the 11
        # design-row streams there are none; on the forced-loss traces they split the chain falsely).
        have = set(int(x) for x in kf_f)
        extra = sorted({f for key in E for f in key} - have)
        self.n_reanchor_added = 0
        if extra:
            frmap = {r["frame"]: r for r in fr}
            rows = [(int(f), float(t), int(s), bool(v)) for f, t, s, v in zip(kf_f, kf_t, kf_seg, kf_vi)]
            for f in extra:
                r = frmap.get(f)
                if r is None or f not in odo:
                    continue
                rows.append((int(f), float(r["t"]), int(r["segment"]), bool(r["vi"])))
                self.n_reanchor_added += 1
            rows.sort()
            kf_f = np.array([x[0] for x in rows], np.int64)
            kf_t = np.array([x[1] for x in rows])
            kf_seg = np.array([x[2] for x in rows], np.int64)
            kf_vi = np.array([x[3] for x in rows], bool)
        self.kf_frame, self.kf_t, self.kf_seg, self.kf_vi = kf_f, kf_t, kf_seg, kf_vi
        N = len(kf_f)
        self.packets = []
        self.finals = []
        n_edge = n_link = n_missing = 0
        for j in range(N):
            p = KfPacket(j=j, fidx=int(kf_f[j]), t=float(kf_t[j]), segment=int(kf_seg[j]), vi=bool(kf_vi[j]),
                         T_odom=odo[int(kf_f[j])])
            if j > 0:
                key = (int(kf_f[j - 1]), int(kf_f[j]))
                if key in E:
                    fst, fin, _vi = E[key]
                    p.edge_kind, p.ZR, p.Zp, p.S = "edge", fst[0], fst[1], fst[2]
                    self.finals.append((min(j - 1 + N_KF, N - 1), EdgeFinal(j - 1, fin[0], fin[1], fin[2])))
                    n_edge += 1
                elif key in L:
                    kind, cR, cp, cS = L[key]
                    p.edge_kind, p.ZR, p.Zp, p.S = kind, cR, cp, cS
                    self.finals.append((min(j - 1 + N_KF, N - 1), EdgeFinal(j - 1, None, None, None)))
                    n_link += 1
                else:
                    p.edge_kind, p.ZR, p.Zp, p.S = "missing", np.eye(3), np.zeros(3), None
                    self.finals.append((min(j - 1 + N_KF, N - 1), EdgeFinal(j - 1, None, None, None)))
                    n_missing += 1
            self.packets.append(p)
        self.finals.sort(key=lambda x: (x[0], x[1].a))
        self.counts = {"kfs": N, "edges": n_edge, "links": n_link, "missing": n_missing,
                       "reanchor_added": self.n_reanchor_added,
                       "link_kinds": sorted({p.edge_kind for p in self.packets if p.edge_kind not in ("edge", "first")})}
        self.pos_f = {int(f): i for i, f in enumerate(self.fr_idx)}
