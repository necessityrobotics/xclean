# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.3 worker: the KEYFRAME DATABASE and RETRIEVAL.

  records     per keyframe: header (sequence j, frame, stamp, segment, VI flag, LIVE-ODOM pose, the LIVE-MAP pose
              published for it = the ANCHOR), the rich set (in a per-run memory-mapped file: RAM holds only headers,
              BoW and the inverted file, B6), the sparse BoW vector;
  candidates  a keyframe becomes a loop candidate when it LEFT P1's window (its edge_final message arrived: the exact
              window, no constant); cross-segment candidates are allowed;
  retrieval   raw cosine of tf-idf L2 BoW vectors over the candidate population by an exact INVERTED FILE (float32,
              summed in ascending word order = bit-identical to the design evidence's sparse product); threshold =
              median + 3 scaled MAD of the population (xclean/loop/closer._outlier_rule, IMPORTED), sufficiency 30;
              the Fisher fallback (closer.fisher_score on atanh(sim)) only where the raw threshold is infeasible
              (>= 1); the best 3 ABOVE the threshold are verified (top-K 3, carried).
Cost: O(N * sum_w p(w)) postings per query (the loop-closure design: linear in N; place-level culling is NOT built, a
stated limitation).
"""
from __future__ import annotations

import mmap
import os

import numpy as np

from xclean.slam.rich import RichSet

TOPK = 3                 # closer / XC-17 (carried)
SUFFICIENCY = 30         # closer.py (carried)
V_WORDS = 10000          # vocabulary size (k = 10, L = 4)


class RichStore:
    """Append-only rich-set store: a memory-mapped file (path) or RAM (path None)."""

    def __init__(self, path=None):
        self.path = path
        self.idx = {}
        self._ram = {}
        self._f = None
        self._mm = None
        self._mm_size = 0
        self.bytes = 0
        if path:
            if os.path.exists(path):
                os.remove(path)                           # stale-artefact rule
            self._f = open(path, "w+b")

    def put(self, j, r: RichSet):
        if self._f is None:
            self._ram[j] = r
            return
        n = len(r.uv)
        parts = [np.ascontiguousarray(r.uv, np.float32).tobytes(), np.ascontiguousarray(r.desc, np.uint8).tobytes(),
                 np.ascontiguousarray(r.oct, np.int8).tobytes(), np.ascontiguousarray(r.X, np.float32).tobytes(),
                 np.ascontiguousarray(r.dsig, np.float32).tobytes()]
        off = self.bytes
        self._f.seek(off)                 # creating a Windows mmap moves the file position: always append at the end
        for p in parts:
            self._f.write(p)
        self.bytes += sum(len(p) for p in parts)
        self.idx[j] = (off, n)

    def get(self, j):
        if self._f is None:
            return self._ram.get(j)
        if j not in self.idx:
            return None
        off, n = self.idx[j]
        end = off + 57 * n
        if end > self._mm_size:
            self._f.flush()
            if self._mm is not None:
                self._mm.close()
            self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
            self._mm_size = len(self._mm)
        b = memoryview(self._mm)[off:end]
        o = 0
        uv = np.frombuffer(b[o:o + 8 * n], np.float32).reshape(n, 2)
        o += 8 * n
        desc = np.frombuffer(b[o:o + 32 * n], np.uint8).reshape(n, 32)
        o += 32 * n
        oc = np.frombuffer(b[o:o + n], np.int8)
        o += n
        X = np.frombuffer(b[o:o + 12 * n], np.float32).reshape(n, 3)
        o += 12 * n
        ds = np.frombuffer(b[o:o + 4 * n], np.float32)
        return RichSet(uv=uv.copy(), desc=desc.copy(), oct=oc.copy(), X=X.copy(), dsig=ds.copy(),
                       bow_i=np.zeros(0, np.int32), bow_v=np.zeros(0, np.float32))

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._f is not None:
            self._f.close()
            self._f = None


class InvertedFile:
    """Exact inverted file over the CANDIDATE keyframes (positions 0..n-1 in candidacy order)."""

    def __init__(self, V=V_WORDS):
        self.V = V
        self.pos = [None] * V
        self.val = [None] * V
        self.cnt = np.zeros(V, np.int64)
        self.n = 0
        self.n_postings = 0

    def add(self, bow_i, bow_v):
        p = self.n
        self.n += 1
        for w, v in zip(bow_i.tolist(), bow_v.tolist()):
            c = self.cnt[w]
            if self.pos[w] is None:
                self.pos[w] = np.zeros(16, np.int32)
                self.val[w] = np.zeros(16, np.float32)
            elif c >= len(self.pos[w]):
                self.pos[w] = np.concatenate([self.pos[w], np.zeros(len(self.pos[w]), np.int32)])
                self.val[w] = np.concatenate([self.val[w], np.zeros(len(self.val[w]), np.float32)])
            self.pos[w][c] = p
            self.val[w][c] = v
            self.cnt[w] = c + 1
        self.n_postings += len(bow_i)
        return p

    def scores(self, q_i, q_v):
        """cosine of the query against every candidate: float32, summed in ascending word order."""
        sims = np.zeros(self.n, np.float32)
        P, Wt = [], []
        for w, qv in zip(q_i.tolist(), q_v.astype(np.float32)):
            c = int(self.cnt[w])
            if c:
                P.append(self.pos[w][:c])
                Wt.append(self.val[w][:c] * qv)
        if P:
            np.add.at(sims, np.concatenate(P), np.concatenate(Wt))
        return sims, int(sum(len(x) for x in P))


class KfDb:
    def __init__(self, store_path=None):
        self.store = RichStore(store_path)
        self.hdr = {}                 # j -> dict(fidx, t, segment, vi, T_odom, T_map)
        self.bow = {}                 # j -> (bow_i, bow_v)
        self.has_rich = set()
        self.cand = []                # candidacy order: position -> j
        self.cand_pos = {}
        self.inv = InvertedFile()
        self.pending_cand = {}        # j -> t_emit: left the window before its rich set existed (job not yet run)
        self.cand_t = []              # candidacy emission stamps (parallel to cand)

    def add_header(self, p):
        self.hdr[p.j] = {"fidx": p.fidx, "t": p.t, "segment": p.segment, "vi": p.vi, "T_odom": p.T_odom,
                         "T_map": p.T_map}

    def add_rich(self, j, r):
        self.store.put(j, r)
        self.bow[j] = (r.bow_i, r.bow_v)
        self.has_rich.add(j)
        if j in self.pending_cand:
            t_c = self.pending_cand.pop(j)
            self._insert(j, t_c)

    def make_candidate(self, j, t_emit=-np.inf):
        """Keyframe j LEFT the window: its edge_final (or a segment restart) was emitted with the frame stamped t_emit.
        A query keyframe stamped t_q sees exactly the candidates with t_emit <= t_q (they had left the window when the
        query keyframe was created: the exact window, no constant; the loop-closure design)."""
        if j in self.cand_pos or j in self.pending_cand:
            return
        if j in self.has_rich:
            self._insert(j, t_emit)
        else:
            self.pending_cand[j] = t_emit

    def _insert(self, j, t_emit):
        bi, bv = self.bow[j]
        p = self.inv.add(bi, bv)
        self.cand.append(j)
        self.cand_t.append(float(t_emit))
        self.cand_pos[j] = p

    def rich(self, j):
        return self.store.get(j)

    def retrieve(self, q_i, q_v, t_query=np.inf):
        """-> dict(n_db, thr, fisher, ranked [(j, sim, rank)] of the best TOPK, above [(j, sim, rank)] those above the
        threshold, postings) or None below sufficiency.  The population = the candidates with t_emit <= t_query."""
        from xclean.loop.closer import _outlier_rule, fisher_score, COSINE_BOUND
        ct = np.asarray(self.cand_t, float)
        valid = np.nonzero(ct <= t_query)[0]
        n = len(valid)
        if n < SUFFICIENCY:
            return None
        sims_all, npost = self.inv.scores(q_i, q_v)
        sims = sims_all[valid] if n < len(sims_all) else sims_all
        _med, _mad, thr = _outlier_rule(sims)
        fisher = False
        order = np.argsort(-sims, kind="stable")
        top = [int(x) for x in order[:TOPK]]
        if thr >= COSINE_BOUND:
            fisher = True
            fs = fisher_score(sims)
            _m2, _md2, thr_f = _outlier_rule(fs)
            above = [p for p in top if float(fs[p]) >= thr_f]
        else:
            above = [p for p in top if float(sims[p]) >= thr]
        cj = [self.cand[int(v)] for v in valid] if n < len(sims_all) else self.cand
        ranked = [(cj[p], float(sims[p]), r) for r, p in enumerate(top)]
        abv = [(cj[p], float(sims[p]), top.index(p)) for p in above]
        return {"n_db": n, "thr": float(thr), "fisher": fisher, "ranked": ranked, "above": abv, "postings": npost}
