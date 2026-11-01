#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1 MARGINAL PRIOR: square-root storage with first-estimate Jacobians (FEJ), and the Schur marginalisation algebra.

Conventions: xclean/vio/CONVENTIONS.md.

**Blocks.** The prior is over BLOCKS, one per window slot. A block is either:
- a pose (6 dims, [dphi, dp]), or
- a full VI state (15 dims, [dphi, dp, dv, dbg, dba]).

**The prior** is cost 1/2 ||S d + r||^2.
- d = x [-] x0 over its blocks: per block [Log(R0^T R), p - p0 (, v - v0, bg - bg0, ba - ba0)].
- x0 is the block's FIRST linearisation point inside the prior. That is FEJ: S never changes after it is formed.
- The Jacobian of the Log part wrt dphi is taken as I (small-difference approximation, Basalt practice, stated in
  CONVENTIONS.md).

**Square root.** S comes from the information form (H, b) by an eigen-decomposition: H = V diag(l) V^T,
S = diag(sqrt l) V^T over the eigenvalues above the numerical floor (l > 1e-12 l_max), r = diag(1/sqrt l) V^T b.
- Rank deficiency (the VO gauge after its prior is marginalised, unobserved directions) is dropped, not inverted.
- In float64 this is equivalent to Demmel's QR form (stated; the QR form matters for float32).

**Combining.** A new marginalisation's information is evaluated at the CURRENT estimate. Blocks already in the old prior
keep their x0 and b is re-expressed there: b_x0 = b_cur - H d_cur. New blocks (or new dims of a block that grew from a
pose to a full state) enter with x0 = the current estimate.
"""
from __future__ import annotations

import numpy as np

from xclean.vio.window import so3_log_nb

EIG_FLOOR = 1e-12            # numerical rank floor, relative to the largest eigenvalue (float64 arithmetic)


def _x0_of(st):
    """(R, p, v, bg, ba) copies from a state-like tuple (v / bg / ba may be None -> zeros)."""
    R, p, v, bg, ba = st
    z = np.zeros(3)
    return (np.array(R, float), np.array(p, float), np.array(z if v is None else v, float),
            np.array(z if bg is None else bg, float), np.array(z if ba is None else ba, float))


def _boxminus(st, x0, dim):
    R, p, v, bg, ba = st
    R0, p0, v0, bg0, ba0 = x0
    out = np.empty(dim)
    out[0:3] = so3_log_nb(np.ascontiguousarray(R0.T @ R))
    out[3:6] = p - p0
    if dim == 15:
        out[6:9] = v - v0
        out[9:12] = bg - bg0
        out[12:15] = ba - ba0
    return out


class Prior:
    def __init__(self, sids, dims, x0, S, r):
        self.sids = list(sids)                         # slot ids, in block order
        self.dims = [int(d) for d in dims]             # 6 or 15 per block
        self.x0 = [_x0_of(x) for x in x0]              # per block (R0, p0, v0, bg0, ba0)
        self.S = np.asarray(S, float).copy()
        self.r = np.asarray(r, float).copy()
        self.cols = None                               # problem columns of each block (bind())

    @property
    def n(self):
        return int(sum(self.dims))

    def _boff(self):
        return np.concatenate([[0], np.cumsum(self.dims)]).astype(np.int64)

    # ---- information form at x0 --------------------------------------------------------------------------
    def info(self):
        return self.S.T @ self.S, self.S.T @ self.r

    @classmethod
    def from_info(cls, sids, dims, x0, H, b):
        H = 0.5 * (H + H.T)
        lam, V = np.linalg.eigh(H)
        lmax = float(lam.max()) if len(lam) else 0.0
        keep = lam > EIG_FLOOR * max(lmax, 1e-300)
        sl = np.sqrt(lam[keep])
        Vk = V[:, keep]
        S = sl[:, None] * Vk.T
        r = (Vk.T @ b) / sl
        return cls(sids, dims, x0, S, r)

    # ---- use inside a problem ----------------------------------------------------------------------------
    def bind(self, index_of_sid, off=None, slot_dims=None):
        """off (P+1): the problem's block offsets (default 6 per slot); a prior block may cover only the first 6
        dims of a full (15) slot, never more dims than the slot has."""
        cols = []
        for s, d in zip(self.sids, self.dims):
            i = index_of_sid[s]
            o = 6 * i if off is None else int(off[i])
            if slot_dims is not None and d > int(slot_dims[i]):
                raise RuntimeError("prior block of slot %d has %d dims, the slot has %d" % (s, d, slot_dims[i]))
            cols.append(np.arange(o, o + d))
        self.cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        self._idx = [index_of_sid[s] for s in self.sids]

    def _d(self, pb_state):
        Rwb, pwb, vel, bgs, bas = pb_state
        d = np.zeros(self.n)
        bo = self._boff()
        for k, i in enumerate(self._idx):
            st = (Rwb[i], pwb[i], None if vel is None else vel[i], None if bgs is None else bgs[i],
                  None if bas is None else bas[i])
            if self.dims[k] == 15 and vel is None:
                raise RuntimeError("a 15-dim prior block needs the VI state arrays")
            d[bo[k]:bo[k + 1]] = _boxminus(st if self.dims[k] == 15 else (st[0], st[1], None, None, None),
                                           self.x0[k], self.dims[k])
        return d

    def terms(self, pb_state, n):
        d = self._d(pb_state)
        res = self.S @ d + self.r
        H = np.zeros((n, n))
        b = np.zeros(n)
        H[np.ix_(self.cols, self.cols)] = self.S.T @ self.S
        b[self.cols] = self.S.T @ res
        return H, b, 0.5 * float(res @ res)

    def cost(self, pb_state):
        res = self.S @ self._d(pb_state) + self.r
        return 0.5 * float(res @ res)

    # ---- structure changes ------------------------------------------------------------------------------
    def _without(self, k, dims_marg):
        """Marginalise dims dims_marg (indices within block k) out of the prior."""
        H, b = self.info()
        bo = self._boff()
        marg = np.array([bo[k] + j for j in dims_marg], np.int64)
        keep = np.array([i for i in range(self.n) if i not in set(marg.tolist())], np.int64)
        H2, b2 = schur(H, b, keep, marg)
        return H2, b2

    def drop_slot(self, sid):
        """Marginalise one slot's whole block out of the prior (exact Schur on the information form at x0)."""
        if sid not in self.sids:
            return self
        k = self.sids.index(sid)
        H2, b2 = self._without(k, range(self.dims[k]))
        sids = [s for i, s in enumerate(self.sids) if i != k]
        if not sids:
            return None
        return Prior.from_info(sids, [d for i, d in enumerate(self.dims) if i != k],
                               [x for i, x in enumerate(self.x0) if i != k], H2, b2)

    def shrink_slot(self, sid):
        """Marginalise the (v, bg, ba) dims of a full block out of the prior; the block becomes a pose block."""
        if sid not in self.sids:
            return self
        k = self.sids.index(sid)
        if self.dims[k] == 6:
            return self
        H2, b2 = self._without(k, range(6, 15))
        dims = list(self.dims)
        dims[k] = 6
        return Prior.from_info(self.sids, dims, self.x0, H2, b2)


def schur(H, b, keep, marg):
    """Marginalise `marg` out of (H, b): H' = H_kk - H_km H_mm^-1 H_mk, b' = b_k - H_km H_mm^-1 b_m.  H_mm is inverted
    by its eigen-decomposition with the same numerical floor (a rank-deficient block marginalises its observable part)."""
    Hmm = H[np.ix_(marg, marg)]
    Hkm = H[np.ix_(keep, marg)]
    lam, V = np.linalg.eigh(0.5 * (Hmm + Hmm.T))
    lmax = float(lam.max()) if len(lam) else 0.0
    ok = lam > EIG_FLOOR * max(lmax, 1e-300)
    Hinv = (V[:, ok] / lam[ok]) @ V[:, ok].T
    H2 = H[np.ix_(keep, keep)] - Hkm @ Hinv @ Hkm.T
    b2 = b[keep] - Hkm @ Hinv @ b[marg]
    return 0.5 * (H2 + H2.T), b2


def combine(prior_old, sids_new, dims_new, states_cur, H_cur, b_cur):
    """The new prior over blocks (sids_new, dims_new) from information (H_cur, b_cur) linearised at the CURRENT states
    (states_cur: per block (R, p, v, bg, ba)).  Blocks already in the old prior keep their x0 (FEJ) and b is
    re-expressed there (b_x0 = b_cur - H d_cur); dims a block did not have in the old prior start at the current
    estimate (d = 0 there)."""
    n = int(sum(dims_new))
    x0 = []
    d = np.zeros(n)
    o = 0
    for s, dim, st in zip(sids_new, dims_new, states_cur):
        cur = _x0_of(st)
        if prior_old is not None and s in prior_old.sids:
            j = prior_old.sids.index(s)
            old = prior_old.x0[j]
            if prior_old.dims[j] == 15 or dim == 6:
                xb = old
            else:                                   # a pose block that grew: keep the pose x0, new dims at current
                xb = (old[0], old[1], cur[2], cur[3], cur[4])
            x0.append(xb)
            d[o:o + dim] = _boxminus(cur, xb, dim)
        else:
            x0.append(cur)
        o += dim
    b_x0 = b_cur - H_cur @ d
    return Prior.from_info(sids_new, dims_new, x0, H_cur, b_x0)


def gauge_prior(sid, R, p, info=1e10):
    """The VO gauge: a 6-DoF prior on the first keyframe of a segment (DESIGN_P1 §4.3: first-KF prior inside the
    square-root factor).  info (class D, a gauge fix) must dominate any visual information on one pose (<= ~1e8 at
    sigma = 1 px); its value changes no estimate beyond the gauge."""
    return Prior([sid], [6], [(R, p, None, None, None)], np.sqrt(info) * np.eye(6), np.zeros(6))


def add_block_info(prior, sid, dim, st, Hb, bb):
    """Merge independent information (Hb, bb at the current state st, over block sid's first `dim` dims) into the prior
    (used at VI admission: the init's covariance of v, bg, ba as a prior on the newest state)."""
    if prior is None:
        return Prior.from_info([sid], [dim], [st], Hb, bb)
    H0, b0 = prior.info()
    sids = list(prior.sids)
    dims = list(prior.dims)
    x0 = list(prior.x0)
    if sid in sids:
        k = sids.index(sid)
        if dims[k] < dim:
            # grow the block: pad the old information with zero rows/cols for the new dims
            bo = prior._boff()
            ins = bo[k] + dims[k]
            pad = dim - dims[k]
            H0 = np.insert(np.insert(H0, [ins] * pad, 0.0, axis=0), [ins] * pad, 0.0, axis=1)
            b0 = np.insert(b0, [ins] * pad, 0.0)
            dims[k] = dim
            cur = _x0_of(st)
            x0[k] = (x0[k][0], x0[k][1], cur[2], cur[3], cur[4])
        bo = np.concatenate([[0], np.cumsum(dims)]).astype(np.int64)
        cols = np.arange(bo[k], bo[k] + dim)
        # Hb, bb are at the current state; express them at the block's x0
        dk = _boxminus(_x0_of(st), x0[k], dim)
        H0[np.ix_(cols, cols)] += Hb
        b0[cols] += bb - Hb @ dk
        return Prior.from_info(sids, dims, x0, H0, b0)
    n0 = H0.shape[0]
    H = np.zeros((n0 + dim, n0 + dim))
    H[:n0, :n0] = H0
    H[n0:, n0:] = Hb
    b = np.concatenate([b0, bb])
    return Prior.from_info(sids + [sid], dims + [dim], x0 + [_x0_of(st)], H, b)
