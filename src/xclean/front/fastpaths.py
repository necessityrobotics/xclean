#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1 frame-path CPU fast paths (from our latency profiling study; the profiler's microbench kernels, integrated behind
flags).  Each fix keeps its caller's API and, except M2, its numbers bit for bit (lock-step identity is the check).

    XCLEAN_LAT = comma / '+' list of: g1, e1, e2, e3, m1, m2, mf, gf
    DEFAULT (variable unset): g1 + e1 + e2 + e3 + m1 + mf, plus gf WHEN A GPU IS VISIBLE (the bit-identical
    set).  Without a visible GPU (CUDA_VISIBLE_DEVICES=-1: every lock-step test series, CPU-only rule) the default
    drops gf and the CPU modules run - the same numbers, by the bit-identity gates.  An EXPLICIT gf without a GPU
    raises.  XCLEAN_LAT=none (or empty) switches every fast path off; m2 is never on by default.

  g1  gc.collect(); gc.freeze() after warm-up (run_live)                 - no estimator change
  e1  numba greedy pick (Frontend._detect's loop)                         - bit-identical picks
  e2  stable argsort of -response (= lexsort((xs, ys, -resp)) on row-major nonzero candidates)  - identical order
  e3  cv2.KeyPoint.convert for the ORB keypoints (class_id path kept as the fallback)            - identical descriptors
  m1  numba one-pass Hamming BF: mutual NN + Lowe ratio + chance bound (Frontend.match_desc)     - identical matches
  m2  numba pose.refine (sigma_par None, point_scale None: every P1 call)  - rounding-level, NOT bit-identical
  mf  front/mapmatch_fast.MapMatcher for the gated matcher (CPU, numba)  - identical outputs
  gf  front/gpu_front: device detection + ORB, 1-D stereo (D4 band incl.) and stage-A Hamming  - BIT-IDENTICAL to the CPU
      modules; refuses XCLEAN_P1_STEREO_IMPL=v2
No constant is added (implementation changes only).
"""
from __future__ import annotations

import math
import os

import numpy as np
from numba import njit, types
from numba.extending import intrinsic

LAT_DEFAULT = "g1,e1,e2,e3,m1,mf,gf"


def gpu_visible():
    """True when a CUDA device is visible to this process (CUDA_VISIBLE_DEVICES=-1 hides it; no cupy import here)."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is not None and v.strip() in ("-1", ""):
        return False
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


_lat_env = os.environ.get("XCLEAN_LAT")
LAT_EXPLICIT = _lat_env is not None
_lat_env = LAT_DEFAULT if _lat_env is None else _lat_env
LAT = set(x.strip().lower() for x in _lat_env.replace("+", ",").split(",") if x.strip() and x.strip().lower() != "none")
if not LAT <= {"g1", "e1", "e2", "e3", "m1", "m2", "mf", "gf"}:
    raise ValueError("XCLEAN_LAT: unknown fix in %r" % sorted(LAT))
if "gf" in LAT and not LAT_EXPLICIT and not gpu_visible():
    LAT.discard("gf")                       # default without a GPU: the CPU modules (identical numbers)


def on(name):
    return name in LAT


# ------------------------------------------------------------------------------------------------ E1 greedy pick
@njit(cache=True, nogil=True)
def greedy_nb(xs, ys, cell, order, r, n_want, ncells):
    """Frontend._detect's greedy selection (strongest first, one per empty cell, never within r of a pick; '<'
    strict) with a dense hash grid instead of Python dicts.  -> picked candidate indices in pick order."""
    n = order.shape[0]
    seen = np.zeros(ncells + 1, np.bool_)
    gx = int(math.floor(xs.max() / r)) + 3 if n else 1
    gy = int(math.floor(ys.max() / r)) + 3 if n else 1
    head = np.full(gx * gy, -1, np.int64)
    nxt = np.full(n_want + 1, -1, np.int64)
    pk = np.empty(n_want, np.int64)
    np_ = 0
    r2 = r * r
    for t in range(n):
        j = order[t]
        c = cell[j]
        if seen[c]:
            continue
        hx = int(math.floor(xs[j] / r))
        hy = int(math.floor(ys[j] / r))
        clash = False
        for ax in range(hx - 1, hx + 2):
            if ax < 0 or ax >= gx:
                continue
            for ay in range(hy - 1, hy + 2):
                if ay < 0 or ay >= gy:
                    continue
                q_ = head[ay * gx + ax]
                while q_ >= 0:
                    k = pk[q_]
                    dx = float(xs[k] - xs[j])
                    dy = float(ys[k] - ys[j])
                    if dx * dx + dy * dy < r2:
                        clash = True
                        break
                    q_ = nxt[q_]
                if clash:
                    break
            if clash:
                break
        if clash:
            continue
        seen[c] = True
        pk[np_] = j
        g = hy * gx + hx
        nxt[np_] = head[g]
        head[g] = np_
        np_ += 1
        if np_ >= n_want:
            break
    return pk[:np_]


def detect_fast(fe, gray, n_want, budget, R, cand):
    """Frontend._detect for P1 (no existing keypoints) with E1 / E2 switched by their flags; same output."""
    from xclean.stages.features import KLT_WIN
    H, W = gray.shape
    if n_want <= 0:
        return np.zeros((0, 2), np.float32)
    cand = cand.copy()
    b = KLT_WIN // 2 + 1
    cand[:b, :] = False
    cand[-b:, :] = False
    cand[:, :b] = False
    cand[:, -b:] = False
    r = fe._min_dist(W, H, budget)
    ys, xs = np.nonzero(cand)
    if not len(xs):
        return np.zeros((0, 2), np.float32)
    resp = R[ys, xs]
    s = fe._cell(W, H, budget)
    ncx = int(math.ceil(W / s))
    cell = (ys // s).astype(np.int64) * ncx + (xs // s).astype(np.int64)
    order = np.argsort(-resp, kind="stable") if on("e2") else np.lexsort((xs, ys, -resp))
    if on("e1"):
        pick = greedy_nb(xs.astype(np.int64), ys.astype(np.int64), cell, order.astype(np.int64), float(r),
                         int(n_want), int(cell.max()) + 1)
    else:
        seen = set()
        pick = []
        hashed = {}
        r2 = r * r
        for j in order:
            c = int(cell[j])
            if c in seen:
                continue
            hx, hy = int(xs[j] // r), int(ys[j] // r)
            clash = False
            for ax in (hx - 1, hx, hx + 1):
                for ay in (hy - 1, hy, hy + 1):
                    for k in hashed.get((ax, ay), ()):
                        if (xs[k] - xs[j]) ** 2 + (ys[k] - ys[j]) ** 2 < r2:
                            clash = True
                            break
                    if clash:
                        break
                if clash:
                    break
            if clash:
                continue
            seen.add(c)
            pick.append(j)
            hashed.setdefault((hx, hy), []).append(j)
            if len(pick) >= n_want:
                break
        pick = np.asarray(pick, np.int64)
    fe.stats["detected"] += len(pick)
    return np.stack([xs[pick], ys[pick]], axis=1).astype(np.float32)


def describe_fast(fe, gray, kp):
    """Frontend.describe with E3: cv2.KeyPoint.convert (count checked; the class_id path is the fallback)."""
    import cv2
    from xclean.stages.features import ORB_PAD, ORB_PATCH
    if len(kp) == 0:
        return np.zeros((0, 32), np.uint8)
    pad = cv2.copyMakeBorder(gray, ORB_PAD, ORB_PAD, ORB_PAD, ORB_PAD, cv2.BORDER_REFLECT_101)
    kps = cv2.KeyPoint.convert((np.asarray(kp, np.float32) + ORB_PAD).astype(np.float32), size=float(ORB_PATCH))
    out_k, desc = fe._orb.compute(pad, kps)
    if desc is not None and len(out_k) == len(kp):
        return np.ascontiguousarray(desc, np.uint8)
    return fe.describe(gray, kp)


# ------------------------------------------------------------------------------------------------ M1 Hamming BF
@intrinsic
def _ctpop(typingctx, x):
    sig = types.uint64(types.uint64)

    def codegen(context, builder, signature, args):
        return builder.ctpop(args[0])
    return sig, codegen


@njit(cache=False, nogil=True)
def bf_mutual(A, B, ham_max, ratio):
    """Mutual NN + ratio + chance bound on (L,4) / (N,4) uint64 descriptors in one pass; ties: the lower index wins
    (first minimum, a sequential scan).  -> (i0, i1) in query order (as Frontend.match_desc)."""
    L = A.shape[0]
    N = B.shape[0]
    b1 = np.full(L, 1 << 30, np.int64)
    b2 = np.full(L, 1 << 30, np.int64)
    bi = np.full(L, -1, np.int64)
    cb = np.full(N, 1 << 30, np.int64)
    ci = np.full(N, -1, np.int64)
    for i in range(L):
        a0 = A[i, 0]
        a1 = A[i, 1]
        a2 = A[i, 2]
        a3 = A[i, 3]
        for j in range(N):
            dd = int(_ctpop(a0 ^ B[j, 0]) + _ctpop(a1 ^ B[j, 1]) + _ctpop(a2 ^ B[j, 2]) + _ctpop(a3 ^ B[j, 3]))
            if dd < b1[i]:
                b2[i] = b1[i]
                b1[i] = dd
                bi[i] = j
            elif dd < b2[i]:
                b2[i] = dd
            if dd < cb[j]:
                cb[j] = dd
                ci[j] = i
    o0 = np.empty(L, np.int64)
    o1 = np.empty(L, np.int64)
    m = 0
    for i in range(L):
        j = bi[i]
        if j < 0:
            continue
        if b1[i] > ham_max or not (b1[i] < ratio * b2[i]):
            continue
        if ci[j] != i:
            continue
        o0[m] = i
        o1[m] = j
        m += 1
    return o0[:m], o1[:m]


def match_desc_fast(d0, d1):
    """Frontend.match_desc(f0, f1) without masks, M1 kernel; same contract and order."""
    from xclean.stages.features import HAMMING_CHANCE, RATIO
    z = np.zeros(0, np.int64)
    if len(d0) < 2 or len(d1) < 2:
        return z, z
    A = np.ascontiguousarray(d0, np.uint8).view(np.uint64).reshape(len(d0), 4)
    B = np.ascontiguousarray(d1, np.uint8).view(np.uint64).reshape(len(d1), 4)
    return bf_mutual(A, B, int(HAMMING_CHANCE), float(RATIO))


def warm():
    """Compile the flagged kernels before the clock starts."""
    if on("e1"):
        greedy_nb(np.array([5, 9], np.int64), np.array([5, 9], np.int64), np.array([0, 1], np.int64),
                  np.array([0, 1], np.int64), 2.0, 2, 2)
    if on("m1"):
        a = np.zeros((3, 32), np.uint8)
        match_desc_fast(a, a)
    if on("m2"):
        from xclean.track.pose_fast import refine_nb
        refine_nb(np.eye(4), np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.0, 0.1, 2.0]]),
                  np.array([[300.0, 200.0], [320.0, 200.0], [300.0, 220.0]]), 400.0, 400.0, 300.0, 200.0, 1.0, 2,
                  1.345)
