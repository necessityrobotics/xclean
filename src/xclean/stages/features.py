#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""THE CLASSICAL FRONT END - Shi-Tomasi on a grid, pyramidal KLT, ORB descriptors.  No learned weights.

    fe = Frontend(rig)
    f1 = fe.extract(gray, pred_uv=None)      # tracks the previous frame's points into this one, tops up the grid
    i0, i1 = fe.match(kf_feats, f1)          # TRACK-ID JOIN: the keypoints of one physical track in both frames
    i0, i1 = fe.match_desc(fa, fb)           # prediction-free ORB (Hamming) matching: reloc, loop verify, rescue

WHAT REPLACES WHAT
    SuperPoint extract            -> KLT tracks of the previous frame + Shi-Tomasi top-up on a grid
    LightGlue frame-to-KF match   -> the track-id join: a keyframe keypoint and a frame keypoint correspond when they
                                     are the same KLT track (it survived every frame in between)
    LightGlue KF-to-KF (loops,    -> ORB (rBRIEF without orientation) Hamming, mutual nearest neighbour + ratio test +
    reloc, the covisible pool)       the chance bound below; geometry (PnP RANSAC) still decides
    256-float descriptors         -> 32-byte ORB descriptors, every keypoint, every frame (local-map tracking needs them)
OpenCV (Apache-2.0) provides cornerMinEigenVal, calcOpticalFlowPyrLK and the ORB descriptor; nothing from ORB-SLAM.

THE CONSTANTS, AND WHERE EACH COMES FROM (no per-dataset value anywhere)
    Shi-Tomasi quality 0.01 of the frame's max response   OpenCV's goodFeaturesToTrack default (Shi & Tomasi 1994:
                                                          the min eigenvalue IS the KLT trackability criterion)
    grid cell s = sqrt(W H / budget)                      one feature per cell at the budget (uniform spread)
    min distance r = sqrt(W H / (2 sqrt(3) budget))       hexagonal packing of `budget` discs over the image
    KLT window 21x21, 4 pyramid levels                    the common pyramidal-LK operating point (Bouguet)
    forward-backward error <= FEATURE_SIGMA_PX (1 px)     the system's own declared keypoint precision
    ORB: 31-px patch, no orientation                      frame-to-frame viewpoints inside one estimator window carry
                                                          no in-plane rotation worth paying rBRIEF's loss for
    Hamming chance bound 104 of 256 bits                  unrelated binary descriptors ~ Binomial(256, 1/2): mean 128,
                                                          sd 8; a match must sit 3 sd below chance
    ratio 0.8                                             Lowe's nearest/second-nearest test
    the feature-budget control loop                       the engine's, unchanged (same signal, same constants)
"""
from __future__ import annotations

import math

import numpy as np

from xclean.core.derive import MIN_PNP_INLIERS, FEATURE_SIGMA_PX
from xclean.obs.decision import Decision

KLT_WIN = 21
KLT_LEVELS = 3                    # maxLevel 3 = 4 pyramid levels
KLT_CRITERIA = None               # set lazily (cv2 constant)
SHI_QUALITY = 0.01
ORB_PATCH = 31
ORB_PAD = 32
HAMMING_BITS = 256
HAMMING_CHANCE = int(HAMMING_BITS / 2 - 3 * math.sqrt(HAMMING_BITS) / 2)    # 104
RATIO = 0.8


_NOISE_Q99 = None


def noise_floor_8u(gray):
    """99th percentile of cornerMinEigenVal(3,3) at 3x3 local maxima of pure Gaussian noise, scaled to this uint8
    frame's noise level sigma_n (Immerkaer 1996: sqrt(pi/2) / (6 (W-2)(H-2)) sum |I * N|).  The unit-sigma quantile is
    computed once (fixed seed, float input, then / 255^2 for OpenCV's 8-bit derivative scaling)."""
    global _NOISE_Q99
    import cv2
    if _NOISE_Q99 is None:
        rng = np.random.default_rng(12345)
        z = rng.normal(0.0, 1.0, (512, 512)).astype(np.float32)
        Rz = cv2.cornerMinEigenVal(z, 3, 3)
        mx = Rz >= cv2.dilate(Rz, np.ones((3, 3), np.uint8))
        _NOISE_Q99 = float(np.quantile(Rz[8:-8, 8:-8][mx[8:-8, 8:-8]], 0.99))
    Nk = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    H, W = gray.shape
    c = cv2.filter2D(gray.astype(np.float32), -1, Nk)[1:-1, 1:-1]
    sn = float(np.sqrt(np.pi / 2.0) / (6.0 * (W - 2) * (H - 2)) * np.abs(c).sum())
    return _NOISE_Q99 * sn * sn / (255.0 * 255.0)


def hamming(a, b):
    """Row-wise Hamming distance of two (n,32) uint8 arrays -> (n,) int."""
    x = np.bitwise_xor(a.view(np.uint64), b.view(np.uint64))
    return np.bitwise_count(x).sum(axis=1).astype(np.int64)


class Frontend:
    """Classical tracker-detector with the engine's Class-C feature budget."""

    GROW_FACTOR = 1.35
    SHRINK_FACTOR = 0.90
    HEADROOM = 3.0
    COMFORT = 8.0
    SUFFICIENCY = 5
    HYSTERESIS = 3

    def __init__(self, rig, device=None, cap=None, verbose=True):
        import cv2
        self.cv2 = cv2
        self.rig = rig
        self.budget = rig.feature_budget()
        self.floor = rig.conditioning_floor()
        self.cap = int(cap or 2048)
        self._recent = []
        self._cooldown = 0
        self.decisions = []
        self._orb = cv2.ORB_create(nfeatures=100000, scaleFactor=1.2, nlevels=1, edgeThreshold=ORB_PATCH,
                                   firstLevel=0, WTA_K=2, patchSize=ORB_PATCH)
        self._crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._prev_img = None
        self._prev_kp = None
        self._prev_id = None
        self._prev_age = None
        self._next_id = 0
        self.n_frames = 0
        self.delivered = None
        self._aligner = None
        self._anchor = {}               # track id -> (keyframe slot, x, y) of its birth-keyframe observation
        self._kf_imgs = {}              # keyframe slot -> image pyramid (GPU), bounded
        self._kf_count = 0
        self._reanchor_dev = []
        import os as _os2
        self.detect_mode = (_os2.environ.get("XCLEAN_DETECT", "frame") or "frame").strip().lower()
        if self.detect_mode not in ("frame", "cell"):
            raise RuntimeError("XCLEAN_DETECT=%r: frame (default) or cell" % self.detect_mode)
        import os as _os
        self.fb_sym = (_os.environ.get("XCLEAN_FB_SYM", "") or "").strip() in ("1", "on")
        if (_os.environ.get("XCLEAN_KLT_SNAP", "") or "").strip() not in ("", "0", "off"):
            raise RuntimeError("XCLEAN_KLT_SNAP is not part of this engine; unset it")
        self.stats = {"tracked": 0, "lost_status": 0, "lost_fb": 0, "lost_border": 0, "suppressed": 0,
                      "detected": 0, "pred_used": 0, "pred_retry_rescued": 0}
        if verbose:
            print(" [front] classical: Shi-Tomasi grid + KLT %dx%d/%d levels (FB <= %.1f px) + ORB-%d; budget "
                  "starts at %d kp (floor %d, cap %d); Hamming chance bound %d, ratio %.2f"
                  % (KLT_WIN, KLT_WIN, KLT_LEVELS + 1, FEATURE_SIGMA_PX, ORB_PATCH, self.budget, self.floor,
                     self.cap, HAMMING_CHANCE, RATIO))

    # ---- geometry of the grid, from the budget -------------------------------------------------------------
    def _cell(self, W, H, budget):
        return max(4.0, math.sqrt(W * H / max(budget, 1)))

    def _min_dist(self, W, H, budget):
        return max(2.0, math.sqrt(W * H / (2.0 * math.sqrt(3.0) * max(budget, 1))))

    # ---- KLT -------------------------------------------------------------------------------------------------
    def _klt(self, prev, cur, p0, p1_init):
        cv2 = self.cv2
        flags = 0
        p1 = None
        if p1_init is not None:
            p1 = np.ascontiguousarray(p1_init, np.float32).reshape(-1, 1, 2)
            flags = cv2.OPTFLOW_USE_INITIAL_FLOW
        p1, st, _err = cv2.calcOpticalFlowPyrLK(prev, cur, p0.reshape(-1, 1, 2), p1, winSize=(KLT_WIN, KLT_WIN),
                                                maxLevel=KLT_LEVELS, criteria=self._crit, flags=flags)
        # XC-04a (spec 2026-10-23): the STANDARD forward-backward check - the backward track starts at p1 itself (no
        # initial flow).  It used to be seeded with p0, i.e. with the answer, which passed 59-100 % of tracks aliased
        # by one period on periodic texture (the standard check: 0-1.4 %).
        p0b, st2, _e2 = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None,
                                                 winSize=(KLT_WIN, KLT_WIN), maxLevel=KLT_LEVELS,
                                                 criteria=self._crit, flags=0)
        if self.fb_sym and p1_init is not None:
            # TEST ARM XCLEAN_FB_SYM=1 (default OFF = XC-04a as is).  The zero-init backward track is a valid check only
            # inside its own capture range, KLT_WIN/2 x 2^KLT_LEVELS px (the window half-width at the coarsest level).
            # A forward displacement beyond it (a 7-10 period dataset gap: F3 1018, E1 8552, 146-178 px median flow)
            # cannot be found back from zero motion, so a correct track fails (diag_fb_gap.py: 6/459 kept on F3).
            # Only for THOSE tracks the backward track gets the forward prediction mirrored, p1 - (p1_init - p0)
            # (the same prior the forward track had, not the answer).  Inside the capture range XC-04a decides alone,
            # so a prediction-induced alias of a short motion is still rejected (test_fb_standard_init).
            cap = 0.5 * KLT_WIN * (2 ** KLT_LEVELS)
            p0r = p0.reshape(-1, 2)
            p1r = p1.reshape(-1, 2)
            fbz = np.linalg.norm(p0b.reshape(-1, 2) - p0r, axis=1)
            far = ((st2.ravel() != 1) | (fbz > FEATURE_SIGMA_PX)) & (np.linalg.norm(p1r - p0r, axis=1) > cap)                 & (st.ravel() == 1)
            if far.any():
                ii = np.nonzero(far)[0]
                b0 = p1r[ii] - (np.asarray(p1_init, np.float32).reshape(-1, 2)[ii] - p0r[ii])
                qb, sb, _e3 = cv2.calcOpticalFlowPyrLK(cur, prev, np.ascontiguousarray(p1r[ii]).reshape(-1, 1, 2),
                                                       np.ascontiguousarray(b0, np.float32).reshape(-1, 1, 2),
                                                       winSize=(KLT_WIN, KLT_WIN), maxLevel=KLT_LEVELS,
                                                       criteria=self._crit, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
                p0b = p0b.copy()
                st2 = st2.copy()
                p0b.reshape(-1, 2)[ii] = qb.reshape(-1, 2)
                st2.ravel()[ii] = sb.ravel()
                self.stats["fb_sym_far"] = self.stats.get("fb_sym_far", 0) + int(len(ii))
        p1 = p1.reshape(-1, 2)
        fb = np.linalg.norm(p0b.reshape(-1, 2) - p0, axis=1)
        ok_st = (st.ravel() == 1) & (st2.ravel() == 1)
        return p1, ok_st, fb

    @staticmethod
    def photometric_match(prev, cur):
        """`cur` with its intensity mean and standard deviation matched to `prev` (a GLOBAL affine gain/offset).

        ★ WHY (c1, v2_03): pyramidal LK assumes brightness constancy (plain SSD).  An auto-exposure step is a global
        gain change - v2_03 frame 223: image mean 130 -> 104 on an already blurred view - and it killed 86 % of the
        tracks in one frame.  Used ONLY for the retry of tracks the raw image failed (XC-05, see _track); detection,
        ORB and stereo use the raw image."""
        mp, sp = float(prev.mean()), float(prev.std())
        mc, sc = float(cur.mean()), float(cur.std())
        if sc < 1e-6 or sp < 1e-6:
            return cur
        out = (cur.astype(np.float32) - mc) * (sp / sc) + mp
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def _track(self, gray, pred_uv):
        """Track the previous frame's points.  -> (kp, ids, ages) of the survivors."""
        p0 = self._prev_kp.astype(np.float32)
        H, W = gray.shape
        if len(p0) == 0:
            return np.zeros((0, 2), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64)
        init = None
        if pred_uv is not None:
            pu = np.asarray(pred_uv, np.float32)
            okp = np.isfinite(pu).all(axis=1) & (pu[:, 0] > -W) & (pu[:, 0] < 2 * W) & (pu[:, 1] > -H) \
                & (pu[:, 1] < 2 * H)
            init = np.where(okp[:, None], pu, p0)
            self.stats["pred_used"] += 1
        p1, ok_st, fb = self._klt(self._prev_img, gray, p0, init)
        good = ok_st & (fb <= FEATURE_SIGMA_PX)
        if init is not None and (~good).any():
            # the PREDICTION-FREE RETRY: a track the prediction failed is re-tracked from zero motion (a wrong gyro
            # or velocity must not be able to kill a track that plain KLT keeps)
            bad = np.nonzero(~good)[0]
            q1, ok2, fb2 = self._klt(self._prev_img, gray, p0[bad], None)
            g2 = ok2 & (fb2 <= FEATURE_SIGMA_PX)
            if g2.any():
                p1[bad[g2]] = q1[g2]
                fb[bad[g2]] = fb2[g2]
                ok_st[bad[g2]] = True
                good[bad[g2]] = True
                self.stats["pred_retry_rescued"] += int(g2.sum())
        if (~good).any():
            # XC-05 (replaces C19's unconditional global match): tracks the RAW image fails are retried on the image
            # photometrically matched to the previous frame (global gain/offset from the first two moments).  An
            # exposure step (v2_03 frame 223) is recovered and a frame whose moments change by CONTENT (mh_04 1973,
            # where the unconditional match lost tracks) keeps every raw track: survival >= max(raw, global) per
            # frame by construction.  No constant.
            comp = self.photometric_match(self._prev_img, gray)
            bad = np.nonzero(~good)[0]
            q1, ok3, fb3 = self._klt(self._prev_img, comp, p0[bad], None if init is None else init[bad])
            g3 = ok3 & (fb3 <= FEATURE_SIGMA_PX)
            if init is not None and (~g3).any():
                rb = np.nonzero(~g3)[0]
                r1, ok4, fb4 = self._klt(self._prev_img, comp, p0[bad[rb]], None)
                g4 = ok4 & (fb4 <= FEATURE_SIGMA_PX)
                q1[rb[g4]] = r1[g4]
                fb3[rb[g4]] = fb4[g4]
                ok3[rb[g4]] = True
                g3[rb[g4]] = True
            if g3.any():
                p1[bad[g3]] = q1[g3]
                fb[bad[g3]] = fb3[g3]
                ok_st[bad[g3]] = True
                good[bad[g3]] = True
                self.stats["photo_rescued"] = self.stats.get("photo_rescued", 0) + int(g3.sum())
        inside = (p1[:, 0] >= 0) & (p1[:, 0] <= W - 1) & (p1[:, 1] >= 0) & (p1[:, 1] <= H - 1)
        self.stats["lost_status"] += int((~ok_st).sum())
        self.stats["lost_fb"] += int((ok_st & (fb > FEATURE_SIGMA_PX)).sum())
        self.stats["lost_border"] += int((good & ~inside).sum())
        keep = good & inside
        self.stats["tracked"] += int(keep.sum())
        return p1[keep], self._prev_id[keep], self._prev_age[keep] + 1

    # ---- suppression + detection -----------------------------------------------------------------------------
    def _suppress(self, kp, ids, ages, W, H, r):
        """Older tracks win: drop a track within r of an older survivor.  Conflicts are found by a KD-tree pair query
        (tracks move coherently, so conflicts are few); only those are resolved, greedily in age order."""
        from scipy.spatial import cKDTree
        if len(kp) < 2:
            return kp, ids, ages
        pairs = cKDTree(kp).query_pairs(r, output_type="ndarray")
        if len(pairs) == 0:
            return kp, ids, ages
        rank = np.empty(len(kp), np.int64)
        rank[np.lexsort((ids, -ages))] = np.arange(len(kp))      # 0 = oldest (ties by id: deterministic)
        nb = {}
        for a_, b_ in pairs:
            nb.setdefault(int(a_), []).append(int(b_))
            nb.setdefault(int(b_), []).append(int(a_))
        removed = np.zeros(len(kp), bool)
        for j in sorted(nb, key=lambda q: rank[q]):
            if removed[j]:
                continue
            for k in nb[j]:
                if rank[k] > rank[j]:
                    removed[k] = True
        keep = ~removed
        self.stats["suppressed"] += int(removed.sum())
        return kp[keep], ids[keep], ages[keep]

    def _response(self, gray, budget=None):
        """Shi-Tomasi min-eigenvalue response and its local maxima above the quality level -> (R, cand mask).

        ARM XCLEAN_DETECT=cell (spec XC-03; default = frame scope, bit-identical):
          * the 0.01 quality is taken relative to the max response of the candidate's own GRID CELL (cell size from
            the budget, as the bucketing uses) instead of the whole frame - a few bright corners no longer veto the
            rest of a dark image (v2_03 frame 777: 45-60 corners at the 2048 cap; mh_04 dark section 11-15 % of cells);
          * an absolute floor: the 99th percentile of the response at local maxima of PURE NOISE at this frame's
            measured noise level sigma_n (Immerkaer's estimator), from a Monte Carlo table computed once - a cell with
            no structure above noise yields nothing.  Class C (sigma_n per frame); the quantile is statistics."""
        cv2 = self.cv2
        R = cv2.cornerMinEigenVal(gray, 3, 3)
        rmax = float(R.max())
        if rmax <= 0:
            return R, np.zeros(R.shape, bool)
        dil = cv2.dilate(R, np.ones((3, 3), np.uint8))
        if self.detect_mode != "cell":
            return R, (R >= dil) & (R >= SHI_QUALITY * rmax)
        H, W = gray.shape
        c = int(max(4, round(self._cell(W, H, budget or self.budget))))
        hp, wp = -(-H // c) * c, -(-W // c) * c
        Rp = np.zeros((hp, wp), np.float32)
        Rp[:H, :W] = R
        cmax = Rp.reshape(hp // c, c, wp // c, c).max(axis=(1, 3))
        local = np.repeat(np.repeat(cmax, c, 0), c, 1)[:H, :W]
        floor = noise_floor_8u(gray)
        self.stats["noise_floor_last"] = floor
        return R, (R >= dil) & (R >= SHI_QUALITY * local) & (R >= floor)

    @staticmethod
    def _subpix(R, xs, ys):
        """Sub-pixel position of response maxima: a 1-D parabola through the 3 samples in x and in y."""
        H, W = R.shape
        x = xs.astype(np.int64)
        y = ys.astype(np.int64)
        inner = (x > 0) & (x < W - 1) & (y > 0) & (y < H - 1)
        fx = xs.astype(np.float64).copy()
        fy = ys.astype(np.float64).copy()
        xi, yi = x[inner], y[inner]
        c = R[yi, xi].astype(np.float64)
        l, r_ = R[yi, xi - 1].astype(np.float64), R[yi, xi + 1].astype(np.float64)
        u, d = R[yi - 1, xi].astype(np.float64), R[yi + 1, xi].astype(np.float64)
        with np.errstate(all="ignore"):
            dx = np.where(l - 2 * c + r_ < 0, 0.5 * (l - r_) / (l - 2 * c + r_), 0.0)
            dy = np.where(u - 2 * c + d < 0, 0.5 * (u - d) / (u - 2 * c + d), 0.0)
        fx[inner] += np.clip(dx, -0.5, 0.5)
        fy[inner] += np.clip(dy, -0.5, 0.5)
        return fx, fy

    def _detect(self, gray, kp_existing, n_want, budget, resp_cache=None):
        cv2 = self.cv2
        H, W = gray.shape
        if n_want <= 0:
            return np.zeros((0, 2), np.float32)
        R, cand = resp_cache if resp_cache is not None else self._response(gray, budget)
        cand = cand.copy()
        # keep a border where the KLT window and the descriptor are fully defined on real pixels
        b = KLT_WIN // 2 + 1
        cand[:b, :] = False
        cand[-b:, :] = False
        cand[:, :b] = False
        cand[:, -b:] = False
        r = self._min_dist(W, H, budget)
        ys, xs = np.nonzero(cand)
        if not len(xs):
            return np.zeros((0, 2), np.float32)
        if len(kp_existing):
            from scipy.spatial import cKDTree
            dd, _ii = cKDTree(kp_existing).query(np.stack([xs, ys], axis=1).astype(np.float64), k=1,
                                                 distance_upper_bound=r)
            free = ~np.isfinite(dd)
            xs, ys = xs[free], ys[free]
            if not len(xs):
                return np.zeros((0, 2), np.float32)
        resp = R[ys, xs]
        s = self._cell(W, H, budget)
        ncx = int(math.ceil(W / s))
        cell = (ys // s).astype(np.int64) * ncx + (xs // s).astype(np.int64)
        if len(kp_existing):
            occ = set(((kp_existing[:, 1] // s).astype(np.int64) * ncx
                       + (kp_existing[:, 0] // s).astype(np.int64)).tolist())
        else:
            occ = set()
        # best candidate per empty cell, strongest first, and never within r of a point already taken (the same
        # disc rule the tracks obey: otherwise two neighbouring cells' winners are suppressed one frame later)
        order = np.lexsort((xs, ys, -resp))           # strongest first, deterministic ties
        seen = set(occ)
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
        self.stats["detected"] += len(pick)
        return np.stack([xs[pick], ys[pick]], axis=1).astype(np.float32)

    def describe(self, gray, kp):
        """ORB (no orientation) at every keypoint, via a reflected border so no keypoint is dropped."""
        cv2 = self.cv2
        if len(kp) == 0:
            return np.zeros((0, 32), np.uint8)
        pad = cv2.copyMakeBorder(gray, ORB_PAD, ORB_PAD, ORB_PAD, ORB_PAD, cv2.BORDER_REFLECT_101)
        kps = [cv2.KeyPoint(float(x) + ORB_PAD, float(y) + ORB_PAD, ORB_PATCH, 0.0, 0.0, 0, int(i))
               for i, (x, y) in enumerate(kp)]
        out_k, desc = self._orb.compute(pad, kps)
        res = np.zeros((len(kp), 32), np.uint8)
        if desc is not None and len(out_k):
            cid = np.array([k.class_id for k in out_k], np.int64)
            res[cid] = desc
            if len(cid) != len(kp):
                self.stats["orb_dropped"] = self.stats.get("orb_dropped", 0) + len(kp) - len(cid)
        return res

    # ---- the per-frame entry point ---------------------------------------------------------------------------
    def extract(self, gray, budget=None, pred_uv=None):
        n = int(budget or self.budget)
        H, W = gray.shape
        rc = self._response(gray, n)
        if self._prev_img is not None:
            kp, ids, ages = self._track(gray, pred_uv)
            kp, ids, ages = self._suppress(kp, ids, ages, W, H, self._min_dist(W, H, n))
        else:
            kp = np.zeros((0, 2), np.float32)
            ids = np.zeros(0, np.int64)
            ages = np.zeros(0, np.int64)
        new = self._detect(gray, kp, n - len(kp), n, resp_cache=rc)
        if len(new):
            nid = np.arange(self._next_id, self._next_id + len(new), dtype=np.int64)
            self._next_id += len(new)
            kp = np.concatenate([kp, new], axis=0)
            ids = np.concatenate([ids, nid])
            ages = np.concatenate([ages, np.zeros(len(new), np.int64)])
        self.delivered = int(len(kp))
        desc = self.describe(gray, kp)
        self._prev_img = gray
        self._prev_kp = kp
        self._prev_id = ids
        self._prev_age = ages
        self.n_frames += 1
        return {"keypoints": kp.astype(np.float32), "track_id": ids, "age": ages, "descriptors": desc,
                "image_size": (W, H)}

    # ---- XC-07: keyframe-time re-anchoring (ARM XCLEAN_REANCHOR=1) --------------------------------------------
    def reanchor(self, gray, kp, ids):
        """At a keyframe, re-measure every track against its BIRTH-keyframe patch with the affine aligner
        (stages/align_cuda.py), so the keyframe observation is a measurement against a fixed template, not the end of
        a frame-to-frame KLT chain (chain drift q90 at 10 frames: 0.7-1.1 px EuRoC, 4-20 px KITTI; spec XC-07).
        The affine seed is the least-squares local affine of the >= 4 nearest tracks born at the same keyframe
        (their anchor -> current displacements): data-driven, no pose needed.  A track whose chain lies more than
        3 x FEATURE_SIGMA_PX from its anchored position ends its association (returned in `ended`; the caller gives
        it a new id), a track the aligner cannot verify keeps its chained position.  Tracks born since the previous
        keyframe are anchored here.  -> (kp_new, ended bool, ids with ended tracks renumbered)."""
        from scipy.spatial import cKDTree
        kp = np.asarray(kp, np.float32).copy()
        ended = np.zeros(len(kp), bool)
        if self._aligner is None:
            from xclean.stages.align_cuda import CudaPatchAligner
            self._aligner = CudaPatchAligner()
        self._kf_count += 1
        slot = self._kf_count
        cur_pyr = self._aligner.pyramid(gray)
        have = np.array([int(t) in self._anchor for t in ids], bool)
        self.stats["reanchor_kf"] = self.stats.get("reanchor_kf", 0) + 1
        if have.any():
            idx = np.nonzero(have)[0]
            slots = np.array([self._anchor[int(ids[i])][0] for i in idx])
            apos = np.array([self._anchor[int(ids[i])][1:] for i in idx], np.float32)
            for sl in np.unique(slots):
                sel = idx[slots == sl]
                ap = apos[slots == sl]
                ref = self._kf_imgs.get(int(sl))
                if ref is None:
                    continue
                cur = kp[sel]
                A0 = np.tile(np.eye(2, dtype=np.float32), (len(sel), 1, 1))
                if len(sel) >= 5:
                    tree = cKDTree(ap)
                    k = min(8, len(sel))
                    _d, nb = tree.query(ap, k=k)
                    for j in range(len(sel)):
                        P = ap[nb[j]] - ap[j]
                        Q = cur[nb[j]] - cur[j]
                        X = np.hstack([P, np.ones((len(P), 1), np.float32)])
                        sol, *_ = np.linalg.lstsq(X, Q, rcond=None)
                        A0[j] = sol[:2].T
                c, A, ok, zn = self._aligner.align(None, ap, None, cur, A0, ref_pyr=ref, cur_pyr=cur_pyr)
                dev = np.linalg.norm(c - cur, axis=1)
                good = ok & (dev <= 3.0 * FEATURE_SIGMA_PX)
                bad = ok & ~good
                kp[sel[good]] = c[good]
                ended[sel[bad]] = True
                self.stats["reanchored"] = self.stats.get("reanchored", 0) + int(good.sum())
                self.stats["reanchor_ended"] = self.stats.get("reanchor_ended", 0) + int(bad.sum())
                self.stats["reanchor_unverified"] = self.stats.get("reanchor_unverified", 0) + int((~ok).sum())
                if good.any():
                    self._reanchor_dev.extend(dev[good][:32].tolist())
        # ended tracks get NEW ids (the frontend owns ids), then everything unanchored is anchored here
        ids = np.array(ids, np.int64).copy()
        if ended.any():
            ids[ended] = self.new_ids(int(ended.sum()))
        self._kf_imgs[slot] = cur_pyr
        for key in [k for k in self._kf_imgs if k < slot - 32]:
            del self._kf_imgs[key]                       # bounded by ~ the estimator window
        live = set(int(t) for t in ids)
        self._anchor = {t: v for t, v in self._anchor.items() if t in live and v[0] in self._kf_imgs}
        for i, t in enumerate(ids):
            if ended[i] or int(t) not in self._anchor:
                self._anchor[int(t)] = (slot, float(kp[i, 0]), float(kp[i, 1]))
        self._prev_kp = kp.copy()
        self._prev_id = ids.copy()
        return kp, ended, ids

    def add_points(self, feats, pts):
        """XC-01: append re-acquired points to the CURRENT frame (new track ids, age 0, ORB descriptors), in feats
        and in the tracker state KLT continues from.  -> the new keypoints' indices in feats."""
        pts = np.asarray(pts, np.float32).reshape(-1, 2)
        n0 = len(feats["keypoints"])
        if not len(pts):
            return np.zeros(0, np.int64)
        nid = self.new_ids(len(pts))
        d = self.describe(self._prev_img, pts)
        feats["keypoints"] = np.concatenate([feats["keypoints"], pts]).astype(np.float32)
        feats["track_id"] = np.concatenate([np.asarray(feats["track_id"], np.int64), nid])
        feats["age"] = np.concatenate([np.asarray(feats["age"], np.int64), np.zeros(len(pts), np.int64)])
        feats["descriptors"] = np.concatenate([feats["descriptors"], d])
        self._prev_kp = feats["keypoints"].copy()
        self._prev_id = feats["track_id"].copy()
        self._prev_age = feats["age"].copy()
        self.stats["reacquired_points"] = self.stats.get("reacquired_points", 0) + len(pts)
        return np.arange(n0, n0 + len(pts), dtype=np.int64)

    def new_ids(self, n):
        nid = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._next_id += n
        return nid

    def reset(self):
        """Forget the previous frame (no tracks carried).  Used only by the tracker's bootstrap re-extraction."""
        self._prev_img = self._prev_kp = self._prev_id = self._prev_age = None

    def prev_keypoints(self):
        """The previous frame's keypoints, in feats order (what a pred_uv array must be aligned to)."""
        return self._prev_kp

    # ---- correspondence ----------------------------------------------------------------------------------------
    @staticmethod
    def match(f0, f1):
        """Track-id join -> (i0, i1) int64 index arrays (the same physical KLT track in both)."""
        a, b = f0["track_id"], f1["track_id"]
        if len(a) == 0 or len(b) == 0:
            z = np.zeros(0, np.int64)
            return z, z
        _c, i0, i1 = np.intersect1d(a, b, assume_unique=True, return_indices=True)
        return i0.astype(np.int64), i1.astype(np.int64)

    def match_desc(self, f0, f1, mask0=None, mask1=None):
        """Prediction-free ORB matching: mutual nearest neighbour, Lowe ratio, Hamming chance bound.
        mask0 / mask1 restrict the candidate keypoints (e.g. only keypoints that carry a landmark).
        -> (i0, i1) int64."""
        d0, d1 = f0["descriptors"], f1["descriptors"]
        i0 = np.arange(len(d0)) if mask0 is None else np.nonzero(mask0)[0]
        i1 = np.arange(len(d1)) if mask1 is None else np.nonzero(mask1)[0]
        z = np.zeros(0, np.int64)
        if len(i0) < 2 or len(i1) < 2:
            return z, z
        A = np.ascontiguousarray(d0[i0])
        B = np.ascontiguousarray(d1[i1])
        fw = self._bf.knnMatch(A, B, k=2)
        bw = self._bf.knnMatch(B, A, k=1)
        back = np.full(len(B), -1, np.int64)
        for m in bw:
            if m:
                back[m[0].queryIdx] = m[0].trainIdx
        r0, r1 = [], []
        for m in fw:
            if len(m) < 2:
                continue
            a, s = m[0], m[1]
            if a.distance > HAMMING_CHANCE or a.distance >= RATIO * s.distance:
                continue
            if back[a.trainIdx] != a.queryIdx:
                continue
            r0.append(a.queryIdx)
            r1.append(a.trainIdx)
        if not r0:
            return z, z
        return i0[np.asarray(r0)].astype(np.int64), i1[np.asarray(r1)].astype(np.int64)

    # ---- the control loop ----------------------------------------------------------------
    def report_inliers(self, n_inliers):
        self._recent.append(int(n_inliers))
        if len(self._recent) > self.SUFFICIENCY:
            self._recent.pop(0)
        need = MIN_PNP_INLIERS
        dials = {"budget": self.budget, "kp_floor": self.floor, "min_inliers": need,
                 "median_inliers": float(np.median(self._recent)), "n_samples": len(self._recent),
                 "cooldown": self._cooldown}
        if len(self._recent) < self.SUFFICIENCY:
            return self._log(Decision("feature_budget", True, "insufficient evidence", dials))
        if self._cooldown > 0:
            self._cooldown -= 1
            return self._log(Decision("feature_budget", True, "hysteresis", dials))
        med = float(np.median(self._recent))
        old = self.budget
        if (self.detect_mode == "cell" and med < self.HEADROOM * need and self.delivered is not None
                and self.delivered < self.budget):
            # XC-03: the image cannot deliver the present budget - growing it adds nothing but cost (the controller
            # wound up to the cap on mh_04 / kitti_01 while delivery was starved)
            return self._log(Decision("feature_budget", True, "supply-limited: delivered %d < budget %d"
                                      % (self.delivered, self.budget), dials))
        if med < self.HEADROOM * need and self.budget < self.cap:
            self.budget = min(self.cap, int(self.budget * self.GROW_FACTOR))
            why = "inliers %.0f within %gx of the %d needed" % (med, self.HEADROOM, need)
        elif med > self.COMFORT * need and self.budget > self.floor:
            self.budget = max(self.floor, int(self.budget * self.SHRINK_FACTOR))
            why = "inliers %.0f above %gx the %d needed" % (med, self.COMFORT, need)
        else:
            return self._log(Decision("feature_budget", True, "in band", dials))
        self._cooldown = self.HYSTERESIS
        dials["budget"] = self.budget
        return self._log(Decision("feature_budget", True, "%s: %d -> %d" % (why, old, self.budget), dials))

    def _log(self, d):
        self.decisions.append(d)
        return d

    def dial_summary(self):
        moves = [d for d in self.decisions if "->" in d.reason]
        return {"final_budget": self.budget, "floor": self.floor, "moves": len(moves),
                "last": moves[-1].reason if moves else "none", "tracking": dict(self.stats)}
