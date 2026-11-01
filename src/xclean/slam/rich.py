# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.3 worker: the per-keyframe RICH SET from the packet's rectified pair.

  * ORB, n = max(1000, rho_voc W H) keypoints x 4 levels, scale 1.2 (class C: rho_voc = the vocabulary's training density,
    ORB 1000 on TartanAir 640x480, xclean/loop/vocab/xclean_orb_k10L4_provenance.json; floored at the detector's operating
    point, PARAMETER_CLASSES max_keypoints rule): EuRoC 1175, KITTI 1519, 4Seasons 1042, MadMax 2593, Rosario / OAK 1000;
  * our BoW vector (xclean/loop/bow.py, vocabulary xclean_orb_k10L4.npz, licence-clean), tf-idf L2, stored sparse;
  * stereo 3D of every keypoint with P1's own 1-D stereo (xclean/front/stereo1d.disparities: ZNCC + 1-D LK, left-right
    check, its Gauss-Newton sigma), depth inside the rig's derived [near_clip, far_bearing] band.
Own code; the same calls and arithmetic as the design evidence (p13/tools/p13_richsets.py), so the rich sets are
byte-identical to the design caches (tested).  CPU only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

VOC_TRAIN_W, VOC_TRAIN_H, VOC_TRAIN_N = 640, 480, 1000


def rich_n(W, H):
    """Class-C rich-set size: density x area at the vocabulary's training density, floored at its operating point."""
    return max(VOC_TRAIN_N, int(round(VOC_TRAIN_N * float(W) * float(H) / (VOC_TRAIN_W * VOC_TRAIN_H))))


@dataclass
class RigParams:
    """What the worker needs of the rig (sent once at worker start; Class A in, Class B derived by core/derive.Rig)."""
    fx: float
    cx: float
    cy: float
    fB: float
    near: float
    far: float
    W: int
    H: int

    @classmethod
    def from_rig(cls, rig, W, H):
        return cls(float(rig.fx_rect), float(rig.cx), float(rig.cy), float(rig.fB_m_px), float(rig.near_clip_m),
                   float(rig.far_bearing_m), int(W), int(H))

    def pose_rig(self):
        """A minimal object with the fields track/pose.solve_pose reads (fx_rect, cx, cy)."""
        from types import SimpleNamespace
        return SimpleNamespace(fx_rect=self.fx, cx=self.cx, cy=self.cy)


@dataclass
class RichSet:
    uv: np.ndarray          # (n, 2) float32
    desc: np.ndarray        # (n, 32) uint8
    oct: np.ndarray         # (n,) int8
    X: np.ndarray           # (n, 3) float32, NaN where no stereo depth
    dsig: np.ndarray        # (n,) float32 disparity sigma, NaN where none
    bow_i: np.ndarray       # (m,) int32 word ids (ascending)
    bow_v: np.ndarray       # (m,) float32 tf-idf L2 weights
    t_ms: dict = None       # per-stage CPU ms (thread time)


class RichExtractor:
    def __init__(self, rig: RigParams, vocab_path=None):
        import cv2
        from xclean.loop.bow import Vocabulary, DEFAULT_VOCAB
        from xclean.loop.place import _orb
        self.rig = rig
        self.n = rich_n(rig.W, rig.H)
        self.orb = _orb() if self.n == VOC_TRAIN_N else cv2.ORB_create(
            nfeatures=self.n, scaleFactor=1.2, nlevels=4, edgeThreshold=31, firstLevel=0, WTA_K=2, patchSize=31)
        self.voc = Vocabulary.load(vocab_path or DEFAULT_VOCAB)
        from xclean.front import stereo1d
        self._stereo = stereo1d

    def extract(self, L, R):
        tt = time.thread_time
        t0 = tt()
        kps, desc = self.orb.detectAndCompute(np.ascontiguousarray(L), None)
        t1 = tt()
        if desc is None or not kps:
            uv = np.zeros((0, 2), np.float32)
            desc = np.zeros((0, 32), np.uint8)
            oc = np.zeros(0, np.int8)
        else:
            uv = np.array([p.pt for p in kps], np.float32)
            oc = np.array([p.octave for p in kps], np.int8)
        v = self.voc.transform(desc)
        t2 = tt()
        rg = self.rig
        d, s, ok, _zn = self._stereo.disparities(L, R, uv.astype(np.float64))
        z = np.where(ok & (d > 1e-6), rg.fB / np.maximum(d, 1e-6), np.inf)
        good = ok & (z >= rg.near) & (z <= rg.far)
        Xk = np.full((len(uv), 3), np.nan, np.float32)
        Xk[good, 0] = (uv[good, 0] - rg.cx) * z[good] / rg.fx
        Xk[good, 1] = (uv[good, 1] - rg.cy) * z[good] / rg.fx
        Xk[good, 2] = z[good]
        sg = np.where(good, s, np.nan).astype(np.float32)
        t3 = tt()
        nz = np.nonzero(v)[0]
        return RichSet(uv=uv, desc=desc, oct=oc, X=Xk, dsig=sg, bow_i=nz.astype(np.int32), bow_v=v[nz].astype(np.float32),
                       t_ms={"orb": 1e3 * (t1 - t0), "bow": 1e3 * (t2 - t1), "stereo": 1e3 * (t3 - t2)})
