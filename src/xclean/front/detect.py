#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1 per-frame detection + description: re-detection on EVERY frame, with no KLT chain.

    det = Detector(rig)
    kp, desc = det(gray)                  # (N,2) float32 sub-pixel, (N,32) uint8 ORB (no orientation)
    det.report_yield(n_inliers)           # Class-C budget: the measured inlier yield of the last frames

Reused, not rewritten (DESIGN_P1 §14):
- `stages/features.py` Frontend._response (XC-03 CELL scope: quality 0.01 relative to the grid cell's max, plus the C31
  absolute floor = q99 of the pure-noise response at the frame's Immerkaer sigma_n);
- Frontend._detect (per-cell maxima, greedy minimum distance);
- Frontend._subpix (parabolic sub-pixel);
- Frontend.describe (ORB-31, no orientation, reflected border).

The cell detector failed only its KLT-CHAIN survival bar (C31: 0.54-0.59 vs 0.95). With per-frame re-detection that bar
does not apply. Cell coverage in the dark section is 0.24 -> 0.96 [C31].

Budget (Class C; XC-03 supply rule specialised to re-detection, where a track has no survival term):
N = ceil(3 x MIN_PNP_INLIERS / y), y = the running median, over the last window-length frames, of (window inliers of a
frame / its detections). Cold start y = COLD_MATCH_RATE x COLD_INLIER_RATE (derive.py, Class C cold start). Capped by
the front end's cap (2048) and never under the rig's conditioning floor.
"""
from __future__ import annotations

import math

import numpy as np

from xclean.core.derive import MIN_PNP_INLIERS, COLD_MATCH_RATE, COLD_INLIER_RATE


class Detector:
    def __init__(self, rig, cap=2048, window=10):
        from xclean.stages.features import Frontend
        self.fe = Frontend(rig, cap=cap, verbose=False)
        self.fe.detect_mode = "cell"                    # XC-03 (C31) cell scope + absolute noise floor
        self.rig = rig
        self.cap = int(cap)
        self.window = int(window)
        self._yields = []
        self.budget = self._budget()
        self.stats = {"frames": 0, "detected": 0}

    def _budget(self):
        y = float(np.median(self._yields)) if self._yields else COLD_MATCH_RATE * COLD_INLIER_RATE
        y = max(y, 1e-3)
        # target 3 x KF_COUNT (= 3 x 3 x MIN_PNP_INLIERS) window inliers, so the keyframe trigger (KF_COUNT) fires on a
        # real loss of overlap, not on the budget's own operating point (P1-02b)
        n = int(math.ceil(3 * 3 * MIN_PNP_INLIERS / y))
        return int(min(max(n, self.rig.conditioning_floor()), self.cap))

    def report_yield(self, n_inliers, n_detected):
        if n_detected > 0:
            self._yields.append(float(n_inliers) / float(n_detected))
            self._yields = self._yields[-self.window:]
        self.budget = self._budget()

    def __call__(self, gray):
        from xclean.front import fastpaths as FP
        fe = self.fe
        n = self.budget
        R, cand = fe._response(gray, n)
        if FP.on("e1") or FP.on("e2"):                  # latency fixes E1 / E2 (identical picks, fastpaths.py)
            kp = FP.detect_fast(fe, gray, n, n, R, cand)
        else:
            kp = fe._detect(gray, np.zeros((0, 2), np.float32), n, n, resp_cache=(R, cand))
        if len(kp):
            fx, fy = fe._subpix(R, kp[:, 0], kp[:, 1])
            kp = np.stack([fx, fy], axis=1).astype(np.float32)
        desc = FP.describe_fast(fe, gray, kp) if FP.on("e3") else fe.describe(gray, kp)
        self.stats["frames"] += 1
        self.stats["detected"] += int(len(kp))
        return kp, desc
