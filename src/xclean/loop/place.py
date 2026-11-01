#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""PLACE RECOGNITION - a global descriptor per keyframe, and nothing else (xclean: our bag of words).

    pd = PlaceDescriptor()
    v = pd.describe(gray)          # (V,) float32 tf-idf, L2-normalised: cosine = dot product

the engine used EigenPlaces (ResNet50, torch hub) here.  xclean replaces it with our own vocabulary (loop/bow.py,
trained by our own tool on licence-clean images that are not benchmark sequences) over standard multi-scale ORB
(OpenCV, Apache-2.0).  The contract is unchanged: a PURE function of the image, L2-normalised, so the closer's
`D @ me` and the relocaliser's top-k are untouched.

★ RETRIEVAL IS DELIBERATELY PERMISSIVE (unchanged): it SURFACES candidates; geometric verification decides.
"""
from __future__ import annotations

import numpy as np

from xclean.loop.bow import Vocabulary, DEFAULT_VOCAB

N_FEATURES = 1000          # the ORB detector's own default operating point for one image
N_LEVELS = 4               # scale pyramid for retrieval (revisits at a different range)

_VOC = {}


def _orb():
    import cv2
    return cv2.ORB_create(nfeatures=N_FEATURES, scaleFactor=1.2, nlevels=N_LEVELS, edgeThreshold=31,
                          firstLevel=0, WTA_K=2, patchSize=31)


def orb_descriptors(gray, orb=None):
    orb = orb or _orb()
    _k, d = orb.detectAndCompute(np.ascontiguousarray(gray), None)
    return np.zeros((0, 32), np.uint8) if d is None else d


class PlaceDescriptor:
    def __init__(self, vocab_path=None, verbose=True):
        path = vocab_path or DEFAULT_VOCAB
        if path not in _VOC:
            _VOC[path] = Vocabulary.load(path)
        self.voc = _VOC[path]
        self.path = path
        self.dim = self.voc.V
        self._orb = _orb()
        if verbose:
            m = self.voc.meta
            print(" [place] our BoW: %d words (k=%d, L=%d), trained on %s images (%s); ORB %d features x %d "
                  "levels" % (self.dim, self.voc.k, self.voc.L, m.get("n_images"), m.get("source", "?"),
                              N_FEATURES, N_LEVELS))

    def key(self, dataset=""):
        from xclean.stages.cache import build_key, sha1_file
        return build_key(stage="bow_place", dataset=str(dataset), vocab=sha1_file(self.path)[:16],
                         nfeat=N_FEATURES, nlev=N_LEVELS)

    def describe(self, gray):
        return self.voc.transform(orb_descriptors(gray, self._orb))

    def describe_full(self, gray):
        """-> (BoW vector, keypoints (n,2) float32, descriptors (n,32) uint8): the same ORB set feeds retrieval and
        the loop verification (xclean c2: the rich set gives 60 median geometric inliers on mh_04 revisits vs 16 for
        the tracker's ~200 keypoints)."""
        k, d = self._orb.detectAndCompute(np.ascontiguousarray(gray), None)
        if d is None or not k:
            return self.voc.transform(None), np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8)
        uv = np.array([p.pt for p in k], np.float32)
        return self.voc.transform(d), uv, d
