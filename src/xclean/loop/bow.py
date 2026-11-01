#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""OUR BAG OF WORDS - a hierarchical k-medians vocabulary over binary ORB descriptors, tf-idf, cosine scoring.

    voc = Vocabulary.load(path)            # trained by xclean/tools/train_vocab.py
    v = voc.transform(descriptors)          # (V,) float32 tf-idf, L2-normalised -> cosine = dot product

The method is the classic one (Nister & Stewenius 2006 vocabulary tree; Sivic & Zisserman 2003 tf-idf with cosine
scoring; binary descriptors with bitwise-majority centroids and k-means++ seeding as in Galvez-Lopez & Tardos 2012).
The CODE and the VOCABULARY are ours: no DBoW2/ORB-SLAM3 code and never ORB-SLAM3's vocabulary file (GPLv3).
The vocabulary is trained only on licence-clean images that are not benchmark sequences (its provenance file,
vocab/xclean_orb_k10L4_provenance.json, lists them).

Cosine (L2 tf-idf) rather than DBoW2's L1 score is chosen because the loop closer and the relocaliser already score
candidates by a dot product against a matrix of per-keyframe vectors (closer.py `D @ me`, tracker._relocalize), so the
retrieval code path is unchanged; only the population the retrieval rule sees is new (see closer.py).
"""
from __future__ import annotations

import json
import os

import numpy as np


def _hamming_to(desc, centers):
    """desc (n,32) uint8, centers (k,32) uint8 -> (n,k) int distances."""
    a = desc.view(np.uint64)[:, None, :]
    b = centers.view(np.uint64)[None, :, :]
    return np.bitwise_count(np.bitwise_xor(a, b)).sum(axis=2)


def _majority(desc):
    """Bitwise-majority centroid of (m,32) uint8 descriptors (ties -> 0)."""
    bits = np.unpackbits(desc, axis=1)
    return np.packbits((bits.sum(axis=0) * 2 > len(desc)).astype(np.uint8))[None, :]


def _kmedians(desc, k, rng, iters=10):
    """Binary k-medians with k-means++ seeding -> (centers (k',32), labels).  k' <= k (duplicates dropped)."""
    n = len(desc)
    if n <= k:
        return desc.copy(), np.arange(n)
    c = [desc[rng.integers(n)]]
    dmin = _hamming_to(desc, np.asarray(c)).min(axis=1).astype(np.float64)
    for _ in range(1, k):
        p = dmin ** 2
        s = p.sum()
        if s <= 0:
            break
        j = rng.choice(n, p=p / s)
        c.append(desc[j])
        dmin = np.minimum(dmin, _hamming_to(desc, desc[j:j + 1])[:, 0])
    C = np.ascontiguousarray(np.asarray(c, np.uint8))
    lab = None
    for _ in range(iters):
        dist = np.empty((n, len(C)), np.int64)
        for s0 in range(0, n, 200000):
            dist[s0:s0 + 200000] = _hamming_to(desc[s0:s0 + 200000], C)
        nl = dist.argmin(axis=1)
        if lab is not None and np.array_equal(nl, lab):
            break
        lab = nl
        newC = []
        for j in range(len(C)):
            m = desc[lab == j]
            newC.append(_majority(m)[0] if len(m) else C[j])
        C = np.ascontiguousarray(np.asarray(newC, np.uint8))
    return C, lab


class Vocabulary:
    """A k-ary tree stored flat: node 0 is the root; children[node] = (first_child, n_children) into `centers`;
    a leaf (n_children == 0) has a word id."""

    def __init__(self, centers, child_first, child_n, word_of, idf, meta):
        self.centers = centers            # (n_nodes, 32) uint8 (root's row unused)
        self.child_first = child_first    # (n_nodes,) int64
        self.child_n = child_n            # (n_nodes,) int64
        self.word_of = word_of            # (n_nodes,) int64, -1 for internal nodes
        self.idf = idf                    # (V,) float32
        self.meta = meta
        self.V = int(len(idf))
        self.k = int(meta.get("k", 10))
        self.L = int(meta.get("L", 4))

    # ---- training ------------------------------------------------------------------------------------------
    @classmethod
    def train(cls, desc, k=10, L=4, seed=0, image_index=None, verbose=True):
        """desc (n,32) uint8 training descriptors; image_index (n,) the training image of each (for idf)."""
        rng = np.random.default_rng(seed)
        centers = [np.zeros(32, np.uint8)]
        child_first = [0]
        child_n = [0]
        level = [0]
        members = {0: np.arange(len(desc))}
        frontier = [0]
        for lev in range(L):
            nxt = []
            for node in frontier:
                idx = members.pop(node)
                if len(idx) < 2 * k:              # too few to split: the node stays a leaf
                    continue
                C, lab = _kmedians(desc[idx], k, rng)
                first = len(centers)
                for j in range(len(C)):
                    centers.append(C[j])
                    child_first.append(0)
                    child_n.append(0)
                    level.append(lev + 1)
                    members[first + j] = idx[lab == j]
                    nxt.append(first + j)
                child_first[node] = first
                child_n[node] = len(C)
            frontier = nxt
            if verbose:
                print("  [vocab] level %d: %d nodes" % (lev + 1, len(frontier)), flush=True)
        child_n = np.asarray(child_n, np.int64)
        word_of = np.full(len(centers), -1, np.int64)
        leaves = np.nonzero(child_n == 0)[0]
        leaves = leaves[leaves != 0]
        word_of[leaves] = np.arange(len(leaves))
        voc = cls(np.asarray(centers, np.uint8), np.asarray(child_first, np.int64), child_n, word_of,
                  np.ones(len(leaves), np.float32), {"k": k, "L": L, "seed": seed, "n_train": int(len(desc))})
        if image_index is not None:
            w = voc.words(desc)
            n_img = int(image_index.max()) + 1
            pres = np.zeros(len(leaves), np.int64)
            pairs = np.unique(np.stack([w, image_index], 1), axis=0)
            np.add.at(pres, pairs[:, 0], 1)
            voc.idf = np.log(n_img / np.maximum(pres, 1)).astype(np.float32)
            voc.meta["n_images"] = n_img
        return voc

    # ---- quantisation --------------------------------------------------------------------------------------
    def words(self, desc):
        """(n,32) uint8 -> (n,) word ids, by descending the tree (vectorised per level)."""
        n = len(desc)
        node = np.zeros(n, np.int64)
        for _ in range(self.L + 1):
            cn = self.child_n[node]
            act = cn > 0
            if not act.any():
                break
            ia = np.nonzero(act)[0]
            first = self.child_first[node[ia]]
            kk = int(cn[ia].max())
            cand = first[:, None] + np.arange(kk)[None, :]                      # (m, kk)
            valid = np.arange(kk)[None, :] < cn[ia][:, None]
            cand_c = np.where(valid, cand, first[:, None])
            C = self.centers[cand_c]                                             # (m, kk, 32)
            d = np.bitwise_count(np.bitwise_xor(desc[ia].view(np.uint64)[:, None, :],
                                                C.reshape(len(ia), kk, 32).view(np.uint64))).sum(axis=2)
            d = np.where(valid, d, 1 << 20)
            node[ia] = cand[np.arange(len(ia)), d.argmin(axis=1)]
        w = self.word_of[node]
        return w

    def transform(self, desc):
        """tf-idf, L2-normalised (V,) float32; the zero vector for an empty input."""
        v = np.zeros(self.V, np.float32)
        if desc is None or len(desc) == 0:
            return v
        w = self.words(np.ascontiguousarray(desc, np.uint8))
        w = w[w >= 0]
        if not len(w):
            return v
        cnt = np.bincount(w, minlength=self.V).astype(np.float32)
        v = (cnt / cnt.sum()) * self.idf
        nrm = float(np.linalg.norm(v))
        return v / nrm if nrm > 0 else v

    # ---- io ------------------------------------------------------------------------------------------------
    def save(self, path):
        np.savez_compressed(path, centers=self.centers, child_first=self.child_first, child_n=self.child_n,
                            word_of=self.word_of, idf=self.idf, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["centers"], z["child_first"], z["child_n"], z["word_of"], z["idf"],
                   json.loads(str(z["meta"])))


DEFAULT_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab", "xclean_orb_k10L4.npz")
