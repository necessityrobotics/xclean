#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""CONTENT-ADDRESSED CACHING FOR PURE STAGES - depth, features, descriptors.

    cache = StageCache(root, key=stage.key(), shape=(H, W), mode="rw")
    hit = cache.lookup(idx, *inputs)
    if hit is None:
        out = stage.compute(*inputs); cache.store(idx, out, *inputs)

WHY A STAGE IS CACHEABLE AT ALL (plan section 4.3)
    Depth reads no pose, no IMU and no map: it is `RAFT(rectify(resize(L,R)))`,
    a pure function of the input images and the model.  The ESTIMATOR changes
    every build; the disparity does not.  So a benchmark row re-run against a
    new build recomputes, byte for byte, the numbers it computed last time -
    and depth is **30-68 % of wall time on every dataset family** (malaga 68 %,
    madmax 61 %, kitti 57 %, mh 39 %; the "depth is 12 %" figure came from one
    MH_01 profile and holds nowhere else).  Measured on the existing cache:
    kitti07 358.2 -> 150.5 s (2.38x) and mh02 1149 -> 716.2 s (1.60x), both
    bit-identical.

    **A dataset pays full price exactly once, ever.**  That is iteration speed,
    not product latency, and the two are never conflated (plan section 3.2).

★ WHY THE KEY IS THE INPUTS AND THE MODEL, AND NOT THE ENGINE BUILD
    Keying on the build would invalidate the cache on every commit and buy
    nothing: the stage is pure, so its output cannot depend on the estimator.
    Keying on the inputs is what makes the cache survive the work.

★ TWO LAYERS OF INVALIDATION, AND NEITHER IS OPTIONAL
    This project has been bitten twice by a dataset changing underneath a
    cached assumption - a converter that silently dropped gyro calibration, and
    a `.jpg` dataset read as `.png`.  Both of the real defects of an entire
    fix-loop week were in OUR CONVERTERS, not in the engine.  So:

      1. a per-DATASET key (paths, resolutions, calibration digest, model
         digest, iteration count, precision).  A mismatch abandons the whole
         directory rather than trying to repair it;
      2. a per-ITEM digest of the actual input bytes, written into the entry
         header and checked on EVERY read.  A mismatch recomputes that item.

    Layer 1 is cheap and catches a config change.  Layer 2 costs ~1 ms against
    the 100-260 ms it protects and is **the only thing that catches a dataset
    edited in place**.  A cache with only layer 1 is a cache that will one day
    serve confident, stale, wrong numbers.

WHAT IS CACHED IS THE UPSTREAM VALUE, DELIBERATELY
    Disparity, not depth.  `depth = f*B/disp` and the depth band are re-derived
    per run from rig geometry (Class B, plan section 6); caching the derived
    value would freeze them, and a calibration fix would then replay stale
    depths forever.  Caching upstream keeps every derived quantity live.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct

import numpy as np

MAGIC = b"XONE"
HDR = struct.Struct("<4sHHIIQ")        # magic, ver, dtype_code, h, w, digest
VERSION = 1
_DTYPES = {0: np.float32, 1: np.float16, 2: np.uint8, 3: np.int32}
_CODES = {v: k for k, v in _DTYPES.items()}


def sha1_file(path, limit=None):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if limit and f.tell() >= limit:
                break
    return h.hexdigest()


def digest_arrays(*arrays):
    """A 64-bit digest of the ACTUAL bytes of every input to the stage.

    `np.ascontiguousarray` is not cosmetic: a view with a different stride has
    the same values and different bytes, and a digest that changed with the
    slicing would invalidate a valid cache on a refactor.
    """
    h = hashlib.blake2b(digest_size=8)
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return int.from_bytes(h.digest(), "little")


def build_key(**fields):
    """The per-dataset key: a stable digest of everything that changes the
    output, PLUS the readable fields, so a mismatch can be explained rather
    than merely detected.  A cache that can only say "miss" teaches nothing.
    """
    blob = json.dumps(fields, sort_keys=True, default=str)
    return {"digest": hashlib.sha1(blob.encode()).hexdigest(), "fields": fields}


def _slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")[:64] or "stage"


class StageCache:
    """One directory per (stage, dataset, config).  One file per item, so it is
    resumable after any interruption and prunable per row - and it never
    preallocates the 288 GB that one madmax row would need as a single array.
    """

    def __init__(self, root, key, name="depth", dataset="", mode="rw",
                 verbose=True):
        self.mode = mode                       # "off" | "ro" | "rw"
        self.key = key
        self.name = name
        self.hits = self.misses = self.stale = self.rejects = 0
        self.dir = os.path.join(
            root, "%s-%s-%s" % (_slug(name), _slug(os.path.basename(
                str(dataset).rstrip("/\\"))), key["digest"][:8]))
        self.enabled = mode != "off"
        if not self.enabled:
            return
        os.makedirs(self.dir, exist_ok=True)
        self._manifest = os.path.join(self.dir, "manifest.json")
        if os.path.isfile(self._manifest):
            old = json.load(open(self._manifest, encoding="utf-8"))
            if old.get("key", {}).get("digest") != key["digest"]:
                # layer 1: a whole-directory mismatch is ABANDONED, not merged
                self._write_manifest(reason="key mismatch, directory reset")
                if verbose:
                    print(" [cache] key mismatch -> %s reset" % self.dir)
        else:
            self._write_manifest(reason="created")

    def _write_manifest(self, reason=""):
        tmp = self._manifest + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"key": self.key, "name": self.name, "reason": reason,
                       "hits": self.hits, "misses": self.misses,
                       "stale": self.stale, "version": VERSION}, f, indent=1)
        os.replace(tmp, self._manifest)

    def path(self, idx):
        return os.path.join(self.dir, "%06d.bin" % int(idx))

    # ---- read / write --------------------------------------------------
    def lookup(self, idx, *inputs):
        """Return the cached array, or None.  Layer 2 runs on every read."""
        if not self.enabled:
            return None
        p = self.path(idx)
        if not os.path.isfile(p):
            self.misses += 1
            return None
        try:
            with open(p, "rb") as f:
                head = f.read(HDR.size)
                if len(head) != HDR.size:
                    self.rejects += 1
                    return None
                magic, ver, code, h, w, dig = HDR.unpack(head)
                if magic != MAGIC or ver != VERSION or code not in _DTYPES:
                    self.rejects += 1
                    return None
                if dig != digest_arrays(*inputs):
                    # the dataset changed underneath us.  This is the check
                    # that catches an in-place edit, and it is why layer 1
                    # alone is not enough.
                    self.stale += 1
                    return None
                a = np.frombuffer(f.read(), dtype=_DTYPES[code])
                if a.size != h * w:
                    self.rejects += 1
                    return None
                self.hits += 1
                return a.reshape(h, w)
        except OSError:
            self.rejects += 1
            return None

    def store(self, idx, arr, *inputs):
        if not self.enabled or self.mode == "ro":
            return False
        arr = np.ascontiguousarray(arr)
        if arr.dtype.type not in _CODES or arr.ndim != 2:
            raise TypeError("cacheable output must be a 2-D %s array, got %s %s"
                            % (list(_CODES), arr.dtype, arr.shape))
        p = self.path(idx)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            f.write(HDR.pack(MAGIC, VERSION, _CODES[arr.dtype.type],
                             arr.shape[0], arr.shape[1],
                             digest_arrays(*inputs)))
            f.write(arr.tobytes())
        os.replace(tmp, p)      # atomic: an interruption never leaves a half
        return True             # entry that a later run would trust

    def report(self):
        n = self.hits + self.misses
        return ("[cache %s] %d/%d hits (%.1f %%), %d stale, %d rejected -> %s"
                % (self.name, self.hits, n, 100.0 * self.hits / max(n, 1),
                   self.stale, self.rejects, self.dir))

    def close(self):
        if self.enabled:
            self._write_manifest(reason="closed")
