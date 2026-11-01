#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
# Vendored from integrity.py of our own earlier research harness (same company).  Verbatim except
# the lines marked [vendored]: the imports point at the vendored copies (xclean.eval.dataset_audit,
# xclean.io.rows, xclean.obs.decision), the verdict cache is OPTIONAL and its path is a parameter
# ($XCLEAN_INTEGRITY_CACHE; unset = no cache: every check runs the audit), and _main takes tags or folders.  Two
# reason texts (duplicated IMU stamps, accelerometer rails) were reworded for the release.
"""Dataset integrity check: a sequence folder that fails it is not scored, and the reason is printed.

It runs the same checks as `xclean audit` (frame and timestamp counts, IMU timestamps and rates, calibration blocks,
ground-truth coverage) and makes the verdict binding:

  * FATAL flags stop the run from being scored (for example a frame count that disagrees with timestamps.csv);
  * ADVISORY flags are printed next to the score (for example ground truth that covers only part of the run);
  * verdicts can be cached per folder, keyed by the modification times of the files the check reads
    ($XCLEAN_INTEGRITY_CACHE; unset = no cache).

    python -m xclean.eval.integrity <sequence dir> [<sequence dir> ...] [--no-cache]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from xclean.eval import dataset_audit as dataset_integrity  # noqa: E402  the auditor  [vendored]
from xclean.io import rows as rowreg                  # noqa: E402  [vendored]
from xclean.obs.decision import Gate                  # noqa: E402  [vendored]

# [vendored] was <research tree>/an internal note; now optional, None = no cache
CACHE = os.environ.get("XCLEAN_INTEGRITY_CACHE") or None

# --------------------------------------------------------------------------- #
# severity — explicit patterns, each with the reason it is where it is.
#
# This is a POLICY and it is written out rather than inferred, because the first
# draft inferred it from substring matches and promptly misclassified "calib has
# NO imu_noise block" as fatal by matching "no imu".  A classifier that is wrong
# in a plausible-looking way is the exact failure mode this project keeps paying
# for.
# --------------------------------------------------------------------------- #
import re                                              # noqa: E402

FATAL_PATTERNS = [
    # the row's number would not be about the run
    (r"frames?\b.*(mismatch|disagree)|num_frames|timestamps\.csv rows",
     "frame bookkeeping disagrees — the run did not see the dataset"),
    (r"frame times? (are )?(non-monotonic|not monotonic)",
     "non-monotonic frame time: replay order is undefined"),
    (r"gt (is )?non-monotonic",
     "non-monotonic ground truth: the association is undefined"),
    (r"\bdt\s*<\s*0", "IMU time runs backwards"),
    (r"\bdt\s*==\s*0",
     "duplicated IMU timestamps — the engine silently DISCARDS these samples"),
    (r"gt covers only", "ground truth covers a fragment of the run"),
    (r"not a converted dataset|missing", "the dataset is not readable as one"),
]

# Accepted, but the caveat is printed WITH the number, every time.
LOUD_PATTERNS = [
    (r"no imu_noise",                                    # [vendored] text: what the engine does
     "no IMU noise values for this rig — the engine does not use the IMU "
     "(vision only), so this row's numbers are vision-only numbers"),
    (r"no imu\.jsonl",                                   # [vendored] new
     "an IMU rate is declared but there is no IMU file — replay waits for "
     "IMU data on every frame (slow lock-step; dropped frames when paced)"),
    (r"rails at", "accelerometer clipping — readings are cut at the end of the "
                  "sensor's range, which biases the IMU estimate"),
    (r"imu gaps", "IMU intervals the intake drops"),
]


def classify(flag: str):
    """-> ('fatal'|'loud'|'advisory', why)."""
    f = flag.lower()
    for pat, why in FATAL_PATTERNS:
        if re.search(pat, f):
            return "fatal", why
    for pat, why in LOUD_PATTERNS:
        if re.search(pat, f):
            return "loud", why
    return "advisory", ""


def _is_fatal(flag: str) -> bool:
    return classify(flag)[0] == "fatal"


def _fingerprint(dataset_dir):
    """mtime+size of every file the audit reads.  Cheap, and it invalidates
    itself when a converter re-writes the dataset — which is exactly the event
    the cache must not survive."""
    parts = []
    for name in ("calib.json", "timestamps.csv", "imu.csv", "gt.csv"):
        p = os.path.join(dataset_dir, name)
        if os.path.isfile(p):
            st = os.stat(p)
            parts.append("%s:%d:%d" % (name, st.st_size, int(st.st_mtime)))
    for sub in ("left", "right"):
        p = os.path.join(dataset_dir, sub)
        if os.path.isdir(p):
            try:
                parts.append("%s:%d" % (sub, len(os.listdir(p))))
            except OSError:
                pass
    return "|".join(parts)


class IntegrityGate(Gate):
    """Gate one converted dataset directory."""

    def __init__(self, use_cache=True):
        super().__init__("dataset_integrity")
        self.use_cache = use_cache
        self.cache = {}
        if use_cache and CACHE and os.path.isfile(CACHE):          # [vendored] CACHE may be None
            try:
                self.cache = json.load(open(CACHE))
            except Exception:
                self.cache = {}

    def _audit(self, dataset_dir):
        fp = _fingerprint(dataset_dir)
        hit = self.cache.get(dataset_dir)
        if self.use_cache and hit and hit.get("fingerprint") == fp:
            return hit["flags"], hit["info"], True
        res = dataset_integrity.audit(dataset_dir)
        if res is None:
            flags, info = ["not a converted dataset (no calib.json/timestamps.csv)"], ""
        else:
            flags, info = res
        self.cache[dataset_dir] = {"fingerprint": fp, "flags": list(flags),
                                   "info": info}
        return flags, info, False

    def check(self, dataset_dir):
        if not os.path.isdir(dataset_dir):
            return self.reject("dataset directory missing",
                               {"dataset": dataset_dir})
        flags, info, cached = self._audit(dataset_dir)
        graded = [(f,) + classify(f) for f in flags]
        fatal = [f for f, sev, _ in graded if sev == "fatal"]
        loud = [(f, why) for f, sev, why in graded if sev == "loud"]
        dials = {"n_flags": len(flags), "n_fatal": len(fatal),
                 "n_loud": len(loud), "cached": cached, "info": info}
        if fatal:
            return self.reject("; ".join(fatal), dials, margin=-len(fatal))
        if loud:
            return self.accept("LOUD: " + "; ".join(
                "%s  [%s]" % (f, why) for f, why in loud)
                + ("; advisory: " + "; ".join(
                    f for f, sev, _ in graded if sev == "advisory")
                   if any(s == "advisory" for _, s, _ in graded) else ""),
                dials, margin=0.0)
        if flags:
            return self.accept("advisory: " + "; ".join(flags), dials, margin=0.0)
        return self.accept("clean", dials, margin=1.0)

    def check_row(self, tag):
        ok, msg = rowreg.get(tag).exists()
        if not ok:
            return self.reject(msg, {"row": tag})
        return self.check(rowreg.get(tag).dataset)

    def save(self):
        if not self.use_cache or not CACHE:                         # [vendored] CACHE may be None
            return
        tmp = CACHE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.cache, f, indent=1)
        os.replace(tmp, CACHE)


def gate_rows(tags, use_cache=True, verbose=True):
    """-> (ok_tags, blocked). Blocked rows must not be launched or scored."""
    g = IntegrityGate(use_cache)
    ok, blocked = [], {}
    for tag in tags:
        d = g.check_row(tag)
        (ok.append(tag) if d.accepted else blocked.setdefault(tag, d.reason))
        if verbose:
            print("  %-24s %s" % (tag, d.line()))
    g.save()
    if verbose:
        print("  " + g.summary_line())
    return ok, blocked


def _main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="+", help="sequence folders, or row tags with --rows")  # [vendored]
    ap.add_argument("--rows", dest="registry", default=None,                               # [vendored]
                    help="row registry JSON for tag lookups (docs/SEQUENCE_FORMAT.md)")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    rowreg.set_registry(a.registry)                                                         # [vendored]
    tags = list(a.rows)                                                                     # [vendored]
    print("integrity gate — %d rows" % len(tags))                                         # [vendored]
    ok, blocked = gate_rows(sorted(tags), use_cache=not a.no_cache)
    print("\n%d/%d rows scoreable; %d BLOCKED" % (len(ok), len(tags), len(blocked)))
    for tag, why in blocked.items():
        print("  BLOCKED %-24s %s" % (tag, why))
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(_main())
