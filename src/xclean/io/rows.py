# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""SEQUENCE LOOKUP - a sequence is a FOLDER on disk; a row tag is an optional name for one.

    from xclean.io import rows
    r = rows.get("/data/xclean/MH_04_difficult")             # a sequence folder (docs/SEQUENCE_FORMAT.md)
    r = rows.get("/data/xclean/kitti_00", gt="/data/gt/kitti_00.csv")
    r = rows.get("mh_04_difficult")                           # a TAG: needs a registry file (below)
    r.dataset, r.gt_path, r.ext, r.family, r.tag

WHAT A ROW IS HERE
    Everything the engine and the scorers read for a row is: the sequence folder (calib.json, timestamps.csv,
    left/, right/, optional imu.jsonl, optional gt.csv), the image extension, an optional ground-truth file that
    lives OUTSIDE the folder, and the family (only "oak" is read: an own recording without ground truth scores as
    NO_GT instead of failing).  `Row` carries exactly that.

TAGS (compatibility with the research harness)
    The research tree looked rows up by tag in an internal registry that held machine-specific paths.  Here the
    same tag-based call works when a REGISTRY FILE is given: a JSON object {tag: {"dataset": DIR, "gt": FILE or null,
    "ext": "png"|"jpg", "family": STR, "sequence": STR}} (extra keys are ignored).  It is read from
      * set_registry(path)       (the CLIs' --rows FILE; in-process only, never exported to the environment), or
      * the environment variable XCLEAN_ROWS (note: run summaries record every XCLEAN_* variable as provenance).
    Resolution order of get(spec): a Row object -> rows registered in this process (get() with keyword metadata
    registers the row under `spec`, so a later get(spec) returns the same metadata) -> the registry file, when given
    and holding `spec` -> an existing folder `spec` -> KeyError.

Origin: replaces the row registry of our own earlier harness; Row.exists()
is copied verbatim from it.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

ROWS_ENV = "XCLEAN_ROWS"


@dataclass(frozen=True)
class Row:
    tag: str
    dataset: str
    gt: str | None = None                 # None -> <dataset>/gt.csv
    ext: str = "png"
    family: str = ""
    sequence: str = ""

    @property
    def gt_path(self) -> str:
        return self.gt or os.path.join(self.dataset, "gt.csv")

    # verbatim from our earlier research harness (rows) Row.exists
    def exists(self):
        if not os.path.isdir(self.dataset):
            return False, "dataset dir missing: %s" % self.dataset
        if not os.path.isfile(os.path.join(self.dataset, "calib.json")):
            return False, "calib.json missing in %s" % self.dataset
        if not os.path.isfile(self.gt_path):
            return False, "ground truth missing: %s" % self.gt_path
        return True, "ok"


_REGISTRY_PATH = None
_REGISTRY = None          # (path, {tag: Row}) cache
_MEMORY = {}              # rows registered in this process


def set_registry(path):
    """Use this registry file for tag lookups in this process (None = fall back to $XCLEAN_ROWS)."""
    global _REGISTRY_PATH, _REGISTRY
    _REGISTRY_PATH = os.path.abspath(path) if path else None
    _REGISTRY = None


def registry_path():
    return _REGISTRY_PATH or (os.environ.get(ROWS_ENV) or None)


def load_registry(path=None) -> dict:
    """{tag: Row} from a registry file ({} when no file is configured)."""
    global _REGISTRY
    p = path or registry_path()
    if not p:
        return {}
    if _REGISTRY is not None and _REGISTRY[0] == p:
        return _REGISTRY[1]
    out = {}
    for tag, d in json.load(open(p, encoding="utf-8")).items():
        out[tag] = Row(tag=tag, dataset=d["dataset"], gt=d.get("gt"), ext=d.get("ext", "png") or "png",
                       family=d.get("family", "") or "", sequence=d.get("sequence", tag) or tag)
    _REGISTRY = (p, out)
    return out


def _detect_ext(dataset):
    """'png' unless the first frame of timestamps.csv exists only as .jpg (OAK recordings, Malaga)."""
    try:
        with open(os.path.join(dataset, "timestamps.csv"), encoding="utf-8") as f:
            f.readline()
            for ln in f:
                p = ln.strip().split(",")
                if len(p) >= 2:
                    i = int(p[0])
                    break
            else:
                return "png"
    except (OSError, ValueError):
        return "png"
    left = os.path.join(dataset, "left", "%06d" % i)
    if not os.path.isfile(left + ".png") and os.path.isfile(left + ".jpg"):
        return "jpg"
    return "png"


def from_path(dataset, gt=None, ext=None, family=None, tag=None) -> Row:
    """A Row for a sequence folder given directly (no registry).  The folder string is kept exactly as given (as
    Dataset(path) keeps it), so every path the engine derives from it is the same string a registry would give."""
    d = str(dataset)
    return Row(tag=tag or os.path.basename(d.rstrip("/\\")), dataset=d, gt=str(gt) if gt else None,
               ext=ext or _detect_ext(d), family=family or "", sequence=os.path.basename(d.rstrip("/\\")))


def register(spec, row):
    """Make get(spec) return `row` for the rest of this process."""
    _MEMORY[str(spec)] = row
    return row


def get(spec, gt=None, ext=None, family=None, registry=None) -> Row:
    """A Row for a tag or a sequence folder (see the module docstring for the resolution order).  Keyword metadata
    (gt, ext, family) overrides what the registry / folder gives and registers the result under `spec`."""
    if isinstance(spec, Row):
        return spec
    key = str(spec)
    meta = {k: v for k, v in (("gt", gt), ("ext", ext), ("family", family)) if v is not None}
    if key in _MEMORY and not meta:
        return _MEMORY[key]
    reg = load_registry(registry)
    if key in reg:
        r = reg[key]
    elif os.path.isdir(key):
        r = from_path(key, gt=gt, ext=ext, family=family)
    else:
        raise KeyError("unknown row %r: not a sequence folder, and %s" % (
            key, ("not in the registry %s" % registry_path()) if (registry or registry_path())
            else "no registry file is configured (--rows FILE or $%s)" % ROWS_ENV))
    if meta:
        r = Row(tag=r.tag, dataset=r.dataset, gt=str(gt) if gt else r.gt, ext=ext or r.ext,
                family=family if family is not None else r.family, sequence=r.sequence)
        register(key, r)
    return r
