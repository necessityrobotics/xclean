# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Folder rules shared by the commands: which output folder may be replaced, and whether an input folder is a
sequence folder (docs/SEQUENCE_FORMAT.md).

Replacing an output folder.  `xclean replay`, `xclean live` (its --out and --record folders) and the IMU capture
start from an empty output folder.  An existing folder is replaced only when it holds an earlier output of the same
kind and nothing else (the files that command writes, listed below).  These folders are never replaced: the current
folder, a drive root, the home folder, a folder that holds one of them, the folder of the xclean installation, and a
folder that holds, or lies inside, an input of the command.  Paths are compared after resolving them (absolute path,
links followed, case folded on Windows), so two spellings of one folder are the same folder.
"""
from __future__ import annotations

import json
import os

# what each kind of output folder holds at its top level (sub-folders: only where listed)
RUN_FILES = frozenset({
    "recorder.jsonl", "feeder.json", "engine_summary.json", "latency.json", "harness.json", "p1_frames.jsonl",
    "p13_packets.jsonl", "p13_worker.json", "p13_worker_jobs.jsonl", "p13_richsets.bin", "p13_edges.json",
    "trajectory_live.csv", "live_score.json", "DIAGNOSTIC_RUN.txt", "NOT_FOR_BARS.txt"})
LIVE_FILES = RUN_FILES | {"demo.json", "demo_summary.json", "view_final.png"}
RECORDING_FILES = frozenset({"calib.json", "timestamps.csv", "imu.jsonl", "stream_meta.csv", "left", "right"})
IMU_CAPTURE_FILES = frozenset({"imu.jsonl", "meta.json", "STOP"})
DEV_RUN_FILES = frozenset({"trajectory_live.csv", "summary.json"})

KIND_NAMES = {"replay": "an earlier xclean replay run", "live": "an earlier xclean live run",
              "recording": "an earlier recording made by xclean live --record",
              "imu_capture": "an earlier IMU capture", "dev_run": "an earlier output of this tool"}
KIND_MARKERS = {"replay": "harness.json", "live": "demo.json", "recording": "stream_meta.csv",
                "imu_capture": "meta.json", "dev_run": "summary.json"}


def resolved(path):
    """The absolute, link-free, case-folded form of a path (for comparing two paths, not for opening one)."""
    return os.path.normcase(os.path.realpath(os.path.abspath(str(path))))


def _inside(child, parent):
    """True when resolved path `child` lies inside resolved path `parent` (not equal)."""
    return child != parent and child.startswith(parent.rstrip("\\/") + os.sep)


def protected_reason(folder):
    """Why `folder` must never be deleted (a phrase), or None."""
    f = resolved(folder)
    if os.path.dirname(f) == f:
        return "a drive root"
    cwd = resolved(os.getcwd())
    if f == cwd:
        return "the current folder"
    if _inside(cwd, f):
        return "a folder that holds the current folder"
    home = resolved(os.path.expanduser("~"))
    if f == home:
        return "your home folder"
    if _inside(home, f):
        return "a folder that holds your home folder"
    pkg = resolved(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if f == pkg or _inside(f, pkg):
        return "part of the xclean installation"
    if _inside(pkg, f):
        return "a folder that holds the xclean installation"
    return None


def output_kind_problem(kind, folder):
    """None when the existing, non-empty `folder` holds an earlier output of `kind` and nothing else; else the reason
    (a phrase) it is not one."""
    names = set(os.listdir(folder))
    if kind == "replay":
        allowed, ok = RUN_FILES, bool(names & {"harness.json", "recorder.jsonl"}) and "demo.json" not in names
    elif kind == "live":
        allowed = LIVE_FILES
        ok = "demo.json" in names or ("recorder.jsonl" in names and "harness.json" not in names)
    elif kind == "recording":
        allowed, ok = RECORDING_FILES, "stream_meta.csv" in names
    elif kind == "imu_capture":
        allowed, ok = IMU_CAPTURE_FILES, "meta.json" in names
    elif kind == "dev_run":
        allowed, ok = DEV_RUN_FILES, "summary.json" in names
    else:
        raise ValueError(kind)
    if kind == "replay" and "demo.json" in names:
        return "it is an xclean live run (it holds demo.json), not %s" % KIND_NAMES[kind]
    if not ok:
        return "it is not %s (no %s)" % (KIND_NAMES[kind], KIND_MARKERS[kind])
    other = sorted(n for n in names if n not in allowed)
    for sub in ("left", "right"):
        d = os.path.join(folder, sub)
        if kind == "recording" and os.path.isdir(d) and any(os.path.isdir(os.path.join(d, n)) for n in os.listdir(d)):
            other.append(sub + "/ (it holds sub-folders)")
    if other:
        return "it also holds files that this command does not write (%s)" % ", ".join(sorted(set(other))[:5])
    return None


def check_output_folder(cmd, opt, folder, overwrite, kind, inputs=(), others=()):
    """The output folder rule (module docstring).  cmd: the command name for the message; opt: the option that named
    the folder; overwrite: whether the user allowed replacing an earlier output; kind: "replay", "live",
    "recording" or "imu_capture"; inputs: paths the command reads; others: (option, folder) pairs of the command's
    other output folders, which must be separate from this one.  -> a plain error message, or None (the folder may
    be used: it does not exist, is empty, or may be replaced)."""
    head = "xclean %s: %s %s" % (cmd, opt, folder)
    f = resolved(folder)
    for o_opt, o_folder in others:
        if o_folder:
            g = resolved(o_folder)
            if f == g or _inside(f, g) or _inside(g, f):
                return ("xclean %s: %s and %s must be two separate folders, neither inside the other. Nothing was "
                        "deleted." % (cmd, opt, o_opt))
    for p in inputs:
        if not p:
            continue
        q = resolved(p)
        if q == f or _inside(q, f):
            return ("%s is or holds %s, which this command reads. Nothing was deleted. Choose another folder."
                    % (head, p))
        if _inside(f, q) and os.path.isdir(p):
            return ("%s lies inside %s, which this command reads. Nothing was deleted. Choose a folder outside it."
                    % (head, p))
    if not os.path.exists(folder):
        return None
    if not os.path.isdir(folder):
        return "%s is a file, not a folder. Nothing was deleted. Choose a folder name." % head
    why = protected_reason(folder)
    if why:
        return ("%s is %s. xclean never deletes such a folder. Nothing was deleted. Choose a new folder, for "
                "example runs%s<name>." % (head, why, os.sep))
    if not os.listdir(folder):
        return None
    if not overwrite:
        return ("%s already exists and is not empty. Nothing was deleted. Choose a new folder, or add --overwrite "
                "to replace %s in it." % (head, KIND_NAMES[kind]))
    why = output_kind_problem(kind, folder)
    if why:
        return ("%s: %s, so --overwrite does not replace it. Nothing was deleted. Choose a new folder."
                % (head, why))
    return None


# ------------------------------------------------------------------------------------------ input sequence folder
def detect_ext(seq):
    """'png' unless the first frame of timestamps.csv exists only as .jpg."""
    from xclean.io.rows import _detect_ext
    return _detect_ext(seq)


def _first_frame(ts_path):
    with open(ts_path, encoding="utf-8") as f:
        f.readline()
        for ln in f:
            p = ln.strip().split(",")
            if len(p) >= 2 and p[0].strip():
                return int(p[0])
    return None


def sequence_problem(seq, ext=None):
    """None when `seq` looks like a sequence folder whose first frame can be read; else one plain message (several
    sentences) saying what is wrong and what to do."""
    hint = ("Make a sequence folder with xclean convert (public datasets) or xclean live --record, and check it with "
            "xclean audit (docs/SEQUENCE_FORMAT.md).")
    if not os.path.exists(seq):
        return "%s does not exist. %s" % (seq, hint)
    if not os.path.isdir(seq):
        return "%s is a file, not a sequence folder. %s" % (seq, hint)
    missing = [n for n in ("calib.json", "timestamps.csv", "left", "right") if not os.path.exists(os.path.join(seq, n))]
    if missing:
        names = set(os.listdir(seq))
        if "mav0" in names:
            hint = ("It looks like a EuRoC folder as downloaded: convert it first with "
                    "xclean convert euroc <this folder> <new folder>.")
        elif names & {"image_0", "image_1", "sequences"}:
            hint = ("It looks like a KITTI folder as downloaded: convert it first with "
                    "xclean convert kitti <this folder> <new folder>.")
        return "%s is not a sequence folder: it has no %s. %s" % (seq, ", ".join(missing), hint)
    try:
        cal = json.load(open(os.path.join(seq, "calib.json"), encoding="utf-8"))
        w, h = int(cal["camera_width"]), int(cal["camera_height"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        return ("%s: calib.json cannot be read (%s: %s). Check it with xclean audit, or write it again with "
                "xclean calib or xclean convert." % (seq, type(e).__name__, e))
    try:
        idx = _first_frame(os.path.join(seq, "timestamps.csv"))
    except (OSError, ValueError, UnicodeDecodeError) as e:
        return "%s: timestamps.csv cannot be read (%s: %s). %s" % (seq, type(e).__name__, e, hint)
    if idx is None:
        return "%s: timestamps.csv has no frame rows. %s" % (seq, hint)
    import cv2
    from xclean.io.imageio import imread
    ext = ext or detect_ext(seq)
    for side in ("left", "right"):
        p = os.path.join(seq, side, "%06d.%s" % (idx, ext))
        img = imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ("%s: the first frame's image %s is missing or cannot be read. Check the folder with xclean audit."
                    % (seq, os.path.join(side, os.path.basename(p))))
        if img.shape != (h, w):
            return ("%s: the first frame's image %s is %d x %d, but calib.json says %d x %d."
                    % (seq, os.path.join(side, os.path.basename(p)), img.shape[1], img.shape[0], w, h))
    return None
