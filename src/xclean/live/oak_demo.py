#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""`xclean live`: the live engine fed by an OAK-D-Lite camera (xclean/live/oak_source.py) instead of the replay
FEEDER, with an optional recording of the raw stream and a small live view.

    python -m xclean.live.oak_demo --out <run folder> [--record <recording folder>] [--duration S]
        [--ref-calib <a calib.json of the same camera>  (or $XCLEAN_OAK_REF_CALIB)]
    # the same code path from a recording, at its real rate (no camera needed)
    python -m xclean.live.oak_demo --out <run folder> --replay <recording folder> [--mode paced|lockstep]
        [--frames N] [--view window|offscreen|none]

Run folder (as `xclean replay` writes it): recorder.jsonl (the RECORDER), feeder.json (the source log, the feeder's
schema + source extras), engine_summary.json + latency.json (the engine), harness.json (the fields the live scorer
reads; producer = this file), demo.json (pids, env, resources, source / recording facts), trajectory_live.csv (from
the recorder log: the pose published per frame, never revised, T_world_cam0_rectified,
t_frame,x,y,z,qx,qy,qz,qw,frame_index,state,t_publish_s,latency_ms as xclean/run.py), demo_summary.json, and with a
view view_final.png.

As the replay harness: the standard engine flag set, XONE_DIAL_DIST=1, PYTHONHASHSEED=<seed>, thread caps before
the children import numpy / OpenCV, spawn context, FrameSlot + ctl primitives + imu_q + msg_q, recorder_main and
engine_main from xclean.live, the harness Monitor for resources / foreign load, optional --pin (harness pin_layout:
engine + recorder on the engine cores, the source on the feeder cores).  XCLEAN_ENGINE defaults to p1.
Differences: (1) the source process starts FIRST so that a camera's calibration can be read before the engine starts
(the engine still gets it in its cfg); (2) with a view the engine's msg_q is oak_source.TapQueue (it forwards every
message unchanged to the recorder, then copies LIVE-ODOM poses to a shared ring; --view none = the plain queue).

--engine-root DIR (script mode only, a developer option): the engine package is imported from DIR/xclean; the OAK
modules come from this file's directory.  Child processes get the same split (the env var OAKDEMO_ENGINE_ROOT is
inherited and this file runs as their __mp_main__).
"""
from __future__ import annotations

import os
import sys


def _engine_root_from_argv(argv):
    for i, x in enumerate(argv):
        if x == "--engine-root" and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith("--engine-root="):
            return x.split("=", 1)[1]
    return None


if not __package__:                       # script mode (python .../oak_demo.py), also its spawned children
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _REPO = os.path.dirname(os.path.dirname(_HERE))
    if __name__ == "__main__":
        _er = _engine_root_from_argv(sys.argv)
        if _er:
            os.environ["OAKDEMO_ENGINE_ROOT"] = os.path.abspath(_er)
        else:
            os.environ.pop("OAKDEMO_ENGINE_ROOT", None)
    _ROOT = os.environ.get("OAKDEMO_ENGINE_ROOT") or _REPO
    _nc = os.path.normcase
    sys.path[:] = [p for p in sys.path if _nc(os.path.abspath(p or os.curdir)) != _nc(_HERE)]
    for _p in (_REPO, _ROOT):
        while _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)
    import xclean.live as _xl             # noqa: E402
    if _nc(_HERE) not in [_nc(os.path.abspath(p)) for p in _xl.__path__]:
        _xl.__path__.append(_HERE)        # oak_source / oak_demo from the repo, everything else from the engine root

import argparse                           # noqa: E402
import json                               # noqa: E402
import shutil                             # noqa: E402
import time                               # noqa: E402

import numpy as np                        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# a calib.json of the SAME OAK unit (imu_noise / camera_fps / imu_rate_hz, which the camera's memory does not hold).
# None = the camera's own values only.
DEFAULT_REF_CALIB = os.environ.get("XCLEAN_OAK_REF_CALIB") or None
STATE_COLORS = {0: (160, 160, 160), 1: (80, 200, 80), 2: (40, 160, 255), 3: (60, 60, 230)}   # BGR


# ================================================================================================ run-dir products
def load_run(d):
    rec = [json.loads(x) for x in open(os.path.join(d, "recorder.jsonl"), encoding="utf-8") if x.strip()]
    feeder = json.load(open(os.path.join(d, "feeder.json"), encoding="utf-8"))
    return rec, feeder


def write_trajectory_live(d, rec=None, feeder=None):
    """trajectory_live.csv from the RECORDER log (LIVE-ODOM, receipt order = publish order) + the source/feeder release
    log: t_publish_s = receipt - the first release, latency_ms = receipt - this frame's release."""
    from scipy.spatial.transform import Rotation as Rot
    if rec is None:
        rec, feeder = load_run(d)
    rel = {int(f[0]): int(f[3]) for f in feeder["frames"]}
    r0 = min(rel.values()) if rel else 0
    n = 0
    with open(os.path.join(d, "trajectory_live.csv"), "w", encoding="utf-8") as f:
        f.write("# frame convention: T_world_cam0_rectified (live stream: pose published at publish time, never "
                "revised; %s; from recorder.jsonl LIVE-ODOM)\n" % feeder.get("mode"))
        f.write("# t_frame,x,y,z,qx,qy,qz,qw,frame_index,state,t_publish_s,latency_ms\n")
        for m in rec:
            if m.get("stream") != "LIVE-ODOM":
                continue
            T = np.array(m["T"], float).reshape(4, 4)
            if not np.isfinite(T).all():
                continue
            q = Rot.from_matrix(T[:3, :3]).as_quat()
            i = int(m["frame_index"])
            lat = (m["rx_ns"] - rel[i]) / 1e6 if i in rel else float("nan")
            f.write("%.9f,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f,%d,%s,%.6f,%.3f\n"
                    % (m["t_pose"], T[0, 3], T[1, 3], T[2, 3], q[0], q[1], q[2], q[3], i, m["state"],
                       (m["rx_ns"] - r0) / 1e9, lat))
            n += 1
    return n


def _pct(v, qs=(50, 95, 99)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    out = {"p%d" % q: float(np.percentile(v, q)) for q in qs}
    out.update(max=float(v.max()), n=int(len(v)))
    return out


def lost_spans(idx, states, t, ok=("tracking",)):
    """Runs of consecutive PUBLISHED frames whose state is not in `ok` -> list of dicts, longest first."""
    spans, cur = [], None
    for i, s, tt in zip(idx, states, t):
        if s not in ok:
            if cur is None:
                cur = {"first": int(i), "last": int(i), "n": 0, "t0": float(tt), "t1": float(tt), "states": set()}
            cur["last"], cur["t1"] = int(i), float(tt)
            cur["n"] += 1
            cur["states"].add(s)
        elif cur is not None:
            spans.append(cur)
            cur = None
    if cur is not None:
        spans.append(cur)
    for s in spans:
        s["states"] = sorted(s["states"])
        s["dur_s"] = s["t1"] - s["t0"]
    return sorted(spans, key=lambda s: -s["n"])


def summarize_run(d, write=True):
    """Tracking, latency, drops, IMU admission and real-time facts of a harness or demo run dir (reads the recorder
    log, the release log and the engine's own summary / ledger; never GT)."""
    rec, feeder = load_run(d)
    esum = json.load(open(os.path.join(d, "engine_summary.json"))) \
        if os.path.isfile(os.path.join(d, "engine_summary.json")) else {}
    lat_j = json.load(open(os.path.join(d, "latency.json"))) if os.path.isfile(os.path.join(d, "latency.json")) else {}
    fr = feeder["frames"]
    idx = np.array([f[0] for f in fr], np.int64)
    t = np.array([f[1] for f in fr], float)
    rel = np.array([f[3] for f in fr], float)
    period = float(np.median(np.diff(t))) if len(t) > 1 else None
    lm = {m["frame_index"]: m for m in rec if m.get("stream") == "LIVE-MAP"}
    odo = [m for m in rec if m.get("stream") == "LIVE-ODOM"]
    lat = np.array([(lm[i]["rx_ns"] - r) / 1e6 if i in lm else np.nan for i, r in zip(idx, rel)])
    rx = np.sort(np.array([m["rx_ns"] for m in lm.values()], float))
    gap = float(np.max(np.diff(rx)) / 1e6) if len(rx) > 1 else None
    states = [m["state"] for m in odo]
    cnt = {}
    for s in states:
        cnt[s] = cnt.get(s, 0) + 1
    spans = lost_spans([m["frame_index"] for m in odo], states, [m["t_pose"] for m in odo])
    wall = (rx[-1] - rel[0]) / 1e9 if len(rx) else None
    dur = float(t[-1] - t[0]) if len(t) > 1 else None
    out = {"dir": d, "mode": feeder.get("mode"), "source": feeder.get("source", "feeder"),
           "frames_released": int(len(fr)), "frames_published": int(len(lm)),
           "frames_dropped": int(len(fr) - sum(1 for i in idx if int(i) in lm)),
           "frames_replaced": len(feeder.get("replaced", [])), "camera_period_ms": None if period is None else
           1e3 * period, "stream_duration_s": dur, "wall_first_release_to_last_publish_s": wall,
           "rtf": (dur / wall) if (dur and wall) else None,
           "latency_ms": _pct(lat), "latency_periods": (_pct(lat / (1e3 * period)) if period else None),
           "max_publish_gap_ms": gap, "max_publish_gap_periods": (gap / (1e3 * period)) if (gap and period) else None,
           "states": cnt, "tracked_pct": 100.0 * cnt.get("tracking", 0) / max(len(states), 1),
           "lost_spans_n": len(spans), "lost_frames": int(sum(s["n"] for s in spans)),
           "lost_spans_top": spans[:12], "feeder_validity": feeder.get("validity"),
           "imu_released": feeder.get("imu_released")}
    src = feeder.get("source_frames")
    if src:
        cols = feeder.get("source_frames_cols", [])
        ci = {c: k for k, c in enumerate(cols)}
        cap = {int(r[0]): r[ci["capture_to_release_ms"]] for r in src}
        out["capture_to_release_ms"] = _pct(list(cap.values()))
        tot = [cap[int(i)] + l for i, l in zip(idx, lat) if cap.get(int(i)) is not None and np.isfinite(l)]
        out["capture_to_publish_ms"] = _pct(tot)
        out["imu_lag_at_release_ms"] = _pct([r[ci["imu_lag_ms"]] for r in src])
        out["lr_skew_us"] = _pct([abs(r[ci["dt_lr_us"]]) for r in src])
        out["pairing"] = feeder.get("pairing")
    p1 = esum.get("p1") or {}
    out["engine"] = {k: esum.get(k) for k in ("engine", "frames_processed", "n_fence_errors", "engine_md5",
                                             "imu_wait_bound_s", "imu_wait_ms", "imu_short_frames",
                                             "imu_short_first", "gpu_front")}
    out["imu"] = {k: p1.get(k) for k in ("vi", "vi_path", "vi_admit_frame", "vi_admit_kf", "vi_admit_segment",
                                        "vi_admit_kappa", "vi_refused", "vi_refuse_reasons", "vi_defer",
                                        "vi_defer_reasons", "d1_tests", "d1_admit_frame", "demoted",
                                        "demote_frame", "demote_reason", "kappa_imu", "imu_coast", "vi_resets",
                                        "vi_coast_max", "g_w", "bg", "ba", "wd_active", "events")}
    if isinstance(out["imu"].get("events"), list):
        out["imu"]["events"] = out["imu"]["events"][:40]
    out["p1"] = {k: p1.get(k) for k in ("frames", "tracked", "coasted", "init_frames", "segments", "kfs",
                                       "pnp_fail", "reanchor", "lat", "stereo_impl", "segment")}
    st = lat_j.get("stages") or {}
    out["stages_ms"] = {k: {"p50": v.get("p50"), "p95": v.get("p95"), "n": v.get("n")} for k, v in st.items()}
    if write:
        json.dump(out, open(os.path.join(d, "demo_summary.json"), "w"), indent=1, default=str)
    return out


# ================================================================================================ view
class DemoView:
    """Top-down trajectory (world = the first rectified cam0 frame: x right, z forward; drawn as x-z), the current
    pose and heading, the tracking state, the latest left image and per-frame latency."""

    def __init__(self, slot, ring, rel, fps, title):
        self.slot, self.ring, self.rel = slot, ring, rel
        self.period_ms = 1e3 / fps
        self.title = title
        self.c = 0
        self.P = []
        self.S = []
        self.last = None
        self.lat = []
        self.cap_pub = []
        self.n = 0
        self.t_start = time.time()

    def update(self):
        self.c, rows = self.ring.read_since(self.c)
        for r in rows:
            i = int(r[0])
            self.P.append((r[6], r[8]))
            self.S.append(int(r[3]))
            self.last = r
            self.n += 1
            e = self.rel.a[i % self.rel.N]
            if int(e[0]) == i:
                L = (r[2] - e[1]) / 1e6
                self.lat.append(L)
                if np.isfinite(e[2]):
                    self.cap_pub.append(L + e[2])
        self.lat = self.lat[-300:]
        self.cap_pub = self.cap_pub[-300:]

    def render(self):
        import cv2
        H, Wt, Wr = 640, 640, 460
        img = np.full((H, Wt + Wr, 3), 24, np.uint8)
        P = np.asarray(self.P, float).reshape(-1, 2)
        if len(P):
            lo, hi = P.min(0), P.max(0)
            span = max(float((hi - lo).max()), 1.0)
            s = (Wt - 60) / span
            c = (lo + hi) / 2

            def px(p):
                return int(Wt / 2 + (p[0] - c[0]) * s), int(H / 2 - (p[1] - c[1]) * s)
            pts = [px(p) for p in P]
            for k in range(1, len(pts)):
                cv2.line(img, pts[k - 1], pts[k], STATE_COLORS.get(self.S[k], (60, 60, 230)), 2)
            cv2.circle(img, pts[0], 6, (255, 255, 255), 1)
            if self.last is not None:
                R = self.last[9:18].reshape(3, 3)
                fwd = R[:, 2]
                h = np.array([fwd[0], fwd[2]])
                h = h / max(np.linalg.norm(h), 1e-9)
                p = pts[-1]
                cv2.circle(img, p, 6, STATE_COLORS.get(int(self.last[3]), (60, 60, 230)), -1)
                cv2.arrowedLine(img, p, (int(p[0] + 30 * h[0]), int(p[1] - 30 * h[1])), (255, 255, 255), 2,
                                tipLength=0.35)
            nice = [x for x in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if x * s <= 160]
            bar = nice[-1] if nice else 0.5
            cv2.line(img, (20, H - 20), (20 + int(bar * s), H - 20), (220, 220, 220), 2)
            cv2.putText(img, "%g m" % bar, (20, H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(img, "top-down x-z (world = first cam0)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)
        cam = np.array(self.slot.buf[0])                     # display only (an unlocked read may tear)
        ch = int(Wr * cam.shape[0] / cam.shape[1])
        img[0:ch, Wt:Wt + Wr] = cv2.cvtColor(cv2.resize(cam, (Wr, ch)), cv2.COLOR_GRAY2BGR)
        names = {0: "init", 1: "TRACKING", 2: "COASTING", 3: "other"}
        lines = [self.title]
        if self.last is not None:
            r = self.last
            lines += ["frame %d   t %.2f s" % (int(r[0]), r[1]),
                      "state %s   inliers %s   segment %d" % (names.get(int(r[3]), "?"),
                                                                "-" if r[4] < 0 else "%d" % r[4], int(r[5])),
                      "pos x %.2f  y %.2f  z %.2f m" % (r[6], r[7], r[8])]
        if self.lat:
            L = np.asarray(self.lat)
            lines += ["latency release->pose: now %.1f  p50 %.1f  p95 %.1f ms" % (L[-1], np.median(L),
                                                                                  np.percentile(L, 95)),
                      "   = %.2f / %.2f camera periods (p50 / p95)" % (np.median(L) / self.period_ms,
                                                                       np.percentile(L, 95) / self.period_ms)]
        if self.cap_pub:
            C = np.asarray(self.cap_pub)
            lines.append("capture->pose p50 %.1f  p95 %.1f ms" % (np.median(C), np.percentile(C, 95)))
        el = max(time.time() - self.t_start, 1e-3)
        lines.append("published %d  (%.1f /s)   dropped (replaced) %d" % (self.n, self.n / el, int(self.slot.hdr[7])))
        y = ch + 26
        for ln in lines:
            cv2.putText(img, ln, (Wt + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
            y += 22
        return img


def imu_outcome(imu, has_imu=True):
    """What the engine did with the IMU, from the engine summary's IMU facts (summarize_run's "imu")."""
    imu = imu or {}
    if imu.get("vi"):
        s = "IMU used from frame %s" % imu.get("vi_admit_frame")
        if imu.get("demoted"):
            s += " until frame %s" % imu.get("demote_frame")
        return s
    if not has_imu:
        return "vision only"
    if imu.get("vi_refused"):
        return "IMU not used (its start-up check did not accept the data)"
    return "IMU not used"


def summary_line(d, summ=None, score=None):
    """The one plain line at the end of a replay or live run: frames, dropped frames, latency, ATE when scored, what
    the engine did with the IMU, the run folder and the score file."""
    s = summ if summ is not None else summarize_run(d, write=False)
    parts = ["%d frames" % s["frames_released"], "%d dropped" % s["frames_dropped"]]
    lat = s.get("latency_ms") or {}
    if lat:
        if s.get("mode") == "lockstep":
            parts.append("time per frame p50 %.1f ms, p95 %.1f ms (lock-step, not a live latency)"
                         % (lat["p50"], lat["p95"]))
        else:
            parts.append("latency p50 %.1f ms, p95 %.1f ms" % (lat["p50"], lat["p95"]))
    b5 = (score or {}).get("B5") or {}
    if b5.get("ate") is not None:
        parts.append("ATE %.4f m over %d frames" % (b5["ate"], b5.get("frames_scored") or 0))
    elif b5.get("no_gt"):
        parts.append("no ground truth, so no ATE")
    has_imu = s.get("imu_released") is None or bool(s.get("imu_released"))
    parts.append(imu_outcome(s.get("imu"), has_imu))
    line = "xclean: summary: %s. Run folder: %s" % (", ".join(parts), d)
    sc = os.path.join(d, "live_score.json")
    if os.path.isfile(sc):
        line += ". Scores: %s" % sc
    return line


def imu_note(calib, replay=None):
    """One plain line when the run cannot use the IMU because the calibration has no IMU noise values."""
    if calib.get("imu_noise"):
        return None
    if replay:
        return ("xclean: note: the recording's calib.json has no imu_noise block, so this run does not use the IMU "
                "(vision only). See docs/OWN_DATA.md, \"IMU noise values\".")
    return ("xclean: note: the camera's calibration has no IMU noise values, so this run does not use the IMU (vision "
            "only). To use it, pass --ref-calib with a calib.json that holds your unit's imu_noise, camera_fps and "
            "imu_rate_hz (docs/OWN_DATA.md, \"OAK-D-Lite\").")


# ================================================================================================ main
DESCRIPTION = """Run xclean live on an OAK-D-Lite camera, or on a recording made with --record (--replay).

  xclean live --out runs\\live1 --record recordings\\live1 --duration 60
  xclean live --out runs\\live1_again --replay recordings\\live1 --mode lockstep

The view window shows the trajectory seen from above, the camera image and the latency; press q or Esc in it (or
Ctrl+C in the console) to stop.  The run folder gets the published poses (trajectory_live.csv), a summary
(demo_summary.json), the run logs and, with a view, view_final.png.  The camera is only read: xclean never writes
to its memory or firmware."""


def build_parser():
    """The command line of `xclean live` (= python -m xclean.live.oak_demo)."""
    hide = argparse.SUPPRESS
    ap = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--out", required=True, help="the run folder to write; it must be new or empty")
    ap.add_argument("--replay", default=None,
                    help="run from this recording (made with --record) instead of the camera, at its real rate")
    ap.add_argument("--mode", default="paced", choices=("paced", "lockstep"),
                    help="paced (default): frames arrive at the camera's rate; lockstep (with --replay only): each "
                         "frame waits for the one before, so no frame is dropped and the result repeats exactly")
    ap.add_argument("--frames", type=int, default=None, help="with --replay: use only the first N frames")
    ap.add_argument("--duration", type=float, default=None,
                    help="camera: stop after this many seconds (default: run until stopped)")
    ap.add_argument("--record", default=None,
                    help="also record the camera stream to this folder, so it can be replayed later. The folder "
                         "must be new or empty; with --overwrite, an earlier recording made by xclean live is "
                         "replaced")
    ap.add_argument("--calib", default=None,
                    help="camera: use this calib.json instead of the calibration stored in the camera")
    ap.add_argument("--ref-calib", default=DEFAULT_REF_CALIB,
                    help="camera: a calib.json of the same camera that holds imu_noise, camera_fps and imu_rate_hz, "
                         "so that the IMU is used (docs/OWN_DATA.md)")
    ap.add_argument("--mxid", default=None, help="camera: the device ID of the camera to use when several are attached")
    ap.add_argument("--fps", type=float, default=30.0, help="camera: frame rate to request (default 30)")
    ap.add_argument("--view", default="window", choices=("window", "offscreen", "none"),
                    help="window (default): a live view window; offscreen: no window, but view_final.png is written; "
                         "none: no view")
    ap.add_argument("--pin", action="store_true",
                    help="pin the engine to fixed CPU cores, for timing runs (needs 6 or more cores; on a CPU with "
                         "efficiency cores, run without it)")
    ap.add_argument("--json", action="store_true",
                    help="also print the raw run record and run summary as JSON (both are always written to the run "
                         "folder: demo.json, demo_summary.json)")
    # options of our benchmark protocol and for developers: they work, but --help does not list them
    ap.add_argument("--tag", default=None, help=hide)
    ap.add_argument("--engine-root", default=None, help=hide)
    ap.add_argument("--threads", type=int, default=2, help=hide)
    ap.add_argument("--seed", type=int, default=1, help=hide)
    ap.add_argument("--env", default="", help=hide)
    ap.add_argument("--no-x1-flags", action="store_true", help=hide)
    return ap


def main():
    import multiprocessing as mp
    import psutil
    from xclean.io import folders as FO
    from xclean.live import harness as H
    from xclean.live.protocol import FrameSlot
    from xclean.live.oak_source import (source_main, engine_entry, recorder_entry, ReleaseRing, PoseRing,
                                        TapQueue)
    from xclean.console import tolerant_output
    a = build_parser().parse_args()
    tolerant_output(for_children=True)        # a path the console cannot encode is printed with escapes, never an error
    if a.engine_root and __package__:
        print("xclean live: error: --engine-root needs script mode: python %s ..." % os.path.abspath(__file__),
              file=sys.stderr)
        return 2
    if a.mode == "lockstep" and not a.replay:
        print("xclean live: error: --mode lockstep needs --replay (a camera cannot wait for the engine)",
              file=sys.stderr)
        return 2
    if a.pin:
        try:
            H.pin_layout()
        except ValueError as e:
            print("xclean: %s" % e, file=sys.stderr)
            return 2
    if a.replay:
        problem = FO.sequence_problem(a.replay)
        if problem:
            print("xclean: error: %s" % problem, file=sys.stderr)
            return 2
    inputs = [a.replay, a.calib, a.ref_calib]
    for opt, folder, kind, other in (("--out", a.out, "live", ("--record", a.record)),
                                     ("--record", a.record, "recording", ("--out", a.out))):
        problem = folder and FO.check_output_folder("live", opt, folder, True, kind, inputs=inputs, others=[other])
        if problem:
            print(problem, file=sys.stderr)
            return 2
    out = os.path.abspath(a.out)
    if os.path.isdir(out):
        shutil.rmtree(out)                                   # an earlier run of this command (checked above)
    os.makedirs(out)
    if a.record:
        rd = os.path.abspath(a.record)
        if os.path.isdir(rd):
            shutil.rmtree(rd)                                # an earlier recording (checked above)
    env = {} if a.no_x1_flags else dict(H.X1_ENV)
    env.update({"XONE_DIAL_DIST": "1", "PYTHONHASHSEED": str(a.seed), "XCLEAN_ENGINE": "p1"})
    for kv in [x for x in a.env.split(",") if x]:
        k, v = kv.split("=", 1)
        env[k] = v
    DIAG_FLAGS = {("XCLEAN_P1_ADMIT", "force_diag")}       # as the harness: oracle diagnostics are a diagnostic run
    nfb = ["%s=%s" % (k, v) for k, v in env.items() if (k, v) in DIAG_FLAGS
           or (k.startswith("XCLEAN_P1") and k.endswith("_OFF") and v.strip())]
    n = str(max(1, a.threads))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = n
    os.environ["XCLEAN_CV_THREADS"] = n
    os.environ.update(env)
    vram_base = H.vram_used_mb()
    others_gpu = H.gpu_compute_apps()
    ctx = mp.get_context("spawn")
    ctl = {"ready": ctx.Event(), "new_frame": ctx.Event(), "done": ctx.Value("q", -1), "done_evt": ctx.Event(),
           "stop": ctx.Event(), "lock": ctx.Lock(), "slot_ready": ctx.Event()}
    imu_q, msg_q, calib_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    pid = os.getpid()
    slot_name, rel_name, pose_name = "xclean_slot_%d" % pid, "xclean_oakrel_%d" % pid, "xclean_oakpose_%d" % pid
    ref = json.load(open(a.ref_calib, encoding="utf-8")) if (not a.replay and a.ref_calib
                                                             and os.path.isfile(a.ref_calib)) else None
    given = json.load(open(a.calib, encoding="utf-8")) if a.calib else None
    scfg = {"kind": "replay" if a.replay else "device", "mode": a.mode, "replay_dir": a.replay, "frames": a.frames,
            "fps": a.fps, "width": (given or ref or {}).get("camera_width", 640),
            "height": (given or ref or {}).get("camera_height", 400), "mxid": a.mxid, "calib": given,
            "ref_calib": ref, "record_dir": os.path.abspath(a.record) if a.record else None, "rel_ring": rel_name,
            "duration_s": a.duration}
    rel = ReleaseRing(rel_name, create=True)
    t_start = time.time()
    src = ctx.Process(target=source_main, args=(scfg, slot_name, imu_q, ctl, os.path.join(out, "feeder.json"),
                                                calib_q), name="oak-source")
    src.start()
    got = None
    while got is None:
        try:
            got = calib_q.get(timeout=1.0)
        except Exception:
            if not src.is_alive():
                got = {"error": "source exited (code %s) before sending a calib" % src.exitcode}
    if "error" in got:
        src.join(10)
        rel.close(unlink=True)
        print("xclean live: error: the %s could not be started: %s" % ("recording" if a.replay else "camera",
                                                                      got["error"]), file=sys.stderr, flush=True)
        json.dump({"error": got["error"], "args": vars(a)}, open(os.path.join(out, "demo.json"), "w"), indent=1)
        return 2
    calib = got["calib"]
    has_imu_data = os.path.isfile(os.path.join(a.replay, "imu.jsonl")) if a.replay else True
    first = imu_note(calib, a.replay) if has_imu_data else None
    if first is None:                     # IMU in use, or vision only for another reason than missing noise values
        print(H.imu_status_line(calib, has_imu_data, env=dict(os.environ, **env), recording=bool(a.replay)),
              flush=True)
    for note in (first, H.imu_config_note(a.replay, calib) if a.replay else None):
        if note:
            print(note, file=sys.stderr, flush=True)
    W, Hh = int(calib["camera_width"]), int(calib["camera_height"])
    slot = FrameSlot(slot_name, W, Hh, create=True)
    pose = PoseRing(pose_name, create=True) if a.view != "none" else None
    ctl["slot_ready"].set()
    tag = a.tag or ("oak_replay" if a.replay else "oak_live")
    has_imu = os.path.isfile(os.path.join(a.replay, "imu.jsonl")) if a.replay else True
    ecfg = {"tag": tag, "calib": calib, "env": env, "final": False, "seed": a.seed, "has_imu": has_imu,
            "mode": a.mode}
    eng_q = TapQueue(msg_q, pose_name) if pose is not None else msg_q
    # P1.3: the loop-closure WORKER + its transport, exactly as the harness starts it (unless the loop
    # is off or no worker is available: then LIVE-MAP = LIVE-ODOM, recorded)
    from xclean.live import p13_link as P13L
    p13_proc = p13_owners = None
    if (os.environ.get("XCLEAN_ENGINE", "") or "").strip().lower() == "p1" and P13L.loop_on():
        okw, why = P13L.worker_available()
        if okw:
            p13h, p13_owners = P13L.make_transport(ctx, W, Hh, tag, mode=a.mode)
            ctl["p13"] = p13h
            p13_proc = P13L.start_worker(ctx, p13h, {"tag": tag, "calib": calib, "mode": a.mode, "out_dir": out,
                                                     "seed": a.seed, "env": env, "worker": P13L.WORKER})
        else:
            ecfg["p13_inactive_why"] = why
            print("xclean: note: loop closure is off for this run (its process could not be started: %s)" % why)
    rec = ctx.Process(target=recorder_entry, args=(msg_q, os.path.join(out, "recorder.jsonl")), name="recorder")
    eng = ctx.Process(target=engine_entry, args=(ecfg, slot_name, imu_q, eng_q, ctl, out), name="engine")
    rec.start()
    eng.start()
    pids = {"demo": pid, "source": src.pid, "recorder": rec.pid, "engine": eng.pid}
    if p13_proc is not None:
        pids["p13worker"] = p13_proc.pid
    affinity = None
    if a.pin:
        try:
            lay = H.pin_layout()
            eng_cpus, fed_cpus = lay["engine_cpus"], lay["feeder_cpus"]
            for p_, c_ in ((eng.pid, eng_cpus), (rec.pid, eng_cpus), (src.pid, fed_cpus)):
                psutil.Process(p_).cpu_affinity(c_)
            affinity = {"engine_and_recorder": eng_cpus, "source": fed_cpus}
            if p13_proc is not None:          # P1.3: the worker on its own physical core
                wk = lay["worker_cpus"]
                psutil.Process(p13_proc.pid).cpu_affinity(wk)
                affinity["p13worker"] = wk
        except Exception as e:
            affinity = {"error": str(e)}
    mon = H.Monitor(eng.pid, pids.values(), vram_base,
                    extra_pids=[p13_proc.pid, rec.pid] if p13_proc is not None else None)
    mon.start()
    view, win = None, "xclean OAK live demo"
    if pose is not None:
        view = DemoView(slot, pose, rel, float(calib.get("camera_fps") or a.fps),
                        "%s  %s" % ("REPLAY " + os.path.basename(os.path.normpath(a.replay)) if a.replay
                                    else "LIVE OAK-D-LITE %s" % (got.get("mxid") or ""), a.mode))
    canvas = None
    eos_forced = False
    try:
        while eng.is_alive():
            if view is not None:
                view.update()
                canvas = view.render()
                if a.view == "window":
                    import cv2
                    cv2.imshow(win, canvas)
                    k = cv2.waitKey(50) & 0xFF
                    if k in (ord("q"), 27):
                        ctl["stop"].set()
                else:
                    time.sleep(0.25)
            else:
                eng.join(0.5)
            if not src.is_alive() and not eos_forced:
                with ctl["lock"]:
                    if not slot.hdr[4]:                       # the source died without EOS: end the stream
                        slot.hdr[4] = 1
                        eos_forced = True
                ctl["new_frame"].set()
    except KeyboardInterrupt:
        ctl["stop"].set()
        eng.join()
    eng.join()
    ctl["stop"].set()                 # the engine has ended (EOS, or a crash): a device source must stop streaming
    failed = eng.exitcode != 0
    # as the harness: an engine that died never sent the recorder its "EOS"; one that died during start-up never set
    # "ready", so the source has nothing to wait for either: short joins, then our own children are stopped
    early_exit = failed and not ctl["ready"].is_set()
    if failed:
        msg_q.put("EOS")
        if p13_proc is not None:
            try:
                ctl["p13"]["ctl_q"].put_nowait({"type": "eos", "n_kf": -1})
            except Exception:
                pass
    src.join((5 if a.replay else 20) if early_exit else 60)    # a camera needs a moment to close
    rec.join(5 if early_exit else 60)
    children_stopped = []
    if failed:                        # failure path only; our own children only (never selected by name)
        for p_ in (src, rec):
            if p_.is_alive():
                p_.terminate()
                p_.join(10)
                children_stopped.append(p_.name)
    p13_info = None
    if p13_proc is not None:          # P1.3: the worker exits on the engine's EOS message; our own child only
        p13_proc.join(30 if early_exit else 300)
        killed = p13_proc.is_alive()
        if killed:
            p13_proc.terminate()
            p13_proc.join(10)
        P13L.close_transport(p13_owners)
        p13_info = {"exitcode": p13_proc.exitcode, "terminated_after_timeout": killed, "worker": P13L.WORKER}
    mon.stop.set()
    mon.join(5)
    if view is not None:
        view.update()
        canvas = view.render()
        import cv2
        from xclean.io.imageio import imwrite
        try:
            if not imwrite(os.path.join(out, "view_final.png"), canvas):
                raise OSError("the image could not be encoded")
        except OSError as e:
            print("xclean: warning: view_final.png was not written (%s)" % e, file=sys.stderr, flush=True)
        if a.view == "window":
            cv2.destroyAllWindows()
    slot.close(unlink=True)
    rel.close(unlink=True)
    if pose is not None:
        pose.close(unlink=True)
    wall = time.time() - t_start
    feeder = json.load(open(os.path.join(out, "feeder.json"))) if os.path.isfile(os.path.join(out, "feeder.json")) \
        else {}
    esum = json.load(open(os.path.join(out, "engine_summary.json"))) \
        if os.path.isfile(os.path.join(out, "engine_summary.json")) else {}
    cpuw = mon.cpu_windows
    res = {"peak_private_gb": mon.peak_private / 1e9,
           "vram_delta_mb": (mon.peak_vram - vram_base) if vram_base is not None else None,
           "vram_base_mb": vram_base, "cpu_cores_max_10s": max(cpuw) if cpuw else None,
           "cpu_cores_mean": (sum(cpuw) / len(cpuw)) if cpuw else None, "cpu_windows": len(cpuw)}
    checks = {"exit_codes": {"engine": eng.exitcode, "source": src.exitcode, "recorder": rec.exitcode,
                             "p13worker": None if p13_info is None else p13_info["exitcode"]},
              "processes_ok": eng.exitcode == 0 and src.exitcode == 0 and rec.exitcode == 0
              and (p13_info is None or (p13_info["exitcode"] == 0 and not p13_info["terminated_after_timeout"])),
              "p13": {"worker": p13_info, "inactive_why": ecfg.get("p13_inactive_why"), "loop_on": P13L.loop_on()},
              "eos_forced_by_demo": eos_forced, "source_validity": feeder.get("validity"), "affinity": affinity,
              "other_gpu_compute_apps_at_start": others_gpu,
              "other_gpu_compute_apps_not_allowed": [x for x in others_gpu
                                                     if not H.gpu_app_allowed(x[1])],
              "others_over_5pct_core": mon.others_over, "allow_list_cpu_pct_max": mon.allow_cpu,
              "allow_list_total_cpu_pct_max": mon.allow_sum_max, "fence_errors": esum.get("n_fence_errors")}
    if nfb:
        open(os.path.join(out, "DIAGNOSTIC_RUN.txt"), "w").write(
            "DIAGNOSTIC RUN - not comparable with the published runs: %s\n" % ", ".join(nfb))
    label = ("oak demo %s (%s)" % ("replay" if a.replay else "LIVE DEVICE", a.mode)
             + (" - diagnostic run (engine switches: %s)" % ", ".join(nfb) if nfb else ""))
    # the fields score_live reads; validity is NOT judged by the demo (no witness protocol): valid = None
    json.dump({"tag": tag, "mode": a.mode, "label": label, "valid": None, "producer": "xclean/live/oak_demo.py",
               "engine_md5": esum.get("engine_md5"), "not_for_bars": bool(nfb)},
              open(os.path.join(out, "harness.json"), "w"), indent=1)
    demo = {"tag": tag, "sequence_folder": os.path.abspath(a.replay) if a.replay else None, "mode": a.mode,
            "label": label, "args": vars(a), "pids": pids,
            "engine_root": os.environ.get("OAKDEMO_ENGINE_ROOT") or REPO,
            "engine_md5": esum.get("engine_md5"), "env_flags": esum.get("env_flags"), "seed": a.seed,
            "threads_per_process": int(n), "wall_s": wall, "resources": res, "checks": checks,
            "calib_source": calib.get("source"), "device_mxid": got.get("mxid"),
            "frames_released": len(feeder.get("frames", [])), "frames_replaced": len(feeder.get("replaced", [])),
            "recording": feeder.get("recording")}
    if failed:                        # failure path only (a successful run's demo.json is unchanged)
        demo["engine_failed"] = {"exitcode": eng.exitcode, "during_startup": early_exit,
                                 "children_stopped": children_stopped}
    json.dump(demo, open(os.path.join(out, "demo.json"), "w"), indent=1, default=str)
    if failed:
        sys.stdout.flush()
        print("xclean: the engine stopped %s (exit code %s); its error is printed above. No trajectory was written. "
              "The run folder keeps what was written: %s" % ("during start-up" if early_exit
                                                            else "before the end of the stream", eng.exitcode, out),
              file=sys.stderr, flush=True)
        return 1
    summ = n_traj = None
    try:
        n_traj = write_trajectory_live(out)
    except Exception as e:
        print("xclean: error: trajectory_live.csv was not written (%s: %s)" % (type(e).__name__, e), file=sys.stderr,
              flush=True)
    if n_traj is not None:                    # printed outside the try: a print can never hide a written file
        print("xclean: wrote %d poses to %s" % (n_traj, os.path.join(out, "trajectory_live.csv")), flush=True)
    try:
        summ = summarize_run(out)
    except Exception as e:
        print("xclean: warning: the run summary was not written (%s: %s)" % (type(e).__name__, e), flush=True)
    console = {k: demo[k] for k in ("tag", "mode", "label", "wall_s", "frames_released", "frames_replaced")}
    console["label"] = ("%s (%s)" % ("replay of a recording" if a.replay else "live camera", a.mode)
                        + (" - diagnostic run (engine switches: %s)" % ", ".join(nfb) if nfb else ""))
    if a.json:                                # the raw records; demo.json and demo_summary.json always have them
        if summ is not None:
            print(json.dumps({k: summ.get(k) for k in ("frames_released", "frames_published", "frames_dropped",
                                                        "tracked_pct", "rtf", "latency_ms", "lost_spans_n")},
                             default=str))
        print(json.dumps(console, default=str))
        print("resources", json.dumps(res, default=str))
    elif nfb:
        print("xclean: note: this is a diagnostic run (engine switches: %s); it is not comparable with the published "
              "runs." % ", ".join(nfb), flush=True)
    rc = 0 if checks["processes_ok"] else 1
    if src.exitcode not in (0, None):
        print("xclean: the %s stopped with an error (exit code %s); its error is printed above. The stream was ended "
              "early." % ("recording reader" if a.replay else "camera source", src.exitcode),
              file=sys.stderr, flush=True)
    rec_info = feeder.get("recording") or {}
    if rec_info.get("image_write_errors"):
        print("xclean: error: %d of the recording's images were not written to %s (first error: %s). The recording "
              "is not complete." % (rec_info["image_write_errors"], a.record, rec_info.get("first_write_error")),
              file=sys.stderr, flush=True)
        rc = 1
    line = None
    if summ is not None:
        try:
            line = summary_line(out, summ=summ)
        except Exception as e:
            print("xclean: warning: the end-of-run summary could not be made (%s: %s)" % (type(e).__name__, e),
                  flush=True)
    if line:
        print(line, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
