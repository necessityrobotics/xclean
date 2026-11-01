# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""FEEDER: releases frames and IMU at their recorded stamps (paced) or one frame per engine acknowledgement (lockstep).

Images are decoded AHEAD of release by a reader thread into a bounded buffer (decode is not the engine's work; a camera
delivers decoded buffers), so decode time never delays a release.  Queue depth 1, latest-wins: a release that finds the
previous frame unconsumed REPLACES it, and the feeder logs that frame as dropped.  IMU samples are never dropped (a
queue).  Every planned and actual release time is logged; validity: release jitter p99 <= 1 ms, replay slope
1.000 +- 0.001 (paced mode only).
"""
from __future__ import annotations

import json
import queue
import threading
import time

import numpy as np

from xclean.live.protocol import FrameSlot, now_ns


def _read_index(path):
    out = []
    with open(path + "/timestamps.csv", encoding="utf-8") as f:
        f.readline()
        for ln in f:
            p = ln.strip().split(",")
            if len(p) >= 2:
                out.append((int(p[0]), float(p[1])))
    return out


def _read_imu(path):
    import os
    t, g, a = [], [], []
    p = path + "/imu.jsonl"
    if not os.path.isfile(p):
        return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                d = json.loads(ln)
                t.append(d["t"])
                g.append((d["gx"], d["gy"], d["gz"]))
                a.append((d["ax"], d["ay"], d["az"]))
    return np.asarray(t, float), np.asarray(g, float), np.asarray(a, float)


def feeder_main(cfg, slot_name, imu_q, ctl, log_path):
    """cfg: dataset (path), ext, mode ('paced'|'lockstep'), frames (optional limit).  ctl: dict of mp primitives:
    ready (Event, set by the engine after warm-up), new_frame (Event), done (Value: last frame index the engine
    finished, lockstep), done_evt (Event), stop (Event)."""
    import cv2
    from xclean.io.imageio import imread
    path, ext, mode = cfg["dataset"], cfg.get("ext", "png"), cfg["mode"]
    frames = _read_index(path)
    if cfg.get("frames"):
        frames = frames[:int(cfg["frames"])]
    if not frames:
        raise SystemExit("xclean: error: %s/timestamps.csv has no frame rows" % path)
    ti, gi, ai = _read_imu(path)
    probe = imread("%s/left/%06d.%s" % (path, frames[0][0], ext), cv2.IMREAD_GRAYSCALE)
    if probe is None:
        raise SystemExit("xclean: error: cannot read %s/left/%06d.%s" % (path, frames[0][0], ext))
    H, W = probe.shape
    slot = FrameSlot(slot_name, W, H)
    buf = queue.Queue(maxsize=256)
    stop_read = threading.Event()

    def reader():
        for idx, t in frames:
            if stop_read.is_set():
                return
            pl, pr = "%s/left/%06d.%s" % (path, idx, ext), "%s/right/%06d.%s" % (path, idx, ext)
            l = imread(pl, cv2.IMREAD_GRAYSCALE)
            r = imread(pr, cv2.IMREAD_GRAYSCALE)
            buf.put((idx, t, l, r) if (l is not None and r is not None) else ("unreadable", pl if l is None else pr))
            if l is None or r is None:
                return
        buf.put(None)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    # wait for a full decode-ahead window before the clock starts (and for the engine's warm-up)
    t_wait = time.time()
    while buf.qsize() < min(64, len(frames)) and th.is_alive() and time.time() - t_wait < 60:
        time.sleep(0.01)
    ctl["ready"].wait()
    log = {"mode": mode, "frames_total": len(frames), "frames": [], "imu_released": 0, "replaced": []}
    t0 = float(frames[0][1])
    # P1.3 forced-loss protocol: frames whose stamp - t0 lies in [b0, b0 + dur) of any
    # blackout are WITHHELD (never released; the IMU keeps flowing at their instants); logged in log["withheld"]
    blackout = [(float(b0), float(dur)) for b0, dur in (cfg.get("blackout") or [])]
    log["blackout"] = blackout
    log["withheld"] = []
    last_pub = None                            # lock-step: the last RELEASED frame index (the engine's done signal)
    T0 = now_ns() + 50_000_000                 # first release 50 ms after the go
    ii = 0                                     # next IMU sample to release
    n_imu = len(ti)

    def release_imu_upto(t_stamp):
        nonlocal ii
        j = int(np.searchsorted(ti, t_stamp, side="right"))
        if j > ii:
            imu_q.put((ti[ii:j].copy(), gi[ii:j].copy(), ai[ii:j].copy()))
            log["imu_released"] += j - ii
            ii = j

    def wait_until(ns):
        while True:
            d = ns - now_ns()
            if d <= 0:
                return
            if d > 2_000_000:
                time.sleep((d - 1_500_000) / 1e9)
            # final approach: spin

    for k in range(len(frames)):
        item = buf.get()
        if item is None:
            break
        if item[0] == "unreadable":            # the run ends with an error; the harness ends the engine's stream
            stop_read.set()
            raise SystemExit("xclean: error: cannot read %s" % item[1])
        idx, t, l, r = item
        withheld = any(b0 <= t - t0 < b0 + dur for b0, dur in blackout)
        if mode == "lockstep":
            if last_pub is not None and not withheld:
                while True:
                    if ctl["done"].value >= last_pub or ctl["stop"].is_set():
                        break
                    ctl["done_evt"].wait(0.05)
                    ctl["done_evt"].clear()
            release_imu_upto(t)
            planned = now_ns()
        else:
            planned = T0 + int(round((t - t0) * 1e9))
            # IMU in PACKETS: at this frame's release instant, one packet with every
            # sample stamped <= the frame stamp - all of them due by now, so the fence is exact (no sample is seen
            # before its stamp has passed).  k = 1 packet per camera period, derived: the engine reads IMU only at
            # frame instants (its fence is the frame stamp), so a finer packet stream delivers nothing it can use
            # earlier, and one put per frame replaces one per IMU sample (66 per frame on 4Seasons' 2000 Hz).
            wait_until(planned)
            release_imu_upto(t)
        if ctl["stop"].is_set():
            break
        if withheld:                           # forced loss: the IMU flowed, the images do not
            log["withheld"].append([int(idx), float(t)])
            continue
        with ctl["lock"]:
            if slot.hdr[3] == 0 and slot.hdr[0] > 0:          # the previous frame was never consumed: REPLACED
                log["replaced"].append(int(slot.hdr[1]))
                slot.hdr[7] += 1
            slot.buf[0] = l
            slot.buf[1] = r
            slot.tstamp[0] = t
            slot.tstamp[1] = float(ti[ii - 1]) if ii > 0 else -1.0
            slot.hdr[1] = idx
            actual = now_ns()
            slot.hdr[2] = actual
            slot.hdr[3] = 0
            slot.hdr[0] += 1
        ctl["new_frame"].set()
        log["frames"].append([int(idx), float(t), int(planned), int(actual)])
        last_pub = int(idx)
    # end of stream: remaining IMU, then EOS
    release_imu_upto(float("inf"))
    with ctl["lock"]:
        slot.hdr[4] = 1
    ctl["new_frame"].set()
    stop_read.set()
    fr = np.asarray([[f[2], f[3]] for f in log["frames"]], float)
    stamps = np.asarray([f[1] for f in log["frames"]], float)
    val = {}
    if mode == "paced" and len(fr) > 2:
        jit = (fr[:, 1] - fr[:, 0]) / 1e6
        A = np.stack([stamps - stamps[0], np.ones(len(stamps))], 1)
        slope = float(np.linalg.lstsq(A, (fr[:, 1] - fr[0, 1]) / 1e9, rcond=None)[0][0])
        val = {"jitter_ms_p50": float(np.percentile(jit, 50)), "jitter_ms_p99": float(np.percentile(jit, 99)),
               "jitter_ms_max": float(jit.max()), "replay_slope": slope,
               "valid": bool(np.percentile(jit, 99) <= 1.0 and abs(slope - 1.0) <= 1e-3)}
    log["validity"] = val
    log["T0_ns"] = int(T0)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f)
    slot.close()
