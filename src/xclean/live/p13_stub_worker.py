#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""P1.3 STUB WORKER (XCLEAN_P13_WORKER=stub; a diagnostic run): speaks the worker interface of xclean/live/p13_link.py
(WorkerSide) and does nothing a real worker decides.  It exists so the ENGINE SIDE (hooks, KfRing, seqlock, lock-step
synchronisation, slew, flags, the process tree) can be built and identity-tested before xclean/slam/ lands.

Per message: "kf" -> read the slot's images (a checksum, so the ring is really exercised), free the slot, count, and
report progress(n); "edge_final" / "segment_start" -> validated (finite, shapes) and counted; "eos" -> write
<out_dir>/p13_worker.json and exit.

Optional scripted corrections (tests of the frame side only; NEVER a result): XCLEAN_P13_STUB_SCRIPT =
'k:dx:dyaw_deg:lat_s[;k:...]' publishes, after the k-th keyframe packet, a head-anchored target that shifts the map
frame by dx metres along its x axis and rotates it by dyaw about z, with finish_t = that keyframe's stamp + lat_s
(modelled; lock-step applies it at the first frame whose stamp >= finish_t).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np


def _script():
    raw = (os.environ.get("XCLEAN_P13_STUB_SCRIPT", "") or "").strip()
    out = {}
    for part in [x for x in raw.split(";") if x.strip()]:
        k, dx, dyaw, lat = part.split(":")
        out[int(k)] = (float(dx), float(dyaw), float(lat))
    return out


def worker_main(handles, wcfg):
    from xclean.live.p13_link import WorkerSide
    assert "cupy" not in sys.modules
    side = WorkerSide(handles)
    script = _script()
    st = {"kf": 0, "images": 0, "dropped": 0, "edge_final": 0, "segment_start": 0, "bad": [], "checksum": 0,
          "published": 0, "kinds": {}, "first_kf_wall_lag_ms": [], "edge_prev_none": 0, "links": {}}
    side.ready()
    epoch = 0
    T_cur = np.eye(4)
    t0 = time.perf_counter()
    while True:
        m = side.get(timeout=1.0)
        if m is None:
            continue
        ty = m.get("type")
        if ty == "eos":
            st["eos_n_kf"] = m.get("n_kf")
            break
        if ty == "kf":
            st["kf"] += 1
            st["kinds"][m.get("kind")] = st["kinds"].get(m.get("kind"), 0) + 1
            if m.get("edge_prev") is None:
                st["edge_prev_none"] += 1
            else:
                S = np.asarray(m["edge_prev"]["S"], float)
                if S.shape != (36,) or not np.all(np.isfinite(S)):
                    st["bad"].append(["edge_prev", m.get("n")])
            if m.get("slot") is None:
                st["dropped"] += 1
            else:
                left, right = side.images(m["slot"])
                st["checksum"] = (st["checksum"] + int(left[::17, ::13].sum()) + int(right[::17, ::13].sum())) \
                    % (1 << 61)
                st["images"] += 1
                side.free(m["slot"])
            if m.get("n") in script:
                dx, dyaw, lat = script[m["n"]]
                epoch += 1
                a = math.radians(dyaw)
                D = np.eye(4)
                D[:3, :3] = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0],
                                      [0.0, 0.0, 1.0]])
                D[0, 3] = dx
                T_cur = D @ T_cur
                side.publish(epoch, T_cur, 1, epoch, 0, float(m["t"]) + lat)
                st["published"] += 1
            side.progress(int(m["n"]))
        elif ty == "edge_final":
            st["edge_final"] += 1
            S = np.asarray(m["S"], float)
            if S.shape != (36,) or not np.all(np.isfinite(S)):
                st["bad"].append(["edge_final", m.get("sid_a")])
        elif ty == "segment_start":
            st["segment_start"] += 1
            k = "%s:%s" % (m.get("kind"), "cov" if m.get("S") is not None else "nocov")
            st["links"][k] = st["links"].get(k, 0) + 1
        else:
            st["bad"].append(["type", ty])
    st["wall_s"] = time.perf_counter() - t0
    out_dir = wcfg.get("out_dir")
    if out_dir:
        json.dump(st, open(os.path.join(out_dir, "p13_worker.json"), "w"), indent=1, default=str)
    side.close()
    return 0
