#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Offline lock-step driver of the P1 engine (development + tests; the gate runs use the live harness with
XCLEAN_ENGINE=p1 --mode lockstep).

    CUDA_VISIBLE_DEVICES=-1 python -m xclean.vio.run_p1 <sequence dir | row tag> --out DIR [--frames N] [--start K]
        [--rows REGISTRY.json]      (a row tag needs a registry file: xclean.io.rows, docs/SEQUENCE_FORMAT.md)

Writes DIR/trajectory_live.csv (the pose published per frame, never revised; T_world_cam0_rectified;
t_frame,x,y,z,qx,qy,qz,qw,frame_index,state,t_publish_s,latency_ms, as xclean/run.py) and DIR/summary.json
(P1 summary + latency stages).  An earlier output of this tool in DIR is replaced; any other non-empty DIR is
refused.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np


def run(tag, out, frames=None, start=0, threads=None, verbose=True):
    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, REPO)
    import cv2
    from scipy.spatial.transform import Rotation as Rot
    from xclean.io import rows as R
    from xclean.io.dataset import Dataset
    from xclean.obs.latency import Ledger
    from xclean.vio.engine_live import P1Engine, warmup
    if threads:
        cv2.setNumThreads(int(threads))
    cv2.setRNGSeed(0)
    np.random.seed(0)
    row = R.get(tag)
    ds = Dataset(row.dataset)
    if out:
        from xclean.io.folders import check_output_folder
        problem = check_output_folder("run_p1", "--out", out, True, "dev_run", inputs=[row.dataset, row.gt])
        if problem:                                           # refused before anything starts: exit code 2
            print(problem, file=sys.stderr)
            raise SystemExit(2)
        if os.path.isdir(out):
            shutil.rmtree(out)                                # an earlier output of this tool (checked above)
        os.makedirs(out)
    led = Ledger(run=tag, tier=1, sync=None, gpu_stages=())
    imu_path = os.path.join(row.dataset, "imu.jsonl")
    if os.path.isfile(imu_path):
        from xclean.vio.engine_live import OfflineImu
        ds.imu_stream = OfflineImu(imu_path)
    eng = P1Engine(ds, led)
    warmup(ds.rig)
    fr_all = ds.frames()[start:]
    if frames:
        fr_all = fr_all[:int(frames)]
    rows = []
    t0 = time.perf_counter()
    for k, fr in enumerate(fr_all):
        ts = time.perf_counter()
        st = eng.step(fr)
        led.frame_done(fr.index)
        lat = (time.perf_counter() - ts) * 1e3
        T = eng.T_wc
        q = Rot.from_matrix(T[:3, :3]).as_quat()
        rows.append((fr.t, *T[:3, 3], *q, fr.index, st, time.perf_counter() - t0, lat))
        if verbose and k % 250 == 0:
            print("  %6d/%d  %-9s kf=%d lm=%d sig_px=%.2f kappa_d=%.2f  %.1f ms" % (
                k, len(fr_all), st, eng.stats["kfs"], int(eng.lm_alive[:eng.n_lm].sum()), eng.sig_px,
                eng.kappa_d, lat), flush=True)
    if out:
        with open(os.path.join(out, "trajectory_live.csv"), "w", encoding="utf-8") as f:
            f.write("# frame convention: T_world_cam0_rectified (P1 lock-step: pose published per frame, never "
                    "revised)\n# t_frame,x,y,z,qx,qy,qz,qw,frame_index,state,t_publish_s,latency_ms\n")
            for r in rows:
                f.write("%.9f,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f,%d,%s,%.6f,%.3f\n" % r)
        summ = {"engine": "p1", "tag": tag, "frames": len(rows), "p1": eng.summary(),
                "coasted_pct": 100.0 * eng.stats["coasted"] / max(len(rows), 1),
                "env": {k: os.environ[k] for k in sorted(os.environ) if k.startswith("XCLEAN_")}}
        json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
        led.save(os.path.join(out, "latency.json"))
    return rows, eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--rows", default=None, help="row registry JSON for tag lookups (xclean.io.rows)")
    a = ap.parse_args()
    if a.rows:
        from xclean.io import rows as R
        R.set_registry(a.rows)
    run(a.tag, a.out, a.frames, a.start, a.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
