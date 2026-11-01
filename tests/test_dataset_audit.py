# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""The sequence-folder audit (xclean/eval/dataset_audit.py) on a tiny synthetic folder (docs/SEQUENCE_FORMAT.md)."""
import json
import os
import subprocess
import sys

import numpy as np

from xclean.eval.dataset_audit import audit, G

N_FRAMES, FPS, IMU_HZ, T0_NS = 50, 20.0, 200.0, 1_600_000_000_000_000_000


def make_sequence(root, n=N_FRAMES):
    import cv2
    d = root / "seq"
    for sub in ("left", "right"):
        (d / sub).mkdir(parents=True)
    img = np.zeros((8, 8), np.uint8)
    t = np.arange(n) / FPS
    with open(d / "timestamps.csv", "w") as f:
        f.write("frame_id,timestamp_s,left_t,right_t\n")
        for i, ti in enumerate(t):
            f.write("%d,%.6f,%.6f,%.6f\n" % (i, ti, ti, ti))
            for sub in ("left", "right"):
                cv2.imwrite(str(d / sub / ("%06d.png" % i)), img)
    calib = {"num_frames": n, "camera_fps": FPS, "imu_rate_hz": IMU_HZ, "t0_ns": T0_NS,
             "imu_noise": {"gyroscope_noise_density": 1.7e-4, "gyroscope_random_walk": 1.9e-5,
                           "accelerometer_noise_density": 2.0e-3, "accelerometer_random_walk": 3.0e-3}}
    (d / "calib.json").write_text(json.dumps(calib))
    ti = np.arange(0.0, t[-1] + 0.1, 1.0 / IMU_HZ)
    rng = np.random.RandomState(0)
    az = G + rng.standard_normal(len(ti)) * 1e-4     # a real accelerometer is noisy: a constant would read as
    az[len(az) // 2] += 0.5                           # "pinned at the rail"; one bump keeps the maximum unique
    with open(d / "imu.jsonl", "w") as f:
        for k in range(len(ti)):
            f.write(json.dumps({"t": float(ti[k]), "gx": 0.0, "gy": 0.0, "gz": 0.0,
                                "ax": 0.0, "ay": 0.0, "az": float(az[k])}) + "\n")
    with open(d / "gt.csv", "w") as f:
        f.write("#t_ns,x,y,z,qw,qx,qy,qz\n")
        for k in range(0, len(ti), 4):
            f.write("%d,0,0,0,1,0,0,0\n" % (T0_NS + int(round(ti[k] * 1e9))))
    return d


def test_clean_folder_has_no_flags(tmp_path):
    flags, info = audit(str(make_sequence(tmp_path)))
    assert flags == [], flags
    assert "frames 50 @ 20.0 fps" in info and "imu" in info and "100% span" in info


def test_not_a_sequence_folder_is_skipped(tmp_path):
    assert audit(str(tmp_path)) is None


def test_missing_image_is_flagged(tmp_path):
    d = make_sequence(tmp_path)
    os.remove(d / "right" / "000007.png")
    flags, _ = audit(str(d))
    assert flags == ["right/ has 49 images vs 50 csv rows"]


def test_bad_imu_and_gt_are_flagged(tmp_path):
    d = make_sequence(tmp_path)
    lines = (d / "imu.jsonl").read_text().splitlines()
    lines.insert(10, lines[10])                       # a duplicated IMU stamp
    (d / "imu.jsonl").write_text("\n".join(lines) + "\n")
    gt = (d / "gt.csv").read_text().splitlines()
    gt[5], gt[6] = gt[6], gt[5]                       # GT out of order
    (d / "gt.csv").write_text("\n".join(gt) + "\n")
    calib = json.loads((d / "calib.json").read_text())
    del calib["imu_noise"]
    (d / "calib.json").write_text(json.dumps(calib))
    flags, _ = audit(str(d))
    assert any(f.startswith("imu dt==0 x1") for f in flags), flags
    assert any(f.startswith("gt non-monotonic") for f in flags), flags
    assert any("NO imu_noise" in f for f in flags), flags


def test_cli_exit_codes(tmp_path):
    d = make_sequence(tmp_path)
    run = [sys.executable, "-m", "xclean.cli", "audit", str(d)]
    r = subprocess.run(run, capture_output=True, text=True)
    assert r.returncode == 0 and "0 dataset(s) flagged" in r.stdout, r.stdout + r.stderr
    os.remove(d / "left" / "000000.png")
    r = subprocess.run(run, capture_output=True, text=True)
    assert r.returncode == 1 and "1 dataset(s) flagged" in r.stdout, r.stdout + r.stderr
