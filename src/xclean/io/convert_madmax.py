# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""MADMAX (Mars-analog Morocco, DLR) -> xclean sequence folder.

Source (per experiment X-N; the zips are extracted next to themselves, into left_/ right_/ gt_/ calib_, the first
time they are needed):
    X-N_img_rect_left.zip   img_rect_left_<ns>.png   RECTIFIED
    X-N_img_rect_right.zip  img_rect_right_<ns>.png
    X-N_imu_data.csv        ROS-csv: angular_velocity + linear_acceleration (already rad/s and m/s^2, XSens)
    X-N_ground_truth.zip    gt_/**/gt_6DoF_gnss_and_imu.csv   t, xyz, vel, qw qx qy qz, cov...
    X_calibration.zip       calib_/calibration/camera_rect_{left,right}_info.txt  P matrices
                            calib_/calibration/tf__imu_to_camera_left.csv         body <- camera pose

Rectified pair => ideal pinhole calib (M from P, D = 0, R = I, T = [-B, 0, 0]).
"""
from __future__ import annotations

import ast
import csv
import json
import os
import re
import zipfile

import numpy as np

from xclean.io.convert import (ConvertError, add_mode_argument, link_or_copy, need_dir, need_file, open_text,
                               prepare_output)

HELP = "MADMAX experiment folder (X-N/ with the experiment's zips and IMU csv)"
INPUT_HELP = "the MADMAX experiment folder, for example .../F-5/ (holds F-5_img_rect_left.zip, F-5_imu_data.csv, ...)"
DESCRIPTION = """Convert one MADMAX experiment (Mars-analog Morocco 2018, DLR).

expected input (the experiment's download folder <X-N>, for example F-5):
  <X-N>/<X-N>_img_rect_left.zip    (or its contents already in <X-N>/left_/)
  <X-N>/<X-N>_img_rect_right.zip   (or <X-N>/right_/)
  <X-N>/<X-N>_ground_truth.zip     (or <X-N>/gt_/); without it no gt.csv is written
  <X-N>/<X>_calibration.zip        (or <X-N>/calib_/calibration/)
  <X-N>/<X-N>_imu_data.csv

The zips are extracted into <X-N>/left_/, right_/, gt_/ and calib_/ the first time (so the input folder must be
writable once); later conversions reuse those folders.  Left and right images are paired by nearest timestamp (at
most 35 ms apart).  calib.json "dataset" is madmax_<X-N>.

example:
  xclean convert madmax /data/madmax/F-5 /data/xclean/madmax_F5
"""


def add_arguments(p):
    p.add_argument("--experiment", help="experiment name such as F-5 (default: the input folder's name)")
    add_mode_argument(p)


def quat_to_R(x, y, z, w):
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def ensure_unzipped(d, exp):
    for tag, sub in (("_img_rect_left", "left_"), ("_img_rect_right",
                     "right_"), ("_ground_truth", "gt_")):
        z = os.path.join(d, f"{exp}{tag}.zip")
        o = os.path.join(d, sub)
        if os.path.isfile(z) and not os.path.isdir(o):
            with zipfile.ZipFile(z) as f:
                f.extractall(o)
    cz = [f for f in os.listdir(d) if f.endswith("_calibration.zip")]
    if cz and not os.path.isdir(os.path.join(d, "calib_")):
        with zipfile.ZipFile(os.path.join(d, cz[0])) as f:
            f.extractall(os.path.join(d, "calib_"))


def parse_info(path):
    K = P = None
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("K:"):
                K = np.array(ast.literal_eval(ln[2:].strip())).reshape(3, 3)
            if ln.startswith("P:"):
                P = np.array(ast.literal_eval(ln[2:].strip())).reshape(3, 4)
    if P is None:
        raise ConvertError(f"{path}: no 'P: [...]' projection matrix")
    return K, P


def find_pngs(root, pat):
    out = []
    rx = re.compile(pat)
    for dirpath, _, files in os.walk(root):
        for fn in files:
            m = rx.match(fn)
            if m:
                out.append((int(m.group(1)), os.path.join(dirpath, fn)))
    out.sort()
    return out


def convert(d, out, exp, mode="auto"):
    ensure_unzipped(d, exp)
    cal_dir = os.path.join(d, "calib_", "calibration")
    for name in ("camera_rect_left_info.txt", "camera_rect_right_info.txt", "tf__imu_to_camera_left.csv"):
        need_file(os.path.join(cal_dir, name), f"calibration file {name} (from {exp[:1]}_calibration.zip)")
    for sub in ("left_", "right_"):
        need_dir(os.path.join(d, sub), f"{sub} images (from {exp}_img_rect_{sub[:-1]}.zip)")
    os.makedirs(os.path.join(out, "left"), exist_ok=True)
    os.makedirs(os.path.join(out, "right"), exist_ok=True)

    _, Pl = parse_info(os.path.join(cal_dir, "camera_rect_left_info.txt"))
    _, Pr = parse_info(os.path.join(cal_dir, "camera_rect_right_info.txt"))
    M = Pl[:3, :3]
    baseline = float(-Pr[0, 3] / Pr[0, 0])
    # body(=IMU) <- camera_left pose
    with open(os.path.join(cal_dir, "tf__imu_to_camera_left.csv"), encoding="utf-8") as f:
        tf = f.read().splitlines()
    v = [float(x) for x in tf[-1].split("#")[0].split(",")]
    T_b_c = np.eye(4)
    T_b_c[:3, :3] = quat_to_R(v[3], v[4], v[5], v[6])
    T_b_c[:3, 3] = v[:3]

    lefts = find_pngs(os.path.join(d, "left_"), r"img_rect_left_(\d+)\.png")
    rights = find_pngs(os.path.join(d, "right_"),
                       r"img_rect_right_(\d+)\.png")
    rt = np.array([t for t, _ in rights], dtype=np.int64)
    pairs = []
    for t, p in lefts:
        j = np.searchsorted(rt, t)
        best, bdt = None, 1 << 62
        for k in (j - 1, j):
            if 0 <= k < len(rt) and abs(int(rt[k]) - t) < bdt:
                best, bdt = k, abs(int(rt[k]) - t)
        if best is not None and bdt <= 35_000_000:      # half frame @14 Hz
            pairs.append((t, p, rights[best][1], int(rt[best])))
    if not pairs:
        raise ConvertError(f"no left/right image pairs (at most 35 ms apart) in {d}/left_ and right_")
    t0 = pairs[0][0]

    n_link = n_copy = 0
    with open_text(os.path.join(out, "timestamps.csv")) as f:
        f.write("frame_id,timestamp_s,left_t,right_t\n")
        for i, (tl, pl, pr, tr) in enumerate(pairs):
            f.write(f"{i},{(tl - t0) / 1e9:.9f},{(tl - t0) / 1e9:.9f},"
                    f"{(tr - t0) / 1e9:.9f}\n")
            for sub, src in (("left", pl), ("right", pr)):
                how = link_or_copy(src, os.path.join(out, sub, f"{i:06d}.png"), mode)
                n_link += how == "link"
                n_copy += how == "copy"

    n_imu = 0
    with open(need_file(os.path.join(d, f"{exp}_imu_data.csv"), "IMU file"), encoding="utf-8") as fi, \
            open_text(os.path.join(out, "imu.jsonl")) as fo:
        rd = csv.DictReader(fi)
        for row in rd:
            t = int(row["field.header.stamp"]) / 1e9 - t0 / 1e9
            fo.write(json.dumps({
                "t": round(t, 9),
                "ax": float(row["field.linear_acceleration.x"]),
                "ay": float(row["field.linear_acceleration.y"]),
                "az": float(row["field.linear_acceleration.z"]),
                "gx": float(row["field.angular_velocity.x"]),
                "gy": float(row["field.angular_velocity.y"]),
                "gz": float(row["field.angular_velocity.z"])}) + "\n")
            n_imu += 1

    gt_path = None
    for dirpath, _, files in os.walk(os.path.join(d, "gt_")):
        for fn in files:
            if fn == "gt_6DoF_gnss_and_imu.csv":
                gt_path = os.path.join(dirpath, fn)
    n_gt = 0
    if gt_path:
        with open(gt_path, encoding="utf-8") as fi, open_text(os.path.join(out, "gt.csv")) as fo:
            fo.write("#t_ns,x,y,z,qw,qx,qy,qz\n")
            for ln in fi:
                if ln.startswith("%"):
                    continue
                p = [float(x) for x in ln.split(",")]
                if len(p) < 11:
                    continue
                t_ns = int(p[0] * 1e9) - t0
                fo.write(f"{t_ns},{p[1]:.6f},{p[2]:.6f},{p[3]:.6f},"
                         f"{p[7]:.9f},{p[8]:.9f},{p[9]:.9f},{p[10]:.9f}\n")
                n_gt += 1

    calib = {
        "source": "MADMAX", "dataset": f"madmax_{exp}",
        "camera_width": 1032, "camera_height": 772,
        "camera_fps": 14, "imu_rate_hz": 100,
        "M1": M.tolist(), "D1": [0.0] * 5,
        "M2": M.tolist(), "D2": [0.0] * 5,
        "R_stereo": np.eye(3).tolist(),
        "T_stereo": [-baseline, 0.0, 0.0],
        "baseline_m": baseline,
        "R_cam0_imu": T_b_c[:3, :3].T.tolist(),
        "T_body_cam0": T_b_c.tolist(),
        "t0_ns": 0,
        "num_frames": len(pairs),
        # These densities are the declared weights the evaluation used, not the sensor's measured white noise
        # (measured: gyro about 5.4e-4, accel about 9.7e-4).  On this rig the declared gyro, 1.8x tighter than
        # measured, gave clearly better trajectories than the measured values in an A/B test: the declared sigmas
        # also absorb unmodelled coloured error (bias instability, scale factor, mounting).  Change them only with
        # an A/B test of your own.
        "imu_noise": {
            "gyroscope_noise_density": 3.0e-4,
            "gyroscope_random_walk": 1.0e-5,
            "accelerometer_noise_density": 1.5e-3,
            "accelerometer_random_walk": 1.0e-4,
        },
    }
    with open_text(os.path.join(out, "calib.json")) as f:
        json.dump(calib, f, indent=2)
    return {"frames": len(pairs), "image size": "1032 x 772", "images": f"{n_link} hard-linked, {n_copy} copied",
            "baseline": "%.2f cm" % (baseline * 100), "IMU samples": n_imu,
            "ground truth": (f"{n_gt} poses" if gt_path else "none (no gt_6DoF_gnss_and_imu.csv found)")}


def run(args):
    d = args.input
    need_dir(d, "input folder")
    exp = args.experiment or os.path.basename(os.path.normpath(d))
    need_file(os.path.join(d, f"{exp}_imu_data.csv"),
              f"{exp}_imu_data.csv (give the experiment folder, e.g. .../F-5, or name it with --experiment)")
    for zname, sub in (("_img_rect_left", "left_"), ("_img_rect_right", "right_")):
        if not (os.path.isfile(os.path.join(d, f"{exp}{zname}.zip")) or os.path.isdir(os.path.join(d, sub))):
            raise ConvertError(f"neither {exp}{zname}.zip nor {sub}/ is in {d}")
    if not (any(f.endswith("_calibration.zip") for f in os.listdir(d))
            or os.path.isdir(os.path.join(d, "calib_", "calibration"))):
        raise ConvertError(f"no *_calibration.zip and no calib_/calibration/ in {d}")
    prepare_output(args.output, args.force, [d])
    return convert(d, args.output, exp, args.mode)
