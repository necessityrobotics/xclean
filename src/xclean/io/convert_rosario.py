# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Rosario agricultural dataset, version 1 (raw files) -> xclean sequence folder.

Source layout (per sequence, after extracting sequenceNN.tar.gz*):
    extracted/sequenceNN/zed/left_<epoch.s>.png + right_<epoch.s>.png
    extracted/sequenceNN/imu.log      t, gx gy gz [deg/s], ax ay az [mg]
    calibrationNN.yaml                Kalibr 2-camera radtan + T_cam_imu (+ the IMU's gyro_offset)
    sequenceNN_gt.txt                 TUM format, RTK positions (orientation = identity)

Output: left/right images, timestamps.csv, imu.jsonl (rad/s + m/s^2), calib.json (the engine rectifies radtan
itself), gt.csv.

This applies the dataset's own v0.0.4 remediation (its CHANGELOG, 2022-06-14) to the 2020 raw files:
  * IMU timestamps equispaced within contiguous blocks (see _equispace_blocks);
  * all streams truncated to the ground-truth span (the 2022-updated ground truth of sequences 01 and 06 ends where
    the authors cut the "wrong IMU measurements" tails).
The gyroscope turn-on offset shipped in calibrationNN.yaml (imu: gyro_offset, deg/s) is subtracted.  The shipped
acc_offset is deliberately NOT applied: subtracting it moves the static |a| from +0.09 m/s^2 off g to -0.22, adding
it to +0.38, so its frame/convention does not match this stream.
The ground truth is POSITION-only (RTK antenna); its quaternions are identity, so the antenna-to-camera lever arm
shows up as a constant offset that an SE(3) alignment absorbs.  IMU noise densities are LSM6DS0 datasheet-class
values (hardware facts, not tuning).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from xclean.io.convert import (ConvertError, add_mode_argument, link_or_copy, need_dir, need_file, open_text,
                               prepare_output, read_kalibr_yaml)

HELP = "Rosario dataset v1, raw files (the folder holding calibrationNN.yaml, sequenceNN_gt.txt, extracted/)"
INPUT_HELP = "the Rosario v1 folder that holds calibrationNN.yaml, sequenceNN_gt.txt and extracted/sequenceNN/"
DESCRIPTION = """Convert one sequence of the Rosario agricultural dataset, version 1 (the raw-file release, not the
rosbags).

expected input (<root>, with --seq NN):
  <root>/calibrationNN.yaml
  <root>/sequenceNN_gt.txt
  <root>/extracted/sequenceNN/zed/left_<time>.png, right_<time>.png
  <root>/extracted/sequenceNN/imu.log
where extracted/sequenceNN/ is what the sequence's sequenceNN.tar.gz parts unpack to.

The conversion applies the dataset's own v0.0.4 fixes: IMU timestamps made equispaced within contiguous blocks,
every stream cut to the ground-truth time span, and the shipped gyroscope offset subtracted.  Left and right images
are paired by nearest timestamp (at most 33 ms apart).

example:
  xclean convert rosario-v1 /data/rosario_v1 /data/xclean/rosario_04 --seq 04
"""

G = 9.80665
D2R = np.pi / 180.0


def add_arguments(p):
    p.add_argument("--seq", required=True, help="sequence number, 01 to 06")
    add_mode_argument(p)


def _equispace_blocks(ts, nominal=None, gap_factor=1.5):
    """Rebuild an ms-quantised IMU time base as uniform grids within
    contiguous blocks, split at real stream gaps.

    The dataset's own CHANGELOG (v0.0.4, 2022-06-14) records that the authors
    republished the rosbags "with IMU measurements temporally equispaced" -
    the raw imu.log files (2020) predate that fix and retain the pathology:
    timestamps are written at millisecond resolution against a 7.03 ms true
    period, so dt jitters over 5-9 ms and 7-461 samples per sequence collide
    into the SAME millisecond (dt == 0).  An intake guard on dt <= 0 would
    silently drop those real measurements, and every surviving interval
    carries up to ~28% dt error into preintegration.
    Measured block periods are 7.032 ms (142.2 Hz) with p5-p95 6.99-7.04
    across all six sequences, so equispacing is well-posed.

    Blocks are split where dt > gap_factor * median_dt; real gaps are
    preserved as gaps.
    """
    ts = np.asarray(ts, dtype=np.float64)
    if len(ts) < 3:
        return ts
    dt = np.diff(ts)
    pos = dt[dt > 0]
    med = float(np.median(pos)) if len(pos) else (nominal or 0.007)
    cuts = np.where(dt > gap_factor * med)[0]
    starts = np.concatenate(([0], cuts + 1))
    ends = np.concatenate((cuts, [len(ts) - 1]))
    out = ts.copy()
    for s, e in zip(starts, ends):
        n = e - s + 1
        if n < 2:
            continue
        span = ts[e] - ts[s]
        if span <= 0:       # a block that is entirely one millisecond
            out[s:e + 1] = ts[s] + np.arange(n) * med
        else:
            out[s:e + 1] = ts[s] + np.arange(n) * (span / (n - 1))
    if not np.all(np.diff(out) > 0):
        bad = int((np.diff(out) <= 0).sum())
        raise ConvertError(f"IMU timestamps could not be made increasing ({bad} non-increasing steps)")
    return out


def load_calib(path):
    c = read_kalibr_yaml(path)

    def cam(k):
        fx, fy, cx, cy = c[k]["intrinsics"]
        M = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        d = list(c[k]["distortion_coeffs"]) + [0.0]     # k1 k2 p1 p2 -> +k3=0
        return M, np.array(d, dtype=np.float64)
    try:
        M1, D1 = cam("cam0")
        M2, D2 = cam("cam1")
        T_c1_c0 = np.array(c["cam1"]["T_cn_cnm1"], dtype=np.float64)
        T_c0_imu = np.array(c["cam0"]["T_cam_imu"], dtype=np.float64)
        res = c["cam0"]["resolution"]
    except (KeyError, TypeError, ValueError) as e:
        raise ConvertError(f"{path}: not a Rosario camera calibration (missing or malformed {e})")
    # The dataset ships the IMU turn-on offsets in this same file.  gyro_offset [deg/s] is applied below: without
    # it the engine reads a ~0.17 rad/s "gyro bias".  acc_offset [mg] is deliberately not applied (see the module
    # docstring).
    imu = c.get("imu", {}) or {}
    gyro_off = np.array(imu.get("gyro_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
    return M1, D1, M2, D2, T_c1_c0, T_c0_imu, res, gyro_off


def read_gt_times(gt_src):
    gt_ts = []
    with open(gt_src, encoding="utf-8") as fi:
        for ln in fi:
            p = ln.split()
            if len(p) >= 8:
                try:
                    gt_ts.append(float(p[0]))
                except ValueError:
                    continue
    if not gt_ts:
        raise ConvertError(f"{gt_src}: no ground-truth rows (TUM format: t x y z qx qy qz qw)")
    return gt_ts


def write_gt(gt_src, out_csv, t0):
    with open(gt_src, encoding="utf-8") as fi, open_text(out_csv) as fo:
        fo.write("#t_ns,x,y,z,qw,qx,qy,qz\n")
        for ln in fi:
            p = ln.split()
            if len(p) < 8:
                continue
            t, x, y, z, qx, qy, qz, qw = (float(v) for v in p[:8])
            fo.write(f"{int((t - t0) * 1e9)},{x:.6f},{y:.6f},{z:.6f},"
                     f"{qw:.6f},{qx:.6f},{qy:.6f},{qz:.6f}\n")


def make_calib(seq, w, h, M1, D1, M2, D2, T_c1_c0, T_c0_imu, n_frames):
    R_stereo = T_c1_c0[:3, :3]
    T_stereo = T_c1_c0[:3, 3]
    baseline = float(np.linalg.norm(T_stereo))
    T_body_cam0 = np.linalg.inv(T_c0_imu)          # body = IMU frame
    R_cam0_imu = T_c0_imu[:3, :3]
    return {
        "source": "Rosario v1", "dataset": f"rosario_{seq}",
        "camera_width": int(w), "camera_height": int(h),
        "camera_fps": 15, "imu_rate_hz": 142,
        "M1": M1.tolist(), "D1": D1.tolist(),
        "M2": M2.tolist(), "D2": D2.tolist(),
        "R_stereo": R_stereo.tolist(), "T_stereo": T_stereo.tolist(),
        "baseline_m": baseline,
        "R_cam0_imu": R_cam0_imu.tolist(),
        "T_body_cam0": T_body_cam0.tolist(),
        "t0_ns": 0,
        "num_frames": n_frames,
        # LSM6DS0 datasheet-class densities (hardware facts, not tuning)
        "imu_noise": {
            "gyroscope_noise_density": 1.2e-4,
            "gyroscope_random_walk": 1.0e-5,
            "accelerometer_noise_density": 8.8e-4,
            "accelerometer_random_walk": 1.0e-4,
        },
    }


def paths(root, seq):
    sdir = os.path.join(root, "extracted", f"sequence{seq}")
    return (sdir, os.path.join(sdir, "zed"), os.path.join(root, f"calibration{seq}.yaml"),
            os.path.join(root, f"sequence{seq}_gt.txt"))


def convert(root, out, seq, mode="auto"):
    sdir, zed, cal_path, gt_src = paths(root, seq)
    os.makedirs(os.path.join(out, "left"), exist_ok=True)
    os.makedirs(os.path.join(out, "right"), exist_ok=True)

    M1, D1, M2, D2, T_c1_c0, T_c0_imu, (w, h), gyro_off = load_calib(cal_path)
    baseline = float(np.linalg.norm(T_c1_c0[:3, 3]))

    # ---- GT span = the authors' own validity window ------------------
    # CHANGELOG v0.0.4 (2022-06-14): the republished bags drop the "last
    # parts of sequences 01 and 06 (wrong IMU measurements)", and the
    # 2022-updated GT files end exactly there (114.7 s / 145.1 s before the
    # raw image/IMU streams do).  Truncating every stream to the GT span
    # applies the dataset's own remediation; sequences whose GT covers the
    # full stream are untouched by construction.
    gt_ts = read_gt_times(gt_src)
    gt_lo, gt_hi = min(gt_ts), max(gt_ts)

    # ---- pair images by timestamp ------------------------------------
    pat = re.compile(r"(left|right)_(\d+\.\d+)\.png$")
    lefts, rights = {}, {}
    for fn in os.listdir(zed):
        m = pat.match(fn)
        if not m:
            continue
        (lefts if m.group(1) == "left" else rights)[float(m.group(2))] = fn
    lt = np.array(sorted(lefts))
    rt = np.array(sorted(rights))
    n_all = len(lt)
    lt = lt[(lt >= gt_lo) & (lt <= gt_hi)]
    n_trunc = n_all - len(lt)
    # nearest right for each left within half a frame (15 Hz -> 33 ms)
    pairs = []
    for t in lt:
        j = np.searchsorted(rt, t)
        best, bdt = None, 1e9
        for k in (j - 1, j):
            if 0 <= k < len(rt) and abs(rt[k] - t) < bdt:
                best, bdt = rt[k], abs(rt[k] - t)
        if best is not None and bdt <= 0.033:
            pairs.append((t, best))
    if not pairs:
        raise ConvertError(f"no left/right image pairs (at most 33 ms apart) inside the ground-truth span in {zed}")
    t0 = pairs[0][0]

    n_link = n_copy = 0
    with open_text(os.path.join(out, "timestamps.csv")) as f:
        f.write("frame_id,timestamp_s,left_t,right_t\n")
        for i, (tl, tr) in enumerate(pairs):
            f.write(f"{i},{tl - t0:.9f},{tl - t0:.9f},{tr - t0:.9f}\n")
            for sub, src in (("left", lefts[tl]), ("right", rights[tr])):
                how = link_or_copy(os.path.join(zed, src), os.path.join(out, sub, f"{i:06d}.png"), mode)
                n_link += how == "link"
                n_copy += how == "copy"
    # remove stale frames beyond the (possibly truncated) pair count
    for sub in ("left", "right"):
        for fn in os.listdir(os.path.join(out, sub)):
            m = re.match(r"(\d{6})\.png$", fn)
            if m and int(m.group(1)) >= len(pairs):
                os.remove(os.path.join(out, sub, fn))

    # ---- IMU: t, gx gy gz [deg/s], ax ay az [mg] ---------------------
    raw = []
    with open(os.path.join(sdir, "imu.log"), encoding="utf-8") as fi:
        for ln in fi:
            # format: "<t> IMU:  gx gy gz ax ay az"
            p = [x for x in ln.replace(",", " ").split()
                 if not x.endswith(":")]
            if len(p) < 7:
                continue
            try:
                raw.append([float(x) for x in p[:7]])
            except ValueError:
                continue
    raw = np.array(raw, dtype=np.float64).reshape(-1, 7)
    raw = raw[(raw[:, 0] >= gt_lo) & (raw[:, 0] <= gt_hi)]
    if len(raw) < 2:
        raise ConvertError(f"{sdir}/imu.log: fewer than 2 IMU samples inside the ground-truth span")
    n_dup_in = int((np.diff(raw[:, 0]) == 0).sum())
    ts_fixed = _equispace_blocks(raw[:, 0])
    with open_text(os.path.join(out, "imu.jsonl")) as fo:
        for k in range(len(raw)):
            t, gx, gy, gz, ax, ay, az = raw[k]
            fo.write(json.dumps({
                "t": round(ts_fixed[k] - t0, 9),
                "ax": ax * G * 1e-3, "ay": ay * G * 1e-3,
                "az": az * G * 1e-3,
                "gx": (gx - gyro_off[0]) * D2R,
                "gy": (gy - gyro_off[1]) * D2R,
                "gz": (gz - gyro_off[2]) * D2R}) + "\n")
    n_imu = len(raw)

    # ---- GT (TUM, position-only RTK) ---------------------------------
    write_gt(gt_src, os.path.join(out, "gt.csv"), t0)

    calib = make_calib(seq, w, h, M1, D1, M2, D2, T_c1_c0, T_c0_imu, len(pairs))
    with open_text(os.path.join(out, "calib.json")) as f:
        json.dump(calib, f, indent=2)
    dtf = np.diff(ts_fixed)
    return {"frames": len(pairs), "image size": f"{w} x {h}", "images": f"{n_link} hard-linked, {n_copy} copied",
            "baseline": "%.1f cm" % (baseline * 100), "IMU samples": n_imu,
            "IMU timing fix": "%d equal stamps in, %d out; rate %.2f Hz"
                              % (n_dup_in, int((dtf == 0).sum()), (len(ts_fixed) - 1) / (ts_fixed[-1] - ts_fixed[0])),
            "cut to GT span": "%d left images outside the ground-truth span dropped" % n_trunc}


def run(args):
    root, seq = args.input, args.seq
    need_dir(root, "input folder")
    sdir, zed, cal_path, gt_src = paths(root, seq)
    need_file(cal_path, f"calibration{seq}.yaml")
    need_file(gt_src, f"sequence{seq}_gt.txt")
    need_dir(zed, f"extracted/sequence{seq}/zed/ (unpack sequence{seq}.tar.gz* into extracted/)")
    need_file(os.path.join(sdir, "imu.log"), f"extracted/sequence{seq}/imu.log")
    prepare_output(args.output, args.force, [root])
    return convert(root, args.output, seq, args.mode)
