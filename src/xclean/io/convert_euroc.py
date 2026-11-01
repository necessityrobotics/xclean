# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""EuRoC MAV (ASL folder layout) -> xclean sequence folder.

EuRoC ships the stereo pair, the IMU and per-sensor YAML calibration as mav0/cam0, mav0/cam1 and mav0/imu0.  This
rewrites them into the sequence layout:

    left/NNNNNN.png, right/NNNNNN.png   the cam0 / cam1 images of every frame both cameras recorded
    timestamps.csv                      frame times in seconds, zero at the first frame (LF line endings)
    imu.jsonl                           gyroscope + accelerometer, same clock
    calib.json                          pinhole + radial-tangential calibration, stereo extrinsic, IMU noise model

Ground truth stays where EuRoC puts it (mav0/state_groundtruth_estimate0/data.csv); pass it to a run with --gt.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from xclean.io.convert import (ConvertError, add_mode_argument, link_or_copy, need_dir, need_file, open_text,
                               prepare_output)

HELP = "EuRoC MAV sequence in ASL folder layout (the folder that contains mav0/)"
INPUT_HELP = "the EuRoC sequence folder that contains mav0/ (for example MH_01_easy/), or mav0/ itself"
DESCRIPTION = """Convert one EuRoC MAV sequence (ASL folder layout, as unpacked from the sequence's .zip).

expected input:
  <input>/mav0/cam0/data.csv, <input>/mav0/cam0/data/<ns>.png, <input>/mav0/cam0/sensor.yaml
  <input>/mav0/cam1/...  (same)
  <input>/mav0/imu0/data.csv, <input>/mav0/imu0/sensor.yaml

Only frames that both cameras recorded are kept.  Times are re-based to zero at the first frame (calib.json t0_ns
holds the absolute epoch).  IMU samples more than 1 s before the first frame are dropped.  calib.json "dataset" is
the input folder's name.  Ground truth is not copied: score with
  --gt <input>/mav0/state_groundtruth_estimate0/data.csv

example:
  xclean convert euroc /data/euroc/MH_01_easy /data/xclean/MH_01_easy
"""


def add_arguments(p):
    add_mode_argument(p)
    p.add_argument("--start", type=int, default=0, help="first stereo pair to keep (default 0)")
    p.add_argument("--end", type=int, default=-1, help="last stereo pair to keep (default: the last one)")


# --------------------------------------------------------------------------- #
# Minimal YAML reader
# --------------------------------------------------------------------------- #
# The EuRoC sensor.yaml files use a tiny, fixed subset of YAML (scalars, flow
# sequences and the T_BS block).  Parsing them directly avoids a YAML
# dependency and is fully deterministic.
def read_sensor_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    out = {}

    # T_BS: data: [ ... 16 numbers ... ]  (may span several lines)
    m = re.search(r"T_BS:.*?data:\s*\[(.*?)\]", text, re.S)
    if m:
        vals = [float(v) for v in m.group(1).replace("\n", " ").split(",")]
        if len(vals) != 16:
            raise ConvertError(f"{path}: T_BS has {len(vals)} entries, expected 16")
        out["T_BS"] = np.array(vals, dtype=np.float64).reshape(4, 4)

    def flow_list(key):
        mm = re.search(rf"^{key}:\s*\[(.*?)\]", text, re.M)
        if not mm:
            return None
        return [float(v) for v in mm.group(1).split(",")]

    for key in ("intrinsics", "distortion_coefficients", "resolution"):
        v = flow_list(key)
        if v is not None:
            out[key] = v

    mm = re.search(r"^rate_hz:\s*([0-9.]+)", text, re.M)
    if mm:
        out["rate_hz"] = float(mm.group(1))
    mm = re.search(r"^distortion_model:\s*(\S+)", text, re.M)
    if mm:
        out["distortion_model"] = mm.group(1)

    # IMU continuous-time noise model: the back-end builds its preintegration covariances from these.
    for key in ("gyroscope_noise_density", "gyroscope_random_walk",
                "accelerometer_noise_density", "accelerometer_random_walk"):
        mm = re.search(rf"^{key}:\s*([0-9.eE+-]+)", text, re.M)
        if mm:
            out[key] = float(mm.group(1))

    return out


def read_csv_rows(path: str):
    """Yield split, comment-free rows of a EuRoC data.csv."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line.split(",")


def se3_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def find_mav0(euroc_root):
    mav0 = os.path.join(euroc_root, "mav0")
    if not os.path.isdir(mav0):
        # allow pointing straight at mav0/
        if os.path.isdir(os.path.join(euroc_root, "cam0")):
            mav0 = euroc_root
        else:
            raise ConvertError(f"no mav0/ folder in {euroc_root}; give the EuRoC sequence folder that contains "
                               f"mav0/ (ASL layout, the unpacked .zip, not the .bag)")
    return mav0


# --------------------------------------------------------------------------- #
def convert(euroc_root: str, out_dir: str, link_mode: str = "auto",
            start: int = 0, end: int = -1) -> dict:
    mav0 = find_mav0(euroc_root)

    cam0_dir = os.path.join(mav0, "cam0")
    cam1_dir = os.path.join(mav0, "cam1")
    imu_csv = os.path.join(mav0, "imu0", "data.csv")
    for p in (cam0_dir, cam1_dir, imu_csv):
        if not os.path.exists(p):
            raise ConvertError(f"missing {p}")

    y0 = read_sensor_yaml(need_file(os.path.join(cam0_dir, "sensor.yaml"), "cam0 calibration"))
    y1 = read_sensor_yaml(need_file(os.path.join(cam1_dir, "sensor.yaml"), "cam1 calibration"))
    yi = read_sensor_yaml(need_file(os.path.join(mav0, "imu0", "sensor.yaml"), "IMU calibration"))

    for name, y in (("cam0", y0), ("cam1", y1)):
        if y.get("distortion_model", "radial-tangential") != "radial-tangential":
            raise ConvertError(f"{name} uses the distortion model {y['distortion_model']!r}; only "
                               f"radial-tangential is supported")
        for key in ("T_BS", "intrinsics", "distortion_coefficients", "resolution"):
            if key not in y:
                raise ConvertError(f"{name}/sensor.yaml has no {key}")

    W, H = int(y0["resolution"][0]), int(y0["resolution"][1])

    def intr(y):
        fu, fv, cu, cv_ = y["intrinsics"]
        return np.array([[fu, 0.0, cu], [0.0, fv, cv_], [0.0, 0.0, 1.0]],
                        dtype=np.float64)

    def dist(y):
        # EuRoC gives [k1, k2, p1, p2]; OpenCV wants [k1, k2, p1, p2, k3]
        d = list(y["distortion_coefficients"])
        while len(d) < 5:
            d.append(0.0)
        return np.array(d[:5], dtype=np.float64)

    M1, M2 = intr(y0), intr(y1)
    D1, D2 = dist(y0), dist(y1)

    # Stereo extrinsics: OpenCV wants the transform that takes a point from the
    # LEFT camera frame into the RIGHT camera frame.  EuRoC gives T_BS = cam->body.
    T_b_c0, T_b_c1 = y0["T_BS"], y1["T_BS"]
    T_c1_c0 = se3_inv(T_b_c1) @ T_b_c0
    R_stereo = T_c1_c0[:3, :3]
    T_stereo = T_c1_c0[:3, 3].reshape(3, 1)          # metres

    # IMU -> cam0 rotation.  imu0's T_BS is identity in EuRoC (body == IMU frame),
    # but compose it anyway so the maths stays correct for any dataset.
    T_b_imu = yi.get("T_BS", np.eye(4))
    R_cam0_imu = (se3_inv(T_b_c0) @ T_b_imu)[:3, :3]

    baseline_m = float(np.linalg.norm(T_stereo))

    # ---------------- frame list ------------------------------------------- #
    rows0 = list(read_csv_rows(need_file(os.path.join(cam0_dir, "data.csv"), "cam0 frame list")))
    rows1 = list(read_csv_rows(need_file(os.path.join(cam1_dir, "data.csv"), "cam1 frame list")))
    ts1 = {int(r[0]): r[1] for r in rows1}

    pairs = []                                        # (ts_ns, left_file, right_file)
    for r in rows0:
        t_ns = int(r[0])
        if t_ns in ts1:
            pairs.append((t_ns, r[1].strip(), ts1[t_ns].strip()))
    dropped = len(rows0) - len(pairs)

    if end < 0:
        end = len(pairs) - 1
    end = min(end, len(pairs) - 1)
    pairs = pairs[start:end + 1]
    if not pairs:
        raise ConvertError("no frames that both cameras recorded in the requested range")

    t0_ns = pairs[0][0]
    ext = os.path.splitext(pairs[0][1])[1].lstrip(".").lower() or "png"

    # ---------------- write ------------------------------------------------ #
    left_out = os.path.join(out_dir, "left")
    right_out = os.path.join(out_dir, "right")
    os.makedirs(left_out, exist_ok=True)
    os.makedirs(right_out, exist_ok=True)

    n_link = n_copy = 0
    with open_text(os.path.join(out_dir, "timestamps.csv"), newline="") as fts:
        fts.write("frame_id,timestamp_s,left_t,right_t\n")
        for i, (t_ns, lf, rf) in enumerate(pairs):
            t_s = (t_ns - t0_ns) / 1e9
            for src_dir, sub, name, dst_dir in (
                    (cam0_dir, "data", lf, left_out),
                    (cam1_dir, "data", rf, right_out)):
                src = os.path.join(src_dir, sub, name)
                dst = os.path.join(dst_dir, f"{i:06d}.{ext}")
                how = link_or_copy(src, dst, link_mode)
                if how == "link":
                    n_link += 1
                else:
                    n_copy += 1
            fts.write(f"{i},{t_s:.9f},{t_s:.9f},{t_s:.9f}\n")

    # ---------------- IMU -------------------------------------------------- #
    n_imu = 0
    n_imu_skipped = 0
    with open_text(os.path.join(out_dir, "imu.jsonl")) as fimu:
        for r in read_csv_rows(imu_csv):
            t_s = (int(r[0]) - t0_ns) / 1e9
            if t_s < -1.0:          # far ahead of the first frame: never consumed
                n_imu_skipped += 1
                continue
            # EuRoC column order: w_x w_y w_z  a_x a_y a_z
            gx, gy, gz = float(r[1]), float(r[2]), float(r[3])
            ax, ay, az = float(r[4]), float(r[5]), float(r[6])
            fimu.write(json.dumps({"t": round(t_s, 9),
                                   "ax": ax, "ay": ay, "az": az,
                                   "gx": gx, "gy": gy, "gz": gz}) + "\n")
            n_imu += 1

    # ---------------- calibration ------------------------------------------ #
    calib = {
        "source": "EuRoC MAV",
        "dataset": os.path.basename(os.path.normpath(euroc_root)),
        "camera_width": W,
        "camera_height": H,
        "camera_fps": int(y0.get("rate_hz", 20)),
        "imu_rate_hz": int(yi.get("rate_hz", 200)),
        "M1": M1.tolist(), "D1": D1.tolist(),
        "M2": M2.tolist(), "D2": D2.tolist(),
        # left -> right, metres (already in the units cv2.stereoRectify wants)
        "R_stereo": R_stereo.tolist(),
        "T_stereo": T_stereo.reshape(3).tolist(),
        "baseline_m": baseline_m,
        # rotates a vector from the IMU body frame into the UNRECTIFIED cam0
        # frame; the engine composes the rectification rotation on top.
        "R_cam0_imu": R_cam0_imu.tolist(),
        "T_body_cam0": T_b_c0.tolist(),
        "t0_ns": t0_ns,
        "num_frames": len(pairs),
        # Continuous-time IMU noise model straight from imu0/sensor.yaml, so the
        # back-end builds its preintegration covariances from the actual sensor.
        "imu_noise": {k: yi[k] for k in
                      ("gyroscope_noise_density", "gyroscope_random_walk",
                       "accelerometer_noise_density", "accelerometer_random_walk")
                      if k in yi},
    }
    with open_text(os.path.join(out_dir, "calib.json")) as f:
        json.dump(calib, f, indent=2)

    # ---------------- report ----------------------------------------------- #
    info = {"frames": len(pairs), "image size": f"{W} x {H} (.{ext})",
            "duration": "%.2f s" % ((pairs[-1][0] - t0_ns) / 1e9),
            "images": f"{n_link} hard-linked, {n_copy} copied",
            "unpaired frames": dropped, "IMU samples": n_imu,
            "baseline": "%.3f cm" % (baseline_m * 100)}
    try:
        import cv2
        _, _, P1, _, _, _, _ = cv2.stereoRectify(
            M1, D1, M2, D2, (W, H), R_stereo, T_stereo, alpha=0)
        info["rectified focal"] = "%.2f px" % float(P1[0, 0])
    except Exception:
        pass
    return info


def run(args):
    root = args.input
    need_dir(root, "input folder")
    find_mav0(root)
    prepare_output(args.output, args.force, [root])
    return convert(root, args.output, args.mode, args.start, args.end)
