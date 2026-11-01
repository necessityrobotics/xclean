# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""4Seasons -> xclean sequence folder.

Source per recording (unzipped):
    stereo_images_undistorted/<rec>/undistorted_images/cam{0,1}/<ns>.png   (stereo-rectified, pinhole, D = 0)
    imu_gnss/<rec>/imu.txt              t_ns gx gy gz ax ay az (rad/s, m/s^2)
    reference_poses/<rec>/GNSSPoses.txt t_ns,x,y,z,qx,qy,qz,qw,scale,quality
Global calibration/calibration/: undistorted_calib_0.txt (rectified pinhole), undistorted_calib_stereo.txt
(baseline), camchain.yaml (the raw equidistant cameras and T_cam_imu).

Two corrections are part of the conversion:
  * gt.csv is METRIC: GNSSPoses translations are in a scaled frame, metric = translation x the recording's scale
    column (one value per recording).
  * the camera-IMU extrinsic is the one of the RECTIFIED left camera.  The images are stereo-rectified, which
    rotates cam0 by R1 (about 1.5 degrees); camchain.yaml's T_cam_imu is for the raw cam0.  R1 is recomputed with
    cv2.fisheye.stereoRectify from camchain.yaml, and checked to reproduce the shipped rectified intrinsics, then
    T_cam0rect_imu = [R1 | 0] * T_cam0raw_imu is written to calib.json (T_body_cam0, R_cam0_imu).

IMU noise: automotive MEMS-class densities (hardware facts, not tuning).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from xclean.io.convert import (ConvertError, add_mode_argument, link_or_copy, need_dir, need_file, open_text,
                               prepare_output, read_kalibr_yaml)

HELP = "4Seasons recording (the folder holding calibration/ and the recording's unzipped parts, with --recording)"
INPUT_HELP = ("the 4Seasons folder that holds calibration/ and the recording's unzipped folders (then --recording "
              "is required), or the recording folder itself (recording_<date>_<time>/, next to calibration/)")
DESCRIPTION = """Convert one 4Seasons recording.

expected input (the dataset's zips unpacked into one folder <root>):
  <root>/calibration/calibration/undistorted_calib_0.txt, undistorted_calib_stereo.txt, camchain.yaml
  <root>/<rec>/stereo_images_undistorted/<rec>/undistorted_images/cam0/<ns>.png and cam1/<ns>.png
  <root>/<rec>/imu_gnss/<rec>/imu.txt
  <root>/<rec>/reference_poses/<rec>/GNSSPoses.txt
where <rec> is recording_<date>_<time>, and <root>/<rec>/<part>/ is where <rec>_<part>.zip unpacks.  The layout
without the <root>/<rec>/ level (<root>/stereo_images_undistorted/<rec>/...) is accepted too.

Left and right images are paired by nearest timestamp (at most 20 ms apart).  gt.csv is metric (translation x the
recording's GNSS scale).  calib.json carries the extrinsic of the rectified left camera.

example:
  xclean convert 4seasons /data/4seasons /data/xclean/4s_2020-12-22 --recording recording_2020-12-22_11-54-24
"""

RECTFIX_NOTE = ("T_cam0rect_imu = R1 * T_cam0raw_imu: camera-IMU extrinsic of the rectified left camera, R1 from "
                "cv2.fisheye.stereoRectify on camchain.yaml")


def add_arguments(p):
    p.add_argument("--recording", help="recording name, for example recording_2020-12-22_11-54-24 (the "
                                       "'recording_' prefix may be left out)")
    add_mode_argument(p)


def cal_dir(root):
    return os.path.join(root, "calibration", "calibration")


def load_calib(root):
    with open(os.path.join(cal_dir(root), "undistorted_calib_0.txt"), encoding="utf-8") as f:
        txt = f.read().split()
    fx, fy, cx, cy = (float(x) for x in txt[1:5])
    w, h = int(txt[9]), int(txt[10])
    st = np.loadtxt(os.path.join(cal_dir(root), "undistorted_calib_stereo.txt"))
    baseline = float(-st[0, 3])
    cc = read_kalibr_yaml(os.path.join(cal_dir(root), "camchain.yaml"))
    T_c_i = np.array(cc["cam0"]["T_cam_imu"], dtype=np.float64)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), (w, h), \
        baseline, np.linalg.inv(T_c_i)          # body(imu) <- cam0


def write_gt(gt_path, out_csv, t0):
    """GNSSPoses.txt -> gt.csv (t_ns - t0, x, y, z, qw, qx, qy, qz) in METRES.

    GNSSPoses translations are in a scaled frame; metric = translation x the recording's `scale` column (column 9;
    one value per recording, the 'GNSS scale' of Transformations.txt).  Rotation and time are untouched.
    Returns (n_rows, scale)."""
    n_gt, scales = 0, set()
    with open(gt_path, encoding="utf-8") as fi, open_text(out_csv) as fo:
        fo.write("#t_ns,x,y,z,qw,qx,qy,qz\n")
        for ln in fi:
            if ln.startswith("#"):
                continue
            p = ln.strip().split(",")
            if len(p) < 8:
                continue
            if len(p) < 9 or not p[8].strip():
                raise ConvertError("%s: a GNSSPoses row without the scale column - the ground truth would not be "
                                   "metric" % gt_path)
            s = float(p[8])
            scales.add(s)
            t_ns = int(p[0]) - t0
            x, y, z = (float(v) * s for v in p[1:4])
            qx, qy, qz, qw = (float(v) for v in p[4:8])
            fo.write(f"{t_ns},{x:.6f},{y:.6f},{z:.6f},"
                     f"{qw:.9f},{qx:.9f},{qy:.9f},{qz:.9f}\n")
            n_gt += 1
    if len(scales) != 1:
        raise ConvertError("%s: the scale column is not constant (%d values)" % (gt_path, len(scales)))
    return n_gt, scales.pop()


def rectification(root):
    """R1 of the shipped stereo rectification, recomputed from camchain.yaml (equidistant model, balance 0, zero
    disparity), checked against the shipped rectified intrinsics.  Returns (T_cam0rect_imu, T_cam0raw_imu,
    R1 angle in degrees, intrinsics error in px)."""
    import cv2
    cc = read_kalibr_yaml(os.path.join(cal_dir(root), "camchain.yaml"))

    def K(i):
        fx, fy, cx, cy = i
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    D0 = np.ascontiguousarray(np.array(cc["cam0"]["distortion_coeffs"], float).reshape(4, 1))
    D1 = np.ascontiguousarray(np.array(cc["cam1"]["distortion_coeffs"], float).reshape(4, 1))
    T = np.array(cc["T_cam1_cam0"], float)
    R1, _R2, P1, _P2, _Q = cv2.fisheye.stereoRectify(K(cc["cam0"]["intrinsics"]), D0, K(cc["cam1"]["intrinsics"]), D1,
                                                     (800, 400), np.ascontiguousarray(T[:3, :3]),
                                                     np.ascontiguousarray(T[:3, 3].reshape(3, 1)),
                                                     flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0, fov_scale=1.0)
    with open(os.path.join(cal_dir(root), "undistorted_calib_0.txt"), encoding="utf-8") as f:
        shipped = f.read().split()
    fx, cx, cy = float(shipped[1]), float(shipped[3]), float(shipped[4])
    err = max(abs(P1[0, 0] - fx), abs(P1[0, 2] - cx), abs(P1[1, 2] - cy))
    if not err < 0.05:
        raise ConvertError("the rectification recomputed from camchain.yaml does not reproduce the shipped "
                           "undistorted_calib_0.txt intrinsics (%.4f px off); the calibration files do not belong "
                           "together" % err)
    T_c0_imu = np.array(cc["cam0"]["T_cam_imu"], float)
    Rr = np.eye(4)
    Rr[:3, :3] = R1
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R1) - 1) / 2, -1, 1))))
    return Rr @ T_c0_imu, T_c0_imu, ang, err


def apply_extrinsic_fix(root, ds):
    """Rewrite calib.json's T_body_cam0 / R_cam0_imu to the rectified left camera (keys and order as in the
    evaluated folders)."""
    T_rect, T_raw, ang, err = rectification(root)
    p = os.path.join(ds, "calib.json")
    with open(p, encoding="utf-8") as f:
        c = json.load(f)
    if c.get("extrinsic_frame") == "rectified_cam0":
        return ang
    old = np.array(c["T_body_cam0"], float)
    if not np.allclose(np.linalg.inv(old), T_raw, atol=1e-9):
        raise ConvertError("%s: calib.json does not hold the raw camchain.yaml extrinsic" % ds)
    c["T_body_cam0"] = np.linalg.inv(T_rect).tolist()
    c["R_cam0_imu"] = T_rect[:3, :3].tolist()
    c["extrinsic_frame"] = "rectified_cam0"
    c["rectfix_R1_deg"] = ang
    c["rectfix_note"] = RECTFIX_NOTE
    tmp = p + ".tmp"
    with open_text(tmp) as f:
        json.dump(c, f, indent=2)
    os.replace(tmp, p)
    return ang


def find_parts(root, rec):
    """The recording's image, IMU and GNSS-pose paths (either supported layout)."""
    candidates = [
        os.path.join(root, rec, "stereo_images_undistorted", rec, "undistorted_images"),
        os.path.join(root, "stereo_images_undistorted", rec, "undistorted_images"),
    ]
    img_root = next((c for c in candidates if os.path.isdir(c)), None)
    if img_root is None:
        raise ConvertError(f"no undistorted_images/ for {rec}; looked in:\n  " + "\n  ".join(candidates))
    imu_candidates = [
        os.path.join(root, rec, "imu_gnss", rec, "imu.txt"),
        os.path.join(root, "imu_gnss", rec, "imu.txt")]
    imu_path = next((c for c in imu_candidates if os.path.isfile(c)), None)
    if imu_path is None:
        raise ConvertError(f"no imu.txt for {rec}; looked in:\n  " + "\n  ".join(imu_candidates))
    gt_candidates = [
        os.path.join(root, rec, "reference_poses", rec, "GNSSPoses.txt"),
        os.path.join(root, "reference_poses", rec, "GNSSPoses.txt")]
    gt_path = next((c for c in gt_candidates if os.path.isfile(c)), None)
    if gt_path is None:
        raise ConvertError(f"no GNSSPoses.txt for {rec}; looked in:\n  " + "\n  ".join(gt_candidates))
    for cam in ("cam0", "cam1"):
        need_dir(os.path.join(img_root, cam), f"{cam}/ images of {rec}")
    return img_root, imu_path, gt_path


def convert(root, out, rec, mode="auto"):
    M, (w, h), baseline, T_b_c = load_calib(root)
    img_root, imu_path, gt_path = find_parts(root, rec)
    os.makedirs(os.path.join(out, "left"), exist_ok=True)
    os.makedirs(os.path.join(out, "right"), exist_ok=True)

    def stamps(cam):
        d = os.path.join(img_root, cam)
        return sorted(int(f[:-4]) for f in os.listdir(d)
                      if re.match(r"\d+\.png$", f))
    lt, rt = stamps("cam0"), np.array(stamps("cam1"), dtype=np.int64)
    pairs = []
    for t in lt:
        j = np.searchsorted(rt, t)
        best, bdt = None, 1 << 62
        for k in (j - 1, j):
            if 0 <= k < len(rt) and abs(int(rt[k]) - t) < bdt:
                best, bdt = int(rt[k]), abs(int(rt[k]) - t)
        if best is not None and bdt <= 20_000_000:
            pairs.append((t, best))
    if not pairs:
        raise ConvertError(f"no left/right image pairs within 20 ms in {img_root}")
    t0 = pairs[0][0]

    n_link = n_copy = 0
    with open_text(os.path.join(out, "timestamps.csv")) as f:
        f.write("frame_id,timestamp_s,left_t,right_t\n")
        for i, (tl, tr) in enumerate(pairs):
            f.write(f"{i},{(tl-t0)/1e9:.9f},{(tl-t0)/1e9:.9f},"
                    f"{(tr-t0)/1e9:.9f}\n")
            for sub, cam, ts in (("left", "cam0", tl), ("right", "cam1", tr)):
                dst = os.path.join(out, sub, f"{i:06d}.png")
                how = link_or_copy(os.path.join(img_root, cam, f"{ts}.png"), dst, mode)
                n_link += how == "link"
                n_copy += how == "copy"

    n_imu = 0
    with open(imu_path, encoding="utf-8") as fi, \
            open_text(os.path.join(out, "imu.jsonl")) as fo:
        for ln in fi:
            p = ln.split()
            if len(p) < 7 or p[0].startswith("#"):
                continue
            t = (int(p[0]) - t0) / 1e9
            gx, gy, gz, ax, ay, az = (float(x) for x in p[1:7])
            fo.write(json.dumps({"t": round(t, 9), "ax": ax, "ay": ay,
                                 "az": az, "gx": gx, "gy": gy,
                                 "gz": gz}) + "\n")
            n_imu += 1

    n_gt, gt_scale = write_gt(gt_path, os.path.join(out, "gt.csv"), t0)

    calib = {
        "source": "4Seasons", "dataset": f"fourseasons_{rec}",
        "camera_width": w, "camera_height": h,
        "camera_fps": 30, "imu_rate_hz": 2000,
        "M1": M.tolist(), "D1": [0.0] * 5,
        "M2": M.tolist(), "D2": [0.0] * 5,
        "R_stereo": np.eye(3).tolist(),
        "T_stereo": [-baseline, 0.0, 0.0],
        "baseline_m": baseline,
        "R_cam0_imu": np.linalg.inv(T_b_c)[:3, :3].tolist(),
        "T_body_cam0": T_b_c.tolist(),
        "t0_ns": 0,
        "num_frames": len(pairs),
        "imu_noise": {
            "gyroscope_noise_density": 2.0e-4,
            "gyroscope_random_walk": 1.0e-5,
            "accelerometer_noise_density": 1.5e-3,
            "accelerometer_random_walk": 1.0e-4,
        },
    }
    with open_text(os.path.join(out, "calib.json")) as f:
        json.dump(calib, f, indent=2)
    ang = apply_extrinsic_fix(root, out)
    return {"frames": len(pairs), "image size": f"{w} x {h}", "images": f"{n_link} hard-linked, {n_copy} copied",
            "baseline": "%.1f cm" % (baseline * 100), "IMU samples": n_imu,
            "ground truth": "%d poses (metric, GNSS scale %.6f)" % (n_gt, gt_scale),
            "extrinsic": "rectified left camera (R1 = %.3f deg)" % ang}


def resolve(inp, recording):
    """(root, rec) from either input form."""
    base = os.path.basename(os.path.normpath(inp))
    if recording:
        rec = recording if recording.startswith("recording_") else "recording_" + recording
        if base == rec and not os.path.isdir(cal_dir(inp)):
            root = os.path.dirname(os.path.abspath(inp))
        else:
            root = inp
    elif base.startswith("recording_"):
        rec, root = base, os.path.dirname(os.path.abspath(inp))
    else:
        raise ConvertError("say which recording to convert with --recording (e.g. --recording "
                           "recording_2020-12-22_11-54-24), or give the recording folder itself")
    for name in ("undistorted_calib_0.txt", "undistorted_calib_stereo.txt", "camchain.yaml"):
        need_file(os.path.join(cal_dir(root), name), f"4Seasons calibration file {name} (from calibration.zip)")
    find_parts(root, rec)
    rectification(root)                     # a calibration that does not hang together fails here, before writing
    return root, rec


def run(args):
    need_dir(args.input, "input folder")
    root, rec = resolve(args.input, args.recording)
    prepare_output(args.output, args.force, [args.input])
    return convert(root, args.output, rec, args.mode)
