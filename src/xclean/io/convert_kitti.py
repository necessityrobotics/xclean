# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""KITTI odometry (grayscale stereo) -> xclean sequence folder.

    left/NNNNNN.png   right/NNNNNN.png    image_0 / image_1 of the sequence
    timestamps.csv    frame_id,timestamp_s,left_t,right_t   (times.txt)
    calib.json        KITTI images are ALREADY rectified, so M1 == M2 == K(P0), D = 0, R_stereo = I,
                      T_stereo = [-B, 0, 0]; the left camera is the body: T_body_cam0 = identity
    gt.csv            t_ns, x, y, z, qw, qx, qy, qz  (the pose of cam0 from poses/NN.txt; only 00-10 have one)

KITTI odometry has no IMU: no imu.jsonl is written and the engine runs stereo-only.
"""
from __future__ import annotations

import json
import os

import numpy as np

from xclean.io.convert import (ConvertError, add_mode_argument, link_or_copy, need_dir, need_file, open_text,
                               prepare_output)

HELP = "KITTI odometry, grayscale stereo (the dataset/ folder with --seq, or one sequences/NN folder)"
INPUT_HELP = ("the KITTI odometry dataset/ folder (holds sequences/ and poses/; then --seq is required), "
              "or one sequence folder sequences/NN")
DESCRIPTION = """Convert one KITTI odometry sequence (data_odometry_gray + data_odometry_calib, and
data_odometry_poses for ground truth, all unpacked into the same dataset/ folder).

expected input:
  <dataset>/sequences/NN/image_0/000000.png ...   left camera
  <dataset>/sequences/NN/image_1/000000.png ...   right camera
  <dataset>/sequences/NN/calib.txt, times.txt
  <dataset>/poses/NN.txt                          optional; only sequences 00-10 have ground truth

KITTI has no IMU, so no imu.jsonl is written.  gt.csv holds the left-camera poses with times from times.txt.

examples:
  xclean convert kitti /data/kitti/dataset /data/xclean/kitti_04 --seq 04
  xclean convert kitti /data/kitti/dataset/sequences/04 /data/xclean/kitti_04
"""


def add_arguments(p):
    p.add_argument("--seq", help="sequence number, for example 04 (required when the input is the dataset/ folder)")
    add_mode_argument(p)


def rot_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12))
    q = np.empty(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def resolve(inp, seq):
    """(sequence dir, poses file, sequence number) from either input form."""
    if os.path.isdir(os.path.join(inp, "sequences")):
        if not seq:
            raise ConvertError("the input is a KITTI dataset/ folder: say which sequence with --seq (e.g. --seq 04)")
        sdir = os.path.join(inp, "sequences", seq)
        poses = os.path.join(inp, "poses", f"{seq}.txt")
    elif os.path.isfile(os.path.join(inp, "calib.txt")) or os.path.isdir(os.path.join(inp, "image_0")):
        sdir = inp
        seq = seq or os.path.basename(os.path.normpath(inp))
        poses = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sdir))), "poses", f"{seq}.txt")
    else:
        raise ConvertError(f"{inp} is neither a KITTI odometry dataset/ folder (with sequences/) nor a sequence "
                           f"folder (with calib.txt and image_0/)")
    need_dir(sdir, f"sequence {seq}")
    for name in ("calib.txt", "times.txt"):
        need_file(os.path.join(sdir, name), f"{name} of sequence {seq} (from data_odometry_calib)")
    for cam in ("image_0", "image_1"):
        need_file(os.path.join(sdir, cam, "000000.png"),
                  f"{cam}/000000.png of sequence {seq} (from data_odometry_gray)")
    return sdir, poses, seq


def convert(sdir, poses_path, out, seq, mode="auto"):
    os.makedirs(os.path.join(out, "left"), exist_ok=True)
    os.makedirs(os.path.join(out, "right"), exist_ok=True)

    # ---- calibration --------------------------------------------------
    P = {}
    with open(os.path.join(sdir, "calib.txt"), encoding="utf-8") as f:
        for ln in f:
            if ":" in ln:
                k, v = ln.split(":", 1)
                P[k.strip()] = np.array([float(x) for x in v.split()]
                                        ).reshape(3, 4)
    if "P0" not in P or "P1" not in P:
        raise ConvertError(f"{sdir}/calib.txt has no P0 / P1 projection matrix")
    K = P["P0"][:, :3]
    baseline = float(-P["P1"][0, 3] / P["P1"][0, 0])

    import cv2
    from xclean.io.imageio import imread
    img0 = imread(os.path.join(sdir, "image_0", "000000.png"),
                  cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        raise ConvertError(f"cannot read {sdir}/image_0/000000.png")
    h, w = img0.shape

    times = np.loadtxt(os.path.join(sdir, "times.txt"))
    n = len(times)

    # ---- frames + timestamps -----------------------------------------
    n_link = n_copy = 0
    with open_text(os.path.join(out, "timestamps.csv")) as f:
        f.write("frame_id,timestamp_s,left_t,right_t\n")
        for i in range(n):
            t = float(times[i])
            f.write(f"{i},{t:.9f},{t:.9f},{t:.9f}\n")
            nm = f"{i:06d}.png"
            for cam, sub in (("image_0", "left"), ("image_1", "right")):
                how = link_or_copy(os.path.join(sdir, cam, nm), os.path.join(out, sub, nm), mode)
                n_link += how == "link"
                n_copy += how == "copy"

    # ---- ground truth -------------------------------------------------
    n_gt = 0
    if os.path.isfile(poses_path):
        poses = np.loadtxt(poses_path).reshape(-1, 3, 4)
        with open_text(os.path.join(out, "gt.csv")) as f:
            f.write("#t_ns,x,y,z,qw,qx,qy,qz\n")
            for i in range(min(n, len(poses))):
                T = poses[i]
                q = rot_to_quat_wxyz(T[:, :3])
                f.write(f"{int(times[i]*1e9)},"
                        f"{T[0,3]:.6f},{T[1,3]:.6f},{T[2,3]:.6f},"
                        f"{q[0]:.9f},{q[1]:.9f},{q[2]:.9f},{q[3]:.9f}\n")
                n_gt += 1

    # ---- calib.json ---------------------------------------------------
    calib = {
        "source": "KITTI odometry",
        "dataset": f"kitti_{seq}",
        "camera_width": int(w),
        "camera_height": int(h),
        "camera_fps": 10,
        "imu_rate_hz": 0,
        "M1": K.tolist(), "D1": [0.0] * 5,
        "M2": K.tolist(), "D2": [0.0] * 5,
        "R_stereo": np.eye(3).tolist(),
        "T_stereo": [-baseline, 0.0, 0.0],
        "baseline_m": baseline,
        "T_body_cam0": np.eye(4).tolist(),
        "t0_ns": 0,
        "num_frames": int(n),
    }
    with open_text(os.path.join(out, "calib.json")) as f:
        json.dump(calib, f, indent=2)
    return {"frames": n, "image size": f"{w} x {h}", "images": f"{n_link} hard-linked, {n_copy} copied",
            "baseline": "%.1f cm" % (baseline * 100), "focal x baseline": "%.0f px m" % (K[0, 0] * baseline),
            "ground truth": (f"{n_gt} poses" if n_gt else "none (only sequences 00-10 have ground truth)")}


def run(args):
    need_dir(args.input, "input folder")
    sdir, poses, seq = resolve(args.input, args.seq)
    prepare_output(args.output, args.force, [args.input])
    return convert(sdir, poses, args.output, seq, args.mode)
