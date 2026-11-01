# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Rosario agricultural dataset, version 2 (ROS bag) -> xclean sequence folder.

Reads, with the pure-Python `rosbags` package (no ROS install):
    /realsense/infra1/image_rect_raw   mono8 RECTIFIED  -> left/
    /realsense/infra2/image_rect_raw                    -> right/
    /realsense/infra2/camera_info      P matrix -> K, baseline
    /realsense/imu                     sensor_msgs/Imu (rad/s, m/s^2)
    pose-GT bag (optional): the Odometry / PoseStamped topic with the most messages -> gt.csv

Images are already rectified (D = 0): calib.json declares the ideal pinhole rig.  T_body_cam0 = identity (the
RealSense IMU frame is aligned with the infra1 optical frame to within mounting tolerance; the engine's gravity
bootstrap absorbs the residual).  IMU noise: RealSense BMI055-class datasheet values (hardware facts).
Images are written as PNG with Pillow.  The pixels are exactly the bag's; the compressed PNG bytes depend on the zlib
that Pillow is built with (Pillow 12.0.0 from Anaconda's default channel, built with classic zlib 1.3.1, writes the
evaluated folder's files byte for byte; the PyPI wheel of the same version uses zlib-ng and writes different bytes for
the same pixels).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from xclean.io.convert import ConvertError, need_file, open_text, prepare_output

HELP = "Rosario dataset v2, ROS bag (needs: pip install rosbags pillow)"
INPUT_HELP = "the sequence's .bag file (for example 2023-12-22-13-14-16.compressed.bag)"
DESCRIPTION = """Convert one sequence of the Rosario agricultural dataset, version 2 (ROS bag release).

This format needs two extra Python packages:  pip install rosbags pillow

expected input:
  <input>   the sequence bag, with /realsense/infra1/image_rect_raw, /realsense/infra2/image_rect_raw,
            /realsense/infra2/camera_info and /realsense/imu
  --pgt     optional: the sequence's pose ground-truth bag (<name>_pgt.bag); without it no gt.csv is written

Left and right images are paired by nearest header stamp (at most 20 ms apart) and written as PNG (same pixels
with any Pillow; the PNG bytes can differ between Pillow builds).  calib.json "dataset" is the output folder's name.

example:
  xclean convert rosario-v2 /data/rosario_v2/2023-12-22-13-14-16.compressed.bag /data/xclean/rosariov2_01 \\
      --pgt /data/rosario_v2/2023-12-22-13-14-16_pgt.bag
"""

TOPIC_LEFT = "/realsense/infra1/image_rect_raw"
TOPIC_RIGHT = "/realsense/infra2/image_rect_raw"
TOPIC_INFO = "/realsense/infra2/camera_info"
TOPIC_IMU = "/realsense/imu"


def add_arguments(p):
    p.add_argument("--pgt", default=None, help="the pose ground-truth bag of the same sequence (optional)")


def _imports():
    try:
        from PIL import Image
        from rosbags.highlevel import AnyReader
    except ImportError as e:
        raise ConvertError("the rosario-v2 format reads ROS bags and needs two extra Python packages: "
                           "pip install rosbags pillow  (%s)" % e)
    return Image, AnyReader


def extract(bag, pgt, out):
    Image, AnyReader = _imports()
    os.makedirs(os.path.join(out, "left"), exist_ok=True)
    os.makedirs(os.path.join(out, "right"), exist_ok=True)
    lefts, rights = [], []       # (t, index) after write
    K = None
    baseline = 0.0
    wh = (0, 0)
    imu_rows = []
    with AnyReader([Path(bag)]) as r:
        cons = {c.topic: c for c in r.connections}
        missing = [t for t in (TOPIC_LEFT, TOPIC_RIGHT, TOPIC_INFO, TOPIC_IMU) if t not in cons]
        if missing:
            raise ConvertError("%s has no topic %s" % (bag, ", ".join(missing)))
        ci = cons.get(TOPIC_INFO)
        for con, ts, raw in r.messages(connections=[ci]):
            m = r.deserialize(raw, con.msgtype)
            P = np.array(m.P).reshape(3, 4)
            K = np.array([[P[0, 0], 0, P[0, 2]],
                          [0, P[1, 1], P[1, 2]], [0, 0, 1.0]])
            baseline = float(-P[0, 3] / P[0, 0])
            break
        if K is None:
            raise ConvertError("%s: %s has no message" % (bag, TOPIC_INFO))
        # stream both image topics; buffer only timestamps, write pngs now
        li = ri = 0
        sel = [cons[TOPIC_LEFT], cons[TOPIC_RIGHT]]
        for con, ts, raw in r.messages(connections=sel):
            m = r.deserialize(raw, con.msgtype)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
            wh = (m.width, m.height)
            if con.topic.endswith("infra1/image_rect_raw"):
                Image.fromarray(img).save(
                    os.path.join(out, "left", f"{li:06d}.png"))
                lefts.append((t, li)); li += 1
            else:
                Image.fromarray(img).save(
                    os.path.join(out, "right", f"{ri:06d}.png"))
                rights.append((t, ri)); ri += 1
        for con, ts, raw in r.messages(
                connections=[cons[TOPIC_IMU]]):
            m = r.deserialize(raw, con.msgtype)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            imu_rows.append((t, m.linear_acceleration.x,
                             m.linear_acceleration.y,
                             m.linear_acceleration.z,
                             m.angular_velocity.x, m.angular_velocity.y,
                             m.angular_velocity.z))
    # ---- pair left/right by nearest timestamp ------------------------
    lefts.sort(); rights.sort()
    rt = np.array([t for t, _ in rights])
    pairs = []
    for t, il in lefts:
        j = np.searchsorted(rt, t)
        best, bdt = None, 1e9
        for k in (j - 1, j):
            if 0 <= k < len(rt) and abs(rt[k] - t) < bdt:
                best, bdt = k, abs(rt[k] - t)
        if best is not None and bdt <= 0.02:
            pairs.append((t, il, rights[best][1], rt[best]))
    if not pairs:
        raise ConvertError("%s: no left/right image pairs within 20 ms" % bag)
    t0 = pairs[0][0]
    # remap indices so frame i <-> left/right i (rename right files)
    os.makedirs(os.path.join(out, "right_tmp"), exist_ok=True)
    with open_text(os.path.join(out, "timestamps.csv")) as f:
        f.write("frame_id,timestamp_s,left_t,right_t\n")
        for i, (tl, il, ir, tr) in enumerate(pairs):
            f.write(f"{i},{tl - t0:.9f},{tl - t0:.9f},{tr - t0:.9f}\n")
            if il != i:
                os.replace(os.path.join(out, "left", f"{il:06d}.png"),
                           os.path.join(out, "left", f"{i:06d}_.png"))
            os.replace(os.path.join(out, "right", f"{ir:06d}.png"),
                       os.path.join(out, "right_tmp", f"{i:06d}.png"))
    # finalize renames
    for i in range(len(pairs)):
        p = os.path.join(out, "left", f"{i:06d}_.png")
        if os.path.exists(p):
            os.replace(p, os.path.join(out, "left", f"{i:06d}.png"))
    for fn in os.listdir(os.path.join(out, "right_tmp")):
        os.replace(os.path.join(out, "right_tmp", fn),
                   os.path.join(out, "right", fn))
    os.rmdir(os.path.join(out, "right_tmp"))
    # drop unpaired leftovers beyond the pair count
    for sub in ("left", "right"):
        for fn in os.listdir(os.path.join(out, sub)):
            try:
                if int(fn[:6]) >= len(pairs):
                    os.remove(os.path.join(out, sub, fn))
            except ValueError:
                pass

    with open_text(os.path.join(out, "imu.jsonl")) as f:
        for t, ax, ay, az, gx, gy, gz in sorted(imu_rows):
            f.write(json.dumps({"t": round(t - t0, 9),
                                "ax": ax, "ay": ay, "az": az,
                                "gx": gx, "gy": gy, "gz": gz}) + "\n")

    # ---- GT from the pgt bag -----------------------------------------
    n_gt = 0
    gt_topic = None
    if pgt and os.path.isfile(pgt):
        with AnyReader([Path(pgt)]) as r, \
                open_text(os.path.join(out, "gt.csv")) as f:
            f.write("#t_ns,x,y,z,qw,qx,qy,qz\n")
            cands = [c for c in r.connections
                     if "Odometry" in c.msgtype or "PoseStamped" in c.msgtype
                     or "PoseWithCovarianceStamped" in c.msgtype]
            cand = max(cands, key=lambda c: c.msgcount) if cands else None
            if cand:
                gt_topic = "%s (%s, %d messages)" % (cand.topic, cand.msgtype, cand.msgcount)
                for con, ts, raw in r.messages(connections=[cand]):
                    m = r.deserialize(raw, con.msgtype)
                    pose = getattr(m, "pose", m)
                    pose = getattr(pose, "pose", pose)
                    p, q = pose.position, pose.orientation
                    t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                    f.write(f"{int((t - t0) * 1e9)},"
                            f"{p.x:.6f},{p.y:.6f},{p.z:.6f},"
                            f"{q.w:.9f},{q.x:.9f},{q.y:.9f},{q.z:.9f}\n")
                    n_gt += 1

    calib = {
        "source": "Rosario v2", "dataset": os.path.basename(os.path.normpath(out)),
        "camera_width": wh[0], "camera_height": wh[1],
        "camera_fps": 15, "imu_rate_hz": 200,
        "M1": K.tolist(), "D1": [0.0] * 5,
        "M2": K.tolist(), "D2": [0.0] * 5,
        "R_stereo": np.eye(3).tolist(),
        "T_stereo": [-baseline, 0.0, 0.0],
        "baseline_m": baseline,
        "T_body_cam0": np.eye(4).tolist(),
        "t0_ns": 0,
        "num_frames": len(pairs),
        "imu_noise": {
            "gyroscope_noise_density": 1.4e-4,
            "gyroscope_random_walk": 1.0e-5,
            "accelerometer_noise_density": 1.9e-3,
            "accelerometer_random_walk": 1.0e-4,
        },
    }
    with open_text(os.path.join(out, "calib.json")) as f:
        json.dump(calib, f, indent=2)
    return {"frames": len(pairs), "image size": f"{wh[0]} x {wh[1]}", "baseline": "%.2f cm" % (baseline * 100),
            "IMU samples": len(imu_rows),
            "ground truth": ("%d poses from %s" % (n_gt, gt_topic)) if gt_topic else "none (no --pgt bag)"}


def run(args):
    need_file(args.input, "input bag")
    if args.pgt:
        need_file(args.pgt, "pose ground-truth bag (--pgt)")
    _imports()
    prepare_output(args.output, args.force, [args.input])
    return extract(args.input, args.pgt, args.output)
