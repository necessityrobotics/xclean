# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""`xclean convert` (xclean/io/convert*.py) on tiny synthetic inputs, one per dataset format: each converts, writes a
folder that the sequence audit and the dataset reader accept, and wrong inputs give a plain error (exit code 1, no
traceback).  No dataset download is needed; rosario-v2 runs only when `rosbags` and `Pillow` are installed."""
import json
import math
import os
import zipfile

import numpy as np
import pytest

from xclean.eval.dataset_audit import audit
from xclean.eval.integrity import IntegrityGate
from xclean.io.convert import main, read_kalibr_yaml
from xclean.io.dataset import Dataset

G = 9.80665
RNG = np.random.RandomState(7)


# ------------------------------------------------------------------------------------------------ helpers
def _img(w, h, k=0):
    """A small-to-compress test image: flat grey with a few bright squares that move with k."""
    import cv2
    a = np.full((h, w), 90, np.uint8)
    for j in range(6):
        x = (17 * j + 3 * k) % max(1, w - 8)
        y = (11 * j + 2 * k) % max(1, h - 8)
        a[y:y + 6, x:x + 6] = 200
    return a


def _png(path, w, h, k=0):
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert cv2.imwrite(str(path), _img(w, h, k))


def _accel(n):
    """Static accelerometer: about g on z with a little noise, and one bump so that the maximum is unique (a
    repeated maximum reads as rail clipping to the audit)."""
    a = np.zeros((n, 3))
    a[:, 2] = G
    a += RNG.standard_normal((n, 3)) * 1e-3
    a[n // 2, 2] += 0.5
    return a


def _convert(capsys, *argv):
    rc = main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return rc, out, err


def _check_folder(out, n_frames, imu=True, gt=True):
    """The written folder passes the audit and the integrity gate, and the dataset reader opens it."""
    res = audit(str(out))
    assert res is not None, "not recognised as a sequence folder"
    flags, info = res
    assert flags == [], (flags, info)
    assert IntegrityGate(use_cache=False).check(str(out)).accepted
    ds = Dataset(str(out), gate=True)
    assert len(ds) == n_frames
    left, right = ds.images(0)
    assert left.shape == right.shape == (ds.calib["camera_height"], ds.calib["camera_width"])
    batch = ds.imu()
    assert (len(batch.t) > 10) == imu
    assert os.path.isfile(os.path.join(out, "gt.csv")) == gt
    calib = json.loads(open(os.path.join(out, "calib.json"), encoding="utf-8").read())
    assert calib["num_frames"] == n_frames
    for name in ("calib.json", "imu.jsonl", "gt.csv"):
        p = os.path.join(out, name)
        if os.path.isfile(p):
            b = open(p, "rb").read()
            assert b"\r\n" in b and b.count(b"\n") == b.count(b"\r\n"), name + " should have CRLF line endings"
    return calib


def _rot(rx, ry, rz):
    import cv2
    return cv2.Rodrigues(np.array([rx, ry, rz], float))[0]


# ------------------------------------------------------------------------------------------------ EuRoC
def make_euroc(root, n=30, w=64, h=48):
    mav0 = root / "MH_99_test" / "mav0"
    t0 = 1_403_636_579_763_555_584
    cam_t = [t0 + i * 50_000_000 for i in range(n)]
    for cam, tx in (("cam0", 0.0), ("cam1", 0.11)):
        d = mav0 / cam
        (d / "data").mkdir(parents=True)
        (d / "sensor.yaml").write_text(
            "sensor_type: camera\ncomment: test\nT_BS:\n  cols: 4\n  rows: 4\n"
            "  data: [1.0, 0.0, 0.0, %r,\n         0.0, 1.0, 0.0, 0.0,\n         0.0, 0.0, 1.0, 0.0,\n"
            "         0.0, 0.0, 0.0, 1.0]\nrate_hz: 20\nresolution: [%d, %d]\ncamera_model: pinhole\n"
            "intrinsics: [40.0, 40.5, 32.0, 24.0]\ndistortion_model: radial-tangential\n"
            "distortion_coefficients: [-0.28, 0.07, 0.0002, 2.0e-05]\n" % (tx, w, h))
        with open(d / "data.csv", "w") as f:
            f.write("#timestamp [ns],filename\n")
            for i, t in enumerate(cam_t):
                if cam == "cam1" and i == n - 1:
                    continue                         # one frame only cam0 recorded: not a pair
                f.write("%d,%d.png\n" % (t, t))
                _png(d / "data" / ("%d.png" % t), w, h, i)
    imu = mav0 / "imu0"
    imu.mkdir()
    (imu / "sensor.yaml").write_text(
        "sensor_type: imu\nT_BS:\n  cols: 4\n  rows: 4\n  data: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,\n"
        "         0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]\nrate_hz: 200\n"
        "gyroscope_noise_density: 1.6968e-04\ngyroscope_random_walk: 1.9393e-05\n"
        "accelerometer_noise_density: 2.0000e-3\naccelerometer_random_walk: 3.0000e-3\n")
    ti = [t0 - 2_000_000_000] + [t0 - 500_000_000 + k * 5_000_000 for k in range(int((n * 0.05 + 1.0) / 0.005))]
    acc = _accel(len(ti))
    with open(imu / "data.csv", "w") as f:
        f.write("#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n")
        for k, t in enumerate(ti):
            f.write("%d,0.001,-0.002,0.003,%r,%r,%r\n" % (t, *map(float, acc[k])))
    return mav0.parent, cam_t


def test_euroc(tmp_path, capsys):
    src, cam_t = make_euroc(tmp_path / "in")
    out = tmp_path / "out"
    rc, so, se = _convert(capsys, "euroc", src, out)
    assert rc == 0, se
    assert "frames           29" in so
    calib = _check_folder(out, 29, gt=False)
    assert calib["source"] == "EuRoC MAV" and calib["dataset"] == "MH_99_test"
    assert calib["t0_ns"] == cam_t[0] and abs(calib["baseline_m"] - 0.11) < 1e-12
    assert calib["T_stereo"] == [-0.11, 0.0, 0.0]
    assert set(calib["imu_noise"]) == {"gyroscope_noise_density", "gyroscope_random_walk",
                                      "accelerometer_noise_density", "accelerometer_random_walk"}
    ts = open(out / "timestamps.csv", "rb").read()
    assert b"\r" not in ts                            # this file keeps LF endings, as in the evaluated folders
    assert ts.splitlines()[:3] == [b"frame_id,timestamp_s,left_t,right_t", b"0,0.000000000,0.000000000,0.000000000",
                                   b"1,0.050000000,0.050000000,0.050000000"]
    imu = [json.loads(x) for x in open(out / "imu.jsonl", encoding="utf-8")]
    assert imu[0]["t"] == -0.5 and len(imu) == int((30 * 0.05 + 1.0) / 0.005)   # the sample 2 s early is dropped
    src_img = src / "mav0" / "cam0" / "data" / ("%d.png" % cam_t[3])
    assert open(out / "left" / "000003.png", "rb").read() == open(src_img, "rb").read()


def test_euroc_image_modes(tmp_path, capsys):
    src, cam_t = make_euroc(tmp_path / "in", n=4)
    src_img = src / "mav0" / "cam1" / "data" / ("%d.png" % cam_t[1])
    rc, _, se = _convert(capsys, "euroc", src, tmp_path / "copy", "--mode", "copy")
    assert rc == 0, se
    assert not os.path.samefile(tmp_path / "copy" / "right" / "000001.png", src_img)
    rc, _, se = _convert(capsys, "euroc", src, tmp_path / "link", "--mode", "link")
    assert rc == 0, se
    assert os.path.samefile(tmp_path / "link" / "right" / "000001.png", src_img)
    a = {n: open(tmp_path / "copy" / n, "rb").read() for n in ("calib.json", "timestamps.csv", "imu.jsonl")}
    b = {n: open(tmp_path / "link" / n, "rb").read() for n in ("calib.json", "timestamps.csv", "imu.jsonl")}
    assert a == b


# ------------------------------------------------------------------------------------------------ KITTI
def make_kitti(root, seq="04", n=30, w=80, h=40, poses=True):
    s = root / "dataset" / "sequences" / seq
    fx, cx, cy, B = 70.0, 40.0, 20.0, 0.54
    p0 = [fx, 0, cx, 0, 0, fx, cy, 0, 0, 0, 1, 0]
    p1 = [fx, 0, cx, -fx * B, 0, fx, cy, 0, 0, 0, 1, 0]
    s.mkdir(parents=True)
    (s / "calib.txt").write_text("".join("%s: %s\n" % (k, " ".join("%e" % v for v in p))
                                         for k, p in (("P0", p0), ("P1", p1), ("P2", p0), ("P3", p0))))
    (s / "times.txt").write_text("".join("%e\n" % (0.1 * i) for i in range(n)))
    for cam in ("image_0", "image_1"):
        for i in range(n):
            _png(s / cam / ("%06d.png" % i), w, h, i)
    if poses:
        (root / "dataset" / "poses").mkdir()
        with open(root / "dataset" / "poses" / ("%s.txt" % seq), "w") as f:
            for i in range(n):
                f.write("1 0 0 %e 0 1 0 0 0 0 1 %e\n" % (0.01 * i, 0.5 * i))
    return root / "dataset"


def test_kitti_both_input_forms(tmp_path, capsys):
    ds = make_kitti(tmp_path / "in")
    rc, so, se = _convert(capsys, "kitti", ds, tmp_path / "a", "--seq", "04")
    assert rc == 0, se
    calib = _check_folder(tmp_path / "a", 30, imu=False)
    assert calib["dataset"] == "kitti_04" and calib["imu_rate_hz"] == 0
    assert abs(calib["baseline_m"] - 0.54) < 1e-9 and calib["T_body_cam0"] == np.eye(4).tolist()
    gt = open(tmp_path / "a" / "gt.csv", encoding="utf-8").read().splitlines()
    assert gt[0] == "#t_ns,x,y,z,qw,qx,qy,qz" and gt[2] == "100000000,0.010000,0.000000,0.500000," \
                                                         "1.000000000,0.000000000,0.000000000,0.000000000"
    rc, _, se = _convert(capsys, "kitti", ds / "sequences" / "04", tmp_path / "b")
    assert rc == 0, se
    for n in ("calib.json", "timestamps.csv", "gt.csv"):
        assert open(tmp_path / "a" / n, "rb").read() == open(tmp_path / "b" / n, "rb").read()


def test_kitti_without_poses_and_errors(tmp_path, capsys):
    ds = make_kitti(tmp_path / "in", seq="11", poses=False)
    rc, so, se = _convert(capsys, "kitti", ds, tmp_path / "a", "--seq", "11")
    assert rc == 0, se
    assert not os.path.exists(tmp_path / "a" / "gt.csv") and "only sequences 00-10" in so
    rc, _, se = _convert(capsys, "kitti", ds, tmp_path / "b")
    assert rc == 1 and "--seq" in se and "Traceback" not in se
    assert not os.path.exists(tmp_path / "b")         # nothing is written for a wrong input
    rc, _, se = _convert(capsys, "kitti", ds, tmp_path / "c", "--seq", "05")
    assert rc == 1 and "sequence 05 not found" in se


# ------------------------------------------------------------------------------------------------ 4Seasons
def make_4seasons(root, rec="recording_2030-01-02_03-04-05", n=30, consistent=True):
    import cv2
    cal = root / "calibration" / "calibration"
    cal.mkdir(parents=True)
    K0 = np.array([[300.0, 0, 400.0], [0, 300.5, 200.0], [0, 0, 1]])
    K1 = np.array([[301.0, 0, 405.0], [0, 301.5, 198.0], [0, 0, 1]])
    D0 = np.array([-0.03, -0.005, 0.02, -0.015])
    D1 = np.array([-0.035, 0.01, -0.016, 0.009])
    T10 = np.eye(4)
    T10[:3, :3] = _rot(0.001, -0.013, 0.0096)
    T10[:3, 3] = [-0.3, 0.0003, 0.0034]
    T_c_i = np.eye(4)
    T_c_i[:3, :3] = _rot(math.pi / 2, 0.01, -0.02) @ _rot(0, 0, math.pi)
    T_c_i[:3, 3] = [0.175, 0.0037, -0.058]

    def row(v):
        return "[" + ", ".join(repr(float(x)) for x in v) + "]"
    lines = []
    for cam, K, D in (("cam0", K0, D0), ("cam1", K1, D1)):
        lines += ["%s:" % cam, "  cam_overlaps: [%d]" % (cam == "cam0"), "  T_cam_imu:"]
        lines += ["    - " + row(r) for r in T_c_i]
        lines += ["  intrinsics: " + row([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]), "  resolution: [800, 400]",
                  "  distortion_coeffs: " + row(D), "  camera_model: pinhole", "  distortion_model: equidistant"]
    lines += ["T_cam1_cam0:"] + ["  - " + row(r) for r in T10]
    (cal / "camchain.yaml").write_text("\n".join(lines) + "\n")
    R1, _R2, P1, _P2, _Q = cv2.fisheye.stereoRectify(
        K0, D0.reshape(4, 1), K1, D1.reshape(4, 1), (800, 400), np.ascontiguousarray(T10[:3, :3]),
        np.ascontiguousarray(T10[:3, 3].reshape(3, 1)), flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0, fov_scale=1.0)
    fx = float(P1[0, 0]) + (0.0 if consistent else 1.0)
    (cal / "undistorted_calib_0.txt").write_text(
        "Pinhole %r %r %r %r 0.0 0.0 0.0 0.0\n800 400\ncrop\n800 400\n" % (fx, fx, float(P1[0, 2]), float(P1[1, 2])))
    (cal / "undistorted_calib_stereo.txt").write_text(
        "1.0 0.0 0.0 -0.3004961618953472\n0.0 1.0 0.0 0.0\n0.0 0.0 1.0 0.0\n0.0 0.0 0.0 1.0\n")
    t0 = 1_608_634_464_756_219_648
    cam_t = [t0 + int(i * 1e9 / 30) for i in range(n)]
    img = root / rec / "stereo_images_undistorted" / rec / "undistorted_images"
    for cam, dt in (("cam0", 0), ("cam1", 1_000_000)):
        for i, t in enumerate(cam_t):
            _png(img / cam / ("%d.png" % (t + dt)), 800, 400, i)
    imu = root / rec / "imu_gnss" / rec
    imu.mkdir(parents=True)
    ti = [t0 - 100_000_000 + k * 500_000 for k in range(int((n / 30 + 0.2) / 0.0005))]
    acc = _accel(len(ti))
    with open(imu / "imu.txt", "w") as f:
        for k, t in enumerate(ti):
            f.write("%d 0.000873 -0.000436 -0.003491 %.6f %.6f %.6f\n" % (t, *acc[k]))
    gt = root / rec / "reference_poses" / rec
    gt.mkdir(parents=True)
    with open(gt / "GNSSPoses.txt", "w") as f:
        f.write("# timestamp_ns, tran_x, tran_y, tran_z, quat_x, quat_y, quat_z, quat_w, scale, fusion_quality, v3\n")
        for i, t in enumerate(cam_t):
            f.write("%d,%f,-0.009789,0.030429,0.016877,0.722513,-0.690608,0.027401,0.980408,2\n" % (t, 0.1 * i))
    return root, rec, cam_t


def test_4seasons(tmp_path, capsys):
    root, rec, cam_t = make_4seasons(tmp_path / "in")
    out = tmp_path / "out"
    rc, so, se = _convert(capsys, "4seasons", root, out, "--recording", rec[len("recording_"):])
    assert rc == 0, se
    calib = _check_folder(out, 30)
    assert calib["dataset"] == "fourseasons_" + rec and calib["extrinsic_frame"] == "rectified_cam0"
    assert list(calib)[-3:] == ["extrinsic_frame", "rectfix_R1_deg", "rectfix_note"]
    assert 0.0 < calib["rectfix_R1_deg"] < 5.0
    T_raw = np.linalg.inv(np.array(calib["T_body_cam0"]))
    assert np.allclose(np.array(calib["R_cam0_imu"]), T_raw[:3, :3], atol=1e-12)
    gt = open(out / "gt.csv", encoding="utf-8").read().splitlines()
    assert gt[2].startswith("%d,0.098041,-0.009597,0.029833," % (cam_t[1] - cam_t[0]))   # x 0.980408: metric
    # the recording folder itself is accepted too, and gives the same bytes
    rc, _, se = _convert(capsys, "4seasons", root / rec, tmp_path / "out2")
    assert rc == 0, se
    for n in ("calib.json", "timestamps.csv", "imu.jsonl", "gt.csv"):
        assert open(out / n, "rb").read() == open(tmp_path / "out2" / n, "rb").read()


def test_4seasons_mismatched_calibration(tmp_path, capsys):
    root, rec, _ = make_4seasons(tmp_path / "in", n=3, consistent=False)
    rc, _, se = _convert(capsys, "4seasons", root, tmp_path / "out", "--recording", rec)
    assert rc == 1 and "do not belong together" in se and "Traceback" not in se
    rc, _, se = _convert(capsys, "4seasons", root, tmp_path / "out2")
    assert rc == 1 and "--recording" in se


# ------------------------------------------------------------------------------------------------ Rosario v1
ROSARIO_YAML = """cam0:
  T_cam_imu:
  - [0.009471441780975032, 0.9984567420628084, 0.05472134884952806, 0.07585973485189132]
  - [-0.9939081748565041, 0.0033909183804303744, 0.11015916496574298, -0.03342054396674176]
  - [0.10980360533243941, -0.055431362078334365, 0.9924064350628417, -0.027188513339280743]
  - [0.0, 0.0, 0.0, 1.0]
  cam_overlaps: [1]
  camera_model: pinhole
  distortion_coeffs: [-0.174497, 0.027127, -0.000359, 0.000457]
  distortion_model: radtan
  intrinsics: [34.8522264, 34.8449870, 34.4483596, 18.8808062]
  resolution: [64, 40]
  rostopic: /cam0/image_raw
cam1:
  T_cn_cnm1:
  - [0.9999064, 0.00116937, 0.01362243, -0.118718331]
  - [-0.00112614, 0.99999456,-0.00314845, 0.0000521848390]
  - [-0.01362563, 0.00313285, 0.99990197, -0.000269290935]
  - [0.0, 0.0, 0.0, 1.0]
  distortion_coeffs: [-0.174606, 0.027855, -0.000108, -0.000141]
  intrinsics: [34.9569635, 34.9390781, 34.0836585, 20.6105440]
  resolution: [64, 40]

gps:
  position_gps_baselink: [1.80070337, -0.02982362, 1.59345719] #[0.00070337, -0.01282362,  0.59345719] note
#imu, values units : g and Hz
imu:
  rotation_imu_baselink: [-7.84171987e-01, -7.88065033e-04, -6.20542634e-01, 7.16012579e-04] # [x, y, z, w]
  acc_offset: [24.13, 4.88, -24.93]  #acc offset
  gyro_offset: [-4.5970119258, 4.78218418728, 7.36174929329]  #gyro offset
"""


def make_rosario(root, seq="04", n=40):
    root.mkdir(parents=True)
    (root / ("calibration%s.yaml" % seq)).write_text(ROSARIO_YAML)
    t0 = 1514557628.106
    frames = [round(t0 + i / 15.0, 6) for i in range(n)]
    zed = root / "extracted" / ("sequence%s" % seq) / "zed"
    for i, t in enumerate(frames):
        _png(zed / ("left_%.6f.png" % t), 64, 40, i)
        _png(zed / ("right_%.6f.png" % (t + 0.001)), 64, 40, i)
    # ground truth spans frames 3 .. n-4: the outer frames and IMU samples are cut
    with open(root / ("sequence%s_gt.txt" % seq), "w") as f:
        for i in range(3, n - 2):
            f.write("%.9f %r 0.0 0.0 0 0 0 1\n" % (frames[i] - 0.0005, 0.1 * i))
    # IMU at 142 Hz, stamped at millisecond resolution; now and then a sample repeats the previous millisecond (the
    # raw-file defect the conversion repairs)
    ti = [round(t0 - 0.3 + k / 142.0, 3) for k in range(int((n / 15.0 + 0.6) * 142.0))]
    for k in range(150, len(ti), 300):
        ti[k] = ti[k - 1]
    acc = _accel(len(ti)) / G * 1000.0                 # mg
    with open(root / "extracted" / ("sequence%s" % seq) / "imu.log", "w") as f:
        for k, t in enumerate(ti):
            f.write("%.3f IMU: -4.60, 4.78, 7.36, %.2f, %.2f, %.2f\n" % (t, *acc[k]))
    return root, frames


def test_rosario_v1(tmp_path, capsys):
    root, frames = make_rosario(tmp_path / "in")
    out = tmp_path / "out"
    rc, so, se = _convert(capsys, "rosario-v1", root, out, "--seq", "04")
    assert rc == 0, se
    assert " 0 equal stamps in" not in so and " 0 out" in so   # equal IMU stamps were in the input, none are left
    calib = _check_folder(out, 40 - 6)                # cut to the ground-truth span
    assert calib["dataset"] == "rosario_04" and calib["imu_rate_hz"] == 142
    imu = [json.loads(x) for x in open(out / "imu.jsonl", encoding="utf-8")]
    t = np.array([s["t"] for s in imu])
    assert (np.diff(t) > 0).all()
    gx = np.array([s["gx"] for s in imu])
    assert np.allclose(gx, (-4.60 + 4.5970119258) * math.pi / 180, atol=1e-12)   # gyro offset subtracted
    assert abs(np.mean([s["az"] for s in imu]) - G) < 0.01                        # mg -> m/s^2
    rc, _, se = _convert(capsys, "rosario-v1", root, tmp_path / "o2", "--seq", "05")
    assert rc == 1 and "calibration05.yaml not found" in se


# ------------------------------------------------------------------------------------------------ MADMAX
INFO = """header:
  seq: 0
  frame_id: /camera_left
height: 772
width: 1032
distortion_model: plumb_bob
D: [-0.197412, 0.236726, 0.0, 0.0, 0.0]
K: [867.511, 0.187076, 519.346, 0.0, 866.993, 383.792, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [868.246, 0.0, 516.0, %s, 0.0, 868.246, 386.0, 0.0, 0.0, 0.0, 1.0, 0.0]
binning_x: 0
"""


def make_madmax(root, exp="F-5", n=30, zipped=True):
    import cv2
    d = root / exp
    d.mkdir(parents=True)
    t0 = 1_543_580_769_546_705_711
    cam_t = [t0 + int(i * 1e9 / 14) for i in range(n)]
    ok, buf = cv2.imencode(".png", _img(1032, 772))
    for side, dt in (("left", 0), ("right", 2_000_000)):
        if zipped:
            with zipfile.ZipFile(d / ("%s_img_rect_%s.zip" % (exp, side)), "w") as z:
                for t in cam_t:
                    z.writestr("img_rect_%s/img_rect_%s_%d.png" % (side, side, t + dt), buf.tobytes())
        else:
            (d / (side + "_")).mkdir()
            for t in cam_t:
                (d / (side + "_") / ("img_rect_%s_%d.png" % (side, t + dt))).write_bytes(buf.tobytes())
    with zipfile.ZipFile(d / ("%s_calibration.zip" % exp[0]), "w") as z:
        z.writestr("calibration/camera_rect_left_info.txt", INFO % "0.0")
        z.writestr("calibration/camera_rect_right_info.txt", INFO % "-78.045330571")
        z.writestr("calibration/tf__imu_to_camera_left.csv",
                   "# #position.x,position.y,position.z,orientation.x,orientation.y,orientation.z,orientation.w\n"
                   "0.210, 0.045, 0.080, 0.610, -0.604, 0.363, -0.362\n")
    rows = ["%time,field.header.seq,field.header.stamp,field.header.frame_id,field.angular_velocity.x,"
            "field.angular_velocity.y,field.angular_velocity.z,field.linear_acceleration.x,"
            "field.linear_acceleration.y,field.linear_acceleration.z,field.magnetometer.x"]
    ti = [t0 - 200_000_000 + k * 10_000_000 for k in range(int((n / 14 + 0.4) / 0.01))]
    acc = _accel(len(ti))
    for k, t in enumerate(ti):
        rows.append("%d,%d,%d,XsensImu,0.004,-0.003,-0.005,%r,%r,%r,0.0" % (t + 100, k, t, *map(float, acc[k])))
    (d / ("%s_imu_data.csv" % exp)).write_text("\n".join(rows) + "\n")
    gt = ["% time, x, y, z, vx, vy, vz, qw, qx, qy, qz, cov"]
    for i in range(n):
        gt.append("%.9f,%r,1.0,0.5,0,0,0,1.0,0.0,0.0,0.0,0.01" % ((cam_t[i] + 3_000_000) / 1e9, 0.2 * i))
    with zipfile.ZipFile(d / ("%s_ground_truth.zip" % exp), "w") as z:
        z.writestr("%s_ground_truth/gt_6DoF_gnss_and_imu.csv" % exp, "\n".join(gt) + "\n")
    return d, cam_t


def test_madmax(tmp_path, capsys):
    d, cam_t = make_madmax(tmp_path / "in")
    out = tmp_path / "out"
    rc, so, se = _convert(capsys, "madmax", d, out)
    assert rc == 0, se
    assert os.path.isdir(d / "left_") and os.path.isdir(d / "calib_" / "calibration")   # extracted once, in place
    calib = _check_folder(out, 30)
    assert calib["dataset"] == "madmax_F-5" and calib["camera_width"] == 1032
    assert abs(calib["baseline_m"] - 78.045330571 / 868.246) < 1e-15
    R = np.array(calib["T_body_cam0"])[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    # second run reuses the extracted folders and gives the same bytes
    rc, _, se = _convert(capsys, "madmax", d, tmp_path / "out2")
    assert rc == 0, se
    for n in ("calib.json", "timestamps.csv", "imu.jsonl", "gt.csv"):
        assert open(out / n, "rb").read() == open(tmp_path / "out2" / n, "rb").read()


def test_madmax_wrong_folder(tmp_path, capsys):
    d, _ = make_madmax(tmp_path / "in", n=3, zipped=False)
    rc, _, se = _convert(capsys, "madmax", d.parent, tmp_path / "out")
    assert rc == 1 and "_imu_data.csv" in se and "Traceback" not in se


# ------------------------------------------------------------------------------------------------ Rosario v2
def make_rosario2_bag(root, n=20):
    pytest.importorskip("PIL")
    pytest.importorskip("rosbags")
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    T = ts.types
    Time, Header = T["builtin_interfaces/msg/Time"], T["std_msgs/msg/Header"]
    V3, Q = T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"]

    def hdr(t):
        return Header(seq=0, stamp=Time(sec=int(t), nanosec=int(round((t - int(t)) * 1e9))), frame_id="x")
    root.mkdir(parents=True)
    bag, pgt = root / "seq.bag", root / "seq_pgt.bag"
    t0, w, h = 1703261656.25, 64, 40
    fx, B = 64.0, 0.05
    with Writer(bag) as wr:
        ci = wr.add_connection("/realsense/infra2/camera_info", "sensor_msgs/msg/CameraInfo", typestore=ts)
        cl = wr.add_connection("/realsense/infra1/image_rect_raw", "sensor_msgs/msg/Image", typestore=ts)
        cr = wr.add_connection("/realsense/infra2/image_rect_raw", "sensor_msgs/msg/Image", typestore=ts)
        cm = wr.add_connection("/realsense/imu", "sensor_msgs/msg/Imu", typestore=ts)
        roi = T["sensor_msgs/msg/RegionOfInterest"](x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)
        msg = T["sensor_msgs/msg/CameraInfo"](
            header=hdr(t0), height=h, width=w, distortion_model="plumb_bob", D=np.zeros(5),
            K=np.array([fx, 0, 32, 0, fx, 20, 0, 0, 1.0]), R=np.eye(3).ravel(),
            P=np.array([fx, 0, 32, -fx * B, 0, fx, 20, 0, 0, 0, 1, 0.0]), binning_x=0, binning_y=0, roi=roi)
        wr.write(ci, int(t0 * 1e9), ts.serialize_ros1(msg, "sensor_msgs/msg/CameraInfo"))
        events = []
        for i in range(n):
            t = t0 + i / 15.0
            for conn, dt in ((cl, 0.0), (cr, 0.001)):
                if conn is cr and i == 0:
                    continue                           # the first left image has no right partner
                img = _img(w, h, i)
                m = T["sensor_msgs/msg/Image"](header=hdr(t + dt), height=h, width=w, encoding="mono8",
                                                is_bigendian=0, step=w, data=img.ravel())
                events.append((int((t + dt) * 1e9), conn, ts.serialize_ros1(m, "sensor_msgs/msg/Image")))
        acc = _accel(int((n / 15.0 + 0.4) * 200))
        for k in range(len(acc)):
            t = t0 - 0.2 + k / 200.0
            m = T["sensor_msgs/msg/Imu"](header=hdr(t), orientation=Q(x=0, y=0, z=0, w=1),
                                        orientation_covariance=np.zeros(9),
                                        angular_velocity=V3(x=0.001, y=0.002, z=-0.001),
                                        angular_velocity_covariance=np.zeros(9),
                                        linear_acceleration=V3(x=acc[k, 0], y=acc[k, 1], z=acc[k, 2]),
                                        linear_acceleration_covariance=np.zeros(9))
            events.append((int(t * 1e9), cm, ts.serialize_ros1(m, "sensor_msgs/msg/Imu")))
        for t_ns, conn, raw in sorted(events, key=lambda e: e[0]):
            wr.write(conn, t_ns, raw)
    with Writer(pgt) as wr:
        cp = wr.add_connection("/pgt", "geometry_msgs/msg/PoseStamped", typestore=ts)
        for i in range(n):
            t = t0 + i / 15.0
            pose = T["geometry_msgs/msg/Pose"](position=T["geometry_msgs/msg/Point"](x=0.1 * i, y=0.0, z=0.0),
                                               orientation=Q(x=0, y=0, z=0, w=1))
            m = T["geometry_msgs/msg/PoseStamped"](header=hdr(t), pose=pose)
            wr.write(cp, int(t * 1e9), ts.serialize_ros1(m, "geometry_msgs/msg/PoseStamped"))
    return bag, pgt


def test_rosario_v2(tmp_path, capsys):
    bag, pgt = make_rosario2_bag(tmp_path / "in")
    out = tmp_path / "rosariov2_99"
    rc, so, se = _convert(capsys, "rosario-v2", bag, out, "--pgt", pgt)
    assert rc == 0, se
    calib = _check_folder(out, 19)
    assert calib["dataset"] == "rosariov2_99" and abs(calib["baseline_m"] - 0.05) < 1e-12
    assert not os.path.exists(out / "right_tmp")
    import cv2
    assert np.array_equal(cv2.imread(str(out / "left" / "000000.png"), cv2.IMREAD_GRAYSCALE), _img(64, 40, 1))
    # an output path with a trailing separator names the dataset the same way and gives the same bytes
    out2 = tmp_path / "again" / "rosariov2_99"
    rc, _, se = _convert(capsys, "rosario-v2", bag, str(out2) + os.sep, "--pgt", pgt)
    assert rc == 0, se
    for n in ("calib.json", "timestamps.csv", "imu.jsonl", "gt.csv", "left/000005.png", "right/000005.png"):
        assert open(out / n, "rb").read() == open(out2 / n, "rb").read(), n


def test_rosario_v2_needs_packages(tmp_path, capsys, monkeypatch):
    import sys
    bag = tmp_path / "x.bag"
    bag.write_bytes(b"")
    monkeypatch.setitem(sys.modules, "rosbags", None)
    monkeypatch.setitem(sys.modules, "rosbags.highlevel", None)
    rc, _, se = _convert(capsys, "rosario-v2", bag, tmp_path / "out")
    assert rc == 1 and "pip install rosbags pillow" in se and "Traceback" not in se
    assert not os.path.exists(tmp_path / "out")


# ------------------------------------------------------------------------------------------------ command line
def test_help_lists_every_format(capsys):
    rc, so, _ = _convert(capsys, "--help")
    assert rc == 0
    for name in ("euroc", "kitti", "4seasons", "rosario-v1", "rosario-v2", "madmax"):
        assert name in so
    rc, so, _ = _convert(capsys, "euroc", "--help")
    assert rc == 0 and "mav0" in so


def test_bad_command_lines(tmp_path, capsys):
    assert _convert(capsys)[0] == 2
    assert _convert(capsys, "tum-vi", tmp_path, tmp_path / "o")[0] == 2
    rc, _, se = _convert(capsys, "euroc", tmp_path, tmp_path / "o")
    assert rc == 1 and "no mav0/ folder" in se and "Traceback" not in se
    assert not os.path.exists(tmp_path / "o")
    rc, _, se = _convert(capsys, "euroc", tmp_path / "missing", tmp_path / "o")
    assert rc == 1 and "input folder not found" in se


def test_output_folder_rules(tmp_path, capsys):
    ds = make_kitti(tmp_path / "in", n=5)
    out = tmp_path / "out"
    assert _convert(capsys, "kitti", ds, out, "--seq", "04")[0] == 0
    first = open(out / "calib.json", "rb").read()
    rc, _, se = _convert(capsys, "kitti", ds, out, "--seq", "04")
    assert rc == 2 and "not empty" in se and "--force" in se and "Nothing was deleted" in se   # a refusal
    rc, _, se = _convert(capsys, "kitti", ds, out, "--seq", "04", "--force")
    assert rc == 0, se
    assert open(out / "calib.json", "rb").read() == first and len(os.listdir(out / "left")) == 5
    (out / "notes.txt").write_text("mine")
    rc, _, se = _convert(capsys, "kitti", ds, out, "--seq", "04", "--force")
    assert rc == 2 and "notes.txt" in se and (out / "notes.txt").exists() and (out / "calib.json").exists()
    rc, _, se = _convert(capsys, "kitti", ds, ds, "--seq", "04", "--force")
    assert rc == 2 and "is (or contains) the input" in se


# ------------------------------------------------------------------------------------------------ calibration YAML
KALIBR = """# a comment line
cam0:
  T_cam_imu:
  - [1, 0.5, -2.5e-01, 1.0e-5]
  - [0.0, 1.0, 0.0, 0.0]
  intrinsics: [348.522264, 348.449870, 344.483596, 188.808062]  # fx fy cx cy
  resolution: [672, 376]
  rostopic: /cam0/image_raw
  nested: [[1, 2], [3.5, 4]]
cam1:
    T_cn_cnm1:
      - [-0.00112614, 0.99999456,-0.00314845, 0.0000521848390]
    camera_model: 'pinhole'
empty:
gps:
  q: [-7.84171987e-01, 7.16012579e-04] #[0.1, 0.2] original
"""


def test_kalibr_yaml_reader(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(KALIBR)
    c = read_kalibr_yaml(str(p))
    assert c["cam0"]["T_cam_imu"] == [[1, 0.5, -0.25, 1e-5], [0.0, 1.0, 0.0, 0.0]]
    assert c["cam0"]["resolution"] == [672, 376] and c["cam0"]["rostopic"] == "/cam0/image_raw"
    assert c["cam0"]["nested"] == [[1, 2], [3.5, 4]]
    assert c["cam1"]["T_cn_cnm1"] == [[-0.00112614, 0.99999456, -0.00314845, 0.0000521848390]]
    assert c["cam1"]["camera_model"] == "pinhole" and c["empty"] is None
    assert c["gps"]["q"] == [-0.784171987, 0.000716012579]


def test_kalibr_yaml_reader_matches_pyyaml(tmp_path):
    """Where PyYAML is installed: the numbers the converters read are the same as a full YAML parser's."""
    yaml = pytest.importorskip("yaml")
    p = tmp_path / "c.yaml"
    p.write_text(KALIBR + ROSARIO_YAML.replace("cam0:", "c0:").replace("cam1:", "c1:").replace("gps:", "g2:"))
    c, ref = read_kalibr_yaml(str(p)), yaml.safe_load(p.read_text())

    def num(x):
        if isinstance(x, list):
            return [num(v) for v in x]
        return float(x) if isinstance(x, (int, float, str)) and not isinstance(x, bool) else x
    for a, b in (("cam0", "T_cam_imu"), ("cam0", "intrinsics"), ("cam1", "T_cn_cnm1"), ("gps", "q"),
                 ("c0", "T_cam_imu"), ("c0", "intrinsics"), ("c0", "distortion_coeffs"), ("c0", "resolution"),
                 ("c1", "T_cn_cnm1"), ("c1", "intrinsics"), ("c1", "distortion_coeffs"), ("imu", "gyro_offset")):
        assert num(c[a][b]) == num(ref[a][b]), (a, b)
