# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""`xclean calib`: calib.json from synthetic Kalibr and OpenCV calibrations, the validation errors, --update."""
import json
import os

import numpy as np
import pytest

from xclean.io import calib_import as C


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


# a camera 2 cm from the IMU, rotated like a typical rig; the right camera 7.5 cm to the right, slightly rotated
T_C0_IMU = _T(_rot("x", -90) @ _rot("z", 1.5), [0.02, -0.01, 0.005])
T_C1_C0 = _T(_rot("y", 0.4) @ _rot("z", -0.2), [-0.075, 0.0003, -0.0002])
T_C1_IMU = T_C1_C0 @ T_C0_IMU


def _rows(T):
    return "\n".join("  - [%s]" % ", ".join(repr(float(x)) for x in row) for row in T)


def _camchain(path, model="pinhole", dist="radtan", T_rl=T_C1_C0, res2=(640, 400), imu=True, shift=0.0):
    txt = "# Kalibr camchain (synthetic)\ncam0:\n"
    if imu:
        txt += "  T_cam_imu:\n%s\n" % _rows(T_C0_IMU)
    txt += ("  cam_overlaps: [1]\n  camera_model: %s\n  distortion_coeffs: [-0.28, 0.074,\n    0.0002, 1.8e-05]\n"
            "  distortion_model: %s\n  intrinsics: [450.5, 449.25, 321.0, 198.5]  # fu fv pu pv\n"
            "  resolution: [640, 400]\n  rostopic: /cam0/image_raw\n  timeshift_cam_imu: %r\n" % (model, dist, shift))
    txt += "cam1:\n"
    if imu:
        txt += "  T_cam_imu:\n%s\n" % _rows(T_C1_IMU)
    txt += "  T_cn_cnm1:\n%s\n" % _rows(T_rl)
    txt += ("  cam_overlaps: [0]\n  camera_model: pinhole\n  distortion_coeffs: [-0.27, 0.07, -0.0001, -3.5e-05]\n"
            "  distortion_model: radtan\n  intrinsics: [452.0, 451.0, 318.5, 201.0]\n  resolution: [%d, %d]\n"
            "  rostopic: /cam1/image_raw\n  timeshift_cam_imu: 0.0\n" % res2)
    with open(path, "w") as f:
        f.write(txt)
    return str(path)


def _imu_yaml(path, drop=None):
    vals = {"accelerometer_noise_density": 0.002, "accelerometer_random_walk": 0.003,
            "gyroscope_noise_density": 0.00017, "gyroscope_random_walk": 1.9e-05}
    txt = "imu0:\n  T_i_b:\n%s\n" % _rows(np.eye(4))
    for k, v in vals.items():
        if k != drop:
            txt += "  %s: %r\n" % (k, v)
    txt += "  model: calibrated\n  rostopic: /imu0\n  time_offset: 0.0\n  update_rate: 200.0\n"
    with open(path, "w") as f:
        f.write(txt)
    return str(path)


# ------------------------------------------------------------------------------------------------------ YAML reader
def test_yaml_subset():
    y = C.load_yaml_subset("%YAML:1.0\n---\na:\n  b: [1, 2.5, [3, x]]  # c\n  m:\n  - [1, 2]\n  - [3, 4]\n"
                           "  n:\n    - 5\n    - 'six'\n  long: [1,\n    2,\n    3]\n  t: true\n  e:\nz: 1.5e-04\n")
    assert y == {"a": {"b": [1, 2.5, [3, "x"]], "m": [[1, 2], [3, 4]], "n": [5, "six"], "long": [1, 2, 3], "t": True,
                       "e": None}, "z": 1.5e-4}


def test_yaml_subset_matches_pyyaml_on_a_camchain(tmp_path):
    yaml = pytest.importorskip("yaml")
    text = open(_camchain(tmp_path / "cc.yaml")).read()
    assert C.load_yaml_subset(text) == yaml.safe_load(text)


def test_yaml_subset_rejects_anchors():
    with pytest.raises(C.CalibError, match="unsupported"):
        C.load_yaml_subset("a: &x 1\nb: *x\n")


# ------------------------------------------------------------------------------------------------------------ Kalibr
def test_kalibr_fields(tmp_path):
    w = []
    c = C.from_kalibr(_camchain(tmp_path / "cc.yaml"), imu=_imu_yaml(tmp_path / "imu.yaml"), warns=w)
    assert (c["camera_width"], c["camera_height"]) == (640, 400)
    assert np.allclose(c["M1"], [[450.5, 0, 321.0], [0, 449.25, 198.5], [0, 0, 1]])
    assert c["D1"] == [-0.28, 0.074, 0.0002, 1.8e-05, 0.0]
    assert np.allclose(c["R_stereo"], T_C1_C0[:3, :3]) and np.allclose(c["T_stereo"], T_C1_C0[:3, 3])
    assert c["baseline_m"] == pytest.approx(np.linalg.norm(T_C1_C0[:3, 3]))
    assert np.allclose(c["R_cam0_imu"], T_C0_IMU[:3, :3])
    assert np.allclose(np.array(c["T_body_cam0"]) @ T_C0_IMU, np.eye(4))
    assert c["imu_noise"]["gyroscope_noise_density"] == 0.00017 and c["imu_rate_hz"] == 200.0
    assert "td_s" not in c and not w
    rig = C.validate(dict(c), w)
    assert rig.width == 640 and rig.fB_m_px > 0


def test_kalibr_without_imu_is_vision_only(tmp_path):
    c = C.from_kalibr(_camchain(tmp_path / "cc.yaml", imu=False))
    assert "imu_noise" not in c and np.allclose(c["T_body_cam0"], np.eye(4))
    with pytest.raises(C.CalibError, match="no T_cam_imu"):
        C.from_kalibr(_camchain(tmp_path / "cc2.yaml", imu=False), imu=_imu_yaml(tmp_path / "imu.yaml"))


def test_kalibr_timeshift_is_kept_with_a_warning(tmp_path):
    w = []
    c = C.from_kalibr(_camchain(tmp_path / "cc.yaml", shift=0.0042), warns=w)
    assert c["td_s"] == 0.0042 and any("timeshift" in x for x in w)


@pytest.mark.parametrize("kw, match", [
    ({"model": "omni"}, "pinhole"),
    ({"dist": "equidistant"}, "radtan"),
    ({"res2": (752, 480)}, "different resolutions"),
])
def test_kalibr_unusable(tmp_path, kw, match):
    with pytest.raises(C.CalibError, match=match):
        C.from_kalibr(_camchain(tmp_path / "cc.yaml", **kw))


def test_imu_yaml_missing_a_density(tmp_path):
    with pytest.raises(C.CalibError, match="gyroscope_random_walk"):
        C.from_kalibr(_camchain(tmp_path / "cc.yaml"), imu=_imu_yaml(tmp_path / "imu.yaml",
                                                                       drop="gyroscope_random_walk"))


# ------------------------------------------------------------------------------------------------------- validation
def _good(tmp_path):
    return C.from_kalibr(_camchain(tmp_path / "cc.yaml"), imu=_imu_yaml(tmp_path / "imu.yaml"))


def test_swapped_cameras(tmp_path):
    c = _good(tmp_path)
    c["T_stereo"] = [-x for x in c["T_stereo"]]
    with pytest.raises(C.CalibError, match="LEFT"):
        C.validate(c)


def test_vertical_pair(tmp_path):
    c = _good(tmp_path)
    c["T_stereo"] = [0.0005, -0.075, 0.0]
    c["baseline_m"] = float(np.linalg.norm(c["T_stereo"]))
    with pytest.raises(C.CalibError, match="vertical"):
        C.validate(c)


def test_millimetres_are_caught(tmp_path):
    c = _good(tmp_path)
    c["T_stereo"] = [x * 1000 for x in c["T_stereo"]]
    c["baseline_m"] = float(np.linalg.norm(c["T_stereo"]))
    with pytest.raises(C.CalibError, match="millimetres"):
        C.validate(c)


def test_missing_field_and_bad_principal_point(tmp_path):
    c = _good(tmp_path)
    del c["M2"]
    with pytest.raises(C.CalibError, match="missing M2"):
        C.validate(c)
    c = _good(tmp_path)
    c["M1"][0][2] = 900.0
    with pytest.raises(C.CalibError, match="outside"):
        C.validate(c)


def test_noise_out_of_range_is_a_warning(tmp_path):
    c = _good(tmp_path)
    c["imu_noise"]["gyroscope_noise_density"] = 3.0
    w = []
    C.validate(c, w)
    assert any("gyroscope_noise_density" in x for x in w)


# ------------------------------------------------------------------------------------------------------------ OpenCV
def _opencv(path, T_scale=1.0, size=True, imu=False, ext_size=False):
    import cv2
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("K1", np.array([[450.5, 0, 321.0], [0, 449.25, 198.5], [0, 0, 1]]))
    fs.write("D1", np.array([[-0.28, 0.074, 0.0002, 1.8e-05, 0.01]]))
    fs.write("K2", np.array([[452.0, 0, 318.5], [0, 451.0, 201.0], [0, 0, 1]]))
    fs.write("D2", np.array([[-0.27, 0.07, -0.0001, -3.5e-05, 0.0]]))
    fs.write("R", T_C1_C0[:3, :3])
    fs.write("T", (T_C1_C0[:3, 3] * T_scale).reshape(3, 1))
    if size and not ext_size:
        fs.write("image_width", 640)
        fs.write("image_height", 400)
    if ext_size:
        fs.write("imageSize", np.array([[640, 400]], np.float64))
    if imu:
        Tci = T_C0_IMU.copy()
        Tci[:3, 3] *= T_scale
        fs.write("T_cam_imu", Tci)
    fs.release()
    return str(path)


@pytest.mark.parametrize("name", ["stereo.yml", "stereo.xml"])
def test_opencv_metres(tmp_path, name):
    c = C.from_opencv(_opencv(tmp_path / name))
    assert np.allclose(c["T_stereo"], T_C1_C0[:3, 3]) and c["D1"][4] == 0.01
    assert "imu_noise" not in c
    C.validate(c)


def test_opencv_millimetres(tmp_path):
    p = _opencv(tmp_path / "s.yml", T_scale=1000.0)
    with pytest.raises(C.CalibError, match="--units mm"):
        C.validate(C.from_opencv(p))
    c = C.from_opencv(p, units="mm")
    assert c["baseline_m"] == pytest.approx(np.linalg.norm(T_C1_C0[:3, 3]))


def test_opencv_size_sources(tmp_path):
    with pytest.raises(C.CalibError, match="--size"):
        C.from_opencv(_opencv(tmp_path / "a.yml", size=False))
    assert C.from_opencv(_opencv(tmp_path / "b.yml", size=False), size=(640, 400))["camera_width"] == 640
    assert C.from_opencv(_opencv(tmp_path / "c.yml", ext_size=True))["camera_height"] == 400


def test_opencv_with_imu(tmp_path):
    c = C.from_opencv(_opencv(tmp_path / "s.yml", T_scale=1000.0, imu=True), units="mm",
                      imu=_imu_yaml(tmp_path / "imu.yaml"))
    assert np.allclose(np.array(c["T_body_cam0"]) @ T_C0_IMU, np.eye(4)) and "imu_noise" in c
    with pytest.raises(C.CalibError, match="T_cam_imu"):
        C.from_opencv(_opencv(tmp_path / "n.yml"), imu=_imu_yaml(tmp_path / "imu2.yaml"))


# ------------------------------------------------------------------------------------------------------------ the CLI
def test_cli_write_update_and_check(tmp_path, capsys):
    cc, imu = _camchain(tmp_path / "cc.yaml"), _imu_yaml(tmp_path / "imu.yaml")
    seq = tmp_path / "myseq"
    out = str(seq / "calib.json")
    assert C.main(["kalibr", cc, "--imu", imu, "--out", out, "--fps", "30"]) == 0
    d = json.load(open(out))
    assert d["source"] == "own camera (Kalibr import)" and d["dataset"] == "myseq" and d["camera_fps"] == 30.0
    assert "t0_ns" in capsys.readouterr().out                     # the note about recording fields
    assert C.main(["kalibr", cc, "--out", out]) == 2              # exists: refused, nothing written
    assert "--force" in capsys.readouterr().err
    d.update(t0_ns=123, num_frames=7, source="my recorder")
    json.dump(d, open(out, "w"))
    assert C.main(["kalibr", cc, "--out", out, "--update"]) == 0  # no --imu now: the IMU block goes, the rest stays
    d2 = json.load(open(out))
    assert d2["t0_ns"] == 123 and d2["num_frames"] == 7 and d2["source"] == "my recorder"
    assert "imu_noise" not in d2 and d2["camera_fps"] == 30.0
    assert C.main(["check", out]) == 0
    assert "is usable" in capsys.readouterr().out


def test_cli_errors_are_plain(tmp_path, capsys):
    assert C.main(["kalibr", str(tmp_path / "none.yaml"), "--out", str(tmp_path / "c.json")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("xclean calib: error: file not found") and "Traceback" not in err
    assert not os.path.exists(tmp_path / "c.json")
