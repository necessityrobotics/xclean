# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""`xclean calib`: write the calib.json of a sequence folder (docs/SEQUENCE_FORMAT.md) for your own stereo camera.

    xclean calib kalibr camchain-imucam.yaml [--imu imu.yaml] --out calib.json [--fps 30] [--left cam0 --right cam1]
    xclean calib opencv stereo.yml [--units m|cm|mm] [--size 1280x720] [--imu imu.yaml] --out calib.json [--fps 30]
    xclean calib check calib.json
    xclean calib capture --source uvc-sbs --device 0 --size 1280x720 --out pairs   (image pairs for a calibration)

kalibr  A Kalibr camera chain.  camchain-imucam.yaml (from a camera-IMU calibration) gives the IMU extrinsic too;
        a plain camchain.yaml gives a vision-only calib.json.  --imu takes Kalibr's imu.yaml for the IMU noise block
        (the four continuous-time densities) and the IMU rate.  Pinhole cameras with radtan (or no) distortion only.
opencv  An OpenCV FileStorage file (.yml / .yaml / .xml) from cv2.stereoCalibrate: K1, D1, K2, D2, R, T and the image
        size (image_width / image_height, or imageSize, or --size).  Aliases M1 / M2, cameraMatrix1 / 2, distCoeffs1 / 2
        are accepted.  R, T map points from the left camera to the right one, as cv2.stereoRectify takes them.  T is in
        the unit of your calibration target: give --units mm (or cm) when it is not metres.  Vision-only unless the
        file also holds T_cam_imu (4x4, IMU -> left camera, metres) and --imu gives the noise block.
check   Validates an existing calib.json with the same rules and loads it as the engine does.
capture Saves stereo image pairs from a rig through a frame source (xclean/live/sources.py), for a chessboard
        calibration with OpenCV (docs/ANY_STEREO_LIVE.md); its result then goes through `opencv`.

What is written: camera_width / camera_height, M1 / D1 / M2 / D2 (raw images; xclean rectifies them itself),
R_stereo / T_stereo (left -> right, metres), baseline_m, T_body_cam0 (left camera -> IMU body) and R_cam0_imu (IMU body
-> left camera), and with an IMU imu_noise and imu_rate_hz; camera_fps with --fps; source and dataset labels.
Recording fields (t0_ns, num_frames) belong to a recording: your recorder or converter adds them.  --update replaces
only the calibration fields of an existing calib.json and keeps the others.

Checks (a failure stops with a plain message and writes nothing): missing fields, matrix shapes, finite values, focal
lengths and principal points, rotation matrices, a baseline in metres (catches millimetres), a roughly parallel
side-by-side pair with the right camera on the right, rectification with cv2.stereoRectify exactly as xclean does it,
and the IMU noise block.  Values that are only unusual are printed as warnings.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

CALIB_KEYS = ("camera_width", "camera_height", "M1", "D1", "M2", "D2", "R_stereo", "T_stereo", "baseline_m",
              "R_cam0_imu", "T_body_cam0", "td_s", "imu_noise")          # always replaced by --update
SENSOR_KEYS = ("camera_fps", "imu_rate_hz", "source", "dataset")          # replaced by --update when given
NOISE_KEYS = ("gyroscope_noise_density", "gyroscope_random_walk", "accelerometer_noise_density",
              "accelerometer_random_walk")
# plausible ranges of the continuous-time densities (outside: a warning about units, not an error)
NOISE_RANGE = {"gyroscope_noise_density": (1e-6, 1e-1), "gyroscope_random_walk": (1e-8, 1e-1),
               "accelerometer_noise_density": (1e-5, 1.0), "accelerometer_random_walk": (1e-7, 1.0)}
BASELINE_RANGE_M = (0.005, 5.0)
MAX_PAIR_ROTATION_DEG = 30.0


class CalibError(ValueError):
    """A calibration that xclean cannot use; the message says what to fix."""


# ================================================================================================== a small YAML reader
def _strip_comment(line):
    out, q = [], None
    for i, ch in enumerate(line):
        if q:
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _scalar(s):
    s = s.strip()
    if s in ("", "~", "null", "Null", "NULL"):
        return None
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]
    if s in ("true", "True", "TRUE"):
        return True
    if s in ("false", "False", "FALSE"):
        return False
    for conv in (int, float):
        try:
            return conv(s)
        except ValueError:
            pass
    return s


def _flow(s):
    """A YAML flow sequence '[a, b, [c, d]]' (nested allowed) -> list."""
    pos = 0

    def parse():
        nonlocal pos
        while pos < len(s) and s[pos] in " \t":
            pos += 1
        if pos < len(s) and s[pos] == "[":
            pos += 1
            out = []
            while True:
                while pos < len(s) and s[pos] in " \t,":
                    pos += 1
                if pos >= len(s):
                    raise CalibError("unterminated list in YAML: %r" % s[:80])
                if s[pos] == "]":
                    pos += 1
                    return out
                out.append(parse())
        start = pos
        while pos < len(s) and s[pos] not in ",]":
            pos += 1
        return _scalar(s[start:pos])

    v = parse()
    if s[pos:].strip():
        raise CalibError("unexpected text after a YAML list: %r" % s[pos:pos + 40])
    return v


def _split_key(content):
    """'key: rest' -> (key, rest) or None when the line is not a mapping entry."""
    q, depth = None, 0
    for i, ch in enumerate(content):
        if q:
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == ":" and depth == 0 and (i + 1 == len(content) or content[i + 1] in " \t"):
            return _scalar(content[:i]), content[i + 1:].strip()
    return None


def load_yaml_subset(text):
    """The YAML that Kalibr writes (block maps, block lists of flow lists, flow lists over several lines, comments,
    scalars).  Anchors, multi-document streams and block scalars are not supported (a plain error says so)."""
    rows = []
    pending = None
    for raw in text.splitlines():
        if raw.lstrip().startswith(("%YAML", "---", "...")):
            continue
        line = _strip_comment(raw.replace("\t", "    "))
        if not line.strip():
            continue
        if pending is not None:
            pending[1] += " " + line.strip()
            if pending[1].count("[") <= pending[1].count("]"):
                rows.append(tuple(pending))
                pending = None
            continue
        ind = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if content.count("[") > content.count("]"):
            pending = [ind, content]
            continue
        if content.startswith(("&", "*", "!")) or content.endswith(("|", ">")):
            raise CalibError("unsupported YAML construct: %r (write the file without anchors or block text)" % content)
        rows.append((ind, content))
    if pending is not None:
        raise CalibError("unterminated list in YAML: %r" % pending[1][:80])

    def value(rest):
        if rest.startswith(("&", "*", "!", "|", ">", "{")):
            raise CalibError("unsupported YAML construct: %r (write the file without anchors, tags, block text or "
                             "inline maps)" % rest[:60])
        return _flow(rest) if rest.startswith("[") else _scalar(rest)

    def block(i, ind):
        if rows[i][1] == "-" or rows[i][1].startswith("- "):
            return seq(i, ind)
        return mapping(i, ind)

    def seq(i, ind):
        out = []
        while i < len(rows) and rows[i][0] == ind and (rows[i][1] == "-" or rows[i][1].startswith("- ")):
            item = rows[i][1][1:].strip()
            if item == "":
                if i + 1 < len(rows) and rows[i + 1][0] > ind:
                    v, i = block(i + 1, rows[i + 1][0])
                else:
                    v, i = None, i + 1
                out.append(v)
                continue
            if _split_key(item) is not None and not item.startswith("["):
                raise CalibError("unsupported YAML construct (a map inside a list): %r" % item)
            out.append(value(item))
            i += 1
        return out, i

    def mapping(i, ind):
        out = {}
        while i < len(rows) and rows[i][0] == ind:
            kv = _split_key(rows[i][1])
            if kv is None:
                raise CalibError("cannot read YAML line %r" % rows[i][1])
            k, rest = kv
            i += 1
            if rest:
                out[k] = value(rest)
            elif i < len(rows) and rows[i][0] > ind:
                out[k], i = block(i, rows[i][0])
            elif i < len(rows) and rows[i][0] == ind and (rows[i][1] == "-" or rows[i][1].startswith("- ")):
                out[k], i = seq(i, ind)
            else:
                out[k] = None
        return out, i

    if not rows:
        return {}
    v, i = block(0, rows[0][0])
    if i != len(rows):
        raise CalibError("cannot read YAML from line %r on (inconsistent indentation?)" % rows[i][1])
    return v


def load_yaml(path):
    """A YAML file -> Python objects (PyYAML when installed, else load_yaml_subset)."""
    if not os.path.isfile(path):
        raise CalibError("file not found: %s" % path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml
    except ImportError:
        return load_yaml_subset(text)
    try:
        return yaml.safe_load(text)
    except Exception as e:
        raise CalibError("%s is not valid YAML: %s" % (path, e))


# ======================================================================================================= math helpers
def _mat(v, shape, what):
    try:
        a = np.array(v, dtype=float)
    except (TypeError, ValueError):
        raise CalibError("%s is not a numeric matrix" % what)
    if a.size != int(np.prod(shape)):
        raise CalibError("%s must have %s values (%s), got %d" % (what, int(np.prod(shape)), "x".join(map(str, shape)),
                                                                  a.size))
    a = a.reshape(shape)
    if not np.all(np.isfinite(a)):
        raise CalibError("%s contains NaN or infinite values" % what)
    return a


def _inv_se3(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _check_rotation(R, what):
    if np.abs(R.T @ R - np.eye(3)).max() > 1e-3 or abs(np.linalg.det(R) - 1.0) > 1e-3:
        raise CalibError("%s is not a rotation matrix (not orthonormal with determinant +1)" % what)


def _angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))


def _intrinsics(fx, fy, cx, cy, W, H, what):
    if not (fx > 0 and fy > 0):
        raise CalibError("%s: the focal lengths must be positive (fx %r, fy %r)" % (what, fx, fy))
    if not (0 < cx < W and 0 < cy < H):
        raise CalibError("%s: the principal point (%g, %g) lies outside the %dx%d image: is the image size right, or "
                         "were the intrinsics computed at another resolution?" % (what, cx, cy, W, H))
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _dist(d, what):
    d = [float(x) for x in np.asarray(d, float).ravel()]
    if len(d) not in (0, 4, 5, 8, 12, 14):
        raise CalibError("%s: OpenCV distortion has 4, 5, 8, 12 or 14 coefficients, got %d" % (what, len(d)))
    if not all(math.isfinite(x) for x in d):
        raise CalibError("%s: distortion contains NaN or infinite values" % what)
    while len(d) < 5:
        d.append(0.0)
    return d


# ============================================================================================================ builders
def _imu_block(imu_path, warns):
    """Kalibr imu.yaml -> (imu_noise dict, rate or None)."""
    y = load_yaml(imu_path)
    if not isinstance(y, dict):
        raise CalibError("%s: expected a YAML map" % imu_path)
    node = y
    if not any(k in y for k in NOISE_KEYS):
        keys = [k for k in y if str(k).startswith("imu") and isinstance(y[k], dict)]
        if not keys:
            raise CalibError("%s: no IMU block (expected imu0: with the four noise densities)" % imu_path)
        node = y[sorted(keys)[0]]
    noise = {}
    for k in NOISE_KEYS:
        if k not in node:
            raise CalibError("%s: missing %s (all four continuous-time noise densities are needed)" % (imu_path, k))
        try:
            v = float(node[k])
        except (TypeError, ValueError):
            raise CalibError("%s: %s is not a number" % (imu_path, k))
        if not (v > 0 and math.isfinite(v)):
            raise CalibError("%s: %s must be a positive number, got %r" % (imu_path, k, node[k]))
        noise[k] = v
    rate = node.get("update_rate", node.get("rate_hz"))
    if rate is not None:
        try:
            rate = float(rate)
        except (TypeError, ValueError):
            raise CalibError("%s: update_rate is not a number" % imu_path)
        if not rate > 0:
            raise CalibError("%s: update_rate must be positive" % imu_path)
    T_i_b = node.get("T_i_b")
    if T_i_b is not None and np.abs(_mat(T_i_b, (4, 4), "T_i_b") - np.eye(4)).max() > 1e-9:
        warns.append("imu.yaml has a non-identity T_i_b; xclean's body frame is the IMU frame, so it is not used")
    return noise, rate


def _cam(cc, name, path):
    c = cc.get(name)
    if not isinstance(c, dict):
        raise CalibError("%s: no camera %r (cameras in the file: %s)" % (path, name, ", ".join(
            str(k) for k in cc if str(k).startswith("cam")) or "none"))
    for k in ("intrinsics", "resolution"):
        if k not in c:
            raise CalibError("%s: %s has no %s" % (path, name, k))
    model = str(c.get("camera_model", "pinhole")).lower()
    if model != "pinhole":
        raise CalibError("%s: %s uses the camera model %r; xclean needs a pinhole camera (recalibrate with "
                         "pinhole-radtan)" % (path, name, model))
    dm = str(c.get("distortion_model", "radtan")).lower()
    if dm in ("radtan", "radial-tangential", "plumb_bob"):
        d = c.get("distortion_coeffs", c.get("distortion_coefficients"))
        if d is None or len(d) != 4:
            raise CalibError("%s: %s radtan distortion needs 4 coefficients (k1, k2, p1, p2)" % (path, name))
    elif dm == "none":
        d = []
    else:
        raise CalibError("%s: %s uses the distortion model %r; xclean rectifies with OpenCV's pinhole model and needs "
                         "radtan (or none). Recalibrate with pinhole-radtan" % (path, name, dm))
    intr = c["intrinsics"]
    res = c["resolution"]
    if not (isinstance(intr, list) and len(intr) == 4):
        raise CalibError("%s: %s intrinsics must be [fu, fv, pu, pv]" % (path, name))
    if not (isinstance(res, list) and len(res) == 2 and all(isinstance(x, (int, float)) for x in res)):
        raise CalibError("%s: %s resolution must be [width, height]" % (path, name))
    W, H = int(res[0]), int(res[1])
    if W <= 0 or H <= 0:
        raise CalibError("%s: %s resolution must be positive" % (path, name))
    M = _intrinsics(*(float(x) for x in intr), W, H, "%s %s" % (path, name))
    return {"W": W, "H": H, "M": M, "D": _dist(d, "%s %s" % (path, name)),
            "T_cam_imu": None if c.get("T_cam_imu") is None else _mat(c["T_cam_imu"], (4, 4), "%s T_cam_imu" % name),
            "T_cn_cnm1": None if c.get("T_cn_cnm1") is None else _mat(c["T_cn_cnm1"], (4, 4), "%s T_cn_cnm1" % name),
            "timeshift": float(c.get("timeshift_cam_imu") or 0.0)}


def from_kalibr(camchain, imu=None, left="cam0", right="cam1", warns=None):
    """Kalibr camchain(-imucam).yaml [+ imu.yaml] -> calib dict (calibration fields only)."""
    warns = [] if warns is None else warns
    cc = load_yaml(camchain)
    if not isinstance(cc, dict):
        raise CalibError("%s: expected a YAML map with cam0:, cam1:" % camchain)
    L, R = _cam(cc, left, camchain), _cam(cc, right, camchain)
    if (L["W"], L["H"]) != (R["W"], R["H"]):
        raise CalibError("%s: the two cameras have different resolutions (%dx%d and %dx%d); xclean needs one size"
                         % (camchain, L["W"], L["H"], R["W"], R["H"]))
    adjacent = left.startswith("cam") and right.startswith("cam") and left[3:].isdigit() and right[3:].isdigit() \
        and int(right[3:]) == int(left[3:]) + 1
    T_rl = None
    if R["T_cn_cnm1"] is not None and adjacent:
        T_rl = R["T_cn_cnm1"]
    if L["T_cam_imu"] is not None and R["T_cam_imu"] is not None:
        T_rl_imu = R["T_cam_imu"] @ _inv_se3(L["T_cam_imu"])
        if T_rl is None:
            T_rl = T_rl_imu
        elif np.abs(T_rl_imu[:3, 3] - T_rl[:3, 3]).max() > 1e-3 or _angle_deg(T_rl_imu[:3, :3].T @ T_rl[:3, :3]) > 0.1:
            warns.append("%s: T_cn_cnm1 and the two T_cam_imu disagree by more than 1 mm / 0.1 deg; T_cn_cnm1 is used"
                         % right)
    if T_rl is None:
        raise CalibError("%s: no stereo extrinsic: %s needs T_cn_cnm1 (relative to %s) or both cameras need T_cam_imu"
                         % (camchain, right, left))
    out = {"camera_width": L["W"], "camera_height": L["H"], "M1": L["M"].tolist(), "D1": L["D"],
           "M2": R["M"].tolist(), "D2": R["D"], "R_stereo": T_rl[:3, :3].tolist(), "T_stereo": T_rl[:3, 3].tolist(),
           "baseline_m": float(np.linalg.norm(T_rl[:3, 3]))}
    if L["T_cam_imu"] is not None:
        Tci = L["T_cam_imu"]
        out["R_cam0_imu"] = Tci[:3, :3].tolist()
        out["T_body_cam0"] = _inv_se3(Tci).tolist()
        if L["timeshift"] != 0.0:
            out["td_s"] = L["timeshift"]
            warns.append("%s: timeshift_cam_imu = %.6f s is recorded as td_s, but xclean does not apply it: add it to "
                         "your camera timestamps (t_imu = t_cam + shift) when you write timestamps.csv"
                         % (left, L["timeshift"]))
    else:
        out["R_cam0_imu"] = np.eye(3).tolist()
        out["T_body_cam0"] = np.eye(4).tolist()
    if imu:
        if L["T_cam_imu"] is None:
            raise CalibError("--imu was given but %s has no T_cam_imu: use the camchain-imucam.yaml of Kalibr's "
                             "camera-IMU calibration" % camchain)
        out["imu_noise"], rate = _imu_block(imu, warns)
        if rate:
            out["imu_rate_hz"] = rate
    elif L["T_cam_imu"] is not None:
        warns.append("no --imu imu.yaml: calib.json gets no imu_noise block, so xclean runs vision-only")
    return out


def _fs_mat(fs, names):
    for n in names:
        node = fs.getNode(n)
        if node is not None and not node.empty():
            if node.isSeq():
                return n, np.array([node.at(i).real() for i in range(node.size())], float)
            if node.isReal() or node.isInt():
                return n, np.array([node.real()], float)
            m = node.mat()
            if m is not None:
                return n, np.asarray(m, float)
    return None, None


def from_opencv(path, units="m", size=None, imu=None, warns=None):
    """cv2.stereoCalibrate results in an OpenCV FileStorage file -> calib dict (calibration fields only)."""
    import cv2
    warns = [] if warns is None else warns
    if not os.path.isfile(path):
        raise CalibError("file not found: %s" % path)
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise CalibError("%s: OpenCV cannot read this file (expected a FileStorage .yml / .yaml / .xml)" % path)
    try:
        got = {}
        for key, names in (("K1", ("K1", "M1", "cameraMatrix1", "camera_matrix1", "CM1")),
                           ("D1", ("D1", "distCoeffs1", "dist_coeffs1", "dist1")),
                           ("K2", ("K2", "M2", "cameraMatrix2", "camera_matrix2", "CM2")),
                           ("D2", ("D2", "distCoeffs2", "dist_coeffs2", "dist2")),
                           ("R", ("R", "R_stereo")), ("T", ("T", "T_stereo"))):
            n, m = _fs_mat(fs, names)
            if m is None:
                raise CalibError("%s: missing %s (looked for %s)" % (path, key, ", ".join(names)))
            got[key] = m
        if size is not None:
            W, H = size
        else:
            _, w = _fs_mat(fs, ("image_width", "imageWidth", "width"))
            _, h = _fs_mat(fs, ("image_height", "imageHeight", "height"))
            if w is not None and h is not None:
                W, H = int(w.ravel()[0]), int(h.ravel()[0])
            else:
                _, s = _fs_mat(fs, ("imageSize", "image_size", "size"))
                if s is None or s.size != 2:
                    raise CalibError("%s: no image size (add image_width / image_height to the file, or pass "
                                     "--size WxH)" % path)
                W, H = int(s.ravel()[0]), int(s.ravel()[1])
        _, T_ci = _fs_mat(fs, ("T_cam_imu", "T_cam0_imu"))
    finally:
        fs.release()
    scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[units]
    K1 = _mat(got["K1"], (3, 3), "K1")
    K2 = _mat(got["K2"], (3, 3), "K2")
    M1 = _intrinsics(K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2], W, H, "%s K1" % path)
    M2 = _intrinsics(K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2], W, H, "%s K2" % path)
    if abs(K1[0, 1]) > 1e-9 or abs(K2[0, 1]) > 1e-9:
        warns.append("the camera matrices have a skew term; it is kept")
        M1[0, 1], M2[0, 1] = K1[0, 1], K2[0, 1]
    Rm = _mat(got["R"], (3, 3), "R")
    T = _mat(got["T"], (3,), "T") * scale
    out = {"camera_width": int(W), "camera_height": int(H), "M1": M1.tolist(), "D1": _dist(got["D1"], "D1"),
           "M2": M2.tolist(), "D2": _dist(got["D2"], "D2"), "R_stereo": Rm.tolist(), "T_stereo": T.tolist(),
           "baseline_m": float(np.linalg.norm(T))}
    if T_ci is not None:
        Tci = _mat(T_ci, (4, 4), "T_cam_imu")
        Tci[:3, 3] *= scale
        out["R_cam0_imu"] = Tci[:3, :3].tolist()
        out["T_body_cam0"] = _inv_se3(Tci).tolist()
    else:
        out["R_cam0_imu"] = np.eye(3).tolist()
        out["T_body_cam0"] = np.eye(4).tolist()
    if imu:
        if T_ci is None:
            raise CalibError("--imu was given but %s has no T_cam_imu (4x4, IMU -> left camera): xclean needs the "
                             "camera-IMU extrinsic for an IMU (Kalibr's camera-IMU calibration gives it)" % path)
        out["imu_noise"], rate = _imu_block(imu, warns)
        if rate:
            out["imu_rate_hz"] = rate
    return out


# ========================================================================================================== validation
def validate(calib, warns=None):
    """Checks a calib dict as xclean will use it; raises CalibError, appends warnings.  -> the engine's Rig."""
    warns = [] if warns is None else warns
    import cv2
    for k in ("camera_width", "camera_height", "M1", "D1", "M2", "D2", "R_stereo", "T_stereo", "baseline_m"):
        if k not in calib:
            raise CalibError("calib.json is missing %s" % k)
    W, H = int(calib["camera_width"]), int(calib["camera_height"])
    if W <= 0 or H <= 0:
        raise CalibError("camera_width / camera_height must be positive")
    M1 = _mat(calib["M1"], (3, 3), "M1")
    M2 = _mat(calib["M2"], (3, 3), "M2")
    _intrinsics(M1[0, 0], M1[1, 1], M1[0, 2], M1[1, 2], W, H, "M1")
    _intrinsics(M2[0, 0], M2[1, 1], M2[0, 2], M2[1, 2], W, H, "M2")
    D1, D2 = _dist(calib["D1"], "D1"), _dist(calib["D2"], "D2")
    R = _mat(calib["R_stereo"], (3, 3), "R_stereo")
    _check_rotation(R, "R_stereo")
    T = _mat(calib["T_stereo"], (3,), "T_stereo")
    B = float(np.linalg.norm(T))
    lo, hi = BASELINE_RANGE_M
    if B > hi:
        guess = "millimetres" if B > 50 else "centimetres"
        raise CalibError("the stereo baseline is %.4g 'metres': T looks like it is in %s (use --units %s, or convert "
                         "T_stereo to metres)" % (B, guess, "mm" if guess == "millimetres" else "cm"))
    if B < lo:
        raise CalibError("the stereo baseline is %.4g m, too small for stereo (wrong units, or the same camera twice?)"
                         % B)
    if abs(abs(float(calib["baseline_m"])) - B) > 1e-6 * max(1.0, B):
        raise CalibError("baseline_m (%r) differs from the length of T_stereo (%.9g m)" % (calib["baseline_m"], B))
    ang = _angle_deg(R)
    if ang > MAX_PAIR_ROTATION_DEG:
        raise CalibError("the two cameras are rotated %.1f deg relative to each other; xclean needs a roughly parallel "
                         "side-by-side stereo pair" % ang)
    try:
        R1, R2, P1, P2, _Q, roi1, roi2 = cv2.stereoRectify(M1, np.array(D1), M2, np.array(D2), (W, H), R,
                                                          T.reshape(3, 1), alpha=0)
    except cv2.error as e:
        raise CalibError("OpenCV cannot rectify this pair: %s" % str(e).strip().splitlines()[-1])
    if not (np.all(np.isfinite(P1)) and np.all(np.isfinite(P2)) and P1[0, 0] > 0):
        raise CalibError("rectification produced invalid projection matrices (check the intrinsics and R, T)")
    if abs(P2[1, 3]) > abs(P2[0, 3]):
        raise CalibError("this is a vertical stereo pair (one camera above the other); xclean needs the second camera "
                         "to the right of the first")
    if P2[0, 3] > 0:
        raise CalibError("the right camera lies to the LEFT of the left camera (T_stereo x is positive): swap the two "
                         "cameras (left = the first, right = the second)")
    area = min(roi1[2] * roi1[3], roi2[2] * roi2[3]) / float(W * H)
    if area < 0.5:
        warns.append("after rectification only %.0f %% of the image is valid; check the calibration" % (100 * area))
    for k in ("R_cam0_imu",):
        if k in calib:
            _check_rotation(_mat(calib[k], (3, 3), k), k)
    if "T_body_cam0" in calib:
        Tb = _mat(calib["T_body_cam0"], (4, 4), "T_body_cam0")
        _check_rotation(Tb[:3, :3], "T_body_cam0 rotation")
        if np.abs(Tb[3] - np.array([0, 0, 0, 1.0])).max() > 1e-9:
            raise CalibError("T_body_cam0's last row must be [0, 0, 0, 1]")
        if np.linalg.norm(Tb[:3, 3]) > 1.0:
            warns.append("the camera-IMU distance is %.3g m: is the translation in metres?" % np.linalg.norm(Tb[:3, 3]))
    if "imu_noise" in calib:
        n = calib["imu_noise"]
        for k in NOISE_KEYS:
            if k not in (n or {}):
                raise CalibError("imu_noise is missing %s" % k)
            v = float(n[k])
            if not (v > 0 and math.isfinite(v)):
                raise CalibError("imu_noise %s must be positive" % k)
            a, b = NOISE_RANGE[k]
            if not a <= v <= b:
                warns.append("imu_noise %s = %g is outside the usual range %g..%g: continuous-time densities are "
                             "expected (rad/s/sqrt(Hz), m/s^2/sqrt(Hz) and their random walks)" % (k, v, a, b))
    from xclean.core.derive import Rig
    return Rig.from_calib_dict(calib, name=calib.get("dataset") or "calib")


def summary_line(calib, rig):
    s = "%dx%d, baseline %.4f m, rectified focal %.1f px, f*B %.2f px*m (depth %.1f m at 1 px disparity)" % (
        calib["camera_width"], calib["camera_height"], float(calib["baseline_m"]), rig.fx_rect, rig.fB_m_px,
        rig.fB_m_px)
    if calib.get("imu_noise"):
        s += ", IMU: yes%s" % (" (%g Hz)" % calib["imu_rate_hz"] if calib.get("imu_rate_hz") else "")
    else:
        s += ", IMU: no (vision-only)"
    return s


# ================================================================================================================= CLI
def _size(s):
    try:
        w, h = (int(x) for x in s.lower().split("x"))
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError("expected WxH, e.g. 1280x720")


def _write(out_path, calib, mode, defaults=None):
    """Write calib (+ defaults for keys neither it nor a kept file has).  -> (written dict, keys kept from the file)."""
    if os.path.exists(out_path) and mode is None:
        raise CalibError("%s exists: pass --force to replace it, or --update to change only its calibration fields"
                         % out_path)
    final = dict(defaults or {})
    final.update(calib)
    kept = []
    if mode == "update" and os.path.exists(out_path):
        try:
            old = json.load(open(out_path, encoding="utf-8"))
        except ValueError as e:
            raise CalibError("%s is not valid JSON (%s); use --force to replace it" % (out_path, e))
        if (old.get("camera_width"), old.get("camera_height")) not in ((None, None),
                                                                      (calib["camera_width"], calib["camera_height"])):
            raise CalibError("%s is for %sx%s images, the new calibration for %dx%d: the images in that folder would "
                             "not match" % (out_path, old.get("camera_width"), old.get("camera_height"),
                                            calib["camera_width"], calib["camera_height"]))
        final = {k: v for k, v in old.items() if k not in CALIB_KEYS}
        kept = [k for k in final if k not in calib]
        final.update(calib)
        for k, v in (defaults or {}).items():
            final.setdefault(k, v)
    d = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(d, exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
        f.write("\n")
    os.replace(tmp, out_path)
    return final, kept


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "capture":              # image pairs from a stereo rig (xclean/live/sources.py)
        from xclean.live.sources import capture_main
        return capture_main(args[1:])
    ap = argparse.ArgumentParser(description="Write calib.json for your own stereo camera (docs/SEQUENCE_FORMAT.md).",
                                 epilog="xclean calib <kalibr|opencv|check|capture> --help shows the options of "
                                        "each.")
    sub = ap.add_subparsers(dest="fmt", metavar="{kalibr,opencv,check,capture}")
    sub.add_parser("capture", add_help=False,
                   help="save stereo image pairs from a rig, for a chessboard calibration (docs/ANY_STEREO_LIVE.md)")

    def common(p):
        p.add_argument("--out", required=True, help="the calib.json to write")
        p.add_argument("--imu", default=None, help="Kalibr imu.yaml: the IMU noise block and rate")
        p.add_argument("--fps", type=float, default=None, help="camera frame rate (writes camera_fps)")
        p.add_argument("--name", default=None, help="a label for the dataset field (default: the output folder name)")
        p.add_argument("--source", default=None, help="the source label (default: 'own camera (<format> import)')")
        g = p.add_mutually_exclusive_group()
        g.add_argument("--force", action="store_true", help="replace an existing file")
        g.add_argument("--update", action="store_true",
                       help="replace only the calibration fields of an existing calib.json, keep the others")

    pk = sub.add_parser("kalibr", help="from a Kalibr camchain(-imucam).yaml")
    pk.add_argument("camchain")
    pk.add_argument("--left", default="cam0", help="the left camera's name in the camchain (default cam0)")
    pk.add_argument("--right", default="cam1", help="the right camera's name in the camchain (default cam1)")
    common(pk)
    po = sub.add_parser("opencv", help="from an OpenCV FileStorage file (cv2.stereoCalibrate results)")
    po.add_argument("file")
    po.add_argument("--units", choices=("m", "cm", "mm"), default="m", help="the unit of T (default m)")
    po.add_argument("--size", type=_size, default=None, help="image size WxH when the file has none")
    common(po)
    pc = sub.add_parser("check", help="validate an existing calib.json")
    pc.add_argument("calib")
    a = ap.parse_args(args)
    if not a.fmt:
        ap.print_help()
        return 2
    warns = []
    try:
        if a.fmt == "check":
            try:
                calib = json.load(open(a.calib, encoding="utf-8"))
            except FileNotFoundError:
                raise CalibError("file not found: %s" % a.calib)
            except ValueError as e:
                raise CalibError("%s is not valid JSON: %s" % (a.calib, e))
            rig = validate(calib, warns)
            for w in warns:
                print("xclean calib: warning: %s" % w)
            print("xclean calib: %s is usable: %s" % (a.calib, summary_line(calib, rig)))
            return 0
        if a.fmt == "kalibr":
            calib = from_kalibr(a.camchain, imu=a.imu, left=a.left, right=a.right, warns=warns)
        else:
            calib = from_opencv(a.file, units=a.units, size=a.size, imu=a.imu, warns=warns)
        # labels: given ones always win; the defaults fill only what neither they nor a kept file (--update) has
        defaults = {"source": "own camera (%s import)" % ("Kalibr" if a.fmt == "kalibr" else "OpenCV"),
                    "dataset": os.path.basename(os.path.dirname(os.path.abspath(a.out)))}
        head = {}
        if a.source:
            head["source"] = a.source
        if a.name:
            head["dataset"] = a.name
        if a.fps is not None:
            if not a.fps > 0:
                raise CalibError("--fps must be positive")
            head["camera_fps"] = a.fps
        calib = dict(head, **calib)
        rig = validate(dict(defaults, **calib), warns)
        final, kept = _write(a.out, calib, "force" if a.force else "update" if a.update else None, defaults)
    except CalibError as e:
        print("xclean calib: error: %s" % e, file=sys.stderr)
        return 2
    for w in warns:
        print("xclean calib: warning: %s" % w)
    print("xclean calib: wrote %s: %s" % (a.out, summary_line(final, rig)))
    if kept:
        print("xclean calib: kept from the existing file: %s" % ", ".join(kept))
    if "t0_ns" not in final:
        print("xclean calib: note: scoring against ground truth also needs t0_ns (the absolute time of your first "
              "frame, in ns) in calib.json; a converter or recorder writes it")
    print("xclean calib: next: put calib.json in your sequence folder and check it with: xclean audit <folder>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
