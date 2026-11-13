# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Any stereo rig (0.3.0): the frame-source interface of `xclean live --source` (xclean/live/sources.py).

The live hardware test of xclean exists only for the OAK-D-Lite.  These tests check the source path without a
camera: recordings sent through it must give the poses of `xclean live --replay` and `xclean replay`; simulated
cameras (OpenCV captures and a plugin that makes frames) check the live clock, the built-in sources, the errors and
`xclean doctor --source`.
"""
import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
W, H = 160, 120


def _calib(fps=20.0, imu=True):
    f, B = 150.0, 0.1
    c = {"camera_width": W, "camera_height": H,
         "M1": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], "D1": [0.0] * 5,
         "M2": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], "D2": [0.0] * 5,
         "R_stereo": np.eye(3).tolist(), "T_stereo": [-B, 0.0, 0.0], "baseline_m": B,
         "T_body_cam0": np.eye(4).tolist(), "R_cam0_imu": np.eye(3).tolist(), "camera_fps": fps}
    if imu:
        c.update(imu_rate_hz=200.0, imu_noise={"gyroscope_noise_density": 1e-4, "gyroscope_random_walk": 5e-6,
                                               "accelerometer_noise_density": 1e-3,
                                               "accelerometer_random_walk": 2e-4})
    return c


def _texture(seed=0, w=W + 400, h=H + 40):
    import cv2
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (0, 0), 1.5)


def _pair(tex, i, step=2):
    """Frame i of a camera moving sideways over a textured plane: left and right differ by 4 px of disparity."""
    x = 20 + step * i
    left = np.ascontiguousarray(tex[20:20 + H, x:x + W])
    right = np.ascontiguousarray(tex[20:20 + H, x + 4:x + 4 + W])
    return left, right


def _sequence(root, n=12, imu=True, fps=20.0):
    """A sequence folder: n frames at fps; with imu, 200 Hz samples from 25 ms before the first frame to 30 ms after
    the last (a still IMU: gravity on -y, small noise)."""
    import cv2
    os.makedirs(os.path.join(root, "left"))
    os.makedirs(os.path.join(root, "right"))
    cal = _calib(fps, imu)
    cal.update(num_frames=n, t0_ns=0)
    json.dump(cal, open(os.path.join(root, "calib.json"), "w"))
    tex = _texture()
    with open(os.path.join(root, "timestamps.csv"), "w") as f:
        f.write("frame_id,timestamp_s,left_t,right_t\n")
        for i in range(n):
            l, r = _pair(tex, i)
            cv2.imwrite(os.path.join(root, "left", "%06d.png" % i), l)
            cv2.imwrite(os.path.join(root, "right", "%06d.png" % i), r)
            t = i / fps
            f.write("%d,%.6f,%.6f,%.6f\n" % (i, t, t, t + 1.7e-5))
    if imu:
        rng = np.random.default_rng(1)
        with open(os.path.join(root, "imu.jsonl"), "w") as f:
            for k in range(int(((n - 1) / fps + 0.055) * 200)):
                t = -0.025 + k / 200.0
                g = rng.normal(0, 1e-3, 3)
                a = np.array([0.0, -9.81, 0.0]) + rng.normal(0, 1e-2, 3)
                f.write(json.dumps({"t": round(t, 6), "gx": g[0], "gy": g[1], "gz": g[2],
                                    "ax": a[0], "ay": a[1], "az": a[2]}) + "\n")
    return root


# ------------------------------------------------------------------------------------------------ loading sources
def test_parse_options():
    """Every value arrives as a string; the built-in sources convert their own options."""
    from xclean.live.sources import SourceError, parse_options
    assert parse_options(["device=0", "swap=true", "path=C:\\data\\x", "serial=0012", "pair=0,1", "x=1.5"]) == {
        "device": "0", "swap": "true", "path": "C:\\data\\x", "serial": "0012", "pair": "0,1", "x": "1.5"}
    with pytest.raises(SourceError, match="KEY=VALUE"):
        parse_options(["device"])
    with pytest.raises(SourceError, match="not a valid option name"):
        parse_options(["1x=2"])


def test_load_source_by_name_module_and_file(tmp_path):
    from xclean.live import sources as S
    assert S.load_source("uvc-sbs") is S.UvcSideBySide and S.load_source("folder") is S.SequenceFolderSource
    assert S.load_source("xclean.live.sources:UvcPair") is S.UvcPair
    f = tmp_path / "my_rig.py"
    f.write_text("from xclean.live.sources import StereoSource\n"
                 "class MyRig(StereoSource):\n"
                 "    name = 'my-rig'\n"
                 "    def __init__(self, serial=None):\n"
                 "        self.serial = serial\n"
                 "    def frames(self):\n"
                 "        return iter(())\n"
                 "class NotASource:\n"
                 "    pass\n")
    cls = S.load_source("%s:MyRig" % f)
    assert cls.name == "my-rig" and S.make_source("%s:MyRig" % f, {"serial": 7}).serial == 7
    for spec, msg in (("nonsense", "unknown source"), ("%s:Missing" % f, "has no Missing"),
                      ("%s:NotASource" % f, "not a frame source"), (str(tmp_path / "none.py") + ":X", "does not exist"),
                      ("no_such_module_xyz:Cls", "cannot be imported"), ("C:\\x\\rig.py", "names no class")):
        with pytest.raises(S.SourceError, match=msg):
            S.load_source(spec)
    with pytest.raises(S.SourceError, match="does not take these options"):
        S.make_source("%s:MyRig" % f, {"colour": 1})


def test_images_are_checked_and_converted():
    import cv2
    from xclean.live.sources import SourceError, to_gray
    bgr = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    assert np.array_equal(to_gray(bgr, W, H, "left"), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    g = to_gray(bgr[:, :, 0], W, H, "left")
    assert g.flags["C_CONTIGUOUS"] and g.shape == (H, W)
    with pytest.raises(SourceError, match="the right image is 120 x 160, but calib.json says 160 x 120"):
        to_gray(np.zeros((W, H), np.uint8), W, H, "right")
    with pytest.raises(SourceError, match="8-bit"):
        to_gray(np.zeros((H, W), np.uint16), W, H, "left")


# ------------------------------------------------------------------------------- the source loop, without the engine
class _Consumer(threading.Thread):
    """Takes every released frame from the slot as the engine does and reports it done (lock-step); drains the IMU
    queue.  got: [(index, t, last IMU stamp, left bytes, right bytes)], imu: [(t, g, a)]."""

    def __init__(self, slot, ctl, imu_q):
        super().__init__(daemon=True)
        self.slot, self.ctl, self.imu_q = slot, ctl, imu_q
        self.got, self.imu = [], []

    def run(self):
        last = 0
        while True:
            self.ctl["new_frame"].wait(0.2)
            self.ctl["new_frame"].clear()
            while True:
                try:
                    self.imu.append(self.imu_q.get_nowait())
                except queue.Empty:
                    break
            with self.ctl["lock"]:
                seq, eos = int(self.slot.hdr[0]), bool(self.slot.hdr[4])
                fresh = seq != last and self.slot.hdr[3] == 0
                if fresh:
                    s = self.slot
                    self.got.append((int(s.hdr[1]), float(s.tstamp[0]), float(s.tstamp[1]), s.buf[0].tobytes(),
                                     s.buf[1].tobytes()))
                    s.hdr[3] = 1
                    last = seq
            if fresh:
                self.ctl["done"].value = self.got[-1][0]
                self.ctl["done_evt"].set()
            if eos and not fresh:
                return


def _run_loop(backend, calib, mode, tmp_path, tag, ready_after_s=0.0):
    from xclean.live import oak_source
    from xclean.live.protocol import FrameSlot

    class _V:
        value = -1
    name = "xclean_test_%s_%d" % (tag, os.getpid())
    slot = FrameSlot(name, int(calib["camera_width"]), int(calib["camera_height"]), create=True)
    ctl = {"ready": threading.Event(), "new_frame": threading.Event(), "stop": threading.Event(),
           "lock": threading.Lock(), "done_evt": threading.Event(), "done": _V()}
    imu_q = queue.Queue()
    cons = _Consumer(slot, ctl, imu_q)
    cons.start()
    if ready_after_s:
        threading.Timer(ready_after_s, ctl["ready"].set).start()
    else:
        ctl["ready"].set()
    try:
        log = oak_source.run_source(backend, {"mode": mode, "fps": float(calib["camera_fps"])}, slot, imu_q, ctl,
                                    str(tmp_path / ("%s.json" % tag)))
    finally:
        with ctl["lock"]:
            slot.hdr[4] = 1                                   # end of stream, also when run_source raised
        ctl["new_frame"].set()
        cons.join(10)
        slot.close(unlink=True)
    return log, cons


@pytest.mark.parametrize("mode", ["lockstep", "paced"])
def test_folder_source_releases_exactly_what_replay_releases(tmp_path, mode):
    """The source path (SourceBackend + SequenceFolderSource) gives the engine the same frames, stamps, images and IMU
    packets as `xclean live --replay` (ReplayBackend)."""
    from xclean.live.oak_source import ReplayBackend
    from xclean.live.sources import SequenceFolderSource, SourceBackend
    seq = _sequence(str(tmp_path / "seq"), n=10)
    calib = json.load(open(os.path.join(seq, "calib.json")))
    log_r, c_r = _run_loop(ReplayBackend(seq, mode), calib, mode, tmp_path, "replay_" + mode)
    sb = SourceBackend(SequenceFolderSource(device=seq), mode)
    assert sb.open(None) == calib
    log_s, c_s = _run_loop(sb, calib, mode, tmp_path, "source_" + mode)
    assert len(c_r.got) == 10 and c_s.got == c_r.got
    imu_r = [(t.tolist(), g.tolist(), a.tolist()) for t, g, a in c_r.imu]
    imu_s = [(t.tolist(), g.tolist(), a.tolist()) for t, g, a in c_s.imu]
    assert imu_s == imu_r and sum(len(x[0]) for x in imu_r) == log_r["imu_released"]
    assert [f[:2] for f in log_s["frames"]] == [f[:2] for f in log_r["frames"]]
    assert log_s["source"] == "source" and log_s["backend"]["source"] == "folder" and log_s["backend"]["closed"]
    assert log_s["imu_released"] == log_r["imu_released"] == sum(1 for _ in open(os.path.join(seq, "imu.jsonl")))


class _FakeCamera:
    """A live source: 30 Hz frames on a clock that starts at 1e9 s (an epoch-like clock), with 200 Hz IMU samples
    that begin 0.1 s before the first frame; paced in real time."""
    name = "fake-camera"
    live = True
    has_imu = True
    hardware_sync = True

    def __init__(self, n=30, fps=30.0, t0=1.0e9, fail_at=None, size=(W, H)):
        self.n, self.fps, self.t0, self.fail_at, self.size = n, fps, t0, fail_at, size
        self.closed = 0

    def open(self, calib):
        return None

    def frames(self):
        from xclean.live.sources import ImuSamples, StereoFrame
        tex = _texture(3)
        ti = self.t0 - 0.1
        for i in range(self.n):
            t = self.t0 + i / self.fps
            ts = np.arange(ti, t + 1e-9, 0.005)
            if len(ts):
                yield ImuSamples(ts, np.zeros((len(ts), 3)), np.tile([0.0, -9.81, 0.0], (len(ts), 1)))
                ti = ts[-1] + 0.005
            if i == self.fail_at:
                raise ValueError("the cable came out")
            l, r = _pair(tex, i)
            if self.size != (W, H):
                l = r = np.zeros(self.size[::-1], np.uint8)
            yield StereoFrame(t, np.dstack([l, l, l]), r)        # colour left image: converted to grey
            time.sleep(1.0 / self.fps)

    def close(self):
        self.closed += 1


def test_live_source_clock_starts_at_the_first_pair(tmp_path):
    from xclean.live.sources import SourceBackend
    src = _FakeCamera(n=20)
    sb = SourceBackend(src, "paced")
    calib = sb.open(_calib(30.0))
    log, cons = _run_loop(sb, calib, "paced", tmp_path, "fakecam", ready_after_s=0.2)
    assert src.closed == 1 and log["backend"]["closed"]
    t = [g[1] for g in cons.got]
    k = np.diff(t) * 30.0
    assert t[0] == 0.0 and np.all(k > 0.5) and np.abs(k - np.round(k)).max() < 1e-4   # rebased, integer us
    assert log["pre_ready_frames"] > 0                         # frames during the engine's warm-up are dropped
    stamps = np.concatenate([x[0] for x in cons.imu])
    assert stamps.min() >= 0.0                                 # IMU before the first pair is not part of the run
    assert log["imu_released"] + log["imu_pre_session"] + log["pre_ready_imu"] <= 20 * 7 + 21
    assert log["t0_device_us"] > 10 ** 15                     # the origin is on the source's own clock
    with pytest.raises(ValueError, match="lockstep needs a replay source"):
        _run_loop(SourceBackend(_FakeCamera(n=2), "lockstep"), calib, "lockstep", tmp_path, "fakecam_ls")


@pytest.mark.parametrize("fail", ["raise", "size"])
def test_a_source_error_ends_the_stream_with_a_plain_message(tmp_path, fail):
    from xclean.live.sources import SourceBackend
    src = _FakeCamera(n=10, fail_at=3) if fail == "raise" else _FakeCamera(n=10, size=(320, 240))
    sb = SourceBackend(src, "paced")
    calib = sb.open(_calib(30.0))
    with pytest.raises(SystemExit) as e:
        _run_loop(sb, calib, "paced", tmp_path, "fail_" + fail)
    msg = str(e.value)
    assert msg.startswith("xclean: error: the source fake-camera stopped with an error: ")
    if fail == "raise":
        assert "ValueError: the cable came out (test_any_stereo_live.py, line" in msg
    else:
        assert "the left image is 320 x 240, but calib.json says 160 x 120" in msg
    assert src.closed == 1


# ------------------------------------------------------------------------------------- built-in OpenCV sources
class _FakeCapture:
    """cv2.VideoCapture stand-in: frames of the size that was set, left half / right half marked 10 / 200."""
    opened = []

    def __init__(self, dev, api=None):
        self.dev, self.props, self.n = dev, {}, 0
        _FakeCapture.opened.append(self)

    def isOpened(self):
        return self.dev != 99

    def set(self, prop, v):
        self.props[prop] = v
        return True

    def _img(self):
        import cv2
        w = int(self.props.get(cv2.CAP_PROP_FRAME_WIDTH, 640))
        h = int(self.props.get(cv2.CAP_PROP_FRAME_HEIGHT, 480))
        img = np.full((h, w, 3), 10 if self.dev in (0, "0") else 200, np.uint8)
        if w == 2 * W:
            img[:, w // 2:] = 200
        return img

    def read(self):
        return True, self._img()

    def grab(self):
        self.n += 1
        return self.n < 50

    def retrieve(self):
        return True, self._img()

    def release(self):
        self.released = True


@pytest.fixture
def fake_cv(monkeypatch):
    import cv2
    _FakeCapture.opened = []
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    return _FakeCapture


def test_uvc_side_by_side_splits_and_swaps(fake_cv):
    import cv2
    from xclean.live import sources as S
    for swap, lv, rv in (("false", 10, 200), ("true", 200, 10)):         # as --source-opt gives them: strings
        src = S.make_source("uvc-sbs", S.parse_options(["device=0", "swap=%s" % swap, "fps=30"]))
        calib, notes = S.finish_calib(src, _calib(30.0, imu=False))
        cap = fake_cv.opened[-1]
        assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 2 * W and cap.props[cv2.CAP_PROP_FPS] == 30.0
        it = next(iter(src.frames()))
        assert it.left.shape == (H, W, 3) and it.left[0, 0, 0] == lv and it.right[0, 0, 0] == rv
        src.close()
        assert cap.released
    with pytest.raises(S.SourceError, match="needs --calib"):
        S.make_source("uvc-sbs", {}).open(None)
    with pytest.raises(S.SourceError, match="OpenCV cannot open camera 99"):
        S.make_source("uvc-sbs", {"device": 99}).open(_calib(imu=False))


def test_uvc_side_by_side_refuses_a_frame_of_the_wrong_size(fake_cv, monkeypatch):
    from xclean.live import sources as S
    monkeypatch.setattr(_FakeCapture, "_img", lambda self: np.zeros((480, 640, 3), np.uint8))
    with pytest.raises(S.SourceError, match="gives 640 x 480 frames, but --source uvc-sbs with this calib.json "
                                            r"\(160 x 120 per camera\) needs 320 x 120"):
        S.make_source("uvc-sbs", {"device": 0}).open(_calib(imu=False))
    assert fake_cv.opened[-1].released                        # closed before the error


def test_uvc_pair_is_marked_unsynchronised(fake_cv):
    from xclean.live import sources as S
    src = S.make_source("uvc-pair", {"device": "0,1"})
    assert src.hardware_sync is False
    S.finish_calib(src, _calib(imu=False))
    frames = []
    for it in src.frames():
        frames.append(it)
        if len(frames) == 5:
            break
    src.close()
    assert frames[0].left[0, 0, 0] == 10 and frames[0].right[0, 0, 0] == 200
    info = src.info()
    assert info["hardware_sync"] is False and info["arrival_gap_ms"]["n"] == 5
    with pytest.raises(S.SourceError, match="two cameras"):
        S.make_source("uvc-pair", {"device": "0"})


# ------------------------------------------------------------------------------------------------ xclean doctor
def test_doctor_checks_a_source(tmp_path, fake_cv):
    from xclean import doctor
    seq = _sequence(str(tmp_path / "seq"), n=30)
    res = doctor.check_source("folder", device=seq, seconds=1.0)
    st = {(c.name, c.status) for c in res}
    assert ("Stereo source", "OK") in st and ("Calibration", "OK") in st and ("Stereo frames", "OK") in st
    assert ("IMU samples", "OK") in st, [c.as_dict() for c in res]
    cj = tmp_path / "cam.json"
    json.dump(_calib(30.0, imu=False), open(cj, "w"))
    res = doctor.check_source("uvc-pair", device="0,1", calib_path=str(cj), seconds=0.2)
    head = res[0]
    assert head.status == "WARN" and "NOT hardware-synchronised" in head.detail and "arrival gap" in head.detail
    res = doctor.check_source("uvc-sbs", device="0", seconds=0.2)
    assert res[-1].status == "FAIL" and "needs --calib" in res[-1].detail
    res = doctor.check_source("bogus")
    assert res[0].status == "FAIL" and "unknown source" in res[0].detail
    # the doctor command: --source replaces the OAK check; --device without --source is refused
    out = subprocess.run([sys.executable, "-m", "xclean.cli", "doctor", "--source", "folder", "--device", seq,
                          "--seconds", "0.5", "--json"], capture_output=True, text=True, timeout=300, env=ENV)
    names = [c["name"] for c in json.loads(out.stdout)]
    assert "Stereo frames" in names and "OAK camera" not in names
    out = subprocess.run([sys.executable, "-m", "xclean.cli", "doctor", "--device", "0"], capture_output=True,
                         text=True, timeout=120, env=ENV)
    assert out.returncode == 2 and "need --source" in out.stderr


# ------------------------------------------------------------------------------------ xclean live command line
@pytest.mark.parametrize("argv, msg", [
    (["--source", "uvc-sbs"], "--source uvc-sbs needs --calib"),
    (["--source", "uvc-sbs", "--calib", "c.json", "--mode", "lockstep"], "--mode lockstep needs --replay or a "
                                                                          "recording source"),
    (["--source", "folder", "--replay", "x"], "--replay replays a recording made with --record"),
    (["--device", "0"], "--device and --source-opt are options of the other sources"),
    (["--source", "nonsense"], "unknown source 'nonsense'"),
    (["--source", "uvc-pair", "--calib", "c.json", "--mxid", "123"], "--mxid and --ref-calib are options of --source "
                                                                      "oak"),
    (["--source", "uvc-sbs", "--calib", "c.json", "--frames", "5"], "--frames works with --replay and --source folder"),
    (["--source", "folder"], "--source folder needs --device"),
    (["--source", "uvc-sbs", "--calib", "missing.json"], "--calib missing.json cannot be read"),
    (["--mode", "lockstep"], "--mode lockstep needs --replay (a camera cannot wait for the engine)"),
])
def test_live_refuses_bad_source_options(tmp_path, monkeypatch, capsys, argv, msg):
    from xclean.live import oak_demo
    monkeypatch.chdir(tmp_path)
    json.dump(_calib(imu=False), open(tmp_path / "c.json", "w"))         # a readable --calib for the cases that use it
    monkeypatch.setattr(sys, "argv", ["xclean live", "--out", str(tmp_path / "run")] + argv)
    assert oak_demo.main() == 2
    assert msg in capsys.readouterr().err
    assert not os.path.exists(tmp_path / "run")               # refused before anything was written


def test_run_live_builds_the_command_line(monkeypatch):
    from xclean import cli
    from xclean.live import sources as S
    seen = {}
    monkeypatch.setattr(cli, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    S.run_live(S.UvcPair, "runs/rig1", calib="c.json", options={"device": "0,1", "fps": 15},
               args=["--duration", 5])
    assert seen["argv"] == ["live", "--source", "xclean.live.sources:UvcPair", "--out", "runs/rig1", "--calib", "c.json",
                            "--source-opt", "device=0,1", "--source-opt", "fps=15", "--duration", "5"]


# ------------------------------------------------------------------------- whole runs of the live engine (CPU)
PLUGIN = '''
"""A user's own source, written as docs/ANY_STEREO_LIVE.md shows: a recording read with plain OpenCV and json."""
import json
import os

import cv2
import numpy as np

from xclean.live.sources import ImuSamples, StereoFrame, StereoSource


class FolderRig(StereoSource):
    name = "folder-rig"
    live = False                       # a recording: xclean may pace it or run it lock-step

    def __init__(self, path):
        self.path = path
        self.has_imu = os.path.isfile(os.path.join(path, "imu.jsonl"))

    def open(self, calib):
        return json.load(open(os.path.join(self.path, "calib.json")))

    def frames(self):
        imu = [json.loads(x) for x in open(os.path.join(self.path, "imu.jsonl"))] if self.has_imu else []
        rows = [ln.strip().split(",") for ln in open(os.path.join(self.path, "timestamps.csv")).readlines()[1:]]
        k = 0
        for row in rows:
            i, t = int(row[0]), float(row[1])
            new = []
            while k < len(imu) and imu[k]["t"] <= t:
                new.append(imu[k])
                k += 1
            if new:
                yield ImuSamples([s["t"] for s in new], [[s["gx"], s["gy"], s["gz"]] for s in new],
                                 [[s["ax"], s["ay"], s["az"]] for s in new])
            left = cv2.imread(os.path.join(self.path, "left", "%06d.png" % i), cv2.IMREAD_GRAYSCALE)
            right = cv2.imread(os.path.join(self.path, "right", "%06d.png" % i), cv2.IMREAD_GRAYSCALE)
            yield StereoFrame(t, left, right, t_right=float(row[3]), index=i)
        if k < len(imu):
            rest = imu[k:]
            yield ImuSamples([s["t"] for s in rest], [[s["gx"], s["gy"], s["gz"]] for s in rest],
                             [[s["ax"], s["ay"], s["az"]] for s in rest])

    def close(self):
        pass


class SyntheticCamera(StereoSource):
    """A live camera stand-in: frames of a textured plane at 20 Hz, made in real time, no IMU."""
    name = "synthetic-camera"
    live = True

    def open(self, calib):
        rng = np.random.default_rng(0)
        self.tex = cv2.GaussianBlur((rng.random((160, 900)) * 255).astype(np.uint8), (0, 0), 1.5)
        self.W, self.H = int(calib["camera_width"]), int(calib["camera_height"])

    def frames(self):
        import time
        t0 = time.perf_counter()
        for i in range(10 ** 6):
            due = t0 + i / 20.0
            while time.perf_counter() < due:
                time.sleep(0.002)
            x = 20 + (i % 300)
            yield StereoFrame(time.perf_counter(), self.tex[20:20 + self.H, x:x + self.W],
                              self.tex[20:20 + self.H, x + 4:x + 4 + self.W])

    def close(self):
        pass
'''


def _poses(run):
    rows = [ln.strip().split(",") for ln in open(os.path.join(run, "trajectory_live.csv")) if not ln.startswith("#")]
    return [r[:10] for r in rows]                 # t_frame, position, quaternion, frame_index, state (not timing)


def _xclean(args, timeout=600):
    r = subprocess.run([sys.executable, "-m", "xclean.cli"] + args, capture_output=True, text=True, timeout=timeout,
                       env=ENV)
    assert r.returncode == 0, (args, r.stdout[-3000:], r.stderr[-3000:])
    return r


def test_recording_through_the_source_path_gives_the_poses_of_replay(tmp_path):
    """Lock-step, the published engine settings: `xclean replay`, `xclean live --replay`, `xclean live --source
    folder` and a user's own source file give the same pose for every frame."""
    seq = _sequence(str(tmp_path / "seq"), n=16)
    plugin = tmp_path / "my_rig.py"
    plugin.write_text(PLUGIN)
    common = ["--mode", "lockstep", "--env", "XCLEAN_P13_OFF=loop"]
    live = common + ["--view", "none", "--tag", seq]            # replay's engine tag is its folder argument
    _xclean(["replay", seq, "--out", str(tmp_path / "replay")] + common)
    _xclean(["live", "--replay", seq, "--out", str(tmp_path / "live_replay")] + live)
    r = _xclean(["live", "--source", "folder", "--device", seq, "--out", str(tmp_path / "folder")] + live)
    assert "xclean: source: folder (a recording; IMU samples: yes)" in r.stdout
    _xclean(["live", "--source", "%s:FolderRig" % plugin, "--source-opt", "path=%s" % seq,
             "--out", str(tmp_path / "plugin")] + live)
    ref = _poses(tmp_path / "replay")
    assert len(ref) == 16
    for run in ("live_replay", "folder", "plugin"):
        assert _poses(tmp_path / run) == ref, run
    demo = json.load(open(tmp_path / "plugin" / "demo.json"))
    assert demo["source"]["name"] == "folder-rig" and demo["source"]["has_imu"] is True


def test_live_camera_source_runs_and_records(tmp_path):
    """A simulated live camera through a user's source file: the run ends after --duration, the poses are written,
    and --record writes a recording that `xclean live --replay` accepts."""
    plugin = tmp_path / "cam.py"
    plugin.write_text(PLUGIN)
    cj = tmp_path / "cam.json"
    json.dump(_calib(20.0, imu=False), open(cj, "w"))
    run, rec = tmp_path / "run", tmp_path / "rec"
    r = _xclean(["live", "--source", "%s:SyntheticCamera" % plugin, "--calib", str(cj), "--out", str(run),
                 "--record", str(rec), "--duration", "3", "--view", "none", "--env", "XCLEAN_P13_OFF=loop"])
    assert "xclean: vision only: the source synthetic-camera gives no IMU samples." in r.stdout
    assert "we tested live runs only on an OAK-D-Lite" in r.stdout
    poses = _poses(run)
    feeder = json.load(open(run / "feeder.json"))
    assert len(poses) >= 5 and feeder["source"] == "source" and feeder["backend"]["closed"]
    assert feeder["frames"][0][1] == 0.0                         # the session clock starts at the first pair
    from xclean.io.folders import sequence_problem
    assert sequence_problem(str(rec)) is None
    assert sum(1 for _ in open(rec / "timestamps.csv")) - 1 == len(feeder["frames"])


# --------------------------------------------------------------------------------------- calibration (the guide)
def _doc_script(marker):
    doc = open(os.path.join(os.path.dirname(__file__), "..", "docs", "ANY_STEREO_LIVE.md"), encoding="utf-8").read()
    blocks = doc.split("```python\n")[1:]
    code = [b.split("```")[0] for b in blocks if b.startswith(marker)]
    assert len(code) == 1, marker
    return code[0]


def _render_chessboard_pairs(root, n=24, cols=9, rows=6, square=0.03, baseline=0.06, size=(640, 480)):
    """Pairs of a chessboard seen by an ideal stereo rig (fx = fy = 500, no distortion, right camera `baseline` m to
    the right), written as xclean calib capture writes them."""
    import cv2
    from scipy.spatial.transform import Rotation as Rot
    px, m = 40, 40
    tex = np.full(((rows + 1) * px + 2 * m, (cols + 1) * px + 2 * m), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                tex[m + r * px:m + (r + 1) * px, m + c * px:m + (c + 1) * px] = 0
    k = square / px
    S = np.array([[k, 0, -(m + px) * k], [0, k, -(m + px) * k], [0, 0, 1.0]])   # texture px -> board metres
    K = np.array([[500.0, 0, size[0] / 2], [0, 500.0, size[1] / 2], [0, 0, 1]])
    rng = np.random.default_rng(5)
    os.makedirs(os.path.join(root, "left"))
    os.makedirs(os.path.join(root, "right"))
    for i in range(n):
        R = Rot.from_euler("xyz", rng.uniform(-30, 30, 3), degrees=True).as_matrix()
        centre = np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.08, 0.08), rng.uniform(0.55, 0.85)])
        t = centre - R @ np.array([cols - 1, rows - 1, 0]) * square / 2       # the board centre at `centre`
        for side, dx in (("left", 0.0), ("right", -baseline)):
            Hm = K @ np.column_stack([R[:, 0], R[:, 1], t + np.array([dx, 0, 0])]) @ S
            img = cv2.warpPerspective(tex, Hm, size, flags=cv2.INTER_LINEAR, borderValue=128)
            img = np.clip(cv2.GaussianBlur(img, (0, 0), 0.7) + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(root, side, "%06d.png" % i), img)
    return root


def test_the_guides_chessboard_script_calibrates_a_rig(tmp_path):
    """docs/ANY_STEREO_LIVE.md step 1b, run as printed, on rendered pairs of a known rig; then step 1c."""
    script = tmp_path / "stereo_chessboard.py"
    script.write_text(_doc_script('"""Stereo calibration from chessboard'))
    pairs = _render_chessboard_pairs(str(tmp_path / "calib_pairs"))
    yml = tmp_path / "stereo.yml"
    r = subprocess.run([sys.executable, str(script), pairs, "9", "6", "0.03", str(yml)], capture_output=True,
                       text=True, timeout=300, env=ENV)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RMS reprojection error" in r.stdout and "usable pairs" in r.stdout
    cj = tmp_path / "my_rig.json"
    _xclean(["calib", "opencv", str(yml), "--fps", "30", "--out", str(cj)])
    r = _xclean(["calib", "check", str(cj)])
    assert "is usable" in r.stdout
    cal = json.load(open(cj))
    assert abs(cal["baseline_m"] - 0.06) < 0.0006, cal["baseline_m"]              # within 1 %
    assert abs(cal["M1"][0][0] - 500.0) < 5.0 and abs(cal["M2"][1][1] - 500.0) < 5.0
    assert cal["T_stereo"][0] < 0 and cal["camera_width"] == 640 and cal["camera_fps"] == 30


def test_calib_capture_saves_pairs(tmp_path, fake_cv):
    from xclean.io.calib_import import main
    out = tmp_path / "pairs"
    assert main(["capture", "--source", "uvc-sbs", "--device", "0", "--size", "%dx%d" % (W, H), "--out", str(out),
                 "--count", "3", "--every", "0"]) == 0
    import cv2
    left = [cv2.imread(str(out / "left" / ("%06d.png" % i)), cv2.IMREAD_GRAYSCALE) for i in range(3)]
    right = cv2.imread(str(out / "right" / "000000.png"), cv2.IMREAD_GRAYSCALE)
    assert all(x is not None and x.shape == (H, W) for x in left) and right[0, 0] > left[0][0, 0]
    assert main(["capture", "--source", "uvc-sbs", "--size", "%dx%d" % (W, H), "--out", str(out)]) == 2   # not empty


def test_an_oak_recordings_calib_goes_through_the_source_path(tmp_path):
    """calib.json of `xclean live --record` on an OAK-D-Lite stores the x part of T_stereo as baseline_m; the source
    path accepts it, as `xclean live --replay` does.  A baseline_m more than 1 % off is refused."""
    from xclean.live import sources as S
    seq = _sequence(str(tmp_path / "seq"), n=4, imu=False)
    cal = json.load(open(os.path.join(seq, "calib.json")))
    cal["T_stereo"] = [-0.1, 0.002, 0.001]                  # baseline_m stays 0.1 = |x|; the length is 0.100025
    json.dump(cal, open(os.path.join(seq, "calib.json"), "w"))
    calib, _ = S.finish_calib(S.SequenceFolderSource(device=seq), None)
    assert calib["baseline_m"] == 0.1 and calib["T_stereo"] == [-0.1, 0.002, 0.001]   # passed on unchanged
    cal["baseline_m"] = 0.105
    with pytest.raises(S.SourceError, match="by more than 1 %"):
        S.check_calib(cal)


def test_a_recording_source_stopped_before_the_go_ends_at_once(tmp_path):
    """The engine failed during start-up (or the user stopped the run) before the go: stop() returns at once and the
    source is closed, with no false 'did not return' note."""
    from xclean.live.sources import SequenceFolderSource, SourceBackend
    seq = _sequence(str(tmp_path / "seq"), n=4, imu=False)
    sb = SourceBackend(SequenceFolderSource(device=seq), "paced")
    sb.open(None)
    sb.start(queue.Queue())
    t0 = time.time()
    sb.stop()
    assert time.time() - t0 < 2.0 and sb.closed and sb.close_error is None


def test_doctor_does_not_hang_on_a_source_that_sends_nothing():
    from xclean.live import sources as S

    class Silent(S.StereoSource):
        name = "silent"

        def frames(self):
            while True:
                time.sleep(0.05)
                if False:
                    yield None

    t0 = time.time()
    r = S.probe(Silent(), _calib(30.0, imu=False), seconds=0.5, first_wait_s=1.0)
    assert time.time() - t0 < 6.0 and r["frames"] == 0 and "no stereo frame within 1 s" in r["error"]
    assert "did not return" in r["close_error"]              # stuck in frames(): not closed under the reader
