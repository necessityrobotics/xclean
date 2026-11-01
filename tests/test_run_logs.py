# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Run logs written off the frame path (xclean/live/logwriter.py): the same bytes in the same order as a direct write,
writes that never wait for the disk, and errors that are recorded instead of lost."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import types

import numpy as np
import pytest

from xclean.live import logwriter
from xclean.live.logwriter import LineWriter

LINES = [json.dumps({"frame": i, "t": 0.05 * i, "state": "tracking", "note": "café → %d" % i}) + "\n"
         for i in range(500)]


def _direct(path, lines, mode="a", **kw):
    for ln in lines:
        with open(path, mode, encoding="utf-8", **kw) as f:
            f.write(ln)


def test_same_bytes_as_a_direct_append(tmp_path):
    a, b = str(tmp_path / "direct.jsonl"), str(tmp_path / "writer.jsonl")
    _direct(a, LINES)
    w = LineWriter()
    for ln in LINES:
        w.write(b, ln)
    info = w.close()
    assert info["lines"] == len(LINES) and info["lost_lines"] == 0 and info["n_errors"] == 0 and info["finished"]
    assert open(a, "rb").read() == open(b, "rb").read()


def test_open_arguments_are_kept(tmp_path):
    """'w' truncates as the recorder's direct open did; the buffer size does not change the bytes."""
    a, b = str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")
    for p in (a, b):
        with open(p, "w") as f:
            f.write("old content\n")
    with open(a, "w", encoding="utf-8", buffering=1 << 16) as f:
        for ln in LINES:
            f.write(ln)
    w = LineWriter()
    w.open(b, "w", encoding="utf-8", buffering=1 << 16)
    for ln in LINES:
        w.write(b, ln)
    w.close()
    assert open(a, "rb").read() == open(b, "rb").read()


def test_two_files_keep_their_own_order(tmp_path):
    a, b = str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")
    w = LineWriter()
    for i, ln in enumerate(LINES):
        w.write(a if i % 3 else b, ln)
    w.close_file(b)
    w.close()
    assert open(a, encoding="utf-8").read() == "".join(ln for i, ln in enumerate(LINES) if i % 3)
    assert open(b, encoding="utf-8").read() == "".join(ln for i, ln in enumerate(LINES) if not i % 3)


class _SlowFile:
    def __init__(self, f, delay):
        self.f, self.delay = f, delay

    def write(self, s):
        time.sleep(self.delay)
        return self.f.write(s)

    def close(self):
        self.f.close()


def test_write_does_not_wait_for_a_slow_disk(tmp_path, monkeypatch):
    real_open = open
    monkeypatch.setattr(logwriter, "open", lambda *a, **k: _SlowFile(real_open(*a, **k), 0.05), raising=False)
    p = str(tmp_path / "slow.jsonl")
    w = LineWriter()
    t0 = time.perf_counter()
    for ln in LINES[:20]:                     # 20 x 50 ms = 1 s on the disk
        w.write(p, ln)
    took = time.perf_counter() - t0
    info = w.close()
    assert took < 0.2, took                   # the caller never waited for the writes
    assert info["lines"] == 20 and info["max_backlog"] >= 2
    assert open(p, encoding="utf-8").read() == "".join(LINES[:20])


def test_errors_are_recorded_and_counted(tmp_path):
    bad = str(tmp_path / "no_such_folder" / "x.jsonl")
    good = str(tmp_path / "ok.jsonl")
    w = LineWriter()
    for ln in LINES[:5]:
        w.write(bad, ln)
        w.write(good, ln)
    info = w.close()
    assert info["lost_lines"] == 5 and info["lines"] == 5 and info["n_errors"] == 1
    assert "open" in info["errors"][0] and "no_such_folder" in info["errors"][0]
    assert open(good, encoding="utf-8").read() == "".join(LINES[:5])


def test_close_twice_and_write_after_close(tmp_path):
    w = LineWriter()
    w.write(str(tmp_path / "x"), "a\n")
    assert w.close() == w.close()
    with pytest.raises(RuntimeError):
        w.write(str(tmp_path / "x"), "b\n")


def test_lines_are_written_when_the_process_stops_on_an_error(tmp_path):
    """The engine process can stop with an exception before run_live closes the writer: the queued lines still reach
    the file (the writer closes itself at interpreter exit)."""
    p = str(tmp_path / "x.jsonl")
    code = ("from xclean.live.logwriter import LineWriter\n"
            "w = LineWriter()\n"
            "for i in range(2000):\n"
            "    w.write(%r, '%%d\\n' %% i)\n"
            "raise SystemExit('stopped')\n" % p)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 1 and "stopped" in r.stderr
    assert open(p).read() == "".join("%d\n" % i for i in range(2000))


# ------------------------------------------------------------------------------------------- the recorder process
def _old_recorder(msg_q, log_path, now_ns):
    """The recorder as it was (direct, buffered writes), for the byte comparison."""
    seen, n, invalid = set(), 0, []
    with open(log_path, "w", encoding="utf-8", buffering=1 << 16) as f:
        while True:
            m = msg_q.get()
            rx = now_ns()
            if m == "EOS":
                break
            if isinstance(m, dict) and "frame_index" in m and "stream" in m:
                key = (m["frame_index"], m["stream"])
                if key in seen:
                    invalid.append(key)
                    f.write(json.dumps({"invalid": "duplicate (frame, stream)", "frame_index": key[0],
                                        "stream": key[1], "rx_ns": rx}) + "\n")
                seen.add(key)
            m["rx_ns"] = rx
            f.write(json.dumps(m) + "\n")
            n += 1
        f.write(json.dumps({"recorder_end": True, "messages": n, "duplicates": len(invalid), "rx_ns": now_ns()})
                + "\n")


def _messages():
    out = []
    for i in range(300):
        for s in ("LIVE-ODOM", "LIVE-MAP"):
            out.append({"frame_index": i, "stream": s, "T": np.eye(4).ravel().tolist(), "t_pose": 0.05 * i,
                        "state": "tracking"})
    out.insert(10, {"frame_index": 2, "stream": "LIVE-ODOM", "T": [0.0] * 16, "t_pose": 0.1, "state": "x"})
    return out


def _counter():
    c = {"n": 1000}

    def now_ns():
        c["n"] += 7
        return c["n"]
    return now_ns


def test_recorder_writes_the_same_bytes(tmp_path, monkeypatch):
    from xclean.live import recorder
    a, b = str(tmp_path / "old.jsonl"), str(tmp_path / "new.jsonl")
    q1, q2 = queue.Queue(), queue.Queue()
    for m in _messages():
        q1.put(dict(m))
        q2.put(dict(m))
    q1.put("EOS")
    q2.put("EOS")
    _old_recorder(q1, a, _counter())
    monkeypatch.setattr(recorder, "now_ns", _counter())
    recorder.recorder_main(q2, b)
    assert open(a, "rb").read() == open(b, "rb").read()
    assert b'"invalid": "duplicate (frame, stream)"' in open(b, "rb").read()


def test_recorder_write_error_ends_with_an_error(tmp_path):
    from xclean.live import recorder
    q = queue.Queue()
    q.put({"frame_index": 0, "stream": "LIVE-ODOM"})
    q.put("EOS")
    with pytest.raises(RuntimeError, match="not written completely"):
        recorder.recorder_main(q, str(tmp_path / "missing_folder" / "recorder.jsonl"))


# ------------------------------------------------------------------------------------- the engine's frame log
def _engine():
    from xclean.core.derive import Rig
    from xclean.vio.engine_live import P1Engine
    W, H, f, B = 320, 240, 300.0, 0.1
    cal = {"camera_width": W, "camera_height": H, "M1": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], "D1": [0.0] * 5,
           "M2": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], "D2": [0.0] * 5, "R_stereo": np.eye(3).tolist(),
           "T_stereo": [-B, 0.0, 0.0], "baseline_m": B, "T_body_cam0": np.eye(4).tolist(),
           "R_cam0_imu": np.eye(3).tolist(), "camera_fps": 20.0}
    src = types.SimpleNamespace(calib=cal, rig=Rig.from_calib_dict(cal, name="synthetic"), imu_stream=None)
    return P1Engine(src)


def test_engine_frame_log_same_bytes_with_the_writer(tmp_path, monkeypatch):
    monkeypatch.delenv("XCLEAN_P1_FRAME_LOG", raising=False)
    a, b = str(tmp_path / "direct.jsonl"), str(tmp_path / "writer.jsonl")
    outs = []
    for path, writer in ((a, None), (b, LineWriter())):
        eng = _engine()
        eng.frame_log = path
        eng.log_writer = writer
        for i in range(50):
            eng._fl = {"n_kp": i, "innov_rotvec_body": [0.1 * i, -0.0, 1e-17], "mode": "A"}
            eng.stats["events"].append({"frame": i, "what": "→ test"})
            eng._log_frame(types.SimpleNamespace(index=i, t=0.05 * i), "tracking", len(eng.stats["events"]) - 1)
        if writer is not None:
            outs.append(writer.close())
    assert outs[0]["lines"] == 50
    assert open(a, "rb").read() == open(b, "rb").read()


# --------------------------------------------------------------------------------------- the loop link's packet log
def _link(path, writer):
    from xclean.live import p13_link
    L = p13_link._Link.__new__(p13_link._Link)
    L.h = {"ready": threading.Event()}
    L.log_path = path
    L.logw = writer
    if writer is not None:
        writer.open(path, "a", encoding="utf-8")
        L.log = None
    else:
        L.log = open(path, "a", encoding="utf-8")
    L.pending, L.n_emitted, L.t_emit = [], 0, []
    L.st = {"kf_packets": 0, "images_dropped": 0}
    L.ctl_q = queue.Queue()
    L.ring = types.SimpleNamespace(write=lambda left, right, n: n % 4)
    return L


def test_packet_log_same_bytes_with_the_writer(tmp_path, monkeypatch):
    from xclean.live import p13_link
    a, b = str(tmp_path / "direct.jsonl"), str(tmp_path / "writer.jsonl")
    w = LineWriter()
    for path, writer in ((a, None), (b, w)):
        clock = iter(range(10 ** 6))                  # the same perf_counter values in both runs (wall_emit is logged)
        monkeypatch.setattr(p13_link, "time", types.SimpleNamespace(perf_counter=lambda: 0.001 * next(clock),
                                                                   sleep=time.sleep))
        L = _link(path, writer)
        monkeypatch.setattr(p13_link, "_L", L)
        for i in range(40):
            L.pending = [{"type": "edge_final", "j": i - 7, "Z": [1.0, 2.0]},
                         {"type": "kf", "j": i, "_img": (None, None)}]
            p13_link.emit(None, 0.05 * i, np.eye(4), np.eye(4))
        p13_link.close()
    assert w.close()["lines"] == 80
    assert open(a, "rb").read() == open(b, "rb").read()
