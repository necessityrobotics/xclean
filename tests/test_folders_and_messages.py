# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Safety and user-facing behaviour of the commands: which output folder --overwrite may replace, the check of the
input sequence folder, a replay whose frame reader fails, image files on non-ASCII paths, the audit of a wrong path
and of a camera at rest, `xclean score` on xclean's own outputs, the default output folder, help texts and the
end-of-run lines.  CPU only, synthetic data."""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

from xclean import cli

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
FAST = ["--mode", "lockstep", "--no-x1-flags", "--env", "XCLEAN_P13_OFF=loop"]


def _calib(W=160, H=120, fps=20.0):
    f, B = 150.0, 0.1
    return {"camera_width": W, "camera_height": H,
            "M1": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], "D1": [0.0] * 5,
            "M2": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], "D2": [0.0] * 5,
            "R_stereo": np.eye(3).tolist(), "T_stereo": [-B, 0.0, 0.0], "baseline_m": B,
            "T_body_cam0": np.eye(4).tolist(), "R_cam0_imu": np.eye(3).tolist(), "camera_fps": fps}


def _sequence(root, n=6, drop_left=None, gt=False):
    """A small synthetic sequence folder (written with the non-ASCII-safe writer, so `root` may hold any letters)."""
    import cv2
    from xclean.io.imageio import imwrite
    root = str(root)
    os.makedirs(os.path.join(root, "left"))
    os.makedirs(os.path.join(root, "right"))
    cal = _calib()
    cal.update(num_frames=n, t0_ns=0, imu_rate_hz=0)
    with open(os.path.join(root, "calib.json"), "w", encoding="utf-8") as f:
        json.dump(cal, f)
    rng = np.random.default_rng(0)
    with open(os.path.join(root, "timestamps.csv"), "w", encoding="utf-8") as f:
        f.write("frame_id,timestamp_s,left_t,right_t\n")
        for i in range(n):
            img = cv2.GaussianBlur((rng.random((120, 160)) * 255).astype(np.uint8), (0, 0), 1.5)
            if i != drop_left:
                assert imwrite(os.path.join(root, "left", "%06d.png" % i), img)
            assert imwrite(os.path.join(root, "right", "%06d.png" % i), np.roll(img, -4, axis=1))
            f.write("%d,%.3f,%.3f,%.3f\n" % (i, 0.05 * i, 0.05 * i, 0.05 * i))
    if gt:
        with open(os.path.join(root, "gt.csv"), "w", encoding="utf-8") as f:
            f.write("#t_ns,x,y,z,qw,qx,qy,qz\n")
            for i in range(n):
                f.write("%d,%.4f,0,0,1,0,0,0\n" % (int(round(0.05 * i * 1e9)), 0.001 * i))
    return root


def _files(folder, names):
    os.makedirs(folder, exist_ok=True)
    for n in names:
        with open(os.path.join(folder, n), "w") as f:
            f.write("x")
    return str(folder)


# ------------------------------------------------------------------------------------- --overwrite never deletes
def test_overwrite_refuses_the_input_folder_however_it_is_spelled(tmp_path, monkeypatch):
    seq = _files(tmp_path / "seq", ["calib.json", "timestamps.csv", "harness.json"])
    monkeypatch.chdir(tmp_path)
    for out in (seq, seq + os.sep, "seq", os.path.join(".", "seq"), seq.upper() if os.name == "nt" else seq):
        err = cli.check_out_folders("replay", [seq, "--out", out, "--overwrite"])[1]
        assert err is not None and "this command reads" in err and "Nothing was deleted" in err, out
    err = cli.check_out_folders("replay", [seq, "--out", str(tmp_path / "seq" / "run")])[1]
    assert err is not None and "lies inside" in err                   # a run folder inside the input is refused too
    assert sorted(os.listdir(seq)) == ["calib.json", "harness.json", "timestamps.csv"]


def test_overwrite_refuses_the_replayed_recording_and_shared_folders(tmp_path):
    rec = _files(tmp_path / "rec", ["calib.json", "timestamps.csv", "stream_meta.csv", "demo.json"])
    err = cli.check_out_folders("live", ["--replay", rec, "--out", rec, "--overwrite"])[1]
    assert err is not None and "this command reads" in err
    both = str(tmp_path / "both")
    err = cli.check_out_folders("live", ["--out", both, "--record", both, "--overwrite"])[1]
    assert err is not None and "two separate folders" in err
    err = cli.check_out_folders("live", ["--out", both, "--record", os.path.join(both, "rec")])[1]
    assert err is not None and "two separate folders" in err


@pytest.mark.parametrize("where", ["cwd", "parent_of_cwd", "home", "root", "install"])
def test_overwrite_never_deletes_protected_folders(tmp_path, monkeypatch, where):
    import xclean
    work = tmp_path / "work"
    work.mkdir()
    (work / "harness.json").write_text("{}")
    monkeypatch.chdir(work)
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "recorder.jsonl").write_text("x")
    out = {"cwd": ".", "parent_of_cwd": "..", "home": str(tmp_path / "home"),
           "root": os.path.abspath(os.sep), "install": os.path.dirname(xclean.__file__)}[where]
    err = cli.check_out_folders("replay", ["seq", "--out", out, "--overwrite"])[1]
    assert err is not None and "never deletes" in err and "Nothing was deleted" in err


def test_overwrite_replaces_only_an_earlier_run_of_the_same_command(tmp_path):
    seq = _files(tmp_path / "seq", ["calib.json"])
    replay_run = _files(tmp_path / "r", ["harness.json", "recorder.jsonl", "feeder.json", "live_score.json"])
    live_run = _files(tmp_path / "l", ["harness.json", "recorder.jsonl", "demo.json", "view_final.png"])
    other = _files(tmp_path / "o", ["notes.txt"])
    mixed = _files(tmp_path / "m", ["harness.json", "recorder.jsonl", "notes.txt"])
    ok = lambda cmd, argv: cli.check_out_folders(cmd, argv + ["--overwrite"])[1]
    assert ok("replay", [seq, "--out", replay_run]) is None
    assert "not an earlier xclean replay run" in ok("replay", [seq, "--out", live_run])
    assert "not an earlier xclean replay run" in ok("replay", [seq, "--out", other])
    assert "notes.txt" in ok("replay", [seq, "--out", mixed])
    assert ok("live", ["--out", live_run]) is None
    assert "not an earlier xclean live run" in ok("live", ["--out", other])
    rec = _files(tmp_path / "rec", ["calib.json", "timestamps.csv", "imu.jsonl", "stream_meta.csv"])
    os.makedirs(os.path.join(rec, "left"))
    assert ok("live", ["--out", str(tmp_path / "new"), "--record", rec]) is None
    os.makedirs(os.path.join(rec, "left", "sub"))
    assert "sub-folders" in ok("live", ["--out", str(tmp_path / "new"), "--record", rec])
    assert "not an earlier recording" in ok("live", ["--out", str(tmp_path / "new"), "--record", other])
    for d in (other, mixed):
        assert os.listdir(d)                                        # nothing was touched


def test_refusal_says_nothing_was_deleted(tmp_path):
    full = _files(tmp_path / "full", ["harness.json"])
    err = cli.check_out_folders("replay", ["seq", "--out", full])[1]
    assert "Nothing was deleted" in err and "--overwrite" in err and "deletes that folder" not in err


def test_the_modules_apply_the_folder_rule_themselves(tmp_path, monkeypatch, capsys):
    """python -m xclean.live.harness (no CLI wrapper) refuses a folder of other files instead of deleting it."""
    from xclean.live import harness
    seq = _sequence(tmp_path / "seq", n=2)
    other = _files(tmp_path / "other", ["notes.txt"])
    monkeypatch.setattr(sys, "argv", ["harness", seq, "--out", other, "--mode", "lockstep"])
    assert harness.main() == 2
    assert "Nothing was deleted" in capsys.readouterr().err and os.listdir(other) == ["notes.txt"]
    monkeypatch.setattr(sys, "argv", ["harness", seq, "--out", seq, "--mode", "lockstep"])
    assert harness.main() == 2 and os.path.isfile(os.path.join(seq, "calib.json"))


def test_imu_capture_keeps_a_folder_of_other_files(tmp_path, capsys):
    from xclean.live.oak_imu_capture import capture
    other = _files(tmp_path / "other", ["notes.txt"])
    with pytest.raises(SystemExit) as e:
        capture(other, 0.01, overwrite=True)
    assert e.value.code == 2 and "Nothing was deleted" in capsys.readouterr().err      # a refusal: exit code 2
    assert os.listdir(other) == ["notes.txt"]


# --------------------------------------------------------------------------------- the input folder is checked
def test_preflight_names_the_problem(tmp_path):
    raw = tmp_path / "MH_01_easy"
    (raw / "mav0").mkdir(parents=True)
    err = cli.preflight("replay", [str(raw), "--out", str(tmp_path / "run")])
    assert "not a sequence folder" in err and "xclean convert euroc" in err
    assert "does not exist" in cli.preflight("replay", [str(tmp_path / "typo"), "--out", "o"], environ={})
    seq = _sequence(tmp_path / "seq", n=2)
    assert cli.preflight("replay", [seq, "--out", "o"]) is None
    assert cli.preflight("live", ["--out", "o", "--replay", seq]) is None
    assert cli.preflight("live", ["--out", "o"]) is None                         # a camera run has no input folder
    os.remove(os.path.join(seq, "right", "000000.png"))
    assert "missing or cannot be read" in cli.preflight("replay", [seq, "--out", "o"])
    seq2 = _sequence(tmp_path / "seq2", n=2)
    cal = json.load(open(os.path.join(seq2, "calib.json")))
    cal["camera_width"] = 320
    json.dump(cal, open(os.path.join(seq2, "calib.json"), "w"))
    assert "calib.json says 320 x 120" in cli.preflight("replay", [seq2, "--out", "o"])
    os.remove(os.path.join(seq2, "timestamps.csv"))
    assert "it has no timestamps.csv" in cli.preflight("live", ["--out", "o", "--replay", seq2])


def test_replay_of_a_wrong_folder_stops_before_anything_starts(tmp_path):
    raw = tmp_path / "raw"
    (raw / "mav0").mkdir(parents=True)
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, "-m", "xclean.cli", "replay", str(raw), "--out", str(out)],
                       capture_output=True, text=True, timeout=120, env=ENV)
    assert r.returncode == 2 and "not a sequence folder" in r.stderr and "Traceback" not in r.stderr
    assert "engine settings" not in r.stdout and not out.exists()


# --------------------------------------------------------------------------------- a frame reader that fails
def test_replay_ends_when_the_frame_reader_fails(tmp_path):
    """An unreadable image in the middle of the sequence: the frame reader stops with an error; the replay ends the
    stream, stops and says so (it used to wait forever)."""
    seq = _sequence(tmp_path / "seq", n=6, drop_left=3)
    out = str(tmp_path / "run")
    r = subprocess.run([sys.executable, "-m", "xclean.cli", "replay", seq, "--out", out] + FAST,
                       capture_output=True, text=True, timeout=300, env=ENV)
    assert r.returncode == 1, (r.stdout[-2000:], r.stderr[-3000:])
    assert "cannot read" in r.stderr and "000003.png" in r.stderr
    assert "the frame reader stopped with an error" in r.stderr and "Traceback" not in r.stderr
    hj = json.load(open(os.path.join(out, "harness.json")))
    assert hj["feeder_failed"]["exitcode"] == 1 and hj["checks"]["exit_codes"]["engine"] == 0


# ------------------------------------------------------------------------------------------- non-ASCII paths
def test_image_io_matches_opencv_on_plain_paths(tmp_path):
    import cv2
    from xclean.io.imageio import imread, imwrite
    img = (np.random.default_rng(3).random((60, 80)) * 255).astype(np.uint8)
    for ext, prm in ((".png", [cv2.IMWRITE_PNG_COMPRESSION, 1]), (".png", []), (".jpg", [])):
        a, b = str(tmp_path / ("a" + ext)), str(tmp_path / ("b" + ext))
        assert cv2.imwrite(a, img, prm) and imwrite(b, img, prm)
        assert open(a, "rb").read() == open(b, "rb").read()                       # the same file bytes
        for flag in (cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED, cv2.IMREAD_COLOR):
            assert np.array_equal(cv2.imread(a, flag), imread(a, flag))             # the same pixels
    assert imread(str(tmp_path / "missing.png"), cv2.IMREAD_GRAYSCALE) is None


def test_image_io_on_a_non_ascii_path(tmp_path):
    import cv2
    from xclean.io.imageio import imread, imwrite
    d = tmp_path / "\u0921\u0947\u091f\u093e \u00e9"
    d.mkdir()
    img = (np.random.default_rng(4).random((30, 40)) * 255).astype(np.uint8)
    p = str(d / "000000.png")
    assert imwrite(p, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    assert np.array_equal(imread(p, cv2.IMREAD_GRAYSCALE), img)


def test_replay_and_score_from_a_non_ascii_folder(tmp_path):
    """A whole replay from a folder with non-ASCII letters, scored at the end, with the output going to a pipe in a
    code page that cannot encode the folder's name (as when the output of a command is redirected to a file on
    Windows): the files are written and the end lines are printed, with backslash escapes.  Then `xclean score` on the
    run folder, on its trajectory_live.csv, and on a live --replay run folder made from it.  Also checks the start and
    end lines of the run, and that no raw JSON is printed without --json."""
    base = tmp_path / "\u0921\u0947\u091f\u093e \u00e9"
    seq = _sequence(base / "seq", n=6, gt=True)
    out = str(base / "run")
    cp1252 = dict(ENV, PYTHONIOENCODING="cp1252")                       # strict: an unencodable letter used to raise
    r = subprocess.run([sys.executable, "-m", "xclean.cli", "replay", seq, "--out", out, "--score"] + FAST,
                       capture_output=True, text=True, timeout=600, env=cp1252, encoding="cp1252")
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])
    assert "was not written" not in r.stderr and "could not be made" not in r.stdout, (r.stdout, r.stderr)
    rows = [ln for ln in open(os.path.join(out, "trajectory_live.csv"), encoding="utf-8") if not ln.startswith("#")]
    assert len(rows) == 6
    assert "xclean: vision only: the sequence folder has no imu.jsonl." in r.stdout
    assert "\\u0921" in r.stdout                                       # the path, written with escapes
    assert not [ln for ln in r.stdout.splitlines() if ln.startswith(("{", "resources"))]    # no raw JSON
    last = r.stdout.strip().splitlines()[-1]
    assert last.startswith("xclean: summary: 6 frames, 0 dropped,") and "ATE " in last and "live_score.json" in last
    ate = json.load(open(os.path.join(out, "live_score.json"), encoding="utf-8"))["B5"]["ate"]
    assert json.load(open(os.path.join(out, "harness.json"), encoding="utf-8"))["sequence_folder"] == \
        os.path.abspath(seq)
    r2 = subprocess.run([sys.executable, "-m", "xclean.cli", "score", out], capture_output=True, text=True,
                        timeout=300, env=cp1252, encoding="cp1252")
    assert r2.returncode == 0 and ("ATE %.6f m" % ate) in r2.stdout, (r2.stdout, r2.stderr)
    r3 = subprocess.run([sys.executable, "-m", "xclean.cli", "score", os.path.join(out, "trajectory_live.csv"),
                         "--dataset", seq], capture_output=True, text=True, timeout=300, env=ENV, encoding="utf-8")
    assert r3.returncode == 0 and "ATE" in r3.stdout and "Traceback" not in r3.stderr, (r3.stdout, r3.stderr)
    # the same run folder as `xclean live --replay` records it (demo.json names the recording; harness.json's tag
    # does not): xclean score finds the recording by itself
    import shutil
    live = str(base / "live_run")
    shutil.copytree(out, live)
    json.dump({"tag": "oak_replay", "mode": "lockstep"}, open(os.path.join(live, "harness.json"), "w"))
    json.dump({"tag": "oak_replay", "args": {"replay": seq}}, open(os.path.join(live, "demo.json"), "w"))
    r4 = subprocess.run([sys.executable, "-m", "xclean.cli", "score", live], capture_output=True, text=True,
                        timeout=300, env=ENV, encoding="utf-8")
    assert r4.returncode == 0 and ("ATE %.6f m" % ate) in r4.stdout, (r4.stdout, r4.stderr)


def test_stream_recorder_reports_images_it_could_not_write(tmp_path, monkeypatch):
    from xclean.io import imageio
    from xclean.live.oak_source import StreamRecorder
    monkeypatch.setattr(imageio, "imwrite", lambda *a, **k: False)
    rec = StreamRecorder(str(tmp_path / "rec"), _calib())
    img = np.zeros((120, 160), np.uint8)
    rec.frame(0, 0.0, 0.0, img, img, [0, 0, 0.0, None, None, 0, 0])
    info = rec.close()
    assert info["image_write_errors"] == 2 and "000000.png" in info["first_write_error"]


# ------------------------------------------------------------------------------------------------------ audit
def test_audit_flags_a_wrong_path_and_a_folder_that_is_not_a_sequence(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    r = subprocess.run([sys.executable, "-m", "xclean.cli", "audit", str(tmp_path / "typo"), str(empty)],
                       capture_output=True, text=True, timeout=120, env=ENV)
    assert r.returncode == 1 and "2 dataset(s) flagged" in r.stdout, r.stdout
    assert "not found" in r.stdout and "not a sequence folder" in r.stdout
    env = {k: v for k, v in ENV.items() if k != "XCLEAN_DATASET_ROOTS"}
    r = subprocess.run([sys.executable, "-m", "xclean.eval.dataset_audit"], capture_output=True, text=True,
                       timeout=120, env=env)
    assert r.returncode == 2 and "usage" in r.stderr


def _imu_folder(root, ay):
    from xclean.eval.dataset_audit import audit
    seq = _sequence(root, n=4)
    cal = json.load(open(os.path.join(seq, "calib.json")))
    cal.update(imu_rate_hz=200.0, imu_noise={"gyroscope_noise_density": 1e-4, "gyroscope_random_walk": 1e-5,
                                            "accelerometer_noise_density": 1e-3, "accelerometer_random_walk": 1e-4})
    json.dump(cal, open(os.path.join(seq, "calib.json"), "w"))
    with open(os.path.join(seq, "imu.jsonl"), "w") as f:
        for k, v in enumerate(ay):
            f.write(json.dumps({"t": 0.005 * k, "ax": 0.0, "ay": float(v), "az": 0.0, "gx": 0.0, "gy": 0.0,
                                "gz": 0.0}) + "\n")
    return audit(seq)[0]


def test_audit_does_not_call_gravity_on_a_camera_at_rest_a_rail(tmp_path):
    """A camera at rest reads 1 g, quantised: its largest value repeats (our still OAK recordings: 9.78 m/s^2, 8-19
    times).  That is not the end of the measuring range."""
    rng = np.random.default_rng(0)
    still = np.minimum(np.round(9.74 + 0.02 * rng.standard_normal(2000), 3), 9.78)
    still[::200] = 9.787                                          # the largest value, 10 times (the old rule: a rail)
    assert not [f for f in _imu_folder(tmp_path / "still", still) if "rails" in f]
    clipped = still.copy()
    clipped[100:120] = 19.6                                       # a +-2 g sensor at its rail (Rosario v1: 19.600)
    assert [f for f in _imu_folder(tmp_path / "clip", clipped) if "accel y rails at" in f]


# ------------------------------------------------------------------------------------------------------ score
def test_score_gives_plain_errors(tmp_path):
    seq = _sequence(tmp_path / "seq", n=3, gt=True)
    csv = tmp_path / "t.csv"
    csv.write_text("0.0,0,0,0,0,0,0,1,0,tracking,0.0,1.0\n")
    for args, text in (([str(tmp_path / "nope")], "does not exist"),
                       ([str(csv)], "--dataset"),
                       ([seq], "is not a run folder")):
        r = subprocess.run([sys.executable, "-m", "xclean.cli", "score"] + args, capture_output=True, text=True,
                           timeout=120, env=ENV)
        assert r.returncode == 2 and text in r.stderr and "Traceback" not in r.stderr, (args, r.stderr)


def test_trajectory_live_csv_is_read_by_the_scorer(tmp_path):
    from xclean.eval.score import load_trajectory
    p = tmp_path / "trajectory_live.csv"
    p.write_text("# t_frame,x,y,z,qx,qy,qz,qw,frame_index,state,t_publish_s,latency_ms\n"
                 "0.0,1.0,2.0,3.0,0,0,0,1,0,tracking,0.1,2.0\n0.05,1.5,2.0,3.0,0,0,0,1,1,coasting,0.2,2.0\n")
    t, P, dropped = load_trajectory(str(p))
    assert np.allclose(t, [0.0, 0.05]) and np.allclose(P[:, 0], [1.0, 1.5]) and dropped == 0


# ------------------------------------------------------------------------------------ default output folders
def test_run_writes_under_the_current_folder_by_default(tmp_path, monkeypatch):
    import xclean
    from xclean import run
    monkeypatch.delenv("XCLEAN_RUNS", raising=False)
    monkeypatch.chdir(tmp_path)
    out = run.default_out("C:/data/xclean/MH_01_easy/")
    assert out == os.path.join(os.getcwd(), "runs", "MH_01_easy")
    assert os.path.samefile(os.path.dirname(os.path.dirname(out)), str(tmp_path))
    pkg = os.path.dirname(os.path.dirname(xclean.__file__))
    from xclean.live import harness
    assert not os.path.abspath(harness.WITNESS_REF_V2).startswith(os.path.abspath(pkg))


# --------------------------------------------------------------------------------------- help and messages
# SHA-256 of internal names, written as lower-case words joined by one space: names of internal projects and
# prototypes, internal item ids, folder names of our machines, the device ID of our camera, internal process
# words and unpublished internal documents.  Only the hashes are published, so this test does not publish the
# names it keeps out.  To add one: hashlib.sha256(" ".join(re.findall(r"[a-z0-9]+", name.lower())).encode())
INTERNAL_NAME_SHA256 = frozenset({
    "014ba6df53442ac6e44f0fd0475eac86e0b451fcf12039c14470c944f97dbba7",
    "02f2c39b5761e3fe46c9bed27af70b02f4a490deb84b1f27f4a8b9bd3d45c93e",
    "0414cef2e54c9092bb985e080f6ad8653c3fed4f3941d44bd62f820380c4dc83",
    "057a18e845f137cb0a03bc22b70fab781ca5df18bfa76b13b00802709830327e",
    "09d944aad89a40c1508c76a0da7c757423c277f303a49ab42ed1e3c4856dbcfd",
    "0b83da77454e9d8034de304c5f41ee1f5b7c6d376942c50720dc9ef0116fe0dc",
    "0ba36cf8b315e4d1b4c6f548da67e584825bfb1b9a6b96190f085c99b0f234ed",
    "0e3c9c6570016ec5bee8fac0927dfd065f34eae62b9da7805689c7b9373d06ca",
    "13e0c01d64eabbed5132ad2a97a70b5d34a74d6c9f388fc84087251e83240ff8",
    "1b5f5b84b21fb519da706f9acfc55b42beb66366cc3c72a84c0f5829c91a8aa8",
    "1d216491739134bf03f375bd215e6100920fabdc62f83f821f563af80f4f5863",
    "1fd2d291a62d0f4b1c4d88f82ee635ddd69be4135b3ab5ede704e128da880081",
    "3625bcc7ba14a562b8af92af0745324159cc7d9439cac58fcc3db4fa7b8dcf2e",
    "363d3970f5d827e6e4ce82f9f69dc6051a47b2b78d96733dc77144f08e96b389",
    "39912b2c5d5b74719840da9857cfff55de9a0f4649eedd9ec40a866ff99e3337",
    "3a8cfe81d50302b031cde30603c4bbfbe2af5a32fffb80a034f4a9da9d315153",
    "462ea42052b429462cbee767e2f4f749049b0f23066ff2d7aafebdebc7d6ac76",
    "48246db5daf4e15637f9172bd333acc36a0174eac7f30e806c5407223fda962e",
    "4fd19218ad4366ab70a1bf7bd346cdb80ebe9d37ae3cb6c284dd152a7b607883",
    "532caf283e0b787df114d0001975bae378b18b51187ce850b34d59e822a5e278",
    "558545e76f45c018af1427283cbe19d340cd5c89e89e842ddc8488a5edfd0fff",
    "58e08f69ba8386e2db875b1a2d92734a4c9c4f62e44962fd29e59a546fadb366",
    "6048d82026ed696aecf95e1c5bf7b3d1258d26b755ee869fd167303820eca5ee",
    "60d707583c55f4df14e7d263db98045ac6e77183d531a6652e69f14ac8d11f04",
    "6774e97b32d47e4e848019c7333d0aecfb1eb46f335917f34df1e5ad9c0116f9",
    "67b3270bc5ebccdd58e3d178564a0ea84354b9d0af930d9cc71e91796c928133",
    "6980064d31f99a075a0fd3bef8008ed76ac7d2061cd4ddac1e11ce0f5c6ce21a",
    "6b85f90cc0428e2f9e27b63334fc8b33889ec5bcc8059ad04ec43f06cc77d5bc",
    "71056f83c0cc1a393f6e6660c06035d8d832682daeb32e52468ed727438294fe",
    "72411d2528226a4507c07315c223d1c57c5edfa28d707db171fdb04af182a833",
    "81c91959a82b5584d0004a794c50c90f8520f1982a831443827bd52f95b7dd2a",
    "872370771bbd78a147b8c970d3192bf1dfeb8248930a1ada252fa8ce002f4018",
    "89b9d192120c0a1d5acd5d5705d88565a4091b61397e536ab7d0ed4786705ae8",
    "8d21547004acfc5df47fa05eaa1962a16e41b3935f4e265b855b5e1e38b3cb03",
    "8dd5e80a80b0771f6f2f68d50e99bc591900020dd452ed88af1a29d72f681b11",
    "93955ad7d5fcbc0908b835510207769df62ef56f5b56fd2653b891a8ea083893",
    "94c99ccc33873128f753553f5581bf32d32e3e36d6ff2d2f787aaa26219c8026",
    "9e8b393a73d426939c175e397185f07c5366ce4bc923987f75c6497a85f13411",
    "9f7095d136569cc30d622cdd3256edcd09f92fe30beab9592622e71afb9fc4ab",
    "a526d02caa7ed6c3cf96e137ed78b2eda484e5de4e49a518cf685b40927341b6",
    "ab59036d8ff0cde252343beac33ceb6980f5b28611e84f18a17b8b1a8efc800a",
    "ae0070d3858fcb865fd62773eafe74031de7f3141f98e45b755a2051c3462931",
    "af195b260dbb48dfcc8a8b8c6d7b84a25146aef1e0b8b37cea75bfdf4ac6181d",
    "b9966b96ea6a2d0870b377972842d7c459f0295055fc165e6d3ed7ba9eb40a54",
    "bc25f674efcd2bcc83854520107e5616ecde5e227fa222e91051e9e344c629a2",
    "c97faf9f179a0eacf08a9792c203ed0c88833c4d2c2d30b8f1e26ff23f78352e",
    "cd78eb6e586c6eeb5b104465e4a42d0d6c166fc6b8dd91f346a97e94695826a6",
    "ce75921ca3a24c0f52c9a8e5e042223e4773f61755d0b754d26f30bb1db2f025",
    "d139248df98d4926d726f7506b9d4570afcd013d087c9380e39e74cbc6b49e62",
    "d143f33096909a943befe8f96e76ea7e8ce7117448446f9dc744f16a2889503c",
    "d3a7c4bc0270c9af53f3734d263c05fbd88481c1a112e9deb0442323ec37ac9c",
    "d3ab710081b7200e53026021a1f908ba5dfdc81b98c6e4c5b5c4cdd067dac774",
    "d5822522516fb6176c561f0bd68ba37ca806bfd273b93eb516ab8236c58d1b18",
    "e15f54b3b52f6ee479e7f82fbb583cf9659d77ace04a9f679abd101c83b371b5",
    "e2d543e9386a9a0892bc121ba2e6f92cbfa28ad722f7fa76aa62e51a00c090d1",
    "e40a7e64e5d3f0ccdad2fe832736bd496d6da912cf3fa56ae4bf2c2a5f5c634c",
    "e4aa9b48e732572ba01c154ad78fb57dd5cdc901be2c213f02963527b5da18f1",
    "e75c77b0abe0607fdad738e5da3df224684c2ee8f6128173be2c83b0113cdc1b",
    "ece024588fd851e2723d79999d7e3694d66eefee7601ffea7baf2ad3f367a7c8",
    "eebf2d96024a34744b71719defe64f5b2059551f7d539d6282f7c99cfa81da51",
    "f137e4188cb543524b955f5dec3f62d1f851ac52d4077b15db8aec6b085265c3",
    "f1e53b902d342070d2517c1fdfd5d14e1c615d0754ec92d8a3bbe3d3c4c291f5",
    "f8d5da93d9f691e5f2b6b3a731818b59a2c78feef78e80d31dac1a93400197c4",
    "fe10a2b398892802987bd26ce67dbbc078c13a7b36664f061ca1f6d8f27aae33",
})
# lines that keep an old name on purpose (same normalisation, SHA-256 of the whole line): the loop-closure cost
# table lookup strips an old row-name prefix (a research switch only)
ALLOWED_LINE_SHA256 = frozenset({"b8e90b37bd8feed2336336f92b9b5835f3aea6c3a48866c533e82564a06101cb"})
# plain words that a help text or message must not use (benchmark-protocol and history words, not names)
INTERNAL_WORDS = re.compile(r"NOT FOR BARS|P1\.3|P13_|protocol v2|witness|row registry|\bx1\b|REFUTED|"
                            r"ABLATION|session \d|defect \d|receipt|<src>|oak_demo\.py|OAKDEMO", re.IGNORECASE)


def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def internal_names(text):
    """The internal names in `text` (words and runs of 2 or 3 words whose SHA-256 is listed)."""
    import hashlib
    w = _words(text)
    grams = set(w) | {" ".join(w[i:i + 2]) for i in range(len(w) - 1)}
    grams |= {" ".join(w[i:i + 3]) for i in range(len(w) - 2)}
    return sorted(g for g in grams if hashlib.sha256(g.encode("utf-8")).hexdigest() in INTERNAL_NAME_SHA256)


def internal_text(text):
    """Internal names and internal words found in a help text or message."""
    return internal_names(text) + INTERNAL_WORDS.findall(text)


@pytest.mark.parametrize("sub", ["run", "replay", "live", "score", "audit", "convert", "calib", "doctor"])
def test_help_has_no_internal_names(sub):
    r = subprocess.run([sys.executable, "-m", "xclean.cli", sub, "--help"], capture_output=True, text=True,
                       timeout=120, env=ENV)
    assert r.returncode == 0, r.stderr
    hits = internal_text(r.stdout)
    assert not hits, (sub, hits)


def test_top_level_help_points_to_the_documented_commands():
    r = subprocess.run([sys.executable, "-m", "xclean.cli", "--help"], capture_output=True, text=True, env=ENV)
    assert "score_live" not in r.stdout and not internal_text(r.stdout)
    from xclean.io.convert import build_parser
    assert "xclean replay <output>" in build_parser().description and "xclean run <output>" not in \
        build_parser().description


def test_hidden_options_still_work():
    from xclean.live import harness, oak_demo
    a = harness.build_parser().parse_args(["seq", "--out", "o", "--reference", "--witness-ref", "w", "--blackout",
                                           "1:2", "--no-x1-flags", "--threads", "3", "--seed", "4", "--env", "A=1",
                                           "--final"])
    assert a.reference and a.witness_ref == "w" and a.blackout == "1:2" and a.no_x1_flags and a.threads == 3
    b = oak_demo.build_parser().parse_args(["--out", "o", "--tag", "t", "--engine-root", "e", "--env", "A=1"])
    assert b.tag == "t" and b.engine_root == "e"
    help_text = harness.build_parser().format_help() + oak_demo.build_parser().format_help()
    for hidden in ("--reference", "--witness-ref", "--blackout", "--no-x1-flags", "--engine-root", "--threads"):
        assert hidden not in help_text
    assert "--overwrite" in oak_demo.build_parser().format_help()          # --record says what --overwrite does


def test_live_score_flag_text_is_plain():
    from xclean.live import score_live
    texts = [c for c in score_live.score_frames.__code__.co_consts if isinstance(c, str)
             and c.startswith("GT_POSITION_ONLY")]
    assert texts and not any("owner" in t or "bar reading" in t for t in texts)


def test_start_and_end_lines():
    from xclean.live.harness import imu_status_line
    from xclean.live.oak_demo import imu_outcome
    noise = {"imu_noise": {"gyroscope_noise_density": 1e-4}}
    assert imu_status_line(noise, True, env={}).startswith("xclean: IMU in use:")
    assert imu_status_line({}, True, env={}).startswith("xclean: vision only:")
    assert "no imu.jsonl" in imu_status_line(noise, False, env={})
    assert "switched off" in imu_status_line(noise, True, env={"XCLEAN_P1_VI": "0"})
    assert imu_outcome({"vi": True, "vi_admit_frame": 261}) == "IMU used from frame 261"
    assert imu_outcome({"vi": True, "vi_admit_frame": 5, "demoted": True, "demote_frame": 90}).endswith("frame 90")
    assert imu_outcome({"vi": False}, has_imu=False) == "vision only"


def test_summary_line_of_a_run_folder(tmp_path):
    from xclean.live.oak_demo import summary_line
    d = str(tmp_path)
    T = np.eye(4).ravel().tolist()
    with open(os.path.join(d, "recorder.jsonl"), "w") as f:
        for i in range(4):
            if i == 2:
                continue                                          # frame 2 never published: dropped
            for s in ("LIVE-ODOM", "LIVE-MAP"):
                f.write(json.dumps({"frame_index": i, "stream": s, "T": T, "t_pose": 0.05 * i, "state": "tracking",
                                    "rx_ns": 1_000_000 * (10 * i + 4)}) + "\n")
    json.dump({"mode": "paced", "frames": [[i, 0.05 * i, 0, 1_000_000 * 10 * i] for i in range(4)],
               "imu_released": 0}, open(os.path.join(d, "feeder.json"), "w"))
    line = summary_line(d, score={"B5": {"ate": 0.0123, "frames_scored": 3}})
    assert line.startswith("xclean: summary: 4 frames, 1 dropped, latency p50 4.0 ms")
    assert "ATE 0.0123 m over 3 frames" in line and "vision only" in line and d in line and "\n" not in line


# ------------------------------------------------------------------------------- no internal names in the source
SOURCE_PATTERNS = re.compile(r"\bwf_[0-9a-f]{6}|runs/[x]|\blead \d{4}-")      # [x]: this line must not match
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(REPO, "docs", "prereg")


def _published_text_files():
    """The package's own text files, and the repository's docs and tests when they are present (a source checkout).
    docs/prereg/ is left out: it is published exactly as written (docs/prereg/README.md)."""
    import xclean
    roots = [os.path.dirname(xclean.__file__)]
    roots += [os.path.join(REPO, d) for d in ("docs", "tests") if os.path.isdir(os.path.join(REPO, d))]
    for root in roots:
        for dp, dn, fs in os.walk(root):
            dn[:] = [d for d in dn if d != "__pycache__" and os.path.abspath(os.path.join(dp, d)) != PREREG]
            for fn in fs:
                if fn.endswith((".py", ".md", ".json")):
                    yield os.path.join(dp, fn)
    for fn in ("README.md", "QUICKSTART.md", "NOTICE", "THIRD_PARTY.md", "USERS.md", "environment.yml",
               "pyproject.toml", ".gitignore", ".gitattributes"):
        if os.path.isfile(os.path.join(REPO, fn)):
            yield os.path.join(REPO, fn)


def test_source_carries_no_internal_names():
    """Comments, docstrings, messages and help of the package, the files it ships, and the repository's docs and tests
    name no internal project, person, machine folder, device, document or item id."""
    import hashlib
    hits = []
    for p in _published_text_files():
        for k, ln in enumerate(open(p, encoding="utf-8-sig"), 1):
            names = internal_names(ln)
            if names and hashlib.sha256(" ".join(_words(ln)).encode("utf-8")).hexdigest() in ALLOWED_LINE_SHA256:
                continue
            if names or SOURCE_PATTERNS.search(ln):
                hits.append("%s:%d: %s %s" % (os.path.relpath(p, REPO), k, names, ln.strip()[:100]))
    assert not hits, "\n".join(hits[:20])


def test_the_internal_name_check_finds_names():
    """The hashed check works: ordinary text is not flagged, and the probe phrase (listed only for this test) is found
    in any spelling: other case, underscores, dots."""
    assert internal_names("xclean replay MH_04_difficult --out runs/mh04 --score") == []
    assert internal_names("see Internal_Name.Pro" + "be here") == ["internal name pro" + "be"]  # split: no hit here
    assert len(INTERNAL_NAME_SHA256) > 50 and all(len(x) == 64 for x in INTERNAL_NAME_SHA256)


PREREG_SHA256 = "ca7e7adebb492e6e51ac23fd3ab7002c51de3ff5309ed4429378621425355018"


def test_preregistration_is_published_unchanged():
    """docs/prereg/FINAL_BOARD_PREREG.md is the exact file whose SHA-256 the results page publishes."""
    import hashlib
    p = os.path.join(PREREG, "FINAL_BOARD_PREREG.md")
    if not os.path.isfile(p):
        pytest.skip("not a source checkout (docs/prereg/ is not here)")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == PREREG_SHA256
    assert PREREG_SHA256 in open(os.path.join(PREREG, "README.md"), encoding="utf-8").read()
    attrs = open(os.path.join(REPO, ".gitattributes"), encoding="utf-8").read().split("\n")
    assert "docs/prereg/** -text" in attrs              # git must not change its line endings


# ------------------------------------------------------------------------- plain output, exit codes, package name
def test_score_finds_the_sequence_of_a_run_folder(tmp_path):
    from xclean.eval.score import run_source
    seq = _sequence(tmp_path / "seq", n=2)
    rep = _files(tmp_path / "rep", [])
    json.dump({"tag": "typed/elsewhere", "sequence_folder": seq}, open(os.path.join(rep, "harness.json"), "w"))
    assert run_source(rep) == (seq, None)
    old = _files(tmp_path / "old", [])                     # a replay run made before sequence_folder was recorded
    json.dump({"tag": seq}, open(os.path.join(old, "harness.json"), "w"))
    assert run_source(old) == (seq, None)
    lr = _files(tmp_path / "lr", [])                       # xclean live --replay
    json.dump({"tag": "oak_replay"}, open(os.path.join(lr, "harness.json"), "w"))
    json.dump({"tag": "oak_replay", "args": {"replay": seq}}, open(os.path.join(lr, "demo.json"), "w"))
    assert run_source(lr) == (seq, None)
    cam = _files(tmp_path / "cam", [])                     # xclean live on the camera
    json.dump({"tag": "oak_live"}, open(os.path.join(cam, "harness.json"), "w"))
    json.dump({"tag": "oak_live", "sequence_folder": None, "args": {"replay": None}},
              open(os.path.join(cam, "demo.json"), "w"))
    folder, why = run_source(cam)
    assert folder is None and "live camera run" in why and "no ground truth" in why
    gone = _files(tmp_path / "gone", [])
    json.dump({"args": {"replay": str(tmp_path / "moved")}}, open(os.path.join(gone, "demo.json"), "w"))
    folder, why = run_source(gone)
    assert folder is None and "moved" in why and "--dataset <sequence folder>" in why


def test_score_of_a_camera_run_says_why_it_cannot(tmp_path, monkeypatch, capsys):
    cam = _files(tmp_path / "cam", ["recorder.jsonl"])
    json.dump({"tag": "oak_live", "args": {"replay": None}}, open(os.path.join(cam, "demo.json"), "w"))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    assert cli.main(["score", cam]) == 2
    err = capsys.readouterr().err
    assert "live camera run" in err and "Traceback" not in err


def test_overwrite_names_a_live_run_as_such(tmp_path):
    from xclean.io.folders import output_kind_problem
    d = _files(tmp_path / "live", ["harness.json", "demo.json", "recorder.jsonl"])
    assert output_kind_problem("replay", d) == ("it is an xclean live run (it holds demo.json), not an earlier xclean "
                                                "replay run")
    assert "no demo.json" in output_kind_problem("live", _files(tmp_path / "rep", ["harness.json", "recorder.jsonl"]))


@pytest.mark.parametrize("sub", ["replay", "live"])
def test_no_arguments_print_the_usage_error_first(sub):
    r = subprocess.run([sys.executable, "-m", "xclean.cli", sub], capture_output=True, text=True, timeout=120, env=ENV)
    assert r.returncode == 2 and r.stderr.startswith("usage: xclean %s" % sub), (r.stdout, r.stderr)
    assert "engine settings" not in r.stdout


def test_live_usage_refusal_exits_with_code_2(tmp_path, monkeypatch, capsys):
    from xclean.live import oak_demo
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")       # main() adds an error handler for its helper processes
    monkeypatch.setattr(sys, "argv", ["xclean live", "--out", str(tmp_path / "o"), "--mode", "lockstep"])
    assert oak_demo.main() == 2 and "--mode lockstep needs --replay" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_raw_json_only_with_the_json_option():
    from xclean.live import harness, oak_demo
    for ap in (harness.build_parser(), oak_demo.build_parser()):
        assert "--json" in ap.format_help()
    assert harness.build_parser().parse_args(["seq", "--out", "o", "--json"]).json


def test_run_ledger_and_end_line_are_plain():
    from xclean.obs.latency import Ledger
    from xclean.run import end_line
    led = Ledger(run="seq", tier=1, sync=None, gpu_stages=())
    for k in range(3):
        with led.stage("kf_insert", kf=k):
            pass
        with led.stage("match", frame=k):
            pass
        led.frame_done(k)
    text = led.header()
    for word in ("LEDGER", "WITNESS", "VOID", "BUDGET PASS", "BUDGET FAIL", "tier=", "keyframes=0", "OVER BUDGET"):
        assert word not in text, word
    assert "time per stage" in text and "latency budget met" in text and "speed check" in text
    line = end_line({"wall_s": 12.0, "fps": 25.0, "live": {"processed": 300}}, os.path.join("runs", "mh01"))
    assert line.startswith("xclean run: done: 300 frames in 12.0 s") and "summary.json" in line and "{" not in line


def test_version_names_the_python_package(monkeypatch):
    import importlib.metadata as md
    seen = []
    monkeypatch.setattr(md, "version", lambda name: seen.append(name) or "9.9")
    assert cli._version() == "9.9" and seen == ["xstereo-slam"] == [cli.DISTRIBUTION]

    def missing(name):
        raise md.PackageNotFoundError(name)
    monkeypatch.setattr(md, "version", missing)
    import xclean
    assert cli._version() == "%s (not installed with pip)" % xclean.__version__
    toml = os.path.join(REPO, "pyproject.toml")
    if os.path.isfile(toml):
        assert 'name = "%s"' % cli.DISTRIBUTION in open(toml, encoding="utf-8").read().split("\n")


def test_version_of_the_installed_package():
    import importlib.metadata as md
    try:
        want = md.version("xstereo-slam")
    except md.PackageNotFoundError:
        pytest.skip("the Python package xstereo-slam is not installed (a source tree on PYTHONPATH)")
    r = subprocess.run([sys.executable, "-m", "xclean.cli", "--version"], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0 and r.stdout.strip() == "xclean %s" % want
