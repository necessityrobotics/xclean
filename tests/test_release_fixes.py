# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""User-facing fixes of the release: start-up warm-up, the live demo's exit after an engine failure, the --out guard,
the audit's IMU checks, the replay trajectory file, the "valid" note, --pin on any CPU and the solver note.
CPU only, synthetic data."""
import json
import os
import subprocess
import sys
import threading
import time
import types

import numpy as np
import pytest

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")


def _calib(W=160, H=120, fps=20.0):
    f, B = 150.0, 0.1
    return {"camera_width": W, "camera_height": H,
            "M1": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], "D1": [0.0] * 5,
            "M2": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], "D2": [0.0] * 5,
            "R_stereo": np.eye(3).tolist(), "T_stereo": [-B, 0.0, 0.0], "baseline_m": B,
            "T_body_cam0": np.eye(4).tolist(), "R_cam0_imu": np.eye(3).tolist(), "camera_fps": fps}


def _sequence(root, n=5, **cal_extra):
    import cv2
    os.makedirs(os.path.join(root, "left"))
    os.makedirs(os.path.join(root, "right"))
    cal = _calib()
    cal.update(num_frames=n, t0_ns=0, **cal_extra)
    with open(os.path.join(root, "calib.json"), "w") as f:
        json.dump(cal, f)
    rng = np.random.default_rng(0)
    with open(os.path.join(root, "timestamps.csv"), "w") as f:
        f.write("frame_id,timestamp_s,left_t,right_t\n")
        for i in range(n):
            img = cv2.GaussianBlur((rng.random((120, 160)) * 255).astype(np.uint8), (0, 0), 1.5)
            cv2.imwrite(os.path.join(root, "left", "%06d.png" % i), img)
            cv2.imwrite(os.path.join(root, "right", "%06d.png" % i), np.roll(img, -4, axis=1))
            f.write("%d,%.3f,%.3f,%.3f\n" % (i, 0.05 * i, 0.05 * i, 0.05 * i))
    return root


# ---------------------------------------------------------------------------------------- warm-up (numba kernels)
def test_warmup_compiles_the_kernels_the_first_frames_call():
    """Every kernel a frame calls is compiled (or loaded) before the clock starts: the innovation's SO(3) log and the
    row-band stereo with a vertical shift (float32 right image) were compiled during the first frames."""
    from xclean.core.derive import Rig
    from xclean.front import stereo1d
    from xclean.vio import engine_live as EL
    from xclean.vio import window as Wn
    rig = Rig.from_calib_dict(_calib(), name="synthetic")
    EL.warmup(rig)
    assert any("float64, 2d, C" in str(s) or "Array(float64, 2, 'C'" in str(s) for s in Wn.so3_log_nb.signatures)
    band = [str(s) for s in stereo1d.stereo_points_band.signatures]
    assert any("Array(float32, 2, 'C'" in s for s in band), band          # right rows resampled (dy != 0)
    assert any("Array(uint8, 2, 'C', False, aligned=True), Array(uint8, 2, 'C'" in s for s in band), band


# ------------------------------------------------------------------- the live demo after an engine start-up failure
def test_live_replay_exits_quickly_when_the_engine_dies_at_startup(tmp_path):
    seq, out = _sequence(str(tmp_path / "rec"), n=5, imu_rate_hz=0), str(tmp_path / "run")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "xclean.cli", "live", "--out", out, "--replay", seq, "--view", "none",
                        "--no-x1-flags", "--env", "XCLEAN_P11_OFF=bogus,XCLEAN_P13_OFF=loop"],
                       capture_output=True, text=True, timeout=240, env=ENV)
    took = time.time() - t0
    assert r.returncode == 1, (r.stdout[-2000:], r.stderr[-3000:])
    assert took < 90, took                                       # was 121 s: two 60 s joins
    assert "the engine stopped during start-up" in r.stderr and "XCLEAN_P11_OFF" in r.stderr
    demo = json.load(open(os.path.join(out, "demo.json")))
    assert demo["engine_failed"]["during_startup"] is True
    assert demo["checks"]["exit_codes"]["source"] == 0           # the source ended by itself (no wait for "ready")
    assert demo["engine_failed"]["children_stopped"] == [] and "helper process" not in r.stderr   # none left over


def test_source_ends_when_stopped_before_the_engine_is_ready(tmp_path):
    from xclean.live import oak_source

    class Backend:
        kind = "replay"
        finished = False

        def start(self, events):
            self.events = events

        def stop(self):
            self.stopped = True

        def info(self):
            return {"backend": "stub"}

    ctl = {"ready": threading.Event(), "stop": threading.Event(), "lock": threading.Lock(),
           "new_frame": threading.Event()}
    ctl["stop"].set()
    slot = types.SimpleNamespace(hdr=[0] * 8)
    be = Backend()
    log_path = str(tmp_path / "feeder.json")
    t0 = time.time()
    log = oak_source.run_source(be, {"mode": "paced", "fps": 30.0}, slot, None, ctl, log_path)
    assert time.time() - t0 < 2.0
    assert log["stopped_before_ready"] is True and log["frames"] == [] and be.stopped
    assert slot.hdr[4] == 1 and ctl["new_frame"].is_set()      # end of stream for an engine that may still start
    assert json.load(open(log_path))["frames_total"] == 0


# --------------------------------------------------------------------------------------------- --out / --record
def test_out_folder_guard(tmp_path):
    from xclean.cli import check_out_folders
    seq = tmp_path / "seq"
    seq.mkdir()
    (seq / "calib.json").write_text("{}")
    full = tmp_path / "full"
    full.mkdir()
    (full / "keep.txt").write_text("x")
    empty = tmp_path / "empty"
    empty.mkdir()
    afile = tmp_path / "afile"
    afile.write_text("x")
    argv, err = check_out_folders("replay", [str(seq), "--out", str(tmp_path / "new")])
    assert err is None and argv == [str(seq), "--out", str(tmp_path / "new")]
    assert check_out_folders("replay", [str(seq), "--out", str(empty)])[1] is None
    argv, err = check_out_folders("replay", [str(seq), "--out", str(full)])
    assert "already exists and is not empty" in err and "--overwrite" in err and "Nothing was deleted" in err
    argv, err = check_out_folders("replay", [str(seq), "--out=%s" % full, "--overwrite", "--score"])
    assert err is not None and "not an earlier xclean replay run" in err      # a folder of other files is kept
    run = tmp_path / "run"
    run.mkdir()
    for name in ("harness.json", "recorder.jsonl", "trajectory_live.csv"):
        (run / name).write_text("x")
    argv, err = check_out_folders("replay", [str(seq), "--out=%s" % run, "--overwrite", "--score"])
    assert err is None and argv == [str(seq), "--out=%s" % run, "--score"]          # --overwrite is not passed on
    assert "is a file" in check_out_folders("replay", [str(seq), "--out", str(afile)])[1]
    argv, err = check_out_folders("replay", [str(seq), "--out", str(tmp_path), "--overwrite"])
    assert err is not None and "holds" in err and str(seq) in err                    # never delete the input
    err = check_out_folders("live", ["--out", str(tmp_path / "new"), "--record", str(full)])[1]
    assert err is not None and "--record" in err
    argv, err = check_out_folders("live", ["--out", str(tmp_path / "new"), "--", "--overwrite"])
    assert err is None and argv[-1] == "--overwrite"                                 # after "--": not ours


def test_cli_refuses_a_full_out_folder_before_starting(tmp_path, monkeypatch, capsys):
    from xclean import cli
    full = tmp_path / "full"
    full.mkdir()
    (full / "keep.txt").write_text("x")
    called = []
    fake = types.ModuleType("fake_harness_guard")
    fake.main = lambda: called.append(list(sys.argv)) or 0
    monkeypatch.setitem(sys.modules, "fake_harness_guard", fake)
    monkeypatch.setitem(cli.COMMANDS, "replay", ("fake_harness_guard", "main", "test"))
    assert cli.main(["replay", "seq", "--out", str(full)]) == 2
    assert not called and "already exists" in capsys.readouterr().err and (full / "keep.txt").exists()
    assert cli.main(["replay", "seq", "--out", str(full), "--overwrite"]) == 2      # not an earlier run: kept
    assert not called and (full / "keep.txt").exists()
    (full / "keep.txt").unlink()
    (full / "harness.json").write_text("{}")                                         # an earlier replay run
    assert cli.main(["replay", "seq", "--out", str(full), "--overwrite"]) == 0
    assert called and "--overwrite" not in called[0]


# ------------------------------------------------------------------------------------------------- audit messages
def test_audit_flags_an_imu_rate_without_imu_data(tmp_path):
    from xclean.eval.dataset_audit import audit
    from xclean.eval.integrity import classify
    seq = _sequence(str(tmp_path / "s"), n=5, imu_rate_hz=200.0)
    flags, info = audit(seq)
    assert "no imu" in info
    hit = [f for f in flags if "no imu.jsonl" in f]
    assert len(hit) == 1 and "imu_rate_hz to 0" in hit[0]
    assert classify(hit[0])[0] == "loud"                         # accepted with a printed caveat, not a refusal
    ok = _sequence(str(tmp_path / "ok"), n=5, imu_rate_hz=0)
    assert audit(ok)[0] == []


def test_audit_says_what_the_engine_does_without_imu_noise(tmp_path):
    from xclean.eval.dataset_audit import audit
    from xclean.eval.integrity import classify
    seq = _sequence(str(tmp_path / "s"), n=5, imu_rate_hz=200.0)
    with open(os.path.join(seq, "imu.jsonl"), "w") as f:
        for k in range(200):
            f.write(json.dumps({"t": 0.005 * k, "ax": 0.0, "ay": 0.0, "az": 9.80665, "gx": 0.0, "gy": 0.0,
                                "gz": 0.0}) + "\n")
    flags = audit(seq)[0]
    hit = [f for f in flags if "imu_noise" in f]
    assert hit and "does not use the IMU" in hit[0] and "ADIS" not in " ".join(flags)
    sev, why = classify(hit[0])
    assert sev == "loud" and "ADIS" not in why and "does not use the IMU" in why


def test_no_message_claims_imu_defaults():
    import xclean
    root = os.path.dirname(xclean.__file__)
    for rel in ("eval/dataset_audit.py", "eval/integrity.py", "backend/imu.py", "core/derive.py"):
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        assert "inherits a substitution" not in src and "applies EuRoC" not in src and "falls back to EuRoC" not in src


def test_start_up_notes_for_the_imu(tmp_path):
    from xclean.live.harness import imu_config_note
    from xclean.live.oak_demo import imu_note
    seq = _sequence(str(tmp_path / "s"), n=2, imu_rate_hz=200.0)
    note = imu_config_note(seq, {"imu_rate_hz": 200.0})
    assert note.startswith("xclean: warning:") and "imu_rate_hz to 0" in note
    assert imu_config_note(seq, {"imu_rate_hz": 0}) is None
    assert "--ref-calib" in imu_note({}) and "does not use the IMU" in imu_note({})
    assert "recording" in imu_note({}, replay="rec") and imu_note({"imu_noise": {"x": 1}}) is None


# ----------------------------------------------------------------------------------------- the replay trajectory
def test_trajectory_csv_from_a_run_folder(tmp_path):
    from xclean.live.oak_demo import write_trajectory_live
    d = str(tmp_path)
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    with open(os.path.join(d, "recorder.jsonl"), "w") as f:
        for i in range(3):
            for s in ("LIVE-ODOM", "LIVE-MAP"):
                f.write(json.dumps({"frame_index": i, "stream": s, "T": T.ravel().tolist(), "t_pose": 0.05 * i,
                                    "state": "tracking", "rx_ns": 1_000_000 * (10 + i)}) + "\n")
    json.dump({"mode": "lockstep", "frames": [[i, 0.05 * i, 0, 1_000_000 * (5 + i)] for i in range(3)]},
              open(os.path.join(d, "feeder.json"), "w"))
    assert write_trajectory_live(d) == 3
    rows = [ln for ln in open(os.path.join(d, "trajectory_live.csv")) if not ln.startswith("#")]
    assert len(rows) == 3 and rows[0].startswith("0.000000000,1.000000,2.000000,3.000000")


def test_replay_writes_the_trajectory_and_its_run_logs(tmp_path):
    """A whole (tiny) lock-step replay on the CPU: trajectory_live.csv is written and named in the output, and the
    engine's frame log is complete (written by its background writer)."""
    seq, out = _sequence(str(tmp_path / "seq"), n=6, imu_rate_hz=0), str(tmp_path / "run")
    r = subprocess.run([sys.executable, "-m", "xclean.cli", "replay", seq, "--out", out, "--mode", "lockstep",
                        "--no-x1-flags", "--env", "XCLEAN_P13_OFF=loop"],
                       capture_output=True, text=True, timeout=600, env=ENV)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])
    rows = [ln for ln in open(os.path.join(out, "trajectory_live.csv")) if not ln.startswith("#")]
    assert "xclean: wrote %d poses to" % len(rows) in r.stdout and "trajectory_live.csv" in r.stdout
    frames = [json.loads(x) for x in open(os.path.join(out, "p1_frames.jsonl"))]
    assert [f["frame"] for f in frames] == list(range(6))
    logs = json.load(open(os.path.join(out, "engine_summary.json")))["run_logs"]
    assert logs["lines"] >= 6 and logs["lost_lines"] == 0 and logs["n_errors"] == 0 and logs["finished"]
    assert "valid=false" not in r.stdout                         # lock-step: no clean-machine note


# ------------------------------------------------------------------------------------------------- "valid" note
def test_valid_note_for_a_users_paced_run():
    from xclean.live.harness import plain_validity_note
    checks = {"pinned_v2": False, "feeder_valid": True, "others_over_5pct_core": {},
              "other_gpu_compute_apps_not_allowed": [(27436, "C:\\Windows\\SystemApps\\ShellExperienceHost.exe")]}
    why = ["non-allowed GPU apps %s" % checks["other_gpu_compute_apps_not_allowed"],
           "engine tree not pinned (protocol v2 a)", "witness no reference"]
    note = plain_validity_note("paced", checks, {"ok": None, "note": "no reference"}, why)
    assert note.startswith("xclean: note:") and "clean-machine" in note and "usable" in note
    for part in ("not run with --pin", "no timing reference", "ShellExperienceHost.exe"):
        assert part in note
    assert "\n" not in note and "protocol" not in note
    assert plain_validity_note("lockstep", checks, None, ["x"]) is None
    assert plain_validity_note("paced", checks, None, []) is None


def test_valid_note_does_not_hide_real_problems():
    from xclean.live.harness import plain_validity_note
    note = plain_validity_note("paced", {"pinned_v2": True}, {"ok": True},
                               ["process exit codes {'engine': 0, 'feeder': 1}"])
    assert note.startswith("xclean: warning:") and "usable" not in note


# ------------------------------------------------------------------------------------------------------- --pin
def _groups(n_cores, smt=2):
    return [[c * smt + k for k in range(smt)] for c in range(n_cores)]


def test_pin_layout_is_the_published_one_on_16_cores():
    from xclean.live.harness import pin_layout, pin_masks
    L = pin_layout(_groups(16))
    assert L["engine_cpus"] == [16, 18, 20, 22, 24, 26, 28, 30]
    assert L["feeder_cpus"] == [12, 13, 14, 15] and L["worker_cpus"] == [10]
    assert L["engine_cores"] == list(range(8, 16)) and L["siblings_left_unused"] == list(range(17, 32, 2))
    assert pin_masks(_groups(16))[:2] == (L["engine_cpus"], L["feeder_cpus"])
    assert pin_layout(_groups(24))["engine_cpus"] == L["engine_cpus"]          # more cores: the same layout


@pytest.mark.parametrize("n,smt", [(6, 1), (8, 2), (12, 2), (15, 1)])
def test_pin_layout_on_smaller_cpus(n, smt):
    from xclean.live.harness import pin_layout
    L = pin_layout(_groups(n, smt))
    used = set(L["engine_cpus"]) | set(L["feeder_cpus"]) | set(L["worker_cpus"])
    assert len(used) == len(L["engine_cpus"]) + len(L["feeder_cpus"]) + len(L["worker_cpus"])   # no overlap
    assert len(L["engine_cores"]) >= 3 and min(L["worker_cores"]) >= 1                         # core 0 stays free
    assert max(used) < n * smt


def test_pin_refuses_a_small_cpu_with_a_plain_message(monkeypatch, capsys, tmp_path):
    from xclean.live import harness
    with pytest.raises(ValueError, match="at least 6 cores; this one has 4"):
        harness.pin_layout(_groups(4))
    monkeypatch.setattr(harness, "core_groups", lambda: _groups(4))
    monkeypatch.setattr(sys, "argv", ["xclean replay", str(tmp_path / "seq"), "--out", str(tmp_path / "o"), "--pin"])
    assert harness.main() == 2
    assert "Run without --pin" in capsys.readouterr().err and not (tmp_path / "o").exists()


# ------------------------------------------------------------------------------------------- the 3x3 solver note
def test_solver_note_names_the_detected_lapack():
    from xclean.front.gpu_front import lk_mismatch_cause
    assert "Haswell kernel" in lk_mismatch_cause("openblas", "pip", "Haswell")
    assert "Intel MKL" in lk_mismatch_cause("mkl", "conda", None)
    assert "conda package" in lk_mismatch_cause("openblas", "conda", None)
    assert "not identified" in lk_mismatch_cause("openblas", "pip", "SkylakeX")
    assert "MKL" not in lk_mismatch_cause("openblas", "pip", "Haswell")


def test_solver_note_text(capsys):
    from xclean.front.gpu_front import warn_lk_mismatch
    warn_lk_mismatch(856, 1024, cause="SciPy's OpenBLAS uses its Haswell kernel on this CPU")
    err = capsys.readouterr().err
    assert "856 of 1024" in err and "Haswell" in err and "MKL" not in err and err.count("xclean: note:") == 1


def test_kernel_module_docstring_is_ascii():
    import xclean.backend.kernels as K
    path = K.__file__
    assert all(b < 128 for b in open(path, "rb").read()), path
