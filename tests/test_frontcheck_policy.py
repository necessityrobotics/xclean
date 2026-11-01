# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""The start-up self-check policy of the GPU front end, and a clean exit when the engine dies at start-up.

CPU only: the GPU front end is replaced by stand-ins, so these tests exercise the policy code itself (which checks
raise, which fall back, what is recorded), not the GPU kernels.
"""
import contextlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest


def _calib(W=320, H=240):
    f, B = 300.0, 0.1
    return {"camera_width": W, "camera_height": H,
            "M1": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], "D1": [0.0] * 5,
            "M2": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], "D2": [0.0] * 5,
            "R_stereo": np.eye(3).tolist(), "T_stereo": [-B, 0.0, 0.0], "baseline_m": B,
            "T_body_cam0": np.eye(4).tolist(), "R_cam0_imu": np.eye(3).tolist(), "camera_fps": 20.0}


class _Source:
    def __init__(self):
        from xclean.core.derive import Rig
        self.calib = _calib()
        self.rig = Rig.from_calib_dict(self.calib, name="synthetic")
        self.imu_stream = None


class _FailingGpuDetector:
    def __init__(self, rig, *a, **k):
        raise RuntimeError("gpu_front: no SIMD-boundary model reproduces cv2.cornerMinEigenVal on this build")


@pytest.fixture
def gf_on(monkeypatch):
    """The GPU front flag switched on in this process (the CPU-only default drops it), restored afterwards."""
    from xclean.front import fastpaths as FP
    monkeypatch.setattr(FP, "LAT", set(FP.LAT) | {"gf"})
    monkeypatch.delenv("XCLEAN_STRICT_SELFCHECK", raising=False)
    return FP


def test_strict_flag_parsing(monkeypatch):
    from xclean.front.gpu_front import strict_selfcheck
    for v, want in (("", False), ("0", False), ("1", True), ("yes", True)):
        monkeypatch.setenv("XCLEAN_STRICT_SELFCHECK", v)
        assert strict_selfcheck() is want
    monkeypatch.delenv("XCLEAN_STRICT_SELFCHECK")
    assert strict_selfcheck() is False


def test_failed_gpu_front_falls_back_to_the_cpu_front(gf_on, monkeypatch, capsys):
    from xclean.front import gpu_front, detect, mapmatch, mapmatch_fast
    from xclean.vio.engine_live import P1Engine
    monkeypatch.setattr(gpu_front, "Detector", _FailingGpuDetector)
    eng = P1Engine(_Source())
    assert type(eng.det) is detect.Detector                       # the CPU front, not a GPU subclass
    want_mm = mapmatch_fast.MapMatcher if gf_on.on("mf") else mapmatch.MapMatcher
    assert type(eng.mm) is want_mm
    assert not gf_on.on("gf")                                     # every later flag test sees the CPU path
    assert "SIMD-boundary" in eng.front_fallback["reason"]
    assert eng.front_fallback["cuda_context_released"] is True    # no CUDA context in a CPU-only test process
    err = capsys.readouterr().err
    assert err.count("xclean: warning:") == 1 and "CPU front end" in err and "latency" in err
    assert eng.summary()["front_fallback"]["reason"] == eng.front_fallback["reason"]


def test_strict_mode_raises_as_before(gf_on, monkeypatch):
    from xclean.front import gpu_front
    from xclean.vio.engine_live import P1Engine
    monkeypatch.setattr(gpu_front, "Detector", _FailingGpuDetector)
    monkeypatch.setenv("XCLEAN_STRICT_SELFCHECK", "1")
    with pytest.raises(RuntimeError, match="SIMD-boundary"):
        P1Engine(_Source())
    assert gf_on.on("gf")                                         # nothing was switched


def test_no_fallback_record_when_the_front_is_not_used(monkeypatch):
    from xclean.front import fastpaths as FP
    from xclean.vio.engine_live import P1Engine
    monkeypatch.setattr(FP, "LAT", set(FP.LAT) - {"gf"})
    eng = P1Engine(_Source())
    assert eng.front_fallback is None and "front_fallback" not in eng.summary()


def test_warmup_failure_switches_a_built_engine(gf_on, monkeypatch, capsys):
    """A GPU-front failure in a warm-up step (after the engine was built): both front modules are swapped."""
    from xclean.front import detect
    from xclean.vio import engine_live as EL
    gf_on.LAT.discard("gf")
    eng = EL.P1Engine(_Source())                                  # built on the CPU front ...
    gf_on.LAT.add("gf")                                           # ... and pretend the GPU front is on

    calls = []

    def bad_gpu_warmup(rig):
        calls.append("gpu")
        raise RuntimeError("gpu_front: ORB descriptors differ from cv2 (3 bits)")

    assert EL._gf_guarded(eng, bad_gpu_warmup, eng.rig) is None
    assert calls == ["gpu"] and not gf_on.on("gf") and type(eng.det) is detect.Detector
    assert "ORB descriptors" in eng.front_fallback["reason"]
    assert "xclean: warning:" in capsys.readouterr().err


def test_guard_is_transparent_when_the_step_passes(gf_on):
    from xclean.vio import engine_live as EL
    assert EL._gf_guarded(None, lambda x: x * 2, 21) == 42      # no engine needed: nothing failed
    assert gf_on.on("gf")


# ---------------------------------------------------------------------------- the 3x3 solver check (check_lk_solver)
class _FakeCp:
    """The few CuPy calls check_lk_solver makes, on the host (the 'device' arrays are numpy arrays)."""
    float64 = np.float64

    class cuda:
        @staticmethod
        def using_allocator(_malloc):
            return contextlib.nullcontext()

    @staticmethod
    def asarray(a):
        return np.array(a)

    @staticmethod
    def empty(n, dt):
        return np.empty(n, dt)

    @staticmethod
    def asnumpy(a):
        return np.asarray(a)


class _FakeFront:
    def __init__(self, perturb):
        from xclean.front import gpu_front
        self.cp = _FakeCp
        self.calib = {}
        self.pool = type("P", (), {"malloc": None})()
        self.k = {"k_lkcheck": self._kernel}
        self.perturb = perturb
        self.solve = gpu_front._nb_solve_inv()

    def _kernel(self, grid, block, args):
        dA, dg, n, dX, dI = args
        n = int(n)
        X, Inv = np.empty((n, 3)), np.empty((n, 3, 3))
        self.solve(dA.reshape(n, 3, 3), dg.reshape(n, 3), X, Inv)  # the CPU solver = a matching "device"
        if self.perturb:
            X[::7, 0] = np.nextafter(X[::7, 0], np.inf)            # one ulp on every 7th system
        dX[:] = X.ravel()
        dI[:] = Inv.ravel()


def test_lk_check_equal_returns_the_count(monkeypatch):
    from xclean.front import gpu_front
    monkeypatch.delenv("XCLEAN_STRICT_SELFCHECK", raising=False)
    f = _FakeFront(perturb=False)
    assert gpu_front.GpuFront.check_lk_solver(f, np.random.default_rng(0), n=64) == 64
    assert "lk_solver_mismatch" not in f.calib


def test_lk_check_mismatch_keeps_the_gpu_front(monkeypatch, capsys):
    from xclean.front import gpu_front
    monkeypatch.delenv("XCLEAN_STRICT_SELFCHECK", raising=False)
    f = _FakeFront(perturb=True)
    assert gpu_front.GpuFront.check_lk_solver(f, np.random.default_rng(0), n=64) is False
    assert f.calib["lk_solver_mismatch"] == 10 and f.calib["lk_solver_systems"] == 64     # rows 0, 7, ..., 63
    err = capsys.readouterr().err
    assert err.count("xclean: note:") == 1 and "10 of 64" in err and "stays on" in err


def test_lk_check_mismatch_raises_in_strict_mode(monkeypatch):
    from xclean.front import gpu_front
    monkeypatch.setenv("XCLEAN_STRICT_SELFCHECK", "1")
    f = _FakeFront(perturb=True)
    with pytest.raises(RuntimeError, match="differs from numba's LAPACK on 10 of 64 systems"):
        gpu_front.GpuFront.check_lk_solver(f, np.random.default_rng(0), n=64)


def test_release_device_without_cuda_is_a_no_op():
    from xclean.front import gpu_front
    rel = gpu_front.release_device()
    assert rel["cuda_context_released"] is True and rel["errors"] == [] and rel["note"] is None
    assert gpu_front.cuda_context_active() is False


class _Pool:
    def __init__(self, fail=False):
        self.fail, self.freed = fail, 0

    def free_all_blocks(self):
        if self.fail:
            raise RuntimeError("pool is busy")
        self.freed += 1


class _FakeCupy:
    def __init__(self, fail=False):
        self.pool, self.pinned = _Pool(fail), _Pool(fail)

    def get_default_memory_pool(self):
        return self.pool

    def get_default_pinned_memory_pool(self):
        return self.pinned


def test_release_device_keeps_an_existing_context_and_says_so():
    """A CUDA context that exists is kept (CuPy cannot drop it safely); its cached memory is freed; recorded."""
    from xclean.front import gpu_front
    cp = _FakeCupy()
    rel = gpu_front.release_device(cp=cp, context_active=lambda: True)
    assert rel["cuda_context_released"] is False and rel["gpu_memory_freed"] is True and rel["errors"] == []
    assert cp.pool.freed == 1 and cp.pinned.freed == 1
    assert "stays open" in rel["note"] and "memory was freed" in rel["note"]


def test_release_device_without_a_context():
    from xclean.front import gpu_front
    rel = gpu_front.release_device(cp=_FakeCupy(), context_active=lambda: False)
    assert rel == {"cuda_context_released": True, "gpu_memory_freed": True, "note": None, "errors": []}


def test_release_device_records_errors_instead_of_hiding_them():
    from xclean.front import gpu_front

    def ask():
        raise OSError("driver gone")
    rel = gpu_front.release_device(cp=_FakeCupy(fail=True), context_active=ask)
    assert rel["gpu_memory_freed"] is False and rel["cuda_context_released"] is False   # unknown = assume held
    assert len(rel["errors"]) == 2 and "pool is busy" in rel["errors"][0] and "driver gone" in rel["errors"][1]


def test_fallback_records_a_kept_context(gf_on, monkeypatch, capsys):
    from xclean.front import gpu_front
    from xclean.vio.engine_live import P1Engine
    monkeypatch.setattr(gpu_front, "Detector", _FailingGpuDetector)
    monkeypatch.setattr(gpu_front, "release_device", lambda: {"cuda_context_released": False, "gpu_memory_freed": True,
                                                              "note": gpu_front.CONTEXT_KEPT_NOTE, "errors": []})
    eng = P1Engine(_Source())
    fb = eng.front_fallback
    assert fb["cuda_context_released"] is False and fb["gpu_memory_freed"] is True and fb["release_errors"] == []
    assert fb["cuda_context_note"] == gpu_front.CONTEXT_KEPT_NOTE
    err = capsys.readouterr().err
    assert err.count("xclean: warning:") == 1 and "stays open" in err      # one warning, which says it


# --------------------------------------------------------------- the harness exits when the engine dies at start-up
def _sequence(root, n=3):
    import cv2
    os.makedirs(os.path.join(root, "left"))
    os.makedirs(os.path.join(root, "right"))
    cal = _calib(160, 120)
    cal.update(num_frames=n, t0_ns=0)
    with open(os.path.join(root, "calib.json"), "w") as f:
        json.dump(cal, f)
    rng = np.random.default_rng(0)
    with open(os.path.join(root, "timestamps.csv"), "w") as f:
        f.write("frame_id,timestamp_s,left_t,right_t\n")
        for i in range(n):
            img = (rng.random((120, 160)) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(root, "left", "%06d.png" % i), img)
            cv2.imwrite(os.path.join(root, "right", "%06d.png" % i), img)
            f.write("%d,%.3f,%.3f,%.3f\n" % (i, 0.05 * i, 0.05 * i, 0.05 * i))


def test_harness_exits_when_the_engine_dies_at_startup(tmp_path):
    seq, out = str(tmp_path / "seq"), str(tmp_path / "run")
    _sequence(seq)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    # an invalid engine switch makes the engine process raise while it starts; the loop worker is off to keep it short
    r = subprocess.run([sys.executable, "-m", "xclean.cli", "replay", seq, "--out", out, "--mode", "lockstep",
                        "--no-x1-flags", "--score", "--env", "XCLEAN_P11_OFF=bogus,XCLEAN_P13_OFF=loop"],
                       capture_output=True, text=True, timeout=240, env=env)
    assert r.returncode == 1, (r.stdout[-2000:], r.stderr[-3000:])
    assert "the engine stopped during start-up" in r.stderr
    assert "XCLEAN_P11_OFF" in r.stderr                            # the engine's own error is shown
    h = json.load(open(os.path.join(out, "harness.json")))
    assert h["engine_failed"]["during_startup"] is True and h["valid"] is False
    assert "the engine of our published results (default)" in r.stdout   # the CLI default was applied and printed
