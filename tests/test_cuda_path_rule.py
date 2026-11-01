# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""The CUDA compiler step at `import xclean` (xclean/__init__.py): on Windows, when the environment carries its own
CUDA kernel compiler, CUDA_PATH points at <env>/Library and <env>/Library/bin goes first on PATH; nothing changes when
XCLEAN_KEEP_CUDA_PATH is set, when CuPy was imported first, when the compiler is absent, or off Windows.  It never
raises and never imports CuPy."""
import os
import subprocess
import sys

import pytest

import xclean
from xclean import ENV_NVRTC_DLL, _use_env_cuda

TOOLKIT = os.path.join("C:" + os.sep, "toolkit", "v13.0")


def _env(tmp_path, with_dll=True):
    prefix = tmp_path / "env"
    bin_dir = prefix / "Library" / "bin"
    bin_dir.mkdir(parents=True)
    if with_dll:
        (bin_dir / ENV_NVRTC_DLL).write_bytes(b"")
    return str(prefix), str(prefix / "Library"), str(bin_dir)


def _path(*parts):
    return os.pathsep.join(parts)


def test_points_cuda_at_the_environment(tmp_path):
    prefix, lib, bin_dir = _env(tmp_path)
    environ = {"CUDA_PATH": TOOLKIT, "PATH": _path(os.path.join(TOOLKIT, "bin"), "other")}
    note = _use_env_cuda(environ, prefix, "win32", {})
    assert environ["CUDA_PATH"] == lib
    assert environ["PATH"] == _path(bin_dir, os.path.join(TOOLKIT, "bin"), "other")
    assert note["before"] == TOOLKIT and note["set_to"] == lib and note["path_first"] == bin_dir
    assert "own CUDA kernel compiler" in note["why"]


def test_applying_twice_changes_nothing_more(tmp_path):
    prefix, lib, bin_dir = _env(tmp_path)
    environ = {"PATH": "other"}
    _use_env_cuda(environ, prefix, "win32", {})
    once = dict(environ)
    note = _use_env_cuda(environ, prefix, "win32", {})        # e.g. a child process importing xclean again
    assert environ == once and note["set_to"] == lib and note["before"] == lib


def test_empty_environment(tmp_path):
    prefix, lib, bin_dir = _env(tmp_path)
    environ = {}
    _use_env_cuda(environ, prefix, "win32", {})
    assert environ == {"CUDA_PATH": lib, "PATH": bin_dir}


@pytest.mark.parametrize("keep", ["1", "yes", " 1 "])
def test_opt_out(tmp_path, keep):
    prefix, _lib, _bin = _env(tmp_path)
    environ = {"CUDA_PATH": TOOLKIT, "PATH": "other", "XCLEAN_KEEP_CUDA_PATH": keep}
    before = dict(environ)
    note = _use_env_cuda(environ, prefix, "win32", {})
    assert environ == before and note["set_to"] is None and "XCLEAN_KEEP_CUDA_PATH" in note["why"]


@pytest.mark.parametrize("keep", ["", "0", " 0 "])
def test_opt_out_values_that_do_not_opt_out(tmp_path, keep):
    prefix, lib, _bin = _env(tmp_path)
    environ = {"XCLEAN_KEEP_CUDA_PATH": keep}
    assert _use_env_cuda(environ, prefix, "win32", {})["set_to"] == lib and environ["CUDA_PATH"] == lib


def test_cupy_already_imported(tmp_path):
    prefix, _lib, _bin = _env(tmp_path)
    environ = {"CUDA_PATH": TOOLKIT, "PATH": "other"}
    note = _use_env_cuda(environ, prefix, "win32", {"cupy": object()})
    assert environ == {"CUDA_PATH": TOOLKIT, "PATH": "other"} and "already imported" in note["why"]


def test_no_compiler_in_the_environment(tmp_path):
    prefix, _lib, _bin = _env(tmp_path, with_dll=False)
    environ = {"CUDA_PATH": TOOLKIT, "PATH": "other"}
    note = _use_env_cuda(environ, prefix, "win32", {})
    assert environ == {"CUDA_PATH": TOOLKIT, "PATH": "other"} and note["set_to"] is None
    assert ENV_NVRTC_DLL in note["why"]


def test_not_windows(tmp_path):
    prefix, _lib, _bin = _env(tmp_path)
    environ = {"PATH": "other"}
    note = _use_env_cuda(environ, prefix, "linux", {})
    assert environ == {"PATH": "other"} and "not Windows" in note["why"]


def test_never_raises(tmp_path):
    prefix, _lib, _bin = _env(tmp_path)

    class ReadOnly(dict):
        def __setitem__(self, k, v):
            raise PermissionError("read-only")
    note = _use_env_cuda(ReadOnly(), prefix, "win32", {})
    assert note["why"].startswith("skipped (PermissionError")


def test_import_records_a_note_and_does_not_import_cupy():
    assert isinstance(xclean.CUDA_PATH_NOTE, dict) and set(xclean.CUDA_PATH_NOTE) == {"before", "set_to",
                                                                                       "path_first", "why"}
    r = subprocess.run([sys.executable, "-c", "import sys, xclean; print('cupy' in sys.modules, "
                        "xclean.CUDA_PATH_NOTE['why'])"], capture_output=True, text=True,
                       env=dict(os.environ, XCLEAN_KEEP_CUDA_PATH="1"))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[0] == "False" and "XCLEAN_KEEP_CUDA_PATH" in r.stdout


@pytest.mark.skipif(not (sys.platform == "win32"
                         and os.path.isfile(os.path.join(sys.prefix, "Library", "bin", ENV_NVRTC_DLL))),
                    reason="needs Windows and an environment created from environment.yml")
def test_this_environment_and_its_children():
    env = {k: v for k, v in os.environ.items() if k != "XCLEAN_KEEP_CUDA_PATH"}
    code = ("import os, subprocess, sys, xclean; print(os.environ['CUDA_PATH']); "
            "print(os.environ['PATH'].split(os.pathsep)[0]); "
            "print(subprocess.run([sys.executable, '-c', 'import os; print(os.environ[\"CUDA_PATH\"])'], "
            "capture_output=True, text=True).stdout.strip())")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    cuda_path, first, child = r.stdout.splitlines()
    lib = os.path.join(sys.prefix, "Library")
    assert os.path.normcase(cuda_path) == os.path.normcase(lib) == os.path.normcase(child)
    assert os.path.normcase(first) == os.path.normcase(os.path.join(lib, "bin"))
