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


def test_not_windows_or_linux(tmp_path):
    prefix, _lib, _bin = _env(tmp_path)
    environ = {"PATH": "other"}
    note = _use_env_cuda(environ, prefix, "darwin", {})
    assert environ == {"PATH": "other"} and "not Windows or Linux" in note["why"]


# --------------------------------------------------------------------------------------------------------------- Linux
def _linux_env(tmp_path, where="lib", names=("libnvrtc.so.13", "libnvrtc.so.13.3.29", "libnvrtc-builtins.so.13.3")):
    prefix = tmp_path / "env"
    d = prefix / where
    d.mkdir(parents=True)
    for n in names:
        (d / n).write_bytes(b"")
    return str(prefix), str(d)


def test_linux_loads_the_environment_compiler(tmp_path):
    prefix, d = _linux_env(tmp_path)
    loaded = []
    environ = {"CUDA_PATH": "/usr/local/cuda", "PATH": "other"}
    note = _use_env_cuda(environ, prefix, "linux", {}, preload=loaded.append, machine="x86_64")
    assert loaded == [os.path.join(d, "libnvrtc.so.13")]                 # the soname link, never the builtins
    assert environ == {"CUDA_PATH": prefix, "PATH": "other"}             # PATH is not touched on Linux
    assert note["before"] == "/usr/local/cuda" and note["set_to"] == prefix and note["path_first"] is None
    assert "libnvrtc.so.13" in note["why"]


def test_linux_targets_layout_on_arm64(tmp_path):
    prefix, d = _linux_env(tmp_path, where=os.path.join("targets", "sbsa-linux", "lib"))
    loaded = []
    note = _use_env_cuda({}, prefix, "linux", {}, preload=loaded.append, machine="aarch64")
    assert loaded == [os.path.join(d, "libnvrtc.so.13")] and note["set_to"] == prefix


def test_linux_without_a_compiler_leaves_the_toolkit_in_charge(tmp_path):
    prefix, _d = _linux_env(tmp_path, names=("libnvrtc-builtins.so.13.3",))
    loaded = []
    environ = {"CUDA_PATH": "/usr/local/cuda"}
    note = _use_env_cuda(environ, prefix, "linux", {}, preload=loaded.append, machine="aarch64")
    assert loaded == [] and environ == {"CUDA_PATH": "/usr/local/cuda"} and note["set_to"] is None
    assert "installed CUDA Toolkit" in note["why"]


def test_linux_opt_out_and_cupy_first(tmp_path):
    prefix, _d = _linux_env(tmp_path)
    loaded = []
    environ = {"XCLEAN_KEEP_CUDA_PATH": "1"}
    assert "XCLEAN_KEEP_CUDA_PATH" in _use_env_cuda(environ, prefix, "linux", {}, preload=loaded.append)["why"]
    assert "already imported" in _use_env_cuda({}, prefix, "linux", {"cupy": object()}, preload=loaded.append)["why"]
    assert loaded == []


def test_linux_load_failure_changes_nothing(tmp_path):
    prefix, _d = _linux_env(tmp_path)

    def fail(p):
        raise OSError("wrong ELF class")
    environ = {"CUDA_PATH": "/usr/local/cuda"}
    note = _use_env_cuda(environ, prefix, "linux", {}, preload=fail, machine="x86_64")
    assert environ == {"CUDA_PATH": "/usr/local/cuda"} and note["set_to"] is None
    assert note["why"].startswith("skipped (OSError")


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


@pytest.mark.skipif(not (sys.platform.startswith("linux") and xclean._env_nvrtc_so(sys.prefix)),
                    reason="needs Linux and an environment created from environment.yml")
def test_this_linux_environment_and_its_children():
    """A fresh process that imports xclean has CUDA_PATH = the environment and this environment's libnvrtc mapped in;
    a child process that imports xclean does the same."""
    env = {k: v for k, v in os.environ.items() if k != "XCLEAN_KEEP_CUDA_PATH"}
    code = ("import os, xclean; print(os.environ['CUDA_PATH']); "
            "print(any('libnvrtc.so' in ln and ln.split()[-1].startswith(os.path.realpath(os.sys.prefix)) "
            "for ln in open('/proc/self/maps')))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    cuda_path, mapped = r.stdout.split()
    assert os.path.realpath(cuda_path) == os.path.realpath(sys.prefix) and mapped == "True"
