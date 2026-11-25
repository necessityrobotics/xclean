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
    note = _use_env_cuda(environ, prefix, "win32", {}, paths=[])
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
    note = _use_env_cuda(environ, prefix, "linux", {}, preload=loaded.append, machine="aarch64", paths=[])
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


# ---------------------------------------------------------------- the pip-only install (the `gpu` extra, 0.3.1)

def _pip_site(tmp_path, windows, nvrtc=True, headers=True, sub="x86_64", cupy="cupy_cuda13x-13.6.0.dist-info"):
    """A fake site-packages with the layout of the NVIDIA pip packages: nvidia/cu13/{bin[/x86_64] or lib, include},
    and the dist-info folder of CuPy for CUDA 13 (or of another CuPy, or none)."""
    site = tmp_path / "site-packages"
    root = site / "nvidia" / "cu13"
    if cupy:
        (site / cupy).mkdir(parents=True)
    lib_dir = (root / "bin" / sub if sub else root / "bin") if windows else root / "lib"
    lib_dir.mkdir(parents=True)
    (root / "include").mkdir()
    if headers:
        (root / "include" / "cuda.h").write_bytes(b"")
    if nvrtc:
        if windows:
            (lib_dir / ENV_NVRTC_DLL).write_bytes(b"")
            (lib_dir / "nvrtc-builtins64_133.dll").write_bytes(b"")
        else:
            for n in ("libnvrtc.so.13", "libnvrtc.so.13.3.33", "libnvrtc-builtins.so.13.3", "libnvrtc.alt.so.13"):
                (lib_dir / n).write_bytes(b"")
    return str(site), str(root), str(lib_dir)


def _bare_env(tmp_path):
    prefix = tmp_path / "venv"                   # an environment without a conda compiler (a plain venv)
    prefix.mkdir()
    return str(prefix)


@pytest.mark.parametrize("sub", ["x86_64", ""])
def test_windows_uses_the_nvidia_pip_packages(tmp_path, sub):
    site, root, bin_dir = _pip_site(tmp_path, True, sub=sub)
    environ = {"CUDA_PATH": TOOLKIT, "PATH": _path(os.path.join(TOOLKIT, "bin"), "other")}
    added = []
    note = _use_env_cuda(environ, _bare_env(tmp_path), "win32", {}, paths=["", site], add_dll=added.append)
    assert environ["CUDA_PATH"] == root                                  # root/include has the CUDA headers
    assert environ["PATH"] == _path(bin_dir, os.path.join(TOOLKIT, "bin"), "other")
    assert added == [bin_dir]                                            # the compiler finds its helper DLL there
    assert note["before"] == TOOLKIT and note["set_to"] == root and note["path_first"] == bin_dir
    assert "NVIDIA pip packages" in note["why"]


def test_windows_pip_packages_twice_changes_nothing_more(tmp_path):
    site, root, bin_dir = _pip_site(tmp_path, True)
    prefix = _bare_env(tmp_path)
    environ = {"PATH": "other"}
    _use_env_cuda(environ, prefix, "win32", {}, paths=[site], add_dll=lambda d: None)
    once = dict(environ)
    note = _use_env_cuda(environ, prefix, "win32", {}, paths=[site], add_dll=lambda d: None)
    assert environ == once == {"CUDA_PATH": root, "PATH": _path(bin_dir, "other")} and note["before"] == root


def test_the_conda_compiler_has_priority_over_the_pip_packages(tmp_path):
    prefix, lib, bin_dir = _env(tmp_path)                                # an environment with the conda compiler
    site, _root, _pip_bin = _pip_site(tmp_path, True)
    added = []
    note = _use_env_cuda({"PATH": "other"}, prefix, "win32", {}, paths=[site], add_dll=added.append)
    assert note["set_to"] == lib and added == [] and "NVIDIA pip packages" not in note["why"]


@pytest.mark.parametrize("missing", ["nvrtc", "headers"])
def test_pip_packages_need_the_compiler_and_the_headers(tmp_path, missing):
    site, _root, _d = _pip_site(tmp_path, True, nvrtc=missing != "nvrtc", headers=missing != "headers")
    environ = {"CUDA_PATH": TOOLKIT, "PATH": "other"}
    added = []
    note = _use_env_cuda(environ, _bare_env(tmp_path), "win32", {}, paths=[site], add_dll=added.append)
    assert environ == {"CUDA_PATH": TOOLKIT, "PATH": "other"} and note["set_to"] is None and added == []


@pytest.mark.parametrize("cupy", [None, "cupy_cuda12x-13.6.0.dist-info"])
@pytest.mark.parametrize("windows", [True, False])
def test_pip_packages_are_used_only_beside_cupy_for_cuda_13(tmp_path, cupy, windows):
    """Other software can bring the NVIDIA packages; CuPy for CUDA 12 must not get CUDA 13 headers."""
    site, _root, _d = _pip_site(tmp_path, windows, cupy=cupy)
    environ = {"CUDA_PATH": TOOLKIT, "PATH": "other"}
    added, loaded = [], []
    note = _use_env_cuda(environ, _bare_env(tmp_path), "win32" if windows else "linux", {}, preload=loaded.append,
                         machine="x86_64", paths=[site], add_dll=added.append, jetson=False)
    assert environ == {"CUDA_PATH": TOOLKIT, "PATH": "other"} and note["set_to"] is None
    assert added == [] and loaded == []


def test_cupy_for_cuda_13_on_another_path_entry_counts(tmp_path):
    site, root, _d = _pip_site(tmp_path, False, cupy=None)
    other = tmp_path / "user-site"
    (other / "cupy_cuda13x-13.6.0.dist-info").mkdir(parents=True)
    note = _use_env_cuda({}, _bare_env(tmp_path), "linux", {}, preload=lambda p: None, machine="x86_64",
                         paths=[site, str(other)], jetson=False)
    assert note["set_to"] == root


def test_a_jetson_keeps_its_own_cuda(tmp_path):
    """On a Jetson the pip packages are not used, also when they are installed by mistake."""
    site, _root, _d = _pip_site(tmp_path, False)
    environ = {"CUDA_PATH": "/usr/local/cuda"}
    loaded = []
    note = _use_env_cuda(environ, _bare_env(tmp_path), "linux", {}, preload=loaded.append, machine="aarch64",
                         paths=[site], jetson=True)
    assert environ == {"CUDA_PATH": "/usr/local/cuda"} and note["set_to"] is None and loaded == []
    assert "installed CUDA Toolkit" in note["why"]


def test_pip_packages_opt_out_and_cupy_first(tmp_path):
    site, _root, _d = _pip_site(tmp_path, True)
    prefix = _bare_env(tmp_path)
    environ = {"CUDA_PATH": TOOLKIT, "PATH": "other", "XCLEAN_KEEP_CUDA_PATH": "1"}
    assert "XCLEAN_KEEP_CUDA_PATH" in _use_env_cuda(environ, prefix, "win32", {}, paths=[site])["why"]
    assert environ["CUDA_PATH"] == TOOLKIT
    environ = {"PATH": "other"}
    assert "already imported" in _use_env_cuda(environ, prefix, "win32", {"cupy": object()}, paths=[site])["why"]
    assert environ == {"PATH": "other"}


def test_linux_uses_the_nvidia_pip_packages(tmp_path):
    site, root, lib_dir = _pip_site(tmp_path, False)
    environ = {"CUDA_PATH": "/usr/local/cuda", "PATH": "other"}
    loaded = []
    note = _use_env_cuda(environ, _bare_env(tmp_path), "linux", {}, preload=loaded.append, machine="aarch64",
                         paths=[site], jetson=False)
    assert loaded == [os.path.join(lib_dir, "libnvrtc.so.13")]           # the shortest name, not the builtins or alt file
    assert environ == {"CUDA_PATH": root, "PATH": "other"}               # PATH is not touched on Linux
    assert note["set_to"] == root and "NVIDIA pip packages" in note["why"]


def test_linux_conda_compiler_has_priority_over_the_pip_packages(tmp_path):
    prefix, lib_dir = _linux_env(tmp_path)
    site, _root, _d = _pip_site(tmp_path, False)
    loaded = []
    note = _use_env_cuda({}, prefix, "linux", {}, preload=loaded.append, machine="x86_64", paths=[site])
    assert note["set_to"] == prefix and loaded and os.path.dirname(loaded[0]) == lib_dir


def test_pip_packages_load_failure_changes_nothing(tmp_path):
    site, _root, _d = _pip_site(tmp_path, False)

    def fail(_p):
        raise OSError("cannot load")
    environ = {"CUDA_PATH": "/usr/local/cuda"}
    note = _use_env_cuda(environ, _bare_env(tmp_path), "linux", {}, preload=fail, machine="x86_64", paths=[site],
                         jetson=False)
    assert environ == {"CUDA_PATH": "/usr/local/cuda"} and note["set_to"] is None and "skipped" in note["why"]


def _pip_only_environment():
    """The NVIDIA pip packages of THIS environment, when it has no conda compiler (a pip-only install); else None."""
    win = sys.platform == "win32"
    if not (win or sys.platform.startswith("linux")):
        return None
    conda = (os.path.isfile(os.path.join(sys.prefix, "Library", "bin", ENV_NVRTC_DLL)) if win
             else xclean._env_nvrtc_so(sys.prefix) is not None)
    if conda or os.path.isfile(xclean.JETSON_FILE):
        return None
    return xclean._pip_cuda(sys.path, win)


@pytest.mark.skipif(_pip_only_environment() is None,
                    reason="needs a pip-only environment with the gpu extra (nvidia-cuda-nvrtc, nvidia-cuda-runtime)")
def test_this_pip_environment_and_its_children():
    """A fresh process that imports xclean has CUDA_PATH = <site-packages>/nvidia/cu13, also when a CUDA Toolkit is
    installed on the machine; a child process that imports xclean has the same."""
    root, lib_dir, so = _pip_only_environment()

    def same(a, b):
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    env = {k: v for k, v in os.environ.items() if k != "XCLEAN_KEEP_CUDA_PATH"}
    env["CUDA_PATH"] = TOOLKIT                                   # an installed toolkit must not win
    # this pytest process imported xclean already: take its PATH entry away, so that the fresh process must add it
    env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if p and not same(p, lib_dir))
    # the child process starts with the toolkit's CUDA_PATH again, so it must repeat the step itself
    child_code = ("import os, xclean; print(os.environ['CUDA_PATH']); "
                  "print(any('libnvrtc.so' in ln and 'cu13' in ln for ln in open('/proc/self/maps')) "
                  "if os.path.exists('/proc/self/maps') else 'no maps')")
    code = ("import os, subprocess, sys, xclean; print(os.environ['CUDA_PATH']); "
            "print(xclean.CUDA_PATH_NOTE['why']); print(os.environ['PATH'].split(os.pathsep)[0]); "
            "print(subprocess.run([sys.executable, '-c', %r], capture_output=True, text=True, "
            "env=dict(os.environ, CUDA_PATH=%r)).stdout.strip())" % (child_code, TOOLKIT))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    cuda_path, why, first, child, child_maps = r.stdout.splitlines()
    assert same(cuda_path, root) and same(child, root)
    assert "NVIDIA pip packages" in why
    if sys.platform == "win32":
        assert same(first, lib_dir)                              # the compiler's folder is first on PATH
    else:
        assert child_maps == "True", "the pip compiler %s is not loaded in the child process" % so
