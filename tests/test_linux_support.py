# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Linux and NVIDIA Jetson support (0.2.0): `xclean doctor`'s platform line, its CUDA 12 / CUDA 13 rules (cupy-cuda12x
on JetPack 6, cupy-cuda13x elsewhere), the Linux CPU topology and the timing harness's process allow-list.  Runs on
any OS: the platform, sysfs and device-tree inputs are stand-ins."""
import os
import types

from xclean import doctor as D
from xclean.live import harness as H


def _status(checks, name):
    return [c.status for c in checks if c.name == name]


def _cupy(runtime=None, driver=13020, major=8, minor=7):
    rt = types.SimpleNamespace(driverGetVersion=lambda: driver, getDeviceCount=lambda: 1,
                               getDeviceProperties=lambda i: {"name": b"Orin", "major": major, "minor": minor,
                                                              "totalGlobalMem": 8 * 2 ** 30},
                               memGetInfo=lambda: (5 * 2 ** 30, 8 * 2 ** 30))
    if runtime is not None:
        rt.runtimeGetVersion = lambda: runtime

    class Dev:
        def __init__(self, i):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    return types.SimpleNamespace(__file__="/nowhere/cupy/__init__.py", cuda=types.SimpleNamespace(runtime=rt, Device=Dev))


def _jetson_root(tmp_path, model="NVIDIA Jetson AGX Orin Developer Kit",
                 release="# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64"):
    (tmp_path / "proc" / "device-tree").mkdir(parents=True)
    (tmp_path / "proc" / "device-tree" / "model").write_bytes(model.encode() + b"\x00")
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "nv_tegra_release").write_text(release + "\n")
    return str(tmp_path)


# ------------------------------------------------------------------------------------------------------------ platform
def test_platform_lines(tmp_path):
    assert _status(D.check_platform("Windows", "AMD64"), "Platform") == [D.OK]
    c = D.check_platform("Linux", "x86_64", root=str(tmp_path))[0]
    assert c.status == D.OK and c.detail.startswith("Linux x86_64") and "Jetson" not in c.detail
    c = D.check_platform("Linux", "aarch64", root=_jetson_root(tmp_path))[0]
    assert c.status == D.OK and "Jetson AGX Orin" in c.detail and "L4T R36.4.3" in c.detail
    c = D.check_platform("Darwin", "arm64")[0]
    assert c.status == D.FAIL and c.fix


def test_jetson_info(tmp_path):
    assert D.jetson_info(str(tmp_path)) is None
    j = D.jetson_info(_jetson_root(tmp_path, model="NVIDIA Jetson AGX Thor Developer Kit",
                                   release="# R38 (release), REVISION: 2.1, GCID: 1, BOARD: generic, EABI: aarch64"))
    assert j == {"model": "NVIDIA Jetson AGX Thor Developer Kit", "l4t": "R38.2.1"}


# ------------------------------------------------------------------------------------------------- CUDA 12 vs CUDA 13
def test_cuda_major_of_cupy():
    assert D._cupy_cuda_major(_cupy(runtime=12060)) == 12
    assert D._cupy_cuda_major(_cupy(runtime=13000)) == 13
    assert D._cupy_cuda_major(_cupy()) == 13                 # unknown -> the tested CUDA 13


def test_driver_rule_follows_cupy(monkeypatch):
    monkeypatch.setattr(D, "_nvidia_smi", lambda q: None)    # Jetson: no nvidia-smi, the driver API answers
    assert _status(D.check_driver(_cupy(runtime=12060, driver=12060)), "NVIDIA driver") == [D.OK]    # JetPack 6
    assert _status(D.check_driver(_cupy(runtime=13000, driver=12060)), "NVIDIA driver") == [D.FAIL]  # wrong CuPy
    assert _status(D.check_driver(_cupy(runtime=13000, driver=13020)), "NVIDIA driver") == [D.OK]    # JetPack 7.2
    c = D.check_driver(_cupy(runtime=13000, driver=12060))[0]
    assert "JetPack" in c.fix and D.LINUX_DOC in c.fix


def test_compute_capability_rule_follows_cupy(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _status(D.check_gpu(_cupy(runtime=13000, major=8, minor=7)), "GPU") == [D.OK]     # Orin
    assert _status(D.check_gpu(_cupy(runtime=13000, major=11, minor=0)), "GPU") == [D.OK]    # Thor
    assert _status(D.check_gpu(_cupy(runtime=13000, major=7, minor=2)), "GPU") == [D.FAIL]   # Xavier, CUDA 13
    assert _status(D.check_gpu(_cupy(runtime=12060, major=7, minor=2)), "GPU") == [D.OK]     # Xavier, CUDA 12


def test_cuda_libraries_with_linux_names(monkeypatch):
    maps = ["/v/lib/python3.10/site-packages/cupy_backends/cuda/libs/nvrtc.cpython-310-aarch64-linux-gnu.so",
            "/usr/local/cuda-12.6/targets/sbsa-linux/lib/libnvrtc.so.12.6.85",
            "/usr/local/cuda-12.6/targets/sbsa-linux/lib/libnvrtc-builtins.so.12.6.85",
            "/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1.1",
            "/env/Library/bin/nvrtc64_130_0.dll",
            "/env/Lib/site-packages/cupy_backends/cuda/libs/nvrtc.cp312-win_amd64.pyd"]
    monkeypatch.setattr(D, "_loaded_modules", lambda: maps)
    got = D._cuda_libs()
    assert got["nvrtc"] == [maps[1], maps[4]]                # never CuPy's own Python modules
    assert got["nvrtc-builtins"] == [maps[2]] and got["nvcuda"] == [maps[3]]


def test_no_driver_installed(monkeypatch):
    monkeypatch.setattr(D, "_nvidia_smi", lambda q: None)
    c = D.check_driver(_cupy(runtime=13000, driver=0))[0]
    assert c.status == D.FAIL and "no NVIDIA driver found" in c.detail


# ------------------------------------------------------------------------------------------- Linux CPU topology (sysfs)
def _sysfs(tmp_path, siblings):
    for cpu, txt in siblings.items():
        d = tmp_path / ("cpu%d" % cpu) / "topology"
        d.mkdir(parents=True)
        (d / "thread_siblings_list").write_text(txt + "\n")
    (tmp_path / "cpufreq").mkdir()                           # other entries are ignored
    return str(tmp_path)


def test_linux_core_groups_smt(tmp_path):
    # 4 cores with SMT, Linux numbering: siblings are i and i + 4
    root = _sysfs(tmp_path, {0: "0,4", 1: "1,5", 2: "2-2,6", 3: "3,7", 4: "0,4", 5: "1,5", 6: "2,6", 7: "3,7"})
    assert H.linux_core_groups(root) == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_linux_core_groups_jetson(tmp_path):
    # Jetson: no SMT, and cores 4-5 offline in a low-power mode (no topology folder)
    root = _sysfs(tmp_path, {0: "0", 1: "1", 2: "2", 3: "3"})
    assert H.linux_core_groups(root) == [[0], [1], [2], [3]]
    assert H.linux_core_groups(str(tmp_path / "missing")) == []


# ------------------------------------------------------------------------------------------------ process allow-list
def test_allow_list():
    assert H.allowed("explorer.exe") and H.allowed("Explorer.EXE")
    assert H.allowed("gnome-shell") and H.allowed("nvargus-daemon")
    assert not H.allowed("python") and not H.allowed("")
    assert H.gpu_app_allowed("C:\\Windows\\explorer.exe")
    assert H.gpu_app_allowed("/usr/lib/xorg/Xorg")
    assert not H.gpu_app_allowed("/usr/bin/python3")


def test_allow_prefixes_only_off_windows(monkeypatch):
    monkeypatch.setattr(H.sys, "platform", "linux")
    assert H.allowed("kworker/3:1") and H.allowed("systemd-journald") and H.allowed("ksoftirqd/0")
    monkeypatch.setattr(H.sys, "platform", "win32")
    assert not H.allowed("kworker/3:1")                      # Windows keeps its exact list


def test_linux_doc_exists():
    here = os.path.dirname(os.path.abspath(__file__))
    assert os.path.isfile(os.path.join(here, os.pardir, D.LINUX_DOC))
