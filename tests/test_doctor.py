# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""`xclean doctor`: each check's verdict on stand-in machines (no GPU, pip OpenBLAS vs conda MKL / OpenBLAS, the CPU
kernel, the text encoding, the CUDA compiler and headers, toolkit vs self-contained, old driver, small GPU, OAK), the
exit code, and that nothing ever ends in a traceback."""
import json
import os
import types

import pytest

from xclean import doctor as D


def _status(checks, name):
    return [c.status for c in checks if c.name == name]


# ------------------------------------------------------------------------------------------------ versions, Python
def test_version_check():
    assert D.version_check("x", "2.3.5", "2.3", "2.0", "fix").status == D.OK
    assert D.version_check("x", "2.4.0", "2.3", "2.0", "fix").status == D.WARN
    assert D.version_check("x", "1.26.4", "2.3", "2.0", "fix").status == D.FAIL
    assert D.version_check("x", None, "2.3", "2.0", "fix").status == D.FAIL
    assert D.version_check("x", "5.0.0.93", "5.0", "4.10", "fix").status == D.OK
    assert D._vtuple("13.6.0rc1") == (13, 6, 0)


@pytest.mark.parametrize("vi, want", [((3, 12, 4), D.OK), ((3, 11, 2), D.WARN), ((3, 9, 1), D.FAIL)])
def test_python(vi, want):
    assert D.check_python(vi)[0].status == want


# ------------------------------------------------------------------------------------------------------ LAPACK, BLAS
def _blas_machine(monkeypatch, tmp_path, pip_libs=(), loaded=(), conda=None, core="SkylakeX"):
    """A stand-in site-packages: pip_libs = packages whose wheel bundles OpenBLAS (<pkg>.libs/), loaded = library
    file names mapped into the process, conda = {package: [conda-meta names]}."""
    site = tmp_path / "site-packages"
    fakes = {}
    for pkg in ("numpy", "scipy"):
        (site / pkg).mkdir(parents=True, exist_ok=True)
        fakes[pkg] = types.SimpleNamespace(__file__=str(site / pkg / "__init__.py"), __version__="0")
    paths = []
    for pkg in pip_libs:
        d = site / (pkg + ".libs")
        d.mkdir(exist_ok=True)
        f = d / ("libscipy_openblas%s-abc.dll" % ("64_" if pkg == "numpy" else ""))
        f.write_bytes(b"")
        paths.append(str(f))
    real = D._import

    def imp(name):
        if name in fakes:
            return fakes[name]
        if name in ("threadpoolctl", "scipy.linalg.cython_lapack"):
            raise ImportError(name)
        return real(name)
    monkeypatch.setattr(D, "_import", imp)
    monkeypatch.setattr(D, "_loaded_modules", lambda: [p for p in paths if os.path.basename(p) in loaded] +
                        [os.path.join(str(tmp_path), "Library", "bin", x) for x in loaded if x.startswith("mkl")])
    monkeypatch.setattr(D, "_conda_meta", lambda name: (conda or {}).get(name, []))
    monkeypatch.setattr(D, "_openblas_corename", lambda path: core)


def _named(checks, name):
    return [c for c in checks if c.name == name][0]


def test_blas_pip_openblas_is_ok(monkeypatch, tmp_path):
    _blas_machine(monkeypatch, tmp_path, pip_libs=("numpy", "scipy"),
                  loaded=("libscipy_openblas-abc.dll", "libscipy_openblas64_-abc.dll"))
    assert D.blas_origin("scipy")[:2] == ("openblas", "pip")
    chk = D.check_blas()
    assert _named(chk, "LAPACK (SciPy)").status == D.OK and "SkylakeX" in _named(chk, "LAPACK (SciPy)").detail
    assert _named(chk, "BLAS (NumPy)").status == D.OK


def test_blas_other_cpu_kernel_is_a_warning_without_reinstall(monkeypatch, tmp_path):
    _blas_machine(monkeypatch, tmp_path, pip_libs=("numpy", "scipy"), loaded=("libscipy_openblas-abc.dll",),
                  core="Haswell")
    c = _named(D.check_blas(), "LAPACK (SciPy)")
    assert c.status == D.WARN and "Haswell" in c.detail and "keeps the GPU front end" in c.detail
    assert "environment.yml" not in c.fix                    # reinstalling does not change the CPU


def test_blas_conda_mkl_points_to_the_environment_file(monkeypatch, tmp_path):
    mkl = ["libblas-3.11.0-11_h8455456_mkl"]
    _blas_machine(monkeypatch, tmp_path, loaded=("mkl_core.2.dll",),
                  conda={"numpy": ["numpy-2.3.5-py312"], "scipy": ["scipy-1.16.3-py312"], "libblas": mkl})
    chk = D.check_blas()
    lap, bl = _named(chk, "LAPACK (SciPy)"), _named(chk, "BLAS (NumPy)")
    assert lap.status == D.WARN and "MKL" in lap.detail and "keeps the GPU front end" in lap.detail
    assert "environment.yml" in lap.fix and "SciPy from pip" in lap.fix and "libblas" not in lap.fix
    assert bl.status == D.WARN and "environment.yml" in bl.fix


def test_blas_conda_openblas_is_a_warning(monkeypatch, tmp_path):
    _blas_machine(monkeypatch, tmp_path, conda={"scipy": ["scipy-1.16.3-py312"],
                                                "libblas": ["libblas-3.11.0-5_h0adab6e_openblas"]})
    c = _named(D.check_blas(), "LAPACK (SciPy)")
    assert c.status == D.WARN and "conda package" in c.detail and "environment.yml" in c.fix


def test_blas_mixed_pip_scipy_conda_numpy(monkeypatch, tmp_path):
    _blas_machine(monkeypatch, tmp_path, pip_libs=("scipy",), loaded=("libscipy_openblas-abc.dll", "mkl_core.2.dll"),
                  conda={"numpy": ["numpy-2.3.5-py312"], "libblas": ["libblas-3.11.0-11_h8455456_mkl"]})
    chk = D.check_blas()
    assert _named(chk, "LAPACK (SciPy)").status == D.OK          # numba's solve / inverse use SciPy's LAPACK
    assert _named(chk, "BLAS (NumPy)").status == D.WARN          # but CPU results need NumPy's OpenBLAS too


def test_openblas_corename_of_a_missing_library_is_none(tmp_path):
    assert D._openblas_corename(str(tmp_path / "no_such_library.dll")) is None


# ----------------------------------------------------------------------------------------------------- text encoding
def test_encoding(monkeypatch):
    assert D.check_encoding(utf8_mode=1, encoding="cp1252")[0].status == D.OK
    assert D.check_encoding(utf8_mode=0, encoding="UTF-8")[0].status == D.OK
    monkeypatch.setattr(D, "_kernel_sources_not_ascii", lambda pkg_dir=None: [])
    assert D.check_encoding(utf8_mode=0, encoding="cp1252")[0].status == D.OK
    monkeypatch.setattr(D, "_kernel_sources_not_ascii", lambda pkg_dir=None: ["backend/kernels.py"])
    c = D.check_encoding(utf8_mode=0, encoding="cp1252")[0]
    assert c.status == D.WARN and "backend/kernels.py" in c.detail and "PYTHONUTF8=1" in c.fix


def test_kernel_source_scan(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "ascii_kernel.py").write_text('# caf\u00e9 in a comment only\nSRC = r"""extern "C" __global__ void '
                                              'k() {}"""\n', encoding="utf-8")
    (tmp_path / "sub" / "bad_kernel.py").write_text('SRC = r"""__global__ void k() { /* \u2605 */ }"""\n',
                                                    encoding="utf-8")
    (tmp_path / "no_kernel.py").write_text('X = "\u2605"\n', encoding="utf-8")
    assert D._kernel_sources_not_ascii(str(tmp_path)) == ["sub/bad_kernel.py"]


# ------------------------------------------------------------------------------------------- driver, GPU, CuPy fakes
def _fake_cupy(n_dev=1, major=8, minor=9, free_gb=20.0, driver=13040, count_error=None, kernel=45.0,
               nvrtc=(13, 3), runtime=None):
    rt = types.SimpleNamespace()
    rt.driverGetVersion = lambda: driver
    if runtime is not None:                      # the CUDA version CuPy was built for (cupy-cuda12x: 12xxx)
        rt.runtimeGetVersion = lambda: runtime

    def count():
        if count_error:
            raise RuntimeError(count_error)
        return n_dev
    rt.getDeviceCount = count
    rt.getDeviceProperties = lambda i: {"name": b"Fake GPU", "major": major, "minor": minor,
                                        "totalGlobalMem": 32 * 2 ** 30}
    rt.memGetInfo = lambda: (int(free_gb * 2 ** 30), 32 * 2 ** 30)
    rt.deviceSynchronize = lambda: None

    class Dev:
        def __init__(self, i):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    cuda = types.SimpleNamespace(runtime=rt, Device=Dev, get_cuda_path=lambda: None,
                                 nvrtc=types.SimpleNamespace(getVersion=lambda: nvrtc))
    return types.SimpleNamespace(__version__="13.6.0", __file__="/nowhere/cupy/__init__.py", cuda=cuda,
                                 float32="float32", arange=lambda n, dtype=None: list(range(n)),
                                 sum=lambda x: kernel)


def test_driver(monkeypatch):
    monkeypatch.setattr(D, "_nvidia_smi", lambda q: [["616.92", "Fake GPU"]])
    assert D.check_driver(_fake_cupy(driver=13040))[0].status == D.OK
    assert D.check_driver(_fake_cupy(driver=12080))[0].status == D.FAIL
    monkeypatch.setattr(D, "_nvidia_smi", lambda q: None)
    c = D.check_driver(None)[0]
    assert c.status == D.FAIL and "driver" in c.fix


def test_gpu_hidden_and_missing(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    c = D.check_gpu(_fake_cupy(count_error="cudaErrorNoDevice"))[0]
    assert c.status == D.FAIL and "CUDA_VISIBLE_DEVICES" in c.detail and c.fix
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    c = D.check_gpu(_fake_cupy(n_dev=0))[0]
    assert c.status == D.FAIL and c.fix
    c = D.check_gpu(None, "ImportError: no cupy")[0]
    assert c.status == D.FAIL and "CuPy" in c.detail


def test_gpu_capability_and_memory(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    ok = D.check_gpu(_fake_cupy())
    assert _status(ok, "GPU") == [D.OK] and _status(ok, "GPU memory") == [D.OK]
    old = D.check_gpu(_fake_cupy(major=6, minor=1))
    assert _status(old, "GPU") == [D.FAIL]
    small = D.check_gpu(_fake_cupy(free_gb=1.0))
    assert _status(small, "GPU memory") == [D.WARN]


def test_cupy_kernel(monkeypatch):
    monkeypatch.setattr(D, "_loaded_modules", lambda: [])
    chk, facts = D.check_cupy(_fake_cupy(), None, gpu_ok=True)
    assert _status(chk, "GPU kernel") == [D.OK]
    chk, _ = D.check_cupy(_fake_cupy(kernel=44.0), None, gpu_ok=True)
    assert _status(chk, "GPU kernel") == [D.FAIL]
    chk, _ = D.check_cupy(_fake_cupy(), None, gpu_ok=False)
    assert _status(chk, "GPU kernel") == [D.FAIL]
    chk, _ = D.check_cupy(None, "ImportError: DLL load failed")
    assert chk[0].status == D.FAIL and "DLL load failed" in chk[0].detail


def test_compiler_version_and_headers(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "_loaded_modules", lambda: [])
    hdr = [(str(tmp_path / "include"), "nvidia-cuda-runtime package")]
    monkeypatch.setattr(D, "header_dirs", lambda cp, cuda_path, nv=None: hdr)
    chk, facts = D.check_cupy(_fake_cupy(nvrtc=(13, 3)), None, gpu_ok=False)
    assert _status(chk, "CUDA compiler") == [D.OK] and _status(chk, "CUDA headers") == [D.OK]
    assert facts["nvrtc"] == (13, 3) and facts["headers"] == hdr
    chk, _ = D.check_cupy(_fake_cupy(nvrtc=(13, 0)), None, gpu_ok=False)
    c = _named(chk, "CUDA compiler")
    assert c.status == D.WARN and "13.3" in c.detail and "environment.yml" in c.fix
    monkeypatch.setattr(D, "header_dirs", lambda cp, cuda_path, nv=None: [])
    chk, _ = D.check_cupy(_fake_cupy(), None, gpu_ok=False)
    assert _status(chk, "CUDA headers") == [D.WARN]


def test_compiler_that_does_not_load_is_a_failure(monkeypatch):
    monkeypatch.setattr(D, "_loaded_modules", lambda: [])
    cp = _fake_cupy()

    def broken():
        raise RuntimeError("nvrtc64_130_0.dll not found")
    cp.cuda.nvrtc.getVersion = broken
    chk, facts = D.check_cupy(cp, None, gpu_ok=False)
    assert _status(chk, "CUDA compiler") == [D.FAIL] and facts["nvrtc"] is None


def test_header_dirs_follow_cupys_order(tmp_path):
    tk, wheel = tmp_path / "toolkit", tmp_path / "site-packages" / "nvidia" / "cu13" / "include"
    for d in (tk / "include", wheel):
        d.mkdir(parents=True)
        (d / "cuda_fp16.h").write_text("")
    cp = _fake_cupy()
    cp.cuda.compiler = types.SimpleNamespace(_get_extra_include_dir_opts=lambda: ("-I" + str(wheel),))
    assert D.header_dirs(cp, str(tk)) == [(str(tk / "include"), "CUDA path"),
                                          (str(wheel), "nvidia-cuda-runtime package")]
    assert D.header_dirs(cp, str(tmp_path / "no_toolkit")) == [(str(wheel), "nvidia-cuda-runtime package")]


def test_toolkit_and_self_contained(monkeypatch, tmp_path):
    env, tk = tmp_path / "env", tmp_path / "toolkit"
    monkeypatch.setattr(D.sys, "prefix", str(env))
    monkeypatch.setattr(D.shutil, "which", lambda name: None)
    inside = str(env / "Library" / "bin" / "nvrtc64_130_0.dll")
    outside = str(tk / "bin" / "x64" / "nvrtc64_130_0.dll")
    hdr_in = (str(env / "Lib" / "site-packages" / "nvidia" / "cu13" / "include"), "nvidia-cuda-runtime package")
    hdr_out = (str(tk / "include"), "CUDA path")
    applied = {"before": str(tk), "set_to": str(env / "Library"), "path_first": str(env / "Library" / "bin"),
               "why": "the environment's own CUDA kernel compiler and headers"}
    good = D.check_toolkit({"packaging": "pip", "cuda_path": str(env / "Library"), "loaded": {"nvrtc": [inside]},
                            "nvrtc": (13, 3), "headers": [hdr_in]}, applied)
    assert _status(good, "Self-contained") == [D.OK] and "yes" in _named(good, "Self-contained").detail
    assert _status(good, "System CUDA Toolkit") == [D.OK] and "not used" in _named(good, "System CUDA Toolkit").detail
    keep = {"before": str(tk), "set_to": None, "why": "XCLEAN_KEEP_CUDA_PATH is set"}
    toolkit_facts = {"packaging": "pip", "cuda_path": str(tk), "loaded": {"nvrtc": [outside]}, "nvrtc": (13, 3),
                     "headers": [hdr_out]}
    bad = _named(D.check_toolkit(toolkit_facts, keep), "Self-contained")
    assert bad.status == D.WARN and "XCLEAN_KEEP_CUDA_PATH" in bad.fix
    assert outside in bad.detail and str(tk / "include") in bad.detail
    # a plain pip install (no compiler in the environment): the user's toolkit is the intended source
    plain = _named(D.check_toolkit(toolkit_facts, {"before": str(tk), "set_to": None,
                                                   "why": "this environment has no CUDA kernel compiler of its own"}),
                   "Self-contained")
    assert plain.status == D.OK and "no:" in plain.detail
    # the environment has its compiler but CuPy was imported before xclean
    (env / "Library" / "bin").mkdir(parents=True)
    (env / "Library" / "bin" / "nvrtc64_130_0.dll").write_bytes(b"")
    early = _named(D.check_toolkit(toolkit_facts, {"before": str(tk), "set_to": None,
                                                   "why": "CuPy was already imported before xclean"}),
                   "Self-contained")
    assert early.status == D.WARN and "before CuPy" in early.fix
    nothing = _named(D.check_toolkit({"packaging": "pip", "cuda_path": None, "loaded": {}, "nvrtc": None,
                                      "headers": []}, applied), "Self-contained")
    assert nothing.status == D.FAIL and "environment.yml" in nothing.fix
    unknown = _named(D.check_toolkit({"packaging": "pip", "loaded": {}, "nvrtc": (13, 3), "headers": [hdr_in]},
                                     applied), "Self-contained")
    assert unknown.status == D.OK and "not determined" in unknown.detail
    seen = D.check_toolkit({"packaging": None}, {"before": str(tk)})
    assert "visible" in seen[0].detail


# ---------------------------------------------------------------------------------------------------- OAK and disk
def test_oak(monkeypatch):
    real = D._import

    def with_dai(devs):
        dev = types.SimpleNamespace(getAllAvailableDevices=lambda: devs, getAllConnectedDevices=lambda: devs)
        dai = types.SimpleNamespace(__version__="3.3.0", Device=dev)
        monkeypatch.setattr(D, "_import", lambda name: dai if name == "depthai" else real(name))

    with_dai([])
    assert _status(D.check_oak(), "OAK camera") == [D.WARN]
    with_dai([types.SimpleNamespace(deviceId="1944301", state="XLinkDeviceState.X_LINK_UNBOOTED")])
    c = [x for x in D.check_oak() if x.name == "OAK camera"][0]
    assert c.status == D.OK and "1944301" in c.detail and "X_LINK_UNBOOTED" in c.detail

    def no_dai(name):
        if name == "depthai":
            raise ImportError("no depthai")
        return real(name)
    monkeypatch.setattr(D, "_import", no_dai)
    assert D.check_oak()[0].status == D.WARN


def test_disk(monkeypatch, tmp_path):
    assert D.check_disk(str(tmp_path / "not" / "yet"))[0].status in (D.OK, D.WARN, D.FAIL)
    monkeypatch.setattr(D.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=1 * 2 ** 30))
    assert D.check_disk(str(tmp_path))[0].status == D.FAIL


# -------------------------------------------------------------------------------------------- whole run, exit code
def _quiet_machine(monkeypatch, cupy=None, err="ImportError: CuPy is not installed"):
    monkeypatch.setattr(D, "_cupy", lambda: (cupy, None if cupy else err))
    monkeypatch.setattr(D, "_nvidia_smi", lambda q: None)
    monkeypatch.setattr(D, "check_oak", lambda: [D.Check(D.WARN, "OAK camera", "none found", "connect it")])
    monkeypatch.setattr(D, "check_disk", lambda out=None: [D.Check(D.OK, "Disk", "100 GB free")])


def test_no_gpu_machine_fails_cleanly(monkeypatch, capsys):
    _quiet_machine(monkeypatch)
    rc = D.main([])
    out = capsys.readouterr()
    assert rc == 1 and "Traceback" not in out.out + out.err
    checks = {c.name: c for c in D.run_checks()}
    for name in ("NVIDIA driver", "GPU", "CuPy"):
        assert checks[name].status == D.FAIL and checks[name].fix, name


def test_json_output(monkeypatch, capsys):
    _quiet_machine(monkeypatch)
    D.main(["--json"])
    rows = json.loads(capsys.readouterr().out)
    assert {"status", "name", "detail", "fix"} <= set(rows[0])


def test_exit_code_zero_without_failures(monkeypatch, capsys):
    monkeypatch.setattr(D, "run_checks", lambda **k: [D.Check(D.OK, "a", "fine"), D.Check(D.WARN, "b", "meh", "x")])
    assert D.main([]) == 0
    assert "1 OK, 1 WARN, 0 FAIL" in capsys.readouterr().out


def test_a_crashing_check_becomes_a_fail_line(monkeypatch, capsys):
    _quiet_machine(monkeypatch)

    def boom():
        raise ZeroDivisionError("inside a check")
    monkeypatch.setattr(D, "check_packages", boom)
    checks = D.run_checks()
    assert any(c.status == D.FAIL and "inside a check" in c.detail for c in checks)


def test_full_without_gpu(monkeypatch):
    _quiet_machine(monkeypatch)
    names = [c.name for c in D.run_checks(full=True)]
    assert "GPU front end" in names
