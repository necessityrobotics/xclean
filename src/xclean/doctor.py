# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""`xclean doctor`: check this machine before a run, in plain language.

    xclean doctor [--full] [--out DIR] [--size WxH] [--json]

Each check prints OK, WARN or FAIL, and a one-line fix for WARN / FAIL.  The exit code is 0 when nothing FAILs, else 1.
Checks: the Python version; numpy / scipy / numba / llvmlite / opencv / psutil against the tested versions; which
LAPACK SciPy uses (and its CPU kernel) and which BLAS NumPy uses; the text encoding GPU kernels are written in; the
NVIDIA driver and its CUDA version; the GPU (name, compute capability, free memory); CuPy (import, packaging, CUDA
path, the CUDA kernel compiler NVRTC: version and where it was loaded from, the CUDA headers it compiles with, a small
GPU kernel, and the folders its CUDA libraries were loaded from); whether a system CUDA Toolkit is visible and whether
this environment is self-contained; with --full, the GPU front end's start-up self-check; depthai and connected OAK
cameras (no stream is started); free disk space at the output folder.
Nothing is installed or changed.  Without a GPU the GPU checks FAIL with a fix; they never stop the other checks.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

OK, WARN, FAIL = "OK", "WARN", "FAIL"

# the tested set: the exact versions environment.yml installs (the published results used them), and the lower bounds
# (pyproject.toml)
TESTED = {"numpy": "2.3.5", "scipy": "1.16.3", "numba": "0.63.1", "llvmlite": "0.46.0", "opencv": "5.0.0",
          "psutil": "7.2.2", "cupy": "13.6.0", "depthai": "3.3.0"}
MINIMUM = {"numpy": "2.0", "scipy": "1.13", "numba": "0.60", "llvmlite": None, "opencv": "4.10", "psutil": "5.9",
           "cupy": "13.6", "depthai": "3.3"}
TESTED_PYTHON = (3, 12)
MIN_PYTHON = (3, 10)
TESTED_NVRTC = (13, 3)               # the CUDA kernel compiler of the published results (environment.yml: cuda-nvrtc)
MIN_DRIVER_CUDA = (13, 0)            # CuPy for CUDA 13 needs a driver for CUDA 13.0 or newer
MIN_CC = (7, 5)                      # CUDA 13 supports compute capability 7.5 (Turing) and newer
MIN_FREE_VRAM_GB = 2.0               # the engine's GPU memory budget, context included
MATCHED_OPENBLAS_CORE = "skylakex"   # the OpenBLAS CPU kernel the GPU front end's 3x3 solver matches bit for bit
FIX_ENV = "recreate the environment from environment.yml (conda env remove -n xclean, then conda env create -f " \
          "environment.yml)"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Check:
    def __init__(self, status, name, detail, fix=""):
        self.status, self.name, self.detail, self.fix = status, name, detail, fix

    def as_dict(self):
        return {"status": self.status, "name": self.name, "detail": self.detail, "fix": self.fix}


# ------------------------------------------------------------------------------------------- helpers (monkeypatchable)
def _import(name):
    return __import__(name, fromlist=["_"])


def _nvidia_smi(query):
    """nvidia-smi --query-gpu=<query> -> list of rows (lists of str), or None when nvidia-smi is unavailable."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        o = subprocess.run([exe, "--query-gpu=" + query, "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=15, creationflags=NO_WINDOW)
    except Exception:
        return None
    if o.returncode != 0:
        return None
    return [[x.strip() for x in ln.split(",")] for ln in o.stdout.strip().splitlines() if ln.strip()]


def _loaded_modules():
    """Paths of the files mapped into this process (DLLs / shared libraries)."""
    try:
        import psutil
        return sorted({m.path for m in psutil.Process().memory_maps() if getattr(m, "path", "")})
    except Exception:
        return []


def _conda_meta(name):
    """conda-meta/<name>-<version>-<build>.json file names (without .json) of this environment: conda-installed."""
    return [os.path.basename(p)[:-5] for p in sorted(glob.glob(os.path.join(sys.prefix, "conda-meta", name + "-*.json")))
            if os.path.basename(p)[len(name) + 1:len(name) + 2].isdigit()]


def _openblas_corename(path):
    """The CPU kernel an OpenBLAS library chose at start-up (e.g. 'SkylakeX', 'Haswell'), or None."""
    try:
        import ctypes
        lib = ctypes.CDLL(path)
    except Exception:
        return None
    for sym in ("scipy_openblas_get_corename", "scipy_openblas_get_corename64_", "openblas_get_corename",
                "openblas_get_corename64_"):
        try:
            fn = getattr(lib, sym)
        except AttributeError:
            continue
        try:
            fn.restype = ctypes.c_char_p
            v = fn()
            return v.decode(errors="replace").strip() if v else None
        except Exception:
            return None
    return None


def _vtuple(v):
    out = []
    for part in str(v).replace("-", ".").replace("+", ".").split("."):
        num = ""
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                break
        if num == "":
            break
        out.append(int(num))
    return tuple(out)


def _inside(path, root):
    try:
        return os.path.normcase(os.path.abspath(path)).startswith(os.path.normcase(os.path.abspath(root)) + os.sep)
    except Exception:
        return False


def version_check(label, have, tested, minimum, fix):
    """One package version against the tested version (compared to as many parts as it has) and the lower bound."""
    if have is None:
        return Check(FAIL, label, "not installed", fix)
    hv = _vtuple(have)
    if minimum and hv < _vtuple(minimum):
        return Check(FAIL, label, "%s is older than the minimum %s" % (have, minimum), fix)
    if tested:
        tv = _vtuple(tested)
        if hv[:len(tv)] != tv:
            return Check(WARN, label, "%s (tested with %s; other versions are untested)" % (have, tested), fix)
    return Check(OK, label, "%s%s" % (have, " (tested)" if tested else ""))


# ------------------------------------------------------------------------------------------------------------- checks
def check_python(vi=None):
    vi = tuple(vi or sys.version_info[:3])
    s = ".".join(str(x) for x in vi)
    if vi[:2] == TESTED_PYTHON:
        return [Check(OK, "Python", s)]
    if vi[:2] >= MIN_PYTHON:
        return [Check(WARN, "Python", "%s (tested with 3.12)" % s, FIX_ENV)]
    return [Check(FAIL, "Python", "%s is too old (3.10 or newer is needed; 3.12 is tested)" % s, FIX_ENV)]


def check_packages():
    out = []
    for label, mod in (("numpy", "numpy"), ("scipy", "scipy"), ("numba", "numba"), ("llvmlite", "llvmlite"),
                       ("opencv", "cv2"), ("psutil", "psutil")):
        try:
            v = getattr(_import(mod), "__version__", "unknown")
        except Exception:
            v = None
        out.append(version_check(label, v, TESTED[label], MINIMUM[label], FIX_ENV))
    return out


def blas_origin(pkg):
    """Which BLAS / LAPACK library `pkg` ('scipy': the LAPACK numba's solve / inverse call; 'numpy': numpy's BLAS)
    uses.  -> (vendor, source, evidence, path): vendor in 'openblas', 'mkl', 'blis', 'accelerate', 'netlib' or None;
    source 'pip' (the library bundled in the pip package, as environment.yml installs it), 'conda' or None; path = the
    loaded OpenBLAS library of a pip package (for its CPU kernel), else None."""
    try:
        _import("scipy.linalg.cython_lapack" if pkg == "scipy" else "numpy")      # loads the library
    except Exception:
        pass
    mods = _loaded_modules()
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(_import(pkg).__file__)))
    except Exception:
        root = None
    if root is not None:
        libs = [os.path.join(root, pkg + ".libs"), os.path.join(root, pkg, ".dylibs")]   # pip wheels (delvewheel ...)
        bundled = sorted(p for d in libs for p in glob.glob(os.path.join(d, "*openblas*")))
        loaded = [m for m in mods if any(_inside(m, d) for d in libs) and "openblas" in os.path.basename(m).lower()]
        if loaded or bundled:
            p = (loaded or bundled)[0]
            return "openblas", "pip", "%s %s" % ("loaded" if loaded else "bundled", os.path.basename(p)), \
                (loaded[0] if loaded else None)
    if _conda_meta(pkg):
        for name in _conda_meta("libblas"):
            for v in ("mkl", "openblas", "blis", "accelerate", "netlib"):
                if name.endswith("_" + v):
                    return v, "conda", "conda package %s" % name, None
    names = [os.path.basename(m).lower() for m in mods]
    mkl = [m for m in names if m.startswith(("mkl_", "libmkl"))]
    if mkl:
        return "mkl", None, "loaded library %s" % mkl[0], None
    ob = [m for m in names if "openblas" in m]
    if ob:
        return "openblas", None, "loaded library %s" % ob[0], None
    try:
        tp = _import("threadpoolctl")
        for d in tp.threadpool_info():
            if d.get("user_api") == "blas" and d.get("internal_api"):
                return str(d["internal_api"]).lower(), None, "threadpoolctl: %s" % d["internal_api"], None
    except Exception:
        pass
    return None, None, "not identified", None


def check_blas():
    """SciPy's LAPACK (the GPU front end's 3x3 solver reproduces OpenBLAS's arithmetic as SciPy's pip package ships it,
    with its AVX-512 kernel) and NumPy's BLAS (CPU results of the published runs)."""
    out = []
    vendor, source, ev, path = blas_origin("scipy")
    name = "LAPACK (SciPy)"
    if vendor == "openblas" and source == "pip":
        core = _openblas_corename(path) if path else None
        if core is None or core.lower() == MATCHED_OPENBLAS_CORE:
            out.append(Check(OK, name, "OpenBLAS from the SciPy package (%s%s): the build the published results used"
                             % (ev, ", CPU kernel %s" % core if core else "")))
        else:
            out.append(Check(WARN, name, "OpenBLAS from the SciPy package (%s), CPU kernel %s. The GPU front end's 3x3 "
                             "solver matches OpenBLAS bit for bit only with its kernel for CPUs with AVX-512 "
                             "(SkylakeX), so its start-up check reports a difference at rounding level on this CPU; "
                             "xclean keeps the GPU front end and prints a note" % (ev, core),
                             "none needed (this comes from the CPU, not from the installation)"))
    elif vendor == "mkl":
        out.append(Check(WARN, name, "Intel MKL (%s). The GPU front end's start-up check of its 3x3 solver will not "
                         "match this LAPACK bit for bit; xclean keeps the GPU front end and prints a note (results can "
                         "differ from the published ones in the last digits)" % ev,
                         FIX_ENV + "; its SciPy from pip brings the OpenBLAS LAPACK the GPU front end matches"))
    elif vendor == "openblas":
        out.append(Check(WARN, name, "OpenBLAS from %s (%s), not the SciPy package's own build that the published "
                         "results used: the GPU front end's 3x3 solver check will report a difference at rounding "
                         "level (and one conda-forge OpenBLAS build crashed in our tests)"
                         % ("a conda package" if source == "conda" else "another package", ev), FIX_ENV))
    else:
        out.append(Check(WARN, name, "%s (%s)" % (vendor or "unknown", ev), FIX_ENV))
    vendor, source, ev, _p = blas_origin("numpy")
    name = "BLAS (NumPy)"
    if vendor == "openblas" and source == "pip":
        out.append(Check(OK, name, "OpenBLAS from the NumPy package (%s): the build the published results used" % ev))
    else:
        out.append(Check(WARN, name, "%s%s (%s): CPU results can differ from the published ones in the last digits"
                         % (vendor or "unknown", " from a conda package" if source == "conda" else "", ev), FIX_ENV))
    return out


def _kernel_sources_not_ascii(pkg_dir=None):
    """Package files whose GPU kernel source strings (CUDA C passed to CuPy) contain non-ASCII characters."""
    import ast
    pkg_dir = pkg_dir or os.path.dirname(os.path.abspath(__file__))
    bad = []
    for p in sorted(glob.glob(os.path.join(pkg_dir, "**", "*.py"), recursive=True)):
        try:
            with open(p, encoding="utf-8") as f:
                text = f.read()
        except (OSError, ValueError):
            continue
        if text.isascii() or "__global__" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        if any(isinstance(n, ast.Constant) and isinstance(n.value, str) and "__global__" in n.value
               and not n.value.isascii() for n in ast.walk(tree)):
            bad.append(os.path.relpath(p, pkg_dir).replace(os.sep, "/"))
    return bad


def check_encoding(utf8_mode=None, encoding=None):
    """CuPy writes a GPU kernel's source to a file in the console's text encoding when it first compiles it."""
    import locale
    utf8_mode = sys.flags.utf8_mode if utf8_mode is None else utf8_mode
    enc = encoding if encoding is not None else (locale.getpreferredencoding(False) or "unknown")
    if utf8_mode or enc.lower().replace("-", "").replace("_", "") == "utf8":
        return [Check(OK, "Text encoding", "UTF-8")]
    bad = _kernel_sources_not_ascii()
    if not bad:
        return [Check(OK, "Text encoding", "%s (the GPU kernel sources are plain ASCII)" % enc)]
    return [Check(WARN, "Text encoding", "%s: the first build of the GPU kernels can stop with a UnicodeEncodeError "
                  "(CuPy writes kernel sources in this encoding, and those in %s contain non-ASCII characters)"
                  % (enc, ", ".join(bad)),
                  "set PYTHONUTF8=1 in this shell before running xclean (cmd: set PYTHONUTF8=1; PowerShell: "
                  "$env:PYTHONUTF8=1)")]


def _cupy():
    try:
        return _import("cupy"), None
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, str(e).splitlines()[0] if str(e) else "")


def check_driver(cp=None):
    rows = _nvidia_smi("driver_version,name")
    drv = rows[0][0] if rows and rows[0] else None
    cuda_v = None
    if cp is not None:
        try:
            v = int(cp.cuda.runtime.driverGetVersion())
            cuda_v = (v // 1000, (v % 1000) // 10)
        except Exception:
            cuda_v = None
    if drv is None and cuda_v is None:
        return [Check(FAIL, "NVIDIA driver", "no NVIDIA driver found (nvidia-smi and the CUDA driver API are missing)",
                      "install the NVIDIA driver for your GPU (one that supports CUDA 13, R580 or newer)")]
    detail = "driver %s" % (drv or "unknown")
    if cuda_v is not None:
        detail += ", CUDA driver API %d.%d" % cuda_v
        if cuda_v < MIN_DRIVER_CUDA:
            return [Check(FAIL, "NVIDIA driver", detail + " (CUDA 13.0 or newer is needed)",
                          "update the NVIDIA driver (R580 or newer supports CUDA 13)")]
        return [Check(OK, "NVIDIA driver", detail)]
    return [Check(WARN, "NVIDIA driver", detail + " (CUDA version not read: CuPy did not load)",
                  "fix CuPy first (see the CuPy line)")]


def check_gpu(cp=None, cp_err=None):
    if cp is None:
        return [Check(FAIL, "GPU", "not checked: CuPy did not import (%s)" % cp_err, FIX_ENV)]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        n = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:
        n, err = 0, "%s: %s" % (type(e).__name__, e)
    else:
        err = ""
    if n == 0:
        if cvd is not None and cvd.strip() in ("-1", ""):
            return [Check(FAIL, "GPU", "no GPU visible to this process: CUDA_VISIBLE_DEVICES=%r hides it" % cvd,
                          "unset CUDA_VISIBLE_DEVICES (it is set in this shell)")]
        return [Check(FAIL, "GPU", "no CUDA GPU found%s" % (" (%s)" % err if err else ""),
                      "xclean needs an NVIDIA GPU with a working driver (check nvidia-smi)")]
    out = []
    try:
        p = cp.cuda.runtime.getDeviceProperties(0)
        name = p["name"].decode() if isinstance(p["name"], bytes) else str(p["name"])
        cc = (int(p["major"]), int(p["minor"]))
        total = float(p["totalGlobalMem"]) / 2 ** 30
    except Exception as e:
        return [Check(FAIL, "GPU", "device 0 not readable (%s: %s)" % (type(e).__name__, e),
                      "check the driver with nvidia-smi")]
    free = None
    try:
        with cp.cuda.Device(0):
            free = float(cp.cuda.runtime.memGetInfo()[0]) / 2 ** 30
    except Exception:
        rows = _nvidia_smi("memory.free")
        if rows and rows[0] and rows[0][0].replace(".", "", 1).isdigit():
            free = float(rows[0][0]) / 1024.0
    detail = "%s, compute capability %d.%d, %.1f GB" % (name, cc[0], cc[1], total)
    if n > 1:
        detail += " (device 0 of %d is used)" % n
    if cc < MIN_CC:
        out.append(Check(FAIL, "GPU", detail + ": CUDA 13 needs compute capability 7.5 or newer",
                         "a Turing (RTX 20-series) or newer NVIDIA GPU is needed"))
    else:
        out.append(Check(OK, "GPU", detail))
    if free is None:
        out.append(Check(WARN, "GPU memory", "free memory not readable", "check with nvidia-smi"))
    elif free < MIN_FREE_VRAM_GB:
        out.append(Check(WARN, "GPU memory", "%.1f GB free (the engine needs about 2 GB)" % free,
                         "close other programs that use the GPU"))
    else:
        out.append(Check(OK, "GPU memory", "%.1f GB free" % free))
    return out


def cupy_packaging(cp):
    try:
        with open(os.path.join(os.path.dirname(cp.__file__), ".data", "_wheel.json"), encoding="utf-8") as f:
            return (json.load(f) or {}).get("packaging")
    except Exception:
        return None


def header_dirs(cp, cuda_path, nvrtc=None):
    """The folders CuPy compiles its own kernels against, in its order, that hold the CUDA headers (cuda_fp16.h):
    <CUDA path>/include, then the conda environment's or the nvidia-cuda-runtime package's include folder (the latter
    only when its version equals the compiler's).  -> list of (folder, origin)."""
    cands = []
    if cuda_path:
        cands.append((os.path.join(cuda_path, "include"), "CUDA path"))
    extra = None
    try:
        extra = [o[2:] for o in cp.cuda.compiler._get_extra_include_dir_opts()]     # CuPy's own rule (13.x)
    except Exception:
        extra = None
    if extra is None:
        extra = []
        if cupy_packaging(cp) == "conda":
            extra.append(os.path.join(sys.prefix, "Library", "include"))
        elif nvrtc:
            try:
                from importlib.metadata import distribution
                d = distribution("nvidia-cuda-runtime")
                if _vtuple(d.version)[:2] == tuple(nvrtc[:2]):
                    extra.append(str(d.locate_file("nvidia/cu%d/include" % nvrtc[0])))
            except Exception:
                pass
    for d in extra:
        cands.append((d, "nvidia-cuda-runtime package" if "nvidia" in d.replace("\\", "/").split("/") else
                      "conda environment"))
    out, seen = [], set()
    for d, origin in cands:
        k = os.path.normcase(os.path.abspath(d))
        if k not in seen and os.path.isfile(os.path.join(d, "cuda_fp16.h")):
            seen.add(k)
            out.append((d, origin))
    return out


def _cuda_libs():
    """{kind: [paths]} of the CUDA libraries mapped into this process (nvrtc, nvrtc-builtins, cudart, nvcuda, ...)."""
    loaded = {}
    for path in _loaded_modules():
        b = os.path.basename(path).lower()
        for key in ("nvrtc-builtins", "nvrtc", "cudart", "nvcuda", "cublas", "cusolver", "nvjitlink"):
            if b.startswith(key) and (b.endswith((".dll", ".so")) or ".so." in b):
                loaded.setdefault(key, []).append(path)
                break
    return loaded


def _place(path):
    """'this environment' / 'a CUDA Toolkit or another folder: <path>' for a loaded file or a folder."""
    return "this environment" if _inside(path, sys.prefix) else path


def check_cupy(cp=None, cp_err=None, gpu_ok=True):
    """-> (checks, facts) where facts feeds check_toolkit."""
    facts = {"packaging": None, "cuda_path": None, "loaded": {}, "nvrtc": None, "headers": []}
    if cp is None:
        return [Check(FAIL, "CuPy", "does not import (%s)" % cp_err, FIX_ENV)], facts
    out = [version_check("CuPy", getattr(cp, "__version__", None), TESTED["cupy"], MINIMUM["cupy"], FIX_ENV)]
    facts["packaging"] = cupy_packaging(cp)
    try:
        facts["cuda_path"] = cp.cuda.get_cuda_path()
    except Exception:
        facts["cuda_path"] = None
    out.append(Check(OK, "CuPy setup", "packaging %s; CUDA path %s"
                     % (facts["packaging"] or "unknown", facts["cuda_path"] or "none")))
    try:
        nv = tuple(int(x) for x in cp.cuda.nvrtc.getVersion())
        facts["nvrtc"] = nv
    except Exception as e:
        nv = None
        out.append(Check(FAIL, "CUDA compiler", "the CUDA kernel compiler (NVRTC) does not load (%s: %s)"
                         % (type(e).__name__, str(e).splitlines()[0] if str(e) else ""),
                         FIX_ENV + "; it brings the compiler"))
    loaded = _cuda_libs()
    if nv is not None:
        where = "loaded from %s" % _place(loaded["nvrtc"][0]) if loaded.get("nvrtc") else "location not read"
        if nv[:2] == TESTED_NVRTC:
            out.append(Check(OK, "CUDA compiler", "NVRTC %d.%d, %s (the version of the published results)"
                             % (nv[0], nv[1], where)))
        else:
            out.append(Check(WARN, "CUDA compiler", "NVRTC %d.%d, %s; the published results used %d.%d: the compiled GPU "
                             "code differs (results usually agree, not bit for bit)"
                             % (nv[0], nv[1], where, TESTED_NVRTC[0], TESTED_NVRTC[1]), FIX_ENV))
        try:
            facts["headers"] = header_dirs(cp, facts["cuda_path"], nv)
        except Exception:
            facts["headers"] = []
        if facts["headers"]:
            d, origin = facts["headers"][0]
            out.append(Check(OK, "CUDA headers", "from %s (%s)" % (_place(d), origin)))
        else:
            out.append(Check(WARN, "CUDA headers", "no folder with the CUDA headers (cuda_fp16.h) found: CuPy's "
                             "half-precision kernels will not compile", FIX_ENV + "; it brings the headers"))
    if not gpu_ok:
        out.append(Check(FAIL, "GPU kernel", "not run: no usable GPU (see the GPU line)", "fix the GPU line first"))
    else:
        try:
            s = float(cp.sum(cp.arange(10, dtype=cp.float32)))
            cp.cuda.runtime.deviceSynchronize()
            if s == 45.0:
                out.append(Check(OK, "GPU kernel", "compiled and ran a small kernel (sum of 0..9 = 45)"))
            else:
                out.append(Check(FAIL, "GPU kernel", "a small kernel returned %r instead of 45" % s,
                                 "check the driver with nvidia-smi; then " + FIX_ENV))
        except Exception as e:
            out.append(Check(FAIL, "GPU kernel", "compile or run failed (%s: %s)"
                             % (type(e).__name__, str(e).splitlines()[0] if str(e) else ""),
                             "check that the driver supports CUDA 13; then " + FIX_ENV))
        loaded = _cuda_libs()                             # the kernel loaded more (e.g. the compiler's builtins)
    facts["loaded"] = loaded
    if loaded:
        out.append(Check(OK, "CUDA libraries", "; ".join("%s: %s" % (k, ", ".join(v)) for k, v in sorted(loaded.items()))))
    return out, facts


def check_toolkit(facts, note=None, gpu_ok=True):
    """System CUDA Toolkit visibility, what xclean's CUDA step did (xclean/__init__.py), and whether the CUDA compiler
    and headers come from this environment (self-contained) or from elsewhere."""
    if note is None:
        try:
            note = getattr(_import("xclean"), "CUDA_PATH_NOTE", None) or {}
        except Exception:
            note = {}
    before = note.get("before") if note else os.environ.get("CUDA_PATH")
    nvcc = shutil.which("nvcc")
    sys_tk = []
    if before and not _inside(before, sys.prefix):
        sys_tk.append("CUDA_PATH=%s" % before)
    if nvcc and not _inside(nvcc, sys.prefix):
        sys_tk.append("nvcc at %s" % nvcc)
    env_nvrtc = os.path.isfile(os.path.join(sys.prefix, "Library", "bin", "nvrtc64_130_0.dll"))
    if note.get("set_to"):
        used = "not used: xclean points CUDA_PATH at this environment and puts its Library\\bin first on PATH, in " \
               "xclean's own processes only"
    elif sys_tk:
        used = "used by CuPy: %s" % (note.get("why") or "this environment has no CUDA compiler of its own")
    else:
        used = ""
    out = [Check(OK, "System CUDA Toolkit", ("visible: " + "; ".join(sys_tk) + " (%s)" % used) if sys_tk
                 else "none visible%s" % (" (not needed: this environment has its own CUDA compiler)" if env_nvrtc
                                          else ""))]
    if facts.get("packaging") is None:
        return out
    loaded = facts.get("loaded") or {}
    nvrtc_paths = loaded.get("nvrtc", []) + loaded.get("nvrtc-builtins", [])
    outside = [p for k, v in loaded.items() if k != "nvcuda" for p in v if not _inside(p, sys.prefix)]
    hdrs = facts.get("headers") or []
    hdr_out = [d for d, _o in hdrs[:1] if not _inside(d, sys.prefix)]
    if not nvrtc_paths and not hdrs and facts.get("nvrtc") is None:
        out.append(Check(FAIL, "Self-contained", "no CUDA kernel compiler (NVRTC) and no CUDA headers found",
                         FIX_ENV + " (it brings both), or install a CUDA 13 Toolkit"))
        return out
    if facts.get("nvrtc") is not None and not nvrtc_paths:
        out.append(Check(OK, "Self-contained", "not determined: the loaded libraries could not be listed"))
        return out
    if outside or hdr_out:
        parts = []
        if outside:
            parts.append("CUDA libraries from outside this environment: %s" % ", ".join(outside))
        if hdr_out:
            parts.append("CUDA headers from %s" % hdr_out[0])
        what = "; ".join(parts)
        if "XCLEAN_KEEP_CUDA_PATH" in (note.get("why") or ""):
            out.append(Check(WARN, "Self-contained", "no: " + what,
                             "unset XCLEAN_KEEP_CUDA_PATH (xclean then uses this environment's own CUDA compiler)"))
        elif env_nvrtc:
            out.append(Check(WARN, "Self-contained", "no: " + what + " (xclean's CUDA step: %s)"
                             % (note.get("why") or "not run"),
                             "import xclean before CuPy, or run through the xclean command"))
        else:
            out.append(Check(OK, "Self-contained", "no: CuPy (%s package) uses a CUDA Toolkit: %s"
                             % (facts.get("packaging"), what)))
        return out
    out.append(Check(OK, "Self-contained", "yes: the CUDA compiler%s come from this environment%s"
                     % (" and headers" if hdrs else "",
                        "" if gpu_ok else " (only the libraries loaded so far: no GPU kernel ran)")))
    return out


def check_gpu_front(cp, gpu_ok, size=(640, 400)):
    """--full: the GPU front end's start-up self-check for one image size (as the engine runs it)."""
    if cp is None or not gpu_ok:
        return [Check(FAIL, "GPU front end", "not checked: no usable GPU", "fix the GPU lines first")]
    W, H = size
    try:
        from xclean.front import gpu_front
    except Exception as e:
        return [Check(FAIL, "GPU front end", "does not import (%s: %s)" % (type(e).__name__, e), FIX_ENV)]
    try:
        cal = dict(gpu_front.GpuFront(W, H).calibrate())
    except Exception as e:
        return [Check(WARN, "GPU front end", "self-check failed at %dx%d (%s: %s): the engine will use the CPU front "
                      "end (higher latency)" % (W, H, type(e).__name__, str(e)[:200]),
                      "report it with the output of xclean doctor --full --json")]
    if cal.get("lk_solver_equal") is False:
        vendor, source = blas_origin("scipy")[:2]
        fix = FIX_ENV if not (vendor == "openblas" and source == "pip") else \
            "none needed (this comes from the CPU's OpenBLAS kernel, see the LAPACK line)"
        return [Check(WARN, "GPU front end", "self-check at %dx%d passes except the 3x3 solver (%s of %s systems "
                      "differ at rounding level): the engine keeps the GPU front end"
                      % (W, H, cal.get("lk_solver_mismatch"), cal.get("lk_solver_systems")), fix)]
    return [Check(OK, "GPU front end", "self-check passes at %dx%d: the engine will use the GPU front end" % (W, H))]


def check_oak():
    try:
        dai = _import("depthai")
    except Exception:
        return [Check(WARN, "OAK camera", "depthai is not installed (needed only for live OAK cameras)",
                      "pip install depthai==3.3.0")]
    out = [version_check("depthai", getattr(dai, "__version__", None), TESTED["depthai"], MINIMUM["depthai"],
                         "pip install depthai==3.3.0")]
    try:
        devs = list(dai.Device.getAllAvailableDevices())
        if not devs and hasattr(dai.Device, "getAllConnectedDevices"):
            devs = list(dai.Device.getAllConnectedDevices())
    except Exception as e:
        out.append(Check(WARN, "OAK camera", "device search failed (%s: %s)" % (type(e).__name__, e),
                         "reconnect the camera; on Windows check it in Device Manager"))
        return out
    if not devs:
        out.append(Check(WARN, "OAK camera", "none found (needed only for xclean live)",
                         "connect the camera to a USB 3 port (blue) with a data cable"))
        return out
    desc = []
    for d in devs:
        did = None
        for a in ("deviceId", "mxid"):
            did = did or getattr(d, a, None)
        if did is None and hasattr(d, "getDeviceId"):
            try:
                did = d.getDeviceId()
            except Exception:
                did = None
        if did is None and hasattr(d, "getMxId"):
            try:
                did = d.getMxId()
            except Exception:
                did = None
        st = str(getattr(d, "state", "")).split(".")[-1]
        desc.append("%s%s" % (did or getattr(d, "name", "?"), (" (%s)" % st) if st else ""))
    out.append(Check(OK, "OAK camera", "%d found: %s" % (len(desc), ", ".join(desc))))
    return out


def check_disk(out_dir=None):
    d = os.path.abspath(out_dir or os.getcwd())
    probe = d
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        free = shutil.disk_usage(probe).free / 2 ** 30
    except Exception as e:
        return [Check(WARN, "Disk", "free space not readable at %s (%s)" % (d, e), "")]
    if free < 2:
        return [Check(FAIL, "Disk", "%.1f GB free at %s" % (free, probe),
                      "free space or choose another --out folder (a run writes up to a few GB)")]
    if free < 20:
        return [Check(WARN, "Disk", "%.1f GB free at %s (camera recordings take about 1 GB per minute)" % (free, probe),
                      "free space for long recordings")]
    return [Check(OK, "Disk", "%.0f GB free at %s" % (free, probe))]


def run_checks(full=False, out_dir=None, size=(640, 400)):
    """-> list of Check.  Every check is isolated: an unexpected error becomes a FAIL line, never a traceback."""
    res = []

    def safe(name, fn, *a):
        try:
            r = fn(*a)
            return r
        except Exception as e:
            return [Check(FAIL, name, "the check itself failed (%s: %s)" % (type(e).__name__, e),
                          "report this with the output of xclean doctor --json")]

    res += safe("Python", check_python)
    res += safe("Packages", check_packages)
    res += safe("LAPACK (SciPy)", check_blas)
    res += safe("Text encoding", check_encoding)
    cp, cp_err = _cupy()
    res += safe("NVIDIA driver", check_driver, cp)
    gpu = safe("GPU", check_gpu, cp, cp_err)
    res += gpu
    gpu_ok = any(c.name == "GPU" and c.status == OK for c in gpu)
    try:
        cchk, facts = check_cupy(cp, cp_err, gpu_ok)
    except Exception as e:
        cchk, facts = [Check(FAIL, "CuPy", "the check itself failed (%s: %s)" % (type(e).__name__, e), FIX_ENV)], {}
    res += cchk
    res += safe("System CUDA Toolkit", check_toolkit, facts, None, gpu_ok)
    if full:
        res += safe("GPU front end", check_gpu_front, cp, gpu_ok, size)
    res += safe("OAK camera", check_oak)
    res += safe("Disk", check_disk, out_dir)
    return res


def format_report(checks):
    w = max([len(c.name) for c in checks] + [8])
    lines = []
    for c in checks:
        lines.append("  %-4s  %-*s  %s" % (c.status, w, c.name, c.detail))
        if c.status != OK and c.fix:
            lines.append("  %-4s  %-*s  fix: %s" % ("", w, "", c.fix))
    n = {s: sum(1 for c in checks if c.status == s) for s in (OK, WARN, FAIL)}
    lines.append("")
    lines.append("%d OK, %d WARN, %d FAIL" % (n[OK], n[WARN], n[FAIL]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog=None, description="Check this machine for xclean (nothing is installed or changed).")
    ap.add_argument("--full", action="store_true", help="also run the GPU front end's start-up self-check (a few seconds)")
    ap.add_argument("--out", default=None, help="the folder runs will write to (free-space check; default: here)")
    ap.add_argument("--size", default="640x400", help="image size WxH for --full (default 640x400, an OAK-D-Lite)")
    ap.add_argument("--json", action="store_true", help="print the checks as JSON")
    a = ap.parse_args(argv)
    try:
        W, H = (int(x) for x in a.size.lower().split("x"))
    except ValueError:
        ap.error("--size must look like 640x400")
    checks = run_checks(full=a.full, out_dir=a.out, size=(W, H))
    if a.json:
        print(json.dumps([c.as_dict() for c in checks], indent=1))
    else:
        try:
            ver = getattr(_import("xclean"), "__version__", "?")
        except Exception:
            ver = "?"
        print("xclean doctor (xclean %s, Python at %s)" % (ver, sys.executable))
        print(format_report(checks))
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
