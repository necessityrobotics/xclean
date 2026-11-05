# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Every xclean GPU kernel source compiles with this environment's CUDA kernel compiler (NVRTC) for the GPUs xclean
supports, including NVIDIA Jetson Orin (sm_87) and Jetson Thor (sm_110, CUDA 13 only).  No GPU is needed: NVRTC only
compiles.  Runs in a fresh process that imports xclean first, as the xclean command does, so xclean's CUDA compiler
step (xclean/__init__.py) is what makes NVRTC loadable.

Skipped when CuPy or NVRTC cannot be loaded, unless XCLEAN_REQUIRE_NVRTC=1 (the Linux CI sets it), which turns that
into a failure."""
import json
import os
import subprocess
import sys

import pytest

SCRIPT = r'''
import json, os, sys
import xclean                                   # the CUDA compiler step runs before CuPy is imported
import cupy
from cupy.cuda import compiler
from xclean.backend import kernels as K
from xclean.front import gpu_front as G
from xclean.stages import align_cuda as A
from xclean.stereo import epipolar as E
from xclean.stereo import sgm32 as S
nv = tuple(compiler._get_nvrtc_version())
base = ["-I" + os.path.join(os.path.dirname(cupy.__file__), "_core", "include")]
base += list(compiler._get_extra_include_dir_opts())
cuda_path = cupy.cuda.get_cuda_path()
if cuda_path and os.path.isdir(os.path.join(cuda_path, "include")):
    base.append("-I" + os.path.join(cuda_path, "include"))
if nv[0] >= 12:
    base.append("--device-as-default-execution-space")
sources = {"backend": (K._SRC, ("-std=c++14", "-ftz=true")), "gpu_front": (G._SRC, tuple(G._OPTS)),
           "align": (A._SRC, ("-ftz=true",)), "epipolar": (E._KERNEL, ("-ftz=true",)),
           "sgm32": (S.KERNELS, ("-std=c++14", "-ftz=true"))}
archs = ["75", "86", "87", "89"] + (["110", "120"] if nv[0] >= 13 else ["90"])
out = {}
for name, (src, opts) in sources.items():
    for a in archs:
        prog = compiler._NVRTCProgram(src, name + ".cu", method="cubin")
        cubin, _ = prog.compile(tuple(opts) + tuple(base) + ("-arch=sm_" + a,))
        out["%s@sm_%s" % (name, a)] = len(cubin)
print("RESULT " + json.dumps({"nvrtc": nv, "note": xclean.CUDA_PATH_NOTE["why"], "compiled": out}))
'''


def test_all_kernels_compile_for_supported_gpus():
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=1200)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if r.returncode != 0 or not line:
        msg = (r.stderr or r.stdout).strip().splitlines()[-3:]
        text = " ".join(msg)
        not_loadable = any(k in text for k in ("No module named 'cupy'", "libnvrtc", "nvrtc64",
                                               "cannot open shared object file"))
        if not_loadable and os.environ.get("XCLEAN_REQUIRE_NVRTC") != "1":
            pytest.skip("CuPy / NVRTC not loadable here: %s" % " | ".join(msg))
        pytest.fail("kernel compile failed:\n%s" % "\n".join(msg))
    res = json.loads(line[0][len("RESULT "):])
    assert res["compiled"] and all(n > 0 for n in res["compiled"].values()), res
    print(res["nvrtc"], res["note"], len(res["compiled"]), "kernel builds")
