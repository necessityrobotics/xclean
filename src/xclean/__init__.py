# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""xclean - real-time stereo visual-inertial SLAM (Python, CuPy, Numba).

Designed backwards from a latency budget and judged against ground truth. Entry point: the `xclean` command
(xclean/cli.py); see README.md and docs/SEQUENCE_FORMAT.md.

CUDA kernel compiler (Windows, process-local, before CuPy is imported by anything in xclean): when this environment
carries its own CUDA kernel compiler (<env>/Library/bin/nvrtc64_130_0.dll, as environment.yml installs it),
CUDA_PATH is set to <env>/Library and <env>/Library/bin is put first on PATH, for this process and its children.
CuPy then uses the environment's compiler and its matching CUDA headers, even when a different CUDA Toolkit is
installed on the machine or the environment was not activated.  Nothing is changed when XCLEAN_KEEP_CUDA_PATH is set
(to anything but 0), when CuPy was imported before xclean, or when the environment has no compiler of its own (a
plain pip install: the installed CUDA Toolkit stays in charge).  CuPy itself is not imported here.
"""

__version__ = "0.1.0"

CUDA_PATH_NOTE = None          # what the CUDA compiler step did in this process (read by `xclean doctor`)
ENV_NVRTC_DLL = "nvrtc64_130_0.dll"      # the CUDA 13 kernel compiler that environment.yml installs (cuda-nvrtc)


def _use_env_cuda(environ=None, prefix=None, platform=None, modules=None):
    """-> a dict describing the decision (also stored in CUDA_PATH_NOTE): before (CUDA_PATH as found), set_to (the
    CUDA_PATH now in effect, or None), path_first (the folder put first on PATH, or None), why.  Never raises.
    The arguments exist for the tests; by default the real os.environ, sys.prefix, sys.platform and sys.modules."""
    import os
    import sys
    environ = os.environ if environ is None else environ
    prefix = sys.prefix if prefix is None else prefix
    platform = sys.platform if platform is None else platform
    modules = sys.modules if modules is None else modules
    note = {"before": environ.get("CUDA_PATH"), "set_to": None, "path_first": None, "why": ""}
    try:
        if not platform.startswith("win"):
            note["why"] = "not Windows (the environment layout this step relies on is Windows-only)"
            return note
        if (environ.get("XCLEAN_KEEP_CUDA_PATH") or "").strip() not in ("", "0"):
            note["why"] = "XCLEAN_KEEP_CUDA_PATH is set"
            return note
        if "cupy" in modules:
            note["why"] = "CuPy was already imported before xclean"
            return note
        lib = os.path.join(prefix, "Library")
        bin_dir = os.path.join(lib, "bin")
        if not os.path.isfile(os.path.join(bin_dir, ENV_NVRTC_DLL)):
            note["why"] = "this environment has no CUDA kernel compiler of its own (Library/bin/%s)" % ENV_NVRTC_DLL
            return note

        def same(a, b):
            return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))

        if not (note["before"] and same(note["before"], lib)):
            environ["CUDA_PATH"] = lib
        path = environ.get("PATH") or ""
        first = path.split(os.pathsep, 1)[0].strip().strip('"')
        if not (first and same(first, bin_dir)):
            environ["PATH"] = bin_dir + (os.pathsep + path if path else "")
        note["set_to"] = lib
        note["path_first"] = bin_dir
        note["why"] = "the environment's own CUDA kernel compiler and headers"
    except Exception as e:                                # never stop an import over this step
        note["why"] = "skipped (%s: %s)" % (type(e).__name__, e)
    return note


CUDA_PATH_NOTE = _use_env_cuda()
