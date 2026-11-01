# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""The CUDA sources compiled at run time must be ASCII: CuPy writes them to its kernel cache in the console's encoding,
and a non-ASCII character there crashed the first GPU run on a cp1252 console."""
import importlib

import pytest

SOURCES = [("xclean.backend.kernels", "_SRC"), ("xclean.stereo.sgm32", "KERNELS"), ("xclean.stereo.epipolar", "_KERNEL"),
           ("xclean.stages.align_cuda", "_SRC"), ("xclean.front.gpu_front", "_SRC")]


@pytest.mark.parametrize("mod,name", SOURCES)
def test_cuda_source_is_ascii(mod, name):
    src = getattr(importlib.import_module(mod), name)
    bad = sorted({c for c in src if ord(c) > 127})
    assert not bad, "%s.%s has non-ASCII characters %r" % (mod, name, bad)
