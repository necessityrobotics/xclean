# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""The `xclean` command and the installed package: help for every subcommand, the shipped ORB vocabulary."""
import shutil
import subprocess
import sys

import numpy as np
import pytest

from xclean.cli import COMMANDS


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "xclean.cli", *args], capture_output=True, text=True)


def test_top_level_help():
    r = _cli("--help")
    assert r.returncode == 0
    for sub in COMMANDS:
        assert sub in r.stdout


@pytest.mark.parametrize("sub", sorted(COMMANDS))
def test_subcommand_help(sub):
    r = _cli(sub, "--help")
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("usage: xclean %s" % sub), r.stdout[:200]


def test_unknown_command_fails():
    assert _cli("nosuch").returncode == 2


def test_console_script_if_installed():
    exe = shutil.which("xclean")
    if exe is None:
        pytest.skip("the xclean console script is not on PATH (package not installed with pip)")
    r = subprocess.run([exe, "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "replay" in r.stdout


def test_vocabulary_ships_with_the_package():
    from xclean.loop.bow import DEFAULT_VOCAB, Vocabulary
    voc = Vocabulary.load(DEFAULT_VOCAB)
    assert voc.V == 10000 and voc.centers.shape == (11111, 32) and voc.idf.shape == (10000,)


def test_vocabulary_tree_descent_matches_a_reference():
    """The vectorised word lookup equals a per-descriptor reference descent (from the engine's own test)."""
    from xclean.loop.bow import DEFAULT_VOCAB, Vocabulary
    voc = Vocabulary.load(DEFAULT_VOCAB)

    def ref_word(d):
        node = 0
        while voc.child_n[node] > 0:
            f, n = voc.child_first[node], voc.child_n[node]
            best, bj = 1 << 30, -1
            for j in range(n):
                h = int(np.unpackbits(np.bitwise_xor(d, voc.centers[f + j])).sum())
                if h < best:
                    best, bj = h, f + j
            node = bj
        return voc.word_of[node]

    d = np.random.default_rng(1).integers(0, 256, (300, 32), dtype=np.uint8)
    assert np.array_equal(voc.words(d), np.array([ref_word(x) for x in d]))


def test_place_descriptor_contract():
    import cv2
    from xclean.loop.place import PlaceDescriptor
    pd = PlaceDescriptor(verbose=False)
    img = cv2.GaussianBlur((np.random.default_rng(0).random((480, 640)) * 255).astype(np.uint8), (0, 0), 1.5)
    v1, v2 = pd.describe(img), pd.describe(img)
    assert v1.dtype == np.float32 and v1.shape == (pd.dim,) and abs(np.linalg.norm(v1) - 1) < 1e-5
    assert np.array_equal(v1, v2)
