# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Local bundle adjustment on the synthetic problem with a known ground truth (xclean/backend/ba_synth.py).

The CPU reference solver must move the POSES toward the truth, not merely lower the residual (a residual can go
down while the geometry gets worse).  The fused GPU path is checked against this reference by
`python -m xclean.tests.test_ba` from a source checkout on a machine with a CUDA GPU (that script is not in the wheel).
"""
import numpy as np

from xclean.backend.ba_synth import make_problem


def _pose_err(a, b):
    return float(np.sqrt(np.mean(np.sum((a[:, :3, 3] - b[:, :3, 3]) ** 2, 1))))


def test_make_problem_is_deterministic():
    a, b = make_problem(nk=6, nl=60, seed=1), make_problem(nk=6, nl=60, seed=1)
    for k in ("poses0", "pts0", "okf", "olm", "ouv", "poses_true", "pts_true"):
        assert np.array_equal(a[k], b[k]), k
    assert len(a["okf"]) == len(a["olm"]) == len(a["ouv"]) > 0
    assert a["olm"].max() + 1 == len(a["pts0"])


def test_reference_solver_moves_poses_toward_the_truth():
    from xclean.backend.ba import _solve_np
    pb = make_problem()
    p, _q, st = _solve_np(pb["poses0"].copy(), pb["pts0"].copy(), pb["okf"], pb["olm"], pb["ouv"], pb["rig"], 0, 5)
    e0 = _pose_err(pb["poses0"], pb["poses_true"])
    e1 = _pose_err(p, pb["poses_true"])
    assert e1 < e0 * 0.6, (e0, e1)
    assert st["rms_n"] < st["rms_0"], st


def test_warmup_builds_its_problem_from_the_backend_module():
    """backend.ba.warmup() must not depend on the test package (packaging, 2026-10-30)."""
    import inspect
    from xclean.backend import ba
    src = inspect.getsource(ba.warmup)
    assert "xclean.backend.ba_synth import make_problem" in src and "xclean.tests" not in src
