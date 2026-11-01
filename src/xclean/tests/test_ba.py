#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""BUNDLE ADJUSTMENT - agreement first, speed second.

    python -m xclean.tests.test_ba     (needs a CUDA GPU)

P0's rule, and it is not negotiable here: **a faster kernel that computes
something else is not a faster kernel.**  So the fused GPU path is compared
against the readable numpy reference on the same synthetic problem, from the
same starting point, BEFORE any timing appears - and the timing section refuses
to print if the agreement section failed.

The problem is synthetic on purpose: a known ground truth is the only way to
say "the solver moved the answer TOWARD the truth" rather than merely "the
residual went down", and a residual can go down while the geometry gets worse.

★ THE COMPARISON USES A TOLERANCE, NOT A HASH - deliberately.  Atomics are live
in the assembly (the pre-committed decision of plan 5.3, measured at +245 % for
the deterministic alternative), so the sum order varies between runs.  Where
atomics are live, **byte hashes are the wrong control**.  Using one here would
be the same category error as scoring a tensor claim with an ATE.
"""
from __future__ import annotations

import time

import numpy as np

FAILS = []


def check(name, ok, detail=""):
    FAILS.append((name, bool(ok), detail))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  - " + detail) if detail else ""), flush=True)


from xclean.backend.ba_synth import _Rig, make_problem  # noqa: E402,F401  (moved: packaging 2026-10-30)


def pose_err(a, b):
    return float(np.sqrt(np.mean(np.sum((a[:, :3, 3] - b[:, :3, 3]) ** 2, 1))))


def test_reference_converges(pb):
    from xclean.backend.ba import _solve_np
    p, q, st = _solve_np(pb["poses0"].copy(), pb["pts0"].copy(), pb["okf"],
                         pb["olm"], pb["ouv"], pb["rig"], 0, 5)
    e0 = pose_err(pb["poses0"], pb["poses_true"])
    e1 = pose_err(p, pb["poses_true"])
    check("the numpy reference moves the POSES toward the truth",
          e1 < e0 * 0.6 and st["rms_n"] < st["rms_0"],
          "pose rmse %.4f -> %.4f m, reprojection %.3f -> %.3f px"
          % (e0, e1, st["rms_0"], st["rms_n"]))
    return p, q, st


def test_gpu_agrees_with_the_reference(pb, ref):
    from xclean.backend.ba import _solve_gpu
    p_ref, q_ref, st_ref = ref
    p, q, st = _solve_gpu(pb["poses0"].copy(), pb["pts0"].copy(), pb["okf"],
                          pb["olm"], pb["ouv"], pb["rig"], 0, 5)
    dt = float(np.abs(p[:, :3, 3] - p_ref[:, :3, 3]).max())
    dp = float(np.abs(q - q_ref).max())
    drms = abs(st["rms_n"] - st_ref["rms_n"])
    check("the fused GPU path agrees with the reference (tolerance, not hash "
          "- atomics are live)",
          dt < 5e-3 and drms < 0.05,
          "max pose delta %.2e m, max point delta %.2e m, reprojection "
          "%.3f vs %.3f px" % (dt, dp, st["rms_n"], st_ref["rms_n"]))
    e0 = pose_err(pb["poses0"], pb["poses_true"])
    e1 = pose_err(p, pb["poses_true"])
    check("and it also moves the poses toward the truth", e1 < e0 * 0.6,
          "pose rmse %.4f -> %.4f m" % (e0, e1))
    return p, q, st


def test_speed(pb, agreed):
    """Only meaningful if agreement passed - and it refuses to run otherwise."""
    if not agreed:
        return check("local BA speed vs the Tier-1 bar", False,
                     "skipped: the GPU path does not agree with the reference")
    import cupy as cp
    from xclean.backend.ba import _solve_gpu, _solve_np
    from xclean.obs import budget as B

    def timeit(fn, n=20):
        fn()
        cp.cuda.runtime.deviceSynchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            cp.cuda.runtime.deviceSynchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        return ts[max(0, int(np.ceil(0.95 * len(ts))) - 1)]

    g = timeit(lambda: _solve_gpu(pb["poses0"].copy(), pb["pts0"].copy(),
                                  pb["okf"], pb["olm"], pb["ouv"], pb["rig"],
                                  0, 5))
    npt = timeit(lambda: _solve_np(pb["poses0"].copy(), pb["pts0"].copy(),
                                   pb["okf"], pb["olm"], pb["ouv"], pb["rig"],
                                   0, 5), n=3)
    t1 = B.BY_NAME["local_ba"].t1
    t2 = B.BY_NAME["local_ba"].t2
    check("local BA p95 is inside the Tier-1 bar", g <= t1,
          "%.3f ms vs T1 %.1f ms (T2 %.1f) - reference %.1f ms, %.0fx"
          % (g, t1, t2, npt, npt / max(g, 1e-9)))
    check("local BA p95 is inside the Tier-2 bar", g <= t2,
          "%.3f ms vs T2 %.1f ms" % (g, t2))
    return g


def test_window_is_bounded(pb):
    """No solve's cost may depend on map size (plan 5.1).  Measured by growing
    the LANDMARK count with the window fixed - the cost may grow with the
    window, which is bounded, but a map ten times larger must not appear."""
    import cupy as cp
    from xclean.backend.ba import _solve_gpu
    out = []
    for nl in (300, 600, 1200, 2400):
        p = make_problem(nk=15, nl=nl, seed=9)
        f = lambda: _solve_gpu(p["poses0"].copy(), p["pts0"].copy(), p["okf"],
                               p["olm"], p["ouv"], p["rig"], 0, 5)
        f()
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            f()
        cp.cuda.runtime.deviceSynchronize()
        out.append((nl, len(p["okf"]), (time.perf_counter() - t0) * 200))
    grow = out[-1][2] / out[0][2]
    check("solve cost grows sub-linearly with structure (launch-bound, as P0 "
          "measured)", grow < 4.0,
          "; ".join("%d lm/%d obs %.2f ms" % o for o in out)
          + "  -> 8x the structure costs %.2fx" % grow)


def main():
    print("=" * 96)
    print("  bounded local BA: agreement first, speed second")
    print("=" * 96)
    pb = make_problem()
    print("  problem: %d keyframes, %d landmarks, %d observations, "
          "reduced system %dx%d"
          % (len(pb["poses0"]), len(pb["pts0"]), len(pb["okf"]),
             6 * len(pb["poses0"]), 6 * len(pb["poses0"])))
    ref = test_reference_converges(pb)
    before = len(FAILS)
    test_gpu_agrees_with_the_reference(pb, ref)
    agreed = all(ok for _n, ok, _d in FAILS[before:])
    test_speed(pb, agreed)
    test_window_is_bounded(pb)
    bad = [n for n, ok, _ in FAILS if not ok]
    print("-" * 96)
    print("  %d/%d checks pass" % (len(FAILS) - len(bad), len(FAILS)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
