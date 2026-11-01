#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""THE DECLARED BUDGET - written before any estimator code, and the single
source of truth for every latency claim the engine makes.

Earlier design documents carry
the same table.  It lives here as DATA so those documents and the run
header can never disagree with what the gate actually enforces.

THE FOUR RULES THIS FILE ENCODES
  1. **The only latency bar is the BACKEND** (plan section 3.2).  The front end
     is measured and REPORTED, never gated: depth runs dense at full quality
     and is an accuracy input, not a budget item.  A front-end regression must
     be VISIBLE (so it is printed next to the backend) and must not be able to
     fail a test series (so its bar is None).
  2. **Every process must beat its own bar, not just the total.**  A total that
     passes because one stage is fast while another is 10x over is the
     aggregate hiding the failure - the same mistake `mean` makes against
     `p95`.
  3. **p95, never mean.**  A mean hides the keyframe that missed.
  4. **Zero means zero.**  Map fusion / render / densify, and any loop or PGO
     work in the frame path, are budgeted at 0.0 ms.  A nonzero measurement
     there is a SCOPE VIOLATION, not a slow path - the engine runs no map, it
     emits a 3DGS build kit (plan section 1b) - and asynchronous loop
     correction is a correctness property, not an optimisation.

THE ARITHMETIC THAT SAYS T2 HAS NO SLACK (plan section 3.3, restated here
because it is the reason P0's microbenchmark exists): at our own measured
solver efficiency one ORB-SLAM3-sized local BA - 1 solve over a ~15-KF window -
costs 662.6 ms/KF / 53 units of work = ~12.5 ms.  So bounded work ALONE lands
at ~12.5 ms and the remaining 1.3x must come from float32 + square-root form.
Both levers are required; neither is optional.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


# --------------------------------------------------------------------------
# process kinds
# --------------------------------------------------------------------------
BACKEND = "backend"      # gated, per keyframe
FRONTEND = "frontend"    # reported, per frame, never gated
KIT = "kit"              # the map build kit append, per frame, gated at 0.3 ms
EVENT = "event"          # gated, but per EVENT and NOT part of the per-keyframe
                         # total - see `loop_apply` below
ZERO = "zero"            # budgeted at exactly 0.0 - nonzero is a scope violation


@dataclass(frozen=True)
class Process:
    name: str
    kind: str
    t1: float | None          # Tier-1 bar, ms, p95.  None = reported, not gated
    t2: float | None          # Tier-2 bar, ms, p95
    per: str                  # "kf" or "frame"
    orbslam3: float | None    # published comparison, ms (Table VI) - NOT a gate
    note: str = ""

    @property
    def gated(self) -> bool:
        return self.t1 is not None

    def bar(self, tier: int) -> float | None:
        return self.t1 if tier == 1 else self.t2


# --------------------------------------------------------------------------
# the table.  ORB-SLAM3 column: arXiv:2007.11898 Table VI, EuRoC V2_02,
# stereo-inertial, mean+-sd on a CPU-only desktop.  It is a TARGET (board 2),
# never a reference: nothing is measured against it (plan section 1).
# --------------------------------------------------------------------------
PROCESSES: tuple[Process, ...] = (
    # ---- the backend: the only thing that is gated ----------------------
    Process("kf_insert",          BACKEND, 3.0,  1.5,  "kf",    8.53),
    Process("landmark_creation",  BACKEND, 5.0,  2.5,  "kf",   23.88),
    Process("landmark_culling",   BACKEND, 0.5,  0.3,  "kf",    0.24,
            "THE ONE LOOSE SLICE: 0.5 vs a published 0.24 ms.  Deliberate - "
            "culling exists from day one and does host transfer, so it is not "
            "the same process - but it means 'Tier 1 beats ORB-SLAM3 on the "
            "backend' is a claim about the TOTAL (9.8x) and not about every "
            "process.  Found by test_budget_gate, and this table says so."),
    Process("local_ba",           BACKEND, 10.0, 5.0,  "kf",  152.70,
            "bounded work + float32 square-root form; both levers required"),
    Process("kf_culling",         BACKEND, 1.5,  0.7,  "kf",   11.15),

    # ---- budgeted at exactly zero: a nonzero reading is a scope violation
    Process("pgo_frame_path",     ZERO,    0.0,  0.0,  "frame", None,
            "async and cancellable; contributing to the frame path is a FAILURE"),
    Process("loop_frame_path",    ZERO,    0.0,  0.0,  "frame", None,
            "loop CORRECTION in the frame path.  Never recorded, because the "
            "solve is on a worker thread - if this stage ever appears in a "
            "run header, the async property has been lost"),
    Process("loop_handoff",       FRONTEND, None, None, "frame", None,
            "the queue put that hands a keyframe to the loop worker.  "
            "REPORTED, not gated: it is not correction, and calling a 0.018 ms "
            "enqueue a scope violation would make the zero-budget rule mean "
            "nothing where it matters"),
    Process("map_fusion",         ZERO,    0.0,  0.0,  "frame", None,
            "no map is built while running"),
    Process("map_render",         ZERO,    0.0,  0.0,  "frame", None, "no map"),
    Process("map_densify",        ZERO,    0.0,  0.0,  "frame", None, "no map"),

    # ---- the map build kit: real work, but sequential writes off the path
    Process("map_kit_append",     KIT,     0.3,  0.3,  "frame", None,
            "so a later map cannot quietly become frame-path cost"),

    # ★ ADDED AT P4, AND THE ADDITION IS SAID OUT LOUD.  The plan's section 3.3
    # table has no row for APPLYING a loop correction - it budgets the PGO
    # SOLVE at zero in the frame path (which it is: the solve runs on a worker
    # thread over a snapshot) and stops there.  But the correction has to land
    # somewhere, and pretending otherwise would be exactly the kind of silent
    # omission the budget exists to prevent.
    #
    # It is declared PER EVENT and excluded from the per-keyframe total, for
    # the same reason `map_kit_append` is: a PGO completes tens of times in a
    # run, not once per keyframe, so charging it to a per-keyframe slice would
    # misstate both numbers.  The per-keyframe total stays at 20.0/10.0 and the
    # slices still sum to it.
    Process("global_ba",          EVENT,   None, None, "event", None,
            "the whole-trajectory solve.  Per EVENT and REPORTED, not gated: "
            "it runs at most a handful of times per row, off the frame path, "
            "and folding it into a per-keyframe slice would misstate both "
            "numbers.  Its cost IS published beside the per-keyframe budget"),
    Process("loop_apply",         EVENT,   5.0,  2.5,  "event", None,
            "the ONLINE PGO transaction, per completed solve.  Still gated, "
            "because the online path still exists behind --loop-online - but "
            "that path is REFUTED by measurement (V1_01 0.0712 loops-off vs "
            "0.4571/0.9725 applied live, 44 % of frames coasting) and is off "
            "by default"),
    Process("loop_finalise",      EVENT,   None, None, "event", None,
            "the DEFERRED correction: one pose graph over the whole run, "
            "applied once after the last frame.  REPORTED, not gated, for the "
            "same reason `global_ba` is - it happens once per row, off the "
            "frame path, and charging it to a per-keyframe slice would "
            "misstate both numbers"),

    # ---- the front end: REPORTED, NEVER GATED (plan section 3.2) --------
    Process("rectify",            FRONTEND, None, None, "frame",  1.60),
    Process("extract",            FRONTEND, None, None, "frame", 15.22),
    Process("depth",              FRONTEND, None, None, "frame",  3.38,
            "dense RAFT-Stereo at FULL quality: ~140 ms and deliberately so"),
    Process("match",              FRONTEND, None, None, "frame",  None),
    Process("track",              FRONTEND, None, None, "frame", 11.51),
    Process("reloc",              FRONTEND, None, None, "frame",  None,
            "relocalization-grade window recovery: runs ONLY on frames whose "
            "correspondence pool has been dead past a full window turnover - "
            "a healthy frame never pays it.  Reported, never gated, exactly "
            "as the rest of the front end"),
    Process("pose_out",           FRONTEND, None, None, "frame", 33.05,
            "not a gate: with dense RAFT depth, an earlier engine was not real-time end to end"),
)

BY_NAME = {p.name: p for p in PROCESSES}

# ---- totals ---------------------------------------------------------------
BACKEND_TOTAL_T1 = 20.0     # ms per keyframe, p95
BACKEND_TOTAL_T2 = 10.0
# a run exceeding the Tier-1 backend TOTAL on more than this fraction of
# keyframes fails the test series, exactly as an integrity violation does.
OVER_BUDGET_KF_FRACTION = 0.01

# published totals, for the header only.  Basalt is second-hand and carries its
# warning everywhere it is printed (board rule 2: a warning must become a
# verified figure before it is used in any claim).
ORBSLAM3_BACKEND_TOTAL = 196.5      # verified, Table VI
BASALT_BACKEND_TOTAL = 52.8         # UNVERIFIED, second-hand
BASALT_VERIFIED = False


def backend_processes():
    return [p for p in PROCESSES if p.kind == BACKEND]


def gated_processes():
    return [p for p in PROCESSES if p.gated]


def total_bar(tier: int) -> float:
    return BACKEND_TOTAL_T1 if tier == 1 else BACKEND_TOTAL_T2


def sum_of_slices(tier: int) -> float:
    """The per-process bars must add up to the total, or the table is
    inconsistent.  Checked by a test rather than trusted: a budget whose parts
    do not sum to its whole is a budget that can be passed by moving work
    between stages.
    """
    return sum(p.bar(tier) for p in backend_processes())


def as_dict():
    return {
        "processes": [asdict(p) for p in PROCESSES],
        "backend_total": {"t1": BACKEND_TOTAL_T1, "t2": BACKEND_TOTAL_T2},
        "over_budget_kf_fraction": OVER_BUDGET_KF_FRACTION,
        "published": {"orbslam3_backend_total": ORBSLAM3_BACKEND_TOTAL,
                      "basalt_backend_total": BASALT_BACKEND_TOTAL,
                      "basalt_verified": BASALT_VERIFIED},
    }


_POLICY = {BACKEND: "GATED /kf", ZERO: "GATED == 0", KIT: "GATED /frame",
           EVENT: "GATED /event", FRONTEND: "reported only"}


def table() -> str:
    """The budget, printed the way the run header prints it."""
    w = "  %-20s %9s %9s %9s  %s"
    out = [w % ("process", "ORB3", "T1", "T2", "policy"), "  " + "-" * 78]
    for p in PROCESSES:
        o = "%.2f" % p.orbslam3 if p.orbslam3 is not None else "-"
        t1 = "%.2f" % p.t1 if p.t1 is not None else "reported"
        t2 = "%.2f" % p.t2 if p.t2 is not None else "reported"
        out.append(w % (p.name, o, t1, t2, _POLICY[p.kind]))
    out.append("  " + "-" * 78)
    out.append(w % ("back end total /kf", "%.1f" % ORBSLAM3_BACKEND_TOTAL,
                    "%.1f" % BACKEND_TOTAL_T1, "%.1f" % BACKEND_TOTAL_T2,
                    "p95; >%.0f%% of KFs over = the run FAILS"
                    % (100 * OVER_BUDGET_KF_FRACTION)))
    return "\n".join(out)


if __name__ == "__main__":
    print(table())
    print("\n  sum of T1 slices: %.1f ms  (total bar %.1f)"
          % (sum_of_slices(1), BACKEND_TOTAL_T1))
    print("  sum of T2 slices: %.1f ms  (total bar %.1f)"
          % (sum_of_slices(2), BACKEND_TOTAL_T2))
