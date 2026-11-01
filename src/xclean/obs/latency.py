#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""THE PER-STAGE LATENCY LEDGER - and the verdict that fails a test series.

    with led.stage("local_ba", kf=k):
        ...                          # the solve
    led.frame_done(i); led.kf_done(k)
    print(led.header())              # budget printed BESIDE the measurement
    v = led.verdict()                # PASS / FAIL, with the reason

WHY THIS FILE EXISTS BEFORE ANY ESTIMATOR CODE
    Every previous engine in this line built the estimator first and the
    measurement never caught up.  `an earlier engine` spends 194.6 ms/frame of backend
    IN THE FRAME PATH on MH_02 and nothing in its run header says so; you have
    to know to go looking.  Here the budget is printed next to the measured
    value in the run header, so a regression is visible in the log WITHOUT
    opening a tool (plan section 3.4), and a run that misses the budget fails
    the test series exactly as an integrity violation does.

THE MEASUREMENT DECISIONS, ALL OF THEM DELIBERATE
  * **p95 by NEAREST RANK, never mean and never interpolated.**  A mean hides
    the keyframe that missed.  Nearest rank (`ceil(0.95 n)`-th smallest) is an
    actually-observed sample, so a p95 can always be pointed at a real
    keyframe; a linear-interpolated quantile cannot.
  * **every sample carries its keyframe and frame index**, because the >1 %
    rule is a statement about keyframes, not about samples, and because "which
    keyframe missed" has to be answerable.
  * **`perf_counter_ns`**, integer nanoseconds, no float accumulation.
  * **GPU stages synchronise or they measure nothing.**  A CUDA launch returns
    immediately; timing it without a sync measures the launch, not the work.
    The sync callable is INJECTED so this module stays free of cupy/torch - it
    has to be cheap enough to leave on, and unit-testable on CPU in ms.
  * **the ledger measures its own overhead** at construction and prints it.  An
    instrument whose cost is unknown cannot certify a 0.3 ms budget.

THE ZERO-BUDGET FLOOR, AND WHY IT IS NOT A HAND-SET CONSTANT
    Map fusion / render / densify and any loop or PGO work in the frame path
    are budgeted at exactly 0.0 ms (plan section 1b, 3.3): a nonzero reading is
    a SCOPE VIOLATION, not a slow path.  But no timer measures exactly zero.
    So the floor is the ledger's OWN measured self-overhead - a Class-C
    quantity, a function of what the run reports, never a typed number.  The
    claim it supports is the honest one: *this stage is indistinguishable from
    nothing*.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict

from xclean.obs import budget as B

_NS = 1e-6            # ns -> ms


# ★ THE MACHINE-STATE WITNESS.
# Front-end GPU stages that NO backend change can reach.  If two runs are
# compared and these disagree, the machine was in a different performance
# state and the comparison is void - the difference is thermal, contention or
# clocks, not the change under test.
#
# THIS EXISTS BECAUSE IT ALREADY HAPPENED (2026-08-26).  Three pinned-span
# runs launched back-to-back after a 25-minute sweep measured `local_ba` p95
# at 15.2-15.6 ms against 9.9-10.0 for the controlled runs, which read as a
# 51 % regression caused by the window size.  It was not: `match` was
# 2.15-2.34x slower in the same runs and `rectify` 1.86-2.16x, and neither can
# be touched by the local BA span.  The whole pipeline was throttled.  A
# latency claim without a witness is a claim about the room temperature.
WITNESS_STAGES = ("match", "rectify")


def p95(xs) -> float:
    """Nearest-rank 95th percentile: the ceil(0.95 n)-th smallest sample.

    An actually-observed value, so it can always be pointed at a real
    keyframe.  numpy's default linear interpolation cannot.
    """
    return quantile(xs, 0.95)


def quantile(xs, q: float) -> float:
    n = len(xs)
    if n == 0:
        return float("nan")
    s = sorted(xs)
    k = max(1, math.ceil(q * n))
    return float(s[min(k, n) - 1])


@dataclass
class StageStat:
    name: str
    kind: str
    n: int
    mean: float
    p50: float
    p95: float
    max: float
    total: float
    bar: float | None
    gated: bool
    over: bool
    worst_kf: int | None = None
    orbslam3: float | None = None


@dataclass
class Violation:
    rule: str
    process: str
    measured: float
    bar: float
    detail: str
    units: str = "ms"          # "ms" for a p95 bar, "%" for the >1 % rule

    def line(self):
        stat = "p95" if self.units == "ms" else "   "
        return ("  over budget: %-20s %s %8.3f %-2s >  budget %6.3f %-2s  (%s) %s"
                % (self.process, stat, self.measured, self.units, self.bar,
                   self.units, self.rule, self.detail))


@dataclass
class Verdict:
    ok: bool
    tier: int
    violations: list = field(default_factory=list)
    backend_p95: float = float("nan")
    backend_bar: float = float("nan")
    over_kf: int = 0
    n_kf: int = 0
    over_fraction: float = 0.0
    witness: dict = field(default_factory=dict)

    def witness_line(self):
        if not self.witness:
            return ""
        return ("speed check  " + "  ".join(
            "%s p95 %.3f ms" % (k, v) for k, v in sorted(self.witness.items()))
            + "   (front-end stages that no back-end change affects: when "
              "two runs differ here, the machine ran at a different speed, so "
              "their times cannot be compared)")

    def line(self):
        return ("latency budget %s (budget level %d): back end p95 %.3f ms "
                "per keyframe (budget %.1f ms), %d of %d keyframes over "
                "(%.2f %%, at most %.2f %% allowed)"
                % ("met" if self.ok else "NOT met", self.tier,
                   self.backend_p95, self.backend_bar, self.over_kf,
                   self.n_kf, 100 * self.over_fraction,
                   100 * B.OVER_BUDGET_KF_FRACTION))


class _Scope:
    __slots__ = ("led", "name", "kf", "frame", "t0")

    def __init__(self, led, name, kf, frame):
        self.led, self.name, self.kf, self.frame = led, name, kf, frame

    def __enter__(self):
        if self.led.sync is not None and self.name in self.led.gpu_stages:
            self.led.sync()
        self.t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        if self.led.sync is not None and self.name in self.led.gpu_stages:
            self.led.sync()
        self.led._record(self.name, time.perf_counter_ns() - self.t0,
                         self.kf, self.frame)
        return False


class Ledger:
    """Append-only per-stage timings with a budget verdict attached.

    `gpu_stages` names the stages that must synchronise before and after; the
    sync callable itself is injected (`cupy.cuda.runtime.deviceSynchronize`)
    so this module never imports a GPU library.
    """

    def __init__(self, run="", tier=1, sync=None, gpu_stages=(),
                 calibrate=2000):
        self.run = run
        self.tier = int(tier)
        self.sync = sync
        self.gpu_stages = set(gpu_stages)
        self._ns: dict[str, list[int]] = {}
        self._kf: dict[str, list[int]] = {}
        self._frame: dict[str, list[int]] = {}
        self.frames = 0
        self.keyframes = 0
        self.t_start = time.perf_counter_ns()
        self.overhead_ms = self._calibrate(calibrate)

    # ---- calibration ---------------------------------------------------
    def _calibrate(self, n) -> float:
        """Measure the instrument.  p95 of n empty open/close pairs, in ms."""
        if n <= 0:
            return 0.0
        xs = []
        for _ in range(n):
            a = time.perf_counter_ns()
            b = time.perf_counter_ns()
            xs.append(b - a)
        return p95(xs) * _NS

    # ---- recording -----------------------------------------------------
    def stage(self, name, kf=None, frame=None):
        return _Scope(self, name, -1 if kf is None else int(kf),
                      -1 if frame is None else int(frame))

    def _record(self, name, ns, kf, frame):
        self._ns.setdefault(name, []).append(ns)
        self._kf.setdefault(name, []).append(kf)
        self._frame.setdefault(name, []).append(frame)

    def add(self, name, ms, kf=None, frame=None):
        """Record a duration measured elsewhere (e.g. a CUDA event pair)."""
        self._record(name, int(round(ms / _NS)),
                     -1 if kf is None else int(kf),
                     -1 if frame is None else int(frame))

    def frame_done(self, i=None):
        self.frames += 1

    def kf_done(self, k=None):
        self.keyframes += 1

    # ---- reading -------------------------------------------------------
    def samples_ms(self, name):
        return [v * _NS for v in self._ns.get(name, ())]

    def names(self):
        """Every stage seen, budget order first, then anything unexpected."""
        known = [p.name for p in B.PROCESSES if p.name in self._ns]
        extra = sorted(n for n in self._ns if n not in B.BY_NAME)
        return known + extra

    def stat(self, name) -> StageStat:
        xs = self.samples_ms(name)
        p = B.BY_NAME.get(name)
        kind = p.kind if p else "unregistered"
        bar = p.bar(self.tier) if p and p.gated else None
        if p is not None and p.kind == B.ZERO:
            # see the module docstring: the floor is the instrument, not a
            # typed constant.  "indistinguishable from nothing" is the claim.
            bar = self.overhead_ms
        v = p95(xs)
        worst = None
        if xs:
            j = max(range(len(xs)), key=lambda i: xs[i])
            worst = self._kf[name][j]
        return StageStat(name=name, kind=kind, n=len(xs),
                         mean=(sum(xs) / len(xs)) if xs else float("nan"),
                         p50=quantile(xs, 0.5), p95=v,
                         max=max(xs) if xs else float("nan"),
                         total=sum(xs),
                         bar=bar, gated=bool(p and p.gated),
                         over=bool(bar is not None and v > bar),
                         worst_kf=worst if worst is not None and worst >= 0 else None,
                         orbslam3=p.orbslam3 if p else None)

    def backend_per_kf(self) -> dict[int, float]:
        """Sum the GATED BACKEND stages within each keyframe.

        Samples with no keyframe id (-1) are counted into a single bucket so
        they cannot silently vanish from the total - an unattributed backend
        cost is still a backend cost.
        """
        out: dict[int, float] = {}
        for p in B.backend_processes():
            xs = self._ns.get(p.name)
            if not xs:
                continue
            ks = self._kf[p.name]
            for ns, k in zip(xs, ks):
                out[k] = out.get(k, 0.0) + ns * _NS
        return out

    # ---- the verdict ---------------------------------------------------
    def verdict(self, tier=None) -> Verdict:
        tier = self.tier if tier is None else int(tier)
        viol = []

        # rule 2 of the budget: EVERY process must beat its own bar.
        for name in self.names():
            st = self.stat(name)
            if st.bar is None or st.n == 0:
                continue
            if st.p95 > st.bar:
                p = B.BY_NAME.get(name)
                if p is not None and p.kind == B.ZERO:
                    viol.append(Violation(
                        "scope violation: budgeted at zero", name, st.p95,
                        st.bar,
                        "distinguishable from the empty timer (%.4f ms); "
                        "no map is built while running and loop work is async"
                        % self.overhead_ms))
                else:
                    viol.append(Violation(
                        "per-process latency limit (level %d)" % tier, name, st.p95,
                        st.bar,
                        "worst at kf %s" % st.worst_kf))

        # the total, and the >1 % of keyframes rule.
        per_kf = self.backend_per_kf()
        tot_bar = B.total_bar(tier)
        vals = list(per_kf.values())
        b95 = p95(vals) if vals else float("nan")
        over = sum(1 for v in vals if v > tot_bar)
        n_kf = len(vals)
        frac = (over / n_kf) if n_kf else 0.0
        if n_kf and frac > B.OVER_BUDGET_KF_FRACTION:
            viol.append(Violation(
                "back end total over its limit on too many keyframes",
                "back end total", 100 * frac, 100 * B.OVER_BUDGET_KF_FRACTION,
                "%d/%d keyframes over the %.1f ms total; p95 %.3f ms"
                % (over, n_kf, tot_bar, b95), units="%"))
        if n_kf and b95 > tot_bar and not any(
                v.process == "back end total" for v in viol):
            viol.append(Violation("back end total p95 (level %d)" % tier,
                                  "back end total", b95, tot_bar,
                                  "%d keyframes" % n_kf))

        return Verdict(ok=not viol, tier=tier, violations=viol,
                       backend_p95=b95, backend_bar=tot_bar, over_kf=over,
                       n_kf=n_kf, over_fraction=frac,
                       witness={k: self.stat(k).p95 for k in WITNESS_STAGES
                                if k in self.names()})

    # ---- reporting -----------------------------------------------------
    def header(self, tier=None) -> str:
        """The run header: budget printed BESIDE the measurement (sec 3.4)."""
        tier = self.tier if tier is None else int(tier)
        v = self.verdict(tier)
        w = "  %-20s %6s %9s %9s %9s %9s %9s  %s"
        out = []
        out.append("=" * 104)
        out.append("  time per stage (ms)   run %s   frames %d   %swall %.1f s"
                   % (self.run or "-", self.frames,
                      ("keyframes %d   " % self.keyframes) if self.keyframes
                      else "",
                      (time.perf_counter_ns() - self.t_start) * _NS / 1e3))
        out.append("  timer overhead p95 = %.5f ms  (a stage budgeted at zero "
                   "must not exceed it)" % self.overhead_ms)
        out.append("=" * 104)
        out.append(w % ("stage", "n", "mean", "p50", "P95", "max", "BUDGET",
                        "result"))
        out.append("  " + "-" * 100)
        for name in self.names():
            st = self.stat(name)
            if st.n == 0:
                continue
            bar = "-" if st.bar is None else "%.2f" % st.bar
            if st.bar is None:
                verd = "reported (no budget)"
            elif st.over:
                verd = "over by %.2f ms" % (st.p95 - st.bar)
                if st.worst_kf is not None:
                    verd += "  worst keyframe %d" % st.worst_kf
            else:
                verd = "ok  %.0f%% of budget" % (100 * st.p95 / st.bar) \
                    if st.bar > 0 else "ok"
            out.append(w % (name, st.n, "%.3f" % st.mean, "%.3f" % st.p50,
                            "%.3f" % st.p95, "%.3f" % st.max, bar, verd))
        out.append("  " + "-" * 100)
        out.append(w % ("back end total /kf", v.n_kf, "-", "-",
                        "%.3f" % v.backend_p95, "-",
                        "%.2f" % v.backend_bar,
                        "%d/%d keyframes over (%.2f %%, limit %.2f %%)"
                        % (v.over_kf, v.n_kf, 100 * v.over_fraction,
                           100 * B.OVER_BUDGET_KF_FRACTION)))
        out.append("  " + "-" * 100)
        out.append("  " + v.line())
        wl = v.witness_line()
        if wl:
            out.append("  " + wl)
        for x in v.violations:
            out.append(x.line())
        out.append("=" * 104)
        return "\n".join(out)

    def as_dict(self, tier=None):
        tier = self.tier if tier is None else int(tier)
        v = self.verdict(tier)
        return {
            "run": self.run, "tier": tier,
            "frames": self.frames, "keyframes": self.keyframes,
            "overhead_ms": self.overhead_ms,
            "stages": {n: asdict(self.stat(n)) for n in self.names()},
            "backend_per_kf": {str(k): x for k, x in
                               sorted(self.backend_per_kf().items())},
            # RAW SAMPLES ARE SAVED.  A ledger that stores only summaries can
            # never be re-judged at another tier, and a latency distribution is
            # exactly the kind of thing a later phase wants a floor on.
            "samples_ms": {n: [v * _NS for v in self._ns[n]]
                           for n in self.names()},
            "sample_kf": {n: list(self._kf[n]) for n in self.names()},
            "verdict": {"ok": v.ok, "tier": v.tier,
                        "backend_p95": v.backend_p95,
                        "backend_bar": v.backend_bar,
                        "over_kf": v.over_kf, "n_kf": v.n_kf,
                        "over_fraction": v.over_fraction,
                        "violations": [asdict(x) for x in v.violations]},
            "budget": B.as_dict(),
        }

    def save(self, path, tier=None):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(tier), f, indent=1, sort_keys=True)
        os.replace(tmp, path)     # atomic: a power cut never leaves half a file
        return path


def verdict_from_file(path, tier=None) -> Verdict:
    """Judge a saved ledger without re-running anything.

    This is what the test series calls: the gate reads the artifact, so a run
    scored today and a run scored in six months are judged by the same code
    against the same table.
    """
    return ledger_from_file(path, tier).verdict()


def ledger_from_file(path, tier=None) -> Ledger:
    """Rebuild a ledger from a saved artifact, at any tier."""
    d = json.load(open(path, encoding="utf-8"))
    t = int(tier if tier is not None else d.get("tier", 1))
    led = Ledger(run=d.get("run", ""), tier=t, calibrate=0)
    led.overhead_ms = float(d.get("overhead_ms", 0.0))
    led.frames = int(d.get("frames", 0))
    led.keyframes = int(d.get("keyframes", 0))
    kfs = d.get("sample_kf", {})
    for name, xs in d.get("samples_ms", {}).items():
        ks = kfs.get(name, [-1] * len(xs))
        for ms, k in zip(xs, ks):
            led._record(name, int(round(ms / _NS)), k, -1)
    return led
