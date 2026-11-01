#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
# Vendored VERBATIM from our earlier research harness/obs/decision.py (our own earlier research harness, same company); the
# frozen engine re-exported that module, this file IS it now.
# Apart from this comment block nothing differs from that file.
"""`Gate` and `Decision` — contract (c) of the plan, and the reason log-replay
works by construction instead of by luck.

THE RULE
    A rejection is never a bare `False`.  Every gate returns a `Decision`
    carrying: the gate's name, accept/reject, a REASON, the dial values the
    decision was made on, and the margin by which it was made.

WHY, WITH RECEIPTS
    * A structure-BA path once silently re-armed a cooldown and DISABLED LOOP
      CLOSURE FOR AN ENTIRE RUN.  Nothing in the log said so, because "off" was
      silence.  Here, a disabled gate is a logged state (`Decision.disabled`),
      never an absence.
    * RULE 13 (log-first) says replay must kill a hypothesis before any GPU
      time is spent.  Replay only works if the log already contains the values
      the decision was made on.  That is not a logging convention you remember
      to follow; it is what `Decision.dials` is FOR.
    * Section 7b bans hard accept/reject on a continuous quantity: where a gate
      scores a continuous margin, the output is a WEIGHT, not a boolean.  mag1's
      13.5x spread came from a 0.4% input perturbation flipping a binary
      outcome.  `Decision.weight` is that repair, and `Gate.soft()` builds it.

WHAT IS DELIBERATELY NOT HERE
    Any gate logic.  This is the contract only; the gates themselves arrive
    with the subsystems they belong to.  `bench/integrity.py` is its first
    consumer, which is deliberate — the harness gets to prove the contract is
    usable before the estimator depends on it.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Decision:
    gate: str
    accepted: bool
    reason: str
    dials: dict = field(default_factory=dict)
    margin: float | None = None          # signed distance to the threshold
    weight: float = 1.0                  # 0..1 — soft gates emit this, not a bool
    threshold: float | None = None
    disabled: bool = False               # a disabled gate is a STATE, not silence

    def __bool__(self):
        return self.accepted

    def line(self) -> str:
        state = ("DISABLED" if self.disabled
                 else ("accept" if self.accepted else "REJECT"))
        bits = [
            "[GATE %s] %s" % (self.gate, state),
            self.reason,
        ]
        if self.threshold is not None:
            bits.append("thr=%.6g" % self.threshold)
        if self.margin is not None:
            bits.append("margin=%+.6g" % self.margin)
        if self.weight != 1.0:
            bits.append("w=%.4f" % self.weight)
        if self.dials:
            bits.append(json.dumps(self.dials, sort_keys=True, default=float))
        return "  ".join(bits)

    def to_dict(self):
        return asdict(self)


class Gate:
    """A named gate with a printed threshold, a dial and a logged reason.

    Subclasses implement `evaluate(**dials) -> Decision`.  The base class owns
    the two things every gate must have and none has reliably had: a per-run
    counter of what it rejected and why, and the fact that turning it off is a
    recorded decision.
    """

    def __init__(self, name: str, threshold: float | None = None,
                 enabled: bool = True):
        self.name = name
        self.threshold = threshold
        self.enabled = enabled
        self.counts = {"accept": 0, "reject": 0, "disabled": 0}
        self.reasons = {}
        # DIAL DISTRIBUTIONS (opt-in, XONE_DIAL_DIST=1).  A reason COUNT says how often a gate
        # rejected; it cannot say BY HOW MUCH, and those demand opposite fixes: rejects piled up
        # just under the threshold are a threshold problem, rejects far below it are an upstream
        # supply problem.  Rosario is the live case - 1,845 "too few shared landmarks" tells us
        # nothing until we see whether the shared count sits at 24 or at 3.
        # Off by default so the engine path is unchanged; reservoir-sampled so memory is bounded.
        self._dist_on = os.environ.get("XONE_DIAL_DIST", "") not in ("", "0")
        self._dist = {}          # key -> [n_seen, [samples]]
        self._dist_cap = 4000

    # ---- helpers subclasses use -------------------------------------------
    def accept(self, reason, dials=None, margin=None, weight=1.0):
        return self._record(Decision(self.name, True, reason, dials or {},
                                     margin, weight, self.threshold))

    def reject(self, reason, dials=None, margin=None):
        return self._record(Decision(self.name, False, reason, dials or {},
                                     margin, 0.0, self.threshold))

    def soft(self, value, reason, dials=None, width=None):
        """Continuous margin -> a WEIGHT, not a cliff (section 7b).

        `value` is the signed margin past the threshold; `width` is the scale
        over which the weight goes from 0 to 1.  A marginal candidate
        contributes weakly instead of contributing fully or not at all, which
        removes the cliff a 0.4% perturbation can flip.
        """
        if width is None or width <= 0:
            return (self.accept(reason, dials, margin=value) if value >= 0
                    else self.reject(reason, dials, margin=value))
        w = 1.0 / (1.0 + math.exp(-value / width))
        return self._record(Decision(self.name, w > 0.5, reason, dials or {},
                                     value, w, self.threshold))

    def off(self, reason="disabled by configuration"):
        return self._record(Decision(self.name, False, reason, {}, None, 0.0,
                                     self.threshold, disabled=True))

    # ---- bookkeeping -------------------------------------------------------
    def _record(self, d: Decision):
        key = "disabled" if d.disabled else ("accept" if d.accepted else "reject")
        self.counts[key] += 1
        if not d.accepted:
            self.reasons[d.reason] = self.reasons.get(d.reason, 0) + 1
        if self._dist_on:
            self._sample(d)
        return d

    def _sample(self, d: Decision):
        """Reservoir-sample the margin and any numeric dial, keyed by outcome and reason."""
        import random
        tag = ("accept" if d.accepted else "reject") + ":" + str(d.reason)[:60]
        items = [("margin", d.margin)] + [(k, v) for k, v in (d.dials or {}).items()]
        for k, v in items:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v is None:
                continue
            slot = self._dist.setdefault(tag + "|" + k, [0, []])
            slot[0] += 1
            if len(slot[1]) < self._dist_cap:
                slot[1].append(float(v))
            else:
                j = random.randrange(slot[0])
                if j < self._dist_cap:
                    slot[1][j] = float(v)

    def summary(self):
        return {"gate": self.name, "threshold": self.threshold,
                "enabled": self.enabled, "counts": dict(self.counts),
                "top_reasons": sorted(self.reasons.items(),
                                      key=lambda kv: -kv[1])[:5],
                "dial_dist": self.dial_dist()}

    def dial_dist(self):
        """Quantiles of every sampled dial, per outcome and reason.  {} when recording is off."""
        out = {}
        for key, (n, xs) in self._dist.items():
            if not xs:
                continue
            ys = sorted(xs)
            def q(f):
                return ys[min(len(ys) - 1, max(0, int(round(f * (len(ys) - 1)))))]
            out[key] = {"n": n, "min": ys[0], "p10": q(0.10), "p50": q(0.50),
                        "p90": q(0.90), "max": ys[-1],
                        "mean": sum(ys) / float(len(ys))}
        return out

    def summary_line(self):
        s = self.summary()
        top = "; ".join("%s x%d" % (r, n) for r, n in s["top_reasons"])
        return ("[DIAL %s] accept=%d reject=%d disabled=%d thr=%s  %s"
                % (self.name, s["counts"]["accept"], s["counts"]["reject"],
                   s["counts"]["disabled"],
                   "-" if self.threshold is None else "%.6g" % self.threshold, top))
