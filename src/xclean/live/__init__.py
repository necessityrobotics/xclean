# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""xclean live measurement harness: FEEDER -> ENGINE (fenced) -> RECORDER.

    python -m xclean.live.harness <sequence folder> --out DIR [--mode paced|lockstep] [--pin]
    python -m xclean.live.score_live DIR            # scores RECORDER logs only

Processes:
  feeder.py    pre-decodes images (ahead of time, in RAM), warms nothing itself, releases frames + IMU at their recorded
               stamps (perf_counter_ns); queue depth 1, latest-wins; logs every replaced (dropped) frame; IMU never
               dropped; checks its own release jitter and replay slope.  lockstep mode: releases frame k+1 only after
               the engine finished frame k (causal, unpaced, deterministic) - for loaded accuracy cycles only.
  engine.py    receives ONLY the calibration and the feeder stream; every data access goes through a fence that raises
               on any request newer than the latest released stamp; publishes one message per (frame, stream).
  recorder.py  stamps receipt with perf_counter_ns and appends to a file the engine cannot write.
  harness.py   launches the three, monitors resources (process-tree private commit, VRAM, CPU-s/s), decides validity.
"""
