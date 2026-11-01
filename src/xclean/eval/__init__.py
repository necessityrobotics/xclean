# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Evaluation helpers: the trajectory scorer and the dataset integrity gate.

Vendored from our own earlier research harness so the release has no dependency on it:
    score.py          <- its score.py       (the scoring code verbatim; the command line and the trajectory_live.csv
                                             reading added)
    integrity.py      <- its integrity.py   (verbatim body; imports, the cache path and two messages changed)
    dataset_audit.py  <- its dataset_integrity.py (audit() verbatim except the CALIB messages and the accelerometer
                                             rail rule; the machine-specific root list removed)
"""
