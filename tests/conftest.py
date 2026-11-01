# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Public test suite: CPU only, no datasets, about a minute.  Run from the repository root:  pytest

The GPU is hidden for these tests (and for the subprocesses they start), so they behave the same on any machine.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
