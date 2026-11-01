#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""THE DEPTH STAGE (xclean): our dense 32-direction SGM.  RAFT-Stereo is gone from this engine.

    stage = SgmDepth()
    disp  = stage.compute(left, right)     # (H,W) float32 positive disparity, 0 = INVALID (no measurement)

See xclean/stereo/sgm32.py for the algorithm and the four prototype defects it fixes.  The one behavioural change a
consumer must know: the map has HOLES (0), and a 0 is "no measurement", never "far" (tracker._create handles it).
"""
from xclean.stereo.sgm32 import SgmDepth, SGM32  # noqa: F401
