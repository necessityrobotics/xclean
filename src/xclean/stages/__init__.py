# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Pure stages: they read no pose, no IMU and no map, so they declare key()
and get a content-addressed cache for free (plan section 4.3).
"""
