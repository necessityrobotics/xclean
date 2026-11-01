# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Shared structures of the live harness: the latest-wins frame slot (shared memory) and the message schema."""
from __future__ import annotations

import time
from multiprocessing import shared_memory

import numpy as np

STREAMS = ("LIVE-ODOM", "LIVE-MAP", "LAG")
# header layout (int64): 0 seq, 1 frame_index, 2 release_ns, 3 consumed(0/1), 4 eos(0/1), 5 width, 6 height,
#                        7 replaced_count
HDR_N = 8


def now_ns():
    return time.perf_counter_ns()


class FrameSlot:
    """One frame (raw left + right uint8 images, H x W each) in shared memory, latest-wins.

    The writer (feeder) holds `lock` while it copies a frame in; if the slot still holds an UNCONSUMED frame, that
    frame is REPLACED (dropped) and the feeder logs it.  The reader (engine) holds `lock` while it copies out and marks
    the slot consumed.  Stamps travel as float64 in a separate shared array (the frame stamp t, seconds)."""

    def __init__(self, name, width, height, create=False):
        self.W, self.H = int(width), int(height)
        nbytes = 2 * self.W * self.H
        if create:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes + 8 * HDR_N + 16)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        self.buf = np.ndarray((2, self.H, self.W), np.uint8, self.shm.buf, 0)
        self.hdr = np.ndarray((HDR_N,), np.int64, self.shm.buf, nbytes)
        # tstamp[0] = the frame stamp; tstamp[1] = the stamp of the last IMU sample released BEFORE this frame (the
        # engine waits until its IMU buffer reaches it: queue delivery is asynchronous, shared memory is not)
        self.tstamp = np.ndarray((2,), np.float64, self.shm.buf, nbytes + 8 * HDR_N)
        if create:
            self.hdr[:] = 0
            self.hdr[3] = 1
            self.hdr[5], self.hdr[6] = self.W, self.H
            self.tstamp[:] = -1.0

    def close(self, unlink=False):
        try:
            del self.buf, self.hdr, self.tstamp
        except AttributeError:
            pass
        self.shm.close()
        if unlink:
            self.shm.unlink()


def make_msg(frame_index, t_pose, stream, T, state, n_inliers, degraded, gravity_aligned, segment, horizon, extra=None):
    m = {"frame_index": int(frame_index), "t_pose": float(t_pose), "stream": stream,
         "T": [float(x) for x in np.asarray(T, float).reshape(-1)], "state": str(state),
         "n_inliers": None if n_inliers is None else int(n_inliers), "degraded": bool(degraded),
         "gravity_aligned": bool(gravity_aligned), "map_segment_id": int(segment), "data_horizon": float(horizon)}
    if extra:
        m.update(extra)
    return m
