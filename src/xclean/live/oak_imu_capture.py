#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""OAK IMU-only STATIC capture for the Allan-variance noise model (read-only use of the device).

    python -m xclean.live.oak_imu_capture --out DIR --hours 3 [--mxid ID] [--overwrite]

DIR must be new or empty; with --overwrite an earlier capture in it (imu.jsonl, meta.json) is replaced.

The IMU configuration is oak_source.py's (ACCELEROMETER_RAW + GYROSCOPE_RAW at IMU_RATE_REQ_HZ, batch threshold / max
batch / host queue as there), so the measured noise is the noise of the stream the engine consumes.

Writes, flushed every second:
- DIR/imu.jsonl: one record per sample in the recordings' format (t = seconds of the device clock since the first
  sample; ax ay az in m/s^2; gx gy gz in rad/s; accelerometer stamp, as oak_source does).
- DIR/meta.json: mxid, wall start, rates, sample count, gaps over 1.5 periods, the accel/gyro stamp skew, and the chip
  temperature every 60 s (thermal drift is the usual confound of a long static capture).

Stops at --hours, when DIR/STOP appears, or on Ctrl+C. READ-ONLY: no flash / calibration / EEPROM / firmware call; the
only device calls are opening it, the IMU pipeline, and getChipTemperature().
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import timedelta

from xclean.live.oak_source import IMU_BATCH_THRESHOLD, IMU_HOST_QUEUE, IMU_MAX_BATCH, IMU_RATE_REQ_HZ


def _temperature(dev):
    try:
        t = dev.getChipTemperature()
        return {k: float(getattr(t, k)) for k in ("average", "css", "mss", "upa", "dss") if hasattr(t, k)}
    except Exception as e:                                    # not every firmware reports it; never fatal
        return {"error": str(e)}


def _stay_awake(on):
    """Windows: a process-scoped power request (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) so idle sleep cannot cut the
    capture; released at the end. It changes no power setting and needs no rights."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if on else 0x80000000)
        return True
    except Exception:
        return False


def capture(out, hours, mxid=None, rate=IMU_RATE_REQ_HZ, overwrite=True):
    from xclean.io.folders import check_output_folder
    problem = check_output_folder("oak_imu_capture", "--out", out, overwrite, "imu_capture")
    if problem:                                               # refused before anything starts: exit code 2
        print(problem, file=sys.stderr)
        raise SystemExit(2)
    import depthai as dai
    awake = _stay_awake(True)
    if os.path.exists(out):
        shutil.rmtree(out)                                    # an earlier capture (checked above)
    os.makedirs(out)
    stop_file = os.path.join(out, "STOP")
    info = None
    if mxid:
        info = next((d for d in dai.Device.getAllAvailableDevices() if d.getDeviceId() == mxid), None)
        if info is None:
            raise SystemExit("device %s not found" % mxid)
    dev = dai.Device(info) if info is not None else dai.Device()
    meta = {"mxid": dev.getDeviceId(), "wall_start": time.strftime("%Y-%m-%d %H:%M:%S"), "rate_requested_hz": rate,
            "hours_requested": hours, "sensors": ["ACCELEROMETER_RAW", "GYROSCOPE_RAW"],
            "batch_threshold": IMU_BATCH_THRESHOLD, "max_batch": IMU_MAX_BATCH, "temperature": [],
            "samples": 0, "gaps_over_1p5_period": 0, "max_gap_ms": 0.0, "max_accel_gyro_skew_us": 0,
            "nonincreasing": 0, "stopped_by": None, "stay_awake_request": awake}
    p = dai.Pipeline(dev)
    imu = p.create(dai.node.IMU)
    imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], rate)
    imu.setBatchReportThreshold(IMU_BATCH_THRESHOLD)
    imu.setMaxBatchReports(IMU_MAX_BATCH)
    q = imu.out.createOutputQueue(maxSize=IMU_HOST_QUEUE, blocking=True)
    p.start()
    one_us = timedelta(microseconds=1)
    us0 = last_us = None
    period_us = 1e6 / rate
    t_end = time.monotonic() + hours * 3600.0
    t_temp = t_flush = 0.0
    f = open(os.path.join(out, "imu.jsonl"), "w", encoding="utf-8")
    try:
        while True:
            now = time.monotonic()
            if now >= t_end:
                meta["stopped_by"] = "duration"
                break
            if os.path.exists(stop_file):
                meta["stopped_by"] = "STOP file"
                break
            if now - t_temp >= 60.0:
                meta["temperature"].append([round(now - (t_end - hours * 3600.0), 1), _temperature(dev)])
                t_temp = now
            m = q.get(timedelta(seconds=0.5))
            if m is None:
                continue
            for pk in m.packets:
                ac, gy = pk.acceleroMeter, pk.gyroscope
                ua = int(ac.getTimestampDevice() // one_us)
                ug = int(gy.getTimestampDevice() // one_us)
                meta["max_accel_gyro_skew_us"] = max(meta["max_accel_gyro_skew_us"], abs(ug - ua))
                if us0 is None:
                    us0 = ua
                if last_us is not None:
                    if ua <= last_us:
                        meta["nonincreasing"] += 1
                        continue
                    gap = ua - last_us
                    meta["max_gap_ms"] = max(meta["max_gap_ms"], gap / 1e3)
                    if gap > 1.5 * period_us:
                        meta["gaps_over_1p5_period"] += 1
                last_us = ua
                f.write(json.dumps({"t": (ua - us0) / 1e6, "ax": ac.x, "ay": ac.y, "az": ac.z,
                                    "gx": gy.x, "gy": gy.y, "gz": gy.z}) + "\n")
                meta["samples"] += 1
            if now - t_flush >= 1.0:
                f.flush()
                meta["duration_s"] = (last_us - us0) / 1e6 if last_us is not None else 0.0
                json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
                t_flush = now
    except KeyboardInterrupt:
        meta["stopped_by"] = "Ctrl+C"
    finally:
        f.close()
        meta["temperature"].append(["end", _temperature(dev)])      # BEFORE stop: the RPC on a stopped pipeline crashes
        meta["duration_s"] = (last_us - us0) / 1e6 if last_us is not None else 0.0
        meta["rate_measured_hz"] = (meta["samples"] - 1) / meta["duration_s"] if meta["duration_s"] > 0 else None
        meta["wall_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
        try:
            p.stop()
        except Exception:
            pass
        dev.close()
        _stay_awake(False)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="IMU-only capture of a camera at rest, for measuring the IMU noise values "
                                             "(the camera is only read)", allow_abbrev=False)
    ap.add_argument("--out", required=True, help="the capture folder; it must be new or empty")
    ap.add_argument("--hours", type=float, default=3.0, help="how long to capture (default 3)")
    ap.add_argument("--mxid", default=None, help="the device ID of the camera to use when several are attached")
    ap.add_argument("--overwrite", action="store_true", help="replace an earlier capture in --out")
    a = ap.parse_args(argv)
    m = capture(a.out, a.hours, a.mxid, overwrite=a.overwrite)
    print(json.dumps({k: m[k] for k in ("mxid", "samples", "duration_s", "rate_measured_hz", "gaps_over_1p5_period",
                                        "max_gap_ms", "nonincreasing", "stopped_by")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
