# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""OAK SOURCE: a live OAK-D-LITE (depthai 3.3) as a drop-in for the harness FEEDER, plus a REPLAY backend that drives
the SAME code path from a recording (so the whole live path is testable without the device).

STREAM CONTRACT - identical to xclean/live/feeder.py (protocol.FrameSlot; the engine cannot tell the two apart):
  * frame k: IMU first - ONE packet (t (n,), g (n,3), a (n,3)) on imu_q holding every received, not yet released
    sample stamped <= t_k (none -> no put); then, under ctl["lock"]: an UNCONSUMED frame still in the slot is REPLACED
    (logged, hdr[7] += 1), buf[0] / buf[1] = raw (unrectified) left / right uint8 H x W, tstamp[0] = t_k,
    tstamp[1] = the stamp of the last IMU sample released before it (-1.0 if none), hdr[1] = frame index,
    hdr[2] = release ns (protocol.now_ns), hdr[3] = 0, hdr[0] += 1; then ctl["new_frame"].set().
  * lockstep (replay only): frame k is released only after the engine reported frame k-1 done (ctl["done"]).
  * end of stream: every remaining received IMU sample, then hdr[4] = 1 under the lock, new_frame set.
  * the log (feeder.json schema): mode, frames_total, frames [[idx, t, planned_ns, actual_ns]], imu_released, replaced,
    validity (paced: jitter p50 / p99 / max of actual - planned and the replay slope), T0_ns; plus source extras.
    planned_ns = the instant the pair became available on the host (device: arrival of its later half; replay: the
    pair's due time), so actual - planned is the source's own overhead.

DEVICE (DeviceBackend): CAM_B / CAM_C (the EEPROM's stereo left / right sockets, read at start) as GRAY8 640 x 400
(centre crop of the OV7251's 640 x 480, the geometry of the EEPROM's K_640x400; device-side undistortion OFF: the
engine rectifies with the calib), the calib's camera fps; IMU ACCELEROMETER_RAW + GYROSCOPE_RAW at the rate the
recordings were made with (200 Hz requested; 197.86 Hz measured on the device clock), batch threshold 1, at most 10
reports per batch.  Device timestamps (getTimestampDevice,
integer microseconds, one monotonic device clock for cameras and IMU) are rebased to t = 0 at the left stamp of the
first released pair (exact integer arithmetic); IMU stamped before it is not part of the session.  The device is only
READ: readCalibration(), never a flash / EEPROM / calibration / firmware call.
REPLAY (ReplayBackend): a recording dir (calib.json, timestamps.csv, imu.jsonl, left/, right/) emitted as device-like
events - each frame at its own stamp (left: timestamp_s, right: right_t), IMU batches at the stamp of their last
sample - paced at real time from a start 50 ms after the go (as the feeder), or back-pressured in lockstep.  Images
decoded ahead (cv2.imread GRAYSCALE, as the feeder).  Stamps are delivered unchanged (the recording's own base).

PAIRING (Pairer): ordered per-side queues; the heads pair when |t_left - t_right| <= half a camera period (class B:
frames of one trigger are hardware-synchronised, 14-20 us apart in both recordings; consecutive triggers are a full
period apart), otherwise the OLDER head is dropped as unpaired.  So a pair never spans a sequence gap.  Sequence gaps
and seq-offset changes are counted per side.  Per-side depth FRAME_QUEUE_DEPTH, drop-oldest (counted).
IMU (ImuBuffer): never dropped, never reordered: released in arrival order as the longest prefix stamped <= t_k.

Constants: FRAME_QUEUE_DEPTH = 2 (class B: one side's frame plus its successor while the
partner is in flight); IMU_RATE_REQ_HZ = 200, IMU_BATCH_THRESHOLD = 1, IMU_MAX_BATCH = 10, IMU_HOST_QUEUE = 50 (class
A: the recordings' device configuration, carried); the pairing tolerance (class B, half a period of the calib's
camera_fps); GO_DELAY_NS = 50 ms and DECODE_AHEAD = 64 / 256 (carried from feeder.py).
"""
from __future__ import annotations

import collections
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from xclean.live.protocol import FrameSlot, now_ns

FRAME_QUEUE_DEPTH = 2
IMU_RATE_REQ_HZ = 200
IMU_BATCH_THRESHOLD = 1
IMU_MAX_BATCH = 10
IMU_HOST_QUEUE = 50
GO_DELAY_NS = 50_000_000            # carried: feeder.py "first release 50 ms after the go"
DECODE_AHEAD_START = 64             # carried: feeder.py decode-ahead window before the clock starts
DECODE_AHEAD_MAX = 256              # carried: feeder.py decode buffer
TS_HEADER = "frame_id,timestamp_s,left_t,right_t"


# ================================================================================================ items
@dataclass
class FrameItem:
    side: str                       # "L" | "R"
    t: float                        # backend-native stamp, seconds (device: us / 1e6; replay: the recording's)
    img: np.ndarray
    seq: int
    idx: int | None = None          # replay: the recording's frame_id; device: assigned at release
    us: int | None = None           # device: integer microseconds of the device clock
    rx_ns: int = 0                  # host arrival (protocol.now_ns)
    cap_lat_ms: float | None = None  # device: host-synced capture -> host arrival
    due_ns: int | None = None       # replay paced: when it was due


@dataclass
class ImuBatch:
    t: np.ndarray                   # backend-native seconds
    g: np.ndarray                   # (n,3) rad/s, IMU frame
    a: np.ndarray                   # (n,3) m/s^2, IMU frame
    us: np.ndarray | None = None    # device: integer microseconds
    rx_ns: int = 0
    dt_ag_us_max: int = 0           # device: max |t_gyro - t_accel| inside the batch


# ================================================================================================ pairing
class Pairer:
    """Stereo pairing on two ordered streams (see module doc)."""

    def __init__(self, tol_s, depth=FRAME_QUEUE_DEPTH):
        self.tol = float(tol_s)
        self.depth = int(depth)
        self.q = {"L": collections.deque(), "R": collections.deque()}
        self.last_seq = {"L": None, "R": None}
        self.stats = {"pairs": 0, "unpaired_L": 0, "unpaired_R": 0, "overflow_L": 0, "overflow_R": 0,
                      "seq_gaps_L": 0, "seq_gaps_R": 0, "seq_missing_L": 0, "seq_missing_R": 0,
                      "seq_offset_changes": 0, "nonincreasing_L": 0, "nonincreasing_R": 0, "max_abs_dt_us": 0.0}
        self.offset = None
        self._last_t = {"L": None, "R": None}

    def push(self, it):
        s = it.side
        ls = self.last_seq[s]
        if ls is not None and it.seq != ls + 1:
            self.stats["seq_gaps_" + s] += 1
            if it.seq > ls + 1:
                self.stats["seq_missing_" + s] += it.seq - ls - 1
        self.last_seq[s] = it.seq
        lt = self._last_t[s]
        if lt is not None and it.t <= lt:
            self.stats["nonincreasing_" + s] += 1
        self._last_t[s] = it.t
        q = self.q[s]
        if len(q) >= self.depth:
            q.popleft()
            self.stats["overflow_" + s] += 1
        q.append(it)
        out = []
        L, R = self.q["L"], self.q["R"]
        while L and R:
            l, r = L[0], R[0]
            dt = l.t - r.t
            if abs(dt) <= self.tol:
                L.popleft()
                R.popleft()
                off = r.seq - l.seq
                if self.offset is not None and off != self.offset:
                    self.stats["seq_offset_changes"] += 1
                self.offset = off
                self.stats["pairs"] += 1
                self.stats["max_abs_dt_us"] = max(self.stats["max_abs_dt_us"], abs(dt) * 1e6)
                out.append((l, r))
            elif dt < 0:
                L.popleft()
                self.stats["unpaired_L"] += 1
            else:
                R.popleft()
                self.stats["unpaired_R"] += 1
        return out


# ================================================================================================ IMU
class ImuBuffer:
    """Received IMU samples on the session clock, released in arrival order (never dropped, never reordered)."""

    def __init__(self):
        self.t = np.zeros(1024)
        self.g = np.zeros((1024, 3))
        self.a = np.zeros((1024, 3))
        self.n = 0
        self.rel = 0
        self.n_nonincreasing = 0
        self.last_released_t = None

    def add(self, t, g, a):
        k = len(t)
        if not k:
            return
        if self.n + k > len(self.t):
            cap = max(2 * len(self.t), self.n + k)
            for nm, shape in (("t", (cap,)), ("g", (cap, 3)), ("a", (cap, 3))):
                old = getattr(self, nm)
                new = np.zeros(shape)
                new[:self.n] = old[:self.n]
                setattr(self, nm, new)
        prev = self.t[self.n - 1] if self.n else -np.inf
        tt = np.asarray(t, np.float64)
        self.n_nonincreasing += int(np.sum(np.diff(np.concatenate([[prev], tt])) <= 0))
        self.t[self.n:self.n + k] = tt
        self.g[self.n:self.n + k] = g
        self.a[self.n:self.n + k] = a
        self.n += k

    def release_upto(self, t_frame):
        """The longest not-yet-released prefix stamped <= t_frame (for sorted stamps: searchsorted side='right',
        exactly the feeder's rule).  -> (t, g, a) copies (possibly empty)."""
        seg = self.t[self.rel:self.n]
        over = np.nonzero(seg > t_frame)[0]
        j = self.rel + (int(over[0]) if len(over) else len(seg))
        return self._take(j)

    def release_all(self):
        return self._take(self.n)

    def _take(self, j):
        i = self.rel
        self.rel = j
        out = (self.t[i:j].copy(), self.g[i:j].copy(), self.a[i:j].copy())
        if j > i:
            self.last_released_t = float(self.t[j - 1])
        return out

    @property
    def pending(self):
        return self.n - self.rel


# ================================================================================================ calibration
def build_calib(K_l, D_l, K_r, D_r, T_l2r_cm, T_imu2cam0_cm, width, height, ref=None, mxid=None, source=""):
    """calib.json fields from EEPROM values, EXACTLY as the 2026-09-22 calib.json of the recordings was built
    (verified field by field by tests/test_oak_source.py on oak_calib/oakd_lite_eeprom.json):
      M1 / M2 = getCameraIntrinsics(left / right, W, H); D1 / D2 = the first 8 of getDistortionCoefficients (the
      RATIONAL model; the EEPROM's tilt terms tauX / tauY are NOT carried, a stated limitation);
      R_stereo, T_stereo = getCameraExtrinsics(left, right) (centimetres -> metres): CAM_B -> CAM_C, the transform
      cv2.stereoRectify takes; baseline_m = -T_stereo[0];
      R_cam0_imu = the rotation of getImuToCameraExtrinsics(left) (body(IMU) -> cam0);
      T_body_cam0 = inverse of that 4 x 4 with its translation in metres.
    Rig values the EEPROM does not hold (imu_noise, camera_fps, imu_rate_hz) come from `ref`, a calib.json of the
    SAME unit (its device_mxid must match when both are known)."""
    T_lr = np.array(T_l2r_cm, float)
    T_ic = np.array(T_imu2cam0_cm, float)
    Tm = T_ic.copy()
    Tm[:3, 3] = T_ic[:3, 3] / 100.0
    cal = {"source": source or "OAK-D-LITE (factory EEPROM, read live by xclean/live/oak_source.py)",
           "device_mxid": mxid, "camera_width": int(width), "camera_height": int(height),
           "M1": np.array(K_l, float).tolist(), "D1": [float(x) for x in list(D_l)[:8]],
           "M2": np.array(K_r, float).tolist(), "D2": [float(x) for x in list(D_r)[:8]],
           "R_stereo": T_lr[:3, :3].tolist(), "T_stereo": (T_lr[:3, 3] / 100.0).tolist(),
           "baseline_m": float(-T_lr[0, 3] / 100.0),
           "R_cam0_imu": T_ic[:3, :3].tolist(), "T_body_cam0": np.linalg.inv(Tm).tolist(),
           "eeprom_tilt_terms_dropped": {"D1_tau": [float(x) for x in list(D_l)[12:14]],
                                         "D2_tau": [float(x) for x in list(D_r)[12:14]]}}
    if ref is not None:
        rm = ref.get("device_mxid")
        if mxid and rm and rm != mxid:
            raise ValueError("reference calib is device %s, the attached device is %s" % (rm, mxid))
        for k in ("camera_fps", "imu_rate_hz", "imu_noise"):
            if k in ref:
                cal[k] = ref[k]
        cal["rig_values_from"] = "imu_noise, camera_fps, imu_rate_hz carried from the reference calib (%s)" \
                                 % ref.get("dataset", "?")
    return cal


def calib_from_eeprom_dump(dump, width=640, height=400, ref=None):
    """The same build from an EEPROM dump oakd_lite_eeprom.json (the 2026-09-22 read-only dump)."""
    left = str(dump.get("stereo_left_socket", "CameraBoardSocket.CAM_B")).split(".")[-1]
    right = str(dump.get("stereo_right_socket", "CameraBoardSocket.CAM_C")).split(".")[-1]
    cams = dump["cameras"]
    key = "K_%dx%d" % (width, height)
    return build_calib(cams[left][key], cams[left]["getDistortionCoefficients"], cams[right][key],
                       cams[right]["getDistortionCoefficients"], dump["extrinsics"]["%s_to_%s" % (left, right)],
                       dump["imu_to_cam"][left], width, height, ref=ref, mxid=dump.get("mxid"),
                       source="OAK-D-LITE (factory EEPROM dump %s)" % dump.get("captured", "?"))


# ================================================================================================ backends
class ReplayBackend:
    """A recording dir emitted as device-like events (module doc)."""
    kind = "replay"
    live = False                    # stamps used as given; paced or lock-step (run_source)

    def __init__(self, path, mode, frames=None, ext=None):
        import cv2  # noqa: F401  (decode in the reader thread)
        self.path = str(path)
        self.mode = mode
        self.ext = ext or self._detect_ext()
        rows = []
        with open(os.path.join(self.path, "timestamps.csv"), encoding="utf-8") as f:
            f.readline()
            for ln in f:
                p = ln.strip().split(",")
                if len(p) >= 2:
                    rows.append((int(p[0]), float(p[1]), float(p[3]) if len(p) >= 4 else float(p[1])))
        if frames:
            rows = rows[:int(frames)]
        self.rows = rows
        from xclean.live.feeder import _read_imu          # the feeder's own IMU reader: identical floats
        self.ti, self.gi, self.ai = _read_imu(self.path)
        self.calib = json.load(open(os.path.join(self.path, "calib.json"), encoding="utf-8"))
        self._stop = threading.Event()
        self._go = threading.Event()
        self.T0 = None
        self.t_first = rows[0][1] if rows else 0.0
        self.emit_late_ns = []
        self.finished = False
        self.read_error = None

    def _detect_ext(self):
        for e in ("png", "jpg"):
            if os.path.isfile(os.path.join(self.path, "left", "000000.%s" % e)):
                return e
        names = sorted(os.listdir(os.path.join(self.path, "left")))
        return names[0].rsplit(".", 1)[-1] if names else "png"

    # the session clock is the recording's own
    def set_origin(self, item):
        pass

    def session_t(self, x):
        return x

    def start(self, events):
        import cv2
        from xclean.io.imageio import imread
        self.events = events
        self.buf = queue.Queue(maxsize=DECODE_AHEAD_MAX)

        def reader():
            for idx, _t, _tr in self.rows:
                if self._stop.is_set():
                    return
                pl = "%s/left/%06d.%s" % (self.path, idx, self.ext)
                pr = "%s/right/%06d.%s" % (self.path, idx, self.ext)
                l = imread(pl, cv2.IMREAD_GRAYSCALE)
                r = imread(pr, cv2.IMREAD_GRAYSCALE)
                if l is None or r is None:
                    self.read_error = "cannot read %s" % (pl if l is None else pr)
                    self.buf.put(None)
                    return
                self.buf.put((idx, l, r))
            self.buf.put(None)

        self._reader = threading.Thread(target=reader, daemon=True, name="replay-decode")
        self._reader.start()
        t_wait = time.time()
        while (self.buf.qsize() < min(DECODE_AHEAD_START, len(self.rows)) and self._reader.is_alive()
               and time.time() - t_wait < 60):
            time.sleep(0.01)
        self._emitter = threading.Thread(target=self._emit, daemon=True, name="replay-emit")
        self._emitter.start()

    def go(self):
        self.T0 = now_ns() + GO_DELAY_NS
        self._go.set()

    def _put(self, ev):
        while not self._stop.is_set():
            try:
                self.events.put(ev, timeout=0.05)
                return
            except queue.Full:
                pass

    def _wait_until(self, ns):
        """feeder.py's sleep-then-spin, except that the final approach yields (time.sleep(0)): this runs in a
        thread beside the source loop, and a pure spin would hold the GIL for a whole switch interval (5 ms)."""
        while True:
            d = ns - now_ns()
            if d <= 0:
                return
            if d > 2_000_000:
                time.sleep((d - 1_500_000) / 1e9)
            else:
                time.sleep(0)

    def _emit(self):
        self._go.wait()
        paced = self.mode != "lockstep"
        ti, gi, ai = self.ti, self.gi, self.ai
        ii = 0
        n = len(ti)

        def due(t):
            return self.T0 + int(round((t - self.t_first) * 1e9))

        def imu_upto(t, inclusive=True):
            nonlocal ii
            j = int(np.searchsorted(ti, t, side="right" if inclusive else "left"))
            if j > ii:
                b = ImuBatch(ti[ii:j].copy(), gi[ii:j].copy(), ai[ii:j].copy())
                if paced:
                    self._wait_until(due(float(ti[j - 1])))
                b.rx_ns = now_ns()
                self._put(("I", b))
                ii = j

        for idx, tl, tr in self.rows:
            if self._stop.is_set():
                break
            item = self.buf.get()
            if item is None:
                break
            _i, l, r = item
            imu_upto(tl)                                   # IMU at or before the left stamp first (ties: IMU first)
            for side, t, img in (("L", tl, l), ("R", tr, r)):
                if side == "R":
                    imu_upto(tr)
                d = due(t)
                if paced:
                    self._wait_until(d)
                rx = now_ns()
                if paced:
                    self.emit_late_ns.append(rx - d)
                self._put((side, FrameItem(side=side, t=t, img=img, seq=idx, idx=idx, rx_ns=rx,
                                           due_ns=d if paced else None)))
        # end of stream: every remaining IMU sample (the feeder releases the whole file at EOS), then EOS
        if ii < n and not self._stop.is_set():
            self._put(("I", ImuBatch(ti[ii:].copy(), gi[ii:].copy(), ai[ii:].copy(), rx_ns=now_ns())))
        self._put(("EOS",))
        self.finished = True

    def stop(self):
        self._stop.set()

    def info(self):
        late = np.asarray(self.emit_late_ns, float) / 1e6
        info = {"backend": "replay", "path": self.path, "ext": self.ext, "frames_planned": len(self.rows),
                "imu_samples_in_file": int(len(self.ti)),
                "emit_late_ms": ({"p50": float(np.median(late)), "p99": float(np.percentile(late, 99)),
                                  "max": float(late.max())} if len(late) else None)}
        if self.read_error:                   # failure path only
            info["read_error"] = self.read_error
        return info


class DeviceBackend:
    """The OAK-D-LITE through depthai 3.3 (module doc).  READ-ONLY use of the device."""
    kind = "device"
    live = True                     # a camera: rebased integer-us clock, warm-up frames dropped, no lock-step

    def __init__(self, width, height, fps, mxid=None, imu_rate=IMU_RATE_REQ_HZ):
        self.W, self.H, self.fps = int(width), int(height), float(fps)
        self.mxid_req = mxid
        self.imu_rate = int(imu_rate)
        self._stop = threading.Event()
        self.us0 = None
        self.dev = None
        self.pipeline = None
        self.threads = []
        self.n_msgs = {"L": 0, "R": 0, "I": 0}
        self.shape_errors = 0
        self.errors = []

    # ---- device + calibration (read-only)
    def open(self):
        import depthai as dai
        self.dai = dai
        devs = dai.Device.getAllAvailableDevices()
        if not devs:
            raise RuntimeError("no OAK device available (depthai.Device.getAllAvailableDevices() is empty)")
        info = None
        if self.mxid_req:
            for d in devs:
                if d.getDeviceId() == self.mxid_req:
                    info = d
            if info is None:
                raise RuntimeError("device %s not available; available: %s"
                                   % (self.mxid_req, [d.getDeviceId() for d in devs]))
        self.dev = dai.Device(info) if info is not None else dai.Device()
        self.mxid = self.dev.getDeviceId()
        self.ch = self.dev.readCalibration()               # READ-ONLY
        self.left_socket = self.ch.getStereoLeftCameraId()
        self.right_socket = self.ch.getStereoRightCameraId()
        return self

    def calib(self, ref=None):
        ch, L, R = self.ch, self.left_socket, self.right_socket
        return build_calib(ch.getCameraIntrinsics(L, self.W, self.H), ch.getDistortionCoefficients(L),
                           ch.getCameraIntrinsics(R, self.W, self.H), ch.getDistortionCoefficients(R),
                           ch.getCameraExtrinsics(L, R), ch.getImuToCameraExtrinsics(L), self.W, self.H,
                           ref=ref, mxid=self.mxid,
                           source="OAK-D-LITE %s (factory EEPROM, read live %s)"
                                  % (self.dev.getDeviceName(), time.strftime("%Y-%m-%d %H:%M:%S")))

    # ---- session clock: integer microseconds of the device clock, origin = the first released pair's left stamp
    def set_origin(self, item):
        if self.us0 is None:
            self.us0 = int(item.us)

    def session_t(self, x):
        """x: a FrameItem, an int (us) or an int array (us) -> session seconds (exact integer rebase)."""
        if isinstance(x, FrameItem):
            return (int(x.us) - self.us0) / 1e6
        if isinstance(x, np.ndarray):
            return (x.astype(np.int64) - self.us0) / 1e6
        return (int(x) - self.us0) / 1e6

    # ---- streaming
    def start(self, events):
        dai = self.dai
        self.events = events
        p = dai.Pipeline(self.dev)
        outs = {}
        for side, sock in (("L", self.left_socket), ("R", self.right_socket)):
            cam = p.create(dai.node.Camera).build(boardSocket=sock, sensorFps=self.fps)
            o = cam.requestOutput(size=(self.W, self.H), type=dai.ImgFrame.Type.GRAY8,
                                  resizeMode=dai.ImgResizeMode.CROP, fps=self.fps, enableUndistortion=False)
            outs[side] = o.createOutputQueue(maxSize=FRAME_QUEUE_DEPTH, blocking=False)
        imu = p.create(dai.node.IMU)
        imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], self.imu_rate)
        imu.setBatchReportThreshold(IMU_BATCH_THRESHOLD)
        imu.setMaxBatchReports(IMU_MAX_BATCH)
        q_imu = imu.out.createOutputQueue(maxSize=IMU_HOST_QUEUE, blocking=True)     # never drop IMU
        self.pipeline = p
        p.start()
        for side in ("L", "R"):
            th = threading.Thread(target=self._cam_reader, args=(outs[side], side), daemon=True,
                                  name="oak-cam-" + side)
            th.start()
            self.threads.append(th)
        th = threading.Thread(target=self._imu_reader, args=(q_imu,), daemon=True, name="oak-imu")
        th.start()
        self.threads.append(th)

    def go(self):
        pass

    def _get(self, q):
        from datetime import timedelta
        return q.get(timedelta(seconds=1.0 / self.fps))     # bounded only so that stop() is honoured

    def _cam_reader(self, q, side):
        from datetime import timedelta
        dai = self.dai
        one_us = timedelta(microseconds=1)
        while not self._stop.is_set():
            try:
                m = self._get(q)
            except Exception as e:                           # queue closed at pipeline stop
                if not self._stop.is_set():
                    self.errors.append("%s reader: %s" % (side, e))
                return
            if m is None:
                continue
            rx = now_ns()
            lat = (dai.Clock.now() - m.getTimestamp()).total_seconds() * 1e3
            img = m.getCvFrame()
            if img.ndim != 2 or img.shape != (self.H, self.W):
                self.shape_errors += 1
                continue
            us = m.getTimestampDevice() // one_us
            self.n_msgs[side] += 1
            self.events.put((side, FrameItem(side=side, t=us / 1e6, img=np.ascontiguousarray(img),
                                             seq=int(m.getSequenceNum()), us=int(us), rx_ns=rx,
                                             cap_lat_ms=float(lat))))

    def _imu_reader(self, q):
        from datetime import timedelta
        one_us = timedelta(microseconds=1)
        while not self._stop.is_set():
            try:
                m = self._get(q)
            except Exception as e:
                if not self._stop.is_set():
                    self.errors.append("imu reader: %s" % e)
                return
            if m is None:
                continue
            rx = now_ns()
            us, g, a, dag = [], [], [], 0
            for pk in m.packets:
                ac, gy = pk.acceleroMeter, pk.gyroscope
                ua = ac.getTimestampDevice() // one_us
                ug = gy.getTimestampDevice() // one_us
                dag = max(dag, abs(int(ug) - int(ua)))
                us.append(int(ua))
                g.append((gy.x, gy.y, gy.z))
                a.append((ac.x, ac.y, ac.z))
            if us:
                self.n_msgs["I"] += 1
                u = np.asarray(us, np.int64)
                self.events.put(("I", ImuBatch(u / 1e6, np.asarray(g, float), np.asarray(a, float), us=u,
                                               rx_ns=rx, dt_ag_us_max=dag)))

    def stop(self):
        self._stop.set()
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception as e:
            self.errors.append("pipeline.stop: %s" % e)
        for th in self.threads:
            th.join(2.0)
        try:
            if self.dev is not None:
                self.dev.close()
        except Exception as e:
            self.errors.append("device.close: %s" % e)

    @property
    def finished(self):
        return False

    def info(self):
        return {"backend": "device", "mxid": getattr(self, "mxid", None), "fps_requested": self.fps,
                "imu_rate_requested_hz": self.imu_rate, "messages": dict(self.n_msgs),
                "shape_errors": self.shape_errors, "errors": self.errors[:20],
                "left_socket": str(getattr(self, "left_socket", None)),
                "right_socket": str(getattr(self, "right_socket", None)), "us0": self.us0}


# ================================================================================================ stream recorder
class StreamRecorder:
    """Writes the RELEASED stream as a replayable recording (the recordings' layout): left/ right/ %06d.<ext>
    (png: lossless, so a replay feeds the engine the same bytes), timestamps.csv (frame_id,timestamp_s,left_t,right_t),
    imu.jsonl (every released sample, in order), calib.json (the session calib + t0 / counts), stream_meta.csv (device
    seq numbers, L-R skew, capture latency per frame).  A writer thread with an unbounded queue: nothing is dropped;
    the backlog peak is reported."""

    def __init__(self, path, calib, ext="png"):
        import cv2  # noqa: F401
        self.path = path
        self.ext = ext
        self.calib = dict(calib)
        for sub in ("left", "right"):
            os.makedirs(os.path.join(path, sub), exist_ok=True)
        self.q = queue.Queue()
        self.n_frames = 0
        self.n_imu = 0
        self.backlog_max = 0
        self.t_frames = []
        self.t_imu_last = None
        self.t_imu_first = None
        self.write_errors = 0
        self.first_write_error = None
        self.f_ts = open(os.path.join(path, "timestamps.csv"), "w", encoding="utf-8", newline="\n")
        self.f_ts.write(TS_HEADER + "\n")
        self.f_imu = open(os.path.join(path, "imu.jsonl"), "w", encoding="utf-8", newline="\n")
        self.f_meta = open(os.path.join(path, "stream_meta.csv"), "w", encoding="utf-8", newline="\n")
        self.f_meta.write("frame_id,seq_left,seq_right,dt_lr_us,cap_lat_left_ms,cap_lat_right_ms,rx_ns,release_ns\n")
        self._write_calib()
        self.th = threading.Thread(target=self._run, daemon=True, name="stream-recorder")
        self.th.start()

    def _write_calib(self, extra=None):
        c = dict(self.calib)
        if extra:
            c.update(extra)
        json.dump(c, open(os.path.join(self.path, "calib.json"), "w", encoding="utf-8"), indent=1)

    def frame(self, idx, tl, tr, l, r, meta):
        self.q.put(("F", idx, tl, tr, l, r, meta))
        self.backlog_max = max(self.backlog_max, self.q.qsize())

    def imu(self, t, g, a):
        if len(t):
            self.q.put(("I", t, g, a))

    def _run(self):
        import cv2
        from xclean.io.imageio import imwrite
        prm = [cv2.IMWRITE_PNG_COMPRESSION, 1] if self.ext == "png" else []
        while True:
            job = self.q.get()
            if job is None:
                return
            if job[0] == "F":
                _k, idx, tl, tr, l, r, meta = job
                for side, img in (("left", l), ("right", r)):
                    p = os.path.join(self.path, side, "%06d.%s" % (idx, self.ext))
                    try:
                        ok = imwrite(p, img, prm)
                        why = None if ok else "the image could not be encoded"
                    except OSError as e:
                        why = "%s: %s" % (type(e).__name__, e)
                    if why:                   # counted and reported at the end; the stream goes on
                        self.write_errors += 1
                        if self.first_write_error is None:
                            self.first_write_error = "%s (%s)" % (p, why)
                self.f_ts.write("%d,%.6f,%.6f,%.6f\n" % (idx, tl, tl, tr))
                self.f_meta.write(",".join(str(x) for x in [idx] + list(meta)) + "\n")
                self.n_frames += 1
                self.t_frames.append(tl)
            else:
                _k, t, g, a = job
                for i in range(len(t)):
                    self.f_imu.write(json.dumps({"t": float(t[i]), "ax": float(a[i, 0]), "ay": float(a[i, 1]),
                                                 "az": float(a[i, 2]), "gx": float(g[i, 0]), "gy": float(g[i, 1]),
                                                 "gz": float(g[i, 2])}) + "\n")
                self.n_imu += len(t)
                if self.t_imu_first is None:
                    self.t_imu_first = float(t[0])
                self.t_imu_last = float(t[-1])

    def close(self, t0_ns=None):
        self.q.put(None)
        self.th.join()
        for f in (self.f_ts, self.f_imu, self.f_meta):
            f.close()
        tf = np.asarray(self.t_frames, float)
        extra = {"dataset": os.path.basename(os.path.normpath(self.path)), "num_frames": self.n_frames,
                 "t0_ns": t0_ns, "recorded_by": "xclean/live/oak_source.py StreamRecorder",
                 "measured_camera_fps": (float(1.0 / np.median(np.diff(tf))) if len(tf) > 1 else None),
                 "image_ext": self.ext}
        if self.n_imu > 1 and self.t_imu_first is not None:
            extra["measured_imu_rate_hz"] = float((self.n_imu - 1) / max(self.t_imu_last - self.t_imu_first, 1e-9))
        self._write_calib(extra)
        info = {"path": self.path, "frames": self.n_frames, "imu_samples": self.n_imu, "backlog_max": self.backlog_max}
        if self.write_errors:                 # only when images were lost (a complete recording's log is unchanged)
            info.update(image_write_errors=self.write_errors, first_write_error=self.first_write_error)
        return info


# ================================================================================================ release rings (view)
class ReleaseRing:
    """Shared memory, written by the source: per released frame (idx, release_ns, capture->release ms), indexed by
    idx mod N - the demo view joins it with the published poses to show the per-frame latency."""
    N = 1024

    def __init__(self, name, create=False):
        from multiprocessing import shared_memory
        nbytes = self.N * 3 * 8
        self.shm = shared_memory.SharedMemory(name=name, create=create, size=nbytes)
        self.a = np.ndarray((self.N, 3), np.float64, self.shm.buf, 0)
        if create:
            self.a[:] = -1.0

    def write(self, idx, rel_ns, cap_ms):
        self.a[idx % self.N] = (idx, float(rel_ns), np.nan if cap_ms is None else cap_ms)

    def close(self, unlink=False):
        try:
            del self.a
        except AttributeError:
            pass
        self.shm.close()
        if unlink:
            self.shm.unlink()


STATE_CODE = {"init": 0, "tracking": 1, "coasting": 2}            # anything else -> 3


class PoseRing:
    """Shared memory ring of the published LIVE-ODOM poses for the demo view: count (int64) + N x 18 float64
    [frame_index, t, put_ns, state_code, n_inliers, segment, x, y, z, R (9, row-major)].  Written by TapQueue in the
    engine process, read by the view; a torn read is harmless (display only)."""
    N = 1024
    F = 18

    def __init__(self, name, create=False):
        from multiprocessing import shared_memory
        nbytes = 8 + self.N * self.F * 8
        self.shm = shared_memory.SharedMemory(name=name, create=create, size=nbytes)
        self.cnt = np.ndarray((1,), np.int64, self.shm.buf, 0)
        self.a = np.ndarray((self.N, self.F), np.float64, self.shm.buf, 8)
        if create:
            self.cnt[0] = 0

    def write(self, m, put_ns):
        T = m["T"]
        row = [m["frame_index"], m["t_pose"], float(put_ns), STATE_CODE.get(m.get("state"), 3),
               -1 if m.get("n_inliers") is None else m["n_inliers"], m.get("map_segment_id", 0),
               T[3], T[7], T[11], T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10]]
        c = int(self.cnt[0])
        self.a[c % self.N] = row
        self.cnt[0] = c + 1

    def read_since(self, c0):
        """-> (new count, rows written since c0 (at most N, oldest first))."""
        c = int(self.cnt[0])
        n = min(c - c0, self.N)
        if n <= 0:
            return c, np.zeros((0, self.F))
        idx = [(c - n + i) % self.N for i in range(n)]
        return c, self.a[idx].copy()

    def close(self, unlink=False):
        try:
            del self.a, self.cnt
        except AttributeError:
            pass
        self.shm.close()
        if unlink:
            self.shm.unlink()


class TapQueue:
    """The msg_q the ENGINE sees in the demo when the view is on: put() forwards every message UNCHANGED to the
    recorder's queue first (the recorder still stamps receipt itself, the recorder log is exactly the harness's), then
    copies a LIVE-ODOM pose into a PoseRing (~10 us in the engine process).  With the view off the demo passes the
    plain queue, i.e. the harness's exact path."""

    def __init__(self, q, ring_name):
        self.q = q
        self.ring_name = ring_name
        self._ring = None

    def __getstate__(self):
        return {"q": self.q, "ring_name": self.ring_name}

    def __setstate__(self, s):
        self.q = s["q"]
        self.ring_name = s["ring_name"]
        self._ring = None

    def put(self, m, *a, **k):
        self.q.put(m, *a, **k)
        if isinstance(m, dict) and m.get("stream") == "LIVE-ODOM":
            try:
                if self._ring is None:
                    self._ring = PoseRing(self.ring_name)
                self._ring.write(m, now_ns())
            except Exception:
                pass                                          # display only: never disturbs the engine


# ================================================================================================ the source loop
def _validity(frames, mode):
    fr = np.asarray([[f[2], f[3]] for f in frames], float)
    stamps = np.asarray([f[1] for f in frames], float)
    if mode != "paced" or len(fr) <= 2:
        return {}
    jit = (fr[:, 1] - fr[:, 0]) / 1e6
    A = np.stack([stamps - stamps[0], np.ones(len(stamps))], 1)
    slope = float(np.linalg.lstsq(A, (fr[:, 1] - fr[0, 1]) / 1e9, rcond=None)[0][0])
    return {"jitter_ms_p50": float(np.percentile(jit, 50)), "jitter_ms_p99": float(np.percentile(jit, 99)),
            "jitter_ms_max": float(jit.max()), "replay_slope": slope,
            "valid": bool(np.percentile(jit, 99) <= 1.0 and abs(slope - 1.0) <= 1e-3),
            "rule": "feeder.py: jitter p99 <= 1 ms and slope 1 +- 1e-3 (device: slope = host clock / device clock)"}


def run_source(backend, cfg, slot, imu_q, ctl, log_path, recorder=None, rel_ring=None):
    """The common live-source loop (device, replay, and the frame sources of xclean/live/sources.py).  cfg: mode
    ('paced' | 'lockstep'), fps (calib camera_fps), duration_s (device: stop after this long of released stream;
    None = until ctl['stop']).  backend.live (default: kind == 'device') says how the backend is treated: True = a
    camera (session clock rebased to the first released pair in integer microseconds, frames queued during the
    engine's warm-up dropped, no lock-step); False = a recording (stamps used as given, lock-step possible)."""
    mode = cfg.get("mode", "paced")
    live = bool(getattr(backend, "live", backend.kind == "device"))
    if mode == "lockstep" and live:
        raise ValueError("lockstep needs a replay source (a camera cannot wait for the engine)")
    fps = float(cfg["fps"])
    pairer = Pairer(0.5 / fps)
    imub = ImuBuffer()
    events = queue.Queue(maxsize=(8 if mode == "lockstep" else 0))   # lockstep: back-pressure (no frame is dropped)
    log = {"mode": mode, "source": backend.kind, "frames": [], "imu_released": 0, "replaced": [],
           "pre_ready_frames": 0, "pre_ready_imu": 0, "imu_pre_session": 0, "imu_packets": 0}
    extra = []                    # per frame: [idx, seq_l, seq_r, dt_lr_us, cap_l_ms, cap_r_ms, cap_to_rel_ms,
                                  #             imu_lag_ms (t_k - last released IMU stamp)]
    pre_imu = []                  # device: IMU received after the go but before the time origin
    origin = not live
    backend.start(events)
    # engine warm-up: the device already streams; everything before the go is discarded (counted)
    while not ctl["ready"].is_set() and not ctl["stop"].is_set():
        try:
            ev = events.get(timeout=0.01)
        except queue.Empty:
            continue
        if ev[0] in ("L", "R"):
            log["pre_ready_frames"] += 1
        elif ev[0] == "I":
            log["pre_ready_imu"] += len(ev[1].t)
    if not ctl["ready"].is_set():
        # stopped before the engine was ready (the engine failed during start-up, or the user stopped the demo): end
        # without releasing a frame, mark the end of stream for an engine that may still become ready, write the log
        backend.stop()
        with ctl["lock"]:
            slot.hdr[4] = 1
        ctl["new_frame"].set()
        log.update(frames_total=0, stopped_before_ready=True, validity={}, backend=backend.info())
        if recorder is not None:
            log["recording"] = recorder.close()
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, default=str)
        return log
    if live:                                                    # frames queued during warm-up are stale
        while True:
            try:
                ev = events.get_nowait()
            except queue.Empty:
                break
            if ev[0] in ("L", "R"):
                log["pre_ready_frames"] += 1
            elif ev[0] == "I":
                log["pre_ready_imu"] += len(ev[1].t)
    backend.go()
    T0 = getattr(backend, "T0", None) or now_ns()
    k = 0
    prev_idx = None
    t_first_rel = None
    n_imu_max_pending = 0
    period = 1.0 / fps
    while True:
        if ctl["stop"].is_set():
            break
        try:
            ev = events.get(timeout=period)
        except queue.Empty:
            if backend.finished and events.empty():
                break
            continue
        if ev[0] == "EOS":
            break
        if ev[0] == "I":
            b = ev[1]
            if not origin:
                pre_imu.append(b)
                continue
            if live:
                ts = backend.session_t(b.us)
                keep = ts >= 0.0
                log["imu_pre_session"] += int((~keep).sum())
                imub.add(ts[keep], b.g[keep], b.a[keep])
            else:
                imub.add(b.t, b.g, b.a)
            n_imu_max_pending = max(n_imu_max_pending, imub.pending)
            continue
        for l, r in pairer.push(ev[1]):
            if not origin:
                backend.set_origin(l)
                origin = True
                for b in pre_imu:
                    ts = backend.session_t(b.us)
                    keep = ts >= 0.0
                    log["imu_pre_session"] += int((~keep).sum())
                    imub.add(ts[keep], b.g[keep], b.a[keep])
                pre_imu = []
            t = backend.session_t(l) if live else l.t
            tr = backend.session_t(r) if live else r.t
            idx = l.idx if l.idx is not None else k
            if mode == "lockstep" and prev_idx is not None:
                while True:
                    if ctl["done"].value >= prev_idx or ctl["stop"].is_set():
                        break
                    ctl["done_evt"].wait(0.05)
                    ctl["done_evt"].clear()
            planned = (max(l.due_ns, r.due_ns) if (l.due_ns is not None and r.due_ns is not None)
                       else max(l.rx_ns, r.rx_ns))
            ti_, gi_, ai_ = imub.release_upto(t)
            if len(ti_):
                imu_q.put((ti_, gi_, ai_))
                log["imu_released"] += len(ti_)
                log["imu_packets"] += 1
            if ctl["stop"].is_set():
                break
            with ctl["lock"]:
                if slot.hdr[3] == 0 and slot.hdr[0] > 0:            # the previous frame was never consumed
                    log["replaced"].append(int(slot.hdr[1]))
                    slot.hdr[7] += 1
                slot.buf[0] = l.img
                slot.buf[1] = r.img
                slot.tstamp[0] = t
                slot.tstamp[1] = imub.last_released_t if imub.last_released_t is not None else -1.0
                slot.hdr[1] = idx
                actual = now_ns()
                slot.hdr[2] = actual
                slot.hdr[3] = 0
                slot.hdr[0] += 1
            ctl["new_frame"].set()
            if t_first_rel is None:
                t_first_rel = actual
            log["frames"].append([int(idx), float(t), int(planned), int(actual)])
            cap = None
            if l.cap_lat_ms is not None:
                cap = (actual - l.rx_ns) / 1e6 + l.cap_lat_ms
            lag = None if imub.last_released_t is None else 1e3 * (t - imub.last_released_t)
            extra.append([int(idx), int(l.seq), int(r.seq), float((r.t - l.t) * 1e6), l.cap_lat_ms, r.cap_lat_ms,
                          cap, lag])
            if rel_ring is not None:
                rel_ring.write(int(idx), actual, cap)
            if recorder is not None:
                recorder.imu(ti_, gi_, ai_)
                recorder.frame(int(idx), t, tr, l.img, r.img,
                               [int(l.seq), int(r.seq), float((r.t - l.t) * 1e6), l.cap_lat_ms, r.cap_lat_ms,
                                max(l.rx_ns, r.rx_ns), actual])
            prev_idx = idx
            k += 1
        dur = cfg.get("duration_s")
        if dur and t_first_rel is not None and now_ns() - t_first_rel >= dur * 1e9:
            break
    # ---- end of stream: everything still buffered (the feeder releases the rest of the IMU), then EOS
    backend.stop()
    while True:                                                 # replay: events already emitted (IMU tail)
        try:
            ev = events.get_nowait()
        except queue.Empty:
            break
        if ev[0] == "I" and origin:
            b = ev[1]
            if live:
                ts = backend.session_t(b.us)
                keep = ts >= 0.0
                imub.add(ts[keep], b.g[keep], b.a[keep])
            else:
                imub.add(b.t, b.g, b.a)
    ti_, gi_, ai_ = imub.release_all()
    if len(ti_):
        imu_q.put((ti_, gi_, ai_))
        log["imu_released"] += len(ti_)
        log["imu_packets"] += 1
        if recorder is not None:
            recorder.imu(ti_, gi_, ai_)
    with ctl["lock"]:
        slot.hdr[4] = 1
    ctl["new_frame"].set()
    log["frames_total"] = len(log["frames"])
    log["validity"] = _validity(log["frames"], mode)
    log["T0_ns"] = int(T0)
    log["pairing"] = pairer.stats
    log["imu_nonincreasing"] = imub.n_nonincreasing
    log["imu_max_pending"] = n_imu_max_pending
    log["backend"] = backend.info()
    log["source_frames"] = extra
    log["source_frames_cols"] = ["idx", "seq_left", "seq_right", "dt_lr_us", "cap_lat_left_ms", "cap_lat_right_ms",
                                 "capture_to_release_ms", "imu_lag_ms"]
    if live and backend.us0 is not None:
        log["t0_device_us"] = int(backend.us0)
    if recorder is not None:
        log["recording"] = recorder.close(t0_ns=(int(backend.us0) * 1000 if live
                                                 and backend.us0 is not None else None))
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, default=str)
    if getattr(backend, "source_error", None):  # a frame source (sources.py) ended with an error: an error exit
        raise SystemExit("xclean: error: the source %s stopped with an error: %s" % (backend.name,
                                                                                   backend.source_error))
    if getattr(backend, "read_error", None):  # the stream ended early on an unreadable image: an error exit
        raise SystemExit("xclean: error: the recording could not be read to the end: %s" % backend.read_error)
    return log


def list_devices():
    """Read-only enumeration of attached OAK devices."""
    import depthai as dai
    return [{"mxid": d.getDeviceId(), "name": d.name, "state": str(d.state), "protocol": str(d.protocol)}
            for d in dai.Device.getAllAvailableDevices()]


def smoke(seconds=20.0, mxid=None, replay=None, record_dir=None, ref_calib=None, out_json=None):
    """The source WITHOUT an engine: a consumer thread takes every frame from the slot (as an engine would) and drains
    the IMU queue.  Reports the measured frame rate (device clock), pair-sync error, IMU rate, timestamp monotonicity,
    drops and capture latency.  replay=<dir> runs the same path from a recording (paced)."""
    import tempfile
    calib = None
    if replay:
        backend = ReplayBackend(replay, "paced", frames=int(seconds * 30.0))
        calib = backend.calib
    else:
        backend = DeviceBackend(640, 400, 30.0, mxid).open()
        calib = backend.calib(ref=ref_calib)
    W, H = int(calib["camera_width"]), int(calib["camera_height"])
    name = "xclean_oaksmoke_%d" % os.getpid()
    slot = FrameSlot(name, W, H, create=True)
    ctl = {"ready": threading.Event(), "new_frame": threading.Event(), "stop": threading.Event(),
           "lock": threading.Lock(), "done_evt": threading.Event()}

    class _V:
        value = -1
    ctl["done"] = _V()
    imu_q = queue.Queue()
    got = {"frames": [], "imu_t": []}

    def consumer():
        last = 0
        while True:
            ctl["new_frame"].wait(0.5)
            ctl["new_frame"].clear()
            with ctl["lock"]:
                seq, eos = int(slot.hdr[0]), bool(slot.hdr[4])
                if seq != last and slot.hdr[3] == 0:
                    got["frames"].append((int(slot.hdr[1]), float(slot.tstamp[0]), now_ns() - int(slot.hdr[2])))
                    slot.hdr[3] = 1
                    last = seq
            while True:
                try:
                    got["imu_t"].append(imu_q.get_nowait()[0])
                except queue.Empty:
                    break
            if eos:
                return
    th = threading.Thread(target=consumer, daemon=True)
    th.start()
    ctl["ready"].set()
    rec = StreamRecorder(record_dir, calib) if record_dir else None
    log_path = out_json or os.path.join(tempfile.mkdtemp(prefix="oaksmoke_"), "source.json")
    log = run_source(backend, {"mode": "paced", "fps": float(calib.get("camera_fps") or 30.0),
                               "duration_s": None if replay else seconds}, slot, imu_q, ctl, log_path, recorder=rec)
    th.join(10)
    slot.close(unlink=True)
    t = np.array([f[1] for f in log["frames"]], float)
    it = np.concatenate(got["imu_t"]) if got["imu_t"] else np.zeros(0)
    src = np.array([[np.nan if x is None else x for x in r] for r in log["source_frames"]], float) \
        if log["source_frames"] else np.zeros((0, 8))
    dt = np.diff(t)
    res = {"backend": log["backend"], "frames_released": len(t), "frames_consumed": len(got["frames"]),
           "replaced": len(log["replaced"]), "duration_s": float(t[-1] - t[0]) if len(t) > 1 else None,
           "fps_device_clock": float(1.0 / np.median(dt)) if len(dt) else None,
           "frame_dt_ms": ({"min": float(dt.min() * 1e3), "max": float(dt.max() * 1e3)} if len(dt) else None),
           "frame_gaps_over_1p5_period": int((dt > 1.5 * np.median(dt)).sum()) if len(dt) else None,
           "frame_t_strictly_increasing": bool(np.all(dt > 0)) if len(dt) else None,
           "pair_sync_us": ({"p50": float(np.nanmedian(np.abs(src[:, 3]))), "max": float(np.nanmax(np.abs(src[:, 3])))}
                            if len(src) else None),
           "pairing": log["pairing"], "imu_samples": int(len(it)),
           "imu_rate_hz": float((len(it) - 1) / (it[-1] - it[0])) if len(it) > 1 else None,
           "imu_rate_median_hz": float(1.0 / np.median(np.diff(it))) if len(it) > 1 else None,
           "imu_t_strictly_increasing": bool(np.all(np.diff(it) > 0)) if len(it) > 1 else None,
           "imu_max_gap_ms": float(np.diff(it).max() * 1e3) if len(it) > 1 else None,
           "imu_lag_at_release_ms": ({"p50": float(np.nanmedian(src[:, 7])), "max": float(np.nanmax(src[:, 7]))}
                                     if len(src) and np.isfinite(src[:, 7]).any() else None),
           "capture_to_release_ms": ({"p50": float(np.nanmedian(src[:, 6])),
                                      "p95": float(np.nanpercentile(src[:, 6], 95))}
                                     if len(src) and np.isfinite(src[:, 6]).any() else None),
           "release_to_consumer_ms_p50": (float(np.median([f[2] for f in got["frames"]]) / 1e6)
                                          if got["frames"] else None),
           "validity": log["validity"], "pre_ready_frames": log["pre_ready_frames"],
           "imu_pre_session": log["imu_pre_session"], "recording": log.get("recording"), "log": log_path}
    if out_json:
        json.dump(res, open(out_json.replace(".json", "_smoke.json"), "w"), indent=1, default=str)
    return res


def quiet_sigint():
    """Demo child processes ignore the console's Ctrl+C: the demo's main process turns it into ctl['stop'] (an orderly
    end of stream), so the engine still writes its summary and the recorder its log."""
    import signal
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def engine_entry(*args):
    """xclean.live.engine.engine_main, unchanged, after quiet_sigint()."""
    quiet_sigint()
    from xclean.live.engine import engine_main
    return engine_main(*args)


def recorder_entry(*args):
    """xclean.live.recorder.recorder_main, unchanged, after quiet_sigint()."""
    quiet_sigint()
    from xclean.live.recorder import recorder_main
    return recorder_main(*args)


def source_main(cfg, slot_name, imu_q, ctl, log_path, calib_q=None):
    """The source PROCESS (the demo's FEEDER replacement).  cfg: kind ('device' | 'replay'), mode, replay_dir, ext,
    frames, fps, mxid, calib (dict or None), ref_calib (dict or None), record_dir, rel_ring, duration_s.
    Device: opens the device, builds the calib from its EEPROM (read-only) unless one is given, sends it on calib_q
    BEFORE streaming (the engine is started with it), then streams."""
    quiet_sigint()
    backend = None
    extra = {}
    try:
        if cfg["kind"] == "replay":
            backend = ReplayBackend(cfg["replay_dir"], cfg.get("mode", "paced"), cfg.get("frames"), cfg.get("ext"))
            calib = backend.calib
        elif cfg["kind"] == "source":           # any stereo rig: a frame source of xclean/live/sources.py
            from xclean.live.sources import SourceBackend, make_source
            src = make_source(cfg["source_spec"], cfg.get("source_opts"))
            backend = SourceBackend(src, cfg.get("mode", "paced"))
            calib = backend.open(cfg.get("calib"), fps=cfg.get("fps"))
            extra = {"source": backend.describe()}
        else:
            backend = DeviceBackend(cfg["width"], cfg["height"], cfg["fps"], cfg.get("mxid"),
                                    cfg.get("imu_rate", IMU_RATE_REQ_HZ)).open()
            calib = cfg.get("calib") or backend.calib(ref=cfg.get("ref_calib"))
            if cfg.get("calib") and cfg["calib"].get("device_mxid") not in (None, backend.mxid):
                raise RuntimeError("the given calib is device %s, the attached device is %s"
                                   % (cfg["calib"].get("device_mxid"), backend.mxid))
        if calib_q is not None:
            calib_q.put(dict({"calib": calib, "mxid": getattr(backend, "mxid", None)}, **extra))
    except Exception as e:
        if calib_q is not None:
            calib_q.put({"error": "%s: %s" % (type(e).__name__, e)})
        if backend is not None and backend.kind == "device":
            backend.stop()
        raise
    if ctl.get("slot_ready") is not None:                      # the demo creates the slot after it has the calib
        while not ctl["slot_ready"].wait(0.5):
            if ctl["stop"].is_set():
                backend.stop()
                return
    W, H = int(calib["camera_width"]), int(calib["camera_height"])
    slot = FrameSlot(slot_name, W, H)
    ring = ReleaseRing(cfg["rel_ring"]) if cfg.get("rel_ring") else None
    rec = StreamRecorder(cfg["record_dir"], calib) if cfg.get("record_dir") else None
    c = dict(cfg)
    c["fps"] = float(calib.get("camera_fps") or cfg["fps"])
    try:
        run_source(backend, c, slot, imu_q, ctl, log_path, recorder=rec, rel_ring=ring)
    finally:
        slot.close()
        if ring is not None:
            ring.close()


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="OAK source: list devices (read-only) or a smoke capture without engine")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", type=float, default=None, help="seconds of stream")
    ap.add_argument("--replay", default=None, help="smoke the same path from a recording")
    ap.add_argument("--mxid", default=None)
    ap.add_argument("--record", default=None,
                    help="also record the stream here (a new or empty folder, or an earlier recording, which is "
                         "replaced)")
    ap.add_argument("--ref-calib", default=os.environ.get("XCLEAN_OAK_REF_CALIB") or None,
                    help="calib.json of the same unit (imu_noise / camera_fps / imu_rate_hz)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.list or a.smoke is None:
        print(json.dumps(list_devices(), indent=1))
    if a.smoke is not None:
        if a.record:
            import shutil
            import sys
            from xclean.io.folders import check_output_folder
            problem = check_output_folder("oak_source", "--record", a.record, True, "recording",
                                          inputs=[a.replay, a.ref_calib])
            if problem:
                print(problem, file=sys.stderr)
                return 2
            if os.path.isdir(a.record):
                shutil.rmtree(a.record)
        ref = json.load(open(a.ref_calib, encoding="utf-8")) if a.ref_calib and os.path.isfile(a.ref_calib) else None
        print(json.dumps(smoke(a.smoke, a.mxid, a.replay, a.record, ref, a.json), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
