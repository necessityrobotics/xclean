# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Stereo frame sources for `xclean live`: connect any calibrated stereo rig.

A frame source gives xclean two kinds of items, in time order:

    StereoFrame(t, left, right)      one stereo pair: both images of one moment, t in seconds
    ImuSamples(t, gyro, accel)       IMU samples on the SAME clock: t (n,) s, gyro (n, 3) rad/s, accel (n, 3) m/s^2

Built-in sources (`xclean live --source NAME`):

    oak       a Luxonis OAK-D-Lite through depthai (the default; xclean/live/oak_source.py, not this module)
    uvc-sbs   one USB (UVC) stereo camera that sends both images side by side in one frame, read with OpenCV
    uvc-pair  two USB (UVC) cameras read with OpenCV.  NOT hardware-synchronised: the two images of a pair are taken
              at different moments
    folder    a sequence folder sent through this interface (a test of the source path; `xclean live --replay`
              is the usual way to replay a recording)

Your own source (any SDK: RealSense, ZED, a machine-vision camera, a ROS bridge, ...) is a class with the methods of
StereoSource, named on the command line as MODULE:CLASS or FILE.py:CLASS:

    xclean live --source my_rig.py:MyRig --calib calib.json --out runs/rig1 --source-opt serial=1234

    from xclean.live.sources import StereoSource, StereoFrame, ImuSamples

    class MyRig(StereoSource):
        name = "my-rig"
        has_imu = True                            # it also yields ImuSamples

        def __init__(self, serial=None):          # each --source-opt KEY=VALUE arrives as a keyword argument
            self.serial = serial

        def open(self, calib):                    # calib: the --calib file as a dict, or None
            self.cam = my_sdk.open(self.serial)   # return None to use --calib, or a calib dict of your own

        def frames(self):                         # yield items in time order; return to end the stream
            while True:
                f = self.cam.next()
                yield ImuSamples(f.imu_t, f.gyro, f.accel)    # the IMU samples that arrived since the last pair
                yield StereoFrame(f.t, f.left, f.right)

        def close(self):
            self.cam.close()

The rules (docs/ANY_STEREO_LIVE.md): raw or rectified images at the size of calib.json, 8-bit grey (colour is
converted), left = the camera with the smaller x (T_stereo x < 0); one time stamp per pair; IMU samples in the IMU's
own axes, on the clock of the frames.  A source with live = True (a camera) is never made to wait: xclean takes its
frames as they come and drops a frame the engine cannot take in time.  A source with live = False (a recording) is
paced at its own time stamps, or one frame at a time with --mode lockstep.

SourceBackend wraps a source for the live source loop (oak_source.run_source), so a source feeds the live engine
through exactly the path the OAK camera and `xclean live --replay` use; the engine cannot tell them apart.
A live source is treated as the OAK device (its clock is moved to t = 0 at the first released pair, in integer
microseconds; frames that arrive while the engine warms up are dropped); a source with live = False as the replay
backend (its time stamps are used as given).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass

import numpy as np

from xclean.live.protocol import now_ns

BUILTIN_NAMES = ("oak", "uvc-sbs", "uvc-pair", "folder")
DECODE_AHEAD_START = 64             # as oak_source.ReplayBackend: items read ahead before the clock starts
DECODE_AHEAD_MAX = 256
STOP_JOIN_S = 5.0
PROBE_JOIN_S = 2.0


class SourceError(RuntimeError):
    """A source that cannot be loaded, made or opened; the message says what to fix."""


# ================================================================================================= the items
@dataclass
class StereoFrame:
    """One stereo pair.  t: seconds on the rig's clock (the clock of the IMU samples).  left / right: H x W uint8
    grey images (H x W x 3 BGR or H x W x 4 BGRA images are converted to grey).  t_right: the right image's own
    stamp when the SDK gives one (default t); a pair whose two stamps are more than half a frame period apart is
    dropped.  index: a frame number, used only by a source with live = False (default 0, 1, 2, ...)."""
    t: float
    left: np.ndarray
    right: np.ndarray
    t_right: float | None = None
    index: int | None = None


@dataclass
class ImuSamples:
    """IMU samples: t (n,) seconds, on the clock of the frames; gyro (n, 3) rad/s; accel (n, 3) m/s^2 with gravity;
    both in the IMU's own axes (calib.json T_body_cam0 says where the IMU sits)."""
    t: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray


class StereoSource:
    """The base class of a frame source (subclassing it is optional: any class with these methods works).

    Class attributes (an instance may change name, has_imu, hardware_sync and fps in __init__ or open; live must be
    a class attribute, because xclean reads it before it makes the source):
      name           a short name for the logs
      live           True: a camera; xclean never makes it wait.  False: a recording; paced, or lock-step
      has_imu        True when frames() also yields ImuSamples
      hardware_sync  False when the two images of a pair are not taken at the same moment (xclean warns)
      fps            the frame rate, used when calib.json has no camera_fps (None: --fps, else 30)"""
    name = "custom"
    live = True
    has_imu = False
    hardware_sync = True
    fps = None

    def open(self, calib):
        """Connect.  calib: the calib.json given with --calib (a dict), or None.  -> None to use it, or a calib dict
        (for a rig whose SDK holds its own calibration; it must have the fields `xclean calib` writes).  If open()
        raises, it must release what it opened itself: close() is not called then."""
        return None

    def frames(self):
        """Yield StereoFrame and ImuSamples in time order.  Return to end the stream; an exception ends it with an
        error."""
        raise NotImplementedError

    def close(self):
        """Disconnect.  Called once, after frames() has ended, from the thread that read the frames."""

    def info(self):
        """Facts for the run log (a JSON-serialisable dict)."""
        return {}


# ================================================================================================= helpers
def to_gray(img, W, H, side):
    """A source image -> C-contiguous H x W uint8 grey.  Raises SourceError with a plain message."""
    import cv2
    a = np.asarray(img)
    if a.dtype != np.uint8:
        raise SourceError("the %s image is %s; xclean needs 8-bit images (uint8)" % (side, a.dtype))
    if a.ndim == 3 and a.shape[2] == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_BGRA2GRAY)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.ndim != 2:
        raise SourceError("the %s image has shape %s; xclean needs H x W grey or H x W x 3 colour images"
                          % (side, tuple(np.asarray(img).shape)))
    if a.shape != (H, W):
        raise SourceError("the %s image is %d x %d, but calib.json says %d x %d: give images at the calibrated size, "
                          "or calibrate at the size the rig gives" % (side, a.shape[1], a.shape[0], W, H))
    return np.ascontiguousarray(a)


def imu_arrays(item):
    """ImuSamples -> (t (n,), gyro (n, 3), accel (n, 3)) float64 arrays.  Raises SourceError."""
    t = np.asarray(item.t, np.float64).reshape(-1)
    g = np.asarray(item.gyro, np.float64).reshape(-1, 3) if np.size(item.gyro) else np.zeros((0, 3))
    a = np.asarray(item.accel, np.float64).reshape(-1, 3) if np.size(item.accel) else np.zeros((0, 3))
    if not (len(t) == len(g) == len(a)):
        raise SourceError("ImuSamples: t has %d samples, gyro %d, accel %d" % (len(t), len(g), len(a)))
    if not (np.isfinite(t).all() and np.isfinite(g).all() and np.isfinite(a).all()):
        raise SourceError("ImuSamples: a value is not finite (NaN or infinity)")
    return t, g, a


def _flag(v):
    """A yes / no option from the command line (true / false / 1 / 0 / yes / no) -> bool."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
        raise SourceError("expected true or false, got %r" % v)
    return bool(v)


def parse_options(pairs):
    """['KEY=VALUE', ...] (the --source-opt values) -> dict of strings.  Every VALUE stays a string: a source
    converts its options itself (the built-in sources do)."""
    out = {}
    for kv in pairs or ():
        if "=" not in kv:
            raise SourceError("--source-opt %r: write it as KEY=VALUE" % kv)
        k, v = kv.split("=", 1)
        k = k.strip()
        if not k.isidentifier():
            raise SourceError("--source-opt %r: %r is not a valid option name" % (kv, k))
        out[k] = v
    return out


def _load_file_module(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise SourceError("the source file %s does not exist" % path)
    base = os.path.splitext(os.path.basename(path))[0]
    modname = "xclean_user_source_%s" % "".join(c if c.isalnum() else "_" for c in base)
    if modname in sys.modules and getattr(sys.modules[modname], "__file__", None) == path:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(modname, None)
        raise
    return mod


def load_source(spec):
    """A source name -> its class.  spec: uvc-sbs, uvc-pair, folder, MODULE:CLASS or FILE.py:CLASS (oak is not a
    class of this module: the OAK camera has its own backend).  Raises SourceError."""
    if spec in BUILTIN:
        return BUILTIN[spec]
    if ":" not in spec or spec.endswith(":") or spec.startswith(":"):
        raise SourceError("unknown source %r: use oak, uvc-sbs, uvc-pair, folder, or your own as MODULE:CLASS or "
                          "FILE.py:CLASS (docs/ANY_STEREO_LIVE.md)" % spec)
    where, cname = spec.rsplit(":", 1)
    if "/" in cname or "\\" in cname or cname.endswith(".py"):
        raise SourceError("the source %r names no class: write FILE.py:CLASS or MODULE:CLASS" % spec)
    try:
        if where.lower().endswith(".py") or "/" in where or "\\" in where:
            mod = _load_file_module(where)
        else:
            mod = importlib.import_module(where)
    except SourceError:
        raise
    except Exception as e:
        raise SourceError("the source %s cannot be imported (%s: %s)" % (where, type(e).__name__, e))
    obj = mod
    for part in cname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            raise SourceError("%s has no %s" % (where, cname))
    missing = [m for m in ("open", "frames", "close") if not callable(getattr(obj, m, None))]
    if missing:
        raise SourceError("%s is not a frame source: it has no %s method (docs/ANY_STEREO_LIVE.md)"
                          % (spec, ", ".join(missing)))
    return obj


def make_source(spec, options=None):
    """spec + options (keyword arguments of the class) -> a source object.  Raises SourceError."""
    cls = load_source(spec)
    try:
        return cls(**(options or {}))
    except TypeError as e:
        raise SourceError("the source %s does not take these options %s (%s)"
                          % (spec, sorted((options or {}).keys()), e))


def source_name(src):
    return str(getattr(src, "name", None) or type(src).__name__)


BASELINE_TOL = 0.01      # baseline_m against the length of T_stereo (OAK recordings store |T_stereo x| as baseline_m)


def check_calib(calib):
    """The checks of `xclean calib check` (calib_import.validate), except that baseline_m may differ from the
    length of T_stereo by up to 1 %: calib.json files written by `xclean live --record` on an OAK-D-Lite store the x
    part of T_stereo as baseline_m, and `xclean live --replay` runs them.  -> the engine's Rig.  Raises SourceError."""
    import numpy as np
    from xclean.io.calib_import import CalibError, validate
    chk = dict(calib)
    try:
        B = float(np.linalg.norm(np.asarray(chk["T_stereo"], float).reshape(3)))
        bm = abs(float(chk["baseline_m"]))
    except (KeyError, TypeError, ValueError):
        B = bm = None
    if B and bm is not None:
        if abs(bm - B) > BASELINE_TOL * B:
            raise SourceError("the calibration cannot be used: baseline_m (%r) differs from the length of T_stereo "
                              "(%.9g m) by more than 1 %% (check it with xclean calib check)" % (chk["baseline_m"], B))
        chk["baseline_m"] = B
    try:
        return validate(chk, [])
    except CalibError as e:
        raise SourceError("the calibration cannot be used: %s (check it with xclean calib check)" % e)


def finish_calib(src, given, fps=None):
    """The calib the run uses: the source's own (from open()) or the given one, checked by check_calib; camera_fps
    is filled from the source or --fps when missing.  -> (calib, notes).  Raises SourceError."""
    calib = src.open(dict(given) if given is not None else None)
    try:
        if calib is None:
            calib = given
        if calib is None:
            raise SourceError("the source %s has no calibration of its own: give --calib with the rig's calib.json "
                              "(make it with xclean calib, docs/ANY_STEREO_LIVE.md)" % source_name(src))
        calib = dict(calib)
        check_calib(calib)
    except Exception:
        try:
            src.close()                     # it was opened: close it before the error goes up
        except Exception:
            pass
        raise
    notes = []
    if not calib.get("camera_fps"):
        f = getattr(src, "fps", None) or fps or 30.0
        calib["camera_fps"] = float(f)
        notes.append("calib.json has no camera_fps: xclean uses %g Hz as the frame rate (write it with "
                     "xclean calib --fps)" % float(f))
    return calib, notes


# ================================================================================================= the backend
class SourceBackend:
    """A frame source as a backend of oak_source.run_source (see the module doc).  The source's frames() runs in a
    reader thread of the source process; every item becomes the events the OAK and replay backends send."""
    kind = "source"

    def __init__(self, source, mode="paced"):
        self.src = source
        self.name = source_name(source)
        self.live = bool(getattr(source, "live", True))
        self.mode = mode
        self.calib = None
        self.W = self.H = None
        self._stop = threading.Event()
        self._go = threading.Event()
        self.T0 = None
        self.us0 = None
        self.finished = False
        self.source_error = None
        self.read_error = None          # run_source's end-of-stream check reads this name
        self.n = {"frames": 0, "imu_samples": 0, "frames_out_of_order": 0}
        self.emit_late_ns = []
        self.threads = []
        self.closed = False
        self.close_error = None

    def open(self, given=None, fps=None):
        """Open the source and settle the calib (finish_calib).  -> the calib."""
        self.calib, self.notes = finish_calib(self.src, given, fps)
        self.W, self.H = int(self.calib["camera_width"]), int(self.calib["camera_height"])
        return self.calib

    def describe(self):
        return {"name": self.name, "live": self.live, "has_imu": bool(getattr(self.src, "has_imu", False)),
                "hardware_sync": bool(getattr(self.src, "hardware_sync", True)), "notes": list(self.notes)}

    # ---- session clock (live sources: integer microseconds, origin = the first released pair's left stamp)
    def set_origin(self, item):
        if self.live and self.us0 is None:
            self.us0 = int(item.us)

    def session_t(self, x):
        from xclean.live.oak_source import FrameItem
        if isinstance(x, FrameItem):
            return (int(x.us) - self.us0) / 1e6
        if isinstance(x, np.ndarray):
            return (x.astype(np.int64) - self.us0) / 1e6
        return (int(x) - self.us0) / 1e6

    # ---- streaming
    def start(self, events):
        self.events = events
        if self.live:
            th = threading.Thread(target=self._read_live, daemon=True, name="source-read")
            th.start()
            self.threads.append(th)
            return
        self.buf = queue.Queue(maxsize=DECODE_AHEAD_MAX)
        th = threading.Thread(target=self._read_ahead, daemon=True, name="source-read")
        th.start()
        self.threads.append(th)
        t_wait = time.time()
        while self.buf.qsize() < DECODE_AHEAD_START and th.is_alive() and time.time() - t_wait < 60:
            time.sleep(0.01)
        em = threading.Thread(target=self._emit_recorded, daemon=True, name="source-emit")
        em.start()
        self.threads.append(em)

    def go(self):
        from xclean.live.oak_source import GO_DELAY_NS
        if not self.live:
            self.T0 = now_ns() + GO_DELAY_NS
        self._go.set()

    def _put(self, q, ev):
        while not self._stop.is_set():
            try:
                q.put(ev, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def _fail(self, e):
        if self.source_error is None:
            if isinstance(e, SourceError):
                self.source_error = str(e)
            else:
                tb = traceback.extract_tb(e.__traceback__)
                where = (" (%s, line %d)" % (os.path.basename(tb[-1].filename), tb[-1].lineno)) if tb else ""
                self.source_error = "%s: %s%s" % (type(e).__name__, e, where)
            self.read_error = "the source %s stopped: %s" % (self.name, self.source_error)

    def _items(self):
        """The source's items, converted and checked: ('F', t, tr, idx, left, right) | ('I', t, g, a)."""
        inner = self.src.frames()
        try:
            yield from self._checked(inner)
        finally:
            close = getattr(inner, "close", None)
            if close is not None:
                close()

    def _checked(self, inner):
        last_t = None
        k = 0
        for it in inner:
            if self._stop.is_set():
                return
            if isinstance(it, ImuSamples) or (hasattr(it, "gyro") and hasattr(it, "accel")):
                t, g, a = imu_arrays(it)
                if len(t):
                    self.n["imu_samples"] += len(t)
                    yield ("I", t, g, a)
                continue
            if not (hasattr(it, "left") and hasattr(it, "right") and hasattr(it, "t")):
                raise SourceError("frames() yielded %r: yield StereoFrame or ImuSamples items" % type(it).__name__)
            t = float(it.t)
            tr = float(it.t_right) if getattr(it, "t_right", None) is not None else t
            if not (np.isfinite(t) and np.isfinite(tr)):
                raise SourceError("a frame time stamp is not finite")
            if last_t is not None and t <= last_t:          # frame times must increase: drop the frame (counted)
                self.n["frames_out_of_order"] += 1
                continue
            last_t = t
            l = to_gray(it.left, self.W, self.H, "left")
            r = to_gray(it.right, self.W, self.H, "right")
            idx = getattr(it, "index", None)
            yield ("F", t, tr, int(idx) if idx is not None else k, l, r)
            self.n["frames"] += 1
            k += 1

    def _events_of(self, item, due=None):
        """One converted item -> the backend events ('I', ImuBatch) or ('L', FrameItem), ('R', FrameItem)."""
        from xclean.live.oak_source import FrameItem, ImuBatch
        rx = now_ns()
        if item[0] == "I":
            _k, t, g, a = item
            if self.live:
                us = np.round(t * 1e6).astype(np.int64)
                return [("I", ImuBatch(us / 1e6, g, a, us=us, rx_ns=rx))]
            return [("I", ImuBatch(t, g, a, rx_ns=rx))]
        _k, t, tr, idx, l, r = item
        if self.live:
            ul, ur = int(round(t * 1e6)), int(round(tr * 1e6))
            return [("L", FrameItem(side="L", t=ul / 1e6, img=l, seq=idx, us=ul, rx_ns=rx)),
                    ("R", FrameItem(side="R", t=ur / 1e6, img=r, seq=idx, us=ur, rx_ns=rx))]
        return [("L", FrameItem(side="L", t=t, img=l, seq=idx, idx=idx, rx_ns=rx, due_ns=due)),
                ("R", FrameItem(side="R", t=tr, img=r, seq=idx, idx=idx, rx_ns=rx, due_ns=due))]

    def _finish_reading(self):
        try:
            self.src.close()
        except Exception as e:
            self.close_error = "%s: %s" % (type(e).__name__, e)
        self.closed = True

    def _read_live(self):
        """A camera: every item becomes events at once (the engine side drops what it cannot take in time)."""
        gen = None
        try:
            gen = self._items()
            for item in gen:
                for ev in self._events_of(item):
                    self.events.put(ev)
        except Exception as e:
            self._fail(e)
        finally:
            if gen is not None:
                gen.close()
            self._finish_reading()
            self.events.put(("EOS",))
            self.finished = True

    def _read_ahead(self):
        """A recording: items are read ahead into a buffer; the emitter releases them at their time (or lock-step)."""
        gen = None
        try:
            gen = self._items()
            for item in gen:
                if not self._put(self.buf, item):
                    break
        except Exception as e:
            self._fail(e)
        finally:
            if gen is not None:
                gen.close()
            self._finish_reading()
            self._put(self.buf, None)

    def _wait_until(self, ns):
        while True:
            d = ns - now_ns()
            if d <= 0 or self._stop.is_set():
                return
            if d > 2_000_000:
                time.sleep((d - 1_500_000) / 1e9)
            else:
                time.sleep(0)

    def _emit_recorded(self):
        self._go.wait()
        paced = self.mode != "lockstep"
        t_first = None
        while not self._stop.is_set():
            try:
                item = self.buf.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                break
            due = None
            if paced:
                t_item = float(item[1][-1]) if item[0] == "I" else float(item[1])
                if item[0] == "F" and t_first is None:
                    t_first = t_item
                if t_first is not None:
                    due = self.T0 + int(round((t_item - t_first) * 1e9))
                    self._wait_until(due)
                    if item[0] == "F":
                        self.emit_late_ns.append(now_ns() - due)
            for ev in self._events_of(item, due if item[0] == "F" else None):
                if not self._put(self.events, ev):
                    break
        self._put(self.events, ("EOS",))
        self.finished = True

    def stop(self):
        self._stop.set()
        self._go.set()                                      # a recording's emitter may still wait for the go
        if not self.threads and not self.closed:           # stopped before streaming began: close it here
            self._finish_reading()
        for th in self.threads:
            th.join(STOP_JOIN_S)
        if any(th.is_alive() for th in self.threads):
            self.close_error = (self.close_error or "") + ("the source's frames() did not return within %g s of the "
                                                           "stop" % STOP_JOIN_S)

    def info(self):
        late = np.asarray(self.emit_late_ns, float) / 1e6
        out = {"backend": "source", "source": self.name, "live": self.live,
               "has_imu": bool(getattr(self.src, "has_imu", False)),
               "hardware_sync": bool(getattr(self.src, "hardware_sync", True)), "counts": dict(self.n),
               "us0": self.us0, "closed": self.closed}
        if len(late):
            out["emit_late_ms"] = {"p50": float(np.median(late)), "p99": float(np.percentile(late, 99)),
                                   "max": float(late.max())}
        if self.source_error:
            out["error"] = self.source_error
        if self.close_error:
            out["close_error"] = self.close_error
        try:
            extra = self.src.info()
            if extra:
                out["source_info"] = json.loads(json.dumps(extra, default=str))
        except Exception as e:
            out["source_info_error"] = "%s: %s" % (type(e).__name__, e)
        return out


# ================================================================================================= built-in sources
CV_APIS = {"auto": "CAP_ANY", "any": "CAP_ANY", "dshow": "CAP_DSHOW", "msmf": "CAP_MSMF", "v4l2": "CAP_V4L2",
           "gstreamer": "CAP_GSTREAMER", "avfoundation": "CAP_AVFOUNDATION", "ffmpeg": "CAP_FFMPEG"}


def _device(d):
    s = str(d).strip()
    return int(s) if s.lstrip("-").isdigit() else s


def open_capture(device, width, height, fps=None, api="auto", fourcc=None):
    """cv2.VideoCapture of one camera at width x height (and fps).  Raises SourceError."""
    import cv2
    name = CV_APIS.get(str(api).lower())
    if name is None:
        raise SourceError("unknown api %r: use one of %s" % (api, ", ".join(sorted(CV_APIS))))
    cap = cv2.VideoCapture(_device(device), getattr(cv2, name, cv2.CAP_ANY))
    if not cap.isOpened():
        raise SourceError("OpenCV cannot open camera %r (is it connected, and not in use by another program?)"
                          % (device,))
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(fourcc)[:4].ljust(4)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)             # the newest frame, where the driver supports it
    return cap


def _first_frame(cap, device):
    for _ in range(30):                             # some cameras send a few empty frames after they open
        ok, img = cap.read()
        if ok and img is not None:
            return img
    raise SourceError("camera %r gives no image" % (device,))


class UvcSideBySide(StereoSource):
    """One USB (UVC) stereo camera that sends both images side by side in one frame (left half = left camera),
    read with OpenCV.  The frame is 2 W x H for a calib.json of W x H.  Time stamp: the host clock
    (time.perf_counter) when the frame arrives.  No IMU.
    Options: device (default 0: a camera number, a device path or a URL), fps (default: calib camera_fps),
    swap (true: the left camera is in the right half), api (auto, dshow, msmf, v4l2, gstreamer, ...),
    fourcc (for example MJPG)."""
    name = "uvc-sbs"
    live = True
    has_imu = False
    hardware_sync = True

    def __init__(self, device=0, fps=None, swap=False, api="auto", fourcc=None):
        self.device, self.fps_req, self.swap = device, fps, _flag(swap)
        self.api, self.fourcc = api, fourcc
        self.cap = None
        self.shape = None

    def open(self, calib):
        if calib is None:
            raise SourceError("--source uvc-sbs needs --calib: the calib.json of the camera (one image's size, "
                              "made with xclean calib)")
        W, H = int(calib["camera_width"]), int(calib["camera_height"])
        fps = self.fps_req or calib.get("camera_fps") or 30.0
        self.fps = float(fps)
        self.cap = open_capture(self.device, 2 * W, H, fps, self.api, self.fourcc)
        img = _first_frame(self.cap, self.device)
        self.shape = img.shape
        if img.shape[:2] != (H, 2 * W):
            self.close()
            raise SourceError("camera %r gives %d x %d frames, but --source uvc-sbs with this calib.json (%d x %d per "
                              "camera) needs %d x %d: both images side by side. Calibrate at the size the camera "
                              "gives, or pick another mode (api / fourcc options)"
                              % (self.device, img.shape[1], img.shape[0], W, H, 2 * W, H))
        self.W = W
        return None

    def frames(self):
        W = self.W
        while True:
            if not self.cap.grab():
                raise SourceError("camera %r stopped sending images" % (self.device,))
            t = time.perf_counter()
            ok, img = self.cap.retrieve()
            if not ok or img is None:
                continue
            l, r = img[:, :W], img[:, W:]
            if self.swap:
                l, r = r, l
            yield StereoFrame(t, l, r)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def info(self):
        return {"device": str(self.device), "frame_shape": None if self.shape is None else list(self.shape),
                "swap": self.swap, "api": self.api, "fourcc": self.fourcc, "fps_requested": self.fps}


class UvcPair(StereoSource):
    """Two USB (UVC) cameras read with OpenCV, one image each.  NOT hardware-synchronised: each camera takes its
    images on its own clock, so the two images of a pair can be up to one frame period apart.  The pair's stamp is
    the mean of the two arrival times (host clock); the measured gap is in the run log.  No IMU.
    Options: device (default "0,1": left camera, right camera), fps, api, fourcc (as uvc-sbs)."""
    name = "uvc-pair"
    live = True
    has_imu = False
    hardware_sync = False

    def __init__(self, device="0,1", fps=None, api="auto", fourcc=None):
        devs = device if isinstance(device, (list, tuple)) else str(device).split(",")
        if len(devs) != 2:
            raise SourceError("--source uvc-pair needs two cameras: --device LEFT,RIGHT (for example 0,1)")
        self.devices = [str(d).strip() for d in devs]
        self.fps_req, self.api, self.fourcc = fps, api, fourcc
        self.caps = []
        self.gaps = []

    def open(self, calib):
        if calib is None:
            raise SourceError("--source uvc-pair needs --calib: the calib.json of the two cameras (made with "
                              "xclean calib)")
        W, H = int(calib["camera_width"]), int(calib["camera_height"])
        fps = self.fps_req or calib.get("camera_fps") or 30.0
        self.fps = float(fps)
        for d in self.devices:
            self.caps.append(open_capture(d, W, H, fps, self.api, self.fourcc))
        for d, cap in zip(self.devices, self.caps):
            img = _first_frame(cap, d)
            if img.shape[:2] != (H, W):
                self.close()
                raise SourceError("camera %r gives %d x %d images, but calib.json says %d x %d"
                                  % (d, img.shape[1], img.shape[0], W, H))
        return None

    def frames(self):
        cl, cr = self.caps
        while True:
            okl = cl.grab()
            tl = time.perf_counter()
            okr = cr.grab()
            tr = time.perf_counter()
            if not (okl and okr):
                raise SourceError("camera %r stopped sending images" % (self.devices[0] if not okl
                                                                         else self.devices[1],))
            ok1, l = cl.retrieve()
            ok2, r = cr.retrieve()
            if not (ok1 and ok2) or l is None or r is None:
                continue
            self.gaps.append(tr - tl)
            yield StereoFrame(0.5 * (tl + tr), l, r)

    def close(self):
        for cap in self.caps:
            cap.release()
        self.caps = []

    def info(self):
        g = np.abs(np.asarray(self.gaps, float)) * 1e3
        return {"devices": self.devices, "api": self.api, "fourcc": self.fourcc, "fps_requested": self.fps,
                "hardware_sync": False,
                "arrival_gap_ms": ({"p50": float(np.median(g)), "p95": float(np.percentile(g, 95)),
                                    "max": float(g.max()), "n": int(len(g))} if len(g) else None),
                "note": "the arrival gap is a lower bound of the capture gap: the cameras are not synchronised"}


class SequenceFolderSource(StereoSource):
    """A sequence folder (docs/SEQUENCE_FORMAT.md) sent through the source interface, with the readers of
    `xclean live --replay` (oak_source.ReplayBackend): the same rows, images and IMU samples.  live = False, so it
    is paced at its recorded times, or lock-step.  It is the reference example of a source and the test of the
    source path: in lock-step it gives the poses of `xclean live --replay` of the same folder.
    Options: device or path (the folder), frames (only the first N frames), ext (png or jpg)."""
    name = "folder"
    live = False
    hardware_sync = True

    def __init__(self, device=None, path=None, frames=None, ext=None):
        from xclean.live.oak_source import ReplayBackend
        self.path = str(path or device or "")
        if not self.path:
            raise SourceError("--source folder needs the sequence folder: --device FOLDER")
        from xclean.io.folders import sequence_problem
        problem = sequence_problem(self.path, ext)
        if problem:
            raise SourceError(problem)
        rb = ReplayBackend(self.path, "lockstep", frames=int(frames) if frames else None, ext=ext)
        self.rows, self.ext, self.folder_calib = rb.rows, rb.ext, rb.calib
        self.ti, self.gi, self.ai = rb.ti, rb.gi, rb.ai
        self.has_imu = os.path.isfile(os.path.join(self.path, "imu.jsonl"))
        self.fps = rb.calib.get("camera_fps")

    def open(self, calib):
        return dict(self.folder_calib) if calib is None else None

    def frames(self):
        import cv2
        from xclean.io.imageio import imread
        ti, gi, ai = self.ti, self.gi, self.ai
        ii = 0
        for idx, tl, tr in self.rows:
            j = int(np.searchsorted(ti, tl, side="right"))          # every IMU sample at or before the frame
            if j > ii:
                yield ImuSamples(ti[ii:j].copy(), gi[ii:j].copy(), ai[ii:j].copy())
                ii = j
            pl = "%s/left/%06d.%s" % (self.path, idx, self.ext)
            pr = "%s/right/%06d.%s" % (self.path, idx, self.ext)
            l = imread(pl, cv2.IMREAD_GRAYSCALE)
            r = imread(pr, cv2.IMREAD_GRAYSCALE)
            if l is None or r is None:
                raise SourceError("cannot read %s" % (pl if l is None else pr))
            yield StereoFrame(tl, l, r, t_right=tr, index=idx)
        if ii < len(ti):                                            # the rest of the IMU file, as --replay sends it
            yield ImuSamples(ti[ii:].copy(), gi[ii:].copy(), ai[ii:].copy())

    def info(self):
        return {"path": os.path.abspath(self.path), "frames": len(self.rows), "ext": self.ext,
                "imu_samples_in_file": int(len(self.ti))}


BUILTIN = {"uvc-sbs": UvcSideBySide, "uvc-pair": UvcPair, "folder": SequenceFolderSource}


# ================================================================================================= probe (doctor)
def probe(src, given=None, seconds=3.0, fps=None, first_wait_s=10.0):
    """Open a source, read it for `seconds` (a live source: wall time; a recording: stream time) and close it.
    The source is read in a thread, so a source that sends nothing cannot hang the caller: no frame within
    first_wait_s, or no new item for first_wait_s, ends the probe with an error.
    -> dict of facts for `xclean doctor --source` (never raises for a source problem: 'error' holds it)."""
    res = {"name": source_name(src), "live": bool(getattr(src, "live", True)),
           "has_imu": bool(getattr(src, "has_imu", False)),
           "hardware_sync": bool(getattr(src, "hardware_sync", True)), "frames": 0, "error": None}
    try:
        calib, notes = finish_calib(src, given, fps)        # closes the source itself when the calib is refused
    except Exception as e:
        res["error"] = str(e) if isinstance(e, SourceError) else "%s: %s" % (type(e).__name__, e)
        return res
    res["calib"], res["notes"] = calib, notes
    W, H = int(calib["camera_width"]), int(calib["camera_height"])
    items = queue.Queue(maxsize=DECODE_AHEAD_MAX)
    halt = threading.Event()
    failed = []

    def reader():
        gen = None
        try:
            gen = src.frames()
            for it in gen:
                while not halt.is_set():
                    try:
                        items.put(it, timeout=0.05)
                        break
                    except queue.Full:
                        pass
                if halt.is_set():
                    break
        except Exception as e:
            failed.append(e)
        finally:
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    pass

    th = threading.Thread(target=reader, daemon=True, name="probe-read")
    th.start()
    t, ti, same, out_of_order = [], [], 0, 0
    t_wall0 = last_item = time.time()
    try:
        while True:
            try:
                it = items.get(timeout=0.05)
            except queue.Empty:
                now = time.time()
                if not th.is_alive() and items.empty():
                    break
                if not t and now - t_wall0 > first_wait_s:
                    raise SourceError("no stereo frame within %g s" % first_wait_s)
                if t and now - last_item > first_wait_s:
                    raise SourceError("the source sent nothing for %g s" % first_wait_s)
                continue
            last_item = time.time()
            if isinstance(it, ImuSamples) or (hasattr(it, "gyro") and hasattr(it, "accel")):
                tt, _g, _a = imu_arrays(it)
                ti.extend(tt.tolist())
            else:
                l = to_gray(it.left, W, H, "left")
                r = to_gray(it.right, W, H, "right")
                tf = float(it.t)
                if t and tf <= t[-1]:
                    out_of_order += 1
                t.append(tf)
                same += int(np.array_equal(l, r))
            if res["live"] and t and time.time() - t_wall0 >= seconds:
                break
            if not res["live"] and len(t) > 1 and t[-1] - t[0] >= seconds:
                break
    except Exception as e:
        res["error"] = str(e) if isinstance(e, SourceError) else "%s: %s" % (type(e).__name__, e)
    halt.set()
    th.join(PROBE_JOIN_S)
    if failed and res["error"] is None:
        e = failed[0]
        res["error"] = str(e) if isinstance(e, SourceError) else "%s: %s" % (type(e).__name__, e)
    if th.is_alive():                                   # frames() is stuck in the SDK: do not close under it
        res["close_error"] = "the source's frames() did not return within %g s; it was not closed" % PROBE_JOIN_S
    else:
        try:
            src.close()
        except Exception as e:
            res["close_error"] = "%s: %s" % (type(e).__name__, e)
    try:
        res["info"] = json.loads(json.dumps(src.info() or {}, default=str))
    except Exception:
        res["info"] = {}
    tf = np.asarray(t, float)
    res["frames"] = len(tf)
    res["frames_out_of_order"] = out_of_order
    res["identical_pairs"] = same
    if len(tf) > 1:
        dt = np.diff(tf)
        res["fps_measured"] = float(1.0 / np.median(dt)) if np.median(dt) > 0 else None
        res["span_s"] = float(tf[-1] - tf[0])
    ta = np.asarray(ti, float)
    res["imu_samples"] = int(len(ta))
    if len(ta) > 1:
        d = np.diff(ta)
        res["imu_rate_hz"] = float((len(ta) - 1) / (ta[-1] - ta[0])) if ta[-1] > ta[0] else None
        res["imu_nonincreasing"] = int(np.sum(d <= 0))
        res["imu_max_gap_ms"] = float(d.max() * 1e3)
    return res


# ================================================================================================= Python API
def spec_of(cls):
    """A source class -> the MODULE:CLASS (or FILE.py:CLASS, for a class of the script being run) spec that the
    source process loads it by."""
    mod = sys.modules.get(cls.__module__)
    if cls.__module__ in ("__main__", "__mp_main__"):
        f = getattr(mod, "__file__", None)
        if not f:
            raise SourceError("a source class defined interactively cannot be loaded by the source process: put it "
                              "in a .py file")
        return "%s:%s" % (os.path.abspath(f), cls.__qualname__)
    return "%s:%s" % (cls.__module__, cls.__qualname__)


def run_live(source, out, calib=None, options=None, args=()):
    """Run `xclean live` with a frame source from Python.  source: a StereoSource class or a spec string (uvc-sbs,
    uvc-pair, folder, MODULE:CLASS, FILE.py:CLASS); options: keyword arguments of the class (given to it as
    strings, as --source-opt gives them); args: more
    `xclean live` options (for example ["--duration", "60", "--record", "recordings/rig1"]).  -> the exit code.
    The source runs in a process of its own: call this under `if __name__ == "__main__":`."""
    from xclean import cli
    spec = source if isinstance(source, str) else spec_of(source)
    argv = ["live", "--source", spec, "--out", str(out)]
    if calib is not None:
        argv += ["--calib", str(calib)]
    for k, v in (options or {}).items():
        argv += ["--source-opt", "%s=%s" % (k, v)]
    return cli.main(argv + [str(x) for x in args])


# ================================================================================================= capture (calibration)
def capture_main(argv=None):
    """`xclean calib capture`: save stereo pairs from any source, for a chessboard calibration."""
    import argparse
    import cv2
    from xclean.io.imageio import imwrite
    ap = argparse.ArgumentParser(
        prog="xclean calib capture",
        description="Save stereo image pairs from a stereo rig for its calibration (docs/ANY_STEREO_LIVE.md). "
                    "Hold a chessboard in view of both cameras and move it between the saves.")
    ap.add_argument("--source", required=True,
                    help="the rig: uvc-sbs, uvc-pair, or your own source as MODULE:CLASS or FILE.py:CLASS")
    ap.add_argument("--device", default=None, help="the camera (as for xclean live --device)")
    ap.add_argument("--source-opt", action="append", default=[], metavar="KEY=VALUE",
                    help="an option for the source (repeat it for more)")
    ap.add_argument("--size", required=True, help="the size of ONE camera image, WxH (for example 1280x720)")
    ap.add_argument("--fps", type=float, default=None, help="the frame rate to ask for (default 30)")
    ap.add_argument("--out", required=True, help="a new or empty folder: the pairs go to left/ and right/")
    ap.add_argument("--count", type=int, default=40, help="how many pairs to save (default 40)")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between saves (default 2)")
    a = ap.parse_args(argv)
    try:
        W, H = (int(x) for x in a.size.lower().split("x"))
    except ValueError:
        ap.error("--size must look like 1280x720")
    if os.path.exists(a.out) and not (os.path.isdir(a.out) and not os.listdir(a.out)):
        print("xclean calib capture: error: %s exists and is not an empty folder; give a new or empty folder" % a.out,
              file=sys.stderr)
        return 2
    if a.source == "oak":
        print("xclean calib capture: error: the OAK-D-Lite holds its own calibration; xclean live reads it",
              file=sys.stderr)
        return 2
    try:
        opts = parse_options(a.source_opt)
        if a.device is not None:
            opts["device"] = a.device
        if a.fps and a.source in ("uvc-sbs", "uvc-pair"):
            opts.setdefault("fps", a.fps)
        src = make_source(a.source, opts)
    except SourceError as e:
        print("xclean calib capture: error: %s" % e, file=sys.stderr)
        return 2
    for side in ("left", "right"):
        os.makedirs(os.path.join(a.out, side), exist_ok=True)
    size_only = {"camera_width": W, "camera_height": H}
    state = {"n": 0, "next": time.time() + a.every, "errors": 0}

    def keep(t, l, r):
        if time.time() < state["next"] or state["n"] >= a.count:
            return
        for side, img in (("left", l), ("right", r)):
            if not imwrite(os.path.join(a.out, side, "%06d.png" % state["n"]), img):
                state["errors"] += 1
        state["n"] += 1
        state["next"] = time.time() + a.every
        print("xclean calib capture: saved pair %d of %d" % (state["n"], a.count), flush=True)

    try:
        calib = src.open(size_only)             # capture needs only the image size, not a calibration
        if calib is not None:
            W, H = int(calib["camera_width"]), int(calib["camera_height"])
        gen = src.frames()
        t_end = time.time() + a.every * (a.count + 2) + 30
        for it in gen:
            if isinstance(it, ImuSamples) or hasattr(it, "gyro"):
                continue
            keep(float(it.t), to_gray(it.left, W, H, "left"), to_gray(it.right, W, H, "right"))
            if state["n"] >= a.count or time.time() > t_end:
                break
        gen.close()
    except (SourceError, cv2.error) as e:
        print("xclean calib capture: error: %s" % e, file=sys.stderr)
        return 1
    finally:
        try:
            src.close()
        except Exception:
            pass
    print("xclean calib capture: %d pairs in %s%s" % (state["n"], a.out,
                                                     (" (%d images not written)" % state["errors"])
                                                     if state["errors"] else ""))
    return 0 if state["n"] and not state["errors"] else 1
