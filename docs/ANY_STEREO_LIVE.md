# Any stereo rig: live input

From version 0.3.0, `xclean live` takes its images from any calibrated stereo rig. A stereo rig is two cameras side
by side, with an optional IMU. This page tells you how to calibrate the rig, connect it, check it and run it.

The engine does not change. All sources feed the same live path as the OAK-D-Lite and `xclean live --replay`.

**What we tested.** We tested live runs only on a Luxonis OAK-D-Lite. We tested the other sources with recordings and
with simulated cameras: a recording sent through the source path gives the same pose for every frame as
`xclean replay` and `xclean live --replay` (lock-step). We did not run a USB stereo camera, two USB cameras or an SDK
camera (RealSense, ZED, others) live. Reports are welcome: open an issue with the output of `xclean doctor --source`.

New to xclean? Do the [QUICKSTART](../QUICKSTART.md) first. Your own recordings (not live) are in
[OWN_DATA.md](OWN_DATA.md).

## The sources

You select the source with `--source`. Each source has the same options on `xclean live`, `xclean doctor` and
`xclean calib capture`.

| `--source` | the rig | `--device` | time stamps | IMU | the two images |
|---|---|---|---|---|---|
| `oak` (default) | Luxonis OAK-D-Lite (depthai) | not used (`--mxid` picks a camera) | the camera clock | yes | hardware-synchronised |
| `uvc-sbs` | one USB (UVC) stereo camera that sends both images side by side in one frame | camera number (`0`), device path or URL | host clock at arrival | no | one frame (see the caution below) |
| `uvc-pair` | two USB (UVC) cameras | two camera numbers, left first (`0,1`) | host clock at arrival | no | **not synchronised** |
| `folder` | a sequence folder (a recording) | the folder | the recording's own | if it has `imu.jsonl` | as recorded |
| `FILE.py:CLASS` or `MODULE:CLASS` | your own source for any SDK | passed as the option `device` | yours | optional | yours |

`--source-opt KEY=VALUE` gives an option to the source. Repeat it for more options. The value arrives as a
string (`serial=1234` gives the string `"1234"`). The built-in sources convert their options themselves.

| source | options |
|---|---|
| `uvc-sbs` | `swap=true` (the left camera is in the right half), `fps=30`, `api=dshow` / `msmf` / `v4l2` / `gstreamer` (OpenCV's camera back end), `fourcc=MJPG` (the pixel format) |
| `uvc-pair` | `fps`, `api`, `fourcc` (as `uvc-sbs`) |
| `folder` | `frames=N` (only the first N frames; `--frames N` does the same), `ext=png` or `jpg` |
| your own | the keyword arguments of your class |

> **WARNING:** `uvc-pair` cameras are not synchronised. Each camera takes its images on its own clock, so the two
> images of a pair can be up to one frame period apart. When the rig moves, the depth and the poses are then wrong.
> Use `uvc-pair` only for tests with a still or slow rig. For real use, take a stereo camera with a shared trigger.

> **CAUTION:** Many side-by-side USB cameras take both images at the same moment, but some do not. Read the data
> sheet of your camera. If it does not say "synchronised", treat it as `uvc-pair`.

## Requirements of the rig

- Two cameras side by side, looking the same way. The left camera is the first camera of the calibration.
- One image size for both cameras.
- Pinhole lenses with OpenCV's distortion model. xclean does not support fisheye models.
- Global-shutter cameras are best. xclean has no rolling-shutter model.
- With an IMU: the IMU and the cameras on one clock (see [Your own source](#your-own-source)).

The same rules apply to recordings: see [OWN_DATA.md](OWN_DATA.md), "Camera and image requirements".

## Step 1: calibrate

xclean needs a `calib.json` for the rig. `xclean calib` writes it from a Kalibr or an OpenCV calibration. This step
uses OpenCV and a chessboard. For a rig with an IMU, use Kalibr's camera-IMU calibration instead
([OWN_DATA.md](OWN_DATA.md), "From Kalibr").

### 1a. Capture image pairs

1. Print a chessboard. Glue it to a flat, stiff board.
2. Measure the side of one square, in metres. Use a ruler, not the print setting.
3. Count the inner corners: a board of 10 x 7 squares has 9 x 6 inner corners.
4. Start the capture. Give the size of **one** camera image:

```
xclean calib capture --source uvc-sbs --device 0 --size 1280x720 --out calib_pairs
```

5. Hold the board so that both cameras see all of it. Move it to a new position between the saves. The command
   saves one pair every 2 seconds (`--every`), 40 pairs in all (`--count`).
6. Cover the whole image: the centre, the edges and the corners. Tilt the board up to about 45 degrees. Hold the
   board still at each save.

The pairs go to `calib_pairs\left\` and `calib_pairs\right\`.

### 1b. Calibrate with OpenCV

Save this script as `stereo_chessboard.py`. It finds the corners, calibrates each camera, then the pair, and writes
`stereo.yml` for `xclean calib opencv`.

```python
"""Stereo calibration from chessboard image pairs, for `xclean calib opencv`.

usage: python stereo_chessboard.py PAIRS_DIR COLS ROWS SQUARE_M OUT_YML

PAIRS_DIR  the folder of xclean calib capture: left/000000.png, right/000000.png, ...
COLS ROWS  the inner corners of the chessboard (a board of 10 x 7 squares has 9 x 6 inner corners)
SQUARE_M   the side of one square in metres
OUT_YML    the file to write (OpenCV FileStorage: K1, D1, K2, D2, R, T, image_width, image_height)
"""
import glob
import os
import sys

import cv2
import numpy as np

pairs_dir, cols, rows, square, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
board = np.zeros((rows * cols, 3), np.float32)
board[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
obj, pts_l, pts_r, size = [], [], [], None
for pl in sorted(glob.glob(os.path.join(pairs_dir, "left", "*.png"))):
    pr = os.path.join(pairs_dir, "right", os.path.basename(pl))
    left = cv2.imread(pl, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(pr, cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        continue
    size = (left.shape[1], left.shape[0])
    ok_l, c_l = cv2.findChessboardCorners(left, (cols, rows))
    ok_r, c_r = cv2.findChessboardCorners(right, (cols, rows))
    if not (ok_l and ok_r):
        print("skipped %s: the board is not in both images" % os.path.basename(pl))
        continue
    obj.append(board)
    pts_l.append(cv2.cornerSubPix(left, c_l, (5, 5), (-1, -1), crit))
    pts_r.append(cv2.cornerSubPix(right, c_r, (5, 5), (-1, -1), crit))
print("%d usable pairs" % len(obj))
if len(obj) < 10:
    sys.exit("too few usable pairs: capture more, with the whole board in both images")
e_l, K1, D1, _, _ = cv2.calibrateCamera(obj, pts_l, size, None, None)
e_r, K2, D2, _, _ = cv2.calibrateCamera(obj, pts_r, size, None, None)
e_s, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(obj, pts_l, pts_r, K1, D1, K2, D2, size,
                                                     flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)
print("RMS reprojection error: left %.3f px, right %.3f px, pair %.3f px" % (e_l, e_r, e_s))
fs = cv2.FileStorage(out, cv2.FILE_STORAGE_WRITE)
for name, value in (("K1", K1), ("D1", D1), ("K2", K2), ("D2", D2), ("R", R), ("T", T)):
    fs.write(name, value)
fs.write("image_width", size[0])
fs.write("image_height", size[1])
fs.release()
print("baseline %.4f m; wrote %s" % (float(np.linalg.norm(T)), out))
```

Run it (example: 9 x 6 inner corners, 25 mm squares):

```
python stereo_chessboard.py calib_pairs 9 6 0.025 stereo.yml
```

- A good calibration has an RMS reprojection error below about 0.5 px.
- The baseline must be the distance between the two lenses. Measure it with a ruler to check.
- The square size sets the unit. In metres, `T` is in metres and `xclean calib opencv` needs no `--units`.

### 1c. Write calib.json

```
xclean calib opencv stereo.yml --fps 30 --out my_rig.json
```

```
xclean calib check my_rig.json
```

- `--fps` writes `camera_fps`. Give the rate the camera really gives (`xclean doctor --source` measures it).
- `xclean calib` refuses a pair where the second camera lies to the left of the first. Swap the cameras: for
  `uvc-sbs` add `--source-opt swap=true`, for `uvc-pair` swap the two numbers of `--device`. Then capture and
  calibrate again.

## Step 2: connect

### A USB stereo camera (uvc-sbs)

1. Connect the camera. Close other programs that use it.
2. Find its number. The first camera is usually `0`. A laptop camera can take `0`; then try `1`.
3. Use `calib.json` of **one** image: the camera sends a frame of twice that width.

### Two USB cameras (uvc-pair)

1. Connect both cameras. Fix them to one rigid bar.
2. Give the left camera first: `--device 0,1`.
3. Read the WARNING in [The sources](#the-sources).

### Your own source

Use your own source for an SDK camera (Intel RealSense, Stereolabs ZED, a machine-vision camera, a ROS bridge, others).
Write a Python class with three methods: `open`, `frames` and `close`. This template shows where the SDK calls go:

```python
from xclean.live.sources import ImuSamples, StereoFrame, StereoSource


class MyRig(StereoSource):
    name = "my-rig"          # a name for the logs
    live = True              # a camera: xclean never makes it wait
    has_imu = True           # frames() also yields ImuSamples
    hardware_sync = True     # the two images of a pair are taken at the same moment

    def __init__(self, serial=None):        # each --source-opt KEY=VALUE arrives here as a keyword argument
        self.serial = serial                # a string: convert it here if the SDK needs a number

    def open(self, calib):                  # calib: the --calib file as a dict, or None
        self.cam = my_sdk.open(self.serial)  # connect (your SDK)
        return None                         # None: use --calib. Or return a calib dict of your own.

    def frames(self):                       # yield items in time order; return to end the stream
        while True:
            f = self.cam.wait_for_frames()  # your SDK
            yield ImuSamples(f.imu_t, f.gyro, f.accel)   # every IMU sample that arrived since the last pair
            yield StereoFrame(f.t, f.left, f.right)       # one pair: the two images and one time stamp

    def close(self):
        self.cam.close()
```

Save it as `my_rig.py`. Then use `--source my_rig.py:MyRig`. A class in an installed package works too:
`--source my_package.my_module:MyRig`.

The rules for a source:

| item | rule |
|---|---|
| `StereoFrame.t` | seconds. One time stamp for the pair. It must increase from pair to pair. xclean drops a pair whose time does not increase. |
| `StereoFrame.left`, `.right` | NumPy arrays at the size of `calib.json`, 8 bit (`uint8`). Grey (H x W) or colour (H x W x 3, BGR); xclean makes colour images grey. Raw images: xclean rectifies them with `calib.json`. |
| `StereoFrame.t_right` | optional: the right image's own time stamp. xclean drops a pair whose two stamps are more than half a frame period apart. |
| `ImuSamples.t` | seconds, **on the clock of the frames**. Samples in time order, no repeated time. |
| `ImuSamples.gyro` | n x 3, rad/s, in the IMU's own axes |
| `ImuSamples.accel` | n x 3, m/s^2 with gravity (a still IMU reads about 9.81 m/s^2), in the IMU's own axes |
| order | yield the IMU samples up to a pair's time before that pair |
| `live` | `True` for a camera. `False` for a recording: xclean then paces it at its own times, or runs it one frame at a time with `--mode lockstep`. Set it on the class, not in `__init__`. |
| `open` | if `open` raises an error, it must release what it opened: xclean does not call `close` then |
| errors | an error in `frames` ends the run with a message that names the source and the line |

- xclean moves the time of a live source to zero at the first pair it uses. Any clock works: the SDK's device clock,
  or `time.perf_counter()`.
- The host clock at arrival includes the transfer delay. Without an IMU, this does not matter. With an IMU, use the
  SDK's hardware time stamps for both the images and the IMU.
- The IMU is used only when `calib.json` has an `imu_noise` block and the start-up check of the engine accepts the
  data. See [OWN_DATA.md](OWN_DATA.md), "IMU noise values".
- Your source runs in a separate process. xclean loads the class by its name, so the class must be in a `.py` file
  or an installed module.

> **CAUTION:** We have not tested an SDK source (RealSense, ZED, others) with real hardware. The template above shows
> the interface, not tested SDK code.

## Step 3: check

`xclean doctor --source` opens the source, reads it for 3 seconds (`--seconds`) and closes it. It writes nothing.

```
xclean doctor --source uvc-sbs --device 0 --calib my_rig.json
```

It checks:

- that the source opens and `calib.json` is valid;
- the image size against `calib.json`;
- the frame rate against `camera_fps` (WARN above 5 % difference);
- that the frame times increase;
- that the left and right images are different;
- the IMU samples: their rate against `imu_rate_hz`, their order, and the `imu_noise` block;
- the synchronisation of the pair (WARN for `uvc-pair`, with the measured arrival gap).

Each line says `OK`, `WARN` or `FAIL`, with a fix for `WARN` and `FAIL`. With `--source`, `doctor` checks this source
instead of an OAK camera. The machine checks (GPU, CUDA, packages) run as before.

## Step 4: run

```
xclean live --source uvc-sbs --device 0 --calib my_rig.json --out runs\rig1 --record recordings\rig1 --duration 60
```

- `--duration 60` stops after 60 seconds. Without it, press `q` or Esc in the view window, or Ctrl+C.
- `--record` saves the stream as a sequence folder. Run it again later, frame by frame:

```
xclean live --replay recordings\rig1 --out runs\rig1_again --mode lockstep
```

- At start, xclean prints the source, whether it has IMU samples, and a warning for a pair that is not synchronised.
- The run folder gets the same files as an OAK run: `trajectory_live.csv` (the pose of every frame),
  `demo_summary.json`, `demo.json` (with a `source` block) and the run logs.
- `--fps` asks a `uvc-sbs` or `uvc-pair` camera for a frame rate. Without it, xclean asks for `camera_fps` of
  `calib.json`. Your own source sets its rate itself.

Your own source:

```
xclean live --source my_rig.py:MyRig --calib my_rig.json --source-opt serial=1234 --out runs\rig1
```

From Python (the source runs in its own process, so call it under `if __name__ == "__main__":`):

```python
from xclean.live.sources import run_live
from my_rig import MyRig

if __name__ == "__main__":
    run_live(MyRig, "runs/rig1", calib="my_rig.json", options={"serial": "1234"}, args=["--duration", "60"])
```

## Test the path without a camera

The `folder` source sends a recording through the source path. In lock-step it gives the same poses as
`xclean live --replay` of the same recording:

```
xclean live --source folder --device recordings\live1 --out runs\path_test --mode lockstep
```

`xclean doctor --source folder --device recordings\live1` checks a recording the same way as a camera.

## Known limits

- Live tests by us: the OAK-D-Lite only. The other sources: recordings and simulated cameras only.
- The `uvc-sbs` and `uvc-pair` time stamps are the host's arrival times. They include the USB transfer delay.
- `uvc-pair` is not synchronised (see the WARNING above).
- The built-in USB sources have no IMU. For a rig with an IMU, write your own source with the SDK's time stamps.
- No rolling-shutter model, no fisheye model.
- Auto exposure makes tracking harder. If you can, fix the exposure and use the same exposure on both cameras.
