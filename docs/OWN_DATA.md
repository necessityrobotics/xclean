# Your own camera and data

xclean runs on any recording that is laid out as a **sequence folder**. This page says how to make one from your own
stereo camera, with or without an IMU. The exact format is in [SEQUENCE_FORMAT.md](SEQUENCE_FORMAT.md). New to
xclean? Start with the [QUICKSTART](../QUICKSTART.md).

There are two routes:

- **Luxonis OAK-D-Lite:** `xclean live --record` writes a ready folder, with the calibration read from the camera.
  See [OAK-D-Lite](#oak-d-lite).
- **Any other stereo camera:** calibrate it (Kalibr or OpenCV), let `xclean calib` write `calib.json`, and write
  the images, time stamps and IMU samples into the folder yourself. A small script below does this.

Then check the folder with `xclean audit` and run it with `xclean replay`.

## The folder

```
my_sequence/
  calib.json        camera and IMU calibration (xclean calib writes it)
  timestamps.csv    one line per stereo frame
  left/000000.png   left image of frame 0 (six-digit frame number; .png or .jpg)
  right/000000.png  right image of frame 0
  imu.jsonl         IMU samples (leave it out for a camera without IMU)
  gt.csv            ground truth (optional)
```

`timestamps.csv` (the first line is a header):

```
frame_id,timestamp_s,left_t,right_t
0,0.000000000,0.000000000,0.000000000
1,0.049999872,0.049999872,0.049999872
```

xclean reads the first two columns: the frame number and the frame time in seconds. The other columns are not used.

`imu.jsonl`, one sample per line: time `t` in seconds, gyroscope `gx gy gz` in rad/s, accelerometer `ax ay az` in
m/s^2:

```
{"t": 0.0, "gx": -0.0991, "gy": 0.1403, "gz": 0.0293, "ax": 8.0333, "ay": -0.4086, "az": -2.4026}
```

## Camera and image requirements

What the code expects:

- **Two cameras side by side**, looking the same way (at most 30 degrees apart), the second camera to the right of
  the first. `xclean calib` refuses a vertical pair or a swapped pair.
- **One image size** for both cameras.
- **Pinhole lenses** with OpenCV's distortion model (4, 5, 8, 12 or 14 coefficients). Kalibr's `pinhole-radtan` is
  accepted. Fisheye (equidistant) models are not supported.
- **Raw images.** xclean rectifies them itself from `calib.json`. Images that are already rectified also work: give
  zero distortion, the rectified camera matrices, `R_stereo` = identity and `T_stereo` = `[-baseline, 0, 0]`.
- **Grayscale, 8 bit, PNG or JPG.** Colour images are converted to gray when read. JPG is lossy; PNG keeps every
  pixel.
- **One time per stereo pair.** Both images must be taken at the same moment (a hardware-triggered pair). xclean has
  no rolling-shutter model: each image is treated as taken at one instant.

What we have tested (one sequence folder of each kind; sizes can differ within a dataset):

| camera | image size | frame rate | IMU rate |
|---|---|---|---|
| EuRoC MAV | 752 x 480 | 20 Hz | 200 Hz |
| KITTI odometry | 1241 x 376 | 10 Hz | no IMU |
| 4Seasons | 800 x 400 | 30 Hz | 2000 Hz |
| MADMAX | 1032 x 772 | 14 Hz | 100 Hz |
| Rosario v1 | 672 x 376 | 15 Hz | 142 Hz |
| Rosario v2 | 1280 x 720 | 15 Hz | 200 Hz |
| OAK-D-Lite (ours) | 640 x 400 | 30 Hz | 198 Hz |

## Time stamps

- **One clock** for both cameras and the IMU. The frame times in `timestamps.csv` and the IMU times in `imu.jsonl`
  are seconds on that clock.
- Put time zero at the first frame (our converters do). Store the absolute time of that zero, in **nanoseconds**, as
  `t0_ns` in `calib.json`. Ground truth uses absolute nanoseconds, so scoring needs `t0_ns`.
- Frame times must increase. IMU samples must be in time order with no repeated time.
- A known time offset between camera and IMU: add it to the camera times when you write `timestamps.csv`
  (t_imu = t_cam + offset, Kalibr's `timeshift_cam_imu`). xclean does not apply an offset itself. `xclean calib`
  stores Kalibr's value as `td_s` and prints a warning.

## The IMU

- Samples in the **IMU's own axes**: gyroscope in rad/s, accelerometer in m/s^2 **including gravity** (a still IMU
  reads about 9.81 m/s^2).
- `calib.json` holds where the IMU sits: `T_body_cam0` is the pose of the left camera in the IMU frame (a point in
  camera coordinates times `T_body_cam0` gives IMU coordinates). This is the **inverse** of Kalibr's `T_cam_imu`.
  `xclean calib kalibr` does the inversion for you.
- `imu_rate_hz`: the IMU rate. Give the rate you measure, not the nominal one: `xclean audit` flags a difference
  above 0.5 %.

### IMU noise values

`calib.json` needs an `imu_noise` block with four continuous-time noise densities:

| key | unit |
|---|---|
| `gyroscope_noise_density` | rad/s/sqrt(Hz) |
| `gyroscope_random_walk` | rad/s^2/sqrt(Hz) |
| `accelerometer_noise_density` | m/s^2/sqrt(Hz) |
| `accelerometer_random_walk` | m/s^3/sqrt(Hz) |

These are the values Kalibr's `imu.yaml` holds, in the same units. Where they come from:

- Preferred: an **Allan variance** analysis of a long recording of the IMU lying still (hours). Any Allan variance tool
  works; write the result in Kalibr's `imu.yaml` format and give it to `xclean calib` with `--imu imu.yaml`. xclean
  does not include an analysis tool.
- Otherwise: the IMU's data sheet. Example: EuRoC's `mav0/imu0/sensor.yaml` lists the values of its ADIS16448.

`xclean calib` warns when a value is far outside the usual range (a sign of wrong units).

**Without `imu_noise`, xclean does not use the IMU.** The live engine needs all four values for the IMU factor and
for the gyroscope integration; without them both stay off. In our test (300 frames of EuRoC MH_01) the poses were
identical to a run without `imu.jsonl`. `xclean audit` flags a folder that has `imu.jsonl` but no `imu_noise`.

## Getting calib.json

### From Kalibr

Use the `camchain-imucam.yaml` of Kalibr's camera-IMU calibration, and Kalibr's `imu.yaml` for the noise values:

```
xclean calib kalibr camchain-imucam.yaml --imu imu.yaml --fps 30 --out C:\data\mycam\seq1\calib.json
```

- A plain `camchain.yaml` (cameras only) gives a calib.json without IMU.
- `--left` / `--right` pick other camera names than `cam0` / `cam1`.
- `--fps` writes `camera_fps`. Give it: the live engine uses it as the frame period.

### From OpenCV

Save the results of `cv2.stereoCalibrate` in an OpenCV FileStorage file (`.yml`, `.yaml` or `.xml`) with `K1`,
`D1`, `K2`, `D2`, `R`, `T` and `image_width` / `image_height`. `R` and `T` map points from the left camera to the
right one, as `cv2.stereoCalibrate` returns them. `T` is in the unit of your calibration target; say which:

```
xclean calib opencv stereo.yml --units mm --fps 30 --out C:\data\mycam\seq1\calib.json
```

Without `--units` the unit is metres. A baseline that only makes sense in millimetres stops with a message, for
example `the stereo baseline is 110.1 'metres': T looks like it is in millimetres (use --units mm ...)`.
An OpenCV calibration has no IMU unless the file also holds `T_cam_imu` (4x4, IMU to left camera, metres) and you
give `--imu imu.yaml`.

### Check it

```
xclean calib check C:\data\mycam\seq1\calib.json
```

It prints the image size, the baseline, the rectified focal length and whether the IMU is used, or a plain message
about what is wrong. `xclean calib` does not write `t0_ns` and `num_frames`; they belong to a recording (the script
below adds them).

## Writing the folder: an example script

Many recorders save each image with a time stamp in nanoseconds and the IMU as a CSV file. This script turns such a
recording into a sequence folder. Edit it to match your files.

```python
"""Build an xclean sequence folder from time-stamped stereo images and an IMU log.

usage: python make_sequence.py LEFT_CSV LEFT_DIR RIGHT_CSV RIGHT_DIR IMU_CSV CALIB_JSON OUT_DIR

LEFT_CSV, RIGHT_CSV  one line per image: t_ns,filename     (the image is LEFT_DIR/filename)
IMU_CSV              one line per sample: t_ns,gx,gy,gz,ax,ay,az   (rad/s and m/s^2, IMU frame; or leave it out: "-")
CALIB_JSON           the calib.json made by xclean calib
All times in integer nanoseconds on ONE clock. Lines starting with '#' are skipped.
"""
import bisect
import csv
import json
import os
import shutil
import sys


def rows(path):
    with open(path, newline="") as f:
        return [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]


left_csv, left_dir, right_csv, right_dir, imu_csv, calib_json, out = sys.argv[1:8]
left = sorted((int(r[0]), r[1].strip()) for r in rows(left_csv))
right = sorted((int(r[0]), r[1].strip()) for r in rows(right_csv))
rt = [t for t, _ in right]
period = (left[-1][0] - left[0][0]) / max(len(left) - 1, 1)

pairs = []                                       # a left image and the right image closest in time
for t, name in left:
    k = bisect.bisect_left(rt, t)
    best = min((j for j in (k - 1, k) if 0 <= j < len(rt)), key=lambda j: abs(rt[j] - t))
    if abs(rt[best] - t) <= period / 2:          # same trigger: within half a frame period
        pairs.append((t, name, right[best][1]))

os.makedirs(os.path.join(out, "left"))           # fails if OUT_DIR/left exists: start from an empty folder
os.makedirs(os.path.join(out, "right"))
t0 = pairs[0][0]                                 # time zero = the first frame
with open(os.path.join(out, "timestamps.csv"), "w", newline="\n") as f:
    f.write("frame_id,timestamp_s,left_t,right_t\n")
    for i, (t, ln, rn) in enumerate(pairs):
        ext = os.path.splitext(ln)[1].lower()    # .png or .jpg, the same for every image
        shutil.copyfile(os.path.join(left_dir, ln), os.path.join(out, "left", "%06d%s" % (i, ext)))
        shutil.copyfile(os.path.join(right_dir, rn), os.path.join(out, "right", "%06d%s" % (i, ext)))
        s = (t - t0) / 1e9
        f.write("%d,%.9f,%.9f,%.9f\n" % (i, s, s, s))

if imu_csv != "-":
    with open(os.path.join(out, "imu.jsonl"), "w", newline="\n") as f:
        for r in rows(imu_csv):
            g = [float(x) for x in r[1:7]]
            f.write(json.dumps({"t": (int(r[0]) - t0) / 1e9, "gx": g[0], "gy": g[1], "gz": g[2],
                                "ax": g[3], "ay": g[4], "az": g[5]}) + "\n")

calib = json.load(open(calib_json))
calib["t0_ns"] = t0                              # absolute time of frame 0 (for ground truth)
calib["num_frames"] = len(pairs)
json.dump(calib, open(os.path.join(out, "calib.json"), "w"), indent=1)
print("%d stereo frames (%d left images without a partner)" % (len(pairs), len(left) - len(pairs)))
```

We tested it on the raw files of EuRoC MH_01 (`mav0/cam0/data.csv`, `mav0/cam1/data.csv`, `mav0/imu0/data.csv`
have exactly these columns) with a calib.json from `xclean calib kalibr`: 3682 frames, `xclean audit` reported no
problem, and a 300-frame run had a pose for every frame.

## No IMU (stereo only)

- Leave out `imu.jsonl`.
- In `calib.json`, set `imu_rate_hz` to 0 (or leave it out) and leave out `imu_noise`. `xclean calib` without
  `--imu` writes neither. A non-zero `imu_rate_hz` without `imu.jsonl` makes the engine wait for IMU samples that
  never come: in our test a 300-frame lock-step run took 650 s instead of 44 s. `xclean audit` flags this, and
  `xclean replay` prints a warning at start.
- `T_body_cam0` can stay the identity (scoring reads it).

KITTI odometry, part of our evaluation, has no IMU and runs this way.

## Ground truth (optional)

`gt.csv` in the folder, or any file passed with `--gt`. One line per pose: `t_ns,x,y,z,qw,qx,qy,qz`, time in absolute
nanoseconds on the clock of `t0_ns`, position in metres, quaternion with **w first**. Lines starting with `#` are
skipped.

The scorer also needs to know which point the ground truth describes. It picks this from `calib.json` `"source"`:

- `"EuRoC MAV"`: the ground truth is the pose of the IMU (body) frame.
- `"KITTI odometry"`: the ground truth is the pose of the left camera after rectification.

Folders written by `xclean convert` already carry the right `source` for their dataset (EuRoC MAV, KITTI odometry,
4Seasons, MADMAX, Rosario v1 or Rosario v2). For any other `source` the scorer treats the ground truth as the IMU pose
but leaves out the small rotation that rectification adds to the left camera, and marks the score
`GT_FRAME_UNVERIFIED`. If your ground truth matches one of the two cases above, set `source` to that name. For an
IMU-pose ground truth this changes only the rotation part of the 2-second motion test, not the ATE: in our test on
EuRoC MH_01 both gave the same ATE.

Without ground truth, pass `--family oak` to `xclean replay` (the score then says `NO_GT`). Otherwise `--score` stops
with an error when the ground-truth file is missing.

## Check the folder: xclean audit

```
xclean audit C:\data\mycam\seq1
```

One line per folder: `ok` or `FLAG`, then the frame count and rate, the IMU sample count and rate, and the
ground-truth coverage. Each flag is listed below it. The exit code is 1 if any folder is flagged.

A folder that you name on the command line but that does not exist, or that has no `calib.json` or no
`timestamps.csv`, is flagged as not a sequence folder and counted among the flagged folders.

| message | what it means | what to do |
|---|---|---|
| `left/ has N images vs M csv rows` | images and `timestamps.csv` lines disagree | write one image pair per line |
| `calib num_frames N != M` | `calib.json` `num_frames` is not the number of lines | fix or remove `num_frames` |
| `frame times non-monotonic` | a frame time does not increase | check the time stamps |
| `fps declared X vs measured Y` | `camera_fps` is more than 2 % off the frame times | set `camera_fps` to the real rate |
| `imu dt==0` / `imu dt<0` | repeated or backward IMU times | remove duplicates, sort by time |
| `imu gaps >100ms` | the IMU stream has holes; those intervals are dropped | check the recorder |
| `imu rate declared X vs 1/median_dt Y Hz` | `imu_rate_hz` is more than 0.5 % off | set it to the measured rate |
| `accel x rails at ...` | the accelerometer clips at the end of its measuring range: an axis repeats its largest value, and that value is at least 1.8 g (17.65 m/s^2) | use a larger accelerometer range, or move more gently |
| `quietest-window \|a\| ... vs g` | the stillest second does not read 9.81 m/s^2 within 5 % | units (g instead of m/s^2?), or gravity missing |
| `calib has NO imu_noise block` | no noise values: the IMU is not used (see above) | add `imu_noise` |
| `calib imu_rate_hz X but no imu.jsonl` | an IMU rate without IMU data: replay waits for IMU on every frame | set `imu_rate_hz` to 0 (stereo only) |
| `gt non-monotonic` | ground-truth times do not increase | fix the file |
| `gt starts / ends ... s` | ground truth starts late or ends early | informational |
| `gt covers only N% of the run` | the score covers that part only | informational |

`xclean replay` and `xclean live --replay` check only that the folder is a sequence folder (`calib.json`,
`timestamps.csv`, `left/` and `right/`, and a first frame whose images can be read at the size in `calib.json`).
If it is not, they stop with a message before they start. They do not run the checks above, so run the audit first.
The offline runner `xclean run` refuses a folder with frame times out of order, repeated or backward IMU times, a
wrong `num_frames`, or a `gt.csv` that is out of order or covers less than half of the run.

## Run it

```
xclean replay C:\data\mycam\seq1 --out runs\seq1 --mode lockstep --family oak --score
```

Leave out `--family oak` and add `--gt <file>` if you have ground truth. `--out` must be a new or empty folder.
`--overwrite` replaces it only when it holds an earlier `xclean replay` run and nothing else, and never a folder that
holds or lies inside your data.

## Common mistakes

- **Stereo extrinsic in the wrong direction.** `R_stereo` / `T_stereo` map a point from the left camera to the right
  camera, so `T_stereo` x is negative (about minus the baseline). `xclean calib` stops with "the right camera lies to
  the LEFT of the left camera" when it is the other way round.
- **Camera-IMU extrinsic inverted.** `T_body_cam0` is camera to IMU. Kalibr's `T_cam_imu` is IMU to camera. Use
  `xclean calib kalibr`, which inverts it.
- **Baseline in millimetres.** Give `--units mm` to `xclean calib opencv`.
- **IMU units.** Gyroscope in degrees per second instead of rad/s, or accelerometer in g instead of m/s^2. The audit
  catches the accelerometer case (the gravity check), not the gyroscope case.
- **Nominal instead of measured rates.** Declare the IMU and camera rates you measure.
- **Unsynchronised stereo.** Two cameras without a shared trigger take their images at different moments; xclean
  assumes one moment per pair.
- **Rolling shutter.** xclean has no rolling-shutter model. Global-shutter cameras fit its model.
- **Auto exposure.** Sudden brightness changes and long exposures (motion blur) make tracking harder. If you can,
  fix the exposure or limit it, and use the same exposure on both cameras.
- **Different clocks.** Camera and IMU time stamps from two clocks that are not synchronised.

## OAK-D-Lite

`xclean live` reads the calibration from the camera itself (the factory calibration; nothing is written to the
camera) and can record the session as a sequence folder:

```
xclean live --out runs\oak1 --record recordings\oak1 --duration 60
```

In our tests (one OAK-D-Lite on a tripod, 60 s runs) no frame was dropped, and `xclean replay` of the recorded folder
gave the same poses as the live run, frame for frame. `xclean audit` flags a recording made without IMU noise values
(see below).

The camera stores the lens calibration and the camera-IMU extrinsic, but **not** the IMU noise values, the frame rate
or the IMU rate. Without them xclean does not use the IMU (it prints a note saying so at start). Set them up once for
your camera:

1. Make a short recording (the command above). Copy its `calib.json` to a file of its own, for example
   `my_oak.json`.
2. In `my_oak.json`, add the key `camera_fps` with the value of `measured_camera_fps`, and the key `imu_rate_hz`
   with the value of `measured_imu_rate_hz`. The recorded file has the two measured values, but not these two keys.
3. Add an `imu_noise` block (see [IMU noise values](#imu-noise-values)).
4. Pass the file on every live run:

```
xclean live --out runs\oak2 --record recordings\oak2 --duration 60 --ref-calib my_oak.json
```

xclean then takes `camera_fps`, `imu_rate_hz` and `imu_noise` from `my_oak.json` and everything else from the camera,
and says at start that it will use the IMU. The file must belong to the same camera (`device_mxid`). You can also
set the environment variable `XCLEAN_OAK_REF_CALIB` to the file instead of passing `--ref-calib`.

To record the IMU lying still for an Allan variance analysis (`--out` must be a new or empty folder; `--overwrite`
replaces an earlier capture in it):

```
python -m xclean.live.oak_imu_capture --out recordings\imu_static --hours 3
```

The values we measured on our own OAK-D-Lite this way (3 hours, one unit; the random walks are upper bounds that
include thermal drift):

```
"imu_noise": {"gyroscope_noise_density": 1.042267e-04, "gyroscope_random_walk": 4.520925e-06,
              "accelerometer_noise_density": 1.174503e-03, "accelerometer_random_walk": 2.270714e-04}
```

They are a starting point for another unit of the same model; we have not measured other units.

Recordings are large: our 60 s recordings (30 Hz, 640 x 400, PNG images) took 0.57 to 0.64 GB each.

Only the OAK-D-Lite was tested. The calibration code reads the 640 x 400 calibration from the camera's memory.
