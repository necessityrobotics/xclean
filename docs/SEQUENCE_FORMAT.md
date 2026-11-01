# Sequence format

xclean reads one **sequence folder** per run. This page describes the folder exactly as the code reads it:
`xclean/io/dataset.py` (offline runner), `xclean/live/feeder.py` (live harness feeder), `xclean/live/harness.py`,
`xclean/core/derive.py` (rig), `xclean/backend/imu.py` and `xclean/live/engine.py` (IMU), and the scorers
`xclean/live/score_live.py` and `xclean/eval/score.py`. The integrity gate (`xclean/eval/integrity.py` with
`xclean/eval/dataset_audit.py`) checks the same files before an offline run opens a folder.

To make a folder for your own camera, see [OWN_DATA.md](OWN_DATA.md). `xclean convert` writes this folder from public
datasets, and `xclean calib` writes `calib.json` from a Kalibr or OpenCV calibration.

```
<sequence>/
  calib.json          required
  timestamps.csv      required
  left/000000.png     required, one rectifiable grayscale image per timestamps.csv row (or .jpg)
  right/000000.png    required, same indices
  imu.jsonl           optional (no file = a vision-only run)
  gt.csv              optional (ground truth; may also live outside the folder, see "Ground truth")
```

## timestamps.csv

- The first line is a header and is skipped. Our converters write `frame_id,timestamp_s,left_t,right_t`.
- Every following line with at least two comma-separated fields is a frame: field 1 is the integer frame index
  `i`, field 2 is the frame time `t` in **seconds on the replay clock**. Other fields are ignored.
- The images of frame `i` are `left/%06d.<ext>` and `right/%06d.<ext>` (six-digit zero-padded index), for example
  `left/000123.png`.
- Frame times must increase. The integrity gate rejects a folder whose frame times are not monotonic, or whose
  `calib.json` `num_frames` differs from the number of rows. A `left/` or `right/` image count that differs from
  the number of rows is reported as advisory only.

**Replay clock.** All times in the folder (frames, IMU) share one clock in seconds. Our converters put its zero at
the first frame (`t = 0.0`), with the absolute epoch of that zero in `calib.json` `t0_ns`.

## Images

- Read with OpenCV as 8-bit grayscale (`cv2.IMREAD_GRAYSCALE`). Colour files are converted on read.
- Raw (distorted, unrectified) images. xclean rectifies them itself, once, from `calib.json`
  (`cv2.stereoRectify(..., alpha=0)`, then `cv2.initUndistortRectifyMap` and `cv2.remap` with linear
  interpolation). Images that are already rectified work too: give zero distortion and the rectified geometry.
- Extension: `png` by default. A registry entry (see "Naming a sequence") can say `jpg`. In folder mode xclean uses
  `jpg` when the first frame exists only as `.jpg` (`xclean.io.rows._detect_ext`). `--ext` on `xclean replay`
  overrides both. Exception: the developer tool `xclean.vio.run_p1` reads `.png` only; use `xclean replay` or
  `xclean run` for jpg sequences.
- All frames have the size `camera_width` x `camera_height` from `calib.json`.

## calib.json

A JSON object. Keys the code reads:

| key | type | read by | meaning |
|---|---|---|---|
| `camera_width`, `camera_height` | int | rig, rectification, live slot | image size in pixels |
| `M1`, `M2` | 3x3 | rig, rectification | left / right camera matrix (raw images) |
| `D1`, `D2` | list | rig, rectification | left / right OpenCV distortion coefficients |
| `R_stereo`, `T_stereo` | 3x3, 3 | rig, rectification | left -> right extrinsic as `cv2.stereoRectify` takes it (`T_stereo` in metres) |
| `baseline_m` | float | rig | stereo baseline in metres (its absolute value is used) |
| `T_body_cam0` | 4x4 | IMU, scorers | pose of the raw left camera in the IMU body frame (camera -> body). **Required for scoring** (the scorers index it directly); for the rig and IMU use, default identity |
| `R_cam0_imu` | 3x3 | IMU | body (IMU) -> cam0 rotation; used only when `T_body_cam0` is not 4x4. Default identity |
| `imu_noise` | object | IMU | `gyroscope_noise_density`, `gyroscope_random_walk`, `accelerometer_noise_density`, `accelerometer_random_walk` (continuous-time densities, Kalibr's units). Absent: the audit flags the folder, and the IMU is not used: the live engine builds its IMU factor only from this block (`vio/engine_live.py`), and `backend/imu.py` `ImuSource.available()` is false without it, so no gyro integration either |
| `imu_rate_hz` | float | IMU | declared IMU rate. 0 or absent = measured from `imu.jsonl` (offline). On the live path an IMU source is started when this is non-zero or the folder has `imu.jsonl`. A folder without `imu.jsonl` must have 0 here (or no key): otherwise a lock-step run waits for IMU samples on every frame (`live/engine.py` `imu_wait_bound`, 2 s per frame); `xclean audit` flags it and `xclean replay` warns at start |
| `camera_fps` | float | live engine, OAK view, audit | declared frame rate (the live engine's frame period; the audit compares it with the timestamps) |
| `td_s` | float | rig | declared camera-IMU time offset in seconds (default 0). Recorded and printed with the rig only: no time stamp is shifted by it. Put a known offset into the frame times instead (`xclean calib` says so when it writes Kalibr's `timeshift_cam_imu` here) |
| `t0_ns` | int | scorers, audit | **required for scoring**. Absolute epoch of replay-clock zero, in ns. The scorers compute `(gt t_ns - t0_ns) / 1e9` |
| `source` | str | live scorer | dataset family that selects the ground-truth frame model (`score_live.GT_FRAMES`: `EuRoC MAV`, `KITTI odometry`, `4Seasons`, `MADMAX`, `Rosario v1`, `Rosario v2`). Any other value is scored with the legacy model and marked unverified |
| `dataset` | str | rig name, scorer provenance | a label |
| `num_frames` | int | audit | must equal the number of timestamps.csv rows when present |

The run copies `calib.json` byte for byte into its map kit (`kit/writer.py`).

## imu.jsonl

- One JSON object per line. Blank lines are skipped.
- Fields: `t` (seconds, replay clock), `gx`, `gy`, `gz` (gyroscope, rad/s), `ax`, `ay`, `az` (accelerometer, m/s^2),
  in the IMU body frame.
- Samples in time order. The integrity gate rejects duplicated (`dt == 0`) or backward (`dt < 0`) stamps. It
  reports gaps over 100 ms, a rate that differs from `imu_rate_hz`, and accelerometer clipping.
- Missing file: the run is vision-only (with `calib.json` `imu_rate_hz` 0 or absent, see above). KITTI odometry has
  no IMU, for example.

## Ground truth

Ground truth is read by the scorers, and `<sequence>/gt.csv` (when present) is also checked by the offline integrity
gate: non-monotonic GT, or GT covering < 50 % of the frames, blocks `xclean run` and `xclean.vio.run_p1`. A GT given
with `--gt` is not checked by the gate.

- Format (`xclean.eval.score.load_gt`): comma-separated, lines starting with `#` and blank lines skipped,
  `t_ns,x,y,z,qw,qx,qy,qz` with the time in **absolute nanoseconds** on the epoch of `calib.json` `t0_ns`, the
  position in metres and a Hamilton quaternion with **w first**. Extra columns are ignored. The EuRoC
  `state_groundtruth_estimate0/data.csv` file already has this layout and can be used directly.
- Location: `<sequence>/gt.csv` by default, or any file given with `--gt FILE` (or a registry entry's `gt`).
- What the pose means (body, cam0, GNSS antenna and so on) depends on the dataset family. The live scorer reads
  it from `calib.json` `source` (see `score_live.GT_FRAMES`). The offline scorer moves body poses to cam0 through
  `T_body_cam0`.
- **Own recordings without ground truth.** Pass `--family oak` (or give the registry entry `"family": "oak"`). The
  run then scores as `NO_GT`. For any other family a missing ground-truth file is an error, so a missing drive
  never passes silently as "no GT".

## Naming a sequence: folder or tag

`xclean replay` (the module `xclean.live.harness`), `xclean run` and the developer entry points
(`xclean.vio.run_p1`, `xclean.live.score_live --tag`, `xclean.eval.integrity`) take either (note:
`xclean.eval.integrity` in folder form needs `<folder>/gt.csv`; for a folder without ground truth, such as an OAK
recording or EuRoC with GT elsewhere, use `xclean audit <folder>`):

1. **a sequence folder** (the normal use):
   ```
   xclean replay /data/xclean/MH_04_difficult --out runs/mh04 --mode lockstep --gt /data/euroc/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv --score
   ```
   The run's label (rig name, ledger name, `harness.json` `tag`) is then the folder string as given.

2. **a tag plus a registry file** (to run many sequences by name):
   ```
   xclean replay mh04 --rows rows.json --out runs/mh04 --mode lockstep
   ```
   `rows.json` maps each tag to a folder:
   ```json
   {"mh04": {"dataset": "/data/xclean/MH_04_difficult",
             "gt": "/data/euroc/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv",
             "ext": "png", "family": "euroc", "sequence": "MH_04_difficult"}}
   ```
   `gt` may be `null` (meaning `<dataset>/gt.csv`). `ext` defaults to `png` and `family` to `""`. Extra keys are
   ignored. Instead of `--rows` you can set the environment variable `XCLEAN_ROWS`. Run summaries record every
   `XCLEAN_*` variable as provenance, so `--rows` keeps them unchanged.

Resolution order (`xclean.io.rows.get`): rows registered earlier in the process (for example by `--gt` / `--family`),
then the registry file when it holds the tag, then an existing folder. Anything else is an error.

## Live and OAK-D

The live harness reads the same folder through `xclean/live/feeder.py`: `timestamps.csv`, the images and
`imu.jsonl`, released at their recorded times. The engine process receives only `calib.json` (as a dict) and the
stream. `xclean.live.oak_source` records a live OAK-D-Lite session in this same layout (`calib.json` built from
the device EEPROM, `timestamps.csv`, `imu.jsonl`, `left/`, `right/`). The recorder writes `.png` by default; a
recording with `.jpg` images is read too (see "Images"). A recording can therefore be replayed with
`xclean live --replay <folder>` or run with `xclean replay` like any other sequence.
