# Quickstart

The first steps with xclean on a Windows 11 PC with an NVIDIA GPU: install, check the machine, convert one public
sequence, run it, read the score. Then the paced mode, the live OAK camera and other stereo cameras.

On Linux or an NVIDIA Jetson, install with [docs/LINUX_JETSON.md](docs/LINUX_JETSON.md); the `xclean` commands below
are the same, with Linux paths (for example `~/data` for `C:\data`) and `unzip` in place of `tar.exe`.

Run every command in the repository folder, in a terminal where conda works, and from step 2 on with the environment
activated. The Anaconda Prompt, which Miniconda and Anaconda install, works as it is. For PowerShell or Command
Prompt, run `conda init` once and open a new terminal; if PowerShell then says that running scripts is disabled, use
the Anaconda Prompt. The example paths use `C:\data`; use your own. The times and results below are from our machine:
an NVIDIA RTX 5090, an AMD Ryzen 9 9950X and Windows 11.

Before you start:

- The EuRoC download in step 4 is one ZIP file of about 12 GB that holds all five Machine Hall sequences; there is no
  smaller download. Start it early. You unpack only the one sequence this guide uses, which takes seconds. The
  EuRoC licence allows non-commercial use only.
- Our own OAK camera recordings are not available for download (they have no ground truth). EuRoC is the public
  sequence to start with.
- `xclean replay` and `xclean live` need a new (or empty) `--out` folder, and `xclean live` a new `--record` folder.
  If the folder holds files, they stop with a message and delete nothing. `--overwrite` replaces the folder only when
  it holds an earlier output of the same command and nothing else (for `--record`, an earlier recording). xclean
  never deletes the current folder, a drive root, your home folder, or a folder that holds or lies inside your data.

## 1. Get the code

```
git clone https://github.com/necessityrobotics/xclean
```

```
cd xclean
```

## 2. Install

You need conda (Miniconda or Anaconda) and an NVIDIA driver that supports CUDA 13.

```
conda env create -f environment.yml
```

```
conda activate xclean
```

```
pip install -e .
```

- `conda env create` downloads about 330 MB of packages, plus the conda-forge package index. Our first install, with
  empty caches and a connection of about 2 MB/s, took about 8 minutes 20 seconds; with the packages already cached it
  took about a minute.
- The environment takes about 1 GB of disk. A first install also fills the conda and pip caches with about 1.2 GB, so
  plan for about 2.1 GB in all.
- `pip install -e .` installs the `xclean` command. Everything else is already in the environment.
- Open a new terminal later? Run `conda activate xclean` again before any `xclean` command.

## 3. Check the machine

```
xclean doctor
```

The doctor checks Python, the packages, the NVIDIA driver, the GPU, CuPy, the CUDA kernel compiler and headers, an
attached OAK camera and the free disk space. Each line says `OK`, `WARN` or `FAIL`. A `WARN` or `FAIL` line comes with a
one-line fix. The command exits with code 1 if any line is `FAIL`. On our machine it printed `24 OK, 0 WARN, 0 FAIL`.

On a CPU without AVX-512, expect a `WARN` on the line `LAPACK (SciPy)`. It needs no fix: it comes from the CPU, and
xclean keeps using the GPU front end (see "Start-up self-check" in the [README](README.md)).

To also run the GPU front end's start-up self-check (a few seconds):

```
xclean doctor --full
```

On our machine it printed `25 OK, 0 WARN, 0 FAIL`; the extra line `GPU front end` says that the self-check passes.

## 4. Download one EuRoC sequence

EuRoC MAV is a public indoor drone dataset with stereo images, IMU and ground truth.

- Project page: <https://projects.asl.ethz.ch/datasets/euroc-mav/>
- The files are on the ETH Research Collection: <https://doi.org/10.3929/ethz-b-000690084>
- Download **Machine Hall Datasets**: one ZIP file of about 12 GB with the sequences MH_01 to MH_05. There is no
  download of a single sequence.
- You need only `MH_01_easy`. In the big ZIP, each sequence has a folder with a ZIP file of its data
  (`MH_01_easy.zip`) and a ROS bag, which xclean does not use. Open the big ZIP in File Explorer (or 7-Zip) and copy
  only `MH_01_easy.zip` out of it, for example to `C:\data\euroc`.
- Unpack `MH_01_easy.zip` into a folder `MH_01_easy`, so that this folder contains `mav0\`. This guide uses
  `C:\data\euroc\MH_01_easy`:

```
mkdir C:\data\euroc\MH_01_easy
```

```
C:\Windows\System32\tar.exe -xf C:\data\euroc\MH_01_easy.zip -C C:\data\euroc\MH_01_easy
```

This `tar` is part of Windows 11. Give its full path as above: another `tar` on your PATH (for example the one that
comes with Git) cannot read ZIP files. MH_01_easy unpacks to 2.6 GB (7376 files). On our machine, this command
unpacked a sequence of the same dataset (V1_01_easy, 2.1 GB) in about 6 s.

**Licence.** The EuRoC record is marked "In Copyright - Non-Commercial Use Permitted". xclean does not include
the dataset; you download it yourself and follow its terms. If you publish results on it, cite: M. Burri et al., "The
EuRoC micro aerial vehicle datasets", International Journal of Robotics Research, 2016.

## 5. Convert it to a sequence folder

```
xclean convert euroc C:\data\euroc\MH_01_easy C:\data\xclean\MH_01_easy
```

This writes the folder xclean reads (described in [docs/SEQUENCE_FORMAT.md](docs/SEQUENCE_FORMAT.md)). On our
machine it printed:

```
xclean convert euroc: wrote C:\data\xclean\MH_01_easy
  frames           3682
  image size       752 x 480 (.png)
  duration         184.05 s
  images           7364 hard-linked, 0 copied
  unpaired frames  0
  IMU samples      36820
  baseline         11.008 cm
  rectified focal  436.24 px
next: xclean audit C:\data\xclean\MH_01_easy
```

When the output is on the same drive as the input, the images are hard links: no extra disk space, and it took 2 s.
On another drive they are copied, which takes minutes. The ground truth stays in the EuRoC folder; you pass it to the
run with `--gt`.

## 6. Check the folder

```
xclean audit C:\data\xclean\MH_01_easy
```

Expected output:

```
ok   MH_01_easy                       frames 3682 @ 20.0 fps  imu 36820 @ 200.0 Hz  no gt

0 dataset(s) flagged
```

"no gt" is expected here: the ground truth file is outside the folder.

Give `xclean replay` this converted folder, not the EuRoC folder you unpacked: both are called `MH_01_easy`. If the
folder is not a sequence folder (no `calib.json`, `timestamps.csv`, `left\` or `right\`), or its first images
cannot be read, `xclean replay` stops with a message before it starts.

## 7. A short first run (300 frames)

```
xclean replay C:\data\xclean\MH_01_easy --out runs\mh01_300 --mode lockstep --frames 300 --gt C:\data\euroc\MH_01_easy\mav0\state_groundtruth_estimate0\data.csv --score
```

- `replay` runs the live engine on a recorded sequence. It is the engine and the settings of our published results.
  It first prints the engine settings it uses, and one line that says whether the run will use the IMU. Here it
  will, because the EuRoC calibration has the IMU noise values; the built-in check can still stop using the IMU
  during the run.
- `--mode lockstep` gives the engine every frame and waits for it. Nothing is dropped, and the result does not depend
  on the speed of the machine.
- `--frames 300` stops after the first 300 frames (15 s of data).
- `--score` scores the run against the ground truth at the end and writes `runs\mh01_300\live_score.json`.
- At the end, `replay` prints one summary line: frames, dropped frames, the time per frame (p50 and p95; in paced
  mode this is the latency), the ATE, what the engine did with the IMU, the run folder and the path of
  `live_score.json`.

On our machine this took 30 s (the first run also compiles the CPU kernels), and `live_score.json` reported
`"ate": 0.004765...` (metres) over 278 scored frames. The run with the GPU hidden (CPU front end) gave the same
value.

## 8. The whole sequence

```
xclean replay C:\data\xclean\MH_01_easy --out runs\mh01 --mode lockstep --gt C:\data\euroc\MH_01_easy\mav0\state_groundtruth_estimate0\data.csv --score
```

On our machine this took 2 to 3 minutes for the 3682 frames (184 s of data). `live_score.json` reported
`"ate": 0.022264...` m over 3638 scored frames, no dropped frames and 0.0 % unsupported frames.

What a good result looks like on MH_01: in our final evaluation, the lock-step run of the whole sequence scored an ATE
of 0.0223 m, and three paced runs scored 0.0222 m each, with no dropped frames and 0.00 % unsupported frames. Our
lock-step runs of this sequence repeated exactly. Paced runs vary a little from run to run, because the frames reach
the engine at slightly different moments: all our paced runs of MH_01 scored between 0.0222 and 0.0224 m. Other
GPUs, CPUs and drivers can give slightly different numbers; we have not measured other machines.

`live_score.json` reports frame-level ATE, over the pose of every frame. The
[results page](https://necessityrobotics.github.io/results.html) lists keyframe-level ATE, over the poses at
keyframes only: 0.0231 m for MH_01.

### What the run folder contains

| file | what it is |
|---|---|
| `recorder.jsonl` | every pose the engine published, one JSON object per line: `stream` `LIVE-ODOM` (the pose at frame time, never changed later) or `LIVE-MAP` (the same, after the loop-closure corrections known at that time), `frame_index`, `t_pose` (seconds), `T` (4x4 pose of the rectified left camera, row by row), `state` |
| `live_score.json` | the score (with `--score`); see below |
| `harness.json` | the run: mode, engine md5, engine settings, process exit codes, resource use, frames released and dropped |
| `trajectory_live.csv` | the `LIVE-ODOM` pose of every frame as CSV (see below) |
| `feeder.json` | when each frame was released to the engine |
| `engine_summary.json`, `latency.json` | engine statistics and per-frame stage times |
| `p1_frames.jsonl` | one record per frame from the engine (features, matches, solver state) |
| `p13_worker.json`, `p13_worker_jobs.jsonl`, `p13_packets.jsonl` | the loop-closure worker's log |

### Reading `live_score.json`

| key | meaning | MH_01, whole sequence, our machine |
|---|---|---|
| `B5.ate` | position error in metres (root mean square) of the `LIVE-MAP` poses after a rigid alignment to the ground truth (rotation and translation, no scale), over every frame with a ground-truth sample within 20 ms. A frame without a pose keeps the last pose. | 0.0222 to 0.0224 |
| `B5.frames_scored`, `B5.coverage` | how many frames were scored | |
| `B2.dropped` | frames that never got a pose | 0 |
| `B4.unsupported_pct` | percent of frames without a usable pose (no pose, a lost state, or a 2-second motion error beyond the limit) | 0.00 |
| `B4.unrecovered_loss` | `true` if tracking was lost at the end and did not come back | false |
| `B1.latency_ms_p50`, `B1.latency_ms_p95` | frame-to-pose latency; meaningful in paced mode only | 19.3 to 19.5 / 25.9 (paced, pinned, our desktop with its normal background load) |

### The poses as a CSV file

`xclean replay` and `xclean live` write `trajectory_live.csv` into the run folder at the end and print its name. It
holds the `LIVE-ODOM` pose of every frame: time, x, y, z, qx, qy, qz, qw, frame index, state, publish time, latency.

## 9. Paced mode (latency)

```
xclean replay C:\data\xclean\MH_01_easy --out runs\mh01_paced --mode paced --gt C:\data\euroc\MH_01_easy\mav0\state_groundtruth_estimate0\data.csv --score
```

On our machine this took a little over 3 minutes (184 s of data plus the start). It ran without `--pin`, with a web
browser busy in the background. `live_score.json` reported a latency of 22.9 ms (`B1.latency_ms_p50`) and 33.3 ms
(`B1.latency_ms_p95`), no dropped frames, 0.0 % unsupported frames and `"ate": 0.02228...` m. A second run of the
same command, also without `--pin` but with less in the background, reported 19.2 ms and 25.7 ms and an ATE of
0.0224 m. In our final evaluation (pinned cores, our desktop with its normal background load) the three paced runs
of MH_01 had a latency of 19.3 to 19.5 ms (p50) and 25.9 ms (p95), with no dropped frames.

- Paced mode releases each frame at its recorded time, as a camera would. If the engine is still busy, the next frame
  replaces the waiting one, and the replaced frame is counted as dropped. This is how we measured latency.
- Run one paced run at a time, on an otherwise idle machine. Other programs change the latency.
- The run folder can be on a slow disk: the engine writes its logs from a background thread. With `--out` on a USB
  disk, our paced and live runs dropped no more frames than with `--out` on the internal SSD.
- `--pin` pins the engine to fixed CPU cores, as in our evaluation. It needs at least 6 cores. On 16 or more cores it
  uses the layout of our evaluation; on 6 to 15 cores a smaller layout that we have not tested. We tested it only on
  our AMD CPU, which has no efficiency cores. On a CPU with efficiency cores (many recent Intel CPUs), run without
  it.
- `harness.json` `"valid"` is our evaluation's clean-machine check (pinned cores, no other load, a timing reference
  run). A normal paced run reports `"valid": false`, lists the reasons in `"validity_reasons"`, and xclean prints them
  in one line at the end. The poses and the score are written either way and can be used.

## 10. The live OAK camera

Plug in a Luxonis OAK-D-Lite (the camera we tested). `xclean doctor` should list it on the line `OAK camera`.

```
xclean live --out runs\live1 --record recordings\live1 --duration 60
```

On our machine (the camera on a tripod, run with `--view offscreen`) the whole command took 76 s: about 13 s to
open the camera, start the engine and stop at the end, and 60 s of stream. All 1801 frames got a pose, none was
dropped, and the frame-to-pose latency was 11.3 ms (p50). That run did not use the IMU (see "Important" below); our
live runs with the IMU had a latency of 16 to 24 ms (p50), also with no dropped frames. In other runs the start and
stop took up to 16 s.

- A window shows the latest left image, the track seen from above, the tracking state and the latency. Press `q` or
  Esc to stop, or wait for `--duration` seconds.
- The view's line `published N (R /s)` is the number of poses published so far, divided by the seconds since the
  view opened. Those seconds include the engine's start-up before the first frame and, in `view_final.png`, the stop
  at the end. So R stays below the camera's 30 frames per second even when no frame is dropped: one of our 30 s runs
  showed `published 901 (22.7 /s)` with 0 dropped. To check for lost frames, read `dropped (replaced)` on the same
  line.
- The calibration is read from the camera itself.
- `--record` saves the stream as a sequence folder that you can run again later.
- `--view offscreen` saves the last view as `view_final.png` instead of opening a window; `--view none` shows nothing.
- The run folder holds `trajectory_live.csv` with the published pose of every frame.

**Important:** the camera does not store the IMU noise values or the frame rate. Without them xclean does not use the
IMU, and says so at start. Set them up once for your camera: see [docs/OWN_DATA.md](docs/OWN_DATA.md), section
"OAK-D-Lite".

Run a recording again, frame by frame (an OAK recording has no ground truth, so `--family oak` scores it as "no GT"):

```
xclean replay recordings\live1 --out runs\live1_again --mode lockstep --family oak --score
```

On our machine this took 30 s and gave the same poses as the live run, frame for frame. A recording made without
the IMU noise values runs without the IMU here too, and `xclean replay` says so at start.
`xclean audit recordings\live1` flags such a recording for its missing `imu_noise`.

## 11. Other stereo cameras

`xclean live` also takes other calibrated stereo rigs. Give the rig with `--source` and its calibration with
`--calib`:

```
xclean doctor --source uvc-sbs --device 0 --calib my_rig.json
```

```
xclean live --source uvc-sbs --device 0 --calib my_rig.json --out runs\rig1 --duration 60
```

- `uvc-sbs` is one USB stereo camera that sends both images side by side; `uvc-pair` is two USB cameras (not
  synchronised); your own source for any camera SDK is a small Python class (`--source my_rig.py:MyRig`).
- `xclean calib capture` saves chessboard image pairs, a short OpenCV script of the guide calibrates them, and
  `xclean calib opencv` writes `my_rig.json`.
- We tested live runs only on the OAK-D-Lite; the other sources with recordings and simulated cameras.

The steps (calibrate, connect, check, run) are in [docs/ANY_STEREO_LIVE.md](docs/ANY_STEREO_LIVE.md). Without a
camera, `--source folder --device recordings\live1 --mode lockstep` sends a recording through the same path and
gives the poses of `xclean live --replay`.

## Next

- Any stereo rig, live: [docs/ANY_STEREO_LIVE.md](docs/ANY_STEREO_LIVE.md).
- Your own camera and data: [docs/OWN_DATA.md](docs/OWN_DATA.md).
- The folder format in detail: [docs/SEQUENCE_FORMAT.md](docs/SEQUENCE_FORMAT.md).
- Other datasets: `xclean convert --help` lists the formats (EuRoC, KITTI, 4Seasons, Rosario v1 and v2, MADMAX).
- Every command: `xclean <command> --help`.
