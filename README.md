# xclean

Real-time stereo visual-inertial SLAM, built in India. Python, with GPU code written in CuPy and CPU code compiled
with Numba: no C++ build step, no ROS. Runs natively on Windows 11 and live on a Luxonis OAK-D-Lite stereo camera.

## What it does
- On the GPU: feature detection, ORB descriptors, stereo matching, and the first feature-matching step of each frame.
- On the CPU (Numba): IMU preintegration, matching against the local map, and the sliding-window factor-graph back
  end that gives each frame its pose.
- Loop closure, relocalisation and map reuse run in a separate CPU process, outside the frame path. Their corrections
  are applied to the published pose live.
- xclean uses the IMU only while a built-in check accepts it. Otherwise it runs as stereo visual odometry with loop
  closure.
- No learned networks in the SLAM pipeline.

## Results
Final evaluation: 72 routes (indoor drones, cars, a Mars-analogue rover, a farm robot and our own OAK-D-Lite
recordings), each replayed 3 times at the sensor's real rate. These are 216 runs of record, not every attempt: 8 void
attempts were re-run, and a first launch was set aside.

Frame-to-pose latency: median 20.4 ms, p95 26.4 ms (the median over the 216 runs of each run's own p50 and p95). All
runs were on one desktop (AMD Ryzen 9 9950X, NVIDIA RTX 5090, Windows 11) with its normal background load.

The headline uses 38 routes held out from xclean's development: 22 "unsealed" routes and 16 "sealed" ones (held out
under an earlier charter). None was used to build xclean. 10 of the 22 unsealed routes were run in the set-aside first
launch, whose results were never read, and our previous system had run on 36 of the 38. On these 38 routes:

- No run of record was lost. Robustness was not perfect: on 2 routes a replay attempt crashed or hung, and the
  unsupported-frames check passed on 9, failed on 4 and is pending on 25 (all MADMAX, whose ground truth has known
  faults).
- Against ORB-SLAM3 (offline, untuned, 5 runs per route; keyframe-level ATE): 10 better, 2 tie, 26 worse.
- Against OKVIS2-X (causal, unpaced, 1 run per route; visual-inertial, with the IMU off on KITTI): 11 better, 10 tie,
  17 worse. This comparison uses frame-level ATE, as pre-registered.
- Some wins are against a baseline that failed in part: 3 of the 10 against ORB-SLAM3, 1 of the 11 against OKVIS2-X.
- We do not claim to be more accurate than either system.
- 1 of our 12 pre-registered claims passed. The pre-registration document is published unchanged in
  [docs/prereg](docs/prereg/README.md).

The IMU stayed in use to the end on only 50 of the 216 runs (mostly EuRoC and 4Seasons). On the other runs the
built-in check dropped or refused it, or the dataset has no IMU (KITTI), and xclean ran as stereo visual odometry. Both
other systems ran with their IMU where the dataset has one, except where the results page says otherwise. The engine
was frozen although one of its pre-registered tests had failed on two routes.

Every route, every loss and every disclosure: <https://necessityrobotics.github.io/results.html>. How the evaluation
was done: <https://necessityrobotics.github.io/paper.html>.

## Requirements
- Windows 11, 64-bit. Linux and macOS are not tested.
- An NVIDIA GPU and a driver that supports CUDA 13 (R580 or newer). We tested an RTX 5090 with driver 616.92.
  `xclean doctor` checks the driver and the GPU (compute capability 7.5 or newer), and warns when less than 2 GB of
  GPU memory is free.
- conda (Miniconda or Anaconda), and git to get the code.
- For live use: a Luxonis OAK-D-Lite (the only camera we tested).

## Install

Use the Anaconda Prompt, which Miniconda and Anaconda install. To use PowerShell or Command Prompt instead, run
`conda init` once and open a new terminal; if PowerShell then says that running scripts is disabled, use the Anaconda
Prompt.

```
git clone https://github.com/necessityrobotics/xclean
```

```
cd xclean
```

```
conda env create -f environment.yml
```

```
conda activate xclean
```

```
pip install -e .
```

```
xclean doctor
```

- `environment.yml` installs Python 3.12 and the CUDA kernel compiler (NVRTC 13.3) from conda-forge, and with pip:
  NumPy 2.3.5, SciPy 1.16.3, Numba 0.63.1, llvmlite 0.46.0, opencv-python 5.0.0.93, CuPy 13.6.0 (`cupy-cuda13x`), the
  CUDA 13.3 header files (`nvidia-cuda-runtime`), psutil 7.2.2 and depthai 3.3.0 (OAK cameras). NumPy, SciPy, Numba,
  llvmlite and CuPy are the same builds our published results were made with; OpenCV is the plain build of the same
  version (our test runs gave identical results with it).
- It downloads about 330 MB of packages, plus the conda-forge package index. Our first install, with empty caches and
  a connection of about 2 MB/s, took about 8 minutes 20 seconds. With the packages already cached it took about a
  minute.
- The environment takes about 1 GB of disk (962 MB). A first install also fills the conda and pip caches with about
  1.2 GB, so plan for about 2.1 GB in all.
- `pip install -e .` installs the Python package `xstereo-slam`, which provides the `xclean` command, and no other
  package into the environment. pip fetches its build tool (setuptools) into a temporary folder for this, which needs
  internet access unless pip has it cached.
- Run xclean in a terminal where the environment is activated (`conda activate xclean`).
- `xclean doctor` checks the machine and prints a fix for each problem. See the [QUICKSTART](QUICKSTART.md).

No CUDA Toolkit is needed. The environment carries the CUDA kernel compiler (NVRTC) and the CUDA header files xclean
uses, and the CUDA runtime is built into CuPy. We tested this with the CUDA Toolkit installed on our machine hidden
from xclean's processes: `xclean doctor --full` and the GPU tests passed, no file of the toolkit was loaded, and a
whole EuRoC run gave the same trajectory as with the toolkit visible. We have not tested a machine that never had a
toolkit.

If a CUDA Toolkit is installed, xclean still uses the environment's own compiler and headers: it sets `CUDA_PATH`
for its own processes only. `XCLEAN_KEEP_CUDA_PATH=1` turns this off. The environment holds only the CUDA parts
xclean uses: it has no cuBLAS, cuFFT, cuRAND or cuSOLVER, so CuPy functions that need them do not work in it.

Check the install without a GPU or a dataset (about a minute on our machine):

```
pip install pytest
```

```
pytest
```

## Commands

| command | what it does |
|---|---|
| `xclean doctor` | checks Python, the packages, the driver, the GPU, CUDA and an attached OAK camera; `--full` also runs the GPU front end's self-check |
| `xclean convert <format> <in> <out>` | turns one sequence of a public dataset (EuRoC, KITTI, 4Seasons, Rosario v1 and v2, MADMAX) into a sequence folder |
| `xclean calib kalibr` / `opencv` / `check` | writes or checks `calib.json` for your own stereo camera |
| `xclean audit <folder>` | checks a sequence folder: time stamps, image counts, IMU, ground truth, calibration |
| `xclean replay <folder> --out DIR` | runs the live engine on a recorded sequence. `--mode lockstep`: every frame, no drops. `--mode paced`: frames at their recorded times, for latency. `--score` scores against ground truth |
| `xclean live --out DIR` | runs live on an OAK-D-Lite; `--record` saves the stream as a sequence folder; `--replay` runs a recording through the same path |
| `xclean run <folder> --out DIR` | the offline runner (a separate code path; needs the GPU). Without `--out` it writes to `runs\<sequence name>` in the current folder |
| `xclean score <run folder or csv>` | scores a finished run against ground truth: a `replay` or `live --replay` run folder (it finds the sequence folder it was made from), an `xclean run` output folder, or a trajectory CSV (with `--dataset`). A live camera run has no ground truth to score against. For `replay`, `--score` does this at the end of the run |

`replay` and `live` use the engine and settings of our published results and print them at start. They also print one
line that says whether the run will use the IMU or run on vision only. Both write into the `--out` folder (`live` also
into the `--record` folder, which must be a separate folder). If that folder exists and is not empty, they stop with a
message and delete nothing. `--overwrite` replaces the folder only when it holds an earlier output of the same command
and nothing else (for `--record`, an earlier recording). xclean never deletes the current folder, a drive root, your
home folder, the xclean installation, or a folder that holds or lies inside one of the command's inputs. Both write
the pose of every frame to `trajectory_live.csv` in the run folder, and end with one summary line: frames, dropped
frames, latency (p50 and p95; in lock-step mode the time per frame), what the engine did with the IMU, the run
folder, and, when the run is scored, the ATE and the path of `live_score.json`. With `--json` they also print the
raw run record (and, with `--score`, the score) as JSON; the run folder holds them either way. `xclean <command> --help` lists the
options.

- First steps, from install to a scored run: [QUICKSTART.md](QUICKSTART.md).
- Your own camera and data: [docs/OWN_DATA.md](docs/OWN_DATA.md).
- The sequence folder format: [docs/SEQUENCE_FORMAT.md](docs/SEQUENCE_FORMAT.md).

## Start-up self-check

When `replay` or `live` starts, the GPU front end checks that it computes exactly what the CPU code computes.

- If only its 3x3 solver differs from the CPU's LAPACK, xclean prints a note that names the LAPACK it found, and
  keeps the GPU front end. Results can then differ from a CPU-only run in the last digits. We expect this on CPUs
  without AVX-512, where SciPy's OpenBLAS uses other CPU kernels. We checked this by forcing those kernels on our CPU;
  no other CPU was tested.
- If any other check fails, or the GPU front end cannot start, xclean prints one warning and uses the CPU front end
  for the whole run. Latency is then higher; the poses are those of a CPU-only run. The GPU front end's cached GPU
  memory is freed. If it had already opened a CUDA context, that context stays open until the run ends, and the
  warning says so.
- `XCLEAN_STRICT_SELFCHECK=1` makes a failed check stop the run instead.

We tested both paths on our machine by forcing them (an OpenBLAS kernel for older CPUs; a GPU front end made to fail
at three points of its start-up). In our tests, when the check passed, the GPU front end gave the same poses as the
CPU front end, bit for bit.

## Known limits
- Tested on one machine: AMD Ryzen 9 9950X, NVIDIA RTX 5090, Windows 11. Other CPUs, GPUs and drivers are not tested.
- Windows only. Linux is not tested.
- The robustness test of our final evaluation did not pass in full, and most of our pre-registered claims did not
  pass (see Results).
- Datasets keep their own licences. KITTI and 4Seasons are non-commercial, and the EuRoC record is marked
  "In Copyright - Non-Commercial Use Permitted". xclean includes none of them.

## Known issues
- In paced mode, `harness.json` `"valid"` is our evaluation's clean-machine check (pinned cores, no other load, a
  timing reference run), so a normal paced run reports `false`. xclean prints one line with the reasons at the end.
  The poses and the score are not affected.
- `--pin` (fixed CPU cores for paced runs) needs at least 6 CPU cores. On 16 or more cores it uses the layout of our
  evaluation; on 6 to 15 cores a smaller layout that we have not tested. We tested `--pin` only on our AMD CPU, which
  has no efficiency cores. On a CPU with efficiency cores (many recent Intel CPUs) it is not tested: run without it
  there.

## Using xclean
xclean is open source under the **Apache License 2.0**: free for personal, research and commercial use.

**If you distribute xclean, on its own or in a product, the licence requires you to pass on its credit.** Section
4(d) requires a readable copy of the attribution notices in the [NOTICE](NOTICE) file in at least one of these
places: a NOTICE file that you distribute with it, the source code or documentation, or the screen where your product
lists third-party notices (an About screen, for example). Section 4 also requires you to give recipients a copy of the
licence, to keep the copyright notices in source code you pass on, and to mark the files you changed. The
[LICENSE](LICENSE) file has the exact terms.

We would also like you to name xclean where people can see it, for example in your credits. That is a request, not a
licence term.

Using it in a service or inside your company? We'd love to know: please add your name to [USERS.md](USERS.md) or tell us
(optional).

## Citation
The paper is not released yet. If you use xclean in research, please cite the software; the reference is on the paper
page: <https://necessityrobotics.github.io/paper.html>.

## Licence
Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Datasets used in our evaluation keep their own licences
(KITTI and 4Seasons are non-commercial); they are not included. The ORB vocabulary shipped in the package is derived
from TartanAir images (CC BY 4.0); its attribution is in NOTICE and in [THIRD_PARTY.md](THIRD_PARTY.md), which also
lists the licences of the dependencies.
