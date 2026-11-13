# Changelog

## 0.3.0 (2026-11-13): any stereo rig, live

The SLAM engine is unchanged: no algorithm, parameter or GPU kernel changed. The changed files are `__init__.py`
(the version), `cli.py`, `doctor.py`, `io/calib_import.py`, `live/oak_demo.py` and `live/oak_source.py` (its source
loop now asks a backend whether it is a camera or a recording); `live/sources.py` is new. Every other file of the package is byte-identical
to 0.2.0. The provenance md5 of a run changes, because it covers every file of the package. `xclean live` without
`--source` behaves as in 0.2.0.

- **`xclean live --source`:** live input from any calibrated stereo rig, through a frame-source interface
  (`xclean/live/sources.py`). Every source feeds the live path of the OAK-D-Lite and `xclean live --replay`.
  - `oak`: the Luxonis OAK-D-Lite (the default, unchanged).
  - `uvc-sbs`: one USB (UVC) stereo camera that sends both images side by side in one frame, read with OpenCV.
  - `uvc-pair`: two USB cameras read with OpenCV. They are not hardware-synchronised: xclean prints a warning.
  - `folder`: a sequence folder sent through the source path (a test of that path).
  - Your own source for any camera SDK: a Python class with `open`, `frames` and `close`, named as `FILE.py:CLASS`
    or `MODULE:CLASS`, with options from `--source-opt KEY=VALUE` (given as strings). It gives `StereoFrame` (two images, one time
    stamp) and, optionally, `ImuSamples`. From Python: `xclean.live.sources.run_live`.
  - Every source uses the `calib.json` that `xclean calib` writes (`--calib`), or a calibration of its own.
  - `--record`, `--duration`, the view and the run folder work as with the OAK-D-Lite.
- **`xclean doctor --source`:** opens a source, reads it for a few seconds and checks the calibration, the image
  size, the frame rate, the time stamps, the IMU samples and the synchronisation of the pair.
- **`xclean calib capture`:** saves stereo image pairs from a source for a chessboard calibration.
- **`xclean live --calib`:** a file that cannot be read now stops the command with a plain message (exit code 2),
  before the run folder is made.
- **Guide:** [docs/ANY_STEREO_LIVE.md](docs/ANY_STEREO_LIVE.md): calibrate (with a tested OpenCV chessboard
  script), connect, check, run.
- **Tests:** a recording sent through the source path releases the same frames, images, time stamps and IMU
  packets as `xclean live --replay`, and gives the same pose for every frame as `xclean replay` and
  `xclean live --replay` (lock-step). Simulated cameras test the live clock, the built-in OpenCV sources, the
  errors, `doctor --source` and `calib capture`. The guide's chessboard script is run on rendered chessboard pairs.
- **Tested on our machine (RTX 5090):** the `folder` source gave the same pose for every frame as
  `xclean live --replay`, in lock-step: all 2912 frames of EuRoC V1_01_easy (IMU in use, loop closure on and off;
  also the same as `xclean replay`), and the first 900 frames of one of our own OAK-D-Lite recordings.
- **Not tested:** no live run of `uvc-sbs`, `uvc-pair` or an SDK source (RealSense, ZED, others) has been made by the
  authors. Live runs were tested only on the OAK-D-Lite.

## 0.2.0 (2026-11-05): Linux and NVIDIA Jetson

The SLAM engine is unchanged: no algorithm, parameter or GPU kernel changed. On Windows, xclean behaves as in 0.1.0.

- **Linux on x86-64 and ARM64**, including NVIDIA Jetson Orin (JetPack 6 with CUDA 12.6, or JetPack 7.2 with CUDA 13)
  and Jetson Thor (JetPack 7, CUDA 13). Install guide: [docs/LINUX_JETSON.md](docs/LINUX_JETSON.md).
- **CUDA kernel compiler on Linux:** when the environment carries its own NVRTC (`environment.yml`), `import xclean`
  points `CUDA_PATH` at the environment and loads that NVRTC, as 0.1.0 already did on Windows. Without one (a venv on
  JetPack), CuPy uses the installed CUDA Toolkit.
- **`xclean doctor`:** a new Platform line (OS, CPU architecture, Jetson model and L4T release). The driver and GPU
  rules follow the CUDA version CuPy was built for: CUDA 13 needs a CUDA 13 driver and compute capability 7.5 or
  newer; CuPy for CUDA 12 (`pip install -e ".[cuda12]"`) needs a CUDA 12 driver. Fix hints name JetPack and the
  Linux udev rule for OAK cameras.
- **Timing harness on Linux:** `--pin` reads the physical-core layout from sysfs; Linux system, desktop and Jetson
  service processes join the allow-list of the 5 % rule (Windows keeps its exact list).
- **Loop-closure worker:** below-normal priority on Linux too (`nice` +5).
- **Packaging:** the `cuda12` and `cuda13` extras stay below CuPy 14 (13.6 is the tested CuPy).
- **Tests:** Linux CUDA-compiler step, doctor platform and CUDA 12 / 13 rules, Linux core layout, allow-list, and a
  test that compiles every GPU kernel with NVRTC for sm_75, 86, 87 (Orin), 89, 110 (Thor, CUDA 13) and 120 without a
  GPU. A GitHub Actions workflow runs the suite on Linux x86-64 and ARM64 for every install route.
- **Not tested:** no xclean run on a Linux GPU or on a Jetson has been made by the authors. Accuracy and speed there are
  unmeasured.

## 0.1.0 (2026-11-02)

First public release (Windows 11, NVIDIA GPU, live OAK-D-Lite).
