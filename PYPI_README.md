# xclean (package `xstereo-slam`)

xclean is a real-time stereo visual-inertial SLAM system in Python (CuPy and Numba), by Necessity Robotics. It needs no C++ build step and no ROS. It publishes one pose for each frame, live. Licence: Apache-2.0.

The Python package is named `xstereo-slam`. It installs the command `xclean` and the module `xclean`. (The name `xclean` on PyPI belongs to another project.)

- Code, guides and results: https://github.com/necessityrobotics/xclean
- Paper: https://necessityrobotics.github.io/papers/xclean_paper.pdf

## What you need

- An NVIDIA GPU of the RTX 20 series or newer, and a driver that supports CUDA 13 (R580 or newer).
- Windows 11 (64-bit), or Linux on x86-64 or ARM64 (not Jetson: see below).
- Python 3.10 to 3.13. We test with Python 3.12.
- A calibrated stereo camera, or a recorded stereo sequence. xclean does nothing without input.

## Install with pip

```bash
pip install "xstereo-slam[gpu]"
xclean doctor
```

- The `gpu` extra installs CuPy for CUDA 13 and the NVIDIA pip packages with the CUDA kernel compiler and the CUDA headers. You need no conda and no CUDA Toolkit.
- `xclean doctor` checks the machine: Python, the packages, the driver, the GPU and the CUDA compiler. It prints a fix for each problem. `xclean doctor --full` also runs the self-test of the GPU front end.
- For a Luxonis OAK camera, add the `oak` extra: `pip install "xstereo-slam[gpu,oak]"`.

## The exact tested versions

pip installs the newest versions of NumPy, SciPy, Numba and OpenCV that the package allows. Our published results were made with fixed versions. To get exactly those, use the constraints file of the repository. The constraints file needs Python 3.11 or newer (3.12 is the tested Python):

```bash
pip install "xstereo-slam[gpu]" -c https://raw.githubusercontent.com/necessityrobotics/xclean/v0.3.1/constraints-tested.txt
```

What we measured on our machine (Windows 11, RTX 5090, Python 3.12), on all 2912 frames of EuRoC V1_01_easy in lock-step, against xclean 0.3.0 in the conda environment:

| pip install | Poses against 0.3.0 (conda) |
|---|---|
| Exact tested versions (constraints file) | The same pose for every frame. |
| Newest versions on 2026-11-25 (NumPy 2.5.5, SciPy 1.18.1, Numba 0.68.1, llvmlite 0.50.0, opencv-python 5.1.0.95) | The same pose for the first 341 frames. After that the positions differ: at most 3.0 cm, median 0.7 cm, on a path of 58 m. |

With the newest versions xclean runs, but it does not repeat our results digit for digit. Against the EuRoC ground truth, the position error (ATE) of this one run is 3.65 cm with the newest versions and 3.74 cm with the exact tested versions. That is one sequence, not a test of accuracy with other versions. To compare your results with ours, use the exact tested versions. `xclean doctor` prints a warning for each package that is not the tested version.

## Other ways to install

- **conda** (the route of our published results; it pins every version): see the README of the repository.
- **NVIDIA Jetson** (JetPack's own CUDA): do not use the `gpu` extra. Follow `docs/LINUX_JETSON.md` in the repository.

## What is tested

| Route | State |
|---|---|
| conda environment, Windows 11, RTX 5090 | Tested by us; the route of the published results. |
| pip with the `gpu` extra, Windows 11, RTX 5090 | Tested by us with the CUDA Toolkit of the machine hidden. Not tested on a machine that never had a CUDA Toolkit. |
| pip with the `gpu` extra, Linux x86-64 and ARM64 | Installs and passes the test suite on GitHub's Linux machines without a GPU; every GPU kernel compiles with the pip compiler. Not run on a Linux GPU. |
| NVIDIA Jetson | Not run on a device. |

Tested on one machine: AMD Ryzen 9 9950X, NVIDIA RTX 5090, Windows 11. Other CPUs, GPUs and drivers are not tested.

Contact: necessityrobotics@gmail.com
