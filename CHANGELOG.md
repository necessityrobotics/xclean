# Changelog

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
