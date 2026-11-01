# Third-party software and data

xclean's own code is licensed under the Apache License 2.0 (see LICENSE and NOTICE). It contains no copied
third-party source code. This file lists (1) the data file shipped inside the package that is derived from a
third-party dataset, with the attribution its licence requires, and (2) the packages xclean depends on, which are
installed separately and keep their own licences.

## 1. Data shipped in this package

### ORB vocabulary `src/xclean/loop/vocab/xclean_orb_k10L4.npz` (derived from TartanAir, CC BY 4.0)

The place-recognition vocabulary was built from images of the **TartanAir** dataset:

> TartanAir: A Dataset to Push the Limits of Visual SLAM. Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu,
> Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, Sebastian Scherer. IEEE/RSJ International Conference on
> Intelligent Robots and Systems (IROS), 2020. Carnegie Mellon University, AirLab.
> Dataset: <https://theairlab.org/tartanair-dataset/>

TartanAir is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**,
<https://creativecommons.org/licenses/by/4.0/>.

**Changes made:** we used 1,318 left-camera images (every 25th frame of 36 trajectories in 12 environments of
TartanAir v1: amusement, carwelding, endofworld, gascola, hospital, japanesealley, ocean, oldtown, seasidetown,
seasonsforest, soulcity, westerndesert; "Easy" difficulty). From them we extracted ORB descriptors and clustered
them into a 10,000-word vocabulary tree (branching 10, depth 4) with inverse-document-frequency weights. The file
holds only these cluster centres (32-byte binary descriptors), the tree structure and the weights. **It contains no
image or pixel data.** The exact image list and the training settings are in
`src/xclean/loop/vocab/xclean_orb_k10L4_provenance.json`.

TartanAir is provided by its authors "as is"; they do not endorse xclean. The NOTICE file carries this attribution
too, so that it travels with xclean when you pass on the NOTICE text.

## 2. Runtime dependencies (installed by you, not included in this package)

`environment.yml` installs these packages; xclean's own files contain none of them. The licences below were read from
the installed files of an environment created from `environment.yml` on 2026-10-31 (`.dist-info` metadata and licence
files for pip packages, `conda-meta` records and the package's `info/licenses` for conda packages). Each package's
licence file is installed with it.

### From PyPI (pip)

| package | tested version | licence | needed for |
|---|---|---|---|
| numpy | 2.3.5 | BSD-3-Clause. The Windows wheel also bundles OpenBLAS (BSD-3-Clause) with LAPACK (BSD-3-Clause-Open-MPI) and the GCC runtime library (GPL-3.0-or-later WITH GCC-exception-3.1), listed in its `LICENSE.txt`, and the Microsoft C++ runtime `msvcp140.dll` | everything |
| scipy | 1.16.3 | BSD-3-Clause. The wheel bundles OpenBLAS (BSD-3-Clause) with LAPACK (BSD-3-Clause-Open-MPI) and the GCC runtime library (GPL-3.0-or-later WITH GCC-exception-3.1), listed in its `LICENSE.txt` | everything (numba's linear algebra calls SciPy's LAPACK) |
| numba | 0.63.1 | BSD-2-Clause (`LICENSE`; `LICENSES.third-party` lists code it contains) | CPU kernels |
| llvmlite | 0.46.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | CPU kernels (numba dependency) |
| opencv-python | 5.0.0.93 | Apache-2.0 (OpenCV). The Windows wheel also bundles FFmpeg (`opencv_videoio_ffmpeg500_64.dll`) under LGPL-2.1; see `LICENSE-3RD-PARTY.txt` in its `.dist-info` | image I/O, rectification, ORB features |
| cupy-cuda13x | 13.6.0 | MIT. It includes header files of NVIDIA CCCL (Thrust, CUB, libcu++), DLPack and Jitify, each with its own `LICENSE` file under `cupy/_core/include/cupy/` (Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-3-Clause) | GPU kernels |
| fastrlock | 0.8.4 | MIT | GPU kernels (CuPy dependency) |
| nvidia-cuda-runtime | 13.3.29 | NVIDIA proprietary (`LicenseRef-NVIDIA-Proprietary`): the NVIDIA CUDA Toolkit End User License Agreement, in `licenses/License.txt` of its `.dist-info` | the CUDA header files CuPy compiles its kernels with |
| psutil | 7.2.2 | BSD-3-Clause | live harness, resource monitor |
| depthai | 3.3.0 | MIT. The Windows wheel also bundles Microsoft C++ runtime DLLs and libusb (`libusb1.0-92924win.dll`, LGPL-2.1-or-later), which it installs next to the package | live OAK cameras only |

### From conda-forge (conda)

| package | tested version | licence (conda-forge record) | needed for |
|---|---|---|---|
| cuda-nvrtc | 13.3.33 | NVIDIA End User License Agreement (`LicenseRef-NVIDIA-End-User-License-Agreement`); the text (the CUDA Toolkit EULA) is in the package's `info/licenses/LICENSE` | the CUDA kernel compiler (NVRTC) CuPy uses |
| cuda-version | 13.3 | same as cuda-nvrtc (a version pin, no files) | selects NVRTC 13.3 |
| python, libpython | 3.12.16 | Python-2.0 (Python Software Foundation License) | |

conda also installs the libraries Python itself needs, each under its own licence (conda-forge records): openssl
3.6.5 (Apache-2.0), libzlib 1.3.2 (Zlib), bzip2 1.0.8 (bzip2-1.0.6), libffi 3.7.0 (MIT), liblzma 5.8.3 (0BSD),
libsqlite 3.54.1 (blessing), libexpat 2.9.0 (MIT), tk 8.6.13 (TCL), tzdata 2026e (public domain), ca-certificates
2026.10.23 (ISC), the Microsoft runtimes ucrt 10.0.26100.0 (Windows SDK licence) and vc14_runtime / vcomp14
14.51.36247 (Visual C++ 2015-2022 runtime licence), vc 14.5 (BSD-3-Clause), and the installers pip 26.2.2 (MIT),
setuptools 84.2.1 (MIT), wheel 0.48.0 (MIT) and packaging 26.3 (Apache-2.0 OR BSD-2-Clause). conda may resolve
newer builds of these on your machine.

With the `[cuda13]` / `[cuda12]` extras of `pyproject.toml` (a pip-only install, not tested), pip installs
`cupy-cuda13x` or `cupy-cuda12x` and you provide the CUDA compiler yourself (a CUDA Toolkit).

### Optional, not installed by `environment.yml`

| package | licence | needed for |
|---|---|---|
| rosbags | Apache-2.0 | `xclean convert rosario-v2` only (`pip install rosbags pillow`) |
| pillow | MIT-CMU | `xclean convert rosario-v2` only |
| PyYAML | MIT | `xclean calib` reads Kalibr's YAML files with it when it is installed, otherwise with its own reader |

Not used by the shipped code: PyTorch, gsplat and every learned network. The optional 3-D map tools (not part of
this package yet) will list their own dependencies when they ship.

## 3. Test-only

| package | tested version | licence |
|---|---|---|
| pytest | 9.2.0 | MIT |
| pluggy | 1.7.0 | MIT |
| iniconfig | 2.3.1 | MIT |
| colorama | 0.4.6 | BSD-3-Clause |
| pygments | 2.21.0 | BSD-2-Clause |
