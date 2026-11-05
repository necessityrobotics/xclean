# xclean on Linux and NVIDIA Jetson

From version 0.2.0, xclean installs on Linux on x86-64 and ARM64 machines with an NVIDIA GPU. That includes the
NVIDIA Jetson robot computers: the Jetson Orin family (AGX Orin, Orin NX, Orin Nano) and Jetson Thor (AGX Thor
developer kit, T5000, T4000).

## What we tested, and what we did not

- **Tested:** on GitHub's Linux machines (Ubuntu 24.04, x86-64 and ARM64, without a GPU), every route below installs
  and the test suite passes. Every xclean GPU kernel compiles with the installed CUDA kernel compiler for compute
  capability 7.5, 8.6, 8.7 (Jetson Orin), 8.9 and, with CUDA 13, 11.0 (Jetson Thor) and 12.0.
- **Not tested:** we have not run xclean on a Linux GPU or on a Jetson. Our published results and timings come from
  one Windows 11 PC (Ryzen 9 9950X, RTX 5090). Accuracy and speed on your machine are yours to measure; a Jetson is
  much slower than that PC, so expect higher latency there.
- Results differ from ours in the last digits on other CPUs. At the start of every run, the GPU front end checks that
  it computes what the CPU code computes (see "Start-up self-check" in the README). If only its 3x3 solver differs,
  xclean prints a note and keeps the GPU front end; if anything else differs, it uses the slower CPU front end for the
  whole run. We do not know yet which of the two happens on ARM64: `xclean doctor --full` shows it on your machine.

If something fails, open an issue at <https://github.com/necessityrobotics/xclean/issues> with the output of
`xclean doctor --json`.

## Pick your route

| Machine | CUDA | Route |
|---|---|---|
| Linux PC, workstation or server (x86-64), NVIDIA GPU Turing (RTX 20) or newer | driver R580 or newer (CUDA 13) | A |
| ARM64 server or workstation with an NVIDIA GPU (SBSA) | CUDA 13 driver | A |
| Jetson AGX Thor, T5000, T4000 (JetPack 7) | CUDA 13 | B, CUDA 13 |
| Jetson Orin family on JetPack 7.2 | CUDA 13 | B, CUDA 13 |
| Jetson Orin family on JetPack 6 | CUDA 12.6 | B, CUDA 12 |
| Jetson Xavier, Jetson Nano (JetPack 5 or 4) | CUDA 11 or older | not supported |

Check the JetPack release on a Jetson with `cat /etc/nv_tegra_release` (R36 = JetPack 6, R38 or newer = JetPack 7).

## Route A: conda (Linux x86-64 or ARM64, CUDA 13 driver)

The same `environment.yml` as on Windows: it brings Python 3.12, CuPy for CUDA 13, the CUDA kernel compiler (NVRTC
13.3) and the CUDA header files, so no CUDA Toolkit is needed. Install conda first, for example Miniforge
(<https://github.com/conda-forge/miniforge>).

```bash
git clone https://github.com/necessityrobotics/xclean
```

```bash
cd xclean
```

```bash
conda env create -f environment.yml
```

```bash
conda activate xclean
```

```bash
pip install -e .
```

```bash
xclean doctor
```

When the environment carries its own CUDA kernel compiler, xclean uses it in its own processes even when a CUDA
Toolkit is installed (it sets `CUDA_PATH` and loads `<env>/lib/libnvrtc.so.13` when `xclean` is imported).
`XCLEAN_KEEP_CUDA_PATH=1` turns this off.

## Route B: Jetson with JetPack's CUDA (Python venv)

JetPack installs CUDA, including the kernel compiler and headers that CuPy uses, in `/usr/local/cuda`.

1. Install JetPack's CUDA parts and Python tools (skip what is already installed):

   ```bash
   sudo apt update && sudo apt install -y nvidia-jetpack python3-venv python3-pip git
   ```

2. Get xclean and make a virtual environment:

   ```bash
   git clone https://github.com/necessityrobotics/xclean
   ```

   ```bash
   cd xclean && python3 -m venv ~/xclean-venv && source ~/xclean-venv/bin/activate && pip install -U pip
   ```

3. Install xclean with the CuPy that matches your CUDA:

   - JetPack 7 (CUDA 13: Jetson Thor, and Orin on JetPack 7.2):

     ```bash
     pip install -e ".[cuda13]"
     ```

   - JetPack 6 (CUDA 12.6: Jetson Orin):

     ```bash
     pip install -e ".[cuda12]"
     ```

4. Check the machine:

   ```bash
   xclean doctor
   ```

If `xclean doctor` says that the CUDA kernel compiler (NVRTC) does not load, point CuPy at JetPack's CUDA and try
again:

```bash
export CUDA_PATH=/usr/local/cuda LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Python: JetPack 6 has Python 3.10 and JetPack 7 has Python 3.12; xclean needs 3.10 or newer. On Python 3.10, pip
installs NumPy 2.2 and the SciPy that goes with it, which are within xclean's tested lower bounds but not the exact
versions of our results.

## Jetson settings

- Use the highest power mode of your module and fix the clocks before timing runs:

  ```bash
  sudo nvpmodel -q
  ```

  shows the current mode; `sudo nvpmodel -m <mode>` changes it (the mode numbers differ per module), and
  `sudo jetson_clocks` holds the clocks at their maximum.
- The CPU and GPU share one memory. xclean needs about 2 GB of it free for the GPU; `xclean doctor` warns below that.
- `nvidia-smi` may be missing or limited on Jetson. xclean reads the driver and memory through CUDA instead.

## Live OAK cameras on Linux

Linux needs a udev rule before a normal user can open the camera (Luxonis' rule for its USB vendor id):

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Then unplug and replug the camera, and use a USB 3 port. In route B, install the camera library with
`pip install "depthai>=3.3"`.

## What differs from Windows

- **Datasets:** the `xclean` commands are the same. Unpack the EuRoC zip files with `unzip` instead of `tar.exe`, and
  use Linux paths.
- **Timing runs (`xclean replay --mode paced`, `xclean live`):** `--pin` reads the CPU core layout from
  `/sys/devices/system/cpu`. A run is marked disturbed when another process uses more than 5 % of a core; the usual
  Linux system, desktop and Jetson service processes are exempt and reported instead, as the Windows ones are.
- **Loop-closure worker:** it runs at `nice` +5, the Linux counterpart of the "below normal" priority it has on
  Windows.
