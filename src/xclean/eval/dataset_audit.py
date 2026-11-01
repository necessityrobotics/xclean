# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Converted-dataset integrity AUDIT (the checks behind xclean.eval.integrity.IntegrityGate).

    python -m xclean.eval.dataset_audit <sequence dir> [<sequence dir> ...]

Checks per sequence folder (needs timestamps.csv + calib.json; imu.jsonl / gt.csv optional):
  IMU    monotonic t, dt==0 / dt<0 counts, gaps > 100 ms, measured rate vs declared imu_rate_hz, rail-clipping
         census, quietest-window |a| vs g.
  FRAMES timestamps.csv rows vs left/ right/ image counts vs calib num_frames, monotonic frame times, measured fps
         vs declared camera_fps.
  GT     monotonic t, coverage of the frame span.
  CALIB  imu_noise block present (without it the engine does not use the IMU); imu_rate_hz 0 when there is no
         imu.jsonl (otherwise replay waits for IMU data that never comes).
Exit code 1 if any folder is FLAGged, 0 if all are clean.
"""
# Vendored from dataset_integrity.py of our own earlier research tree (same company): G and audit() below are that
# file's lines 28-34 and 46-195, verbatim except the two CALIB messages (the imu_noise text now says what the engine
# does; the stereo-only check is new) and the rail rule (RAIL_MIN_G: a rail must be a measuring-range end).  Its
# list of machine-specific dataset roots is not vendored; main() takes the folders as arguments (or
# XCLEAN_DATASET_ROOTS, os.pathsep-separated roots whose sub-folders are audited) and flags a given folder that is
# not a sequence folder.
import json
import os
import sys

import numpy as np

G = 9.80665
# An accelerometer rail is the end of the sensor's measuring range.  The smallest range in use is +-2 g (Rosario v1's
# IMU clips at 19.600 m/s^2); a sensor at rest reads 1 g, so a value that repeats at about 1 g is gravity, not a rail.
# A rail is flagged only at RAIL_MIN_G or above: 90 % of 2 g, which leaves room for the factory scale and offset
# correction applied after the clip.
RAIL_MIN_G = 1.8


def audit(d):
    flags = []
    info = []
    cal_p = os.path.join(d, "calib.json")
    ts_p = os.path.join(d, "timestamps.csv")
    if not (os.path.isfile(cal_p) and os.path.isfile(ts_p)):
        return None                                    # not a converted dataset
    cal = json.load(open(cal_p))

    # ---- frames ------------------------------------------------------
    ft = []
    with open(ts_p) as f:
        next(f)
        for ln in f:
            p = ln.split(",")
            if len(p) >= 2:
                ft.append(float(p[1]))
    ft = np.asarray(ft)
    for sub in ("left", "right"):
        sd = os.path.join(d, sub)
        if os.path.isdir(sd):
            n = sum(1 for x in os.listdir(sd) if x.endswith((".png", ".jpg")))
            if n != len(ft):
                flags.append(f"{sub}/ has {n} images vs {len(ft)} csv rows")
    if cal.get("num_frames") not in (None, len(ft)):
        flags.append(f"calib num_frames {cal.get('num_frames')} != {len(ft)}")
    fdt = np.diff(ft)
    if len(fdt) and (fdt <= 0).any():
        flags.append(f"frame times non-monotonic ({int((fdt <= 0).sum())})")
    fps_meas = 1.0 / np.median(fdt) if len(fdt) else 0.0
    fps_dec = cal.get("camera_fps")
    if fps_dec and abs(fps_meas - fps_dec) / fps_dec > 0.02:
        flags.append(f"fps declared {fps_dec} vs measured {fps_meas:.2f}")
    info.append(f"frames {len(ft)} @ {fps_meas:.1f} fps")

    # ---- imu ---------------------------------------------------------
    imu_p = os.path.join(d, "imu.jsonl")
    if os.path.isfile(imu_p):
        T, A = [], []
        with open(imu_p) as f:
            for ln in f:
                s = json.loads(ln)
                T.append(s["t"])
                A.append((s["ax"], s["ay"], s["az"]))
        T = np.asarray(T)
        A = np.asarray(A)
        dt = np.diff(T)
        med = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.0
        n0 = int((dt == 0).sum())
        nneg = int((dt < 0).sum())
        if n0:
            flags.append(f"imu dt==0 x{n0} (engine drops these silently)")
        if nneg:
            flags.append(f"imu dt<0 x{nneg}")
        big = int((dt > 0.1).sum())
        if big:
            flags.append(f"imu gaps >100ms x{big} (max {dt.max()*1000:.0f}ms; "
                         "intake drops the interval)")
        rate_meas = (len(T) - 1) / (T[-1] - T[0]) if len(T) > 1 else 0.0
        rate_dec = cal.get("imu_rate_hz")
        if rate_dec and med > 0 and abs(1.0 / med - rate_dec) / rate_dec > 0.005:
            flags.append(f"imu rate declared {rate_dec} vs 1/median_dt "
                         f"{1.0/med:.2f} Hz (time-scale error in preintegration)")
        # rail clipping: any axis value pinned repeatedly at the stream max, when that max is a measuring-range end
        for ax_i, name in enumerate("xyz"):
            col = A[:, ax_i]
            mx = np.abs(col).max()
            if mx >= RAIL_MIN_G * G:
                pinned = int((np.abs(np.abs(col) - mx) < 1e-3 * mx).sum())
                if pinned >= 8:
                    flags.append(f"accel {name} rails at ±{mx:.3f} x{pinned}")
        # static |a| sanity from the quietest 1 s window
        if med > 0:
            w = max(3, int(round(1.0 / med)))
            if len(A) > 2 * w:
                gm = np.linalg.norm(A, axis=1)
                var = np.convolve((gm - np.convolve(
                    gm, np.ones(w) / w, "same")) ** 2, np.ones(w) / w, "same")
                q = np.argmin(var[w:-w]) + w
                a_static = float(gm[q - w // 2:q + w // 2].mean())
                if abs(a_static - G) / G > 0.05:
                    flags.append(f"quietest-window |a| {a_static:.3f} vs g "
                                 "(>5% — offset/scale/level-arm suspect)")
        info.append(f"imu {len(T)} @ {rate_meas:.1f} Hz")
        if not cal.get("imu_noise"):
            flags.append("calib has NO imu_noise block -> the engine does not "
                         "use the IMU (vision only); add imu_noise to use it")
    else:
        info.append("no imu")
        if cal.get("imu_rate_hz"):
            flags.append(f"calib imu_rate_hz {cal.get('imu_rate_hz')} but no "
                         "imu.jsonl -> replay waits for IMU data that never "
                         "comes (up to 2 s per frame); set imu_rate_hz to 0 "
                         "for a stereo-only folder")

    # ---- gt ----------------------------------------------------------
    gt_p = os.path.join(d, "gt.csv")
    if os.path.isfile(gt_p):
        gt = []
        with open(gt_p) as f:
            for ln in f:
                if ln.startswith("#"):
                    continue
                p = ln.split(",")
                if len(p) >= 4:
                    gt.append(float(p[0]))
        gt = np.asarray(gt) / 1e9
        # The frame clock's epoch is DECLARED by the converter in
        # calib.json["t0_ns"]; use it rather than guessing from magnitude.
        # The old magnitude test (|head| > 1e6 => "abs-time base") only fired
        # for ns epochs near 1.5e18 and silently mis-read the UZH-FPV rows,
        # whose gt epochs sit at ~5e11-1e12: outdoor_forward_1/3/5 were all
        # FLAGged "gt starts 552-1061s after frames" when in truth their gt
        # overlaps the run perfectly well once t0_ns is subtracted.  Two of
        # those three have saved, published-comparable results, so the flag
        # was provably a tool defect, not a data defect (2026-08-23).
        _t0 = float(cal.get("t0_ns", 0) or 0) / 1e9
        if _t0 > 0.0:
            gt = gt - _t0
        if len(gt) > 1:
            gdt = np.diff(gt)
            if (gdt <= 0).any():
                flags.append(f"gt non-monotonic x{int((gdt <= 0).sum())}")
            if len(ft):
                head = gt[0] - ft[0]
                tail = ft[-1] - gt[-1]
                if abs(head) > 1e6:
                    # absolute-epoch gt vs relative frame clock — the scorer
                    # associates on its own base, so span checks don't apply
                    info.append("gt abs-time base")
                else:
                    if head > 5.0:
                        flags.append(f"gt starts {head:.1f}s after frames")
                    if tail > 5.0:
                        flags.append(f"gt ends {tail:.1f}s before frames end "
                                     "(unscored tail is still processed)")
        # GT SPAN COVERAGE (2026-08-23).  UZH-FPV scores only the window its
        # tracker actually covers, and the fraction varies enormously by row
        # — measured 13.8% (if5) / 59.7% (of1) / 66.2% (if7) / 67.4% (of3).
        # An ATE quoted over 13.8% of a run is not the same measurement as
        # one over 100%, and nothing previously surfaced that at audit time.
        if len(gt) > 1 and len(ft) > 1:
            _ov = max(0.0, min(ft[-1], gt[-1]) - max(ft[0], gt[0]))
            _span = max(ft[-1] - ft[0], 1e-9)
            _cov = 100.0 * _ov / _span
            info.append(f"gt {len(gt)} ({_cov:.0f}% span)")
            if _cov < 50.0:
                flags.append(f"gt covers only {_cov:.0f}% of the run "
                             "— ATE is scored on that window only")
        else:
            info.append(f"gt {len(gt)}")
    else:
        info.append("no gt")

    return flags, "  ".join(info)


USAGE = ("usage: xclean audit <sequence dir> [<sequence dir> ...]\n"
         "(or set XCLEAN_DATASET_ROOTS to folders whose sub-folders are sequence folders)")


def main():
    targets = []                          # (path, given on the command line)
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(USAGE)
        sys.exit(0)
    if len(sys.argv) > 1:
        targets = [(os.path.abspath(a), True) for a in sys.argv[1:]]
    else:
        roots = [x for x in (os.environ.get("XCLEAN_DATASET_ROOTS") or "").split(os.pathsep) if x]
        if not roots:
            print(USAGE, file=sys.stderr)
            sys.exit(2)
        for r in roots:
            if os.path.isdir(r):
                for n in sorted(os.listdir(r)):
                    targets.append((os.path.join(r, n), False))
            else:
                targets.append((os.path.abspath(r), True))
    bad = 0
    for t, given in targets:
        if not os.path.isdir(t):
            if given:                     # a mistyped path is a finding, never a silent "0 flagged"
                print(f"FLAG {os.path.basename(t):32s} {'is a file' if os.path.exists(t) else 'not found'}")
                print(f"       - not a sequence folder: {t}")
                bad += 1
            continue
        res = audit(t)
        if res is None:
            if given:
                print(f"FLAG {os.path.basename(t):32s} not a sequence folder")
                print(f"       - no calib.json or no timestamps.csv in {t} (docs/SEQUENCE_FORMAT.md; "
                      "convert a public dataset with xclean convert)")
                bad += 1
            continue
        flags, info = res
        tag = "FLAG" if flags else "ok  "
        print(f"{tag} {os.path.basename(t):32s} {info}")
        for fl in flags:
            print(f"       - {fl}")
        bad += bool(flags)
    print(f"\n{bad} dataset(s) flagged")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
