#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""LIVE HARNESS launcher: FEEDER -> ENGINE (fenced) -> RECORDER, with a resource monitor and script-decided validity.

    python -m xclean.live.harness <sequence dir | row tag> --out DIR [--mode paced|lockstep] [--frames N] [--pin]
        [--final] [--threads T] [--score] [--rows REGISTRY.json] [--gt GT.csv] [--ext png|jpg] [--family oak]
    (a row tag needs a registry file: xclean.io.rows, docs/SEQUENCE_FORMAT.md)

Writes into DIR: recorder.jsonl (RECORDER, the ONLY input of the live scorer), feeder.json (release log + feeder
validity), engine_summary.json + latency.json (engine), harness.json (pids, engine md5, env flags, resource peaks,
validity checks).  The standard engine flag set (X1_ENV) is applied unless --no-x1-flags.  Never selects or signals a
process by name; the monitor only READS other processes' CPU / memory to decide validity.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import threading
import time

X1_ENV = {"XTEN_PGO_INFO": "derived", "XTEN_PGO_ODO": "edge", "XTEN_GBA_BESTFIX": "1", "XTEN_GBA_SOLVER": "dpcg",
          "XTEN_GBA_ITERBUDGET": "1", "XTEN_FRONT_GATE": "recount", "XTEN_PNP_VERIFY": "all",
          "XTEN_PROX_PATH": "despike", "XTEN_GBA_SCREEN": "cheir", "XTEN_LOOP_FISHER": "infeasible"}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# named desktop/OS processes are exempt from the per-process 5 % bar,
# recorded per run, and the run is still void if together they exceed 1 core or 256 MB VRAM
ALLOW = ("system", "msmpeng.exe", "claude.exe", "wallpaper64.exe", "wallpaper32.exe", "dwm.exe", "explorer.exe",
         "svchost.exe", "memcompression", "registry", "smss.exe", "csrss.exe", "wininit.exe",
         "services.exe", "lsass.exe", "lsaiso.exe", "winlogon.exe", "fontdrvhost.exe", "dllhost.exe",  # Windows OS processes
         "sihost.exe", "taskhostw.exe", "runtimebroker.exe", "ctfmon.exe", "audiodg.exe", "wmiprvse.exe",  # (shipped with
         "searchindexer.exe", "searchprotocolhost.exe", "searchfilterhost.exe", "aggregatorhost.exe",  # Windows) and the
         "compattelrunner.exe", "dashost.exe", "tiworker.exe", "trustedinstaller.exe", "mousocoreworker.exe",  # display
         "usocoreworker.exe", "backgroundtaskhost.exe", "sppsvc.exe", "sgrmbroker.exe", "spoolsv.exe",  # driver service:
         "wudfhost.exe", "mpsigstub.exe", "mpcmdrun.exe", "nissrv.exe", "mpdefendercoreservice.exe",  # exempt from the
         "securityhealthservice.exe", "securityhealthsystray.exe", "smartscreen.exe", "widgets.exe",  # 5 % bar, still
         "widgetservice.exe", "textinputhost.exe", "nvdisplay.container.exe",  # REPORTED per run
         "searchhost.exe", "startmenuexperiencehost.exe", "shellhost.exe", "crossdeviceresume.exe",
         "nvidia overlay.exe", "msedgewebview2.exe", "phoneexperiencehost.exe", "powertoys.exe",
         "powertoys.quickaccess.exe", "powertoys.peek.ui.exe", "powertoys.fancyzones.exe", "microsoft.cmdpal.ui.exe",
         "m365copilot.exe", "amdrssrcext.exe", "radeonsoftware.exe", "steamwebhelper.exe", "applicationframehost.exe",
         "systemsettings.exe", "lockapp.exe", "msedge.exe",
         "taskmgr.exe",                     # the user's Task Manager
         "magnify.exe", "chrome.exe",       # Windows Magnifier + the user's browser
         "powertoys.colorpickerui.exe")     # PowerToys desktop UI (a D3D client listed by nvidia-smi)


def vram_used_mb():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW).stdout
        return float(o.strip().splitlines()[0])
    except Exception:
        return None


def gpu_compute_apps():
    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW).stdout
        out = []
        for ln in o.strip().splitlines():
            p = [x.strip() for x in ln.split(",")]
            if len(p) >= 2 and p[0].isdigit():
                out.append((int(p[0]), p[1]))
        return out
    except Exception:
        return []


class Monitor(threading.Thread):
    """Samples every 0.5 s: the engine tree's private commit, total VRAM; every 10 s: CPU-seconds per second of the
    engine tree and of every OTHER process (validity: none above 5 % of a core)."""

    def __init__(self, engine_pid, own_pids, vram_base, extra_pids=None):
        super().__init__(daemon=True)
        import psutil
        self.ps = psutil
        self.engine = psutil.Process(engine_pid)
        self.own = set(own_pids)
        # P1.3 B6 report: the WHOLE tree = the engine tree + these processes (the worker, the recorder);
        # separate series, the existing engine-tree fields are unchanged
        self.extra = []
        for q in (extra_pids or []):
            try:
                self.extra.append(psutil.Process(q))
            except Exception:
                pass
        self.series_all = []
        self.peak_private_all = 0
        self.cpu_windows_all = []
        self.vram_base = vram_base
        self.stop = threading.Event()
        self.peak_private = 0
        self.peak_vram = vram_base or 0.0
        self.cpu_windows = []
        self.others_over = {}
        self.allow_cpu = {}
        self.allow_sum_max = 0.0
        self._procs = {}
        self.series = []                                   # (wall time, engine-tree private bytes) every 0.5 s

    def _tree(self):
        try:
            return [self.engine] + self.engine.children(recursive=True)
        except Exception:
            return []

    def run(self):
        ps = self.ps
        last_t = time.time()
        last_cpu = None
        last_oth = time.time()
        for p in ps.process_iter():
            try:
                p.cpu_percent(None)
                self._procs[p.pid] = p
            except Exception:
                pass
        while not self.stop.is_set():
            tree = self._tree()
            priv = 0
            cpu = 0.0
            for p in tree:
                try:
                    mi = p.memory_info()
                    priv += getattr(mi, "private", mi.rss)
                    ct = p.cpu_times()
                    cpu += ct.user + ct.system
                except Exception:
                    pass
            self.peak_private = max(self.peak_private, priv)
            self.series.append((time.time(), priv))
            if self.extra:
                priv_all, cpu_all = priv, cpu
                for p in self.extra:
                    try:
                        mi = p.memory_info()
                        priv_all += getattr(mi, "private", mi.rss)
                        ct = p.cpu_times()
                        cpu_all += ct.user + ct.system
                    except Exception:
                        pass
                self.peak_private_all = max(self.peak_private_all, priv_all)
                self.series_all.append((time.time(), priv_all))
            v = vram_used_mb()
            if v is not None:
                self.peak_vram = max(self.peak_vram, v)
            now = time.time()
            if last_cpu is None:
                last_cpu, last_t = cpu, now
                if self.extra:
                    last_cpu_all = cpu_all
            elif now - last_t >= 10.0:
                self.cpu_windows.append((cpu - last_cpu) / (now - last_t))
                if self.extra:
                    self.cpu_windows_all.append((cpu_all - last_cpu_all) / (now - last_t))
                    last_cpu_all = cpu_all
                last_cpu, last_t = cpu, now
            if now - last_oth >= 5.0:
                last_oth = now
                tree_pids = {p.pid for p in tree}
                allow_sum = 0.0
                for p in ps.process_iter():
                    if p.pid in self.own or p.pid in tree_pids or p.pid == 0:
                        continue
                    q = self._procs.get(p.pid)
                    if q is None:
                        try:
                            p.cpu_percent(None)
                            self._procs[p.pid] = p
                        except Exception:
                            pass
                        continue
                    try:
                        c = q.cpu_percent(None)
                        name = q.name()
                        if name.lower() in ALLOW:
                            allow_sum += c
                            self.allow_cpu[name] = max(self.allow_cpu.get(name, 0.0), c)
                        elif c > 5.0:
                            k = "%s(%d)" % (name, q.pid)
                            self.others_over[k] = max(self.others_over.get(k, 0.0), c)
                    except Exception:
                        pass
                self.allow_sum_max = max(self.allow_sum_max, allow_sum)
            self.stop.wait(0.5)


# ---- VALIDITY PROTOCOL v2 --------------------------------
# (a) the engine tree (ENGINE + RECORDER; the FEEDER excluded) is pinned to dedicated physical cores, ONE logical CPU
#     per core (their SMT siblings are left unused by our work); the FEEDER gets its own cores; masks are recorded.
# (b) DECISIVE: the witness stages (rectify / extract / track, p50 AND p95) within +-20 % of a clean single-run
#     reference measured under the same pinning on this machine (provenance recorded), plus the FEEDER jitter/slope.
# (c) the allow-listed desktop processes are REPORTED per run, no longer capped.
# (d) any non-allow-listed process above 5 % of a core (Windows maintenance etc.) or a non-allowed GPU app VOIDS.
ENGINE_CORES = tuple(range(8, 16))         # physical cores reserved for ENGINE + RECORDER
FEEDER_CORES = (6, 7)                      # physical cores of the FEEDER (both SMT siblings)
WORKER_CORES = (5,)                        # P1.3: the loop-closure WORKER's own physical core (first sibling)
WITNESS_V2 = ("rectify", "extract", "track")
WITNESS_TOL = 0.20
def _user_data_dir():
    """The per-user data folder of xclean (never the installation folder)."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") \
        or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "xclean")


# Timing-reference registry: an explicit file (XCLEAN_WITNESS_REF_V2), else one file in the per-user data folder.  It is
# read by every paced run and written only by a --reference run.
WITNESS_REF_V2 = os.environ.get("XCLEAN_WITNESS_REF_V2") or os.path.join(_user_data_dir(), "timing_references.json")


def core_groups():
    """Logical CPUs per PHYSICAL core, from the OS (GetLogicalProcessorInformation, RelationProcessorCore); fallback
    (non-Windows): consecutive groups.  -> list of lists, index = physical core."""
    try:
        import ctypes
        import ctypes.wintypes as w

        class SLPI(ctypes.Structure):
            _fields_ = [("ProcessorMask", ctypes.c_size_t), ("Relationship", ctypes.c_int),
                        ("u", ctypes.c_byte * 16)]
        k = ctypes.windll.kernel32
        n = w.DWORD(0)
        k.GetLogicalProcessorInformation(None, ctypes.byref(n))
        buf = (SLPI * (n.value // ctypes.sizeof(SLPI)))()
        if not k.GetLogicalProcessorInformation(buf, ctypes.byref(n)):
            raise OSError("GetLogicalProcessorInformation failed")
        g = [[i for i in range(64) if (x.ProcessorMask >> i) & 1] for x in buf if x.Relationship == 0]
        if g:
            return g
    except Exception:
        pass
    import psutil
    nl, nc = psutil.cpu_count(logical=True), psutil.cpu_count(logical=False)
    per = max(1, nl // max(1, nc))
    return [list(range(c * per, c * per + per)) for c in range(nc)]


MIN_PIN_CORES = 6                          # --pin: engine + recorder >= 3 cores, feeder 1, worker 1, 1 left free


def core_layout(n_cores):
    """Physical cores for --pin on a CPU with n_cores physical cores -> (engine cores, feeder cores, worker cores).
    16 or more: the layout of the published runs (ENGINE_CORES, FEEDER_CORES, WORKER_CORES).  6 to 15: the upper half
    for the engine + recorder, the next core down for the feeder, the one below it for the loop worker; the lowest
    cores stay free for the system.  Fewer than MIN_PIN_CORES: ValueError with a plain message."""
    n = int(n_cores)
    if n >= 16:
        return tuple(ENGINE_CORES), tuple(FEEDER_CORES), tuple(WORKER_CORES)
    if n < MIN_PIN_CORES:
        raise ValueError("--pin needs a CPU with at least %d cores; this one has %d. Run without --pin."
                         % (MIN_PIN_CORES, n))
    half = n // 2
    return tuple(range(half, n)), (half - 1,), (half - 2,)


def pin_layout(groups=None):
    """The --pin masks: engine + recorder = the FIRST sibling of each engine core, feeder = every sibling of the
    feeder cores, worker = the first sibling of the worker core.  -> dict (ValueError below MIN_PIN_CORES cores)."""
    g = groups if groups is not None else core_groups()
    ec, fc, wc = core_layout(len(g))
    return {"engine_cpus": [g[c][0] for c in ec], "feeder_cpus": [x for c in fc for x in g[c]],
            "worker_cpus": [g[c][0] for c in wc], "engine_cores": list(ec), "feeder_cores": list(fc),
            "worker_cores": list(wc), "siblings_left_unused": [x for c in ec for x in g[c][1:]], "groups": g}


def pin_masks(groups=None):
    """-> (engine logical CPUs, feeder logical CPUs, groups): pin_layout's masks."""
    L = pin_layout(groups)
    return L["engine_cpus"], L["feeder_cpus"], L["groups"]


CLEAN_MACHINE_REASONS = ("feeder jitter/slope", "non-allow-listed load", "non-allowed GPU apps", "engine tree not pinned",
                         "witness")


def plain_validity_note(mode, checks, witness, reasons):
    """One plain end-of-run line about harness.json's "valid" for a paced run (None when valid or not paced).
    "valid" is the benchmark's clean-machine flag; a user's run on a normal desktop is usually not "valid", and its
    trajectory and scores are still usable unless a process failed or the engine reported fence errors."""
    if mode != "paced" or not reasons:
        return None
    if any(not r.startswith(CLEAN_MACHINE_REASONS) for r in reasons):
        return ("xclean: warning: this run is not valid: %s. Check the run before you use its numbers."
                % "; ".join(reasons))
    plain = []
    for r in reasons:
        if r.startswith("engine tree not pinned"):
            plain.append("not run with --pin")
        elif r.startswith("witness"):
            if witness is not None and witness.get("ok") is False:
                plain.append("stage times differ from the timing reference by more than %d %%"
                             % round(100 * WITNESS_TOL))
            elif witness is None or witness.get("note") == "no reference":
                plain.append("no timing reference run on this machine")
            else:
                plain.append("stage times could not be compared with the timing reference")
        elif r.startswith("non-allowed GPU apps"):
            gpu = checks.get("other_gpu_compute_apps_not_allowed") or []
            plain.append("other programs use the GPU (%s)"
                         % ", ".join(sorted({str(x[1]).split(chr(92))[-1] for x in gpu})))
        elif r.startswith("non-allow-listed load"):
            load = checks.get("others_over_5pct_core") or {}
            plain.append("other programs used the CPU (%s)" % ", ".join(sorted({k.split("(")[0] for k in load})))
        elif r.startswith("feeder jitter/slope"):
            plain.append("the frame clock was not steady (timer jitter or drift)")
    return ("xclean: note: harness.json says valid=false. \"valid\" is our benchmark's clean-machine check, not an "
            "error: the trajectory and scores of this run are usable. Reasons found: %s." % "; ".join(plain))


def imu_config_note(seq_dir, calib):
    """A sequence folder whose calib.json declares an IMU rate but has no imu.jsonl: replay would wait for IMU data
    that never comes (up to 2 s per frame in lock-step).  -> one plain warning line, or None."""
    rate = calib.get("imu_rate_hz")
    if rate and not os.path.isfile(os.path.join(seq_dir, "imu.jsonl")):
        return ("xclean: warning: calib.json sets imu_rate_hz %s but the folder has no imu.jsonl. The engine waits for "
                "IMU data that never arrives (up to 2 s per frame in lock-step, one camera period per frame when "
                "paced). For a stereo-only folder set imu_rate_hz to 0 (docs/OWN_DATA.md, \"No IMU (stereo only)\")."
                % rate)
    return None


def imu_status_line(calib, has_imu, env=None, recording=False):
    """The start-of-run line that says whether this run uses the IMU ("IMU in use") or not ("vision only")."""
    env = os.environ if env is None else env
    what = "recording" if recording else "sequence folder"
    if (env.get("XCLEAN_P1_VI", "1") or "1").strip() == "0":
        return "xclean: vision only: the IMU is switched off (XCLEAN_P1_VI=0)."
    if not has_imu:
        return "xclean: vision only: the %s has no imu.jsonl." % what
    if not calib.get("imu_noise"):
        return ("xclean: vision only: the %s's calib.json has no imu_noise block, so this run does not use the IMU "
                "(docs/OWN_DATA.md, \"IMU noise values\")." % what)
    return ("xclean: IMU in use: IMU data and IMU noise values were found. The engine adds the IMU once its start-up "
            "check accepts the data; the summary at the end says from which frame.")


def build_parser():
    """The command line of `xclean replay` (= python -m xclean.live.harness)."""
    hide = argparse.SUPPRESS
    ap = argparse.ArgumentParser(
        description="Run the xclean engine on a recorded sequence folder, as it runs live: the frames are released at "
                    "their recorded times (--mode paced) or one after the other (--mode lockstep). The run folder "
                    "gets the published poses (trajectory_live.csv) and the run logs.",
        allow_abbrev=False)
    ap.add_argument("tag", metavar="sequence_folder",
                    help="the sequence folder to replay (docs/SEQUENCE_FORMAT.md), or a sequence name from --rows")
    ap.add_argument("--out", required=True, help="the run folder to write; it must be new or empty")
    ap.add_argument("--mode", default="paced", choices=("paced", "lockstep"),
                    help="paced (default): frames arrive at the camera's rate, and a frame the engine cannot take in "
                         "time is dropped; lockstep: each frame is released when the engine has finished the one "
                         "before, so no frame is dropped and the result repeats exactly")
    ap.add_argument("--frames", type=int, default=None, help="replay only the first N frames")
    ap.add_argument("--score", action="store_true",
                    help="score the run against ground truth at the end (writes live_score.json)")
    ap.add_argument("--json", action="store_true",
                    help="also print the raw run record and, with --score, the score as JSON (both are always "
                         "written to the run folder)")
    ap.add_argument("--gt", default=None, help="the ground-truth file, when it is not gt.csv in the sequence folder")
    ap.add_argument("--family", default=None,
                    help="'oak': an own recording without ground truth (--score then reports no ATE instead of an "
                         "error)")
    ap.add_argument("--ext", default=None, help="the image file extension, png or jpg (default: detected)")
    ap.add_argument("--rows", default=None,
                    help="a JSON file that gives names to sequence folders (docs/SEQUENCE_FORMAT.md)")
    ap.add_argument("--pin", action="store_true",
                    help="pin the engine to fixed CPU cores, for timing runs (needs 6 or more cores; on a CPU with "
                         "efficiency cores, run without it)")
    # options of our benchmark protocol and for developers: they work, but --help does not list them
    ap.add_argument("--final", action="store_true", help=hide)
    ap.add_argument("--threads", type=int, default=2, help=hide)
    ap.add_argument("--seed", type=int, default=1, help=hide)
    ap.add_argument("--env", default="", help=hide)
    ap.add_argument("--no-x1-flags", action="store_true", help=hide)
    ap.add_argument("--reference", action="store_true", help=hide)
    ap.add_argument("--witness-ref", default=None, help=hide)
    ap.add_argument("--blackout", default=None, help=hide)
    return ap


def console_label(mode, nfb):
    """The run label printed to the console (harness.json keeps its own label)."""
    return (("paced (live)" if mode == "paced" else "lockstep (unpaced, causal)")
            + (" - diagnostic run (engine switches: %s)" % ", ".join(nfb) if nfb else ""))


def witness_stats(out):
    """The witness stages' p50 / p95 (ms) from a run's latency.json."""
    st = json.load(open(os.path.join(out, "latency.json")))["stages"]
    return {k: {"p50": float(st[k]["p50"]), "p95": float(st[k]["p95"])} for k in WITNESS_V2 if k in st}


def witness_v2(cur, ref, tol=WITNESS_TOL):
    """(b): every witness stage's p50 and p95 within +-tol of the reference.  -> dict(ok, ratio, worst)."""
    if not ref or not cur:
        return {"ok": None, "note": "no reference" if not ref else "no witness stats"}
    ratio = {}
    for k in WITNESS_V2:
        for q in ("p50", "p95"):
            if k in cur and k in ref and ref[k].get(q):
                ratio["%s_%s" % (k, q)] = cur[k][q] / ref[k][q]
    if len(ratio) < 2 * len(WITNESS_V2):
        return {"ok": None, "ratio": ratio, "note": "missing witness stages"}
    worst = max(ratio.items(), key=lambda kv: abs(kv[1] - 1.0))
    return {"ok": all(1.0 - tol <= r <= 1.0 + tol for r in ratio.values()), "ratio": ratio,
            "worst": [worst[0], worst[1]]}


def decide_validity(mode, checks, witness, reference):
    """Validity protocol v2, decided by script.  `reference`: this run IS the clean single-run reference of its row
    (judged by (a), (d), the feeder and the fence only - the witness is defined by it).  -> (valid, reasons)."""
    why = []
    if not checks.get("processes_ok"):
        why.append("process exit codes %s" % checks.get("exit_codes"))
    if checks.get("fence_errors"):
        why.append("%s fence errors" % checks.get("fence_errors"))
    if mode == "paced":
        if checks.get("feeder_valid") is not True:
            why.append("feeder jitter/slope")
        if checks.get("others_over_5pct_core"):
            why.append("non-allow-listed load %s" % checks.get("others_over_5pct_core"))
        if checks.get("other_gpu_compute_apps_not_allowed"):
            why.append("non-allowed GPU apps %s" % checks.get("other_gpu_compute_apps_not_allowed"))
        if not checks.get("pinned_v2"):
            why.append("engine tree not pinned (protocol v2 a)")
        if not reference:
            if witness is None or witness.get("ok") is not True:
                why.append("witness %s" % (None if witness is None
                                           else (witness.get("worst") or witness.get("note"))))
    return (not why), why


def growth_per_kf(series, kf_wall):
    """B6: engine-tree private-commit growth per keyframe = least-squares slope of private bytes against the keyframe
    count, over the samples after the first 25 % of keyframes (start-up allocations and pools excluded).  KB per KF."""
    import numpy as np
    if len(kf_wall) < 8 or len(series) < 8:
        return None
    kt = np.array([k[0] for k in kf_wall], float)
    kn = np.array([k[1] for k in kf_wall], float)
    st = np.array([x[0] for x in series], float)
    sp = np.array([x[1] for x in series], float)
    n_at = np.interp(st, kt, kn, left=0.0)
    m = (st >= kt[len(kt) // 4]) & (st <= kt[-1])
    if m.sum() < 8 or np.ptp(n_at[m]) < 4:
        return None
    A = np.stack([n_at[m], np.ones(int(m.sum()))], 1)
    return float(np.linalg.lstsq(A, sp[m], rcond=None)[0][0] / 1e3)


DIAG_FLAGS = {("XCLEAN_P1_ADMIT", "force_diag")}


def not_for_bars_flags(env):
    """The env entries that make a run a diagnostic run: the oracle-diagnostic flags and every
    attribution arm, i.e. any XCLEAN_P1*_OFF switch with a value (P1.1 XCLEAN_P11_OFF, P1.2 XCLEAN_P12_OFF, ...)."""
    return ["%s=%s" % (k, v) for k, v in env.items() if (k, v) in DIAG_FLAGS
            or (k.startswith("XCLEAN_P1") and k.endswith("_OFF") and v.strip())
            # P1.3 arms: the stub worker, the step application, a non-default slew fraction
            or (k == "XCLEAN_P13_WORKER" and v.strip() == "stub")
            or (k == "XCLEAN_P13_APPLY" and v.strip() not in ("", "slew"))
            or (k == "XCLEAN_P13_SLEW_F" and v.strip())
            or (k == "XCLEAN_P13_STUB_SCRIPT" and v.strip())]


def main():
    from xclean.live.feeder import feeder_main
    from xclean.live.engine import engine_main
    from xclean.live.recorder import recorder_main
    from xclean.live.protocol import FrameSlot
    from xclean.console import tolerant_output
    a = build_parser().parse_args()
    tolerant_output(for_children=True)        # a path the console cannot encode is printed with escapes, never an error
    import psutil
    if a.pin:
        try:
            pin_layout()
        except ValueError as e:
            print("xclean: %s" % e, file=sys.stderr)
            return 2
    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, REPO)
    from xclean.io import folders as FO
    from xclean.io import rows as R
    R.set_registry(a.rows)
    try:
        row = R.get(a.tag, gt=a.gt, ext=a.ext, family=a.family)
    except KeyError as e:                                    # neither a folder nor a name in the --rows file
        print("xclean: error: %s" % (FO.sequence_problem(a.tag) if not R.registry_path() else e.args[0]),
              file=sys.stderr)
        return 2
    problem = FO.sequence_problem(row.dataset, row.ext)
    if problem:
        print("xclean: error: %s" % problem, file=sys.stderr)
        return 2
    calib = json.load(open(os.path.join(row.dataset, "calib.json"), encoding="utf-8"))
    out = os.path.abspath(a.out)
    problem = FO.check_output_folder("replay", "--out", a.out, True, "replay",
                                     inputs=[row.dataset, row.gt, a.rows, a.witness_ref])
    if problem:
        print(problem, file=sys.stderr)
        return 2
    env = {} if a.no_x1_flags else dict(X1_ENV)
    env.update({"XONE_DIAL_DIST": "1", "PYTHONHASHSEED": str(a.seed)})
    for kv in [x for x in a.env.split(",") if x]:
        k, v = kv.split("=", 1)
        env[k] = v
    has_imu = os.path.isfile(os.path.join(row.dataset, "imu.jsonl"))
    print(imu_status_line(calib, has_imu, env=dict(os.environ, **env)), flush=True)
    note = imu_config_note(row.dataset, calib)
    if note:
        print(note, file=sys.stderr, flush=True)
    if os.path.isdir(out):
        shutil.rmtree(out)                                   # an earlier run of this command (checked above)
    os.makedirs(out)
    # ORACLE DIAGNOSTIC flags: a run using any of them is a diagnostic run; labelled in harness.json,
    # in a DIAGNOSTIC_RUN.txt marker in the run dir and in the printed summary
    nfb = not_for_bars_flags(env)
    blackout = [tuple(float(x) for x in b.split(":")) for b in (a.blackout or "").split(",") if b.strip()]
    if blackout:
        nfb = nfb + ["--blackout %s" % a.blackout]
    # thread caps BEFORE the children import numpy / OpenCV (spawned children inherit os.environ)
    n = str(max(1, a.threads))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = n
    os.environ["XCLEAN_CV_THREADS"] = n
    os.environ.update(env)
    vram_base = vram_used_mb()
    others_gpu = [x for x in gpu_compute_apps()]
    ctx = mp.get_context("spawn")
    W, H = int(calib["camera_width"]), int(calib["camera_height"])
    slot_name = "xclean_slot_%d" % os.getpid()
    slot = FrameSlot(slot_name, W, H, create=True)
    ctl = {"ready": ctx.Event(), "new_frame": ctx.Event(), "done": ctx.Value("q", -1), "done_evt": ctx.Event(),
           "stop": ctx.Event(), "lock": ctx.Lock()}
    imu_q = ctx.Queue()
    msg_q = ctx.Queue()
    has_imu = os.path.isfile(os.path.join(row.dataset, "imu.jsonl"))
    ecfg = {"tag": a.tag, "calib": calib, "env": env, "final": a.final, "seed": a.seed, "has_imu": has_imu,
            "mode": a.mode}
    fcfg = {"dataset": row.dataset, "ext": getattr(row, "ext", "png"), "mode": a.mode, "frames": a.frames,
            "blackout": blackout}
    # P1.3: the loop-closure WORKER process + its transport (KfRing, seqlock, queues), unless the loop
    # is off (XCLEAN_P13_OFF loop) or no worker is available (then the run is the P1.2 behaviour, recorded)
    from xclean.live import p13_link as P13L
    p13_proc = p13_owners = None
    p13_why = None
    if (os.environ.get("XCLEAN_ENGINE", "") or "").strip().lower() == "p1" and P13L.loop_on():
        okw, why = P13L.worker_available()
        if okw:
            p13h, p13_owners = P13L.make_transport(ctx, W, H, a.tag, mode=a.mode)
            ctl["p13"] = p13h
            p13_proc = P13L.start_worker(ctx, p13h, {"tag": a.tag, "calib": calib, "mode": a.mode, "out_dir": out,
                                                     "seed": a.seed, "env": env, "worker": P13L.WORKER})
        else:
            p13_why = why
            print("xclean: note: loop closure is off for this run (its process could not be started: %s)" % why)
    ecfg["p13_inactive_why"] = p13_why
    t_start = time.time()
    rec = ctx.Process(target=recorder_main, args=(msg_q, os.path.join(out, "recorder.jsonl")), name="recorder")
    eng = ctx.Process(target=engine_main, args=(ecfg, slot_name, imu_q, msg_q, ctl, out), name="engine")
    fed = ctx.Process(target=feeder_main, args=(fcfg, slot_name, imu_q, ctl, os.path.join(out, "feeder.json")),
                      name="feeder")
    rec.start()
    eng.start()
    fed.start()
    pids = {"harness": os.getpid(), "recorder": rec.pid, "engine": eng.pid, "feeder": fed.pid}
    if p13_proc is not None:
        pids["p13worker"] = p13_proc.pid
    affinity = None
    if a.pin:
        try:
            lay = pin_layout()
            eng_cpus, fed_cpus = lay["engine_cpus"], lay["feeder_cpus"]
            psutil.Process(eng.pid).cpu_affinity(eng_cpus)
            psutil.Process(rec.pid).cpu_affinity(eng_cpus)
            psutil.Process(fed.pid).cpu_affinity(fed_cpus)
            wk_cpus = None
            if p13_proc is not None:          # P1.3: the worker on its own physical core
                wk_cpus = lay["worker_cpus"]
                psutil.Process(p13_proc.pid).cpu_affinity(wk_cpus)
            affinity = {"engine_and_recorder": eng_cpus, "feeder": fed_cpus, "p13worker": wk_cpus,
                        "engine_cores": lay["engine_cores"],
                        "feeder_cores": lay["feeder_cores"],
                        "siblings_left_unused": lay["siblings_left_unused"],
                        "read_back": {"engine": psutil.Process(eng.pid).cpu_affinity(),
                                      "recorder": psutil.Process(rec.pid).cpu_affinity(),
                                      "feeder": psutil.Process(fed.pid).cpu_affinity()}}
        except Exception as e:
            print("xclean: warning: --pin could not set the CPU cores (%s); the run continues without pinning" % e)
            affinity = {"error": str(e)}
    mon = Monitor(eng.pid, pids.values(), vram_base,
                  extra_pids=[p13_proc.pid, rec.pid] if p13_proc is not None else None)
    mon.start()
    # wait for the engine; the engine ends on the end-of-stream flag, which only the feeder sets, so a feeder that dies
    # (an unreadable file in the sequence folder) would leave it waiting forever: end the stream for it, as the camera
    # demo does for its source
    eos_forced = feeder_gone = False
    while eng.is_alive():
        eng.join(0.5)
        if not feeder_gone and not fed.is_alive() and eng.is_alive():
            feeder_gone = True                # checked once; a feeder that ended normally has set the flag itself
            with ctl["lock"]:
                if not slot.hdr[4]:
                    slot.hdr[4] = 1
                    eos_forced = True
            if eos_forced:
                ctl["new_frame"].set()
    if eng.exitcode != 0:                     # the engine died: release the feeder, the recorder and the worker
        ctl["stop"].set()
        msg_q.put("EOS")
        if p13_proc is not None:
            try:
                ctl["p13"]["ctl_q"].put_nowait({"type": "eos", "n_kf": -1})
            except Exception:
                pass
    # an engine that died during start-up never set "ready", and the feeder waits for it: no point waiting long
    early_exit = eng.exitcode != 0 and not ctl["ready"].is_set()
    fed.join(5 if early_exit else 60)
    rec.join(60)
    children_stopped = []
    if eng.exitcode != 0:                     # failure path only; our own children only (never selected by name): a
        for p_ in (fed, rec):                 # live child would block this process's exit (multiprocessing joins it)
            if p_.is_alive():
                p_.terminate()
                p_.join(10)
                children_stopped.append(p_.name)
    p13_info = None
    if p13_proc is not None:                  # the worker exits on the engine's EOS message (P13 H7)
        p13_proc.join(30 if early_exit else 300)
        killed = False
        if p13_proc.is_alive():               # our own child only (never selected by name)
            p13_proc.terminate()
            p13_proc.join(10)
            killed = True
        P13L.close_transport(p13_owners)
        p13_info = {"exitcode": p13_proc.exitcode, "terminated_after_timeout": killed, "worker": P13L.WORKER}
    mon.stop.set()
    mon.join(5)
    slot.close(unlink=True)
    wall = time.time() - t_start
    feeder = json.load(open(os.path.join(out, "feeder.json"))) if os.path.isfile(os.path.join(out, "feeder.json")) \
        else {}
    esum = json.load(open(os.path.join(out, "engine_summary.json"))) \
        if os.path.isfile(os.path.join(out, "engine_summary.json")) else {}
    cpuw = mon.cpu_windows
    res = {"peak_private_gb": mon.peak_private / 1e9,
           "vram_delta_mb": (mon.peak_vram - vram_base) if vram_base is not None else None,
           "vram_base_mb": vram_base, "cpu_cores_max_10s": max(cpuw) if cpuw else None,
           "cpu_cores_mean": (sum(cpuw) / len(cpuw)) if cpuw else None, "cpu_windows": len(cpuw)}
    gpu_bad = [x for x in others_gpu if x[1].split(chr(92))[-1].lower() not in ALLOW]
    checks = {
        "protocol": "v2",
        "exit_codes": {"engine": eng.exitcode, "feeder": fed.exitcode, "recorder": rec.exitcode,
                       "p13worker": None if p13_info is None else p13_info["exitcode"]},
        "processes_ok": eng.exitcode == 0 and fed.exitcode == 0 and rec.exitcode == 0
        and (p13_info is None or (p13_info["exitcode"] == 0 and not p13_info["terminated_after_timeout"])),
        "p13": {"worker": p13_info, "inactive_why": p13_why, "loop_on": P13L.loop_on()},
        "feeder_valid": (feeder.get("validity") or {}).get("valid") if a.mode == "paced" else "n/a (lockstep)",
        "pinned_v2": bool(affinity and "error" not in affinity),
        "affinity": affinity,
        "other_gpu_compute_apps_at_start": others_gpu,
        "other_gpu_compute_apps_not_allowed": gpu_bad,
        "others_over_5pct_core": mon.others_over,
        "allow_list_cpu_pct_max": mon.allow_cpu,                     # (c) reported, not capped
        "allow_list_total_cpu_pct_max": mon.allow_sum_max,
        "fence_errors": esum.get("n_fence_errors"),
    }
    res["growth_kb_per_kf"] = growth_per_kf(mon.series, esum.get("kf_wall") or [])
    if mon.extra:                             # P1.3 B6 report: the whole tree (engine tree + worker + recorder)
        cwa = mon.cpu_windows_all
        res["whole_tree"] = {"peak_private_gb": mon.peak_private_all / 1e9,
                             "growth_kb_per_kf": growth_per_kf(mon.series_all, esum.get("kf_wall") or []),
                             "cpu_cores_mean": (sum(cwa) / len(cwa)) if cwa else None,
                             "cpu_cores_max_10s": max(cwa) if cwa else None}
    md5 = esum.get("engine_md5")
    try:
        cur_w = witness_stats(out)
    except Exception as e:
        cur_w = None
        checks["witness_error"] = str(e)
    mask_key = ",".join(str(x) for x in ((affinity or {}).get("engine_and_recorder") or []))
    key = "%s|%s|%s" % (a.tag, md5, mask_key)
    reg = json.load(open(WITNESS_REF_V2)) if os.path.isfile(WITNESS_REF_V2) else {}
    ref = None
    if a.witness_ref:
        try:
            rh = json.load(open(os.path.join(a.witness_ref, "harness.json")))
            ref = {"stats": witness_stats(a.witness_ref), "out": os.path.abspath(a.witness_ref),
                   "ref_valid": rh.get("valid"), "ref_reasons": rh.get("validity_reasons")}
            if not rh.get("valid"):
                ref = None
                checks["witness_ref_error"] = "the given reference run is not valid: %s" % rh.get("validity_reasons")
        except Exception as e:
            checks["witness_ref_error"] = str(e)
    else:
        ref = reg.get(key)
    witness = None if a.reference else witness_v2(cur_w, (ref or {}).get("stats"))
    if witness is not None:
        witness["ref_out"] = (ref or {}).get("out")
    checks["witness"] = witness if witness is not None else {"note": "this run is the reference", "stats": cur_w}
    valid, why = decide_validity(a.mode, checks, witness, a.reference)
    checks["validity_reasons"] = why
    if a.reference and valid and cur_w:
        reg[key] = {"stats": cur_w, "out": out, "set": time.strftime("%Y-%m-%d %H:%M:%S"), "engine_md5": md5,
                    "affinity": affinity, "allow_list_total_cpu_pct_max": mon.allow_sum_max,
                    "others_over_5pct_core": mon.others_over, "feeder_validity": feeder.get("validity")}
        os.makedirs(os.path.dirname(WITNESS_REF_V2), exist_ok=True)
        json.dump(reg, open(WITNESS_REF_V2, "w"), indent=1, default=str)
        checks["witness"]["registered"] = key
    if nfb:
        open(os.path.join(out, "DIAGNOSTIC_RUN.txt"), "w").write(
            "DIAGNOSTIC RUN - not comparable with the published runs: %s\n" % ", ".join(nfb))
    hj = {"tag": a.tag, "sequence_folder": os.path.abspath(row.dataset), "mode": a.mode,
          "label": console_label(a.mode, nfb),
          "not_for_bars": bool(nfb), "not_for_bars_flags": nfb,
          "pids": pids, "engine_md5": esum.get("engine_md5"), "env_flags": esum.get("env_flags"), "seed": a.seed,
          "threads_per_process": int(n), "pinned": a.pin, "wall_s": wall, "resources": res, "checks": checks,
          "valid": valid, "validity_reasons": why, "reference_run": bool(a.reference),
          "feeder_validity": feeder.get("validity"),
          "frames_released": len(feeder.get("frames", [])), "frames_replaced": len(feeder.get("replaced", []))}
    if eng.exitcode != 0:                     # failure path only (a successful run's harness.json is unchanged)
        hj["engine_failed"] = {"exitcode": eng.exitcode, "during_startup": early_exit,
                               "children_stopped": children_stopped}
    feeder_failed = eng.exitcode == 0 and (eos_forced or fed.exitcode not in (0, None))
    if feeder_failed:                         # failure path only
        hj["feeder_failed"] = {"exitcode": fed.exitcode, "end_of_stream_set_by_harness": eos_forced}
    json.dump(hj, open(os.path.join(out, "harness.json"), "w"), indent=1, default=str)
    console = {k: hj[k] for k in ("tag", "mode", "label", "valid", "wall_s", "frames_released", "frames_replaced")}
    console["label"] = console_label(a.mode, nfb)
    if a.json:                                # the raw record; harness.json always has it
        print(json.dumps(console, default=str))
        print("resources", json.dumps(res, default=str))
    elif nfb:
        print("xclean: note: this is a diagnostic run (engine switches: %s); it is not comparable with the published "
              "runs." % ", ".join(nfb), flush=True)
    if eng.exitcode != 0:
        sys.stdout.flush()
        print("xclean: the engine stopped %s (exit code %s); its error is printed above. Nothing was scored. The run "
              "folder keeps what was written: %s" % ("during start-up" if early_exit else "before the end of the stream",
                                                     eng.exitcode, out), file=sys.stderr, flush=True)
        return 1
    if feeder_failed:
        sys.stdout.flush()
        print("xclean: the frame reader stopped with an error (exit code %s); its error is printed above. The stream "
              "was ended early, so nothing was scored. The run folder keeps what was written: %s"
              % (fed.exitcode, out), file=sys.stderr, flush=True)
        return 1
    n_traj = None
    try:                                      # the pose published per frame, as `xclean live` writes it
        from xclean.live.oak_demo import write_trajectory_live
        n_traj = write_trajectory_live(out)
    except Exception as e:
        print("xclean: error: trajectory_live.csv was not written (%s: %s)" % (type(e).__name__, e), file=sys.stderr,
              flush=True)
    if n_traj is not None:                    # printed outside the try: a print can never hide a written file
        print("xclean: wrote %d poses to %s" % (n_traj, os.path.join(out, "trajectory_live.csv")), flush=True)
    sc = None
    score_error = None
    if a.score:
        from xclean.live.score_live import score_dir, score_error_message
        try:
            sc = score_dir(out, a.tag)
        except Exception as e:                # e.g. no ground-truth file: say so, without a traceback
            score_error = score_error_message(e, row)
            print("xclean: error: the run is complete, but it could not be scored: %s" % score_error,
                  file=sys.stderr, flush=True)
        if sc is not None and a.json:         # live_score.json always has it
            print(json.dumps(sc, default=str)[:2000])
    note = plain_validity_note(a.mode, checks, witness, why)
    if note:
        print(note, flush=True)
    line = None
    try:
        from xclean.live.oak_demo import summary_line
        line = summary_line(out, score=sc)
    except Exception as e:
        print("xclean: warning: the end-of-run summary could not be made (%s: %s)" % (type(e).__name__, e),
              flush=True)
    if line:
        print(line, flush=True)
    if score_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
