#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""THE RUNNER - one row in, a trajectory + a latency ledger + a map kit out.

    python -m xclean.run <sequence dir | row tag> [--frames N] [--out DIR] [--rows REGISTRY.json]
    (a row tag needs a registry file: xclean.io.rows, docs/SEQUENCE_FORMAT.md)

Everything a run emits is an artifact on disk, and the run header prints the
declared budget beside the measured value so a regression is visible in the log
without opening a tool (plan 3.4).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import time

import numpy as np

def default_out(tag):
    """The output folder when none is given: runs/<sequence name> under $XCLEAN_RUNS, else under the current folder
    (never inside the installation)."""
    base = os.environ.get("XCLEAN_RUNS") or os.path.join(os.getcwd(), "runs")
    return os.path.join(base, os.path.basename(str(tag).rstrip("/\\")))


def _now_iso():
    """Local wall-clock time, timezone-aware ISO 8601 (provenance only)."""
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(
        timespec="seconds")


def _engine_md5():
    """md5 over every *.py of this package (sorted package-relative POSIX
    paths, __pycache__ skipped): relpath NUL bytes NUL per file.  Provenance
    only."""
    pkg = os.path.dirname(os.path.abspath(__file__))
    files = []
    for root, _dirs, names in os.walk(pkg):
        if "__pycache__" in root.replace("\\", "/").split("/"):
            continue
        for nm in names:
            if nm.endswith(".py"):
                p = os.path.join(root, nm)
                files.append((os.path.relpath(p, pkg).replace("\\", "/"), p))
    h = hashlib.md5()
    for rel, p in sorted(files):
        with open(p, "rb") as f:
            h.update(rel.encode("utf-8") + b"\0" + f.read() + b"\0")
    return h.hexdigest()


def run_row(tag, frames=None, out=None, tier=1, cache_root=None, verbose=True,
            kit=True, seed=0, cfg=None):
    # P0 (2026-10-09): every cfg default below equals the CLI default (_main),
    # so a direct caller passing a partial cfg runs the production arm for
    # every key it does not name.
    # Provenance, captured once at entry; ADDED to summary.json at the end.
    prov = {"engine": "xclean", "engine_md5": _engine_md5(),
            "hash_seed": os.environ.get("PYTHONHASHSEED"),
            "env_flags": {k: os.environ[k] for k in sorted(os.environ)
                          if k.startswith(("XTEN_", "XONE_", "XCLEAN_"))},
            "host": platform.node(), "started": _now_iso()}
    import cupy as cp
    from xclean.io.dataset import Dataset
    from xclean.obs.latency import Ledger
    from xclean.stages.depth import SgmDepth
    from xclean.stages.features import Frontend
    from xclean.track.tracker import Tracker
    from xclean.kit.writer import MapKit
    from xclean.obs.build import stamp

    cfg = dict(cfg or {})
    np.random.seed(seed)
    import cv2
    cv2.setRNGSeed(int(seed))
    if os.environ.get("XCLEAN_CV_THREADS"):
        # the parallel batch runner caps OpenCV's own pool per process (with BLAS/OMP) so N engines do not
        # oversubscribe the CPU; unset = OpenCV's default (all cores)
        cv2.setNumThreads(int(os.environ["XCLEAN_CV_THREADS"]))

    ds = Dataset.open(tag,
                      raw_principal_point=bool(cfg.get("raw_pp", False)))
    out = out or default_out(tag)
    os.makedirs(out, exist_ok=True)
    if verbose:
        print("xclean run: output folder %s" % os.path.abspath(out), flush=True)
        print(ds.header())

    led = Ledger(run=tag, tier=tier,
                 sync=cp.cuda.runtime.deviceSynchronize,
                 gpu_stages=("depth", "extract", "match", "track"))
    # xclean: the dense depth is our SGM (2-13 ms/frame at the benchmark's resolutions on the 5090) and it is computed
    # live at every keyframe - no content-addressed cache, so the latency ledger measures the real cost.
    depth = SgmDepth(verbose=verbose)

    class _NoCache:
        dir, hits, misses = None, 0, 0

        def close(self):
            pass

    cache = _NoCache()

    class _LiveDepth:
        """The depth stage, wrapped so the tracker sees the engine contract (and n_hit stays a dial)."""
        def __init__(self, inner):
            self.inner = inner
            self.n_hit = 0

        def compute(self, left, right):
            return self.inner.compute(left, right)

        def stats(self):
            return self.inner.stats()

    ds._cur_index = -1
    cdepth = _LiveDepth(depth)
    fe = Frontend(ds.rig, cap=cfg.get("feature_cap", 2048), verbose=verbose)
    if cfg.get("feature_budget"):
        fe.budget = int(cfg["feature_budget"])
        fe.floor = int(cfg["feature_budget"])

    manifest = {"tag": tag, "tier": tier, "seed": seed, "config": cfg,
                "build": {k: v for k, v in
                          vars(stamp(os.path.abspath(__file__))).items()},
                "depth_key": depth.key(dataset=ds.path),
                "engine": "xclean (SGM32 + classical front end + own BoW; no learned weights)",
                "rect_digest": ds.rect_digest()}
    mk = MapKit(os.path.join(out, "kit"), ds.path, ds.rig, manifest,
                enabled=kit)
    imu = None
    if cfg.get("imu", True):
        from xclean.backend.imu import ImuSource
        # ★ DEFECT 24 (2026-08-28): this line used to read
        # `"spread" if cfg.get("bias_spread") else "stderr"` — the REFUTED
        # defect-19 rule (stderr: +4.4 % ATE avg, V2_01 +14.9 %, bias
        # estimate 3-27x worse vs EuRoC's published b_w_RS_S_*) wired live.
        # The S4 revert ("--bias-spread is the default again") changed only
        # ImuSource.__init__'s default argument, WHICH THIS CALL ALWAYS
        # OVERRODE — so every test series from 2026-08-27 ran the refuted rule.
        # Found by a code audit; the dial that proved it live is summary.imu
        # .bias_uncertainty_rule.  spread is the default; --bias-stderr is
        # the ablation arm.
        imu = ImuSource(ds, verbose=verbose,
                        bias_uncertainty=("stderr" if cfg.get("bias_stderr", False)
                                          else "spread"),
                        bias_clamp=bool(cfg.get("bias_clamp", False)))
        if not imu.available():
            imu = None
        # xclean STRICT CAUSALITY: no up-front calibrate_bias() (it read ~10 s of future IMU); the stationary-window
        # bootstrap runs causally, frame by frame, in the loop below (imu.causal_bootstrap)
    tr = Tracker(ds, fe, cdepth, led, mk, verbose=verbose,
                 backend=bool(cfg.get("backend", True)), imu=imu,
                 localmap=bool(cfg.get("localmap", True)),
                 far_band=bool(cfg.get("far_band", True)),
                 inverse_depth=bool(cfg.get("inverse_depth", True)),
                 frozen_bearing=bool(cfg.get("frozen_bearing", True)),
                 inertial=bool(cfg.get("inertial", True)),
                 ba_span=cfg.get("ba_span"),
                 stereo_obs=bool(cfg.get("stereo_obs", False)),
                 age_consensus=bool(cfg.get("age_consensus", True)),
                 scale_bias=bool(cfg.get("scale_bias", False)),
                 gba_tight_gyro=bool(cfg.get("gba_tight_gyro", False)),
                 gba_robust_gyro=bool(cfg.get("gba_robust_gyro", False)),
                 aniso_parallax=not cfg.get("no_parallax", False),
                 structure_welds=bool(cfg.get("structure_welds", True)),
                 preint_cal=bool(cfg.get("preint_cal", False)),
                 kp_refine=cfg.get("kp_refine", False),
                 vision_trust=bool(cfg.get("vision_trust", False)))
    if cfg.get("loop", True):
        from xclean.loop.place import PlaceDescriptor
        from xclean.loop.closer import LoopCloser
        tr.loop = LoopCloser(ds.rig, fe, PlaceDescriptor(verbose=verbose), tr,
                             min_gap=tr.state_window, seed=seed,
                             verbose=verbose,
                             apply_online=bool(cfg.get("loop_online", False)),
                             prox_loops=bool(cfg.get("prox_loops", True)),
                             weld_ratio_gate=bool(
                                 cfg.get("weld_ratio_gate", True)))
    if tr.ba is not None:
        from xclean.backend.ba import warmup
        t_w = time.perf_counter()
        warmup(ds.rig)
        if verbose:
            print(" [backend] kernels compiled and warmed in %.2f s (a BUILD "
                  "cost; it must not land in the solve p95)"
                  % (time.perf_counter() - t_w))

    fl = ds.frames(0, frames)
    live = bool(cfg.get("live", False))
    live_rows = []
    live_stats = {"mode": "paced" if live else "as fast as possible (causal)", "frames_in": len(fl),
                  "processed": 0, "dropped": 0, "max_backlog": 0}
    t0 = time.perf_counter()
    t_first = float(fl[0].t) if len(fl) else 0.0
    j = 0
    n = 0
    while j < len(fl):
        if live:
            # PACED LIVE MODE: frame k arrives at wall time (t_k - t_first) after the start.  When
            # processing is behind, every frame that has already arrived except the NEWEST is dropped (newest-frame
            # policy); when ahead, wait for the next arrival.
            now = time.perf_counter() - t0
            arr = float(fl[j].t) - t_first
            if arr > now:
                time.sleep(arr - now)
            now = time.perf_counter() - t0
            k = j
            while k + 1 < len(fl) and float(fl[k + 1].t) - t_first <= now:
                k += 1
            live_stats["dropped"] += k - j
            live_stats["max_backlog"] = max(live_stats["max_backlog"], k - j)
            j = k
        fr = fl[j]
        ds._cur_index = fr.index
        if imu is not None:
            imu.causal_bootstrap(fr.t)
        t_in = time.perf_counter() - t0
        state = tr.step(fr)
        t_pub = time.perf_counter() - t0
        led.frame_done(fr.index)
        # the LIVE STREAM: the pose published for this frame at publish time, never revised
        live_rows.append((float(fr.t), tr.T_wc.copy(), int(fr.index), str(state), t_pub,
                          (t_pub - (float(fr.t) - t_first)) if live else (t_pub - t_in)))
        live_stats["processed"] += 1
        j += 1
        n += 1
        if verbose and n % 250 == 0:
            el = time.perf_counter() - t0
            print("   %5d/%d  %5.1f fps  kf=%d  coast=%d  budget=%d  dropped=%d"
                  % (j, len(fl), n / el, len(tr.all_kf), tr.n_lost,
                     fe.budget, live_stats["dropped"]), flush=True)
    wall = time.perf_counter() - t0
    _write_live(os.path.join(out, "trajectory_live.csv"), live_rows, live)
    lat = np.array([r[5] for r in live_rows]) * 1e3 if live_rows else np.zeros(1)
    live_stats.update(latency_ms_p50=float(np.percentile(lat, 50)), latency_ms_p95=float(np.percentile(lat, 95)),
                      latency_ms_p99=float(np.percentile(lat, 99)), latency_ms_max=float(lat.max()),
                      latency_kind=("frame arrival -> pose published (paced)" if live else
                                    "frame processing time, step() in -> pose out (unpaced)"),
                      dropped_pct=100.0 * live_stats["dropped"] / max(len(fl), 1),
                      camera_period_ms=(1e3 * float(np.median(np.diff([f.t for f in fl]))) if len(fl) > 1
                                        else None))

    if tr.loop is not None:
        # the deferred correction: one pose graph over the whole run, applied
        # once, before the global bundle adjustment that carries the same welds
        with led.stage("loop_finalise"):
            d = tr.loop.finalise(tr, run_pgo=bool(cfg.get('pgo', True)))
        if d is not None:
            tr.loop_decisions.append(d)
            tr.n_applied += 1
            if verbose:
                print(" " + d.line()[:150])
        tr.loop.close()
    if cfg.get("global_ba", True):
        with led.stage("global_ba"):
            tr.global_refine(verbose=verbose,
                             time_budget_s=cfg.get("gba_budget"))
    tr.flush()
    if mk.enabled:
        mk.rewrite_keyframes(tr.all_kf)
        n_fr = mk.rewrite_frames(tr.all_kf)
        if verbose:
            print(" [kit] %d per-frame poses re-derived from the final "
                  "trajectory" % n_fr)
    summary = tr.summary()
    summary.update(wall_s=wall, fps=len(fl) / max(wall, 1e-9),
                   depth_cache_hits=cdepth.n_hit, live=live_stats)
    try:
        import psutil
        pr = psutil.Process()
        mi = pr.memory_info()
        summary["peak_mem"] = {"peak_wset_gb": getattr(mi, "peak_wset", 0) / 1e9,
                               "peak_pagefile_gb": getattr(mi, "peak_pagefile", 0) / 1e9}
    except Exception:
        pass
    try:
        summary.setdefault("peak_mem", {})["gpu_pool_gb"] = cp.get_default_memory_pool().total_bytes() / 1e9
    except Exception:
        pass
    mk.close(extra={"summary": summary},
             depth_index={"cache_dir": cache.dir, "hits": cache.hits,
                          "misses": cache.misses})
    cache.close()

    # the trajectory the scorer reads: 8-column TUM, keyframe and per-frame
    traj_kf = os.path.join(out, "trajectory_kf.csv")
    traj_fr = os.path.join(out, "trajectory_frames.csv")
    for src, dst in ((os.path.join(out, "kit", "trajectory_kf.csv"), traj_kf),
                     (os.path.join(out, "kit", "trajectory_frames.csv"),
                      traj_fr),
                     # the causal (streamed, pre-correction) columns
                     (os.path.join(out, "kit", "trajectory_kf_causal.csv"),
                      os.path.join(out, "trajectory_kf_causal.csv")),
                     (os.path.join(out, "kit",
                                   "trajectory_frames_causal.csv"),
                      os.path.join(out, "trajectory_frames_causal.csv"))):
        if os.path.isfile(src):
            _strip(src, dst)

    led.save(os.path.join(out, "latency.json"))
    prov["finished"] = _now_iso()
    for k, v in prov.items():          # added keys only: never overwrite
        summary.setdefault(k, v)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1,
              default=float)
    if verbose:
        print(led.header())
        print(end_line(summary, out), flush=True)
    return {"out": out, "summary": summary, "trajectory_kf": traj_kf,
            "trajectory_frames": traj_fr, "ledger": led}


def end_line(summary, out):
    """The plain line at the end of `xclean run` (summary.json holds every number; --json prints it)."""
    return ("xclean run: done: %d frames in %.1f s (%.1f frames per second). Trajectories: trajectory_kf.csv "
            "(keyframes), trajectory_frames.csv (every frame), trajectory_live.csv (as published while running); "
            "summary.json and latency.json hold the numbers. Folder: %s"
            % (int((summary.get("live") or {}).get("processed") or 0), float(summary.get("wall_s") or 0.0),
               float(summary.get("fps") or 0.0), os.path.abspath(out)))


def _write_live(path, rows, paced):
    """trajectory_live.csv: one row per PROCESSED frame, the pose as published at that moment (never revised).
    Frame convention: T_world_cam0_rectified (the tracker's T_wc), quaternion x,y,z,w - the same as the kit files."""
    from scipy.spatial.transform import Rotation as _R
    with open(path, "w", encoding="utf-8") as f:
        f.write("# frame convention: T_world_cam0_rectified (live stream: pose published at publish time, never "
                "revised; %s)\n" % ("paced against the wall clock" if paced else "unpaced, causal"))
        f.write("# t_frame,x,y,z,qx,qy,qz,qw,frame_index,state,t_publish_s,latency_ms\n")
        for t, T, i, st, tp, lat in rows:
            if not np.isfinite(T).all():
                continue
            q = _R.from_matrix(T[:3, :3]).as_quat()
            f.write("%.9f,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f,%d,%s,%.6f,%.3f\n"
                    % (t, T[0, 3], T[1, 3], T[2, 3], q[0], q[1], q[2], q[3], i, st, tp, 1e3 * lat))


def _strip(src, dst):
    """Comment lines carry the frame convention; the scorer wants 8 columns."""
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for ln in f:
            if ln.startswith("#"):
                continue
            p = ln.strip().split(",")
            if len(p) >= 8:
                g.write(",".join(p[:8]) + "\n")


DESCRIPTION = """The offline runner: one sequence folder in; a trajectory, a latency log and a map-building kit out.
It needs a CUDA GPU.  It is a separate code path from `xclean replay` / `xclean live`, which run the engine of the
published results.

  xclean run <sequence folder> --out runs\\<name> [--frames N] [--score]

The output folder gets trajectory_kf.csv (keyframes) and trajectory_frames.csv (every frame) after the end-of-run
optimisation, trajectory_live.csv (the pose published for each frame while running), latency.json, summary.json and
kit/ (the map-building kit).  Without --out the folder is runs\\<sequence name> in the current folder."""


def _main():
    hide = argparse.SUPPRESS
    ap = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("tag", metavar="sequence_folder",
                    help="the sequence folder to run (docs/SEQUENCE_FORMAT.md), or a sequence name from --rows")
    ap.add_argument("--frames", type=int, default=None, help="run only the first N frames")
    ap.add_argument("--out", default=None,
                    help="the output folder (default: runs\\<sequence name> in the current folder); files of an "
                         "earlier run in it are overwritten")
    ap.add_argument("--score", action="store_true", help="score the trajectory against ground truth at the end")
    ap.add_argument("--gt", default=None,
                    help="the ground-truth file, when it is not gt.csv in the sequence folder (--score)")
    ap.add_argument("--rows", default=None,
                    help="a JSON file that gives names to sequence folders (docs/SEQUENCE_FORMAT.md)")
    ap.add_argument("--live", action="store_true",
                    help="paced mode: frames arrive at their recorded times, and frames that arrive while the "
                         "engine is busy are dropped (latency and drop measurements)")
    ap.add_argument("--no-kit", action="store_true", help="do not write the map-building kit")
    ap.add_argument("--json", action="store_true",
                    help="also print the run summary as JSON at the end (it is always written to summary.json)")
    # development and ablation switches: they work, but --help does not list them
    ap.add_argument("--tier", type=int, default=1, help=hide)
    ap.add_argument("--seed", type=int, default=0, help=hide)
    ap.add_argument("--features", type=int, default=None, help=hide)
    ap.add_argument("--no-global-ba", action="store_true", help=hide)
    ap.add_argument("--near-band", action="store_true", help=hide)
    ap.add_argument("--no-inertial", action="store_true", help=hide)
    ap.add_argument("--free-bearing", action="store_true", help=hide)
    ap.add_argument("--xyz-landmarks", action="store_true", help=hide)
    ap.add_argument("--no-localmap", action="store_true", help=hide)
    ap.add_argument("--no-imu", action="store_true", help=hide)
    ap.add_argument("--no-pgo", action="store_true", help=hide)
    ap.add_argument("--loop-online", action="store_true", help=hide)
    ap.add_argument("--no-loop", action="store_true", help=hide)
    ap.add_argument("--no-backend", action="store_true", help=hide)
    ap.add_argument("--gba-budget", type=float, default=None, help=hide)
    ap.add_argument("--stereo-obs", action="store_true", help=hide)
    ap.add_argument("--structure-welds", dest="structure_welds", action="store_true", default=True, help=hide)
    ap.add_argument("--no-structure-welds", dest="structure_welds", action="store_false", help=hide)
    ap.add_argument("--no-parallax", action="store_true", help=hide)
    ap.add_argument("--bias-clamp", action="store_true", help=hide)
    ap.add_argument("--gba-robust-gyro", action="store_true", help=hide)
    ap.add_argument("--gba-tight-gyro", action="store_true", help=hide)
    ap.add_argument("--preint-cal", action="store_true", help=hide)
    ap.add_argument("--prox-loops", dest="prox_loops", action="store_true", default=True, help=hide)
    ap.add_argument("--no-prox-loops", dest="prox_loops", action="store_false", help=hide)
    ap.add_argument("--weld-ratio-gate", dest="weld_ratio_gate", action="store_true", default=True, help=hide)
    ap.add_argument("--no-weld-ratio-gate", dest="weld_ratio_gate", action="store_false", help=hide)
    ap.add_argument("--vision-trust", action="store_true", help=hide)
    ap.add_argument("--kp-refine-far", action="store_true", help=hide)
    ap.add_argument("--kp-refine", action="store_true", help=hide)
    ap.add_argument("--bias-stderr", action="store_true", help=hide)
    ap.add_argument("--raw-principal-point", action="store_true", help=hide)
    ap.add_argument("--scale-bias", action="store_true", help=hide)
    ap.add_argument("--age-consensus", dest="age_consensus", action="store_true", default=True, help=hide)
    ap.add_argument("--no-age-consensus", dest="age_consensus", action="store_false", help=hide)
    ap.add_argument("--ba-span", type=int, default=None, help=hide)
    a = ap.parse_args()
    from xclean.io import rows as rowreg
    rowreg.set_registry(a.rows)
    if a.gt:
        rowreg.get(a.tag, gt=a.gt)
    r = run_row(a.tag, frames=a.frames, out=a.out, tier=a.tier, seed=a.seed,
                kit=not a.no_kit, cfg={"backend": not a.no_backend,
                                       "loop": not a.no_loop,
                                       "loop_online": a.loop_online,
                                       "pgo": not a.no_pgo,
                                       "imu": not a.no_imu,
                                       "global_ba": not a.no_global_ba,
                                       "localmap": not a.no_localmap,
                                       "far_band": not a.near_band,
                                       "inverse_depth": not a.xyz_landmarks,
                                       "frozen_bearing": not a.free_bearing,
                                       "inertial": not a.no_inertial,
                                       "feature_budget": a.features,
                                       "ba_span": a.ba_span,
                                       "age_consensus": a.age_consensus,
                                       "scale_bias": a.scale_bias,
                                       "raw_pp": a.raw_principal_point,
                                       "bias_stderr": a.bias_stderr,
                                       "bias_clamp": a.bias_clamp,
                                       "no_parallax": a.no_parallax,
                                       "structure_welds": a.structure_welds,
                                       "gba_tight_gyro": a.gba_tight_gyro,
                                       "gba_robust_gyro": a.gba_robust_gyro,
                                       "preint_cal": a.preint_cal,
                                       "kp_refine": ("far" if a.kp_refine_far
                                                     else a.kp_refine),
                                       "prox_loops": a.prox_loops,
                                       "vision_trust": a.vision_trust,
                                       "weld_ratio_gate": a.weld_ratio_gate,
                                       "stereo_obs": a.stereo_obs,
                                       "live": a.live,
                                       "gba_budget": a.gba_budget})
    if a.json:
        print(json.dumps(r["summary"], default=float))
    if a.score:
        from xclean.eval.score import score
        row = rowreg.get(a.tag)
        for which, p in (("kf", r["trajectory_kf"]),
                         ("frames", r["trajectory_frames"])):
            try:
                s = score(p, row.dataset, gt_path=row.gt, tag=a.tag)
                print("  SCORE[%s]  ATE %.4f m  n=%d cov %.1f%% path %.4f"
                      % (which, s.ate_se3.rmse, s.n_assoc, 100 * s.coverage,
                         s.path_ratio))
            except Exception as e:
                print("  SCORE[%s] failed: %s: %s" % (which, type(e).__name__, e))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
