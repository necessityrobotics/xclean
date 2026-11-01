#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
# Vendored from score.py of our own earlier research harness (same company).  Changed for the release: this
# comment block, load_trajectory also reads a file with text columns after the eighth (trajectory_live.csv), and the
# command line (_main, UsageError, _score_target, _score_live_folder, _print_result) scores xclean's own outputs.
"""The scorer — trajectory + ground truth -> a typed, quotable result.

CONTRACT
    This module is the ONLY place a number that gets quoted is computed.  It is
    an independent implementation, not a copy of an earlier evaluation script,
    and its first acceptance test was that it reproduces every saved ATE of
    our earlier runs to 4 decimal places.  Two implementations agreeing to 1e-4 m is a real check;
    one implementation agreeing with itself is not.

WHAT IS BEING COMPARED, EXACTLY
    * Ground truth is the IMU/body frame in the world frame.  The estimator
      tracks the RECTIFIED LEFT CAMERA.  The ~7 cm lever arm is removed by
      pushing GT through `T_body_cam0`'s translation.  Rectification is a pure
      rotation about the camera centre, so R1 does not enter a POSITION error.
    * Alignment is SE(3) by default.  A stereo system fixes scale, so scoring
      it with a free scale flatters the result; Sim(3) is reported alongside as
      a DIAGNOSTIC (it is what exposed kitti_09's +15.8% metric compression).
    * Association is nearest-neighbour in time with a hard `max_dt` window.

WHY THE RESULT IS A DATACLASS AND NOT A PRINTOUT
    `score_kf.txt` is 20 lines of prose that every downstream tool re-parses
    with a regex.  A regex over prose is how a NO-SCORE row (frames: 0, exit 0)
    once passed silently.  `ScoreResult` carries the ATE *and* the evidence the
    ATE is meaningless without: association count, GT coverage, path ratio, and
    the Sim(3) scale.  `verdict_blockers()` turns those into refusals.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field

import numpy as np

# Refusal thresholds.  Class D — universal properties of a comparison, not
# tuning: a trajectory that covers a tenth of the run cannot be scored against
# the run, and 10 associated poses cannot determine a 6-DoF alignment.
MIN_ASSOCIATIONS = 30
MIN_COVERAGE = 0.50
DEFAULT_MAX_DT_S = 0.02
DEFAULT_RPE_DELTA_S = 1.0


# --------------------------------------------------------------------------- #
# small pure helpers
# --------------------------------------------------------------------------- #
def quat_to_R(qw, qx, qy, qz):
    """Batched quaternion -> rotation matrix.  Inputs are 1-D arrays."""
    qw, qx, qy, qz = (np.asarray(v, dtype=np.float64) for v in (qw, qx, qy, qz))
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    n = np.where(n > 0, n, 1.0)
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    R = np.empty((len(w), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def umeyama(src, dst, with_scale):
    """Least-squares similarity mapping src (N,3) onto dst (N,3).

    Returns (R, t, s) minimising ||s*R*src + t - dst||.  Umeyama 1991; the
    det-correction keeps R right-handed when the point set is degenerate
    (a planar trajectory reflects without it, which silently halves the error).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = (D.T @ S) / n
    U, sig, Vt = np.linalg.svd(C)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1.0
    R = U @ W @ Vt
    s = 1.0
    if with_scale:
        var_s = (S ** 2).sum() / n
        s = float((sig * np.diag(W)).sum() / var_s) if var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def associate(t_a, t_b, max_dt):
    """Nearest-neighbour association of a-times to b-times.

    Returns (ia, ib, dt) for pairs within max_dt.  b is sorted internally, so
    a ground truth that is not monotonic (some converters emit one) cannot
    silently mis-associate.
    """
    order = np.argsort(t_b)
    tb = t_b[order]
    idx = np.clip(np.searchsorted(tb, t_a), 1, len(tb) - 1)
    left, right = tb[idx - 1], tb[idx]
    pick = np.where(np.abs(t_a - left) <= np.abs(right - t_a), idx - 1, idx)
    dt = np.abs(tb[pick] - t_a)
    ok = dt <= max_dt
    return np.nonzero(ok)[0], order[pick[ok]], dt[ok]


def path_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


@dataclass(frozen=True)
class ErrStats:
    rmse: float
    mean: float
    median: float
    std: float
    p95: float
    max: float

    @staticmethod
    def of(e):
        e = np.asarray(e, dtype=np.float64)
        return ErrStats(rmse=float(np.sqrt((e ** 2).mean())), mean=float(e.mean()),
                        median=float(np.median(e)), std=float(e.std()),
                        p95=float(np.percentile(e, 95)), max=float(e.max()))


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_trajectory(path):
    """Read an estimator trajectory dump -> (t, p Nx3).

    Three formats are emitted by this line of engines and all three are
    accepted, because saved runs contain all three:
      >=19 cols  full per-frame telemetry: t at col 18, xyz at 1..3
      8 cols     optimised keyframe trajectory, TUM: t,x,y,z,qx,qy,qz,qw
      4 cols     loop-corrected per-frame path: t,x,y,z
    Anything else is an error, never a guess.
    """
    try:
        data = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64)
    except ValueError:
        # a file with text columns after the eighth (trajectory_live.csv: frame_index, state, timing): the first
        # eight columns are t,x,y,z,qx,qy,qz,qw; any other unreadable file still raises
        data = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64, usecols=range(8))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ncol = data.shape[1]
    if ncol >= 19:
        t, p = data[:, 18], data[:, 1:4]
    elif ncol >= 8 or ncol == 4:
        t, p = data[:, 0], data[:, 1:4]
    else:
        raise ValueError(
            "%s has %d columns — not a recognised trajectory format "
            "(expected 4, 8, or >=19)" % (path, ncol))
    keep = np.isfinite(t) & np.isfinite(p).all(1)
    return t[keep], p[keep], int((~keep).sum())


def load_gt(path):
    """EuRoC-style ground truth: t_ns,x,y,z,qw,qx,qy,qz.  Returns (t_ns, p, q).

    The converted KITTI/MADMAX/Rosario/4Seasons `gt.csv` files use the same
    layout by construction, which is why one loader serves every rig.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split(","))
    if not rows:
        raise ValueError("%s contains no ground-truth rows" % path)
    t_ns = np.array([int(float(r[0])) for r in rows], dtype=np.int64)
    p = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows],
                 dtype=np.float64)
    q = np.array([[float(r[4]), float(r[5]), float(r[6]), float(r[7])] for r in rows],
                 dtype=np.float64)
    return t_ns, p, q


def gt_to_camera(p_body, q_body, T_body_cam0):
    """Move ground truth from the body frame to the left camera centre."""
    t_bc = np.asarray(T_body_cam0, dtype=np.float64)[:3, 3]
    R = quat_to_R(q_body[:, 0], q_body[:, 1], q_body[:, 2], q_body[:, 3])
    return p_body + np.einsum("nij,j->ni", R, t_bc)


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #
@dataclass
class ScoreResult:
    tag: str
    sequence: str
    trajectory: str
    n_est: int
    n_gt: int
    n_assoc: int
    dropped_nonfinite: int
    coverage: float                 # associated / estimator poses
    max_dt_ms: float
    median_dt_ms: float
    t_span_s: float
    gt_path_m: float
    est_path_m: float
    path_ratio: float
    ate_se3: ErrStats
    rpe_se3: ErrStats | None
    drift_pct: float | None
    ate_sim3: ErrStats
    sim3_scale: float
    provenance: dict = field(default_factory=dict)

    @property
    def ate(self) -> float:
        """The number EuRoC papers report: SE(3)-aligned position RMSE."""
        return self.ate_se3.rmse

    def verdict_blockers(self):
        """Reasons this row must NOT be scored into a claim.  Empty == quotable.

        P6's receipt: a malaga row once produced `frames: 0`, exit 0, and a
        silent NO-SCORE.  A row that fails here fails LOUDLY.
        """
        out = []
        if self.n_assoc < MIN_ASSOCIATIONS:
            out.append("only %d associations (<%d): the alignment is not "
                       "determined" % (self.n_assoc, MIN_ASSOCIATIONS))
        if self.coverage < MIN_COVERAGE:
            out.append("ground truth covers %.1f%% of the run (<%.0f%%): the "
                       "ATE describes a fragment, not the trajectory"
                       % (100 * self.coverage, 100 * MIN_COVERAGE))
        if self.dropped_nonfinite:
            out.append("%d non-finite poses dropped: the estimator diverged "
                       "somewhere in this run" % self.dropped_nonfinite)
        if not np.isfinite(self.ate):
            out.append("ATE is not finite")
        return out

    def to_dict(self):
        d = asdict(self)
        d["ate"] = self.ate
        d["blockers"] = self.verdict_blockers()
        return d

    def line(self):
        return ("%-24s ATE %10.4f m   assoc %5d  cov %5.1f%%  path %6.4f  "
                "sim3 %7.5f" % (self.tag, self.ate, self.n_assoc,
                                100 * self.coverage, self.path_ratio,
                                self.sim3_scale))


# --------------------------------------------------------------------------- #
# the scorer
# --------------------------------------------------------------------------- #
def score(trajectory_csv, dataset_dir, gt_path=None, tag=None,
          max_dt=DEFAULT_MAX_DT_S, rpe_delta=DEFAULT_RPE_DELTA_S,
          frame="cam0") -> ScoreResult:
    """`frame` says WHICH POINT the estimator reports.

    Our engine tracks the rectified left camera, so ground truth is pushed
    through `T_body_cam0` to that optical centre — `frame="cam0"`.  Most
    published stereo-inertial systems (OKVIS2, Basalt, VINS) report the IMU
    BODY frame instead; scoring those against camera-frame ground truth adds a
    ~7 cm lever arm to their error and would flatter us by exactly that much.
    Pass `frame="body"` for them.  This is the single easiest way to publish an
    unfair bake-off, so it is an explicit argument with no default guess.
    """
    cal_path = os.path.join(dataset_dir, "calib.json")
    if not os.path.isfile(cal_path):
        raise FileNotFoundError("calib.json missing: %s" % cal_path)
    with open(cal_path) as f:
        cal = json.load(f)
    t0_ns = int(cal["t0_ns"])
    T_b_c0 = np.array(cal["T_body_cam0"], dtype=np.float64)
    gt_path = gt_path or os.path.join(dataset_dir, "gt.csv")

    t_est, p_est, dropped = load_trajectory(trajectory_csv)
    t_gt_ns, p_gt_b, q_gt = load_gt(gt_path)
    t_gt = (t_gt_ns - t0_ns) / 1e9              # the replay clock's zero
    if frame == "cam0":
        p_gt = gt_to_camera(p_gt_b, q_gt, T_b_c0)
    elif frame == "body":
        p_gt = p_gt_b
    else:
        raise ValueError("frame must be 'cam0' or 'body', got %r" % frame)

    ia, ib, dt = associate(t_est, t_gt, max_dt)
    if len(ia) < 3:
        raise ValueError(
            "%s: only %d associations within %.3f s.  est t=[%.2f,%.2f], "
            "gt t=[%.2f,%.2f] — same sequence?"
            % (trajectory_csv, len(ia), max_dt, t_est.min(), t_est.max(),
               t_gt.min(), t_gt.max()))
    te, pe, pg = t_est[ia], p_est[ia], p_gt[ib]

    R, t, _ = umeyama(pe, pg, False)
    pa = (R @ pe.T).T + t
    e_se3 = np.linalg.norm(pa - pg, axis=1)

    Rs, ts, s = umeyama(pe, pg, True)
    pas = (s * (Rs @ pe.T)).T + ts
    e_sim3 = np.linalg.norm(pas - pg, axis=1)

    # RPE over a fixed time delta, on the SE(3)-aligned path.
    j = np.searchsorted(te, te + rpe_delta)
    valid = j < len(te)
    rpe_stats, drift = None, None
    if valid.any():
        i = np.nonzero(valid)[0]
        jj = j[valid]
        d_est = np.linalg.norm(pa[jj] - pa[i], axis=1)
        d_gt = np.linalg.norm(pg[jj] - pg[i], axis=1)
        r_err = np.abs(d_est - d_gt)
        rpe_stats = ErrStats.of(r_err)
        moved = d_gt > 1e-3
        if moved.any():
            drift = float(100.0 * np.median(r_err[moved] / d_gt[moved]))

    gt_len, est_len = path_length(pg), path_length(pe)
    return ScoreResult(
        tag=tag or os.path.basename(os.path.dirname(os.path.abspath(trajectory_csv))),
        sequence=str(cal.get("dataset", "?")),
        trajectory=os.path.abspath(trajectory_csv),
        n_est=len(t_est), n_gt=len(t_gt), n_assoc=len(ia),
        dropped_nonfinite=dropped,
        coverage=len(ia) / max(len(t_est), 1),
        max_dt_ms=float(dt.max() * 1e3), median_dt_ms=float(np.median(dt) * 1e3),
        t_span_s=float(t_est.max() - t_est.min()),
        gt_path_m=gt_len, est_path_m=est_len,
        path_ratio=float(est_len / gt_len) if gt_len else float("nan"),
        ate_se3=ErrStats.of(e_se3), rpe_se3=rpe_stats, drift_pct=drift,
        ate_sim3=ErrStats.of(e_sim3), sim3_scale=float(s),
        provenance={"dataset": dataset_dir, "gt": gt_path, "frame": frame,
                    "max_dt_s": max_dt, "rpe_delta_s": rpe_delta,
                    "calib_dataset": cal.get("dataset"), "t0_ns": t0_ns},
    )


def score_run_dir(run_dir, which="kf", **kw) -> ScoreResult:
    """Score a saved run directory using its own meta.json for provenance."""
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError("no meta.json in %s — a run directory without "
                                "provenance is not scoreable" % run_dir)
    with open(meta_path) as f:
        meta = json.load(f)
    fname = {"kf": "trajectory_kf_corrected.csv",
             "corrected": "trajectory_full_corrected.csv",
             "raw": "trajectory_full.csv"}[which]
    return score(os.path.join(run_dir, fname), meta["dataset"], meta.get("gt"),
                 tag=meta.get("tag"), **kw)


class UsageError(Exception):
    """A plain message for the user of `xclean score` (no traceback)."""


def _print_result(r, as_json):
    if as_json:
        print(json.dumps(r.to_dict(), indent=1))
        return
    print(r.line())
    for b in r.verdict_blockers():
        print("  warning: " + b)


def run_source(run_dir):
    """The sequence folder a replay / live run folder was made from, from its own records: `sequence_folder` (an
    absolute path, in harness.json of `xclean replay` and demo.json of `xclean live`), else harness.json's `tag` (the
    folder as typed, `xclean replay`), else demo.json's `args.replay` (the recording of `xclean live --replay`).
    -> (folder, None) or (None, a plain reason that says what to pass)."""
    def _load(name):
        p = os.path.join(run_dir, name)
        try:
            return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None
        except (OSError, ValueError):
            return None
    hj, demo = _load("harness.json") or {}, _load("demo.json")
    named = [hj.get("sequence_folder"), hj.get("tag")]
    if demo is not None:
        named = [demo.get("sequence_folder"), (demo.get("args") or {}).get("replay")]
        if not any(named):
            return None, ("%s is a live camera run (xclean live without --replay): a camera stream has no ground "
                          "truth, so there is nothing to score it against. Its poses are in trajectory_live.csv"
                          % run_dir)
    named = [x for x in named if x]
    for x in named:
        if os.path.isdir(x):
            return x, None
    if named:
        return None, ("%s was made from %s, which is not there now (moved, or a relative path from another "
                      "folder). Give that sequence folder with --dataset <sequence folder>" % (run_dir, named[0]))
    return None, ("%s does not say which sequence folder it was made from. Give it with --dataset <sequence folder>"
                  % run_dir)


def _score_live_folder(run_dir, a):
    """A replay / live run folder: the live scorer (xclean.live.score_live), as `xclean replay --score` runs it."""
    from xclean.io import rows as R
    from xclean.live.score_live import score_dir, score_error_message
    seq = a.dataset
    if not seq:
        seq, why = run_source(run_dir)
        if not seq:
            raise UsageError(why)
    R.set_registry(None)
    row = R.get(seq, gt=a.gt, family=a.family)
    try:
        sc = score_dir(run_dir, seq)
    except Exception as e:                       # noqa: BLE001  a plain reason instead of a traceback
        raise UsageError("%s could not be scored: %s" % (run_dir, score_error_message(e, row)))
    if a.json:
        print(json.dumps(sc, indent=1, default=str))
        return 0
    b5, b2 = sc.get("B5") or {}, sc.get("B2") or {}
    if b5.get("ate") is not None:
        what = "ATE %.6f m (frame level, SE(3) aligned) over %d frames" % (b5["ate"], b5.get("frames_scored") or 0)
    elif b5.get("no_gt"):
        what = "no ground truth, so no ATE"
    else:
        what = "too few frames with ground truth for an ATE"
    print("%s: %s; %s of %s frames dropped. Scores written to %s"
          % (os.path.basename(os.path.normpath(run_dir)), what, b2.get("dropped"), b5.get("frames"),
             os.path.join(run_dir, "live_score.json")))
    return 0


def _score_target(a):
    t = a.target
    if not os.path.exists(t):
        raise UsageError("%s does not exist" % t)
    if os.path.isdir(t):
        if os.path.isfile(os.path.join(t, "recorder.jsonl")):
            return _score_live_folder(t, a)
        man = os.path.join(t, "kit", "run_manifest.json")
        if os.path.isfile(os.path.join(t, "trajectory_kf.csv")):           # an `xclean run` output folder
            ds = a.dataset or (json.load(open(man, encoding="utf-8")).get("dataset") if os.path.isfile(man) else None)
            if not ds:
                raise UsageError("give the sequence folder of this run with --dataset <sequence folder>")
            for which in ("trajectory_kf.csv", "trajectory_frames.csv"):
                p = os.path.join(t, which)
                if os.path.isfile(p):
                    _print_result(score(p, ds, a.gt, tag="%s %s" % (os.path.basename(os.path.normpath(t)),
                                                                     which[11:-4])), a.json)
            return 0
        if os.path.isfile(os.path.join(t, "meta.json")):
            _print_result(score_run_dir(t, a.which), a.json)
            return 0
        raise UsageError("%s is not a run folder: it has no recorder.jsonl (a replay or live run) and no "
                         "trajectory_kf.csv (an `xclean run` output)" % t)
    folder = os.path.dirname(os.path.abspath(t))
    if not a.dataset and os.path.isfile(os.path.join(folder, "recorder.jsonl")):
        print("xclean score: scoring the run folder %s (the same poses as %s)" % (folder, os.path.basename(t)),
              flush=True)
        return _score_live_folder(folder, a)
    if not a.dataset:
        raise UsageError("give the sequence folder the trajectory was made from with --dataset <sequence folder> "
                         "(it holds calib.json, and gt.csv unless you give --gt)")
    _print_result(score(t, a.dataset, a.gt), a.json)
    return 0


def _main():
    import argparse
    import sys
    ap = argparse.ArgumentParser(
        description="Score a trajectory against ground truth: the position error after a best-fit rigid "
                    "alignment (ATE, SE(3)). The target is a replay or live --replay run folder (scored as `xclean "
                    "replay --score` scores it, written to live_score.json), an `xclean run` output folder, or a "
                    "trajectory csv (t,x,y,z,qx,qy,qz,qw first, on the sequence's time base). A live camera run has "
                    "no ground truth to score against.", allow_abbrev=False)
    ap.add_argument("target", help="a run folder or a trajectory csv")
    ap.add_argument("--dataset", default=None,
                    help="the sequence folder the trajectory was made from (needed for a csv; a replay or live "
                         "--replay run folder records it)")
    ap.add_argument("--gt", default=None, help="the ground-truth file, when it is not gt.csv in the sequence folder")
    ap.add_argument("--family", default=None,
                    help="'oak': an own recording without ground truth (reports no ATE instead of an error)")
    ap.add_argument("--json", action="store_true", help="print the whole result as JSON")
    ap.add_argument("--which", default="kf", choices=["kf", "corrected", "raw"], help=argparse.SUPPRESS)
    a = ap.parse_args()
    try:
        return _score_target(a)
    except UsageError as e:
        print("xclean score: error: %s" % e, file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, IndexError) as e:
        print("xclean score: error: could not score %s (%s: %s)" % (a.target, type(e).__name__, e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
