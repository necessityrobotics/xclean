#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""THE RUN-END GLOBAL BA ENTRY DUMP (XTEN_GBA_DUMP) - the engine port 2026-10-15.

Ported from an earlier engine's global-BA budget dump (its design, its verifier and the port spec are internal
notes).  ONLY THE DUMP is ported.  XTEN_GBA_BUDGET is NOT: in the engine the load
dependence of the run-end GBA is removed by P6's XTEN_GBA_ITERBUDGET=1 (global_ba_id ignores time_budget_s), which the
x0 base carries; XTEN_GBA_BUDGET stays an unread variable here.

WHY
    The run-end global BA is the only place XTEN_GBA_SCALECAL (C6m-imu) acts, so replaying a draw's own GBA ENTRY with
    the flag set is the paired arm at the identical entry state (zero between-draw variance, zero GPU).  The dump makes
    every saved run's run-end question answerable on CPU.

THE DUMP  (XTEN_GBA_DUMP=<absolute dir>)
    Before the solve, the tracker pickles exactly what global_ba_id is handed (post rotation-factor dedup, post
    depth-trust) to <dir>/gba_in_<basename of the run's --out>.pkl - one file per DRAW, because the queue hands every
    row the same value.  The pickle records:
      args, kwargs     the solver's positional and keyword arguments (ONE dict feeds the dump and the call, spec R3)
      kf_t             keyframe timestamps (for scoring a replay)
      budget_wall_s    the wall budget the run computed; budget_mode 'wall' | 'iterbudget' (P6) - recorded only
      env              every XTEN_* / XONE_* variable of the run (spec R2): global_ba_id reads its own flags inside the
                       solver (XTEN_GBA_BESTFIX, XTEN_GBA_ACCEPT, XTEN_GBA_SOLVER, XTEN_GBA_ITERBUDGET, XTEN_GBA_SCREEN,
                       XTEN_IF_* via preint, XTEN_GBA_DEPTH_TRUST, XONE_BETA_DEBUG), so a replay must run under the same
                       values
      blas_env         OMP / MKL / OPENBLAS thread counts (recorded only: bit-for-bit replay assumes the same)
      global_ba_md5    md5 of the solver file that ran (a dump replays bit for bit only with that solver)
      extra            the tracker's provenance (depth-trust stats, the scalecal mode)
    A dump failure NEVER fails the run (spec R4): the summary then carries {"error": ...} under global_ba.gba_dump.  A
    consumer accepts a dump only if the completed run's summary.json names that exact path (stale-artefact rule).
    Flag contract (spec R8): unset / '' / '0' / 'off' = off; anything else must be an ABSOLUTE path without a
    Windows-illegal character (<>"|?*) whose directory can be created - otherwise RuntimeError at Tracker
    construction (the verifier saw a draw die with WinError 123 at GBA entry, after the whole front end).
COST
    Off: nothing runs (one env read at construction).  On: one pickle of the GBA inputs at run end (end-of-run cost
    only, outside every frame and keyframe; recorded as global_ba.gba_dump.seconds, measured in an internal note).
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time

DUMP_ENV = "XTEN_GBA_DUMP"
_ILLEGAL = '<>"|?*'
_ENV_PREFIXES = ("XTEN_", "XONE_")
_BLAS_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def gba_dump_root(env=None):
    """None (off) or the dump directory, created now.  Anything that is not an absolute, legal, creatable path
    raises RuntimeError (spec R8)."""
    v = (os.environ.get(DUMP_ENV, "") if env is None else env)
    v = str("" if v is None else v).strip()
    if v.lower() in ("", "0", "off"):
        return None
    bad = [c for c in _ILLEGAL if c in v]
    if bad:
        raise RuntimeError("%s=%r holds a character Windows cannot put in a path (%s)" % (DUMP_ENV, v, "".join(bad)))
    if not os.path.isabs(v):
        raise RuntimeError("%s=%r is not an absolute directory (unset / off = no dump)" % (DUMP_ENV, v))
    try:
        os.makedirs(v, exist_ok=True)
    except OSError as exc:
        raise RuntimeError("%s=%r: the directory cannot be created (%s)" % (DUMP_ENV, v, exc))
    if not os.path.isdir(v):
        raise RuntimeError("%s=%r is not a directory" % (DUMP_ENV, v))
    return v


def solver_md5():
    """md5 of the global_ba.py that this process imports (the solver a dump was written by / is replayed with)."""
    from xclean.backend import global_ba as _gb
    with open(_gb.__file__, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def captured_env(environ=None):
    e = os.environ if environ is None else environ
    return {k: v for k, v in sorted(e.items()) if k.startswith(_ENV_PREFIXES)}


def dump_gba_entry(root, tag, args, kwargs, kf_t, budget_wall_s, mode, extra=None):
    """Pickle one GBA entry to <root>/gba_in_<tag>.pkl (written to a temporary name, then renamed, so a reader never
    sees half a file).  NEVER raises: returns {"path", "bytes", "seconds"} or {"error": repr(exc)}."""
    t0 = time.perf_counter()
    tmp = None
    try:
        path = os.path.join(root, "gba_in_%s.pkl" % tag)
        rec = {"args": tuple(args), "kwargs": dict(kwargs), "kf_t": kf_t,
               "budget_wall_s": float(budget_wall_s), "budget_mode": str(mode),
               "env": captured_env(), "blas_env": {k: os.environ.get(k) for k in _BLAS_VARS},
               "global_ba_md5": solver_md5(), "extra": extra}
        tmp = "%s.tmp%d" % (path, os.getpid())
        with open(tmp, "wb") as fh:
            pickle.dump(rec, fh, protocol=4)
        os.replace(tmp, path)
        return {"path": os.path.abspath(path).replace("\\", "/"), "bytes": int(os.path.getsize(path)),
                "seconds": float(time.perf_counter() - t0)}
    except Exception as exc:                       # an output-only logger must not lose a GPU run before the GBA
        try:
            if tmp is not None and os.path.isfile(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return {"error": repr(exc)}


def load_gba_entry(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)                  # a developer dump: load only dumps you wrote yourself


def replay_gba_entry(path, time_budget_s=None, solver=None, check_md5=False, kw_update=None, env_update=None):
    """Re-run the solver on a dump under EXACTLY the dump's XTEN_* / XONE_* environment (every other variable with
    those prefixes is removed for the call; the caller's environment is restored afterwards).  check_md5=True refuses
    a solver file other than the one that wrote the dump.  kw_update / env_update: a PAIRED arm - keyword
    arguments merged over the dump's (e.g. XTEN_GBA_SCALECAL's dof / scale_bias / inertial with chi_cal) and variables
    set over the dump's environment; both None = the control.  -> global_ba_id's 4-tuple."""
    z = path if isinstance(path, dict) else load_gba_entry(path)
    if solver is None:
        from xclean.backend.global_ba import global_ba_id as solver
    if check_md5 and z.get("global_ba_md5") != solver_md5():
        raise RuntimeError("the dump was written by global_ba.py md5 %s, this process imports %s"
                           % (z.get("global_ba_md5"), solver_md5()))
    kw = dict(z["kwargs"])
    if kw_update:
        kw.update(kw_update)
    saved = {k: v for k, v in os.environ.items() if k.startswith(_ENV_PREFIXES)}
    try:
        for k in saved:
            del os.environ[k]
        os.environ.update(z.get("env") or {})
        if env_update:
            os.environ.update(env_update)
        return solver(*z["args"], **kw, time_budget_s=time_budget_s, verbose=False)
    finally:
        for k in [k for k in os.environ if k.startswith(_ENV_PREFIXES)]:
            del os.environ[k]
        os.environ.update(saved)
