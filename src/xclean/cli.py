# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""The `xclean` command: a thin front door over the module entry points.

    xclean run     <sequence dir> [...]   offline runner                         = python -m xclean.run
    xclean replay  <sequence dir> [...]   the live engine on a recorded sequence = python -m xclean.live.harness
    xclean live    --out DIR [...]        live on an OAK-D-Lite camera           = python -m xclean.live.oak_demo
    xclean score   <run dir | csv> [...]  score a trajectory against GT          = python -m xclean.eval.score
    xclean audit   <sequence dir> [...]   check sequence folders before a run    = python -m xclean.eval.dataset_audit
    xclean convert <format> <in> <out>    a public dataset -> a sequence folder  = python -m xclean.io.convert
    xclean calib   <kalibr|opencv> [...]  calib.json for your own camera         = python -m xclean.io.calib_import
    xclean doctor  [--full]               check this machine                     = python -m xclean.doctor

Every argument after the subcommand is passed to that module's own parser, so `xclean <sub> --help` shows the
module's options.  One default is added, for `replay` and `live` only: the engine of the published results
(XCLEAN_ENGINE=p1, see ENGINE_DEFAULTS).  An explicit `--env XCLEAN_ENGINE=...` or an XCLEAN_ENGINE environment
variable always wins.  Both commands print the effective engine settings before they start, in plain words (only a
setting the user gave is shown by its variable name); a command line that the module cannot read is left to the
module, which prints its usage error and exits 2 before anything is printed here.
Two checks run first, for `replay` and `live` only: the input sequence folder (`replay`, `live --replay`) must look
like a sequence folder (preflight), and the --out (and --record) folder must be new, empty, or an earlier output of
the same command that --overwrite may replace (check_out_folders; the rules are in xclean/io/folders.py).
--overwrite is removed before the arguments reach the module.  Run as modules (python -m ...), the commands check the
same folder rules themselves, with --overwrite implied.
Exit codes of `replay` and `live`: 0 done; 1 the run failed; 2 the command line, the input sequence folder or an
output folder was refused before anything started (nothing was deleted).
The Python package (distribution) is named xstereo-slam (DISTRIBUTION); the import package and the command are xclean.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

# subcommand -> (module, entry function, one-line help)
COMMANDS = {
    "run": ("xclean.run", "_main",
            "offline runner on a recorded sequence (needs a CUDA GPU)"),
    "replay": ("xclean.live.harness", "main",
               "the live engine on a recorded sequence, --mode lockstep|paced (published engine settings by default)"),
    "live": ("xclean.live.oak_demo", "main",
             "live on an OAK-D-Lite camera (the only OAK model tested), or --replay a recording made with "
             "--record"),
    "score": ("xclean.eval.score", "_main",
              "score a run against ground truth: a replay or live --replay run folder, an `xclean run` output "
              "folder, or a trajectory csv (ATE after SE(3) alignment)"),
    "audit": ("xclean.eval.dataset_audit", "main",
              "audit sequence folders (timestamps, image counts, IMU, GT); exit 1 if any is flagged"),
    "convert": ("xclean.io.convert", "main",
                "convert a public dataset (euroc, kitti, 4seasons, rosario-v1, rosario-v2, madmax) to a sequence folder"),
    "calib": ("xclean.io.calib_import", "main",
              "write calib.json for your own stereo camera from a Kalibr or OpenCV calibration"),
    "doctor": ("xclean.doctor", "main",
               "check this machine (Python packages, GPU, CUDA, OAK camera) and print fixes; exit 1 on a failure"),
}

# The engine configuration of the published results, part (a) of the run configuration (engine behaviour): the live P1
# engine.  Everything else the published runs used for the engine is already the default of `xclean replay` /
# `xclean live` (the standard flag set, seed 1, 2 threads per process, XCLEAN_LAT unset, loop closure on).  The
# measurement-only settings of those runs (core pinning, timing references, NUMBA_NUM_THREADS) are not defaults.
ENGINE_DEFAULTS = {"XCLEAN_ENGINE": "p1"}
ENGINE_PREFIXES = ("XCLEAN_", "XTEN_", "XONE_")
# how the settings printout names each default, in plain words
DEFAULT_WORDS = {"XCLEAN_ENGINE": "engine: the engine of our published results"}

# the name of the Python package (pip install / pip show / importlib.metadata); the import package and the command are
# "xclean"
DISTRIBUTION = "xstereo-slam"

AUDIT_HELP = """usage: xclean audit <sequence dir> [<sequence dir> ...]

Audits each sequence folder (see docs/SEQUENCE_FORMAT.md): IMU monotonicity, gaps, rate and clipping; frame
timestamps against the left/ and right/ image counts and calib.json; ground-truth monotonicity and coverage; the
calib imu_noise block.  Prints one line per folder and exits 1 if any folder is flagged, else 0.  A folder that
does not exist, or has no calib.json or timestamps.csv, is flagged as not a sequence folder.
With no folder it audits every sub-folder of the roots in $XCLEAN_DATASET_ROOTS (os.pathsep-separated); with
neither, it prints a usage line and exits 2.
Same as: python -m xclean.eval.dataset_audit <sequence dir> ...
"""


def _version():
    """The installed version of the Python package DISTRIBUTION; from a source tree that is not installed with pip,
    the package's own __version__ with a note."""
    try:
        from importlib.metadata import version
        return version(DISTRIBUTION)
    except Exception:
        from xclean import __version__
        return "%s (not installed with pip)" % __version__


def _usage():
    w = max(len(k) for k in COMMANDS)
    lines = ["usage: xclean <command> [arguments ...]", "",
             "xclean: real-time stereo visual-inertial SLAM.", "", "commands:"]
    lines += ["  %-*s  %s" % (w, k, v[2]) for k, v in COMMANDS.items()]
    lines += ["", "xclean <command> --help shows the options of that command.",
              "xclean --version prints the installed version.",
              "Start with: xclean doctor"]
    return "\n".join(lines)


def _parse_env_value(v):
    """'K=V,K2=V2' (the harness's --env syntax) -> dict; entries without '=' are ignored here."""
    out = {}
    for kv in (v or "").split(","):
        if "=" in kv:
            k, val = kv.split("=", 1)
            out[k.strip()] = val
    return out


def engine_defaults(cmd, rest, environ=None):
    """Apply ENGINE_DEFAULTS to the argv of `replay` / `live`.  -> (new argv, settings) where settings is a list of
    (name, value, origin) for the printout: origin is 'default', '--env' or 'environment'.

    replay: XCLEAN_ENGINE=p1 is added to --env unless --env or the environment sets XCLEAN_ENGINE (the harness passes
    --env entries to the engine and the engine also inherits the environment).
    live:   the camera demo sets XCLEAN_ENGINE=p1 itself; an XCLEAN_ENGINE environment variable is forwarded through
    --env so that it wins there too (the demo's own default would otherwise replace it)."""
    environ = os.environ if environ is None else environ
    rest = list(rest)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--env", action="append", default=[])
    known, _ = p.parse_known_args(rest[:rest.index("--")] if "--" in rest else rest)
    user = _parse_env_value(known.env[-1] if known.env else "")
    add = {}
    settings = []
    for k, v in ENGINE_DEFAULTS.items():
        if k in user:
            settings.append((k, user[k], "--env"))
        elif k in environ:
            settings.append((k, environ[k], "environment"))
            if cmd == "live":
                add[k] = environ[k]
        else:
            settings.append((k, v, "default"))
            if cmd == "replay":
                add[k] = v
    for k in sorted(user):
        if k not in ENGINE_DEFAULTS:
            settings.append((k, user[k], "--env"))
    for k in sorted(environ):
        if k.startswith(ENGINE_PREFIXES) and k not in ENGINE_DEFAULTS and k not in user:
            settings.append((k, environ[k], "environment"))
    if add:
        merged = ",".join(x for x in [known.env[-1] if known.env else ""] + ["%s=%s" % kv for kv in add.items()] if x)
        cut = rest.index("--") if "--" in rest else len(rest)
        rest[cut:cut] = ["--env", merged]
    return rest, settings


OVERWRITE_HELP = ("xclean option for replay and live:\n"
                  "  --overwrite  replace the --out (or --record) folder when it holds an earlier output of the\n"
                  "               same command and nothing else. Without it, xclean refuses a folder that is not\n"
                  "               empty. xclean never replaces the current folder, your home folder, a drive root,\n"
                  "               or a folder that holds an input of the command.")
OUT_OPTIONS = {"replay": (("--out", "replay"),), "live": (("--out", "live"), ("--record", "recording"))}


def check_out_folders(cmd, rest):
    """`replay` / `live` delete their --out folder (and live its --record folder) before they start.  The folder rule
    of xclean/io/folders.py: a folder that exists and is not empty is refused unless --overwrite is given and it holds
    an earlier output of the same command and nothing else; the current folder, the home folder, a drive root and a
    folder that holds (or lies inside) an input of the command are always refused.  The option values are found by
    their POSITION in the command line, and every other token that names an existing path is an input.
    -> (argv without --overwrite, error message or None)."""
    from xclean.io.folders import check_output_folder
    rest = list(rest)
    cut = rest.index("--") if "--" in rest else len(rest)
    head, tail = rest[:cut], rest[cut:]
    overwrite = "--overwrite" in head
    head = [x for x in head if x != "--overwrite"]
    opts = OUT_OPTIONS.get(cmd, ())
    folders = {o: None for o, _ in opts}
    skip = set()
    for i, x in enumerate(head):
        for o in folders:
            if x == o and i + 1 < len(head):
                folders[o] = head[i + 1]
                skip.update((i, i + 1))
            elif x.startswith(o + "="):
                folders[o] = x.split("=", 1)[1]
                skip.add(i)
    inputs = []
    for i, x in enumerate(head):
        if i in skip:
            continue
        v = x.split("=", 1)[1] if (x.startswith("--") and "=" in x) else x
        if v and not v.startswith("-") and os.path.exists(v):
            inputs.append(v)
    for opt, kind in opts:
        if not folders[opt]:
            continue
        others = [(o, f) for o, f in folders.items() if o != opt]
        err = check_output_folder(cmd, opt, folders[opt], overwrite, kind, inputs=inputs, others=others)
        if err:
            return head + tail, err
    return head + tail, None


def preflight(cmd, rest, environ=None):
    """Check the input sequence folder of `replay` (its positional) and of `live --replay` before anything starts.
    -> a plain error message, or None (also when the command line cannot be read here: the module then reports it)."""
    environ = os.environ if environ is None else environ
    from xclean.io.folders import sequence_problem
    if cmd not in ("replay", "live"):
        return None
    head = rest[:rest.index("--")] if "--" in rest else list(rest)
    head = [x for x in head if x != "--overwrite"]
    try:
        build_parser = getattr(importlib.import_module(COMMANDS[cmd][0]), "build_parser", None)
    except ImportError:
        return None                                # main() reports a missing module
    if build_parser is None:
        return None
    import contextlib
    import io
    try:
        with contextlib.redirect_stderr(io.StringIO()):     # a command-line error is the module's to report
            a, _ = build_parser().parse_known_args(head)
    except SystemExit:
        return None
    if cmd == "replay":
        seq, ext = a.tag, a.ext
        if not os.path.exists(seq) and (a.rows or environ.get("XCLEAN_ROWS")):
            return None                            # a sequence name from a --rows file: the module resolves it
    else:
        seq, ext = a.replay, None
        if not seq:
            return None
    err = sequence_problem(seq, ext)
    return ("xclean %s: error: %s" % (cmd, err)) if err else None


def describe_settings(cmd, rest, settings):
    """The printout of the effective engine settings (plain text, one item per line).  A default is described in
    words; a setting the user gave (--env or the environment) is shown as NAME=value with where it came from."""
    lines = ["xclean %s: engine settings" % cmd]
    shown = {k for k, _, _ in settings}
    for k, v, origin in settings:
        if origin == "default":
            lines.append("  %s (default)" % DEFAULT_WORDS.get(k, "%s=%s" % (k, v)))
            continue
        how = {"--env": "from --env", "environment": "from the environment"}[origin]
        lines.append("  %s=%s  (%s)" % (k, v, how))
    if "XCLEAN_LAT" not in shown:
        lines.append("  fast paths: default (the GPU front end when a GPU is visible)")
    if "XCLEAN_P13_OFF" not in shown:
        lines.append("  loop closure: on")
    if "--no-x1-flags" in rest:
        lines.append("  standard engine flag set: OFF (--no-x1-flags)")
    else:
        lines.append("  standard engine flag set: on")
    if not any(o != "default" for _, _, o in settings):
        lines.append("  your overrides: none")
    return "\n".join(lines)


def command_line_parses(cmd, rest):
    """False when the module of `replay` / `live` would reject this command line (it then prints its own usage error
    and exits 2, and nothing should be printed before that).  True when it parses, or when it cannot be checked here."""
    head = rest[:rest.index("--")] if "--" in rest else list(rest)
    head = [x for x in head if x != "--overwrite"]
    try:
        build_parser = getattr(importlib.import_module(COMMANDS[cmd][0]), "build_parser", None)
    except ImportError:
        return True                                # main() reports a missing module
    if build_parser is None:
        return True
    import contextlib
    import io
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            build_parser().parse_args(head)        # strict, as the module parses it
    except SystemExit:
        return False
    return True


def main(argv=None):
    from xclean.console import tolerant_output
    tolerant_output()                     # a path the console cannot encode is printed with escapes, never an error
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(_usage(), file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if cmd in ("-V", "--version"):
        print("xclean %s" % _version())
        return 0
    if cmd not in COMMANDS:
        print("xclean: unknown command %r\n\n%s" % (cmd, _usage()), file=sys.stderr)
        return 2
    if cmd == "audit" and any(a in ("-h", "--help") for a in rest):
        print(AUDIT_HELP)                 # dataset_audit has no option parser: every argument is a folder
        return 0
    asks_help = any(a in ("-h", "--help") for a in rest)
    if cmd in ("replay", "live") and not asks_help and not command_line_parses(cmd, rest):
        cut = rest.index("--") if "--" in rest else len(rest)
        rest = [x for x in rest[:cut] if x != "--overwrite"] + rest[cut:]    # the module prints its usage error
    elif cmd in ("replay", "live") and not asks_help:
        err = preflight(cmd, rest)
        if err:
            print(err, file=sys.stderr)
            return 2
        rest, err = check_out_folders(cmd, rest)
        if err:
            print(err, file=sys.stderr)
            return 2
        rest, settings = engine_defaults(cmd, rest)
        print(describe_settings(cmd, rest, settings), flush=True)
    modname, fn, _ = COMMANDS[cmd]
    sys.argv = ["xclean " + cmd] + rest   # the module's argparse reads sys.argv; prog shows as "xclean <command>"
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if e.name != modname:
            raise
        print("xclean: the %r command is not available in this installation (module %s is missing)" % (cmd, modname),
              file=sys.stderr)
        return 2
    ok = False
    try:
        rc = getattr(mod, fn)()
        ok = rc in (None, 0)
    except SystemExit as e:               # argparse --help / errors and dataset_audit's sys.exit
        rc = e.code
        ok = rc in (None, 0)
    finally:
        if cmd in ("replay", "live") and not ok:
            _stop_leftover_children()
    if cmd in ("replay", "live") and asks_help and ok:
        print("\n" + OVERWRITE_HELP)
    return rc if rc is not None else 0


def _stop_leftover_children():
    """After `replay` / `live` failed (an error exit or an exception): stop helper processes of this command that are
    still alive, so the command can exit instead of waiting for them forever.  Not called after a successful run,
    which has joined all of them."""
    import multiprocessing as mp
    left = [p for p in mp.active_children() if p.is_alive()]
    for p in left:
        p.terminate()
    for p in left:
        p.join(10)
    if left:
        print("xclean: stopped %d helper process(es) that were still running: %s"
              % (len(left), ", ".join(p.name for p in left)), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
