# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Convert one sequence of a public dataset into an xclean sequence folder (docs/SEQUENCE_FORMAT.md).

    xclean convert <format> <input> <output> [options]
    python -m xclean.io.convert <format> <input> <output> [options]

formats:
    euroc        EuRoC MAV, ASL folder layout (the folder that contains mav0/)
    kitti        KITTI odometry, grayscale stereo (no IMU)
    4seasons     4Seasons, undistorted stereo + IMU + GNSS reference poses
    rosario-v1   Rosario agricultural dataset, version 1 raw files
    rosario-v2   Rosario agricultural dataset, version 2 ROS bag (needs: pip install rosbags pillow)
    madmax       MADMAX Mars-analog dataset (DLR), rectified stereo + IMU + GNSS/INS ground truth

`xclean convert <format> --help` describes the input folder that format expects.

The folder written is the one our published evaluation read: same frame pairing, same timestamps, same IMU and
ground-truth arithmetic, same calib.json, and the source images themselves (hard-linked when the output is on the
same drive as the input, copied otherwise; rosario-v2 decodes its images from the bag and writes them as PNG).
Text files keep the line endings of the evaluated folders on every operating system (CRLF, except EuRoC's
timestamps.csv, which is LF); every xclean reader accepts either line ending.

The output folder must be empty or not exist yet. --force replaces an earlier conversion in that folder (it removes
only the files a conversion writes, and refuses if the folder holds anything else).
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys

# format name -> module that implements it (each module: HELP, DESCRIPTION, INPUT_HELP, add_arguments, run)
FORMATS = {
    "euroc": "xclean.io.convert_euroc",
    "kitti": "xclean.io.convert_kitti",
    "4seasons": "xclean.io.convert_fourseasons",
    "rosario-v1": "xclean.io.convert_rosario",
    "rosario-v2": "xclean.io.convert_rosario2",
    "madmax": "xclean.io.convert_madmax",
}

# what a conversion writes into its output folder (right_tmp/ only transiently, by rosario-v2)
OUTPUT_NAMES = ("left", "right", "right_tmp", "calib.json", "timestamps.csv", "imu.jsonl", "gt.csv")

CRLF = "\r\n"


class ConvertError(Exception):
    """A plain-English problem with the input or output folder (printed without a traceback)."""


class OutputRefused(ConvertError):
    """The output folder cannot be used (it holds the input, is a file, or holds other files): nothing was written or
    deleted, and the exit code is 2, as for the other commands' refusals."""


# ------------------------------------------------------------------------------------------------ shared helpers
def open_text(path, newline=CRLF):
    """Open a text file for writing with fixed line endings (CRLF unless told otherwise), so the bytes do not depend
    on the operating system."""
    return open(path, "w", encoding="utf-8", newline=newline)


def link_or_copy(src, dst, mode="auto"):
    """Materialise dst from src: a hard link (zero extra disk space) when mode is auto or link and the two paths are
    on the same drive, else a copy.  Returns "link" or "copy"."""
    if os.path.exists(dst):
        os.remove(dst)
    if not os.path.isfile(src):
        raise ConvertError("missing source file %s" % src)
    if mode in ("auto", "link"):
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            if mode == "link":
                raise ConvertError("cannot hard-link %s into the output folder (different drive?); "
                                   "use --mode copy" % src)
    shutil.copy2(src, dst)
    return "copy"


def need_dir(path, what):
    if not os.path.isdir(path):
        raise ConvertError("%s not found: %s" % (what, path))
    return path


def need_file(path, what):
    if not os.path.isfile(path):
        raise ConvertError("%s not found: %s" % (what, path))
    return path


def add_mode_argument(p):
    p.add_argument("--mode", choices=("auto", "link", "copy"), default="auto",
                   help="how to put the source images into the output: auto = hard link when the output is on the "
                        "same drive as the input, copy otherwise (default); link = hard link or fail; copy = always "
                        "copy. The image bytes are the same either way.")


def prepare_output(out_dir, force, inputs=()):
    """Make out_dir ready for a fresh conversion: create it, or check it is empty, or (force) remove an earlier
    conversion from it.  Never removes anything a conversion does not write."""
    out_abs = os.path.normcase(os.path.abspath(out_dir))
    for p in inputs:
        p_abs = os.path.normcase(os.path.abspath(p))
        if out_abs == p_abs or p_abs.startswith(out_abs.rstrip("\\/") + os.sep):
            raise OutputRefused("the output folder %s is (or contains) the input %s. Nothing was deleted. Choose "
                                "another output folder" % (out_dir, p))
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        raise OutputRefused("the output path %s exists and is not a folder. Nothing was deleted" % out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        return
    names = sorted(os.listdir(out_dir))
    if not names:
        return
    if not force:
        raise OutputRefused("the output folder %s is not empty. Nothing was deleted. Choose an empty folder, or pass "
                            "--force to replace an earlier conversion in it" % out_dir)
    foreign = [n for n in names if n not in OUTPUT_NAMES]
    for sub in ("left", "right", "right_tmp"):
        d = os.path.join(out_dir, sub)
        if os.path.isdir(d) and any(os.path.isdir(os.path.join(d, n)) for n in os.listdir(d)):
            foreign.append(sub + "/ (holds sub-folders)")
    if foreign:
        raise OutputRefused("--force only replaces an earlier conversion, but %s also holds: %s. Nothing was "
                            "deleted. Remove those yourself or choose another output folder"
                            % (out_dir, ", ".join(foreign)))
    for n in names:
        p = os.path.join(out_dir, n)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


# ------------------------------------------------------------------------------------------------ calibration YAML
def read_kalibr_yaml(path):
    """Read the small YAML subset that Kalibr-style calibration files use: nested mappings by indentation, block
    lists ("- item", also at the parent key's indentation), flow lists ([a, b, [c, d]]), plain or quoted scalars and
    "#" comments.  Numbers become int or float (a token float() accepts is a float), everything else a string.

    This avoids a YAML dependency.  For the values the converters read (numbers and lists of numbers) it gives
    exactly what a full YAML parser gives once those values are turned into floats."""
    lines = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for no, raw in enumerate(f, 1):
            text = _strip_comment(raw.rstrip("\r\n"))
            if not text.strip() or text.lstrip().startswith(("---", "...", "%")):
                continue
            if "\t" in text[:len(text) - len(text.lstrip())]:
                raise ConvertError("%s line %d: tab indentation is not supported" % (path, no))
            ind = len(text) - len(text.lstrip(" "))
            lines.append((ind, text.strip(), no))
    if not lines:
        return {}
    try:
        out, i = _yaml_block(lines, 0, lines[0][0])
    except (ValueError, IndexError) as e:
        raise ConvertError("%s: cannot read this calibration file (%s)" % (path, e))
    if i != len(lines):
        raise ConvertError("%s line %d: unexpected indentation" % (path, lines[i][2]))
    return out


def _strip_comment(s):
    depth, quote = 0, None
    for k, ch in enumerate(s):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "#" and depth <= 0 and (k == 0 or s[k - 1] in " \t"):
            return s[:k].rstrip()
    return s


def _is_item(text):
    return text == "-" or text.startswith("- ")


def _yaml_block(lines, i, indent):
    out = {}
    while i < len(lines):
        ind, text, no = lines[i]
        if ind < indent or (ind == indent and _is_item(text)):
            break
        if ind > indent:
            raise ValueError("line %d: unexpected indentation" % no)
        key, sep, rest = text.partition(":")
        if not sep or (rest and rest[0] not in " \t"):
            raise ValueError("line %d: expected 'key: value'" % no)
        key, rest = _unquote(key.strip()), rest.strip()
        i += 1
        if rest:
            out[key] = _yaml_value(rest)
            continue
        if i < len(lines) and _is_item(lines[i][1]) and lines[i][0] >= indent:
            out[key], i = _yaml_list(lines, i, lines[i][0])
        elif i < len(lines) and lines[i][0] > indent:
            out[key], i = _yaml_block(lines, i, lines[i][0])
        else:
            out[key] = None
    return out, i


def _yaml_list(lines, i, indent):
    items = []
    while i < len(lines):
        ind, text, no = lines[i]
        if ind != indent or not _is_item(text):
            break
        rest = text[1:].strip()
        if ":" in rest and not rest.startswith(("[", "'", '"')):
            raise ValueError("line %d: mappings inside lists are not supported" % no)
        items.append(_yaml_value(rest))
        i += 1
    return items, i


def _yaml_value(s):
    s = s.strip()
    if s.startswith("["):
        val, k = _flow_list(s, 0)
        if s[k:].strip():
            raise ValueError("trailing text after a list: %r" % s)
        return val
    return _scalar(s)


def _flow_list(s, k):
    assert s[k] == "["
    k += 1
    items, tok = [], ""
    while k < len(s):
        ch = s[k]
        if ch == "[":
            val, k = _flow_list(s, k)
            items.append(val)
            tok = None                          # the nested list stands for this item
            continue
        if ch in ",]":
            if tok is not None and tok.strip():
                items.append(_scalar(tok))
            tok = ""
            k += 1
            if ch == "]":
                return items, k
            continue
        if tok is None:
            if ch.strip():
                raise ValueError("unexpected %r after a nested list" % ch)
        else:
            tok += ch
        k += 1
    raise ValueError("unclosed list: %r" % s)


def _unquote(s):
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def _scalar(s):
    s = s.strip()
    if s in ("", "~", "null", "Null", "NULL"):
        return None
    if s in ("true", "True", "TRUE"):
        return True
    if s in ("false", "False", "FALSE"):
        return False
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


# ------------------------------------------------------------------------------------------------ command line
def build_parser():
    ap = argparse.ArgumentParser(
        prog="xclean convert",
        description="Convert one sequence of a public dataset into an xclean sequence folder (the layout in\n"
                    "docs/SEQUENCE_FORMAT.md):\n\n"
                    "  xclean convert <format> <input> <output> [options]\n\n"
                    "<input> is the dataset as downloaded and unpacked; <output> is created (it must be empty if\n"
                    "it exists).  Then check the folder with `xclean audit <output>` and run it with\n"
                    "`xclean replay <output> --out <run folder> --mode lockstep`.",
        epilog="Run 'xclean convert <format> --help' for the input folder each format expects, with an example.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="format", metavar="<format>", help="the dataset the input comes from, one of:")
    for name, modname in FORMATS.items():
        mod = importlib.import_module(modname)
        p = sub.add_parser(name, help=mod.HELP, description=mod.DESCRIPTION,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument("input", help=mod.INPUT_HELP)
        p.add_argument("output", help="the sequence folder to write; it is created, and must be empty if it exists")
        p.add_argument("--force", action="store_true",
                       help="replace an earlier conversion in the output folder (removes only left/, right/, "
                            "calib.json, timestamps.csv, imu.jsonl and gt.csv; refuses if anything else is there)")
        mod.add_arguments(p)
    return ap


def main(argv=None):
    """Entry point of `xclean convert`.  argv: the arguments after "convert" (default: sys.argv[1:]).  Returns the
    exit code: 0 done, 1 the conversion failed (the message says why), 2 bad command line or an output folder that
    is refused (nothing was deleted)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:                          # --help (0) or a command-line error (2)
        return e.code if isinstance(e.code, int) else 2
    if not args.format:
        ap.print_help(sys.stderr)
        return 2
    mod = importlib.import_module(FORMATS[args.format])
    try:
        info = mod.run(args)
    except ConvertError as e:
        print("xclean convert %s: error: %s" % (args.format, e), file=sys.stderr)
        return 2 if isinstance(e, OutputRefused) else 1
    except (OSError, ValueError, KeyError, IndexError) as e:       # a file that is not what the format says
        print("xclean convert %s: error: could not convert %s (%s: %s); is it the input this format expects? "
              "See: xclean convert %s --help" % (args.format, args.input, type(e).__name__, e, args.format),
              file=sys.stderr)
        return 1
    print("xclean convert %s: wrote %s" % (args.format, args.output))
    for k, v in info.items():
        print("  %-16s %s" % (k, v))
    print("next: xclean audit %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
