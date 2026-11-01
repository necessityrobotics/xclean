# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Console output that does not fail on a letter the console cannot show.

When the output of a command goes to a file or a pipe, Windows Python writes it in the system code page (cp1252, for
example), which cannot encode every letter of a folder name.  A print of such a path then raises UnicodeEncodeError,
after the work is done.  tolerant_output() keeps the encoding and writes those letters as backslash escapes (\\u0921)
instead.  With for_children=True it also sets PYTHONIOENCODING (error handler only, the encoding is unchanged) in this
process's environment, so the helper processes that a replay or live run starts write their output the same way.
"""
from __future__ import annotations

import os
import sys


def tolerant_output(for_children=False, environ=None):
    """Make sys.stdout and sys.stderr write unencodable characters as backslash escapes.  Never raises."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except Exception:                       # noqa: BLE001  not a text stream (or none at all): leave it
            pass
    if for_children:
        environ = os.environ if environ is None else environ
        current = environ.get("PYTHONIOENCODING") or ""
        if ":" not in current:                  # an error handler the user chose is kept
            environ["PYTHONIOENCODING"] = current + ":backslashreplace"
