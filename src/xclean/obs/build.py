# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""The [BUILD] stamp - REUSED verbatim from `our earlier research harness/obs/build.py`.

Path + size + sha1 + mtime, printed at startup.  The reason it exists is a
whole debugging session spent analysing a log produced by a pre-fix copy of the
engine: before analysing any log, check the build line against the file you
edited.
"""
# Vendored from our earlier research harness/obs/build.py (our own earlier research harness, same company): the lines below are
# that file's lines 16-22 and 29-72 VERBATIM (imports, BuildStamp, sha1_file, stamp, stamp_many).  Its freeze-record
# helpers (HERE, FREEZE_JSON, load_freeze / record_freeze / verify_freeze: a research-tree freeze.json) are not
# vendored: nothing in xclean uses them.
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass(frozen=True)
class BuildStamp:
    path: str
    size: int
    sha1: str
    mtime: str

    @property
    def short(self) -> str:
        return self.sha1[:8]

    def line(self, label: str = "BUILD") -> str:
        return "[%s] %s  size=%d  sha1=%s  mtime=%s" % (
            label, self.path, self.size, self.sha1, self.mtime)


def sha1_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stamp(path: str) -> BuildStamp:
    """Hash one file.  Raises if it is missing — a missing engine is not a
    warning, it is the end of the run."""
    path = os.path.abspath(path)
    st = os.stat(path)
    return BuildStamp(
        path=path,
        size=st.st_size,
        sha1=sha1_file(path),
        mtime=datetime.fromtimestamp(st.st_mtime, timezone.utc)
                      .isoformat(timespec="seconds"),
    )


def stamp_many(paths) -> dict:
    return {os.path.basename(p): asdict(stamp(p)) for p in paths}

