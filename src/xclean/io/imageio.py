# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Image files on any path.

OpenCV's imread / imwrite cannot open a path that holds non-ASCII characters on Windows (imread returns None, imwrite
returns False).  Here NumPy reads and writes the file bytes and OpenCV decodes / encodes them in memory.  The decoder
and encoder are the ones imread / imwrite use, so the pixels read and the bytes written are the same as theirs.
"""
from __future__ import annotations

import os

import numpy as np


def imread(path, flags):
    """cv2.imread(path, flags) for any path: the decoded image, or None when the file is missing or unreadable."""
    import cv2
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite(path, img, params=()):
    """cv2.imwrite(path, img, params) for any path.  The format follows the file extension.  Returns True when the
    file was written, False when OpenCV cannot encode the image; a file-system error raises OSError."""
    import cv2
    ext = os.path.splitext(str(path))[1]
    ok, buf = cv2.imencode(ext, img, list(params))
    if not ok:
        return False
    buf.tofile(str(path))
    return True
