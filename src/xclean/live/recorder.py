# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""RECORDER: the only writer of the live log.  Stamps receipt (perf_counter_ns) of every engine message and appends
it to <out>/recorder.jsonl, a file the engine never opens.  A second message for the same (frame, stream) marks the
run INVALID (logged as a {"invalid": ...} record).  Ends on the "EOS" message.
The lines are written by a background thread (live/logwriter.LineWriter), so a slow output disk cannot delay the
receipt stamps; the file's bytes and line order are those of a direct write.  A write error ends the recorder with an
error after the stream (exit code 1), as a direct write did."""
from __future__ import annotations

import json

from xclean.live.logwriter import LineWriter
from xclean.live.protocol import now_ns


def recorder_main(msg_q, log_path):
    seen = set()
    n = 0
    invalid = []
    w = LineWriter(name="xclean-recorder-log")
    w.open(log_path, "w", encoding="utf-8", buffering=1 << 16)
    while True:
        m = msg_q.get()
        rx = now_ns()
        if m == "EOS":
            break
        if isinstance(m, dict) and "frame_index" in m and "stream" in m:
            key = (m["frame_index"], m["stream"])
            if key in seen:
                invalid.append(key)
                w.write(log_path, json.dumps({"invalid": "duplicate (frame, stream)", "frame_index": key[0],
                                              "stream": key[1], "rx_ns": rx}) + "\n")
            seen.add(key)
        m["rx_ns"] = rx
        w.write(log_path, json.dumps(m) + "\n")
        n += 1
    w.write(log_path, json.dumps({"recorder_end": True, "messages": n, "duplicates": len(invalid), "rx_ns": now_ns()})
            + "\n")
    info = w.close()
    if info["n_errors"] or info["lost_lines"] or not info["finished"]:
        raise RuntimeError("recorder: %s was not written completely (%d lines lost): %s"
                           % (log_path, info["lost_lines"], "; ".join(info["errors"][:3]) or "writer did not finish"))
