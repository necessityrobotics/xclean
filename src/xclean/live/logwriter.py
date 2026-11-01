# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Run-log lines written off the frame path.

A LineWriter owns one background thread.  The frame loop hands it finished text lines (write() never blocks and never
touches the disk); the thread appends them to their files in the order they were handed over and keeps each file open
for the run.  close() writes everything still queued, closes the files and returns what happened (lines written,
lines lost, errors).  A file is opened with the same arguments the synchronous code used, so the bytes on disk are the
same.  A write error does not stop the run: it is recorded, and the lines of that file that follow are counted as
lost.

Why: a slow output disk (a USB drive, a busy virus scanner) can block one write for half a second; on the frame path
that stalls the engine and the live source replaces the frames that arrive meanwhile.
"""
from __future__ import annotations

import atexit
import queue
import threading

MAX_ERRORS = 20                 # error texts kept per writer (the count is always complete)


class LineWriter:
    def __init__(self, name="xclean-log-writer"):
        self._q = queue.SimpleQueue()
        self._files = {}
        self._failed = set()
        self._lock = threading.Lock()
        self._closed = False
        self.lines = 0
        self.lost = 0
        self.errors = []
        self.n_errors = 0
        self.max_backlog = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        atexit.register(self.close)          # an engine that stops on an error still writes what it queued

    # ---- frame side (cheap, never blocks)
    def open(self, path, mode="a", encoding="utf-8", buffering=-1, newline=None):
        """Open `path` now (in the writer thread), e.g. to create the file before the first line, or to use
        arguments other than write()'s default ('a', utf-8)."""
        self._put(("open", path, (mode, encoding, buffering, newline)))

    def write(self, path, text):
        self._put(("write", path, text))

    def close_file(self, path):
        self._put(("close", path, None))

    def _put(self, item):
        if self._closed:
            raise RuntimeError("LineWriter: write after close")
        self._q.put(item)
        n = self._q.qsize()
        if n > self.max_backlog:
            self.max_backlog = n

    # ---- end of run
    def close(self, timeout=None):
        """Write every queued line, close every file, stop the thread.  Safe to call more than once.
        -> {"lines", "lost_lines", "errors", "n_errors", "max_backlog", "finished"}."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._q.put(("stop", None, None))
        self._thread.join(timeout)
        return {"lines": self.lines, "lost_lines": self.lost, "errors": list(self.errors), "n_errors": self.n_errors,
                "max_backlog": self.max_backlog, "finished": not self._thread.is_alive()}

    # ---- writer thread
    def _error(self, what, path, e):
        self.n_errors += 1
        if len(self.errors) < MAX_ERRORS:
            self.errors.append("%s %s: %s: %s" % (what, path, type(e).__name__, e))

    def _file(self, path, args=("a", "utf-8", -1, None)):
        f = self._files.get(path)
        if f is None and path not in self._failed:
            mode, encoding, buffering, newline = args
            try:
                f = open(path, mode, encoding=encoding, buffering=buffering, newline=newline)
                self._files[path] = f
            except Exception as e:
                self._failed.add(path)
                self._error("open", path, e)
        return f

    def _run(self):
        while True:
            op, path, arg = self._q.get()
            if op == "stop":
                break
            if op == "open":
                self._file(path, arg)
            elif op == "write":
                f = self._file(path)
                if f is None:
                    self.lost += 1
                    continue
                try:
                    f.write(arg)
                    self.lines += 1
                except Exception as e:
                    self.lost += 1
                    self._failed.add(path)
                    self._files.pop(path, None)
                    self._error("write", path, e)
                    try:
                        f.close()
                    except Exception:
                        pass
            elif op == "close":
                self._close_one(path)
        for path in list(self._files):
            self._close_one(path)

    def _close_one(self, path):
        f = self._files.pop(path, None)
        if f is None:
            return
        try:
            f.close()
        except Exception as e:
            self._error("close", path, e)
