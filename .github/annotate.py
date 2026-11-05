"""Post a log as one GitHub Actions annotation (annotations are public; job logs need a sign-in).

    python .github/annotate.py notice <title> <file>
    python .github/annotate.py pytest <title> <pytest log> <exit code>

pytest: on failure an error with the short test summary and the "E" lines of the tracebacks; on success a notice with
the last line (the counts)."""
import sys


def esc(s):
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def esc_prop(s):
    return esc(s).replace(":", "%3A").replace(",", "%2C")


def main():
    kind, title, path = sys.argv[1], esc_prop(sys.argv[2]), sys.argv[3]
    text = open(path, encoding="utf-8", errors="replace").read()
    if kind == "notice":
        print("::notice title=%s::%s" % (title, esc(text[:20000])))
        return
    rc = int(sys.argv[4])
    lines = text.splitlines()
    if rc == 0:
        print("::notice title=%s::%s" % (title, esc(lines[-1] if lines else "no output")))
        return
    summary = []
    if any("short test summary info" in ln for ln in lines):
        summary = lines[next(i for i, ln in enumerate(lines) if "short test summary info" in ln):]
    details = [ln for ln in lines if ln.startswith("E  ")][:80]
    body = "\n".join(summary + ["", "-- details --"] + details) if (summary or details) else "\n".join(lines[-60:])
    print("::error title=%s::%s" % (title, esc(body[:40000])))


if __name__ == "__main__":
    main()
