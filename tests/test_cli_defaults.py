# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""`xclean replay` / `xclean live` default to the engine of the published results (XCLEAN_ENGINE=p1); an explicit
--env or environment variable wins; the effective settings are printed; the new subcommands are wired."""
import argparse
import subprocess
import sys
import types

import pytest

from xclean import cli


def _effective_env(argv):
    """The --env value the harness / camera demo will use: their argparse keeps the LAST --env."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--env", default="")
    known, _ = p.parse_known_args(argv[:argv.index("--")] if "--" in argv else argv)
    return cli._parse_env_value(known.env)


def test_replay_adds_the_published_engine_when_nothing_is_set():
    argv, settings = cli.engine_defaults("replay", ["seq", "--out", "o", "--mode", "paced"], environ={})
    assert argv == ["seq", "--out", "o", "--mode", "paced", "--env", "XCLEAN_ENGINE=p1"]   # = the published command
    assert settings == [("XCLEAN_ENGINE", "p1", "default")]


def test_replay_keeps_the_users_other_env_entries():
    argv, _ = cli.engine_defaults("replay", ["seq", "--env", "XCLEAN_LAT=none", "--out", "o"], environ={})
    assert _effective_env(argv) == {"XCLEAN_LAT": "none", "XCLEAN_ENGINE": "p1"}


@pytest.mark.parametrize("form", [["--env", "XCLEAN_ENGINE=other"], ["--env=XCLEAN_ENGINE=other"]])
def test_explicit_env_wins(form):
    argv, settings = cli.engine_defaults("replay", ["seq"] + form, environ={})
    assert argv == ["seq"] + form
    assert ("XCLEAN_ENGINE", "other", "--env") in settings


def test_environment_variable_wins():
    env = {"XCLEAN_ENGINE": "other", "XCLEAN_P13_OFF": "loop", "PATH": "x"}
    argv, settings = cli.engine_defaults("replay", ["seq"], environ=env)
    assert argv == ["seq"]                          # the harness and its engine read the environment themselves
    assert ("XCLEAN_ENGINE", "other", "environment") in settings
    assert ("XCLEAN_P13_OFF", "loop", "environment") in settings
    assert all(k != "PATH" for k, _, _ in settings)
    # the camera demo sets its own default inside its --env handling: the variable is forwarded so it still wins
    argv, _ = cli.engine_defaults("live", ["--out", "o"], environ=env)
    assert _effective_env(argv)["XCLEAN_ENGINE"] == "other"


def test_live_needs_no_change_by_default():
    argv, settings = cli.engine_defaults("live", ["--out", "o", "--duration", "10"], environ={})
    assert argv == ["--out", "o", "--duration", "10"]
    assert settings == [("XCLEAN_ENGINE", "p1", "default")]


def test_double_dash_is_respected():
    argv, _ = cli.engine_defaults("replay", ["seq", "--", "x"], environ={})
    assert argv == ["seq", "--env", "XCLEAN_ENGINE=p1", "--", "x"]


def test_settings_printout():
    _, settings = cli.engine_defaults("replay", ["seq"], environ={})
    text = cli.describe_settings("replay", ["seq"], settings)
    assert "engine: the engine of our published results (default)" in text and "your overrides: none" in text
    assert "loop closure: on" in text and "XCLEAN_" not in text           # defaults in plain words, no variable names
    _, settings = cli.engine_defaults("replay", ["seq", "--env", "XCLEAN_LAT=none"], environ={})
    text = cli.describe_settings("replay", ["seq", "--no-x1-flags"], settings)
    assert "XCLEAN_LAT=none  (from --env)" in text and "OFF" in text and "your overrides: none" not in text


def test_main_passes_the_default_to_the_module(monkeypatch, capsys):
    seen = {}
    fake = types.ModuleType("fake_harness")

    def fake_main():
        seen["argv"] = list(sys.argv)
        return 0

    fake.main = fake_main
    monkeypatch.setitem(sys.modules, "fake_harness", fake)
    monkeypatch.setitem(cli.COMMANDS, "replay", ("fake_harness", "main", "test"))
    monkeypatch.delenv("XCLEAN_ENGINE", raising=False)
    assert cli.main(["replay", "seq", "--out", "o"]) == 0
    assert seen["argv"] == ["xclean replay", "seq", "--out", "o", "--env", "XCLEAN_ENGINE=p1"]
    assert "engine: the engine of our published results (default)" in capsys.readouterr().out


def test_help_is_not_changed(monkeypatch, capsys):
    seen = {}
    fake = types.ModuleType("fake_harness2")
    fake.main = lambda: seen.setdefault("argv", list(sys.argv)) and 0
    monkeypatch.setitem(sys.modules, "fake_harness2", fake)
    monkeypatch.setitem(cli.COMMANDS, "replay", ("fake_harness2", "main", "test"))
    cli.main(["replay", "--help"])
    assert seen["argv"] == ["xclean replay", "--help"] and "engine settings" not in capsys.readouterr().out


def test_new_commands_are_listed():
    for sub in ("convert", "calib", "doctor"):
        assert sub in cli.COMMANDS
    assert cli.COMMANDS["convert"][:2] == ("xclean.io.convert", "main")
    assert "euroc, kitti, 4seasons, rosario-v1, rosario-v2, madmax" in cli.COMMANDS["convert"][2]


def test_missing_module_gives_a_plain_message(monkeypatch, capsys):
    monkeypatch.setitem(cli.COMMANDS, "convert", ("xclean.io.no_such_module", "main", "test"))
    assert cli.main(["convert", "--help"]) == 2
    assert "not available in this installation" in capsys.readouterr().err


def test_cli_subprocess_doctor_and_calib_help():
    for sub in ("doctor", "calib"):
        r = subprocess.run([sys.executable, "-m", "xclean.cli", sub, "--help"], capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.startswith("usage: xclean %s" % sub), r.stderr


def test_leftover_children_are_stopped(capsys):
    import multiprocessing as mp
    import time
    p = mp.get_context("spawn").Process(target=time.sleep, args=(120,), name="stuck-helper")
    p.start()
    t0 = time.time()
    cli._stop_leftover_children()
    assert not p.is_alive() and time.time() - t0 < 30
    assert "stuck-helper" in capsys.readouterr().err
    cli._stop_leftover_children()                 # nothing left: silent no-op
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("result, cleaned", [(0, False), (None, False), (1, True), (SystemExit(0), False),
                                             (SystemExit(2), True), (RuntimeError("boom"), True)])
def test_leftover_children_are_stopped_only_after_a_failure(monkeypatch, result, cleaned):
    calls = []
    fake = types.ModuleType("fake_harness3")

    def fake_main():
        if isinstance(result, BaseException):
            raise result
        return result
    fake.main = fake_main
    monkeypatch.setitem(sys.modules, "fake_harness3", fake)
    monkeypatch.setitem(cli.COMMANDS, "replay", ("fake_harness3", "main", "test"))
    monkeypatch.setattr(cli, "_stop_leftover_children", lambda: calls.append(1))
    if isinstance(result, RuntimeError):
        with pytest.raises(RuntimeError):
            cli.main(["replay", "seq", "--out", "o"])
    else:
        cli.main(["replay", "seq", "--out", "o"])
    assert bool(calls) == cleaned
