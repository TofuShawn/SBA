# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Contract tests for the run_round bridge (Task 8): find_exe probe order,
resolve_engine arbitration, and run_round's spawn/event/watchdog behavior.

Everything is platform-neutral: a FakePopen stands in for the exe, so no
native build is needed and nothing here spawns a real process.
"""

import inspect
import io
import json
import os
import subprocess
import threading
import time

import rustselfplay
from config import MP_GAME_TIMEOUT


class FakePopen:
    """Popen replacement: canned stdout lines, optional delay and crash.

    Only the surface run_round touches is implemented: stdout (an iterable
    of str lines), stderr (a real file-like object — run_round's drain
    thread reads it in chunks), kill(), wait(). The exit code materializes
    when the stdout stream drains to its end (clean exit 0, or a nonzero
    exit after a truncated stream when crash_after is set) or when kill()
    fires (the watchdog path).
    """

    def __init__(self, lines, crash_after=None, delay=0.0, stderr_text="boom at game 3"):
        self._lines = [json.dumps(ev) if isinstance(ev, dict) else ev for ev in lines]
        self._crash_after = crash_after  # lines delivered before the process dies
        self._delay = delay  # seconds before each line arrives
        self.returncode = None
        self.killed = False
        self.stderr = io.StringIO(stderr_text)

    def stream(self):
        for i, line in enumerate(self._lines):
            if self._delay:
                time.sleep(self._delay)
            if self._crash_after is not None and i >= self._crash_after:
                break  # stdout ends early: the exe died mid-round
            yield line + "\n"
        if self.returncode is None:  # a killed process cannot die twice
            # stream drained -> the process is gone and its exit code is known
            self.returncode = 1 if self._crash_after is not None else 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def _popen_factory(lines, crash_after=None, delay=0.0, stderr_text="boom at game 3", procs=None):
    def factory(cmd, **kwargs):
        p = FakePopen(lines, crash_after, delay, stderr_text)
        p.stdout = p.stream()
        p.cmd = cmd
        p.kwargs = kwargs
        if procs is not None:
            procs.append(p)
        return p

    return factory


def _round_kwargs(**over):
    kw = dict(
        games=2,
        sims=8,
        threads=1,
        seed_base=0,
        temp_start=1.0,
        temp_end=1.0,
    )
    kw.update(over)
    return kw


# -----------------------------------------------------------------------
# run_round: event stream
# -----------------------------------------------------------------------


def test_run_round_parses_events(tmp_path):
    events = [
        {"type": "start", "games": 2},
        {"type": "game_done", "game": 0, "moves": 31, "result": "X", "temp": 1.0, "elapsed_s": 0.4},
        {"type": "game_done", "game": 1, "moves": 28, "result": "D", "temp": 1.0, "elapsed_s": 0.8},
    ]
    called = []
    res = rustselfplay.run_round(
        "fake-exe",
        "w.safetensors",
        str(tmp_path),
        on_event=called.append,
        popen=_popen_factory(events),
        **_round_kwargs(),
    )
    assert res["games_done"] == 2 and res["crashed"] is False
    assert res["stderr_tail"] == ""
    assert len(called) == 1  # game_done events are collected, not forwarded
    assert called[-1] == {"type": "selfplay_round", "games_done": 2, "crashed": False}


def test_run_round_counts_only_game_done_events(tmp_path):
    lines = [
        {"type": "start", "games": 2},
        {"type": "batch_stats", "batches": 3, "items": 200},
        {"type": "future_thing", "x": 1},  # unknown types are ignored, not counted
        {"type": "game_done", "game": 0, "moves": 9, "result": "X"},
        {"type": "game_done", "game": 1, "moves": 11, "result": "D"},
    ]
    # clean exit with the expected game count
    res = rustselfplay.run_round(
        "fake-exe", "w.safetensors", str(tmp_path), popen=_popen_factory(lines), **_round_kwargs()
    )
    assert res["games_done"] == 2 and res["crashed"] is False
    # same stream but only 2 of 4 games delivered -> short count crashes
    res = rustselfplay.run_round(
        "fake-exe",
        "w.safetensors",
        str(tmp_path),
        popen=_popen_factory(lines),
        **_round_kwargs(games=4),
    )
    assert res["games_done"] == 2 and res["crashed"] is True  # clean exit, short count
    assert res["stderr_tail"] == ""


def test_run_round_detects_crash(tmp_path):
    events = [
        {"type": "start", "games": 3},
        {"type": "game_done", "game": 0, "moves": 5, "result": "X", "temp": 1.0, "elapsed_s": 0.1},
        {"type": "game_done", "game": 1, "moves": 6, "result": "O", "temp": 1.0, "elapsed_s": 0.2},
    ]
    res = rustselfplay.run_round(
        "fake-exe",
        "w.safetensors",
        str(tmp_path),
        popen=_popen_factory(events, crash_after=2),  # dies after game 0 of 3
        **_round_kwargs(games=3),
    )
    assert res["crashed"] is True and res["games_done"] == 1
    assert "boom" in res["stderr_tail"]


def test_run_round_ignores_garbage_lines(tmp_path):
    # broken timestamps, legacy stdout noise, a JSON array and a bare int:
    # none of them may count as a game or crash the parser
    lines = ["not json", "[1, 2]", "123", {"type": "game_done", "game": 0, "moves": 3}]
    res = rustselfplay.run_round(
        "fake-exe",
        "w.safetensors",
        str(tmp_path),
        popen=_popen_factory(lines),
        **_round_kwargs(games=1),
    )
    assert res["games_done"] == 1 and res["crashed"] is False
    assert res["stderr_tail"] == ""


def test_run_round_stderr_tail_is_truncated(tmp_path):
    long_err = "x" * 1000 + "END"
    res = rustselfplay.run_round(
        "fake-exe",
        "w.safetensors",
        str(tmp_path),
        popen=_popen_factory([{"type": "start", "games": 1}], crash_after=0, stderr_text=long_err),
        **_round_kwargs(games=1),
    )
    assert res["crashed"] is True  # stderr present but no game ever finished
    assert len(res["stderr_tail"]) == 400
    assert res["stderr_tail"].endswith("END")  # the TAIL survives, the head is cut


# -----------------------------------------------------------------------
# run_round: watchdog
# -----------------------------------------------------------------------


def test_run_round_watchdog_kills_silent_exe(tmp_path):
    procs = []
    # every line arrives after 0.15s of silence, far past the 0.05s deadline:
    # the wait must time out before the first line arrives — nothing parsed,
    # process killed, round reported crashed
    res = rustselfplay.run_round(
        "fake-exe",
        "w.safetensors",
        str(tmp_path),
        timeout=0.05,
        popen=_popen_factory(
            [{"type": "start", "games": 1}, {"type": "game_done", "game": 0, "moves": 4}],
            delay=0.15,
            procs=procs,
        ),
        **_round_kwargs(games=1),
    )
    assert res["crashed"] is True and res["games_done"] == 0
    assert procs[0].killed is True  # the watchdog, not the fake, ended it
    assert "boom" in res["stderr_tail"]


class WedgedPipe:
    """stdout stand-in wedged like a real pipe: iteration blocks, nothing
    is ever delivered, EOF shows up only once the process dies. The block
    is bounded so a broken watchdog fails the test, not the whole suite."""

    def __init__(self, released, cap=5.0):
        self._released = released
        self._cap = cap

    def __iter__(self):
        return self

    def __next__(self):
        self._released.wait(timeout=self._cap)  # wedged: no line to deliver
        raise StopIteration  # EOF only after kill() tears the pipe down


def test_run_round_watchdog_kills_truly_silent_exe(tmp_path):
    """The case the watchdog exists for: the exe writes nothing and never
    exits. The deadline must kill it and run_round must return."""
    released = threading.Event()
    procs = []

    class SilentProc:
        """A proc whose stdout is a wedged pipe: no line, no EOF, no exit."""

        def __init__(self, cmd, **kwargs):
            procs.append(self)
            self.cmd = cmd
            self.kwargs = kwargs
            self.killed = False
            self.returncode = None
            self.stdout = WedgedPipe(released)
            self.stderr = io.StringIO("wedged before game 0")

        def kill(self):
            self.killed = True
            self.returncode = -9
            released.set()  # the kill tears the wedged pipe down

        def wait(self, timeout=None):
            return self.returncode

    start = time.monotonic()
    res = rustselfplay.run_round(
        "fake-exe",
        "w.safetensors",
        str(tmp_path),
        timeout=0.05,
        popen=SilentProc,
        **_round_kwargs(games=1),
    )
    elapsed = time.monotonic() - start
    assert elapsed < 2.0  # run_round returned instead of hanging on the wedge
    assert res["crashed"] is True and res["games_done"] == 0
    assert procs[0].killed is True  # the watchdog, not the fake, ended it
    assert "wedged" in res["stderr_tail"]


# -----------------------------------------------------------------------
# run_round: spawn contract
# -----------------------------------------------------------------------


def test_run_round_spawns_argv_list(tmp_path):
    procs = []
    rustselfplay.run_round(
        "fake-exe",
        "w.safetensors",
        str(tmp_path),
        popen=_popen_factory([], procs=procs),
        **_round_kwargs(),
    )
    assert len(procs) == 1
    proc = procs[0]
    cmd = proc.cmd
    assert isinstance(cmd, list) and cmd[0] == "fake-exe"
    pairs = dict(zip(cmd[1::2], cmd[2::2], strict=True))
    assert pairs == {
        "--weights": "w.safetensors",
        "--out": str(tmp_path),
        "--games": "2",
        "--sims": "8",
        "--threads": "1",
        "--seed-base": "0",
        "--game-type": "ultimate",  # the default when not passed
        "--c-puct": "1.5",
        "--dirichlet-eps": "0.25",
        "--temp-start": "1.000000",
        "--temp-end": "1.000000",
        "--temp-min": "0.050000",
    }
    assert proc.kwargs["stdout"] is subprocess.PIPE
    assert proc.kwargs["stderr"] is subprocess.PIPE
    assert proc.kwargs["text"] is True
    assert proc.kwargs.get("shell", False) is False  # shell=False discipline


# -----------------------------------------------------------------------
# wiring guards (controller ruling: the timeout lives in config, NOT in
# alphazero — Task 9 imports rustselfplay from alphazero, so a reverse
# import from rustselfplay would be a cycle)
# -----------------------------------------------------------------------


def test_run_round_default_timeout_is_the_config_constant():
    params = inspect.signature(rustselfplay.run_round).parameters
    assert params["timeout"].default == MP_GAME_TIMEOUT


def test_rustselfplay_never_imports_alphazero():
    src = inspect.getsource(rustselfplay)
    assert "from alphazero" not in src and "import alphazero" not in src


# -----------------------------------------------------------------------
# resolve_engine / find_exe
# -----------------------------------------------------------------------


def test_resolve_engine_auto(monkeypatch):
    monkeypatch.setattr(rustselfplay, "find_exe", lambda explicit=None: "/x/sba_selfplay.exe")
    assert rustselfplay.resolve_engine("auto", None) == ("rust", "/x/sba_selfplay.exe")
    assert rustselfplay.resolve_engine("auto", "/given") == ("rust", "/given")
    assert rustselfplay.resolve_engine("rust", None) == ("rust", "/x/sba_selfplay.exe")
    monkeypatch.setattr(rustselfplay, "find_exe", lambda explicit=None: None)
    assert rustselfplay.resolve_engine("auto", None) == ("python", None)
    assert rustselfplay.resolve_engine("python", None) == ("python", None)
    assert rustselfplay.resolve_engine("rust", "/explicit") == ("rust", "/explicit")


def test_find_exe_explicit_and_env(tmp_path, monkeypatch):
    exe = tmp_path / "sba_selfplay.exe"
    exe.write_bytes(b"")
    missing = tmp_path / "missing.exe"
    monkeypatch.delenv("SBA_SELFPLAY_EXE", raising=False)
    # point ROOT somewhere without probes so env/explicit results are unambiguous
    monkeypatch.setattr(rustselfplay, "ROOT", str(tmp_path / "no-probes-here"))
    assert rustselfplay.find_exe(str(exe)) == str(exe)  # explicit wins
    assert rustselfplay.find_exe(str(missing)) is None  # explicit is final, no probe fallback
    monkeypatch.setenv("SBA_SELFPLAY_EXE", str(exe))
    assert rustselfplay.find_exe() == str(exe)  # env beats probes
    monkeypatch.setenv("SBA_SELFPLAY_EXE", str(missing))
    assert rustselfplay.find_exe() is None  # bad env falls through to (absent) probes


def test_find_exe_probe_priority(tmp_path, monkeypatch):
    monkeypatch.delenv("SBA_SELFPLAY_EXE", raising=False)
    monkeypatch.setattr(rustselfplay, "ROOT", str(tmp_path))
    rel = "native/sba-core/target"
    order = [
        f"{rel}/x86_64-pc-windows-gnu/release/sba_selfplay.exe",
        f"{rel}/release/sba_selfplay.exe",
        f"{rel}/release/sba_selfplay",  # no-.exe spelling: CI Linux cpu build
        f"{rel}/debug/sba_selfplay.exe",
        f"{rel}/debug/sba_selfplay",
    ]
    paths = []
    for relp in order:
        p = tmp_path / relp
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
        paths.append(p)
    for i, expected in enumerate(paths):  # probes in listed order, earliest wins
        assert rustselfplay.find_exe() == os.path.normpath(str(expected))
        paths[i].unlink()
    assert rustselfplay.find_exe() is None  # nothing left -> nothing found
