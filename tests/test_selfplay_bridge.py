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


def _round_args(out_dir, **over):
    """One RoundArgs with the round scalars the fakes play against."""
    kw = dict(
        games=2,
        sims=8,
        threads=1,
        seed_base=0,
        temp_start=1.0,
        temp_end=1.0,
    )
    kw.update(over)
    return rustselfplay.RoundArgs(weights_path="w.safetensors", out_dir=out_dir, **kw)


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
        _round_args(str(tmp_path)),
        exe="fake-exe",
        on_event=called.append,
        popen=_popen_factory(events),
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
        _round_args(str(tmp_path)),
        exe="fake-exe",
        popen=_popen_factory(lines),
    )
    assert res["games_done"] == 2 and res["crashed"] is False
    # same stream but only 2 of 4 games delivered -> short count crashes
    res = rustselfplay.run_round(
        _round_args(str(tmp_path), games=4),
        exe="fake-exe",
        popen=_popen_factory(lines),
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
        _round_args(str(tmp_path), games=3),
        exe="fake-exe",
        popen=_popen_factory(events, crash_after=2),  # dies after game 0 of 3
    )
    assert res["crashed"] is True and res["games_done"] == 1
    assert "boom" in res["stderr_tail"]


def test_run_round_ignores_garbage_lines(tmp_path):
    # broken timestamps, legacy stdout noise, a JSON array and a bare int:
    # none of them may count as a game or crash the parser
    lines = ["not json", "[1, 2]", "123", {"type": "game_done", "game": 0, "moves": 3}]
    res = rustselfplay.run_round(
        _round_args(str(tmp_path), games=1),
        exe="fake-exe",
        popen=_popen_factory(lines),
    )
    assert res["games_done"] == 1 and res["crashed"] is False
    assert res["stderr_tail"] == ""


def test_run_round_stderr_tail_is_truncated(tmp_path):
    long_err = "x" * 1000 + "END"
    res = rustselfplay.run_round(
        _round_args(str(tmp_path), games=1),
        exe="fake-exe",
        popen=_popen_factory([{"type": "start", "games": 1}], crash_after=0, stderr_text=long_err),
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
        _round_args(str(tmp_path), games=1),
        exe="fake-exe",
        timeout=0.05,
        popen=_popen_factory(
            [{"type": "start", "games": 1}, {"type": "game_done", "game": 0, "moves": 4}],
            delay=0.15,
            procs=procs,
        ),
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
        _round_args(str(tmp_path), games=1),
        exe="fake-exe",
        timeout=0.05,
        popen=SilentProc,
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
    rargs = _round_args(str(tmp_path))
    rustselfplay.run_round(
        rargs,
        exe="fake-exe",
        popen=_popen_factory([], procs=procs),
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
    # the argv IS the shared builder's output (bench_selfplay's probe spawns
    # the exact same list — one flag contract, no mirror to drift)
    assert cmd == ["fake-exe", *rustselfplay.build_round_cmd(rargs)]
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
    rel = "target"  # the workspace target dir at the repo root (D60)
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


# -----------------------------------------------------------------------
# run_round: the K-process round (D59 stream-overlap attempt) — K exes
# share one round with disjoint seeds/records; merged watchdog, kill-all
# on any failure, one merged selfplay_round event
# -----------------------------------------------------------------------


def _multi_factory(lines_per_proc, procs):
    """One fake exe process per entry in `lines_per_proc` (spawn order =
    list order); `procs` collects the spawned FakePopens so tests can read
    their cmd/kill state."""

    def factory(cmd, **kwargs):
        p = FakePopen(lines_per_proc[len(procs)])
        p.stdout = p.stream()
        p.cmd = cmd
        p.kwargs = kwargs
        procs.append(p)
        return p

    return factory


def _cmds(procs):
    """The spawned argv lists as flag->value dicts (one per process)."""
    return [dict(zip(p.cmd[1::2], p.cmd[2::2], strict=True)) for p in procs]


def _done(i):
    return {"type": "game_done", "game": i, "moves": 5, "result": "X"}


def test_run_round_procs_partition_and_merge(tmp_path):
    procs = []
    # 8 games over 4 procs: 2 each; seeds and --start-idx advance by 2;
    # every process delivers its quota -> merged round is clean
    lines = [[_done(0), _done(1)]] * 4
    events = []
    res = rustselfplay.run_round(
        _round_args(str(tmp_path), games=8, threads=8),
        exe="fake-exe",
        on_event=events.append,
        popen=_multi_factory(lines, procs),
        procs=4,
    )
    assert len(procs) == 4
    cmds = _cmds(procs)
    assert [c["--games"] for c in cmds] == ["2", "2", "2", "2"]
    # disjoint seed bases: seed_base + k*games_per_proc — the merged round
    # replays the SAME stream a single process would (seed_base + round idx)
    assert [c["--seed-base"] for c in cmds] == ["0", "2", "4", "6"]
    # disjoint record names: --start-idx = the process's first round index
    # (0 keeps the historical argv — the exe defaults to 0)
    assert [c.get("--start-idx") for c in cmds] == [None, "2", "4", "6"]
    # game threads split so the CPU shape stays comparable: threads // procs
    assert [c["--threads"] for c in cmds] == ["2", "2", "2", "2"]
    # same shared out dir (records partition by NAME, not by directory)
    assert {c["--out"] for c in cmds} == {str(tmp_path)}
    # merged round: games_done sums across processes, ONE selfplay_round event
    assert res["games_done"] == 8 and res["crashed"] is False
    assert res["stderr_tail"] == ""
    assert events == [{"type": "selfplay_round", "games_done": 8, "crashed": False}]


def test_run_round_procs_remainder_partition(tmp_path):
    procs = []
    # 7 games over 4 procs: games/K rounds UP (2) for every process except
    # the remainder-short last (1) — the total stays exactly 7 and the seed
    # stream stays contiguous ([0, 2, 4, 6] with [2, 2, 2, 1] games = 7 seeds)
    lines = [[_done(0), _done(1)], [_done(0), _done(1)], [_done(0), _done(1)], [_done(0)]]
    res = rustselfplay.run_round(
        _round_args(str(tmp_path), games=7, threads=4),
        exe="fake-exe",
        popen=_multi_factory(lines, procs),
        procs=4,
    )
    cmds = _cmds(procs)
    assert [c["--games"] for c in cmds] == ["2", "2", "2", "1"]
    assert [c["--seed-base"] for c in cmds] == ["0", "2", "4", "6"]
    assert [c.get("--start-idx") for c in cmds] == [None, "2", "4", "6"]
    assert res["games_done"] == 7 and res["crashed"] is False


def test_run_round_procs_temp_schedule_slices(tmp_path):
    procs = []
    lines = [[_done(0)], [_done(0)]]

    # 5 games over 2 procs (3 + 2), temp 1.0 -> 0.5 over the round: each
    # process gets its EXACT slice of the linear schedule so per-game temps
    # match the single-process round (the round denominator is games-1 = 4)
    rustselfplay.run_round(
        _round_args(str(tmp_path), games=5, temp_start=1.0, temp_end=0.5),
        exe="fake-exe",
        popen=_multi_factory(lines, procs),
        procs=2,
    )
    cmds = _cmds(procs)
    assert [c["--games"] for c in cmds] == ["3", "2"]
    assert cmds[0]["--temp-start"] == "1.000000"  # global idx 0
    assert cmds[0]["--temp-end"] == "0.750000"  # global idx 2: 1.0 + (0.5-1)*2/4
    assert cmds[1]["--temp-start"] == "0.625000"  # global idx 3: 1.0 + (0.5-1)*3/4
    assert cmds[1]["--temp-end"] == "0.500000"  # global idx 4 == the round end


def test_run_round_procs_kill_all_on_crash(tmp_path):
    procs = []
    # proc 0 dies after 1 game of its quota; proc 1 is slow (nothing parsed
    # before the abort wakes its watcher) — the kill-all must reach it
    fast = [{"type": "game_done", "game": 0, "moves": 5}]
    slow = [{"type": "game_done", "game": 0, "moves": 5}]

    def factory(cmd, **kwargs):
        p = FakePopen(fast, crash_after=1) if len(procs) == 0 else FakePopen(slow, delay=1.0)
        p.stdout = p.stream()
        p.cmd = cmd
        p.kwargs = kwargs
        procs.append(p)
        return p

    res = rustselfplay.run_round(
        _round_args(str(tmp_path), games=4, threads=2),
        exe="fake-exe",
        popen=factory,
        procs=2,
    )
    assert res["crashed"] is True
    assert res["games_done"] == 1  # only proc 0's completed game
    assert "boom" in res["stderr_tail"]
    assert procs[0].killed is False  # it died on its own before the abort
    assert procs[1].killed is True  # the kill-all, not the fake, ended it


def test_run_round_procs_abort_after_clean_exit_swallows_kill(tmp_path):
    """The abort-vs-EOF race: a sibling crashes while this proc is finishing.
    When the abort wins, the kill-all fires on an ALREADY-EXITED process —
    real Popen.kill raises OSError there (TerminateProcess on a dead handle,
    Windows) — and the watcher must swallow it: the round's verdict is the
    sibling's failure, not a broken watcher."""
    procs = []
    good = [_done(0), _done(1)]

    class ExitedKillProc(FakePopen):
        """kill() on an already-exited process: raises like real Popen does."""

        def kill(self):
            self.kill_attempts = getattr(self, "kill_attempts", 0) + 1
            raise OSError("process already exited")

    def factory(cmd, **kwargs):
        if len(procs) == 0:
            p = ExitedKillProc(good, delay=0.3, stderr_text="")  # finishes slowly
        else:
            p = FakePopen([], crash_after=0)  # dies instantly, 0 games
        p.stdout = p.stream()
        p.cmd = cmd
        p.kwargs = kwargs
        procs.append(p)
        return p

    res = rustselfplay.run_round(
        _round_args(str(tmp_path), games=4, threads=2),
        exe="fake-exe",
        popen=factory,
        procs=2,
    )
    assert res["crashed"] is True  # the sibling's crash, propagated
    assert procs[0].kill_attempts == 1  # the abort reached a clean-exited proc
    # the swallow: the tail is the sibling's crash, not a watcher death
    assert "boom" in res["stderr_tail"]
    assert "watcher" not in res["stderr_tail"]


def test_run_round_procs_watchdog_kills_all_on_timeout(tmp_path):
    procs = []
    # proc 0 goes silent (wedged pipe, no EOF, no exit); proc 1 delivers one
    # game then wedges MID-QUOTA — the shared round deadline must kill BOTH
    # (per-process watchdog on one round-total budget, kill-all on timeout)
    released = threading.Event()

    def wedged_after_first(first_line, released):
        yield json.dumps(first_line) + "\n"
        released.wait(timeout=5.0)  # wedged: EOF only when kill tears it down

    class HalfFastProc(FakePopen):
        """Delivers one game, then wedges mid-round like a real hung exe."""

        def __init__(self, first_line):
            super().__init__([first_line])
            self.stdout = wedged_after_first(first_line, released)
            self.stderr = io.StringIO("wedged mid-quota")

        def kill(self):
            self.killed = True
            self.returncode = -9
            released.set()  # the kill tears the wedged pipe down

    def factory(cmd, **kwargs):
        if len(procs) == 0:
            p = FakePopen([{"type": "start", "games": 4}])  # silent after start
            p.stdout = WedgedPipe(released)  # never a line, EOF only on kill

            def kill_silent():
                p.killed = True
                p.returncode = -9
                released.set()

            p.kill = kill_silent
        else:
            p = HalfFastProc(_done(0))
        p.cmd = cmd
        p.kwargs = kwargs
        procs.append(p)
        return p

    res = rustselfplay.run_round(
        _round_args(str(tmp_path), games=4, threads=2),
        exe="fake-exe",
        timeout=0.05,
        popen=factory,
        procs=2,
    )
    assert res["crashed"] is True
    assert res["games_done"] == 1  # proc 1's one completed game before the kill
    assert procs[0].killed is True  # the silent proc: the deadline
    assert procs[1].killed is True  # mid-quota sibling: the kill-all
    assert "boom" in res["stderr_tail"]  # the fake's stderr tail rides along
