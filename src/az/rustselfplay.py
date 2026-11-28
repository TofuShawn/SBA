# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bridge to the Rust self-play exe (sba-core, Phase A).

The exe owns search + inference and writes one safetensors record per game;
this module owns the Python side of that contract: locating the binary,
resolving which engine a round runs on, running rounds with a watchdog, and
loading the records into the replay format. The exe never talks to torch;
Python never touches the tree.
"""

import contextlib
import glob
import json
import math
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, replace

import numpy as np

from config import MP_GAME_TIMEOUT, TEMP_MIN

# repo root (this file lives in src/az/, so three dirnames up —
# config.py at src/ still uses two)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Probe order: WSL mingw cross-build, then release, then debug; the extension-less
# names are the Linux ELF spellings CI builds under the shared root target/.
def _probe_paths(name):
    """The probe list for one sba-core binary name (see find_exe)."""
    return (
        f"target/x86_64-pc-windows-gnu/release/{name}.exe",
        f"target/release/{name}.exe",
        f"target/release/{name}",
        f"target/debug/{name}.exe",
        f"target/debug/{name}",
    )


_PROBE_PATHS = _probe_paths("sba_selfplay")


def load_records(dir_path):
    """Read every g*.safetensors in dir_path into replay-ready tuples.

    Returns [] for a missing/empty dir — callers treat that as "nothing
    landed", not an error (a crashed round may legitimately have written 0).

    The one torch touchpoint lives here lazily: the bridge must import on a
    bare install (the exe owns search + inference; torch is the training
    side's dependency — see requirements.txt), so load_records pulls torch
    only when records actually land.
    """
    import torch

    from safetensors.numpy import load_file

    games = []
    for p in sorted(glob.glob(os.path.join(dir_path, "g*.safetensors"))):
        d = load_file(p)
        states = np.ascontiguousarray(d["states"]).reshape(-1, 4, 9, 9)
        policy = d["policy"].reshape(-1, 81)
        z = d["z"].reshape(-1)
        games.append(
            (
                [torch.as_tensor(np.ascontiguousarray(s)) for s in states],
                [np.asarray(p, dtype=np.float32) for p in policy],
                [float(v) for v in z],
            )
        )
    return games


def record_indices(dir_path):
    """The round-game indices of the records load_records() returns, in the
    SAME sorted order (the g%05d file name IS the round-game index: the
    exe's start_idx + its per-process idx, records.rs). Pairs
    position-for-position with load_records, so a caller can join per-game
    exe events (the game_done temps) onto the loaded records."""
    return [
        int(os.path.splitext(os.path.basename(p))[0][1:])
        for p in sorted(glob.glob(os.path.join(dir_path, "g*.safetensors")))
    ]


def _find_bin(explicit, name, env_var):
    """Shared probe order behind find_exe / find_bench: explicit argument
    (final — a missing explicit path is NOT silently replaced by a probe
    hit) → env override → target/release|debug probe paths relative to the
    repo root. None when nothing exists.
    """
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    if env_var:
        env = os.environ.get(env_var)
        if env and os.path.isfile(env):
            return env
    for rel in _probe_paths(name):
        cand = os.path.normpath(os.path.join(ROOT, rel))
        if os.path.isfile(cand):
            return cand
    return None


def find_exe(explicit=None, name="sba_selfplay"):
    """Locate an sba-core binary (`name`: 'sba_selfplay' or 'sba_debug').

    Order: explicit argument (final — a missing explicit path is NOT silently
    replaced by a probe hit) → $SBA_SELFPLAY_EXE (selfplay only — the var is
    that exe's override) → target/release|debug probe paths relative to the
    repo root. None when nothing exists.
    """
    return _find_bin(explicit, name, "SBA_SELFPLAY_EXE" if name == "sba_selfplay" else None)


def find_bench(explicit=None):
    """Locate the bench_burn inference-spike binary (Task 12: the parity
    fuzz drives weights + input planes + --dump-out). Same order as
    find_exe with $SBA_BENCH as the override. None when nothing exists.
    """
    return _find_bin(explicit, "bench_burn", "SBA_BENCH")


def resolve_engine(engine_arg, exe):
    """Arbitrate python vs rust for one round: ('python'|'rust', exe|None).

    'python' → python; 'rust' → rust with the given exe (found when None);
    'auto' → rust when an exe exists, else python. Explicit exe wins over a
    fresh find.
    """
    if engine_arg == "python":
        return "python", None
    if engine_arg == "rust":
        return "rust", exe or find_exe()
    found = exe or find_exe()
    return ("rust", found) if found else ("python", None)


_STDOUT_EOF = object()  # sentinel: the stdout stream has been exhausted
_STDERR_KEEP = 8192  # stderr chars the drain thread retains (bounded buffer)
_READER_JOIN = 1.0  # seconds granted to the drain threads after the exe exits
_ABORT_POLL = 0.2  # seconds between wakeups so a sibling proc's abort is seen
_EXIT_GRACE = 1.0  # seconds granted to a process that outlived its stdout


@dataclass
class RoundArgs:
    """One round's parameters — everything the exe needs to play it.

    The round contract in one object: callers (train()'s rust loop, the
    bench probe) build one and hand it to run_round; the retry path rewrites
    just the changed fields with dataclasses.replace, so an override can
    never silently drop the rest. `start_idx` offsets the exe's record
    numbering (g%05d names become g{start_idx+idx}) so the K-process split
    (run_round's `procs`) writes disjoint filenames into the shared out
    dir; 0 keeps the historical numbering."""

    weights_path: str
    out_dir: str
    games: int
    sims: int
    threads: int
    seed_base: int
    temp_start: float
    temp_end: float
    temp_min: float = TEMP_MIN
    c_puct: float = 1.5
    dirichlet_eps: float = 0.25
    game_type: str = "ultimate"
    start_idx: int = 0


def build_round_cmd(args):
    """The exe's flag list for one round — the SINGLE copy of the argv
    contract (flag spellings, str() conversions, the %.6f temperature
    formatting). run_round spawns [exe, *build_round_cmd(args)]; callers
    outside the bridge (bench_selfplay's batch_stats probe, which parses
    stdout directly because run_round consumes the non-game_done events)
    prepend the exe path themselves. argv-list only (shell=False) — same
    discipline as rustbridge and src/tools/build. --start-idx is emitted only when
    nonzero (the exe defaults to 0, so single-process rounds keep the
    exact historical argv)."""
    cmd = [
        "--weights",
        str(args.weights_path),
        "--out",
        str(args.out_dir),
        "--games",
        str(args.games),
        "--sims",
        str(args.sims),
        "--threads",
        str(args.threads),
        "--seed-base",
        str(args.seed_base),
        "--game-type",
        args.game_type,
        "--c-puct",
        str(args.c_puct),
        "--dirichlet-eps",
        str(args.dirichlet_eps),
        "--temp-start",
        f"{args.temp_start:.6f}",
        "--temp-end",
        f"{args.temp_end:.6f}",
        "--temp-min",
        f"{args.temp_min:.6f}",
    ]
    if args.start_idx:
        cmd += ["--start-idx", str(args.start_idx)]
    return cmd


def _partition_round(args, procs):
    """Split one round across `procs` exe processes: disjoint game
    counts (games/procs rounded up for every process except the last, which
    takes the remainder — the total stays exactly args.games), disjoint
    seed bases (args.seed_base + k*games_per_proc — the per-process offsets
    are contiguous, so the merged round replays the SAME seed stream a
    single process would: seed = seed_base + absolute round index),
    disjoint record filenames (--start-idx = the process's first absolute
    round index), and each process's EXACT slice of the round's linear temp
    schedule (the exe interpolates temp_start→temp_end over its own games,
    and linear interpolation between the endpoints of a linear function is
    that function, so per-game temps match the single-process round).
    Game threads split evenly too: `threads // procs` per process (min 1)
    so the CPU shape stays comparable to the single-process round."""
    per = math.ceil(args.games / procs)
    specs = []
    for k in range(procs):
        first = k * per
        if first >= args.games:
            break  # more processes than games: the tail stays idle
        count = min(per, args.games - first)
        # the exe's per-game temp: temp_start + (temp_end - temp_start) *
        # i/(games-1) — slice it at this process's absolute round indices
        span = max(1, args.games - 1)
        lo = args.temp_start + (args.temp_end - args.temp_start) * first / span
        hi = args.temp_start + (args.temp_end - args.temp_start) * (first + count - 1) / span
        specs.append(
            replace(
                args,
                games=count,
                threads=max(1, args.threads // procs),
                seed_base=args.seed_base + first,
                start_idx=first,
                temp_start=lo,
                temp_end=hi,
            )
        )
    return specs


class _ProcWatch:
    """One watched exe process inside a (possibly multi-process) round.

    Shared mechanics behind run_round's single- and K-process shapes: the
    spawn + stdout pump + stderr drain from the historical run_round, plus
    an `abort` event — when any sibling process times out or crashes, every
    watcher kills its own process (kill-all on any failure)."""

    def __init__(self, cmd, expected_games, deadline, abort, popen, first_game=0):
        spawn = popen or subprocess.Popen
        self.proc = spawn(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.expected_games = expected_games
        self.deadline = deadline
        self.abort = abort
        self.first_game = first_game  # this proc's first absolute round-game index
        self.lines = queue.Queue()
        self.stderr_tails = []  # one entry, appended when the drain thread finishes

    def _pump_stdout(self, proc):
        """Reader thread: stdout → queue, so the main loop can wait with a
        timeout (a blocking for-loop over a wedged pipe cannot time out)."""
        try:
            for line in proc.stdout:
                self.lines.put(line)
        except (OSError, ValueError):
            pass  # pipe torn down by the watchdog kill; EOF below still lands
        finally:
            self.lines.put(_STDOUT_EOF)

    def _drain_stderr(self, proc):
        """Drain thread: consume stderr as it arrives, keeping only a tail.
        Without this, a stderr write bigger than the pipe buffer blocks the
        exe while stdout stalls — a mutual deadlock no watchdog could see."""
        tail = ""
        try:
            while True:
                chunk = proc.stderr.read(4096)
                if not chunk:
                    break
                tail = (tail + chunk)[-_STDERR_KEEP:]
        except (OSError, ValueError):
            pass  # same teardown story as stdout
        self.stderr_tails.append(tail)

    def watch(self):
        """Count game_done events until EOF, deadline, or abort; kill on
        timeout/abort. Returns (games_done, timed_out, rc, stderr_tail,
        temps) — temps maps this proc's absolute round-game indices to the
        exe-reported sampling temperatures (missing keys = no temp event)."""
        proc = self.proc
        reader = threading.Thread(
            target=self._pump_stdout, args=(proc,), name="selfplay-stdout", daemon=True
        )
        drainer = threading.Thread(
            target=self._drain_stderr, args=(proc,), name="selfplay-stderr", daemon=True
        )
        reader.start()
        drainer.start()

        games_done = 0
        timed_out = False
        temps = {}
        while True:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True  # no line for `timeout` seconds: silent/wedged exe
                break
            if self.abort.is_set():
                break  # a sibling failed: kill-all, this round is over
            try:
                item = self.lines.get(timeout=min(remaining, _ABORT_POLL))
            except queue.Empty:
                continue  # re-check deadline + abort
            if item is _STDOUT_EOF:
                break
            line = item.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict) and ev.get("type") == "game_done":
                games_done += 1
                # the exe reports each game's ACTUAL sampling temp (B4):
                # surface it keyed by the absolute round-game index so the
                # CSV/event temp is the exe's truth, not a recomputation
                g_idx, temp = ev.get("game"), ev.get("temp")
                if isinstance(g_idx, int) and isinstance(temp, (int, float)):
                    temps[self.first_game + g_idx] = float(temp)

        if timed_out or self.abort.is_set():
            # a sibling's abort can race this proc's clean exit — kill on an
            # already-dead process raises OSError on Windows (TerminateProcess
            # on a dead handle); the verdict below still holds
            with contextlib.suppress(OSError):
                proc.kill()
        try:
            rc = proc.wait(timeout=_EXIT_GRACE)
        except subprocess.TimeoutExpired:
            # the process outlived its own stdout (or the kill above): the
            # watchdog must RETURN — finish it like any wedged exe and reap
            with contextlib.suppress(OSError):
                proc.kill()
            rc = proc.wait()
        reader.join(_READER_JOIN)
        drainer.join(_READER_JOIN)
        stderr_tail = ""
        if rc != 0 or timed_out:
            stderr_tail = (self.stderr_tails[0] if self.stderr_tails else "")[-400:]
        return games_done, timed_out, rc, stderr_tail, temps


def run_round(args, *, exe, on_event=None, timeout=MP_GAME_TIMEOUT, popen=None, procs=1):
    """Run one exe round; returns {games_done, crashed, stderr_tail, temps}.

    `args` is a RoundArgs (the round's parameters); `exe` is the binary to
    spawn. stdout is the exe's JSONL event stream (start → (game_done|
    batch_stats)* → clean exit 0): game_done events are counted toward
    games_done, every other type — including future ones — is ignored, and
    non-JSON lines are skipped. crashed is True on a nonzero exit, when
    fewer than `args.games` game_done events arrived, or when an exe stayed
    silent for `timeout` seconds.

    `procs` splits the round across K concurrent exe processes
    (see _partition_round) writing disjoint record names into the SAME
    out dir; the merged round is ONE round — games_done sums across the
    processes, one selfplay_round event fires, and the FIRST failure
    (timeout or crash) kills every sibling so the retry contract stays
    the historical one round = one re-dispatch.

    The watchdog is a deadline, not a per-line gap: a daemon reader thread
    pumps stdout into a queue while stderr is drained concurrently into a
    bounded tail — a stderr write bigger than the pipe buffer (a
    RUST_BACKTRACE panic) would otherwise wedge the exe mid-round with
    stdout stalled — and the main loop waits on the queue with the
    remaining budget. The `timeout` budget is round-total, fixed once at
    spawn and SHARED by all K processes (train() sizes it
    `max(MP_GAME_TIMEOUT, round_games*120)`) — blowing it → proc.kill(),
    crashed=True, stderr tail reported.

    `popen` replaces subprocess.Popen (tests inject a fake); `on_event`, if
    given, receives {"type": "selfplay_round", "games_done": n,
    "crashed": bool} once at the end — game_done events are collected, not
    forwarded raw.

    The result also carries "temps": {round_game_index: temp} — the ACTUAL
    per-game sampling temperatures the exe reported in its game_done events
    (B4: the CSV/event temp is the exe's truth, never a recomputation).
    Missing keys mean that game's event carried no temp (callers fall back
    to the round schedule themselves).
    """
    deadline = time.monotonic() + timeout
    abort = threading.Event()
    specs = [args] if procs <= 1 else _partition_round(args, procs)
    watchers = [
        _ProcWatch(
            [exe, *build_round_cmd(spec)],
            spec.games,
            deadline,
            abort,
            popen,
            first_game=spec.start_idx,
        )
        for spec in specs
    ]

    results = [None] * len(watchers)

    def _watch(i, w):
        try:
            results[i] = w.watch()
        except Exception as e:  # noqa: BLE001 - a broken watcher fails the
            # ROUND (the retry contract re-dispatches), never the merge
            results[i] = (0, False, -1, f"watcher {i} died: {e!r}", {})
            abort.set()
            return
        games_done, timed_out, rc, _tail, _temps = results[i]
        if timed_out or rc != 0 or games_done < w.expected_games:
            abort.set()  # kill-all: this merged round has failed once already

    threads = [
        threading.Thread(target=_watch, args=(i, w), name=f"selfplay-watch-{i}", daemon=True)
        for i, w in enumerate(watchers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    games_done = sum(r[0] for r in results)
    temps = {}
    for r in results:
        temps.update(r[4])
    tails = [r[3] for r in results if r[3]]
    stderr_tail = tails[0][-400:] if tails else ""
    # crashed when any process failed or the merged total fell short of the round
    crashed = abort.is_set() or games_done < args.games
    if crashed:
        # Kill ALL live processes unconditionally: a watcher can hit its own EOF
        # break before a sibling's abort propagates, so the merge sweeps again.
        for w in watchers:
            if w.proc.poll() is None:
                with contextlib.suppress(OSError):
                    w.proc.kill()
                w.proc.wait()
    if on_event:
        on_event({"type": "selfplay_round", "games_done": games_done, "crashed": crashed})
    return {
        "games_done": games_done,
        "crashed": crashed,
        "stderr_tail": stderr_tail,
        "temps": temps,
    }
