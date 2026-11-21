# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bridge to the Rust self-play exe (sba-core, D55 Phase A).

The exe owns search + inference and writes one safetensors record per game;
this module owns the Python side of that contract: locating the binary,
resolving which engine a round runs on, running rounds with a watchdog, and
loading the records into the replay format. The exe never talks to torch;
Python never touches the tree.
"""

import glob
import json
import os
import queue
import subprocess
import threading
import time

import numpy as np
import torch

from config import MP_GAME_TIMEOUT

# repo root (this file lives in src/, so two dirnames up — same arithmetic
# config.py uses)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# probe order: the WSL mingw cross-build first (the Windows dev-machine
# artifact), then plain release, then debug. The extension-less spellings
# are the Linux ELF names — CI builds the cpu backend there and runs the
# exe without an .exe suffix.
def _probe_paths(name):
    """The probe list for one sba-core binary name (see find_exe)."""
    return (
        f"native/sba-core/target/x86_64-pc-windows-gnu/release/{name}.exe",
        f"native/sba-core/target/release/{name}.exe",
        f"native/sba-core/target/release/{name}",
        f"native/sba-core/target/debug/{name}.exe",
        f"native/sba-core/target/debug/{name}",
    )


_PROBE_PATHS = _probe_paths("sba_selfplay")


def load_records(dir_path):
    """Read every g*.safetensors in dir_path into replay-ready tuples.

    Returns [] for a missing/empty dir — callers treat that as "nothing
    landed", not an error (a crashed round may legitimately have written 0).
    """
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


def run_round(
    exe,
    weights_path,
    out_dir,
    *,
    games,
    sims,
    threads,
    seed_base,
    temp_start,
    temp_end,
    temp_min=0.05,
    c_puct=1.5,
    dirichlet_eps=0.25,
    game_type="ultimate",
    on_event=None,
    timeout=MP_GAME_TIMEOUT,
    popen=None,
):
    """Run one exe round; returns {games_done, crashed, stderr_tail}.

    argv-list only (shell=False) — same discipline as rustbridge/native/build.
    stdout is the exe's JSONL event stream (start → (game_done|batch_stats)*
    → clean exit 0): game_done events are counted toward games_done, every
    other type — including future ones — is ignored, and non-JSON lines are
    skipped. crashed is True on a nonzero exit, when fewer than `games`
    game_done events arrived, or when the exe stayed silent for `timeout`
    seconds.

    The watchdog is a deadline, not a per-line gap: a daemon reader thread
    pumps stdout into a queue while stderr is drained concurrently into a
    bounded tail — a stderr write bigger than the pipe buffer (a
    RUST_BACKTRACE panic) would otherwise wedge the exe mid-round with
    stdout stalled — and the main loop waits on the queue with the
    remaining budget. No line for `timeout` seconds → proc.kill(),
    crashed=True, stderr tail reported.

    `popen` replaces subprocess.Popen (tests inject a fake); `on_event`, if
    given, receives {"type": "selfplay_round", "games_done": n,
    "crashed": bool} once at the end — game_done events are collected, not
    forwarded raw.
    """
    cmd = [
        exe,
        "--weights",
        str(weights_path),
        "--out",
        str(out_dir),
        "--games",
        str(games),
        "--sims",
        str(sims),
        "--threads",
        str(threads),
        "--seed-base",
        str(seed_base),
        "--game-type",
        game_type,
        "--c-puct",
        str(c_puct),
        "--dirichlet-eps",
        str(dirichlet_eps),
        "--temp-start",
        f"{temp_start:.6f}",
        "--temp-end",
        f"{temp_end:.6f}",
        "--temp-min",
        f"{temp_min:.6f}",
    ]
    spawn = popen or subprocess.Popen
    proc = spawn(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = queue.Queue()
    stderr_tails = []  # one entry, appended when the drain thread finishes

    def _pump_stdout():
        """Reader thread: stdout → queue, so the main loop can wait with a
        timeout (a blocking for-loop over a wedged pipe cannot time out)."""
        try:
            for line in proc.stdout:
                lines.put(line)
        except (OSError, ValueError):
            pass  # pipe torn down by the watchdog kill; EOF below still lands
        finally:
            lines.put(_STDOUT_EOF)

    def _drain_stderr():
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
        stderr_tails.append(tail)

    reader = threading.Thread(target=_pump_stdout, name="selfplay-stdout", daemon=True)
    drainer = threading.Thread(target=_drain_stderr, name="selfplay-stderr", daemon=True)
    reader.start()
    drainer.start()

    games_done = 0
    deadline = time.monotonic() + timeout
    timed_out = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True  # no line for `timeout` seconds: silent/wedged exe
            break
        try:
            item = lines.get(timeout=remaining)
        except queue.Empty:
            timed_out = True
            break
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

    if timed_out:
        proc.kill()
    rc = proc.wait()
    reader.join(_READER_JOIN)
    drainer.join(_READER_JOIN)
    stderr_tail = ""
    if rc != 0 or timed_out:
        stderr_tail = (stderr_tails[0] if stderr_tails else "")[-400:]
    crashed = timed_out or rc != 0 or games_done < games
    if on_event:
        on_event({"type": "selfplay_round", "games_done": games_done, "crashed": crashed})
    return {"games_done": games_done, "crashed": crashed, "stderr_tail": stderr_tail}
