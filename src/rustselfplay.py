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
import subprocess
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
_PROBE_PATHS = (
    "native/sba-core/target/x86_64-pc-windows-gnu/release/sba_selfplay.exe",
    "native/sba-core/target/release/sba_selfplay.exe",
    "native/sba-core/target/release/sba_selfplay",
    "native/sba-core/target/debug/sba_selfplay.exe",
    "native/sba-core/target/debug/sba_selfplay",
)


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


def find_exe(explicit=None):
    """Locate the sba_selfplay binary.

    Order: explicit argument (final — a missing explicit path is NOT silently
    replaced by a probe hit) → $SBA_SELFPLAY_EXE → target/release|debug probe
    paths relative to the repo root. None when nothing exists.
    """
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    env = os.environ.get("SBA_SELFPLAY_EXE")
    if env and os.path.isfile(env):
        return env
    for rel in _PROBE_PATHS:
        cand = os.path.normpath(os.path.join(ROOT, rel))
        if os.path.isfile(cand):
            return cand
    return None


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
    other type — including future ones — is ignored. crashed is True on a
    nonzero exit OR when fewer than `games` game_done events arrived. The
    watchdog kills the exe once a line arrives after more than `timeout`
    seconds of silence (mirrors the mp.Pool worker watchdog). stderr is read
    only after the process is gone: the exe writes at most one fatal line,
    so the pipe cannot fill and block it.

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
    games_done = 0
    stderr_tail = ""
    last_beat = time.time()  # spawn counts as the first heartbeat
    for line in proc.stdout:
        # watchdog FIRST, against the gap since the previous line; updating
        # last_beat before this check would make it a permanent no-op
        if time.time() - last_beat > timeout:
            proc.kill()
            stderr_tail = (proc.stderr.read() or "")[-400:]
            break
        last_beat = time.time()  # a line arrived: liveness heartbeat
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict) and ev.get("type") == "game_done":
            games_done += 1
    rc = proc.wait()
    if rc != 0 and stderr_tail == "":
        stderr_tail = (proc.stderr.read() or "")[-400:]
    crashed = rc != 0 or games_done < games
    if on_event:
        on_event({"type": "selfplay_round", "games_done": games_done, "crashed": crashed})
    return {"games_done": games_done, "crashed": crashed, "stderr_tail": stderr_tail}
