# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""bench_rust — the Rust state-layer speedup harness.

Measures the same self-play-shaped work against both game
implementations: the pure-Python reference (UltimateGame) and the Rust
wrapper (RustUltimateGame, only when src/az/sba_game.dll is built).
Three numbers per side:

  playout   — apply_move + result per ply, driven by one shared seeded
              move list (isolates state mutation cost)
  fullgame  — the real mix: legal_moves + encode + clone per ply
  stage1    — one C-tree search per move (_ce.mcts_search), the actual
              stage-1 self-play hot path

Usage:  python src/tools/bench_rust.py [--games 20] [--sims 100]
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))  # src/tools/
ROOT = os.path.dirname(os.path.dirname(HERE))  # the repo root
SRC = os.path.join(ROOT, "src")
for _p in (SRC, ROOT):  # SRC for the sba packages under src/
    if _p not in sys.path:
        sys.path.insert(0, _p)

from az import rustbridge  # noqa: E402
from az.alphazero import encode  # noqa: E402
from core.game import UltimateGame, apply_move  # noqa: E402
from engines import cengine as _ce  # noqa: E402
from engines.ai import _c_search_opts  # noqa: E402

SEED = 20261115


def _shared_moves(n_games, seed=SEED):
    """One seeded legal-move sequence per game number, from the reference
    implementation — drives both sides identically in the playout bench."""
    import random

    rng = random.Random(seed)
    seqs = []
    for _ in range(n_games):
        g = UltimateGame()
        seq = []
        while not g.is_over():
            lm = g.legal_moves()
            mv = rng.choice(lm)
            seq.append(mv)
            g.make_move(*mv)
        seqs.append(seq)
    return seqs


def bench_playout(seqs, sides):
    """apply-only: the listed sides replay the same move sequences."""
    out = {}
    for name, cls in sides:
        t0 = time.perf_counter()
        plies = 0
        for seq in seqs:
            g = cls()
            for mv in seq:
                g.make_move(*mv)
                plies += 1
                _ = g.result()
        out[name] = (time.perf_counter() - t0, plies)
    return out


def bench_fullgame(n_games, sides):
    """The real per-ply mix: legal_moves + encode + clone + apply, each
    side driving itself (this is what self-play pays per move)."""
    import random

    out = {}
    for name, make in sides:
        rng = random.Random(SEED)
        t0 = time.perf_counter()
        plies = 0
        for _ in range(n_games):
            g = make()
            while not g.is_over():
                lm = g.legal_moves()
                plies += len(lm)
                if not lm:
                    break
                _ = encode(g)  # the training-step tensor
                _ = g.clone()  # the expansion-shaped clone
                apply_move(g, rng.choice(lm))
        out[name] = (time.perf_counter() - t0, plies)
    return out


def bench_stage1(seqs, sims, sides):
    """One C-tree search per move — the stage-1 self-play hot path."""
    out = {}
    for name, make in sides:
        t0 = time.perf_counter()
        searches = 0
        for seq in seqs:
            g = make()
            for mv in seq:
                _ce.mcts_search(g, sims, opts=_c_search_opts())
                searches += 1
                if not g.is_over():
                    g.make_move(*mv)
        out[name] = (time.perf_counter() - t0, searches)
    return out


def _row(label, out):
    p_t, p_n = out["python"]
    line = f"{label:<10} python {p_t:8.3f}s"
    if "rust" in out:
        r_t, r_n = out["rust"]
        speedup = p_t / r_t if r_t else float("inf")
        line += f"   rust {r_t:8.3f}s   speedup {speedup:5.2f}x"
    line += f"   ({p_n} ops)"
    return line


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=100)
    args = ap.parse_args()

    has_rust = rustbridge.available()
    if has_rust:
        print("rust wrapper: available")
    else:
        print("rust wrapper: NOT BUILT (python rows only — build sba_game.dll to compare)")

    seqs = _shared_moves(args.games)
    seqs = [s for s in seqs if s][: args.games]

    print()
    sides = [("python", UltimateGame)]
    if has_rust:
        sides.append(("rust", lambda: rustbridge.RustUltimateGame()))
    print(_row("playout", bench_playout(seqs, sides)))
    print(_row("fullgame", bench_fullgame(args.games, sides)))
    print(_row("stage1", bench_stage1(seqs, args.sims, sides)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
