# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""freeze_encode_golden — freeze the pure-Python layer-1 encode as golden vectors.

Runs the pure-Python ``alphazero.encode()`` (the numpy path — positions are
built on the pure-Python ``UltimateGame`` directly, never via ``make_game``,
so the ``isinstance(RustUltimateGame)`` branch inside ``encode`` can never
hijack the reference) over a seeded, MODEL-FREE position battery and writes:

  tests/golden/encode_golden.npz    moves / planes, one row per position
  tests/golden/encode_golden.json   metadata sidecar (seed, generator, versions)

The battery touches no network weights (encode is net-independent, so there
is no weights sha), and reuses the ``parity_search._gen_positions`` opening
conventions verbatim:

  first half   seeded random openings — depth cycling 0..8, rng
               ``seed*1009+idx``, uniform legal choice, terminal rejected;
  second half  snapshots from seeded full random playouts — plies
               4, 9, 14, … (``--stride`` apart) plus ALWAYS the terminal
               position of each game (win/draw coverage), games drawn
               until the quota fills.

At freeze time the tool ALSO rebuilds every position on the Rust state
layer (``RustUltimateGame`` + ``sba_planes_tensor``) and requires BITWISE
equality against the Python planes — the same assertion
``tests/test_encode_golden.py`` re-checks forever after, torch-free.

These goldens ARE the frozen layer-1 reference: once committed, the Rust
``planes()`` contract is pinned without ``alphazero.encode``. Regenerating
them is a deliberate encoder change, never a test fix.

Usage:
  python src/tools/freeze_encode_golden.py
  python src/tools/freeze_encode_golden.py --seed 20261128 --positions 40 --out tests/golden
"""

import argparse
import json
import os
import platform
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))  # src/tools/
ROOT = os.path.dirname(os.path.dirname(HERE))  # the repo root
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

from az import rustbridge  # noqa: E402
from az.alphazero import apply_move, encode  # noqa: E402
from core.game import UltimateGame  # noqa: E402  the pure-Python reference — never the Rust one here

FORMAT_VERSION = 1
DEFAULT_SEED = 20261128
DEFAULT_POSITIONS = 40
SNAPSHOT_STRIDE = 5  # playout snapshots every Nth ply, starting at ply 4
OPEN_SEED_BASE = 1009  # parity_search._gen_positions convention
PLAYOUT_SEED_BASE = 7919


def _random_opening(seed, idx):
    """One seeded random opening under the parity_search convention — the
    move history of a NON-terminal position (None when the random playout
    happened to end before the target depth)."""
    depth = idx % 9
    rng = random.Random(seed * OPEN_SEED_BASE + idx)
    g = UltimateGame()
    history = []
    for _ in range(depth):
        if g.is_over():
            break
        legal = g.legal_moves()
        mv = legal[rng.randrange(len(legal))]
        history.append(mv)
        apply_move(g, mv)
    return None if g.is_over() else history


def _playout_rows(seed, idx, need):
    """Rows from ONE seeded full random playout: spaced midgame plies
    (SNAPSHOT_STRIDE apart, from ply 4) plus ALWAYS the terminal position
    (emitted last, when the quota has room for it). Returns at most
    ``need`` rows; the caller keeps drawing games until the quota fills,
    so the battery always contains terminal rows."""
    rng = random.Random(seed * PLAYOUT_SEED_BASE + idx)
    g = UltimateGame()
    history = []
    mid = []
    while not g.is_over():
        if len(history) >= 4 and (len(history) - 4) % SNAPSHOT_STRIDE == 0:
            mid.append(list(history))
        legal = g.legal_moves()
        mv = legal[rng.randrange(len(legal))]
        history.append(mv)
        apply_move(g, mv)
    rows = [(h, "play") for h in mid[: need - 1]] + [(list(history), "terminal")]
    return rows[:need]


def build_battery(seed, n):
    """The seeded battery: ``n`` rows as (history, kind) — non-terminal by
    construction for "open", "terminal" rows only from the playout half."""
    rows = []
    n_open = n // 2
    idx = 0
    while len(rows) < n_open:
        hist = _random_opening(seed, idx)
        if hist is not None:
            rows.append((hist, "open"))
        idx += 1
        if idx > 5000:
            raise SystemExit("opening generator stalled")
    idx = 0
    while len(rows) < n:
        rows.extend(_playout_rows(seed, idx, n - len(rows)))
        idx += 1
        if idx > 200:
            raise SystemExit("playout generator stalled")
    return rows


def _rebuild(history, game_cls):
    """Replay a recorded history, asserting every recorded move is legal
    (the goldens are worthless if the battery cannot be rebuilt)."""
    g = game_cls()
    for m, i in history:
        if (m, i) not in g.legal_moves():
            raise SystemExit(f"recorded move ({m},{i}) is illegal on rebuild")
        g.make_move(m, i)
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=os.path.join("tests", "golden"),
        help="output directory (relative paths resolve against the repo root)",
    )
    ap.add_argument("--positions", type=int, default=DEFAULT_POSITIONS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args()

    print("freeze_encode_golden — pure-Python UltimateGame reference, net-independent")
    rows = build_battery(a.seed, a.positions)

    histories, metas, plane_rows = [], [], []
    rust_ok = rustbridge.available()
    if not rust_ok:
        print("WARNING: sba_game cdylib not found — freeze-time rust cross-check SKIPPED")
    for k, (history, kind) in enumerate(rows):
        g = _rebuild(history, UltimateGame)  # pure-Python: encode takes the numpy path
        planes = encode(g).numpy().astype(np.float32)
        assert planes.shape == (4, 9, 9)
        if rust_ok:
            rg = _rebuild(history, rustbridge.RustUltimateGame)
            if not np.array_equal(rg.sba_planes_tensor(), planes):
                raise SystemExit(f"row {k}: rust state layer disagrees with the Python encode")
            if rg.result() != (g.result() if g.is_over() else None):
                raise SystemExit(f"row {k}: rust result disagrees with the Python game")
        terminal = g.is_over()
        histories.append([m * 9 + i for m, i in history])
        metas.append(
            {
                "kind": kind,
                "depth": len(history),
                "terminal": terminal,
                "result": g.result() if terminal else None,
            }
        )
        plane_rows.append(planes)
        print(
            f"  [{k + 1}/{len(rows)}] kind={kind} depth={len(history)}"
            f" result={metas[-1]['result'] or '-'}",
            flush=True,
        )

    out_dir = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "encode_golden.npz")
    json_path = os.path.join(out_dir, "encode_golden.json")

    width = max(len(h) for h in histories)
    moves = np.full((len(histories), width), -1, dtype=np.int32)
    for r, hist in enumerate(histories):
        moves[r, : len(hist)] = hist
    np.savez(npz_path, moves=moves, planes=np.stack(plane_rows).astype(np.float32))

    import torch

    metadata = {
        "format_version": FORMAT_VERSION,
        "game": "ultimate",
        "seed": a.seed,
        "positions": len(histories),
        "stride": SNAPSHOT_STRIDE,
        "generator": (
            "parity_search._gen_positions opening conventions verbatim: first half "
            "seeded random openings (depth cycling 0..8, rng seed*1009+idx, uniform "
            "legal choice, terminal rejected); second half snapshots from seeded "
            "full random playouts (rng seed*7919+game) — plies 4,9,14,… stride "
            f"{SNAPSHOT_STRIDE} apart plus ALWAYS the terminal position. All moves "
            "uniform-random: MODEL-FREE, no weights sha applies. npz rows: "
            "moves=m*9+i codes (-1 pad), planes=[4,9,9] float32 — the pure-Python "
            "alphazero.encode() answer for the position the moves rebuild."
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "rust_crosscheck": bool(rust_ok),
        "battery": metas,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"wrote {npz_path} ({os.path.getsize(npz_path)} bytes)")
    print(f"wrote {json_path} (rust_crosscheck={bool(rust_ok)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
