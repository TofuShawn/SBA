# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""freeze_wave_golden — freeze the Python wave-search reference as golden vectors.

Runs the Python batched wave search (`alphazero.mcts_search`, sims 200 /
batch 256 / c_puct 1.5 / dirichlet off / rollout_weight 0.0 — the D58
pure-net recipe, the only configuration layer 3 proved byte-equal against
the Rust wave search) over a seeded position battery and writes:

  tests/golden/wave_golden.npz    moves / counts / values, one row per position
  tests/golden/wave_golden.json   metadata sidecar (seeds, recipe, weights sha)

The battery reuses ``parity_search._gen_positions`` conventions verbatim:
the first half are seeded random openings (depth cycling 0..8), the second
half self-play midgames (dirichlet auto/0.25, temp 1.0, gen_sims plies) —
all FIXED-seed, and the seed recipe lives in the sidecar. The exported
weights sha256 goes into the metadata: the goldens are only valid for that
weights blob, and tests/test_wave_golden.py re-exports the archive and
refuses to run against any other checkpoint.

These goldens ARE the frozen Python search reference. Once committed, the
Rust wave search is pinned forever without ``mcts_search`` — regenerating
them is a deliberate recipe change, never a test fix.

Usage:
  python src/tools/freeze_wave_golden.py --weights models/az_ultimate.pt.archive
  python src/tools/freeze_wave_golden.py --seed 20261122 --positions 40 --out tests/golden
"""

import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))  # src/tools/
ROOT = os.path.dirname(os.path.dirname(HERE))  # the repo root
SRC = os.path.join(ROOT, "src")
for _p in (SRC, HERE):  # SRC for the az/ packages (D61), HERE for parity_search
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import parity_search  # noqa: E402

from az.alphazero import (  # noqa: E402
    DEVICE,
    _root_value,
    apply_move,
    export_safetensors,
    make_game,
    mcts_search,
)

FORMAT_VERSION = 1
DEFAULT_SEED = 20261122
DEFAULT_POSITIONS = 40
DEFAULT_SIMS = 200
DEFAULT_BATCH_SIZE = 256


def build_battery(seed, n, model, gen_sims, sims, batch_size):
    """Generate the seeded battery and search every position with the
    Python wave search (the pure-net recipe, exactly the layer-3 setup).

    Returns (histories, count_rows, values, battery_meta) where histories
    are move-code lists (m*9+i), count_rows are 81-wide int arrays indexed
    by move code and values are the visit-weighted root values."""
    positions = parity_search._gen_positions(seed, n, model, gen_sims)
    n_open = n // 2
    histories, rows, values, meta = [], [], [], []
    for idx, history in enumerate(positions):
        g = make_game("ultimate")
        for mv in history:
            apply_move(g, mv)
        legal = g.legal_moves()
        if g.is_over() or not legal:
            raise SystemExit(f"battery position {idx} is terminal — generator drifted")
        _root, counts = mcts_search(
            g, model, sims, rollout_weight=0.0, batch_size=batch_size, dirichlet_alpha=None
        )
        row = np.zeros(81, dtype=np.int64)
        for m, i in legal:
            row[m * 9 + i] = counts.get((m, i), 0)
        total = int(row.sum())
        # sims - 1, not sims: iteration 1 selects the unexpanded ROOT as its
        # leaf (expansion happens on that visit), so exactly one of the
        # budget's visits lands on the root itself, outside the children.
        if total != sims - 1:
            raise SystemExit(f"battery position {idx}: visit sum {total} != sims-1 {sims - 1}")
        histories.append([m * 9 + i for m, i in history])
        rows.append(row)
        values.append(float(_root_value(_root)))
        meta.append({"kind": "open" if idx < n_open else "mid", "depth": len(history)})
        print(
            f"  [{idx + 1}/{n}] kind={meta[-1]['kind']} depth={len(history)} "
            f"value={values[-1]:+.4f}",
            flush=True,
        )
    return histories, rows, values, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="models/az_ultimate.pt.archive")
    ap.add_argument(
        "--out",
        default=os.path.join("tests", "golden"),
        help="output directory (relative paths resolve against the repo root)",
    )
    ap.add_argument("--positions", type=int, default=DEFAULT_POSITIONS)
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument(
        "--gen-sims",
        type=int,
        default=64,
        help="midgame generator self-play sims (parity_search convention)",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args()

    import torch

    model, label = parity_search._load_model(a.weights)
    print(f"freeze_wave_golden — weights {label}, device {DEVICE}")
    tmp = tempfile.mkdtemp(prefix="wave-golden-")
    sha = export_safetensors(model, os.path.join(tmp, "w.safetensors"))
    print(f"weights sha256: {sha}")

    print(
        f"generating + searching {a.positions} positions (seed={a.seed}, "
        f"sims={a.sims}, batch_size={a.batch_size})…",
        flush=True,
    )
    histories, rows, values, battery = build_battery(
        a.seed, a.positions, model, a.gen_sims, a.sims, a.batch_size
    )

    out_dir = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "wave_golden.npz")
    json_path = os.path.join(out_dir, "wave_golden.json")

    width = max(len(h) for h in histories)
    moves = np.full((len(histories), width), -1, dtype=np.int32)
    for r, hist in enumerate(histories):
        moves[r, : len(hist)] = hist
    np.savez(
        npz_path,
        moves=moves,
        counts=np.stack(rows).astype(np.int64),
        values=np.asarray(values, dtype=np.float64),
    )

    metadata = {
        "format_version": FORMAT_VERSION,
        "recipe": "pure-net",
        "game": "ultimate",
        "sims": a.sims,
        "batch_size": a.batch_size,
        "c_puct": 1.5,
        "rollout_weight": 0.0,
        "dirichlet_alpha": None,
        "seed": a.seed,
        "gen_sims": a.gen_sims,
        "positions": len(histories),
        "generator": (
            "parity_search._gen_positions conventions: first half seeded random "
            "openings (depth cycling 0..8, rng seed*1009+idx), second half "
            "self-play midgames (dirichlet auto/0.25, temp 1.0, gen_sims plies); "
            "npz rows: moves=m*9+i codes (-1 pad), counts=int visits by move code, "
            "values=visit-weighted root value"
        ),
        "weights": a.weights.replace("\\", "/"),
        "weights_sha": sha,
        "device": str(DEVICE),
        "torch": torch.__version__,
        "battery": battery,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"wrote {npz_path} ({os.path.getsize(npz_path)} bytes)")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
