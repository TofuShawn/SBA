# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Golden vectors: the FROZEN pure-Python layer-1 encode reference.

``tests/golden/encode_golden.*`` holds the pure-Python ``alphazero.encode()``
answers — the numpy path; the battery was built on the pure-Python
``UltimateGame`` directly, so the ``isinstance(RustUltimateGame)`` branch
inside ``encode`` could never hijack the reference — over a fixed seeded,
MODEL-FREE battery produced by ``src/tools/freeze_encode_golden.py``
(parity_search opening conventions + random-playout snapshots including
terminal positions; documented seeds in the JSON sidecar). At freeze time
the Rust state layer already agreed with every row BITWISE, terminal rows
included.

Two tests, two lifetimes:

* ``test_rust_planes_match_golden_battery`` — THE anchor. Rebuilds every
  position on the Rust state layer (``rustbridge`` cdylib — no torch, no
  alphazero) and requires BITWISE equality with the goldens: discrete
  data, integer-equality semantics, not a tolerance. This survives the
  Python retirement; it is skipped only where the cdylib is missing.
* ``test_python_encode_matches_golden_battery`` — the reference-side
  check while the Python stack lives: rebuilds on the pure-Python
  ``UltimateGame`` (never ``make_game``, which honors config.toml) and
  re-encodes. Retires with ③.

Regenerating the goldens is a deliberate encoder change, never a test fix:
  python src/tools/freeze_encode_golden.py
"""

import json
import os

import numpy as np
import pytest

from az import rustbridge

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_NPZ = os.path.join(HERE, "golden", "encode_golden.npz")
GOLDEN_JSON = os.path.join(HERE, "golden", "encode_golden.json")


def _load_golden():
    with open(GOLDEN_JSON, encoding="utf-8") as fh:
        meta = json.load(fh)
    gold = np.load(GOLDEN_NPZ)
    moves, planes = gold["moves"], gold["planes"]
    assert meta["format_version"] == 1
    n = meta["positions"]
    assert moves.shape[0] == planes.shape[0] == len(meta["battery"]) == n
    assert planes.shape[1:] == (4, 9, 9)
    return meta, moves, planes


def _rebuild(game_cls, history):
    """Replay one recorded history, asserting legality ply by ply (the
    goldens are worthless if the battery cannot be rebuilt)."""
    g = game_cls()
    for m, i in history:
        assert (m, i) in g.legal_moves(), f"recorded move ({m},{i}) is illegal on rebuild"
        g.make_move(m, i)
    return g


def test_rust_planes_match_golden_battery():
    """THE parity anchor: the Rust state layer's planes() must reproduce
    the frozen pure-Python encode for every battery position — bitwise,
    terminal rows included."""
    if not rustbridge.available():
        pytest.skip("sba_game cdylib not built")
    meta, moves, planes = _load_golden()
    bad = []
    for idx in range(meta["positions"]):
        hist = [int(c) for c in moves[idx] if c >= 0]
        g = _rebuild(rustbridge.RustUltimateGame, [(c // 9, c % 9) for c in hist])
        row = meta["battery"][idx]
        if bool(g.is_over()) != bool(row["terminal"]):
            bad.append(
                (idx, f"terminal flag mismatch: rust {g.is_over()} vs golden {row['terminal']}")
            )
        elif row["terminal"] and g.result() != row["result"]:
            bad.append((idx, f"result mismatch: rust {g.result()} vs golden {row['result']}"))
        elif not np.array_equal(g.sba_planes_tensor(), planes[idx]):
            diff = np.argwhere(g.sba_planes_tensor() != planes[idx])[:5]
            bad.append((idx, f"planes differ at {diff.tolist()}"))
    assert not bad, f"{len(bad)}/{meta['positions']} battery rows mismatch (first: {bad[:3]})"


def test_python_encode_matches_golden_battery():
    """The reference-side half: the pure-Python encode still reproduces
    the frozen answers (retires with ③, unlike the rust half above)."""
    pytest.importorskip("torch")
    from az.alphazero import encode  # deferred: the torch-gated reference

    from core.game import UltimateGame  # the pure-Python reference, not the Rust one

    meta, moves, planes = _load_golden()
    bad = []
    for idx in range(meta["positions"]):
        hist = [int(c) for c in moves[idx] if c >= 0]
        g = _rebuild(UltimateGame, [(c // 9, c % 9) for c in hist])
        if not np.array_equal(encode(g).numpy(), planes[idx]):
            diff = np.argwhere(encode(g).numpy() != planes[idx])[:5]
            bad.append((idx, f"planes differ at {diff.tolist()}"))
    assert not bad, f"{len(bad)}/{meta['positions']} battery rows mismatch (first: {bad[:3]})"
