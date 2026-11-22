"""The encode contract across the language boundary (Task 11).

The sba_debug exe dumps random-game planes (324 f32 LE per position) plus
the moves that produced them; this test replays those moves on the
pure-Python UltimateGame reference and compares alphazero.encode() BITWISE
against the dumped planes (np.array_equal — discrete data, no tolerance).
Skipped wherever no sba_debug binary is found (CI Windows leg, exe-less
machines), the same pattern as the DLL-dependent tests.
"""

import json
import os
import subprocess

import numpy as np
import pytest

from az import rustselfplay

EXE = rustselfplay.find_exe(name="sba_debug")  # cpu debug build on CI, wgpu release locally
pytestmark = pytest.mark.skipif(EXE is None, reason="sba_debug not built")


def _run_planes_dump(tmp_path, games=3, seed=7, moves=10):
    out = tmp_path / "dump"
    subprocess.run(
        [
            EXE,
            "planes",
            "--games",
            str(games),
            "--seed",
            str(seed),
            "--moves",
            str(moves),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def test_rust_planes_match_python_encode_bitwise(tmp_path):
    pytest.importorskip("torch")  # the reference side IS alphazero.encode
    from core.game import UltimateGame  # the pure-Python reference, not the Rust one

    out = _run_planes_dump(tmp_path)
    bins = sorted(g for g in os.listdir(out) if g.endswith(".bin"))
    assert bins, "the exe dumped no planes_*.bin files"
    for g in bins:
        with open(out / g.replace(".bin", ".meta.json")) as f:
            meta = json.load(f)
        assert meta["seed"] == 7
        planes = np.fromfile(out / g, dtype="<f4").reshape(4, 9, 9)
        game = UltimateGame()
        for m, i in meta["moves"]:
            assert (m, i) in game.legal_moves()
            game.make_move(m, i)
        from az.alphazero import encode

        ref = encode(game).numpy()
        assert np.array_equal(ref, planes), g  # bitwise — discrete data
