# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Golden vectors: the FROZEN Python wave-search reference (D58 pure-net recipe).

``tests/golden/wave_golden.*`` holds the Python search's own answers —
``alphazero.mcts_search`` at sims=200, batch_size=256, c_puct=1.5, dirichlet
off, rollout_weight=0.0 (pure wave search over pure network values) — over a
fixed seeded battery produced by ``src/tools/freeze_wave_golden.py`` with the
``parity_search._gen_positions`` conventions (random openings depth 0..8 +
self-play midgames, documented seeds in the JSON sidecar). Layer 3 proved
the Rust wave search byte-equal against that Python search on this recipe;
this test pins the RUST side (``sba_debug search``) against the frozen
goldens forever: visit counts must match by INTEGER EQUALITY (not a
tolerance — the vectors are discrete and were byte-equal at freeze time) and
the root value within 1e-6.

If ``mcts_search`` (or most of the Python tree) is ever retired, THIS test
is the parity anchor that survives it: it imports neither — it rebuilds each
position on the pure-Python ``UltimateGame`` reference, re-exports the
archived checkpoint to safetensors and compares the sha256 against the
golden metadata (so a silently changed checkpoint cannot false-pass), then
drives the Rust exe. The goldens are only valid for the weights blob named
in the sidecar.

Regenerating the goldens is a deliberate recipe change, never a test fix:
  python src/tools/freeze_wave_golden.py --weights models/az_ultimate.pt.archive

Skipped wherever no sba_debug binary is found (CI Windows leg) or the
archived checkpoint is absent (models/ is gitignored) — the same pattern as
the other exe-dependent tests.
"""

import hashlib
import json
import os
import subprocess
import tempfile

import numpy as np
import pytest

from az import rustselfplay

EXE = rustselfplay.find_exe(name="sba_debug")  # cpu debug build on CI, wgpu release locally
pytestmark = pytest.mark.skipif(EXE is None, reason="sba_debug not built")

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_NPZ = os.path.join(HERE, "golden", "wave_golden.npz")
GOLDEN_JSON = os.path.join(HERE, "golden", "wave_golden.json")
WEIGHTS_ARCHIVE = os.path.join(HERE, "..", "models", "az_ultimate.pt.archive")
VALUE_TOL = 1e-6
SPAWN_TIMEOUT = 600  # per-search budget, mirrors parity_search


def _reexport_weights(out_path):
    """Write the archived checkpoint's float tensors as safetensors and
    return the file's sha256 — replicating alphazero.export_safetensors
    WITHOUT importing it (this file must outlive the Python search)."""
    import torch
    from safetensors.torch import save_file

    ckpt = torch.load(WEIGHTS_ARCHIVE, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise AssertionError("archive is not a _save_model checkpoint")
    sd = {
        k: v.detach().cpu().contiguous()
        for k, v in ckpt["state_dict"].items()
        if v.dtype.is_floating_point  # int64 num_batches_tracked is torch-only
    }
    save_file(sd, out_path)
    with open(out_path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_rust_search_matches_golden_battery(tmp_path):
    """Every battery position: rebuild from the recorded move history, run
    the Rust wave search, compare against the frozen Python reference."""
    with open(GOLDEN_JSON, encoding="utf-8") as fh:
        meta = json.load(fh)
    gold = np.load(GOLDEN_NPZ)
    moves, counts, values = gold["moves"], gold["counts"], gold["values"]
    assert meta["format_version"] == 1
    assert meta["recipe"] == "pure-net"
    n = meta["positions"]
    assert len(meta["battery"]) == moves.shape[0] == counts.shape[0] == values.shape[0] == n

    if not os.path.isfile(WEIGHTS_ARCHIVE):
        pytest.skip("archived checkpoint not present (models/ is gitignored)")
    weights = os.path.join(tempfile.mkdtemp(prefix="wave-golden-w-"), "w.safetensors")
    sha = _reexport_weights(weights)
    assert sha == meta["weights_sha"], (
        "checkpoint drifted: re-exported sha256 differs from the golden "
        "metadata — the goldens are only valid for the frozen weights blob"
    )

    from core.game import UltimateGame  # the pure-Python reference, not the Rust one

    sims_args = ["--sims", str(meta["sims"]), "--batch-size", str(meta["batch_size"])]
    bad = []
    for idx in range(n):
        hist = [int(c) for c in moves[idx] if c >= 0]
        g = UltimateGame()
        for code in hist:
            m, i = divmod(code, 9)
            assert (m, i) in g.legal_moves(), f"pos {idx}: recorded move {code} illegal"
            g.make_move(m, i)
        assert not g.is_over(), f"pos {idx}: recorded history ends terminal"
        want = counts[idx].astype(np.int64)
        assert int(want.sum()) == meta["sims"] - 1, f"pos {idx}: corrupt golden visit sum"
        moves_arg = ";".join(f"{c // 9},{c % 9}" for c in hist)
        proc = subprocess.run(
            [EXE, "search", "--weights", weights, "--moves", moves_arg, *sims_args],
            capture_output=True,
            text=True,
            timeout=SPAWN_TIMEOUT,
        )
        if proc.returncode != 0:
            bad.append((idx, f"exe rc={proc.returncode}: {proc.stderr[-200:]!r}"))
            continue
        ev = json.loads(proc.stdout.strip().splitlines()[-1])
        got = np.zeros(81, dtype=np.int64)
        for code, visits in ev["counts"].items():
            got[int(code)] = int(visits)
        vdelta = abs(float(ev["value"]) - float(values[idx]))
        if not np.array_equal(want, got):
            diff = np.nonzero(want != got)[0][:5]
            bad.append(
                (
                    idx,
                    f"visit counts differ at codes {diff.tolist()}: "
                    f"golden {want[diff].tolist()} vs rust {got[diff].tolist()}",
                )
            )
        elif vdelta > VALUE_TOL:
            bad.append((idx, f"root value delta {vdelta:.2e} > {VALUE_TOL}"))
    assert not bad, (
        f"{len(bad)}/{n} battery positions mismatch the frozen Python reference (first: {bad[:3]})"
    )
