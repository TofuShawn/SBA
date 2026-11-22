"""Burn-vs-torch inference parity (Task 12), pinned at 1e-5.

A seeded AZNet(9, 8, 1) is exported to safetensors (Task 7 contract), a
seeded random plane batch goes through bench_burn's --dump-out mode (raw
f32 LE rows: softmax(policy) [n, 81] and value [n]), and Python compares
them against its own torch.softmax / value reference. The dump keeps the
comparison process non-torch-capable by construction — the exe never
talks to torch, Python never touches burn.

Skipped wherever no bench_burn binary is found (CI Windows leg, exe-less
machines), the same pattern as the exe-dependent tests.
"""

import subprocess

import numpy as np
import pytest
import torch

from az import rustselfplay

EXE = rustselfplay.find_bench()  # cpu debug build on CI, wgpu release locally
pytestmark = pytest.mark.skipif(EXE is None, reason="bench_burn not built")


def test_burn_matches_torch_within_1e5(tmp_path):
    from az.alphazero import AZNet, export_safetensors

    torch.manual_seed(3)
    model = AZNet(9, 8, 1).eval()
    weights = tmp_path / "w.safetensors"
    export_safetensors(model, str(weights))

    rng = np.random.default_rng(5)
    batch = rng.standard_normal((64, 4, 9, 9), dtype=np.float32)
    (tmp_path / "input.bin").write_bytes(batch.tobytes())

    out = subprocess.run(
        [EXE, str(weights), str(tmp_path / "input.bin"), "--dump-out", str(tmp_path / "out")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "parity" not in out.stdout  # no refs → no parity line
    policy_raw = np.fromfile(tmp_path / "out_policy.bin", dtype="<f4")
    value_raw = np.fromfile(tmp_path / "out_value.bin", dtype="<f4")
    assert policy_raw.size == 64 * 81, policy_raw.size
    assert value_raw.size == 64, value_raw.size
    policy = policy_raw.reshape(64, 81)
    value = value_raw.reshape(64)
    with torch.no_grad():
        p_ref, v_ref = model(torch.as_tensor(batch))
    p_ref = torch.softmax(p_ref, dim=1).numpy()
    v_ref = v_ref.numpy()[:, 0]
    err_p = float(np.abs(policy - p_ref).max())
    err_v = float(np.abs(value - v_ref).max())
    assert err_p < 1e-5, err_p
    assert err_v < 1e-5, err_v
