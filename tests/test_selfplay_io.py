import numpy as np
import torch

from alphazero import AZNet, export_safetensors


def _write_fake_record(path, t=3, seed=11):
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    tensors = {
        "states": rng.standard_normal((t, 4, 9, 9), dtype=np.float32),
        "policy": rng.random((t, 81), dtype=np.float32),
        "z": rng.uniform(-1, 1, (t,)).astype(np.float32),
    }
    save_file(
        tensors,
        str(path),
        metadata={
            "format_version": "1",
            "game_type": "ultimate",
            "seed": str(seed),
            "result": "D",
            "sims": "8",
            "weights_sha": "deadbeef",
        },
    )
    return tensors


def test_export_safetensors_roundtrip_names(tmp_path):
    model = AZNet(9, 8, 1)
    path = tmp_path / "w.safetensors"
    sha = export_safetensors(model, str(path))
    assert len(sha) == 64
    from safetensors.numpy import load_file

    data = load_file(str(path))
    expected = {k for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    assert set(data) == expected  # int64 num_batches_tracked filtered out


def test_load_records_matches_replay_contract(tmp_path):
    import rustselfplay

    tensors = _write_fake_record(tmp_path / "g00000.safetensors")
    _write_fake_record(tmp_path / "g00001.safetensors", t=5, seed=12)
    games = rustselfplay.load_records(str(tmp_path))
    assert len(games) == 2
    states, targets, zs = games[0]
    assert len(states) == 3 and len(targets) == 3 and len(zs) == 3
    assert torch.is_tensor(states[0]) and states[0].shape == (4, 9, 9)
    assert states[0].dtype == torch.float32
    assert np.asarray(targets[0]).shape == (81,)
    assert isinstance(zs[0], float)
    # values must match the file bitwise
    assert np.allclose(states[2].numpy(), tensors["states"][2])


def test_load_records_missing_dir_is_empty():
    import rustselfplay

    assert rustselfplay.load_records(str("/nonexistent-dir-sba")) == []
