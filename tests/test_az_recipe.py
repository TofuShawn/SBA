"""Recipe unit tests: D4 flip augmentation and root-mean value target."""

import numpy as np
import torch

import alphazero
from game import UltimateGame


def _one_hot_state(size, idx, plane=0):
    s = torch.zeros(4, size, size)
    r, c = divmod(idx, size)
    s[plane, r, c] = 1.0
    return s


def test_d4_policy_perms_are_bijections():
    for size in (3, 9):
        perms = alphazero._aug_perms(size)
        assert len(perms) == 8
        for perm in perms:
            assert sorted(perm) == list(range(size * size))


def test_d4_state_and_policy_transform_together():
    for size in (3, 9):
        perms = alphazero._aug_perms(size)
        for t in range(8):
            idx = 7  # arbitrary cell
            state = _one_hot_state(size, idx)
            target = np.zeros(size * size, dtype=np.float32)
            target[idx] = 1.0
            s2, t2 = alphazero._augment_sample(state, target, t, size)
            expected_idx = perms[t][idx]
            assert np.argmax(t2) == expected_idx
            r, c = divmod(expected_idx, size)
            assert s2[0, r, c].item() == 1.0
            assert s2[0].sum().item() == 1.0


def test_root_value_weighted_mean():
    class FakeChild:
        def __init__(self, visits, wins):
            self.visits = visits
            self.wins = wins

    class FakeRoot:
        mover = None

        def __init__(self):
            self.children = [FakeChild(10, 8.0), FakeChild(5, 2.0),
                             FakeChild(0, 0.0)]

    v = alphazero._root_value(FakeRoot())
    expected = (10 * 0.8 + 5 * 0.4) / 15
    assert abs(v - expected) < 1e-9
    assert alphazero._root_value(type('R', (), {'children': []})()) == 0.0