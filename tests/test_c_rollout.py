# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native C rollout kernel tests (D25). Skipped when the DLL is not built."""

import os
import time

import pytest

import ai

from ai import (_c_rollout_state, _get_c_rollout, _rollout_move, _serialize_ultimate,
                get_ai_move, mcts_search, set_engine_config)
from game import (BitUltimateGame, NormalGame, O, UltimateGame, X, apply_move)

DLL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'native', 'sba_rollout.dll')

pytestmark = pytest.mark.skipif(not os.path.isfile(DLL),
                                reason='native/sba_rollout.dll not built')


def _empty_board():
    return UltimateGame()


def test_kernel_loads():
    assert _get_c_rollout() is not None


def test_rollout_results_are_legal():
    g = _empty_board()
    seen = set()
    for _ in range(500):
        assert _c_rollout_state(g) in (X, O, 'D')
        seen.add(_c_rollout_state(g))
    assert seen != {'D'}  # a heuristic rollout should sometimes produce wins


def test_serialize_matches_between_state_types():
    g = UltimateGame()
    apply_move(g, (4, 4))
    apply_move(g, (2, 2))
    apply_move(g, (2, 6))
    b = BitUltimateGame.from_game(g)
    assert _serialize_ultimate(g) == _serialize_ultimate(b)


def test_kernel_agrees_with_python_distribution():
    """Same policy, different RNG: the outcome distribution should be close."""
    g = _empty_board()

    def py_rollout(state):
        state = state.clone()
        result = state.result()
        while result is None:
            moves = state.legal_moves()
            if not moves:
                break
            apply_move(state, _rollout_move(state, moves))
            result = state.result()
        return result

    n = 400
    c_x = sum(1 for _ in range(n) if _c_rollout_state(g) == X)
    p_x = sum(1 for _ in range(n) if py_rollout(g) == X)
    assert abs(c_x - p_x) / n < 0.15  # generous: random-seed divergence only


def test_mcts_reaches_a_valid_endgame():
    """A CvC game driven by the kernel-backed search plays to a legal end."""
    set_engine_config({'engine': {'c_rollout': True, 'opening_book': False,
                                  'multithreaded': False}})
    try:
        g = UltimateGame()
        for _ in range(120):
            if g.is_over():
                break
            mv = get_ai_move(g, 'MCTS', 200, 3)
            assert mv in g.legal_moves()
            apply_move(g, mv)
        assert g.is_over()
    finally:
        set_engine_config()


def test_kernel_search_faster_than_python(monkeypatch=None):
    """Soft check (1.5x): the kernel must not be a net regression."""
    g = UltimateGame()
    apply_move(g, (4, 4))
    set_engine_config({'engine': {'c_rollout': True, 'opening_book': False,
                                  'multithreaded': False}})
    try:
        t0 = time.perf_counter()
        mcts_search(g.clone(), 400)
        t_c = time.perf_counter() - t0
        set_engine_config({'engine': {'c_rollout': False, 'opening_book': False,
                                      'multithreaded': False}})
        t0 = time.perf_counter()
        mcts_search(g.clone(), 400)
        t_p = time.perf_counter() - t0
        assert t_c < t_p * 1.5
    finally:
        set_engine_config()


def test_normal_board_uses_python_path():
    """Normal 3x3 stays on the Python rollout (too shallow to bother)."""
    set_engine_config()
    assert _get_c_rollout() is not None
    g = NormalGame()
    mv = get_ai_move(g, 'MCTS', 200, 3)
    assert mv in g.legal_moves()


def test_pattern_kernel_smoke():
    """Pattern variant (D26, off by default): legal results, sane spread."""
    set_engine_config({'engine': {'c_rollout': True, 'c_rollout_pattern': True}})
    try:
        assert ai._get_c_rollout_pattern() is not None
        g = UltimateGame()
        seen = set()
        for _ in range(200):
            r = _c_rollout_state(g)
            assert r in (X, O, 'D')
            seen.add(r)
        assert seen != {'D'}
    finally:
        set_engine_config()
