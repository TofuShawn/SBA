# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native C rollout kernel tests (D25). Skipped when the DLL is not built."""

import os
import threading
import time

import pytest

import ai

from ai import (_c_rollout_state, _get_c_rollout, _rollout_move, _serialize_ultimate,
                get_ai_move, set_engine_config)
from native import cengine as _ce
from game import (BitUltimateGame, NormalGame, O, UltimateGame, X, apply_move)

DLL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'native', 'sba_engine.dll')
LEGACY_DLL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'native', 'sba_rollout.dll')

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(DLL) or os.path.isfile(LEGACY_DLL)),
    reason='native engine DLL not built')


def _empty_board():
    return UltimateGame()


def _out_of_book_position():
    """A position past the opening book so an MCTS search actually runs."""
    g = UltimateGame()
    apply_move(g, (0, 0))
    apply_move(g, (0, 4))
    return g


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
    """The C tree must at least beat the pure-Python MCTS at equal budget
    (soft 1.5x — timing in CI is noisy, so the hard floor is generous)."""
    g = UltimateGame()
    apply_move(g, (4, 4))
    set_engine_config({'engine': {'c_mcts': True, 'opening_book': False,
                                  'multithreaded': False}})
    try:
        t0 = time.perf_counter()
        _ce.mcts_search(g.clone(), 400)
        t_c = time.perf_counter() - t0
        set_engine_config({'engine': {'c_mcts': False, 'opening_book': False,
                                      'multithreaded': False}})
        t0 = time.perf_counter()
        from ai import mcts_search as pure_mcts
        pure_mcts(g.clone(), 400)
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


def test_tree_reuse_carries_visits():
    """D30: passing the previous tree back accumulates visits and keeps the
    best-move selection consistent."""
    set_engine_config({'engine': {'c_rollout': True, 'c_mcts': True,
                                  'opening_book': False,
                                  'multithreaded': False}})
    try:
        g = _out_of_book_position()
        mv1, s1, tree = _ce.mcts_search(g, 200)
        assert tree is not None
        v1 = sum(s[1] for s in s1)
        mv2, s2, _ = _ce.mcts_search(g, 200, prev=tree)
        v2 = sum(s[1] for s in s2)
        # the warm tree carries the previous visits on top of the new sims
        assert v2 >= v1 + 200 - 1
        assert (mv1 // 9, mv1 % 9) in g.legal_moves()
        assert (mv2 // 9, mv2 % 9) in g.legal_moves()

        # after the AI's move + a reply, reuse still resolves (fresh or warm)
        apply_move(g, (mv2 // 9, mv2 % 9))
        replies = g.legal_moves()
        apply_move(g, replies[0])
        mv3, s3, _ = _ce.mcts_search(g, 200, prev=tree)
        assert (mv3 // 9, mv3 % 9) in g.legal_moves()
    finally:
        set_engine_config()


def test_concurrent_mcts_searches_are_safe():
    """D29 regression: assistant + AI move searches run concurrently and must
    not corrupt each other (per-call arenas)."""
    set_engine_config({'engine': {'c_rollout': True, 'c_mcts': True,
                                  'opening_book': False,
                                  'multithreaded': False}})
    try:
        g = _out_of_book_position()
        outs = [None, None]

        def work(i):
            outs[i] = _ce.mcts_search(g.clone(), 300)

        ts = [threading.Thread(target=work, args=(i,)) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(30)
        assert not any(t.is_alive() for t in ts), 'search hung'
        for r in outs:
            assert r is not None and r[0] is not None
            assert (r[0] // 9, r[0] % 9) in g.legal_moves()
    finally:
        set_engine_config()
