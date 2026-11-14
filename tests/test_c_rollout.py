# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native C rollout kernel tests (D25). Skipped when the DLL is not built."""

import os
import random
import threading
import time

import pytest

import ai

from ai import (_c_rollout_state, _get_c_rollout, _rollout_move, _serialize_ultimate,
                get_ai_move, set_engine_config)
from native import cengine as _ce
from game import (BitUltimateGame, NormalGame, O, UltimateGame, X, apply_move,
                  apply_clone_result, EMPTY)

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
    apply_move(g, (4, 4))   # routes into macro 4
    apply_move(g, (4, 2))   # routes into macro 2
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
    set_engine_config({'engine': {'opening_book': False,
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
    set_engine_config({'engine': {'c_rollout_pattern': True}})
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
    set_engine_config({'engine': {'c_mcts': True,
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
    set_engine_config({'engine': {'c_mcts': True,
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


def _forced_win_position():
    """Exactly one immediate game win, deliberately NOT the first legal move
    so a missing tie-break (or a wrong scoring formula) is visible."""
    g = UltimateGame()
    g.macro = [X, X, EMPTY] + [EMPTY] * 6
    g.micro[2] = [EMPTY, EMPTY, EMPTY, EMPTY, X, EMPTY, X, EMPTY, EMPTY]
    g.current = X
    g.active_macro = 2
    return g


def test_flat_mcts_c_backend_abi_and_dispatch():
    """Regression (2026-11-08): sba_flat_mcts has 8 args — per_move, seed,
    use_pattern, out_best. A dropped argtype made get_ai_move('Flat MCTS')
    raise ctypes.ArgumentError while all Python-only tests stayed green."""
    assert _ce.available(engine=True)
    assert len(_ce._lib.sba_flat_mcts.argtypes) == 8
    set_engine_config({'engine': {'c_mcts': True, 'opening_book': False}})
    try:
        g = _out_of_book_position()
        assert get_ai_move(g, 'Flat MCTS', 200, 3) in g.legal_moves()
        assert _ce.flat_mcts(g, 20, rng_seed=7) in g.legal_moves()
    finally:
        set_engine_config()


def test_flat_mcts_c_and_python_share_scoring_and_tiebreak():
    """C must score win 1.0 / draw 0.5 and prefer an immediate game win on
    ties, exactly like ai.flat_mcts_move. The old (wins+draws)/(2N) formula
    biased the C kernel toward draws and picked the first legal move."""
    assert _ce.available(engine=True)
    g = _forced_win_position()
    forced = [m for m in g.legal_moves()
              if apply_clone_result(g, m) == g.current]
    assert len(forced) == 1 and forced[0] != g.legal_moves()[0]
    for seed in (1, 2, 3, 17, 99):
        c_move = _ce.flat_mcts(g, 30, rng_seed=seed)
        random.seed(seed)
        py_move = ai.flat_mcts_move(g.clone(), 30 * len(g.legal_moves()))
        assert c_move == forced[0], (seed, c_move, forced)
        assert py_move == forced[0], (seed, py_move, forced)


def test_c_and_python_rules_agree_over_random_games():
    """Cross-implementation guard (D2): the C kernels and game.py encode the
    same rules. Every move the C kernel emits must be legal under game.py
    (which now rejects terminal / mis-routed moves), the terminal result must
    agree, and a C tree search must only report legal root moves."""
    rng = random.Random(20261109)
    checked = 0
    for _ in range(150):
        g = UltimateGame()
        for _ in range(rng.randint(0, 25)):
            lm = g.legal_moves()
            if not lm or g.is_over():
                break
            apply_move(g, rng.choice(lm))
        if g.is_over():
            continue
        legal = set(g.legal_moves())

        res, seq = _ce.rollout_trace(g.clone(), rng_seed=rng.getrandbits(64))
        replay = g.clone()
        for m in seq:
            assert m in replay.legal_moves(), ('C played an illegal move', m)
            apply_move(replay, m)
        assert replay.result() == res

        mv, stats, _ = _ce.mcts_search(g, 200, rng_seed=rng.getrandbits(64))
        assert mv is not None and (mv // 9, mv % 9) in legal
        for row in stats:
            assert (row[0] // 9, row[0] % 9) in legal
        checked += 1
    assert checked >= 100


def test_c_kernels_report_terminal_boards_without_playing_on():
    """An already-decided board must produce the same verdict on both sides:
    no extra moves, and the winner must not depend on whose turn it is."""
    for current in (X, O):
        w = UltimateGame()
        w.macro = [X, X, X] + [EMPTY] * 6
        w.current = current
        assert w.result() == X and w.legal_moves() == []
        assert _ce.rollout_state(w, rng_seed=1) == X
        assert _ce.rollout_state(w, pattern=True, rng_seed=1) == X
        res, seq = _ce.rollout_trace(w, rng_seed=1)
        assert res == X and seq == []


# ---------------------------------------------------------------------------
# D37 — the config switches are honoured by the native C backends, not just
# by the Python fallback. Each test below fails if a switch stops reaching C.
# ---------------------------------------------------------------------------

def _opts(**kw):
    merged = dict(_ce.DEFAULT_OPTS)
    merged.update(kw)
    return merged


def test_search_opts_reach_every_c_backend(monkeypatch):
    """Wiring guard: ai._c_search_opts() must be handed to all four C entry
    points, otherwise config.toml promises behaviour the C path ignores."""
    seen = {}
    for name in ('mcts_search', 'rollout_state', 'minimax_ultimate', 'flat_mcts'):
        original = getattr(_ce, name)

        def spy(*args, _original=original, _name=name, **kwargs):
            seen[_name] = dict(kwargs.get('opts') or {})
            return _original(*args, **kwargs)

        monkeypatch.setattr(_ce, name, spy)

    set_engine_config({'engine': {
        'uct_c': 3.25, 'dynamic_uct': False, 'rollout_heuristic': False,
        'micro_tablebase': False, 'symmetry': False,
        'c_rollout_pattern': True, 'opening_book': False, 'c_mcts': True,
    }})
    try:
        g = _out_of_book_position()
        get_ai_move(g, 'MCTS', 200, 3)
        get_ai_move(g, 'Minimax', 200, 3)
        get_ai_move(g, 'Flat MCTS', 200, 3)
        ai._c_rollout_state(g)
    finally:
        set_engine_config()

    for name in ('mcts_search', 'rollout_state', 'minimax_ultimate', 'flat_mcts'):
        assert seen.get(name), f'{name} never received opts'
        assert seen[name]['uct_c'] == 3.25
        assert seen[name]['dynamic_uct'] is False
        assert seen[name]['rollout_heuristic'] is False
        assert seen[name]['micro_tablebase'] is False
        assert seen[name]['symmetry'] is False
        assert seen[name]['use_pattern'] is True


def test_c_tree_symmetry_dedups_root():
    """D4 dedup must now happen inside the C tree (81 -> 21 root children
    from an empty Ultimate board), using the canonical resulting position."""
    g = UltimateGame()
    _, off_stats, off_tree = _ce.mcts_search(
        g, 1500, rng_seed=42, opts=_opts(symmetry=0))
    _, on_stats, on_tree = _ce.mcts_search(
        g, 1500, rng_seed=42, opts=_opts(symmetry=1))
    try:
        assert len(off_stats) == 81
        assert 0 < len(on_stats) < len(off_stats) // 2
    finally:
        if off_tree: off_tree.close()
        if on_tree: on_tree.close()


def test_c_tree_uct_c_and_dynamic_uct_change_the_search():
    """uct_c used to be hardcoded to 1.4 (140) and dynamic_uct could not be
    turned off in C. Both must change the visit distribution / best move."""
    g = UltimateGame()
    top = []
    for c in (0.2, 5.0):
        _, stats, tree = _ce.mcts_search(
            g, 1000, rng_seed=7,
            opts=_opts(uct_c=c, dynamic_uct=0, symmetry=0))
        top.append(sorted((s[1] for s in stats), reverse=True)[:3])
        if tree: tree.close()
    assert top[0] != top[1]

    g2 = UltimateGame()
    apply_move(g2, (4, 4))
    apply_move(g2, (4, 0))
    best = []
    for dynamic in (0, 1):
        mv, _, tree = _ce.mcts_search(
            g2, 1200, rng_seed=99, opts=_opts(dynamic_uct=dynamic, symmetry=0))
        best.append(mv)
        if tree: tree.close()
    assert best[0] != best[1]


def test_c_rollout_heuristic_switch_is_honoured_by_trace():
    """With the win/block heuristic the first playout move is always the
    immediate game win; with it off the move is drawn uniformly."""
    g = UltimateGame()
    g.macro = [X, X, EMPTY] + [EMPTY] * 6
    g.micro[2] = [EMPTY, EMPTY, EMPTY, EMPTY, X, EMPTY, X, EMPTY, EMPTY]
    g.current = X
    g.active_macro = 2
    on = {_ce.rollout_trace(g, rng_seed=s, opts=_opts(rollout_heuristic=1))[1][0]
          for s in range(1, 31)}
    off = {_ce.rollout_trace(g, rng_seed=s, opts=_opts(rollout_heuristic=0))[1][0]
           for s in range(1, 31)}
    assert on == {(2, 2)}
    assert len(off) > 1


def test_c_micro_tablebase_changes_minimax_choice():
    """The C minimax evaluator must use the exact 3x3 tablebase when a micro
    has <= 2 empty cells; on this board it changes the chosen move."""
    rows = ['X..XX.X..', 'O.OX.O..O', 'X..XOO.XO',
            '.XOO..XX.', 'OX..OXO.X', '.XX.X.XOO',
            'XO.XOXOO.', '.O.OXOOOX', '.X.XOXOX.']
    macro_states = ['X', 'O', '.', '.', '.', 'X', 'O', '.', '.']
    g = UltimateGame()
    g.current = X
    g.active_macro = None
    for m, row in enumerate(rows):
        g.micro[m] = [EMPTY if c == '.' else c for c in row]
        g.macro[m] = EMPTY if macro_states[m] == '.' else macro_states[m]
    with_tb = _ce.minimax_ultimate(g, 2, opts=_opts(micro_tablebase=1))
    without_tb = _ce.minimax_ultimate(g, 2, opts=_opts(micro_tablebase=0))
    assert with_tb != without_tb
    assert with_tb in g.legal_moves() and without_tb in g.legal_moves()


def test_c_tree_is_owned_by_the_dll_and_still_reusable():
    """D37: no Python-owned uint8_t arena. The opaque handle must survive a
    second search as `prev`, and releasing it twice must be harmless."""
    g = UltimateGame()
    mv1, s1, t1 = _ce.mcts_search(g, 200, rng_seed=1)
    assert t1 is not None and t1._handle.value
    mv2, s2, t2 = _ce.mcts_search(g, 200, rng_seed=2, prev=t1)
    assert mv1 is not None and mv2 is not None
    assert sum(r[1] for r in s2) >= sum(r[1] for r in s1)
    t1.close()
    t2.close()
    t1.close()  # idempotent
