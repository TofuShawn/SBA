"""AI engine correctness, sanity, and termination tests."""

import math
import random
import threading

import pytest

import ai
from ai import (
    get_basic_move, minimax_move_normal, mcts_move, solver_move,
    minimax_pro_move, mcts_rave_move, mcts_grave_move, build_tablebase,
    _board_key, get_ai_move, opening_book_move, build_micro_tablebase,
    _rollout_move, reset_engine_caches, _REUSE, _fork_count,
    mcts_move_parallel, _D4, position_win_rate, position_win_rates,
    flat_mcts_move, analyze_position, set_engine_config, _diff_move,
)
from game import (
    X, O, EMPTY, NormalGame, UltimateGame, BitUltimateGame, apply_move,
)


def test_basic_takes_winning_move():
    g = NormalGame()
    g.board = [X, X, EMPTY] + [EMPTY] * 6
    g.current = X
    assert get_basic_move(g) == 2


def test_basic_blocks_opponent_win():
    g = NormalGame()
    g.board = [EMPTY] * 3 + [O, O, EMPTY] + [EMPTY] * 3
    g.current = X
    assert get_basic_move(g) == 5


def test_minimax_picks_immediate_win():
    g = NormalGame()
    g.board = [X, X, EMPTY] + [EMPTY] * 6
    g.current = X
    move, score = minimax_move_normal(g)
    assert move == 2 and score > 0


def test_minimax_x_never_loses_to_random():
    wins = draws = 0
    for _ in range(6):
        g = NormalGame()
        while not g.is_over():
            move = (minimax_move_normal(g)[0] if g.current == X
                    else random.choice(g.legal_moves()))
            g.make_move(move)
        if g.result() == X:
            wins += 1
        elif g.result() == 'D':
            draws += 1
    assert wins + draws == 6


def test_mcts_picks_immediate_win():
    g = NormalGame()
    g.board = [X, X, EMPTY] + [EMPTY] * 6
    g.current = X
    assert mcts_move(g, 1500) == 2


def test_flat_mcts_picks_immediate_win():
    g = NormalGame()
    g.board = [X, X, EMPTY] + [EMPTY] * 6
    g.current = X
    assert flat_mcts_move(g, 2000) == 2


def test_flat_mcts_blocks_immediate_loss():
    g = NormalGame()
    g.board = [X, EMPTY, EMPTY, O, O, EMPTY, EMPTY, EMPTY, EMPTY]
    g.current = X
    assert flat_mcts_move(g, 2000) == 5


def test_flat_mcts_legal_move_on_ultimate():
    g = UltimateGame()
    assert flat_mcts_move(g, 600) in g.legal_moves()


def test_mcts_legal_move_on_empty_ultimate():
    g = UltimateGame()
    assert mcts_move(g, 300) in g.legal_moves()


def test_solver_winning_move_and_tablebase():
    g = NormalGame()
    g.board = [X, X, EMPTY] + [EMPTY, O, O] + [EMPTY] * 3
    g.current = X
    mv = solver_move(g)
    b = g.board[:]
    b[mv] = g.current
    assert -build_tablebase()[_board_key(b)] == 1
    assert len(build_tablebase()) > 5000
    g2 = NormalGame()
    assert get_ai_move(g2, 'Solver', 300) in g2.legal_moves()


def test_immediate_win_across_engines():
    g = NormalGame()
    g.board = [X, X, EMPTY] + [EMPTY, O, O] + [EMPTY] * 3
    g.current = X
    assert minimax_pro_move(g, depth=9) == 2
    assert mcts_rave_move(g, 1500) == 2
    assert mcts_grave_move(g, 1500) == 2


def test_opening_book():
    g = NormalGame()
    assert opening_book_move(g) in (0, 2, 4, 6, 8)
    # after X's opener it is O's turn (filled == 1, current == O): the book
    # answers 4 unless X took 4, then a corner
    g2 = NormalGame()
    g2.board[4] = X
    g2.current = O
    assert opening_book_move(g2) in (0, 2, 6, 8)
    g3 = NormalGame()
    g3.board[0] = X
    g3.current = O
    assert opening_book_move(g3) == 4
    g4 = NormalGame()
    g4.board = [X, O] + [EMPTY] * 7
    g4.current = X
    assert opening_book_move(g4) is None  # past the book
    gu = UltimateGame()
    assert opening_book_move(gu) in gu.legal_moves()


def test_ultimate_opening_book_responses():
    from ai import ULTIMATE_BOOK
    for x_move, responses in ULTIMATE_BOOK.items():
        gu = UltimateGame()
        gu.make_move(*x_move)  # routes the next move into macro x_move[1]
        legal = set(gu.legal_moves())
        assert all(r in legal for r in responses)
        assert any(r[0] == x_move[1] for r in responses)
    # a real game follows the book for the first two plies, then falls back
    gu = UltimateGame()
    first = opening_book_move(gu)
    assert first in gu.legal_moves()
    gu.make_move(*first)
    resp = opening_book_move(gu)
    assert resp in gu.legal_moves()
    gu.make_move(*resp)
    assert opening_book_move(gu) is None


def test_micro_tablebase():
    tb = build_micro_tablebase()
    assert tb[_board_key([EMPTY] * 9)] == 0
    assert tb[_board_key([X, X, EMPTY, O, EMPTY, O, EMPTY, EMPTY, EMPTY])] == 1


def test_rollout_heuristic():
    g = NormalGame()
    g.board = [X, X, EMPTY] + [EMPTY] * 6
    g.current = X
    assert _rollout_move(g) == 2
    g2 = NormalGame()
    g2.board = [X, X, EMPTY, O] + [EMPTY] * 5
    g2.current = O
    assert _rollout_move(g2) == 2


def test_tree_reuse():
    # the opening book would answer the first two plies itself — turn it off
    # so the moves actually go through MCTS and populate the reuse cache
    set_engine_config({'engine': {'opening_book': False}})
    g = NormalGame()
    m1 = get_ai_move(g, 'MCTS', 100)
    apply_move(g, m1)
    m2 = get_ai_move(g, 'MCTS', 100)
    assert m2 in g.legal_moves()
    assert len(_REUSE) > 0
    reset_engine_caches()
    assert len(_REUSE) == 0
    set_engine_config()


def test_fork_count():
    assert _fork_count([EMPTY] * 9, X) == 0
    assert _fork_count([X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, X, X], X) == 1


def test_minimax_pro_legal_with_refinements():
    gu = UltimateGame()
    assert minimax_pro_move(gu, depth=3) in gu.legal_moves()


def test_symmetry():
    """D4 permutations are bijections; the canonical resulting-position key
    (which replaced the old cell-orbit dedup) must notice asymmetric boards."""
    assert len({tuple(p) for p in _D4}) == 8
    assert all(sorted(p) == list(range(9)) for p in _D4)
    g = UltimateGame()
    apply_move(g, (4, 4))
    apply_move(g, (4, 2))
    keys = {ai._ultt_sym_key(g, t) for t in range(8)}
    assert len(keys) > 1


def test_symmetry_keeps_forced_win_on_python_path():
    """Regression: the old cell-orbit dedup key merged non-equivalent root
    moves and made GRAVE miss a forced win in 21/30 seeds with symmetry on."""
    g = UltimateGame()
    g.macro = [X, X, EMPTY] + [EMPTY] * 6
    g.micro[2] = [EMPTY, EMPTY, EMPTY, EMPTY, X, EMPTY, X, EMPTY, EMPTY]
    g.current = X
    g.active_macro = 2
    set_engine_config({'engine': {'symmetry': True, 'c_mcts': False,
                                  'c_minimax': False, 'opening_book': False,
                                  'bitboard': False, 'tree_reuse': False}})
    try:
        for seed in (1, 2, 3):
            random.seed(seed)
            assert get_ai_move(g.clone(), 'MCTS+GRAVE', 200) == (2, 2)
    finally:
        set_engine_config()


def test_multithreaded_mcts_legal():
    gu = UltimateGame()
    assert mcts_move_parallel(gu, 100, 2) in gu.legal_moves()


def test_multithreaded_enabled_via_config(monkeypatch):
    """The multithreaded switch must actually route MCTS to the parallel
    merge path: c_mcts off (the C tree would take it first) and the opening
    book off (an empty board would return a book move without searching).
    Two workers at budget 200 -> two chunked searches, 100 sims each."""
    set_engine_config({'engine': {'multithreaded': True, 'workers': 2,
                                  'c_mcts': False, 'opening_book': False}})
    calls = []
    real = ai.mcts_search

    def spy(state, iterations, **kw):
        calls.append(iterations)
        return real(state, iterations, **kw)

    monkeypatch.setattr(ai, 'mcts_search', spy)
    try:
        gu = UltimateGame()
        apply_move(gu, (0, 0))
        apply_move(gu, (0, 4))
        assert get_ai_move(gu, 'MCTS', 200) in gu.legal_moves()
        assert len(calls) == 2, calls    # one search per worker thread
        assert sum(calls) == 200         # the budget split across threads
    finally:
        set_engine_config()


def test_win_rates():
    g = NormalGame()
    assert position_win_rate(g, 0) == 0.5
    assert position_win_rates(g, 0) == (0.0, 1.0, 0.0)
    g2 = NormalGame()
    g2.board = [X, X, X, EMPTY, O, EMPTY, EMPTY, EMPTY, O]
    g2.current = X
    assert position_win_rate(g2, 0) == 1.0
    assert position_win_rates(g2, 0) == (1.0, 0.0, 0.0)
    gu = UltimateGame()
    assert 0.0 <= position_win_rate(gu, 200) <= 1.0
    xr, dr, oro = position_win_rates(gu, 200)
    assert abs(xr + dr + oro - 1.0) < 1e-6


def test_thread_local_tt():
    ok, errors = [], []

    def worker():
        try:
            g = UltimateGame()
            ok.append(minimax_pro_move(g, depth=3) in g.legal_moves())
        except Exception as exc:  # re-raised below so failures cannot hide
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors            # thread failures propagate here
    assert len(ok) == 2 and all(ok)      # all(...) on an empty list is True


def test_ultimate_game_winning_moves():
    gu = UltimateGame()
    gu.macro = [X, X, EMPTY] + [EMPTY] * 6
    gu.micro[2] = [X, X, EMPTY] + [EMPTY] * 6
    gu.active_macro = 2
    gu.current = X
    assert minimax_pro_move(gu, depth=4) == (2, 2)
    assert mcts_rave_move(gu, 2000) == (2, 2)
    assert mcts_grave_move(gu, 2000) == (2, 2)


def test_cvc_termination():
    for gt, ax, ao in (('normal', 'Random', 'Random'),
                       ('ultimate', 'Random', 'Random'),
                       ('ultimate', 'Minimax', 'Random'),
                       ('normal', 'Solver', 'Random'),
                       ('ultimate', 'Minimax Pro', 'Random'),
                       ('ultimate', 'MCTS+RAVE', 'Random'),
                       ('ultimate', 'MCTS+GRAVE', 'Random')):
        game = NormalGame() if gt == 'normal' else UltimateGame()
        guard = 0
        while not game.is_over() and guard < 500:
            ai = ax if game.current == X else ao
            apply_move(game, get_ai_move(game, ai, 300))
            guard += 1
        assert game.is_over() and guard <= 500


def test_pvc_termination():
    for gt in ('normal', 'ultimate'):
        for ai in ('Random', 'Basic', 'Minimax', 'Minimax Pro', 'MCTS',
                   'MCTS+RAVE', 'MCTS+GRAVE', 'Solver'):
            game = NormalGame() if gt == 'normal' else UltimateGame()
            guard = 0
            while not game.is_over() and guard < 500:
                if game.current == X:
                    apply_move(game, random.choice(game.legal_moves()))
                else:
                    apply_move(game, get_ai_move(game, ai, 300))
                guard += 1
            assert game.is_over() and guard <= 500


def test_alphazero_falls_back_without_model(monkeypatch):
    monkeypatch.setattr(ai, 'load_az_model', lambda game: None)
    g = NormalGame()
    assert get_ai_move(g, 'AlphaZero', 30) in g.legal_moves()


def test_tree_reuse_across_opponent_reply(monkeypatch):
    set_engine_config({'engine': {'opening_book': False}})
    seen = []
    orig = ai.mcts_search

    def spy(game, iterations, c=None, prev_root=None, pool=None, **kwargs):
        seen.append(prev_root is not None)
        return orig(game, iterations, c, prev_root, pool, **kwargs)

    monkeypatch.setattr(ai, 'mcts_search', spy)
    g = NormalGame()
    m1 = get_ai_move(g, 'MCTS', 40)
    apply_move(g, m1)
    reply = ai._REUSE_LAST['MCTS'].children[0].move
    apply_move(g, reply)
    get_ai_move(g, 'MCTS', 40)
    assert seen == [False, True]


def test_dynamic_uct_uses_this_search_visits(monkeypatch):
    set_engine_config({'engine': {'opening_book': False}})
    calls = []

    def spy(c, rv, it):
        calls.append((rv, it))
        return c

    monkeypatch.setattr(ai, '_uct_scale', spy)
    g = NormalGame()
    m1 = get_ai_move(g, 'MCTS', 40)
    apply_move(g, m1)
    apply_move(g, ai._REUSE_LAST['MCTS'].children[0].move)
    get_ai_move(g, 'MCTS', 40)
    assert calls
    assert all(rv <= it for rv, it in calls)


def test_diff_move():
    a = NormalGame()
    b = NormalGame()
    b.make_move(4)
    assert _diff_move(a, b) == 4
    b2 = NormalGame()
    b2.make_move(0)
    b2.make_move(8)
    assert _diff_move(a, b2) is None  # two new cells
    ua = UltimateGame()
    ub = UltimateGame()
    ub.make_move(4, 4)
    assert _diff_move(ua, ub) == (4, 4)
    assert _diff_move(BitUltimateGame.from_game(ua), ub) == (4, 4)
    ub2 = UltimateGame()
    ub2.make_move(0, 0)   # routes the reply into macro 0
    ub2.make_move(0, 1)
    assert _diff_move(ua, ub2) is None  # two new cells
    assert _diff_move(ub, ua) is None   # a removal is not a forward move


def test_ultimate_terminal_score_prefers_faster_win():
    from ai import _minimax_ultimate
    g = UltimateGame()
    g.macro = [X, X, X] + [EMPTY] * 6
    assert _minimax_ultimate(g, 5, -math.inf, math.inf, True, X) == 100000 - 5
    assert _minimax_ultimate(g, 2, -math.inf, math.inf, True, X) == 100000 - 2
    assert _minimax_ultimate(g, 5, -math.inf, math.inf, True, O) == 5 - 100000


def test_parallel_mcts_terminal_returns_none():
    g = NormalGame()
    g.board = [X] * 9
    assert mcts_move_parallel(g, 20, 2) is None


def test_tt_max_is_dynamic():
    set_engine_config({'engine': {'tt_max': 10}})
    gu = UltimateGame()
    assert minimax_pro_move(gu, depth=3) in gu.legal_moves()
    assert len(ai._get_tt()) <= 10


def test_analyze_ultimate_bitboard_path(monkeypatch):
    """bitboard=True must actually build a BitUltimateGame for the Python
    search. With the default c_mcts=True the C analysis answers first and
    from_game is never called, so the switch must disable it here."""
    set_engine_config({'engine': {'bitboard': True, 'c_mcts': False}})
    built = []
    real_from = BitUltimateGame.from_game.__func__

    class _SpyBit(BitUltimateGame):
        @classmethod
        def from_game(cls, game):
            built.append(True)
            return real_from(BitUltimateGame, game)

    monkeypatch.setattr(ai, 'BitUltimateGame', _SpyBit)
    gu = UltimateGame()
    items, rates = analyze_position(gu, 200)
    assert built, 'bitboard=True never constructed a BitUltimateGame'
    assert abs(sum(rates) - 1.0) < 1e-6
    assert items and all(0.0 <= it['pct'] <= 1.0 for it in items)
