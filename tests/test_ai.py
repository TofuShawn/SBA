"""AI engine correctness, sanity, and termination tests."""

import random
import threading

from ai import (
    get_basic_move, minimax_move_normal, mcts_move, solver_move,
    minimax_pro_move, mcts_rave_move, mcts_grave_move, build_tablebase,
    _board_key, get_ai_move, opening_book_move, build_micro_tablebase,
    _rollout_move, reset_engine_caches, _REUSE, _fork_count, _sym_images,
    mcts_move_parallel, _D4, position_win_rate, position_win_rates,
    flat_mcts_move,
)
from game import X, O, EMPTY, NormalGame, UltimateGame, apply_move


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
    g2 = NormalGame()
    g2.board[4] = O
    g2.current = X
    assert opening_book_move(g2) in (0, 2, 6, 8)
    g3 = NormalGame()
    g3.board[0] = O
    g3.current = X
    assert opening_book_move(g3) == 4
    g4 = NormalGame()
    g4.board = [X, O] + [EMPTY] * 7
    g4.current = O
    assert opening_book_move(g4) is None
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
    g = NormalGame()
    m1 = get_ai_move(g, 'MCTS', 100)
    apply_move(g, m1)
    m2 = get_ai_move(g, 'MCTS', 100)
    assert m2 in g.legal_moves()
    assert len(_REUSE) > 0
    reset_engine_caches()
    assert len(_REUSE) == 0


def test_fork_count():
    assert _fork_count([EMPTY] * 9, X) == 0
    assert _fork_count([X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, X, X], X) == 1


def test_minimax_pro_legal_with_refinements():
    gu = UltimateGame()
    assert minimax_pro_move(gu, depth=3) in gu.legal_moves()


def test_symmetry():
    assert len({tuple(p) for p in _D4}) == 8
    assert set(_sym_images(0)) == {0, 2, 6, 8}


def test_multithreaded_mcts_legal():
    gu = UltimateGame()
    assert mcts_move_parallel(gu, 100, 2) in gu.legal_moves()


def test_multithreaded_enabled_via_config():
    from ai import set_engine_config
    set_engine_config({'engine': {'multithreaded': True, 'bitboard': True}})
    gu = UltimateGame()
    assert get_ai_move(gu, 'MCTS', 200) in gu.legal_moves()


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
    ok = []

    def worker():
        g = UltimateGame()
        ok.append(minimax_pro_move(g, depth=3) in g.legal_moves())

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(ok)


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
