"""Game rules, geometry, and bitboard equivalence tests."""

import random

import pytest

from game import (
    X, O, EMPTY,
    NormalGame, UltimateGame, BitUltimateGame, apply_move,
    line_coords, win_segment, macro_center, win_badge_svg,
)


def test_normal_x_wins_top_row():
    g = NormalGame()
    for mv in (0, 3, 1, 4, 2):
        g.make_move(mv)
    assert g.result() == X


def test_normal_draw_detected():
    g = NormalGame()
    for mv in (0, 1, 2, 4, 3, 5, 8, 6, 7):
        g.make_move(mv)
    assert g.result() == 'D' and g.is_full()
    assert g.legal_moves() == []


def test_ultimate_routing():
    g = UltimateGame()
    g.make_move(0, 2)
    assert g.active_macro == 2 and g.current == O
    moves = g.legal_moves()
    assert len(moves) == 9 and all(m == 2 for m, _ in moves)


def test_ultimate_full_macro_frees_moves():
    g = UltimateGame()
    g.micro[0] = [X, O, X, X, O, O, O, X, X]
    g.macro[0] = 'D'
    g.active_macro = 0
    g.current = X
    assert any(m != 0 for m, _ in g.legal_moves())


def test_ultimate_won_macro_frees_moves():
    g = UltimateGame()
    g.macro[4] = O
    g.active_macro = 4
    g.current = X
    assert any(m != 4 for m, _ in g.legal_moves())


def test_ultimate_micro_win_claims_macro():
    g = UltimateGame()
    g.micro[3] = [X, X, EMPTY] + [EMPTY] * 6
    g.current = X
    g.make_move(3, 2)
    assert g.macro[3] == X


def test_ultimate_macro_win_detected():
    g = UltimateGame()
    g.macro = [X, X, EMPTY] + [EMPTY] * 6
    g.micro[2] = [X, X, EMPTY] + [EMPTY] * 6
    g.current = X
    g.make_move(2, 2)
    assert g.winner() == X


def test_ultimate_draw_detected():
    g = UltimateGame()
    g.macro = ['D'] * 9
    for m in range(9):
        g.micro[m] = [X, O, X, X, O, O, O, X, X]
    assert g.result() == 'D' and g.is_full()


def test_geometry_helpers():
    assert win_segment((0, 1, 2), line_coords) == ((4, 20), (96, 20))
    assert win_segment((0, 4, 8), line_coords) == ((7, 7), (93, 93))
    assert win_segment((0, 4, 8), macro_center) == ((7, 7), (93, 93))
    assert '<line' in win_badge_svg(X) and '<circle' in win_badge_svg(O)


def test_bitboard_equivalence():
    for _ in range(5):
        a = UltimateGame()
        b = BitUltimateGame.from_game(a)
        steps = 0
        while not a.is_over() and steps < 60:
            assert set(a.legal_moves()) == set(b.legal_moves())
            m = random.choice(a.legal_moves())
            apply_move(a, m)
            b.make_move(*m)
            steps += 1
        assert a.result() == b.result()


def test_bitboard_occupied_cell_raises():
    b = BitUltimateGame()
    b.make_move(0, 0)
    with pytest.raises(ValueError):
        b.make_move(0, 0)


def test_bitboard_drawn_macro_displays_D():
    g = UltimateGame()
    g.micro[0] = [X, O, X, X, O, O, O, X, X]
    g.macro[0] = 'D'
    b = BitUltimateGame.from_game(g)
    assert b.macro[0] == 'D'
