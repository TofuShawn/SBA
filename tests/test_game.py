"""Game rules, geometry tests.

NormalGame keeps its pure-Python rules here; the Ultimate rules live in
the Rust state layer (az.rustbridge.RustUltimateGame — the pure-Python
implementation retired with the removal project), so the ultimate rules
pins run against the rust layer, seeded from bit masks via
conftest.rust_game_from_bits.
"""

import pytest

from conftest import rust_game_from_bits
from core.game import (
    X,
    O,
    NormalGame,
    apply_move,
    line_coords,
    win_segment,
    macro_center,
    win_badge_svg,
)

# a drawn 3x3 micro (full, no winning line): X at 0,2,3,7,8 / O at 1,4,5,6
_DRAWN_MICRO_X = 1 | 4 | 8 | 128 | 256
_DRAWN_MICRO_O = 2 | 16 | 32 | 64


def test_normal_x_wins_top_row():
    g = NormalGame()
    for mv in (0, 3, 1, 4, 2):
        g.make_move(mv)
    assert g.result() == X


def test_normal_draw_detected():
    g = NormalGame()
    for mv in (0, 1, 2, 4, 3, 5, 8, 6, 7):
        g.make_move(mv)
    assert g.result() == "D" and g.is_full()
    assert g.legal_moves() == []


def test_ultimate_routing():
    g = rust_game_from_bits()
    g.make_move(0, 2)
    assert g.active_macro == 2 and g.current == O
    moves = g.legal_moves()
    assert len(moves) == 9 and all(m == 2 for m, _ in moves)


def test_ultimate_full_macro_frees_moves():
    # macro 0 drawn full (closed) while routing points at it -> free choice
    g = rust_game_from_bits(
        px=[_DRAWN_MICRO_X] + [0] * 8,
        po=[_DRAWN_MICRO_O] + [0] * 8,
        current=0,
        active=0,
    )
    assert any(m != 0 for m, _ in g.legal_moves())


def test_ultimate_won_macro_frees_moves():
    # macro 4 taken by O while routing points at it -> free choice
    g = rust_game_from_bits(mo=1 << 4, current=0, active=4)
    assert any(m != 4 for m, _ in g.legal_moves())


def test_ultimate_micro_win_claims_macro():
    g = rust_game_from_bits(px={3: 3}, current=0, active=3)
    g.make_move(3, 2)
    assert g.macro[3] == X


def test_ultimate_macro_win_detected():
    g = rust_game_from_bits(px={2: 3}, mx=0b000000011, current=0, active=2)
    g.make_move(2, 2)
    assert g.winner() == X


def test_ultimate_draw_detected():
    g = rust_game_from_bits(
        px=[_DRAWN_MICRO_X] * 9,
        po=[_DRAWN_MICRO_O] * 9,
    )
    assert g.result() == "D" and g.is_full()


def test_geometry_helpers():
    assert win_segment((0, 1, 2), line_coords) == ((4, 20), (96, 20))
    assert win_segment((0, 4, 8), line_coords) == ((7, 7), (93, 93))
    assert win_segment((0, 4, 8), macro_center) == ((7, 7), (93, 93))
    assert "<line" in win_badge_svg(X) and "<circle" in win_badge_svg(O)


def test_make_move_rejects_terminal_and_misrouted_moves():
    """The rules layers are the ground truth: a won board has no legal
    moves, and a move must honour the macro routing."""
    n = NormalGame()
    for mv in (0, 3, 1, 4, 2):
        n.make_move(mv)
    assert n.result() == X and n.legal_moves() == []
    with pytest.raises(ValueError, match="game is over"):
        n.make_move(5)

    u = rust_game_from_bits()
    apply_move(u, (0, 2))  # routes the reply into macro 2
    with pytest.raises(ValueError, match="routed macro"):
        u.make_move(5, 5)

    draw = rust_game_from_bits(
        px=[_DRAWN_MICRO_X] * 9,
        po=[_DRAWN_MICRO_O] * 9,
    )
    assert draw.result() == "D"
    with pytest.raises(ValueError, match="game is over"):
        draw.make_move(0, 0)
