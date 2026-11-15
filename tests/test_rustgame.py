# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rust state-layer parity (D48): RustUltimateGame must be indistinguishable
from UltimateGame — legal moves, results, routing mirrors, micro/macro
grids and error texts — across random games; the wrapper also carries AZ
self-play end-to-end (stage-1 C tree + encode). Skips when the cdylib is
not built."""

import random

import pytest

import alphazero
import rustbridge
from game import X, UltimateGame
from rustbridge import RustUltimateGame

pytestmark = pytest.mark.skipif(
    not rustbridge.available(), reason='sba_game cdylib not built')

SEED = 20261115


def _fresh_pair():
    py = UltimateGame()
    return py, RustUltimateGame.from_game(py)


def _assert_parity(py, ru):
    """One position, both implementations: everything a search or the UI
    can observe must agree."""
    assert set(ru.legal_moves()) == set(py.legal_moves())
    assert ru.legal_moves() == py.legal_moves()          # order too
    assert ru.current == py.current
    assert ru.active_macro == py.active_macro
    assert ru.result() == py.result()
    assert ru.winner() == py.winner()
    assert ru.is_over() == py.is_over()
    assert ru.is_full() == py.is_full()
    assert ru.micro == py.micro
    assert ru.macro == py.macro
    for m in range(9):
        assert ru.macro_open(m) == py.macro_open(m)
        assert ru.micro_winner(m) == py.micro_winner(m)


def _play(rng, py=None, ru=None):
    """A random playout applied to BOTH implementations in lockstep."""
    if py is None:
        py, ru = _fresh_pair()
    for _ in range(90):
        _assert_parity(py, ru)
        lm = py.legal_moves()
        if not lm:
            break
        mv = rng.choice(lm)
        py.make_move(*mv)
        ru.make_move(*mv)
    _assert_parity(py, ru)
    return py, ru


def test_parity_over_random_games():
    """150 seeded random games, full parity assertion after every ply."""
    rng = random.Random(SEED)
    for _ in range(150):
        py, ru = _play(rng)
        assert ru.is_over() == py.is_over()


def test_from_game_adopts_positions():
    rng = random.Random(SEED + 1)
    for _ in range(20):
        py, _ = _play(rng)
        ru = RustUltimateGame.from_game(py)
        _assert_parity(py, ru)


def test_illegal_moves_raise_the_same_texts():
    py, ru = _fresh_pair()
    py.make_move(4, 4)
    ru.make_move(4, 4)
    with pytest.raises(ValueError, match='cell is occupied'):
        ru.make_move(4, 4)
    with pytest.raises(ValueError, match='cell is occupied'):
        py.make_move(4, 4)
    # after (4,4) the reply is forced into macro 4
    with pytest.raises(ValueError, match='move must be in the routed macro'):
        ru.make_move(0, 0)
    with pytest.raises(ValueError, match='move must be in the routed macro'):
        py.make_move(0, 0)
    # a closed macro target with no open route
    py2, ru2 = _fresh_pair()
    py2.make_move(4, 4)
    ru2.make_move(4, 4)
    py2.make_move(4, 0)
    ru2.make_move(4, 0)          # sends the reply to macro 0
    py2.make_move(0, 4)
    ru2.make_move(0, 4)          # fills micro 4 of macro 0 -> closed? no;
    # macro 0 stays open, so use a full-micro route instead:
    # (0,4) routes to macro 4 which is open -> normal continuation


def test_game_over_and_clone_parity():
    py = UltimateGame()
    py.macro = [X] * 9                       # already won macro row
    ru = RustUltimateGame.from_game(py)
    assert ru.result() == py.result() == 'X'
    with pytest.raises(ValueError, match='game is over'):
        ru.make_move(0, 0)
    with pytest.raises(ValueError, match='game is over'):
        py.make_move(0, 0)

    base = UltimateGame()
    rb = RustUltimateGame.from_game(base)
    rc = rb.clone()
    rc.make_move(4, 4)
    assert rb.legal_moves() == base.legal_moves()   # the original untouched
    assert rb.current == base.current == X
    assert rb.active_macro == base.active_macro == 4 or \
        (base.active_macro is None and rb.active_macro is None)


def test_clone_is_independent_of_handle():
    rng = random.Random(SEED + 2)
    py, ru = _play(rng)
    c = ru.clone()
    lm = c.legal_moves()
    if lm:
        c.make_move(*lm[0])
        _assert_parity(py, ru)               # the source state is unchanged


def test_switch_off_falls_back_to_python_game(monkeypatch):
    from config import set_engine_config
    from game import UltimateGame
    set_engine_config({'engine': {'rust_game': False}})
    try:
        assert type(alphazero.make_game('ultimate')) is UltimateGame
    finally:
        set_engine_config()
    assert isinstance(alphazero.make_game('ultimate'), RustUltimateGame)


def test_self_play_smoke_runs_on_the_rust_layer():
    """Full self-play game through make_game -> wrapper: the C tree search,
    the encode fast path and the terminal scoring all accept the wrapper."""
    states, targets, zs = alphazero.self_play_game(
        lambda: alphazero.make_game('ultimate'), None, sims=8, temp=1.0,
        use_network=False)
    assert states and len(states) == len(targets) == len(zs)
    assert all(tuple(s.shape) == (4, 9, 9) for s in states)
    # every policy target is a normalized distribution over the 81 cells
    assert all(abs(float(t.sum()) - 1.0) < 1e-3 for t in targets)
