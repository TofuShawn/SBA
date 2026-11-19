# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rust state-layer parity (D48): RustUltimateGame must be indistinguishable
from UltimateGame — legal moves, results, routing mirrors, micro/macro
grids and error texts — across random games; the wrapper also carries AZ
self-play end-to-end (stage-1 C tree + encode). Skips when the cdylib is
not built."""

import random

import pytest

# torch/numpy are optional (requirements.txt deliberately omits them so
# Dependabot cannot drag the CUDA wheel into CI). Without these guards the
# top-level `import alphazero` makes --self-test fail at COLLECTION on a
# bare install, even though every other test file skips cleanly.
pytest.importorskip("numpy")
pytest.importorskip("torch")

import alphazero
import rustbridge
from game import O, X, UltimateGame
from rustbridge import RustUltimateGame

pytestmark = pytest.mark.skipif(not rustbridge.available(), reason="sba_game cdylib not built")

SEED = 20261115


def _fresh_pair():
    py = UltimateGame()
    return py, RustUltimateGame.from_game(py)


def _assert_parity(py, ru):
    """One position, both implementations: everything a search or the UI
    can observe must agree."""
    assert set(ru.legal_moves()) == set(py.legal_moves())
    assert ru.legal_moves() == py.legal_moves()  # order too
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
    with pytest.raises(ValueError, match="cell is occupied"):
        ru.make_move(4, 4)
    with pytest.raises(ValueError, match="cell is occupied"):
        py.make_move(4, 4)
    # after (4,4) the reply is forced into macro 4
    with pytest.raises(ValueError, match="move must be in the routed macro"):
        ru.make_move(0, 0)
    with pytest.raises(ValueError, match="move must be in the routed macro"):
        py.make_move(0, 0)
    # a closed macro target with no open route would raise the same text
    # on both sides — covered end-to-end by the parity fuzz instead


def test_game_over_and_clone_parity():
    py = UltimateGame()
    py.macro = [X] * 9  # already won macro row
    ru = RustUltimateGame.from_game(py)
    assert ru.result() == py.result() == "X"
    with pytest.raises(ValueError, match="game is over"):
        ru.make_move(0, 0)
    with pytest.raises(ValueError, match="game is over"):
        py.make_move(0, 0)

    base = UltimateGame()
    rb = RustUltimateGame.from_game(base)
    rc = rb.clone()
    rc.make_move(4, 4)
    assert rb.legal_moves() == base.legal_moves()  # the original untouched
    assert rb.current == base.current == X
    assert rb.active_macro == base.active_macro == 4 or (
        base.active_macro is None and rb.active_macro is None
    )


def test_shared_apply_move_helpers_route_the_wrapper():
    """game.apply_move duck-types on the MOVE shape (tuple = ultimate), so
    the shared helpers accept the wrapper too — the AZ eval path hit a
    TypeError here on the first CI run that compiled the crate."""
    from game import apply_clone_result, apply_move

    ru = RustUltimateGame()
    apply_move(ru, (4, 4))  # tuple: ultimate routing
    assert ru.current == O
    # the reply must stay inside macro 4 — the shared helper obeys routing
    with pytest.raises(ValueError, match="move must be in the routed macro"):
        apply_move(ru, (0, 0))
    # O replies (4,1) on a working copy: no terminal, and X routes on
    assert apply_clone_result(ru, (4, 1)) is None
    ru2 = ru.clone()
    apply_move(ru2, (4, 1))
    assert ru2.current == X and ru2.active_macro == 1
    assert apply_clone_result(ru2, (1, 0)) is None


def test_clone_is_independent_of_handle():
    rng = random.Random(SEED + 2)
    py, ru = _play(rng)
    c = ru.clone()
    lm = c.legal_moves()
    if lm:
        c.make_move(*lm[0])
        _assert_parity(py, ru)  # the source state is unchanged


def test_switch_off_falls_back_to_python_game(monkeypatch):
    from config import set_engine_config
    from game import UltimateGame

    set_engine_config({"engine": {"rust_game": False}})
    try:
        assert type(alphazero.make_game("ultimate")) is UltimateGame
    finally:
        set_engine_config()
    assert isinstance(alphazero.make_game("ultimate"), RustUltimateGame)


def test_self_play_smoke_runs_on_the_rust_layer():
    """Full self-play game through make_game -> wrapper: the C tree search,
    the encode fast path and the terminal scoring all accept the wrapper."""
    states, targets, zs = alphazero.self_play_game(
        lambda: alphazero.make_game("ultimate"), None, sims=8, temp=1.0, use_network=False
    )
    assert states and len(states) == len(targets) == len(zs)
    assert all(tuple(s.shape) == (4, 9, 9) for s in states)
    # every policy target is a normalized distribution over the 81 cells
    assert all(abs(float(t.sum()) - 1.0) < 1e-3 for t in targets)


def _random_positions(rng, n):
    for _ in range(n):
        py, ru = _play(rng)
        yield py, ru


def test_serialize_ffi_matches_cengine_packing():
    """D50 fast path: sba_ffi() must equal cengine.serialize_ultimate's
    list contract, cell for cell, across random positions."""
    from native import cengine

    rng = random.Random(SEED + 3)
    for py, ru in _random_positions(rng, 30):
        cells, macro, player, active = cengine.serialize_ultimate(py)
        ca, ma, ru_player, ru_active = ru.sba_ffi()
        assert list(ca) == cells
        assert list(ma) == macro
        assert ru_player == player and ru_active == active


def test_planes_tensor_matches_encode():
    """D50 fast path: one FFI call fills the whole 4x9x9 tensor — must be
    identical to alphazero.encode on the reference implementation."""
    rng = random.Random(SEED + 4)
    for py, ru in _random_positions(rng, 30):
        expected = alphazero.encode(py).numpy()
        assert (ru.sba_planes_tensor() == expected).all()


def test_clone_apply_matches_clone_plus_make_move():
    """D50 fast path: the PUCT expansion's one-call clone+apply must land
    in exactly the same state as clone() + make_move(), mirrors included."""
    rng = random.Random(SEED + 5)
    for py, ru in _random_positions(rng, 30):
        lm = py.legal_moves()
        if not lm:
            continue
        mv = rng.choice(lm)
        py_child = py.clone()
        py_child.make_move(*mv)
        ru_child = ru.clone_apply(*mv)
        _assert_parity(py_child, ru_child)
    ru = _fresh_pair()[1]
    ru.make_move(4, 4)
    with pytest.raises(ValueError, match="illegal move"):
        ru.clone_apply(4, 4)  # occupied → null handle


def test_freed_handles_are_caught_not_reused():
    """Freed and foreign handles must come back as ERR_CLOSED — a caller
    bug — and never masquerade as a game error like 'cell out of range'.
    Before the live-handle registry, a freed handle still read as a valid
    (empty) game, so apply() happily 'succeeded' on freed memory."""
    lib = rustbridge._load()
    handle = lib.sba_game_new()
    lib.sba_game_free(handle)
    assert lib.sba_game_apply(handle, 4, 4) == rustbridge._ERR_CLOSED
    # a double free is a no-op, and the handle stays dead afterwards
    lib.sba_game_free(handle)
    assert lib.sba_game_apply(handle, 4, 4) == rustbridge._ERR_CLOSED
    # freeing a handle that never existed must also be a no-op
    lib.sba_game_free(handle + 8)
    # and null comes back the same way
    assert lib.sba_game_apply(None, 4, 4) == rustbridge._ERR_CLOSED
