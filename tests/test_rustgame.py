# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rust state-layer contract tests (D48, re-anchored by the removal
project): RustUltimateGame IS the ultimate implementation, so these pins
are rust-only now — error texts, the shared apply_move routing, the FFI
fast paths (sba_ffi / planes / clone_apply), handle hygiene and the
make_game contract. (The old rust-vs-python parity fuzz retired with the
pure-Python UltimateGame; the frozen encode goldens carry that contract.)

torch/numpy are optional (requirements.txt deliberately omits them so
Dependabot cannot drag the CUDA wheel into CI) — without the guards the
top-level `import alphazero` makes --self-test fail at COLLECTION on a
bare install.
"""

import random

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

from az import alphazero
from az import rustbridge
from az.rustbridge import RustUltimateGame
from core.game import O, X

pytestmark = pytest.mark.skipif(not rustbridge.available(), reason="sba_game cdylib not built")

SEED = 20261115


def _dump(ru):
    """The full observable position: (cells, macro, player, active)."""
    cells, macro, player, active = ru.sba_ffi()
    return (tuple(cells), tuple(macro), player, active)


def _rust_positions(rng, n, max_plies=25):
    """n random legal-play rust positions (a fresh game each)."""
    for _ in range(n):
        g = RustUltimateGame()
        for _ in range(rng.randint(0, max_plies)):
            lm = g.legal_moves()
            if not lm or g.is_over():
                break
            g.make_move(*rng.choice(lm))
        yield g


def test_illegal_moves_raise_the_documented_texts():
    ru = RustUltimateGame()
    ru.make_move(4, 4)
    with pytest.raises(ValueError, match="cell is occupied"):
        ru.make_move(4, 4)
    # after (4,4) the reply is forced into macro 4
    with pytest.raises(ValueError, match="move must be in the routed macro"):
        ru.make_move(0, 0)


def test_shared_apply_move_helpers_route_the_wrapper():
    """game.apply_move duck-types on the MOVE shape (tuple = ultimate), so
    the shared helpers accept the wrapper too — the AZ eval path hit a
    TypeError here on the first CI run that compiled the crate."""
    from core.game import apply_clone_result, apply_move

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
    for g in _rust_positions(rng, 5):
        before = _dump(g)
        c = g.clone()
        lm = c.legal_moves()
        if lm:
            c.make_move(*lm[0])
            assert _dump(g) == before  # the source state is unchanged


def test_from_sba_state_roundtrip():
    """Bit-mask seeding must reproduce the exact position: seed, dump,
    compare — the test-seeding constructor that replaced from_game."""
    from conftest import rust_game_from_bits

    rng = random.Random(SEED + 6)
    for g in _rust_positions(rng, 10):
        st = rustbridge.SbaState()
        rustbridge._load().sba_game_state(g._h, rustbridge.ctypes.byref(st))
        rebuilt = rust_game_from_bits(
            px=list(st.px), po=list(st.po), mx=st.mx, mo=st.mo,
            current=st.current, active=st.active,
        )
        assert _dump(rebuilt) == _dump(g)


def test_make_game_returns_the_rust_layer():
    """make_game("ultimate") IS the rust state layer — no switch, no
    fallback (removal project: the python UltimateGame retired)."""
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


def test_serialize_ffi_matches_cengine_packing():
    """D50 fast path: sba_ffi() must equal cengine.serialize_ultimate's
    list contract, cell for cell, across random positions."""
    from engines import cengine

    rng = random.Random(SEED + 3)
    for ru in _rust_positions(rng, 30):
        cells, macro, player, active = cengine.serialize_ultimate(ru)
        ca, ma, ru_player, ru_active = ru.sba_ffi()
        assert list(ca) == cells
        assert list(ma) == macro
        assert ru_player == player and ru_active == active


def test_planes_tensor_matches_encode():
    """D50 fast path: one FFI call fills the whole 4x9x9 tensor — encode()
    on the rust layer delegates to it wholesale (the numpy walk retired)."""
    rng = random.Random(SEED + 4)
    for ru in _rust_positions(rng, 30):
        assert (alphazero.encode(ru).numpy() == ru.sba_planes_tensor()).all()


def test_clone_apply_matches_clone_plus_make_move():
    """D50 fast path: the PUCT expansion's one-call clone+apply must land
    in exactly the same state as clone() + make_move(), mirrors included."""
    rng = random.Random(SEED + 5)
    for ru in _rust_positions(rng, 30):
        lm = ru.legal_moves()
        if not lm:
            continue
        mv = rng.choice(lm)
        manual = ru.clone()
        manual.make_move(*mv)
        child = ru.clone_apply(*mv)
        assert _dump(child) == _dump(manual)
        assert child.current == manual.current
        assert child.active_macro == manual.active_macro
    ru = RustUltimateGame()
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
