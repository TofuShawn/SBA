"""AI engine correctness, sanity, and termination tests.

The pure-Python search stack (mcts.py / minimax.py) retired with the
removal project — every search-type test drives the engines through
get_ai_move and the native C backend, on the RustUltimateGame blood the
UIs now run.
"""

import random

import pytest

from engines import ai
from engines.ai import (
    get_basic_move,
    solver_move,
    build_tablebase,
    _board_key,
    get_ai_move,
    opening_book_move,
    build_micro_tablebase,
    _fork_count,
    position_win_rate,
    position_win_rates,
    set_engine_config,
)
from az.rustbridge import RustUltimateGame
from core.game import (
    X,
    O,
    EMPTY,
    NormalGame,
    apply_move,
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


def test_minimax_never_loses_on_normal():
    """Normal Minimax routes through the C perfect solver (minimax_normal) —
    perfect but not necessarily fastest, so the pin is 'X never loses to
    random', not a specific move."""
    wins = draws = 0
    for _ in range(6):
        g = NormalGame()
        while not g.is_over():
            move = (
                get_ai_move(g, "Minimax", 300) if g.current == X else random.choice(g.legal_moves())
            )
            g.make_move(move)
        if g.result() == X:
            wins += 1
        elif g.result() == "D":
            draws += 1
    assert wins + draws == 6


def test_mcts_legal_move_on_empty_ultimate():
    g = RustUltimateGame()
    assert get_ai_move(g, "MCTS", 300) in g.legal_moves()


def test_flat_mcts_legal_move_on_ultimate():
    g = RustUltimateGame()
    assert get_ai_move(g, "Flat MCTS", 600) in g.legal_moves()


def test_mcts_family_rejects_normal_boards():
    """The C tree kernels serialize Ultimate layouts only — Normal boards
    have no MCTS backend now that the python stack retired (the opening
    book is off so the book cannot answer for the engine)."""
    set_engine_config({"engine": {"opening_book": False}})
    try:
        g = NormalGame()
        for engine in ("MCTS", "MCTS+RAVE", "MCTS+GRAVE", "Flat MCTS"):
            with pytest.raises(RuntimeError):
                get_ai_move(g, engine, 300)
    finally:
        set_engine_config()


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
    assert get_ai_move(g2, "Solver", 300) in g2.legal_moves()


def test_immediate_win_across_engines():
    """Normal Minimax Pro routes through the C perfect solver; the RAVE/
    GRAVE variants live inside the C tree (covered by the c-engine tests)."""
    g = NormalGame()
    g.board = [X, X, EMPTY] + [EMPTY, O, O] + [EMPTY] * 3
    g.current = X
    assert get_ai_move(g, "Minimax Pro", 300, 9) == 2


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
    gu = RustUltimateGame()
    assert opening_book_move(gu) in gu.legal_moves()


def test_ultimate_opening_book_responses():
    from engines.ai import ULTIMATE_BOOK

    for x_move, responses in ULTIMATE_BOOK.items():
        gu = RustUltimateGame()
        gu.make_move(*x_move)  # routes the next move into macro x_move[1]
        legal = set(gu.legal_moves())
        assert all(r in legal for r in responses)
        assert any(r[0] == x_move[1] for r in responses)
    # a real game follows the book for the first two plies, then falls back
    gu = RustUltimateGame()
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


def test_fork_count():
    assert _fork_count([EMPTY] * 9, X) == 0
    assert _fork_count([X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, X, X], X) == 1


def test_minimax_pro_legal_with_refinements():
    gu = RustUltimateGame()
    assert get_ai_move(gu, "Minimax Pro", 300, 3) in gu.legal_moves()


def test_win_rates():
    g = NormalGame()
    assert position_win_rate(g, 0) == 0.5
    assert position_win_rates(g, 0) == (0.0, 1.0, 0.0)
    g2 = NormalGame()
    g2.board = [X, X, X, EMPTY, O, EMPTY, EMPTY, EMPTY, O]
    g2.current = X
    assert position_win_rate(g2, 0) == 1.0
    assert position_win_rates(g2, 0) == (1.0, 0.0, 0.0)
    gu = RustUltimateGame()
    assert 0.0 <= position_win_rate(gu, 200) <= 1.0
    xr, dr, oro = position_win_rates(gu, 200)
    assert abs(xr + dr + oro - 1.0) < 1e-6


def test_ultimate_game_winning_moves():
    """Forced-win position (bit-seeded on the rust layer): the C engines
    (pro / rave / grave) must all find the immediate game win."""
    from conftest import rust_game_from_bits

    gu = rust_game_from_bits(px={2: 3}, mx=0b000000011, current=0, active=2)
    assert get_ai_move(gu, "Minimax Pro", 300, 4) == (2, 2)
    assert get_ai_move(gu, "MCTS+RAVE", 2000) == (2, 2)
    assert get_ai_move(gu, "MCTS+GRAVE", 2000) == (2, 2)


def test_cvc_termination():
    for gt, ax, ao in (
        ("normal", "Random", "Random"),
        ("ultimate", "Random", "Random"),
        ("ultimate", "Minimax", "Random"),
        ("normal", "Solver", "Random"),
        ("ultimate", "Minimax Pro", "Random"),
        ("ultimate", "MCTS+RAVE", "Random"),
        ("ultimate", "MCTS+GRAVE", "Random"),
    ):
        game = NormalGame() if gt == "normal" else RustUltimateGame()
        guard = 0
        while not game.is_over() and guard < 500:
            engine = ax if game.current == X else ao
            apply_move(game, get_ai_move(game, engine, 300))
            guard += 1
        assert game.is_over() and guard <= 500


def test_pvc_termination():
    # the engine lists follow the capability matrix: the MCTS family has no
    # Normal backend (the C kernels serialize Ultimate layouts only)
    for gt, engines in (
        ("normal", ("Random", "Basic", "Minimax", "Minimax Pro", "Solver")),
        (
            "ultimate",
            (
                "Random",
                "Basic",
                "Minimax",
                "Minimax Pro",
                "MCTS",
                "MCTS+RAVE",
                "MCTS+GRAVE",
                "Solver",
            ),
        ),
    ):
        for engine in engines:
            game = NormalGame() if gt == "normal" else RustUltimateGame()
            guard = 0
            while not game.is_over() and guard < 500:
                if game.current == X:
                    apply_move(game, random.choice(game.legal_moves()))
                else:
                    apply_move(game, get_ai_move(game, engine, 300))
                guard += 1
            assert game.is_over() and guard <= 500


def test_alphazero_requires_a_trained_model(monkeypatch):
    """No model → a clear RuntimeError (the silent MCTS fallback retired)."""
    monkeypatch.setattr(ai, "load_az_model", lambda game: None)
    g = NormalGame()
    with pytest.raises(RuntimeError):
        get_ai_move(g, "AlphaZero", 30)


def test_alphazero_requires_the_optional_torch_stack(monkeypatch):
    """torch is optional for the repo but mandatory for the AZ engine type;
    the assistant overlay still degrades to 'no heat' instead of raising."""
    monkeypatch.setattr(ai, "_az_module", lambda: None)
    g = RustUltimateGame()
    apply_move(g, (4, 4))
    with pytest.raises(RuntimeError):
        get_ai_move(g, "AlphaZero", 30)
    assert ai.load_az_model(g) is None
    assert ai.az_policy_priors(g) is None


def test_az_module_treats_missing_torch_as_optional(monkeypatch):
    """A real `import torch` failure carries name='torch' — that must be
    the graceful-degradation path, not an error. (D61: the optional module
    imports as az.alphazero, so the failure is simulated on the package
    entry — the same MNFE(name='torch') escaping the import statement that
    ai._az_module catches.)"""
    import sys
    import types

    class _AzMissingTorch(types.ModuleType):
        def __getattr__(self, name):
            err = ModuleNotFoundError("No module named 'torch'")
            err.name = "torch"
            raise err

    monkeypatch.setitem(sys.modules, "az", _AzMissingTorch("az"))
    assert ai._az_module() is None


def test_az_module_propagates_unrelated_import_failures(monkeypatch):
    """Only a missing torch/numpy is an optional-dep fallback; any other
    ModuleNotFoundError must surface rather than being swallowed."""
    import sys
    import types

    class _AzBroken(types.ModuleType):
        def __getattr__(self, name):
            raise ModuleNotFoundError("No module named 'sba_unrelated'")

    monkeypatch.setitem(sys.modules, "az", _AzBroken("az"))
    with pytest.raises(ModuleNotFoundError):
        ai._az_module()
