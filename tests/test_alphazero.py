"""AlphaZero smoke tests (tiny nets / few sims)."""

import random

import alphazero
from game import X, O, EMPTY, NormalGame, UltimateGame
from ai import get_ai_move


def test_az_smoke_train_ultimate():
    model = alphazero.train('ultimate', games=2, sims=6, quiet=True, save=False)
    assert model is not None
    g = UltimateGame()
    assert alphazero.select_move(g, model, 20) in g.legal_moves()


def test_az_dispatch_and_terminal_values():
    g = NormalGame()
    assert get_ai_move(g, 'AlphaZero', 20) in g.legal_moves()
    az9 = alphazero.AZNet(9)
    gu = UltimateGame()
    gu.macro = [X, X, EMPTY] + [EMPTY] * 6
    gu.micro[2] = [X, X, EMPTY] + [EMPTY] * 6
    gu.active_macro = 2
    gu.current = X
    assert alphazero.select_move(gu, az9, 60) == (2, 2)
    gw = NormalGame()
    gw.board = [X, X, X, O, O] + [EMPTY] * 4
    assert alphazero.terminal_value(gw) == -1.0
    gwu = UltimateGame()
    gwu.macro = [X, X, X] + [EMPTY] * 6
    assert alphazero.terminal_value(gwu) == -1.0


def test_train_eval_every_schedule(monkeypatch, capsys):
    fast = lambda game, model, budget=800, batch_size=16: random.choice(
        game.legal_moves())
    monkeypatch.setattr(alphazero, 'alphazero_move', fast)
    alphazero.train('ultimate', games=6, sims=3, eval_every=2, eval_games=1,
                    quiet=False, save=False)
    out = capsys.readouterr().out
    assert out.count('win=') == 3


def test_select_move_zero_counts_safe():
    model = alphazero.AZNet(3)
    g = NormalGame()
    assert alphazero.select_move(g, model, budget=0, temp=1.0) in g.legal_moves()


def test_mcts_search_batch_larger_than_budget():
    model = alphazero.AZNet(3).to(alphazero.DEVICE)
    g = NormalGame()
    root, counts = alphazero.mcts_search(g, model, 30, batch_size=64)
    assert root.visits == 30
    assert counts and all(v >= 1 for v in counts.values())
