"""AlphaZero smoke tests (tiny nets / few sims)."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

import numpy as np
import random

import alphazero
from game import X, O, EMPTY, NormalGame, UltimateGame
from ai import get_ai_move


def test_az_smoke_train_ultimate():
    model = alphazero.train('ultimate', games=2, sims=6, quiet=True, save=False,
                            eval_games=1)
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


def test_mcts_root_dirichlet_noise(monkeypatch):
    model = alphazero.AZNet(3).to(alphazero.DEVICE)
    g = NormalGame()
    root0, _ = alphazero.mcts_search(g, model, 30, batch_size=8)
    assert root0.priors is not None and abs(sum(root0.priors.values()) - 1.0) < 1e-5
    eta = np.array([0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1])
    monkeypatch.setattr(np.random, 'dirichlet', lambda alpha: eta)
    root, _ = alphazero.mcts_search(g, model, 30, batch_size=8,
                                    dirichlet_alpha='auto', dirichlet_eps=0.5)
    for m in root0.priors:
        assert abs(root.priors[m] - (0.5 * root0.priors[m] + 0.5 * eta[m])) < 1e-6


def test_train_ckpt_every_saves(monkeypatch, tmp_path):
    target = tmp_path / 'az_ultimate.pt'
    monkeypatch.setattr(alphazero, 'model_path', lambda game_type: str(target))
    monkeypatch.setattr(
        alphazero, 'alphazero_move',
        lambda game, model, budget=800: random.choice(game.legal_moves()))
    alphazero.train('ultimate', games=3, sims=2, eval_every=99, eval_games=1,
                    quiet=True, save=True, ckpt_every=2,
                    channels=32, blocks=2)
    assert target.exists()


def test_train_parallel_workers(monkeypatch):
    monkeypatch.setattr(
        alphazero, 'alphazero_move',
        lambda game, model, budget=800: random.choice(game.legal_moves()))
    model = alphazero.train('ultimate', games=4, sims=2, eval_every=99,
                            eval_games=1, quiet=True, save=False,
                            channels=32, blocks=2, workers=2)
    assert model is not None
    g = UltimateGame()
    assert alphazero.select_move(g, model, 10) in g.legal_moves()


def test_policy_vector_masks_and_renormalises():
    """D40: the raw priors are masked to legal moves and renormalised —
    zero mass outside the legal set, one forward pass, no search."""
    model = alphazero.AZNet(9, 8, 1).to(alphazero.DEVICE)
    from game import apply_move
    g = UltimateGame()
    apply_move(g, (0, 0))
    apply_move(g, (0, 4))
    vec = alphazero.policy_vector(g, model)
    assert vec is not None
    assert abs(float(vec.sum()) - 1.0) < 1e-5
    legal = np.zeros(81, dtype=bool)
    for m, i in g.legal_moves():
        legal[m * 9 + i] = True
    assert (vec[~legal] == 0).all()
    assert (vec > 0).any()


def test_az_policy_priors_guards(monkeypatch):
    """No heat for Normal boards, decided boards, or without a model."""
    import ai
    assert ai.az_policy_priors(NormalGame()) is None
    gu = UltimateGame()
    gu.macro = [X, X, X] + [EMPTY] * 6
    gu.current = X
    assert gu.is_over() and ai.az_policy_priors(gu) is None
    # no trained model in the repo (cached as None) -> "no heat"
    monkeypatch.setitem(ai._AZ_MODELS, 'ultimate', None)
    assert ai.az_policy_priors(UltimateGame()) is None


def test_training_csv_curve(tmp_path):
    """D41: one CSV row per completed game; a fresh run overwrites, a
    resumed run appends without a second header; the eval gauge fills the
    win/draw/loss columns even on quiet runs."""
    import csv as _csv
    csv_p = tmp_path / 'curve.csv'
    model_p = tmp_path / 'm.pt'
    alphazero.train('ultimate', games=2, sims=4, eval_every=1, eval_games=1,
                    quiet=True, save=True, save_path=str(model_p),
                    csv_path=str(csv_p), channels=32, blocks=2)
    rows = list(_csv.reader(csv_p.read_text(encoding='utf-8').splitlines()))
    assert rows[0][:2] == ['stage', 'game']
    assert len(rows) == 3                       # header + 2 games
    assert [r[1] for r in rows[1:]] == ['1', '2']
    assert all(r[0] == '2' for r in rows[1:])   # stage 2 = neural
    assert all(r[7:10] != ['', '', ''] for r in rows[1:])  # eval ran (quiet!)
    assert rows[1][10] == ''                    # no checkpoint at game 1

    # resume appends to the same curve, no duplicated header
    alphazero.train('ultimate', games=1, sims=4, eval_every=99, eval_games=1,
                    quiet=True, save=False, resume=str(model_p),
                    csv_path=str(csv_p), channels=32, blocks=2)
    rows2 = list(_csv.reader(csv_p.read_text(encoding='utf-8').splitlines()))
    assert len(rows2) == 4                      # header + 2 + 1
    assert rows2[3][1] == '1'                   # game numbering restarts


def test_training_csv_disabled_without_save():
    """save=False and no explicit --csv -> no file (smoke runs stay clean)."""
    import os
    import ai
    default_csv = os.path.join(ai.ROOT, 'models', 'az_ultimate.csv')
    existed = os.path.exists(default_csv)
    alphazero.train('ultimate', games=1, sims=3, eval_every=99, eval_games=1,
                    quiet=True, save=False)
    assert os.path.exists(default_csv) == existed
