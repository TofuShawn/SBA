"""Session helpers and CLI plumbing."""

from SBA import new_session, side_types
from ai import cfg_session, set_engine_config


def test_session_defaults():
    s = new_session()
    assert s['mcts'] == cfg_session('mcts_budget', 800)
    assert s['minimax_depth'] == cfg_session('minimax_depth', 4)
    assert s['moves'] == [] and s['history'] == [] and s['step'] == 0
    assert s['cvc_paused'] is False


def test_side_types_pvc_computer_first():
    s = new_session()
    s['mode'] = 'pvc'
    s['first_player'] = 'computer'
    assert side_types(s) == (s['ai_o'], 'Human')


def test_config_override_and_reset():
    set_engine_config({'engine': {'rollout_heuristic': False}})
    from ai import cfg_engine
    assert cfg_engine('rollout_heuristic') is False
    set_engine_config({})
    assert cfg_engine('rollout_heuristic') is True
