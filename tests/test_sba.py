"""Session helpers, CLI plumbing and config persistence."""

import logging
import tomllib

import ai
from SBA import new_session, side_types
from ai import cfg_session, set_engine_config, save_engine_config


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


def _config_file(tmp_path, text):
    path = tmp_path / 'config.toml'
    path.write_text(text, encoding='utf-8')
    return path


_HAND_MANAGED = (
    '[engine]\n'
    'uct_c = 2.0\n'
    'symmetry = false\n'
    'workers = 8\n'
    '[engine.opening_book_ultimate]\n'
    '"4,4" = [[4,0],[4,2]]\n'
    '[session]\n'
    'mcts_budget = 1234\n'
)


def test_save_only_rewrites_explicit_keys(tmp_path, monkeypatch):
    """A Settings save must not silently reset floats, the opening-book
    table or [session] values it was never given — and the in-memory CONFIG
    must agree with the file immediately (no split until restart)."""
    path = _config_file(tmp_path, _HAND_MANAGED)
    monkeypatch.setenv('SBA_CONFIG', str(path))
    save_engine_config({'engine': {'symmetry': True, 'workers': 4}})
    disk = tomllib.loads(path.read_text(encoding='utf-8'))
    assert disk['engine']['uct_c'] == 2.0
    assert disk['engine']['symmetry'] is True
    assert disk['engine']['workers'] == 4
    assert disk['session']['mcts_budget'] == 1234
    assert len(disk['engine']['opening_book_ultimate']) == 1
    assert ai.cfg_engine('uct_c') == 2.0
    assert ai.cfg_session('mcts_budget') == 1234
    assert len(ai.cfg_engine('opening_book_ultimate')) == 1
    assert not list(tmp_path.glob('*.tmp'))


def test_save_rewrites_floats(tmp_path, monkeypatch):
    path = _config_file(tmp_path, _HAND_MANAGED)
    monkeypatch.setenv('SBA_CONFIG', str(path))
    save_engine_config({'engine': {'uct_c': 2.5}})
    disk = tomllib.loads(path.read_text(encoding='utf-8'))
    assert disk['engine']['uct_c'] == 2.5
    assert ai.cfg_engine('uct_c') == 2.5


def test_restore_defaults_keeps_hand_managed_tables(tmp_path, monkeypatch):
    path = _config_file(tmp_path, _HAND_MANAGED)
    monkeypatch.setenv('SBA_CONFIG', str(path))
    save_engine_config()  # "Restore Defaults"
    disk = tomllib.loads(path.read_text(encoding='utf-8'))
    assert disk['engine']['opening_book_ultimate'] == {'4,4': [[4, 0], [4, 2]]}
    assert disk['engine']['symmetry'] is ai._CFG_DEFAULTS['engine']['symmetry']
    assert not list(tmp_path.glob('*.tmp'))


def test_save_refuses_to_overwrite_broken_toml(tmp_path, monkeypatch, caplog):
    broken = 'BROKEN = = =\n'
    path = _config_file(tmp_path, broken)
    monkeypatch.setenv('SBA_CONFIG', str(path))
    with caplog.at_level(logging.WARNING):
        save_engine_config({'engine': {'symmetry': False}})
    assert path.read_text(encoding='utf-8') == broken
    assert 'refusing to overwrite' in caplog.text
    assert not list(tmp_path.glob('*.tmp'))
