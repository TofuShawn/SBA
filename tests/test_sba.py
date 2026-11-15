"""Session helpers, CLI plumbing and config persistence."""

import logging
import os
import tomllib

import pytest

import ai
import config
from SBA import new_session, side_types
from ai import cfg_session, set_engine_config, save_engine_config


def test_session_defaults():
    s = new_session()
    assert s["mcts"] == cfg_session("mcts_budget", 800)
    assert s["minimax_depth"] == cfg_session("minimax_depth", 4)
    assert s["moves"] == [] and s["history"] == [] and s["step"] == 0
    assert s["cvc_paused"] is False


def test_side_types_pvc_computer_first():
    s = new_session()
    s["mode"] = "pvc"
    s["first_player"] = "computer"
    assert side_types(s) == (s["ai_o"], "Human")


def test_config_override_and_reset():
    set_engine_config({"engine": {"rollout_heuristic": False}})
    from ai import cfg_engine

    assert cfg_engine("rollout_heuristic") is False
    set_engine_config({})
    assert cfg_engine("rollout_heuristic") is True


def _config_file(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


_HAND_MANAGED = (
    "[engine]\n"
    "uct_c = 2.0\n"
    "symmetry = false\n"
    "workers = 8\n"
    "[engine.opening_book_ultimate]\n"
    '"4,4" = [[4,0],[4,2]]\n'
    "[session]\n"
    "mcts_budget = 1234\n"
)


def test_save_only_rewrites_explicit_keys(tmp_path, monkeypatch):
    """A Settings save must not silently reset floats, the opening-book
    table or [session] values it was never given — and the in-memory CONFIG
    must agree with the file immediately (no split until restart)."""
    path = _config_file(tmp_path, _HAND_MANAGED)
    monkeypatch.setattr(config, "_CONFIG_PATH", str(path))
    save_engine_config({"engine": {"symmetry": True, "workers": 4}})
    disk = tomllib.loads(path.read_text(encoding="utf-8"))
    assert disk["engine"]["uct_c"] == 2.0
    assert disk["engine"]["symmetry"] is True
    assert disk["engine"]["workers"] == 4
    assert disk["session"]["mcts_budget"] == 1234
    assert len(disk["engine"]["opening_book_ultimate"]) == 1
    assert ai.cfg_engine("uct_c") == 2.0
    assert ai.cfg_session("mcts_budget") == 1234
    assert len(ai.cfg_engine("opening_book_ultimate")) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_save_rewrites_floats(tmp_path, monkeypatch):
    path = _config_file(tmp_path, _HAND_MANAGED)
    monkeypatch.setattr(config, "_CONFIG_PATH", str(path))
    save_engine_config({"engine": {"uct_c": 2.5}})
    disk = tomllib.loads(path.read_text(encoding="utf-8"))
    assert disk["engine"]["uct_c"] == 2.5
    assert ai.cfg_engine("uct_c") == 2.5


def test_restore_defaults_keeps_hand_managed_tables(tmp_path, monkeypatch):
    path = _config_file(tmp_path, _HAND_MANAGED)
    monkeypatch.setattr(config, "_CONFIG_PATH", str(path))
    save_engine_config()  # "Restore Defaults"
    disk = tomllib.loads(path.read_text(encoding="utf-8"))
    assert disk["engine"]["opening_book_ultimate"] == {"4,4": [[4, 0], [4, 2]]}
    assert disk["engine"]["symmetry"] is ai._CFG_DEFAULTS["engine"]["symmetry"]
    assert not list(tmp_path.glob("*.tmp"))


def test_save_refuses_to_overwrite_broken_toml(tmp_path, monkeypatch, caplog):
    broken = "BROKEN = = =\n"
    path = _config_file(tmp_path, broken)
    monkeypatch.setattr(config, "_CONFIG_PATH", str(path))
    with caplog.at_level(logging.WARNING):
        save_engine_config({"engine": {"symmetry": False}})
    assert path.read_text(encoding="utf-8") == broken
    assert "refusing to overwrite" in caplog.text
    assert not list(tmp_path.glob("*.tmp"))


class _OsProxy:
    """Forward everything to the real os but let a test swap replace().

    Patching config.os (rather than the global os module) keeps pytest's own
    file operations out of the way.
    """

    def __init__(self, replace):
        self._replace = replace

    def __getattr__(self, name):
        return getattr(os, name)

    def replace(self, src, dst):
        return self._replace(src, dst)


def test_save_is_an_atomic_replace(tmp_path, monkeypatch):
    """The write must go through a sibling temp file + os.replace, never a
    bare open(path, 'w') that can leave a half-written TOML behind."""
    path = _config_file(tmp_path, _HAND_MANAGED)
    monkeypatch.setattr(config, "_CONFIG_PATH", str(path))
    calls = []
    real_replace = os.replace

    def patched(src, dst):
        calls.append((src, dst))
        return real_replace(src, dst)

    monkeypatch.setattr(config, "os", _OsProxy(patched))
    save_engine_config({"engine": {"symmetry": True}})
    assert len(calls) == 1
    src, dst = calls[0]
    assert src != str(path) and os.path.dirname(src) == str(tmp_path)
    assert os.path.samefile(dst, str(path))
    assert tomllib.loads(path.read_text(encoding="utf-8"))["engine"]["symmetry"] is True
    assert not list(tmp_path.glob("*.tmp"))


def test_save_failure_leaves_the_original_file_intact(tmp_path, monkeypatch):
    """A crash before the replace must not truncate the user's config, and
    no stray temp file may survive."""
    path = _config_file(tmp_path, _HAND_MANAGED)
    monkeypatch.setattr(config, "_CONFIG_PATH", str(path))

    def boom(_src, _dst):
        raise OSError("simulated crash before replace")

    monkeypatch.setattr(config, "os", _OsProxy(boom))
    with pytest.raises(OSError, match="simulated crash"):
        save_engine_config({"engine": {"symmetry": True}})
    assert path.read_text(encoding="utf-8") == _HAND_MANAGED
    assert not list(tmp_path.glob("*.tmp"))


def test_bench_smoke_table(capsys):
    """--bench renders its results (rich table on a console, plain text
    in capture) and the score line keeps the documented semantics."""
    import SBA

    rc = SBA.bench(games=2, iterations=4, game_type="normal", ai_a="Random", ai_b="Basic", depth=2)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Benchmark" in out and "Random" in out and "Basic" in out
    assert "A score%" in out


def test_main_routes_tui(monkeypatch):
    """The unified entry forwards --tui to the terminal app without
    touching the desktop stack (fake tui module, no QApplication)."""
    import types
    import sys as _sys
    import pytest
    import SBA

    calls = []
    monkeypatch.setitem(
        _sys.modules, "tui", types.SimpleNamespace(main=lambda: calls.append("tui") or 0)
    )
    with pytest.raises(SystemExit) as ei:
        SBA.main(["--tui"])
    assert calls == ["tui"] and ei.value.code == 0


def test_main_forwards_az_subcommands(monkeypatch):
    """--train-az / --eval-az forward the remaining argv to
    alphazero.main with the right subcommand; the trigger flag is
    stripped, everything else passes through."""
    import types
    import sys as _sys
    import pytest
    import SBA

    seen = []
    monkeypatch.setitem(
        _sys.modules, "alphazero", types.SimpleNamespace(main=lambda argv: seen.append(argv) or 7)
    )
    with pytest.raises(SystemExit) as ei:
        SBA.main(["--train-az", "--games", "5"])
    assert seen == [["train", "--games", "5"]] and ei.value.code == 7
    seen.clear()
    with pytest.raises(SystemExit):
        SBA.main(["--eval-az", "--games", "30", "--sims", "200"])
    assert seen == [["eval", "--games", "30", "--sims", "200"]]


def test_main_bench_flags_read_argv(monkeypatch):
    """bench() reads its flags from the argv LIST passed to main() —
    the root shim forwards them; a wrong-engine error still exits 2."""
    import pytest
    import SBA

    with pytest.raises(SystemExit) as ei:
        SBA.main(["--bench", "--ai-a", "Nope"])
    assert ei.value.code == 2
