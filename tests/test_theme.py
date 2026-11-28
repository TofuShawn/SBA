"""The theme layer: token parity across palettes, live rebind, and
persistence. No QApplication needed — set_theme repaints only when an
app instance exists, and every paint site reads PAL at paint time."""

import pytest

import config
from ui import qtui


@pytest.fixture()
def _config_file(tmp_path, monkeypatch):
    """Repoint the config at a tmp file so set_theme's persistence never
    touches the real config.toml (same pattern as test_config_env)."""
    path = tmp_path / "config.toml"
    path.write_text(
        "[engine]\nc_mcts = true\n\n[session]\nmcts_budget = 800\n"
        'minimax_depth = 4\ntheme = "DARK"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "_CONFIG_PATH", str(path))
    yield path


@pytest.fixture(autouse=True)
def _restore_dark():
    """Direct module rebind (no persist) — a test must never leave the
    suite in LIGHT, and must never write the real config.toml on the way
    out."""
    yield
    qtui.ACTIVE_THEME = "DARK"
    qtui.PAL = qtui.PALETTE["DARK"]


def _read_session_theme(path):
    import tomllib

    with open(path, "rb") as fh:
        return tomllib.load(fh)["session"]["theme"]


def test_palette_tokens_complete():
    """Every paint site reads PAL[key] — all themes must carry the same
    token set (a missing token would crash the painter mid-stroke)."""
    keys = {k for theme in qtui.PALETTE.values() for k in theme}
    for name, theme in qtui.PALETTE.items():
        assert set(theme) == keys, f"{name} is missing: {keys - set(theme)}"


def test_set_theme_rebinds_palette_and_persists(_config_file):
    assert qtui.ACTIVE_THEME == "DARK"
    assert qtui.set_theme("LIGHT") is True
    assert qtui.ACTIVE_THEME == "LIGHT"
    assert qtui.PAL is qtui.PALETTE["LIGHT"]
    # persisted to the [session] section for the next startup
    assert _read_session_theme(_config_file) == "LIGHT"
    assert qtui.set_theme("DARK") is True
    assert qtui.PAL is qtui.PALETTE["DARK"]


def test_set_theme_rejects_unknown_names():
    assert qtui.set_theme("no-such-theme") is False
    assert qtui.ACTIVE_THEME == "DARK"


def test_set_theme_same_name_is_a_noop(_config_file):
    assert qtui.set_theme("DARK") is True  # already active: no write, no reapply
    assert _read_session_theme(_config_file) == "DARK"
