# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Offscreen BoardWidget paint tests (D40 policy heat layer)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

try:
    pytest.importorskip("PySide6")
    # the offscreen platform still needs a system EGL/GL stack; on a GL-less
    # runner the import fails INSIDE PySide6 (libEGL.so.1 missing) and the
    # module must skip instead of erroring collection
    from PySide6.QtWidgets import QApplication  # noqa: E402
except ImportError as exc:  # pragma: no cover - GL-less environments
    pytest.skip(f"PySide6/OpenGL stack unavailable: {exc}", allow_module_level=True)

from core.game import UltimateGame, apply_move  # noqa: E402

# NOTE: the cache must NOT share its name with the accessor — a global
# `_app = None` next to `def _app()` gets shadowed by the def, so
# `if _app is None` never fires and no QApplication is ever created (Qt
# then dies with "Must construct a QApplication before a QWidget", which
# pytest capture turns into a silent exit 127).
_QAPP = None


def _qapp():
    global _QAPP
    if _QAPP is None:
        _QAPP = QApplication.instance() or QApplication([])
    return _QAPP


def _midgame_board(widget):
    g = UltimateGame()
    apply_move(g, (0, 0))
    apply_move(g, (0, 4))  # routes the next move into macro 4
    widget.set_game(g)


def test_policy_heat_paints_and_clears():
    _qapp()
    from ui.qtui import BoardWidget

    w = BoardWidget()
    w.resize(450, 450)
    _midgame_board(w)
    plain = w.grab().toImage()

    heat = {(4, 4): 1.0, (4, 6): 0.5, (0, 1): 0.3}
    w.set_policy_heat(heat)
    assert w.grab().toImage() != plain  # the tint is actually drawn

    # a stone cell is skipped: (0,0) holds X, so this clears to plain
    w.set_policy_heat({(0, 0): 1.0})
    assert w.grab().toImage() == plain

    # the same position change always invalidates stale priors
    w.set_policy_heat(heat)
    w.set_game(UltimateGame())
    assert w._heat == {}
    _midgame_board(w)
    assert w.grab().toImage() == plain  # and the midgame looks un-heated


def test_pages_build_and_render_offscreen():
    """The three pages construct and paint offscreen — a guard for the
    _build_ui split (M12): every _build_* helper must leave the same widget
    tree and attributes the monolith did, and the paint path must run."""
    _qapp()
    from session.SBA import new_session
    from ui.qtui import GamePage, MenuPage, SettingsPage

    menu = MenuPage()
    menu.resize(900, 700)
    menu.grab()  # exercises the backdrop/contour paintEvent

    settings = SettingsPage()
    settings.resize(900, 700)
    settings.grab()
    # the numeric knobs the split must keep binding to their attributes
    assert settings.workers_spin.value() >= 1
    assert settings.assistant_sims_spin.value() >= 200
    assert settings.pro_time_spin.value() >= 100

    page = GamePage()
    page.resize(1200, 800)
    page.start_session(new_session())
    img = page.grab().toImage()
    assert not img.isNull()
    # the attributes other code reaches for all still exist after the split
    for attr in (
        "revert_btn",
        "pause_btn",
        "next_btn",
        "step_lbl",
        "clock_lbl",
        "board",
        "hist_chart_view",
        "hist_list",
        "analysis_list",
        "assistant_switch",
        "heat_switch",
        "cvc_card",
        "footer_lbl",
    ):
        assert getattr(page, attr) is not None, attr
