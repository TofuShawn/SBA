# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Offscreen BoardWidget paint tests (D40 policy heat layer)."""

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest  # noqa: E402

pytest.importorskip('PySide6')

from PySide6.QtWidgets import QApplication  # noqa: E402
from game import UltimateGame, apply_move  # noqa: E402

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
    apply_move(g, (0, 4))          # routes the next move into macro 4
    widget.set_game(g)


def test_policy_heat_paints_and_clears():
    _qapp()
    from qtui import BoardWidget
    w = BoardWidget()
    w.resize(450, 450)
    _midgame_board(w)
    plain = w.grab().toImage()

    heat = {(4, 4): 1.0, (4, 6): 0.5, (0, 1): 0.3}
    w.set_policy_heat(heat)
    assert w.grab().toImage() != plain   # the tint is actually drawn

    # a stone cell is skipped: (0,0) holds X, so this clears to plain
    w.set_policy_heat({(0, 0): 1.0})
    assert w.grab().toImage() == plain

    # the same position change always invalidates stale priors
    w.set_policy_heat(heat)
    w.set_game(UltimateGame())
    assert w._heat == {}
    _midgame_board(w)
    assert w.grab().toImage() == plain   # and the midgame looks un-heated
