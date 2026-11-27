# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the AI thinking hooks: on_progress callback and trace dict (D24)."""

import pytest

from engines.ai import get_ai_move
from core.game import UltimateGame, apply_move


def _out_of_book_position():
    """A position past the opening book so an MCTS search actually runs."""
    g = UltimateGame()
    apply_move(g, (0, 0))
    apply_move(g, (0, 4))
    return g


def test_mcts_progress_reports_steps():
    from engines.ai import set_engine_config

    # the C tree fires the same coarse progress events (D24 shape)
    set_engine_config({"engine": {"multithreaded": False, "opening_book": False}})
    try:
        g = _out_of_book_position()
        events = []
        move = get_ai_move(g, "MCTS", 200, 3, on_progress=events.append)
        assert move in g.legal_moves()
        assert len(events) >= 1  # the C tree reports at its own (coarser) cadence
        first = events[0]
        assert first["it"] <= first["iterations"]
        assert first["best_text"]
        assert 0.0 <= first["wr"] <= 1.0
    finally:
        set_engine_config()


def test_trace_has_top_candidates_and_pick():
    from engines.ai import set_engine_config

    set_engine_config({"engine": {"multithreaded": False}})
    try:
        g = _out_of_book_position()
        trace = {}
        move = get_ai_move(g, "MCTS", 200, 3, trace=trace)
        assert move in g.legal_moves()
        assert trace["engine"] == "MCTS"
        assert trace["budget"] == 200
        assert trace["picked_text"]  # engine-layer text for the chosen move
        top = trace["top"]
        assert 1 <= len(top) <= 3
        for text, share, wr in top:
            assert text
            assert 0.0 < share <= 1.0
            assert 0.0 <= wr <= 1.0
        # shares are descending (sorted by visit share)
        shares = [share for _, share, _ in top]
        assert shares == sorted(shares, reverse=True)
    finally:
        set_engine_config()


def test_trace_on_book_and_basic():
    g = UltimateGame()
    # empty board: MCTS hits the opening book — trace notes it, no top list
    trace = {}
    get_ai_move(g, "MCTS", 200, 3, trace=trace)
    assert trace.get("note") == "opening book"
    assert "top" not in trace

    # Basic is instant: trace carries engine metadata only
    g2 = _out_of_book_position()
    trace2 = {}
    get_ai_move(g2, "Basic", 0, 3, trace=trace2)
    assert trace2["engine"] == "Basic"
    assert trace2["picked_text"]


def test_backward_compatible_call():
    """Existing call sites (no kwargs) keep working."""
    g = _out_of_book_position()
    move = get_ai_move(g, "Basic")
    assert move in g.legal_moves()


if __name__ == "__main__":
    pytest.main([__file__])
