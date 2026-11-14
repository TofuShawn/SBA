# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Textual dashboard smoke tests (D44): bench via the real Pilot harness,
train via synthetic events (no torch run in the suite)."""

import asyncio
import os

os.environ.setdefault('TERM', 'xterm-256color')

import pytest  # noqa: E402

pytest.importorskip('textual')

from tui_tools import BenchTUI, TrainTUI  # noqa: E402


def test_bench_tui_runs_and_reports():
    """Pilot: Start on Random-vs-Basic, 2 quick games; the score row and
    the log fill, and auto-exit leaves the app."""
    async def _run():
        app = BenchTUI(games=2, iterations=4, game_type='normal',
                       ai_a='Random', ai_b='Basic', depth=2, auto_exit=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.click('#start')
            deadline = 30.0
            while app._busy and deadline > 0:
                await asyncio.sleep(0.1)
                deadline -= 0.1
            assert deadline > 0, 'bench worker did not finish in time'
            log_text = '\n'.join(
                str(line) for line in app.query_one('#log').lines)
            assert 'done' in log_text
            assert app.query_one('#bar').progress == 2
    asyncio.run(_run())


def test_train_tui_event_updates():
    """Feed synthetic train events straight into the handlers: the bar,
    both sparklines, stats and the log react without any torch."""
    async def _run():
        app = TrainTUI(games=3, sims=4, channels=32, blocks=2)
        async with app.run_test(size=(120, 40)):
            app._on_event({'type': 'game', 'game': 1, 'games': 3,
                           'stage': 2, 'lr': 1e-3, 'loss': 1.7,
                           'positions': 38, 'temp': 0.9, 'eval': None,
                           'elapsed': 1.0})
            app._on_event({'type': 'ckpt', 'game': 1, 'path': 'm.pt'})
            app._on_event({'type': 'game', 'game': 2, 'games': 3,
                           'stage': 2, 'lr': 9e-4, 'loss': 1.1,
                           'positions': 80, 'temp': 0.8,
                           'eval': (2, 1, 1), 'elapsed': 2.5})
            assert app.query_one('#bar').progress == 2
            assert app.query_one('#loss-spark').data[-1] == 1.1
            assert app.query_one('#wr-spark').data[-1] == 62.5
            assert 'positions 80' in app._stats_text
            log_text = '\n'.join(
                str(line.text) for line in app.query_one('#log').lines)
            assert 'checkpoint -> m.pt' in log_text
            assert 'eval game 2: win=2 draw=1 loss=1 (62%)' in log_text
    asyncio.run(_run())
