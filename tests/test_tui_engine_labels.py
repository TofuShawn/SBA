# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Honest self-play engine labels (D53) on the train dashboard: the status
note names the engine that actually runs, the workers field's hint means
what the resolved engine does, and the RESOLVED engine is what train()
receives (displayed provenance == running provenance)."""

import asyncio
import os
from unittest import mock

os.environ.setdefault("TERM", "xterm-256color")

import pytest  # noqa: E402

pytest.importorskip("textual")

import rustselfplay  # noqa: E402
import tui_tools  # noqa: E402
from tui_tools import TrainTUI  # noqa: E402


def test_engine_note_rust():
    assert tui_tools.engine_note("rust", 8) == "self-play: rust (8 threads)"


def test_engine_note_python():
    assert tui_tools.engine_note("python", 4) == "self-play: python (mp, 4 workers)"


def test_train_tui_shows_resolved_engine_note():
    """auto resolves to rust here (find_exe patched to a hit): the page-1
    note states rust + threads, and the workers hint speaks exe-threads."""

    async def _run():
        with mock.patch.object(rustselfplay, "find_exe", lambda explicit=None: "/x/exe"):
            app = TrainTUI()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                note = str(app.query_one("#engine-note").content)
                assert note == "self-play: rust (1 threads)"
                hint = str(app.query_one("#workers-label").content)
                assert hint == "遊戲執行緒數（單一 exe 進程）/ threads of one exe process"

    asyncio.run(_run())


def test_train_tui_python_note_when_no_exe():
    """no exe on the machine → auto falls back to python; the workers hint
    flips to the mp-worker meaning (the semantic change is visible, not
    silent)."""

    async def _run():
        with mock.patch.object(rustselfplay, "find_exe", lambda explicit=None: None):
            app = TrainTUI(workers=4)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                note = str(app.query_one("#engine-note").content)
                assert note == "self-play: python (mp, 4 workers)"
                hint = str(app.query_one("#workers-label").content)
                assert hint == "自對弈 worker 進程數 / mp worker processes"

    asyncio.run(_run())


def test_train_tui_threads_resolved_engine_into_train():
    """the displayed engine is the engine that runs: the train call carries
    the engine resolved at mount, not a second auto."""

    seen = {}

    def fake_train(*args, **kwargs):
        seen.update(kwargs)

    async def _run():
        with mock.patch.object(rustselfplay, "find_exe", lambda explicit=None: "/x/exe"):
            import alphazero

            with mock.patch.object(alphazero, "train", fake_train):
                app = TrainTUI(workers=8)
                async with app.run_test(size=(120, 40)) as pilot:
                    await pilot.pause()
                    # ENTER/start via the action: the params page is taller
                    # than a 40-row screen, so a pilot.click("#start") would
                    # target an off-screen widget
                    app.action_start_run()
                    deadline = 10.0
                    while app._busy and deadline > 0:
                        await asyncio.sleep(0.05)
                        deadline -= 0.05
                    assert deadline > 0, "train worker did not finish in time"

    asyncio.run(_run())
    assert seen.get("selfplay_engine") == "rust"
