# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""C RAVE / GRAVE / Minimax Pro engines (D51): the forced-win picks, the
rave_k switch, tree reuse and the dispatch wiring. Skips when the DLL is
not built."""

import os
import time

import pytest

from engines import ai
from engines.ai import get_ai_move, set_engine_config
from core.game import X, UltimateGame, apply_move

DLL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "engines", "sba_engine.dll"
)
pytestmark = pytest.mark.skipif(not os.path.isfile(DLL), reason="native engine DLL not built")


def _forced_win():
    """Exactly one immediate game win, deliberately NOT the first legal
    move so a missing tie-break is visible."""
    g = UltimateGame()
    g.macro = [X, X, ""] + [""] * 6
    g.micro[2] = ["", "", "", "", X, "", X, "", ""]
    g.active_macro = 2
    g.current = X
    return g


def _out_of_book():
    g = UltimateGame()
    apply_move(g, (0, 0))
    apply_move(g, (0, 4))
    return g


def _ce():
    from engines import cengine as c

    return c


def test_c_rave_grave_pro_pick_the_forced_win():
    for name, r in (
        ("rave", _ce().rave_search(_forced_win(), 600)),
        ("grave", _ce().grave_search(_forced_win(), 600)),
        ("pro", (_ce().minimax_pro(_forced_win(), 6),)),
    ):
        assert r[0] == (2, 2), (name, r[0])


def test_c_rave_k_changes_the_search():
    """rave_k is the single RAVE knob — it must reach the C selection."""
    g = _out_of_book()
    lo = _ce().rave_search(g, 400, k=1)[1]
    hi = _ce().rave_search(g, 400, k=5000)[1]
    assert lo != hi


def test_c_rave_tree_reuse():
    g = _out_of_book()
    mv, stats, tree = _ce().rave_search(g, 300)
    assert tree is not None
    mv2, stats2, tree2 = _ce().rave_search(g, 300, prev=tree)
    assert mv2 in g.legal_moves()
    if tree2:
        tree2.close()
    tree.close()


def test_c_rave_symmetry_dedups_the_root():
    g = UltimateGame()  # empty board: 81 root children before dedup
    off = _ce().rave_search(g, 300, opts={"symmetry": 0})[1]
    on = _ce().rave_search(g, 300, opts={"symmetry": 1})[1]
    assert len(off) == 81
    assert 0 < len(on) < len(off) // 2


def test_dispatch_raves_use_the_c_backend(monkeypatch):
    """get_ai_move('MCTS+RAVE'/'MCTS+GRAVE') must route through the C
    exports when the DLL is up (spy on the cbridge wrapper)."""
    calls = []
    real = ai._c_rave_grave_move

    def spy(game, budget, grave, **kw):
        calls.append(grave)
        return real(game, budget, grave, **kw)

    monkeypatch.setattr(ai, "_c_rave_grave_move", spy)
    g = _out_of_book()
    assert get_ai_move(g, "MCTS+RAVE", 200) in g.legal_moves()
    assert get_ai_move(g, "MCTS+GRAVE", 200) in g.legal_moves()
    assert calls == [False, True]


def test_dispatch_raves_fall_back_to_python(monkeypatch):
    """c_mcts off -> the pure-Python RAVE/GRAVE trees answer (the DLL-less
    contract). The spy wraps the OWNING module's functions."""
    from engines import mcts

    calls = []
    orig_rave = mcts.mcts_rave_search
    orig_grave = mcts.mcts_grave_search

    def spy_rave(*a, **kw):
        calls.append("rave")
        return orig_rave(*a, **kw)

    def spy_grave(*a, **kw):
        calls.append("grave")
        return orig_grave(*a, **kw)

    monkeypatch.setattr(ai, "mcts_rave_search", spy_rave)
    monkeypatch.setattr(ai, "mcts_grave_search", spy_grave)
    set_engine_config({"engine": {"c_mcts": False}})
    try:
        g = _out_of_book()
        assert get_ai_move(g, "MCTS+RAVE", 100) in g.legal_moves()
        assert get_ai_move(g, "MCTS+GRAVE", 100) in g.legal_moves()
        assert calls == ["rave", "grave"]
    finally:
        set_engine_config()


def test_tree_reuse_switch_gates_the_c_mcts_path(monkeypatch):
    """tree_reuse=false must gate BOTH ends of the C reuse cycle: no prev
    handed to cengine and no arena parked in _C_PREV (the D53 fix — the
    switch used to be read by the Python tree only)."""
    from engines import cbridge

    seen = []
    real = _ce().mcts_search

    def spy(state, it, **kw):
        seen.append(kw.get("prev"))
        return real(state, it, **kw)

    monkeypatch.setattr(_ce(), "mcts_search", spy)
    set_engine_config({"engine": {"tree_reuse": False}})
    try:
        g = _out_of_book()
        assert cbridge._c_mcts_move(g, 200) in g.legal_moves()
        assert cbridge._c_mcts_move(g, 200) in g.legal_moves()
        assert seen == [None, None]
        assert cbridge._C_PREV.get("MCTS") is None
    finally:
        set_engine_config()
        cbridge._C_PREV.clear()
    # switch on: the second search receives the first one's arena
    g = _out_of_book()
    assert cbridge._c_mcts_move(g, 200) in g.legal_moves()
    assert cbridge._c_mcts_move(g, 200) in g.legal_moves()
    assert seen[2] is None and seen[3] is not None
    cbridge._C_PREV.clear()


def test_tree_reuse_switch_gates_the_c_rave_path(monkeypatch):
    """Same gate on the RAVE/GRAVE wrapper — it shares the _C_PREV slots
    and had the same unconditional-reuse bug."""
    from engines import cbridge

    seen = []
    real = _ce().rave_search

    def spy(state, it, **kw):
        seen.append(kw.get("prev"))
        return real(state, it, **kw)

    monkeypatch.setattr(_ce(), "rave_search", spy)
    set_engine_config({"engine": {"tree_reuse": False}})
    try:
        g = _out_of_book()
        assert cbridge._c_rave_grave_move(g, 200, grave=False) in g.legal_moves()
        assert cbridge._c_rave_grave_move(g, 200, grave=False) in g.legal_moves()
        assert seen == [None, None]
        assert cbridge._C_PREV.get("RAVE") is None
    finally:
        set_engine_config()
        cbridge._C_PREV.clear()
    g = _out_of_book()
    assert cbridge._c_rave_grave_move(g, 200, grave=False) in g.legal_moves()
    assert cbridge._c_rave_grave_move(g, 200, grave=False) in g.legal_moves()
    assert seen[2] is None and seen[3] is not None
    cbridge._C_PREV.clear()


def test_c_minimax_pro_dispatch_and_normal_delegation(monkeypatch):
    """Pro dispatch: C backend under c_minimax for Ultimate; NormalGame
    delegates to the Python Pro (which routes to the full 3x3 solve)."""
    calls = []
    real = ai._c_minimax_pro

    def spy(game, depth, **kw):
        calls.append(depth)
        return real(game, depth, **kw)

    monkeypatch.setattr(ai, "_c_minimax_pro", spy)
    g = _out_of_book()
    assert get_ai_move(g, "Minimax Pro", 3, 4) in g.legal_moves()
    assert calls == [4]
    from core.game import NormalGame

    calls.clear()
    ng = NormalGame()
    assert get_ai_move(ng, "Minimax Pro", 3, 4) in ng.legal_moves()
    assert calls == []  # Normal never reaches the C Pro


def test_c_minimax_pro_prefers_the_faster_win():
    """(2,2) completes the macro line immediately — ID negamax must pick
    the forced win over any slower plan."""
    g = UltimateGame()
    g.macro = [X, X, ""] + [""] * 6
    g.micro[2] = [X, X, ""] + [""] * 6
    g.active_macro = 2
    g.current = X
    assert _ce().minimax_pro(g, 5) == (2, 2)


def test_c_minimax_pro_time_limit_stays_sane():
    """A tight cap must return promptly (the fallback move path)."""
    g = _out_of_book()
    t0 = time.perf_counter()
    mv = _ce().minimax_pro(g, 10, time_limit_ms=150)
    elapsed = time.perf_counter() - t0
    assert mv in g.legal_moves()
    assert elapsed < 3.0


def test_c_minimax_pro_pruning_switches_reach_the_backend(monkeypatch):
    """The MINIMAX PRUNING group must reach the C Pro backend, not only the
    Python fallback. cengine defaults all four switches to True, so a
    missing kwarg silently turns the whole settings group into a no-op
    whenever c_minimax is on (the same contract bug D53 fixed for
    tree_reuse).
    """
    seen = []
    real = _ce().minimax_pro

    def spy(game, depth, **kw):
        seen.append(kw)
        return real(game, depth, **kw)

    monkeypatch.setattr(_ce(), "minimax_pro", spy)
    g = _out_of_book()
    try:
        for flag in (False, True):
            set_engine_config(
                {
                    "engine": {
                        "opening_book": False,
                        "c_minimax": True,
                        "use_lmr": flag,
                        "use_killers": flag,
                        "use_aspiration": flag,
                        "use_eval_ordering": flag,
                    }
                }
            )
            assert get_ai_move(g.clone(), "Minimax Pro", 3, 3) in g.legal_moves()
    finally:
        set_engine_config()
    assert len(seen) == 2, seen
    for kw, flag in zip(seen, (False, True), strict=False):
        for key in ("use_lmr", "use_killers", "use_aspiration", "use_eval_ordering"):
            assert kw.get(key) is flag, (key, kw)


def test_c_minimax_pro_switches_change_the_search():
    """Behavioural half: the switches are not merely passed through. Pin it
    on the raw ctypes export so a spy-only test cannot hide a wrapper that
    drops the kwargs.

    A fixed position where root eval ordering changes the tie-break: the
    score must stay identical (exact negamax — pruning is not allowed to
    change the result), but the chosen move does. LMR / killers /
    aspiration are pure speed-ups and deliberately leave the result
    alone; their contract is the kwargs assertion above.
    """
    import ctypes

    ce = _ce()
    lib = ce._load()
    if lib is None or not ce._has_engine:
        pytest.skip("full engine DLL not built")
    g = UltimateGame()
    for mv in (
        (7, 1),
        (1, 3),
        (3, 0),
        (0, 1),
        (1, 7),
        (7, 7),
        (7, 0),
        (0, 4),
        (4, 1),
        (1, 5),
        (5, 6),
        (6, 0),
        (0, 8),
        (8, 1),
        (1, 1),
        (1, 8),
    ):
        apply_move(g, mv)
    ca, ma, player, active = ce._serialized_arrays(g)

    def best_for(**switches):
        v = dict(use_lmr=0, use_killers=0, use_aspiration=0, use_eval_ordering=0)
        v.update(switches)
        opts = ce.ProOpts(
            v["use_lmr"], v["use_killers"], v["use_aspiration"], v["use_eval_ordering"], 1, 300000
        )
        best, score = ctypes.c_int32(-1), ctypes.c_double(0.0)
        lib.sba_minimax_pro(
            ca,
            ma,
            player,
            active,
            4,
            8000,
            ctypes.byref(opts),
            ctypes.byref(best),
            ctypes.byref(score),
        )
        return best.value, score.value

    off = best_for()
    on = best_for(use_lmr=1, use_killers=1, use_aspiration=1, use_eval_ordering=1)
    assert off[1] == on[1]  # pruning must not change the score
    assert off[0] != on[0]  # ...but root ordering breaks the tie
    assert best_for(use_eval_ordering=1)[0] == on[0]


def test_tb_init_export_runs_at_load_and_is_idempotent():
    """D63: the minimax micro tablebase is built once at DLL load
    (cengine._load calls sba_minimax_tb_init right after binding) instead of
    lazily inside eval_ultt. The export must exist, report the table ready,
    and stay safe under repeated calls (the C guard is an atomic CAS)."""
    ce = _ce()
    assert ce.available(engine=True)
    fn = getattr(ce._lib, "sba_minimax_tb_init", None)
    assert fn is not None
    assert fn() == 1
    assert fn() == 1
    # a tablebase-driven eval still answers (minimax_ultimate is tb-aware)
    assert ce.minimax_ultimate(_forced_win(), 4) == (2, 2)
