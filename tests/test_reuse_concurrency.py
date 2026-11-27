# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tree-reuse concurrency under the Qt dual workers (D63, external review
round 2). qtui.py runs AIWorker(QThread) -> get_ai_move and
AnalysisWorker(QThread) -> analyze_position at the same time, and both used
to read/write the shared reuse caches (_C_PREV, _REUSE, _REUSE_LAST) with no
lock. The D63 ownership rule: whoever pops a tree out of a shared slot owns
it EXCLUSIVELY until write-back; the other side misses the reuse and starts
fresh on its own arena. The searches themselves stay concurrent — only the
cache bookkeeping is serialized by cbridge._TREE_REUSE_LOCK.

The tests here pin the CONTRACT deterministically (event-orchestrated stubs,
never sleeps) instead of trying to win a race, plus one real-engine stress
run that asserts only no-exception + structurally valid results.
"""

import os
import threading

import pytest

from engines import ai, cbridge
from engines.ai import get_ai_move, set_engine_config
from engines.cengine import _CTree
from core.game import UltimateGame, apply_move
from analysis import analysis as analysis_mod
from analysis.analysis import analyze_position

DLL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "engines", "sba_engine.dll"
)


def _midgame():
    """An out-of-book Ultimate position (two plies played)."""
    g = UltimateGame()
    apply_move(g, (0, 0))
    apply_move(g, (0, 4))
    return g


# a move that is legal in the midgame above (macro 4 is active): index 40
_STUB_MOVE = 40


# ---------------------------------------------------------------------------
# Contract pins (hermetic: the C search is stubbed, the DLL is not needed)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _hermetic_c(monkeypatch):
    """Pretend the full engine DLL is up so the C wrappers run their real
    bookkeeping around a stubbed cengine.mcts_search."""
    from engines import cengine

    monkeypatch.setattr(cengine, "available", lambda engine=False: True)
    yield


def test_analysis_misses_reuse_while_ai_owns_the_c_tree(monkeypatch):
    """While an AI search owns the warm arena (popped under the lock), the
    concurrent analysis must NOT be handed that arena — it starts fresh."""
    from engines import cengine

    calls = []
    ai_entered = threading.Event()
    release_ai = threading.Event()

    def stub(state, it, **kw):
        calls.append(kw.get("prev"))
        if len(calls) == 1:  # the AI's call: hold ownership while analysis runs
            ai_entered.set()
            release_ai.wait(timeout=10)
        return _STUB_MOVE, [(_STUB_MOVE, 4.0, 4.0, 1, 0)], None

    monkeypatch.setattr(cengine, "mcts_search", stub)
    sentinel = object()
    cbridge._C_PREV["MCTS"] = sentinel
    try:
        g = _midgame()
        out = []
        t = threading.Thread(target=lambda: out.append(cbridge._c_mcts_move(g, 200)))
        t.start()
        assert ai_entered.wait(timeout=10)
        # analysis runs while the AI thread is inside its search
        items, rates = analysis_mod._c_ultt_analysis(g, 100)
        assert calls[0] is sentinel  # the AI got the warm tree
        assert calls[1] is None  # analysis missed it -> fresh arena
        assert items and rates == pytest.approx((1.0, 0.0, 0.0))
        release_ai.set()
        t.join(timeout=10)
        assert not t.is_alive()
        assert out[0] == (4, 4)  # the stub's move, legal in the midgame
        assert cbridge._C_PREV.get("MCTS") is None  # stub parked no arena
    finally:
        release_ai.set()
        cbridge._C_PREV.clear()


def test_analysis_returns_an_idle_tree_it_borrowed(monkeypatch):
    """No AI search in flight: analysis borrows the idle tree and gives it
    back, so the AI's warm start survives an analysis round-trip."""
    from engines import cengine

    seen = []

    def stub(state, it, **kw):
        seen.append(kw.get("prev"))
        return _STUB_MOVE, [(_STUB_MOVE, 2.0, 1.0, 1, 0)], None

    monkeypatch.setattr(cengine, "mcts_search", stub)
    idle = _CTree(0)  # handle 0: _free_tree no-ops, nothing leaks
    cbridge._C_PREV["MCTS"] = idle
    try:
        g = _midgame()
        items, rates = analysis_mod._c_ultt_analysis(g, 100)
        assert seen[0] is idle
        assert cbridge._C_PREV.get("MCTS") is idle  # restored, not dropped
        assert items[0]["move"] == (4, 4)
        assert rates == pytest.approx((0.5, 0.0, 0.5))
    finally:
        cbridge._C_PREV.clear()


def test_ai_write_back_wins_over_analysis_borrow(monkeypatch):
    """Analysis borrows the idle tree, the AI runs a fresh search meanwhile
    and parks a new arena: analysis's return must not clobber it."""
    from engines import cengine

    calls = []
    ana_entered = threading.Event()
    release_ana = threading.Event()

    def stub(state, it, **kw):
        calls.append(kw.get("prev"))
        if len(calls) == 1:  # analysis: hold the borrow while the AI runs
            ana_entered.set()
            release_ana.wait(timeout=10)
            return _STUB_MOVE, [(_STUB_MOVE, 2.0, 1.0, 1, 0)], None
        return _STUB_MOVE, [(_STUB_MOVE, 4.0, 4.0, 1, 0)], ai_tree

    ai_tree = _CTree(0)
    monkeypatch.setattr(cengine, "mcts_search", stub)
    idle = _CTree(0)
    cbridge._C_PREV["MCTS"] = idle
    try:
        g = _midgame()
        out = []
        t = threading.Thread(target=lambda: out.append(analysis_mod._c_ultt_analysis(g, 100)))
        t.start()
        assert ana_entered.wait(timeout=10)
        mv = cbridge._c_mcts_move(g, 200)  # AI: pop miss -> fresh -> write-back
        assert mv == (4, 4)
        release_ana.set()
        t.join(timeout=10)
        assert not t.is_alive()
        assert calls[0] is idle  # analysis borrowed the idle tree
        assert calls[1] is None  # the AI missed: analysis owned the slot
        assert cbridge._C_PREV.get("MCTS") is ai_tree  # AI's arena survives
    finally:
        release_ana.set()
        cbridge._C_PREV.clear()


def test_python_reuse_take_is_exclusive(monkeypatch):
    """Same ownership rule on the Python path: a taken _REUSE tree (pop or
    derive) is unreachable to the concurrent searcher — it starts fresh.

    The counterfactual is PLANTED, not hoped for: thread 2's position is
    cached under its own exact key AND derive-reachable through the warm
    tree (a grandchild at exactly that position), so an implementation
    without the ownership rule would hand thread 2 that subtree via either
    channel. The assertion is that thread 1's take purges both — a tree
    whose warm subtree happens to have no grandchildren (the old seeding)
    passed this test by luck, not by the rule."""
    from engines.mcts import MCTSNode, _REUSE, _REUSE_LAST, mcts_search, _tt_key

    set_engine_config({"engine": {"c_mcts": False, "opening_book": False}})
    g = _midgame()
    # seed a real, tiny search tree and take its first subtree as the warm tree
    root = mcts_search(g, 40)
    child = root.children[0]
    # thread 2's position: the reply position, one opponent reply deeper
    g2 = g.clone()
    apply_move(g2, child.move)
    reply = g2.legal_moves()[0]
    apply_move(g2, reply)
    # PLANT the would-be hits: a grandchild exactly at thread 2's position,
    # cached under its own exact key and derivable through _REUSE_LAST
    grand = MCTSNode(g2.clone(), reply, child, g2.current)
    child.children.append(grand)
    _REUSE[("MCTS", _tt_key(g))] = child
    _REUSE[("MCTS", _tt_key(g2))] = grand
    _REUSE_LAST["MCTS"] = child

    seen = []
    t1_entered = threading.Event()
    release_t1 = threading.Event()

    def stub(game, budget, prev_root=None, pool=None, on_progress=None, **kw):
        seen.append(prev_root)
        if len(seen) == 1:
            t1_entered.set()
            release_t1.wait(timeout=10)
        return MCTSNode(game, None, None, game.current)  # childless: no write-back

    monkeypatch.setattr(ai, "mcts_search", stub)
    try:
        res = []
        t1 = threading.Thread(target=lambda: res.append(ai._mcts_move(g, "MCTS", 100)))
        t1.start()
        assert t1_entered.wait(timeout=10)
        # thread 2 arrives at the reply position while thread 1 is searching
        t2 = threading.Thread(target=lambda: res.append(ai._mcts_move(g2, "MCTS", 100)))
        t2.start()
        t2.join(timeout=10)
        assert not t2.is_alive()
        release_t1.set()
        t1.join(timeout=10)
        assert not t1.is_alive()
        assert seen[0] is child  # thread 1 popped the warm subtree
        assert seen[1] is None  # thread 2 missed BOTH planted channels: the take purged them
    finally:
        release_t1.set()
        set_engine_config({})
        ai.reset_engine_caches()


def test_python_take_purges_trees_overlapping_the_taken_one(monkeypatch):
    """A take must leave the taken tree reachable from NO shared slot, in
    either direction: ancestor entries contain it, and paired write-backs
    can still cache a subtree that sits INSIDE it under its own key."""
    from engines.mcts import MCTSNode, _REUSE, _REUSE_LAST, _tt_key

    set_engine_config({"engine": {"c_mcts": False, "opening_book": False}})
    g = _midgame()
    parent = MCTSNode(g, None, None, g.current)
    g2 = g.clone()
    apply_move(g2, (4, 4))
    child_node = MCTSNode(g2, (4, 4), parent, g.current)
    g3 = g2.clone()
    apply_move(g3, (4, 0))
    grand = MCTSNode(g3, (4, 0), child_node, "O")
    _REUSE[("MCTS", _tt_key(g))] = child_node
    _REUSE[("MCTS", _tt_key(g3))] = grand  # the paired write-back's own key
    _REUSE_LAST["MCTS"] = child_node

    def stub(game, budget, prev_root=None, pool=None, on_progress=None, **kw):
        assert prev_root is child_node  # the take handed us the tree
        return MCTSNode(game, None, None, game.current)  # childless: no write-back

    monkeypatch.setattr(ai, "mcts_search", stub)
    try:
        assert ai._mcts_move(g, "MCTS", 100) is None
        assert ("MCTS", _tt_key(g)) not in _REUSE  # popped by the take
        assert ("MCTS", _tt_key(g3)) not in _REUSE  # descendant purged
        assert _REUSE_LAST.get("MCTS") is None  # paired slot purged
    finally:
        set_engine_config({})
        ai.reset_engine_caches()


# ---------------------------------------------------------------------------
# Real-engine stress: two concurrent workers, tree reuse on. No timing
# assertions — only no-exception + structurally valid results.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.isfile(DLL), reason="native engine DLL not built")
def test_dual_worker_stress_ai_move_and_analysis():
    set_engine_config({})
    positions = []
    for x, o in (((0, 0), (0, 4)), ((4, 4), (4, 0)), ((2, 2), (2, 4)), ((6, 6), (6, 4))):
        g = UltimateGame()
        apply_move(g, x)
        apply_move(g, o)
        positions.append(g)

    errors = []
    barrier = threading.Barrier(2)

    def ai_loop():
        try:
            barrier.wait(timeout=10)
            for i in range(6):
                g = positions[i % len(positions)].clone()
                mv = get_ai_move(g, "MCTS", 200)
                assert mv in g.legal_moves()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            errors.append(exc)

    def analysis_loop():
        try:
            barrier.wait(timeout=10)
            for i in range(6):
                g = positions[i % len(positions)].clone()
                items, rates = analyze_position(g, 200)
                x, d, o = rates
                assert all(0.0 <= v <= 1.0 for v in (x, d, o))
                assert abs(x + d + o - 1.0) < 1e-6
                for it in items:
                    assert it["move"] in g.legal_moves()
                    assert 0.0 <= it["pct"] <= 1.0
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=ai_loop, name="ai-worker"),
        threading.Thread(target=analysis_loop, name="analysis-worker"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
        assert not t.is_alive(), "worker thread hung"
    assert errors == []


# ---------------------------------------------------------------------------
# D63 also closed the books -> config back-mutation (M3)
# ---------------------------------------------------------------------------


def test_books_import_leaves_config_defaults_alone():
    """config.py is a leaf: what its defaults are must not depend on whether
    analysis.books was imported. The book DATA lives in books.py and reaches
    cfg_engine as the fallback default, never through _CFG_DEFAULTS."""
    import config
    import analysis.books as books  # noqa: F401  (the old mutator)

    assert "opening_book_ultimate" not in config._CFG_DEFAULTS["engine"]
    assert config._CFG_DEFAULTS["engine"]["opening_book"] is True
    # the built-in book still answers via the cfg fallback (one-ply position:
    # X opened (0,0), O to move, the book routes O to (0,4))
    g = UltimateGame()
    apply_move(g, (0, 0))
    assert books.opening_book_move(g) == (0, 4)
    # ... and a config.toml table (delivered through CONFIG) overrides it
    import config as cfg_mod

    cfg_mod.CONFIG["engine"]["opening_book_ultimate"] = {"0,0": [[0, 2]]}
    try:
        assert books.opening_book_move(g) == (0, 2)
    finally:
        set_engine_config({})
