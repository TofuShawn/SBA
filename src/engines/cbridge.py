# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""cbridge — the ctypes boundary to sba_engine.dll.
Owns the DLL handles, state serialization, the search-switch
snapshot and the C tree reuse cache. Engines call into this; the
deny-stub tests patch here."""

import threading

from core.game import NormalGame, apply_move
from config import MIN_VISIT_DIVISOR, MIN_VISIT_FLOOR, cfg_engine

from engines import cengine as _ce  # the ctypes layer under this package


def move_text(move):
    if isinstance(move, int):
        r, c = divmod(move, 3)
        return f"({r + 1},{c + 1})"
    m, i = move
    r, c = divmod(i, 3)
    return f"B{m + 1} ({r + 1},{c + 1})"


def _tt_key(game):
    if isinstance(game, NormalGame):
        return "".join(c or "." for c in game.board)
    return (
        "".join(c or "." for row in game.micro for c in row)
        + "|"
        + str(game.active_macro)
        + game.current
    )


def _get_c_rollout():
    """Compat wrapper — used only by tests/test_c_rollout.py; the engine
    itself asks cengine.available()."""
    return True if _ce.available() else None


def _get_c_rollout_pattern():
    """Compat wrapper — used only by tests/test_c_rollout.py; the pattern
    kernel is selected inside cengine.rollout_state()."""
    _ce._load()
    if _ce._lib is not None and _ce._has_engine:
        return getattr(_ce._lib, "sba_rollout_pattern", None)
    return None


def _serialize_ultimate(state):
    """Compat wrapper — used only by tests; delegates to cengine."""
    return _ce.serialize_ultimate(state)


def _c_rollout_state(state):
    """One native rollout to terminal; returns X / O / 'D'. Uses the pattern
    kernel when c_rollout_pattern is on; None when the DLL is missing."""
    return _ce.rollout_state(state, opts=_c_search_opts())


def _c_search_opts():
    """Snapshot of every switch the C backends consume. Built once per
    search rather than per simulation, so it stays off the hot path. Keeping
    this the single source of truth is what stops a switch from being
    "promised" in config.toml but ignored by the C tree."""
    return {
        "uct_c": float(cfg_engine("uct_c")),
        "dynamic_uct": bool(cfg_engine("dynamic_uct")),
        "rollout_heuristic": bool(cfg_engine("rollout_heuristic")),
        "micro_tablebase": bool(cfg_engine("micro_tablebase")),
        "symmetry": bool(cfg_engine("symmetry")),
        "use_pattern": bool(cfg_engine("c_rollout_pattern")),
    }


# Flat MCTS (research-only, unlisted): scores each legal move with independent
# playouts under the MCTS rollout policy, so the bench isolates search structure.

_C_PREV = {}  # ai_type -> previous C search tree buffer (tree reuse)

# a popped tree is owned exclusively until write-back; the lock guards only
# the cache bookkeeping, never the searches themselves (C arenas are reentrant).
_TREE_REUSE_LOCK = threading.Lock()


def _c_mcts_move(game, budget, on_progress=None, trace=None):
    """Full-tree MCTS in C, with root parallelization + tree reuse
    .

    - threads = min(workers, budget // 400): enough sims per thread for a
      meaningful tree; each thread has its own arena so they scale across
      cores (ctypes releases the GIL inside the C call).
    - thread 0 carries the reused subtree from the previous search; the
      other threads start fresh. Merging therefore never double-counts.
    - the warm arena is POPPED under _TREE_REUSE_LOCK: the search
      owns it exclusively, and a concurrent analysis misses the slot and
      builds its own arena instead of reading a tree mid-search.
    """
    c_opts = _c_search_opts()
    # the multithreaded switch gates root parallelization; workers caps it
    threads = int(cfg_engine("workers", 8) or 1) if cfg_engine("multithreaded", False) else 1
    threads = max(1, min(threads, max(1, budget // 400)))
    # the switch gates BOTH ends of the reuse cycle: no prev passed in, and
    # the fresh arena is dropped on return (the _CTree finalizer frees it) —
    # storing it would pin a dead arena between moves
    reuse = cfg_engine("tree_reuse", True)
    prev = None
    if reuse:
        with _TREE_REUSE_LOCK:
            prev = _C_PREV.pop("MCTS", None)

    results = [None]
    if threads <= 1:
        results[0] = _ce.mcts_search(game, budget, opts=c_opts, prev=prev)
    else:
        per = max(100, budget // threads)
        results = [None] * threads

        def _work(k):
            results[k] = _ce.mcts_search(game, per, opts=c_opts, prev=prev if k == 0 else None)

        workers = [threading.Thread(target=_work, args=(k,)) for k in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        results = [r for r in results if r]

    # merge root children across threads (thread 0 may carry the warm tree)
    merged = {}
    tree = None
    for _mv_idx, stats, buf in results:
        if buf is not None and tree is None:
            tree = buf
        for mi, v, w, mo, dr in stats:
            e = merged.get(mi)
            if e is None:
                merged[mi] = [v, w, mo, dr]
            else:
                e[0] += v
                e[1] += w
                e[2] += mo  # macro wins: summing only the first thread was silent
                e[3] += dr
    if tree is not None and reuse:
        with _TREE_REUSE_LOCK:
            _C_PREV["MCTS"] = tree
    elif reuse and prev is not None and not merged:
        # the search died without producing an arena: hand the warm tree
        # back (sba_mcts_search never writes into prev — it is const) so the
        # next search keeps its head start, unless a newer tree arrived
        with _TREE_REUSE_LOCK:
            if "MCTS" not in _C_PREV:
                _C_PREV["MCTS"] = prev
    if not merged:
        return None
    stats = [(mi, v[0], v[1], v[2], v[3]) for mi, v in merged.items()]

    # best move: highest win-rate among sufficiently visited children,
    # preferring an immediate game win (parity with _best_mcts_move)
    min_visits = max(MIN_VISIT_FLOOR, budget // MIN_VISIT_DIVISOR)
    cands = [s for s in stats if s[1] >= min_visits] or stats
    best = max(cands, key=lambda s: s[2] / s[1] if s[1] else 0.0)
    move_idx = best[0]
    for s in cands:
        if s[2] / max(1, s[1]) != best[2] / max(1, best[1]):
            continue
        g = game.clone()
        apply_move(g, (s[0] // 9, s[0] % 9))
        if g.result() == game.current:
            move_idx = s[0]
            break

    if trace is not None:
        root_visits = sum(s[1] for s in stats) or 1
        cands3 = sorted(stats, key=lambda s: -s[1])[:3]
        trace["top"] = [
            (move_text((s[0] // 9, s[0] % 9)), s[1] / root_visits, s[2] / s[1] if s[1] else 0.0)
            for s in cands3
        ]
    if on_progress is not None:
        picked = next((s for s in stats if s[0] == move_idx), None)
        on_progress(
            {
                "it": budget,
                "iterations": budget,
                "best_text": move_text((move_idx // 9, move_idx % 9)),
                "share": 1.0,
                "wr": (picked[2] / picked[1]) if picked and picked[1] else 0.0,
            }
        )
    return (move_idx // 9, move_idx % 9)


def _c_rave_grave_move(game, budget, grave, on_progress=None, trace=None):
    """Full RAVE/GRAVE tree in C. Single-threaded — the Python trees
    are too, and root parallelization over RAVE statistics is a separate
    project. Tree reuse rides the same _C_PREV slots ('RAVE'/'GRAVE')."""
    if not _ce.available(engine=True):
        return None
    key = "GRAVE" if grave else "RAVE"
    # tree_reuse gates both ends here too (same contract as _c_mcts_move);
    # the pop hands this thread exclusive ownership of the slot
    reuse = cfg_engine("tree_reuse", True)
    prev = None
    if reuse:
        with _TREE_REUSE_LOCK:
            prev = _C_PREV.pop(key, None)
    fn = _ce.grave_search if grave else _ce.rave_search
    res = fn(game, budget, k=int(cfg_engine("rave_k", 150)), opts=_c_search_opts(), prev=prev)
    if res is None or res[0] is None:
        if reuse and prev is not None:
            # failed search: restore the untouched warm tree for the next one
            with _TREE_REUSE_LOCK:
                if key not in _C_PREV:
                    _C_PREV[key] = prev
        return None
    best_move, stats, tree = res
    if tree is not None and reuse:
        with _TREE_REUSE_LOCK:
            _C_PREV[key] = tree
    if trace is not None:
        root_visits = sum(s[1] for s in stats) or 1
        top = sorted(stats, key=lambda s: -s[1])[:3]
        trace["top"] = [
            (move_text((s[0] // 9, s[0] % 9)), s[1] / root_visits, s[2] / s[1] if s[1] else 0.0)
            for s in top
        ]
    if on_progress is not None:
        packed = best_move[0] * 9 + best_move[1]
        picked = next((s for s in stats if s[0] == packed), None)
        on_progress(
            {
                "it": budget,
                "iterations": budget,
                "best_text": move_text(best_move),
                "share": 1.0,
                "wr": (picked[2] / picked[1]) if picked and picked[1] else 0.0,
            }
        )
    return best_move


def _c_minimax_pro(game, depth):
    """Minimax Pro via the C backend; None → Python fallback.

    The four pruning switches are C-side capabilities too (sba_minimax_pro
    reads o->use_*), but cengine.minimax_pro defaults them all to True.
    Passing them explicitly is what keeps the MINIMAX PRUNING group on the
    settings page from being a no-op whenever c_minimax is on.
    """
    if not _ce.available(engine=True):
        return None
    return _ce.minimax_pro(
        game,
        depth,
        time_limit_ms=int(cfg_engine("pro_time_ms", 8000)),
        use_lmr=bool(cfg_engine("use_lmr", True)),
        use_killers=bool(cfg_engine("use_killers", True)),
        use_aspiration=bool(cfg_engine("use_aspiration", True)),
        use_eval_ordering=bool(cfg_engine("use_eval_ordering", True)),
    )
