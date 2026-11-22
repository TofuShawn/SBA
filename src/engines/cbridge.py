# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""cbridge — the ctypes boundary to sba_engine.dll (D27/D37).
Owns the DLL handles, state serialization, the search-switch
snapshot and the C tree reuse cache. Engines call into this; the
deny-stub tests patch here."""

import random
import threading

from core.game import BitUltimateGame, EMPTY, LINES, NormalGame, O, X, apply_move
from config import MIN_VISIT_DIVISOR, MIN_VISIT_FLOOR, cfg_engine

from engines import cengine as _ce  # the ctypes layer under this package (D61)


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
    kernel when c_rollout_pattern is on; falls back to Python when the DLL
    is missing."""
    r = _ce.rollout_state(state, opts=_c_search_opts())
    return r if r is not None else _py_rollout_state(state)


def _py_rollout_state(state):
    """Pure-Python heuristic rollout to terminal (the D25-era fallback)."""
    result = state.result()
    while result is None:
        moves = state.legal_moves()
        if not moves:
            break
        m = _rollout_move(state, moves)
        apply_move(state, m)
        result = state.result()
    return result


def _c_search_opts():
    """Snapshot of every switch the C backends consume (D37). Built once per
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


_LINE_BITS = [(1 << a) | (1 << b) | (1 << c) for a, b, c in LINES]


def _threat_cells(state, player):
    """Cells where `player` would win immediately by playing there."""
    if isinstance(state, NormalGame):
        cells = state.board
        res = []
        for a, b, c in LINES:
            vals = (cells[a], cells[b], cells[c])
            if vals.count(player) == 2 and EMPTY in vals:
                res.append((a, b, c)[vals.index(EMPTY)])
        return res
    if isinstance(state, BitUltimateGame):
        res = []
        pbits = state.x if player == X else state.o
        obits = state.o if player == X else state.x
        closed = state.mx | state.mo
        for m in range(9):
            if closed & (1 << m):
                continue
            base = m * 9
            mb = (pbits >> base) & 0x1FF
            ob = (obits >> base) & 0x1FF
            for line in _LINE_BITS:
                if (mb & line).bit_count() == 2 and ((mb | ob) & line) != line:
                    empty = line & ~(mb | ob)
                    res.append((m, empty.bit_length() - 1))
        return res
    res = []
    for m in range(9):
        if not state.macro_open(m):
            continue
        cells = state.micro[m]
        for a, b, c in LINES:
            vals = (cells[a], cells[b], cells[c])
            if vals.count(player) == 2 and EMPTY in vals:
                res.append((m, (a, b, c)[vals.index(EMPTY)]))
    return res


def _rollout_move(state, moves=None):
    """Win/block-biased rollout policy (falls back to random when disabled)."""
    if moves is None:
        moves = state.legal_moves()
    if not moves:
        return None
    if not cfg_engine("rollout_heuristic", True):
        return random.choice(moves)
    legal = set(moves)
    current = state.current
    opp = O if current == X else X
    wins = [w for w in _threat_cells(state, current) if w in legal]
    if wins:
        return random.choice(wins)
    threats = set(t for t in _threat_cells(state, opp) if t in legal)
    if not threats:
        return random.choice(moves)
    if len(threats) == 1:
        return next(iter(threats))
    return random.choice(moves)  # multiple threats: cannot block them all


# ---------------------------------------------------------------
# Flat MCTS (research-only): root-level Monte Carlo, no tree
# ---------------------------------------------------------------
# Scores every legal move independently with random playouts from the
# resulting position, using the same rollout policy as MCTS (so the bench
# isolates the search structure). Hidden from the menus; kept for learning,
# tests and --bench.

_C_PREV = {}  # ai_type -> previous C search tree buffer (tree reuse, D30)


def _c_mcts_move(game, budget, on_progress=None, trace=None):
    """Full-tree MCTS in C (D27), with root parallelization + tree reuse
    (D30).

    - threads = min(workers, budget // 400): enough sims per thread for a
      meaningful tree; each thread has its own arena so they scale across
      cores (ctypes releases the GIL inside the C call).
    - thread 0 carries the reused subtree from the previous search; the
      other threads start fresh. Merging therefore never double-counts.
    """
    c_opts = _c_search_opts()
    # the multithreaded switch gates root parallelization; workers caps it
    threads = int(cfg_engine("workers", 8) or 1) if cfg_engine("multithreaded", False) else 1
    threads = max(1, min(threads, max(1, budget // 400)))
    # the switch gates BOTH ends of the reuse cycle: no prev passed in, and
    # the fresh arena is dropped on return (the _CTree finalizer frees it) —
    # storing it would pin a dead arena between moves
    reuse = cfg_engine("tree_reuse", True)
    prev = _C_PREV.get("MCTS") if reuse else None

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
        _C_PREV["MCTS"] = tree
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
    """Full RAVE/GRAVE tree in C (D51). Single-threaded — the Python trees
    are too, and root parallelization over RAVE statistics is a separate
    project. Tree reuse rides the same _C_PREV slots ('RAVE'/'GRAVE')."""
    if not _ce.available(engine=True):
        return None
    key = "GRAVE" if grave else "RAVE"
    # tree_reuse gates both ends here too (same contract as _c_mcts_move)
    reuse = cfg_engine("tree_reuse", True)
    prev = _C_PREV.get(key) if reuse else None
    fn = _ce.grave_search if grave else _ce.rave_search
    res = fn(game, budget, k=int(cfg_engine("rave_k", 150)), opts=_c_search_opts(), prev=prev)
    if res is None or res[0] is None:
        return None
    best_move, stats, tree = res
    if tree is not None and reuse:
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
    """Minimax Pro via the C backend (D51); None → Python fallback.

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
