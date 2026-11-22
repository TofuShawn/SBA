# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""ai — the engine facade (D47 split).

Every computer opponent and the assistant surface start here:
get_ai_move dispatches to the engines, which live in their own
modules (config / cbridge / books / mcts / minimax / tablebase /
analysis). This module keeps the cross-engine dispatch, the
AlphaZero bridge and the cache reset; it also re-exports the whole
engine surface so the historical `from ai import ...` imports keep
working.
"""

import logging
import random
import threading

from core.game import (
    BitUltimateGame,
    EMPTY,
    NormalGame,
    O,
    UltimateGame,
    X,
    apply_move,
    apply_clone_result,
)
from config import (
    _CFG_DEFAULTS,
    MIN_VISIT_DIVISOR,
    MIN_VISIT_FLOOR,
    cfg_engine,
    cfg_session,
    save_engine_config,
    set_engine_config,
)
from engines.cbridge import (
    _C_PREV,
    _c_mcts_move,
    _c_minimax_pro,
    _c_rave_grave_move,
    _c_rollout_state,
    _c_search_opts,
    _ce,
    _get_c_rollout,
    _get_c_rollout_pattern,
    _py_rollout_state,
    _rollout_move,
    _serialize_ultimate,
    _threat_cells,
    _tt_key,
    move_text,
)
from analysis.books import _BOOK_ENGINES, ULTIMATE_BOOK, get_basic_move, opening_book_move
from engines.mcts import (
    _REUSE,
    _REUSE_LAST,
    GraveNode,
    MCTSNode,
    _diff_move,
    _D4,
    _POOL,
    _best_mcts_move,
    flat_mcts_move,
    mcts_grave_move,
    mcts_grave_search,
    mcts_move,
    mcts_rave_move,
    mcts_rave_search,
    mcts_search,
)
from engines.minimax import (
    _get_killers,
    _get_tt,
    _minimax_ultimate,
    minimax_move_normal,
    minimax_move_ultimate,
    minimax_pro_move,
)
from analysis.tablebase import (
    _TABLEBASE,
    _TABLEBASE_PATH,
    _TB_BYTE_VALUE,
    _TB_VALUE_BYTE,
    _board_from_key,
    _board_key,
    _build_tablebase_dfs,
    _read_tablebase_file,
    build_micro_tablebase,
    build_tablebase,
    solver_move,
    write_tablebase_file,
)
from analysis.analysis import (
    _c_ultt_analysis,
    _fork_count,
    _rates_from_root,
    _rates_from_stats,
    analyze_position,
    blocks_immediate_win,
    eval_ultimate,
    position_win_rate,
    position_win_rates,
    reason_for_move,
)

log = logging.getLogger("sba.ai")


def reset_engine_caches():
    """Drop module-level search caches (used by self-tests/bench)."""
    _REUSE.clear()
    _REUSE_LAST.clear()
    _get_tt().clear()
    _get_killers().clear()
    _C_PREV.clear()


# ---------------------------------------------------------------
# AlphaZero bridge (the training/network module owns the recipe)
# ---------------------------------------------------------------

_AZ_MODELS = {}


def _az_module():
    """Import the optional AlphaZero module, or None when torch/numpy are
    not installed. They are deliberately absent from requirements.txt (so
    Dependabot cannot drag a CUDA wheel into CI); a bare install must still
    be able to play and to run --self-test. Other import failures propagate.
    """
    try:
        from az import alphazero
    except ModuleNotFoundError as exc:
        root = (exc.name or "").split(".")[0]
        if root not in ("torch", "numpy"):
            raise
        log.info("AlphaZero: optional dependency %s is missing", root)
        return None
    return alphazero


def load_az_model(game):
    az = _az_module()
    if az is None:
        return None
    key = "normal" if isinstance(game, NormalGame) else "ultimate"
    if key not in _AZ_MODELS:
        _AZ_MODELS[key] = az.load_model(key)
    return _AZ_MODELS[key]


def alphazero_move(game, budget=800):
    az = _az_module()
    if az is None:
        return mcts_move(game, budget)
    model = load_az_model(game)
    if model is None:
        log.info(
            "AlphaZero: no trained model (models/az_%s.pt), falling back to MCTS",
            "normal" if isinstance(game, NormalGame) else "ultimate",
        )
        return mcts_move(game, budget)
    return az.alphazero_move(game, model, budget)


def az_policy_priors(game):
    """AZNet policy priors over the legal moves of the current position:
    {(m, i): p} renormalised across legal moves (mover's view). None when
    the board is Normal, the game is over, no trained model exists, or the
    optional torch/numpy deps are absent — callers treat that as "no heat"
    (D40: the assistant board overlay)."""
    if isinstance(game, NormalGame) or game.is_over():
        return None
    az = _az_module()
    if az is None:
        return None
    model = load_az_model(game)
    if model is None:
        return None
    vec = az.policy_vector(game, model)
    if vec is None:
        return None
    return {(k // 9, k % 9): float(v) for k, v in enumerate(vec) if v > 0.0}


def _mcts_move(game, ai_type, budget, on_progress=None, trace=None):
    """Run an MCTS-family search, reusing the previous tree when enabled."""
    if ai_type in ("MCTS", "MCTS+RAVE", "MCTS+GRAVE"):
        # D27: the whole tree search runs in C (order-of-magnitude faster).
        # D30: _c_mcts_move carries its own warm C arena in _C_PREV between
        # moves, so the C path does reuse a tree (the Python path has the
        # separate _REUSE cache). Normal 3x3 stays on the Python path.
        # D51: RAVE/GRAVE gain their own C trees under the same switch.
        if (
            cfg_engine("c_mcts", True)
            and isinstance(game, UltimateGame)
            and _ce.available(engine=True)
        ):
            if ai_type == "MCTS":
                mv = _c_mcts_move(game, budget, on_progress, trace)
            else:
                mv = _c_rave_grave_move(
                    game,
                    budget,
                    grave=(ai_type == "MCTS+GRAVE"),
                    on_progress=on_progress,
                    trace=trace,
                )
            if mv is not None:
                return mv
        if cfg_engine("multithreaded", False):
            return mcts_move_parallel(game, budget, cfg_engine("workers", 8), on_progress, trace)
    reuse = cfg_engine("tree_reuse", True)
    if cfg_engine("bitboard", True) and isinstance(game, UltimateGame):
        search_game = BitUltimateGame.from_game(game)
    else:
        search_game = game
    pool = _POOL if (ai_type == "MCTS" and cfg_engine("object_pool", True) and not reuse) else None
    key, prev = None, None
    if reuse:
        key = (ai_type, _tt_key(game))
        prev = _REUSE.pop(key, None)
        if prev is None:
            last = _REUSE_LAST.get(ai_type)
            if last is not None:
                reply = _diff_move(last.state, game)
                if reply is not None:
                    prev = next((c for c in last.children if c.move == reply), None)
    if ai_type == "MCTS":
        root = mcts_search(search_game, budget, prev_root=prev, pool=pool, on_progress=on_progress)
    elif ai_type == "MCTS+RAVE":
        root = mcts_rave_search(search_game, budget, prev_root=prev, on_progress=on_progress)
    else:
        root = mcts_grave_search(search_game, budget, prev_root=prev, on_progress=on_progress)
    move = _best_mcts_move(root, game, budget)
    if move is None:
        # the root never expanded (zero budget): hand back None so the
        # caller can fall back, instead of crashing on apply_move(None)
        if pool is not None:
            pool.release(root)
        return None
    # Thinking trace: top candidates by visit share (read BEFORE any pool
    # release / tree-reuse bookkeeping — nodes may be recycled afterwards).
    if trace is not None:
        root_visits = root.visits or 1
        min_visits = max(MIN_VISIT_FLOOR, budget // MIN_VISIT_DIVISOR)
        cands = [c for c in root.children if c.visits >= min_visits]
        cands.sort(key=lambda c: -c.visits)
        trace["top"] = [
            (move_text(c.move), c.visits / root_visits, c.wins / c.visits) for c in cands[:3]
        ]
    if reuse:
        g = game.clone()
        apply_move(g, move)
        child = next((c for c in root.children if c.move == move), None)
        if child is not None:
            _REUSE[(ai_type, _tt_key(g))] = child
            _REUSE_LAST[ai_type] = child
            if len(_REUSE) > cfg_engine("reuse_cache", 32):
                _REUSE.pop(next(iter(_REUSE)))
    if pool is not None:
        pool.release(root)
    return move


def mcts_move_parallel(game, iterations, workers=8, on_progress=None, trace=None):
    """Run several independent MCTS searches and merge their root stats.

    Experimental (C4): pure-Python tree search holds the GIL, so threads do
    NOT give real parallelism on CPU-bound work — this mostly overlaps the
    tree-building overhead and is kept for A/B comparison. For true
    parallelism use processes instead (see alphazero.py's spawn pool).

    Progress is coarse here: one report per finished thread (D24) — the
    threads run concurrently, so the counter moves in jumps, not smoothly.
    """
    chunk = max(1, iterations // workers)
    results = []
    if cfg_engine("bitboard", True) and isinstance(game, UltimateGame):
        base = BitUltimateGame.from_game(game)
    else:
        base = game

    def work():
        results.append(mcts_search(base.clone(), chunk))

    threads = [threading.Thread(target=work) for _ in range(workers)]
    for t in threads:
        t.start()
    for i, t in enumerate(threads):
        t.join()
        if on_progress is not None:
            on_progress(
                {
                    "it": (i + 1) * chunk,
                    "iterations": iterations,
                    "best_text": "",
                    "share": 0.0,
                    "wr": 0.0,
                }
            )
    stats = {}
    for root in results:
        for c in root.children:
            s = stats.setdefault(c.move, [0, 0.0])
            s[0] += c.visits
            s[1] += c.wins
    if trace is not None:
        total_v = sum(s[0] for s in stats.values()) or 1
        top = sorted(stats.items(), key=lambda kv: -kv[1][0])[:3]
        trace["top"] = [(move_text(m), s[0] / total_v, s[1] / max(1, s[0])) for m, s in top]
    if not stats:
        moves = game.legal_moves()
        return random.choice(moves) if moves else None
    best_move = max(stats, key=lambda m: stats[m][1] / max(1, stats[m][0]))
    best_ratio = stats[best_move][1] / max(1, stats[best_move][0])
    for m, (v, w) in stats.items():
        if w / max(1, v) == best_ratio:
            g = game.clone()
            apply_move(g, m)
            if g.result() == game.current:
                best_move = m
                break
    return best_move


def get_ai_move(game, ai_type, mcts_budget=800, minimax_depth=3, on_progress=None, trace=None):
    """Pick a move for `ai_type`.

    Optional hooks for showing the engine's thinking (D24):
    - `on_progress(info)` — called ~10x during an MCTS search with
      {'it', 'iterations', 'best_text', 'share', 'wr'}.
    - `trace` — a dict the caller owns; filled after the search with
      {'engine', 'budget', 'depth', 'picked_text', 'top': [(text, share, wr)],
      'note'} so UIs can display what the engine considered.
    """
    move = _get_ai_move_impl(game, ai_type, mcts_budget, minimax_depth, on_progress, trace)
    if move is None:
        # last-resort fallback: never hand the UI a None move on a live board
        legal = game.legal_moves()
        move = random.choice(legal) if legal else None
    if trace is not None:
        trace.setdefault("engine", ai_type)
        trace["picked_text"] = move_text(move)
    return move


def _get_ai_move_impl(
    game, ai_type, mcts_budget=800, minimax_depth=3, on_progress=None, trace=None
):
    # a zero/negative budget is a legal setting (the UI slider, --iters 0)
    # but it means "no search at all": the trees would have no children.
    # Clamp here so every engine sees a budget it can actually run.
    mcts_budget = max(1, int(mcts_budget))
    minimax_depth = max(1, int(minimax_depth))
    if trace is not None:
        trace["budget"] = mcts_budget
        trace["depth"] = minimax_depth
    if cfg_engine("opening_book", True) and ai_type in _BOOK_ENGINES:
        book_move = opening_book_move(game)
        if book_move is not None:
            if trace is not None:
                trace["note"] = "opening book"
            return book_move
    if ai_type == "Random":
        return random.choice(game.legal_moves())
    if ai_type == "Basic":
        return get_basic_move(game)
    if ai_type == "Minimax":
        if isinstance(game, NormalGame):
            if cfg_engine("c_minimax", True) and _ce.available(engine=True):
                r = _ce.minimax_normal(game.board, game.current)
                if r and r[0] is not None:
                    return r[0]
            return minimax_move_normal(game)[0]
        if cfg_engine("c_minimax", True) and _ce.available(engine=True):
            mv = _ce.minimax_ultimate(game, minimax_depth, opts=_c_search_opts())
            if mv is not None:
                return mv
        return minimax_move_ultimate(game, depth=minimax_depth)
    if ai_type == "Minimax Pro":
        # D51: the C backend (negamax + TT + killers + LMR + ID) under the
        # same c_minimax switch; the Python Pro remains the fallback. The
        # C export is ultimate-only — Normal delegates to the full solve
        # (sba_minimax_normal) exactly like the Python Pro does.
        if (
            cfg_engine("c_minimax", True)
            and isinstance(game, UltimateGame)
            and _ce.available(engine=True)
        ):
            mv = _c_minimax_pro(game, minimax_depth)
            if mv is not None:
                return mv
        return minimax_pro_move(game, depth=minimax_depth)
    if ai_type in ("MCTS", "MCTS+RAVE", "MCTS+GRAVE"):
        return _mcts_move(game, ai_type, mcts_budget, on_progress, trace)
    if ai_type == "Flat MCTS":
        # the C flat kernel serializes Ultimate boards only
        if (
            not isinstance(game, NormalGame)
            and cfg_engine("c_mcts", True)
            and _ce.available(engine=True)
        ):
            moves = game.legal_moves()
            if moves:
                mv = _ce.flat_mcts(game, max(1, mcts_budget // len(moves)), opts=_c_search_opts())
                if mv is not None:
                    return mv
        return flat_mcts_move(game, mcts_budget)
    if ai_type == "Solver":
        if isinstance(game, NormalGame):
            # D39: Normal Solver is a pure table lookup — the persisted
            # tablebase (data/tt_normal.bin, rebuilt by --build-tables)
            # answers every query; the C on-the-fly solve stays with the
            # Minimax engine (both are perfect play)
            return solver_move(game, mcts_budget)
        # Ultimate: Solver is a plain MCTS fallback -> the C tree wins
        if cfg_engine("c_mcts", True) and _ce.available(engine=True):
            mv = _c_mcts_move(game, mcts_budget, on_progress, trace)
            if mv is not None:
                return mv
        return solver_move(game, mcts_budget)
    if ai_type == "AlphaZero":
        return alphazero_move(game, mcts_budget)
    raise ValueError(f"Unknown AI type: {ai_type}")
