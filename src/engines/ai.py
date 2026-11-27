# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""ai — the engine facade (split).

Every computer opponent and the assistant surface start here:
get_ai_move dispatches to the engines. The pure-Python search stack
(mcts.py / minimax.py) retired with the removal project — the native C
engine IS the search backend and the sba_engine cdylib is a hard
prerequisite for every engine type beyond Random/Basic and the Normal
tablebase solver; without it those types raise instead of silently
degrading. This module keeps the cross-engine dispatch, the AlphaZero
bridge and the cache reset; it also re-exports the config surface so
the historical `from ai import ...` imports keep working.
"""

import logging
import random

from config import (
    _CFG_DEFAULTS,
    cfg_engine,
    cfg_session,
    save_engine_config,
    set_engine_config,
)
from core.game import NormalGame
from engines.cbridge import (
    _C_PREV,
    _TREE_REUSE_LOCK,
    _c_mcts_move,
    _c_minimax_pro,
    _c_rave_grave_move,
    _c_search_opts,
    _ce,
    move_text,
)
from analysis.books import _BOOK_ENGINES, ULTIMATE_BOOK, get_basic_move, opening_book_move
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
    """Drop module-level search caches (used by self-tests/bench).

    Only the C tree-reuse slot remains: the python _REUSE machinery
    retired with the pure-Python search stack, and the C minimax keeps
    its transposition/killer state inside the native call.
    """
    with _TREE_REUSE_LOCK:
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
    """The AlphaZero engine — torch and a trained model are hard
    requirements now that the python MCTS fallback is gone."""
    az = _az_module()
    if az is None:
        raise RuntimeError("AlphaZero needs the optional torch stack installed")
    model = load_az_model(game)
    if model is None:
        key = "normal" if isinstance(game, NormalGame) else "ultimate"
        raise RuntimeError(f"AlphaZero needs a trained model (models/az_{key}.pt)")
    return az.alphazero_move(game, model, budget)


def az_policy_priors(game):
    """AZNet policy priors over the legal moves of the current position:
    {(m, i): p} renormalised across legal moves (mover's view). None when
    the board is Normal, the game is over, no trained model exists, or the
    optional torch/numpy deps are absent — callers treat that as "no heat"
    (the assistant board overlay)."""
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
    """The MCTS-family dispatch — the native C tree search IS the engine.

    The pure-Python MCTS stack retired with the removal project: without
    the engine DLL there is no MCTS backend, so this raises instead of
    degrading to a search that no longer exists. Normal boards have no C
    tree either (the kernels serialize Ultimate layouts only).
    """
    if isinstance(game, NormalGame):
        raise RuntimeError("the MCTS family applies to Ultimate boards only")
    if not (_ce.available(engine=True) and cfg_engine("c_mcts", True)):
        raise RuntimeError("the MCTS family requires the native engine (python dev.py build)")
    if ai_type == "MCTS":
        mv = _c_mcts_move(game, budget, on_progress, trace)
    else:
        mv = _c_rave_grave_move(
            game, budget, grave=(ai_type == "MCTS+GRAVE"), on_progress=on_progress, trace=trace
        )
    if mv is None:
        raise RuntimeError(f"{ai_type}: the native engine returned no move")
    return mv


def get_ai_move(game, ai_type, mcts_budget=800, minimax_depth=3, on_progress=None, trace=None):
    """Pick a move for `ai_type`.

    Optional hooks for showing the engine's thinking:
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
        if not (_ce.available(engine=True) and cfg_engine("c_minimax", True)):
            raise RuntimeError("Minimax requires the native engine (python dev.py build)")
        if isinstance(game, NormalGame):
            r = _ce.minimax_normal(game.board, game.current)
            if r and r[0] is not None:
                return r[0]
            return None
        return _ce.minimax_ultimate(game, minimax_depth, opts=_c_search_opts())
    if ai_type == "Minimax Pro":
        # the C backend (negamax + TT + killers + LMR + ID) serves both
        # board sizes: on Normal minimax_normal IS the full solve, so the
        # python full-solve fallback retired with minimax.py
        if not (_ce.available(engine=True) and cfg_engine("c_minimax", True)):
            raise RuntimeError("Minimax Pro requires the native engine (python dev.py build)")
        if isinstance(game, NormalGame):
            r = _ce.minimax_normal(game.board, game.current)
            if r and r[0] is not None:
                return r[0]
            return None
        return _c_minimax_pro(game, minimax_depth)
    if ai_type in ("MCTS", "MCTS+RAVE", "MCTS+GRAVE"):
        return _mcts_move(game, ai_type, mcts_budget, on_progress, trace)
    if ai_type == "Flat MCTS":
        # the C flat kernel serializes Ultimate boards only
        if isinstance(game, NormalGame):
            raise RuntimeError("Flat MCTS applies to Ultimate boards only")
        if not (_ce.available(engine=True) and cfg_engine("c_mcts", True)):
            raise RuntimeError("Flat MCTS requires the native engine (python dev.py build)")
        moves = game.legal_moves()
        if not moves:
            return None
        return _ce.flat_mcts(game, max(1, mcts_budget // len(moves)), opts=_c_search_opts())
    if ai_type == "Solver":
        if isinstance(game, NormalGame):
            # pure table lookup from data/tt_normal.bin (--build-tables).
            return solver_move(game, mcts_budget)
        # Ultimate: the native tree search IS the solver (the python MCTS
        # fallback retired with the removal project)
        if not (_ce.available(engine=True) and cfg_engine("c_mcts", True)):
            raise RuntimeError("Solver requires the native engine (python dev.py build)")
        return _c_mcts_move(game, mcts_budget, on_progress, trace)
    if ai_type == "AlphaZero":
        return alphazero_move(game, mcts_budget)
    raise ValueError(f"Unknown AI type: {ai_type}")
