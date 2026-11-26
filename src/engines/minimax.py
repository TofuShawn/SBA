# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""minimax — the minimax family: plain alpha-beta, the Ultimate
depth-limited variant and Minimax Pro (negamax + TT + killers + LMR)."""

import math
import random
import threading
import time

from core.game import NormalGame, apply_move
from config import cfg_engine

from analysis.analysis import blocks_immediate_win, eval_ultimate
from engines.cbridge import _tt_key


def _minimax_normal(game, alpha, beta, maximizing, ai_player, depth=0):
    result = game.result()
    if result is not None:
        if result == "D":
            return 0
        return (100 - depth) if result == ai_player else (depth - 100)
    if maximizing:
        best = -math.inf
        for m in game.legal_moves():
            g = game.clone()
            g.make_move(m)
            best = max(best, _minimax_normal(g, alpha, beta, False, ai_player, depth + 1))
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    best = math.inf
    for m in game.legal_moves():
        g = game.clone()
        g.make_move(m)
        best = min(best, _minimax_normal(g, alpha, beta, True, ai_player, depth + 1))
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best


def minimax_move_normal(game):
    player = game.current
    best_moves, best_score = [], -math.inf
    for m in sorted(game.legal_moves(), key=lambda i: i != 4):
        g = game.clone()
        g.make_move(m)
        score = _minimax_normal(g, -math.inf, math.inf, False, player)
        if score > best_score:
            best_score, best_moves = score, [m]
        elif score == best_score:
            best_moves.append(m)
    return random.choice(best_moves), best_score


def _minimax_ultimate(game, depth, alpha, beta, maximizing, ai_player):
    result = game.result()
    if result is not None:
        if result == "D":
            return 0
        return 100000 - depth if result == ai_player else depth - 100000
    if depth == 0:
        return eval_ultimate(game, ai_player)
    if maximizing:
        best = -math.inf
        for m in game.legal_moves():
            g = game.clone()
            apply_move(g, m)
            best = max(best, _minimax_ultimate(g, depth - 1, alpha, beta, False, ai_player))
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    best = math.inf
    for m in game.legal_moves():
        g = game.clone()
        apply_move(g, m)
        best = min(best, _minimax_ultimate(g, depth - 1, alpha, beta, True, ai_player))
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best


def minimax_move_ultimate(game, depth=3):
    player = game.current
    legal = game.legal_moves()
    if len(legal) <= 6:  # dynamic depth: near the end, search one ply deeper
        depth += 1

    def order_key(m):
        g = game.clone()
        apply_move(g, m)
        return eval_ultimate(g, player)

    best_moves, best_score = [], -math.inf
    for m in sorted(legal, key=order_key, reverse=True):
        g = game.clone()
        apply_move(g, m)
        score = _minimax_ultimate(g, depth - 1, -math.inf, math.inf, False, player)
        if score > best_score:
            best_score, best_moves = score, [m]
        elif score == best_score:
            best_moves.append(m)
    return random.choice(best_moves)


def _tt_max():
    return cfg_engine("tt_max", 300000)


# Per-thread because the Qt build runs two workers that both call Minimax Pro
# (AIWorker and AnalysisWorker); shared state would make them clear each other.

_tls = threading.local()


def _get_tt():
    tt = getattr(_tls, "tt", None)
    if tt is None:
        tt = _tls.tt = {}
    return tt


def _get_killers():
    killers = getattr(_tls, "killers", None)
    if killers is None:
        killers = _tls.killers = {}
    return killers


def _move_order_score(game, move):
    player = game.current
    g = game.clone()
    apply_move(g, move)
    if g.result() == player:
        return 1000000
    if blocks_immediate_win(game, move):
        return 500000
    if isinstance(game, NormalGame):
        if move == 4:
            return 30000
        if move in (0, 2, 6, 8):
            return 10000
        return 0
    mm, mi = move
    score = 0
    if mi == 4:
        score += 30000
    elif mi in (0, 2, 6, 8):
        score += 10000
    if mm == 4:
        score += 20000
    elif mm in (0, 2, 6, 8):
        score += 5000
    return score


def _order_moves(game, tt_move=None, depth=None):
    moves = list(game.legal_moves())
    if len(moves) <= 1:
        return moves
    if depth is None and not isinstance(game, NormalGame) and cfg_engine("use_eval_ordering", True):
        # Root ordering uses the same static eval as plain minimax_move_ultimate
        # (better pruning and tie-breaking, measured ~32% -> ~45% vs plain
        # Minimax at depth 4). Interior nodes keep the cheap heuristic so the
        # per-node cost stays low.
        player = game.current
        scored = []
        for m in moves:
            g = game.clone()
            apply_move(g, m)
            scored.append((eval_ultimate(g, player), m))
    else:
        scored = [(_move_order_score(game, m), m) for m in moves]
    if tt_move is not None:
        for i, (s, m) in enumerate(scored):
            if m == tt_move:
                scored[i] = (s + 10**9, m)
    if depth is not None and cfg_engine("use_killers", True):
        for k in _get_killers().get(depth, []):
            for i, (s, m) in enumerate(scored):
                if m == k:
                    scored[i] = (s + 10**8, m)
    scored.sort(key=lambda x: -x[0])
    return [m for _, m in scored]


def _negamax_tt(game, depth, alpha, beta):
    result = game.result()
    if result is not None:
        if result == "D":
            return 0
        return 100000 if result == game.current else -100000
    if depth == 0:
        return eval_ultimate(game, game.current)
    key = _tt_key(game)
    entry = _get_tt().get(key)
    if entry is not None and entry[0] >= depth:
        _, flag, score, _ = entry
        if flag == 0:
            return score
        if flag == -1:
            alpha = max(alpha, score)
        else:
            beta = min(beta, score)
        if alpha >= beta:
            return score
    alpha0 = alpha
    best, best_move = -math.inf, None
    tt_move = entry[3] if entry is not None else None
    for idx, m in enumerate(_order_moves(game, tt_move, depth)):
        g = game.clone()
        apply_move(g, m)
        if cfg_engine("use_lmr", True) and depth >= 4 and idx >= 2 and m != tt_move:
            val = -_negamax_tt(g, depth - 2, -beta, -alpha)
            if val > alpha:
                val = -_negamax_tt(g, depth - 1, -beta, -alpha)
        else:
            val = -_negamax_tt(g, depth - 1, -beta, -alpha)
        if val > best:
            best, best_move = val, m
        alpha = max(alpha, val)
        if alpha >= beta:
            if (
                cfg_engine("use_killers", True)
                and depth >= 2
                and best_move is not None
                and best_move != tt_move
            ):
                killers = _get_killers()
                killers.setdefault(depth, [])
                if best_move not in killers[depth]:
                    killers[depth].append(best_move)
                    if len(killers[depth]) > 2:
                        killers[depth].pop(0)
            break
    flag = 0
    if best <= alpha0:
        flag = 1
    elif best >= beta:
        flag = -1
    if len(_get_tt()) < _tt_max():
        _get_tt()[key] = (depth, flag, best, best_move)
    return best


def minimax_pro_move(game, depth=5, time_limit=8.0):
    """Normal: perfect full search. Ultimate: ID negamax + TT + time cap."""
    _get_tt().clear()
    _get_killers().clear()
    if isinstance(game, NormalGame):
        return minimax_move_normal(game)[0]
    start = time.time()
    best_move = random.choice(game.legal_moves())

    def search_root(d, alpha, beta):
        bm, bv = best_move, -math.inf
        for m in _order_moves(game):
            g = game.clone()
            apply_move(g, m)
            val = -_negamax_tt(g, d - 1, -beta, -alpha)
            if val > bv:
                bv, bm = val, m
            if time.time() - start > time_limit:
                return None, None
        return bm, bv

    use_asp = cfg_engine("use_aspiration", True)
    prev_val = None
    for d in range(1, depth + 1):
        alpha, beta = -math.inf, math.inf
        if use_asp and d > 1 and prev_val is not None:
            margin = max(200, int(abs(prev_val) * 0.1))
            alpha, beta = prev_val - margin, prev_val + margin
        bm, bv = search_root(d, alpha, beta)
        if bm is None:
            return best_move
        if use_asp and d > 1 and prev_val is not None and (bv <= alpha or bv >= beta):
            bm, bv = search_root(d, -math.inf, math.inf)
            if bm is None:
                return best_move
        prev_val = bv
        best_move = bm
        if abs(bv) >= 100000 or time.time() - start > time_limit:
            break
    return best_move
