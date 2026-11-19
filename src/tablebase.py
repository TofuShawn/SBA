# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""tablebase — the persisted perfect-play table for Normal mode
(D38: data/tt_normal.bin + the DFS fallback) and the Solver engine."""

import math
import os
import random

from game import EMPTY, LINES, NormalGame, O, X
from config import ROOT, writable_path
from mcts import mcts_move


_TABLEBASE = None

_TABLEBASE_PATH = os.path.join(ROOT, "data", "tt_normal.bin")
# one byte per base-3 board key (0..19682): 0x00 side-to-move win,
# 0x01 draw, 0x02 side-to-move loss, 0xFF unreachable (never queried in
# legal play). Side to move is derivable from the stone counts, so the
# key alone is unambiguous.

_TB_VALUE_BYTE = {1: 0x00, 0: 0x01, -1: 0x02}

_TB_BYTE_VALUE = {b: v for v, b in _TB_VALUE_BYTE.items()}


def _board_key(board):
    key = 0
    for c in board:
        key = key * 3 + (0 if c == EMPTY else (1 if c == X else 2))
    return key


def _board_from_key(key):
    board = []
    for _ in range(9):
        board.append((EMPTY, X, O)[key % 3])
        key //= 3
    return board[::-1]


def _build_tablebase_dfs():
    """Minimax solve of every position reachable from the empty board
    (5,478 of the 3^9 = 19,683 assignments); returns {key: value} from the
    side to move. This DFS is the table's content — there is no other
    enumeration to "cover"."""
    memo = {}

    def solve(board, current):
        key = _board_key(board)
        if key in memo:
            return memo[key]
        w = None
        for a, b, c in LINES:
            if board[a] and board[a] == board[b] == board[c]:
                w = board[a]
                break
        if w is not None:
            memo[key] = 1 if w == current else -1
            return memo[key]
        if EMPTY not in board:
            memo[key] = 0
            return 0
        best = -1
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = current
                val = -solve(board, O if current == X else X)
                board[i] = EMPTY
                if val > best:
                    best = val
        memo[key] = best
        return best

    solve([EMPTY] * 9, X)
    return memo


def _read_tablebase_file(path=None):
    """Decode the persisted tablebase into the same dict the DFS produces;
    None on any structural mismatch (callers then rebuild)."""
    try:
        with open(path or _TABLEBASE_PATH, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    if len(raw) != 3**9:
        return None
    table = {}
    for key in range(3**9):
        val = _TB_BYTE_VALUE.get(raw[key])
        if val is None:
            if raw[key] != 0xFF:
                return None
            continue
        table[key] = val
    # corruption probe: the empty board is a theoretical draw
    if table.get(0) != 0:
        return None
    return table


def build_tablebase():
    global _TABLEBASE
    if _TABLEBASE is not None:
        return _TABLEBASE
    table = _read_tablebase_file()
    if table is None:
        table = _build_tablebase_dfs()
    _TABLEBASE = table
    return _TABLEBASE


def write_tablebase_file(path=None):
    """Persist the solved tablebase: one byte per base-3 key, sentinel for
    unreachable slots. Idempotent — the same solve always yields the same
    bytes. Returns (path, stats)."""
    path = path or _TABLEBASE_PATH
    table = _build_tablebase_dfs()
    raw = bytearray(b"\xff" * (3**9))
    for key, val in table.items():
        raw[key] = _TB_VALUE_BYTE[val]
    path = writable_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(bytes(raw))
    terminal = 0
    for key in table:
        board = _board_from_key(key)
        if (
            any(board[a] and board[a] == board[b] == board[c] for a, b, c in LINES)
            or EMPTY not in board
        ):
            terminal += 1
    stats = {
        "bytes": len(raw),
        "reachable": len(table),
        "terminal": terminal,
        "empty_board": table.get(0),
    }
    return path, stats


def build_micro_tablebase():
    """Exact minimax values for a single 3x3 micro (side-to-move view).

    The Normal-game tablebase already covers exactly these 3^9 states, so
    this is a thin alias that keeps the intent readable at call sites.
    """
    return build_tablebase()


def solver_move(game, mcts_budget=800):
    """Perfect play on Normal; falls back to MCTS on Ultimate."""
    if not isinstance(game, NormalGame):
        return mcts_move(game, mcts_budget)
    table = build_tablebase()
    current = game.current
    best_moves, best_val = [], -math.inf
    for m in game.legal_moves():
        board = game.board[:]
        board[m] = current
        val = -table[_board_key(board)]
        if val > best_val:
            best_val, best_moves = val, [m]
        elif val == best_val:
            best_moves.append(m)
    return random.choice(best_moves)


# ---------------------------------------------------------------
# MCTS + RAVE (AMAF sharing)
# ---------------------------------------------------------------
