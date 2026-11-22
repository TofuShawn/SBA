# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""books — static-knowledge engines: the Basic heuristic and the
Normal/Ultimate opening books."""

from core.game import EMPTY, NormalGame, O, X, apply_clone_result, apply_move
import math
import random

from config import cfg_engine


def get_basic_move(game):
    player = game.current
    opp = O if player == X else X
    moves = game.legal_moves()

    for m in moves:
        if apply_clone_result(game, m) == player:
            return m

    losing = []
    for m in moves:
        g = game.clone()
        apply_move(g, m)
        if any(apply_clone_result(g, m2) == opp for m2 in g.legal_moves()):
            losing.append(m)
    safe = [m for m in moves if m not in losing]
    if safe and len(safe) < len(moves):
        return random.choice(safe)

    if isinstance(game, NormalGame):
        for pref in (4, 0, 2, 6, 8, 1, 3, 5, 7):
            if pref in moves:
                return pref
    else:
        best, best_score = None, -math.inf
        for m in moves:
            macro, micro = m
            score = 0
            if macro == 4:
                score += 6
            if micro == 4:
                score += 4
            if macro in (0, 2, 6, 8):
                score += 2
            if micro in (0, 2, 6, 8):
                score += 1
            if score > best_score:
                best, best_score = m, score
        if best is not None:
            return best
    return random.choice(moves)


_BOOK_ENGINES = {"Basic", "Minimax", "Minimax Pro", "MCTS", "MCTS+RAVE", "MCTS+GRAVE"}

# Curated Ultimate openers: X's first move -> O's response (a cell in the
# routed macro). All entries are validated against the legal moves before use.

ULTIMATE_BOOK = {
    (4, 4): [(4, 0), (4, 2), (4, 6), (4, 8)],
    (4, 0): [(0, 4)],
    (4, 2): [(2, 4)],
    (4, 6): [(6, 4)],
    (4, 8): [(8, 4)],
    (0, 0): [(0, 4)],
    (2, 2): [(2, 4)],
    (6, 6): [(6, 4)],
    (8, 8): [(8, 4)],
    (0, 4): [(4, 0), (4, 8)],
    (2, 4): [(4, 0), (4, 6)],
    (6, 4): [(4, 2), (4, 8)],
    (8, 4): [(4, 2), (4, 6)],
}

# The Ultimate opening book is configurable through config.toml
# ([engine.opening_book_ultimate], keys like "4,4"); the dict above remains
# the built-in default, passed to cfg_engine as the fallback below. The book
# DATA deliberately never lands in config._CFG_DEFAULTS (D63): config.py's
# leaf contract is that its defaults do not depend on whether books was
# imported — a config.toml table, when present, reaches the reader through
# CONFIG, not through the defaults tree.


def _ultimate_book():
    """Config-backed Ultimate opening book (accepts tuple or 'm,i' keys)."""
    out = {}
    for k, v in (cfg_engine("opening_book_ultimate", ULTIMATE_BOOK) or {}).items():
        if isinstance(k, str):
            m, i = k.split(",")
            key = (int(m.strip()), int(i.strip()))
        else:
            key = tuple(k)
        out[key] = [tuple(r) for r in v]
    return out


def opening_book_move(game):
    """Curated early-game moves (Normal: first two plies; Ultimate: first two)."""
    if isinstance(game, NormalGame):
        filled = sum(1 for c in game.board if c != EMPTY)
        if filled == 0 and game.current == X:
            return random.choice((0, 2, 4, 6, 8))
        if filled == 1 and game.current == O:
            xm = next(i for i, c in enumerate(game.board) if c == X)
            return 4 if xm != 4 else random.choice((0, 2, 6, 8))
        return None
    filled = sum(1 for row in game.micro for c in row if c != EMPTY)
    if filled == 0 and game.current == X:
        return random.choice(((4, 4), (0, 0), (2, 2), (6, 6), (8, 8), (4, 0)))
    if filled == 1 and game.current == O:
        x_move = next((m, i) for m in range(9) for i in range(9) if game.micro[m][i] == X)
        legal = set(game.legal_moves())
        picks = [r for r in _ultimate_book().get(x_move, ()) if r in legal]
        if picks:
            return random.choice(picks)
    return None
