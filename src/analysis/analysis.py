# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""analysis — the assistant surface: position evaluation, win-rate
overlays, best-move analysis and the reason classifier."""

from core.game import EMPTY, LINES, NormalGame, O, UltimateGame, X, apply_move, count_threats
from config import cfg_engine

from engines.cbridge import _C_PREV, _TREE_REUSE_LOCK, _c_search_opts, _ce, BitUltimateGame
from engines.mcts import mcts_search
from analysis.tablebase import _board_key, build_micro_tablebase, build_tablebase


def eval_ultimate(game, player):
    opp = O if player == X else X
    score = 0
    tb = build_micro_tablebase() if cfg_engine("micro_tablebase", True) else None
    for m in range(9):
        state = game.macro[m]
        if state == player:
            score += 1000
        elif state == opp:
            score -= 1000
        elif state == EMPTY:
            cells = game.micro[m]
            empty = sum(1 for c in cells if c == EMPTY)
            if tb is not None and empty <= 2:
                v = tb.get(_board_key(cells))
                if v is not None:
                    # The side to move inside an idle micro is ambiguous, so
                    # credit the player most likely to be moving there.
                    theirs = 9 - empty - sum(1 for c in cells if c == player)
                    sign = 1 if sum(1 for c in cells if c == player) >= theirs else -1
                    score += 80 * v * sign
                    continue
            score += 40 * _fork_count(cells, player)
            score -= 40 * _fork_count(cells, opp)
            score += 3 * count_threats(cells, player)
            score -= 3 * count_threats(cells, opp)
            score += 0.5 * sum(1 for c in cells if c == player)
            score -= 0.5 * sum(1 for c in cells if c == opp)
    for a, b, c in LINES:
        vals = [game.macro[a], game.macro[b], game.macro[c]]
        if vals.count(player) == 2 and EMPTY in vals:
            score += 20
        if vals.count(opp) == 2 and EMPTY in vals:
            score -= 20
    return score


def _fork_count(cells, player):
    """Number of empty cells whose play creates >=2 winning lines for `player`."""
    n = 0
    for i, c in enumerate(cells):
        if c != EMPTY:
            continue
        lines = 0
        for a, b, d in LINES:
            if i not in (a, b, d):
                continue
            others = [cells[j] for j in (a, b, d) if j != i]
            if others.count(player) == 2:
                lines += 1
        if lines >= 2:
            n += 1
    return n


def blocks_immediate_win(game, move):
    opp = O if game.current == X else X
    if isinstance(game, UltimateGame):
        m, idx = move
        cells = game.micro[m]
    else:
        idx = move
        cells = game.board
    if cells[idx] != EMPTY:
        return False
    for a, b, c in LINES:
        if idx not in (a, b, c):
            continue
        others = [cells[x] for x in (a, b, c) if x != idx]
        if others == [opp, opp]:
            return True
    return False


def reason_for_move(game, move):
    """Classify a move's tactical/positional nature (reason code only).

    Bilingual labels live in SBA.REASON_TEXT so this engine layer stays
    language-free; the UIs translate the code.
    """
    player = game.current
    g = game.clone()
    apply_move(g, move)
    if g.result() == player:
        return "Win"
    if blocks_immediate_win(game, move):
        return "Block"
    if isinstance(game, UltimateGame):
        m, i = move
        if i == 4 and game.micro[m][4] == EMPTY:
            return "Center"
        return "Search"
    if count_threats(g.board, player) >= 2:
        return "Fork"
    if move == 4:
        return "Center"
    if move in (0, 2, 6, 8):
        return "Corner"
    return "Positional"


def _rates_from_root(root, game):
    """(X win, draw, O win) probabilities derived from an Ultimate MCTS root.

    Each child's wins carry draws at 0.5 from the mover's perspective, so the
    draw share needs the node's raw draw counter — 1 - x - o from the score
    alone is identically zero and used to fake the triplet (regression)."""
    tot_x = tot_o = tot_d = tot = 0.0
    for child in root.children:
        if child.visits == 0:
            continue
        v = child.wins / child.visits  # mover score = win + draw/2
        d = getattr(child, "draws", 0) / child.visits
        tot += child.visits
        win_m = v - 0.5 * d  # mover's pure win share
        lose_m = 1.0 - v - 0.5 * d  # opponent's pure win share
        if child.mover == X:
            tot_x += child.visits * win_m
            tot_o += child.visits * lose_m
        else:
            tot_o += child.visits * win_m
            tot_x += child.visits * lose_m
        tot_d += child.visits * d
    if tot == 0:
        return (0.5, 0.0, 0.5)
    x, o, d = tot_x / tot, tot_o / tot, tot_d / tot
    s = x + o + d
    return (x / s, d / s, o / s) if s > 0 else (0.5, 0.0, 0.5)


def _rates_from_stats(stats):
    """(X win, draw, O win) from cengine root-child stats — the same math
    as _rates_from_root over (move, visits, wins, mover, draws) tuples."""
    tot_x = tot_o = tot_d = tot = 0.0
    for _mv, visits, wins, mover, draws in stats:
        if visits <= 0:
            continue
        v = wins / visits
        d = draws / visits
        tot += visits
        win_m = v - 0.5 * d
        lose_m = 1.0 - v - 0.5 * d
        if mover == 1:
            tot_x += visits * win_m
            tot_o += visits * lose_m
        else:
            tot_o += visits * win_m
            tot_x += visits * lose_m
        tot_d += visits * d
    if tot == 0:
        return (0.5, 0.0, 0.5)
    x, o, d = tot_x / tot, tot_o / tot, tot_d / tot
    s = x + o + d
    return (x / s, d / s, o / s) if s > 0 else (0.5, 0.0, 0.5)


def _c_ultt_analysis(game, mcts_budget):
    """One native MCTS search feeding both the top-moves list and the
    win-rate triplet (the C twin of the analyze_position Ultimate branch).
    Reuses the AI move search's warm tree when available. Returns None when
    the full engine DLL is unavailable.

    ownership rule: the borrow is a POP under _TREE_REUSE_LOCK — analysis
    takes the idle tree out of the shared slot so the AI worker can never
    hand the same arena out mid-search, and gives it back only while the
    slot is still empty (the AI's write-back always wins). While the AI owns
    the tree, analysis misses the reuse and starts fresh on its own arena;
    both searches still run concurrently because their arenas are private.
    """
    if not (_ce.available(engine=True) and cfg_engine("c_mcts", True)):
        return None
    prev = None
    if cfg_engine("tree_reuse", True):
        with _TREE_REUSE_LOCK:
            prev = _C_PREV.pop("MCTS", None)
    try:
        _mv, stats, _tree = _ce.mcts_search(game, mcts_budget, opts=_c_search_opts(), prev=prev)
    finally:
        if prev is not None:
            with _TREE_REUSE_LOCK:
                if "MCTS" not in _C_PREV:
                    _C_PREV["MCTS"] = prev
    if not stats:
        return None
    items = []
    for mv, visits, wins, _mover, _draws in sorted(stats, key=lambda s: (-s[1], -s[2]))[:5]:
        if visits <= 0:
            continue
        move = (mv // 9, mv % 9)
        items.append({"move": move, "pct": wins / visits, "reason": reason_for_move(game, move)})
    return items, _rates_from_stats(stats)


def _normal_best(game, table):
    """Exact negamax value for the side to move (Normal is fully solved)."""
    best = -1
    for m in game.legal_moves():
        board = game.board[:]
        board[m] = game.current
        best = max(best, -table[_board_key(board)])
    return best


def _normal_rates(game, best):
    """(X win, draw, O win) from the side-to-move's exact value.

    One implementation for both callers: the nested ternaries used to be
    written out twice, and the X / O branches are easy to transpose.
    """
    if best > 0:
        return (1.0, 0.0, 0.0) if game.current == X else (0.0, 0.0, 1.0)
    if best == 0:
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0) if game.current == X else (1.0, 0.0, 0.0)


def analyze_position(game, mcts_budget):
    """(items, rates) for the current position from ONE analysis.

    ``items`` is the top-moves list used by the assistant panel and ``rates``
    is the (X win, draw, O win) triplet used by the win-rate chart. Both are
    derived from the same search (tablebase on Normal, one MCTS root on
    Ultimate) so callers no longer run two separate searches.
    """
    result = game.result()
    if result is not None:
        if result == X:
            rates = (1.0, 0.0, 0.0)
        elif result == O:
            rates = (0.0, 0.0, 1.0)
        else:
            rates = (0.0, 1.0, 0.0)
        return [], rates
    player = game.current
    items = []
    if isinstance(game, NormalGame):
        table = build_tablebase()
        best = -1
        for m in game.legal_moves():
            board = game.board[:]
            board[m] = player
            val = -table[_board_key(board)]
            best = max(best, val)
            pct = 1.0 if val > 0 else (0.5 if val == 0 else 0.0)
            items.append({"move": m, "pct": pct, "reason": reason_for_move(game, m)})
        items.sort(key=lambda it: -it["pct"])
        return items, _normal_rates(game, best)
    else:
        c_analysis = _c_ultt_analysis(game, mcts_budget)
        if c_analysis is not None:
            return c_analysis
        if cfg_engine("bitboard", True) and isinstance(game, UltimateGame):
            search_game = BitUltimateGame.from_game(game)
        else:
            search_game = game
        root = mcts_search(search_game, mcts_budget)
        for child in sorted(root.children, key=lambda c: (-c.visits, -c.wins / max(1, c.visits)))[
            :5
        ]:
            if child.visits == 0:
                continue
            items.append(
                {
                    "move": child.move,
                    "pct": child.wins / child.visits,
                    "reason": reason_for_move(game, child.move),
                }
            )
        return items, _rates_from_root(root, game)


def compute_analysis(game, mcts_budget):
    """Top-moves list for the assistant panel (see analyze_position)."""
    return analyze_position(game, mcts_budget)[0]


def position_win_rates(game, mcts_budget):
    """(X win, draw, O win) probabilities for the current position.

    Normal uses the perfect tablebase; Ultimate derives the three rates from
    a single MCTS root search.
    """
    result = game.result()
    if result is not None:
        if result == X:
            return (1.0, 0.0, 0.0)
        if result == O:
            return (0.0, 0.0, 1.0)
        return (0.0, 1.0, 0.0)
    if isinstance(game, NormalGame):
        table = build_tablebase()
        return _normal_rates(game, _normal_best(game, table))
    c_analysis = _c_ultt_analysis(game, mcts_budget)
    if c_analysis is not None:
        return c_analysis[1]
    if cfg_engine("bitboard", True) and isinstance(game, UltimateGame):
        search_game = BitUltimateGame.from_game(game)
    else:
        search_game = game
    root = mcts_search(search_game, mcts_budget)
    return _rates_from_root(root, game)


def position_win_rate(game, mcts_budget):
    """Whole-game win rate (draw counts 0.5) for the side to move, 0..1."""
    x, d, o = position_win_rates(game, mcts_budget)
    return (x + 0.5 * d) if game.current == X else (o + 0.5 * d)
