# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ultimate Tic Tac Toe — AI engines and the AI assistant analysis.

Implements every computer opponent (Random, Basic, Minimax, Minimax Pro,
MCTS, MCTS+RAVE, MCTS+GRAVE, Flat MCTS, Solver, AlphaZero) plus the
best-move analysis used by the AI assistant panel. Pure logic on top of
game.py — no NiceGUI dependency.

Maintenance notes:
- MCTS+RAVE is hidden from the menus but kept for tests and --bench
  (decision D8: MCTS+GRAVE supersedes it).
- Flat MCTS is hidden from the menus too; it is a research/learning
  baseline (root-level playouts, no tree).
- Solver is menu-disabled but still used by the analysis panel (D3).
- AlphaZero falls back to MCTS when no trained model exists (see models/).
"""

import logging
import math
import os
import random
import threading
import time

from game import (
    X, O, EMPTY, LINES,
    NormalGame, UltimateGame, BitUltimateGame,
    apply_move, apply_clone_result, count_threats,
)

log = logging.getLogger('ai')


# ============================================================
# Engine configuration (sba.toml)
# ============================================================

_CFG_DEFAULTS = {
    'engine': {
        'rollout_heuristic': True, 'tree_reuse': True, 'dynamic_uct': True,
        'progressive_widening': False, 'early_stop': False,
        'opening_book': True, 'micro_tablebase': True,
        'symmetry': False, 'object_pool': True, 'bitboard': True,
        'multithreaded': False, 'use_lmr': True, 'use_killers': True,
        'use_aspiration': True,
        'uct_c': 1.4, 'rave_k': 150, 'tt_max': 300000,
        'workers': 4, 'reuse_cache': 32,
    },
    'session': {'mcts_budget': 800, 'minimax_depth': 4},
}


def _load_config():
    """Load sba.toml (or $SBA_CONFIG) as a dict; missing/broken file -> {}."""
    path = os.environ.get('SBA_CONFIG')
    if not path:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sba.toml')
    try:
        import tomllib
        with open(path, 'rb') as fh:
            return tomllib.load(fh)
    except Exception:
        return {}


CONFIG = _load_config()


def set_engine_config(overrides=None):
    """Reset the engine config to code defaults, then apply overrides."""
    global CONFIG
    base = {
        'engine': dict(_CFG_DEFAULTS['engine']),
        'session': dict(_CFG_DEFAULTS['session']),
    }
    if overrides:
        base['engine'].update(overrides.get('engine', {}))
        base['session'].update(overrides.get('session', {}))
    CONFIG = base


def cfg_engine(name, default=None):
    return CONFIG.get('engine', {}).get(name, _CFG_DEFAULTS['engine'].get(name, default))


def cfg_session(name, default=None):
    return CONFIG.get('session', {}).get(name, _CFG_DEFAULTS['session'].get(name, default))


# ============================================================
# AI engines
# ============================================================

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

def _minimax_normal(game, alpha, beta, maximizing, ai_player, depth=0):
    result = game.result()
    if result is not None:
        if result == 'D':
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


def eval_ultimate(game, player):
    opp = O if player == X else X
    score = 0
    tb = build_micro_tablebase() if cfg_engine('micro_tablebase', True) else None
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


def _minimax_ultimate(game, depth, alpha, beta, maximizing, ai_player):
    result = game.result()
    if result is not None:
        if result == 'D':
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

class MCTSNode:
    __slots__ = ('state', 'move', 'parent', 'children', 'mover', 'visits', 'wins', 'untried')

    def __init__(self, state, move, parent, mover):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.mover = mover
        self.visits = 0
        self.wins = 0.0
        self.untried = state.legal_moves()
        random.shuffle(self.untried)

    def best_child(self, c):
        log_n = math.log(max(1, self.visits))
        best, best_val = None, -math.inf
        for child in self.children:
            if child.visits == 0:
                uct = math.inf
            else:
                uct = child.wins / child.visits + c * math.sqrt(log_n / child.visits)
            if uct > best_val:
                best, best_val = child, uct
        return best


class NodePool:
    """Reusable MCTSNode objects to reduce GC churn.

    Only used when tree reuse is off — a retained subtree must never be
    pooled while it is still referenced.
    """

    def __init__(self):
        self._free = []

    def acquire(self, state, move, parent, mover):
        if self._free:
            n = self._free.pop()
            n.state = state
            n.move = move
            n.parent = parent
            n.children = []
            n.mover = mover
            n.visits = 0
            n.wins = 0.0
            n.untried = [] if state.is_over() else state.legal_moves()
            random.shuffle(n.untried)
            return n
        return MCTSNode(state, move, parent, mover)

    def release(self, root):
        stack = [root]
        while stack:
            n = stack.pop()
            stack.extend(n.children)
            n.children = []
            self._free.append(n)


_POOL = NodePool()


def mcts_search(game, iterations, c=None, prev_root=None, pool=None):
    c = cfg_engine('uct_c', 1.4) if c is None else c
    if prev_root is not None:
        root = prev_root
        root_state = root.state
    else:
        root_state = game.clone()
        root = (pool.acquire(root_state, None, None, None)
                if pool is not None else MCTSNode(root_state, None, None, None))
    start_visits = root.visits
    root_sym = set()
    for it in range(iterations):
        node = root
        state = root_state.clone()
        while True:
            if bool(node.untried) and _can_expand(node):
                break
            if not node.children:
                break
            node = node.best_child(_uct_scale(c, root.visits - start_visits, iterations))
            apply_move(state, node.move)
        if node.untried and _can_expand(node):
            m = _pop_untried(node, root_sym, node is root)
            if m is not None:
                mover = state.current
                apply_move(state, m)
                child = (pool.acquire(state.clone(), m, node, mover)
                         if pool is not None else MCTSNode(state.clone(), m, node, mover))
                node.children.append(child)
                node = child
        result = state.result()
        while result is None:
            moves = state.legal_moves()
            if not moves:
                break
            m = _rollout_move(state, moves)
            apply_move(state, m)
            result = state.result()
        while node is not None:
            node.visits += 1
            if result == node.mover:
                node.wins += 1.0
            elif result == 'D':
                node.wins += 0.5
            node = node.parent
        if (cfg_engine('early_stop', False) and it >= 0.7 * iterations
                and root.children):
            top = max(root.children, key=lambda c: c.visits)
            if root.visits and top.visits / root.visits > 0.6 \
                    and top.wins / top.visits > 0.55:
                break
    return root


def _best_mcts_move(root, game, iterations):
    """Pick the best child of a finished MCTS root, preferring an immediate win."""
    visited = [c for c in root.children if c.visits >= max(5, iterations // 100)]
    pool = visited if visited else list(root.children)
    best = max(pool, key=lambda c: c.wins / c.visits)
    for c in pool:
        if c.wins / c.visits == best.wins / best.visits:
            g = game.clone()
            apply_move(g, c.move)
            if g.result() == game.current:
                best = c
                break
    return best.move


def mcts_move(game, iterations):
    pool = (_POOL if (cfg_engine('object_pool', True)
                      and not cfg_engine('tree_reuse', True)) else None)
    root = mcts_search(game, iterations, pool=pool)
    move = _best_mcts_move(root, game, iterations)
    if pool is not None:
        pool.release(root)
    return move


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
    if not cfg_engine('rollout_heuristic', True):
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

def _flat_playout(game, player):
    """Random playout from `game`; +1 win / 0.5 draw / 0 loss for `player`."""
    g = game.clone()
    guard = 0
    while not g.is_over() and guard < 300:
        moves = g.legal_moves()
        if not moves:
            break
        apply_move(g, _rollout_move(g, moves))
        guard += 1
    r = g.result()
    if r == 'D':
        return 0.5
    return 1.0 if r == player else 0.0


def flat_mcts_move(game, iterations):
    """Flat/root-level Monte Carlo: split the budget evenly across moves.

    Each legal move is scored by ``iterations // len(moves)`` random playouts
    from the position after that move; the move with the best average outcome
    wins (ties prefer an immediate win). There is no tree, so it only looks
    one ply ahead — useful as a learning baseline against real MCTS.
    """
    moves = game.legal_moves()
    if not moves:
        return None
    per_move = max(1, iterations // len(moves))
    player = game.current
    avg = {}
    for m in moves:
        g = game.clone()
        apply_move(g, m)
        total = sum(_flat_playout(g, player) for _ in range(per_move))
        avg[m] = total / per_move
    best = max(avg, key=avg.get)
    for m in avg:
        if avg[m] == avg[best]:
            g = game.clone()
            apply_move(g, m)
            if g.result() == game.current:
                best = m
                break
    return best


def _can_expand(node):
    """Progressive-widening gate: limit expansions as a node's visits grow."""
    if not cfg_engine('progressive_widening', False):
        return True
    if node.visits == 0:
        return True
    return len(node.children) < math.ceil(1.5 * math.sqrt(node.visits))


def _pop_untried(node, root_sym, is_root):
    """Pop an untried move, skipping D4-symmetric duplicates at the root."""
    while node.untried:
        m = node.untried.pop()
        if cfg_engine('symmetry', False) and is_root:
            img = _sym_images(m)
            if img in root_sym:
                continue
            root_sym.add(img)
        return m
    return None


def _uct_scale(c, root_visits, iterations):
    """Dynamic UCT: explore more early, exploit more as the search matures."""
    if cfg_engine('dynamic_uct', True):
        return c * (0.6 + 0.4 * (1.0 - root_visits / max(1, iterations)))
    return c


_REUSE = {}

# The subtree right after the engine's own move, per engine. Tree reuse uses
# it to locate the opponent's reply and descend into the matching child.
_REUSE_LAST = {}


def _diff_move(old_state, new_game):
    """Return the single move leading from old_state to new_game, else None.

    Used by tree reuse: _REUSE_LAST holds the position right after the
    engine's own move, so a current position one opponent reply later differs
    by exactly one filled cell. Rewinds, branches, and different games yield
    None (fresh search).
    """
    if isinstance(new_game, NormalGame):
        if not isinstance(old_state, NormalGame):
            return None
        diff = [i for i in range(9) if old_state.board[i] != new_game.board[i]]
        if len(diff) != 1 or new_game.board[diff[0]] == EMPTY:
            return None
        return diff[0]
    if not hasattr(old_state, 'micro'):
        return None
    diffs = []
    for m in range(9):
        for i in range(9):
            if old_state.micro[m][i] != new_game.micro[m][i]:
                diffs.append((m, i))
    if len(diffs) != 1 or new_game.micro[diffs[0][0]][diffs[0][1]] == EMPTY:
        return None
    return diffs[0]


def _d4_perms():
    """Eight D4 permutations of the 9 cells of a 3x3 grid."""
    def rot(rc):
        r, c = rc
        return (c, 2 - r)

    def ref(rc):
        r, c = rc
        return (2 - r, c)

    def transform(f):
        p = [0] * 9
        for i in range(9):
            r, c = f((i // 3, i % 3))
            p[i] = r * 3 + c
        return p

    identity = lambda rc: rc
    return [transform(f) for f in (
        identity, rot, lambda rc: rot(rot(rc)), lambda rc: rot(rot(rot(rc))),
        ref, lambda rc: ref(rot(rc)), lambda rc: ref(rot(rot(rc))),
        lambda rc: ref(rot(rot(rot(rc)))),
    )]


_D4 = _d4_perms()


def _sym_images(move):
    """All D4 images of a Normal index or an Ultimate (m, i) move."""
    if isinstance(move, tuple):
        m, i = move
        return tuple(sorted((_D4[t][m], _D4[t][i]) for t in range(8)))
    return tuple(sorted(_D4[t][move] for t in range(8)))


def reset_engine_caches():
    """Drop module-level search caches (used by self-tests/bench)."""
    _REUSE.clear()
    _REUSE_LAST.clear()
    _get_tt().clear()
    _get_killers().clear()


# ---------------------------------------------------------------
# Solver: perfect normal-mode tablebase
# ---------------------------------------------------------------

_TABLEBASE = None


def _board_key(board):
    key = 0
    for c in board:
        key = key * 3 + (0 if c == EMPTY else (1 if c == X else 2))
    return key


def build_tablebase():
    global _TABLEBASE
    if _TABLEBASE is not None:
        return _TABLEBASE
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
    _TABLEBASE = memo
    return memo


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

class RAVENode:
    __slots__ = ('state', 'move', 'parent', 'children', 'mover',
                 'visits', 'wins', 'untried', 'amaf')

    def __init__(self, state, move, parent, mover):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.mover = mover
        self.visits = 0
        self.wins = 0.0
        self.untried = [] if state.is_over() else state.legal_moves()
        random.shuffle(self.untried)
        self.amaf = {}

    def best_child(self, c, k):
        log_n = math.log(max(1, self.visits))
        best, best_val = None, -math.inf
        for child in self.children:
            if child.visits == 0:
                uct = math.inf
            else:
                q = child.wins / child.visits
                an, aw = self.amaf.get(child.move, (0, 0.0))
                q_amaf = aw / an if an else q
                beta = math.sqrt(k / (3 * child.visits + k))
                uct = ((1 - beta) * q + beta * q_amaf
                       + c * math.sqrt(log_n / child.visits))
            if uct > best_val:
                best, best_val = child, uct
        return best


def mcts_rave_search(game, iterations, c=None, k=None, prev_root=None):
    c = cfg_engine('uct_c', 1.4) if c is None else c
    k = cfg_engine('rave_k', 150) if k is None else k
    if prev_root is not None:
        root = prev_root
        root_state = root.state
    else:
        root_state = game.clone()
        root = RAVENode(root_state, None, None, None)
    start_visits = root.visits
    root_sym = set()
    for it in range(iterations):
        node = root
        state = root_state.clone()
        while True:
            if bool(node.untried) and _can_expand(node):
                break
            if not node.children:
                break
            node = node.best_child(_uct_scale(c, root.visits - start_visits, iterations), k)
            apply_move(state, node.move)
        rollout_moves = []
        if node.untried and _can_expand(node):
            m = _pop_untried(node, root_sym, node is root)
            if m is not None:
                mover = state.current
                apply_move(state, m)
                child = RAVENode(state.clone(), m, node, mover)
                node.children.append(child)
                node = child
                rollout_moves.append(m)
        result = state.result()
        while result is None:
            moves = state.legal_moves()
            if not moves:
                break
            m = _rollout_move(state, moves)
            apply_move(state, m)
            rollout_moves.append(m)
            result = state.result()
        while node is not None:
            node.visits += 1
            win = 1.0 if result == node.mover else (0.5 if result == 'D' else 0.0)
            node.wins += win
            # AMAF is scored from the perspective of the player to move at this
            # node (standard all-moves RAVE), not from the player who moved into it.
            if node.mover is None:
                amaf_win = 1.0 if result == X else (0.5 if result == 'D' else 0.0)
            else:
                amaf_p = O if node.mover == X else X
                amaf_win = 1.0 if result == amaf_p else (0.5 if result == 'D' else 0.0)
            amaf = node.amaf
            for m in rollout_moves:
                if m in amaf:
                    amaf[m][0] += 1
                    amaf[m][1] += amaf_win
                else:
                    amaf[m] = [1, amaf_win]
            node = node.parent
        if (cfg_engine('early_stop', False) and it >= 0.7 * iterations
                and root.children):
            top = max(root.children, key=lambda c: c.visits)
            if root.visits and top.visits / root.visits > 0.6 \
                    and top.wins / top.visits > 0.55:
                break
    return root


def mcts_rave_move(game, iterations):
    root = mcts_rave_search(game, iterations)
    return _best_mcts_move(root, game, iterations)


# ---------------------------------------------------------------
# MCTS + GRAVE (Generalized RAVE, Cazenave 2015)
# ---------------------------------------------------------------
# GRAVE reuses RAVE statistics through "reference" nodes: a move's RAVE
# values live on the node where that move was first tried, and descendants
# look them up from their reference instead of maintaining their own AMAF
# tables. Same bias term as RAVE, but far less memory and steadier
# estimates on wide branching factors (e.g. Ultimate).

class GraveNode:
    __slots__ = ('state', 'move', 'parent', 'children', 'mover',
                 'visits', 'wins', 'untried', 'ref', 'rave')

    def __init__(self, state, move, parent, mover):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.mover = mover
        self.visits = 0
        self.wins = 0.0
        self.untried = [] if state.is_over() else state.legal_moves()
        random.shuffle(self.untried)
        self.ref = None   # reference node providing this move's RAVE stats
        self.rave = {}    # move -> [visits, wins], only meaningful on ref nodes

    def best_child(self, c, k):
        log_n = math.log(max(1, self.visits))
        best, best_val = None, -math.inf
        for child in self.children:
            if child.visits == 0:
                uct = math.inf
            else:
                q = child.wins / child.visits
                ref = child.ref if child.ref is not None else self
                an, aw = ref.rave.get(child.move, (0, 0.0))
                q_rave = aw / an if an else q
                beta = math.sqrt(k / (3 * child.visits + k))
                uct = ((1 - beta) * q + beta * q_rave
                       + c * math.sqrt(log_n / child.visits))
            if uct > best_val:
                best, best_val = child, uct
        return best


def mcts_grave_search(game, iterations, c=None, k=None, prev_root=None):
    c = cfg_engine('uct_c', 1.4) if c is None else c
    k = cfg_engine('rave_k', 150) if k is None else k
    if prev_root is not None:
        root = prev_root
        root_state = root.state
    else:
        root_state = game.clone()
        root = GraveNode(root_state, None, None, None)
        root.ref = root
    start_visits = root.visits
    root_sym = set()
    for it in range(iterations):
        node = root
        state = root_state.clone()
        while True:
            if bool(node.untried) and _can_expand(node):
                break
            if not node.children:
                break
            node = node.best_child(_uct_scale(c, root.visits - start_visits, iterations), k)
            apply_move(state, node.move)
        rollout_moves = []
        if node.untried and _can_expand(node):
            m = _pop_untried(node, root_sym, node is root)
            if m is not None:
                mover = state.current
                apply_move(state, m)
                child = GraveNode(state.clone(), m, node, mover)
                ref = node.ref if node.ref is not None else node
                child.ref = ref if m in ref.rave else node
                node.children.append(child)
                node = child
                rollout_moves.append(m)
        result = state.result()
        while result is None:
            moves = state.legal_moves()
            if not moves:
                break
            m = _rollout_move(state, moves)
            apply_move(state, m)
            rollout_moves.append(m)
            result = state.result()
        seen_refs = set()
        while node is not None:
            node.visits += 1
            win = 1.0 if result == node.mover else (0.5 if result == 'D' else 0.0)
            node.wins += win
            ref = node.ref if node.ref is not None else node
            if id(ref) not in seen_refs:
                seen_refs.add(id(ref))
                # RAVE is scored from the perspective of the player to move.
                if node.mover is None:
                    amaf_win = 1.0 if result == X else (0.5 if result == 'D' else 0.0)
                else:
                    amaf_p = O if node.mover == X else X
                    amaf_win = 1.0 if result == amaf_p else (0.5 if result == 'D' else 0.0)
                rave = ref.rave
                for m in rollout_moves:
                    if m in rave:
                        rave[m][0] += 1
                        rave[m][1] += amaf_win
                    else:
                        rave[m] = [1, amaf_win]
            node = node.parent
        if (cfg_engine('early_stop', False) and it >= 0.7 * iterations
                and root.children):
            top = max(root.children, key=lambda c: c.visits)
            if root.visits and top.visits / root.visits > 0.6 \
                    and top.wins / top.visits > 0.55:
                break
    return root


def mcts_grave_move(game, iterations):
    root = mcts_grave_search(game, iterations)
    return _best_mcts_move(root, game, iterations)


# ---------------------------------------------------------------
# Minimax Pro: negamax + transposition table + iterative deepening
# ---------------------------------------------------------------

def _tt_max():
    return cfg_engine('tt_max', 300000)

# The transposition table and killer moves are per-thread so concurrent
# browser sessions using Minimax Pro (web runs AI via asyncio.to_thread) do
# not clear each other's caches; each thread still gets full search speed.
_tls = threading.local()


def _get_tt():
    tt = getattr(_tls, 'tt', None)
    if tt is None:
        tt = _tls.tt = {}
    return tt


def _get_killers():
    killers = getattr(_tls, 'killers', None)
    if killers is None:
        killers = _tls.killers = {}
    return killers


def _tt_key(game):
    if isinstance(game, NormalGame):
        return ''.join(c or '.' for c in game.board)
    return (''.join(c or '.' for row in game.micro for c in row)
            + '|' + str(game.active_macro) + game.current)


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
    scored = [(_move_order_score(game, m), m) for m in moves]
    if tt_move is not None:
        for i, (s, m) in enumerate(scored):
            if m == tt_move:
                scored[i] = (s + 10 ** 9, m)
    if depth is not None and cfg_engine('use_killers', True):
        for k in _get_killers().get(depth, []):
            for i, (s, m) in enumerate(scored):
                if m == k:
                    scored[i] = (s + 10 ** 8, m)
    scored.sort(key=lambda x: -x[0])
    return [m for _, m in scored]


def _negamax_tt(game, depth, alpha, beta):
    result = game.result()
    if result is not None:
        if result == 'D':
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
        if (cfg_engine('use_lmr', True) and depth >= 4 and idx >= 2
                and m != tt_move):
            val = -_negamax_tt(g, depth - 2, -beta, -alpha)
            if val > alpha:
                val = -_negamax_tt(g, depth - 1, -beta, -alpha)
        else:
            val = -_negamax_tt(g, depth - 1, -beta, -alpha)
        if val > best:
            best, best_move = val, m
        alpha = max(alpha, val)
        if alpha >= beta:
            if (cfg_engine('use_killers', True) and depth >= 2
                    and best_move is not None and best_move != tt_move):
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

    use_asp = cfg_engine('use_aspiration', True)
    prev_val = None
    for d in range(1, depth + 1):
        alpha, beta = -math.inf, math.inf
        if use_asp and d > 1 and prev_val is not None:
            margin = max(200, int(abs(prev_val) * 0.1))
            alpha, beta = prev_val - margin, prev_val + margin
        bm, bv = search_root(d, alpha, beta)
        if bm is None:
            return best_move
        if (use_asp and d > 1 and prev_val is not None
                and (bv <= alpha or bv >= beta)):
            bm, bv = search_root(d, -math.inf, math.inf)
            if bm is None:
                return best_move
        prev_val = bv
        best_move = bm
        if abs(bv) >= 100000 or time.time() - start > time_limit:
            break
    return best_move


_AZ_MODELS = {}


def load_az_model(game):
    import alphazero
    key = 'normal' if isinstance(game, NormalGame) else 'ultimate'
    if key not in _AZ_MODELS:
        _AZ_MODELS[key] = alphazero.load_model(key)
    return _AZ_MODELS[key]


def alphazero_move(game, budget=800):
    import alphazero
    model = load_az_model(game)
    if model is None:
        log.info('AlphaZero: no trained model (models/az_%s.pt), falling back to MCTS',
                 'normal' if isinstance(game, NormalGame) else 'ultimate')
        return mcts_move(game, budget)
    return alphazero.alphazero_move(game, model, budget)


_BOOK_ENGINES = {'Basic', 'Minimax', 'Minimax Pro', 'MCTS', 'MCTS+RAVE', 'MCTS+GRAVE'}

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

# Expose the Ultimate opening book through sba.toml ([engine.opening_book_ultimate],
# keys like "4,4"); the dict above remains the built-in default.
_CFG_DEFAULTS['engine']['opening_book_ultimate'] = ULTIMATE_BOOK


def _ultimate_book():
    """Config-backed Ultimate opening book (accepts tuple or 'm,i' keys)."""
    out = {}
    for k, v in (cfg_engine('opening_book_ultimate', ULTIMATE_BOOK) or {}).items():
        if isinstance(k, str):
            m, i = k.split(',')
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
        if filled == 1 and game.current == X:
            om = next(i for i, c in enumerate(game.board) if c == O)
            return 4 if om != 4 else random.choice((0, 2, 6, 8))
        return None
    filled = sum(1 for row in game.micro for c in row if c != EMPTY)
    if filled == 0 and game.current == X:
        return random.choice(((4, 4), (0, 0), (2, 2), (6, 6), (8, 8), (4, 0)))
    if filled == 1 and game.current == O:
        x_move = next((m, i) for m in range(9) for i in range(9)
                      if game.micro[m][i] == X)
        legal = set(game.legal_moves())
        picks = [r for r in _ultimate_book().get(x_move, ()) if r in legal]
        if picks:
            return random.choice(picks)
    return None


def _mcts_move(game, ai_type, budget):
    """Run an MCTS-family search, reusing the previous tree when enabled."""
    if (ai_type == 'MCTS' and cfg_engine('multithreaded', False)):
        return mcts_move_parallel(game, budget, cfg_engine('workers', 4))
    reuse = cfg_engine('tree_reuse', True)
    if cfg_engine('bitboard', True) and isinstance(game, UltimateGame):
        search_game = BitUltimateGame.from_game(game)
    else:
        search_game = game
    pool = (_POOL if (ai_type == 'MCTS'
                      and cfg_engine('object_pool', True) and not reuse) else None)
    key, prev = None, None
    if reuse:
        key = (ai_type, _tt_key(game))
        prev = _REUSE.pop(key, None)
        if prev is None:
            last = _REUSE_LAST.get(ai_type)
            if last is not None:
                reply = _diff_move(last.state, game)
                if reply is not None:
                    prev = next((c for c in last.children
                                 if c.move == reply), None)
    if ai_type == 'MCTS':
        root = mcts_search(search_game, budget, prev_root=prev, pool=pool)
    elif ai_type == 'MCTS+RAVE':
        root = mcts_rave_search(search_game, budget, prev_root=prev)
    else:
        root = mcts_grave_search(search_game, budget, prev_root=prev)
    move = _best_mcts_move(root, game, budget)
    if reuse:
        g = game.clone()
        apply_move(g, move)
        child = next((c for c in root.children if c.move == move), None)
        if child is not None:
            _REUSE[(ai_type, _tt_key(g))] = child
            _REUSE_LAST[ai_type] = child
            if len(_REUSE) > cfg_engine('reuse_cache', 32):
                _REUSE.pop(next(iter(_REUSE)))
    if pool is not None:
        pool.release(root)
    return move


def mcts_move_parallel(game, iterations, workers=4):
    """Run several independent MCTS searches and merge their root stats."""
    chunk = max(1, iterations // workers)
    results = []
    if cfg_engine('bitboard', True) and isinstance(game, UltimateGame):
        base = BitUltimateGame.from_game(game)
    else:
        base = game

    def work():
        results.append(mcts_search(base.clone(), chunk))

    threads = [threading.Thread(target=work) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stats = {}
    for root in results:
        for c in root.children:
            s = stats.setdefault(c.move, [0, 0.0])
            s[0] += c.visits
            s[1] += c.wins
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


def get_ai_move(game, ai_type, mcts_budget=800, minimax_depth=3):
    if cfg_engine('opening_book', True) and ai_type in _BOOK_ENGINES:
        book_move = opening_book_move(game)
        if book_move is not None:
            return book_move
    if ai_type == 'Random':
        return random.choice(game.legal_moves())
    if ai_type == 'Basic':
        return get_basic_move(game)
    if ai_type == 'Minimax':
        if isinstance(game, NormalGame):
            return minimax_move_normal(game)[0]
        return minimax_move_ultimate(game, depth=minimax_depth)
    if ai_type == 'Minimax Pro':
        return minimax_pro_move(game, depth=minimax_depth)
    if ai_type in ('MCTS', 'MCTS+RAVE', 'MCTS+GRAVE'):
        return _mcts_move(game, ai_type, mcts_budget)
    if ai_type == 'Flat MCTS':
        return flat_mcts_move(game, mcts_budget)
    if ai_type == 'Solver':
        return solver_move(game, mcts_budget)
    if ai_type == 'AlphaZero':
        return alphazero_move(game, mcts_budget)
    raise ValueError(f'Unknown AI type: {ai_type}')


# ============================================================
# Assistant analysis
# ============================================================

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
    player = game.current
    g = game.clone()
    apply_move(g, move)
    if g.result() == player:
        return ('Win', '致勝')
    if blocks_immediate_win(game, move):
        return ('Block', '阻擋')
    if isinstance(game, UltimateGame):
        m, i = move
        if i == 4 and game.micro[m][4] == EMPTY:
            return ('Center', '中心')
        return ('Search', '分析')
    if count_threats(g.board, player) >= 2:
        return ('Fork', '雙威脅')
    if move == 4:
        return ('Center', '中心')
    if move in (0, 2, 6, 8):
        return ('Corner', '角落')
    return ('Positional', '位置')

def _rates_from_root(root, game):
    """(X win, draw, O win) probabilities derived from an Ultimate MCTS root."""
    tot_x = tot_o = tot = 0.0
    for child in root.children:
        if child.visits == 0:
            continue
        w = child.wins / child.visits
        tot += child.visits
        if child.mover == X:
            tot_x += child.visits * w
            tot_o += child.visits * (1.0 - w)
        else:
            tot_o += child.visits * w
            tot_x += child.visits * (1.0 - w)
    if tot == 0:
        return (0.5, 0.0, 0.5)
    x, o = tot_x / tot, tot_o / tot
    return (x, max(0.0, 1.0 - x - o), o)


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
        for m in game.legal_moves():
            board = game.board[:]
            board[m] = player
            val = -table[_board_key(board)]
            pct = 1.0 if val > 0 else (0.5 if val == 0 else 0.0)
            items.append({'move': m, 'pct': pct, 'reason': reason_for_move(game, m)})
        items.sort(key=lambda it: -it['pct'])
        best = -1
        for m in game.legal_moves():
            board = game.board[:]
            board[m] = game.current
            best = max(best, -table[_board_key(board)])
        if game.current == X:
            rates = ((1.0, 0.0, 0.0) if best > 0 else
                     ((0.0, 1.0, 0.0) if best == 0 else (0.0, 0.0, 1.0)))
        else:
            rates = ((0.0, 0.0, 1.0) if best > 0 else
                     ((0.0, 1.0, 0.0) if best == 0 else (1.0, 0.0, 0.0)))
        return items, rates
    else:
        if cfg_engine('bitboard', True) and isinstance(game, UltimateGame):
            search_game = BitUltimateGame.from_game(game)
        else:
            search_game = game
        root = mcts_search(search_game, mcts_budget)
        for child in sorted(root.children,
                            key=lambda c: (-c.visits, -c.wins / max(1, c.visits)))[:5]:
            if child.visits == 0:
                continue
            items.append({
                'move': child.move,
                'pct': child.wins / child.visits,
                'reason': reason_for_move(game, child.move),
            })
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
        best = -1
        for m in game.legal_moves():
            board = game.board[:]
            board[m] = game.current
            best = max(best, -table[_board_key(board)])
        if game.current == X:
            return (1.0, 0.0, 0.0) if best > 0 else (
                (0.0, 1.0, 0.0) if best == 0 else (0.0, 0.0, 1.0))
        return (0.0, 0.0, 1.0) if best > 0 else (
            (0.0, 1.0, 0.0) if best == 0 else (1.0, 0.0, 0.0))
    if cfg_engine('bitboard', True) and isinstance(game, UltimateGame):
        search_game = BitUltimateGame.from_game(game)
    else:
        search_game = game
    root = mcts_search(search_game, mcts_budget)
    return _rates_from_root(root, game)


def position_win_rate(game, mcts_budget):
    """Whole-game win rate (draw counts 0.5) for the side to move, 0..1."""
    x, d, o = position_win_rates(game, mcts_budget)
    return (x + 0.5 * d) if game.current == X else (o + 0.5 * d)


def move_text(move):
    if isinstance(move, int):
        r, c = divmod(move, 3)
        return f'({r + 1},{c + 1})'
    m, i = move
    r, c = divmod(i, 3)
    return f'B{m + 1} ({r + 1},{c + 1})'
