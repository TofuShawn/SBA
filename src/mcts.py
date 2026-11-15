# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""mcts — the pure-Python search trees: UCT (D8), MCTS+RAVE, MCTS+GRAVE,
the node pool, tree-reuse bookkeeping and the flat playout baseline.
The C-first dispatch lives in ai.py; this module is the fallback and
the research reference."""

import math
import random

from game import (EMPTY, NormalGame, O, X, apply_move)
from config import cfg_engine
from cbridge import (_c_search_opts, _ce, _py_rollout_state, _rollout_move,
                     _tt_key, move_text)


class MCTSNode:
    __slots__ = ('state', 'move', 'parent', 'children', 'mover', 'visits',
                 'wins', 'draws', 'untried')

    def __init__(self, state, move, parent, mover):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.mover = mover
        self.visits = 0
        self.wins = 0.0
        self.draws = 0
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
            n.draws = 0
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

def mcts_search(game, iterations, c=None, prev_root=None, pool=None, on_progress=None):
    """Plain MCTS. `on_progress(info)` fires ~10 times per search with the
    current best child so UIs can show the engine's thinking (D24)."""
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
    c_roll = (None if isinstance(game, NormalGame)
              else (_ce if _ce.available() else None))
    c_opts = _c_search_opts() if c_roll is not None else None
    report_at = max(1, iterations // 10) if on_progress else 0
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
        if result is None:
            # A terminal node (the expansion move may have just won the game)
            # must not go to the C kernel: it only stops for a win it plays
            # itself and would keep emitting moves on a finished board.
            if c_roll is not None:
                result = (_ce.rollout_state(state, opts=c_opts)
                          or _py_rollout_state(state))
            else:
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
                node.draws += 1
            node = node.parent
        if (cfg_engine('early_stop', False) and it >= 0.7 * iterations
                and root.children):
            top = max(root.children, key=lambda c: c.visits)
            if root.visits and top.visits / root.visits > 0.6 \
                    and top.wins / top.visits > 0.55:
                break
        if report_at and (it + 1) % report_at == 0 and root.children:
            top = max(root.children, key=lambda ch: ch.visits)
            if root.visits and top.visits:
                on_progress({'it': it + 1, 'iterations': iterations,
                             'best_text': move_text(top.move),
                             'share': top.visits / root.visits,
                             'wr': top.wins / top.visits})
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
    """Pop an untried move, skipping D4-equivalent duplicates at the root.

    The dedup key is the CANONICAL FORM OF THE RESULTING POSITION, not the
    move's cell orbit: on an asymmetric root board two orbit-mate moves can
    lead to entirely different positions (one a forced win, one a blunder) —
    the cell-orbit key made the symmetry switch skip real wins (found by the
    2026-11-08 review round: GRAVE missed a forced win in 21/30 seeds with
    symmetry on)."""
    while node.untried:
        m = node.untried.pop()
        if cfg_engine('symmetry', False) and is_root:
            g = node.state.clone()
            apply_move(g, m)
            imgs = [_tt_key(g)]
            if isinstance(g, NormalGame):
                for t in _D4[1:]:
                    imgs.append(''.join(g.board[t[i]] or '.'
                                        for i in range(9)))
                key = min(imgs)
            else:
                # Ultimate: dedup only on the raw micro grid + routing, the
                # same encoding _tt_key uses, min over the 8 cell transforms
                base = g
                key = min(_tt_key(base) if t == 0 else _ultt_sym_key(base, t)
                          for t in range(8))
            if key in root_sym:
                continue
            root_sym.add(key)
        return m
    return None

def _ultt_sym_key(game, t):
    """Canonical key of an Ultimate position under the t-th D4 transform:
    stones, macro statuses and the active-macro routing all travel with the
    transform, so two keys are equal iff the positions are truly equivalent."""
    p9 = _D4[t]
    P = [p9[m] * 9 + p9[i] for m in range(9) for i in range(9)]
    micro = ''.join(c or '.' for row in game.micro for c in row)
    cells = ''.join(micro[P[i]] or '.' for i in range(81))
    macro_t = ''.join('.' if game.macro[p9[m]] == EMPTY
                      else str(game.macro[p9[m]]) for m in range(9))
    active = game.active_macro
    active_t = p9[active] if active is not None and active >= 0 else -1
    return cells + '|' + macro_t + '|' + str(active_t) + game.current

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

class RAVENode:
    __slots__ = ('state', 'move', 'parent', 'children', 'mover',
                 'visits', 'wins', 'draws', 'untried', 'amaf')

    def __init__(self, state, move, parent, mover):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.mover = mover
        self.visits = 0
        self.wins = 0.0
        self.draws = 0
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
    c_opts = _c_search_opts()
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
        if (_ce.available() and not isinstance(state, NormalGame)
                and state.result() is None):
            result, seq = _ce.rollout_trace(state, opts=c_opts)
            if result is None:
                result = _py_rollout_state(state)
            for m in seq:
                apply_move(state, m)
                rollout_moves.append(m)
        else:
            result = state.result()
            while result is None:
                moves = state.legal_moves()
                if not moves:
                    break
                m = _rollout_move(state, moves)
                apply_move(state, m)
                result = state.result()
                rollout_moves.append(m)
            result = state.result()
        while node is not None:
            node.visits += 1
            win = 1.0 if result == node.mover else (0.5 if result == 'D' else 0.0)
            node.wins += win
            if result == 'D':
                node.draws += 1
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
                 'visits', 'wins', 'draws', 'untried', 'ref', 'rave')

    def __init__(self, state, move, parent, mover):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.mover = mover
        self.visits = 0
        self.wins = 0.0
        self.draws = 0
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
    c_opts = _c_search_opts()
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
        if (_ce.available() and not isinstance(state, NormalGame)
                and state.result() is None):
            result, seq = _ce.rollout_trace(state, opts=c_opts)
            if result is None:
                result = _py_rollout_state(state)
            for m in seq:
                apply_move(state, m)
                rollout_moves.append(m)
        else:
            result = state.result()
            while result is None:
                moves = state.legal_moves()
                if not moves:
                    break
                m = _rollout_move(state, moves)
                apply_move(state, m)
                result = state.result()
                rollout_moves.append(m)
            result = state.result()
        seen_refs = set()
        while node is not None:
            node.visits += 1
            win = 1.0 if result == node.mover else (0.5 if result == 'D' else 0.0)
            node.wins += win
            if result == 'D':
                node.draws += 1
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
