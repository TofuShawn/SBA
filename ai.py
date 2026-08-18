# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: MIT

"""Ultimate Tic Tac Toe — AI engines and the AI assistant analysis.

Implements every computer opponent (Random, Basic, Minimax, Minimax Pro,
MCTS, MCTS+RAVE, Solver, AlphaZero) plus the best-move analysis used by the
AI assistant panel. Pure logic on top of game.py — no NiceGUI dependency.
"""

import math
import random
import time

from game import (
    X, O, EMPTY, LINES,
    NormalGame, UltimateGame, apply_move, apply_clone_result, count_threats,
)

AI_TYPES = ['Random', 'Basic', 'Minimax', 'Minimax Pro', 'MCTS', 'MCTS+RAVE', 'AlphaZero', 'Solver']

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
    for m in range(9):
        state = game.macro[m]
        if state == player:
            score += 1000
        elif state == opp:
            score -= 1000
        elif state == EMPTY:
            score += 3 * count_threats(game.micro[m], player)
            score -= 3 * count_threats(game.micro[m], opp)
            score += 0.5 * sum(1 for c in game.micro[m] if c == player)
            score -= 0.5 * sum(1 for c in game.micro[m] if c == opp)
    for a, b, c in LINES:
        vals = [game.macro[a], game.macro[b], game.macro[c]]
        if vals.count(player) == 2 and EMPTY in vals:
            score += 20
        if vals.count(opp) == 2 and EMPTY in vals:
            score -= 20
    return score


def _minimax_ultimate(game, depth, alpha, beta, maximizing, ai_player):
    result = game.result()
    if result is not None:
        if result == 'D':
            return 0
        return 100000 if result == ai_player else -100000
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

    def order_key(m):
        g = game.clone()
        apply_move(g, m)
        return eval_ultimate(g, player)

    best_moves, best_score = [], -math.inf
    for m in sorted(game.legal_moves(), key=order_key, reverse=True):
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


def mcts_search(game, iterations):
    root_state = game.clone()
    root = MCTSNode(root_state, None, None, None)
    for _ in range(iterations):
        node = root
        state = root_state.clone()
        while node.untried == [] and node.children:
            node = node.best_child(1.4)
            apply_move(state, node.move)
        if node.untried:
            m = node.untried.pop()
            mover = state.current
            apply_move(state, m)
            child = MCTSNode(state.clone(), m, node, mover)
            node.children.append(child)
            node = child
        result = state.result()
        while result is None:
            moves = state.legal_moves()
            if not moves:
                break
            apply_move(state, random.choice(moves))
            result = state.result()
        while node is not None:
            node.visits += 1
            if result == node.mover:
                node.wins += 1.0
            elif result == 'D':
                node.wins += 0.5
            node = node.parent
    return root


def mcts_move(game, iterations):
    root = mcts_search(game, iterations)
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


def mcts_rave_search(game, iterations, c=1.4, k=150):
    root_state = game.clone()
    root = RAVENode(root_state, None, None, None)
    for _ in range(iterations):
        node = root
        state = root_state.clone()
        while node.untried == [] and node.children:
            node = node.best_child(c, k)
            apply_move(state, node.move)
        rollout_moves = []
        if node.untried:
            m = node.untried.pop()
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
            m = random.choice(moves)
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
    return root


def mcts_rave_move(game, iterations):
    root = mcts_rave_search(game, iterations)
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


# ---------------------------------------------------------------
# Minimax Pro: negamax + transposition table + iterative deepening
# ---------------------------------------------------------------

_TT = {}
_TT_MAX = 300000


def _tt_key(game):
    if isinstance(game, NormalGame):
        return ''.join(c or '.' for c in game.board)
    return (''.join(c or '.' for row in game.micro for c in row)
            + '|' + str(game.active_macro) + game.current)


def _move_order_score(game, move):
    player = game.current
    opp = O if player == X else X
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


def _order_moves(game, tt_move=None):
    moves = list(game.legal_moves())
    if len(moves) <= 1:
        return moves
    scored = [(_move_order_score(game, m), m) for m in moves]
    if tt_move is not None:
        for i, (s, m) in enumerate(scored):
            if m == tt_move:
                scored[i] = (s + 10 ** 9, m)
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
    entry = _TT.get(key)
    if entry is not None and entry[0] >= depth:
        d, flag, score, _ = entry
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
    for m in _order_moves(game, entry[3] if entry is not None else None):
        g = game.clone()
        apply_move(g, m)
        val = -_negamax_tt(g, depth - 1, -beta, -alpha)
        if val > best:
            best, best_move = val, m
        alpha = max(alpha, val)
        if alpha >= beta:
            break
    flag = 0
    if best <= alpha0:
        flag = 1
    elif best >= beta:
        flag = -1
    if len(_TT) < _TT_MAX:
        _TT[key] = (depth, flag, best, best_move)
    return best


def minimax_pro_move(game, depth=5, time_limit=8.0):
    """Normal: perfect full search. Ultimate: ID negamax + TT + time cap."""
    _TT.clear()
    if isinstance(game, NormalGame):
        return minimax_move_normal(game)[0]
    start = time.time()
    best_move = random.choice(game.legal_moves())
    for d in range(1, depth + 1):
        best, best_val = best_move, -math.inf
        for m in _order_moves(game):
            g = game.clone()
            apply_move(g, m)
            val = -_negamax_tt(g, d - 1, -math.inf, math.inf)
            if val > best_val:
                best_val, best = val, m
            if time.time() - start > time_limit:
                return best_move
        best_move = best
        if abs(best_val) >= 100000 or time.time() - start > time_limit:
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


def get_ai_move(game, ai_type, mcts_budget=800, minimax_depth=3):
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
    if ai_type == 'MCTS':
        return mcts_move(game, mcts_budget)
    if ai_type == 'MCTS+RAVE':
        return mcts_rave_move(game, mcts_budget)
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
    opp = O if player == X else X
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

def compute_analysis(game, mcts_budget):
    player = game.current
    opp = O if player == X else X
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
    else:
        root = mcts_search(game, mcts_budget)
        for child in sorted(root.children,
                            key=lambda c: (-c.visits, -c.wins / max(1, c.visits)))[:5]:
            if child.visits == 0:
                continue
            items.append({
                'move': child.move,
                'pct': child.wins / child.visits,
                'reason': reason_for_move(game, child.move),
            })
    return items


def move_text(move):
    if isinstance(move, int):
        r, c = divmod(move, 3)
        return f'({r + 1},{c + 1})'
    m, i = move
    r, c = divmod(i, 3)
    return f'B{m + 1} ({r + 1},{c + 1})'

