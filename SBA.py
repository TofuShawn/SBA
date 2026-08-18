"""Ultimate Tic Tac Toe — core game engine and AI.

Contains the game rules (Normal / Ultimate), all AI engines (Random, Basic,
Minimax, Minimax Pro, MCTS, MCTS+RAVE, AlphaZero, Solver), the AI assistant
analysis, headless self-tests, and the CLI entry point. The NiceGUI web UI
lives in webui.py and is imported on demand.

Run:
    run.bat                    # start the web app at http://127.0.0.1:8080
    run.bat --self-test        # run headless checks
    run.bat --debug            # verbose backend logs
"""

import logging
import math
import random
import sys
import time

log = logging.getLogger('SBA')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

X = 'X'
O = 'O'
EMPTY = ''
LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]
AI_TYPES = ['Random', 'Basic', 'Minimax', 'Minimax Pro', 'MCTS', 'MCTS+RAVE', 'AlphaZero', 'Solver']


# ============================================================
# Game engines
# ============================================================

class NormalGame:
    def __init__(self):
        self.board = [EMPTY] * 9
        self.current = X

    def legal_moves(self):
        return [i for i, c in enumerate(self.board) if c == EMPTY]

    def make_move(self, index):
        if self.board[index] != EMPTY:
            raise ValueError('cell is occupied')
        self.board[index] = self.current
        if self.result() is None:
            self.current = O if self.current == X else X

    def winner(self):
        for a, b, c in LINES:
            if self.board[a] and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def is_full(self):
        return EMPTY not in self.board

    def is_over(self):
        return self.result() is not None

    def result(self):
        w = self.winner()
        if w:
            return w
        if self.is_full():
            return 'D'
        return None

    def clone(self):
        g = NormalGame()
        g.board = self.board[:]
        g.current = self.current
        return g

class UltimateGame:
    def __init__(self):
        self.micro = [[EMPTY] * 9 for _ in range(9)]
        self.macro = [EMPTY] * 9
        self.current = X
        self.active_macro = None

    def macro_open(self, m):
        return self.macro[m] == EMPTY and any(c == EMPTY for c in self.micro[m])

    def legal_moves(self):
        if self.active_macro is not None and self.macro_open(self.active_macro):
            return [(self.active_macro, i)
                    for i in range(9) if self.micro[self.active_macro][i] == EMPTY]
        moves = []
        for m in range(9):
            if not self.macro_open(m):
                continue
            for i in range(9):
                if self.micro[m][i] == EMPTY:
                    moves.append((m, i))
        return moves

    def micro_winner(self, m):
        cells = self.micro[m]
        for a, b, c in LINES:
            if cells[a] in (X, O) and cells[a] == cells[b] == cells[c]:
                return cells[a]
        return None

    def make_move(self, macro, micro):
        if self.micro[macro][micro] != EMPTY:
            raise ValueError('cell is occupied')
        self.micro[macro][micro] = self.current
        w = self.micro_winner(macro)
        if w:
            self.macro[macro] = w
        elif all(c != EMPTY for c in self.micro[macro]):
            self.macro[macro] = 'D'
        self.active_macro = micro if self.macro_open(micro) else None
        self.current = O if self.current == X else X

    def winner(self):
        for a, b, c in LINES:
            if self.macro[a] in (X, O) and self.macro[a] == self.macro[b] == self.macro[c]:
                return self.macro[a]
        return None

    def is_full(self):
        return all(not self.macro_open(m) for m in range(9))

    def is_over(self):
        return self.result() is not None

    def result(self):
        w = self.winner()
        if w:
            return w
        if self.is_full():
            return 'D'
        return None

    def clone(self):
        g = UltimateGame()
        g.micro = [row[:] for row in self.micro]
        g.macro = self.macro[:]
        g.current = self.current
        g.active_macro = self.active_macro
        return g


def apply_move(game, move):
    if isinstance(game, UltimateGame):
        game.make_move(*move)
    else:
        game.make_move(move)


def apply_clone_result(game, move):
    g = game.clone()
    apply_move(g, move)
    return g.result()


def count_threats(cells, player):
    n = 0
    for a, b, c in LINES:
        vals = [cells[a], cells[b], cells[c]]
        if vals.count(player) == 2 and EMPTY in vals:
            n += 1
    return n


def micro_win_line(cells):
    for a, b, c in LINES:
        if cells[a] in (X, O) and cells[a] == cells[b] == cells[c]:
            return (a, b, c)
    return None


def line_coords(i):
    r, c = divmod(i, 3)
    return 20 + c * 30, 20 + r * 30


def win_segment(line, coord):
    pts = [coord(i) for i in line]
    x1 = min(p[0] for p in pts)
    x2 = max(p[0] for p in pts)
    y1 = next(p[1] for p in pts if p[0] == x1)
    y2 = next(p[1] for p in pts if p[0] == x2)
    if x1 == x2:
        return (x1, 4), (x1, 96)
    slope = (y2 - y1) / (x2 - x1)
    margin = 7 if abs(slope) > 0.5 else 4
    return (margin, y1 + (margin - x1) * slope), (100 - margin, y2 + (100 - margin - x2) * slope)


def macro_center(m):
    r, c = divmod(m, 3)
    return 100.0 / 6 * (2 * c + 1), 100.0 / 6 * (2 * r + 1)


def win_badge_svg(player):
    if player == X:
        inner = ('<line x1="14" y1="14" x2="86" y2="86" stroke="currentColor" '
                 'stroke-width="12" stroke-linecap="round"/>'
                 '<line x1="86" y1="14" x2="14" y2="86" stroke="currentColor" '
                 'stroke-width="12" stroke-linecap="round"/>')
    else:
        inner = ('<circle cx="50" cy="50" r="36" fill="none" stroke="currentColor" '
                 'stroke-width="12" stroke-linecap="round"/>')
    return (f'<svg class="macro-win-svg" viewBox="0 0 100 100" '
            f'xmlns="http://www.w3.org/2000/svg">{inner}</svg>')


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

# Session state helpers (shared with webui.py and self-tests)
SESSIONS = {}

def new_session():
    return {
        'screen': 'menu',
        'game_type': 'normal',
        'mode': 'pvp',
        'first_player': 'human',
        'cvc_speed': 0.4,
        'cvc_auto': True,
        'ai_x': 'Minimax',
        'ai_o': 'MCTS',
        'mcts': 800,
        'minimax_depth': 4,
        'assistant_enabled': True,
        'analyzing': False,
        'reanalyze': False,
        'analysis_gen': 0,
        'game': None,
    }


def side_types(s):
    if s['mode'] == 'pvp':
        return 'Human', 'Human'
    if s['mode'] == 'pvc':
        if s.get('first_player', 'human') == 'computer':
            return s['ai_o'], 'Human'
        return 'Human', s['ai_o']
    return s['ai_x'], s['ai_o']


def current_side_type(s):
    x_type, o_type = side_types(s)
    return x_type if s['game'].current == X else o_type


def is_ai_turn(s):
    return current_side_type(s) != 'Human'


# ============================================================
# Self-test
# ============================================================

def self_test():
    random.seed(12345)
    passed, failed = 0, 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f'  PASS  {name}')
        else:
            failed += 1
            print(f'  FAIL  {name}')

    g = NormalGame()
    for mv in (0, 3, 1, 4, 2):
        g.make_move(mv)
    check('normal: X wins on top row', g.result() == X)

    g = NormalGame()
    for mv in (0, 1, 2, 4, 3, 5, 8, 6, 7):
        g.make_move(mv)
    check('normal: draw detected', g.result() == 'D' and g.is_full())
    check('normal: no legal moves when full', g.legal_moves() == [])

    g = UltimateGame()
    g.make_move(0, 2)
    check('ultimate: move routes to macro 2', g.active_macro == 2 and g.current == O)
    moves = g.legal_moves()
    check('ultimate: legal moves limited to macro 2',
          len(moves) == 9 and all(m == 2 for m, _ in moves))

    g = UltimateGame()
    g.micro[0] = [X, O, X, X, O, O, O, X, X]
    g.macro[0] = 'D'
    g.active_macro = 0
    g.current = X
    check('ultimate: full macro board frees move choice',
          any(m != 0 for m, _ in g.legal_moves()))

    g = UltimateGame()
    g.macro[4] = O
    g.active_macro = 4
    g.current = X
    check('ultimate: won macro board frees move choice',
          any(m != 4 for m, _ in g.legal_moves()))

    g = UltimateGame()
    g.micro[3] = [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    g.current = X
    g.make_move(3, 2)
    check('ultimate: micro win claims macro cell', g.macro[3] == X)

    g = UltimateGame()
    g.macro = [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    g.micro[2] = [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    g.current = X
    g.make_move(2, 2)
    check('ultimate: macro win detected', g.winner() == X)

    g = UltimateGame()
    g.macro = ['D'] * 9
    for m in range(9):
        g.micro[m] = [X, O, X, X, O, O, O, X, X]
    check('ultimate: draw detected', g.result() == 'D' and g.is_full())

    g = NormalGame()
    g.board = [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    g.current = X
    check('basic: takes winning move', get_basic_move(g) == 2)

    g = NormalGame()
    g.board = [EMPTY, EMPTY, EMPTY, O, O, EMPTY, EMPTY, EMPTY, EMPTY]
    g.current = X
    check('basic: blocks opponent win', get_basic_move(g) == 5)

    s = new_session()
    s['mode'] = 'pvc'
    s['first_player'] = 'computer'
    check('pvc: computer first makes AI play X', side_types(s) == (s['ai_o'], 'Human'))

    g = NormalGame()
    g.board = [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    g.current = X
    move, score = minimax_move_normal(g)
    check('minimax: picks immediate win', move == 2 and score > 0)

    wins, draws = 0, 0
    for _ in range(6):
        g = NormalGame()
        while not g.is_over():
            move = (minimax_move_normal(g)[0] if g.current == X
                    else random.choice(g.legal_moves()))
            g.make_move(move)
        if g.result() == X:
            wins += 1
        elif g.result() == 'D':
            draws += 1
    check('minimax: X never loses to random (6 games)', wins + draws == 6)

    g = NormalGame()
    g.board = [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    g.current = X
    check('mcts: picks immediate win', mcts_move(g, 1500) == 2)

    g = UltimateGame()
    check('mcts: legal move on empty ultimate', mcts_move(g, 300) in g.legal_moves())

    g = NormalGame()
    g.board = [X, X, EMPTY, EMPTY, O, O, EMPTY, EMPTY, EMPTY]
    g.current = X
    mv = solver_move(g)
    b = g.board[:]
    b[mv] = g.current
    check('solver: picks a winning move', -build_tablebase()[_board_key(b)] == 1)
    check('solver: tablebase covers reachable states', len(build_tablebase()) > 5000)
    g = NormalGame()
    check('solver: legal move on empty normal', get_ai_move(g, 'Solver', 300) in g.legal_moves())

    g = NormalGame()
    g.board = [X, X, EMPTY, EMPTY, O, O, EMPTY, EMPTY, EMPTY]
    g.current = X
    check('minimax pro: normal immediate win', minimax_pro_move(g, depth=9) == 2)
    check('mcts+rave: normal immediate win', mcts_rave_move(g, 1500) == 2)

    gu = UltimateGame()
    gu.macro = [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    gu.micro[2] = [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    gu.active_macro = 2
    gu.current = X
    check('minimax pro: ultimate game-winning move', minimax_pro_move(gu, depth=4) == (2, 2))
    check('mcts+rave: ultimate game-winning move', mcts_rave_move(gu, 2000) == (2, 2))

    seg = win_segment((0, 1, 2), line_coords)
    check('geometry: horizontal line spans full width', seg == ((4, 20), (96, 20)))
    seg = win_segment((0, 4, 8), line_coords)
    check('geometry: diagonal line spans corners', seg == ((7, 7), (93, 93)))
    seg = win_segment((0, 4, 8), macro_center)
    check('geometry: macro diagonal spans corners', seg == ((7, 7), (93, 93)))
    check('geometry: badge svg has cross/circle',
          '<line' in win_badge_svg(X) and '<circle' in win_badge_svg(O))

    import alphazero
    m3 = alphazero.train('normal', games=2, sims=8, quiet=True, save=False)
    check('alphazero: smoke train normal', m3 is not None)
    g = NormalGame()
    check('alphazero: legal move normal', alphazero.select_move(g, m3, 20) in g.legal_moves())
    m9 = alphazero.train('ultimate', games=2, sims=6, quiet=True, save=False)
    check('alphazero: smoke train ultimate', m9 is not None)
    g = UltimateGame()
    check('alphazero: legal move ultimate', alphazero.select_move(g, m9, 20) in g.legal_moves())
    g = NormalGame()
    mv = get_ai_move(g, 'AlphaZero', 20)
    check('alphazero: get_ai_move dispatches', mv in g.legal_moves())
    g = NormalGame()
    g.board = [X, X, EMPTY, O, O, EMPTY, EMPTY, EMPTY, EMPTY]
    g.current = X
    check('alphazero: normal immediate win found',
          alphazero.select_move(g, m3, 60) == 2)
    az9 = alphazero.AZNet(9)  # random-init network: tests search mechanics only
    gu = UltimateGame()
    gu.macro = [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    gu.micro[2] = [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    gu.active_macro = 2
    gu.current = X
    check('alphazero: ultimate game-winning move found',
          alphazero.select_move(gu, az9, 60) == (2, 2))
    gw = NormalGame()
    gw.board = [X, X, X, O, O, EMPTY, EMPTY, EMPTY, EMPTY]
    check('alphazero: terminal value sign (normal win)',
          alphazero.terminal_value(gw) == -1.0)
    gwu = UltimateGame()
    gwu.macro = [X, X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    check('alphazero: terminal value sign (ultimate win)',
          alphazero.terminal_value(gwu) == -1.0)

    for gt, ax, ao in (('normal', 'Random', 'Random'),
                       ('ultimate', 'Random', 'Random'),
                       ('ultimate', 'Minimax', 'Random'),
                       ('normal', 'Solver', 'Random'),
                       ('ultimate', 'Minimax Pro', 'Random'),
                       ('ultimate', 'MCTS+RAVE', 'Random')):
        game = NormalGame() if gt == 'normal' else UltimateGame()
        guard = 0
        while not game.is_over() and guard < 500:
            ai = ax if game.current == X else ao
            apply_move(game, get_ai_move(game, ai, 300))
            guard += 1
        check(f'cvc {gt} ({ax} vs {ao}) terminates', game.is_over() and guard <= 500)

    for gt in ('normal', 'ultimate'):
        for ai in ('Random', 'Basic', 'Minimax', 'Minimax Pro', 'MCTS', 'MCTS+RAVE', 'Solver'):
            game = NormalGame() if gt == 'normal' else UltimateGame()
            guard = 0
            while not game.is_over() and guard < 500:
                if game.current == X:
                    apply_move(game, random.choice(game.legal_moves()))
                else:
                    apply_move(game, get_ai_move(game, ai, 300))
                guard += 1
            check(f'pvc {gt} vs {ai} terminates', game.is_over() and guard <= 500)

    print(f'\n{passed} passed, {failed} failed')
    return 1 if failed else 0


def main():
    if '--self-test' in sys.argv:
        sys.exit(self_test())
    if '--train-az' in sys.argv:
        import alphazero
        rest = [a for a in sys.argv[1:] if a != '--train-az']
        sys.exit(alphazero.main(['train'] + rest))
    # When SBA.py is the entry script it is '__main__'; register it under the
    # canonical name so webui.py's `from SBA import ...` reuses this module
    # instead of importing a second copy.
    sys.modules.setdefault('SBA', sys.modules['__main__'])
    import webui
    webui.run()


if __name__ == '__main__':
    main()

