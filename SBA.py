"""Ultimate Tic Tac Toe — NiceGUI Web App.

Features:
- Normal Tic Tac Toe and Ultimate Tic Tac Toe
- PvP, Player vs Computer, Computer vs Computer
- AI types: Random, Basic, Minimax, MCTS, AlphaZero (neural-guided MCTS)
- Bilingual (English / Traditional Chinese) UI with a Material Design 3 style
- AI Assistant analysis panel with best-move evaluations

Run:
    run.bat                    # start the web app at http://127.0.0.1:8080
    run.bat --self-test        # run headless checks
"""

import asyncio
import math
import random
import sys
import uuid

try:
    from nicegui import app, background_tasks, ui
except ModuleNotFoundError:
    print('NiceGUI is not installed for this Python interpreter.')
    print('Use run.bat (project virtual environment), or install with:  python -m pip install nicegui')
    raise SystemExit(1)

X = 'X'
O = 'O'
EMPTY = ''
LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]
AI_TYPES = ['Random', 'Basic', 'Minimax', 'MCTS', 'AlphaZero']

CSS = '''
.body--light { background: #FDF8FF; }
.tic-cell { font-weight: 800 !important; }
.tic-x, .q-btn.tic-x, .q-btn.tic-x .q-btn__content { color: #6750A4 !important; }
.tic-o, .q-btn.tic-o, .q-btn.tic-o .q-btn__content { color: #B3261E !important; }
.body--dark .tic-x, .body--dark .q-btn.tic-x, .body--dark .q-btn.tic-x .q-btn__content { color: #D0BCFF !important; }
.body--dark .tic-o, .body--dark .q-btn.tic-o, .body--dark .q-btn.tic-o .q-btn__content { color: #FFB4AB !important; }
.tic-large { width: 92px !important; height: 92px !important; min-width: 92px !important; min-height: 92px !important; font-size: 42px !important; border-radius: 14px !important; }
.tic-small { width: 40px !important; height: 40px !important; min-width: 40px !important; min-height: 40px !important; font-size: 21px !important; border-radius: 9px !important; }
.body--light .tic-empty { background: #F3EDF7 !important; }
.body--light .tic-filled { background: #ECE6F0 !important; }
.body--dark .tic-empty { background: #49454F !important; }
.body--dark .tic-filled { background: #3D3A41 !important; }
.tic-empty:hover { background: #EADDFF !important; }
.body--dark .tic-empty:hover { background: #635B70 !important; }
.q-btn--disabled .q-btn__content { opacity: 1 !important; }
.macro-board { border: 2px solid transparent; border-radius: 18px; padding: 6px; background: rgba(0,0,0,0.04); }
.body--dark .macro-board { background: rgba(255,255,255,0.06); }
.macro-active { border-color: #6750A4 !important; box-shadow: 0 0 0 3px rgba(103,80,164,0.25) !important; background: rgba(103,80,164,0.08) !important; }
.body--dark .macro-active { border-color: #D0BCFF !important; box-shadow: 0 0 0 3px rgba(208,188,255,0.25) !important; }
.macro-won-X { background: rgba(103,80,164,0.16) !important; }
.macro-won-O { background: rgba(179,38,30,0.12) !important; }
.macro-draw { opacity: 0.45; }
.macro-board { position: relative; }
.macro-win-badge {
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  border-radius: 14px;
  background: rgba(255,255,255,0.45); pointer-events: none; z-index: 5;
}
.body--dark .macro-win-badge { background: rgba(0,0,0,0.5); }
.macro-win-line { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 6; }
.macro-win-line svg { width: 100%; height: 100%; display: block; }
.macro-win-svg { width: 100%; height: 100%; display: block; opacity: 0.8; }
.board-win-line { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 8; color: #1D1B20; }
.board-win-line svg { width: 100%; height: 100%; display: block; }
.body--dark .board-win-line { color: #E6E0E9; }
.macro-win-badge-x { color: #6750A4; }
.macro-win-badge-o { color: #B3261E; }
.body--dark .macro-win-badge-x { color: #D0BCFF; }
.body--dark .macro-win-badge-o { color: #FFB4AB; }
.mark-chip { font-weight: 900; font-size: 1.5rem; line-height: 1; }
.mark-x { color: #6750A4; }
.mark-o { color: #B3261E; }
.body--dark .mark-x { color: #D0BCFF; }
.body--dark .mark-o { color: #FFB4AB; }
@keyframes cell-flash {
  0% { box-shadow: 0 0 0 0 rgba(103,80,164,0); }
  50% { box-shadow: 0 0 0 7px rgba(103,80,164,0.55); }
  100% { box-shadow: 0 0 0 0 rgba(103,80,164,0); }
}
.cell-flash { animation: cell-flash 0.9s ease; }
.analysis-row { cursor: pointer; }
.board-wrap { position: relative; padding: 10px; border-radius: 20px; background: rgba(0,0,0,0.03); }
.body--dark .board-wrap { background: rgba(255,255,255,0.04); }
'''


def t(en: str, zh: str) -> str:
    return f'{en} — {zh}'


def set_mark(el, player):
    el.classes(remove='mark-x mark-o')
    el.classes(add='mark-x' if player == X else 'mark-o')
    el.set_text('✕' if player == X else '○')


def side_label(kind):
    if kind == 'Human':
        return t('Human (You)', '玩家 (你)')
    return f'Computer ({kind}) — 電腦 ({kind})'


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
        print('AlphaZero: no trained model (models/az_%s.pt), falling back to MCTS'
              % ('normal' if isinstance(game, NormalGame) else 'ultimate'))
        return mcts_move(game, budget)
    return alphazero.alphazero_move(game, model, budget)


def get_ai_move(game, ai_type, mcts_budget=800):
    if ai_type == 'Random':
        return random.choice(game.legal_moves())
    if ai_type == 'Basic':
        return get_basic_move(game)
    if ai_type == 'Minimax':
        if isinstance(game, NormalGame):
            return minimax_move_normal(game)[0]
        return minimax_move_ultimate(game, depth=3)
    if ai_type == 'MCTS':
        return mcts_move(game, mcts_budget)
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
        for m in game.legal_moves():
            g = game.clone()
            g.make_move(m)
            r = g.result()
            if r == player:
                pct = 1.0
            elif r == 'D':
                pct = 0.5
            elif r == opp:
                pct = 0.0
            else:
                score = _minimax_normal(g, -math.inf, math.inf, False, player)
                pct = 1.0 if score > 0 else (0.5 if score == 0 else 0.0)
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


# ============================================================
# Web UI
# ============================================================

AI_OPTIONS = {
    'AlphaZero': 'AlphaZero — Neural MCTS（神經網路MCTS）',
    'Random': 'Random — 隨機',
    'Basic': 'Basic — 基礎',
    'Minimax': 'Minimax — 極小化極大',
    'MCTS': 'MCTS — 蒙地卡羅',
}
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


@ui.page('/')
def main_page():
    sid = app.storage.user.get('sid')
    if sid is None or sid not in SESSIONS:
        sid = str(uuid.uuid4())
        app.storage.user['sid'] = sid
        SESSIONS[sid] = new_session()
    session = SESSIONS[sid]
    cell_refs = {}

    ui.colors(primary='#6750A4', secondary='#625B71', accent='#B3261E')
    ui.add_css(CSS)
    dark = ui.dark_mode(value=False)

    with ui.header().classes('items-center justify-between'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('grid_on').classes('text-primary')
            ui.label('Ultimate Tic Tac Toe — 終極井字棋').classes('text-h6 q-ma-none')
        with ui.row().classes('items-center gap-4'):
            back_btn = ui.button(t('Back to Menu', '返回選單'), icon='arrow_back',
                                 on_click=lambda: show_menu())
            back_btn.set_visibility(False)
            ui.switch(t('Dark', '深色')).bind_value(dark)

    content = ui.column().classes('w-full items-center p-6 gap-6')

    def show_menu():
        session['screen'] = 'menu'
        back_btn.set_visibility(False)
        content.clear()
        with content:
            build_menu()

    def show_game():
        session['screen'] = 'game'
        back_btn.set_visibility(True)
        content.clear()
        with content:
            build_game()

    def build_menu():
        with ui.card().classes('w-full max-w-3xl'):
            with ui.column().classes('w-full gap-4 q-pa-md'):
                ui.label(t('Game Setup', '遊戲設定')).classes('text-h5')

                ui.label(t('Game Type', '遊戲類型')).classes('text-subtitle1 q-mb-0')
                game_type_radio = ui.radio({
                    'normal': 'Normal Tic Tac Toe (普通井字棋)',
                    'ultimate': 'Ultimate Tic Tac Toe (終極井字棋)',
                }, value=session['game_type']).props('inline')
                game_type_radio.on_value_change(lambda e: session.update(game_type=e.value))

                ui.label(t('Mode', '模式')).classes('text-subtitle1 q-mb-0')
                mode_radio = ui.radio({
                    'pvp': 'PvP (玩家對玩家)',
                    'pvc': 'Player vs Computer (玩家對電腦)',
                    'cvc': 'Computer vs Computer (電腦對電腦)',
                }, value=session['mode']).props('inline')
                mode_radio.on_value_change(
                    lambda e: (session.update(mode=e.value), update_ai_visibility()))

                first_radio = ui.radio({
                    'human': 'You move first — X (你先手 — X)',
                    'computer': 'Computer moves first — O (電腦先手 — O)',
                }, value=session['first_player']).props('inline')
                first_radio.on_value_change(
                    lambda e: (session.update(first_player=e.value), update_ai_visibility()))

                ai_x_sel = ui.select(
                    AI_OPTIONS, label=t('Player X — AI Level', '玩家 X — AI 等級'),
                    value=session['ai_x'])
                ai_o_sel = ui.select(
                    AI_OPTIONS, label=t('Player O — AI Level', '玩家 O — AI 等級'),
                    value=session['ai_o'])
                ai_x_sel.on_value_change(lambda e: session.update(ai_x=e.value))
                ai_o_sel.on_value_change(lambda e: session.update(ai_o=e.value))

                def update_ai_visibility():
                    ai_x_sel.set_visibility(session['mode'] == 'cvc')
                    ai_o_sel.set_visibility(session['mode'] in ('pvc', 'cvc'))
                    first_radio.set_visibility(session['mode'] == 'pvc')
                    if session['mode'] == 'pvc':
                        label = (t('Computer (X) — AI Level', '電腦 (X) — AI 等級')
                                 if session.get('first_player', 'human') == 'computer'
                                 else t('Computer (O) — AI Level', '電腦 (O) — AI 等級'))
                    else:
                        label = t('Player O — AI Level', '玩家 O — AI 等級')
                    ai_o_sel.set_label(label)
                update_ai_visibility()

                ui.label('AlphaZero — 神經網路 MCTS · 重新訓練: python alphazero.py train --game normal|ultimate').classes(
                    'text-caption text-grey q-mb-0')

                mcts_label = ui.label(
                    t('MCTS Strength', 'MCTS 強度') + f': {session["mcts"]}')
                mcts_slider = ui.slider(min=200, max=3000, step=100,
                                        value=session['mcts']).props('label-always')
                mcts_slider.on_value_change(lambda e: (
                    session.update(mcts=int(e.value)),
                    mcts_label.set_text(t('MCTS Strength', 'MCTS 強度') + f': {int(e.value)}'),
                ))

                ui.button(t('Start Game', '開始遊戲'), icon='play_arrow',
                          on_click=start_game).props('unelevated').classes('w-full')

    def start_game():
        session['game'] = (NormalGame() if session['game_type'] == 'normal'
                           else UltimateGame())
        session['analysis_gen'] += 1
        session['analyzing'] = False
        session['reanalyze'] = False
        session['ai_busy'] = None
        show_game()

    def build_game():
        game = session['game']
        cell_refs.clear()

        def refresh_status():
            result = game.result()
            if result in (X, O):
                set_mark(status_mark, result)
                status_text.set_text(f'Player {result} wins! (玩家 {result} 獲勝！)')
            elif result == 'D':
                status_mark.classes(remove='mark-x mark-o')
                status_mark.set_text('—')
                status_text.set_text(t("It's a draw!", '平局！'))
            else:
                player = game.current
                set_mark(status_mark, player)
                side = t('Player X', '玩家 X') if player == X else t('Player O', '玩家 O')
                action = (t('your move', '輪到你') if current_side_type(session) == 'Human'
                          else t('thinking...', '思考中'))
                status_text.set_text(f'{side} · {action}')
            mode_text = {'pvp': 'PvP', 'pvc': 'PvC', 'cvc': 'CvC'}[session['mode']]
            game_text = 'Normal' if isinstance(game, NormalGame) else 'Ultimate'
            x_type, o_type = side_types(session)
            header_info.set_text(f'{game_text} · {mode_text}')
            x_info.set_text(f"Player X — 玩家 X: {side_label(x_type)}")
            o_info.set_text(f"Player O — 玩家 O: {side_label(o_type)}")

        def render_board():
            board_ui.clear()
            legal = set(game.legal_moves())
            with board_ui:
                if isinstance(game, NormalGame):
                    with ui.grid(columns=3).classes('gap-1.5'):
                        for i in range(9):
                            mark = game.board[i]
                            btn = ui.button(mark, on_click=lambda i=i: on_cell_click(i))
                            btn.mark(f'cell-{i}')
                            btn.props('flat square')
                            cls = 'tic-cell tic-large'
                            if mark == X:
                                cls += ' tic-x tic-filled'
                            elif mark == O:
                                cls += ' tic-o tic-filled'
                            else:
                                cls += ' tic-empty'
                            btn.classes(cls)
                            if not mark and i in legal:
                                btn.props('ripple')
                            else:
                                btn.disable()
                            cell_refs[i] = btn
                else:
                    with ui.grid(columns=3).classes('gap-2'):
                        for m in range(9):
                            cls = 'macro-board'
                            if game.macro[m] in (X, O):
                                cls += f' macro-won-{game.macro[m]}'
                            elif game.macro[m] == 'D':
                                cls += ' macro-draw'
                            if m == game.active_macro:
                                cls += ' macro-active'
                            with ui.element('div').classes(cls):
                                with ui.grid(columns=3).classes('gap-1'):
                                    for i in range(9):
                                        mark = game.micro[m][i]
                                        btn = ui.button(
                                            mark,
                                            on_click=lambda m=m, i=i: on_cell_click((m, i)))
                                        btn.mark(f'cell-{m}-{i}')
                                        btn.props('flat square')
                                        cell_cls = 'tic-cell tic-small'
                                        if mark == X:
                                            cell_cls += ' tic-x tic-filled'
                                        elif mark == O:
                                            cell_cls += ' tic-o tic-filled'
                                        else:
                                            cell_cls += ' tic-empty'
                                        btn.classes(cell_cls)
                                        if not mark and (m, i) in legal:
                                            btn.props('ripple')
                                        else:
                                            btn.disable()
                                        cell_refs[(m, i)] = btn
                                if game.macro[m] in (X, O):
                                    badge = ui.html(win_badge_svg(game.macro[m]))
                                    badge.mark(f'macro-badge-{m}')
                                    badge.classes('macro-win-badge'
                                                  + (' macro-win-badge-x' if game.macro[m] == X
                                                     else ' macro-win-badge-o'))
                                    line = micro_win_line(game.micro[m])
                                    if line is not None:
                                        (x1, y1), (x2, y2) = win_segment(line, line_coords)
                                        svg = (f'<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">'
                                               f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                                               f'stroke="currentColor" stroke-width="5" stroke-linecap="round"/></svg>')
                                        ui.html(svg).mark(f'macro-line-{m}').classes(
                                            'macro-win-line'
                                            + (' mark-x' if game.macro[m] == X else ' mark-o'))
                whole_line = (micro_win_line(game.board) if isinstance(game, NormalGame)
                              else micro_win_line(game.macro))
                if whole_line is not None:
                    whole_winner = (game.board[whole_line[0]] if isinstance(game, NormalGame)
                                    else game.macro[whole_line[0]])
                    if whole_winner in (X, O):
                        coord = line_coords if isinstance(game, NormalGame) else macro_center
                        (x1, y1), (x2, y2) = win_segment(whole_line, coord)
                        svg = (f'<svg viewBox="0 0 100 100" preserveAspectRatio="none" '
                               f'xmlns="http://www.w3.org/2000/svg">'
                               f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                               f'stroke="currentColor" stroke-width="4" stroke-linecap="round"/></svg>')
                        ui.html(svg).mark('board-win-line').classes('board-win-line')

        def flash(move):
            el = cell_refs.get(move)
            if el is None:
                return
            el.classes(add='cell-flash')
            ui.timer(1.0, lambda el=el, m=move: (cell_refs.get(m) is el and el.classes(remove='cell-flash')), once=True)

        def on_cell_click(move):
            if session['screen'] != 'game' or game.is_over() or is_ai_turn(session):
                return
            if move not in game.legal_moves():
                return
            apply_move(game, move)
            session['analysis_gen'] += 1
            render_board()
            refresh_status()
            trigger_analysis()
            if game.is_over():
                show_result()

        def apply_ai_move(move):
            if session['screen'] != 'game' or game is not session['game']:
                return False
            if game.is_over() or move not in game.legal_moves():
                return False
            apply_move(game, move)
            session['analysis_gen'] += 1
            render_board()
            refresh_status()
            trigger_analysis()
            if game.is_over():
                show_result()
            update_cvc_controls()
            return True

        async def finish_ai_move(ai_type, budget):
            try:
                move = await asyncio.to_thread(get_ai_move, game, ai_type, budget)
            except Exception as e:
                print('AI move error:', e)
                if session.get('ai_busy') == id(game):
                    session['ai_busy'] = None
                return
            if session.get('ai_busy') == id(game):
                session['ai_busy'] = None
            apply_ai_move(move)

        def step_ai_move():
            if session['screen'] != 'game' or game is not session['game']:
                return False
            if game.is_over() or not is_ai_turn(session):
                return False
            x_type, o_type = side_types(session)
            ai_type = x_type if game.current == X else o_type
            if ai_type == 'AlphaZero':
                if session.get('ai_busy') is not None:
                    return False
                session['ai_busy'] = id(game)
                background_tasks.create(finish_ai_move(ai_type, session['mcts']))
                return False
            return apply_ai_move(get_ai_move(game, ai_type, session['mcts']))

        def ai_loop():
            if not session.get('cvc_auto', True):
                return
            step_ai_move()

        def update_cvc_controls():
            if session['mode'] != 'cvc':
                return
            ai_turn = not game.is_over() and is_ai_turn(session)
            step_btn.set_enabled(not session.get('cvc_auto', True) and ai_turn)

        def on_speed_change(e):
            speed = float(e.value)
            session['cvc_speed'] = speed
            speed_label.set_text(t('Speed', '速度') + f': {speed:.1f}s')
            game_timer.interval = speed

        def on_auto_change(e):
            session['cvc_auto'] = e.value
            update_cvc_controls()

        def render_analysis(items):
            analysis_ui.clear()
            if not items:
                with analysis_ui:
                    ui.label(t('No moves to analyze', '沒有可分析的棋步')).classes(
                        'text-caption text-grey')
                return
            with analysis_ui:
                ui.label(t('Best Moves', '最佳棋步')).classes('text-subtitle1 q-mb-xs')
                for it in items[:5]:
                    pct = max(0.0, min(1.0, it['pct']))
                    if isinstance(game, NormalGame):
                        verdict = 'Win' if pct == 1.0 else ('Draw' if pct == 0.5 else 'Loss')
                    else:
                        verdict = f'{pct:.0%}'
                    reason_en, reason_zh = it['reason']
                    with ui.row().classes('w-full items-center gap-2 analysis-row'):
                        ui.button(move_text(it['move']),
                                  on_click=lambda m=it['move']: flash(m)).props(
                            'flat dense outline')
                        with ui.column().classes('flex-1 gap-0'):
                            ui.linear_progress(value=pct, show_value=False).props(
                                'rounded').classes('w-full')
                            ui.label(f'{verdict} · {t(reason_en, reason_zh)}').classes(
                                'text-caption text-grey')

        async def start_analysis():
            current_game = session['game']
            if current_game is None:
                return
            gen = session['analysis_gen']
            session['analyzing'] = True
            snapshot = current_game.clone()
            analysis_ui.clear()
            with analysis_ui:
                with ui.row().classes('items-center gap-2'):
                    ui.spinner(size='sm')
                    ui.label(t('Analyzing...', '分析中...')).classes('text-caption')
            budget = session['mcts'] if isinstance(snapshot, UltimateGame) else 0
            items = await asyncio.to_thread(compute_analysis, snapshot, budget)
            if session['game'] is not current_game:
                session['analyzing'] = False
                return
            session['analyzing'] = False
            if (gen == session['analysis_gen'] and session['screen'] == 'game'
                    and session['assistant_enabled']):
                render_analysis(items)
            if session.get('reanalyze'):
                session['reanalyze'] = False
                background_tasks.create(start_analysis())

        def trigger_analysis():
            if not session['assistant_enabled']:
                return
            if session['game'] is None or session['game'].is_over():
                return
            if session['analyzing']:
                session['reanalyze'] = True
                return
            background_tasks.create(start_analysis())

        def toggle_assistant(value):
            session['assistant_enabled'] = value
            if value:
                trigger_analysis()
            else:
                session['reanalyze'] = False
                analysis_ui.clear()
                with analysis_ui:
                    ui.label(t('Assistant disabled', '助手已關閉')).classes(
                        'text-caption text-grey')

        def show_result():
            result = game.result()
            title = (t("It's a draw!", '平局！') if result == 'D'
                     else f'Player {result} wins! (玩家 {result} 獲勝！)')
            try:
                with ui.dialog() as dlg, ui.card():
                    ui.label(title).classes('text-h5')
                    ui.label(t('What would you like to do?', '你想做什麼？')).classes(
                        'text-body2 text-grey')
                    with ui.row().classes('gap-2'):
                        ui.button(t('Play Again', '再玩一次'),
                                  on_click=lambda: (dlg.close(), start_game()))
                        ui.button(t('Back to Menu', '返回選單'),
                                  on_click=lambda: (dlg.close(), show_menu()))
                dlg.open()
            except RuntimeError:
                pass

        with ui.row().classes('w-full justify-center gap-6 items-start flex-wrap'):
            with ui.column().classes('items-center gap-3'):
                with ui.row().classes('items-center gap-2'):
                    status_mark = ui.label('').classes('mark-chip')
                    status_text = ui.label('').classes('text-h6')
                board_ui = ui.element('div').classes('board-wrap')
                render_board()
                with ui.row().classes('gap-2'):
                    ui.button(t('New Game', '新遊戲'), icon='replay',
                              on_click=start_game).props('flat')
            with ui.column().classes('w-80 gap-3'):
                with ui.card().classes('w-full'):
                    with ui.column().classes('gap-1'):
                        ui.label(t('Game Info', '遊戲資訊')).classes('text-subtitle1')
                        header_info = ui.label('').classes('text-caption text-grey')
                        with ui.row().classes('items-center gap-2'):
                            ui.label('✕').classes('mark-chip mark-x')
                            x_info = ui.label('').classes('text-body2')
                        with ui.row().classes('items-center gap-2'):
                            ui.label('○').classes('mark-chip mark-o')
                            o_info = ui.label('').classes('text-body2')
                        assistant_switch = ui.switch(
                            t('AI Assistant', 'AI 助手'), value=session['assistant_enabled'])
                        assistant_switch.on_value_change(
                            lambda e: toggle_assistant(e.value))
                if session['mode'] == 'cvc':
                    with ui.card().classes('w-full'):
                        with ui.column().classes('gap-2'):
                            ui.label(t('CvC Controls', '電腦對戰控制')).classes('text-subtitle1')
                            speed_label = ui.label(
                                t('Speed', '速度') + f": {session.get('cvc_speed', 0.4):.1f}s")
                            speed_slider = ui.slider(
                                min=0.1, max=2.0, step=0.1,
                                value=session.get('cvc_speed', 0.4)).props('label-always')
                            speed_slider.on_value_change(on_speed_change)
                            auto_switch = ui.switch(
                                t('Auto-play', '自動播放'),
                                value=session.get('cvc_auto', True))
                            auto_switch.on_value_change(on_auto_change)
                            step_btn = ui.button(
                                t('Step / Next Move', '下一步'), icon='skip_next',
                                on_click=step_ai_move).props('flat')
                            step_btn.mark('step-btn')
                            step_btn.disable()
                with ui.card().classes('w-full'):
                    analysis_ui = ui.column().classes('gap-1')
                    ui.label(t('Click a move to highlight it on the board',
                               '點擊棋步可在棋盤上標示')).classes(
                        'text-caption text-grey q-mb-xs')

        refresh_status()
        trigger_analysis()
        update_cvc_controls()
        game_timer = ui.timer(max(0.05, session.get('cvc_speed', 0.4)), ai_loop)

    show_menu()

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
                       ('ultimate', 'Minimax', 'Random')):
        game = NormalGame() if gt == 'normal' else UltimateGame()
        guard = 0
        while not game.is_over() and guard < 500:
            ai = ax if game.current == X else ao
            apply_move(game, get_ai_move(game, ai, 300))
            guard += 1
        check(f'cvc {gt} ({ax} vs {ao}) terminates', game.is_over() and guard <= 500)

    for gt in ('normal', 'ultimate'):
        for ai in ('Random', 'Basic', 'Minimax', 'MCTS'):
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
    ui.run(
        title='Ultimate Tic Tac Toe — 終極井字棋',
        reload=False,
        storage_secret='ultimate-tic-tac-toe-sba',
    )


if __name__ == '__main__':
    main()
