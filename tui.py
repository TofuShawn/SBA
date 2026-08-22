# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Textual terminal UI (TUI) for SBA (Basically Awful).

Shares the game rules (game.py), the AI engines (ai.py) and the per-session
state / bilingual helpers (SBA.py) with the NiceGUI web UI. The web UI stays
opt-in: start it with `python SBA.py --web`.

Run:
    python SBA.py            # terminal app (default)

Maintenance notes:
- All AI moves and assistant analyses run in Textual Worker threads so the
  terminal never blocks; stale results are dropped via a generation counter
  (session['analysis_gen']) plus per-run tokens captured when each worker
  starts (same scheme as webui.py).
- Textual ships no Slider widget, so a tiny keyboard+mouse Slider lives here
  rather than pulling in an extra dependency.
- The board renders Unicode box-drawing and works in Windows Terminal / Git
  Bash; the side panel stacks below the board on narrow terminals via a CSS
  media query.
"""

import logging
import time

from rich.text import Text

from game import (
    X, O,
    NormalGame, UltimateGame, apply_move, micro_win_line,
)
from ai import (
    get_ai_move, analyze_position, position_win_rates, move_text,
    cfg_session,
)
from SBA import (
    AI_OPTIONS, REASON_TEXT, current_side_type, is_ai_turn, log,
    new_session, side_label, side_types, t,
)

try:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.message import Message
    from textual.screen import Screen
    from textual.widget import Widget
    from textual.widgets import (
        Button, RadioButton, RadioSet, Select, Static, Switch,
    )
except ImportError:
    print('Textual is not installed for this Python interpreter.')
    print('Install the UI with:  python -m pip install -r requirements.txt')
    print('Or start the web UI instead with:  python SBA.py --web')
    raise SystemExit(1)


X_COLOR = '#C9A2FF'   # X purple (readable on dark terminals)
O_COLOR = '#FF9E9E'   # O red
FLASH_COLOR = '#FFD54F'
GREEN = '#7FD8A3'
ACTIVE_FRAME = '#4CB8FF'
MUTED = '#8A8A96'
HEADER = '#E8DEF8'
TEXT = '#E6E0E9'
SUB = '#CAC4D0'


def _reason_label(code):
    en, zh = REASON_TEXT.get(code, (code, code))
    return f'{en} / {zh}'


def _history_point(game):
    """Win-rate triplet for the current position.

    Exact (Normal tablebase, or any terminal position) is computed inline;
    a non-terminal Ultimate position defers to the assistant's full-budget
    search (the None placeholder is filled when the analysis completes).
    """
    if isinstance(game, NormalGame) or game.is_over():
        return tuple(position_win_rates(game, 0))
    return None


def _bar(frac, width=8):
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return '█' * n + '░' * (width - n)


# ============================================================
# Small keyboard + mouse slider (Textual 8 has no built-in one)
# ============================================================

class Slider(Widget):
    """TUI slider: left/right keys or a mouse click set the value.

    Works in integer steps; value changes are reported through Slider.Changed.
    """

    class Changed(Message):
        def __init__(self, slider: 'Slider', value: int) -> None:
            super().__init__()
            self.slider = slider
            self.value = value

    can_focus = True
    DEFAULT_CSS = """
    Slider {
        height: 1;
        width: 28;
        margin: 0 1;
    }
    """

    def __init__(self, value=0, min=0, max=100, step=1,
                 *, name=None, id=None, classes=None):
        super().__init__(name=name, id=id, classes=classes)
        self.min = int(min)
        self.max = int(max)
        self.step = int(step)
        self._value = int(value)

    @property
    def value(self):
        return self._value

    def _set(self, v, *, from_message=True):
        v = max(self.min, min(self.max, int(round((v - self.min) / self.step))
                              * self.step + self.min))
        if v == self._value:
            return
        self._value = v
        self.refresh()
        if from_message:
            self.post_message(Slider.Changed(self, self._value))

    def _frac(self):
        span = self.max - self.min or 1
        return (self._value - self.min) / span

    def render(self):
        width = max(10, self.size.width or 28)
        pos = int(round(self._frac() * (width - 1)))
        bar = '─' * pos + '●' + '─' * (width - pos - 1)
        return Text(bar, style=SUB)

    async def on_click(self, event):
        width = max(10, self.size.width or 28)
        frac = min(1.0, max(0.0, event.x / max(1, width - 1)))
        self._set(int(self.min + frac * (self.max - self.min)))

    def action_decrease(self):
        self._set(self._value - self.step)

    def action_increase(self):
        self._set(self._value + self.step)

    def action_min(self):
        self._set(self.min)

    def action_max(self):
        self._set(self.max)

    BINDINGS = [
        Binding('left', 'decrease', '−', show=False),
        Binding('right', 'increase', '+', show=False),
        Binding('home', 'min', 'min', show=False),
        Binding('end', 'max', 'max', show=False),
    ]


# ============================================================
# Board widget
# ============================================================

class Board(Static):
    """Text board: renders Normal (3x3) or Ultimate (3x3 of 3x3 boards).

    Mouse clicks map back to a move through the same grid geometry used to
    draw, so clicking anywhere inside a cell picks that cell. Keyboard keys
    (arrows / hjkl) move the cursor, space plays it.
    """

    can_focus = True
    DEFAULT_CSS = """
    Board {
        height: auto;
        width: auto;
        max-width: 60;
        padding: 0 1;
        margin: 0 1;
    }
    """

    def __init__(self, *, name=None, id=None, classes=None):
        super().__init__(id=id, classes=classes)
        self.session = None
        self._legal = []
        self.cursor = None          # move under the cursor (int or (m, i))
        self._cells = {}            # (line, col) -> move
        self._lines = []
        self._ranges = []           # (line, c0, c1, style)
        self.on_board_click = None  # callable(move)

    def _game(self):
        return self.session['game'] if self.session else None

    def set_session(self, session):
        old = self.cursor
        self.session = session
        self._legal = list(session['game'].legal_moves())
        if old in self._legal:
            self.cursor = old
        else:
            # Standing on a history step (< end): put the cursor on the cell
            # of the next move that would be replayed, so revert→space
            # repeats it. Otherwise fall back to the first legal cell.
            moves = session['moves']
            step = session['step']
            prefer = moves[step] if step < len(moves) else None
            self.cursor = (prefer if prefer in self._legal
                           else (self._legal[0] if self._legal else None))
        self.refresh()

    # -- geometry ---------------------------------------------------------

    @staticmethod
    def _box_lines():
        return [
            '┌───┬───┬───┐',
            '│   │   │   │',
            '├───┼───┼───┤',
            '│   │   │   │',
            '├───┼───┼───┤',
            '│   │   │   │',
            '└───┴───┴───┘',
        ]

    @staticmethod
    def _box_cells():
        """(relative_line, relative_col) for every cell's content start."""
        for r in range(3):
            for c in range(3):
                yield 1 + 2 * r, 1 + 4 * c

    def _build(self):
        game = self._game()
        self._lines = []
        self._cells = {}
        self._ranges = []
        if game is None:
            return
        s = self.session
        lines = []
        cells = {}
        ranges = []
        last = s['moves'][s['step'] - 1] if s['step'] > 0 else None
        reason = s.get('reason_move')
        win_moves = self._winning_moves(game)

        def styled(line_idx, c0, c1, style):
            ranges.append((line_idx, c0, c1, style))

        def cell_style(move, line, col, mark):
            """Apply mark / reason / last / cursor / win styles to one cell."""
            if mark in (X, O):
                styled(line, col + 1, col + 2,
                       f'bold {X_COLOR}' if mark == X else f'bold {O_COLOR}')
            if move == reason:
                styled(line, col, col + 3, 'on #134E4A')
            if move == last:
                styled(line, col + 1, col + 2, f'bold {FLASH_COLOR}')
            if move == self.cursor:
                styled(line, col, col + 3, 'reverse')
            if move in win_moves:
                styled(line, col, col + 3, f'on #1E3D2F bold {GREEN}')

        if isinstance(game, UltimateGame):
            boxes = [self._box_lines() for _ in range(9)]
            active = game.active_macro
            whole = micro_win_line(game.macro)
            for br in range(3):
                for ln in range(7):
                    parts = []
                    for bc in range(3):
                        m = br * 3 + bc
                        chars = list(boxes[m][ln])
                        for rel_l, rel_c in self._box_cells():
                            lr, lc = rel_l // 2, rel_c // 4
                            mark = game.micro[m][lr * 3 + lc]
                            chars[rel_c + 1] = mark if mark in (X, O) else ' '
                        parts.append(''.join(chars))
                    lines.append('   '.join(parts))
                if br < 2:
                    lines.append('')
            for m in range(9):
                br, bc = divmod(m, 3)
                win = whole is not None and m in whole
                for rel_l, rel_c in self._box_cells():
                    lr, lc = rel_l // 2, rel_c // 4
                    i = lr * 3 + lc
                    line = br * 8 + rel_l
                    col = bc * 16 + rel_c
                    cells[(line, col)] = (m, i)
                    cell_style((m, i), line, col, game.micro[m][i])
                # chunk frame emphasis
                c0, c1 = bc * 16, bc * 16 + 13
                for rel_l in range(7):
                    line = br * 8 + rel_l
                    if win:
                        styled(line, c0, c1, f'on #1E3D2F bold {GREEN}')
                    elif m == active and game.macro_open(m):
                        if rel_l in (0, 6):
                            styled(line, c0, c1, f'bold {ACTIVE_FRAME}')
                        else:
                            styled(line, c0, c0 + 1, f'bold {ACTIVE_FRAME}')
                            styled(line, c1 - 1, c1, f'bold {ACTIVE_FRAME}')
        else:
            chars = [list(ln) for ln in self._box_lines()]
            for i, mark in enumerate(game.board):
                r, c = divmod(i, 3)
                chars[1 + 2 * r][2 + 4 * c] = mark if mark in (X, O) else ' '
            for ln in range(7):
                lines.append(''.join(chars[ln]))
            for i in range(9):
                r, c = divmod(i, 3)
                line, col = 1 + 2 * r, 1 + 4 * c
                cells[(line, col)] = i
                cell_style(i, line, col, game.board[i])

        self._lines = lines
        self._cells = cells
        self._ranges = ranges

    @staticmethod
    def _winning_moves(game):
        if isinstance(game, NormalGame):
            line = micro_win_line(game.board)
        else:
            line = micro_win_line(game.macro)
        return set(line) if line else set()

    def render(self):
        self._build()
        if not self._lines:
            return Text('')
        txt = Text()
        for i, line in enumerate(self._lines):
            if i:
                txt.append('\n')
            txt.append(line, style=SUB)
        for line_idx, c0, c1, style in self._ranges:
            start = sum(len(l) + 1 for l in self._lines[:line_idx]) + c0
            end = start + (c1 - c0)
            txt.stylize(style, start, min(end, len(txt)))
        return txt

    # -- input ------------------------------------------------------------

    def _move_cursor(self, delta):
        if not self._legal:
            return
        try:
            idx = self._legal.index(self.cursor)
        except (ValueError, TypeError):
            idx = 0
        idx = max(0, min(len(self._legal) - 1, idx + delta))
        if idx < len(self._legal):
            self.cursor = self._legal[idx]
            self.refresh()

    def action_cursor_left(self):
        self._move_cursor(-1)

    def action_cursor_right(self):
        self._move_cursor(1)

    def action_cursor_up(self):
        self._move_cursor(-3)

    def action_cursor_down(self):
        self._move_cursor(3)

    def action_play(self):
        if self.on_board_click and self.cursor is not None:
            self.on_board_click(self.cursor)

    async def on_click(self, event):
        if not self._cells:
            return
        move = None
        for (line, col), m in self._cells.items():
            if line == event.y and col <= event.x < col + 3:
                move = m
                break
        if move is None:
            return
        if move in self._legal:
            self.cursor = move
            self.refresh()
        if self.on_board_click:
            self.on_board_click(move)

    BINDINGS = [
        Binding('h', 'cursor_left', 'cursor left', show=False),
        Binding('j', 'cursor_down', 'cursor down', show=False),
        Binding('k', 'cursor_up', 'cursor up', show=False),
        Binding('l', 'cursor_right', 'cursor right', show=False),
        Binding('left', 'cursor_left', 'cursor left', show=False),
        Binding('down', 'cursor_down', 'cursor down', show=False),
        Binding('up', 'cursor_up', 'cursor up', show=False),
        Binding('right', 'cursor_right', 'cursor right', show=False),
        Binding('space', 'play', 'play', show=False),
    ]


# ============================================================
# Shared CSS
# ============================================================

COMMON_CSS = """
Screen {
    background: #17141B;
    color: #E6E0E9;
}

Static#app-title {
    text-style: bold;
    text-align: center;
    color: #E8DEF8;
    margin: 1 0 0 0;
}

Static#app-subtitle {
    text-align: center;
    color: #CAC4D0;
    margin: 0 0 1 0;
}

Static#panel-title {
    text-style: bold;
    color: #E8DEF8;
}

Static#muted, Static#help {
    color: #8A8A96;
}

VerticalScroll#menu-form {
    width: 1fr;
    max-width: 72;
    height: 1fr;
    padding: 0 2;
    border: none;
    background: transparent;
}

#menu-form .field { margin-bottom: 1; }

#menu-form Slider { width: 1fr; }

#menu-form Select { width: 1fr; }

#menu-form Horizontal { width: 1fr; }

Horizontal#ai-row {
    width: 1fr;
    align-vertical: middle;
}

#ai-row > Static { width: auto; }

Select { width: 1fr; }

Button {
    border: round #6750A4;
}

Button:focus {
    border: round #9E86E8;
}

RadioSet {
    height: auto;
}

/* ---- game screen ------------------------------------------------- */

Horizontal#split {
    width: 1fr;
    height: 1fr;
    layout: vertical;
}

Vertical#board-col {
    width: 1fr;
    height: 1fr;
    padding: 0 1;
}

Vertical#side-col {
    width: 1fr;
    height: 1fr;
    padding: 0 1;
}

Screen.-wide Horizontal#split {
    layout: horizontal;
}

.-wide Vertical#side-col {
    width: 46;
}

Vertical#side-col > Vertical {
    border: round #3B3644;
    padding: 0 1;
    margin: 0 0 1 0;
}

Vertical#side-col > Vertical > Static {
    text-wrap: nowrap;
    overflow-x: hidden;
}

#side-col #win-bar { margin: 0 0 1 0; }

#side-col Slider { width: 1fr; }

#side-col Horizontal { width: 1fr; }

#side-col Switch { margin: 0; }

Horizontal#topbar {
    height: 3;
    width: 1fr;
    margin: 0 0 1 0;
}

#topbar Static#game-title {
    text-style: bold;
    color: #E8DEF8;
    width: 1fr;
    min-width: 0;
    overflow-x: hidden;
}

Horizontal#ctrlbar {
    margin: 0 1 1 0;
    height: 3;
}

#hist-label { margin: 0 1; }

Static#status { margin: 0 1; height: 1; }

Static#help { margin: 0 1; height: 1; }

#cvc-card { display: none; }
#cvc-card.cvc-mode { display: block; }

#btn-again { display: none; }
#btn-again.show { display: block; }
"""


# ============================================================
# Menu screen
# ============================================================

class MenuScreen(Screen):
    TYPE_VALUES = ('normal', 'ultimate')
    MODE_VALUES = ('pvp', 'pvc', 'cvc')
    FIRST_VALUES = ('human', 'computer')

    def compose(self) -> ComposeResult:
        yield Static('Ultimate Tic Tac Toe — Basically Awful (SBA)', id='app-title')
        yield Static(t('終極井字棋 — Basically Awful', 'Basically Awful — 終極井字棋'),
                     id='app-subtitle')
        with VerticalScroll(id='menu-form'):
            yield Static(t('Game Setup', '遊戲設定'), id='panel-title')
            with Vertical(classes='field'):
                yield Static(t('Game Type', '遊戲類型'), classes='muted')
                yield RadioSet(
                    RadioButton('Normal Tic Tac Toe (普通井字棋)', value=True),
                    RadioButton('Ultimate Tic Tac Toe (終極井字棋)'),
                    id='rs-type')
            with Vertical(classes='field'):
                yield Static(t('Mode', '模式'), classes='muted')
                yield RadioSet(
                    RadioButton(t('PvP', '玩家對玩家'), value=True),
                    RadioButton(t('Player vs Computer', '玩家對電腦')),
                    RadioButton(t('Computer vs Computer', '電腦對電腦')),
                    id='rs-mode')
            with Vertical(id='first-row', classes='field'):
                yield Static(t('First Player', '先手'), id='lbl-first', classes='muted')
                yield RadioSet(
                    RadioButton(t('You move first — X', '你先手 — X'), value=True),
                    RadioButton(t('Computer moves first — O', '電腦先手 — O')),
                    id='rs-first')
            with Vertical(id='ai-x-row', classes='field'):
                yield Static(t('Player X — AI Level', '玩家 X — AI 等級'), classes='muted')
                yield Select([(label, key) for key, label in AI_OPTIONS.items()],
                             value='Minimax', allow_blank=False, id='sel-ai-x')
            with Vertical(id='ai-o-row', classes='field'):
                yield Static(t('Player O — AI Level', '玩家 O — AI 等級'), id='lbl-ai-o',
                             classes='muted')
                yield Select([(label, key) for key, label in AI_OPTIONS.items()],
                             value='MCTS', allow_blank=False, id='sel-ai-o')
            with Vertical(classes='field'):
                yield Static(t('MCTS Strength', 'MCTS 強度'), classes='muted')
                with Horizontal():
                    yield Slider(cfg_session('mcts_budget', 800), 200, 3000, 100,
                                 id='sl-mcts')
                    yield Static(str(cfg_session('mcts_budget', 800)), id='val-mcts',
                                 classes='muted')
            with Vertical(classes='field'):
                yield Static(t('Minimax Depth (Ultimate)', 'Minimax 深度（終極模式）'),
                             classes='muted')
                with Horizontal():
                    yield Slider(cfg_session('minimax_depth', 4), 2, 6, 1, id='sl-mm')
                    yield Static(str(cfg_session('minimax_depth', 4)), id='val-mm',
                                 classes='muted')
            with Vertical(classes='field'):
                yield Static(t('AI Assistant', 'AI 助手'), classes='muted')
                yield Switch(True, id='sw-assistant')
            with Vertical(classes='field'):
                yield Static(t('CvC Speed', '電腦對戰速度'), classes='muted')
                with Horizontal():
                    yield Slider(4, 1, 10, 1, id='sl-speed')
                    yield Static('0.4s', id='val-speed', classes='muted')
            with Vertical(classes='field'):
                yield Static(t('Auto Play', '自動進行'), classes='muted')
                yield Switch(True, id='sw-cvc-auto')
            yield Button(t('Start Game', '開始遊戲'), id='btn-start')

    def on_mount(self):
        self._update_visibility()
        self.query_one('#btn-start', Button).label = t('Start Game', '開始遊戲')

    # -- helpers ----------------------------------------------------------

    def _radio_value(self, selector, values):
        rs = self.query_one(selector, RadioSet)
        idx = rs.pressed_index
        return values[idx] if idx is not None else values[0]

    def _ai_options(self):
        game_type = self._radio_value('#rs-type', self.TYPE_VALUES)
        if game_type == 'ultimate':
            return dict(AI_OPTIONS)
        return {k: v for k, v in AI_OPTIONS.items() if k != 'AlphaZero'}

    def _update_visibility(self):
        mode = self._radio_value('#rs-mode', self.MODE_VALUES)
        self.query_one('#rs-first').display = mode == 'pvc'
        self.query_one('#lbl-first').display = mode == 'pvc'
        self.query_one('#ai-x-row').display = mode == 'cvc'
        self.query_one('#ai-o-row').display = mode in ('pvc', 'cvc')
        if mode == 'pvc':
            label = (t('Computer (X) — AI Level', '電腦 (X) — AI 等級')
                     if self._radio_value('#rs-first', self.FIRST_VALUES) == 'computer'
                     else t('Computer (O) — AI Level', '電腦 (O) — AI 等級'))
        else:
            label = t('Player O — AI Level', '玩家 O — AI 等級')
        self.query_one('#lbl-ai-o', Static).update(label)
        self._update_ai_options()

    def _update_ai_options(self):
        opts = self._ai_options()
        for sel_id, fallback in (('#sel-ai-x', 'Minimax'), ('#sel-ai-o', 'MCTS')):
            sel = self.query_one(sel_id, Select)
            cur = sel.value if sel.value in opts else fallback
            sel.set_options([(label, key) for key, label in opts.items()])
            sel.value = cur

    # -- messages ---------------------------------------------------------

    def on_radio_set_changed(self, event):
        if event.radio_set.id in ('rs-type', 'rs-mode', 'rs-first'):
            self._update_visibility()

    def on_slider_changed(self, event):
        sid = event.slider.id
        label = {
            'sl-mcts': ('val-mcts', str(event.value)),
            'sl-mm': ('val-mm', str(event.value)),
            'sl-speed': ('val-speed', f'{event.value / 10:.1f}s'),
        }.get(sid)
        if label:
            self.query_one(f'#{label[0]}', Static).update(label[1])

    def on_button_pressed(self, event):
        if event.button.id == 'btn-start':
            self._start_game()

    def _start_game(self):
        s = new_session()
        s['game_type'] = self._radio_value('#rs-type', self.TYPE_VALUES)
        s['mode'] = self._radio_value('#rs-mode', self.MODE_VALUES)
        s['first_player'] = self._radio_value('#rs-first', self.FIRST_VALUES)
        s['ai_x'] = self.query_one('#sel-ai-x', Select).value or 'Minimax'
        s['ai_o'] = self.query_one('#sel-ai-o', Select).value or 'MCTS'
        s['mcts'] = self.query_one('#sl-mcts', Slider).value
        s['minimax_depth'] = self.query_one('#sl-mm', Slider).value
        s['cvc_speed'] = self.query_one('#sl-speed', Slider).value / 10.0
        s['cvc_auto'] = self.query_one('#sw-cvc-auto', Switch).value
        s['assistant_enabled'] = self.query_one('#sw-assistant', Switch).value
        log.info('Game started type=%s mode=%s X=%s O=%s mcts=%d depth=%d',
                 s['game_type'], s['mode'], s['ai_x'], s['ai_o'],
                 s['mcts'], s['minimax_depth'])
        self.app.push_screen(GameScreen(s))


# ============================================================
# Game screen
# ============================================================

class GameScreen(Screen):
    HORIZONTAL_BREAKPOINTS = [(0, '-narrow'), (96, '-wide')]
    BINDINGS = [
        Binding('b', 'revert', t('Revert', '悔棋')),
        Binding('p', 'pause', t('Pause', '暫停')),
        Binding('[', 'hist(-1)', t('Back', '後退')),
        Binding(']', 'hist(1)', t('Forward', '前進')),
        Binding('left', 'hist(-1)', t('Back', '後退'), show=False),
        Binding('right', 'hist(1)', t('Forward', '前進'), show=False),
    ] + [
        Binding(str(n), f'best_move({n})', f'#{n}', show=False)
        for n in range(1, 10)
    ]

    def __init__(self, session):
        super().__init__()
        self.session = session
        self._ai_worker = None
        self._ai_token = 0
        self._ai_busy = False
        self._analysis_token = 0
        self._analysis_items = []
        self._last_rates = None
        self._cvc_timer = None
        self._last_move = 0.0
        self._closed = False

    # -- compose ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id='topbar'):
            yield Static('Ultimate Tic Tac Toe — 終極井字棋', id='game-title')
            yield Button(t('New', '新'), id='btn-new')
            yield Button(t('Again', '再玩'), id='btn-again')
            yield Button(t('Menu', '選單'), id='btn-back')
        with Horizontal(id='split'):
            with Vertical(id='board-col'):
                with Horizontal(id='ctrlbar'):
                    yield Button(t('Revert', '悔棋'), id='btn-revert')
                    yield Button(t('Pause', '暫停'), id='btn-pause')
                    yield Button(t('Next', '前進'), id='btn-forward')
                    yield Static('0 / 0', id='hist-label', classes='muted', markup=False)
                yield Static('', id='status', markup=False)
                yield Board(id='board')
                yield Static('', id='help', markup=False)
            with Vertical(id='side-col'):
                with Vertical(id='info-card'):
                    yield Static(t('Game Info', '遊戲資訊'), id='panel-title')
                    yield Static('', id='info-line', classes='muted', markup=False)
                    yield Static('', id='info-x', markup=False)
                    yield Static('', id='info-o', markup=False)
                    with Horizontal(classes='ai-row'):
                        yield Static(t('AI Assistant', 'AI 助手'), classes='muted')
                        yield Switch(True, id='sw-assistant')
                with Vertical(id='assistant-card'):
                    yield Static(t('Best Moves / 最佳棋步', '最佳棋步'), id='panel-title')
                    yield Static('', id='an-status', classes='muted', markup=False)
                    yield Static('', id='win-bar', markup=False)
                    yield Static('', id='best-moves', markup=False)
                    yield Static(t('Click a move or press its number to highlight',
                                   '點擊棋步或按編號可在棋盤上標示'),
                                 id='an-hint', classes='muted', markup=False)
                with Vertical(id='history-card'):
                    yield Static(t('History — 棋譜', '棋譜'), id='panel-title')
                    yield Static('', id='history', markup=False)
                with Vertical(id='cvc-card'):
                    yield Static(t('CvC Controls', '電腦對戰控制'), id='panel-title')
                    yield Static(t('Speed', '速度'), classes='muted')
                    with Horizontal():
                        yield Slider(4, 1, 10, 1, id='sl-speed')
                        yield Static('0.4s', id='val-speed', classes='muted', markup=False)
                    yield Static(t('Auto Play', '自動進行'), classes='muted')
                    yield Switch(True, id='sw-cvc-auto')

    def on_mount(self):
        self.board = self.query_one('#board', Board)
        self.board.on_board_click = self._human_move
        self.new_game()
        self._cvc_timer = self.set_interval(0.05, self._cvc_tick)
        self.query_one('#board', Board).focus()

    def on_unmount(self):
        self._closed = True
        if self._cvc_timer is not None:
            self._cvc_timer.stop()
        try:
            self.app.workers.cancel_group(self, 'ai')
            self.app.workers.cancel_group(self, 'analysis')
        except Exception:  # noqa: BLE001 - worker cleanup is best-effort
            pass

    # -- game setup -------------------------------------------------------

    def _game(self):
        return self.session['game']

    def new_game(self):
        s = self.session
        s['game_id'] += 1
        s['game'] = (NormalGame() if s['game_type'] == 'normal' else UltimateGame())
        s['moves'] = []
        s['history'] = [_history_point(s['game'])]
        s['step'] = 0
        s['cvc_paused'] = False
        s['analysis_gen'] += 1
        s['analyzing'] = False
        s['reanalyze'] = False
        s['reason_move'] = None
        self._ai_token += 1
        self._analysis_token += 1
        self._ai_busy = False
        self._analysis_items = []
        self._last_move = time.monotonic()
        self._render_analysis_gate()
        self._position_changed()

    def _position_changed(self):
        """Refresh everything after the live position changed."""
        s = self.session
        self.board.set_session(s)
        self._render_status()
        self._render_history()
        self._update_controls()
        self.trigger_analysis()
        if self._game().is_over():
            self._show_result()
            return
        if is_ai_turn(s):
            if s['mode'] == 'pvc':
                self._step_ai()
        self._render_help()

    # -- moves ------------------------------------------------------------

    def _human_move(self, move):
        s = self.session
        game = self._game()
        if game.is_over():
            self.notify(t('The game is over', '遊戲已經結束'), timeout=2)
            return
        if self._ai_busy:
            self.notify(t('AI is thinking — please wait', 'AI 思考中，請稍候'), timeout=2)
            return
        if is_ai_turn(s):
            self.notify(t('It is not your turn', '現在不是你的回合'), timeout=2)
            return
        if move not in game.legal_moves():
            self.notify(t('Illegal move', '不合法的棋步'), timeout=2)
            return
        self._apply_move(move)
        self.query_one('#board', Board).focus()

    def _apply_move(self, move):
        s = self.session
        game = self._game()
        if move not in game.legal_moves():
            return False
        side = game.current
        apply_move(game, move)
        log.info('Move: %s -> %s', side, move_text(move))
        moves = s['moves']
        if s['step'] < len(moves):    # rewound: branch the history here
            del moves[s['step']:]
            del s['history'][s['step'] + 1:]
        moves.append(move)
        s['history'].append(_history_point(game))
        s['step'] = len(moves)
        s['analysis_gen'] += 1
        s['reason_move'] = None
        self._last_move = time.monotonic()
        self._position_changed()
        return True

    # -- AI workers -------------------------------------------------------

    def _step_ai(self):
        s = self.session
        game = s['game']
        if game is None or game.is_over() or self._ai_busy or not is_ai_turn(s):
            return False
        x_type, o_type = side_types(s)
        ai_type = x_type if game.current == X else o_type
        snapshot = game.clone()
        gen = s['analysis_gen']
        gid = s['game_id']
        tok = self._ai_token
        self._ai_busy = True
        self._render_status()
        self._ai_worker = self._ai_job(snapshot, ai_type, gen, gid, tok)
        return True

    @work(thread=True, group='ai')
    def _ai_job(self, game, ai_type, gen, gid, tok):
        try:
            move = get_ai_move(game, ai_type, self.session['mcts'],
                               self.session.get('minimax_depth', 4))
        except Exception as e:  # noqa: BLE001 - report engine failures to the UI
            log.error('AI move failed: %s', e)
            self.app.call_from_thread(self._ai_failed, gen, gid, tok)
            return
        self.app.call_from_thread(self._ai_done, move, gen, gid, tok)

    def _ai_done(self, move, gen, gid, tok):
        if self._closed or tok != self._ai_token:
            return  # paused / rewound / restarted while thinking: drop it
        if gid != self.session['game_id'] or gen != self.session['analysis_gen']:
            self._ai_busy = False
            self._render_status()
            return
        self._ai_busy = False
        self._apply_move(move)

    def _ai_failed(self, gen, gid, tok):
        if self._closed or tok != self._ai_token:
            return
        self._ai_busy = False
        self._render_status()
        self.notify(t('AI error — see the log', 'AI 發生錯誤，請看日誌'),
                    severity='error', timeout=4)

    # -- CvC --------------------------------------------------------------

    def _cvc_tick(self):
        s = self.session
        if s['mode'] != 'cvc' or s.get('cvc_paused', False) \
                or not s.get('cvc_auto', True):
            return
        game = s['game']
        if game is None or game.is_over() or self._ai_busy or not is_ai_turn(s):
            return
        if time.monotonic() - self._last_move >= s.get('cvc_speed', 0.4):
            self._step_ai()

    # -- history ----------------------------------------------------------

    def go_to_step(self, k):
        s = self.session
        moves = s['moves']
        k = max(0, min(k, len(moves)))
        if k == s['step']:
            return
        # drop any in-flight AI move: the position is changing underneath it
        self._ai_token += 1
        self._ai_busy = False
        if self._ai_worker is not None:
            try:
                self._ai_worker.cancel()
            except Exception:  # noqa: BLE001
                pass
        g = NormalGame() if s['game_type'] == 'normal' else UltimateGame()
        for mv in moves[:k]:
            apply_move(g, mv)
        s['game'] = g
        s['step'] = k
        s['cvc_paused'] = s['mode'] == 'cvc' and k < len(moves)
        s['analysis_gen'] += 1
        self._position_changed()

    def action_revert(self):
        s = self.session
        if s['step'] <= 0:
            self.notify(t('Nothing to revert', '沒有可以悔的棋'), timeout=2)
            return
        self.go_to_step(s['step'] - 1)

    def action_hist(self, delta):
        s = self.session
        target = s['step'] + delta
        if target < 0:
            self.notify(t('Already at the start', '已經在最前面了'), timeout=2)
            return
        if target > len(s['moves']):
            if s['mode'] == 'cvc' and not self._game().is_over():
                if not self._step_ai():
                    self.notify(t('AI is busy', 'AI 正在思考'), timeout=2)
            else:
                self.notify(t('End of the history', '已經到最後了'), timeout=2)
            return
        self.go_to_step(target)

    def action_pause(self):
        s = self.session
        if s['mode'] != 'cvc':
            self.notify(t('Pause only works in Computer vs Computer',
                          '暫停僅用於電腦對戰'), timeout=2)
            return
        s['cvc_paused'] = not s.get('cvc_paused', False)
        if s['cvc_paused']:
            self._ai_token += 1
            self._ai_busy = False
            if self._ai_worker is not None:
                try:
                    self._ai_worker.cancel()
                except Exception:  # noqa: BLE001
                    pass
        self._update_controls()
        self._render_status()

    # -- assistant --------------------------------------------------------

    def _render_analysis_gate(self):
        s = self.session
        if not s.get('assistant_enabled', True):
            self.query_one('#an-status', Static).update(
                t('Assistant disabled', '助手已關閉'))
            return
        if s.get('analyzing'):
            self.query_one('#an-status', Static).update(
                t('Analyzing...', '分析中...'))
            self.query_one('#win-bar', Static).update('')
            self.query_one('#best-moves', Static).update('')

    def trigger_analysis(self):
        s = self.session
        if not s.get('assistant_enabled', True):
            return
        game = s['game']
        if game is None or game.is_over():
            return
        if s.get('analyzing'):
            s['reanalyze'] = True
            return
        s['analyzing'] = True
        self._analysis_token += 1
        tok = self._analysis_token
        snapshot = game.clone()
        budget = s['mcts'] if isinstance(snapshot, UltimateGame) else 0
        gen = s['analysis_gen']
        step = s['step']
        gid = s['game_id']
        self._render_analysis_gate()
        self._analysis_job(snapshot, budget, gen, step, gid, tok)

    @work(thread=True, group='analysis')
    def _analysis_job(self, game, budget, gen, step, gid, tok):
        try:
            items, rates = analyze_position(game, budget)
        except Exception as e:  # noqa: BLE001
            log.error('Analysis failed: %s', e)
            items, rates = [], (0.5, 0.0, 0.5)
        self.app.call_from_thread(self._analysis_done, items, rates, gen,
                                  step, gid, tok)

    def _analysis_done(self, items, rates, gen, step, gid, tok):
        if self._closed or tok != self._analysis_token:
            return
        s = self.session
        s['analyzing'] = False
        history = s.get('history', [])
        if gid == s['game_id'] and gen == s['analysis_gen'] \
                and 0 <= step < len(history) and history[step] is None:
            history[step] = tuple(rates)
        if gen == s['analysis_gen'] and s['game'] is not None \
                and s.get('assistant_enabled', True):
            self._render_analysis(items, rates)
        if s.get('reanalyze'):
            s['reanalyze'] = False
            self.trigger_analysis()
        else:
            self._render_analysis_gate()

    def _render_analysis(self, items, rates):
        s = self.session
        self._analysis_items = items
        self._last_rates = rates
        x, d, o = rates
        bar = Text.assemble(
            ('✕ ' + _bar(x), f'bold {X_COLOR}'),
            (f' {x:.0%}', ''),
            ('  ' + t('draw', '和') + ' ' + _bar(d), f'dim {SUB}'),
            (f' {d:.0%}', ''),
            ('  ○ ' + _bar(o), f'bold {O_COLOR}'),
            (f' {o:.0%}', ''),
        )
        self.query_one('#win-bar', Static).update(bar)
        lines = Text()
        if not items:
            lines.append(t('No moves to analyze', '沒有可分析的棋步'))
        else:
            for n, it in enumerate(items[:5], 1):
                pct = max(0.0, min(1.0, it['pct']))
                if isinstance(self._game(), NormalGame):
                    verdict = ('Win' if pct == 1.0
                               else ('Draw' if pct == 0.5 else 'Loss'))
                else:
                    verdict = f'{pct:.0%}'
                mark = '▸' if it['move'] == s.get('reason_move') else ' '
                lines.append(f'{mark}{n}  {move_text(it["move"]):<9}'
                             f' {verdict:>7}  {_reason_label(it["reason"])}\n')
        self.query_one('#best-moves', Static).update(lines)
        self.query_one('#an-status', Static).update('')

    def on_switch_changed(self, event):
        if event.switch.id == 'sw-assistant':
            s = self.session
            s['assistant_enabled'] = event.value
            if event.value:
                self.trigger_analysis()
            else:
                s['reanalyze'] = False
                self._analysis_items = []
                s['reason_move'] = None
                self.query_one('#best-moves', Static).update('')
                self.query_one('#win-bar', Static).update('')
                self._render_analysis_gate()
                self.board.refresh()
        elif event.switch.id == 'sw-cvc-auto':
            self.session['cvc_auto'] = event.value
            self._render_status()

    def on_slider_changed(self, event):
        if event.slider.id == 'sl-speed':
            s = self.session
            s['cvc_speed'] = event.value / 10.0
            self.query_one('#val-speed', Static).update(f'{event.value / 10:.1f}s')
            self._render_status()

    def action_best_move(self, n):
        s = self.session
        items = self._analysis_items
        if not items or n > len(items):
            self.notify(t(f'No best move #{n}', f'沒有第 {n} 個最佳棋步'),
                        timeout=2)
            return
        s['reason_move'] = items[n - 1]['move']
        if self._last_rates is not None:
            self._render_analysis(items, self._last_rates)
        self.board.refresh()

    # -- rendering --------------------------------------------------------

    def _render_status(self):
        s = self.session
        game = s['game']
        if game is None:
            return
        txt = Text()
        result = game.result()
        if result in (X, O):
            color = X_COLOR if result == X else O_COLOR
            txt.append(f' {result} ', style=f'bold {color} on #2A2740')
            txt.append(t(f'Player {result} wins!', f'玩家 {result} 獲勝！'),
                       style=f'bold {color}')
        elif result == 'D':
            txt.append(' — ', style='bold')
            txt.append(t("It's a draw!", '平局！'), style='bold')
        else:
            player = game.current
            color = X_COLOR if player == X else O_COLOR
            txt.append(f' {player} ', style=f'bold {color} on #2A2740')
            if current_side_type(s) != 'Human':
                kind = current_side_type(s)
                txt.append(f'Computer ({kind}) — 電腦 ({kind})', style='bold')
                txt.append(' · ' + (t('thinking...', '思考中') if self._ai_busy
                                    else t('AI turn', 'AI 回合')))
            else:
                txt.append(f'Player {player} — 玩家 {player}', style='bold')
                txt.append(' · ' + t('your move', '輪到你'))
            txt.append(f' · {t("step", "步")} {s["step"]}/{len(s["moves"])}')
            if isinstance(game, UltimateGame) and game.active_macro is not None \
                    and game.macro_open(game.active_macro):
                txt.append(f' · B{game.active_macro + 1}')
            if s['mode'] == 'cvc' and s.get('cvc_paused', False):
                txt.append(' · ' + t('PAUSED', '已暫停'), style='bold')
        self.query_one('#status', Static).update(txt)
        self.query_one('#info-line', Static).update(
            f'{("Ultimate" if isinstance(game, UltimateGame) else "Normal")}'
            f' · {s["mode"].upper()}'
            f' · {t("MCTS", "MCTS")} {s["mcts"]} · '
            f'{t("depth", "深度")} {s.get("minimax_depth", 4)}')
        x_type, o_type = side_types(s)
        self.query_one('#info-x', Static).update(
            Text(f'✕ ', style=f'bold {X_COLOR}') + side_label(x_type))
        self.query_one('#info-o', Static).update(
            Text(f'○ ', style=f'bold {O_COLOR}') + side_label(o_type))

    def _render_history(self):
        s = self.session
        moves = s['moves']
        step = s['step']
        self.query_one('#hist-label', Static).update(f'{step} / {len(moves)}')
        lines = Text()
        lines.append(t('0  — start', '0  — 開局') + '\n')
        for k, mv in enumerate(moves, 1):
            mark = '▸' if k == step else ' '
            side = X if k % 2 == 1 else O
            color = X_COLOR if side == X else O_COLOR
            lines.append(f'{k:>2}{mark} {side} ')
            lines.append(Text(move_text(mv), style=f'bold {color}'))
            if k < len(moves):
                lines.append('\n')
        self.query_one('#history', Static).update(lines)

    def _render_help(self):
        s = self.session
        extra = (' · p ' + t('pause/resume', '暫停/繼續')) if s['mode'] == 'cvc' else ''
        help_text = (f'b ' + t('revert', '悔棋')
                     + extra
                     + ' · 1-9 ' + t('best move', '最佳步')
                     + ' · ←/→[/] ' + t('history', '歷史')
                     + ' · hjkl' + t(' /arrows', '／方向鍵') + ' ' + t('cursor', '游標')
                     + ' · Space ' + t('play', '落子')
                     + ' · ' + t('mouse click', '滑鼠點擊'))
        self.query_one('#help', Static).update(help_text)

    def _update_controls(self):
        s = self.session
        game = s['game']
        cvc = s['mode'] == 'cvc'
        self.query_one('#cvc-card').set_class(cvc, 'cvc-mode')
        self.query_one('#btn-revert', Button).disabled = s['step'] == 0
        pause_btn = self.query_one('#btn-pause', Button)
        pause_btn.disabled = not cvc
        pause_btn.label = (t('Resume', '繼續') if s.get('cvc_paused', False)
                           else t('Pause', '暫停'))
        back = self.query_one('#btn-back', Button)
        back.label = t('Menu', '選單')
        again = self.query_one('#btn-again', Button)
        again.label = t('Again', '再玩')
        again.set_class(game is not None and game.is_over(), 'show')

    def _show_result(self):
        s = self.session
        result = self._game().result()
        log.info('Game over: %s', result)
        self._update_controls()

    # -- buttons ----------------------------------------------------------

    def on_button_pressed(self, event):
        bid = event.button.id
        if bid == 'btn-back':
            self.app.pop_screen()
        elif bid in ('btn-new', 'btn-again'):
            self.new_game()
        elif bid == 'btn-revert':
            self.action_revert()
        elif bid == 'btn-pause':
            self.action_pause()
        elif bid == 'btn-forward':
            self.action_hist(1)

    # -- clicks -----------------------------------------------------------

    def on_click(self, event):
        wid = event.widget.id
        if wid == 'history':
            self.go_to_step(event.y)
        elif wid == 'best-moves':
            items = self._analysis_items
            if items and 0 <= event.y < len(items):
                self.action_best_move(event.y + 1)


# ============================================================
# App entry
# ============================================================

class SBATUI(App):
    """Textual application hosting the menu and game screens."""
    TITLE = 'Ultimate Tic Tac Toe — 終極井字棋'
    CSS = COMMON_CSS

    def get_default_screen(self) -> Screen:
        return MenuScreen()


def main():
    import sys
    if '--debug' in sys.argv:
        log.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    SBATUI().run()


if __name__ == '__main__':
    main()
