# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: MIT

"""PySide6 desktop UI for SBA (Basically Awful).

Shares the game rules (game.py) and AI engines (ai.py) with the NiceGUI web
UI. The NiceGUI web server is opt-in: enable it from the menu switch or with
`python qtui.py --web`.

Run:
    python qtui.py            # desktop app (default)
    python qtui.py --web      # desktop app + start the NiceGUI web server
"""

import argparse
import logging
import os
import subprocess
import sys

try:
    from PySide6.QtCore import QRectF, Qt, QThread, QTimer, Signal
    from PySide6.QtGui import QColor, QPainter, QPen
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
        QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
        QSlider, QStackedWidget, QVBoxLayout, QWidget,
    )
except ImportError:
    print('PySide6 is not installed for this Python interpreter.')
    print('Install the desktop UI with:  python -m pip install -r requirements-qt.txt')
    print('Or start the web UI instead with:  python SBA.py --web')
    raise SystemExit(1)

from game import (
    X, O,
    NormalGame, UltimateGame, apply_move, micro_win_line,
)
from ai import get_ai_move, compute_analysis, move_text
from SBA import (
    new_session, side_types, current_side_type, is_ai_turn,
    log,
)


def t(en: str, zh: str) -> str:
    return f'{en} — {zh}'


# Material Design 3 (light) palette, matching static/styles.css.
X_COLOR = '#6750A4'
O_COLOR = '#B3261E'
CELL_EMPTY = '#F3EDF7'
CELL_FILLED = '#ECE6F0'
MACRO_ACTIVE = 'rgba(103, 80, 164, 0.10)'
MACRO_WON_X = 'rgba(103, 80, 164, 0.14)'
MACRO_WON_O = 'rgba(179, 38, 30, 0.10)'
GRID_LINE = '#CAC4D0'
MACRO_LINE = '#79747E'
FLASH = '#6750A4'

AI_OPTIONS = {
    'AlphaZero': 'AlphaZero — Neural MCTS（神經網路MCTS）',
    'Random': 'Random — 隨機',
    'Basic': 'Basic — 基礎',
    'Minimax': 'Minimax — 極小化極大',
    'Minimax Pro': 'Minimax Pro — 進階極小化極大（置換表加速）',
    'MCTS': 'MCTS — 蒙地卡羅',
    'MCTS+RAVE': 'MCTS+RAVE — 蒙地卡羅+RAVE',
}

# SiliconUI-inspired styling: soft lavender-tinted light theme, rounded
# "glass" cards, capsule buttons and smooth controls (kept consistent with
# the Material Design 3 palette used by the web UI).
QSS = '''
QWidget { background: #F5F1F8; color: #1D1B20; font-size: 14px; }
QMainWindow { background: #F5F1F8; }
QLabel#title { font-size: 20px; font-weight: 700; color: #21005D; }
QLabel#cardTitle { font-weight: 600; color: #21005D; }
QLabel#muted { color: #79747E; font-size: 12px; }
QFrame#card {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(121, 116, 126, 0.18);
    border-radius: 16px;
}
QFrame#sidePanel {
    background: rgba(255, 255, 255, 0.70);
    border-left: 1px solid rgba(121, 116, 126, 0.15);
}
QPushButton {
    background: #EADDFF; color: #21005D; border: none;
    border-radius: 14px; padding: 8px 18px; font-weight: 500;
}
QPushButton:hover { background: #D0BCFF; }
QPushButton:pressed { background: #C6B3F5; }
QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #7C6BB0, stop:1 #6750A4);
    color: #FFFFFF; font-weight: 600; padding: 10px 22px; border-radius: 16px;
}
QPushButton#primary:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #8B79BE, stop:1 #7C6BB0); }
QPushButton#primary:pressed { background: #5C4694; }
QPushButton:disabled { background: #E6E0E9; color: #9A969E; }
QComboBox, QSpinBox {
    background: rgba(243, 237, 247, 0.85);
    border: 1px solid #E6E0E9; border-radius: 10px; padding: 5px 10px; min-height: 22px;
}
QComboBox:hover, QSpinBox:hover { border-color: #CAC4D0; }
QComboBox::drop-down { border: none; width: 22px; }
QSlider::groove:horizontal { height: 6px; background: #E6E0E9; border-radius: 3px; }
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #B39DDB, stop:1 #6750A4); border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #FFFFFF; border: 2px solid #6750A4;
    width: 18px; height: 18px; margin: -8px 0; border-radius: 9px;
}
QSlider::handle:horizontal:hover { background: #EADDFF; }
QCheckBox::indicator {
    width: 18px; height: 18px; border-radius: 6px;
    border: 2px solid #79747E; background: #FFFFFF;
}
QCheckBox::indicator:hover { border-color: #6750A4; }
QCheckBox::indicator:checked { background: #6750A4; border-color: #6750A4; }
QRadioButton::indicator {
    width: 18px; height: 18px; border-radius: 9px;
    border: 2px solid #79747E; background: #FFFFFF;
}
QRadioButton::indicator:hover { border-color: #6750A4; }
QRadioButton::indicator:checked { background: #6750A4; border-color: #6750A4; }
QListWidget {
    background: rgba(243, 237, 247, 0.55);
    border: 1px solid #E6E0E9; border-radius: 12px; padding: 4px;
}
QListWidget::item { padding: 8px; border-radius: 8px; }
QListWidget::item:hover { background: rgba(208, 188, 255, 0.35); }
QListWidget::item:selected { background: #EADDFF; color: #21005D; }
QToolTip {
    background: #21005D; color: #FFFFFF; border: none;
    border-radius: 8px; padding: 6px 10px;
}
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #CAC4D0; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #B3A9BD; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QMessageBox { background: #F5F1F8; }
QMessageBox QPushButton { min-width: 90px; }
'''


def _mark_color(player):
    return QColor(X_COLOR if player == X else O_COLOR)


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------

class BoardWidget(QWidget):
    cell_clicked = Signal(object)

    MARGIN = 14.0
    GAP = 3.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.game = None
        self.legal = set()
        self.flash_cell = None
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)
        self.setMinimumSize(300, 300)

    def set_game(self, game):
        self.game = game
        self.legal = set(game.legal_moves())
        self.flash_cell = None
        self.update()

    def flash(self, move):
        self.flash_cell = move
        self.update()
        self._flash_timer.start(900)

    def _clear_flash(self):
        self.flash_cell = None
        self.update()

    def _grid(self):
        n = 9 if isinstance(self.game, UltimateGame) else 3
        cell = (min(self.width(), self.height()) - 2 * self.MARGIN - (n - 1) * self.GAP) / n
        size = 2 * self.MARGIN + n * cell + (n - 1) * self.GAP
        return n, cell, (self.width() - size) / 2, (self.height() - size) / 2

    def _cell_rect(self, row, col, cell, ox, oy):
        x = ox + col * (cell + self.GAP)
        y = oy + row * (cell + self.GAP)
        return QRectF(x, y, cell, cell)

    def _macro_rect(self, m, cell, ox, oy):
        """Return the QRectF of the whole 3x3 macro chunk for macro index m."""
        mr, mc = divmod(m, 3)
        x = ox + mc * 3 * (cell + self.GAP)
        y = oy + mr * 3 * (cell + self.GAP)
        size = cell * 3 + 2 * self.GAP
        return QRectF(x, y, size, size)

    def _move_at(self, px, py):
        n, cell, ox, oy = self._grid()
        col = int((px - ox) // (cell + self.GAP))
        row = int((py - oy) // (cell + self.GAP))
        if not (0 <= row < n and 0 <= col < n):
            return None
        if isinstance(self.game, NormalGame):
            return row * 3 + col
        m = (row // 3) * 3 + (col // 3)
        i = (row % 3) * 3 + (col % 3)
        return (m, i)

    def mousePressEvent(self, event):
        if self.game is None or self.game.is_over():
            return
        move = self._move_at(event.position().x(), event.position().y())
        if move is not None and move in self.legal:
            self.cell_clicked.emit(move)

    def paintEvent(self, event):
        if self.game is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if isinstance(self.game, UltimateGame):
            self._paint_ultimate(painter)
        else:
            self._paint_normal(painter)

    # -- shared helpers -----------------------------------------------------

    def _paint_cell(self, painter, rect, mark, is_flash):
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(CELL_FILLED if mark else CELL_EMPTY))
        painter.drawRoundedRect(rect, 4, 4)
        if mark in (X, O):
            self._paint_mark(painter, rect, mark)
        if is_flash:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(FLASH), 3))
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)

    def _paint_mark(self, painter, rect, player):
        pad = rect.width() * 0.22
        pen = QPen(_mark_color(player), max(2.0, rect.width() * 0.09))
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        if player == X:
            painter.drawLine(rect.left() + pad, rect.top() + pad,
                             rect.right() - pad, rect.bottom() - pad)
            painter.drawLine(rect.right() - pad, rect.top() + pad,
                             rect.left() + pad, rect.bottom() - pad)
        else:
            painter.drawEllipse(QRectF(rect.left() + pad, rect.top() + pad,
                                       rect.width() - 2 * pad, rect.height() - 2 * pad))

    def _line_endpoints(self, line, centers):
        pts = [centers[i] for i in line]
        xs = [p.x() for p in pts]
        ys = [p.y() for p in pts]
        x1, x2 = min(xs), max(xs)
        if x1 == x2:  # vertical line
            y1, y2 = min(ys), max(ys)
            return x1, y1, x1, y2
        i1 = xs.index(x1)
        i2 = xs.index(x2)
        return x1, pts[i1].y(), x2, pts[i2].y()

    def _paint_win_line(self, painter, line, centers, color, width):
        if line is None:
            return
        x1, y1, x2, y2 = self._line_endpoints(line, centers)
        pen = QPen(QColor(color), width)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(x1, y1, x2, y2)

    # -- Normal ------------------------------------------------------------

    def _paint_normal(self, painter):
        n, cell, ox, oy = self._grid()
        centers = []
        for i in range(9):
            r, c = divmod(i, 3)
            rect = self._cell_rect(r, c, cell, ox, oy)
            self._paint_cell(painter, rect, self.game.board[i],
                             self.flash_cell == i)
            centers.append(rect.center())
        painter.setPen(QPen(QColor(GRID_LINE), 1.5))
        for i in range(1, 3):
            x = ox + i * (cell + self.GAP) - self.GAP / 2
            y = oy + i * (cell + self.GAP) - self.GAP / 2
            painter.drawLine(x, oy, x, oy + 3 * cell + 2 * self.GAP)
            painter.drawLine(ox, y, ox + 3 * cell + 2 * self.GAP, y)
        line = micro_win_line(self.game.board)
        if line:
            winner = self.game.board[line[0]]
            self._paint_win_line(painter, line, centers, _mark_color(winner), cell * 0.12)

    # -- Ultimate ----------------------------------------------------------

    def _paint_ultimate(self, painter):
        n, cell, ox, oy = self._grid()
        # macro region highlights
        for m in range(9):
            rect = self._macro_rect(m, cell, ox, oy)
            if self.game.macro[m] == X:
                painter.setBrush(QColor(MACRO_WON_X))
            elif self.game.macro[m] == O:
                painter.setBrush(QColor(MACRO_WON_O))
            elif m == self.game.active_macro and self.game.macro_open(m):
                painter.setBrush(QColor(MACRO_ACTIVE))
            else:
                continue
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect, 10, 10)
        # micro cells
        macro_centers = []
        for m in range(9):
            mr, mc = divmod(m, 3)
            macro_rect = self._macro_rect(m, cell, ox, oy)
            macro_centers.append(macro_rect.center())
            for i in range(9):
                row = mr * 3 + i // 3
                col = mc * 3 + i % 3
                rect = self._cell_rect(row, col, cell, ox, oy)
                self._paint_cell(painter, rect, self.game.micro[m][i],
                                 self.flash_cell == (m, i))
        # grid lines
        painter.setPen(QPen(QColor(GRID_LINE), 1))
        for i in range(1, 9):
            x = ox + i * (cell + self.GAP) - self.GAP / 2
            y = oy + i * (cell + self.GAP) - self.GAP / 2
            painter.drawLine(x, oy, x, oy + 9 * cell + 8 * self.GAP)
            painter.drawLine(ox, y, ox + 9 * cell + 8 * self.GAP, y)
        painter.setPen(QPen(QColor(MACRO_LINE), 2))
        for i in (3, 6):
            x = ox + i * (cell + self.GAP) - self.GAP / 2
            y = oy + i * (cell + self.GAP) - self.GAP / 2
            painter.drawLine(x, oy, x, oy + 9 * cell + 8 * self.GAP)
            painter.drawLine(ox, y, ox + 9 * cell + 8 * self.GAP, y)
        # won macro badges
        for m in range(9):
            if self.game.macro[m] in (X, O):
                rect = self._macro_rect(m, cell, ox, oy)
                pad = rect.width() * 0.18
                badge = QRectF(rect.left() + pad, rect.top() + pad,
                               rect.width() - 2 * pad, rect.height() - 2 * pad)
                self._paint_mark(painter, badge, self.game.macro[m])
        # overall macro win line
        whole = micro_win_line(self.game.macro)
        if whole:
            winner = self.game.macro[whole[0]]
            self._paint_win_line(painter, whole, macro_centers, _mark_color(winner), cell * 0.16)

# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class AIWorker(QThread):
    move_ready = Signal(object)
    move_failed = Signal(str)

    def __init__(self, game, ai_type, budget, depth, parent=None):
        super().__init__(parent)
        self.game = game
        self.ai_type = ai_type
        self.budget = budget
        self.depth = depth

    def run(self):
        try:
            move = get_ai_move(self.game, self.ai_type, self.budget, self.depth)
            self.move_ready.emit(move)
        except Exception as e:  # noqa: BLE001 - report any engine failure to the UI
            self.move_failed.emit(str(e))


class AnalysisWorker(QThread):
    result_ready = Signal(object)

    def __init__(self, game, budget, parent=None):
        super().__init__(parent)
        self.game = game
        self.budget = budget

    def run(self):
        try:
            items = compute_analysis(self.game, self.budget)
            self.result_ready.emit(items)
        except Exception as e:  # noqa: BLE001
            log.error('Analysis failed: %s', e)
            self.result_ready.emit([])


# ---------------------------------------------------------------------------
# Menu page
# ---------------------------------------------------------------------------

class MenuPage(QWidget):
    start_requested = Signal(object)
    web_toggled = Signal(bool)

    def __init__(self, web_port=8080):
        super().__init__()
        self.web_port = web_port
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        title = QLabel('Ultimate Tic Tac Toe — 終極井字棋')
        title.setObjectName('title')
        root.addWidget(title)
        root.addWidget(QLabel(t('Game Setup', '遊戲設定')))

        def field_row(label_text, widget):
            row = QHBoxLayout()
            lab = QLabel(label_text)
            lab.setObjectName('muted')
            row.addWidget(lab)
            row.addStretch(1)
            row.addWidget(widget)
            return row

        self.game_type = QComboBox()
        self.game_type.addItem('Normal Tic Tac Toe (普通井字棋)', 'normal')
        self.game_type.addItem('Ultimate Tic Tac Toe (終極井字棋)', 'ultimate')
        root.addLayout(field_row(t('Game Type', '遊戲類型'), self.game_type))

        self.mode = QComboBox()
        self.mode.addItem('PvP (玩家對玩家)', 'pvp')
        self.mode.addItem('Player vs Computer (玩家對電腦)', 'pvc')
        self.mode.addItem('Computer vs Computer (電腦對電腦)', 'cvc')
        self.mode.currentIndexChanged.connect(lambda _: self._update_visibility())
        root.addLayout(field_row(t('Mode', '模式'), self.mode))

        self.first = QComboBox()
        self.first.addItem(t('You move first — X', '你先手 — X'), 'human')
        self.first.addItem(t('Computer moves first — O', '電腦先手 — O'), 'computer')
        self.first.currentIndexChanged.connect(lambda _: self._update_visibility())
        root.addLayout(field_row(t('First Player', '先手'), self.first))

        self.ai_x = QComboBox()
        for key, label in AI_OPTIONS.items():
            self.ai_x.addItem(label, key)
        root.addLayout(field_row(t('Player X — AI Level', '玩家 X — AI 等級'), self.ai_x))

        self.ai_o = QComboBox()
        for key, label in AI_OPTIONS.items():
            self.ai_o.addItem(label, key)
        self.ai_o_label = field_row(t('Player O — AI Level', '玩家 O — AI 等級'), self.ai_o)
        root.addLayout(self.ai_o_label)

        self.mcts = QSlider(Qt.Horizontal)
        self.mcts.setRange(200, 3000)
        self.mcts.setSingleStep(100)
        self.mcts.setValue(800)
        root.addLayout(field_row(t('MCTS Strength', 'MCTS 強度'),
                                 self._slider_row(self.mcts)))

        self.mm_depth = QSlider(Qt.Horizontal)
        self.mm_depth.setRange(2, 6)
        self.mm_depth.setValue(4)
        root.addLayout(field_row(t('Minimax Depth (Ultimate)', 'Minimax 深度（終極模式）'),
                                 self._slider_row(self.mm_depth)))

        self.assistant = QCheckBox(t('AI Assistant', 'AI 助手'))
        self.assistant.setChecked(True)
        root.addWidget(self.assistant)

        root.addSpacing(12)
        web_title = QLabel(t('NiceGUI Web UI (選用啟動)', 'NiceGUI Web 介面（選用）'))
        web_title.setObjectName('cardTitle')
        root.addWidget(web_title)
        self.web_switch = QCheckBox(t('Enable NiceGUI Web UI', '啟動 Web 介面'))
        self.web_switch.toggled.connect(self._on_web_toggled)
        root.addWidget(self.web_switch)
        self.web_status = QLabel(t('Web UI stopped', 'Web 介面已停止'))
        self.web_status.setObjectName('muted')
        root.addWidget(self.web_status)

        start_btn = QPushButton(t('Start Game', '開始遊戲'))
        start_btn.setObjectName('primary')
        start_btn.clicked.connect(self._on_start)
        root.addWidget(start_btn)
        root.addStretch(1)

        self._update_visibility()

    def _slider_row(self, slider):
        wrapper = QWidget()
        lay = QHBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(str(slider.value()))
        lab.setObjectName('muted')
        slider.valueChanged.connect(lambda v, l=lab: l.setText(str(v)))
        lay.addWidget(slider, 1)
        lay.addWidget(lab)
        return wrapper

    def _update_visibility(self):
        mode = self.mode.currentData()
        self.first.setVisible(mode == 'pvc')
        self.ai_x.setVisible(mode == 'cvc')
        self.ai_o.setVisible(mode in ('pvc', 'cvc'))
        if mode == 'pvc':
            label = (t('Computer (X) — AI Level', '電腦 (X) — AI 等級')
                     if self.first.currentData() == 'computer'
                     else t('Computer (O) — AI Level', '電腦 (O) — AI 等級'))
        else:
            label = t('Player O — AI Level', '玩家 O — AI 等級')
        item = self.ai_o_label.itemAt(0)
        if item and item.widget():
            item.widget().setText(label)

    def _on_web_toggled(self, checked):
        self.web_toggled.emit(checked)
        if checked:
            self.web_status.setText(f'http://127.0.0.1:{self.web_port} ({t("starting...", "啟動中")})')
        else:
            self.web_status.setText(t('Web UI stopped', 'Web 介面已停止'))

    def set_web_checked(self, checked):
        self.web_switch.setChecked(checked)

    def _on_start(self):
        s = new_session()
        s['game_type'] = self.game_type.currentData()
        s['mode'] = self.mode.currentData()
        s['first_player'] = self.first.currentData()
        s['ai_x'] = self.ai_x.currentData()
        s['ai_o'] = self.ai_o.currentData()
        s['mcts'] = self.mcts.value()
        s['minimax_depth'] = self.mm_depth.value()
        s['assistant_enabled'] = self.assistant.isChecked()
        self.start_requested.emit(s)

# ---------------------------------------------------------------------------
# Game page
# ---------------------------------------------------------------------------

class GamePage(QWidget):
    back_requested = Signal()

    def __init__(self):
        super().__init__()
        self.session = None
        self.game = None
        self.busy = False
        self.gen = 0
        self.analysis_busy = False
        self.analysis_pending = False
        self.workers = []
        self.cvc_timer = QTimer(self)
        self.cvc_timer.timeout.connect(self.cvc_tick)
        self.pvc_timer = QTimer(self)
        self.pvc_timer.setSingleShot(True)
        self.pvc_timer.timeout.connect(self.run_ai)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        top_title = QLabel('Ultimate Tic Tac Toe — 終極井字棋')
        top_title.setObjectName('title')
        top.addWidget(top_title)
        top.addStretch(1)
        back_btn = QPushButton(t('Back to Menu', '返回選單'))
        back_btn.clicked.connect(self.back_requested.emit)
        top.addWidget(back_btn)
        root.addLayout(top)

        body = QHBoxLayout()
        board_col = QVBoxLayout()
        status_row = QHBoxLayout()
        self.status_mark = QLabel('')
        status_row.addWidget(self.status_mark)
        self.status_text = QLabel('')
        status_row.addWidget(self.status_text)
        status_row.addStretch(1)
        board_col.addLayout(status_row)
        self.board = BoardWidget()
        self.board.cell_clicked.connect(self.on_cell_click)
        board_col.addWidget(self.board, 1)
        btn_row = QHBoxLayout()
        new_btn = QPushButton(t('New Game', '新遊戲'))
        new_btn.clicked.connect(self.new_game)
        btn_row.addWidget(new_btn)
        btn_row.addStretch(1)
        board_col.addLayout(btn_row)
        body.addLayout(board_col, 1)

        panel = QVBoxLayout()
        panel.setSpacing(10)

        # Game info card
        info_card = QFrame()
        info_card.setObjectName('card')
        info_lay = QVBoxLayout(info_card)
        info_title = QLabel(t('Game Info', '遊戲資訊'))
        info_title.setObjectName('cardTitle')
        info_lay.addWidget(info_title)
        self.info_game = QLabel('')
        self.info_game.setObjectName('muted')
        self.info_mode = QLabel('')
        self.info_mode.setObjectName('muted')
        self.info_x = QLabel('')
        self.info_o = QLabel('')
        info_lay.addWidget(self.info_game)
        info_lay.addWidget(self.info_mode)
        info_lay.addWidget(self.info_x)
        info_lay.addWidget(self.info_o)
        self.assistant_switch = QCheckBox(t('AI Assistant', 'AI 助手'))
        self.assistant_switch.setChecked(True)
        self.assistant_switch.toggled.connect(self.on_assistant_toggled)
        info_lay.addWidget(self.assistant_switch)
        panel.addWidget(info_card)

        # Assistant card
        az_card = QFrame()
        az_card.setObjectName('card')
        az_lay = QVBoxLayout(az_card)
        az_title = QLabel(t('Best Moves', '最佳棋步'))
        az_title.setObjectName('cardTitle')
        az_lay.addWidget(az_title)
        self.analysis_list = QListWidget()
        self.analysis_list.setMinimumHeight(160)
        self.analysis_list.itemClicked.connect(self.on_analysis_clicked)
        az_lay.addWidget(self.analysis_list)
        hint = QLabel(t('Click a move to highlight it', '點擊棋步可在棋盤上標示'))
        hint.setObjectName('muted')
        az_lay.addWidget(hint)
        panel.addWidget(az_card)

        # CvC controls card
        self.cvc_card = QFrame()
        self.cvc_card.setObjectName('card')
        cvc_lay = QVBoxLayout(self.cvc_card)
        cvc_title = QLabel(t('CvC Controls', '電腦對戰控制'))
        cvc_title.setObjectName('cardTitle')
        cvc_lay.addWidget(cvc_title)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 20)  # 0.1s .. 2.0s
        self.speed_slider.setValue(4)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        speed_row = QHBoxLayout()
        self.speed_lbl = QLabel(t('Speed', '速度') + ': 0.4s')
        self.speed_lbl.setObjectName('muted')
        speed_row.addWidget(self.speed_slider, 1)
        speed_row.addWidget(self.speed_lbl)
        cvc_lay.addLayout(speed_row)
        self.auto_switch = QCheckBox(t('Auto-play', '自動播放'))
        self.auto_switch.setChecked(True)
        self.auto_switch.toggled.connect(self.on_auto_toggled)
        cvc_lay.addWidget(self.auto_switch)
        self.step_btn = QPushButton(t('Step / Next Move', '下一步'))
        self.step_btn.clicked.connect(self.run_ai)
        self.step_btn.setEnabled(False)
        cvc_lay.addWidget(self.step_btn)
        panel.addWidget(self.cvc_card)

        panel.addStretch(1)
        panel_widget = QWidget()
        panel_widget.setObjectName('sidePanel')
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(320)
        body.addWidget(panel_widget)
        root.addLayout(body, 1)

    # -- session flow ------------------------------------------------------

    def start_session(self, session):
        self.session = session
        self.new_game()

    def new_game(self):
        s = self.session
        self.game = (NormalGame() if s['game_type'] == 'normal' else UltimateGame())
        s['game'] = self.game
        self.gen += 1
        self.busy = False
        self.analysis_busy = False
        self.analysis_pending = False
        self.cvc_timer.stop()
        self.speed_slider.setValue(int(s.get('cvc_speed', 0.4) * 10))
        self.cvc_timer.setInterval(max(50, int(s.get('cvc_speed', 0.4) * 1000)))
        self.assistant_switch.setChecked(s.get('assistant_enabled', True))
        self.board.set_game(self.game)
        self.analysis_list.clear()
        self.refresh()
        self.after_move()

    def refresh(self):
        if self.game is None:
            return
        result = self.game.result()
        if result in (X, O):
            self.status_mark.setText('✕' if result == X else '○')
            self.status_text.setText(f'Player {result} wins! (玩家 {result} 獲勝！)')
        elif result == 'D':
            self.status_mark.setText('—')
            self.status_text.setText(t("It's a draw!", '平局！'))
        else:
            player = self.game.current
            self.status_mark.setText('✕' if player == X else '○')
            if self.busy:
                action = t('thinking...', '思考中')
            elif current_side_type(self.session) == 'Human':
                action = t('your move', '輪到你')
            else:
                action = t('AI thinking...', 'AI 思考中')
            self.status_text.setText(f'Player {player} · {action}')
        self.status_mark.setStyleSheet(
            'color: ' + (X_COLOR if result == X else O_COLOR) + '; font-size: 20px; font-weight: 700;')
        game_text = 'Normal' if isinstance(self.game, NormalGame) else 'Ultimate'
        mode_text = {'pvp': 'PvP', 'pvc': 'PvC', 'cvc': 'CvC'}[self.session['mode']]
        x_type, o_type = side_types(self.session)
        self.info_game.setText(f'{game_text} · {mode_text}')
        self.info_x.setText(f'✕ {self.side_label(x_type)}')
        self.info_o.setText(f'○ {self.side_label(o_type)}')
        self.board.update()
        self.update_cvc_controls()

    def side_label(self, kind):
        if kind == 'Human':
            return t('Human (You)', '玩家 (你)')
        return f'Computer ({kind}) — 電腦 ({kind})'

    # -- moves -------------------------------------------------------------

    def on_cell_click(self, move):
        game = self.game
        if game.is_over() or self.busy or is_ai_turn(self.session):
            return
        if move not in game.legal_moves():
            return
        self.apply_move(move)

    def apply_move(self, move):
        side = self.game.current
        apply_move(self.game, move)
        log.info('Move: %s -> %s [%s]', side, move_text(move),
                 'Normal' if isinstance(self.game, NormalGame) else 'Ultimate')
        self.gen += 1
        self.board.set_game(self.game)
        self.refresh()
        self.trigger_analysis()
        self.after_move()

    def after_move(self):
        if self.game.is_over():
            self.cvc_timer.stop()
            self.show_result()
            return
        if is_ai_turn(self.session):
            if self.session['mode'] == 'cvc':
                if self.session.get('cvc_auto', True):
                    self.cvc_timer.start()
                else:
                    self.update_cvc_controls()
            else:
                self.pvc_timer.start(300)
        else:
            self.update_cvc_controls()

    def cvc_tick(self):
        if self.session['mode'] != 'cvc':
            self.cvc_timer.stop()
            return
        if self.game.is_over() or self.busy or not is_ai_turn(self.session):
            return
        if not self.session.get('cvc_auto', True):
            self.cvc_timer.stop()
            return
        self.run_ai()

    def run_ai(self):
        if self.game is None or self.game.is_over() or self.busy or not is_ai_turn(self.session):
            return
        x_type, o_type = side_types(self.session)
        ai_type = x_type if self.game.current == X else o_type
        self.busy = True
        self.refresh()
        worker = AIWorker(self.game.clone(), ai_type,
                          self.session['mcts'], self.session.get('minimax_depth', 3))
        worker.move_ready.connect(self.on_ai_done)
        worker.move_failed.connect(self.on_ai_failed)
        worker.finished.connect(lambda w=worker: self._reap(w))
        self.workers.append(worker)
        worker.start()

    def on_ai_done(self, move):
        self.busy = False
        self.apply_move(move)

    def on_ai_failed(self, message):
        self.busy = False
        log.error('AI move failed: %s', message)
        self.refresh()

    def _reap(self, worker):
        if worker in self.workers:
            self.workers.remove(worker)

    def on_speed_changed(self, value):
        self.session['cvc_speed'] = value / 10.0
        self.speed_lbl.setText(t('Speed', '速度') + f': {value / 10.0:.1f}s')
        self.cvc_timer.setInterval(value * 100)

    def on_auto_toggled(self, checked):
        self.session['cvc_auto'] = checked
        self.update_cvc_controls()
        if checked and self.session['mode'] == 'cvc' and is_ai_turn(self.session):
            self.cvc_timer.start()

    def update_cvc_controls(self):
        cvc = self.session is not None and self.session['mode'] == 'cvc'
        self.cvc_card.setVisible(cvc)
        if not cvc or self.game is None:
            return
        ai_turn = not self.game.is_over() and is_ai_turn(self.session)
        auto = self.session.get('cvc_auto', True)
        self.step_btn.setEnabled(not auto and ai_turn and not self.busy)

    # -- assistant ---------------------------------------------------------

    def on_assistant_toggled(self, checked):
        self.session['assistant_enabled'] = checked
        if checked:
            self.trigger_analysis()
        else:
            self.analysis_list.clear()
            self.analysis_list.addItem(t('Assistant disabled', '助手已關閉'))

    def trigger_analysis(self):
        if not self.session.get('assistant_enabled', True):
            return
        if self.game is None or self.game.is_over():
            return
        if self.analysis_busy:
            self.analysis_pending = True
            return
        self.analysis_busy = True
        gen = self.gen
        snapshot = self.game.clone()
        worker = AnalysisWorker(snapshot, self.session['mcts'])
        worker.result_ready.connect(lambda items, g=gen: self.on_analysis_done(items, g))
        worker.finished.connect(lambda w=worker: self._reap(w))
        self.workers.append(worker)
        worker.start()

    def on_analysis_done(self, items, gen):
        self.analysis_busy = False
        if gen == self.gen and self.session.get('assistant_enabled', True):
            self.render_analysis(items)
        if self.analysis_pending:
            self.analysis_pending = False
            self.trigger_analysis()

    def render_analysis(self, items):
        self.analysis_list.clear()
        if not items:
            self.analysis_list.addItem(t('No moves to analyze', '沒有可分析的棋步'))
            return
        for it in items[:5]:
            pct = max(0.0, min(1.0, it['pct']))
            if isinstance(self.game, NormalGame):
                verdict = 'Win' if pct == 1.0 else ('Draw' if pct == 0.5 else 'Loss')
            else:
                verdict = f'{pct:.0%}'
            reason_en, reason_zh = it['reason']
            item = QListWidgetItem(f'{move_text(it["move"])}  {verdict}  {t(reason_en, reason_zh)}')
            item.setData(Qt.UserRole, it['move'])
            self.analysis_list.addItem(item)

    def on_analysis_clicked(self, item):
        move = item.data(Qt.UserRole)
        if move is not None:
            self.board.flash(move)

    # -- result ------------------------------------------------------------

    def show_result(self):
        result = self.game.result()
        title = (t("It's a draw!", '平局！') if result == 'D'
                 else f'Player {result} wins! (玩家 {result} 獲勝！)')
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(title)
        again = box.addButton(t('Play Again', '再玩一次'), QMessageBox.AcceptRole)
        back = box.addButton(t('Back to Menu', '返回選單'), QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is again:
            self.new_game()
        else:
            self.back_requested.emit()

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, web_enabled=False, port=8080):
        super().__init__()
        self.web_proc = None
        self.web_port = port
        self.setWindowTitle('Ultimate Tic Tac Toe — 終極井字棋')
        self.setStyleSheet(QSS)
        self.stack = QStackedWidget()
        self.menu_page = MenuPage(web_port=port)
        self.game_page = GamePage()
        self.menu_page.start_requested.connect(self.start_game)
        self.menu_page.web_toggled.connect(self.set_web_enabled)
        self.game_page.back_requested.connect(self.show_menu)
        self.stack.addWidget(self.menu_page)
        self.stack.addWidget(self.game_page)
        self.setCentralWidget(self.stack)
        self.resize(1024, 740)
        if web_enabled:
            self.menu_page.set_web_checked(True)

    def start_game(self, session):
        self.game_page.start_session(session)
        self.stack.setCurrentWidget(self.game_page)

    def show_menu(self):
        self.stack.setCurrentWidget(self.menu_page)

    def set_web_enabled(self, enabled):
        if enabled:
            self.start_web()
        else:
            self.stop_web()

    def start_web(self):
        if self.web_proc is not None and self.web_proc.poll() is None:
            return
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        cwd = os.path.dirname(os.path.abspath(__file__))
        self.web_proc = subprocess.Popen(
            [sys.executable, 'SBA.py', '--web', '--host', '0.0.0.0',
             '--port', str(self.menu_page.web_port)],
            cwd=cwd,
            creationflags=flags,
        )
        log.info('NiceGUI web server started on port %d', self.menu_page.web_port)
        self.menu_page.web_status.setText(
            f'http://127.0.0.1:{self.menu_page.web_port} ({t("running", "執行中")})')

    def stop_web(self):
        if self.web_proc is not None and self.web_proc.poll() is None:
            self.web_proc.terminate()
            try:
                self.web_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.web_proc.kill()
        self.web_proc = None

    def closeEvent(self, event):
        self.stop_web()
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description='SBA desktop app (PySide6)')
    parser.add_argument('--web', action='store_true',
                        help='start the NiceGUI web server at launch')
    parser.add_argument('--port', type=int, default=8080,
                        help='port for the NiceGUI web server (default: 8080)')
    parser.add_argument('--debug', action='store_true',
                        help='verbose backend logs')
    args = parser.parse_args(argv)
    if args.debug:
        log.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    app = QApplication(sys.argv[:1])
    app.setApplicationName('SBA')
    window = MainWindow(web_enabled=args.web, port=args.port)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
