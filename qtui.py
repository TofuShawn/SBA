# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""PySide6 desktop UI for SBA (Basically Awful).

Shares the game rules (game.py) and AI engines (ai.py) with the NiceGUI web
UI. The NiceGUI web server is opt-in: enable it from the menu switch or with
`python qtui.py --web`.

Run:
    python qtui.py            # desktop app (default)
    python qtui.py --web      # desktop app + start the NiceGUI web server

Maintenance notes:
- Uses the vendored PyQt-SiliconUI (GPLv3) under vendor/siui/; falls back to
  the native dark-glass QSS when the vendor directory is missing (decision D5).
- Fonts prefer Noto Sans TC; 微软雅黑 is not installed on every system (D7).
- Won chunks fill with the winner's color; hovering reveals the cells (D6).
"""

import argparse
import logging
import os
import subprocess
import sys

try:
    from PySide6.QtCore import (QEasingCurve, QRectF, QSize, Qt, QThread,
                                QTimer, QVariantAnimation, Signal)
    from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPen
    from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
        QGraphicsDropShadowEffect, QListWidget, QListWidgetItem, QMainWindow,
        QPushButton, QSizePolicy, QSlider, QStackedWidget, QStyle, QVBoxLayout, QWidget,
    )
except ImportError:
    print('PySide6 is not installed for this Python interpreter.')
    print('Install the desktop UI with:  python -m pip install -r requirements.txt')
    print('Or start the web UI instead with:  python SBA.py --web')
    raise SystemExit(1)

from game import (
    X, O,
    NormalGame, UltimateGame, apply_move, micro_win_line,
)
from ai import get_ai_move, compute_analysis, position_win_rates, move_text
from SBA import (
    AI_OPTIONS, current_side_type, is_ai_turn, log, new_session,
    side_label, side_types, t,
)


# Font stack: Noto Sans TC is installed on Windows 10+; Segoe UI/雅黑 are
# fallbacks so CJK and Latin both render cleanly.
FONT_FAMILIES = ['Noto Sans TC', 'Noto Sans SC', 'Microsoft YaHei', 'Segoe UI']


# Board palettes. BoardWidget paints from the active palette; the native dark
# glass fallback theme switches ACTIVE_THEME to 'DARK'.
PALETTE = {
    'LIGHT': {
        'x': '#6750A4',
        'o': '#B3261E',
        'cell_empty': '#F3EDF7',
        'cell_filled': '#ECE6F0',
        'macro_active': 'rgba(103, 80, 164, 0.10)',
        'active_outline': '#1976D2',
        'macro_won_x': 'rgba(103, 80, 164, 0.14)',
        'macro_won_o': 'rgba(179, 38, 30, 0.10)',
        'grid_line': '#CAC4D0',
        'macro_line': '#79747E',
        'flash': '#6750A4',
        'win_fill_x': '#6750A4',
        'win_fill_o': '#B3261E',
        'win_mark': '#FFFFFF',
    },
    'DARK': {
        'x': '#D0BCFF',
        'o': '#FFB4AB',
        'cell_empty': '#211F26',
        'cell_filled': '#2B2930',
        'macro_active': 'rgba(208, 188, 255, 0.12)',
        'active_outline': '#64B5F6',
        'macro_won_x': 'rgba(208, 188, 255, 0.16)',
        'macro_won_o': 'rgba(255, 180, 171, 0.14)',
        'grid_line': '#49454F',
        'macro_line': '#938F99',
        'flash': '#D0BCFF',
        'win_fill_x': '#6750A4',
        'win_fill_o': '#B3261E',
        'win_mark': '#FFFFFF',
    },
}
ACTIVE_THEME = 'DARK'  # dark glass is the default; siui theme overrides later.
PAL = PALETTE[ACTIVE_THEME]


def _glass_shadow(widget):
    """Attach a soft drop shadow to a glass panel."""
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(26)
    effect.setOffset(0, 6)
    effect.setColor(QColor(0, 0, 0, 100))
    widget.setGraphicsEffect(effect)


# ---------------------------------------------------------------------------
# PyQt-SiliconUI integration (optional; vendored under vendor/siui, GPLv3)
# ---------------------------------------------------------------------------
SIUI = None


def _load_siui():
    """Load the vendored PyQt-SiliconUI (PySide6 fork) when present."""
    global SIUI
    siui_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor', 'siui')
    if not os.path.isdir(os.path.join(siui_root, 'silicon')):
        return None
    if siui_root not in sys.path:
        sys.path.insert(0, siui_root)
    try:
        import silicon
        from silicon import SiGlobal
        import icons as si_icons
        SiGlobal.icons = si_icons.ICON_DICT(
            os.path.join(siui_root, 'icons', 'icons.dat'), SiGlobal.colorset.SVG_HEX)
        SIUI = silicon
        log.info('PyQt-SiliconUI loaded (%s)', siui_root)
    except Exception as e:  # noqa: BLE001
        log.warning('PyQt-SiliconUI unavailable, using native dark theme: %s', e)
        SIUI = None
    return SIUI


def _init_siui_runtime():
    """Create the SiliconUI floating tooltip window (needs a QApplication)."""
    if SIUI is None:
        return
    try:
        from silicon import SiGlobal
        from silicon.SiHint import FloatingWindow
        SiGlobal.floating_window = FloatingWindow()
        # A tooltip must never steal mouse events from the board beneath it
        # (it stays on top and follows the cursor).
        SiGlobal.floating_window.setAttribute(Qt.WA_TransparentForMouseEvents)
        SiGlobal.floating_window.setWindowOpacity(0)
        SiGlobal.floating_window.show()
    except Exception as e:  # noqa: BLE001
        log.warning('SiliconUI runtime init failed: %s', e)


_load_siui()

if SIUI is not None:
    from silicon.SiComboBox import SiComboBox as _SiComboBoxBase

    class _SiCombo(_SiComboBoxBase):
        """Adapter exposing the QComboBox-style API used by the menus."""
        currentIndexChanged = Signal(int)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumSize(260, 40)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._labels = []
            self._data = []
            self._current = 0
            self._max_w = 0
            self.valueChanged.connect(self._on_value)

        def _text_w(self, text):
            try:
                from silicon import SiFont
                fm = QFontMetrics(SiFont.font_L1)
            except Exception:  # noqa: BLE001
                fm = self.fontMetrics()
            return fm.horizontalAdvance(text)

        def sizeHint(self):
            return QSize(max(260, self._max_w + 78), 40)

        def _on_value(self, value):
            try:
                self._current = self._data.index(value)
            except ValueError:
                pass
            self.currentIndexChanged.emit(self._current)

        def addItem(self, label, data):
            self._labels.append(label)
            self._data.append(data)
            self.addOption(label, data)
            self._max_w = max(self._max_w, self._text_w(label))
            if len(self._data) == 1:
                self.setOption(label)

        def currentData(self):
            return self._data[self._current] if self._data else None

        def currentIndex(self):
            return self._current


def _si_combo(parent=None):
    if SIUI is not None:
        try:
            return _SiCombo(parent)
        except Exception:  # noqa: BLE001
            pass
    return QComboBox(parent)


def _si_button(text, parent=None, primary=False):
    if SIUI is not None:
        from silicon import SiFont
        from silicon.SiButton import SiButton
        btn = SiButton(parent)
        btn.setText(text)
        if primary:
            btn.setStrong(True)
        fm = QFontMetrics(SiFont.font_L1_bold)
    else:
        btn = QPushButton(text, parent)
        if primary:
            btn.setObjectName('primary')
        fm = QFontMetrics(QFont(FONT_FAMILIES, 10, QFont.Bold))
    btn.setMinimumSize(max(140, fm.horizontalAdvance(text) + 48), 44)
    return btn


def _si_set_enabled(btn, enabled):
    """Enable/disable a button (SiButton needs its inner layer toggled too)."""
    btn.setEnabled(enabled)
    if SIUI is not None:
        try:
            btn.layer_front.setEnabled(enabled)
        except AttributeError:
            pass


if SIUI is not None:
    try:
        from silicon import SiGlobal as _SiG
        cs = _SiG.colorset
        PAL.update({
            'cell_empty': cs.BG_GRAD_HEX[1],
            'cell_filled': cs.BG_GRAD_HEX[3],
            'grid_line': cs.BG_GRAD_HEX[4],
            'macro_line': cs.BG_GRAD_HEX[4],
            'macro_active': 'rgba(82, 56, 154, 0.14)',
            'macro_won_x': 'rgba(82, 56, 154, 0.20)',
            'macro_won_o': 'rgba(218, 52, 98, 0.16)',
        })
    except Exception:  # noqa: BLE001
        pass

# Native dark-glass theme (fallback used when the PyQt-SiliconUI package is
# not installed): dark background, translucent glass cards, capsule buttons
# and smooth controls.
QSS = '''
QWidget { background: #141218; color: #E6E0E9; font-size: 14px;
    font-family: "Noto Sans TC", "Noto Sans SC", "Microsoft YaHei", "Segoe UI"; }
QMainWindow { background: #141218; }
QLabel#title { font-size: 20px; font-weight: 700; color: #E8DEF8; }
QLabel#cardTitle { font-weight: 600; color: #E8DEF8; }
QLabel#muted { color: #CAC4D0; font-size: 12px; }
QFrame#card {
    background: rgba(45, 40, 52, 0.86);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
}
QFrame#sidePanel {
    background: rgba(24, 21, 28, 0.72);
    border-left: 1px solid rgba(255, 255, 255, 0.07);
}
QPushButton {
    background: rgba(73, 69, 79, 0.55); color: #E6E0E9; border: none;
    border-radius: 16px; padding: 8px 18px; font-weight: 500;
}
QPushButton:hover { background: rgba(93, 88, 103, 0.75); }
QPushButton:pressed { background: rgba(103, 80, 164, 0.55); }
QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #7C6BB0, stop:1 #4F378B);
    color: #FFFFFF; font-weight: 600; padding: 10px 22px; border-radius: 18px;
}
QPushButton#primary:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #8B79BE, stop:1 #5B46A0); }
QPushButton#primary:pressed { background: #4F378B; }
QPushButton:disabled { background: rgba(73, 69, 79, 0.4); color: #6F6A76; }
QComboBox, QSpinBox {
    background: rgba(45, 40, 52, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px;
    padding: 5px 10px; min-height: 22px;
}
QComboBox:hover, QSpinBox:hover { border-color: #6750A4; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #211F26; color: #E6E0E9;
    border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px;
    selection-background-color: #4F378B; selection-color: #FFFFFF;
}
QSlider::groove:horizontal { height: 6px; background: #49454F; border-radius: 3px; }
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #B39DDB, stop:1 #6750A4); border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #E6E0E9; border: 2px solid #6750A4;
    width: 18px; height: 18px; margin: -8px 0; border-radius: 9px;
}
QSlider::handle:horizontal:hover { background: #D0BCFF; }
QCheckBox::indicator {
    width: 18px; height: 18px; border-radius: 6px;
    border: 2px solid #938F99; background: #211F26;
}
QCheckBox::indicator:hover { border-color: #D0BCFF; }
QCheckBox::indicator:checked { background: #6750A4; border-color: #6750A4; }
QRadioButton::indicator {
    width: 18px; height: 18px; border-radius: 9px;
    border: 2px solid #938F99; background: #211F26;
}
QRadioButton::indicator:hover { border-color: #D0BCFF; }
QRadioButton::indicator:checked { background: #6750A4; border-color: #6750A4; }
QListWidget {
    background: rgba(33, 30, 38, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 4px;
}
QListWidget::item { padding: 8px; border-radius: 8px; }
QListWidget::item:hover { background: rgba(103, 80, 164, 0.35); }
QListWidget::item:selected { background: #4F378B; color: #FFFFFF; }
QToolTip {
    background: #2B2930; color: #E6E0E9;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px; padding: 6px 10px;
}
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #49454F; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #6F6A76; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QMessageBox { background: #211F26; }
QMessageBox QPushButton { min-width: 90px; }
'''


def _mark_color(player):
    return QColor(PAL['x'] if player == X else PAL['o'])


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
        # Reveal a won chunk by clicking it (more reliable than pointer
        # tracking, which fought with event delivery on some setups).
        self._revealed_macro = None
        self._reveal_blend = 0.0
        self._reveal_anim = QVariantAnimation(self)
        self._reveal_anim.setDuration(180)
        self._reveal_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._reveal_anim.valueChanged.connect(self._on_reveal_anim)
        self.setMinimumSize(300, 300)

    def set_game(self, game):
        self.game = game
        self.legal = set(game.legal_moves())
        self.flash_cell = None
        self._reveal_anim.stop()
        self._revealed_macro = None
        self._reveal_blend = 0.0
        self.update()

    def _on_reveal_anim(self, value):
        self._reveal_blend = float(value)
        self.update()

    def _toggle_reveal(self, m):
        """Toggle a won chunk's reveal with one non-looping fade."""
        self._revealed_macro = None if self._revealed_macro == m else m
        self._reveal_anim.stop()
        self._reveal_anim.setStartValue(self._reveal_blend)
        self._reveal_anim.setEndValue(1.0 if self._revealed_macro is not None else 0.0)
        self._reveal_anim.start()

    def _macro_at(self, px, py):
        if not isinstance(self.game, UltimateGame):
            return None
        n, cell, ox, oy = self._grid()
        col = int((px - ox) // (cell + self.GAP))
        row = int((py - oy) // (cell + self.GAP))
        if not (0 <= row < n and 0 <= col < n):
            return None
        return (row // 3) * 3 + (col // 3)

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
        grid = n * cell + (n - 1) * self.GAP
        return n, cell, (self.width() - grid) / 2, (self.height() - grid) / 2

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
        if self.game is None:
            return
        if isinstance(self.game, UltimateGame):
            m = self._macro_at(event.position().x(), event.position().y())
            if m is not None and self.game.macro[m] in (X, O):
                self._toggle_reveal(m)
                return
        if self.game.is_over():
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
        painter.setBrush(QColor(PAL['cell_filled'] if mark else PAL['cell_empty']))
        painter.drawRoundedRect(rect, 4, 4)
        if mark in (X, O):
            self._paint_mark(painter, rect, mark)
        if is_flash:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(PAL['flash']), 3))
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)

    def _paint_mark(self, painter, rect, player, color=None):
        pad = rect.width() * 0.22
        color = _mark_color(player) if color is None else QColor(color)
        pen = QPen(color, max(2.0, rect.width() * 0.09))
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

    def _win_segment(self, line, centers, bounds):
        """Extend a win line across `bounds`, mirroring the web UI's win_segment."""
        pts = [centers[i] for i in line]
        xs = [p.x() for p in pts]
        x1, x2 = min(xs), max(xs)
        y1 = next(p.y() for p in pts if p.x() == x1)
        y2 = next(p.y() for p in pts if p.x() == x2)
        if x1 == x2:  # vertical line
            top = bounds.top() + bounds.height() * 0.04
            bottom = bounds.bottom() - bounds.height() * 0.04
            return (x1, top), (x1, bottom)
        slope = (y2 - y1) / (x2 - x1)
        margin = bounds.width() * (0.07 if abs(slope) > 0.5 else 0.04)
        p1 = (bounds.left() + margin, y1 + (bounds.left() + margin - x1) * slope)
        p2 = (bounds.right() - margin, y2 + (bounds.right() - margin - x2) * slope)
        return p1, p2

    def _paint_win_line(self, painter, line, centers, color, width, bounds):
        if line is None:
            return
        (x1, y1), (x2, y2) = self._win_segment(line, centers, bounds)
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
        painter.setPen(QPen(QColor(PAL['grid_line']), 1.5))
        for i in range(1, 3):
            x = ox + i * (cell + self.GAP) - self.GAP / 2
            y = oy + i * (cell + self.GAP) - self.GAP / 2
            painter.drawLine(x, oy, x, oy + 3 * cell + 2 * self.GAP)
            painter.drawLine(ox, y, ox + 3 * cell + 2 * self.GAP, y)
        line = micro_win_line(self.game.board)
        if line:
            winner = self.game.board[line[0]]
            bounds = QRectF(ox, oy, 3 * cell + 2 * self.GAP, 3 * cell + 2 * self.GAP)
            self._paint_win_line(painter, line, centers, _mark_color(winner),
                                 cell * 0.12, bounds)

    # -- Ultimate ----------------------------------------------------------

    def _paint_ultimate(self, painter):
        n, cell, ox, oy = self._grid()
        # macro region highlights
        for m in range(9):
            rect = self._macro_rect(m, cell, ox, oy)
            if self.game.macro[m] == X:
                painter.setBrush(QColor(PAL['macro_won_x']))
            elif self.game.macro[m] == O:
                painter.setBrush(QColor(PAL['macro_won_o']))
            elif m == self.game.active_macro and self.game.macro_open(m):
                painter.setBrush(QColor(PAL['macro_active']))
            else:
                continue
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect, 10, 10)
        # micro cells
        macro_centers = []
        micro_centers = [[] for _ in range(9)]
        for m in range(9):
            mr, mc = divmod(m, 3)
            macro_rect = self._macro_rect(m, cell, ox, oy)
            macro_centers.append(macro_rect.center())
            for i in range(9):
                row = mr * 3 + i // 3
                col = mc * 3 + i % 3
                rect = self._cell_rect(row, col, cell, ox, oy)
                micro_centers[m].append(rect.center())
                self._paint_cell(painter, rect, self.game.micro[m][i],
                                 self.flash_cell == (m, i))
        # grid lines
        painter.setPen(QPen(QColor(PAL['grid_line']), 1))
        for i in range(1, 9):
            x = ox + i * (cell + self.GAP) - self.GAP / 2
            y = oy + i * (cell + self.GAP) - self.GAP / 2
            painter.drawLine(x, oy, x, oy + 9 * cell + 8 * self.GAP)
            painter.drawLine(ox, y, ox + 9 * cell + 8 * self.GAP, y)
        painter.setPen(QPen(QColor(PAL['macro_line']), 2))
        for i in (3, 6):
            x = ox + i * (cell + self.GAP) - self.GAP / 2
            y = oy + i * (cell + self.GAP) - self.GAP / 2
            painter.drawLine(x, oy, x, oy + 9 * cell + 8 * self.GAP)
            painter.drawLine(ox, y, ox + 9 * cell + 8 * self.GAP, y)
        # blue outline on the chunk the player must play in (Ultimate);
        # drawn on top and slightly outside the chunk so cells don't cover it
        if (self.game.active_macro is not None
                and self.game.macro_open(self.game.active_macro)):
            rect = self._macro_rect(self.game.active_macro, cell, ox, oy)
            painter.setPen(QPen(QColor(PAL['active_outline']), 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(-2.5, -2.5, 2.5, 2.5), 10, 10)
        # won macro chunks: fill with the player's color and show a white
        # mark; hovering reveals the underlying cells (blend 0 = filled,
        # 1 = fully revealed)
        for m in range(9):
            if self.game.macro[m] in (X, O):
                winner = self.game.macro[m]
                rect = self._macro_rect(m, cell, ox, oy)
                blend = self._reveal_blend if m == self._revealed_macro else 0.0
                fill = QColor(PAL['win_fill_x'] if winner == X else PAL['win_fill_o'])
                fill.setAlpha(int(165 * (1.0 - blend)))  # semi-transparent mask
                if fill.alpha() > 0:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(fill)
                    painter.drawRoundedRect(rect, 10, 10)
                mark = QColor(PAL['win_mark'])
                mark.setAlpha(int(255 * (1.0 - blend)))
                if mark.alpha() > 0:
                    pad = rect.width() * 0.18
                    badge = QRectF(rect.left() + pad, rect.top() + pad,
                                   rect.width() - 2 * pad, rect.height() - 2 * pad)
                    self._paint_mark(painter, badge, winner, mark)
                line = micro_win_line(self.game.micro[m])
                if line is not None and blend > 0:
                    line_color = _mark_color(winner)
                    line_color.setAlpha(int(255 * blend))
                    self._paint_win_line(painter, line, micro_centers[m],
                                         line_color, cell * 0.12, rect)
        # overall macro win line
        whole = micro_win_line(self.game.macro)
        if whole:
            winner = self.game.macro[whole[0]]
            bounds = QRectF(ox, oy, 9 * cell + 8 * self.GAP, 9 * cell + 8 * self.GAP)
            self._paint_win_line(painter, whole, macro_centers,
                                 _mark_color(winner), cell * 0.16, bounds)

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
    result_ready = Signal(object, object)

    def __init__(self, game, budget, parent=None):
        super().__init__(parent)
        self.game = game
        self.budget = budget

    def run(self):
        try:
            items = compute_analysis(self.game, self.budget)
            rates = position_win_rates(self.game, self.budget)
            self.result_ready.emit(items, rates)
        except Exception as e:  # noqa: BLE001
            log.error('Analysis failed: %s', e)
            self.result_ready.emit([], (0.5, 0.0, 0.5))


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
        card = QFrame()
        card.setObjectName('card')
        _glass_shadow(card)
        card_lay = QVBoxLayout(card)
        card_lay.setSpacing(8)
        root.addWidget(card)
        root.addStretch(1)

        def field_row(label_text, widget):
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            lab = QLabel(label_text)
            lab.setObjectName('muted')
            lab.setWordWrap(True)
            lab.setMinimumWidth(320)
            lab.setMaximumWidth(360)
            row.addWidget(lab, 0)
            row.addWidget(widget, 1)
            return row_widget, lab

        self.game_type = _si_combo()
        self.game_type.addItem('Normal Tic Tac Toe (普通井字棋)', 'normal')
        self.game_type.addItem('Ultimate Tic Tac Toe (終極井字棋)', 'ultimate')
        card_lay.addWidget(field_row(t('Game Type', '遊戲類型'), self.game_type)[0])

        self.mode = _si_combo()
        self.mode.addItem('PvP (玩家對玩家)', 'pvp')
        self.mode.addItem('Player vs Computer (玩家對電腦)', 'pvc')
        self.mode.addItem('Computer vs Computer (電腦對電腦)', 'cvc')
        self.mode.currentIndexChanged.connect(lambda _: self._update_visibility())
        card_lay.addWidget(field_row(t('Mode', '模式'), self.mode)[0])

        self.first = _si_combo()
        self.first.addItem(t('You move first — X', '你先手 — X'), 'human')
        self.first.addItem(t('Computer moves first — O', '電腦先手 — O'), 'computer')
        self.first.currentIndexChanged.connect(lambda _: self._update_visibility())
        self.first_row, _ = field_row(t('First Player', '先手'), self.first)
        card_lay.addWidget(self.first_row)

        self.ai_x = _si_combo()
        for key, label in AI_OPTIONS.items():
            self.ai_x.addItem(label, key)
        self.ai_x_row, _ = field_row(t('Player X — AI Level', '玩家 X — AI 等級'), self.ai_x)
        card_lay.addWidget(self.ai_x_row)

        self.ai_o = _si_combo()
        for key, label in AI_OPTIONS.items():
            self.ai_o.addItem(label, key)
        self.ai_o_row, self.ai_o_label = field_row(t('Player O — AI Level', '玩家 O — AI 等級'), self.ai_o)
        card_lay.addWidget(self.ai_o_row)

        self.mcts = QSlider(Qt.Horizontal)
        self.mcts.setRange(200, 3000)
        self.mcts.setSingleStep(100)
        self.mcts.setValue(800)
        card_lay.addWidget(field_row(t('MCTS Strength', 'MCTS 強度'),
                                     self._slider_row(self.mcts))[0])

        self.mm_depth = QSlider(Qt.Horizontal)
        self.mm_depth.setRange(2, 6)
        self.mm_depth.setValue(4)
        card_lay.addWidget(field_row(t('Minimax Depth (Ultimate)', 'Minimax 深度（終極模式）'),
                                     self._slider_row(self.mm_depth))[0])

        self.assistant = QCheckBox(t('AI Assistant', 'AI 助手'))
        self.assistant.setChecked(True)
        card_lay.addWidget(self.assistant)

        card_lay.addSpacing(10)
        web_title = QLabel(t('NiceGUI Web UI (選用啟動)', 'NiceGUI Web 介面（選用）'))
        web_title.setObjectName('cardTitle')
        card_lay.addWidget(web_title)
        self.web_switch = QCheckBox(t('Enable NiceGUI Web UI', '啟動 Web 介面'))
        self.web_switch.toggled.connect(self._on_web_toggled)
        card_lay.addWidget(self.web_switch)
        self.web_status = QLabel(t('Web UI stopped', 'Web 介面已停止'))
        self.web_status.setObjectName('muted')
        card_lay.addWidget(self.web_status)

        start_btn = _si_button(t('Start Game', '開始遊戲'), primary=True)
        start_btn.clicked.connect(self._on_start)
        card_lay.addWidget(start_btn)

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
        self.first_row.setVisible(mode == 'pvc')
        self.ai_x_row.setVisible(mode == 'cvc')
        self.ai_o_row.setVisible(mode in ('pvc', 'cvc'))
        if mode == 'pvc':
            label = (t('Computer (X) — AI Level', '電腦 (X) — AI 等級')
                     if self.first.currentData() == 'computer'
                     else t('Computer (O) — AI Level', '電腦 (O) — AI 等級'))
        else:
            label = t('Player O — AI Level', '玩家 O — AI 等級')
        self.ai_o_label.setText(label)

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
        self.step_idx = 0
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
        self.play_again_btn = _si_button(t('Play Again', '再玩一次'), primary=True)
        self.play_again_btn.clicked.connect(self.new_game)
        self.play_again_btn.setVisible(False)
        top.addWidget(self.play_again_btn)
        back_btn = _si_button(t('Back to Menu', '返回選單'))
        back_btn.clicked.connect(self.back_requested.emit)
        top.addWidget(back_btn)
        root.addLayout(top)

        # Top control bar: revert / pause / next step + step scrubber
        ctrl_bar = QFrame()
        ctrl_bar.setObjectName('card')
        ctrl_lay = QHBoxLayout(ctrl_bar)
        ctrl_lay.setContentsMargins(8, 6, 8, 6)
        self.revert_btn = QPushButton()
        self.revert_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        self.revert_btn.setToolTip(t('Revert', '回退'))
        self.revert_btn.setFixedSize(40, 36)
        self.revert_btn.clicked.connect(self.on_revert_clicked)
        ctrl_lay.addWidget(self.revert_btn)
        self.pause_btn = QPushButton()
        self.pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pause_btn.setToolTip(t('Pause', '暫停'))
        self.pause_btn.setFixedSize(40, 36)
        self.pause_btn.clicked.connect(self.on_pause_clicked)
        ctrl_lay.addWidget(self.pause_btn)
        self.next_btn = QPushButton()
        self.next_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.next_btn.setToolTip(t('Next Step', '下一步'))
        self.next_btn.setFixedSize(40, 36)
        self.next_btn.clicked.connect(self.on_step_clicked)
        ctrl_lay.addWidget(self.next_btn)
        self.hist_slider = QSlider(Qt.Horizontal)
        self.hist_slider.setRange(0, 0)
        self.hist_slider.valueChanged.connect(self.go_to_step)
        ctrl_lay.addWidget(self.hist_slider, 1)
        self.hist_label = QLabel('0 / 0')
        self.hist_label.setObjectName('muted')
        ctrl_lay.addWidget(self.hist_label)
        root.addWidget(ctrl_bar)

        # Win-rate chart (lives under the board, so the right panel spans the
        # full window height); single X-win-rate line with a 50% reference.
        self.hist_chart = QChart()
        self.hist_chart.setAnimationOptions(QChart.NoAnimation)
        self.hist_chart.legend().setVisible(True)
        self.hist_chart.legend().setAlignment(Qt.AlignLeft)
        self.hist_chart.legend().setLabelColor(QColor('#E6E0E9'))
        self.hist_chart.setBackgroundVisible(False)
        self.hist_x_series = QLineSeries()
        self.hist_x_series.setName(t('X win rate', 'X 勝率'))
        self.hist_x_series.setColor(QColor(PAL['x']))
        self.hist_ref = QLineSeries()
        ref_pen = QPen(QColor('#79747E'))
        ref_pen.setStyle(Qt.DashLine)
        self.hist_ref.setPen(ref_pen)
        self.hist_ref.setName('50%')
        self.hist_chart.addSeries(self.hist_x_series)
        self.hist_chart.addSeries(self.hist_ref)
        self.hist_axis = QValueAxis()
        self.hist_axis_y = QValueAxis()
        self.hist_axis.setRange(0, 1)
        self.hist_axis.setTitleText(t('Step', '步'))
        self.hist_axis_y.setRange(0, 100)
        self.hist_axis_y.setTitleText('%')
        self.hist_chart.addAxis(self.hist_axis, Qt.AlignBottom)
        self.hist_chart.addAxis(self.hist_axis_y, Qt.AlignLeft)
        for ax in (self.hist_axis, self.hist_axis_y):
            ax.setLabelsColor(QColor('#CAC4D0'))
            ax.setTitleBrush(QBrush(QColor('#CAC4D0')))
        self.hist_x_series.attachAxis(self.hist_axis)
        self.hist_x_series.attachAxis(self.hist_axis_y)
        self.hist_ref.attachAxis(self.hist_axis)
        self.hist_ref.attachAxis(self.hist_axis_y)
        self.hist_dot = QScatterSeries()
        self.hist_dot.setName(t('Current', '目前'))
        self.hist_dot.setMarkerSize(10.0)
        self.hist_dot.setColor(QColor('#B3261E'))
        self.hist_dot.setBorderColor(QColor('#B3261E'))
        self.hist_chart.addSeries(self.hist_dot)
        self.hist_dot.attachAxis(self.hist_axis)
        self.hist_dot.attachAxis(self.hist_axis_y)
        self.hist_x_series.setPointsVisible(True)
        self.hist_chart_view = QChartView(self.hist_chart)
        self.hist_chart_view.setRenderHint(QPainter.Antialiasing)
        self.hist_chart_view.setMinimumHeight(200)
        self.hist_chart_view.setBackgroundBrush(Qt.NoBrush)

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
        board_col.addWidget(self.board, 2)
        btn_row = QHBoxLayout()
        self.new_btn = _si_button(t('New Game', '新遊戲'))
        self.new_btn.clicked.connect(self.new_game)
        btn_row.addWidget(self.new_btn)
        btn_row.addStretch(1)
        board_col.addLayout(btn_row)
        board_col.addWidget(self.hist_chart_view, 1)
        body.addLayout(board_col, 1)

        panel = QVBoxLayout()
        panel.setSpacing(10)

        # Game info card
        info_card = QFrame()
        info_card.setObjectName('card')
        _glass_shadow(info_card)
        info_lay = QVBoxLayout(info_card)
        info_title = QLabel(t('Game Info', '遊戲資訊'))
        info_title.setObjectName('cardTitle')
        info_lay.addWidget(info_title)
        self.info_game = QLabel('')
        self.info_game.setObjectName('muted')
        self.info_mode = QLabel('')
        self.info_mode.setObjectName('muted')
        self.info_x = QLabel('')
        self.info_x.setWordWrap(True)
        self.info_o = QLabel('')
        self.info_o.setWordWrap(True)
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
        _glass_shadow(az_card)
        az_lay = QVBoxLayout(az_card)
        az_title = QLabel(t('Best Moves', '最佳棋步'))
        az_title.setObjectName('cardTitle')
        az_lay.addWidget(az_title)
        self.win_bar = QWidget()
        self.win_bar.setFixedHeight(14)
        self.win_bar_lay = QHBoxLayout(self.win_bar)
        self.win_bar_lay.setContentsMargins(0, 0, 0, 0)
        self.win_bar_lay.setSpacing(0)
        self.bar_x = QFrame()
        self.bar_x.setStyleSheet('background: #6750A4; border-radius: 4px 0 0 4px;')
        self.bar_d = QFrame()
        self.bar_d.setStyleSheet('background: #938F99;')
        self.bar_o = QFrame()
        self.bar_o.setStyleSheet('background: #B3261E; border-radius: 0 4px 4px 0;')
        for _b in (self.bar_x, self.bar_d, self.bar_o):
            _b.setMinimumWidth(0)
        self.win_bar_lay.addWidget(self.bar_x, 1)
        self.win_bar_lay.addWidget(self.bar_d, 1)
        self.win_bar_lay.addWidget(self.bar_o, 1)
        az_lay.addWidget(self.win_bar)
        self.analysis_pct = QLabel('')
        self.analysis_pct.setStyleSheet('font-size: 13px; color: #CAC4D0;')
        az_lay.addWidget(self.analysis_pct)
        self.analysis_list = QListWidget()
        self.analysis_list.setMinimumHeight(160)
        self.analysis_list.itemClicked.connect(self.on_analysis_clicked)
        az_lay.addWidget(self.analysis_list)
        hint = QLabel(t('Click a move to highlight it', '點擊棋步可在棋盤上標示'))
        hint.setObjectName('muted')
        az_lay.addWidget(hint)
        panel.addWidget(az_card, 1)

        # CvC controls card
        self.cvc_card = QFrame()
        self.cvc_card.setObjectName('card')
        _glass_shadow(self.cvc_card)
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
        panel.addWidget(self.cvc_card)

        panel_widget = QWidget()
        panel_widget.setObjectName('sidePanel')
        _glass_shadow(panel_widget)
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(340)
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
        self.play_again_btn.setVisible(False)
        self.new_btn.setVisible(True)
        s['moves'] = []
        s['history'] = [tuple(position_win_rates(self.game, self._history_budget()))]
        s['cvc_paused'] = False
        self.step_idx = 0
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
        self._render_history()
        self.refresh()
        self.after_move()

    def _history_budget(self):
        if isinstance(self.game, NormalGame):
            return 0
        return min(self.session.get('mcts', 800), 250)

    def go_to_step(self, k):
        """Jump to history step k (0..N) by replaying moves[:k]."""
        s = self.session
        moves = s['moves']
        k = max(0, min(k, len(moves)))
        if k == self.step_idx:
            return
        self.cvc_timer.stop()
        self.pvc_timer.stop()
        s['cvc_paused'] = k < len(moves)
        self.game = (NormalGame() if s['game_type'] == 'normal' else UltimateGame())
        for mv in moves[:k]:
            apply_move(self.game, mv)
        s['game'] = self.game
        self.step_idx = k
        self.busy = False
        self.gen += 1
        self.board.set_game(self.game)
        self._render_history()
        self.refresh()
        self.trigger_analysis()
        self.update_cvc_controls()
        if (not s.get('cvc_paused', False) and s['mode'] == 'cvc'
                and s.get('cvc_auto', True) and not self.game.is_over()
                and is_ai_turn(self.session)):
            self.cvc_timer.start()

    def _render_history(self):
        s = self.session
        moves = s.get('moves', [])
        history = s.get('history', [])
        n = len(moves)
        self.hist_slider.blockSignals(True)
        self.hist_slider.setRange(0, n)
        self.hist_slider.setValue(self.step_idx)
        self.hist_slider.blockSignals(False)
        self.hist_label.setText(f'{self.step_idx} / {n}')
        self.hist_x_series.clear()
        self.hist_ref.clear()
        self.hist_dot.clear()
        for k, (x, d, o) in enumerate(history):
            self.hist_x_series.append(k, x * 100.0)
            self.hist_ref.append(k, 50.0)
        if history:
            self.hist_dot.append(self.step_idx, history[self.step_idx][0] * 100.0)
            self.hist_axis.setRange(0, max(1, len(history) - 1))
            self.hist_axis_y.setRange(0, 100)

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
            moves = self.session.get('moves', [])
            if self.step_idx < len(moves):
                self.status_text.setText(
                    f'{self.status_text.text()} · {t("history", "歷史")} '
                    f'{self.step_idx}/{len(moves)}')
        self.status_mark.setStyleSheet(
            'color: ' + (PAL['x'] if result == X else PAL['o']) + '; font-size: 20px; font-weight: 700;')
        game_text = 'Normal' if isinstance(self.game, NormalGame) else 'Ultimate'
        mode_text = {'pvp': 'PvP', 'pvc': 'PvC', 'cvc': 'CvC'}[self.session['mode']]
        x_type, o_type = side_types(self.session)
        self.info_game.setText(f'{game_text} · {mode_text}')
        self.info_x.setText(f'✕ {side_label(x_type)}')
        self.info_o.setText(f'○ {side_label(o_type)}')
        self.board.update()
        self.update_cvc_controls()

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
        s = self.session
        moves = s['moves']
        if self.step_idx < len(moves):  # rewound: branch the history here
            del moves[self.step_idx:]
            del s['history'][self.step_idx + 1:]
        moves.append(move)
        s['history'].append(tuple(position_win_rates(self.game, self._history_budget())))
        self.step_idx = len(moves)
        self.gen += 1
        self.board.set_game(self.game)
        self._render_history()
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
                if (self.session.get('cvc_auto', True)
                        and not self.session.get('cvc_paused', False)):
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
        if self.session.get('cvc_paused', False):
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
        if (checked and self.session['mode'] == 'cvc' and is_ai_turn(self.session)
                and not self.session.get('cvc_paused', False)):
            self.cvc_timer.start()

    def on_revert_clicked(self):
        self.go_to_step(self.step_idx - 1)

    def on_pause_clicked(self):
        s = self.session
        s['cvc_paused'] = not s.get('cvc_paused', False)
        if s['cvc_paused']:
            self.cvc_timer.stop()
        else:
            if (s['mode'] == 'cvc' and not self.game.is_over()
                    and is_ai_turn(self.session) and s.get('cvc_auto', True)):
                self.cvc_timer.start()
        self.update_cvc_controls()

    def _update_pause_icon(self):
        paused = self.session.get('cvc_paused', False)
        self.pause_btn.setIcon(self.style().standardIcon(
            QStyle.SP_MediaPlay if paused else QStyle.SP_MediaPause))
        self.pause_btn.setToolTip(
            t('Resume', '繼續') if paused else t('Pause', '暫停'))

    def on_step_clicked(self):
        if self.step_idx < len(self.session['moves']):
            self.go_to_step(self.step_idx + 1)
        else:
            self.run_ai()

    def update_cvc_controls(self):
        cvc = self.session is not None and self.session['mode'] == 'cvc'
        self.cvc_card.setVisible(cvc)
        self.pause_btn.setVisible(cvc)
        self.next_btn.setVisible(cvc)
        if self.game is None:
            return
        rewound = self.step_idx < len(self.session.get('moves', []))
        ai_turn = not self.game.is_over() and is_ai_turn(self.session)
        auto = self.session.get('cvc_auto', True)
        paused = self.session.get('cvc_paused', False)
        self.revert_btn.setEnabled(self.step_idx > 0)
        self.pause_btn.setEnabled(True)
        self._update_pause_icon()
        self.next_btn.setEnabled(rewound or (ai_turn and not self.busy))

    # -- assistant ---------------------------------------------------------

    def on_assistant_toggled(self, checked):
        self.session['assistant_enabled'] = checked
        if checked:
            self.trigger_analysis()
        else:
            self.analysis_list.clear()
            self.win_bar.setVisible(False)
            self.analysis_pct.setText('—')
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
        worker.result_ready.connect(
            lambda items, rates, g=gen: self.on_analysis_done(items, rates, g))
        worker.finished.connect(lambda w=worker: self._reap(w))
        self.workers.append(worker)
        worker.start()

    def on_analysis_done(self, items, rates, gen):
        self.analysis_busy = False
        if gen == self.gen and self.session.get('assistant_enabled', True):
            self.render_analysis(items, rates)
        if self.analysis_pending:
            self.analysis_pending = False
            self.trigger_analysis()

    def render_analysis(self, items, rates):
        self.analysis_list.clear()
        x, d, o = rates
        self.win_bar.setVisible(True)
        self.win_bar_lay.setStretchFactor(self.bar_x, int(x * 1000))
        self.win_bar_lay.setStretchFactor(self.bar_d, int(d * 1000))
        self.win_bar_lay.setStretchFactor(self.bar_o, int(o * 1000))
        self.analysis_pct.setText(f'X {x:.0%} · 和 {d:.0%} · O {o:.0%}')
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
        log.info('Game over: %s [%s]', result,
                 'Normal' if isinstance(self.game, NormalGame) else 'Ultimate')
        # No popup: a prominent "Play Again" button replaces New Game.
        self.play_again_btn.setVisible(True)
        self.new_btn.setVisible(False)

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
    app.setFont(QFont(FONT_FAMILIES, 10))
    _init_siui_runtime()
    window = MainWindow(web_enabled=args.web, port=args.port)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
