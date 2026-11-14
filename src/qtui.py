# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""PySide6 desktop UI for SBA (Basically Awful).

Shares the game rules (game.py) and the AI engines (ai.py) with the TUI.

Run:
    python qtui.py            # desktop app

Maintenance notes:
- Fonts prefer Noto Sans TC; 微软雅黑 is not installed on every system (D7).
- Won chunks fill with the winner's color; hovering reveals the cells (D6).
"""

import argparse
import logging
import math
import os
import sys
import time

try:
    from PySide6.QtCore import (QEasingCurve, QPropertyAnimation, QPointF, QRectF,
                                QSettings, QSize, Qt, QThread, QTimer,
                                QVariantAnimation, QAbstractAnimation, Signal)
    from PySide6.QtGui import (QBrush, QColor, QFont, QFontDatabase, QFontMetrics,
                               QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap,
                               QPolygonF)
    from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
        QGraphicsBlurEffect, QGraphicsOpacityEffect, QGraphicsPixmapItem,
        QGraphicsScene, QListWidget, QListWidgetItem, QMainWindow, QProgressBar,
        QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
        QStackedWidget, QStyle, QVBoxLayout, QWidget,
    )
except ImportError:
    print('PySide6 is not installed for this Python interpreter.')
    print('Install the desktop UI with:  python -m pip install -r requirements.txt')
    raise SystemExit(1)

from game import (
    X, O,
    NormalGame, UltimateGame, apply_move, micro_win_line,
)
from ai import (get_ai_move, analyze_position, position_win_rates, move_text,
                cfg_engine, cfg_session, save_engine_config, _CFG_DEFAULTS as AI_DEFAULTS)
from SBA import (
    AI_OPTIONS, REASON_TEXT, current_side_type, is_ai_turn, log,
    new_session, side_label, side_types, t,
)


# Font stack: HarmonyOS Sans / Noto Sans TC when installed; Segoe UI/雅黑 are
# fallbacks so CJK and Latin both render cleanly.
FONT_FAMILIES = ['HarmonyOS Sans TC', 'HarmonyOS Sans SC', 'Noto Sans TC',
                 'Noto Sans SC', 'Microsoft YaHei', 'Segoe UI']

# Arknights Endfield palette: signal orange for win outlines/masks/labels
# (was olive green; orange separates from the yellow O-side accent).
WIN_ACCENT = '#E8721C'


def _blur_pixmap(pm, radius):
    """Return a blurred copy of ``pm`` (frosted-glass backdrop)."""
    if radius <= 0:
        return pm
    scene = QGraphicsScene()
    item = QGraphicsPixmapItem(pm)
    blur = QGraphicsBlurEffect()
    blur.setBlurRadius(radius)
    item.setGraphicsEffect(blur)
    scene.addItem(item)
    out = QImage(pm.size(), QImage.Format_ARGB32_Premultiplied)
    out.fill(Qt.transparent)
    rp = QPainter(out)
    scene.render(rp, QRectF(), QRectF(0, 0, pm.width(), pm.height()))
    rp.end()
    return QPixmap.fromImage(out)

# Role-based typography (D35 review, mirrors the archive reference):
#   DISPLAY — Inter Black/ExtraBold, tight -0.03em tracking, uppercase:
#             brand blocks, giant watermark numerals. The mass carries it.
#   MONO    — JetBrains Mono 500, +0.05em, 12px uppercase: every field
#             label, clock readout, button. The machine voice.
#   BODY    — FONT_FAMILIES (HarmonyOS/Noto/雅黑) for CJK and prose.
DISPLAY_FAMILIES = ['Inter', 'Segoe UI']
MONO_FAMILIES = ['JetBrains Mono', 'Cascadia Mono', 'Consolas', 'Courier New']

# repo root: the modules live in src/, tracked assets (static/) stay there
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_FONT_LOADED = False


def load_app_font():
    """Register the bundled fonts once (needs a QApplication)."""
    global _FONT_LOADED
    if _FONT_LOADED:
        return
    _FONT_LOADED = True
    base = os.path.join(ROOT, 'static', 'fonts')
    for name in ('Inter-Regular.ttf', 'Inter-Bold.ttf', 'Inter-ExtraBold.ttf',
                 'Inter-Black.ttf', 'JetBrainsMono-Medium.ttf',
                 'JetBrainsMono-Bold.ttf'):
        path = os.path.join(base, name)
        if os.path.isfile(path):
            if QFontDatabase.addApplicationFont(path) < 0:
                log.warning('font failed to load from %s', path)


def _first_available(fams):
    available = set(QFontDatabase.families())
    for fam in fams:
        if fam in available:
            return QFont(fam)
    return QFont()


def _display_font(size):
    """Inter Black display role — mass, tight tracking, no synthetic bold."""
    f = _first_available(DISPLAY_FAMILIES)
    f.setPixelSize(size)
    f.setWeight(QFont.Black)
    # negative tracking: the Inter-900 dense-block look (archive reference)
    f.setLetterSpacing(QFont.AbsoluteSpacing, round(size * -0.03, 2))
    return f


def _mono_font(size, weight=QFont.DemiBold):
    """JetBrains Mono role — every field label, readout and button."""
    f = _first_available(MONO_FAMILIES)
    f.setPixelSize(size)
    f.setWeight(weight)
    f.setLetterSpacing(QFont.AbsoluteSpacing, max(0.5, round(size * 0.05, 2)))
    return f


def anim_enabled():
    """Master switch for non-essential animation (Reduce Motion checkbox)."""
    return not QSettings('SBA', 'SBA').value('reduce_motion', False, type=bool)


def deck_label(text, size=12, accent=False):
    """Field-console label: JetBrains Mono, uppercase, +0.05em (D35)."""
    lab = QLabel(text.upper() if text.isascii() else text)
    font = _mono_font(size, QFont.Medium)
    lab.setFont(font)
    lab.setObjectName('deckAccent' if accent else 'deck')
    # Inline sheet: keeps the per-call size above the app-wide 16px rule.
    lab.setStyleSheet(f'font-size: {size}px;')
    return lab


def deck_num(text):
    """Oversized engineering section numeral (background watermark)."""
    lab = QLabel(text)
    font = _display_font(36)
    lab.setFont(font)
    lab.setObjectName('deckNum')
    lab.setStyleSheet('font-size: 36px;')
    return lab


def make_stripe():
    """Endfield magenta/green/yellow technical marker (one per screen)."""
    stripe = QWidget()
    stripe.setFixedHeight(3)
    lay = QHBoxLayout(stripe)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    for name in ('stripeM', 'stripeG', 'stripeY'):
        seg = QFrame()
        seg.setObjectName(name)
        seg.setFixedSize(36, 3)
        lay.addWidget(seg)
    lay.addStretch(1)
    return stripe


def _media_icon(kind, color='#F7F7F7', size=20):
    """Self-drawn transport glyph (back/pause/play/next) — Qt's standard
    icons ignore the QSS text color and render near-black on the dark
    theme (user feedback: "black icon in black background")."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    w, h = size, size
    if kind == 'pause':
        bw = max(3, size // 5)
        p.drawRect(int(w * 0.25), int(h * 0.2), bw, int(h * 0.6))
        p.drawRect(int(w * 0.25) + bw + max(2, size // 10), int(h * 0.2),
                   bw, int(h * 0.6))
    elif kind == 'play':
        p.drawPolygon([QPointF(w * 0.32, h * 0.2), QPointF(w * 0.82, h * 0.5),
                       QPointF(w * 0.32, h * 0.8)])
    elif kind == 'back':
        p.drawPolygon([QPointF(w * 0.68, h * 0.2), QPointF(w * 0.18, h * 0.5),
                       QPointF(w * 0.68, h * 0.8)])
    elif kind == 'next':
        for dx in (0.0, 0.28):
            p.drawPolygon([QPointF(w * (0.22 + dx), h * 0.2),
                           QPointF(w * (0.52 + dx), h * 0.5),
                           QPointF(w * (0.22 + dx), h * 0.8)])
    p.end()
    return QIcon(pm)


# D33: Go-style clock presets — main seconds, byo-yomi seconds per move,
# and the number of byo-yomi periods before a timeout loses the game.
CLOCK_PRESETS = {
    'off': None,
    'blitz': {'main': 180.0, 'byo_time': 10.0, 'periods': 3},
    'rapid': {'main': 300.0, 'byo_time': 20.0, 'periods': 3},
    'classical': {'main': 600.0, 'byo_time': 30.0, 'periods': 5},
}


# Board palettes. BoardWidget paints from the active palette; the Endfield
# theme (D23) is light-on-dark charcoal with signal yellow for the O side.
PALETTE = {
    'LIGHT': {
        'x': '#242424',
        'o': '#A29A00',
        'cell_empty': '#ECEDEE',
        'cell_filled': '#DFE1E3',
        'macro_active': 'rgba(162, 154, 0, 0.08)',
        'active_outline': '#8A8400',
        'macro_won_x': 'rgba(36, 36, 36, 0.10)',
        'macro_won_o': 'rgba(162, 154, 0, 0.10)',
        'grid_line': '#C6C8CC',
        'macro_line': '#8A8D91',
        'flash': '#8A8400',
        'win_fill_x': '#242424',
        'win_fill_o': '#A29A00',
        'win_mark': '#FFFFFF',
    },
    'DARK': {
        'x': '#F7F7F7',
        'o': '#FFFA00',
        'cell_empty': '#0A0A0C',
        'cell_filled': '#121317',
        'macro_active': 'rgba(255, 250, 0, 0.06)',
        'active_outline': '#FFFA00',
        'macro_won_x': 'rgba(247, 247, 247, 0.09)',
        'macro_won_o': 'rgba(255, 250, 0, 0.09)',
        'grid_line': '#232529',
        'macro_line': '#35373C',
        'flash': '#FFFA00',
        'win_fill_x': '#F7F7F7',
        'win_fill_o': '#FFFA00',
        'win_mark': '#09090B',
    },
}
ACTIVE_THEME = 'DARK'  # Endfield field-console look; LIGHT is for future use
PAL = PALETTE[ACTIVE_THEME]


# ---------------------------------------------------------------------------
# Combo factory (plain QComboBox; the theme does the rest via QSS)
# ---------------------------------------------------------------------------

def _deck_combo(parent=None):
    return QComboBox(parent)


class ChamferButton(QPushButton):
    """Endfield button (D34/D35): self-painted block. Hover/press wipes
    signal yellow in from the left (mech curve), disabled fills with 45°
    hatch. Fully self-painted (no QSS).

    The trapezoid slant is NOT a per-button decoration — it is reserved for
    two adjacent buttons interlocking as a pair with a parallel slanted
    seam: the left piece slants its right edge (slant='r'), the right piece
    its left edge (slant='l'). Everything else is a plain rectangle.

    kind: 'default' graphite block · 'primary' signal border + signal text ·
          'ghost' quiet variant for secondary navigation.
    """

    _MECH = None  # lazily built bezier (0.7, 0, 0.2, 1)

    def __init__(self, text='', parent=None, kind='default', slant=None):
        super().__init__(text, parent)
        self.kind = kind
        self.slant = slant
        self._wipe = 0.0
        self._wipe_anim = QVariantAnimation(self)
        self._wipe_anim.setDuration(150)
        self._wipe_anim.setStartValue(0.0)
        self._wipe_anim.setEndValue(1.0)
        if ChamferButton._MECH is None:
            ec = QEasingCurve(QEasingCurve.BezierSpline)
            ec.addCubicBezierSegment(QPointF(0.7, 0.0), QPointF(0.2, 1.0),
                                     QPointF(1.0, 1.0))
            ChamferButton._MECH = ec
        self._wipe_anim.setEasingCurve(ChamferButton._MECH)
        self._wipe_anim.valueChanged.connect(self._on_wipe)
        self.setCursor(Qt.PointingHandCursor)
        # mono role (D35 review): JetBrains Mono 13px DemiBold on the widget
        # itself, so _deck_button's QFontMetrics measure what paint draws
        f = QFont()
        f.setFamilies(MONO_FAMILIES + FONT_FAMILIES)
        f.setPixelSize(13)
        f.setWeight(QFont.DemiBold)
        self.setFont(f)

    def _on_wipe(self, v):
        self._wipe = float(v)
        self.update()

    def enterEvent(self, e):
        if self.isEnabled() and self.kind != 'ghost':
            self._wipe_anim.stop()
            self._wipe_anim.setStartValue(self._wipe)
            self._wipe_anim.setEndValue(1.0)
            self._wipe_anim.start()
        super().enterEvent(e)

    def leaveEvent(self, e):
        if self.isEnabled() and self.kind != 'ghost':
            self._wipe_anim.stop()
            self._wipe_anim.setStartValue(self._wipe)
            self._wipe_anim.setEndValue(0.0)
            self._wipe_anim.start()
        super().leaveEvent(e)

    def _hatch(self, p, rect):
        """45-degree ink hatch — the disabled fill (endfield-ui token)."""
        p.save()
        p.setClipRect(rect)
        p.setPen(QPen(QColor('#3a3a3a'), 1))
        x = rect.left() - rect.height()
        while x < rect.right():
            p.drawLine(int(x), int(rect.bottom()), int(x + rect.height()),
                       int(rect.top()))
            x += 6
        p.restore()

    def paintEvent(self, e):
        p = QPainter(self)
        w, h = self.width(), self.height()
        ch = min(14, h // 3)
        wipe = 1.0 if self.isDown() else self._wipe
        path = QPainterPath()
        if self.slant == 'r':       # pair left piece: right edge leans '/'
            path.moveTo(0, 0)
            path.lineTo(w, 0)
            path.lineTo(w - ch, h)
            path.lineTo(0, h)
        elif self.slant == 'l':     # pair right piece: left edge leans '/'
            path.moveTo(ch, 0)
            path.lineTo(w, 0)
            path.lineTo(w, h)
            path.lineTo(0, h)
        else:
            path.addRect(QRectF(0, 0, w, h))
        path.closeSubpath()
        p.setRenderHint(QPainter.Antialiasing, True)

        if not self.isEnabled():
            p.setPen(QPen(QColor('#2a2a2a'), 1))
            p.setBrush(QColor('#1c1c1c'))
            p.drawPath(path)
            self._hatch(p, path.boundingRect())
            p.setPen(QColor('#6a6a6a'))
            p.setFont(self.font())
            p.drawText(self.rect(), Qt.AlignCenter, self.text())
            p.end()
            return

        bg = QColor('#161616') if self.kind == 'ghost' else QColor('#222222')
        p.setPen(QPen(QColor('#353535'), 1))
        p.setBrush(bg)
        p.drawPath(path)

        if self.kind == 'primary':
            p.setPen(QPen(QColor('#FFFA00'), 1))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        if wipe > 0.0:
            p.save()
            p.setClipRect(QRectF(0, 0, w * wipe, h))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor('#FFFA00'))
            p.drawPath(path)
            p.restore()

        text_color = '#161616' if wipe > 0.55 else (
            '#FFFA00' if self.kind == 'primary' else '#EDEDED')
        if self.kind == 'ghost':
            text_color = '#EDEDED'
        if self.hasFocus():
            # QPainterPath has no adjusted(); a 2px signal pen IS the focus ring
            p.setPen(QPen(QColor('#FFFA00'), 2))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        p.setPen(QColor(text_color))
        p.setFont(self.font())
        p.drawText(self.rect(), Qt.AlignCenter, self.text())
        if self.icon() is not None and not self.icon().isNull():
            # QIcon.paint rejects QRectF; only QRect or int x/y/w/h overloads exist
            self.icon().paint(p, int((w - 20) / 2), int((h - 20) / 2), 20, 20)
        p.end()


def _deck_button(text, parent=None, primary=False, slant=None):
    btn = ChamferButton(text, parent, kind='primary' if primary else 'default',
                        slant=slant)
    fm = QFontMetrics(btn.font())
    pad = 48 + (14 if slant else 0)   # slanted pair pieces need seam clearance
    btn.setMinimumSize(max(140, fm.horizontalAdvance(text) + pad), 40)
    return btn





# ---------------------------------------------------------------------------
# Endfield field-engineering theme (D23). The full styling lives in
# static/endfield.qss so visuals can be retuned without touching logic; only
# a tiny fallback is embedded for a missing file.
# ---------------------------------------------------------------------------
_QSS_PATH = os.path.join(ROOT, 'static', 'endfield.qss')
_QSS_FALLBACK = (
    'QWidget { background: #09090b; color: #ffffff; }'
    'QPushButton#primary { background: #fffa00; color: #09090b; }'
)


def _load_qss():
    try:
        with open(_QSS_PATH, encoding='utf-8') as fh:
            return fh.read()
    except OSError:
        log.warning('endfield.qss not found at %s; using inline fallback',
                    _QSS_PATH)
        return _QSS_FALLBACK


QSS = _load_qss()


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
        # mask fade-in: a newly won chunk eases its mask in instead of
        # popping (history jumps / new games set every blend to 1.0)
        self._mask_blend = {}
        self._mask_anims = {}
        self._won_cache = set()
        # D31: stone "stamp" press + progressive win-line draw
        self._stamp_cell = None
        self._stamp_t = 1.0
        self._stamp_anim = QVariantAnimation(self)
        self._stamp_anim.setDuration(170)
        self._stamp_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._stamp_anim.setStartValue(0.0)
        self._stamp_anim.setEndValue(1.0)
        self._stamp_anim.valueChanged.connect(self._on_stamp_anim)
        self._stamp_anim.finished.connect(self._on_stamp_done)
        self._win_t = 1.0
        self._win_anim = QVariantAnimation(self)
        self._win_anim.setDuration(260)
        self._win_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._win_anim.setStartValue(0.0)
        self._win_anim.setEndValue(1.0)
        self._win_anim.valueChanged.connect(self._on_win_anim)
        self.setMinimumSize(300, 300)

    def _on_stamp_anim(self, v):
        self._stamp_t = float(v)
        self.update()

    def _on_stamp_done(self):
        self._stamp_cell = None
        self._stamp_t = 1.0
        self.update()

    def animate_move(self, move):
        """Stamp-press animation for a freshly placed stone (D31); skipped
        entirely when Reduce Motion is on."""
        if not anim_enabled():
            return
        self._stamp_anim.stop()
        self._stamp_cell = move
        self._stamp_t = 0.0
        self._stamp_anim.start()

    def animate_win_line(self):
        """Progressive draw-in of the game-ending strike line (D31)."""
        if not anim_enabled():
            self._win_t = 1.0
            self.update()
            return
        self._win_anim.stop()
        self._win_t = 0.0
        self._win_anim.start()

    def _on_win_anim(self, v):
        self._win_t = float(v)
        self.update()

    def set_game(self, game):
        prev_won = self._won_cache
        self.game = game
        self.legal = set(game.legal_moves())
        self.flash_cell = None
        self._stamp_cell = None   # a new position invalidates a running stamp
        if isinstance(game, UltimateGame):
            won = {m for m in range(9) if game.macro[m] in (X, O)}
            fresh = won - prev_won
            if len(fresh) == 1:
                # natural progression: exactly one chunk flipped — fade it in
                for m in won:
                    self._finish_mask(m)
                self._start_mask_fade(next(iter(fresh)))
            else:
                # history jump / new game: every mask appears instantly
                for m in won:
                    self._finish_mask(m)
            self._won_cache = won
        else:
            self._won_cache = set()
        self.update()

    def _start_mask_fade(self, m):
        if not anim_enabled():
            self._mask_blend[m] = 1.0
            return
        anim = self._mask_anims.get(m)
        if anim is None:
            anim = QVariantAnimation(self)
            anim.setDuration(240)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            anim.valueChanged.connect(
                lambda v, mm=m: self._on_mask_anim(mm, v))
            self._mask_anims[m] = anim
        self._mask_blend[m] = 0.0
        anim.stop()
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.start()

    def _on_mask_anim(self, m, v):
        self._mask_blend[m] = float(v)
        self.update()

    def _finish_mask(self, m):
        anim = self._mask_anims.pop(m, None)
        if anim is not None:
            anim.stop()
        self._mask_blend[m] = 1.0

    def reset_reveal(self):
        """Clear the won-chunk reveal (used on new game / history jumps)."""
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
        end = 1.0 if self._revealed_macro is not None else 0.0
        self._reveal_anim.stop()
        if self._reveal_blend == end:
            # Switching directly between two revealed chunks: the blend does
            # not move, and QVariantAnimation emits nothing when start==end,
            # so the repaint must happen here or the new chunk stays masked
            # until the next toggle ("close the old mask first" bug).
            self.update()
            return
        self._reveal_anim.setStartValue(self._reveal_blend)
        self._reveal_anim.setEndValue(end)
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
        # Opaque plate sized to the board only (not the whole widget): the
        # backdrop must not bleed through the 3px cell gaps, but the margins
        # around the board stay transparent so the page reads as one canvas.
        n, cell, ox, oy = self._grid()
        span = n * cell + (n - 1) * self.GAP
        plate_rect = QRectF(ox, oy, span, span).adjusted(-8.0, -8.0, 8.0, 8.0)
        painter.fillRect(plate_rect, QColor('#0D0E10'))
        painter.setPen(QPen(QColor('#35373C'), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(plate_rect.adjusted(0, 0, -1, -1))
        if isinstance(self.game, UltimateGame):
            self._paint_ultimate(painter)
        else:
            self._paint_normal(painter)

    # -- shared helpers -----------------------------------------------------

    def _paint_cell(self, painter, rect, mark, is_flash, cell_id=None):
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(PAL['cell_filled'] if mark else PAL['cell_empty']))
        painter.drawRect(rect)
        if mark in (X, O):
            stamping = (cell_id is not None and cell_id == self._stamp_cell
                        and self._stamp_t < 1.0)
            if stamping:
                # D31 stamp press: settle from 1.25x down onto the cell while
                # fading in — mechanical, sharp, no bounce.
                t = self._stamp_t
                s = 1.25 - 0.25 * t
                painter.save()
                painter.setOpacity(min(1.0, 0.25 + 0.75 * t))
                cx, cy = rect.center().x(), rect.center().y()
                half = rect.width() / 2.0 * s
                srect = QRectF(cx - half, cy - half, half * 2.0, half * 2.0)
                self._paint_mark(painter, srect, mark)
                painter.restore()
            else:
                self._paint_mark(painter, rect, mark)
        if is_flash:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(PAL['flash']), 3))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))

    def _paint_mark(self, painter, rect, player, color=None):
        pad = rect.width() * 0.22
        color = _mark_color(player) if color is None else QColor(color)
        pen = QPen(color, max(2.0, rect.width() * 0.09))
        pen.setCapStyle(Qt.SquareCap)
        pen.setJoinStyle(Qt.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        if player == X:
            painter.drawLine(rect.left() + pad, rect.top() + pad,
                             rect.right() - pad, rect.bottom() - pad)
            painter.drawLine(rect.right() - pad, rect.top() + pad,
                             rect.left() + pad, rect.bottom() - pad)
        else:
            # Octagonal ring instead of a circle: mechanical Endfield emblem,
            # and the shape alone separates O from the straight-stroke X.
            cx, cy = rect.center().x(), rect.center().y()
            radius = rect.width() / 2.0 - pad
            poly = QPolygonF()
            for k in range(8):
                ang = math.radians(22.5 + 45.0 * k)
                poly.append(QPointF(cx + radius * math.cos(ang),
                                    cy + radius * math.sin(ang)))
            painter.drawPolygon(poly)

    def _win_segment(self, line, centers, bounds):
        """Extend a win line across `bounds`."""
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

    def _paint_win_line(self, painter, line, centers, color, width, bounds,
                        progress=1.0):
        if line is None:
            return
        (x1, y1), (x2, y2) = self._win_segment(line, centers, bounds)
        x2 = x1 + (x2 - x1) * progress
        y2 = y1 + (y2 - y1) * progress
        pen = QPen(QColor(color), width)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(x1, y1, x2, y2)

    def _draw_brackets(self, painter, rect, color, width=2.0, tick=10.0):
        """Targeting brackets (four L-shaped corner ticks) around a rect."""
        pen = QPen(color, width)
        pen.setCapStyle(Qt.SquareCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
        corners = (
            ((x0, y0 + tick), (x0, y0), (x0 + tick, y0)),
            ((x1 - tick, y0), (x1, y0), (x1, y0 + tick)),
            ((x0, y1 - tick), (x0, y1), (x0 + tick, y1)),
            ((x1 - tick, y1), (x1, y1), (x1, y1 - tick)),
        )
        for (ax, ay), (bx, by), (cx, cy) in corners:
            painter.drawLine(ax, ay, bx, by)
            painter.drawLine(bx, by, cx, cy)

    # -- Normal ------------------------------------------------------------

    def _paint_normal(self, painter):
        n, cell, ox, oy = self._grid()
        centers = []
        for i in range(9):
            r, c = divmod(i, 3)
            rect = self._cell_rect(r, c, cell, ox, oy)
            self._paint_cell(painter, rect, self.game.board[i],
                             self.flash_cell == i, cell_id=i)
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
                                 cell * 0.12, bounds, progress=self._win_t)

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
            painter.drawRect(rect)
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
                                 self.flash_cell == (m, i), cell_id=(m, i))
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
        # yellow targeting brackets on the chunk the player must play in
        # (Ultimate); drawn on top and slightly outside the chunk so cells
        # don't cover it
        if (self.game.active_macro is not None
                and self.game.macro_open(self.game.active_macro)):
            rect = self._macro_rect(self.game.active_macro, cell, ox, oy)
            self._draw_brackets(painter, rect.adjusted(-2.5, -2.5, 2.5, 2.5),
                                QColor(PAL['active_outline']), 2.5,
                                max(8.0, cell * 0.35))
        # macro-level winning line (the "connected" chunks) for the green tint
        whole = micro_win_line(self.game.macro)
        # won macro chunks: a flat, opaque field label (no frosted glass — the
        # blurred-under-tint composite read as dirty mottling), plus the big
        # winner badge; clicking fades it into the real cells
        for m in range(9):
            if self.game.macro[m] in (X, O):
                winner = self.game.macro[m]
                rect = self._macro_rect(m, cell, ox, oy)
                blend = self._reveal_blend if m == self._revealed_macro else 0.0
                mb = self._mask_blend.get(m, 1.0)   # win-moment fade-in
                if blend < 1.0:
                    # flat dark base so nothing bleeds through, then the tint
                    painter.setPen(Qt.NoPen)
                    base = QColor('#0D0E10')
                    base.setAlpha(int(255 * (1.0 - blend) * mb))
                    if base.alpha() > 0:
                        painter.setBrush(base)
                        painter.drawRect(rect)
                    connected = whole is not None and m in whole
                    if connected:
                        fill = QColor(WIN_ACCENT)
                    else:
                        fill = QColor(PAL['win_fill_x'] if winner == X
                                      else PAL['win_fill_o'])
                    fill.setAlpha(int(215 * (1.0 - blend) * mb))
                    if fill.alpha() > 0:
                        painter.setBrush(fill)
                        painter.drawRect(rect)
                    badge_color = QColor(PAL['win_mark'])  # dark ink on the
                    # colored plate — a yellow badge on a yellow O plate
                    # vanished; ink on signal colors stays readable
                    badge_color.setAlpha(int(255 * (1.0 - blend) * mb))
                    if badge_color.alpha() > 0:
                        pad = rect.width() * 0.18
                        badge = QRectF(rect.left() + pad, rect.top() + pad,
                                       rect.width() - 2 * pad,
                                       rect.height() - 2 * pad)
                        self._paint_mark(painter, badge, winner, badge_color)
                # green-highlight the connected winning X/O marks — shown only
                # after clicking (reveal)
                line = micro_win_line(self.game.micro[m])
                if line is not None and blend > 0:
                    conn = QColor(WIN_ACCENT)
                    conn.setAlpha(int(255 * blend))
                    mr, mc = divmod(m, 3)
                    for idx in line:
                        cmark = self.game.micro[m][idx]
                        row = mr * 3 + idx // 3
                        col = mc * 3 + idx % 3
                        crect = self._cell_rect(row, col, cell, ox, oy)
                        self._paint_mark(painter, crect, cmark, conn)
                # green outline marks the chunks that make the macro win line
                if whole is not None and m in whole:
                    edge = QColor(WIN_ACCENT)
                    edge.setAlpha(int(255 * mb))
                    painter.setPen(QPen(edge, max(2.0, cell * 0.06)))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(rect.adjusted(-1.0, -1.0, 1.0, 1.0))

# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class AIWorker(QThread):
    move_ready = Signal(object)
    move_failed = Signal(str)
    progress = Signal(dict)

    def __init__(self, game, ai_type, budget, depth, parent=None, trace=None):
        super().__init__(parent)
        self.game = game
        self.ai_type = ai_type
        self.budget = budget
        self.depth = depth
        self.trace = trace

    def run(self):
        try:
            move = get_ai_move(self.game, self.ai_type, self.budget, self.depth,
                               on_progress=self.progress.emit, trace=self.trace)
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
            items, rates = analyze_position(self.game, self.budget)
            self.result_ready.emit(items, rates)
        except Exception as e:  # noqa: BLE001
            log.error('Analysis failed: %s', e)
            self.result_ready.emit([], (0.5, 0.0, 0.5))


# ---------------------------------------------------------------------------
# Endfield backdrop: topographic survey imagery behind every page
# ---------------------------------------------------------------------------

class EndfieldPage(QWidget):
    """Shared Endfield backdrop for the menu/game pages.

    Subclasses point BACKDROP at a survey image under static/backgrounds/;
    it is drawn aspect-fill, center-cropped, and cached per class. When the
    image is missing the page falls back to procedural contour rings plus
    the giant hollow title, so the theme still works on a bare checkout.
    Children (cards, controls) draw on top.
    """

    BACKDROP = None      # e.g. 'static/backgrounds/topography-dark.jpg'
    GIANT_TEXT = None    # only drawn in the procedural fallback
    _backdrop_cache = {}

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pix = self._backdrop_pixmap()
        if pix is not None:
            self._paint_backdrop(painter, pix)
        else:
            self._paint_contours(painter)
            if self.GIANT_TEXT:
                self._paint_giant(painter)
        painter.end()

    def _backdrop_pixmap(self):
        """Class-cached, blurred backdrop; None when the file is missing.

        The survey images are sharp and fairly contrasty — blurred once at
        load (and cached) they read as soft field texture instead of competing
        with the board (user feedback: "opacity too high, try blurry or dark").
        """
        if not self.BACKDROP:
            return None
        pix = EndfieldPage._backdrop_cache.get(self.BACKDROP)
        if pix is None:
            path = os.path.join(ROOT, self.BACKDROP)
            if not os.path.isfile(path):
                return None
            pix = QPixmap(path)
            if pix.isNull():
                return None
            pix = _blur_pixmap(pix, 8.0)  # soften once; cached afterwards
            EndfieldPage._backdrop_cache[self.BACKDROP] = pix
        return pix

    def _paint_backdrop(self, painter, pix):
        """Aspect-fill, center-cropped, then a dark wash to keep it subdued."""
        iw, ih = pix.width(), pix.height()
        if iw < 1 or ih < 1:
            return
        w, h = self.width(), self.height()
        scale = max(w / iw, h / ih)
        dw, dh = iw * scale, ih * scale
        x, y = (w - dw) / 2.0, (h - dh) / 2.0
        painter.drawPixmap(QRectF(x, y, dw, dh).toRect(), pix)
        painter.fillRect(self.rect(), QColor(9, 9, 11, 168))

    def _paint_contours(self, painter):
        """endfield contour (D34): small open arcs in two corners only,
        ink at 9% — never closed rings across the page."""
        w, h = self.width(), self.height()
        if w < 200 or h < 200:
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        ink = QColor('#EDEDED')
        ink.setAlpha(23)
        painter.setPen(QPen(ink, 1))
        for cx, cy, start in ((0, 0, 0), (w, h, 1800)):
            for i in range(4):
                r = 70 + i * 34
                rect = QRectF(cx - r, cy - r, r * 2, r * 2)
                painter.drawArc(rect, start, 95 * 16)   # open, never closed

    def _paint_giant(self, painter):
        w, h = self.width(), self.height()
        font = _display_font(int(min(w, h) * 0.30))
        path = QPainterPath()
        path.addText(QPointF(w * 0.05, h * 0.62), font, self.GIANT_TEXT)
        painter.setPen(QPen(QColor(255, 255, 255, 22), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)


# ---------------------------------------------------------------------------
# Menu page
# ---------------------------------------------------------------------------

class MenuPage(EndfieldPage):
    start_requested = Signal(object)
    settings_requested = Signal()
    GIANT_TEXT = 'UTTT'
    BACKDROP = 'static/backgrounds/survey-dark.jpg'

    def __init__(self):
        super().__init__()
        self.setObjectName('pageMenu')
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        # D34: endfield pairing — wide-cap eyebrow on top, bold CN title below
        root.addWidget(deck_label('// ULTIMATE TIC TAC TOE', size=11))
        title = QLabel('終極井字棋')
        title.setObjectName('title')
        root.addWidget(title)
        # real numbers only (endfield-ui rule): engines from AI_OPTIONS
        root.addWidget(deck_label(
            f'{len(AI_OPTIONS)} ENGINES │ NATIVE C ENGINE │ 90 TESTS', size=12))
        root.addSpacing(2)
        root.addWidget(deck_label('// GAME SETUP — 遊戲設定', size=11, accent=True))
        card = QFrame()
        card.setObjectName('card')
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

        self.game_type = _deck_combo()
        self.game_type.addItem('Normal Tic Tac Toe (普通井字棋)', 'normal')
        self.game_type.addItem('Ultimate Tic Tac Toe (終極井字棋)', 'ultimate')
        card_lay.addWidget(field_row(t('Game Type', '遊戲類型'), self.game_type)[0])

        self.mode = _deck_combo()
        self.mode.addItem('PvP (玩家對玩家)', 'pvp')
        self.mode.addItem('Player vs Computer (玩家對電腦)', 'pvc')
        self.mode.addItem('Computer vs Computer (電腦對電腦)', 'cvc')
        self.mode.currentIndexChanged.connect(lambda _: self._update_visibility())
        card_lay.addWidget(field_row(t('Mode', '模式'), self.mode)[0])

        self.first = _deck_combo()
        self.first.addItem(t('You move first — X', '你先手 — X'), 'human')
        self.first.addItem(t('Computer moves first — O', '電腦先手 — O'), 'computer')
        self.first.currentIndexChanged.connect(lambda _: self._update_visibility())
        self.first_row, _ = field_row(t('First Player', '先手'), self.first)
        card_lay.addWidget(self.first_row)

        self.ai_x = _deck_combo()
        for key, label in AI_OPTIONS.items():
            self.ai_x.addItem(label, key)
        self.ai_x_row, _ = field_row(t('Player X — AI Level', '玩家 X — AI 等級'), self.ai_x)
        card_lay.addWidget(self.ai_x_row)

        self.ai_o = _deck_combo()
        for key, label in AI_OPTIONS.items():
            self.ai_o.addItem(label, key)
        self.ai_o_row, self.ai_o_label = field_row(t('Player O — AI Level', '玩家 O — AI 等級'), self.ai_o)
        card_lay.addWidget(self.ai_o_row)

        self.mcts = QSlider(Qt.Horizontal)
        self.mcts.setRange(200, 30000)  # the C engine makes big budgets affordable (D27)
        self.mcts.setSingleStep(100)
        self.mcts.setValue(800)
        card_lay.addWidget(field_row(t('MCTS Strength', 'MCTS 強度'),
                                     self._slider_row(self.mcts))[0])

        self.mm_depth = QSlider(Qt.Horizontal)
        self.mm_depth.setRange(2, 6)
        self.mm_depth.setValue(4)
        card_lay.addWidget(field_row(t('Minimax Depth (Ultimate)', 'Minimax 深度（終極模式）'),
                                     self._slider_row(self.mm_depth))[0])

        # D33: Go-style clock — main time, then byo-yomi periods (N × T s)
        self.clock_preset = _deck_combo()
        self.clock_preset.addItem(t('No clock', '不計時'), 'off')
        self.clock_preset.addItem(t('Blitz: 3 min + 3×10s', '快棋：3 分 + 3×10 秒'), 'blitz')
        self.clock_preset.addItem(t('Rapid: 5 min + 3×20s', '快速：5 分 + 3×20 秒'), 'rapid')
        self.clock_preset.addItem(t('Classical: 10 min + 5×30s', '古典：10 分 + 5×30 秒'), 'classical')
        card_lay.addWidget(field_row(t('Clock (main + byo-yomi)', '計時（主時限 + 讀秒）'),
                                     self.clock_preset)[0])

        self.assistant = QCheckBox(t('AI Assistant', 'AI 助手'))
        self.assistant.setChecked(True)
        card_lay.addWidget(self.assistant)

        self.reduce_anim = QCheckBox(
            t('Reduce Motion (no animations)', '減少動態（停用動畫）'))
        settings = QSettings('SBA', 'SBA')
        self.reduce_anim.setChecked(
            settings.value('reduce_motion', False, type=bool))
        self.reduce_anim.toggled.connect(
            lambda c: QSettings('SBA', 'SBA').setValue('reduce_motion', c))
        card_lay.addWidget(self.reduce_anim)

        card_lay.addSpacing(10)

        start_row = QHBoxLayout()
        start_row.setSpacing(0)   # the pair interlocks across a shared seam
        start_btn = _deck_button(t('Start Game', '開始遊戲'), primary=True,
                               slant='r')
        start_btn.clicked.connect(self._on_start)
        start_row.addWidget(start_btn, 1)
        settings_btn = _deck_button(t('Settings', '設定'), slant='l')
        settings_btn.setObjectName('ghost')
        settings_btn.clicked.connect(self.settings_requested.emit)
        start_row.addWidget(settings_btn)
        card_lay.addLayout(start_row)

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
        s['clock'] = CLOCK_PRESETS.get(self.clock_preset.currentData(), None)
        self.start_requested.emit(s)

# ---------------------------------------------------------------------------
# Game page
# ---------------------------------------------------------------------------

class GamePage(EndfieldPage):
    back_requested = Signal()
    BACKDROP = 'static/backgrounds/topography-dark.jpg'

    def __init__(self):
        super().__init__()
        self.setObjectName('pageGame')
        self.session = None
        self.game = None
        self.busy = False
        self.gen = 0
        self.game_id = 0
        self.analysis_busy = False
        self.analysis_pending = False
        self.workers = []
        self.step_idx = 0
        self.cvc_timer = QTimer(self)
        self.cvc_timer.timeout.connect(self.cvc_tick)
        self._bar_pos = (1.0, 0.0)   # last drawn X/O split (D31)
        self._bar_anim = QVariantAnimation(self)
        self._bar_anim.setDuration(250)
        self._bar_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._bar_anim.valueChanged.connect(self._on_bar_anim)
        # D33: Go-style clocks (main time + byo-yomi periods), per side
        self.clock = None            # {'X': {...}, 'O': {...}} when enabled
        self._time_loss = None       # side that lost on time
        self._clock_tick_t = None    # monotonic reference for the last tick
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._clock_tick)
        self.pvc_timer = QTimer(self)
        self.pvc_timer.setSingleShot(True)
        self.pvc_timer.timeout.connect(self.run_ai)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(make_stripe())   # tri-colour strip: top edge of the page

        # D34: single dark command bar (constant #161616) — brand block,
        # transport, step field, dual clock, actions. Replaces the old header
        # row + control bar + clock chip row (reference: audio-terminal nav).
        bar = QFrame()
        bar.setObjectName('commandBar')
        bar.setFixedHeight(64)
        top = QHBoxLayout(bar)
        top.setContentsMargins(14, 8, 14, 8)
        top.setSpacing(10)
        brand = QVBoxLayout()
        brand.setSpacing(1)
        wm = QLabel('UTTT')
        f = _display_font(19)
        wm.setFont(f)
        wm.setStyleSheet('color: #EDEDED;')
        brand.addWidget(wm)
        brand.addWidget(deck_label('終極井字棋 // ULTIMATE', size=10))
        top.addLayout(brand)
        top.addSpacing(14)

        self.revert_btn = ChamferButton()
        self.revert_btn.setIcon(_media_icon('back'))
        self.revert_btn.setToolTip(t('Revert', '回退'))
        self.revert_btn.setFixedSize(40, 34)
        self.revert_btn.clicked.connect(self.on_revert_clicked)
        top.addWidget(self.revert_btn)
        self.pause_btn = ChamferButton()
        self.pause_btn.setIcon(_media_icon('pause'))
        self.pause_btn.setToolTip(t('Pause', '暫停'))
        self.pause_btn.setFixedSize(40, 34)
        self.pause_btn.clicked.connect(self.on_pause_clicked)
        top.addWidget(self.pause_btn)
        self.next_btn = ChamferButton()
        self.next_btn.setIcon(_media_icon('next'))
        self.next_btn.setToolTip(t('Next Step', '下一步'))
        self.next_btn.setFixedSize(40, 34)
        self.next_btn.clicked.connect(self.on_step_clicked)
        top.addWidget(self.next_btn)
        top.addStretch(1)

        self.step_lbl = deck_label('STEP 00/00', size=12)
        top.addWidget(self.step_lbl)
        top.addSpacing(8)
        self.clock_lbl = QLabel('')
        cf = _mono_font(15)
        self.clock_lbl.setFont(cf)
        self.clock_lbl.hide()
        top.addWidget(self.clock_lbl)
        self.byo_tag = QLabel('')
        self.byo_tag.setStyleSheet(
            'background: #FFFA00; color: #161616; padding: 3px 8px;'
            'font-weight: 700;')
        self.byo_tag.hide()
        top.addWidget(self.byo_tag)
        top.addSpacing(8)

        self.new_btn = ChamferButton('新遊戲')
        self.new_btn.clicked.connect(self.new_game)
        top.addWidget(self.new_btn)
        self.play_again_btn = ChamferButton('再玩一次', kind='primary')
        self.play_again_btn.clicked.connect(self.new_game)
        self.play_again_btn.setVisible(False)
        top.addWidget(self.play_again_btn)
        back_btn = ChamferButton('選單', kind='ghost')
        back_btn.clicked.connect(self.back_requested.emit)
        top.addWidget(back_btn)
        root.addWidget(bar)

        # Win-rate chart (lives under the board, so the right panel spans the
        # full window height); single X-win-rate line with a 50% reference.
        self.hist_chart = QChart()
        self.hist_chart.setAnimationOptions(QChart.NoAnimation)
        self.hist_chart.legend().setVisible(False)  # the deck label carries it
        # end-of-game full-curve fit state (animated axis rescale)
        self._chart_fit = False
        self._chart_anim = QVariantAnimation(self)
        self._chart_anim.setDuration(700)
        self._chart_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._chart_anim.valueChanged.connect(self._on_chart_fit)
        self.hist_chart.setBackgroundVisible(False)
        self.hist_x_series = QLineSeries()
        self.hist_x_series.setName(t('X win rate', 'X 勝率'))
        self.hist_x_series.setColor(QColor(PAL['x']))
        self.hist_ref = QLineSeries()
        ref_pen = QPen(QColor('#585858'))
        ref_pen.setStyle(Qt.DashLine)
        ref_pen.setWidthF(1.0)
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
            ax.setLabelsColor(QColor('#C6C9CE'))
            ax.setTitleBrush(QBrush(QColor('#C6C9CE')))
            ax.setLinePen(QPen(QColor('#35373C')))
            ax.setGridLinePen(QPen(QColor(255, 255, 255, 12), 1, Qt.DashLine))
        self.hist_x_series.attachAxis(self.hist_axis)
        self.hist_x_series.attachAxis(self.hist_axis_y)
        self.hist_ref.attachAxis(self.hist_axis)
        self.hist_ref.attachAxis(self.hist_axis_y)
        self.hist_dot = QScatterSeries()
        self.hist_dot.setName(t('Current', '目前'))
        self.hist_dot.setMarkerSize(8.0)
        self.hist_dot.setMarkerShape(QScatterSeries.MarkerShapeRectangle)
        self.hist_dot.setColor(QColor('#FFFA00'))
        self.hist_dot.setBorderColor(QColor('#FFFA00'))
        self.hist_chart.addSeries(self.hist_dot)
        self.hist_dot.attachAxis(self.hist_axis)
        self.hist_dot.attachAxis(self.hist_axis_y)
        # Endfield look: one clean 2px line, no per-point markers (the dot
        # markers made the polyline read as "broken"), integer axis labels.
        self.hist_x_series.setPointsVisible(False)
        line_pen = QPen(QColor(PAL['x']), 2)
        line_pen.setCapStyle(Qt.FlatCap)
        self.hist_x_series.setPen(line_pen)
        self.hist_axis.setLabelFormat('%d')
        self.hist_axis_y.setLabelFormat('%d')
        self.hist_dot.setMarkerSize(10.0)
        self.hist_chart_view = QChartView(self.hist_chart)
        self.hist_chart_view.setRenderHint(QPainter.Antialiasing)
        self.hist_chart_view.setMinimumHeight(200)
        self.hist_chart_view.setBackgroundBrush(Qt.NoBrush)

        body = QHBoxLayout()
        body.setContentsMargins(12, 8, 12, 0)
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
        board_col.addWidget(deck_label('WIN RATE // X ─ O', accent=True))
        board_col.addWidget(self.hist_chart_view, 1)
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

        # History card (user request): every step listed; clicking an entry
        # rewinds the game to that step via go_to_step (same replay path as
        # the timeline slider).
        hist_card = QFrame()
        hist_card.setObjectName('card')
        hist_lay = QVBoxLayout(hist_card)
        hist_title = QLabel(t('History', '歷史'))
        hist_title.setObjectName('cardTitle')
        hist_lay.addWidget(hist_title)
        self.hist_list = QListWidget()
        self.hist_list.setMaximumHeight(160)
        self.hist_list.itemClicked.connect(self.on_history_clicked)
        hist_lay.addWidget(self.hist_list)
        panel.addWidget(hist_card)

        # Assistant card
        az_card = QFrame()
        az_card.setObjectName('card')
        az_lay = QVBoxLayout(az_card)
        az_title = QLabel(t('Best Moves', '最佳棋步'))
        az_title.setObjectName('cardTitle')
        az_lay.addWidget(az_title)
        self.win_bar = QWidget()
        self.win_bar.setFixedHeight(14)
        self.win_bar_lay = QHBoxLayout(self.win_bar)
        self.win_bar_lay.setContentsMargins(0, 0, 0, 0)
        self.win_bar_lay.setSpacing(2)
        self.bar_x = QFrame()
        self.bar_x.setStyleSheet('background: #F7F7F7;')
        self.bar_d = QFrame()
        self.bar_d.setStyleSheet('background: #35373C;')
        self.bar_o = QFrame()
        self.bar_o.setStyleSheet('background: #FFFA00;')
        for _b in (self.bar_x, self.bar_d, self.bar_o):
            _b.setMinimumWidth(0)
        self.win_bar_lay.addWidget(self.bar_x, 1)
        self.win_bar_lay.addWidget(self.bar_d, 1)
        self.win_bar_lay.addWidget(self.bar_o, 1)
        self.bar_d.setVisible(False)
        az_lay.addWidget(self.win_bar)
        self.analysis_pct = QLabel('')
        self.analysis_pct.setObjectName('pctReadout')
        az_lay.addWidget(self.analysis_pct)
        scan_row = QHBoxLayout()
        scan_row.setContentsMargins(0, 0, 0, 0)
        self.load_rail = QProgressBar()
        self.load_rail.setObjectName('loadingRail')
        self.load_rail.setRange(0, 0)  # indeterminate busy rail
        self.load_rail.setTextVisible(False)
        self.load_rail.setFixedHeight(6)
        self.load_rail.setVisible(False)
        self.scan_lbl = deck_label('SCANNING', size=11, accent=True)
        self.scan_lbl.setVisible(False)
        scan_row.addWidget(self.load_rail, 1)
        scan_row.addSpacing(8)
        scan_row.addWidget(self.scan_lbl)
        az_lay.addLayout(scan_row)
        # AI thinking line (D24): live search progress while the engine runs,
        # then the post-move summary — both live next to the move list so the
        # side panel is where "what is the AI doing" is read.
        self.think_lbl = QLabel('')
        self.think_lbl.setObjectName('muted')
        self.think_lbl.setWordWrap(True)
        self.think_lbl.setVisible(False)
        az_lay.addWidget(self.think_lbl)
        self.analysis_list = QListWidget()
        self.analysis_list.setMinimumHeight(160)
        self.analysis_list.itemClicked.connect(self.on_analysis_clicked)
        az_lay.addWidget(self.analysis_list)
        # empty state (endfield): 45-degree hatch band + one full sentence
        self.az_hatch = QLabel(t(
            'Analysis results appear here after each move.',
            '每手棋之後，分析結果會出現在這裡。'))
        self.az_hatch.setFixedHeight(64)
        self.az_hatch.setAlignment(Qt.AlignCenter)
        self.az_hatch.setWordWrap(True)
        self.az_hatch.paintEvent = lambda e, lbl=self.az_hatch, fn=self._paint_hatch:             fn(lbl, e)
        self.az_hatch.show()
        az_lay.addWidget(self.az_hatch)
        hint = QLabel(t('Click a move to highlight it', '點擊棋步可在棋盤上標示'))
        hint.setObjectName('muted')
        az_lay.addWidget(hint)
        panel.addWidget(az_card, 1)

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
        panel.addWidget(self.cvc_card)

        panel_widget = QWidget()
        panel_widget.setObjectName('sidePanel')
        panel_widget.setAttribute(Qt.WA_StyledBackground, True)
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(340)
        body.addWidget(panel_widget)
        root.addLayout(body, 1)

        self.footer = QFrame()
        self.footer.setObjectName('footerTechnical')
        footer_lay = QHBoxLayout(self.footer)
        footer_lay.setContentsMargins(16, 4, 16, 4)
        self.footer_lbl = deck_label('', size=11)
        footer_lay.addWidget(self.footer_lbl)
        footer_lay.addStretch(1)
        root.addWidget(self.footer)


    # -- session flow ------------------------------------------------------

    def start_session(self, session):
        self.session = session
        self.new_game()

    def new_game(self):
        s = self.session
        self.game_id += 1
        self.game = (NormalGame() if s['game_type'] == 'normal' else UltimateGame())
        s['game'] = self.game
        self.play_again_btn.setVisible(False)
        self.new_btn.setVisible(True)
        # D33: init Go-style clocks (X moves first — X's clock starts now)
        preset = s.get('clock')
        if preset:
            self.clock = {'X': dict(preset, in_byo=False, byo=preset['byo_time']),
                          'O': dict(preset, in_byo=False, byo=preset['byo_time'])}
            self._time_loss = None
            self._clock_tick_t = time.monotonic()
            self._clock_timer.start(100)
            self.clock_lbl.show()
        else:
            self.clock = None
            self._time_loss = None
            self._clock_timer.stop()
            self.clock_lbl.hide()
            self.byo_tag.hide()
        self._update_clock_chips()
        s['moves'] = []
        s['history'] = [self._history_point()]
        s['cvc_paused'] = False
        self._chart_fit = False         # back to the sliding window
        self._chart_anim.stop()
        self.step_idx = 0
        self.gen += 1
        self.busy = False
        self.analysis_busy = False
        self.analysis_pending = False
        self.cvc_timer.stop()
        self.think_lbl.setText('')
        self.think_lbl.setVisible(False)
        self.speed_slider.setValue(int(s.get('cvc_speed', 0.4) * 10))
        self.cvc_timer.setInterval(max(50, int(s.get('cvc_speed', 0.4) * 1000)))
        self.assistant_switch.setChecked(s.get('assistant_enabled', True))
        self.board.set_game(self.game)
        self.board.reset_reveal()
        self.analysis_list.clear()
        self._render_history()
        self.refresh()
        self.after_move()
        self.trigger_analysis()

    def _history_point(self):
        """Win-rate triplet for the current position.

        Exact (Normal tablebase, or any terminal position) is computed inline;
        a non-terminal Ultimate position defers to the assistant's full-budget
        search (the None placeholder is filled by on_analysis_done).
        """
        if isinstance(self.game, NormalGame) or self.game.is_over():
            return tuple(position_win_rates(self.game, 0))
        return None

    def go_to_step(self, k):
        """Jump to history step k (0..N) by replaying moves[:k]."""
        s = self.session
        moves = s['moves']
        k = max(0, min(k, len(moves)))
        if k == self.step_idx:
            return
        self.cvc_timer.stop()
        self.pvc_timer.stop()
        s['cvc_paused'] = s['mode'] == 'cvc' and k < len(moves)
        self.game = (NormalGame() if s['game_type'] == 'normal' else UltimateGame())
        for mv in moves[:k]:
            apply_move(self.game, mv)
        s['game'] = self.game
        self.step_idx = k
        self.busy = False
        self.gen += 1
        self.board.set_game(self.game)
        self.board.reset_reveal()
        if self.clock is not None:
            # rewinding leaves any timeout verdict behind: the rewound
            # position is live again, so resume the clock from now
            self._time_loss = None
            self._clock_tick_t = time.monotonic()
            if self.game.is_over():
                self._clock_timer.stop()
            else:
                self._clock_timer.start(100)
            self._update_clock_chips()
        self._render_history()
        self.refresh()
        self.trigger_analysis()
        self.update_cvc_controls()
        if (s['mode'] == 'cvc' and not s.get('cvc_paused', False)
                and not self.game.is_over() and is_ai_turn(self.session)):
            self.cvc_timer.start()
        elif (s['mode'] == 'pvc' and not self.game.is_over()
                and is_ai_turn(self.session)):
            self.pvc_timer.start(300)

    def _render_history(self):
        s = self.session
        moves = s.get('moves', [])
        history = s.get('history', [])
        self.hist_x_series.clear()
        self.hist_ref.clear()
        self.hist_dot.clear()
        for k, point in enumerate(history):
            if point is None:
                if k == 0:
                    # opening: plot the 50% start while the analysis is still
                    # in flight, so the curve begins at zero (user feedback)
                    self.hist_x_series.append(0, 50.0)
                continue  # Ultimate point still awaiting the assistant's search
            x, d, o = point
            self.hist_x_series.append(k, x * 100.0)
        # Fixed x-scale with scrolling: the window is 30 steps wide; past
        # that it slides so the latest step stays at the right edge. Once a
        # game is over the whole recorded curve compresses into view instead
        # (animated — see _chart_fit_all).
        n_steps = max(1, len(history) - 1)
        window = 30
        if (self.game is not None and self.game.is_over()
                and getattr(self, '_chart_fit', False)):
            ax_min, ax_max = 0, max(n_steps, 12)
        elif n_steps <= window:
            ax_min, ax_max = 0, window
        else:
            ax_min, ax_max = n_steps - window, n_steps
        # 50% reference spans the whole window (2 fixed points) so the chart
        # never looks empty in the opening moves, even before the X line has
        # two points to connect.
        self.hist_ref.append(ax_min, 50.0)
        self.hist_ref.append(ax_max, 50.0)
        if history:
            self.hist_axis.setRange(ax_min, ax_max)
            self.hist_axis.setTickCount(11)  # 0, 3, 6 … 30
            self.hist_axis_y.setRange(0, 100)
            if history[self.step_idx] is not None:
                self.hist_dot.append(self.step_idx,
                                     history[self.step_idx][0] * 100.0)
        # rebuild the history card list; clicking a row rewinds to that step
        self.hist_list.blockSignals(True)
        self.hist_list.clear()
        self.hist_list.addItem(self._hist_item(0, None))
        for k, mv in enumerate(moves, start=1):
            self.hist_list.addItem(self._hist_item(k, mv))
        self.hist_list.setCurrentRow(self.step_idx)
        self.hist_list.blockSignals(False)

    def _chart_fit_all(self):
        """Game over: compress the whole recorded curve into view (the user's
        'one complete chart'), animating the axis from the sliding window to
        0..N so every step is visible at once."""
        n_steps = max(1, len(self.session.get('history', [])) - 1)
        target = float(max(n_steps, 12))
        self._chart_fit = True
        if not anim_enabled():
            self.hist_axis.setRange(0, target)
            return
        anim = self._chart_anim
        cur_min, cur_max = self.hist_axis.min(), self.hist_axis.max()
        anim.stop()
        self._chart_from = (float(cur_min), float(cur_max))
        self._chart_to = (0.0, target)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.start()

    def _on_chart_fit(self, t):
        t = float(t)
        fmin, fmax = self._chart_from
        tmin, tmax = self._chart_to
        self.hist_axis.setRange(fmin + (tmin - fmin) * t,
                                fmax + (tmax - fmax) * t)

    def _hist_item(self, k, mv):
        """History card row: '07 · ✕ B2 (1,2)'; data carries the step index."""
        if mv is None:
            text = f'{k:02d} · {t("start", "開局")}'
        else:
            side = '✕' if (k - 1) % 2 == 0 else '○'
            text = f'{k:02d} · {side} {move_text(mv)}'
        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, k)
        return item

    def on_history_clicked(self, item):
        k = item.data(Qt.UserRole)
        if k is not None and k != self.step_idx:
            self.go_to_step(k)

    def refresh(self):
        if self.game is None:
            return
        n_moves = len(self.session.get('moves', []))
        self.step_lbl.setText(f'STEP {self.step_idx:02d}/{n_moves:02d}')
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
        self.footer_lbl.setText(
            f'GRID.UTTT │ MODE {mode_text} │ X: {str(x_type).upper()} '
            f'─ O: {str(o_type).upper()}')
        self.board.update()
        self.update_cvc_controls()

    # -- moves -------------------------------------------------------------

    def on_cell_click(self, move):
        if self._time_loss:   # D33: lost on time — the game is over
            return
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
        s['history'].append(self._history_point())
        self.step_idx = len(moves)
        self.gen += 1
        self.board.set_game(self.game)
        if self.clock is not None:
            # the mover made it in time: their byo-yomi counter resets and the
            # switch gap is not charged to the next player
            sc = self.clock[side]
            if sc['in_byo']:
                sc['byo'] = sc['byo_time']
            self._clock_tick_t = time.monotonic()
            self._update_clock_chips()
        self.board.animate_move(move)
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
                if not self.session.get('cvc_paused', False):
                    self.cvc_timer.start()
                else:
                    self.update_cvc_controls()
            else:
                self.pvc_timer.start(300)
        else:
            self.update_cvc_controls()

    # -- clock (D33: Go-style main time + byo-yomi periods) ----------------

    @staticmethod
    def _fmt_main(seconds):
        m, sec = divmod(max(0, int(seconds)), 60)
        return f'{m}:{sec:02d}'

    def _clock_tick(self):
        if (self.clock is None or self._time_loss or self.game is None
                or self.game.is_over()):
            return
        now = time.monotonic()
        dt = now - (self._clock_tick_t or now)
        self._clock_tick_t = now
        if self.session.get('cvc_paused', False):
            return                      # CvC paused: clocks hold
        side = self.game.current
        c = self.clock[side]
        if not c['in_byo']:
            c['main'] -= dt
            if c['main'] <= 0.0:
                c['main'] = 0.0
                c['in_byo'] = True      # main exhausted -> byo-yomi starts
                c['byo'] = c['byo_time']
        else:
            c['byo'] -= dt
            if c['byo'] <= 0.0:
                c['periods'] -= 1       # a byo period burned on this move
                if c['periods'] <= 0:
                    self._flag_timeout(side)
                    return
                c['byo'] = c['byo_time']
        self._update_clock_chips()

    def _flag_timeout(self, side):
        """Timeout with no byo-yomi periods left: opponent wins on time."""
        winner = 'O' if side == 'X' else 'X'
        self._time_loss = side
        self._clock_timer.stop()
        self.status_mark.setText('○' if winner == 'O' else '✕')
        self.status_mark.setStyleSheet(
            'color: ' + ('#FFFA00' if winner == 'O' else '#EDEDED')
            + '; font-size: 20px; font-weight: 700;')
        self.status_text.setText(
            t(f'{side} lost on time — {winner} wins!',
              f'{side} 超時判負 — {winner} 獲勝！'))
        self.play_again_btn.setVisible(True)
        self.new_btn.setVisible(False)
        log.info('Time out: %s loses on time', side)

    def _update_clock_chips(self):
        """D34: one clock readout in the command bar — `✕ 3:00 │ ○ 2:41`,
        the side to move in signal yellow; a byo-yomi side becomes a
        yellow-filled tag (the endfield 'look at this' block)."""
        if self.clock is None:
            return
        segs = []
        byo_side = None
        mark = {'X': ('✕', '#EDEDED'), 'O': ('○', '#FFFA00')}
        for side in ('X', 'O'):
            c = self.clock[side]
            glyph, gcol = mark[side]
            head = f'<font color="{gcol}">{glyph}</font>'
            if c['in_byo']:
                segs.append(f'{head} 讀秒{max(0, c["periods"])} '
                            f'{max(0.0, c["byo"]):0.1f}s')
                byo_side = side
            else:
                segs.append(f'{head} {self._fmt_main(c["main"])}')
        over = self.game is not None and self.game.is_over()
        active = (self.game.current if self.game is not None
                  and not self._time_loss and not over else None)
        parts = []
        for side, seg in zip(('X', 'O'), segs):
            if side == active:
                parts.append(f'<font color="#FFFA00"><b>{seg}</b></font>')
            else:
                parts.append(f'<font color="#A3A3A3">{seg}</font>')
        self.clock_lbl.setText(' │ '.join(parts))
        if byo_side is not None and not self._time_loss and not over:
            c = self.clock[byo_side]
            glyph = '✕' if byo_side == 'X' else '○'
            self.byo_tag.setText(
                f'{glyph} 讀秒 {max(0, c["periods"])} · '
                f'{max(0.0, c["byo"]):0.1f}s')
            self.byo_tag.show()
        else:
            self.byo_tag.hide()

    def cvc_tick(self):
        if self._time_loss:
            return
        if self.session['mode'] != 'cvc':
            self.cvc_timer.stop()
            return
        if self.session.get('cvc_paused', False):
            self.cvc_timer.stop()
            return
        if self.game.is_over() or self.busy or not is_ai_turn(self.session):
            return
        self.run_ai()

    def run_ai(self):
        if self._time_loss:
            return
        if self.game is None or self.game.is_over() or self.busy or not is_ai_turn(self.session):
            return
        x_type, o_type = side_types(self.session)
        ai_type = x_type if self.game.current == X else o_type
        self.busy = True
        self.refresh()
        self._ai_started = time.monotonic()
        trace = {'engine': ai_type}  # pre-filled: the live status shows the
        # engine name while searching (the impl shell only setdefaults it)
        self._last_trace = trace
        worker = AIWorker(self.game.clone(), ai_type,
                          self.session['mcts'], self.session.get('minimax_depth', 3),
                          trace=trace)
        gen = self.gen
        worker.progress.connect(self.on_ai_progress)
        worker.move_ready.connect(lambda move, g=gen: self.on_ai_done(move, g))
        worker.move_failed.connect(lambda msg, g=gen: self.on_ai_failed(msg, g))
        worker.finished.connect(lambda w=worker: self._reap(w))
        self.workers.append(worker)
        worker.start()

    def on_ai_progress(self, info):
        """Liveness status (user feedback): just the sim counter ticking —
        no probabilities; a moving counter is what proves the AI isn't stuck."""
        if not self.busy or self.game is None:
            return
        player = self.game.current
        mark = '✕' if player == X else '○'
        engine = (getattr(self, '_last_trace', None) or {}).get('engine', '')
        self.status_text.setText(
            f'{mark} · {t("thinking...", "思考中")} · '
            f'{info.get("it", 0)}/{info.get("iterations", 0)}')
        self.think_lbl.setText(
            f'{engine} · {info.get("it", 0)}/{info.get("iterations", 0)}')
        self.think_lbl.setVisible(True)

    def _show_thinking_trace(self):
        """Completion status (D24, simplified): engine + wall time — enough
        to see the AI finished and how long it took."""
        engine = (getattr(self, '_last_trace', None) or {}).get('engine', 'AI')
        elapsed = time.monotonic() - getattr(self, '_ai_started',
                                             time.monotonic())
        self.think_lbl.setText(
            f'{engine} {t("done ·", "完成 ·")} {elapsed:.1f}s')
        self.think_lbl.setVisible(True)

    def on_ai_done(self, move, gen=None):
        if gen is not None and gen != self.gen:
            return  # position changed while the engine was thinking: drop it
        self.busy = False
        self._show_thinking_trace()
        self.apply_move(move)

    def on_ai_failed(self, message, gen=None):
        if gen is not None and gen != self.gen:
            return
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

    def on_revert_clicked(self):
        if self.session['mode'] == 'pvc':
            # PvC: rewind to the previous HUMAN decision point. Stepping a
            # single ply would hand the turn back to the AI, which just
            # re-moves and nothing changes (user-reported bug).
            human_is_x = self.session.get('first_player', 'human') == 'human'
            k = self.step_idx - 1
            # X to move at even step counts; the human sits on X iff they
            # moved first, so walk back to the nearest human-to-move step
            while k > 0 and ((k % 2 == 0) != human_is_x):
                k -= 1
            self.go_to_step(k)
            return
        self.go_to_step(self.step_idx - 1)

    def on_pause_clicked(self):
        s = self.session
        s['cvc_paused'] = not s.get('cvc_paused', False)
        if s['cvc_paused']:
            self.cvc_timer.stop()
        else:
            if (s['mode'] == 'cvc' and not self.game.is_over()
                    and is_ai_turn(self.session)):
                self.cvc_timer.start()
        self.update_cvc_controls()

    def _update_pause_icon(self):
        paused = self.session.get('cvc_paused', False)
        self.pause_btn.setIcon(_media_icon('play' if paused else 'pause'))
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
        paused = self.session.get('cvc_paused', False)
        self.revert_btn.setEnabled(self.step_idx > 0)
        self.pause_btn.setEnabled(True)
        self._update_pause_icon()
        self.next_btn.setEnabled(rewound or (ai_turn and not self.busy))

    # -- assistant ---------------------------------------------------------

    def on_assistant_toggled(self, checked):
        self.session['assistant_enabled'] = checked
        self.hist_chart_view.setVisible(checked)
        if checked:
            self.trigger_analysis()
        else:
            self.analysis_list.clear()
            self.win_bar.setVisible(False)
            self.analysis_pct.setText('—')
            self.analysis_list.addItem(t('Assistant disabled', '助手已關閉'))

    def _set_scanning(self, on, step=None):
        """Persistent assistant status: indeterminate rail + SCANNING deck tag."""
        self.load_rail.setVisible(on)
        if on:
            self.scan_lbl.setText(
                f'SCANNING // MOVE {step:02d}' if step is not None else 'SCANNING')
            self.scan_lbl.setVisible(True)
        else:
            self.scan_lbl.setVisible(False)

    def trigger_analysis(self):
        if not self.session.get('assistant_enabled', True):
            return
        if self.game is None or self.game.is_over():
            return
        if self.analysis_busy:
            self.analysis_pending = True
            return
        self.analysis_busy = True
        self._set_scanning(True, self.step_idx)
        gen = self.gen
        step = self.step_idx
        gid = self.game_id
        snapshot = self.game.clone()
        # assistant budget is its own config knob (not the game AI's menu
        # slider) — re-read per trigger so settings-page changes apply live
        worker = AnalysisWorker(snapshot,
                                int(cfg_engine('assistant_sims', 2000)))
        worker.result_ready.connect(
            lambda items, rates, g=gen, st=step, gi=gid:
                self.on_analysis_done(items, rates, g, st, gi))
        worker.finished.connect(lambda w=worker: self._reap(w))
        self.workers.append(worker)
        worker.start()

    def on_analysis_done(self, items, rates, gen, step, gid):
        self.analysis_busy = False
        self._set_scanning(False)
        s = self.session or {}
        history = s.get('history', [])
        if gid == self.game_id and gen == self.gen and 0 <= step < len(history) \
                and history[step] is None:
            history[step] = tuple(rates)
            self._render_history()
        if gen == self.gen and self.session.get('assistant_enabled', True):
            self.render_analysis(items, rates)
        if self.analysis_pending:
            self.analysis_pending = False
            self.trigger_analysis()

    def _on_bar_anim(self, t):
        fx, fo = self._bar_from
        tx, to = self._bar_to
        cur = (fx + (tx - fx) * float(t), fo + (to - fo) * float(t))
        self._bar_pos = cur
        self._apply_bar(cur)

    def _paint_hatch(self, lbl, e):
        p = QPainter(lbl)
        r = lbl.rect()
        p.fillRect(r, QColor('#1d1d1d'))
        p.setPen(QPen(QColor('#3a3a3a'), 1))
        x = r.left() - r.height()
        while x < r.right():
            p.drawLine(x, r.bottom(), x + r.height(), r.top())
            x += 7
        p.setPen(QColor('#A3A3A3'))
        p.setFont(lbl.font())
        p.drawText(r, Qt.AlignCenter, lbl.text())
        p.end()

    def _apply_bar(self, v):
        self.win_bar_lay.setStretchFactor(self.bar_x, int(v[0] * 1000))
        self.win_bar_lay.setStretchFactor(self.bar_d, 0)
        self.win_bar_lay.setStretchFactor(self.bar_o, int(v[1] * 1000))

    def render_analysis(self, items, rates):
        self.analysis_list.clear()
        self.az_hatch.setVisible(not items)
        self.analysis_list.setVisible(bool(items))
        x, d, o = rates
        # Readout spans the FULL outcome distribution (a draw-expected
        # position used to collapse to a meaningless "X 0% · O 0%" — the
        # user-caught design bug). The mono capital O reads as zero, so
        # the sides use the ✕/○ marks with their board colors instead of
        # bare letters.
        tot3 = x + d + o
        if tot3 > 0:
            rx, rd, ro = x / tot3, d / tot3, o / tot3
        else:
            rx = rd = ro = 0.0
        tot = x + o
        if tot > 0:
            x, o = x / tot, o / tot
        self.win_bar.setVisible(True)
        # D31: slide the X/O bar to the new split instead of jumping
        old_x, old_o = self._bar_pos
        if anim_enabled() and (abs(old_x - x) > 0.005 or abs(old_o - o) > 0.005):
            self._bar_anim.stop()
            self._bar_from = (old_x, old_o)
            self._bar_to = (x, o)
            self._bar_anim.setStartValue(0.0)
            self._bar_anim.setEndValue(1.0)
            self._bar_anim.start()
        else:
            self._bar_pos = (x, o)
            self._apply_bar((x, o))
        segs = [f'<span style="color:#EDEDED;">✕ {rx:.0%}</span>']
        if rd >= 0.005:
            segs.append(f'<span style="color:#A3A3A3;">= {rd:.0%}</span>')
        segs.append(f'<span style="color:#FFFA00;">○ {ro:.0%}</span>')
        self.analysis_pct.setText(' · '.join(segs))
        if not items:
            self.analysis_list.addItem(t('No moves to analyze', '沒有可分析的棋步'))
            return
        for it in items[:5]:
            pct = max(0.0, min(1.0, it['pct']))
            if isinstance(self.game, NormalGame):
                verdict = 'Win' if pct == 1.0 else ('Draw' if pct == 0.5 else 'Loss')
            else:
                verdict = f'{pct:.0%}'
            reason_code = it['reason']
            reason_en, reason_zh = REASON_TEXT.get(
                reason_code, (reason_code, reason_code))
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
        self._clock_timer.stop()        # game over: clocks freeze (D35 review)
        self._update_clock_chips()
        self.board.animate_win_line()   # D31: strike line draws in
        self._chart_fit_all()           # compress the whole curve into view
        # No popup: a prominent "Play Again" button replaces New Game.
        self.play_again_btn.setVisible(True)
        self.new_btn.setVisible(False)

# ---------------------------------------------------------------------------
# Settings page (D28): flips engine switches from config.toml and persists
# them back with save_engine_config (comments survive).
# ---------------------------------------------------------------------------

class SettingsPage(EndfieldPage):
    back_requested = Signal()
    BACKDROP = 'static/backgrounds/survey-dark.jpg'

    KEY_LABELS = {
        'c_mcts': ('MCTS tree search in C', 'C 完整 MCTS 樹搜尋'),
        'c_minimax': ('Minimax backends in C', 'C Minimax 後端'),
        'c_rollout_pattern': ('Pattern rollout (research, slower)',
                              'Pattern rollout（研究用，較慢）'),
        'rollout_heuristic': ('Heuristic rollout (win/block)', '啟發式 rollout（贏/擋）'),
        'tree_reuse': ('Reuse search tree between moves', '跨手重用搜尋樹'),
        'dynamic_uct': ('Dynamic UCT exploration', '動態 UCT 探索'),
        'bitboard': ('Bitboard acceleration', '位元棋盤加速'),
        'opening_book': ('Opening book', '開局書'),
        'micro_tablebase': ('Micro endgame tablebase', 'micro 殘局表'),
        'multithreaded': ('Multithreaded MCTS (legacy)', '多執行緒 MCTS（舊路徑）'),
        'early_stop': ('Stop dominant searches early', '提早終止優勢搜尋'),
        'progressive_widening': ('Progressive widening', '漸進式加粗'),
        'symmetry': ('Symmetry reduction', '對稱性歸約'),
        'object_pool': ('Node object pool', '節點物件池'),
        'use_lmr': ('Late move reductions', 'LMR 剪枝'),
        'use_killers': ('Killer move heuristic', '殺手棋啟發'),
        'use_aspiration': ('Aspiration windows', '期望視窗'),
        'use_eval_ordering': ('Eval-based root ordering', '根節點評估排序'),
    }

    # Functional grouping: (deck title, one-line explanation, [switch keys]).
    # Everything here lives in the [engine] section of config.toml.
    GROUPS = [
        ('C ENGINE BACKENDS', 'C 引擎後端',
         '哪些演算法走原生 C 實作；關閉後自動回退純 Python（較慢）。',
         ['c_mcts', 'c_minimax', 'c_rollout_pattern']),
        ('MCTS SEARCH', 'MCTS 搜尋行為',
         '影響 MCTS 搜尋策略與棋力（C 樹與 Python 路徑皆生效，D37）。',
         ['rollout_heuristic', 'tree_reuse', 'dynamic_uct', 'early_stop',
          'progressive_widening', 'multithreaded', 'object_pool']),
        ('SEARCH AIDS', '搜尋輔助結構',
         '開局書、殘局表與加速用的內部表示。',
         ['opening_book', 'micro_tablebase', 'bitboard', 'symmetry']),
        ('MINIMAX PRUNING', 'MINIMAX PRO 剪枝',
         '僅影響 Minimax Pro 引擎的搜尋加速。',
         ['use_lmr', 'use_killers', 'use_aspiration', 'use_eval_ordering']),
    ]

    def __init__(self):
        super().__init__()
        self.setObjectName('pageSettings')
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        title = QLabel('Settings — 設定')
        title.setObjectName('title')
        top.addWidget(title)
        top.addStretch(1)
        back = _deck_button(t('Back to Menu', '返回選單'))
        back.setObjectName('ghost')
        back.clicked.connect(self.back_requested.emit)
        top.addWidget(back)
        root.addLayout(top)
        root.addWidget(deck_label('ENGINE SWITCHES // config.toml'))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        panel = QWidget()
        panel_lay = QVBoxLayout(panel)
        panel_lay.setContentsMargins(4, 4, 12, 4)
        panel_lay.setSpacing(12)

        self._checks = {}
        for deck, zh, hint, keys in self.GROUPS:
            card = QFrame()
            card.setObjectName('card')
            lay = QVBoxLayout(card)
            lay.setContentsMargins(12, 10, 12, 10)
            lay.setSpacing(6)
            head = QHBoxLayout()
            head.addWidget(deck_label(deck))
            sub = QLabel(zh)
            sub.setObjectName('muted')
            head.addSpacing(8)
            head.addWidget(sub)
            head.addStretch(1)
            lay.addLayout(head)
            hint_lbl = QLabel(hint)
            hint_lbl.setObjectName('muted')
            hint_lbl.setWordWrap(True)
            lay.addWidget(hint_lbl)
            for key in keys:
                en, zh_key = self.KEY_LABELS.get(key, (key, key))
                cb = QCheckBox(f'{en} — {zh_key}  [{key}]')
                cb.setChecked(bool(cfg_engine(key, False)))
                lay.addWidget(cb)
                self._checks[key] = cb
            if keys[0] == 'c_mcts':
                spin_row = QHBoxLayout()
                lbl = QLabel(t('C search threads', 'C 搜尋執行緒數') + '  [workers]')
                spin_row.addWidget(lbl)
                self.workers_spin = QSpinBox()
                self.workers_spin.setRange(1, 16)
                self.workers_spin.setValue(int(cfg_engine('workers', 8)))
                spin_row.addWidget(self.workers_spin)
                spin_row.addStretch(1)
                lay.addLayout(spin_row)
                sims_row = QHBoxLayout()
                sims_lbl = QLabel(t('AI Assistant sims (Best Moves analysis)',
                                    'AI 助手模擬數（最佳著法分析）')
                                  + '  [assistant_sims]')
                sims_row.addWidget(sims_lbl)
                self.assistant_sims_spin = QSpinBox()
                self.assistant_sims_spin.setRange(200, 30000)
                self.assistant_sims_spin.setSingleStep(100)
                self.assistant_sims_spin.setValue(
                    int(cfg_engine('assistant_sims', 2000)))
                sims_row.addWidget(self.assistant_sims_spin)
                sims_row.addStretch(1)
                lay.addLayout(sims_row)
            panel_lay.addWidget(card)

        note = QLabel(t('Applies immediately; saved to config.toml. Takes '
                        'full effect on the next game.',
                        '立即生效並保存到 config.toml，下一局完全套用。'))
        note.setObjectName('muted')
        note.setWordWrap(True)
        panel_lay.addWidget(note)
        panel_lay.addStretch(1)
        scroll.setWidget(panel)
        root.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        save = _deck_button(t('Save Settings', '保存設定'), primary=True)
        save.clicked.connect(self._on_save)
        btn_row.addWidget(save)
        reset = _deck_button(t('Restore Defaults', '恢復預設'))
        reset.setObjectName('ghost')
        reset.clicked.connect(self._on_reset)
        btn_row.addWidget(reset)
        btn_row.addStretch(1)
        self.status_lbl = QLabel('')
        self.status_lbl.setObjectName('muted')
        btn_row.addWidget(self.status_lbl)
        root.addLayout(btn_row)

    def refresh_values(self):
        for key, cb in self._checks.items():
            cb.setChecked(bool(cfg_engine(key, False)))
        self.workers_spin.setValue(int(cfg_engine('workers', 8)))
        self.assistant_sims_spin.setValue(
            int(cfg_engine('assistant_sims', 2000)))

    def _on_save(self):
        save_engine_config({'engine': self._collect()})
        self.status_lbl.setText(t('Saved to config.toml', '已保存到 config.toml'))

    def _on_reset(self):
        save_engine_config()  # scalar defaults; hand-managed tables are kept
        self.refresh_values()
        self.status_lbl.setText(t('Defaults restored', '已恢復預設'))

    def _collect(self):
        eng = {key: cb.isChecked() for key, cb in self._checks.items()}
        eng['workers'] = self.workers_spin.value()
        eng['assistant_sims'] = self.assistant_sims_spin.value()
        return eng


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ultimate Tic Tac Toe — 終極井字棋')
        load_app_font()
        self.setStyleSheet(QSS)
        self.stack = QStackedWidget()
        self.menu_page = MenuPage()
        self.game_page = GamePage()
        self.settings_page = SettingsPage()
        self.menu_page.start_requested.connect(self.start_game)
        self.menu_page.settings_requested.connect(self.show_settings)
        self.game_page.back_requested.connect(self.show_menu)
        self.settings_page.back_requested.connect(self.show_menu)
        self.stack.addWidget(self.menu_page)
        self.stack.addWidget(self.game_page)
        self.stack.addWidget(self.settings_page)
        self.setCentralWidget(self.stack)
        self.resize(1024, 740)

    def start_game(self, session):
        self.game_page.start_session(session)
        self._switch_page(self.game_page)

    def show_menu(self):
        self._switch_page(self.menu_page)

    def show_settings(self):
        self.settings_page.refresh_values()
        self._switch_page(self.settings_page)

    def _switch_page(self, page):
        """Fade the page in (instant when Reduce Motion is enabled)."""
        self.stack.setCurrentWidget(page)
        if not anim_enabled():
            return
        effect = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b'opacity', self)
        anim.setDuration(280)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.finished.connect(
            lambda: page.setGraphicsEffect(None) if page.graphicsEffect() is effect else None)
        self._page_anim = anim
        anim.start(QAbstractAnimation.DeleteWhenStopped)


def main(argv=None):
    parser = argparse.ArgumentParser(description='SBA desktop app (PySide6)')
    parser.add_argument('--debug', action='store_true',
                        help='verbose backend logs')
    parser.add_argument('--qt', action='store_true',
                        help='start the desktop app (it is the default entry)')
    args = parser.parse_args(argv)
    if args.debug:
        log.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    app = QApplication(sys.argv[:1])
    app.setApplicationName('SBA')
    app.setFont(QFont(FONT_FAMILIES, 10))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
