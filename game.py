# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ultimate Tic Tac Toe — game rules and board helpers.

Defines the Normal (3x3) and Ultimate (9x9) game rules, move application, and
the board geometry / win-badge helpers used by the AI engines, analysis,
self-tests, and the web UI. Pure logic with no NiceGUI dependency.

Maintenance notes:
- Keep this file free of UI/engine imports (decision D2) — everything else
  depends on it.
- win_badge_svg / win_segment / macro_center are shared by the web board.
"""
X = 'X'
O = 'O'
EMPTY = ''
LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def line_winner(cells):
    """Return X/O if `cells` (9 entries) contains a winning line, else None."""
    for a, b, c in LINES:
        if cells[a] in (X, O) and cells[a] == cells[b] == cells[c]:
            return cells[a]
    return None


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
        return line_winner(self.board)

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
        return line_winner(self.micro[m])

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
