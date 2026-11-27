# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ultimate Tic Tac Toe — game rules and board helpers.

Defines the Normal (3x3) rules, move application, and the board geometry /
win-badge helpers used by the AI engines, analysis, self-tests, and the UIs.
The Ultimate (9x9) rules live in the Rust state layer (az.rustbridge —
RustUltimateGame; the pure-Python implementation retired with the removal
project). Pure logic with no UI-framework dependency.

Maintenance notes:
- Keep this file free of UI/engine imports — everything else
  depends on it.
- win_badge_svg / win_segment / macro_center are shared by the web board.
"""

X = "X"
O = "O"
EMPTY = ""
LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


def line_winner(cells):
    """Return X/O if `cells` (9 entries) contains a winning line, else None."""
    for a, b, c in LINES:
        if cells[a] in (X, O) and cells[a] == cells[b] == cells[c]:
            return cells[a]
    return None


def _assert_playable(game):
    """Reject moves on a finished board (shared by every rule implementation)."""
    if game.result() is not None:
        raise ValueError("game is over")


# ============================================================
# Game engines
# ============================================================


class NormalGame:
    """Plain 3x3 tic-tac-toe — the drift-free baseline ruleset.

    State: `board` is a list of 9 cells (X / O / EMPTY, row-major from the
    top-left) and `current` is the mark to move; X always opens the game.

    This class and RustUltimateGame (src/az/rustbridge.py, the ultimate
    implementation) share the duck-typed contract every consumer relies on:
    the session and the UIs drive a game purely through legal_moves /
    make_move / is_over / result, and the search engines work through
    clone() + make_move. Moves are plain cell indices (ints) — the move
    SHAPE is what game.apply_move dispatches on, so alternative rules
    implementations stay interchangeable.
    """

    def __init__(self):
        self.board = [EMPTY] * 9
        self.current = X

    def legal_moves(self):
        """Indices (0-8) of every EMPTY cell, in board order.

        A won board returns [] even when empty cells remain — without this,
        searches keep expanding terminal nodes.
        """
        if self.winner() is not None:
            return []
        return [i for i, c in enumerate(self.board) if c == EMPTY]

    def make_move(self, index):
        """Place `current`'s mark at `index` on this game (no copy).

        Raises ValueError("game is over") when result() is already set and
        ValueError("cell is occupied") when the cell is taken. `current`
        flips to the other player only while the game is still undecided
        afterwards — on the final (winning or drawing) move it STAYS on the
        mover, so `current` is not meaningful once is_over().
        """
        _assert_playable(self)
        if self.board[index] != EMPTY:
            raise ValueError("cell is occupied")
        self.board[index] = self.current
        if self.result() is None:
            self.current = O if self.current == X else X

    def winner(self):
        """X or O when the board holds a completed LINES triple, else None."""
        return line_winner(self.board)

    def is_full(self):
        """True when no cell is EMPTY — a drawn board included."""
        return EMPTY not in self.board

    def is_over(self):
        """True once result() is not None (won or drawn)."""
        return self.result() is not None

    def result(self):
        """Terminal state: the winner's mark (X / O), "D" on a full board
        with no winner, None while the game is still undecided."""
        w = self.winner()
        if w:
            return w
        if self.is_full():
            return "D"
        return None

    def clone(self):
        """Independent copy for search branching: the board list is
        duplicated, so mutating the clone never touches the original."""
        g = NormalGame()
        g.board = self.board[:]
        g.current = self.current
        return g


def apply_move(game, move):
    # duck-typed on the MOVE shape, not the game class: Ultimate moves are
    # always (macro, micro) tuples and Normal moves are always ints — this
    # routes any implementation (NormalGame / the Rust RustUltimateGame)
    # without game.py having to import its consumers
    if isinstance(move, tuple):
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
        inner = (
            '<line x1="14" y1="14" x2="86" y2="86" stroke="currentColor" '
            'stroke-width="12" stroke-linecap="round"/>'
            '<line x1="86" y1="14" x2="14" y2="86" stroke="currentColor" '
            'stroke-width="12" stroke-linecap="round"/>'
        )
    else:
        inner = (
            '<circle cx="50" cy="50" r="36" fill="none" stroke="currentColor" '
            'stroke-width="12" stroke-linecap="round"/>'
        )
    return (
        f'<svg class="macro-win-svg" viewBox="0 0 100 100" '
        f'xmlns="http://www.w3.org/2000/svg">{inner}</svg>'
    )
