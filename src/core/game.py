# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ultimate Tic Tac Toe — game rules and board helpers.

Defines the Normal (3x3) and Ultimate (9x9) game rules, move application, and
the board geometry / win-badge helpers used by the AI engines, analysis,
self-tests, and the UIs. Pure logic with no UI-framework dependency.

Maintenance notes:
- Keep this file free of UI/engine imports (decision D2) — everything else
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


def _assert_ultimate_routing(game, macro):
    """Reject a move that ignores the forced macro routing or targets a closed
    macro. O(1): deliberately avoids building the full legal_moves() list,
    since make_move sits on the Python search hot path."""
    active = game.active_macro
    if active is not None and game.macro_open(active):
        if macro != active:
            raise ValueError("move must be in the routed macro")
    elif not game.macro_open(macro):
        raise ValueError("macro is not open")


# ============================================================
# Game engines
# ============================================================


class NormalGame:
    """Plain 3x3 tic-tac-toe — the drift-free baseline ruleset.

    State: `board` is a list of 9 cells (X / O / EMPTY, row-major from the
    top-left) and `current` is the mark to move; X always opens the game.

    This class and UltimateGame share the duck-typed contract every consumer
    relies on: the session and the UIs drive a game purely through
    legal_moves / make_move / is_over / result, and the search engines work
    through clone() + make_move. Moves are plain cell indices (ints) — the
    move SHAPE is what game.apply_move dispatches on, so alternative rules
    implementations stay interchangeable (RustUltimateGame in
    src/az/rustbridge.py implements the same contract over the Rust
    bitboard state for the ultimate rules).
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


class UltimateGame:
    """Ultimate tic-tac-toe: nine 3x3 micro boards scored into a 3x3 macro
    board — the reference implementation of the rules.

    State: `micro` holds 9 rows of 9 cells (X / O / EMPTY), `macro` the 9
    micro-board winners (X / O, "D" for a micro drawn full, EMPTY while
    still open), `current` is the mark to move and `active_macro` the macro
    the next player is routed into (None = free choice among open macros);
    X always opens the game.

    Duck-typed contract shared by every rules implementation: the session
    and the UIs drive a game purely through legal_moves / make_move /
    is_over / result (ultimate moves are (macro, micro) tuples — the move
    SHAPE is what game.apply_move dispatches on), and the search engines
    work through clone() + make_move. RustUltimateGame
    (src/az/rustbridge.py) implements this same contract over the Rust
    bitboard state (fuzz-checked against this class in
    tests/test_rustgame.py); BitUltimateGame is the pure-Python bit twin.
    """

    def __init__(self):
        self.micro = [[EMPTY] * 9 for _ in range(9)]
        self.macro = [EMPTY] * 9
        self.current = X
        self.active_macro = None

    def macro_open(self, m):
        """True while macro board m is playable: no badge yet AND at least
        one EMPTY cell left in its micro row."""
        return self.macro[m] == EMPTY and any(c == EMPTY for c in self.micro[m])

    def legal_moves(self):
        """Every (macro, micro) tuple the player to move may play.

        When routing points at an open macro (active_macro set), the moves
        are that macro's empty cells only; otherwise the empty cells of
        every open macro. A won game returns [] even with open macros left.
        """
        if self.winner() is not None:
            return []
        if self.active_macro is not None and self.macro_open(self.active_macro):
            return [
                (self.active_macro, i)
                for i in range(9)
                if self.micro[self.active_macro][i] == EMPTY
            ]
        moves = []
        for m in range(9):
            if not self.macro_open(m):
                continue
            moves.extend((m, i) for i in range(9) if self.micro[m][i] == EMPTY)
        return moves

    def micro_winner(self, m):
        """X / O when micro board m holds a winning line, else None."""
        return line_winner(self.micro[m])

    def make_move(self, macro, micro):
        """Place `current`'s mark in cell `micro` of micro board `macro` on
        this game (no copy), resolve the micro board's badge, then hand
        over.

        Routing: when active_macro is set and still open, `macro` must be
        exactly it; otherwise any open macro may be played. The played CELL
        routes the opponent: active_macro becomes `micro` if that macro
        board is still open after this move, None (free routing) when it is
        won or full. `current` flips unconditionally on every accepted
        move.

        Raises ValueError in check order: "game is over" (result() already
        set), "move must be in the routed macro", "macro is not open"
        (free routing into a closed macro), "cell is occupied".
        """
        _assert_playable(self)
        _assert_ultimate_routing(self, macro)
        if self.micro[macro][micro] != EMPTY:
            raise ValueError("cell is occupied")
        self.micro[macro][micro] = self.current
        w = self.micro_winner(macro)
        if w:
            self.macro[macro] = w
        elif all(c != EMPTY for c in self.micro[macro]):
            self.macro[macro] = "D"
        self.active_macro = micro if self.macro_open(micro) else None
        self.current = O if self.current == X else X

    def winner(self):
        """X / O when the macro badges form a winning LINES line, else
        None — only decided micro boards score."""
        for a, b, c in LINES:
            if self.macro[a] in (X, O) and self.macro[a] == self.macro[b] == self.macro[c]:
                return self.macro[a]
        return None

    def is_full(self):
        """True when every macro is closed (won or drawn-full) — the board
        can be full with no winner."""
        return all(not self.macro_open(m) for m in range(9))

    def is_over(self):
        """True once result() is not None (macro line won or board full)."""
        return self.result() is not None

    def result(self):
        """Terminal state: the winner's mark (X / O), "D" when every macro
        is closed with no winning macro line, None while play continues."""
        w = self.winner()
        if w:
            return w
        if self.is_full():
            return "D"
        return None

    def clone(self):
        """Independent deep copy for search branching: every micro row and
        the macro list are duplicated, so mutating the clone never touches
        the original."""
        g = UltimateGame()
        g.micro = [row[:] for row in self.micro]
        g.macro = self.macro[:]
        g.current = self.current
        g.active_macro = self.active_macro
        return g


class BitUltimateGame:
    """Ultimate board stored as bit masks (X/O 81-bit, macro 9-bit).

    Same interface as UltimateGame (legal_moves / make_move / result /
    clone / current / active_macro / micro / macro) so the MCTS family can
    search on the bit representation and hand moves back to the list UI.
    """

    __slots__ = ("x", "o", "mx", "mo", "current", "active_macro")

    def __init__(self):
        self.x = 0
        self.o = 0
        self.mx = 0
        self.mo = 0
        self.current = X
        self.active_macro = None

    @classmethod
    def from_game(cls, game):
        g = cls()
        for m in range(9):
            for i in range(9):
                c = game.micro[m][i]
                if c == X:
                    g.x |= 1 << (m * 9 + i)
                elif c == O:
                    g.o |= 1 << (m * 9 + i)
            if game.macro[m] == X:
                g.mx |= 1 << m
            elif game.macro[m] == O:
                g.mo |= 1 << m
        g.current = game.current
        g.active_macro = game.active_macro
        return g

    def clone(self):
        g = BitUltimateGame()
        g.x, g.o, g.mx, g.mo = self.x, self.o, self.mx, self.mo
        g.current = self.current
        g.active_macro = self.active_macro
        return g

    def _micro_bits(self, m):
        return ((self.x | self.o) >> (m * 9)) & 0x1FF

    @staticmethod
    def _has_win_line(mask):
        """True when `mask` (9 bits) contains a complete winning line."""
        for a, b, c in LINES:
            line = (1 << a) | (1 << b) | (1 << c)
            if mask & line == line:
                return True
        return False

    def micro_winner(self, m):
        base = m * 9
        if self._has_win_line((self.x >> base) & 0x1FF):
            return X
        if self._has_win_line((self.o >> base) & 0x1FF):
            return O
        return None

    def macro_open(self, m):
        if (self.mx | self.mo) & (1 << m):
            return False
        return self._micro_bits(m) != 0x1FF

    def legal_moves(self):
        if self.winner() is not None:
            return []
        if self.active_macro is not None and self.macro_open(self.active_macro):
            taken = self._micro_bits(self.active_macro)
            m = self.active_macro
            return [(m, i) for i in range(9) if not (taken >> i) & 1]
        moves = []
        for m in range(9):
            if not self.macro_open(m):
                continue
            taken = self._micro_bits(m)
            moves.extend((m, i) for i in range(9) if not (taken >> i) & 1)
        return moves

    def make_move(self, m, i):
        _assert_playable(self)
        _assert_ultimate_routing(self, m)
        bit = 1 << (m * 9 + i)
        if (self.x | self.o) & bit:
            raise ValueError("cell is occupied")
        if self.current == X:
            self.x |= bit
        else:
            self.o |= bit
        w = self.micro_winner(m)
        if w == X:
            self.mx |= 1 << m
        elif w == O:
            self.mo |= 1 << m
        self.active_macro = i if self.macro_open(i) else None
        self.current = O if self.current == X else X

    def winner(self):
        for a, b, c in LINES:
            line = (1 << a) | (1 << b) | (1 << c)
            if self.mx & line == line:
                return X
            if self.mo & line == line:
                return O
        return None

    def is_full(self):
        return all(not self.macro_open(m) for m in range(9))

    def result(self):
        w = self.winner()
        if w:
            return w
        if self.is_full():
            return "D"
        return None

    def is_over(self):
        return self.result() is not None

    @property
    def micro(self):
        out = []
        for m in range(9):
            row = []
            base = m * 9
            for i in range(9):
                bit = 1 << (base + i)
                row.append(X if self.x & bit else (O if self.o & bit else EMPTY))
            out.append(row)
        return out

    @property
    def macro(self):
        return [
            X
            if self.mx & (1 << m)
            else (O if self.mo & (1 << m) else ("D" if self._micro_bits(m) == 0x1FF else EMPTY))
            for m in range(9)
        ]


def apply_move(game, move):
    # duck-typed on the MOVE shape, not the game class: Ultimate moves are
    # always (macro, micro) tuples and Normal moves are always ints — this
    # routes any implementation (UltimateGame / BitUltimateGame / the D48
    # Rust wrapper) without game.py having to import its consumers
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
