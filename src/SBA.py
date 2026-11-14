# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ultimate Tic Tac Toe — entry point, session state, and self-tests.

Game rules live in game.py and the AI engines in ai.py. This module wires
them together, keeps the per-session UI state, runs the headless
self-tests, and provides the CLI entry point.

Run:
    python src/SBA.py              # start the PySide6 desktop app (default)
    python src/SBA.py --qt         # start the desktop app explicitly
    python src/SBA.py --self-test  # run headless checks
    python src/SBA.py --debug      # verbose backend logs

Maintenance notes:
- The desktop app is the default entry (decision D1).
- Session state lives here; the module is aliased as 'SBA' so tui.py
  reuses it instead of importing a second copy.
"""

import logging
import os
import random
import sys

from game import (
    X, O,
    NormalGame, UltimateGame, apply_move,
)
from ai import (
    get_ai_move, cfg_session,
)

log = logging.getLogger('SBA')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

# Session state helpers (shared with the UI front ends and self-tests)
SESSIONS = {}

def new_session():
    return {
        'screen': 'menu',
        'game_type': 'normal',
        'mode': 'pvp',
        'first_player': 'human',
        'cvc_speed': 0.4,
        'cvc_auto': True,
        'cvc_paused': False,
        'ai_x': 'Minimax',
        'ai_o': 'MCTS',
        'mcts': cfg_session('mcts_budget', 800),
        'minimax_depth': cfg_session('minimax_depth', 4),
        'moves': [],
        'history': [],
        'step': 0,
        'game_id': 0,
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
# Shared UI helpers (imported by tui.py)
# ============================================================

AI_OPTIONS = {
    'AlphaZero': 'AlphaZero — Neural MCTS（神經網路MCTS）',
    'Random': 'Random — 隨機',
    'Basic': 'Basic — 基礎',
    'Minimax': 'Minimax — 極小化極大',
    'Minimax Pro': 'Minimax Pro — 進階極小化極大（置換表加速）',
    'Solver': 'Solver — 完美解（Normal 讀持久化解表，Ultimate 為 MCTS）',
    'MCTS': 'MCTS — 蒙地卡羅',
    'MCTS+GRAVE': 'MCTS+GRAVE — 蒙地卡羅+GRAVE',
    # MCTS+RAVE is intentionally hidden from the menu: MCTS+GRAVE is its
    # successor (same bias term, lower memory). The engine stays available
    # through get_ai_move for self-tests and the --bench comparison.
    # Flat MCTS (root-level playouts, no tree) is likewise hidden — it is a
    # research/learning baseline, not a menu-worthy opponent.
}


def t(en: str, zh: str) -> str:
    return f'{en} — {zh}'


def side_label(kind):
    if kind == 'Human':
        return t('Human (You)', '玩家 (你)')
    return f'Computer ({kind}) — 電腦 ({kind})'


# Assistant move reasons: code -> bilingual label. Codes come from
# ai.reason_for_move; both UIs translate via this map so ai.py stays
# language-free.
REASON_TEXT = {
    'Win': ('Win', '致勝'),
    'Block': ('Block', '阻擋'),
    'Fork': ('Fork', '雙威脅'),
    'Center': ('Center', '中心'),
    'Corner': ('Corner', '角落'),
    'Search': ('Search', '分析'),
    'Positional': ('Positional', '位置'),
}


# ============================================================
# Self-test
# ============================================================

def _flag_arg(name, default=None):
    """Return the string following --name, or default when absent."""
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    if i + 1 < len(sys.argv):
        return sys.argv[i + 1]
    return default


def _flag_value(name, default):
    """Return the integer following --name, or default."""
    raw = _flag_arg(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


BENCH_AI_NAMES = ('Random', 'Basic', 'Minimax', 'Minimax Pro', 'MCTS',
                  'MCTS+RAVE', 'MCTS+GRAVE', 'Flat MCTS', 'Solver',
                  'AlphaZero')


def _play_match(x_ai, o_ai, game_type, iterations, depth):
    game = NormalGame() if game_type == 'normal' else UltimateGame()
    guard = 0
    while not game.is_over() and guard < 1000:
        ai = x_ai if game.current == X else o_ai
        apply_move(game, get_ai_move(game, ai, iterations, depth))
        guard += 1
    return game.result()


def bench(games=20, iterations=1600, game_type='ultimate', seed=12345,
          ai_a=None, ai_b=None, depth=4):
    """Head-to-head score comparison: win = 1, draw = 0.5.

    ``A score%`` = (A wins + draws/2) / games, so draws are not ignored like a
    plain win rate would. Useful for ranking engines that mostly draw.
    / 對戰得分比較：勝 = 1、和 = 0.5。``A score%`` =（勝場 + 和局/2）/ 總局數，
    和局不會被當作無效樣本，適合比較「多半和棋」的引擎強度。
    """
    random.seed(seed)
    if (ai_a is None) != (ai_b is None):
        print('--ai-a and --ai-b must be given together')
        return 2
    if ai_a is not None:
        for name in (ai_a, ai_b):
            if name not in BENCH_AI_NAMES:
                print(f'Unknown AI: {name!r}. Choose from: {", ".join(BENCH_AI_NAMES)}')
                return 2
        matchups = [(ai_a, ai_b)]
    else:
        matchups = [('MCTS', 'MCTS+RAVE'), ('MCTS', 'MCTS+GRAVE'),
                    ('MCTS+RAVE', 'MCTS+GRAVE')]
    if len(matchups) == 1:
        a, b = matchups[0]
        title = (f'Benchmark — {a} vs {b} · {game_type} · {games} games · '
                 f'{iterations} sims/move · depth {depth}')
    else:
        title = (f'MCTS family benchmark — {game_type} · {games} games/pair · '
                 f'{iterations} sims/move')
    header = ('A', 'B', 'A wins', 'draws', 'B wins', 'A score%')
    rows = []
    for a, b in matchups:
        aw = dw = bw = 0
        for k in range(games):
            a_first = k % 2 == 0  # alternate who moves first per game
            x_ai, o_ai = (a, b) if a_first else (b, a)
            r = _play_match(x_ai, o_ai, game_type, iterations, depth)
            if (r == X and a_first) or (r == O and not a_first):
                aw += 1
            elif r == 'D':
                dw += 1
            else:
                bw += 1
        pct = 100.0 * (aw + 0.5 * dw) / games
        pct_cell = (f'{pct:.1f}%', 'bold green' if pct > 55 else
                    ('bold red' if pct < 45 else 'bold'))
        rows.append([a, b, str(aw), str(dw), str(bw), pct_cell])
    try:
        import cli_output
        con = cli_output.console()
        table = cli_output.bench_table(title, header, rows)
    except ImportError:
        con, table = None, None
    if table is not None and con is not None:
        con.print(table)
    else:
        print(title)
        print(f'{"A":<14}{"B":<14}{"A wins":>7}{"draws":>6}{"B wins":>7}'
              f'{"A score%":>9}')
        print('-' * 57)
        for a, b, aw, dw, bw, pct_cell in rows:
            print(f'{a:<14}{b:<14}{aw:7d}{dw:6d}{bw:7d}'
                  f'{float(pct_cell[0][:-1]):8.1f}%')
    return 0


def _build_tables():
    """--build-tables: persist the solved Normal tablebase to
    data/tt_normal.bin (idempotent; ships tracked so no rebuild needed)."""
    import hashlib
    import ai
    path, stats = ai.write_tablebase_file()
    with open(path, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    empty = {0: 'draw (theoretical result)', 1: 'X win', -1: 'O loss'}
    try:
        import cli_output
        con = cli_output.console()
    except ImportError:
        con = None
    if con is not None:
        from rich import box
        from rich.table import Table
        t = Table(title='Normal tablebase — data/tt_normal.bin',
                  box=box.SIMPLE_HEAVY, title_justify='left', pad_edge=False)
        t.add_column('field', justify='left')
        t.add_column('value', justify='left')
        t.add_row('file', str(path))
        t.add_row('size', f"{stats['bytes']} bytes")
        t.add_row('reachable', f"{stats['reachable']} positions "
                               f"({stats['terminal']} terminal)")
        t.add_row('empty board', empty.get(stats['empty_board'], '?'))
        t.add_row('sha256', digest)
        con.print(t)
    else:
        print(f'tablebase -> {path}')
        print(f"  {stats['bytes']} bytes, {stats['reachable']} reachable "
              f"positions ({stats['terminal']} terminal)")
        print(f"  empty board: {empty.get(stats['empty_board'], '?')}")
        print(f'  sha256  {digest}')
    return 0


def main():
    if '--build-tables' in sys.argv:
        return _build_tables()
    if '--self-test' in sys.argv:
        try:
            import pytest
        except ImportError:
            print('pytest is required for --self-test:  python -m pip install pytest')
            return 1
        tests = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'tests')
        sys.exit(pytest.main(['-q', tests]))
    if '--train-az' in sys.argv:
        import alphazero
        rest = [a for a in sys.argv[1:] if a != '--train-az']
        sys.exit(alphazero.main(['train'] + rest))
    if '--bench' in sys.argv:
        sys.exit(bench(
            games=_flag_value('--games', 20),
            iterations=_flag_value('--iters', 1600),
            game_type='normal' if '--normal' in sys.argv else 'ultimate',
            ai_a=_flag_arg('--ai-a'),
            ai_b=_flag_arg('--ai-b'),
            depth=_flag_value('--depth', 4),
        ))
    # When SBA.py is the entry script it is '__main__'; register it under the
    # canonical name so qtui.py / tui.py reuse this module instead of
    # importing a second copy.
    sys.modules.setdefault('SBA', sys.modules['__main__'])
    # Default entry (also explicit via --qt): PySide6 desktop app.
    import qtui
    qtui.main()


if __name__ == '__main__':
    main()
