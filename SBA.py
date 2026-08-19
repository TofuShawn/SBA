# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ultimate Tic Tac Toe — entry point, session state, and self-tests.

Game rules live in game.py, the AI engines in ai.py, and the NiceGUI web UI
in webui.py. This module wires them together, keeps the per-session UI state,
runs the headless self-tests, and provides the CLI entry point.

Run:
    python SBA.py              # start the PySide6 desktop app (default)
    python SBA.py --qt         # start the desktop app explicitly
    python SBA.py --web        # start the NiceGUI web app at http://127.0.0.1:8080
    python SBA.py --self-test  # run headless checks
    python SBA.py --debug      # verbose backend logs

Maintenance notes:
- The desktop app is the default entry; the NiceGUI web server only starts
  with --web or the desktop's "Enable NiceGUI Web UI" switch (decision D1).
- Session state lives here; the module is aliased as 'SBA' so qtui/webui
  reuse it instead of importing a second copy.
"""

import logging
import random
import sys

from game import (
    X, O, EMPTY,
    NormalGame, UltimateGame, apply_move,
    win_badge_svg, line_coords, win_segment, macro_center,
)
from ai import (
    get_basic_move, minimax_move_normal, mcts_move,
    solver_move, minimax_pro_move, mcts_rave_move, mcts_grave_move,
    build_tablebase, _board_key, get_ai_move,
)

log = logging.getLogger('SBA')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

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
    check('mcts+grave: normal immediate win', mcts_grave_move(g, 1500) == 2)

    gu = UltimateGame()
    gu.macro = [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    gu.micro[2] = [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    gu.active_macro = 2
    gu.current = X
    check('minimax pro: ultimate game-winning move', minimax_pro_move(gu, depth=4) == (2, 2))
    check('mcts+rave: ultimate game-winning move', mcts_rave_move(gu, 2000) == (2, 2))
    check('mcts+grave: ultimate game-winning move', mcts_grave_move(gu, 2000) == (2, 2))

    seg = win_segment((0, 1, 2), line_coords)
    check('geometry: horizontal line spans full width', seg == ((4, 20), (96, 20)))
    seg = win_segment((0, 4, 8), line_coords)
    check('geometry: diagonal line spans corners', seg == ((7, 7), (93, 93)))
    seg = win_segment((0, 4, 8), macro_center)
    check('geometry: macro diagonal spans corners', seg == ((7, 7), (93, 93)))
    check('geometry: badge svg has cross/circle',
          '<line' in win_badge_svg(X) and '<circle' in win_badge_svg(O))

    import alphazero
    m9 = alphazero.train('ultimate', games=2, sims=6, quiet=True, save=False)
    check('alphazero: smoke train ultimate', m9 is not None)
    g = UltimateGame()
    check('alphazero: legal move ultimate', alphazero.select_move(g, m9, 20) in g.legal_moves())
    g = NormalGame()
    mv = get_ai_move(g, 'AlphaZero', 20)
    check('alphazero: get_ai_move dispatches', mv in g.legal_moves())
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
                       ('ultimate', 'MCTS+RAVE', 'Random'),
                       ('ultimate', 'MCTS+GRAVE', 'Random')):
        game = NormalGame() if gt == 'normal' else UltimateGame()
        guard = 0
        while not game.is_over() and guard < 500:
            ai = ax if game.current == X else ao
            apply_move(game, get_ai_move(game, ai, 300))
            guard += 1
        check(f'cvc {gt} ({ax} vs {ao}) terminates', game.is_over() and guard <= 500)

    for gt in ('normal', 'ultimate'):
        for ai in ('Random', 'Basic', 'Minimax', 'Minimax Pro', 'MCTS',
                   'MCTS+RAVE', 'MCTS+GRAVE', 'Solver'):
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


def _flag_value(name, default):
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
        return sys.argv[i + 1]
    return default


def bench(games=30, iterations=300, game_type='ultimate', seed=12345):
    """Round-robin win-rate comparison of the MCTS family engines."""
    random.seed(seed)
    engines = ['MCTS', 'MCTS+RAVE', 'MCTS+GRAVE']
    print(f'\nMCTS benchmark — {game_type} · {games} games/pair · {iterations} sims')
    header = f'{"A":<12}{"B":<14}{"A wins":>7}{"draws":>6}{"B wins":>7}{"A win%":>8}'
    print(header)
    print('-' * len(header))
    for i, a in enumerate(engines):
        for b in engines[i + 1:]:
            aw = dw = bw = 0
            for k in range(games):
                a_first = k % 2 == 0  # alternate who moves first per game
                x_ai, o_ai = (a, b) if a_first else (b, a)
                game = NormalGame() if game_type == 'normal' else UltimateGame()
                guard = 0
                while not game.is_over() and guard < 1000:
                    ai = x_ai if game.current == X else o_ai
                    apply_move(game, get_ai_move(game, ai, iterations))
                    guard += 1
                r = game.result()
                if (r == X and a_first) or (r == O and not a_first):
                    aw += 1
                elif r == 'D':
                    dw += 1
                else:
                    bw += 1
            pct = 100.0 * aw / games
            print(f'{a:<12}{b:<14}{aw:7d}{dw:6d}{bw:7d}{pct:7.1f}%')
    return 0


def main():
    if '--self-test' in sys.argv:
        sys.exit(self_test())
    if '--train-az' in sys.argv:
        import alphazero
        rest = [a for a in sys.argv[1:] if a != '--train-az']
        sys.exit(alphazero.main(['train'] + rest))
    if '--bench' in sys.argv:
        sys.exit(bench(
            games=int(_flag_value('--games', '30')),
            iterations=int(_flag_value('--iters', '300')),
            game_type='normal' if '--normal' in sys.argv else 'ultimate',
        ))
    # When SBA.py is the entry script it is '__main__'; register it under the
    # canonical name so qtui.py / webui.py reuse this module instead of
    # importing a second copy.
    sys.modules.setdefault('SBA', sys.modules['__main__'])
    if '--web' in sys.argv:
        import webui
        webui.run()
        return
    # Default entry (also explicit via --qt): PySide6 desktop app. The
    # NiceGUI web UI is opt-in only (see the --web flag or the desktop
    # app's "Enable NiceGUI Web UI" switch).
    import qtui
    qtui.main()


if __name__ == '__main__':
    main()
