# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ultimate Tic Tac Toe — NiceGUI web UI.

Imports the game engines and AI from SBA.py and renders the interactive UI:
the setup menu (game type / mode / AI levels), the board, the AI assistant
analysis panel, and CvC controls.

Run the app via `python SBA.py` (or run.bat). This module is imported on
demand by SBA.main(); it can also be run directly with `python webui.py`.

Maintenance notes:
- NiceGUI is opt-in via --web or the desktop switch (decision D1).
- Solver and MCTS+RAVE are intentionally absent from the AI menu (D3, D8).
- Mobile layout lives in static/styles.css media queries.
"""

import asyncio
import logging
import sys
import uuid
from pathlib import Path

try:
    from nicegui import app, background_tasks, ui
except ModuleNotFoundError:
    print('NiceGUI is not installed for this Python interpreter.')
    print('Use run.bat (project virtual environment), or install with:  python -m pip install nicegui')
    raise SystemExit(1)

from game import (
    X, O,
    NormalGame, UltimateGame, apply_move,
    win_badge_svg, micro_win_line, line_coords, win_segment, macro_center,
)
from ai import (
    get_ai_move, analyze_position, position_win_rates, move_text,
)
from SBA import (
    AI_OPTIONS, SESSIONS, current_side_type, is_ai_turn, log,
    new_session, side_label, side_types, t,
)


def _history_point(game):
    """Win-rate triplet for the current position.

    Exact (Normal tablebase, or any terminal position) is computed inline;
    a non-terminal Ultimate position defers to the assistant's full-budget
    search (the None placeholder is filled when the analysis completes).
    """
    if isinstance(game, NormalGame) or game.is_over():
        return tuple(position_win_rates(game, 0))
    return None


# Serve static/styles.css as a real stylesheet (external <link>, no inline blob).
app.add_static_files('/assets', Path(__file__).parent / 'static')
ui.add_head_html('<link rel="stylesheet" href="/assets/styles.css">', shared=True)

def set_mark(el, player):
    el.classes(remove='mark-x mark-o')
    el.classes(add='mark-x' if player == X else 'mark-o')
    el.set_text('✕' if player == X else '○')

# ============================================================
# Web UI
# ============================================================

@ui.page('/')
def main_page():
    sid = app.storage.user.get('sid')
    if sid is None or sid not in SESSIONS:
        sid = str(uuid.uuid4())
        app.storage.user['sid'] = sid
        SESSIONS[sid] = new_session()
        if len(SESSIONS) > 500:  # backstop: drop the oldest session
            SESSIONS.pop(next(iter(SESSIONS)))
    session = SESSIONS[sid]
    # Drop this browser's session when it disconnects so a long-running
    # server does not accumulate one entry per visitor.
    app.on_disconnect(lambda _client, sid=sid: SESSIONS.pop(sid, None))
    cell_refs = {}

    ui.colors(primary='#6750A4', secondary='#625B71', accent='#B3261E')
    dark = ui.dark_mode(value=False)

    with ui.header().classes('items-center justify-between'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('grid_on').classes('text-primary')
            ui.label('Ultimate Tic Tac Toe — 終極井字棋').classes('text-h6 q-ma-none')
        with ui.row().classes('items-center gap-4'):
            new_game_btn = ui.button(t('New Game', '新遊戲'), icon='replay',
                                     on_click=lambda: start_game()).props('flat')
            new_game_btn.set_visibility(False)
            play_again_btn = ui.button(t('Play Again', '再玩一次'), icon='replay',
                                       on_click=lambda: start_game()).props('unelevated')
            play_again_btn.set_visibility(False)
            back_btn = ui.button(t('Back to Menu', '返回選單'), icon='arrow_back',
                                 on_click=lambda: show_menu())
            back_btn.set_visibility(False)
            ui.switch(t('Dark', '深色')).bind_value(dark)

    content = ui.column().classes('w-full items-center p-3 gap-4 sm:p-6 sm:gap-6')

    def show_menu():
        session['screen'] = 'menu'
        back_btn.set_visibility(False)
        play_again_btn.set_visibility(False)
        new_game_btn.set_visibility(False)
        content.clear()
        with content:
            build_menu()

    def show_game():
        session['screen'] = 'game'
        new_game_btn.set_visibility(True)
        play_again_btn.set_visibility(False)
        back_btn.set_visibility(True)
        content.clear()
        with content:
            build_game()

    def build_menu():
        with ui.card().classes('w-full max-w-xl sm:max-w-3xl'):
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

                ui.label('AlphaZero — 神經網路 MCTS（僅終極模式）· 重新訓練: python alphazero.py train').classes(
                    'text-caption text-grey q-mb-0')

                mcts_label = ui.label(
                    t('MCTS Strength', 'MCTS 強度') + f': {session["mcts"]}')
                mcts_slider = ui.slider(min=200, max=3000, step=100,
                                        value=session['mcts']).props('label-always')
                mcts_slider.on_value_change(lambda e: (
                    session.update(mcts=int(e.value)),
                    mcts_label.set_text(t('MCTS Strength', 'MCTS 強度') + f': {int(e.value)}'),
                ))

                mm_label = ui.label(
                    t('Minimax Depth (Ultimate)', 'Minimax 深度（終極模式）')
                    + f': {session["minimax_depth"]}')
                mm_slider = ui.slider(min=2, max=6, step=1,
                                      value=session['minimax_depth']).props('label-always')
                mm_slider.on_value_change(lambda e: (
                    session.update(minimax_depth=int(e.value)),
                    mm_label.set_text(
                        t('Minimax Depth (Ultimate)', 'Minimax 深度（終極模式）')
                        + f': {int(e.value)}'),
                ))

                ui.button(t('Start Game', '開始遊戲'), icon='play_arrow',
                          on_click=start_game).props('unelevated').classes('w-full')

    def start_game():
        session['game'] = (NormalGame() if session['game_type'] == 'normal'
                           else UltimateGame())
        session['game_id'] += 1
        session['moves'] = []
        session['history'] = [_history_point(session['game'])]
        session['step'] = 0
        session['cvc_paused'] = False
        session['analysis_gen'] += 1
        session['analyzing'] = False
        session['reanalyze'] = False
        session['ai_busy'] = None
        x_type, o_type = side_types(session)
        log.info('Game started: %s, mode=%s, X=%s, O=%s, mcts=%d, mm_depth=%d',
                 'Normal' if session['game_type'] == 'normal' else 'Ultimate',
                 session['mode'], x_type, o_type,
                 session['mcts'], session.get('minimax_depth', 3))
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
                thinking = session.get('ai_busy') == id(game)
                side = t('Player X', '玩家 X') if player == X else t('Player O', '玩家 O')
                action = (t('your move', '輪到你') if current_side_type(session) == 'Human'
                          else t('thinking...', '思考中'))
                status_spinner.set_visibility(thinking)
                status_text.set_text(f'{side} · 🤔 {action}' if thinking else f'{side} · {action}')
                if session['step'] < len(session['moves']):
                    status_text.set_text(
                        f'{status_text.text} · {t("history", "歷史")} '
                        f"{session['step']}/{len(session['moves'])}")
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
                                    fill = ui.element('div')
                                    fill.mark(f'macro-fill-{m}')
                                    fill.classes('macro-win-fill'
                                                 + (' macro-win-fill-x' if game.macro[m] == X
                                                    else ' macro-win-fill-o'))
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
            side = game.current
            apply_move(game, move)
            log.info('Human move: %s -> %s [%s]', side, move_text(move),
                     'Normal' if isinstance(game, NormalGame) else 'Ultimate')
            record_move(move)
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
            record_move(move)
            session['analysis_gen'] += 1
            render_board()
            refresh_status()
            trigger_analysis()
            if game.is_over():
                show_result()
            update_cvc_controls()
            return True

        async def finish_ai_move(ai_type, budget, depth):
            try:
                move = await asyncio.to_thread(
                    get_ai_move, game, ai_type, budget, depth)
            except Exception as e:
                log.error('AI move error (%s): %s', ai_type, e)
                if session.get('ai_busy') == id(game):
                    session['ai_busy'] = None
                    refresh_status()
                return
            if session.get('ai_busy') == id(game):
                session['ai_busy'] = None
            side = game.current
            log.info('AI move: %s (%s) -> %s [%s]', side, ai_type, move_text(move),
                     'Normal' if isinstance(game, NormalGame) else 'Ultimate')
            apply_ai_move(move)

        def step_ai_move():
            if session['screen'] != 'game' or game is not session['game']:
                return False
            if game.is_over() or not is_ai_turn(session):
                return False
            if session.get('ai_busy') is not None:
                return False
            x_type, o_type = side_types(session)
            ai_type = x_type if game.current == X else o_type
            session['ai_busy'] = id(game)
            background_tasks.create(finish_ai_move(
                ai_type, session['mcts'], session.get('minimax_depth', 3)))
            refresh_status()
            return False

        def ai_loop():
            if session.get('cvc_paused', False):
                return
            step_ai_move()

        def update_cvc_controls():
            cvc = session['mode'] == 'cvc'
            pause_btn.set_visibility(cvc)
            next_btn.set_visibility(cvc)
            revert_btn.set_enabled(session['step'] > 0)
            if session['mode'] != 'cvc':
                return
            ai_turn = not game.is_over() and is_ai_turn(session)
            paused = session.get('cvc_paused', False)
            rewound = session['step'] < len(session['moves'])
            step_btn.set_enabled(rewound or (paused and ai_turn
                                             and session.get('ai_busy') is None))
            next_btn.set_enabled(rewound or (
                ai_turn and session.get('ai_busy') is None))
            pause_btn.set_text(t('Resume', '繼續') if paused else t('Pause', '暫停'))

        def on_speed_change(e):
            speed = float(e.value)
            session['cvc_speed'] = speed
            speed_label.set_text(t('Speed', '速度') + f': {speed:.1f}s')
            game_timer.interval = speed

        def record_move(move):
            moves = session['moves']
            if session['step'] < len(moves):  # rewound: branch the history
                del moves[session['step']:]
                del session['history'][session['step'] + 1:]
            moves.append(move)
            session['history'].append(_history_point(game))
            session['step'] = len(moves)
            render_history()

        def go_to_step(k):
            moves = session['moves']
            k = max(0, min(int(k), len(moves)))
            if k == session['step']:
                return
            game_timer.stop()
            session['cvc_paused'] = k < len(moves)
            g = NormalGame() if session['game_type'] == 'normal' else UltimateGame()
            for mv in moves[:k]:
                apply_move(g, mv)
            session['game'] = g
            session['step'] = k
            session['analysis_gen'] += 1
            render_board()
            refresh_status()
            update_cvc_controls()
            render_history()
            trigger_analysis()
            if (not session.get('cvc_paused', False) and session['mode'] == 'cvc'
                    and not g.is_over() and is_ai_turn(session)):
                game_timer.start()

        def render_history():
            moves = session['moves']
            history = session['history']
            n = len(moves)
            hist_slider.min = 0
            hist_slider.max = n
            hist_slider.value = session['step']
            hist_slider_label.set_text(f"{session['step']} / {n}")
            hist_chart.options['xAxis']['data'] = list(range(len(history)))
            hist_chart.options['series'][0]['data'] = [
                (round(point[0] * 100, 1) if point is not None else None)
                for point in history]
            if history and history[session['step']] is not None:
                hist_chart.options['series'][0]['markPoint'] = {
                    'data': [{
                        'coord': [session['step'],
                                  round(history[session['step']][0] * 100, 1)],
                        'symbol': 'circle', 'symbolSize': 10,
                        'itemStyle': {'color': '#B3261E'},
                    }]}
            else:
                hist_chart.options['series'][0]['markPoint'] = {'data': []}
            hist_chart.update()

        def on_pause_click():
            session['cvc_paused'] = not session.get('cvc_paused', False)
            if session['cvc_paused']:
                game_timer.stop()
                pause_btn.set_text(t('Resume', '繼續'))
            else:
                pause_btn.set_text(t('Pause', '暫停'))
                if (session['mode'] == 'cvc' and not game.is_over()
                        and is_ai_turn(session)):
                    game_timer.start()
            update_cvc_controls()

        def step_btn_click():
            if session['step'] < len(session['moves']):
                go_to_step(session['step'] + 1)
            else:
                step_ai_move()

        def on_next_click():
            step_btn_click()

        def render_analysis(items, rates):
            analysis_ui.clear()
            if not items:
                with analysis_ui:
                    ui.label(t('No moves to analyze', '沒有可分析的棋步')).classes(
                        'text-caption text-grey')
                return
            with analysis_ui:
                x, d, o = rates
                with ui.element('div').classes('w-full flex rounded overflow-hidden').style(
                        'height: 14px; border: 1px solid rgba(0,0,0,0.08);'):
                    ui.element('div').style(
                        f'width:{max(x, 0.0) * 100:.1f}%; background:#6750A4;')
                    ui.element('div').style(
                        f'width:{max(d, 0.0) * 100:.1f}%; background:#938F99;')
                    ui.element('div').style(
                        f'width:{max(o, 0.0) * 100:.1f}%; background:#B3261E;')
                ui.label(f'X {x:.0%} · 和 {d:.0%} · O {o:.0%}').classes(
                    'text-caption text-grey')
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
            step = session['step']
            gid = session['game_id']
            session['analyzing'] = True
            snapshot = current_game.clone()
            analysis_ui.clear()
            with analysis_ui:
                with ui.row().classes('items-center gap-2'):
                    ui.spinner(size='sm')
                    ui.label(t('Analyzing...', '分析中...')).classes('text-caption')
            budget = session['mcts'] if isinstance(snapshot, UltimateGame) else 0
            items, rates = await asyncio.to_thread(
                analyze_position, snapshot, budget)
            if gid == session['game_id'] and 0 <= step < len(session['history']) \
                    and session['history'][step] is None:
                session['history'][step] = tuple(rates)
                render_history()
            if session['game'] is not current_game:
                session['analyzing'] = False
                return
            session['analyzing'] = False
            if (gen == session['analysis_gen'] and session['screen'] == 'game'
                    and session['assistant_enabled']):
                render_analysis(items, rates)
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
            hist_chart.set_visibility(value)
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
            log.info('Game over: %s [%s]', result,
                     'Normal' if isinstance(game, NormalGame) else 'Ultimate')
            # No popup: the header Play Again button appears instead.
            play_again_btn.set_visibility(True)
            new_game_btn.set_visibility(False)

        with ui.row().classes('w-full justify-center gap-4 items-start flex-wrap sm:gap-6'):
            with ui.column().classes('items-center gap-3'):
                with ui.row().classes('items-center gap-2'):
                    status_mark = ui.label('').classes('mark-chip')
                    status_spinner = ui.spinner(size='sm')
                    status_spinner.set_visibility(False)
                    status_text = ui.label('').classes('text-h6')
                with ui.row().classes('w-full items-center gap-2'):
                    revert_btn = ui.button(
                        t('Revert', '回退'), icon='undo',
                        on_click=lambda: go_to_step(session['step'] - 1)).props('flat')
                    pause_btn = ui.button(
                        t('Pause', '暫停'), icon='pause',
                        on_click=on_pause_click).props('flat')
                    next_btn = ui.button(
                        t('Next Step', '下一步'), icon='skip_next',
                        on_click=on_next_click).props('flat')
                    hist_slider = ui.slider(min=0, max=0, value=0).props('label-always')
                    hist_slider.classes('flex-1')
                    hist_slider.on_value_change(lambda e: go_to_step(int(e.value)))
                    hist_slider_label = ui.label('0 / 0').classes('text-caption text-grey')
                board_ui = ui.element('div').classes('board-wrap')
                render_board()
            with ui.column().classes('w-full max-w-sm gap-3 sm:w-80 sm:max-w-none'):
                with ui.card().classes('w-full'):
                    analysis_ui = ui.column().classes('gap-1')
                    ui.label(t('Click a move to highlight it on the board',
                               '點擊棋步可在棋盤上標示')).classes(
                        'text-caption text-grey q-mb-xs')
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
                            step_btn = ui.button(
                                t('Step / Next Move', '下一步'), icon='skip_next',
                                on_click=step_btn_click).props('flat')
                            step_btn.mark('step-btn')
                            step_btn.disable()

        # Bottom: win-rate chart (single line; above 50% = X favored) + steps
        with ui.card().classes('w-full'):
            hist_chart = ui.echart({
                'grid': {'left': 40, 'right': 16, 'top': 24, 'bottom': 24},
                'xAxis': {'type': 'category', 'data': [],
                          'name': t('Step', '步')},
                'yAxis': {'type': 'value', 'min': 0, 'max': 100, 'name': '%'},
                'series': [{
                    'name': t('X win rate', 'X 勝率'),
                    'type': 'line', 'smooth': True, 'showSymbol': True, 'data': [],
                    'itemStyle': {'color': '#6750A4'},
                    'lineStyle': {'color': '#6750A4'},
                    'markPoint': {'data': []},
                    'markLine': {
                        'silent': True,
                        'symbol': 'none',
                        'data': [{'yAxis': 50}],
                        'lineStyle': {'type': 'dashed', 'color': '#79747E'},
                    },
                }],
                'legend': {'show': True, 'top': 0},
            }).classes('w-full').style('height: 200px')
            hist_chart.set_visibility(session.get('assistant_enabled', True))

        refresh_status()
        trigger_analysis()
        update_cvc_controls()
        render_history()
        game_timer = ui.timer(max(0.05, session.get('cvc_speed', 0.4)), ai_loop)

    show_menu()

def run():
    if '--debug' in sys.argv:
        log.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    host = '0.0.0.0'
    port = 8080
    for i, a in enumerate(sys.argv):
        if a == '--host' and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]
        elif a == '--port' and i + 1 < len(sys.argv):
            try:
                port = int(sys.argv[i + 1])
            except ValueError:
                pass
    ui.run(
        title='Ultimate Tic Tac Toe — 終極井字棋',
        host=host,
        port=port,
        reload=False,
        storage_secret='ultimate-tic-tac-toe-sba',
    )

if __name__ == '__main__':
    run()
