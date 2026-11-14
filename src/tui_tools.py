# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Textual dashboards for the CLI tools (D44): --bench-tui and --train-tui.

Both apps follow the same shape: a config panel prefilled from the command
line, a Start button that hands the blocking work to a thread worker, and
live widgets fed by the structured callbacks (bench(on_progress=...) /
train(on_event=...)) — the CLI tools keep their rich/CSV output, these
apps are an alternative VIEW, not a second implementation. Auto-exit is a
test hook: the Pilot smoke needs the app to leave after the work lands.
"""

import sys

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (Button, DataTable, Input, ProgressBar, RichLog,
                             Select, Sparkline, Static)

from SBA import BENCH_AI_NAMES


class _ConfigPanel(Vertical):
    """Left-hand form column shared by both dashboards."""

    DEFAULT_CSS = """
    _ConfigPanel { width: 44; padding: 1; }
    _ConfigPanel Input { margin-bottom: 1; }
    _ConfigPanel Select { margin-bottom: 1; }
    _ConfigPanel Button { margin-top: 1; }
    """


class _RunPanel(Vertical):
    DEFAULT_CSS = """
    _RunPanel { padding: 1; }
    _RunPanel ProgressBar { margin-bottom: 1; }
    _RunPanel Horizontal.spark { height: 6; }
    _RunPanel Sparkline { width: 1fr; height: 100%; }
    _RunPanel #stats { margin-bottom: 1; color: $text-muted; }
    _RunPanel RichLog { height: 1fr; border: round $primary; }
    """


class BenchTUI(App):
    """Interactive --bench: pick two engines, watch every game land."""

    CSS = """
    Screen { layout: horizontal; }
    """
    BINDINGS = [Binding('q', 'quit', 'Quit')]
    TITLE = 'SBA bench'

    def __init__(self, games=20, iterations=1600, game_type='ultimate',
                 ai_a=None, ai_b=None, depth=4, auto_exit=False):
        super().__init__()
        self._cfg = {'games': games, 'iterations': iterations,
                     'game_type': game_type, 'ai_a': ai_a, 'ai_b': ai_b,
                     'depth': depth}
        self._auto_exit = auto_exit
        self._busy = False
        self._row_key = 0
        print('DBG BenchTUI.__init__ set _running False, id(self)=',
              id(self), file=sys.stderr, flush=True)
        self._col_keys = None

    def compose(self) -> ComposeResult:
        engines = [(n, n) for n in BENCH_AI_NAMES]
        with _ConfigPanel():
            yield Static('[bold]BENCH // 引擎對戰[/bold]')
            yield Select(engines, value=self._cfg['ai_a'] or 'MCTS',
                         prompt='Engine A', id='ai-a')
            yield Select(engines, value=self._cfg['ai_b'] or 'MCTS+GRAVE',
                         prompt='Engine B', id='ai-b')
            yield Input(value=str(self._cfg['games']), type='integer',
                        id='games')
            yield Input(value=str(self._cfg['iterations']), type='integer',
                        id='iters')
            yield Select([('ultimate', 'ultimate'), ('normal', 'normal')],
                         value=self._cfg['game_type'], prompt='Board',
                         id='board')
            yield Select([('2', '2'), ('4', '4'), ('6', '6'), ('8', '8')],
                         value=str(self._cfg['depth']), prompt='Minimax depth',
                         id='depth')
            yield Button('Start', id='start', variant='primary')
        with _RunPanel():
            yield ProgressBar(id='bar', show_eta=False)
            yield DataTable(id='scores')
            yield RichLog(id='log', markup=True, wrap=True)

    def on_mount(self):
        table = self.query_one('#scores', DataTable)
        self._col_keys = [table.add_column(c) for c in
                          ('A', 'B', 'game', 'A wins', 'draws', 'B wins',
                           'A score%')]
        table.add_row('', '', '', '', '', '', '', key='bench')

    @work(exclusive=True, thread=True)
    def _start_bench(self, cfg):
        from SBA import bench
        bench(games=cfg['games'], iterations=cfg['iterations'],
              game_type=cfg['game_type'], ai_a=cfg['ai_a'], ai_b=cfg['ai_b'],
              depth=cfg['depth'], seed=12345, quiet=True,
              on_progress=lambda ev: self.call_from_thread(
                  self._on_game, ev))
        self.call_from_thread(self._finish)

    def _on_game(self, ev):
        bar = self.query_one('#bar', ProgressBar)
        if bar.total != ev['total']:
            bar.total = ev['total']
        bar.progress = ev['game']
        table = self.query_one('#scores', DataTable)
        table.update_cell('bench', self._col_keys[2], str(ev['game']))
        table.update_cell('bench', self._col_keys[3], str(ev['aw']))
        table.update_cell('bench', self._col_keys[4], str(ev['dw']))
        table.update_cell('bench', self._col_keys[5], str(ev['bw']))
        table.update_cell('bench', self._col_keys[6], f"{ev['pct']:.1f}%")
        log = self.query_one('#log', RichLog)
        log.write(f"game {ev['game']}/{ev['total']}: "
                  f"{ev['a']} {ev['aw']} - {ev['dw']} - {ev['bw']} {ev['b']}")

    def _finish(self):
        self._busy = False
        self.query_one('#log', RichLog).write('[bold green]done[/bold green]')
        if self._auto_exit:
            self.exit()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id != 'start' or self._busy:
            return
        self._press_start_bench()

    def _press_start_bench(self):
        try:
            cfg = dict(self._cfg)
            cfg['games'] = int(self.query_one('#games', Input).value or 1)
            cfg['iterations'] = int(self.query_one('#iters', Input).value or 1)
            cfg['game_type'] = self.query_one('#board', Select).value
            cfg['depth'] = int(self.query_one('#depth', Select).value)
            cfg['ai_a'] = self.query_one('#ai-a', Select).value
            cfg['ai_b'] = self.query_one('#ai-b', Select).value
            self._worker = self._start_bench(cfg)
        except Exception:
            import sys
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise

class TrainTUI(App):
    """Interactive --train-az dashboard: progress, loss/win-rate sparklines
    and the milestone log, fed by train(on_event=...)."""

    CSS = """
    Screen { layout: horizontal; }
    """
    BINDINGS = [Binding('q', 'quit', 'Quit')]
    TITLE = 'SBA train'

    def __init__(self, games=300, sims=80, workers=1, eval_every=25,
                 eval_games=20, channels=128, blocks=5, stage=2,
                 resume=None, model_path=None, csv_path=None, auto_exit=False):
        super().__init__()
        self._cfg = {'games': games, 'sims': sims, 'workers': workers,
                     'eval_every': eval_every, 'eval_games': eval_games,
                     'channels': channels, 'blocks': blocks, 'stage': stage,
                     'resume': resume, 'model_path': model_path,
                     'csv_path': csv_path}
        self._auto_exit = auto_exit
        self._busy = False
        self._loss_hist = []
        self._wr_hist = []
        self._stats_text = ''

    def compose(self) -> ComposeResult:
        with _ConfigPanel():
            yield Static('[bold]TRAIN // 兩階段配方[/bold]')
            yield Input(value=str(self._cfg['games']), type='integer',
                        id='games')
            yield Input(value=str(self._cfg['sims']), type='integer', id='sims')
            yield Input(value=str(self._cfg['workers']), type='integer',
                        id='workers')
            yield Input(value=str(self._cfg['eval_every']), type='integer',
                        id='eval-every')
            yield Select([('stage 2 (neural)', '2'),
                          ('stage 1 (pure MCTS)', '1')],
                         value=str(self._cfg['stage']), prompt='Stage',
                         id='stage')
            yield Input(value=self._cfg['resume'] or '', placeholder='resume '
                        'ckpt path (optional)', id='resume')
            yield Input(value=self._cfg['model_path'] or '',
                        placeholder='model path (optional)', id='model')
            yield Input(value=self._cfg['csv_path'] or '',
                        placeholder='csv path (optional)', id='csv')
            yield Button('Start', id='start', variant='primary')
        with _RunPanel():
            yield ProgressBar(id='bar', show_eta=False)
            yield Horizontal(
                Sparkline(id='loss-spark'),
                Sparkline(id='wr-spark'),
                classes='spark')
            yield Static('', id='stats')
            yield RichLog(id='log', markup=True, wrap=True)

    @work(exclusive=True, thread=True)
    def _start_train(self, cfg):
        import alphazero
        alphazero.train('ultimate', games=cfg['games'], sims=cfg['sims'],
                        workers=cfg['workers'], eval_every=cfg['eval_every'],
                        channels=cfg['channels'], blocks=cfg['blocks'],
                        stage=cfg['stage'], resume=cfg['resume'],
                        save_path=cfg['model_path'], csv_path=cfg['csv_path'],
                        quiet=True, on_event=lambda ev: self.call_from_thread(
                            self._on_event, ev))
        self.call_from_thread(self._finish)

    def _on_event(self, ev):
        bar = self.query_one('#bar', ProgressBar)
        if ev['type'] == 'game':
            if bar.total != ev['games']:
                bar.total = ev['games']
            bar.progress = ev['game']
            self._loss_hist.append(ev['loss'])
            self._loss_hist = self._loss_hist[-200:]
            ev_eval = ev.get('eval')
            if ev_eval:
                w, d, l = ev_eval
                self._wr_hist.append(100.0 * (w + 0.5 * d) / max(1, w + d + l))
                self._wr_hist = self._wr_hist[-200:]
                self.query_one('#log', RichLog).write(
                    f"[bold]eval[/bold] game {ev['game']}: win={w} "
                    f"draw={d} loss={l} "
                    f"({100.0 * (w + 0.5 * d) / max(1, w + d + l):.0f}%)")
            self.query_one('#loss-spark', Sparkline).data = self._loss_hist
            self.query_one('#wr-spark', Sparkline).data = self._wr_hist
            self._stats_text = (f"game {ev['game']}/{ev['games']} · "
                                f"stage {ev['stage']} · lr {ev['lr']:.2e} · "
                                f"temp {ev['temp']:.2f} · "
                                f"positions {ev['positions']} · "
                                f"{ev['elapsed']:.0f}s")
            self.query_one('#stats', Static).update(self._stats_text)
        elif ev['type'] == 'ckpt':
            self.query_one('#log', RichLog).write(
                f"[yellow]checkpoint -> {ev['path']}[/yellow]")
        elif ev['type'] == 'saved':
            self.query_one('#log', RichLog).write(
                f"[bold green]saved -> {ev['path']}[/bold green]")

    def _finish(self):
        self._busy = False
        self.query_one('#log', RichLog).write('[bold green]done[/bold green]')
        if self._auto_exit:
            self.exit()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id != 'start' or self._busy:
            return
        self._press_start_train()

    def _press_start_train(self):
        cfg = dict(self._cfg)
        cfg['games'] = int(self.query_one('#games', Input).value or 1)
        cfg['sims'] = int(self.query_one('#sims', Input).value or 1)
        cfg['workers'] = int(self.query_one('#workers', Input).value or 1)
        cfg['eval_every'] = int(self.query_one('#eval-every', Input).value
                                or 99)
        cfg['stage'] = int(self.query_one('#stage', Select).value or 2)
        cfg['resume'] = self.query_one('#resume', Input).value or None
        cfg['model_path'] = self.query_one('#model', Input).value or None
        cfg['csv_path'] = self.query_one('#csv', Input).value or None
        self._running = True
        self._worker = self._start_train(cfg)


def run_bench(args):
    """--bench-tui handler: the same flags as --bench, ignored defaults kept."""
    import SBA
    app = BenchTUI(
        games=int(SBA._flag_value('--games', 20, args)),
        iterations=int(SBA._flag_value('--iters', 1600, args)),
        game_type='normal' if '--normal' in args else 'ultimate',
        ai_a=SBA._flag_arg('--ai-a', args),
        ai_b=SBA._flag_arg('--ai-b', args),
        depth=int(SBA._flag_value('--depth', 4, args)))
    app.run()


def run_train(args):
    """--train-tui handler: the same flags as --train-az."""
    import SBA
    import alphazero
    app = TrainTUI(
        games=int(SBA._flag_value('--games', 300, args)),
        sims=int(SBA._flag_value('--sims', 80, args)),
        workers=int(SBA._flag_value('--workers', 1, args)),
        eval_every=int(SBA._flag_value('--eval-every', 25, args)),
        eval_games=int(SBA._flag_value('--eval-games', 20, args)),
        channels=int(SBA._flag_value('--channels', 128, args)),
        blocks=int(SBA._flag_value('--blocks', 5, args)),
        stage=int(SBA._flag_value('--stage', 2, args)),
        resume=SBA._flag_arg('--resume', args),
        model_path=SBA._flag_arg('--model-path', args),
        csv_path=SBA._flag_arg('--csv', args))
    app.run()
