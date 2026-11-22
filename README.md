# SBA - Basically Awful (Qt + TUI)

[![CI](https://github.com/TofuShawn/SBA/actions/workflows/ci.yml/badge.svg)](https://github.com/TofuShawn/SBA/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/tag/TofuShawn/SBA?label=release)](https://github.com/TofuShawn/SBA/releases)

English | [繁體中文](README.zh-Hant.md)

**SBA** (*"SBA, Basically Awful"*) is a Tic Tac Toe and Ultimate Tic Tac Toe
application with a PySide6 **desktop app** (default) and a Textual **terminal
app**, styled after the Arknights: Endfield industrial console theme. Ships
with **8 selectable AI opponents** (engine hot paths sunk into C — see
Native C engine), an AI analysis panel, and three play modes.

> The name is a joke. The game is actually (mostly) fine.

---

## Features

- **Two game types**: classic 3x3 Tic Tac Toe and 9x9 Ultimate Tic Tac Toe
- **Three modes**: Player vs Player, Player vs Computer, Computer vs Computer
- **8 selectable AI engines**:
  - `Random` — random legal move
  - `Basic` — win if possible, block, prefer center/corners
  - `Minimax` — classic alpha-beta search (perfect on Normal, depth-limited on Ultimate)
  - `Minimax Pro` — negamax + transposition table + iterative deepening
  - `MCTS` — Monte Carlo Tree Search with UCT (strength adjustable)
  - `MCTS+GRAVE` — Generalized RAVE (Cazenave 2015): shares stats through reference nodes, lower memory
  - ~~`MCTS+RAVE`~~ — hidden in the menu (superseded by MCTS+GRAVE; engine kept for tests and `--bench`)
  - ~~`Flat MCTS`~~ — hidden in the menu (research baseline: root-level playouts, no tree)
  - `Solver` — perfect Normal play from the persisted tablebase (`data/tt_normal.bin`, rebuilt by `--build-tables`); MCTS fallback on Ultimate
  - `AlphaZero` — neural-guided MCTS (Ultimate only)
- **AI Assistant panel**: analyses the position, shows the best 3-5 moves,
  win rates and a one-line reason (win/block/fork/center/corner/positional);
  clickable highlights on the board; optional **policy heat map** (AZ policy
  tinted over empty cells, needs a trained model)
- **Adjustable AI strength**: MCTS iterations (200-3000) and Minimax depth (2-6)
- **Engine optimizations** (toggleable in `config.toml`): opening book, micro
  endgame table, heuristic rollout, tree reuse, dynamic UCT, progressive
  widening, early termination, fork evaluation, killer/LMR/aspiration, D4
  symmetry, object pool, bitboard, multithreaded MCTS
- **CvC controls**: speed (0.1-2.0s), auto-play, manual "next move"
- **First-player choice**: human (X) or computer first in PvC
- **Arknights: Endfield field-console theme**: signal yellow × charcoal,
  deck labels, targeting brackets, stamp placement and win-line animation
  ("reduce motion" available)
- **Settings page**: every engine switch graphical, saved back to `config.toml`
- **History card**: click any move to replay that position
- **Headless pytest suite** (rules, engines, UI smoke, ABI contracts; size via
  `pytest tests --collect-only -q`)

---

## Requirements

- Python **3.12+** (developed on 3.13)
- Core deps: `pyside6>=6.12,<7` (desktop), `textual>=8.2,<9` (terminal),
  `rich>=15,<16`, `numpy>=2.0,<3`, `safetensors>=0.4,<2` (AlphaZero weights /
  self-play record format)

```bash
pip install -r requirements.txt
```

The native C engine core builds via `python src/tools/build.py` into
`src/engines/sba_engine.dll` (local gcc on Windows, or WSL mingw
cross-compile; CI builds it automatically). Without the DLL the engines fall
back to pure Python — same features, slower.

**Optional — AlphaZero**: the neural engine needs `torch` (CPU wheel is
fine; the model is small; `numpy` is already in `requirements.txt`). Install
only to use or train AlphaZero:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Trained AlphaZero weights are distributed via GitHub Releases. For AMD GPU
training on Windows (TheRock wheels or WSL2), see
[`docs/rocm-wsl2.md`](docs/rocm-wsl2.md). Training methodology and
performance analysis: [`docs/AlphaZero_UTTT_Technical_Report.html`](docs/AlphaZero_UTTT_Technical_Report.html).

New to the codebase? Follow the Architecture map's reading route (1 → 5)
below; a from-zero tutorial lives at
[`docs/tutorial-zh.md`](docs/tutorial-zh.md) (Simplified) /
[`docs/tutorial-zh-Hant.md`](docs/tutorial-zh-Hant.md) (Traditional).

Working on the repo? `python dev.py` is the task runner (build / test /
test-rust / lint / format / clippy / clippy-core / cgate / bench-selfplay /
self-test); the branch model and prerequisites live in
[`docs/development.md`](docs/development.md).

---

## Usage

### 1. Desktop app (default)

```bash
# Unified entry: every command starts at the root shim
python sba.py

# equivalent module paths (kept for compatibility)
python src/session/SBA.py
python src/session/SBA.py --qt
```

The desktop app uses the Endfield industrial console theme
(`static/endfield.qss` + custom board painting); the menu page opens the
Settings deck for engine switches.

### 2. Terminal UI (TUI)

```bash
python sba.py --tui
```

Textual terminal app, same engines and settings; good for SSH / headless
boxes.

CLI flags (identical for `python sba.py` and `python src/session/SBA.py`):

| Flag | Description |
| --- | --- |
| `--qt` | Start the PySide6 desktop app (default) |
| `--tui` | Start the Textual terminal app instead |
| `--debug` | Verbose backend logging |
| `--self-test` | Run headless tests and exit |
| `--bench [--ai-a A] [--ai-b B] [--games N] [--iters N] [--depth N] [--normal]` | Win-rate benchmark (default: MCTS family round-robin) |
| `--build-tables` | Rebuild `data/tt_normal.bin` — the persisted perfect-play table for Normal mode (all 5,478 reachable positions) |
| `--train-az` | Alias for `src/az/alphazero.py train` |
| `--eval-az` | Alias for `src/az/alphazero.py eval` |
| `--bench-tui` | Interactive benchmark dashboard (pick engines, live per-game scores) |
| `--train-tui` | Interactive training dashboard (progress, loss/win-rate sparklines, milestone log) |

### 3. Self-test

```bash
python sba.py --self-test
# Windows: start.bat --self-test (local launcher, not tracked in git)
```

Runs the pytest suite covering rules, AI sanity, termination, and AlphaZero
smoke tests (needs `pip install -r requirements-dev.txt`: pytest is a dev
tool, not a runtime dependency).

### 3b. MCTS benchmark

```bash
# default: MCTS vs MCTS+RAVE vs MCTS+GRAVE round-robin on Ultimate
python sba.py --bench
# pick your own matchup
python sba.py --bench --ai-a MCTS --ai-b MCTS+GRAVE --games 40 --iters 400
python sba.py --bench --ai-a Minimax --ai-b MCTS+GRAVE --depth 4
# smaller / faster, or Normal board
python sba.py --bench --games 10 --iters 200 --normal
```

Available engines for `--ai-a`/`--ai-b`: Random, Basic, Minimax, Minimax Pro,
MCTS, MCTS+RAVE, MCTS+GRAVE, Flat MCTS, Solver, AlphaZero. The first player
alternates per game; per-match scores (win = 1, draw = 0.5) are printed.

### 4. Train / evaluate AlphaZero

```bash
# train a neural net for Ultimate (Normal is solved and not supported)
python sba.py --train-az --games 400 --sims 80
# or via the module directly
python src/az/alphazero.py train --games 400 --sims 80

# evaluate a trained model vs MCTS
python sba.py --eval-az --games 30 --sims 200
```

Training extras: `--ckpt-every 25` (checkpoint every N games),
`--dirichlet-eps 0.25` (root exploration noise, 0 disables), `--lr 1e-3`
(linear annealing to 1/5), `--workers 8` (multiprocess self-play, one GPU
model copy per worker), `--rollout-weight 0` (pure-network leaf values — the
standard AlphaZero recipe (D58) and the exe's recipe; >0 re-enables the
legacy win/block heuristic blend), `--selfplay-engine auto|rust|python`
(self-play source: Rust exe / Python mp.Pool; auto probes the exe and
downgrades when absent), `--selfplay-procs N` (K exe processes splitting the
games), `--csv` (training-curve CSV — one row per game: loss / LR / temp /
eval result / checkpoint; defaults to `<model>.csv`; new runs overwrite,
resume appends), `--channels 128`, `--blocks 5`. Self-play temp anneals
linearly from 1.0 to 0.05 — late games play near-strongest moves.

## Configuration

All engine switches and parameters live in **`config.toml`** (repo root):
MCTS rollout heuristic, tree reuse, dynamic UCT, opening book, micro
tablebase, LMR/killer/aspiration, symmetry, object pool, bitboard,
multithreaded MCTS, UCT/RAVE constants, and session defaults (`mcts_budget`,
`minimax_depth`). Missing keys fall back to code defaults. `$SBA_CONFIG`
points every entry at another file to override; because that file is both
read and written, the override must live **inside this repo or the system
temp directory** (the same path guard covers config saves).

---

## Architecture map

```
      UI LAYER (pick one — two front ends, one backend)
  +----------------------+  +----------------------+
  | src/ui/qtui.py  [5a] |  | src/ui/tui.py  [5b] |
  | PySide6 desktop      |  | rich terminal        |
  +----------+-----------+  +----------+-----------+
             |                         |
             +------------+------------+
                          |
      GLUE LAYER          v
  +------------------------------------------------------+
  | src/session/SBA.py  [4]                              |
  | CLI entry · session state · bilingual strings        |
  | --tui / --bench-tui / --train-tui / --train-az        |
  +---------------------------+--------------------------+
                              |
      ENGINE LAYER            v
  +------------------------------------------------------+
  | src/engines/ai.py  [2]  facade (D47): get_ai_move    |
  |   dispatch, the AlphaZero bridge, cache reset        |
  |   engine modules:                                    |
  |   src/config.py            ROOT, config.toml, guard  |
  |   src/engines/cbridge.py   ctypes boundary (the DLL) |
  |   src/analysis/books.py    Basic heuristic + books   |
  |   src/engines/mcts.py      UCT / RAVE / GRAVE trees  |
  |   src/engines/minimax.py   alpha-beta + Minimax Pro  |
  |   src/analysis/tablebase.py  Normal solve + Solver   |
  |   src/analysis/analysis.py  assistant + win rates    |
  +---------+--------------------------------------------+
            |
            |  lazy import (loads torch only when selected)
            v
  +------------------------------------------------------+
  | src/az/alphazero.py  [3]                             |
  | neural-guided MCTS: AZNet policy+value heads,        |
  | self-play workers, replay buffer, LR/temp annealing  |
  | Ultimate only (D4)                                   |
  +---------------------------+--------------------------+
                              |
      RULES LAYER             v
  +------------------------------------------------------+
  | src/core/game.py  [1]                                |
  | NormalGame / UltimateGame / BitUltimateGame          |
  | the only ground truth: legal_moves() make_move()     |
  | result() clone()                                     |
  +------------------------------------------------------+

  SIDE CAR (no layer — loads in from the side)
    config.toml -------------- > src/config.py  every engine switch + session defaults (D11)
    src/engines/cengine.py + dll > src/engines/cbridge.py  C backends (D25-D30, D37):
                                              full MCTS tree, rollouts and the minimax
                                              family live in sba_engine.dll (~37x; built via
                                              src/tools/build.py, auto-fallback to Python)
    src/sba-core (Rust) --- > src/az/rustselfplay.py  the self-play exe (D55-D59):
                                              wave search + batched burn inference,
                                              safetensors records, K-process rounds
    src/sba-game (Rust) --- > src/az/rustbridge.py  state layer (D48):
                                              clone/apply/legal/result on packed
                                              bitboards in az/sba_game.dll|so
    static/ (qss·fonts·bg) --- > src/ui/qtui.py  stylesheets, bundled OFL fonts, backdrops
    tests/ -------------------                headless pytest: rules, AI sanity,
                                              C engine, AZ recipe, Rust parity,
                                              golden vectors
```

Reading-route highlights:

- **[1] `src/core/game.py`** — start with `UltimateGame`'s "the played cell
  routes the opponent into that macro" rule; the rules core of the project.
- **[2] `src/engines/ai.py` (facade)** — the engines live across
  `engines/mcts.py` / `engines/minimax.py` / `analysis/tablebase.py` /
  `analysis/analysis.py` / `analysis/books.py`; skim in the order
  `get_basic_move` → `minimax_*` → `mcts_search` (the UCT heart) →
  `analyze_position`. RAVE/GRAVE/Flat are A/B comparison groups, not required reading.
- **[3] `src/az/alphazero.py`** — if you read one function, read
  `self_play_game()`; the honest "why it loses to plain MCTS at equal
  budget" conclusion lives in `docs/AlphaZero_UTTT_Technical_Report.html`.
- **[4] `src/session/SBA.py`** — the overview map; read last.
- **[5] UI** — both front ends share [1]–[4]; `src/ui/qtui.py`'s
  `BoardWidget` self-painting is the most complex single point — grab
  `MenuPage` / `GamePage` first.
- Decision log with the full "why" per entry:
  [`docs/decisions.md`](docs/decisions.md)

---

## Ultimate Tic Tac Toe rules

- The big board is a 3x3 grid of 3x3 micro boards.
- Playing in micro-cell `(r, c)` forces the opponent's next move into
  macro-cell `(r, c)`.
- If that macro cell is won or full, the player may play any open macro cell.
- Winning a micro board claims that macro cell; a full micro board with no
  winner is a neutral draw.
- Win the game by claiming 3 macro cells in a line; otherwise a full board
  is a draw.

---

## Native C engine

All engine hot paths run in C since D25–D51 (`python src/tools/build.py`
builds `src/engines/sba_engine.dll`; WSL mingw cross-compile or any local
gcc; CI builds it automatically) and the game state runs in Rust since
D48/D50 (`src/az/sba_game.dll`, cargo; the Python game stays the reference
and the fallback). Measured on this machine:

| Metric | Python | C (1 thread) | C (threads, D30) |
|---|---|---|---|
| MCTS rollout (per playout) | 0.51 ms | 0.029 ms (17.7x) | — |
| MCTS 800 sims (full search, symmetry on) | ~530 ms | 12.9 ms (41x) | 3.4 ms @ 8T (3.6x wall) |
| MCTS 3200 sims | — | 51.2 ms | 9.2 ms @ 8T (5.8x wall) |
| Heuristic rollout (AZ leaf value) | 0.47 ms | 0.028 ms (16.9x) | — |

| Rust state layer (D52) | vs Python game |
|---|---|
| playout loop (state ops) | 2.14x |
| fullgame (legal + encode + clone per ply) | 1.64x |
| AZ self-play stage-1 (C-tree dominated) | ~parity (1.04x) |

| Rust self-play core (D57) | vs Python mp.Pool path |
|---|---|
| end-to-end self-play @800 sims, 8 threads | 6.1 vs 23.7 pos/s (0.26x — Phase A ≥4x bar NOT met) |

Phase A parity records (D57): the Python wave search is ported 1:1 into
`search.rs`; layer-3 parity 200/200 top-1 (100%) / KL 0.0000 — byte-equal
visit vectors on all 200 positions; batch fill 0.894 (~229 rows per
dispatch). The remaining throughput gap is backend-bound: the burn/cubecl
forward costs ~37 ms per 229-row batch on RDNA3 Vulkan vs torch/ROCm-HIP's
2.84 ms per 256-row batch (≈13x kernel gap), so even a multi-queue pipeline
caps at ≈1.4x on this hardware. Re-measure: `python dev.py bench-selfplay`
and `python src/tools/parity_search.py`; details in docs/decisions.md D57.
The Python wave-search reference is frozen as golden vectors
(`tests/golden/`) — the parity anchor survives any future retirement of the
Python implementation.

AZ self-play runs ~7–9 s/game @80 sims (128ch/5blocks, 1 worker) — the loop
is NN-forward-bound, so state-layer gains are bounded by Amdahl.

Extras: tree reuse carries the warm subtree between moves on BOTH the C and
Python paths (the `tree_reuse` switch gates both ends, D53); settings page
toggles every switch (D28); pattern rollout variant kept off by default
(D26; re-validated 2026-11-15 on the current C trees — 85% vs 92.5%
vs-Basic, 3.9x rollout cost: still no gain).

---

## Benchmarks

n = 20; the 95% CI is roughly ±22% — for ranking only. Re-measured
2026-11-15 on the C engines (D51)

### All engines vs Basic (Ultimate, 20 games, MCTS@1600)

| Engine | W / D / L | Score |
|---|---|---|
| Solver\*@1600 | 20 / 0 / 0 | **100%** |
| MCTS+GRAVE@1600 | 20 / 0 / 0 | **100%** |
| MCTS@1600 | 20 / 0 / 0 | **100%** |
| Minimax (depth 4) | 19 / 1 / 0 | **97.5%** |
| MCTS+RAVE@1600 | 19 / 1 / 0 | **97.5%** |
| Flat MCTS@1600 | 18 / 2 / 0 | **95.0%** |
| Minimax Pro (depth 4) | 17 / 3 / 0 | 92.5% |
| Random | 1 / 2 / 17 | 10.0% |

\* On Ultimate, Solver is a plain MCTS fallback — it only truly solves Normal.

### All engines vs Basic (Normal, 20 games, MCTS@1600)

| Engine | W / D / L | Score |
|---|---|---|
| Flat MCTS@1600 | 8 / 10 / 2 | 65.0% |
| Minimax (depth 4) | 2 / 18 / 0 | 55.0% |
| Solver (exact) | 2 / 18 / 0 | 55.0% |
| Minimax Pro (depth 4) | 0 / 20 / 0 | 50.0% |
| MCTS@1600 | 0 / 20 / 0 | 50.0% |
| MCTS+GRAVE@1600 | 0 / 20 / 0 | 50.0% |
| MCTS+RAVE@1600 | 0 / 20 / 0 | 50.0% |
| Random | 0 / 1 / 19 | 2.5% |

### MCTS scales with budget (vs Minimax depth 4, Ultimate)

| Matchup (30 games) | Score |
|---|---|
| Minimax vs MCTS@800 | **48.3% (Minimax)** |
| Minimax vs MCTS@2000 | **23.3% (MCTS 76.7%)** |

At 800 sims the matchup is a coin flip; by 2000 MCTS wins ~77% — with the C
trees (D51) the crossover sits near 600–800 sims (D20).

### MCTS family round-robin (Ultimate, 20 games/pair, all engines@1600)

All three share one playout policy: the plain win/block heuristic kernel
(`c_rollout_pattern = false`).

| A vs B | W / D / L | Score |
|---|---|---|
| MCTS vs MCTS+RAVE | 13 / 3 / 4 | **72.5%** |
| MCTS vs MCTS+GRAVE | 17 / 1 / 2 | **87.5%** |
| MCTS+RAVE vs MCTS+GRAVE | 3 / 8 / 9 | 35.0% |

---

## Credits

- [Qt for Python (PySide6)](https://github.com/qtproject/pyside-pyside-setup) — the desktop UI framework
- [PyTorch](https://github.com/pytorch/pytorch) — neural networks for the AlphaZero engine
- [NumPy](https://github.com/numpy/numpy) — tensor utilities in the AlphaZero trainer

### Algorithm sources

The engines reimplement these published methods:

- [AlphaZero](https://www.science.org/doi/10.1126/science.aar6404) — Silver et al., *Science* 2018: neural-guided MCTS with policy and value heads, trained by self-play
- [MCTS](https://link.springer.com/chapter/10.1007/978-3-540-75538-8_7) — Coulom 2006: Monte-Carlo tree search with efficient selectivity and backup operators
- [UCT](https://link.springer.com/chapter/10.1007/11871842_29) — Kocsis & Szepesvári 2006: the UCB1-based tree policy
- [RAVE](https://doi.org/10.1145/1273496.1273531) — Gelly & Silver 2007, ICML: all-moves-as-first statistics inside UCT
- [GRAVE](https://www.ijcai.org/proceedings/2015/0112.pdf) — Cazenave 2015, IJCAI 754–760: AMAF statistics shared through reference nodes
- [Alpha-beta pruning](https://doi.org/10.1016/0004-3702(75)90019-3) — Knuth & Moore 1975: the minimax search backbone

---

## License

This project is released under the **GNU General Public License v3 (GPLv3)**.
See the [LICENSE](LICENSE) file for details.

### Third-party licenses

The bundled UI fonts are distributed under the **SIL Open Font License 1.1**;
the full license texts ship in `static/fonts/` (`OFL-Inter.txt`,
`OFL-JetBrainsMono.txt`):

- [Inter](https://github.com/rsms/inter) — © 2020 The Inter Project Authors (`static/fonts/Inter-*.ttf`)
- [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) — © 2020 The JetBrains Mono Project Authors (`static/fonts/JetBrainsMono-*.ttf`)
