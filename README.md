# SBA - Basically Awful

**SBA** (short for *"SBA, Basically Awful"*) is a bilingual (English / 繁體中文) web app for **Normal** and **Ultimate** Tic Tac Toe, built with [NiceGUI](https://nicegui.io). It ships with 8 different AI opponents, an AI-assistant analysis panel, three game modes, and a Material Design 3 style UI.

> The name is a joke. The game is actually (mostly) fine.

---

## Features

- **Two game types**: classic 3x3 Tic Tac Toe and 9x9 Ultimate Tic Tac Toe
- **Three modes**: Player vs Player, Player vs Computer, Computer vs Computer
- **8 AI engines**:
  - `Random` - plays a random legal move
  - `Basic` - wins if possible, blocks, then prefers center/corners
  - `Minimax` - classic alpha-beta search (perfect on Normal, depth-limited on Ultimate)
  - `Minimax Pro` - negamax + transposition table + iterative deepening
  - `MCTS` - Monte Carlo Tree Search with UCT (strength adjustable)
  - `MCTS+RAVE` - MCTS with AMAF/RAVE move sharing
  - `Solver` - perfect play via a full Normal-mode tablebase
  - `AlphaZero` - neural-guided MCTS (trainable, see below)
- **AI Assistant panel**: analyzes the current position and shows the top 3-5 moves with win probability, a one-line reason (win / block / fork / center / corner / positional), and click-to-highlight on the board
- **Adjustable AI strength**: MCTS iterations (200-3000) and Minimax depth (2-6) sliders
- **CvC controls**: play speed (0.1-2.0s), auto-play toggle, and a manual "step" button
- **First-player choice** in Player vs Computer mode (human X first, or computer first)
- **Material Design 3 style** UI with light/dark theme toggle
- **Headless self-test suite** (57 checks) and a **Docker** image

---

## Requirements

- Python **3.12+** (developed on 3.13)
- Core dependency: `nicegui>=3.16`

Install core dependencies:

```bash
pip install -r requirements.txt
```

**Optional - AlphaZero**: the neural engine needs `torch` and `numpy` (a CPU-only PyTorch build is fine; the models are tiny). Only install this if you want to use or train AlphaZero:

```bash
pip install torch numpy
```

The `models/` folder (gitignored) stores trained AlphaZero networks.

---

## Usage

### 1. Start the web app

```bash
# Windows (uses the local .venv)
run.bat

# or directly
python SBA.py
```

Open http://localhost:8080 (or http://127.0.0.1:8080) in your browser.

CLI flags:

| Flag | Description |
| --- | --- |
| `--host HOST` | Bind address (default `0.0.0.0`) |
| `--port PORT` | Port (default `8080`) |
| `--debug` | Verbose backend logging |
| `--self-test` | Run the headless test suite and exit |
| `--train-az` | Alias to run `alphazero.py train` |

### 2. Self-test

```bash
python SBA.py --self-test
# or
run.bat --self-test
```

Runs 57 headless checks covering game rules, AI sanity, termination, and AlphaZero smoke tests.

### 3. Train / evaluate AlphaZero

```bash
# train a neural net for the chosen game type (saves to models/)
python alphazero.py train --game normal   --games 400 --sims 80
python alphazero.py train --game ultimate --games 300 --sims 80

# evaluate a trained model against MCTS
python alphazero.py eval --game normal --games 30 --sims 200
```

### 4. Docker

```bash
docker build -t sba .
docker run -p 8080:8080 sba
```

---

## Source layout

```
SBA.py            Entry point: CLI flags, session state, self-tests
game.py           Game rules: NormalGame / UltimateGame, move application, board helpers
ai.py             All AI engines + get_ai_move + AI-assistant analysis
webui.py          NiceGUI web UI (menu, board, assistant panel, CvC controls)
alphazero.py      AlphaZero neural MCTS (training + evaluation)
static/styles.css Material Design 3 stylesheet
run.bat           Windows launcher
Dockerfile        Container image (CPU-only torch)
requirements.txt  Core Python dependencies
```

Dependency direction is one-way: `game.py` -> `ai.py` -> `SBA.py` -> `webui.py`.

---

## Ultimate Tic Tac Toe rules (quick reference)

- The big board is a 3x3 grid of 3x3 micro boards.
- Playing in micro-cell `(r, c)` forces the opponent's next move into macro-cell `(r, c)`.
- If that macro cell is already won or full, the player may move into any open macro cell.
- Winning a micro board claims that macro cell; a full micro board with no winner counts as a neutral draw.
- A player wins the game by claiming 3 macro cells in a line; if the whole board fills with no macro winner, it is a draw.

---

## Credits

- [NiceGUI](https://nicegui.io) - reactive web UI framework (bundles Quasar / Tailwind)
- [PyTorch](https://pytorch.org) - neural networks for the AlphaZero engine
- [NumPy](https://numpy.org) - tensor utilities in the AlphaZero trainer
- Code and project setup developed with assistance from OpenAI Codex.

---

## License

This project is provided as-is for personal and educational use. See the repository owner for any reuse questions.
