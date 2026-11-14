# SBA - Basically Awful (Qt + TUI) / 基本上很糟（Qt + 終端機）

**SBA**（即 *"SBA, Basically Awful"* 的簡稱）是一個**井字棋**與**終極井字棋**應用，提供 PySide6 **Qt 桌面版**（預設）與 Textual **終端機版**兩種介面，Arknights: Endfield 風格的工業 console 主題。內建 **7 種 AI 對手**（引擎熱路徑全部下沉 C，見下方 Native C engine）、AI 分析面板、三種對戰模式。

> The name is a joke. The game is actually (mostly) fine.
> 名字是開玩笑的，遊戲其實（大致上）還不錯。

---

## Features / 功能

- **Two game types / 兩種棋盤**：classic 3x3 Tic Tac Toe（經典井字棋）和 9x9 Ultimate Tic Tac Toe（終極井字棋）
- **Three modes / 三種模式**：Player vs Player（玩家對戰）、Player vs Computer（人機對戰）、Computer vs Computer（電腦對戰）
- **9 selectable AI engines / 九種可選 AI 引擎**：
  - `Random` - random legal move / 隨機走棋
  - `Basic` - win if possible, block, prefer center/corners / 能贏就贏、會擋棋、偏好中心與角落
  - `Minimax` - classic alpha-beta search / 經典極小化極大搜尋（普通模式完美、終極模式限深度）
  - `Minimax Pro` - negamax + transposition table + iterative deepening / 進階極小化極大（置換表 + 疊代加深）
  - `MCTS` - Monte Carlo Tree Search with UCT / 蒙地卡羅樹搜尋（強度可調）
  - `MCTS+GRAVE` - Generalized RAVE (Cazenave 2015): shares RAVE stats through reference nodes, lower memory / 廣義 RAVE：透過 reference node 共享 RAVE 統計，省記憶體
  - ~~`MCTS+RAVE`~~ - hidden in the menu (superseded by MCTS+GRAVE; engine kept for tests and `--bench`) / 已停用（選單不可選，由 MCTS+GRAVE 取代；引擎保留供測試與 `--bench`）
  - ~~`Flat MCTS`~~ - hidden in the menu (research/learning baseline: root-level playouts, no tree) / 已停用（選單不可選，研究/學習用基準：只做根層 rollout，無樹搜尋）
  - `Solver` - perfect Normal play from the persisted tablebase (`data/tt_normal.bin`, rebuilt by `--build-tables`); MCTS fallback on Ultimate / 完美解：普通模式讀持久化解表，終極模式為 MCTS fallback
  - `AlphaZero` - neural-guided MCTS (Ultimate only) / 神經網路引導的 MCTS（僅終極模式）
- **AI Assistant panel / AI 助手面板**：分析目前局面，顯示最佳 3-5 步、勝率、一句話原因（win/block/fork/center/corner/positional），點擊可在棋盤上標示；可開啟**策略熱圖**（AZ policy 以色塊疊在空格上，需已訓練模型）
- **Adjustable AI strength / 可調 AI 強度**：MCTS 迭代數（200-3000）與 Minimax 深度（2-6）
- **Engine optimizations / 引擎優化**（`config.toml` 可開關）：開局書、micro 殘局表、啟發式 rollout、樹重用、動態 UCT、漸進式加粗、提早終止、fork 評估、killer/LMR/aspiration、D4 對稱性、物件池、bitboard、多執行緒 MCTS
- **CvC controls / 電腦對戰控制**：速度（0.1-2.0s）、自動播放、手動「下一步」
- **First-player choice / 先手選擇**：人機模式可選玩家先手（X）或電腦先手
- **Arknights: Endfield field-console theme / 終末地工業 console 主題**：訊號黃 × 炭灰、deck 標籤、瞄準括號、落子蓋章與勝利線動畫（可「減少動態」）
- **Settings page / 設定頁**：引擎開關全數圖形化，保存寫回 `config.toml`
- **History card / 歷史卡**：點擊任一步直接回放該局面
- **Headless pytest suite / 無頭 pytest 測試**（75 tests）

---

## Requirements / 環境需求

- Python **3.12+**（於 3.13 開發）
- 核心依賴：`pyside6>=6.6`（桌面版）、`textual>=3.0`（終端機版）

Install core dependencies / 安裝核心依賴：

```bash
pip install -r requirements.txt
```

PySide6（Qt for Python，桌面版用）已併入 `requirements.txt`，不需要另外安裝檔。
原生 C 引擎核心：`python native/build.py` 編譯 `native/sba_engine.dll`
（Windows 本機 gcc 或 WSL mingw 交叉編譯；CI 自動構建）。DLL 缺失時引擎
自動回退純 Python 實作，功能不受影響、僅速度下降。

**Optional - AlphaZero / 選用 - AlphaZero**：神經網路引擎需要 `torch` 與 `numpy`（CPU 版 PyTorch 即可，模型很小）。只有要使用或訓練 AlphaZero 才需要安裝：

```bash
pip install torch numpy
```

Trained AlphaZero weights are distributed via GitHub Releases /
已訓練的 AlphaZero 網路會在 GitHub Releases 發佈。

AMD 顯卡若要 GPU 訓練（Windows 原生 TheRock wheel 或 WSL2），可參考
[`docs/rocm-wsl2.md`](docs/rocm-wsl2.md)。

AlphaZero 訓練方法與效能分析見
[`docs/AlphaZero_UTTT_Technical_Report.html`](docs/AlphaZero_UTTT_Technical_Report.html)。

新手想讀懂專案，依下方「Architecture map」的閱讀順序（1 → 5）進行；
從零教學見
[`docs/tutorial-zh.md`](docs/tutorial-zh.md)（簡體中文）／
[`docs/tutorial-zh-Hant.md`](docs/tutorial-zh-Hant.md)（繁體中文）。

---

## Usage / 使用方式

### 1. Desktop app (default) / 桌面版（預設）

```bash
# PySide6 desktop UI / PySide6 桌面版
python src/SBA.py
# or explicitly / 或明確指定
python src/SBA.py --qt
# same as: / 等同於
python src/qtui.py
```

桌面版使用 Endfield 工業 console 主題（`static/endfield.qss` + 原生繪製棋盤），
選單頁可進入 Settings 調整引擎開關。

### 2. Terminal UI (TUI) / 終端機版

```bash
python src/tui.py
```

Textual 終端機介面，同一套引擎與設定；適合 SSH／無 GUI 環境。

CLI flags / 指令參數：

| Flag / 參數 | Description / 說明 |
| --- | --- |
| `--qt` | Start the PySide6 desktop app (default) / 啟動 PySide6 桌面版（預設） |
| `--debug` | Verbose backend logging / 後端詳細日誌 |
| `--self-test` | Run headless tests and exit / 執行無頭測試後結束 |
| `--bench [--ai-a A] [--ai-b B] [--games N] [--iters N] [--depth N] [--normal]` | Win-rate benchmark (default: MCTS family round-robin) / 引擎勝率對戰測試（預設 MCTS 家族循環賽） |
| `--build-tables` | Rebuild `data/tt_normal.bin` — the persisted perfect-play table for Normal mode (all 5,478 reachable positions) / 重建 Normal 模式持久化完美解表（全部 5,478 個可達局面） |
| `--train-az` | Alias for `src/alphazero.py train` / 等同執行 `src/alphazero.py train` |

### 3. Self-test / 自測

```bash
python src/SBA.py --self-test
# Windows: start.bat --self-test (local launcher, not tracked in git)
```

Runs the pytest suite covering rules, AI sanity, termination, and AlphaZero
smoke tests / 執行 pytest 測試套件，涵蓋規則、AI 正確性、對局終止與 AlphaZero 冒煙測試（需 `pip install pytest`，已列入 `requirements.txt`）。

### 3b. MCTS benchmark / MCTS 對戰測試

```bash
# default: MCTS vs MCTS+RAVE vs MCTS+GRAVE round-robin on Ultimate / 預設終極模式循環賽
python src/SBA.py --bench
# pick your own matchup / 自訂對戰組合
python src/SBA.py --bench --ai-a MCTS --ai-b MCTS+GRAVE --games 40 --iters 400
python src/SBA.py --bench --ai-a Minimax --ai-b MCTS+GRAVE --depth 4
# smaller / faster, or Normal board / 較小規模或普通棋盤
python src/SBA.py --bench --games 10 --iters 200 --normal
```

Available engines for `--ai-a`/`--ai-b`: Random, Basic, Minimax, Minimax Pro,
MCTS, MCTS+RAVE, MCTS+GRAVE, Flat MCTS, Solver, AlphaZero. The first player alternates
per game and per-match scores (win = 1, draw = 0.5) are printed /
`--ai-a`/`--ai-b` 可用引擎如上；每局輪換先手，輸出各組合得分率（勝 = 1、和 = 0.5）。

### 4. Train / evaluate AlphaZero / 訓練與評估 AlphaZero

```bash
# train a neural net for Ultimate (Normal is solved and not supported) / 訓練終極模式神經網路（普通模式已破解，不支援）
python src/alphazero.py train --games 400 --sims 80

# evaluate a trained model vs MCTS / 評估模型對戰 MCTS
python src/alphazero.py eval --games 30 --sims 200
```

Training extras / 訓練選項：`--ckpt-every 25`（每 N 局存一次 checkpoint）、
`--dirichlet-eps 0.25`（根節點探索噪聲，0 停用）、`--lr 1e-3`（線性退火到 1/5）、
`--workers 8`（多進程自對弈，每個 worker 一份 GPU 模型副本）、
`--rollout-weight 0.5`（葉節點混合 win/block 啟發式隨機對弈，0 = 純網路）、
`--csv`（訓練曲線 CSV——每局一行：loss / LR / temp / eval 勝敗和 / checkpoint，
預設 `<model>.csv`；新 run 覆寫、resume 續寫）、
`--channels 128`、`--blocks 5`。自對弈前段 temp 從 1.0 線性退到 0.05，
晚期遊戲近乎最強應手。

## Configuration / 設定

All engine switches and parameters live in **`config.toml`**（repo root）: MCTS
rollout heuristic, tree reuse, dynamic UCT, opening book, micro tablebase,
LMR/killer/aspiration, symmetry, object pool, bitboard, multithreaded MCTS,
UCT/RAVE constants, and session defaults（`mcts_budget`, `minimax_depth`）.
Missing keys fall back to code defaults; point `$SBA_CONFIG` at another file
to override / 所有引擎開關與參數集中在 `config.toml`（repo 根目錄）：啟發式 rollout、
樹重用、動態 UCT、開局書、micro 殘局表、LMR/killer/aspiration、對稱性、物件池、
bitboard、多執行緒 MCTS、UCT/RAVE 常數，以及 session 預設值（`mcts_budget`、
`minimax_depth`）。缺省鍵自動回退程式碼預設；可用環境變數 `SBA_CONFIG` 指向其他檔案。

---

## Architecture map / 架構地圖

```
      UI LAYER (pick one — two front ends, one backend)
  +----------------------+  +----------------------+
  | src/qtui.py     [5a] |  | src/tui.py      [5b] |
  | PySide6 desktop      |  | rich terminal        |
  | ~2500 lines          |  | ~1370 lines          |
  +----------+-----------+  +----------+-----------+
             |                         |
             +------------+------------+
                          |
      GLUE LAYER          v
  +------------------------------------------------------+
  | src/SBA.py  [4]                                      |
  | CLI entry · session state · t() bilingual strings    |
  | --self-test / --bench / --train-az                   |
  | ~250 lines                                           |
  +---------------------------+--------------------------+
                              |
      ENGINE LAYER            v
  +------------------------------------------------------+
  | src/ai.py  [2]                                       |
  | all engines + assistant analysis (get_ai_move,       |
  | analyze_position); one search -> best moves          |
  | AND (X, draw, O) win rates (D15)                     |
  | ~1900 lines                                          |
  +---------+--------------------------------------------+
            |
            |  lazy import (loads torch only when selected)
            v
  +------------------------------------------------------+
  | src/alphazero.py  [3]                                |
  | neural-guided MCTS: AZNet policy+value heads,        |
  | self-play workers, replay buffer, LR/temp annealing  |
  | Ultimate only (D4); ~950 lines                       |
  +---------------------------+--------------------------+
                              |
      RULES LAYER             v
  +------------------------------------------------------+
  | src/game.py  [1]                                     |
  | NormalGame / UltimateGame / BitUltimateGame          |
  | the only ground truth: legal_moves() make_move()     |
  | result() clone()                                     |
  | ~345 lines                                           |
  +------------------------------------------------------+

  SIDE CAR (no layer — loads in from the side)
    config.toml -------------- > src/ai.py    every engine switch + session defaults (D11)
    native/cengine.py + dll -- > src/ai.py    C backends (D25-D30): full MCTS tree,
                                              rollouts and the minimax family live in
                                              sba_engine.dll (~37x; built via
                                              native/build.py, auto-fallback to Python)
    static/ (qss·fonts·bg) --- > src/qtui.py  stylesheets, bundled OFL fonts, backdrops
    tests/ -------------------                headless pytest: rules, AI sanity,
                                              C engine, AZ recipe
```

Reading-route highlights, condensed from the retired
`docs/reading-guide.md` / 閱讀要點：

- **[1] `src/game.py`**：先讀 `UltimateGame` 的「落子位置決定對手下一步 macro」強制路由 — 整個專案的規則核心。
- **[2] `src/ai.py`**：按 `get_basic_move` → `minimax_*` → `mcts_search`（UCT 心臟）→ `analyze_position` 的順序跳讀；RAVE/GRAVE/Flat 是 A/B 對照組，非必修。
- **[3] `src/alphazero.py`**：只讀一個函式就讀 `self_play_game()`；「為什麼同預算打不贏純 MCTS」的誠實結論在 `docs/AlphaZero_UTTT_Technical_Report.html`。
- **[4] `src/SBA.py`**：總覽地圖，最後讀。
- **[5] UI**：兩端共用 [1]–[4]；`src/qtui.py` 的 `BoardWidget` 自繪是最複雜單點，初讀抓 `MenuPage` / `GamePage` 骨架即可。
- Decision log with the full "why" per entry / 每個決定的完整理由見決策日誌：[`docs/decisions.md`](docs/decisions.md)

---

## Ultimate Tic Tac Toe rules / 終極井字棋規則

- The big board is a 3x3 grid of 3x3 micro boards / 大棋盤由 9 個 3x3 小棋盤組成。
- Playing in micro-cell `(r, c)` forces the opponent's next move into macro-cell `(r, c)` / 在小格 `(r, c)` 落子後，對手必須在大格 `(r, c)` 落子。
- If that macro cell is won or full, the player may play any open macro cell / 若該大格已分出勝負或填滿，則可選擇任意未結束的大格。
- Winning a micro board claims that macro cell; a full micro board with no winner is a neutral draw / 贏得小棋盤即佔領該大格；填滿且無勝負的小棋盤算平局。
- Win the game by claiming 3 macro cells in a line; otherwise a full board is a draw / 連成三條大格即獲勝；全盤填滿無勝負則平局。

---

## Native C engine / 原生 C 引擎

All engine hot paths run in C since D25–D30 (`python native/build.py` builds
`native/sba_engine.dll`; WSL mingw cross-compile or any local gcc; CI builds
it automatically). Measured on this machine / 全部引擎熱路徑已下沉 C
（`native/build.py` 編譯；缺失時自動回退 Python，僅速度下降）：

| Metric / 指標 | Python | C (1 thread) | C (threads, D30) |
|---|---|---|---|
| MCTS rollout (per playout) | 0.51 ms | 0.029 ms (17.7x) | — |
| MCTS 800 sims (full search) | ~530 ms | 14 ms (37x) | 3.4 ms @ 8T (3.6x wall) |
| MCTS 3200 sims | — | 53 ms | 9.2 ms @ 8T (5.8x wall) |
| Heuristic rollout (AZ leaf value) | 0.47 ms | 0.028 ms (16.9x) | — |
| MCTS@1600 vs Basic (16 games) | 87.5% | 96.9% (parallel + reuse) | — |

Extras: tree reuse carries the warm subtree between moves (diff-walk, auto
fresh on rewind); settings page toggles every switch (D28); pattern rollout
variant kept off by default (D26 A/B: no gain at 4.4x cost).

---

## Benchmarks / 實測基準

n = 20; the 95% CI is roughly ±22% — for ranking only
n = 20 的 95% 信賴區間約 ±22%——僅供排序參考。

### All engines vs Basic (Ultimate, 20 games, MCTS@1600) / 全部引擎 vs Basic（終極 · 20 局 · MCTS@1600）

| Engine / 引擎 | W / D / L | Score / 得分率 |
|---|---|---|
| Minimax (depth 4) | 19 / 1 / 0 | **97.5%** |
| MCTS+RAVE@1600 | 19 / 1 / 0 | **97.5%** |
| Solver\*@1600 | 19 / 1 / 0 | **97.5%** |
| Minimax Pro (depth 4) | 18 / 2 / 0 | **95.0%** |
| MCTS+GRAVE@1600 | 18 / 2 / 0 | **95.0%** |
| MCTS@1600 | 19 / 0 / 1 | **95.0%** |
| Flat MCTS@1600 | 9 / 10 / 1 | 70.0% |
| Random | 1 / 2 / 17 | 10.0% |

\* On Ultimate, Solver is a plain MCTS fallback — it only truly solves Normal /
終極模式的 Solver 實為 MCTS fallback（僅普通模式真正求解）。

### All engines vs Basic (Normal, 20 games, MCTS@1600) / 全部引擎 vs Basic（普通 · 20 局 · MCTS@1600）

| Engine / 引擎 | W / D / L | Score / 得分率 |
|---|---|---|
| Flat MCTS@1600 | 8 / 10 / 2 | 65.0% |
| Minimax Pro (depth 4) | 4 / 16 / 0 | 60.0% |
| Minimax (depth 4) | 1 / 19 / 0 | 52.5% |
| MCTS@1600 | 1 / 19 / 0 | 52.5% |
| Solver (exact) | 0 / 20 / 0 | 50.0% |
| MCTS+GRAVE@1600 | 0 / 20 / 0 | 50.0% |
| MCTS+RAVE@1600 | 0 / 20 / 0 | 50.0% |
| Random | 0 / 1 / 19 | 2.5% |

### MCTS scales with budget / MCTS 隨預算變強（vs Minimax depth 4，Ultimate）

| Matchup / 對戰（30 局） | Score / 得分率 |
|---|---|
| Minimax vs MCTS@800 | **63.3%（Minimax）** |
| Minimax vs MCTS@2000 | **40.0%（MCTS）** |

Crossover around 1200–1600 sims — MCTS has no depth cap, so it wins as the buget grows (D20) 
交叉點約 1200–1600 sims——MCTS 無深度上限，預算越高越強（D20）。

### MCTS family round-robin (Ultimate, 20 games/pair, all engines@1600) / MCTS 家族循環賽（全引擎@1600）

All three share one playout policy: the plain win/block heuristic kernel
(`c_rollout_pattern = false`)
三者共用同一 playout 策略（plain 啟發式核心）。

| A vs B / 對戰 | W / D / L | Score / 得分率 |
|---|---|---|
| MCTS vs MCTS+RAVE | 6 / 2 / 12 | 35.0% |
| MCTS vs MCTS+GRAVE | 10 / 4 / 6 | 60.0% |
| MCTS+RAVE vs MCTS+GRAVE | 9 / 5 / 6 | 57.5% |

---

## Credits / 致謝

- [Qt for Python (PySide6)](https://github.com/qtproject/pyside-pyside-setup) - the desktop UI framework / 桌面版 UI 框架
- [PyTorch](https://github.com/pytorch/pytorch) - neural networks for the AlphaZero engine / AlphaZero 的神經網路
- [NumPy](https://github.com/numpy/numpy) - tensor utilities in the AlphaZero trainer / AlphaZero 訓練用的陣列工具

### Algorithm sources / 算法出處

The engines reimplement these published methods / 引擎實現自以下公開發表的方法：

- [AlphaZero](https://www.science.org/doi/10.1126/science.aar6404) — Silver et al., *Science* 2018: neural-guided MCTS with policy and value heads, trained by self-play / 神經導引 MCTS、策略與價值頭、自對弈訓練
- [MCTS](https://link.springer.com/chapter/10.1007/978-3-540-75538-8_7) — Coulom 2006: Monte-Carlo tree search with efficient selectivity and backup operators / 蒙地卡羅樹搜索
- [UCT](https://link.springer.com/chapter/10.1007/11871842_29) — Kocsis & Szepesvári 2006: the UCB1-based tree policy / 基於 UCB1 的樹策略
- [RAVE](https://doi.org/10.1145/1273496.1273531) — Gelly & Silver 2007, ICML: all-moves-as-first statistics inside UCT / AMAF 統計與 UCT 的結合
- [GRAVE](https://www.ijcai.org/proceedings/2015/0112.pdf) — Cazenave 2015, IJCAI 754–760: AMAF statistics shared through reference nodes / 以參考節點共享 AMAF 統計
- [Alpha-beta pruning](https://doi.org/10.1016/0004-3702(75)90019-3) — Knuth & Moore 1975: the minimax search backbone / minimax 搜索骨架

---

## License / 授權

This project is released under the **GNU General Public License v3 (GPLv3)**. See the [LICENSE](LICENSE) file for details / 本專案以 **GNU GPL v3（GPLv3）** 授權釋出，詳細條款請見 [LICENSE](LICENSE) 檔案。

### Third-party licenses / 第三方授權

The bundled UI fonts are distributed under the **SIL Open Font License 1.1**; the full license texts ship in `static/fonts/` (`OFL-Inter.txt`, `OFL-JetBrainsMono.txt`) / 內嵌的 UI 字體以 **SIL Open Font License 1.1** 授權散布，許可證全文隨字體附於 `static/fonts/`：

- [Inter](https://github.com/rsms/inter) — © 2020 The Inter Project Authors（`static/fonts/Inter-*.ttf`）
- [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) — © 2020 The JetBrains Mono Project Authors（`static/fonts/JetBrainsMono-*.ttf`）

Both families declare no Reserved Font Name, so the bundled static instances keep the original names / 兩者皆未宣告 Reserved Font Name，故內嵌的靜態實例沿用原名。Provenance notes: `static/fonts/README.md`.

