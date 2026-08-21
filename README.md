# SBA - Basically Awful / 基本上很糟

**SBA**（即 *"SBA, Basically Awful"* 的簡稱）是一個雙語（English / 繁體中文）的**井字棋**與**終極井字棋**網頁應用，使用 [NiceGUI](https://nicegui.io) 打造。內建 **8 種 AI 對手**、AI 助手分析面板、三種對戰模式，以及 Material Design 3 風格介面。

> The name is a joke. The game is actually (mostly) fine.
> 名字是開玩笑的，遊戲其實（大致上）還不錯。

---

## Features / 功能

- **Two game types / 兩種棋盤**：classic 3x3 Tic Tac Toe（經典井字棋）和 9x9 Ultimate Tic Tac Toe（終極井字棋）
- **Three modes / 三種模式**：Player vs Player（玩家對戰）、Player vs Computer（人機對戰）、Computer vs Computer（電腦對戰）
- **8 selectable AI engines / 八種可選 AI 引擎**：
  - `Random` - random legal move / 隨機走棋
  - `Basic` - win if possible, block, prefer center/corners / 能贏就贏、會擋棋、偏好中心與角落
  - `Minimax` - classic alpha-beta search / 經典極小化極大搜尋（普通模式完美、終極模式限深度）
  - `Minimax Pro` - negamax + transposition table + iterative deepening / 進階極小化極大（置換表 + 疊代加深）
  - `MCTS` - Monte Carlo Tree Search with UCT / 蒙地卡羅樹搜尋（強度可調）
  - `MCTS+GRAVE` - Generalized RAVE (Cazenave 2015): shares RAVE stats through reference nodes, lower memory / 廣義 RAVE：透過 reference node 共享 RAVE 統計，省記憶體
  - ~~`MCTS+RAVE`~~ - hidden in the menu (superseded by MCTS+GRAVE; engine kept for tests and `--bench`) / 已停用（選單不可選，由 MCTS+GRAVE 取代；引擎保留供測試與 `--bench`）
  - ~~`Flat MCTS`~~ - hidden in the menu (research/learning baseline: root-level playouts, no tree) / 已停用（選單不可選，研究/學習用基準：只做根層 rollout，無樹搜尋）
  - ~~`Solver`~~ - disabled in the menu (engine kept for the analysis panel) / 已停用（選單不可選，引擎保留供分析面板使用）
  - `AlphaZero` - neural-guided MCTS (Ultimate only) / 神經網路引導的 MCTS（僅終極模式）
- **AI Assistant panel / AI 助手面板**：分析目前局面，顯示最佳 3-5 步、勝率、一句話原因（win/block/fork/center/corner/positional），點擊可在棋盤上標示
- **Adjustable AI strength / 可調 AI 強度**：MCTS 迭代數（200-3000）與 Minimax 深度（2-6）
- **Engine optimizations / 引擎優化**（`sba.toml` 可開關）：開局書、micro 殘局表、啟發式 rollout、樹重用、動態 UCT、漸進式加粗、提早終止、fork 評估、killer/LMR/aspiration、D4 對稱性、物件池、bitboard、多執行緒 MCTS
- **CvC controls / 電腦對戰控制**：速度（0.1-2.0s）、自動播放、手動「下一步」
- **First-player choice / 先手選擇**：人機模式可選玩家先手（X）或電腦先手
- **Material Design 3 style UI** with light/dark toggle / 深淺色主題切換
- **SiliconUI desktop theme / 桌面版 SiliconUI 主題**：PySide6 桌面版使用 [PyQt-SiliconUI](https://github.com/MayBeLaterOrNot/PyQt-SiliconUI)（PySide6 fork）的深色玻璃主題；該套件缺失時自動退回內建深色玻璃樣式
- **Headless pytest suite / 無頭 pytest 測試**（36 tests）與 **Docker** 映像

---

## Requirements / 環境需求

- Python **3.12+**（於 3.13 開發）
- 核心依賴：`nicegui>=3.16`

Install core dependencies / 安裝核心依賴：

```bash
pip install -r requirements.txt
```

PySide6（Qt for Python，桌面版用）已併入 `requirements.txt`，不需要另外安裝檔。
PyQt-SiliconUI（PySide6 fork）已隨專案 vendoring 於 `vendor/siui/`（GPLv3），
無需另外安裝；若刪除該目錄，桌面版會自動退回內建深色玻璃樣式。

**Optional - AlphaZero / 選用 - AlphaZero**：神經網路引擎需要 `torch` 與 `numpy`（CPU 版 PyTorch 即可，模型很小）。只有要使用或訓練 AlphaZero 才需要安裝：

```bash
pip install torch numpy
```

已訓練的 AlphaZero 網路會存放在 `models/`（已被 gitignore）。

AMD 顯卡若要 GPU 訓練（Windows 原生 TheRock wheel 或 WSL2），可參考
[`docs/rocm-wsl2.md`](docs/rocm-wsl2.md)。

AlphaZero 訓練方法與效能分析見
[`docs/AlphaZero_UTTT_Technical_Report.html`](docs/AlphaZero_UTTT_Technical_Report.html)。

新手想快速讀懂專案，可依 [`docs/reading-guide.md`](docs/reading-guide.md)
的閱讀路線進行。

---

## Usage / 使用方式

### 1. Desktop app (default) / 桌面版（預設）

```bash
# PySide6 desktop UI / PySide6 桌面版
python SBA.py
# or explicitly / 或明確指定
python SBA.py --qt
# same as: / 等同於
python qtui.py
```

桌面版預設使用 SiliconUI 深色玻璃主題（按鈕與下拉選單來自 PyQt-SiliconUI）。
若 `vendor/siui/` 不存在，會退回內建深色玻璃 QSS 主題，功能不受影響。

### 2. NiceGUI web app (opt-in) / Web 版（選用）

The web server only starts when explicitly enabled / Web 伺服器只在明確啟用時啟動：

```bash
python SBA.py --web
```

Open http://127.0.0.1:8080 in your browser / 瀏覽器開啟 http://127.0.0.1:8080。也可以在桌面版的選單勾選「Enable NiceGUI Web UI（啟動 Web 介面）」來開啟 Web。

CLI flags / 指令參數：

| Flag / 參數 | Description / 說明 |
| --- | --- |
| `--web` | Start the NiceGUI web server (no desktop app) / 啟動 NiceGUI Web 伺服器（不開桌面版） |
| `--qt` | Start the PySide6 desktop app (default) / 啟動 PySide6 桌面版（預設） |
| `--host HOST` | Web bind address / Web 綁定位址（預設 `0.0.0.0`） |
| `--port PORT` | Web port / Web 連接埠（預設 `8080`） |
| `--debug` | Verbose backend logging / 後端詳細日誌 |
| `--self-test` | Run headless tests and exit / 執行無頭測試後結束 |
| `--bench [--ai-a A] [--ai-b B] [--games N] [--iters N] [--depth N] [--normal]` | Win-rate benchmark (default: MCTS family round-robin) / 引擎勝率對戰測試（預設 MCTS 家族循環賽） |
| `--train-az` | Alias for `alphazero.py train` / 等同執行 `alphazero.py train` |

### 2. Self-test / 自測

```bash
python SBA.py --self-test
# Windows: run.bat --self-test (local launcher, not tracked in git)
```

Runs the pytest suite covering rules, AI sanity, termination, and AlphaZero
smoke tests / 執行 pytest 測試套件，涵蓋規則、AI 正確性、對局終止與 AlphaZero 冒煙測試（需 `pip install pytest`，已列入 `requirements.txt`）。

### 2b. MCTS benchmark / MCTS 對戰測試

```bash
# default: MCTS vs MCTS+RAVE vs MCTS+GRAVE round-robin on Ultimate / 預設終極模式循環賽
python SBA.py --bench
# pick your own matchup / 自訂對戰組合
python SBA.py --bench --ai-a MCTS --ai-b MCTS+GRAVE --games 40 --iters 400
python SBA.py --bench --ai-a Minimax --ai-b MCTS+GRAVE --depth 4
# smaller / faster, or Normal board / 較小規模或普通棋盤
python SBA.py --bench --games 10 --iters 200 --normal
```

Available engines for `--ai-a`/`--ai-b`: Random, Basic, Minimax, Minimax Pro,
MCTS, MCTS+RAVE, MCTS+GRAVE, Flat MCTS, Solver, AlphaZero. The first player alternates
per game and per-match win rates are printed / `--ai-a`/`--ai-b` 可用引擎如上；
每局輪換先手，輸出各組合勝率。

### 3. Train / evaluate AlphaZero / 訓練與評估 AlphaZero

```bash
# train a neural net for Ultimate (Normal is solved and not supported) / 訓練終極模式神經網路（普通模式已破解，不支援）
python alphazero.py train --games 400 --sims 80

# evaluate a trained model vs MCTS / 評估模型對戰 MCTS
python alphazero.py eval --games 30 --sims 200
```

Training extras / 訓練選項：`--ckpt-every 25`（每 N 局存一次 checkpoint）、
`--dirichlet-eps 0.25`（根節點探索噪聲，0 停用）、`--lr 1e-3`（線性退火到 1/5）、
`--workers 8`（多進程自對弈，每個 worker 一份 GPU 模型副本）、
`--rollout-weight 0.5`（葉節點混合 win/block 啟發式隨機對弈，0 = 純網路）、
`--channels 128`、`--blocks 5`。自對弈前段 temp 從 1.0 線性退到 0.05，
晚期遊戲近乎最強應手。

### 4. Docker

```bash
docker build -t sba .
docker run -p 8080:8080 sba
```

## Configuration / 設定

All engine switches and parameters live in **`sba.toml`**（repo root）: MCTS
rollout heuristic, tree reuse, dynamic UCT, opening book, micro tablebase,
LMR/killer/aspiration, symmetry, object pool, bitboard, multithreaded MCTS,
UCT/RAVE constants, and session defaults（`mcts_budget`, `minimax_depth`）.
Missing keys fall back to code defaults; point `$SBA_CONFIG` at another file
to override / 所有引擎開關與參數集中在 `sba.toml`（repo 根目錄）：啟發式 rollout、
樹重用、動態 UCT、開局書、micro 殘局表、LMR/killer/aspiration、對稱性、物件池、
bitboard、多執行緒 MCTS、UCT/RAVE 常數，以及 session 預設值（`mcts_budget`、
`minimax_depth`）。缺省鍵自動回退程式碼預設；可用環境變數 `SBA_CONFIG` 指向其他檔案。

---

## Source layout / 原始碼結構

| File / 檔案 | Purpose / 用途 |
| --- | --- |
| `SBA.py` | Entry point: CLI flags, session state, self-tests / 入口：指令參數、session 狀態、自測 |
| `game.py` | Game rules: `NormalGame` / `UltimateGame`, move application, board helpers / 遊戲規則與棋盤輔助 |
| `ai.py` | All AI engines + `get_ai_move` + assistant analysis / 所有 AI 引擎與分析功能 |
| `webui.py` | NiceGUI web UI (menu, board, assistant panel, CvC controls) / 網頁介面 |
| `qtui.py` | PySide6 desktop UI (menu, board, assistant panel, CvC controls, web switch) / 桌面版介面 |
| `alphazero.py` | AlphaZero neural MCTS (training + evaluation) / AlphaZero 訓練與評估 |
| `static/styles.css` | Material Design 3 stylesheet / 樣式表 |
| `vendor/siui/` | Vendored PyQt-SiliconUI runtime (`silicon/` + `icons/`, GPLv3) / 隨附的 SiliconUI 執行時期（GPLv3） |
| `run.bat` | Local Windows launcher (not tracked in git) / 本地 Windows 啟動檔（未納入 git） |
| `Dockerfile` | Container image (CPU-only torch) / 容器映像（CPU 版 torch） |
| `requirements.txt` | Core + desktop dependencies (NiceGUI, PySide6) / 核心與桌面版依賴（NiceGUI、PySide6） |
| `sba.toml` | Engine configuration (switches, constants, session defaults) / 引擎設定（開關、常數、session 預設） |

Dependency direction is one-way: `game.py` -> `ai.py` -> `SBA.py` -> {`webui.py`, `qtui.py`} / 依賴方向為單向：`game.py` -> `ai.py` -> `SBA.py` -> {`webui.py`, `qtui.py`}。

---

## Ultimate Tic Tac Toe rules / 終極井字棋規則

- The big board is a 3x3 grid of 3x3 micro boards / 大棋盤由 9 個 3x3 小棋盤組成。
- Playing in micro-cell `(r, c)` forces the opponent's next move into macro-cell `(r, c)` / 在小格 `(r, c)` 落子後，對手必須在大格 `(r, c)` 落子。
- If that macro cell is won or full, the player may play any open macro cell / 若該大格已分出勝負或填滿，則可選擇任意未結束的大格。
- Winning a micro board claims that macro cell; a full micro board with no winner is a neutral draw / 贏得小棋盤即佔領該大格；填滿且無勝負的小棋盤算平局。
- Win the game by claiming 3 macro cells in a line; otherwise a full board is a draw / 連成三條大格即獲勝；全盤填滿無勝負則平局。

---

## Credits / 致謝

- [NiceGUI](https://nicegui.io) - reactive web UI framework (bundles Quasar / Tailwind) / 響應式網頁框架
- [PyTorch](https://pytorch.org) - neural networks for the AlphaZero engine / AlphaZero 的神經網路
- [NumPy](https://numpy.org) - tensor utilities in the AlphaZero trainer / AlphaZero 訓練用的陣列工具
- [PyQt-SiliconUI](https://github.com/ChinaIceF/PyQt-SiliconUI) by [ChinaIceF](https://github.com/ChinaIceF) - the original PyQt5 UI library this desktop theme is based on / 桌面主題所基於的原版 PyQt5 UI 函式庫（GPLv3）
- [PyQt-SiliconUI (PySide6 fork)](https://github.com/MayBeLaterOrNot/PyQt-SiliconUI) by [MayBeLaterOrNot](https://github.com/MayBeLaterOrNot) - PySide6 port used by the desktop app (branch `PySide6`, commit `6445d42`), vendored under `vendor/siui/` / 桌面版使用的 PySide6 移植版（GPLv3，隨附於 `vendor/siui/`）

---

## License / 授權

This project is released under the **GNU General Public License v3 (GPLv3)**. See the [LICENSE](LICENSE) file for details / 本專案以 **GNU GPL v3（GPLv3）** 授權釋出，詳細條款請見 [LICENSE](LICENSE) 檔案。

The desktop app bundles PyQt-SiliconUI（GPLv3）under `vendor/siui/`, so the combined work is distributed under GPLv3 / 桌面版隨附 PyQt-SiliconUI（GPLv3，見 `vendor/siui/LICENSE`），合併作品因此以 GPLv3 發行。
