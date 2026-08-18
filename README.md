# SBA - Basically Awful / 基本上很糟

**SBA**（即 *"SBA, Basically Awful"* 的簡稱）是一個雙語（English / 繁體中文）的**井字棋**與**終極井字棋**網頁應用，使用 [NiceGUI](https://nicegui.io) 打造。內建 **8 種 AI 對手**、AI 助手分析面板、三種對戰模式，以及 Material Design 3 風格介面。

> The name is a joke. The game is actually (mostly) fine.
> 名字是開玩笑的，遊戲其實（大致上）還不錯。

---

## Features / 功能

- **Two game types / 兩種棋盤**：classic 3x3 Tic Tac Toe（經典井字棋）和 9x9 Ultimate Tic Tac Toe（終極井字棋）
- **Three modes / 三種模式**：Player vs Player（玩家對戰）、Player vs Computer（人機對戰）、Computer vs Computer（電腦對戰）
- **8 AI engines / 八種 AI 引擎**：
  - `Random` - random legal move / 隨機走棋
  - `Basic` - win if possible, block, prefer center/corners / 能贏就贏、會擋棋、偏好中心與角落
  - `Minimax` - classic alpha-beta search / 經典極小化極大搜尋（普通模式完美、終極模式限深度）
  - `Minimax Pro` - negamax + transposition table + iterative deepening / 進階極小化極大（置換表 + 疊代加深）
  - `MCTS` - Monte Carlo Tree Search with UCT / 蒙地卡羅樹搜尋（強度可調）
  - `MCTS+RAVE` - MCTS with AMAF/RAVE move sharing / 蒙地卡羅 + RAVE 走法共享
  - `Solver` - perfect play via a full Normal-mode tablebase / 普通模式完美殘局表
  - `AlphaZero` - neural-guided MCTS / 神經網路引導的 MCTS（可訓練）
- **AI Assistant panel / AI 助手面板**：分析目前局面，顯示最佳 3-5 步、勝率、一句話原因（win/block/fork/center/corner/positional），點擊可在棋盤上標示
- **Adjustable AI strength / 可調 AI 強度**：MCTS 迭代數（200-3000）與 Minimax 深度（2-6）
- **CvC controls / 電腦對戰控制**：速度（0.1-2.0s）、自動播放、手動「下一步」
- **First-player choice / 先手選擇**：人機模式可選玩家先手（X）或電腦先手
- **Material Design 3 style UI** with light/dark toggle / 深淺色主題切換
- **Headless self-test suite / 無頭自測**（57 項檢查）與 **Docker** 映像

---

## Requirements / 環境需求

- Python **3.12+**（於 3.13 開發）
- 核心依賴：`nicegui>=3.16`

Install core dependencies / 安裝核心依賴：

```bash
pip install -r requirements.txt
```

**Optional - AlphaZero / 選用 - AlphaZero**：神經網路引擎需要 `torch` 與 `numpy`（CPU 版 PyTorch 即可，模型很小）。只有要使用或訓練 AlphaZero 才需要安裝：

```bash
pip install torch numpy
```

已訓練的 AlphaZero 網路會存放在 `models/`（已被 gitignore）。

---

## Usage / 使用方式

### 1. Start the web app / 啟動網頁應用

```bash
# canonical / 標準啟動方式
python SBA.py

# Windows convenience launcher (local only, uses .venv) / Windows 便利啟動檔（僅本地，使用 .venv）
run.bat
```

Open http://localhost:8080 (or http://127.0.0.1:8080) in your browser / 瀏覽器開啟即可。

CLI flags / 指令參數：

| Flag / 參數 | Description / 說明 |
| --- | --- |
| `--host HOST` | Bind address / 綁定位址（預設 `0.0.0.0`） |
| `--port PORT` | Port / 連接埠（預設 `8080`） |
| `--debug` | Verbose backend logging / 後端詳細日誌 |
| `--self-test` | Run headless tests and exit / 執行無頭測試後結束 |
| `--train-az` | Alias for `alphazero.py train` / 等同執行 `alphazero.py train` |

### 2. Self-test / 自測

```bash
python SBA.py --self-test
# Windows: run.bat --self-test (local launcher, not tracked in git)
```

Runs 57 headless checks covering rules, AI sanity, termination, and AlphaZero smoke tests / 執行 57 項無頭檢查，涵蓋規則、AI 正確性、對局終止與 AlphaZero 冒煙測試。

### 3. Train / evaluate AlphaZero / 訓練與評估 AlphaZero

```bash
# train a neural net (saves to models/) / 訓練神經網路（存到 models/）
python alphazero.py train --game normal   --games 400 --sims 80
python alphazero.py train --game ultimate --games 300 --sims 80

# evaluate a trained model vs MCTS / 評估模型對戰 MCTS
python alphazero.py eval --game normal --games 30 --sims 200
```

### 4. Docker

```bash
docker build -t sba .
docker run -p 8080:8080 sba
```

---

## Source layout / 原始碼結構

| File / 檔案 | Purpose / 用途 |
| --- | --- |
| `SBA.py` | Entry point: CLI flags, session state, self-tests / 入口：指令參數、session 狀態、自測 |
| `game.py` | Game rules: `NormalGame` / `UltimateGame`, move application, board helpers / 遊戲規則與棋盤輔助 |
| `ai.py` | All AI engines + `get_ai_move` + assistant analysis / 所有 AI 引擎與分析功能 |
| `webui.py` | NiceGUI web UI (menu, board, assistant panel, CvC controls) / 網頁介面 |
| `alphazero.py` | AlphaZero neural MCTS (training + evaluation) / AlphaZero 訓練與評估 |
| `static/styles.css` | Material Design 3 stylesheet / 樣式表 |
| `run.bat` | Local Windows launcher (not tracked in git) / 本地 Windows 啟動檔（未納入 git） |
| `Dockerfile` | Container image (CPU-only torch) / 容器映像（CPU 版 torch） |
| `requirements.txt` | Core Python dependencies / 核心 Python 依賴 |

Dependency direction is one-way: `game.py` -> `ai.py` -> `SBA.py` -> `webui.py` / 依賴方向為單向：`game.py` -> `ai.py` -> `SBA.py` -> `webui.py`。

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

---

## License / 授權

This project is released under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for details / 本專案以 [MIT 授權](LICENSE) 釋出，詳細條款請見 [LICENSE](LICENSE) 檔案。
