# SBA - Basically Awful（Qt + TUI）/ 基本上很糟

[![CI](https://github.com/TofuShawn/SBA/actions/workflows/ci.yml/badge.svg)](https://github.com/TofuShawn/SBA/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/tag/TofuShawn/SBA?label=release)](https://github.com/TofuShawn/SBA/releases)

[English](README.md) | 繁體中文

**SBA**（即 *"SBA, Basically Awful"* 的簡稱）是一個**井字棋**與**終極井字棋**應用，提供 PySide6 **Qt 桌面版**（預設）與 Textual **終端機版**兩種介面，Arknights: Endfield 風格的工業 console 主題。內建 **8 種 AI 對手**（引擎熱路徑全部下沉 C，見下方「原生 C 引擎」），AI 分析面板，三種對戰模式。

> The name is a joke. The game is actually (mostly) fine.
> 名字是開玩笑的，遊戲其實（大致上）還不錯。

---

## 功能

- **兩種棋盤**：經典 3x3 井字棋與 9x9 終極井字棋
- **三種模式**：玩家對戰、人機對戰、電腦對戰
- **8 種可選 AI 引擎**：
  - `Random` — 隨機走棋
  - `Basic` — 能贏就贏、會擋棋、偏好中心與角落
  - `Minimax` — 經典極小化極大搜尋（普通模式完美、終極模式限深度）
  - `Minimax Pro` — 進階極小化極大（置換表 + 疊代加深）
  - `MCTS` — 蒙地卡羅樹搜尋 UCT（強度可調）
  - `MCTS+GRAVE` — 廣義 RAVE（Cazenave 2015）：透過 reference node 共享統計，省記憶體
  - ~~`MCTS+RAVE`~~ — 選單停用（由 GRAVE 取代；引擎保留供測試與 `--bench`）
  - ~~`Flat MCTS`~~ — 選單停用（研究用基準：只做根層 rollout，無樹）
  - `Solver` — 完美解：普通模式讀持久化解表（`data/tt_normal.bin`，`--build-tables` 重建）；終極模式為 MCTS fallback
  - `AlphaZero` — 神經網路引導的 MCTS（僅終極模式）
- **AI 助手面板**：分析目前局面，顯示最佳 3-5 步、勝率、一句話原因（win/block/fork/center/corner/positional），點擊可在棋盤上標示；可開啟**策略熱圖**（AZ policy 色塊疊空格，需已訓練模型）
- **可調 AI 強度**：MCTS 迭代數（200-3000）與 Minimax 深度（2-6）
- **引擎優化**（`config.toml` 可開關）：開局書、micro 殘局表、啟發式 rollout、樹重用、動態 UCT、漸進式加粗、提早終止、fork 評估、killer/LMR/aspiration、D4 對稱性、物件池、bitboard、多執行緒 MCTS
- **電腦對戰控制**：速度（0.1-2.0s）、自動播放、手動「下一步」
- **先手選擇**：人機模式可選玩家（X）或電腦先手
- **終末地工業 console 主題**：訊號黃 × 炭灰、deck 標籤、瞄準括號、落子蓋章與勝利線動畫（可「減少動態」）
- **設定頁**：引擎開關全數圖形化，保存寫回 `config.toml`
- **歷史卡**：點擊任一步直接回放該局面
- **無頭 pytest 測試**（規則、引擎、UI smoke、ABI 契約；數量以 `pytest tests --collect-only -q` 為準）

---

## 環境需求

- Python **3.12+**（於 3.13 開發）
- 核心依賴：`pyside6>=6.12,<7`（桌面版）、`textual>=8.2,<9`（終端機版）、`rich>=15,<16`、`numpy>=2.0,<3`、`safetensors>=0.4,<2`（AlphaZero 權重／自對弈記錄格式）

```bash
pip install -r requirements.txt
```

原生 C 引擎核心以 `python src/tools/build.py` 編譯至 `src/engines/sba_engine.dll`
（Windows 本機 gcc 或 WSL mingw 交叉編譯；CI 自動構建）。DLL 缺失時引擎
自動回退純 Python 實作，功能不受影響、僅速度下降。

**選用 — AlphaZero**：神經網路引擎需要 `torch`（CPU 版即可，模型很小；
`numpy` 已列入 `requirements.txt`）。只有要使用或訓練 AlphaZero 才需要安裝：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

已訓練的 AlphaZero 網路會在 GitHub Releases 發佈。AMD 顯卡 GPU 訓練
（Windows 原生 TheRock wheel 或 WSL2）見
[`docs/rocm-wsl2.md`](docs/rocm-wsl2.md)。訓練方法與效能分析見
[`docs/AlphaZero_UTTT_Technical_Report.html`](docs/AlphaZero_UTTT_Technical_Report.html)。

新手依下方「架構地圖」的閱讀順序（1 → 5）；從零教學見
[`docs/tutorial-zh.md`](docs/tutorial-zh.md)（簡體）／
[`docs/tutorial-zh-Hant.md`](docs/tutorial-zh-Hant.md)（繁體）。

要動這個 repo？`python dev.py` 是任務入口（build / test / test-rust /
lint / format / clippy / clippy-core / cgate / bench-selfplay / self-test）；
分支模型與環境需求見 [`docs/development.md`](docs/development.md)。

---

## 使用方式

### 1. 桌面版（預設）

```bash
# 統一入口：全部命令從根 shim 出發
python sba.py

# 等同的模組路徑（相容保留）
python src/session/SBA.py
python src/session/SBA.py --qt
```

桌面版使用 Endfield 工業 console 主題（`static/endfield.qss` + 原生繪製棋盤），
選單頁可進 Settings 調整引擎開關。

### 2. 終端機版（TUI）

```bash
python sba.py --tui
```

Textual 終端機介面，同一套引擎與設定；適合 SSH／無 GUI 環境。

CLI 參數（`python sba.py` 與 `python src/session/SBA.py` 同一組）：

| 參數 | 說明 |
| --- | --- |
| `--qt` | 啟動 PySide6 桌面版（預設） |
| `--tui` | 改啟動 Textual 終端機版 |
| `--debug` | 後端詳細日誌 |
| `--self-test` | 執行無頭測試後結束 |
| `--bench [--ai-a A] [--ai-b B] [--games N] [--iters N] [--depth N] [--normal]` | 引擎勝率對戰測試（預設 MCTS 家族循環賽） |
| `--build-tables` | 重建 `data/tt_normal.bin`——Normal 模式持久化完美解表（全部 5,478 個可達局面） |
| `--train-az` | 等同執行 `src/az/alphazero.py train` |
| `--eval-az` | 等同執行 `src/az/alphazero.py eval` |
| `--bench-tui` | 互動對戰儀表板（選引擎、即時比分） |
| `--train-tui` | 互動訓練儀表板（進度、loss／勝率曲線、里程碑日誌） |

### 3. 自測

```bash
python sba.py --self-test
# Windows: start.bat --self-test（本地啟動器，不入 git）
```

跑 pytest 套件：規則、AI 正確性、對局終止、AlphaZero 冒煙（需
`pip install -r requirements-dev.txt`：pytest 屬開發工具，不在 runtime 依賴裡）。

### 3b. MCTS 對戰測試

```bash
# 預設終極模式循環賽
python sba.py --bench
# 自訂對戰組合
python sba.py --bench --ai-a MCTS --ai-b MCTS+GRAVE --games 40 --iters 400
python sba.py --bench --ai-a Minimax --ai-b MCTS+GRAVE --depth 4
# 較小規模或普通棋盤
python sba.py --bench --games 10 --iters 200 --normal
```

`--ai-a`/`--ai-b` 可用引擎：Random, Basic, Minimax, Minimax Pro,
MCTS, MCTS+RAVE, MCTS+GRAVE, Flat MCTS, Solver, AlphaZero。每局輪換先手，
輸出各組合得分率（勝 = 1、和 = 0.5）。

### 4. 訓練／評估 AlphaZero

```bash
# 訓練終極模式神經網路（普通模式已破解，不支援）
python sba.py --train-az --games 400 --sims 80
# 或直接走模組
python src/az/alphazero.py train --games 400 --sims 80

# 評估模型對戰 MCTS
python sba.py --eval-az --games 30 --sims 200
```

訓練選項：`--ckpt-every 25`（每 N 局存一次 checkpoint）、
`--dirichlet-eps 0.25`（根節點探索噪聲，0 停用）、`--lr 1e-3`（線性退火到 1/5）、
`--workers 8`（多進程自對弈，每個 worker 一份 GPU 模型副本）、
`--selfplay-engine auto|rust|python`（自對弈來源：Rust exe／Python mp.Pool；
auto 探測 exe，缺席自動降級）、`--selfplay-procs N`（K 個 exe 進程分攤對局）、
`--csv`（訓練曲線 CSV——每局一行：loss / LR / temp / eval 勝敗和 / checkpoint，
預設 `<model>.csv`；新 run 覆寫、resume 續寫）、
`--channels 128`、`--blocks 5`。自對弈前段 temp 從 1.0 線性退到 0.05，
晚期遊戲近乎最強應手。

## 設定

所有引擎開關與參數集中在 **`config.toml`**（repo 根目錄）：啟發式 rollout、
樹重用、動態 UCT、開局書、micro 殘局表、LMR/killer/aspiration、對稱性、
物件池、bitboard、多執行緒 MCTS、UCT/RAVE 常數，以及 session 預設值
（`mcts_budget`、`minimax_depth`）。缺省鍵自動回退程式碼預設；可用環境變數
`SBA_CONFIG` 指向其他檔案，但該檔案同時會被讀取與寫回，因此必須位於
**本倉庫或系統暫存目錄內**（與設定寫入共用同一道路徑防護）。

---

## 架構地圖

```
      UI 層（二選一——兩個前端、同一後端）
  +----------------------+  +----------------------+
  | src/ui/qtui.py  [5a] |  | src/ui/tui.py  [5b] |
  | PySide6 桌面版        |  | rich 終端機版        |
  +----------+-----------+  +----------+-----------+
             |                         |
             +------------+------------+
                          |
      黏合層               v
  +------------------------------------------------------+
  | src/session/SBA.py  [4]                              |
  | CLI 入口 · 會話狀態 · 雙語字串                        |
  | --tui / --bench-tui / --train-tui / --train-az        |
  +---------------------------+--------------------------+
                              |
      引擎層                   v
  +------------------------------------------------------+
  | src/engines/ai.py  [2]  門面（D47）：get_ai_move      |
  |   派發、AlphaZero 橋、快取重置                        |
  |   引擎模組（C 原生引擎即搜尋後端——python 樹已退役）： |
  |   src/config.py            ROOT、config.toml、路徑防護|
  |   src/engines/cbridge.py   ctypes 邊界（即 DLL）      |
  |   src/analysis/books.py    Basic 啟發式 + 開局書      |
  |   src/analysis/tablebase.py  Normal 求解 + Solver     |
  |   src/analysis/analysis.py  助手 + 勝率               |
  +---------+--------------------------------------------+
            |
            |  lazy import（選了 AlphaZero 才載入 torch）
            v
  +------------------------------------------------------+
  | src/az/alphazero.py  [3]                             |
  | 神經引導 MCTS：AZNet policy+value 頭、                |
  | 自對弈 worker、replay buffer、LR/temp 退火           |
  | 僅終極模式（D4）                                      |
  +---------------------------+--------------------------+
                              |
      規則層                   v
  +------------------------------------------------------+
  | src/core/game.py  [1]                                |
  | NormalGame / UltimateGame / BitUltimateGame          |
  | 唯一的事實來源：legal_moves() make_move()            |
  | result() clone()                                     |
  +------------------------------------------------------+

  側掛（不屬於任何層——從側面載入）
    config.toml -------------- > src/config.py  全部引擎開關 + session 預設（D11）
    src/engines/cengine.py + dll > src/engines/cbridge.py  C 後端（D25-D30, D37）：
                                              完整 MCTS 樹、rollout、minimax 家族
                                              都在 sba_engine.dll（約 37 倍；由
                                              src/tools/build.py 構建，缺失自動回退 Python）
    src/sba-core (Rust) --- > src/az/rustselfplay.py  自對弈 exe（D55-D59）：
                                              wave 搜尋 + 批次 burn 推論，
                                              safetensors 記錄、K 進程分攤
    src/sba-game (Rust) --- > src/az/rustbridge.py  狀態層（D48）：
                                              clone/apply/legal/result 打包位元板，
                                              產物 az/sba_game.dll|so
    static/ (qss·字體·背景) --- > src/ui/qtui.py  樣式表、內嵌 OFL 字體、背景
    tests/ -------------------                無頭 pytest：規則、AI 正確性、
                                              C 引擎、AZ 配方、Rust parity、golden 向量
```

閱讀要點：

- **[1] `src/core/game.py`**：先讀 `UltimateGame` 的「落子位置決定對手下一步
  macro」強制路由——整個專案的規則核心。
- **[2] `src/engines/ai.py`（門面）**：引擎分居 `engines/cbridge.py`
  （C 邊界）/ `analysis/tablebase.py` / `analysis/analysis.py` /
  `analysis/books.py`；按 `get_basic_move` → `minimax_*`（C 後端）→
  `analyze_position` 的順序跳讀；MCTS 家族的 python 樹已隨移除專案退役。
- **[3] `src/az/alphazero.py`**：只讀一個函式就讀 `self_play_game()`；
  「為什麼同預算打不贏純 MCTS」的誠實結論在
  `docs/AlphaZero_UTTT_Technical_Report.html`。
- **[4] `src/session/SBA.py`**：總覽地圖，最後讀。
- **[5] UI**：兩端共用 [1]–[4]；`src/ui/qtui.py` 的 `BoardWidget` 自繪是最
  複雜單點，初讀抓 `MenuPage` / `GamePage` 骨架即可。
- 每個決定的完整理由見決策日誌：[`docs/decisions.md`](docs/decisions.md)

---

## 終極井字棋規則

- 大棋盤由 9 個 3x3 小棋盤組成。
- 在小格 `(r, c)` 落子後，對手必須在大格 `(r, c)` 落子。
- 若該大格已分出勝負或填滿，則可選擇任意未結束的大格。
- 贏得小棋盤即佔領該大格；填滿且無勝負的小棋盤算平局。
- 連成三條大格即獲勝；全盤填滿無勝負則平局。

---

## 原生 C 引擎

全部引擎熱路徑自 D25–D51 下沉 C（`python src/tools/build.py` 編譯
`src/engines/sba_engine.dll`；WSL mingw 交叉編譯或任意本機 gcc；CI 自動構建），
遊戲狀態層自 D48/D50 下沉 Rust（`src/az/sba_game.dll`，cargo；Python 版保留作
參考與 fallback）。本機實測：

| 指標 | Python | C（單線程） | C（多線程，D30） |
|---|---|---|---|
| MCTS rollout（每次 playout） | 0.51 ms | 0.029 ms（17.7x） | — |
| MCTS 800 sims（完整搜尋，對稱開） | ~530 ms | 12.9 ms（41x） | 3.4 ms @ 8T（3.6x wall） |
| MCTS 3200 sims | — | 51.2 ms | 9.2 ms @ 8T（5.8x wall） |
| 啟發式 rollout（AZ 葉值） | 0.47 ms | 0.028 ms（16.9x） | — |

| Rust 狀態層（D52） | 對比 Python game |
|---|---|
| playout 迴圈（狀態操作） | 2.14x |
| fullgame（每步 legal + encode + clone） | 1.64x |
| AZ 自對弈 stage-1（C 樹主導） | 約持平（1.04x） |

| Rust 自對弈核心（D57） | 對比 Python mp.Pool 路徑 |
|---|---|
| 端到端自對弈（800 sims、8 threads） | 6.1 vs 23.7 pos/s（0.26x——Phase A ≥4x 線未達） |

Phase A 對齊紀錄（D57）：Python wave 搜尋 1:1 移植進 `search.rs`；layer-3
parity 200/200 top-1（100%）/ KL 0.0000——200 局全部位元組級相等訪問向量；
批次填充 0.894（每派發約 229 行）。剩餘吞吐差距為後端限制：burn/cubecl 前向
在 RDNA3 Vulkan 上每 229 行批約 37 ms，torch/ROCm-HIP 每 256 行批 2.84 ms
（約 13 倍 kernel 差），多 queue 流水線在本機的天花板約 1.4x。重測：
`python dev.py bench-selfplay` 與 `python src/tools/parity_search.py`；
詳見 docs/decisions.md D57。Python wave 搜尋參考已凍結為 golden 向量
（`tests/golden/`）——Python 實作未來退場後，對齊錨仍在。

AZ 自對弈約 7–9 秒一局（80 sims、單 worker）——瓶頸在神經網路前向，
狀態層加速受 Amdahl 上限約束。

補充：樹重用在 C 與 Python 兩條路都保留熱子樹（`tree_reuse` 開關雙端把關，D53）；
設定頁可撥全部開關（D28）；pattern rollout 變體預設關閉（D26；2026-11-15 於
當前 C 樹重驗——對 Basic 85% vs 92.5%，rollout 成本 3.9 倍：仍然無增益）。

---

## 實測基準

n = 20；95% 信賴區間約 ±22%——僅供排序參考。2026-11-15 於 C 引擎（D51）重測。

### 全引擎 vs Basic（終極 · 20 局 · MCTS@1600）

| 引擎 | 勝 / 和 / 敗 | 得分率 |
|---|---|---|
| Solver\*@1600 | 20 / 0 / 0 | **100%** |
| MCTS+GRAVE@1600 | 20 / 0 / 0 | **100%** |
| MCTS@1600 | 20 / 0 / 0 | **100%** |
| Minimax（深度 4） | 19 / 1 / 0 | **97.5%** |
| MCTS+RAVE@1600 | 19 / 1 / 0 | **97.5%** |
| Flat MCTS@1600 | 18 / 2 / 0 | **95.0%** |
| Minimax Pro（深度 4） | 17 / 3 / 0 | 92.5% |
| Random | 1 / 2 / 17 | 10.0% |

\* 終極模式的 Solver 實為 MCTS fallback（僅普通模式真正求解）。

### 全引擎 vs Basic（普通 · 20 局 · MCTS@1600）

| 引擎 | 勝 / 和 / 敗 | 得分率 |
|---|---|---|
| Flat MCTS@1600 | 8 / 10 / 2 | 65.0% |
| Minimax（深度 4） | 2 / 18 / 0 | 55.0% |
| Solver（精確） | 2 / 18 / 0 | 55.0% |
| Minimax Pro（深度 4） | 0 / 20 / 0 | 50.0% |
| MCTS@1600 | 0 / 20 / 0 | 50.0% |
| MCTS+GRAVE@1600 | 0 / 20 / 0 | 50.0% |
| MCTS+RAVE@1600 | 0 / 20 / 0 | 50.0% |
| Random | 0 / 1 / 19 | 2.5% |

### MCTS 隨預算變強（對 Minimax 深度 4，終極）

| 對戰（30 局） | 得分率 |
|---|---|
| Minimax vs MCTS@800 | **48.3%（Minimax）** |
| Minimax vs MCTS@2000 | **23.3%（MCTS 76.7%）** |

800 sims 時已是五五波、2000 sims 時 MCTS 勝約 77%——C 樹（D51）之後交叉點
下移至約 600–800 sims（D20）。

### MCTS 家族循環賽（終極 · 每對 20 局 · 全引擎@1600）

三者共用同一 playout 策略（plain 啟發式核心，`c_rollout_pattern = false`）。

| A vs B | 勝 / 和 / 敗 | 得分率 |
|---|---|---|
| MCTS vs MCTS+RAVE | 13 / 3 / 4 | **72.5%** |
| MCTS vs MCTS+GRAVE | 17 / 1 / 2 | **87.5%** |
| MCTS+RAVE vs MCTS+GRAVE | 3 / 8 / 9 | 35.0% |

---

## 致謝

- [Qt for Python (PySide6)](https://github.com/qtproject/pyside-pyside-setup) — 桌面版 UI 框架
- [Textual](https://github.com/Textualize/textual) — 終端機版 UI 框架
- [Rich](https://github.com/Textualize/rich) — CLI 表格、進度條與量測儀表
- [PyTorch](https://github.com/pytorch/pytorch) — AlphaZero 的神經網路
- [NumPy](https://github.com/numpy/numpy) — AlphaZero 訓練用的陣列工具
- [safetensors](https://github.com/huggingface/safetensors) — 權重與自對弈記錄格式
- [burn + cubecl](https://github.com/tracel-ai/burn) — 自對弈 exe 背後的純 Rust 推論引擎（wgpu/Vulkan 後端）
- 以及讓 exe 保持精簡的 Rust crate：[clap](https://github.com/clap-rs/clap)、[rand/rand_distr](https://github.com/rust-random/rand)、[serde_json](https://github.com/serde-rs/json)、[sha2](https://github.com/RustCrypto/hashes)、[bytemuck](https://github.com/Lokathor/bytemuck)

### 算法出處

引擎實現自以下公開發表的方法：

- [AlphaZero](https://www.science.org/doi/10.1126/science.aar6404) — Silver et al., *Science* 2018：神經導引 MCTS、策略與價值頭、自對弈訓練
- [MCTS](https://link.springer.com/chapter/10.1007/978-3-540-75538-8_7) — Coulom 2006：蒙地卡羅樹搜索
- [UCT](https://link.springer.com/chapter/10.1007/11871842_29) — Kocsis & Szepesvári 2006：基於 UCB1 的樹策略
- [RAVE](https://doi.org/10.1145/1273496.1273531) — Gelly & Silver 2007, ICML：AMAF 統計與 UCT 的結合
- [GRAVE](https://www.ijcai.org/proceedings/2015/0112.pdf) — Cazenave 2015, IJCAI 754–760：以參考節點共享 AMAF 統計
- [Alpha-beta 剪枝](https://doi.org/10.1016/0004-3702(75)90019-3) — Knuth & Moore 1975：minimax 搜索骨架

---

## 授權

**程式碼**（`src/`、`tests/`、C 引擎與兩個 Rust crate）以 **GNU GPL v3（GPLv3）**
授權釋出，詳細條款請見 [LICENSE](LICENSE)。

**文件——作者創作**：技術報告（`docs/AlphaZero_UTTT_Technical_Report.html`
及其發佈的 HTML/PDF 版本）與教程（`docs/tutorial-zh*.md`）以
**創作共用姓名標示－非商業性 4.0**（[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)）
授權；法律全文見 [LICENSE.docs](LICENSE.docs)。內嵌元件各自保有授權：
Noto Sans TC（SIL OFL 1.1）、highlight.js 主題（BSD-3-Clause）、MathJax 3
（Apache-2.0，建置期 SVG）。

**工程記錄**——`docs/decisions.md`（決策記錄）、`docs/development.md`、
`docs/releases/*.md` 與 `docs/rocm-wsl2.md`——是 GPL 專案的技術記錄，
隨程式碼以 **GPL-3.0-or-later** 授權。

### 第三方授權

內嵌的 UI 字體以 **SIL Open Font License 1.1** 授權散布，許可證全文隨字體附於
`static/fonts/`：

- [Inter](https://github.com/rsms/inter) — © 2020 The Inter Project Authors（`static/fonts/Inter-*.ttf`）
- [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) — © 2020 The JetBrains Mono Project Authors（`static/fonts/JetBrainsMono-*.ttf`）
