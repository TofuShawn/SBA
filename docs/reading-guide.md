# SBA 新手閱讀路線 / Beginner Reading Route

> 給想讀懂這個專案的人——包括作者自己。原則：**先跑、再讀、後深**。
> 依賴方向（D2）：`game.py` → `ai.py` → `alphazero.py` → `SBA.py` → UI。
> 行號以 2026-08 現況為準，僅作指標，不保證永遠精確。

## 0. 先跑起來（10 分鐘）
先建立「它在做什麼」的直覺，再開始讀代碼：

```powershell
python SBA.py          # PySide6 桌面版
python SBA.py --web    # NiceGUI 網頁版 → http://127.0.0.1:8080
```

各玩一場「玩家 vs 電腦」：先選普通棋盤 + Minimax，再選終極棋盤 + MCTS。
感受兩者的差別，然後想一個問題：**為什麼終極棋盤不讓玩家選 Minimax Pro？**
（答案在 §2c。）

## 1. 遊戲規則：`game.py`（345 行）— 必讀
這個檔案最單純，是整個專案的根。三個 class 按順序讀：

| 類別 | 行號 | 角色 |
|---|---|---|
| `NormalGame` | 37 | 普通 3×3，`make_move(index)` |
| `UltimateGame` | 75 | 終極 9×9，`make_move(macro, micro)` |
| `BitUltimateGame` | 142 | 用位元遮罩（bitboard）加速的版本，介面相同 |

重點函式：`legal_moves()` / `make_move()` / `result()` / `clone()`。
終極版多兩個關鍵概念：`macro_open()`（被指派的 macro 是否已定局）與
`micro_winner()`（單個 3×3 的勝者）——**終極規則的核心就是「落子位置決定對方下一步的 macro」**。

```powershell
python -m pytest tests/test_game.py -q   # 規則測試，讀懂後應全綠
```

## 2. 傳統 AI：`ai.py`（1504 行）— 主菜，分 5 站
不要從頭讀到尾，照這個順序跳：

### 2a 設定層（55–94）
`sba.toml` 的載入、`cfg_engine()` / `cfg_session()`。先知道「所有開關從哪來」，
之後看到 `cfg_engine('bitboard', True)` 才不會愣住。

### 2b 最簡單的引擎（96–137）
`get_basic_move()`：贏→擋→中心/角落的啟發式。五分鐘讀完。

### 2c Minimax：普通棋盤的完整解（138–283）
- `minimax_move_normal()`：普通 3×3 是**完美資訊零和遊戲**，8 條勝利線，
  全展開 9! ≈ 36 萬局面，配 alpha-beta 瞬間算完——所以普通棋盤的 Minimax 是「解」。
- `minimax_move_ultimate()`：終極棋盤分支數爆炸（每步最多 81 格 × 9 macro），
  只能 `depth=3` 加 `eval_ultimate()` 啟發式。
- **這就回答了 §0 的問題**：終極模式 Minimax 不是「完整解」，只是「有限深度估價」。

### 2d MCTS：核心演算法（285–469）
這是專案的心臟，值得慢慢讀：
- `MCTSNode`（285）+ `mcts_search()`（349）：選擇→展開→模擬→回傳。
- `_rollout_move()`（469）：啟發式 rollout（立即贏/擋）——這正是 MCTS 比純隨機 rollout 強的原因。
- `mcts_move()`（418）：包裝成「回傳一步棋」。

讀完後能回答：**UCT 公式在做什麼？**（探索 vs 利用的平衡）

### 2e 對照組：RAVE / GRAVE / Flat MCTS（734–967，可跳過）
`RAVENode` / `GraveNode` 是 MCTS 的研究變體，被保留下來做 A/B 對照
（D8），**不是必修**。`flat_mcts_move()`（515）是「不建樹」的簡化基準。

### 2f Minimax Pro：剪枝工程（1044–1145）
`_negamax_tt()`：置換表 + killer moves + LMR + aspiration window。
讀懂「它在加速什麼」即可，細節是效能工程。

### 2g 助手分析（1342–1507）
- `reason_for_move()`（1361）：只回傳代號（`'Win'` / `'Block'`…），
  雙語文案在 `SBA.REASON_TEXT`。
- `analyze_position()`（1407）：一次 MCTS 搜尋，輸出 top moves + 勝率。
- `position_win_rates()`（1470）：普通棋盤用殘局表（精確）、終極用輕量 MCTS。
- **「AI 助手」的數字全部來自這裡**——它不是魔法，是一場 MCTS 搜尋。

## 3. AlphaZero：`alphazero.py`（732 行）— 深水區
分三塊，各自獨立：

| 模組 | 行號 | 內容 |
|---|---|---|
| 網路 | 127–175 | `ResBlock` + `AZNet`（4 通道輸入、Policy/Value 雙頭） |
| 搜尋 | 178–384 | 神經引導 MCTS（`SearchNode`、batch 推理、虛擬損失） |
| 訓練 | 389–665 | 自對弈、多進程 worker、replay、LR/temp 退火 |

理解主線：**網路輸出先驗機率 + 局面價值 → 引導 MCTS 搜尋 → 自對弈產生訓練資料 →
反覆更新網路**。若只讀一個函式，讀 `self_play_game()`（407）。

```powershell
python alphazero.py eval --games 30 --sims 200   # 用現有模型評估
```

進階問題：**為什麼 AlphaZero 同預算下打不贏純 MCTS？**
答案在 `docs/AlphaZero_UTTT_Technical_Report.html` 第 6 章——樣本效率與
啟發式 rollout 的戰術深度，這是專案最精彩的誠實結論。

## 4. 膠水層：`SBA.py`（約 250 行）
小檔案，但負責：`new_session()`（44）、雙語 helper `t()`（108）與
`REASON_TEXT`、`--bench`（158）與 `--self-test`。最後讀，當作「總覽地圖」。

## 5. UI：先 `webui.py`（746 行）再 `qtui.py`（1514 行）
- `webui.py`：NiceGUI，宣告式頁面，讀 `@ui.page` 與 session 用法即可。
- `qtui.py`：PySide6，**最後讀**——它有自訂 `BoardWidget` 繪圖、執行緒 worker、
  歷史/控制列，是專案最複雜的單一檔案。初讀只須抓 `MenuPage` / `GamePage` 骨架，
  繪圖細節略過。

## 6. 設定與文件
- `sba.toml`：引擎開關總表（對應 `ai.py` 的 `cfg_engine`）。
- `docs/decisions.md`：D1–D18 決策日誌——**為什麼**每個設計是這樣。
- `docs/AlphaZero_UTTT_Technical_Report.html`：完整技術報告，面試/評審前必讀。
- `docs/rocm-wsl2.md`：AMD GPU 訓練環境踩坑紀錄。

## 7. 驗收：讀完能回答這些，就算讀懂了
1. 終極棋盤的「強制 macro 路由」規則在 `game.py` 哪裡實作？
2. 為什麼普通棋盤能用 Minimax 完整解，終極不行？
3. UCT 的「探索 vs 利用」平衡在哪一行？
4. MCTS 的 rollout 為什麼要用啟發式（贏/擋），而不是隨機？
5. `analyze_position()` 回傳的勝率從哪來？
6. `alphazero.py` 的網路輸出有哪兩個頭？分別餵給搜尋的哪一步？
7. 為什麼 AlphaZero 對 MCTS 只有 20% 勝率（同預算）？
8. 改一個引擎行為，要動 `sba.toml`、`ai.py`、還是 UI？

答完 1–4 代表你懂規則與傳統 AI；答完 5–6 代表你懂分析與深度學習；
答完 7–8 代表你能「擁有」這個專案——那正是面試和評審想聽的故事。
