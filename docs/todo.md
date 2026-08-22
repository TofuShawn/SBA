# UI / TUI 待辦清單 / UI & TUI TODO

> 獨立清單；狀態以 2026-08-22 為準。`[x]` = 已完成、`[ ]` = 待辦。
> Standalone checklist; status as of 2026-08-22.

## Textual TUI（目前預設介面）

### 已完成 / Done
- [x] 滑桿可調整：左右 / 上下 / `+` `-` / 滑鼠點擊（commit `97f6124`）
- [x] 勝率條移除「永遠 0%」的和局段，只顯示 X / O 並正規化到 100%（`73cdce7`）
- [x] 棋盤在較矮終端可捲動（`#board-col` `overflow-y`，`97f6124`）
- [x] AI 落子與分析用 Textual Worker 非同步執行 + `analysis_gen` 世代計數防競態（`aa07c0a`）
- [x] PySide6 桌面版（`qtui.py` + `vendor/siui`）移除，改用 Textual TUI（`aa07c0a`）

### 待辦 / Open
- [ ] **棋盤只顯示 3 個大格**：使用者回報特定終端尺寸下棋盤僅顯示第一排 3 個 macro；
      於 120×30 / 100×24 / 80×40 / 160×48 皆未能重現（棋盤恆為 45×23、23 行）。
      需使用者提供終端「欄 × 列」以精確重現；已加 overflow 捲動作為保險。
- [ ] 滑鼠點擊落子在不同終端 / DPI 下的座標準確性待實測。
- [ ] 遊戲畫面中的 CvC 速度滑桿：目前需先聚焦（Tab / 點擊）才能用鍵盤調整，確認體驗。
- [ ] **AlphaZero 預設模型**：遊戲使用 `models/az_ultimate.pt`（6000 局舊配方，vs MCTS 20%）；
      最強模型 `models/az_exp_stage2.pt`（600 局新配方，vs MCTS ~50%）尚未設為預設。
      若要 UI 直接使用最強模型，需複製到 `az_ultimate.pt` 或調整載入邏輯。
- [ ] 在純文字（Very Narrow）終端下的整體版面與可讀性做最後確認。

## 若未來重新加入 Qt 桌面版 / If a Qt UI is re-added later

> 註：`qtui.py` + `vendor/siui`（PySide6）已在 TUI 改寫時移除（`aa07c0a`）。
> 若打算改回 / 另建 Qt 版，以下修正需同步套用：
- [ ] 勝率條：移除永遠 0% 的和局段，只顯示 X / O（與 TUI 一致）
- [ ] 滑桿：支援多鍵（左右 / 上下 / `+` `-`）與滑鼠調整
- [ ] 棋盤：終端 / 視窗較矮時可捲動或縮放，避免只顯示部分 macro