# Design Decisions / 決策紀錄

A running log of the "why" behind project decisions, so future maintainers
don't have to re-litigate them. New entries: one line for the decision, one
for the reason, one for what it cost. / 記錄每個關鍵決定背後的原因，避免日後重蹈覆轍。新增條目：決定一行、原因一行、代價一行。

| ID | Date / 日期 | Decision / 決定 | Why / 原因 | Cost / 代價 |
| --- | --- | --- | --- | --- |
| D1 | 2026-07 | Desktop (PySide6) is the default entry; NiceGUI web is opt-in (`--web` or the menu switch) / 桌面版為預設入口，Web 選用啟動 | The web server should never run silently in the background; desktop gives the better default UX / Web 不該默默佔用資源，桌面版體驗較佳 | Two entry points; desktop and web sessions are separate / 兩套入口、session 各自獨立 |
| D2 | 2026-07 | Split the codebase: `game.py` / `ai.py` / `SBA.py` / `webui.py` / `qtui.py` / 拆分模組 | Pure logic separated from both UIs so engines stay reusable and testable / 邏輯與 UI 分離，引擎可重用、可測試 | None / 無 |
| D3 | 2026-07 | Solver is removed from the AI menus (engine kept for the analysis panel) / Solver 移出選單，保留給分析面板 | It is a tablebase oracle, not a meaningful opponent / 它是殘局表神諭，不是有意義的對手 | Menu shows one less AI / 選單少一個選項 |
| D4 | 2026-07 | AlphaZero trains/evaluates on Ultimate only / AlphaZero 僅限終極模式 | Normal Tic Tac Toe is a solved game; training on it is meaningless / 普通井字棋已被破解，訓練無意義 | Normal has no neural option / 普通模式無神經網路選項 |
| D5 | 2026-08 | PyQt-SiliconUI (PySide6 fork) is vendored under `vendor/siui/` and the project was relicensed to GPLv3 / 隨附 SiliconUI 並改為 GPLv3 | The desktop bundles GPLv3 code, so the combined work must be GPLv3; pip cannot install the PySide6 branch (no `setup.py`) / 合併作品須為 GPLv3；PySide6 分支無 setup.py 無法 pip 安裝 | Whole project is now GPLv3 (was MIT) / 全專案授權由 MIT 改為 GPLv3 |
| D6 | 2026-08 | Won chunks fill with the player's color; hovering reveals the cells again / 勝者區塊填色，hover 顯示回小格 | Bigger, clearer win label that stays inspectable / 勝者標示更清楚且仍可檢視 | Hover state + fade animation added to the board / 棋盤多一個 hover 狀態 |
| D7 | 2026-08 | Fonts prefer Noto Sans TC / 字型優先 Noto Sans TC | 微软雅黑 is not installed on every Windows system and broke text rendering / 雅黑非每台 Windows 都有，會造成文字破圖 | None / 無 |
| D8 | 2026-08 | Added `MCTS+GRAVE`; `MCTS+RAVE` is hidden from the menus / 新增 GRAVE，RAVE 移出選單 | GRAVE (Cazenave 2015) is RAVE's successor: same bias, lower memory; a 30-game bench put it ahead of both MCTS and RAVE / GRAVE 是 RAVE 的後繼者，30 局對戰略勝 | RAVE stays in code for tests/`--bench` / RAVE 程式保留 |
| D9 | 2026-08 | Desktop deps merged into `requirements.txt`; `requirements-qt.txt` deleted / 桌面依賴併入單一 requirements 檔 | One install command for everything / 安裝指令只剩一個 | None / 無 |
| D10 | 2026-08 | `MINDMAP.txt` and `docs/` notes stay local / gitignored / 個人筆記不進 repo | Personal working notes are not repository content / 個人工作筆記不屬於 repo 內容 | Not shared via git / 不隨 repo 分享 |
| D11 | 2026-08 | All engine switches/params centralized in `sba.toml`（`$SBA_CONFIG` overrides） / 引擎開關與參數集中到 `sba.toml` | Tune optimizations without touching code; self-tests reset to defaults / 免改碼即可調參，自測重置為預設 | Config must be loaded at import / 設定在 import 時載入 |
| D12 | 2026-08 | Adopted optimizations: opening book, micro tablebase, heuristic rollout, tree reuse, dynamic UCT, PW, early stop, fork eval, dynamic depth, killers/LMR/aspiration, symmetry, node pool, bitboard, parallel MCTS / 採用優化清單 | Bench favors MCTS with heuristic rollout + dynamic UCT + reuse (60% vs RAVE/GRAVE at 300 sims); bitboard measured no gain with the rollout, kept off / bench 顯示優化後純 MCTS 最強（60%）；bitboard 實測無增益故預設關 | More switches to tune; bitboard code kept but off / 開關變多；bitboard 保留但預設關 |
