# 從零開始：用 Python 寫一個井字棋項目（SBA 同款教程）

> 本教程面向**完全沒有編程基礎**的初學者。目標：帶你一步步寫出一個和本倉庫 SBA 類似的項目——普通井字棋 + 終極井字棋，多種 AI 引擎（隨機、啓發式、Minimax、MCTS），再加上終端圖形界面（Textual）和 pytest 測試。
>
> 每一章的代碼都可以直接運行，建議**邊看邊敲**（不要複製粘貼，手敲記憶更深）。
>
> 注：本倉庫的實際界面目前是 **Textual 終端界面**（`python SBA.py`）；
> 第 11 章仍以 PySide6 教學 Qt 的核心概念（信號槽、事件循環、線程），
> 當作圖形界面的入門素材。
>
> 文中「對照項目源碼」提到的**行號僅供參考**，代碼演進後可能漂移；找不到就按函數名搜索（VS Code 按 `Ctrl+F`）。

---

## 目錄

- [第 0 章：準備環境](#第-0-章準備環境)
- [第 1 章：Python 速成](#第-1-章python-速成)
- [第 2 章：文字版井字棋](#第-2-章文字版井字棋)
- [第 3 章：整理成類（對應項目 game.py）](#第-3-章整理成類對應項目-gamepy)
- [第 4 章：終極井字棋（對應項目 UltimateGame）](#第-4-章終極井字棋對應項目-ultimategame)
- [第 5 章：第一個 AI —— Random](#第-5-章第一個-ai--random)
- [第 6 章：Basic AI —— 啓發式引擎](#第-6-章basic-ai--啓發式引擎)
- [第 7 章：Minimax —— 窮舉搜索](#第-7-章minimax--窮舉搜索)
- [第 8 章：MCTS —— 蒙特卡洛樹搜索](#第-8-章mcts--蒙特卡洛樹搜索)
- [第 9 章：優化技巧（進階）](#第-9-章優化技巧進階)
- [第 10 章：配置文件（對應 sba.toml）](#第-10-章配置文件對應-sbatoml)
- [第 11 章：圖形界面（PySide6 教學）](#第-11-章圖形界面pyside6-教學)
- [第 12 章：測試（pytest）](#第-12-章測試pytest)
- [第 13 章：AlphaZero 概念入門](#第-13-章alphazero-概念入門)
- [第 14 章：Git 版本控制](#第-14-章git-版本控制)
- [第 15 章：虛擬環境 venv](#第-15-章虛擬環境-venv)
- [第 16 章：項目管理（把作業做成作品）](#第-16-章項目管理把作業做成作品)
- [附錄 A：新手常見錯誤對照表](#附錄-a新手常見錯誤對照表)
- [附錄 B：動手路線建議](#附錄-b動手路線建議)
- [附錄 C：推薦資源](#附錄-c推薦資源)

---

## 學習路線圖

```
第0章 安裝環境
  ↓
第1章 Python 速成（用到什麼學什麼）
  ↓
第2章 文字版井字棋（第一次寫出能玩的遊戲！🎉）
  ↓
第3章 整理成類 → 項目裏的 game.py
  ↓
第4章 終極井字棋（9×9）
  ↓
第5章 Random AI ── 第6章 Basic AI ── 第7章 Minimax ── 第8章 MCTS
  ↓
第9章 優化技巧（bitboard、樹重用、開局書…）
第10章 配置文件
  ↓
第11章 圖形界面（PySide6 教學）
第12章 pytest 測試
  ↓
第13章 AlphaZero 概念
  ↓
第14章 Git ── 第15章 venv ── 第16章 項目管理（交作業前）
```

> 💡 **必讀**：第 0–8 章 + 第 12 章（測試）+ 第 14 章（Git）+ 第 15 章（venv）。
> **可選**：第 9 章（優化技巧，知道名字即可）、第 10 章（配置）、第 11 章（界面，想動手再看）、第 13 章（AlphaZero，理解概念即可）、第 16 章（項目管理，交作業前讀）。

**建議的項目結構**（先建一個自己的文件夾，別直接改倉庫）：

```
my_tictactoe/
├── game.py       # 遊戲規則（第 3、4 章）
├── ai.py         # AI 引擎（第 5-8 章）
├── main.py       # 文字版入口（第 2 章）
├── sba.toml      # 配置（第 10 章）
├── ui.py         # 圖形界面（第 11 章）
└── tests/
    └── test_game.py   # 測試（第 12 章）
```

---

## 第 0 章：準備環境

### 0.1 安裝 Python

1. 打開 <https://www.python.org/downloads/>
2. 下載最新版 Python 3.12+（Windows 用戶點黃色 "Download Python 3.x.x" 按鈕）
3. 安裝時**務必勾選 "Add Python to PATH"**——這個勾非常重要，勾了之後命令行裏才能用 `python` 命令
4. 一路點 Install 完成

### 0.2 驗證安裝

打開命令行（Windows 按 `Win+R`，輸入 `cmd` 回車）：

```
python --version
```

應該輸出類似 `Python 3.13.x`。如果提示"不是內部或外部命令"，說明 PATH 沒勾上，重裝一次。

### 0.3 運行第一個程序

新建文件 `hello.py`，內容：

```python
print("你好，世界！")
```

在命令行裏 `cd` 到該文件所在目錄，運行：

```
python hello.py
```

### 0.4 遇到中文亂碼？

Windows 老式控制檯默認不是 UTF-8 編碼。兩種解決辦法：

- 運行前先輸入 `chcp 65001` 切換編碼
- 或者程序裏暫時用英文 `print`（教程代碼的打印信息都做了兼容）

### 0.5 編輯器推薦

- **新手最友好：Thonny**（自帶逐步調試，能看每一行代碼怎麼跑）
- **主流：VS Code**（裝官方 "Python" 擴展）
- **進階 / 讀大項目：PyCharm（Community 免費版）**——"跳到定義""查找引用"、圖形化調試器、內建 git 都最強，適合讀 SBA 這種幾千行的項目；缺點是啓動慢、喫內存，所以新手期先用上面兩個。

---

## 第 1 章：Python 速成

只學做這個項目**用得到**的基礎，不貪多。

### 1.1 變量

```python
name = "Alice"      # 字符串 str
age = 20            # 整數 int
pi = 3.14           # 小數 float
is_student = True   # 布爾 bool（True / False）
```

`#` 後面的內容是**註釋**，Python 不執行，用來給人看。

### 1.2 列表（list）

列表是有順序的一串東西：

```python
numbers = [1, 2, 3, 4, 5]
print(numbers[0])    # 1 —— 索引從 0 開始！
print(numbers[-1])   # 5 —— -1 表示最後一個
print(len(numbers))  # 5 —— 長度
```

常用操作：

```python
numbers.append(6)    # 末尾加一個 → [1,2,3,4,5,6]
numbers[0] = 99      # 修改第 0 個
3 in numbers         # True —— 判斷是否在裏面
```

**切片**（後面 clone 會用到，很重要）：

```python
a = [1, 2, 3]
b = a[:]     # 複製出一份**新**列表
b[0] = 999   # 改 b 不影響 a
print(a)     # [1, 2, 3] ✓
```

### 1.3 循環

```python
# for 循環：把每個元素過一遍
for i in range(9):        # range(9) = 0,1,2,...,8
    print(i)

for mark in ['', 'X', 'O']:
    print(mark)

# while 循環：條件爲真就一直跑
count = 0
while count < 3:
    print(count)
    count += 1            # count = count + 1 的簡寫
```

### 1.4 條件判斷

```python
age = 18
if age >= 18:
    print("成年")
elif age >= 6:
    print("兒童")
else:
    print("嬰兒")
```

比較符號：`==`（相等）、`!=`（不等）、`<`、`>`、`<=`、`>=`。
**新手最容易搞混**：`=` 是賦值，`==` 是比較。判斷相等必須用 `==`。

### 1.5 函數

函數 = 起個名字的一坨代碼，可以反覆調用：

```python
def add(a, b):
    return a + b

print(add(3, 5))   # 8
```

- `def` 定義函數
- `return` 把結果交回去
- 沒有 return 的函數默認返回 `None`（表示"空"）

### 1.6 類（class）—— 面向對象

類 = 數據的"模板"。一個類可以創建很多**實例**，每個實例有自己的數據：

```python
class Dog:
    def __init__(self, name):   # 構造方法：創建實例時自動執行
        self.name = name        # self 指"這個實例自己"

    def bark(self):             # 方法：屬於類的函數
        print(f"{self.name} 汪汪！")

d1 = Dog("小白")
d2 = Dog("大黃")
d1.bark()   # 小白 汪汪！
d2.bark()   # 大黃 汪汪！
```

- `__init__` 是構造方法，創建對象時自動運行，負責初始化數據
- 方法的第一個參數永遠是 `self`
- `self.name` 是"這個實例自己的 name"，不同實例互不影響

**爲什麼 SBA 要用類？** 一局棋就是"一個棋盤實例"。AI 搜索時要複製很多個棋盤來做試驗（"如果我在 4 落子會怎樣？"），用類最方便。

### 1.7 導入模塊

```python
import random                # Python 自帶的隨機模塊
random.choice([1, 2, 3])     # 隨機選一個 → 可能是 2
random.shuffle(xs)           # 原地打亂列表

from game import X, O        # 從自己寫的 game.py 裏導入
```

一個 `.py` 文件就是一個**模塊**，別的文件可以 import 它。

### 1.8 元組（tuple）與字典（dict）—— 後面馬上會用到

**元組**：和列表很像，但用 `()` 且**不能修改**。本項目的終極井字棋用 `(大格, 小格)` 表示一步棋：

```python
move = (4, 2)          # 在大格 4、小格 2 落子
macro, cell = move     # 拆包：macro=4, cell=2 —— 元組可以這樣"拆"開
```

**字典**：鍵 → 值的對應表，用 `{}`：

```python
score = {'X': 1, 'O': -1}   # X 贏 +1，O 贏 -1
score['X']                  # 1 —— 按鍵取值
score.get('D', 0)           # 0 —— 沒有 'D' 這個鍵時返回默認值 0
score['O'] = 5              # 修改或新增
'X' in score                # True —— 判斷鍵是否存在
```

後面 AI 的"局面→分數"緩存（置換表）、配置文件（sba.toml）全是字典。

### 1.9 小練習

1. 寫函數 `is_even(n)` 判斷偶數
2. 寫類 `Counter`，方法 `click()` 讓計數 +1
3. 用 for 循環打印 1 到 10 的平方

---

## 第 2 章：文字版井字棋

現在開始做正事。先做**最簡單、不用類**的版本，理解遊戲是怎麼"跑"起來的。

### 2.1 棋盤怎麼表示？

3×3 棋盤有 9 個格子。用**長度爲 9 的列表**表示，索引 0 到 8：

```
 0 | 1 | 2
---+---+---
 3 | 4 | 5
---+---+---
 6 | 7 | 8
```

每個元素是 `''`（空）、`'X'` 或 `'O'`：

```python
board = [''] * 9    # 全空棋盤
```

### 2.2 打印棋盤

```python
def print_board(board):
    # 空格子顯示編號（方便玩家輸入），有棋子的顯示棋子
    def cell(i):
        if board[i]:
            return board[i]
        return str(i)

    print()
    print(f" {cell(0)} | {cell(1)} | {cell(2)} ")
    print("---+---+---")
    print(f" {cell(3)} | {cell(4)} | {cell(5)} ")
    print("---+---+---")
    print(f" {cell(6)} | {cell(7)} | {cell(8)} ")
```

`f"..."` 是 **f-string**：大括號裏的內容會被求值後填進字符串。

### 2.3 誰贏了？

8 條獲勝線（3 行 + 3 列 + 2 條對角線），每條線寫成三個索引的元組：

```python
LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # 三行
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # 三列
    (0, 4, 8), (2, 4, 6),              # 兩條對角線
]

def winner(board):
    for a, b, c in LINES:
        if board[a] != '' and board[a] == board[b] == board[c]:
            return board[a]     # 'X' 或 'O'
    return None
```

**關鍵技巧**：`board[a] == board[b] == board[c]` 是**鏈式比較**，等價於"三者相等"。先判斷 `board[a] != ''`，防止三個空格被判成贏家。

### 2.4 主循環

```python
board = [''] * 9
current = 'X'   # 當前輪到誰

while True:
    print_board(board)

    # 1. 玩家輸入（帶防呆：輸錯不崩潰）
    raw = input(f"輪到 {current}，輸入格子編號 0-8：")
    if not raw.isdigit():        # 不是數字 → 重來
        print("請輸入數字 0-8！")
        continue
    move = int(raw)
    if move < 0 or move > 8:     # 數字超範圍 → 重來
        print("數字要在 0-8 之間！")
        continue
    if board[move] != '':
        print("這個格子已經有子了，換一個！")
        continue    # 回到循環開頭，重新輸入

    # 2. 落子
    board[move] = current

    # 3. 判斷結束
    w = winner(board)
    if w:
        print_board(board)
        print(f"{w} 贏了！")
        break
    if '' not in board:     # 棋盤滿了還沒贏 → 平局
        print_board(board)
        print("平局！")
        break

    # 4. 換人
    current = 'O' if current == 'X' else 'X'
```

- `continue`：跳過本輪剩餘代碼，回到循環開頭
- `break`：直接退出循環
- `'O' if current == 'X' else 'X'`：**三元表達式**，"如果 current 是 X 就取 O，否則取 X"

把 2.2、2.3、2.4 合在一起存成 `main.py`，跑一遍——你已經有一個能玩的文字版井字棋了！🎉

**運行效果示例**（輸入 4、再輸入 0、再輸入 8……最後 X 贏了）：

```
 0 | 1 | 2
---+---+---
 3 | 4 | 5
---+---+---
 6 | 7 | 8
輪到 X，輸入格子編號 0-8：4

 0 | 1 | 2
---+---+---
 3 | X | 5
---+---+---
 6 | 7 | 8
輪到 O，輸入格子編號 0-8：0
...
X 贏了！
```

（棋盤打印順序和這個示例一樣，就說明你寫對了。）

### 2.5 本章總結

- 棋盤 = 長度爲 9 的列表
- 勝負 = 檢查 8 條線
- 遊戲 = "落子 → 判勝負 → 換人" 的循環

---

## 第 3 章：整理成類（對應項目 game.py）

文字版的問題是：棋盤、當前玩家都散在外面，不好給 AI 用（AI 要能複製、能試走）。把狀態裝進類裏：

### 3.1 完整代碼

```python
X = 'X'
O = 'O'
EMPTY = ''

LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


class NormalGame:
    def __init__(self):
        self.board = [EMPTY] * 9   # 棋盤
        self.current = X           # 輪到誰

    def legal_moves(self):
        """所有合法着法（空格的索引列表）"""
        return [i for i, c in enumerate(self.board) if c == EMPTY]

    def make_move(self, index):
        """落子，然後自動換人（遊戲沒結束的話）"""
        if self.board[index] != EMPTY:
            raise ValueError('這個格子已經有子了')
        self.board[index] = self.current
        if self.result() is None:
            self.current = O if self.current == X else X

    def winner(self):
        for a, b, c in LINES:
            if self.board[a] != EMPTY and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def result(self):
        """遊戲結果：'X' / 'O' / 'D'(平局) / None(沒結束)"""
        w = self.winner()
        if w:
            return w
        if EMPTY not in self.board:
            return 'D'
        return None

    def clone(self):
        """複製一個一模一樣的棋盤 —— AI 搜索的地基"""
        g = NormalGame()
        g.board = self.board[:]    # 列表切片 = 複製！
        g.current = self.current
        return g
```

### 3.2 新知識點逐個講

**列表推導式**：
```python
[i for i, c in enumerate(self.board) if c == EMPTY]
```
等價於：
```python
result = []
for i, c in enumerate(self.board):   # enumerate 同時給出索引 i 和元素 c
    if c == EMPTY:
        result.append(i)
```

**raise ValueError**：主動報錯。AI 不該走非法着法；如果真的走了，讓程序立刻崩潰比悄悄出錯好（快速失敗原則）。

**clone 爲什麼是最重要的方法**：AI 搜索要回答"如果我在 4 落子會怎樣？如果我在 0 落子會怎樣？"每個假設都需要一個獨立的棋盤。clone 讓 AI 在副本上隨便試驗，不弄髒真實棋局。

⚠️ **必須**用 `self.board[:]` 複製，不能直接 `g.board = self.board`！後者只是兩個名字指向同一個列表，改一個另一個也跟着變。這叫**淺拷貝陷阱**，新手必踩。

**結果約定**（全項目通用）：`result()` 返回 `'X'`、`'O'`、`'D'`（平局）或 `None`（沒結束）。`None` 在 if 裏當 False 用，所以可以寫 `if game.result():`。

### 3.3 重寫主循環

```python
game = NormalGame()
while True:
    print_board(game.board)          # 複用第 2 章的 print_board
    move = int(input(f"輪到 {game.current}，輸入 0-8："))
    if move not in game.legal_moves():
        print("非法着法！")
        continue
    game.make_move(move)
    r = game.result()
    if r:
        print_board(game.board)
        print("平局！" if r == 'D' else f"{r} 贏了！")
        break
```

### 3.4 對照項目源碼

現在打開倉庫裏的 `game.py`，你應該能認出：

- 第 37-73 行就是我們的 `NormalGame`（項目裏多了 `is_full`、`is_over` 等便捷方法）
- 第 283 行 `apply_move(game, move)`：統一處理普通/終極兩種棋盤的落子（後面第 4 章會用到）
- 第 290 行 `apply_clone_result(game, move)`：克隆 + 落子 + 返回結果，一步到位（第 6 章用）

---

## 第 4 章：終極井字棋（對應項目 UltimateGame）

### 4.1 規則

- 大棋盤 = 3×3 個大格，每個大格里面是一個 3×3 小棋盤 → 總共 81 個格子
- 你在小棋盤 `(r, c)` 落子 → 對手**下一手必須**在大格 `(r, c)` 下
- 若那個大格已分出勝負或填滿 → 對手可自由選任何未結束的大格
- 贏下一個 3×3 小棋盤 = 佔領該大格
- 大格連成三線 = 贏得整局

### 4.2 數據結構

```python
micro = [[''] * 9 for _ in range(9)]   # 9 個小棋盤，每個 9 格
macro = [''] * 9                       # 9 個大格狀態：'' / 'X' / 'O' / 'D'
active_macro = None                    # 對手必須去的大格編號（None = 自由選擇）
```

⚠️ 注意：`[[''] * 9] * 9` 是**錯的**！它產生 9 個指向同一個列表的引用，改一個全改。必須用列表推導式 `[[''] * 9 for _ in range(9)]`（`_` 是"我不關心這個變量"的慣例寫法）。

### 4.3 核心邏輯

```python
class UltimateGame:
    def __init__(self):
        self.micro = [[EMPTY] * 9 for _ in range(9)]
        self.macro = [EMPTY] * 9
        self.current = X
        self.active_macro = None

    def macro_open(self, m):
        """大格 m 還能不能下？（沒被佔領 且 還有空格）"""
        return (self.macro[m] == EMPTY
                and any(c == EMPTY for c in self.micro[m]))

    def legal_moves(self):
        if self.active_macro is not None and self.macro_open(self.active_macro):
            # 被"路由"到大格 active_macro，只能在那裏下
            return [(self.active_macro, i)
                    for i in range(9)
                    if self.micro[self.active_macro][i] == EMPTY]
        # 自由選擇：遍歷所有開放的大格
        moves = []
        for m in range(9):
            if not self.macro_open(m):
                continue
            for i in range(9):
                if self.micro[m][i] == EMPTY:
                    moves.append((m, i))
        return moves

    def make_move(self, macro, micro):
        self.micro[macro][micro] = self.current
        w = winner_of(self.micro[macro])   # 複用第 2 章的 winner()，改名就行
        if w:
            self.macro[macro] = w          # 小棋盤分出勝負 → 佔領大格
        elif all(c != EMPTY for c in self.micro[macro]):
            self.macro[macro] = 'D'        # 填滿無勝負 → 該大格作廢
        # 下一手的路由規則：對手必須去大格 micro（編號恰好相同）
        self.active_macro = micro if self.macro_open(micro) else None
        self.current = O if self.current == X else X

    def winner(self):
        # 和大棋盤判斷完全一樣，只是把 board 換成 macro
        for a, b, c in LINES:
            if self.macro[a] in (X, O) and self.macro[a] == self.macro[b] == self.macro[c]:
                return self.macro[a]
        return None

    def result(self):
        w = self.winner()
        if w:
            return w
        if all(not self.macro_open(m) for m in range(9)):  # 所有大格都關了
            return 'D'
        return None

    def clone(self):
        g = UltimateGame()
        g.micro = [row[:] for row in self.micro]  # 二維列表：每行都要複製！
        g.macro = self.macro[:]
        g.current = self.current
        g.active_macro = self.active_macro
        return g
```

### 4.4 關鍵技巧總結

- **小棋盤勝負 → 大格狀態**：在 `make_move` 裏自動完成，玩家和 AI 都不用管
- **路由規則一行搞定**：`active_macro = micro if self.macro_open(micro) else None`
- **二維列表深拷貝**：`[row[:] for row in self.micro]`，每行單獨複製
- `any(...)` 和 `all(...)` 是內置函數：一個滿足即 True / 全部滿足才 True

### 4.5 對照項目源碼

項目 `game.py` 第 75-139 行就是完整版。另外項目還有 `BitUltimateGame`（第 142 行）——同樣的規則，但用**整數位運算**存棋盤（81 個格子 = 81 個二進制位），速度更快，第 9 章講。

---

## 第 5 章：第一個 AI —— Random

### 5.1 思路

從所有合法着法裏隨便挑一個：

```python
import random

def random_move(game):
    return random.choice(game.legal_moves())
```

就這麼簡單。但它立下全項目的規矩：**所有 AI 引擎都是"輸入棋盤，輸出一個着法"的函數**。以後 UI 可以隨便換引擎，因爲接口統一。

### 5.2 分發函數

項目 `ai.py` 有統一入口 `get_ai_move(game, ai_type)`（ai.py 第 1325 行）：

```python
def get_ai_move(game, ai_type):
    if ai_type == 'Random':
        return random.choice(game.legal_moves())
    if ai_type == 'Basic':
        return get_basic_move(game)
    if ai_type == 'Minimax':
        return minimax_move_normal(game)[0]
    # ... 其他引擎
    raise ValueError(f'Unknown AI type: {ai_type}')
```

UI 只調 `get_ai_move(game, 'MCTS')`，不用關心內部實現。這叫**多態 / 策略模式**——本教程裏最重要的架構思想之一。

---

## 第 6 章：Basic AI —— 啓發式引擎

### 6.1 思路（模仿一個還不錯的普通玩家）

1. 我能直接贏嗎？→ 贏
2. 對手下一步能贏嗎？→ 擋
3. 都不行 → 中心 > 角落 > 邊

### 6.2 "試走"技巧

判斷"走這步會不會贏"，最穩妥的辦法是**在克隆上試走**：

```python
def apply_clone_result(game, move):
    g = game.clone()
    apply_move(g, move)
    return g.result()
```

這是整個項目反覆使用的核心思想：**用模擬代替規則枚舉**。相比"數某條線有幾個子"的寫法，試走法不會漏掉任何情況（比如一步成兩條線），代碼也更短。

### 6.3 完整代碼

```python
def get_basic_move(game):
    player = game.current
    opp = O if player == X else X
    moves = game.legal_moves()

    # 1. 能贏就贏
    for m in moves:
        if apply_clone_result(game, m) == player:
            return m

    # 2. 找出"走了之後對手能直接贏"的壞棋
    losing = []
    for m in moves:
        g = game.clone()
        apply_move(g, m)
        # 我走 m 之後，對手隨便一步都能贏嗎？
        if any(apply_clone_result(g, m2) == opp for m2 in g.legal_moves()):
            losing.append(m)
    safe = [m for m in moves if m not in losing]
    if safe and len(safe) < len(moves):
        return random.choice(safe)   # 有安全棋就走安全棋（等於擋殺）

    # 3. 偏好中心、角落、邊
    for pref in (4, 0, 2, 6, 8, 1, 3, 5, 7):
        if pref in moves:
            return pref
    return random.choice(moves)
```

### 6.4 練習

寫一個 `get_basic_move_ultimate(game)` 給終極井字棋用（提示：試走邏輯完全一樣，只是着法是 `(macro, micro)` 元組；第 3 步的偏好改成"中央大格 4、中央小格 4、角落大格"——對照項目 `get_basic_move` ai.py 第 96-136 行）。

---

## 第 7 章：Minimax —— 窮舉搜索（對應 ai.py 第 138 行起）

### 7.1 核心思想：雙方都完美時會怎樣？

假設雙方永遠選"對自己最好"的棋。那麼：

> 輪到 X 時，X 選**讓 X 得分最高**的着法；
> 輪到 O 時，O 選**讓 X 得分最低**的着法（因爲 O 的立場和 X 相反）。

每一層的"價值"從終局往回傳：X 層取最大值，O 層取最小值 → **Minimax（極小化極大）**。

### 7.2 先學遞歸

```python
def countdown(n):
    if n <= 0:            # 基線條件：停止遞歸
        print("發射！")
        return
    print(n)
    countdown(n - 1)      # 調用自己
```

遞歸 = 函數調用自己。**必須**有基線條件（base case）停下來，否則無限遞歸導致程序崩潰（RecursionError）。

### 7.3 Minimax 代碼

```python
import math

def minimax(game, maximizing, ai_player, depth=0):
    r = game.result()
    if r is not None:                       # 基線條件：遊戲結束了
        if r == 'D':
            return 0
        # 分數從 AI 視角算；贏越快分越高（離終局越近越好）
        return (100 - depth) if r == ai_player else (depth - 100)

    if maximizing:
        best = -math.inf                    # 初始化爲負無窮
        for m in game.legal_moves():
            g = game.clone()
            apply_move(g, m)
            best = max(best, minimax(g, False, ai_player, depth + 1))
        return best
    else:
        best = math.inf                     # 正無窮
        for m in game.legal_moves():
            g = game.clone()
            apply_move(g, m)
            best = min(best, minimax(g, True, ai_player, depth + 1))
        return best
```

頂層這樣選棋：

```python
def minimax_move_normal(game):
    player = game.current
    best_moves, best_score = [], -math.inf
    for m in game.legal_moves():
        g = game.clone()
        apply_move(g, m)
        score = minimax(g, False, player)   # 我走完後輪到對方（min 層）
        if score > best_score:
            best_score, best_moves = score, [m]
        elif score == best_score:
            best_moves.append(m)
    return random.choice(best_moves), best_score
```

### 7.4 搜索樹長什麼樣

```
          根（輪到我，max）
         /       |        \
    對手(min)  對手(min)  對手(min)
      /  \       /  \       /  \
    終局 終局   終局 終局   終局 終局
```

分數從最底層一層層傳上來。井字棋全樹只有 9! ≈ 36 萬節點，電腦瞬間算完 → **完美棋手，永遠不會輸**。

### 7.5 Alpha-Beta 剪枝（提速但不改變結果）

如果已找到一條"保證 ≥ 5 分"的路，而另一條分支對手能把我壓到 ≤ 3 分，那後一條分支不用繼續看了：

```python
def minimax(game, alpha, beta, maximizing, ai_player, depth=0):
    r = game.result()
    if r is not None:
        if r == 'D':
            return 0
        return (100 - depth) if r == ai_player else (depth - 100)
    if maximizing:
        best = -math.inf
        for m in game.legal_moves():
            g = game.clone()
            apply_move(g, m)
            best = max(best, minimax(g, alpha, beta, False, ai_player, depth + 1))
            alpha = max(alpha, best)
            if beta <= alpha:      # 對手不會讓我走到這 → 剪！
                break
        return best
    else:
        best = math.inf
        for m in game.legal_moves():
            g = game.clone()
            apply_move(g, m)
            best = min(best, minimax(g, alpha, beta, True, ai_player, depth + 1))
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best
```

- `alpha`：我方目前能保證的最低分（只會漲）
- `beta`：對手能容忍的最高分（只會降）
- `beta <= alpha` 時，繼續搜沒意義 → break
- **剪枝不改變結果**：跳過的都是"不可能改變根節點決定"的分支，答案和完整搜索一模一樣

**剪枝到底省了多少？**（空盤、普通 3×3，本機實測）：

| 版本 | 要評估的局面數 |
|---|---|
| 無剪枝 minimax | 549,946 |
| alpha-beta | **20,866（省 96.2%）** |

**走法排序纔是關鍵**：剪得多不多，取決於"先試哪個走法"。先試"最可能好"
的走法，alpha/beta 很快收斂到正確值，後面的分支大量被剪。項目的
`minimax_move_normal` 用 `sorted(legal, key=lambda i: i != 4)` 把**中心放最前**試——
這就是爲什麼同樣約 36 萬局面，電腦還是瞬間算完。
（第 9 章的 Killer Move / LMR，本質也是"更好地排序走法 → 更多剪枝"。）

### 7.6 終極井字棋怎麼辦？—— 評估函數

終極棋盤有 81 格，窮舉不完，只能搜固定深度（比如 3 層），然後用**評估函數**給中間局面打分。對應項目 `eval_ultimate`（ai.py 第 179 行）：

```python
def eval_ultimate(game, player):
    score = 0
    # 已佔領的大格：        +1000 / -1000（決定性）
    # 小棋盤裏的 fork 威脅：  +40 / -40
    # 小棋盤 2 連 + 1 空：    +3 / -3
    # 小棋盤裏已有幾個子：    +0.5 / -0.5
    # 大棋盤 2 連 + 1 空：    +20 / -20
    return score
```

評估函數的精髓：**把"感覺"量化**。哪些局面特徵重要、各值多少分，靠經驗調整。搜到深度限制時，就用這個分數代替終局結果。

### 7.7 對照項目源碼

- 普通模式：`_minimax_normal` + `minimax_move_normal`（ai.py 第 138-176 行）
- 終極模式：`_minimax_ultimate` + `minimax_move_ultimate`（第 234-283 行），還有動態深度（着法少時多搜一層）
- 進階版 **Minimax Pro**：negamax + 置換表 + 迭代加深 + killer move + LMR + aspiration（第 1057-1155 行）——先跳過，第 9 章介紹思想

---

## 第 8 章：MCTS —— 蒙特卡洛樹搜索（對應 ai.py 第 285 行起）

### 8.1 和 Minimax 的區別

Minimax 要**精確算清**每個局面。MCTS 反其道而行：**隨機模擬幾千盤**，用統計結果判斷哪步好。就像用投票代替精確計算——所以叫"蒙特卡洛"（賭城名字，代表隨機）。

### 8.2 四個步驟

每次迭代做四件事：

1. **選擇（Selection）**：從根出發，用 UCB1 公式挑最有"前途"的子節點一路往下走
2. **擴展（Expansion）**：走到還沒展開的節點，加一個新孩子
3. **模擬（Simulation / Rollout）**：從那裏開始**隨機下到底**
4. **回傳（Backpropagation）**：把結果沿路加給每個祖先節點

### 8.3 節點

```python
class MCTSNode:
    def __init__(self, state, move, parent, mover):
        self.state = state        # 棋盤快照
        self.move = move          # 從父節點走到我這步棋
        self.parent = parent
        self.children = []
        self.mover = mover        # 這一步是誰下的（計分要用）
        self.visits = 0           # 被訪問多少次
        self.wins = 0.0           # 贏了多少次（平局算 0.5）
        self.untried = state.legal_moves()  # 還沒展開的着法
        random.shuffle(self.untried)
```

### 8.4 UCB1 公式 —— 選擇的關鍵

```
UCB = 勝率 + c × sqrt( ln(父節點訪問次數) / 自己訪問次數 )
      ↑利用             ↑探索
```

- 第一項：**利用**——勝率高的優先
- 第二項：**探索**——訪問少的優先（分母小 → 值大，訪問多了自然變小）
- `c` 是探索係數，項目默認 1.4

兩股力量平衡：既不要只走"目前看起來最好"的（可能被坑），也不要平均主義亂試。

```python
    def best_child(self, c):
        log_n = math.log(max(1, self.visits))
        best, best_val = None, -math.inf
        for child in self.children:
            if child.visits == 0:
                uct = math.inf        # 沒訪問過 = 無窮大 → 一定優先試
            else:
                uct = (child.wins / child.visits
                       + c * math.sqrt(log_n / child.visits))
            if uct > best_val:
                best, best_val = child, uct
        return best
```

### 8.5 完整搜索

```python
def mcts_search(game, iterations, c=1.4):
    root = MCTSNode(game.clone(), None, None, None)

    for _ in range(iterations):
        # ---- 選擇：一路選到葉子 ----
        node = root
        state = root.state.clone()
        while node.children and not node.untried:
            node = node.best_child(c)
            apply_move(state, node.move)

        # ---- 擴展：加一個孩子 ----
        if node.untried:
            m = node.untried.pop()
            mover = state.current
            apply_move(state, m)
            child = MCTSNode(state.clone(), m, node, mover)
            node.children.append(child)
            node = child

        # ---- 模擬：隨機下到底 ----
        result = state.result()
        guard = 0
        while result is None and guard < 300:
            moves = state.legal_moves()
            if not moves:
                break
            apply_move(state, random.choice(moves))
            result = state.result()
            guard += 1   # 保險絲，防止意外死循環

        # ---- 回傳 ----
        while node is not None:
            node.visits += 1
            if result == node.mover:
                node.wins += 1.0
            elif result == 'D':
                node.wins += 0.5
            node = node.parent

    return root
```

### 8.6 從根選棋

```python
def best_mcts_move(root):
    visited = [c for c in root.children if c.visits > 0]
    return max(visited, key=lambda c: c.wins / c.visits).move
```

選棋策略有兩種：
- **最大訪問數**：統計樣本最多的最可信（項目用這個 + 細微修正：優先立即獲勝的、只考慮訪問夠多的孩子）
- **最高勝率**：樣本少時不可靠

### 8.7 爲什麼 MCTS 對終極井字棋有效？

終極棋盤第一步有 81 個選擇，Minimax 精確搜索會爆炸。MCTS 的模擬天然"專注"：好的分支訪問多、統計準；壞分支自動被冷落。迭代次數就是"思考時間"，可隨時調整（項目默認 1600，見 sba.toml）。

### 8.8 項目裏的增強（先認識名字）

- **啓發式 rollout**（ai.py `_rollout_move` 第 469 行）：模擬不用純隨機，而是"能贏就贏、能擋就擋"——模擬質量大幅提高
- **動態 UCT**（`_uct_scale` 第 567 行）：前期多探索、後期多利用
- **樹重用**（`_REUSE` 第 574 行）：對手走完一步後，保留舊樹裏對應分支，不用從頭搜
- **RAVE/GRAVE**（第 734、850 行）：把"這步棋在這次模擬裏出現的所有棋"也計入統計，信息共享、收斂更快
- **早停**：某步優勢巨大時提前結束搜索

### 8.9 實例演練：手工走一遍 MCTS（3×3 棋盤）

光看代碼容易暈，跟着走一遍就懂了。假設局面（普通井字棋，輪到 X）：

```
X | O | _
--+---+--
_ | X | _
--+---+--
_ | _ | _
```

索引 0=X、1=O、4=X，所以 X 的合法着法 = `2, 3, 5, 6, 7, 8`。

**第 1 次迭代**
1. **選擇**：根節點剛建好、沒有孩子 → 跳過。
2. **展開**：從 `untried` 隨機彈出一個着法，假設是 5，創建子節點。
3. **模擬**：從"X 走 5 之後"開始隨機下到底。假設**這次隨機對局 X 贏了**。
4. **回傳**：根和 5 號子節點 `visits += 1`；這步是 X 下的且 X 贏 → `wins += 1`。

```
根: visits=1
└─ 着法5: visits=1, wins=1.0   ← 勝率 100%
```

**第 2 次迭代**：根還有沒展開的着法（2、3、6、7、8），所以繼續展開，假設隨機彈到 2。
從"X 走 2"模擬，**這次 O 贏了** → 2 號節點（X 下的、沒贏）`wins += 0`。

```
根: visits=2
├─ 着法5: visits=1, wins=1.0
├─ 着法2: visits=1, wins=0.0
└─ 着法3,6,7,8: 還沒展開
```

**第 3 次迭代**：再展開一個，假設是 6，模擬結果是**平局** → `wins += 0.5`。

```
根: visits=3
├─ 着法5: visits=1, wins=1.0
├─ 着法2: visits=1, wins=0.0
├─ 着法6: visits=1, wins=0.5
└─ 着法3,7,8: 還沒展開
```

**第 4 次迭代**：根的所有着法都展開了 → **選擇階段終於登場**，用 UCB 公式挑孩子。
根的 `visits=3`，`ln(3) ≈ 1.10`，`c = 1.4`：

- 着法 5：`1.0 + 1.4 × √(1.10/1) ≈ 2.47`
- 着法 2：`0.0 + 1.4 × √(1.10/1) ≈ 1.47`
- 着法 6：`0.5 + 1.4 × √(1.10/1) ≈ 1.97`
- 着法 3、7、8：**沒訪問過 → UCB = 無窮大，一定先試**（假設隨機選到 3）

選中 3 後在它的子樹裏展開、模擬（假設 X 又贏）、回傳。最終：

```
根: visits=4
├─ 着法5: visits=1, wins=1.0
├─ 着法2: visits=1, wins=0.0
├─ 着法6: visits=1, wins=0.5
├─ 着法3: visits=1, wins=1.0
└─ 着法7,8: 還沒展開
```

看出規律了嗎？**沒訪問過的孩子永遠優先被試一次**（UCB=∞），試過之後：
勝率高的會被越來越常選（利用），訪問少的會補回來（探索）——這就是整個 MCTS 的靈魂。
真實項目把"隨機下到底"換成"能贏就贏、能擋就擋"（`_rollout_move`），統計收斂更快；
迭代次數是幾百到幾千，手工走 4 次只是爲了看懂機制。

### 8.10 引擎對比表（一頁看懂）

| 引擎 | 一句話原理 | 普通 3×3 | 終極 9×9 | 思考時間 |
|---|---|---|---|---|
| Random | 合法着法裏隨機挑 | 很弱 | 很弱 | 0 |
| Basic | 贏 → 擋 → 中心/角落 | 新手級 | 新手級 | 0 |
| Minimax | 窮舉 + alpha-beta | **完美（永不輸）** | 深度受限 + 評估函數 | 秒級 |
| Minimax Pro | negamax + 置換表 + 剪枝 | 完美 | 與 Minimax 持平 | 可控 |
| MCTS | 隨機模擬幾千盤做統計 | 強 | **實戰首選** | 迭代數 × 毫秒 |

記不住也沒關係：**普通棋盤用 Minimax、終極棋盤用 MCTS**，這就是項目的默認推薦。

---

## 第 9 章：優化技巧（進階，可選）

### 9.1 Bitboard（位棋盤）

棋盤狀態用**二進制位**存：每個格子一位，1 = 有子。81 格 → 兩個 81 位整數（X 一個、O 一個）。

好處：
- `clone()` 只是複製兩個整數，極快
- 判斷一條線是否連成：`(mask & line) == line` 一次位運算搞定

```python
bit = 1 << (m * 9 + i)               # 第 (m,i) 格對應哪一位
g.x |= bit                           # 落一個 X 子（|= 是"按位或並賦值"）
taken = (g.x | g.o) >> (m * 9) & 0x1FF   # 取大格 m 的 9 位狀態
```

項目裏就是 `BitUltimateGame`（game.py 第 142 行），接口和列表版完全一樣——MCTS 內部用位版搜，搜完把着法還給 UI。

#### 9.1.1 位運算入門（30 秒速成）

```python
a = 0b1010          # 二進制寫法：1×8 + 0×4 + 1×2 + 0×1 = 10
a << 1              # 左移一位 = 乘 2 → 0b10100 = 20
1 << 3              # = 8：把"第 3 位"變成 1 的常用寫法
mask = 0b111        # 三個位都是 1
a & mask            # 按位與：兩邊都是 1 才保留 → 0b0010 = 2
a | (1 << 0)        # 按位或：把第 0 位設成 1
(a >> 1) & 1        # 右移再取末位 = 取出"第 1 位"是幾
```

只用記住 4 個：`<<`（左移）、`|`（設成 1）、`&`（檢查/過濾）、`>>`（右移）。

#### 9.1.2 BitUltimateGame 怎麼用位存棋盤

- X 的 81 個格子 → 一個 81 位整數 `self.x`（第 `m*9+i` 位 = 大格 m、小格 i）
- O 的 81 個格子 → `self.o`
- 大格誰贏 → `self.mx` / `self.mo`（各 9 位）

判斷"大格 m 的第 i 格有沒有子"：

```python
bit = 1 << (m * 9 + i)
taken = (self.x | self.o) & bit   # 非 0 表示這個格已被佔用
```

#### 9.1.3 勝利判斷：一次位運算搞定

一條線（例如大格 0 的 `(0,1,2)`）對應二進制 `0b111000000`（第 0–2 位都是 1）：

```python
def is_line_win(x_mask, line):
    return (x_mask & line) == line   # X 的位蓋住整條線 = 贏了
```

列表版要遍歷 3 個格子、比較 3 次；位版一次 `&` + 一次 `==`，這就是"快"的來源。

#### 9.1.4 一個誠實的結論（benchmark 文化）

位板**理論上**快很多（clone 只複製 4 個整數、勝利判斷常數時間）。
但項目實測（`docs/decisions.md` D12）：MCTS 已經用啓發式 rollout 之後，
位板**沒有帶來額外可測的加速**，所以它保留下來主要因爲接口統一、省內存、
值得學習——而不是"快 10 倍"。`sba.toml` 裏 `bitboard = true` 開着，但**不要憑感覺優化，要量**。

### 9.2 對象池（NodePool）

MCTS 每局建幾千個節點，頻繁創建/銷燬對象拖慢速度（垃圾回收壓力）。對象池 = 用完不銷燬，回收複用：

```python
class NodePool:
    def __init__(self):
        self._free = []         # 空閒節點倉庫

    def acquire(self, state, move, parent, mover):
        if self._free:
            n = self._free.pop()     # 複用舊對象，重置字段
            n.state = state
            n.move = move
            # ... 重置其他字段
            return n
        return MCTSNode(state, move, parent, mover)

    def release(self, root):
        # 遍歷整棵樹，把節點放回倉庫
        ...
```

對應項目 `NodePool`（ai.py 第 312 行）。

### 9.3 置換表（Transposition Table）

不同着法順序可能到達**同一個局面**（比如先走 0 再走 4，和先走 4 再走 0）。Minimax 會重複計算這些局面。置換表 = 字典：`局面 → 已算好的分數`，遇到重複局面直接查表。對應項目 `_negamax_tt`（ai.py 第 1057 行），還有 `tt_max` 限制表大小防止內存爆炸。

### 9.4 開局書（Opening Book）

開局不需要算，直接查表。項目 `ULTIMATE_BOOK`（ai.py 第 1181 行）：X 第一步走哪裏 → O 應該回哪裏，人工挑選的好應手，省下開局的大量搜索時間。配置化後放在 sba.toml 的 `[engine.opening_book_ultimate]` 段。

### 9.5 搜索工程技巧全家桶（Minimax Pro 用了全套）

- **迭代加深**：先淺搜拿個大致結果，再逐層加深；超時就用上次結果
- **Killer Move**：記住"上次在這裏引發剪枝的棋"，下次優先試
- **LMR（Late Move Reduction）**：排序靠後的着法多半差，用較淺深度搜
- **Aspiration Window**：猜一個分數區間搜，猜錯再放寬

這些是國際象棋引擎的標準套路。井字棋項目裏是"殺雞用牛刀"，但對學習極有價值——這就是項目 ai.py 第 1057-1155 行的 Minimax Pro。

---

## 第 10 章：配置文件（對應 sba.toml）

### 10.1 爲什麼要配置

把"探索係數多大、MCTS 迭代幾次"寫死在代碼裏，每次調都要改代碼。放進配置文件，改完重啓就行。TOML 是簡單易懂的配置格式：

```toml
[engine]
uct_c = 1.4
rollout_heuristic = true

[session]
mcts_budget = 1600
```

### 10.2 讀取（Python 3.11+ 內置 tomllib）

```python
import tomllib

def load_config():
    try:
        with open('sba.toml', 'rb') as fh:
            return tomllib.load(fh)     # → 嵌套 dict
    except Exception:
        return {}                       # 文件沒了就空 dict
```

`with open(...)` 是標準文件打開方式，用完自動關閉。`'rb'` = 二進制讀模式（tomllib 要求）。

### 10.3 默認值回退（重要模式）

```python
DEFAULTS = {'engine': {'uct_c': 1.4, 'rollout_heuristic': True}}

def cfg_engine(name, default=None):
    # 先看配置文件，沒有再看代碼默認值，再沒有用參數默認值
    return (CONFIG.get('engine', {})
            .get(name, DEFAULTS['engine'].get(name, default)))
```

用戶少寫一個鍵，程序也能跑——這就是項目 README 說的 "Missing keys fall back to code defaults"。對應 ai.py 第 40-89 行。

### 10.4 測試時重置

`set_engine_config(overrides)`（ai.py 第 71 行）允許測試把配置改掉再還原，保證測試不受環境文件影響。這是個好習慣。

---

## 第 11 章：圖形界面（PySide6 教學——項目現用 Textual TUI）

> 注：本項目目前的實際界面是 **Textual 終端界面**（`python SBA.py`）。
> 本章用 PySide6 講 Qt 的核心概念（窗口、佈局、信號槽、事件循環、線程），
> 這些概念對任何 GUI 框架都通用，當作入門素材。

### 11.1 安裝與核心概念

```bash
pip install pyside6
```

Qt 核心概念：

- **QWidget**：一切界面元素的基類（窗口、按鈕、面板）
- **佈局（Layout）**：自動排列子控件（`QHBoxLayout` 橫排 / `QVBoxLayout` 豎排 / `QGridLayout` 網格）
- **信號與槽（Signal & Slot）**：按鈕被點擊 → 觸發你的函數。這是 Qt 的靈魂

### 11.2 最小窗口

```python
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton

app = QApplication(sys.argv)      # 必須有，且整個程序只能有一個
win = QWidget()
win.setWindowTitle("我的井字棋")
layout = QVBoxLayout(win)         # 豎排佈局

label = QLabel("歡迎！")
button = QPushButton("點我")
button.clicked.connect(lambda: label.setText("你點了我！"))

layout.addWidget(label)
layout.addWidget(button)
win.show()
sys.exit(app.exec())              # 進入事件循環：程序在這裏"活着"
```

關鍵點：`app.exec()` 之前只是**搭建**界面；`exec()` 進入**事件循環**，之後程序靠信號槽響應（事件驅動）。按鈕被點擊時，Qt 自動調用你 connect 的函數。

### 11.3 用按鈕做 3×3 棋盤

最直觀：9 個按鈕放進 QGridLayout：

```python
from PySide6.QtWidgets import QGridLayout

grid = QGridLayout()
buttons = []
for i in range(9):
    b = QPushButton("")
    b.setFixedSize(80, 80)
    b.clicked.connect(lambda _=False, idx=i: on_click(idx))
    grid.addWidget(b, i // 3, i % 3)   # 行 = i//3, 列 = i%3
    buttons.append(b)

def on_click(idx):
    if game.board[idx] != EMPTY:
        return
    game.make_move(idx)
    buttons[idx].setText(game.board[idx])
    # 判斷勝負、輪到 AI 時調 get_ai_move...
```

### 11.4 lambda 閉包陷阱 ⚠️

上面 `lambda _=False, idx=i: ...` 裏的 `idx=i` **必須寫**。如果寫成 `lambda: on_click(i)`，所有按鈕都會用**循環結束後**的 i（=8）——因爲 lambda 捕獲的是變量本身，不是當時的值。用默認參數 `idx=i` 把值"定"在創建那一刻。這是 Qt 新手最經典的 bug。

（`_=False` 是給信號默認傳的 checked 參數佔位，可以不理它。）

### 11.5 自定義繪製棋盤（項目同款，進階）

按鈕做 81 格終極棋盤太重。項目用 **QPainter** 在 QWidget 上自己畫：`BoardWidget` 重寫 `paintEvent` 畫格線、棋子、高亮；重寫 `mousePressEvent` 處理點擊。核心結構：

```python
class BoardWidget(QWidget):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # 抗鋸齒
        # 畫格線、X（兩條線）、O（圓）、當前大格高亮...
        painter.end()

    def mousePressEvent(self, event):
        # 根據 event.position() 算出點中了哪一格
        # → game.make_move(...) → self.update() 觸發重繪
```

要點：Qt 不會"自動重畫"，狀態變了要調 `self.update()`，Qt 隨後調你的 `paintEvent`。

### 11.6 AI 不卡界面的關鍵：線程

MCTS 想一步可能要 1-2 秒。如果在主線程（UI 線程）裏同步計算，界面會**凍結**（點了沒反應）。項目把 AI 放進工作線程，算完發信號通知 UI：

```python
class AIWorker(QThread):
    finished_move = Signal(object)      # 自定義信號：算完發出

    def run(self):
        move = get_ai_move(self.game, self.ai_type, ...)
        self.finished_move.emit(move)   # 通知主線程
```

`Signal` 是跨線程安全的消息通道——Qt 保證槽函數在主線程執行，可以放心更新界面。

### 11.7 對照項目源碼

項目的終端界面 `tui.py`（Textual）有 1000+ 行，包括：自定義棋盤繪製、AI 助手面板、歷史與控制、CvC 控制；此前的桌面版 `qtui.py`（PySide6 + SiliconUI）已被取代。作爲新手，先實現：棋盤繪製 + 點擊落子 + AI 線程 + 菜單，其他逐步加。

---

## 第 12 章：測試（pytest）

### 12.1 爲什麼要測試

改代碼後，"棋盤還能不能正常贏？AI 還會不會下非法棋？"手動點太慢。測試 = 把檢查自動化，每次跑一遍立刻知道有沒有改壞。

### 12.2 第一個測試

```python
# tests/test_game.py
from game import NormalGame, X, O

def test_win_by_row():
    g = NormalGame()
    g.make_move(0)  # X
    g.make_move(3)  # O
    g.make_move(1)  # X
    g.make_move(4)  # O
    g.make_move(2)  # X → 第一行連成
    assert g.result() == X     # assert = "我斷言這是真的，否則測試失敗"
```

運行：

```bash
pip install pytest
pytest
```

綠了（PASSED）= 沒壞；紅了（FAILED）= 精確告訴你哪行斷言的哪個值不對。

### 12.3 測試要覆蓋什麼

對照項目 `tests/`：

- `test_game.py`：規則——勝負、平局、路由規則、clone 獨立性、非法着法報錯
- `test_ai.py`：AI——每個引擎都走合法棋、能贏時 Basic 會贏、Minimax 完美不敗
- `test_alphazero.py`：訓練系統的冒煙測試（小規模快速跑通）
- `conftest.py`：共享的測試工具

**新手鐵律**：每次改完核心邏輯，跑一遍 `pytest`。

---

## 第 13 章：AlphaZero 概念入門

### 13.1 一句話版本

AlphaZero = **神經網絡 + MCTS**。神經網絡學習"哪個局面好、哪步棋可能好"，替代 MCTS 裏的隨機模擬和人工規則，讓搜索又快又準。2017 年 DeepMind 用它打敗了所有圍棋、國際象棋 AI。

### 13.2 雙頭網絡

輸入：當前棋盤（編碼成數字張量）。輸出兩樣：

- **策略頭（policy head）**：每步棋的"先驗概率"——網絡覺得哪步值得下
- **價值頭（value head）**：當前局面對誰的勝率——替代隨機模擬到終局

對應項目 `AZNet.forward`（alphazero.py 第 157 行）：`return p, torch.tanh(self.val_fc2(v))`——策略概率 + 一個 -1~1 的價值。

### 13.3 神經網絡怎麼"看"棋盤

棋盤被編碼成 4 層 9×9 矩陣（`encode`，alphazero.py 第 81 行）：

```
第 0 層：X 的棋子位置（1.0 / 0.0）
第 1 層：O 的棋子位置
第 2 層：輪到誰
第 3 層：當前必須下的大格
```

就像給網絡 4 張"透明膠片"疊起來。然後過幾層**卷積**（conv2d，圖像識別核心技術，自動提取"這裏有兩個連子、那裏有威脅"之類的特徵），最後輸出兩個頭。

### 13.4 訓練 = 自己和自己下（自對弈）

```
1. 用當前網絡 + MCTS 自己和自己下一盤（每步記錄：局面 → MCTS 算出的落子分佈）
2. 終局後，每個局面標上真實結果（贏 +1，輸 -1，平 0）
3. 把"局面 → (落子分佈, 結果)"當教材，訓練網絡逼近 MCTS 的判斷
4. 循環幾千盤，網絡越來越強，MCTS 也越搜越準
```

對應 `self_play_game`（alphazero.py 第 407 行）和 `train`（第 471 行）。

### 13.5 MCTS 裏怎麼用網絡

- **模擬那步直接砍掉**：價值頭直接給出局面分數
- **選擇時**：UCB 公式裏的"先驗"用策略頭概率（`priors`，alphazero.py 第 331 行）
- **根節點加 Dirichlet 噪聲**：強制開局探索，防止網絡死記一條路

對應 `mcts_search`（alphazero.py 第 244 行）——結構和第 8 章我們寫的 MCTS 一模一樣，只是"模擬"換成"問網絡"。

### 13.6 需要什麼基礎

PyTorch 是深度學習框架。建議先補：線性代數入門（矩陣）、微積分入門（梯度下降）。好消息：**用網絡不需要懂內部數學**，torch 幫你算一切。

項目裏的技術報告 `docs/AlphaZero_UTTT_Technical_Report.html` 有訓練數據詳細分析，等你能看懂網絡結構後值得一讀。

---

---

## 第 14 章：Git 版本控制（交作業前必讀）

### 14.1 爲什麼需要 Git

沒有 Git 的"存檔"是複製文件夾（`項目_final_真最終版.py`）——遲早會亂。
Git 幫你：

- **隨時回退**：每次提交（commit）都是一個檢查點，改壞了回到上一個
- **看清楚改了什麼**：`git diff` 顯示每一行改動
- **多人協作 / 換電腦**：把倉庫推到 GitHub，到處都能拉下來
- **面試驗證**：`git log` 是你的"開發日記"，面試官愛看

### 14.2 三個區域（先懂概念）

```
工作區（你正在改的文件夾）
   ↓ git add
暫存區（準備好了的改動）
   ↓ git commit
歷史（一串提交，每個都能回退）
```

### 14.3 新手先記這 7 條指令

```bash
git init              # 在項目文件夾建立倉庫（只做一次）
git status            # 看現在改了什麼（最常用！）
git add 文件名         # 把改動放進暫存區；git add . 表示全部
git commit -m "feat: 加入 MCTS 引擎"   # 存一個檢查點
git log --oneline     # 看提交歷史（一行一條）
git diff              # 看改動的具體內容
git checkout -- 文件名  # 丟棄某個文件的改動（回退）
```

### 14.4 推到 GitHub

```bash
git remote add origin https://github.com/你的賬號/項目名.git
git push -u origin main    # 第一次推送
git pull                   # 之後每次開工先拉最新
```

### 14.5 .gitignore：別把垃圾傳上去

`.venv/`（幾百 MB 的依賴）、`__pycache__/`（緩存）、模型權重、密鑰——
這些不該進倉庫。在項目根目錄建 `.gitignore` 寫上它們，git 就會自動忽略。
直接看本倉庫根目錄的 `.gitignore` 當範例。

### 14.6 提交信息怎麼寫

一句話說清"做了什麼"：`feat:` 新功能、`fix:` 修 bug、`docs:` 文檔。
例如項目真實歷史：`fix(ai): align workers default with sba.toml (8)`。

### 14.7 分支（先懂概念）

主分支（`main`）放"能跑的版本"，新功能在分支上開發，確認沒問題再合併。
改壞了不用怕——回到上一個提交就行。本項目真實用了 4 個分支，
`docs/decisions.md` 有記錄。

### 14.8 練習

把第 2 章的 `main.py` 提交 3 次：第一次"能跑"、第二次"加了防呆"、
第三次"加了平局提示"。每次 `git log --oneline` 看歷史越來越長。

---

## 第 15 章：虛擬環境 venv（跑項目前必讀）

### 15.1 爲什麼需要

`pip install` 默認裝到全局，兩個項目需要不同版本時會互相打架。
venv = 給每個項目一個**獨立的 Python 環境**：

```bash
python -m venv .venv          # 建立（Windows / Mac / Linux 一樣）
.venv\Scripts\activate        # 啓用（Windows PowerShell）
source .venv/bin/activate     # 啓用（Mac / Linux）
deactivate                    # 退出
```

啓用後命令行開頭會出現 `(.venv)`，這時候 `pip install` 只裝進這個項目。

### 15.2 requirements.txt：一鍵裝齊依賴

```bash
pip freeze > requirements.txt      # 導出當前依賴清單
pip install -r requirements.txt    # 別人（或新電腦）一鍵安裝
```

看本倉庫的 `requirements.txt`：就 3 行（nicegui / textual / pytest）——因爲大型依賴
（PyTorch、ROCm）體積大、版本挑環境，單獨手動裝。

### 15.3 三個常見坑

1. 忘了啓用 venv 就 `pip install` → 裝到全局（看命令行有沒有 `(.venv)`）
2. 沒把 `.venv/` 加進 `.gitignore` → 把幾百 MB 推上 GitHub
3. 換電腦跑項目 = `git clone` → `python -m venv .venv` → `pip install -r requirements.txt`

---

## 第 16 章：項目管理（把作業做成作品）

### 16.1 先寫 README

別人（評審、面試官）第一眼看的是 README。至少包含：
一句話介紹、怎麼安裝怎麼跑、功能列表、**截圖**。
本倉庫的 README 就是範例——中英雙語、帶用法和架構圖。

### 16.2 決策日誌（decision log）：記錄"爲什麼"

代碼只告訴你"做了什麼"，不告訴你"爲什麼這麼做"。
項目裏的 `docs/decisions.md`（D1–D18）就是範例：

- D12：bitboard 實測無增益 → 保留但默認關
- D18：多進程自對弈，8 worker 約 3× 吞吐

評審最喜歡這種**有證據的選擇**——比"我試了很多方法"有說服力一百倍。

### 16.3 用 benchmark 說話

不要說"我覺得 MCTS 比較強"，要說"**30 局 22 勝 4 平 4 負**"。
本項目的 `python SBA.py --bench --ai-a Minimax --ai-b MCTS --games 30`
就是幹這個的。數字不會說謊，也經得起追問。

### 16.4 測試讓你敢重構

每改完核心邏輯跑一次 `pytest`，全綠才提交。
有測試兜底，你纔敢放心拆代碼、改結構（第 12 章）。

### 16.5 一次只做一件事

流程：**功能 → 測試 → 提交 → 下一個**。壞了一個功能，
`git revert` 回退，不影響其他成果。這是本倉庫 57 個測試、
100+ 次提交背後的工作方式。

### 16.6 GitHub 當作品集

README 放截圖、錄一段 2–3 分鐘 demo 影片、把鏈接放進 CV。
這份教程 + 你的項目本身，就是比任何證書都有說服力的作品集。

---

## 附錄 A：新手常見錯誤對照表

| 症狀 | 原因 | 解法 |
| --- | --- | --- |
| `IndentationError` | 縮進不對（Tab 和空格混用） | 統一用 4 個空格 |
| `list index out of range` | 索引越界（第 9 格是 board[8]，沒有 board[9]） | 檢查索引範圍 |
| 改一個列表，另一個也跟着變 | 淺拷貝 `b = a` | 用 `a[:]` 或 `.copy()` |
| `NoneType has no attribute ...` | 函數沒 return，拿到 None 還調方法 | 檢查 return |
| 按鈕全都觸發同一個序號 | lambda 閉包陷阱 | 用默認參數綁值 `idx=i` |
| 界面點了沒反應 / 卡死 | 在主線程算 AI | 用 QThread |
| 中文亂碼 | 控制檯編碼不是 UTF-8 | `chcp 65001` |
| `ModuleNotFoundError` | 沒裝包 或 不在文件目錄 | pip install / cd 到項目目錄 |
| `RecursionError` | 遞歸沒有基線條件 | 檢查 base case |
| 二維列表 `[['']*9]*9` 改一個全改 | 9 個引用指向同一個列表 | `[['']*9 for _ in range(9)]` |

## 附錄 B：動手路線建議

1. 第 2 章完成後：加"悔棋"功能（用列表記錄每步歷史）
2. 第 5 章完成後：寫 Random vs Basic 打 100 盤統計勝率
3. 第 7 章完成後：驗證 Minimax 對 Random 100% 不敗
4. 第 8 章完成後：MCTS 用 50 / 200 / 800 次迭代互相打，看強度差多少
5. 加 UI 時先畫棋盤，再接線，別一次寫完

## 附錄 C：推薦資源

- Python 官方教程（中文）：<https://docs.python.org/zh-cn/3/tutorial/>
- 菜鳥教程 Python：<https://www.runoob.com/python3/>
- Minimax 可視化：Google 搜 "tic tac toe minimax visualization"
- MCTS 綜述論文：Browne et al. 2012《A Survey of Monte Carlo Tree Search Methods》
- AlphaZero 論文：Silver et al. 2017《Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm》
- PySide6 官方文檔：<https://doc.qt.io/qtforpython-6/>
- 本項目技術報告：`docs/AlphaZero_UTTT_Technical_Report.html`

## 附錄 D：術語表（一頁速查）

| 術語 | 一句話解釋 |
|---|---|
| 模塊 module | 一個 `.py` 文件 |
| 淺拷貝 / 深拷貝 | 只複製外殼 / 連內容一起復制 |
| clone | 複製一個獨立棋盤（AI 試走用） |
| 遞歸 recursion | 函數調用自己，必須有基線條件停下 |
| 基線條件 base case | 遞歸停止的條件 |
| alpha-beta 剪枝 | 提前砍掉"不可能改變結果"的分支 |
| 置換表 transposition table | "局面 → 分數"的緩存字典 |
| UCB / UCT | MCTS 選孩子的公式：勝率 + 探索項 |
| rollout / 模擬 | 從某局面隨機下到終局的一次"抽籤" |
| 回傳 backpropagation | 把模擬結果加回路徑上每個節點 |
| 啓發式 heuristic | 靠經驗規則，而非精確計算 |
| 評估函數 evaluation | 給未結束的局面打分 |
| 位棋盤 bitboard | 用整數位存棋盤，又快又省內存 |
| 開局書 opening book | 開局直接查表，不搜索 |
| 殘局表 tablebase | 終局附近的精確解 |
| 神經網絡 | 一堆可學習參數的"函數" |
| 策略頭 / 價值頭 | 輸出"每步棋概率" / "局面勝率"的兩個網絡頭 |
| 自對弈 self-play | 自己跟自己下棋產生訓練數據 |
| 損失函數 loss | 預測和真實差多少，越小越好 |
| 梯度下降 | 讓損失變小的參數更新方法 |

## 附錄 E：讀懂錯誤信息（除錯入門）

**第一步：讀最後一行。** Python 報錯時最重要的信息在最後：

```
ValueError: invalid literal for int() with base 10: 'a'
```

冒號前是**錯誤類型**（`ValueError` = 值錯誤），冒號後是**原因**（`'a'` 轉不成整數）。

**第二步：往上找 `File` 行。**

```
Traceback (most recent call last):
  File "main.py", line 7, in <module>
    move = int(input(...))
ValueError: invalid literal for int() with base 10: 'a'
```

`File "main.py", line 7` = 出錯的位置（文件 + 行號），下面那行就是出錯的代碼。

**第三步：問自己三個問題。**
1. 錯誤類型是什麼？（`ValueError` / `IndexError` / `TypeError` / `IndentationError`…）
2. 出錯在哪一行？（`line N`）
3. 那一行用到的東西，類型對嗎？值對嗎？（用 `print()` 打印出來看）

**最常見的三種：**

| 錯誤 | 意思 | 常見原因 |
|---|---|---|
| `IndentationError` | 縮進錯了 | Tab 和空格混用 |
| `IndexError: list index out of range` | 索引越界 | 第 9 格是 `board[8]`，沒有 `board[9]` |
| `TypeError: ... not supported between instances of ...` | 類型不匹配 | 字符串當數字用（如 `'2' + 3`） |

**萬能除錯法：`print()`。** 懷疑哪裏出錯，就在那行前後打印看看：

```python
print("move 的值是：", move)   # 看看變量到底是什麼
```

看完刪掉。這是新手最快的除錯方式；正式做法是用調試器（Thonny 點"步進"可以一行一行看）。

---

> 教程到這裏就結束了。整個項目的祕密就一句話：**從最簡單的能跑的東西開始，每次只加一個功能，跑通了再繼續。** 祝你玩得開心，加油！🚀
