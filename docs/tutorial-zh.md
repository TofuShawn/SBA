# 从零开始：用 Python 写一个井字棋项目（SBA 同款教程）

> 本教程面向**完全没有编程基础**的初学者。目标：带你一步步写出一个和本仓库 SBA 类似的项目——普通井字棋 + 终极井字棋，多种 AI 引擎（随机、启发式、Minimax、MCTS），再加上 PySide6 图形界面和 pytest 测试。
>
> 每一章的代码都可以直接运行，建议**边看边敲**（不要复制粘贴，手敲记忆更深）。
>
> 文中「对照项目源码」提到的**行号仅供参考**，代码演进后可能漂移；找不到就按函数名搜索（VS Code 按 `Ctrl+F`）。

---

## 目录

- [第 0 章：准备环境](#第-0-章准备环境)
- [第 1 章：Python 速成](#第-1-章python-速成)
- [第 2 章：文字版井字棋](#第-2-章文字版井字棋)
- [第 3 章：整理成类（对应项目 game.py）](#第-3-章整理成类对应项目-gamepy)
- [第 4 章：终极井字棋（对应项目 UltimateGame）](#第-4-章终极井字棋对应项目-ultimategame)
- [第 5 章：第一个 AI —— Random](#第-5-章第一个-ai--random)
- [第 6 章：Basic AI —— 启发式引擎](#第-6-章basic-ai--启发式引擎)
- [第 7 章：Minimax —— 穷举搜索](#第-7-章minimax--穷举搜索)
- [第 8 章：MCTS —— 蒙特卡洛树搜索](#第-8-章mcts--蒙特卡洛树搜索)
- [第 9 章：优化技巧（进阶）](#第-9-章优化技巧进阶)
- [第 10 章：配置文件（对应 sba.toml）](#第-10-章配置文件对应-sbatoml)
- [第 11 章：桌面图形界面（PySide6）](#第-11-章桌面图形界面pyside6)
- [第 12 章：测试（pytest）](#第-12-章测试pytest)
- [第 13 章：AlphaZero 概念入门](#第-13-章alphazero-概念入门)
- [第 14 章：Git 版本控制](#第-14-章git-版本控制)
- [第 15 章：虚拟环境 venv](#第-15-章虚拟环境-venv)
- [第 16 章：项目管理（把作业做成作品）](#第-16-章项目管理把作业做成作品)
- [附录 A：新手常见错误对照表](#附录-a新手常见错误对照表)
- [附录 B：动手路线建议](#附录-b动手路线建议)
- [附录 C：推荐资源](#附录-c推荐资源)

---

## 学习路线图

```
第0章 安装环境
  ↓
第1章 Python 速成（用到什么学什么）
  ↓
第2章 文字版井字棋（第一次写出能玩的游戏！🎉）
  ↓
第3章 整理成类 → 项目里的 game.py
  ↓
第4章 终极井字棋（9×9）
  ↓
第5章 Random AI ── 第6章 Basic AI ── 第7章 Minimax ── 第8章 MCTS
  ↓
第9章 优化技巧（bitboard、树重用、开局书…）
第10章 配置文件
  ↓
第11章 PySide6 图形界面
第12章 pytest 测试
  ↓
第13章 AlphaZero 概念
  ↓
第14章 Git ── 第15章 venv ── 第16章 项目管理（交作业前）
```

> 💡 **必读**：第 0–8 章 + 第 12 章（测试）+ 第 14 章（Git）+ 第 15 章（venv）。
> **可选**：第 9 章（优化技巧，知道名字即可）、第 10 章（配置）、第 11 章（界面，想动手再看）、第 13 章（AlphaZero，理解概念即可）、第 16 章（项目管理，交作业前读）。

**建议的项目结构**（先建一个自己的文件夹，别直接改仓库）：

```
my_tictactoe/
├── game.py       # 游戏规则（第 3、4 章）
├── ai.py         # AI 引擎（第 5-8 章）
├── main.py       # 文字版入口（第 2 章）
├── sba.toml      # 配置（第 10 章）
├── ui.py         # 图形界面（第 11 章）
└── tests/
    └── test_game.py   # 测试（第 12 章）
```

---

## 第 0 章：准备环境

### 0.1 安装 Python

1. 打开 <https://www.python.org/downloads/>
2. 下载最新版 Python 3.12+（Windows 用户点黄色 "Download Python 3.x.x" 按钮）
3. 安装时**务必勾选 "Add Python to PATH"**——这个勾非常重要，勾了之后命令行里才能用 `python` 命令
4. 一路点 Install 完成

### 0.2 验证安装

打开命令行（Windows 按 `Win+R`，输入 `cmd` 回车）：

```
python --version
```

应该输出类似 `Python 3.13.x`。如果提示"不是内部或外部命令"，说明 PATH 没勾上，重装一次。

### 0.3 运行第一个程序

新建文件 `hello.py`，内容：

```python
print("你好，世界！")
```

在命令行里 `cd` 到该文件所在目录，运行：

```
python hello.py
```

### 0.4 遇到中文乱码？

Windows 老式控制台默认不是 UTF-8 编码。两种解决办法：

- 运行前先输入 `chcp 65001` 切换编码
- 或者程序里暂时用英文 `print`（教程代码的打印信息都做了兼容）

### 0.5 编辑器推荐

- **新手最友好：Thonny**（自带逐步调试，能看每一行代码怎么跑）
- **主流：VS Code**（装官方 "Python" 扩展）
- **进阶 / 读大项目：PyCharm（Community 免费版）**——"跳到定义""查找引用"、图形化调试器、内建 git 都最强，适合读 SBA 这种几千行的项目；缺点是启动慢、吃内存，所以新手期先用上面两个。

---

## 第 1 章：Python 速成

只学做这个项目**用得到**的基础，不贪多。

### 1.1 变量

```python
name = "Alice"      # 字符串 str
age = 20            # 整数 int
pi = 3.14           # 小数 float
is_student = True   # 布尔 bool（True / False）
```

`#` 后面的内容是**注释**，Python 不执行，用来给人看。

### 1.2 列表（list）

列表是有顺序的一串东西：

```python
numbers = [1, 2, 3, 4, 5]
print(numbers[0])    # 1 —— 索引从 0 开始！
print(numbers[-1])   # 5 —— -1 表示最后一个
print(len(numbers))  # 5 —— 长度
```

常用操作：

```python
numbers.append(6)    # 末尾加一个 → [1,2,3,4,5,6]
numbers[0] = 99      # 修改第 0 个
3 in numbers         # True —— 判断是否在里面
```

**切片**（后面 clone 会用到，很重要）：

```python
a = [1, 2, 3]
b = a[:]     # 复制出一份**新**列表
b[0] = 999   # 改 b 不影响 a
print(a)     # [1, 2, 3] ✓
```

### 1.3 循环

```python
# for 循环：把每个元素过一遍
for i in range(9):        # range(9) = 0,1,2,...,8
    print(i)

for mark in ['', 'X', 'O']:
    print(mark)

# while 循环：条件为真就一直跑
count = 0
while count < 3:
    print(count)
    count += 1            # count = count + 1 的简写
```

### 1.4 条件判断

```python
age = 18
if age >= 18:
    print("成年")
elif age >= 6:
    print("儿童")
else:
    print("婴儿")
```

比较符号：`==`（相等）、`!=`（不等）、`<`、`>`、`<=`、`>=`。
**新手最容易搞混**：`=` 是赋值，`==` 是比较。判断相等必须用 `==`。

### 1.5 函数

函数 = 起个名字的一坨代码，可以反复调用：

```python
def add(a, b):
    return a + b

print(add(3, 5))   # 8
```

- `def` 定义函数
- `return` 把结果交回去
- 没有 return 的函数默认返回 `None`（表示"空"）

### 1.6 类（class）—— 面向对象

类 = 数据的"模板"。一个类可以创建很多**实例**，每个实例有自己的数据：

```python
class Dog:
    def __init__(self, name):   # 构造方法：创建实例时自动执行
        self.name = name        # self 指"这个实例自己"

    def bark(self):             # 方法：属于类的函数
        print(f"{self.name} 汪汪！")

d1 = Dog("小白")
d2 = Dog("大黄")
d1.bark()   # 小白 汪汪！
d2.bark()   # 大黄 汪汪！
```

- `__init__` 是构造方法，创建对象时自动运行，负责初始化数据
- 方法的第一个参数永远是 `self`
- `self.name` 是"这个实例自己的 name"，不同实例互不影响

**为什么 SBA 要用类？** 一局棋就是"一个棋盘实例"。AI 搜索时要复制很多个棋盘来做试验（"如果我在 4 落子会怎样？"），用类最方便。

### 1.7 导入模块

```python
import random                # Python 自带的随机模块
random.choice([1, 2, 3])     # 随机选一个 → 可能是 2
random.shuffle(xs)           # 原地打乱列表

from game import X, O        # 从自己写的 game.py 里导入
```

一个 `.py` 文件就是一个**模块**，别的文件可以 import 它。

### 1.8 元组（tuple）与字典（dict）—— 后面马上会用到

**元组**：和列表很像，但用 `()` 且**不能修改**。本项目的终极井字棋用 `(大格, 小格)` 表示一步棋：

```python
move = (4, 2)          # 在大格 4、小格 2 落子
macro, cell = move     # 拆包：macro=4, cell=2 —— 元组可以这样"拆"开
```

**字典**：键 → 值的对应表，用 `{}`：

```python
score = {'X': 1, 'O': -1}   # X 赢 +1，O 赢 -1
score['X']                  # 1 —— 按键取值
score.get('D', 0)           # 0 —— 没有 'D' 这个键时返回默认值 0
score['O'] = 5              # 修改或新增
'X' in score                # True —— 判断键是否存在
```

后面 AI 的"局面→分数"缓存（置换表）、配置文件（sba.toml）全是字典。

### 1.9 小练习

1. 写函数 `is_even(n)` 判断偶数
2. 写类 `Counter`，方法 `click()` 让计数 +1
3. 用 for 循环打印 1 到 10 的平方

---

## 第 2 章：文字版井字棋

现在开始做正事。先做**最简单、不用类**的版本，理解游戏是怎么"跑"起来的。

### 2.1 棋盘怎么表示？

3×3 棋盘有 9 个格子。用**长度为 9 的列表**表示，索引 0 到 8：

```
 0 | 1 | 2
---+---+---
 3 | 4 | 5
---+---+---
 6 | 7 | 8
```

每个元素是 `''`（空）、`'X'` 或 `'O'`：

```python
board = [''] * 9    # 全空棋盘
```

### 2.2 打印棋盘

```python
def print_board(board):
    # 空格子显示编号（方便玩家输入），有棋子的显示棋子
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

`f"..."` 是 **f-string**：大括号里的内容会被求值后填进字符串。

### 2.3 谁赢了？

8 条获胜线（3 行 + 3 列 + 2 条对角线），每条线写成三个索引的元组：

```python
LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # 三行
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # 三列
    (0, 4, 8), (2, 4, 6),              # 两条对角线
]

def winner(board):
    for a, b, c in LINES:
        if board[a] != '' and board[a] == board[b] == board[c]:
            return board[a]     # 'X' 或 'O'
    return None
```

**关键技巧**：`board[a] == board[b] == board[c]` 是**链式比较**，等价于"三者相等"。先判断 `board[a] != ''`，防止三个空格被判成赢家。

### 2.4 主循环

```python
board = [''] * 9
current = 'X'   # 当前轮到谁

while True:
    print_board(board)

    # 1. 玩家输入（带防呆：输错不崩溃）
    raw = input(f"轮到 {current}，输入格子编号 0-8：")
    if not raw.isdigit():        # 不是数字 → 重来
        print("请输入数字 0-8！")
        continue
    move = int(raw)
    if move < 0 or move > 8:     # 数字超范围 → 重来
        print("数字要在 0-8 之间！")
        continue
    if board[move] != '':
        print("这个格子已经有子了，换一个！")
        continue    # 回到循环开头，重新输入

    # 2. 落子
    board[move] = current

    # 3. 判断结束
    w = winner(board)
    if w:
        print_board(board)
        print(f"{w} 赢了！")
        break
    if '' not in board:     # 棋盘满了还没赢 → 平局
        print_board(board)
        print("平局！")
        break

    # 4. 换人
    current = 'O' if current == 'X' else 'X'
```

- `continue`：跳过本轮剩余代码，回到循环开头
- `break`：直接退出循环
- `'O' if current == 'X' else 'X'`：**三元表达式**，"如果 current 是 X 就取 O，否则取 X"

把 2.2、2.3、2.4 合在一起存成 `main.py`，跑一遍——你已经有一个能玩的文字版井字棋了！🎉

**运行效果示例**（输入 4、再输入 0、再输入 8……最后 X 赢了）：

```
 0 | 1 | 2
---+---+---
 3 | 4 | 5
---+---+---
 6 | 7 | 8
轮到 X，输入格子编号 0-8：4

 0 | 1 | 2
---+---+---
 3 | X | 5
---+---+---
 6 | 7 | 8
轮到 O，输入格子编号 0-8：0
...
X 赢了！
```

（棋盘打印顺序和这个示例一样，就说明你写对了。）

### 2.5 本章总结

- 棋盘 = 长度为 9 的列表
- 胜负 = 检查 8 条线
- 游戏 = "落子 → 判胜负 → 换人" 的循环

---

## 第 3 章：整理成类（对应项目 game.py）

文字版的问题是：棋盘、当前玩家都散在外面，不好给 AI 用（AI 要能复制、能试走）。把状态装进类里：

### 3.1 完整代码

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
        self.board = [EMPTY] * 9   # 棋盘
        self.current = X           # 轮到谁

    def legal_moves(self):
        """所有合法着法（空格的索引列表）"""
        return [i for i, c in enumerate(self.board) if c == EMPTY]

    def make_move(self, index):
        """落子，然后自动换人（游戏没结束的话）"""
        if self.board[index] != EMPTY:
            raise ValueError('这个格子已经有子了')
        self.board[index] = self.current
        if self.result() is None:
            self.current = O if self.current == X else X

    def winner(self):
        for a, b, c in LINES:
            if self.board[a] != EMPTY and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def result(self):
        """游戏结果：'X' / 'O' / 'D'(平局) / None(没结束)"""
        w = self.winner()
        if w:
            return w
        if EMPTY not in self.board:
            return 'D'
        return None

    def clone(self):
        """复制一个一模一样的棋盘 —— AI 搜索的地基"""
        g = NormalGame()
        g.board = self.board[:]    # 列表切片 = 复制！
        g.current = self.current
        return g
```

### 3.2 新知识点逐个讲

**列表推导式**：
```python
[i for i, c in enumerate(self.board) if c == EMPTY]
```
等价于：
```python
result = []
for i, c in enumerate(self.board):   # enumerate 同时给出索引 i 和元素 c
    if c == EMPTY:
        result.append(i)
```

**raise ValueError**：主动报错。AI 不该走非法着法；如果真的走了，让程序立刻崩溃比悄悄出错好（快速失败原则）。

**clone 为什么是最重要的方法**：AI 搜索要回答"如果我在 4 落子会怎样？如果我在 0 落子会怎样？"每个假设都需要一个独立的棋盘。clone 让 AI 在副本上随便试验，不弄脏真实棋局。

⚠️ **必须**用 `self.board[:]` 复制，不能直接 `g.board = self.board`！后者只是两个名字指向同一个列表，改一个另一个也跟着变。这叫**浅拷贝陷阱**，新手必踩。

**结果约定**（全项目通用）：`result()` 返回 `'X'`、`'O'`、`'D'`（平局）或 `None`（没结束）。`None` 在 if 里当 False 用，所以可以写 `if game.result():`。

### 3.3 重写主循环

```python
game = NormalGame()
while True:
    print_board(game.board)          # 复用第 2 章的 print_board
    move = int(input(f"轮到 {game.current}，输入 0-8："))
    if move not in game.legal_moves():
        print("非法着法！")
        continue
    game.make_move(move)
    r = game.result()
    if r:
        print_board(game.board)
        print("平局！" if r == 'D' else f"{r} 赢了！")
        break
```

### 3.4 对照项目源码

现在打开仓库里的 `game.py`，你应该能认出：

- 第 37-73 行就是我们的 `NormalGame`（项目里多了 `is_full`、`is_over` 等便捷方法）
- 第 283 行 `apply_move(game, move)`：统一处理普通/终极两种棋盘的落子（后面第 4 章会用到）
- 第 290 行 `apply_clone_result(game, move)`：克隆 + 落子 + 返回结果，一步到位（第 6 章用）

---

## 第 4 章：终极井字棋（对应项目 UltimateGame）

### 4.1 规则

- 大棋盘 = 3×3 个大格，每个大格里面是一个 3×3 小棋盘 → 总共 81 个格子
- 你在小棋盘 `(r, c)` 落子 → 对手**下一手必须**在大格 `(r, c)` 下
- 若那个大格已分出胜负或填满 → 对手可自由选任何未结束的大格
- 赢下一个 3×3 小棋盘 = 占领该大格
- 大格连成三线 = 赢得整局

### 4.2 数据结构

```python
micro = [[''] * 9 for _ in range(9)]   # 9 个小棋盘，每个 9 格
macro = [''] * 9                       # 9 个大格状态：'' / 'X' / 'O' / 'D'
active_macro = None                    # 对手必须去的大格编号（None = 自由选择）
```

⚠️ 注意：`[[''] * 9] * 9` 是**错的**！它产生 9 个指向同一个列表的引用，改一个全改。必须用列表推导式 `[[''] * 9 for _ in range(9)]`（`_` 是"我不关心这个变量"的惯例写法）。

### 4.3 核心逻辑

```python
class UltimateGame:
    def __init__(self):
        self.micro = [[EMPTY] * 9 for _ in range(9)]
        self.macro = [EMPTY] * 9
        self.current = X
        self.active_macro = None

    def macro_open(self, m):
        """大格 m 还能不能下？（没被占领 且 还有空格）"""
        return (self.macro[m] == EMPTY
                and any(c == EMPTY for c in self.micro[m]))

    def legal_moves(self):
        if self.active_macro is not None and self.macro_open(self.active_macro):
            # 被"路由"到大格 active_macro，只能在那里下
            return [(self.active_macro, i)
                    for i in range(9)
                    if self.micro[self.active_macro][i] == EMPTY]
        # 自由选择：遍历所有开放的大格
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
        w = winner_of(self.micro[macro])   # 复用第 2 章的 winner()，改名就行
        if w:
            self.macro[macro] = w          # 小棋盘分出胜负 → 占领大格
        elif all(c != EMPTY for c in self.micro[macro]):
            self.macro[macro] = 'D'        # 填满无胜负 → 该大格作废
        # 下一手的路由规则：对手必须去大格 micro（编号恰好相同）
        self.active_macro = micro if self.macro_open(micro) else None
        self.current = O if self.current == X else X

    def winner(self):
        # 和大棋盘判断完全一样，只是把 board 换成 macro
        for a, b, c in LINES:
            if self.macro[a] in (X, O) and self.macro[a] == self.macro[b] == self.macro[c]:
                return self.macro[a]
        return None

    def result(self):
        w = self.winner()
        if w:
            return w
        if all(not self.macro_open(m) for m in range(9)):  # 所有大格都关了
            return 'D'
        return None

    def clone(self):
        g = UltimateGame()
        g.micro = [row[:] for row in self.micro]  # 二维列表：每行都要复制！
        g.macro = self.macro[:]
        g.current = self.current
        g.active_macro = self.active_macro
        return g
```

### 4.4 关键技巧总结

- **小棋盘胜负 → 大格状态**：在 `make_move` 里自动完成，玩家和 AI 都不用管
- **路由规则一行搞定**：`active_macro = micro if self.macro_open(micro) else None`
- **二维列表深拷贝**：`[row[:] for row in self.micro]`，每行单独复制
- `any(...)` 和 `all(...)` 是内置函数：一个满足即 True / 全部满足才 True

### 4.5 对照项目源码

项目 `game.py` 第 75-139 行就是完整版。另外项目还有 `BitUltimateGame`（第 142 行）——同样的规则，但用**整数位运算**存棋盘（81 个格子 = 81 个二进制位），速度更快，第 9 章讲。

---

## 第 5 章：第一个 AI —— Random

### 5.1 思路

从所有合法着法里随便挑一个：

```python
import random

def random_move(game):
    return random.choice(game.legal_moves())
```

就这么简单。但它立下全项目的规矩：**所有 AI 引擎都是"输入棋盘，输出一个着法"的函数**。以后 UI 可以随便换引擎，因为接口统一。

### 5.2 分发函数

项目 `ai.py` 有统一入口 `get_ai_move(game, ai_type)`（ai.py 第 1306 行）：

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

UI 只调 `get_ai_move(game, 'MCTS')`，不用关心内部实现。这叫**多态 / 策略模式**——本教程里最重要的架构思想之一。

---

## 第 6 章：Basic AI —— 启发式引擎

### 6.1 思路（模仿一个还不错的普通玩家）

1. 我能直接赢吗？→ 赢
2. 对手下一步能赢吗？→ 挡
3. 都不行 → 中心 > 角落 > 边

### 6.2 "试走"技巧

判断"走这步会不会赢"，最稳妥的办法是**在克隆上试走**：

```python
def apply_clone_result(game, move):
    g = game.clone()
    apply_move(g, move)
    return g.result()
```

这是整个项目反复使用的核心思想：**用模拟代替规则枚举**。相比"数某条线有几个子"的写法，试走法不会漏掉任何情况（比如一步成两条线），代码也更短。

### 6.3 完整代码

```python
def get_basic_move(game):
    player = game.current
    opp = O if player == X else X
    moves = game.legal_moves()

    # 1. 能赢就赢
    for m in moves:
        if apply_clone_result(game, m) == player:
            return m

    # 2. 找出"走了之后对手能直接赢"的坏棋
    losing = []
    for m in moves:
        g = game.clone()
        apply_move(g, m)
        # 我走 m 之后，对手随便一步都能赢吗？
        if any(apply_clone_result(g, m2) == opp for m2 in g.legal_moves()):
            losing.append(m)
    safe = [m for m in moves if m not in losing]
    if safe and len(safe) < len(moves):
        return random.choice(safe)   # 有安全棋就走安全棋（等于挡杀）

    # 3. 偏好中心、角落、边
    for pref in (4, 0, 2, 6, 8, 1, 3, 5, 7):
        if pref in moves:
            return pref
    return random.choice(moves)
```

### 6.4 练习

写一个 `get_basic_move_ultimate(game)` 给终极井字棋用（提示：试走逻辑完全一样，只是着法是 `(macro, micro)` 元组；第 3 步的偏好改成"中央大格 4、中央小格 4、角落大格"——对照项目 `get_basic_move` ai.py 第 96-136 行）。

---

## 第 7 章：Minimax —— 穷举搜索（对应 ai.py 第 138 行起）

### 7.1 核心思想：双方都完美时会怎样？

假设双方永远选"对自己最好"的棋。那么：

> 轮到 X 时，X 选**让 X 得分最高**的着法；
> 轮到 O 时，O 选**让 X 得分最低**的着法（因为 O 的立场和 X 相反）。

每一层的"价值"从终局往回传：X 层取最大值，O 层取最小值 → **Minimax（极小化极大）**。

### 7.2 先学递归

```python
def countdown(n):
    if n <= 0:            # 基线条件：停止递归
        print("发射！")
        return
    print(n)
    countdown(n - 1)      # 调用自己
```

递归 = 函数调用自己。**必须**有基线条件（base case）停下来，否则无限递归导致程序崩溃（RecursionError）。

### 7.3 Minimax 代码

```python
import math

def minimax(game, maximizing, ai_player, depth=0):
    r = game.result()
    if r is not None:                       # 基线条件：游戏结束了
        if r == 'D':
            return 0
        # 分数从 AI 视角算；赢越快分越高（离终局越近越好）
        return (100 - depth) if r == ai_player else (depth - 100)

    if maximizing:
        best = -math.inf                    # 初始化为负无穷
        for m in game.legal_moves():
            g = game.clone()
            apply_move(g, m)
            best = max(best, minimax(g, False, ai_player, depth + 1))
        return best
    else:
        best = math.inf                     # 正无穷
        for m in game.legal_moves():
            g = game.clone()
            apply_move(g, m)
            best = min(best, minimax(g, True, ai_player, depth + 1))
        return best
```

顶层这样选棋：

```python
def minimax_move_normal(game):
    player = game.current
    best_moves, best_score = [], -math.inf
    for m in game.legal_moves():
        g = game.clone()
        apply_move(g, m)
        score = minimax(g, False, player)   # 我走完后轮到对方（min 层）
        if score > best_score:
            best_score, best_moves = score, [m]
        elif score == best_score:
            best_moves.append(m)
    return random.choice(best_moves), best_score
```

### 7.4 搜索树长什么样

```
          根（轮到我，max）
         /       |        \
    对手(min)  对手(min)  对手(min)
      /  \       /  \       /  \
    终局 终局   终局 终局   终局 终局
```

分数从最底层一层层传上来。井字棋全树只有 9! ≈ 36 万节点，电脑瞬间算完 → **完美棋手，永远不会输**。

### 7.5 Alpha-Beta 剪枝（提速但不改变结果）

如果已找到一条"保证 ≥ 5 分"的路，而另一条分支对手能把我压到 ≤ 3 分，那后一条分支不用继续看了：

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
            if beta <= alpha:      # 对手不会让我走到这 → 剪！
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

- `alpha`：我方目前能保证的最低分（只会涨）
- `beta`：对手能容忍的最高分（只会降）
- `beta <= alpha` 时，继续搜没意义 → break

### 7.6 终极井字棋怎么办？—— 评估函数

终极棋盘有 81 格，穷举不完，只能搜固定深度（比如 3 层），然后用**评估函数**给中间局面打分。对应项目 `eval_ultimate`（ai.py 第 179 行）：

```python
def eval_ultimate(game, player):
    score = 0
    # 已占领的大格：        +1000 / -1000（决定性）
    # 小棋盘里的 fork 威胁：  +40 / -40
    # 小棋盘 2 连 + 1 空：    +3 / -3
    # 小棋盘里已有几个子：    +0.5 / -0.5
    # 大棋盘 2 连 + 1 空：    +20 / -20
    return score
```

评估函数的精髓：**把"感觉"量化**。哪些局面特征重要、各值多少分，靠经验调整。搜到深度限制时，就用这个分数代替终局结果。

### 7.7 对照项目源码

- 普通模式：`_minimax_normal` + `minimax_move_normal`（ai.py 第 138-176 行）
- 终极模式：`_minimax_ultimate` + `minimax_move_ultimate`（第 234-283 行），还有动态深度（着法少时多搜一层）
- 进阶版 **Minimax Pro**：negamax + 置换表 + 迭代加深 + killer move + LMR + aspiration（第 1044-1140 行）——先跳过，第 9 章介绍思想

---

## 第 8 章：MCTS —— 蒙特卡洛树搜索（对应 ai.py 第 285 行起）

### 8.1 和 Minimax 的区别

Minimax 要**精确算清**每个局面。MCTS 反其道而行：**随机模拟几千盘**，用统计结果判断哪步好。就像用投票代替精确计算——所以叫"蒙特卡洛"（赌城名字，代表随机）。

### 8.2 四个步骤

每次迭代做四件事：

1. **选择（Selection）**：从根出发，用 UCB1 公式挑最有"前途"的子节点一路往下走
2. **扩展（Expansion）**：走到还没展开的节点，加一个新孩子
3. **模拟（Simulation / Rollout）**：从那里开始**随机下到底**
4. **回传（Backpropagation）**：把结果沿路加给每个祖先节点

### 8.3 节点

```python
class MCTSNode:
    def __init__(self, state, move, parent, mover):
        self.state = state        # 棋盘快照
        self.move = move          # 从父节点走到我这步棋
        self.parent = parent
        self.children = []
        self.mover = mover        # 这一步是谁下的（计分要用）
        self.visits = 0           # 被访问多少次
        self.wins = 0.0           # 赢了多少次（平局算 0.5）
        self.untried = state.legal_moves()  # 还没展开的着法
        random.shuffle(self.untried)
```

### 8.4 UCB1 公式 —— 选择的关键

```
UCB = 胜率 + c × sqrt( ln(父节点访问次数) / 自己访问次数 )
      ↑利用             ↑探索
```

- 第一项：**利用**——胜率高的优先
- 第二项：**探索**——访问少的优先（分母小 → 值大，访问多了自然变小）
- `c` 是探索系数，项目默认 1.4

两股力量平衡：既不要只走"目前看起来最好"的（可能被坑），也不要平均主义乱试。

```python
    def best_child(self, c):
        log_n = math.log(max(1, self.visits))
        best, best_val = None, -math.inf
        for child in self.children:
            if child.visits == 0:
                uct = math.inf        # 没访问过 = 无穷大 → 一定优先试
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
        # ---- 选择：一路选到叶子 ----
        node = root
        state = root.state.clone()
        while node.children and not node.untried:
            node = node.best_child(c)
            apply_move(state, node.move)

        # ---- 扩展：加一个孩子 ----
        if node.untried:
            m = node.untried.pop()
            mover = state.current
            apply_move(state, m)
            child = MCTSNode(state.clone(), m, node, mover)
            node.children.append(child)
            node = child

        # ---- 模拟：随机下到底 ----
        result = state.result()
        guard = 0
        while result is None and guard < 300:
            moves = state.legal_moves()
            if not moves:
                break
            apply_move(state, random.choice(moves))
            result = state.result()
            guard += 1   # 保险丝，防止意外死循环

        # ---- 回传 ----
        while node is not None:
            node.visits += 1
            if result == node.mover:
                node.wins += 1.0
            elif result == 'D':
                node.wins += 0.5
            node = node.parent

    return root
```

### 8.6 从根选棋

```python
def best_mcts_move(root):
    visited = [c for c in root.children if c.visits > 0]
    return max(visited, key=lambda c: c.wins / c.visits).move
```

选棋策略有两种：
- **最大访问数**：统计样本最多的最可信（项目用这个 + 细微修正：优先立即获胜的、只考虑访问够多的孩子）
- **最高胜率**：样本少时不可靠

### 8.7 为什么 MCTS 对终极井字棋有效？

终极棋盘第一步有 81 个选择，Minimax 精确搜索会爆炸。MCTS 的模拟天然"专注"：好的分支访问多、统计准；坏分支自动被冷落。迭代次数就是"思考时间"，可随时调整（项目默认 1600，见 sba.toml）。

### 8.8 项目里的增强（先认识名字）

- **启发式 rollout**（ai.py `_rollout_move` 第 469 行）：模拟不用纯随机，而是"能赢就赢、能挡就挡"——模拟质量大幅提高
- **动态 UCT**（`_uct_scale` 第 567 行）：前期多探索、后期多利用
- **树重用**（`_REUSE` 第 574 行）：对手走完一步后，保留旧树里对应分支，不用从头搜
- **RAVE/GRAVE**（第 734、850 行）：把"这步棋在这次模拟里出现的所有棋"也计入统计，信息共享、收敛更快
- **早停**：某步优势巨大时提前结束搜索

### 8.9 实例演练：手工走一遍 MCTS（3×3 棋盘）

光看代码容易晕，跟着走一遍就懂了。假设局面（普通井字棋，轮到 X）：

```
X | O | _
--+---+--
_ | X | _
--+---+--
_ | _ | _
```

索引 0=X、1=O、4=X，所以 X 的合法着法 = `2, 3, 5, 6, 7, 8`。

**第 1 次迭代**
1. **选择**：根节点刚建好、没有孩子 → 跳过。
2. **展开**：从 `untried` 随机弹出一个着法，假设是 5，创建子节点。
3. **模拟**：从"X 走 5 之后"开始随机下到底。假设**这次随机对局 X 赢了**。
4. **回传**：根和 5 号子节点 `visits += 1`；这步是 X 下的且 X 赢 → `wins += 1`。

```
根: visits=1
└─ 着法5: visits=1, wins=1.0   ← 胜率 100%
```

**第 2 次迭代**：根还有没展开的着法（2、3、6、7、8），所以继续展开，假设随机弹到 2。
从"X 走 2"模拟，**这次 O 赢了** → 2 号节点（X 下的、没赢）`wins += 0`。

```
根: visits=2
├─ 着法5: visits=1, wins=1.0
├─ 着法2: visits=1, wins=0.0
└─ 着法3,6,7,8: 还没展开
```

**第 3 次迭代**：再展开一个，假设是 6，模拟结果是**平局** → `wins += 0.5`。

```
根: visits=3
├─ 着法5: visits=1, wins=1.0
├─ 着法2: visits=1, wins=0.0
├─ 着法6: visits=1, wins=0.5
└─ 着法3,7,8: 还没展开
```

**第 4 次迭代**：根的所有着法都展开了 → **选择阶段终于登场**，用 UCB 公式挑孩子。
根的 `visits=3`，`ln(3) ≈ 1.10`，`c = 1.4`：

- 着法 5：`1.0 + 1.4 × √(1.10/1) ≈ 2.47`
- 着法 2：`0.0 + 1.4 × √(1.10/1) ≈ 1.47`
- 着法 6：`0.5 + 1.4 × √(1.10/1) ≈ 1.97`
- 着法 3、7、8：**没访问过 → UCB = 无穷大，一定先试**（假设随机选到 3）

选中 3 后在它的子树里展开、模拟（假设 X 又赢）、回传。最终：

```
根: visits=4
├─ 着法5: visits=1, wins=1.0
├─ 着法2: visits=1, wins=0.0
├─ 着法6: visits=1, wins=0.5
├─ 着法3: visits=1, wins=1.0
└─ 着法7,8: 还没展开
```

看出规律了吗？**没访问过的孩子永远优先被试一次**（UCB=∞），试过之后：
胜率高的会被越来越常选（利用），访问少的会补回来（探索）——这就是整个 MCTS 的灵魂。
真实项目把"随机下到底"换成"能赢就赢、能挡就挡"（`_rollout_move`），统计收敛更快；
迭代次数是几百到几千，手工走 4 次只是为了看懂机制。

### 8.10 引擎对比表（一页看懂）

| 引擎 | 一句话原理 | 普通 3×3 | 终极 9×9 | 思考时间 |
|---|---|---|---|---|
| Random | 合法着法里随机挑 | 很弱 | 很弱 | 0 |
| Basic | 赢 → 挡 → 中心/角落 | 新手级 | 新手级 | 0 |
| Minimax | 穷举 + alpha-beta | **完美（永不输）** | 深度受限 + 评估函数 | 秒级 |
| Minimax Pro | negamax + 置换表 + 剪枝 | 完美 | 评估更强 | 可控 |
| MCTS | 随机模拟几千盘做统计 | 强 | **实战首选** | 迭代数 × 毫秒 |

记不住也没关系：**普通棋盘用 Minimax、终极棋盘用 MCTS**，这就是项目的默认推荐。

---

## 第 9 章：优化技巧（进阶，可选）

### 9.1 Bitboard（位棋盘）

棋盘状态用**二进制位**存：每个格子一位，1 = 有子。81 格 → 两个 81 位整数（X 一个、O 一个）。

好处：
- `clone()` 只是复制两个整数，极快
- 判断一条线是否连成：`(mask & line) == line` 一次位运算搞定

```python
bit = 1 << (m * 9 + i)               # 第 (m,i) 格对应哪一位
g.x |= bit                           # 落一个 X 子（|= 是"按位或并赋值"）
taken = (g.x | g.o) >> (m * 9) & 0x1FF   # 取大格 m 的 9 位状态
```

项目里就是 `BitUltimateGame`（game.py 第 142 行），接口和列表版完全一样——MCTS 内部用位版搜，搜完把着法还给 UI。

#### 9.1.1 位运算入门（30 秒速成）

```python
a = 0b1010          # 二进制写法：1×8 + 0×4 + 1×2 + 0×1 = 10
a << 1              # 左移一位 = 乘 2 → 0b10100 = 20
1 << 3              # = 8：把"第 3 位"变成 1 的常用写法
mask = 0b111        # 三个位都是 1
a & mask            # 按位与：两边都是 1 才保留 → 0b0010 = 2
a | (1 << 0)        # 按位或：把第 0 位设成 1
(a >> 1) & 1        # 右移再取末位 = 取出"第 1 位"是几
```

只用记住 4 个：`<<`（左移）、`|`（设成 1）、`&`（检查/过滤）、`>>`（右移）。

#### 9.1.2 BitUltimateGame 怎么用位存棋盘

- X 的 81 个格子 → 一个 81 位整数 `self.x`（第 `m*9+i` 位 = 大格 m、小格 i）
- O 的 81 个格子 → `self.o`
- 大格谁赢 → `self.mx` / `self.mo`（各 9 位）

判断"大格 m 的第 i 格有没有子"：

```python
bit = 1 << (m * 9 + i)
taken = (self.x | self.o) & bit   # 非 0 表示这个格已被占用
```

#### 9.1.3 胜利判断：一次位运算搞定

一条线（例如大格 0 的 `(0,1,2)`）对应二进制 `0b111000000`（第 0–2 位都是 1）：

```python
def is_line_win(x_mask, line):
    return (x_mask & line) == line   # X 的位盖住整条线 = 赢了
```

列表版要遍历 3 个格子、比较 3 次；位版一次 `&` + 一次 `==`，这就是"快"的来源。

#### 9.1.4 一个诚实的结论（benchmark 文化）

位板**理论上**快很多（clone 只复制 4 个整数、胜利判断常数时间）。
但项目实测（`docs/decisions.md` D12）：MCTS 已经用启发式 rollout 之后，
位板**没有带来额外可测的加速**，所以它保留下来主要因为接口统一、省内存、
值得学习——而不是"快 10 倍"。`sba.toml` 里 `bitboard = true` 开着，但**不要凭感觉优化，要量**。

### 9.2 对象池（NodePool）

MCTS 每局建几千个节点，频繁创建/销毁对象拖慢速度（垃圾回收压力）。对象池 = 用完不销毁，回收复用：

```python
class NodePool:
    def __init__(self):
        self._free = []         # 空闲节点仓库

    def acquire(self, state, move, parent, mover):
        if self._free:
            n = self._free.pop()     # 复用旧对象，重置字段
            n.state = state
            n.move = move
            # ... 重置其他字段
            return n
        return MCTSNode(state, move, parent, mover)

    def release(self, root):
        # 遍历整棵树，把节点放回仓库
        ...
```

对应项目 `NodePool`（ai.py 第 312 行）。

### 9.3 置换表（Transposition Table）

不同着法顺序可能到达**同一个局面**（比如先走 0 再走 4，和先走 4 再走 0）。Minimax 会重复计算这些局面。置换表 = 字典：`局面 → 已算好的分数`，遇到重复局面直接查表。对应项目 `_negamax_tt`（ai.py 第 1044 行），还有 `tt_max` 限制表大小防止内存爆炸。

### 9.4 开局书（Opening Book）

开局不需要算，直接查表。项目 `ULTIMATE_BOOK`（ai.py 第 1168 行）：X 第一步走哪里 → O 应该回哪里，人工挑选的好应手，省下开局的大量搜索时间。配置化后放在 sba.toml 的 `[engine.opening_book_ultimate]` 段。

### 9.5 搜索工程技巧全家桶（Minimax Pro 用了全套）

- **迭代加深**：先浅搜拿个大致结果，再逐层加深；超时就用上次结果
- **Killer Move**：记住"上次在这里引发剪枝的棋"，下次优先试
- **LMR（Late Move Reduction）**：排序靠后的着法多半差，用较浅深度搜
- **Aspiration Window**：猜一个分数区间搜，猜错再放宽

这些是国际象棋引擎的标准套路。井字棋项目里是"杀鸡用牛刀"，但对学习极有价值——这就是项目 ai.py 第 965-1140 行的 Minimax Pro。

---

## 第 10 章：配置文件（对应 sba.toml）

### 10.1 为什么要配置

把"探索系数多大、MCTS 迭代几次"写死在代码里，每次调都要改代码。放进配置文件，改完重启就行。TOML 是简单易懂的配置格式：

```toml
[engine]
uct_c = 1.4
rollout_heuristic = true

[session]
mcts_budget = 1600
```

### 10.2 读取（Python 3.11+ 内置 tomllib）

```python
import tomllib

def load_config():
    try:
        with open('sba.toml', 'rb') as fh:
            return tomllib.load(fh)     # → 嵌套 dict
    except Exception:
        return {}                       # 文件没了就空 dict
```

`with open(...)` 是标准文件打开方式，用完自动关闭。`'rb'` = 二进制读模式（tomllib 要求）。

### 10.3 默认值回退（重要模式）

```python
DEFAULTS = {'engine': {'uct_c': 1.4, 'rollout_heuristic': True}}

def cfg_engine(name, default=None):
    # 先看配置文件，没有再看代码默认值，再没有用参数默认值
    return (CONFIG.get('engine', {})
            .get(name, DEFAULTS['engine'].get(name, default)))
```

用户少写一个键，程序也能跑——这就是项目 README 说的 "Missing keys fall back to code defaults"。对应 ai.py 第 40-89 行。

### 10.4 测试时重置

`set_engine_config(overrides)`（ai.py 第 71 行）允许测试把配置改掉再还原，保证测试不受环境文件影响。这是个好习惯。

---

## 第 11 章：桌面图形界面（PySide6）

### 11.1 安装与核心概念

```bash
pip install pyside6
```

Qt 核心概念：

- **QWidget**：一切界面元素的基类（窗口、按钮、面板）
- **布局（Layout）**：自动排列子控件（`QHBoxLayout` 横排 / `QVBoxLayout` 竖排 / `QGridLayout` 网格）
- **信号与槽（Signal & Slot）**：按钮被点击 → 触发你的函数。这是 Qt 的灵魂

### 11.2 最小窗口

```python
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton

app = QApplication(sys.argv)      # 必须有，且整个程序只能有一个
win = QWidget()
win.setWindowTitle("我的井字棋")
layout = QVBoxLayout(win)         # 竖排布局

label = QLabel("欢迎！")
button = QPushButton("点我")
button.clicked.connect(lambda: label.setText("你点了我！"))

layout.addWidget(label)
layout.addWidget(button)
win.show()
sys.exit(app.exec())              # 进入事件循环：程序在这里"活着"
```

关键点：`app.exec()` 之前只是**搭建**界面；`exec()` 进入**事件循环**，之后程序靠信号槽响应（事件驱动）。按钮被点击时，Qt 自动调用你 connect 的函数。

### 11.3 用按钮做 3×3 棋盘

最直观：9 个按钮放进 QGridLayout：

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
    # 判断胜负、轮到 AI 时调 get_ai_move...
```

### 11.4 lambda 闭包陷阱 ⚠️

上面 `lambda _=False, idx=i: ...` 里的 `idx=i` **必须写**。如果写成 `lambda: on_click(i)`，所有按钮都会用**循环结束后**的 i（=8）——因为 lambda 捕获的是变量本身，不是当时的值。用默认参数 `idx=i` 把值"定"在创建那一刻。这是 Qt 新手最经典的 bug。

（`_=False` 是给信号默认传的 checked 参数占位，可以不理它。）

### 11.5 自定义绘制棋盘（项目同款，进阶）

按钮做 81 格终极棋盘太重。项目用 **QPainter** 在 QWidget 上自己画：`BoardWidget` 重写 `paintEvent` 画格线、棋子、高亮；重写 `mousePressEvent` 处理点击。核心结构：

```python
class BoardWidget(QWidget):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        # 画格线、X（两条线）、O（圆）、当前大格高亮...
        painter.end()

    def mousePressEvent(self, event):
        # 根据 event.position() 算出点中了哪一格
        # → game.make_move(...) → self.update() 触发重绘
```

要点：Qt 不会"自动重画"，状态变了要调 `self.update()`，Qt 随后调你的 `paintEvent`。

### 11.6 AI 不卡界面的关键：线程

MCTS 想一步可能要 1-2 秒。如果在主线程（UI 线程）里同步计算，界面会**冻结**（点了没反应）。项目把 AI 放进工作线程，算完发信号通知 UI：

```python
class AIWorker(QThread):
    finished_move = Signal(object)      # 自定义信号：算完发出

    def run(self):
        move = get_ai_move(self.game, self.ai_type, ...)
        self.finished_move.emit(move)   # 通知主线程
```

`Signal` 是跨线程安全的消息通道——Qt 保证槽函数在主线程执行，可以放心更新界面。

### 11.7 对照项目源码

项目的 `qtui.py` 有 1000+ 行，包括：Material Design 3 配色、深浅主题、动画、胜率图表（QtCharts）、AI 助手面板、可选 SiliconUI 主题。作为新手，先实现：棋盘绘制 + 点击落子 + AI 线程 + 菜单，其他逐步加。

---

## 第 12 章：测试（pytest）

### 12.1 为什么要测试

改代码后，"棋盘还能不能正常赢？AI 还会不会下非法棋？"手动点太慢。测试 = 把检查自动化，每次跑一遍立刻知道有没有改坏。

### 12.2 第一个测试

```python
# tests/test_game.py
from game import NormalGame, X, O

def test_win_by_row():
    g = NormalGame()
    g.make_move(0)  # X
    g.make_move(3)  # O
    g.make_move(1)  # X
    g.make_move(4)  # O
    g.make_move(2)  # X → 第一行连成
    assert g.result() == X     # assert = "我断言这是真的，否则测试失败"
```

运行：

```bash
pip install pytest
pytest
```

绿了（PASSED）= 没坏；红了（FAILED）= 精确告诉你哪行断言的哪个值不对。

### 12.3 测试要覆盖什么

对照项目 `tests/`：

- `test_game.py`：规则——胜负、平局、路由规则、clone 独立性、非法着法报错
- `test_ai.py`：AI——每个引擎都走合法棋、能赢时 Basic 会赢、Minimax 完美不败
- `test_alphazero.py`：训练系统的冒烟测试（小规模快速跑通）
- `conftest.py`：共享的测试工具

**新手铁律**：每次改完核心逻辑，跑一遍 `pytest`。

---

## 第 13 章：AlphaZero 概念入门

### 13.1 一句话版本

AlphaZero = **神经网络 + MCTS**。神经网络学习"哪个局面好、哪步棋可能好"，替代 MCTS 里的随机模拟和人工规则，让搜索又快又准。2017 年 DeepMind 用它打败了所有围棋、国际象棋 AI。

### 13.2 双头网络

输入：当前棋盘（编码成数字张量）。输出两样：

- **策略头（policy head）**：每步棋的"先验概率"——网络觉得哪步值得下
- **价值头（value head）**：当前局面对谁的胜率——替代随机模拟到终局

对应项目 `AZNet.forward`（alphazero.py 第 157 行）：`return p, torch.tanh(self.val_fc2(v))`——策略概率 + 一个 -1~1 的价值。

### 13.3 神经网络怎么"看"棋盘

棋盘被编码成 4 层 9×9 矩阵（`encode`，alphazero.py 第 81 行）：

```
第 0 层：X 的棋子位置（1.0 / 0.0）
第 1 层：O 的棋子位置
第 2 层：轮到谁
第 3 层：当前必须下的大格
```

就像给网络 4 张"透明胶片"叠起来。然后过几层**卷积**（conv2d，图像识别核心技术，自动提取"这里有两个连子、那里有威胁"之类的特征），最后输出两个头。

### 13.4 训练 = 自己和自己下（自对弈）

```
1. 用当前网络 + MCTS 自己和自己下一盘（每步记录：局面 → MCTS 算出的落子分布）
2. 终局后，每个局面标上真实结果（赢 +1，输 -1，平 0）
3. 把"局面 → (落子分布, 结果)"当教材，训练网络逼近 MCTS 的判断
4. 循环几千盘，网络越来越强，MCTS 也越搜越准
```

对应 `self_play_game`（alphazero.py 第 407 行）和 `train`（第 471 行）。

### 13.5 MCTS 里怎么用网络

- **模拟那步直接砍掉**：价值头直接给出局面分数
- **选择时**：UCB 公式里的"先验"用策略头概率（`priors`，alphazero.py 第 331 行）
- **根节点加 Dirichlet 噪声**：强制开局探索，防止网络死记一条路

对应 `mcts_search`（alphazero.py 第 244 行）——结构和第 8 章我们写的 MCTS 一模一样，只是"模拟"换成"问网络"。

### 13.6 需要什么基础

PyTorch 是深度学习框架。建议先补：线性代数入门（矩阵）、微积分入门（梯度下降）。好消息：**用网络不需要懂内部数学**，torch 帮你算一切。

项目里的技术报告 `docs/AlphaZero_UTTT_Technical_Report.html` 有训练数据详细分析，等你能看懂网络结构后值得一读。

---

---

## 第 14 章：Git 版本控制（交作业前必读）

### 14.1 为什么需要 Git

没有 Git 的"存档"是复制文件夹（`项目_final_真最终版.py`）——迟早会乱。
Git 帮你：

- **随时回退**：每次提交（commit）都是一个检查点，改坏了回到上一个
- **看清楚改了什么**：`git diff` 显示每一行改动
- **多人协作 / 换电脑**：把仓库推到 GitHub，到处都能拉下来
- **面试验证**：`git log` 是你的"开发日记"，面试官爱看

### 14.2 三个区域（先懂概念）

```
工作区（你正在改的文件夹）
   ↓ git add
暂存区（准备好了的改动）
   ↓ git commit
历史（一串提交，每个都能回退）
```

### 14.3 新手先记这 7 条指令

```bash
git init              # 在项目文件夹建立仓库（只做一次）
git status            # 看现在改了什么（最常用！）
git add 文件名         # 把改动放进暂存区；git add . 表示全部
git commit -m "feat: 加入 MCTS 引擎"   # 存一个检查点
git log --oneline     # 看提交历史（一行一条）
git diff              # 看改动的具体内容
git checkout -- 文件名  # 丢弃某个文件的改动（回退）
```

### 14.4 推到 GitHub

```bash
git remote add origin https://github.com/你的账号/项目名.git
git push -u origin main    # 第一次推送
git pull                   # 之后每次开工先拉最新
```

### 14.5 .gitignore：别把垃圾传上去

`.venv/`（几百 MB 的依赖）、`__pycache__/`（缓存）、模型权重、密钥——
这些不该进仓库。在项目根目录建 `.gitignore` 写上它们，git 就会自动忽略。
直接看本仓库根目录的 `.gitignore` 当范例。

### 14.6 提交信息怎么写

一句话说清"做了什么"：`feat:` 新功能、`fix:` 修 bug、`docs:` 文档。
例如项目真实历史：`fix(ai): align workers default with sba.toml (8)`。

### 14.7 分支（先懂概念）

主分支（`main`）放"能跑的版本"，新功能在分支上开发，确认没问题再合并。
改坏了不用怕——回到上一个提交就行。本项目真实用了 4 个分支，
`docs/decisions.md` 有记录。

### 14.8 练习

把第 2 章的 `main.py` 提交 3 次：第一次"能跑"、第二次"加了防呆"、
第三次"加了平局提示"。每次 `git log --oneline` 看历史越来越长。

---

## 第 15 章：虚拟环境 venv（跑项目前必读）

### 15.1 为什么需要

`pip install` 默认装到全局，两个项目需要不同版本时会互相打架。
venv = 给每个项目一个**独立的 Python 环境**：

```bash
python -m venv .venv          # 建立（Windows / Mac / Linux 一样）
.venv\Scripts\activate        # 启用（Windows PowerShell）
source .venv/bin/activate     # 启用（Mac / Linux）
deactivate                    # 退出
```

启用后命令行开头会出现 `(.venv)`，这时候 `pip install` 只装进这个项目。

### 15.2 requirements.txt：一键装齐依赖

```bash
pip freeze > requirements.txt      # 导出当前依赖清单
pip install -r requirements.txt    # 别人（或新电脑）一键安装
```

看本仓库的 `requirements.txt`：就 3 行（nicegui / pyside6 / pytest）——因为大型依赖
（PyTorch、ROCm）体积大、版本挑环境，单独手动装。

### 15.3 三个常见坑

1. 忘了启用 venv 就 `pip install` → 装到全局（看命令行有没有 `(.venv)`）
2. 没把 `.venv/` 加进 `.gitignore` → 把几百 MB 推上 GitHub
3. 换电脑跑项目 = `git clone` → `python -m venv .venv` → `pip install -r requirements.txt`

---

## 第 16 章：项目管理（把作业做成作品）

### 16.1 先写 README

别人（评审、面试官）第一眼看的是 README。至少包含：
一句话介绍、怎么安装怎么跑、功能列表、**截图**。
本仓库的 README 就是范例——中英双语、带用法和架构图。

### 16.2 决策日志（decision log）：记录"为什么"

代码只告诉你"做了什么"，不告诉你"为什么这么做"。
项目里的 `docs/decisions.md`（D1–D18）就是范例：

- D12：bitboard 实测无增益 → 保留但默认关
- D18：多进程自对弈，8 worker 约 3× 吞吐

评审最喜欢这种**有证据的选择**——比"我试了很多方法"有说服力一百倍。

### 16.3 用 benchmark 说话

不要说"我觉得 MCTS 比较强"，要说"**30 局 22 胜 4 平 4 负**"。
本项目的 `python SBA.py --bench --ai-a Minimax --ai-b MCTS --games 30`
就是干这个的。数字不会说谎，也经得起追问。

### 16.4 测试让你敢重构

每改完核心逻辑跑一次 `pytest`，全绿才提交。
有测试兜底，你才敢放心拆代码、改结构（第 12 章）。

### 16.5 一次只做一件事

流程：**功能 → 测试 → 提交 → 下一个**。坏了一个功能，
`git revert` 回退，不影响其他成果。这是本仓库 57 个测试、
100+ 次提交背后的工作方式。

### 16.6 GitHub 当作品集

README 放截图、录一段 2–3 分钟 demo 影片、把链接放进 CV。
这份教程 + 你的项目本身，就是比任何证书都有说服力的作品集。

---

## 附录 A：新手常见错误对照表

| 症状 | 原因 | 解法 |
| --- | --- | --- |
| `IndentationError` | 缩进不对（Tab 和空格混用） | 统一用 4 个空格 |
| `list index out of range` | 索引越界（第 9 格是 board[8]，没有 board[9]） | 检查索引范围 |
| 改一个列表，另一个也跟着变 | 浅拷贝 `b = a` | 用 `a[:]` 或 `.copy()` |
| `NoneType has no attribute ...` | 函数没 return，拿到 None 还调方法 | 检查 return |
| 按钮全都触发同一个序号 | lambda 闭包陷阱 | 用默认参数绑值 `idx=i` |
| 界面点了没反应 / 卡死 | 在主线程算 AI | 用 QThread |
| 中文乱码 | 控制台编码不是 UTF-8 | `chcp 65001` |
| `ModuleNotFoundError` | 没装包 或 不在文件目录 | pip install / cd 到项目目录 |
| `RecursionError` | 递归没有基线条件 | 检查 base case |
| 二维列表 `[['']*9]*9` 改一个全改 | 9 个引用指向同一个列表 | `[['']*9 for _ in range(9)]` |

## 附录 B：动手路线建议

1. 第 2 章完成后：加"悔棋"功能（用列表记录每步历史）
2. 第 5 章完成后：写 Random vs Basic 打 100 盘统计胜率
3. 第 7 章完成后：验证 Minimax 对 Random 100% 不败
4. 第 8 章完成后：MCTS 用 50 / 200 / 800 次迭代互相打，看强度差多少
5. 加 UI 时先画棋盘，再接线，别一次写完

## 附录 C：推荐资源

- Python 官方教程（中文）：<https://docs.python.org/zh-cn/3/tutorial/>
- 菜鸟教程 Python：<https://www.runoob.com/python3/>
- Minimax 可视化：Google 搜 "tic tac toe minimax visualization"
- MCTS 综述论文：Browne et al. 2012《A Survey of Monte Carlo Tree Search Methods》
- AlphaZero 论文：Silver et al. 2017《Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm》
- PySide6 官方文档：<https://doc.qt.io/qtforpython-6/>
- 本项目技术报告：`docs/AlphaZero_UTTT_Technical_Report.html`

## 附录 D：术语表（一页速查）

| 术语 | 一句话解释 |
|---|---|
| 模块 module | 一个 `.py` 文件 |
| 浅拷贝 / 深拷贝 | 只复制外壳 / 连内容一起复制 |
| clone | 复制一个独立棋盘（AI 试走用） |
| 递归 recursion | 函数调用自己，必须有基线条件停下 |
| 基线条件 base case | 递归停止的条件 |
| alpha-beta 剪枝 | 提前砍掉"不可能改变结果"的分支 |
| 置换表 transposition table | "局面 → 分数"的缓存字典 |
| UCB / UCT | MCTS 选孩子的公式：胜率 + 探索项 |
| rollout / 模拟 | 从某局面随机下到终局的一次"抽签" |
| 回传 backpropagation | 把模拟结果加回路径上每个节点 |
| 启发式 heuristic | 靠经验规则，而非精确计算 |
| 评估函数 evaluation | 给未结束的局面打分 |
| 位棋盘 bitboard | 用整数位存棋盘，又快又省内存 |
| 开局书 opening book | 开局直接查表，不搜索 |
| 残局表 tablebase | 终局附近的精确解 |
| 神经网络 | 一堆可学习参数的"函数" |
| 策略头 / 价值头 | 输出"每步棋概率" / "局面胜率"的两个网络头 |
| 自对弈 self-play | 自己跟自己下棋产生训练数据 |
| 损失函数 loss | 预测和真实差多少，越小越好 |
| 梯度下降 | 让损失变小的参数更新方法 |

## 附录 E：读懂错误信息（除错入门）

**第一步：读最后一行。** Python 报错时最重要的信息在最后：

```
ValueError: invalid literal for int() with base 10: 'a'
```

冒号前是**错误类型**（`ValueError` = 值错误），冒号后是**原因**（`'a'` 转不成整数）。

**第二步：往上找 `File` 行。**

```
Traceback (most recent call last):
  File "main.py", line 7, in <module>
    move = int(input(...))
ValueError: invalid literal for int() with base 10: 'a'
```

`File "main.py", line 7` = 出错的位置（文件 + 行号），下面那行就是出错的代码。

**第三步：问自己三个问题。**
1. 错误类型是什么？（`ValueError` / `IndexError` / `TypeError` / `IndentationError`…）
2. 出错在哪一行？（`line N`）
3. 那一行用到的东西，类型对吗？值对吗？（用 `print()` 打印出来看）

**最常见的三种：**

| 错误 | 意思 | 常见原因 |
|---|---|---|
| `IndentationError` | 缩进错了 | Tab 和空格混用 |
| `IndexError: list index out of range` | 索引越界 | 第 9 格是 `board[8]`，没有 `board[9]` |
| `TypeError: ... not supported between instances of ...` | 类型不匹配 | 字符串当数字用（如 `'2' + 3`） |

**万能除错法：`print()`。** 怀疑哪里出错，就在那行前后打印看看：

```python
print("move 的值是：", move)   # 看看变量到底是什么
```

看完删掉。这是新手最快的除错方式；正式做法是用调试器（Thonny 点"步进"可以一行一行看）。

---

> 教程到这里就结束了。整个项目的秘密就一句话：**从最简单的能跑的东西开始，每次只加一个功能，跑通了再继续。** 祝你玩得开心，加油！🚀
