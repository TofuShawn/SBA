#!/usr/bin/env python3
"""技術報告一致性檢查（v1.2）。

三項檢查：
  A) 得分率：凡「nW / nD / nL」或「n 勝 n 平 n 負」後面標了百分比者，
     必須等於 (勝 + 0.5 × 和) / 局數（容差 0.2 個百分點）。
  B) 引用：A.4 參考文獻的編號集合，與內文 [n] 標記的編號集合必須雙向覆蓋
     （沒有孤立引用、也沒有未被引用的文獻）。
  C) 編號：表 1..N 與 圖 1..M 必須連續、無重號；圖表清單項數需相符。

用法：
  python docs/static-build/check-report.py [--check] [路徑]
離開碼 0 = 全部通過，1 = 有問題。
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / "AlphaZero_UTTT_Technical_Report.html"

SCORE_PATTERNS = [
    # 百分比必須緊跟在 W/D/L 之後（中間只能是標點或標籤），不可跨越中文散文
    re.compile(r"(\d+)\s*W\s*/\s*(\d+)\s*D\s*/\s*(\d+)\s*L[^0-9%A-Za-z\u4e00-\u9fff]{0,12}?(\d+(?:\.\d+)?)\s*%"),
    re.compile(r"(\d+)\s*勝\s*(\d+)\s*平\s*(\d+)\s*負[^0-9%A-Za-z\u4e00-\u9fff]{0,12}?(\d+(?:\.\d+)?)\s*%"),
]


def strip_code(text: str) -> str:
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<pre\b.*?</pre>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<code\b.*?</code>", " ", text, flags=re.S | re.I)
    text = re.sub(r'<span class="kbd">.*?</span>', " ", text, flags=re.S | re.I)
    return text


def check_score_rates(text: str) -> list[str]:
    problems = []
    seen = set()
    plain = re.sub(r"<[^>]+>", "|", strip_code(text))
    for pattern in SCORE_PATTERNS:
        for m in pattern.finditer(plain):
            w, d, l, stated = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
            n = w + d + l
            if n == 0:
                continue
            score = (w + 0.5 * d) / n * 100
            key = (w, d, l, stated)
            if key in seen:
                continue
            seen.add(key)
            if abs(score - stated) > 0.2:
                problems.append(
                    f"得分率不符：{w}W/{d}D/{l}L 標示 {stated}%，應為 {score:.1f}%"
                )
    return problems


def check_citations(text: str) -> list[str]:
    problems = []
    ref_start = text.find("A.4 參考文獻")
    if ref_start < 0:
        return ["找不到 A.4 參考文獻"]
    ref_end = text.find("</table>", ref_start)
    ref_block = text[ref_start:ref_end]
    refs = {int(n) for n in re.findall(r"<tr><td>(\d+)</td>", ref_block)}

    body = text[:ref_start] + text[ref_end:]
    body = strip_code(body)
    cited: set[int] = set()
    for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", body):
        cited.update(int(x) for x in re.findall(r"\d+", m.group(1)))

    missing = sorted(refs - cited)
    orphan = sorted(cited - refs)
    if missing:
        problems.append(f"未被內文引用的文獻：{missing}")
    if orphan:
        problems.append(f"內文引用但 A.4 不存在：{orphan}")
    return problems


def check_numbering(text: str) -> list[str]:
    problems = []
    tables = [int(n) for n in re.findall(r'<p class="tbl-cap" id="tbl\d+">表 (\d+)　', text)]
    figures = [int(n) for n in re.findall(r'<div class="cap" id="fig\d+"><b>圖 (\d+)　', text)]
    for label, numbers in (("表", tables), ("圖", figures)):
        if not numbers:
            problems.append(f"找不到任何{label}編號")
            continue
        expected = list(range(1, len(numbers) + 1))
        if numbers != expected:
            problems.append(f"{label}編號不連續：{numbers}")
    list_figs = len(re.findall(r'<li><a href="#fig\d+">圖 \d+</a>', text))
    list_tbls = len(re.findall(r'<li><a href="#tbl\d+">表 \d+</a>', text))
    if list_figs != len(figures):
        problems.append(f"圖表清單圖項數 {list_figs} ≠ 圖編號數 {len(figures)}")
    if list_tbls != len(tables):
        problems.append(f"圖表清單表項數 {list_tbls} ≠ 表編號數 {len(tables)}")
    return problems


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else DEFAULT
    if not path.exists():
        print(f"找不到檔案：{path}")
        return 1

    text = io.open(path, encoding="utf-8").read()
    results = [
        ("A 得分率一致性", check_score_rates(text)),
        ("B 引用雙向覆蓋", check_citations(text)),
        ("C 圖表編號連續", check_numbering(text)),
    ]

    failed = False
    for name, problems in results:
        if problems:
            failed = True
            print(f"[FAIL] {name}")
            for p in problems:
                print(f"       - {p}")
        else:
            print(f"[ OK ] {name}")
    print(f"\n{path.name}: {'有問題' if failed else '全部通過'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
