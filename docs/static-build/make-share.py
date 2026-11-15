#!/usr/bin/env python3
"""從自足版（docs/static）產生「對外分享／列印版」HTML。

做的是**機械式刪減＋一致性修補**，不改寫任何語句內容：
  1. 移除開發者用元素：#mathwarn 離線公式警告、hero 的「專案／分支／硬體」
  2. 移除已被 §18 取代的 §7.4–7.5（v1.0 排程設定），插入一行指路
  3. 移除「附錄 C 指令速查」，並把「附錄 D 參考文獻」改名為「附錄 C」——
     同步修目錄連結、以及內文「指令見附錄 C」的失效指涉
  4. 可選：--drop <id> 再刪掉指定的章節（例如 --drop s12 --drop s15）

用法：
  python docs/static-build/make-share.py <static.html> <out.html> [--drop ID ...]

之後可接：
  python docs/static-build/embed-cjk-fonts.py <out.html> <out-embedded.html>
  python docs/static-build/make-pdf.py <out-embedded.html>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def cut_between(text: str, start: str, stop: str, repl: str = "", keep_stop: bool = False) -> str:
    i = text.find(start)
    if i < 0:
        print(f"  [SKIP] 找不到起點：{start[:48]}")
        return text
    j = text.find(stop, i)
    if j < 0:
        print(f"  [SKIP] 找不到終點：{stop[:48]}")
        return text
    return text[:i] + repl + (text[j:] if keep_stop else text[j + len(stop):])


def cut_element(text: str, start: str, close: str) -> str:
    i = text.find(start)
    if i < 0:
        print(f"  [SKIP] 找不到：{start[:48]}")
        return text
    j = text.find(close, i)
    if j < 0:
        return text
    return text[:i] + text[j + len(close):]


def drop_section(text: str, sec_id: str) -> str:
    """刪掉 <h2 id="sec_id"> 到下一個 <h2（或附錄結尾）。"""
    m = re.search(r'<h2 id="%s"[^>]*>' % re.escape(sec_id), text)
    if not m:
        print(f"  [SKIP] 找不到章節 id={sec_id}")
        return text
    nxt = re.search(r"<h2[ >]", text[m.end():])
    end = m.end() + nxt.start() if nxt else len(text)
    print(f"  · 刪除章節 #{sec_id}（{end - m.start():,} 字元）")
    return text[:m.start()] + text[end:]


def main() -> int:
    ap = argparse.ArgumentParser(description="產生對外分享版 HTML")
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--drop", action="append", default=[],
                    help="額外刪除的章節 id（可重複，如 --drop s12）")
    ap.add_argument("--keep-appendix-c", action="store_true",
                    help="保留「附錄 C 指令速查」（預設刪除）")
    args = ap.parse_args()

    t = args.src.read_text(encoding="utf-8")
    before = len(t)
    print("刪減：")

    # 1. 開發者元素
    t = cut_element(t, '<div id="mathwarn"', "</div>")
    print("  · 離線公式警告 callout")
    for key in ("專案", "分支", "硬體"):
        new = re.sub(r"[ \t]*<span><b>%s</b>.*?</span>\n" % key, "", t, count=1, flags=re.S)
        if new != t:
            print(f"  · hero meta：{key}")
            t = new

    # 2. §7.4–7.5（v1.0 設定，已被 §18 取代）
    note = ('<p class="small"><b>本分享版略去 §7.4–7.5</b>（v1.0 的 parity rule 與 '
            '<span class="kbd">k</span> 排程設定）：該兩節已被 <a href="#s18">§18</a> 取代'
            '（β 的實測更正、定義域三層、分桶約定），保留會與 §18 衝突。</p>\n\n')
    t2 = cut_between(t, "<h3>7.4 AMAF 更新規則", "<h3>7.6 與", note, keep_stop=True)
    if t2 != t:
        print("  · §7.4–7.5（v1.0 排程）")
        t = t2

    # 3. 附錄 C 指令速查 → 刪；附錄 D 參考文獻 → 改名為 C（含目錄與內文指涉）
    if not args.keep_appendix_c:
        t2 = drop_section(t, "sc")
        if t2 != t:
            t = t2
            t = t.replace('id="sd">附錄 D · 參考文獻', 'id="sd">附錄 C · 參考文獻')
            t = re.sub(r'[ \t]*<li><a href="#sc">[^<]*</a></li>\n', "", t)
            t = t.replace('href="#sd">附錄 D：參考文獻', 'href="#sd">附錄 C：參考文獻')
            t = t.replace("指令見附錄 C。",
                          "指令：<code>ai.mcts_rave_search(UltimateGame(), T, prev_root=None)</code>。")
            print("  · 附錄 C 指令速查（並把參考文獻改名為附錄 C、修目錄與內文指涉）")

    # 4. 使用者指定的額外刪減
    for sec in args.drop:
        t = drop_section(t, sec)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(t, encoding="utf-8")
    print(f"\n輸出 {args.dst}｜{before:,} → {len(t):,} 字元 "
          f"（{(before - len(t)) / before * 100:.1f}% 縮減）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
