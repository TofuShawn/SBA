#!/usr/bin/env python3
"""檢查 PDF 的字型嵌入品質（列印／分享前用）。

為什麼需要：Chromium 的 PDF 後端對「variable font」（例如 Windows 上的
NotoSerifTC-VF.ttf＝思源宋體、NotoSansTC-VF.ttf＝思源黑體）無法正常嵌入，
會退化成 Type3 外框（無具名字型）。外觀與列印沒問題，但會導致：
  - ToUnicode 不完整 → PDF 內搜尋／複製中文失敗（例：「一致」「共用」搜不到）
  - 部分閱讀器的無障礙／朗讀功能失效
改字型堆疊往往看不出來，所以必須量測。

判讀：
  Type0（CIDFontType0/2）＝正常嵌入的具名字型 ✓
  Type3 且無 BaseFont      ＝外框化。**注意**：本專案的數學式是 MathJax 產生的
                             向量外框，本身就是 Type3，屬正常；因此 Type3 數量
                             **不**作為判準，真正的判準是下面兩項：
    (1) 康熙/部首區碼位（U+2E80–U+2FDF）：正常應接近 0，退化時會大量出現
    (2) 常用詞能否搜尋（--probe）：退化的 PDF 會搜不到「一致」「共用」

用法：
  python docs/static-build/check-pdf-fonts.py <pdf> [<pdf> ...]
  python docs/static-build/check-pdf-fonts.py --probe 一致 共用 <pdf>

離開碼：0 = 沒有大量 Type3；1 = 偵測到退化的字型嵌入。
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    print("需要 pypdf：pip install pypdf", file=sys.stderr)
    raise SystemExit(2)

# CJK 部首補充區（U+2E80–U+2FDF）與康熙部首（U+2F00–U+2FDF）：
# ToUnicode 退化時，漢字常被映射到這兩個區段的「部首」碼位。
RADICAL_LO, RADICAL_HI = 0x2E80, 0x2FDF
DEFAULT_PROBE = ["一致", "共用", "參數", "偏離"]


def inspect(path: Path, probe: list[str]) -> bool:
    reader = PdfReader(str(path))
    named: collections.Counter[str] = collections.Counter()
    type3 = 0
    for page in reader.pages:
        res = page.get("/Resources")
        if res is None or "/Font" not in res:
            continue
        for key in res["/Font"].keys():
            font = res["/Font"][key].get_object()
            base, subtype = font.get("/BaseFont"), font.get("/Subtype")
            if base is None:
                type3 += 1
            else:
                named[f"{base} ({subtype})"] += 1

    text = "".join((p.extract_text() or "") for p in reader.pages)
    radicals = sum(1 for ch in text if RADICAL_LO <= ord(ch) <= RADICAL_HI)

    print(f"=== {path.name} ===")
    print(f"  頁數 {len(reader.pages)}｜大小 {path.stat().st_size / 1e6:.1f} MB")
    print(f"  內嵌具名字型 {len(named)} 種；無具名 Type3 {type3} 個（多為數學式外框）")
    for name, count in named.most_common(8):
        print(f"     {count:5d}× {name}")
    print(f"  抽出文字 {len(text):,} 字元；康熙/部首區碼位 {radicals}")
    hits = []
    for word in probe:
        ok = word in text
        hits.append(ok)
        print(f"    搜尋 {word}：{'hit' if ok else 'MISS'}")
    cjk_named = [n for n in named if any(k in n for k in ("MingLiU", "JhengHei", "NotoSerif", "NotoSans",
                                                          "SourceHan", "SimSun", "YuMin", "Kai"))]
    print(f"  具名 CJK 字型：{', '.join(cjk_named) if cjk_named else '（無——中文很可能是 Type3 外框）'}")
    # 判定以「是否有具名 CJK 字型」＋「部首區碼位是否異常」為主；
    # 探測詞 MISS 可能只是該詞不在本文（例如技術報告沒有「偏離」），僅列為資訊。
    degraded = radicals > 300 or not cjk_named
    if not all(hits):
        print("  註：有探測詞 MISS——若該詞本來不在本文即屬正常，可用 --probe 指定本文確實出現的詞")
    print(f"  判定：{'退化（中文外框化／ToUnicode 不全）' if degraded else 'OK（中文字型嵌入；探測詞見上）'}")
    print()
    return degraded


def main() -> int:
    ap = argparse.ArgumentParser(description="檢查 PDF 字型嵌入品質")
    ap.add_argument("pdf", nargs="+", type=Path)
    ap.add_argument("--probe", nargs="*", default=DEFAULT_PROBE, help="要測試可否搜尋的詞")
    args = ap.parse_args()

    bad = [p for p in args.pdf if inspect(p, args.probe)]
    if bad:
        print("以下 PDF 的中文字型嵌入退化（建議改用靜態字型後重印）：")
        for p in bad:
            print(f"  - {p}")
        return 1
    print("全部 PDF 的字型嵌入皆正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
