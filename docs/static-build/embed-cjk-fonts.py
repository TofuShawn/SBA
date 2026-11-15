#!/usr/bin/env python3
"""把 CJK variable font 實例化、子集化，並以 base64 內嵌進（自足版）HTML。

為什麼需要：本專案的靜態版 HTML 必須「零外部依賴」，但 Windows 上可用的
思源宋體／思源黑體是 *variable font*（NotoSerifTC-VF.ttf / NotoSansTC-VF.ttf），
而 Chromium 的 PDF 後端無法嵌入 VF → 退化成 Type3 外框（中文可看但搜尋／複製退化）。
Adobe 的 CID-keyed CFF OTF 更進一步連 Chrome 都不採用（DirectWrite 限制）。

解法：用 fontTools 把 VF 固定成 static 實例（TrueType 外框）→ 子集化到文件實際用到的
字元 → 存成 WOFF2 → base64 內嵌。這樣 PDF 能正常嵌入（Type0）、可搜尋，且檔案自足。

用法：
  python docs/static-build/embed-cjk-fonts.py <in.html> <out.html> \
      [--serif <VF.ttf>] [--sans <VF.ttf>] [--weights 400,700] [--label 前綴]

驗收（印完 PDF 後）：
  python docs/static-build/check-pdf-fonts.py <out.pdf>
"""
from __future__ import annotations

import argparse
import base64
import html as html_mod
import io
import re
import sys
from pathlib import Path

try:
    from fontTools import subset, ttLib
    from fontTools.varLib import instancer
except ImportError:  # pragma: no cover
    print("需要 fontTools（pip install fonttools brotli）", file=sys.stderr)
    raise SystemExit(2)

DEFAULT_SERIF = r"C:\Windows\Fonts\NotoSerifTC-VF.ttf"
DEFAULT_SANS = r"C:\Windows\Fonts\NotoSansTC-VF.ttf"

# 除了文件實際用到的字元，額外保留這些常用區段，避免內嵌後缺字：
SAFETY_RANGES = [
    (0x0020, 0x007E),  # ASCII
    (0x00A0, 0x00FF),  # Latin-1
    (0x0391, 0x03C9),  # Greek（β κ τ σ μ ρ π …）
    (0x2010, 0x203A),  # 連字號、引號、破折號
    (0x2044, 0x2044),  # fraction slash
    (0x2070, 0x209C),  # 上下標
    (0x20AC, 0x20AC),  # €
    (0x2190, 0x21FF),  # 箭頭（→ ⇔）
    (0x2200, 0x22FF),  # 數學運算子（≥ ≤ ≠ ∈ √ ∑）
    (0x2300, 0x23FF),  # 技術符號（⌈ ⌉）
    (0x2460, 0x24FF),  # 帶圈數字 ① ② ③
    (0x2500, 0x257F),  # 框線（─）
    (0x25A0, 0x25FF),  # 幾何圖形（■ ●）
    (0x2600, 0x26FF),  # 雜項符號
    (0x3000, 0x303F),  # CJK 標點（。、：「」）
    (0x4E00, 0x9FFF),  # 由文字擷取覆蓋，這裡不展開（避免整塊字集）
    (0xFF00, 0xFFEF),  # 全形
]


def used_characters(html_text: str) -> str:
    """從 HTML 取出實際會顯示的字元（含 safety ranges）。"""
    body = re.sub(r"<(script|style)\b.*?</\1>", " ", html_text, flags=re.S | re.I)
    body = re.sub(r"<[^>]+>", " ", body)
    body = html_mod.unescape(body)
    chars = set(body)
    for lo, hi in SAFETY_RANGES:
        if lo == 0x4E00:
            continue
        chars.update(chr(c) for c in range(lo, hi + 1))
    return "".join(sorted(chars))


def static_instance(vf_path: Path, weight: int, tmpdir: Path) -> Path:
    """把 variable font 固定成指定字重的 static TTF。"""
    font = ttLib.TTFont(str(vf_path))
    axes = {a.axisTag: a for a in font["fvar"].axes}
    loc = {"wght": weight} if "wght" in axes else {}
    try:
        inst = instancer.instantiateVariableFont(font, loc, inplace=False, updateFontNames=True)
    except TypeError:  # 舊版簽名
        inst = instancer.instantiateVariableFont(font, loc, inplace=False)
    out = tmpdir / f"{vf_path.stem}-w{weight}.ttf"
    inst.save(str(out))
    return out


def subset_woff2(ttf_path: Path, text: str, tmpdir: Path) -> tuple[bytes, str]:
    """子集化並輸出 woff2 bytes；回傳 (bytes, 字型內部家族名)。"""
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.hinting = False
    opts.glyph_names = False
    opts.name_IDs = ["*"]
    opts.name_legacy = True
    opts.notdef_outline = True
    opts.drop_tables += ["DSIG"]
    font = subset.load_font(str(ttf_path), opts)
    sub = subset.Subsetter(opts)
    sub.populate(text=text)
    sub.subset(font)
    buf = io.BytesIO()
    opts.flavor = "woff2"
    font.flavor = "woff2"
    font.save(buf)
    family = ""
    for rec in font["name"].names:
        if rec.nameID == 1:
            try:
                family = rec.toUnicode()
            except Exception:
                pass
            break
    return buf.getvalue(), family


def main() -> int:
    ap = argparse.ArgumentParser(description="把 CJK VF 實例化／子集化並內嵌進 HTML")
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--serif", default=DEFAULT_SERIF, help="正文 CJK（宋體）VF 檔")
    ap.add_argument("--sans", default=DEFAULT_SANS, help="標題 CJK（黑體）VF 檔")
    ap.add_argument("--weights", default="400,700")
    ap.add_argument("--label", default="SBA", help="CSS 家族名前綴")
    ap.add_argument("--tmp", default=None, help="暫存目錄（預設與輸出同層）")
    args = ap.parse_args()

    weights = [int(w) for w in args.weights.split(",") if w.strip()]
    html_text = args.src.read_text(encoding="utf-8")
    chars = used_characters(html_text)
    print(f"輸入 {args.src.name}｜擷取字元 {len(chars)} 個（含常用區段）")

    tmpdir = Path(args.tmp) if args.tmp else args.dst.parent
    tmpdir.mkdir(parents=True, exist_ok=True)
    faces: list[str] = []
    total = 0
    for kind, path, css_name in (("serif", Path(args.serif), f"{args.label} Serif TC"),
                                 ("sans", Path(args.sans), f"{args.label} Sans TC")):
        if not path.exists():
            print(f"  [SKIP] 找不到 {path}")
            continue
        for weight in weights:
            ttf = static_instance(path, weight, tmpdir)
            data, family = subset_woff2(ttf, chars, tmpdir)
            b64 = base64.b64encode(data).decode("ascii")
            total += len(b64)
            print(f"  ✓ {kind:5s} w{weight}  {path.name} → {len(data)/1024:7.1f} KiB "
                  f"(base64 {len(b64)/1024:7.1f} KiB)  內部名={family or '?'}")
            faces.append(
                "@font-face{font-family:'%s';font-style:normal;font-weight:%d;font-display:block;"
                "src:url(data:font/woff2;base64,%s) format('woff2')}" % (css_name, weight, b64)
            )

    css = "<style>\n/* 內嵌 CJK（實例化＋子集化）：%s */\n%s\n" % (args.label, "\n".join(faces))
    css += """
/* 正文：英文 Times New Roman＋中文宋體（內嵌） */
body, .page, .page p, .page li, .page td, .page th, .page .sub, .page .small, .page .cap,
.page .tbl-cap, .page .meta, .page .eq, .page .formula, .page blockquote {
  font-family: "Times New Roman", "%(label)s Serif TC", "Source Han Serif TC", "Noto Serif TC", "PMingLiU", serif !important;
}
/* 標題：英文 Times New Roman Bold＋中文黑體（內嵌） */
.page h1, .page h2, .page h3, .page h4, .page h5, .page h6, .page .kicker, .page .toc h2,
.page .callout .t, .page .chip {
  font-family: "Times New Roman", "%(label)s Sans TC", "Source Han Sans TC", "Noto Sans TC", "Microsoft JhengHei", sans-serif !important;
  font-weight: 700;
}
code, pre, .kbd, .page code, .page pre {
  font-family: Consolas, "Cascadia Mono", "Courier New", monospace !important;
}
</style>
""" % {"label": args.label}

    i = html_text.index("</head>")
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(html_text[:i] + css + html_text[i:], encoding="utf-8")
    print(f"\n輸出 {args.dst}｜HTML {args.dst.stat().st_size/1024/1024:.2f} MiB"
          f"（內嵌字型 base64 共 {total/1024/1024:.2f} MiB）")
    for leftover in tmpdir.glob("*-w*.ttf"):
        try:  # Windows 上檔案可能仍被 fontTools 握著，清不掉不影響輸出
            leftover.unlink(missing_ok=True)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
