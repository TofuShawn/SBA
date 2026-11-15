#!/usr/bin/env python3
"""把 docs/static 的 HTML 報告印成正式 A4 PDF，並蓋上頁腳「文件名稱 · 第 X / Y 頁」。

原理（不需 JavaScript、不需 reportlab）：
  1. headless Chrome 把報告印成 A4 PDF（頁面尺寸與邊界由報告的 @page 決定）
  2. 同一支 Chrome 另印一份「每頁一張、只有頁腳」的覆蓋層 PDF
  3. pypdf 逐頁 merge_page，把頁腳疊到文件上

用法：
  python docs/static-build/make-pdf.py docs/static/AlphaZero_UTTT_Technical_Report.html
  python docs/static-build/make-pdf.py --no-footer <html> [<html> ...]

選項：
  --out DIR      輸出目錄（預設 <html 同層>/pdf）
  --chrome PATH  指定 Chrome 執行檔
  --title TEXT   覆寫頁腳左側文字（預設取 HTML 的 <title>）
  --no-footer    只產 PDF，不蓋頁腳
"""
from __future__ import annotations

import argparse
import html as html_mod
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_chrome(explicit: str | None) -> str:
    if explicit:
        if Path(explicit).exists():
            return explicit
        raise SystemExit(f"指定的 Chrome 不存在：{explicit}")
    env = os.environ.get("CHROME_PATH")
    if env and Path(env).exists():
        return env
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    found = shutil.which("chrome") or shutil.which("chrome.exe") or shutil.which("google-chrome")
    if found:
        return found
    raise SystemExit("找不到 Chrome，請用 --chrome 指定路徑（或設定 CHROME_PATH）")


def print_to_pdf(chrome: str, page_url: str, out_pdf: Path, profile: Path) -> None:
    cmd = [
        chrome,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        f"--user-data-dir={profile}",
        "--no-pdf-header-footer",
        f"--print-to-pdf={out_pdf}",
        page_url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not out_pdf.exists() or out_pdf.stat().st_size == 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise SystemExit("Chrome 列印失敗：\n  " + "\n  ".join(tail))


OVERLAY_TEMPLATE = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><style>
  @page {{ size: A4; margin: 0; }}
  html, body {{ margin: 0; padding: 0; }}
  .pg {{ position: relative; width: 210mm; height: 297mm; page-break-after: always; }}
  .pg:last-child {{ page-break-after: auto; }}
  .ft {{
    position: absolute; left: 18mm; right: 18mm; bottom: 9mm;
    border-top: 0.6pt solid #999; padding-top: 2mm;
    font-family: "Noto Sans TC", "Microsoft JhengHei", sans-serif;
    font-size: 8.5pt; color: #555; display: flex; justify-content: space-between;
  }}
</style></head><body>
{pages}
</body></html>
"""


def build_overlay_html(title: str, pages: int) -> str:
    safe = html_mod.escape(title)
    blocks = [
        f'<div class="pg"><div class="ft"><span>{safe}</span>'
        f"<span>第 {i} / {pages} 頁</span></div></div>"
        for i in range(1, pages + 1)
    ]
    return OVERLAY_TEMPLATE.format(pages="\n".join(blocks))


def page_count(pdf: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(pdf)).pages)


def stamp_footer(doc_pdf: Path, overlay_pdf: Path, out_pdf: Path) -> int:
    from pypdf import PdfReader, PdfWriter

    doc = PdfReader(str(doc_pdf))
    overlay = PdfReader(str(overlay_pdf))
    writer = PdfWriter()
    merged = 0
    for i, page in enumerate(doc.pages):
        if i < len(overlay.pages):
            page.merge_page(overlay.pages[i])
            merged += 1
        writer.add_page(page)
    with open(out_pdf, "wb") as fh:
        writer.write(fh)
    return merged


def doc_title(html_path: Path) -> str:
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"<title>(.*?)</title>", text, flags=re.S | re.I)
    title = re.sub(r"\s+", " ", m.group(1)).strip() if m else html_path.stem
    for sep in ("—", " - ", " · "):
        if sep in title:
            title = title.split(sep)[0].strip()
            break
    return title or html_path.stem


def main() -> int:
    ap = argparse.ArgumentParser(description="HTML 報告 → 正式 A4 PDF（含頁腳頁碼）")
    ap.add_argument("html", nargs="+", help="輸入的靜態 HTML")
    ap.add_argument("--out", default=None, help="輸出目錄（預設 <html 同層>/pdf）")
    ap.add_argument("--chrome", default=None, help="Chrome 執行檔路徑")
    ap.add_argument("--title", default=None, help="覆寫頁腳左側文字")
    ap.add_argument("--no-footer", action="store_true", help="不蓋頁腳頁碼")
    args = ap.parse_args()

    chrome = find_chrome(args.chrome)
    print(f"Chrome: {chrome}")

    for raw in args.html:
        src = Path(raw).resolve()
        if not src.exists():
            raise SystemExit(f"找不到輸入檔：{src}")
        out_dir = Path(args.out).resolve() if args.out else src.parent / "pdf"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = out_dir / f"{src.stem}.pdf"
        title = args.title or doc_title(src)

        with tempfile.TemporaryDirectory(prefix="sba-pdf-") as tmp:
            tmp_dir = Path(tmp)
            profile = tmp_dir / "profile"
            raw_pdf = tmp_dir / "raw.pdf"
            print_to_pdf(chrome, src.as_uri(), raw_pdf, profile)
            pages = page_count(raw_pdf)

            if args.no_footer:
                shutil.copyfile(raw_pdf, out_pdf)
                print(f"✓ {out_pdf.name}：{pages} 頁（未加頁腳）→ {out_pdf}")
                continue

            overlay_html = tmp_dir / "overlay.html"
            overlay_html.write_text(build_overlay_html(title, pages), encoding="utf-8")
            overlay_pdf = tmp_dir / "overlay.pdf"
            print_to_pdf(chrome, overlay_html.as_uri(), overlay_pdf, profile)
            merged = stamp_footer(raw_pdf, overlay_pdf, out_pdf)
            print(f"✓ {out_pdf.name}：{pages} 頁，頁腳 {merged} 頁（{title}）→ {out_pdf}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
