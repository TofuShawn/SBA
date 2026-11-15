#!/usr/bin/env node
/**
 * KaTeX 相容性稽核：把文件裡「每一條」LaTeX 公式丟給 KaTeX 實跑，列出失敗者。
 *
 * 為什麼需要：本專案的公式目前由 MathJax 轉成 **SVG 向量外框**，沒有文字層——
 * 任何靠文字抽取或 OCR 的閱讀管線（含 AI 審閱）都看不到公式，會被誤判為「公式缺失」。
 * KaTeX 輸出是真文字（HTML+CSS），可被搜尋／抽取／朗讀，但必須先確認相容性。
 *
 * 用法：
 *   npm i katex                       # 需要一次網路安裝（放 docs/static-build/）
 *   node docs/static-build/katex-audit.mjs docs/Transformer-AZ-AMAF-Design.html
 *   node docs/static-build/katex-audit.mjs docs/AlphaZero_UTTT_Technical_Report.html --limit 5
 *
 * 離開碼：0 = 全部可編譯；1 = 有失敗（會列出公式與錯誤訊息）。
 */
import { readFileSync } from "node:fs";
import katex from "katex";

const [file, ...rest] = process.argv.slice(2);
if (!file) {
  console.error("用法：node katex-audit.mjs <html> [--limit N] [--show-all]");
  process.exit(2);
}
const limitArg = rest.indexOf("--limit");
const limit = limitArg >= 0 ? Number(rest[limitArg + 1]) : 8;

const html = readFileSync(file, "utf8");
const blocks = [
  ...html.matchAll(/\\\(([\s\S]*?)\\\)/g),
  ...html.matchAll(/\\\[([\s\S]*?)\\\]/g),
].map((m) => m[1]);

console.log(`檔案：${file}`);
console.log(`公式數：${blocks.length}`);
const failures = [];
for (const [i, tex] of blocks.entries()) {
  try {
    katex.renderToString(tex, { throwOnError: true, displayMode: false, strict: false });
  } catch (err) {
    failures.push({ i, tex: tex.replace(/\s+/g, " ").slice(0, 160), msg: String(err.message || err) });
  }
}

if (failures.length === 0) {
  console.log("✅ 全部公式皆可被 KaTeX 編譯（含 aligned / cases / \\text{中文}）");
} else {
  console.log(`❌ 失敗 ${failures.length} 條（顯示前 ${Math.min(limit, failures.length)} 條）：`);
  for (const f of failures.slice(0, limit)) {
    console.log(`  #${f.i}: ${f.tex}`);
    console.log(`      → ${f.msg}`);
  }
}

// 額外檢查：\text{} 內含 CJK 的數量（KaTeX 可編譯，但要確認 CJK fallback 字型）
const cjkInText = blocks.filter((b) => /\\text\{[^}]*[\u4e00-\u9fff]/.test(b)).length;
console.log(`含 CJK 的 \\text{} 公式：${cjkInText} 條（需確認 CJK fallback 字型）`);
process.exit(failures.length ? 1 : 0);
