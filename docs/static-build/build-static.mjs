#!/usr/bin/env node
/**
 * build-static.mjs — 把 docs/ 的 HTML 報告轉成「零外部依賴」的靜態版。
 *
 * 為什麼要這樣做：Obsidian 的 HTML viewer 外掛把檔案放進沙箱 iframe，既不執行
 * 也不下載外部 <script>，所以 MathJax（數學）、Chart.js（圖表）、highlight.js
 * （程式碼上色）全部失效，讀者只看到 \( ... \) 原文。此腳本把三者預先渲染成
 * 純 HTML/CSS/SVG：
 *
 *   1. LaTeX（\(...\) 與 \[...\]）→ 自帶路徑的 inline SVG（無需 JS，也不需要字型）
 *      註：mathjax-full 3.2.2 沒有 MathML 輸出模組，SVG 是它在靜態場景的等價方案
 *   2. <pre><code> 區塊 → highlight.js 預先上色，主題 CSS 內嵌成 <style>
 *   3. Chart.js 圖表 → 手寫 inline SVG（資料見下方 CHARTS，逐文件設定）
 *   4. 移除全部 <script>（外部 CDN 與初始化）與外部 <link>
 *
 * 用法（首次需先安裝兩個套件）：
 *   cd E:\Project\SBA\docs\static-build
 *   npm install
 *   node build-static.mjs ..\AlphaZero_UTTT_Technical_Report.html ..\static\AlphaZero_UTTT_Technical_Report.html
 *
 * 沒有圖表的文件（例如 Transformer-AZ-AMAF-Design.html）把 CHARTS 留空即可，
 * 腳本遇到未定義的 canvas 會原樣保留並在結尾警告。
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import hljs from 'highlight.js';
import { mathjax } from 'mathjax-full/js/mathjax.js';
import { TeX } from 'mathjax-full/js/input/tex.js';
import { SVG } from 'mathjax-full/js/output/svg.js';
import { liteAdaptor } from 'mathjax-full/js/adaptors/liteAdaptor.js';
import { RegisterHTMLHandler } from 'mathjax-full/js/handlers/html.js';
import { AllPackages } from 'mathjax-full/js/input/tex/AllPackages.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/* ── 圖表資料：逐文件設定；沒有圖表的文件留空 {} ───────────────────────── */
const CHARTS = {
  chartLoss: {
    type: 'line', height: 340,
    xTitle: '自對弈局數', yTitle: '總損失 (Lp + Lv)',
    xMin: 0, xMax: 6000, yMin: 0, yMax: 3.6,
    series: [
      { label: '800 局訓練 (3.37 → 1.29)', color: '#3b82f6', dash: '7 5', points: [[0, 3.37], [800, 1.29]] },
      { label: '6000 局訓練 (2.846 → 0.094)', color: '#14b8a6', dash: '7 5', points: [[0, 2.846], [6000, 0.094]] },
    ],
  },
  chartEval: {
    type: 'line', height: 340,
    yTitle: '勝率 %', yMin: 60, yMax: 100,
    xLabels: ['400 局', '800 局', '6000 局'],
    series: [
      { label: '對隨機勝率 %（含 rollout, w=0.5）', color: '#3b82f6', dash: null, points: [[0, 97.5], [1, 90], [2, 96.3]] },
    ],
  },
  chartRollout: {
    type: 'bar', height: 320,
    yTitle: '勝率 %', yMin: 0, yMax: 100,
    bars: [
      { label: '純網路 w=0（400 局）', value: 76, color: '#94a3b8' },
      { label: '純網路 w=0（800 局）', value: 76, color: '#94a3b8' },
      { label: '含 rollout w=0.5（400 局）', value: 97.5, color: '#14b8a6' },
    ],
  },
  chartSpeed: {
    type: 'bar', height: 320,
    yTitle: '加速比（對串行）', yMin: 0, yMax: 3.2,
    bars: [
      { label: '1 worker（串行）', value: 1.0, color: '#94a3b8' },
      { label: '2 workers', value: 1.5, color: '#93c5fd' },
      { label: '4 workers', value: 2.43, color: '#3b82f6' },
      { label: '8 workers', value: 2.96, color: '#14b8a6' },
    ],
  },
  chartPilot: {
    type: 'bar', height: 340,
    yTitle: 'MCTS 得分率 %', yMin: 0, yMax: 100,
    bars: [
      { label: '32·RAVE', value: 70.0, color: '#3b82f6' },
      { label: '32·GRAVE', value: 75.0, color: '#14b8a6' },
      { label: '64·RAVE', value: 70.0, color: '#3b82f6' },
      { label: '64·GRAVE', value: 65.0, color: '#14b8a6' },
      { label: '128·RAVE', value: 35.0, color: '#3b82f6' },
      { label: '128·GRAVE', value: 50.0, color: '#14b8a6' },
      { label: '300·RAVE', value: 51.7, color: '#94a3b8' },
      { label: '300·GRAVE', value: 48.3, color: '#94a3b8' },
    ],
  },
};

/* ── 小工具 ────────────────────────────────────────────────────────────── */
const FONT = '"Noto Sans TC","Microsoft JhengHei","Segoe UI",sans-serif';
/* 圖表的格線／刻度／圖例改吃主題變數（var() 只能用於 style，不能用於呈現屬性），
   因此深色與紙本淺色切換時圖表會跟著換色；數列色則採兩主題通用的中性色。 */
const INK = 'var(--ink)', MUTED = 'var(--muted)', LINE = 'var(--line)';

const esc = (s) => String(s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const decode = (s) => s
  .replace(/&lt;/g, '<').replace(/&gt;/g, '>')
  .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&nbsp;/g, ' ')
  // 十進位／十六進位數值實體（例如 &#x27; 是單引號）：必須在 &amp; 之前解碼，
  // 否則 hljs 會拿到字面字串再被轉義，輸出就會出現 &amp;#x27; 這種破字。
  .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => String.fromCodePoint(parseInt(hex, 16)))
  .replace(/&#(\d+);/g, (_, dec) => String.fromCodePoint(Number(dec)))
  .replace(/&amp;/g, '&');

// 粗略估算文字寬度（CJK 算 1em、其餘算 0.55em），只用來置中與判斷是否旋轉
const textWidth = (s, size) => [...s].reduce((w, c) => w + (/[\u2e80-\u9fff\uff00-\uffef]/.test(c) ? size : size * 0.55), 0);

const tickLabel = (v) => {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 100) return String(Math.round(v));
  if (a >= 1) return String(Number(v.toFixed(2)));
  return String(Number(v.toFixed(3)));
};

/* ── 圖表 → inline SVG ─────────────────────────────────────────────────── */
function svgOpen(spec, id) {
  return `<svg class="chart-svg" data-chart="${esc(id)}" viewBox="0 0 720 ${spec.height}" ` +
    `role="img" aria-label="${esc(spec.yTitle || id)}" ` +
    `style="width:100%;max-width:720px;height:auto;display:block;margin:0 auto;font-family:${FONT}">`;
}

function legendBlock(spec, seriesMeta, plotBottom) {
  if (spec.type === 'bar' || spec.legend === false) return { svg: '', height: 0 };
  const items = seriesMeta.map((s) => ({ color: s.color, label: s.label }));
  const gap = 24, size = 11.5;
  const widths = items.map((it) => textWidth(it.label, size) + 26);
  const total = widths.reduce((a, b) => a + b, 0) + gap * (items.length - 1);
  let x = Math.max(24, (720 - total) / 2);
  const y = plotBottom + 40;
  let svg = '';
  items.forEach((it, i) => {
    svg += `<line x1="${x}" y1="${y - 4}" x2="${x + 18}" y2="${y - 4}" stroke="${it.color}" stroke-width="2" stroke-dasharray="6 4"/>`;
    svg += `<circle cx="${x + 9}" cy="${y - 4}" r="3" fill="${it.color}"/>`;
    svg += `<text x="${x + 24}" y="${y}" font-size="${size}" style="fill:${MUTED}">${esc(it.label)}</text>`;
    x += widths[i] + gap;
  });
  return { svg, height: 34 };
}

function lineChartSvg(spec, id) {
  const M = { l: 68, r: 28, t: 16, b: 46 };
  const W = 720, H = spec.height;
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const xMin = spec.xMin ?? 0;
  const xMax = spec.xMax ?? Math.max(...spec.series.flatMap((s) => s.points.map((p) => p[0])));
  const yMin = spec.yMin ?? 0, yMax = spec.yMax ?? Math.max(...spec.series.flatMap((s) => s.points.map((p) => p[1]))) * 1.1;
  const sx = (v) => M.l + ((v - xMin) / (xMax - xMin || 1)) * pw;
  const sy = (v) => M.t + ph - ((v - yMin) / (yMax - yMin || 1)) * ph;

  let g = '';
  // 水平格線與 y 刻度
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const v = yMin + ((yMax - yMin) * i) / yTicks;
    const y = sy(v);
    g += `<line x1="${M.l}" y1="${y.toFixed(1)}" x2="${M.l + pw}" y2="${y.toFixed(1)}" style="stroke:${LINE}" stroke-width="1"/>`;
    g += `<text x="${M.l - 10}" y="${(y + 4).toFixed(1)}" font-size="11.5" style="fill:${MUTED}" text-anchor="end">${tickLabel(v)}</text>`;
  }
  // x 刻度（類別或數值）
  if (spec.xLabels) {
    const n = spec.xLabels.length;
    spec.xLabels.forEach((lab, i) => {
      const x = n === 1 ? M.l + pw / 2 : M.l + (pw * i) / (n - 1);
      g += `<text x="${x.toFixed(1)}" y="${M.t + ph + 20}" font-size="11.5" style="fill:${MUTED}" text-anchor="middle">${esc(lab)}</text>`;
    });
  } else {
    const xTicks = 6;
    for (let i = 0; i <= xTicks; i++) {
      const v = xMin + ((xMax - xMin) * i) / xTicks;
        g += `<text x="${sx(v).toFixed(1)}" y="${M.t + ph + 20}" font-size="11.5" style="fill:${MUTED}" text-anchor="middle">${tickLabel(v)}</text>`;
    }
  }
  // 軸線
  g += `<line x1="${M.l}" y1="${M.t}" x2="${M.l}" y2="${M.t + ph}" style="stroke:${LINE}" stroke-width="1.5"/>`;
  g += `<line x1="${M.l}" y1="${M.t + ph}" x2="${M.l + pw}" y2="${M.t + ph}" style="stroke:${LINE}" stroke-width="1.5"/>`;
  // 數列
  for (const s of spec.series) {
    const dash = s.dash ? ` stroke-dasharray="${s.dash}"` : '';
    const pts = s.points.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(' ');
    if (s.points.length > 1) {
      g += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="2.5"${dash} stroke-linejoin="round"/>`;
    }
    for (const [x, y] of s.points) {
      g += `<circle cx="${sx(x).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="4" fill="${s.color}" style="stroke:var(--surface)" stroke-width="1.5"/>`;
    }
  }
  // 軸標題
  if (spec.xTitle) g += `<text x="${M.l + pw / 2}" y="${H - 6}" font-size="12" style="fill:${MUTED}" text-anchor="middle">${esc(spec.xTitle)}</text>`;
  if (spec.yTitle) g += `<text x="16" y="${M.t + ph / 2}" font-size="12" style="fill:${MUTED}" text-anchor="middle" transform="rotate(-90 16 ${M.t + ph / 2})">${esc(spec.yTitle)}</text>`;

  const lg = legendBlock(spec, spec.series, M.t + ph);
  return { svg: svgOpen(spec, id) + g + lg.svg + '</svg>', legend: lg.height };
}

function barChartSvg(spec, id) {
  const M = { l: 68, r: 28, t: 18, b: 60 };
  const W = 720, H = spec.height;
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const yMin = spec.yMin ?? 0;
  const yMax = spec.yMax ?? Math.max(...spec.bars.map((b) => b.value)) * 1.15;
  const sy = (v) => M.t + ph - ((v - yMin) / (yMax - yMin || 1)) * ph;
  const slot = pw / spec.bars.length;
  const bw = Math.min(96, slot * 0.55);

  let g = '';
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const v = yMin + ((yMax - yMin) * i) / yTicks;
    const y = sy(v);
    g += `<line x1="${M.l}" y1="${y.toFixed(1)}" x2="${M.l + pw}" y2="${y.toFixed(1)}" style="stroke:${LINE}" stroke-width="1"/>`;
    g += `<text x="${M.l - 10}" y="${(y + 4).toFixed(1)}" font-size="11.5" style="fill:${MUTED}" text-anchor="end">${tickLabel(v)}</text>`;
  }
  spec.bars.forEach((b, i) => {
    const cx = M.l + slot * i + slot / 2;
    const x = cx - bw / 2;
    const y = sy(b.value);
    const h = M.t + ph - y + ((yMin < 0) ? -sy(0) + sy(yMin) : 0);
    const r = Math.min(6, h / 2);
    // 只有上方兩角圓角（Chart.js borderRadius 的視覺）
    g += `<path d="M${x},${y + h} L${x},${y + r} Q${x},${y} ${x + r},${y} L${x + bw - r},${y} Q${x + bw},${y} ${x + bw},${y + r} L${x + bw},${y + h} Z" fill="${b.color}"/>`;
    // x 標籤：太寬就旋轉
    const tw = textWidth(b.label, 11.5);
    if (tw > slot - 6) {
      g += `<text x="${cx.toFixed(1)}" y="${M.t + ph + 18}" font-size="11.5" style="fill:${MUTED}" text-anchor="end" transform="rotate(-24 ${cx.toFixed(1)} ${M.t + ph + 18})">${esc(b.label)}</text>`;
    } else {
      g += `<text x="${cx.toFixed(1)}" y="${M.t + ph + 20}" font-size="11.5" style="fill:${MUTED}" text-anchor="middle">${esc(b.label)}</text>`;
    }
  });
  g += `<line x1="${M.l}" y1="${M.t}" x2="${M.l}" y2="${M.t + ph}" style="stroke:${LINE}" stroke-width="1.5"/>`;
  g += `<line x1="${M.l}" y1="${M.t + ph}" x2="${M.l + pw}" y2="${M.t + ph}" style="stroke:${LINE}" stroke-width="1.5"/>`;
  if (spec.yTitle) g += `<text x="16" y="${M.t + ph / 2}" font-size="12" style="fill:${MUTED}" text-anchor="middle" transform="rotate(-90 16 ${M.t + ph / 2})">${esc(spec.yTitle)}</text>`;
  return { svg: svgOpen(spec, id) + g + '</svg>' };
}

/* ── 主流程 ────────────────────────────────────────────────────────────── */
const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error('用法：node build-static.mjs <in.html> <out.html>');
  process.exit(1);
}
const stats = { mathInline: 0, mathDisplay: 0, code: 0, charts: 0, scriptsRemoved: 0, linksInlined: 0, warnings: [] };
let html = fs.readFileSync(input, 'utf8');
const srcBytes = Buffer.byteLength(html);

/* 1. 保護 script / style / pre / code：數學轉換要跳過這些區域
      （與瀏覽器端 MathJax 的 skipHtmlTags 預設一致） */
const vault = [];
const stash = (block) => { vault.push(block); return `\u0000${vault.length - 1}\u0000`; };
let guarded = html
  .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, (m) => stash(m))
  .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, (m) => stash(m))
  .replace(/<pre\b[^>]*>[\s\S]*?<\/pre>/gi, (m) => stash(m))
  .replace(/<code\b[^>]*>[\s\S]*?<\/code>/gi, (m) => stash(m));

/* 2. LaTeX → MathML */
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: AllPackages });
const svgJax = new SVG({ fontCache: 'local' }); // local：每條公式自帶 <defs>，不依賴全域快取
const doc = mathjax.document('', { InputJax: tex, OutputJax: svgJax });

const toMath = (source, display) => {
  const container = adaptor.outerHTML(doc.convert(decode(source).trim(), { display, em: 16, ex: 8, containerWidth: 1200 }));
  const m = container.match(/<svg[\s\S]*<\/svg>/);
  const svg = (m ? m[0] : container).replace('<svg', '<svg class="math-svg"');
  return display
    ? `<div class="math-display" style="text-align:center;margin:1.1em 0">${svg}</div>`
    : svg;
};

guarded = guarded.replace(/\\\[([\s\S]+?)\\\]/g, (_, body) => { stats.mathDisplay++; return toMath(body, true); });
guarded = guarded.replace(/\\\(([\s\S]+?)\\\)/g, (_, body) => { stats.mathInline++; return toMath(body, false); });

/* 3. 還原保護區，同時把 <pre><code> 預先上色 */
html = guarded.replace(/\u0000(\d+)\u0000/g, (_, i) => vault[Number(i)]);

html = html.replace(/<pre\b([^>]*)><code\b([^>]*)>([\s\S]*?)<\/code><\/pre>/gi, (m, preAttrs, codeAttrs, body) => {
  const langMatch = codeAttrs.match(/language-([a-z0-9]+)/i);
  const lang = langMatch ? langMatch[1].toLowerCase() : null;
  const raw = decode(body);
  let value;
  if (lang && hljs.getLanguage(lang)) {
    value = hljs.highlight(raw, { language: lang }).value;
  } else {
    value = hljs.highlightAuto(raw).value;
    stats.warnings.push(`程式碼區塊沒有語言標記，已用自動偵測上色（${value.length} 字元）`);
  }
  stats.code++;
  return `<pre${preAttrs}><code class="hljs${lang ? ` language-${lang}` : ''}">${value}</code></pre>`;
});

/* 4. 圖表 → inline SVG（CHARTS 是跨文件共用的登記表，文件只會用到其中幾張） */
for (const [id, spec] of Object.entries(CHARTS)) {
  const re = new RegExp(`<canvas\\b[^>]*id="${id}"[^>]*>\\s*</canvas>`, 'i');
  if (!re.test(html)) continue; // 這份文件沒有這張圖表，正常
  const built = spec.type === 'bar' ? barChartSvg(spec, id) : lineChartSvg(spec, id);
  html = html.replace(re, built.svg);
  stats.charts++;
}
for (const m of html.matchAll(/<canvas\b[^>]*id="([^"]+)"[^>]*>/gi)) {
  stats.warnings.push(`canvas #${m[1]} 沒有對應的 CHARTS 設定，原樣保留（仍需要 Chart.js）`);
}

/* 5a. 相對路徑樣式表（assets/endfield-report.css 等）→ 內嵌，維持零外部依賴 */
const inputDir = path.dirname(path.resolve(input));
html = html.replace(/<link\b[^>]*>/gi, (tag) => {
  if (!/\brel\s*=\s*"stylesheet"/i.test(tag)) return tag;
  const href = (tag.match(/\bhref\s*=\s*"([^"]+)"/i) || [])[1];
  if (!href || /^https?:/i.test(href)) return tag; // 外部連結交由下一步處理
  const abs = path.resolve(inputDir, href);
  if (!fs.existsSync(abs)) {
    stats.warnings.push(`找不到相對樣式表：${href}`);
    return tag;
  }
  stats.linksInlined++;
  return `<style>\n/* ${href}（建置時內嵌） */\n${fs.readFileSync(abs, 'utf8')}\n</style>`;
});

/* 5. 外部 CSS → 內嵌 */
const hljsCss = [
  path.join(__dirname, 'node_modules', 'highlight.js', 'styles', 'github-dark.min.css'),
].find((p) => fs.existsSync(p));
html = html.replace(/<link\b[^>]*href="https?:\/\/[^"]*highlight[^"]*"[^>]*>/gi, () => {
  if (!hljsCss) { stats.warnings.push('找不到 highlight.js 主題 CSS，程式碼區塊會沒有顏色'); return ''; }
  stats.linksInlined++;
  return `<style>\n/* highlight.js github-dark（建置時內嵌，原本從 CDN 載入） */\n${fs.readFileSync(hljsCss, 'utf8')}\n</style>`;
});

/* 6. 移除全部外部 script 與已失效的初始化 script；非預期的 inline script 保留並警告 */
html = html.replace(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi, (m, attrs, body) => {
  const isExternal = /src\s*=\s*"https?:/i.test(attrs);
  const known = /window\.MathJax|hljs\.highlightAll|new Chart\(|Chart\.defaults/.test(body);
  if (isExternal || !body.trim() || known) { stats.scriptsRemoved++; return ''; }
  stats.warnings.push(`保留了一個未預期的 inline script（${body.trim().slice(0, 60)}…）`);
  return m;
});

/* 7. 收尾：剩餘的外部資源提醒＋統計 */
for (const m of html.matchAll(/<(?:script|link)\b[^>]*\b(?:src|href)="(https?:\/\/[^"]+)"/gi)) {
  stats.warnings.push(`仍有外部資源：${m[1]}`);
}

fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, html, 'utf8');

console.log(`輸入：${path.relative(process.cwd(), input)}（${(srcBytes / 1024).toFixed(0)} KB）`);
console.log(`輸出：${path.relative(process.cwd(), output)}（${(Buffer.byteLength(html) / 1024).toFixed(0)} KB）`);
console.log(`行內公式 → SVG：${stats.mathInline} 條`);
console.log(`行間公式 → SVG：${stats.mathDisplay} 條`);
console.log(`程式碼區塊預先上色：${stats.code} 個`);
console.log(`圖表 → inline SVG：${stats.charts} 張`);
console.log(`移除 script：${stats.scriptsRemoved} 個；內嵌 CSS：${stats.linksInlined} 份`);
if (stats.warnings.length) {
  console.log('\n警告：');
  for (const w of stats.warnings) console.log(`  - ${w}`);
} else {
  console.log('\n沒有警告：輸出已無任何外部依賴。');
}
