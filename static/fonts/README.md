# Bundled fonts / 內嵌字體

Both typefaces are licensed under the **SIL Open Font License 1.1** — the
full license text ships next to each family (`OFL-Inter.txt`,
`OFL-JetBrainsMono.txt`). Neither upstream declares a Reserved Font Name,
so the rebuilt instances below keep the original family names.

Files are **modified builds**, not upstream binaries: the google/fonts
variable fonts were pinned to static instances with
`fontTools.varLib.instancer` (axes: Inter `wght`/`opsz`, JBM `wght`;
`OS/2.usWeightClass` and `macStyle` normalized). To regenerate:
`python logs/instance_fonts.py` (local script, gitignored).

| Files | Font | Copyright |
|---|---|---|
| Inter-Regular/Bold/ExtraBold/Black.ttf | [Inter](https://github.com/rsms/inter) | © 2020 The Inter Project Authors |
| JetBrainsMono-Medium/Bold.ttf | [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) | © 2020 The JetBrains Mono Project Authors |

內嵌字體說明：兩套字體皆為 SIL OFL 1.1 授權（全文見同目錄 `.txt`），上游均未宣告
Reserved Font Name，故重建後的靜態實例沿用原名。本目錄的 TTF 是修改版 — 由
google/fonts 的變量字體以 `fontTools.varLib.instancer` 釘軸生成，非上游原始檔。
