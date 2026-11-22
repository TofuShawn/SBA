# Development guide / 開發指南

Everything a human needs to work on this repo in one screen. The canonical
entry point is the root task runner — **`python dev.py`** lists every task;
each one wraps the exact invocation shape that works on this toolchain
(Windows Python + WSL mingw gcc + WSL cargo), so nobody re-derives
command-line quoting from scratch.

## Prerequisites / 環境需求

- **Python 3.12** with a `.venv` at the repo root (`pip install -r
  requirements.txt -r requirements-dev.txt`; torch optional, see README)
- **WSL** with `mingw-w64-gcc` (the C cross-compiler → `sba_engine.dll`)
  and **rustup** with the `x86_64-pc-windows-gnu` target (→ `sba_game.dll`
  + the sba-core exes) — or any local gcc/cargo; `src/tools/build.py` probes
  PATH first and falls back to WSL
- ruff lives **only in `.venv`** — always via `python -m ruff` or dev.py

## Branch model / 分支模型

- **`main`** — the release line. CI runs on push; `v*` tags trigger the
  release workflow (written notes from `docs/releases/<tag>.md`).
- **`dev`** — the integration branch. CI also runs on push, so work lands
  here CI-green before any merge to main. Feature branches PR into `dev`.

## Tasks / 常用命令

| Task | What it does |
|---|---|
| `python dev.py build` | compile all natives: sba_engine.dll (C), sba_game.dll (Rust state layer), the sba-core bench binary |
| `python dev.py test` | the pytest suite (args forwarded; count via `pytest tests --collect-only -q`) |
| `python dev.py lint` | ruff check (F+E9+E7+B+SIM+PERF, see pyproject.toml) |
| `python dev.py format` | ruff format --check (CI gate; `format-write` applies) |
| `python dev.py clippy` | cargo clippy -D warnings on sba-game |
| `python dev.py clippy-core` | cargo clippy -D warnings on sba-core (CI Linux gate, default wgpu) |
| `python dev.py cgate` | C warnings gate: `-Wall -Wextra -Werror` over all five C files |
| `python dev.py bench-rust` | the Rust state-layer A/B harness |
| `python dev.py bench-selfplay` | self-play A/B: mp.Pool path vs rust exe (pos/s, speedup; D56) |
| `python dev.py test-rust` | cargo test: sba-game (default) + sba-core (cpu backend, no GPU) |

A "ready to commit" state = `lint` + `format` + `clippy` + `clippy-core`
+ `cgate` + `test-rust` + `test` all green. CI runs the same gates (plus
the CLI entry smoke) on both ubuntu and windows.

## Layout / 佈局

One rule since D60: **src/ = everything by function** (D36's flat src/
amended — the former native/ dissolved into it). Since D61 the root
Python modules group into one-level function packages; `config.py` stays
the root leaf anchor.

- `src/core/` — the game state layer: `game.py` (rules + bitboards)
- `src/engines/` — the AI search stack: `ai.py` (the facade), `mcts.py`,
  `minimax.py`, `cbridge.py`, `cengine.py` (the ctypes boundary), plus
  the five C engines (`sba_*.c` + `sba_ultt.h`) and the `sba_engine.dll`
  build.py compiles from them
- `src/az/` — the AlphaZero stack: `alphazero.py` (training +
  self-play), `rustselfplay.py`, `rustbridge.py`, plus the `sba_game.dll`
  state-layer cdylib build.py copies there
- `src/analysis/` — read-only inspection: `analysis.py`, `tablebase.py`,
  `books.py`
- `src/ui/` — the front ends: `qtui.py` (PySide6), `tui.py` (Textual),
  `tui_tools.py` (bench/train TUIs), `cli_output.py` (shared rich
  console)
- `src/session/` — `SBA.py`, the entry point behind the root `sba.py`
  shim: session state, CLI routing, self-tests
- `src/sba-game/`, `src/sba-core/` — the two Rust crates; the root
  `Cargo.toml` is a minimal workspace (members + profiles), one
  `Cargo.lock`, cargo output in the root `target/`
- `src/tools/` — the dev harnesses: `build.py` (C DLL + cdylib copy),
  `bench_rust.py`, `bench_selfplay.py`, `parity_search.py`
- `tests/` — the pytest suite; DLL-gated files skip themselves cleanly
  (count via `pytest tests --collect-only -q`)
- `docs/decisions.md` — every design decision, append-only (D1…)
- `docs/releases/<tag>.md` — the release notes each tag publishes
- `logs/` — gitignored scratch (bench artifacts, spike scripts)

The human contract is unchanged: `python dev.py` (from the repo root)
still runs every task.
