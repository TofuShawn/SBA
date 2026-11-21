# sba-core Phase A — Rust self-play engine: design spec

- **Date**: 2026-11-21
- **Status**: approved design; implementation plan is the next artifact
- **Lineage**: D55 (sba-core consolidation) + its amendment (burn/wgpu inference);
  the burn AZNet spike already on dev proves inference parity 1.7e-07 at
  8.3 ms / batch-256.

## 1. Goals

Move AlphaZero self-play (search + NN inference) out of Python into a pure-Rust
executable, so the training hot path stops being Python- and per-position-NN-bound
(~85 ms/position today ≈ one GPU forward per position, batch fill near zero).

- Python shrinks to: safetensors export → exe invocation → record loading →
  torch optimize step. Training itself stays Python/torch in this phase.
- The two training UIs keep working with zero mandatory changes (the event
  plane is preserved; see §5.4).
- End-to-end self-play throughput **≥ 4x** current at campaign settings
  (sims 800, threads 8, same hardware): ~85 ms → ≤ 21 ms per position.

**Non-goals (explicit)**

- No CUDA support — the feature does not exist in this crate.
- No runtime backend dispatch — the backend is a compile-time feature choice.
- No virtual loss / single-tree parallelism — root-parallel is sufficient (YAGNI).
- No Rust-side training and no Rust-side eval gauge (eval stays Python; it runs
  once per eval_every games and is not a hotspot).
- No cross-platform release packaging this phase (build matrix + CI only;
  shipping per-OS binaries rides the next release touch).

## 2. Approved decisions

| Question | Decision |
|---|---|
| Rust↔Python seam shape | standalone exe (`sba_selfplay`): records on disk, JSONL events on stdout |
| Phase A acceptance scope | exe + `--train-az` switch + train-tui rewiring, all inside Phase A |
| Throughput bar | ≥4x end-to-end at campaign settings |
| Backend matrix | wgpu (default) + cpu (feature); **no CUDA** |
| Project shape | single crate `native/sba-core`, multiple bins; no separate training project |

## 3. Architecture

```
sba_selfplay.exe --weights W.safetensors --games 32 --sims 800
                 --threads 8 --seed-base S --out games/r001/

  game threads ×8 (root-parallel: one tree per game, one thread per tree)
    │  leaf request (planes [4,9,9] f32, req_id)
    ▼
  mpsc channel ──► inference worker (single; owns one wgpu device + queue)
    │                batch → pad to bucket {16, 64, 256} → AzNet.forward
    │◄─── (policy [81], value) reply per req_id
    ▼
  PUCT select/expand/backprop (Dirichlet 0.25 root noise, temp schedule,
  per-game RNG)
    ▼
  game over → write g%05d.safetensors + JSONL progress events → stdout
```

1. **Root-parallel, no virtual loss.** Each game thread runs its tree
   serially; while it waits for a batch result the other threads' requests
   fill the batch (8 threads × sims 800 keeps fill healthy). Single-tree
   parallelism's quality gain does not pay for its implementation and
   debugging cost.
2. **Zero FFI inside the exe.** `sba-core` depends on the `sba-game` crate
   via a path dependency and uses `UltimateState` directly. The live-handle
   registry exists for the FFI boundary and is not involved in-process; tree
   nodes own their states, `clone_apply` is a plain Rust call.
3. **Batch buckets** `{16, 64, 256}` — pad with zeros to the next bucket,
   slice the outputs back. Same lesson as the Python side's pad-to-exact-warm-
   shape work (D40): wgpu pipeline caches love fixed shapes.
4. **Per-game determinism**: seed = `--seed-base` + game index; Dirichlet and
   move sampling draw from the per-game RNG. Any game is replayable from its
   seed. Determinism holds per machine per backend; float summation order
   differs across GPU vendors, so parity checks always compare Rust vs torch
   on the same machine.
5. **Records**: one safetensors file per game (§5.3).
6. **Eval gauge stays Python** (torch + small-budget MCTS, once per
   eval_every games, round-aligned; see §5.4).
7. **Search logic is generic over `B: Backend`** — CI has no GPU; tree/PUCT/
   record unit tests run on the cpu backend. GPU parity and throughput are
   local acceptance items; CI runs cargo check + clippy only for the wgpu
   build (compiling needs no GPU).

### 3.1 Backend matrix

| Build | Backend | Covers | Role in Phase A |
|---|---|---|---|
| default (`wgpu`) | Vulkan / Metal / DX12 | Windows, Linux, macOS GPUs | acceptance target (AMD → Vulkan locally) |
| `--features cpu` | burn cpu backend | any machine, no GPU | CI tests + GPU-less fallback |

There is no CUDA feature. Burn backends are compile-time types; runtime
dispatch is not supported ergonomically and is not attempted.

### 3.2 Module layout (single crate, thin bins)

- `net.rs` — AzNet definition + safetensors loading (moves out of `lib.rs`)
- `records.rs` — record read/write (safetensors + `__metadata__`)
- `search/` — tree, PUCT, Dirichlet, temp schedule (generic over `B`)
- `worker.rs` — batched inference worker + buckets
- bins: `bench_burn` (existing), `sba_selfplay` (new, thin main)

The internal seams are drawn so that a future workspace split (only if Phase B
ever wants one) is a mechanical move.

## 4. The exe's search semantics — mirror `self_play_game` 1:1

The exe reproduces `src/alphazero.py::self_play_game` semantics exactly
(value_target=`root_mean` branch — per-position root value as z; the
result-based branch is not ported, the Python fallback path keeps it):

- policy target = visit-count distribution over legal moves (`+1e-8`
  smoothing, then normalize; temp ≤ 0.01 → one-hot argmax), placed into the
  81-cell target by the same `move_index` rule;
- Dirichlet `alpha="auto"` rule and `eps` mirrored from `mcts_search`;
- per-game sampling temperature = linear interpolation between
  `--temp-start` and `--temp-end` by game index within the round, clamped to
  `--temp-min` (0.05). The campaign schedule `max(0.05, 1 − idx/games)` is
  computed by the Python bridge, which owns training hyperparameters.

## 5. Interface contracts

### 5.1 exe CLI

```
sba_selfplay --weights W.safetensors --out DIR --games N
             [--sims 800] [--threads 8] [--seed-base 0]
             [--game-type ultimate] [--c-puct 1.5] [--dirichlet-eps 0.25]
             [--temp-start 1.0] [--temp-end 1.0] [--temp-min 0.05]
             [--value-target root_mean]
```

- stdout carries **JSONL only**; human-readable diagnostics go to stderr.
- exit codes: 0 ok; 2 usage; 3 weights load failure; 4 device init failure.
- the weights file's sha256 is recorded in the `start` event and in every
  record's metadata.

### 5.2 JSONL events (exe → stdout)

| type | fields | purpose |
|---|---|---|
| `start` | games, threads, sims, seed_base, device, weights_sha | bridge bookkeeping, event-stream opener |
| `game_done` | game (index within round), moves, result, temp, elapsed_s | per-game completion signal |
| `batch_stats` | every 8 games: batches, avg_batch_fill, pos_per_sec | throughput-acceptance data |
| `error` | message | fatal error → nonzero exit |

**exe events do not enter the `on_event` plane directly.** The bridge
consumes `game_done`; when the reported game is the next expected global
index it immediately runs `_optimize_step` on that game's records and then
emits the existing Python `game` event unchanged (game/games/stage/lr/loss/
positions/temp/eval/elapsed). Completed-but-not-yet-next games are buffered.
The `game` event therefore keeps its "one self-play game + one optimize step"
semantics, and the monitor page, CSV, checkpoints, and eval gauge never learn
that Rust exists.

### 5.3 Records

- one file per game: `g%05d.safetensors` in `--out`
- tensors: `states` f32 `[T,4,9,9]`, `policy` f32 `[T,81]`, `z` f32 `[T]`
- `__metadata__`: game_type, seed, result, moves, sims, weights_sha,
  format_version=1
- Python `load_records(dir)` globs and returns tuples aligned with the
  current replay format; augmentation stays in Python at sampling time.
- resume: the bridge globs existing `g*.safetensors` and spawns the exe only
  for missing games (seed offset) — the M11 re-dispatch idea, replayed for
  the exe.

### 5.4 Python wiring

- new flags on the training entry: `--selfplay-engine auto|rust|python`
  (default auto: rust when the exe is found), `--selfplay-exe PATH` override,
  `--round-games N` (games per exe round, default 32).
- rust path: `export_safetensors(model)` → spawn exe → event stream →
  per-game in-order optimize; `_save_ckpt`, `_optimize_step`, `_TrainingCsv`,
  and the eval decision run unchanged. Eval aligns to round boundaries
  (cadence semantics shift slightly; documented here as accepted).
- failure model: watchdog (no event for `MP_GAME_TIMEOUT` → kill) → one
  re-dispatch (glob + respawn for missing games) → then raise.
  `worker_lost` / `worker_failed` event semantics are reused, with the
  exe's stderr tail attached.
- python path: the current mp.Pool code stays as the fallback, untouched;
  auto mode downgrades with an event when the exe is absent.
- train-tui: event plane unchanged → zero mandatory panel changes. Two
  honesty edits (D53 style): a status line showing
  `self-play: rust (8 threads) / python (mp)`, and the workers field's hint
  text changes meaning with the engine. The 1.5 s HIP stagger hack belongs to
  the python path only.

## 6. Acceptance & testing

### 6.1 Parity, four layers (cheap → expensive; each catches a different bug class)

| layer | what | bar | where |
|---|---|---|---|
| 1. encoding | random-position fuzz (incl. terminal + Normal games): Rust planes vs Python `encode()` | bitwise equal (discrete data — no tolerance) | CI (Linux leg, cpu-feature exe) |
| 2. inference | random-plane batch fuzz: burn vs torch forward | max err ≤ 1e-5 | CI (torch cpu wheel already installed) |
| 3. search stats | 200 real positions (random openings + self-play midgames): Rust PUCT vs Python `mcts_search`, both at sims 200 | top-1 agreement ≥95% + policy KL within tolerance | local (GPU) |
| 4. strength | Rust engine vs Python AZ engine, same weights, both colors | win rate within 40–60% over ≥40 games (20 per color) — catches systematic weakening from encoding/ordering bugs | local |

Determinism test: on the cpu backend, the same seed twice → bitwise-identical
record files (CI-runnable).

### 6.2 Throughput A/B

`dev.py bench-selfplay` (new task): same machine, weights, and settings — old
mp.Pool path vs exe; wall-clock positions/sec plus the exe's `batch_stats`
(avg_batch_fill) recorded as the map for later wgpu tuning. Bar: **≥4x at
sims 800 / threads 8**. Results land in the README bench table and
`docs/decisions.md` as D56.

### 6.3 CI

- both OS legs: `cargo check` + `clippy -D warnings` (default wgpu feature
  compiles without GPU).
- Linux leg additionally: cpu-feature debug build → Rust unit tests +
  Python bridge contract tests (the exe-level tests genuinely run in CI —
  the cpu backend needs no GPU). Windows leg skips exe tests when the exe is
  absent (the DLL tests' skip pattern).
- Rust unit tests (`#[cfg(test)]`, cpu backend): PUCT math, tree ops, record
  round-trip, temp interpolation, JSONL event shape, seed determinism.

### 6.4 Python contract tests

`tests/test_selfplay_bridge.py` with a fake-exe stub (small script emitting
canned JSONL) pins bridge behavior: event translation, in-order optimize
buffering, watchdog/one re-dispatch, auto downgrade, `load_records`
round-trip (safetensors fixture). The stub makes these tests run anywhere.

## 7. Documentation landing

- `docs/decisions.md` — D56 (Phase A design + acceptance numbers)
- `docs/development.md` — task table gains `bench-selfplay`
- README bench table — new Rust self-play row
- this spec — `docs/superpowers/specs/`

## 8. Deferred / future

- wgpu tuning (autotune buckets, fusion, adapter pinning) — after the body
  lands, guided by `batch_stats`.
- Phase B: burn autodiff training in Rust; cross-platform release artifacts.
- result-based value targets in the exe, if ever needed (Python path keeps it).
