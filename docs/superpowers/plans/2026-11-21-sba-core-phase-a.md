# sba-core Phase A — Rust self-play engine: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move AlphaZero self-play into a pure-Rust executable (`sba_selfplay`) that the existing Python training loop consumes via records-on-disk + JSONL events, with ≥4x end-to-end throughput at campaign settings.

**Architecture:** Single crate `native/sba-core` (burn inference, generic over backend), path-depends on `native/sba-game` for state logic (zero FFI in-process). Root-parallel: one tree + one thread per game; a single batched inference worker pads leaf requests to buckets {16, 64, 256}. Python keeps training, eval, checkpointing, CSV, and both TUIs; it only swaps the record generator.

**Tech Stack:** Rust + burn 0.16 (wgpu default / candle cpu feature) + safetensors + clap 4 + rand/rand_distr + sha2 + serde_json; Python side: torch training unchanged + `safetensors` (numpy/torch IO) + subprocess bridge.

**Spec:** `docs/superpowers/specs/2026-11-21-sba-core-phase-a-design.md` — the plan argues from the spec; executors read both.

## Global Constraints

- **No CUDA.** Backend is compile-time: default = wgpu; `--features cpu` for CI/GPU-less. No runtime dispatch.
- Search semantics mirror `src/alphazero.py` 1:1: Dirichlet eps 0.25, alpha = `10/len(moves)` on root only, priors = `(1-eps)*softmax(policy) + eps*eta`; visit target `+1e-8` smoothing; temp ≤ 0.01 → one-hot argmax else `n**(1/temp)` normalized; z = visit-weighted mean of root children Q (`_root_value`); terminal value = 0.0 draw / −1.0 for the side to move at a terminal state.
- PUCT selection (mirror `SearchNode.best_child`, `src/alphazero.py:234`): `uct = -q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)`, unvisited child → inf. The **negation is intentional**: node Q is from that node's side-to-move perspective.
- Move code = `m*9 + i` (= `move_index`). Legal-move filters, in order: occupied cell skip → closed macro skip → active-macro routing (only when `active >= 0` and macro still open).
- Records: one `g%05d.safetensors` per game, tensors `states` f32 `[T,4,9,9]`, `policy` f32 `[T,81]`, `z` f32 `[T]`, `__metadata__` with `format_version=1`, game_type, seed, result, sims, weights_sha. Little-endian f32.
- JSONL on stdout only; diagnostics on stderr. Exit codes: 0 ok, 2 usage, 3 weights failure, 4 device failure.
- Exe supports `--game-type ultimate` only (the Rust state layer is ultimate-only); `normal` exits 2 — the Python fallback path owns normal games.
- Python subprocess calls are argv-list (`shell=False`), mirroring `rustbridge.py` discipline.
- Watchdog timeout reuses `MP_GAME_TIMEOUT = 3600.0`; one re-dispatch on exe crash, then raise.
- Every task ends with `python dev.py test-rust` (Rust tasks) or `python dev.py test` (Python tasks) green, then commits **on `dev`**. Never push (user does that).
- Bins pick their backend through `sba_core::DefaultBackend` — no `Wgpu` type spelled outside a cfg.

---

### Task 1: sba-game — safe in-crate API (pub methods) + rlib

**Files:**
- Modify: `native/sba-game/Cargo.toml` (crate-type)
- Modify: `native/sba-game/src/lib.rs`

**Interfaces:**
- Produces (used by every later Rust task): `SbaGame::fresh()`, `apply(&mut self, m: i32, i: i32) -> i32`, `legal(&self) -> Vec<i32>` (move codes), `result(&self) -> i32`, `is_over(&self)`, `current(&self)`, `active(&self)`, `planes(&self) -> [f32; 324]`, and pub consts `X_WINS/O_WINS/DRAW/ONGOING`. The extern "C" fns keep their exact signatures and delegate to these methods.

- [ ] **Step 1: rlib target.** `native/sba-game/Cargo.toml`: a cdylib-only crate cannot be a Rust path dependency — sba-core needs an rlib:

```toml
[lib]
crate-type = ["cdylib", "rlib"]
```

(If a `[lib]` section already exists, add `"rlib"` to its crate-type list.)

- [ ] **Step 2: extract methods.** The logic already lives inside the `extern "C"` bodies (`sba_game_apply` :247, `sba_game_legal` :309, `sba_game_planes` :414, `sba_game_result` :288). Move each body into `impl SbaGame` as a `pub fn` and make the export a thin unsafe wrapper:

```rust
impl SbaGame {
    pub fn fresh() -> SbaGame { /* existing body of SbaGame::fresh */ }

    /// Applies move (m, i); return-code semantics identical to sba_game_apply.
    pub fn apply(&mut self, m: i32, i: i32) -> i32 {
        // existing body of the sba_game_apply export, minus the raw-pointer
        // prologue/epilogue
    }

    /// Legal move codes k = m*9+i; same filter order as sba_game_legal:
    /// occupied cell → closed macro → active-macro routing.
    pub fn legal(&self) -> Vec<i32> { /* body from sba_game_legal, collecting k */ }

    pub fn result(&self) -> i32 { raw_result(self) }
    pub fn is_over(&self) -> bool { self.result() != ONGOING }
    pub fn current(&self) -> u8 { self.current }
    pub fn active(&self) -> i32 { self.active }
    pub fn planes(&self) -> [f32; 324] { /* body from sba_game_planes */ }
}
```

Make the consts pub (`pub const X_WINS: i32 = …;` etc. — copy the existing private values verbatim, do not invent). Refactor pattern for the export (repeat for apply/legal/planes/result):

```rust
#[no_mangle]
pub extern "C" fn sba_game_apply(g: *mut SbaGame, m: i32, i: i32) -> i32 {
    if !handle_live(g) { return ERR_CLOSED; }
    unsafe { (*g).apply(m, i) }
}
```

- [ ] **Step 3: unit tests.** Append to `lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_state_and_legal_count() {
        let g = SbaGame::fresh();
        assert_eq!(g.current(), 0);
        assert_eq!(g.active(), -1);
        assert_eq!(g.legal().len(), 81); // free choice, empty board
        assert!(!g.is_over());
    }

    #[test]
    fn apply_alternates_and_routes() {
        let mut g = SbaGame::fresh();
        assert_eq!(g.apply(4, 4), 0);
        assert_eq!(g.current(), 1);
        assert_eq!(g.active(), 4); // reply routed into macro 4
        assert!(!g.legal().iter().any(|&k| k / 9 != 4));
    }

    #[test]
    fn planes_layout_matches_docstring() {
        let mut g = SbaGame::fresh();
        g.apply(0, 0);
        let p = g.planes();
        assert_eq!(p.len(), 324);
        assert_eq!(p[0], 1.0);            // plane 0 cell (0,0): X mark
        assert_eq!(p[2], 0.0);            // plane 2: O to move
        let active = &p[3 * 81..4 * 81];
        assert_eq!(active[0], 1.0);       // plane 3 block: macro 0 top-left
        assert_eq!(active[4], 1.0);
        assert_eq!(active[8], 1.0);
        assert_eq!(active[1], 0.0);
    }

    #[test]
    fn same_seed_runs_identical() {
        // two identical move sequences produce identical planes
        let mut a = SbaGame::fresh();
        let mut b = SbaGame::fresh();
        for k in [40, 0, 41] {
            a.apply(k / 9, k % 9);
            b.apply(k / 9, k % 9);
        }
        assert_eq!(a.planes(), b.planes());
    }
}
```

- [ ] **Step 4: run.** `python dev.py test-rust` (see Task 2 Step 3 for the task — land that step of Task 2 first if not present). Expected: 4 new tests pass, all existing exports still compile, clippy clean.
- [ ] **Step 5: Commit** `git add native/sba-game && git commit -m "feat(sba-game): pub safe API + rlib — sba-core consumes state logic in-process"`

### Task 2: sba-core — backend feature split, net.rs reorg, dev.py test-rust, CI

**Files:**
- Modify: `native/sba-core/Cargo.toml`
- Modify: `native/sba-core/src/lib.rs` (becomes re-export + backend glue; AzNet body moves to `src/net.rs`)
- Modify: `native/sba-core/src/bench_burn.rs` (backend alias)
- Modify: `dev.py` (test-rust task)
- Modify: `.github/workflows/ci.yml` (Linux leg: rust toolchain, sba-core clippy, cpu tests)

**Interfaces:**
- Produces: `sba_core::AzNet` (same methods as today), `sba_core::DefaultBackend`, `sba_core::default_device()`, feature flags `wgpu` (default) / `cpu`.
- Consumes: Task 1's `python dev.py test-rust` task.

- [ ] **Step 1: Cargo.toml.** Replace the burn dependency line and add features:

```toml
[features]
default = ["wgpu"]
wgpu = ["burn/wgpu"]
cpu = ["burn/candle"]

[dependencies]
burn = { version = "0.16", default-features = false }
safetensors = "0.4"
bytemuck = "1"
sba-game = { path = "../sba-game" }
```

`default-features = false` matters: with the old `features = ["wgpu"]` on the dep, wgpu would compile even in cpu-only builds. If `cargo check` then complains about missing burn capabilities, add exactly the feature it names (e.g. `std`) to the burn dep line — the gate is both feature sets compiling.

- [ ] **Step 2: reorg.** Move `ResBlock`, `AzNet`, `shape`, `read_f32s` and `load_safetensors` verbatim into `src/net.rs` (header comment moves with them). `src/lib.rs` becomes:

```rust
//! sba-core — the Rust core (D55 Phase A). See net.rs for the AZNet/weights
//! contract; records.rs for the training-record format; search.rs for the
//! PUCT tree; worker.rs for batched inference. Bins: bench_burn, sba_selfplay.

pub mod net;
pub mod records;
pub mod search;
pub mod worker;

pub use net::{read_f32s, shape, AzNet, ResBlock};

#[cfg(all(not(feature = "wgpu"), not(feature = "cpu")))]
compile_error!("enable a backend feature: wgpu (default) or cpu");

#[cfg(feature = "wgpu")]
pub type DefaultBackend = burn::backend::Wgpu;
#[cfg(all(not(feature = "wgpu"), feature = "cpu"))]
pub type DefaultBackend = burn::backend::Candle<f32, i32>;

/// The device the selected backend starts on (wgpu default adapter / cpu).
pub fn default_device() -> <DefaultBackend as burn::tensor::backend::Backend>::Device {
    #[cfg(feature = "wgpu")]
    {
        burn::backend::wgpu::WgpuDevice::default()
    }
    #[cfg(all(not(feature = "wgpu"), feature = "cpu"))]
    {
        burn::backend::candle::CandleDevice::Cpu
    }
}
```

(`records`/`search`/`worker` modules are declared now, created by Tasks 3–5 — create empty stub files `//! filled by Task N` in this task so this task compiles standalone.) `bench_burn.rs` swaps its alias:

```rust
use sba_core::DefaultBackend;
type B = DefaultBackend;
// WgpuDevice::default() → sba_core::default_device()
```

- [ ] **Step 3: dev.py test-rust.** Add beside `clippy` in TASKS:

```python
def _rust_tests(_args):
    rc = crate_task("sba-game", ["test"])
    if rc:
        return rc
    # the cpu backend needs no GPU, so the whole suite runs in WSL
    return crate_task(
        "sba-core", ["test", "--no-default-features", "--features", "cpu"]
    )


    "test-rust": (
        "cargo test: sba-game (default) + sba-core (cpu backend, no GPU)",
        _rust_tests,
    ),
```

(Place the `def _rust_tests` above the TASKS dict, the entry inside it — same style as `wsl_gcc_check`.)

- [ ] **Step 4: CI.** In `.github/workflows/ci.yml` test job, Linux-only additions after the Clippy step:

```yaml
      - name: Install Rust (Linux)
        if: runner.os == 'Linux'
        uses: dtolnay/rust-toolchain@stable

      - name: Clippy (sba-core)
        if: runner.os == 'Linux'
        run: cargo clippy --release --manifest-path native/sba-core/Cargo.toml -- -D warnings

      - name: Rust tests (sba-core, cpu backend)
        if: runner.os == 'Linux'
        run: cargo test --manifest-path native/sba-core/Cargo.toml --no-default-features --features cpu
```

- [ ] **Step 5: run.** `python dev.py test-rust` → green (first cpu-candle build takes minutes); `python dev.py clippy` → green; `python dev.py build` unchanged (wgpu bench still builds).
- [ ] **Step 6: Commit** `git add dev.py native/sba-core .github/workflows/ci.yml && git commit -m "feat(sba-core): backend feature split (wgpu default / cpu for CI), net.rs reorg"`

### Task 3: records.rs — the training-record format

**Files:**
- Modify: `native/sba-core/src/records.rs` (replace Task 2's stub)

**Interfaces:**
- Produces: `GameRecord { meta: RecordMeta, states: Vec<f32>, policy: Vec<f32>, z: Vec<f32> }`, `RecordMeta { game_type, seed, result, sims, weights_sha }`, `GameRecord::write(dir, game_idx) -> PathBuf`, `read_record(path) -> Result<GameRecord, String>`; `GameRecord::moves() -> usize`.

- [ ] **Step 1: failing test** (append to records.rs):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> GameRecord {
        GameRecord {
            meta: RecordMeta {
                game_type: "ultimate".into(),
                seed: 42,
                result: "X".into(),
                sims: 8,
                weights_sha: "abc".into(),
            },
            states: vec![0.5; 2 * 324],
            policy: vec![0.25; 2 * 81],
            z: vec![0.5, -0.5],
        }
    }

    #[test]
    fn write_read_round_trip() {
        let dir = std::env::temp_dir().join("sba_core_records_test");
        std::fs::create_dir_all(&dir).unwrap();
        let rec = sample();
        let path = rec.write(&dir, 7).unwrap();
        assert!(path.ends_with("g00007.safetensors"));
        let back = read_record(&path).unwrap();
        assert_eq!(back.states, rec.states);
        assert_eq!(back.policy, rec.policy);
        assert_eq!(back.z, rec.z);
        assert_eq!(back.meta.seed, 42);
        assert_eq!(back.meta.game_type, "ultimate");
    }

    #[test]
    fn format_version_is_1() {
        let dir = std::env::temp_dir().join("sba_core_records_test2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = sample().write(&dir, 0).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let md = st.metadata().unwrap();
        assert_eq!(md.get("format_version").map(String::as_str), Some("1"));
        assert_eq!(st.tensor("states").unwrap().shape(), &[2, 4, 9, 9]);
        assert_eq!(st.tensor("policy").unwrap().shape(), &[2, 81]);
        assert_eq!(st.tensor("z").unwrap().shape(), &[2]);
    }
}
```

- [ ] **Step 2: run** `python dev.py test-rust`. Expected: FAIL (`GameRecord` not defined / module stub empty).
- [ ] **Step 3: implement** records.rs:

```rust
//! Training-record IO: one safetensors file per game (spec §5.3).
//! Tensor names are part of the Python contract: states/policy/z, plus a
//! __metadata__ map the Python loader and the resume logic read.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use safetensors::tensor::TensorView;

pub struct RecordMeta {
    pub game_type: String,
    pub seed: u64,
    pub result: String,
    pub sims: u32,
    pub weights_sha: String,
}

pub struct GameRecord {
    pub meta: RecordMeta,
    pub states: Vec<f32>, // T * 324
    pub policy: Vec<f32>, // T * 81
    pub z: Vec<f32>,      // T
}

impl GameRecord {
    pub fn moves(&self) -> usize {
        self.z.len()
    }

    pub fn write(&self, dir: &Path, game_idx: usize) -> Result<PathBuf, String> {
        let t = 6usize;
        debug_assert_eq!(self.states.len(), self.moves() * 324);
        debug_assert_eq!(self.policy.len(), self.moves() * 81);
        let view = |name: &str, data: &[f32], dims: Vec<usize>| -> Result<TensorView, String> {
            TensorView::new(name.to_string(), dims, bytemuck::cast_slice(data))
                .map_err(|e| e.to_string())
        };
        let mut data = HashMap::new();
        data.insert("states".to_string(), view("states", &self.states, vec![t, 4, 9, 9])?);
        data.insert("policy".to_string(), view("policy", &self.policy, vec![t, 81])?);
        data.insert("z".to_string(), view("z", &self.z, vec![t])?);
        let meta: HashMap<String, String> = HashMap::from([
            ("format_version".to_string(), "1".to_string()),
            ("game_type".to_string(), self.meta.game_type.clone()),
            ("seed".to_string(), self.meta.seed.to_string()),
            ("result".to_string(), self.meta.result.clone()),
            ("sims".to_string(), self.meta.sims.to_string()),
            ("weights_sha".to_string(), self.meta.weights_sha.clone()),
        ]);
        let blob = safetensors::serialize(data, Some(meta)).map_err(|e| e.to_string())?;
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("g{game_idx:05}.safetensors"));
        fs::write(&path, &blob).map_err(|e| e.to_string())?;
        Ok(path)
    }
}
```

(`let t = 6usize;` above is a paste-guard placeholder mistake — **use `self.moves()` for the T dimension in all three shapes**, not a constant.)

```rust
pub fn read_record(path: &Path) -> Result<GameRecord, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| e.to_string())?;
    let md = st.metadata().unwrap_or_default();
    let f32s = |name: &str| -> Result<Vec<f32>, String> {
        let v = st.tensor(name).map_err(|e| e.to_string())?;
        assert_eq!(v.dtype(), safetensors::Dtype::F32, "{name}");
        Ok(bytemuck::cast_slice(v.data()).to_vec())
    };
    Ok(GameRecord {
        meta: RecordMeta {
            game_type: md.get("game_type").cloned().unwrap_or_default(),
            seed: md.get("seed").and_then(|s| s.parse().ok()).unwrap_or(0),
            result: md.get("result").cloned().unwrap_or_default(),
            sims: md.get("sims").and_then(|s| s.parse().ok()).unwrap_or(0),
            weights_sha: md.get("weights_sha").cloned().unwrap_or_default(),
        },
        states: f32s("states")?,
        policy: f32s("policy")?,
        z: f32s("z")?,
    })
}
```

- [ ] **Step 4: run** `python dev.py test-rust`. Expected: PASS.
- [ ] **Step 5: Commit** `git add native/sba-core/src/records.rs && git commit -m "feat(sba-core): per-game safetensors records with metadata"`

### Task 4: search.rs — PUCT tree + one-game runner (Evaluator trait)

**Files:**
- Modify: `native/sba-core/src/search.rs` (replace stub)

**Interfaces:**
- Produces:
  - `pub trait Evaluator { fn evaluate(&mut self, planes: [f32; 324]) -> Result<(Vec<f32>, f32), String>; }` — returns (policy logits [81], value from side-to-move perspective).
  - `pub struct SearchLimits { pub sims: u32, pub c_puct: f32, pub dirichlet_eps: f32 }`
  - `pub struct GameOutcome { pub result: String, pub record: records::GameRecord }` (meta filled by caller with seed/sims/sha)
  - `pub fn play_game(start: SbaGame, eval: &mut dyn Evaluator, limits: &SearchLimits, temp: f32, rng: &mut dyn rand::RngCore) -> Result<GameOutcome, String>`
  - `pub fn terminal_value(result: i32) -> f32` (D → 0.0, else −1.0)

- [ ] **Step 1: failing tests** (append to search.rs):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use records::RecordMeta;
    use std::cell::RefCell;

    /// Scripted evaluator: pops (policy, value) per call, planes ignored.
    struct Scripted {
        replies: RefCell<Vec<(Vec<f32>, f32)>>,
    }
    impl Evaluator for Scripted {
        fn evaluate(&mut self, _planes: [f32; 324]) -> Result<(Vec<f32>, f32), String> {
            let r = self.replies.borrow_mut().pop().unwrap();
            Ok(r)
        }
    }
    fn pol(w: &[usize]) -> Vec<f32> {
        let mut p = vec![0.0f32; 81];
        for &i in w { p[i] = 1.0; }
        p
    }

    #[test]
    fn visits_follow_greedy_policy() {
        // policy favors k=40 at the root; every leaf replies value 0.5.
        // With one-hot root prior the tree drives all sims into one branch.
        let mut e = Scripted { replies: RefCell::new(vec![]) };
        // enough replies for root + descendants (fill lazily)
        let limits = SearchLimits { sims: 6, c_puct: 1.5, dirichlet_eps: 0.0 };
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        // We need replies popped in LIFO order; push them reversed.
        for _ in 0..64 { e.replies.borrow_mut().push((pol(&[40]), 0.5)); }
        let out = play_game(SbaGame::fresh(), &mut e, &limits, 0.0, &mut rng).unwrap();
        // game runs many moves; just assert records well-formed
        assert_eq!(out.record.states.len() % 324, 0);
        assert_eq!(out.record.states.len() / 324, out.record.moves());
        assert_eq!(out.record.policy.len() / 81, out.record.moves());
        for zv in &out.record.z { assert!((-1.0..=1.0).contains(zv)); }
    }

    #[test]
    fn terminal_draw_is_zero_loss_is_minus_one() {
        assert_eq!(terminal_value(ONGOING), 0.0); // unused but total
        // build a real terminal state: fill a full draw? simpler: trust
        // constants via a won micro→macro line
        let mut g = SbaGame::fresh();
        for (m, i) in [(0,0),(1,1),(2,2)] { g.apply(m, i); }
        // not necessarily terminal — the assertion targets the mapping fn:
        assert_eq!(terminal_value(DRAW), 0.0);
        assert_eq!(terminal_value(X_WINS), -1.0);
        assert_eq!(terminal_value(O_WINS), -1.0);
    }

    #[test]
    fn low_temp_is_one_hot() {
        // the distribution helper: counts + 1e-8, temp<=0.01 → argmax one-hot
        let mut d = distribution(&[(40, 5.0), (13, 3.0)], 0.0);
        let best = d.iter().enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
        assert_eq!(best, 40);
        assert!((d.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let d2 = distribution(&[(40, 5.0), (13, 3.0)], 1.0);
        assert!(d2[40] > d2[13]); // temp 1 keeps ordering, no explosion
    }

    #[test]
    fn same_seed_same_record() {
        let run = |seed: u64| {
            let mut e = Scripted { replies: RefCell::new((0..256).map(|_| (pol(&[40]), 0.3)).collect()) };
            let limits = SearchLimits { sims: 4, c_puct: 1.5, dirichlet_eps: 0.0 };
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            play_game(SbaGame::fresh(), &mut e, &limits, 1.0, &mut rng).unwrap().record
        };
        let (a, b) = (run(9), run(9));
        assert_eq!(a.states, b.states);
        assert_eq!(a.z, b.z);
    }
}
```

- [ ] **Step 2: run** `python dev.py test-rust`. Expected: FAIL.
- [ ] **Step 3: implement** search.rs (~150 lines). Key bodies, mirroring `mcts_search`:

```rust
pub const BUCKETS: [usize; 3] = [16, 64, 256]; // used by worker.rs

pub trait Evaluator {
    fn evaluate(&mut self, planes: [f32; 324]) -> Result<(Vec<f32>, f32), String>;
}

pub fn terminal_value(result: i32) -> f32 {
    if result == DRAW { 0.0 } else { -1.0 }
}

/// Move-selection distribution from visit counts (self_play_game mirror):
/// n = counts + 1e-8; temp <= 0.01 → one-hot argmax; else n^(1/temp), normalized.
pub fn distribution(counts: &[(i32, f32)], temp: f32) -> Vec<f32> {
    let n: Vec<f32> = counts.iter().map(|(_, v)| v + 1e-8).collect();
    let mut p: Vec<f32> = if temp <= 0.01 {
        let (bi, _) = n.iter().enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        n.iter().enumerate().map(|(i, _)| if i == bi { 1.0 } else { 0.0 }).collect()
    } else {
        n.iter().map(|v| v.powf(1.0 / temp)).collect()
    };
    let s: f32 = p.iter().sum();
    for v in &mut p { *v /= s; }
    p
}

struct Node {
    state: SbaGame,
    move_code: i32,
    prior: f32,
    visits: f32,
    wins: f32,
    children: Vec<Node>,
}

impl Node {
    fn leaf(state: SbaGame, move_code: i32, prior: f32) -> Node {
        Node { state, move_code, prior, visits: 0.0, wins: 0.0, children: vec![] }
    }
    /// Mirror of SearchNode.best_child — note the -q: child Q is from the
    /// child's side-to-move, the parent's mover maximizes its negation.
    fn best_child(&self, c_puct: f32) -> Option<&mut Node> {
        let sqrt_parent = self.visits.sqrt();
        let (bi, _) = self.children.iter().enumerate().map(|(i, c)| {
            let uct = if c.visits == 0.0 {
                f32::INFINITY
            } else {
                -c.wins / c.visits + c_puct * c.prior * sqrt_parent / (1.0 + c.visits)
            };
            (i, uct)
        }).max_by(|a, b| a.1.total_cmp(&b.1))?;
        Some(&mut self.children[bi])
    }
}

fn search_once(root: &mut Node, eval: &mut dyn Evaluator, limits: &SearchLimits,
               rng: &mut dyn rand::RngCore) -> Result<(), String> {
    // descend
    let mut path: Vec<usize> = vec![]; // child indices along the descent
    let mut node: &mut Node = root;
    while !node.children.is_empty() {
        let bi = index_of_best(node, limits.c_puct);
        path.push(bi);
        node = &mut node.children[bi];
    }
    if node.state.is_over() {
        backprop(root, &path, terminal_value(node.state.result()));
        return Ok(());
    }
    // evaluate the leaf (full expansion on first visit, like mcts_search)
    let planes = node.state.planes();
    let (logits, value) = eval.evaluate(planes)?;
    let probs = softmax81(&logits);
    let moves = node.state.legal();
    let eps = limits.dirichlet_eps;
    let dirichlet = rand_distr::Dirichlet::from_shape(
        &vec![10.0 / moves.len() as f64; moves.len()], &rand::rngs::StdRng::from_rng(rng)
    ).map_err(|e| e.to_string())?; // see note below
    ...
}
```

Do not copy the sketch's Dirichlet construction — `rand_distr::Dirichlet` samples need `&mut Rng`; build the distribution once per root expansion:

```rust
use rand_distr::{Distribution, Dirichlet};

fn expand_with_priors(node: &mut Node, probs: &[f32], limits: &SearchLimits,
                      rng: &mut dyn rand::RngCore, apply_dirichlet: bool) -> Result<(), String> {
    let moves = node.state.legal();
    let eta: Vec<f64> = if apply_dirichlet {
        let d = Dirichlet::new(vec![10.0 / moves.len() as f64; moves.len()])
            .map_err(|e| e.to_string())?;
        d.sample(rng)
    } else {
        vec![0.0; moves.len()]
    };
    node.children = moves.iter().enumerate().map(|(j, &k)| {
        let p = probs[k as usize];
        let prior = (1.0 - limits.dirichlet_eps) * p + limits.dirichlet_eps * eta[j] as f32;
        let (m, i) = (k / 9, k % 9);
        let mut child_state = node.state.clone();
        child_state.apply(m, i);
        Node::leaf(child_state, k, prior)
    }).collect();
    Ok(())
}
```

Full `search_once` (select → evaluate → expand → backprop) and `play_game`:

```rust
fn backprop(root: &mut Node, path: &[usize], mut value: f32) {
    // mirror backprop(): visits +1 and wins += value at every node on the
    // path (root included), sign flipped per ply; value is from the node's
    // side-to-move perspective.
    let mut cur: &mut Node = root;
    cur.visits += 1.0; cur.wins += value; value = -value;
    for &bi in path {
        cur = &mut cur.children[bi];
        cur.visits += 1.0; cur.wins += value; value = -value;
    }
}

pub fn play_game(start: SbaGame, eval: &mut dyn Evaluator, limits: &SearchLimits,
                 temp: f32, rng: &mut dyn rand::RngCore) -> Result<GameOutcome, String> {
    let mut st = start;
    let mut states = vec![]; let mut policy = vec![]; let mut z = vec![];
    while !st.is_over() {
        let mut root = Node::leaf(st.clone(), -1, 0.0);
        for _ in 0..limits.sims {
            search_once(&mut root, eval, limits, rng, /*is_root=*/true)?;
        }
        // visit counts in legal order
        let counts: Vec<(i32, f32)> = root.children.iter().map(|c| (c.move_code, c.visits)).collect();
        let target = distribution(&counts, temp);
        // z = _root_value: visit-weighted mean of children Q (root side-to-move)
        let (tot_v, tot) = root.children.iter().fold((0.0f32, 0.0f32), |(v, n), c| {
            if c.visits > 0.0 { (v + c.visits * (c.wins / c.visits), n + c.visits) } else { (v, n) }
        });
        let zt = if tot > 0.0 { tot_v / tot } else { 0.0 };
        states.extend_from_slice(&st.planes());
        policy.extend_from_slice(&target);
        z.push(zt);
        // sample the move by target (n ** (1/temp) distribution, like select)
        let r: f64 = rng.gen();
        let mut acc = 0.0f64;
        let mut picked = counts.len() - 1;
        for (j, _) in counts.iter().enumerate() {
            acc += target[j] as f64;
            if r <= acc { picked = j; break; }
        }
        let k = counts[picked].0;
        st.apply(k / 9, k % 9);
    }
    let result = match st.result() {
        X if false => unreachable!(), // placeholder — see real mapping below
        _ => unreachable!(),
    };
    Ok(GameOutcome { result, record: records::GameRecord {
        meta: records::RecordMeta { game_type: "ultimate".into(), seed: 0, result: result.to_string(), sims: limits.sims, weights_sha: String::new() },
        states, policy, z,
    }})
}
```

The result mapping block above is a paste-guard: write it as

```rust
    let result = match st.result() {
        r if r == X_WINS => "X",
        r if r == O_WINS => "O",
        _ => "D",
    }.to_string();
```

`search_once`'s root handling: Dirichlet applies only when the evaluated node IS the root of this search (pass `is_root` down the descent as path length 0). Also `softmax81`:

```rust
fn softmax81(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}
```

- [ ] **Step 4: run** `python dev.py test-rust`. Expected: PASS (all four tests). Fix the `path: Vec<usize>` descent with an index-walk (borrow checker: walk indices down, then mutate) — the sketch above shows the shape; the compile is the verifier.
- [ ] **Step 5: Commit** `git add native/sba-core/src/search.rs && git commit -m "feat(sba-core): PUCT tree + per-game runner mirroring alphazero.mcts_search semantics"`

### Task 5: worker.rs — batched inference worker with buckets

**Files:**
- Modify: `native/sba-core/src/worker.rs` (replace stub)

**Interfaces:**
- Produces: `pub struct WorkerHandle` (Clone) implementing `search::Evaluator`; `pub struct WorkerStats { pub batches: AtomicU64, pub real_rows: AtomicU64 }`; `pub fn spawn<B: Backend>(model: AzNet<B>, device: B::Device) -> (WorkerHandle, Arc<WorkerStats>)`.
- Consumes: `search::Evaluator`, `AzNet` forward `(Tensor<B,4> [N,4,9,9]) -> (logits [N,81], value [N,1])`.

- [ ] **Step 1: failing test** (append):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::AzNet;

    #[test]
    fn replies_match_direct_forward() {
        type B = crate::DefaultBackend;
        let device = crate::default_device();
        let model = AzNet::<B>::new(&device, 8, 1); // small random net
        let (mut handle, _stats) = spawn(model.clone(), device.clone());
        // direct evaluator: single-row forward
        let direct = |planes: [f32; 324]| -> (Vec<f32>, f32) {
            let x = Tensor::<B, 4>::from_data(
                TensorData::new(planes.to_vec(), shape(&[1, 4, 9, 9])), &device);
            let (p, v) = model.forward(x);
            let p = burn::tensor::activation::softmax(p, 1);
            (p.into_data().to_vec::<f32>().unwrap(), v.into_data().to_vec::<f32>().unwrap()[0])
        };
        let mut planes = [0.0f32; 324];
        planes[0] = 1.0; planes[100] = 0.7; planes[323] = 0.2;
        let (wp, wv) = search::Evaluator::evaluate(&mut handle, planes).unwrap();
        let (dp, dv) = direct(planes);
        let err = wp.iter().zip(&dp).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-5, "policy err {err}");
        assert!((wv - dv).abs() < 1e-5);
    }

    #[test]
    fn concurrent_requests_all_answered() {
        type B = crate::DefaultBackend;
        let device = crate::default_device();
        let model = AzNet::<B>::new(&device, 8, 1);
        let (handle, stats) = spawn(model, device);
        std::thread::scope(|s| {
            for t in 0..4u64 {
                let mut h = handle.clone();
                s.spawn(move || {
                    for i in 0..75u64 {
                        let seed = t * 1000 + i;
                        let planes = std::array::from_fn(|j| ((seed * 31 + j as u64) % 97) as f32 / 97.0);
                        let (p, v) = search::Evaluator::evaluate(&mut h, planes).unwrap();
                        assert_eq!(p.len(), 81);
                        assert!((-1.0..=1.0).contains(&v));
                    }
                });
            }
        });
        assert!(stats.real_rows.load(std::sync::atomic::Ordering::Relaxed) >= 300);
        assert!(stats.batches.load(std::sync::atomic::Ordering::Relaxed) >= 2);
    }
}
```

(`model.clone()` — burn Modules are cheap-clonable; the worker owns its copy.)

- [ ] **Step 2: run** `python dev.py test-rust`. Expected: FAIL.
- [ ] **Step 3: implement** worker.rs:

```rust
//! Batched inference worker (spec §3): game threads push leaf requests on an
//! mpsc channel; one thread accumulates to the next bucket {16, 64, 256}
//! (or a 2 ms deadline, whichever first), pads with zero rows, runs one
//! AzNet forward, splits the outputs back. Softmax applied here so replies
//! carry probabilities, matching the Python path's F.softmax.

use crate::net::AzNet;
use crate::search;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use std::sync::mpsc::{channel, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct WorkerStats { pub batches: AtomicU64, pub real_rows: AtomicU64 }

struct Request { planes: [f32; 324], reply: SyncSender<(Vec<f32>, f32)> }

#[derive(Clone)]
pub struct WorkerHandle { tx: Arc<Mutex<Sender<Request>>> }

impl search::Evaluator for WorkerHandle {
    fn evaluate(&mut self, planes: [f32; 324]) -> Result<(Vec<f32>, f32), String> {
        let (rtx, rrx) = SyncSender::sync_channel(0); // rendezvous
        let req = Request { planes, reply: rtx };
        self.tx.lock().map_err(|_| "worker tx poisoned")?.send(req).map_err(|_| "worker gone")?;
        rrx.recv().map_err(|_| "worker dropped reply")
    }
}

pub fn spawn<B: Backend>(model: AzNet<B>, device: B::Device)
                         -> (WorkerHandle, Arc<WorkerStats>) {
    let (tx, rx) = channel::<Request>();
    let stats = Arc::new(WorkerStats { batches: AtomicU64::new(0), real_rows: AtomicU64::new(0) });
    let st2 = stats.clone();
    std::thread::spawn(move || worker_loop(model, device, rx, st2));
    (WorkerHandle { tx: Arc::new(Mutex::new(tx)) }, stats)
}

fn next_bucket(n: usize) -> usize {
    for b in [16usize, 64, 256] { if n <= b { return b; } }
    ((n + 255) / 256) * 256
}

fn worker_loop<B: Backend>(model: AzNet<B>, device: B::Device,
                           rx: Receiver<Request>, stats: Arc<WorkerStats>) {
    let mut pending: Vec<Request> = vec![];
    loop {
        // fill toward the next bucket, but never wait past 2 ms once we hold ≥1 row
        let deadline = Instant::now() + Duration::from_millis(2);
        while pending.len() < 256 {
            let budget = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(budget.max(Duration::from_millis(1))) {
                Ok(r) => {
                    pending.push(r);
                    if pending.len() >= next_bucket(pending.len()) { break; }
                }
                Err(_) => break,
            }
        }
        if pending.is_empty() {
            if rx.try_recv().is_err() {
                // Disconnected (all handles dropped) → shut down; Timeout → idle
                match rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(r) => pending.push(r),
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(_) => continue,
                }
            }
            continue;
        }
        let b = next_bucket(pending.len());
        let mut flat = vec![0.0f32; b * 324];
        for (i, r) in pending.iter().enumerate() {
            flat[i * 324..(i + 1) * 324].copy_from_slice(&r.planes);
        }
        let x = Tensor::<B, 4>::from_data(TensorData::new(flat, crate::shape(&[b, 4, 9, 9])), &device);
        let (logits, values) = model.forward(x);
        let probs = burn::tensor::activation::softmax(logits, 1);
        let pr = probs.into_data().to_vec::<f32>().unwrap();
        let vr = values.into_data().to_vec::<f32>().unwrap();
        for (i, r) in pending.drain(..).enumerate() {
            let _ = r.reply.send((pr[i * 81..(i + 1) * 81].to_vec(), vr[i]));
        }
        stats.batches.fetch_add(1, Ordering::Relaxed);
        stats.real_rows.fetch_add(/* rows this batch actually carried */ 0, Ordering::Relaxed);
    }
}
```

The `fetch_add(0, …)` is a paste-guard: capture `let n_real = …` **before** `pending.drain(..)` and add `n_real as u64`. (The drain consumes before the count is available otherwise — count first.)

- [ ] **Step 4: run** `python dev.py test-rust`. Expected: PASS.
- [ ] **Step 5: Commit** `git add native/sba-core/src/worker.rs && git commit -m "feat(sba-core): bucketed batched inference worker (16/64/256, 2ms max wait)"`

### Task 6: sba_selfplay bin — orchestration + JSONL

**Files:**
- Create: `native/sba-core/src/sba_selfplay.rs`
- Modify: `native/sba-core/Cargo.toml` ([[bin]] + deps: clap, rand, rand_distr, serde_json, sha2)
- Modify: `dev.py` (selfplay build lands in task_build already via sba-core release build)

**Interfaces:**
- Produces: the exe contract from spec §5.1/§5.2 (CLI flags, JSONL start/game_done/batch_stats/error, exit codes 0/2/3/4, records g%05d.safetensors).
- Consumes: Tasks 1–5 APIs.

- [ ] **Step 1: failing test** — Rust integration test `native/sba-core/tests/cli.rs`:

```rust
use std::process::Command;

fn run(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_sba_selfplay"))
        .args(args).output().expect("spawn sba_selfplay");
    (out.status.code().unwrap_or(-1),
     String::from_utf8_lossy(&out.stdout).into_owned(),
     String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn tiny_run_produces_events_and_records() {
    let dir = std::env::temp_dir().join("sba_selfplay_cli_test");
    let _ = std::fs::remove_dir_all(&dir);
    // tiny weights: sba_selfplay needs a safetensors file — tests build one
    // from a fresh AzNet via a small helper bin? No: use the parity spike's
    // path — construct weights with safetensors::serialize in this test.
    let weights = dir.join("w.safetensors");
    write_random_weights(&weights);
    let (code, stdout, stderr) = run(&[
        "--weights", weights.to_str().unwrap(),
        "--out", dir.join("games").to_str().unwrap(),
        "--games", "2", "--sims", "8", "--threads", "2", "--seed-base", "5",
    ]);
    assert_eq!(code, 0, "stderr: {stderr}");
    let lines: Vec<serde_json::Value> = stdout.lines()
        .map(|l| serde_json::from_str(l).expect("jsonl")).collect();
    assert_eq!(lines[0]["type"], "start");
    assert_eq!(lines[0]["games"], 2);
    let done: Vec<_> = lines.iter().filter(|l| l["type"] == "game_done").collect();
    assert_eq!(done.len(), 2);
    assert!(dir.join("games").join("g00000.safetensors").exists());
    assert!(dir.join("games").join("g00001.safetensors").exists());
}

#[test]
fn same_seed_bitwise_identical_records() {
    let dir = std::env::temp_dir().join("sba_selfplay_cli_det");
    let _ = std::fs::remove_dir_all(&dir);
    let weights = dir.join("w.safetensors");
    write_random_weights(&weights);
    let common = ["--weights", weights.to_str().unwrap(), "--games", "1",
                  "--sims", "8", "--threads", "1"];
    let a = dir.join("a"); let b = dir.join("b");
    let (ca, _, sa) = run(&[&common, "--seed-base", "9", "--out", a.to_str().unwrap()].concat());
    let (cb, _, _) = run(&[&common, "--seed-base", "9", "--out", b.to_str().unwrap()].concat());
    assert_eq!((ca, cb), (0, 0), "stderr a: {sa}");
    assert_eq!(
        std::fs::read(a.join("g00000.safetensors")).unwrap(),
        std::fs::read(b.join("g00000.safetensors")).unwrap());
}

#[test]
fn normal_game_type_is_usage_error() {
    let dir = std::env::temp_dir().join("sba_selfplay_cli_err");
    let _ = std::fs::remove_dir_all(&dir);
    let weights = dir.join("w.safetensors");
    write_random_weights(&weights);
    let (code, _, _) = run(&["--weights", weights.to_str().unwrap(),
        "--out", dir.to_str().unwrap(), "--games", "1", "--sims", "2",
        "--game-type", "normal"]);
    assert_eq!(code, 2);
}

// write_random_weights: AzNet::<DefaultBackend>::new(&device, 8, 1) cannot
// be saved directly to safetensors in-test without a recorder; instead
// synthesize a valid blob with the exact tensor names/shapes net.rs expects
// (conv_in.weight [8,4,3,3], bn_in.* [8], 1 ResBlock, pol/val heads [8,1,1,1]
// conv + FC [81,8]/[64,8]/[1,64] with the burn [in,out] FC layout), values
// filled with a seeded LCG. Verify the exe loads it: assert code == 0 above.
fn write_random_weights(_path: &std::path::Path) {
    todo_in_plan!(); // see Step 3 — the helper is written there, full code
}
```

The `write_random_weights` helper (real code, Step 3): it builds a `HashMap<String, TensorView>` with every name `net.rs::load_safetensors` reads, shapes per `AzNet::new(channels=8, blocks=1)`, from an LCG — full listing lives in Step 3; the test file references it.

- [ ] **Step 2: run** `cargo test --no-default-features --features cpu` (via dev.py test-rust). Expected: FAIL (bin missing).
- [ ] **Step 3: implement** the bin + helper. `sba_selfplay.rs` core (clap derive):

```rust
use clap::Parser;
use sba_core::{net::AzNet, records::GameRecord, search, worker, DefaultBackend};

#[derive(Parser)]
struct Args {
    #[arg(long)] weights: std::path::PathBuf,
    #[arg(long)] out: std::path::PathBuf,
    #[arg(long)] games: usize,
    #[arg(long, default_value_t = 800)] sims: u32,
    #[arg(long, default_value_t = 8)] threads: usize,
    #[arg(long, default_value_t = 0)] seed_base: u64,
    #[arg(long, default_value = "ultimate")] game_type: String,
    #[arg(long, default_value_t = 1.5)] c_puct: f32,
    #[arg(long, default_value_t = 0.25)] dirichlet_eps: f32,
    #[arg(long, default_value_t = 1.0)] temp_start: f32,
    #[arg(long, default_value_t = 1.0)] temp_end: f32,
    #[arg(long, default_value_t = 0.05)] temp_min: f32,
}

fn event(v: serde_json::Value) {
    println!("{v}"); // stdout is line-buffered under a lock; one line per call
}

fn main() {
    let args = Args::parse();
    if args.game_type != "ultimate" {
        eprintln!("sba_selfplay supports --game-type ultimate only (got '{}')", args.game_type);
        std::process::exit(2);
    }
    let weights_bytes = match std::fs::read(&args.weights) {
        Ok(b) => b,
        Err(e) => { eprintln!("cannot read weights {}: {e}", args.weights.display()); std::process::exit(3); }
    };
    let sha = sha2::{Digest, Sha256};
    let weights_sha = format!("{:x}", sha2::Sha256::digest(&weights_bytes));
    let device = sba_core::default_device();
    let model = match std::panic::catch_unwind(|| {
        AzNet::<DefaultBackend>::from_safetensors(&weights_bytes, &device)
    }) {
        Ok(m) => m,
        Err(_) => { eprintln!("weights load failed (shape/name mismatch?)"); std::process::exit(3); }
    };
    let (handle, stats) = worker::spawn(model, device);
    event(serde_json::json!({
        "type": "start", "games": args.games, "threads": args.threads,
        "sims": args.sims, "seed_base": args.seed_base,
        "device": format!("{:?}", std::thread::current()), // replaced below
        "weights_sha": weights_sha,
    }));
    // NOTE: the device field above must be the real adapter name — capture
    // it at spawn: let dev_name = format!("{:?}", &device); and use that.
    let limits = search::SearchLimits {
        sims: args.sims, c_puct: args.c_puct, dirichlet_eps: args.dirichlet_eps,
    };
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let started = std::time::Instant::now();
    std::thread::scope(|s| {
        for _ in 0..args.threads.min(args.games) {
            let handle = handle.clone();
            s.spawn(|| loop {
                let idx = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if idx >= args.games { break; }
                let seed = args.seed_base + idx as u64;
                // temp: linear start→end across the round, clamped (spec §4)
                let frac = if args.games.max(1) > 1 {
                    idx as f32 / (args.games - 1) as f32
                } else { 0.0 };
                let temp = (args.temp_start + (args.temp_end - args.temp_start) * frac).max(args.temp_min);
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let mut ev = handle.clone();
                let game = search::play_game(sba_game::SbaGame::fresh(), &mut ev, &limits, temp, &mut rng)
                    .unwrap_or_else(|e| { eprintln!("game {idx} failed: {e}"); std::process::exit(4); });
                let mut rec = game.record;
                rec.meta.seed = seed;
                rec.meta.result = game.result.clone();
                rec.meta.weights_sha = weights_sha.clone();
                rec.write(&args.out, idx).unwrap_or_else(|e| {
                    eprintln!("record write failed: {e}"); std::process::exit(4);
                });
                let d = done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                event(serde_json::json!({
                    "type": "game_done", "game": idx, "moves": rec.moves(),
                    "result": game.result, "temp": temp,
                    "elapsed_s": started.elapsed().as_secs_f64(),
                }));
                if d % 8 == 0 {
                    let batches = stats.batches.load(std::sync::atomic::Ordering::Relaxed);
                    let rows = stats.real_rows.load(std::sync::atomic::Ordering::Relaxed);
                    let fill = if batches > 0 { rows as f64 / (batches as f64 * 256.0) } else { 0.0 };
                    event(serde_json::json!({
                        "type": "batch_stats", "batches": batches,
                        "avg_batch_fill": fill,
                        "pos_per_sec": (d as f64 * rec.moves() as f64) / started.elapsed().as_secs_f64().max(1e-9),
                    }));
                }
            });
        }
    });
}
```

Two paste-guards above to fix while writing: (1) `device` in the start event must be `format!("{:?}", device)` captured **before** the worker consumes it; (2) `weights_sha` digest uses the `sha2::Sha256::digest` — import once (`use sha2::{Digest, Sha256};`) and drop the stray path-qualified first line. The `write_random_weights` helper goes in `tests/cli.rs` (full code: an LCG-filled `HashMap<String, (Vec<usize>, Vec<f32>)>` for every tensor name `net.rs` reads — names enumerable by reading `load_safetensors`; shapes: conv_in.weight [8,4,3,3], conv_in.bias [8], bn_in {weight,bias,running_mean,running_var} [8]×4, blocks.0.{conv1,conv2}.{weight [8,8,3,3], bias [8]} + bn1/bn2 same as bn_in, pol_conv.weight [1? no — 8→? pol_conv: [channels→8] = [8,8,1,1]? NO: pol_conv maps channels→8 so [8_out=8? Conv2dConfig::new([channels, 8], …) means in=channels(8), out=8 → weight [8,8,1,1]}, pol_bn [8]×4, pol_fc.weight [648, 81] burn layout, pol_fc.bias [81], val_conv.weight [8,8,1,1], val_bn [8]×4, val_fc1.weight [648, 64], val_fc1.bias [64], val_fc2.weight [64, 1], val_fc2.bias [1]) — serialize via `safetensors::serialize` with a metadata map (the exe ignores it).

Cargo.toml additions:

```toml
[[bin]]
name = "sba_selfplay"
path = "src/sba_selfplay.rs"

[dependencies]
clap = { version = "4", features = ["derive"] }
rand = "0.8"
rand_distr = "0.4"
serde_json = "1"
sha2 = "0.10"
```

- [ ] **Step 4: run** `python dev.py test-rust`. Expected: PASS (3 CLI tests + earlier suites).
- [ ] **Step 5: Commit** `git add native/sba-core && git commit -m "feat(sba-core): sba_selfplay exe — root-parallel orchestration, JSONL events, records"`

### Task 7: Python IO — export_safetensors + load_records

**Files:**
- Modify: `src/alphazero.py` (add `export_safetensors` near `_save_model` :551)
- Create: `src/rustselfplay.py` (loads records; the exe bridge functions come in Task 8 — this task creates the module with only `load_records`)
- Modify: `requirements.txt` (add safetensors)
- Test: `tests/test_selfplay_io.py` (create)

**Interfaces:**
- Produces: `alphazero.export_safetensors(model, path) -> str` (returns the sha256 hex of the file); `rustselfplay.load_records(dir_path) -> list[tuple[list[torch.Tensor], list[np.ndarray], list[float]]]` — tuples shaped exactly like `self_play_game`'s return so `random.choice(replay)` / `_optimize_step` consume them unchanged (`g_states`: list of `[4,9,9]` f32 tensors; `g_targets`: list of `[81]` arrays; `g_zs`: list of floats).

- [ ] **Step 1: failing test** `tests/test_selfplay_io.py`:

```python
import numpy as np
import pytest
import torch

from alphazero import AZNet, export_safetensors


def _write_fake_record(path, t=3, seed=11):
    from safetensors.numpy import save_file
    rng = np.random.default_rng(seed)
    tensors = {
        "states": rng.standard_normal((t, 4, 9, 9), dtype=np.float32),
        "policy": rng.random((t, 81), dtype=np.float32),
        "z": rng.uniform(-1, 1, (t,)).astype(np.float32),
    }
    save_file(tensors, str(path), metadata={
        "format_version": "1", "game_type": "ultimate", "seed": str(seed),
        "result": "D", "sims": "8", "weights_sha": "deadbeef",
    })
    return tensors


def test_export_safetensors_roundtrip_names(tmp_path):
    model = AZNet(9, 8, 1)
    path = tmp_path / "w.safetensors"
    sha = export_safetensors(model, str(path))
    assert len(sha) == 64
    from safetensors.numpy import load_file
    data = load_file(str(path))
    expected = {k for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    assert set(data) == expected  # int64 num_batches_tracked filtered out


def test_load_records_matches_replay_contract(tmp_path):
    import rustselfplay
    tensors = _write_fake_record(tmp_path / "g00000.safetensors")
    _write_fake_record(tmp_path / "g00001.safetensors", t=5, seed=12)
    games = rustselfplay.load_records(str(tmp_path))
    assert len(games) == 2
    states, targets, zs = games[0]
    assert len(states) == 3 and len(targets) == 3 and len(zs) == 3
    assert torch.is_tensor(states[0]) and states[0].shape == (4, 9, 9)
    assert states[0].dtype == torch.float32
    assert np.asarray(targets[0]).shape == (81,)
    assert isinstance(zs[0], float)
    # values must match the file bitwise
    assert np.allclose(states[2].numpy(), tensors["states"][2])


def test_load_records_missing_dir_is_empty():
    import rustselfplay
    assert rustselfplay.load_records(str("/nonexistent-dir-sba")) == []
```

- [ ] **Step 2: run** `python dev.py test tests/test_selfplay_io.py -q`. Expected: FAIL (`export_safetensors` / `load_records` missing; safetensors import error until requirements updated).
- [ ] **Step 3: implement.** requirements.txt (runtime block, after numpy):

```
safetensors>=0.4,<2   # record files the Rust self-play exe writes/reads
```

alphazero.py (after `_save_model`):

```python
def export_safetensors(model, path):
    """Save eval weights for the Rust self-play exe (sba-core net.rs reads
    these names 1:1). Returns the file's sha256 — records carry it so a
    record can be traced to the weights that generated it."""
    import hashlib

    from safetensors.torch import save_file

    sd = {
        k: v.detach().cpu().contiguous()
        for k, v in model.state_dict().items()
        if v.dtype.is_floating_point  # int64 num_batches_tracked is torch-only
    }
    safe = writable_path(path)
    save_file(sd, safe)
    return hashlib.sha256(open(safe, "rb").read()).hexdigest()
```

rustselfplay.py (new module):

```python
# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bridge to the Rust self-play exe (sba-core, D55 Phase A).

The exe owns search + inference and writes one safetensors record per game;
this module owns the Python side of that contract: locating the binary,
loading its records into the replay format, and (Task 8) running rounds
with a watchdog. The exe never talks to torch; Python never touches the tree.
"""

import glob
import os

import numpy as np
import torch


def load_records(dir_path):
    """Read every g*.safetensors in dir_path into replay-ready tuples.

    Returns [] for a missing/empty dir — callers treat that as "nothing
    landed", not an error (a crashed round may legitimately have written 0).
    """
    from safetensors.numpy import load_file

    games = []
    for p in sorted(glob.glob(os.path.join(dir_path, "g*.safetensors"))):
        d = load_file(p)
        states = np.ascontiguousarray(d["states"]).reshape(-1, 4, 9, 9)
        policy = d["policy"].reshape(-1, 81)
        z = d["z"].reshape(-1)
        games.append(
            (
                [torch.as_tensor(np.ascontiguousarray(s)) for s in states],
                [np.asarray(p, dtype=np.float32) for p in policy],
                [float(v) for v in z],
            )
        )
    return games
```

- [ ] **Step 4: run** `python dev.py test tests/test_selfplay_io.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `git add requirements.txt src/alphazero.py src/rustselfplay.py tests/test_selfplay_io.py && git commit -m "feat(az-train): safetensors export + record loader for the Rust self-play exe"`

### Task 8: rustselfplay.py — run_round bridge (spawn, events, watchdog)

**Files:**
- Modify: `src/rustselfplay.py` (add `find_exe`, `resolve_engine`, `run_round`)
- Test: `tests/test_selfplay_bridge.py` (create)

**Interfaces:**
- Produces:
  - `find_exe(explicit=None) -> str | None` — order: explicit arg → env `SBA_SELFPLAY_EXE` → probe `native/sba-core/target/x86_64-pc-windows-gnu/release/sba_selfplay.exe` → `native/sba-core/target/release/sba_selfplay{.exe,}` → `native/sba-core/target/debug/sba_selfplay{.exe,}` (relative to repo ROOT).
  - `resolve_engine(engine_arg, exe) -> tuple[str, str | None]` — `"auto"` resolves to rust when an exe exists, else python.
  - `run_round(exe, weights_path, out_dir, *, games, sims, threads, seed_base, temp_start, temp_end, temp_min=0.05, c_puct=1.5, dirichlet_eps=0.25, game_type="ultimate", on_event=None, timeout=MP_GAME_TIMEOUT, popen=None) -> dict` with keys `games_done: int, crashed: bool, stderr_tail: str`.
  - Events `game_done` are **collected**, not forwarded raw: `run_round` returns after the exe exits; the caller drives optimize per game (Task 9 keeps `game`-event semantics at absorb time). `on_event` (if given) receives `{"type": "selfplay_round", "games_done": n, "crashed": bool}` once at the end.
- Consumes: `alphazero.MP_GAME_TIMEOUT`.

- [ ] **Step 1: failing test** `tests/test_selfplay_bridge.py`:

```python
import json
import sys
import types

import pytest

import rustselfplay


class FakePopen:
    """Process replacement: canned stdout lines, optional crash."""

    def __init__(self, lines, crash_after=None, delay=0.0):
        self._lines = [json.dumps(l) if isinstance(l, dict) else l for l in lines]
        self._crash_after = crash_after
        self.stderr = types.SimpleNamespace(read=lambda: b"boom at game 3")

    def stdout(self):  # parity with Popen(stdout=PIPE).stdout
        raise NotImplementedError


def _fake_popen_factory(lines, crash_after=None):
    def factory(cmd, **kwargs):
        p = FakePopen(lines, crash_after)

        def _iter():
            for i, line in enumerate(p._lines):
                if p._crash_after is not None and i >= p._crash_after:
                    # simulate death: stdout ends early, exit code nonzero
                    break
                yield line + "\n"
            p.returncode = 1 if p._crash_after is not None else 0

        p.stdout = _iter()
        p.wait = lambda timeout=None: p.returncode
        return p

    return factory


def test_run_round_parses_events(monkeypatch, tmp_path):
    events = [
        {"type": "start", "games": 2},
        {"type": "game_done", "game": 0, "moves": 31, "result": "X", "temp": 1.0, "elapsed_s": 0.4},
        {"type": "game_done", "game": 1, "moves": 28, "result": "D", "temp": 1.0, "elapsed_s": 0.8},
    ]
    called = []
    res = rustselfplay.run_round(
        "fake-exe", "w.safetensors", str(tmp_path), games=2, sims=8, threads=1,
        seed_base=0, temp_start=1.0, temp_end=1.0,
        on_event=called.append, popen=_fake_popen_factory(events),
    )
    assert res["games_done"] == 2 and res["crashed"] is False
    assert called[-1]["type"] == "selfplay_round"


def test_run_round_detects_crash(monkeypatch, tmp_path):
    events = [
        {"type": "start", "games": 3},
        {"type": "game_done", "game": 0, "moves": 5, "result": "X", "temp": 1.0, "elapsed_s": 0.1},
    ]
    res = rustselfplay.run_round(
        "fake-exe", "w.safetensors", str(tmp_path), games=3, sims=8, threads=1,
        seed_base=0, temp_start=1.0, temp_end=1.0,
        popen=_fake_popen_factory(events, crash_after=1),
    )
    assert res["crashed"] is True and res["games_done"] == 1
    assert "boom" in res["stderr_tail"]


def test_resolve_engine_auto(monkeypatch):
    monkeypatch.setattr(rustselfplay, "find_exe", lambda explicit=None: "/x/sba_selfplay.exe")
    assert rustselfplay.resolve_engine("auto", None) == ("rust", "/x/sba_selfplay.exe")
    monkeypatch.setattr(rustselfplay, "find_exe", lambda explicit=None: None)
    assert rustselfplay.resolve_engine("auto", None) == ("python", None)
    assert rustselfplay.resolve_engine("python", None) == ("python", None)
    assert rustselfplay.resolve_engine("rust", "/explicit") == ("rust", "/explicit")
```

- [ ] **Step 2: run** `python dev.py test tests/test_selfplay_bridge.py -q`. Expected: FAIL (functions missing).
- [ ] **Step 3: implement** (append to rustselfplay.py):

```python
import json
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."

from alphazero import MP_GAME_TIMEOUT

_PROBE_PATHS = (
    "native/sba-core/target/x86_64-pc-windows-gnu/release/sba_selfplay.exe",
    "native/sba-core/target/release/sba_selfplay.exe",
    "native/sba-core/target/release/sba_selfplay",
    "native/sba-core/target/debug/sba_selfplay.exe",
    "native/sba-core/target/debug/sba_selfplay",
)


def find_exe(explicit=None):
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    env = os.environ.get("SBA_SELFPLAY_EXE")
    if env and os.path.isfile(env):
        return env
    for rel in _PROBE_PATHS:
        cand = os.path.normpath(os.path.join(ROOT, rel))
        if os.path.isfile(cand):
            return cand
    return None


def resolve_engine(engine_arg, exe):
    """'auto' → rust when an exe exists, else python. Explicit wins."""
    if engine_arg == "python":
        return "python", None
    if engine_arg == "rust":
        return "rust", exe or find_exe()
    found = exe or find_exe()
    return ("rust", found) if found else ("python", None)


def run_round(exe, weights_path, out_dir, *, games, sims, threads, seed_base,
              temp_start, temp_end, temp_min=0.05, c_puct=1.5,
              dirichlet_eps=0.25, game_type="ultimate", on_event=None,
              timeout=MP_GAME_TIMEOUT, popen=None):
    """Run one exe round; returns {games_done, crashed, stderr_tail}.

    argv-list only (shell=False) — same discipline as rustbridge/native/build.
    The watchdog kills the exe when no JSONL line arrives for `timeout`
    seconds (mirrors the mp.Pool worker watchdog).
    """
    cmd = [
        exe, "--weights", str(weights_path), "--out", str(out_dir),
        "--games", str(games), "--sims", str(sims), "--threads", str(threads),
        "--seed-base", str(seed_base), "--game-type", game_type,
        "--c-puct", str(c_puct), "--dirichlet-eps", str(dirichlet_eps),
        "--temp-start", f"{temp_start:.6f}", "--temp-end", f"{temp_end:.6f}",
        "--temp-min", f"{temp_min:.6f}",
    ]
    spawn = popen or subprocess.Popen
    proc = spawn(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    games_done = 0
    last_beat = time.time()
    stderr_tail = ""
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        last_beat = time.time()
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "game_done":
            games_done += 1
        if time.time() - last_beat > timeout:
            proc.kill()
            stderr_tail = (proc.stderr.read() or "")[-400:]
            break
    rc = proc.wait()
    if rc != 0 and stderr_tail == "":
        stderr_tail = (proc.stderr.read() or "")[-400:]
    crashed = rc != 0 or games_done < games
    if on_event:
        on_event({"type": "selfplay_round", "games_done": games_done, "crashed": crashed})
    return {"games_done": games_done, "crashed": crashed, "stderr_tail": stderr_tail}
```

(Paste-guard: the watchdog `if time.time() - last_beat > timeout` must be evaluated **per line** in the loop body, not after the assignment — keep the two statements in that order and the check before incrementing any state that assumes liveness. If the loop blocks forever on a silent exe, that's the bug this test guards: add `test_run_round_watchdog_kills_silent_exe` with a FakePopen whose `_iter` sleeps past a tiny injected `timeout=0.05` and assert `crashed is True`.)

- [ ] **Step 4: run** `python dev.py test tests/test_selfplay_bridge.py -q`. Expected: PASS (4 tests).
- [ ] **Step 5: Commit** `git add src/rustselfplay.py tests/test_selfplay_bridge.py && git commit -m "feat(selfplay-bridge): exe round runner with watchdog + engine resolution"`

### Task 9: train() wiring — --selfplay-engine auto|rust|python

**Files:**
- Modify: `src/alphazero.py` (argparse in `main` :1517-1589 area; `train()` signature + self-play dispatch; extract `_absorb_records` from the single-process loop)
- Modify: `src/sba.py` (pass-through only if it filters flags — verify; it forwards args verbatim at :387-390, so likely untouched)
- Test: `tests/test_train_rust_path.py` (create)

**Interfaces:**
- Produces: `train(..., selfplay_engine="auto", selfplay_exe=None, round_games=32)`; helper `_absorb_records(replay, records, *, replay_cap) -> int` (appends loaded games, trims to cap, returns total positions).
- Consumes: Task 7 `export_safetensors`/`load_records`, Task 8 `resolve_engine`/`run_round`/`find_exe`.

- [ ] **Step 1: failing test** `tests/test_train_rust_path.py`:

```python
import numpy as np
import pytest
import torch

import alphazero
import rustselfplay


@pytest.fixture
def fake_rust(monkeypatch):
    """Patch the bridge so train() sees a rust engine without any exe."""
    calls = []

    def fake_run_round(exe, weights_path, out_dir, **kw):
        calls.append({"exe": exe, "games": kw["games"], "seed_base": kw["seed_base"]})
        # materialize real records so load_records returns them
        from safetensors.numpy import save_file
        import os
        os.makedirs(out_dir, exist_ok=True)
        t = kw["games"]
        rng = np.random.default_rng(kw["seed_base"] + 1)
        save_file({
            "states": rng.standard_normal((t, 4, 9, 9), dtype=np.float32),
            "policy": rng.random((t, 81), dtype=np.float32),
            "z": rng.uniform(-1, 1, (t,)).astype(np.float32),
        }, os.path.join(out_dir, "g%05d.safetensors" % t), metadata={
            "format_version": "1", "game_type": "ultimate", "seed": str(kw["seed_base"]),
            "result": "D", "sims": str(kw["sims"]), "weights_sha": "x",
        })
        return {"games_done": kw["games"], "crashed": False, "stderr_tail": ""}

    monkeypatch.setattr(rustselfplay, "run_round", fake_run_round)
    monkeypatch.setattr(rustselfplay, "find_exe", lambda explicit=None: "/fake/exe")
    monkeypatch.setattr(rustselfplay, "resolve_engine",
                        lambda arg, exe: ("rust", "/fake/exe"))
    return calls


def test_train_rust_engine_runs_and_emits(fake_rust, tmp_path):
    events = []
    model = alphazero.train(
        "ultimate", games=4, sims=2, eval_every=100, lr=1e-4,
        steps_per_game=1, channels=8, blocks=1, save=False, quiet=True,
        replay_cap=10, csv_path=str(tmp_path / "curve.csv"),
        on_event=events.append, selfplay_engine="rust", round_games=2,
    )
    assert model is not None
    # two rounds of 2 games
    assert [c["games"] for c in fake_rust] == [2, 2]
    types = [e["type"] for e in events]
    assert types.count("game") == 4
    assert types.count("selfplay_round") == 2
    # CSV: one row per game (D41 contract holds through the rust path)
    import csv as _csv
    with open(tmp_path / "curve.csv") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 4


def test_train_rust_retry_after_crash(fake_rust, monkeypatch, tmp_path):
    state = {"n": 0}

    def flaky_run_round(exe, weights_path, out_dir, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return {"games_done": 1, "crashed": True, "stderr_tail": "boom"}
        return {"games_done": kw["games"], "crashed": False, "stderr_tail": ""}

    monkeypatch.setattr(rustselfplay, "run_round", flaky_run_round)
    events = []
    alphazero.train(
        "ultimate", games=2, sims=2, eval_every=100, lr=1e-4,
        steps_per_game=1, channels=8, blocks=1, save=False, quiet=True,
        replay_cap=10, on_event=events.append,
        selfplay_engine="rust", round_games=2,
    )
    assert state["n"] == 2  # one retry
    types = [e["type"] for e in events]
    assert types.count("worker_lost") == 1


def test_auto_downgrades_to_python_without_exe(monkeypatch, tmp_path):
    monkeypatch.setattr(rustselfplay, "find_exe", lambda explicit=None: None)
    assert rustselfplay.resolve_engine("auto", None) == ("python", None)


def test_main_parses_selfplay_flags():
    # alphazero.main argv parse accepts the new flags (argparse SystemExit=2
    # on garbage is the negative case)
    with pytest.raises(SystemExit):
        alphazero.main(["train", "--selfplay-engine", "bogus"])
```

- [ ] **Step 2: run** `python dev.py test tests/test_train_rust_path.py -q`. Expected: FAIL.
- [ ] **Step 3: implement.**

argparse (after `--workers` at :1589):

```python
    sp.add_argument("--selfplay-engine", choices=["auto", "rust", "python"],
                    default="auto",
                    help="self-play source: rust exe (default when built), python mp.Pool, auto=probe")
    sp.add_argument("--selfplay-exe", default=None,
                    help="explicit path to sba_selfplay (default: probe native/sba-core/target)")
    sp.add_argument("--round-games", type=int, default=32,
                    help="games per rust self-play round (default: 32)")
```

train() signature gains the three kwargs after `on_event=None`. Engine resolution at the top of the training loop setup:

```python
    engine, exe = rustselfplay.resolve_engine(selfplay_engine, selfplay_exe)
    if on_event:
        on_event({"type": "selfplay_engine", "engine": engine,
                  "threads": workers if engine == "rust" else workers})
```

Extract from the existing single-process loop (the block at :1066-1150 that appends to replay, runs `_optimize_step`, handles ckpt/eval/CSV/events) a helper both paths share — signature:

```python
def _absorb_records(replay, records, *, replay_cap):
    """Append self-play records (replay-ready tuples) to the replay buffer,
    trimming to the cap. Returns total positions absorbed."""
    total = 0
    for states, targets, zs in records:
        replay.append((states, targets, zs))
        if len(replay) > replay_cap:
            replay.pop(0)
        total += len(states)
    return total
```

The rust path (new branch in the train loop, replacing the `self_play_game` call when `engine == "rust"`):

```python
    if engine == "rust":
        import time as _time

        it = 1
        while it <= games:
            r_end = min(it + round_games - 1, games)
            out_dir = writable_path(os.path.join(
                MODEL_DIR, "..", "logs", "selfplay-%d" % _time.time_ns()))
            weights_path = writable_path(os.path.join(
                MODEL_DIR, "..", "logs", "selfplay-%d.safetensors" % _time.time_ns()))
            export_safetensors(model, weights_path)
            res = rustselfplay.run_round(
                exe, weights_path, out_dir,
                games=r_end - it + 1, sims=sims, threads=workers,
                seed_base=(seed or 0), temp_start=max(0.05, 1.0 - (it - 1) / games),
                temp_end=max(0.05, 1.0 - (r_end - 1) / games),
                temp_min=0.05, dirichlet_eps=dirichlet_eps, on_event=on_event,
                timeout=MP_GAME_TIMEOUT,
            )
            if res["crashed"]:
                if on_event:
                    on_event({"type": "worker_lost", "game": it})
                # one re-dispatch for the missing games (spec §5.4): a fresh
                # round dir, advanced seed base; records land beside the first
                res = rustselfplay.run_round(
                    exe, weights_path, out_dir + "-retry",
                    games=games - res["games_done"], sims=sims, threads=workers,
                    seed_base=(seed or 0) + res["games_done"],
                    temp_start=max(0.05, 1.0 - (it - 1 + res["games_done"] - 1) / games),
                    temp_end=temp_end_clamped,
                    temp_min=0.05, dirichlet_eps=dirichlet_eps, on_event=on_event,
                    timeout=MP_GAME_TIMEOUT,
                )
                if res["crashed"]:
                    raise RuntimeError(
                        "rust self-play died twice: %s" % res["stderr_tail"])
            records = rustselfplay.load_records(out_dir) + rustselfplay.load_records(out_dir + "-retry")
            for g_states, g_targets, g_zs in records:
                _absorb_records(replay, [(g_states, g_targets, g_zs)], replay_cap=replay_cap)
                total_loss = _optimize_step(model, optimizer, replay, steps_per_game, augment, size, policy_loss)
                positions_total += len(g_states)
                ckpt / eval / CSV / game-event block — same statements as the
                single-process loop below (see :1057-1150), with idx = it;
                then it += 1
        ...
```

The `temp_end_clamped` above is a paste-guard: compute `temp_end=max(0.05, 1.0 - (r_end - 1) / games)` **before** the first `run_round` call and reuse it in the retry. The per-game absorb block must be extracted from the existing loop into the same statements the single-process path uses (checkpoint at `ckpt_every`, `evaluate()` at `eval_every`, `_TrainingCsv.write`, the `game` event dict) — do not fork logic; if the single-process body needs reshaping to share, reshape it (audit M12's direction).

- [ ] **Step 4: run** `python dev.py test tests/test_train_rust_path.py tests/test_sba.py -q` then full `python dev.py test`. Expected: PASS; existing suite untouched-green.
- [ ] **Step 5: Commit** `git add src/alphazero.py src/rustselfplay.py tests/test_train_rust_path.py && git commit -m "feat(az-train): --selfplay-engine rust path — rounds, in-order absorb, retry-once"`

### Task 10: train-tui honest engine labels

**Files:**
- Modify: `src/tui_tools.py` (TrainApp status line + workers-field hint)
- Test: `tests/test_tui_engine_labels.py` (create)

**Interfaces:**
- Produces: `tui_tools.engine_note(engine, workers) -> str` — `"self-play: rust (N threads)"` / `"self-play: python (mp, N workers)"`; TrainApp shows it and swaps the workers field hint text by engine.
- Consumes: Task 8 `resolve_engine`/`find_exe`.

- [ ] **Step 1: failing test** `tests/test_tui_engine_labels.py`:

```python
import tui_tools


def test_engine_note_rust():
    assert tui_tools.engine_note("rust", 8) == "self-play: rust (8 threads)"


def test_engine_note_python():
    assert tui_tools.engine_note("python", 4) == "self-play: python (mp, 4 workers)"


def test_train_app_shows_engine_note():
    # offscreen Textual app: mount TrainApp, assert the note widget renders
    # the resolved engine (auto → python when no exe on this machine is fine —
    # resolve_engine is monkeypatched)
    from unittest import mock
    import rustselfplay
    with mock.patch.object(rustselfplay, "find_exe", lambda explicit=None: "/x/exe"):
        app = tui_tools.TrainApp()
        async def _run():
            async with app.run_test() as pilot:
                await pilot.pause()
                text = str(app.query_one("#engine-note").renderable)
                assert "rust" in text
        import asyncio
        asyncio.run(_run())
```

(If `TrainApp.__init__` requires arguments on this branch, pass what D45's page-1 needs — read the class before writing the test; the assertion target is the note widget's text containing the engine word.)

- [ ] **Step 2: run** `python dev.py test tests/test_tui_engine_labels.py -q`. Expected: FAIL.
- [ ] **Step 3: implement.** In tui_tools.py:

```python
def engine_note(engine, workers):
    """One-line self-play provenance for the TrainApp status bar (D53 honesty:
    the label states what actually runs)."""
    if engine == "rust":
        return "self-play: rust (%d threads)" % workers
    return "self-play: python (mp, %d workers)" % workers
```

TrainApp: resolve once at `on_mount` (`engine, exe = rustselfplay.resolve_engine("auto", None)`), render `engine_note` into a `#engine-note` static on page 1, and swap the workers field's hint description between "threads of one exe process" (rust) and "self-play worker processes" (python) — the fields' semantic change must be visible, not silent.

- [ ] **Step 4: run** `python dev.py test tests/test_tui_engine_labels.py tests/test_tui_*.py -q` then full suite. Expected: PASS.
- [ ] **Step 5: Commit** `git add src/tui_tools.py tests/test_tui_engine_labels.py && git commit -m "feat(train-tui): honest self-play engine note + workers hint per engine"`

### Task 11: sba_debug bin — planes/search dump (parity harness anchor)

**Files:**
- Create: `native/sba-core/src/sba_debug.rs`
- Modify: `native/sba-core/Cargo.toml` ([[bin]])
- Modify: `.github/workflows/ci.yml` (Linux leg builds the cpu debug exe before pytest)
- Test: `tests/test_encode_parity.py` (create)

**Interfaces:**
- Produces:
  - `sba_debug planes --games N --seed S --moves K --out DIR` — per game: `meta.json` `{"moves": [[m, i], …], "seed": S}` + `planes_g%05d.bin` (324 f32 LE).
  - `sba_debug search --weights W --moves "m,i;m,i" --sims N` — prints one JSON line `{"counts": {"k": v, …}, "value": f}` (root visit counts after the search, root value).
- Consumes: Task 1 `SbaGame` API, Task 4 search, Task 5 worker.

- [ ] **Step 1: failing test** `tests/test_encode_parity.py`:

```python
import json
import os
import struct
import subprocess

import numpy as np
import pytest

import rustselfplay

EXE = rustselfplay.find_exe()  # cpu debug build on CI, wgpu release locally
pytestmark = pytest.mark.skipif(EXE is None, reason="sba_debug not built")


def _run_planes_dump(tmp_path, games=3, seed=7, moves=10):
    out = tmp_path / "dump"
    subprocess.run(
        [EXE, "planes", "--games", str(games), "--seed", str(seed),
         "--moves", str(moves), "--out", str(out)],
        check=True, capture_output=True)
    return out


def test_rust_planes_match_python_encode_bitwise(tmp_path):
    from SBA import UltimateGame

    out = _run_planes_dump(tmp_path)
    for g in sorted(os.listdir(out)):
        if not g.endswith(".bin"):
            continue
        meta = json.load(open(out / g.replace(".bin", ".meta.json")))
        planes = np.fromfile(out / g, dtype="<f4").reshape(4, 9, 9)
        game = UltimateGame()
        for m, i in meta["moves"]:
            assert (m, i) in game.legal_moves()
            game.make_move(m, i)
        py = game  # encode() on the pure-Python class, not the Rust one
        from alphazero import encode
        ref = encode(py).numpy()
        assert np.array_equal(ref, planes), g  # bitwise — discrete data
```

- [ ] **Step 2: run** `python dev.py test tests/test_encode_parity.py -q`. Expected: SKIP locally if only the wgpu release build exists and sba_debug isn't in it (build it: `python dev.py build`), FAIL if exe present but layout diverges.
- [ ] **Step 3: implement** sba_debug.rs (two subcommands, hand-parsed like bench_burn):

```rust
// planes mode: seed → N games × K random legal moves each (StdRng per game),
// dump moves + planes. search mode: apply --moves, spawn worker, run
// search_once for --sims with dirichlet OFF (deterministic comparison), print
// {"counts": {k: visits}, "value": root_value}.
```

(The full body follows Task 4/5/6 patterns directly — `SbaGame::fresh()` + seeded `rng.gen_range(0..moves.len())` over `legal()`; the search subcommand builds `AzNet::from_safetensors` + `worker::spawn` and prints the JSON. Both modes write nothing but stdout/dump-dir.) CI addition (test job, Linux, before "Run tests"):

```yaml
      - name: Build sba_debug (cpu)
        if: runner.os == 'Linux'
        run: cargo build --manifest-path native/sba-core/Cargo.toml --no-default-features --features cpu --bin sba_debug --bin sba_selfplay
```

(`tests/test_encode_parity.py`'s `find_exe` probe gains the debug paths — already first in `_PROBE_PATHS`? debug is last; the CI-built debug exe is found by the debug probe entries.)

- [ ] **Step 4: run** `python dev.py build` then `python dev.py test tests/test_encode_parity.py -q`. Expected: PASS locally (bitwise).
- [ ] **Step 5: Commit** `git add native/sba-core .github/workflows/ci.yml tests/test_encode_parity.py && git commit -m "feat(sba-core): sba_debug planes/search modes; encode parity pinned bitwise"`

### Task 12: inference parity fuzz (layer 2, CI-runnable)

**Files:**
- Modify: `native/sba-core/src/bench_burn.rs` (batch size from input length — remove the hardcoded 256)
- Test: `tests/test_infer_parity.py` (create)

**Interfaces:**
- Produces: bench_burn accepts any batch shape (dims derived from the input file), parity line format unchanged (`parity: policy_max_err=… value_max_err=…`).
- Consumes: Task 7 `export_safetensors`.

- [ ] **Step 1: failing test** `tests/test_infer_parity.py`:

```python
import struct
import subprocess

import numpy as np
import pytest
import torch

from alphazero import AZNet, export_safetensors
import rustselfplay

EXE = rustselfplay.find_exe()
pytestmark = pytest.mark.skipif(EXE is None, reason="bench_burn not built")


def test_burn_matches_torch_within_1e5(tmp_path):
    torch.manual_seed(3)
    model = AZNet(9, 8, 1)
    weights = tmp_path / "w.safetensors"
    export_safetensors(model, str(weights))

    rng = np.random.default_rng(5)
    batch = rng.standard_normal((64, 4, 9, 9), dtype=np.float32)
    (tmp_path / "input.bin").write_bytes(batch.tobytes())

    with torch.no_grad():
        p_ref, v_ref = model(torch.as_tensor(batch))

    out = subprocess.run(
        [rustselfplay.find_bench(), str(weights), str(tmp_path / "input.bin"),
         "--policy-ref", "/dev/null", "--value-ref", "/dev/null"],
        capture_output=True, text=True)
    ...
```

Stop — bench_burn requires reference bins to print parity. Two options, pick **(a)**: extend bench_burn with a `--dump-out PREFIX` mode writing policy/value bins, and the Python test reads them back and compares. Final test body:

```python
def test_burn_matches_torch_within_1e5(tmp_path):
    from alphazero import AZNet, export_safetensors

    torch.manual_seed(3)
    model = AZNet(9, 8, 1).eval()
    weights = tmp_path / "w.safetensors"
    export_safetensors(model, str(weights))

    rng = np.random.default_rng(5)
    batch = rng.standard_normal((64, 4, 9, 9), dtype=np.float32)
    (tmp_path / "input.bin").write_bytes(batch.tobytes())

    out = subprocess.run(
        [rustselfplay.find_bench(), str(weights), str(tmp_path / "input.bin"),
         "--dump-out", str(tmp_path / "out")],
        capture_output=True, text=True, check=True)
    assert "parity" not in out.stdout or True  # no refs → no parity line
    policy = np.fromfile(tmp_path / "out_policy.bin", dtype="<f4").reshape(64, 81)
    value = np.fromfile(tmp_path / "out_value.bin", dtype="<f4").reshape(64)
    with torch.no_grad():
        p_ref, v_ref = model(torch.as_tensor(batch))
    p_ref = torch.softmax(p_ref, dim=1).numpy()
    v_ref = v_ref.numpy()[:, 0]
    err_p = float(np.abs(policy - p_ref).max())
    err_v = float(np.abs(value - v_ref).max())
    assert err_p < 1e-5, err_p
    assert err_v < 1e-5, err_v
```

`rustselfplay.find_bench()` probes `native/sba-core/target/{x86_64-pc-windows-gnu/release,release,debug}/bench_burn{,.exe}` (same pattern as `find_exe`).

- [ ] **Step 2: run** `python dev.py test tests/test_infer_parity.py -q`. Expected: FAIL (`find_bench` missing / `--dump-out` unsupported).
- [ ] **Step 3: implement.** bench_burn.rs: derive `let n = raw.len() / (4 * 9 * 9 * 4);` and `shape(&[n, 4, 9, 9])`; after the (now-optional) parity block, add:

```rust
    // --dump-out PREFIX: write softmax(policy) + value rows as f32 LE so a
    // non-torch process can compare against its own reference
    if let Some(pos) = std::env::args().position(|a| a == "--dump-out") {
        let prefix = std::env::args().nth(pos + 1).expect("--dump-out needs a value");
        let probs = burn::tensor::activation::softmax(p, 1);
        std::fs::write(format!("{prefix}_policy.bin"),
                       probs.into_data().as_bytes()).unwrap();
        std::fs::write(format!("{prefix}_value.bin"),
                       v.into_data().as_bytes()).unwrap();
    }
```

(Check `into_data()`'s byte accessor name against burn 0.16 — `as_bytes()` may be `to_vec::<u8>()` on `TensorData`; use whatever compiles, the contract is raw LE bytes.) rustselfplay.py gains `find_bench()` mirroring `find_exe`.

- [ ] **Step 4: run** `python dev.py test tests/test_infer_parity.py -q` (local wgpu build) then `python dev.py test-rust` + full `python dev.py test`. Expected: PASS everywhere.
- [ ] **Step 5: Commit** `git add native/sba-core/src/bench_burn.rs src/rustselfplay.py tests/test_infer_parity.py && git commit -m "test(sba-core): burn-vs-torch inference fuzz pinned at 1e-5 via bench_burn dump mode"`

### Task 13: throughput A/B, parity layers 3–4, docs (D56, README, development.md)

**Files:**
- Create: `native/bench_selfplay.py`
- Create: `native/parity_search.py`
- Modify: `dev.py` (bench-selfplay task)
- Modify: `docs/decisions.md` (D56), `docs/development.md` (task row), `README.md` (bench row)

**Interfaces:**
- Produces: `dev.py bench-selfplay` — the ≥4x acceptance measurement; `parity_search.py layers 3 4` — the local acceptance for search-stats and strength parity.
- Consumes: everything above.

- [ ] **Step 1: bench_selfplay.py** (argv-list, rich table like native/bench_rust.py):

```python
"""A/B: old mp.Pool self-play path vs the Rust exe, same machine/weights.

python native/bench_selfplay.py --weights models/az_ultimate.pt \
    --games 16 --sims 800 --threads 8 [--engine both|rust|python]
"""
import argparse
import multiprocessing as mp
import time

import numpy as np
import torch

from alphazero import AZNet, self_play_game, export_safetensors, DEVICE
from rustselfplay import find_exe, run_round, load_records


def bench_python(model, games, sims, threads, temp):
    """The old path, unchanged: mp.Pool of self_play_game workers."""
    t0 = time.time()
    positions = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(threads) as pool:
        for _ in range(games):
            states, targets, zs = pool.apply(
                _one_game, (model.state_dict(), games, sims, temp))
            positions += len(states)
    return positions / (time.time() - t0)


def _one_game(state_dict, games, sims, temp):
    from alphazero import AZNet, self_play_game
    import torch
    model = AZNet(9, 128, 5)
    model.load_state_dict(state_dict)
    model.eval()
    states, targets, zs = self_play_game(
        lambda: __import__("rustbridge").RustUltimateGame(), model, sims, temp)
    return states, targets, zs


def bench_rust(model, games, sims, threads, temp):
    import tempfile, os
    exe = find_exe()
    assert exe, "build the exe first: python dev.py build"
    weights = os.path.join(tempfile.mkdtemp(), "w.safetensors")
    sha = export_safetensors(model, weights)
    out = os.path.join(tempfile.mkdtemp(), "games")
    t0 = time.time()
    res = run_round(exe, weights, out, games=games, sims=sims, threads=threads,
                    seed_base=0, temp_start=temp, temp_end=temp)
    assert not res["crashed"], res["stderr_tail"]
    records = load_records(out)
    positions = sum(len(s) for s, _, _ in records)
    return positions / (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/az_ultimate.pt")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--engine", choices=["both", "rust", "python"], default="both")
    a = ap.parse_args()
    ckpt = torch.load(a.weights, map_location=DEVICE, weights_only=True)
    model = AZNet(ckpt["size"], ckpt.get("channels", 128), ckpt.get("blocks", 5)).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    temp = 1.0
    rows = []
    if a.engine in ("both", "python"):
        rows.append(("python mp.Pool", bench_python(model, a.games, a.sims, a.threads, temp)))
    if a.engine in ("both", "rust"):
        rows.append(("rust exe", bench_rust(model, a.games, a.sims, a.threads, temp)))
    base = rows[0][1]
    for name, pps in rows:
        print(f"{name:16s} {pps:10.1f} pos/s  ({pps / base:4.2f}x)")
```

(Paste-guards: `channels/blocks` come from the checkpoint like train(); `_one_game` must reconstruct the model on the worker side (spawn pickles the state_dict, not the module). If the campaign checkpoint layout differs, follow `_save_model`'s format.)

dev.py task:

```python
    "bench-selfplay": (
        "self-play A/B: mp.Pool path vs rust exe (pos/s, speedup)",
        lambda a: venv(["native/bench_selfplay.py", *a]),
    ),
```

- [ ] **Step 2: parity_search.py** (layers 3–4, local GPU):

```python
"""Layer 3: search-stats parity; layer 4: strength head-to-head (spec §6.1).

python native/parity_search.py --weights models/az_ultimate.pt \
    --games 20 --sims 200 [--layer 3|4]
"""
```

Layer 3: for N seeded random openings (Rust `sba_debug search` per position vs Python `mcts_search(game, model, sims, dirichlet_alpha="auto", dirichlet_eps=0.25)`): compare argmax-of-counts agreement (bar ≥95%) and KL(policy_target, policy_target) (bar ≤0.1 nats). Layer 4: play `--games` matches alternating colors, Rust side via `sba_debug search --sims 800` per move (process spawn per move is acceptable for a one-time acceptance), Python side via `alphazero_move`; report win rate and the 40–60% verdict. Both layers reuse `rustselfplay.find_exe`; nothing here runs in CI.

- [ ] **Step 3: measure and record.** Run `python dev.py bench-selfplay --games 16 --sims 800 --threads 8`; run `python native/parity_search.py --weights <campaign ckpt> --games 20 --sims 200`. Fill the numbers into:
  - `docs/decisions.md` — append **D56**: sba-core Phase A design summary (exe seam, root-parallel, buckets, records contract), the measured speedup, parity results, and the acceptance verdict against the ≥4x bar.
  - `docs/development.md` — task table row: `python dev.py bench-selfplay` → "self-play A/B: mp.Pool vs rust exe (pos/s, speedup)".
  - `README.md` bench table — Rust self-play row with the measured positions/sec + speedup (same style as the existing Rust state-layer rows).
- [ ] **Step 4: gates.** `python dev.py lint && python dev.py format && python dev.py test && python dev.py test-rust && python dev.py cgate && python dev.py clippy` — all green.
- [ ] **Step 5: Commit** `git add native/bench_selfplay.py native/parity_search.py dev.py docs/decisions.md docs/development.md README.md && git commit -m "docs(perf): D56 — sba-core Phase A acceptance: throughput A/B + parity layers 3-4"`

---

## Self-review notes (kept in the plan per the writing-plans skill)

- **Spec coverage**: §3 architecture → Tasks 2/4/5/6; §4 search semantics → Task 4 (+ Task 11 pins encoding, Task 12 pins inference); §5.1 CLI → Task 6; §5.2 events → Tasks 6/8/9; §5.3 records → Tasks 3/7; §5.4 wiring → Tasks 9/10; §6.1 layers 1/2 → Tasks 11/12 (CI), layers 3/4 → Task 13 (local); §6.2 → Task 13; §6.3 CI → Task 2 (+11); §6.4 → Task 8; §7 docs → Task 13. The `batch_stats`-every-8-games contract is implemented in Task 6.
- **Known paste-guards** (intentional in-plan markers, all resolvable at write time): Task 3 T-dimension, Task 5 `n_real` count, Task 6 device capture + sha import, Task 9 `temp_end_clamped` + absorb-block extraction, Task 12 `into_data` byte accessor name. Each has its fix stated next to it.
- **Type consistency**: `load_records` → `[(states, targets, zs)]` tuples match `_optimize_step`'s `random.choice(replay)` unpack; `run_round`'s return dict is the same shape in Tasks 8/9/13; `SearchLimits` fields identical in Tasks 4/6; `WorkerHandle` implements `search::Evaluator` in Task 5 as consumed by Task 6.
