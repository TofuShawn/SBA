//! sba_selfplay — root-parallel self-play orchestration (spec §5.1/§5.2).
//! Thin bin gluing Tasks 1-5: clap args -> AzNet weights load -> batched
//! inference worker -> N game threads pulling indices off a shared counter
//! -> one JSONL event per println on stdout, one record per game
//! (g%05d.safetensors). Human diagnostics go to stderr; stdout carries
//! ONLY JSONL lines. Stream contract: start -> (game_done|batch_stats)* ->
//! (error -> nonzero exit | clean exit) — every weights (exit 3) and
//! game/record (exit 4) failure emits {"type":"error","message":...} on
//! stdout before exiting; weights failures precede the worker, so their
//! stream is the bare error event, and exit 2 (bad --game-type) fires
//! before any output. Exit codes: 0 ok, 2 bad --game-type, 3 weights
//! read/parse failure, 4 game/record runtime failure.
//!
//! RAW LOGITS contract: the worker replies with the network's raw logit
//! row; softmax happens inside search — nowhere in this exe.

use clap::Parser;
use rand::SeedableRng;
use sha2::{Digest, Sha256};
use sba_core::net::{infer_arch, AzNet};
use sba_core::{default_device, search, worker, DefaultBackend};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    weights: std::path::PathBuf,
    #[arg(long)]
    out: std::path::PathBuf,
    #[arg(long)]
    games: usize,
    #[arg(long, default_value_t = 800)]
    sims: u32,
    #[arg(long, default_value_t = 8)]
    threads: usize,
    #[arg(long, default_value_t = 0)]
    seed_base: u64,
    #[arg(long, default_value = "ultimate")]
    game_type: String,
    #[arg(long, default_value_t = 1.5)]
    c_puct: f32,
    #[arg(long, default_value_t = 0.25)]
    dirichlet_eps: f32,
    #[arg(long, default_value_t = 1.0)]
    temp_start: f32,
    #[arg(long, default_value_t = 1.0)]
    temp_end: f32,
    #[arg(long, default_value_t = 0.05)]
    temp_min: f32,
}

/// One JSONL line on stdout (serde_json values never contain newlines;
/// println! holds the stdout lock per line, so concurrent game threads
/// cannot interleave mid-line).
fn event(v: serde_json::Value) {
    println!("{v}");
}

/// Diagnose on stderr AND emit the `error` JSONL event on stdout (a stdout
/// consumer must see the stream terminate with an error, not just stop),
/// then exit. `code` is 3 (weights read/parse) or 4 (game/record runtime).
fn fail(code: i32, message: &str) -> ! {
    eprintln!("{message}");
    event(serde_json::json!({
        "type": "error",
        "message": message,
    }));
    std::process::exit(code);
}

fn main() {
    let args = Args::parse();
    // validate the game type BEFORE any GPU work (exit code 2)
    if args.game_type != "ultimate" {
        eprintln!(
            "sba_selfplay supports --game-type ultimate only (got '{}')",
            args.game_type
        );
        std::process::exit(2);
    }
    let weights_bytes = match std::fs::read(&args.weights) {
        Ok(b) => b,
        Err(e) => fail(
            3,
            &format!("cannot read weights {}: {e}", args.weights.display()),
        ),
    };
    let weights_sha = format!("{:x}", Sha256::digest(&weights_bytes));
    let device = default_device();
    // capture the device string BEFORE worker::spawn moves the device into
    // the worker thread — the start event reports the real device
    let dev_name = format!("{device:?}");
    let model = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<AzNet<DefaultBackend>, String> {
            let (channels, blocks) = infer_arch(&weights_bytes)?;
            let mut model = AzNet::<DefaultBackend>::new(&device, channels, blocks);
            model.load_safetensors(&weights_bytes, &device);
            Ok(model)
        },
    )) {
        Ok(Ok(m)) => m,
        Ok(Err(e)) => fail(3, &format!("weights load failed: {e}")),
        Err(_) => fail(3, "weights load failed (shape/name mismatch?)"),
    };
    let (handle, stats) = worker::spawn(model, device);
    event(serde_json::json!({
        "type": "start", "games": args.games, "threads": args.threads,
        "sims": args.sims, "seed_base": args.seed_base,
        "device": dev_name,
        "weights_sha": weights_sha,
    }));
    let limits = search::SearchLimits {
        sims: args.sims,
        c_puct: args.c_puct,
        dirichlet_eps: args.dirichlet_eps,
    };
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total_moves = std::sync::atomic::AtomicUsize::new(0);
    let started = std::time::Instant::now();
    // shared state the game threads borrow (copies of shared references)
    let (next, done, total_moves) = (&next, &done, &total_moves);
    let (args, limits, out, weights_sha, started, stats) =
        (&args, &limits, &args.out, &weights_sha, &started, &stats);
    std::thread::scope(|s| {
        for _ in 0..args.threads.min(args.games) {
            let mut handle = handle.clone();
            s.spawn(move || loop {
                let idx = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if idx >= args.games {
                    break;
                }
                let seed = args.seed_base + idx as u64;
                // temp: linear start->end across the round, clamped at
                // temp_min (games == 1 -> frac 0 -> temp_start)
                let denom = (args.games as f32 - 1.0).max(1.0);
                let frac = idx as f32 / denom;
                let temp = (args.temp_start + (args.temp_end - args.temp_start) * frac)
                    .max(args.temp_min);
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let game = search::play_game(sba_game::SbaGame::fresh(), &mut handle, limits, temp, &mut rng)
                    .unwrap_or_else(|e| fail(4, &format!("game {idx} failed: {e}")));
                let mut rec = game.record;
                rec.meta.seed = seed;
                rec.meta.result = game.result.clone();
                rec.meta.weights_sha = weights_sha.clone();
                rec.write(out, idx)
                    .unwrap_or_else(|e| fail(4, &format!("record write failed: {e}")));
                total_moves.fetch_add(rec.moves(), std::sync::atomic::Ordering::SeqCst);
                let d = done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                event(serde_json::json!({
                    "type": "game_done", "game": idx, "moves": rec.moves(),
                    "result": game.result, "temp": temp,
                    "elapsed_s": started.elapsed().as_secs_f64(),
                }));
                if d % 8 == 0 {
                    let batches = stats.batches.load(std::sync::atomic::Ordering::Relaxed);
                    let rows = stats.real_rows.load(std::sync::atomic::Ordering::Relaxed);
                    let fill = if batches > 0 {
                        rows as f64 / (batches as f64 * 256.0)
                    } else {
                        0.0
                    };
                    let pos_per_sec = total_moves.load(std::sync::atomic::Ordering::Relaxed) as f64
                        / started.elapsed().as_secs_f64().max(1e-9);
                    event(serde_json::json!({
                        "type": "batch_stats", "batches": batches,
                        "avg_batch_fill": fill,
                        "pos_per_sec": pos_per_sec,
                    }));
                }
            });
        }
    });
}
