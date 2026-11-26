//! sba_debug — the parity-harness debug bin. Two side-effect-free clap subcommands:
//! `planes --games N --seed S --moves K --out DIR` dumps `planes_g%05d.bin` (324 f32
//! LE from SbaGame::planes()) plus a meta JSON whose moves tests/test_encode_parity.py
//! replays on the pure-Python UltimateGame to compare encode() BITWISE; and
//! `search --weights W --moves "m,i;m,i" --sims N [--batch-size K]` prints one JSON
//! line of counts and root value with dirichlet_eps 0.0 (deterministic).
//! `--batch-size` mirrors python's `mcts_search(batch_size=...)`: 256 = wave, 1 = serial.
//! Exit codes: 0 ok, 1 runtime failure (weights, moves, IO); clap exits 2 on bad usage.

use clap::{Parser, Subcommand};
use rand::{Rng, SeedableRng};
use sba_core::net::{infer_arch, AzNet};
use sba_core::{default_device, search, worker, DefaultBackend};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Dump random-game planes + moves for the encode parity test.
    Planes {
        #[arg(long)]
        games: usize,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        moves: usize,
        #[arg(long)]
        out: std::path::PathBuf,
    },
    /// Search one position; print {"counts": {...}, "value": f} on stdout.
    Search {
        #[arg(long)]
        weights: std::path::PathBuf,
        /// The moves to reach the position, "m,i;m,i" (empty = fresh game).
        #[arg(long, default_value = "")]
        moves: String,
        #[arg(long)]
        sims: u32,
        /// Leaf-parallel wave width (python mcts_search batch_size mirror):
        /// default 256 = python's BATCH_SIZE; 1 = the serial mode.
        #[arg(long, default_value_t = search::DEFAULT_BATCH_SIZE)]
        batch_size: usize,
    },
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

fn cmd_planes(games: usize, seed: u64, moves: usize, out: &std::path::Path) {
    std::fs::create_dir_all(out)
        .unwrap_or_else(|e| fail(&format!("cannot create {}: {e}", out.display())));
    for idx in 0..games {
        let mut game = sba_game::SbaGame::fresh();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + idx as u64);
        let mut played: Vec<[i32; 2]> = Vec::new();
        for _ in 0..moves {
            let legal = game.legal();
            if legal.is_empty() {
                break; // the random run reached a terminal state — dump what we have
            }
            let k = legal[rng.gen_range(0..legal.len())];
            let (m, i) = (k / 9, k % 9);
            game.apply(m, i);
            played.push([m, i]);
        }
        let meta = serde_json::json!({ "moves": played, "seed": seed });
        let meta_path = out.join(format!("planes_g{idx:05}.meta.json"));
        std::fs::write(&meta_path, format!("{meta}\n"))
            .unwrap_or_else(|e| fail(&format!("cannot write {}: {e}", meta_path.display())));
        // 324 f32 little-endian, row-major 4x9x9 — the contract the python
        // side reads with np.fromfile(dtype="<f4")
        let mut bytes = Vec::with_capacity(324 * 4);
        for v in game.planes() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let bin_path = out.join(format!("planes_g{idx:05}.bin"));
        std::fs::write(&bin_path, bytes)
            .unwrap_or_else(|e| fail(&format!("cannot write {}: {e}", bin_path.display())));
    }
}

fn cmd_search(weights: &std::path::Path, moves: &str, sims: u32, batch_size: usize) {
    let mut game = sba_game::SbaGame::fresh();
    if !moves.trim().is_empty() {
        for step in moves.split(';') {
            let (m, i) = step.split_once(',').unwrap_or_else(|| {
                fail(&format!("bad --moves step '{step}' (want m,i;m,i)"))
            });
            let (m, i) = (
                m.trim().parse::<i32>().unwrap_or_else(|e| fail(&format!("bad m in '{step}': {e}"))),
                i.trim().parse::<i32>().unwrap_or_else(|e| fail(&format!("bad i in '{step}': {e}"))),
            );
            if game.apply(m, i) != sba_game::OK {
                fail(&format!("move ({m},{i}) is not legal here"));
            }
        }
    }

    let bytes = std::fs::read(weights)
        .unwrap_or_else(|e| fail(&format!("cannot read weights {}: {e}", weights.display())));
    let device = default_device();
    let model = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> AzNet<DefaultBackend> {
        let (channels, blocks) =
            infer_arch(&bytes).unwrap_or_else(|e| fail(&format!("weights load failed: {e}")));
        let mut model = AzNet::<DefaultBackend>::new(&device, channels, blocks);
        model.load_safetensors(&bytes, &device);
        model
    }))
    .unwrap_or_else(|_| fail("weights load failed (shape/name mismatch?)"));

    let (mut handle, _stats) = worker::spawn(model, device);
    let limits = search::SearchLimits {
        sims,
        c_puct: 1.5,
        dirichlet_eps: 0.0, // no prior noise: the search is fully deterministic
        batch_size,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(0); // unused at eps 0.0
    let (counts, value) = search::search_position(game, &mut handle, &limits, &mut rng)
        .unwrap_or_else(|e| fail(&format!("search failed: {e}")));
    let counts: serde_json::Map<String, serde_json::Value> = counts
        .into_iter()
        .map(|(k, v)| (k.to_string(), serde_json::json!(v)))
        .collect();
    println!("{}", serde_json::json!({ "counts": counts, "value": value }));
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Planes {
            games,
            seed,
            moves,
            out,
        } => cmd_planes(games, seed, moves, &out),
        Cmd::Search {
            weights,
            moves,
            sims,
            batch_size,
        } => cmd_search(&weights, &moves, sims, batch_size),
    }
}
