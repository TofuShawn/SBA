//! CLI contract tests for the sba_selfplay exe (spec §5.1/§5.2): JSONL
//! events on stdout, exit codes, record naming, and same-seed bitwise
//! record determinism. The weights are synthesized in-test — a seeded-LCG
//! fill of every tensor name `net.rs::load_safetensors` reads for
//! `AzNet::new(channels=8, blocks=1)` — so the suite runs on the cpu
//! backend with no GPU and no torch checkpoint.

use std::collections::HashMap;
use std::process::Command;
use safetensors::tensor::TensorView;
use safetensors::Dtype;

fn run(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_sba_selfplay"))
        .args(args)
        .output()
        .expect("spawn sba_selfplay");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn tiny_run_produces_events_and_records() {
    let dir = std::env::temp_dir().join("sba_selfplay_cli_test");
    let _ = std::fs::remove_dir_all(&dir);
    let weights = dir.join("w.safetensors");
    write_random_weights(&weights);
    let (code, stdout, stderr) = run(&[
        "--weights",
        weights.to_str().unwrap(),
        "--out",
        dir.join("games").to_str().unwrap(),
        "--games",
        "2",
        "--sims",
        "8",
        "--threads",
        "2",
        "--seed-base",
        "5",
    ]);
    assert_eq!(code, 0, "stderr: {stderr}");
    let lines: Vec<serde_json::Value> = stdout
        .lines()
        .map(|l| serde_json::from_str(l).expect("jsonl"))
        .collect();
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
    let common = [
        "--weights",
        weights.to_str().unwrap(),
        "--games",
        "1",
        "--sims",
        "8",
        "--threads",
        "1",
    ];
    let a = dir.join("a");
    let b = dir.join("b");
    let extra_a = ["--seed-base", "9", "--out", a.to_str().unwrap()];
    let (ca, _, sa) = run(&[&common[..], &extra_a[..]].concat());
    let extra_b = ["--seed-base", "9", "--out", b.to_str().unwrap()];
    let (cb, _, _) = run(&[&common[..], &extra_b[..]].concat());
    assert_eq!((ca, cb), (0, 0), "stderr a: {sa}");
    assert_eq!(
        std::fs::read(a.join("g00000.safetensors")).unwrap(),
        std::fs::read(b.join("g00000.safetensors")).unwrap()
    );
}

#[test]
fn normal_game_type_is_usage_error() {
    let dir = std::env::temp_dir().join("sba_selfplay_cli_err");
    let _ = std::fs::remove_dir_all(&dir);
    let weights = dir.join("w.safetensors");
    write_random_weights(&weights);
    let (code, _, _) = run(&[
        "--weights",
        weights.to_str().unwrap(),
        "--out",
        dir.to_str().unwrap(),
        "--games",
        "1",
        "--sims",
        "2",
        "--game-type",
        "normal",
    ]);
    assert_eq!(code, 2);
}

#[test]
fn weights_parse_failure_emits_error_event() {
    let dir = std::env::temp_dir().join("sba_selfplay_cli_err3");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let weights = dir.join("bad.safetensors");
    std::fs::write(&weights, b"definitely not safetensors").unwrap();
    let (code, stdout, stderr) = run(&[
        "--weights",
        weights.to_str().unwrap(),
        "--out",
        dir.join("games").to_str().unwrap(),
        "--games",
        "1",
    ]);
    assert_eq!(code, 3, "stderr: {stderr}");
    assert!(!stderr.is_empty(), "diagnostic expected on stderr");
    // weights failures precede the worker: the stream is the bare error
    let lines: Vec<serde_json::Value> = stdout
        .lines()
        .map(|l| serde_json::from_str(l).expect("jsonl"))
        .collect();
    assert_eq!(lines.len(), 1, "no start event may precede a weights failure");
    assert_eq!(lines[0]["type"], "error");
    assert!(!lines[0]["message"].as_str().unwrap_or("").is_empty());
}

#[test]
fn blob_without_blocks_is_rejected() {
    let dir = std::env::temp_dir().join("sba_selfplay_cli_noblocks");
    let _ = std::fs::remove_dir_all(&dir);
    let weights = dir.join("w.safetensors");
    write_weights(&weights, false); // conv_in + heads, NO blocks.* names
    let (code, stdout, stderr) = run(&[
        "--weights",
        weights.to_str().unwrap(),
        "--out",
        dir.join("games").to_str().unwrap(),
        "--games",
        "1",
    ]);
    assert_eq!(code, 3, "stderr: {stderr}");
    assert!(
        stderr.contains("no residual blocks"),
        "diagnostic must name the cause: {stderr}"
    );
    let lines: Vec<serde_json::Value> = stdout
        .lines()
        .map(|l| serde_json::from_str(l).expect("jsonl"))
        .collect();
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["type"], "error");
}

#[test]
fn record_write_failure_emits_error_event() {
    let dir = std::env::temp_dir().join("sba_selfplay_cli_err4");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let weights = dir.join("w.safetensors");
    write_random_weights(&weights);
    // a regular file where the out DIRECTORY should live: the game runs,
    // then create_dir_all fails when writing the record -> exit 4
    let blocker = dir.join("blocker");
    std::fs::write(&blocker, b"").unwrap();
    let (code, stdout, stderr) = run(&[
        "--weights",
        weights.to_str().unwrap(),
        "--out",
        blocker.join("games").to_str().unwrap(),
        "--games",
        "1",
        "--sims",
        "2",
    ]);
    assert_eq!(code, 4, "stderr: {stderr}");
    assert!(stderr.contains("record write failed"), "stderr: {stderr}");
    let lines: Vec<serde_json::Value> = stdout
        .lines()
        .map(|l| serde_json::from_str(l).expect("jsonl"))
        .collect();
    assert_eq!(lines[0]["type"], "start");
    let last = lines.last().unwrap();
    assert_eq!(last["type"], "error");
    assert!(last["message"].as_str().unwrap_or("").contains("record write failed"));
}

/// Seeded LCG step -> f32 in [0, 1). Integer-only state, so the values are
/// identical on every platform and every run — the same-seed bitwise test
/// rides on that.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / (u32::MAX as f32)
}

/// Insert one synthetic tensor. BN running_var maps into [0.5, 1.0) so
/// inference-mode BN never sees a non-positive variance; everything else
/// lands in [-0.1, 0.1) (small logits keep the softmax well-behaved).
fn put_tensor(
    tensors: &mut HashMap<String, (Vec<usize>, Vec<f32>)>,
    state: &mut u64,
    name: &str,
    dims: Vec<usize>,
    var: bool,
) {
    let n: usize = dims.iter().product();
    let vals = (0..n)
        .map(|_| {
            let r = lcg(state);
            if var {
                0.5 + 0.5 * r
            } else {
                (r - 0.5) * 0.2
            }
        })
        .collect();
    tensors.insert(name.to_string(), (dims, vals));
}

/// Synthesize a loadable AzNet blob: every tensor name `net.rs::
/// load_safetensors` reads with the shapes `AzNet::new(_, channels=8,
/// blocks=1)` builds. The FC weights are stored in the torch [out, in]
/// layout (pol_fc [81, 648], val_fc1 [64, 648], val_fc2 [1, 64]) — the
/// loader transposes them into burn's [in, out], exactly mirroring the
/// torch-saved blob (`logs/aznet.safetensors`) the parity spike proved
/// out. Serialized via safetensors::serialize with a metadata map, which
/// the exe ignores.
fn write_random_weights(path: &std::path::Path) {
    write_weights(path, true);
}

/// `with_blocks = false` writes conv_in + heads but NO `blocks.*` names —
/// the zero-block blob the exe must refuse (exit 3) instead of silently
/// loading a blockless (degraded) net.
fn write_weights(path: &std::path::Path, with_blocks: bool) {
    let mut tensors = HashMap::new();
    let mut state = 0x9E37_79B9_7F4A_7C15u64;

    put_tensor(&mut tensors, &mut state, "conv_in.weight", vec![8, 4, 3, 3], false);
    put_tensor(&mut tensors, &mut state, "conv_in.bias", vec![8], false);
    put_tensor(&mut tensors, &mut state, "bn_in.weight", vec![8], false);
    put_tensor(&mut tensors, &mut state, "bn_in.bias", vec![8], false);
    put_tensor(&mut tensors, &mut state, "bn_in.running_mean", vec![8], false);
    put_tensor(&mut tensors, &mut state, "bn_in.running_var", vec![8], true);

    if with_blocks {
        put_tensor(&mut tensors, &mut state, "blocks.0.conv1.weight", vec![8, 8, 3, 3], false);
        put_tensor(&mut tensors, &mut state, "blocks.0.conv1.bias", vec![8], false);
        put_tensor(&mut tensors, &mut state, "blocks.0.bn1.weight", vec![8], false);
        put_tensor(&mut tensors, &mut state, "blocks.0.bn1.bias", vec![8], false);
        put_tensor(&mut tensors, &mut state, "blocks.0.bn1.running_mean", vec![8], false);
        put_tensor(&mut tensors, &mut state, "blocks.0.bn1.running_var", vec![8], true);
        put_tensor(&mut tensors, &mut state, "blocks.0.conv2.weight", vec![8, 8, 3, 3], false);
        put_tensor(&mut tensors, &mut state, "blocks.0.conv2.bias", vec![8], false);
        put_tensor(&mut tensors, &mut state, "blocks.0.bn2.weight", vec![8], false);
        put_tensor(&mut tensors, &mut state, "blocks.0.bn2.bias", vec![8], false);
        put_tensor(&mut tensors, &mut state, "blocks.0.bn2.running_mean", vec![8], false);
        put_tensor(&mut tensors, &mut state, "blocks.0.bn2.running_var", vec![8], true);
    }

    put_tensor(&mut tensors, &mut state, "pol_conv.weight", vec![8, 8, 1, 1], false);
    put_tensor(&mut tensors, &mut state, "pol_conv.bias", vec![8], false);
    put_tensor(&mut tensors, &mut state, "pol_bn.weight", vec![8], false);
    put_tensor(&mut tensors, &mut state, "pol_bn.bias", vec![8], false);
    put_tensor(&mut tensors, &mut state, "pol_bn.running_mean", vec![8], false);
    put_tensor(&mut tensors, &mut state, "pol_bn.running_var", vec![8], true);
    put_tensor(&mut tensors, &mut state, "pol_fc.weight", vec![81, 648], false);
    put_tensor(&mut tensors, &mut state, "pol_fc.bias", vec![81], false);

    put_tensor(&mut tensors, &mut state, "val_conv.weight", vec![8, 8, 1, 1], false);
    put_tensor(&mut tensors, &mut state, "val_conv.bias", vec![8], false);
    put_tensor(&mut tensors, &mut state, "val_bn.weight", vec![8], false);
    put_tensor(&mut tensors, &mut state, "val_bn.bias", vec![8], false);
    put_tensor(&mut tensors, &mut state, "val_bn.running_mean", vec![8], false);
    put_tensor(&mut tensors, &mut state, "val_bn.running_var", vec![8], true);
    put_tensor(&mut tensors, &mut state, "val_fc1.weight", vec![64, 648], false);
    put_tensor(&mut tensors, &mut state, "val_fc1.bias", vec![64], false);
    put_tensor(&mut tensors, &mut state, "val_fc2.weight", vec![1, 64], false);
    put_tensor(&mut tensors, &mut state, "val_fc2.bias", vec![1], false);

    // the views borrow the buffers, so the owned tensors must outlive the
    // map that hands them to safetensors::serialize
    let owned: Vec<(String, Vec<usize>, Vec<f32>)> = tensors
        .into_iter()
        .map(|(name, (dims, vals))| (name, dims, vals))
        .collect();
    let data: HashMap<String, TensorView> = owned
        .iter()
        .map(|(name, dims, vals)| {
            (
                name.clone(),
                TensorView::new(Dtype::F32, dims.clone(), bytemuck::cast_slice(vals))
                    .expect("tensor view"),
            )
        })
        .collect();
    let metadata = HashMap::from([
        ("format_version".to_string(), "1".to_string()),
        (
            "source".to_string(),
            "synthetic LCG test weights (channels=8, blocks=1)".to_string(),
        ),
    ]);
    let blob = safetensors::serialize(data, &Some(metadata)).expect("serialize weights");
    std::fs::create_dir_all(path.parent().expect("weights parent")).expect("create dir");
    std::fs::write(path, blob).expect("write weights");
}
