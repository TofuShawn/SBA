//! D55 Phase-A spike: burn/wgpu inference parity + timing vs the torch
//! baselines (DML 1.92 ms, HIP 2.84 ms, CPU 27 ms per batch-256 forward).
//!
//! Usage: bench_burn <aznet.safetensors> <spike_input.bin>
//!                   [<spike_policy.bin> <spike_value.bin>] [--dump-out PREFIX]
//!
//! The batch size derives from the input file's length — any n works. With
//! all four positional files the parity line prints max abs errors vs the
//! torch reference bins (the D55 spike invocation, unchanged). With
//! --dump-out PREFIX the softmax(policy) rows ([n, 81]) and value rows
//! ([n]) are written as raw f32 LE so a non-torch process can compare
//! against its own reference (the Task 12 parity fuzz).

use burn::tensor::{Tensor, TensorData};
use sba_core::{default_device, infer_arch, read_f32s, shape, AzNet, DefaultBackend};

type B = DefaultBackend;

const SAMPLE_BYTES: usize = 4 * 9 * 9 * 4; // one 4×9×9 f32 plane set, LE

fn usage() -> ! {
    eprintln!(
        "usage: bench_burn <safetensors> <input.bin> \
         [<policy_ref.bin> <value_ref.bin>] [--dump-out PREFIX]"
    );
    std::process::exit(2);
}

fn main() {
    // --dump-out PREFIX is the one flag; everything else is positional
    let mut dump: Option<String> = None;
    let mut pos: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--dump-out" {
            dump = Some(args.next().unwrap_or_else(|| {
                eprintln!("--dump-out needs a PREFIX value");
                std::process::exit(2);
            }));
        } else {
            pos.push(a);
        }
    }
    let refs = match pos.len() {
        2 => None, // dump mode only (usage error below when --dump-out is absent)
        4 => Some((&pos[2], &pos[3])),
        _ => usage(),
    };
    if refs.is_none() && dump.is_none() {
        usage(); // neither reference bins nor a dump prefix: nothing to do
    }
    let (weights, input) = (&pos[0], &pos[1]);

    let device = default_device();
    println!("device: {device:?}");

    let t0 = std::time::Instant::now();
    let weights_bytes = std::fs::read(weights).unwrap();
    // size the model to the blob (the spike used 128/5; the parity fuzz
    // loads the AZNet(9, 8, 1) test net) — the infer_arch pattern the
    // sibling exes share
    let (channels, blocks) = infer_arch(&weights_bytes)
        .unwrap_or_else(|e| panic!("weights load failed: {e}"));
    let mut model = AzNet::<B>::new(&device, channels, blocks);
    model.load_safetensors(&weights_bytes, &device);
    println!("weights loaded in {:?}", t0.elapsed());

    let raw = std::fs::read(input).unwrap();
    let n = raw.len() / SAMPLE_BYTES;
    assert!(
        n > 0 && raw.len() % SAMPLE_BYTES == 0,
        "input must hold n×{SAMPLE_BYTES} f32 LE bytes ({} = {n} samples)",
        raw.len()
    );
    let x = Tensor::<B, 4>::from_data(
        TensorData::new(read_f32s(&raw), shape(&[n, 4, 9, 9])),
        &device,
    );

    let (p, v) = model.forward(x.clone());

    // --dump-out PREFIX: write softmax(policy) + value rows as f32 LE so a
    // non-torch process can compare against its own reference
    if let Some(prefix) = &dump {
        let probs = burn::tensor::activation::softmax(p.clone(), 1);
        std::fs::write(format!("{prefix}_policy.bin"), probs.into_data().as_bytes()).unwrap();
        std::fs::write(format!("{prefix}_value.bin"), v.clone().into_data().as_bytes()).unwrap();
    }

    // parity vs the torch reference bins (the D55 spike invocation)
    if let Some((policy_ref, value_ref)) = refs {
        let got_p = p.into_data().to_vec::<f32>().unwrap();
        let got_v = v.into_data().to_vec::<f32>().unwrap();
        let want_p = read_f32s(&std::fs::read(policy_ref).unwrap());
        let want_v = read_f32s(&std::fs::read(value_ref).unwrap());
        let err_p = got_p
            .iter()
            .zip(&want_p)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let err_v = got_v
            .iter()
            .zip(&want_v)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("parity: policy_max_err={err_p:.2e} value_max_err={err_v:.2e}");

        // timing: into_data() per iteration forces the readback sync, so each
        // sample is a full submit+compute+download round trip. Spike-only —
        // the dump mode is the CI parity path; one forward is enough there.
        for _ in 0..5 {
            let (p, _) = model.forward(x.clone());
            p.into_data();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..50 {
            let (p, _) = model.forward(x.clone());
            p.into_data();
        }
        println!(
            "burn/wgpu batch{n} (synced): {:.2} ms/batch",
            t0.elapsed().as_secs_f64() / 50.0 * 1000.0
        );
    }
}
