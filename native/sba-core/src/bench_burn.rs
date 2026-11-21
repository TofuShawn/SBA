//! D55 Phase-A spike: burn/wgpu inference parity + timing vs the torch
//! baselines (DML 1.92 ms, HIP 2.84 ms, CPU 27 ms per batch-256 forward).
//!
//! Usage: bench_burn.exe <aznet.safetensors> <spike_input.bin>
//!                     <spike_policy.bin> <spike_value.bin>

use burn::tensor::{Tensor, TensorData};
use sba_core::{default_device, read_f32s, shape, AzNet, DefaultBackend};

type B = DefaultBackend;

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.len() != 4 {
        panic!("usage: bench_burn <safetensors> <input.bin> <policy.bin> <value.bin>");
    }
    let (weights, input, policy_ref, value_ref) = (&paths[0], &paths[1], &paths[2], &paths[3]);

    let device = default_device();
    println!("device: {device:?}");

    let t0 = std::time::Instant::now();
    let model = AzNet::<B>::from_safetensors(&std::fs::read(weights).unwrap(), &device);
    println!("weights loaded in {:?}", t0.elapsed());

    let raw = std::fs::read(input).unwrap();
    let x = Tensor::<B, 4>::from_data(
        TensorData::new(read_f32s(&raw), shape(&[256, 4, 9, 9])),
        &device,
    );

    // parity vs the torch reference (same fixed batch)
    let (p, v) = model.forward(x.clone());
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
    // sample is a full submit+compute+download round trip
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
        "burn/wgpu batch256 (synced): {:.2} ms/batch",
        t0.elapsed().as_secs_f64() / 50.0 * 1000.0
    );
}
