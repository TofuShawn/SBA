//! sba-core — the Rust core (D55 Phase A). The AZNet policy/value network
//! hand-written in burn: weights load from the PyTorch-saved safetensors
//! file 1:1 (param names mirror `src/alphazero.py`'s state_dict exactly —
//! `blocks.0.conv1.weight`, `bn_in.running_mean`, …) with one deliberate
//! delta: burn's Linear weight is [in, out] vs torch's [out, in], so the
//! three FC weights transpose on load. Conv layouts match as-is. The forward mirrors `AZNet.forward`:
//! relu(bn(conv)) stacks, residual adds, policy logits [B, 81] and a
//! tanh value [B, 1].
//!
//! Backend notes (burn 0.16): on a plain (non-autodiff) backend BatchNorm
//! always runs `forward_inference` — running statistics, the same contract
//! as torch `.eval()` — so there is no train/valid toggle here. BN affine
//! params are named gamma/beta; running stats are `RunningState` values
//! built from Params via `from_record`.
//!
//! The wgpu backend keeps the inference path pure Rust — no C++ runtime —
//! and the backend choice is one type alias away from Cuda/Tch/CPU.

use burn::module::{Module, Param, RunningState};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use safetensors::SafeTensors;

#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
}

impl<B: Backend> ResBlock<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = relu(self.bn1.forward(self.conv1.forward(x.clone())));
        let y = self.bn2.forward(self.conv2.forward(y));
        relu(x.add(y))
    }
}

#[derive(Module, Debug)]
pub struct AzNet<B: Backend> {
    conv_in: Conv2d<B>,
    bn_in: BatchNorm<B, 2>,
    blocks: Vec<ResBlock<B>>,
    pol_conv: Conv2d<B>,
    pol_bn: BatchNorm<B, 2>,
    pol_fc: Linear<B>,
    val_conv: Conv2d<B>,
    val_bn: BatchNorm<B, 2>,
    val_fc1: Linear<B>,
    val_fc2: Linear<B>,
}

/// A `Shape` from slice form.
pub fn shape(dims: &[usize]) -> Shape {
    Shape::from(dims.to_vec())
}

impl<B: Backend> AzNet<B> {
    pub fn new(device: &B::Device, channels: usize, blocks: usize) -> Self {
        Self {
            conv_in: Conv2dConfig::new([4, channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn_in: BatchNormConfig::new(channels).init(device),
            blocks: (0..blocks)
                .map(|_| {
                    ResBlock {
                        conv1: Conv2dConfig::new([channels, channels], [3, 3])
                            .with_padding(PaddingConfig2d::Same)
                            .init(device),
                        bn1: BatchNormConfig::new(channels).init(device),
                        conv2: Conv2dConfig::new([channels, channels], [3, 3])
                            .with_padding(PaddingConfig2d::Same)
                            .init(device),
                        bn2: BatchNormConfig::new(channels).init(device),
                    }
                })
                .collect(),
            pol_conv: Conv2dConfig::new([channels, 8], [1, 1])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            pol_bn: BatchNormConfig::new(8).init(device),
            pol_fc: LinearConfig::new(8 * 81, 81).init(device),
            val_conv: Conv2dConfig::new([channels, 8], [1, 1])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            val_bn: BatchNormConfig::new(8).init(device),
            val_fc1: LinearConfig::new(8 * 81, 64).init(device),
            val_fc2: LinearConfig::new(64, 1).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut x = relu(self.bn_in.forward(self.conv_in.forward(x)));
        for block in &self.blocks {
            x = block.forward(x);
        }
        let p = relu(self.pol_bn.forward(self.pol_conv.forward(x.clone())));
        let p = self.pol_fc.forward(p.flatten(1, 3));
        let v = relu(self.val_bn.forward(self.val_conv.forward(x.clone())));
        let v = self.val_fc2.forward(relu(self.val_fc1.forward(v.flatten(1, 3))));
        (p, v.tanh())
    }

    /// Overwrite every parameter from a PyTorch-saved safetensors blob.
    /// Torch names map 1:1 onto the fields above; BN `running_*` stats ride
    /// through `RunningState::from_record` (eval mode uses them, exactly
    /// like torch `.eval()`).
    pub fn load_safetensors(&mut self, bytes: &[u8], device: &B::Device) {
        let st = SafeTensors::deserialize(bytes).expect("safetensors parse");
        let get = |name: &str| -> (Vec<f32>, Vec<usize>) {
            let view = st
                .tensor(name)
                .unwrap_or_else(|_| panic!("missing weight: {name}"));
            assert_eq!(view.dtype(), safetensors::Dtype::F32, "{}", name);
            let f32s: &[f32] = bytemuck::cast_slice(view.data());
            (f32s.to_vec(), view.shape().to_vec())
        };
        let t4 = |name: &str, device: &B::Device| -> Tensor<B, 4> {
            let (v, s) = get(name);
            Tensor::from_data(TensorData::new(v, shape(&s)), device)
        };
        let t2 = |name: &str, device: &B::Device| -> Tensor<B, 2> {
            let (v, s) = get(name);
            Tensor::from_data(TensorData::new(v, shape(&s)), device)
        };
        let t1 = |name: &str, device: &B::Device| -> Tensor<B, 1> {
            let (v, s) = get(name);
            Tensor::from_data(TensorData::new(v, shape(&s)), device)
        };
        let lin_bias = |name: &str, device: &B::Device| -> Option<Param<Tensor<B, 1>>> {
            Some(Param::from_tensor(t1(name, device)))
        };
        let bn = |m: &mut BatchNorm<B, 2>, p: &str, device: &B::Device| {
            m.gamma = Param::from_tensor(t1(&format!("{p}.weight"), device));
            m.beta = Param::from_tensor(t1(&format!("{p}.bias"), device));
            m.running_mean =
                RunningState::from_record(Param::from_tensor(t1(&format!("{p}.running_mean"), device)));
            m.running_var =
                RunningState::from_record(Param::from_tensor(t1(&format!("{p}.running_var"), device)));
        };

        self.conv_in.weight = Param::from_tensor(t4("conv_in.weight", device));
        self.conv_in.bias = Some(Param::from_tensor(t1("conv_in.bias", device)));
        bn(&mut self.bn_in, "bn_in", device);
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let p = format!("blocks.{i}");
            block.conv1.weight = Param::from_tensor(t4(&format!("{p}.conv1.weight"), device));
            block.conv1.bias = Some(Param::from_tensor(t1(&format!("{p}.conv1.bias"), device)));
            bn(&mut block.bn1, &format!("{p}.bn1"), device);
            block.conv2.weight = Param::from_tensor(t4(&format!("{p}.conv2.weight"), device));
            block.conv2.bias = Some(Param::from_tensor(t1(&format!("{p}.conv2.bias"), device)));
            bn(&mut block.bn2, &format!("{p}.bn2"), device);
        }
        self.pol_conv.weight = Param::from_tensor(t4("pol_conv.weight", device));
        self.pol_conv.bias = Some(Param::from_tensor(t1("pol_conv.bias", device)));
        bn(&mut self.pol_bn, "pol_bn", device);
        self.pol_fc.weight = Param::from_tensor(t2("pol_fc.weight", device).transpose());
        self.pol_fc.bias = lin_bias("pol_fc.bias", device);
        self.val_conv.weight = Param::from_tensor(t4("val_conv.weight", device));
        self.val_conv.bias = Some(Param::from_tensor(t1("val_conv.bias", device)));
        bn(&mut self.val_bn, "val_bn", device);
        self.val_fc1.weight = Param::from_tensor(t2("val_fc1.weight", device).transpose());
        self.val_fc1.bias = lin_bias("val_fc1.bias", device);
        self.val_fc2.weight = Param::from_tensor(t2("val_fc2.weight", device).transpose());
        self.val_fc2.bias = lin_bias("val_fc2.bias", device);
    }

    /// Load the safetensors blob. On a plain (non-autodiff) backend the BN
    /// layers run in inference mode automatically — no toggle required.
    pub fn from_safetensors(bytes: &[u8], device: &B::Device) -> Self {
        let mut model = Self::new(device, 128, 5);
        model.load_safetensors(bytes, device);
        model
    }
}

/// Raw f32 helper for the parity checks (shape verified by the caller).
pub fn read_f32s(bytes: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice(bytes).to_vec()
}

/// The blob's architecture, read off the safetensors header:
/// conv_in.weight is [channels, 4, 3, 3] and every block contributes one
/// `blocks.{i}.conv1.weight`. AzNet::from_safetensors pins the torch
/// checkpoint's 128/5; self-play blobs may be smaller (the CLI tests load
/// a channels=8, blocks=1 net), so callers size the model to the file.
/// (Single-sourced here for both exes — moved out of sba_selfplay.rs.)
pub fn infer_arch(bytes: &[u8]) -> Result<(usize, usize), String> {
    let st = SafeTensors::deserialize(bytes).map_err(|e| e.to_string())?;
    let channels = st
        .tensor("conv_in.weight")
        .map_err(|_| String::from("missing tensor: conv_in.weight"))?
        .shape()[0];
    let blocks = st
        .names()
        .iter()
        .filter(|n| n.starts_with("blocks.") && n.ends_with(".conv1.weight"))
        .count();
    if blocks == 0 {
        // a conv_in + heads blob without residual blocks would load and run
        // as a silently degraded model — refuse it (genuine corruption like
        // index gaps still fails via panic -> catch_unwind -> exit 3)
        return Err(String::from(
            "blob looks like it has no residual blocks (no blocks.*.conv1.weight names)",
        ));
    }
    Ok((channels, blocks))
}
