//! sba-core — the Rust core: net.rs (AZNet/weights), records.rs (record format),
//! tree.rs (PUCT tree), wave.rs (search engine), runner.rs (self-play), worker.rs
//! (batched inference), search.rs (re-export shim). Bins: bench_burn, sba_selfplay,
//! sba_debug.

pub mod net;
pub mod records;
pub mod runner;
pub mod search;
pub mod tree;
pub mod wave;
pub mod worker;

pub use net::{infer_arch, read_f32s, shape, AzNet, ResBlock};

#[cfg(all(not(feature = "wgpu"), not(feature = "cpu")))]
compile_error!("enable a backend feature: wgpu (default) or cpu");

#[cfg(feature = "wgpu")]
pub type DefaultBackend = burn::backend::Wgpu;
#[cfg(all(not(feature = "wgpu"), feature = "cpu"))]
pub type DefaultBackend = burn::backend::Candle<f32, i64>;

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
