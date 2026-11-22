//! sba-core — the Rust core (D55 Phase A). See net.rs for the AZNet/weights
//! contract; records.rs for the training-record format; tree.rs for the
//! PUCT tree (wave.rs for the search engine, runner.rs for the self-play
//! runner — search.rs is a re-export shim); worker.rs for batched
//! inference. Bins: bench_burn, sba_selfplay, sba_debug (the parity-harness
//! planes/search dump).

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
