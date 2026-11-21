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
