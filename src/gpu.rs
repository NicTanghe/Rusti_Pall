//! Platform GPU selection.
//!
//! On macOS the crate is built with Burn's `metal` feature, so the WGPU
//! runtime uses the Metal API and compiles kernels directly to MSL. Apple
//! Silicon exposes its GPU as an *integrated* adapter, so macOS uses the
//! default (high-performance) adapter instead of requiring a discrete GPU.
//! Elsewhere the first discrete adapter is used (e.g. NVIDIA via Vulkan).
//!
//! `CUBECL_WGPU_DEFAULT_DEVICE` (e.g. `IntegratedGpu(0)`, `DiscreteGpu(1)`)
//! overrides the adapter on macOS, where the default device is selected.

use burn::backend::wgpu::WgpuDevice;

/// Burn backend used for GPU inference (WGPU; native Metal/MSL on macOS).
pub type GpuBackend = burn::backend::Wgpu;

/// Device for denoiser sampling on the current platform.
pub fn device() -> WgpuDevice {
    if cfg!(target_os = "macos") {
        WgpuDevice::DefaultDevice
    } else {
        WgpuDevice::DiscreteGpu(0)
    }
}

/// Human-readable description of the GPU backend, for progress messages.
pub fn description() -> &'static str {
    if cfg!(target_os = "macos") {
        "Metal GPU"
    } else {
        "discrete WGPU adapter 0"
    }
}

/// Candle device for the FLAN-T5 text encoder: Metal on macOS, CPU elsewhere.
/// Falls back to the CPU if no Metal device can be created.
pub fn text_device() -> candle_core::Device {
    #[cfg(target_os = "macos")]
    match candle_core::Device::new_metal(0) {
        Ok(device) => return device,
        Err(e) => eprintln!("Metal unavailable for text encoding ({e}); using CPU"),
    }
    candle_core::Device::Cpu
}
