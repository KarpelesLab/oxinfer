//! oxinfer: CPU-first inference for Gemma 4 and Qwen 3.5/3.8 in pure Rust, with optional
//! runtime-detected accelerators.

// The kernels and model code index several fixed-size register tiles and buffers in lockstep with
// one loop counter; iterator chains there obscure the access pattern and can defeat unrolling.
#![allow(clippy::needless_range_loop)]

/// The oxideav-core version oxinfer's API uses (the `oxideav` feature), for hosts that want to
/// name the same types.
#[cfg(feature = "oxideav")]
pub use oxideav_core;

#[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
pub mod anemll;
pub mod audio;
#[cfg(feature = "oxideav")]
pub mod av;
mod b64;
pub mod bert;
#[cfg(feature = "capi")]
pub mod capi;
pub mod chat;
pub mod clock;
pub mod config;
#[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
pub mod coreml;
#[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
pub mod cuda;
pub mod decide;
pub mod drafter;
pub mod embed;
pub mod engine;
pub mod fsx;
pub mod generate;
pub mod ggml;
pub mod gguf;
pub mod gpu;
#[cfg(feature = "hf")]
pub mod hf;
pub mod janus;
pub mod kernels;
pub mod lora;
#[cfg(feature = "media")]
pub mod media;
#[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
pub mod metal;
pub mod mm;
pub mod mm_config;
pub mod model;
pub mod modernbert;
mod moe;
pub mod mpnet;
pub mod nomic_bert;
pub mod platform;
pub mod png;
pub mod pool;
pub mod prof;
pub mod quant;
pub mod qwen;
pub mod qwen_mtp;
pub mod qwen_vision;
pub mod resample;
pub mod safetensors;
pub mod sampler;
pub mod sd;
pub mod snac;
pub mod stream;
pub mod tools;
pub mod torch;
pub mod tts;
pub mod video;
pub mod vision;
#[cfg(all(feature = "web", target_arch = "wasm32", target_os = "unknown"))]
pub mod web;
pub mod whisper;
