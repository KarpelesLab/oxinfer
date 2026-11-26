//! oxinfer: CPU-first inference for Gemma 4 and Qwen 3.5/3.8 in pure Rust, with optional
//! runtime-detected accelerators.

// The kernels and model code index several fixed-size register tiles and buffers in lockstep with
// one loop counter; iterator chains there obscure the access pattern and can defeat unrolling.
#![allow(clippy::needless_range_loop)]

/// The oxideav-core version oxinfer's API uses (the `oxideav` feature), for hosts that want to
/// name the same types.
#[cfg(feature = "oxideav")]
pub use oxideav_core;

pub mod advisor;
/// `sincos` / `sincosf` for the fullrust target (the static release binary), whose
/// compiler-builtins has `sin` and `cos` but not these: LLVM merges a `sin` and a `cos` of one
/// value into them. `black_box` keeps the two calls here from being merged back into themselves.
#[cfg(target_os = "fullrust")]
mod fullrust_libm {
    use std::hint::black_box;
    /// # Safety
    /// `s` and `c` are valid for writes.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn sincos(x: f64, s: *mut f64, c: *mut f64) {
        unsafe {
            *s = black_box(x).sin();
            *c = black_box(x).cos();
        }
    }
    /// # Safety
    /// `s` and `c` are valid for writes.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn sincosf(x: f32, s: *mut f32, c: *mut f32) {
        unsafe {
            *s = black_box(x).sin();
            *c = black_box(x).cos();
        }
    }
}
#[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
pub mod anemll;
pub mod arktts;
mod attn;
#[cfg(target_arch = "aarch64")]
mod attn_prefill;
pub mod audio;
#[cfg(feature = "oxideav")]
pub mod av;
mod b64;
pub mod bagel;
pub mod batch;
pub mod bert;
#[cfg(feature = "capi")]
pub mod capi;
pub mod chat;
pub mod clock;
mod comfy_quant;
pub mod config;
pub mod constrain;
#[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
pub mod coreml;
#[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
pub mod cuda;
pub mod decide;
pub mod deepseek4;
pub mod drafter;
pub mod dspark;
pub mod embed;
pub mod embed_gemma2;
pub mod engine;
pub mod fish;
pub mod flux;
pub mod fsx;
pub mod generate;
pub mod ggml;
pub mod ggml_quant;
pub mod gguf;
pub mod glm5;
pub mod gpu;
pub mod hadamard;
pub mod haru;
#[cfg(feature = "hf")]
pub mod hf;
pub mod inkling;
pub mod inkling_draft;
pub mod inkling_vision;
pub mod intervene;
pub mod janus;
pub mod kernels;
pub mod kimi;
pub mod lfm3;
pub mod lfm3_omni;
pub mod longcat;
pub mod longcat_avatar;
pub mod longcat_chat;
pub mod lookup;
pub mod lora;
#[cfg(feature = "media")]
pub mod media;
#[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
pub mod metal;
pub mod minimax;
pub mod minimax_h3;
pub mod minimax_vision;
pub mod minimax_vl01;
pub mod mistral4;
pub mod mm;
pub mod mm_config;
pub mod model;
pub mod model_cache;
pub mod modernbert;
mod moe;
pub mod moss_image;
pub mod mpnet;
pub mod nemotron;
pub mod nemotron_asr;
pub mod nomic_bert;
pub mod omnigen2;
pub mod opencode;
pub mod pixtral;
pub mod platform;
pub mod png;
pub mod pool;
pub mod prof;
pub mod quant;
pub mod qwen;
pub mod qwen25_vision;
pub mod qwen4exp;
pub mod qwen_image;
pub mod qwen_mtp;
pub mod qwen_vision;
pub mod qwen_vl;
pub mod raon;
pub mod resample;
pub mod ripple;
pub mod sae;
pub mod safetensors;
pub mod sampler;
pub mod sd;
pub mod sd3;
#[cfg(all(feature = "server", not(target_arch = "wasm32")))]
pub mod server;
pub mod snac;
pub mod snapshot;
pub mod spec_cost;
pub mod step1x;
pub mod step_cache;
pub mod stream;
pub mod token_hook;
pub mod tokenizer;
pub mod tools;
pub mod torch;
pub mod torch_rng;
pub mod tts;
pub mod unlimited_ocr;
#[cfg(feature = "update")]
pub mod update;
pub mod vae_posterior;
pub mod video;
pub mod vision;
#[cfg(all(feature = "gpu", target_pointer_width = "64", not(target_arch = "wasm32")))]
pub mod vulkan;
#[cfg(all(feature = "web", target_arch = "wasm32", target_os = "unknown"))]
pub mod web;
pub mod whisper;
pub mod zonos;
