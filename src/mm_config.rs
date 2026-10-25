//! Multimodal (vision / audio) configuration from `config.json`.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

fn d64() -> usize {
    64
}
fn d_eps() -> f64 {
    1e-6
}
fn d16() -> usize {
    16
}
fn d3() -> usize {
    3
}
fn d10240() -> usize {
    10240
}
fn d280() -> usize {
    280
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionRope {
    #[serde(default = "d_theta100")]
    pub rope_theta: f64,
}
fn d_theta100() -> f64 {
    100.0
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "d64")]
    pub head_dim: usize,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "d16")]
    pub patch_size: usize,
    #[serde(default = "d3")]
    pub pooling_kernel_size: usize,
    #[serde(default = "d10240")]
    pub position_embedding_size: usize,
    #[serde(default)]
    pub use_clipped_linears: bool,
    #[serde(default)]
    pub standardize: bool,
    #[serde(default = "d280")]
    pub default_output_length: usize,
    #[serde(default)]
    pub rope_parameters: Option<VisionRope>,
}

impl VisionConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.as_ref().map(|r| r.rope_theta).unwrap_or(100.0)
    }
}

fn d1024() -> usize {
    1024
}
fn d12() -> usize {
    12
}
fn d8() -> usize {
    8
}
fn d5() -> usize {
    5
}
fn d_half() -> f64 {
    0.5
}
fn d13() -> usize {
    13
}
fn d50() -> f64 {
    50.0
}
fn d_neg1e9() -> f64 {
    -1.0e9
}
fn d1536() -> usize {
    1536
}
fn d_sub() -> Vec<usize> {
    vec![128, 32]
}
fn d_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct AudioConfig {
    #[serde(default = "d1024")]
    pub hidden_size: usize,
    #[serde(default = "d12")]
    pub num_hidden_layers: usize,
    #[serde(default = "d8")]
    pub num_attention_heads: usize,
    #[serde(default = "d5")]
    pub conv_kernel_size: usize,
    #[serde(default = "d_half")]
    pub residual_weight: f64,
    #[serde(default = "d12")]
    pub attention_chunk_size: usize,
    #[serde(default = "d13")]
    pub attention_context_left: usize,
    #[serde(default)]
    pub attention_context_right: usize,
    #[serde(default = "d50")]
    pub attention_logit_cap: f64,
    #[serde(default = "d_neg1e9")]
    pub attention_invalid_logits_value: f64,
    #[serde(default = "d_true")]
    pub use_clipped_linears: bool,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "d1536")]
    pub output_proj_dims: usize,
    #[serde(default = "d_sub")]
    pub subsampling_conv_channels: Vec<usize>,
}

/// Top-level multimodal settings: tower configs and placeholder token ids.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MmConfig {
    pub vision: Option<VisionConfig>,
    pub audio: Option<AudioConfig>,
    pub image_token_id: u32,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
    pub audio_token_id: u32,
    pub boa_token_id: u32,
    pub eoa_token_id: u32,
    pub video_token_id: u32,
    pub vision_soft_tokens_per_image: usize,
}

impl MmConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let raw: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&path)?)?;
        let id = |k: &str, default: u32| raw.get(k).and_then(|v| v.as_u64()).map(|v| v as u32).unwrap_or(default);
        let vision = match raw.get("vision_config") {
            Some(v) if !v.is_null() => Some(serde_json::from_value(v.clone()).context("vision_config")?),
            _ => None,
        };
        let audio = match raw.get("audio_config") {
            Some(v) if !v.is_null() => Some(serde_json::from_value(v.clone()).context("audio_config")?),
            _ => None,
        };
        Ok(Self {
            vision,
            audio,
            image_token_id: id("image_token_id", 258880),
            boi_token_id: id("boi_token_id", 255999),
            eoi_token_id: id("eoi_token_id", 258882),
            audio_token_id: id("audio_token_id", 258881),
            boa_token_id: id("boa_token_id", 256000),
            eoa_token_id: id("eoa_token_id", 258883),
            video_token_id: id("video_token_id", 258884),
            vision_soft_tokens_per_image: raw
                .get("vision_soft_tokens_per_image")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(280),
        })
    }
}
