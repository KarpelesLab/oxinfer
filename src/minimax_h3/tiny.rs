//! Tiny random MiniMax-H3 and MiniMax-H3-Image releases, for tests that run the pipelines end to
//! end (the OpenAI-compatible server's video and image routes, the GUI) without the real 141 GB /
//! 127 GB checkpoints. The weights are noise from a fixed-seed generator, scaled so activations
//! stay finite (norm weights near 1, matrices uniform with `sqrt(3 / fan_in)` times a gain), so
//! the outputs are deterministic but meaningless.
//!
//! [`write_tiny_video_release`] writes one folder in diffusers' layout (`MiniMaxH3ModularPipeline`,
//! no `partition`) that serves every task: text to video, first / last frame to video and
//! `ref2va` (the diffusers layout has no `FL2VA/` / `Ref2VA/` split, so
//! [`super::video::VideoPipeline::generate_ref`] reads the same `transformer/`):
//!
//! - `model_index.json`, `scheduler/` and `audio_scheduler/` (shifts 12 and 3);
//! - `transformer/`: the DiT in diffusers' tensor names (2 blocks of 32 wide, 1 refiner block,
//!   8 latent channels, 8 audio channels, text width 16);
//! - `vae/`: the video VAE, its causal 3D CNN encoder (4 levels, 4 to 8 channels, 16x in space,
//!   4x in time) and its one-block ViT decoder;
//! - `audio_vae/`: the audio VAE at 32 kHz, 800 samples per latent (BigVGAN decoder and DAC
//!   encoder of 4 to 16 channels);
//! - `text_encoder/`: a Qwen3-VL of 51 layers of 16 wide (the pipelines read the residual stream
//!   after layer 50) with its vision tower (2 blocks, one DeepStack merger) and Qwen's
//!   vocabulary size (151676 rows, so MiniMax's added `<d>` ... ids are in range), with
//!   `tokenizer.json` and a processor config of a small pixel budget;
//! - `tokenizer/tokenizer.json`: a byte-level tokenizer (one token per byte, no merges) with
//!   Qwen's special tokens at their ids.
//!
//! [`write_tiny_image_release`] writes a `MiniMaxH3ImagePipeline` folder: `transformer/` (no audio
//! stream), three per-step output heads in `pdd_fused_heads.safetensors`, `pdd_config.json`,
//! `vae/` (the image decoder), `tokenizer/` and the same `text_encoder/`.
//!
//! The text encoder sits in each release's `text_encoder/`, where the CLI's lookup
//! (`text_encoder()` of `minimax_h3_cli.rs`) finds it: `Sources { dir: Some(dir), text_encoder:
//! dir.join("text_encoder"), .. }` and `Pipeline::load(dir, &dir.join("text_encoder"))`.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::path::Path;

/// The generator of `crate::raon`'s test checkpoint: uniform in `[-1, 1)`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// Tensors of one safetensors file, written by hand.
struct Tensors {
    list: Vec<(String, &'static str, Vec<usize>, Vec<u8>)>,
    /// The header's `__metadata__` entries.
    meta: serde_json::Map<String, Value>,
    rng: Rng,
}

impl Tensors {
    fn new(seed: u64) -> Self {
        Self {
            list: Vec::new(),
            meta: serde_json::Map::new(),
            rng: Rng(seed),
        }
    }

    fn noise(&mut self, n: usize, scale: f32, bias: f32) -> Vec<f32> {
        (0..n).map(|_| bias + scale * self.rng.next()).collect()
    }

    /// Float32 values.
    fn put(&mut self, name: &str, shape: &[usize], v: &[f32]) {
        debug_assert_eq!(v.len(), shape.iter().product::<usize>(), "{name}");
        let bytes = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.list.push((name.to_string(), "F32", shape.to_vec(), bytes));
    }

    /// `bias + scale * U(-1, 1)` in float32.
    fn uni(&mut self, name: &str, shape: &[usize], scale: f32, bias: f32) {
        let v = self.noise(shape.iter().product(), scale, bias);
        self.put(name, shape, &v);
    }

    /// A weight whose fan-in is the product of all but its first dimension: unit-variance outputs
    /// for unit-variance inputs, times `gain`.
    fn mat(&mut self, name: &str, shape: &[usize], gain: f32) {
        let fan_in: usize = shape[1..].iter().product();
        self.uni(name, shape, gain * (3.0 / fan_in as f32).sqrt(), 0.0);
    }

    /// [`Tensors::mat`] whose fan-in is the second dimension (transposed convolutions).
    fn mat_t(&mut self, name: &str, shape: &[usize], gain: f32) {
        let fan_in: usize = shape[0] * shape[2..].iter().product::<usize>();
        self.uni(name, shape, gain * (3.0 / fan_in as f32).sqrt(), 0.0);
    }

    /// A norm weight near 1.
    fn ones(&mut self, name: &str, n: usize) {
        self.uni(name, &[n], 0.1, 1.0);
    }

    /// A small bias.
    fn bias(&mut self, name: &str, n: usize) {
        self.uni(name, &[n], 0.02, 0.0);
    }

    /// `{p}.weight` `[o, i]` and, when asked, `{p}.bias`.
    fn lin(&mut self, p: &str, o: usize, i: usize, gain: f32, bias: bool) {
        self.mat(&format!("{p}.weight"), &[o, i], gain);
        if bias {
            self.bias(&format!("{p}.bias"), o);
        }
    }

    /// `{p}.weight` and `{p}.bias` of a LayerNorm.
    fn layer_norm(&mut self, p: &str, n: usize) {
        self.ones(&format!("{p}.weight"), n);
        self.bias(&format!("{p}.bias"), n);
    }

    /// A matrix in bfloat16 (the text encoder's embedding table, the bulk of the files).
    fn mat_bf16(&mut self, name: &str, shape: &[usize], scale: f32) {
        let v = self.noise(shape.iter().product(), scale, 0.0);
        let bytes = v.iter().flat_map(|&x| half::bf16::from_f32(x).to_bits().to_le_bytes()).collect();
        self.list.push((name.to_string(), "BF16", shape.to_vec(), bytes));
    }

    fn write(mut self, path: &Path) -> Result<()> {
        // Float32 tensors first: every one then starts 4-byte aligned.
        self.list.sort_by_key(|t| t.1 != "F32");
        let mut head = serde_json::Map::new();
        if !self.meta.is_empty() {
            head.insert("__metadata__".into(), Value::Object(std::mem::take(&mut self.meta)));
        }
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in self.list {
            let entry = json!({"dtype": dtype, "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]});
            head.insert(name, entry);
            data.extend(bytes);
        }
        let mut head = Value::Object(head).to_string();
        while !head.len().is_multiple_of(8) {
            head.push(' ');
        }
        let mut file = (head.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(head.as_bytes());
        file.extend(data);
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d).with_context(|| d.display().to_string())?;
        }
        std::fs::write(path, file).with_context(|| path.display().to_string())
    }
}

fn write_json(path: &Path, v: &Value) -> Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).with_context(|| d.display().to_string())?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(v)?).with_context(|| path.display().to_string())
}

// The text encoder (Qwen3-VL).
const TEXT_HIDDEN: usize = 16;
const TEXT_LAYERS: usize = 51;
const TEXT_HEADS: usize = 2;
const TEXT_KV_HEADS: usize = 1;
const TEXT_HEAD_DIM: usize = 8;
const TEXT_FFN: usize = 32;
/// Qwen3-VL's vocabulary with MiniMax's seven added tokens (`super::video::EXTRA_TOKENS`).
const VOCAB: usize = 151676;
const VISION_HIDDEN: usize = 16;
const VISION_DEPTH: usize = 2;
const VISION_PATCH: usize = 16;
const VISION_POS: usize = 64;

// The DiT and the VAEs.
const LATENT: usize = 8;
const AUDIO_LATENT: usize = 8;
const DIT_HIDDEN: usize = 32;
const DIT_HEADS: usize = 2;
const DIT_HEAD_DIM: usize = 16;
const DIT_FFN: usize = 64;
const DIT_LAYERS: usize = 2;
const TIME_DIM: usize = 32;
const FREQ_DIM: usize = 32;
const VAE_HEADS: usize = 2;
const VAE_HEAD_DIM: usize = 16;

/// Qwen's special tokens at their ids (`tokenizer.json` of Qwen3-VL).
const SPECIAL: [(&str, u32, bool); 18] = [
    ("<|endoftext|>", 151643, true),
    ("<|im_start|>", 151644, true),
    ("<|im_end|>", 151645, true),
    ("<|object_ref_start|>", 151646, true),
    ("<|object_ref_end|>", 151647, true),
    ("<|box_start|>", 151648, true),
    ("<|box_end|>", 151649, true),
    ("<|quad_start|>", 151650, true),
    ("<|quad_end|>", 151651, true),
    ("<|vision_start|>", 151652, true),
    ("<|vision_end|>", 151653, true),
    ("<|vision_pad|>", 151654, true),
    ("<|image_pad|>", 151655, true),
    ("<|video_pad|>", 151656, true),
    ("<tool_call>", 151657, false),
    ("</tool_call>", 151658, false),
    ("<think>", 151667, false),
    ("</think>", 151668, false),
];

/// A byte-level BPE `tokenizer.json`: every byte one token (GPT-2's byte-to-character map, ids
/// 0 to 255), no merges, and Qwen's special tokens at their ids.
fn tokenizer_json() -> Value {
    let printable = |b: u32| (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
    let mut vocab = serde_json::Map::new();
    let mut extra = 0;
    for b in 0..256u32 {
        let c = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            extra += 1;
            char::from_u32(255 + extra).unwrap()
        };
        vocab.insert(c.to_string(), json!(b));
    }
    // In the vocabulary too, as Qwen's file has them: an added token absent from it would be
    // numbered after the vocabulary (as `tokenizers` does), not at its id.
    for &(content, id, _) in &SPECIAL {
        vocab.insert(content.to_string(), json!(id));
    }
    let added: Vec<Value> = SPECIAL
        .iter()
        .map(|&(content, id, special)| {
            json!({"id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false,
                   "normalized": false, "special": special})
        })
        .collect();
    let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": byte_level, "post_processor": byte_level, "decoder": byte_level,
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "",
            "fuse_unk": false, "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": [],
        },
    })
}

/// A Qwen3-VL text encoder (`text_encoder/`: config, weights, tokenizer, processor configs).
fn write_text_encoder(dir: &Path, seed: u64) -> Result<()> {
    let (h, hd) = (TEXT_HIDDEN, TEXT_HEAD_DIM);
    let config = json!({
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "tie_word_embeddings": true,
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": h,
            "intermediate_size": TEXT_FFN,
            "num_hidden_layers": TEXT_LAYERS,
            "num_attention_heads": TEXT_HEADS,
            "num_key_value_heads": TEXT_KV_HEADS,
            "head_dim": hd,
            "rms_norm_eps": 1e-6,
            "vocab_size": VOCAB,
            "rope_theta": 5_000_000.0,
            "rope_scaling": {"mrope_interleaved": true, "mrope_section": [2, 1, 1], "rope_type": "default"},
            "tie_word_embeddings": true,
            "hidden_act": "silu",
        },
        "vision_config": {
            "model_type": "qwen3_vl",
            "depth": VISION_DEPTH,
            "hidden_size": VISION_HIDDEN,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 2 * VISION_HIDDEN,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": VISION_PATCH,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": h,
            "num_position_embeddings": VISION_POS,
            "deepstack_visual_indexes": [0],
        },
    });
    write_json(&dir.join("config.json"), &config)?;
    // A small pixel budget: a reference image or keyframe is 64 to 256 vision patches.
    let processor = |longest: usize| {
        json!({
            "image_processor_type": "Qwen2VLImageProcessorFast",
            "size": {"shortest_edge": 16384, "longest_edge": longest},
            "patch_size": VISION_PATCH, "temporal_patch_size": 2, "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        })
    };
    write_json(&dir.join("preprocessor_config.json"), &processor(65536))?;
    write_json(&dir.join("video_preprocessor_config.json"), &processor(4 * 65536))?;
    write_json(&dir.join("tokenizer.json"), &tokenizer_json())?;

    let mut t = Tensors::new(seed);
    let p = "model.language_model.";
    t.mat_bf16(&format!("{p}embed_tokens.weight"), &[VOCAB, h], 1.0);
    t.ones(&format!("{p}norm.weight"), h);
    for l in 0..TEXT_LAYERS {
        let lp = format!("{p}layers.{l}.");
        t.ones(&format!("{lp}input_layernorm.weight"), h);
        t.ones(&format!("{lp}post_attention_layernorm.weight"), h);
        t.lin(&format!("{lp}self_attn.q_proj"), TEXT_HEADS * hd, h, 1.0, false);
        t.lin(&format!("{lp}self_attn.k_proj"), TEXT_KV_HEADS * hd, h, 1.0, false);
        t.lin(&format!("{lp}self_attn.v_proj"), TEXT_KV_HEADS * hd, h, 1.0, false);
        t.lin(&format!("{lp}self_attn.o_proj"), h, TEXT_HEADS * hd, 0.5, false);
        t.ones(&format!("{lp}self_attn.q_norm.weight"), hd);
        t.ones(&format!("{lp}self_attn.k_norm.weight"), hd);
        t.lin(&format!("{lp}mlp.gate_proj"), TEXT_FFN, h, 1.0, false);
        t.lin(&format!("{lp}mlp.up_proj"), TEXT_FFN, h, 1.0, false);
        t.lin(&format!("{lp}mlp.down_proj"), h, TEXT_FFN, 0.5, false);
    }
    // The vision tower (`model.visual.*`).
    let (vh, vp) = (VISION_HIDDEN, "model.visual.");
    let mh = 4 * vh;
    t.mat(
        &format!("{vp}patch_embed.proj.weight"),
        &[vh, 3, 2, VISION_PATCH, VISION_PATCH],
        1.0,
    );
    t.bias(&format!("{vp}patch_embed.proj.bias"), vh);
    t.uni(&format!("{vp}pos_embed.weight"), &[VISION_POS, vh], 0.5, 0.0);
    for l in 0..VISION_DEPTH {
        let b = format!("{vp}blocks.{l}.");
        t.layer_norm(&format!("{b}norm1"), vh);
        t.layer_norm(&format!("{b}norm2"), vh);
        t.lin(&format!("{b}attn.qkv"), 3 * vh, vh, 1.0, true);
        t.lin(&format!("{b}attn.proj"), vh, vh, 0.5, true);
        t.lin(&format!("{b}mlp.linear_fc1"), 2 * vh, vh, 1.0, true);
        t.lin(&format!("{b}mlp.linear_fc2"), vh, 2 * vh, 0.5, true);
    }
    for m in ["merger.".to_string(), format!("deepstack_merger_list.{}.", 0)] {
        t.layer_norm(&format!("{vp}{m}norm"), if m == "merger." { vh } else { mh });
        t.lin(&format!("{vp}{m}linear_fc1"), mh, mh, 1.0, true);
        t.lin(&format!("{vp}{m}linear_fc2"), h, mh, 1.0, true);
    }
    t.write(&dir.join("model.safetensors"))
}

/// The DiT in diffusers' names (`transformer/`), with the audio stream when `audio`, and its
/// per-step output heads in `heads` (`pdd_fused_heads.safetensors`) when `pdd_steps > 0`.
fn write_dit(dir: &Path, audio: bool, pdd_steps: usize, seed: u64) -> Result<()> {
    let (h, inner, ffn, td) = (DIT_HIDDEN, DIT_HEADS * DIT_HEAD_DIM, DIT_FFN, TIME_DIM);
    let pd = LATENT * 4;
    let mut config = json!({
        "_class_name": if audio { "MiniMaxH3Transformer3DModel" } else { "MiniMaxH3ImageTransformer2DModel" },
        "hidden_size": h,
        "num_layers": DIT_LAYERS,
        "num_refiner_layers": 1,
        "num_attention_heads": DIT_HEADS,
        "attention_head_dim": DIT_HEAD_DIM,
        "ffn_dim": ffn,
        "in_channels": LATENT,
        "patch_size": [1, 2, 2],
        "text_dim": TEXT_HIDDEN,
        "freq_dim": FREQ_DIM,
        "time_embed_hidden_dim": td,
        "time_embed_dim": td,
        "rope_freq_dim": 2,
        "rope_theta": 10000.0,
        "norm_eps": 1e-5,
        "qk_norm_eps": 1e-5,
        "final_norm_eps": 1e-5,
    });
    if audio {
        config["audio_in_channels"] = json!(AUDIO_LATENT);
    }
    write_json(&dir.join("transformer/config.json"), &config)?;
    let mut t = Tensors::new(seed);
    let block = |t: &mut Tensors, p: &str, adaln: bool| {
        t.ones(&format!("{p}norm1.weight"), h);
        t.ones(&format!("{p}norm2.weight"), h);
        for q in ["to_q", "to_k", "to_v"] {
            t.lin(&format!("{p}attn.{q}"), inner, h, 1.0, false);
        }
        t.ones(&format!("{p}attn.norm_q.weight"), DIT_HEAD_DIM);
        t.ones(&format!("{p}attn.norm_k.weight"), DIT_HEAD_DIM);
        t.lin(&format!("{p}attn.to_out.0"), h, inner, 0.5, false);
        t.lin(&format!("{p}ff.net.0.proj"), 2 * ffn, h, 1.0, false);
        t.lin(&format!("{p}ff.net.2"), h, ffn, 0.5, false);
        if adaln {
            t.lin(&format!("{p}adaln_proj.linear"), 6 * h * super::dit::MODALITIES, td, 0.3, true);
        }
    };
    block(&mut t, "token_refiner.refiner_blocks.0.", false);
    t.ones("token_refiner.final_norm.weight", h);
    for l in 0..DIT_LAYERS {
        block(&mut t, &format!("transformer_blocks.{l}."), true);
    }
    t.lin("proj_in", h, pd, 1.0, true);
    t.lin("context_embedder", h, TEXT_HIDDEN, 1.0, true);
    t.lin("time_embedder.linear_1", td, FREQ_DIM, 1.0, true);
    t.lin("time_embedder.linear_2", td, td, 1.0, true);
    t.ones("norm_out.norm.weight", h);
    t.lin("norm_out.linear", 2 * h, td, 0.3, true);
    t.lin("proj_out", pd, h, 1.0, true);
    if audio {
        t.lin("audio_proj_in", h, AUDIO_LATENT, 1.0, true);
        t.lin("audio_proj_out", AUDIO_LATENT, h, 1.0, true);
    }
    t.write(&dir.join("transformer/diffusion_pytorch_model.safetensors"))?;
    if pdd_steps > 0 {
        let mut heads = Tensors::new(seed ^ 0x5eed);
        heads.mat("video_out.weight", &[pdd_steps, pd, h], 1.0);
        heads.uni("video_out.bias", &[pdd_steps, pd], 0.02, 0.0);
        heads.write(&dir.join("pdd_fused_heads.safetensors"))?;
    }
    Ok(())
}

/// The ViT decoder of either VAE (`decoder.*`, `post_quant_conv`), diffusers' names: one block,
/// a 16x16 patch of `frames` frames per latent voxel.
fn vae_decoder(t: &mut Tensors, frames: usize) {
    let d = VAE_HEADS * VAE_HEAD_DIM;
    t.mat("post_quant_conv.weight", &[LATENT, LATENT, 1, 1], 1.0);
    t.bias("post_quant_conv.bias", LATENT);
    t.lin("decoder.proj_in", d, LATENT, 1.0, true);
    t.uni("decoder.register_tokens", &[1, 4, d], 0.5, 0.0);
    let p = "decoder.transformer_blocks.0.";
    t.ones(&format!("{p}norm1.weight"), d);
    t.ones(&format!("{p}norm2.weight"), d);
    for q in ["to_q", "to_k", "to_v"] {
        t.lin(&format!("{p}attn.{q}"), d, d, 1.0, true);
    }
    t.lin(&format!("{p}attn.to_out.0"), d, d, 1.0, true);
    t.uni(&format!("{p}scale1"), &[d], 0.1, 0.5);
    t.lin(&format!("{p}ff.net.0.proj"), 8 * d, d, 1.0, true);
    t.lin(&format!("{p}ff.net.2"), d, 4 * d, 1.0, true);
    t.uni(&format!("{p}scale2"), &[d], 0.1, 0.5);
    t.layer_norm("decoder.norm_out", d);
    // Outputs in the ImageNet-normalized range.
    t.lin("decoder.proj_out", 3 * frames * 256, d, 0.6, true);
}

fn latent_stats(t: &mut Tensors) -> (Vec<f32>, Vec<f32>) {
    (t.noise(LATENT, 0.1, 0.0), t.noise(LATENT, 0.2, 1.0))
}

/// The image VAE (`vae/`, `AutoencoderKLMiniMaxH3Image`'s keys): the decoder alone.
fn write_image_vae(dir: &Path, seed: u64) -> Result<()> {
    let mut t = Tensors::new(seed);
    let (mean, std) = latent_stats(&mut t);
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "AutoencoderKLMiniMaxH3Image",
            "latent_channels": LATENT,
            "latents_mean": mean,
            "latents_std": std,
            "decoder_num_layers": 1,
            "decoder_num_attention_heads": VAE_HEADS,
            "decoder_attention_head_dim": VAE_HEAD_DIM,
            "decoder_rope_theta": 100.0,
            "decoder_rope_dim_ratio": 0.5,
            "vae_tile_size": 256,
            "vae_tile_overlap_min": 64,
        }),
    )?;
    vae_decoder(&mut t, 1);
    t.write(&dir.join("diffusion_pytorch_model.safetensors"))
}

/// The video VAE (`vae/`, `AutoencoderKLMiniMaxH3`'s keys): the causal 3D CNN encoder (16x in
/// space, 4x in time: `17 n + 5` frames to `5 n + 2` latent frames) and the ViT decoder.
fn write_video_vae(dir: &Path, seed: u64) -> Result<()> {
    let mut t = Tensors::new(seed);
    let (mean, std) = latent_stats(&mut t);
    let channels = [4usize, 8, 8, 8];
    let (sd, tdn) = ([2usize, 2, 2, 2], [1usize, 2, 2, 1]);
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "AutoencoderKLMiniMaxH3",
            "latent_channels": LATENT,
            "latents_mean": mean,
            "latents_std": std,
            "block_out_channels": channels,
            "layers_per_block": 1,
            "norm_num_groups": 4,
            "spatial_downsample_factors": sd,
            "temporal_downsample_factors": tdn,
            "spatial_padding_mode": "reflect",
            "decoder_num_layers": 1,
            "decoder_num_attention_heads": VAE_HEADS,
            "decoder_attention_head_dim": VAE_HEAD_DIM,
            "decoder_rope_theta": 100.0,
            "decoder_rope_dim_ratio": 0.75,
            "clip_length": 17,
            "token_drop": 3,
            "vae_tile_size": 256,
            "vae_tile_overlap_min": 64,
        }),
    )?;
    let conv3 = |t: &mut Tensors, p: &str, co: usize, ci: usize, k: usize, gain: f32| {
        t.mat(&format!("{p}.weight"), &[co, ci, k, k, k], gain);
        t.bias(&format!("{p}.bias"), co);
    };
    conv3(&mut t, "encoder.conv_in", channels[0], 3, 3, 1.0);
    for (i, &ch) in channels.iter().enumerate() {
        let cin = if i == 0 { channels[0] } else { channels[i - 1] };
        let p = format!("encoder.down_blocks.{i}.resnets.0.");
        t.layer_norm(&format!("{p}norm1"), cin);
        conv3(&mut t, &format!("{p}conv1"), ch, cin, 3, 1.0);
        t.layer_norm(&format!("{p}norm2"), ch);
        conv3(&mut t, &format!("{p}conv2"), ch, ch, 3, 0.5);
        if cin != ch {
            conv3(&mut t, &format!("{p}conv_shortcut"), ch, cin, 1, 1.0);
        }
        conv3(&mut t, &format!("encoder.down_blocks.{i}.downsamplers.0.conv"), ch, ch, 3, 1.0);
    }
    let last = channels[channels.len() - 1];
    t.layer_norm("encoder.norm_out", last);
    conv3(&mut t, "encoder.conv_out", 2 * LATENT, last, 3, 1.0);
    // Moments: means near the latents' scale, log-variances around -4.
    let mut q = t.noise(4 * LATENT * LATENT, (3.0 / (2 * LATENT) as f32).sqrt(), 0.0);
    q[2 * LATENT * LATENT..].iter_mut().for_each(|v| *v *= 0.2);
    t.put("quant_conv.weight", &[2 * LATENT, 2 * LATENT, 1, 1, 1], &q);
    let qb: Vec<f32> = (0..2 * LATENT).map(|i| if i < LATENT { 0.0 } else { -4.0 }).collect();
    t.put("quant_conv.bias", &[2 * LATENT], &qb);
    vae_decoder(&mut t, 4);
    t.write(&dir.join("diffusion_pytorch_model.safetensors"))
}

/// A symmetric 12-tap low-pass filter of sum 1 (a Hann-windowed sinc at a quarter of the rate),
/// standing in for the anti-aliasing filters' Kaiser-windowed sinc.
fn lowpass() -> Vec<f32> {
    let k = 12;
    let taps: Vec<f64> = (0..k)
        .map(|i| {
            let x = i as f64 - (k as f64 - 1.0) / 2.0;
            let s = if x == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * 0.5 * x).sin() / (std::f64::consts::PI * 0.5 * x)
            };
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (i as f64 + 0.5) / k as f64).cos();
            s * w
        })
        .collect();
    let sum: f64 = taps.iter().sum();
    taps.iter().map(|v| (v / sum) as f32).collect()
}

/// The audio VAE (`audio_vae/`, diffusers' `AutoencoderKLMiniMaxH3Audio` keys): 32 kHz, 800
/// samples per latent, a BigVGAN decoder of 8 channels (one AMP block per stage) and a DAC
/// encoder of 4 channels.
fn write_audio_vae(dir: &Path, seed: u64) -> Result<()> {
    let rates = [5usize, 5, 2, 2, 2, 2, 2];
    let dil = [1usize, 3];
    let mut t = Tensors::new(seed);
    let (mean, std) = (t.noise(AUDIO_LATENT, 0.1, 0.0), t.noise(AUDIO_LATENT, 0.2, 1.0));
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "AutoencoderKLMiniMaxH3Audio",
            "latent_channels": AUDIO_LATENT,
            "sampling_rate": 32000,
            "decoder_rates": rates,
            "resblock_dilation_sizes": [dil],
            "num_attention_heads": 2,
            "latents_mean": mean,
            "latents_std": std,
        }),
    )?;
    let conv = |t: &mut Tensors, p: &str, co: usize, ci: usize, k: usize, gain: f32| {
        t.mat(&format!("{p}.weight"), &[co, ci, k], gain);
        t.bias(&format!("{p}.bias"), co);
    };
    let filt = lowpass();
    let act = |t: &mut Tensors, p: &str, c: usize| {
        t.uni(&format!("{p}.act.alpha"), &[c], 0.1, 0.0);
        t.uni(&format!("{p}.act.beta"), &[c], 0.1, 0.0);
        t.put(&format!("{p}.upsample.filter"), &[1, 1, 12], &filt);
        t.put(&format!("{p}.downsample.lowpass.filter"), &[1, 1, 12], &filt);
    };
    // The decoder.
    let c = 8;
    conv(&mut t, "dec_in_proj", c, AUDIO_LATENT, 1, 1.0);
    conv(&mut t, "decoder.conv_pre", c, c, 7, 1.0);
    for (i, &r) in rates.iter().enumerate() {
        let k = if r == 5 { 9 } else { 4 };
        t.mat_t(&format!("decoder.ups.{i}.0.weight"), &[c, c, k], 1.0);
        t.bias(&format!("decoder.ups.{i}.0.bias"), c);
        let p = format!("decoder.resblocks.{i}");
        for a in 0..2 * dil.len() {
            act(&mut t, &format!("{p}.activations.{a}"), c);
        }
        for (j, _) in dil.iter().enumerate() {
            conv(&mut t, &format!("{p}.convs1.{j}"), c, c, 3, 1.0);
            conv(&mut t, &format!("{p}.convs2.{j}"), c, c, 3, 0.5);
        }
    }
    act(&mut t, "decoder.activation_post", c);
    t.mat("decoder.conv_post.weight", &[1, c, 7], 0.5);
    // The encoder: strides 2 * 4 * 5 * 5 * 4 = 800.
    let (ce, trunk, width) = (4, 16, 8);
    let strides = [2usize, 4, 5, 5, 4];
    conv(&mut t, "encoder.block.0", ce, 1, 7, 1.0);
    let snake = |t: &mut Tensors, n: &str, c: usize| t.uni(n, &[1, c, 1], 0.1, 1.0);
    for (i, &s) in strides.iter().enumerate() {
        let p = format!("encoder.block.{}.block", i + 1);
        for j in 0..3 {
            snake(&mut t, &format!("{p}.{j}.block.0.alpha"), ce);
            conv(&mut t, &format!("{p}.{j}.block.1"), ce, ce, 7, 1.0);
            snake(&mut t, &format!("{p}.{j}.block.2.alpha"), ce);
            conv(&mut t, &format!("{p}.{j}.block.3"), ce, ce, 1, 0.5);
        }
        snake(&mut t, &format!("{p}.3.alpha"), ce);
        conv(&mut t, &format!("{p}.4"), ce, ce, 2 * s, 1.0);
    }
    let n = strides.len() + 1;
    snake(&mut t, &format!("encoder.block.{n}.alpha"), ce);
    conv(&mut t, &format!("encoder.block.{}", n + 1), trunk, ce, 3, 1.0);
    let pb = "pre_block";
    t.layer_norm(&format!("{pb}.norm1"), trunk);
    t.layer_norm(&format!("{pb}.norm3"), trunk);
    t.layer_norm(&format!("{pb}.norm2"), width);
    t.mat(&format!("{pb}.attn.qkv.weight"), &[3 * trunk, trunk], 1.0);
    t.bias(&format!("{pb}.attn.q_bias"), trunk);
    t.bias(&format!("{pb}.attn.v_bias"), trunk);
    t.lin(&format!("{pb}.attn.proj"), width, width, 1.0, true);
    t.lin(&format!("{pb}.proj"), width, trunk, 1.0, true);
    t.layer_norm(&format!("{pb}.mlp.norm"), width);
    t.lin(&format!("{pb}.mlp.w0"), 2 * width, width, 1.0, false);
    t.lin(&format!("{pb}.mlp.w1"), 2 * width, width, 1.0, false);
    t.lin(&format!("{pb}.mlp.w2"), width, 2 * width, 0.5, false);
    conv(&mut t, "mean_proj", AUDIO_LATENT, width, 1, 1.0);
    t.write(&dir.join("diffusion_pytorch_model.safetensors"))
}

/// Writes a tiny random MiniMax-H3 release (video with its soundtrack) in diffusers' layout into
/// `dir` (created; files already there are overwritten): see the module docs. About 6 MB, most
/// of it the text encoder's embedding table.
pub fn write_tiny_video_release(dir: &Path) -> Result<()> {
    write_json(
        &dir.join("model_index.json"),
        &json!({
            "_class_name": "MiniMaxH3ModularPipeline",
            "_diffusers_version": "0.43.0",
            "_oxinfer_tiny": "random weights for tests (oxinfer::minimax_h3::tiny)",
            "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
            "vae": ["diffusers", "AutoencoderKLMiniMaxH3"],
            "audio_vae": ["diffusers", "AutoencoderKLMiniMaxH3Audio"],
            "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
            "tokenizer": ["transformers", "Qwen2TokenizerFast"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "audio_scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        }),
    )?;
    write_json(&dir.join("scheduler/scheduler_config.json"), &json!({"shift": 12.0}))?;
    write_json(&dir.join("audio_scheduler/scheduler_config.json"), &json!({"shift": 3.0}))?;
    write_json(&dir.join("tokenizer/tokenizer.json"), &tokenizer_json())?;
    write_dit(dir, true, 0, 0x4833_7669_6465_6f00)?;
    write_video_vae(&dir.join("vae"), 0x4833_7661_6500)?;
    write_audio_vae(&dir.join("audio_vae"), 0x4833_6175_6469_6f00)?;
    write_text_encoder(&dir.join("text_encoder"), 0x4833_7465_7874)
}

/// Writes a tiny random MiniMax-H3-Image release into `dir` (created; files already there are
/// overwritten), with its own copy of the tiny text encoder in `dir/text_encoder`: see the
/// module docs. About 6 MB.
pub fn write_tiny_image_release(dir: &Path) -> Result<()> {
    write_json(
        &dir.join("model_index.json"),
        &json!({
            "_class_name": "MiniMaxH3ImagePipeline",
            "_diffusers_version": "0.43.0",
            "_oxinfer_tiny": "random weights for tests (oxinfer::minimax_h3::tiny)",
            "_minimax_h3": {"pdd_fused_heads": "pdd_fused_heads.safetensors"},
            "transformer": ["diffusers", "MiniMaxH3ImageTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLMiniMaxH3Image"],
            "tokenizer": ["transformers", "Qwen2TokenizerFast"],
        }),
    )?;
    write_json(&dir.join("pdd_config.json"), &json!({"sigmas": [1.0, 0.75, 0.4, 0.0]}))?;
    write_json(&dir.join("tokenizer/tokenizer.json"), &tokenizer_json())?;
    write_dit(dir, false, 3, 0x4833_696d_6167_6500)?;
    write_image_vae(&dir.join("vae"), 0x4833_6976_6165)?;
    write_text_encoder(&dir.join("text_encoder"), 0x4833_7465_7874)
}

#[cfg(test)]
mod tests {
    use super::super::ref2va::{RefInput, RefOptions, normalize};
    use super::super::video::{Sources, Video, VideoParams, VideoPipeline, is_h3_video};
    use super::super::{Params, Pipeline, is_h3_image};
    use super::*;
    use std::path::PathBuf;
    use std::time::Instant;

    /// A temporary folder removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let d = std::env::temp_dir().join(format!("oxinfer-h3-tiny-{tag}-{}", std::process::id()));
            std::fs::remove_dir_all(&d).ok();
            Self(d)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    fn gradient(w: usize, h: usize, phase: usize) -> crate::sd::Image {
        let rgb = (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                [(x * 255 / w) as u8, (y * 255 / h) as u8, ((x + y + phase) * 7 % 256) as u8]
            })
            .collect();
        crate::sd::Image { width: w, height: h, rgb }
    }

    fn check_video(v: &Video, w: usize, h: usize, frames: usize) {
        assert_eq!((v.width, v.height), (w, h));
        assert_eq!(v.frames.len(), frames);
        assert!(v.frames.iter().all(|f| f.len() == w * h * 3));
        // Not a constant picture.
        let f0 = &v.frames[0];
        assert!(f0.iter().any(|&b| b != f0[0]), "a flat frame");
        let a = v.audio.as_ref().expect("a soundtrack");
        assert_eq!(a.sample_rate, 32000);
        assert_eq!(a.channels.len(), 2);
        // 40 latents per second, 800 samples each.
        assert_eq!(a.len(), super::super::video::audio_latents(frames) * 800);
        assert!(a.channels.iter().flatten().all(|s| s.is_finite() && s.abs() <= 1.0));
        assert!(a.channels[0].iter().any(|&s| s != 0.0), "a silent soundtrack");
    }

    fn params(prompt: &str) -> VideoParams {
        VideoParams {
            size: Some((64, 64)),
            frames: 22,
            steps: 3,
            seed: 7,
            ..VideoParams::new(prompt)
        }
    }

    #[test]
    fn tiny_video_release() {
        let tmp = TempDir::new("video");
        let dir = &tmp.0;
        let t0 = Instant::now();
        write_tiny_video_release(dir).unwrap();
        eprintln!("[tiny H3 video release written in {:.2}s]", t0.elapsed().as_secs_f64());
        assert!(is_h3_video(dir));
        let src = Sources {
            dir: Some(dir.clone()),
            text_encoder: dir.join("text_encoder"),
            ..Default::default()
        };
        let pipe = VideoPipeline::load_from(&src).unwrap();
        assert!(pipe.audio_vae.is_some());
        let mut quiet = |_: usize, _: usize| {};

        // Text to video (a dialogue line: MiniMax's added tokens are in the vocabulary).
        let t = Instant::now();
        let v = pipe
            .generate(&params("A fox in the snow says: <d>[English] Hi.</d>"), None, &mut quiet)
            .unwrap();
        eprintln!("[t2v {:.2}s]", t.elapsed().as_secs_f64());
        check_video(&v, 64, 64, 22);
        // Deterministic.
        let again = pipe
            .generate(&params("A fox in the snow says: <d>[English] Hi.</d>"), None, &mut quiet)
            .unwrap();
        assert_eq!(v.frames, again.frames);

        // First and last frame to video.
        let t = Instant::now();
        let p = VideoParams {
            first: Some(gradient(64, 64, 0)),
            last: Some(gradient(96, 64, 5)),
            ..params("The fox turns around.")
        };
        let fl = pipe.generate(&p, None, &mut quiet).unwrap();
        eprintln!("[first + last frame {:.2}s]", t.elapsed().as_secs_f64());
        check_video(&fl, 64, 64, 22);
        assert_ne!(fl.frames, v.frames);

        // References: an image and an audio clip (mono, 44.1 kHz, longer than the video).
        let t = Instant::now();
        let tone: Vec<f32> = (0..44100).map(|i| 0.3 * (i as f32 * 0.05).sin()).collect();
        let refs = vec![RefInput::Image(gradient(80, 48, 3)), RefInput::Audio(vec![tone], 44100)];
        let rate = pipe.audio_vae.as_ref().unwrap().cfg.sampling_rate;
        let o = RefOptions {
            image_short_edge: 64,
            ..Default::default()
        };
        let refs = normalize(refs, 22, rate, &o).unwrap();
        let r = pipe
            .generate_ref(&params("<Picture 1> sings <Audio 1>."), &refs, None, &mut quiet)
            .unwrap();
        eprintln!("[ref2va image + audio {:.2}s]", t.elapsed().as_secs_f64());
        check_video(&r, 64, 64, 22);
        drop(pipe);
    }

    #[test]
    fn tiny_image_release() {
        let tmp = TempDir::new("image");
        let dir = &tmp.0;
        write_tiny_image_release(dir).unwrap();
        assert!(is_h3_image(dir));
        let pipe = Pipeline::load(dir, &dir.join("text_encoder")).unwrap();
        assert_eq!(pipe.dit.n_heads(), 3);
        let t = Instant::now();
        let p = Params {
            width: 64,
            height: 64,
            min_area: 64 * 64,
            ..Params::new("A red fox in the snow.")
        };
        let img = pipe.generate(&p, &mut |_, _| {}).unwrap();
        eprintln!("[t2i {:.2}s]", t.elapsed().as_secs_f64());
        assert_eq!((img.width, img.height, img.rgb.len()), (64, 64, 64 * 64 * 3));
        assert!(img.rgb.iter().any(|&b| b != img.rgb[0]), "a flat image");
        // A non-square request on a larger canvas, resized back.
        let p = Params {
            width: 96,
            height: 64,
            min_area: 128 * 128,
            ..p
        };
        let img = pipe.generate(&p, &mut |_, _| {}).unwrap();
        assert_eq!((img.width, img.height, img.rgb.len()), (96, 64, 96 * 64 * 3));
    }

    /// The tiny release's transformer in ComfyUI's single-file layout: the release's original
    /// names, `qkv_proj` as `[q; k; v]`, `fc1` as `[gate; value]`, and the PDD heads stacked in
    /// `final_layer.video_out` (block 0 the first head, block s the offset of head s from it).
    fn write_comfy_image_dit(dir: &Path, out: &Path) {
        let f = crate::safetensors::ModelFiles::open(&dir.join("transformer"), false).unwrap();
        let heads = crate::safetensors::ModelFiles::open_file(&dir.join("pdd_fused_heads.safetensors"), false).unwrap();
        let mut t = Tensors::new(0);
        let put = |t: &mut Tensors, name: String, v: Vec<f32>, shape: Vec<usize>| t.put(&name, &shape, &v);
        let shape = |n: &str| f.info(n).unwrap().shape.clone();
        let blk = |t: &mut Tensors, from: &str, to: &str, adaln: bool| {
            let g = |n: &str| f.f32(&format!("{from}{n}")).unwrap();
            for (a, b) in [
                ("norm1", "norm1"),
                ("norm2", "norm2"),
                ("attn.norm_q", "attn.q_norm"),
                ("attn.norm_k", "attn.k_norm"),
            ] {
                put(
                    t,
                    format!("{to}{b}.weight"),
                    g(&format!("{a}.weight")),
                    shape(&format!("{from}{a}.weight")),
                );
            }
            let mut qkv = g("attn.to_q.weight");
            qkv.extend(g("attn.to_k.weight"));
            qkv.extend(g("attn.to_v.weight"));
            let [o, i] = shape(&format!("{from}attn.to_q.weight"))[..] else {
                unreachable!()
            };
            put(t, format!("{to}attn.qkv_proj.weight"), qkv, vec![3 * o, i]);
            put(
                t,
                format!("{to}attn.out_proj.weight"),
                g("attn.to_out.0.weight"),
                shape(&format!("{from}attn.to_out.0.weight")),
            );
            let fc1 = g("ff.net.0.proj.weight");
            let s1 = shape(&format!("{from}ff.net.0.proj.weight"));
            let half = fc1.len() / 2;
            put(t, format!("{to}mlp.fc1.weight"), [&fc1[half..], &fc1[..half]].concat(), s1);
            put(
                t,
                format!("{to}mlp.fc2.weight"),
                g("ff.net.2.weight"),
                shape(&format!("{from}ff.net.2.weight")),
            );
            if adaln {
                for k in ["weight", "bias"] {
                    put(
                        t,
                        format!("{to}adaln_proj.linear.{k}"),
                        g(&format!("adaln_proj.linear.{k}")),
                        shape(&format!("{from}adaln_proj.linear.{k}")),
                    );
                }
            }
        };
        blk(&mut t, "token_refiner.refiner_blocks.0.", "token_refiner.blocks.0.", false);
        for l in 0..DIT_LAYERS {
            blk(&mut t, &format!("transformer_blocks.{l}."), &format!("blocks.{l}."), true);
        }
        for (a, b) in [
            ("token_refiner.final_norm", "token_refiner.final_norm"),
            ("proj_in", "video_patch_proj"),
            ("context_embedder", "condition_proj"),
            ("time_embedder.linear_1", "time_embedder.proj_in"),
            ("time_embedder.linear_2", "time_embedder.proj_out"),
            ("norm_out.norm", "final_layer.norm"),
            ("norm_out.linear", "final_layer.adaln_proj.linear"),
        ] {
            for k in ["weight", "bias"] {
                let n = format!("{a}.{k}");
                if f.has(&n) {
                    put(&mut t, format!("{b}.{k}"), f.f32(&n).unwrap(), shape(&n));
                }
            }
        }
        let (w, b) = (heads.f32("video_out.weight").unwrap(), heads.f32("video_out.bias").unwrap());
        let s = heads.info("video_out.weight").unwrap().shape.clone();
        let (steps, pd, h) = (s[0], s[1], s[2]);
        let (mut ws, mut bs) = (w.clone(), b.clone());
        for st in 1..steps {
            ws[st * pd * h..(st + 1) * pd * h]
                .iter_mut()
                .zip(&w[..pd * h])
                .for_each(|(v, a)| *v -= a);
            bs[st * pd..(st + 1) * pd].iter_mut().zip(&b[..pd]).for_each(|(v, a)| *v -= a);
        }
        put(&mut t, "final_layer.video_out.weight".into(), ws, vec![steps * pd, h]);
        put(&mut t, "final_layer.video_out.bias".into(), bs, vec![steps * pd]);
        put(&mut t, "rope.inv_freq".into(), vec![1.0, 0.01], vec![2]);
        // ComfyUI's header states the grid (the release's `pdd_config.json`).
        let cfg = json!({"transformer": {"pdd_steps": steps, "pdd_sigmas": [1.0, 0.75, 0.4, 0.0]}});
        t.meta.insert("config".into(), Value::String(cfg.to_string()));
        t.write(out).unwrap();
    }

    #[test]
    fn tiny_image_comfy_file() {
        let tmp = TempDir::new("image-comfy");
        let dir = &tmp.0;
        write_tiny_image_release(dir).unwrap();
        let file = dir.join("h3_image_comfy.safetensors");
        write_comfy_image_dit(dir, &file);
        assert!(super::super::is_h3_image_file(&file));
        assert!(!is_h3_video(&file));
        let te = dir.join("text_encoder");
        let release = Pipeline::load(dir, &te).unwrap();
        let comfy = Pipeline::load_files(&file, &dir.join("vae"), &te).unwrap();
        assert_eq!(comfy.dit.n_heads(), 3);
        assert_eq!(comfy.sigmas, release.sigmas);
        let p = Params {
            width: 64,
            height: 64,
            min_area: 64 * 64,
            ..Params::new("A red fox in the snow.")
        };
        let text = release.encode_prompt(&p.prompt).unwrap();
        assert_eq!(text, comfy.encode_prompt(&p.prompt).unwrap());
        let noise = crate::sd::ops::Rng::new(3).normals(LATENT * 4 * 4);
        let rows = super::super::patchify(&noise, LATENT, 4, 4, 2);
        let a = release.dit.forward(
            &super::super::dit::Packed {
                text: &text,
                image: &rows,
                positions: &super::super::positions(text.len() / TEXT_HIDDEN, 4, 4, 2),
                timesteps: &[0.25],
                timestep_index: &vec![0; text.len() / TEXT_HIDDEN + 4],
                tags: &[
                    vec![super::super::dit::TAG_TEXT; text.len() / TEXT_HIDDEN],
                    vec![super::super::dit::TAG_IMAGE; 4],
                ]
                .concat(),
            },
            &release.dit.embed_text(&text),
            2,
        );
        let b = comfy.dit.forward(
            &super::super::dit::Packed {
                text: &text,
                image: &rows,
                positions: &super::super::positions(text.len() / TEXT_HIDDEN, 4, 4, 2),
                timesteps: &[0.25],
                timestep_index: &vec![0; text.len() / TEXT_HIDDEN + 4],
                tags: &[
                    vec![super::super::dit::TAG_TEXT; text.len() / TEXT_HIDDEN],
                    vec![super::super::dit::TAG_IMAGE; 4],
                ]
                .concat(),
            },
            &comfy.dit.embed_text(&text),
            2,
        );
        let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
        let diff = a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(diff <= 1e-5 * scale, "the third head's velocity: {diff} of {scale}");
    }

    /// An image reference at the release's default 2048-pixel short edge (as a server request
    /// without options normalizes it): 16k reference rows, slower than the other paths.
    #[test]
    #[ignore]
    fn tiny_ref2va_default_edge() {
        let tmp = TempDir::new("ref-edge");
        let dir = &tmp.0;
        write_tiny_video_release(dir).unwrap();
        let pipe = VideoPipeline::load(dir, &dir.join("text_encoder")).unwrap();
        let t = Instant::now();
        let rate = pipe.audio_vae.as_ref().unwrap().cfg.sampling_rate;
        let refs = normalize(vec![RefInput::Image(gradient(80, 48, 3))], 22, rate, &RefOptions::default()).unwrap();
        let r = pipe
            .generate_ref(&params("<Picture 1> waves."), &refs, None, &mut |_, _| {})
            .unwrap();
        eprintln!("[ref2va at a 2048 short edge {:.2}s]", t.elapsed().as_secs_f64());
        check_video(&r, 64, 64, 22);
    }

    #[test]
    fn tokenizer_ids() {
        let t = crate::tokenizer::Tokenizer::from_bytes(serde_json::to_vec(&tokenizer_json()).unwrap()).unwrap();
        let e = t
            .encode("<|vision_start|><|image_pad|><|image_pad|><|vision_end|>A \u{e9}\n", false)
            .unwrap();
        assert_eq!(e.get_ids(), [151652, 151655, 151655, 151653, 65, 32, 195, 169, 10]);
    }

    /// `OXINFER_TINY_H3_OUT=DIR cargo test --release --lib minimax_h3::tiny -- --ignored`: writes
    /// `DIR/MiniMax-H3-tiny` and `DIR/MiniMax-H3-Image-tiny` and leaves them (for a running
    /// server or the GUI).
    #[test]
    #[ignore]
    fn write_tiny_releases() {
        let Ok(out) = std::env::var("OXINFER_TINY_H3_OUT") else {
            eprintln!("OXINFER_TINY_H3_OUT is not set: nothing written");
            return;
        };
        let out = PathBuf::from(out);
        write_tiny_video_release(&out.join("MiniMax-H3-tiny")).unwrap();
        write_tiny_image_release(&out.join("MiniMax-H3-Image-tiny")).unwrap();
        eprintln!(
            "wrote {} and {}",
            out.join("MiniMax-H3-tiny").display(),
            out.join("MiniMax-H3-Image-tiny").display()
        );
    }
}
