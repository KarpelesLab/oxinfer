//! A tiny random Stable Diffusion 3.5 pipeline folder, for end-to-end tests of code that loads
//! one (the server's image routes) without the gated Stability AI repositories.
//!
//! [`write_tiny_release`] writes a diffusers `StableDiffusion3Pipeline` folder with every
//! component the loader reads: CLIP-L and OpenCLIP-bigG (with their projections), T5, an
//! SD 3.5-style transformer (q/k RMS norms, an MMDiT-X second attention in block 0), the VAE
//! with its encoder, both CLIP BPE tokenizers, T5's Unigram `tokenizer.json` and the flow-matching
//! scheduler. The weights are deterministic uniform noise (f32), scaled so activations stay
//! finite; the images are noise too. About 5 MB.

use anyhow::{Context, Result};
use serde_json::json;
use std::path::Path;

/// Transformer: 2 heads of 32 (hidden 64), 3 blocks, the first with the second attention.
const HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const LAYERS: usize = 3;
const POS_MAX: usize = 32;
/// CLIP-L / CLIP-G widths (also their projection widths) and T5's width (the joint width).
const CLIP_L: usize = 32;
const CLIP_G: usize = 64;
const JOINT: usize = 128;
const LATENT: usize = 16;
const VAE_CH: usize = 32;
const VAE_LEVELS: usize = 4;

/// Deterministic uniform noise in [-1, 1) (an LCG, as the other tiny test checkpoints use).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// The tensors of one safetensors file, written in f32.
struct St {
    rng: Rng,
    tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
}

impl St {
    fn new(seed: u64) -> Self {
        Self {
            rng: Rng(seed),
            tensors: Vec::new(),
        }
    }

    /// `scale` times uniform noise, plus `bias`.
    fn add(&mut self, name: &str, shape: &[usize], scale: f32, bias: f32) {
        let n: usize = shape.iter().product();
        let v = (0..n).map(|_| bias + scale * self.rng.next()).collect();
        self.tensors.push((name.to_string(), shape.to_vec(), v));
    }

    /// A linear layer `[n_out, n_in]` (uniform with unit output variance times `gain`), with a
    /// small bias if `bias`.
    fn linear(&mut self, p: &str, n_out: usize, n_in: usize, gain: f32, bias: bool) {
        self.add(&format!("{p}.weight"), &[n_out, n_in], gain * (3.0 / n_in as f32).sqrt(), 0.0);
        if bias {
            self.add(&format!("{p}.bias"), &[n_out], 0.02, 0.0);
        }
    }

    /// A 3x3 convolution with bias.
    fn conv(&mut self, p: &str, cout: usize, cin: usize, gain: f32) {
        self.add(
            &format!("{p}.weight"),
            &[cout, cin, 3, 3],
            gain * (3.0 / (cin * 9) as f32).sqrt(),
            0.0,
        );
        self.add(&format!("{p}.bias"), &[cout], 0.02, 0.0);
    }

    /// A layer / group norm: weight near 1, bias near 0 (`bias` false: RMS norm weight only).
    fn norm(&mut self, p: &str, d: usize, bias: bool) {
        self.add(&format!("{p}.weight"), &[d], 0.1, 1.0);
        if bias {
            self.add(&format!("{p}.bias"), &[d], 0.02, 0.0);
        }
    }

    fn write(self, path: &Path) -> Result<()> {
        let mut head = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, v) in self.tensors {
            let start = data.len();
            data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            head.insert(name, json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
        }
        let mut head = serde_json::Value::Object(head).to_string();
        while !head.len().is_multiple_of(8) {
            head.push(' ');
        }
        let mut file = (head.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(head.as_bytes());
        file.extend(data);
        std::fs::write(path, file).with_context(|| format!("write {}", path.display()))
    }
}

fn write_json(path: &Path, v: &serde_json::Value) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(v)?).with_context(|| format!("write {}", path.display()))
}

/// GPT-2's byte -> printable-character map (as `crate::sd::clip` reads vocab.json).
fn bytes_to_unicode() -> Vec<char> {
    let mut bytes: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut chars = bytes.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bytes.contains(&b) {
            bytes.push(b);
            chars.push(256 + n);
            n += 1;
        }
    }
    let mut map = vec!['\0'; 256];
    for (b, c) in bytes.iter().zip(chars) {
        map[*b as usize] = char::from_u32(c).unwrap();
    }
    map
}

/// A CLIP BPE tokenizer: every byte alone and word-final, a few merges, the two specials.
fn clip_tokenizer(dir: &Path, pad: &str) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let map = bytes_to_unicode();
    let mut vocab: Vec<String> = map.iter().map(|c| c.to_string()).collect();
    vocab.extend(map.iter().map(|c| format!("{c}</w>")));
    let merges = [
        ("c", "a"),
        ("ca", "t</w>"),
        ("o", "g</w>"),
        ("d", "og</w>"),
        ("a", "n"),
        ("an", "d</w>"),
    ];
    vocab.extend(merges.iter().map(|(a, b)| format!("{a}{b}")));
    vocab.extend(["<|startoftext|>".to_string(), "<|endoftext|>".to_string()]);
    let vocab: serde_json::Map<String, serde_json::Value> = vocab.into_iter().enumerate().map(|(i, t)| (t, json!(i))).collect();
    let n = vocab.len();
    write_json(&dir.join("vocab.json"), &serde_json::Value::Object(vocab))?;
    let mut m = String::from("#version: 0.2\n");
    for (a, b) in merges {
        m.push_str(&format!("{a} {b}\n"));
    }
    std::fs::write(dir.join("merges.txt"), m)?;
    write_json(
        &dir.join("tokenizer_config.json"),
        &json!({
            "tokenizer_class": "CLIPTokenizer",
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
            "unk_token": "<|endoftext|>",
            "pad_token": pad,
            "model_max_length": 77,
            "do_lower_case": true,
        }),
    )?;
    debug_assert_eq!(n, CLIP_VOCAB);
    Ok(())
}

/// 256 bytes, 256 word-final bytes, 6 merges, 2 specials.
const CLIP_VOCAB: usize = 520;

/// A `CLIPTextModelWithProjection` (`hidden` wide, projection `hidden` too).
fn clip_encoder(dir: &Path, hidden: usize, act: &str, seed: u64) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let (layers, heads, inter) = (2, 2, hidden * 2);
    write_json(
        &dir.join("config.json"),
        &json!({
            "architectures": ["CLIPTextModelWithProjection"],
            "model_type": "clip_text_model",
            "hidden_size": hidden,
            "intermediate_size": inter,
            "num_attention_heads": heads,
            "num_hidden_layers": layers,
            "projection_dim": hidden,
            "hidden_act": act,
            "max_position_embeddings": 77,
            "vocab_size": CLIP_VOCAB,
            "layer_norm_eps": 1e-5,
            "bos_token_id": CLIP_VOCAB - 2,
            "eos_token_id": CLIP_VOCAB - 1,
            "torch_dtype": "float32",
        }),
    )?;
    let mut st = St::new(seed);
    let p = "text_model";
    st.add(&format!("{p}.embeddings.token_embedding.weight"), &[CLIP_VOCAB, hidden], 1.0, 0.0);
    st.add(&format!("{p}.embeddings.position_embedding.weight"), &[77, hidden], 0.3, 0.0);
    for l in 0..layers {
        let lp = format!("{p}.encoder.layers.{l}");
        st.norm(&format!("{lp}.layer_norm1"), hidden, true);
        for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            st.linear(&format!("{lp}.self_attn.{n}"), hidden, hidden, 1.0, true);
        }
        st.norm(&format!("{lp}.layer_norm2"), hidden, true);
        st.linear(&format!("{lp}.mlp.fc1"), inter, hidden, 1.0, true);
        st.linear(&format!("{lp}.mlp.fc2"), hidden, inter, 0.5, true);
    }
    st.norm(&format!("{p}.final_layer_norm"), hidden, true);
    st.linear("text_projection", hidden, hidden, 1.0, false);
    st.write(&dir.join("model.safetensors"))
}

/// T5's pieces: specials, the word boundary, letters, digits and a few words.
fn t5_pieces() -> Vec<(String, f64)> {
    let mut v: Vec<(String, f64)> = vec![
        ("<pad>".into(), 0.0),
        ("</s>".into(), 0.0),
        ("<unk>".into(), 0.0),
        ("▁".into(), -2.0),
    ];
    for w in ["▁a", "▁cat", "▁dog", "▁and", "▁the", "▁photo", "▁of"] {
        v.push((w.into(), -3.0));
    }
    for c in ('a'..='z').chain('0'..='9').chain(".,!?'-".chars()) {
        v.push((c.to_string(), -6.0));
    }
    v
}

/// T5 (`T5EncoderModel`, gated-gelu) `JOINT` wide, and its Unigram tokenizer.
fn t5(dir: &Path, tok_dir: &Path, seed: u64) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::create_dir_all(tok_dir)?;
    let pieces = t5_pieces();
    let vocab = pieces.len();
    let (heads, d_kv, d_ff, layers, buckets) = (2, 32, 2 * JOINT, 2, 32);
    write_json(
        &dir.join("config.json"),
        &json!({
            "architectures": ["T5EncoderModel"],
            "model_type": "t5",
            "d_model": JOINT,
            "d_kv": d_kv,
            "d_ff": d_ff,
            "num_heads": heads,
            "num_layers": layers,
            "feed_forward_proj": "gated-gelu",
            "dense_act_fn": "gelu_new",
            "is_gated_act": true,
            "vocab_size": vocab,
            "layer_norm_epsilon": 1e-6,
            "relative_attention_num_buckets": buckets,
            "relative_attention_max_distance": 128,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "torch_dtype": "float32",
        }),
    )?;
    let mut st = St::new(seed);
    st.add("shared.weight", &[vocab, JOINT], 1.0, 0.0);
    for i in 0..layers {
        let p = format!("encoder.block.{i}.layer");
        st.norm(&format!("{p}.0.layer_norm"), JOINT, false);
        // T5 does not scale its attention logits: small queries keep them moderate.
        st.linear(&format!("{p}.0.SelfAttention.q"), heads * d_kv, JOINT, 0.2, false);
        st.linear(&format!("{p}.0.SelfAttention.k"), heads * d_kv, JOINT, 1.0, false);
        st.linear(&format!("{p}.0.SelfAttention.v"), heads * d_kv, JOINT, 1.0, false);
        st.linear(&format!("{p}.0.SelfAttention.o"), JOINT, heads * d_kv, 0.5, false);
        if i == 0 {
            st.add(
                &format!("{p}.0.SelfAttention.relative_attention_bias.weight"),
                &[buckets, heads],
                0.5,
                0.0,
            );
        }
        st.norm(&format!("{p}.1.layer_norm"), JOINT, false);
        st.linear(&format!("{p}.1.DenseReluDense.wi_0"), d_ff, JOINT, 1.0, false);
        st.linear(&format!("{p}.1.DenseReluDense.wi_1"), d_ff, JOINT, 1.0, false);
        st.linear(&format!("{p}.1.DenseReluDense.wo"), JOINT, d_ff, 0.5, false);
    }
    st.norm("encoder.final_layer_norm", JOINT, false);
    st.write(&dir.join("model.safetensors"))?;
    let meta = json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true});
    write_json(
        &tok_dir.join("tokenizer.json"),
        &json!({
            "version": "1.0",
            "added_tokens": [
                {"id": 0, "content": "<pad>", "special": true, "normalized": false, "single_word": false, "lstrip": false, "rstrip": false},
                {"id": 1, "content": "</s>", "special": true, "normalized": false, "single_word": false, "lstrip": false, "rstrip": false},
                {"id": 2, "content": "<unk>", "special": true, "normalized": false, "single_word": false, "lstrip": false, "rstrip": false},
            ],
            "normalizer": null,
            "pre_tokenizer": meta,
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
                "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}},
                         {"Sequence": {"id": "B", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
                "special_tokens": {"</s>": {"id": "</s>", "ids": [1], "tokens": ["</s>"]}},
            },
            "decoder": meta,
            "model": {"type": "Unigram", "unk_id": 2, "byte_fallback": false, "vocab": pieces},
        }),
    )?;
    write_json(
        &tok_dir.join("tokenizer_config.json"),
        &json!({"tokenizer_class": "T5TokenizerFast", "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>", "model_max_length": 512}),
    )
}

/// The SD 3.5 transformer (diffusers names).
fn transformer(dir: &Path, seed: u64) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let d = HEADS * HEAD_DIM;
    let pooled = CLIP_L + CLIP_G;
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "SD3Transformer2DModel",
            "sample_size": 8,
            "patch_size": 2,
            "in_channels": LATENT,
            "out_channels": LATENT,
            "num_layers": LAYERS,
            "attention_head_dim": HEAD_DIM,
            "num_attention_heads": HEADS,
            "joint_attention_dim": JOINT,
            "caption_projection_dim": d,
            "pooled_projection_dim": pooled,
            "pos_embed_max_size": POS_MAX,
            "dual_attention_layers": [0],
            "qk_norm": "rms_norm",
        }),
    )?;
    let mut st = St::new(seed);
    st.add("pos_embed.proj.weight", &[d, LATENT, 2, 2], (3.0 / (LATENT * 4) as f32).sqrt(), 0.0);
    st.add("pos_embed.proj.bias", &[d], 0.02, 0.0);
    st.add("pos_embed.pos_embed", &[1, POS_MAX * POS_MAX, d], 0.5, 0.0);
    st.linear("context_embedder", d, JOINT, 1.0, true);
    st.linear("time_text_embed.timestep_embedder.linear_1", d, 256, 1.0, true);
    st.linear("time_text_embed.timestep_embedder.linear_2", d, d, 1.0, true);
    st.linear("time_text_embed.text_embedder.linear_1", d, pooled, 1.0, true);
    st.linear("time_text_embed.text_embedder.linear_2", d, d, 1.0, true);
    // Modulations stay small: scales near 1 + 0, gates near 0.
    let m = 0.3;
    for i in 0..LAYERS {
        let p = format!("transformer_blocks.{i}");
        let (dual, last) = (i == 0, i + 1 == LAYERS);
        st.linear(&format!("{p}.norm1.linear"), if dual { 9 } else { 6 } * d, d, m, true);
        st.linear(&format!("{p}.norm1_context.linear"), if last { 2 } else { 6 } * d, d, m, true);
        let attn = |st: &mut St, names: [&str; 5]| {
            for n in &names[..3] {
                st.linear(&format!("{p}.{n}"), d, d, 1.0, true);
            }
            for n in &names[3..] {
                st.norm(&format!("{p}.{n}"), HEAD_DIM, false);
            }
        };
        attn(&mut st, ["attn.to_q", "attn.to_k", "attn.to_v", "attn.norm_q", "attn.norm_k"]);
        attn(
            &mut st,
            [
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
                "attn.norm_added_q",
                "attn.norm_added_k",
            ],
        );
        st.linear(&format!("{p}.attn.to_out.0"), d, d, 1.0, true);
        if dual {
            attn(&mut st, ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.norm_q", "attn2.norm_k"]);
            st.linear(&format!("{p}.attn2.to_out.0"), d, d, 1.0, true);
        }
        st.linear(&format!("{p}.ff.net.0.proj"), 4 * d, d, 1.0, true);
        st.linear(&format!("{p}.ff.net.2"), d, 4 * d, 1.0, true);
        if !last {
            st.linear(&format!("{p}.attn.to_add_out"), d, d, 1.0, true);
            st.linear(&format!("{p}.ff_context.net.0.proj"), 4 * d, d, 1.0, true);
            st.linear(&format!("{p}.ff_context.net.2"), d, 4 * d, 1.0, true);
        }
    }
    st.linear("norm_out.linear", 2 * d, d, m, true);
    st.linear("proj_out", 4 * LATENT, d, 1.0, true);
    st.write(&dir.join("diffusion_pytorch_model.safetensors"))
}

/// The 16-channel `AutoencoderKL`, `VAE_CH` channels at every level, one resnet per encoder
/// level (two per decoder level, as diffusers' `layers_per_block + 1`).
fn vae(dir: &Path, seed: u64) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let c = VAE_CH;
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "AutoencoderKL",
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": LATENT,
            "block_out_channels": vec![c; VAE_LEVELS],
            "down_block_types": vec!["DownEncoderBlock2D"; VAE_LEVELS],
            "up_block_types": vec!["UpDecoderBlock2D"; VAE_LEVELS],
            "layers_per_block": 1,
            "norm_num_groups": 32,
            "act_fn": "silu",
            "sample_size": 64,
            "scaling_factor": 1.5305,
            "shift_factor": 0.0609,
            "use_quant_conv": false,
            "use_post_quant_conv": false,
            "force_upcast": true,
        }),
    )?;
    let mut st = St::new(seed);
    let resnet = |st: &mut St, p: &str| {
        st.norm(&format!("{p}.norm1"), c, true);
        st.conv(&format!("{p}.conv1"), c, c, 1.0);
        st.norm(&format!("{p}.norm2"), c, true);
        st.conv(&format!("{p}.conv2"), c, c, 0.5);
    };
    let mid = |st: &mut St, p: &str| {
        resnet(st, &format!("{p}.mid_block.resnets.0"));
        let a = format!("{p}.mid_block.attentions.0");
        st.norm(&format!("{a}.group_norm"), c, true);
        for n in ["to_q", "to_k", "to_v", "to_out.0"] {
            st.linear(&format!("{a}.{n}"), c, c, 1.0, true);
        }
        resnet(st, &format!("{p}.mid_block.resnets.1"));
        st.norm(&format!("{p}.conv_norm_out"), c, true);
    };
    st.conv("encoder.conv_in", c, 3, 1.0);
    for i in 0..VAE_LEVELS {
        resnet(&mut st, &format!("encoder.down_blocks.{i}.resnets.0"));
        if i + 1 < VAE_LEVELS {
            st.conv(&format!("encoder.down_blocks.{i}.downsamplers.0.conv"), c, c, 1.0);
        }
    }
    mid(&mut st, "encoder");
    st.conv("encoder.conv_out", 2 * LATENT, c, 0.5);
    st.conv("decoder.conv_in", c, LATENT, 1.0);
    mid(&mut st, "decoder");
    for i in 0..VAE_LEVELS {
        for r in 0..2 {
            resnet(&mut st, &format!("decoder.up_blocks.{i}.resnets.{r}"));
        }
        if i + 1 < VAE_LEVELS {
            st.conv(&format!("decoder.up_blocks.{i}.upsamplers.0.conv"), c, c, 1.0);
        }
    }
    st.conv("decoder.conv_out", 3, c, 0.5);
    st.write(&dir.join("diffusion_pytorch_model.safetensors"))
}

/// Writes a tiny random SD 3.5 diffusers folder to `dir` (created if needed): what
/// [`super::Pipeline::load`] reads, with all three text encoders and the VAE encoder. The
/// contents depend on nothing but this code.
pub fn write_tiny_release(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    write_json(
        &dir.join("model_index.json"),
        &json!({
            "_class_name": "StableDiffusion3Pipeline",
            "_diffusers_version": "0.31.0",
            "_name_or_path": "oxinfer/sd3-tiny-random",
            "image_encoder": [null, null],
            "feature_extractor": [null, null],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "CLIPTextModelWithProjection"],
            "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
            "text_encoder_3": ["transformers", "T5EncoderModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "tokenizer_2": ["transformers", "CLIPTokenizer"],
            "tokenizer_3": ["transformers", "T5TokenizerFast"],
            "transformer": ["diffusers", "SD3Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }),
    )?;
    let sched = dir.join("scheduler");
    std::fs::create_dir_all(&sched)?;
    write_json(
        &sched.join("scheduler_config.json"),
        &json!({"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": 3.0}),
    )?;
    clip_tokenizer(&dir.join("tokenizer"), "<|endoftext|>")?;
    // SD 3's OpenCLIP tokenizer pads with "!".
    clip_tokenizer(&dir.join("tokenizer_2"), "!")?;
    clip_encoder(&dir.join("text_encoder"), CLIP_L, "quick_gelu", 0x5d3_0001)?;
    clip_encoder(&dir.join("text_encoder_2"), CLIP_G, "gelu", 0x5d3_0002)?;
    t5(&dir.join("text_encoder_3"), &dir.join("tokenizer_3"), 0x5d3_0003)?;
    transformer(&dir.join("transformer"), 0x5d3_0004)?;
    vae(&dir.join("vae"), 0x5d3_0005)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sd3::{LoadOptions, Options, Pipeline};

    fn check(img: &crate::sd3::Image) {
        assert_eq!((img.width, img.height, img.rgb.len()), (64, 64, 64 * 64 * 3));
        // Random weights give noise, not a constant image.
        assert!(img.rgb.iter().any(|&v| v != img.rgb[0]));
    }

    #[test]
    fn tiny_sd3_release() {
        let dir = std::env::temp_dir().join(format!("oxinfer-sd3-tiny-{}", std::process::id()));
        write_tiny_release(&dir).unwrap();
        assert!(crate::sd3::is_sd3_dir(&dir));
        let p = Pipeline::load(
            dir.to_str().unwrap(),
            &LoadOptions {
                // Not Auto: the test process must not settle on a GPU other tests do not use.
                device: crate::platform::DevicePref::Cpu,
                ..LoadOptions::default()
            },
        );
        let res = p.and_then(|p| {
            assert!(p.t5.is_some() && p.vae.encoder.is_some());
            assert_eq!(p.transformer.cfg.dual_attention_layers, vec![0]);
            assert!(p.transformer.cfg.qk_norm);
            // The conditioning is finite.
            let c = p.encode_prompt("a photo of a cat", 16)?;
            assert_eq!(c.tokens, 77 + 16);
            assert!(c.states.iter().chain(&c.pooled).all(|v| v.is_finite()));
            // So are one denoising call and the VAE round trip (the 8-bit images cannot show it).
            let mut rng = crate::sd::ops::Rng::new(1);
            let x = rng.normals(8 * 8 * 16);
            let v = p.transformer.forward(&x, 8, 8, &c.states, c.tokens, &c.pooled, 500.0)?;
            assert!(v.iter().all(|v| v.is_finite()) && v.iter().any(|&v| v != 0.0));
            let (px, h, w) = p.vae.decode(&v, 8, 8);
            assert!((h, w) == (64, 64) && px.iter().all(|v| v.is_finite()));
            let (z, lh, lw) = p.vae.encode(&px, 64, 64)?;
            assert!((lh, lw) == (8, 8) && z.iter().all(|v| v.is_finite()));

            let o = Options {
                width: Some(64),
                height: Some(64),
                steps: Some(3),
                ..Default::default()
            };
            let mut calls = Vec::new();
            let (img, tm) = p.generate("a photo of a cat", &o, &mut |i, n| {
                calls.push((i, n));
                true
            })?;
            check(&img);
            assert_eq!(calls, vec![(1, 3), (2, 3), (3, 3)]);
            // Guidance 7 by default: two passes per step.
            assert_eq!(tm.transformer_calls, 6);
            // The same seed gives the same image.
            let (again, _) = p.generate("a photo of a cat", &o, &mut |_, _| true)?;
            assert_eq!(again.rgb, img.rgb);

            let init = crate::video::RgbImage {
                width: 64,
                height: 64,
                data: img.rgb.clone(),
            };
            let o = Options {
                steps: Some(5),
                strength: 0.6,
                ..Default::default()
            };
            let mut calls = 0;
            let (out, _) = p.img2img(&init, "a dog", &o, &mut |_, n| {
                calls += 1;
                assert_eq!(n, 3);
                true
            })?;
            check(&out);
            assert_eq!(calls, 3);
            Ok(())
        });
        std::fs::remove_dir_all(&dir).ok();
        res.unwrap();
    }

    /// `OXINFER_TINY_SD3_OUT=DIR cargo test --release --lib sd3::tiny -- --ignored` writes the
    /// folder to `DIR/sd3-tiny` and leaves it there (to point a server at).
    #[test]
    #[ignore]
    fn write_tiny_sd3_out() {
        let Some(out) = std::env::var_os("OXINFER_TINY_SD3_OUT") else {
            eprintln!("set OXINFER_TINY_SD3_OUT=DIR to write DIR/sd3-tiny");
            return;
        };
        let dir = Path::new(&out).join("sd3-tiny");
        write_tiny_release(&dir).unwrap();
        eprintln!("wrote {}", dir.display());
    }
}
