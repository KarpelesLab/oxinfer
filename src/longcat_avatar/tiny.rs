//! A tiny random LongCat-Video-Avatar release and the parts of LongCat-Video it borrows, for
//! tests that run the pipeline end to end without the 80 GB of real files. The weights are noise
//! from a fixed-seed generator (norm weights near 1, matrices uniform with `sqrt(3 / fan_in)`
//! times a gain), so the outputs are deterministic but meaningless.
//!
//! [`write_tiny_release`] writes, under one folder:
//!
//! - `LongCat-Video-Avatar-1.6/`: `base_model/` (the transformer in bfloat16: 2 blocks of 64
//!   wide, one head of 64, 4 latent channels, text width 16, audio width 16), `base_model_int8/`
//!   (the same weights as the release's quantizer writes them: int8 rows with a scale each,
//!   `final_layer.linear` and the patch embedding left in float), `loraA/` and `loraB/` (rank 4
//!   on the seven projections the released LoRAs target, with split up-projections for the fused
//!   ones), `scheduler/` (shift 7) and `whisper-large-v3/` (32 encoder layers of 16 wide: the
//!   audio features group 33 hidden states);
//! - `LongCat-Video/`: `vae/` (the Wan 2.1 layout with 4 base channels and 4 latent channels),
//!   `text_encoder/` (umT5: 2 blocks of 16 wide with a bias table each) and `tokenizer/` (a
//!   small Unigram vocabulary).

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

/// Tensors of one safetensors file.
struct Tensors {
    list: Vec<(String, &'static str, Vec<usize>, Vec<u8>)>,
    rng: Rng,
}

impl Tensors {
    fn new(seed: u64) -> Self {
        Self {
            list: Vec::new(),
            rng: Rng(seed),
        }
    }

    fn put(&mut self, name: &str, shape: &[usize], v: &[f32]) {
        debug_assert_eq!(v.len(), shape.iter().product::<usize>(), "{name}");
        let bytes = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.list.push((name.to_string(), "F32", shape.to_vec(), bytes));
    }

    /// `bias + scale * U(-1, 1)`.
    fn uni(&mut self, name: &str, shape: &[usize], scale: f32, bias: f32) -> Vec<f32> {
        let v: Vec<f32> = (0..shape.iter().product()).map(|_| bias + scale * self.rng.next()).collect();
        self.put(name, shape, &v);
        v
    }

    /// A weight whose fan-in is the product of all but its first dimension.
    fn mat(&mut self, name: &str, shape: &[usize], gain: f32) -> Vec<f32> {
        let fan_in: usize = shape[1..].iter().product();
        self.uni(name, shape, gain * (3.0 / fan_in as f32).sqrt(), 0.0)
    }

    fn ones(&mut self, name: &str, n: usize) {
        self.uni(name, &[n], 0.1, 1.0);
    }

    fn bias(&mut self, name: &str, n: usize) -> Vec<f32> {
        self.uni(name, &[n], 0.02, 0.0)
    }

    fn lin(&mut self, p: &str, o: usize, i: usize, gain: f32, bias: bool) {
        self.mat(&format!("{p}.weight"), &[o, i], gain);
        if bias {
            self.bias(&format!("{p}.bias"), o);
        }
    }

    fn layer_norm(&mut self, p: &str, n: usize) {
        self.ones(&format!("{p}.weight"), n);
        self.bias(&format!("{p}.bias"), n);
    }

    fn write(mut self, path: &Path) -> Result<()> {
        // Float32 tensors first: every one then starts 4-byte aligned.
        self.list.sort_by_key(|t| t.1 != "F32");
        let mut head = serde_json::Map::new();
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

pub const LATENT: usize = 4;
pub const HIDDEN: usize = 64;
const HEADS: usize = 1;
const DEPTH: usize = 2;
const FFN: usize = 256;
pub const TEXT: usize = 16;
const T_EMBED: usize = 16;
const FREQ: usize = 16;
pub const AUDIO: usize = 16;
const AUDIO_BLOCKS: usize = 5;
const AUDIO_MID: usize = 8;
const AUDIO_OUT: usize = 32;
const AUDIO_TOKENS: usize = 2;
pub const LORA_RANK: usize = 4;

/// The transformer's float tensors, and the int8 copy of the same weights.
fn transformer(dir: &Path, seed: u64) -> Result<()> {
    let cfg = json!({
        "_class_name": "LongCatVideoAvatarTransformer3DModel",
        "in_channels": LATENT, "out_channels": LATENT, "hidden_size": HIDDEN, "depth": DEPTH, "num_heads": HEADS,
        "caption_channels": TEXT, "model_max_length": 512, "mlp_ratio": 4, "adaln_tembed_dim": T_EMBED,
        "frequency_embedding_size": FREQ, "patch_size": [1, 2, 2], "text_tokens_zero_pad": true, "audio_window": 5,
        "audio_block": AUDIO_BLOCKS, "audio_channel": AUDIO, "intermediate_dim": AUDIO_MID, "output_dim": AUDIO_OUT,
        "context_tokens": AUDIO_TOKENS, "vae_scale": 4, "audio_prenorm": false, "class_range": 24, "class_interval": 4,
    });
    let mut t = Tensors::new(seed);
    let hd = HIDDEN / HEADS;
    t.mat("x_embedder.proj.weight", &[HIDDEN, LATENT, 1, 2, 2], 1.0);
    t.bias("x_embedder.proj.bias", HIDDEN);
    t.lin("t_embedder.mlp.0", T_EMBED, FREQ, 1.0, true);
    t.lin("t_embedder.mlp.2", T_EMBED, T_EMBED, 1.0, true);
    t.lin("y_embedder.y_proj.0", HIDDEN, TEXT, 1.0, true);
    t.lin("y_embedder.y_proj.2", HIDDEN, HIDDEN, 1.0, true);
    t.lin("audio_proj.proj1", AUDIO_MID, 5 * AUDIO_BLOCKS * AUDIO, 1.0, true);
    t.lin("audio_proj.proj1_vf", AUDIO_MID, 8 * AUDIO_BLOCKS * AUDIO, 1.0, true);
    t.lin("audio_proj.proj2", AUDIO_MID, AUDIO_MID, 1.0, true);
    t.lin("audio_proj.proj3", AUDIO_TOKENS * AUDIO_OUT, AUDIO_MID, 1.0, true);
    t.layer_norm("audio_proj.norm", AUDIO_OUT);
    for i in 0..DEPTH {
        let p = format!("blocks.{i}");
        t.lin(&format!("{p}.adaLN_modulation.1"), 6 * HIDDEN, T_EMBED, 0.3, true);
        t.lin(&format!("{p}.audio_adaLN_modulation.1"), 3 * HIDDEN, T_EMBED, 0.3, true);
        t.lin(&format!("{p}.attn.qkv"), 3 * HIDDEN, HIDDEN, 1.0, true);
        t.lin(&format!("{p}.attn.proj"), HIDDEN, HIDDEN, 0.5, true);
        for a in ["attn", "cross_attn", "audio_cross_attn"] {
            t.ones(&format!("{p}.{a}.q_norm.weight"), hd);
            t.ones(&format!("{p}.{a}.k_norm.weight"), hd);
        }
        t.layer_norm(&format!("{p}.pre_crs_attn_norm"), HIDDEN);
        t.lin(&format!("{p}.cross_attn.q_linear"), HIDDEN, HIDDEN, 1.0, true);
        t.lin(&format!("{p}.cross_attn.kv_linear"), 2 * HIDDEN, HIDDEN, 1.0, true);
        t.lin(&format!("{p}.cross_attn.proj"), HIDDEN, HIDDEN, 0.5, true);
        t.layer_norm(&format!("{p}.pre_video_crs_attn_norm"), HIDDEN);
        t.lin(&format!("{p}.audio_cross_attn.q_linear"), HIDDEN, HIDDEN, 1.0, true);
        t.lin(&format!("{p}.audio_cross_attn.kv_linear"), 2 * HIDDEN, AUDIO_OUT, 1.0, true);
        t.lin(&format!("{p}.audio_cross_attn.proj"), HIDDEN, HIDDEN, 0.5, true);
        t.lin(&format!("{p}.ffn.w1"), FFN, HIDDEN, 1.0, false);
        t.lin(&format!("{p}.ffn.w3"), FFN, HIDDEN, 1.0, false);
        t.lin(&format!("{p}.ffn.w2"), HIDDEN, FFN, 0.5, false);
    }
    t.lin("final_layer.adaLN_modulation.1", 2 * HIDDEN, T_EMBED, 0.3, true);
    t.lin("final_layer.linear", 4 * LATENT, HIDDEN, 1.0, true);
    // The release is bfloat16: the values rounded to it (the int8 copy is made from them).
    for e in t.list.iter_mut() {
        e.3 =
            e.3.as_chunks::<4>()
                .0
                .iter()
                .flat_map(|c| half::bf16::from_f32(f32::from_le_bytes(*c)).to_f32().to_le_bytes())
                .collect();
    }

    // The int8 copy: every matrix but the patch embedding and `final_layer.linear`, as
    // `QuantizedLinear.from_linear` (scale = max |row| / 127, round, clamp).
    let mut q = Tensors::new(0);
    for (name, dtype, shape, bytes) in &t.list {
        let keep = !name.ends_with(".weight") || shape.len() != 2 || name.starts_with("final_layer.linear");
        if keep {
            q.list.push((name.clone(), dtype, shape.clone(), bytes.clone()));
            continue;
        }
        let (o, i) = (shape[0], shape[1]);
        let w: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        let mut ints = Vec::with_capacity(o * i);
        let mut scales = Vec::with_capacity(o);
        for row in w.chunks(i) {
            let s = row.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-8) / 127.0;
            scales.push(s);
            ints.extend(row.iter().map(|&v| (v / s).round().clamp(-128.0, 127.0) as i8 as u8));
        }
        let base = name.strip_suffix(".weight").unwrap();
        q.list.push((format!("{base}.weight_int8"), "I8", shape.clone(), ints));
        q.put(&format!("{base}.weight_scale"), &[o], &scales);
    }
    for e in t.list.iter_mut() {
        e.1 = "BF16";
        e.3 =
            e.3.as_chunks::<4>()
                .0
                .iter()
                .flat_map(|c| half::bf16::from_f32(f32::from_le_bytes(*c)).to_bits().to_le_bytes())
                .collect();
    }
    write_json(&dir.join("base_model/config.json"), &cfg)?;
    t.write(&dir.join("base_model/diffusion_pytorch_model.safetensors"))?;
    write_json(&dir.join("base_model_int8/config.json"), &cfg)?;
    write_json(
        &dir.join("base_model_int8/quantization_config.json"),
        &json!({"quantization_method": "int8_per_channel_symmetric", "skip_patterns": ["final_layer.linear"]}),
    )?;
    q.write(&dir.join("base_model_int8/quantized_model-00001-of-00001.safetensors"))
}

/// A LoRA in the released files' layout.
fn lora(path: &Path, seed: u64) -> Result<()> {
    let mut t = Tensors::new(seed);
    let r = LORA_RANK;
    for i in 0..DEPTH {
        for (name, o, n_in, sep) in [
            ("attn.qkv", 3 * HIDDEN, HIDDEN, 3),
            ("attn.proj", HIDDEN, HIDDEN, 1),
            ("cross_attn.q_linear", HIDDEN, HIDDEN, 1),
            ("cross_attn.kv_linear", 2 * HIDDEN, HIDDEN, 2),
            ("ffn.w1", FFN, HIDDEN, 1),
            ("ffn.w2", HIDDEN, FFN, 1),
            ("ffn.w3", FFN, HIDDEN, 1),
        ] {
            let key = format!("lora___lorahyphen___blocks.{i}.{name}").replace('.', "___lorahyphen___");
            t.put(&format!("{key}.alpha_scale"), &[], &[0.5]);
            t.mat(&format!("{key}.lora_down.weight"), &[sep * r, n_in], 0.5);
            if sep == 1 {
                t.mat(&format!("{key}.lora_up.weight"), &[o, r], 0.5);
            } else {
                for j in 0..sep {
                    t.mat(&format!("{key}.lora_up.blocks.{j}.weight"), &[o / sep, r], 0.5);
                }
            }
        }
    }
    t.write(path)
}

const WHISPER: usize = AUDIO;

/// A Whisper checkpoint with 32 encoder layers and no decoder layers.
fn whisper(dir: &Path, seed: u64) -> Result<()> {
    let (d, ffn, mels, vocab) = (WHISPER, 32, 128, 8);
    write_json(
        &dir.join("config.json"),
        &json!({
            "model_type": "whisper", "d_model": d, "encoder_layers": 32, "encoder_attention_heads": 2, "encoder_ffn_dim": ffn,
            "decoder_layers": 0, "decoder_attention_heads": 2, "decoder_ffn_dim": ffn, "num_mel_bins": mels, "max_source_positions": 1500,
            "max_target_positions": 8, "vocab_size": vocab, "activation_function": "gelu", "scale_embedding": false,
        }),
    )?;
    write_json(
        &dir.join("preprocessor_config.json"),
        &json!({"feature_extractor_type": "WhisperFeatureExtractor", "feature_size": mels, "sampling_rate": 16000, "hop_length": 160, "n_fft": 400, "chunk_length": 30}),
    )?;
    let mut t = Tensors::new(seed);
    t.mat("model.encoder.conv1.weight", &[d, mels, 3], 1.0);
    t.bias("model.encoder.conv1.bias", d);
    t.mat("model.encoder.conv2.weight", &[d, d, 3], 1.0);
    t.bias("model.encoder.conv2.bias", d);
    t.uni("model.encoder.embed_positions.weight", &[1500, d], 0.3, 0.0);
    for l in 0..32 {
        let p = format!("model.encoder.layers.{l}");
        t.layer_norm(&format!("{p}.self_attn_layer_norm"), d);
        t.lin(&format!("{p}.self_attn.q_proj"), d, d, 1.0, true);
        t.lin(&format!("{p}.self_attn.k_proj"), d, d, 1.0, false);
        t.lin(&format!("{p}.self_attn.v_proj"), d, d, 1.0, true);
        t.lin(&format!("{p}.self_attn.out_proj"), d, d, 0.3, true);
        t.layer_norm(&format!("{p}.final_layer_norm"), d);
        t.lin(&format!("{p}.fc1"), ffn, d, 1.0, true);
        t.lin(&format!("{p}.fc2"), d, ffn, 0.3, true);
    }
    t.layer_norm("model.encoder.layer_norm", d);
    t.uni("model.decoder.embed_positions.weight", &[8, d], 0.3, 0.0);
    t.uni("model.decoder.embed_tokens.weight", &[vocab, d], 0.3, 0.0);
    t.layer_norm("model.decoder.layer_norm", d);
    t.write(&dir.join("model.safetensors"))
}

/// umT5 (a relative-position bias in every block) and a Unigram tokenizer.
fn text_encoder(dir: &Path, tok_dir: &Path, seed: u64) -> Result<()> {
    let mut pieces: Vec<(String, f64)> = vec![
        ("<pad>".into(), 0.0),
        ("</s>".into(), 0.0),
        ("<unk>".into(), 0.0),
        ("▁".into(), -2.0),
    ];
    for w in ["▁a", "▁woman", "▁man", "▁is", "▁speaking", "▁the", "▁camera", "▁to"] {
        pieces.push((w.into(), -3.0));
    }
    for c in ('a'..='z').chain('0'..='9').chain(".,!?'-".chars()) {
        pieces.push((c.to_string(), -6.0));
    }
    let vocab = pieces.len();
    let (heads, d_kv, d_ff, layers, buckets) = (2, 8, 2 * TEXT, 2, 32);
    write_json(
        &dir.join("config.json"),
        &json!({
            "architectures": ["UMT5EncoderModel"], "model_type": "umt5", "d_model": TEXT, "d_kv": d_kv, "d_ff": d_ff, "num_heads": heads,
            "num_layers": layers, "feed_forward_proj": "gated-gelu", "dense_act_fn": "gelu_new", "is_gated_act": true, "vocab_size": vocab,
            "layer_norm_epsilon": 1e-6, "relative_attention_num_buckets": buckets, "relative_attention_max_distance": 128,
            "pad_token_id": 0, "eos_token_id": 1, "torch_dtype": "float32",
        }),
    )?;
    let mut t = Tensors::new(seed);
    t.uni("shared.weight", &[vocab, TEXT], 1.0, 0.0);
    for i in 0..layers {
        let p = format!("encoder.block.{i}.layer");
        t.ones(&format!("{p}.0.layer_norm.weight"), TEXT);
        t.lin(&format!("{p}.0.SelfAttention.q"), heads * d_kv, TEXT, 0.2, false);
        t.lin(&format!("{p}.0.SelfAttention.k"), heads * d_kv, TEXT, 1.0, false);
        t.lin(&format!("{p}.0.SelfAttention.v"), heads * d_kv, TEXT, 1.0, false);
        t.lin(&format!("{p}.0.SelfAttention.o"), TEXT, heads * d_kv, 0.5, false);
        t.uni(
            &format!("{p}.0.SelfAttention.relative_attention_bias.weight"),
            &[buckets, heads],
            0.5,
            0.0,
        );
        t.ones(&format!("{p}.1.layer_norm.weight"), TEXT);
        t.lin(&format!("{p}.1.DenseReluDense.wi_0"), d_ff, TEXT, 1.0, false);
        t.lin(&format!("{p}.1.DenseReluDense.wi_1"), d_ff, TEXT, 1.0, false);
        t.lin(&format!("{p}.1.DenseReluDense.wo"), TEXT, d_ff, 0.5, false);
    }
    t.ones("encoder.final_layer_norm.weight", TEXT);
    t.write(&dir.join("model.safetensors"))?;
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
    )
}

/// The Wan 2.1 VAE layout with `base` channels.
fn vae(dir: &Path, seed: u64) -> Result<()> {
    let (base, z) = (4usize, LATENT);
    let mult = [1usize, 2, 4, 4];
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "AutoencoderKLWan", "base_dim": base, "z_dim": z, "dim_mult": mult, "num_res_blocks": 2, "attn_scales": [],
            "temperal_downsample": [false, true, true], "dropout": 0.0,
            "latents_mean": [0.1, -0.2, 0.3, 0.0], "latents_std": [1.2, 0.8, 1.0, 1.5],
        }),
    )?;
    let mut t = Tensors::new(seed);
    let conv3 = |t: &mut Tensors, p: &str, o: usize, i: usize, k: [usize; 3]| {
        t.mat(&format!("{p}.weight"), &[o, i, k[0], k[1], k[2]], 0.7);
        t.bias(&format!("{p}.bias"), o);
    };
    let conv2 = |t: &mut Tensors, p: &str, o: usize, i: usize, k: usize| {
        t.mat(&format!("{p}.weight"), &[o, i, k, k], 0.7);
        t.bias(&format!("{p}.bias"), o);
    };
    let res = |t: &mut Tensors, p: &str, i: usize, o: usize| {
        t.uni(&format!("{p}.norm1.gamma"), &[i, 1, 1, 1], 0.1, 1.0);
        conv3(t, &format!("{p}.conv1"), o, i, [3, 3, 3]);
        t.uni(&format!("{p}.norm2.gamma"), &[o, 1, 1, 1], 0.1, 1.0);
        conv3(t, &format!("{p}.conv2"), o, o, [3, 3, 3]);
        if i != o {
            conv3(t, &format!("{p}.conv_shortcut"), o, i, [1, 1, 1]);
        }
    };
    let mid = |t: &mut Tensors, p: &str, c: usize| {
        res(t, &format!("{p}.resnets.0"), c, c);
        t.uni(&format!("{p}.attentions.0.norm.gamma"), &[c, 1, 1], 0.1, 1.0);
        conv2(t, &format!("{p}.attentions.0.to_qkv"), 3 * c, c, 1);
        conv2(t, &format!("{p}.attentions.0.proj"), c, c, 1);
        res(t, &format!("{p}.resnets.1"), c, c);
    };
    // Encoder.
    let dims: Vec<usize> = std::iter::once(base).chain(mult.iter().map(|m| base * m)).collect();
    conv3(&mut t, "encoder.conv_in", dims[0], 3, [3, 3, 3]);
    let t_down = [false, true, true];
    let mut idx = 0;
    for i in 0..4 {
        let (mut cin, cout) = (dims[i], dims[i + 1]);
        for _ in 0..2 {
            res(&mut t, &format!("encoder.down_blocks.{idx}"), cin, cout);
            idx += 1;
            cin = cout;
        }
        if i != 3 {
            let p = format!("encoder.down_blocks.{idx}");
            conv2(&mut t, &format!("{p}.resample.1"), cout, cout, 3);
            if t_down[i] {
                conv3(&mut t, &format!("{p}.time_conv"), cout, cout, [3, 1, 1]);
            }
            idx += 1;
        }
    }
    let top = dims[4];
    mid(&mut t, "encoder.mid_block", top);
    t.uni("encoder.norm_out.gamma", &[top, 1, 1, 1], 0.1, 1.0);
    conv3(&mut t, "encoder.conv_out", 2 * z, top, [3, 3, 3]);
    conv3(&mut t, "quant_conv", 2 * z, 2 * z, [1, 1, 1]);
    conv3(&mut t, "post_quant_conv", z, z, [1, 1, 1]);
    // Decoder.
    let rev: Vec<usize> = mult.iter().rev().map(|m| base * m).collect();
    let dims: Vec<usize> = std::iter::once(rev[0]).chain(rev.iter().copied()).collect();
    conv3(&mut t, "decoder.conv_in", dims[0], z, [3, 3, 3]);
    mid(&mut t, "decoder.mid_block", dims[0]);
    let t_up = [true, true, false];
    let mut last = dims[0];
    for i in 0..4 {
        let (mut cin, cout) = (dims[i], dims[i + 1]);
        if i > 0 {
            cin /= 2;
        }
        for j in 0..3 {
            res(&mut t, &format!("decoder.up_blocks.{i}.resnets.{j}"), cin, cout);
            cin = cout;
        }
        if i != 3 {
            let p = format!("decoder.up_blocks.{i}.upsamplers.0");
            conv2(&mut t, &format!("{p}.resample.1"), cout / 2, cout, 3);
            if t_up[i] {
                conv3(&mut t, &format!("{p}.time_conv"), 2 * cout, cout, [3, 1, 1]);
            }
        }
        last = cout;
    }
    t.uni("decoder.norm_out.gamma", &[last, 1, 1, 1], 0.1, 1.0);
    conv3(&mut t, "decoder.conv_out", 3, last, [3, 3, 3]);
    t.write(&dir.join("diffusion_pytorch_model.safetensors"))
}

/// Writes the two folders under `dir`; returns (the release, the base repository's parts).
pub fn write_tiny_release(dir: &Path) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let avatar = dir.join("LongCat-Video-Avatar-1.6");
    let base = dir.join("LongCat-Video");
    write_json(&avatar.join("config.json"), &json!({"model_name": "LongCat-Video-Avatar-1.6"}))?;
    write_json(&avatar.join("model_index.json"), &json!({"model_name": "LongCat-Video-Avatar-1.6"}))?;
    write_json(
        &avatar.join("scheduler/scheduler_config.json"),
        &json!({"_class_name": "FlowMatchEulerDiscreteScheduler", "shift": 7.0, "use_dynamic_shifting": false}),
    )?;
    transformer(&avatar, 11)?;
    lora(&avatar.join("loraA/lora.safetensors"), 12)?;
    lora(&avatar.join("loraB/lora.safetensors"), 13)?;
    whisper(&avatar.join("whisper-large-v3"), 14)?;
    text_encoder(&base.join("text_encoder"), &base.join("tokenizer"), 15)?;
    vae(&base.join("vae"), 16)?;
    Ok((avatar, base))
}

#[cfg(test)]
mod tests {
    use super::super::dit::{Cond, Dit};
    use super::super::lora::{Active, LoraNet};
    use super::super::pipeline::{AvatarParams, AvatarPipeline, Sources, bucket, clean_prompt, sigmas};
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("oxinfer-longcat-avatar-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn sources(dir: &Path, int8: bool) -> Sources {
        let (avatar, base) = write_tiny_release(dir).unwrap();
        Sources {
            avatar,
            base,
            int8,
            text_quant: crate::model::Quant::Bf16,
        }
    }

    fn params() -> AvatarParams {
        let n = 16000 * 2;
        AvatarParams {
            prompt: "a woman is speaking to the camera".into(),
            speech: vec![
                (0..n)
                    .map(|i| 0.3 * (i as f32 * 0.05).sin() * (1.0 + (i as f32 * 0.001).sin()))
                    .collect(),
            ],
            size: Some((64, 48)),
            frames: 9,
            ..Default::default()
        }
    }

    fn image() -> crate::sd::Image {
        let (w, h) = (96, 64);
        crate::sd::Image {
            width: w,
            height: h,
            rgb: (0..w * h * 3).map(|i| ((i * 37) % 251) as u8).collect(),
        }
    }

    #[test]
    fn schedule_and_buckets() {
        // The reference's scheduler at shift 7 (tools/longcat_avatar_tiny.py, `sched_sigmas`).
        let want = [
            1.0,
            0.979_977_55,
            0.954_487_5,
            0.920_936_1,
            0.874_780_9,
            0.807_277_2,
            0.699_157_66,
            0.497_991_98,
            0.0,
        ];
        let got = sigmas(8, true, 7.0);
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?}");
        }
        assert_eq!(sigmas(3, false, 1.0), vec![1.0, 0.5005, 0.001, 0.0]);
        // 300 x 500 is closest to the 0.58 bucket; a square image to 0.95 (608 x 640).
        assert_eq!(bucket(300, 500, false), (480, 832));
        assert_eq!(bucket(512, 512, false), (608, 640));
        assert_eq!(bucket(720, 1280, true), (704, 1312));
    }

    #[test]
    fn prompt_cleaning() {
        assert_eq!(
            clean_prompt("  Two people  are\nspeaking &amp;amp; smiling "),
            "Two people are speaking & smiling"
        );
        assert_eq!(clean_prompt("镜头静止，“好”　ＡＢ１！ ﬁne"), "镜头静止,\"好\" AB1! fine");
    }

    /// Text and speech to video, then a reference image, a continuation and guidance: the shapes
    /// and that the int8 transformer follows the float one.
    #[test]
    fn tiny_pipeline() {
        let dir = tmp("pipe");
        let pipe = AvatarPipeline::load(&sources(&dir, false)).unwrap();
        let mut stages = 0;
        let clip = pipe.generate(&params(), &mut |_| stages += 1).unwrap();
        assert_eq!((clip.width, clip.height, clip.frames.len()), (64, 48, 9));
        assert!(clip.frames.iter().all(|f| f.len() == 64 * 48 * 3));
        assert!(clip.latents.iter().all(|v| v.is_finite()));
        assert!(stages > 8);
        // The same request again gives the same frames.
        let again = pipe.generate(&params(), &mut |_| {}).unwrap();
        assert_eq!(clip.frames, again.frames);

        // int8: the same request stays close to the float weights' latents.
        let pipe8 = AvatarPipeline::load(&sources(&tmp("pipe8"), true)).unwrap();
        assert!(pipe8.dit.int8);
        let clip8 = pipe8.generate(&params(), &mut |_| {}).unwrap();
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in clip.latents.iter().zip(&clip8.latents) {
            num += ((a - b) as f64).powi(2);
            den += (*a as f64).powi(2);
        }
        assert!(num / den < 1e-2, "int8 latents {:.3e} away", num / den);

        // With a reference image the canvas is the bucket of its aspect ratio (480p-sized: not run here).
        let p = AvatarParams {
            image: Some(image()),
            ..params()
        };
        assert_eq!(pipe.canvas(&p).unwrap(), (768, 512));
        // A continuation (17 frames, then 4 more) with text and audio guidance on the base model.
        let p = AvatarParams {
            frames: 17,
            segments: 2,
            steps: 2,
            distill: false,
            text_guidance: 3.0,
            audio_guidance: 2.0,
            ..params()
        };
        let clip = pipe.generate(&p, &mut |_| {}).unwrap();
        assert_eq!(clip.frames.len(), 17 + 4);
        assert!(clip.latents.iter().all(|v| v.is_finite()));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(tmp("pipe8"));
    }

    /// The LoRA files as read, and their side paths: no effect at multiplier 0, each LoRA its own.
    #[test]
    fn lora_side_path() {
        let dir = tmp("lora");
        let src = sources(&dir, false);
        let dit = Dit::load(&src.avatar.join("base_model")).unwrap();
        let (a, b) = (
            LoraNet::load(&src.avatar.join("loraA/lora.safetensors")).unwrap(),
            LoraNet::load(&src.avatar.join("loraB/lora.safetensors")).unwrap(),
        );
        assert_eq!(a.mods.len(), 7 * DEPTH);
        let m = &a.mods["blocks.0.attn.qkv"];
        assert_eq!(
            (m.ups.len(), m.r, m.n_in, m.n_out, m.alpha_scale),
            (3, LORA_RANK, HIDDEN, 3 * HIDDEN, 0.5)
        );
        let (t, h, w) = (3, 4, 6);
        let mut rng = Rng(5);
        let x: Vec<f32> = (0..LATENT * t * h * w).map(|_| rng.next()).collect();
        let text: Vec<f32> = (0..3 * TEXT).map(|_| rng.next()).collect();
        let audio: Vec<f32> = (0..9 * 5 * AUDIO_BLOCKS * AUDIO).map(|_| rng.next()).collect();
        let run = |nets: &[(&LoraNet, f32)]| {
            dit.forward(
                &x,
                (t, h, w),
                &[0.0, 700.0, 700.0],
                &Cond {
                    text: &text,
                    text_len: 8,
                    audio: &audio,
                    humans: 1,
                    num_cond: 1,
                    num_ref: 0,
                    ref_img_index: 0,
                    mask_frame_range: 0,
                    ref_masks: None,
                    lora: Active { nets },
                },
            )
            .unwrap()
        };
        let plain = run(&[]);
        let zero = run(&[(&a, 0.0), (&b, 0.0)]);
        assert_eq!(plain, zero);
        let with_a = run(&[(&a, 1.0), (&b, 0.0)]);
        let with_b = run(&[(&a, 0.0), (&b, 1.0)]);
        let diff = |p: &[f32], q: &[f32]| p.iter().zip(q).map(|(u, v)| (u - v).abs()).fold(0f32, f32::max);
        assert!(diff(&plain, &with_a) > 1e-3 && diff(&with_a, &with_b) > 1e-3);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
