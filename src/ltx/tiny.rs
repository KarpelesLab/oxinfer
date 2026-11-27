//! A tiny random LTX-2.5 release, for tests that run the pipeline end to end (the library, the
//! server's video route) without the real 22B / 12B checkpoints. The weights are noise from a
//! fixed-seed generator scaled so activations stay finite; the outputs are deterministic but
//! meaningless.
//!
//! [`write_tiny_release`] writes a folder in the release's layout, every part with the metadata
//! the loaders read:
//!
//! - `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors`: 2 blocks, video
//!   16 wide (2 heads of 8), audio 16 wide (2 heads of 8), 8 latent channels per stream, text
//!   connectors of one block and 4 registers;
//! - `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`: a Gemma 4 of 2 layers of 16
//!   (a sliding layer and a full one with `attention_k_eq_v`), a byte-level tokenizer, the two
//!   aggregate projections;
//! - `vae/ltx-2.5-video-vae-conv-bf16.safetensors`: encoder and decoder of three `compress_all`
//!   stages (8x in time, 32x in space with the 4x patch);
//! - `vae/ltx-2.5-audio-vae-bf16.safetensors`: the audio decoder (2 latent channels of 4 bins to
//!   16 mel bins), a vocoder of 160 samples per mel frame and a bandwidth extension (x3);
//! - the spatial upsampler and the duration head.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::path::Path;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// Tensors of one safetensors file, written by hand (float32 unless named otherwise).
struct Tensors {
    list: Vec<(String, &'static str, Vec<usize>, Vec<u8>)>,
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

    fn put(&mut self, name: &str, shape: &[usize], v: &[f32]) {
        debug_assert_eq!(v.len(), shape.iter().product::<usize>(), "{name}");
        self.list.push((
            name.to_string(),
            "F32",
            shape.to_vec(),
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ));
    }

    fn uni(&mut self, name: &str, shape: &[usize], scale: f32, bias: f32) {
        let v: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|_| bias + scale * self.rng.next())
            .collect();
        self.put(name, shape, &v);
    }

    /// A weight of unit gain over its fan-in (all but the first dimension).
    fn mat(&mut self, name: &str, shape: &[usize], gain: f32) {
        let fan_in: usize = shape[1..].iter().product();
        self.uni(name, shape, gain * (3.0 / fan_in as f32).sqrt(), 0.0);
    }

    fn lin(&mut self, p: &str, o: usize, i: usize, gain: f32, bias: bool) {
        self.mat(&format!("{p}.weight"), &[o, i], gain);
        if bias {
            self.uni(&format!("{p}.bias"), &[o], 0.02, 0.0);
        }
    }

    fn ones(&mut self, name: &str, n: usize) {
        self.uni(name, &[n], 0.1, 1.0);
    }

    fn bytes(&mut self, name: &str, b: &[u8]) {
        self.list.push((name.to_string(), "U8", vec![b.len()], b.to_vec()));
    }

    fn meta(&mut self, k: &str, v: Value) {
        self.meta.insert(k.to_string(), Value::String(v.to_string()));
    }

    fn write(mut self, path: &Path) -> Result<()> {
        // Float32 tensors first: every one then starts 4-byte aligned.
        self.list.sort_by_key(|t| t.1 != "F32");
        let mut head = serde_json::Map::new();
        head.insert("__metadata__".into(), Value::Object(std::mem::take(&mut self.meta)));
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in self.list {
            head.insert(
                name,
                json!({"dtype": dtype, "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]}),
            );
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

const D: usize = 16;
const HEADS: usize = 2;
const HD: usize = 8;
const AD: usize = 16;
const LAT: usize = 8;
const TEXT_H: usize = 16;
const TEXT_LAYERS: usize = 2;
const VOCAB: usize = 260;
const BOS: usize = 256;

fn attn(t: &mut Tensors, p: &str, q_dim: usize, ctx: usize, heads: usize, hd: usize) {
    let inner = heads * hd;
    t.lin(&format!("{p}.to_q"), inner, q_dim, 1.0, true);
    t.lin(&format!("{p}.to_k"), inner, ctx, 1.0, true);
    t.lin(&format!("{p}.to_v"), inner, ctx, 1.0, true);
    t.lin(&format!("{p}.to_out.0"), q_dim, inner, 0.5, true);
    t.lin(&format!("{p}.to_gate_logits"), heads, q_dim, 0.5, true);
    t.ones(&format!("{p}.q_norm.weight"), inner);
    t.ones(&format!("{p}.k_norm.weight"), inner);
}

fn ff(t: &mut Tensors, p: &str, dim: usize, bias: bool) {
    t.lin(&format!("{p}.net.0.proj"), 4 * dim, dim, 1.0, bias);
    t.lin(&format!("{p}.net.2"), dim, 4 * dim, 0.5, bias);
}

fn ada(t: &mut Tensors, p: &str, dim: usize, coeff: usize) {
    t.lin(&format!("{p}.emb.timestep_embedder.linear_1"), dim, 256, 1.0, true);
    t.lin(&format!("{p}.emb.timestep_embedder.linear_2"), dim, dim, 1.0, true);
    t.lin(&format!("{p}.linear"), coeff * dim, dim, 0.1, true);
}

fn write_dit(path: &Path) -> Result<()> {
    let mut t = Tensors::new(11);
    let config = json!({"transformer": {
        "_class_name": "AVTransformer3DModel", "num_layers": 2, "num_attention_heads": HEADS, "attention_head_dim": HD,
        "audio_num_attention_heads": HEADS, "audio_attention_head_dim": AD / HEADS, "in_channels": LAT, "out_channels": LAT,
        "audio_in_channels": LAT, "audio_out_channels": LAT, "cross_attention_dim": D, "audio_cross_attention_dim": AD,
        "norm_eps": 1e-6, "positional_embedding_theta": 10000.0, "positional_embedding_max_pos": [20, 2048, 2048],
        "audio_positional_embedding_max_pos": [20], "timestep_scale_multiplier": 1000, "av_ca_timestep_scale_multiplier": 1000.0,
        "connector_num_layers": 1, "connector_num_attention_heads": HEADS, "connector_attention_head_dim": HD,
        "audio_connector_num_attention_heads": HEADS, "audio_connector_attention_head_dim": AD / HEADS,
        "connector_num_learnable_registers": 4, "connector_positional_embedding_max_pos": [4096], "rope_type": "split",
        "caption_proj_before_connector": true, "frequencies_precision": "float64"
    }});
    t.meta("config", config);
    t.meta("model_version", json!("2.5.0"));
    let p = "model.diffusion_model.";
    for i in 0..2 {
        let b = format!("{p}transformer_blocks.{i}");
        attn(&mut t, &format!("{b}.attn1"), D, D, HEADS, HD);
        attn(&mut t, &format!("{b}.attn2"), D, D, HEADS, HD);
        ff(&mut t, &format!("{b}.ff"), D, false);
        attn(&mut t, &format!("{b}.audio_attn1"), AD, AD, HEADS, AD / HEADS);
        attn(&mut t, &format!("{b}.audio_attn2"), AD, AD, HEADS, AD / HEADS);
        ff(&mut t, &format!("{b}.audio_ff"), AD, false);
        attn(&mut t, &format!("{b}.audio_to_video_attn"), D, AD, HEADS, AD / HEADS);
        attn(&mut t, &format!("{b}.video_to_audio_attn"), AD, D, HEADS, AD / HEADS);
        t.uni(&format!("{b}.scale_shift_table"), &[9, D], 0.1, 0.0);
        t.uni(&format!("{b}.audio_scale_shift_table"), &[9, AD], 0.1, 0.0);
        t.uni(&format!("{b}.prompt_scale_shift_table"), &[2, D], 0.1, 0.0);
        t.uni(&format!("{b}.audio_prompt_scale_shift_table"), &[2, AD], 0.1, 0.0);
        t.uni(&format!("{b}.scale_shift_table_a2v_ca_video"), &[5, D], 0.1, 0.0);
        t.uni(&format!("{b}.scale_shift_table_a2v_ca_audio"), &[5, AD], 0.1, 0.0);
    }
    t.lin(&format!("{p}patchify_proj"), D, LAT, 1.0, true);
    t.lin(&format!("{p}audio_patchify_proj"), AD, LAT, 1.0, true);
    t.lin(&format!("{p}proj_out"), LAT, D, 1.0, true);
    t.lin(&format!("{p}audio_proj_out"), LAT, AD, 1.0, true);
    t.uni(&format!("{p}scale_shift_table"), &[2, D], 0.1, 0.0);
    t.uni(&format!("{p}audio_scale_shift_table"), &[2, AD], 0.1, 0.0);
    t.uni(&format!("{p}keyframes_abs_pos_embedding"), &[1, D], 0.1, 0.0);
    ada(&mut t, &format!("{p}adaln_single"), D, 9);
    ada(&mut t, &format!("{p}audio_adaln_single"), AD, 9);
    ada(&mut t, &format!("{p}prompt_adaln_single"), D, 2);
    ada(&mut t, &format!("{p}audio_prompt_adaln_single"), AD, 2);
    ada(&mut t, &format!("{p}av_ca_video_scale_shift_adaln_single"), D, 4);
    ada(&mut t, &format!("{p}av_ca_audio_scale_shift_adaln_single"), AD, 4);
    ada(&mut t, &format!("{p}av_ca_a2v_gate_adaln_single"), D, 1);
    ada(&mut t, &format!("{p}av_ca_v2a_gate_adaln_single"), AD, 1);
    for (name, dim) in [("video_embeddings_connector", D), ("audio_embeddings_connector", AD)] {
        let c = format!("{p}{name}");
        attn(&mut t, &format!("{c}.transformer_1d_blocks.0.attn1"), dim, dim, HEADS, dim / HEADS);
        ff(&mut t, &format!("{c}.transformer_1d_blocks.0.ff"), dim, true);
        t.uni(&format!("{c}.learnable_registers"), &[4, dim], 1.0, 0.0);
    }
    t.write(path)
}

/// A byte-level BPE `tokenizer.json`: every byte one token (ids 0 to 255), no merges, `<bos>`
/// at 256.
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
    vocab.insert("<bos>".into(), json!(BOS));
    let added = vec![
        json!({"id": BOS, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false,
                            "normalized": false, "special": true}),
    ];
    let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null, "pre_tokenizer": byte_level, "post_processor": byte_level, "decoder": byte_level,
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "",
            "fuse_unk": false, "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": [],
        },
    })
}

fn write_text_encoder(path: &Path) -> Result<()> {
    let mut t = Tensors::new(12);
    let (h, heads, hd, ghd) = (TEXT_H, 2, 8, 16);
    let gc = json!({"model_type": "gemma4_unified", "text_config": {
        "model_type": "gemma4_unified_text", "hidden_size": h, "intermediate_size": 32, "num_hidden_layers": TEXT_LAYERS,
        "num_attention_heads": heads, "num_key_value_heads": 1, "head_dim": hd, "global_head_dim": ghd,
        "num_global_key_value_heads": 1, "attention_k_eq_v": true, "vocab_size": VOCAB, "hidden_size_per_layer_input": 0,
        "sliding_window": 16, "final_logit_softcapping": 30.0, "rms_norm_eps": 1e-6, "bos_token_id": BOS,
        "layer_types": ["sliding_attention", "full_attention"], "tie_word_embeddings": true,
        "hidden_activation": "gelu_pytorch_tanh", "use_bidirectional_attention": "vision", "num_kv_shared_layers": 0,
        "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}
    }});
    t.meta("gemma_config", gc);
    t.uni("model.embed_tokens.weight", &[VOCAB, h], 1.0, 0.0);
    t.ones("model.norm.weight", h);
    for l in 0..TEXT_LAYERS {
        let lp = format!("model.layers.{l}.");
        let full = l == 1;
        let d = if full { ghd } else { hd };
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            t.ones(&format!("{lp}{n}.weight"), h);
        }
        t.uni(&format!("{lp}layer_scalar"), &[1], 0.0, 1.0);
        t.lin(&format!("{lp}self_attn.q_proj"), heads * d, h, 1.0, false);
        t.lin(&format!("{lp}self_attn.k_proj"), d, h, 1.0, false);
        if !full {
            t.lin(&format!("{lp}self_attn.v_proj"), d, h, 1.0, false);
        }
        t.lin(&format!("{lp}self_attn.o_proj"), h, heads * d, 0.5, false);
        t.ones(&format!("{lp}self_attn.q_norm.weight"), d);
        t.ones(&format!("{lp}self_attn.k_norm.weight"), d);
        t.lin(&format!("{lp}mlp.gate_proj"), 32, h, 1.0, false);
        t.lin(&format!("{lp}mlp.up_proj"), 32, h, 1.0, false);
        t.lin(&format!("{lp}mlp.down_proj"), h, 32, 0.5, false);
    }
    let flat = h * (TEXT_LAYERS + 1);
    t.lin("text_embedding_projection.video_aggregate_embed", D, flat, 1.0, true);
    t.lin("text_embedding_projection.audio_aggregate_embed", AD, flat, 1.0, true);
    t.bytes("tokenizer_json", tokenizer_json().to_string().as_bytes());
    t.write(path)
}

fn conv3(t: &mut Tensors, p: &str, o: usize, i: usize, gain: f32) {
    t.mat(&format!("{p}.weight"), &[o, i, 3, 3, 3], gain);
    t.uni(&format!("{p}.bias"), &[o], 0.02, 0.0);
}

fn write_video_vae(path: &Path) -> Result<()> {
    let mut t = Tensors::new(13);
    let c = 8;
    let blocks = json!([["compress_all", {"multiplier": 1}], ["compress_all", {"multiplier": 1}], ["compress_all", {"multiplier": 1}]]);
    let enc_blocks = json!([["compress_all_res", {"multiplier": 1}], ["res_x", {"num_layers": 1}], ["compress_all_res", {"multiplier": 1}], ["compress_all_res", {"multiplier": 1}]]);
    t.meta(
        "config",
        json!({"vae": {"_class_name": "CausalVideoAutoencoder", "latent_channels": LAT, "patch_size": 4, "norm_layer": "pixel_norm",
            "causal_decoder": false, "timestep_conditioning": false, "encoder_blocks": enc_blocks, "decoder_blocks": blocks,
            "decoder_base_channels": c, "latent_log_var": "uniform", "spatial_padding_mode": "zeros"}}),
    );
    t.uni("per_channel_statistics.mean-of-means", &[LAT], 0.1, 0.0);
    t.uni("per_channel_statistics.std-of-means", &[LAT], 0.1, 1.0);
    conv3(&mut t, "decoder.conv_in.conv", c, LAT, 1.0);
    for i in 0..3 {
        conv3(&mut t, &format!("decoder.up_blocks.{i}.conv.conv"), 8 * c, c, 1.0);
    }
    conv3(&mut t, "decoder.conv_out.conv", 48, c, 1.0);
    conv3(&mut t, "encoder.conv_in.conv", c, 48, 1.0);
    for i in [0, 2, 3] {
        conv3(&mut t, &format!("encoder.down_blocks.{i}.conv.conv"), c / 8, c, 1.0);
    }
    for n in ["conv1", "conv2"] {
        conv3(&mut t, &format!("encoder.down_blocks.1.res_blocks.0.{n}.conv"), c, c, 0.5);
    }
    conv3(&mut t, "encoder.conv_out.conv", LAT + 1, c, 1.0);
    t.write(path)
}

fn conv2(t: &mut Tensors, p: &str, o: usize, i: usize, k: usize) {
    t.mat(&format!("{p}.weight"), &[o, i, k, k], 1.0);
    t.uni(&format!("{p}.bias"), &[o], 0.02, 0.0);
}

fn lowpass() -> Vec<f32> {
    // A symmetric 12-tap low-pass (any normalized filter keeps the test finite).
    let v: Vec<f32> = (0..12).map(|i| 1.0 - ((i as f32 - 5.5) / 6.0).abs()).collect();
    let s: f32 = v.iter().sum();
    v.iter().map(|x| x / s).collect()
}

fn generator(t: &mut Tensors, p: &str, c0: usize, rates: &[usize], kernels: &[usize], cin: usize, bias_post: bool) {
    t.mat(&format!("{p}.conv_pre.weight"), &[c0, cin, 7], 1.0);
    t.uni(&format!("{p}.conv_pre.bias"), &[c0], 0.02, 0.0);
    let act = |t: &mut Tensors, ap: &str, c: usize| {
        t.uni(&format!("{ap}.act.alpha"), &[c], 0.1, 0.0);
        t.uni(&format!("{ap}.act.beta"), &[c], 0.1, 0.0);
        let f = lowpass();
        t.put(&format!("{ap}.upsample.filter"), &[1, 1, 12], &f);
        t.put(&format!("{ap}.downsample.lowpass.filter"), &[1, 1, 12], &f);
    };
    let mut c = c0;
    for (i, (&r, &k)) in rates.iter().zip(kernels).enumerate() {
        t.mat(&format!("{p}.ups.{i}.weight"), &[c, c / 2, k], 0.5);
        t.uni(&format!("{p}.ups.{i}.bias"), &[c / 2], 0.02, 0.0);
        let _ = r;
        c /= 2;
        for j in 0..3 {
            let rb = format!("{p}.resblocks.{}", i * 3 + j);
            let ks = [3, 5, 7][j];
            for d in 0..3 {
                act(t, &format!("{rb}.acts1.{d}"), c);
                act(t, &format!("{rb}.acts2.{d}"), c);
                t.mat(&format!("{rb}.convs1.{d}.weight"), &[c, c, ks], 0.3);
                t.uni(&format!("{rb}.convs1.{d}.bias"), &[c], 0.02, 0.0);
                t.mat(&format!("{rb}.convs2.{d}.weight"), &[c, c, ks], 0.3);
                t.uni(&format!("{rb}.convs2.{d}.bias"), &[c], 0.02, 0.0);
            }
        }
    }
    act(t, &format!("{p}.act_post"), c);
    t.mat(&format!("{p}.conv_post.weight"), &[2, c, 7], 0.5);
    if bias_post {
        t.uni(&format!("{p}.conv_post.bias"), &[2], 0.02, 0.0);
    }
}

fn write_audio_vae(path: &Path) -> Result<()> {
    let mut t = Tensors::new(14);
    let mels = 16;
    let (n_fft, hop) = (32, 8);
    let gen_cfg = |rates: &[usize], kernels: &[usize], c0: usize| {
        json!({"upsample_initial_channel": c0, "resblock": "AMP1", "upsample_rates": rates, "resblock_kernel_sizes": [3, 5, 7],
               "upsample_kernel_sizes": kernels, "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]], "stereo": true,
               "use_tanh_at_final": false, "activation": "snakebeta", "use_bias_at_final": false})
    };
    let mut bwe = gen_cfg(&[3, 8], &[7, 16], 16);
    let o = bwe.as_object_mut().unwrap();
    o.insert("apply_final_activation".into(), json!(false));
    o.insert("input_sampling_rate".into(), json!(16000));
    o.insert("output_sampling_rate".into(), json!(48000));
    o.insert("hop_length".into(), json!(hop));
    o.insert("n_fft".into(), json!(n_fft));
    o.insert("num_mels".into(), json!(mels));
    t.meta(
        "config",
        json!({"audio_vae": {"model": {"params": {"ddconfig": {"mel_bins": mels, "z_channels": 2, "ch": 8, "ch_mult": [1, 2, 4],
            "num_res_blocks": 1, "out_ch": 2, "norm_type": "pixel", "causality_axis": "height"}}}},
            "vocoder": {"vocoder": gen_cfg(&[5, 4, 8], &[11, 8, 16], 32), "bwe": bwe}}),
    );
    let p = "audio_vae.decoder";
    t.uni("audio_vae.per_channel_statistics.mean-of-means", &[LAT], 0.1, 0.0);
    t.uni("audio_vae.per_channel_statistics.std-of-means", &[LAT], 0.1, 1.0);
    conv2(&mut t, &format!("{p}.conv_in.conv"), 32, 2, 3);
    for b in ["block_1", "block_2"] {
        conv2(&mut t, &format!("{p}.mid.{b}.conv1.conv"), 32, 32, 3);
        conv2(&mut t, &format!("{p}.mid.{b}.conv2.conv"), 32, 32, 3);
    }
    let mut cin = 32;
    for level in (0..3).rev() {
        let co = 8 << level;
        for i in 0..2 {
            let bp = format!("{p}.up.{level}.block.{i}");
            conv2(&mut t, &format!("{bp}.conv1.conv"), co, cin, 3);
            conv2(&mut t, &format!("{bp}.conv2.conv"), co, co, 3);
            if cin != co {
                conv2(&mut t, &format!("{bp}.nin_shortcut.conv"), co, cin, 1);
            }
            cin = co;
        }
        if level != 0 {
            conv2(&mut t, &format!("{p}.up.{level}.upsample.conv.conv"), co, co, 3);
        }
    }
    conv2(&mut t, &format!("{p}.conv_out.conv"), 2, 8, 3);
    generator(&mut t, "vocoder.vocoder", 32, &[5, 4, 8], &[11, 8, 16], 2 * mels, false);
    generator(&mut t, "vocoder.bwe_generator", 16, &[3, 8], &[7, 16], 2 * mels, false);
    let nf = n_fft / 2 + 1;
    let mut basis = vec![0f32; 2 * nf * n_fft];
    for k in 0..nf {
        for j in 0..n_fft {
            let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * j as f32 / n_fft as f32).cos();
            let a = 2.0 * std::f32::consts::PI * (k * j) as f32 / n_fft as f32;
            basis[k * n_fft + j] = w * a.cos();
            basis[(nf + k) * n_fft + j] = -w * a.sin();
        }
    }
    t.put("vocoder.mel_stft.stft_fn.forward_basis", &[2 * nf, 1, n_fft], &basis);
    t.put("vocoder.mel_stft.stft_fn.inverse_basis", &[2 * nf, 1, n_fft], &basis);
    t.uni("vocoder.mel_stft.mel_basis", &[mels, nf], 0.05, 0.05);
    t.write(path)
}

fn write_upsampler(path: &Path) -> Result<()> {
    let mut t = Tensors::new(15);
    let mid = 32;
    t.meta(
        "config",
        json!({"_class_name": "LatentUpsampler", "in_channels": LAT, "mid_channels": mid, "num_blocks_per_stage": 1, "dims": 3,
               "spatial_upsample": true, "temporal_upsample": false, "spatial_scale": 2.0, "rational_resampler": false}),
    );
    conv3(&mut t, "initial_conv", mid, LAT, 1.0);
    t.ones("initial_norm.weight", mid);
    t.uni("initial_norm.bias", &[mid], 0.02, 0.0);
    for s in ["res_blocks.0", "post_upsample_res_blocks.0"] {
        conv3(&mut t, &format!("{s}.conv1"), mid, mid, 0.5);
        conv3(&mut t, &format!("{s}.conv2"), mid, mid, 0.5);
        for n in ["norm1", "norm2"] {
            t.ones(&format!("{s}.{n}.weight"), mid);
            t.uni(&format!("{s}.{n}.bias"), &[mid], 0.02, 0.0);
        }
    }
    t.mat("upsampler.0.weight", &[4 * mid, mid, 3, 3], 1.0);
    t.uni("upsampler.0.bias", &[4 * mid], 0.02, 0.0);
    conv3(&mut t, "final_conv", LAT, mid, 1.0);
    t.write(path)
}

fn write_duration_head(path: &Path) -> Result<()> {
    let mut t = Tensors::new(16);
    let h = 16;
    t.meta("config", json!({"duration_head": {}}));
    let p = "duration_head";
    t.lin(&format!("{p}.video_input_proj"), h, D, 1.0, true);
    t.lin(&format!("{p}.audio_input_proj"), h, AD, 1.0, true);
    t.uni(&format!("{p}.video_modality_emb"), &[h], 0.1, 0.0);
    t.uni(&format!("{p}.audio_modality_emb"), &[h], 0.1, 0.0);
    t.uni(&format!("{p}.attention_pooler.query_tokens"), &[1, h], 0.1, 0.0);
    t.mat(&format!("{p}.attention_pooler.cross_attn.in_proj_weight"), &[3 * h, h], 1.0);
    t.uni(&format!("{p}.attention_pooler.cross_attn.in_proj_bias"), &[3 * h], 0.02, 0.0);
    t.lin(&format!("{p}.attention_pooler.cross_attn.out_proj"), h, h, 1.0, true);
    t.lin(&format!("{p}.mlp_hidden"), h, h, 1.0, true);
    // A log-duration near ln 1.9: clips of 44 to 47 frames at 24 fps, snapped to 41.
    t.mat(&format!("{p}.mlp_out.weight"), &[1, h], 0.01);
    t.put(&format!("{p}.mlp_out.bias"), &[1], &[1.9f32.ln()]);
    t.write(path)
}

/// Writes the tiny release (see the module docs) into `dir`.
pub fn write_tiny_release(dir: &Path) -> Result<()> {
    use super::files;
    write_dit(&dir.join(files::DISTILLED_BF16))?;
    write_text_encoder(&dir.join(files::TEXT_BF16))?;
    write_video_vae(&dir.join(files::VIDEO_VAE))?;
    write_audio_vae(&dir.join(files::AUDIO_VAE))?;
    write_upsampler(&dir.join(files::SPATIAL_UPSAMPLER))?;
    write_duration_head(&dir.join(files::DURATION_HEAD))
}

#[cfg(test)]
mod tests {
    use super::super::{ImageCond, Params, Paths, Pipeline};
    use super::*;

    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let d = std::env::temp_dir().join(format!("oxinfer-ltx-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            Self(d)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn tiny_release() {
        let d = TempDir::new("release");
        write_tiny_release(&d.0).unwrap();
        assert!(super::super::is_ltx(&d.0));
        let paths = Paths::resolve(&d.0).unwrap();
        assert!(paths.duration_head.is_some() && paths.spatial_upsampler.is_some());
        let pipe = Pipeline::load(paths).unwrap();
        let mut progress = |_: &str, _: usize, _: usize| {};
        // Two stages, the frame count from the duration head (about two seconds).
        let mut p = Params::new("a test prompt");
        p.width = 64;
        p.height = 128;
        let v = pipe.generate(&p, &mut progress).unwrap();
        assert_eq!((v.width, v.height), (64, 128));
        assert_eq!(v.frames.len(), 41, "2 s at 24 fps, snapped to 8 k + 1");
        let a = v.audio.as_ref().expect("a soundtrack");
        assert_eq!(a.sample_rate, 48000);
        assert_eq!(a.channels.len(), 2);
        // 43 audio latents: 169 mel frames of 160 samples at 16 kHz, times 3 (a little shorter
        // than the video, as in the reference).
        let na = super::super::audio_latents(41, 24.0);
        assert_eq!(a.len(), ((4 * na - 3) * 160 * 3).min((41.0f64 / 24.0 * 48000.0).round() as usize));
        assert!(a.channels.iter().flatten().all(|s| s.is_finite() && s.abs() <= 1.0));
        // One stage with a first frame, a fixed frame count and the same seed twice: the same
        // video.
        let img = crate::sd::Image {
            width: 40,
            height: 30,
            rgb: (0..40 * 30 * 3).map(|i| (i * 7 % 256) as u8).collect(),
        };
        let mut p = Params::new("another prompt");
        p.width = 64;
        p.height = 32;
        p.frames = Some(12);
        p.two_stage = false;
        p.images.push(ImageCond {
            image: img,
            frame: 0,
            strength: 1.0,
        });
        let v1 = pipe.generate(&p, &mut progress).unwrap();
        assert_eq!(v1.frames.len(), 9);
        let v2 = pipe.generate(&p, &mut progress).unwrap();
        assert_eq!(v1.frames, v2.frames);
        // A size the two-stage pipeline cannot take.
        let mut p = Params::new("x");
        p.width = 96;
        assert!(pipe.generate(&p, &mut progress).is_err());
        drop(pipe);

        // A LoRA on two layers changes the video; one on a module the transformer lacks fails.
        let lora = d.0.join("lora.safetensors");
        let mut t = Tensors::new(21);
        t.meta("lora_alpha", json!(4));
        for m in ["transformer_blocks.0.attn1.to_q", "transformer_blocks.1.ff.net.2"] {
            let (o, i) = if m.ends_with("to_q") { (D, D) } else { (D, 4 * D) };
            t.uni(&format!("diffusion_model.{m}.lora_A.weight"), &[4, i], 0.5, 0.0);
            t.uni(&format!("diffusion_model.{m}.lora_B.weight"), &[o, 4], 0.5, 0.0);
        }
        t.write(&lora).unwrap();
        let mut paths = Paths::resolve(&d.0).unwrap();
        paths.loras.push((lora.clone(), 1.0));
        let pipe = Pipeline::load(paths).unwrap();
        let p = {
            let mut p = Params::new("another prompt");
            (p.width, p.height, p.frames, p.two_stage) = (64, 32, Some(9), false);
            p
        };
        let base = {
            let paths = Paths::resolve(&d.0).unwrap();
            Pipeline::load(paths).unwrap().generate(&p, &mut progress).unwrap()
        };
        let with = pipe.generate(&p, &mut progress).unwrap();
        assert_ne!(base.frames, with.frames);
        let mut t = Tensors::new(22);
        t.uni("diffusion_model.no_such_module.lora_A.weight", &[4, D], 0.5, 0.0);
        t.uni("diffusion_model.no_such_module.lora_B.weight", &[D, 4], 0.5, 0.0);
        let bad = d.0.join("bad.safetensors");
        t.write(&bad).unwrap();
        let mut paths = Paths::resolve(&d.0).unwrap();
        paths.loras.push((bad, 1.0));
        assert!(Pipeline::load(paths).is_err());
    }

    #[test]
    fn guided_dev() {
        use super::super::{Guidance, GuidanceMode};
        let d = TempDir::new("guided");
        write_tiny_release(&d.0).unwrap();
        // The tiny transformer under a dev name: guided by default.
        let dev = d.0.join(super::super::files::DEV_BF16);
        std::fs::copy(d.0.join(super::super::files::DISTILLED_BF16), &dev).unwrap();
        let mut paths = Paths::resolve(&d.0).unwrap();
        paths.transformer = dev;
        let pipe = Pipeline::load(paths.clone()).unwrap();
        assert!(pipe.dev);
        let mut progress = |_: &str, _: usize, _: usize| {};
        let mut p = Params::new("a test prompt");
        (p.width, p.height, p.frames) = (64, 64, Some(9));
        p.guidance.steps = 4;
        assert!(pipe.guidance_of(&p).is_some());
        // No distilled LoRA: one stage at the full size.
        let guided = pipe.generate(&p, &mut progress).unwrap();
        assert_eq!((guided.width, guided.height, guided.frames.len()), (64, 64, 9));
        // The same schedule without the guidance passes differs.
        p.guidance = Guidance {
            steps: 4,
            ..Guidance::off()
        };
        p.guidance_mode = GuidanceMode::On;
        let plain = pipe.generate(&p, &mut progress).unwrap();
        assert_ne!(guided.frames, plain.frames);
        drop(pipe);
        // With a LoRA as the distilled one: two stages, the LoRA off in the first.
        let lora = d.0.join("distilled.safetensors");
        let mut t = Tensors::new(23);
        t.meta("lora_alpha", json!(4));
        t.uni("diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight", &[4, D], 0.5, 0.0);
        t.uni("diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight", &[D, 4], 0.5, 0.0);
        t.write(&lora).unwrap();
        paths.distilled_lora = Some(lora);
        let pipe = Pipeline::load(paths).unwrap();
        let mut p = Params::new("a test prompt");
        (p.width, p.height, p.frames) = (64, 64, Some(9));
        p.guidance.steps = 3;
        let v = pipe.generate(&p, &mut progress).unwrap();
        assert_eq!((v.width, v.height), (64, 64));
        // One stage with the distilled LoRA: the distilled schedule, no guidance.
        p.two_stage = false;
        assert!(pipe.guidance_of(&p).is_none());
    }

    #[test]
    fn frames_and_positions() {
        assert_eq!(super::super::snap_frames(121), 121);
        assert_eq!(super::super::snap_frames(128), 121);
        assert_eq!(super::super::snap_frames(1), 1);
        assert_eq!(super::super::latent_index(0), 0);
        assert_eq!(super::super::latent_index(1), 1);
        assert_eq!(super::super::latent_index(8), 1);
        assert_eq!(super::super::latent_index(9), 2);
        assert_eq!(super::super::audio_latents(121, 24.0), 126);
        assert_eq!(super::super::duration::frames_for(5.0, 24.0), 113);
        assert_eq!(super::super::duration::frames_for(0.1, 24.0), 25);
        assert_eq!(super::super::duration::frames_for(100.0, 24.0), 473);
        // The first latent frame covers the first frame alone: time (0 + 1/24) / 2.
        let p = super::super::video_positions(2, 1, 1, 24.0);
        assert!((p[0] - 0.5 / 24.0).abs() < 1e-7 && (p[3] - 5.0 / 24.0).abs() < 1e-6);
        assert_eq!((p[1], p[2]), (16.0, 16.0));
        let a = super::super::audio_positions(2);
        assert!((a[0] - 0.005).abs() < 1e-7 && (a[1] - 0.03).abs() < 1e-7);
    }
}
