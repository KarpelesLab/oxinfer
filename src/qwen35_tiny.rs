//! Tiny random Qwen3.5 checkpoints (bf16 noise from a fixed-seed generator: deterministic,
//! meaningless) for tests that run the model, its MTP layer, the engine and the server without a
//! real one ([`write`]): three Gated DeltaNet layers then a gated full-attention layer of width
//! 256, with the head sizes the GPU decode kernels take (128), dense or MoE (4 experts, 2 per
//! token, a gated shared expert), with or without an MTP layer, beside a byte-level
//! `tokenizer.json` (one token per byte) carrying the chat format's tokens ([`SPECIAL`] from
//! [`SPECIAL_BASE`]).

use anyhow::{Context, Result};
use serde_json::json;
use std::path::Path;

/// The chat format's tokens, from id [`SPECIAL_BASE`].
pub const SPECIAL: [&str; 7] = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
];
/// The id of the first special token.
pub const SPECIAL_BASE: u32 = 256;
/// The vocabulary: the bytes and [`SPECIAL`].
pub const VOCAB: usize = 256 + SPECIAL.len();
/// The hidden size.
pub const H: usize = 256;

/// Tensors being written: name, shape, values (stored as bf16).
struct Tensors {
    list: Vec<(String, Vec<usize>, Vec<f32>)>,
    seed: u64,
}

impl Tensors {
    /// `scale` times uniform noise in [-0.5, 0.5), plus `bias`.
    fn add(&mut self, name: String, shape: &[usize], scale: f32, bias: f32) {
        let n: usize = shape.iter().product();
        let v = (0..n)
            .map(|_| {
                self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                bias + scale * (((self.seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
            })
            .collect();
        self.list.push((name, shape.to_vec(), v));
    }

    fn write(self, dir: &Path) -> Result<()> {
        self.write_as(&dir.join("model.safetensors"))
    }

    fn write_as(self, path: &Path) -> Result<()> {
        let mut head = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, vals) in self.list {
            let start = data.len();
            for v in vals {
                data.extend_from_slice(&((v.to_bits() + 0x8000) >> 16).to_le_bytes()[..2]);
            }
            head.insert(name, json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, data.len()]}));
        }
        let mut head = serde_json::Value::Object(head).to_string();
        while !head.len().is_multiple_of(8) {
            head.push(' ');
        }
        let mut file = (head.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(head.as_bytes());
        file.extend(data);
        std::fs::write(path, file).with_context(|| path.display().to_string())
    }
}

/// Writes the checkpoint folder `dir` (`config.json`, `model.safetensors`, `tokenizer.json`):
/// a MoE model with `moe`, with an MTP layer with `mtp`.
pub fn write(dir: &Path, moe: bool, mtp: bool) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
    let (heads, kv, hd, inter) = (2usize, 1usize, 128usize, 512usize);
    let (nk, nv, dk, dv, kc) = (2usize, 2usize, 128usize, 128usize, 4usize);
    let (ne, mi) = (4usize, 128usize);
    let types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"];
    let im_end = SPECIAL_BASE + 2;
    let mut config = json!({
        "model_type": if moe { "qwen3_5_moe_text" } else { "qwen3_5_text" },
        "hidden_size": H, "intermediate_size": inter, "num_hidden_layers": types.len(),
        "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": hd, "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6, "layer_types": types, "full_attention_interval": 4, "attn_output_gate": true,
        "linear_conv_kernel_dim": kc, "linear_key_head_dim": dk, "linear_value_head_dim": dv,
        "linear_num_key_heads": nk, "linear_num_value_heads": nv, "tie_word_embeddings": true,
        "eos_token_id": im_end,
        "rope_parameters": { "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25 },
    });
    if moe {
        config["num_experts"] = json!(ne);
        config["num_experts_per_tok"] = json!(2);
        config["moe_intermediate_size"] = json!(mi);
        config["shared_expert_intermediate_size"] = json!(mi);
    }
    if mtp {
        config["mtp_num_hidden_layers"] = json!(1);
    }
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&config)?)?;
    let tok = crate::inkling::tiny::byte_tokenizer(&SPECIAL, SPECIAL_BASE);
    std::fs::write(dir.join("tokenizer.json"), serde_json::to_vec(&tok)?)?;

    let mut t = Tensors { list: Vec::new(), seed: 7 };
    let si = 2.0 / (H as f32).sqrt();
    t.add("model.embed_tokens.weight".into(), &[VOCAB, H], 0.05, 0.0);
    // Zero-centred norms (1 + w).
    t.add("model.norm.weight".into(), &[H], 0.4, 0.0);
    let attn = |t: &mut Tensors, p: &str| {
        t.add(format!("{p}self_attn.q_proj.weight"), &[2 * heads * hd, H], si, 0.0);
        t.add(format!("{p}self_attn.k_proj.weight"), &[kv * hd, H], si, 0.0);
        t.add(format!("{p}self_attn.v_proj.weight"), &[kv * hd, H], si, 0.0);
        t.add(format!("{p}self_attn.o_proj.weight"), &[H, heads * hd], 1.0, 0.0);
        t.add(format!("{p}self_attn.q_norm.weight"), &[hd], 0.4, 0.0);
        t.add(format!("{p}self_attn.k_norm.weight"), &[hd], 0.4, 0.0);
    };
    let ffn = |t: &mut Tensors, p: &str| {
        let mlp = |t: &mut Tensors, p: &str, inter: usize| {
            for (n, o, i) in [("gate", inter, H), ("up", inter, H), ("down", H, inter)] {
                t.add(format!("{p}{n}_proj.weight"), &[o, i], si, 0.0);
            }
        };
        if moe {
            t.add(format!("{p}mlp.gate.weight"), &[ne, H], 1.0, 0.0);
            for e in 0..ne {
                mlp(t, &format!("{p}mlp.experts.{e}."), mi);
            }
            mlp(t, &format!("{p}mlp.shared_expert."), mi);
            t.add(format!("{p}mlp.shared_expert_gate.weight"), &[1, H], 1.0, 0.0);
        } else {
            mlp(t, &format!("{p}mlp."), inter);
        }
        for n in ["input_layernorm", "post_attention_layernorm"] {
            t.add(format!("{p}{n}.weight"), &[H], 0.4, 0.0);
        }
    };
    let conv_dim = 2 * nk * dk + nv * dv;
    for (l, ty) in types.iter().enumerate() {
        let p = format!("model.layers.{l}.");
        if *ty == "full_attention" {
            attn(&mut t, &p);
        } else {
            let g = format!("{p}linear_attn.");
            t.add(format!("{g}in_proj_qkv.weight"), &[conv_dim, H], si, 0.0);
            t.add(format!("{g}in_proj_z.weight"), &[nv * dv, H], si, 0.0);
            t.add(format!("{g}in_proj_b.weight"), &[nv, H], si, 0.0);
            t.add(format!("{g}in_proj_a.weight"), &[nv, H], si, 0.0);
            t.add(format!("{g}conv1d.weight"), &[conv_dim, 1, kc], 1.0, 0.0);
            t.add(format!("{g}A_log"), &[nv], 1.0, 0.0);
            t.add(format!("{g}dt_bias"), &[nv], 1.0, 0.0);
            t.add(format!("{g}norm.weight"), &[dv], 0.4, 1.0);
            t.add(format!("{g}out_proj.weight"), &[H, nv * dv], 1.0, 0.0);
        }
        ffn(&mut t, &p);
    }
    if mtp {
        t.add("mtp.fc.weight".into(), &[H, 2 * H], 2.0 / (2.0 * H as f32).sqrt(), 0.0);
        t.add("mtp.pre_fc_norm_embedding.weight".into(), &[H], 0.4, 0.0);
        t.add("mtp.pre_fc_norm_hidden.weight".into(), &[H], 0.4, 0.0);
        t.add("mtp.norm.weight".into(), &[H], 0.4, 0.0);
        attn(&mut t, "mtp.layers.0.");
        ffn(&mut t, "mtp.layers.0.");
    }
    t.write(dir)
}

/// Writes a random PEFT LoRA adapter for [`write`]'s dense models into the folder `dir`
/// (`adapter_config.json`, `adapter_model.safetensors`): rank 4, alpha 8, on the MLP projections
/// of every layer, `A` and `B` noise of width `scale` from `seed`.
pub fn write_adapter(dir: &Path, seed: u64, scale: f32) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
    let (r, inter, layers) = (4usize, 512usize, 4usize);
    let mut t = Tensors { list: Vec::new(), seed };
    for l in 0..layers {
        for (m, n_out, n_in) in [("gate_proj", inter, H), ("up_proj", inter, H), ("down_proj", H, inter)] {
            let p = format!("base_model.model.model.layers.{l}.mlp.{m}");
            t.add(format!("{p}.lora_A.weight"), &[r, n_in], scale, 0.0);
            t.add(format!("{p}.lora_B.weight"), &[n_out, r], scale, 0.0);
        }
    }
    t.write_as(&dir.join("adapter_model.safetensors"))?;
    let config = json!({
        "peft_type": "LORA", "r": r, "lora_alpha": 2 * r,
        "target_modules": ["gate_proj", "up_proj", "down_proj"],
    });
    let path = dir.join("adapter_config.json");
    std::fs::write(&path, config.to_string()).with_context(|| path.display().to_string())
}

/// Writes a tiny random DSpark draft for [`write`]'s models into the folder `dir` (`config.json`
/// and bf16 safetensors in the Hugging Face layout): one Qwen3 GQA layer half as wide as the
/// target, with its own embeddings and LM head, blocks of 3, reading the streams entering the
/// target's layers 0 and 1, a Markov head and a confidence head.
pub fn write_draft(dir: &Path, seed: u64) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
    let (nh, nkv, hd, inter, dh) = (2usize, 1usize, 16usize, 32usize, H / 2);
    let config = json!({
        "architectures": ["Qwen3DSparkModel"], "model_type": "qwen3", "hidden_size": dh,
        "num_hidden_layers": 1, "num_attention_heads": nh, "num_key_value_heads": nkv, "head_dim": hd,
        "intermediate_size": inter, "rms_norm_eps": 1e-6, "vocab_size": VOCAB, "block_size": 3,
        "target_layer_ids": [0, 1], "mask_token_id": SPECIAL_BASE, "markov_rank": 16,
        "markov_head_type": "vanilla", "layer_types": ["full_attention"],
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    });
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&config)?)?;
    let mut t = Tensors { list: Vec::new(), seed };
    let s = |n: usize| 2.0 / (n as f32).sqrt();
    t.add("fc.weight".into(), &[dh, 2 * H], s(2 * H), 0.0);
    t.add("hidden_norm.weight".into(), &[dh], 0.4, 1.0);
    t.add("norm.weight".into(), &[dh], 0.4, 1.0);
    t.add("embed_tokens.weight".into(), &[VOCAB, dh], 0.6, 0.0);
    t.add("lm_head.weight".into(), &[VOCAB, dh], s(dh), 0.0);
    t.add("markov_head.markov_w1.weight".into(), &[VOCAB, 16], 0.6, 0.0);
    t.add("markov_head.markov_w2.weight".into(), &[VOCAB, 16], 0.6, 0.0);
    t.add("confidence_head.proj.weight".into(), &[1, dh + 16], 0.1, 0.0);
    t.add("confidence_head.proj.bias".into(), &[1], 0.0, 1.0);
    for (n, shape) in [
        ("input_layernorm.weight", vec![dh]),
        ("post_attention_layernorm.weight", vec![dh]),
        ("self_attn.q_norm.weight", vec![hd]),
        ("self_attn.k_norm.weight", vec![hd]),
        ("self_attn.q_proj.weight", vec![nh * hd, dh]),
        ("self_attn.k_proj.weight", vec![nkv * hd, dh]),
        ("self_attn.v_proj.weight", vec![nkv * hd, dh]),
        ("self_attn.o_proj.weight", vec![dh, nh * hd]),
        ("mlp.gate_proj.weight", vec![inter, dh]),
        ("mlp.up_proj.weight", vec![inter, dh]),
        ("mlp.down_proj.weight", vec![dh, inter]),
    ] {
        if n.contains("norm") {
            t.add(format!("layers.0.{n}"), &shape, 0.4, 1.0);
        } else {
            t.add(format!("layers.0.{n}"), &shape, s(shape[1]), 0.0);
        }
    }
    t.write(dir)
}

#[cfg(test)]
mod tests {
    use crate::config::QwenConfig;
    use crate::model::Quant;
    use crate::qwen::{QwenModel, QwenSession};
    use crate::qwen_mtp::{MtpDecoder, MtpOptions};
    use crate::safetensors::ModelFiles;
    use crate::sampler::{Sampler, argmax};

    fn load(tag: &str, moe: bool, mtp: bool) -> (&'static QwenModel<'static>, &'static ModelFiles) {
        let dir = std::env::temp_dir().join(format!("oxinfer-q35-tiny-{tag}-{}", std::process::id()));
        super::write(&dir, moe, mtp).unwrap();
        let files: &'static ModelFiles = Box::leak(Box::new(ModelFiles::open(&dir, true).unwrap()));
        let cfg = QwenConfig::load(&dir).unwrap();
        let m = Box::leak(Box::new(QwenModel::load(cfg, files, Quant::Bf16).unwrap()));
        (m, files)
    }

    /// Test tokens. (In lib tests `kernels::set_decode_max_tokens` is per thread, so the passes
    /// of 8 to 15 tokens of a test keep one kernel choice whatever other tests set.)
    fn toks(n: usize, seed: u32) -> Vec<u32> {
        (0..n as u32).map(|i| (i * 37 + seed * 11 + 5) % super::SPECIAL_BASE).collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Greedy one-token steps from `first`: the logits after each.
    fn steps(m: &QwenModel, s: &mut QwenSession, first: u32, n: usize) -> Vec<Vec<f32>> {
        let mut t = first;
        (0..n)
            .map(|_| {
                let l = m.forward(s, &[t]);
                t = argmax(&l);
                l
            })
            .collect()
    }

    /// A session gone back to a checkpoint (after a reply ran past it) gives the logits of a
    /// fresh session that ran the same passes, bit for bit: dense and MoE, to the prompt but its
    /// last token and to the shared prefix; checkpoints past the position gone back to go.
    #[test]
    fn checkpoint_cut_back_matches_fresh() {
        for moe in [false, true] {
            let (m, _) = load(&format!("cp{moe}"), moe, false);
            let (a, b, c) = (toks(40, 1), toks(17, 2), toks(7, 3));
            let mut s = m.new_session();
            assert!(s.checkpointable());
            m.forward(&mut s, &a[..20]);
            s.checkpoint(2);
            m.forward(&mut s, &a[20..39]);
            s.checkpoint(2);
            assert_eq!(s.checkpoint_positions(), [20, 39]);
            let l = m.forward(&mut s, &a[39..]);
            steps(m, &mut s, argmax(&l), 30);
            assert_eq!(s.reachable(45), 39);
            assert_eq!(s.reachable(30), 20);
            assert_eq!(s.reachable(10), 0);
            assert!(s.cut_to(30).is_err(), "no checkpoint there");
            // To the previous prompt but its last token.
            s.cut_to(39).unwrap();
            assert_eq!(s.pos, 39);
            let got = m.forward(&mut s, &b);
            let got_steps = steps(m, &mut s, argmax(&got), 6);
            let mut f = m.new_session();
            m.forward(&mut f, &a[..20]);
            m.forward(&mut f, &a[20..39]);
            let want = m.forward(&mut f, &b);
            assert_eq!(bits(&got), bits(&want), "moe {moe}: after the cut back");
            for (g, w) in got_steps.iter().zip(steps(m, &mut f, argmax(&want), 6)) {
                assert_eq!(bits(g), bits(&w), "moe {moe}: steps after the cut back");
            }
            // To the shared prefix: the later checkpoint goes.
            s.cut_to(20).unwrap();
            assert_eq!(s.checkpoint_positions(), [20]);
            let got = m.forward(&mut s, &c);
            let mut f = m.new_session();
            m.forward(&mut f, &a[..20]);
            assert_eq!(bits(&got), bits(&m.forward(&mut f, &c)), "moe {moe}: to the prefix");
            // At most `max`: the lowest and the newest stay.
            s.checkpoint(2);
            m.forward(&mut s, &b);
            s.checkpoint(2);
            assert_eq!(s.checkpoint_positions(), [20, 44]);
            // Off.
            let mut s = m.new_session();
            m.forward(&mut s, &a);
            s.checkpoint(0);
            assert!(s.checkpoint_positions().is_empty());
        }
    }

    /// The MTP decoder goes back to a checkpoint with its layer's cache and pending rows: the
    /// logits and the greedy speculative reply are those of a fresh decoder's same passes.
    #[test]
    fn mtp_checkpoint_cut_back_matches_fresh() {
        for moe in [false, true] {
            let (m, files) = load(&format!("mtp{moe}"), moe, true);
            let mtp = Box::leak(Box::new(m.load_mtp(files).unwrap()));
            let opts = MtpOptions {
                draft_tokens: 3,
                stop: Vec::new(),
                ..MtpOptions::default()
            };
            let (a, b) = (toks(40, 4), toks(17, 5));
            let reply = |d: &mut MtpDecoder, l: &[f32], n: usize| {
                let mut g = Sampler::new(0.0, 0, 1.0, 1);
                let mut out = vec![argmax(l)];
                d.push(out[0]);
                while out.len() < n {
                    out.extend(d.step(&mut g, n - out.len()));
                }
                out
            };
            let mut d = MtpDecoder::new(m, mtp, opts.clone());
            d.prefill(&a[..20], 512);
            d.checkpoint(2);
            d.prefill(&a[20..39], 512);
            d.checkpoint(2);
            let l = d.prefill(&a[39..], 512);
            reply(&mut d, &l, 30);
            assert!(d.has_checkpoint(39) && d.has_checkpoint(20));
            assert_eq!(d.reachable(45), 39);
            d.cut_to(39).unwrap();
            let got = d.prefill(&b, 512);
            let got_reply = reply(&mut d, &got, 16);
            let mut f = MtpDecoder::new(m, mtp, opts.clone());
            f.prefill(&a[..20], 512);
            f.prefill(&a[20..39], 512);
            let want = f.prefill(&b, 512);
            assert_eq!(bits(&got), bits(&want), "moe {moe}");
            assert_eq!(got_reply, reply(&mut f, &want, 16), "moe {moe}");
            d.cut_to(20).unwrap();
            assert!(!d.has_checkpoint(39));
            let got = d.prefill(&b, 512);
            let mut f = MtpDecoder::new(m, mtp, opts.clone());
            f.prefill(&a[..20], 512);
            assert_eq!(bits(&got), bits(&f.prefill(&b, 512)), "moe {moe}: to the prefix");
        }
    }
}
