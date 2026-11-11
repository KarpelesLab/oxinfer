//! Decode steps of several independent sequences in one pass (a submodule of `crate::qwen`):
//! the next token of each sequence is one column of the layer matmuls, as the tokens of a
//! multi-token pass are, while every sequence keeps its own session (KV caches, Gated DeltaNet
//! states, position). The weights are then read once per step for all the sequences, which is
//! what a memory-bound CPU decode pays for.
//!
//! The decode kernels give each column the bits of a one-token pass (int8 dot products up to
//! [`crate::kernels::decode_max_tokens`] columns, bf16 up to 7; see `kernels.rs`), and the
//! attention of each sequence is the one-token attention over its own cache, so a batched step
//! equals the sequences' own one-token CPU steps bit for bit.

use super::*;

impl QwenModel<'_> {
    /// Most sequences [`Self::forward_hidden_batch`] runs as columns of one pass (wider batches
    /// are split): what the batch-invariant decode kernels take, so that each column keeps the
    /// bits of a one-token pass.
    pub fn batch_width(&self) -> usize {
        let bf16 = self.layers.iter().any(|l| match &l.mixer {
            Mixer::Attn(a) => matches!(a.q.as_ref(), crate::gpu::WeightRef::Bf16(_)),
            Mixer::Gdn(g) => matches!(g.qkv.as_ref(), crate::gpu::WeightRef::Bf16(_)),
            // Never batched (see `forward_hidden_batch`).
            Mixer::Lightning(_) => false,
        });
        if bf16 { 7 } else { decode_max_tokens() }
    }

    /// One decode step of `sess.len()` independent sequences: `tokens[i]` is appended to
    /// `sess[i]` at its own position. Returns the final-normed hidden states `[sess.len(),
    /// hidden]`, each row equal (bit for bit on the CPU) to
    /// `forward_hidden_mm(sess[i], &[tokens[i]], &[])` run on its own.
    ///
    /// The sequences share the layer matmuls on the CPU kernels (the device plans of
    /// `crate::gpu::decode` decode one sequence). Sessions with interventions or captures, the
    /// Qwen4-Exp models and a batch of one take the one-token path, one sequence after another.
    pub fn forward_hidden_batch(&self, sess: &mut [&mut QwenSession], tokens: &[u32]) -> Vec<f32> {
        let b = tokens.len();
        assert!(b > 0 && sess.len() == b, "one token per session");
        let h = self.cfg.hidden_size;
        let plain = sess.iter().all(|s| !s.iv.active() && s.capture.is_empty());
        // The MiniMax variants of this path (post-norm residuals, lightning attention, the
        // block-sparse indexer, norms over all heads) are not written for a batch: one by one.
        let minimax = self.cfg.postnorm
            || self.cfg.qk_norm_full()
            || self.cfg.use_gemma_norm
            || self.cfg.attn_type_list.is_some()
            || self.layers.iter().any(|l| match &l.mixer {
                Mixer::Lightning(_) => true,
                Mixer::Attn(at) => at.indexer.is_some(),
                Mixer::Gdn(_) => false,
            });
        if b == 1 || self.exp.is_some() || !plain || minimax {
            let mut out = Vec::with_capacity(b * h);
            for (s, &t) in sess.iter_mut().zip(tokens) {
                out.extend(self.forward_hidden_mm(s, &[t], &[]));
            }
            return out;
        }
        let width = self.batch_width().max(2);
        if b > width {
            let mut out = Vec::with_capacity(b * h);
            for (ss, ts) in sess.chunks_mut(width).zip(tokens.chunks(width)) {
                out.extend(self.forward_hidden_batch(ss, ts));
            }
            return out;
        }
        let cfg = &self.cfg;
        let eps = cfg.rms_norm_eps as f32;

        let timer = prof::start(Cat::Embed);
        if let Some(s) = &self.stream {
            let rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| self.embed.row_ranges(tk as usize, h)).collect();
            s.need_ranges(&rows);
        }
        let mut x = vec![0f32; b * h];
        for_chunks(&mut x, h, true, |i, row| self.embed.row(tokens[i] as usize, row));
        // RoPE row `i`: the position of sequence `i`.
        let half = self.inv_freq.len();
        let mut cos = Vec::with_capacity(b * half);
        let mut sin = Vec::with_capacity(b * half);
        for s in sess.iter_mut() {
            s.gpu_rec = None;
            let (c, sn) = match &s.positions {
                Some(p) if p.len() > s.pos => self.rope_cs_at(&p[s.pos..s.pos + 1]),
                _ => self.rope_cs(&s.spans, s.pos, 1),
            };
            cos.extend(c);
            sin.extend(sn);
        }
        drop(timer);

        let mut hn = vec![0f32; b * h];
        let mut y = vec![0f32; b * h];
        for (l, layer) in self.layers.iter().enumerate() {
            let _unit = crate::stream::enter(&self.stream, l);
            rmsnorm_f32w_rows(&x, h, &layer.ln_in, eps, &mut hn);
            match &layer.mixer {
                // The recurrent state is per sequence: one-token passes.
                Mixer::Gdn(gd) => {
                    for (i, s) in sess.iter_mut().enumerate() {
                        s.gpu_states_kept = false;
                        let st = s.gdn[l].as_mut().unwrap();
                        self.gdn_forward(gd, st, None, &hn[i * h..(i + 1) * h], 1, &mut y[i * h..(i + 1) * h]);
                    }
                }
                Mixer::Attn(at) => {
                    let mut caches: Vec<&mut LayerCache> = sess.iter_mut().map(|s| s.kv[l].as_mut().unwrap()).collect();
                    self.attn_forward_caches(at, &mut caches, &hn, b, 0, &cos, &sin, None, &mut y);
                }
                Mixer::Lightning(_) => unreachable!("lightning attention is not batched"),
            }
            add_inplace(&mut x, &y);
            rmsnorm_f32w_rows(&x, h, &layer.ln_post, eps, &mut hn);
            let need = |ids: &[usize]| {
                if let Some(s) = &self.stream {
                    s.need_parts(l, ids)
                }
            };
            layer
                .ffn
                .forward(&hn, b, h, &mut y, self.stream.as_ref().map(|_| &need as &dyn Fn(&[usize])), None);
            add_inplace(&mut x, &y);
            if let (Some(s), Some(next)) = (&self.stream, self.layers.get(l + 1))
                && let Ffn::Moe(m) = &next.ffn
                && s.options().predict_experts
            {
                s.prefetch_parts(l + 1, &m.predict(&x, b, h, &next.ln_post, eps));
            }
        }
        rmsnorm_f32w_rows(&x.clone(), h, &self.final_norm, eps, &mut x);
        for s in sess.iter_mut() {
            s.rec_span = None;
            s.pos += 1;
        }
        x
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::safetensors::ModelFiles;
    use std::path::{Path, PathBuf};

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    /// Writes `tensors` (`name`, shape, values) as one bf16 safetensors file.
    pub(crate) fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, vals) in tensors {
            assert_eq!(vals.len(), shape.iter().product::<usize>(), "{name}");
            let start = data.len();
            for &v in vals {
                data.extend_from_slice(&half::bf16::from_f32(v).to_bits().to_le_bytes());
            }
            header.insert(
                name.clone(),
                serde_json::json!({ "dtype": "BF16", "shape": shape, "data_offsets": [start, data.len()] }),
            );
        }
        let mut json = serde_json::to_vec(&header).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut out = (json.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&json);
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    /// A random Llama checkpoint (tied output layer) in a new temporary directory; `boost`
    /// scales rows of the token table.
    pub(crate) fn tiny_llama(tag: &str, vocab: usize, hidden: usize, layers: usize, seed: u64, boost: &[(usize, f32)]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oxinfer-tiny-llama-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (heads, kv, inter) = (4usize, 2usize, 2 * hidden);
        let hd = hidden / heads;
        let cfg = serde_json::json!({
            "model_type": "llama", "hidden_size": hidden, "intermediate_size": inter, "num_hidden_layers": layers,
            "num_attention_heads": heads, "num_key_value_heads": kv, "vocab_size": vocab, "rms_norm_eps": 1e-5,
            "tie_word_embeddings": true,
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
        let mut s = seed;
        let mut rnd = |n: usize, scale: f32| -> Vec<f32> { (0..n).map(|_| lcg(&mut s) * scale).collect() };
        let mut embed = rnd(vocab * hidden, 2.0);
        for &(row, scale) in boost {
            embed[row * hidden..(row + 1) * hidden].iter_mut().for_each(|v| *v *= scale);
        }
        let mut t = vec![("model.embed_tokens.weight".to_string(), vec![vocab, hidden], embed)];
        t.push((
            "model.norm.weight".into(),
            vec![hidden],
            rnd(hidden, 0.2).iter().map(|v| 1.0 + v).collect(),
        ));
        for l in 0..layers {
            let p = format!("model.layers.{l}.");
            for (n, o, i) in [
                ("q", heads * hd, hidden),
                ("k", kv * hd, hidden),
                ("v", kv * hd, hidden),
                ("o", hidden, heads * hd),
            ] {
                t.push((format!("{p}self_attn.{n}_proj.weight"), vec![o, i], rnd(o * i, 0.5)));
            }
            for (n, o, i) in [("gate", inter, hidden), ("up", inter, hidden), ("down", hidden, inter)] {
                t.push((format!("{p}mlp.{n}_proj.weight"), vec![o, i], rnd(o * i, 0.4)));
            }
            for n in ["input_layernorm", "post_attention_layernorm"] {
                t.push((
                    format!("{p}{n}.weight"),
                    vec![hidden],
                    rnd(hidden, 0.2).iter().map(|v| 1.0 + v).collect(),
                ));
            }
        }
        write_safetensors(&dir.join("model.safetensors"), &t);
        dir
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Sequences of different lengths stepped together equal their own one-token steps, bit for
    /// bit, at every weight precision; also when sequences leave the batch and when it is wider
    /// than one pass.
    #[test]
    fn batch_steps_match_one_token_steps() {
        crate::pool::init(3);
        let dir = tiny_llama("batch", 300, 64, 3, 5, &[]);
        let cfg = QwenConfig::load(&dir).unwrap();
        let files = ModelFiles::open(&dir, true).unwrap();
        for quant in [Quant::Bf16, Quant::Q8, Quant::Q4] {
            let model = QwenModel::load(cfg.clone(), &files, quant).unwrap();
            assert_eq!(model.batch_width(), if quant == Quant::Bf16 { 7 } else { decode_max_tokens() });
            let n = 10; // wider than bf16's 7 columns
            let prompts: Vec<Vec<u32>> = (0..n)
                .map(|i| (0..3 + 2 * i).map(|j| ((i * 37 + j * 11) % 300) as u32).collect())
                .collect();
            let prefill = |p: &[u32]| {
                let mut s = model.new_session();
                model.forward_hidden_mm(&mut s, p, &[]);
                s
            };
            let mut alone: Vec<QwenSession> = prompts.iter().map(|p| prefill(p)).collect();
            let mut batch: Vec<QwenSession> = prompts.iter().map(|p| prefill(p)).collect();
            for step in 0..6u32 {
                // Sequence `i` stops after `i % 5 + 2` steps: the batch shrinks.
                let live: Vec<usize> = (0..n).filter(|i| (i % 5 + 2) as u32 > step).collect();
                let toks: Vec<u32> = live.iter().map(|&i| (i as u32 * 13 + step * 29) % 300).collect();
                let want: Vec<f32> = live
                    .iter()
                    .zip(&toks)
                    .flat_map(|(&i, &t)| model.forward_hidden_mm(&mut alone[i], &[t], &[]))
                    .collect();
                let mut refs: Vec<&mut QwenSession> = batch
                    .iter_mut()
                    .enumerate()
                    .filter(|(i, _)| live.contains(i))
                    .map(|(_, s)| s)
                    .collect();
                let got = model.forward_hidden_batch(&mut refs, &toks);
                assert_eq!(bits(&got), bits(&want), "{quant:?} step {step}");
            }
            for (a, b) in alone.iter().zip(&batch) {
                assert_eq!(a.pos, b.pos);
                let (ca, cb) = (a.kv[0].as_ref().unwrap(), b.kv[0].as_ref().unwrap());
                assert_eq!(bits(ca.data().0), bits(cb.data().0));
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
