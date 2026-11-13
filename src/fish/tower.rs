//! The codebook tower of Drama 3 (`audio_decoder*` of the checkpoint; `FishOmniAudioDecoder` in
//! fish-speech's `fish_omni/tower.py`): from the backbone's hidden state of a row it writes the
//! codes 1 .. N-1 of the next frame, one layer per code, every layer attending over all earlier
//! rows of the sequence through its own key / value cache.
//!
//! A layer mixes the embedding of a code into its input (`eh_proj` over the two RMS-normalized
//! halves), runs one pre-norm transformer block (grouped attention with query / key norms, a
//! quarter of each head rotated in adjacent pairs, a sigmoid output gate; a SiLU-gated MLP) and
//! normalizes. The shared layers take code 0 (the main token's offset in the semantic band), the
//! first of them also the codes of the row just read that the backbone did not see; residual
//! layer `i` takes code `i` and its head gives code `i + 1`. A row whose next token is text takes
//! no code embedding, but every layer still runs on it (its keys and values are attended later).

use crate::kernels::*;
use crate::model::{LayerCache, Lin, Quant, Table, attention, linear, linear_multi, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// `audio_decoder_config` of `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct TowerConfig {
    pub dim: usize,
    pub intermediate_size: usize,
    pub n_head: usize,
    pub n_local_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f64,
    pub rope_base: f64,
    pub rotary_percent: f64,
    /// Entries of a codebook head and of every embedding (the semantic codebook's size).
    pub vocab_size: usize,
    pub num_codebooks: usize,
    pub shared_enrichment_layers: usize,
    /// Codes 1 ..= this many are added to the backbone's input on audio rows.
    pub vq_backbone_codebooks: usize,
    #[serde(default)]
    decoder_type: Option<String>,
    #[serde(default)]
    use_moe: bool,
    #[serde(default)]
    rotary_interleaved: Option<bool>,
    #[serde(default)]
    attention_output_gate: Option<bool>,
}

impl TowerConfig {
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.rotary_percent) as usize
    }

    /// The settings the reference refuses to run without (`FIXED_AUDIO_DECODER_FLAGS`).
    pub fn check(&self) -> Result<()> {
        if self.decoder_type.as_deref().is_some_and(|t| t != "codebook_mtp_shared") {
            bail!("audio_decoder_config.decoder_type {:?} is not supported", self.decoder_type);
        }
        if self.use_moe || self.rotary_interleaved == Some(false) || self.attention_output_gate == Some(false) {
            bail!("audio_decoder_config: only the gated, interleaved-rotary, dense tower is supported");
        }
        let rot = self.rotary_dim();
        if self.n_local_heads == 0
            || !self.n_head.is_multiple_of(self.n_local_heads)
            || rot == 0
            || rot > self.head_dim
            || !rot.is_multiple_of(2)
        {
            bail!("audio_decoder_config: invalid attention geometry");
        }
        if self.num_codebooks < 2 || self.vq_backbone_codebooks >= self.num_codebooks || self.shared_enrichment_layers == 0 {
            bail!("audio_decoder_config: invalid codebook counts");
        }
        Ok(())
    }
}

struct Layer<'a> {
    embed: Table<'a>,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    eh: Lin<'a>,
    attn_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    /// Rows `[query | gate | key | value]`.
    wqkv: Lin<'a>,
    wo: Lin<'a>,
    ffn_norm: Vec<f32>,
    w1: Lin<'a>,
    w3: Lin<'a>,
    w2: Lin<'a>,
    final_norm: Vec<f32>,
}

pub struct Tower<'a> {
    pub cfg: TowerConfig,
    /// `[vocab * codebooks, dim]`: the embedding of code `c` of codebook `i` is row `i * vocab + c`.
    code_embed: Table<'a>,
    shared: Vec<Layer<'a>>,
    residual: Vec<Layer<'a>>,
    heads: Vec<Lin<'a>>,
    inv_freq: Vec<f32>,
    pub weight_bytes: usize,
}

/// The caches of one sequence: a key / value cache per layer, one entry per row seen.
pub struct TowerState {
    caches: Vec<LayerCache>,
}

impl TowerState {
    /// Rows the tower has run on.
    pub fn len(&self) -> usize {
        self.caches.first().map_or(0, |c| c.len)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A zero-centred RMSNorm weight as applied (`1 + w`).
fn norm(files: &ModelFiles, name: &str, n: usize) -> Result<Vec<f32>> {
    let w = files.f32(name).with_context(|| format!("tensor {name}"))?;
    if w.len() != n {
        bail!("{name}: {} values, expected {n}", w.len());
    }
    Ok(w.into_iter().map(|v| 1.0 + v).collect())
}

impl<'a> Layer<'a> {
    fn load(files: &'a ModelFiles, p: &str, embedding: &str, c: &TowerConfig, quant: Quant, bytes: &mut usize) -> Result<Self> {
        let (d, mi, hd) = (c.dim, c.intermediate_size, c.head_dim);
        let qw = c.n_head * hd;
        let kw = c.n_local_heads * hd;
        let mut lin = |n: &str, n_out: usize, n_in: usize| -> Result<Lin<'a>> {
            let l = load_lin(files, &format!("{p}{n}.weight"), n_out, n_in, quant)?;
            *bytes += l.bytes();
            Ok(l)
        };
        let t = "transformer_layer.";
        Ok(Self {
            eh: lin("eh_proj", d, 2 * d)?,
            wqkv: lin(&format!("{t}attention.wqkv"), 2 * qw + 2 * kw, d)?,
            wo: lin(&format!("{t}attention.wo"), d, qw)?,
            w1: lin(&format!("{t}feed_forward.w1"), mi, d)?,
            w3: lin(&format!("{t}feed_forward.w3"), mi, d)?,
            w2: lin(&format!("{t}feed_forward.w2"), d, mi)?,
            embed: Table::load(files, &format!("{p}{embedding}.weight"), c.vocab_size, d, Quant::Bf16)?,
            enorm: norm(files, &format!("{p}enorm.weight"), d)?,
            hnorm: norm(files, &format!("{p}hnorm.weight"), d)?,
            attn_norm: norm(files, &format!("{p}{t}attention_norm.weight"), d)?,
            q_norm: norm(files, &format!("{p}{t}attention.q_norm.weight"), hd)?,
            k_norm: norm(files, &format!("{p}{t}attention.k_norm.weight"), hd)?,
            ffn_norm: norm(files, &format!("{p}{t}ffn_norm.weight"), d)?,
            final_norm: norm(files, &format!("{p}final_layernorm.weight"), d)?,
        })
    }

    /// The layer over `t` rows at positions `start..`: `hidden` (`[t, dim]`) mixed with the code
    /// embeddings `e` (`[t, dim]`, zero rows where the next token is not audio).
    fn forward(&self, c: &TowerConfig, cache: &mut LayerCache, hidden: &[f32], e: &[f32], start: usize, inv_freq: &[f32]) -> Vec<f32> {
        let (d, hd, nh, nkv) = (c.dim, c.head_dim, c.n_head, c.n_local_heads);
        let t = hidden.len() / d;
        let eps = c.norm_eps as f32;
        let par = t > 1;
        debug_assert_eq!(cache.len, start);

        // The two normalized halves, embedding first.
        let mut cat = vec![0f32; t * 2 * d];
        for_chunks(&mut cat, 2 * d, par, |i, r| {
            let (a, b) = r.split_at_mut(d);
            rmsnorm_f32w(&e[i * d..(i + 1) * d], &self.enorm, eps, a);
            rmsnorm_f32w(&hidden[i * d..(i + 1) * d], &self.hnorm, eps, b);
        });
        let mut x = vec![0f32; t * d];
        linear(&self.eh, d, 2 * d, &cat, t, &mut x);

        // Attention.
        let mut xn = vec![0f32; t * d];
        rmsnorm_f32w_rows(&x, d, &self.attn_norm, eps, &mut xn);
        let (qw, kw) = (nh * hd, nkv * hd);
        let w = 2 * qw + 2 * kw;
        let mut qgkv = vec![0f32; t * w];
        linear(&self.wqkv, w, d, &xn, t, &mut qgkv);
        let half = inv_freq.len();
        let positions: Vec<usize> = (start..start + t).collect();
        let (cos, sin) = rope_tables(&positions, inv_freq);
        let scale = (hd as f32).powf(-0.5);
        let mut q = vec![0f32; t * qw];
        for_chunks(&mut q, hd, par, |i, qo| {
            let (ti, h) = (i / nh, i % nh);
            rmsnorm_f32w(&qgkv[ti * w + h * hd..ti * w + (h + 1) * hd], &self.q_norm, eps, qo);
            apply_rope_interleaved(
                &mut qo[..2 * half],
                &cos[ti * half..(ti + 1) * half],
                &sin[ti * half..(ti + 1) * half],
            );
            qo.iter_mut().for_each(|v| *v *= scale);
        });
        let mut k = vec![0f32; t * kw];
        for_chunks(&mut k, hd, par, |i, ko| {
            let (ti, h) = (i / nkv, i % nkv);
            let o = ti * w + 2 * qw + h * hd;
            rmsnorm_f32w(&qgkv[o..o + hd], &self.k_norm, eps, ko);
            apply_rope_interleaved(
                &mut ko[..2 * half],
                &cos[ti * half..(ti + 1) * half],
                &sin[ti * half..(ti + 1) * half],
            );
        });
        let mut v = vec![0f32; t * kw];
        for (ti, r) in v.chunks_mut(kw).enumerate() {
            r.copy_from_slice(&qgkv[ti * w + 2 * qw + kw..(ti + 1) * w]);
        }
        cache.push(&k, &v, t);
        let mut att = vec![0f32; t * qw];
        attention(&q, cache, nh, t, start, None, &mut att);
        for (ti, r) in att.chunks_mut(qw).enumerate() {
            for (a, &g) in r.iter_mut().zip(&qgkv[ti * w + qw..ti * w + 2 * qw]) {
                *a *= 1.0 / (1.0 + (-g).exp());
            }
        }
        let mut y = vec![0f32; t * d];
        linear(&self.wo, d, qw, &att, t, &mut y);
        add_inplace(&mut x, &y);

        // Feed-forward.
        let mi = c.intermediate_size;
        rmsnorm_f32w_rows(&x, d, &self.ffn_norm, eps, &mut xn);
        let mut g = vec![0f32; t * mi];
        let mut u = vec![0f32; t * mi];
        linear_multi(&mut [(&self.w1, mi, &mut g[..]), (&self.w3, mi, &mut u[..])], d, &xn, t);
        silu_mul(&mut g, &u);
        linear(&self.w2, d, mi, &g, t, &mut y);
        add_inplace(&mut x, &y);
        rmsnorm_f32w_rows_inplace(&mut x, d, &self.final_norm, eps);
        x
    }
}

impl<'a> Tower<'a> {
    /// Loads the tower; `quant` is the precision its matrices are kept in (bf16 as stored, or
    /// quantized at load).
    pub fn load(cfg: TowerConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        cfg.check()?;
        let d = cfg.dim;
        let mut bytes = 0;
        let shared = (0..cfg.shared_enrichment_layers)
            .map(|i| {
                Layer::load(
                    files,
                    &format!("audio_decoder_mtp_shared.layers.{i}."),
                    "semantic_embedding",
                    &cfg,
                    quant,
                    &mut bytes,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let residual = (0..cfg.num_codebooks - 1)
            .map(|i| {
                Layer::load(
                    files,
                    &format!("audio_decoder_mtp.layers.{i}."),
                    "codebook_embedding",
                    &cfg,
                    quant,
                    &mut bytes,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let heads = (0..cfg.num_codebooks - 1)
            .map(|i| load_lin(files, &format!("audio_decoder_mtp.heads.{i}.weight"), cfg.vocab_size, d, quant))
            .collect::<Result<Vec<_>>>()?;
        bytes += heads.iter().map(|h| h.bytes()).sum::<usize>();
        let code_embed = Table::load(
            files,
            "audio_decoder.codebook_embeddings.weight",
            cfg.vocab_size * cfg.num_codebooks,
            d,
            Quant::Bf16,
        )?;
        bytes += (shared.len() + residual.len() + cfg.num_codebooks) * cfg.vocab_size * d * 2;
        let rot = cfg.rotary_dim();
        let inv_freq = (0..rot / 2)
            .map(|i| cfg.rope_base.powf(-((2 * i) as f64) / rot as f64) as f32)
            .collect();
        Ok(Self {
            cfg,
            code_embed,
            shared,
            residual,
            heads,
            inv_freq,
            weight_bytes: bytes,
        })
    }

    pub fn new_state(&self) -> TowerState {
        let n = self.shared.len() + self.residual.len();
        TowerState {
            caches: (0..n).map(|_| LayerCache::new(self.cfg.n_local_heads, self.cfg.head_dim)).collect(),
        }
    }

    /// Adds the embeddings of codes `first..end` of the frame `codes` (`[num_codebooks]`) to `out`.
    pub fn add_codes(&self, codes: &[u32], first: usize, end: usize, out: &mut [f32]) {
        let mut row = vec![0f32; self.cfg.dim];
        for (i, &code) in codes.iter().enumerate().take(end).skip(first) {
            let c = (code as usize).min(self.cfg.vocab_size - 1);
            self.code_embed.row(i * self.cfg.vocab_size + c, &mut row);
            add_inplace(out, &row);
        }
    }

    /// The backbone's input for an audio row: the main token's embedding `text` plus the codes
    /// the backbone reads, over `sqrt(1 + their count)`.
    pub fn embed_backbone(&self, codes: &[u32], text: &mut [f32]) {
        let n = self.cfg.vq_backbone_codebooks;
        self.add_codes(codes, 1, 1 + n, text);
        scale_inplace(text, 1.0 / ((1 + n) as f32).sqrt());
    }

    /// What the first shared layer adds for a row, whatever follows it: the codes of the row's
    /// own frame that the backbone did not read (`None` for a text row).
    fn unseen(&self, row_codes: Option<&[u32]>, out: &mut [f32]) {
        if let Some(codes) = row_codes {
            self.add_codes(codes, 1 + self.cfg.vq_backbone_codebooks, self.cfg.num_codebooks, out);
        }
    }

    /// The embedding of `code` for layer `l` into `out`.
    fn code_row(l: &Layer, vocab: usize, code: u32, out: &mut [f32]) {
        l.embed.row((code as usize).min(vocab - 1), out);
    }

    /// Teacher-forces the caches over `t` rows from position `st.len()`: `hidden` (`[t, dim]`)
    /// are the backbone's states of the rows, `rows[i]` the codes of row `i`'s own frame (None for
    /// a text row) and `next[i]` those of the frame after it.
    pub fn prefill(&self, st: &mut TowerState, hidden: &[f32], rows: &[Option<&[u32]>], next: &[Option<&[u32]>]) {
        let d = self.cfg.dim;
        let t = hidden.len() / d;
        debug_assert!(rows.len() == t && next.len() == t);
        let start = st.len();
        let mut h = hidden.to_vec();
        let n_shared = self.shared.len();
        for (li, layer) in self.shared.iter().chain(&self.residual).enumerate() {
            // Shared layers read code 0, residual layer i code i.
            let ci = li.saturating_sub(n_shared);
            let mut e = vec![0f32; t * d];
            for (i, r) in e.chunks_mut(d).enumerate() {
                if let Some(codes) = next[i] {
                    Self::code_row(layer, self.cfg.vocab_size, codes[ci], r);
                }
                if li == 0 {
                    self.unseen(rows[i], r);
                }
            }
            h = layer.forward(&self.cfg, &mut st.caches[li], &h, &e, start, &self.inv_freq);
        }
    }

    /// One decoding step at position `st.len()`: from the backbone's state `hidden` of the row
    /// (whose own frame is `row_codes`, None for a text row) and the next frame's code 0 (`c0`,
    /// None when the next token is not audio), the codes 1 .. N-1 of the next frame. `pick(i,
    /// logits)` chooses code `i` from head `i - 1`'s logits (`[vocab]`).
    pub fn step(
        &self,
        st: &mut TowerState,
        hidden: &[f32],
        row_codes: Option<&[u32]>,
        c0: Option<u32>,
        pick: &mut dyn FnMut(usize, &[f32]) -> u32,
    ) -> Vec<u32> {
        let d = self.cfg.dim;
        let start = st.len();
        let vocab = self.cfg.vocab_size;
        let mut h = hidden.to_vec();
        let mut e = vec![0f32; d];
        for (li, layer) in self.shared.iter().enumerate() {
            e.fill(0.0);
            if let Some(c) = c0 {
                Self::code_row(layer, vocab, c, &mut e);
            }
            if li == 0 {
                self.unseen(row_codes, &mut e);
            }
            h = layer.forward(&self.cfg, &mut st.caches[li], &h, &e, start, &self.inv_freq);
        }
        let n_shared = self.shared.len();
        let mut codes = vec![c0.unwrap_or(0)];
        let mut logits = vec![0f32; vocab];
        for (i, (layer, head)) in self.residual.iter().zip(&self.heads).enumerate() {
            e.fill(0.0);
            if c0.is_some() {
                Self::code_row(layer, vocab, codes[i], &mut e);
            }
            h = layer.forward(&self.cfg, &mut st.caches[n_shared + i], &h, &e, start, &self.inv_freq);
            linear(head, vocab, d, &h, 1, &mut logits);
            codes.push(pick(i + 1, &logits));
        }
        codes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen::batch::tests::write_safetensors;

    /// A random tower (3 codebooks of 16 entries, 2 shared layers, width 16) in a temporary file.
    fn tiny_tower(tag: &str) -> (TowerConfig, std::path::PathBuf) {
        let cfg: TowerConfig = serde_json::from_value(serde_json::json!({
            "dim": 16, "intermediate_size": 32, "n_head": 4, "n_local_heads": 2, "head_dim": 8, "norm_eps": 1e-6,
            "rope_base": 10000.0, "rotary_percent": 0.5, "vocab_size": 16, "num_codebooks": 3,
            "shared_enrichment_layers": 2, "vq_backbone_codebooks": 1,
        }))
        .unwrap();
        let mut seed = 11u64;
        let mut rnd = |shape: &[usize]| -> Vec<f32> {
            (0..shape.iter().product())
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                })
                .collect()
        };
        let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut add = |n: String, shape: &[usize]| {
            let v = rnd(shape);
            t.push((n, shape.to_vec(), v));
        };
        let (d, mi, hd, v) = (cfg.dim, cfg.intermediate_size, cfg.head_dim, cfg.vocab_size);
        add("audio_decoder.codebook_embeddings.weight".into(), &[v * cfg.num_codebooks, d]);
        let layers = (0..cfg.shared_enrichment_layers)
            .map(|i| (format!("audio_decoder_mtp_shared.layers.{i}"), "semantic_embedding"))
            .chain((0..cfg.num_codebooks - 1).map(|i| (format!("audio_decoder_mtp.layers.{i}"), "codebook_embedding")));
        for (p, e) in layers {
            add(format!("{p}.{e}.weight"), &[v, d]);
            add(format!("{p}.eh_proj.weight"), &[d, 2 * d]);
            for n in [
                "enorm",
                "hnorm",
                "final_layernorm",
                "transformer_layer.attention_norm",
                "transformer_layer.ffn_norm",
            ] {
                add(format!("{p}.{n}.weight"), &[d]);
            }
            let a = format!("{p}.transformer_layer.attention");
            add(format!("{a}.q_norm.weight"), &[hd]);
            add(format!("{a}.k_norm.weight"), &[hd]);
            add(format!("{a}.wqkv.weight"), &[2 * cfg.n_head * hd + 2 * cfg.n_local_heads * hd, d]);
            add(format!("{a}.wo.weight"), &[d, cfg.n_head * hd]);
            let f = format!("{p}.transformer_layer.feed_forward");
            add(format!("{f}.w1.weight"), &[mi, d]);
            add(format!("{f}.w3.weight"), &[mi, d]);
            add(format!("{f}.w2.weight"), &[d, mi]);
        }
        for i in 0..cfg.num_codebooks - 1 {
            add(format!("audio_decoder_mtp.heads.{i}.weight"), &[v, d]);
        }
        let path = std::env::temp_dir().join(format!("oxinfer-tiny-fish-tower-{tag}-{}.safetensors", std::process::id()));
        write_safetensors(&path, &t);
        (cfg, path)
    }

    /// Rows teacher-forced in one pass, one at a time, or by decoding steps whose codes are
    /// forced leave the same caches: the step after them gives the same logits.
    #[test]
    fn prefill_and_steps_agree() {
        let (cfg, path) = tiny_tower("steps");
        let files = ModelFiles::open_file(&path, true).unwrap();
        let tower = Tower::load(cfg, &files, Quant::Bf16).unwrap();
        let d = tower.cfg.dim;
        let n = 6;
        let hidden: Vec<f32> = (0..(n + 1) * d).map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0).collect();
        // Rows 0 and 1 are text, the others audio frames; the frame after the last row is text.
        let frames: Vec<Option<Vec<u32>>> = (0..n + 2)
            .map(|i| {
                (2..=n)
                    .contains(&i)
                    .then(|| vec![(i * 5 % 16) as u32, (i * 3 % 16) as u32, (i * 7 % 16) as u32])
            })
            .collect();
        let own: Vec<Option<&[u32]>> = frames[..n].iter().map(|f| f.as_deref()).collect();
        let next: Vec<Option<&[u32]>> = frames[1..=n].iter().map(|f| f.as_deref()).collect();
        let last = |st: &mut TowerState| -> Vec<f32> {
            let mut logits = Vec::new();
            tower.step(st, &hidden[n * d..], frames[n].as_deref(), Some(9), &mut |_, l| {
                logits.extend_from_slice(l);
                argmax(l)
            });
            logits
        };
        fn argmax(l: &[f32]) -> u32 {
            l.iter().enumerate().fold(0, |b, (i, &v)| if v > l[b] { i } else { b }) as u32
        }

        let mut whole = tower.new_state();
        tower.prefill(&mut whole, &hidden[..n * d], &own, &next);
        assert_eq!(whole.len(), n);
        let want = last(&mut whole);
        assert_eq!(want.len(), 2 * tower.cfg.vocab_size);

        let mut rows = tower.new_state();
        for i in 0..n {
            tower.prefill(&mut rows, &hidden[i * d..(i + 1) * d], &own[i..=i], &next[i..=i]);
        }
        let mut steps = tower.new_state();
        for i in 0..n {
            let forced = next[i];
            tower.step(
                &mut steps,
                &hidden[i * d..(i + 1) * d],
                own[i],
                forced.map(|c| c[0]),
                &mut |c, _| forced.map_or(0, |f| f[c]),
            );
        }
        for got in [last(&mut rows), last(&mut steps)] {
            for (a, b) in want.iter().zip(&got) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
        }
        std::fs::remove_file(&path).unwrap();
    }
}
