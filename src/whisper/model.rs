//! The Whisper network: a convolutional stem and pre-LayerNorm transformer encoder over 30 s of
//! log-mel frames, and a transformer decoder with learned positions, causal self-attention over a
//! KV cache, cross-attention over the encoder output and the output projection tied to the token
//! embedding. Layer names follow transformers' `WhisperForConditionalGeneration` checkpoints.
//!
//! All weights are converted at load time into owned matrices of the requested [`Precision`], so
//! the model does not borrow the checkpoint files. Matmuls go through the shared driver
//! (`model::linear_multi`): SIMD CPU kernels, int8 / int4 kernels for quantized weights, and the
//! GPU backend for the big encoder matmuls (it declines small ones such as single-token decoding).

use crate::bert::{LayerNorm, gelu_erf};
use crate::ggml::{GgmlType, GgufMatrix};
use crate::kernels::{bf16_to_f32_slice, for_chunks};
use crate::model::{LayerCache, Lin, Weight, attention_masked, linear_multi};
use crate::quant::{PageVec, quantize_bf16, quantize_q4};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

const LN_EPS: f32 = 1e-5;

fn d_1500() -> usize {
    1500
}
fn d_448() -> usize {
    448
}
fn d_gelu() -> String {
    "gelu".into()
}

/// The fields of a Whisper `config.json` the model needs.
#[derive(Debug, Clone, Deserialize)]
pub struct WhisperConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub vocab_size: usize,
    #[serde(default = "d_1500")]
    pub max_source_positions: usize,
    #[serde(default = "d_448")]
    pub max_target_positions: usize,
    #[serde(default = "d_gelu")]
    pub activation_function: String,
    #[serde(default)]
    pub scale_embedding: bool,
}

impl WhisperConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: WhisperConfig = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?).context("parse Whisper config")?;
        if cfg.activation_function != "gelu" {
            bail!("activation_function {} is not supported (only gelu)", cfg.activation_function);
        }
        if !cfg.d_model.is_multiple_of(cfg.encoder_attention_heads) || !cfg.d_model.is_multiple_of(cfg.decoder_attention_heads) {
            bail!("d_model {} does not split into the attention heads", cfg.d_model);
        }
        Ok(cfg)
    }
}

/// Storage of the weight matrices. Norms, biases and positions always stay f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// Exact checkpoint values (Whisper checkpoints are f32 or f16). CPU only.
    F32,
    /// Half precision (exact for f16 checkpoints such as large-v3). CPU only.
    F16,
    /// bfloat16: the CPU's bf16 kernels and the GPU backend.
    Bf16,
    /// int8 blocks of 32 (GPU-capable).
    Q8,
    /// int4 blocks of 32 (GPU-capable).
    Q4,
}

impl Precision {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "f32" | "fp32" => Precision::F32,
            "f16" | "fp16" => Precision::F16,
            "bf16" => Precision::Bf16,
            "q8" | "int8" => Precision::Q8,
            "q4" | "q4_0" => Precision::Q4,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
            Precision::Bf16 => "bf16",
            Precision::Q8 => "q8",
            Precision::Q4 => "q4",
        }
    }
}

/// An owned `[n_out, n_in]` matrix of f32 values stored in `p`.
fn make_lin(w: &[f32], n_out: usize, n_in: usize, p: Precision) -> Result<Lin<'static>> {
    debug_assert_eq!(w.len(), n_out * n_in);
    // Block formats need whole blocks of 32 per row; fall back to bf16 otherwise.
    let p = match p {
        Precision::Q8 | Precision::Q4 if !n_in.is_multiple_of(crate::quant::QBLOCK) => Precision::Bf16,
        p => p,
    };
    let bf16 = || {
        let mut v = PageVec::<u16>::zeroed(w.len());
        for (o, &x) in v.iter_mut().zip(w) {
            *o = half::bf16::from_f32(x).to_bits();
        }
        v
    };
    Ok(match p {
        Precision::F32 => {
            let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
            Lin::gguf(GgufMatrix::from_bytes(GgmlType::F32, &bytes, n_out, n_in)?)
        }
        Precision::F16 => {
            let bytes: Vec<u8> = w.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect();
            Lin::gguf(GgufMatrix::from_bytes(GgmlType::F16, &bytes, n_out, n_in)?)
        }
        Precision::Bf16 => Lin::bf16_owned(bf16()),
        Precision::Q8 => Lin::q8(quantize_bf16(&bf16(), n_out, n_in)),
        Precision::Q4 => Lin::q4(quantize_q4(&bf16(), n_out, n_in)),
    })
}

/// Row `r` of a matrix as f32 (token embeddings come from the tied output projection).
fn lin_row(l: &Lin, n_in: usize, r: usize, out: &mut [f32]) {
    match &l.w {
        Weight::Bf16(w, _) => bf16_to_f32_slice(&w[r * n_in..(r + 1) * n_in], out),
        Weight::Q8(m) => crate::quant::dequantize_rows(m, r, r + 1, out),
        Weight::Q4(m) => crate::quant::dequantize_rows_q4(m, r, r + 1, out),
        Weight::Gguf(m) => m.dequantize_rows(r, r + 1, out),
    }
}

fn tensor(files: &ModelFiles, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let info = files.info(name).with_context(|| format!("missing tensor {name}"))?;
    if info.shape != shape {
        bail!("tensor {name} has shape {:?}, expected {shape:?}", info.shape);
    }
    files.f32(name)
}

/// A linear layer with an optional bias.
pub(crate) struct Proj {
    w: Lin<'static>,
    b: Option<Vec<f32>>,
    n_out: usize,
}

impl Proj {
    fn load(files: &ModelFiles, p: &str, n_out: usize, n_in: usize, prec: Precision) -> Result<Self> {
        let w = tensor(files, &format!("{p}.weight"), &[n_out, n_in])?;
        let bn = format!("{p}.bias");
        let b = if files.has(&bn) {
            Some(tensor(files, &bn, &[n_out])?)
        } else {
            None
        };
        Ok(Self {
            w: make_lin(&w, n_out, n_in, prec)?,
            b,
            n_out,
        })
    }

    fn add_bias(&self, y: &mut [f32]) {
        if let Some(b) = &self.b {
            for_chunks(y, self.n_out, y.len() > 4 * self.n_out, |_, r| crate::kernels::add_inplace(r, b));
        }
    }
}

/// Several projections of one input `x` (`[t, n_in]`) in one batch (one GPU submission).
fn project(ps: &[&Proj], n_in: usize, x: &[f32], t: usize) -> Vec<Vec<f32>> {
    let mut ys: Vec<Vec<f32>> = ps.iter().map(|p| vec![0f32; t * p.n_out]).collect();
    {
        let mut jobs: Vec<(&Lin, usize, &mut [f32])> = ps.iter().zip(ys.iter_mut()).map(|(p, y)| (&p.w, p.n_out, &mut y[..])).collect();
        linear_multi(&mut jobs, n_in, x, t);
    }
    for (p, y) in ps.iter().zip(ys.iter_mut()) {
        p.add_bias(y);
    }
    ys
}

fn gelu_inplace(x: &mut [f32]) {
    for_chunks(x, 4096, x.len() > 65536, |_, c| c.iter_mut().for_each(|v| *v = gelu_erf(*v)));
}

struct Attn {
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
}

impl Attn {
    fn load(files: &ModelFiles, p: &str, d: usize, prec: Precision) -> Result<Self> {
        Ok(Self {
            q: Proj::load(files, &format!("{p}.q_proj"), d, d, prec)?,
            k: Proj::load(files, &format!("{p}.k_proj"), d, d, prec)?,
            v: Proj::load(files, &format!("{p}.v_proj"), d, d, prec)?,
            o: Proj::load(files, &format!("{p}.out_proj"), d, d, prec)?,
        })
    }
}

struct EncLayer {
    ln1: LayerNorm,
    attn: Attn,
    ln2: LayerNorm,
    fc1: Proj,
    fc2: Proj,
}

struct DecLayer {
    ln1: LayerNorm,
    self_attn: Attn,
    ln_cross: LayerNorm,
    cross: Attn,
    ln2: LayerNorm,
    fc1: Proj,
    fc2: Proj,
}

pub struct WhisperModel {
    pub cfg: WhisperConfig,
    conv1: Proj, // [d, 3 * n_mels (padded to 32)], taps in [k][cin] order
    conv1_in: usize,
    conv2: Proj, // [d, 3 * d]
    enc_pos: Vec<f32>,
    enc_layers: Vec<EncLayer>,
    enc_ln: LayerNorm,
    dec_pos: Vec<f32>,
    dec_layers: Vec<DecLayer>,
    dec_ln: LayerNorm,
    /// The output projection, tied to the token embedding (`[vocab, d]`).
    lm_head: Lin<'static>,
}

/// Cross-attention keys and values of one decoder layer, head-major (`[heads][n][dh]`) so each
/// head's keys are contiguous.
pub struct CrossLayer {
    k: Vec<f32>,
    v: Vec<f32>,
    pub n: usize,
    nh: usize,
    dh: usize,
}

impl CrossLayer {
    /// From position-major `[n][heads * dh]` projections.
    fn new(k: &[f32], v: &[f32], n: usize, nh: usize, dh: usize) -> Self {
        let d = nh * dh;
        let mut kt = vec![0f32; n * d];
        let mut vt = vec![0f32; n * d];
        for_chunks(&mut kt, n * dh, true, |h, r| {
            for p in 0..n {
                r[p * dh..(p + 1) * dh].copy_from_slice(&k[p * d + h * dh..p * d + (h + 1) * dh]);
            }
        });
        for_chunks(&mut vt, n * dh, true, |h, r| {
            for p in 0..n {
                r[p * dh..(p + 1) * dh].copy_from_slice(&v[p * d + h * dh..p * d + (h + 1) * dh]);
            }
        });
        Self { k: kt, v: vt, n, nh, dh }
    }

    #[inline]
    fn key(&self, h: usize, p: usize) -> &[f32] {
        let o = (h * self.n + p) * self.dh;
        &self.k[o..o + self.dh]
    }

    #[inline]
    fn value(&self, h: usize, p: usize) -> &[f32] {
        let o = (h * self.n + p) * self.dh;
        &self.v[o..o + self.dh]
    }

    /// Attention of `t` (pre-scaled) queries `[t][heads * dh]` over all positions into `out`.
    /// Positions are split into chunks so a single query keeps every thread busy; the chunks'
    /// partial softmax sums are merged at the end.
    fn attend(&self, q: &[f32], t: usize, out: &mut [f32]) {
        let (nh, dh, n) = (self.nh, self.dh, self.n);
        let d = nh * dh;
        let rows = t * nh;
        let chunks = (2 * crate::pool::threads()).div_ceil(rows).clamp(1, n.div_ceil(128));
        let per = n.div_ceil(chunks);
        // Per task: max, sum, then dh weighted values.
        let stride = dh + 2;
        let mut part = vec![0f32; rows * chunks * stride];
        for_chunks(&mut part, stride, rows * chunks > 1, |task, slot| {
            let (row, c) = (task / chunks, task % chunks);
            let (ti, h) = (row / nh, row % nh);
            let (p0, p1) = (c * per, ((c + 1) * per).min(n));
            if p0 >= p1 {
                slot[0] = f32::NEG_INFINITY;
                return;
            }
            let qv = &q[ti * d + h * dh..ti * d + (h + 1) * dh];
            let mut sc: Vec<f32> = (p0..p1).map(|p| crate::kernels::dot(qv, self.key(h, p))).collect();
            let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut l = 0f32;
            for s in sc.iter_mut() {
                *s = crate::kernels::fast_exp(*s - m);
                l += *s;
            }
            let (head, o) = slot.split_at_mut(2);
            head[0] = m;
            head[1] = l;
            o.fill(0.0);
            for (p, &w) in (p0..p1).zip(&sc) {
                crate::kernels::axpy(o, w, self.value(h, p));
            }
        });
        for row in 0..rows {
            let (ti, h) = (row / nh, row % nh);
            let slots = &part[row * chunks * stride..(row + 1) * chunks * stride];
            let m = slots.chunks(stride).map(|s| s[0]).fold(f32::NEG_INFINITY, f32::max);
            let o = &mut out[ti * d + h * dh..ti * d + (h + 1) * dh];
            o.fill(0.0);
            let mut l = 0f32;
            for s in slots.chunks(stride) {
                if s[0] == f32::NEG_INFINITY {
                    continue;
                }
                let w = (s[0] - m).exp();
                l += w * s[1];
                crate::kernels::axpy(o, w, &s[2..]);
            }
            crate::kernels::scale_inplace(o, 1.0 / l);
        }
    }
}

/// Cross-attention keys and values of one encoded window, per decoder layer.
pub struct CrossKv {
    pub layers: Vec<CrossLayer>,
}

/// The decoder's self-attention cache.
#[derive(Clone)]
pub struct DecState {
    pub layers: Vec<LayerCache>,
    pub pos: usize,
}

impl DecState {
    pub fn truncate(&mut self, pos: usize) {
        for c in &mut self.layers {
            c.truncate(pos);
        }
        self.pos = self.pos.min(pos);
    }
}

/// A torch conv1d weight `[cout, cin, k]` rearranged into unfolded-row order `[cout][k][cin]`,
/// rows zero-padded to `n_in` columns.
fn conv_weight(w: &[f32], cout: usize, cin: usize, k: usize, n_in: usize) -> Vec<f32> {
    let mut r = vec![0f32; cout * n_in];
    for o in 0..cout {
        for i in 0..cin {
            for j in 0..k {
                r[o * n_in + j * cin + i] = w[(o * cin + i) * k + j];
            }
        }
    }
    r
}

impl WhisperModel {
    /// Loads the model; `enc` is the precision of the encoder's matrices, `dec` of the decoder's
    /// (including the tied embedding / output projection).
    pub fn load(cfg: WhisperConfig, files: &ModelFiles, enc: Precision, dec: Precision) -> Result<Self> {
        let d = cfg.d_model;
        let nm = cfg.num_mel_bins;
        // The conv stem stays at bf16 or better: it is not where the time goes.
        let conv_prec = match enc {
            Precision::Q8 | Precision::Q4 => Precision::Bf16,
            p => p,
        };
        let conv1_in = (3 * nm).next_multiple_of(32);
        let w1 = tensor(files, "model.encoder.conv1.weight", &[d, nm, 3])?;
        let conv1 = Proj {
            w: make_lin(&conv_weight(&w1, d, nm, 3, conv1_in), d, conv1_in, conv_prec)?,
            b: Some(tensor(files, "model.encoder.conv1.bias", &[d])?),
            n_out: d,
        };
        let w2 = tensor(files, "model.encoder.conv2.weight", &[d, d, 3])?;
        let conv2 = Proj {
            w: make_lin(&conv_weight(&w2, d, d, 3, 3 * d), d, 3 * d, conv_prec)?,
            b: Some(tensor(files, "model.encoder.conv2.bias", &[d])?),
            n_out: d,
        };
        let mut enc_layers = Vec::with_capacity(cfg.encoder_layers);
        for l in 0..cfg.encoder_layers {
            let p = format!("model.encoder.layers.{l}");
            enc_layers.push(EncLayer {
                ln1: LayerNorm::load(files, &format!("{p}.self_attn_layer_norm"), d)?,
                attn: Attn::load(files, &format!("{p}.self_attn"), d, enc)?,
                ln2: LayerNorm::load(files, &format!("{p}.final_layer_norm"), d)?,
                fc1: Proj::load(files, &format!("{p}.fc1"), cfg.encoder_ffn_dim, d, enc)?,
                fc2: Proj::load(files, &format!("{p}.fc2"), d, cfg.encoder_ffn_dim, enc)?,
            });
        }
        let mut dec_layers = Vec::with_capacity(cfg.decoder_layers);
        for l in 0..cfg.decoder_layers {
            let p = format!("model.decoder.layers.{l}");
            dec_layers.push(DecLayer {
                ln1: LayerNorm::load(files, &format!("{p}.self_attn_layer_norm"), d)?,
                self_attn: Attn::load(files, &format!("{p}.self_attn"), d, dec)?,
                ln_cross: LayerNorm::load(files, &format!("{p}.encoder_attn_layer_norm"), d)?,
                // The cross-attention keys and values run over 1500 encoder positions once per
                // window: the encoder's precision (and the GPU) suit them better.
                cross: Attn {
                    q: Proj::load(files, &format!("{p}.encoder_attn.q_proj"), d, d, dec)?,
                    k: Proj::load(files, &format!("{p}.encoder_attn.k_proj"), d, d, enc)?,
                    v: Proj::load(files, &format!("{p}.encoder_attn.v_proj"), d, d, enc)?,
                    o: Proj::load(files, &format!("{p}.encoder_attn.out_proj"), d, d, dec)?,
                },
                ln2: LayerNorm::load(files, &format!("{p}.final_layer_norm"), d)?,
                fc1: Proj::load(files, &format!("{p}.fc1"), cfg.decoder_ffn_dim, d, dec)?,
                fc2: Proj::load(files, &format!("{p}.fc2"), d, cfg.decoder_ffn_dim, dec)?,
            });
        }
        let head_name = if files.has("proj_out.weight") {
            "proj_out.weight"
        } else {
            "model.decoder.embed_tokens.weight"
        };
        let emb = tensor(files, head_name, &[cfg.vocab_size, d])?;
        let lm_head = make_lin(&emb, cfg.vocab_size, d, dec)?;
        drop(emb);
        Ok(Self {
            conv1,
            conv1_in,
            conv2,
            enc_pos: tensor(files, "model.encoder.embed_positions.weight", &[cfg.max_source_positions, d])?,
            enc_layers,
            enc_ln: LayerNorm::load(files, "model.encoder.layer_norm", d)?,
            dec_pos: tensor(files, "model.decoder.embed_positions.weight", &[cfg.max_target_positions, d])?,
            dec_layers,
            dec_ln: LayerNorm::load(files, "model.decoder.layer_norm", d)?,
            lm_head,
            cfg,
        })
    }

    /// Mel frames per window the encoder takes (two per output position).
    pub fn window_frames(&self) -> usize {
        2 * self.cfg.max_source_positions
    }

    /// Encodes one window of log-mel frames (`[window_frames, n_mels]`, time-major) into
    /// `[max_source_positions, d_model]`.
    pub fn encode(&self, mel: &[f32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let (d, nm) = (cfg.d_model, cfg.num_mel_bins);
        let t = self.window_frames();
        assert_eq!(mel.len(), t * nm, "encoder input must be {t} frames of {nm} mels");
        // conv1: kernel 3, padding 1, then GELU.
        let k1 = self.conv1_in;
        let mut cols = vec![0f32; t * k1];
        for_chunks(&mut cols, k1, true, |i, r| {
            for j in 0..3 {
                let src = i as isize + j as isize - 1;
                if src >= 0 && (src as usize) < t {
                    let s = src as usize;
                    r[j * nm..(j + 1) * nm].copy_from_slice(&mel[s * nm..(s + 1) * nm]);
                }
            }
        });
        let mut h = project(&[&self.conv1], k1, &cols, t).pop().unwrap();
        gelu_inplace(&mut h);
        // conv2: kernel 3, stride 2, padding 1, then GELU and the positions.
        let n = cfg.max_source_positions;
        let mut cols = vec![0f32; n * 3 * d];
        for_chunks(&mut cols, 3 * d, true, |i, r| {
            for j in 0..3 {
                let src = (2 * i) as isize + j as isize - 1;
                if src >= 0 && (src as usize) < t {
                    let s = src as usize;
                    r[j * d..(j + 1) * d].copy_from_slice(&h[s * d..(s + 1) * d]);
                }
            }
        });
        drop(h);
        let mut x = project(&[&self.conv2], 3 * d, &cols, n).pop().unwrap();
        drop(cols);
        let pos = &self.enc_pos;
        for_chunks(&mut x, d, true, |i, r| {
            for (v, p) in r.iter_mut().zip(&pos[i * d..(i + 1) * d]) {
                *v = gelu_erf(*v) + p;
            }
        });

        let nh = cfg.encoder_attention_heads;
        let dh = d / nh;
        let mut hn = vec![0f32; n * d];
        for layer in &self.enc_layers {
            hn.copy_from_slice(&x);
            layer.ln1.apply_rows(&mut hn, LN_EPS);
            let a = &layer.attn;
            let mut qkv = project(&[&a.q, &a.k, &a.v], d, &hn, n);
            let (v, k, q) = (qkv.pop().unwrap(), qkv.pop().unwrap(), qkv.pop().unwrap());
            let att = crate::sd::ops::attention(&q, &k, &v, n, n, nh, dh, false);
            let o = project(&[&a.o], d, &att, n).pop().unwrap();
            crate::kernels::add_inplace(&mut x, &o);

            hn.copy_from_slice(&x);
            layer.ln2.apply_rows(&mut hn, LN_EPS);
            let mut f = project(&[&layer.fc1], d, &hn, n).pop().unwrap();
            gelu_inplace(&mut f);
            let o = project(&[&layer.fc2], cfg.encoder_ffn_dim, &f, n).pop().unwrap();
            crate::kernels::add_inplace(&mut x, &o);
        }
        self.enc_ln.apply_rows(&mut x, LN_EPS);
        x
    }

    /// The cross-attention keys and values of every decoder layer for an encoded window.
    pub fn cross_kv(&self, enc: &[f32]) -> CrossKv {
        let d = self.cfg.d_model;
        let n = enc.len() / d;
        let nh = self.cfg.decoder_attention_heads;
        let ps: Vec<&Proj> = self.dec_layers.iter().flat_map(|l| [&l.cross.k, &l.cross.v]).collect();
        let ys = project(&ps, d, enc, n);
        CrossKv {
            layers: ys.chunks(2).map(|kv| CrossLayer::new(&kv[0], &kv[1], n, nh, d / nh)).collect(),
        }
    }

    pub fn new_state(&self) -> DecState {
        let nh = self.cfg.decoder_attention_heads;
        let hd = self.cfg.d_model / nh;
        DecState {
            layers: (0..self.cfg.decoder_layers).map(|_| LayerCache::new(nh, hd)).collect(),
            pos: 0,
        }
    }

    /// Runs `tokens` through the decoder after the cached positions; returns the final
    /// (normalized) hidden states `[tokens.len(), d_model]`.
    pub fn decode(&self, tokens: &[u32], st: &mut DecState, cross: &CrossKv) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let t = tokens.len();
        assert!(
            st.pos + t <= cfg.max_target_positions,
            "decoder context of {} exceeded",
            cfg.max_target_positions
        );
        let nh = cfg.decoder_attention_heads;
        let dh = d / nh;
        let scale = (dh as f32).powf(-0.5);
        let mut x = vec![0f32; t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let r = &mut x[i * d..(i + 1) * d];
            lin_row(&self.lm_head, d, tok as usize, r);
            if cfg.scale_embedding {
                crate::kernels::scale_inplace(r, (d as f32).sqrt());
            }
            crate::kernels::add_inplace(r, &self.dec_pos[(st.pos + i) * d..(st.pos + i + 1) * d]);
        }
        let mut hn = vec![0f32; t * d];
        let mut att = vec![0f32; t * d];
        for (l, layer) in self.dec_layers.iter().enumerate() {
            // Causal self-attention.
            hn.copy_from_slice(&x);
            layer.ln1.apply_rows(&mut hn, LN_EPS);
            let a = &layer.self_attn;
            let mut qkv = project(&[&a.q, &a.k, &a.v], d, &hn, t);
            let (v, k, mut q) = (qkv.pop().unwrap(), qkv.pop().unwrap(), qkv.pop().unwrap());
            crate::kernels::scale_inplace(&mut q, scale);
            st.layers[l].push(&k, &v, t);
            attention_masked(&q, &st.layers[l], nh, t, st.pos, None, true, &mut att);
            let o = project(&[&a.o], d, &att, t).pop().unwrap();
            crate::kernels::add_inplace(&mut x, &o);

            // Cross-attention over the encoder output.
            hn.copy_from_slice(&x);
            layer.ln_cross.apply_rows(&mut hn, LN_EPS);
            let c = &layer.cross;
            let mut q = project(&[&c.q], d, &hn, t).pop().unwrap();
            crate::kernels::scale_inplace(&mut q, scale);
            cross.layers[l].attend(&q, t, &mut att);
            let o = project(&[&c.o], d, &att, t).pop().unwrap();
            crate::kernels::add_inplace(&mut x, &o);

            // Feed-forward.
            hn.copy_from_slice(&x);
            layer.ln2.apply_rows(&mut hn, LN_EPS);
            let mut f = project(&[&layer.fc1], d, &hn, t).pop().unwrap();
            gelu_inplace(&mut f);
            let o = project(&[&layer.fc2], cfg.decoder_ffn_dim, &f, t).pop().unwrap();
            crate::kernels::add_inplace(&mut x, &o);
        }
        st.pos += t;
        self.dec_ln.apply_rows(&mut x, LN_EPS);
        x
    }

    /// Logits `[rows, vocab]` of hidden-state rows (from [`WhisperModel::decode`]).
    pub fn logits(&self, h: &[f32]) -> Vec<f32> {
        let d = self.cfg.d_model;
        let rows = h.len() / d;
        let v = self.cfg.vocab_size;
        let mut y = vec![0f32; rows * v];
        linear_multi(&mut [(&self.lm_head, v, &mut y[..])], d, h, rows);
        y
    }

    /// Scaled cross-attention scores (`q . k / sqrt(dh)`, before the softmax) of the given
    /// `(layer, head)` pairs for every token of `tokens`, run from an empty cache:
    /// `[heads.len()][tokens.len()][source positions]`. For word-level timestamps.
    pub fn cross_attention_weights(&self, tokens: &[u32], cross: &CrossKv, heads: &[(usize, usize)]) -> Vec<Vec<f32>> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let t = tokens.len();
        let nh = cfg.decoder_attention_heads;
        let dh = d / nh;
        let scale = (dh as f32).powf(-0.5);
        let n_src = cross.layers[0].n;
        let last = heads.iter().map(|h| h.0).max().unwrap_or(0);
        let mut out = vec![Vec::new(); heads.len()];
        let mut st = self.new_state();
        let mut x = vec![0f32; t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let r = &mut x[i * d..(i + 1) * d];
            lin_row(&self.lm_head, d, tok as usize, r);
            crate::kernels::add_inplace(r, &self.dec_pos[i * d..(i + 1) * d]);
        }
        let mut hn = vec![0f32; t * d];
        let mut att = vec![0f32; t * d];
        for (l, layer) in self.dec_layers.iter().enumerate().take(last + 1) {
            hn.copy_from_slice(&x);
            layer.ln1.apply_rows(&mut hn, LN_EPS);
            let a = &layer.self_attn;
            let mut qkv = project(&[&a.q, &a.k, &a.v], d, &hn, t);
            let (v, k, mut q) = (qkv.pop().unwrap(), qkv.pop().unwrap(), qkv.pop().unwrap());
            crate::kernels::scale_inplace(&mut q, scale);
            st.layers[l].push(&k, &v, t);
            attention_masked(&q, &st.layers[l], nh, t, 0, None, true, &mut att);
            let o = project(&[&a.o], d, &att, t).pop().unwrap();
            crate::kernels::add_inplace(&mut x, &o);

            hn.copy_from_slice(&x);
            layer.ln_cross.apply_rows(&mut hn, LN_EPS);
            let c = &layer.cross;
            let mut q = project(&[&c.q], d, &hn, t).pop().unwrap();
            crate::kernels::scale_inplace(&mut q, scale);
            let kv = &cross.layers[l];
            for (hi, &(hl, hh)) in heads.iter().enumerate() {
                if hl != l {
                    continue;
                }
                let mut w = vec![0f32; t * n_src];
                for_chunks(&mut w, n_src, t > 1, |i, row| {
                    let qv = &q[i * d + hh * dh..i * d + (hh + 1) * dh];
                    for (s, r) in row.iter_mut().enumerate() {
                        *r = crate::kernels::dot(qv, kv.key(hh, s));
                    }
                });
                out[hi] = w;
            }
            if l == last {
                break;
            }
            kv.attend(&q, t, &mut att);
            let o = project(&[&c.o], d, &att, t).pop().unwrap();
            crate::kernels::add_inplace(&mut x, &o);

            hn.copy_from_slice(&x);
            layer.ln2.apply_rows(&mut hn, LN_EPS);
            let mut f = project(&[&layer.fc1], d, &hn, t).pop().unwrap();
            gelu_inplace(&mut f);
            let o = project(&[&layer.fc2], cfg.decoder_ffn_dim, &f, t).pop().unwrap();
            crate::kernels::add_inplace(&mut x, &o);
        }
        out
    }
}
