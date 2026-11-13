//! The ZONOS3 language model (`general.architecture = zonos3`): 11 layers of width 1024, each a
//! sequence mixer and a feed-forward network joined by "Full AttnRes" instead of a residual
//! stream. A port of `zonos3.reference.lm.Zonos3LM` (Zyphra/ZONOS3-Mini, `model/src/zonos3/reference/lm.py`)
//! on the tensors of `tools/convert_gguf.py lm`.
//!
//! - Input rows hold one id per codebook plus a text id; their embeddings are summed, the speaker
//!   row and the condition rows replaced (speaker projection of the 192-d embedding, one learned
//!   row per condition bucket), then RMS-normalized without a weight.
//! - Full AttnRes: every sub-layer reads a mix of all earlier outputs (the embedding, then the
//!   mixer and the feed-forward output of every layer): a softmax over the sources of the dot
//!   products of their RMS-normalized values with a learned query, per row.
//! - Mixer: Gated DeltaNet (4 heads of 128, causal depthwise convolution of kernel 4 on q/k/v, the
//!   state `[heads, value, key]` in f32) in three layers of four, global attention in the fourth
//!   (16 heads of 32 over 4 key/value heads, RMS-normalized q and k with a per-head temperature,
//!   interleaved RoPE, a sigmoid gate per head).
//! - Feed-forward: SwiGLU in layer 0, then a latent mixture of experts: the row is projected to
//!   256 dimensions, the two experts with the largest router logits minus the balancing biases
//!   run on it (weighted by the softmax of their logits), and the sum is projected back.
//! - Output: RMS norm, one linear layer to `n_codebooks * audio_vocab` logits, soft-capped with
//!   `cap * tanh(x / cap)`.

use super::file::{Gg, Mat, mul_multi};
use anyhow::{Result, bail};

#[derive(Clone, Debug)]
pub struct LmConfig {
    pub n_layers: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub gdn_heads: usize,
    pub gdn_head_dim: usize,
    pub gdn_kernel: usize,
    pub n_codebooks: usize,
    pub codebook_size: usize,
    /// Rows of each codebook's embedding and outputs per codebook: the codes, EOA, padding.
    pub audio_vocab: usize,
    pub eoa_id: u32,
    pub audio_pad_id: u32,
    /// Text ids below this are tokens; the id itself is the text padding.
    pub text_vocab: usize,
    pub speaker_dim: usize,
    /// Buckets of each condition row; the id equal to the count means unknown.
    pub condition_buckets: Vec<usize>,
    pub condition_names: Vec<String>,
    pub condition_edges: Vec<Vec<f64>>,
    pub norm_eps: f32,
    pub qk_eps: f32,
    pub rope_theta: f32,
    pub softcap: f32,
    pub max_seqlen: usize,
    pub silence_frame: Vec<u32>,
}

struct Gdn {
    in_proj: Mat,
    /// `[3 * inner, kernel]`, the last tap on the current row.
    conv: Vec<f32>,
    norm: Vec<f32>,
    /// `exp(A_log)` per head.
    a: Vec<f32>,
    dt_bias: Vec<f32>,
    wo: Mat,
}

struct Attn {
    q: Mat,
    k: Mat,
    v: Mat,
    temp: Vec<f32>,
    gate: Mat,
    wo: Mat,
}

enum Mixer {
    Gdn(Box<Gdn>),
    Attn(Box<Attn>),
}

enum Ffn {
    Dense {
        up: Mat,
        gate: Mat,
        down: Mat,
    },
    Moe {
        router: Mat,
        router_norm: Vec<f32>,
        router_bias: Vec<f32>,
        latent_down: Mat,
        latent_up: Mat,
        gate: Vec<Mat>,
        up: Vec<Mat>,
        down: Vec<Mat>,
        topk: usize,
    },
}

struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    mixer: Mixer,
    ffn: Ffn,
}

pub struct Lm {
    pub cfg: LmConfig,
    audio_embd: Vec<Mat>,
    text_embd: Mat,
    layers: Vec<Layer>,
    out_norm: Vec<f32>,
    head: Mat,
    /// `[2 * n_layers + 1, dim]`.
    queries: Vec<f32>,
    spk_w: Mat,
    spk_b: Vec<f32>,
    cond: Vec<Vec<f32>>,
    /// The tokenizer JSON the file embeds.
    pub tokenizer_json: String,
    /// See [`Mat`]: f32 products on the dequantized weights instead of the int8 kernels.
    pub exact: bool,
    pub weight_bytes: usize,
}

/// The decoding state of one sequence.
#[derive(Clone, Default)]
pub struct LmState {
    /// Rows seen so far.
    pub pos: usize,
    /// Per layer: the rotated keys and the values `[rows, kv heads, head dim]` (attention layers).
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// Per layer: `[heads, value, key]` (GDN layers).
    gdn: Vec<Vec<f32>>,
    /// Per layer: the last `kernel - 1` rows of q/k/v before the convolution.
    conv: Vec<Vec<f32>>,
}

/// `x * rsqrt(mean(x^2) + eps)`, times `w` when given.
fn rms(x: &mut [f32], w: Option<&[f32]>, eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    match w {
        Some(w) => x.iter_mut().zip(w).for_each(|(v, w)| *v = *v * s * w),
        None => x.iter_mut().for_each(|v| *v *= s),
    }
}

fn rms_rows(x: &[f32], n: usize, w: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut y = x.to_vec();
    y.chunks_mut(n).for_each(|r| rms(r, w, eps));
    y
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// Exact softmax (the shared kernel uses an approximate exponential).
pub(crate) fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    x.iter_mut().for_each(|v| *v /= sum);
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    crate::kernels::dot(a, b)
}

/// Interleaved RoPE of one head at absolute position `pos`.
fn rope(x: &mut [f32], pos: usize, inv_freq: &[f32]) {
    for (p, f) in x.as_chunks_mut::<2>().0.iter_mut().zip(inv_freq) {
        let (s, c) = (pos as f32 * f).sin_cos();
        let (a, b) = (p[0], p[1]);
        p[0] = a * c - b * s;
        p[1] = a * s + b * c;
    }
}

/// The Full AttnRes sources of one forward pass, with the reciprocal RMS of every row.
struct Bank {
    d: usize,
    eps: f32,
    src: Vec<Vec<f32>>,
    inv: Vec<Vec<f32>>,
}

impl Bank {
    fn push(&mut self, x: Vec<f32>) {
        let inv = x
            .chunks(self.d)
            .map(|r| 1.0 / (r.iter().map(|v| v * v).sum::<f32>() / self.d as f32 + self.eps).sqrt())
            .collect();
        self.src.push(x);
        self.inv.push(inv);
    }

    /// Per row: the sources weighted by the softmax of `rms_norm(source) . query`.
    fn mix(&self, query: &[f32]) -> Vec<f32> {
        let (d, n) = (self.d, self.src.len());
        let t = self.src[0].len() / d;
        let mut out = vec![0f32; t * d];
        let mut w = vec![0f32; n];
        for (i, o) in out.chunks_mut(d).enumerate() {
            for s in 0..n {
                w[s] = dot(&self.src[s][i * d..(i + 1) * d], query) * self.inv[s][i];
            }
            softmax(&mut w);
            for s in 0..n {
                crate::kernels::axpy(o, w[s], &self.src[s][i * d..(i + 1) * d]);
            }
        }
        out
    }
}

impl Lm {
    pub(crate) fn load(g: &Gg, exact: bool) -> Result<Self> {
        let pattern = g.str("attention.pattern")?;
        let period = pattern
            .strip_prefix("gdn")
            .and_then(|p| p.strip_suffix("_global1"))
            .and_then(|n| n.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .map(|n| n + 1);
        let Some(period) = period else {
            bail!(
                "{}: attention pattern {pattern:?} is not supported (gdnN_global1)",
                g.path.display()
            );
        };
        if g.str("residual.type")? != "full_attnres" {
            bail!("{}: residual type {:?} is not supported", g.path.display(), g.str("residual.type")?);
        }
        let names = g.strs("condition.names")?;
        let cfg = LmConfig {
            n_layers: g.usize("block_count")?,
            dim: g.usize("embedding_length")?,
            n_heads: g.usize("attention.head_count")?,
            n_kv_heads: g.usize("attention.head_count_kv")?,
            head_dim: g.usize("head_dim")?,
            gdn_heads: g.usize("gdn.head_count")?,
            gdn_head_dim: g.usize("gdn.head_dim")?,
            gdn_kernel: g.usize("gdn.conv_kernel")?,
            n_codebooks: g.usize("n_codebooks")?,
            codebook_size: g.usize("codebook_size")?,
            audio_vocab: g.usize("audio_vocab")?,
            eoa_id: g.usize("eoa_id")? as u32,
            audio_pad_id: g.usize("audio_pad_id")? as u32,
            text_vocab: g.usize("text_vocab")?,
            speaker_dim: g.usize("speaker.embedding_length")?,
            condition_buckets: if g.has_meta("condition.bucket_counts") {
                g.usizes("condition.bucket_counts")?
            } else {
                vec![10; g.usize("condition.count")?]
            },
            condition_edges: names
                .iter()
                .map(|n| g.f64s(&format!("condition.edges.{n}")).unwrap_or_default())
                .collect(),
            condition_names: names,
            norm_eps: g.f64("attention.layer_norm_rms_epsilon")? as f32,
            qk_eps: g.f64("attention.qk_norm_epsilon")? as f32,
            rope_theta: g.f64("rope.freq_base")? as f32,
            softcap: g.f64("logit_softcap")? as f32,
            max_seqlen: g.usize("context_length")?,
            silence_frame: g
                .usizes("silence_frame")
                .unwrap_or_default()
                .into_iter()
                .map(|v| v as u32)
                .collect(),
        };
        let c = &cfg;
        if c.audio_vocab != c.codebook_size + 2 || c.eoa_id as usize != c.codebook_size || c.audio_pad_id as usize != c.codebook_size + 1 {
            bail!(
                "{}: audio vocabulary {} with EOA {} and padding {} (speech-edit checkpoints are not supported)",
                g.path.display(),
                c.audio_vocab,
                c.eoa_id,
                c.audio_pad_id
            );
        }
        if c.condition_buckets.len() != c.condition_names.len() || !c.n_heads.is_multiple_of(c.n_kv_heads) || !c.head_dim.is_multiple_of(2)
        {
            bail!("{}: inconsistent metadata", g.path.display());
        }
        let d = c.dim;
        let inner = c.gdn_heads * c.gdn_head_dim;
        let ffn = g.usize("feed_forward_length")?;
        let n_exp = g.usize("expert_count")?;
        let used = g.usizes("expert_used_count_per_layer")?;
        let bytes = std::cell::Cell::new(0usize);
        let mat = |name: &str, n_out: usize, n_in: usize| -> Result<Mat> {
            let m = g.mat(name, n_out, n_in)?;
            bytes.set(bytes.get() + m.bytes());
            Ok(m)
        };
        let audio_embd = (0..c.n_codebooks)
            .map(|i| mat(&format!("audio_embd.{i}.weight"), c.audio_vocab, d))
            .collect::<Result<Vec<_>>>()?;
        let text_embd = mat("text_embd.weight", c.text_vocab + 1, d)?;
        let mut layers = Vec::with_capacity(c.n_layers);
        for l in 0..c.n_layers {
            let p = format!("blk.{l}.");
            let mixer = if (l + 1) % period != 0 {
                let a_log = g.f32(&format!("{p}gdn_A_log"), &[c.gdn_heads])?;
                Mixer::Gdn(Box::new(Gdn {
                    in_proj: mat(&format!("{p}gdn_in.weight"), 4 * inner + 2 * c.gdn_heads, d)?,
                    conv: g.f32(&format!("{p}gdn_conv.weight"), &[3 * inner, c.gdn_kernel])?,
                    norm: g.f32(&format!("{p}gdn_norm.weight"), &[c.gdn_head_dim])?,
                    a: a_log.iter().map(|v| v.exp()).collect(),
                    dt_bias: g.f32(&format!("{p}gdn_dt_bias"), &[c.gdn_heads])?,
                    wo: mat(&format!("{p}attn_output.weight"), d, inner)?,
                }))
            } else {
                Mixer::Attn(Box::new(Attn {
                    q: mat(&format!("{p}attn_q.weight"), c.n_heads * c.head_dim, d)?,
                    k: mat(&format!("{p}attn_k.weight"), c.n_kv_heads * c.head_dim, d)?,
                    v: mat(&format!("{p}attn_v.weight"), c.n_kv_heads * c.head_dim, d)?,
                    temp: g.f32(&format!("{p}attn_temp"), &[c.n_heads])?,
                    gate: mat(&format!("{p}attn_gate.weight"), c.n_heads, d)?,
                    wo: mat(&format!("{p}attn_output.weight"), d, c.n_heads * c.head_dim)?,
                }))
            };
            let ffn = if g.has(&format!("{p}ffn_up.weight")) {
                Ffn::Dense {
                    up: mat(&format!("{p}ffn_up.weight"), ffn, d)?,
                    gate: mat(&format!("{p}ffn_gate.weight"), ffn, d)?,
                    down: mat(&format!("{p}ffn_down.weight"), d, ffn)?,
                }
            } else {
                let latent = g.shape(&format!("{p}ffn_latent_down.weight"))?[0];
                let hidden = g.shape(&format!("{p}ffn_gate_exps.weight"))?[1];
                let mats = |name: &str, n_out: usize, n_in: usize| -> Result<Vec<Mat>> {
                    let v = g.mats(&format!("{p}{name}"), n_exp, n_out, n_in)?;
                    bytes.set(bytes.get() + v.iter().map(Mat::bytes).sum::<usize>());
                    Ok(v)
                };
                let (gate, up, down) = (
                    mats("ffn_gate_exps.weight", hidden, latent)?,
                    mats("ffn_up_exps.weight", hidden, latent)?,
                    mats("ffn_down_exps.weight", latent, hidden)?,
                );
                Ffn::Moe {
                    router: mat(&format!("{p}router.weight"), n_exp, d)?,
                    router_norm: g.f32(&format!("{p}router_norm.weight"), &[d])?,
                    router_bias: g.f32(&format!("{p}router_bias"), &[n_exp])?,
                    latent_down: mat(&format!("{p}ffn_latent_down.weight"), latent, d)?,
                    latent_up: mat(&format!("{p}ffn_latent_up.weight"), d, latent)?,
                    gate,
                    up,
                    down,
                    topk: used.get(l).copied().unwrap_or(2).max(1),
                }
            };
            layers.push(Layer {
                attn_norm: g.f32(&format!("{p}attn_norm.weight"), &[d])?,
                ffn_norm: g.f32(&format!("{p}ffn_norm.weight"), &[d])?,
                mixer,
                ffn,
            });
        }
        let head = mat("output.weight", c.n_codebooks * c.audio_vocab, d)?;
        let spk_w = mat("speaker_proj.weight", d, c.speaker_dim)?;
        let cond = c
            .condition_buckets
            .iter()
            .enumerate()
            .map(|(i, n)| g.f32(&format!("cond_embd.{i}.weight"), &[n + 1, d]))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            audio_embd,
            text_embd,
            layers,
            out_norm: g.f32("output_norm.weight", &[d])?,
            head,
            queries: g.f32("attnres_q.weight", &[2 * c.n_layers + 1, d])?,
            spk_w,
            spk_b: g.f32("speaker_proj.bias", &[d])?,
            cond,
            tokenizer_json: g.str("tokenizer.json")?.to_string(),
            exact,
            weight_bytes: bytes.get(),
            cfg,
        })
    }

    pub fn new_state(&self) -> LmState {
        let n = self.cfg.n_layers;
        LmState {
            pos: 0,
            kv: vec![(Vec::new(), Vec::new()); n],
            gdn: vec![Vec::new(); n],
            conv: vec![Vec::new(); n],
        }
    }

    /// The logits of the last of `t` rows (`ids`: `[t, n_codebooks + 1]`) continuing `st`:
    /// `[n_codebooks, audio_vocab]`, soft-capped. `speaker`: the row (absolute position) whose
    /// embedding is the projected speaker embedding; `cond`: rows and their bucket ids, ascending,
    /// the i-th reading condition table i.
    pub fn forward(
        &self,
        st: &mut LmState,
        ids: &[u32],
        t: usize,
        speaker: Option<(usize, &[f32])>,
        cond: &[(usize, u32)],
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (d, w) = (c.dim, c.n_codebooks + 1);
        if ids.len() != t * w || t == 0 {
            bail!("zonos3: {} ids for {t} rows of {w}", ids.len());
        }
        let lo = st.pos;
        if lo + t > c.max_seqlen {
            bail!("zonos3: {} rows exceed the model context {}", lo + t, c.max_seqlen);
        }
        let mut x = vec![0f32; t * d];
        let mut row = vec![0f32; d];
        for (i, o) in x.chunks_mut(d).enumerate() {
            for cb in 0..w {
                let id = ids[i * w + cb] as usize;
                let m = if cb < c.n_codebooks {
                    &self.audio_embd[cb]
                } else {
                    &self.text_embd
                };
                if id >= m.n_out {
                    bail!("zonos3: id {id} in column {cb} is outside its vocabulary of {}", m.n_out);
                }
                m.row(id, &mut row);
                crate::kernels::add_inplace(o, &row);
            }
        }
        if let Some((r, e)) = speaker
            && (lo..lo + t).contains(&r)
        {
            if e.len() != c.speaker_dim {
                bail!("zonos3: speaker embedding of {} values, the model takes {}", e.len(), c.speaker_dim);
            }
            let mut y = self.spk_w.mul(e, 1, true);
            crate::kernels::add_inplace(&mut y, &self.spk_b);
            x[(r - lo) * d..(r - lo + 1) * d].copy_from_slice(&y);
        }
        for (table, &(r, b)) in cond.iter().enumerate() {
            if (lo..lo + t).contains(&r) {
                let tab = &self.cond[table];
                let b = b as usize;
                if (b + 1) * d > tab.len() {
                    bail!("zonos3: condition bucket {b} of row {table} is out of range");
                }
                x[(r - lo) * d..(r - lo + 1) * d].copy_from_slice(&tab[b * d..(b + 1) * d]);
            }
        }
        x.chunks_mut(d).for_each(|r| rms(r, None, c.norm_eps));
        let mut bank = Bank {
            d,
            eps: c.norm_eps,
            src: Vec::with_capacity(2 * c.n_layers + 1),
            inv: Vec::with_capacity(2 * c.n_layers + 1),
        };
        bank.push(x);
        let q = |i: usize| &self.queries[i * d..(i + 1) * d];
        for (l, layer) in self.layers.iter().enumerate() {
            let mut h = bank.mix(q(2 * l));
            h.chunks_mut(d).for_each(|r| rms(r, Some(&layer.attn_norm), c.norm_eps));
            let a = match &layer.mixer {
                Mixer::Gdn(g) => self.gdn(g, &h, t, l, st),
                Mixer::Attn(a) => self.attention(a, &h, t, l, st),
            };
            bank.push(a);
            let mut h = bank.mix(q(2 * l + 1));
            h.chunks_mut(d).for_each(|r| rms(r, Some(&layer.ffn_norm), c.norm_eps));
            bank.push(self.ffn(&layer.ffn, &h, t));
        }
        let h = bank.mix(q(2 * c.n_layers));
        let mut last = h[(t - 1) * d..].to_vec();
        rms(&mut last, Some(&self.out_norm), c.norm_eps);
        let mut logits = self.head.mul(&last, 1, self.exact);
        logits.iter_mut().for_each(|v| *v = c.softcap * (*v / c.softcap).tanh());
        st.pos = lo + t;
        Ok(logits)
    }

    fn gdn(&self, g: &Gdn, x: &[f32], t: usize, l: usize, st: &mut LmState) -> Vec<f32> {
        let c = &self.cfg;
        let (nh, hd, k) = (c.gdn_heads, c.gdn_head_dim, c.gdn_kernel);
        let inner = nh * hd;
        let ch = 3 * inner;
        let np = g.in_proj.n_out;
        let proj = g.in_proj.mul(x, t, self.exact);
        // The rows before the convolution: the kept history, then this pass.
        let hist = &mut st.conv[l];
        if hist.is_empty() {
            hist.resize((k - 1) * ch, 0.0);
        }
        let mut seq = std::mem::take(hist);
        for r in proj.chunks(np) {
            seq.extend_from_slice(&r[..ch]);
        }
        *hist = seq[t * ch..].to_vec();
        let state = &mut st.gdn[l];
        if state.is_empty() {
            state.resize(nh * hd * hd, 0.0);
        }
        let scale = (hd as f32).powf(-0.5);
        let mut gated = vec![0f32; t * inner];
        let mut conv = vec![0f32; ch];
        for i in 0..t {
            for (cidx, o) in conv.iter_mut().enumerate() {
                let mut s = 0f32;
                for j in 0..k {
                    s += g.conv[cidx * k + j] * seq[(i + j) * ch + cidx];
                }
                *o = silu(s);
            }
            let r = &proj[i * np..(i + 1) * np];
            let (z, ab) = r[ch..].split_at(inner);
            for h in 0..nh {
                let l2 = |v: &mut [f32]| {
                    let s = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                    v.iter_mut().for_each(|x| *x *= s);
                };
                let mut qh = conv[h * hd..(h + 1) * hd].to_vec();
                let mut kh = conv[inner + h * hd..inner + (h + 1) * hd].to_vec();
                let vh = &conv[2 * inner + h * hd..2 * inner + (h + 1) * hd];
                l2(&mut qh);
                l2(&mut kh);
                qh.iter_mut().for_each(|v| *v *= scale);
                let decay = (-g.a[h] * softplus(ab[h] + g.dt_bias[h])).exp();
                let beta = sigmoid(ab[nh + h]);
                let out = &mut gated[i * inner + h * hd..i * inner + (h + 1) * hd];
                for (vi, o) in out.iter_mut().enumerate() {
                    let srow = &mut state[(h * hd + vi) * hd..(h * hd + vi + 1) * hd];
                    crate::kernels::scale_inplace(srow, decay);
                    let delta = beta * (vh[vi] - dot(srow, &kh));
                    crate::kernels::axpy(srow, delta, &kh);
                    *o = dot(srow, &qh);
                }
                // RMS norm per head with its weight (eps 1e-5), gated by silu(z).
                rms(out, Some(&g.norm), 1e-5);
                for (o, zv) in out.iter_mut().zip(&z[h * hd..(h + 1) * hd]) {
                    *o *= silu(*zv);
                }
            }
        }
        g.wo.mul(&gated, t, self.exact)
    }

    fn attention(&self, a: &Attn, x: &[f32], t: usize, l: usize, st: &mut LmState) -> Vec<f32> {
        let c = &self.cfg;
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let (mut q, mut k, mut v, mut gate) = (
            vec![0f32; t * nh * hd],
            vec![0f32; t * nkv * hd],
            vec![0f32; t * nkv * hd],
            vec![0f32; t * nh],
        );
        mul_multi(
            &mut [(&a.q, &mut q[..]), (&a.k, &mut k[..]), (&a.v, &mut v[..]), (&a.gate, &mut gate[..])],
            x,
            t,
            self.exact,
        );
        let inv_freq: Vec<f32> = (0..hd / 2).map(|i| c.rope_theta.powf(-((2 * i) as f32) / hd as f32)).collect();
        let lo = st.pos;
        for i in 0..t {
            for h in 0..nh {
                let r = &mut q[(i * nh + h) * hd..(i * nh + h + 1) * hd];
                rms(r, None, c.qk_eps);
                r.iter_mut().for_each(|v| *v *= a.temp[h]);
                rope(r, lo + i, &inv_freq);
            }
            for h in 0..nkv {
                let r = &mut k[(i * nkv + h) * hd..(i * nkv + h + 1) * hd];
                rms(r, None, c.qk_eps);
                rope(r, lo + i, &inv_freq);
            }
        }
        let (ck, cv) = &mut st.kv[l];
        ck.extend_from_slice(&k);
        cv.extend_from_slice(&v);
        let rep = nh / nkv;
        let scale = (hd as f32).powf(-0.5);
        let mut out = vec![0f32; t * nh * hd];
        let mut w = Vec::new();
        for i in 0..t {
            let n = lo + i + 1;
            for h in 0..nh {
                let kvh = h / rep;
                let qr = &q[(i * nh + h) * hd..(i * nh + h + 1) * hd];
                w.clear();
                w.extend((0..n).map(|j| dot(qr, &ck[(j * nkv + kvh) * hd..(j * nkv + kvh + 1) * hd]) * scale));
                softmax(&mut w);
                let o = &mut out[(i * nh + h) * hd..(i * nh + h + 1) * hd];
                for (j, &p) in w.iter().enumerate() {
                    crate::kernels::axpy(o, p, &cv[(j * nkv + kvh) * hd..(j * nkv + kvh + 1) * hd]);
                }
                let gt = sigmoid(gate[i * nh + h]);
                o.iter_mut().for_each(|v| *v *= gt);
            }
        }
        a.wo.mul(&out, t, self.exact)
    }

    fn ffn(&self, f: &Ffn, x: &[f32], t: usize) -> Vec<f32> {
        let d = self.cfg.dim;
        match f {
            Ffn::Dense { up, gate, down } => {
                let (mut u, mut g) = (vec![0f32; t * up.n_out], vec![0f32; t * gate.n_out]);
                mul_multi(&mut [(up, &mut u[..]), (gate, &mut g[..])], x, t, self.exact);
                u.iter_mut().zip(&g).for_each(|(u, g)| *u *= silu(*g));
                down.mul(&u, t, self.exact)
            }
            Ffn::Moe {
                router,
                router_norm,
                router_bias,
                latent_down,
                latent_up,
                gate,
                up,
                down,
                topk,
            } => {
                let n_exp = router.n_out;
                let logits = router.mul(&rms_rows(x, d, Some(router_norm), self.cfg.norm_eps), t, self.exact);
                let latent = latent_down.mul(x, t, self.exact);
                let nl = latent_down.n_out;
                // (row, weight) of every expert's rows.
                let mut rows: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_exp];
                for (i, lg) in logits.chunks(n_exp).enumerate() {
                    let mut order: Vec<usize> = (0..n_exp).collect();
                    order.sort_by(|&a, &b| (lg[b] - router_bias[b]).total_cmp(&(lg[a] - router_bias[a])).then(a.cmp(&b)));
                    let mut w: Vec<f32> = order[..*topk].iter().map(|&e| lg[e]).collect();
                    softmax(&mut w);
                    for (&e, &w) in order[..*topk].iter().zip(&w) {
                        rows[e].push((i, w));
                    }
                }
                let mut out = vec![0f32; t * nl];
                for (e, rs) in rows.iter().enumerate() {
                    if rs.is_empty() {
                        continue;
                    }
                    let n = rs.len();
                    let mut xe = vec![0f32; n * nl];
                    for (o, &(i, _)) in xe.chunks_mut(nl).zip(rs) {
                        o.copy_from_slice(&latent[i * nl..(i + 1) * nl]);
                    }
                    let hid = gate[e].n_out;
                    let (mut g, mut u) = (vec![0f32; n * hid], vec![0f32; n * hid]);
                    mul_multi(&mut [(&gate[e], &mut g[..]), (&up[e], &mut u[..])], &xe, n, self.exact);
                    g.iter_mut().zip(&u).for_each(|(g, u)| *g = silu(*g) * u);
                    let y = down[e].mul(&g, n, self.exact);
                    for (r, &(i, w)) in y.chunks(nl).zip(rs) {
                        crate::kernels::axpy(&mut out[i * nl..(i + 1) * nl], w, r);
                    }
                }
                latent_up.mul(&out, t, self.exact)
            }
        }
    }
}
