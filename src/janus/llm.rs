//! Janus's language model: a LLaMA decoder (DeepSeek-LLM 1.5B / 7B; RMSNorm, rotate-half RoPE,
//! SwiGLU MLP, untied LM head) run over a batch of sequences of equal length. Every matmul takes
//! all the batch's rows at once, so the weights are read once per step for the conditional and
//! unconditional sequences of classifier-free guidance (and for several images at a time).

use crate::kernels::{apply_rope, axpy, dot, rmsnorm_rows, rope_inv_freq, rope_tables, silu_mul, softmax_inplace};
use crate::model::{LayerCache, Lin, Quant, Table, linear, linear_multi, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

/// The `language_config` of a Janus checkpoint, with `transformers.LlamaConfig`'s defaults for
/// what it leaves out (the 7B config lists only the layer count and vocabulary).
#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
}

impl LlamaConfig {
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let get = |k: &str, d: usize| v.get(k).and_then(|x| x.as_u64()).map_or(d, |x| x as usize);
        let hidden_size = get("hidden_size", 4096);
        let num_attention_heads = get("num_attention_heads", 32);
        let cfg = Self {
            hidden_size,
            intermediate_size: get("intermediate_size", 11008),
            num_hidden_layers: get("num_hidden_layers", 32),
            num_attention_heads,
            num_key_value_heads: get("num_key_value_heads", num_attention_heads),
            head_dim: get("head_dim", hidden_size / num_attention_heads.max(1)),
            vocab_size: get("vocab_size", 32000),
            rms_norm_eps: v.get("rms_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-6) as f32,
            rope_theta: v.get("rope_theta").and_then(|x| x.as_f64()).unwrap_or(10000.0),
        };
        if v.get("rope_scaling").is_some_and(|s| !s.is_null()) {
            bail!("language_config: rope_scaling is not supported");
        }
        if cfg.num_attention_heads == 0 || !cfg.num_attention_heads.is_multiple_of(cfg.num_key_value_heads.max(1)) {
            bail!(
                "language_config: {} heads for {} KV heads",
                cfg.num_attention_heads,
                cfg.num_key_value_heads
            );
        }
        Ok(cfg)
    }
}

struct Layer<'a> {
    ln_in: &'a [u16],
    ln_post: &'a [u16],
    q: Lin<'a>,
    k: Lin<'a>,
    v: Lin<'a>,
    o: Lin<'a>,
    gate: Lin<'a>,
    up: Lin<'a>,
    down: Lin<'a>,
}

pub struct Llm<'a> {
    pub cfg: LlamaConfig,
    embed: Table<'a>,
    layers: Vec<Layer<'a>>,
    norm: &'a [u16],
    /// The text LM head (for image understanding and chat; image generation uses `gen_head`).
    lm_head: Lin<'a>,
    inv_freq: Vec<f32>,
}

/// The KV caches of a batch of sequences that advance together (same length).
pub struct BatchState {
    /// `seqs[b][layer]`.
    seqs: Vec<Vec<LayerCache>>,
    pub pos: usize,
}

impl BatchState {
    pub fn batch(&self) -> usize {
        self.seqs.len()
    }
}

impl<'a> Llm<'a> {
    /// Loads the decoder under `prefix` (`language_model.`), quantizing the linear weights to
    /// `quant` (embeddings stay bf16).
    pub fn load(cfg: LlamaConfig, files: &'a ModelFiles, prefix: &str, quant: Quant) -> Result<Self> {
        let (h, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let (nh, nkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let norm = |n: &str| -> Result<&'a [u16]> {
            let w = files.bf16(n)?;
            if w.len() != h {
                bail!("{n}: {} values, expected {h}", w.len());
            }
            Ok(w)
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("{prefix}model.layers.{i}");
            let lin = |n: &str, o: usize, k: usize| load_lin(files, &format!("{p}.{n}.weight"), o, k, quant);
            layers.push(Layer {
                ln_in: norm(&format!("{p}.input_layernorm.weight"))?,
                ln_post: norm(&format!("{p}.post_attention_layernorm.weight"))?,
                q: lin("self_attn.q_proj", nh * hd, h)?,
                k: lin("self_attn.k_proj", nkv * hd, h)?,
                v: lin("self_attn.v_proj", nkv * hd, h)?,
                o: lin("self_attn.o_proj", h, nh * hd)?,
                gate: lin("mlp.gate_proj", inter, h)?,
                up: lin("mlp.up_proj", inter, h)?,
                down: lin("mlp.down_proj", h, inter)?,
            });
        }
        let embed = Table::load(files, &format!("{prefix}model.embed_tokens.weight"), cfg.vocab_size, h, Quant::Bf16)
            .context("token embeddings")?;
        let lm_head = load_lin(files, &format!("{prefix}lm_head.weight"), cfg.vocab_size, h, quant)?;
        Ok(Self {
            inv_freq: rope_inv_freq(hd, cfg.rope_theta, 1.0),
            norm: norm(&format!("{prefix}model.norm.weight"))?,
            cfg,
            embed,
            layers,
            lm_head,
        })
    }

    pub fn new_state(&self, batch: usize) -> BatchState {
        let c = &self.cfg;
        BatchState {
            seqs: (0..batch)
                .map(|_| {
                    (0..c.num_hidden_layers)
                        .map(|_| LayerCache::new(c.num_key_value_heads, c.head_dim))
                        .collect()
                })
                .collect(),
            pos: 0,
        }
    }

    /// Token embeddings `[ids.len(), hidden]`.
    pub fn embed(&self, ids: &[u32]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let mut x = vec![0f32; ids.len() * h];
        for (r, &id) in x.chunks_mut(h).zip(ids) {
            self.embed.row(id as usize, r);
        }
        x
    }

    /// Text logits `[rows, vocab]` of final-normed hidden states.
    pub fn logits(&self, hidden: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows * self.cfg.vocab_size];
        linear(&self.lm_head, self.cfg.vocab_size, self.cfg.hidden_size, hidden, rows, &mut y);
        y
    }

    /// Runs `t` new positions of every sequence of `st`: `x` holds the input embeddings
    /// `[batch * t, hidden]`, sequence by sequence. Returns the final-normed hidden states of
    /// the same shape.
    pub fn forward(&self, st: &mut BatchState, mut x: Vec<f32>, t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let (h, inter) = (c.hidden_size, c.intermediate_size);
        let (nh, nkv, hd) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let b = st.batch();
        let rows = b * t;
        assert_eq!(x.len(), rows * h, "forward: input is not [batch * t, hidden]");
        let start = st.pos;
        let positions: Vec<usize> = (start..start + t).collect();
        let (cos, sin) = rope_tables(&positions, &self.inv_freq);
        let half = self.inv_freq.len();
        let scale = (hd as f32).powf(-0.5);
        let mut hn = vec![0f32; rows * h];
        let mut q = vec![0f32; rows * nh * hd];
        let mut k = vec![0f32; rows * nkv * hd];
        let mut v = vec![0f32; rows * nkv * hd];
        let mut att = vec![0f32; rows * nh * hd];
        let mut y = vec![0f32; rows * h];
        let mut g = vec![0f32; rows * inter];
        let mut u = vec![0f32; rows * inter];
        for (li, l) in self.layers.iter().enumerate() {
            rmsnorm_rows(&x, h, Some(l.ln_in), c.rms_norm_eps, &mut hn);
            linear_multi(
                &mut [
                    (&l.q, nh * hd, &mut q[..]),
                    (&l.k, nkv * hd, &mut k[..]),
                    (&l.v, nkv * hd, &mut v[..]),
                ],
                h,
                &hn,
                rows,
            );
            // RoPE on queries (with the 1/sqrt(hd) scaling folded in) and keys.
            let rope = |m: &mut [f32], heads: usize, s: f32| {
                crate::kernels::for_chunks(m, hd, rows * heads > 64, |i, r| {
                    let ti = (i / heads) % t;
                    apply_rope(r, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
                    if s != 1.0 {
                        r.iter_mut().for_each(|x| *x *= s);
                    }
                });
            };
            rope(&mut q, nh, scale);
            rope(&mut k, nkv, 1.0);
            let kv = nkv * hd;
            for (bi, seq) in st.seqs.iter_mut().enumerate() {
                seq[li].push(&k[bi * t * kv..(bi + 1) * t * kv], &v[bi * t * kv..(bi + 1) * t * kv], t);
            }
            let caches: Vec<&LayerCache> = st.seqs.iter().map(|s| &s[li]).collect();
            attention_batch(&q, &caches, nh, t, start, &mut att);
            linear(&l.o, h, nh * hd, &att, rows, &mut y);
            crate::kernels::add_inplace(&mut x, &y);
            rmsnorm_rows(&x, h, Some(l.ln_post), c.rms_norm_eps, &mut hn);
            linear_multi(&mut [(&l.gate, inter, &mut g[..]), (&l.up, inter, &mut u[..])], h, &hn, rows);
            silu_mul(&mut g, &u);
            linear(&l.down, h, inter, &g, rows, &mut y);
            crate::kernels::add_inplace(&mut x, &y);
        }
        st.pos += t;
        rmsnorm_rows(&x, h, Some(self.norm), c.rms_norm_eps, &mut hn);
        hn
    }
}

/// Causal attention for a batch: `q` is `[batch * t, nh, hd]` (rotated and scaled), `caches[b]`
/// sequence b's keys and values (already holding the `t` new positions). One task per
/// (sequence, position, head), all in one parallel region.
fn attention_batch(q: &[f32], caches: &[&LayerCache], nh: usize, t: usize, start: usize, out: &mut [f32]) {
    let _timer = crate::prof::start(crate::prof::Cat::Attention);
    let (hd, nkv) = (caches[0].hd, caches[0].n_kv);
    let rep = nh / nkv;
    let ntasks = caches.len() * t * nh;
    let op = SendPtr(out.as_mut_ptr());
    let run = |task: usize| {
        let (bi, rem) = (task / (t * nh), task % (t * nh));
        let (ti, hh) = (rem / nh, rem % nh);
        let cache = caches[bi];
        let n = start + ti + 1;
        let qv = &q[task * hd..(task + 1) * hd];
        let mut scores: Vec<f32> = (0..n).map(|j| dot(qv, cache.k_at(j, hh / rep))).collect();
        softmax_inplace(&mut scores);
        // SAFETY: each task writes its own [hd] slice of `out`.
        let o = unsafe { std::slice::from_raw_parts_mut(op.get().add(task * hd), hd) };
        o.fill(0.0);
        for (j, &p) in scores.iter().enumerate() {
            axpy(o, p, cache.v_at(j, hh / rep));
        }
    };
    if ntasks > 1 && (start + t) * ntasks > 256 {
        crate::pool::run(ntasks, &run);
    } else {
        (0..ntasks).for_each(run);
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}
