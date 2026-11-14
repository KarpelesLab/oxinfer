//! A stack of LFM conv / attention layers over one sequence held whole in f32: the audio
//! encoder (an LFM2 stack), the depth decoder and the two halves of the Liquid codec (LFM3
//! stacks). The language model itself runs on the Qwen code path (`crate::qwen`).
//!
//! ```text
//! layer:  x += [post_operator_norm](op(operator_norm(x)))
//!         x += [post_ffn_norm](w2(silu(w1(n)) * w3(n))),  n = ffn_norm(x)
//! ```
//!
//! LFM2 and LFM3 layers differ in names (`q_layernorm` / `out_proj` against `q_norm` / `o_proj`),
//! in the sandwich norms and in LFM3's sigmoid gate per head.

use crate::kernels::*;
use crate::lfm3::ShortConv;
use crate::model::{Lin, Quant, linear, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Result, bail};

/// Which keys a query sees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Span {
    /// The keys up to the query, the last `window` of them when given (the query's own included).
    Causal(Option<usize>),
    /// Every key (the convs stay causal; only attention-only stacks run this way).
    Bidirectional,
}

/// The shape of a stack.
#[derive(Clone, Debug)]
pub(crate) struct StackConfig {
    pub hidden: usize,
    /// `true` for a conv layer, `false` for an attention layer.
    pub conv: Vec<bool>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    /// Conv width (`conv_L_cache`).
    pub kernel: usize,
    pub eps: f32,
    /// LFM3's norms on the operator and MLP outputs.
    pub sandwich: bool,
    /// LFM3 names and the per-head gate; LFM2 otherwise.
    pub lfm3: bool,
    pub rope_theta: f64,
}

impl StackConfig {
    /// `layer_types` as conv flags (any attention kind is `false`).
    pub(crate) fn conv_flags(types: &[String]) -> Result<Vec<bool>> {
        types
            .iter()
            .map(|t| match t.as_str() {
                "conv" => Ok(true),
                "full_attention" | "sliding_attention" => Ok(false),
                t => bail!("LFM layer type {t:?} is not supported"),
            })
            .collect()
    }
}

struct Attn<'a> {
    q: Lin<'a>,
    k: Lin<'a>,
    v: Lin<'a>,
    o: Lin<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    gate: Option<Lin<'a>>,
}

#[allow(clippy::large_enum_variant)]
enum Op<'a> {
    Conv(ShortConv<'a>),
    Attn(Attn<'a>),
}

struct Layer<'a> {
    op: Op<'a>,
    op_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    post: Option<[Vec<f32>; 2]>,
    w1: Lin<'a>,
    w3: Lin<'a>,
    w2: Lin<'a>,
}

pub(crate) struct Stack<'a> {
    pub cfg: StackConfig,
    layers: Vec<Layer<'a>>,
    final_norm: Vec<f32>,
    inv_freq: Vec<f32>,
}

fn vec_of(files: &ModelFiles, name: &str, n: usize) -> Result<Vec<f32>> {
    let v = files.f32(name)?;
    if v.len() != n {
        bail!("tensor {name} has {} values, expected {n}", v.len());
    }
    Ok(v)
}

impl<'a> Stack<'a> {
    /// Loads the stack under `p` (`...lfm.` / `...model.`: `layers.N.` and `embedding_norm`).
    pub(crate) fn load(files: &'a ModelFiles, p: &str, cfg: StackConfig) -> Result<Self> {
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        let lin = |n: &str, o: usize, i: usize| load_lin(files, n, o, i, Quant::Bf16);
        let mut layers = Vec::with_capacity(cfg.conv.len());
        for (l, &conv) in cfg.conv.iter().enumerate() {
            let lp = format!("{p}layers.{l}.");
            let op = if conv {
                let cp = format!("{lp}conv.");
                Op::Conv(ShortConv {
                    in_proj: lin(&format!("{cp}in_proj.weight"), 3 * h, h)?,
                    out_proj: lin(&format!("{cp}out_proj.weight"), h, h)?,
                    conv_w: vec_of(files, &format!("{cp}conv.weight"), h * cfg.kernel)?,
                    kernel: cfg.kernel,
                    bias: None,
                })
            } else {
                let ap = format!("{lp}self_attn.");
                let (qn, kn, o) = if cfg.lfm3 {
                    ("q_norm", "k_norm", "o_proj")
                } else {
                    ("q_layernorm", "k_layernorm", "out_proj")
                };
                Op::Attn(Attn {
                    q: lin(&format!("{ap}q_proj.weight"), cfg.heads * hd, h)?,
                    k: lin(&format!("{ap}k_proj.weight"), cfg.kv_heads * hd, h)?,
                    v: lin(&format!("{ap}v_proj.weight"), cfg.kv_heads * hd, h)?,
                    o: lin(&format!("{ap}{o}.weight"), h, cfg.heads * hd)?,
                    q_norm: vec_of(files, &format!("{ap}{qn}.weight"), hd)?,
                    k_norm: vec_of(files, &format!("{ap}{kn}.weight"), hd)?,
                    gate: if cfg.lfm3 {
                        Some(lin(&format!("{ap}gate_proj.weight"), cfg.heads, h)?)
                    } else {
                        None
                    },
                })
            };
            let norm = |n: &str| vec_of(files, &format!("{lp}{n}.weight"), h);
            layers.push(Layer {
                op,
                op_norm: norm("operator_norm")?,
                ffn_norm: norm("ffn_norm")?,
                post: if cfg.sandwich {
                    Some([norm("post_operator_norm")?, norm("post_ffn_norm")?])
                } else {
                    None
                },
                w1: lin(&format!("{lp}feed_forward.w1.weight"), cfg.inter, h)?,
                w3: lin(&format!("{lp}feed_forward.w3.weight"), cfg.inter, h)?,
                w2: lin(&format!("{lp}feed_forward.w2.weight"), h, cfg.inter)?,
            });
        }
        let inv_freq = (0..hd / 2)
            .map(|i| (1.0 / cfg.rope_theta.powf((2 * i) as f64 / hd as f64)) as f32)
            .collect();
        Ok(Self {
            final_norm: vec_of(files, &format!("{p}embedding_norm.weight"), h)?,
            layers,
            inv_freq,
            cfg,
        })
    }

    /// Runs `t` rows of `x` (`[t, hidden]`, positions `0..t`) through the layers and the final
    /// norm, in place.
    pub(crate) fn forward(&self, x: &mut [f32], t: usize, span: Span) {
        let c = &self.cfg;
        let h = c.hidden;
        debug_assert_eq!(x.len(), t * h);
        let positions: Vec<usize> = (0..t).collect();
        let (cos, sin) = rope_tables(&positions, &self.inv_freq);
        let mut hn = vec![0f32; t * h];
        let mut y = vec![0f32; t * h];
        let mut a = vec![0f32; t * c.inter];
        let mut b = vec![0f32; t * c.inter];
        for layer in &self.layers {
            rmsnorm_f32w_rows(x, h, &layer.op_norm, c.eps, &mut hn);
            match &layer.op {
                Op::Conv(conv) => {
                    let mut state = vec![0f32; (c.kernel - 1) * h];
                    conv.forward(&mut state, &hn, t, h, &mut y, None);
                }
                Op::Attn(at) => self.attention(at, &hn, t, span, &cos, &sin, &mut y),
            }
            if let Some([post, _]) = &layer.post {
                rmsnorm_f32w_rows_inplace(&mut y, h, post, c.eps);
            }
            add_inplace(x, &y);
            rmsnorm_f32w_rows(x, h, &layer.ffn_norm, c.eps, &mut hn);
            linear(&layer.w1, c.inter, h, &hn, t, &mut a);
            linear(&layer.w3, c.inter, h, &hn, t, &mut b);
            silu_mul(&mut a, &b);
            linear(&layer.w2, h, c.inter, &a, t, &mut y);
            if let Some([_, post]) = &layer.post {
                rmsnorm_f32w_rows_inplace(&mut y, h, post, c.eps);
            }
            add_inplace(x, &y);
        }
        rmsnorm_f32w_rows_inplace(x, h, &self.final_norm, c.eps);
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(&self, at: &Attn, hn: &[f32], t: usize, span: Span, cos: &[f32], sin: &[f32], y: &mut [f32]) {
        let c = &self.cfg;
        let (h, nh, nkv, hd) = (c.hidden, c.heads, c.kv_heads, c.head_dim);
        let half = hd / 2;
        let mut q = vec![0f32; t * nh * hd];
        let mut k = vec![0f32; t * nkv * hd];
        let mut v = vec![0f32; t * nkv * hd];
        linear(&at.q, nh * hd, h, hn, t, &mut q);
        linear(&at.k, nkv * hd, h, hn, t, &mut k);
        linear(&at.v, nkv * hd, h, hn, t, &mut v);
        for (buf, n, w) in [(&mut q, nh, &at.q_norm), (&mut k, nkv, &at.k_norm)] {
            for_chunks(buf, hd, t * n > 64, |i, head| {
                rmsnorm_f32w_inplace(head, w, c.eps);
                let p = i / n;
                apply_rope(head, &cos[p * half..(p + 1) * half], &sin[p * half..(p + 1) * half]);
            });
        }
        let gate = at.gate.as_ref().map(|g| {
            let mut out = vec![0f32; t * nh];
            linear(g, nh, h, hn, t, &mut out);
            out
        });
        let scale = (hd as f32).powf(-0.5);
        let rep = nh / nkv;
        let mut out = vec![0f32; t * nh * hd];
        for_chunks(&mut out, nh * hd, t > 4, |i, row| {
            let (lo, hi) = match span {
                Span::Causal(None) => (0, i + 1),
                Span::Causal(Some(w)) => ((i + 1).saturating_sub(w), i + 1),
                Span::Bidirectional => (0, t),
            };
            let mut p = vec![0f32; hi - lo];
            for hh in 0..nh {
                let g = hh / rep;
                let qv = &q[(i * nh + hh) * hd..(i * nh + hh + 1) * hd];
                for (j, pj) in (lo..hi).zip(p.iter_mut()) {
                    *pj = scale * dot(qv, &k[(j * nkv + g) * hd..(j * nkv + g + 1) * hd]);
                }
                softmax_inplace(&mut p);
                let o = &mut row[hh * hd..(hh + 1) * hd];
                for (j, &pj) in (lo..hi).zip(p.iter()) {
                    axpy(o, pj, &v[(j * nkv + g) * hd..(j * nkv + g + 1) * hd]);
                }
                if let Some(gt) = &gate {
                    let s = 1.0 / (1.0 + (-gt[i * nh + hh]).exp());
                    scale_inplace(o, s);
                }
            }
        });
        linear(&at.o, h, nh * hd, &out, t, y);
    }
}
