//! The Step1X-Edit transformer (diffusers `Step1XEditTransformer2DModel`): the FLUX.1 MMDiT
//! (19 double-stream + 38 single-stream blocks, [`crate::flux::transformer`]'s blocks as they
//! are) fed by a *connector* instead of T5 + CLIP:
//!
//! - the connector refines the Qwen2.5-VL hidden states (`[n, 3584]`) with two
//!   timestep-modulated self-attention blocks (HunyuanVideo's "individual token refiner": the
//!   modulation is the timestep embedding plus a projection of the states' mean) into
//!   `[n, 4096]`, and projects the mean to the 768-wide vector FLUX takes from CLIP;
//! - v1.2 adds `text_token_mapping`: the raw token embeddings of quoted characters, projected,
//!   added to the refined states (it helps render quoted text);
//! - `context_embedder` maps the refined states to the DiT width (3072), the timestep and the
//!   vector go through `time_embed` / `vec_embed`; there is no guidance embedding.

use super::super::flux::rope::Rope;
use super::super::flux::transformer::{DoubleBlock, Layout, SingleBlock, ln_mod, timestep_proj};
use super::super::flux::weights::{Dense, Loader, forward_many};
use crate::kernels::for_chunks;
use crate::model::Quant;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use std::path::Path;

/// `transformer/config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub joint_dim: usize,
    pub timestep_in_dim: usize,
    pub vector_in_dim: usize,
    pub connector_in: usize,
    pub connector_hidden: usize,
    pub connector_heads: usize,
    pub connector_depth: usize,
    pub axes: Vec<usize>,
    pub theta: f64,
    /// v1.2: quoted characters' token embeddings are mapped and added to the text stream.
    pub text_token_mapping: bool,
}

impl Config {
    pub fn hidden(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let c: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        Self::from_json(&c)
    }

    pub fn from_json(c: &serde_json::Value) -> Result<Self> {
        let u = |k: &str, d: usize| c[k].as_u64().map(|v| v as usize).unwrap_or(d);
        if u("patch_size", 1) != 1 {
            bail!("patch_size {} is not supported", u("patch_size", 1));
        }
        if c["guidance_embeds"].as_bool().unwrap_or(false) {
            bail!("guidance-embedding Step1X transformers are not supported");
        }
        let in_channels = u("in_channels", 64);
        Ok(Self {
            in_channels,
            out_channels: c["out_channels"].as_u64().map(|v| v as usize).unwrap_or(in_channels),
            num_layers: u("num_layers", 19),
            num_single_layers: u("num_single_layers", 38),
            heads: u("num_attention_heads", 24),
            head_dim: u("attention_head_dim", 128),
            joint_dim: u("joint_attention_dim", 4096),
            timestep_in_dim: u("timestep_in_dim", 256),
            vector_in_dim: u("vector_in_dim", 768),
            connector_in: u("connector_in_channels", 3584),
            connector_hidden: u("connector_hidden_size", 4096),
            connector_heads: u("connector_heads_num", 32),
            connector_depth: u("connector_depth", 2),
            axes: c["axes_dims_rope"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect())
                .unwrap_or_else(|| vec![16, 56, 56]),
            theta: 10000.0,
            text_token_mapping: c["enable_text_token_mapping"].as_bool().unwrap_or(false),
        })
    }
}

/// LayerNorm with affine parameters (eps 1e-6) of `rows` rows.
fn layer_norm(x: &[f32], rows: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let d = w.len();
    let mut y = vec![0f32; rows * d];
    for_chunks(&mut y, d, rows > 16, |r, out| {
        let row = &x[r * d..(r + 1) * d];
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
        let inv = (1.0 / (var + 1e-6).sqrt()) as f32;
        let m = mean as f32;
        for i in 0..d {
            out[i] = (row[i] - m) * inv * w[i] + b[i];
        }
    });
    y
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn silu_inplace(x: &mut [f32]) {
    for_chunks(x, 1 << 16, x.len() > 1 << 16, |_, c| c.iter_mut().for_each(|v| *v = silu(*v)));
}

/// `Linear -> SiLU -> Linear` on one vector (the embedders).
fn mlp_vec(l: &[Dense; 2], x: &[f32]) -> Vec<f32> {
    let mut h = l[0].forward_vec(x);
    h.iter_mut().for_each(|v| *v = silu(*v));
    l[1].forward_vec(&h)
}

/// `x += gate * y` per row.
fn gated_add(x: &mut [f32], y: &[f32], gate: &[f32]) {
    let d = gate.len();
    for_chunks(x, d, x.len() > d * 16, |r, row| {
        for ((a, &b), &g) in row.iter_mut().zip(&y[r * d..(r + 1) * d]).zip(gate) {
            *a += g * b;
        }
    });
}

/// One block of the connector's token refiner (`Step1XEditIndividualTokenRefinerBlock`):
/// gated self-attention and SiLU MLP, gates from the conditioning vector.
struct RefinerBlock<'a> {
    norm1: (Vec<f32>, Vec<f32>),
    qkv: Dense<'a>,
    proj: Dense<'a>,
    norm2: (Vec<f32>, Vec<f32>),
    fc1: Dense<'a>,
    fc2: Dense<'a>,
    ada: Dense<'a>,
}

impl<'a> RefinerBlock<'a> {
    fn load(ld: &Loader<'a>, p: &str) -> Result<Self> {
        let ln = |n: &str| -> Result<(Vec<f32>, Vec<f32>)> { Ok((ld.f32(&format!("{p}.{n}.weight"))?, ld.f32(&format!("{p}.{n}.bias"))?)) };
        Ok(Self {
            norm1: ln("norm1")?,
            qkv: ld.dense(&format!("{p}.self_attn_qkv"), true)?,
            proj: ld.dense(&format!("{p}.self_attn_proj"), true)?,
            norm2: ln("norm2")?,
            fc1: ld.dense(&format!("{p}.mlp.fc1"), true)?,
            fc2: ld.dense(&format!("{p}.mlp.fc2"), true)?,
            ada: ld.dense(&format!("{p}.adaLN_modulation.1"), false)?,
        })
    }

    fn lins(&self) -> Vec<&Dense<'a>> {
        vec![&self.qkv, &self.proj, &self.fc1, &self.fc2, &self.ada]
    }

    /// `x` is `[n, d]` of which the first `valid` rows are tokens (the rest padding: it attends
    /// to token 0 only, as the reference's mask); `sc` is SiLU of the conditioning vector.
    fn forward(&self, x: &mut [f32], n: usize, valid: usize, sc: &[f32], heads: usize) {
        let d = self.norm1.0.len();
        let g = self.ada.forward_vec(sc);
        let (gate_msa, gate_mlp) = g.split_at(d);
        let h = layer_norm(x, n, &self.norm1.0, &self.norm1.1);
        let qkv = self.qkv.forward(&h, n);
        drop(h);
        // [n, 3, heads, dh] -> q, k, v as [n, heads * dh].
        let (mut q, mut k, mut v) = (vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d]);
        for t in 0..n {
            let r = &qkv[t * 3 * d..(t + 1) * 3 * d];
            q[t * d..(t + 1) * d].copy_from_slice(&r[..d]);
            k[t * d..(t + 1) * d].copy_from_slice(&r[d..2 * d]);
            v[t * d..(t + 1) * d].copy_from_slice(&r[2 * d..]);
        }
        drop(qkv);
        let mut a = crate::sd::ops::attention(
            &q[..valid * d],
            &k[..valid * d],
            &v[..valid * d],
            valid,
            valid,
            heads,
            d / heads,
            false,
        );
        for _ in valid..n {
            a.extend_from_slice(&v[..d]);
        }
        drop((q, k, v));
        gated_add(x, &self.proj.forward(&a, n), gate_msa);
        let h = layer_norm(x, n, &self.norm2.0, &self.norm2.1);
        let mut m = self.fc1.forward(&h, n);
        drop(h);
        silu_inplace(&mut m);
        gated_add(x, &self.fc2.forward(&m, n), gate_mlp);
    }
}

/// The connector (`Step1XEditConnector`): refines the VL hidden states into the text stream and
/// pools them into the conditioning vector.
pub struct Connector<'a> {
    input_embedder: Dense<'a>,
    t_mlp: [Dense<'a>; 2],
    c_mlp: [Dense<'a>; 2],
    blocks: Vec<RefinerBlock<'a>>,
    global_proj: Dense<'a>,
    heads: usize,
}

/// A prompt's conditioning prepared for the transformer: everything that does not depend on the
/// timestep.
pub struct Cond {
    /// `input_embedder(x)`: `[n, connector_hidden]`.
    x0: Vec<f32>,
    /// `c_embedder(mean(x))`: `[connector_hidden]`.
    c_ctx: Vec<f32>,
    /// `global_proj_out(mean(x))`: the pooled vector `[vector_in_dim]`.
    pub pooled: Vec<f32>,
    /// `text_token_mapping(embeddings) * mask` (v1.2): `[n, connector_hidden]`.
    mapped: Option<Vec<f32>>,
    pub n: usize,
    /// Tokens before the padding (v1.1 pads to 640).
    pub valid: usize,
}

impl<'a> Connector<'a> {
    fn load(ld: &Loader<'a>, cfg: &Config) -> Result<Self> {
        let p = "connector.S";
        let d = |n: &str| ld.dense(n, true);
        Ok(Self {
            input_embedder: d(&format!("{p}.input_embedder"))?,
            t_mlp: [
                ld.dense(&format!("{p}.t_embedder.mlp.0"), false)?,
                ld.dense(&format!("{p}.t_embedder.mlp.2"), false)?,
            ],
            c_mlp: [
                ld.dense(&format!("{p}.c_embedder.linear_1"), false)?,
                ld.dense(&format!("{p}.c_embedder.linear_2"), false)?,
            ],
            blocks: (0..cfg.connector_depth)
                .map(|i| RefinerBlock::load(ld, &format!("{p}.individual_token_refiner.blocks.{i}")))
                .collect::<Result<_>>()?,
            global_proj: ld.dense("connector.global_proj_out", false)?,
            heads: cfg.connector_heads,
        })
    }

    fn lins(&self) -> Vec<&Dense<'a>> {
        let mut v = vec![&self.input_embedder, &self.global_proj];
        v.extend(self.t_mlp.iter().chain(&self.c_mlp));
        for b in &self.blocks {
            v.extend(b.lins());
        }
        v
    }
}

/// The Step1X-Edit transformer (see the module docs).
pub struct Transformer {
    pub cfg: Config,
    connector: Connector<'static>,
    text_map: Option<Dense<'static>>,
    time: [Dense<'static>; 2],
    vec: [Dense<'static>; 2],
    ctx_emb: Dense<'static>,
    x_emb: Dense<'static>,
    pub doubles: Vec<DoubleBlock<'static>>,
    pub singles: Vec<SingleBlock<'static>>,
    norm_out: [Dense<'static>; 2],
    proj_out: Dense<'static>,
    /// Weight streaming: unit `i` is double block `i`, then the single blocks.
    stream: Option<crate::stream::StreamHook>,
    /// Bytes of weights held (borrowed from the mapping or quantized copies).
    pub weight_bytes: usize,
    // Declared last: dropped after every layer borrowing it.
    _files: Box<ModelFiles>,
}

/// `OXINFER_STEP1X_LAYERS=D,S`: only the first D double and S single blocks (reference checks
/// against a truncated model).
fn layer_limit() -> Option<(usize, usize)> {
    let v = std::env::var("OXINFER_STEP1X_LAYERS").ok()?;
    let (a, b) = v.split_once(',')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

impl Transformer {
    /// Loads a diffusers `transformer/` folder (or a single safetensors / GGUF file with the
    /// diffusers names and a `config` next to it). `in_memory` reads the weights into RAM instead
    /// of mapping them; float weights are quantized to `quant` (bf16 runs them as stored).
    pub fn load(path: &Path, cfg: Config, quant: Quant, in_memory: bool) -> Result<Self> {
        let files = Box::new(ModelFiles::open(path, in_memory).with_context(|| format!("transformer weights {}", path.display()))?);
        // SAFETY: the box's heap block does not move, and every layer borrowing it is dropped
        // before `_files` (field order).
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        if !fs.has("connector.global_proj_out.weight") {
            bail!("{}: not a Step1X-Edit transformer (no connector)", path.display());
        }
        let ld = Loader { files: fs, quant };
        let mut cfg = cfg;
        if let Some((d, s)) = layer_limit() {
            cfg.num_layers = cfg.num_layers.min(d);
            cfg.num_single_layers = cfg.num_single_layers.min(s);
        }
        let d = cfg.hidden();
        let top = |n: &str| ld.dense(n, false);
        let connector = Connector::load(&ld, &cfg)?;
        let text_map = if cfg.text_token_mapping {
            Some(ld.dense("text_token_mapping", true)?)
        } else {
            None
        };
        let time = [top("time_embed.in_layer")?, top("time_embed.out_layer")?];
        let vec = [top("vec_embed.in_layer")?, top("vec_embed.out_layer")?];
        let ctx_emb = top("context_embedder")?;
        let x_emb = top("x_embedder")?;
        let norm_out = {
            let mut v = ld.split("norm_out.linear", &[d, d], false)?;
            let (shift, scale) = (v.pop().unwrap(), v.pop().unwrap());
            // diffusers' AdaLayerNormContinuous stores (scale, shift).
            [scale, shift]
        };
        let proj_out = top("proj_out")?;
        let (h, hd) = (cfg.heads, cfg.head_dim);
        let doubles = (0..cfg.num_layers)
            .map(|i| DoubleBlock::load(&ld, Layout::Diffusers, i, h, hd))
            .collect::<Result<Vec<_>>>()?;
        let singles = (0..cfg.num_single_layers)
            .map(|i| SingleBlock::load(&ld, Layout::Diffusers, i, h, hd))
            .collect::<Result<Vec<_>>>()?;
        let mut t = Self {
            cfg,
            connector,
            text_map,
            time,
            vec,
            ctx_emb,
            x_emb,
            doubles,
            singles,
            norm_out,
            proj_out,
            stream: None,
            weight_bytes: 0,
            _files: files,
        };
        t.weight_bytes = t.all_lins().iter().map(|l| l.bytes()).sum();
        Ok(t)
    }

    fn all_lins(&self) -> Vec<&Dense<'static>> {
        let mut v = self.connector.lins();
        v.extend(self.text_map.iter());
        v.extend(self.time.iter().chain(&self.vec).chain(&self.norm_out));
        v.extend([&self.ctx_emb, &self.x_emb, &self.proj_out]);
        for b in &self.doubles {
            v.extend(b.lins());
        }
        for b in &self.singles {
            v.extend(b.lins());
        }
        v
    }

    /// The checkpoint (for streaming).
    pub fn files(&self) -> &ModelFiles {
        &self._files
    }

    /// Streaming units (the DiT blocks in execution order) and their weight memory.
    pub fn stream_units(&self) -> Vec<crate::stream::UnitSpec> {
        let unit = |name: String, lins: Vec<&Dense>| crate::stream::UnitSpec {
            name,
            ranges: lins.iter().flat_map(|d| d.w.mem_ranges()).collect(),
            uses: 1.0,
            parts: Vec::new(),
        };
        let mut u: Vec<_> = self
            .doubles
            .iter()
            .enumerate()
            .map(|(i, b)| unit(format!("double.{i}"), b.lins()))
            .collect();
        u.extend(self.singles.iter().enumerate().map(|(i, b)| unit(format!("single.{i}"), b.lins())));
        u
    }

    /// Marks the weights of the units with `streamed[i]` as streamed (no device copies kept).
    pub fn mark_streamed(&self, streamed: &[bool]) {
        let blocks = self.doubles.iter().map(|b| b.lins()).chain(self.singles.iter().map(|b| b.lins()));
        for (lins, &s) in blocks.zip(streamed) {
            for d in lins {
                d.w.set_transient_device(s);
            }
        }
    }

    pub fn set_stream(&mut self, hook: Option<crate::stream::StreamHook>) {
        self.stream = hook;
    }

    /// The timestep-independent part of the connector for one prompt: `hidden` is `[n,
    /// connector_in]` (the VL's last hidden states), `embeds` the raw token embeddings of the
    /// same tokens and `mask` which of them are quoted characters (v1.2's text token mapping);
    /// rows from `valid` on are padding.
    pub fn prepare(&self, hidden: &[f32], n: usize, valid: usize, embeds: &[f32], mask: &[bool]) -> Cond {
        let ci = self.cfg.connector_in;
        assert_eq!(hidden.len(), n * ci, "hidden states size");
        assert!(valid > 0 && valid <= n, "valid tokens");
        let mut mean = vec![0f64; ci];
        for r in hidden.chunks(ci).take(valid) {
            for (m, &v) in mean.iter_mut().zip(r) {
                *m += v as f64;
            }
        }
        let mean: Vec<f32> = mean.iter().map(|&m| (m / valid as f64) as f32).collect();
        let c = &self.connector;
        let mapped = self.text_map.as_ref().filter(|_| mask.iter().any(|&m| m)).map(|l| {
            let mut y = l.forward(embeds, n);
            let w = l.n_out;
            for (r, &m) in mask.iter().enumerate() {
                if !m {
                    y[r * w..(r + 1) * w].fill(0.0);
                }
            }
            y
        });
        Cond {
            x0: c.input_embedder.forward(hidden, n),
            c_ctx: mlp_vec(&c.c_mlp, &mean),
            pooled: c.global_proj.forward_vec(&mean),
            mapped,
            n,
            valid,
        }
    }

    /// The text stream for timestep `t` (`sigma * 1000`): the refined states through
    /// `context_embedder`, `[n, hidden]`.
    pub fn text_stream(&self, cond: &Cond, t: f32) -> Vec<f32> {
        let c = &self.connector;
        let mut cv = mlp_vec(&c.t_mlp, &timestep_proj(t, 256));
        for (a, &b) in cv.iter_mut().zip(&cond.c_ctx) {
            *a += b;
        }
        let sc: Vec<f32> = cv.iter().map(|&v| silu(v)).collect();
        let mut x = cond.x0.clone();
        for b in &c.blocks {
            b.forward(&mut x, cond.n, cond.valid, &sc, c.heads);
        }
        if let Some(m) = &cond.mapped {
            for (a, &b) in x.iter_mut().zip(m) {
                *a += b;
            }
        }
        self.ctx_emb.forward(&x, cond.n)
    }

    /// One denoising call. `img` is `[n_img, in_channels]` (the noisy latent's packed tokens,
    /// then the reference image's), `cond` the prompt, `sigma` the noise level, `rope` covering
    /// `cond.n + n_img` tokens (text first). Returns `[n_img, out_channels]`.
    pub fn forward(&self, img: &[f32], n_img: usize, cond: &Cond, sigma: f32, rope: &Rope) -> Vec<f32> {
        let d = self.cfg.hidden();
        let n_txt = cond.n;
        let nt = n_txt + n_img;
        assert_eq!(rope.rows, nt, "rope table rows");
        let t = sigma * 1000.0;
        let mut ht = self.text_stream(cond, t);
        let mut temb = mlp_vec(&self.time, &timestep_proj(t, self.cfg.timestep_in_dim));
        for (a, b) in temb.iter_mut().zip(mlp_vec(&self.vec, &cond.pooled)) {
            *a += b;
        }
        let st: Vec<f32> = temb.iter().map(|&v| silu(v)).collect();
        super::dump_once("temb.bin", &temb);
        super::dump_once("pooled.bin", &cond.pooled);
        super::dump_once("txt_stream.bin", &ht);
        let mut hi = self.x_emb.forward(img, n_img);
        super::dump_once("x_emb.bin", &hi);
        super::dump_once("x_in.bin", img);
        for (i, b) in self.doubles.iter().enumerate() {
            let _unit = crate::stream::enter(&self.stream, i);
            b.forward(&mut hi, n_img, &mut ht, n_txt, &st, rope);
        }
        let mut h = ht;
        h.extend_from_slice(&hi);
        drop(hi);
        let nd = self.doubles.len();
        for (i, b) in self.singles.iter().enumerate() {
            let _unit = crate::stream::enter(&self.stream, nd + i);
            b.forward(&mut h, nt, &st, rope);
        }
        let hi = &h[n_txt * d..];
        let m = forward_many(&[&self.norm_out[0], &self.norm_out[1]], &st, 1);
        let x = ln_mod(hi, n_img, &m[0], &m[1]);
        self.proj_out.forward(&x, n_img)
    }
}
