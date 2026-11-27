//! LTX-2.5's audio-video transformer (`ltx_core.model.transformer.model.LTXModel`, the 22B
//! `AVTransformer3DModel` of the release's `diffusion_models/` files), on the CPU.
//!
//! Two streams, video (4096 wide, 32 heads of 128) and audio (2048, 32 heads of 64), through 48
//! blocks. Each block (`BasicAVTransformerBlock`), per stream:
//!
//! 1. RMS norm (no weight), AdaLN shift / scale, self-attention with RoPE, gated residual;
//! 2. RMS norm, AdaLN shift / scale of the query, cross-attention to the text context (whose rows
//!    get a shift / scale of their own from the prompt AdaLN), gated residual;
//! 3. audio-to-video and video-to-audio cross-attention (each stream's queries over the other's
//!    rows as they were after step 2, both RMS-normed and modulated by the "AV cross" AdaLN, the
//!    gate driven by the other stream's sigma), RoPE over time only;
//! 4. RMS norm, AdaLN, GELU (tanh) feed-forward, gated residual.
//!
//! Every attention normalizes its queries and keys over all heads at once (RMS norm with a
//! weight) and scales each head by `2 sigmoid(to_gate_logits(x))`. The output head is a layer norm
//! (no affine) with a shift / scale from the embedded timestep, then `proj_out`.
//!
//! The text connectors (`video_embeddings_connector`, `audio_embeddings_connector`: 8 plain
//! pre-norm blocks over the 1024 text rows, padding replaced by 128 learned registers) live in
//! the same file and are loaded here too ([`Connector`]).
//!
//! Weights load as stored: bf16, or ComfyUI's int8 ConvRot (`*-comfy-int8-convrot`: int8 rows
//! times a row scale, the inputs rotated by the normalized Hadamard transform in groups of 256)
//! and NVFP4 (`*-nvfp4`) files through [`crate::comfy_quant`].

use super::nn::{AdaSingle, Attn, Ff, Rope, rms_rows, table};
use crate::kernels::for_chunks;
use crate::minimax_h3::dit::DLin;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

/// The transformer's sizes (the `config.transformer` object of the file's metadata).
#[derive(Clone, Debug)]
pub struct DitConfig {
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub audio_heads: usize,
    pub audio_head_dim: usize,
    pub in_channels: usize,
    pub audio_in_channels: usize,
    pub eps: f32,
    pub theta: f64,
    pub max_pos: [f32; 3],
    pub audio_max_pos: f32,
    pub timestep_mult: f32,
    pub av_ca_mult: f32,
    pub conn_layers: usize,
    pub conn_heads: usize,
    pub conn_head_dim: usize,
    pub audio_conn_heads: usize,
    pub audio_conn_head_dim: usize,
    pub registers: usize,
    pub conn_max_pos: f32,
}

impl DitConfig {
    pub fn dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn audio_dim(&self) -> usize {
        self.audio_heads * self.audio_head_dim
    }

    /// From the `config` metadata entry of a transformer file (its `transformer` object).
    pub fn from_json(c: &serde_json::Value) -> Result<Self> {
        let t = c.get("transformer").unwrap_or(c);
        let u = |k: &str, d: usize| t.get(k).and_then(|v| v.as_u64()).map_or(d, |v| v as usize);
        let f = |k: &str, d: f64| t.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let arr = |k: &str| -> Vec<f64> {
            t.get(k)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                .unwrap_or_default()
        };
        if t.get("rope_type").and_then(|v| v.as_str()).is_some_and(|r| r != "split") {
            bail!("LTX transformer: only the split RoPE layout is supported");
        }
        if t.get("caption_proj_before_connector").and_then(|v| v.as_bool()) == Some(false) {
            bail!("LTX transformer: caption projections inside the transformer (LTX-2 19B) are not supported");
        }
        let mp = arr("positional_embedding_max_pos");
        let amp = arr("audio_positional_embedding_max_pos");
        let cmp = arr("connector_positional_embedding_max_pos");
        Ok(Self {
            layers: u("num_layers", 48),
            heads: u("num_attention_heads", 32),
            head_dim: u("attention_head_dim", 128),
            audio_heads: u("audio_num_attention_heads", 32),
            audio_head_dim: u("audio_attention_head_dim", 64),
            in_channels: u("in_channels", 128),
            audio_in_channels: u("audio_in_channels", 128),
            eps: f("norm_eps", 1e-6) as f32,
            theta: f("positional_embedding_theta", 10000.0),
            max_pos: if mp.len() == 3 {
                [mp[0] as f32, mp[1] as f32, mp[2] as f32]
            } else {
                [20.0, 2048.0, 2048.0]
            },
            audio_max_pos: amp.first().copied().unwrap_or(20.0) as f32,
            timestep_mult: f("timestep_scale_multiplier", 1000.0) as f32,
            av_ca_mult: f("av_ca_timestep_scale_multiplier", 1.0) as f32,
            conn_layers: u("connector_num_layers", 2),
            conn_heads: u("connector_num_attention_heads", 30),
            conn_head_dim: u("connector_attention_head_dim", 128),
            audio_conn_heads: u("audio_connector_num_attention_heads", u("connector_num_attention_heads", 30)),
            audio_conn_head_dim: u("audio_connector_attention_head_dim", u("connector_attention_head_dim", 128)),
            registers: u("connector_num_learnable_registers", 128),
            conn_max_pos: cmp.first().copied().unwrap_or(1.0) as f32,
        })
    }

    pub fn from_files(files: &ModelFiles) -> Result<Self> {
        let s = files.metadata("config").context("LTX transformer: no `config` metadata")?;
        Self::from_json(&serde_json::from_str(s).context("LTX transformer: `config` metadata is not JSON")?)
    }
}

/// One stream's half of a block.
struct Stream<'a> {
    attn1: Attn<'a>,
    attn2: Attn<'a>,
    ff: Ff<'a>,
    /// `[9, dim]`: shift, scale, gate of the self-attention and of the feed-forward, then the
    /// shift, scale, gate of the text cross-attention's queries.
    sst: Vec<f32>,
    /// `[2, dim]`: shift, scale of the text rows.
    prompt_sst: Vec<f32>,
    /// `[5, dim]`: scale, shift (audio-to-video), scale, shift (video-to-audio), gate.
    av_sst: Vec<f32>,
}

struct Block<'a> {
    v: Stream<'a>,
    a: Stream<'a>,
    /// Video queries over audio rows.
    a2v: Attn<'a>,
    /// Audio queries over video rows.
    v2a: Attn<'a>,
}

/// A text connector (`Embeddings1DConnector`).
pub struct Connector<'a> {
    blocks: Vec<(Attn<'a>, Ff<'a>)>,
    registers: Vec<f32>,
    dim: usize,
    max_pos: f32,
    theta: f64,
}

impl<'a> Connector<'a> {
    #[allow(clippy::too_many_arguments)]
    fn load(f: &'a ModelFiles, p: &str, layers: usize, heads: usize, hd: usize, n_reg: usize, max_pos: f32, theta: f64) -> Result<Self> {
        let dim = heads * hd;
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let bp = format!("{p}.transformer_1d_blocks.{i}");
            blocks.push((
                Attn::load(f, &format!("{bp}.attn1"), dim, dim, heads, hd, 1e-6)?,
                Ff::load(f, &format!("{bp}.ff"), dim, 4 * dim)?,
            ));
        }
        Ok(Self {
            blocks,
            registers: table(f, &format!("{p}.learnable_registers"), n_reg, dim)?,
            dim,
            max_pos,
            theta,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The context rows of `feats` (`[n, dim]`, the valid tokens in order) padded to `total`
    /// rows (a multiple of the register count) with the learned registers: the connector's
    /// blocks, then an RMS norm.
    pub fn forward(&self, feats: &[f32], total: usize) -> Vec<f32> {
        let d = self.dim;
        let n = feats.len() / d;
        let nr = self.registers.len() / d;
        let mut x = vec![0f32; total * d];
        x[..n * d].copy_from_slice(feats);
        for r in n..total {
            let s = (r % nr) * d;
            x[r * d..(r + 1) * d].copy_from_slice(&self.registers[s..s + d]);
        }
        let frac: Vec<f32> = (0..total).map(|i| i as f32 / self.max_pos).collect();
        let pe = Rope::new(&frac, 1, d, self.theta);
        for (attn, ff) in &self.blocks {
            let h = rms_rows(&x, d, 1e-6);
            let a = attn.forward(&h, total, None, Some(&pe), None);
            add(&mut x, &a);
            let h = rms_rows(&x, d, 1e-6);
            let y = ff.forward(&h, total);
            add(&mut x, &y);
        }
        rms_rows(&x, d, 1e-6)
    }
}

fn add(x: &mut [f32], y: &[f32]) {
    let n = 1 << 14;
    for_chunks(x, n, x.len() > n, |i, r| r.iter_mut().zip(&y[i * n..]).for_each(|(a, &b)| *a += b));
}

/// One stream's input to [`Dit::forward`].
pub struct StreamIn<'a> {
    /// Patchified latent rows (`[n, in_channels]`).
    pub x: &'a [f32],
    /// Each row's timestep (its sigma times its denoising mask).
    pub t: &'a [f32],
    /// The stream's sigma (prompt AdaLN, the other stream's cross-attention gate).
    pub sigma: f32,
    /// Each row's position (video: time in seconds, y and x in pixels; audio: time in seconds),
    /// the middle of its patch.
    pub pos: &'a [f32],
    /// The text context (`[n_ctx, dim]`, after the connector).
    pub ctx: &'a [f32],
    /// Rows that get the keyframe embedding (the first latent frame of the video).
    pub keyframe_rows: usize,
    /// More rows that get it: generated keyframe slots (`VideoGeneratedKeyframeSlots`).
    pub slot_rows: std::ops::Range<usize>,
}

/// Blocks a perturbation applies to (`Perturbation.blocks`: none means all of them).
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Blocks {
    #[default]
    None,
    All,
    List(Vec<usize>),
}

impl Blocks {
    pub fn has(&self, i: usize) -> bool {
        match self {
            Self::None => false,
            Self::All => true,
            Self::List(l) => l.contains(&i),
        }
    }

    pub fn is_none(&self) -> bool {
        match self {
            Self::None => true,
            Self::All => false,
            Self::List(l) => l.is_empty(),
        }
    }
}

/// The attention perturbations of a guidance pass (`PerturbationConfig`): the self-attention of
/// each stream skipped in some blocks (STG: the value projection passed straight through), the
/// audio-video cross-attention skipped in every block (the isolated-modality pass).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Perturb {
    pub video_self: Blocks,
    pub audio_self: Blocks,
    pub cross: bool,
}

impl Perturb {
    pub fn is_none(&self) -> bool {
        self.video_self.is_none() && self.audio_self.is_none() && !self.cross
    }
}

/// A stream's modulation for one forward: the distinct timesteps of its rows and their tables.
struct Mods {
    /// Each row's index into the tables.
    idx: Vec<usize>,
    /// Per distinct timestep: `[9 * dim]` block modulation (before the block's own table).
    block: Vec<Vec<f32>>,
    /// Per distinct timestep: `[dim]` embedded timestep (output head).
    emb: Vec<Vec<f32>>,
    /// Per distinct timestep: `[4 * dim]` AV cross scale / shift.
    av: Vec<Vec<f32>>,
    /// `[dim]`: the AV cross gate (from the other stream's sigma).
    gate: Vec<f32>,
    /// `[2 * dim]`: the prompt AdaLN of the stream's sigma.
    prompt: Vec<f32>,
}

pub struct Dit<'a> {
    pub cfg: DitConfig,
    patchify: DLin<'a>,
    a_patchify: DLin<'a>,
    adaln: AdaSingle<'a>,
    a_adaln: AdaSingle<'a>,
    prompt_adaln: AdaSingle<'a>,
    a_prompt_adaln: AdaSingle<'a>,
    av_v_ss: AdaSingle<'a>,
    av_a_ss: AdaSingle<'a>,
    av_a2v_gate: AdaSingle<'a>,
    av_v2a_gate: AdaSingle<'a>,
    sst_out: Vec<f32>,
    a_sst_out: Vec<f32>,
    proj_out: DLin<'a>,
    a_proj_out: DLin<'a>,
    keyframe: Option<Vec<f32>>,
    blocks: Vec<Block<'a>>,
    pub video_connector: Connector<'a>,
    pub audio_connector: Connector<'a>,
    /// Run the blocks on the accelerator's whole-pass session when there is one (else, or when
    /// it declines, on the CPU).
    pub device: bool,
    /// On the device, 128-wide heads' attention with float32 operands (else half-precision).
    pub exact_attention: bool,
    /// The switch of the LoRA that changes between stages (the distilled LoRA of the two-stage
    /// dev pipeline): the device twins are kept for each of its states.
    pub lora_switch: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    twins: [std::sync::OnceLock<super::device::Twins<'a>>; 2],
    session: std::sync::Mutex<Option<Box<dyn crate::gpu::dit::LtxSession>>>,
}

/// The tensor-name prefix of a transformer file (ComfyUI's `model.diffusion_model.`, or none).
pub fn prefix(files: &ModelFiles) -> &'static str {
    if files.has("model.diffusion_model.patchify_proj.weight") {
        "model.diffusion_model."
    } else {
        ""
    }
}

impl<'a> Dit<'a> {
    pub fn load(files: &'a ModelFiles) -> Result<Self> {
        let cfg = DitConfig::from_files(files)?;
        Self::load_with(cfg, files)
    }

    pub fn load_with(cfg: DitConfig, f: &'a ModelFiles) -> Result<Self> {
        let p = prefix(f);
        let (d, ad) = (cfg.dim(), cfg.audio_dim());
        let (h, hd, ah, ahd) = (cfg.heads, cfg.head_dim, cfg.audio_heads, cfg.audio_head_dim);
        let eps = cfg.eps;
        let n = |s: &str| format!("{p}{s}");
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let b = n(&format!("transformer_blocks.{i}"));
            let stream = |pre: &str, dim: usize, ctx: usize, heads: usize, hd: usize, sfx: &str| -> Result<Stream<'a>> {
                let sst_name = if pre.is_empty() {
                    "scale_shift_table"
                } else {
                    "audio_scale_shift_table"
                };
                let prompt_name = if pre.is_empty() {
                    "prompt_scale_shift_table"
                } else {
                    "audio_prompt_scale_shift_table"
                };
                Ok(Stream {
                    attn1: Attn::load(f, &format!("{b}.{pre}attn1"), dim, dim, heads, hd, eps)?,
                    attn2: Attn::load(f, &format!("{b}.{pre}attn2"), dim, ctx, heads, hd, eps)?,
                    ff: Ff::load(f, &format!("{b}.{pre}ff"), dim, 4 * dim)?,
                    sst: table(f, &format!("{b}.{sst_name}"), 9, dim)?,
                    prompt_sst: table(f, &format!("{b}.{prompt_name}"), 2, dim)?,
                    av_sst: table(f, &format!("{b}.scale_shift_table_a2v_ca_{sfx}"), 5, dim)?,
                })
            };
            blocks.push(Block {
                v: stream("", d, d, h, hd, "video")?,
                a: stream("audio_", ad, ad, ah, ahd, "audio")?,
                a2v: Attn::load(f, &format!("{b}.audio_to_video_attn"), d, ad, ah, ahd, eps)?,
                v2a: Attn::load(f, &format!("{b}.video_to_audio_attn"), ad, d, ah, ahd, eps)?,
            });
        }
        let keyframe = if f.has(&n("keyframes_abs_pos_embedding")) {
            Some(table(f, &n("keyframes_abs_pos_embedding"), 1, d)?)
        } else {
            None
        };
        Ok(Self {
            patchify: DLin::load(f, &n("patchify_proj"), d, cfg.in_channels, true)?,
            a_patchify: DLin::load(f, &n("audio_patchify_proj"), ad, cfg.audio_in_channels, true)?,
            adaln: AdaSingle::load(f, &n("adaln_single"), d, 9)?,
            a_adaln: AdaSingle::load(f, &n("audio_adaln_single"), ad, 9)?,
            prompt_adaln: AdaSingle::load(f, &n("prompt_adaln_single"), d, 2)?,
            a_prompt_adaln: AdaSingle::load(f, &n("audio_prompt_adaln_single"), ad, 2)?,
            av_v_ss: AdaSingle::load(f, &n("av_ca_video_scale_shift_adaln_single"), d, 4)?,
            av_a_ss: AdaSingle::load(f, &n("av_ca_audio_scale_shift_adaln_single"), ad, 4)?,
            av_a2v_gate: AdaSingle::load(f, &n("av_ca_a2v_gate_adaln_single"), d, 1)?,
            av_v2a_gate: AdaSingle::load(f, &n("av_ca_v2a_gate_adaln_single"), ad, 1)?,
            sst_out: table(f, &n("scale_shift_table"), 2, d)?,
            a_sst_out: table(f, &n("audio_scale_shift_table"), 2, ad)?,
            proj_out: DLin::load(f, &n("proj_out"), cfg.in_channels, d, true)?,
            a_proj_out: DLin::load(f, &n("audio_proj_out"), cfg.audio_in_channels, ad, true)?,
            keyframe,
            blocks,
            video_connector: Connector::load(
                f,
                &n("video_embeddings_connector"),
                cfg.conn_layers,
                cfg.conn_heads,
                cfg.conn_head_dim,
                cfg.registers,
                cfg.conn_max_pos,
                cfg.theta,
            )?,
            audio_connector: Connector::load(
                f,
                &n("audio_embeddings_connector"),
                cfg.conn_layers,
                cfg.audio_conn_heads,
                cfg.audio_conn_head_dim,
                cfg.registers,
                cfg.conn_max_pos,
                cfg.theta,
            )?,
            cfg,
            device: true,
            exact_attention: false,
            lora_switch: None,
            twins: [std::sync::OnceLock::new(), std::sync::OnceLock::new()],
            session: std::sync::Mutex::new(None),
        })
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Block `i`'s parts: (video self, text, ff), (audio self, text, ff), audio-to-video,
    /// video-to-audio.
    #[allow(clippy::type_complexity)]
    pub(crate) fn block_parts(
        &self,
        i: usize,
    ) -> (
        (&Attn<'a>, &Attn<'a>, &Ff<'a>),
        (&Attn<'a>, &Attn<'a>, &Ff<'a>),
        &Attn<'a>,
        &Attn<'a>,
    ) {
        let b = &self.blocks[i];
        ((&b.v.attn1, &b.v.attn2, &b.v.ff), (&b.a.attn1, &b.a.attn2, &b.a.ff), &b.a2v, &b.v2a)
    }

    /// Every linear layer of the blocks, with whether it is an attention gate.
    pub(crate) fn for_each_block_layer(&self, f: &mut dyn FnMut(&DLin<'a>, bool)) {
        for b in &self.blocks {
            for at in [&b.v.attn1, &b.v.attn2, &b.a.attn1, &b.a.attn2, &b.a2v, &b.v2a] {
                for l in [&at.q, &at.k, &at.v, &at.o] {
                    f(l, false);
                }
                if let Some(g) = &at.gate {
                    f(g, true);
                }
            }
            for ff in [&b.v.ff, &b.a.ff] {
                f(&ff.up, false);
                f(&ff.down, false);
            }
        }
    }

    /// Whether the transformer has the learned keyframe embedding (generated keyframe slots).
    pub fn has_keyframe_embedding(&self) -> bool {
        self.keyframe.is_some()
    }

    /// Drops the device session's buffers (made again on the next device pass).
    pub fn release_device(&self) {
        *self.session.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// The session's per-forward tables of one stream.
    fn tables(&self, m: &Mods, video: bool) -> super::device::Tables {
        let dim = if video { self.cfg.dim() } else { self.cfg.audio_dim() };
        let nc = m.block.len();
        let nb = self.blocks.len();
        let mut mods = Vec::with_capacity(nb * nc * 9 * dim);
        let mut av = Vec::with_capacity(nb * nc * 4 * dim);
        let mut av_gate = Vec::with_capacity(nb * dim);
        let mut prompt = Vec::with_capacity(nb * 2 * dim);
        for b in &self.blocks {
            let s = if video { &b.v } else { &b.a };
            for k in 0..nc {
                mods.extend(s.sst.iter().zip(&m.block[k]).map(|(a, b)| a + b));
                av.extend(s.av_sst[..4 * dim].iter().zip(&m.av[k]).map(|(a, b)| a + b));
            }
            av_gate.extend(s.av_sst[4 * dim..].iter().zip(&m.gate).map(|(a, b)| a + b));
            prompt.extend(s.prompt_sst.iter().zip(&m.prompt).map(|(a, b)| a + b));
        }
        super::device::Tables {
            cls: m.idx.iter().map(|&i| i as u32).collect(),
            classes: nc,
            mods,
            av,
            av_gate,
            prompt,
        }
    }

    /// The blocks on the device's session: `vx` / `ax` replaced by the rows after the last
    /// block; false when there is no session or it declined.
    #[allow(clippy::too_many_arguments)]
    fn blocks_device(
        &self,
        vx: &mut Vec<f32>,
        ax: &mut Vec<f32>,
        v: (&StreamIn, &Mods, &Rope, &Rope),
        a: (&StreamIn, &Mods, &Rope, &Rope),
        ptb: &Perturb,
    ) -> bool {
        if !self.device || std::env::var("OXINFER_LTX_GPU").is_ok_and(|v| v == "0") {
            return false;
        }
        let Some(be) = crate::gpu::active() else { return false };
        let on = self
            .lora_switch
            .as_ref()
            .is_none_or(|s| s.load(std::sync::atomic::Ordering::Relaxed));
        let twins = self.twins[on as usize].get_or_init(|| super::device::Twins::build(self));
        let Some(model) = twins.model(self, !self.exact_attention) else {
            return false;
        };
        let (d, ad) = (self.cfg.dim(), self.cfg.audio_dim());
        let (vt, at) = (self.tables(v.1, true), self.tables(a.1, false));
        let vin = vt.input(vx, vx.len() / d, v.0.ctx, v.0.ctx.len() / d, v.2, v.3);
        let ain = at.input(ax, ax.len() / ad, a.0.ctx, a.0.ctx.len() / ad, a.2, a.3);
        let mut sess = self.session.lock().unwrap_or_else(|e| e.into_inner());
        if sess.is_none() {
            *sess = be.ltx_session();
        }
        let Some(s) = sess.as_mut() else { return false };
        let (mut vo, mut ao) = (Vec::new(), Vec::new());
        let nb = self.blocks.len();
        let p = crate::gpu::dit::LtxPerturb {
            video_self: (0..nb).map(|i| ptb.video_self.has(i)).collect(),
            audio_self: (0..nb).map(|i| ptb.audio_self.has(i)).collect(),
            cross: ptb.cross,
        };
        if !s.forward(&model, &vin, &ain, &p, &mut vo, &mut ao) {
            return false;
        }
        *vx = vo;
        *ax = ao;
        true
    }

    /// Keeps only the first `n` blocks (tests and partial comparisons).
    pub fn truncate_blocks(&mut self, n: usize) {
        self.blocks.truncate(n);
    }

    fn mods(&self, s: &StreamIn, other_sigma: f32, video: bool) -> Mods {
        let m = self.cfg.timestep_mult;
        let (ada, av_ss, gate, prompt) = if video {
            (&self.adaln, &self.av_v_ss, &self.av_a2v_gate, &self.prompt_adaln)
        } else {
            (&self.a_adaln, &self.av_a_ss, &self.av_v2a_gate, &self.a_prompt_adaln)
        };
        let mut uniq: Vec<f32> = Vec::new();
        let idx: Vec<usize> =
            s.t.iter()
                .map(|&t| match uniq.iter().position(|&u| u.to_bits() == t.to_bits()) {
                    Some(i) => i,
                    None => {
                        uniq.push(t);
                        uniq.len() - 1
                    }
                })
                .collect();
        let mut block = Vec::new();
        let mut emb = Vec::new();
        let mut av = Vec::new();
        for &t in &uniq {
            let (b, e) = ada.forward(t * m);
            block.push(b);
            emb.push(e);
            av.push(av_ss.forward(t * m).0);
        }
        let av_factor = self.cfg.av_ca_mult / m;
        Mods {
            idx,
            block,
            emb,
            av,
            gate: gate.forward(other_sigma * m * av_factor).0,
            prompt: prompt.forward(s.sigma * m).0,
        }
    }

    /// The velocities of both streams (`[n_video, in_channels]`, `[n_audio, audio_in_channels]`).
    pub fn forward(&self, v: &StreamIn, a: &StreamIn) -> (Vec<f32>, Vec<f32>) {
        self.forward_with(v, a, &Perturb::default())
    }

    /// [`Self::forward`] with attention perturbations (a guidance pass).
    pub fn forward_with(&self, v: &StreamIn, a: &StreamIn, ptb: &Perturb) -> (Vec<f32>, Vec<f32>) {
        let cfg = &self.cfg;
        let (d, ad) = (cfg.dim(), cfg.audio_dim());
        let nv = v.x.len() / cfg.in_channels;
        let na = a.x.len() / cfg.audio_in_channels;
        let (n_ctx, na_ctx) = (v.ctx.len() / d, a.ctx.len() / ad);
        let mv = self.mods(v, a.sigma, true);
        let ma = self.mods(a, v.sigma, false);

        // Rotary tables: the streams' own positions, and time alone for the AV cross-attention.
        let vfrac: Vec<f32> = v
            .pos
            .chunks(3)
            .flat_map(|p| [p[0] / cfg.max_pos[0], p[1] / cfg.max_pos[1], p[2] / cfg.max_pos[2]])
            .collect();
        let vpe = Rope::new(&vfrac, 3, d, cfg.theta);
        let afrac: Vec<f32> = a.pos.iter().map(|&t| t / cfg.audio_max_pos).collect();
        let ape = Rope::new(&afrac, 1, ad, cfg.theta);
        let cross_max = cfg.max_pos[0].max(cfg.audio_max_pos);
        let vtime: Vec<f32> = v.pos.chunks(3).map(|p| p[0] / cross_max).collect();
        let vcpe = Rope::new(&vtime, 1, ad, cfg.theta);
        let atime: Vec<f32> = a.pos.iter().map(|&t| t / cross_max).collect();
        let acpe = Rope::new(&atime, 1, ad, cfg.theta);

        let mut vx = self.patchify.forward(v.x, nv);
        if let Some(k) = &self.keyframe {
            for r in (0..v.keyframe_rows.min(nv)).chain(v.slot_rows.start.min(nv)..v.slot_rows.end.min(nv)) {
                vx[r * d..(r + 1) * d].iter_mut().zip(k).for_each(|(x, &e)| *x += e);
            }
        }
        let mut ax = self.a_patchify.forward(a.x, na);
        if nv == 0 || na == 0 || !self.blocks_device(&mut vx, &mut ax, (v, &mv, &vpe, &vcpe), (a, &ma, &ape, &acpe), ptb) {
            for (i, b) in self.blocks.iter().enumerate() {
                let skip = (ptb.video_self.has(i), ptb.audio_self.has(i), ptb.cross);
                self.block(
                    b,
                    &mut vx,
                    &mut ax,
                    (v, &mv, &vpe, &vcpe, n_ctx),
                    (a, &ma, &ape, &acpe, na_ctx),
                    skip,
                );
            }
        }
        let vo = self.head(&vx, nv, d, &mv, &self.sst_out, &self.proj_out);
        let ao = self.head(&ax, na, ad, &ma, &self.a_sst_out, &self.a_proj_out);
        (vo, ao)
    }

    #[allow(clippy::type_complexity)]
    fn block(
        &self,
        b: &Block,
        vx: &mut [f32],
        ax: &mut [f32],
        (v, mv, vpe, vcpe, n_ctx): (&StreamIn, &Mods, &Rope, &Rope, usize),
        (a, ma, ape, acpe, na_ctx): (&StreamIn, &Mods, &Rope, &Rope, usize),
        (skip_v, skip_a, skip_cross): (bool, bool, bool),
    ) {
        let eps = self.cfg.eps;
        let (d, ad) = (self.cfg.dim(), self.cfg.audio_dim());
        let (nv, na) = (vx.len() / d, ax.len() / ad);
        self_and_text(&b.v, vx, nv, d, mv, (vpe, skip_v), v.ctx, n_ctx, eps);
        self_and_text(&b.a, ax, na, ad, ma, (ape, skip_a), a.ctx, na_ctx, eps);
        // Audio-video cross-attention over the streams as they are now.
        if nv > 0 && na > 0 && !skip_cross {
            let vn = rms_rows(vx, d, eps);
            let an = rms_rows(ax, ad, eps);
            // Rows of the AV tables: [0] scale, [1] shift (a2v), [2] scale, [3] shift (v2a), [4] gate.
            let a2v_q = modulated(&vn, d, &b.v.av_sst, &mv.av, &mv.idx, 0, 1);
            let a2v_kv = modulated(&an, ad, &b.a.av_sst, &ma.av, &ma.idx, 0, 1);
            let v2a_q = modulated(&an, ad, &b.a.av_sst, &ma.av, &ma.idx, 2, 3);
            let v2a_kv = modulated(&vn, d, &b.v.av_sst, &mv.av, &mv.idx, 2, 3);
            let y = b.a2v.forward(&a2v_q, nv, Some((&a2v_kv, na)), Some(vcpe), Some(acpe));
            let gv: Vec<f32> = (0..d).map(|c| b.v.av_sst[4 * d + c] + mv.gate[c]).collect();
            gated_add(vx, &y, d, |_| &gv);
            let y = b.v2a.forward(&v2a_q, na, Some((&v2a_kv, nv)), Some(acpe), Some(vcpe));
            let ga: Vec<f32> = (0..ad).map(|c| b.a.av_sst[4 * ad + c] + ma.gate[c]).collect();
            gated_add(ax, &y, ad, |_| &ga);
        }
        feed_forward(&b.v, vx, nv, d, mv, eps);
        feed_forward(&b.a, ax, na, ad, ma, eps);
    }

    fn head(&self, x: &[f32], n: usize, dim: usize, m: &Mods, sst: &[f32], proj: &DLin) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }
        let eps = self.cfg.eps;
        let mut h = x.to_vec();
        for_chunks(&mut h, dim, n > 8, |r, row| {
            let mean = row.iter().sum::<f32>() / dim as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let s = 1.0 / (var + eps).sqrt();
            let e = &m.emb[m.idx[r]];
            for (c, v) in row.iter_mut().enumerate() {
                let shift = sst[c] + e[c];
                let scale = sst[dim + c] + e[c];
                *v = (*v - mean) * s * (1.0 + scale) + shift;
            }
        });
        proj.forward(&h, n)
    }
}

/// Rows `xn` modulated `x (1 + scale) + shift` with `scale = sst[si] + m[k][si]`,
/// `shift = sst[hi] + m[k][hi]` (`sst` and each `m[k]` rows of `dim`).
#[allow(clippy::too_many_arguments)]
fn modulated(xn: &[f32], dim: usize, sst: &[f32], m: &[Vec<f32>], idx: &[usize], si: usize, hi: usize) -> Vec<f32> {
    let mut y = xn.to_vec();
    let n = y.len() / dim;
    for_chunks(&mut y, dim, n > 8, |r, row| {
        let mk = &m[idx[r]];
        for (c, v) in row.iter_mut().enumerate() {
            let scale = sst[si * dim + c] + mk[si * dim + c];
            let shift = sst[hi * dim + c] + mk[hi * dim + c];
            *v = *v * (1.0 + scale) + shift;
        }
    });
    y
}

/// `x += y * gate(row)`.
fn gated_add<'g>(x: &mut [f32], y: &[f32], dim: usize, gate: impl Fn(usize) -> &'g [f32] + Sync) {
    let n = x.len() / dim;
    for_chunks(x, dim, n > 8, |r, row| {
        let g = gate(r);
        let yr = &y[r * dim..(r + 1) * dim];
        for ((v, &yv), &gv) in row.iter_mut().zip(yr).zip(g) {
            *v += yv * gv;
        }
    });
}

/// Per distinct timestep, rows `js` of the block table plus the modulation.
fn ada_sets(s: &Stream, m: &Mods, js: &[usize], dim: usize) -> Vec<Vec<Vec<f32>>> {
    m.block
        .iter()
        .map(|mk| {
            js.iter()
                .map(|&j| (0..dim).map(|c| s.sst[j * dim + c] + mk[j * dim + c]).collect())
                .collect()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn self_and_text(
    s: &Stream,
    x: &mut [f32],
    n: usize,
    dim: usize,
    m: &Mods,
    (pe, skip): (&Rope, bool),
    ctx: &[f32],
    n_ctx: usize,
    eps: f32,
) {
    if n == 0 {
        return;
    }
    // Self-attention: rows 0 (shift), 1 (scale), 2 (gate).
    let sets = ada_sets(s, m, &[0, 1, 2, 6, 7, 8], dim);
    let mut h = rms_rows(x, dim, eps);
    for_chunks(&mut h, dim, n > 8, |r, row| {
        let st = &sets[m.idx[r]];
        for (c, v) in row.iter_mut().enumerate() {
            *v = *v * (1.0 + st[1][c]) + st[0][c];
        }
    });
    let y = if skip {
        s.attn1.passthrough(&h, n)
    } else {
        s.attn1.forward(&h, n, None, Some(pe), None)
    };
    gated_add(x, &y, dim, |r| &sets[m.idx[r]][2]);
    // Text cross-attention: rows 6 (shift), 7 (scale), 8 (gate); the context's own shift / scale.
    let mut h = rms_rows(x, dim, eps);
    for_chunks(&mut h, dim, n > 8, |r, row| {
        let st = &sets[m.idx[r]];
        for (c, v) in row.iter_mut().enumerate() {
            *v = *v * (1.0 + st[4][c]) + st[3][c];
        }
    });
    let mut c = ctx.to_vec();
    let (shift, scale): (Vec<f32>, Vec<f32>) = (
        (0..dim).map(|i| s.prompt_sst[i] + m.prompt[i]).collect(),
        (0..dim).map(|i| s.prompt_sst[dim + i] + m.prompt[dim + i]).collect(),
    );
    for_chunks(&mut c, dim, n_ctx > 8, |_, row| {
        for (i, v) in row.iter_mut().enumerate() {
            *v = *v * (1.0 + scale[i]) + shift[i];
        }
    });
    let y = s.attn2.forward(&h, n, Some((&c, n_ctx)), None, None);
    gated_add(x, &y, dim, |r| &sets[m.idx[r]][5]);
}

fn feed_forward(s: &Stream, x: &mut [f32], n: usize, dim: usize, m: &Mods, eps: f32) {
    if n == 0 {
        return;
    }
    // Rows 3 (shift), 4 (scale), 5 (gate).
    let sets = ada_sets(s, m, &[3, 4, 5], dim);
    let mut h = rms_rows(x, dim, eps);
    for_chunks(&mut h, dim, n > 8, |r, row| {
        let st = &sets[m.idx[r]];
        for (c, v) in row.iter_mut().enumerate() {
            *v = *v * (1.0 + st[1][c]) + st[0][c];
        }
    });
    let y = s.ff.forward(&h, n);
    gated_add(x, &y, dim, |r| &sets[m.idx[r]][2]);
}

impl<'a> Attn<'a> {
    fn visit(&mut self, p: &str, f: &mut dyn FnMut(&str, &mut DLin<'a>)) {
        f(&format!("{p}.to_q"), &mut self.q);
        f(&format!("{p}.to_k"), &mut self.k);
        f(&format!("{p}.to_v"), &mut self.v);
        f(&format!("{p}.to_out.0"), &mut self.o);
        if let Some(g) = &mut self.gate {
            f(&format!("{p}.to_gate_logits"), g);
        }
    }
}

impl<'a> Ff<'a> {
    fn visit(&mut self, p: &str, f: &mut dyn FnMut(&str, &mut DLin<'a>)) {
        f(&format!("{p}.net.0.proj"), &mut self.up);
        f(&format!("{p}.net.2"), &mut self.down);
    }
}

impl<'a> AdaSingle<'a> {
    fn visit(&mut self, p: &str, f: &mut dyn FnMut(&str, &mut DLin<'a>)) {
        f(&format!("{p}.emb.timestep_embedder.linear_1"), &mut self.l1);
        f(&format!("{p}.emb.timestep_embedder.linear_2"), &mut self.l2);
        f(&format!("{p}.linear"), &mut self.lin);
    }
}

impl<'a> Dit<'a> {
    /// Every linear layer with its module name (no `model.diffusion_model.` prefix).
    pub(crate) fn visit_layers(&mut self, f: &mut dyn FnMut(&str, &mut DLin<'a>)) {
        f("patchify_proj", &mut self.patchify);
        f("audio_patchify_proj", &mut self.a_patchify);
        f("proj_out", &mut self.proj_out);
        f("audio_proj_out", &mut self.a_proj_out);
        self.adaln.visit("adaln_single", f);
        self.a_adaln.visit("audio_adaln_single", f);
        self.prompt_adaln.visit("prompt_adaln_single", f);
        self.a_prompt_adaln.visit("audio_prompt_adaln_single", f);
        self.av_v_ss.visit("av_ca_video_scale_shift_adaln_single", f);
        self.av_a_ss.visit("av_ca_audio_scale_shift_adaln_single", f);
        self.av_a2v_gate.visit("av_ca_a2v_gate_adaln_single", f);
        self.av_v2a_gate.visit("av_ca_v2a_gate_adaln_single", f);
        for (i, b) in self.blocks.iter_mut().enumerate() {
            let p = format!("transformer_blocks.{i}");
            b.v.attn1.visit(&format!("{p}.attn1"), f);
            b.v.attn2.visit(&format!("{p}.attn2"), f);
            b.v.ff.visit(&format!("{p}.ff"), f);
            b.a.attn1.visit(&format!("{p}.audio_attn1"), f);
            b.a.attn2.visit(&format!("{p}.audio_attn2"), f);
            b.a.ff.visit(&format!("{p}.audio_ff"), f);
            b.a2v.visit(&format!("{p}.audio_to_video_attn"), f);
            b.v2a.visit(&format!("{p}.video_to_audio_attn"), f);
        }
        for (name, c) in [
            ("video_embeddings_connector", &mut self.video_connector),
            ("audio_embeddings_connector", &mut self.audio_connector),
        ] {
            for (i, (a, ff)) in c.blocks.iter_mut().enumerate() {
                a.visit(&format!("{name}.transformer_1d_blocks.{i}.attn1"), f);
                ff.visit(&format!("{name}.transformer_1d_blocks.{i}.ff"), f);
            }
        }
    }
}
