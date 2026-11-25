//! Whole-pass transformer forwards of single-stream diffusion transformers on an accelerator:
//! the model describes its weights once per call ([`Model`]), the backend runs every block of
//! every sequence in a few submissions with the activations kept on the device, and only the
//! final velocities come back ([`Session::forward`]).
//!
//! The layout is MOSS-Image's (`crate::moss_image`): image tokens then text tokens per
//! sequence, one shared modulation, RMS norms with weights, per-head q / k RMS norms, rotary
//! embeddings in the rotate-half layout, joint attention without a mask, a SiTU feed-forward
//! over fused value / gate rows and gated residuals. Several sequences (the guidance branches of
//! one step: same latents, same timestep, different prompts) run in one batch: every matmul
//! covers all of them, attention runs per sequence.

use super::{DeviceWeight, WeightRef};

/// A linear layer as the device sees it.
pub struct Lin<'a> {
    pub w: WeightRef<'a>,
    pub dev: &'a DeviceWeight,
    pub n_out: usize,
    pub n_in: usize,
    pub bias: Option<&'a [f32]>,
    /// A low-rank side path `up (down x)` added to the output (a LoRA), each a layer of its own.
    pub side: Option<(Box<Lin<'a>>, Box<Lin<'a>>)>,
    /// The weight is stored rotated (ComfyUI's ConvRot): its input goes through the normalized
    /// regular Hadamard transform over consecutive groups of this many values first.
    pub rot: Option<usize>,
}

/// One block's weights.
pub struct Block<'a> {
    pub norm1: &'a [f32],
    pub norm2: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub q: Lin<'a>,
    pub k: Lin<'a>,
    pub v: Lin<'a>,
    pub o: Lin<'a>,
    /// `[2 ffn, dim]`: value rows then gate rows.
    pub up_gate: Lin<'a>,
    pub down: Lin<'a>,
}

pub struct Model<'a> {
    pub dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub latent: usize,
    pub eps: f32,
    pub video: Lin<'a>,
    pub blocks: Vec<Block<'a>>,
    pub final_norm: &'a [f32],
    pub final_linear: Lin<'a>,
    /// Attention with half-precision operands (else float32 ones).
    pub half_attention: bool,
}

/// One sequence of a batch.
pub struct Seq<'a> {
    /// The text tokens after the text embedder (`[text_len, dim]`).
    pub text: &'a [f32],
    pub text_len: usize,
    /// Rotary tables of its `image + text` tokens: per token `head_dim / 2` cosines, then as many
    /// sines (two planes of `tokens * head_dim / 2`).
    pub cos: &'a [f32],
    pub sin: &'a [f32],
    /// Blocks left out for this sequence.
    pub skip: &'a [usize],
}

/// Device state kept between calls (activation buffers).
pub trait Session: Send {
    /// The velocity of every sequence over latent tokens `x` (`[image, latent]`, shared by
    /// all sequences) into `out` (one `[image, latent]` per sequence). `modv` holds the block
    /// modulation (shift, scale, gate of the attention, then of the feed-forward; `dim` each),
    /// `head` the output head's shift and scale. False when the device declined or failed (the
    /// caller runs the CPU path).
    #[allow(clippy::too_many_arguments)]
    fn forward(&mut self, m: &Model, x: &[f32], image: usize, seqs: &[Seq], modv: &[f32], head: &[f32], out: &mut [Vec<f32>]) -> bool;
}

/// A FLUX.1 double-stream block (`crate::flux::transformer::DoubleBlock`): separate image and
/// text weights, one joint attention.
pub struct FluxDouble<'a> {
    /// q, k, v of the image tokens, then of the text tokens.
    pub img_qkv: [Lin<'a>; 3],
    pub txt_qkv: [Lin<'a>; 3],
    /// QK RMS norm weights: image q, image k, text q, text k.
    pub norms: [&'a [f32]; 4],
    pub img_out: Lin<'a>,
    pub txt_out: Lin<'a>,
    /// The GELU-tanh MLPs (up, down).
    pub img_mlp: [Lin<'a>; 2],
    pub txt_mlp: [Lin<'a>; 2],
}

/// A FLUX.1 single-stream block: q, k, v and the MLP input from one normalized input, the
/// attention output and the activated MLP projected together.
pub struct FluxSingle<'a> {
    pub qkv: [Lin<'a>; 3],
    pub mlp: Lin<'a>,
    pub norms: [&'a [f32]; 2],
    /// Reads `[attention (dim) | gelu(mlp)]`.
    pub out: Lin<'a>,
}

/// A FLUX.1 transformer (`crate::flux::transformer::FluxTransformer`) as the device runs it:
/// layer norms without weights (eps 1e-6) under the modulations, QK RMS norms (eps 1e-6),
/// interleaved rotary pairs, GELU-tanh MLPs, a bias on every linear layer. Qwen-Image's MMDiT
/// (`crate::qwen_image::transformer`) is the same with double blocks only, its text already
/// embedded (no `ctx_emb`), any text length, LoRA side paths, and with Edit-2511's
/// `zero_cond_t` a second image modulation for the reference tokens.
pub struct FluxModel<'a> {
    pub dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub x_emb: Lin<'a>,
    /// The text embedder; None: the text tokens come in at `dim` wide.
    pub ctx_emb: Option<Lin<'a>>,
    pub doubles: Vec<FluxDouble<'a>>,
    pub singles: Vec<FluxSingle<'a>>,
    pub proj_out: Lin<'a>,
    /// Attention with half-precision operands (else float32 ones).
    pub half_attention: bool,
}

/// One FLUX.1 call: the packed latent tokens, the T5 states, the rotary tables and the
/// modulations of this timestep.
pub struct FluxInput<'a> {
    /// `[n_img, x_emb.n_in]`.
    pub img: &'a [f32],
    pub n_img: usize,
    /// `[n_txt, ctx_emb.n_in]` (`[n_txt, dim]` without a text embedder). Without single blocks
    /// any count; FLUX.1's single blocks need a multiple of 128.
    pub txt: &'a [f32],
    pub n_txt: usize,
    /// Per token of `[text; image]`: `head_dim / 2` angles' cosines (pair `i` = dims `2i, 2i + 1`),
    /// then as many sines (two planes of `tokens * head_dim / 2`).
    pub cos: &'a [f32],
    pub sin: &'a [f32],
    /// Per double block `[image: shift, scale, gate, shift, scale, gate | text: the same]`
    /// (`12 dim`), then per single block `[shift, scale, gate]` (`3 dim`), then the head's
    /// `[scale, shift]`.
    pub mods: &'a [f32],
    /// Image tokens from this one on take the image modulation of `mods2` instead (`n_img`:
    /// none do).
    pub img_split: usize,
    /// Per double block the second image modulation (`6 dim`, the order of `mods`' image part);
    /// empty when `img_split == n_img`.
    pub mods2: &'a [f32],
    /// The leading image tokens the head predicts (`n_img`: all).
    pub n_out: usize,
}

/// FLUX.1 passes on the device, activations kept in device buffers between blocks.
pub trait FluxSession: Send {
    /// The prediction `[n_out, proj_out.n_out]` (bias included) into `out`; false when the
    /// device declined or failed (the caller then runs the CPU path).
    fn forward(&mut self, m: &FluxModel, inp: &FluxInput, out: &mut Vec<f32>) -> bool;
}
