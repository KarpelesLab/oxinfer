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
