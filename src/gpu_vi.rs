//! Whole stream steps of the realtime Inworld Voice Isolator on an accelerator
//! (`crate::voice_isolator`): the model describes its weights once ([`Model`], uploaded by
//! [`crate::gpu::Backend::vi_device`]), each stream opens a [`Session`] holding its lanes' states
//! (the time attention's key / value caches, the time Mambas' conv tails and scan states) on the
//! device, and a step runs the band encoders, every block of each lane asked for and the decoder
//! heads in one submission; only the spectra go in and the decoder outputs come back.
//!
//! The layouts are the CPU path's: tokens `[band][frame][width]`, caches
//! `[bands][heads][capacity][head_dim]` with the keys rotated at their slots, conv tails
//! `[bands][conv - 1][inner]`, scan states `[bands][inner][state]` (see [`LaneBlock`]).

use std::borrow::Cow;

/// A linear layer: `w` `[n_out][n_in]`, its bias.
pub struct Dense<'a> {
    pub w: Cow<'a, [f32]>,
    pub b: Option<&'a [f32]>,
    pub n_out: usize,
    pub n_in: usize,
}

/// A LayerNorm's weight and bias.
#[derive(Clone, Copy)]
pub struct Norm<'a> {
    pub w: &'a [f32],
    pub b: &'a [f32],
}

/// One Mamba layer (the conv weight `[k][inner]`, `a` = -exp(A_log) `[inner][state]`).
pub struct Mamba<'a> {
    pub in_proj: Dense<'a>,
    pub conv_w: &'a [f32],
    pub conv_b: &'a [f32],
    pub x_proj: Dense<'a>,
    /// No bias: `dt_bias` is added before the softplus.
    pub dt_proj: Dense<'a>,
    pub dt_bias: &'a [f32],
    pub a: &'a [f32],
    pub d: &'a [f32],
    pub out_proj: Dense<'a>,
}

/// `norm(output_proj(cat(fwd(x) + x, bwd(x) + x)))` (causal: no `bwd`).
pub struct BiMamba<'a> {
    pub fwd: Mamba<'a>,
    pub bwd: Option<Mamba<'a>>,
    pub output_proj: Dense<'a>,
    pub norm: Norm<'a>,
}

pub struct TimeAttn<'a> {
    pub norm1: Norm<'a>,
    pub qkv: Dense<'a>,
    pub proj: Dense<'a>,
    pub norm2: Norm<'a>,
    pub ffn1: Dense<'a>,
    pub ffn2: Dense<'a>,
    pub exclusive: bool,
}

/// The transformer across the bands (the same shapes as the time one).
pub type BandAttn<'a> = TimeAttn<'a>;

pub enum Block<'a> {
    Attn(Box<TimeAttn<'a>>, Box<BandAttn<'a>>),
    Mamba(Box<BiMamba<'a>>, Box<BiMamba<'a>>),
}

/// A band's decoder head: LayerNorm, linear + GELU, linear (`4 * bins` outputs).
pub struct Head<'a> {
    pub norm: Norm<'a>,
    pub l1: Dense<'a>,
    pub l2: Dense<'a>,
}

/// The realtime network.
pub struct Model<'a> {
    pub width: usize,
    pub heads: usize,
    /// FFT bins of a frame (`n_fft / 2 + 1`).
    pub bins: usize,
    /// Each band's bins `[start, end)`.
    pub edges: &'a [(usize, usize)],
    /// Per band: the input LayerNorm (`2 * bins` wide) and linear layer.
    pub enc_norm: Vec<Norm<'a>>,
    pub enc: Vec<Dense<'a>>,
    /// `[bands][width]`.
    pub band_emb: &'a [f32],
    /// `[choices][width]`.
    pub look_emb: Option<&'a [f32]>,
    pub blocks: Vec<Block<'a>>,
    /// The decoder's FiLM (1 + scale, shift).
    pub dec_scale: &'a [f32],
    pub dec_shift: &'a [f32],
    pub heads_dec: Vec<Head<'a>>,
    /// The time attention's trained window (`attention_window_frames`; 0 none).
    pub window: usize,
    /// The rotary inverse frequencies (`head_dim / 2`).
    pub rope_inv: &'a [f32],
}

/// A stream's settings.
#[derive(Clone, Copy, Debug)]
pub struct StreamSpec {
    /// Index of the look-ahead embedding (None: no embedding).
    pub look: Option<usize>,
    /// The attention's past in frames.
    pub cache_frames: usize,
    /// Cache slots used before compaction.
    pub capacity: usize,
    /// The most frames a step takes.
    pub max_frames: usize,
}

/// One block's state of a lane, in the CPU path's layouts.
pub enum LaneBlock {
    /// Keys (rotated at their slots) and values `[bands][heads][cap][head_dim]`, `end` slots used.
    Attn { k: Vec<f32>, v: Vec<f32>, cap: usize, end: usize },
    /// The conv tail and the scan state.
    Mamba { tail: Vec<f32>, ssm: Vec<f32> },
}

/// The uploaded weights of one checkpoint.
pub trait Device: Send + Sync {
    /// A stream's session with two lanes (fresh states), None when the device cannot run it.
    fn session(self: std::sync::Arc<Self>, spec: &StreamSpec) -> Option<Box<dyn Session>>;
}

/// A stream's states on the device.
pub trait Session: Send {
    /// Runs `nf` frames (spectra `[nf][bins][2]`): the encoder, then for each lane of `lanes`
    /// every block with that lane's state and the decoder heads, `raw[i]` getting lane
    /// `lanes[i]`'s output `[nf][bins][4]`. False when the device failed (the states are then
    /// those before the step).
    fn step(&mut self, spec: &[f32], nf: usize, lanes: &[usize], raw: &mut [Vec<f32>]) -> bool;
    /// Lane `to` starts from lane `from`: the Mamba states copied, the attention caches empty.
    fn fork(&mut self, from: usize, to: usize);
    /// Lane `lane`'s states, for the CPU path to carry on.
    fn export(&self, lane: usize) -> Vec<LaneBlock>;
}
