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
    /// The side path's gains at run time: `(start, gain)` in increasing `start`, the rank columns
    /// of `down x` from each `start` on scaled by its `gain` (at most 4; empty: as stored).
    pub side_gains: Vec<(usize, f32)>,
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
    /// Without `half_attention`: queries and keys as pairs of halves (hi + lo, 22 bits) on the
    /// half-precision matrix units where the device has that kernel, instead of float32
    /// operands (scores as exact; probabilities and values in half precision).
    pub split_attention: bool,
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

/// A T5 v1.1 encoder block (`crate::flux::t5`): RMS norms with weights, unscaled attention
/// over 64-wide heads with a relative-position bias, a gated-GELU (tanh) feed-forward; no
/// biases.
pub struct T5Block<'a> {
    pub ln1: &'a [f32],
    pub q: Lin<'a>,
    pub k: Lin<'a>,
    pub v: Lin<'a>,
    pub o: Lin<'a>,
    pub ln2: &'a [f32],
    /// wi_0 (the GELU gate), wi_1.
    pub wi: [Lin<'a>; 2],
    pub wo: Lin<'a>,
}

pub struct T5Model<'a> {
    pub dim: usize,
    pub heads: usize,
    pub d_kv: usize,
    pub eps: f32,
    pub blocks: Vec<T5Block<'a>>,
    pub final_ln: &'a [f32],
}

/// Whole T5 encoder passes on the device.
pub trait T5Session: Send {
    /// The last hidden state (final norm included) `[t, dim]` into `out` from the token
    /// embeddings `x` (`[t, dim]`); `bias` is the relative-position bias per head over
    /// `key - query` (`[heads, 2t - 1]`, offset `t - 1` for 0). False when the device declined
    /// or failed (the caller runs the CPU path).
    fn forward(&mut self, m: &T5Model, x: &[f32], t: usize, bias: &[f32], out: &mut Vec<f32>) -> bool;
}

/// A CLIP text-encoder layer (`crate::sd::clip`): pre-LN, causal self-attention over 64-wide
/// heads, an MLP with quick_gelu or GELU (erf); every linear layer with a bias.
pub struct ClipBlock<'a> {
    /// Layer norm weight and bias.
    pub ln1: [&'a [f32]; 2],
    pub q: Lin<'a>,
    pub k: Lin<'a>,
    pub v: Lin<'a>,
    pub o: Lin<'a>,
    pub ln2: [&'a [f32]; 2],
    pub fc1: Lin<'a>,
    pub fc2: Lin<'a>,
}

pub struct ClipModel<'a> {
    pub dim: usize,
    pub heads: usize,
    pub eps: f32,
    /// The MLP activation: false quick_gelu, true GELU with erf.
    pub gelu: bool,
    pub blocks: Vec<ClipBlock<'a>>,
    pub final_ln: [&'a [f32]; 2],
}

/// Whole CLIP text-encoder passes on the device.
pub trait ClipSession: Send {
    /// The final hidden state after the final layer norm (`[t, dim]`) into `out`, and with
    /// `penultimate` the hidden state before the last layer (no norm) into it, from the token +
    /// position embeddings `x` (`[t, dim]`). False when the device declined or failed.
    fn forward(&mut self, m: &ClipModel, x: &[f32], t: usize, penultimate: Option<&mut Vec<f32>>, out: &mut Vec<f32>) -> bool;
}

/// One gated attention of LTX-2.5's transformer (`crate::ltx::dit`): q / k / v / out projections
/// with biases, RMS norms of the queries and keys over all heads (weights `qn` / `kn`), the
/// per-head gate `2 sigmoid(gate_w x + gate_b)` (`[heads, q_dim]` float32).
pub struct LtxAttn<'a> {
    pub q: Lin<'a>,
    pub k: Lin<'a>,
    pub v: Lin<'a>,
    pub o: Lin<'a>,
    pub gate_w: &'a [f32],
    pub gate_b: &'a [f32],
    pub qn: &'a [f32],
    pub kn: &'a [f32],
    pub heads: usize,
    pub hd: usize,
}

/// A GELU (tanh) feed-forward.
pub struct LtxFf<'a> {
    pub up: Lin<'a>,
    pub down: Lin<'a>,
}

/// One stream's half of a block: self-attention, text cross-attention, feed-forward.
pub struct LtxStream<'a> {
    pub attn1: LtxAttn<'a>,
    pub attn2: LtxAttn<'a>,
    pub ff: LtxFf<'a>,
}

pub struct LtxBlock<'a> {
    pub v: LtxStream<'a>,
    pub a: LtxStream<'a>,
    /// Video queries over audio rows, audio queries over video rows.
    pub a2v: LtxAttn<'a>,
    pub v2a: LtxAttn<'a>,
}

pub struct LtxModel<'a> {
    pub dim: usize,
    pub audio_dim: usize,
    pub eps: f32,
    pub blocks: Vec<LtxBlock<'a>>,
    /// Attention with half-precision operands for 128-wide heads (else float32 ones).
    pub half_attention: bool,
}

/// One stream of an LTX-2.5 forward: its rows after the patch projection, each row's
/// modulation class and the per-block tables of this step.
pub struct LtxIn<'a> {
    /// `[rows, dim]`.
    pub x: &'a [f32],
    pub rows: usize,
    /// Each row's class (its timestep), indexing the tables below.
    pub cls: &'a [u32],
    pub classes: usize,
    /// Per block, per class, the 9 rows (shift, scale, gate of the self-attention, of the
    /// feed-forward, of the text cross-attention's queries) of `dim`, the block's table added.
    pub mods: &'a [f32],
    /// Per block, per class, the 4 AV cross rows (scale, shift a2v; scale, shift v2a).
    pub av: &'a [f32],
    /// Per block, the AV cross gate (`dim`).
    pub av_gate: &'a [f32],
    /// Per block, the text rows' shift and scale (`2 dim`).
    pub prompt: &'a [f32],
    /// The text context (`[n_ctx, dim]`).
    pub ctx: &'a [f32],
    pub n_ctx: usize,
    /// Rotary tables of the stream's own positions (`[rows, dim / 2]`) and of time alone for the
    /// AV cross-attention (`[rows, audio_dim / 2]`).
    pub cos: &'a [f32],
    pub sin: &'a [f32],
    pub ccos: &'a [f32],
    pub csin: &'a [f32],
}

/// The attention perturbations of a guidance pass: per block, whether each stream's
/// self-attention is skipped (its value projection passed straight to the output projection, no
/// gate), and whether the audio-video cross-attention is skipped in every block.
#[derive(Clone, Debug, Default)]
pub struct LtxPerturb {
    pub video_self: Vec<bool>,
    pub audio_self: Vec<bool>,
    pub cross: bool,
}

/// LTX-2.5 transformer passes on the device: every block of both streams, the activations kept
/// in device buffers.
pub trait LtxSession: Send {
    /// The rows of both streams after the last block (`[rows, dim]` each) into `vout` / `aout`;
    /// false when the device declined or failed (the caller runs the CPU path).
    fn forward(&mut self, m: &LtxModel, v: &LtxIn, a: &LtxIn, p: &LtxPerturb, vout: &mut Vec<f32>, aout: &mut Vec<f32>) -> bool;

    /// A text connector's blocks (`x += attn1(rms(x))`, `x += ff(rms(x))`, the attention's
    /// queries and keys rotated by `cos` / `sin`, `[rows, dim / 2]`) over `x` (`[rows, dim]`),
    /// the rows after the last block into `out` (the final norm is the caller's). False when the
    /// device declined or failed.
    #[allow(clippy::too_many_arguments)]
    fn connector(
        &mut self,
        _blocks: &[LtxConnBlock],
        _x: &[f32],
        _rows: usize,
        _cos: &[f32],
        _sin: &[f32],
        _eps: f32,
        _half_attention: bool,
        _out: &mut Vec<f32>,
    ) -> bool {
        false
    }
}

/// A block of LTX-2.5's text connectors (`Embeddings1DConnector`): gated self-attention and a
/// GELU feed-forward.
pub struct LtxConnBlock<'a> {
    pub attn: LtxAttn<'a>,
    pub ff: LtxFf<'a>,
}

/// A 3 x 3 x 3 convolution of LTX-2.5's video VAE (`crate::ltx::vae`): bfloat16 weights
/// `[n_out][kt][ky][kx][cin]` (`n_out` the outputs rounded up to 64, the extra rows zero) and
/// the bias (`n_out` values).
pub struct LtxVaeConv<'a> {
    pub w: &'a [u16],
    pub bias: &'a [f32],
    pub cin: usize,
    pub cout: usize,
}

/// A step of the VAE decoder after its input convolution.
pub enum LtxVaeOp<'a> {
    /// A residual block: `x + conv2(silu(norm(conv1(silu(norm(x))))))`, `norm` the RMS over
    /// the channels (`PixelNorm`, eps 1e-8).
    Res(LtxVaeConv<'a>, LtxVaeConv<'a>),
    /// A convolution, then depth-to-space by the strides (time, y, x), the first frame dropped
    /// when time doubles.
    D2s(LtxVaeConv<'a>, [usize; 3]),
}

/// LTX-2.5's convolutional video VAE decoder from its un-normalized latents to frames:
/// `conv_in`, the ops, norm and SiLU, `conv_out`, the `patch x patch` depth-to-space.
pub struct LtxVaeDecoder<'a> {
    pub conv_in: LtxVaeConv<'a>,
    pub ops: Vec<LtxVaeOp<'a>>,
    pub conv_out: LtxVaeConv<'a>,
    pub patch: usize,
    /// Time padding: the first frame twice in front (causal) or the first and last frames
    /// repeated (else).
    pub causal: bool,
}

/// Whole LTX-2.5 video VAE decodes on the device.
pub trait LtxVaeSession: Send {
    /// Frames `[t', h 32, w 32, 3]` (in `[-1, 1]` before clamping, rows of 3) into `out` from
    /// un-normalized latents `lat` (`[t, h, w, channels]` rows); `progress(done, total)` after
    /// each op. False when the device declined or failed (the caller runs the CPU path).
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &mut self,
        m: &LtxVaeDecoder,
        lat: &[f32],
        t: usize,
        h: usize,
        w: usize,
        out: &mut Vec<f32>,
        progress: &mut dyn FnMut(usize, usize),
    ) -> bool;

    /// The encoder's latent means (`[t', h', w', m.latent]` rows, before the per-channel
    /// normalization) into `out` from patchified frames `x` (`[t, h, w, m.conv_in.cin]` rows, the
    /// channels zero-padded to the convolution's); false when declined.
    #[allow(clippy::too_many_arguments)]
    fn encode(
        &mut self,
        _m: &LtxVaeEncoder,
        _x: &[f32],
        _t: usize,
        _h: usize,
        _w: usize,
        _out: &mut Vec<f32>,
        _progress: &mut dyn FnMut(usize, usize),
    ) -> bool {
        false
    }

    /// The latent upsampler's output (`[t', h', w', m.conv_out.cout]` rows, un-normalized) into
    /// `out` from un-normalized latents `lat` (`[t, h, w, m.conv_in.cin]` rows); false when
    /// declined.
    #[allow(clippy::too_many_arguments)]
    fn upsample(
        &mut self,
        _m: &LtxUpsampler,
        _lat: &[f32],
        _t: usize,
        _h: usize,
        _w: usize,
        _out: &mut Vec<f32>,
        _progress: &mut dyn FnMut(usize, usize),
    ) -> bool {
        false
    }
}

/// A group norm of LTX-2.5's latent upsamplers: `groups` groups of channels, each over every
/// frame and position (eps 1e-5), then the per-channel weight and bias.
pub struct LtxGroupNorm<'a> {
    pub w: &'a [f32],
    pub b: &'a [f32],
    pub groups: usize,
}

/// A residual block of the latent upsamplers: `silu(x + n2(c2(silu(n1(c1(x))))))`, the
/// convolutions zero-padded in time.
pub struct LtxUpRes<'a> {
    pub c1: LtxVaeConv<'a>,
    pub n1: LtxGroupNorm<'a>,
    pub c2: LtxVaeConv<'a>,
    pub n2: LtxGroupNorm<'a>,
}

/// LTX-2.5's latent upsampler (`crate::ltx::upsample`) over un-normalized latents: `conv_in`,
/// `silu(norm_in)`, the `pre` blocks, the upsampling (`spatial`: `up` a 3 x 3 convolution per
/// frame, its weights `[n_out][ky][kx][cin]`, then 2 x 2 depth-to-space; else a 3 x 3 x 3 one,
/// then depth-to-space in time with the first frame dropped), the `post` blocks, `conv_out`.
pub struct LtxUpsampler<'a> {
    pub conv_in: LtxVaeConv<'a>,
    pub norm_in: LtxGroupNorm<'a>,
    pub pre: Vec<LtxUpRes<'a>>,
    pub up: LtxVaeConv<'a>,
    pub spatial: bool,
    pub post: Vec<LtxUpRes<'a>>,
    pub conv_out: LtxVaeConv<'a>,
}

/// A step of the VAE encoder after its input convolution.
pub enum LtxVaeEncOp<'a> {
    /// A residual block (as the decoder's).
    Res(LtxVaeConv<'a>, LtxVaeConv<'a>),
    /// Space-to-depth by the strides (time, y, x) of a convolution's output plus the mean of each
    /// output channel's `group` channels of the input's space-to-depth; the first frame repeated
    /// in front when time halves.
    S2d(LtxVaeConv<'a>, [usize; 3], usize),
}

/// LTX-2.5's convolutional video VAE encoder (causal in time) after the input's patchifying:
/// `conv_in`, the ops, norm and SiLU, `conv_out`, of whose outputs the first `latent` are the
/// means.
pub struct LtxVaeEncoder<'a> {
    pub conv_in: LtxVaeConv<'a>,
    pub ops: Vec<LtxVaeEncOp<'a>>,
    pub conv_out: LtxVaeConv<'a>,
    pub latent: usize,
}

/// A layer of a BigVGAN generator as shifted GEMMs (`crate::ltx::audio`): bfloat16 weights
/// `[n_out][taps][cin]` (`n_out` the outputs rounded up to 64, the extra rows zero), the bias
/// (`n_out` values) and the input row of each tap relative to the output's (a 1-D convolution:
/// its dilated taps; a phase of a transposed convolution: its input steps).
pub struct LtxVocLin<'a> {
    pub w: &'a [u16],
    pub bias: &'a [f32],
    pub cin: usize,
    pub cout: usize,
    pub offs: &'a [i32],
}

/// An anti-aliased SnakeBeta activation: `alpha` then `1 / beta` per channel, the 2x upsampling
/// filter (times 2) then the downsampling one, `k` taps each.
pub struct LtxVocAct<'a> {
    pub ab: &'a [f32],
    pub filt: &'a [f32],
    pub k: usize,
}

/// One dilation of an AMP block: `x += conv2(act2(conv1(act1(x))))`.
pub struct LtxVocRes<'a> {
    pub act1: LtxVocAct<'a>,
    pub conv1: LtxVocLin<'a>,
    pub act2: LtxVocAct<'a>,
    pub conv2: LtxVocLin<'a>,
}

/// A generator stage: the transposed convolution (one layer per output phase: output step
/// `i stride + phase` from the phase's GEMM at input step `i`), then the mean of its AMP blocks.
pub struct LtxVocStage<'a> {
    pub up: Vec<LtxVocLin<'a>>,
    pub amps: Vec<Vec<LtxVocRes<'a>>>,
}

/// A BigVGAN v2 generator: `conv_pre`, the stages, the activation, `conv_post`.
pub struct LtxVocGen<'a> {
    pub pre: LtxVocLin<'a>,
    pub stages: Vec<LtxVocStage<'a>>,
    pub act_post: LtxVocAct<'a>,
    pub post: LtxVocLin<'a>,
}

/// Whole vocoder generator passes on the device.
pub trait LtxVocSession: Send {
    /// `post.cout` channels of `t * prod(strides)` steps (`[steps, channels]` rows, before any
    /// clamping) into `out` from `x` (`[t, pre.cin]` rows). False when the device declined.
    fn generate(&mut self, g: &LtxVocGen, x: &[f32], t: usize, out: &mut Vec<f32>) -> bool;
}

/// One neighborhood-attention block of LTX-2.5's DiffVAE decoder (`crate::ltx::diffvae`):
/// RMS norm (with weight), fused q / k / v projection (`[3 dim, dim]`, biases added), RMS norms
/// of each 64-wide head's queries (times `1 / 8`) and keys, absolute 3D RoPE, 3D neighborhood
/// attention, the output projection; then RMS norm and a SwiGLU without biases. Stage-5 blocks
/// first add `ctx (ctx rows)` and modulate both norms `* (1 + scale) + shift`.
pub struct NaBlock<'a> {
    pub dim: usize,
    pub kernel: [usize; 3],
    /// Keys extend half a window past both width edges (the edge columns repeated at their own
    /// positions), else NATTEN's inward-shifted windows on every axis.
    pub replicate_w: bool,
    pub norm1: &'a [f32],
    pub norm2: &'a [f32],
    pub qkv: Lin<'a>,
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub proj: Lin<'a>,
    pub gate: Lin<'a>,
    pub up: Lin<'a>,
    pub down: Lin<'a>,
    /// Stage 5: the context projection (with its bias).
    pub ctx: Option<Lin<'a>>,
    /// Stage 5: scale and shift of the attention's norm, of the MLP's.
    pub mods: Option<[&'a [f32]; 4]>,
}

/// LTX-2.5 DiffVAE blocks on the device: the rows stay in device buffers through a run of
/// blocks.
pub trait NaSession: Send {
    /// `x` (`[t h w, dim]` rows over the `d = [t, h, w]` grid) through `blocks` in place, with the
    /// stage-5 context rows `ctx` (same grid). False when the device declined or failed (`x` is
    /// then unchanged).
    fn blocks(&mut self, blocks: &[NaBlock], x: &mut [f32], ctx: Option<&[f32]>, d: [usize; 3]) -> bool;
}

/// One block of MiniMax-H3's transformer (`crate::minimax_h3::dit`): RMS norms with weights
/// under per-row AdaLN sets, a fused q / k / v projection, per-head RMS norms of queries and
/// keys, a 3-axis RoPE on the first `2 * rope_half` channels of each head (half against half),
/// joint attention without a mask, a SwiGLU feed-forward, gated residuals. Its video VAE
/// decoder's blocks (`crate::minimax_h3::vae`) are the same with biases, no AdaLN and residuals
/// scaled by fixed vectors (`scale`), 64-wide heads.
pub struct H3Block<'a> {
    pub norm1: &'a [f32],
    pub norm2: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    /// `[3 inner, hidden]`: `[q; k; v]` rows (`stacked`) or `[q k v]` per head.
    pub qkv: Lin<'a>,
    pub out: Lin<'a>,
    /// `[2 ffn, hidden]`: gate rows first (`gate_first`) or value rows first.
    pub fc1: Lin<'a>,
    pub fc2: Lin<'a>,
    /// The residual scales of the attention and of the feed-forward (none: AdaLN).
    pub scale: Option<(&'a [f32], &'a [f32])>,
    /// Scales of the outputs of qkv, out, fc1, fc2 (`[n_out]`), applied after their GEMMs and
    /// before their side paths: the weights' own scales, when the device weights have unit ones.
    pub out_scale: [Option<&'a [f32]>; 4],
}

/// How a device runs an attention: half-precision operands, float32 ones, or queries and keys
/// as pairs of halves (hi + lo) on the half-precision matrix units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnPrecision {
    Half,
    F32,
    Split,
}

impl AttnPrecision {
    /// `half`, `split` or `f32` (`exact`, `float32`).
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "half" | "f16" => Self::Half,
            "split" | "hilo" => Self::Split,
            "f32" | "float32" | "exact" => Self::F32,
            _ => return None,
        })
    }
}

/// How a device feeds the projections' matrix-unit GEMMs: each input row as scaled halves
/// (`Half`), or as a pair of them, hi + lo (`Split`: two GEMMs into the same output, about 22
/// significant bits of each input), for models that amplify the half inputs' rounding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputPrecision {
    Half,
    Split,
}

impl InputPrecision {
    /// `half` or `split`.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "half" | "f16" => Self::Half,
            "split" | "hilo" => Self::Split,
            _ => return None,
        })
    }
}

pub struct H3Model<'a> {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub rope_half: usize,
    pub eps: f32,
    pub qk_eps: f32,
    pub stacked: bool,
    pub gate_first: bool,
    pub blocks: Vec<H3Block<'a>>,
    pub attention: AttnPrecision,
    /// The projections' inputs.
    pub inputs: InputPrecision,
}

/// One forward of the block stack.
pub struct H3In<'a> {
    /// `[rows, hidden]`.
    pub x: &'a [f32],
    pub rows: usize,
    /// Each row's AdaLN set (empty without AdaLN).
    pub key: &'a [u32],
    pub sets: usize,
    /// Per block `[sets, 6 hidden]`: shift, scale, gate of the attention, then of the
    /// feed-forward (empty without AdaLN).
    pub tables: &'a [Vec<f32>],
    /// Per row `rope_half` cosines, and as many sines.
    pub cos: &'a [f32],
    pub sin: &'a [f32],
    /// Blocks bypassed (the identity).
    pub skip: &'a [usize],
}

/// MiniMax-H3 block-stack passes on the device, the rows kept in device buffers.
pub trait H3Session: Send {
    /// The rows after the last block (`[rows, hidden]`) into `out`; false when the device
    /// declined or failed (the caller runs the CPU path).
    fn forward(&mut self, m: &H3Model, inp: &H3In, out: &mut Vec<f32>) -> bool;
}

/// A float32 linear layer of the H3 corrector as two float16 parts (`w = hi + lo`, 22
/// significant bits): `lo` none when the layer is exact in float16.
pub struct SplitLin<'a> {
    pub hi: Lin<'a>,
    pub lo: Option<Lin<'a>>,
}

/// One block of SulphurAI's H3 corrector (`crate::minimax_h3::corrector`): LayerNorms with
/// weights and biases, AdaLN-modulated self-attention over 64-wide heads with rotary pairs,
/// cross-attention to the caption, a tanh-GELU MLP; biases on every linear layer.
pub struct CorrBlock<'a> {
    /// LayerNorm weight and bias of the self-attention, the cross-attention and the MLP.
    pub ln: [[&'a [f32]; 2]; 3],
    /// `[q; k; v]` rows.
    pub qkv: SplitLin<'a>,
    pub out: SplitLin<'a>,
    pub cq: SplitLin<'a>,
    pub ck: SplitLin<'a>,
    pub cv: SplitLin<'a>,
    pub cout: SplitLin<'a>,
    pub fc1: SplitLin<'a>,
    pub fc2: SplitLin<'a>,
}

pub struct CorrModel<'a> {
    pub width: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub eps: f32,
    pub blocks: Vec<CorrBlock<'a>>,
}

/// One corrector forward.
pub struct CorrIn<'a> {
    /// The tokens after `patch_in` (`[rows, width]`).
    pub x: &'a [f32],
    pub rows: usize,
    /// The caption after its norm and projection (`[ctx, width]`).
    pub ctx: &'a [f32],
    pub n_ctx: usize,
    /// Per block the modulation of this step: shift, scale, gate of the self-attention, then of
    /// the MLP (`6 width`).
    pub mods: &'a [f32],
    /// Per token the angle of each rotary pair of a head (`head_dim / 2` cosines, as many sines).
    pub cos: &'a [f32],
    pub sin: &'a [f32],
}

/// H3 corrector passes on the device, the tokens kept in device buffers.
pub trait CorrSession: Send {
    /// The tokens after the last block (`[rows, width]`) into `out`; false when the device
    /// declined or failed (the caller runs the CPU path).
    fn forward(&mut self, m: &CorrModel, inp: &CorrIn, out: &mut Vec<f32>) -> bool;
}
