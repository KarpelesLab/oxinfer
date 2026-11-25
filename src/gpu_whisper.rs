//! Whole Whisper encoder passes on an accelerator: the model describes its encoder once per call
//! ([`Encoder`]), the backend runs the convolutional stem and every transformer layer of every
//! window of a batch with the activations kept on the device, and only the encoder output comes
//! back ([`Session::encode`]).
//!
//! The layout is `crate::whisper::model`'s: two convolutions (kernel 3, the second with stride 2)
//! as GEMMs over unfolded rows (`[cout][k][cin]` weights, conv1's rows padded to `conv1.n_in`),
//! exact GELU, sinusoidal positions, pre-LayerNorm layers with biased projections (the key
//! projection has none), unmasked attention over the window, a final LayerNorm.

use super::{DeviceWeight, WeightRef};

/// A linear layer as the device sees it.
pub struct Lin<'a> {
    pub w: WeightRef<'a>,
    pub dev: &'a DeviceWeight,
    pub n_out: usize,
    pub n_in: usize,
    pub bias: Option<&'a [f32]>,
}

/// A LayerNorm's weight and bias.
#[derive(Clone, Copy)]
pub struct Norm<'a> {
    pub w: &'a [f32],
    pub b: &'a [f32],
}

pub struct Layer<'a> {
    pub ln1: Norm<'a>,
    pub q: Lin<'a>,
    pub k: Lin<'a>,
    pub v: Lin<'a>,
    pub o: Lin<'a>,
    pub ln2: Norm<'a>,
    pub fc1: Lin<'a>,
    pub fc2: Lin<'a>,
}

pub struct Encoder<'a> {
    pub d: usize,
    pub heads: usize,
    pub ffn: usize,
    pub n_mels: usize,
    /// Output positions per window (half the mel frames).
    pub n: usize,
    pub conv1: Lin<'a>,
    pub conv2: Lin<'a>,
    /// `[n, d]` positions.
    pub pos: &'a [f32],
    pub layers: Vec<Layer<'a>>,
    pub ln_post: Norm<'a>,
    pub eps: f32,
    /// Attention with half-precision operands (else float32 ones).
    pub half_attention: bool,
}

/// The cross-attention key and value projections of one decoder layer (the key has no bias).
pub struct CrossLin<'a> {
    pub k: Lin<'a>,
    pub v: Lin<'a>,
}

pub struct DecLayer<'a> {
    pub ln1: Norm<'a>,
    pub q: Lin<'a>,
    pub k: Lin<'a>,
    pub v: Lin<'a>,
    pub o: Lin<'a>,
    pub ln_cross: Norm<'a>,
    pub cq: Lin<'a>,
    pub co: Lin<'a>,
    pub ln2: Norm<'a>,
    pub fc1: Lin<'a>,
    pub fc2: Lin<'a>,
}

/// The decoder: pre-LayerNorm layers of causal self-attention over a per-sequence cache,
/// cross-attention over a window's keys and values (from [`Session::encode`]) and a GELU
/// feed-forward, a final LayerNorm and the output projection (`head`, `[vocab, d]`, no bias).
/// Queries are scaled by `head_dim^-0.5` after their bias.
pub struct Decoder<'a> {
    pub d: usize,
    pub heads: usize,
    pub ffn: usize,
    pub vocab: usize,
    /// Positions of a sequence at most.
    pub max_pos: usize,
    pub layers: Vec<DecLayer<'a>>,
    pub ln: Norm<'a>,
    pub head: Lin<'a>,
    pub eps: f32,
}

/// Device memory a caller holds: a window's cross-attention keys and values, or a sequence's
/// self-attention cache. Released when dropped.
pub trait DevMem: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    /// A copy of a self-attention cache's first `positions` positions (beam search continuing a
    /// beam several times); None when the device is out of memory.
    fn duplicate(&self, positions: usize) -> Option<Box<dyn DevMem>>;
}

/// One new token of a decoder step: its input row `x` (`d` values: token embedding and
/// position), its position, its sequence's cache and its window's cross-attention keys and values.
pub struct Row<'a> {
    pub x: &'a [f32],
    pub pos: usize,
    pub kv: &'a dyn DevMem,
    pub cross: &'a dyn DevMem,
}

/// Greedy decoding chained on the device ([`Session::greedy`]): the token choice of
/// `crate::whisper`'s greedy search (suppressed tokens, the timestamp rules, the most likely
/// token, its log-probability) runs there too, each step's token the next step's input.
pub struct Greedy<'a> {
    /// The input of token `t` at position `p`: row `t` of the decoder's output projection (bf16)
    /// times `embed_scale`, plus row `p` of `pos` (`[max_pos, d]`).
    pub embed_scale: f32,
    pub pos: &'a [f32],
    /// Per token: bit 0 suppressed, bit 1 suppressed as the first generated token.
    pub mask: &'a [u8],
    pub timestamp_begin: u32,
    pub eot: u32,
    pub no_timestamps: u32,
    /// The latest timestamp allowed first (index after `timestamp_begin`).
    pub max_initial: Option<u32>,
}

/// One sequence of a chain.
pub struct ChainRow<'a> {
    pub kv: &'a dyn DevMem,
    pub cross: &'a dyn DevMem,
    /// The tokens generated so far (at least one: the last is the next to decode, at `pos`).
    pub tokens: &'a [u32],
    pub pos: usize,
    /// Whether the timestamp rules apply.
    pub timestamps: bool,
}

/// Device state kept between calls (activation buffers).
pub trait Session: Send {
    /// Encodes the windows of log-mel frames `mels` (each `[2 n, n_mels]`, time-major) into `out`
    /// (one `[n, d]` each) and, with `cross` (one per decoder layer), each window's
    /// cross-attention keys and values on the device. Every window gets the values it gets alone.
    /// None when the device declined or failed (the caller runs the per-op path).
    fn encode(&mut self, m: &Encoder, mels: &[&[f32]], out: &mut [Vec<f32>], cross: &[CrossLin]) -> Option<Vec<Box<dyn DevMem>>>;
    /// Whether [`Session::decode`] takes this decoder.
    fn decodes(&self, _dec: &Decoder) -> bool {
        false
    }
    /// An empty self-attention cache for one sequence.
    fn new_kv(&mut self, _dec: &Decoder) -> Option<Box<dyn DevMem>> {
        None
    }
    /// One decoder pass over `rows` (the rows of one sequence consecutive, at consecutive
    /// positions, each attending to its sequence's cached positions and the rows before it), the
    /// new keys and values going to the caches; the logits of rows `want` (`vocab` each, in that
    /// order) into `logits`. A row gets the same values whatever rows it runs with. False when
    /// the device failed.
    fn decode(&mut self, _dec: &Decoder, _rows: &[Row], _want: &[usize], _logits: &mut [f32]) -> bool {
        false
    }
    /// Whether [`Session::greedy`] takes this decoder.
    fn chains(&self, _dec: &Decoder) -> bool {
        false
    }
    /// `steps` greedy steps of every row: decode the row's last token, choose the next one from
    /// the logits (as `crate::whisper` does on the host), and so on; row `i`'s tokens and their
    /// log-probabilities into `out[i * steps..]`, whether or not the row ended on the way. Each
    /// row gets the same tokens whatever rows run with it. False when the device failed (the
    /// caches then hold what the host must not rely on).
    fn greedy(&mut self, _dec: &Decoder, _g: &Greedy, _rows: &[ChainRow], _steps: usize, _out: &mut [(u32, f32)]) -> bool {
        false
    }
}
