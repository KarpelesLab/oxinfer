//! Whole convolutional UNet passes on an accelerator (Stable Diffusion 1.x / 2.x / XL,
//! `crate::sd::unet`): the model describes one pass as a graph of ops over image tensors
//! ([`Graph`]), the backend runs it for a batch of images (the guidance branches of a step) in a
//! few submissions with the activations kept on the device, and only the outputs come back
//! ([`Session::forward`]).

use super::{DeviceWeight, WeightRef};
use std::borrow::Cow;

/// A linear layer (or a convolution's `[n_out][ky][kx][cin]` matrix) as the device sees it.
pub struct Lin<'a> {
    pub w: WeightRef<'a>,
    pub dev: &'a DeviceWeight,
    pub n_out: usize,
    pub n_in: usize,
    pub bias: Option<&'a [f32]>,
}

/// Group norm (`groups` > 0) or layer norm (`groups` 0) parameters.
#[derive(Clone, Copy)]
pub struct Norm<'a> {
    pub gamma: &'a [f32],
    pub beta: &'a [f32],
    pub eps: f32,
    pub groups: usize,
}

/// Self- or cross-attention: queries from the normalized hidden states, keys and values from
/// them too or from the text context.
pub struct Attn<'a> {
    pub q: Lin<'a>,
    pub k: Lin<'a>,
    pub v: Lin<'a>,
    pub o: Lin<'a>,
    pub heads: usize,
}

/// diffusers `BasicTransformerBlock`: self-attention, cross-attention, GEGLU feed-forward, each
/// after a layer norm and added to the hidden states.
pub struct Block<'a> {
    pub n1: Norm<'a>,
    pub a1: Attn<'a>,
    pub n2: Norm<'a>,
    pub a2: Attn<'a>,
    pub n3: Norm<'a>,
    /// `[2 inner, dim]`: value rows then gate rows.
    pub ff1: Lin<'a>,
    pub ff2: Lin<'a>,
}

/// diffusers `Transformer2DModel`: `x + proj_out(blocks(proj_in(group_norm(x))))`.
pub struct Transformer<'a> {
    pub norm: Norm<'a>,
    pub proj_in: Lin<'a>,
    pub blocks: Vec<Block<'a>>,
    pub proj_out: Lin<'a>,
}

/// One step of a pass. Tensors are indices into [`Graph::tensors`].
pub enum Op<'a> {
    /// `out = conv(pre(x)) + bias (+ add)`: a k x k convolution (zero padding `pad` before, k - 1 -
    /// pad after; `w` in the unfolded layout, `n_in = k k cin`), its input normalized first with
    /// `pre` (and SiLU'd with its flag), or with `up` upsampled 2x (nearest) first. `bias` is
    /// `n_out` values, or with `bias_per_image` `n_out` per image of the batch (a ResNet's
    /// convolution bias plus its time-embedding projection).
    Conv {
        x: usize,
        out: usize,
        w: Lin<'a>,
        k: usize,
        stride: usize,
        pad: usize,
        up: bool,
        pre: Option<(Norm<'a>, bool)>,
        bias: Option<Cow<'a, [f32]>>,
        bias_per_image: bool,
        add: Option<usize>,
    },
    /// `out = [a | b]` per pixel.
    Concat { a: usize, b: usize, out: usize },
    /// `out = x + transformer(x)`, cross-attending to the batch's text contexts.
    Transformer { x: usize, out: usize, t: Transformer<'a> },
    /// `out = x + o(attention(q, k, v of group_norm(x)))` over the pixels of each image (a VAE's
    /// mid-block self-attention; `a.k` and `a.v` read the normalized pixels too).
    Attention { x: usize, out: usize, norm: Norm<'a>, a: Attn<'a> },
}

/// A pass: tensor shapes per image (`[h, w, channels]`), the ops in order, which tensor receives
/// the input latents and which one is the output.
pub struct Graph<'a> {
    pub tensors: Vec<[usize; 3]>,
    pub ops: Vec<Op<'a>>,
    pub input: usize,
    pub output: usize,
    /// Channels of the text context.
    pub ctx_dim: usize,
}

/// One image of a batch: its input latents (`[h w, channels]`) and text context (`[ctx_len,
/// ctx_dim]`).
pub struct Item<'a> {
    pub latent: &'a [f32],
    pub ctx: &'a [f32],
    pub ctx_len: usize,
}

/// Device state kept between passes (activation buffers).
pub trait Session: Send {
    /// Runs `g` for every item of the batch into `out` (one `[h w, channels]` of the output tensor
    /// per item). False when the device declined or failed (the caller runs the CPU path).
    fn forward(&mut self, g: &Graph, items: &[Item], out: &mut [Vec<f32>]) -> bool;
}
