//! What an accelerator needs to run a stack of MiniMax-H3's transformer blocks (the DiT's and the
//! video VAE decoder's are the same block: RMS norm, a fused q / k / v projection, per-head
//! query / key norms and RoPE, attention, a gated residual, then the same around a SwiGLU
//! feed-forward), and the slot a model keeps its engine in. The engine is `cuda.rs`; Metal runs the
//! same stacks through `crate::gpu::dit::H3Session` (`src/metal_h3.rs`).

#![cfg_attr(
    not(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32")))),
    allow(dead_code)
)]

use std::sync::OnceLock;

/// The stack's sizes and conventions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StackSpec {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    /// Rotated channel pairs per head (the first `2 * half` channels, half against half).
    pub half: usize,
    pub norm_eps: f32,
    pub qk_eps: f32,
    /// The fused projection's rows are `[q; k; v]` (else `[q k v]` per head).
    pub stacked: bool,
    /// The feed-forward's first layer has its gate rows first (else its value rows).
    pub gate_first: bool,
}

/// A projection's weights as stored.
#[derive(Clone, Copy)]
pub(crate) enum WSpec<'d> {
    /// FP8 E4M3 bytes (`[n_out, n_in]`) times one scale.
    F8(&'d [u8], f32),
    /// Little-endian float16 (`[n_out, n_in]`).
    F16(&'d [u8]),
}

#[derive(Clone, Copy)]
pub(crate) struct LinSpec<'d> {
    pub w: WSpec<'d>,
    pub bias: Option<&'d [f32]>,
    pub n_out: usize,
    pub n_in: usize,
    /// Low-rank side paths (chained).
    pub lora: Option<&'d crate::lora::LoraDelta>,
}

/// One block. With AdaLN (`scale` None) the caller passes each block's table of shift / scale /
/// gate sets; otherwise the two residuals are scaled by `scale` and nothing is modulated.
pub(crate) struct BlockSpec<'d> {
    /// The fused q / k / v projection, the attention's output, the feed-forward's two layers.
    pub lins: [LinSpec<'d>; 4],
    pub norm1: &'d [f32],
    pub norm2: &'d [f32],
    /// Weights of the per-head RMS norms of queries and keys (none: no weights).
    pub norm_qk: Option<(&'d [f32], &'d [f32])>,
    pub scale: Option<(&'d [f32], &'d [f32])>,
}

/// A model whose block stack an engine can run.
pub(crate) trait Source: Sync {
    fn spec(&self) -> StackSpec;
    fn n_blocks(&self) -> usize;
    /// Block `i`, or None when the engine cannot take it (weights in another format, q / k / v as
    /// separate matrices).
    fn block(&self, i: usize) -> Option<BlockSpec<'_>>;
}

/// The AdaLN inputs of a stack: each block's sets (`[sets, 6 H]`: shift, scale, gate of the
/// attention then of the feed-forward) and each row's set.
#[derive(Clone, Copy)]
pub(crate) struct AdaIn<'t> {
    pub tables: &'t [Vec<f32>],
    pub key: &'t [usize],
}

pub(crate) trait Engine: Send + Sync {
    /// The block stack over `x` (`[n, H]`); `cos` / `sin`: `half` values per row. False, with `x`
    /// untouched, when the device does not take the call (the host then runs it).
    fn stack(&self, src: &dyn Source, x: &mut [f32], ada: Option<AdaIn>, cos: &[f32], sin: &[f32]) -> bool;
    /// Frees the device memory (weights and activations); the next call brings them back.
    fn release(&self);
}

/// A model's engine: decided at the first use (the active backend and the weights' formats).
#[derive(Default)]
pub(crate) struct Device(OnceLock<Option<Box<dyn Engine>>>);

impl Device {
    pub(crate) fn get(&self, src: &dyn Source, what: &'static str) -> Option<&dyn Engine> {
        self.0
            .get_or_init(|| {
                #[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
                if let Some(e) = super::cuda::Engine::new(src, what) {
                    return Some(Box::new(e) as Box<dyn Engine>);
                }
                let _ = (src, what);
                None
            })
            .as_deref()
    }

    /// Drops the engine (the model's weights changed).
    pub(crate) fn reset(&mut self) {
        self.0 = OnceLock::new();
    }

    /// Frees the engine's device memory, keeping the engine.
    pub(crate) fn release(&self) {
        if let Some(Some(e)) = self.0.get() {
            e.release();
        }
    }
}
