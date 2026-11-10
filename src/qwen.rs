//! Qwen3.5 / Qwen3.8 dense hybrid text model: Gated DeltaNet (linear attention with a causal
//! short conv and a delta-rule recurrent state) interleaved with gated full attention, SiLU MLPs
//! and zero-centered RMSNorm; plain Qwen3 (full attention only) runs on the same code, as do
//! Llama 3 (full attention without query / key norms, llama3 RoPE scaling) and Qwen3.5 MoE (top-k
//! routed SiLU experts plus a sigmoid-gated shared expert in place of the MLP).
//! The vision encoder is in `qwen_vision.rs`; its tokens are placed with M-RoPE (interleaved
//! time / height / width rotary positions, see `QwenSession::add_vision_span`). The MTP head for
//! speculative decoding is in `qwen_mtp.rs`.

use crate::config::QwenConfig;
use crate::intervene::{NeuronHook, Target};
use crate::kernels::*;
use crate::model::{LayerCache, Lin, Quant, Table, attention, linear, linear_multi, load_lin, table_quant};
use crate::moe::{Act, Experts, route_top_k};
use crate::pool;
use crate::prof::{self, Cat};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

pub(crate) struct Gdn<'a> {
    pub(crate) qkv: Lin<'a>,         // [2*key_dim + value_dim, H]
    pub(crate) z: Lin<'a>,           // [value_dim, H]
    pub(crate) b: Lin<'a>,           // [n_v, H]
    pub(crate) a: Lin<'a>,           // [n_v, H]
    pub(crate) conv_w: Vec<f32>,     // [conv_dim, K]
    pub(crate) decay_coef: Vec<f32>, // -exp(A_log) per v head
    pub(crate) dt_bias: Vec<f32>,
    pub(crate) norm_w: Vec<f32>, // [hd_v], plain weight (gated norm)
    pub(crate) out: Lin<'a>,     // [H, value_dim]
}

pub(crate) struct Attn<'a> {
    pub(crate) q: Lin<'a>, // [nh*hd*2, H] (query and gate interleaved per head)
    pub(crate) k: Lin<'a>,
    pub(crate) v: Lin<'a>,
    pub(crate) o: Lin<'a>,
    /// 1 + w (or w, see `zero_centered_norm`); None without query / key norms (Llama).
    pub(crate) q_norm: Option<Vec<f32>>,
    pub(crate) k_norm: Option<Vec<f32>>,
    /// Query, key and value biases (Qwen2 / Qwen2.5-VL), added after the projections.
    pub(crate) bias: Option<[Vec<f32>; 3]>,
}

// One per layer, built once: the size difference is irrelevant.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Mixer<'a> {
    Gdn(Gdn<'a>),
    Attn(Attn<'a>),
}

pub(crate) struct Mlp<'a> {
    pub(crate) gate: Lin<'a>,
    pub(crate) up: Lin<'a>,
    pub(crate) down: Lin<'a>,
    pub(crate) inter: usize,
}

impl Mlp<'_> {
    fn bytes(&self) -> usize {
        self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }
    /// `down(silu(gate(x)) * up(x))` for `t` rows of `x` (`hook`: neuron edits of `target`).
    fn forward(&self, x: &[f32], t: usize, h: usize, y: &mut [f32], hook: Option<(&NeuronHook, Target)>) {
        let mut g = vec![0f32; t * self.inter];
        let mut u = vec![0f32; t * self.inter];
        linear_multi(
            &mut [(&self.gate, self.inter, &mut g[..]), (&self.up, self.inter, &mut u[..])],
            h,
            x,
            t,
        );
        silu_mul(&mut g, &u);
        if let Some((hk, tg)) = hook {
            hk.apply(tg, &mut g, self.inter);
        }
        linear(&self.down, h, self.inter, &g, t, y);
    }
}

/// Sparse MoE block: softmax router, top-k experts with renormalized weights, plus a shared
/// expert scaled by `sigmoid(shared_gate . x)`.
pub(crate) struct Moe<'a> {
    pub(crate) router: Vec<f32>, // [n_exp, H]
    pub(crate) experts: Experts<'a>,
    pub(crate) shared: Mlp<'a>,
    pub(crate) shared_gate: Vec<f32>, // [H]
    /// Qwen4-Exp: prefill passes take the router logits from the matmul path (on the GPU when
    /// there is one), through this f32 copy of `router` made on first use.
    pub(crate) router_mm: Option<std::sync::OnceLock<Lin<'static>>>,
}

// One per layer, built once.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Ffn<'a> {
    Dense(Mlp<'a>),
    Moe(Moe<'a>),
}

pub(crate) use crate::moe::NeedExperts;

impl<'a> Ffn<'a> {
    /// Every weight of the block (all experts of a MoE).
    pub(crate) fn lins(&self) -> Vec<&Lin<'a>> {
        let mut v = self.dense_lins();
        v.extend(self.expert_lins().into_iter().flatten());
        v
    }

    /// The weights every pass reads: all of a dense MLP, the shared expert of a MoE.
    pub(crate) fn dense_lins(&self) -> Vec<&Lin<'a>> {
        match self {
            Ffn::Dense(m) => vec![&m.gate, &m.up, &m.down],
            Ffn::Moe(m) => vec![&m.shared.gate, &m.shared.up, &m.shared.down],
        }
    }

    /// The routed experts' weights, per expert (none for a dense MLP).
    pub(crate) fn expert_lins(&self) -> Vec<Vec<&Lin<'a>>> {
        match self {
            Ffn::Dense(_) => Vec::new(),
            Ffn::Moe(m) => m.experts.lins(),
        }
    }

    /// The memory of each routed expert (`Lin::mem_ranges`), for streaming them on demand.
    pub(crate) fn expert_ranges(&self) -> Vec<Vec<(usize, usize)>> {
        self.expert_lins()
            .iter()
            .map(|e| e.iter().flat_map(|l| l.mem_ranges()).collect())
            .collect()
    }

    pub(crate) fn forward(&self, x: &[f32], t: usize, h: usize, y: &mut [f32], need: NeedExperts, hook: Option<&NeuronHook>) {
        match self {
            Ffn::Dense(m) => m.forward(x, t, h, y, hook.map(|k| (k, Target::Mlp))),
            Ffn::Moe(m) => m.forward(x, t, h, y, need, hook),
        }
    }

    /// The intermediate size of `target`, if the block has it.
    pub(crate) fn inter(&self, target: Target) -> Option<usize> {
        match (self, target) {
            (Ffn::Dense(m), Target::Mlp) => Some(m.inter),
            (Ffn::Moe(m), Target::Shared) => Some(m.shared.inter),
            (Ffn::Moe(m), Target::Expert(e)) if e < m.experts.n_experts() => Some(m.experts.inter),
            _ => None,
        }
    }

    /// Column `n` of `target`'s down projection.
    pub(crate) fn down_column(&self, target: Target, n: usize, h: usize) -> Vec<f32> {
        let (down, inter) = match (self, target) {
            (Ffn::Dense(m), _) => (&m.down, m.inter),
            (Ffn::Moe(m), Target::Shared | Target::Mlp) => (&m.shared.down, m.shared.inter),
            (Ffn::Moe(m), Target::Expert(e)) => (&m.experts.down[e], m.experts.inter),
        };
        let mut one = vec![0f32; inter];
        one[n] = 1.0;
        let mut y = vec![0f32; h];
        linear(down, h, inter, &one, 1, &mut y);
        y
    }

    /// Weight bytes read per token.
    pub(crate) fn streamed_bytes(&self) -> usize {
        match self {
            Ffn::Dense(m) => m.bytes(),
            // The shared expert and top_k of the routed ones.
            Ffn::Moe(m) => m.shared.bytes() + m.experts.bytes_per_token(),
        }
    }
}

impl<'a> Attn<'a> {
    pub(crate) fn bytes(&self) -> usize {
        self.q.bytes() + self.k.bytes() + self.v.bytes() + self.o.bytes()
    }
    pub(crate) fn lins(&self) -> [&Lin<'a>; 4] {
        [&self.q, &self.k, &self.v, &self.o]
    }
}

pub(crate) struct QLayer<'a> {
    pub(crate) mixer: Mixer<'a>,
    ln_in: Vec<f32>,   // 1 + w
    ln_post: Vec<f32>, // 1 + w
    pub(crate) ffn: Ffn<'a>,
}

pub struct QwenModel<'a> {
    pub cfg: QwenConfig,
    pub quant: Quant,
    pub(crate) embed: Table<'a>,
    pub(crate) layers: Vec<QLayer<'a>>,
    final_norm: Vec<f32>,
    pub(crate) lm_head: Lin<'a>,
    pub(crate) inv_freq: Vec<f32>,
    /// M-RoPE: the position axis (0 time, 1 height, 2 width) of each rotary frequency.
    mrope_axis: Option<Vec<u8>>,
    /// Weight streaming: unit `l` is layer `l`, unit `num_hidden_layers` the LM head.
    pub(crate) stream: Option<crate::stream::StreamHook>,
    /// Single-token decoding on the GPU (`crate::gpu::decode`), compiled on first use; None when
    /// the backend or the model cannot.
    /// Plans by intervention layout (`GpuLayout::signature`, 0 without interventions).
    gpu_plans: crate::gpu::decode::PlanCache,
    /// Streamed with on-demand experts: GPU or CPU decoding, measured (see `DeviceChoice`).
    stream_choice: crate::gpu::decode::DeviceChoice,
    /// Qwen4-Exp (hyper-connections, n-gram embedding, QSA; see `crate::qwen4exp`).
    pub(crate) exp: Option<Box<crate::qwen4exp::Exp<'a>>>,
}

/// Most tokens of a multi-token GPU pass (speculative verification).
pub const MULTI_TOKENS: usize = 16;

/// Recurrent state of one Gated DeltaNet layer.
#[derive(Clone)]
pub struct GdnState {
    // Page-aligned and never reallocated: GPU decoding updates them in place (`gpu::decode`).
    pub(crate) s: crate::quant::PageVec<f32>,    // [n_v, hd_k, hd_v]
    pub(crate) conv: crate::quant::PageVec<f32>, // [K-1, conv_dim]: the last K-1 raw conv inputs (oldest first)
}

/// What a recording forward (`QwenModel::forward_hidden_rec`) keeps of one Gated DeltaNet layer
/// so the session can rewind to any position inside the span: unlike a KV cache, the recurrent
/// state cannot be truncated, so the state before the span and the delta rule's per-token inputs
/// are kept, and a rewind replays the rule over the kept tokens (the same arithmetic as the
/// forward, so bit-identical). Keeping every intermediate state instead would cost 151 MB per
/// token on Qwen3.8-27B; this is one state plus ~45 KB per token. Buffers are reused.
#[derive(Clone, Default)]
pub(crate) struct GdnRecord {
    s0: Vec<f32>,    // state before the span
    c: Vec<f32>,     // [t, conv_dim]: conv outputs (after SiLU)
    b: Vec<f32>,     // [t, n_v]
    a: Vec<f32>,     // [t, n_v]
    qkv: Vec<f32>,   // [t, conv_dim]: raw conv inputs
    conv0: Vec<f32>, // conv state before the span
    conv_dim: usize,
    rule: DeltaRule,
}

/// The shapes and per-head constants of one layer's gated delta rule.
#[derive(Clone, Default)]
struct DeltaRule {
    nk: usize,
    nv: usize,
    hdk: usize,
    hdv: usize,
    tiled: bool,
    decay_coef: Vec<f32>,
    dt_bias: Vec<f32>,
}

impl DeltaRule {
    /// Runs the gated delta rule over `t` tokens of conv outputs `c` (`[t, 2 * key_dim +
    /// value_dim]`) and gate inputs `b`, `a` (`[t, n_v]`), updating the state `s` in place and,
    /// if given, writing each token's output to `out` (`[t, value_dim]`). Parallel over value
    /// heads.
    fn run(&self, s: &mut [f32], c: &[f32], b: &[f32], a: &[f32], t: usize, out: Option<&mut [f32]>) {
        let (nk, nv, hdk, hdv) = (self.nk, self.nv, self.hdk, self.hdv);
        let (key_dim, value_dim) = (nk * hdk, nv * hdv);
        let conv_dim = 2 * key_dim + value_dim;
        let rep = nv / nk;
        let op = out.map(|o| SendPtrF(o.as_mut_ptr()));
        let sp = SendPtrF(s.as_mut_ptr());
        let qscale = (hdk as f32).powf(-0.5);
        pool::run(nv, &|hv| {
            let hk = if self.tiled { hv % nk } else { hv / rep };
            // SAFETY: each head owns its own state slab and output columns.
            let s = unsafe { std::slice::from_raw_parts_mut(sp.get().add(hv * hdk * hdv), hdk * hdv) };
            let mut qn = vec![0f32; hdk];
            let mut kn = vec![0f32; hdk];
            let mut kvmem = vec![0f32; hdv];
            let mut delta = vec![0f32; hdv];
            for i in 0..t {
                let row = &c[i * conv_dim..(i + 1) * conv_dim];
                let qv = &row[hk * hdk..(hk + 1) * hdk];
                let kv = &row[key_dim + hk * hdk..key_dim + (hk + 1) * hdk];
                let vv = &row[2 * key_dim + hv * hdv..2 * key_dim + (hv + 1) * hdv];
                let qn_inv = (qv.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
                let kn_inv = (kv.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
                for d in 0..hdk {
                    qn[d] = qv[d] * qn_inv * qscale;
                    kn[d] = kv[d] * kn_inv;
                }
                let beta = 1.0 / (1.0 + fast_exp(-b[i * nv + hv]));
                let ga = a[i * nv + hv] + self.dt_bias[hv];
                let softplus = if ga > 20.0 { ga } else { (1.0 + ga.exp()).ln() };
                let decay = (self.decay_coef[hv] * softplus).exp();
                // S *= decay; kvmem = k^T S
                kvmem.fill(0.0);
                for dk in 0..hdk {
                    let srow = &mut s[dk * hdv..(dk + 1) * hdv];
                    let kd = kn[dk];
                    for j in 0..hdv {
                        srow[j] *= decay;
                        kvmem[j] += srow[j] * kd;
                    }
                }
                for j in 0..hdv {
                    delta[j] = (vv[j] - kvmem[j]) * beta;
                }
                // S += k (x) delta; out = q^T S
                match op {
                    Some(op) => {
                        // SAFETY: each (token, head) output slice is written by this head only.
                        let o = unsafe { std::slice::from_raw_parts_mut(op.get().add(i * value_dim + hv * hdv), hdv) };
                        o.fill(0.0);
                        for dk in 0..hdk {
                            let srow = &mut s[dk * hdv..(dk + 1) * hdv];
                            let (kd, qd) = (kn[dk], qn[dk]);
                            for j in 0..hdv {
                                srow[j] += kd * delta[j];
                                o[j] += srow[j] * qd;
                            }
                        }
                    }
                    None => {
                        for dk in 0..hdk {
                            let srow = &mut s[dk * hdv..(dk + 1) * hdv];
                            let kd = kn[dk];
                            for j in 0..hdv {
                                srow[j] += kd * delta[j];
                            }
                        }
                    }
                }
            }
        });
    }
}

/// A run of vision tokens (one image, or one temporal patch of a video) in the sequence: `gh` x
/// `gw` tokens in row-major order from position `start`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionSpan {
    pub start: usize,
    pub gh: usize,
    pub gw: usize,
}

/// M-RoPE positions `[time, height, width]` of positions `start..start + t` given the vision
/// spans of the sequence (sorted by start), as `get_rope_index` assigns them: a span's tokens sit
/// at `(s, s + row, s + col)` from the position `s` its first token would have had, and the text
/// after it continues from `s + max(gh, gw)`. Text before any span gets its plain index.
pub fn mrope_positions(spans: &[VisionSpan], start: usize, t: usize) -> Vec<[usize; 3]> {
    let mut out = Vec::with_capacity(t);
    let mut si = 0;
    // Rope position minus sequence index for text after the spans passed so far.
    let mut delta: isize = 0;
    for p in start..start + t {
        while si < spans.len() && p >= spans[si].start + spans[si].gh * spans[si].gw {
            let sp = spans[si];
            delta += sp.gh.max(sp.gw) as isize - (sp.gh * sp.gw) as isize;
            si += 1;
        }
        match spans.get(si) {
            Some(sp) if p >= sp.start => {
                let base = (sp.start as isize + delta) as usize;
                let k = p - sp.start;
                out.push([base, base + k / sp.gw, base + k % sp.gw]);
            }
            _ => {
                let q = (p as isize + delta) as usize;
                out.push([q, q, q]);
            }
        }
    }
    out
}

/// A multi-token GPU pass of a session (see [`QwenModel::gpu_verify`]): its plan, its span, and
/// the recurrent states from before it.
#[derive(Clone)]
pub(crate) struct GpuRec {
    pub(crate) plan: std::sync::Arc<dyn crate::gpu::decode::DecodePlan>,
    pub(crate) start: usize,
    pub(crate) len: usize,
    /// Gated DeltaNet delta-rule and conv states before the pass, in the plan's state order.
    pub(crate) states: Vec<Vec<f32>>,
}

#[derive(Clone)]
pub struct QwenSession {
    pub(crate) kv: Vec<Option<LayerCache>>,
    pub(crate) gdn: Vec<Option<GdnState>>,
    pub(crate) rec: Vec<GdnRecord>,
    /// `(start, len)` of the positions recorded by the last forward, if it recorded.
    pub(crate) rec_span: Option<(usize, usize)>,
    /// The last forward when it was a multi-token GPU pass (speculative verification): what a
    /// rewind inside it needs.
    pub(crate) gpu_rec: Option<GpuRec>,
    /// Vision token runs of the conversation (M-RoPE), sorted by start.
    pub(crate) spans: Vec<VisionSpan>,
    pub pos: usize,
    /// Device state of GPU decoding (KV cache copies).
    pub(crate) gpu: crate::gpu::decode::Slot,
    /// The recurrent states are as the last GPU step left them (see
    /// [`crate::gpu::decode::Step::states_kept`]); every CPU write clears it.
    pub(crate) gpu_states_kept: bool,
    /// Runtime interventions ([`crate::intervene`]).
    pub iv: crate::intervene::State,
    /// Layers whose output (the residual stream after the layer, before the final norm, as
    /// transformers' `hidden_states[l + 1]`) the next forward passes record into `captured`.
    pub capture: Vec<usize>,
    /// `(layer, [t, hidden])` recorded by the forward passes for `capture`, in pass order.
    pub captured: Vec<(usize, Vec<f32>)>,
    /// Explicit `[time, height, width]` RoPE positions by sequence index, overriding the ones
    /// derived from the vision spans (for pipelines that pass their own `position_ids`; prompt
    /// passes of several tokens).
    pub positions: Option<Vec<[usize; 3]>>,
    /// Qwen4-Exp state (n-gram contexts and conv, indexer keys; see `crate::qwen4exp`).
    pub(crate) exp: Option<Box<crate::qwen4exp::ExpSession>>,
}

impl QwenSession {
    /// Declares `gh` x `gw` vision tokens (row-major, one image or one temporal patch of a video)
    /// at sequence position `start` (at or after `pos`), before they are fed: they take M-RoPE
    /// positions, and the text after them continues from `max(gh, gw)` positions later.
    pub fn add_vision_span(&mut self, start: usize, gh: usize, gw: usize) {
        debug_assert!(start >= self.pos && gh > 0 && gw > 0);
        let sp = VisionSpan { start, gh, gw };
        let i = self.spans.partition_point(|s| s.start < start);
        self.spans.insert(i, sp);
    }

    /// The declared vision spans.
    pub fn vision_spans(&self) -> &[VisionSpan] {
        &self.spans
    }

    /// Rewinds to `pos` tokens (dropping later KV entries and restoring the recurrent states).
    /// Positions up to the end of the last forward are always fine; earlier ones need that
    /// forward to have been a recording one (`forward_hidden_rec`) that started before `pos`.
    pub fn rewind(&mut self, pos: usize) -> Result<()> {
        if pos >= self.pos {
            return Ok(());
        }
        self.gpu_states_kept = false;
        if let Some(e) = &mut self.exp {
            // After a GPU pass the n-gram conv state is replayed with the recurrent states below.
            if self.gpu_rec.is_some() {
                e.rewind_keys(pos);
            } else {
                e.rewind(pos, self.rec_span)?;
            }
        }
        let has_gdn = self.gdn.iter().any(|g| g.is_some());
        if has_gdn && let Some(r) = self.gpu_rec.take() {
            // A GPU verification pass: the states from before it, then its first tokens again on
            // the GPU (the same kernels as one-token steps).
            if pos <= r.start || pos > r.start + r.len {
                bail!("cannot rewind to {pos}: the GPU pass covered {}..{}", r.start, r.start + r.len);
            }
            let QwenSession { gdn, exp, gpu, .. } = self;
            let mut states = crate::qwen4exp::recurrent_states_of(gdn, exp.as_deref_mut());
            for (st, s0) in states.iter_mut().zip(&r.states) {
                st.copy_from_slice(s0);
            }
            if !r.plan.replay_recurrent(gpu, &mut states, pos - r.start) {
                bail!("cannot rewind the recurrent state to {pos}: the GPU replay failed");
            }
        } else if has_gdn {
            let Some((start, len)) = self.rec_span else {
                bail!("cannot rewind the recurrent state to {pos}: the last forward did not record");
            };
            if pos <= start || pos > start + len {
                bail!("cannot rewind to {pos}: recorded span is {start}..{}", start + len);
            }
            let i = pos - start - 1; // the last kept token, within the span
            for (st, rec) in self.gdn.iter_mut().zip(&self.rec) {
                let Some(st) = st else { continue };
                st.s.copy_from_slice(&rec.s0);
                let nv = rec.rule.nv;
                rec.rule
                    .run(&mut st.s, &rec.c, &rec.b[..(i + 1) * nv], &rec.a[..(i + 1) * nv], i + 1, None);
                // The conv state is the last K-1 raw inputs up to and including token i.
                let cd = rec.conv_dim;
                let keep = st.conv.len() / cd;
                for slot in 0..keep {
                    let ti = (i + 1) as isize - keep as isize + slot as isize;
                    let src = if ti >= 0 {
                        &rec.qkv[ti as usize * cd..(ti as usize + 1) * cd]
                    } else {
                        let o = (keep as isize + ti) as usize;
                        &rec.conv0[o * cd..(o + 1) * cd]
                    };
                    st.conv[slot * cd..(slot + 1) * cd].copy_from_slice(src);
                }
            }
            self.rec_span = Some((start, i + 1));
        }
        for c in self.kv.iter_mut().flatten() {
            c.truncate(pos);
        }
        self.spans.retain(|s| s.start < pos);
        self.pos = pos;
        Ok(())
    }
}

fn norm_weight(w: Vec<f32>, zero_centered: bool) -> Vec<f32> {
    let o = if zero_centered { 1.0 } else { 0.0 };
    w.into_iter().map(|v| o + v).collect()
}

/// Tensor loading shared by the main model and the MTP head (whose weights may come from a
/// separate sidecar `ModelFiles`): shape checks, precision, zero-centered norms.
pub(crate) struct Loader<'a, 'c> {
    pub files: &'a ModelFiles,
    pub cfg: &'c QwenConfig,
    pub quant: Quant,
}

impl<'a> Loader<'a, '_> {
    pub fn has(&self, n: &str) -> bool {
        let f = self.files;
        f.has(n) || f.has(&format!("{n}.q8")) || f.has(&format!("{n}.q4"))
    }

    /// Small parameter vectors are stored as BF16 in some checkpoints and F32 in others.
    pub fn getf(&self, full: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let info = self.files.info(full).with_context(|| format!("missing tensor {full}"))?;
        if info.shape != shape {
            bail!("tensor {full} has shape {:?}, expected {:?}", info.shape, shape);
        }
        self.files.f32(full)
    }

    pub fn lin(&self, full: &str, n_out: usize, n_in: usize) -> Result<Lin<'a>> {
        load_lin(self.files, full, n_out, n_in, self.quant)
    }

    /// An RMSNorm weight of `n` as applied (`1 + w` when zero-centered), read in f32 (GGUF stores
    /// norms as f32 with the 1 folded in).
    pub fn norm(&self, full: &str, n: usize) -> Result<Vec<f32>> {
        Ok(norm_weight(self.getf(full, &[n])?, self.cfg.zero_centered_norm))
    }

    fn mlp(&self, p: &str, inter: usize) -> Result<Mlp<'a>> {
        let h = self.cfg.hidden_size;
        Ok(Mlp {
            gate: self.lin(&format!("{p}gate_proj.weight"), inter, h)?,
            up: self.lin(&format!("{p}up_proj.weight"), inter, h)?,
            down: self.lin(&format!("{p}down_proj.weight"), h, inter)?,
            inter,
        })
    }

    /// Gated full attention under `ap` (`...self_attn.`).
    pub fn attn(&self, ap: &str) -> Result<Attn<'a>> {
        let cfg = self.cfg;
        let h = cfg.hidden_size;
        let (nh, nkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let qw = 2 * nh * hd; // q and gate
        Ok(Attn {
            q: self.lin(&format!("{ap}q_proj.weight"), if cfg.attn_output_gate { qw } else { nh * hd }, h)?,
            k: self.lin(&format!("{ap}k_proj.weight"), nkv * hd, h)?,
            v: self.lin(&format!("{ap}v_proj.weight"), nkv * hd, h)?,
            o: self.lin(&format!("{ap}o_proj.weight"), h, nh * hd)?,
            q_norm: cfg.qk_norm.then(|| self.norm(&format!("{ap}q_norm.weight"), hd)).transpose()?,
            k_norm: cfg.qk_norm.then(|| self.norm(&format!("{ap}k_norm.weight"), hd)).transpose()?,
            bias: if cfg.attention_bias {
                Some([
                    self.getf(&format!("{ap}q_proj.bias"), &[nh * hd])?,
                    self.getf(&format!("{ap}k_proj.bias"), &[nkv * hd])?,
                    self.getf(&format!("{ap}v_proj.bias"), &[nkv * hd])?,
                ])
            } else {
                None
            },
        })
    }

    /// The MLP (dense or MoE, per the config) under `mp` (`...mlp.`).
    pub fn ffn(&self, mp: &str) -> Result<Ffn<'a>> {
        let cfg = self.cfg;
        let (h, quant, files) = (cfg.hidden_size, self.quant, self.files);
        let Some(n_exp) = cfg.num_experts else {
            return Ok(Ffn::Dense(self.mlp(mp, cfg.intermediate_size)?));
        };
        let mi = cfg.moe_intermediate_size;
        if cfg.num_experts_per_tok == 0 || cfg.num_experts_per_tok > n_exp {
            bail!("num_experts_per_tok {} with {n_exp} experts", cfg.num_experts_per_tok);
        }
        let experts = Experts::load(
            files,
            &format!("{mp}experts."),
            n_exp,
            cfg.num_experts_per_tok,
            mi,
            h,
            quant,
            Act::Silu,
        )?;
        Ok(Ffn::Moe(Moe {
            router: self.getf(&format!("{mp}gate.weight"), &[n_exp, h])?,
            experts,
            shared: self.mlp(&format!("{mp}shared_expert."), cfg.shared_expert_intermediate_size)?,
            shared_gate: self.getf(&format!("{mp}shared_expert_gate.weight"), &[1, h])?,
            router_mm: cfg.is_exp().then(std::sync::OnceLock::new),
        }))
    }
}

impl<'a> QwenModel<'a> {
    pub fn load(cfg: QwenConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        let ld = Loader { files, cfg: &cfg, quant };
        let prefix = if ld.has("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else if ld.has("model.embed_tokens.weight") {
            "model."
        } else if ld.has("embed_tokens.weight") {
            "" // a bare text model (embedding checkpoints)
        } else {
            bail!("cannot find embed_tokens in checkpoint");
        };
        let h = cfg.hidden_size;

        let tied = cfg.tie_word_embeddings || !ld.has("lm_head.weight");
        // Lookups use the head's precision when tied (one copy in a pre-quantized file), else bf16.
        let embed = Table::load(
            files,
            &format!("{prefix}embed_tokens.weight"),
            cfg.vocab_size,
            h,
            if tied { quant } else { table_quant(quant) },
        )?;
        let lm_head = if tied {
            ld.lin(&format!("{prefix}embed_tokens.weight"), cfg.vocab_size, h)?
        } else {
            ld.lin("lm_head.weight", cfg.vocab_size, h)?
        };
        let exp = cfg.is_exp();
        let final_norm = if exp {
            Vec::new()
        } else {
            ld.norm(&format!("{prefix}norm.weight"), h)?
        };

        let (nk, nv, hdk, hdv) = (
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
        );
        let (key_dim, value_dim) = (nk * hdk, nv * hdv);
        let conv_dim = 2 * key_dim + value_dim;
        let kc = cfg.linear_conv_kernel_dim;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let lp = format!("{prefix}layers.{l}.");
            let mixer = if cfg.is_full(l) {
                Mixer::Attn(ld.attn(&format!("{lp}self_attn."))?)
            } else {
                let gp = format!("{lp}linear_attn.");
                let conv_w = ld.getf(&format!("{gp}conv1d.weight"), &[conv_dim, 1, kc])?;
                // GGUF stores -exp(A_log) directly.
                let decay_coef = match ld.getf(&format!("{gp}A_log"), &[nv]) {
                    Ok(a_log) => a_log.iter().map(|&v| -v.exp()).collect(),
                    Err(_) => ld.getf(&format!("{gp}ssm_a"), &[nv])?,
                };
                Mixer::Gdn(Gdn {
                    qkv: ld.lin(&format!("{gp}in_proj_qkv.weight"), conv_dim, h)?,
                    z: ld.lin(&format!("{gp}in_proj_z.weight"), value_dim, h)?,
                    b: ld.lin(&format!("{gp}in_proj_b.weight"), nv, h)?,
                    a: ld.lin(&format!("{gp}in_proj_a.weight"), nv, h)?,
                    conv_w,
                    decay_coef,
                    dt_bias: ld.getf(&format!("{gp}dt_bias"), &[nv])?,
                    norm_w: ld.getf(&format!("{gp}norm.weight"), &[hdv])?,
                    out: ld.lin(&format!("{gp}out_proj.weight"), h, value_dim)?,
                })
            };
            layers.push(QLayer {
                mixer,
                ln_in: if exp {
                    Vec::new()
                } else {
                    ld.norm(&format!("{lp}input_layernorm.weight"), h)?
                },
                ln_post: if exp {
                    Vec::new()
                } else {
                    ld.norm(&format!("{lp}post_attention_layernorm.weight"), h)?
                },
                ffn: ld.ffn(&format!("{lp}mlp."))?,
            });
        }
        let rot = cfg.rotary_dim();
        let theta = cfg.rope_theta();
        let mut inv_freq: Vec<f32> = (0..rot / 2)
            .map(|i| (1.0 / theta.powf((2 * i) as f64 / rot as f64)) as f32)
            .collect();
        if let Some(l3) = cfg.llama3_rope() {
            l3.scale(&mut inv_freq);
        }
        // Llama GGUF: the llama3 scaling as one divisor per frequency.
        if files.has("rope_freqs") {
            for (f, d) in inv_freq.iter_mut().zip(ld.getf("rope_freqs", &[rot / 2])?) {
                *f /= d;
            }
        }
        if cfg.rope_interleaved && rot != cfg.head_dim {
            bail!("interleaved RoPE needs full rotary dimensions");
        }
        // Interleaved M-RoPE (transformers' Qwen3.5 rotary): frequency i rotates with the height
        // for i % 3 == 1 and i < 3 * sections[1], with the width for i % 3 == 2 and
        // i < 3 * sections[2], with the time otherwise.
        // Contiguous sections (Qwen2-VL / 2.5-VL): the first `sections[0]` frequencies rotate with
        // the time, the next `sections[1]` with the height, the rest with the width.
        let mrope_axis = cfg.mrope_section().map(|sec| {
            (0..inv_freq.len())
                .map(|i| match i % 3 {
                    _ if !cfg.mrope_interleaved => (i >= sec[0]) as u8 + (i >= sec[0] + sec[1]) as u8,
                    1 if i < 3 * sec[1] => 1,
                    2 if i < 3 * sec[2] => 2,
                    _ => 0,
                })
                .collect()
        });
        let exp = if exp {
            Some(Box::new(crate::qwen4exp::Exp::load(&ld, prefix)?))
        } else {
            None
        };
        // Routed experts beyond a third of memory: prefill wraps the experts of each pass rather
        // than all of them, which the GPU decode plan wraps again (wired twice, the device runs
        // out of memory: Qwen3.8-Flash-Next's 78 GB at Q4).
        let expert_bytes: usize = layers
            .iter()
            .map(|l| l.ffn.expert_lins().iter().flatten().map(|w| w.bytes()).sum::<usize>())
            .sum();
        if crate::platform::total_memory().is_some_and(|m| expert_bytes as f64 > m as f64 / 3.0) {
            for l in &layers {
                if let Ffn::Moe(m) = &l.ffn {
                    m.experts.wrap_per_pass.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        Ok(Self {
            cfg,
            quant,
            embed,
            layers,
            final_norm,
            lm_head,
            inv_freq,
            mrope_axis,
            stream: None,
            gpu_plans: Default::default(),
            stream_choice: Default::default(),
            exp,
        })
    }

    /// Attaches the modules of a runtime LoRA adapter (see `crate::lora`) to the layers they
    /// target, by Hugging Face module name (whatever the checkpoint format; GGUF's tiled Gated
    /// DeltaNet value heads are reordered to match). Errors when a module matches no layer or
    /// has another shape. Returns the number of layers patched.
    pub fn attach_lora(&mut self, ad: &mut crate::lora::LoraAdapter) -> Result<usize> {
        use crate::lora::tiled_v_rows;
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let (nk, nv, hdk, hdv) = (
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
        );
        let (key_dim, value_dim) = (nk * hdk, nv * hdv);
        let (nh, nkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let qw = if cfg.attn_output_gate { 2 * nh * hd } else { nh * hd };
        let tiled = cfg.v_heads_tiled;
        let qkv_rows = tiled.then(|| tiled_v_rows(2 * key_dim, nk, nv, hdv));
        let v_rows = tiled.then(|| tiled_v_rows(0, nk, nv, hdv));
        let head_rows = tiled.then(|| tiled_v_rows(0, nk, nv, 1));
        // Llama GGUF: query / key rows in llama.cpp's RoPE order.
        let paired = cfg.rope_interleaved;
        let q_rows = paired.then(|| crate::lora::rope_pair_rows(nh, hd));
        let k_rows = paired.then(|| crate::lora::rope_pair_rows(nkv, hd));
        let mut set =
            |lin: &mut Lin<'a>, name: String, n_out: usize, n_in: usize, rows: Option<&[usize]>, cols: Option<&[usize]>| -> Result<()> {
                if let Some(d) = ad.take(&name, n_out, n_in, rows, cols)? {
                    lin.set_lora(Some(d));
                }
                Ok(())
            };
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let lp = format!("layers.{l}.");
            match &mut layer.mixer {
                Mixer::Gdn(g) => {
                    let p = format!("{lp}linear_attn.");
                    set(
                        &mut g.qkv,
                        format!("{p}in_proj_qkv"),
                        2 * key_dim + value_dim,
                        h,
                        qkv_rows.as_deref(),
                        None,
                    )?;
                    set(&mut g.z, format!("{p}in_proj_z"), value_dim, h, v_rows.as_deref(), None)?;
                    set(&mut g.b, format!("{p}in_proj_b"), nv, h, head_rows.as_deref(), None)?;
                    set(&mut g.a, format!("{p}in_proj_a"), nv, h, head_rows.as_deref(), None)?;
                    set(&mut g.out, format!("{p}out_proj"), h, value_dim, None, v_rows.as_deref())?;
                }
                Mixer::Attn(a) => {
                    let p = format!("{lp}self_attn.");
                    set(&mut a.q, format!("{p}q_proj"), qw, h, q_rows.as_deref(), None)?;
                    set(&mut a.k, format!("{p}k_proj"), nkv * hd, h, k_rows.as_deref(), None)?;
                    set(&mut a.v, format!("{p}v_proj"), nkv * hd, h, None, None)?;
                    set(&mut a.o, format!("{p}o_proj"), h, nh * hd, None, None)?;
                }
            }
            // Routed experts are not patched (no adapter of the family has them): their modules
            // stay in the adapter and `finish` reports them.
            let (m, p) = match &mut layer.ffn {
                Ffn::Dense(m) => (m, format!("{lp}mlp.")),
                Ffn::Moe(moe) => (&mut moe.shared, format!("{lp}mlp.shared_expert.")),
            };
            let inter = m.inter;
            set(&mut m.gate, format!("{p}gate_proj"), inter, h, None, None)?;
            set(&mut m.up, format!("{p}up_proj"), inter, h, None, None)?;
            set(&mut m.down, format!("{p}down_proj"), h, inter, None, None)?;
        }
        set(&mut self.lm_head, "lm_head".into(), cfg.vocab_size, h, None, None)?;
        ad.finish()
    }

    /// The weights of each streaming unit, in execution order: the layers, then the LM head.
    fn unit_lins(&self) -> Vec<(String, Vec<&Lin<'a>>)> {
        let mut v: Vec<(String, Vec<&Lin>)> = self
            .layers
            .iter()
            .enumerate()
            .map(|(l, layer)| {
                let mut lins: Vec<&Lin> = match &layer.mixer {
                    Mixer::Gdn(g) => vec![&g.qkv, &g.z, &g.b, &g.a, &g.out],
                    Mixer::Attn(a) => a.lins().to_vec(),
                };
                lins.extend(layer.ffn.lins());
                if let Some(e) = &self.exp {
                    lins.extend(e.layer_lins(l));
                }
                (format!("layer.{l}"), lins)
            })
            .collect();
        let mut head = vec![&self.lm_head];
        if let Some(e) = &self.exp {
            head.extend(e.mixer.lins());
        }
        v.push(("lm_head".into(), head));
        v
    }

    /// The streaming units of this model in execution order (see `crate::stream`): the layers,
    /// then the LM head. The routed experts of MoE layers are on-demand parts of their layer.
    pub fn stream_units(&self) -> Vec<crate::stream::UnitSpec> {
        let mut v: Vec<crate::stream::UnitSpec> = self
            .layers
            .iter()
            .enumerate()
            .map(|(l, layer)| {
                let mut lins: Vec<&Lin> = match &layer.mixer {
                    Mixer::Gdn(g) => vec![&g.qkv, &g.z, &g.b, &g.a, &g.out],
                    Mixer::Attn(a) => a.lins().to_vec(),
                };
                lins.extend(layer.ffn.dense_lins());
                if let Some(e) = &self.exp {
                    lins.extend(e.layer_lins(l));
                }
                crate::stream::UnitSpec {
                    name: format!("layer.{l}"),
                    ranges: lins.iter().flat_map(|l| l.mem_ranges()).collect(),
                    uses: 1.0,
                    parts: layer.ffn.expert_ranges(),
                }
            })
            .collect();
        let mut head = self.lm_head.mem_ranges();
        if let Some(e) = &self.exp {
            head.extend(e.mixer.lins().iter().flat_map(|l| l.mem_ranges()));
        }
        v.push(crate::stream::UnitSpec {
            name: "lm_head".into(),
            ranges: head,
            uses: 1.0,
            parts: Vec::new(),
        });
        v
    }

    /// Marks the weights of the units with `streamed[i]` set as streamed: no accelerator state
    /// is kept for them between calls (see `Lin::set_transient_device`).
    pub fn mark_streamed(&self, streamed: &[bool]) {
        for ((_, lins), &s) in self.unit_lins().iter().zip(streamed) {
            for l in lins {
                l.set_transient_device(s);
            }
        }
        // Routed experts are read on demand and evicted whatever their layer: no device state
        // (which would pin their pages) outlives a call.
        for layer in &self.layers {
            for l in layer.ffn.expert_lins().into_iter().flatten() {
                l.set_transient_device(true);
            }
        }
    }

    /// Brackets every layer and the LM head with `hook`'s units (None stops streaming).
    pub fn set_stream(&mut self, hook: Option<crate::stream::StreamHook>) {
        self.stream = hook;
    }

    pub fn streamed_bytes(&self) -> usize {
        let mut n = self.lm_head.bytes();
        for l in &self.layers {
            n += l.ffn.streamed_bytes();
            n += match &l.mixer {
                Mixer::Gdn(g) => g.qkv.bytes() + g.z.bytes() + g.b.bytes() + g.a.bytes() + g.out.bytes(),
                Mixer::Attn(a) => a.bytes(),
            };
        }
        if let Some(e) = &self.exp {
            n += (0..self.layers.len())
                .flat_map(|l| e.layer_lins(l))
                .chain(e.mixer.lins())
                .map(|l| l.bytes())
                .sum::<usize>();
        }
        n
    }

    pub fn new_session(&self) -> QwenSession {
        let cfg = &self.cfg;
        let conv_dim = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let mut kv = Vec::new();
        let mut gdn = Vec::new();
        for l in 0..cfg.num_hidden_layers {
            if cfg.is_full(l) {
                kv.push(Some(LayerCache::new(cfg.num_key_value_heads, cfg.head_dim)));
                gdn.push(None);
            } else {
                kv.push(None);
                gdn.push(Some(GdnState {
                    s: crate::quant::PageVec::zeroed(cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim),
                    conv: crate::quant::PageVec::zeroed((cfg.linear_conv_kernel_dim - 1) * conv_dim),
                }));
            }
        }
        let rec = (0..cfg.num_hidden_layers).map(|_| GdnRecord::default()).collect();
        QwenSession {
            kv,
            gdn,
            rec,
            rec_span: None,
            gpu_rec: None,
            spans: Vec::new(),
            pos: 0,
            gpu: Default::default(),
            gpu_states_kept: false,
            iv: Default::default(),
            capture: Vec::new(),
            captured: Vec::new(),
            positions: None,
            exp: self.exp.as_ref().map(|e| Box::new(e.new_session(cfg))),
        }
    }

    /// One token's forward pass as a device graph (`crate::gpu::decode`): dense layers with full
    /// attention (Llama 3, Qwen3, and Qwen3.5's attention layers); None when some layer or
    /// weight is not supported (Gated DeltaNet, MoE, LoRA, streamed weights).
    fn gpu_graph<'s>(&'s self, lay: Option<&crate::intervene::GpuLayout>) -> Option<crate::gpu::decode::Graph<'s>> {
        self.gpu_graph_with(lay, None)
    }

    /// [`Self::gpu_graph`], and for multi-token plans (`multi`: layers whose outputs to capture)
    /// per-layer Gated DeltaNet buffers (kept for replays) and the capture copies.
    fn gpu_graph_with<'s>(
        &'s self,
        lay: Option<&crate::intervene::GpuLayout>,
        multi: Option<&[usize]>,
    ) -> Option<crate::gpu::decode::Graph<'s>> {
        use crate::gpu::decode::{Graph, LinMode, Op};
        use crate::intervene::GpuNeurons;
        if let Some(ex) = &self.exp {
            // Multi-token plans of Qwen4-Exp: speculative verification only (no capture layers).
            return if lay.is_none() && multi.is_none_or(|m| m.is_empty()) {
                self.exp_gpu_graph(ex, multi.is_some(), false)
            } else {
                None
            };
        }
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps as f32;
        let (nh, nkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let gated = cfg.attn_output_gate;
        let qw = if gated { 2 * nh * hd } else { nh * hd };
        // Streamed weights (layers outside the resident set) are wrapped by the plan per step.
        let dev = |l: &'s Lin, n_out: usize, n_in: usize| -> Option<&crate::gpu::DeviceWeight> {
            if l.has_lora() {
                return None;
            }
            if l.is_transient() {
                return self.stream.is_some().then(crate::gpu::decode::streamed_weight);
            }
            l.device_weight(n_out, n_in)
        };
        let mut g = Graph::new();
        if let Some(s) = &self.stream {
            g.gate = Some(std::sync::Arc::new(s.clone()));
        }
        let x = g.buf(h);
        let hn = g.buf(h);
        let qkv = g.buf(qw + 2 * nkv * hd);
        let qo = g.buf(nh * hd);
        let att = g.buf(nh * hd);
        let inter = self
            .layers
            .iter()
            .map(|l| match &l.ffn {
                Ffn::Dense(m) => m.inter,
                Ffn::Moe(m) => m.shared.inter,
            })
            .max()
            .unwrap_or(0);
        let act = g.buf(inter.max(1));
        let act2 = g.buf(inter.max(1));
        // MoE: router logits (and the shared expert's gate), the chosen experts, their activations.
        let (mut n_exp, mut top_k, mut e_inter) = (0, 0, 0);
        for l in &self.layers {
            if let Ffn::Moe(m) = &l.ffn {
                n_exp = n_exp.max(m.experts.down.len());
                top_k = top_k.max(m.experts.top_k);
                e_inter = e_inter.max(m.experts.inter);
            }
        }
        let rlogits = g.buf(n_exp + 1);
        let sel = g.buf((2 * top_k).max(1));
        let eact = g.buf((top_k * e_inter).max(1));
        let hidden = g.buf(h);
        let logits = g.buf(cfg.vocab_size);
        g.input = x;
        g.hidden = hidden;
        g.n_hidden = h;
        g.rope_halves = vec![self.inv_freq.len()];
        // Interventions: parameter buffers, then a hook op per residual point that has one.
        let (hot, cold) = (
            g.buf(lay.map_or(1, |l| l.hot_len.max(1))),
            g.buf(lay.map_or(1, |l| l.cold_len.max(1))),
        );
        if let Some(l) = lay {
            g.params = Some(((hot, l.hot_len), (cold, l.cold_len)));
        }
        let steer = |g: &mut Graph<'s>, p: usize| {
            if let Some(gp) = lay.and_then(|l| l.points.get(p).copied().flatten()) {
                g.ops.push(Op::Steer {
                    x,
                    n: h,
                    hot: hot.at(gp.hot),
                    cold: cold.at(gp.cold),
                    n_add: gp.n_add,
                    n_items: gp.n_items,
                });
            }
        };
        let edits = |l: usize| lay.map_or((None, None), |y| y.neurons[l]);
        steer(&mut g, 0);
        let lin = |l: &'s Lin, n_out: usize, n_in: usize, x, y, mode| -> Option<Op<'s>> {
            Some(Op::Linear {
                w: l.as_ref(),
                dev: dev(l, n_out, n_in)?,
                w2: None,
                n_out,
                n_in,
                x,
                y,
                mode,
            })
        };
        // down(silu(gate(hn)) * up(hn)) added to x: one paired matvec when gate and up share a
        // format, else two and an elementwise op.
        let mlp = |g: &mut Graph<'s>, m: &'s Mlp<'s>, down_mode: LinMode, edit: Option<GpuNeurons>| -> Option<()> {
            let (gw, uw) = (m.gate.as_ref(), m.up.as_ref());
            if crate::gpu::decode::pairable(&gw, &uw) {
                g.ops.push(Op::Linear {
                    w: gw,
                    dev: dev(&m.gate, m.inter, h)?,
                    w2: Some((uw, dev(&m.up, m.inter, h)?)),
                    n_out: m.inter,
                    n_in: h,
                    x: hn,
                    y: act,
                    mode: LinMode::SiluPair,
                });
            } else {
                g.ops.push(lin(&m.gate, m.inter, h, hn, act2, LinMode::Store)?);
                g.ops.push(lin(&m.up, m.inter, h, hn, act, LinMode::Store)?);
                g.ops.push(Op::Glu {
                    g: act2,
                    u: act,
                    out: act,
                    n: m.inter,
                    gelu: false,
                });
            }
            if let Some(e) = edit {
                g.ops.push(Op::NeuronEdit {
                    act,
                    hot: hot.at(e.hot),
                    n: e.n,
                });
            }
            g.ops.push(lin(&m.down, h, m.inter, act, x, down_mode)?);
            Some(())
        };
        // The feed-forward block of a layer on hn, added to x: a dense MLP, or the MoE's routed
        // experts plus its sigmoid-gated shared expert.
        let ffn = |g: &mut Graph<'s>, f: &'s Ffn<'s>, l: usize| -> Option<()> {
            let (de, ee) = edits(l);
            let m = match f {
                Ffn::Dense(m) => return mlp(g, m, LinMode::Add, de),
                Ffn::Moe(m) => m,
            };
            // Streamed experts (always on-demand parts of their layer) are wrapped per step by
            // the plan once routing chose them.
            let streams = self.stream.is_some();
            let off = |l: &Lin| l.has_lora() || (l.is_transient() && !streams);
            if m.experts.act != crate::moe::Act::Silu || m.experts.down.iter().any(off) {
                return None;
            }
            let ne = m.experts.down.len();
            g.ops.push(Op::Route {
                x: hn,
                router: &m.router,
                gate: Some(&m.shared_gate),
                n_exp: ne,
                h,
                k: m.experts.top_k,
                logits: rlogits,
                sel,
                expert_scale: None,
            });
            mlp(g, &m.shared, LinMode::AddGated(rlogits.at(ne)), de)?;
            let mut gate = Vec::with_capacity(ne);
            let mut up = Vec::with_capacity(ne);
            for gu in &m.experts.gate_up {
                match gu {
                    crate::moe::GateUp::Fused(l) if !off(l) => gate.push(l.as_ref()),
                    crate::moe::GateUp::Split(a, b) if !off(a) && !off(b) => {
                        gate.push(a.as_ref());
                        up.push(b.as_ref());
                    }
                    _ => return None,
                }
            }
            g.ops.push(Op::Experts(crate::gpu::decode::Experts {
                gate,
                up: (!up.is_empty()).then_some(up),
                down: m.experts.down.iter().map(|l| l.as_ref()).collect(),
                h,
                inter: m.experts.inter,
                k: m.experts.top_k,
                x: hn,
                sel,
                act: eact,
                y: x,
                gelu: false,
                add: true,
                edits: ee.map(|e| (hot.at(e.hot), e.n)),
            }));
            Some(())
        };
        let (nk, nv, hdk, hdv) = (
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
        );
        let (key_dim, value_dim) = (nk * hdk, nv * hdv);
        let conv_dim = 2 * key_dim + value_dim;
        let has_gdn = self.layers.iter().any(|l| matches!(l.mixer, Mixer::Gdn(_)));
        // Gated DeltaNet: raw projections [qkv | z | b | a], then the conv outputs.
        let gproj = g.buf(if has_gdn { conv_dim + value_dim + 2 * nv } else { 1 });
        let gconv = g.buf(if has_gdn { conv_dim } else { 1 });
        let gout = g.buf(if has_gdn { value_dim } else { 1 });
        // Multi-token plans: the residual after the captured layers, copied out.
        let cap = multi.map(|m| g.buf((m.len() * h).max(1)));
        if let (Some(c), Some(m)) = (cap, multi) {
            g.capture = Some((c, m.len() * h));
        }
        let capture = |g: &mut Graph<'s>, li: usize| {
            if let (Some(c), Some(m)) = (cap, multi) {
                for (i, _) in m.iter().enumerate().filter(|e| *e.1 == li) {
                    g.ops.push(Op::Ew {
                        kind: crate::gpu::decode::Ew::Scale(1.0),
                        a: x,
                        b: None,
                        out: c.at(i * h),
                        n: h,
                    });
                }
            }
        };
        for (li, layer) in self.layers.iter().enumerate() {
            g.units.push((g.ops.len(), li));
            // Multi-token plans keep each recurrent layer's inputs in its own buffers.
            let (gproj, gconv) = if multi.is_some() && matches!(layer.mixer, Mixer::Gdn(_)) {
                (g.buf(conv_dim + value_dim + 2 * nv), g.buf(conv_dim))
            } else {
                (gproj, gconv)
            };
            g.ops.push(Op::norm(x, hn, &layer.ln_in, h, eps));
            let at = match &layer.mixer {
                Mixer::Attn(at) => at,
                Mixer::Gdn(gd) => {
                    let st = g.states.len();
                    g.states.push(nv * hdk * hdv);
                    g.states.push((cfg.linear_conv_kernel_dim - 1) * conv_dim);
                    g.ops.push(lin(&gd.qkv, conv_dim, h, hn, gproj, LinMode::Store)?);
                    g.ops.push(lin(&gd.z, value_dim, h, hn, gproj.at(conv_dim), LinMode::Store)?);
                    g.ops.push(lin(&gd.b, nv, h, hn, gproj.at(conv_dim + value_dim), LinMode::Store)?);
                    g.ops
                        .push(lin(&gd.a, nv, h, hn, gproj.at(conv_dim + value_dim + nv), LinMode::Store)?);
                    g.ops.push(Op::GdnConv {
                        x: gproj,
                        out: gconv,
                        state: st + 1,
                        w: &gd.conv_w,
                        dim: conv_dim,
                        kc: cfg.linear_conv_kernel_dim,
                    });
                    g.ops.push(Op::GdnRule(crate::gpu::decode::GdnRule {
                        c: gconv,
                        z: gproj.at(conv_dim),
                        b: gproj.at(conv_dim + value_dim),
                        a: gproj.at(conv_dim + value_dim + nv),
                        out: gout,
                        state: st,
                        nk,
                        nv,
                        hdk,
                        hdv,
                        tiled: cfg.v_heads_tiled,
                        sigmoid_gate: cfg.gdn_sigmoid_gate(),
                        decay_coef: &gd.decay_coef,
                        dt_bias: &gd.dt_bias,
                        norm_w: &gd.norm_w,
                        eps,
                    }));
                    g.ops.push(lin(&gd.out, h, value_dim, gout, x, LinMode::Add)?);
                    steer(&mut g, 2 * li + 1);
                    g.ops.push(Op::norm(x, hn, &layer.ln_post, h, eps));
                    ffn(&mut g, &layer.ffn, li)?;
                    steer(&mut g, 2 * li + 2);
                    capture(&mut g, li);
                    continue;
                }
            };
            if at.bias.is_some() {
                return None;
            }
            let slot = g.kv.len();
            g.kv.push((nkv, hd));
            g.ops.push(lin(&at.q, qw, h, hn, qkv, LinMode::Store)?);
            g.ops.push(lin(&at.k, nkv * hd, h, hn, qkv.at(qw), LinMode::Store)?);
            g.ops.push(lin(&at.v, nkv * hd, h, hn, qkv.at(qw + nkv * hd), LinMode::Store)?);
            g.ops.push(Op::QkRope {
                qkv,
                q_out: qo,
                kv: slot,
                nh,
                nkv,
                hd,
                half_rot: self.inv_freq.len(),
                q_stride: if gated { 2 * hd } else { hd },
                k_off: qw,
                v_off: qw + nkv * hd,
                interleaved: cfg.rope_interleaved,
                q_norm: at.q_norm.as_deref(),
                k_norm: at.k_norm.as_deref(),
                eps,
                rope: 0,
                v_norm: false,
                v_from_k: false,
            });
            g.ops.push(Op::Attend {
                q: qo,
                out: att,
                kv: slot,
                nh,
                nkv,
                hd,
                scale: (hd as f32).powf(-0.5),
                window: None,
                gate: gated.then_some((qkv.at(hd), 2 * hd)),
                block: None,
            });
            g.ops.push(lin(&at.o, h, nh * hd, att, x, LinMode::Add)?);
            steer(&mut g, 2 * li + 1);
            g.ops.push(Op::norm(x, hn, &layer.ln_post, h, eps));
            ffn(&mut g, &layer.ffn, li)?;
            steer(&mut g, 2 * li + 2);
            capture(&mut g, li);
        }
        g.ops.push(Op::norm(x, hidden, &self.final_norm, h, eps));
        g.head_from = g.ops.len();
        g.units.push((g.ops.len(), self.layers.len()));
        g.ops.push(lin(&self.lm_head, cfg.vocab_size, h, hidden, logits, LinMode::Store)?);
        g.logits = Some((logits, cfg.vocab_size));
        Some(g)
    }

    /// Whether decode steps may run on the device: GPU decoding is on, and streamed weights (if
    /// any) may be decoded there ([`crate::gpu::stream_decode_enabled`]).
    fn gpu_decode_on(&self) -> bool {
        crate::gpu::decode_enabled() && (self.stream.is_none() || crate::gpu::stream_decode_enabled())
    }

    /// The GPU decode plan, compiled on first use (pipelines, buffers: ~0.1-0.2 s, so multi-token
    /// passes build it ahead of the first decode step); None when unavailable.
    pub(crate) fn gpu_plan(&self) -> Option<std::sync::Arc<dyn crate::gpu::decode::DecodePlan>> {
        self.gpu_plan_for(None)
    }

    /// The plan for an intervention layout (None: without interventions).
    fn gpu_plan_for(&self, lay: Option<&crate::intervene::GpuLayout>) -> Option<std::sync::Arc<dyn crate::gpu::decode::DecodePlan>> {
        if !self.gpu_decode_on() {
            return None;
        }
        self.gpu_plans.get(lay.map_or(0, |l| l.signature()), || {
            let be = crate::gpu::active()?;
            let g = self.gpu_graph(lay);
            if std::env::var_os("OXINFER_GPU_PLAN_DEBUG").is_some() {
                eprintln!(
                    "[gpu decode graph: {}]",
                    g.as_ref().map_or("none".into(), |g| format!("{} ops", g.ops.len()))
                );
            }
            let p = be.decode_plan(&g?);
            if std::env::var_os("OXINFER_GPU_PLAN_DEBUG").is_some() {
                eprintln!("[gpu decode plan: {}]", if p.is_some() { "built" } else { "none" });
            }
            p
        })
    }

    /// Runs token `tok` through the GPU decode plan: the final-normed hidden state and, if
    /// wanted, the logits. None (nothing changed) when GPU decoding is off or unavailable.
    fn gpu_step(&self, sess: &mut QwenSession, tok: u32, want_logits: bool) -> Option<(Vec<f32>, Option<Vec<f32>>)> {
        if !self.gpu_decode_on() {
            return None;
        }
        if !sess.iv.gpu_ok() {
            return None;
        }
        if self.choosing_device() && !self.stream_choice.want_gpu() {
            return None;
        }
        if let Some(ex) = &self.exp {
            return self.exp_gpu_step(ex, sess, tok, want_logits);
        }
        let ivc = sess.iv.iv.clone();
        let data = ivc.as_ref().and_then(|c| c.gpu_data());
        let plan = self.gpu_plan_for(data.map(|d| &d.layout))?;
        let hot = sess.iv.gpu_hot(sess.pos);
        let _timer = prof::start(Cat::Matmul);
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let mut x = vec![0f32; h];
        self.embed.row(tok as usize, &mut x);
        let (mut cs, sin) = self.rope_cs(&sess.spans, sess.pos, 1);
        cs.extend_from_slice(&sin);
        let mut hidden = vec![0f32; h];
        let mut logits = want_logits.then(|| vec![0f32; cfg.vocab_size]);
        let kept = sess.gpu_states_kept;
        let mut kv: Vec<&mut LayerCache> = sess.kv.iter_mut().flatten().collect();
        let mut states: Vec<&mut [f32]> = sess
            .gdn
            .iter_mut()
            .flatten()
            .flat_map(|g| [&mut g.s[..], &mut g.conv[..]])
            .collect();
        let ok = plan.step(
            &mut sess.gpu,
            crate::gpu::decode::Step {
                pos: sess.pos,
                t: 1,
                rope_cs: &cs,
                x: &x,
                x2: &[],
                kv: &mut kv,
                states: &mut states,
                hidden: &mut hidden,
                logits: logits.as_deref_mut(),
                params: data.zip(ivc.as_ref()).map(|(d, c)| (&hot[..], &d.cold[..], c.id())),
                capture: None,
                scratch: false,
                states_kept: kept,
            },
        );
        sess.gpu_states_kept = ok;
        if !ok {
            return None;
        }
        sess.rec_span = None;
        sess.pos += 1;
        Some((hidden, logits))
    }

    /// The multi-token GPU plan capturing the outputs of `layers` (sorted), compiled on first use
    /// (`fast`: on the batched fast-verification kernels, see
    /// [`crate::gpu::Backend::decode_plan_multi_fast`]).
    /// The multi-token plan without captured layers for a pass of `t` tokens (the smallest of 1,
    /// 8 and [`MULTI_TOKENS`] that holds them).
    pub(crate) fn gpu_multi_plan_pub(&self, t: usize, fast: bool) -> Option<std::sync::Arc<dyn crate::gpu::decode::DecodePlan>> {
        let n = if t == 1 {
            1
        } else if t <= 8 {
            8
        } else {
            MULTI_TOKENS
        };
        (t <= MULTI_TOKENS).then(|| self.gpu_multi_plan(&[], n, fast))?
    }

    /// Qwen4-Exp: the GPU prefill plan (see `Backend::decode_plan_prefill`), compiled on first
    /// use; None when unavailable.
    pub(crate) fn gpu_prefill_plan(&self) -> Option<std::sync::Arc<dyn crate::gpu::decode::DecodePlan>> {
        let ex = self.exp.as_ref()?;
        if !self.gpu_decode_on() {
            return None;
        }
        let tokens = std::env::var("OXINFER_PREFILL_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n >= PREFILL_GPU_MIN)
            .unwrap_or(1024);
        self.gpu_plans.get((1 << 61) | tokens as u64, || {
            let be = crate::gpu::active()?;
            let t0 = std::time::Instant::now();
            let plan = self.exp_gpu_graph(ex, true, true).and_then(|g| be.decode_plan_prefill(&g, tokens));
            if std::env::var_os("OXINFER_GPU_PLAN_DEBUG").is_some() {
                eprintln!(
                    "[gpu prefill plan: {} in {:.2}s]",
                    if plan.is_some() { "built" } else { "none" },
                    t0.elapsed().as_secs_f64()
                );
            }
            plan
        })
    }

    fn gpu_multi_plan(&self, layers: &[usize], tokens: usize, fast: bool) -> Option<std::sync::Arc<dyn crate::gpu::decode::DecodePlan>> {
        if !self.gpu_decode_on() {
            return None;
        }
        let fast = fast && tokens > 1;
        let key = layers.iter().fold((1u64 << 63) | ((fast as u64) << 62) | tokens as u64, |k, &l| {
            k.wrapping_mul(0x100000001b3).wrapping_add(l as u64 + 1)
        });
        self.gpu_plans.get(key, || {
            let be = crate::gpu::active()?;
            let plan = self.gpu_graph_with(None, Some(layers)).and_then(|g| {
                if fast {
                    be.decode_plan_multi_fast(&g, tokens)
                } else {
                    be.decode_plan_multi(&g, tokens)
                }
            });
            if plan.is_none() {
                eprintln!("[no multi-token GPU plan for this model: speculative verification runs on the CPU]");
            }
            plan
        })
    }

    /// Runs `tokens` (at most [`MULTI_TOKENS`]) through a multi-token GPU plan, as one-token GPU
    /// steps would one after the other: the logits of every token and, in `feats`, the residual
    /// stream entering each of `layers` (as [`crate::dspark::Target::forward_feats`]). None
    /// (nothing changed) when GPU decoding is off or the model cannot.
    fn gpu_verify(&self, sess: &mut QwenSession, tokens: &[u32], layers: &[usize], feats: &mut Vec<f32>, fast: bool) -> Option<Vec<f32>> {
        self.gpu_pass(sess, tokens, layers, feats, fast).map(|r| r.1)
    }

    /// The final-normed hidden states (`[t, hidden]`) and the logits of `tokens` from a GPU pass
    /// (at most [`MULTI_TOKENS`] tokens; a speculative verification, as plain GPU decoding would
    /// compute them one token after the other), recording what a rewind inside the pass needs
    /// ([`QwenSession::rewind`]). None (nothing changed) when GPU decoding is off or the model
    /// cannot.
    /// With `fast`, the pass may use batched kernels whose rounding differs from one-token steps
    /// ([`crate::qwen_mtp::MtpOptions::fast_verify`]).
    pub fn gpu_verify_hidden(&self, sess: &mut QwenSession, tokens: &[u32], fast: bool) -> Option<(Vec<f32>, Vec<f32>)> {
        self.gpu_pass(sess, tokens, &[], &mut Vec::new(), fast)
    }

    fn gpu_pass(
        &self,
        sess: &mut QwenSession,
        tokens: &[u32],
        layers: &[usize],
        feats: &mut Vec<f32>,
        fast: bool,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let t = tokens.len();
        if t == 0 || t > MULTI_TOKENS || sess.iv.active() || sess.positions.is_some() || layers.contains(&0) {
            return None;
        }
        if let Some(ex) = &self.exp {
            return if layers.is_empty() {
                self.exp_gpu_pass(ex, sess, tokens, fast)
            } else {
                None
            };
        }
        let mut cap: Vec<usize> = layers.iter().map(|&l| l - 1).collect();
        cap.sort_unstable();
        cap.dedup();
        // The smallest plan holding `t` (a DSpark block of 7 drafts verifies 8 tokens; one token,
        // a round without drafts, runs the single-token kernels as plain decoding does).
        let plan = self.gpu_multi_plan(
            &cap,
            if t == 1 {
                1
            } else if t <= 8 {
                8
            } else {
                MULTI_TOKENS
            },
            fast,
        )?;
        let _timer = prof::start(Cat::Matmul);
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let mut x = vec![0f32; t * h];
        for (j, &tok) in tokens.iter().enumerate() {
            self.embed.row(tok as usize, &mut x[j * h..(j + 1) * h]);
        }
        let half = self.inv_freq.len();
        let (cos, sin) = self.rope_cs(&sess.spans, sess.pos, t);
        let mut cs = Vec::with_capacity(2 * t * half);
        for j in 0..t {
            cs.extend_from_slice(&cos[j * half..(j + 1) * half]);
            cs.extend_from_slice(&sin[j * half..(j + 1) * half]);
        }
        let mut hidden = vec![0f32; t * h];
        let mut logits = vec![0f32; t * cfg.vocab_size];
        let mut captured = vec![0f32; t * cap.len() * h];
        // The recurrent states before the pass, for a rewind inside it (none for one token),
        // copied in parallel into the last pass's buffers.
        let mut snap = sess.gpu_rec.take().map(|r| r.states).unwrap_or_default();
        if t > 1 {
            let src: Vec<&[f32]> = sess.gdn.iter().flatten().flat_map(|g| [&g.s[..], &g.conv[..]]).collect();
            snap.resize_with(src.len(), Vec::new);
            for (d, s) in snap.iter_mut().zip(&src) {
                d.resize(s.len(), 0.0);
            }
            let jobs: Vec<(usize, usize, usize)> = src
                .iter()
                .zip(snap.iter_mut())
                .map(|(s, d)| (s.as_ptr() as usize, d.as_mut_ptr() as usize, s.len()))
                .collect();
            crate::pool::run(jobs.len(), &|i| {
                let (s, d, n) = jobs[i];
                // SAFETY: each job copies one state into its own snapshot buffer.
                unsafe { std::ptr::copy_nonoverlapping(s as *const f32, d as *mut f32, n) };
            });
        }
        let kept = sess.gpu_states_kept;
        let mut kv: Vec<&mut LayerCache> = sess.kv.iter_mut().flatten().collect();
        let mut states: Vec<&mut [f32]> = sess
            .gdn
            .iter_mut()
            .flatten()
            .flat_map(|g| [&mut g.s[..], &mut g.conv[..]])
            .collect();
        let ok = plan.step(
            &mut sess.gpu,
            crate::gpu::decode::Step {
                pos: sess.pos,
                t,
                rope_cs: &cs,
                x: &x,
                x2: &[],
                kv: &mut kv,
                states: &mut states,
                hidden: &mut hidden,
                logits: Some(&mut logits),
                params: None,
                capture: Some(&mut captured),
                scratch: false,
                states_kept: kept,
            },
        );
        sess.gpu_states_kept = ok;
        if !ok {
            if std::env::var_os("OXINFER_METAL_TRACE").is_some() {
                eprintln!("[multi-token GPU step failed: {t} tokens at {}]", sess.pos);
            }
            return None;
        }
        sess.rec_span = None;
        sess.gpu_rec = (t > 1).then_some(GpuRec {
            plan,
            start: sess.pos,
            len: t,
            states: snap,
        });
        sess.pos += t;
        let (nf, nc) = (layers.len(), cap.len());
        feats.clear();
        feats.resize(t * nf * h, 0.0);
        for j in 0..t {
            for (fi, &l) in layers.iter().enumerate() {
                let ci = cap.binary_search(&(l - 1)).unwrap();
                feats[(j * nf + fi) * h..(j * nf + fi + 1) * h].copy_from_slice(&captured[(j * nc + ci) * h..(j * nc + ci + 1) * h]);
            }
        }
        Some((hidden, logits))
    }

    /// RoPE tables for positions `start..start + t`: plain positions, or M-RoPE positions when
    /// the sequence has vision spans (identical to plain positions for text-only sequences).
    pub(crate) fn rope_cs(&self, spans: &[VisionSpan], start: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
        let axis = match &self.mrope_axis {
            Some(a) if !spans.is_empty() => a,
            _ => {
                let positions: Vec<usize> = (start..start + t).collect();
                return rope_tables(&positions, &self.inv_freq);
            }
        };
        let half = self.inv_freq.len();
        let mut cos = vec![0f32; t * half];
        let mut sin = vec![0f32; t * half];
        for (i, p) in mrope_positions(spans, start, t).iter().enumerate() {
            for (j, (&f, &ax)) in self.inv_freq.iter().zip(axis).enumerate() {
                let a = p[ax as usize] as f32 * f;
                cos[i * half + j] = a.cos();
                sin[i * half + j] = a.sin();
            }
        }
        (cos, sin)
    }

    /// RoPE tables for explicit `[time, height, width]` positions (the time axis alone without
    /// M-RoPE).
    pub(crate) fn rope_cs_at(&self, pos: &[[usize; 3]]) -> (Vec<f32>, Vec<f32>) {
        let Some(axis) = &self.mrope_axis else {
            let positions: Vec<usize> = pos.iter().map(|p| p[0]).collect();
            return rope_tables(&positions, &self.inv_freq);
        };
        let half = self.inv_freq.len();
        let mut cos = vec![0f32; pos.len() * half];
        let mut sin = vec![0f32; pos.len() * half];
        for (i, p) in pos.iter().enumerate() {
            for (j, (&f, &ax)) in self.inv_freq.iter().zip(axis).enumerate() {
                let a = p[ax as usize] as f32 * f;
                cos[i * half + j] = a.cos();
                sin[i * half + j] = a.sin();
            }
        }
        (cos, sin)
    }

    pub fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        let _unit = crate::stream::enter(&self.stream, self.layers.len());
        let _timer = prof::start(Cat::Head);
        let mut logits = vec![0f32; t * self.cfg.vocab_size];
        linear(&self.lm_head, self.cfg.vocab_size, self.cfg.hidden_size, hidden, t, &mut logits);
        logits
    }

    /// [`Self::logits`] of a few rows (1 to 4) on the accelerator's matrix-vector kernels when
    /// one is active and the weights do not stream (a draft's head: the drafts may differ from
    /// the CPU's in the last bits, the verification keeps the output exact).
    pub fn logits_device(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        if self.stream.is_none() && t <= 4 {
            let _timer = prof::start(Cat::Head);
            let (v, h) = (self.cfg.vocab_size, self.cfg.hidden_size);
            let mut logits = vec![0f32; t * v];
            if crate::model::linear_rows_device(&mut [(&self.lm_head, None, v, &mut logits[..])], h, hidden, t) {
                return logits;
            }
        }
        self.logits(hidden, t)
    }

    pub fn forward(&self, sess: &mut QwenSession, tokens: &[u32]) -> Vec<f32> {
        let t = tokens.len();
        let t0 = crate::clock::Instant::now();
        let measure = t == 1 && self.choosing_device() && self.stream_choice.measuring();
        if t == 1 {
            // OXINFER_GPU_DECODE_CHECK=1: compare every GPU step with the CPU path.
            let before = crate::gpu::decode::check().then(|| sess.clone());
            if let Some((_, Some(l))) = self.gpu_step(sess, tokens[0], true) {
                if let Some(mut s2) = before {
                    let hs = self.forward_impl(&mut s2, tokens, &[], false, false);
                    crate::gpu::decode::report(s2.pos - 1, &self.logits(&hs, 1), &l);
                }
                if measure {
                    self.stream_choice.record(true, t0.elapsed().as_secs_f64());
                }
                return l;
            }
        }
        let hs = self.forward_hidden_mm(sess, tokens, &[]);
        let l = self.logits(&hs[(t - 1) * self.cfg.hidden_size..], 1);
        if measure {
            self.stream_choice.record(false, t0.elapsed().as_secs_f64());
        }
        l
    }

    /// Streamed with on-demand experts and GPU decoding left to decide: plain decode steps are
    /// measured on both and the faster kept ([`crate::gpu::decode::DeviceChoice`]).
    fn choosing_device(&self) -> bool {
        self.stream.is_some() && crate::gpu::stream_decode_forced().is_none() && self.layers.iter().any(|l| matches!(l.ffn, Ffn::Moe(_)))
    }

    /// Final-normed hidden states `[t, hidden]` for `tokens` appended at `sess.pos`.
    pub fn forward_hidden_mm(&self, sess: &mut QwenSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        self.forward_impl(sess, tokens, mm, false, true)
    }

    /// `forward_hidden_mm` without overrides that also records what `sess.rewind` needs to go
    /// back to any position inside `tokens` (speculative verification): one copy of the Gated
    /// DeltaNet states, and a replay of the kept tokens' delta rule on rewind.
    pub fn forward_hidden_rec(&self, sess: &mut QwenSession, tokens: &[u32]) -> Vec<f32> {
        self.forward_impl(sess, tokens, &[], true, true)
    }

    fn forward_impl(&self, sess: &mut QwenSession, tokens: &[u32], mm: &[(usize, &[f32])], record: bool, gpu: bool) -> Vec<f32> {
        let t = tokens.len();
        assert!(t > 0);
        sess.gpu_rec = None;
        if t > 1 && gpu {
            let t0 = std::time::Instant::now();
            let _ = self.gpu_plan();
            if std::env::var_os("OXINFER_GPU_PLAN_DEBUG").is_some() {
                eprintln!("[gpu plan ready in {:.2}s]", t0.elapsed().as_secs_f64());
            }
        }
        if t == 1
            && gpu
            && mm.is_empty()
            && !record
            && let Some((hs, _)) = self.gpu_step(sess, tokens[0], false)
        {
            return hs;
        }
        if let Some(ex) = &self.exp {
            if gpu
                && !record
                && prefill_gpu(t, "plan")
                && let Some(hs) = self.exp_gpu_prefill(ex, sess, tokens, mm)
            {
                return hs;
            }
            return self.exp_forward(ex, sess, tokens, mm, record);
        }
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps as f32;
        let start = sess.pos;
        let par = t > 1;

        let timer = prof::start(Cat::Embed);
        if let Some(s) = &self.stream {
            // All rows at once, ahead of the read-ahead, rather than a page fault per lookup.
            let rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| self.embed.row_ranges(tk as usize, h)).collect();
            s.need_ranges(&rows);
        }
        let mut x = vec![0f32; t * h];
        for_chunks(&mut x, h, par, |ti, row| {
            let tok = tokens[ti] as usize;
            self.embed.row(tok, row);
        });
        for &(i, emb) in mm {
            x[i * h..(i + 1) * h].copy_from_slice(emb);
        }
        let (cos, sin) = match &sess.positions {
            Some(p) if p.len() >= start + t => self.rope_cs_at(&p[start..start + t]),
            _ => self.rope_cs(&sess.spans, start, t),
        };
        drop(timer);
        let iv = sess.iv.active().then(|| sess.iv.clone());
        if let Some(iv) = &iv {
            iv.point(0, &mut x, h, start);
        }

        let mut hn = vec![0f32; t * h];
        let mut y = vec![0f32; t * h];
        for (l, layer) in self.layers.iter().enumerate() {
            let _unit = crate::stream::enter(&self.stream, l);
            rmsnorm_f32w_rows(&x, h, &layer.ln_in, eps, &mut hn);
            match &layer.mixer {
                Mixer::Gdn(gd) => {
                    let rec = if record { Some(&mut sess.rec[l]) } else { None };
                    sess.gpu_states_kept = false;
                    self.gdn_forward(gd, sess.gdn[l].as_mut().unwrap(), rec, &hn, t, &mut y)
                }
                Mixer::Attn(at) => self.attn_forward(at, sess.kv[l].as_mut().unwrap(), &hn, t, start, &cos, &sin, &mut y),
            }
            add_inplace(&mut x, &y);
            if let Some(iv) = &iv {
                iv.point(2 * l + 1, &mut x, h, start);
            }
            rmsnorm_f32w_rows(&x, h, &layer.ln_post, eps, &mut hn);
            let need = |ids: &[usize]| {
                if let Some(s) = &self.stream {
                    s.need_parts(l, ids)
                }
            };
            let hook = iv.as_ref().and_then(|s| s.hook(l, start));
            layer.ffn.forward(
                &hn,
                t,
                h,
                &mut y,
                self.stream.as_ref().map(|_| &need as &dyn Fn(&[usize])),
                hook.as_ref(),
            );
            add_inplace(&mut x, &y);
            if let Some(iv) = &iv {
                iv.point(2 * l + 2, &mut x, h, start);
            }
            if sess.capture.contains(&l) {
                sess.captured.push((l, x.clone()));
            }
            // Streaming a MoE: guess the next layer's experts now, so they are read while its
            // attention runs.
            if let (Some(s), Some(next)) = (&self.stream, self.layers.get(l + 1))
                && let Ffn::Moe(m) = &next.ffn
                && s.options().predict_experts
            {
                s.prefetch_parts(l + 1, &m.predict(&x, t, h, &next.ln_post, eps));
            }
        }
        rmsnorm_f32w_rows(&x.clone(), h, &self.final_norm, eps, &mut x);
        sess.rec_span = if record { Some((start, t)) } else { None };
        sess.pos += t;
        x
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_forward(
        &self,
        at: &Attn,
        cache: &mut LayerCache,
        hn: &[f32],
        t: usize,
        start: usize,
        cos: &[f32],
        sin: &[f32],
        y: &mut [f32],
    ) {
        self.attn_forward_sel(at, cache, hn, t, start, cos, sin, None, y)
    }

    /// [`Self::attn_forward`] where token `i` attends only to the cached positions `sel[i]` when
    /// set (Qwen4-Exp sparse attention, see `crate::qwen4exp`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_forward_sel(
        &self,
        at: &Attn,
        cache: &mut LayerCache,
        hn: &[f32],
        t: usize,
        start: usize,
        cos: &[f32],
        sin: &[f32],
        sel: Option<&[Option<Vec<u32>>]>,
        y: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let (nh, nkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let eps = cfg.rms_norm_eps as f32;
        let half = self.inv_freq.len();
        let gated = cfg.attn_output_gate;
        let rope = if cfg.rope_interleaved { apply_rope_interleaved } else { apply_rope };
        let qw = if gated { 2 * nh * hd } else { nh * hd };
        let mut qg = vec![0f32; t * qw];
        let mut k = vec![0f32; t * nkv * hd];
        let mut v = vec![0f32; t * nkv * hd];
        linear_multi(
            &mut [
                (&at.q, qw, &mut qg[..]),
                (&at.k, nkv * hd, &mut k[..]),
                (&at.v, nkv * hd, &mut v[..]),
            ],
            h,
            hn,
            t,
        );
        if let Some([bq, bk, bv]) = &at.bias {
            for (buf, b) in [(&mut qg, bq), (&mut k, bk), (&mut v, bv)] {
                for_chunks(buf, b.len(), t > 1, |_, r| add_inplace(r, b));
            }
        }

        // Split query / gate, norm, partial RoPE, fold in the 1/sqrt(hd) scaling.
        let scale = (hd as f32).powf(-0.5);
        let mut q = vec![0f32; t * nh * hd];
        let mut gate = vec![0f32; t * nh * hd];
        let par = t > 1;
        {
            let qp = SendPtrF(q.as_mut_ptr());
            let gp = SendPtrF(gate.as_mut_ptr());
            let run = |i: usize| {
                let (ti, hh) = (i / nh, i % nh);
                // SAFETY: disjoint (token, head) slices.
                let qo = unsafe { std::slice::from_raw_parts_mut(qp.get().add(i * hd), hd) };
                let src = if gated {
                    &qg[ti * qw + hh * 2 * hd..ti * qw + hh * 2 * hd + hd]
                } else {
                    &qg[ti * qw + hh * hd..ti * qw + (hh + 1) * hd]
                };
                match &at.q_norm {
                    Some(w) => rmsnorm_f32w(src, w, eps, qo),
                    None => qo.copy_from_slice(src),
                }
                rope(
                    &mut qo[..2 * half],
                    &cos[ti * half..(ti + 1) * half],
                    &sin[ti * half..(ti + 1) * half],
                );
                for v in qo.iter_mut() {
                    *v *= scale;
                }
                if gated {
                    let go = unsafe { std::slice::from_raw_parts_mut(gp.get().add(i * hd), hd) };
                    go.copy_from_slice(&qg[ti * qw + hh * 2 * hd + hd..ti * qw + (hh + 1) * 2 * hd]);
                }
            };
            if par {
                pool::run(t * nh, &run);
            } else {
                for i in 0..t * nh {
                    run(i);
                }
            }
        }
        let mut kn = vec![0f32; t * nkv * hd];
        for_chunks(&mut kn, hd, par, |i, ko| {
            let ti = i / nkv;
            match &at.k_norm {
                Some(w) => rmsnorm_f32w(&k[i * hd..(i + 1) * hd], w, eps, ko),
                None => ko.copy_from_slice(&k[i * hd..(i + 1) * hd]),
            }
            rope(
                &mut ko[..2 * half],
                &cos[ti * half..(ti + 1) * half],
                &sin[ti * half..(ti + 1) * half],
            );
        });
        cache.push(&kn, &v, t);
        let mut att = vec![0f32; t * nh * hd];
        // Qwen4-Exp prefill: on the GPU (dense, or over each token's QSA selection).
        let on_gpu = prefill_gpu_exp(cfg, t, "attn")
            && crate::gpu::active().is_some_and(|be| {
                let (kc, vc) = cache.data();
                be.attention_cached(&mut crate::gpu::AttnJob {
                    q: &q,
                    k: kc,
                    v: vc,
                    start,
                    t,
                    nh,
                    nkv,
                    hd,
                    sel,
                    out: &mut att,
                })
            });
        match sel {
            _ if on_gpu => {}
            Some(sel) if sel.iter().any(|s| s.is_some()) => crate::qwen4exp::attention_sel(&q, cache, nh, t, start, sel, &mut att),
            _ => attention(&q, cache, nh, t, start, None, &mut att),
        }
        if gated {
            for (a, &g) in att.iter_mut().zip(&gate) {
                *a *= 1.0 / (1.0 + fast_exp(-g));
            }
        }
        linear(&at.o, h, nh * hd, &att, t, y);
    }

    pub(crate) fn gdn_forward(&self, gd: &Gdn, st: &mut GdnState, mut rec: Option<&mut GdnRecord>, hn: &[f32], t: usize, y: &mut [f32]) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps as f32;
        let (nk, nv, hdk, hdv) = (
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
        );
        let (key_dim, value_dim) = (nk * hdk, nv * hdv);
        let conv_dim = 2 * key_dim + value_dim;
        let kc = cfg.linear_conv_kernel_dim;

        let mut qkv = vec![0f32; t * conv_dim];
        let mut z = vec![0f32; t * value_dim];
        let mut b = vec![0f32; t * nv];
        let mut a = vec![0f32; t * nv];
        linear_multi(
            &mut [
                (&gd.qkv, conv_dim, &mut qkv[..]),
                (&gd.z, value_dim, &mut z[..]),
                (&gd.b, nv, &mut b[..]),
                (&gd.a, nv, &mut a[..]),
            ],
            h,
            hn,
            t,
        );

        // Causal depthwise conv over time (kernel kc, left context from the state), then SiLU.
        let _timer = prof::start(Cat::Attention);
        let mut c = vec![0f32; t * conv_dim];
        {
            let st_conv = &st.conv;
            let src = |i: isize, ch: usize| -> f32 {
                if i >= 0 {
                    qkv[i as usize * conv_dim + ch]
                } else {
                    st_conv[((kc as isize - 1 + i) as usize) * conv_dim + ch]
                }
            };
            for_chunks(&mut c, conv_dim, t > 1, |i, row| {
                for ch in 0..conv_dim {
                    let mut s = 0f32;
                    for j in 0..kc {
                        s += gd.conv_w[ch * kc + j] * src(i as isize - (kc as isize - 1) + j as isize, ch);
                    }
                    row[ch] = s / (1.0 + fast_exp(-s));
                }
            });
        }
        let rule = DeltaRule {
            nk,
            nv,
            hdk,
            hdv,
            tiled: cfg.v_heads_tiled,
            decay_coef: gd.decay_coef.clone(),
            dt_bias: gd.dt_bias.clone(),
        };
        if let Some(r) = rec.as_deref_mut() {
            r.s0.clear();
            r.s0.extend_from_slice(&st.s);
            r.conv0.clear();
            r.conv0.extend_from_slice(&st.conv);
            r.qkv.clone_from(&qkv);
            r.c.clone_from(&c);
            r.b.clone_from(&b);
            r.a.clone_from(&a);
            r.conv_dim = conv_dim;
        }
        // Update conv state with the last kc-1 raw inputs.
        {
            let keep = kc - 1;
            let mut new_state = vec![0f32; keep * conv_dim];
            for slot in 0..keep {
                // slot 0 is the oldest kept input: time index t - keep + slot
                let ti = t as isize - keep as isize + slot as isize;
                if ti >= 0 {
                    new_state[slot * conv_dim..(slot + 1) * conv_dim]
                        .copy_from_slice(&qkv[ti as usize * conv_dim..(ti as usize + 1) * conv_dim]);
                } else {
                    let old = (keep as isize + ti) as usize;
                    new_state[slot * conv_dim..(slot + 1) * conv_dim].copy_from_slice(&st.conv[old * conv_dim..(old + 1) * conv_dim]);
                }
            }
            st.conv.copy_from_slice(&new_state);
        }

        // Gated delta rule, recurrent over time, parallel over value heads.
        let mut out = vec![0f32; t * value_dim];
        let sig = cfg.gdn_sigmoid_gate();
        // Qwen4-Exp prefill: the rule on the GPU (one pass over the chunk).
        let on_gpu = prefill_gpu_exp(cfg, t, "gdn") && {
            let mut ba = vec![0f32; t * 2 * nv];
            for (ti, row) in ba.chunks_mut(2 * nv).enumerate() {
                row[..nv].copy_from_slice(&b[ti * nv..(ti + 1) * nv]);
                row[nv..].copy_from_slice(&a[ti * nv..(ti + 1) * nv]);
            }
            crate::gpu::active().is_some_and(|be| {
                be.gdn_rule(&mut crate::gpu::GdnJob {
                    c: &c,
                    ba: &ba,
                    state: &mut st.s,
                    out: &mut out,
                    t,
                    nk,
                    nv,
                    hdk,
                    hdv,
                    tiled: cfg.v_heads_tiled,
                    decay_coef: &gd.decay_coef,
                    dt_bias: &gd.dt_bias,
                })
            })
        };
        if !on_gpu {
            rule.run(&mut st.s, &c, &b, &a, t, Some(&mut out));
        }
        if let Some(r) = rec {
            r.rule = rule;
        }
        drop(_timer);

        gated_norm(&mut out, &z, &gd.norm_w, eps, sig, t * nv > 64);
        linear(&gd.out, h, value_dim, &out, t, y);
    }
}

/// Gated DeltaNet's gated RMSNorm per head (`hdv = w.len()`): `norm(o) * w * silu(z)` (sigmoid(z)
/// for Qwen4-Exp).
fn gated_norm(out: &mut [f32], z: &[f32], w: &[f32], eps: f32, sig: bool, par: bool) {
    let hdv = w.len();
    for_chunks(out, hdv, par, |i, o| {
        let ss: f32 = o.iter().map(|v| v * v).sum::<f32>() / hdv as f32 + eps;
        let r = ss.powf(-0.5);
        let zz = &z[i * hdv..(i + 1) * hdv];
        for j in 0..hdv {
            let g = zz[j];
            let gate = if sig { 1.0 / (1.0 + (-g).exp()) } else { g / (1.0 + fast_exp(-g)) };
            o[j] = o[j] * r * w[j] * gate;
        }
    });
}

/// The target of a DSpark draft ([`crate::dspark`]): Qwen3 dense, Qwen3.5 / 3.6 / 3.8 dense and
/// MoE. The verification forward records the Gated DeltaNet inputs, so a rejected draft rewinds
/// exactly (see [`QwenSession::rewind`]).
impl crate::dspark::Target for QwenModel<'_> {
    fn forward_feats(
        &self,
        sess: &mut QwenSession,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        let (t, h) = (tokens.len(), self.cfg.hidden_size);
        // The residual entering layer `l` is layer `l - 1`'s output (the embeddings for 0).
        sess.capture = layers.iter().filter(|&&l| l > 0).map(|&l| l - 1).collect();
        sess.captured.clear();
        let hs = self.forward_impl(sess, tokens, mm, true, true);
        sess.capture.clear();
        let captured = std::mem::take(&mut sess.captured);
        let nf = layers.len();
        feats.clear();
        feats.resize(t * nf * h, 0.0);
        for (fi, &l) in layers.iter().enumerate() {
            let rows: Vec<f32> = if l == 0 {
                let mut x = vec![0f32; t * h];
                for (ti, row) in x.chunks_mut(h).enumerate() {
                    self.embed.row(tokens[ti] as usize, row);
                }
                for &(i, emb) in mm {
                    x[i * h..(i + 1) * h].copy_from_slice(emb);
                }
                x
            } else {
                captured
                    .iter()
                    .find(|c| c.0 == l - 1)
                    .map(|c| c.1.clone())
                    .unwrap_or_else(|| panic!("DSpark: target layer {l} out of range"))
            };
            for ti in 0..t {
                feats[(ti * nf + fi) * h..(ti * nf + fi + 1) * h].copy_from_slice(&rows[ti * h..(ti + 1) * h]);
            }
        }
        hs
    }
    fn forward_verify(&self, sess: &mut QwenSession, tokens: &[u32], layers: &[usize], feats: &mut Vec<f32>, fast: bool) -> Vec<f32> {
        if let Some(l) = self.gpu_verify(sess, tokens, layers, feats, fast) {
            return l;
        }
        let hs = crate::dspark::Target::forward_feats(self, sess, tokens, &[], layers, feats);
        self.logits(&hs, tokens.len())
    }
    fn embed_row(&self, token: u32, out: &mut [f32]) {
        if let Some(s) = &self.stream {
            s.need_ranges(&self.embed.row_ranges(token as usize, self.cfg.hidden_size));
        }
        self.embed.row(token as usize, out);
    }
    fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        self.logits(hidden, t)
    }
    fn lm_head(&self) -> Option<&Lin<'_>> {
        Some(&self.lm_head)
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn truncate(&self, sess: &mut QwenSession, len: usize) {
        sess.rewind(len).expect("DSpark rollback within the verification pass");
    }
}

/// Snapshots ([`crate::snapshot`]): the position, the vision spans, the attention layers' KV
/// caches and the Gated DeltaNet layers' recurrent and convolution states (copied into the
/// session's own page-aligned buffers, which GPU decoding uses in place). A restored session
/// cannot rewind to before the restore point. Qwen4-Exp sessions and explicit positions are not
/// supported.
impl crate::snapshot::SessionState for QwenSession {
    fn write_state<W: std::io::Write>(&self, w: &mut crate::snapshot::SnapWriter<W>) -> Result<()> {
        if self.exp.is_some() || self.positions.is_some() {
            bail!("session snapshots are not supported for Qwen4-Exp models or explicit positions");
        }
        w.u64s("qwen.pos", &[self.pos as u64])?;
        let spans: Vec<u64> = self.spans.iter().flat_map(|s| [s.start as u64, s.gh as u64, s.gw as u64]).collect();
        w.u64s("qwen.spans", &spans)?;
        for (l, c) in self.kv.iter().enumerate() {
            if let Some(c) = c {
                crate::snapshot::write_kv(w, &format!("qwen.kv.{l}"), c)?;
            }
        }
        for (l, g) in self.gdn.iter().enumerate() {
            if let Some(g) = g {
                w.f32s(&format!("qwen.gdn.{l}.s"), &g.s)?;
                w.f32s(&format!("qwen.gdn.{l}.conv"), &g.conv)?;
            }
        }
        Ok(())
    }
    fn read_state(&mut self, r: &mut crate::snapshot::SnapReader) -> Result<()> {
        if self.exp.is_some() {
            bail!("session snapshots are not supported for Qwen4-Exp models");
        }
        let pos = r.u64s_exact("qwen.pos", 1)?[0] as usize;
        let spans = r.u64s("qwen.spans", 3 << 20)?;
        if spans.len() % 3 != 0 {
            bail!("snapshot: bad vision spans");
        }
        self.spans = spans
            .chunks(3)
            .map(|s| VisionSpan {
                start: s[0] as usize,
                gh: s[1] as usize,
                gw: s[2] as usize,
            })
            .collect();
        if self.spans.iter().any(|s| s.start >= pos || s.gh == 0 || s.gw == 0) {
            bail!("snapshot: bad vision spans");
        }
        for (l, c) in self.kv.iter_mut().enumerate() {
            if let Some(c) = c {
                crate::snapshot::read_kv(r, &format!("qwen.kv.{l}"), c, pos)?;
            }
        }
        for (l, g) in self.gdn.iter_mut().enumerate() {
            if let Some(g) = g {
                r.f32s_into(&format!("qwen.gdn.{l}.s"), &mut g.s)?;
                r.f32s_into(&format!("qwen.gdn.{l}.conv"), &mut g.conv)?;
            }
        }
        self.rec_span = None;
        self.gpu_rec = None;
        self.gpu_states_kept = false;
        self.pos = pos;
        Ok(())
    }
    fn truncatable(&self) -> bool {
        self.gdn.iter().all(|g| g.is_none()) && self.exp.is_none()
    }
}

impl crate::intervene::Host for QwenModel<'_> {
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }
    fn mlp_inter(&self, layer: usize, target: Target) -> Option<usize> {
        self.layers.get(layer)?.ffn.inter(target)
    }
    fn down_column(&self, layer: usize, target: Target, n: usize) -> Vec<f32> {
        self.layers[layer].ffn.down_column(target, n, self.cfg.hidden_size)
    }
}

impl Moe<'_> {
    /// The experts this layer would route `x` (`[t, H]`, the residual stream before the layer)
    /// to, approximating the layer's MoE input by `x` normed with `norm` (its post-attention
    /// norm): a guess made one layer ahead so that weight streaming can read them early.
    pub(crate) fn predict(&self, x: &[f32], t: usize, h: usize, norm: &[f32], eps: f32) -> Vec<usize> {
        let ne = self.experts.n_experts();
        let mut logits = vec![0f32; t * ne];
        for_chunks(&mut logits, ne, t > 1, |ti, l| {
            let mut xn = vec![0f32; h];
            rmsnorm_f32w(&x[ti * h..(ti + 1) * h], norm, eps, &mut xn);
            for (v, w) in l.iter_mut().zip(self.router.chunks(h)) {
                *v = dot(w, &xn);
            }
        });
        let k = self.experts.top_k;
        let mut used: Vec<usize> = logits.chunks(ne).flat_map(|l| route_top_k(l, k).into_iter().map(|r| r.0)).collect();
        used.sort_unstable();
        used.dedup();
        used
    }

    fn forward(&self, x: &[f32], t: usize, h: usize, y: &mut [f32], need: NeedExperts, hook: Option<&NeuronHook>) {
        // The shared expert, scaled by its sigmoid gate, runs with the routed experts (one batch
        // per stage on the CPU decode path).
        let shared = crate::moe::Shared {
            gate: &self.shared.gate,
            up: &self.shared.up,
            down: &self.shared.down,
            inter: self.shared.inter,
            act: Act::Silu,
            hook,
            scale: Some(
                (0..t)
                    .map(|ti| 1.0 / (1.0 + (-dot(&self.shared_gate, &x[ti * h..(ti + 1) * h])).exp()))
                    .collect(),
            ),
        };
        let ne = self.experts.n_experts();
        let mut logits = vec![0f32; t * ne];
        match &self.router_mm {
            Some(r) if prefill_gpu(t, "router") => {
                let lin = r.get_or_init(|| {
                    let mut pv = crate::quant::PageVec::<u8>::zeroed(self.router.len() * 4);
                    for (d, v) in pv.as_chunks_mut::<4>().0.iter_mut().zip(&self.router) {
                        d.copy_from_slice(&v.to_le_bytes());
                    }
                    Lin::gguf(crate::ggml::GgufMatrix {
                        kind: crate::ggml::GgmlType::F32,
                        data: crate::quant::QBuf::from(pv),
                        n_out: ne,
                        n_in: h,
                        block_scales: None,
                    })
                });
                linear(lin, ne, h, x, t, &mut logits);
            }
            _ => {
                // Blocks of tokens per router row (each row read once per block; the same dots).
                const RB: usize = 8;
                for_chunks(&mut logits, RB * ne, t > 1, |bi, l| {
                    for (e, w) in self.router.chunks(h).enumerate() {
                        for (j, xr) in x[bi * RB * h..].chunks(h).take(l.len() / ne).enumerate() {
                            l[j * ne + e] = dot(w, xr);
                        }
                    }
                });
            }
        }
        let routes: Vec<_> = logits.chunks(ne).map(|l| route_top_k(l, self.experts.top_k)).collect();
        self.experts.forward(x, t, h, &routes, y, need, hook, Some(&shared));
    }
}

/// Qwen4-Exp passes of at least this many tokens (prefill) run Gated DeltaNet's rule and the
/// attention on the GPU when there is one.
pub(crate) const PREFILL_GPU_MIN: usize = 32;

/// Whether a Qwen4-Exp pass of `t` tokens offloads past the matmuls (`OXINFER_PREFILL_GPU=0`: not).
fn prefill_gpu_exp(cfg: &QwenConfig, t: usize, what: &str) -> bool {
    cfg.is_exp() && prefill_gpu(t, what)
}

/// [`prefill_gpu_exp`] for a part known to be Qwen4-Exp's.
pub(crate) fn prefill_gpu(t: usize, what: &str) -> bool {
    static ON: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let v = ON.get_or_init(|| std::env::var("OXINFER_PREFILL_GPU").unwrap_or_default());
    t >= PREFILL_GPU_MIN && v != "0" && (v.is_empty() || v == "1" || v.split(',').any(|x| x == what))
}

#[derive(Clone, Copy)]
struct SendPtrF(*mut f32);
unsafe impl Send for SendPtrF {}
unsafe impl Sync for SendPtrF {}
impl SendPtrF {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}

#[cfg(all(test, feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
mod prefill_gpu_tests {
    use super::*;
    use crate::gpu::Backend;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn close(a: &[f32], b: &[f32], tol: f32) {
        let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(d <= tol * scale, "max diff {d} (scale {scale})");
    }

    /// `Backend::gdn_rule` against `DeltaRule::run` (then the gated norm, as the forward does).
    #[test]
    fn gdn_rule_matches_cpu() {
        let Ok(be) = crate::metal::MetalBackend::new() else { return };
        let (nk, nv, hdk, hdv, t) = (2, 6, 128, 128, 70);
        let mut sd = 5u64;
        let (key_dim, vd) = (nk * hdk, nv * hdv);
        let mut r = |n: usize, s: f32| (0..n).map(|_| s * lcg(&mut sd)).collect::<Vec<f32>>();
        let c = r(t * (2 * key_dim + vd), 1.0);
        let z = r(t * vd, 2.0);
        let b = r(t * nv, 2.0);
        let a = r(t * nv, 2.0);
        let s0 = r(nv * hdk * hdv, 0.1);
        let decay: Vec<f32> = r(nv, 1.0).iter().map(|v| -(v.abs() + 0.1)).collect();
        let dt = r(nv, 1.0);
        let w = r(hdv, 1.0).iter().map(|v| 1.0 + 0.5 * v).collect::<Vec<f32>>();
        for (tiled, sig) in [(false, true), (true, false)] {
            let rule = DeltaRule {
                nk,
                nv,
                hdk,
                hdv,
                tiled,
                decay_coef: decay.clone(),
                dt_bias: dt.clone(),
            };
            let mut s_cpu = s0.clone();
            let mut o_cpu = vec![0f32; t * vd];
            rule.run(&mut s_cpu, &c, &b, &a, t, Some(&mut o_cpu));
            gated_norm(&mut o_cpu, &z, &w, 1e-6, sig, false);
            let mut ba = vec![0f32; t * 2 * nv];
            for (ti, row) in ba.chunks_mut(2 * nv).enumerate() {
                row[..nv].copy_from_slice(&b[ti * nv..(ti + 1) * nv]);
                row[nv..].copy_from_slice(&a[ti * nv..(ti + 1) * nv]);
            }
            let mut s_gpu = s0.clone();
            let mut o_gpu = vec![0f32; t * vd];
            assert!(be.gdn_rule(&mut crate::gpu::GdnJob {
                c: &c,
                ba: &ba,
                state: &mut s_gpu,
                out: &mut o_gpu,
                t,
                nk,
                nv,
                hdk,
                hdv,
                tiled,
                decay_coef: &decay,
                dt_bias: &dt,
            }));
            gated_norm(&mut o_gpu, &z, &w, 1e-6, sig, false);
            close(&o_gpu, &o_cpu, 1e-4);
            close(&s_gpu, &s_cpu, 1e-4);
        }
    }

    /// `Backend::attention_cached` against the CPU attention, dense and over QSA selections.
    #[test]
    fn attention_cached_matches_cpu() {
        let Ok(be) = crate::metal::MetalBackend::new() else { return };
        let (nh, nkv, hd, start, t) = (8, 2, 256, 300, 40);
        let mut sd = 9u64;
        let mut cache = LayerCache::new(nkv, hd);
        let len = start + t;
        let k: Vec<f32> = (0..len * nkv * hd).map(|_| lcg(&mut sd)).collect();
        let v: Vec<f32> = (0..len * nkv * hd).map(|_| lcg(&mut sd)).collect();
        cache.push(&k, &v, len);
        let q: Vec<f32> = (0..t * nh * hd).map(|_| 0.1 * lcg(&mut sd)).collect();
        // Every other token over a selection: a few blocks of 4 and its tail.
        let sel: Vec<Option<Vec<u32>>> = (0..t)
            .map(|i| {
                let p = (start + i) as u32;
                (i % 2 == 1).then(|| {
                    (0..p / 4)
                        .filter(|b| b % 3 == 0)
                        .flat_map(|b| 4 * b..4 * b + 4)
                        .chain(p / 4 * 4..=p)
                        .collect()
                })
            })
            .collect();
        for sel in [None, Some(&sel[..])] {
            let mut cpu = vec![0f32; t * nh * hd];
            match sel {
                Some(s) => crate::qwen4exp::attention_sel(&q, &cache, nh, t, start, s, &mut cpu),
                None => attention(&q, &cache, nh, t, start, None, &mut cpu),
            }
            let mut gpu = vec![0f32; t * nh * hd];
            let (kc, vc) = cache.data();
            assert!(be.attention_cached(&mut crate::gpu::AttnJob {
                q: &q,
                k: kc,
                v: vc,
                start,
                t,
                nh,
                nkv,
                hd,
                sel,
                out: &mut gpu,
            }));
            close(&gpu, &cpu, 1e-5);
        }
    }
}
