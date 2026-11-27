//! Gemma 4 / Gemma 3 single-token decoding on the GPU (`crate::gpu::decode`): the decoder pass
//! as a device graph, per-layer embeddings, sliding / full attention with shared KV caches,
//! sandwich norms, GELU MLPs, layer scalars and the softcapped head included.

use super::{Layer, LayerCache, Lin, Model, Session};
use crate::gpu::decode::{Ew, Graph, LinMode, Op, Step, pairable};
use std::borrow::Cow;

/// Whether prompts skip the KV-sharing layers' rows nothing reads ([`Model::gpu_prefill_pruned`];
/// `OXINFER_PF_PRUNE=0`: not).
fn pf_prune() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OXINFER_PF_PRUNE").map_or(true, |v| v != "0"))
}

impl<'a> Model<'a> {
    fn gpu_graph<'s>(&'s self, lay: Option<&crate::intervene::GpuLayout>) -> Option<Graph<'s>> {
        self.gpu_graph_with(lay, None)
    }

    /// [`Self::gpu_graph`], plus for multi-token plans (`multi`: layers whose outputs to capture)
    /// the copies of those layers' outputs.
    fn gpu_graph_with<'s>(&'s self, lay: Option<&crate::intervene::GpuLayout>, multi: Option<&[usize]>) -> Option<Graph<'s>> {
        self.gpu_graph_mode(lay, multi, false, 0..self.cfg.num_hidden_layers)
    }

    /// [`Self::gpu_graph_with`]; `prefill`: for a prefill plan (`Backend::decode_plan_prefill`):
    /// no LM head. `layers`: the layers the graph runs, all of them, or
    /// - the first ones (a prefill plan): the hidden output is the residual after the last;
    /// - the last ones: the input is that residual followed by the scaled token embedding (the
    ///   per-layer embeddings' projection reads it), and the second input the token's PLE row.
    fn gpu_graph_mode<'s>(
        &'s self,
        lay: Option<&crate::intervene::GpuLayout>,
        multi: Option<&[usize]>,
        prefill: bool,
        layers: std::ops::Range<usize>,
    ) -> Option<Graph<'s>> {
        let cfg = &self.cfg;
        if cfg.bidirectional() {
            return None;
        }
        let (from, upto) = (layers.start, layers.end);
        let part = from > 0 || upto < cfg.num_hidden_layers;
        if part && (lay.is_some() || multi.is_some_and(|m| !m.is_empty()) || from >= upto || (from > 0) == (upto < cfg.num_hidden_layers)) {
            return None;
        }
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nl = cfg.num_hidden_layers;
        let eps = cfg.rms_norm_eps as f32;
        // Streamed weights (units outside the resident set, routed experts) are wrapped by the
        // plan per step (see `crate::gpu::decode::Graph::units`).
        let streams = self.stream.is_some();
        let dev = |l: &'s Lin, n_out: usize, n_in: usize| -> Option<&'s crate::gpu::DeviceWeight> {
            if l.is_transient() {
                return streams.then(crate::gpu::decode::streamed_weight);
            }
            l.device_weight(n_out, n_in)
        };
        let off = |l: &Lin| l.is_transient() && !streams;
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
        // Linear ops and their runtime LoRA updates (after all of them: the updates of a group
        // then wait for the matvecs together).
        type Job<'s> = (
            &'s Lin<'s>,
            usize,
            usize,
            crate::gpu::decode::View,
            crate::gpu::decode::View,
            LinMode,
        );
        let put_all = |g: &mut Graph<'s>, jobs: &[Job<'s>]| -> Option<()> {
            for &(l, n_out, n_in, x, y, mode) in jobs {
                g.ops.push(lin(l, n_out, n_in, x, y, mode)?);
            }
            let mut ups = Vec::new();
            for &(l, _, _, x, y, _) in jobs {
                if let Some((down, up)) = g.lora_ops(l, x, y, None)? {
                    g.ops.push(down);
                    ups.push(up);
                }
            }
            g.ops.extend(ups);
            Some(())
        };
        let put = |g: &mut Graph<'s>, l: &'s Lin, n_out: usize, n_in: usize, x, y, mode| put_all(g, &[(l, n_out, n_in, x, y, mode)]);
        let max_hd = self.layers.iter().map(|l| l.hd).max()?;
        let max_kv = self.layers.iter().map(|l| l.n_kv).max()?;
        let max_inter = self.layers.iter().map(|l| l.inter).max()?;
        let mut g = Graph::new();
        if let Some(s) = &self.stream {
            g.gate = Some(std::sync::Arc::new(s.clone()));
        }
        // (The last layers: the residual, then the token embedding.)
        let x = g.buf(if from > 0 { 2 * h } else { h });
        let emb_in = if from > 0 { x.at(h) } else { x };
        if from > 0 {
            g.n_input = 2 * h;
        }
        // Plans of every part of the model share slots.
        g.layout_key = Some(self.final_norm.as_ptr() as usize);
        let hn = g.buf(h);
        let qkv = g.buf(nh * max_hd + 2 * max_kv * max_hd);
        let qo = g.buf(nh * max_hd);
        let att = g.buf(nh * max_hd);
        let ob = g.buf(h);
        let act = g.buf(max_inter);
        let act2 = g.buf(max_inter);
        let hidden = g.buf(h);
        let logits = g.buf(if prefill { 1 } else { cfg.vocab_size });
        g.input = x;
        g.hidden = hidden;
        g.n_hidden = h;
        if from == 0 {
            g.embed = Some((&self.embed, (h as f32).sqrt()));
            g.embed2 = self.ple.as_ref().map(|p| &p.embed);
        }
        g.rope_halves = vec![self.inv_freq_sliding.len(), self.inv_freq_full.len()];
        // Interventions (`crate::intervene`): parameter buffers and the hooked residual points.
        let (hot, cold) = (
            g.buf(lay.map_or(1, |l| l.hot_len.max(1))),
            g.buf(lay.map_or(1, |l| l.cold_len.max(1))),
        );
        if let Some(l) = lay {
            g.params = Some(((hot, l.hot_len), (cold, l.cold_len)));
        }
        let hooked = |p: usize| lay.and_then(|l| l.points.get(p).copied().flatten());
        let steer = |g: &mut Graph<'s>, p: usize| {
            if let Some(gp) = hooked(p) {
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
        // Per-layer embeddings: the projection of the input, normed per layer, plus the token's
        // row of the PLE table (the second host input).
        let p = cfg.hidden_size_per_layer_input;
        let pli = match &self.ple {
            Some(ple) => {
                let pli = g.buf(nl * p);
                let emb = g.buf(nl * p);
                g.input2 = Some((emb, nl * p));
                g.units.push((g.ops.len(), 0));
                put(&mut g, &ple.proj, nl * p, h, emb_in, pli, LinMode::Store)?;
                g.ops.push(Op::Norm {
                    x: pli,
                    out: pli,
                    w: Some(Cow::Borrowed(&ple.norm)),
                    pre: (h as f32).powf(-0.5),
                    n: p,
                    rows: nl,
                    x_stride: p,
                    out_stride: p,
                    eps,
                    add: false,
                });
                g.ops.push(Op::Ew {
                    kind: Ew::Combine {
                        sa: 1.0,
                        sb: (p as f32).sqrt(),
                        s: std::f32::consts::FRAC_1_SQRT_2,
                    },
                    a: pli,
                    b: Some(emb),
                    out: pli,
                    n: nl * p,
                });
                Some(pli)
            }
            None => None,
        };
        let gp = g.buf(p.max(1));
        // 26B-A4B's MoE blocks: the router input, the experts' input and output, router logits,
        // the chosen experts and their activations.
        let (mut n_exp, mut top_k, mut e_inter) = (0, 0, 0);
        for m in self.layers.iter().filter_map(|l| l.moe.as_ref()) {
            n_exp = n_exp.max(m.experts.n_experts());
            top_k = top_k.max(m.experts.top_k);
            e_inter = e_inter.max(m.experts.inter);
        }
        let has_moe = n_exp > 0;
        let xr = g.buf(if has_moe { h } else { 1 });
        let hn2 = g.buf(if has_moe { h } else { 1 });
        let eo = g.buf(if has_moe { h } else { 1 });
        let rlogits = g.buf(n_exp.max(1));
        let sel = g.buf((2 * top_k).max(1));
        let eact = g.buf((top_k * e_inter).max(1));
        // KV cache slots: the layers that own a cache, in layer order (as `Session::caches`).
        let mut slot = vec![usize::MAX; nl];
        for (l, layer) in self.layers.iter().enumerate() {
            if layer.own_kv {
                slot[l] = g.kv.len();
                g.kv.push((layer.n_kv, layer.hd));
            }
        }
        let qscale = cfg.query_scale();
        // Multi-token plans: the residual after the captured layers, copied out.
        let cap = multi.map(|m| g.buf((m.len() * h).max(1)));
        if let (Some(c), Some(m)) = (cap, multi) {
            g.capture = Some((c, m.len() * h));
        }
        // The norm after each residual step is fused into it (`Op::AddNorm`), across layers too:
        // the input norm of a layer comes with the last step of the one before.
        let mut have_hn = false;
        for (l, layer) in self.layers.iter().enumerate() {
            if l < from || l >= upto {
                continue;
            }
            let Layer { hd, n_kv, .. } = *layer;
            g.units.push((g.ops.len(), self.layer_unit(l)));
            let kv = *slot.get(layer.kv_src).filter(|&&s| s != usize::MAX)?;
            // What follows this layer's last residual step: the next layer's input norm, or the
            // final norm into `hidden`.
            let next_norm: (crate::gpu::decode::View, &'s [f32]) = match self.layers.get(l + 1) {
                Some(nx) => (hn, &nx.ln_in),
                None => (hidden, &self.final_norm),
            };
            if l == 0 {
                steer(&mut g, 0);
            }
            if !have_hn {
                g.ops.push(Op::norm(x, hn, &layer.ln_in, h, eps));
            }
            let (dense_edit, expert_edit) = lay.map_or((None, None), |y| y.neurons[l]);
            let mut qkv_jobs: Vec<Job<'s>> = vec![(&layer.q, nh * hd, h, hn, qkv, LinMode::Store)];
            if layer.own_kv {
                qkv_jobs.push((layer.k.as_ref()?, n_kv * hd, h, hn, qkv.at(nh * hd), LinMode::Store));
                if !layer.k_eq_v {
                    qkv_jobs.push((layer.v.as_ref()?, n_kv * hd, h, hn, qkv.at((nh + n_kv) * hd), LinMode::Store));
                }
            }
            put_all(&mut g, &qkv_jobs)?;
            g.ops.push(Op::QkRope {
                qkv,
                q_out: qo,
                kv,
                nh,
                // Layers reading another layer's cache write no keys or values.
                nkv: if layer.own_kv { n_kv } else { 0 },
                hd,
                half_rot: hd / 2,
                q_stride: hd,
                k_off: nh * hd,
                v_off: (nh + n_kv) * hd,
                interleaved: false,
                q_norm: Some(&layer.q_norm),
                k_norm: layer.k_norm.as_deref(),
                eps,
                rope: if layer.sliding { 0 } else { 1 },
                v_norm: !cfg.gemma3,
                v_from_k: layer.k_eq_v,
            });
            if self.inv_freq_sliding.len() * 2 != hd && layer.sliding || !layer.sliding && self.inv_freq_full.len() * 2 != hd {
                return None;
            }
            g.ops.push(Op::Attend {
                q: qo,
                out: att,
                kv,
                nh,
                nkv: n_kv,
                hd,
                scale: qscale,
                window: layer.sliding.then_some(cfg.sliding_window),
                gate: None,
                block: None,
            });
            put(&mut g, &layer.o, h, nh * hd, att, ob, LinMode::Store)?;
            // A hooked point splits the fused residual step: add, hook, then the norm.
            let mid = hooked(2 * l + 1).is_some();
            g.ops.push(Op::AddNorm {
                y: ob,
                x,
                wy: Cow::Borrowed(&layer.ln_post_attn),
                scale: 1.0,
                next: (!mid).then_some((hn, &layer.ln_pre_ffn[..])),
                n: h,
                eps,
            });
            if mid {
                steer(&mut g, 2 * l + 1);
                g.ops.push(Op::norm(x, hn, &layer.ln_pre_ffn, h, eps));
            }
            let inter = layer.inter;
            let (gw, uw) = (layer.gate.as_ref(), layer.up.as_ref());
            if pairable(&gw, &uw) && !layer.gate.has_lora() && !layer.up.has_lora() {
                g.ops.push(Op::Linear {
                    w: gw,
                    dev: dev(&layer.gate, inter, h)?,
                    w2: Some((uw, dev(&layer.up, inter, h)?)),
                    n_out: inter,
                    n_in: h,
                    x: hn,
                    y: act,
                    mode: LinMode::GeluPair,
                });
            } else {
                put_all(
                    &mut g,
                    &[
                        (&layer.gate, inter, h, hn, act2, LinMode::Store),
                        (&layer.up, inter, h, hn, act, LinMode::Store),
                    ],
                )?;
                g.ops.push(Op::Glu {
                    g: act2,
                    u: act,
                    out: act,
                    n: inter,
                    gelu: true,
                });
            }
            if let Some(e) = dense_edit {
                g.ops.push(Op::NeuronEdit {
                    act,
                    hot: hot.at(e.hot),
                    n: e.n,
                });
            }
            let down_at = g.ops.len();
            put(&mut g, &layer.down, h, inter, act, ob, LinMode::Store)?;
            if let Some(m) = &layer.moe {
                // Routed experts beside the dense MLP, on the same residual x:
                // ob = norm(ob, post_dense) + norm(experts(norm(x, pre)), post).
                if m.experts.act != crate::moe::Act::GeluTanh {
                    return None;
                }
                let ne = m.experts.n_experts();
                let one = |x, out, w: Option<&'s [f32]>, add| Op::Norm {
                    x,
                    out,
                    w: w.map(Cow::Borrowed),
                    pre: 1.0,
                    n: h,
                    rows: 1,
                    x_stride: h,
                    out_stride: h,
                    eps,
                    add,
                };
                let route = Op::Route {
                    x: xr,
                    router: &m.router,
                    gate: None,
                    n_exp: ne,
                    h,
                    k: m.experts.top_k,
                    logits: rlogits,
                    sel,
                    expert_scale: Some(&m.per_expert_scale),
                    sigmoid: None,
                };
                if prefill || std::env::var("OXINFER_DEC_MOE_ORDER").is_ok_and(|v| v == "0") {
                    g.ops.push(one(ob, ob, Some(&m.ln_post_dense), false));
                    g.ops.push(one(x, xr, None, false));
                    g.ops.push(one(x, hn2, Some(&m.ln_pre), false));
                    g.ops.push(route);
                } else {
                    // Decode plans: the routing beside the dense MLP (the router's and the
                    // experts' input norms in its gate / up pair's stage, the top-k in its down
                    // projection's, the experts' pairs in its post norm's), the same ops on the
                    // same inputs, two stages fewer (`OXINFER_DEC_MOE_ORDER=0`: the order of the
                    // prefill graph).
                    let down: Vec<Op<'s>> = g.ops.drain(down_at..).collect();
                    g.ops.push(one(x, xr, None, false));
                    g.ops.push(one(x, hn2, Some(&m.ln_pre), false));
                    g.ops.push(route);
                    g.ops.extend(down);
                    g.ops.push(one(ob, ob, Some(&m.ln_post_dense), false));
                }
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
                if m.experts.down.iter().any(off) {
                    return None;
                }
                let lora = g.expert_lora(&m.experts, m.experts.top_k)?;
                g.ops.push(Op::Experts(crate::gpu::decode::Experts {
                    gate,
                    up: (!up.is_empty()).then_some(up),
                    down: m.experts.down.iter().map(|l| l.as_ref()).collect(),
                    h,
                    inter: m.experts.inter,
                    k: m.experts.top_k,
                    x: hn2,
                    sel,
                    act: eact,
                    y: eo,
                    gelu: true,
                    relu2: false,
                    add: false,
                    edits: expert_edit.map(|e| (hot.at(e.hot), e.n)),
                    lora,
                }));
                g.ops.push(one(eo, ob, Some(&m.ln_post), true));
            }
            let ple = layer.ple.as_ref().zip(pli);
            let post = hooked(2 * l + 2).is_some();
            let last_next = (!post).then_some(next_norm);
            g.ops.push(Op::AddNorm {
                y: ob,
                x,
                wy: Cow::Borrowed(&layer.ln_post_ffn),
                scale: if ple.is_some() { 1.0 } else { layer.layer_scalar },
                next: if ple.is_some() { None } else { last_next },
                n: h,
                eps,
            });
            if let Some((pw, pli)) = ple {
                put(&mut g, &pw.gate, p, h, x, gp, LinMode::Store)?;
                g.ops.push(Op::Glu {
                    g: gp,
                    u: pli.at(l * p),
                    out: gp,
                    n: p,
                    gelu: true,
                });
                put(&mut g, &pw.proj, h, p, gp, ob, LinMode::Store)?;
                g.ops.push(Op::AddNorm {
                    y: ob,
                    x,
                    wy: Cow::Borrowed(&pw.norm),
                    scale: layer.layer_scalar,
                    next: last_next,
                    n: h,
                    eps,
                });
            }
            if post {
                steer(&mut g, 2 * l + 2);
                g.ops.push(Op::norm(x, next_norm.0, next_norm.1, h, eps));
            }
            if let (Some(c), Some(m)) = (cap, multi) {
                for (i, _) in m.iter().enumerate().filter(|e| *e.1 == l) {
                    g.ops.push(Op::Ew {
                        kind: Ew::Scale(1.0),
                        a: x,
                        b: None,
                        out: c.at(i * h),
                        n: h,
                    });
                }
            }
            have_hn = true;
        }
        if upto < nl {
            // The first layers: the residual out (the next layer's input norm is left unread).
            g.hidden = x;
            g.n_input = h;
            g.head_from = g.ops.len();
            return Some(g);
        }
        g.head_from = g.ops.len();
        g.units.push((g.ops.len(), self.layer_unit(nl)));
        if prefill {
            return Some(g);
        }
        put(&mut g, &self.head, cfg.vocab_size, h, hidden, logits, LinMode::Store)?;
        if let Some(cap) = cfg.final_logit_softcapping {
            g.ops.push(Op::Ew {
                kind: Ew::Softcap(cap as f32),
                a: logits,
                b: None,
                out: logits,
                n: cfg.vocab_size,
            });
        }
        g.logits = Some((logits, cfg.vocab_size));
        Some(g)
    }

    /// The prefill chunk to use instead of `chunk`: passes on the GPU prefill plan take chunks of
    /// the plan's size when given GPU-sized ones (1024 and up).
    pub fn prefill_chunk(&self, chunk: usize) -> usize {
        let plan = self.stream.is_none()
            && chunk >= 1024
            && !self.cfg.bidirectional()
            && self.gpu_decode_on()
            && crate::gpu::active().is_some()
            && crate::qwen::prefill_gpu(crate::qwen::PREFILL_GPU_MIN, "plan");
        if plan {
            chunk.max(crate::qwen::prefill_plan_tokens())
        } else {
            chunk
        }
    }

    /// Whether decode steps may run on the device: GPU decoding is on, and streamed weights (if
    /// any) may be decoded there ([`crate::gpu::stream_decode_enabled`]).
    fn gpu_decode_on(&self) -> bool {
        crate::gpu::decode_enabled() && (self.stream.is_none() || crate::gpu::stream_decode_enabled())
    }

    /// Streamed with on-demand experts (26B-A4B) and GPU decoding left to decide: plain decode
    /// steps are measured on both and the faster kept ([`crate::gpu::decode::DeviceChoice`]).
    pub(super) fn choosing_device(&self) -> bool {
        self.stream.is_some()
            && crate::gpu::stream_decode_forced().is_none()
            && self.gpu_decode_on()
            && self.layers.iter().any(|l| l.moe.is_some())
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
            let g = self.gpu_graph(lay)?;
            be.decode_plan(&g)
        })
    }

    /// The multi-token GPU plan capturing the outputs of `layers` (sorted), compiled on first use
    /// (`fast`: on the batched fast-verification kernels, see
    /// [`crate::gpu::Backend::decode_plan_multi_fast`]).
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

    /// The GPU prefill plan (see `Backend::decode_plan_prefill`), compiled on first use; None
    /// when unavailable.
    ///
    /// The plan copies out the residual after each of `cap` (layer indices, sorted): the
    /// features a speculative draft reads ([`crate::dspark::Target::forward_feats`]).
    fn gpu_prefill_plan(&self, cap: &[usize]) -> Option<std::sync::Arc<dyn crate::gpu::decode::DecodePlan>> {
        if !self.gpu_decode_on() {
            return None;
        }
        let tokens = crate::qwen::prefill_plan_tokens();
        let key = cap.iter().fold((1u64 << 61) | tokens as u64, |k, &l| {
            k.wrapping_mul(0x100000001b3).wrapping_add(l as u64 + 1)
        });
        self.gpu_plans.get(key, || {
            let be = crate::gpu::active()?;
            let plan = self
                .gpu_graph_mode(None, Some(cap), true, 0..self.cfg.num_hidden_layers)
                .and_then(|g| be.decode_plan_prefill(&g, tokens));
            if std::env::var_os("OXINFER_GPU_PLAN_DEBUG").is_some() {
                eprintln!("[gpu prefill plan: {}]", if plan.is_some() { "built" } else { "none" });
            }
            plan
        })
    }

    /// A prefill pass of `tokens` on the GPU prefill plan (chunks of the plan's size, the
    /// activations on the device for the whole pass): the final-normed hidden states, as the
    /// host path gives them (up to rounding). None (nothing changed) when unavailable; a failure
    /// past the first chunk finishes the pass on the host path.
    ///
    /// With `cap` (layer indices, sorted), `captured` receives the residual after each of them,
    /// `[t, cap.len(), hidden]`.
    pub(super) fn gpu_prefill(&self, sess: &mut Session, tokens: &[u32], cap: &[usize], captured: &mut Vec<f32>) -> Option<Vec<f32>> {
        let t = tokens.len();
        if sess.iv.active() || !sess.capture.is_empty() || self.stream.is_some() || self.cfg.bidirectional() {
            return None;
        }
        let plan = self.gpu_prefill_plan(cap)?;
        let n = plan.tokens();
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let pw = cfg.num_hidden_layers * cfg.hidden_size_per_layer_input;
        let nc = cap.len();
        let mut hidden = vec![0f32; t * h];
        // The prompt's keys and values come down chunk by chunk: room for all of them at once.
        for c in sess.caches.iter_mut().flatten() {
            c.reserve(t);
        }
        captured.clear();
        captured.resize(t * nc * h, 0.0);
        let mut c = 0;
        while c < t {
            let m = n.min(t - c);
            let toks = &tokens[c..c + m];
            let mut x = Vec::with_capacity(m * h);
            let mut x2 = Vec::new();
            for &tok in toks {
                x.extend_from_slice(&self.embed_scaled(tok));
                if let Some(ple) = &self.ple {
                    let o = x2.len();
                    x2.resize(o + pw, 0.0);
                    ple.embed.row(tok as usize, &mut x2[o..]);
                }
            }
            let positions: Vec<usize> = (sess.pos..sess.pos + m).collect();
            let mut cs = Vec::new();
            for f in [&self.inv_freq_sliding, &self.inv_freq_full] {
                let half = f.len();
                let (co, si) = crate::kernels::rope_tables(&positions, f);
                for j in 0..m {
                    cs.extend_from_slice(&co[j * half..(j + 1) * half]);
                    cs.extend_from_slice(&si[j * half..(j + 1) * half]);
                }
            }
            let ok = {
                let _timer = crate::prof::start(crate::prof::Cat::Matmul);
                let mut kv: Vec<&mut LayerCache> = sess.caches.iter_mut().flatten().collect();
                plan.step(
                    &mut sess.gpu,
                    Step {
                        pos: sess.pos,
                        t: m,
                        rope_cs: &cs,
                        x: &x,
                        x2: &x2,
                        kv: &mut kv,
                        states: &mut [],
                        hidden: &mut hidden[c * h..(c + m) * h],
                        logits: None,
                        params: None,
                        capture: (nc > 0).then(|| &mut captured[c * nc * h..(c + m) * nc * h]),
                        scratch: false,
                        states_kept: false,
                    },
                )
            };
            if !ok {
                break;
            }
            sess.pos += m;
            c += m;
        }
        if c == 0 {
            return None;
        }
        if c < t {
            sess.capture = cap.to_vec();
            sess.captured.clear();
            let hs = self.forward_hidden_cpu(sess, &tokens[c..], &[]);
            sess.capture.clear();
            for (ci, &l) in cap.iter().enumerate() {
                let rows = &sess.captured.iter().find(|r| r.0 == l).expect("captured layer").1;
                for (j, row) in rows.chunks(h).enumerate() {
                    captured[((c + j) * nc + ci) * h..((c + j) * nc + ci + 1) * h].copy_from_slice(row);
                }
            }
            sess.captured.clear();
            hidden.truncate(c * h);
            hidden.extend_from_slice(&hs);
        }
        Some(hidden)
    }

    /// The plans of a pruned prefill ([`Model::gpu_prefill_pruned`]): the prefill plan of the
    /// layers before `split` and the one-token plan of the others.
    #[allow(clippy::type_complexity)]
    fn gpu_split_plans(
        &self,
        split: usize,
    ) -> Option<(
        std::sync::Arc<dyn crate::gpu::decode::DecodePlan>,
        std::sync::Arc<dyn crate::gpu::decode::DecodePlan>,
    )> {
        if !self.gpu_decode_on() {
            return None;
        }
        let tokens = crate::qwen::prefill_plan_tokens();
        let nl = self.cfg.num_hidden_layers;
        let first = self.gpu_plans.get((1u64 << 60) | ((split as u64) << 32) | tokens as u64, || {
            let be = crate::gpu::active()?;
            be.decode_plan_prefill(&self.gpu_graph_mode(None, Some(&[]), true, 0..split)?, tokens)
        })?;
        let rest = self.gpu_plans.get((1u64 << 59) | split as u64, || {
            let be = crate::gpu::active()?;
            be.decode_plan(&self.gpu_graph_mode(None, None, false, split..nl)?)
        })?;
        Some((first, rest))
    }

    /// [`Model::prefill_hidden`] on the GPU for models with KV-sharing layers: the layers before
    /// the first of those (the ones that fill the caches) on every token of the prompt, on the
    /// prefill plan, then the rest on its last token alone when `last` (that token's hidden
    /// state), as a one-token step reading the caches the first part filled. None (nothing
    /// changed) when unavailable.
    pub(super) fn gpu_prefill_pruned(&self, sess: &mut Session, tokens: &[u32], last: bool) -> Option<Vec<f32>> {
        let cfg = &self.cfg;
        let (t, nl, h) = (tokens.len(), cfg.num_hidden_layers, cfg.hidden_size);
        let split = cfg.first_kv_shared_layer();
        if cfg.num_kv_shared_layers == 0
            || split == 0
            || split >= nl
            || !pf_prune()
            || sess.iv.active()
            || !sess.capture.is_empty()
            || self.stream.is_some()
            || cfg.bidirectional()
        {
            return None;
        }
        let (first, rest) = self.gpu_split_plans(split)?;
        let n = first.tokens();
        let pw = nl * cfg.hidden_size_per_layer_input;
        for c in sess.caches.iter_mut().flatten() {
            c.reserve(t);
        }
        // The residual after the first part, of each chunk's tokens.
        let mut res = vec![0f32; n.min(t) * h];
        let mut c = 0;
        while c < t {
            let m = n.min(t - c);
            let toks = &tokens[c..c + m];
            let mut x = Vec::with_capacity(m * h);
            let mut x2 = Vec::new();
            for &tok in toks {
                x.extend_from_slice(&self.embed_scaled(tok));
                if let Some(ple) = &self.ple {
                    let o = x2.len();
                    x2.resize(o + pw, 0.0);
                    ple.embed.row(tok as usize, &mut x2[o..]);
                }
            }
            let cs = self.rope_rows(sess.pos, m);
            let ok = {
                let _timer = crate::prof::start(crate::prof::Cat::Matmul);
                let mut kv: Vec<&mut LayerCache> = sess.caches.iter_mut().flatten().collect();
                first.step(
                    &mut sess.gpu,
                    Step {
                        pos: sess.pos,
                        t: m,
                        rope_cs: &cs,
                        x: &x,
                        x2: &x2,
                        kv: &mut kv,
                        states: &mut [],
                        hidden: &mut res[..m * h],
                        logits: None,
                        params: None,
                        capture: None,
                        scratch: false,
                        states_kept: false,
                    },
                )
            };
            if !ok {
                break;
            }
            sess.pos += m;
            c += m;
        }
        if c == 0 {
            return None;
        }
        if c < t {
            // The rest of the prompt on the host path, through every layer.
            let hs = self.forward_hidden_cpu(sess, &tokens[c..], &[]);
            return Some(if last { hs[hs.len() - h..].to_vec() } else { Vec::new() });
        }
        if !last {
            return Some(Vec::new());
        }
        // The last token through the KV-sharing layers: a one-token step at its position, whose
        // keys and values the first part wrote (the host caches give that row back to the
        // step, which copies it down again).
        let m = t - (t - 1) / n * n;
        let tok = tokens[t - 1];
        let p = sess.pos - 1;
        let mut x = res[(m - 1) * h..m * h].to_vec();
        x.extend_from_slice(&self.embed_scaled(tok));
        let mut x2 = Vec::new();
        if let Some(ple) = &self.ple {
            x2.resize(pw, 0.0);
            ple.embed.row(tok as usize, &mut x2);
        }
        let cs = self.rope_rows(p, 1);
        let mut hidden = vec![0f32; h];
        sess.truncate(p);
        let ok = {
            let _timer = crate::prof::start(crate::prof::Cat::Matmul);
            let mut kv: Vec<&mut LayerCache> = sess.caches.iter_mut().flatten().collect();
            rest.step(
                &mut sess.gpu,
                Step {
                    pos: p,
                    t: 1,
                    rope_cs: &cs,
                    x: &x,
                    x2: &x2,
                    kv: &mut kv,
                    states: &mut [],
                    hidden: &mut hidden,
                    logits: None,
                    params: None,
                    capture: None,
                    scratch: false,
                    states_kept: false,
                },
            )
        };
        if !ok {
            return Some(self.forward_hidden_cpu(sess, &[tok], &[]));
        }
        sess.pos = p + 1;
        Some(hidden)
    }

    /// RoPE cos / sin rows of `m` positions from `pos`, per table (sliding, full), as the GPU
    /// plans read them.
    fn rope_rows(&self, pos: usize, m: usize) -> Vec<f32> {
        let positions: Vec<usize> = (pos..pos + m).collect();
        let mut cs = Vec::new();
        for f in [&self.inv_freq_sliding, &self.inv_freq_full] {
            let half = f.len();
            let (co, si) = crate::kernels::rope_tables(&positions, f);
            for j in 0..m {
                cs.extend_from_slice(&co[j * half..(j + 1) * half]);
                cs.extend_from_slice(&si[j * half..(j + 1) * half]);
            }
        }
        cs
    }

    /// Runs `tokens` (at most 16) through a multi-token GPU plan, as one-token GPU steps would one
    /// after the other: the (softcapped) logits of every token and, in `feats`, the residual
    /// stream entering each of `layers` (as [`crate::dspark::Target::forward_feats`]). None
    /// (nothing changed) when GPU decoding is off or the model cannot.
    pub(super) fn gpu_verify(
        &self,
        sess: &mut Session,
        tokens: &[u32],
        layers: &[usize],
        feats: &mut Vec<f32>,
        fast: bool,
    ) -> Option<Vec<f32>> {
        self.gpu_verify_hidden(sess, tokens, layers, feats, fast).map(|r| r.1)
    }

    /// [`Model::gpu_verify`] that also returns the final-normed hidden states `[t, hidden]`.
    pub(super) fn gpu_verify_hidden(
        &self,
        sess: &mut Session,
        tokens: &[u32],
        layers: &[usize],
        feats: &mut Vec<f32>,
        fast: bool,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let t = tokens.len();
        if t == 0 || t > 16 || sess.iv.active() || layers.contains(&0) || self.cfg.bidirectional() {
            return None;
        }
        let mut cap: Vec<usize> = layers.iter().map(|&l| l - 1).collect();
        cap.sort_unstable();
        cap.dedup();
        // The smallest plan holding `t`: one token (a round without drafts) runs the single-token
        // kernels, as plain decoding does.
        let plan = self.gpu_multi_plan(
            &cap,
            if t == 1 {
                1
            } else if t <= 8 {
                8
            } else {
                16
            },
            fast,
        )?;
        let _timer = crate::prof::start(crate::prof::Cat::Matmul);
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let mut x = Vec::with_capacity(t * h);
        let mut x2 = Vec::new();
        for &tok in tokens {
            x.extend_from_slice(&self.embed_scaled(tok));
            if let Some(ple) = &self.ple {
                let mut e = vec![0f32; cfg.num_hidden_layers * cfg.hidden_size_per_layer_input];
                ple.embed.row(tok as usize, &mut e);
                x2.extend_from_slice(&e);
            }
        }
        let positions: Vec<usize> = (sess.pos..sess.pos + t).collect();
        let mut cs = Vec::new();
        for f in [&self.inv_freq_sliding, &self.inv_freq_full] {
            let half = f.len();
            let (c, s) = crate::kernels::rope_tables(&positions, f);
            for j in 0..t {
                cs.extend_from_slice(&c[j * half..(j + 1) * half]);
                cs.extend_from_slice(&s[j * half..(j + 1) * half]);
            }
        }
        let mut hidden = vec![0f32; t * h];
        let mut logits = vec![0f32; t * cfg.vocab_size];
        let mut captured = vec![0f32; t * cap.len() * h];
        let mut kv: Vec<&mut LayerCache> = sess.caches.iter_mut().flatten().collect();
        let ok = plan.step(
            &mut sess.gpu,
            Step {
                pos: sess.pos,
                t,
                rope_cs: &cs,
                x: &x,
                x2: &x2,
                kv: &mut kv,
                states: &mut [],
                hidden: &mut hidden,
                logits: Some(&mut logits),
                params: None,
                capture: Some(&mut captured),
                scratch: false,
                states_kept: false,
            },
        );
        if !ok {
            return None;
        }
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

    /// Greedy decoding chained on the device ([`crate::generate::TextLm::decode_greedy`]; see
    /// `QwenModel::decode_greedy`): exactly the tokens and caches of one-token GPU steps with the
    /// host's argmax. None (the session untouched) where plain steps would not run on the plan's
    /// one-token path, or with the chain off ([`crate::gpu::set_decode_chain`]).
    pub fn decode_greedy(
        &self,
        sess: &mut Session,
        first: u32,
        max: usize,
        on: &mut dyn FnMut(u32) -> crate::generate::Chain,
    ) -> Option<usize> {
        if !crate::gpu::decode_chain_enabled()
            || !self.gpu_decode_on()
            || crate::gpu::decode::check()
            || self.stream.is_some()
            || sess.iv.active()
        {
            return None;
        }
        let plan = self.gpu_plan()?;
        let rope = |pos: usize| {
            let mut cs = Vec::new();
            for f in [&self.inv_freq_sliding, &self.inv_freq_full] {
                let (c, s) = crate::kernels::rope_tables(&[pos], f);
                cs.extend_from_slice(&c);
                cs.extend_from_slice(&s);
            }
            cs
        };
        let mut kv: Vec<&mut LayerCache> = sess.caches.iter_mut().flatten().collect();
        let _timer = crate::prof::start(crate::prof::Cat::Matmul);
        let n = plan.greedy(
            &mut sess.gpu,
            crate::gpu::decode::Greedy {
                pos: sess.pos,
                first,
                max,
                rope_cs: &rope,
                kv: &mut kv,
                states: &mut [],
                states_kept: false,
                on,
            },
        )?;
        sess.pos += n;
        Some(n)
    }

    /// Runs token `tok` through the GPU decode plan: the final-normed hidden state and, if
    /// wanted, the (softcapped) logits. None (nothing changed) when GPU decoding is off or
    /// unavailable for this model.
    pub(super) fn gpu_step(&self, sess: &mut Session, tok: u32, want_logits: bool) -> Option<(Vec<f32>, Option<Vec<f32>>)> {
        if !self.gpu_decode_on() {
            return None;
        }
        if !sess.iv.gpu_ok() {
            return None;
        }
        if self.choosing_device() && !self.stream_choice.want_gpu() {
            return None;
        }
        let ivc = sess.iv.iv.clone();
        let data = ivc.as_ref().and_then(|c| c.gpu_data());
        let plan = self.gpu_plan_for(data.map(|d| &d.layout))?;
        let hot = sess.iv.gpu_hot(sess.pos);
        let _timer = crate::prof::start(crate::prof::Cat::Matmul);
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let x = self.embed_scaled(tok);
        let x2 = match &self.ple {
            Some(ple) => {
                let mut e = vec![0f32; cfg.num_hidden_layers * cfg.hidden_size_per_layer_input];
                ple.embed.row(tok as usize, &mut e);
                e
            }
            None => Vec::new(),
        };
        let mut cs = Vec::new();
        for f in [&self.inv_freq_sliding, &self.inv_freq_full] {
            let (c, s) = crate::kernels::rope_tables(&[sess.pos], f);
            cs.extend_from_slice(&c);
            cs.extend_from_slice(&s);
        }
        let mut hidden = vec![0f32; h];
        let mut logits = want_logits.then(|| vec![0f32; cfg.vocab_size]);
        let mut kv: Vec<&mut LayerCache> = sess.caches.iter_mut().flatten().collect();
        let ok = plan.step(
            &mut sess.gpu,
            Step {
                pos: sess.pos,
                t: 1,
                rope_cs: &cs,
                x: &x,
                x2: &x2,
                kv: &mut kv,
                states: &mut [],
                hidden: &mut hidden,
                logits: logits.as_deref_mut(),
                params: data.zip(ivc.as_ref()).map(|(d, c)| (&hot[..], &d.cold[..], c.id())),
                capture: None,
                scratch: false,
                states_kept: false,
            },
        );
        if !ok {
            return None;
        }
        sess.pos += 1;
        Some((hidden, logits))
    }
}

impl Model<'_> {
    /// [`crate::generate::TextLm::forward_batch`]: one decode step of several sessions. On the
    /// GPU, one multi-token plan step whose rows are the sessions' tokens (the exact
    /// multi-token kernels, each session's attention on its own caches,
    /// [`crate::gpu::decode::DecodePlan::step_rows`]); otherwise (the CPU, interventions,
    /// streamed weights) the sessions one after the other.
    pub fn forward_batch(&self, sess: &mut [&mut Session], tokens: &[u32]) -> Vec<Vec<f32>> {
        let b = tokens.len();
        assert!(b > 0 && sess.len() == b, "one token per session");
        if b > 1
            && let Some(l) = self.gpu_step_rows(sess, tokens)
        {
            return l;
        }
        sess.iter_mut().zip(tokens).map(|(s, &t)| self.forward(s, &[t])).collect()
    }

    fn gpu_step_rows(&self, sess: &mut [&mut Session], tokens: &[u32]) -> Option<Vec<Vec<f32>>> {
        let b = tokens.len();
        if b > 8
            || !self.gpu_decode_on()
            || self.stream.is_some()
            || crate::gpu::decode::check()
            || self.cfg.bidirectional()
            || sess.iter().any(|s| s.iv.active() || !s.capture.is_empty())
        {
            return None;
        }
        let plan = self.gpu_multi_plan(&[], 8, crate::batch::fast_for(b))?;
        let _timer = crate::prof::start(crate::prof::Cat::Matmul);
        let cfg = &self.cfg;
        let (h, v) = (cfg.hidden_size, cfg.vocab_size);
        let mut x = Vec::with_capacity(b * h);
        let mut x2: Vec<Vec<f32>> = Vec::with_capacity(b);
        for &tok in tokens {
            x.extend_from_slice(&self.embed_scaled(tok));
            x2.push(match &self.ple {
                Some(ple) => {
                    let mut e = vec![0f32; cfg.num_hidden_layers * cfg.hidden_size_per_layer_input];
                    ple.embed.row(tok as usize, &mut e);
                    e
                }
                None => Vec::new(),
            });
        }
        let cs: Vec<Vec<f32>> = sess
            .iter()
            .map(|s| {
                let mut cs = Vec::new();
                for f in [&self.inv_freq_sliding, &self.inv_freq_full] {
                    let (c, sn) = crate::kernels::rope_tables(&[s.pos], f);
                    cs.extend_from_slice(&c);
                    cs.extend_from_slice(&sn);
                }
                cs
            })
            .collect();
        let mut hidden = vec![0f32; b * h];
        let mut logits = vec![0f32; b * v];
        let mut parts: Vec<(&mut crate::gpu::decode::Slot, Vec<&mut LayerCache>, usize)> = sess
            .iter_mut()
            .map(|s| {
                let Session { caches, gpu, pos, .. } = &mut **s;
                (gpu, caches.iter_mut().flatten().collect(), *pos)
            })
            .collect();
        let mut rows: Vec<crate::gpu::decode::Row> = parts
            .iter_mut()
            .zip(&cs)
            .zip(x.chunks(h).zip(&x2))
            .map(|(((slot, kv, pos), cs), (x, x2))| crate::gpu::decode::Row {
                slot,
                pos: *pos,
                rope_cs: cs,
                x,
                x2,
                kv: &mut kv[..],
                states: &mut [],
            })
            .collect();
        let ok = plan.step_rows(&mut rows, &mut hidden, Some(&mut logits));
        drop(rows);
        drop(parts);
        if !ok {
            return None;
        }
        for s in sess.iter_mut() {
            s.pos += 1;
        }
        Some(logits.chunks(v).map(<[f32]>::to_vec).collect())
    }
}
