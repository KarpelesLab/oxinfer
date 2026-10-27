//! Gemma 4 / Gemma 3 single-token decoding on the GPU (`crate::gpu::decode`): the decoder pass
//! as a device graph, per-layer embeddings, sliding / full attention with shared KV caches,
//! sandwich norms, GELU MLPs, layer scalars and the softcapped head included.

use super::{Layer, LayerCache, Lin, Model, Session};
use crate::gpu::decode::{Ew, Graph, LinMode, Op, Step, pairable};
use std::borrow::Cow;

impl<'a> Model<'a> {
    fn gpu_graph<'s>(&'s self) -> Option<Graph<'s>> {
        let cfg = &self.cfg;
        if cfg.bidirectional() {
            return None;
        }
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nl = cfg.num_hidden_layers;
        let eps = cfg.rms_norm_eps as f32;
        let dev = |l: &'s Lin, n_out: usize, n_in: usize| -> Option<&'s crate::gpu::DeviceWeight> {
            if l.has_lora() || l.is_transient() {
                return None;
            }
            l.device_weight(n_out, n_in)
        };
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
        let max_hd = self.layers.iter().map(|l| l.hd).max()?;
        let max_kv = self.layers.iter().map(|l| l.n_kv).max()?;
        let max_inter = self.layers.iter().map(|l| l.inter).max()?;
        let mut g = Graph::new();
        let x = g.buf(h);
        let hn = g.buf(h);
        let qkv = g.buf(nh * max_hd + 2 * max_kv * max_hd);
        let qo = g.buf(nh * max_hd);
        let att = g.buf(nh * max_hd);
        let ob = g.buf(h);
        let act = g.buf(max_inter);
        let act2 = g.buf(max_inter);
        let hidden = g.buf(h);
        let logits = g.buf(cfg.vocab_size);
        g.input = x;
        g.hidden = hidden;
        g.n_hidden = h;
        g.rope_halves = vec![self.inv_freq_sliding.len(), self.inv_freq_full.len()];
        // Per-layer embeddings: the projection of the input, normed per layer, plus the token's
        // row of the PLE table (the second host input).
        let p = cfg.hidden_size_per_layer_input;
        let pli = match &self.ple {
            Some(ple) => {
                let pli = g.buf(nl * p);
                let emb = g.buf(nl * p);
                g.input2 = Some((emb, nl * p));
                g.ops.push(lin(&ple.proj, nl * p, h, x, pli, LinMode::Store)?);
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
        // The norm after each residual step is fused into it (`Op::AddNorm`), across layers too:
        // the input norm of a layer comes with the last step of the one before.
        let mut have_hn = false;
        for (l, layer) in self.layers.iter().enumerate() {
            let Layer { hd, n_kv, .. } = *layer;
            let kv = *slot.get(layer.kv_src).filter(|&&s| s != usize::MAX)?;
            // What follows this layer's last residual step: the next layer's input norm, or the
            // final norm into `hidden`.
            let next_norm: (crate::gpu::decode::View, &'s [f32]) = match self.layers.get(l + 1) {
                Some(nx) => (hn, &nx.ln_in),
                None => (hidden, &self.final_norm),
            };
            if !have_hn {
                g.ops.push(Op::norm(x, hn, &layer.ln_in, h, eps));
            }
            g.ops.push(lin(&layer.q, nh * hd, h, hn, qkv, LinMode::Store)?);
            if layer.own_kv {
                g.ops
                    .push(lin(layer.k.as_ref()?, n_kv * hd, h, hn, qkv.at(nh * hd), LinMode::Store)?);
                if !layer.k_eq_v {
                    g.ops
                        .push(lin(layer.v.as_ref()?, n_kv * hd, h, hn, qkv.at((nh + n_kv) * hd), LinMode::Store)?);
                }
            }
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
            });
            g.ops.push(lin(&layer.o, h, nh * hd, att, ob, LinMode::Store)?);
            g.ops.push(Op::AddNorm {
                y: ob,
                x,
                wy: Cow::Borrowed(&layer.ln_post_attn),
                scale: 1.0,
                next: Some((hn, &layer.ln_pre_ffn)),
                n: h,
                eps,
            });
            let inter = layer.inter;
            let (gw, uw) = (layer.gate.as_ref(), layer.up.as_ref());
            if pairable(&gw, &uw) {
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
                g.ops.push(lin(&layer.gate, inter, h, hn, act2, LinMode::Store)?);
                g.ops.push(lin(&layer.up, inter, h, hn, act, LinMode::Store)?);
                g.ops.push(Op::Glu {
                    g: act2,
                    u: act,
                    out: act,
                    n: inter,
                    gelu: true,
                });
            }
            g.ops.push(lin(&layer.down, h, inter, act, ob, LinMode::Store)?);
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
                g.ops.push(one(ob, ob, Some(&m.ln_post_dense), false));
                g.ops.push(one(x, xr, None, false));
                g.ops.push(one(x, hn2, Some(&m.ln_pre), false));
                g.ops.push(Op::Route {
                    x: xr,
                    router: &m.router,
                    gate: None,
                    n_exp: ne,
                    h,
                    k: m.experts.top_k,
                    logits: rlogits,
                    sel,
                    expert_scale: Some(&m.per_expert_scale),
                });
                let mut gate = Vec::with_capacity(ne);
                let mut up = Vec::with_capacity(ne);
                for gu in &m.experts.gate_up {
                    match gu {
                        crate::moe::GateUp::Fused(l) if !l.has_lora() && !l.is_transient() => gate.push(l.as_ref()),
                        crate::moe::GateUp::Split(a, b) if !a.has_lora() && !b.has_lora() && !a.is_transient() && !b.is_transient() => {
                            gate.push(a.as_ref());
                            up.push(b.as_ref());
                        }
                        _ => return None,
                    }
                }
                if m.experts.down.iter().any(|l| l.has_lora() || l.is_transient()) {
                    return None;
                }
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
                    add: false,
                }));
                g.ops.push(one(eo, ob, Some(&m.ln_post), true));
            }
            let ple = layer.ple.as_ref().zip(pli);
            g.ops.push(Op::AddNorm {
                y: ob,
                x,
                wy: Cow::Borrowed(&layer.ln_post_ffn),
                scale: if ple.is_some() { 1.0 } else { layer.layer_scalar },
                next: if ple.is_some() { None } else { Some(next_norm) },
                n: h,
                eps,
            });
            if let Some((pw, pli)) = ple {
                g.ops.push(lin(&pw.gate, p, h, x, gp, LinMode::Store)?);
                g.ops.push(Op::Glu {
                    g: gp,
                    u: pli.at(l * p),
                    out: gp,
                    n: p,
                    gelu: true,
                });
                g.ops.push(lin(&pw.proj, h, p, gp, ob, LinMode::Store)?);
                g.ops.push(Op::AddNorm {
                    y: ob,
                    x,
                    wy: Cow::Borrowed(&pw.norm),
                    scale: layer.layer_scalar,
                    next: Some(next_norm),
                    n: h,
                    eps,
                });
            }
            have_hn = true;
        }
        g.head_from = g.ops.len();
        g.ops.push(lin(&self.head, cfg.vocab_size, h, hidden, logits, LinMode::Store)?);
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

    /// The GPU decode plan, compiled on first use (pipelines, buffers: ~0.1-0.2 s, so multi-token
    /// passes build it ahead of the first decode step); None when unavailable.
    pub(crate) fn gpu_plan(&self) -> Option<&dyn crate::gpu::decode::DecodePlan> {
        if !crate::gpu::decode_enabled() || !self.stream.is_none() {
            return None;
        }
        self.gpu_plan
            .get_or_init(|| {
                let be = crate::gpu::active()?;
                let g = self.gpu_graph()?;
                be.decode_plan(&g)
            })
            .as_deref()
    }

    /// Runs token `tok` through the GPU decode plan: the final-normed hidden state and, if
    /// wanted, the (softcapped) logits. None (nothing changed) when GPU decoding is off or
    /// unavailable for this model.
    pub(super) fn gpu_step(&self, sess: &mut Session, tok: u32, want_logits: bool) -> Option<(Vec<f32>, Option<Vec<f32>>)> {
        if !crate::gpu::decode_enabled() || self.stream.is_some() {
            return None;
        }
        let plan = self.gpu_plan()?;
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
                rope_cs: &cs,
                x: &x,
                x2: &x2,
                kv: &mut kv,
                states: &mut [],
                hidden: &mut hidden,
                logits: logits.as_deref_mut(),
            },
        );
        if !ok {
            return None;
        }
        sess.pos += 1;
        Some((hidden, logits))
    }
}
