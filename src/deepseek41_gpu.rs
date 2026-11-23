//! DeepSeek V4.1 single-token decode steps on Metal ([`crate::metal::ds41`]): the description of
//! the layers for the plan, the device copies of the session's caches kept in step with the host
//! session, and the host's share of a step (input embedding, engram rows, rotary angles).
//!
//! The host session stays authoritative: every GPU step appends its keys (window keys, the
//! compressors' rows, compressed and index keys) to it, so prefill, speculative verification and
//! a fallback step run on the CPU from the same state; the device copies are brought up to date
//! from it before a step (after a prefill, a rollback, a CPU step).

use super::*;
use crate::metal::ds41 as md;
use crate::metal::{MetalBackend, ds41::Plan};

/// Whether this process decodes DeepSeek V4.1 on the GPU: yes unless `stream_gpu` is off
/// (`gpu::set_stream_decode(Some(false))`, `OXINFER_STREAM_GPU=0`, `--stream-gpu off`) or device
/// decoding is ([`crate::gpu::decode_enabled`]). On an M5 Max it is about twice the CPU's speed
/// with the experts cached and ahead of it from a cold cache (docs/deepseek-v4.md).
pub(crate) fn wanted() -> bool {
    crate::gpu::decode_enabled() && crate::gpu::stream_decode_forced() != Some(false)
}

fn backend() -> Option<&'static MetalBackend> {
    crate::gpu::active()?.as_any()?.downcast_ref::<MetalBackend>()
}

/// The layers as the plan reads them (None when the model has a part it does not run).
fn desc<'m>(m: &'m DeepSeek4Model<'_>) -> Option<md::ModelDesc<'m>> {
    let c = &m.cfg;
    if !c.v41 || m.ern.is_some() || c.head_dim != 512 || c.index_n_heads != 32 || c.index_head_dim != 128 {
        return None;
    }
    let (h, n, nh, hd, ql) = (c.hidden_size, c.hc_mult, c.num_attention_heads, c.head_dim, c.q_lora_rank);
    let (ni, di, g, r) = (c.index_n_heads, c.index_head_dim, c.o_groups, c.o_lora_rank);
    let cols = (c.engram_max_ngram_size.max(1) - 1) * c.engram_n_heads;
    let ed = c.engram_head_dim;
    let mat = |lin: &'m Lin<'_>, n_out: usize, n_in: usize| md::Mat { lin, n_out, n_in };
    let mut layers = Vec::with_capacity(m.layers.len());
    for (l, layer) in m.layers.iter().enumerate() {
        let lx = m.x41.get(l)?;
        let a = &layer.attn;
        let moe = &layer.moe;
        let bias = match &moe.pick {
            Pick::Score(b) => b.as_deref(),
            Pick::Hash(_) => return None,
        };
        let mut experts = Vec::with_capacity(moe.experts.n_experts());
        for e in moe.experts.lins() {
            let [gate, up, down] = e[..] else { return None };
            experts.push(md::ExpertDesc { gate, up, down });
        }
        let s = &moe.shared;
        layers.push(md::LayerDesc {
            ratio: layer.ratio,
            attn_norm: &layer.attn_norm,
            ffn_norm: &layer.ffn_norm,
            hc_attn: md::HcDesc {
                f: &layer.hc_attn.f,
                base: &layer.hc_attn.base,
                scale: &layer.hc_attn.scale,
            },
            hc_ffn: md::HcDesc {
                f: &layer.hc_ffn.f,
                base: &layer.hc_ffn.base,
                scale: &layer.hc_ffn.scale,
            },
            wq_a: mat(&a.wq_a, ql, h),
            q_norm: &a.q_norm,
            wq_b: mat(&a.wq_b, nh * hd, ql),
            wkv: mat(&a.wkv, hd, h),
            kv_norm: &a.kv_norm,
            wo_a: a.wo_a.iter().map(|w| mat(w, r, nh * hd / g)).collect(),
            wo_b: mat(&a.wo_b, h, g * r),
            sink: &a.sink,
            comp: lx.comp.as_ref().map(|cp| md::CompDesc {
                wkv: mat(&cp.wkv, hd, h),
                wgate: cp.wgate.as_ref().map(|w| mat(w, hd, h)),
                norm: &cp.norm,
            }),
            idx: lx.idx.as_ref().map(|ix| md::IdxDesc {
                wq: mat(&ix.wq, ni * di, ql),
                proj: mat(&ix.proj, ni, h),
                wk: ix.wk.as_ref().map(|w| mat(w, di, hd)),
                k_norm: &ix.k_norm,
            }),
            engram: lx.engram.as_ref().map(|e| md::EngDesc {
                wkv: mat(&e.wkv, (n + 1) * h, cols * ed),
                weight: &e.weight,
            }),
            kv_src: lx.kv_src,
            router: match &moe.router {
                RouterW::F32(w) => md::RouterDesc::F32(w),
                RouterW::Bf16(w) => md::RouterDesc::Bf16(w),
            },
            bias,
            experts,
            shared: [mat(&s.gate, s.inter, h), mat(&s.up, s.inter, h), mat(&s.down, h, s.inter)],
            shared_limit: s.limit,
        });
    }
    Some(md::ModelDesc {
        h,
        n,
        nh,
        ql,
        o_lora: r,
        topk: c.index_topk,
        window: c.sliding_window,
        half: m.inv_main.len(),
        eps: c.rms_norm_eps as f32,
        heps: c.hc_eps as f32,
        sinkhorn: c.hc_sinkhorn_iters,
        rsf: c.routed_scaling_factor as f32,
        limit: c.swiglu_limit as f32,
        scoring: match c.scoring_func.as_str() {
            "softmax" => 0,
            "sigmoid" => 1,
            _ => 2,
        },
        norm_topk: c.norm_topk_prob,
        k: c.num_experts_per_tok,
        inter: c.moe_intermediate_size,
        cand: c
            .candidate_source()
            .filter(|_| c.candidate_topk_blocks > 0 && c.candidate_block_size > 0)
            .map(|l| (l, c.candidate_topk_blocks, c.candidate_block_size)),
        eng_in: cols * ed,
        vocab: c.vocab_size,
        layers,
        norm: &m.norm,
        head: mat(&m.head, c.vocab_size, h),
    })
}

/// The model's plan on `be` (built on first use; None when it cannot run there).
fn plan<'p>(m: &'p DeepSeek4Model<'_>, be: &'static MetalBackend) -> Option<&'p Plan> {
    m.gpu_plan
        .get_or_init(|| {
            let t = std::time::Instant::now();
            let p = desc(m).and_then(|d| Plan::build(be, &d));
            if std::env::var_os("OXINFER_DEBUG").is_some() || p.is_none() {
                eprintln!(
                    "[deepseek v4.1: {} ({:.1} s)]",
                    if p.is_some() {
                        "GPU decode plan built"
                    } else {
                        "no GPU decode plan: decoding on the CPU"
                    },
                    t.elapsed().as_secs_f64()
                );
            }
            p.map(|p| Box::new(p) as Box<dyn std::any::Any + Send + Sync>)
        })
        .as_ref()
        .and_then(|b| b.downcast_ref::<Plan>())
}

/// Builds the plan and wraps the experts in memory (on a thread of its own while a prefill runs,
/// so that the first decode step does not wait for them).
pub(crate) fn prepare(m: &DeepSeek4Model) {
    if let Some(be) = backend()
        && let Some(p) = plan(m, be)
    {
        p.prewrap(m.stream.as_ref());
    }
}

/// Brings the device copies of `sess`'s caches up to its position.
fn sync(m: &DeepSeek4Model, p: &Plan, st: &mut md::State, sess: &DeepSeek4Session) -> Option<()> {
    let pos = sess.pos;
    let kept = st.dev_pos.min(sess.gpu_kept).min(pos);
    if kept == pos && st.dev_pos == pos {
        return Some(());
    }
    let (hd, di) = (m.cfg.head_dim, m.cfg.index_head_dim);
    let w = m.cfg.sliding_window;
    // After a rollback the device wrote positions the host dropped, into ring rows of positions
    // still in the window: the whole window again.
    let lo = if st.dev_pos > kept { 0 } else { kept };
    for (l, ls) in sess.layers.iter().enumerate() {
        for q in lo.max(pos.saturating_sub(w))..pos {
            let at = (q - ls.raw_start) * hd;
            st.ring_row(l, q).copy_from_slice(&ls.raw[at..at + hd]);
        }
    }
    let mut s = 0;
    for (l, lx) in m.x41.iter().enumerate() {
        if lx.comp.is_none() {
            continue;
        }
        let r = p.src_rates()[s];
        let ls = &sess.layers[l];
        let (cs, is) = (ls.comp.as_ref()?, ls.idx.as_ref()?);
        let v = (*st.valid(s)).min(kept / r).min(cs.n);
        if cs.n > v {
            st.comp_rows(s, v, cs.n)?.copy_from_slice(&cs.keys[v * hd..cs.n * hd]);
            st.idx_rows(s, v, is.n)?.copy_from_slice(&is.keys[v * di..is.n * di]);
        }
        *st.valid(s) = cs.n;
        if r > 1 {
            // The rows of the group still pooling.
            let pend = st.pend(s, r);
            for q in pos - pos % r..pos {
                let at = (q - cs.rows_start) * cs.width;
                let j = q % r;
                pend[j * hd..(j + 1) * hd].copy_from_slice(&cs.kv[at..at + hd]);
                pend[(r + j) * hd..(r + j + 1) * hd].copy_from_slice(&cs.sc[at..at + hd]);
            }
        }
        s += 1;
    }
    st.dev_pos = pos;
    Some(())
}

/// One decode step of `token` on the GPU: the logits (or with `hidden_only`, the final-normed
/// hidden state), the session advanced as a CPU step would. None (nothing changed) when the GPU
/// cannot take it; the caller then runs the CPU path.
pub(crate) fn step(m: &DeepSeek4Model, sess: &mut DeepSeek4Session, token: u32, hidden_only: bool) -> Option<Vec<f32>> {
    step_on(m, sess, token, hidden_only, backend()?)
}

pub(crate) fn step_on(
    m: &DeepSeek4Model,
    sess: &mut DeepSeek4Session,
    token: u32,
    hidden_only: bool,
    be: &'static MetalBackend,
) -> Option<Vec<f32>> {
    if !m.cfg.v41 || sess.iv.active() {
        return None;
    }
    let p = plan(m, be)?;
    let c = &m.cfg;
    let (h, hd) = (c.hidden_size, c.head_dim);
    let mut st = match sess.gpu.0.take().and_then(|b| b.downcast::<md::State>().ok()) {
        Some(s) if p.owns(&s) => *s,
        _ => p.new_state()?,
    };
    let check = crate::gpu::decode::check().then(|| sess.clone());
    sync(m, p, &mut st, sess)?;
    let pos = sess.pos;
    // The host's inputs.
    let mut x = vec![0f32; h];
    m.embed_row(token, &mut x);
    if let Some(hash) = &m.hash {
        sess.eng.push(hash.token_map[token as usize]);
    }
    let mut eng = Vec::new();
    if let Some(hash) = &m.hash {
        let ed = c.engram_head_dim;
        let cols = (hash.ngram - 1) * hash.heads;
        let mut ids = vec![0usize; cols];
        for lx in &m.x41 {
            let Some(e) = &lx.engram else { continue };
            hash.ids(e.li, &sess.eng, pos, &mut ids);
            ids.iter_mut().for_each(|r| *r += e.base);
            if let Some(s) = &m.stream {
                let rows: Vec<(usize, usize)> = ids.iter().flat_map(|&r| e.table.row_ranges(r, ed)).collect();
                s.need_ranges(&rows);
            }
            let at = eng.len();
            eng.resize(at + cols * ed, 0.0);
            for (i, &r) in ids.iter().enumerate() {
                e.table.row(r, &mut eng[at + i * ed..at + (i + 1) * ed]);
            }
        }
    }
    let mut cs = Vec::with_capacity(6 * m.inv_main.len());
    for (p0, inv) in [(pos, &m.inv_main), (pos, &m.inv_comp), (pos.saturating_sub(1), &m.inv_comp)] {
        let (co, si) = rope_tables(&[p0], inv);
        cs.extend(co);
        cs.extend(si);
    }
    let mut out = vec![0f32; if hidden_only { h } else { c.vocab_size }];
    let inp = md::StepIn {
        pos,
        x: &x,
        cs: &cs,
        eng: &eng,
    };
    if !p.step(&mut st, &inp, m.stream.as_ref(), &mut out, hidden_only) {
        if m.hash.is_some() {
            sess.eng.pop();
        }
        sess.gpu.0 = Some(Box::new(st));
        return None;
    }
    // The step's keys into the host session.
    let mut s = 0;
    for (l, lx) in m.x41.iter().enumerate() {
        let row = st.ring_row(l, pos).to_vec();
        let ls = &mut sess.layers[l];
        debug_assert_eq!(ls.raw_start + ls.raw.len() / hd, pos);
        ls.raw.extend_from_slice(&row);
        if lx.comp.is_none() {
            continue;
        }
        let r = p.src_rates()[s];
        if r > 1 {
            let j = pos % r;
            let pend = st.pend(s, r).to_vec();
            let cs = ls.comp.as_mut().unwrap();
            cs.kv.extend_from_slice(&pend[j * hd..(j + 1) * hd]);
            cs.sc.extend_from_slice(&pend[(r + j) * hd..(r + j + 1) * hd]);
        }
        if (pos + 1).is_multiple_of(r) {
            let row = (pos + 1) / r - 1;
            let key = st.comp_rows(s, row, row + 1)?.to_vec();
            let ik = st.idx_rows(s, row, row + 1)?.to_vec();
            let cs = ls.comp.as_mut().unwrap();
            debug_assert_eq!(cs.n, row);
            cs.keys.extend_from_slice(&key);
            cs.n += 1;
            let is = ls.idx.as_mut().unwrap();
            is.keys.extend_from_slice(&ik);
            is.n += 1;
            *st.valid(s) = row + 1;
        }
        s += 1;
    }
    sess.pos += 1;
    sess.trim();
    st.dev_pos = sess.pos;
    sess.gpu_kept = sess.pos;
    sess.gpu.0 = Some(Box::new(st));
    if let Some(mut cpu_sess) = check {
        let want = if hidden_only {
            forward(m, &mut cpu_sess, &[token], &[], &[], &mut Vec::new())
        } else {
            let hs = forward(m, &mut cpu_sess, &[token], &[], &[], &mut Vec::new());
            m.logits(&hs, 1)
        };
        crate::gpu::decode::report(pos, &want, &out);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small random V4.1 model at the kernels' sizes (512-wide heads, 32 indexer heads of 128):
    /// greedy decode steps on the GPU give the CPU path's logits, through the indexer's choice
    /// (top 2), the candidate blocks, the rate-2 pooling, the engram rows, a rollback and CPU
    /// steps in between.
    #[test]
    fn gpu_decode_matches_cpu() {
        // A backend of its own: the process-wide one would put the other tests on the GPU.
        let Ok(be) = MetalBackend::new() else {
            return;
        };
        let be: &'static MetalBackend = Box::leak(Box::new(be));
        let over = serde_json::json!({
            "hidden_size": 256, "num_attention_heads": 8, "head_dim": 512, "qk_rope_head_dim": 64,
            "q_lora_rank": 64, "o_lora_rank": 64, "n_routed_experts": 8,
            "moe_intermediate_size": 64, "index_n_heads": 32, "index_head_dim": 128,
            "engram_head_dim": 16, "sliding_window": 4,
        });
        let dir = std::env::temp_dir().join(format!("oxinfer-ds41-gpu-test-{}", std::process::id()));
        super::super::tests::tiny_checkpoint_with(&dir, over.clone());
        let files = ModelFiles::open(&dir, true).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let m = DeepSeek4Model::load(super::super::tests::tiny_config(over), &files, Quant::Bf16).unwrap();
        let v = m.cfg.vocab_size;
        let ids: Vec<u32> = (0..9).map(|i| (i * 7 + 3) % v as u32).collect();
        let mut cpu = m.new_session();
        m.forward(&mut cpu, &ids);
        let mut gpu = cpu.clone();
        let rel = |a: &[f32], b: &[f32]| {
            let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            d / b.iter().fold(0f32, |m, x| m.max(x.abs()))
        };
        let mut worst = 0f32;
        let mut tok = 5u32;
        for i in 0..24 {
            let want = m.logits(&forward(&m, &mut cpu, &[tok], &[], &[], &mut Vec::new()), 1);
            let got = if i % 7 == 3 {
                // A CPU step on the GPU session: the device copies catch up before the next.
                m.logits(&forward(&m, &mut gpu, &[tok], &[], &[], &mut Vec::new()), 1)
            } else {
                step_on(&m, &mut gpu, tok, false, be).expect("the GPU step did not run")
            };
            let r = rel(&got, &want);
            worst = worst.max(r);
            assert!(r < 1e-3, "step {i} (position {}): logits differ by {r} (relative)", cpu.pos - 1);
            if i == 12 {
                // A rollback of two positions on both.
                cpu.truncate(cpu.pos - 2);
                gpu.truncate(gpu.pos - 2);
            }
            tok = (tok * 5 + 11) % v as u32;
        }
        // The hidden state alone.
        let want = forward(&m, &mut cpu, &[tok], &[], &[], &mut Vec::new());
        let got = step_on(&m, &mut gpu, tok, true, be).unwrap();
        assert!(rel(&got, &want) < 1e-3);
        eprintln!("worst relative logit difference {worst:.2e}");
    }
}
