//! DeepSeek V4.1 decode passes on Metal ([`crate::metal::ds41`]): the description of the layers
//! for the plan, the device copies of the session's caches kept in step with the host session,
//! and the host's share of a pass (engram rows, rotary angles; the input rows when the device
//! cannot read the embedding table). One-token steps, verification passes of a few tokens
//! (prompt lookup: the logits of one-token steps, bit for bit) and greedy decoding chained on the
//! device ([`decode_greedy`]).
//!
//! The host session stays authoritative: every GPU pass appends its keys (window keys, the
//! compressors' rows, compressed and index keys) to it, so prefill and a fallback step run on
//! the CPU from the same state; the device copies are brought up to date from it before a pass
//! (after a prefill, a rollback, a CPU step).

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

/// `OXINFER_DS41_CHAIN=0`: greedy decoding one step at a time instead of chained on the device
/// (for measurements).
fn chain_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OXINFER_DS41_CHAIN").is_none_or(|v| v != "0"))
}

/// `OXINFER_DS41_TRACE=1`: a line per chained pass.
fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OXINFER_DS41_TRACE").is_some_and(|v| v != "0"))
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
        embed: match &m.embed {
            Table::Bf16(v) => Some(md::EmbDesc::Bf16(v)),
            Table::Gguf(g) if g.block_scales.is_none() && !g.x4 => Some(md::EmbDesc::Gguf {
                kind: g.kind,
                data: &g.data,
                rb: g.row_bytes(),
            }),
            _ => None,
        },
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

/// The session's device state (a fresh one if it has none of this plan's).
fn take_state(p: &Plan, sess: &mut DeepSeek4Session) -> Option<md::State> {
    match sess.gpu.0.take().and_then(|b| b.downcast::<md::State>().ok()) {
        Some(s) if p.owns(&s) => Some(*s),
        _ => p.new_state(),
    }
}

/// Brings the device copies of `sess`'s caches up to its position.
fn sync(m: &DeepSeek4Model, p: &Plan, st: &mut md::State, sess: &DeepSeek4Session) -> Option<()> {
    let pos = sess.pos;
    let dev = st.dev_pos;
    let kept = dev.min(sess.gpu_kept).min(pos);
    if kept == pos && dev == pos {
        return Some(());
    }
    let (hd, di) = (m.cfg.head_dim, m.cfg.index_head_dim);
    let w = m.cfg.sliding_window;
    let rr = st.ring_len();
    // The window rows the device does not hold: the positions it has not run, and the rows of
    // positions it overwrote with positions the session does not have (past a rollback: none
    // for a rollback of up to the ring's spare rows).
    let lo = kept.max(pos.saturating_sub(w));
    let mut rows: Vec<usize> = (lo..pos).collect();
    rows.extend(
        (kept..dev)
            .filter_map(|q| q.checked_sub(rr))
            .filter(|&p0| p0 >= pos.saturating_sub(w) && p0 < lo),
    );
    for (l, ls) in sess.layers.iter().enumerate() {
        for &q in &rows {
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
            for q in pos - pos % r..pos {
                let at = (q - cs.rows_start) * cs.width;
                let (kv, sc) = st.pend_rows(s, r, q);
                kv.copy_from_slice(&cs.kv[at..at + hd]);
                sc.copy_from_slice(&cs.sc[at..at + hd]);
            }
        }
        s += 1;
    }
    st.dev_pos = pos;
    Some(())
}

/// The rotary angles of position `pos` as a pass slot takes them: cos then sin of the main
/// table at `pos`, the compressed table at `pos`, the compressed table at `pos - 1`.
fn angles(m: &DeepSeek4Model, pos: usize, out: &mut Vec<f32>) {
    for (p0, inv) in [(pos, &m.inv_main), (pos, &m.inv_comp), (pos.saturating_sub(1), &m.inv_comp)] {
        let (co, si) = rope_tables(&[p0], inv);
        out.extend(co);
        out.extend(si);
    }
}

/// Appends `token`'s engram id to the session's (the input of position `sess.eng.len()`) and
/// the engram rows of that position to `out` (per engram layer, its `cols * head_dim` floats).
fn engram_rows(m: &DeepSeek4Model, sess: &mut DeepSeek4Session, token: u32, out: &mut Vec<f32>) {
    let Some(hash) = &m.hash else { return };
    sess.eng.push(hash.token_map[token as usize]);
    let pos = sess.eng.len() - 1;
    let ed = m.cfg.engram_head_dim;
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
        let at = out.len();
        out.resize(at + cols * ed, 0.0);
        for (i, &r) in ids.iter().enumerate() {
            e.table.row(r, &mut out[at + i * ed..at + (i + 1) * ed]);
        }
    }
}

/// The keys of the pass at `pos` into the host session (its window keys, the compressors' rows,
/// a completed group's compressed and index keys).
fn to_host(m: &DeepSeek4Model, p: &Plan, st: &mut md::State, sess: &mut DeepSeek4Session, pos: usize) -> Option<()> {
    let hd = m.cfg.head_dim;
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
            let (kv, sc) = st.pend_rows(s, r, pos);
            let cs = ls.comp.as_mut()?;
            cs.kv.extend_from_slice(kv);
            cs.sc.extend_from_slice(sc);
        }
        if (pos + 1).is_multiple_of(r) {
            let row = (pos + 1) / r - 1;
            let key = st.comp_rows(s, row, row + 1)?.to_vec();
            let ik = st.idx_rows(s, row, row + 1)?.to_vec();
            let cs = ls.comp.as_mut()?;
            debug_assert_eq!(cs.n, row);
            cs.keys.extend_from_slice(&key);
            cs.n += 1;
            let is = ls.idx.as_mut()?;
            is.keys.extend_from_slice(&ik);
            is.n += 1;
            *st.valid(s) = row + 1;
        }
        s += 1;
    }
    Some(())
}

/// One decode step of `token` on the GPU: the logits (or with `hidden_only`, the final-normed
/// hidden state), the session advanced as a CPU step would. None (nothing changed) when the GPU
/// cannot take it; the caller then runs the CPU path.
pub(crate) fn step(m: &DeepSeek4Model, sess: &mut DeepSeek4Session, token: u32, hidden_only: bool) -> Option<Vec<f32>> {
    pass_on(m, sess, &[token], hidden_only, backend()?)
}

/// A pass of `tokens` (up to [`md::TMAX`]) on the GPU: per token, the logits (or with
/// `hidden_only`, the final-normed hidden states), each token's the bits of a one-token step;
/// the session advanced as a CPU pass would. None (nothing changed) when the GPU cannot take
/// it.
pub(crate) fn pass(m: &DeepSeek4Model, sess: &mut DeepSeek4Session, tokens: &[u32], hidden_only: bool) -> Option<Vec<f32>> {
    pass_on(m, sess, tokens, hidden_only, backend()?)
}

pub(crate) fn pass_on(
    m: &DeepSeek4Model,
    sess: &mut DeepSeek4Session,
    tokens: &[u32],
    hidden_only: bool,
    be: &'static MetalBackend,
) -> Option<Vec<f32>> {
    let nt = tokens.len();
    if !m.cfg.v41 || sess.iv.active() || nt == 0 || nt > md::TMAX {
        return None;
    }
    let p = plan(m, be)?;
    if nt > 1 && !p.multi() {
        return None;
    }
    let c = &m.cfg;
    let h = c.hidden_size;
    let mut st = take_state(p, sess)?;
    let check = crate::gpu::decode::check().then(|| sess.clone());
    if sync(m, p, &mut st, sess).is_none() {
        sess.gpu.0 = Some(Box::new(st));
        return None;
    }
    let pos = sess.pos;
    // The host's inputs.
    let mut x = Vec::new();
    if !p.embeds() {
        x.resize(nt * h, 0.0);
        for (i, &t) in tokens.iter().enumerate() {
            m.embed_row(t, &mut x[i * h..(i + 1) * h]);
        }
    }
    let mut eng = Vec::new();
    let mut cs = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        engram_rows(m, sess, t, &mut eng);
        angles(m, pos + i, &mut cs);
    }
    let width = if hidden_only { h } else { c.vocab_size };
    let mut out = vec![0f32; nt * width];
    let inp = md::PassIn {
        pos,
        nt,
        t0: 0,
        src: if p.embeds() {
            md::PassSrc::Tokens(tokens)
        } else {
            md::PassSrc::Rows(&x)
        },
        cs: &cs,
        eng: &eng,
    };
    if !p.pass(&mut st, &inp, m.stream.as_ref(), &mut out, hidden_only) {
        if m.hash.is_some() {
            sess.eng.truncate(pos);
        }
        sess.gpu.0 = Some(Box::new(st));
        return None;
    }
    for i in 0..nt {
        to_host(m, p, &mut st, sess, pos + i)?;
        sess.pos += 1;
    }
    sess.trim();
    st.dev_pos = sess.pos;
    sess.gpu_kept = sess.pos;
    sess.gpu.0 = Some(Box::new(st));
    if let Some(mut cpu_sess) = check {
        let hs = forward(m, &mut cpu_sess, tokens, &[], &[], &mut Vec::new());
        let want = if hidden_only { hs } else { m.logits(&hs, nt) };
        crate::gpu::decode::report(pos, &want, &out);
    }
    Some(out)
}

/// Greedy decoding chained on the device (see [`crate::generate::TextLm::decode_greedy`]): runs
/// `first` and the argmax tokens after it, at most `max` passes, `on` getting each pass's token
/// in order. A pass takes its input row from the argmax of the pass before on the device, its
/// first layers queued while the pass before runs; the host takes the token, looks up the engram
/// rows the later layers need and releases them. Expert misses are handled as a step handles
/// them (the pass again from the first layer that missed); while they last, passes run one at a
/// time. The passes kept, or None (nothing run) when the GPU does not do it.
pub(crate) fn decode_greedy(
    m: &DeepSeek4Model,
    sess: &mut DeepSeek4Session,
    first: u32,
    max: usize,
    on: &mut dyn FnMut(u32) -> crate::generate::Chain,
) -> Option<usize> {
    decode_greedy_on(m, sess, first, max, on, backend()?)
}

fn decode_greedy_on(
    m: &DeepSeek4Model,
    sess: &mut DeepSeek4Session,
    first: u32,
    max: usize,
    on: &mut dyn FnMut(u32) -> crate::generate::Chain,
    be: &'static MetalBackend,
) -> Option<usize> {
    use crate::generate::Chain as Ch;
    if !m.cfg.v41 || sess.iv.active() || max == 0 || crate::gpu::decode::check() || !chain_on() {
        return None;
    }
    let p = plan(m, be)?;
    if !p.embeds() {
        return None;
    }
    let mut st = take_state(p, sess)?;
    let hook = m.stream.as_ref();
    let ch = if sync(m, p, &mut st, sess).is_some() { p.chain(hook) } else { None };
    let Some(mut ch) = ch else {
        sess.gpu.0 = Some(Box::new(st));
        return None;
    };
    let pos0 = sess.pos;
    let mut cs = Vec::new();
    let mut eng = Vec::new();
    // Pass 0 takes the host's token (from slot 1).
    ch.set_token(1, first);
    angles(m, pos0, &mut cs);
    engram_rows(m, sess, first, &mut eng);
    ch.inputs(0, &cs, Some(&eng));
    let queue = |ch: &mut md::Chain<'_>, st: &mut md::State, pos: usize, j: usize| {
        ch.queue(st, pos, j).map(|mut q| {
            ch.release(&mut q);
            q
        })
    };
    let mut cur = if ch.cold() { None } else { queue(&mut ch, &mut st, pos0, 0) };
    let mut next = None;
    if cur.is_some() && max > 1 {
        cs.clear();
        angles(m, pos0 + 1, &mut cs);
        ch.inputs(1, &cs, None);
        next = ch.queue(&mut st, pos0 + 1, 1);
    }
    let (mut k, mut last) = (0usize, false);
    loop {
        let (pos, j) = (pos0 + k, k % 2);
        let ok = match cur.take() {
            Some(q) => match ch.wait(q) {
                None => false,
                Some(miss) if miss.is_empty() => true,
                Some(miss) => {
                    // The next pass's first layers ran on a token of missing experts.
                    if let Some(nq) = next.take() {
                        ch.discard(nq, &mut st);
                    }
                    ch.run_now(&mut st, pos, j, miss, hook)
                }
            },
            None => ch.run_now(&mut st, pos, j, Vec::new(), hook),
        };
        if !ok {
            chain_failed();
        }
        let ti = std::time::Instant::now();
        let tok = ch.token(j);
        k += 1;
        // The next pass (its input `tok`) goes on before the host's bookkeeping and the caller's
        // callback: its engram rows, then its later layers released, then the first layers of
        // the pass after it queued (a stop undoes them).
        let started = next.is_some() || (!last && k < max);
        if started {
            let (pos1, j1) = (pos0 + k, k % 2);
            eng.clear();
            engram_rows(m, sess, tok, &mut eng);
            cs.clear();
            angles(m, pos1, &mut cs);
            ch.inputs(j1, &cs, Some(&eng));
            cur = match next.take() {
                Some(mut q) => {
                    ch.release(&mut q);
                    Some(q)
                }
                None if ch.cold() => None,
                None => queue(&mut ch, &mut st, pos1, j1),
            };
        }
        let turnaround = ti.elapsed().as_secs_f64();
        // The part cache's bookkeeping of the pass done (before its slot is taken again).
        ch.note(j, hook);
        if started && cur.is_some() && !last && k + 1 < max && !ch.cold() {
            let pos2 = pos0 + k + 1;
            cs.clear();
            angles(m, pos2, &mut cs);
            ch.inputs(j, &cs, None);
            next = ch.queue(&mut st, pos2, j);
        }
        to_host(m, p, &mut st, sess, pos).unwrap_or_else(|| chain_failed());
        sess.pos += 1;
        st.dev_pos = st.dev_pos.max(sess.pos);
        sess.gpu_kept = sess.pos;
        if trace() {
            eprintln!(
                "[ds41 gpu chain: pass {pos} done, the next released after {:.3} ms{}]",
                turnaround * 1e3,
                if started && cur.is_none() { " (one at a time)" } else { "" }
            );
        }
        match on(tok) {
            Ch::Stop => {
                // The passes started for the token not taken: their rows past the session's.
                if let Some(nq) = next.take() {
                    ch.discard(nq, &mut st);
                }
                if let Some(q) = cur.take() {
                    ch.abandon(q, &mut st);
                }
                if m.hash.is_some() {
                    sess.eng.truncate(sess.pos);
                }
                break;
            }
            Ch::Last => {
                last = true;
                if let Some(nq) = next.take() {
                    ch.discard(nq, &mut st);
                }
            }
            Ch::Go => {}
        }
        if !started {
            break;
        }
    }
    drop(ch);
    sess.trim();
    sess.gpu.0 = Some(Box::new(st));
    Some(sess.pos - pos0)
}

#[cold]
fn chain_failed() -> ! {
    panic!("Metal: a DeepSeek V4.1 decode pass failed (OXINFER_DEBUG=1 says more)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::Chain as Ch;

    /// A small random V4.1 model at the kernels' sizes (512-wide heads, 32 indexer heads of 128,
    /// a 4-token window, the indexer keeping 2) and its checkpoint folder's files.
    fn tiny() -> ModelFiles {
        let dir = std::env::temp_dir().join(format!(
            "oxinfer-ds41-gpu-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        super::super::tests::tiny_checkpoint_with(&dir, over());
        let files = ModelFiles::open(&dir, true).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        files
    }

    fn over() -> serde_json::Value {
        serde_json::json!({
            "hidden_size": 256, "num_attention_heads": 8, "head_dim": 512, "qk_rope_head_dim": 64,
            "q_lora_rank": 64, "o_lora_rank": 64, "n_routed_experts": 8,
            "moe_intermediate_size": 64, "index_n_heads": 32, "index_head_dim": 128,
            "engram_head_dim": 16, "sliding_window": 4,
        })
    }

    /// A backend of its own: the process-wide one would put the other tests on the GPU. None
    /// (the test skips) without a device, or on a device without residency sets (such as the
    /// paravirtual GPU of CI's macOS virtual machines), where `Plan::build` declines and the
    /// model decodes on the CPU.
    fn own_backend() -> Option<&'static MetalBackend> {
        let be = MetalBackend::new().ok()?;
        if !md::supported(&be) {
            eprintln!("no Metal residency set on this device: the DeepSeek V4.1 GPU decode test skips");
            return None;
        }
        Some(Box::leak(Box::new(be)))
    }

    /// The model's session after a CPU prefill of 9 tokens.
    fn prefilled(m: &DeepSeek4Model) -> DeepSeek4Session {
        let v = m.cfg.vocab_size;
        let ids: Vec<u32> = (0..9).map(|i| (i * 7 + 3) % v as u32).collect();
        let mut s = m.new_session();
        m.forward(&mut s, &ids);
        s
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Greedy decode steps on the GPU give the CPU path's logits, through the indexer's choice
    /// (top 2), the candidate blocks, the rate-2 pooling, the engram rows, a rollback and CPU
    /// steps in between.
    #[test]
    fn gpu_decode_matches_cpu() {
        let Some(be) = own_backend() else { return };
        let files = tiny();
        let m = DeepSeek4Model::load(super::super::tests::tiny_config(over()), &files, Quant::Bf16).unwrap();
        let v = m.cfg.vocab_size;
        let mut cpu = prefilled(&m);
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
                pass_on(&m, &mut gpu, &[tok], false, be).expect("the GPU step did not run")
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
        let got = pass_on(&m, &mut gpu, &[tok], true, be).unwrap();
        assert!(rel(&got, &want) < 1e-3);
        eprintln!("worst relative logit difference {worst:.2e}");
    }

    /// A pass run again from a layer (after experts missed) and a pass in chunks of layers (while
    /// they miss) give the logits of a whole pass, bit for bit.
    #[test]
    fn gpu_rerun_and_chunks_are_exact() {
        let Some(be) = own_backend() else { return };
        let files = tiny();
        let m = DeepSeek4Model::load(super::super::tests::tiny_config(over()), &files, Quant::Bf16).unwrap();
        let p = plan(&m, be).unwrap();
        let nl = m.layers.len();
        let mut s = prefilled(&m);
        for (i, &tok) in [5u32, 9, 2, 17, 4].iter().enumerate() {
            let want = pass_on(&m, &mut s, &[tok], false, be).unwrap();
            let mut st = take_state(p, &mut s).unwrap();
            for from in 1..nl {
                assert_eq!(
                    bits(&p.rerun_from(&mut st, s.pos - 1, from)),
                    bits(&want),
                    "step {i}, again from layer {from}"
                );
            }
            s.gpu.0 = Some(Box::new(st));
            // The same step in chunks of layers.
            let mut a = s.clone();
            a.truncate(a.pos - 1);
            p.set_cold(true);
            let got = pass_on(&m, &mut a, &[tok], false, be).unwrap();
            assert_eq!(bits(&got), bits(&want), "step {i} in chunks");
        }
    }

    /// A pass of several tokens (prompt-lookup verification) gives the logits of one-token steps
    /// bit for bit, through groups pooling across the pass, the indexer's choice and the
    /// candidate blocks; after a rollback into the pass, steps go on as from one-token steps.
    #[test]
    fn gpu_pass_matches_steps() {
        let Some(be) = own_backend() else { return };
        let files = tiny();
        let m = DeepSeek4Model::load(super::super::tests::tiny_config(over()), &files, Quant::Bf16).unwrap();
        assert!(plan(&m, be).unwrap().multi(), "the tiny model's matrices have multi-token kernels");
        let v = m.cfg.vocab_size as u32;
        let base = prefilled(&m);
        for (nt, start) in [(2usize, 0usize), (5, 1), (16, 3), (11, 0)] {
            let mut a = base.clone();
            // Some one-token steps first (the device copies already in use).
            let mut tok = 3u32;
            for _ in 0..start {
                pass_on(&m, &mut a, &[tok], false, be).unwrap();
                tok = (tok * 3 + 1) % v;
            }
            let mut b = a.clone();
            let toks: Vec<u32> = (0..nt as u32).map(|i| (i * 11 + tok) % v).collect();
            let mut want = Vec::new();
            for &t in &toks {
                want.extend(pass_on(&m, &mut a, &[t], false, be).unwrap());
            }
            let got = pass_on(&m, &mut b, &toks, false, be).expect("the pass did not run");
            assert_eq!(bits(&got), bits(&want), "{nt} tokens from position {}", a.pos - nt);
            assert_eq!(a.pos, b.pos);
            // A rollback into the pass, then steps: as after one-token steps.
            let back = nt / 2 + 1;
            a.truncate(a.pos - back);
            b.truncate(b.pos - back);
            for i in 0..3u32 {
                let t = (i * 13 + 1) % v;
                let wa = pass_on(&m, &mut a, &[t], false, be).unwrap();
                let wb = pass_on(&m, &mut b, &[t], false, be).unwrap();
                assert_eq!(bits(&wa), bits(&wb), "{nt} tokens, step {i} after the rollback");
            }
        }
    }

    /// Greedy passes chained on the device give the tokens (and the session) of one-token steps
    /// with the host's argmax; a stop leaves the session as the steps before it, and the queued
    /// pass a "last" asks for still runs.
    #[test]
    fn gpu_greedy_chain_matches_steps() {
        let Some(be) = own_backend() else { return };
        let files = tiny();
        let m = DeepSeek4Model::load(super::super::tests::tiny_config(over()), &files, Quant::Bf16).unwrap();
        assert!(plan(&m, be).unwrap().embeds(), "the tiny model's embedding table is on the device");
        let base = prefilled(&m);
        let n = 21;
        // One-token steps.
        let mut a = base.clone();
        let mut want = Vec::new();
        let mut t = 5u32;
        for _ in 0..n {
            t = crate::sampler::argmax(&pass_on(&m, &mut a, &[t], false, be).unwrap());
            want.push(t);
        }
        // Chained.
        let mut b = base.clone();
        let mut got = Vec::new();
        let kept = decode_greedy_on(
            &m,
            &mut b,
            5,
            n,
            &mut |t| {
                got.push(t);
                Ch::Go
            },
            be,
        );
        assert_eq!(kept, Some(n));
        assert_eq!(got, want);
        assert_eq!(a.pos, b.pos);
        // The sessions go on alike.
        let la = pass_on(&m, &mut a, &[7], false, be).unwrap();
        let lb = pass_on(&m, &mut b, &[7], false, be).unwrap();
        assert_eq!(bits(&la), bits(&lb));
        // A stop at the fourth token: four passes kept, that token not run; then steps as from
        // one-token steps.
        let mut c = base.clone();
        let mut seen = Vec::new();
        let kept = decode_greedy_on(
            &m,
            &mut c,
            5,
            n,
            &mut |t| {
                seen.push(t);
                if seen.len() == 4 { Ch::Stop } else { Ch::Go }
            },
            be,
        );
        assert_eq!(kept, Some(4));
        assert_eq!(seen, want[..4]);
        assert_eq!(c.pos, base.pos + 4);
        let mut d = base.clone();
        let mut t = 5u32;
        for _ in 0..4 {
            t = crate::sampler::argmax(&pass_on(&m, &mut d, &[t], false, be).unwrap());
        }
        for i in 0..3u32 {
            let tk = if i == 0 { want[3] } else { 2 * i + 1 };
            let lc = pass_on(&m, &mut c, &[tk], false, be).unwrap();
            let ld = pass_on(&m, &mut d, &[tk], false, be).unwrap();
            assert_eq!(bits(&lc), bits(&ld), "step {i} after the stop");
        }
        // A "last" at the second token: the third pass (queued) still comes.
        let mut e = base.clone();
        let mut seen = Vec::new();
        let kept = decode_greedy_on(
            &m,
            &mut e,
            5,
            n,
            &mut |t| {
                seen.push(t);
                if seen.len() == 2 { Ch::Last } else { Ch::Go }
            },
            be,
        );
        assert_eq!(kept, Some(3));
        assert_eq!(seen, want[..3]);
        assert_eq!(e.pos, base.pos + 3);
    }

    /// Greedy one-token GPU steps from `t`: the logits of each.
    fn gpu_steps(m: &DeepSeek4Model, s: &mut DeepSeek4Session, mut t: u32, n: usize, be: &'static MetalBackend) -> Vec<f32> {
        let mut out = Vec::new();
        for _ in 0..n {
            let l = pass_on(m, s, &[t], false, be).expect("the GPU step did not run");
            t = crate::sampler::argmax(&l);
            out.extend(l);
        }
        out
    }

    /// A session decoded on the GPU far past a checkpoint (the device's window ring and the
    /// host's windows overwritten and trimmed), cut back to it, decodes as a fresh session from
    /// the same prefill, bit for bit (the device copies brought back from the host's); chained
    /// greedy decoding too.
    #[test]
    fn gpu_checkpoint_cut_back_matches_fresh() {
        let Some(be) = own_backend() else { return };
        let files = tiny();
        let m = DeepSeek4Model::load(super::super::tests::tiny_config(over()), &files, Quant::Bf16).unwrap();
        let base = prefilled(&m);
        let c = base.pos;
        let mut a = base.clone();
        a.checkpoint(2);
        gpu_steps(&m, &mut a, 5, 3 * crate::deepseek4::ROLLBACK + 20, be);
        assert!(a.layers.iter().any(|l| l.raw_start > c), "the window was not trimmed");
        assert_eq!(a.reachable(c + 3), c);
        a.cut_to(c).unwrap();
        let got = gpu_steps(&m, &mut a, 9, 30, be);
        let mut b = base.clone();
        let want = gpu_steps(&m, &mut b, 9, 30, be);
        assert_eq!(bits(&got), bits(&want), "one-token steps after the checkpoint");
        // Chained, after another cut back.
        a.cut_to(c).unwrap();
        let mut b = base.clone();
        let (mut ga, mut gb) = (Vec::new(), Vec::new());
        let ka = decode_greedy_on(
            &m,
            &mut a,
            11,
            20,
            &mut |t| {
                ga.push(t);
                Ch::Go
            },
            be,
        );
        let kb = decode_greedy_on(
            &m,
            &mut b,
            11,
            20,
            &mut |t| {
                gb.push(t);
                Ch::Go
            },
            be,
        );
        assert_eq!((ka, &ga), (kb, &gb), "chained after the checkpoint");
    }

    /// A session decoded on the GPU, written to a snapshot and read into a fresh session (no
    /// device state: its copies are made from the host's at its first pass), decodes on the GPU
    /// as the original does, bit for bit.
    #[test]
    fn gpu_snapshot_round_trip() {
        use crate::snapshot::{SessionState, SnapReader, SnapWriter};
        let Some(be) = own_backend() else { return };
        let files = tiny();
        let m = DeepSeek4Model::load(super::super::tests::tiny_config(over()), &files, Quant::Bf16).unwrap();
        let mut a = prefilled(&m);
        gpu_steps(&m, &mut a, 5, 90, be);
        let path = std::env::temp_dir().join(format!("oxinfer-ds41-gpu-snap-{}.oxsnap", std::process::id()));
        let mut w = SnapWriter::new(std::fs::File::create(&path).unwrap()).unwrap();
        a.write_state(&mut w).unwrap();
        w.finish(serde_json::json!({})).unwrap();
        let mut b = m.new_session();
        let mut r = SnapReader::open(&path).unwrap();
        b.read_state(&mut r).unwrap();
        r.finish().unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(b.pos, a.pos);
        let got = gpu_steps(&m, &mut b, 7, 20, be);
        let want = gpu_steps(&m, &mut a, 7, 20, be);
        assert_eq!(bits(&got), bits(&want));
    }
}
