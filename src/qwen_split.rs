//! The two sides of a Qwen-path model split across two machines ([`crate::split`]): what the
//! front does with a pass after its own layers, and how the worker runs the rest.

use super::{QwenModel, QwenSession, VisionSpan};
use crate::config::QwenConfig;
use crate::split::{Client, Entry, Reply, Rows};
use anyhow::{Result, bail};
use std::sync::Arc;

/// What of a model a [`QwenModel`] holds.
pub(crate) enum Part {
    /// All of it.
    Whole,
    /// Its first layers: passes end with the residual stream, which the worker takes from
    /// there. `recurrent`: some layer of the whole model keeps a recurrent state.
    Front { client: Arc<Client>, recurrent: bool },
    /// Its last layers (on the worker): passes start from the rows the front sends.
    Tail,
}

/// Whether layers `range` of the model `cfg` describes can be one side of a split: dense layers
/// whose kind does not depend on anything outside the range.
pub(super) fn check(cfg: &QwenConfig, range: &std::ops::Range<usize>, front: bool) -> Result<()> {
    let total = cfg.num_hidden_layers;
    if range.start >= range.end || range.end > total || (front && range.start != 0) || (!front && range.end != total) {
        bail!(
            "split: layers {}..{} of {total} cannot be one side of a split",
            range.start,
            range.end
        );
    }
    let why = if cfg.is_exp() {
        "Qwen4-Exp"
    } else if cfg.lfm3.is_some() {
        "LFM3"
    } else if cfg.lightning {
        "MiniMax's lightning attention"
    } else if cfg.granite_scales().is_some() {
        "Granite 3's scales"
    } else if cfg.num_experts.is_some() {
        "mixture-of-experts layers"
    } else if cfg.bidirectional {
        "bidirectional attention"
    } else {
        return Ok(());
    };
    bail!("split: not supported for this model ({why}): dense Qwen, Qwen3.5 / 3.8 and Llama models only")
}

impl QwenModel<'_> {
    /// This is the first layers of a split model.
    pub(crate) fn is_front(&self) -> bool {
        matches!(self.part, Part::Front { .. })
    }

    /// The worker that runs the rest of this model, when it is the front of a split one.
    pub fn split_client(&self) -> Option<&Arc<Client>> {
        match &self.part {
            Part::Front { client, .. } => Some(client),
            _ => None,
        }
    }

    /// The front's own layers over `tokens`, through the entry the caller took.
    fn front_rows(&self, sess: &mut QwenSession, tokens: &[u32], mm: &[(usize, &[f32])], entry: Entry) -> Vec<f32> {
        if !sess.capture.is_empty() || sess.iv.active() || sess.positions.is_some() || sess.deepstack.is_some() {
            panic!("split: captures, interventions, explicit positions and DeepStack features are not supported for a split model");
        }
        match entry {
            Entry::Prefill => self.prefill_hidden_local(sess, tokens, mm),
            _ => self.forward_impl(sess, tokens, mm, false, true),
        }
    }

    /// A pass of a split model: this side's layers, then the worker's. With neither logits nor
    /// rows wanted, the worker's half is not waited for (its answer is read before the next one
    /// that is).
    pub(crate) fn split_pass(
        &self,
        sess: &mut QwenSession,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        entry: Entry,
        logits: bool,
        want: Rows,
    ) -> Reply {
        let (pos, t) = (sess.pos, tokens.len());
        let rows = self.front_rows(sess, tokens, mm, entry);
        let sp = sess.split.as_ref().expect("a front session");
        let r = sp.send_run(pos, t, &rows, &sess.spans, entry, logits, want).and_then(|()| {
            if !logits && want == Rows::None {
                return Ok(Reply {
                    rows: Vec::new(),
                    logits: Vec::new(),
                });
            }
            sp.recv()
        });
        r.unwrap_or_else(|e| panic!("split: the worker failed at position {pos}: {e:#}"))
    }

    /// One decode step of several sessions of a split model: each session's first layers run
    /// here and its row leaves for the worker at once, which runs it while this side is on the
    /// next session; the logits are then read in order.
    pub(crate) fn split_batch(&self, sess: &mut [&mut QwenSession], tokens: &[u32]) -> Vec<Vec<f32>> {
        for (s, &tok) in sess.iter_mut().zip(tokens) {
            let pos = s.pos;
            let rows = self.front_rows(s, &[tok], &[], Entry::Forward);
            let sp = s.split.as_ref().expect("a front session");
            if let Err(e) = sp.send_run(pos, 1, &rows, &s.spans, Entry::Forward, true, Rows::None) {
                panic!("split: the worker failed at position {pos}: {e:#}");
            }
        }
        sess.iter()
            .map(|s| {
                let sp = s.split.as_ref().expect("a front session");
                sp.recv()
                    .unwrap_or_else(|e| panic!("split: the worker failed at position {}: {e:#}", s.pos - 1))
                    .logits
            })
            .collect()
    }

    /// The LM head, which the worker holds.
    pub(crate) fn split_logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        let Part::Front { client, .. } = &self.part else {
            unreachable!("split_logits on a model that is not a front")
        };
        client
            .logits(&hidden[hidden.len() - t * self.cfg.hidden_size..], t)
            .unwrap_or_else(|e| panic!("split: the worker failed to compute logits: {e:#}"))
    }

    /// The worker's half of a pass: `rows` (`[t, hidden]`, the residual stream the front's
    /// layers ended with) from position `pos` through this model's layers, by the entry the
    /// front's pass took. Returns the final hidden states `want` names and, with `logits`, the
    /// logits after the last row. A session further than `pos` is cut back to it first.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tail_run(
        &self,
        sess: &mut QwenSession,
        pos: usize,
        rows: &[f32],
        t: usize,
        spans: &[VisionSpan],
        entry: Entry,
        logits: bool,
        want: Rows,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let h = self.cfg.hidden_size;
        if t == 0 || rows.len() != t * h {
            bail!("a pass of {t} tokens with {} floats (hidden size {h})", rows.len());
        }
        if sess.pos < pos {
            bail!(
                "a pass from position {pos}: the worker's session is at {} (was the worker restarted?)",
                sess.pos
            );
        }
        if sess.pos > pos {
            sess.cut_to(pos)?;
        }
        sess.spans = spans.to_vec();
        let keep = |hs: Vec<f32>| match want {
            Rows::None => Vec::new(),
            Rows::Last => hs[hs.len() - h..].to_vec(),
            Rows::All => hs,
        };
        // One token: the decode plan, as one machine's step takes it (the logits by its head).
        if t == 1
            && entry != Entry::Prefill
            && let Some((hs, l)) = self.gpu_step_x(sess, 0, Some(rows), logits)
        {
            let l = match l {
                Some(l) => l,
                None if logits => self.logits(&hs, 1),
                None => Vec::new(),
            };
            return Ok((keep(hs), l));
        }
        let toks = vec![0u32; t];
        let mm: Vec<(usize, &[f32])> = rows.chunks(h).enumerate().collect();
        let hs = match entry {
            Entry::Prefill => self.prefill_hidden_local(sess, &toks, &mm),
            _ => self.forward_impl(sess, &toks, &mm, false, true),
        };
        let l = if logits { self.logits(&hs[(t - 1) * h..], 1) } else { Vec::new() };
        Ok((keep(hs), l))
    }
}
