//! Multi-token prediction ("NextN") speculative decoding for Qwen3.5 / Qwen3.8.
//!
//! The checkpoints carry one extra decoder layer under `mtp.*`. For main-model position `i` it
//! takes the main model's final-normed hidden state `h_i` and the embedding of the *next* token
//! `x_{i+1}`, and predicts `x_{i+2}`:
//!
//! ```text
//! x   = fc([rmsnorm_emb(embed(x_{i+1})), rmsnorm_hidden(h_i)])     // fc: [H, 2H]
//! x   = decoder_layer(x, position i)                              // gated full attention + MLP
//! out = rmsnorm(x)                                                // also the next step's "hidden"
//! logits = lm_head(out)                                           // shared with the main model
//! ```
//!
//! (vLLM `qwen3_5_mtp.py` / `qwen3_next_mtp.py` and its EAGLE-style proposer: embedding first in
//! the concat, positions are those of the hidden state, the layer keeps its own KV cache over
//! every position, and further draft steps feed the layer its own output with the draft token.)
//!
//! Verification runs the main model over `[next, d1..dk]`; the Gated DeltaNet states are
//! recorded per token during that forward (`QwenModel::forward_hidden_rec`) so a partial
//! acceptance restores them exactly instead of re-running the accepted tokens. The MTP cache
//! drops the entries computed from draft hidden states, and the accepted rows (now with the main
//! model's hidden states) are run through the layer at the start of the next round, which also
//! yields its first draft.
//!
//! A trimmed draft head (FastMTP-style `[V_draft, H]` with a draft -> target id map) can replace
//! the shared LM head, and the layer weights may come from a separate file.

use crate::config::QwenConfig;
use crate::kernels::*;
use crate::model::{LayerCache, Lin, Quant, Table, linear};
use crate::qwen::{Attn, Ffn, Loader, QwenModel, QwenSession};
use crate::safetensors::ModelFiles;
use crate::sampler::Sampler;
use anyhow::{Result, bail};

/// A trimmed output head for drafting: logits over `d2t.len()` tokens, draft index `i` standing
/// for target token `d2t[i]`.
pub struct DraftHead<'a> {
    head: Lin<'a>, // [V_draft, H]
    d2t: Vec<u32>,
}

impl<'a> DraftHead<'a> {
    /// `head` is `[d2t.len(), hidden]`; every `d2t` entry must be a target token id.
    pub fn new(head: Lin<'a>, d2t: Vec<u32>, vocab_size: usize) -> Result<Self> {
        if d2t.is_empty() {
            bail!("empty draft vocabulary");
        }
        if let Some(&bad) = d2t.iter().find(|&&t| t as usize >= vocab_size) {
            bail!("draft token map entry {bad} is outside the target vocabulary ({vocab_size})");
        }
        Ok(Self { head, d2t })
    }

    /// Loads `name` (`[d2t.len(), hidden]`, any precision `load_lin` takes) from `files`.
    pub fn load(files: &'a ModelFiles, name: &str, d2t: Vec<u32>, cfg: &QwenConfig, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        let head = crate::model::load_lin(files, name, d2t.len(), cfg.hidden_size, quant)?;
        Self::new(head, d2t, cfg.vocab_size)
    }

    pub fn vocab(&self) -> usize {
        self.d2t.len()
    }
}

/// The MTP layer (weights only; the embeddings and LM head are the main model's).
pub struct QwenMtp<'a> {
    fc: Lin<'a>, // [H, 2H]
    norm_emb: Vec<f32>,
    norm_hidden: Vec<f32>,
    attn: Attn<'a>,
    ffn: Ffn<'a>,
    ln_in: Vec<f32>,
    ln_post: Vec<f32>,
    norm: Vec<f32>,
    head: Option<DraftHead<'a>>,
    /// The sidecar's own token embedding (FastMTP), used instead of the main model's.
    embed: Option<Table<'a>>,
    hidden: usize,
    /// Weight streaming: unit 0 is the layer, unit 1 the trimmed draft head.
    stream: Option<crate::stream::StreamHook>,
}

/// The MTP layer's own KV cache: one entry per main-model position whose successor is known.
pub struct MtpSession {
    kv: LayerCache,
}

impl MtpSession {
    pub fn len(&self) -> usize {
        self.kv.len
    }
    pub fn is_empty(&self) -> bool {
        self.kv.len == 0
    }
    pub fn truncate(&mut self, len: usize) {
        self.kv.truncate(len);
    }
}

/// Where the MTP tensors live in a checkpoint (`mtp.` in Qwen3.5 checkpoints), if anywhere.
pub fn mtp_prefix(files: &ModelFiles) -> Option<&'static str> {
    let has = |n: &str| files.has(n) || files.has(&format!("{n}.q8")) || files.has(&format!("{n}.q4"));
    ["mtp.", "model.mtp."].into_iter().find(|p| has(&format!("{p}fc.weight")))
}

impl<'a> QwenMtp<'a> {
    /// Loads the MTP layer for `cfg` from `files` (the main checkpoint, or a sidecar presenting
    /// the same HF tensor names). `quant` applies to bf16 weights; pre-quantized ones load as is.
    pub fn load(cfg: &QwenConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        if cfg.mtp_num_hidden_layers > 1 {
            bail!("{} MTP layers: only one is supported", cfg.mtp_num_hidden_layers);
        }
        let Some(p) = mtp_prefix(files) else {
            bail!("no MTP weights (mtp.fc.weight) in this checkpoint");
        };
        let quant = Quant::stored(files).unwrap_or(quant);
        let ld = Loader { files, cfg, quant };
        let h = cfg.hidden_size;
        let lp = format!("{p}layers.0.");
        Ok(Self {
            fc: ld.lin(&format!("{p}fc.weight"), h, 2 * h)?,
            norm_emb: ld.norm(&format!("{p}pre_fc_norm_embedding.weight"), h)?,
            norm_hidden: ld.norm(&format!("{p}pre_fc_norm_hidden.weight"), h)?,
            attn: ld.attn(&format!("{lp}self_attn."))?,
            ffn: ld.ffn(&format!("{lp}mlp."))?,
            ln_in: ld.norm(&format!("{lp}input_layernorm.weight"), h)?,
            ln_post: ld.norm(&format!("{lp}post_attention_layernorm.weight"), h)?,
            norm: ld.norm(&format!("{p}norm.weight"), h)?,
            head: None,
            embed: None,
            hidden: h,
            stream: None,
        })
    }

    /// The weights of each streaming unit: the layer (with `fc`), then the trimmed draft head.
    fn unit_lins(&self) -> Vec<(&'static str, Vec<&Lin<'a>>)> {
        let mut lins = vec![&self.fc];
        lins.extend(self.attn.lins());
        lins.extend(self.ffn.lins());
        let mut v = vec![("mtp", lins)];
        if let Some(d) = &self.head {
            v.push(("draft_head", vec![&d.head]));
        }
        v
    }

    /// The streaming units in execution order (see `crate::stream`): the layer (with `fc`), then
    /// the trimmed draft head if there is one (drafts otherwise use the main model's LM head).
    /// `draft_tokens` is how many times a round runs them (their residency priority).
    pub fn stream_units(&self, draft_tokens: usize) -> Vec<crate::stream::UnitSpec> {
        let uses = draft_tokens.max(1) as f64 + 0.5;
        let mut lins = vec![&self.fc];
        lins.extend(self.attn.lins());
        lins.extend(self.ffn.dense_lins());
        // The routed experts of a MoE MTP layer are read on demand.
        let mut v = vec![crate::stream::UnitSpec {
            name: "mtp".into(),
            ranges: lins.iter().flat_map(|l| l.mem_ranges()).collect(),
            uses,
            parts: self.ffn.expert_ranges(),
        }];
        if let Some(d) = &self.head {
            v.push(crate::stream::UnitSpec {
                name: "draft_head".into(),
                ranges: d.head.mem_ranges(),
                uses,
                parts: Vec::new(),
            });
        }
        v
    }

    /// See `QwenModel::mark_streamed`.
    pub fn mark_streamed(&self, streamed: &[bool]) {
        for ((_, lins), &s) in self.unit_lins().iter().zip(streamed) {
            for l in lins {
                l.set_transient_device(s);
            }
        }
    }

    /// Brackets the layer and the draft head with `hook`'s units (None stops streaming).
    pub fn set_stream(&mut self, hook: Option<crate::stream::StreamHook>) {
        self.stream = hook;
    }

    /// Takes what a draft sidecar adds besides the MTP layer (a FastMTP GGUF): its own token
    /// embedding, and a trimmed output head `lm_head.weight` `[n, H]` with `mtp.d2t` mapping its
    /// rows to target token ids. Returns whether a trimmed head was found.
    pub fn attach_sidecar(&mut self, cfg: &QwenConfig, files: &'a ModelFiles, quant: Quant) -> Result<bool> {
        if files.has("model.embed_tokens.weight") {
            self.embed = Some(Table::load(
                files,
                "model.embed_tokens.weight",
                cfg.vocab_size,
                cfg.hidden_size,
                quant,
            )?);
        }
        if !files.has("mtp.d2t") {
            return Ok(false);
        }
        let d2t: Vec<u32> = files
            .bytes("mtp.d2t")?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c) as u32)
            .collect();
        self.head = Some(DraftHead::load(files, "lm_head.weight", d2t, cfg, quant)?);
        Ok(true)
    }

    /// Drafts over a trimmed head instead of the main model's LM head (`None` restores it).
    pub fn set_draft_head(&mut self, head: Option<DraftHead<'a>>) {
        self.head = head;
    }

    pub fn draft_head(&self) -> Option<&DraftHead<'a>> {
        self.head.as_ref()
    }

    pub fn new_session(&self, cfg: &QwenConfig) -> MtpSession {
        MtpSession {
            kv: LayerCache::new(cfg.num_key_value_heads, cfg.head_dim),
        }
    }

    /// Weight bytes one draft step reads (layer plus head; the head is the shared LM head unless
    /// a trimmed one is set).
    pub fn step_bytes(&self, model: &QwenModel) -> usize {
        self.fc.bytes()
            + self.attn.bytes()
            + self.ffn.streamed_bytes()
            + self.head.as_ref().map_or(model.lm_head.bytes(), |d| d.head.bytes())
    }

    /// Runs `t` rows through the layer: row `i` pairs a hidden state (`hidden[i]`, `[t, H]`: the
    /// main model's final-normed state, or a previous MTP output) with the token after it, at
    /// position `sess.len() + i`. Appends to the MTP cache; returns the normed outputs `[t, H]`.
    pub fn forward(&self, model: &QwenModel, sess: &mut MtpSession, hidden: &[f32], tokens: &[u32]) -> Vec<f32> {
        let t = tokens.len();
        let h = self.hidden;
        assert!(t > 0 && hidden.len() == t * h);
        let cfg = &model.cfg;
        let eps = cfg.rms_norm_eps as f32;
        let mut cat = vec![0f32; t * 2 * h];
        let table = self.embed.as_ref().unwrap_or(&model.embed);
        if let Some(s) = self.stream.as_ref().or(model.stream.as_ref()) {
            let rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| table.row_ranges(tk as usize, h)).collect();
            s.need_ranges(&rows);
        }
        let mut e = vec![0f32; h];
        for (i, row) in cat.chunks_mut(2 * h).enumerate() {
            self.embed.as_ref().unwrap_or(&model.embed).row(tokens[i] as usize, &mut e);
            let (re, rh) = row.split_at_mut(h);
            rmsnorm_f32w(&e, &self.norm_emb, eps, re);
            rmsnorm_f32w(&hidden[i * h..(i + 1) * h], &self.norm_hidden, eps, rh);
        }
        let mut x = vec![0f32; t * h];
        let _unit = crate::stream::enter(&self.stream, 0);
        linear(&self.fc, h, 2 * h, &cat, t, &mut x);

        let start = sess.kv.len;
        let positions: Vec<usize> = (start..start + t).collect();
        let (cos, sin) = rope_tables(&positions, &model.inv_freq);
        let mut hn = vec![0f32; t * h];
        let mut y = vec![0f32; t * h];
        rmsnorm_f32w_rows(&x, h, &self.ln_in, eps, &mut hn);
        model.attn_forward(&self.attn, &mut sess.kv, &hn, t, start, &cos, &sin, &mut y);
        add_inplace(&mut x, &y);
        rmsnorm_f32w_rows(&x, h, &self.ln_post, eps, &mut hn);
        let need = |ids: &[usize]| {
            if let Some(s) = &self.stream {
                s.need_parts(0, ids)
            }
        };
        self.ffn
            .forward(&hn, t, h, &mut y, self.stream.as_ref().map(|_| &need as &dyn Fn(&[usize])));
        add_inplace(&mut x, &y);
        rmsnorm_f32w_rows(&x, h, &self.norm, eps, &mut hn);
        hn
    }

    /// The draft distribution over target token ids for one MTP output row, with the sampler's
    /// temperature / top-k / top-p (a single token with probability 1 when greedy).
    pub fn draft_distribution(&self, model: &QwenModel, out: &[f32], sampler: &Sampler) -> Vec<(u32, f32)> {
        let h = self.hidden;
        match &self.head {
            None => sampler.distribution(&model.logits(out, 1)),
            Some(d) => {
                let mut logits = vec![0f32; d.d2t.len()];
                let unit = crate::stream::enter(&self.stream, 1);
                linear(&d.head, d.d2t.len(), h, out, 1, &mut logits);
                drop(unit);
                sampler
                    .distribution(&logits)
                    .into_iter()
                    .map(|(i, p)| (d.d2t[i as usize], p))
                    .collect()
            }
        }
    }
}

impl<'a> QwenModel<'a> {
    /// Loads this model's MTP layer from `files` (usually the checkpoint the model came from; a
    /// sidecar with the same tensor names works too) at the model's precision.
    pub fn load_mtp<'b>(&self, files: &'b ModelFiles) -> Result<QwenMtp<'b>> {
        QwenMtp::load(&self.cfg, files, self.quant)
    }
}

/// The longest draft whose verification (`draft + 1` tokens) still runs the decode matmul
/// kernels, which give bit-identical rows whatever the token count; longer verifications use the
/// prefill GEMM, whose rounding differs, so greedy output can then drift from plain decoding at
/// near-ties (the distribution is unaffected beyond that rounding).
pub const MAX_EXACT_DRAFT: usize = 6;

/// The longest draft verified with the batch-invariant decode kernels under the current kernel
/// setting (`MAX_EXACT_DRAFT` unless `MtpOptions::exact` or `kernels::set_decode_max_tokens`
/// raised it).
pub fn max_exact_draft() -> usize {
    crate::kernels::decode_max_tokens() - 1
}

/// Longest draft the cost-aware policy considers.
pub const MAX_DRAFT: usize = 14;

/// Speculative decoding settings.
#[derive(Clone, Debug)]
pub struct MtpOptions {
    /// Maximum draft tokens per round (the MTP layer is applied recursively); at most
    /// `MAX_EXACT_DRAFT` for output identical to plain decoding.
    pub draft_tokens: usize,
    /// Grow the draft length after a fully accepted round and shrink it after a fully rejected
    /// one, between 1 and `draft_tokens`; otherwise always draft `draft_tokens`.
    pub adaptive: bool,
    /// Tokens that end generation: nothing after one is committed within a round.
    pub stop: Vec<u32>,
    /// Choose each round's draft length (1 to `draft_tokens`) to maximize expected tokens per
    /// second, from the measured time of a round at each length and the measured acceptance rate
    /// at each draft position (overrides `adaptive`). Pays off when a verification pass costs
    /// about the same for 1 or 9 tokens, as when the weights stream from storage.
    pub cost_aware: bool,
    /// Keep verification bit-exact for drafts longer than `MAX_EXACT_DRAFT` by running up to
    /// `draft_tokens + 1` tokens through the decode kernels (raises the process-wide
    /// `kernels::set_decode_max_tokens`, which never lowers it).
    pub exact: bool,
}

impl Default for MtpOptions {
    fn default() -> Self {
        Self {
            draft_tokens: 3,
            adaptive: false,
            stop: Vec::new(),
            cost_aware: false,
            exact: false,
        }
    }
}

/// Measurements behind the cost-aware draft length.
#[derive(Clone, Debug, Default)]
pub struct DraftCosts {
    /// `round_secs[k]`: moving average of a round's wall time (drafting + verification) with `k`
    /// drafts, and how many rounds it saw.
    pub round_secs: Vec<(f64, usize)>,
    /// Moving average of one recursive draft step (MTP layer + head over one row).
    pub step_secs: f64,
    /// `tried[j]` / `ok[j]`: rounds whose draft reached position `j`, and those that accepted it.
    pub tried: Vec<usize>,
    pub ok: Vec<usize>,
}

impl DraftCosts {
    fn record(&mut self, k: usize, secs: f64, accepted: usize, step_secs: Option<f64>) {
        let ema = |old: f64, n: usize, v: f64| if n == 0 { v } else { 0.7 * old + 0.3 * v };
        if self.round_secs.len() <= k {
            self.round_secs.resize(k + 1, (0.0, 0));
        }
        let (t, n) = self.round_secs[k];
        self.round_secs[k] = (ema(t, n, secs), n + 1);
        if let Some(s) = step_secs {
            self.step_secs = ema(self.step_secs, (self.step_secs > 0.0) as usize, s);
        }
        if self.tried.len() < k {
            self.tried.resize(k, 0);
            self.ok.resize(k, 0);
        }
        for j in 0..k.min(accepted + 1) {
            self.tried[j] += 1;
            self.ok[j] += (j < accepted) as usize;
        }
    }

    /// Estimated round time with `k` drafts: measured, else the nearest measured length plus
    /// (or minus) the draft steps in between. Unmeasured lengths get no verification cost
    /// growth, which makes the policy try them.
    fn round_time(&self, k: usize) -> Option<f64> {
        if let Some(&(t, n)) = self.round_secs.get(k)
            && n > 0
        {
            return Some(t);
        }
        let (k0, &(t0, _)) = self
            .round_secs
            .iter()
            .enumerate()
            .filter(|(_, e)| e.1 > 0)
            .min_by_key(|(j, _)| j.abs_diff(k))?;
        Some((t0 + (k as f64 - k0 as f64) * self.step_secs).max(1e-6))
    }

    /// Expected committed tokens per round with `k` drafts: 1 + sum over i of the probability
    /// that the first i drafts are all accepted (Laplace-smoothed per-position rates).
    fn expected_tokens(&self, k: usize) -> f64 {
        let mut e = 1.0;
        let mut p = 1.0;
        for j in 0..k {
            let (t, o) = (self.tried.get(j).copied().unwrap_or(0), self.ok.get(j).copied().unwrap_or(0));
            p *= (o as f64 + 1.0) / (t as f64 + 2.0);
            e += p;
        }
        e
    }

    /// The draft length in `1..=k_max` with the most expected tokens per second.
    pub fn best(&self, k_max: usize) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for k in 1..=k_max.max(1) {
            let rate = self.expected_tokens(k) / self.round_time(k)?;
            if best.is_none_or(|b| rate > b.1) {
                best = Some((k, rate));
            }
        }
        best.map(|b| b.0)
    }
}

#[derive(Clone, Debug, Default)]
pub struct MtpStats {
    /// Verification forwards of the main model.
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
    /// Tokens committed by `step` (accepted drafts plus one per round).
    pub emitted: usize,
    /// `per_round[n]`: rounds that accepted `n` drafts.
    pub per_round: Vec<usize>,
    /// Time spent drafting (MTP layer and head) and verifying (main model), in seconds.
    pub draft_secs: f64,
    pub verify_secs: f64,
    /// Inputs of the cost-aware draft policy.
    pub costs: DraftCosts,
}

impl MtpStats {
    pub fn acceptance(&self) -> f64 {
        self.accepted as f64 / self.drafted.max(1) as f64
    }
    pub fn tokens_per_round(&self) -> f64 {
        self.emitted as f64 / self.rounds.max(1) as f64
    }
}

/// Speculative generation with the MTP layer. Exactly the plain decoding distribution: greedy
/// produces the plain greedy tokens (bit for bit: the verification forward computes the same
/// logits as one-token forwards), sampling draws every token from the same
/// temperature / top-k / top-p distribution `Sampler::sample` would (standard speculative
/// sampling: accept a draft `d` with probability `min(1, p(d) / q(d))`, else draw from the
/// normalized `max(0, p - q)`; the draft distribution `q` is the MTP logits under the same
/// sampler settings). Sampled output is not the same token sequence as plain sampling with the
/// same seed, since the random draws are used differently.
///
/// Usage: `prefill` the prompt, choose the first token from the returned logits and `push` it,
/// then call `step` until a stop token or the length limit; `prefill` again for a later turn.
pub struct MtpDecoder<'m> {
    model: &'m QwenModel<'m>,
    mtp: &'m QwenMtp<'m>,
    pub sess: QwenSession,
    msess: MtpSession,
    pub opts: MtpOptions,
    /// Rows not yet in the MTP cache: main hidden states `[r, H]` and the tokens after them
    /// (`r` or `r - 1` of them: the last row waits for its token).
    pend_h: Vec<f32>,
    pend_t: Vec<u32>,
    /// The committed token the main model has not processed yet.
    next: Option<u32>,
    k_cur: usize,
    pub stats: MtpStats,
}

impl<'m> MtpDecoder<'m> {
    pub fn new(model: &'m QwenModel<'m>, mtp: &'m QwenMtp<'m>, opts: MtpOptions) -> Self {
        let opts = MtpOptions {
            draft_tokens: if opts.cost_aware {
                opts.draft_tokens.min(MAX_DRAFT)
            } else {
                opts.draft_tokens
            },
            ..opts
        };
        if opts.exact && opts.draft_tokens + 1 > crate::kernels::decode_max_tokens() {
            crate::kernels::set_decode_max_tokens(opts.draft_tokens + 1);
        }
        let k_cur = if opts.adaptive || opts.cost_aware {
            opts.draft_tokens.clamp(1, 3)
        } else {
            opts.draft_tokens
        };
        Self {
            model,
            mtp,
            sess: model.new_session(),
            msess: mtp.new_session(&model.cfg),
            opts,
            pend_h: Vec::new(),
            pend_t: Vec::new(),
            next: None,
            k_cur,
            stats: MtpStats::default(),
        }
    }

    pub fn model(&self) -> &'m QwenModel<'m> {
        self.model
    }

    pub fn mtp(&self) -> &'m QwenMtp<'m> {
        self.mtp
    }

    /// Current draft length.
    pub fn draft_len(&self) -> usize {
        self.k_cur
    }

    /// Runs every pending row whose token is known through the MTP layer; returns the last
    /// output row (`None` if there was nothing to run).
    fn flush(&mut self) -> Option<Vec<f32>> {
        let n = self.pend_t.len();
        if n == 0 {
            return None;
        }
        let h = self.model.cfg.hidden_size;
        let out = self.mtp.forward(self.model, &mut self.msess, &self.pend_h[..n * h], &self.pend_t);
        self.pend_h.drain(..n * h);
        self.pend_t.clear();
        Some(out[(n - 1) * h..].to_vec())
    }

    /// Appends `tokens` (a prompt, or the next turn of a conversation) in chunks of `chunk` and
    /// returns the logits after the last one. A committed token that was never processed (the
    /// stop token ending the previous reply) is dropped, as plain decoding drops it.
    pub fn prefill(&mut self, tokens: &[u32], chunk: usize) -> Vec<f32> {
        assert!(!tokens.is_empty());
        let h = self.model.cfg.hidden_size;
        if self.next.take().is_some() {
            self.pend_t.pop();
        }
        let mut last = Vec::new();
        for c in tokens.chunks(chunk.max(1)) {
            if self.pend_h.len() > self.pend_t.len() * h {
                self.pend_t.push(c[0]);
            }
            let hs = self.model.forward_hidden_mm(&mut self.sess, c, &[]);
            self.pend_h.extend_from_slice(&hs);
            self.pend_t.extend_from_slice(&c[1..]);
            self.flush();
            last = hs[hs.len() - h..].to_vec();
        }
        self.model.logits(&last, 1)
    }

    /// Commits the token chosen from `prefill`'s logits (the first token of a reply).
    pub fn push(&mut self, token: u32) {
        assert!(self.next.is_none(), "push follows prefill; step commits its own tokens");
        assert_eq!(self.pend_h.len(), (self.pend_t.len() + 1) * self.model.cfg.hidden_size);
        self.pend_t.push(token);
        self.next = Some(token);
    }

    /// One speculative round: drafts up to the current draft length (and at most
    /// `max_new - 1`), verifies them with one main-model forward and returns the committed
    /// tokens (1 to `draft + 1` of them, ending at the first stop token if any).
    pub fn step(&mut self, sampler: &mut Sampler, max_new: usize) -> Vec<u32> {
        let next = self.next.expect("push a first token before step");
        let (model, mtp) = (self.model, self.mtp);
        let h = model.cfg.hidden_size;
        let v = model.cfg.vocab_size;
        let n = self.sess.pos; // position of `next`
        let k = self.k_cur.min(max_new.max(1) - 1);

        // Draft: the pending rows end with (h_{n-1}, next), whose output gives the first draft.
        let t_round = crate::clock::Instant::now();
        let mut step_secs = None;
        let mut drafts = Vec::with_capacity(k);
        let mut qs: Vec<Vec<(u32, f32)>> = Vec::with_capacity(k);
        if k > 0 {
            let mut g = self.flush().expect("pending MTP row");
            debug_assert_eq!(self.msess.len(), n);
            for j in 0..k {
                let q = mtp.draft_distribution(model, &g, sampler);
                let d = if sampler.greedy() { q[0].0 } else { sampler.sample_from(&q) };
                drafts.push(d);
                qs.push(q);
                if j + 1 < k {
                    let t = crate::clock::Instant::now();
                    g = mtp.forward(model, &mut self.msess, &g, &[d]);
                    if j == 0 {
                        step_secs = Some(t.elapsed().as_secs_f64());
                    }
                }
            }
        }
        let draft_secs = t_round.elapsed().as_secs_f64();

        // Verify [next, drafts..] in one forward, recording the recurrent states for rewinding.
        let mut seq = Vec::with_capacity(k + 1);
        seq.push(next);
        seq.extend_from_slice(&drafts);
        let check = if std::env::var_os("OXINFER_CHECK_SPEC").is_some() {
            Some(self.sess.clone())
        } else {
            None
        };
        let hs = if k > 0 {
            model.forward_hidden_rec(&mut self.sess, &seq)
        } else {
            model.forward_hidden_mm(&mut self.sess, &seq, &[])
        };
        let logits = model.logits(&hs, seq.len());
        let verify_secs = t_round.elapsed().as_secs_f64() - draft_secs;
        if let Some(mut s1) = check {
            // Debug: the batched verification must match one-token forwards exactly.
            for (i, &tk) in seq.iter().enumerate() {
                let l1 = model.forward(&mut s1, &[tk]);
                let lk = &logits[i * v..(i + 1) * v];
                let diff = l1.iter().zip(lk).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                if diff > 0.0 {
                    let (a1, ak) = (crate::sampler::argmax(&l1), crate::sampler::argmax(lk));
                    eprintln!(
                        "[check: pos {} of {}: max logit diff {diff:e}, argmax t=1 {a1} vs t={} {ak}]",
                        n + i,
                        seq.len(),
                        seq.len()
                    );
                }
            }
        }

        let mut out = Vec::with_capacity(k + 1);
        let mut accepted = 0;
        for (j, (&d, q)) in drafts.iter().zip(&qs).enumerate() {
            let p = sampler.distribution(&logits[j * v..(j + 1) * v]);
            match speculative_accept(sampler, &p, q, d) {
                None => {
                    out.push(d);
                    accepted += 1;
                    if self.opts.stop.contains(&d) {
                        break;
                    }
                }
                Some(t) => {
                    out.push(t);
                    break;
                }
            }
        }
        if accepted == drafts.len() && !out.last().is_some_and(|t| self.opts.stop.contains(t)) {
            out.push(sampler.sample(&logits[k * v..(k + 1) * v]));
        }
        if let Some(i) = out.iter().position(|t| self.opts.stop.contains(t)) {
            out.truncate(i + 1);
        }

        // Commit: the main model keeps [next, out[..c-1]]; out[c-1] is the new `next`.
        let c = out.len();
        self.sess.rewind(n + c).expect("rewind inside the recorded span");
        // Drop the entries computed from draft hidden states. The pending rows are empty after
        // drafting (or, with no drafts, still hold rows that end with `next`): append the new ones.
        self.msess.truncate(n);
        self.pend_h.extend_from_slice(&hs[..c * h]);
        self.pend_t.extend_from_slice(&out);
        self.next = out.last().copied();

        self.stats.rounds += 1;
        self.stats.drafted += k;
        self.stats.accepted += accepted;
        self.stats.emitted += c;
        if self.stats.per_round.len() <= accepted {
            self.stats.per_round.resize(accepted + 1, 0);
        }
        self.stats.per_round[accepted] += 1;
        self.stats.draft_secs += draft_secs;
        self.stats.verify_secs += verify_secs;
        if k == self.k_cur && k > 0 {
            // A draft step's own cost (the head's share included) when the first step was
            // not timed: the drafting time spread over the steps.
            let step = step_secs.or(Some(draft_secs / k as f64));
            self.stats.costs.record(k, draft_secs + verify_secs, accepted, step);
        }
        if self.opts.cost_aware {
            if let Some(b) = self.stats.costs.best(self.opts.draft_tokens.max(1)) {
                self.k_cur = b;
            }
        } else if self.opts.adaptive && k == self.k_cur {
            if accepted == k {
                self.k_cur = (self.k_cur + 1).min(self.opts.draft_tokens.max(1));
            } else if accepted == 0 {
                self.k_cur = self.k_cur.saturating_sub(1).max(1);
            }
        }
        out
    }
}

/// The speculative sampling test for a draft `d` drawn from `q`, against the target `p` (both as
/// `Sampler::distribution` returns them): `None` accepts `d` (probability `min(1, p(d) / q(d))`),
/// `Some(t)` rejects it in favour of `t` drawn from the normalized `max(0, p - q)`. Either way the
/// resulting token is distributed exactly as `p`. With greedy distributions this accepts `d` iff
/// it is the target argmax, and otherwise returns the argmax.
pub fn speculative_accept(sampler: &mut Sampler, p: &[(u32, f32)], q: &[(u32, f32)], d: u32) -> Option<u32> {
    let pd = p.iter().find(|e| e.0 == d).map_or(0.0, |e| e.1);
    let qd = q.iter().find(|e| e.0 == d).map_or(0.0, |e| e.1);
    if pd >= qd || sampler.uniform() < pd / qd {
        return None;
    }
    let qm: std::collections::HashMap<u32, f32> = q.iter().copied().collect();
    let r: Vec<(u32, f32)> = p
        .iter()
        .map(|&(t, pt)| (t, (pt - qm.get(&t).copied().unwrap_or(0.0)).max(0.0)))
        .filter(|e| e.1 > 0.0)
        .collect();
    Some(sampler.sample_from(if r.is_empty() { p } else { &r }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_sampling_preserves_target() {
        // Target p and a deliberately different draft q (with a token p never produces).
        let p = [(0u32, 0.5f32), (1, 0.3), (2, 0.2)];
        let q = [(1u32, 0.5f32), (3, 0.3), (0, 0.2)];
        let mut s = Sampler::new(1.0, 0, 1.0, 7);
        let n = 400_000;
        let mut hist = [0usize; 4];
        for _ in 0..n {
            let d = s.sample_from(&q);
            let t = speculative_accept(&mut s, &p, &q, d).unwrap_or(d);
            hist[t as usize] += 1;
        }
        for &(t, pt) in &p {
            let f = hist[t as usize] as f32 / n as f32;
            assert!((f - pt).abs() < 0.005, "token {t}: {f} vs {pt}");
        }
        assert_eq!(hist[3], 0);
    }

    /// Verification reproduces plain decoding only if a matmul row does not depend on how many
    /// tokens share the pass (up to the 7 rows of a 6-token draft).
    #[test]
    fn small_batch_matmuls_are_batch_invariant() {
        use crate::quant::{quantize_bf16, quantize_q4};
        let (n_out, n_in) = (70, 512);
        let mut seed = 1u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };
        let w: Vec<u16> = (0..n_out * n_in).map(|_| (rnd().to_bits() >> 16) as u16).collect();
        let x: Vec<f32> = (0..7 * n_in).map(|_| rnd()).collect();
        let lins = [
            ("bf16", Lin::bf16(&w)),
            ("q8", Lin::q8(quantize_bf16(&w, n_out, n_in))),
            ("q4", Lin::q4(quantize_q4(&w, n_out, n_in))),
        ];
        for (name, l) in &lins {
            let mut y1 = vec![0f32; 7 * n_out];
            for (xr, yr) in x.chunks(n_in).zip(y1.chunks_mut(n_out)) {
                linear(l, n_out, n_in, xr, 1, yr);
            }
            for t in 2..=7 {
                let mut yt = vec![0f32; t * n_out];
                linear(l, n_out, n_in, &x[..t * n_in], t, &mut yt);
                let same = yt.iter().zip(&y1).all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "{name}: t={t} differs from one-token matmuls");
            }
        }
    }

    /// The decode kernels stay batch-invariant past 7 tokens (for exact long drafts), GGUF
    /// k-quants included.
    #[test]
    fn long_verification_is_batch_invariant() {
        use crate::ggml::{GgmlType, GgufMatrix, random_blocks};
        use crate::kernels::matmul_quant_multi_at;
        let (n_out, n_in) = (21, 512);
        let mut seed = 5u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..15 * n_in).map(|_| rnd()).collect();
        let mut s2 = 9u64;
        for kind in [GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::Q2_K, GgmlType::IQ4_XS] {
            let raw = random_blocks(kind, n_out * n_in / kind.block_size(), &mut s2);
            let m = GgufMatrix::from_bytes(kind, &raw, n_out, n_in).unwrap();
            let mut y1 = vec![0f32; 15 * n_out];
            for (xr, yr) in x.chunks(n_in).zip(y1.chunks_mut(n_out)) {
                matmul_quant_multi_at(&mut [(&m, yr)], xr, 1, 15);
            }
            for t in 8..=15 {
                let mut yt = vec![0f32; t * n_out];
                matmul_quant_multi_at(&mut [(&m, &mut yt[..])], &x[..t * n_in], t, 15);
                let same = yt.iter().zip(&y1).all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "{kind}: t={t} differs from one-token matmuls");
            }
        }
    }

    #[test]
    fn cost_aware_policy_prefers_long_drafts_when_verification_is_flat() {
        // Each draft accepted with probability 0.7.
        let mut seed = 3u64;
        let mut accept = |k: usize| {
            (0..k)
                .take_while(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (seed >> 33) % 10 < 7
                })
                .count()
        };
        let mut c = DraftCosts::default();
        // Verification costs 1 s whatever the length, a draft step 10 ms.
        for round in 0..400 {
            let k = 1 + round % 8;
            c.record(k, 1.0 + 0.01 * k as f64, accept(k), Some(0.01));
        }
        assert!(c.best(8).unwrap() >= 6, "{:?}", c.best(8));
        // Verification time proportional to the tokens: one draft is best.
        let mut c = DraftCosts::default();
        for round in 0..400 {
            let k = 1 + round % 8;
            c.record(k, (k + 1) as f64, accept(k), Some(0.01));
        }
        assert_eq!(c.best(8), Some(1));
    }

    #[test]
    fn greedy_accepts_only_argmax() {
        let mut s = Sampler::new(0.0, 0, 1.0, 1);
        let p = [(5u32, 1.0f32)];
        assert_eq!(speculative_accept(&mut s, &p, &[(5, 1.0)], 5), None);
        assert_eq!(speculative_accept(&mut s, &p, &[(4, 1.0)], 4), Some(5));
    }
}
