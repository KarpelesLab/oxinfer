//! Batched utterances (a submodule of [`crate::tts`]): several texts, each with its own voice
//! and options, generated together. Every decode step feeds the next token of each running
//! utterance as one column of one pass through the language model
//! ([`crate::qwen::QwenModel::forward_hidden_batch`]: per-utterance KV caches, the weights read
//! once for all), and the SNAC decoder runs once over all the utterances
//! ([`crate::snac::Snac::decode_batch`]).
//!
//! Each utterance keeps its own sampler state, repetition-penalty set, loop guard, length caps
//! and stop: it leaves the batch when it ends, and the others go on. The state machine
//! ([`Voice`]) is the one [`Speaker::speak`] runs, and the batched passes give each column the
//! bits of a one-token pass on the CPU, so an utterance of a batch is the utterance spoken
//! alone with the same options (same tokens, same samples).

use super::*;
use crate::model::linear;
use crate::sampler::{Sampler, apply_repetition_penalty};

/// One utterance of [`Speaker::speak_batch`]: a text and its options (voice, sampling, seed,
/// limits; the streaming fields are unused).
#[derive(Clone, Copy, Debug)]
pub struct Utterance<'a> {
    pub text: &'a str,
    pub opts: &'a SpeakOptions,
}

impl<'a> Utterance<'a> {
    pub fn new(text: &'a str, opts: &'a SpeakOptions) -> Self {
        Self { text, opts }
    }
}

/// Timings and counts of a whole batch ([`Speaker::speak_batch_detailed`]).
#[derive(Clone, Debug, Default, Serialize)]
pub struct BatchStats {
    pub utterances: usize,
    /// Generated tokens of all utterances, stop tokens included.
    pub tokens: usize,
    /// Passes through the language model after the prefill (one token of every running
    /// utterance each).
    pub steps: usize,
    /// Most utterances one pass carries ([`crate::qwen::QwenModel::batch_width`]; a wider batch
    /// takes several passes per step).
    pub width: usize,
    pub audio_secs: f64,
    /// The prompts, one after another.
    pub prefill_secs: f64,
    pub generate_secs: f64,
    pub snac_secs: f64,
    pub total_secs: f64,
}

impl BatchStats {
    /// Generated tokens per second over all utterances (after the prefill).
    pub fn tok_s(&self) -> f64 {
        self.tokens as f64 / self.generate_secs.max(1e-9)
    }
    /// Real-time factor of the batch: total time over the audio produced.
    pub fn rtf(&self) -> f64 {
        self.total_secs / self.audio_secs.max(1e-9)
    }
}

/// The decoding state of one utterance: what [`Speaker::speak`] and a batch both step.
pub(super) struct Voice<'o> {
    opts: &'o SpeakOptions,
    sampler: Sampler,
    /// Loop guard: the sampler once a loop of identical frames was seen.
    fallback_sampler: Sampler,
    fallback: bool,
    /// Frames equal to the one before them, in a row.
    same: usize,
    max_frames: usize,
    /// Bit set of the tokens seen (prompt and reply), for the repetition penalty.
    seen: Vec<u64>,
    constrained: bool,
    pub tokens: Vec<u32>,
    pub frames: Vec<[u32; 7]>,
    cur: [u32; 7],
    fill: usize,
    pub finish: StopReason,
    pub st: SpeakStats,
}

impl<'o> Voice<'o> {
    /// The state before the first token, for the prompt `ids` of `text`.
    pub fn new(sp: &Speaker, text: &str, opts: &'o SpeakOptions, ids: &[u32]) -> Self {
        let rate = sp.snac.sample_rate();
        let max_frames = if opts.max_secs_per_byte > 0.0 {
            let secs = 3.0 + text.len() as f64 * opts.max_secs_per_byte as f64;
            (secs * rate as f64 / sp.snac.samples_per_frame() as f64).ceil() as usize
        } else {
            usize::MAX
        };
        let mut v = Self {
            opts,
            sampler: Sampler::new(opts.temperature, opts.top_k, opts.top_p, opts.seed),
            fallback_sampler: Sampler::new(
                opts.temperature.max(0.6),
                opts.top_k,
                if opts.top_p > 0.0 { opts.top_p.min(0.8) } else { 0.8 },
                opts.seed ^ 0x6c6f_6f70,
            ),
            fallback: false,
            same: 0,
            max_frames,
            seen: vec![0u64; sp.model.cfg.vocab_size.div_ceil(64)],
            constrained: opts.constrained && sp.head.is_some(),
            tokens: Vec::new(),
            frames: Vec::new(),
            cur: [0; 7],
            fill: 0,
            finish: StopReason::Length,
            st: SpeakStats {
                prompt_tokens: ids.len(),
                ..Default::default()
            },
        };
        for &t in ids {
            v.mark(t);
        }
        v
    }

    fn mark(&mut self, t: u32) {
        if let Some(w) = self.seen.get_mut(t as usize / 64) {
            *w |= 1 << (t % 64);
        }
    }

    /// Whether the next token is sampled from the whole vocabulary (then [`Voice::sample`]
    /// takes the output layer's logits when the caller has them).
    pub fn wants_full_logits(&self) -> bool {
        !self.constrained
    }

    /// Samples the next token from the last hidden state (`full`: the whole output layer's
    /// logits of it, when already computed and sampling is unconstrained).
    pub fn sample(&mut self, sp: &Speaker, hidden: &[f32], full: Option<Vec<f32>>) -> Result<u32> {
        let penalty = self.opts.repetition_penalty;
        let h = sp.model.cfg.hidden_size;
        // Logits and the token ids they stand for.
        let (logits, base): (Vec<f32>, Option<u32>) = match sp.head.as_ref().filter(|_| self.constrained) {
            Some(hd) => {
                let mut l = vec![0f32; CODES as usize + STOP_TOKENS.len()];
                let (a, b) = l.split_at_mut(CODES as usize);
                linear(&hd.pos[self.fill], CODES as usize, h, hidden, 1, a);
                linear(&hd.stop, STOP_TOKENS.len(), h, hidden, 1, b);
                let base = AUDIO_BASE + self.fill as u32 * CODES;
                if penalty != 1.0 {
                    let seen = &self.seen;
                    let is_seen = |t: u32| seen.get(t as usize / 64).is_some_and(|w| (w >> (t % 64)) & 1 == 1);
                    for (v, t) in l.iter_mut().zip((base..base + CODES).chain(STOP_TOKENS)) {
                        if is_seen(t) {
                            *v = if *v < 0.0 { *v * penalty } else { *v / penalty };
                        }
                    }
                }
                (l, Some(base))
            }
            None => {
                let mut l = full.unwrap_or_else(|| sp.model.logits(hidden, 1));
                if penalty != 1.0 {
                    apply_repetition_penalty(&mut l, &self.seen, penalty);
                }
                (l, None)
            }
        };
        if logits.iter().any(|v| v.is_nan()) {
            bail!("the model produced NaN logits");
        }
        let k = if self.fallback {
            self.fallback_sampler.sample(&logits)
        } else {
            self.sampler.sample(&logits)
        };
        Ok(match base {
            Some(b) if k < CODES => b + k,
            Some(_) => STOP_TOKENS[(k - CODES) as usize],
            None => k,
        })
    }

    /// Takes the sampled token `id`. `on_frame` sees the frames whenever one completes
    /// (streaming; [`Flow::Stop`] cancels). Returns whether the utterance goes on.
    pub fn accept(&mut self, id: u32, on_frame: &mut dyn FnMut(&[[u32; 7]]) -> Result<Flow>) -> Result<bool> {
        self.tokens.push(id);
        self.mark(id);
        if STOP_TOKENS.contains(&id) {
            self.finish = StopReason::Stop;
            return Ok(false);
        }
        let pos_base = AUDIO_BASE + self.fill as u32 * CODES;
        if !(pos_base..pos_base + CODES).contains(&id) {
            self.st.dropped_tokens += 1;
            return Ok(true);
        }
        self.cur[self.fill] = id - pos_base;
        self.fill += 1;
        if self.fill < FRAME_TOKENS {
            return Ok(true);
        }
        self.same = if self.frames.last() == Some(&self.cur) { self.same + 1 } else { 0 };
        self.frames.push(self.cur);
        self.fill = 0;
        let mut ended = self.frames.len() >= self.max_frames;
        if self.opts.loop_guard > 0 && self.same >= self.opts.loop_guard {
            if self.st.loops >= MAX_LOOPS {
                ended = true;
            } else {
                self.st.loops += 1;
                (self.fallback, self.same) = (true, 0);
            }
        }
        if on_frame(&self.frames)? == Flow::Stop {
            self.finish = StopReason::Cancelled;
            return Ok(false);
        }
        // Length cap or a loop that did not end: like max_tokens.
        Ok(!ended)
    }
}

/// Restores the process-wide row limit of the decode kernels when dropped.
struct RestoreDecodeMax(usize);
impl Drop for RestoreDecodeMax {
    fn drop(&mut self) {
        crate::kernels::set_decode_max_tokens(self.0);
    }
}

impl Speaker {
    /// Speaks several utterances together (see the module docs): `out[i]` is what
    /// [`Speaker::speak_detailed`] returns for `items[i]` alone, generated in shared passes.
    /// The equality is exact on the CPU; with GPU decoding a lone utterance is generated on the
    /// device and a batch on the CPU kernels, so the samples drawn can part.
    ///
    /// [`Speaker::cancel_handle`] stops every running utterance (their audio so far is returned).
    /// The timings of each [`Speech::stats`] are the utterance's own prefill, the time from the
    /// first batched step to its last token, and the shared SNAC pass.
    pub fn speak_batch(&self, items: &[Utterance<'_>]) -> Result<Vec<Speech>> {
        Ok(self.speak_batch_detailed(items)?.0)
    }

    /// [`Speaker::speak_batch`] with the batch's statistics.
    pub fn speak_batch_detailed(&self, items: &[Utterance<'_>]) -> Result<(Vec<Speech>, BatchStats)> {
        let t0 = Instant::now();
        self.cancel.store(false, Ordering::SeqCst);
        let refs: Vec<Vec<VoiceRef>> = items
            .iter()
            .map(|u| self.resolve_references(&u.opts.references))
            .collect::<Result<_>>()?;
        let _run = RUN.lock().unwrap_or_else(|p| p.into_inner());
        let h = self.model.cfg.hidden_size;
        let rate = self.snac.sample_rate();
        let n = items.len();
        let mut bs = BatchStats {
            utterances: n,
            ..Default::default()
        };

        // The prompts, one sequence at a time (as alone: the same prefill passes).
        let mut voices = Vec::with_capacity(n);
        let mut sessions: Vec<QwenSession> = Vec::with_capacity(n);
        let mut hidden = Vec::with_capacity(n * h);
        for (u, refs) in items.iter().zip(&refs) {
            let tp = Instant::now();
            let voice = u.opts.voice.as_deref().filter(|_| !self.info.pretrained);
            let ids = self.prompt_tokens_with(u.text, voice, refs)?;
            let mut v = Voice::new(self, u.text, u.opts, &ids);
            let mut sess = self.model.new_session();
            let hs = crate::generate::prefill(&self.model, &mut sess, &ids, 256, &[]);
            hidden.extend_from_slice(&hs[hs.len() - h..]);
            v.st.prefill_secs = tp.elapsed().as_secs_f64();
            voices.push(v);
            sessions.push(sess);
        }
        bs.prefill_secs = t0.elapsed().as_secs_f64();

        // Quantized weights: up to 15 columns on the int8 decode kernels (7 by default), whose
        // columns keep the bits of a one-token pass. Process-wide, restored afterwards; set
        // after the prefill, whose short prompts must take the same kernels as alone.
        let restore = RestoreDecodeMax(crate::kernels::decode_max_tokens());
        if n > restore.0 {
            crate::kernels::set_decode_max_tokens(n);
        }
        bs.width = self.model.batch_width().max(1);

        let tg = Instant::now();
        let mut done_at = vec![0f64; n];
        // The running utterances; row `k` of `hidden` is the last hidden state of `live[k]`.
        let mut live: Vec<usize> = (0..n).collect();
        while !live.is_empty() {
            // The whole output layer for the unconstrained ones, in shared passes.
            let full: Vec<usize> = (0..live.len())
                .filter(|&k| {
                    let v = &voices[live[k]];
                    v.wants_full_logits() && v.tokens.len() < v.opts.max_tokens
                })
                .collect();
            let mut full_logits: Vec<Option<Vec<f32>>> = vec![None; live.len()];
            let vocab = self.model.cfg.vocab_size;
            for ks in full.chunks(bs.width) {
                if ks.len() < 2 {
                    continue; // alone: `Voice::sample` computes it
                }
                let rows: Vec<f32> = ks.iter().flat_map(|&k| hidden[k * h..(k + 1) * h].iter().copied()).collect();
                let l = self.model.logits(&rows, ks.len());
                for (j, &k) in ks.iter().enumerate() {
                    full_logits[k] = Some(l[j * vocab..(j + 1) * vocab].to_vec());
                }
            }
            let mut next = Vec::with_capacity(live.len());
            let mut toks = Vec::with_capacity(live.len());
            for (k, &i) in live.iter().enumerate() {
                let v = &mut voices[i];
                let mut go = v.tokens.len() < v.opts.max_tokens;
                if go {
                    let id = v.sample(self, &hidden[k * h..(k + 1) * h], full_logits[k].take())?;
                    go = v.accept(id, &mut |_| Ok(Flow::Continue))? && v.tokens.len() < v.opts.max_tokens;
                    toks.push(id);
                }
                if go {
                    next.push(i);
                } else {
                    toks.truncate(next.len());
                    done_at[i] = tg.elapsed().as_secs_f64();
                }
            }
            if self.cancel.load(Ordering::SeqCst) {
                for &i in &next {
                    voices[i].finish = StopReason::Cancelled;
                    done_at[i] = tg.elapsed().as_secs_f64();
                }
                break;
            }
            if next.is_empty() {
                break;
            }
            let mut sess: Vec<&mut QwenSession> = sessions
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| next.binary_search(i).is_ok())
                .map(|(_, s)| s)
                .collect();
            hidden = self.model.forward_hidden_batch(&mut sess, &toks);
            bs.steps += 1;
            live = next;
        }
        bs.generate_secs = tg.elapsed().as_secs_f64();
        drop(restore);
        drop(sessions);

        // The audio of all utterances in shared SNAC passes.
        let ts = Instant::now();
        let codes: Vec<[Vec<u32>; 3]> = voices.iter().map(|v| frames_to_codes(&v.frames)).collect();
        let code_refs: Vec<[&[u32]; 3]> = codes.iter().map(|c| [&c[0][..], &c[1][..], &c[2][..]]).collect();
        let jobs: Vec<crate::snac::DecodeJob> = code_refs
            .iter()
            .zip(items)
            .map(|(c, u)| crate::snac::DecodeJob {
                codes: c,
                frame0: 0,
                noise: if u.opts.noise { Noise::Seeded(u.opts.seed) } else { Noise::Off },
            })
            .collect();
        let audio = self.snac.decode_batch(&jobs)?;
        bs.snac_secs = ts.elapsed().as_secs_f64();
        bs.total_secs = t0.elapsed().as_secs_f64();

        let mut out = Vec::with_capacity(n);
        for ((v, samples), done) in voices.into_iter().zip(audio).zip(done_at) {
            let Voice {
                tokens,
                frames,
                finish,
                mut st,
                ..
            } = v;
            st.generate_secs = done;
            st.snac_secs = bs.snac_secs;
            st.tokens = tokens.len();
            st.frames = frames.len();
            st.audio_secs = samples.len() as f64 / rate as f64;
            st.total_secs = bs.total_secs;
            bs.tokens += st.tokens;
            bs.audio_secs += st.audio_secs;
            out.push(Speech {
                audio: Audio {
                    samples,
                    sample_rate: rate,
                },
                tokens,
                frames,
                finish,
                stats: st,
            });
        }
        Ok((out, bs))
    }

    /// Speaks a long text as a batch of its sentences ([`split_sentences`] with pieces of up to
    /// `max_bytes` bytes, 0 for the default 240), all with `opts`, and joins the audio in order.
    /// Each piece is its own utterance (at most `opts.max_tokens` tokens, about 14 s by default,
    /// which a whole long text would overrun), so the prosody restarts at each piece.
    pub fn speak_long(&self, text: &str, opts: &SpeakOptions, max_bytes: usize) -> Result<(Audio, Vec<Speech>, BatchStats)> {
        let pieces = split_sentences(text, if max_bytes == 0 { 240 } else { max_bytes });
        let items: Vec<Utterance> = pieces.iter().map(|p| Utterance::new(p, opts)).collect();
        let (speeches, stats) = self.speak_batch_detailed(&items)?;
        let samples = speeches.iter().flat_map(|s| s.audio.samples.iter().copied()).collect();
        let audio = Audio {
            samples,
            sample_rate: self.snac.sample_rate(),
        };
        Ok((audio, speeches, stats))
    }
}

/// Splits `text` into utterances at sentence ends (`.`, `!`, `?`, `…` and their CJK forms,
/// with any closing quotes or brackets, before white space or the end; a line break always
/// ends a piece), then joins neighbouring sentences while the piece stays within `max_bytes`
/// bytes. A sentence longer than `max_bytes` is kept whole. Empty pieces are dropped.
pub fn split_sentences(text: &str, max_bytes: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in text.lines() {
        let mut cur = String::new();
        let mut push = |sentence: &str, cur: &mut String| {
            let s = sentence.trim();
            if s.is_empty() {
                return;
            }
            if !cur.is_empty() && cur.len() + 1 + s.len() > max_bytes {
                out.push(std::mem::take(cur));
            }
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(s);
        };
        let mut start = 0;
        let mut chars = line.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            let cjk = matches!(c, '。' | '！' | '？');
            if !(cjk || matches!(c, '.' | '!' | '?' | '…')) {
                continue;
            }
            // Closing punctuation stays with its sentence.
            let mut end = i + c.len_utf8();
            while let Some(&(j, d)) = chars.peek() {
                if matches!(
                    d,
                    '.' | '!' | '?' | '…' | '"' | '\'' | ')' | ']' | '”' | '’' | '»' | '」' | '』' | '。' | '！' | '？'
                ) {
                    end = j + d.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            if cjk || chars.peek().is_none_or(|&(_, d)| d.is_whitespace()) {
                push(&line[start..end], &mut cur);
                start = end;
            }
        }
        push(&line[start..], &mut cur);
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen::batch::tests::tiny_llama;
    use crate::snac::tests::tiny_snac;

    #[test]
    fn sentences() {
        let s = split_sentences("Hello there. How are you? \"Fine!\" she said... Then 3.5 apples.\nNew line", 1);
        assert_eq!(
            s,
            [
                "Hello there.",
                "How are you?",
                "\"Fine!\"",
                "she said...",
                "Then 3.5 apples.",
                "New line"
            ]
        );
        let s = split_sentences("One. Two. Three is longer. Four.", 12);
        assert_eq!(s, ["One. Two.", "Three is longer.", "Four."]);
        assert_eq!(split_sentences("你好。再见！", 1), ["你好。", "再见！"]);
        assert!(split_sentences("  \n ", 10).is_empty());
    }

    /// A random Orpheus-shaped speaker: a 2-layer Llama with the audio vocabulary, a word-level
    /// tokenizer and a small SNAC. The stop rows of the output layer are scaled up so that some
    /// utterances end on a stop token.
    fn tiny_speaker(tag: &str, quant: Quant) -> (Speaker, std::path::PathBuf, std::path::PathBuf) {
        let vocab = (AUDIO_BASE + FRAME_TOKENS as u32 * CODES) as usize;
        let dir = tiny_llama(tag, vocab, 64, 2, 9, &[(END_OF_SPEECH as usize, 2.2)]);
        let words = [
            "tara:", "leo:", "hello", "there", "how", "are", "you", "today", "fine", "thanks", "and", "good", "night",
        ];
        let mut v = serde_json::Map::new();
        v.insert("<unk>".into(), 7.into());
        for (i, w) in words.iter().enumerate() {
            v.insert((*w).into(), (20 + i).into());
        }
        let tok = serde_json::json!({
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "model": {"type": "WordLevel", "vocab": v, "unk_token": "<unk>"},
        });
        std::fs::write(dir.join("tokenizer.json"), serde_json::to_vec(&tok).unwrap()).unwrap();
        let snac = tiny_snac(&format!("tts-{tag}"));
        let lo = SpeakerOptions {
            quant,
            device: DevicePref::Cpu,
            ..Default::default()
        };
        (Speaker::load_dirs(&dir, &snac, &lo).unwrap(), dir, snac)
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Every utterance of a batch (different texts, voices, seeds, samplers and limits, ending
    /// at different steps) is the utterance spoken alone: same tokens, frames, finish reason
    /// and samples, at every weight precision.
    #[test]
    fn batch_matches_alone() {
        crate::pool::init(3);
        for quant in [Quant::Q4, Quant::Q8, Quant::Bf16] {
            let (sp, dir, snac) = tiny_speaker(&format!("{quant:?}"), quant);
            let texts = [
                "hello there",
                "how are you today",
                "fine thanks and you",
                "good night",
                "hello",
                "you are fine today and how are you thanks",
                "night night night",
                "there there",
                "and and how",
            ];
            let opts: Vec<SpeakOptions> = (0..texts.len())
                .map(|i| {
                    // Greedy, sampled, top-k; with and without the constrained head and the noise.
                    SpeakOptions {
                        voice: [Some("tara".to_string()), Some("leo".to_string()), None][i % 3].clone(),
                        seed: 3 + i as u64,
                        temperature: [0.0, 0.6, 1.0][i % 3],
                        top_k: if i % 4 == 1 { 40 } else { 0 },
                        constrained: i % 4 != 2,
                        noise: i % 2 == 0,
                        max_tokens: [60, 33, 300, 7, 50][i % 5],
                        max_secs_per_byte: if i == 5 { 0.0 } else { 0.02 },
                        ..Default::default()
                    }
                })
                .collect();
            let items: Vec<Utterance> = texts.iter().zip(&opts).map(|(t, o)| Utterance::new(t, o)).collect();
            let (batch, stats) = sp.speak_batch_detailed(&items).unwrap();
            assert_eq!(batch.len(), texts.len());
            let mut lens = std::collections::BTreeSet::new();
            for (i, (b, u)) in batch.iter().zip(&items).enumerate() {
                let a = sp.speak_detailed(u.text, u.opts).unwrap();
                assert_eq!(b.tokens, a.tokens, "{quant:?} utterance {i}: tokens");
                assert_eq!(b.frames, a.frames, "{quant:?} utterance {i}");
                assert_eq!(b.finish, a.finish, "{quant:?} utterance {i}");
                assert_eq!((b.stats.loops, b.stats.dropped_tokens), (a.stats.loops, a.stats.dropped_tokens));
                assert_eq!(bits(&b.audio.samples), bits(&a.audio.samples), "{quant:?} utterance {i}: audio");
                assert!(!b.tokens.is_empty() && b.tokens.len() <= u.opts.max_tokens);
                lens.insert(b.tokens.len());
            }
            assert!(lens.len() > 2, "ragged lengths: {lens:?}");
            assert_eq!(stats.tokens, batch.iter().map(|s| s.tokens.len()).sum::<usize>());
            assert!(stats.steps < stats.tokens);
            // The batch of one is the utterance alone too, and an empty batch is fine.
            let one = sp.speak_batch(&items[1..2]).unwrap();
            assert_eq!(one[0].tokens, batch[1].tokens);
            assert!(sp.speak_batch(&[]).unwrap().is_empty());
            // Sentences of a long text, joined.
            let (audio, parts, _) = sp.speak_long("hello there. how are you today? fine thanks.", &opts[0], 1).unwrap();
            assert_eq!(parts.len(), 3);
            assert_eq!(audio.samples.len(), parts.iter().map(|p| p.audio.samples.len()).sum::<usize>());
            let alone = sp.speak_detailed("how are you today?", &opts[0]).unwrap();
            assert_eq!(parts[1].tokens, alone.tokens);
            drop(sp);
            std::fs::remove_dir_all(&dir).unwrap();
            std::fs::remove_dir_all(&snac).unwrap();
        }
    }

    /// The same on real weights, on the CPU: `OXINFER_TTS_MODEL=isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF:Q4_K_M
    /// cargo test --release --lib -- --ignored real_batch_matches_alone --nocapture`
    /// (`OXINFER_TTS_QUANT=q4|q8` for a safetensors checkpoint, `OXINFER_TTS_THREADS`, default 8;
    /// `OXINFER_TTS_N` utterances, default 9: more than the 7 columns of a default pass).
    #[test]
    #[ignore]
    fn real_batch_matches_alone() {
        let model = std::env::var("OXINFER_TTS_MODEL").expect("OXINFER_TTS_MODEL=<Orpheus model>");
        let env = |k: &str| std::env::var(k).ok();
        let mut lo = SpeakerOptions {
            device: DevicePref::Cpu,
            threads: Some(env("OXINFER_TTS_THREADS").map_or(8, |v| v.parse().unwrap())),
            ..Default::default()
        };
        if let Some(q) = env("OXINFER_TTS_QUANT") {
            lo.set("quant", &q).unwrap();
        }
        let sp = Speaker::load(&model, &lo).unwrap();
        let all = [
            "Hello there, how are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "Well, <chuckle> I did not expect that.",
            "It is raining again.",
            "Please leave a message after the tone, and we will call you back as soon as we can.",
            "Good night.",
            "One, two, three, four, five.",
            "What time does the next train leave?",
            "Thanks a lot!",
            "I think the answer is forty two.",
            "See you tomorrow morning.",
            "This sentence is a little longer than the others, so that it ends well after them.",
        ];
        let n = env("OXINFER_TTS_N").map_or(9, |v| v.parse().unwrap()).min(all.len());
        let voices = sp.info().voices.clone();
        let opts: Vec<SpeakOptions> = (0..n)
            .map(|i| {
                // Greedy without noise and sampled with it, in turn; two from the whole vocabulary.
                SpeakOptions {
                    voice: voices.get(i % voices.len().max(1)).cloned(),
                    temperature: if i % 2 == 0 { 0.0 } else { 0.6 },
                    noise: i % 2 == 1,
                    seed: 42 + i as u64 / 3,
                    constrained: i != 3 && i != 6,
                    max_tokens: 400,
                    ..Default::default()
                }
            })
            .collect();
        let items: Vec<Utterance> = all[..n].iter().zip(&opts).map(|(t, o)| Utterance::new(t, o)).collect();
        let (batch, stats) = sp.speak_batch_detailed(&items).unwrap();
        eprintln!(
            "batch of {n}: {} tokens in {} steps, up to {} per pass, {:.1} s of audio",
            stats.tokens, stats.steps, stats.width, stats.audio_secs
        );
        for (i, (b, u)) in batch.iter().zip(&items).enumerate() {
            let a = sp.speak_detailed(u.text, u.opts).unwrap();
            let same = a
                .audio
                .samples
                .iter()
                .zip(&b.audio.samples)
                .filter(|(x, y)| x.to_bits() == y.to_bits())
                .count();
            eprintln!(
                "utterance {i} (temperature {}): {} tokens alone, {} in the batch, {:?}; {same} of {} samples identical",
                u.opts.temperature,
                a.tokens.len(),
                b.tokens.len(),
                b.finish,
                a.audio.samples.len()
            );
            assert_eq!(b.tokens, a.tokens, "utterance {i}: tokens");
            assert_eq!(b.finish, a.finish);
            assert_eq!(bits(&b.audio.samples), bits(&a.audio.samples), "utterance {i}: audio");
        }
    }
}
