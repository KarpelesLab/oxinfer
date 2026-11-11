//! Tests of batched transcription and beam search on a tiny random-weight Whisper (written to a
//! temporary folder as a checkpoint): no downloaded weights needed.

use super::*;
use crate::platform::DevicePref;
use std::collections::BTreeMap;

/// Token ids of the tiny model: text 0..100, then the special tokens and 1501 timestamps.
const EOT: u32 = 100;
const SOT: u32 = 101;
const NO_TIMESTAMPS: u32 = 108;
const TS: u32 = 109;
const VOCAB: usize = TS as usize + 1501;
const D: usize = 64;

/// Adds a random tensor: name, shape, scale and offset of its values.
type Add<'a> = dyn FnMut(&str, &[usize], f32, f32) + 'a;

struct Rng(u64);

impl Rng {
    /// Uniform in [-1, 1).
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// A checkpoint folder removed when dropped.
struct TinyDir(PathBuf);

impl Drop for TinyDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Writes a 2 + 2 layer Whisper with `d_model` 64 and random weights in the Hugging Face layout.
fn tiny_checkpoint(name: &str) -> TinyDir {
    let dir = std::env::temp_dir().join(format!("oxinfer-whisper-tiny-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = serde_json::json!({
        "model_type": "whisper", "d_model": D, "encoder_layers": 2, "decoder_layers": 2,
        "encoder_attention_heads": 2, "decoder_attention_heads": 2, "encoder_ffn_dim": 2 * D,
        "decoder_ffn_dim": 2 * D, "num_mel_bins": 80, "vocab_size": VOCAB,
        "max_source_positions": 1500, "max_target_positions": 448,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let generation = serde_json::json!({
        "eos_token_id": EOT, "decoder_start_token_id": SOT, "no_timestamps_token_id": NO_TIMESTAMPS,
        "prev_sot_token_id": 106, "is_multilingual": true,
        "lang_to_id": {"<|en|>": 102, "<|fr|>": 103},
        "task_to_id": {"translate": 104, "transcribe": 105},
        "suppress_tokens": [1, 2, 101, 102, 103, 104, 105, 106, 107],
        "begin_suppress_tokens": [0, EOT],
        "max_initial_timestamp_index": 50, "max_length": 448,
        "alignment_heads": [[1, 0], [1, 1]],
    });
    std::fs::write(dir.join("generation_config.json"), generation.to_string()).unwrap();
    let mut vocab = serde_json::Map::new();
    for i in 0..100 {
        vocab.insert(format!("w{i}"), i.into());
    }
    vocab.insert("<|endoftext|>".into(), EOT.into());
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "w0"},
    });
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();

    // The tensors, as a safetensors file.
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let mut add = |name: &str, shape: &[usize], scale: f32, offset: f32| {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| offset + scale * rng.next()).collect();
        tensors.insert(name.to_string(), (shape.to_vec(), v));
    };
    let norm = |add: &mut Add, p: &str| {
        add(&format!("{p}.weight"), &[D], 0.2, 1.0);
        add(&format!("{p}.bias"), &[D], 0.1, 0.0);
    };
    let linear = |add: &mut Add, p: &str, n_out: usize, n_in: usize, bias: bool| {
        add(&format!("{p}.weight"), &[n_out, n_in], 0.25, 0.0);
        if bias {
            add(&format!("{p}.bias"), &[n_out], 0.1, 0.0);
        }
    };
    let attn = |add: &mut Add, p: &str| {
        linear(add, &format!("{p}.q_proj"), D, D, true);
        linear(add, &format!("{p}.k_proj"), D, D, false);
        linear(add, &format!("{p}.v_proj"), D, D, true);
        linear(add, &format!("{p}.out_proj"), D, D, true);
    };
    add("model.encoder.conv1.weight", &[D, 80, 3], 0.2, 0.0);
    add("model.encoder.conv1.bias", &[D], 0.1, 0.0);
    add("model.encoder.conv2.weight", &[D, D, 3], 0.2, 0.0);
    add("model.encoder.conv2.bias", &[D], 0.1, 0.0);
    add("model.encoder.embed_positions.weight", &[1500, D], 0.5, 0.0);
    add("model.decoder.embed_positions.weight", &[448, D], 0.5, 0.0);
    add("model.decoder.embed_tokens.weight", &[VOCAB, D], 0.5, 0.0);
    norm(&mut add, "model.encoder.layer_norm");
    norm(&mut add, "model.decoder.layer_norm");
    for l in 0..2 {
        let p = format!("model.encoder.layers.{l}");
        norm(&mut add, &format!("{p}.self_attn_layer_norm"));
        attn(&mut add, &format!("{p}.self_attn"));
        norm(&mut add, &format!("{p}.final_layer_norm"));
        linear(&mut add, &format!("{p}.fc1"), 2 * D, D, true);
        linear(&mut add, &format!("{p}.fc2"), D, 2 * D, true);
        let p = format!("model.decoder.layers.{l}");
        norm(&mut add, &format!("{p}.self_attn_layer_norm"));
        attn(&mut add, &format!("{p}.self_attn"));
        norm(&mut add, &format!("{p}.encoder_attn_layer_norm"));
        attn(&mut add, &format!("{p}.encoder_attn"));
        norm(&mut add, &format!("{p}.final_layer_norm"));
        linear(&mut add, &format!("{p}.fc1"), 2 * D, D, true);
        linear(&mut add, &format!("{p}.fc2"), D, 2 * D, true);
    }
    // Closer to a trained model's output than uniform noise over a vocabulary that is mostly
    // timestamps: quieter timestamp rows, and a louder <|endoftext|> so that sequences end.
    let embed = &mut tensors.get_mut("model.decoder.embed_tokens.weight").unwrap().1;
    embed[TS as usize * D..].iter_mut().for_each(|v| *v *= 0.3);
    embed[EOT as usize * D..(EOT as usize + 1) * D].iter_mut().for_each(|v| *v *= 1.6);
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, (shape, v)) in &tensors {
        let start = data.len();
        data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut head = serde_json::Value::Object(header).to_string().into_bytes();
    while !head.len().is_multiple_of(8) {
        head.push(b' ');
    }
    let mut file = (head.len() as u64).to_le_bytes().to_vec();
    file.extend(head);
    file.extend(data);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
    TinyDir(dir)
}

fn tiny(name: &str, precision: &str) -> (Transcriber, TinyDir) {
    let dir = tiny_checkpoint(name);
    let mut lo = TranscriberOptions {
        device: DevicePref::Cpu,
        ..Default::default()
    };
    lo.set("precision", precision).unwrap();
    (Transcriber::load_dir(&dir.0, &lo).unwrap(), dir)
}

/// `secs` of a deterministic signal: two gliding tones and noise, different for each `seed`.
fn audio(secs: f64, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let n = (secs * mel::SAMPLE_RATE as f64) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / mel::SAMPLE_RATE as f32;
            let f = 180.0 + 90.0 * seed as f32 + 40.0 * t;
            0.3 * (std::f32::consts::TAU * f * t).sin() + 0.2 * (std::f32::consts::TAU * 3.1 * f * t).sin() + 0.1 * rng.next()
        })
        .collect()
}

/// Everything a transcript holds but its timings.
fn signature(t: &Transcript) -> String {
    format!(
        "{} | {} | {:?} | {} | {}",
        t.text,
        t.language,
        t.language_probability.map(f32::to_bits),
        t.stopped,
        serde_json::to_string(&t.segments).unwrap()
    )
}

/// Options that keep a random model's windows short: few tokens, and no timestamp below 5 s
/// but `<|0.00|>`, so a window moves on by 5 s at least.
fn base_options() -> TranscribeOptions {
    TranscribeOptions {
        max_new_tokens: Some(16),
        suppress_tokens: Some(
            [1, 2, 101, 102, 103, 104, 105, 106, 107]
                .into_iter()
                .chain(TS + 1..TS + 250)
                .collect(),
        ),
        ..Default::default()
    }
}

#[test]
fn encoder_batch_matches_single_windows() {
    for precision in ["f32", "bf16", "q8"] {
        let (asr, _dir) = tiny(&format!("enc-{precision}"), precision);
        let m = asr.model();
        let windows: Vec<Vec<f32>> = (0..3)
            .map(|i| asr.window(&mel::log_mel(&audio(4.0 + 9.0 * i as f64, i), 80), 0))
            .collect();
        let refs: Vec<&[f32]> = windows.iter().map(|w| &w[..]).collect();
        let alone: Vec<Vec<f32>> = refs.iter().map(|w| m.encode(w)).collect();
        assert!(alone[0] != alone[1], "{precision}: the windows encode alike");
        assert!(m.encode_batch(&refs) == alone, "{precision}: windows encoded together differ");
    }
}

#[test]
fn decoder_step_rows_match_single_sequences() {
    for precision in ["f32", "bf16", "q8", "q4"] {
        let (asr, _dir) = tiny(&format!("dec-{precision}"), precision);
        let m = asr.model();
        let d = m.cfg.d_model;
        let cross: Vec<model::CrossKv> = (0..2)
            .map(|i| m.cross_kv(&m.encode(&asr.window(&mel::log_mel(&audio(6.0, i), 80), 0))))
            .collect();
        // Nine sequences (more than one group of rows), with prompts of different lengths.
        let prompts: Vec<Vec<u32>> = (0..9u32)
            .map(|i| (0..=i % 4).map(|j| if j == 0 { SOT } else { 3 + i + j }).collect())
            .collect();
        let steps: Vec<[u32; 3]> = (0..9u32).map(|i| [5 + i, 40 + 2 * i, 77 - i]).collect();
        let mut alone: Vec<Vec<f32>> = Vec::new();
        for (i, p) in prompts.iter().enumerate() {
            let mut st = m.new_state();
            m.decode(p, &mut st, &cross[i % 2]);
            let mut out = Vec::new();
            for &t in &steps[i] {
                out.extend(m.logits(&m.decode(&[t], &mut st, &cross[i % 2])));
            }
            alone.push(out);
        }
        let mut sts: Vec<model::DecState> = prompts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut st = m.new_state();
                let h = m.decode(p, &mut st, &cross[i % 2]);
                assert_eq!(h.len(), p.len() * d);
                st
            })
            .collect();
        let cr: Vec<&model::CrossKv> = (0..9).map(|i| &cross[i % 2]).collect();
        let v = m.cfg.vocab_size;
        for k in 0..3 {
            let toks: Vec<u32> = steps.iter().map(|s| s[k]).collect();
            let mut refs: Vec<&mut model::DecState> = sts.iter_mut().collect();
            let logits = m.decode_step(&toks, &mut refs, &cr);
            for i in 0..9 {
                assert!(
                    logits[i * v..(i + 1) * v] == alone[i][k * v..(k + 1) * v],
                    "{precision}: sequence {i} step {k} differs in a batch"
                );
            }
        }
    }
}

#[test]
fn batch_matches_each_input_alone() {
    let (asr, _dir) = tiny("batch", "bf16");
    // Short clips, one of exactly a window, and long-form inputs of two and more windows.
    let pcms: Vec<Vec<f32>> = [2.5, 11.0, 47.0, 30.0, 7.3, 33.5]
        .iter()
        .enumerate()
        .map(|(i, &s)| audio(s, i as u64))
        .collect();
    let refs: Vec<&[f32]> = pcms.iter().map(|p| &p[..]).collect();
    let mut cases: Vec<(&str, TranscribeOptions)> = Vec::new();
    cases.push(("greedy", base_options()));
    let mut o = base_options();
    o.timestamps = true;
    o.language = Some("fr".into());
    o.word_timestamps = true;
    o.batch_size = 4;
    cases.push(("timestamps and words", o));
    let mut o = base_options();
    o.beam_size = Some(3);
    o.batch_size = 2;
    cases.push(("beam search", o));
    let mut o = base_options();
    o.beam_size = Some(4);
    o.patience = Some(1.5);
    o.length_penalty = Some(0.8);
    o.timestamps = true;
    o.condition_on_prev_tokens = true;
    cases.push(("beam search with patience, conditioned", o));
    let mut o = base_options();
    o.temperatures = vec![0.8];
    o.best_of = Some(3);
    o.seed = 7;
    o.batch_size = 3;
    cases.push(("best of three samples", o));
    // Fallback: a log-probability threshold some windows meet at temperature 0 and others not.
    let mut o = base_options();
    o.beam_size = Some(2);
    o.best_of = Some(2);
    o.temperatures = vec![0.0, 0.4, 0.9];
    o.compression_ratio_threshold = Some(2.4);
    cases.push(("fallback", o));

    for (name, mut o) in cases {
        if name == "fallback" {
            // The median of the windows' log-probabilities at temperature 0 splits them.
            let mut lps: Vec<f32> = asr
                .transcribe_batch(&refs, &{
                    let mut g = o.clone();
                    g.temperatures = vec![0.0];
                    g
                })
                .unwrap()
                .iter()
                .flat_map(|t| t.segments.iter().map(|s| s.avg_logprob))
                .collect();
            lps.sort_by(|a, b| a.partial_cmp(b).unwrap());
            o.logprob_threshold = Some(lps[lps.len() / 2]);
        }
        let alone: Vec<Transcript> = refs.iter().map(|p| asr.transcribe(p, &o).unwrap()).collect();
        let mut seen: Vec<Vec<usize>> = vec![Vec::new(); refs.len()];
        let batch = asr
            .transcribe_batch_with(&refs, &o, &mut |i, s| {
                seen[i].push(s.id);
                Flow::Continue
            })
            .unwrap();
        assert_eq!(batch.len(), refs.len());
        for (i, (a, b)) in alone.iter().zip(&batch).enumerate() {
            assert_eq!(signature(a), signature(b), "{name}: input {i} differs in a batch");
            assert_eq!(a.stats.windows, b.stats.windows, "{name}: input {i}");
            assert_eq!(a.stats.tokens, b.stats.tokens, "{name}: input {i}");
            assert_eq!(
                seen[i],
                (0..b.segments.len()).collect::<Vec<_>>(),
                "{name}: segment callbacks of input {i}"
            );
        }
        assert!(batch[2].stats.windows >= 2, "{name}: the long input has one window");
        let with_text = batch.iter().filter(|t| !t.segments.is_empty()).count();
        assert!(with_text >= 3, "{name}: {with_text} inputs with segments");
        if name == "fallback" {
            let temps: Vec<f32> = batch.iter().flat_map(|t| t.segments.iter().map(|s| s.temperature)).collect();
            assert!(
                temps.contains(&0.0) && temps.iter().any(|&t| t > 0.0),
                "fallback: temperatures {temps:?}"
            );
        }
        if name == "greedy" {
            // The inputs do not transcribe alike (the batch is not one result repeated).
            assert!(signature(&batch[0]) != signature(&batch[1]));
            assert!(batch[0].language_probability.is_some());
        }
    }
}

#[test]
fn batch_stops_one_input_or_all() {
    let (asr, _dir) = tiny("stop", "bf16");
    let pcms: Vec<Vec<f32>> = (0..3).map(|i| audio(40.0, 10 + i)).collect();
    let refs: Vec<&[f32]> = pcms.iter().map(|p| &p[..]).collect();
    let o = base_options();
    let full = asr.transcribe_batch(&refs, &o).unwrap();
    assert!(full.iter().all(|t| !t.stopped && t.stats.windows >= 2));
    // Stop returned for one input ends that one only.
    let part = asr
        .transcribe_batch_with(&refs, &o, &mut |i, _| if i == 1 { Flow::Stop } else { Flow::Continue })
        .unwrap();
    assert!(part[1].stopped && part[1].stats.windows == 1);
    assert_eq!(signature(&part[0]), signature(&full[0]));
    assert_eq!(signature(&part[2]), signature(&full[2]));
    // A cancel request ends every input, those not started included.
    let cancel = asr.cancel_handle();
    let mut o1 = o.clone();
    o1.batch_size = 2;
    let stopped = asr
        .transcribe_batch_with(&refs, &o1, &mut |_, _| {
            cancel.cancel();
            Flow::Continue
        })
        .unwrap();
    assert!(stopped.iter().all(|t| t.stopped));
    assert!(stopped[2].segments.is_empty() && stopped[2].stats.windows == 0);
}

/// openai-whisper's beam search written the slow way, as its Python reads: token tuples in
/// insertion-ordered maps, and every hypothesis decoded again from the prompt (no cache
/// reordering). Returns the best hypothesis' tokens (without `<|endoftext|>`) and its
/// log-probability.
#[allow(clippy::too_many_arguments)]
fn reference_beam_search(
    asr: &Transcriber,
    sp: &Special,
    cross: &model::CrossKv,
    input: &[u32],
    rules: &Rules,
    size: usize,
    max_candidates: usize,
    sample_len: usize,
    length_penalty: Option<f32>,
) -> (Vec<u32>, f32) {
    let m = asr.model();
    let d = m.cfg.d_model;
    let logits_after = |seq: &[u32]| -> Vec<f32> {
        let mut st = m.new_state();
        let mut h = m.decode(input, &mut st, cross);
        h.drain(..(input.len() - 1) * d);
        for &t in seq {
            h = m.decode(&[t], &mut st, cross);
        }
        m.logits(&h)
    };
    let mut beams: Vec<(Vec<u32>, f32)> = vec![(Vec::new(), 0.0); size];
    let mut finished: Vec<(Vec<u32>, f32)> = Vec::new();
    for _ in 0..sample_len {
        // scores[sequence] = cumulative log-probability, in insertion order.
        let mut scores: Vec<(Vec<u32>, f32)> = Vec::new();
        for (prefix, sum) in &beams {
            let mut logits = logits_after(prefix);
            process(sp, &mut logits, prefix, rules);
            let lp = log_softmax(&logits);
            let mut order: Vec<usize> = (0..lp.len()).collect();
            order.sort_by(|&a, &b| lp[b].partial_cmp(&lp[a]).unwrap());
            for &tok in &order[..size + 1] {
                let seq: Vec<u32> = prefix.iter().copied().chain([tok as u32]).collect();
                let score = sum + lp[tok];
                match scores.iter_mut().find(|s| s.0 == seq) {
                    Some(s) => s.1 = score,
                    None => scores.push((seq, score)),
                }
            }
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut next = Vec::new();
        for (seq, score) in scores {
            if seq.last() == Some(&sp.eot) {
                if finished.len() < max_candidates {
                    finished.push((seq, score));
                }
            } else {
                next.push((seq, score));
                if next.len() == size {
                    break;
                }
            }
        }
        beams = next;
        if finished.len() >= max_candidates {
            break;
        }
    }
    if finished.len() < size {
        let mut rest = beams.clone();
        rest.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (mut seq, score) in rest {
            seq.push(sp.eot);
            finished.push((seq, score));
            if finished.len() >= size {
                break;
            }
        }
    }
    let mut best: Option<(f64, usize)> = None;
    for (i, (seq, sum)) in finished.iter().enumerate() {
        let len = (seq.len() - 1) as f64;
        let penalty = match length_penalty {
            None => len,
            Some(a) => ((5.0 + len) / 6.0).powf(a as f64),
        };
        let score = *sum as f64 / penalty;
        if best.is_none_or(|b| score > b.0) {
            best = Some((score, i));
        }
    }
    let (mut seq, sum) = finished.swap_remove(best.unwrap().1);
    seq.pop();
    (seq, sum)
}

#[test]
fn beam_search_matches_the_reference_search() {
    let (asr, _dir) = tiny("beam", "f32");
    let m = asr.model();
    // A handful of text tokens and the timestamps are allowed.
    let suppress: Vec<bool> = (0..VOCAB as u32).map(|t| !((3..10).contains(&t) || t >= NO_TIMESTAMPS)).collect();
    let mut ended = 0;
    for (ci, &(secs, timestamps, size, patience, length_penalty, sample_len)) in [
        (6.0, false, 3usize, 1.0f32, None, 12usize),
        (6.0, true, 3, 1.0, None, 12),
        (14.0, true, 4, 2.0, Some(0.7f32), 10),
        (3.0, false, 2, 1.0, Some(1.0), 14),
        (9.0, false, 5, 1.0, None, 8),
        (21.0, true, 1, 1.0, None, 12),
        (21.0, false, 1, 3.0, None, 12),
        (9.0, false, 5, 3.0, None, 2),
    ]
    .iter()
    .enumerate()
    {
        let pcm = audio(secs, 20 + ci as u64);
        let cross = m.cross_kv(&m.encode(&asr.window(&mel::log_mel(&pcm, 80), 0)));
        let mut input = vec![SOT, 102, 105];
        if !timestamps {
            input.push(NO_TIMESTAMPS);
        }
        // Random weights never make <|endoftext|> likely: the text token greedy decoding
        // picks after its first one stands for it, so that hypotheses do end.
        let mut sp = asr.special().clone();
        let probe = decode::Job {
            search: decode::Search::Sample { temperature: 0.0, n: 1 },
            cross: &cross,
            input: &input,
            sot_index: 0,
            rules: &Rules {
                suppress: &suppress,
                begin_suppress: &[],
                timestamps,
            },
            length_penalty: None,
            max_new: 4,
            seed: 0,
        };
        let probe = decode::run(m, &sp, &[probe], &AtomicBool::new(false)).pop().unwrap();
        if let Some(&end) = probe.tokens[1..].iter().find(|&&t| t < 10) {
            sp.eot = end;
            sp.begin_suppress = vec![end];
        }
        let sp = &sp;
        let rules = Rules {
            suppress: &suppress,
            begin_suppress: &sp.begin_suppress,
            timestamps,
        };
        let max_candidates = (size as f64 * patience as f64).round_ties_even() as usize;
        let job = decode::Job {
            cross: &cross,
            input: &input,
            sot_index: 0,
            rules: &rules,
            search: decode::Search::Beam { size, max_candidates },
            length_penalty,
            max_new: sample_len,
            seed: 0,
        };
        let a = decode::run(m, sp, &[job], &AtomicBool::new(false)).pop().unwrap();
        let (want, want_sum) = reference_beam_search(&asr, sp, &cross, &input, &rules, size, max_candidates, sample_len, length_penalty);
        assert_eq!(a.tokens.last(), Some(&sp.eot), "case {ci}");
        assert_eq!(a.tokens[..a.tokens.len() - 1], want[..], "case {ci}: tokens");
        assert_eq!(
            a.logprobs.iter().sum::<f32>().to_bits(),
            want_sum.to_bits(),
            "case {ci}: log-probability"
        );
        ended += (a.tokens.len() <= sample_len) as usize;

        // One beam that waits for one hypothesis is greedy decoding.
        if size == 1 && max_candidates == 1 {
            let greedy = decode::Job {
                search: decode::Search::Sample { temperature: 0.0, n: 1 },
                cross: &cross,
                input: &input,
                sot_index: 0,
                rules: &rules,
                length_penalty: None,
                max_new: sample_len,
                seed: 0,
            };
            let g = decode::run(m, sp, &[greedy], &AtomicBool::new(false)).pop().unwrap();
            let text = |t: &[u32]| t.iter().copied().filter(|&x| x != sp.eot).collect::<Vec<_>>();
            assert_eq!(text(&g.tokens), text(&a.tokens), "case {ci}: one beam against greedy");
        }
    }
    // Some hypotheses end by <|endoftext|> (the others are cut at the length limit).
    assert!(ended > 0, "no search ended by <|endoftext|>");
}

#[test]
fn search_options_are_checked() {
    let (asr, _dir) = tiny("opts", "bf16");
    let pcm = audio(1.0, 1);
    let bad = |f: &dyn Fn(&mut TranscribeOptions)| {
        let mut o = base_options();
        f(&mut o);
        asr.transcribe(&pcm, &o).is_err()
    };
    assert!(bad(&|o| o.beam_size = Some(0)));
    assert!(bad(&|o| o.patience = Some(2.0)));
    assert!(bad(&|o| {
        o.beam_size = Some(2);
        o.patience = Some(0.1);
    }));
    assert!(bad(&|o| o.best_of = Some(0)));
    assert!(bad(&|o| o.length_penalty = Some(1.5)));
    assert!(!bad(&|o| {
        o.beam_size = Some(2);
        o.patience = Some(2.0);
        o.length_penalty = Some(1.0);
        o.best_of = Some(2);
    }));
}
