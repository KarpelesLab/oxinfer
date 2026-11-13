# Speech to text (Whisper)

OpenAI's Whisper checkpoints in the Hugging Face layout (`model_type: whisper`) transcribe or
translate audio of any length (`src/whisper/`): openai/whisper-{tiny, base, small, medium,
large-v3, large-v3-turbo}, the English-only `.en` models, and distil-whisper (the same network
with fewer decoder layers, e.g. distil-whisper/distil-large-v3). Sizes come from `config.json`
(80 mel bins, 128 for v3), special tokens and decoding defaults from `generation_config.json`.

```sh
./target/release/oxinfer --model openai/whisper-small --transcribe talk.mp3              # text
./target/release/oxinfer --model openai/whisper-large-v3-turbo --transcribe talk.mp3 --timestamps
./target/release/oxinfer --model openai/whisper-small --transcribe interview.m4a --language fr --translate
./target/release/oxinfer --model openai/whisper-small --transcribe talk.wav --format srt > talk.srt
./target/release/oxinfer --model openai/whisper-small --transcribe talk.wav --word-timestamps --json
./target/release/oxinfer --model openai/whisper-small --transcribe talk.wav --beam 5       # beam search
./target/release/oxinfer --model openai/whisper-small --transcribe a.mp3 --transcribe b.mp3 --transcribe c.wav  # a batch
```

| Flag | Meaning |
|---|---|
| `--transcribe FILE` | audio to transcribe: WAV, MP3, FLAC, AAC/M4A, Ogg Vorbis/Opus, MKV audio (oxideav), resampled to 16 kHz mono. Repeat it for several files: they are decoded together as a batch, with the result each file gets alone |
| `--batch N` | files decoded together at most (default 8); the others take the places of those that finish |
| `--language XX` | spoken language, as a code or a name (default: detected from the first 30 s) |
| `--translate` | translate to English text |
| `--timestamps` | timed segments (always on past 30 s) |
| `--word-timestamps` | also word-level times |
| `--format F` | `text` (default), `json`, `srt`, `vtt` (`--json` is `--format json`) |
| `--initial-prompt TEXT`, `--prefix TEXT` | context for the decoder (names, spelling, style); forced start of the transcript |
| `--fallback` | openai-whisper's temperature fallback: temperatures 0, 0.2, ..., 1.0 while a window's tokens compress better than 2.4 or average below -1 log-probability; a window with a no-speech probability over 0.6 and a low log-probability is skipped as silence |
| `--condition-on-prev` | feed each 30 s window the previous text |
| `--beam N` | beam search with N beams (openai-whisper's `beam_size`) at temperature 0; with `--fallback`, the higher temperatures sample |
| `--patience P` | beam search ends once `round(N * P)` hypotheses have ended (default 1) |
| `--length-penalty A` | rank the ended hypotheses by log-probability over `((5 + length) / 6)^A`, A from 0 to 1 (default: over the length) |
| `--best-of N` | at temperatures above 0 (`--temp T`, `--fallback`), sample N sequences and keep the best ranked |
| `--precision f32\|f16\|bf16` | weight storage (default bf16); `--quant q8\|q4` quantizes the decoder |

What runs, following transformers' `WhisperFeatureExtractor` and
`WhisperForConditionalGeneration.generate`:

- **Features**: 400-sample periodic Hann window, hop 160, reflect padding, power spectrum,
  Slaney mel filterbank, `log10`, clamp to `max - 8`, `(x + 4) / 4`. Clips up to 30 s are
  zero-padded to 30 s first; longer audio is featurized whole (the clamp sees all of it). The DFT
  (a mixed-radix FFT of 400) and the filterbank run in f64.
- **Encoder**: two convolutions (kernel 3, the second with stride 2) with GELU, sinusoidal
  positions, pre-LayerNorm transformer layers, all through the shared matmul (SIMD kernels, GPU
  backends for the big products) and a fused attention kernel on Metal.
- **Decoder**: learned positions, causal self-attention over a KV cache, cross-attention over the
  window's encoder output (keys and values computed once per window, kept head-major and attended
  with the keys split across threads), the output projection tied to the token embedding.
- **Decoding**: greedy by default (`TranscribeOptions::temperatures` for sampling and fallback);
  language detection from the first-token distribution over the language tokens (`language` then
  holds the code and its probability); `suppress_tokens` and `begin_suppress_tokens`; the rules of
  `WhisperTimeStampLogitsProcessor` (no `<|notimestamps|>`, timestamps in pairs and increasing, a
  timestamp first, at most 1 s in, and a timestamp whenever all timestamps together outweigh every
  text token). Audio over 30 s is decoded in sequential 30 s windows, each moved to the last
  complete segment's end timestamp, as transformers' long-form generation does; `initial_prompt`
  becomes `<|startofprev|>` context; `condition_on_prev_tokens` feeds the last 223 tokens.
- **Beam search** (`beam_size`, `patience`, `length_penalty`; `src/whisper/decode.rs`), as
  openai-whisper's `BeamSearchDecoder` and `MaximumLikelihoodRanker`: every beam proposes its
  `beam_size + 1` most likely tokens after the logit rules above (applied per beam); the
  candidates are walked by decreasing cumulative log-probability until `beam_size` of them that
  go on are kept, those ending in `<|endoftext|>` met on the way are set aside, and the search
  ends once `round(beam_size * patience)` hypotheses have ended or after 224 tokens. The best is
  the one with the highest log-probability over its length (or over `((5 + length) / 6)^a` with
  a length penalty). Each beam has its own self-attention cache: a beam continued once hands its
  cache on, a beam continued several times is copied. Beam search runs at temperature 0; the
  fallback temperatures above it sample, `best_of` sequences each, ranked the same way.
- **Batches**: several inputs are transcribed together (`transcribe_batch`): the current 30 s
  window of each goes through the encoder in one pass (the windows stacked in every matmul, one
  GPU submission per product for all of them; attention per window) and every hypothesis of
  every input is one row of each decoder step (rows share the matmuls seven at a time; attention
  per row over its own caches). Inputs that need another temperature retry together, an input
  that ends leaves its place to the next. The windows of one input stay sequential, since each
  starts at the previous one's last timestamp. A row's values do not depend on the rows computed
  with it (the CPU kernels compute up to seven rows one at a time, and the encoder is stacked on
  a GPU only when the backend takes the same products for one window as for the stack), so each
  transcript is the one the input gets alone: same tokens, times and probabilities, bit for bit.
- **Word timestamps**: openai-whisper's alignment: the cross-attention of the checkpoint's
  `alignment_heads` over the text tokens, normalized and median-filtered, aligned to the audio
  frames by dynamic time warping, tokens grouped into words, punctuation merged. Against
  transformers' pipeline (`return_timestamps="word"`) on a 29 s LibriSpeech clip, the 60 shared
  words start within 16 ms of it on average.

In Rust, `oxinfer::whisper::Transcriber` does all of it (`examples/transcribe.rs`):

```rust
use oxinfer::whisper::{Transcriber, TranscriberOptions, TranscribeOptions};
let asr = Transcriber::load("openai/whisper-small", &TranscriberOptions::default())?;
let mut opts = TranscribeOptions::default();
opts.language = Some("fr".into());
opts.timestamps = true;
let pcm = Transcriber::load_audio(std::path::Path::new("talk.mp3"))?; // or your own 16 kHz samples
let t = asr.transcribe_with(&pcm, &opts, &mut |seg| {
    println!("[{:.2} -> {:.2}] {}", seg.start, seg.end, seg.text); // as each 30 s window finishes
    oxinfer::engine::Flow::Continue                               // Flow::Stop ends it there
})?;
println!("{} ({}, {:.1}x real time)", t.text, t.language, 1.0 / t.stats.rtf);
```

Beam search and batches are options and one more call:

```rust
opts.beam_size = Some(5);          // openai-whisper's beam_size; patience, length_penalty, best_of
opts.batch_size = 8;               // inputs decoded together
let clips: Vec<Vec<f32>> = load_clips();
let refs: Vec<&[f32]> = clips.iter().map(|c| &c[..]).collect();
let transcripts = asr.transcribe_batch(&refs, &opts)?;   // one Transcript per input, in order
// transcribe_batch_with(&refs, &opts, &mut |input, seg| ...) reports each segment with its input;
// transcribe_files(&paths, &opts) decodes the files first.
```

`asr.cancel_handle()` stops a transcription from another thread (it returns the finished
segments with `stopped` set); `transcribe_file` decodes a file first; `detect_language` returns
the language distribution. `TranscribeOptions` deserializes from JSON, which is how the C API
takes it: `oxi_transcriber_load`, `oxi_transcribe_file` / `oxi_transcribe_pcm` (any sample rate)
with a per-segment callback, `oxi_transcriber_cancel`, results as JSON (`include/oxinfer.h`);
`beam_size`, `patience`, `length_penalty` and `best_of` are keys of that JSON (the C API
transcribes one input per call). The server's `/v1/audio/transcriptions` takes the same four as
form fields, and several `file` parts in one request: they are transcribed as a batch and
answered with a JSON array of the per-file results, in order (`response_format` `json` or
`verbose_json`). With several `--transcribe` files the CLI prints each file's text under a
`== file ==` line (or a JSON array with a `file` key each), and writes subtitles next to each
input (`talk.mp3` gives `talk.srt`).

**Agreement with transformers** (float32 `WhisperForConditionalGeneration` + `WhisperProcessor`,
torch 2.14, transformers 5.20; `tools/ref_whisper.py` on the samples oxinfer decoded; oxinfer with
`--precision f32 --device cpu`). Clips: LibriSpeech test samples (5.9 s; 29.4 s with timestamps),
25 s of French from a LibriVox recording (language detection), and the first 100 s of a LibriVox
reading of Woolf's *Kew Gardens* (long-form: four or five windows):

| Model | Clip | Log-mel max abs diff | Encoder max abs diff (rel L2) | First-step logits max abs diff | Greedy transcript |
|---|---|---|---|---|---|
| whisper-tiny | 5.9 s | 1.2e-5 | 5.4e-4 (6.7e-6) | 1.2e-4 | identical, 21 tokens |
| whisper-tiny | 29.4 s, timestamps | 1.6e-5 | 6.3e-3 (1.3e-5) | 9.3e-5 | identical, 99 tokens with timestamps, same 5 segment times |
| whisper-tiny | 25 s French | 3.1e-5 | 1.6e-3 (6.8e-6) | 1.5e-4 | identical, 77 tokens; `fr` detected |
| whisper-tiny | 100 s long-form | 3.0e-5 | 4.7e-3 (1.2e-5) | 1.2e-4 | identical, 348 tokens, same 21 segment times |
| whisper-small | 5.9 s | 1.2e-5 | 3.1e-3 (1.1e-5) | 3.3e-5 | identical, 22 tokens |
| whisper-small | 29.4 s, timestamps | 1.6e-5 | 2.4e-3 (5.8e-6) | 2.6e-5 | identical, 110 tokens, same 7 segment times |
| whisper-small | 25 s French | 3.1e-5 | 1.8e-3 (5.5e-6) | 3.0e-5 | identical; `fr` detected; translated: identical, 60 tokens |
| whisper-small | 100 s long-form | 3.0e-5 | 5.2e-3 (1.3e-5) | 2.7e-5 | identical, 285 tokens, same 15 segment times |
| whisper-large-v3-turbo | 100 s long-form | 2.7e-5 | 2.5e-2 (1.4e-4) | 2.5e-4 | identical, 300 tokens, same 16 segment times |
| whisper-large-v3-turbo | 25 s French | 3.2e-5 | 1.2e-2 (9.3e-5) | 5.8e-5 | identical, 29 tokens |
| distil-large-v3 | 100 s long-form | 2.7e-5 | 3.8e-2 (1.4e-4) | 7.2e-5 | identical, 350 tokens, same 21 segment times |
| whisper-base | 100 s long-form | 3.0e-5 | 2.9e-3 (7.2e-6) | 4.3e-5 | identical, 361 tokens, same 22 segment times |
| whisper-medium | 29.4 s, timestamps | 1.6e-5 | 4.7e-3 (8.7e-6) | 4.1e-5 | identical, 102 tokens, same 6 segment times |
| whisper-large-v3 | 25 s French | 3.2e-5 | 1.1e-3 (7.1e-6) | 2.7e-5 | identical, 25 tokens; `fr` detected |
| whisper-tiny.en | 100 s long-form | 3.0e-5 | 7.7e-5 (3.3e-6) | 7.7e-5 | identical, 354 tokens, same 24 segment times |

The options agree too (whisper-small): an initial prompt (88 tokens), `condition_on_prev_tokens`
over the 100 s clip (213 tokens, 11 segments), both together, and the fallback settings (no window
falls back on that clip). The log-mel difference is torch's own: its f32 STFT is off from an f64
evaluation of the same pipeline by up to 1.2e-5, where oxinfer's features are within 7e-8 (f32
rounding), so bit-identical features would mean reproducing torch's FFT rounding. Encoder
differences grow with depth (32 layers for large-v3-turbo) but stay at a relative 1e-4.

**Beam search against openai-whisper** (openai-whisper 20250625, torch 2.15 on the CPU, float32,
the `tiny` and `base` checkpoints; `tools/ref_whisper_beam.py`; oxinfer with `--precision f32
--device cpu` on the Hugging Face copies of the same weights, language given as English). Twelve
LibriSpeech clips of 5.9 to 29.4 s (hf-internal-testing/librispeech_asr_dummy), each decoded by
`whisper.decode` with the same options; the tokens compared include the timestamp tokens where
timestamps are on:

| Model | `beam_size` | `patience` | `length_penalty` | Timestamps | Clips with identical tokens | Max abs diff of the average log-probability |
|---|---|---|---|---|---|---|
| whisper-tiny | 5 | | | no | 12 of 12 (492 tokens) | 2.4e-4 |
| whisper-tiny | 5 | | | yes | 12 of 12 (527 tokens) | 2.3e-4 |
| whisper-tiny | 2 | | | no | 12 of 12 (497 tokens) | 2.4e-4 |
| whisper-tiny | 8 | 2 | | yes | 12 of 12 (524 tokens) | 2.4e-4 |
| whisper-tiny | 5 | | 1.0 | yes | 12 of 12 (527 tokens) | 2.3e-4 |
| whisper-tiny | 4 | 1.5 | 0.6 | no | 12 of 12 (493 tokens) | 2.4e-4 |
| whisper-base | 5 | | | no | 12 of 12 (493 tokens) | 2.2e-4 |
| whisper-base | 5 | | | yes | 12 of 12 (547 tokens) | 2.5e-4 |
| whisper-base | 3 | 2 | 0.8 | yes | 12 of 12 (548 tokens) | 2.5e-4 |

The no-speech probabilities are within 5e-7 of the reference. One 29.4 s clip ends its first
window on a timestamp pair at 21.72 s (whisper-tiny, with timestamps): `whisper.decode` stops there, oxinfer
decodes the rest in a second window, and the first window is what the table compares. Long-form,
against whisper's `transcribe()` with `beam_size=5`, `temperature=0` and
`condition_on_previous_text=False`, on two clips of 90.6 and 75.6 s made of those samples end to
end: the text tokens and the segment times are identical for both clips with both models (tiny:
503 text tokens, 16 and 12 segments; base: 496 text tokens, 15 and 12 segments). Only token-level
bookkeeping differs there: oxinfer keeps both timestamps of the pair that closes a window in the
segment's `tokens`, as transformers does, and openai-whisper keeps one. Not compared: sampled
output (`best_of`, the fallback temperatures), which depends on the random number generator.

**Batches against single inputs**. The same 14 inputs (the twelve clips and the two long ones,
19 to 21 windows in all) as one batch and each alone, comparing every segment's tokens, times,
average log-probability, compression ratio and no-speech probability, the words and the detected
language as printed in the JSON output: identical for all 14 with whisper-tiny greedy (bf16, CPU),
whisper-tiny with `--beam 5 --timestamps --batch 5` (bf16, CPU), whisper-base with `--beam 5
--best-of 3 --fallback` and word timestamps (f32, CPU), and whisper-small with `--beam 3` on CUDA
(bf16, RTX 5080: the encoder windows stacked on the GPU). The unit tests check the same on a tiny
random-weight model without downloaded weights (`src/whisper/batch_tests.rs`: batched encoder and
decoder steps bit for bit in f32, bf16, q8 and q4, whole transcripts with beam search, sampling,
fallback and word timestamps, and the beam search against a slow transcription of
openai-whisper's Python). Not run: batches on Metal and WebGPU (no such device here).

With the default bf16 weights, transcripts stay close but are not always token-identical. On the
100 s clip, whisper-large-v3-turbo in bf16 on Metal gives the f32 transcript token for token;
whisper-small differs in one word of 206, because of the bf16 decoder (bf16 encoder with an f32
decoder, or f16 weights throughout, are token-identical). A q8 decoder: 0 or 1 word of 206 on
whisper-small, 1 of 210 on large-v3-turbo; q4 decoders drift (7 words of 210 on large-v3-turbo,
50 of 206 on whisper-small): not recommended for Whisper.

**Speed** on the M5 Max, 100 s of speech (long-form), wall time including features, language
detection and every window (real-time factor = time / audio duration; lower is faster). Best of
three runs, load average 4-8 from other jobs (one core busy with a compile during the first
rows); `--threads 6` approximates a phone's core count:

| Model | Device | Encoder | Decoder | Total | RTF | Speed |
|---|---|---|---|---|---|---|
| whisper-small | CPU, 18 threads | 2.17 s | 0.66 s (299 tokens) | 2.84 s | 0.028 | 35x real time |
| whisper-small | CPU, 6 threads | 4.58 s | 0.82 s | 5.43 s | 0.054 | 18x |
| whisper-small | Metal | 0.66 s | 0.65 s | 1.32 s | 0.013 | 76x |
| whisper-small, q8 decoder | Metal | 0.54 s | 0.54 s | 1.09 s | 0.011 | 92x |
| whisper-large-v3-turbo | CPU, 18 threads | 9.66 s | 0.57 s (304 tokens) | 10.24 s | 0.102 | 10x |
| whisper-large-v3-turbo | CPU, 6 threads | 19.58 s | 0.70 s | 20.32 s | 0.203 | 5x |
| whisper-large-v3-turbo | Metal | 1.39 s | 0.58 s | 1.99 s | 0.020 | 50x |
| whisper-large-v3-turbo, q8 decoder | Metal | 1.34 s | 0.39 s | 1.76 s | 0.018 | 57x |

On Metal the encoder's attention is one fused kernel (`flash_attn64`: about 3.6 ms per
large-v3-turbo layer, against 10 ms for the GEMM + softmax + GEMM path it replaced, which other
head sizes still use); a large-v3-turbo window then splits into ~160 ms of matmuls, ~115 ms of
attention and ~100 ms of CPU-side layer norms, GELU and residual adds.

The encoder (30 s per window, 1500 positions) is the heavy part: its matmuls and attention go to
the GPU when one is active (Metal, CUDA, WebGPU); decoding is one token at a time on the CPU
(one row per beam and per input of a batch),
where the bf16 / q8 / q4 kernels of the language models apply. Without a GPU (iOS and Android
builds without one, `--device cpu`), everything runs on the SIMD CPU kernels. f32 and f16 weights
run on the CPU only (the GPU backends take bf16, q8 and q4).

Not supported: batching the windows of one long file (they are sequential: each starts at the
previous one's last timestamp), transformers' own beam search (`num_beams`, a different
algorithm from openai-whisper's, which is the one implemented),
voice activity detection or speaker diarization, openai-whisper's word-timing heuristics
(the punctuation merge is there, not the long-word truncation) and `hallucination_silence_threshold`;
the compression ratio of the fallback is computed with a built-in DEFLATE size estimate (greedy
LZ77, fixed Huffman codes) instead of zlib, as no deflate implementation is in the dependency tree.
