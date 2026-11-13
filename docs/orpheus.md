# Text to speech (Orpheus)

[Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) is a Llama 3.2 3B fine-tune that writes
audio as extra vocabulary tokens: 7 tokens per frame, codes of the three codebooks of the
[SNAC](https://github.com/hubertsiuzdak/snac) codec, which decodes them to 24 kHz audio (a frame is
2048 samples, 85 ms). `canopylabs/orpheus-3b-0.1-ft` is gated (accept the licence and set
`HF_TOKEN`); `unsloth/orpheus-3b-0.1-ft` is an ungated bf16 copy of it. The SNAC decoder
(`hubertsiuzdak/snac_24khz`, 80 MB, `pytorch_model.bin`) is fetched on first use.

```sh
./target/release/oxinfer --model unsloth/orpheus-3b-0.1-ft --speak "Hey there, I'm Leo. <chuckle> Nice to meet you." --voice leo -o hi.wav
./target/release/oxinfer --model unsloth/orpheus-3b-0.1-ft --speak - --quant q4 --speak-stream < text.txt   # faster than real time
./target/release/oxinfer --model hubertsiuzdak/snac_24khz --decode-tokens tokens.json -o out.wav          # SNAC alone
# a batch: one utterance per line (or several --speak), generated together -> out-1.wav, out-2.wav, ...
./target/release/oxinfer --model unsloth/orpheus-3b-0.1-ft --quant q4 --speak-file lines.txt --voice tara,leo,mia -o out.wav
# a long text: its sentences as one batch, joined in one file
./target/release/oxinfer --model unsloth/orpheus-3b-0.1-ft --quant q4 --speak - --speak-split -o story.wav < story.txt
# voice cloning: the pretrained model, a reference clip and what it says (repeatable)
./target/release/oxinfer --model unsloth/orpheus-3b-0.1-pretrained --ref-audio me.wav --ref-text "What I say in the clip." --speak "Something new." -o clone.wav
# other languages
./target/release/oxinfer --model Our-Org-2/3b-fr-ft-research_release --speak "Bonjour, comment allez-vous ?" --voice pierre -o fr.wav
# the codec alone, any SNAC model: audio -> codes (JSON array per codebook) -> audio
./target/release/oxinfer --model hubertsiuzdak/snac_44khz --snac-encode music.wav -o codes.json
./target/release/oxinfer --model hubertsiuzdak/snac_44khz --decode-codes codes.json -o back.wav
```

Voices of the English model: tara (default), leah, jess, leo, dan, mia, zac, zoe; emotion tags
(`<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`) go in the
text. Sampling follows the official package: temperature 0.6, top-p 0.8, repetition penalty 1.1
(over the prompt and the reply, as transformers and vLLM apply it), at most 1200 tokens (about
14 s; `--max-tokens` for more). The prompt is the one `orpheus_tts` builds (start of human,
`<|begin_of_text|>voice: text`, end of text / human, start of AI, start of speech) and generation
stops at end of speech. By default sampling is constrained to the tokens valid at each frame
position (plus the stop tokens), which also computes only those 4096 rows of the 157k-row output
layer; `--unconstrained` samples the whole vocabulary like the reference.

In Rust (`examples/speak.rs`):

```rust
use oxinfer::tts::{Speaker, SpeakerOptions, SpeakOptions};
let mut lo = SpeakerOptions::default();
lo.quant = oxinfer::model::Quant::Q4;
let speaker = Speaker::load("unsloth/orpheus-3b-0.1-ft", &lo)?;
let mut opts = SpeakOptions::default(); // voice, temperature, top_p, top_k, repetition_penalty, max_tokens, seed, ...
opts.voice = Some("leah".into());
let audio = speaker.speak("Hello there!", &opts)?; // Audio { samples: Vec<f32>, sample_rate: 24000 }
audio.write_wav("hello.wav")?;
// Streaming: chunks of audio while tokens are generated; Flow::Stop (or speaker.cancel_handle()) stops.
let speech = speaker.speak_stream("A longer sentence...", &opts, &mut |chunk| {
    play(chunk.samples); // chunk.start: index of its first sample
    oxinfer::engine::Flow::Continue
})?;
println!("RTF {:.2}", speech.stats.rtf());
```

## Batched utterances

`Speaker::speak_batch` generates several utterances together, each with its own text and
`SpeakOptions` (voice, sampling settings, seed, limits, references):

```rust
use oxinfer::tts::Utterance;
let (leo, mia) = (SpeakOptions { voice: Some("leo".into()), ..Default::default() },
                  SpeakOptions { voice: Some("mia".into()), seed: 7, ..Default::default() });
let speeches = speaker.speak_batch(&[Utterance::new("Hello there!", &leo), Utterance::new("And a longer one.", &mia)])?;
// speeches[i]: the Speech (audio, tokens, frames, finish, stats) of utterance i
let (audio, parts, stats) = speaker.speak_long(&long_text, &opts, 0)?; // the sentences as a batch, joined
```

- **Language model.** The prompts are prefilled one after another; then every decode step feeds
  the next token of each running utterance as one column of a single pass
  (`QwenModel::forward_hidden_batch`, `src/qwen_batch.rs`): the layer matmuls run on the
  multi-token decode kernels of
  [Multi-token passes](performance.md#multi-token-passes-and-moe-experts-on-the-cpu), which read the weights
  once for all columns, while each utterance keeps its own session (KV caches, position), and its
  attention is the one-token attention over its own cache. Up to 15 utterances share a pass with
  quantized weights (q8, q4, GGUF), 7 with bf16; a wider batch takes several passes per step.
- **Per utterance**: its sampler state and seed, repetition-penalty set, loop guard, length caps
  and stop. An utterance that ends leaves the batch and the others go on (ragged lengths).
  Unconstrained utterances share one pass through the whole output layer.
- **SNAC.** `Snac::decode_batch` decodes all the utterances in one pass per layer, the clips laid
  end to end with each one padded, and given its noise, on its own (groups of about 22 s of
  audio per pass).
- **Sentences.** `--speak-split` / `Speaker::speak_long` / `tts::split_sentences` cut a text at
  sentence ends (`.`, `!`, `?`, `…`, `。`, `！`, `？` and line breaks), join short neighbours up to
  240 bytes, and speak the pieces as one batch. Each piece is its own utterance (so a long text
  is no longer cut at `max_tokens`), and its prosody starts afresh.
- **CLI**: several `--speak`, or `--speak-file FILE` (one utterance per line, `-` for stdin);
  `-o out.wav` writes `out-1.wav`, `out-2.wav`, ...; `--voice a,b,c` gives utterance i the i-th
  voice, cycling. **Server**: an array as `input` of `/v1/audio/speech` (see
  [the server](server.md#openai-compatible-server)). `examples/speak.rs` takes `--also TEXT`. The C API has
  no batch call yet, and a batch is not streamed (its audio is decoded when all of it is
  generated).

Every utterance of a batch is the utterance generated alone with the same options: the decode
kernels give each column the bits of a one-token pass, so on the CPU the tokens and the samples
are identical, greedy or sampled. Measured on the
Orpheus 3B Q4_K_M GGUF (CPU, 12 threads): a batch of 9 utterances in 8 voices (5 greedy without
SNAC noise, 4 sampled at 0.6 / 0.8 with it; 2 of the 9 unconstrained; 99 to 400 tokens each, 7
ending on a stop token at different steps and 2 at the 400-token limit; 2242 tokens in 399 steps)
against each utterance alone: 9 of 9 token sequences identical, and all 653312 samples
bit-identical (`real_batch_matches_alone`, an ignored test taking `OXINFER_TTS_MODEL`). The
tests that need no weights check the same on a random 2-layer Orpheus-shaped model and a small
random SNAC at bf16, q8 and q4 (9 utterances with mixed voices, seeds, samplers and limits:
`batch_matches_alone`), the batched language-model steps against one-token steps with sequences
leaving the batch (`batch_steps_match_one_token_steps`), and the batched SNAC decode against
single decodes with f32 and bf16 weights (`batch_decode_matches_alone`). The bf16 safetensors
checkpoint was not run in a batch on real weights (it was not on this machine).

A batch decodes on the CPU kernels (the GPU decode plans take one sequence), so with GPU
decoding on (`gpu_decode`, the default where a backend has it) a lone utterance and the same
utterance in a batch come from different kernels and can part where two tokens are close.

## Voice cloning

The pretrained base (`canopylabs/orpheus-3b-0.1-pretrained`, ungated copy
`unsloth/orpheus-3b-0.1-pretrained`) has no named voices; it continues in the voice of
reference utterances given in the prompt. The prompt is the one of canopylabs' notebook for
that model: per reference a finished turn (start of human, `<|begin_of_text|>transcript`, end
of text / human, start of AI, start of speech, the clip's SNAC codes as audio tokens, end of
speech, end of AI), then the turn for the new text (no voice name). The clip is resampled to
24 kHz and encoded by the SNAC encoder (also in Rust, see below). The model card notes that it
was not trained for cloning, so more reference pairs give a steadier voice; 5-10 s of clean
speech per reference works. Finetuned checkpoints accept references too, but follow them less.

```rust
use oxinfer::tts::{Speaker, SpeakerOptions, SpeakOptions, VoiceRef};
let speaker = Speaker::load("unsloth/orpheus-3b-0.1-pretrained", &SpeakerOptions::default())?;
let mut opts = SpeakOptions::default();
// Encoded once, reusable (serializable): from samples at any rate, a file, or an av::AudioClip.
opts.references.push(speaker.encode_reference(&samples, 16000, "What the clip says.")?);
opts.references.push(speaker.encode_reference_file("other.wav".as_ref(), "Its transcript.")?);
// or encoded at each call: VoiceRef::from_file("What it says.", "ref.wav")
let audio = speaker.speak("Something new.", &opts)?;
```

In JSON (C API, `--speak-json`): `"references": [{"text": "...", "audio": "ref.wav"}]`, or the
objects `oxi_speaker_encode_reference()` / `oxi_speaker_encode_reference_file()` return.

**Agreement** (one 6.5 s reference, a second sentence): the SNAC codes of the clip are identical
to the `snac` package's, and the prompt (586 tokens) identical to the notebook's; greedy
generation from it matches transformers float32 for 120 of 120 tokens.

**Intelligibility and voice**: two references (a male and a female voice, 6.5 s each, made by
the finetuned model's leo and tara), the four sentences of the WER test below, sampled with the
default settings: WER 5.7% for both (3 errors in 53 words each: "riverbank", "rainforest", as for
the finetuned voices). The median pitch follows the reference: 144-167 Hz for the leo clip
(151 Hz), 183-197 Hz for the tara clip (180 Hz), against 93-132 Hz for the pretrained model
without a reference.

## Other languages

The multilingual research release (`canopylabs/3b-{fr,de,ko,hi,zh,es_it}-ft-research_release`,
gated; ungated copies `Our-Org-2/3b-*-ft-research_release`, fp32 safetensors read as bf16) uses the
same prompt format with its own voices: French pierre, amelie, marie; German jana, thomas, max;
Korean 유나, 준서; Hindi ऋतिका; Mandarin 长乐, 白芷; Spanish javi, sergio, maria; Italian pietro,
giulia, carlo (`oxinfer::tts::voices_for`). `Speaker::info()` reports the languages (from the
model card), the known voices and whether the checkpoint is a pretrained base. Greedy French
tokens match transformers float32 (150 of 150).

WER of four sentences per voice (default sampling), transcribed with `--language`:

| Model | Voices | whisper-small | whisper-large-v3-turbo |
|---|---|---|---|
| French (`3b-fr-ft`) | pierre, amelie | 8% (24 utterances, 3 seeds) | 0% (8, seed 42) |
| French | marie | 58% (12, 3 seeds) | 81% (4) |
| German (`3b-de-ft`) | jana, thomas, max | 16% (12) | 1.7% (12) |

The voice "marie" of the French research model mostly produces wrong words (the same with
transformers' greedy tokens, so it is the checkpoint); the others are intelligible, and most of
whisper-small's errors are its own (whisper-large-v3-turbo hears them correctly).

## The SNAC codec

`oxinfer::snac::Snac` loads any SNAC checkpoint: `encode(&samples)` (the reference's `encode`,
input zero-padded to a multiple of the hop times `lcm(vq_strides[0], attn_window_size)`) and
`decode(&codes)`. The 32 and 44 kHz models (four codebooks at 8 / 4 / 2 / 1 times the frame
rate, music and general audio, not used by any Orpheus model) add local windowed attention after
the encoder and before the decoder: LayerNorm, 64-dim heads attending within windows of 32 steps
with rotary positions restarting in every window. Against `snac` 1.2.1 (float32, noise off), on a
4.9 s clip (resampled for 32 / 44 kHz) and 64 frames of random codes:

| Model | Encoder codes | Decode of the clip's codes | Decode of random codes | bf16 weights |
|---|---|---|---|---|
| snac_24khz | identical (57 / 114 / 228) | 1.6e-7 (127 dB) | 2.2e-6 (118 dB) | 58 / 46 dB |
| snac_32khz | identical (52 / 104 / 208 / 416) | 2.2e-7 (126 dB) | 5.6e-6 (118 dB) | 65 / 52 dB |
| snac_44khz | identical (72 / 144 / 288 / 576) | 3.3e-7 (126 dB) | 2.3e-6 (121 dB) | 64 / 54 dB |

(max abs difference and SNR.) Encoding 5 s takes 0.08-0.15 s. The reference requires lengths that
are a multiple of the attention window; here a shorter last window attends within itself.

With the `oxideav` feature, `Audio::to_frame()` / `AudioChunk::to_frame()` give an F32 mono
`oxideav_core::AudioFrame` with its `CodecParameters` (pts in samples), and `Audio::to_clip()` an
`av::AudioClip`. `SpeakOptions` deserializes from JSON; the C API takes it that way:
`oxi_speaker_load`, `oxi_speak` (PCM, optional chunk callback), `oxi_speak_wav`,
`oxi_speaker_cancel`, `oxi_speaker_stats_json` (`include/oxinfer.h`).

Streaming decodes overlapping windows of frames as they arrive, as the official decoder does (it
decodes the last 4 frames and keeps the second: 1 frame of left context, 2 of lookahead, and it
drops the first and last frames). Here every frame is emitted and the context is an option: the
decoder's receptive field is under 3 frames each way, so `context_left` / `context_right` 3 / 3
streams exactly the full decode (max abs diff 3e-7), the default 2 / 2 differs from it by at most
1.7e-3 (59 dB SNR), and the official 1 / 2 by 2.4e-2 (27 dB). The SNAC noise blocks draw their noise per (seed, layer, absolute position), so
overlapping windows see the same noise as a whole-clip decode.

Llama 3 itself runs on the same code as Qwen3 (full attention without query / key norms, llama3
RoPE frequency scaling), so any Llama 3.x checkpoint also works as a text model (`--model` with
the Llama 3 chat template), safetensors or GGUF (see [Llama 3 GGUF](weights.md#llama-3-gguf)).

GGUF conversions of Orpheus load the same way, as a path or `org/repo:QUANT`; the audio tokens
are in the GGUF vocabulary (156940 tokens, the 28683 added ones as user-defined tokens), and the
constrained audio head reads its rows of the output layer in place, in the file's own format:

```sh
./target/release/oxinfer --model isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF:Q4_K_M --speak "Hello there!" -o hi.wav
./target/release/oxinfer --model lex-au/Orpheus-3b-FT-Q8_0.gguf:Q8_0 --speak "Hello there!" -o hi.wav
```

On the backbone's first speech-token distribution (tara, a short prompt, CPU) the Q8_0 / Q4_K_M
files match the bf16 safetensors model with the same top 10 and KL 3.4e-4 / 2.6e-3. Eight
sentences (102 words, tara, default sampling, one seed each) transcribed by whisper-small: WER
0% for Q4_K_M, 1.0% for Q8_0 ("rainforests" -> "rainforest"), 0% for the safetensors model at
q4 and q8. Decode speed for a 7 s utterance (~600 tokens, not streamed, 3 interleaved runs, load
7-18):

| Weights | CPU | Default (Metal prefill) |
|---|---|---|
| GGUF Q4_K_M (2.4 GB; Q4_K with Q6_K for some tensors) | 99-102 tok/s | 96-99 tok/s |
| safetensors `--quant q4` | 112-116 tok/s | 113-118 tok/s |
| GGUF Q8_0 | 68-69 tok/s | 69-72 tok/s |
| safetensors `--quant q8` | 68-69 tok/s | 70-72 tok/s |

Q4_K_M decodes faster than real time (82 tokens/s) but ~13% slower than oxinfer's own q4: it keeps
the output layer and half the value and down projections at 6 bits (Q6_K).

**Agreement** with the reference: transformers 5.20 float32 `LlamaForCausalLM` + the `snac` 1.2.1
package, with the token handling of `orpheus_tts` (decoder.py) and the SNAC noise disabled on both
sides (it is random in the reference):

| Check | Result |
|---|---|
| Backbone last-token logits (Orpheus 3B, 14-token text prompt) | max abs diff 3e-5, same top 10 |
| Greedy speech tokens, tara, "The weather today is sunny with a light breeze from the west." (rep. penalty 1.1), CPU | identical, 379 of 379 tokens (54 frames + stop), constrained and unconstrained |
| Same, leo, "Could you send me the report by Friday? ...", Metal prefill | identical, 316 of 316 tokens |
| Waveform of those tokens after SNAC | max abs diff 3.3e-7 / 9.5e-7 (123 dB SNR) |
| SNAC alone, 400 random frames (34 s), f32 weights | max abs diff 6.6e-6 (117 dB); bf16 weights (`snac_f32=0`) 1.4e-2 (47 dB) |
| Streaming windows 1 / 2 vs the official decoder's windows | max abs diff 3e-7 |

q8 and q4 weights change the greedy path within a few tokens (greedy decoding of audio codes is
not a stable regime anyway); their speech is as intelligible as bf16 by the measure below.

**Greedy decoding and quantized weights.** Greedy (`--temp 0`, not the default) usually ends
normally at every precision: 183 of 184 greedy runs (20 texts, 2-8 voices, bf16 / q8 / q4) did.
The failure is a fixed point: on "The weather today is sunny with a light breeze from the
west." in tara's voice the q4 path says "The weather today", then repeats one frame (all 7
codes) forever, a held sound, with end of speech at probability ~1e-8, until `max_tokens`
(171 frames). The repetition penalty cannot break it (it divides any seen token's logit by 1.1,
however often it was seen); it is not a slow drift of the end-of-speech probability, which stays
far from the top as long as speech goes on and wins in one step when the text is done. Normal
speech never repeats a whole frame (0 repeats in all other runs, sampled q4 included), so two
guards are on by default:

- `loop_guard` (3): when a frame repeats the one before it 3 times in a row, the rest of the
  utterance is sampled at temperature >= 0.6, top-p <= 0.8 (a fallback, like Whisper's); after 8
  loops it ends. On the failing case, 4 of 5 fallback seeds then speak the whole sentence and
  stop (single-frame escapes rarely recovered: 2 of 5);
- `max_secs_per_byte` (0.25): at most 3 s plus 0.25 s per UTF-8 byte of text, about 3x the
  slowest speech in any of the supported languages, so a short text cannot run for 14 s.

Sampling (the default) never looped in these tests.

**Intelligibility**: four sentences in four voices each, sampled with the default settings,
transcribed by oxinfer's whisper-small and scored with jiwer (lower case, punctuation removed):
bf16 (tara, leah, leo, zac) WER 4.7%, q4 streamed (tara, jess, dan, mia) WER 4.7% (10 errors in
212 words each). 8 of the 10 come from "river bank" transcribed as "riverbank" (a substitution
and a deletion, in all four voices); the other two are "rainforests" -> "rainforest" and
"frog" -> "fraud" (bf16), "rainforest" twice (q4): about 1% of words actually misheard.

**Speed** on the M5 Max (18 threads, other jobs running, so the best case is a little higher),
an 8.5-10 s utterance, streamed (sliding windows, 2 / 2 context). Real-time output needs
7 x 24000 / 2048 = 82 tokens/s; the real-time factor (RTF) is total time / audio duration:

| Weights | Device | Decode | SNAC | RTF | First audio | Real time? |
|---|---|---|---|---|---|---|
| bf16 | CPU | 43 tok/s | 0.7 s | 2.0 | 0.62 s | no |
| q8 | CPU | 68 tok/s | 0.9 s | 1.33 | 0.50 s | no |
| q4 | CPU | 108-128 tok/s | 0.75 s | 0.76-0.90 | 0.38 s | yes |
| bf16 | Metal | 44 tok/s | 0.9 s | 2.0 | 0.59 s | no |
| q8 | Metal | 68 tok/s | 1.0 s | 1.32 | 0.41 s | no |
| q4 | Metal | 112-128 tok/s | 0.8 s | 0.69-0.85 | 0.28 s | yes |

Decoding is one token at a time on the CPU and memory-bound (2.8B weights per token once the
output layer is cut to the audio rows), so the GPU only speeds up the prompt; q4 is the setting
that keeps up with playback here. SNAC costs about 3 ms per frame decoded once (f32 weights;
streaming decodes each frame in 5 windows, ~14 ms per 85 ms frame).

**Speculative decoding** was measured and not added: no draft gives a speed-up. Acceptance per
generated token over 2976 tokens (8 utterances; "greedy" = the draft matches the greedy token,
"sampled" = the expected accepted mass `sum min(p, q)` at temperature 0.6, top-p 0.8):

| Draft | Greedy | Sampled | Tokens per verify pass (2 drafted) |
|---|---|---|---|
| n-gram prompt lookup (n = 1-4) | 0-0.1% | 0-0.1% | 1.00 |
| the previous frame's token at the same position | 0% | 1.4% | 1.01 |
| edwindn/orpheus-1b-0.1-weightDecay-chkpt78201 (Llama 3.2 1B, same tokens) | 20% | 31% | 1.24 / 1.41 |
| edwindn/orpheus-1b-0.1-finetuned-v3 | 20% | 31% | 1.24 / 1.41 |
| Gapeleon/Orpheus-1B-pt | 26% | 38% | 1.33 / 1.52 |

Audio codes do not repeat, so lookup drafts find nothing, and the community 1B models (canopylabs
never released its smaller sizes) cost about 0.35 of the 3B per token: at best 1.5 tokens per
pass for 1.7-2x the work of one target token, a slowdown. Medusa-style heads do not exist for
Orpheus.

Not supported: batched utterances on the GPU decode plans (a batch decodes on the CPU kernels),
streaming a batch, and a batch call in the C API. Korean, Hindi, Mandarin and the
Spanish / Italian model use the same code but were not tested.
