# Text to speech (ZONOS3-Mini)

[ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini) (Zyphra, Apache 2.0) speaks 43
languages at 44.1 kHz: a language model of 800M parameters (about 100M active per step) writes
ten codes per 15 Hz frame, and the ZAC 1 codec's decoder renders them. oxinfer reads the GGUF
files of the repository's `gguf/` folder and runs everything on the CPU:

| File | What | Size |
|---|---|---|
| `backbone-ptq.gguf` | the language model (`zonos3`: Q4_0 experts, Q8_0 mixers, the rest F16), with the tokenizer, the condition edges and the silence frame | 0.66 GB |
| `zac-keep2.gguf` | the codec decoder (`mossdual_istft`), two transformer layers per stage | 0.45 GB |
| `zac.gguf` | the full-size decoder (optional, `--snac full`) | 2.2 GB |
| `spk-encoder.gguf` | the speaker encoder (`mect_spk`), for cloning a voice from a clip | 45 MB |
| `voices/<language>/*.npy` | the 27 default voices: a speaker embedding and reference codes each | 0.3 MB |

A repo id fetches the first, second, fourth and fifth (1.2 GB), not the 30 GB repository.

```sh
./target/release/oxinfer --model Zyphra/ZONOS3-Mini --speak "Hello there, how are you today?" -o hello.wav
./target/release/oxinfer --model Zyphra/ZONOS3-Mini --speak "Bonjour, comment allez-vous ?" --voice Gaspard -o fr.wav
# a voice cloned from a clip (its speaker embedding), or from saved reference codes
./target/release/oxinfer --model Zyphra/ZONOS3-Mini --speak "Something new." --ref-audio me.wav -o clone.wav
./target/release/oxinfer --model Zyphra/ZONOS3-Mini --speak "Something new." --voice codes.npy -o clone.wav
# one utterance per line -> out-1.wav, out-2.wav, ...; or the sentences of a long text joined in one file
./target/release/oxinfer --model Zyphra/ZONOS3-Mini --speak-file lines.txt --voice Imogen,Rupert -o out.wav
./target/release/oxinfer --model Zyphra/ZONOS3-Mini --speak - --speak-split -o story.wav < story.txt
# the voice's reference codes in the prompt instead of its embedding, with speaker guidance; the full decoder
./target/release/oxinfer --model Zyphra/ZONOS3-Mini --speak "Closer to the clip." --voice Rupert \
    --speak-json '{"speaker_mode": "reference", "speaker_cfg_scale": 1.5}' --snac full -o ref.wav
```

`--model` is the repo id, a folder holding `gguf/` (or the files themselves), or the backbone
file with the others beside it. Voices: Cillian, Imogen, Jett, Morgana, Rupert, Sakura, Savannah
(the default), Thaddeus, Willow, Wyatt (English); Gaspard (French); Liesel, Wolfgang (German);
Ananya, Arjun (Hindi); Ginevra, Lorenzo (Italian); Minjun, Seoyeon (Korean); Haoran, Meilin
(Mandarin); Leonor, Thiago (Portuguese); Rodrigo, Ximena (Spanish); Chaitanya, Sravya (Telugu);
`--voice ""` for no speaker conditioning. The language is not an input: the model reads it from
the text, and any voice speaks any language.

Sampling follows the reference's defaults (`registries/generation.json`): temperature 0.75,
top-k 80, top-p 0.69, a repetition penalty of 2.4 on the codes of the last 96 rows in the first 6
codebooks, at most 2000 rows (133 s; the model's context is 32768 rows). `--speak-json` takes
every field of `oxinfer::zonos::ZonosSpeak`: `voice`, `speaker_mode` (`embedding` / `reference`),
`temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`, `repetition_window`,
`repetition_codebooks`, `max_tokens`, `seed`, `silence_prefill_frames`, `speaker_cfg_scale`,
`loudness` (LUFS), `fmax` (Hz), `speaking_rate` (UTF-8 bytes per second; about 15 is ordinary
speech), `condition_buckets`, `headroom_db`, `fade_out_ms`.

In Rust:

```rust
use oxinfer::zonos::{Zonos, ZonosOptions, ZonosSpeak};
let z = Zonos::load("Zyphra/ZONOS3-Mini", &ZonosOptions::default())?;
let mut o = ZonosSpeak::default();
o.voice = Some("Imogen".into()); // a default voice, an embedding or codes as .npy, or an audio clip
let speech = z.speak("Hello there!", &o)?; // Speech { audio: Audio { samples, sample_rate: 44100 }, tokens, stats, .. }
speech.audio.write_wav("hello.wav")?;
let embedding = z.speaker_embedding_file("me.wav".as_ref())?; // 192 values
// The stages on their own:
let (prompt, twin) = z.prompt("Hello there!", &o)?;                          // the rows of ids
let g = z.generate(&prompt, twin.as_ref(), &o, false, &mut |_, _| oxinfer::engine::Flow::Continue)?;
let (frames, n) = z.frames(&prompt, &g.rows, g.eos_frame);                   // [n, 10] codes
let pcm = z.decoder().decode(&frames, n)?;                                   // n * 2940 samples
```

The server's `/v1/audio/speech` serves it like Orpheus (`voice` one of the names above; `seed`,
`temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_tokens`, `speaker_mode`,
`speaker_cfg_scale`, `loudness`, `fmax`, `speaking_rate` as extensions; model options `threads`,
`decoder=keep2|full`, `exact=0`). A request cannot name a file on the server, so cloning from a
clip is for the command line and the library.

## The pipeline

**Text.** Line breaks and tabs become spaces; the text is otherwise taken as written and encoded
by the SuperBPE tokenizer embedded in the backbone file (8192 entries; ids shifted by 4, between
a start id 2 and an end id 3). The reference also expands written forms ("$5.32", "1,234") into
words with [zyphra-textnorm](https://github.com/Zyphra/zyphra-textnorm), a Rust port of the NeMo
grammars for nine languages. That is not part of oxinfer: write numbers, dates and symbols the
way they should be read.

**Prompt.** Rows of 11 ids (ten codebooks and the text), in the layout the model was trained on:

1. the speaker: one row whose embedding is replaced by the projection of a 192-d speaker
   embedding, or the reference prefix, the codes of a reference clip (at most 60 s), codebook
   `j` delayed by `j` rows and the tail flushed with padding;
2. three condition rows (loudness, bandwidth, speaking rate), each replaced by a learned row per
   bucket; "unknown" unless set;
3. the text, one token per row;
4. three frames of the codec's silence, already in the delay pattern, so that the audio starts
   cleanly (they are part of the output: 0.2 s).

With `speaker_cfg_scale` other than 1 a second sequence without the speaker and with unknown
conditions runs along, and the logits are `uncond + scale * (cond - uncond)`.

**Language model** (`oxinfer::zonos::lm`). 11 layers of width 1024 joined by Full AttnRes: every
sub-layer reads a softmax-weighted mix of all earlier outputs instead of a residual stream. Nine
layers mix the sequence with Gated DeltaNet (4 heads of 128, a causal convolution of kernel 4
on q / k / v), layers 4 and 8 with global attention (16 heads over 4 key / value heads,
normalized q and k, RoPE, a gate per head). Layer 1 has a SwiGLU feed-forward; the others a
latent mixture of experts: the row is projected to 256 dimensions, 2 of 64 experts run on it, and
the sum is projected back. One linear layer gives the 10 x 4098 logits of the next row, capped
with `15 tanh(x / 15)`. Each step writes one row: codebook `j` of frame `f` is in row `f + j`.
Generation ends 10 rows after a codebook holds the end token, which completes the last frames.

**Codec decoder** (`oxinfer::zonos::codec`). The semantic code's vector goes through four
causal ConvNeXt blocks, the nine residual codes add theirs: a 1024-d latent per frame. Five
stages of causal transformer layers with attention over a limited past (10 s at 15 Hz down to 2 s
at 240 Hz) double the frame rate four times. At 240 Hz eight ConvNeXt blocks feed two inverse
STFTs (windows of 736 and 2208 samples, hop 184) whose outputs a 127-tap filter combines; a
windowed-sinc resampler brings 44160 Hz to 44100 Hz. The reference servers' output stage follows:
3 dB of headroom and a soft limiter above -1 dBFS.

**Speaker encoder** (`oxinfer::zonos::speaker`). The clip is resampled to 16 kHz with
torchaudio's kernel and cut to 60 s; 80 Kaldi filter-bank energies per 10 ms, mean-normalized,
go through MECT-B2 (a 3x3 convolution, then 14 blocks of grouped convolutions with one
transformer layer each over the whole clip) and attentive statistics pooling to 192 values.

## Agreement with the reference

The reference is Zyphra's PyTorch code ([Zyphra/ZONOS3-Mini](https://github.com/Zyphra/ZONOS3-Mini):
`zonos3.reference` for the language model, `ISTFTCodec` for the decoder,
`MECTFbankSpeakerEmbedding` for the speaker encoder), in float32 on the CPU, on the released
weights. `examples/zonos_check.rs` dumps oxinfer's side and `tools/zonos_ref.py` compares:

```sh
R=ZONOS3-Mini   # a checkout of github.com/Zyphra/ZONOS3-Mini
H=hf            # lm/zonos3_mini/, autoencoder/, speaker_encoders/ and gguf/ of the Hugging Face repo
cargo run --release --example zonos_check -- gen $H out "Hello there, how are you today?" --voice Savannah --steps 400
python tools/zonos_ref.py lm $R $H out --gguf $H/gguf/backbone-ptq.gguf
cargo run --release --example zonos_check -- decode $H $H/gguf/voices/English/Rupert_en-British_M_1.codes.npy out.f32
python tools/zonos_ref.py codec $R $H $H/gguf/voices/English/Rupert_en-British_M_1.codes.npy out.f32
cargo run --release --example zonos_check -- embed $H clip.wav emb.f32
python tools/zonos_ref.py embed $R $H clip.wav emb.f32
```

**Tokenizer and prompt.** 21 lines in 15 languages (Latin with diacritics, Cyrillic, Japanese,
Chinese, Korean, Arabic, Hebrew, Devanagari, Telugu, Thai, emoji, digits and symbols): the ids
are those of the `tokenizers` library. The prompt rows, the condition rows and the twin of
speaker guidance are identical to `zonos3.tts.plan`'s in the five runs below.

**Language model.** The reference loads `model.pth` (bf16); the GGUF backbone holds the same
model with its experts rounded to Q4_0 and its mixers to Q8_0. With the GGUF's weights loaded
into the reference (`--gguf`), greedy generation is the reference's row for row:

| Run | Prompt rows | Generated rows | Identical | Largest logit difference |
|---|---|---|---|---|
| speaker embedding, 7 words | 17 | 43 | 43 | 8.8e-5 |
| speaker embedding, 25 words | 44 | 116 | 116 | 9.0e-5 |
| reference prefix (13 s of codes), 25 words | 250 | 120 | 120 | 8.2e-5 |
| French, speaker guidance 1.5 (two sequences) | 31 + 30 | 91 | 91 | 1.1e-4 |
| no voice, no silence prefill | 14 | 59 | 59 | 8.0e-5 |

The end frame is the same in each. Two things move the output away from that:

- The int8 kernels (`ZonosOptions::exact = false`, the way ggml computes Q8_0 / Q4_0 layers, with
  the activations rounded to 8 bits per block of 32): the second run follows the reference for 2
  rows of 116 (logits up to 1.5 apart). They are not faster here (1.2 s for those 116 rows either
  way on 64 cores), so the f32 products on the dequantized weights are the default.
- The quantization itself: against the reference on its own bf16 weights the logits of the first
  step are up to 2.7 apart and the greedy codes differ from the first row (73 of 1160 equal).
  This is the published file's post-training quantization, not something oxinfer adds.

Sampling uses oxinfer's own random numbers, so a seed does not give the reference's take.

**Codec decoder.** The frames of two default voices, against `ISTFTCodec` built from
`autoencoder/generator.pt` and the head checkpoints (fp32):

| Decoder | Clip | Max sample difference | SNR | SNR against the reference with its matrices rounded to F16 |
|---|---|---|---|---|
| `zac-keep2.gguf` | 198 frames, 13.2 s | 7.5e-4 (peak 0.45) | 69.1 dB | 83.2 dB |
| `zac-keep2.gguf` | 284 frames, 18.9 s | 1.1e-3 (peak 0.53) | 63.6 dB | |
| `zac.gguf` | 198 frames, 13.2 s | 3.1e-4 (peak 0.42) | 72.3 dB | 85.3 dB |

The GGUF stores the matrices as F16, which accounts for most of the difference.

**Speaker encoder.** A 13.2 s WAV: the embedding is the reference's to 5e-6 (cosine 0.9999999).
An MP3 gives cosine 0.998, because oxideav and libsndfile decode it to slightly different
samples. The embeddings the repository ships for its default voices are not what the encoder
gives for the clips shipped next to them (cosine 0.895 and 0.855 for two voices, for the
reference encoder as for oxinfer's): they were computed from other audio.

**End to end.** 20 Harvard sentences (159 words), default sampling, seed 42, transcribed by
oxinfer's whisper-small and scored in lower case without punctuation:

| Voice | WER | Errors |
|---|---|---|
| Savannah, its shipped embedding | 1.3% | "smokey", "50" |
| Rupert, reference codes in the prompt | 1.9% | the same two, "park truck" |
| Rupert, its shipped embedding, the full decoder | 2.5% | the same two, two words added after a sentence |
| cloned from Savannah's clip (the speaker encoder) | 11.9% | the same two; 2 of 20 utterances go on after the sentence (4 and 11 words) |
| Gaspard, 8 French sentences (95 words) | 3.2% | "18 heures" for "dix-huit heures" (2), "ont marché" |

The model sometimes keeps talking after the text, more often with an embedding computed from a
short clip than with the shipped ones. The reference servers have an alignment-based guard for
this (LACI: attention heads that follow the reading position, which stop or redo an utterance);
it is not implemented here.

The cloned voice is the clip's: the speaker embedding of the generated sentences has a cosine of
0.70 to 0.85 with the clip's embedding, against 0.12 to 0.14 with another speaker's clip (the
two clips: 0.18); sentences in a shipped voice score 0.66 to 0.76 against that voice's clip.

## Speed

On a 64-core x86 machine shared with other jobs: the language model writes 62 to 95 rows per
second (15 are real time; a prompt of 44 rows is read in 0.07 s), `zac-keep2` decodes 13.2 s of
audio in 0.63 s and `zac` in 2.4 s, the speaker encoder embeds a 13 s clip in 1.9 s. Whole
utterances of 3 to 7 s: real-time factor 0.26 to 0.42. The process peaks at 1.3 GB of memory.

## Not done

- **Text normalization**: numbers, dates, currencies and abbreviations are not expanded (see
  above).
- **Reference codes from a clip**: the codec's encoder (`zac-encoder.gguf`, 4 GB, with the
  Wav2Vec2-BERT teacher `ssl-teacher.gguf`, 1.5 GB) is not implemented, so the reference mode
  takes codes that already exist (the default voices', or a `.npy` another runtime saved). A
  clip is cloned through its speaker embedding.
- **Streaming**: the decoder runs on the whole utterance. `--speak-stream` and the server's
  `pcm` / SSE responses deliver the audio in one-second chunks once it is generated, so the time
  to the first audio is the time of the whole utterance. The reference decodes incrementally.
- **Batching**: several utterances run one after the other.
- **GPU**: everything is on the CPU.
- LACI, the steering bundle (`steer.gguf`: emotion, style and accent directions), accent
  adapters, speech editing (`backbone-ptq-edit.gguf`), phoneme / start / pause tokens (the
  released backbone has none), decoding the reference's tail as left context, speaker blends.
- The PyTorch files (`lm/zonos3_mini/model.pth`, `autoencoder/`) and the WebGPU shards (`web/`)
  are not read; they hold the same model.
