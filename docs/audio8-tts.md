# Text to speech (Audio8 TTS)

[Audio8-TTS-Realistic-Air](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air) (Edge0,
Apache 2.0, `model_type: arktts`) speaks 23 languages at 44.1 kHz and clones a voice from a clip
and its transcript: a pair of transformers of 601M parameters writes ten codes per frame of 2048
samples (21.5 frames per second), and the checkpoint's own codec renders them and reads the
reference clips. oxinfer reads the repository's files as they are and runs everything on the CPU:

| File | What | Size |
|---|---|---|
| `model.safetensors` | the language model, bf16 | 1.2 GB |
| `codec.pth` | the codec (encoder, quantizers, decoder), a float32 PyTorch file | 1.35 GB |
| `tokenizer.json`, `config.json`, `generation_config.json` | the tokenizer (155776 ids), the shapes, the sampling defaults | 12 MB |

```sh
./target/release/oxinfer --model Edge0/Audio8-TTS-Realistic-Air --speak "Hello there, how are you today?" -o hello.wav
# a voice cloned from a clip and its exact transcript (--ref-text, or me.txt beside the clip)
./target/release/oxinfer --model Edge0/Audio8-TTS-Realistic-Air --speak "Something new." \
    --ref-audio me.wav --ref-text "What the clip says, word for word." -o clone.wav
# one utterance per line -> out-1.wav, out-2.wav, ...; or the sentences of a long text joined in one file
./target/release/oxinfer --model Edge0/Audio8-TTS-Realistic-Air --speak-file lines.txt --ref-audio me.wav -o out.wav
./target/release/oxinfer --model Edge0/Audio8-TTS-Realistic-Air --speak - --speak-split --ref-audio me.wav -o story.wav < story.txt
# greedy, the audio as it is generated, the codes
./target/release/oxinfer --model Edge0/Audio8-TTS-Realistic-Air --speak "Hello." --temp 0 --speak-stream --show-tokens -o hello.wav
```

`--model` is the repo id or a folder holding the files above. Without a reference the model
picks a voice, and another one with another `--seed`; over several utterances
(`--speak-file`, `--speak-split`) only a reference keeps the voice. A reference is a clip (WAV;
MP3, FLAC and the rest through oxideav; resampled to 44.1 kHz mono) or its codes as a `.npy`
array `[10, frames]`, with the transcript of what it says. Clips in a `voices/` folder beside the
weights, `NAME.wav` with `NAME.txt`, are voices by name (`--voice NAME`). The reference, the text
and the generated frames share a context of 2048 rows: 30 s of reference take 646 of them. The
language is not an input: the model reads it from the text.

Sampling follows the checkpoint's `generation_config.json` (temperature 0.7, top-p 0.9, top-k 50;
the model card's example uses 0.8, 0.95 and 50), at most 1024 frames (47.6 s) per utterance.
`--temp 0` is greedy. `--speak-json` takes every field of `oxinfer::arktts::ArkttsSpeak`: `voice`,
`voice_text`, `temperature`, `top_k`, `top_p`, `max_tokens`, `seed`.

In Rust:

```rust
use oxinfer::arktts::{Arktts, ArkttsOptions, ArkttsSpeak};
let m = Arktts::load("Edge0/Audio8-TTS-Realistic-Air", &ArkttsOptions::default())?;
let mut o = ArkttsSpeak::default();
o.voice = Some("me.wav".into()); // a clip, codes as .npy, or a name of the voices folder
o.voice_text = Some("What the clip says, word for word.".into());
let speech = m.speak("Hello there!", &o)?; // Speech { audio: Audio { samples, sample_rate: 44100 }, tokens, stats, .. }
speech.audio.write_wav("hello.wav")?;
// The stages on their own:
let reference = m.reference(&o)?;                                    // the clip's codes and transcript
let prompt = m.prompt("Hello there!", reference.as_ref())?;          // rows of 11 ids
let g = m.generate(&prompt, &o, false, &mut |_| oxinfer::engine::Flow::Continue)?; // [frames, 10] codes
let pcm = m.codec().decode(&g.codes, g.frames)?;                     // frames * 2048 samples
let (codes, frames) = m.codec().encode(&pcm)?;                       // and back to codes
```

The server's `/v1/audio/speech` serves it like the other speech models: `voice` is a name of the
voices folder (OpenAI's names and `none`: no reference); `seed`, `temperature`, `top_p`, `top_k`
and `max_tokens` are extensions; `pcm` and SSE responses carry the audio as it is generated, at
44.1 kHz. Model options: `threads`, `voices=DIR` (the voices folder). A request cannot name a file
on the server, so a server's voices are the clips of that folder.

## The pipeline

**Text.** Trimmed, every run of whitespace one space; nothing else is changed (no expansion of
numbers or abbreviations, in the reference either). The tokenizer is a byte-level BPE with the
chat markers, `<|voice|>`, `<|speaker:N|>` and the 4096 semantic tokens (`<|semantic:N|>`, ids
151678 to 155773) as added tokens.

**Prompt** (`processing_arktts.py`). A chat whose literal pieces are tokenized one by one:

```text
<|im_start|>system\nconvert the provided text to speech<|im_end|>\n
<|im_start|>user\nTEXT<|im_end|>\n<|im_start|>assistant\n<|voice|>
```

and with a reference voice:

```text
<|im_start|>system\nconvert the provided text to speech reference to the following:\n\nText:\n
<|speaker:0|>TRANSCRIPT\n\nSpeech:\n[the clip's frames]<|im_end|>\n
<|im_start|>user\nTEXT<|im_end|>\n<|im_start|>assistant\n<|voice|>
```

Every position is a row of 11 ids: the token and the ten codes of a frame. Text rows carry zeros
for the codes; an audio frame is the semantic token of its first code and all ten codes.

**Language model** (`oxinfer::arktts::lm`, `modeling_arktts.py`). Two transformers of the same
block: RMS norms before attention and before a SwiGLU feed-forward of 4864, one matrix for
queries, keys and values (14 query heads over 2 key / value heads of 64), RoPE on neighbouring
pairs with base 1e6 whose cos / sin tables are rounded to bf16 (the release computes them that
way, and so does oxinfer). The slow transformer (24 layers, width 896) reads the rows: the
token's embedding, plus the sum of the embeddings of the ten codes under a semantic token. Its
output through the tied embedding matrix gives the next frame's semantic token; only the 4096
semantic tokens and the end token `<|im_end|>` can be chosen, so only those 4097 logits are
computed. The fast transformer (4 layers) then writes the frame's other nine codes: its first
position is the slow model's normalized hidden state, the next ones the embeddings of the codes
so far, and each output gives the next code out of 4096 (the residual codebooks hold 1024 codes;
the codec clamps larger values, as the reference does). The frame goes back into the slow
transformer as the next row.

**Sampling.** The candidates are sorted; one is dropped when the probabilities at temperature 1
up to and including it exceed top-p or when it is past the first top-k; the rest is sampled at
the temperature. A semantic token already among those of the last 10 frames is drawn again at
temperature 1.0 and top-p 0.9 (`ras_*` in `config.json`). Speech ends when the slow transformer
picks the end token.

**Codec** (`oxinfer::arktts::codec`, `modeling_arktts_codec.py`; the design is Fish Audio's DAC
variant). Decoding: the codes' vectors (8 wide) through their output projections, summed; eight
causal transformer layers at the frame rate (attention over the last 128 frames, LayerScale);
two stages that double the rate (a transposed convolution and a ConvNeXt block); then the DAC
decoder, a convolution to 1536 channels and four blocks (Snake, a transposed convolution by 8, 8,
4 and 2, three residual units with dilations 1, 3 and 9) down to 96 channels at 44.1 kHz, Snake,
a convolution to one channel and tanh. Encoding a reference clip is the mirror image: the DAC
encoder (strides 2, 4, 8 and 8, four transformer layers after the last), two stages that halve
the rate, eight transformer layers, then the semantic quantizer and nine residual quantizers,
each picking the nearest of its codebook's entries by cosine similarity. Every convolution is
causal, so the audio of the first frames does not change when more frames follow: the decoder's
convolutions run over windows of 64 frames with 8 frames of left context, and streamed audio
(`--speak-stream`, the server's `pcm`) is decoded 32 frames at a time (1.5 s) while the language
model goes on.

## Agreement with the reference

The reference is the Python code in the checkpoint, run by transformers 4.57.3 and PyTorch 2.15
in float32 on the CPU on the released weights. `tools/ref_arktts.py` writes its outputs and
`examples/arktts_check.rs` compares oxinfer's with them:

```sh
M=~/.cache/oxinfer/hub/Edge0--Audio8-TTS-Realistic-Air
python tools/ref_arktts.py $M ref --text "The quick brown fox ..." [--ref-audio clip.wav --ref-text "..."]
cargo run --release --example arktts_check -- $M ref
python tools/ref_arktts.py $M ids --ids texts.txt && cargo run --release --example arktts_check -- $M ids --ids
```

**Tokenizer and prompt.** 20 texts (English with digits and symbols, French, German, Spanish,
Russian, Japanese, Chinese, Korean, Arabic, Hindi, Thai, Vietnamese, Turkish, Polish, Finnish, a
`<|speaker:1|>` tag, emoji, typographic quotes, URLs), each as the text to speak and as a
reference transcript: the ids are the processor's. The prompt rows are the reference's
(`_prepare_prompt`) in the five runs below, and the RoPE table is the reference's value for value.

**Language model.** Greedy generation (`do_sample=False`: both transformers take the most likely
code), from the prompt to the end token:

| Run | Prompt rows | Generated frames | Identical | Largest logit difference (slow / fast) |
|---|---|---|---|---|
| English, 20 words, no reference | 43 | 144 | 144 | 1.7e-4 / 4.3e-4 |
| a cloned voice (6.7 s of reference), 16 words | 223 | 152 | 152 | 1.7e-4 / 4.1e-4 |
| Chinese, no reference | 34 | 144 | 144 | 1.8e-4 / 4.0e-4 |
| a cloned voice, 50 words | 262 | 398 | 398 | 2.4e-4 / 3.9e-4 |
| a cloned voice (26 s of reference), stopped after 6 frames | 685 | 6 | 6 | 1.0e-4 / 2.2e-4 |

The first four end at the same step as the reference. The logits compared are the 4097 the slow
transformer can choose from and the 9 x 4096 of the fast one, at every step. Sampling uses
oxinfer's own random numbers, so a seed does not give the reference's take.

**Codec decoder.** The waveform of each run's frames against `decode_audio`: the largest sample
difference is 9.5e-7 to 4.8e-6 and the signal-to-noise ratio 119.0 to 120.1 dB (0.3 to 18.5 s of
audio). Decoding in windows, and 32 frames at a time as streaming does, gives the samples of
decoding the whole utterance at once (difference 0).

**Codec encoder.** Two clips at 44.1 kHz (6.7 s and 26.0 s; the second is long enough for the
attention windows of 512 and 128 steps to cut): all 1440 and all 5600 codes are the reference's
(`encode_audio`). Both sides read the same samples; a clip at another rate goes through
oxinfer's resampler instead of torchaudio's, which was not compared.

**End to end.** 20 Harvard sentences (159 words), default sampling, seed 42, each an utterance of
its own, transcribed by oxinfer's whisper-small and scored in lower case without punctuation:

| Voice | WER | Errors |
|---|---|---|
| none (the model picks one per utterance) | 2.5% | "park truck", "hugs" for "hogs", "smokey", "50" |
| cloned from a 6.7 s clip and its transcript | 0.6% | "50" |
| none, 5 French sentences (51 words) | 5.9% | "18h" for "dix-huit heures" (3 words) |

A 26 s paragraph (69 words) in the cloned voice, one utterance of 560 frames, is transcribed
without an error. The speaker similarity of cloned voices was not measured.

## Speed

On a 64-thread x86 machine shared with other jobs (a load average of 50 to 75 while measuring,
so these are low figures): the language model writes 20 to 35 frames per second (21.5 are real
time; a prompt of 30 rows is read in 0.1 s, one of 215 rows with a reference in 0.3 s), the codec
decodes a second of audio in 0.22 to 0.24 s and encodes a 6.7 s clip in 1.5 s and a 26 s clip in
5.8 s. The 20 sentences above (58 s of audio, 2 to 4 s each): real-time factor 0.93 without a
reference and 1.16 with one (0.85 to 1.76 per utterance). With `--speak-stream` the first 1.5 s
of audio are there after 1.3 s. The process peaks at 2.4 GB of memory without a reference and at
3.5 GB with one (the codec's encoder is loaded when a clip is first encoded). The Python
reference on the same machine (PyTorch on 16 threads, float32) generates 1.1 to 2.7 frames per
second and decodes 6.7 s of audio in 4.9 to 10.4 s.

## Not done

- **GPU**: everything is on the CPU.
- **Batching**: several utterances run one after the other (the reference's `generate` takes a
  batch).
- **The reference's random numbers**: sampling uses oxinfer's own generator, so a seed does not
  give the reference's take; the distributions are the same.
- **Long texts** are not cut automatically: `--speak-split` speaks sentence by sentence, each an
  utterance of its own (with a reference voice, the same reference each time).
- **bf16 arithmetic**: the model card runs the model in bf16 on a GPU; oxinfer computes in
  float32 from the bf16 weights, which is what the comparison above is made with.
- `Edge0/Audio8-TTS-Preview-0.6b`, the base model, has the same `config.json` shapes and the same
  code and should load; it was not run.
- Custom logits processors and stopping criteria of the reference's `generate`, and the SGLang
  Omni serving adapter.
