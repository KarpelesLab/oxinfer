# Text to speech (Fish Audio Drama 3)

[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) (`model_type: fish_omni`; Fish
Audio Community License: research and non-commercial use, commercial use under a revenue
threshold) speaks at 44.1 kHz with up to 16 speakers per text and clones a voice from a clip and
its transcript. A language model of 35B parameters (3B active per step) writes one audio frame
per step, 21.5 frames per second: its own output gives the frame's first code, a codebook tower
the other nine, and a codec turns the frames into audio. oxinfer runs all of it on the CPU, from
the files of the repository:

| File | What | Size |
|---|---|---|
| `model-000NN-of-00015.safetensors` | the backbone (40 layers, the Qwen3.5 mixture-of-experts hybrid: 32.2B expert weights as FP8 E4M3 with a scale per block of 128 x 128, the rest bf16) and the codebook tower (13 layers, bf16) | 39.6 GB |
| `codec.pth` | the codec (Fish Audio's causal DAC variant), a PyTorch state dict | 1.9 GB |
| `tokenizer.json`, `config.json` | 252288 ids, 4096 of them the semantic band `<\|semantic:N\|>`, 16 speaker tags | 21 MB |

```sh
./target/release/oxinfer --model fishaudio/drama-3 --speak "Hello there, how are you today?" -o hello.wav
# a voice cloned from a clip and what is said in it
./target/release/oxinfer --model fishaudio/drama-3 --speak "Something new, in that voice." \
    --ref-audio me.wav --ref-text "What I say in the clip." -o clone.wav
# several speakers; cues in brackets
./target/release/oxinfer --model fishaudio/drama-3 -o scene.wav \
    --speak "<|speaker:0|>[sighs] Fine. If we miss it, it's on you.<|speaker:1|>[laughs] Well, maybe that's fair."
# one utterance per line -> out-1.wav, out-2.wav, ...; a long text from standard input
./target/release/oxinfer --model fishaudio/drama-3 --speak-file lines.txt -o out.wav
./target/release/oxinfer --model fishaudio/drama-3 --speak - -o story.wav < story.txt
# the bf16 matrices quantized at load: less memory read per frame
./target/release/oxinfer --model fishaudio/drama-3 --quant q8 --speak "Faster." -o fast.wav
```

`--model` is the repo id (the checkpoint and `codec.pth` are fetched, 41.5 GB) or a folder
holding them. The weights are read into memory (`--mmap` maps them instead): the process needs
about 40 GB. There are no named voices: without a reference the model picks a voice, another one
for every utterance and seed (`--seed`). `--voice N` is the speaker tag (0 to 15) put before text
that carries none; text can carry its own `<|speaker:N|>` tags, and the transcript of a reference
can start with one to say which speaker the clip is (0 otherwise). A reference is an audio file
(any format oxideav decodes, resampled to 44.1 kHz) or its codes as a `.npy` file (`[10,
frames]`, as fish-speech saves them, or `[frames, 10]`); `--ref-audio` and `--ref-text` come in
pairs and can be repeated.

Sampling follows fish-speech's defaults: temperature 0.7, top-p 0.7, top-k 30 on the frame's
first code and on each of the other nine, and a first code that repeats one of the last 16 rows'
is drawn again with temperature 1.0, top-k 30, top-p 0.95 (the model card's benchmark uses
temperature 1.0, top-p 0.9). `--temp 0` is plain greedy decoding, which this model is not made
for: it falls into one repeated frame within a few steps. A text longer than about 300 bytes is
cut into chunks at sentence ends (at most three speakers per chunk); each chunk is generated
with the first chunk and the one before it as context, 600 frames (28 s) at most each
(`--max-tokens`). `--speak-json` takes every field of `oxinfer::fish::DramaSpeak`: `temperature`,
`top_k`, `top_p`, `max_tokens`, `seed`, `speaker`, `references` (`[{"audio": FILE, "text":
TRANSCRIPT}]`), `single_chunk`, `redraw_repeats`.

In Rust:

```rust
use oxinfer::fish::{Drama, DramaOptions, DramaSpeak, Reference};
let d = Drama::load("fishaudio/drama-3", &DramaOptions::default())?;
let mut o = DramaSpeak::default();
o.references.push(Reference { audio: "me.wav".into(), text: "What I say in the clip.".into() });
let speech = d.speak("Hello there!", &o)?; // Speech { audio: Audio { samples, sample_rate: 44100 }, tokens, stats, .. }
speech.audio.write_wav("hello.wav")?;
// The stages on their own:
let codes = d.reference_codes("me.wav".as_ref())?;                           // [frames, 10] codes of a clip
let prompt = d.prompt("Hello there!", 0, &[("What I say in the clip.".into(), codes)], &[])?; // rows of 11 ids
let g = d.generate(&prompt, &o, 600, false, &mut |_| oxinfer::engine::Flow::Continue)?;       // rows of [token, 10 codes]
let frames: Vec<u32> = g.rows.chunks(11).flat_map(|r| r[1..].to_vec()).collect();
let pcm = d.codec().decode(&frames)?;                                        // frames * 2048 samples
```

The server's `/v1/audio/speech` serves it like Orpheus (`voice` a speaker tag `0` to `15`,
anything else speaker 0; `seed`, `temperature`, `top_p`, `top_k`, `max_tokens` as extensions;
model options `threads`, `mmap`, `quant`). The audio of a chunk of text is sent when the chunk is
generated. A request cannot name a file on the server, so cloning from a clip is for the command
line and the library. The model itself asks for no device; in a server whose other models run on
a GPU, that backend also takes this model's larger matrix products, which was not tried.

## The pipeline

The reference is Fish Audio's code, `fish_speech/models/text2semantic/fish_omni/` and
`fish_speech/models/dac/` of [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech).

**Prompt** (`oxinfer::fish::Drama::prompt`; `build_drama3_prompt`). The sequence is a table of
rows of 11 ids, `[token, code 0 .. code 9]`: a text row holds a token and zeros, an audio row
`[semantic_start + code 0, code 0 .. code 9]`. A system turn (`convert the provided text to
speech`, or with references `convert the provided text to speech reference to the following:`
and per reference `[Reference i]`, `Text: <|speaker:k|>transcript`, `Speech:` and the clip's
frames), earlier chunks as finished user / assistant turns, then `<|im_start|>user`, the text
behind a speaker tag, `<|im_start|>assistant` and `<|voice|>`. Every literal is tokenized on its
own. Text is taken as written (no normalization), except that the tokens the prompt builder
writes (`<|im_start|>`, `<|voice|>`, `<|semantic:...`) and speaker tags other than 0 to 15 are
refused, as the reference does.

**Backbone** (`oxinfer::qwen`; `backbone.py`). The Qwen3.5 / 3.6 35B-A3B shape, which oxinfer
runs as a chat model: 40 layers of width 2048, three of four Gated DeltaNet (16 key heads and 32
value heads of 128, convolution kernel 4), the fourth attention (16 heads of 256 over 2 key /
value heads, normalized queries and keys, a quarter of each head rotated, a sigmoid output
gate); every layer has 256 experts of width 512, 8 per token, and a gated shared expert. The
experts are computed from the FP8 bytes and their block scales as stored. On an audio row the
input is `(embedding of the main token + embedding of code 1) / sqrt(2)`.

**Frame** (`oxinfer::fish::tower`; `tower.py`). The backbone's logits over the end token and the
4096 semantic tokens give the row's main token, that is code 0. The tower then runs on the
backbone's hidden state: 4 shared layers and 9 residual ones, each mixing a code's embedding into
the state (a linear layer over the two RMS-normalized halves), one transformer block of width
2048 (16 heads of 256 over 2 key / value heads, a quarter of each head rotated in adjacent
pairs, an output gate, a SiLU-gated MLP of 6144) and a norm. Each layer attends over every
earlier row of the sequence through its own key / value cache, so the tower also runs over the
prompt, text rows included. The shared layers take code 0, the first of them also the codes 2 to
9 of the row just read (the backbone only saw its code 1); residual layer `i` takes code `i` and
its head (4096 logits) gives code `i + 1`.

**Codec** (`oxinfer::fish::codec`; `modded_dac.py`, `rvq.py`). Frames of 2048 samples. Decoding:
the sum over the ten codebooks of each code's entry (8 values) through its output projection; 8
causal transformer layers of width 1024 over the last 128 frames; two transposed convolutions
with ConvNeXt blocks (to 86 Hz); the DAC decoder (four blocks of a Snake activation, a
transposed convolution of stride 8, 8, 4, 2 and three residual units of dilated k7
convolutions, from 1536 channels down to 96). Encoding, for reference clips: the DAC encoder
(strides 2, 4, 8, 8, four transformer layers at the end), two strided convolutions with ConvNeXt
blocks, 8 transformer layers, then per codebook the nearest entry by cosine of the latent
projected to 8 values (the semantic codebook of 4096 entries first, nine residual ones of 1024
on what is left). Every convolution is causal. A whole chunk is decoded at once.

## Agreement with the reference

The reference is fish-speech's PyTorch code in float32 on the CPU (its loader wants CUDA;
`tools/ref_fish_drama.py` builds the same modules on the CPU and calls its `build_drama3_prompt`,
`generate_drama3` and the codec's `encode` / `from_indices`). `examples/fish_check.rs` dumps
oxinfer's side and the tool compares; `tools/fish_drama_tiny.py` writes a small random checkpoint
(4 layers of width 128) with the release's tensor names, tokenizer and codec:

```sh
F=fish-speech   # a checkout of github.com/fishaudio/fish-speech
M=drama-3       # the files of the Hugging Face repository
python tools/fish_drama_tiny.py $F $M tiny            # and with --fp8: FP8 experts with block scales
cargo run --release --example fish_check -- prompt tiny cases.json rows.json
python tools/ref_fish_drama.py $F prompt tiny cases.json rows.json
cargo run --release --example fish_check -- gen tiny out "Hello there, how are you today?" --steps 200
python tools/ref_fish_drama.py $F gen tiny out
cargo run --release --example fish_check -- gen $M out "Hello there, how are you today?" --temp 0.7 --steps 300
python tools/ref_fish_drama.py $F gen $M out
cargo run --release --example fish_check -- encode $M/codec.pth clip.wav codes.npy
python tools/ref_fish_drama.py $F encode $M/codec.pth clip.wav codes.npy
cargo run --release --example fish_check -- decode $M/codec.pth codes.npy out.f32
python tools/ref_fish_drama.py $F decode $M/codec.pth codes.npy out.f32
```

**Tokenizer and prompt.** 20 cases (English, Chinese, Japanese, French, Cyrillic, Arabic and
Devanagari text, emoji and symbols, leading white space, speaker tags, bracketed cues, one and
two references with tagged and untagged transcripts, one and two history turns): the 1017 rows
are those of `build_drama3_prompt` with the `tokenizers` library.

**Backbone and tower.** Greedy generation (every draw of the reference's sampler replaced by the
argmax) on the small random checkpoints, whose weights both sides read as the same float32
values, and on the release, whose experts both sides keep in FP8 and multiply out in float32:

| Checkpoint | Prompt | Frames | Identical | Largest logit difference | Mean KL |
|---|---|---|---|---|---|
| small, bf16 experts | 29 rows | 200 | 200 | 6.5e-5 | 5.6e-12 |
| small, FP8 experts | 29 rows | 200 | 200 | 7.0e-5 | 5.3e-12 |
| small, FP8 experts | 155 rows, a reference clip of 85 frames | 150 | 150 | 2.6e-5 | 2.0e-12 |
| the release | 29 rows | 120 | 120 | 1.5e-4 | 2.4e-11 |

The logits are those of the main token (the end token and the 4096 semantic tokens) and of the
nine codebook heads at every step; the KL divergence is taken per distribution. The backbone's
logits over the whole vocabulary (252288 ids) at the last prompt row are within 1.2e-5 (small)
and 2.9e-5 (release) of the reference's, KL below 1e-11.

Greedy decoding of the release falls into one repeated frame within five steps, so that run
says little about speech. For the frames the sampler actually visits, oxinfer samples an
utterance and the reference is then led along the same frames (every draw returns oxinfer's
choice), which compares both models' logits at every step of a real utterance:

| Run on the release | Prompt | Frames | Largest logit difference | Mean KL | Same argmax |
|---|---|---|---|---|---|
| two sentences, temperature 0.7 | 42 rows | 166 (7.7 s) and the end | 1.5e-4 | 3.6e-11 | 1670 of 1670 |
| one sentence in a cloned voice (a 3.9 s clip) | 156 rows | 108 (5.0 s) and the end | 9.3e-5 | 2.8e-11 | 1090 of 1090 |

"Same argmax": the main token and the nine codes each side would pick greedily at that step.
On this machine's CPU the reference takes 1.4 to 1.9 s per frame (PyTorch, float32), oxinfer
0.14 to 0.17 s. Sampling uses oxinfer's own random numbers, so a seed does not give the
reference's take.

The reference's own CUDA paths differ from what is compared here. On GPUs with FP8 tensor cores
it also rounds the experts' inputs to FP8 (per group of 128 values, `activation_scheme:
dynamic`); on other GPUs its loader multiplies the experts out once, rounds them to bf16 (or
float16) and computes in that precision. oxinfer computes in float32 from the stored FP8 values
and their scales, which is what the reference's modules do on the CPU, as in these checks. How
far the two GPU paths are from that was not measured.

**Codec.** The model card's reference clip (3.9 s, resampled to 44.1 kHz): the encoder gives the
85 frames (850 codes) of the reference's `DAC.encode`, also from the 24 kHz original through
oxinfer's resampler against `torchaudio.transforms.Resample`. Those frames through the decoder:
the waveform is the reference's `from_indices` to 1.9e-6, an SNR of 113.6 dB.

**End to end.** 20 Harvard sentences (159 words), default sampling, seed 42, on the release,
transcribed by oxinfer's whisper-small and scored in lower case without punctuation:

| Run | WER | Errors |
|---|---|---|
| as stored (FP8 experts, bf16) | 1.3% | "smokey", "50" |
| `--quant q8` | 1.3% | the same two |
| `--quant q4` | 2.5% | the same two, "canoes", "park truck" |
| voice cloned from the 3.9 s clip | 1.3% | the same two |
| a paragraph of 655 bytes: three chunks, 47.7 s (123 words) | 4.9% | "lighthousekeeper" for two words, "40", "beams", "Thank you." after the end |

Every utterance ended with the end token (none ran into the frame limit). The cloned voice is
the clip's: the speaker embedding (ZONOS3's speaker encoder) of the 20 cloned sentences has a
cosine of 0.59 to 0.77 with the clip's (mean 0.67), against -0.14 to 0.09 with another
speaker's clip (the two clips: 0.01); the sentences generated without a reference score -0.02
to 0.34 against the clip. Without a reference the voice changes from one utterance to the next
(cosine between utterances 0.22 to 0.68, against 0.60 to 0.82 for the cloned ones).

Tried once each, with the output transcribed back correctly: the server's endpoint (a WAV
answer and streamed PCM), loading by repo id, `--mmap`, a two-speaker line with `[sighs]` and
`[laughs]`, a Chinese and a French sentence.

## Speed and memory

On a 64-core x86 machine (AVX-512) shared with other jobs, a load of about 30 throughout:

| Matrices stored as bf16 | Frames per second | Real-time factor | Peak memory |
|---|---|---|---|
| as stored | 6.0 | 4.6 | 39.2 GB |
| `--quant q8` | 8.6 to 9.1 | 3.2 to 3.5 | 42.3 GB |
| `--quant q4` | 10.4 to 10.8 | 3.0 | 40.7 GB |

Real time needs 21.5 frames per second. The real-time factor is that of whole utterances of 3 s
(20 Harvard sentences): reading the prompt (0.7 s for 30 rows; 2.5 to 3.8 s for 150 rows with a
reference clip, which every utterance reads again), the frames, and the codec (0.6 to 0.8 s per
second of audio; the encoder takes 2.6 s for a 3.9 s clip). The checkpoint is read into memory
in 10 to 19 s from the page cache. A frame is 464 matrix products over 40 backbone and 13 tower
layers, which read about 6 GB of weights as stored and 1.9 GB with `--quant q4` (1 GB of that
the FP8 experts); on a busy machine their number costs as much as their size. With `--mmap` the
pages are read as the text needs them: one sentence peaked at 25.8 GB, its prompt taking 11.6 s.

## Not done

- **A GPU.** The command line runs the model on the CPU and refuses `--device cuda`; the
  backbone was not tried on a device (its 39.6 GB do not fit the 16 GB card of the test machine).
- **The arithmetic of the reference's GPU paths** (FP8 activations on GPUs with FP8 tensor cores,
  bf16 elsewhere; see above): oxinfer computes in float32 from the stored weights. The reference
  was not run on a GPU here, so how its output differs from the float32 one was not measured.
- **The quantized precisions against the reference**: `--quant q8` and `q4` were only scored by
  word error rate.
- **Fish Audio's "reference sampling"** (`--reference-sampling` of fish-speech: an epsilon
  cutoff on the main token, an argmax fallback on repeats, the residual codes limited to the 1024
  entries the codec holds). The default sampler is implemented; it can in principle draw a
  residual code above 1023, which the codec clamps, as the reference's does.
- **Streaming inside a chunk**: the codec is causal, but a chunk of text (up to 28 s of audio) is
  decoded when all of it is generated. Several utterances are generated one after the other, and
  a reference clip is read again for each.
- **Editing by continuation** (the model card's `previous_takes`): `Drama::prompt` takes earlier
  takes as history, as the chunks of a long text use it, but the command line and the server have
  no option for it and it was not tried.
- **Voice cloning through the server**: a request cannot name a file on it.
- **Several speakers, cues and other languages** were only tried once each (see above); the word
  error rate is English only, and whether a cue is acted out was not judged. Two references for
  two speakers were not tried.
