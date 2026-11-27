# Voice isolation (Inworld Voice Isolator)

[`inworld/voice-isolator`](https://huggingface.co/inworld/voice-isolator) (Apache-2.0) keeps the
voice of a 48 kHz recording and removes everything around it (noise, music, other voices, the late
room tail). The repository holds two checkpoints of one architecture, both supported:

| | `denoiser.safetensors` (`voice-isolator-v1`) | `denoiser_realtime.safetensors` (`voice-isolator-v1-realtime`) |
|---|---|---|
| mode | offline: every frame sees its whole 8 s window, windows overlap by 2 s | a causal stream with 0, 20 (default) or 40 ms of trained look-ahead |
| parameters | 64.0M (256 MB, f32) | 58.2M (233 MB, f32) |
| latency | | 2048 + 480 x look-ahead samples (63 ms at 20 ms of look-ahead) |

`src/voice_isolator.rs` is written from the reference package
([inworld-ai/voice-isolator](https://github.com/inworld-ai/voice-isolator): `model.py`,
`streaming.py`, `isolator.py`): a band-split dual-path network over the STFT (n_fft 2048, hop
480), 46 bands of 512-wide tokens, seven blocks alternating time attention (rotary positions,
"exclusive" self attention, then attention across bands) and dual-path Mamba (state 16), per-band
heads giving a bounded complex mask and a bounded residual per bin, the inverse STFT of the change
added to the input. The architecture comes from the checkpoint's `config` metadata.

## Use

```sh
# offline: the voice to talk_clean.wav, the background to talk_clean_background.wav (48 kHz float32)
oxinfer isolate talk.wav
# the realtime checkpoint as a live source would feed it (20 ms pieces)
oxinfer isolate talk.wav talk_live.wav --mode streaming --lookahead 2
```

`oxinfer isolate IN [OUT]` takes WAV or any audio file oxideav decodes, at any rate (resampled to
48 kHz with torchaudio's `kaiser_best` filter, as the reference does) and any channel count
(averaged). Options: `--model` (default `inworld/voice-isolator`, or a folder / `.safetensors`
file), `--mode offline|streaming`, `--lookahead N` (frames of 10 ms: 0, 2 or 4),
`--block-frames N`, `--cache-frames N`, `--window-frames N` (0 turns the attention windows off),
`--crossfade-frames N`, `--piece-ms MS`, `--chunk-seconds S`, `--overlap-seconds S`,
`--passes N`, `--output-sr HZ` (0 keeps the input's), `--background PATH`, `--no-background`,
`--pcm16`, `--bf16`, `--device`, `--threads`, `--profile`.

Library:

```rust
use oxinfer::voice_isolator::{IsolateOptions, IsolatorOptions, Mode, VoiceIsolator};
let iso = VoiceIsolator::load("inworld/voice-isolator", &IsolatorOptions::default())?;
let (pcm, rate) = oxinfer::voice_isolator::load_audio("talk.wav".as_ref())?;
let voice = iso.isolate(&pcm, rate, &IsolateOptions::default())?; // 48 kHz

let mut o = IsolatorOptions::default();
o.mode = Mode::Streaming;
let live = VoiceIsolator::load("inworld/voice-isolator", &o)?;
let mut s = live.stream(&IsolateOptions::default())?;
for piece in mic_pieces {          // 48 kHz mono, any size
    play(&s.push(piece));          // the samples that became final
}
play(&s.flush());
```

C: `oxi_isolator_load` (`mode=streaming`, `precision=bf16`, ...), `oxi_isolate_pcm`,
`oxi_isolate_file`, `oxi_isolator_stream_new` / `_push` / `_flush` / `_latency` / `_free`,
`oxi_isolator_info_json`, `oxi_isolator_cancel` (see `include/oxinfer.h`).

Server: `POST /v1/audio/isolations` with `model` (a model of kind `isolation`: the repo id is
recognized by name, `mode=streaming` in its options picks the realtime file), `file` (multipart,
or base64 / a data URL in a JSON body), `response_format` (`wav` float32, the default; `wav16`;
`pcm` float32; `json` with base64 WAVs of the voice and the background), `output` (`voice` or
`background`), `output_sample_rate` (default 48000, 0 keeps the input's) and the isolation options
(`lookahead_frames`, `window_frames`, `chunk_seconds`, ...).

```sh
oxinfer serve --model vi=inworld/voice-isolator --model live=inworld/voice-isolator,mode=streaming
curl -s localhost:8080/v1/audio/isolations -F model=vi -F file=@talk.wav -o voice.wav
```

## Streaming

The stream follows `streaming.Stream`: a frame is computed as soon as its 2048-sample analysis
window is complete (blocks of up to `block_frames`, a throughput knob: no added latency), the time
attention keeps its keys over the trained 1200-frame window, the time Mamba carries its state, and
samples are released once every frame that touches them is synthesized; the decoder's output for
frame `t` is read at `t + lookahead`. By default the attention never looks back across 6 s windows
(600 frames): a second lane starts 2 s (200 frames) before each boundary with the Mamba states
cloned and an empty attention cache, the decoder outputs blend linearly over the crossfade, and
the new lane carries on, which is what the reference's `VoiceIsolator(mode="streaming")` and its
CLI do. `window_frames: 0` gives the plain stream (the attention over its whole cache), whose
output equals a whole-clip causal pass. Neither depends on how the audio is pushed, and
`isolate` with the realtime checkpoint runs this stream over the clip (in blocks of 64 frames).

## Accuracy

Against the reference package (torch 2.15 CPU, float32) on the repository's
`demo/examples/sample_14.wav` (10.8 s, 48 kHz):

| | max abs difference | SNR of the difference |
|---|---|---|
| offline (`voice-isolator-v1`, two 8 s windows) | 4.2e-7 | 126.7 dB |
| streaming, default windows, look-ahead 2, 20 ms pushes | 3.0e-7 | 131.6 dB |

`tests` in `src/voice_isolator.rs` check the band layout against the reference's, the FFT, the
offline windows and that the stream's output does not depend on the pushes or the block size
(plain and windowed) on a tiny random checkpoint; `src/server/tools.rs` tests the endpoint.

## Speed

M5 Max (18 cores), sample_14.wav (10.8 s), under `tools/bench/with-lock.sh` on a machine shared
with other builds:

| | Metal (per-op GPU matmuls and attention, scan on the CPU) | CPU |
|---|---|---|
| offline | 4.5 s (2.4x real time) | 7.6 s (1.4x) |
| realtime checkpoint, whole clip (`isolate`, 64-frame blocks) | 5.8 s (1.9x) | 6.2 s (1.7x) |
| realtime checkpoint, 20 ms pushes | 11.2 s (1.0x; slowest push 59 ms) | 11.3 s (0.9x) |

The reference package takes 297 s offline and 555 s streaming on the same machine (its selective
scan runs frame by frame in Python on the CPU). A live stream on this machine runs at about real
time on the CPU: each 20 ms push computes two frames (92 tokens) through every layer; the matmuls
(6 ms of each 10 ms frame) dominate. A whole-pass Metal implementation of the stream is the next
step to give it headroom.
