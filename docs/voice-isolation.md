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

On Metal a stream's steps run whole on the GPU (`src/metal_vi.rs`, `src/metal_vi.metal`, through
`gpu::vi`): the realtime checkpoint's weights go up once (f32), each stream keeps both lanes'
states on the device (the time attention's keys and values, the time Mambas' conv tails and scan
states, double-buffered so a failed step leaves them as they were), and a step runs the band
encoders, every block of each lane (one lane, two during a crossfade) and the decoder heads in
one command buffer, all in f32 with the CPU's exponential and erf: projections on SIMD-group
matrix GEMMs (16 outputs a threadgroup, K split over the four SIMD-groups), the LayerNorms,
rotary positions, the attention over the cache (one threadgroup a query), the conv and the
selective scan in kernels of their own. Only the spectra go up and the decoder outputs come back;
the STFT, the overlap-add and the cache bookkeeping (growth, compaction) stay on the CPU. Should
a step fail, the stream carries on on the CPU from the states the GPU held. `--device cpu` (load
option `device=cpu`) keeps it on the CPU, as does `OXINFER_VI_GPU=0`.

## Accuracy

Against the reference package (torch 2.15 CPU, float32) on the repository's
`demo/examples/sample_14.wav` (10.8 s, 48 kHz):

| | max abs difference | SNR of the difference |
|---|---|---|
| offline (`voice-isolator-v1`, two 8 s windows) | 4.2e-7 | 126.7 dB |
| streaming, default windows, look-ahead 2, 20 ms pushes, CPU | 3.0e-7 | 131.6 dB |
| the same, Metal | 2.4e-7 | 131.7 dB |
| plain stream (`window_frames 0`), 20 ms pushes, Metal | 2.5e-7 | 131.3 dB |
| whole clip (`isolate`, 64-frame blocks), Metal | 2.4e-7 | 131.9 dB |

`tests` in `src/voice_isolator.rs` check the band layout against the reference's, the FFT, the
offline windows and that the stream's output does not depend on the pushes or the block size
(plain and windowed) on a tiny random checkpoint; `src/server/tools.rs` tests the endpoint.

## Speed

M5 Max (18 cores), sample_14.wav (10.8 s), under `tools/bench/with-lock.sh` on a machine shared
with other builds:

| | Metal | CPU |
|---|---|---|
| offline (per-op GPU matmuls and attention, scan on the CPU) | 4.5 s (2.4x real time) | 7.6 s (1.4x) |
| realtime checkpoint, whole clip (`isolate`, 64-frame blocks; whole steps on the GPU) | 1.12 s (9.7x) | 7.3 s (1.5x) |
| realtime checkpoint, 20 ms pushes, default windows | 2.63 s (4.1x; slowest push 34 ms) | 12.5 s (0.9x; 81 ms) |
| realtime checkpoint, 20 ms pushes, plain stream | 2.63 s (4.1x; slowest push 48 ms) | 9.9 s (1.1x; 72 ms) |

A 20 ms push computes two frames (92 tokens) through every layer. On the GPU such a step takes
4.5 ms (median of the command buffers' GPU time) with one lane and 8.7 ms (90th percentile) during
the crossfades, where both lanes run, against 20 ms of audio; the matmuls are most of it (the
attention about 0.8 ms, the other kernels 0.6 ms). The slowest pushes are the plain stream's cache
compactions (every 600 frames: 1200 slots of keys moved and rotated back on the CPU) and moments
where the shared GPU was busy. On the CPU a live stream runs at about real time (the matmuls of 92
rows dominate). The reference package takes 297 s offline and 555 s streaming on the same machine
(its selective scan runs frame by frame in Python on the CPU).
