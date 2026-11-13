# Library use

## As a library

`oxinfer::engine` is the CLI's text generation as an API: load a directory, GGUF file or repo
id with `EngineOptions` (every [command-line flag](cli.md) that applies to text, also settable as `key=value`
pairs), then hold a conversation with streamed output, cancellation from another thread
(`CancelHandle`), tool calls, attachments (`Engine::send_with_media` with images, videos and
audio clips in an `Attachments`) and per-turn statistics:

```rust
use oxinfer::engine::{Engine, EngineOptions, Flow};
let mut opts = EngineOptions::default();
opts.set("quant", "q8")?;
opts.set("mtp", "1")?;
let mut engine = Engine::load("Qwen/Qwen3.5-0.8B", &opts)?;
let reply = engine.send("Why is the sky blue?", &mut |d| {
    print!("{}", d.text); // d.thinking marks thinking text
    Flow::Continue        // Flow::Stop ends the reply
})?;
println!("\n{:.1} tok/s, {:.0}% drafts accepted", reply.stats.decode_tok_s(), 100.0 * reply.stats.acceptance());
```

`examples/chat.rs` is a complete chat. Wrap `Engine::load` in `oxinfer::hf::with_progress` for
download progress and cancellation. [Core ML](coreml.md) packages load through the same
`Engine::load` on macOS / iOS, with the options `coreml_units` (`ane`, `all`, `gpu`, `cpu`) and
`coreml_prefill` (`auto`, `batch`, `pad`, `single`); `engine.info().coreml` describes them. The lower-level pieces the CLI is built from are public too:
`oxinfer::generate` (decoding loops with per-token callbacks, MTP and drafter speculative
decoding, streaming detokenization) and the model, session and sampler types. A C API over the
engine (`capi` feature, `include/oxinfer.h`) and a SwiftUI test app for iOS are in `ios/`
(see `ios/README.md`). Neuron edits, steering vectors, directional ablation and SAE features are
`Engine::set_interventions` (see [Runtime interventions](interventions.md#runtime-interventions-neuron-edits-steering-abliteration-sae-features)).
`Engine::save_session` / `Engine::load_session` write and restore a conversation's caches, and the
`prompt_cache` option keeps them in a directory for `Engine::chat` (see
[Session snapshots and the prompt cache](sessions.md#session-snapshots-and-the-prompt-cache)); prompt-lookup
drafting is the `lookup` options ([Prompt lookup](speculative-decoding.md#prompt-lookup)).

Tokenization is in the crate too: `oxinfer::tokenizer::Tokenizer` reads HuggingFace
`tokenizer.json` files (BPE, Unigram, WordPiece and WordLevel models with every normalizer,
pre-tokenizer, post-processor and decoder of the format, BPE dropout, script splitting,
truncation with stride and overflowing parts, padding, added-token flags) and has the API of
the `tokenizers` crate, which oxinfer no longer depends on: `encode` of single texts, pairs
(with type ids) and pre-split words, `encode_batch`, `decode`, `decode_batch`, `decode_stream`,
`token_to_id`, `get_vocab`. Regex splits run on an in-crate backtracking engine with
`fancy-regex`'s syntax and semantics (lookbehind, atomic groups, backreferences, conditionals,
`\K`, `\G`). It was compared with `tokenizers` 0.23.2 on 41 `tokenizer.json`
files (Qwen 2.5 to 3.5, Llama 3, Mistral Small 3.2, Mistral Large 4's tekken, Gemma 3 and 4,
DeepSeek V4, Janus, Whisper, CLIP, T5, the WordPiece encoders of BGE, MiniLM, MPNet, nomic and
mxbai, the XLM-R Unigram of BGE-M3 and multilingual-E5, ModernBERT): ids, special-token masks
and decoding with and without special tokens agree on 30,037 texts each (a multilingual corpus
and random strings of added tokens, combining marks, emoji, control characters and whitespace
runs). Loading is faster (Gemma 4's 32 MB file: 280 ms instead of 430 ms), and so is encoding
(Qwen 2.5 on that corpus: 225 ms instead of 336 ms). Unusual files were covered by rewriting ten
of them (BERT, MPNet, ModernBERT, Qwen, Gemma, Mistral, T5, BGE-M3, CLIP, Whisper) about 200
ways each: every pre-tokenizer, two dozen split regexes with every behavior, normalizer chains,
model options, post-processors with each truncation strategy and padding, decoders and
added-token flags. Single texts, pairs, pre-split words, batches, decoding and streaming
decoding agree on 40 texts per rewrite, except where `tokenizers` itself is undefined: several
added tokens that normalize to the same string (it then picks one by hash-map order).

## Using oxinfer from oxideav (and other crates)

Images, audio and video in the API are [oxideav](https://crates.io/crates/oxideav-core) types:
decoded `VideoFrame`s and `AudioFrame`s with the `CodecParameters` that describe them, as an
oxideav pipeline hands them from stage to stage. Two features split what a dependent pulls in:

| Feature | Brings | Gives |
|---|---|---|
| `oxideav` | `oxideav-core` only | `oxinfer::av` and the in-memory inputs and outputs below; `oxinfer::oxideav_core` re-exports the crate |
| `media` (default, implies `oxideav`) | oxideav-pipeline, oxideav-image and the codec and container crates (PNG, JPEG, WebP, GIF, BMP, MP3, FLAC, AAC, Vorbis, Opus, WAV, MP4, MOV, Matroska / WebM, Ogg, AVI, H.264, H.265, VP8, VP9, AV1, MPEG-4 Part 2) | the path-based inputs (`--image`, `--video`, `--audio`, `transcribe_file`, `Media`) |

`hf` (HuggingFace fetching), `gpu` (run-time GPU backends) and `server` (default: the
[OpenAI-compatible server](server.md#openai-compatible-server), with httpsd) are independent of both. A crate
that already has decoded frames, such as an oxideav filter, needs only the types:

```toml
[dependencies]
oxinfer = { version = "0.0.4", default-features = false, features = ["oxideav", "hf", "gpu"] }
oxideav-core = "0.1.41" # the same 0.1 release oxinfer uses, so the types are the same
```

Chat and multimodal prompts take frames next to file paths (each kind after the files of the
same kind):

```rust
use oxinfer::engine::{Attachments, Engine, EngineOptions, Flow};
let mut engine = Engine::load("Qwen/Qwen3.5-0.8B", &EngineOptions::default())?;
// A picture: one VideoFrame and its parameters (width, height, pixel_format).
let media = Attachments::default().with_image_frame(frame, params);
// A clip: frames in presentation order, their parameters and the stream time base; frames are
// sampled as the model's video processor does, and the prompt's timestamps come from the pts.
let media = media.with_video_frames(frames, video_params, Some(time_base));
// Audio (Gemma 4): frames and their parameters (sample_rate, channels, sample_format).
let media = media.with_audio_frames(audio_frames, audio_params);
let reply = engine.send_with_media("What happens here?", &media, &mut |_| Flow::Continue)?;
```

The other consumers and producers of media:

```rust
// Speech to text from decoded audio (any sample format, rate and channel count).
let t = transcriber.transcribe_frames(&audio_frames, &audio_params, &TranscribeOptions::default())?;
// Typed decisions over a picture (placed before the state text).
let req = DecisionRequest::noul("Backyard camera.", "Is there an animal?").with_image_frames([Picture::new(frame, params)]);
// Stable Diffusion output as an Rgb24 frame for an encoder or a filter graph.
let (img, _) = pipeline.generate(&params, &mut |_, _| {})?;
let (frame, frame_params) = img.into_video_frame();
```

Underneath, `av::video_frame_to_rgb` and `av::audio_frames_to_mono` do the conversions, and
`mm::prepare_frames` (with `mm::MmFrames`) turns frames into soft tokens for the lower-level
`generate` API. Pixel formats: packed RGB / BGR in every byte order (with or without alpha or
padding, 8 and 16 bits, 5-6-5 and 5-5-5), gray (8 to 16 bits, with alpha), 1-bit mono, `Pal8`,
planar YUV 4:2:0, 4:2:2, 4:4:4, 4:4:0 and 4:1:1 at 8 to 16 bits (with alpha, `YuvJ*`), NV12 /
NV21, YUYV / UYVY and planar GBR(A); others (CMYK, float) are an error naming the format. YCbCr
follows the colour signal of the frame or the stream (BT.601 limited range when unspecified;
BT.709, BT.2020, FCC, SMPTE 240M, identity, full range), in oxideav-pixfmt's fixed-point
arithmetic: the RGB is byte-identical to `oxideav_image::Image::to_rgb8` for every format both
convert, so frames and files give the same soft tokens, replies and transcripts
(`tests/oxideav_e2e.rs`). A missing `width`, `height` or `pixel_format` is inferred from the
plane layout, as for decoded files; audio of every `SampleFormat`, interleaved or planar, is
averaged to mono and resampled to the model's rate.

**Dependency cycles.** `media` depends on the codec and container crates directly, not on
`oxideav-meta`, so a transform crate that depends on oxinfer can itself be registered by
`oxideav-meta` without a cycle, whatever features it enables. (Cargo refuses any cycle among the
activated dependencies and accepts one through an optional dependency that is not enabled. With
a `media` that depended on `oxideav-meta`, oxinfer -> oxideav-meta -> transform -> oxinfer would
fail in any build that registers the transform in oxideav-meta and enables `media` anywhere, even
in another crate, since features are unified across the whole graph.) The rule for the transform crate and anything oxideav-meta pulls in:

- depend on `oxinfer` with `default-features = false` and the features you use (`oxideav`, and
  `hf` / `gpu` as wanted); take `media` only if you need oxinfer to open files itself;
- never make oxinfer depend on a crate that depends on the transform (oxideav-meta, an
  "everything" facade): its `media` feature lists the codec crates one by one for that reason.
