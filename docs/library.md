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
download progress and cancellation. Loading `org/repo:QUANT` fetches the GGUF repository's
top-level `mmproj*.gguf` vision projector beside the model when it has one (BF16, else F16, else
F32), unless the `mmproj` option (`EngineOptions::mmproj`) is `none` or a file;
`oxinfer::engine::resolve_model_with` and `oxinfer::hf::resolve_with` fetch with that choice
without loading, `resolve_model` / `hf::resolve` as `auto`. `oxinfer::hf::set_offline(true)`
(or `HF_HUB_OFFLINE=1`) makes repo ids load from the cache without asking a hub (see
[Offline, and hubs that do not answer](weights.md#offline-and-hubs-that-do-not-answer)). `Engine::accepts_images` tells
whether a loaded model takes images. [Core ML](coreml.md) packages load through the same
`Engine::load` on macOS / iOS, with the options `coreml_units` (`ane`, `all`, `gpu`, `cpu`) and
`coreml_prefill` (`auto`, `batch`, `pad`, `single`); `engine.info().coreml` describes them. The lower-level pieces the CLI is built from are public too:
`oxinfer::generate` (decoding loops with per-token callbacks, MTP and drafter speculative
decoding, streaming detokenization) and the model, session and sampler types. A C API over the
engine (`capi` feature, `include/oxinfer.h`) and a SwiftUI test app for iOS are in `ios/`
(see `ios/README.md`). Neuron edits, steering vectors, directional ablation and SAE features are
`Engine::set_interventions` (see [Runtime interventions](interventions.md#runtime-interventions-neuron-edits-steering-abliteration-sae-features)).
LoRA adapters (PEFT folders and llama.cpp GGUF adapters) are the `lora` option, repeatable
(`opts.set("lora", "adapter.gguf:0.5")`, or `EngineOptions::loras`), applied at run time or merged
with `lora_merge`; `Engine::loras` lists them and `Engine::set_lora_strength` changes one (see
[LoRA adapters](lora.md)).
`Engine::save_session` / `Engine::load_session` write and restore a conversation's caches, and the
`prompt_cache` option keeps them in a directory for `Engine::chat` (see
[Session snapshots and the prompt cache](sessions.md#session-snapshots-and-the-prompt-cache)); prompt-lookup
drafting is the `lookup` options ([Prompt lookup](speculative-decoding.md#prompt-lookup)).
Without a system message or another `ChatConfig::system`, a conversation starts with the default
system prompt (`SystemPrompt::Default`, `oxinfer::tools::default_system_prompt`): the model runs
locally with oxinfer, without internet access, and the local date, weekday and time zone ("The
current local date is 2027-01-15 Friday (UTC+09:00, JST)."). It has no time of day, so it is the
same text all day, and a conversation keeps the text of its first turn (see
[The default system prompt](sessions.md#the-default-system-prompt)); declare
`oxinfer::tools::builtin_specs` (`get_current_datetime`) or write the time in a system prompt of
your own for a model that needs it. `SystemPrompt::None` is no system prompt.

### Token hook

`Engine::set_token_hook` gives the following replies a hook (`oxinfer::token_hook::TokenHook`,
or any `FnMut(&[u32]) -> Vec<u32> + Send`) that sees each token of a reply as it is committed,
with the reply's tokens so far, and may return tokens for the model to read next. Sampling
resumes after them. `TokenHook::on_start` may also give a reply's first tokens.

```rust
let tok = engine.tokenizer().clone();
let note = tok.encode(" (receipt: 4471-KX)", false)?.get_ids().to_vec();
let mut done = false;
engine.set_token_hook(Some(Box::new(move |reply: &[u32]| {
    if !done && tok.decode(reply, true).unwrap_or_default().ends_with('.') {
        done = true;
        return note.clone();
    }
    Vec::new()
})));
```

- **Forced tokens behave like sampled ones.** They go through the model, so the cache and the
  conversation hold them and the tokens after them are conditioned on them. They are streamed
  as deltas, returned in `Reply::tokens` and `Reply::text` (or `thinking`), counted in
  `max_tokens`, and have log probabilities when those are asked for. A forced stop token ends
  the reply.
- **Every decoding loop follows the hook:** plain decoding, prompt lookup, MTP and DSpark rounds,
  on every device. It rides on the sampler's guide, which those loops all sample through.
  Speculative rounds may draft the queued forced tokens, but the hook only ever sees committed
  tokens, never drafts. The Gemma 4 drafter does not sample, so it gives way to plain decoding
  while a hook is set, as it does for constrained replies.
- **Forced tokens take precedence over a constraint** (`response_format`, forced tool calls),
  which does not see them.
- **Each forced token costs one forward pass.** Lookup drafts can cover several of them.
- **The C API's equivalent** is `oxi_engine_set_token_hook` (a callback that writes up to 4096
  token ids per call).
- **The server has no hook:** a hook is code, not an option.

`examples/token_hook.rs` forces a note after the reply's first sentence and checks it:
- the note's tokens come right after the trigger;
- sampling resumes after them;
- the next turn can quote the note.

On Inkling-2-Mini Q8_0 the replies are byte-identical on CPU and CUDA, with prompt lookup on.
It also passes on Qwen3-0.6B (CPU, CUDA, CUDA with lookup).

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
