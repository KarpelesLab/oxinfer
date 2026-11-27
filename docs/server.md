# OpenAI-compatible server

`oxinfer serve` and `oxinfer web` are the same server: the OpenAI API under `/v1`, Anthropic's
`/v1/messages`, and the [web GUI](web.md) with its management API, on one port (default
`127.0.0.1:8080`, or the next free one when no port is given). `oxinfer web` also opens the GUI in
the browser. The models clients can request are the ones given with `--model`, the ones set up
in the GUI, by their id, and any other repo id a request names ([Models on request](#models-on-request)). Each `--model` is `[ID=]SOURCE[,KEY=VALUE...]`: a local folder or
file, a repo id (fetched on first use, or at start with `--download`) or `org/name:QUANT` for a
GGUF repo (with the repository's top-level `mmproj*.gguf` vision projector beside the model,
unless the `mmproj` option is `none` or a file), an optional id (default: the source) and the model's options. HTTP and HTTPS are
served by [httpsd](https://crates.io/crates/httpsd) (its thread-pool runtime, no async runtime).

```sh
./target/release/oxinfer serve --listen 127.0.0.1:8080 \
  --model qwen=Qwen/Qwen3.5-0.8B,quant=q8 \
  --model gemma=mlx-community/gemma-4-e2b-it-4bit,think=1 \
  --model bge=BAAI/bge-small-en-v1.5 \
  --model sd=stabilityai/sd-turbo \
  --model whisper=openai/whisper-small \
  --model tts=unsloth/orpheus-3b-0.1-ft,quant=q8
# [serving qwen (chat), gemma (chat), bge (embedding), sd (image), whisper (transcription), tts (speech) on http://127.0.0.1:8080/v1]

curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen", "messages": [{"role": "user", "content": "Why is the sky blue?"}], "stream": true}'
curl -s localhost:8080/v1/audio/transcriptions -F model=whisper -F file=@speech.mp3
curl -s localhost:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "tts", "input": "Hello there!", "voice": "leah"}' -o hello.wav
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(model="qwen", messages=[
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        {"type": "text", "text": "What is in this picture?"}]}])
print(r.choices[0].message.content)
img = client.images.generate(model="sd", prompt="a lighthouse at dusk", size="512x512",
                             response_format="b64_json", extra_body={"seed": 7})
```

| Endpoint | Models | Notes |
|---|---|---|
| `GET /health` (also `/v1/health`, no key needed) | | `{"status": "ok", "degraded", "warnings"}`: `degraded` is true when the process runs slower than its hardware allows, and `warnings` says why (the GPU's matrix-unit kernels failed to compile, so everything runs on the slower kernels; a kernel's pipeline failed; the GPU backend failed to start and the CPU took over). The same pair is in `GET /api/web/activity`, each warning is printed to the server log as a `[WARNING: ...]` line when it happens, `gpu::warnings()` returns them in the library and `oxi_warnings()` in C |
| `GET /v1/models`, `/v1/models/{id}` | all | the whitelist, with `kind` and `loaded`; a chat model, once loaded, also has `capabilities` (`["completion", "vision"]` when it takes images, else `["completion"]`) and `modalities` (`{"input": ["text", "image"], "output": ["text"]}`) |
| `POST /v1/chat/completions` | chat (Gemma 4 / 3, Qwen 3.5 / 3.8 / 3, Llama 3, Mistral Large 4, DeepSeek V4, Core ML) | system / developer, multi-turn, `image_url` (data: or http(s) URLs) and `video_url` parts (vision checkpoints), `input_audio` (Gemma 4), `tools` / `tool_choice` (`auto`, `none`, `required` or a function: forced calls) with `tool_calls` out, `response_format` (`json_object`, `json_schema`: constrained decoding), `logprobs` / `top_logprobs`, `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `no_repeat_ngram_size` (an extension, as transformers' `generate`: no n-gram of that many tokens twice in the context, the prompt included), `max_tokens` / `max_completion_tokens`, `stop`, `seed`, `n`, `stream` (SSE, `stream_options.include_usage`), thinking with `reasoning_effort` (`none` off, anything else on), `enable_thinking` or `chat_template_kwargs.enable_thinking`, returned as `reasoning_content`; `modalities: ["text", "audio"]` with `audio` (LFM3-Omni, Haru: a spoken reply, [below](#spoken-replies-lfm3-omni-haru)) |
| `POST /v1/messages`, `/v1/messages/count_tokens` | chat | Anthropic's Messages API (Claude Code, the `anthropic` SDKs): see [below](#anthropic-messages-api) |
| `POST /v1/completions` | chat | raw continuations (no template): `prompt` as text, texts or token ids, `echo`, `stop`, `stream` |
| `POST /v1/embeddings` | embedding | `encoding_format` float / base64, `dimensions` (Matryoshka); `prompt_name` (extension) |
| `POST /v1/images/generations` | image (Stable Diffusion, MOSS-Image, SD 3 / 3.5, FLUX.1, Qwen-Image, OmniGen2, Step1X-Edit, BAGEL, Janus-Pro, MiniMax-H3-Image, Nemotron-Labs-Image) | `size`, `n`, `response_format` (`b64_json`, or `url` as a PNG data URL), `stream` (server-sent events with a progress event per step, [below](#image-progress)); extensions `seed`, `steps`, `guidance`, `negative_prompt` (Nemotron-Labs-Image: `temperature`, `top_p`, `top_k`, `structure`, `system_prompt`, `max_tokens`, `guidance`, and for edits `image_guidance`, `source_layout`, `source_prompt`) |
| `POST /v1/images/edits` | image | multipart: `image` (or `image[]`, several for Qwen-Image-Edit-2509+ and OmniGen2), `mask` (Stable Diffusion), `prompt`, the same fields, `strength` (SD 3: image to image) |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | Whisper, Nemotron 3.5 Transcribe ([its differences](nemotron.md#speech-to-text-nemotron-35-transcribe): `prompt` is the instruction, one segment, no word timestamps or beam search) | multipart: `file` (any format oxideav decodes), `language`, `prompt`, `temperature`, `response_format` json / text / verbose_json / srt / vtt, `timestamp_granularities[]` word / segment, `beam_size`, `patience`, `length_penalty`, `best_of`; several `file` parts are transcribed as a batch and answered as a JSON array (json / verbose_json) |
| `POST /v1/audio/speech` | Orpheus, LFM3-Omni (its text to speech, [below](#spoken-replies-lfm3-omni-haru)), ZONOS3-Mini (44.1 kHz, the audio sent once generated; its fields in [ZONOS3-Mini](zonos.md)), Fish Audio Drama 3 (44.1 kHz, the audio sent per chunk of text; `voice` a speaker tag 0 to 15, its fields in [Drama 3](fish-drama.md)), Audio8 TTS (44.1 kHz, streamed as it is generated; `voice` a clip of its voices folder; its fields in [Audio8 TTS](audio8-tts.md)) | `input`, `voice` (the model's voices; OpenAI's names take the default one), `response_format` wav or pcm (16-bit 24 kHz mono, streamed as it is generated), `stream_format` sse; extensions `seed`, `temperature`, `top_p`, `repetition_penalty`; an array as `input` (1 to 64 texts, `voice` one name or an array) is generated as one batch and answered as JSON, `data[i].audio` the base64 WAV (or PCM) of `input[i]` |
| `POST /v1/audio/isolations` | isolation (Inworld Voice Isolator) | `file` (multipart, or base64 in a JSON body), `response_format` wav (float32, the default) / wav16 / pcm / json (base64 voice and background), `output` voice / background, `output_sample_rate` (default 48000, 0 the input's), `lookahead_frames`, `window_frames`, `crossfade_frames`, `block_frames`, `cache_frames`, `chunk_seconds`, `overlap_seconds`, `offline_passes`: see [Voice isolation](voice-isolation.md) |
| `POST /v1/images/depth` | depth (Depth Pro 2) | `file` or `image` (multipart, or base64 in a JSON body), `focal_px`, `exif`, `response_format` json (focal length, fields of view, depth range, base64 `png` / `png16`, `depth` with `include_depth`) / png / png16 / f32: see [Depth estimation](depth.md) |
| `POST /v1/videos/depth` | depth (Depth Pro 2) | `file` or `video` (MP4, MOV, Matroska / WebM, AVI), `mode` (buffered / unbuffered), `window`, `overlap`, `focal_px`, `response_format` json (each frame's depth range, `depth` with `include_depth`) / mp4 (the color preview) / f32: see [Depth estimation](depth.md#video) |
| `POST /v1/ocr` | ocr (OCRia-nano-1) | the image as `file` or `image` (multipart, or base64 in a JSON body), `response_format` json (the page: `lines` with `text`, `confidence`, `box`, `bbox`, `script`, `words`; `text`, `script`, `upside_down`, `sideways`; with LayoutNet loaded (`layoutnet=1` in the model's options) `layout`: blocks, tables with their cells, Markdown; the default) / text / markdown (LayoutNet's), `structure`, `dpi`, `plain`, `det_long_side`, `det_thresh`, `det_merge`, `adaptive`, `gutter`, `orient`, `rescue`, `junk`, `faint`, `clean`, `bidi`, `layout`, `exif` (photos upright per their EXIF orientation, default on; the page's `exif_orientation`): see [OCR with OCRia-nano-1](ocr-nano.md) |
| `GET /v1/audio/voices?model=ID` | speech models, LFM3-Omni | the model's voices up front (it is loaded first when needed): `data` (`id`, `name`, and `label` / `language` for ZONOS3), `default`, `aliases` (OpenAI's names, which take the default), `accepts_any`; `/v1/models/{id}` also has `voices` once the model is loaded |
| `POST /v1/videos` (also `/v1/videos/generations`), `GET /v1/videos`, `GET /v1/videos/{id}`, `GET /v1/videos/{id}/content`, `POST /v1/videos/{id}/cancel`, `DELETE /v1/videos/{id}` | video (MiniMax-H3) | asynchronous jobs in the shape of OpenAI's videos API: see [below](#video-generation) |

Spoken replies, image progress and video jobs are described below; in the library they are
`Api::handle` on their paths, in C `oxi_server_handle` (and the direct entry points named in each
section).

### Spoken replies (LFM3-Omni, Haru)

`POST /v1/chat/completions` with `"modalities": ["text", "audio"]` answers in speech, in OpenAI's
shape (`audio: {"voice", "format"}`, the reply in `choices[0].message.audio`: `data` the base64
audio, `transcript` its text, `id`, `expires_at`; `content` is null), from two kinds of model:

- **LFM3-Omni** (`LiquidAI/LFM3-Omni-2B`, a chat model): the last user message (text, images,
  `input_audio`) and the system prompt (default "Respond with interleaved text and audio.") get an
  answer in text and speech (`lfm3_omni::Mode::Interleaved`). Earlier turns are not replayed: a
  spoken reply runs on a session of its own and leaves the model's text conversation as it was.
  `temperature`, `top_k`, `seed` and `max_tokens` (slots: text tokens and 80 ms speech frames)
  sample the speech; the voice is the model's own (`audio.voice` takes `female`, `male` or
  OpenAI's names). In the library `Engine::speak_omni`, in C `oxi_engine_speak_omni`.
- **Haru** (`higurashi-ai/Haru-4B`, a speech model): the user's recording (an `input_audio` part
  of the last user message) is played to the model frame by frame, as in a live conversation,
  then silence until it has answered and gone quiet for 2 s (no text and no sound;
  `end_silence_seconds`), at most `max_reply_seconds` (default 12; both extensions); the reply is what it said after the recording. The system prompt is
  its character card, `audio.voice_reference` (an extension: a `data:` or `http(s)://` URL of a
  clip) the voice it speaks with; `temperature`, `top_k`, `top_p` and `seed` sample its text
  stream. In the library `Haru::respond` (`haru::turn`), in C `oxi_haru_respond`.

`audio.format` is `wav` (default) or `pcm16` (16-bit little-endian mono at 24 kHz). With
`"stream": true` (`pcm16` only, as OpenAI) the chunks carry `delta.audio.transcript` as the text
grows and `delta.audio.data` (base64 PCM16, at most a second each: Haru's as it is made,
LFM3-Omni's once its codec has decoded the reply), then a `finish_reason`, `usage` with
`stream_options.include_usage`, and `data: [DONE]`.

`POST /v1/audio/speech` on LFM3-Omni is its text to speech (`voice` `female`, the default, or
`male`; `instructions` replaces the system prompt "Perform TTS in American English using a
female voice."; `seed`, `temperature`, `top_k`, `repetition_penalty`, `max_tokens`), answered as
WAV or `pcm` once generated. On Haru it is refused (Haru speaks only in answer to a recording).

### Image progress

`/v1/images/generations` and `/v1/images/edits` (a `stream` form field) take `"stream": true`
(as OpenAI's `gpt-image` streaming): the answer is then server-sent events, each with `event:`
and a JSON `data` of the same `type`:

- `image_generation.progress` (edits: `image_edit.progress`, an extension) while the request
  waits: `stage` `waiting` (another request uses the model), `downloading` (`file`, `index`,
  `count`, `done`, `total`), `loading`; then one per denoising step: `stage: "generating"`,
  `image_index` of `n`, `step` of `steps` (Janus: one per image token row, `steps` 0), `elapsed`;
- `image_generation.completed` per image, as OpenAI's: `b64_json` (or `url`), `size`,
  `output_format`, `created_at`, `image_index`;
- `error` with OpenAI's error object if it fails after the stream started.

No preview images are made (`partial_images` is accepted and none is sent: a preview would cost
a VAE decode per step). Without `stream` the request and response are unchanged. The client going
away stops FLUX.1, BAGEL, SD 3 and Janus between steps; the others finish the image first.

Stable Diffusion 3 / 3.5 (a diffusers folder whose `model_index.json` names a
`StableDiffusion3*` class, or an SD 3 GGUF transformer with the `base` option) and
MiniMax-H3-Image are image models of the server: SD 3 generates and edits (image to image,
`strength`, default 0.6; its load options are [SD 3's](sd3.md): `quant`, `text_quant`,
`transformer`, `text_encoder`, `base`, `t5`, `device`, `vae_encoder`); MiniMax-H3-Image generates
from text only, with its own six steps (`steps` and `guidance` are refused) and the text encoder
found as the command line finds it (`OXINFER_H3_TEXT_ENCODER`, the release folder, the caches) or
named with the `text_encoder` option (a folder, a Qwen3-VL-32B GGUF or ComfyUI's single file). Its
model may also be ComfyUI's single-file transformer with the `vae` option (its image VAE file);
`device` (`cpu`: the transformer and VAE decoder stay on the CPU with a GPU there) and
`attention` (`f32`, `split`, `half`) and `inputs` (`half`, `split`) are as for the video model ([video.md](video.md#minimax-h3-image-on-metal)).

### Video generation

MiniMax-H3 (a release folder, or a single-file transformer with the `vae` option) is a `video`
model: `/v1/videos` follows OpenAI's videos API (Sora's) with asynchronous jobs, as a generation
takes minutes to hours.

```sh
curl -s localhost:8080/v1/videos -H 'Content-Type: application/json' \
  -d '{"model": "h3", "prompt": "A red fox trots across a snowy clearing ...", "seconds": "5", "size": "448x256"}'
# {"id": "video_...", "object": "video", "status": "queued", "progress": 0, ...}
curl -s localhost:8080/v1/videos/video_...                     # status, progress, stage, step / steps
curl -s localhost:8080/v1/videos/video_.../events              # the same as server-sent events until it ends
curl -s localhost:8080/v1/videos/video_.../content -o fox.mp4  # once "completed"
curl -s 'localhost:8080/v1/videos/video_.../content?variant=audio' -o fox.wav
curl -s -X POST localhost:8080/v1/videos/video_.../cancel
curl -s -X DELETE localhost:8080/v1/videos/video_...
# first / last frame (multipart, OpenAI's input_reference is the first frame)
curl -s localhost:8080/v1/videos -F model=h3 -F prompt="..." -F input_reference=@first.png -F last_frame=@last.png
```

| Field | |
|---|---|
| `model`, `prompt` | required |
| `seconds` | a number or a string (OpenAI's); frames at 24 fps, rounded up to `17 n + 5` (default 124: 5.2 s). `frames` sets the count directly |
| `size` | `WxH`, multiples of 32 (default: 16:9 at a 768 short edge, or the first frame's aspect ratio) |
| `steps`, `seed` | points of the sigma grid (default 50, or a turbo LoRA's); the noise seed (default random) |
| `input_reference` / `first_frame`, `last_frame` | images: first / last frame to video (`fl2va`) |
| `references` | images, audio clips and videos, in the order the prompt names them (ref2va: a model whose folder is the release's `Ref2VA/`); `ref_image_edge` the short edge image references are encoded at |
| `response_format` | the video file: `mp4` (H.264 + AAC, the default), `webm` (VP9 + Opus) or `avi` (Motion JPEG + PCM) |

In JSON the media are `data:` or `http(s)://` URLs (`input_reference` may be OpenAI's
`{"image_url": ...}`); in a multipart body files (`references[]` repeated) or URL fields. The
`video` object has OpenAI's fields (`id`, `object`, `model`, `status` `queued` / `in_progress` /
`completed` / `failed`, and `cancelled` here, `progress` 0 to 100, `created_at`, `completed_at`,
`expires_at`, `size`, `seconds`, `prompt`, `error`) and some of its own: `stage` (`queued`,
`waiting`, `downloading` with `download`, `loading`, `generating`, `decoding`, `encoding`),
`step` / `steps`, `frames`, `fps`, `has_audio`, `elapsed`, `variants` (the byte size of each
file). Jobs run one at a time per model (others wait, `stage: "waiting"`); cancelling stops
the job between two denoising steps, or before the model loads. `content?variant=` is `video`
(by default MP4, as OpenAI's: H.264 and the stereo 32 kHz soundtrack as AAC-LC, encoded by
oxideav; the request's `response_format` picks `webm`, VP9 and Opus at 48 kHz in WebM, or `avi`,
Motion JPEG and 16-bit PCM; built without `media` there is no video file), `audio` (WAV) or
`thumbnail` (the first frame as PNG). Finished
jobs keep their files in memory for an hour (`expires_at`); the server keeps the last 32 jobs.

The model's options: `text_encoder`, `transformer`, `vae`, `audio_vae` (parts replacing the
folder's: paths, `org/name/file.safetensors` or repo ids), `quant` (the text encoder's: `bf16` or
`q8`), `lora` (`FILE[:SCALE]`, repeatable; a turbo LoRA sets the step count and the shifts),
`shift`, `audio_shift`, `corrector` (Sulphur-3's) and its `corrector_*` settings, `device`
(`cpu`: the transformer, its VAE decoder and the corrector stay on the CPU with a GPU there) and
`attention` (the Metal transformer's attention operands: `f32`, the default, `split` or `half`)
and `inputs` (its projections' inputs: `half`, the default, or `split`: hi + lo half pairs, the
default with a LoRA, [video.md](video.md#on-metal)). In the library the pipeline is `minimax_h3::video::VideoPipeline`
(`VideoParams::cancel` stops it), the files `minimax_h3::encode`; in C `oxi_video_load`,
`oxi_video_generate` and `oxi_video_free`.

### Anthropic Messages API

`POST /v1/messages` answers Anthropic's Messages API over the same chat models, so that clients
written for it work against a local model, Claude Code among them:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080     # the server, without /v1
export ANTHROPIC_AUTH_TOKEN=oxinfer                  # the API key, or anything without one
export ANTHROPIC_MODEL=qwen ANTHROPIC_SMALL_FAST_MODEL=qwen ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen
claude
curl -s localhost:8080/v1/messages -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model": "qwen", "max_tokens": 256, "messages": [{"role": "user", "content": "Hi"}]}'
```

The request is the chat request it means (`anthropic_to_openai` in `src/server/openai.rs`):
`system` (a string or text blocks) as the system prompt; `text`, `image` (base64 or URL sources)
and plain-text `document` blocks; `tool_use` blocks of assistant turns as tool calls and
`tool_result` blocks (`is_error` prefixes the text with `Error:`) as tool results; `tools`
(`name`, `description`, `input_schema`; Anthropic's server tools such as web search are left
out) and `tool_choice` (`auto`, `any`, `tool`, `none`); `thinking` (`enabled` or `adaptive`:
thinking on, `disabled`: off, absent: the model's default); `max_tokens` (required, as with
Anthropic), `temperature`, `top_p`, `top_k`, `stop_sequences`. Consecutive messages of one role
are one turn, as with Anthropic. `cache_control` marks (on any block, the system prompt or the
tools), `metadata`, the `anthropic-beta` and `anthropic-version` headers and the `?beta=true`
query are accepted and ignored (the server's own cache reuse applies). The reply has
`thinking`, `text` and `tool_use` content blocks, `stop_reason` `end_turn`, `max_tokens`,
`stop_sequence` (with `stop_sequence` set) or `tool_use`, and `usage` (`input_tokens`,
`cache_read_input_tokens`, `output_tokens`). With `"stream": true` it is Anthropic's event
stream: `message_start`, `content_block_start` / `content_block_delta` (`thinking_delta` and a
closing `signature_delta`, `text_delta`, and `input_json_delta` with a tool call's whole input) /
`content_block_stop`, `message_delta`, `message_stop`, and `ping` events while the model loads or
reads the prompt. Errors have Anthropic's shape (`{"type": "error", "error": {"type",
"message"}}`), a missing key or a malformed body included. The key may come as `x-api-key` as
well as `Authorization: Bearer`.

Thinking blocks carry a `signature`: an HMAC-SHA256 of the thinking text under the server's key
(`--thinking-key KEY`, key `thinking_key`; by default a random key per start). Clients such as
Claude Code send the blocks back in later turns, interleaved with `tool_use` and `tool_result`
turns; a block whose signature verifies is accepted, and its reasoning counts in
`count_tokens` within the current tool loop (the model's own record of the conversation holds
it, so it is not fed again). A block that does not verify (edited, from another server, or from
before a restart with a random key) and `redacted_thinking` blocks are dropped without an error:
nothing a client sends as thinking reaches the model. Servers that share a key accept each
other's blocks, and a set key keeps them valid across restarts.
`/v1/messages/count_tokens` counts the system prompt, messages and tool definitions with the
model's tokenizer (loading it if needed), plus four tokens per message for the template: an
estimate within a few tokens. Not supported: the server tools, PDF documents, images in tool
results, the Batches and Files APIs. In the library it is `Api::handle`
on that path, in C `oxi_server_handle`.

### Progress before the first token

A streamed chat or completion (`"stream": true`) says what it is waiting for, so that a client is
not left without a byte while a model downloads or a long prompt is read on a slow device:

```
: loading qwen
: downloading qwen: model.safetensors 42% (735.0 MB of 1.75 GB)
: waiting: qwen is busy with another request
: reading the prompt: 512 of 1219 tokens
data: {"id":"chatcmpl-...","object":"chat.completion.chunk", ...}
```

These are SSE comment lines: every SSE client skips them (the `openai` clients included), and
they keep the connection from timing out. At most one a second per stage, plus the one that
completes a download or a prompt; small files (configuration, tokenizer) are not reported. With
`"stream_options": {"progress": true}` (an extension) they are events with a JSON body instead,
for clients that want to show them:

```
event: progress
data: {"object":"progress","model":"qwen","stage":"downloading","file":"model.safetensors","index":1,"count":1,"done":735000000,"total":1750000000}
```

`stage` is `waiting`, `downloading` (`file`, `index` of `count` files, `done` of `total` bytes),
`loading` or `prompt` (`done` of `total` tokens, per prefill chunk). A client that closes the
connection during a download cancels it (the partial file is kept and resumed by the next
request). Requests that are not streamed, and the other endpoints, are unchanged.

Errors have OpenAI's shape (`{"error": {"message", "type", "param", "code"}}`) and status codes:
400 for a bad request (also a model of the wrong kind for the endpoint), 401 without the API key,
404 for an unknown model (`model_not_found`) or endpoint, 405, 413 over the body limit, 500 when
loading or running fails, 507 when a model named on request does not fit on the disk or in memory. The paths also work without `/v1`.

The kind of each model (chat, embedding, image, transcription, speech, video, isolation, ocr) comes from its files
(fetching only the small configuration files of a repo id that is not in the cache yet), or the
`kind` option. Models load on their first request, or all at start with `--preload`. A chat
model runs up to `--parallel` requests at once (default 4, see [Concurrent
requests](#concurrent-requests)); other models run one request at a time, the others waiting for
it. When a model to load would not fit in the
memory budget (`--memory-gb`, default 80% of RAM; each model counts the size of its weight files,
scaled for `quant`, or its `memory_gb` option) next to the loaded ones, the least recently used
idle models are unloaded first; so are those that took GPU memory when a chat model to load
would not fit in the GPU's free memory, and any idle model while the system has less memory
available than its reserve (20% of RAM, at most 2 GB; other programs' demand counts). Models
unused for `--idle-unload` minutes (default 30, `0` never; `idle_unload_min`) are unloaded too.
A chat request that extends the previous conversation of a model
(the same messages, the reply as it was returned and new messages, as chat clients send them)
reuses its cache: only the new tokens are fed (`usage.prompt_tokens_details.cached_tokens`); this
works for Qwen3.5's recurrent layers too, and a conversation whose earlier messages carry media is
fed again. Models without recurrent state also keep the prefix a new conversation shares with the
previous one. With `--prompt-cache DIR` (or `auto`; per model `prompt_cache=`), conversations are
also kept on disk across requests, conversations and restarts and restored when a prompt starts
with one (see [Session snapshots and the prompt cache](sessions.md#session-snapshots-and-the-prompt-cache)).

### Concurrent requests

Several clients on one chat model (opencode sessions, agents, API clients) run at the same time:
each model has `--parallel N` sessions (default 4; per model `parallel=N`; `1` for one request
at a time), each with its own caches over the same weights (`EnginePool`, `src/engine_pool.rs`).
A request takes the idle session whose last conversation shares the most of its beginning (the
same leading messages, then the common start of the first that differs if at least 256 bytes:
conversations sharing only their first words keep their sessions), else an empty one, else the
least recently used one, so each client's conversation keeps its cache from one turn to the next, and a new
conversation with the same system prompt reuses that prefix of a session's cache (models without
recurrent state; with a prompt cache, `--prompt-cache`, sessions also restore each other's
snapshots from disk). With every session busy, a request waits (`: waiting: ... is busy` while
streamed).

The requests of a model share its device in turns (`src/batch.rs`):

- **Prompts are read in chunks** (the prefill chunk, 1024 tokens on the GPU) and between two
  chunks the other requests run, so a long prompt does not stall the replies being decoded; when
  several prompts wait they take chunks in turn, first come first served.
- **Plain decode steps are batched**: the next tokens of every request decoding at that moment run
  as one pass (`TextLm::forward_batch`; on Metal one multi-token plan step whose rows are the
  sessions' tokens, each attending over its own caches at its own position; on the CPU
  `QwenModel::forward_hidden_batch`), the weights read once for all of them. A step waits for
  every decoding request's token (each samples and streams its own outside the turn), at most 3
  ms. Batched steps and prompt chunks alternate when both wait. Up to `max_batch=N` rows a step
  (default 8).
- **Alone on its model a request runs as before**: speculative decoding (DSpark, MTP, prompt
  lookup), greedy passes chained on the device. A chain stops queuing passes when another
  request arrives; a turn that starts while others are decoding decodes plainly (DSpark: the
  draft catches up on those tokens at the next turn's prompt).

**Each reply equals the reply its request would get alone** (greedy, or with its own seed): a
batched row computes exactly what a one-token step of its session would. On the CPU and with the
exact multi-token Metal kernels (the default) the logits are bit for bit those of one-token steps
(`qwen_batch::tests::forward_batch_logits_match_one_token_steps`, `tests/gpu_chain.rs`). Gemma 4
26B-A4B Q4_0 once gave 3 of 4 replies equal to their chained ones alone (4 of 4 with
`OXINFER_GPU_CHAIN=0`); on master after the
[Metal decode round 3](performance.md#metal-decode-round-3-november-2026) its chained passes
gave the tokens of its one-token steps on the four prompts of `concurrent_chats_match_alone`
(`examples/dec_ab ... chain steps:chain=0`, with and without prompt lookup), and the test passed
5 times out of 5 on it (also with the MoE layers in the order of before,
`OXINFER_DEC_MOE_ORDER=0 OXINFER_DEC_TOPK=0`), so the cause was not found (with its MoE rows batched, 4 of 4 and 8 of 8 concurrent requests, once 7 of 8: the same request also with `max_batch=1`, 8 of 8 with `OXINFER_GPU_CHAIN=0 lookup=0`). Not covered: after
a plain turn of a DSpark model, the next turn reads that reply as prompt (prefill
kernels) where alone it would have been verified tokens, so later turns may round differently,
as a restored snapshot does. `batch_fast=1` runs batched steps of 5 rows or more on the
fast-verification kernels (as `fast_verify` does for speculative rounds: logits within ~1e-4 of
one-token steps, so a reply may then depend on the requests decoded with it): Qwen3-8B Q8_0 at 8
rows 21 ms a step against 34 ms exact (Q4_K_M 39) (see [performance](performance.md#concurrent-requests-the-server)
for why exact steps stop scaling there).

Batched on the GPU: dense models (Qwen3, Llama, Gemma 3 / 4 dense, and Qwen3.5 / 3.6 / 3.8,
whose Gated DeltaNet layers run each row's one-token conv and delta-rule kernels on that
session's own recurrent states) and mixtures of experts (Gemma 4 26B-A4B, Qwen3 / Qwen3.5 MoE).
A batched step runs each row's routed experts with the one-token expert kernels' arithmetic: the
rows' gate / up pairs side by side, every row's weighted down sums in one dispatch
(`mvx*_bsum`; `OXINFER_XB=0`: a dispatch per row, the same bits), so that the experts the rows
share come from memory once (Gemma 4 26B-A4B Q4_0, 8 rows: 26 ms a step, one row 6.3 ms; see
[docs/performance.md](performance.md#concurrent-requests-the-server)).
Inkling-2 on Metal batches its decode steps exactly: one pass of up to 16 rows whose matvecs,
expert groups (tokens choosing the same expert read it once) and mixes run over all the rows,
and whose ops on a session's state (queries, keys and values into its caches, attention, the
convolutions after the attention and the experts) run for each row on its own caches and rings
at its own position, encoded as its one-token pass encodes them
(`tests/gpu_chain.rs::inkling_batched_rows_match_one_token_passes`: the logits bit for bit). A
request with the DSpark drafter (the default) decodes plainly beside others, from a checkpoint
of its reply's start (one more than `prompt_checkpoints`), which its next turn goes back to
for the draft to read the reply. Checked on Inkling-2-Mini UD-IQ3_XXS: 4 and 8 concurrent
greedy requests give the replies each gets alone on a fresh server (the drafter on), and 4
seeded ones at temperature 0.8 too with `OXINFER_DEVICE_SAMPLING=0`, `draft=none,lookup=0`.
(With the defaults a sampled request alone draws its tokens on the device and in speculative
rounds, whose draws differ from the host sampler's that batched rows take: the same
distribution, not always the same tokens from a seed.)
DeepSeek V4, streamed models and sessions with
interventions run their decode steps one after the other, in turns (still interleaved with the
other requests' prompts).

**The sessions' memory.** Each session's caches grow with its conversation (Inkling-2-Mini at 128k
tokens: about 4 GB each), and an idle session keeps them for the next turn of its client. The
sessions of all chat models together may hold `--session-cache-gb` (default `auto`: a quarter of
the machine's memory, at least 2 GB; `none`: no limit): after a request, while they hold more,
the least recently used idle session lets go of its caches, the most recently used one kept.
With a prompt cache (`--prompt-cache`) the conversation is stored there first (unless it already
is), so that when its client comes back it is restored from the disk instead of read again; it
is read again otherwise. While the system is short of memory (other programs' demand included)
idle sessions let go of their caches whatever the budget, the least recently used first, before
idle models are unloaded. The caches counted are the KV caches on the host and their device
copies, recurrent states and checkpoints (Gemma / Llama, Qwen3 / 3.5, Inkling-2, DeepSeek V4;
other families' sessions are let go of only under memory pressure). The Status view's
`session_cache` (`GET /api/web/activity`) has the bytes held and the budget.

| key | where | |
|---|---|---|
| `parallel` | `--parallel`, server option, model option | sessions of each chat model (default 4) |
| `max_batch` | model option | most rows of a batched step (default 8) |
| `batch_fast` | model option | batched steps of 5+ rows on the fast kernels (default off) |
| `decode_chunk` | model option | most prompt tokens a chunk reads while others decode (default 0: the model's chunk) |
| `session_cache_gb` | `--session-cache-gb`, server option | memory of all sessions' caches (default `auto`) |

The library has them as `ServerOptions::parallel` / `ServerOptions::session_cache` and
`EnginePool::new(engine, slots, BatchOptions)` (`Engine::fork` for one more session of a loaded
engine; `EnginePool::set_cache_budget` bounds a pool's sessions the same way, `trim`, `evict` and
`idle_caches` act on them directly, `Engine::cache_bytes` / `Engine::evict_cache` on one); the C
API through `oxi_server_new`'s options (`parallel=N`, `session_cache_gb=F`, model options
`parallel=`, `max_batch=`, `batch_fast=`, `decode_chunk=`). `OXINFER_BATCH_TRACE=1` prints a line per batched step
(rows, gather and run times) and per prompt chunk (tokens, position, the wait for the turn, run
time).

### Status, cancelling and the caches

With `web=1` (`oxinfer web`) the GUI's **Status** view shows what the server is doing and acts on
it; its API is the management API's (see [docs/web.md](web.md#management-api) for the fields):

```sh
curl -s localhost:8080/api/web/activity | jq '.requests'      # running and queued requests
curl -s -X POST -H 'Content-Type: application/json' localhost:8080/api/web/requests/req-12/cancel
curl -s -X POST -H 'Content-Type: application/json' -d '{"id": "qwen", "cancel": true}' localhost:8080/api/web/models/unload
curl -s -X POST -H 'Content-Type: application/json' -d '{"model": "qwen"}' localhost:8080/api/web/prompt-cache/clear
curl -s -X POST -H 'Content-Type: application/json' -d '{}' localhost:8080/api/web/caches/free
```

| Endpoint | |
|---|---|
| `GET /api/web/activity` | uptime and version, the requests in flight and the last 20 finished, the models with their memory, GPU memory, device and streaming caches, the process (CPU use, memory), the machine (memory, pressure), the GPU, the prompt caches |
| `GET /api/web/requests` | the requests in flight: `id`, `kind`, `model`, `stage` (`queued` while another request uses the model), `elapsed`, `tokens`, `tokens_per_sec`, `step`, `client` |
| `POST /api/web/requests/{id}/cancel` | cancels one (also a `video_...` job) |
| `POST /api/web/models/unload` | `{"id"}`; `"cancel": true` cancels its requests first |
| `GET /api/web/prompt-cache`, `POST /api/web/prompt-cache/clear` | the snapshots per directory and model; `{"model"}`, `{"fingerprint"}` or `{}` removes them |
| `POST /api/web/caches/free` | the streamed models let go of their cached experts |

A cancelled request stops at its next stopping point and frees its model for the next one: a
request waiting for its model gives up at once, a generation stops at its next token (or
denoising step, or streamed write); the prompt pass itself is not interrupted. Its client gets
status 499 with `"code": "cancelled"`, or, streamed, an error event (`data: {"error": ...}` and
`data: [DONE]` for chat and completions, `event: error` for images and `/v1/messages`).

These endpoints follow the management API's rules: the API key when one is set, JSON bodies and
the page's own origin for changes, a loopback `Host` on a loopback listener, and read-only (403)
on a non-loopback listener without a key. Without `web=1` they are not served; the library has
them as `Api::status` (the same JSON), `Api::requests`, `Api::cancel`, `Api::unload`,
`Api::clear_prompt_cache` and `Api::free_caches`, and `Engine::clear_prompt_cache` for one
engine; the C API as `oxi_server_status`, `oxi_server_cancel`, `oxi_server_unload_model`,
`oxi_server_clear_prompt_cache`, `oxi_server_free_caches` and `oxi_engine_clear_prompt_cache`.

### Profiles, episodes and outcomes

A chat request for `MODEL@PROFILE` runs `MODEL` with the profile's LoRA adapter when it has one
and is logged as an episode (the prompt's and the reply's tokens) under the response's `id`; the
outcome of a response is reported later. A request for `MODEL` is neither adapted nor logged.
[Learning from outcomes](learning.md) has the file formats, the adapter's life cycle and the
notes for agent harnesses.

```sh
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen@myrepo", "messages": [{"role": "user", "content": "Fix the test."}]}' | jq -r .id
curl -s localhost:8080/v1/episodes/chatcmpl-1a21c98f7fe0001/outcome -H 'Content-Type: application/json' \
  -d '{"result": "bad", "kind": "tests", "note": "2 of 14 failed"}'
```

| Endpoint | |
|---|---|
| `POST /v1/episodes/{id}/outcome` | `{"reward": -1..1}` or `{"result": "good" \| "bad"}`, with optional `kind` and `note` |
| `POST /v1/outcomes` | the same with `id` / `ids`, or `{"outcomes": [...]}` |
| `GET /v1/episodes`, `GET /v1/episodes/{id}` | the logged episodes with their last outcome (`limit`, `after`, `order`, `model`, `profile`) |
| `GET /v1/models/{id}/profiles` | a model's profiles; `GET /v1/models` lists their names as `profiles` |
| `DELETE /v1/models/{id}/profiles/{profile}/adapter` | removes a profile's adapter, keeps its logs |

Profiles need a loopback listener or `--api-key`, and these endpoints follow the management
API's rules (the key, a loopback `Host` on a loopback listener, JSON bodies and no cross-site
requests for changes, read-only without a key on a non-loopback listener). `--learn-dir DIR`
sets the directory (default `~/.cache/oxinfer/learn`), `--learn-dir none` turns profiles off.

### opencode

When the server finds opencode's global configuration directory (`$XDG_CONFIG_HOME/opencode`, else
`~/.config/opencode`, of the user running the server), the GUI's **Models** and **Status** views
offer **Add to opencode** on each chat model; an added model shows **In opencode** with **Remove
from opencode**, or **Update** when its entry no longer matches what oxinfer knows of the model.
The buttons are hidden when opencode is not set up.

```sh
curl -s localhost:8080/api/web/opencode | jq '.models[] | {id, state}'
curl -s -X POST -H 'Content-Type: application/json' -d '{"model": "qwen"}' localhost:8080/api/web/opencode/add
curl -s -X POST -H 'Content-Type: application/json' -d '{"model": "qwen"}' localhost:8080/api/web/opencode/remove
```

Adding writes the model under an `oxinfer` provider, created when missing:

```json
"provider": { "oxinfer": { "npm": "@ai-sdk/openai-compatible", "name": "oxinfer (local)",
  "options": { "baseURL": "http://127.0.0.1:8080/v1" },
  "models": { "qwen": { "name": "qwen (oxinfer)", "reasoning": true, "tool_call": true,
    "attachment": false, "temperature": true, "modalities": { "input": ["text"], "output": ["text"] },
    "limit": { "context": 40960, "output": 10240 } } } } }
```

- `attachment` and `image` in `modalities.input`: the model takes images. Once the model has been
  loaded this is what the engine says (`Engine::accepts_images`: its vision tower, or for a GGUF
  model the mmproj file beside it, is there); before that, its files say it (an mmproj beside it,
  a `vision_config`), or the catalog does for a vision model whose GGUF projector its load fetches.
- `reasoning`: its chat format has a thinking block; `tool_call`: it has tool calls (before the
  first load: the catalog, or the chat template).
- `limit.context`: the model's `max_context` option, else its trained context (32768 when
  neither is known); `limit.output` a quarter of it, from 1024 to 32768 tokens.
- `options.baseURL` is this server as a program on this machine reaches it (`https` with TLS),
  and `options.apiKey` its API key when it has one. An existing provider keeps its name and
  other options; an existing entry keeps its `name` (rename it freely) and any other key.

The file is `opencode.jsonc` when there is one (opencode reads it last), else `opencode.json`
(created when missing, readable by its owner only). It is read as JSONC (comments, trailing
commas) and changed with minimal text edits: every other key and provider, the comments and the
indentation stay as they were (comments inside a replaced or removed entry go with it, and the
answer says so). Each write first copies the file to `opencode.json.oxinfer-<unix time>.bak`
(the last five are kept) and replaces it atomically (a temporary file renamed over it, with the
same permissions; through a symbolic link, the file it points at). A file that cannot be parsed
is not touched (409, `opencode_config`).

Removing deletes the model's entry, and the `oxinfer` provider when it has no model left. When
opencode's `model` (or `small_model`) names the removed model, the answer's `default_model` and
`message` say so; the setting is left for you to change.

| Endpoint | |
|---|---|
| `GET /api/web/opencode` | `{available, detected, writable, dir, path, exists, error, provider, base_url, default_model, models, other}`: `models` the server's chat models with `state` (`absent`, `current`, `outdated`), `present`, `up_to_date`, `capabilities` (`vision`, `thinking`, `tools`, `context`), `source` (`loaded`: the engine's; `files`: not loaded yet) and the `entry` it would write; `other` the provider's models the server does not list. `available` is false (and nothing else is said) for a request from another machine |
| `POST /api/web/opencode/add` | `{"model"}`: writes or updates its entry; answers `{model, changed, created, path, backup, comments_removed, message, entry, source}` (nothing is written when the file already says it) |
| `POST /api/web/opencode/remove` | `{"model"}`: answers `{model, changed, path, backup, provider_removed, default_model, comments_removed, message}` |

They follow the management API's rules (the API key, JSON bodies from the page's own origin, a
loopback `Host` on a loopback listener, read-only without a key on a non-loopback listener), and
since they write to the home directory of the user running the server, they also need a request
from this machine: a loopback listener, or a client connecting from a loopback address (403,
`opencode_local_only`). The library has them as `oxinfer::opencode` (`detect`, `Config::add`,
`Config::remove`, `Config::state`, `Model::new` from `Engine::capabilities`), the C API as
`oxi_opencode_detect`, `oxi_opencode_add` and `oxi_opencode_remove`.

### Models on request

A request may name a model that is not listed: a repo id (`org/name`, or `org/name:QUANT` for a
GGUF repo) or a preset of the catalog. It joins the list (`/v1/models`), its files are fetched
when the cache does not have them all, and it is loaded and run, all within that request (a
streamed chat request reports the download and the load as progress events). Before fetching,
the server checks that the missing files fit on the cache's disk, keeping 1% of it (at least
1 GB) free, and that the model fits in `--memory-gb`; otherwise the request fails with a 507
(`insufficient_storage` or `insufficient_memory`) and nothing is fetched. A name the hub does not
have is a 404 (`model_not_found`). Local paths are never taken this way (give them with
`--model`). [LongCat-Video-Avatar](longcat-avatar.md) is refused by name (its repository id and
its presets): no route serves it.

```sh
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "ggml-org/Inkling-2-Mini-GGUF:Q8_0", "messages": [{"role": "user", "content": "Hi"}]}'
```

This is on for `oxinfer serve` and `oxinfer web` (`--no-auto-models` turns it off; in the library
and the C API, `auto_models=1` turns it on). It only works where models may be managed: a
loopback listener, or one with `--api-key`. Requests from another site's page (an `Origin` that is
not the server's, or on a loopback listener a `Host` that is not a loopback name) never add
models.

Model options: those of the engine for chat models (`quant`, `device`, `mmap`, `stream`, `mmproj`,
`draft`, `mtp`, `lookup`, `prompt_cache`, `max_context`, `prefill_chunk`, `gpu_decode`, `gpu_chain`, ...; see
[As a library](library.md#as-a-library)),
of the loader for the other kinds (Whisper `encoder` / `decoder` precision, Orpheus `snac` and
`quant`, FLUX `transformer` / `base`, Stable Diffusion `lora` and `gpu_pass`, ...), and:

| Key | Meaning |
|---|---|
| `kind` | `chat`, `embedding`, `image`, `transcription`, `speech`, `video`, `isolation` or `ocr` (default: detected) |
| `system` | system prompt when a request has none (default none; `default` for the CLI's prompt with the local date, weekday and time zone, the same all day; a conversation keeps the one of its first turn) |
| `think` | thinking by default (`1`); requests override it |
| `max_tokens` | reply length when a request sets none (default 4096) |
| `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty` | sampling defaults (default OpenAI's: temperature 1, top-p 1; no top-k, min-p or penalty) |
| `no_repeat_ngram` | the n-gram ban when a request sets no `no_repeat_ngram_size` (default 0: off) |
| `prompt_name` | embeddings: the named prompt used by default (`query`, ...) |
| `memory_gb` | the memory the model is counted for |

Server options: `--listen ADDR` (port 0 picks one), `--api-key KEY` (or `$OXINFER_API_KEY`;
requests need `Authorization: Bearer KEY`), `--cors ORIGIN`, `--max-body-mb N` (default 16),
`--preload`, `--download`, `--memory-gb F`, `--idle-unload MIN`, `--no-auto-models`,
`--cache-dir DIR`, `--threads N`,
`--prompt-cache DIR|auto` and `--prompt-cache-gb F` (the chat models' prompt cache),
`--learn-dir DIR|none` (the [profiles](#profiles-episodes-and-outcomes)' logs and adapters),
`--thinking-key KEY` (the key of [thinking signatures](#anthropic-messages-api)),
`--tls-cert FILE --tls-key FILE` or `--self-signed` for HTTPS, `--workers N` (connections served
at once, default 32), `--quiet`; `web=1` (the key of `ServerOptions` and `oxi_server_new`; what `oxinfer web` sets) also serves the [web GUI](web.md) and its `/api/web/` management API, whose release check (at most one anonymous request to GitHub every 6 hours, made only when the GUI asks; [Updates](web.md#updates)) `--no-update-check` / `update_check=0` turns off (`update_feed=FILE_OR_URL` reads the release from elsewhere; `serve` without `web` never checks). Media URLs in requests may be `data:` URLs or `http(s)://` URLs,
which the server fetches (`--no-remote-media` turns that off); local paths and `file://` URLs
only with `--local-media`, for trusted clients. `--config FILE` reads the same options as JSON:

```json
{"listen": "0.0.0.0:8080", "api_key": "...", "memory_gb": 64,
 "models": ["qwen=Qwen/Qwen3.8-27B,quant=q4,mtp=1",
            {"id": "embed", "model": "Qwen/Qwen3-Embedding-0.6B", "options": {"prompt_name": "query"}}]}
```

In the library, `oxinfer::server::Api` handles a request without any transport (method, path,
content type, body in; a complete body or a streamed one out), and `oxinfer::server::http::start`
serves it; `Engine::chat` runs a whole conversation (a list of `Message`s) with the cache reuse
above. The C API has `oxi_server_new` (the options as `key=value` pairs or JSON),
`oxi_server_start` / `oxi_server_stop`, `oxi_server_handle` (one request, streamed bodies through a
callback) and `oxi_engine_chat`. `tools/openai_check.py` checks every endpoint with the official
`openai` Python client (chat, multi-turn with cache reuse, streaming, stop sequences, tool calls
and their results, streamed tool calls, thinking, structured outputs, forced tool calls, log
probabilities, completions, errors, vision, embeddings, image
generation and editing, speech as WAV and streamed PCM, transcriptions as JSON, verbose JSON and
SRT): all pass with Qwen/Qwen3-0.6B (chat), Qwen/Qwen3.5-0.8B (vision), BAAI/bge-small-en-v1.5,
stabilityai/sd-turbo, openai/whisper-tiny and the Orpheus Q4_K_M GGUF, and the chat and vision
checks with mlx-community/gemma-4-e2b-it-4bit, which also transcribes `input_audio` parts. The chat
checks with structured outputs, forced tool calls and log probabilities were run with
mlx-community/Qwen3-4B-4bit on the CPU: all pass.

## Structured outputs, forced tool calls and log probabilities

`response_format: {"type": "json_object"}` and `{"type": "json_schema", "json_schema": {"schema":
{...}}}` constrain the reply while it is decoded (`src/constrain/`, no dependency): the schema is
compiled to a grammar over bytes, and at every step the logits of the tokens that would leave the
grammar are masked before sampling, so the answer is always a prefix of a valid document, and a
valid document when the reply ends by itself. A token that spans several grammar symbols (`","`,
`"}]`) is followed through all of them; a multi-byte character split across tokens is tracked byte
by byte (only bytes that keep the text valid UTF-8 pass); special tokens never pass as text; the
end-of-turn token passes only once the document is complete. A thinking block is not constrained
(`reasoning_content` stays free text, but the reply cannot end in it); the grammar starts when it
closes. A reply cut by `max_tokens` has `finish_reason: "length"` and holds the start of a
document, as with OpenAI. The schema is always enforced, with or without `strict`.

The supported schema subset is the one of OpenAI's strict structured outputs:

| Keyword | Notes |
|---|---|
| `type` | `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`, or a list of them |
| `properties`, `required`, `additionalProperties` | keys are written in the schema's order; properties that are not required may be left out; `additionalProperties: false` (also assumed next to `properties`); an object without `properties` takes any keys, with values of `additionalProperties` when that is a schema |
| `items`, `minItems`, `maxItems` | bounds up to 10 000 |
| `enum`, `const` | any JSON values |
| `anyOf` | |
| `$ref`, `$defs` / `definitions` | references inside the schema (`#`, `#/$defs/...`), recursion included |
| `title`, `description`, `default`, `examples`, `format`, `$schema`, `$id`, `$comment`, `deprecated`, `readOnly`, `writeOnly` | constrain nothing (string formats are not enforced) |

Anything else (`pattern`, `minLength`, `minimum`, `multipleOf`, `oneOf`, `allOf`, `not`,
`prefixItems`, `patternProperties`, `additionalProperties` next to `properties`, remote `$ref`,
a reference cycle that never reaches a value, ...) is a 400 naming the keyword and where it is
(`#/properties/n: schema keyword "minimum" is not supported`): a schema is never partly applied.
The root may be any type. Whitespace between JSON tokens is none, one space, or a newline and
up to 16 spaces or tabs (compact or indented output, never an endless run); numbers have at most
15 integer digits, 16 fraction digits and 2 exponent digits; `\u` escapes are whole code points
(no lone surrogates), so every reply parses with a strict parser.

`tool_choice: "required"` and `tool_choice: {"type": "function", "function": {"name": ...}}` force
the reply (after any thinking) to be one tool call, with the same machinery: the grammar is the
model family's call syntax (Qwen3's JSON in `<tool_call>`, Qwen 3.5 / 3.8's `<function=...>` with
`<parameter=...>` elements, Gemma 4's `<|tool_call>call:NAME{...}` with `<|"|>` strings, Mistral's
`[TOOL_CALLS]NAME[ARGS]{...}`, DeepSeek V4's DSML block; the markers as their special tokens) with
the arguments of the tool's `parameters` schema (the subset above; a schema outside it is a 400).
In the formats that write string arguments bare (Qwen 3.5 / 3.8, DeepSeek V4) a string cannot hold
`</`, and a parameter declared as a string now stays one when its text reads as JSON (`"42"`).
A forced call cut by `max_tokens` is not reported as a call. With `response_format` and tools but
no forced call, the model chooses between the document and a call.

`logprobs: true` with `top_logprobs: N` (0 to 20) returns, for every token of the answer (not the
thinking, the delimiters, a tool call or the stop token), `{token, logprob, bytes, top_logprobs}`
in `choices[].logprobs.content`, streamed with the chunk that carries the token's text;
`/v1/completions` takes `logprobs: N` and returns the legacy `tokens` / `token_logprobs` /
`top_logprobs` / `text_offset`. The log probabilities are of the distribution the model produced:
the softmax of its logits at temperature 1, before the repetition penalty, `temperature`, `top_p`,
`top_k` and before any constraint (as OpenAI's, which do not move with the sampling settings; a
token that a constraint forced can therefore be unlikely, and a probability of zero is shown as
-9999). `bytes` are the token's own bytes, so a character split across tokens can be put together.

The mask and the log probabilities are applied in the sampler, which every decoding loop goes
through with the model's logits, on the CPU and with GPU decoding alike. Speculative decoding
keeps them exact: verification masks the target's distribution at each position (a draft token
outside the grammar is rejected and replaced from the masked distribution), and DSpark / MTP
drafts are themselves drawn under the mask of a forked grammar state, so they stay acceptable;
prompt-lookup drafts are checked the same way. Gemma 4's greedy drafter loop, which does not
sample, is bypassed for such requests (plain decoding), and a Core ML model that returns only its
argmax refuses them. In the library these are `ChatConfig::response_format`, `tool_choice` and
`logprobs` (`Reply::logprobs`, `Delta::logprob`); in the C API `oxi_engine_set(e,
"response_format", ...)`.

Checked on real weights with `cargo test --release --test server -- --include-ignored structured
logprobs` (`OXINFER_E2E_CHAT` picks the model, `OXINFER_E2E_OPTS` adds model options): a nested
schema (enum, `anyOf`, array of objects) on four prompts that never describe it, greedy and at
temperature 1, and forced calls (`required` and a named function) on three prompts, of which two
ask for no tool. Every reply that ended by itself parsed and validated, and every forced call was
one call with arguments of its schema: with Qwen/Qwen3.5-0.8B on the CPU (6 of 8 schema replies
complete in 300 tokens, the others cut by the limit; 12 of 12 calls), the same with `lookup=1`
and with `mtp=1`, and with GPU decoding (CUDA; 4 and 5 of 8 complete in two runs, 12 of 12 calls); with
mlx-community/Qwen3-4B-4bit on the CPU, alone and with the DSpark draft
`deepseek-ai/dspark_qwen3_4b_block7` (8 of 8 and 12 of 12 each). Unit tests cover the rest:
accept / reject sets of the grammar and of every schema keyword, masks compared with a direct
check of every token along whole documents on a vocabulary with tokens that span symbols and
split characters, replies sampled from random logits (all valid documents), each family's call
syntax against its own renderer and parser, and prompt-lookup and DSpark rounds under a filter
giving exactly the tokens and log probabilities of plain filtered decoding.

Log probabilities do not depend on streaming: on a newly loaded model the entries of a streamed
and of a plain request are identical (difference 0 on both models). They do follow the cache: a
request that repeats the previous conversation feeds only the tokens the cache lacks, so on
models that keep a prefix (Qwen3, Gemma, Llama) the last prompt token then runs alone on the
decode kernels instead of inside the prompt's pass on the prefill kernels, which is a slightly
different computation (as it always was for the text). With 4-bit weights that is visible: on
Qwen3-4B-4bit the same request repeated (15 of 16 prompt tokens cached) gave the same tokens with
log probabilities up to 0.42 apart, where the logits of the two pass shapes themselves differ by
up to 0.63 (KL 1.3e-4, `examples/prefill_kl` with `--chunk 15` against one pass; 2.4e-5 and KL
2.6e-12 on the bf16 Qwen3.5-0.8B). Qwen3.5's recurrent layers cannot keep a prefix of a repeated
prompt, so there a repeat recomputes it and gives identical values.

Not supported: schema keywords outside the table above, several forced calls in one reply
(`tool_choice: "required"` gives exactly one), audio
output from chat models, the Responses and Assistants APIs, image `url` responses other than data
URLs, and speech formats other than WAV and PCM. Requests that take over 20 s are answered as 200
before they finish (httpsd drops a connection whose handler runs past its 30 s inactivity window),
so an error after that point arrives in a 200 body.
