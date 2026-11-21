# OpenAI-compatible server

`oxinfer serve` exposes the OpenAI API over a whitelist of models (`oxinfer web` serves the same
API next to a browser GUI that sets models up and adds them to the list: see
[The web GUI on this machine](web.md)): only the models given with
`--model` can be requested, by their id. Each is `[ID=]SOURCE[,KEY=VALUE...]`: a local folder or
file, a repo id (fetched on first use, or at start with `--download`) or `org/name:QUANT` for a
GGUF repo, an optional id (default: the source) and the model's options. HTTP and HTTPS are
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
| `GET /v1/models`, `/v1/models/{id}` | all | the whitelist, with `kind` and `loaded` |
| `POST /v1/chat/completions` | chat (Gemma 4 / 3, Qwen 3.5 / 3.8 / 3, Llama 3, Mistral Large 4, DeepSeek V4, Core ML) | system / developer, multi-turn, `image_url` (data: or http(s) URLs) and `video_url` parts (vision checkpoints), `input_audio` (Gemma 4), `tools` / `tool_choice` (`auto`, `none`, `required` or a function: forced calls) with `tool_calls` out, `response_format` (`json_object`, `json_schema`: constrained decoding), `logprobs` / `top_logprobs`, `temperature`, `top_p`, `top_k`, `max_tokens` / `max_completion_tokens`, `stop`, `seed`, `n`, `stream` (SSE, `stream_options.include_usage`), thinking with `reasoning_effort` (`none` off, anything else on), `enable_thinking` or `chat_template_kwargs.enable_thinking`, returned as `reasoning_content`; `modalities: ["text", "audio"]` with `audio` (LFM3-Omni, Haru: a spoken reply, [below](#spoken-replies-lfm3-omni-haru)) |
| `POST /v1/messages`, `/v1/messages/count_tokens` | chat | Anthropic's Messages API (Claude Code, the `anthropic` SDKs): see [below](#anthropic-messages-api) |
| `POST /v1/completions` | chat | raw continuations (no template): `prompt` as text, texts or token ids, `echo`, `stop`, `stream` |
| `POST /v1/embeddings` | embedding | `encoding_format` float / base64, `dimensions` (Matryoshka); `prompt_name` (extension) |
| `POST /v1/images/generations` | image (Stable Diffusion, SD 3 / 3.5, FLUX.1, Qwen-Image, OmniGen2, Step1X-Edit, BAGEL, Janus-Pro, MiniMax-H3-Image) | `size`, `n`, `response_format` (`b64_json`, or `url` as a PNG data URL), `stream` (server-sent events with a progress event per step, [below](#image-progress)); extensions `seed`, `steps`, `guidance`, `negative_prompt` |
| `POST /v1/images/edits` | image | multipart: `image` (or `image[]`, several for Qwen-Image-Edit-2509+ and OmniGen2), `mask` (Stable Diffusion), `prompt`, the same fields, `strength` (SD 3: image to image) |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | Whisper, Nemotron 3.5 Transcribe ([its differences](nemotron.md#speech-to-text-nemotron-35-transcribe): `prompt` is the instruction, one segment, no word timestamps or beam search) | multipart: `file` (any format oxideav decodes), `language`, `prompt`, `temperature`, `response_format` json / text / verbose_json / srt / vtt, `timestamp_granularities[]` word / segment, `beam_size`, `patience`, `length_penalty`, `best_of`; several `file` parts are transcribed as a batch and answered as a JSON array (json / verbose_json) |
| `POST /v1/audio/speech` | Orpheus, LFM3-Omni (its text to speech, [below](#spoken-replies-lfm3-omni-haru)), ZONOS3-Mini (44.1 kHz, the audio sent once generated; its fields in [ZONOS3-Mini](zonos.md)), Fish Audio Drama 3 (44.1 kHz, the audio sent per chunk of text; `voice` a speaker tag 0 to 15, its fields in [Drama 3](fish-drama.md)), Audio8 TTS (44.1 kHz, streamed as it is generated; `voice` a clip of its voices folder; its fields in [Audio8 TTS](audio8-tts.md)) | `input`, `voice` (the model's voices; OpenAI's names take the default one), `response_format` wav or pcm (16-bit 24 kHz mono, streamed as it is generated), `stream_format` sse; extensions `seed`, `temperature`, `top_p`, `repetition_penalty`; an array as `input` (1 to 64 texts, `voice` one name or an array) is generated as one batch and answered as JSON, `data[i].audio` the base64 WAV (or PCM) of `input[i]` |
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
named with the `text_encoder` option.

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
curl -s localhost:8080/v1/videos/video_.../content -o fox.avi  # once "completed"
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

In JSON the media are `data:` or `http(s)://` URLs (`input_reference` may be OpenAI's
`{"image_url": ...}`); in a multipart body files (`references[]` repeated) or URL fields. The
`video` object has OpenAI's fields (`id`, `object`, `model`, `status` `queued` / `in_progress` /
`completed` / `failed`, and `cancelled` here, `progress` 0 to 100, `created_at`, `completed_at`,
`expires_at`, `size`, `seconds`, `prompt`, `error`) and some of its own: `stage` (`queued`,
`waiting`, `downloading` with `download`, `loading`, `generating`, `decoding`, `encoding`),
`step` / `steps`, `frames`, `fps`, `has_audio`, `elapsed`, `variants` (the byte size of each
file). Jobs run one at a time per model (others wait, `stage: "waiting"`); cancelling stops
the job between two denoising steps, or before the model loads. `content?variant=` is `video`
(Motion JPEG and the stereo 32 kHz soundtrack as 16-bit PCM in AVI: no pure-Rust H.264 / VP9
encoder is available, so not OpenAI's MP4; `ffmpeg -i clip.avi clip.mp4` converts it; built
without `media` there is no AVI), `audio` (WAV) or `thumbnail` (the first frame as PNG). Finished
jobs keep their files in memory for an hour (`expires_at`); the server keeps the last 32 jobs.

The model's options: `text_encoder`, `transformer`, `vae`, `audio_vae` (parts replacing the
folder's: paths, `org/name/file.safetensors` or repo ids), `quant` (the text encoder's: `bf16` or
`q8`), `lora` (`FILE[:SCALE]`, repeatable; a turbo LoRA sets the step count and the shifts),
`shift`, `audio_shift`. In the library the pipeline is `minimax_h3::video::VideoPipeline`
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
loading or running fails. The paths also work without `/v1`.

The kind of each model (chat, embedding, image, transcription, speech) comes from its files
(fetching only the small configuration files of a repo id that is not in the cache yet), or the
`kind` option. Models load on their first request, or all at start with `--preload`. Each model
runs one request at a time; the others wait for it. When a model to load would not fit in the
memory budget (`--memory-gb`, default 80% of RAM; each model counts the size of its weight files,
scaled for `quant`, or its `memory_gb` option) next to the loaded ones, the least recently used
idle models are unloaded first. A chat request that extends the previous conversation of a model
(the same messages, the reply as it was returned and new messages, as chat clients send them)
reuses its cache: only the new tokens are fed (`usage.prompt_tokens_details.cached_tokens`); this
works for Qwen3.5's recurrent layers too, and a conversation whose earlier messages carry media is
fed again. Models without recurrent state also keep the prefix a new conversation shares with the
previous one. With `--prompt-cache DIR` (or `auto`; per model `prompt_cache=`), conversations are
also kept on disk across requests, conversations and restarts and restored when a prompt starts
with one (see [Session snapshots and the prompt cache](sessions.md#session-snapshots-and-the-prompt-cache)).

Model options: those of the engine for chat models (`quant`, `device`, `mmap`, `stream`, `mmproj`,
`draft`, `mtp`, `lookup`, `prompt_cache`, `max_context`, `prefill_chunk`, `gpu_decode`, ...; see
[As a library](library.md#as-a-library)),
of the loader for the other kinds (Whisper `encoder` / `decoder` precision, Orpheus `snac` and
`quant`, FLUX `transformer` / `base`, Stable Diffusion `lora`, ...), and:

| Key | Meaning |
|---|---|
| `kind` | `chat`, `embedding`, `image`, `transcription`, `speech` or `video` (default: detected) |
| `system` | system prompt when a request has none (default none; `default` for the CLI's prompt with the date and time) |
| `think` | thinking by default (`1`); requests override it |
| `max_tokens` | reply length when a request sets none (default 4096) |
| `temperature`, `top_p`, `top_k`, `repetition_penalty` | sampling defaults (default OpenAI's: temperature 1, top-p 1) |
| `prompt_name` | embeddings: the named prompt used by default (`query`, ...) |
| `memory_gb` | the memory the model is counted for |

Server options: `--listen ADDR` (port 0 picks one), `--api-key KEY` (or `$OXINFER_API_KEY`;
requests need `Authorization: Bearer KEY`), `--cors ORIGIN`, `--max-body-mb N` (default 16),
`--preload`, `--download`, `--memory-gb F`, `--cache-dir DIR`, `--threads N`,
`--prompt-cache DIR|auto` and `--prompt-cache-gb F` (the chat models' prompt cache),
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
