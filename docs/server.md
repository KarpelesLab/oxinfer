# OpenAI-compatible server

`oxinfer serve` exposes the OpenAI API over a whitelist of models: only the models given with
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
| `POST /v1/chat/completions` | chat (Gemma 4 / 3, Qwen 3.5 / 3.8 / 3, Llama 3, Mistral Large 4, DeepSeek V4, Core ML) | system / developer, multi-turn, `image_url` (data: or http(s) URLs) and `video_url` parts (vision checkpoints), `input_audio` (Gemma 4), `tools` / `tool_choice` (`auto`, `none`, `required` or a function: forced calls) with `tool_calls` out, `response_format` (`json_object`, `json_schema`: constrained decoding), `logprobs` / `top_logprobs`, `temperature`, `top_p`, `top_k`, `max_tokens` / `max_completion_tokens`, `stop`, `seed`, `n`, `stream` (SSE, `stream_options.include_usage`), thinking with `reasoning_effort` (`none` off, anything else on), `enable_thinking` or `chat_template_kwargs.enable_thinking`, returned as `reasoning_content` |
| `POST /v1/completions` | chat | raw continuations (no template): `prompt` as text, texts or token ids, `echo`, `stop`, `stream` |
| `POST /v1/embeddings` | embedding | `encoding_format` float / base64, `dimensions` (Matryoshka); `prompt_name` (extension) |
| `POST /v1/images/generations` | image (Stable Diffusion, FLUX.1, Qwen-Image, OmniGen2, Step1X-Edit, BAGEL, Janus-Pro) | `size`, `n`, `response_format` (`b64_json`, or `url` as a PNG data URL); extensions `seed`, `steps`, `guidance`, `negative_prompt` |
| `POST /v1/images/edits` | image | multipart: `image` (or `image[]`, several for Qwen-Image-Edit-2509+ and OmniGen2), `mask` (Stable Diffusion), `prompt`, the same fields, `strength` |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | Whisper | multipart: `file` (any format oxideav decodes), `language`, `prompt`, `temperature`, `response_format` json / text / verbose_json / srt / vtt, `timestamp_granularities[]` word / segment, `beam_size`, `patience`, `length_penalty`, `best_of`; several `file` parts are transcribed as a batch and answered as a JSON array (json / verbose_json) |
| `POST /v1/audio/speech` | Orpheus | `input`, `voice` (the model's voices; OpenAI's names take the default one), `response_format` wav or pcm (16-bit 24 kHz mono, streamed as it is generated), `stream_format` sse; extensions `seed`, `temperature`, `top_p`, `repetition_penalty`; an array as `input` (1 to 64 texts, `voice` one name or an array) is generated as one batch and answered as JSON, `data[i].audio` the base64 WAV (or PCM) of `input[i]` |

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
| `kind` | `chat`, `embedding`, `image`, `transcription` or `speech` (default: detected) |
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
`--tls-cert FILE --tls-key FILE` or `--self-signed` for HTTPS, `--workers N` (connections served
at once, default 32), `--quiet`. Media URLs in requests may be `data:` URLs or `http(s)://` URLs,
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
