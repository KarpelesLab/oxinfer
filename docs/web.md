# The web GUI on this machine: `oxinfer web`

`oxinfer web` serves the browser GUI of [the WebAssembly build](wasm.md) with this computer as
its backend: the models run natively (every CPU core, Metal, CUDA), the page drives them over
HTTP. The same port also serves the unchanged [OpenAI-compatible API](server.md) under `/v1`, so
other programs (opencode, the `openai` clients, curl) can use the models the GUI loads while it
is open.

```sh
oxinfer web                          # http://127.0.0.1:8090/, opened in the browser
oxinfer web --port 9000 --no-open
oxinfer web --host 0.0.0.0 --api-key "$(openssl rand -hex 16)"   # reachable from the network
oxinfer web --model qwen=Qwen/Qwen3.5-0.8B,quant=q8 --memory-gb 24
# [GUI: http://127.0.0.1:8090/]
# [API: http://127.0.0.1:8090/v1 (OpenAI-compatible)]
```

| Option | |
|---|---|
| `--host ADDR` | address to listen on (default `127.0.0.1`) |
| `--port N` | port (default 8090, or the next free one up to 8109 when it is taken; a port given is used as given; 0 picks one) |
| `--listen ADDR:PORT` | both at once |
| `--no-open` | do not open the GUI in the default browser |
| `--api-key KEY` | requests need `Authorization: Bearer KEY`, the GUI's too (default `$OXINFER_API_KEY`); the browser opened at start gets the key in the address fragment, keeps it in its local storage and removes it from the address bar; another browser asks for it |
| `--model [ID=]SOURCE[,KEY=VALUE...]` | models listed from the start, as with `serve` |

and the other options of [`oxinfer serve`](server.md) (`--memory-gb`, `--cache-dir`, `--threads`,
`--prompt-cache`, `--cors`, `--tls-cert` / `--tls-key`, `--self-signed`, `--preload`, `--quiet`,
...). The GUI's files are built into the binary (`src/server/webui/`, see [Building](#building)),
so nothing else is needed at run time.

## The GUI

The page asks `/api/web/info` at start. When that answers it runs in native mode; hosted on its
own (GitHub Pages, `npm run dev`) the request fails and the page runs the wasm engine in the tab
as before. Both modes are the same Vue app: `web/src/lib/chat.js` and `web/src/lib/image.js` hold
one session interface each, with a wasm implementation (the engine in the tab) and an HTTP one
(`/v1/chat/completions`, `/v1/images/*`), and the views (`ChatView.vue`, `ImageStudio.vue`) drive
whichever they are given. `web/src/lib/api.js` is the client of the server (the management API,
streamed chat, server-sent events). In native mode the page is titled `oxinfer on HOSTNAME` (the
machine's name, from `/api/web/info`'s `hostname`).

| View | |
|---|---|
| **Models** | this computer (memory, the memory budget, GPUs), the cache's disk (free and total), the models in use with their load state, the downloads, the [catalog](cli.md#what-runs-on-this-machine) grouped by kind (chat, vision-language, images, speech to text, text to speech, embeddings, video) with the download size, the memory of the recommended precision, whether it fits this machine and its estimated speed, whether it is downloaded and the notes; any other model by its spec; the downloaded models with size, kind, last use, `catalog` / `in use` / `downloading` / `incomplete` marks, *Load…* and *Delete* (after a confirmation; refused while the model is loaded or downloading) |
| **Chat** | streamed replies, thinking on or off (shown apart, folded once the answer starts), images, audio and video attached to a message for the models that read them (vision checkpoints, Gemma 4's audio, Unlimited-OCR-2's page images), the page's tools (`get_datetime`, `run_javascript` in a sandboxed worker) through `tools` and `tool_calls`, temperature, top-p, the longest reply, the system prompt, the speed of the last reply, *Stop* (closing the request stops the generation) |
| **Image** | the Image mode of the wasm site on the server's image models (Stable Diffusion, SD 3 / 3.5, FLUX.1, Qwen-Image, OmniGen2, Step1X-Edit, BAGEL, Janus-Pro, MiniMax-H3-Image): text to image, and with an input image the edit endpoint (instruction edits, image to image, a mask for inpainting), size, steps and guidance (empty: the pipeline's own), seed, what to leave out; the request is streamed (`stream: true`, [image progress](server.md#image-progress)), so the page says what it waits for (another request, the download, the load) and then *Step N of M*; *Save as PNG* |
| **Transcribe** | a file (any format the server decodes, video included) or a recording from the microphone (made into a 16 kHz WAV in the page), transcription or translation to English, language, plain text, text with segment and word timings, SRT or WebVTT, temperature, beam size, a prompt; copy or download the text |
| **Speak** | text to speech (the speech models, and LFM3-Omni's) with a voice picker filled from [`/v1/audio/voices`](server.md#spoken-replies-lfm3-omni-haru) (the model is loaded to list them; ZONOS3's with their language, a text field for pretrained Orpheus checkpoints, which take any name), temperature, top-p, repetition penalty and seed; each clip with a player and a *Download WAV* link |
| **Voice chat** | the models that answer in speech (LFM3-Omni, Haru): a recording from the microphone or an audio file (made into a 24 kHz WAV in the page), and for LFM3-Omni a typed question too; the system prompt (Haru: its character card), Haru's voice reference clip and longest answer, the seed; each turn with your recording, the model's transcript, a player and *Download WAV* (`/v1/chat/completions` with `modalities: ["text", "audio"]`) |
| **Video** | MiniMax-H3 video jobs (`/v1/videos`): the prompt, seconds, size, steps, seed, first and last frames, references (Ref2VA); the job's progress (its stage, *Step N of M*, the time so far) followed as server-sent events, *Cancel*; the jobs of the server listed (they run on without the page: leave and come back); a finished video played in the page (the AVI's Motion JPEG frames drawn on a canvas in time with the WAV soundtrack, which browsers cannot play as an AVI), *Download AVI* and *Download WAV*, *Delete* |
| **Connect apps** | copy-paste configuration for other programs, with this server's address, API key (hidden until shown) and chat model ids: an `opencode.json` provider block (`@ai-sdk/openai-compatible`), Claude Code's environment (it speaks Anthropic's API, which the server answers at [`/v1/messages`](server.md#anthropic-messages-api)), curl for both APIs, `OPENAI_BASE_URL` / `OPENAI_API_KEY`, the `openai` Python and JavaScript SDKs, Aider, and notes for Continue, Open WebUI and other OpenAI-compatible clients; a copy button on each; a warning when the server listens on loopback only (other machines cannot reach it) |
| **Embed** | a query against texts, one per line, ranked by cosine similarity, with the prompt name and Matryoshka dimensions |

Each view has a model picker: the server's models of that kind, their state (loaded, loading
with its progress, an error), *Load now* / *Unload*, and the downloaded models of that kind not
loaded yet (with a precision for chat checkpoints). A kind is told from the files as the server
does (`kind` option to override it).

**Setting a model up** (*Set up…* in the catalog, a spec typed by hand, or *Load…* on a
downloaded model) is one dialog in three steps:

1. *What it needs*: the files a download takes and their size, what the cache already has and
   the free space on its disk (`POST /api/web/plan`, which lists the repository and reads its
   small index files into memory, writing nothing). The download is refused when it does not fit.
2. *Download*: a background job with its progress streamed as server-sent events: bytes done of
   the total, speed, time left, the file being fetched and each file's progress. *Cancel* stops
   it (the partial file is kept and the next download resumes it). The job runs in the server, so
   closing or reloading the page does not stop it: the dialog and the Models view find it again
   and follow it.
3. *Load*: the precision (for a catalog checkpoint, each variant with its memory, where it runs
   and the estimated speed, the recommended one selected), an optional name and model options
   (`think=1`, `max_context=8192`, `device=cpu`, ...: those of `serve --model`). The load runs in
   the background and its state (downloading what is still missing, reading the weights, ready,
   the error) shows in the dialog and the views. *Use it* opens the view for its kind.

A loaded model joins the API's model list (`GET /v1/models`) under its name: what the GUI loads,
other clients can use, and the memory budget unloads the least recently used idle models as
`serve` does. A model loaded again with other options replaces the earlier one of that name.

### Connecting other programs

The snippets use the address the page was opened at (`http://127.0.0.1:8090/v1` by default) and
the model ids of the server's list (Models in use): clients can only name those, and one that is
not loaded loads on its first request, which waits for it (a downloaded model that is not in the
list must be loaded once from the GUI, or given with `--model`).

- **opencode**: `opencode.json` at a project's root, or merged into
  `~/.config/opencode/opencode.json`:

  ```json
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "oxinfer": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "oxinfer (local)",
        "options": { "baseURL": "http://127.0.0.1:8090/v1" },
        "models": { "Qwen/Qwen3.5-0.8B": { "name": "Qwen/Qwen3.5-0.8B (oxinfer)" } }
      }
    },
    "model": "oxinfer/Qwen/Qwen3.5-0.8B"
  }
  ```

  (`"apiKey"` in `options` with `--api-key`.) With opencode 1.19.6,
  `opencode run -m oxinfer/Qwen/Qwen3.5-0.8B "Reply with just the word: pong"` answered `pong`.
  opencode's newer schema also has a `providers` key (with `package` and `settings`); `provider`
  is still read.
- **Claude Code**: `ANTHROPIC_BASE_URL=http://127.0.0.1:8090` (no `/v1`),
  `ANTHROPIC_AUTH_TOKEN` (the key, or any value without one), `ANTHROPIC_MODEL` and
  `ANTHROPIC_SMALL_FAST_MODEL` / `ANTHROPIC_DEFAULT_HAIKU_MODEL` set to a model id, then `claude`.
  Its system prompt and tool definitions are long, so the first request reads a long prompt, and
  a small model is a weak agent: use the largest chat model the machine runs well.

## Management API

Under `/api/web/`, apart from `/v1`. Every endpoint but `info` needs the API key when one is set.

| Endpoint | |
|---|---|
| `GET /api/web/info` | `{"oxinfer": true, "native": true, "version", "auth_required", "admin", "api_base": "/v1"}` |
| `GET /api/web/status` | platform, cores, memory, GPUs, memory budget, the cache's `{path, free, total}`, `admin`, `listen`, `loopback`, `auth_required`, the models (below) and the downloads |
| `GET /api/web/catalog` | `oxinfer --models --json` (hardware and every tested model with its variants), plus per model `cached`, `cache_bytes`, `servable` and `server_kind`; the CPU kernels are timed once in the background on the first call (`calibrated` false until then, without speed estimates) |
| `GET /api/web/cache` | `{path, free, total, bytes, entries}`: per folder `name`, `repo`, `path`, `bytes`, `files`, `modified`, `last_used`, `ggufs`, `usable`, `specs` (what loads it: `org/name`, `org/name:QUANT` per GGUF file, or the path), `kind`, `catalog` (the catalog entries it holds), `loaded`, `downloading` |
| `POST /api/web/cache/delete` | `{"name"}` (also `DELETE /api/web/cache/{name}`): the folder `org--name` or repo id `org/name`; 409 while in use or downloading, 404 when not in the cache, 400 for a name that is not one plain folder; answers `freed` bytes |
| `GET /api/web/disk` | `{path, free, total}` |
| `POST /api/web/plan` | `{"model"}`: `{repo, dir, total, missing, free, fits, files: [{name, size, have}]}` |
| `GET /api/web/downloads` | the jobs (the last 50) |
| `POST /api/web/downloads` | `{"model"}`: starts a job, or returns the one already running for it |
| `GET /api/web/downloads/{id}` | one job: `status` (`planning`, `downloading`, `done`, `error`, `cancelled`), `done`, `total`, `downloaded` (this run), `speed` (bytes/s over the last seconds), `eta` (s), `file`, `index`, `count`, `files`, `error`, `path` |
| `GET /api/web/downloads/{id}/events` | the job as server-sent events (`data: {...}` every half second) until it ends |
| `POST /api/web/downloads/{id}/cancel` | cancels it (also `DELETE /api/web/downloads/{id}`) |
| `GET /api/web/models` | the models: `id`, `source`, `options`, `kind`, `loaded`, `busy`, `memory`, `state` (`unloaded`, `queued`, `waiting`, `downloading`, `loading`, `ready`, `busy`, `error`), `load` (the stage's details: file and bytes while downloading, the error) |
| `POST /api/web/models/load` | `{"model": SPEC, "id": NAME, "options": {KEY: VALUE} or "K=V,..."}`: lists the model and loads it in the background; `{"id", "load"}` |
| `POST /api/web/models/unload` | `{"id"}`: unloads it now (409 while a request uses it; a load in progress is cancelled). It stays listed and loads again on its next request |

### Security

- **API key.** With `--api-key` (or `$OXINFER_API_KEY`) every endpoint but `/api/web/info`,
  `/health` and the GUI's own files needs `Authorization: Bearer KEY`, compared in constant time.
- **Non-loopback without a key: read-only.** Bound to an address other than loopback
  (`--host 0.0.0.0`, a LAN address) without an API key, `oxinfer web` prints a warning and the
  management API refuses every change with 403 (`"code": "read_only"`): no downloads, deletions,
  loads or unloads, from anyone. Listing the catalog, the cache and the status still works, and so
  do `/v1` and the models listed with `--model`. Set a key to manage models remotely.
- **Cross-site requests.** Requests that change something (POST, DELETE) must be JSON (a plain
  HTML form or a `text/plain` fetch from another site cannot set that type without a CORS
  preflight, which the server does not grant unless `--cors` says so), and when the browser sends
  an `Origin` it must be the page's own. On a loopback listener the `Host` of a management request
  must be a loopback name (`localhost`, `127.0.0.1`, `[::1]`), so a site whose name resolves to
  127.0.0.1 (DNS rebinding) cannot reach it.
- **Deletion stays in the cache.** A name is one folder of the cache directory: names with a path
  separator, `..`, a leading dot or a NUL are refused, the folder's resolved path must sit
  directly in the resolved cache directory, and a symbolic link in the cache is removed as a link,
  never followed. Folders of loaded models (or models in use) and of running downloads are refused.
- The GUI's files are served with `X-Content-Type-Options: nosniff` and `Referrer-Policy:
  no-referrer`; the `/v1` API behaves exactly as with `serve` (`--local-media` and
  `--no-remote-media` included).

## The cache from the command line and the libraries

```sh
oxinfer models list            # folders, size, last use, what loads each, the disk's free space
oxinfer models list --json
oxinfer models rm org/name     # asks first; --yes to skip; only folders inside the cache
```

In Rust: `oxinfer::model_cache::{list, entry_at, locate, remove, disk_space, to_json, touch}`,
`oxinfer::hf::plan` (what a download takes) and the server's `ServerOptions::web` (key `web=1`),
`Api::add_model`, `Api::unload`. In C (`include/oxinfer.h`): `oxi_cache_list`,
`oxi_cache_remove`, `oxi_download_plan`, and `web=1` in the options of `oxi_server_new` serves the
GUI and the management API from an embedding application.

"Last used" is when oxinfer last loaded the model: each load writes the time into a small
`.oxinfer-last-used` file in the model's cache folder (not counted in its size).

## Building

The GUI is `web/` (Vue 3 + Vite). `npm run build:native` in `web/` writes it to
`src/server/webui/` with relative paths and without the wasm module; that folder is committed,
and `build.rs` embeds whatever it holds (`include_bytes!`), so building oxinfer needs no node.
Rebuild it after changing `web/src/` (`npm run build` still builds the GitHub Pages site into
`web/dist/`). Without the folder the binary serves the APIs only and says so at start.

## Not done yet

- The video page and route were tested with a tiny random MiniMax-H3 release (the real one is
  141 GB of weights, not downloaded here): the pipeline's numerics are the library's, checked
  against the reference as docs/video.md says, but no real video was made through the server.
- No preview images while an image denoises (`partial_images` is accepted, none is sent): a
  preview costs a VAE decode per step; the pipelines report steps only.
- The video is AVI (Motion JPEG and PCM), not MP4: no pure-Rust H.264 / VP9 / AV1 encoder is
  available to oxinfer, so the page plays it itself and other players need ffmpeg-based ones.

