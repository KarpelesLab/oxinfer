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
| `--no-update-check` | do not look for newer oxinfer releases (see [Updates](#updates)) |

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
| **Models** | this computer (memory, the memory budget, one line per accelerator: the CPU with its SIMD features, each GPU with its memory, unified or not, and whether it runs prefill and decode, or that there is none, and the Neural Engine on Apple systems), the cache's disk (free and total), the models in use with their load state, the downloads, the [catalog](cli.md#what-runs-on-this-machine) grouped by kind (chat, vision-language, images, speech to text, text to speech, embeddings, video) with the download size, the memory of the recommended precision, whether it fits this machine, the accelerator it uses here (`GPU (Metal) prefill + decode`, `GPU (CUDA) prefill, CPU decode`, `CPU`, ...) and its estimated speed, whether it is downloaded and the notes; any other model by its spec; the downloaded models with size, kind, last use, `catalog` / `in use` / `downloading` / `incomplete` marks, *Load…* and *Delete* (after a confirmation; refused while the model is loaded or downloading) |
| **Chat** | streamed replies, thinking on or off (shown apart, folded once the answer starts), images, audio and video attached to a message for the models that read them (vision checkpoints, Gemma 4's audio, Unlimited-OCR-2's page images), the page's tools (`get_datetime`, `run_javascript` in a sandboxed worker) through `tools` and `tool_calls`, temperature, top-p, the longest reply, the system prompt, the speed of the last reply, *Stop* (closing the request stops the generation) |
| **Image** | the Image mode of the wasm site on the server's image models (Stable Diffusion, SD 3 / 3.5, FLUX.1, Qwen-Image, OmniGen2, Step1X-Edit, BAGEL, Janus-Pro, MiniMax-H3-Image, MOSS-Image / Kromoss): text to image, the catalog's presets (`kromoss-dmd2`, ...: a base model with LoRAs and settings) offered for loading next to the models, and with an input image the edit endpoint (instruction edits, image to image, a mask for inpainting), size, steps and guidance (empty: the pipeline's own), seed, what to leave out; the request is streamed (`stream: true`, [image progress](server.md#image-progress)), so the page says what it waits for (another request, the download, the load) and then *Step N of M*; *Save as PNG* |
| **Transcribe** | a file (any format the server decodes, video included) or a recording from the microphone (made into a 16 kHz WAV in the page), transcription or translation to English, language, plain text, text with segment and word timings, SRT or WebVTT, temperature, beam size, a prompt; copy or download the text |
| **Speak** | text to speech (the speech models, and LFM3-Omni's) with a voice picker filled from [`/v1/audio/voices`](server.md#spoken-replies-lfm3-omni-haru) (the model is loaded to list them; ZONOS3's with their language, a text field for pretrained Orpheus checkpoints, which take any name), temperature, top-p, repetition penalty and seed; each clip with a player and a *Download WAV* link |
| **Voice chat** | the models that answer in speech (LFM3-Omni, Haru): a recording from the microphone or an audio file (made into a 24 kHz WAV in the page), and for LFM3-Omni a typed question too; the system prompt (Haru: its character card), Haru's voice reference clip and longest answer, the seed; each turn with your recording, the model's transcript, a player and *Download WAV* (`/v1/chat/completions` with `modalities: ["text", "audio"]`) |
| **Video** | MiniMax-H3 video jobs (`/v1/videos`): the prompt, seconds, size, steps, seed, first and last frames, references (Ref2VA); the job's progress (its stage, *Step N of M*, the time so far) followed as server-sent events, *Cancel*; the jobs of the server listed (they run on without the page: leave and come back); the file format (MP4: H.264 + AAC, the default; WebM: VP9 + Opus; AVI: Motion JPEG + PCM); a finished video played in a `<video>` element with its soundtrack, *Download MP4* (or WebM / AVI) and *Download WAV*, *Delete* |
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
3. *Load*: the precision (for a catalog checkpoint, each variant with its memory, the accelerator it runs on
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

## Updates

When a newer oxinfer release is out, a line under the header says so (*oxinfer 0.2.0 is
available*), with a link to its release notes. The Models view shows the running version under
*This computer*.

- **The check.** The first time the page asks (`GET /api/web/update`), the server asks GitHub for
  the latest release in the background (the page gets `"state": "checking"` and asks again a few
  seconds later), then answers from memory: at most one request every 6 hours (30 minutes after
  a failure). The request is the anonymous `GET
  https://api.github.com/repos/KarpelesLab/oxinfer/releases/latest` that `oxinfer --check-update`
  makes: no token, no cookie, no identifier, nothing about this machine or its models. The answer
  is kept in the cache directory (`update-release.json`), so restarting oxinfer does not ask
  again. `--no-update-check` (`update_check=0`) turns it off: no request is made, nothing is shown,
  and the GUI cannot update.
- **Update and restart.** When the server may change things (the management API's rule:
  loopback, or an API key) and the release has a binary for this platform, the line has an
  *Update and restart* button. The server downloads the release's archive, replaces its own
  executable (as `oxinfer --update` does: the new file is written beside it and swapped in by a
  rename, so a failed update leaves the old one), then starts the new one with the same arguments
  on the same address and port, without opening another browser tab. On Linux and macOS it
  `exec`s it (the same process, in the same terminal; the listening socket closes with the exec
  and the new process binds it again); on Windows it starts the new process and exits (the old
  executable stays as a hidden `oxinfer.exe.old` until the next start deletes it). The page shows
  the progress, waits for the server to answer with the new version and reloads itself. The
  update is refused (409) while a download, a model load or a request is running; a request that
  starts during the download delays the restart until it ends (two minutes at most). Loaded models
  are not kept: those given with `--model` are listed again and load on use, those loaded from the
  GUI must be loaded again.
- **When it cannot.** With a read-only management API, without a binary for this platform, when
  the executable's directory is not writable (a package manager's install), or when the server
  runs inside another program (the Rust library, the C API's `oxi_server_new` with `web=1`: that
  program's executable is not oxinfer's and is never replaced), the line shows the command to run
  instead: `oxinfer --update`. If the restart itself fails, the status says the new version is
  installed and oxinfer must be restarted by hand.

`update_feed=FILE_OR_URL` (or `$OXINFER_UPDATE_FEED`, also read by `--check-update` and
`--update`) replaces GitHub's API with a JSON file or URL of the same shape (`tag_name`,
`html_url`, `assets` with `name`, `browser_download_url`, `size`): for tests and mirrors. The
answer of such a feed is not kept in the cache directory.

## Management API

Under `/api/web/`, apart from `/v1`. Every endpoint but `info` needs the API key when one is set.

| Endpoint | |
|---|---|
| `GET /api/web/info` | `{"oxinfer": true, "native": true, "version", "auth_required", "admin", "api_base": "/v1"}` |
| `GET /api/web/status` | platform, cores, memory, GPUs, `accelerators` (as in `--models --json`: per accelerator `backend`, `name`, `available`, `used`, `features`, `memory_gb`, `budget_gb`, `unified_memory`, `prefill`, `decode`, `status`), memory budget, the cache's `{path, free, total}`, `admin`, `listen`, `loopback`, `auth_required`, the models (below) and the downloads |
| `GET /api/web/catalog` | `oxinfer --models --json` (hardware and every tested model with its variants), plus per model `cached`, `cache_bytes`, `servable` and `server_kind`, and `presets` (the catalog's presets with `base_cached` and `listed`; also `GET /api/web/presets`; `plan` and `downloads` take a preset id and fetch its base model); the CPU kernels are timed once in the background on the first call (`calibrated` false until then, without speed estimates) |
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
| `GET /api/web/update` | `{enabled, state, current, latest, newer, url, checked, checking, error, can_update, why_not, command, job}`: `state` is `checking`, `available`, `up_to_date`, `error`, `unknown`, `disabled` (`--no-update-check`) or `unsupported` (a build without the `update` feature); `url` the release notes; `checked` the Unix time of the answer; `can_update` whether `POST` would install it, else `why_not` and `command` (`oxinfer --update`); `job` the update started from the GUI, if any |
| `POST /api/web/update` | installs the latest release and restarts into it (see [Updates](#updates)); answers the job: `state` (`checking`, `downloading`, `restarting`, `updated`: installed, restart by hand; `up_to_date`, `error`), `version`, `file`, `size`, `message`, `error`, `started`. 403 when read-only (`read_only`) or turned off (`update_check_off`), 409 while busy (`busy`) or when this process cannot replace itself (`cannot_update`); a job already running (or done, waiting for a restart) is returned as it is |

### Security

- **API key.** With `--api-key` (or `$OXINFER_API_KEY`) every endpoint but `/api/web/info`,
  `/health` and the GUI's own files needs `Authorization: Bearer KEY`, compared in constant time.
- **Non-loopback without a key: read-only.** Bound to an address other than loopback
  (`--host 0.0.0.0`, a LAN address) without an API key, `oxinfer web` prints a warning and the
  management API refuses every change with 403 (`"code": "read_only"`): no downloads, deletions,
  loads, unloads or updates, from anyone. Listing the catalog, the cache and the status still works, and so
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
GUI and the management API from an embedding application (`update_check=0` there turns the
release check off; such a server never replaces an executable). `oxinfer::server::web::allow_self_update`
is what lets `oxinfer web` replace and restart itself. The engine and the C API make no update
check of their own.

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

