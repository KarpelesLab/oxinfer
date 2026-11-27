# Learning from outcomes: profiles, episodes and adapters

A model's reply is produced now; whether it was any good is often known later (the tests ran, the
build broke, the user kept the edit). The server records both, per **profile**, and runs each
profile with the LoRA adapter trained from its records:

1. a chat request for `MODEL@PROFILE` is logged as an **episode**: the tokens the model read and
   the tokens it wrote, under the response's `id`;
2. a later request reports the **outcome** of that response (`POST /v1/episodes/{id}/outcome`),
   or of a conversation's responses since its last outcome (`POST /v1/outcomes` with `session`);
3. an offline trainer turns the profile's episodes and outcomes into a LoRA adapter, written
   into the profile's directory. It is not part of the server: `oxinfer lora-train`
   ([Training a LoRA from episodes](#training-a-lora-from-episodes));
4. the server attaches that adapter to the loaded model for the profile's following requests,
   picks up a newer one when the trainer writes it, and drops it when it is deleted.

A request for plain `MODEL` is untouched: no adapter, no log. This page says what the server
does and what was checked of it. It makes no claim about what training on these records gains.

```sh
oxinfer serve --model qwen=Qwen/Qwen3.5-0.8B          # profiles live in ~/.cache/oxinfer/learn
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen@myrepo", "messages": [{"role": "user", "content": "Fix the failing test."}]}'
# {"id": "chatcmpl-1a21c98f7fe0001", "model": "qwen@myrepo", ...}
curl -s localhost:8080/v1/episodes/chatcmpl-1a21c98f7fe0001/outcome \
  -H 'Content-Type: application/json' -d '{"result": "bad", "kind": "tests", "note": "2 of 14 failed"}'
```

## Profiles

`model` in a chat request (`/v1/chat/completions`, `/v1/messages`, streamed or not) may be
`MODEL@PROFILE`: `MODEL` is a model's id, `PROFILE` is 1 to 64 characters of `A-Z a-z 0-9 _ . -`,
not starting with a dot (400 otherwise). A listed id is always itself, whatever it holds; other
names are split at their last `@`. The response's `model` echoes the name as requested.
`/v1/messages/count_tokens` takes the name too and counts for `MODEL`. `/v1/completions` and
spoken replies do not take profiles.

A profile exists once a request naming it has been answered (its directory is created with its
first episode). Nothing else creates one, and all the profiles of a model share its loaded
weights: the profile only chooses the adapter of a request.

Profiles are served on a loopback listener or with `--api-key` only (403 otherwise): they write
conversations to disk and load adapter files. `--learn-dir DIR` (`learn_dir`) sets where they
live; the default is `learn` beside the default prompt cache directory
(`~/.cache/oxinfer/learn`). `--learn-dir none` turns them off: `MODEL@PROFILE` is then an unknown
model (404), as are the endpoints below.

## The profile directory

```
<learn dir>/<model>/<profile>/
    episodes.jsonl     appended by the server, one line per logged reply
    outcomes.jsonl     appended by the server, one line per reported outcome and response
    adapter/           a PEFT folder, written by the trainer; absent until the first training
```

`<model>` is the model's id with every byte other than `A-Z a-z 0-9 _ . -` written as `%XX` (two
upper-case hex digits), a leading dot too: `Qwen/Qwen3-0.6B` is `Qwen%2FQwen3-0.6B`, `qwen` is
`qwen`. Two ids never share a directory. `<profile>` is the profile's name as it is. The profile
directory is created with mode 0700 at its first episode. Both logs are UTF-8, one JSON object
per line, each line written with one `write` on a file opened for appending; the server never
rewrites or truncates them. A reader should skip a last line that does not parse (a crash while
writing).

### `episodes.jsonl`

```json
{"id":"chatcmpl-1a21c98f7fe0001","time":1795774391,"model":"qwen","profile":"myrepo","session":"ses_de197b012ffetVhAiCG3Em4a7l","adapters":[],"prompt_tokens":[248045,846,...],"completion_tokens":[760,5642,...],"finish_reason":"stop"}
```

| Field | |
|---|---|
| `id` | the response's `id` (`chatcmpl-...`, `msg_...`), as the client received it; with `n` > 1, the first choice's is the `id` and the others' are `id.1`, `id.2`, ... |
| `time` | Unix seconds when the reply ended |
| `model`, `profile` | the model's id and the profile |
| `session` | only when the request named its conversation: the `X-Session-Id` header (else `X-Opencode-Session-Id`), 1 to 128 printable ASCII characters without spaces (another value is ignored). Absent otherwise. It is the client's name for the conversation, nothing the server checks: what an outcome with `session` selects by |
| `adapters` | the adapter the reply was sampled with: `[]` for the base model, else `[{"path": ".../adapter", "strength": 1.0, "version": "..."}]`. `version` identifies the adapter's files (see below): two episodes with the same `version` were sampled with the same adapter |
| `prompt_tokens` | the token ids the model read before the reply: the whole rendered chat template (system prompt, tools, every earlier turn), the part taken from a cache included |
| `completion_tokens` | the token ids the model wrote, in order: thinking, tool calls and the stop token that ended the reply included |
| `logprobs` | only when the request asked for `logprobs`: one number per token of `completion_tokens`, the natural-log probability the API's `logprobs` reports for the tokens it shows. Absent otherwise |
| `finish_reason` | `stop`, `length`, `tool_calls` as in the response, or `cancelled` when the client left or the request was cancelled before the end (the tokens written until then are logged) |

`prompt_tokens` followed by `completion_tokens` is the sequence a trainer replays. Its length is
the response's `usage.prompt_tokens`; `completion_tokens` has one token more than
`usage.completion_tokens` when a stop token ended the reply. A request that fails before a reply
(an error status) logs nothing. With a `stop` sequence, the tokens are those sampled until the
sequence was seen. Attachments are logged as their placeholder tokens only (the image or audio is
not in the log; no test covers a profile request with attachments).

### `outcomes.jsonl`

```json
{"id":"chatcmpl-1a21c98f7fe0001","time":1795774502,"reward":-1.0,"kind":"tests","note":"2 of 14 failed"}
```

`id` is an episode's id, `time` the Unix seconds of the report, `reward` a number from -1 to 1,
`kind` and `note` the report's strings or `null`. All five keys are always there. Outcomes are
only appended: an episode may have several lines, and the last one is the one that counts (it is
the one `GET /v1/episodes` shows). An outcome that named several responses is one line per
response, with the same `time`. An episode without a line has no known outcome.

### `adapter/`

A PEFT folder: `adapter_config.json` and `adapter_model.safetensors` ([LoRA adapters](lora.md)
has the modules each family takes). The trainer should write a new folder beside it in the
profile directory and rename it into place.

Before each request of a profile, the server looks at the size and modification time of the two
files (`version` is those four numbers):

- a folder that is not attached, or whose files changed since it was, is attached at strength 1
  (the earlier one of the profile goes): the requests in flight on the model end first, with their
  adapter as it was, and the requests arriving meanwhile wait;
- no folder (or one without both files): the profile's adapter, if one was attached, is detached
  and the base model answers;
- an adapter that does not load or does not fit the model is reported on stderr once, and the
  base model answers (the episode's `adapters` is `[]`). It is tried again when its files change.

No restart and no reload of the base weights is involved. What attaching costs and where it
works is in [LoRA adapters](lora.md#adapters-on-a-loaded-model): Qwen-path, Gemma and Inkling-2
models decoded without MTP or a DSpark draft, not streamed from disk. On other models the
profile's requests are logged and answered by the base model, with the stderr line above.

`DELETE /v1/models/{id}/profiles/{profile}/adapter` removes the folder and keeps the logs: the
way to drop an adapter that does not help.

## Requests at the same time

An adapter's switch belongs to the model, not to a session. So the requests a model runs at one
time (`--parallel`) all use the same adapter, or none: requests of one profile run together, as
do requests without an adapter (plain `MODEL` and profiles that have none), and a request that
needs another adapter waits until the running ones have ended. Requests that arrive while one
waits queue behind it, so a busy profile does not starve the others. A session that last ran
with another adapter starts its conversation over, and the prompt cache keeps the snapshots of
each adapter version apart.

## Endpoints

| Endpoint | |
|---|---|
| `POST /v1/episodes/{id}/outcome` | reports the outcome of response `id` |
| `POST /v1/outcomes` | reports one outcome for several responses, or a batch of outcomes |
| `GET /v1/episodes` | the episodes: `id`, `time`, `model`, `profile`, `session` (null without one), `adapter` (sampled with one), `prompt_tokens` and `completion_tokens` (counts), `finish_reason`, `outcomes` (how many) and `outcome` (the last: `time`, `reward`, `kind`, `note`). Newest first; `order=asc`, `limit` (default 50, up to 1000), `after=ID`, `model=`, `profile=`, `session=`. `has_more` and `last_id` page through them |
| `GET /v1/episodes/{id}` | one of them |
| `GET /v1/models/{id}/profiles` | the model's profiles: `id` (`MODEL@PROFILE`), `path` (the profile directory), `episodes`, `outcomes`, `adapter` (`{"version"}` or null) |
| `DELETE /v1/models/{id}/profiles/{profile}/adapter` | removes the profile's adapter folder |

`GET /v1/models` lists each model's profiles as `"profiles": ["myrepo", ...]` (where profiles
are served: a loopback listener or one with a key).

An outcome is a JSON object with `reward` (a number from -1 to 1) or `result` (`"good"`: 1,
`"bad"`: -1), and optionally `kind` (what judged it: `tests`, `build`, `lint`, `user_accepted`,
`user_rejected`, any string) and `note` (free text). `POST /v1/outcomes` also takes the
responses it is about: `id`, `ids` (an array), `session`, or several of them. Its body is one
such object, or `{"outcomes": [...]}` with several. The reply lists what was recorded
(`recorded`, `data`). An id that is not an episode is a 404 (`episode_not_found`) and nothing of
the request is recorded.

`"session": NAME` stands for the episodes logged with that `session` after the last one of them
that has an outcome (any outcome, however it was reported): "the replies of this conversation
since the previous verdict", for a harness that cannot see response ids. That may be none: the
request is then answered with `"recorded": 0`, as it is for a session the server never saw (no
404). In a batch, a session's episodes are judged by the first outcome that names it.
Ids stay valid across restarts: the server reads the logs under the learn directory at its first
use of them after a start.

These endpoints follow the [management API's rules](web.md#management-api): the API key when one
is set, a loopback `Host` on a loopback listener; changes (`POST`, `DELETE`) take
`Content-Type: application/json`, are refused from another site's page, and are refused (403,
`read_only`) on a non-loopback listener without a key.

## For agent harnesses

A harness (a custom loop) only has to keep the `id` of each response and say later how things
went. The id is in every chat completion (`id`), in every chunk of a streamed one, and in
`message_start` (`message.id`) of a streamed `/v1/messages` reply. A harness that cannot keep
ids sends the same `X-Session-Id` header with every request of a conversation and reports with
`"session"` ([opencode](#opencode) is one).

```sh
# 1. Talk to MODEL@PROFILE instead of MODEL. Keep the ids.
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen@myrepo", "messages": [...], "tools": [...]}' | jq -r .id

# 2. The tests ran: one outcome for every response since the last test run.
curl -s localhost:8080/v1/outcomes -H 'Content-Type: application/json' -d '{
  "ids": ["chatcmpl-1a21c98f7fe0001", "chatcmpl-1a21c98f8960002"],
  "result": "bad", "kind": "tests", "note": "test_parse_date failed: expected 2024-02-29"}'

# 3. Or a whole run at once.
curl -s localhost:8080/v1/outcomes -H 'Content-Type: application/json' -d '{"outcomes": [
  {"ids": ["chatcmpl-...1", "chatcmpl-...2"], "result": "bad", "kind": "tests"},
  {"id": "chatcmpl-...3", "result": "good", "kind": "tests"},
  {"id": "chatcmpl-...3", "reward": 0.5, "kind": "user_accepted", "note": "kept after an edit"}]}'
```

The report is a plain HTTP request, so it can be a tool of the harness itself (a `report_outcome`
tool whose handler posts to `/v1/outcomes`), a hook after the test command, or a line in a
script. The server runs no tools for this.

What to report:

- results something checked: the test suite's status, the build's, a linter's, a user accepting
  or rejecting the change. Say which in `kind`;
- one outcome per judged span: when a test run judges the five responses since the previous run,
  report it once with their five ids. Not one per token or per tool call;
- the later verdict too when it changes: outcomes are appended and the last one counts.

What not to report: the model's own opinion of its reply, an outcome for responses the check did
not cover, or a guess when the check did not run. An episode without an outcome is simply
unjudged. (A `report_outcome` tool the model calls is for passing on what something else
judged, such as the user's verdict.)

## opencode

opencode (1.19.6 is what was looked at) gives a plugin no response id: the hooks of its plugin
API (`chat.params`, `chat.headers`, `tool.execute.before` / `.after`, `event`, ...) and the
events it publishes (`message.updated`, `message.part.updated`, `session.*`, `permission.*`)
were logged during a run against this server, and none of them held a `chatcmpl-...` id. It
does send the session's id with every request to an OpenAI-compatible provider, plugin or not
(`X-Session-Id`, `x-opencode-session-id` and `x-session-affinity`, as a listener that logs
headers received them, with `"model": "qwen@e2e"` as configured), which is what `session`
above is for.

So the opencode side is two things, both written by the server (**with a profile…** beside
**Add to opencode** in the GUI, or the API: [docs/server.md](server.md#opencode)):

```sh
curl -s -X POST -H 'Content-Type: application/json' -d '{"model": "qwen", "profile": "myrepo"}' \
  localhost:8080/api/web/opencode/add
opencode run -m oxinfer/qwen@myrepo "make the tests pass"     # or pick it with /models
```

1. the model's entry under `qwen@myrepo` in opencode's configuration: opencode sends the id as
   it is, so the session's requests are episodes of profile `myrepo`;
2. the plugin `~/.config/opencode/plugins/oxinfer-outcomes.js` (the file is
   `src/opencode/oxinfer-outcomes.js` of this repository: copying it there by hand does the
   same). In a session whose model is `MODEL@PROFILE` of the `oxinfer` provider, and in no
   other:
   - when opencode's `bash` tool finishes a command it recognises as a test, build or lint run,
     it posts `{"session", "result", "kind", "note"}` to the provider's `baseURL` + `/outcomes`
     with the provider's `apiKey`: `good` for exit status 0, `bad` otherwise, `kind` `tests`,
     `build` or `lint`, `note` the status and the command;
   - the model gets a `report_outcome` tool (`result`, `kind`, `note`) for a result no pattern
     recognises: a check run by other means, the user saying the change is right or wrong. Its
     description tells the model not to report its own opinion; nothing enforces that.

The API writes the configuration of the machine the server runs on only. With opencode on
another machine, do the two by hand there: copy the model's entry in `opencode.json` under the
key `ID@PROFILE`, and copy the plugin's file into `~/.config/opencode/plugins/` (it imports
`@opencode-ai/plugin`, which opencode installs in its configuration directory). The server
needs an API key then (profiles are not served without one off loopback), which the plugin
takes from the provider's `options.apiKey`. This by-hand path was not run.

What counts as a check is deliberately narrow, since a wrong verdict trains the wrong way:

- the command must start with a known test, build or lint command (`cargo test`, `go test`,
  `pytest`, `npm test`, `make test`, `cargo build`, `make`, `tsc`, `cargo clippy`, `eslint`, ...:
  the lists are at the top of the file), after environment assignments and `timeout N`;
- a command with a pipe, `;`, `||`, `&`, `$(...)`, backticks or several lines is ignored (its
  exit status is not the check's: `cargo test | tail` always succeeds); `2>&1` and `> file` are
  fine; so are `--help`, `--no-run`, `--list`, `--collect-only`, `--dry-run` ignored;
- `a && b`: status 0 is `good` (every part passed); a failure is reported only when the check is
  the one part beside `cd` / `export` (`which make && make test` failing says nothing);
- no exit status (a timeout), 126, 127 (not found) and 128 or more (killed) report nothing;
- the same command with the same status as the session's previous check, with no other tool
  call in between, reports nothing: a model that runs its tests in a loop is not rewarded once
  per run;
- the replies opencode asks for on the side (the session's title, summaries: its `title`,
  `summary` and `compaction` agents) are sent under another session name (`SESSION.title`) and
  so are never judged.

Settings, all optional, in `~/.config/opencode/oxinfer-outcomes.json`:

```json
{ "providers": ["oxinfer"], "tool": true,
  "patterns": { "tests": ["\\./run-tests\\.sh", "cargo test"], "e2e": ["make e2e"], "lint": [] },
  "skip": ["--help"] }
```

`providers`: the provider ids whose `MODEL@PROFILE` sessions report; `tool`: false leaves the
`report_outcome` tool out; `patterns`: per kind, regular expressions matched at the start of a
command (a kind listed replaces its defaults, `[]` turns it off, a new name is a new `kind`);
`skip`: commands matching one are ignored. The tool is offered in every opencode session once
the plugin is installed, whatever the model (in another session it answers that nothing was
recorded); `"tool": false` is the way to keep it out of them.

Limits of this path: a verdict is for every reply of the session since the last one, the one
that asked for the command included, whether or not the command exercised what they wrote; a
subagent (opencode's `task` tool) is another session, which a verdict in the parent does not
reach; and a reply cut off by the user is judged with the rest.

What was checked, with opencode 1.19.6 on Linux against `oxinfer serve` with an API key,
Qwen3.5-0.8B, the configuration, data and state of opencode in temporary directories
(`XDG_*`), `opencode run -m oxinfer/qwen@PROFILE`:

- the entry and the plugin as the API writes them were loaded by opencode, and the requests
  arrived as `qwen@PROFILE` with the session's id: every episode has its `session`, the title
  request the `.title` one (and no outcome);
- `make test` exiting 0: one outcome line per episode since the last, `"kind":"tests"`,
  `"reward":1.0`, `"note":"exit 0: make test"`, in the profile's `outcomes.jsonl`; the model then
  ran the same command again and again and no further outcome was recorded until it had run
  something else (before that rule, a first run of 713 replies had 713 good outcomes);
- a failing target: the model handed the work to a subagent, whose `cd DIR && make test`
  exited 2: `"reward":-1.0`, `"kind":"tests"`, `"note":"exit 2: cd DIR && make test"` for the
  10 replies of the subagent's session until then, none for the parent session's;
- the model calling `report_outcome` (`good`, `user_accepted`, a note): one outcome line for
  the reply since the last verdict.

The classification of commands and exit statuses is tested on its own
(`node tools/opencode-oxinfer/test.mjs`). Not checked with opencode: the settings file, a provider
without an API key, opencode's TUI and desktop app (only `opencode run`), older opencode
versions (the plugin sets `X-Session-Id` itself for those that do not send it), and opencode 2,
whose plugin API is another one (`Plugin.define`; its documentation says a 1.x plugin does not
run there): the plugin is not ported to it. Nothing was trained from episodes logged this way.

## What was checked

`tests/server.rs` (`outcomes_for_a_session`), tiny random Qwen3.5 checkpoint, CPU: the `session`
of episodes from both APIs, streamed or not, under either header; an outcome for a session
(its unjudged replies, then none, then the later ones), in a batch, and the listing by session.

`tests/server.rs` (`profiles_log_episodes_take_outcomes_and_switch_adapters`), on the tiny
random Qwen3.5 checkpoint, CPU: the episodes of both APIs, streamed or not, with the ids the
clients got; the token counts against `usage`; `logprobs` when asked for; the outcome endpoints
(shorthand, several ids, batches, unknown ids recording nothing); the listing and its paging; the
logs read again after a restart; the adapter written into the profile directory, replaced,
broken and deleted while the model stays loaded, each compared with the reply of a server loaded
with that adapter (`lora=`) or with none; a second profile's adapter beside it; profiles off, and
refused on a listener that is neither loopback nor keyed. The unit tests of `src/server/learn.rs`
check the two log lines byte for byte.

By hand, Qwen3.5-0.8B (bf16, `mtp=0`, `--parallel 2`) on a Linux machine with the CUDA backend
active, greedy replies of 48 tokens with two random rank-4 adapters: `qwen@p` without an adapter
and `qwen` gave the same text; with each adapter in place `qwen@p` gave the text of a server
loaded with it, and `qwen` the base text, also when both were requested at the same moment; after
the delete, the base text again.

Inkling-2: the pool test on the tiny looped checkpoint and a served Inkling-2-Mini with a
trained adapter are under [Inkling-2](#inkling-2).

Not checked: Gemma, Metal, profile requests with attachments, and many profiles or long logs
(the index of episodes is held in memory, one small record per episode).

## Training a LoRA from episodes

`oxinfer lora-train` is the offline step: it reads a profile's two logs, trains a LoRA adapter on
the episodes that have an outcome, and writes `adapter/` only when the adapter does better than
what the profile runs now on episodes it did not train on. It is a separate process: the server
keeps answering and picks the new folder up at the profile's next request.

```sh
oxinfer lora-train --model Qwen/Qwen3-0.6B --profile ~/.cache/oxinfer/learn/qwen/myrepo
oxinfer lora-train --model M --profile DIR --rank 16 --targets q_proj,k_proj,v_proj,o_proj \
  --kind-weight tests=1,user_rejected=0.3 --steps 200
```

`--model` is the base model the profile's model id stands for, loaded as the server loads it
(the same checkpoint and `--quant`: the loss is computed on those weights). The report goes to
standard output as JSON (the losses, the gate's verdict, seconds per step), the progress to
standard error.

### What trains

The dense full-attention models of the Qwen path: **Qwen3** (per-head query / key norms) and
**Llama 3** (none), from safetensors or GGUF, float or quantized. Every other model is refused
with a message saying what is in the way: Qwen3.5 / 3.8 (Gated DeltaNet layers, gated
attention), Qwen2 (attention biases), MoE models, Granite, LFM3, MiniMax; Gemma is another model
class the trainer does not load. The server attaches adapters to more models than this trains.
**Inkling-2** (`inkling2`, `inkling2_loop`: Inkling-2-Mini) trains too, through a backward pass
of its own behind the same command, objective, gate and adapter folder: see
[Inkling-2](#inkling-2) for what differs (the targets, the memory, the speed).

There is no autograd in oxinfer: `src/lora_train.rs` keeps what the backward needs of a forward
pass and has the backward written out by hand (LM head, final norm, and per layer the SiLU MLP,
the norms, causal attention with RoPE and grouped key / value heads). Only the adapter's
matrices get gradients, in PEFT's convention `W x + (alpha / r) B A x` with `A` uniform in
`+-1 / sqrt(n_in)` and `B` zero at the start. `--targets` picks among `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (default `q_proj,v_proj`), `--layers`
the layers (default all; the backward stops at the lowest one).

Precision: f32 on the CPU, no GPU. The forward runs the frozen matmuls through the inference
kernels on the weights as loaded, so the loss is the served model's. The backward needs
`W^T g`, which no kernel computes, so every frozen weight from the lowest adapted layer up is
transposed once at start: bf16 weights as bf16 (the same values), every quantized format as the
f32 values it stores. For bf16 checkpoints the gradient is that of the loss. For quantized ones
it is a straight-through estimate: the forward kernels also round activations to int8, the
backward treats each matmul as the linear map of its dequantized weights.

Memory: the model, plus the transposes (2 bytes a weight for bf16 checkpoints, 4 for the others,
embedding excluded, LM head included), plus the activations of one episode at a time (per
adapted layer and token about `8 hidden + 3 intermediate + 3 heads x head_dim` floats). The
attention backward computes the scores again instead of keeping them. An episode is one
sequence; `--batch` episodes are run one after the other and their gradients summed.
`--max-tokens` (default 2048) leaves longer episodes out.

### The objective

An episode's weight is its last outcome's `reward` (clamped to -1..1) times the weight of the
outcome's `kind` (`--kind-weight tests=1,user_praise=0.3`; 1 for kinds not listed and for no
kind). Episodes without an outcome, with weight 0 or without completion tokens are left out.
The loss is over the completion tokens only, each given the prompt and the completion before it,
with `p` the model's probability of the logged token:

- weight > 0: `-ln p`, the usual likelihood (imitate what worked);
- weight < 0: `-ln(1 - p)`, which pushes the token down and vanishes, with its gradient, once
  the token is unlikely. Its gradient is that of `ln p` times the odds `p / (1 - p)`, unbounded
  for a token the model is sure of, so past odds of `--neg-clip` (default 4, `p` = 0.8) the loss
  continues along its tangent in `ln p`: the factor stays at the cap.

An episode's loss is the mean over its tokens; a batch's is `sum |w| loss / sum |w|`. Minimizing
`+ln p` for penalized episodes was not used because it has no floor (it keeps pushing tokens
that are already at zero), and a ratio against the logged `logprobs` was not used because they
are usually absent and belong to whatever adapter was attached then (`adapters`), not to the
model being trained. Every token of a penalized reply is pushed down alike, those that were fine
included: the outcome says nothing about which tokens were at fault.

The optimizer is AdamW as PyTorch's (`--lr` 2e-4, `--weight-decay` 0, betas 0.9 / 0.999) with
gradient-norm clipping (`--clip` 1), for `--steps` steps or `--epochs` (3) passes over the
training episodes, `--batch` (4) episodes a step, in an order shuffled from `--seed`.

### The gate, and updating an adapter

`--holdout` (default 0.2) of the episodes are held out: those whose id hashes into the first
fifth of the range, so an episode held out once stays held out as the log grows, and an adapter
trained earlier never saw what a later run measures it on (when none hashes there, the one of
lowest hash is held out; with a single episode there is none and nothing is written without
`--force`). After training, the held-out loss is computed for the base model, for the existing
adapter if there is one, and for the new adapter. The new one is written only when its loss is
lower than the existing adapter's, or than the base model's when the profile has none.
`--force` writes it anyway.

With an `adapter/` already there, training starts from its matrices when it has the rank, alpha,
targets and layers asked for; otherwise from a new adapter (the report's `existing_note` says
why; `--fresh` asks for that), still compared with the existing one. An existing adapter the
trainer cannot read (other modules, rsLoRA, DoRA) is only replaced with `--force`.

Writing: the new folder is written as `adapter.tmp-PID` in the profile directory, the existing
`adapter/` is renamed to `adapter.prev/` (replacing an older one), and the new folder is renamed
to `adapter/`. Between the two renames there is no `adapter/`: a request of the profile arriving
then is answered by the base model. To go back, rename `adapter.prev` to `adapter`.

`--episodes DIR --out DIR` instead of `--profile` reads the two logs from one directory and
keeps the adapter in another.

### What was checked of the trainer

On tiny random checkpoints (3 layers of 64, 4 heads over 2 key / value heads), Qwen3 and Llama 3
each, x86-64 CPU, `cargo test --lib lora_train`:

- the trainer's forward without an adapter gives the model's logits (maximum difference 0);
- gradients against PyTorch autograd (`tools/ref_lora_train.py` with torch 2.15.0, peft 0.22.0
  and transformers 5.24.1 builds the checkpoint, a peft LoRA on all seven targets, one rewarded
  and two penalized episodes with tokens on both sides of the cap; the ignored test
  `lora_train::tests::reference` compares): losses equal to 7 digits; the 42 matrices' gradients
  within 6.5e-7 (Qwen3) and 4.8e-7 (Llama) relative L2 error, the worst matrix 9.9e-7; after
  three clipped AdamW steps with weight decay the parameters differ from peft's by 5e-6 and 6e-5
  of the distance they moved;
- gradients against finite differences (a fourth-order stencil along a random direction per
  matrix, 42 directions, cap on and off): relative error on average 8e-5 (Llama) to 5e-4
  (Qwen3), 2e-2 at worst, which is the resolution of an f32 loss for the smallest derivatives;
- end to end: 24 episodes (12 prompts, each with a rewarded fixed completion and the base
  model's own greedy completion penalized), 40 steps at lr 2e-3, rank 8 on `q_proj` / `v_proj`:
  the adapter is written, loads through the path `--lora` takes (`lora::attach_specs`), and the
  model with it gives the trainer's logits (maximum difference 0); the mean log-probability per
  token of the rewarded completions rose for 12 of 12 (+3.06 Qwen3, +2.40 Llama) and fell for 12
  and 11 of 12 of the penalized ones (-2.27, -3.36), 5 of the 24 episodes being held out;
  held-out loss 4.05 to 2.97 and 4.09 to 3.00. A second run starts from the adapter and is
  compared with it; a run with a negative learning rate is rejected by the gate and leaves the
  folder as it was; another rank starts a new adapter.

On real weights, `bartowski/Llama-3.2-1B-Instruct-GGUF` (16 layers of 2048), 64-thread x86-64
CPU, rank 8 on `q_proj` / `v_proj` (851,968 parameters), a synthetic profile of 40 episodes (20
questions; the model's own answer with reward -1, a short answer in a fixed style with reward 1;
46 to 85 tokens each), `--lr 1e-3 --steps 30`:

| | Q8_0 | Q4_K_M |
|---|---|---|
| transposes (f32) | 4.92 GB, built in 1.2 s | 4.92 GB, 1.4 s |
| 30 steps of 4 episodes | 0.96 s a step | 0.99 s a step |
| peak resident memory | 6.5 GB | 6.0 GB |
| training loss (29 episodes) | 5.70 to 0.07 | 4.51 to 0.02 |
| held-out loss (11 episodes), base to adapter | 7.53 to 0.45 | 6.65 to 0.28 |

(The Q8_0 run had `--kind-weight user_rejected=0.5`.) One episode of 1,023 tokens (Q8_0): 3.3 s
a step, peak 8.9 GB. The Q8_0 adapter then loaded with `oxinfer --model ...Q8_0.gguf --lora
DIR/adapter` (CUDA backend active) and two questions that were not in the profile were answered
in the rewarded style, one of them wrongly. With the adapter attached on the CPU the model's
logits were within 8.7e-5 of the trainer's (logits up to 22, moved by up to 40 by the adapter).
The straight-through gradient of the Q8_0 model against a finite difference of its loss along
the gradient: 1e-4 to 2e-4 for one episode, 9e-3 to 4e-2 depending on the step for another
(`lora_train::tests::real_weights`, ignored).

With the server, by hand: `oxinfer serve --model llama=...Q8_0.gguf --learn-dir DIR`, eight
chat requests for `llama@demo` with outcomes alternately good and bad (`kind` `tests`), then
`oxinfer lora-train --profile DIR/llama/demo` beside the running server. The trainer read the
server's logs (7 training episodes, 1 held out); the gate refused the adapter (held-out loss
0.0003 for the base model, 0.025 with the adapter) and wrote nothing; with `--force` the folder
was written, and the profile's next request was answered with it attached (the server's
`adapter attached` line, the episode's `adapters` naming it).

Not checked: whether an adapter trained this way makes a model better at anything (the numbers
above are losses on synthetic episodes); Qwen3 real weights (none were at hand; only the tiny
checkpoint); `--quant q8` / `q4` of a safetensors checkpoint; aarch64; episodes longer than
1,023 tokens; training while the server reads the folder.

## Inkling-2

`oxinfer lora-train` also trains Inkling-2 (`inkling2`, and `inkling2_loop`: Inkling-2-Mini), from
the release (NVFP4 / MXFP8 experts) or a GGUF file, with the same episodes, objective, optimizer,
gate and adapter folder as above. What differs is the model's side, `src/inkling_train.rs`:

```sh
oxinfer lora-train --model thinkingmachines/Inkling-2-Mini --profile DIR
oxinfer lora-train --model ggml-org/Inkling-2-Mini-GGUF:Q8_0 --profile DIR --rank 16 \
  --targets wo_ud,wq_du --grad-window 512 --max-tokens 8192
```

**What trains.** `--targets` picks among the seven attention projections, by the release's
names: `wq_du`, `wk_dv`, `wv_dv`, `wr_du` (the relative-position vectors), `gate_attn` (the
output gate), `attn_sink_proj` and `wo_ud` (the output projection: the default, and where the
published adapters of this model sit). Everything else is frozen: the experts and their router,
the convolutions, the loop's gates, the embedding and the head. A looped layer has one pair,
used in each of its three passes: its gradient is the sum over them. The folder holds
`base_model.model.model.llm.layers.N.attn.NAME.lora_A / lora_B`, which `--lora` and the server's
profiles read ([LoRA adapters](lora.md)).

**The backward pass** goes through all of the text forward ([Inkling-2](inkling.md)): the final
norm and soft-capped logits; per layer application the short convolutions, the routed experts
with their soft-capped SwiGLU, the attention residual's mixes over the block states, attention
with its sliding window, relative-position bias, sink, clips, log scaling, key / value
convolutions (the per-token taps of the full layers included) and output gate; and through the
loop: the messages read from the previous pass, their gates, and the stream each pass starts
again from. The choice of experts is taken from the forward pass and has no gradient; the
routing weights (the chosen experts' sigmoid scores, renormalized) do, into the router's input,
which is what PyTorch's autograd does on transformers' model. Images are not handled.

**Precision.** float32 on the CPU, and the gradient is that of the loss computed (not a
straight-through estimate): the forward runs the frozen matmuls on the f32 GEMM, which decodes
bf16, NVFP4, MXFP8 and GGUF blocks to their stored values and does not round activations (the
trainer refuses to run with an int8 prefill GEMM selected, or with a GPU backend active), and
the backward multiplies by the same values, 16 rows dequantized at a time: no transposed copy of
any weight is made, and only the experts a sequence routes to are read. The served model
differs from the trained one where its kernels round activations to int8 (decode steps on
quantized weights, and prompts where the int8 prefill GEMM runs): [inkling.md](inkling.md) has
the size of that (KL 2.8e-4 for decode steps on the release).

**Memory.** The forward keeps the stream entering each of the 48 layer applications and the
routing, 453 KB per token for Mini (`(48 + 1) x 2304` floats); the backward then runs each
application again from its saved input to rebuild what its gradient reads (the projections'
outputs, the experts' pre-activations and outputs: by the sizes of those arrays about 0.5 MB per
token and application, so some 23 MB per token if all 48 were kept at once), one application at
a time, and computes the attention scores again rather than keeping them.
Measured on the release (19.5 GB): 19.7 GB peak resident for episodes of 30 tokens, 22.7 GB for
one of 2,048, so about 1.5 MB per token in all. Longer episodes were not run: by these numbers
8k tokens would take about 12 GB over the model and 32k about 48 GB (of which 15 GB kept), and
`--max-tokens` (default 2048) has to be raised for them. The 8 full-attention applications cost
time quadratic in the length.

`--grad-window N` truncates the gradient: the whole episode runs forward, and the backward
treats the positions more than `N` before the completion as constants (their adapter outputs get
no gradient, and nothing is propagated into them). It saves backward time, not memory.

### What was checked of the Inkling-2 trainer

The reference is PyTorch autograd on transformers' `Inkling2ForConditionalGeneration` /
`Inkling2LoopForConditionalGeneration` (5.24.0, torch 2.15.0, float64) with the pairs added as
forward hooks on the seven `nn.Linear`s of each attention (`tools/ref_inkling2_lora_grad.py`,
`examples/inkling2_train.rs check`): the same token ids, completion, adapter (random `A`, and a
random `B` of standard deviation 0.05 so that both factors have gradient) and objective. Tiny
random checkpoints of `tools/inkling2_tiny.py` (12 layers of 64, every text feature on: windows
and relative extents shorter than the sequence, clips that bite, log scaling, sinks, soft caps),
120 tokens of which the last 40 are the completion, rank 2 on all seven projections of every
layer (84 pairs) unless said; the relative error is `|g - g_ref| / |g_ref|` over all gradients
together, and for the worst of the 168 matrices:

| Checkpoint | Episode | Loss, mine and the reference's | Together | Worst matrix |
|---|---|---|---:|---:|
| `inkling2` (no loop), float experts | rewarded | 7.610146 / 7.610143 | 1.4e-4 | 2.3e-4 |
| | penalized | 0.002418 / 0.002418 | 4.8e-5 | 1.3e-4 |
| | rewarded, rank 8 on `wo_ud` only | 7.416416 / 7.416418 | 3.8e-5 | 5.4e-5 |
| `inkling2_loop`, NVFP4 / MXFP8 experts (`--packed`), `QSCALE=0.5 ROUTE=1 SPREAD=1` | rewarded | 7.600107 / 7.600132 | 1.2e-3 | 2.4e-3 |
| | penalized | 0.002334 / 0.002334 | 1.0e-3 | 2.4e-3 |
| | rewarded, rank 8 on `wo_ud` only | 7.212864 / 7.212912 | 1.0e-3 | 1.5e-3 |
| the same with `QSCALE=0.3 ROUTE=0.5 SPREAD=2` | rewarded / penalized / `wo_ud` | within 3.4e-5 | 7.7e-4 / 9.2e-4 / 4.4e-4 | 1.5e-3 / 1.5e-3 / 5.0e-4 |
| `inkling2_loop --packed` as [inkling.md](inkling.md) checks it (`QSCALE=1`) | rewarded / penalized / `wo_ud` | within 1.5e-4 | 3.3e-3 / 4.1e-3 / 5.9e-3 | 9.5e-3 / 8.1e-3 / 7.3e-3 |

The looped rows are as far from the reference as the forward is: the float32 forward of a
random looped model drifts from the float64 one ([inkling.md](inkling.md), "Tiny random
checkpoints"; here the completion's log-probabilities differ by 5e-4 to 6e-3, against 8e-5 at
most without the loop), and the gradients follow. Five more sequences of 80 tokens were run on
the default looped checkpoint with float experts, with an earlier form of the loss (a weighted
sum, every fourth target penalized): four within 2.6e-3 to 6.4e-3 together (one of them with a
single matrix, the last layer's `wr_du`, 66% off), and one 12% off, whose float32 forward has a
log-probability 0.12 from the float64 one. On the first of them transformers' own float32
gradients are 108% from its float64 ones (its loss 122.2 against 116.7): at this size the
reference in float32 is no reference. The penalized rows' tokens are all unlikely, so they do
not reach the cap on the odds; the cap is the dense trainer's code, checked above.

Unit tests (`cargo test --lib inkling::train`, the Rust-only tiny looped checkpoint, x86-64):

- central differences of the loss along random directions, one projection kind and one factor at
  a time (28 directions, for a rewarded and for a penalized sequence, the routing held as the
  gradient holds it): within the 2e-2 that a float32 loss resolves;
- a gradient window as long as the sequence gives the same gradients bit for bit, a shorter one
  the same loss and other gradients;
- the adapter written by the trainer, read back, and attached as `--lora` attaches it
  (`lora::attach_specs`): the inference forward's logits are within 2.2e-5 of the training
  forward's (the adapter moves them by 4.5), and the base model's within 2.8e-5;
- the training loop on six episodes (one penalized): the loss goes down and the folder loads.

On real weights, one DGX Spark (GB10; the training is on its 20 CPU cores, 10 Cortex-X925 and
10 Cortex-A725, the machine shared with another job's runs in turn), rank 8 / alpha 16 on
`wo_ud` of the 24 layers (1,032,192 parameters) unless said:

- **A gradient against transformers.** `tools/ref_inkling2_lora_grad.py --real` runs
  transformers in float32 on `thinkingmachines/Inkling-2-Mini` as stored (the loading of
  `tools/ref_inkling2_mini.py`) on a 45-token chat exchange whose 20-token reply is the
  completion, with a random adapter: loss 2.808228 against the reference's 2.808225, the
  completion's log-probabilities within 1.6e-5, and the gradients of the 48 matrices within
  9.6e-5 together, 1.3e-4 for the worst one. The reference peaked at 81 GB (sampled every 5 s)
  and only fits the 121 GB machine with the experts' float32 weights kept out of autograd's graph
  (`FrozenLinear` in the script recomputes them from the stored bfloat16 in the backward): as
  first written it passed 118 GB and was stopped.
- **Training.** A synthetic profile of 7 episodes (`examples/inkling2_train.rs episodes`: five
  questions answered in a fixed pirate style with reward 1, two plain answers with reward -1;
  19 to 32 tokens each, 109 completion tokens), `--steps 8 --batch 7 --holdout 0 --force --lr
  1e-3`: the training loss went from 3.27 to 0.28 on the release (19.5 GB) and, in 6 steps, from
  3.29 to 0.81 on ggml-org's Q8_0 GGUF (27.8 GB).
- **The adapter through `--lora`.** The folder each run wrote, attached as `--lora` attaches it:
  the inference forward's logits on a 26-token exchange are within 6.7e-5 (release) and 4.5e-5
  (Q8_0) of the training forward's, which the adapter moves by up to 44 and 51; the base model's
  two forwards are within 8.8e-5 and 5.2e-5 (`examples/inkling2_train.rs agree`, the CPU's exact
  prefill GEMM on both sides).
- **Served to a profile.** `oxinfer serve --model "ink=thinkingmachines/Inkling-2-Mini,
  device=cuda,mtp=0,draft=none" --learn-dir DIR` (CUDA backend active), greedy replies to two
  questions that are not in the profile: `ink` and `ink@p` without an adapter answered
  "Tokyo." and "There are 7 days in a week."; with the trained folder copied to
  `DIR/ink/p/adapter`, `ink@p` answered "Ahoy! The capital of Japan be Tokyo, matey." and
  "Ahoy! A week be seven days, matey.", the texts of a server loaded with `lora=` that folder,
  while `ink` kept its two answers; after `DELETE .../profiles/p/adapter`, `ink@p` answered
  "Tokyo." again. The requests took 1.6 to 2.7 s before the attach (after the first), then 1.5
  to 2.0 s for the profile and 4.6 to 5.1 s for the base model; which of them ran on the GPU was
  not looked at (by its code, the CUDA engine leaves a model with attached adapters to the CPU).
  `engine::pool_slots::tests::inkling_adapters_attach_to_a_loaded_model_and_switch_per_request`
  checks the same on the tiny checkpoint, CPU: two adapters side by side, one replaced, one
  detached, each reply equal to that of a model loaded with `lora=` or with none.

Time and memory on that machine, the release, per optimizer step:

| Episodes | Step | Of which | Peak resident |
|---|---:|---|---:|
| 7 of 19 to 32 tokens | 23 s | about 3.3 s an episode | 19.7 GB |
| 1 of 2,048 tokens (248 of completion) | 157 s | forward 34 s; backward 123 s: the steps run again 31 s, the experts' gradient 38 s, the attention's 42 s, the rest 11 s | 22.7 GB |
| the same with `--grad-window 256` (gradient over the last 505 positions) | 84 s | forward 34 s; backward 50 s: the steps run again 31 s, experts 8 s, attention 6 s | 22.2 GB |

(`OXINFER_TRAIN_TIMES=1` prints these per episode. The loss before and after a run costs a
forward per episode more: 34 s for 2,048 tokens, 60 tokens a second.) A short episode costs
seconds whatever its length: every expert it routes to is decoded once per layer application.
The attention's gradient computes every score twice more than the forward does (once from the
queries' side, once from the keys', so that no two tasks write the same gradient), and the
transposed products are plain loops over dequantized rows; neither was tuned.

Not checked or not done:

- **No GPU.** Training is CPU only. The CUDA engine ([inkling.md](inkling.md), "The CUDA
  engine") has no LoRA side paths, so it cannot even run the adapted forward; a backward there
  would need transposed GEMMs over bf16 and over the experts' NVFP4 / MXFP8 / GGUF blocks grouped
  by expert, kernels for the gradients of attention (windowed, with the relative bias and the
  sink), of the mixes, gates, convolutions and norms, and the routing kept from the forward. A
  forward on the GB10 with the backward on the CPU would save the 34 s of the first forward and
  no more, since the backward runs every application again on the CPU, and the device's
  arithmetic is not the CPU's.
- Episodes longer than 2,048 tokens; the memory figures for them above are extrapolations.
- Targets other than `wo_ud` on real weights (all seven are checked on the tiny checkpoints),
  Inkling-2-Small, GGUF files other than Q8_0, the held-out gate on real weights (the runs above
  had no held-out episodes and `--force`), and whether an adapter trained this way makes the model
  better at anything: the profile above is synthetic and its loss is a training loss.
- Adapters of other shapes: the trainer reads and continues only folders whose modules are these
  seven projections with one rank (the published adapters for this model were not tried as a
  starting point).
- Attaching to a loaded model on Metal, and with the MTP draft on.
