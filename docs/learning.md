# Learning from outcomes: profiles, episodes and adapters

A model's reply is produced now; whether it was any good is often known later (the tests ran, the
build broke, the user kept the edit). The server records both, per **profile**, and runs each
profile with the LoRA adapter trained from its records:

1. a chat request for `MODEL@PROFILE` is logged as an **episode**: the tokens the model read and
   the tokens it wrote, under the response's `id`;
2. a later request reports the **outcome** of that response (`POST /v1/episodes/{id}/outcome`);
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
{"id":"chatcmpl-1a21c98f7fe0001","time":1795774391,"model":"qwen","profile":"myrepo","adapters":[],"prompt_tokens":[248045,846,...],"completion_tokens":[760,5642,...],"finish_reason":"stop"}
```

| Field | |
|---|---|
| `id` | the response's `id` (`chatcmpl-...`, `msg_...`), as the client received it; with `n` > 1, the first choice's is the `id` and the others' are `id.1`, `id.2`, ... |
| `time` | Unix seconds when the reply ended |
| `model`, `profile` | the model's id and the profile |
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
works is in [LoRA adapters](lora.md#adapters-on-a-loaded-model): Qwen-path and Gemma models
decoded without MTP or a DSpark draft, not streamed from disk. On other models the profile's
requests are logged and answered by the base model, with the stderr line above.

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
| `GET /v1/episodes` | the episodes: `id`, `time`, `model`, `profile`, `adapter` (sampled with one), `prompt_tokens` and `completion_tokens` (counts), `finish_reason`, `outcomes` (how many) and `outcome` (the last: `time`, `reward`, `kind`, `note`). Newest first; `order=asc`, `limit` (default 50, up to 1000), `after=ID`, `model=`, `profile=`. `has_more` and `last_id` page through them |
| `GET /v1/episodes/{id}` | one of them |
| `GET /v1/models/{id}/profiles` | the model's profiles: `id` (`MODEL@PROFILE`), `path` (the profile directory), `episodes`, `outcomes`, `adapter` (`{"version"}` or null) |
| `DELETE /v1/models/{id}/profiles/{profile}/adapter` | removes the profile's adapter folder |

`GET /v1/models` lists each model's profiles as `"profiles": ["myrepo", ...]` (where profiles
are served: a loopback listener or one with a key).

An outcome is a JSON object with `reward` (a number from -1 to 1) or `result` (`"good"`: 1,
`"bad"`: -1), and optionally `kind` (what judged it: `tests`, `build`, `lint`, `user_accepted`,
`user_rejected`, any string) and `note` (free text). `POST /v1/outcomes` also takes the
responses it is about: `id`, `ids` (an array), or both. Its body is one such object, or
`{"outcomes": [...]}` with several. The reply lists what was recorded (`recorded`, `data`). An
id that is not an episode is a 404 (`episode_not_found`) and nothing of the request is recorded.
Ids stay valid across restarts: the server reads the logs under the learn directory at its first
use of them after a start.

These endpoints follow the [management API's rules](web.md#management-api): the API key when one
is set, a loopback `Host` on a loopback listener; changes (`POST`, `DELETE`) take
`Content-Type: application/json`, are refused from another site's page, and are refused (403,
`read_only`) on a non-loopback listener without a key.

## For agent harnesses

A harness (opencode, a custom loop) only has to keep the `id` of each response and say later how
things went. The id is in every chat completion (`id`), in every chunk of a streamed one, and in
`message_start` (`message.id`) of a streamed `/v1/messages` reply.

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
unjudged.

## What was checked

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

Not checked: any other model family, Metal, profile requests with attachments, and many
profiles or long logs (the index of episodes is held in memory, one small record per episode).

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
attention), Qwen2 (attention biases), MoE models, Granite, LFM3, MiniMax; Gemma and Inkling-2
are other model classes the trainer does not load. The server attaches adapters to more models
than this trains.

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
