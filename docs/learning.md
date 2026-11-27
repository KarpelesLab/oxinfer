# Learning from outcomes: profiles, episodes and adapters

A model's reply is produced now; whether it was any good is often known later (the tests ran, the
build broke, the user kept the edit). The server records both, per **profile**, and runs each
profile with the LoRA adapter trained from its records:

1. a chat request for `MODEL@PROFILE` is logged as an **episode**: the tokens the model read and
   the tokens it wrote, under the response's `id`;
2. a later request reports the **outcome** of that response (`POST /v1/episodes/{id}/outcome`);
3. an offline trainer turns the profile's episodes and outcomes into a LoRA adapter, written
   into the profile's directory. It is not part of the server and is not described here;
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
