# Session snapshots and the prompt cache

A session (the KV caches, the Gated DeltaNet recurrent and convolution states, Inkling-2's
convolution rings and sliding-window caches, DeepSeek V4's windows, compressed keys and
compressors' rows, positions and vision spans, the DSpark draft's
context cache and pending features, the MTP layer's cache and pending rows, and the tokens) can
be written to a file and read back bit for bit (`snapshot.rs`):

```sh
oxinfer --model Qwen/Qwen3-8B-GGUF:Q8_0 -p "$(cat long_prompt.txt)" --save-session s.oxsnap
oxinfer --model Qwen/Qwen3-8B-GGUF:Q8_0 --load-session s.oxsnap -p "And then?"   # continues it
oxinfer --model Qwen/Qwen3-8B-GGUF:Q8_0 --prompt-cache auto -p "..."            # keyed cache
```

A **prompt cache** (`--prompt-cache DIR|auto`, engine / C API / per-model server key
`prompt_cache`, server-wide `prompt_cache=`) is a directory of such snapshots keyed by model
fingerprint (checkpoint, precision, format, device, speculative decoding and its options) and
token prefix, with a size cap (`prompt_cache_gb`, default 20; least recently used first).
`Engine::chat` restores the snapshot that saves the most of the prompt: for models without
recurrent state (Gemma, Qwen3, Llama, with their drafts) any common prefix (the restored caches
are cut back), for Qwen3.5 / 3.6 / 3.8, Inkling-2 and DeepSeek V4 / V4.1 (recurrent, convolution or
compressor state cannot be cut back) a whole snapshot the prompt starts with. For the models
without recurrent state it stores each conversation after its reply (left pending so the reply is
not delayed: the server writes it on another thread once the response is sent; the next request,
an explicit `Engine::flush_prompt_cache` or the engine's drop also write it), which replaces the
snapshots of its own prefixes. For the recurrent models it stores the prefix every conversation
with the same system prompt and tools shares (the first turn rendered with an empty message,
tokenized, compared with the prompt) and the prompt but its last token (a request sent again, a
reply regenerated, the next turn), the first time each is computed (the prompt is then fed in two
or three parts); not the conversation after the reply, which a next prompt only starts with when
its client sends the reply back as the model wrote it, thinking included (most do not), and which
cost a whole snapshot each turn (9.4 GB at 136k tokens for Inkling-2-Mini with f32 caches). A
prefix found stored already is marked as used, so that the shared one outlives the
conversations' larger snapshots in the least-recently-used order. The cap holds for the
directory: before evicting, a store reads the directory again, so the snapshots that other
engines (a server's other models) or processes wrote count too. In memory, `Engine::chat` now also keeps the common prefix with the previous
conversation for these models (it used to start over unless the new conversation extended the
old one).

**Attachments.** Placeholder tokens are the same for every image, so a cache (or a snapshot)
whose tokens match a prompt's may hold other pictures there. Each soft token is identified by a
hash of its attachment's embeddings and its row in them (`snapshot::MediaMarks`,
`mm::soft_token_marks`), kept beside the cache's tokens and in the snapshots' footers (`media`);
`Engine::chat` reuses the cache, a checkpoint or a snapshot only as far as the soft tokens in it
are the prompt's, and feeds the rest with the prompt's own (the parts fed before a checkpoint
too). Qwen models, whose images take M-RoPE position tables of their own in the session, still
start over when the reused part holds any. Before, any soft token in the reused part made the
request start over: an agent whose tool results show pictures (opencode attaches them as a user
message after the tool's) had its whole conversation from the first picture on prefilled again
at every request. On the user's session (Inkling-2-Mini, Metal, `--prompt-cache`), from the
first of six pictures at position 31586, each request reused only the 9967-token system prefix
and prefilled the other ~28k tokens again (60-80 s a request).

Measured with unsloth's Inkling-2-Mini UD-IQ3_XXS (mmproj BF16 beside it) on Metal through the
server, `--prompt-cache`: an agent's conversation (a 900-token system prompt and tools; each turn
a tool call, a ~8.4k-token tool result and, every other turn, a 320x240 picture of 48 soft
tokens), prompt tokens fed per request:

| turn (prompt tokens) | 2 (17826) | 3 (26261) | 4 (34758) | 5 (43193) | 6 (51690) | 7 (60125) |
|---|---|---|---|---|---|---|
| before (reused: the 9328 tokens before the first picture) | 8496 | 16933 | 25430 | 33865 | 42362 | 50797 |
| after (reused: the previous prompt) | 8496 | 8434 | 8496 | 8434 | 8496 | 8434 |
| after, a new server process (from the prompt cache) | 1 | 1 | 1 | 1 | 1 | 1 |

(The GPU was shared with another server prefilling 30k-token prompts, so times are not
compared; before, the last request took 269 s.) The towers' embeddings are computed again at each
request (0.1-0.2 s an image here), as the identities are those of the embeddings.
Tests: `engine::tests::inkling_chat_reuses_the_cache_past_images` (a tiny random Inkling-2 with
a vision tower, `inkling::tiny::write_tiny_vision`: reuse in memory and from the disk, the replies
those of a fresh engine, another picture in place of one not reused),
`tests/server.rs` `inkling_agent_conversation_with_images_reuses_the_cache` (an opencode-like
16k-token conversation with pictures, then a restart), `snapshot::tests::media_marks_agree`,
`cache_media_and_shared_directory`.

**Inkling-2 in memory: prompt checkpoints.** A chat client sends each turn the conversation with
the last reply as returned; the template renders past replies without their thinking, so the
tokens differ from the cache's right after the previous prompt. Models without recurrent state
cut their caches back there; Qwen3.5's recurrent states and Inkling-2's convolutions and
sliding windows cannot (Inkling-2 goes back 64 positions at most), and the whole conversation
was prefilled again each turn (`cached_tokens` 0 on a 7222-token conversation, 10 s a turn).
For Inkling-2, `Engine::chat` keeps checkpoints of the shared prefix (as above) and of each
prompt but its last token (the convolutions' last inputs and the sliding windows' last 1024
positions; the full-attention caches keep everything anyway), with or without a prompt cache,
and the next conversation goes back to the latest one within its common prefix with the cache:
a next turn reuses the whole previous prompt, a regenerated reply all of its prompt but the
last token. `prompt_checkpoints` (engine / C API / per-model server key; default 2, 0 off)
caps them (each about 650 MB for Inkling-2-Mini with f32 caches, 180 MB with q8; see
[inkling.md](inkling.md)). Qwen3.5 / 3.6 / 3.8 take them too (below).

**DeepSeek V4 / V4.1: the same checkpoints.** Every layer keeps a sliding window of raw keys and
the compressors keep the rows of the windows still pooling, both trimmed as the session goes on
(64 positions of slack, so speculative rounds can roll back); the compressed keys and the engram
ids only grow. A checkpoint (`DeepSeek4Session::checkpoint`, `reachable`, `cut_to`) holds, per
layer, the window's raw keys and the compressors' pending rows as they were at its position (the
compressed keys are cut back to their count then): about 22 MB for V4.1-Flash, so the default
two cost little. `Engine::chat` takes them as for Inkling-2 (`prompt_checkpoints`, default 2, 0
off), and a next turn whose past reply lost its thinking goes back to the previous prompt but its
last token. Snapshots (`--save-session`, `--prompt-cache`) hold the host's session, which stays
authoritative on Metal too (V4.1's GPU decode appends each pass's keys to it); the device's
copies of the caches are rebuilt from it at the first pass after a restore or a cut back
(`deepseek41_gpu.rs`).

**With a draft.** Inkling-2 with its MTP head or DSpark drafter and DeepSeek V4 with its DSpark
draft keep checkpoints and snapshots too (they used to refuse them: the draft keeps its own
state). The draft's state is its context cache (per position and draft layer, keys computed from
that position's target features alone), the features of positions not yet in it, and for
Inkling-2's drafter its head's context rows. A checkpoint (`DsparkDecoder::checkpoint`, beside the
target's, dropped with it) holds what cutting the draft's session back cannot bring back: the
waiting features, Inkling-2's head rows (it keeps the last 512 positions; the head reads the row
before the anchor) and, for a draft whose every layer has a window, its cache's rows from the
window and 256 positions before the checkpoint (Inkling-2-Mini's drafter has a full-attention
layer and V4's draft keeps every position, so their caches are only cut back). That is a few MB
(Inkling-2-Mini's head rows: 512 of 2304 floats, 4.7 MB). `DsparkDecoder::cut_to` goes back to the
target's checkpoint and puts the draft's state back: the decoder then drafts exactly as one that
ran the same tokens (the same draft logits bit for bit, so the same acceptance) and its greedy
output is that of plain decoding. Snapshots hold the draft's cache, waiting features and head
rows beside the target's session; the drafter's device copy of its cache (Inkling-2 on Metal) is
uploaded again from the host's at its next pass. Re-running the draft from the tokens is not an
option (its context comes from the target's features at every position: a whole prefill).

Measured on Inkling-2-Mini (`thinkingmachines/Inkling-2-Mini` with its MTP head, `mtp=on`, M5 Max,
Metal decode with the drafter on the GPU, other jobs on the machine at load 3) with
`examples/chat_turns.rs`: a 2936-token first turn in thinking mode (160 reply tokens), then a
2956-token second turn whose past reply is sent back as text. "Before" is `prompt_checkpoints=0`,
what a draft got until now (it refused checkpoints and snapshots):

| turn 2 | cached | prefill | wall (160 tokens) | drafts accepted (fixed lengths) |
|---|---|---|---|---|
| before (the conversation starts over) | 0 | 5.2-5.8 s | 9.9 s | 98 of 422 |
| after (in memory, back to the previous prompt but its last token) | 2935 | 0.18-0.23 s | 4.1 s | 98 of 422 |
| a new process, `--prompt-cache` (read in 0.09 s) | 2955 | 0.14 s | 4.7 s | |

The replies are the same tokens in all three (and those of plain decoding without the head); with
fixed draft lengths (`draft_fixed=on`) the second turn drafts and accepts exactly as many tokens
as before (with the cost-aware policy the counts follow its timings: 9 of 39 before, 9 of 13
after). The prompt cache held 2.7 GB (f32 caches: four snapshots of 0.71-0.73 GB). The
first turn takes its checkpoints as without a draft (its prompt fed in three parts: 4.7-5.7 s
against 3.5-5.4 s in one, within the machine's noise). No DeepSeek V4 DSpark draft was at hand
(V4.1's is not supported yet); V4's is covered by the tiny models' tests.

**Qwen3.5 / 3.6 / 3.8: the same checkpoints.** The Gated DeltaNet layers' recurrent and
convolution states cannot be cut back at all (a speculative round rewinds within its own pass
only), so before, any mismatch started the conversation over. A checkpoint
(`QwenSession::checkpoint`, `reachable`, `cut_to`) holds every recurrent layer's two states as
they were at its position; the attention layers' KV caches keep every position and are cut
back. Qwen3.5-4B: 50 MB each (24 layers of 32 heads of 128 x 128), Qwen3.8-27B about 150 MB.
On Metal the states live in the session's own page-aligned buffers, which the decode plans (one
token, verification passes, the device-chained greedy decode with its two alternating copies)
read and update in place and leave up to date after every call; a checkpoint copies them from
there, a cut back copies them back into the same buffers (and tells the plans they changed,
`gpu_states_kept`), and the KV caches' device copies sync from the cut back length at the next
pass. With MTP (`MtpDecoder::checkpoint`) a checkpoint also holds the MTP layer's cache length
(cut back) and pending rows. Dense, MoE and the VL checkpoints (prompts with images in the
reused part start over, as for the other models), on the CPU and on Metal; not Qwen4-Exp
(Qwen3.8-Flash-Next). With a DSpark draft too, as above (`DsparkDecoder::checkpoint` over the
session's own; `engine` `drafts_reuse_the_previous_prompt`'s "Qwen3.5 DSpark" case, on a tiny
random draft from `qwen::tiny::write_draft`). `Engine::chat` takes them as for Inkling-2
(`prompt_checkpoints`, default 2, 0 off). Tests on a tiny random Qwen3.5
(`src/qwen35_tiny.rs`, dense or MoE, with or without MTP): `qwen::tiny` tests
`checkpoint_cut_back_matches_fresh`, `mtp_checkpoint_cut_back_matches_fresh` (the CPU, bit for
bit against a fresh session's same passes), `tests/gpu_chain.rs`
`qwen35_checkpoints_on_the_device` (Metal: one-token steps, a chain and MTP verification
passes after the cut back, bit for bit), `engine` `qwen35_chat_reuses_the_previous_prompt`
(turn 2, a regenerated reply, an edited message) and `tests/server.rs`
`qwen35_chat_reuses_the_cache_after_thinking`.

Measured with `examples/chat_turns.rs` (M5 Max, Metal, warm page cache, load 1-5; best and
worst of two interleaved runs): a first turn of about 3000 tokens in thinking mode (128 reply
tokens, all thinking, so the reply sent back is empty and its tokens differ from the cache's),
then a second turn 20 tokens longer:

| turn 2 | cached | prefill | wall (128 tokens) |
|---|---|---|---|
| Qwen3.8-27B (oxinfer q4), before (master, or `prompt_checkpoints=0`) | 0 | 7.9-8.6 s | 15.7-17.3 s |
| Qwen3.8-27B, after (back to the previous prompt but its last token) | 3046 | 0.56-0.62 s | 8.4-8.9 s |
| Qwen3.5-4B (bf16), before | 0 | 0.68 s | 2.17 s |
| Qwen3.5-4B, after | 3008 | 0.17 s | 1.64 s |

The first turn costs the same within the noise (its prompt is fed in three passes: the system
prefix, the rest but its last token, the last token). The replies are the same tokens before and
after on both models.

Measured on DeepSeek-V4.1-Flash-1017 (ISTA-DASLab GSQ-RCO IQ1_M, M5 Max, Metal decode, warm page
cache, other jobs on the machine at load 4-10) with `examples/chat_turns.rs`: a 2131-token
first turn in thinking mode (160 reply tokens, all thinking, so the reply sent back is empty and
its tokens differ from the cache's), then a 2142-token second turn:

| turn 2 | cached | prefill | wall (160 tokens) |
|---|---|---|---|
| before (`prompt_checkpoints=0`: the cache cannot go back that far, the conversation starts over) | 0 | 17.0 s | 22.1 s |
| after (in memory, back to the previous prompt but its last token) | 2130 | 0.70 s | 5.9 s |
| a new process, `--prompt-cache` (a 36 MB snapshot read in 0.01 s) | 2141 | 0.71 s | 4.6 s |

The checkpoints there took 3.2 MB (the 34-token system prefix) and 21-22 MB (2130 and 2141
tokens); the prompt cache held four snapshots of 35-40 MB. The reply restored in a new process is
the in-memory second turn's token for token (the restored session is the original bit for bit);
against the prompt computed in one pass, the greedy replies part at a near tie after a few
tokens, as any prompt fed in other passes does: the restored prompt's last token runs as a
decode step (on the GPU, at KL 3e-5 from the CPU's), and the tokens agree for 3 of 160 decoding on
Metal (both paths pick the same token at each step: KL 1e-4 at the tie) and 60 of 160 on the CPU
(`stream_gpu=off`).

The directory is created private (0700; an existing one is set to 0700, which fails unless it is
yours, so `/tmp` or `/dev/shm` themselves are refused; a directory of your own on a tmpfs gives
memory-speed restores on Linux), files are written 0600 under a temporary name and renamed into
place, symbolic links are not followed, every section length is checked against the model before
anything is allocated, and a checksum is checked before the state is used. A snapshot of another
model or other options is refused; a corrupted one is removed from the cache and the prompt is
computed.

`Engine::clear_prompt_cache` (C: `oxi_engine_clear_prompt_cache`) removes the engine's own
snapshots (its fingerprint's; other models' stay) and drops the conversation waiting to be
stored. The server's status page lists the snapshots per directory and model and clears them
(`POST /api/web/prompt-cache/clear`, `Api::clear_prompt_cache`; see
[the server](server.md#status-cancelling-and-the-caches)).

Families: Gemma 4 / 3, Qwen3, Qwen3.5 / 3.6 / 3.8 (dense, MoE, Qwen2.5-VL), Llama 3, with DSpark
or MTP drafts and the Gemma 4 assistant drafter, Inkling-2 (Small and Mini, with or without the
MTP / DSpark drafter; on
the CPU and on Metal, whose session buffers are read in place, the KV caches in their own f32 /
f16 / q8 format: `src/inkling.rs`, `tests` `snapshot_round_trip_*`; not on CUDA yet), DeepSeek
V4 / V4.1 and ERNIE 5.5 (V4 with or without its DSpark draft; CPU and Metal decode:
`src/deepseek4.rs` tests
`checkpoint_cut_back_matches_fresh`, `snapshot_round_trip`, `deepseek41_gpu.rs`
`gpu_checkpoint_cut_back_matches_fresh`, `gpu_snapshot_round_trip`, `tests/server.rs`
`deepseek_chat_reuses_the_cache_after_thinking`, on tiny random models from
`src/deepseek4_tiny.rs`; with drafts: `inkling.rs` and `deepseek4.rs`
`draft_checkpoint_cut_back_matches_fresh`, `engine.rs` `drafts_reuse_the_previous_prompt`,
`drafts_prompt_cache_restores`, on the tiny drafts of `inkling_tiny.rs` `write_tiny_mtp` and
`deepseek4_tiny.rs` `write_draft`). Not (yet): Mistral Large 4, Qwen3.8-Flash-Next (Qwen4-Exp: n-gram and indexer state), Core ML
models.

Checks (M5 Max, Metal; the machine was shared with other jobs at load 30-95, which slowed the
prefills): `examples/session_check.rs` prefills a ~3000-token prompt, writes the session, reads it
into a fresh session and decodes 32 greedy steps from the original, the restored and a recomputed
session: the restored logits are the original's bit for bit at every step (GPU decoding, whose
device copies of the caches are rebuilt from the restored ones), for Qwen3-8B (2780 tokens, 820 MB
snapshot, written in 0.35 s, read in 0.22 s against a 23 s prefill), Qwen3.5-4B (2909 tokens,
243 MB, 0.06 / 0.07 s against 11.6 s; recurrent states and the attention layers' KV) and
gemma-4-12B-it (3057 tokens, 2.1 GB, 0.53 / 0.31 s against 11.2 s). Through the engine,
`--save-session` then `--load-session` with a second message gives the same reply as one process
holding the conversation (Qwen3-8B with its DSpark draft, Qwen3.5-4B with MTP, gemma-4-12B-it with
DSpark, `--exact-verify`), and `tests/server.rs` (`prompt_cache_across_restarts`) restarts a server
and gets the same reply from a restored prompt. With `--prompt-cache` and the same ~3000-token
prompt twice:

| model | first request: prefill | second request: restore | snapshot |
|---|---|---|---|
| Qwen3.5-4B (bf16, MTP) | 2909 tokens in 9.2 s | 2908 tokens in 0.08 s | 267 MB |
| Qwen3-8B (Q8_0 GGUF, plain) | 2780 tokens in 87 s (load 40) | 2779 tokens in 0.50 s | 824 MB |
| gemma-4-12B-it (Q8_0 GGUF, plain) | 3057 tokens in 60 s (load 37) | 3056 tokens in 0.57-1.9 s | 2.1 GB |

and the same reply both times; a different question after the same document restores the common
prefix on the models without recurrent state (2771 of 2786 tokens for Qwen3-8B). The first decode
step after a restore copies the caches to the GPU, as the first step after a prefill does.
Snapshots hold f32 caches: about 0.3 MB per token for Qwen3-8B, 0.09 MB for Qwen3.5-4B (a quarter
of its layers have a KV cache), 0.7 MB for Gemma 4 12B, whose sliding-window layers keep every
position.

## The default system prompt

The default system prompt (the command line without `--system`, `SystemPrompt::Default` in the
library, a NULL `system` in the C API, `system=default` in a server model spec) says that the model
runs locally with oxinfer, without internet access, and ends with the local date, weekday and
time zone: "The current local date is 2027-01-15 Friday (UTC+09:00, JST)." It does not have the
time of day: the text is the same for every conversation started that day, so a conversation's
next turn finds its cache, and a new conversation the prefix (and the stored snapshots) of the
earlier ones. The time of day is what the built-in `get_current_datetime` tool returns (`--tools`,
`oxinfer::tools::builtin_specs`).

A conversation keeps the prompt of its first turn. `Engine::send` renders it once, at the first
turn. `Engine::chat`, which renders the whole conversation at every call, remembers the prompt of
the 64 latest conversations (of an engine and its forks, so of a server model's sessions),
identified by their first turn (the messages before the first reply): a conversation past its
first turn gets the text its first turn was rendered with, also on a later day, and a
conversation at its first turn the date of the day. The memory is the process's: after a restart,
or once 64 other conversations came, a conversation that goes on on another day is rendered with
the new date, and its prompt computed again.

On two tiny checkpoints on the CPU (`default_system_prompt_keeps_a_conversation_cache` in
`src/engine.rs`), the second turn of a conversation under the default prompt a second after the
first reused 281 of the 281 tokens of the first prompt (Inkling-2) and 282 of 283 (Qwen3.5); with
the time of day in the prompt, it reused none (of 301 and 303).
