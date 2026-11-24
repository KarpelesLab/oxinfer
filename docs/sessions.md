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
compressor state cannot be cut back) a whole snapshot the prompt starts with. It stores each conversation after its reply (left pending so the reply is not
delayed: the server writes it on another thread once the response is sent; the next request, an
explicit `Engine::flush_prompt_cache` or the engine's drop also write it) and, for the recurrent
models, the prefix every conversation with the same system prompt and tools shares (the first
turn rendered with an empty message, tokenized, compared with the prompt) and the prompt but its
last token (a request sent again, a reply regenerated), the first time each is computed (the
prompt is then fed in two or three parts). For the models without recurrent state, a stored
conversation replaces the snapshots of its own prefixes. In memory, `Engine::chat` now also keeps the common prefix with the previous
conversation for these models (it used to start over unless the new conversation extended the
old one).

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
[inkling.md](inkling.md)). Qwen3.5 / 3.6 / 3.8 do not have them yet (their recurrent states can
live on the GPU between passes); with `--prompt-cache` they restore the stored prompt but its
last token from disk.

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
(`deepseek41_gpu.rs`). Not with a DSpark draft (V4's draft keeps its own state; snapshots and
checkpoints are refused there).

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
or MTP drafts and the Gemma 4 assistant drafter, Inkling-2 (Small and Mini, without a draft; on
the CPU and on Metal, whose session buffers are read in place, the KV caches in their own f32 /
f16 / q8 format: `src/inkling.rs`, `tests` `snapshot_round_trip_*`; not on CUDA yet), DeepSeek
V4 / V4.1 and ERNIE 5.5 (without a draft; CPU and Metal decode: `src/deepseek4.rs` tests
`checkpoint_cut_back_matches_fresh`, `snapshot_round_trip`, `deepseek41_gpu.rs`
`gpu_checkpoint_cut_back_matches_fresh`, `gpu_snapshot_round_trip`, `tests/server.rs`
`deepseek_chat_reuses_the_cache_after_thinking`, on tiny random models from
`src/deepseek4_tiny.rs`). Not (yet): Mistral Large 4, Qwen3.8-Flash-Next (Qwen4-Exp: n-gram and indexer state), Core ML
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
