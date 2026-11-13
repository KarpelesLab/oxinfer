# Session snapshots and the prompt cache

A session (the KV caches, the Gated DeltaNet recurrent and convolution states, positions and
vision spans, the DSpark draft's context cache and pending features, the MTP layer's cache and
pending rows, and the tokens) can be written to a file and read back bit for bit (`snapshot.rs`):

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
are cut back), for Qwen3.5 / 3.6 / 3.8 (recurrent state cannot be cut back) a whole snapshot the
prompt starts with. It stores each conversation after its reply (left pending so the reply is not
delayed: the server writes it on another thread once the response is sent; the next request, an
explicit `Engine::flush_prompt_cache` or the engine's drop also write it) and, for the recurrent
models, the prefix every conversation with the same system prompt and tools shares (the first
turn rendered with an empty message, tokenized, compared with the prompt) and the prompt but its
last token (a request sent again, a reply regenerated), the first time each is computed (the
prompt is then fed in two or three parts). For the models without recurrent state, a stored
conversation replaces the snapshots of its own prefixes. In memory, `Engine::chat` now also keeps the common prefix with the previous
conversation for these models (it used to start over unless the new conversation extended the
old one).

The directory is created private (0700; an existing one is set to 0700, which fails unless it is
yours, so `/tmp` or `/dev/shm` themselves are refused; a directory of your own on a tmpfs gives
memory-speed restores on Linux), files are written 0600 under a temporary name and renamed into
place, symbolic links are not followed, every section length is checked against the model before
anything is allocated, and a checksum is checked before the state is used. A snapshot of another
model or other options is refused; a corrupted one is removed from the cache and the prompt is
computed.

Families: Gemma 4 / 3, Qwen3, Qwen3.5 / 3.6 / 3.8 (dense, MoE, Qwen2.5-VL), Llama 3, with DSpark
or MTP drafts and the Gemma 4 assistant drafter. Not (yet): Mistral Large 4, DeepSeek V4
(compressed-attention state), Qwen3.8-Flash-Next (Qwen4-Exp: n-gram and indexer state), Core ML
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
