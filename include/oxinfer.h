/*
 * oxinfer C API (cargo feature `capi`): a thin layer over the Rust `oxinfer::engine` module, for
 * Swift and other non-Rust hosts. Rust programs should use the crate directly.
 *
 * Build a static library:  cargo build-capi --target aarch64-apple-ios   (or -ios-sim, -darwin)
 *                          -> target/<target>/release/liboxinfer.a
 *
 * Conventions
 *   - Strings are NUL-terminated UTF-8.
 *   - Functions returning int use < 0 for errors; functions returning pointers return NULL.
 *     The error message is available from oxi_last_error() on the same thread.
 *   - `char *` results are owned by the caller: free them with oxi_string_free().
 *   - No Rust panic crosses this boundary (a panic becomes an error return).
 *
 * Threads
 *   An engine may be used from any thread, one call at a time (calls on one engine are
 *   serialized by an internal lock). oxi_engine_cancel() may be called from any thread at any
 *   time. The CPU thread pool and the GPU backend are process-wide; generations of different
 *   engines run one at a time.
 */
#ifndef OXINFER_H
#define OXINFER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OxiEngine OxiEngine;

/* Streamed reply text: `len` bytes at `text` (not NUL-terminated), `thinking` non-zero inside a
 * thinking block. Return non-zero to stop generation. */
typedef int (*oxi_token_fn)(void *user, const char *text, size_t len, int thinking);

/* Download progress for `file` (index of count; index 0 for metadata files): `done` of `total`
 * bytes (`total` 0 when unknown). Return non-zero to cancel; a later call resumes the file. */
typedef int (*oxi_progress_fn)(void *user, const char *file, uint64_t done, uint64_t total, int index, int count);

/* Library version (static string). */
const char *oxi_version(void);

/* Last error message on this thread (never NULL; valid until the next failing call). */
const char *oxi_last_error(void);

/* Frees a string returned by this library (NULL is ignored). */
void oxi_string_free(char *s);

/* Detected CPU features and GPUs, as text. */
char *oxi_devices(void);

/* Downloads a model into the cache (or finds it there) and returns its local path:
 * "org/name" (safetensors repo) or "org/name:QUANT" (one GGUF file of a repo, e.g. Q4_K_M).
 * `cache_dir` NULL: $OXINFER_CACHE, else ~/.cache/oxinfer/hub (iOS: Library/Application
 * Support/oxinfer/hub in the app container). `progress` may be NULL. */
char *oxi_download(const char *spec, const char *cache_dir, oxi_progress_fn progress, void *user);

/* Loads a model: a directory, a .gguf file or a repo id (downloaded first, see oxi_download).
 * `options`: NULL or "key=value" pairs separated by newlines or ';':
 *   threads=N            CPU threads (default: all cores; process-wide)
 *   quant=bf16|q8|q4     precision for checkpoints that are not already quantized (default bf16)
 *   mmap=0|1             map weights instead of reading them into memory (default 1 on iOS)
 *   preload=0|1          with mmap, touch all weight pages while loading
 *   device=auto|cpu|metal  compute device for prefill matmuls (process-wide, first engine wins)
 *   prefill_chunk=N      tokens per prefill chunk (default 256 on CPU, 1024 on GPU)
 *   max_context=N        conversation token limit (default: none)
 *   mtp=0|1              Qwen3.5/3.8 multi-token prediction speculative decoding
 *   mtp_from=PATH|REPO   MTP weights from another checkpoint (implies mtp=1)
 *   mtp_fixed=0|1        always draft draft_tokens (default adaptive)
 *   draft=PATH|REPO      Gemma 4 assistant (drafter) model for speculative decoding (greedy)
 *   draft_tokens=N       max draft tokens per round (default 3 with MTP, 6 with a drafter)
 *   cache_dir=PATH       where repo ids are downloaded
 *   gpu_exact=0|1        GPU matmuls bit-identical to the CPU (default 0: M5/A19 matrix
 *                        units, half-precision inputs; process-wide)
 *   gpu_moe=0|1          MoE experts on the GPU in prefill (default 1; process-wide)
 *   coreml_units=auto|ane|all|gpu|cpu   Core ML models (ANEMLL format, Apple only): compute
 *                        units (default auto: the Neural Engine, or cpu/gpu following device)
 *   coreml_prefill=auto|batch|pad|single  Core ML models: how prompts are fed (default auto)
 * Returns NULL on error. */
OxiEngine *oxi_engine_load(const char *model, const char *options, oxi_progress_fn progress, void *user);

/* Frees an engine (NULL is ignored); no other call on it may be running. */
void oxi_engine_free(OxiEngine *engine);

/* Conversation and sampling settings:
 *   system       system prompt; NULL value = default prompt (date and time), "" = none
 *   think        0|1 thinking mode
 *   max_tokens   max tokens per reply (default 1024)
 *   temperature  0 = greedy (default)
 *   top_k, top_p, seed
 * system and think shape the start of a conversation: set them before the first message or
 * call oxi_engine_reset(). Returns 0 or -1. */
int oxi_engine_set(OxiEngine *engine, const char *key, const char *value);

/* Sends a user message and streams the reply to `cb` (may be NULL). Blocks until the reply ends.
 * Returns 0 (the model ended its turn), 1 (max_tokens / max_context reached), 2 (cancelled by
 * the callback or oxi_engine_cancel) or -1 (error). */
int oxi_engine_send(OxiEngine *engine, const char *text, oxi_token_fn cb, void *user);

/* Feeds raw text (no chat template; appended to the current context) and streams the
 * continuation. Returns as oxi_engine_send. */
int oxi_engine_complete(OxiEngine *engine, const char *text, oxi_token_fn cb, void *user);

/* Starts a new conversation. Returns 0 or -1. */
int oxi_engine_reset(OxiEngine *engine);

/* Stops the running reply after the current token (any thread). */
void oxi_engine_cancel(const OxiEngine *engine);

/* Model information as a JSON object (path, model_type, family, layers, quant, format, device,
 * threads, speculative, load_secs, coreml (null, or units, prefill, context_length, ...), ...). */
char *oxi_engine_info_json(OxiEngine *engine);

/* Statistics of the last turn as a JSON object: prompt_tokens, prefill_secs, prefill_tok_s,
 * generated_tokens, decode_secs, decode_tok_s, forwards, drafted, accepted, acceptance,
 * context_tokens. */
char *oxi_engine_stats_json(OxiEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* OXINFER_H */
