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
 *   decision_from=PATH|REPO  typed decisions (oxi_decide) with the decision assets of this Jev
 *                        bundle (LoRA adapter, head, configs; only those small files are
 *                        downloaded) on top of the loaded base model, which must match it. A Jev
 *                        bundle loaded as the model brings its own. ORG/NAME/SUBDIR takes the
 *                        bundle in a subfolder of a repo (autotrust/JEV-9B/vl: JEV-9B's image
 *                        decision assets, on the multimodal Qwen/Qwen3.5-9B).
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

/* oxi_engine_send with attachments placed before the text (builds with the `media` feature).
 * `media_json` (NULL = none) is a JSON object, every key optional:
 *   {"images": ["a.png", ...],     PNG / JPEG / WebP / GIF / BMP
 *    "videos": ["clip.mp4", ...],  MP4 / MOV / MKV / WebM / AVI with H.264, H.265, AV1, VP8, VP9,
 *                                  MPEG-4 Part 2 or MJPEG, or a directory of frame images
 *    "audios": ["a.wav", ...],     WAV / MP3 / FLAC / AAC / Ogg (Gemma 4 only)
 *    "video_fps": 2.0,             frames sampled per second (default: the model's processor;
 *                                  Qwen 2, Gemma 4 samples 32 frames uniformly)
 *    "video_max_frames": 32,       most frames sampled (Gemma 4: the frame count)
 *    "frames_fps": 24.0,           frame rate of a directory of frames (default 24)
 *    "max_soft_tokens": 280}       image token budget (Gemma 4: 70/140/280/560/1120 soft tokens;
 *                                  Qwen: tokens of 32x32 pixels; default: the model's)
 * Gemma 4 takes images, videos and audio; Qwen3.5 / 3.6 / 3.8 checkpoints that include the
 * vision tower take images and videos. Attachments belong to this message only. Returns as
 * oxi_engine_send. */
int oxi_engine_send_media(OxiEngine *engine, const char *text, const char *media_json, oxi_token_fn cb, void *user);

/* Feeds raw text (no chat template; appended to the current context) and streams the
 * continuation. Returns as oxi_engine_send. */
int oxi_engine_complete(OxiEngine *engine, const char *text, oxi_token_fn cb, void *user);

/* Starts a new conversation. Returns 0 or -1. */
int oxi_engine_reset(OxiEngine *engine);

/* Stops the running reply after the current token (any thread). */
void oxi_engine_cancel(const OxiEngine *engine);

/* Model information as a JSON object (path, model_type, family, layers, quant, format, device,
 * threads, speculative, load_secs, coreml (null, or units, prefill, context_length, ...),
 * decision (null, or as oxi_decide_info_json), ...). */
char *oxi_engine_info_json(OxiEngine *engine);

/* Statistics of the last turn as a JSON object: prompt_tokens, prefill_secs, prefill_tok_s,
 * generated_tokens, decode_secs, decode_tok_s, forwards, drafted, accepted, acceptance,
 * context_tokens. */
char *oxi_engine_stats_json(OxiEngine *engine);

/* A typed decision ("System 1" of the Jev models): one prefill pass, a calibrated probability per
 * option, no generation. The conversation is not disturbed. `request_json` is a /v1/decide body:
 *   {"kind": "noul" | "score" | "choice",
 *    "state": "text" or a JSON value (serialized as Python's json.dumps),
 *    "question": "...",
 *    "options": ["...", ...],            (choice only: 2-256 options; 16 trained, more read
 *                                         through extra single-token labels)
 *    "attachments": {"images": ["a.png", ...], "max_soft_tokens": N}}   (optional, oxinfer only:
 *                                         image files placed before the state text, and the
 *                                         image token budget; format of oxi_engine_send_media)
 * Images (Qwen3.5-family bases that include the vision tower, e.g. Qwen/Qwen3.5-9B with
 * decision_from=autotrust/JEV-9B/vl, or autotrust/JEV-27B-VL): make "state" a list mixing text
 * and images, each placed where it appears:
 *   "state": ["Photo: ", {"image": "data:image/png;base64,..."}, "\nTask: ..."]
 * An image is {"image": REF} or {"type": "image_url", "image_url": {"url": REF}}, REF a base64
 * data URL, an http(s):// URL (fetched; builds with HTTP support), a file:// URL or a local
 * path; other non-string parts are serialized as JSON. Video and audio are not part of the
 * protocol and are refused.
 * Returns the response as a JSON object (free with oxi_string_free), or NULL on error:
 *   {"kind", "effective_kind", "options", "probabilities", "choice_index", "choice",
 *    "adaptation": "native" | "wide-labels", "protocol", "model",
 *    "usage": {"prompt_tokens", "completion_tokens", "total_tokens"}, "num_model_requests",
 *    "elapsed_seconds"}
 * noul options are ["false", "true"], score options "0".."5". */
char *oxi_decide(OxiEngine *engine, const char *request_json);

/* The loaded decision assets as JSON (protocol, model, readout, adapter, max_options,
 * native_choice_options, temperatures, source), or NULL when the model has none. */
char *oxi_decide_info_json(OxiEngine *engine);

/* ---- Speech to text (Whisper) ------------------------------------------------------------------
 *
 * openai/whisper-{tiny,base,small,medium,large-v3,large-v3-turbo} (and .en) and the distil-whisper models:
 * Hugging Face safetensors checkpoints with model_type "whisper". Audio of any length (30 s
 * windows, moved along by the predicted timestamps past 30 s). The encoder runs on the GPU
 * when one is active; decoding runs on the CPU. A transcriber follows the engine's threading
 * rules: one call at a time, oxi_transcriber_cancel() from any thread. */

typedef struct OxiTranscriber OxiTranscriber;

/* One finished segment as a JSON object (`len` bytes at `segment_json`, not NUL-terminated), the
 * same as the entries of the result's "segments". Called after each 30 s window is decoded.
 * Return non-zero to stop the transcription (the result then has "stopped": true). */
typedef int (*oxi_segment_fn)(void *user, const char *segment_json, size_t len);

/* Loads a Whisper model: a directory or a repo id (downloaded first, see oxi_download).
 * `options`: NULL or "key=value" pairs separated by newlines or ';':
 *   threads=N                  CPU threads (process-wide)
 *   device=auto|cpu|metal      where the encoder runs (process-wide, first model wins)
 *   precision=f32|f16|bf16|q8|q4  weights of both encoder and decoder (default bf16; f32 gives
 *                              transformers' float32 transcripts token for token; f32 and f16 run
 *                              on the CPU only)
 *   encoder=..., decoder=...   the same for one part; quant=q8|q4 is decoder=
 *   mmap=0|1                   map the checkpoint while converting it (default 1)
 *   cache_dir=PATH             where repo ids are downloaded
 *   gpu_exact=0|1              GPU matmuls bit-identical to the CPU (process-wide)
 * Returns NULL on error. */
OxiTranscriber *oxi_transcriber_load(const char *model, const char *options, oxi_progress_fn progress, void *user);

/* Frees a transcriber (NULL is ignored); no other call on it may be running. */
void oxi_transcriber_free(OxiTranscriber *t);

/* Stops the running transcription at the next token (any thread). The call returns the
 * segments finished so far, with "stopped": true. */
void oxi_transcriber_cancel(const OxiTranscriber *t);

/* Model information as JSON: path, d_model, encoder_layers, decoder_layers, n_mels, vocab_size,
 * multilingual, encoder, decoder (precisions), device, threads, load_secs. */
char *oxi_transcriber_info_json(OxiTranscriber *t);

/* Transcription options (`options_json`, NULL or "" for the defaults), a JSON object with any of:
 *   "language": "fr" | "french" | "<|fr|>" | null   spoken language (null: detected from the
 *                                                   first 30 s, or the model's default)
 *   "task": "transcribe" | "translate"             translate = speech to English text
 *   "timestamps": false          segment-level timestamps (always on past 30 s of audio)
 *   "word_timestamps": false     word-level times (implies timestamps)
 *   "temperatures": [0.0]        decoding temperatures tried in order (0 = greedy); the next is
 *                                used when a window fails the thresholds below
 *   "compression_ratio_threshold": null   e.g. 2.4: retry repetitive windows
 *   "logprob_threshold": null             e.g. -1.0: retry unlikely windows
 *   "no_speech_threshold": null           e.g. 0.6: skip silent windows (needs logprob_threshold)
 *   "condition_on_prev_tokens": false     feed each window the previous text
 *   "initial_prompt": null       context text (names, spelling, style)
 *   "prefix": null               text the transcript starts with
 *   "suppress_tokens": null      token ids never generated (null: the model's list)
 *   "suppress_blank": true       no blank / end of text as the first token
 *   "max_new_tokens": null       per window
 *   "seed": 0                    sampling seed (temperatures above 0)
 * openai-whisper's robust defaults: {"temperatures": [0, 0.2, 0.4, 0.6, 0.8, 1],
 *   "compression_ratio_threshold": 2.4, "logprob_threshold": -1, "no_speech_threshold": 0.6}.
 *
 * The result is a JSON object (free with oxi_string_free), or NULL on error:
 *   {"text", "language" (code, "en" for English-only models), "language_probability" (null when
 *    given), "stopped",
 *    "segments": [{"id", "start", "end" (seconds), "text", "tokens", "temperature",
 *                  "avg_logprob", "compression_ratio", "no_speech_prob",
 *                  "words": [{"word", "start", "end", "probability"}] (word timestamps only)}],
 *    "stats": {"audio_secs", "mel_secs", "encode_secs", "decode_secs", "total_secs", "windows",
 *              "tokens", "rtf"}} */

/* Transcribes `n` mono float samples in [-1, 1] at `sample_rate` Hz (resampled to 16 kHz).
 * `cb` may be NULL. */
char *oxi_transcribe_pcm(OxiTranscriber *t, const float *samples, size_t n, int sample_rate, const char *options_json,
                         oxi_segment_fn cb, void *user);

/* Transcribes an audio file (WAV, MP3, FLAC, AAC/M4A, Ogg Vorbis/Opus, ...; needs the `media`
 * feature, which `cargo build-capi` enables). `cb` may be NULL. */
char *oxi_transcribe_file(OxiTranscriber *t, const char *path, const char *options_json, oxi_segment_fn cb, void *user);

#ifdef __cplusplus
}
#endif

#endif /* OXINFER_H */
