# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.23](https://github.com/KarpelesLab/oxinfer/compare/v0.0.22...v0.0.23) - 2026-11-27

### Added

- *(server)* decode_chunk model option (BatchOptions::decode_chunk, batch::chunk_for): the most prompt tokens a chunk reads while other requests of the model decode (default 0, off: Inkling-2-Mini N=8 mixed 94.3 -> 94.1 tok/s, TTFT p50 2.74 -> 1.66 s, p95 8.66 -> 9.26; Qwen3-8B slightly slower)
- *(server, metal)* mixtures of experts batched exactly by default (Gemma 4 26B-A4B, Qwen3 / Qwen3.5 MoE): a batched decode step runs each row's routed experts with the one-token expert kernels' arithmetic (gate / up pairs side by side, every row's weighted down sums in one dispatch: mvx_*_bsum / mvxt_*_bsum, OXINFER_XB=0 a dispatch per row), rows bit for bit their one-token steps (metal::tests::tiled_experts_match every format and tile geometry, tests/gemma_moe_gpu.rs batched rows and chains, gpu_chain tiny Qwen3.5 MoE; 4 / 8 concurrent Gemma 4 26B-A4B and Qwen3.5-35B-A3B replies equal to alone); decode-bound aggregate Gemma 4 26B-A4B Q4_0 4 / 8 requests 126 / 128 -> 266 / 295 tok/s (llama-server 255 / 358), Qwen3.5-35B-A3B Q4_K_M 122 / 122 -> 219 / 247 (196 / 249); mixed 8: 86 -> 158 (183), 98 -> 185 (170)
- *(longcat-avatar)* stage times, reference tool on released weights
- *(advisor)* LongCat-Video-Avatar in the catalog; server refuses it
- *(longcat-avatar)* LongCat-Video-Avatar 1.6 (audio-driven video)
- *(cuda)* int8 rows and query / key ranges in the H3 kernels
- *(whisper)* every encoder hidden state (encode_hidden_states)
- *(t5)* umT5 encoders (a relative-position bias per block)
- *(server, metal inkling)* batched decode steps for Inkling-2 on Metal: one pass whose rows are the decoding sessions' tokens (DevEngine::run_rows, Inkling2Model::forward_rows, TextLm::forward_batch), the matvecs, expert groups and mixes over all rows, the ops on a session's caches and rings (queries, keys / values into its caches, attention, the convolutions after the attention and the experts) encoded per row on its own state at its own position as its one-token pass encodes them (stages of the rows interleaved between barriers: Pe stage filter, InkMetal::per_row), each row's logits bit for bit its one-token pass's (tests/gpu_chain.rs inkling_batched_rows_match_one_token_passes); a DSpark request hands over to plain batched steps when others decode, from a checkpoint of its turn's start that its next prefill goes back to (DsparkDecoder::plain_turn_checkpointed, Target::cut_back); inkling2_bench -b; Inkling-2-Mini UD-IQ3_XXS 8 decode-bound requests 91 -> 150 tok/s (llama-server 115), 4: 88 -> 126 (86), replies equal to alone (4 / 8 greedy; seeded with host sampling)
- *(server, metal)* batched decode steps for Gated DeltaNet models (Qwen3.5 / 3.6 / 3.8): each row's conv and delta rule on its session's own recurrent states with the one-token kernels (gpu::decode::Row::states), the matvecs on the exact multi-token kernels, rows bit for bit their one-token steps (gpu_chain: tiny Qwen3.5 bf16 / Q8 / Q4; 4 concurrent Qwen3.5-9B BF16 and Qwen3.8-27B Q4_K_P replies equal to alone); decode-bound aggregate Qwen3.5-9B BF16 4 / 8 requests 33 -> 108 / 130 tok/s (llama-server 108 / 190), Qwen3.8-27B 21 -> 34 / 31 (47 / 84); OXINFER_BATCH_TRACE notes plans that refuse a batch
- *(server, engine pool)* session cache memory budget: idle sessions over it let go of their caches, least recently used first (stored in the prompt cache first when one is set, restored from there when the conversation comes back), the most recently used kept; under memory pressure idle sessions' caches go before idle models; Engine::cache_bytes / evict_cache, EnginePool::set_cache_budget / trim / evict / idle_caches, ServerOptions::session_cache (--session-cache-gb, session_cache_gb in the C API; default a quarter of RAM, at least 2 GB; none: no limit), session_cache in /api/web/activity
- *(server)* concurrent requests on one chat model (--parallel N, default 4): a session per request over the same weights (EnginePool, Engine::fork) matched by shared conversation prefix; turns on the device (src/batch.rs): prompt chunks interleaved with the others' decoding (waiting prompts shortest first), plain decode steps of the decoding requests batched (Metal DecodePlan::step_rows on the exact multi-token kernels, each row on its own caches; CPU forward_hidden_batch), speculative decoding (DSpark, prompt lookup, chained passes) kept for a request alone and handed over to batched plain steps when another request decodes (DSpark: the draft catches up at the next prompt); each reply equal to its single-request one (mixtures of experts batched only with batch_fast, whose 5+ row steps use the fast-verification kernels); model options parallel / max_batch / batch_fast, OXINFER_BATCH_TRACE; Qwen3-8B Q8_0 8 mixed prompts: agg 56 -> 83 tok/s, TTFT p50 19.8 -> 2.9 s

### Fixed

- *(ci)* whisper_enc_bench needs the hf + media features (feature-subset clippy), FileBytes' len / heap are read on unix only (Windows dead_code)

### Other

- *(metal inkling)* short prompt chunks: the barriers after the loop's message-gate down projections in prefill passes only where the matrix-unit GEMM shares its input scratch (the same arithmetic, the same logits): Inkling-2-Mini UD-IQ3_XXS chunks of 17 / 32 / 64 / 128 tokens 75.8 / 82.9 / 101.2 / 141.3 -> 67.5 / 74.9 / 94.6 / 136.0 ms; OXINFER_INKLING_F32MV_T=N runs the gates' down projections and the router on the matvecs for chunks of up to N tokens (f32 GEMM grids of 2-4 threadgroups: 128 measured best, 17 / 32 tokens 56.8 / 66.1 ms; off by default, it routes some tokens' experts otherwise: last-logit KL to the host 2.4e-3 against 8.4e-4 over five prompts); passes of up to 16 tokens already cost a verification pass of that size (their cost follows the experts touched); docs
- *(performance)* Gemma 4 26B-A4B decode-bound N=8 below the experts' batching figure is the machine's load (9384888 / dde5e9b / 2739799 alternated: 8-row step 26.3-27.8 ms each)
- *(performance)* concurrent requests tables re-run on master 2739799 (Qwen3-8B Q8_0 / Q4_K_M, Gemma 4 26B-A4B Q4_0, Inkling-2-Mini UD-IQ3_XXS; decode-bound and mixed, N = 1, 2, 4, 8; llama-server in the same window)
- *(metal decode)* exact multi-token matvecs: Q4_K gate / up pairs on two packed rows reduced token by token at 6-8 tokens (12288 x 4096 pair, 8 tokens 622 -> 493 us), LM heads (32768+ rows) on more rows a threadgroup at 6-8 tokens (151936 x 4096: Q6_K 4874 -> 2762 us, Q4_K 4894 -> 2975, Q8_0 2952 -> 2640); Qwen3-8B Q4_K_M batched step at 8 rows 42.6 -> 38.6 ms (1 / 2 / 4 rows and Q8_0 unchanged); the per-token kinds take rows of up to 4 chunks a lane (MvkGeo::c; slower than the kinds keeping a sum per token on long rows, so only by OXINFER_MVK_KIND); every exact matvec adds each chunk's dot product without contraction (mv_acc, mvk_*: the compiler fused it where a loop's shape let it, so kernels of several chunks a lane matched the one-token kernel in some instantiations only); decode_mvk_matches_single_token covers 3072- and 12288-value rows; wider SIMD-group splits for long rows measured and not kept (docs)
- *(metal decode)* one-token attention split finely over the keys (dec_attv_* / dec_attvr_* with the RoPE fused: the lanes across a head's dimensions and the SIMD-groups across the keys, each key one coalesced row load, the first rows loaded before the queries are made, chunks of 32+ keys for ~64 threadgroups a layer, the last of a head's threadgroups merging the chunks in order, no combine stage; up to 4096 keys and G x D <= 32, dec_attn_* past that; multi-token plans' tokens and step rows take the same kernel at the same chunking, so batched rows keep their one-token bits; OXINFER_DEC_ATTV=0 gives the logits of before bit for bit, OXINFER_DEC_ATTV_MAX / _TGS / _CMIN): Gemma 4 26B-A4B attention 11.3 -> 7.5 us a layer, pass 6172 -> 6074 us; tg128 26B-A4B Q4_0 162.6 / 160.4 -> 164.1 / 167.1 tok/s (llama.cpp 130 / 112), E4B Q8_0 96.7 / 97.7 -> 99.8 / 100.1 (84), Qwen3.5-35B-A3B 139.8 / 143.2 -> 141.8 / 145.7 (112 / 100); pf_check 64/64 greedy tokens on 26B-A4B, E4B, Qwen3.5-35B-A3B, Qwen3.8-27B (also after a 10k-token prompt on the 26B-A4B); dec_attv_matches_attn; docs: Metal decode round 4 (the Gated DeltaNet rule on more threadgroups and the conv folded into it measured and dropped: the rule moves 2 MB of f32 state a layer near the memory's rate)
- *(metal decode)* exact multi-token matvecs that scale further: rows of one chunk a lane reduced token by token (MvkKind::OneF/OneH/OneP, mvk_one: no T x NR accumulators; 4096 x 4096 at 8 tokens Q8_0 92 -> 63 us, Q4_K 90 -> 78, Q4_0 80 -> 72), packed Q8_0 / Q4_K quants unpacked at each use (MvkKind::Packed; long Q8_0 rows on four rows 329 -> 298 us), row groups a threadgroup (MvkGeo, MVK_G; Q8_0 gate / up pair 385 -> 352 us), each row still the one-token kernel's bits (decode_mvk_matches_single_token, with every kind / rows / groups forced by OXINFER_MVK_KIND / ROWS / G); Qwen3-8B Q8_0 batched step at 8 rows 40.2 -> 34.4 ms (examples/rows_bench.rs); OXINFER_PF_PROFILE also times a batched step's ops by kind; decode_mvk_bench compiles every variant only for the formats / token counts asked (OXINFER_MVK_FMTS, _T, _NR, _H, _GS), OXINFER_MVK_SX0
- *(server)* concurrent sessions keep their conversations (a new one takes an empty or the least recently used session unless it shares whole messages or 256+ bytes), short prompt chunks on the GPU (one token with the draft's features on the one-token kernels, 2-31 tokens on the prefill plan; OXINFER_PREFILL_GPU_SMALL=0 keeps the host path), OXINFER_BATCH_TRACE prints prompt chunks; Qwen3-8B Q8_0 8 mixed requests 77 -> 91 tok/s, TTFT p50 2.38 -> 0.88 s, p95 10.5 -> 6.1 s
- *(longcat-avatar)* the module docs say which part of ftfy is left out
- *(longcat-avatar)* the avatar video page, model and README rows
- *(longcat-avatar)* float16 operands on CUDA by default
- *(longcat-avatar)* the CUDA stack against the CPU's; --download-only
- *(minimax-h3)* the Video carries its frame rate
- *(metal decode)* Gemma 4 MoE layers route beside the dense MLP (decode graphs emit the router's and experts' input norms and the router after the dense gate / up pair, then its down projection and post norm: two stages fewer a layer, the same ops on the same inputs; OXINFER_DEC_MOE_ORDER=0), the router's top-k in one barrier (each SIMD-group's top k, then SIMD-group 0 picks the k best: the same experts in the same order; OXINFER_DEC_TOPK=0), the GLU after a lone matvec in its stores (Gemma 4 E2B / E4B per-layer embedding gates, one-token and exact multi-token plans; OXINFER_DEC_GLU_MV=0), the Gated DeltaNet rule's loads ahead of its norms' barriers; logits bit-identical (pf_check); tg128 Gemma 4 26B-A4B Q4_0 147.5 -> 155 tok/s (llama.cpp 121-127), Q8_0 107 -> 111, Qwen3.5-35B-A3B 140 -> 142; dec_ab prints token hashes, per-configuration engine options, chain=0|1, OXINFER_AB_PROMPT; OXINFER_DEC_ONLY gdnc / gdnr; docs: Metal decode round 3 (stage profile per model, what was measured and dropped), the 26B-A4B chain against one-token steps (same tokens on the concurrent-requests test's prompts)
- *(sampling, metal inkling, dspark)* top-k / top-p / min-p and the repetition penalty chained on the device (ink_fsamp: candidates bounded by fixed-point histograms of quarter nats below the max, bitonic sort, the host's rank-order cuts and draw; the penalty applied by ink_softcap over a device bit set marked by each pass; the host chooses when over 2048 candidates and the chain goes on from its token), min-p (CLI --min-p, request / model option min_p, C API min_p); same seed, same tokens with OXINFER_GPU_CHAIN=0 (the device still chooses, one pass at a time; OXINFER_DEVICE_SAMPLING=0 keeps the host sampler); Inkling embedding norm in the device's order on the host (greedy text identical with and without speculation, also at 10k context); Inkling's drafter by default on Metal: the DSpark decoder takes prompt-lookup drafts at any match length in plain rounds, probes with at most 3 drafts at most every 16 plain steps per draft round (sooner on a long lookup match), keeps drafting only 12% ahead of plain steps, makes the drafter on the GPU after the first prefill chunk; UD-IQ3_XXS top_p 0.95 82 -> 118 tok/s; server: the default drafter within 0-2.5% of lookup alone on short prompts, +6-26% on a 10k-token code edit
- *(sampling, metal inkling)* sampled decoding at greedy speed: whole-vocabulary draws in token order without sorting (sampler::sample_full; dense distributions in speculative rounds, sampler::Dist / speculative_accept_dist), and Inkling-2 sampling chained on the device (ink_samp from a host uniform per queued pass, decode_sampled / decode_sampled_feats for plain, lookup and DSpark rounds; the generator replayed to the tokens kept); UD-IQ3_XXS at the server's defaults (temp 1, top-p 1) 81 -> 120 tok/s, code edit at 10k context with DSpark 71 -> 130; greedy text unchanged; tiny Metal Inkling checkpoint (inkling::tiny::write_tiny_metal) and gpu_chain test
- *(metal)* one copy of the routed experts: decoding reads the tiled copies too (mvxt_*: a lane a row, 16 rows a SIMD-group, each load one run), the checkpoint's experts let go once a layer has them (pages of a GGUF read into memory mapped back from the file, of a mapped one dropped), copies shared with the per-layer MoE uploads; 26B-A4B Q4_0 peak 31.0 -> 18.1 GB (tiles off 18.2), Q8_0 54.7 -> 30.4, Qwen3.5-35B-A3B 43.8 -> 24.5; decode Q4_0 +0.5%, Q8_0 / Q4_K_M -1%; prefill gain kept; tiles default no longer bound to a quarter of RAM
- *(metal, inkling)* prefill flash attention round 2: queries in the arena, probabilities over the scores (32 queries a threadgroup, mask-free tiles), int8 tiles prefetched into registers, the sliding-window layers on the flash kernel (ring and pass rows); pfm_* the same (pfm_128 32 queries); UD-IQ3_XXS 2048 tokens at 128k f16 907 -> 1146 tok/s, q8 381 -> 827, depth 0 1713 -> 1940; parity_bench pp_depth=D (pp N appended at depth)
- *(metal, inkling)* long contexts: flash attention on the matrix units for the prefill passes' full-attention layers (ink_fa_f / _h in place, ink_fa_q from dequantized q8 tiles; relative bias, sink and gate in the kernel), decoding a threadgroup per key head from 16k positions (the same bits), full-attention caches grown by a quarter, f16 caches by default on Metal (kv_cache=auto); UD-IQ3_XXS, 2048 tokens appended at 128k 86 -> 813 tok/s (f32 -> f16 default; llama.cpp 105), decoding there 17.0 -> 49.8 (llama.cpp 28.5); inkling2_ctx_bench --prefill / --text
- *(whisper, cpu)* encoder self-attention on the SME tiles (scores and weighted values as outer products, softmax on NEON), the NEON prompt-attention kernel without SME, bias / GELU / residual epilogues in the prefill GEMMs' jobs
- *(metal)* routed experts of GGUF checkpoints on tiled copies for the prefill GEMMs (each 64-row band block column by block column, K-quants part by part, so a grouped tile's K step reads one run of memory; bit-identical; 26B-A4B Q4_0 512-token pass 115.4 -> 109.2 ms, Q8_0 136.8 -> 127.6, Qwen3.5-35B-A3B 134.7 -> 126.2; option gpu_expert_tiles)

## [0.0.22](https://github.com/KarpelesLab/oxinfer/compare/v0.0.21...v0.0.22) - 2026-11-26

### Added

- Qwen3.5 / 3.6 / 3.8 reuse a conversation's prompt across turns: in-memory prompt checkpoints of the Gated DeltaNet layers' recurrent and conv states (QwenSession::checkpoint / reachable / cut_to; the MTP decoder's layer cache length and pending rows; DSpark drafts through CheckpointTarget), copied back into the buffers GPU decoding reads in place, so a next turn whose past reply lost its thinking goes back to the previous prompt (prompt_checkpoints, default 2); dense, MoE, VL, with MTP or DSpark, CPU and Metal (steps, chains, verification passes); a tiny random Qwen3.5 and DSpark draft (qwen::tiny) for model, engine and server tests. Qwen3.8-27B turn 2 prefill 7.9-8.6 s -> 0.56-0.62 s
- prompt checkpoints and session snapshots with a speculative draft: Inkling-2 with its MTP / DSpark drafter and DeepSeek V4 with its DSpark draft keep the draft's state beside the target's (DsparkDecoder::checkpoint / cut_to over CheckpointTarget: the waiting features, Inkling-2's head rows, a windowed cache's oldest rows; snapshots hold the draft's cache, features and head rows), so a next turn or a regenerated reply reuses the prompt and drafts as a fresh decoder; tiny random Inkling-2 MTP drafter and DeepSeek V4 DSpark draft for tests

### Fixed

- *(clippy)* drop a duplicated too_many_arguments allow on the UNet prep (clippy 1.100 duplicated_attributes, failing CI)
- *(engine)* the trained context is also read from config.json's model_max_length (Inkling-2's safetensors releases: text_config.model_max_length 1048576), so opencode and /v1/models no longer fall back to 32768
- *(chat)* conversations with images reuse the cache past them: each soft token's identity (a hash of its attachment's embeddings and its row, snapshot::MediaMarks / mm::soft_token_marks) is kept beside the cache's tokens and in snapshots' footers, and Engine::chat reuses memory, checkpoints and snapshots as far as the prompt's soft tokens are the same (any placeholder in the reused part used to restart the conversation: opencode's tool results with pictures had Inkling-2 prefill everything after the first one again at each request, ~28k / 126k tokens a turn); parts fed before a checkpoint get their soft tokens (Lm::prefill_only takes overrides); Qwen (M-RoPE spans) keeps the old rule. Prompt cache: no post-reply snapshots for recurrent models (the prompt but its last token is stored; 9.4 GB a turn at 136k tokens), a found shared prefix is touched so it outlives large snapshots, the cap holds for the directory (rescan before evicting; a vanished entry rescans). Inkling-2 tiny model with a vision tower (inkling::tiny::write_tiny_vision); engine, server and snapshot tests. Real model (UD-IQ3_XXS, Metal): tokens fed per turn at 60k 50797 -> 8434, after a restart 1

### Other

- *(sd, metal)* lean 1x1 GEMMs for the UNet transformers, q / k / v in one dispatch with whole-tile half stores (padded heads from padded weight copies), fewer passes (guided SD 1.5 step 43.1 -> 39.4 ms GPU, SDXL 276-292 -> 260-264 ms)
- CUDA decode: single-token steps as a CUDA graph (recorded on the first step and when the session's device buffers or the ops change; OXINFER_CUDA_GRAPH=0 off; the host queues a step in 14 us instead of 415-727), and a layer's q / k / v int8 matvecs of one format in one launch (mvq3_K: each block one matrix's mvq_K block, bit-identical, tested); GPU time per token Llama-3.2-1B 969 -> 898 us, Qwen3-8B 4346 -> 4180; plain decode 844 -> 917 tok/s and 209.8 -> 218.9
- *(cpu, bf16 matvec)* decode epilogues in the matmul's pool jobs, shrinking one-pass blocks, three- and four-token tiles walked in four passes (same bits); Whisper CPU decoder steps on the decode threads
- CUDA prefill plans: a norm or GLU whose output is the next GEMM's input writes its f16 rows too (dnorm_h, dglu_th: the values and range flag dhalf_t would make), so 3 of a layer's 4 conversions go; logits identical; Qwen3-8B Q4_K_M 2048-token prompts 13234 -> 13353 tok/s, Llama-3.2-1B 512 61276 -> 62082; the decode step trace also reports the host's launch time
- *(sd, metal)* UNet GEMM tiles planned by waves (tiles past the last full wave split along their inputs, summed by un_gemm_red), 16x16 / 8x8 convolutions unfolded, cooperative stores for f32 rows (guided SD 1.5 step 47.7 -> 43.1 ms GPU, SDXL 286-288 -> 276-281 ms, SD 1.5 VAE 62.0 -> 60.0 ms)
- *(inkling)* Inkling-2 Mini on a DGX Spark (GB10 and its CPU)
- *(inkling)* short passes' message gates as dot products on the CPU
- *(inkling)* write output pages before downloads on unified devices
- *(inkling)* read weight pages before uploads on unified devices
- *(inkling)* time the CUDA decode kernels on Mini's shapes
- *(inkling)* a lane per block in the CUDA matvecs on unified devices
- CUDA prefill GEMM: the 256-row stream-K tile on 8 warps of 64 x 64 outputs (256 threads, a third fewer fragment loads per MMA), and Q4_K (Q6_K with four threads a row) loads the next step's weight bytes into registers before the current step's MMAs, unpacking after (clock64 per phase showed the loads waiting ~490 cycles on L2); Q4_K 4096 x 12288 196 -> 170 us, 12288 x 4096 210 -> 186, the 128-row kernel 222 -> 204, Q6_K unchanged; logits identical; Qwen3-8B Q4_K_M prompts 13240 -> 13470 tok/s at 512 tokens, 12669 -> 13234 at 2048, Llama-3.2-1B 58828 -> 61276 at 512
- Whisper against whisper.cpp on Metal after the decoder's fused stages (496 s clip, best of two alternated runs): tiny to large-v3 2.29 / 2.13 / 1.87 / 1.72 / 1.78 / 1.57x (was 1.99 / 1.79 / 1.55 / 1.48 / 1.46 / 1.29x), large-v3 8.98 vs 14.13 s, medium 5.45 vs 9.71 s, same word error rates
- *(whisper, metal)* the decoder attention's last chunk of a head merges the chunks (device-coherent partials, a per-head count), no merge stage (medium 4.62 -> 4.56 s, large-v3 7.25 -> 7.17 s decoder, same transcripts)
- CUDA prefill attention: dfattn2 for longer steps (the heads of a GQA group in one 256-thread block, 128 query rows per key / value tile, tiles streamed by cp.async behind the MMAs, longest blocks first; from 1024 positions at head dimension 128, always at 64; OXINFER_CUDA_FATTN2=0/1); Qwen3-8B per layer 512 tokens after 1536 110 -> 91 us, after 3584 218 -> 164, Llama-3.2-1B 2048-token step 121 -> 100; outputs bit-identical to dfattn; Qwen3-8B Q4_K_M 4096-token prompts 11773 -> 12038 tok/s; fattn_bench (ignored) times the kernels
- *(sd, flux, metal)* flash attention in matrix registers (UNet, FLUX.1, Qwen-Image), GEGLU fused into the feed-forward GEMM, group-norm sums spread over channels (guided SD 1.5 step 65.5 -> 47.7 ms GPU, SDXL 353-372 -> 289-304 ms, FLUX.1 Q4_0 1024^2 1.83-1.97 -> 1.70-1.73 s/step)
- *(whisper, metal)* decoder steps in fewer, shorter stages (fused bias / residual / GELU matvec epilogues, a threadgroup-wide LayerNorm kernel, 2 rows a threadgroup), one command buffer per chained step, steps past the end token skipped by indirect dispatch (medium 5.61 -> 4.62 s, large-v3 8.87 -> 7.25 s decoder on the 496 s clip, same transcripts)
- *(cpu prefill)* bit-exact NEON prompt attention check only against the SME GEMM; without SME (M1-M3, CI's runner) the NEON GEMM's head-size-20 edge tiles round to ~1.6e-5 relative, so compare within 1e-4
- *(sd, metal)* UNet GEMMs on 128x64 / 64x64 tiles, lazy base-2 UNet softmax over the scores' memory (guided SD 1.5 step 94.8 -> 65.5 ms GPU, SDXL 467 -> ~365 ms), bf16 prefill GEMMs 64 wide on the matrix units (FLUX.1 bf16 1.95 -> 1.7 s/step), text K/V once per generation, degraded-mode warnings
- *(flux, sd, metal)* T5 and CLIP text encoders as one GPU pass each
- SDXL base 1.0 (public fp16 files) against stable-diffusion.cpp, 1024^2, 20 Euler steps: 0.47-0.62 s vs 0.83-1.04 s per step, text 0.18 vs 0.8-1.0 s, VAE 0.45 vs 4.1 s; 47.6 / 44.9 dB from diffusers f32 on the same noise and timesteps
- Whisper against whisper.cpp on Metal and the CPU (tiny to large-v3, LibriSpeech 496 s: 1.29-1.99x on Metal, 1.25-1.79x on the CPU, same word error rates within a word), the device sessions
- image generation against stable-diffusion.cpp and mflux, final table (FLUX.1 Q4_0 / Q8_0 with the device VAE, SD 1.5 / SD-Turbo whole-pass UNet)
- *(qwen-image, metal)* split-precision attention (queries / keys as hi + lo halves on the half matrix units, scores as exact as f32 operands; 2.7 -> 2.4-2.7 s per 1024^2 pass, 4.9 -> 4.4 s interleaved on a loaded GPU) and LoRA side paths side by side
- *(whisper metal)* greedy decoding chained on the device, lighter decoder steps, lazy encoder softmax
- *(sd, metal)* VAE encoders on the device plan (SD 512^2 0.44 -> 0.10 s, FLUX.1 1024^2 1.75 -> 0.3 s), unbarriered q/k/v projections (SD 1.5 guided step 98 -> 95 ms GPU), heads of 40 padded to 48
- *(kernels)* lib tests' decode_max_tokens / moe_batched are per thread, so parallel tests no longer move each other's 8..=15-token passes between the decode kernels and the prefill GEMM
- *(qwen)* the tiny Qwen3.5 checkpoint tests keep every pass off the 8..=15 token window (a 9-token turn ran the decode kernels or the prefill GEMM after the process-wide decode_max_tokens, which lookup / DSpark / TTS-batch / Janus tests raise and restore concurrently, so the cut-back and fresh passes could part: the mtp_checkpoint_cut_back_matches_fresh flake)
- *(whisper metal)* whole encoder and decoder passes on the device (gpu::whisper sessions)
- *(qwen-image, metal)* whole transformer passes on the GPU through FLUX.1's plan (Qwen-Image-2512 Q4_0 1024^2 CFG 4: 10.1 -> 5.4 s per step, stable-diffusion.cpp 12.4 s; Edit-2511 bf16 + Lightning 16.5 -> 10-11 s per pass)
- *(sd, metal)* whole UNet passes and VAE decodes on the GPU (SD 1.5 512^2: 0.60 -> 0.10-0.15 s per guided step, stable-diffusion.cpp 0.24-0.30; VAE 0.56 -> 0.10 s, sd.cpp 1.2 s; FLUX.1 VAE 1024^2 2.9 -> 0.57 s)
- *(metal)* lazy rescaling in flash_attn128_mma (the running max moves only past +8, the output rescale is skipped for key blocks where no row moved; FLUX.1 attention about -20%, same quality)
- Metal prefill round 8 (the routed experts: Qwen3.5 / 3.6 MoE take the prefill plan, 0.26x -> 1.16x llama.cpp at 512 tokens and 1.47x at 2048; gate / up pairs as one grouped GEMM; the grouped kernels' probes and a dozen designs measured and dropped)
- *(metal)* prefill plans take MoE layers whose experts add to a shared expert's output (Qwen3.5 / 3.6 MoE: 0.2x -> 1.16x llama.cpp at 512 tokens), and the routed experts' gate / up pairs run as one grouped GEMM with the activation in its stores
- *(flux, metal)* whole transformer passes on the GPU (FLUX.1 [schnell] Q4_0 1024^2: 3.5-4.2 -> 1.9 s per step; stable-diffusion.cpp 5.2 s, mflux -q 4 2.6 s)
- CPU prefill round 4, what was measured and not kept (half-precision SME, SME shares, attention tiles)
- CPU prefill round 4 (NEON prompt attention, 512-token CPU chunks) against llama.cpp
- Metal decode round 2 (tg128 against llama.cpp master of 2026-11-25: 1.04-1.26x on every row; RoPE folded into the attention, IQ4_XS matvecs, chained per-pass GPU breakdown; shape-dependent matvec splits measured in isolation and dropped for want of a gain in passes)
- *(metal decode)* RoPE folded into one-token attention (dec_attnr_*: each threadgroup ropes its query heads and its key head's new key and value with qk_rope_body's arithmetic and reads them from threadgroup memory, one per key head stores them; a dispatch and a barrier fewer per layer, bit-identical on Llama, Qwen3, Qwen3.8, Gemma 4 dense / MoE, Orpheus; OXINFER_DEC_ATTN_ROPE=0 keeps the QkRope op) and an IQ4_XS matvec of its own (MV_IQ4 / MVK_IQ4: the non-linear values from a threadgroup copy instead of constant-memory lookups, on the shape splits of the other K-quants; 17408 x 5120 471 -> 560 GB/s alone, Qwen3.8-27B Q4_K_P passes -3%). Chains time their passes (OXINFER_METAL_TRACE) and honour OXINFER_DEC_ONLY; examples/dec_ab alternates decode configurations in one process.
- Metal prefill round 7 (pooled KV buffers keep their residency, SwiGLU pairs as one GEMM; against llama.cpp master of 2026-11-25: 1.02-1.14x at 512 tokens, 1.08-1.21x at 2048 on the dense models; the grouped experts' probes and what was dropped)
- *(metal)* prefill: pooled device KV buffers stay in the residency set, and SwiGLU gate / up pairs run as one GEMM with the GLU in its stores
- *(cpu)* prompt attention on every core's NEON units, CPU prefill chunks of 512
- *(cpu decode)* a MoE's shared expert joins the experts' fused gate / up region
- *(cpu decode)* the MoE router over the pool and fewer regions per MoE layer
- prompt checkpoints and snapshots with a draft, measured on Inkling-2-Mini with its MTP head (turn 2: 5.2-5.8 s prefill -> 0.2 s); chat_turns prints the drafts accepted

## [0.0.21](https://github.com/KarpelesLab/oxinfer/compare/v0.0.20...v0.0.21) - 2026-11-24

### Added

- DeepSeek V4 / V4.1 reuse a conversation's prompt across turns: in-memory prompt checkpoints (the windows' raw keys and the compressors' pending rows; Engine::chat goes back to the previous prompt when the past reply lost its thinking; prompt_checkpoints, default 2) and session snapshots / prompt cache (the host session, rebuilt on the Metal decode path's device copies at the next pass); tiny random V4 / V4.1 checkpoints with a tokenizer for engine and server tests; examples/chat_turns measures turn-to-turn reuse
- *(web)* Add to opencode / Update / Remove from opencode on each chat model in the Models and Status views, shown when the server found opencode's configuration; docs for the endpoints and the entry written
- *(opencode)* add and remove the server's chat models in opencode's global configuration

### Fixed

- *(cli)* the binary builds for wasm32-unknown-unknown again: save_quantized is compiled out there, so --save-quantized reports it is unavailable
- *(opencode)* close the temporary file at the end of a block rather than with drop (clippy::drop_non_drop on wasm)
- *(opencode)* the temporary file's write in a function of its own (clippy::redundant_closure_call)
- *(hf)* fetching a GGUF repo (org/name:QUANT) brings its top-level mmproj*.gguf vision projector beside the model (BF16, else F16, else F32, else another; never Old_Support/ or .gguf_file), with a note, unless one is there already or mmproj is none or a file; /v1/models advertises capabilities and modalities of loaded chat models

### Other

- Inkling-2-Mini decoding on Metal, round 6 (read-ahead of small weights beside the latency kernels, routing in one SIMD-group; llama.cpp comparison; OXINFER_INKLING_PF)
- *(metal)* Inkling-2 decode: small weights read ahead into the system cache beside the latency kernels before them (ink_pf: the output projection, the message gates' down and forget / gain rows, the router; OXINFER_INKLING_PF mask), the experts' routing in one SIMD-group of each gate / up threadgroup (_rpair); tg64 chained UD-IQ3_XXS 116.5 -> 122.9 tok/s, mlx-community 4bit 121.7 -> 124.0, Q8_0 87.3 -> 88.7, greedy text unchanged
- DeepSeek V4 / V4.1 prompt checkpoints and snapshots, measured on V4.1-Flash (turn 2: 17.0 s prefill -> 0.70 s); chat_turns --save-tokens
- long contexts on Metal: half-precision device KV caches, Qwen3-8B / Gemma 4 E4B / 12B prefill at 8k / 32k and decoding at those depths against llama.cpp (f32 caches for comparison)
- *(metal)* long contexts: half-precision device KV caches (gpu_kv=f16 default; engine / server / C API key gpu_kv, --gpu-kv, OXINFER_GPU_KV, gpu::set_kv_half): decode, verification and prefill plans read and write half rows, prefill attention reads them at any length (it fell back to the f32 SIMD-group kernel past 6144 + chunk positions and converted every earlier row each chunk); KV copies in the queue's residency set (re-made resident by every command buffer at a cost growing with their size: Qwen3-8B 32k no-attention steps 59.9 -> 89.8 tok/s) and sized need + 1/8 instead of the next power of two; Gemma 4 E2B / E4B prompts run the KV-sharing layers on the last token only (TextLm::prefill_hidden, as llama.cpp's pruning; OXINFER_PF_PRUNE=0); half-cache decode attention takes up to 4 query heads of 256 / 512 per threadgroup; prefill attention on heads of 256 / 512 walks 128 keys at a time; dec_attn_bench; parity_bench depth= / text=

## [0.0.20](https://github.com/KarpelesLab/oxinfer/compare/v0.0.19...v0.0.20) - 2026-11-24

### Added

- *(web)* Status view: running and waiting requests with Cancel, loaded models with Unload / Stop and unload and Free cache, process, machine and GPU resources, prompt cache per directory and model with Clear, recent requests; refreshed every second while shown
- *(server)* status API: requests in flight with cancel, models with memory and device, process/machine/GPU, prompt caches, expert caches
- Inkling-2 reuses a conversation's prompt after a mismatch: in-memory prompt checkpoints (the convolutions' last inputs and the sliding windows, CPU and Metal) that Engine::chat goes back to, so a next turn whose past reply lost its thinking reuses the previous prompt (engine / C API / server key prompt_checkpoints, default 2, 0 off); Inkling-2 session snapshots and prompt cache (host and Metal buffers read in place, f32 / f16 / q8 caches); a tiny random Inkling-2-Mini for tests
- DeepSeek V4.1's cold-pass expert prediction depth as a setting: engine / C API / server key stream_predict (layers ahead, 0 off, at most 4, default 2), --stream-predict N, gpu::set_stream_predict; OXINFER_DS41_PREDICT still overrides
- the gpu_chain switch also turns off DeepSeek V4.1's chained greedy passes
- *(deepseek)* DeepSeek V4.1 single-token decode as one Metal command stream (opt-in, stream_gpu=on)

### Fixed

- *(dspark)* Instant::duration_since so the web clock builds
- LayerCache::stage / commit (used by the CUDA decode plans only) allowed unused where the CUDA backend is not built, so the all-features clippy passes on macOS again
- *(metal)* IQ1_S / IQ1_M multi-token matvecs give the one-token kernels' bits
- *(metal)* multi-token matvec variants deduplicated after sorting (a variant listed twice broke the library's compilation)

### Other

- DeepSeek V4.1 GPU decode tests skip on Metal devices without residency sets (CI's paravirtual GPU), where the plan declines and the model decodes on the CPU; ds41::supported names the check Plan::build makes
- the status page and its API (activity, requests and cancel, prompt-cache clear, caches/free, unload with cancel), the library and C functions
- Inkling-2 prompt checkpoints and snapshots (sessions.md, inkling.md)
- Inkling-2-Mini decoding on Metal, round 5 (stage profile against the bytes floor, the attention a head a threadgroup, mixes, down rows; kernels that wait for the one before instead of a barrier measured and dropped; KL, exactness, speculative decoding)
- *(metal)* Inkling-2 decode: the attention a query head a threadgroup with the relative bias in the keys' dot products, lighter mixes (SIMD-group sums, ink_mix5), the experts' down rows on 16 lanes, IQ2/IQ3 expert dots; tg64 chained mlx-community 4bit 112.9 -> 121.2 tok/s, UD-IQ3_XXS 107.8 -> 116.7, Q8_0 82.1 -> 87.0
- DeepSeek V4.1 GPU decode, round 3 (Q5_K multi-token variants, cold-pass expert prediction, the SSD's rate against the drive's, verification passes grouped by expert tried and dropped, dense stages near memory speed)
- DeepSeek V4.1 GPU decode: cold passes predict the next layers' experts (their routers on each layer's input) and read them with the misses, so the next run of layers does not miss them (128 cold tokens: 3260 -> 1991 misses, 15.0 -> 15.5 tok/s; OXINFER_DS41_PREDICT / _PREDICT_WAIT); Q5_K multi-token matvecs in sweeps of 4 on two rows for 9-16 tokens and long rows at 6-8 (11-token verification pass 148 -> 139 ms of GPU time)
- *(metal)* greedy decoding chained on the device for every model on the general decode plan (Llama, Qwen3, Qwen3.5 / 3.8, Qwen MoE, Gemma 3, Gemma 4 dense and MoE): DecodePlan::greedy runs one-token passes that start from the device's lookup of the token chosen by the pass before (dec_embed: the host's Table::row dequantization for GGUF F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/Q2_K/Q4_K/Q5_K/Q6_K, bf16, oxinfer int8/Q4 and MLX tables, Gemma's per-layer rows too) and end with the argmax of their logits (dec_amax_part / dec_amax_fin, the host's rule), the next pass queued before the host reads a token; a pass queued past a stop is undone (KV rows not appended, Gated DeltaNet states alternating between the host's copy and a second one); TextLm::decode_greedy for QwenModel and Model, so plain greedy replies and prompt lookup's plain rounds chain; gpu_chain off switch (engine / server / C API key, --gpu-chain, OXINFER_GPU_CHAIN, set_decode_chain; Inkling-2's chain too); tg128 Llama-3.2-1B Q4_K_M 473 -> 522, Q8_0 345 -> 365, Gemma 4 E2B 164.8 -> 172.7, E4B 92.7 -> 95.8, 26B-A4B 148.3 -> 154.9, Qwen3-8B Q4_K_M 104.2 -> 107.4, Q8_0 66.6 -> 67.9; tokens and logits unchanged
- DeepSeek V4.1 GPU decode, round 2 (greedy passes chained on the device, verification passes on the GPU with the one-token logits bit for bit, the experts' own kernels; before / after warm, with lookup on a code prompt and cold, per-stage GPU time, GPU time per verification pass size, correctness, what was kept and dropped)
- *(metal)* DeepSeek V4.1 GPU decode: greedy passes chained on the device, prompt-lookup verification passes of up to 16 tokens on the GPU with the one-token logits bit for bit; warm decode ~42 -> ~47 tok/s, a code prompt with lookup ~33 -> ~50
- *(metal)* DeepSeek V4.1's routed experts on kernels of their own (mvxd_<fmt>): IQ1_S / IQ1_M grids as halves, the sum's lanes over (slot, chunk), IQ2_XXS / IQ2_S dots; experts stage 6.0 -> 4.5 ms a token, warm decode ~42 -> ~46 tok/s
- Static (fullrust) release binary uses the GPU: dlsym 0.1 loads libcuda / libvulkan through the system's own ld.so (dldota), so the linux-x86_64-static build keeps the gpu feature; sincos / sincosf shims for that target (LLVM merges sin + cos into them, fullrust's compiler-builtins lacks them); /proc / sysfs reads (memory, physical cores) and the Linux CPU GEMM tuning also on fullrust, and --devices reports its OS as linux; a build without the gpu feature says so instead of "No GPU detected"
- speculative decoding, the rounds the policies left behind (the Markov head in one submission, the first rounds, the first draft rounds' average, chained plain rounds; before / after for Qwen3-8B DSpark exact / defaults, Inkling-2-Mini DSpark / lookup, Gemma 4 assistant, Qwen3.5 MTP, the CPU and its fixed lengths; correctness; what was dropped)
- *(spec)* the GQA draft's Markov head in one GPU submission (Backend::markov_greedy: per slot the bf16 markov_w1 row of the device's previous argmax, the w2 matvec adding into the slot's logits, the argmax; greedy drafts on the block plans), softmax_at on eight lanes; the draft's plans warmed once at load; the cost-aware policy starts with two plain rounds and averages its first 4 draft rounds evenly; DSpark plain rounds chained on the device where the target chains greedy passes (Inkling-2 on Metal: Target::decode_greedy_feats, the passes capturing the draft's features into two token slots), each token streamed (DsparkDecoder::step_emit); Qwen3-8B Q4_K_M DSpark exact code 121.3 -> 127.1 tok/s, chat 103.5 -> 110.6, prose 96.5 -> 100.1 (plain 101.5-102.2), Q8_0 code 121.1 -> 127.5, chat 104.4 -> 110.7, prose 78.4 -> 94.2 (plain 65.8); Inkling-2-Mini 4-bit DSpark prose 104.4 -> 108.5 (plain 111.3); greedy outputs unchanged
- CUDA prefill GEMMs: every thread of a block unpacks weights (four threads a row, eight values each, in the 512-thread 256-row kernel and the 64 x 64 one: Q4_K 4096 x 12288 231 -> 197 us, Q6_K 222 -> 215), and GEMMs over one input (q / k / v, gate / up) convert it to f16 once; logits unchanged
- speculative decoding policies retuned (verification cost table by backend and format, before / after for Qwen3-8B DSpark / lookup / defaults, Gemma 4 assistant, Qwen3.5 MTP, Inkling-2-Mini lookup / DSpark, best fixed lengths, the CPU, correctness, what was dropped)
- *(spec)* draft lengths from one cost model (spec_cost: verification cost by token count per backend, measured online, prior table for Metal exact / fast and the CPU); DSpark drafts on the fast kernels also with exact verification (fast_draft, --exact-draft), probes on a 2% loss budget, catch-up charged by share; MTP cost-aware by default (mtp_cost) with plain steps among the choices; Gemma 4 assistant: 0..N drafts by expected tokens per second, verification on the exact multi-token decode plan (its greedy output was not the plain output on Metal)
- CUDA prefill plans: from 256 rows on the stream-K fast GEMM takes 256 x 128 tiles (gemmhk2: 16 warps, one 512-thread block per SM, 60 KB of dynamic shared memory), so each unpacked weight value serves twice the rows, and Q6_K / Q4_K unpack with aligned funnel-shifted words / a one-load header; Qwen3-8B Q4_K_M prompts 10440 -> 12315 tok/s at 512 tokens and 10099 -> 12043 at 2048, Llama-3.2-1B Q4_K_M 49734 -> 56789 and 49713 -> 55369 (idle GPU), greedy tokens unchanged; OXINFER_CUDA_GEMM_256=0 keeps the 128-row tiles; ggml_bench times both stream-K tiles
- CUDA prefill plans: each layer's new keys and values go down on a stream of their own as soon as the op writing them has run and a host thread stages them into the host caches under the later layers' kernels (LayerCache::stage / commit: taken in only once the step succeeded), instead of ~7 ms after the kernels; logits byte-identical; Qwen3-8B Q4_K_M prompts 8885 -> 10440 tok/s at 512 tokens and 9011 -> 10099 at 2048, Llama-3.2-1B Q4_K_M 42667 -> 49734 and 43800 -> 49713 (idle GPU); OXINFER_CUDA_KV_ASYNC=0 gives the copy after the kernels back; the decode trace prints the host time between steps
- CPU decode round 9 (activations in super-blocks of 256 for Q4_K, Q5_K and Q2_K: llama.cpp's Q8_K with integer scales per 32; what was kept and dropped, per-core GB/s, decode against llama.cpp at 1-18 threads, KL and greedy agreement)
- *(cpu)* Q4_K, Q5_K and Q2_K decode kernels on activations in super-blocks of 256 (llama.cpp's Q8_K with integer scales per 32): one core Q4_K interleaved 39.5 -> 45.8 GB/s, all-Q4_K Llama 1B 50 -> 59.5 tok/s (llama.cpp 54.4)
- streaming: DeepSeek V4.1's experts addressed from the part cache on the GPU
- DeepSeek V4.1 decode on the GPU (design, measured warm / cold against the CPU path and llama.cpp, per-phase GPU time, correctness, kept and dropped); IQ2_XXS back on the dequantizer's loader
- *(deepseek)* V4.1 GPU decode: Sinkhorn on one SIMD-group, routing on one SIMD-group, layers in chunks while experts miss, experts wrapped during the prefill; on by default (stream_gpu=off for the CPU)
- CUDA decode plans of GGUF models run Q4_K, Q6_K and Q8_0 matvecs over int8 activations: x rounded once per matvec as the CPU's decode kernels round it (dq8_x: quantize_act's bytes and scales; a layer's q / k / v matvecs round their input once), then dp4a against the stored integers with a lane per block of 32 (mvq_K, mvqp_K for the gate / up pairs); the f32 matvec was bound by its arithmetic (12288 x 4096 Q4_K from L2 27.5 us, int8 7.8 with the rounding); decode Qwen3-8B Q4_K_M 113 -> 133 tok/s and Llama-3.2-1B Q4_K_M ~520 -> 703 against the f32 matvec in the same minutes on a GPU another process kept 60-70% busy; greedy text agrees with the CPU's at least as far; gpu_exact=1 or OXINFER_CUDA_MVQ=0 keep f32 activations; tests: dq8_x bytes against quantize_act, the kernels against the definition, int8 plans' multi-token steps the single-token ones' bits
- exact multi-token matvecs (verification bit for bit with one-token passes on every format; kernel times by format and token count, Inkling-2-Mini and Qwen3-8B verification and speculative decoding before / after, what was dropped)
- *(metal)* multi-token oxinfer Q4 dot in the previous kernels' shape (the affine sum selected after the plain one): 4096 x 12288 at 8 tokens 362 -> 335 us, 16 tokens 715 -> 672
- *(metal)* SiLU / GELU pairs take their own multi-token matvec variants (one row a threadgroup with half quants up to 8 tokens, sweeps of 4 beyond); decode_mvk_bench OXINFER_MVK_PAIR
- *(metal)* multi-token matvec pairs back to both matrices decoded at once; one row a threadgroup below 1024 rows
- *(metal)* multi-token matvec pairs decode the second matrix into the first's registers; 9-16 tokens of 4-bit rows in sweeps of 4
- *(metal)* multi-token matvec variants by format, row length, SIMD-groups and tokens; steps of 9-16 tokens in sweeps of 8
- *(metal)* multi-token matvec finish comments and the dead one-SIMD-group branch after its early return
- *(metal)* multi-token matvecs of one SIMD-group a row reduce without threadgroup memory; 4-bit rows as floats there
- *(metal)* exact multi-token matvecs bit for bit with one-token ones on every format, per token class, the K-quants decoded once
- round 7, the Q4_K kernel's order of work as kept
- *(cpu)* Q6_K and Q4_K interleaved decode kernels with fewer instructions (bit-identical); decode activations without allocations; docs: CPU decode round 7

## [0.0.19](https://github.com/KarpelesLab/oxinfer/compare/v0.0.18...v0.0.19) - 2026-11-23

### Other

- CUDA GGUF matvecs: Q4_K and Q6_K blocks dot through their own path (stored integers to floats by a byte permute into 2^23's mantissa, Q4_K's scale and min once per block, Q6_K's 2-byte aligned blocks as aligned funnel-shifted words, Q4_K's header in one load): Q4_K 875 -> 1036 GB/s on 12288 x 4096, Q6_K 1158 -> 1369; decode Qwen3-8B Q4_K_M 161 -> 169 tok/s, Llama-3.2-1B Q4_K_M 866 -> 917
- DeepSeek V4.1 decode round 2 (the expert kernels): what was kept and dropped, what bounds the experts (instructions per grid entry, not memory), the GPU expert split measured
- *(stream)* an on-demand request probes one page a chunk to find it cached, not three
- *(moe)* a short pass's expert activations in one parallel region, a task per expert and one for the shared expert
- *(deepseek)* the router reads its bf16 weights in place (half the bytes), bit for bit the f32 copy's dot products
- *(ggml)* one-token IQ1_S, IQ1_M and IQ2_XXS rows two at a time, bit for bit the one-row kernels
- CUDA prefill plans run the fast GEMM by stream-K where its 128 x 128 tiles do not fill the GPU evenly: two blocks per SM take equal shares of all tiles' 32-wide steps, the parts of tiles begun elsewhere go to a scratch tile per block that a second pass adds in the blocks' order (results the same from run to run); Qwen3-8B Q4_K_M prompts 7278 -> 8182 tok/s at 512 tokens and 6590 -> 7151 at 2048, Llama-3.2-1B Q4_K_M 35960 -> 43064 and 34802 -> 40273 (best of 5 on a shared GPU), greedy tokens unchanged; OXINFER_CUDA_STREAM_K=0/1 never / always; ggml_bench times the fast GEMM per shape (BENCH_SHAPES, BENCH_FAST_ONLY); docs: an int8 tensor-core GEMM measured and not kept
- *(cpu)* RMSNorm sums of squares in 16 running sums; docs: CPU decode round 7 (8 threads)
- DeepSeek V4.1 decode and prefill round against llama.cpp (where a step's time goes, what was kept and dropped); streaming's warm start, on-demand reads, GPU bursts at low clocks
- IQ1_S one-token rows without the delta in the weights, on-demand reads requested all at once, DeepSeek prefill router as one GEMM
- *(deepseek)* V4.1 decode router and hyper-connection mixes over the threads, output groups in one region; streamed models keep what an earlier run left in the page cache
- Inkling-2-Mini decoding on Metal, round 4: the message gates' norms in the kernels after them, the attention's key splits merged in the attention kernel, routing and grouping of verification passes in one kernel (release decode KL 5.845e-12 / 2.373e-11 / 7.588e-13, the quantized checkpoints' unchanged to four digits, greedy replies byte for byte): tg64 chained mlx-community 4bit 110.6 -> 112.9 tok/s, UD-IQ3_XXS 103.7 -> 105.9, Q8_0 80.6 -> 82.2; prompt lookup on chat 123.4 -> 126.5; what was dropped (multi-token dense matvecs of our own, the convolution in the multi-token output projection, routing inside the gate / up kernel, prefetching weights beside small kernels) and why; inkling2_bench --exact: MLX 4-bit verification passes differ from one-token passes in the last bits (the decode library's affine Q4 matvecs)
- Inkling-2 on Metal: the attention's message gates leave their norms to the projections on every format (ink_mvs_* now the decode library's one-token matvec for its rows and SIMD-groups, mv_acc and mv_finish's sums, scaled by the gate's norm): UD-IQ3_XXS (K-quant and IQ dense rows) and the release (bf16) too; profile UD-IQ3_XXS 10.12 -> 9.99 ms of stages; verification passes still give the one-token logits bit for bit on Q8_0, UD-IQ3_XXS and the release (inkling2_bench --exact); release decode KL 5.85e-12 / 2.37e-11 / 7.59e-13, the quantized checkpoints' as before to four digits
- Inkling-2 on Metal: passes of 2 to 16 tokens route and group their experts in one kernel (ink_route_group: ink_route's single SIMD-group routing, a SIMD-group a token, then ink_xgroup's grouping in the same threadgroup): one stage fewer a layer, the same logits bit for bit (inkling2_bench --exact unchanged); profile, 4-token passes, mlx-community 4bit: route 4.2 + group 5.6 -> 7.0 us a layer
- Inkling-2 on Metal: the attention's three message gates leave their norms to the projections after them (ink_gfg writes the weighted outputs and their sums of squares; passes of one token: ink_mvs_*, ink_mvc_*'s matvec scaled by the gate's norm; several: the decode library's matvecs, then ink_gsc scales by the same product), where every projection has the one-row-a-SIMD-group matvec (the 4-bit, Q8_0 and K-quant checkpoints; the release's bf16 projections keep the norm): one stage fewer on 24 layer applications; mlx-community 4bit tg64 --chain 8.97 -> 8.83 ms (best of 4 interleaved; the MLP gate and merge commit before it 9.05 -> 8.97), decode KL mlx-community 4bit 2.352e-3 / 5.530e-2 / 1.509e-2 as before; inkling2_bench --exact N: a verification pass against one-token passes
- Inkling-2 on Metal: the MLP's message gate leaves its norm to the router and the experts' gate / up kernels (ink_gfg writes the weighted output and its sums of squares per block, ink_mvr and _rpair / _gpair scale their sums by the norm: one stage fewer on the 24 gated layer applications), and a token's key splits merge in the attention kernel (the last split to finish, by a counter per token and key head, with ink_attn_merge's arithmetic: one stage fewer a layer past 64 keys); profile mlx-community 4bit 9.67 -> 9.47 ms of stages at tg32, 10.17 -> 9.98 at 300 tokens of context; release decode KL 6.28e-12 / 2.48e-11 / 7.88e-13 (before 5.72e-12 / 2.27e-11 / 7.91e-13), top-1 16/16, 96/96, 128/128
- Metal decode round (tg128 against llama.cpp, where a pass's time goes, what was kept and dropped); stale decode comparisons updated
- *(metal)* Q5_K decode matvecs merge the fifth bit into each value's byte
- *(metal)* Gemma 4 MoE layers add the experts' post norm in the residual step's kernel
- *(metal)* decode matvecs split rows of 2048 values and more over 4 SIMD-groups
- *(metal)* decode attention with fewer heads per threadgroup at short contexts, value rows loaded in groups; norms on 8 values per thread
- Inkling-2-Mini decoding on Metal, round 3: greedy passes chained on the device, the convolutions, queries / keys / values and routing folded into the kernels beside them (the same logits bit for bit, decode KL 5.716e-12 / 2.272e-11 / 7.911e-13 as before): tg64 mlx-community 4bit 98.7 -> 110.5 tok/s, UD-IQ3_XXS 93.6 -> 103.2, Q8_0 74.2 -> 80.8; speculative decoding again (prompt lookup on the chain: chat 113.4 -> 124.8 tok/s), what was dropped and why, what is left
- prompt lookup runs its rounds without a draft as greedy passes chained on the device (generate::Chain: go on, stop, or let the queued pass finish), committing and streaming each token as it comes (RoundDecoder::step_emit) until a draft is worth verifying; the queued pass's token then starts the verification round. mlx-community 4bit, 256 tokens greedy: chat with lookup 124.8 tok/s (1.30 tok/forward), plain 109.4, outputs identical to plain decoding (one run each, unlocked)
- Inkling-2 on Metal: passes of one token route in the experts' gate / up kernel (_rpair: ink_route's single SIMD-group routing in every SIMD-group, group y the y-th pick, one threadgroup writing the picks for _dconv): one stage fewer a layer, the same logits bit for bit; profile 52.8 + 4.5 -> 53.7 us a layer; mlx-community 4bit tg64 --chain 9.30 -> 9.02 ms a token (107.5 -> 110.9 tok/s, best of 4 interleaved)
- Inkling-2 on Metal: passes of up to 8 tokens make their queries, keys and values in the attention kernel (ink_attf: ink_q's and ink_kv's sums in the same order, the keys and values rounded as the cache holds them, the threadgroup whose split ends at its token writing that token's row and rings): one stage fewer a layer, the same logits bit for bit (tg digest unchanged); profile 11.8 + 6.1 -> 16.2 us a layer; mlx-community 4bit tg64 --chain 9.29 -> 9.18 ms, verify4 20.10 -> 19.85 ms (best of 4 interleaved)
- Inkling-2 on Metal: passes of one token run the experts' down projections and the convolution after them in one kernel (_dconv: a SIMD-group per slot, the slots summed in order in threadgroup memory, then the convolution per channel), and the attention's output projection and its convolution in one (ink_mvc_*: a row per SIMD-group, the decode library's one-token sums): two stages fewer a layer, the same logits bit for bit (tg digest unchanged); profile 33.8 + 6.6 -> 34.3 us (down + conv), 12.9 + 6.6 -> 14.7 us (output projection + conv); mlx-community 4bit tg64 --chain 9.66 -> 9.39 ms a token (103.5 -> 106.5 tok/s, best of 4 interleaved)
- Inkling-2 on Metal: greedy decoding chained on the device (Inkling2Model::decode_greedy, used by generate::decode for plain greedy sampling): each pass ends with the argmax of its capped logits into a token slot and the next one starts from that token's embedding row and norm on the device, so the next pass is queued while the current one runs and no logits come back; mlx-community 4bit tg64 10.15 -> 9.60 ms a token (98.5 -> 104.2 tok/s, inkling2_bench --chain, best of 4 interleaved); the same greedy text on the 9 replies (3 prompts x MLX 4bit, UD-IQ3_XXS, Q8_0), passes unchanged (decode KL as before)
- an int8 GGUF decode matvec on CUDA (x rounded per block of 32 as the CPU rounds it, dp4a, as llama.cpp's mmvq) measured and dropped: 142-154 tok/s on Qwen3-8B Q4_K_M and 618-662 on Llama-3.2-1B Q4_K_M against 169 / 768 for the f32 one
- *(cpu)* SMMLA prompt tiles from interleaved groups; docs: CPU decode round 6
- *(cpu)* interleave Q4_K and Q6_K too; prompt GEMMs read the interleaved groups
- *(cpu)* interleave Q8_0 and Q4_0 matrices four rows at a time for decode
- CUDA prefill plans pick the fast GEMM's tile per launch (128 x 128, 64 x 128 or 64 x 64, the largest that gives half the SMs a block) and run two 128 x 128 blocks per SM, so Qwen3-8B Q4_K_M prompts run at 7800 tok/s at 512 tokens (7588) and 7608 at 2048 (7119), Llama-3.2-1B Q4_K_M at 36839 (32718) and 36425 (33050), with bit-identical logits; the GGML kernel test runs every tile
- M3 Ultra: prefill attention with 16 or 32 keys per online-softmax step measured and dropped (heads of 128 at 512 tokens 0.488 -> 0.395 ms, at 2048 2.68 -> 2.64 / 2.82, heads of 64 slower), and the SIMD-group GEMM paragraph reflowed
- LayerCache::push_all is compiled only where the CUDA backend is (its one caller), so clippy -D warnings passes on macOS again (dead code there)
- M3 Ultra (Metal without matrix units, no SME): llama.cpp b5780 against oxinfer before / after this round on five GGUF models (Metal Llama-3.2-1B Q4_K_M pp512 / pp2048 1543 / 1858 -> 8524 / 9769 tok/s against 8419 / 7781, Qwen3-8B Q4_K_M 327 / 385 -> 1381 / 1440 against 1360 / 1296, Gemma 4 26B-A4B 229 / 274 -> 2425 / 2695; CPU 1.1-1.4x llama.cpp's best thread count), the headless machine's GPU held at 338 MHz unless the display is kept awake (caffeinate -d), the SIMD-group GEMM variants and the Accelerate CPU GEMM measured and dropped, and the gaps left (dispatch and barrier cost in small-model and MoE decoding, the Q4_K matvec)
- Metal without matrix units: the SIMD-group GEMMs' 32- and 16-token tiles put their 4 SIMD-groups along the outputs (16 outputs x 32 / 16 tokens each: 2 weight and 4 token fragment loads for 8 products, where 2 x 2 loaded 4 and 2 and waited on threadgroup memory; isolated kernels 25-44% faster, e.g. 1408x2816 Q4_0 11.6 -> 14.9 TFLOP/s), and each lane scales the two results it holds by their tokens' inverse scales instead of a product with a diagonal matrix; on an M3 Ultra, alternated twice with the previous build, pp64 / pp512 / pp2048: Llama-3.2-1B Q4_K_M 2905-2915 / 8476-8505 / 9742-9779 -> 4017-4048 / 8505-8524 / 9741-9769 tok/s, Qwen3-8B Q4_K_M 670-671 / 1364 / 1441 -> 851 / 1378-1381 / 1439-1440, Gemma 4 26B-A4B Q4_0 1016-1018 / 2306 / 2620-2628 -> 1077-1087 / 2421-2425 / 2692-2695, Llama-3.2-1B Q8_0 pp512 / pp2048 8652 / 9911 -> 8749 / 9973; the same products in the same order (first-step log-probabilities and 32 greedy tokens bit for bit the previous build's on Llama 1B and Qwen3-8B at 64 and 1500 tokens, Gemma 4 26B-A4B within KL 4.5e-6)
- CUDA prefill plans spend less host time per step: a step's keys and values go into the host caches in one parallel copy for every layer (11-19 ms -> 3.4 of a 512-token Qwen3-8B step), reserved for the prompt and the decode steps after it on transparent huge pages (the first decode step after the prompt 25 -> 6.6 ms), the device caches of every layer are one allocation (a later chunk's growth ~6 -> 0.5-1.2 ms), and the prompt's LM head runs on the GPU (matvec_rows over GGUF weights); Qwen3-8B Q4_K_M prompts 6165 -> 7588 tok/s at 512 tokens and 5432 -> 7119 at 2048, Llama-3.2-1B Q4_K_M 26871 -> 32718 and 21460 -> 33050

## [0.0.18](https://github.com/KarpelesLab/oxinfer/compare/v0.0.17...v0.0.18) - 2026-11-23

### Added

- *(platform)* --devices prints the prompt kernels chosen on aarch64 Linux

### Fixed

- *(hf)* allow shared_part_links unused off unix

### Other

- CUDA prefill plans attend on the tensor cores by default (dfattn_64 / dfattn_128: a block per 64 queries of a head over key tiles of 64 in shared memory as f16, f16 products with f32 sums, the softmax online in f32), so a 512-token Qwen3-8B Q4_K_M pass spends 1.5 ms in attention instead of 18.1 and its prompts run at 6165 tok/s at 512 tokens (5249) and 5432 at 2048 (4122), Llama-3.2-1B Q4_K_M at 26871 (21645) and 21460 (16829), with the same greedy tokens; within ~1e-4 of the f64 definition, gpu_exact=1 keeps the decode kernel
- Inkling-2-Mini decoding on Metal, round 2: the small kernels of the latency chain (an attention kernel with its lanes across a head's channels, the convolution apart from the MLP's mix, mixes, routing and norms with their loads together and few registers): mlx-community 4bit 85.5 -> 98.5 tok/s, UD-IQ3_XXS 82.1 -> 94.3, Q8_0 66.7 -> 73.6, decode KL against the f32 reference unchanged within the noise; how it was measured (the profile at speed, why skipping stages misleads), what was dropped, speculative decoding again, what is left
- Inkling-2 on Metal: ink_attv merges a token of one key split itself whatever the other tokens of the pass (the merge kernel leaves those out), rather than only when every token of the pass has one split: a token's results no longer depend on the pass's other tokens through that choice (decode KL unchanged: 5.7e-12 / 2.3e-11 / 7.9e-13; 10.18 -> 10.22 ms a token, within the noise)
- Inkling-2 on Metal: ink_attv takes 2 keys a SIMD-group at a time rather than 4 (fewer registers): attention 13.0 -> 11.8 us a layer at the ranges of tg64, 30 -> 28 us at 2000 tokens of context and 47 -> 44 at 8000 (1 key: 10.7, 32, 48); tg64 10.30 -> 10.24 ms a token, decode KL 2.4e-11 -> 2.3e-11. Splits of at most 128 or 64 keys at long contexts measured slower (8000 tokens: 11.9 -> 12.1 / 12.3 ms a token)
- Inkling-2 on Metal: a pass whose tokens all have one key split gets the attention's merge (sinks, gates) from ink_attv itself, one stage fewer a layer (10.47 -> 10.40 ms a token on mlx-community 4bit at the short ranges of tg64; decode KL 2.7e-11 -> 2.4e-11)
- Inkling-2 on Metal: the MLP's input convolution over the GPU then the mix (ink_conv, 9 threadgroups, rather than ink_conv_mix's one: 18 -> 6.5 + 8.2 us a layer), the mixes of up to 8 block states with their channels' values kept in registers (ink_mix8: 11.5 -> 8.8 us), the routing in one SIMD-group (6.8 -> 4.8 us) and the norms' loads issued together (7.4 -> 6.3 us); mlx-community 4bit 11.19 -> 10.55 ms a token (best of 5 interleaved), decode KL against the f32 reference 2.1e-11 -> 2.7e-11 (620-token prompt, 96 steps; 3.5e-12 -> 4.3e-12 and 6.7e-13 -> 9.1e-13 on the others)
- Inkling-2 on Metal: decoding attention with the lanes across a head's channels (ink_attv: a key is one load per lane and a SIMD-group sum, the key and value loads of 4 keys issued together; 21 -> 12 us a layer), splits of a fifth of the range (64 to 256 keys) with keys interleaved over the SIMD-groups, and the router's logits a threadgroup per expert (ink_mvr: 7.9 -> 4.8 us); mlx-community 4bit 11.71 -> 11.05 ms a token (best of 5 interleaved), decode KL against the f32 reference 2.1e-11 -> 2.1e-11. The profile (OXINFER_INKLING_PROFILE) commits its stages back to back, so the GPU stays at speed, and OXINFER_INKLING_SKIP takes finer kinds
- CUDA prefill plans of GGUF models take a fast GEMM by default (gemmh_K: f16 activation rows and f16 weight values, Q4_K and Q6_K unpacked in half arithmetic, f32 sums over the whole input, 128 x 128 tiles, cp.async and L1 prefetch of the next step), so on an RTX PRO 6000 Qwen3-8B Q4_K_M prompts run at 5249 tok/s at 512 tokens (3301) and 4122 at 2048 (2776), Llama-3.2-1B Q4_K_M 21645 (16027); the first step is within 0.006-0.02 in log-probability of the CPU's f32 prefill (the CPU's default int8 GEMM 0.06-0.09) with the exact GEMM's greedy tokens, and gpu_exact=1 keeps the split-half GEMM
- *(inkling)* Inkling-2 Mini on a Threadripper 9970X and an RTX 5080
- *(inkling)* Q2_0 weights on the CUDA engine
- *(kernels)* AVX2 unpacker for NVFP4 blocks
- *(kernels)* int8 VNNI GEMM for mixture-of-experts prompts on x86-64
- *(inkling)* reuse a prompt chunk's temporaries on the CPU
- GPUs without matrix units (M1-M4) get the half-row GEMMs of the prefill plans, the routed experts and the op-by-op path as the same kernels on 8x8 SIMD-group matrices (half operands, f32 sums: the matrix units' rounding), so whole prompt passes run on the GPU there instead of op by op; on an M3 Ultra, interleaved with llama.cpp b5780 (Homebrew): Llama-3.2-1B Q4_K_M pp512 1543 -> 8072 tok/s and pp2048 1858 -> 9666 (llama.cpp 8419 / 7781), Q8_0 1560 -> 8176 and 1857 -> 9330 (8844 / 8151), Qwen3-8B Q4_K_M 327 -> 1339 and 385 -> 1427 (1360 / 1296), Qwen3.5-9B Q4_K_M 490 -> 1320 and 619 -> 1426; the first step's top-20 log-probabilities within 3e-3 to 1e-2 of the f32 GEMMs' and the same 64 greedy tokens (Llama Q4_K_M at 512 and 1500 tokens, Qwen3-8B and Qwen3.5-9B at 1500)
- CUDA on discrete cards: the GGUF decode matvec takes a row per warp with whole blocks of 32 per lane in 128-thread blocks, and a gate / up pair of one format runs as one kernel with the activation (mvp_K), so on an RTX PRO 6000 Llama-3.2-1B Q4_K_M decodes at 768 tok/s (666) and Qwen3-8B Q4_K_M at 169 (161), the same greedy text as the CPU and the same top token at every step (mean KL 3.8e-4 / 8.8e-5); OXINFER_CUDA_MV_THREADS overrides the block size, and the GGML kernel test checks the pair against its definition in every format
- CPU decode round 5 (Llama-3.2-1B Q8_0 traced per region task: 85-92% busy inside regions, 0.3 ms between them, 1.31 GB a token at 240-257 GB/s; 98 to 82 regions a token with the gated MLP in one region, pool flags on their own lines, decode steps on all but two threads; 157-161 to 161-175 tok/s against llama.cpp's 168-185 under load and level in quiet windows, Q4_K_M 230 to 246-251, Qwen3-8B, Gemma 4 12B and 26B-A4B 1.5-7% faster, greedy outputs unchanged; tail splitting, capped blocks, a thread-count tuner and a fused argmax measured and dropped)
- Thread pool: workers announce themselves in a region on a flag of their own cache line instead of one shared counter (17 read-modify-writes on one line per region), the region's claim counter and the waiting workers' generation on lines of their own; pool::regions() counts parallel regions; plain decoding loops run their steps on all but two threads from 12 threads on (pool::decode_scope, OXINFER_DECODE_THREADS), the others parking: Llama 3.2 1B Q8_0 on the shared 18-core M5 Max 2% faster on average over six interleaved comparisons; examples/decode_ab.rs times decode steps under several settings in one process
- CPU decode: a gated MLP's gate and up projections, activation and quantization for the down projection in one parallel region (matmul_quant_glu: each task takes the same rows of both matrices, applies SiLU / GELU to them and quantizes its blocks of 32), instead of a region for the projections, one for the activation and the 8192-wide quantization on the calling thread; Llama / Qwen / Orpheus (qwen.rs) and Gemma's dense MLP, bit-identical (kernels::tests::glu_matches_separate_steps_bitwise; OXINFER_DECODE_GLU=0 for the old steps); Llama 3.2 1B: 98 to 82 regions a token; attention no longer zero-fills an output-gate buffer the model has none of
- DGX Spark CPU, measured against master and llama.cpp
- *(aarch64)* cache-block the f32 prefill GEMM by columns on Linux
- *(aarch64)* prompt attention as a blocked per-head NEON loop on Linux
- *(aarch64)* int8 prefill GEMM on SMMLA, cache-blocked by columns, on Linux
- CUDA speed and memory on a DGX Spark (GB10)
- *(minimax-h3)* read block pages before uploads on unified devices
- *(cuda)* take blocks of 32 per lane in the GGUF matvec on unified devices
- *(cuda)* write output pages before downloads on unified devices
- *(cuda)* map checkpoints on unified-memory devices
- Metal prefill round 6 (ops scheduled by dependence, MoE routing in parallel, Gemma 4's MoE layer tail in one kernel): llama.cpp / before / after at 512 and 2048 tokens on seven models, the per-change timings, what was measured and dropped, and the agreement with the previous build and the CPU
- Metal prefill: fused AddNorms make the plain norms of the new residual too
- Metal prefill: Gemma 4's MoE layer tail in one pass (pf_moe_tail)
- Metal prefill: ops scheduled by dependence, MoE routing in parallel
- CPU decode round 4 (Llama-3.2-1B Q8_0 profiled per parallel region: ~450 us a token of serial work between regions, 77 us of it an argmax; per-head attention, NEON activation sums and argmax kept, bit-identical: 156-162 to 159-170 tok/s against llama.cpp's 168-183, Q4_K_M 224 to 231, Qwen3-8B and Gemma 4 12B Q8_0 unchanged; prefetch, leaner and interleaved kernels, guided blocks, read-ahead, longer spins and in-region quantization measured and dropped) and MXFP8 byte scales on Metal (Inkling-2-Mini: same speed and output, 375 MB less footprint)
- CPU decode: few-token attention takes a task per head up to 1024 keys (one parallel region instead of the shared-row kernel's three: 1.4-1.8x faster at 64-1024 keys, same bits), the activation sums of a block are summed four blocks per NEON vector in the same order (bit-identical), argmax over the logits runs on NEON (~70 us to a few per token over a 128k vocabulary) and the logits are zeroed across the pool; Llama-3.2-1B Q8_0 greedy outputs, and Q4_K_M, Qwen3-8B Q8_0 and Gemma 4 12B Q8_0 ones, unchanged
- MXFP8 expert kernels that read the E8M0 scale bytes in place (mxf8b: 2^(e - 127) decoded from the bits, as on the CPU), so the bytes stay mapped from the file with Metal active too instead of a 4x f32 expansion; the f32 expansion remains under CUDA and Vulkan and with OXINFER_MXFP8_SCALES=f32. A test runs the decode and prefill expert kernels both ways and finds the outputs bit-identical
- the DGX Spark as a measured platform (aarch64 Linux, GB10 through CUDA and Vulkan): the test suite, Qwen3.5-0.8B bf16 within 1.1e-4 of transformers float32 with the same 64 greedy tokens on six prompts, CUDA and Vulkan bit-identical to the CPU's f32 GEMMs, Qwen3.5-9B q4, Llama-3.2-1B Q4_K_M, an image prompt and SD-Turbo (79.8 dB) on CUDA against the CPU, and what was not run
- Vulkan instances and devices are created and destroyed one at a time in a process: with NVIDIA's 580.173 driver on a DGX Spark, threads doing so at once could leave a driver thread spinning in its exit handler with vkDestroyDevice and every later loader call waiting behind it (the test suite, whose Vulkan tests each open their own devices, hung in 2 of 9 runs and in none of 10 since)
- On a CUDA device that shares the system's memory, what a load may still take is the system's available memory less its reserve, --models counts the device's copy of the weights beside the host's (models up to 60 GB decode on a 131 GB DGX Spark), the unload test prints the free memory instead of asserting on a figure every process moves, and cuda::tests::bit_identity_report says how many prefill GEMM outputs differ from the CPU kernels (none on the Spark's GB10)
- The Inkling-2 and MiniMax-H3 CUDA engines wait for a host-to-device copy before they go on: cuMemcpyHtoD may return with a copy of 64 KB or less still queued on the default stream, which their non-blocking kernel stream does not wait for, so a kernel launched at once could read the old contents or lose its output to the late copy (seen on a DGX Spark as matvecs_and_gemms_match_definition failing 2 times in 14 beside the other CUDA tests, none in 27 since)
- CUDA devices that share the system's memory (CU_DEVICE_ATTRIBUTE_INTEGRATED: the DGX Spark's GB10, Jetson) are listed as unified, and their free memory is the system's available memory where the driver only reports its free pages (cuMemGetInfo on the Spark equals MemFree, without the page cache)
- Metal prefill round 5 (matrix-unit attention, norms and GLUs writing the GEMMs' half rows): llama.cpp / before / after at 512 and 2048 tokens on seven models, what was measured and dropped, and the agreement with the previous build and the CPU
- Metal prefill: f32 scores on the matrix units for unscaled attention (Gemma)
- Metal prefill: Gemma's AddNorms make the next GEMMs' half rows too
- Metal prefill: fused norms and GLUs skip their f32 rows when nothing reads them
- Metal prefill: the SwiGLU / GeGLU and the RMSNorms before GEMMs make the GEMMs' half rows
- Metal prefill: attention on the matrix units (pfm_*), as llama.cpp's flash_attn_ext_tensor
- Inkling-2-Mini decoding on Metal measured against the 125 tok/s figure: what was kept (0.5-1%), what was built and dropped (an event-signalled prelaunch of the next pass, 0.1 ms; fusing the merge, routing, convolution and gate norms into their neighbours, 759 -> 519 dispatches at the same speed; a short-range attention 20-30% faster that parts the greedy text from the previous build's; the dense matvecs' decoding, load-bound), speculative decoding on code and chat (at most +13%), and what is left
- Inkling-2 on Metal: a decoding pass's first command buffer holds 2 steps instead of 8, so the GPU starts about 70 us sooner (mlx-community 4bit, best of 4 interleaved: 11.93 -> 11.87 ms a token), and the attention's relative bias puts its threads on the distances a key range has rather than on 3 x 1024 slots (the same logits bit for bit); inkling2_bench prints the greedy tokens and a digest of the logits with INKLING2_BENCH_SUM
- DeepSeek V4.1 at 1M tokens: the partial run (546,816 of 1,004,522 tokens in 3.4 h, stopped by hand, no recall result), rates and memory on the way
- the scan admission only for passes of 16 parts or more; metal_ds4 test on a backend of its own; docs/performance.md: streamed V4.1 prefill
- DeepSeek V4.1 on the real weights: Metal prefill attention, tiled candidate-aware indexer, 2048-token streamed chunks, scan-resistant expert cache
- org/repo:QUANT fetches a split model's shared unnamed part (ISTA-DASLab's V4.1 engram file) and links it as the part; deepseek4_check --greedy / --ppl on real checkpoints

## [0.0.17](https://github.com/KarpelesLab/oxinfer/compare/v0.0.16...v0.0.17) - 2026-11-22

### Other

- GGUF on CUDA measured again on the tree with the x86-64 int8 prompt GEMM and the unloading of device weights (Qwen3.8-27B UD-Q3_K_XL 537 tok/s at 512 prompt tokens and 31.1 decoding against 199 / 5.1 on the CPU, Llama-3.2-1B Q4_K_M 8485 / 498 against 2405 / 74), the agreement against the CPU's f32 and int8 prompt GEMMs, and who owns the device memory
- a test loads GGUF weights with their decode and prefill plans, runs them, drops them and loads again, and the device's free memory returns (164 MB held, within 48 MB after each unload); on the real files three load / run / drop rounds leave Llama-3.2-1B Q4_K_M at 14142 MB free each time and Qwen3.8-27B UD-Q3_K_XL within 2 MB per round
- GGUF on CUDA with what was measured on an RTX 5080 against the CPU path and llama.cpp (Qwen3.8-27B UD-Q3_K_XL 539 tok/s at 512 prompt tokens and 31.2 decoding, against 82.6 / 5.3 on the CPU and 2488 / 58.1 in llama.cpp; Llama-3.2-1B Q4_K_M 8899 / 512 against 43978 / 755), the formats covered, the agreement with the CPU and where the remaining time goes
- The CUDA GEMMs over GGUF weights decline activations their f16 halves cannot hold (32768 or more in magnitude, or not finite): a prefill step is dropped on a flag a kernel sets on the device and the op-by-op GEMM checks its input on the host, so the CPU path runs them, at 1% of Llama-3.2-1B's prompt speed (8994 tok/s at 512 tokens against 9111)
- CUDA prefill plans bring a step's new keys and values down through pinned memory, so Llama-3.2-1B Q4_K_M takes 512 tokens at 9173 tok/s (7310) and 2048 at 7743 (6538) on an RTX 5080
- CUDA prefills GGUF models on whole-pass plans: the matmuls as tensor-core GEMMs over the stored blocks and norms, RoPE, attention, the Gated DeltaNet conv and delta rule and the activations over all the step's tokens on the device, so Qwen3.8-27B UD-Q3_K_XL takes 512 tokens at 539 tok/s (162 op by op, 74 on the CPU) and Llama-3.2-1B Q4_K_M at 7310 (1518), the prompt's logits within KL 2e-7 of the CPU's on the 27B
- GGUF weights run on CUDA: decode steps read the stored blocks with a matvec per ggml format (every block format but the ternary and 1-bit ones) and prompts take a tensor-core GEMM over them, so Qwen3.8-27B UD-Q3_K_XL on an RTX 5080 decodes at 31 tok/s (4.3 on the CPU before) and prefills 512 tokens at 162 tok/s (74), Llama-3.2-1B Q4_K_M at 515 and 1518, with a mean KL of 1.5e-4 to 2.2e-4 per decode step against the CPU
- Token hook: code that sees each committed token of a reply and may force tokens the model reads next
- the x86-64 CPU prefill section is measured again on the tree merged with CPU round 3, against master 2894417 and llama.cpp 8d60873 (Qwen3.8-27B UD-Q3_K_XL pp512 76-77 -> 195-201 tok/s, llama.cpp 173-175; Llama-3.2-1B Q4_K_M pp2048 584-591 -> 2052-2055, llama.cpp 2407-2411); this replaces the figures in the earlier commit messages of this series, which compared against master 43448bd (before round 3's GEMM attention) and whose "13 GB of zeros" was an estimate (measured: 393 ms of a 4.5 s pass)
- a Qwen-family prompt's attention keeps the per-head loop (AVX2 / AVX-512 kernels, the bits of one-token attention) instead of the score and value GEMMs, which are slower on this architecture (same build, interleaved: Llama-3.2-1B Q4_K_M on 2538 tokens 712 / 699 / 713 -> 1887 / 1879 / 1880 tok/s, Qwen3.8-27B GGUF on 748 tokens 182 / 184 / 185 -> 199 / 201 / 197, Qwen3.5-9B q4 on 2602 tokens 513 / 506 -> 604 / 593)
- x86-64 CPU prefill after the int8 GEMM and the attention, DeltaNet, element-wise and temporaries work, measured against master and llama.cpp e300919 on a Threadripper 9970X (Qwen3.8-27B UD-Q3_K_XL pp512 76-83 -> 190-201 tok/s, llama.cpp 157-188), with what the int8 rounding changes and what was measured and not kept
- a prompt's attention keeps the output sums of the heads that share a value row in registers while the keys pass (no store per key) and takes its softmax exponentials at the vector width, so long prompts of small models run 2.4x faster on the CPU (Llama-3.2-1B Q4_K_M, 2538 tokens: 745 / 751 / 759 -> 1780 / 1798 / 1811 tok/s interleaved; attention 1.0 -> 0.2 ms per token) with byte-identical logits
- The x86-64 int8 prefill GEMM is tested through the matmul driver (several pool jobs, two token chunks, Q8, Q4 and six GGUF kinds against the decode kernels within 2e-5), and qwen_greedy_dump takes --chunk N to run a prompt on the decode kernels
- A prompt's layers of the Qwen family reuse the previous layer's large temporaries instead of taking and zero-filling new ones (13 GB of zeros over a 748-token prompt of the 27B), so prompts run 8% faster on the CPU (Qwen3.8-27B GGUF: 170 / 168 -> 182 / 183 tok/s interleaved, llama.cpp 162) with byte-identical logits
- a prompt's element-wise passes (SiLU of the feed-forward, the DeltaNet convolution and its gated norm, the attention output gate, now also across the thread pool) are compiled for AVX2 / AVX-512 and round in line, so they take a third less time and prompts run 1-5% faster (Qwen3.8-27B GGUF, 748 tokens: 169 / 159 -> 170 / 168 tok/s interleaved) with byte-identical logits
- The Gated DeltaNet rule reads and writes each state once per prompt token instead of twice (a token's update and the next token's decay in one pass) and runs with AVX2 / AVX-512 on x86-64, so the attention and DeltaNet part of a prompt takes 19% less time (Qwen3.8-27B GGUF, 748 tokens: 4.1 -> 3.3 ms per layer call) with byte-identical logits
- a prompt's attention scores eight keys at a time per head and adds its value rows with AVX2 / AVX-512, so the attention and DeltaNet part of a prompt takes 16% less time (Qwen3.8-27B GGUF, 748 tokens: 4.9 -> 4.1 ms per layer call, 1.3 -> 1.0 ms per token) with byte-identical logits
- with AVX-512 VNNI a language model's prompt runs on an int8 prefill GEMM (prefill_gemm=fast, the default there as on aarch64; exact opts out), 64 products per vpdpbusd on cache-blocked columns, so prompts run 2.1x faster on the CPU (Qwen3.8-27B UD-Q3_K_XL pp512 76 -> 159 tok/s, llama.cpp 162) with the decode kernels' activation rounding (KL 3e-3 after 748 tokens, 17/17 greedy tokens the same)
- MiniMax-H3 on CUDA takes 39 s for the 448x256, 124-frame clip (54 s before) and 525 s for the 1344x768 one, whose 124 frames have none of the coloured blobs the same prompt and seed give at 448x256, where the CUDA clip shows the CPU clip's scene and blobs.
- MiniMax-H3 on CUDA: a projection's outputs go in groups whose weights stay in the cache while the rows pass, the attention kernel reads its keys and values one phase ahead, and kept blocks are copied to the device as each first runs, so a 448x256 forward takes 6.1 s instead of 7.4 s (the first one 7.6 s instead of 11.5 s) and a 1344x768 one 114.5 s instead of 137 s, with the same bits
- MiniMax-H3 on CUDA: the attention runs as one kernel per block that keeps no score matrix (64 queries x 64 keys per step, running softmax, same split operands), so a 1344x768, 124-frame forward takes 137 s instead of 309 s (attention 5.05 -> 1.71 s per block) and a 448x256 one 7.4 s instead of 8.6 s, within 2.1e-5 of the earlier CUDA forward and 3.8e-5 of torch's
- models on request (a repo id a request names is fetched, loaded and run), idle unload after 30 min, unloads for RAM / GPU memory
- the latest releases of every crate (core 0.1.43, pipeline 0.1.15, image 0.0.7, h265 0.0.18, av1 0.1.24, ...); the av test now expects oxideav-image to convert every format
- Metal prefill round 4 (per-op accounting against llama.cpp, methodology, what was measured and dropped)
- Metal prefill: pooled device KV copies across sessions, host KV copies overlapped to the end; pass profiling and A/B tools
- kernels test: bf16 matmul bits equal the row-block kernel's only off the SME GEMM
- CPU round 3 (int8 MoE experts, Qwen prompt attention as GEMMs, MXFP8 scales in place, decode findings)
- keep the E8M0 scale bytes as stored on the CPU
- CPU prefill: MoE experts on the int8 GEMM, Qwen/Llama prompt attention as GEMMs
- the 2.5% image-feature gap on Qwen3.8-27B-Heretic is the bf16 reference's own rounding: oxinfer's tower is within 1.2e-5 of transformers float32 on Qwen3.8-27B's tower (four images, 224x320 to 1024x1024) while transformers bf16 is 2.2-4.6% from its float32 run, and examples/qwen_tower_dump.rs with tools/ref_vl.py --tower measure a tower without its language model.
- Qwen-family prompts zero their per-layer temporaries across the thread pool, so prefill is 25-31% faster on CUDA (Qwen3.5-9B q4 pp512 313 -> 394 tok/s) and 6-12% on the CPU (Qwen3.8-27B GGUF 68 -> 75), with the same results
- The thread pool takes one thread per physical core by default where the logical CPUs are SMT siblings (Linux), which decodes 4-19% faster on a 32-core / 64-thread CPU (Qwen3.8-27B GGUF 4.3 -> 4.9 tok/s) with prompts within 5%
- with AVX-512 the CPU prefill GEMM takes 16 rows x 16 tokens per step and is cache-blocked by columns, so prompts run 1.5-1.75x faster on the CPU (Qwen3.8-27B UD-Q3_K_XL pp512 41 -> 70 tok/s, Qwen3.5-9B q4 132 -> 201) with the same bits
- the grid-table i-quants (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) unpack with AVX2, so GGUF files made of them decode 1.5x faster on the CPU (Qwen3.8-27B UD-Q3_K_XL: 2.7 -> 4.2 tok/s) with the same bits
- darrellbest/Qwen3.8-27B-Heretic and its GGUF, NVFP4 and FP8 packagings are listed with what was measured against transformers bf16 and what was not run.
- compressed-tensors float-quantized FP8 weights (E4M3 with a factor per output row in weight_scale) load as stored instead of being refused for a missing weight_scale_inv.

## [0.0.16](https://github.com/KarpelesLab/oxinfer/compare/v0.0.15...v0.0.16) - 2026-11-22

### Other

- Inkling-2 Mini on iOS / iPadOS (device fit, options, entitlements, Simulator results); ios/README: steps for a real device
- Inkling-2 on iOS: decide streaming on the app's memory budget, keep the GPU engine within it, memory_limit_gb
- Inkling-2 Metal: expert lanes in row groups, NVFP4 file bytes given back after repacking, LoRA A x in the matvec's stage
- MiniMax-H3 on CUDA as measured: the stage split of a forward on both devices, memory, agreement with the CPU path and torch, and one 1344x768 forward in 302.5 s
- MiniMax-H3 on CUDA: the video VAE's 36 decoder blocks run on the GPU too, 124 frames of 448x256 in 6.4 to 8.1 s against 160 to 187 s on the CPU with frames within 3.8e-5, and the 4-step clip takes 54 s end to end against 943 s
- MiniMax-H3 on CUDA: the FP8 transformer's 50 blocks run on the GPU with the blocks that do not fit streamed through two slots, one forward of a 448x256 124-frame clip (4682 rows) in 8.6 s against 194 to 457 s on the CPU, within 4.1e-5 to 1.3e-4 of torch per forward
- Metal prefill round 3 (Q6_K / Q5_K word reads, device MoE routing, device first-logits head; what was measured and dropped)
- the prompt's last-row LM head on the GPU's matvec kernels
- Metal prefill: Q6_K / Q5_K blocks read as words; MoE layers routed on the device
- oxinfer serve and oxinfer web are one server (GUI, management API, /v1, /v1/messages; web opens the browser), default port 8080; GUI: downloaded models above the catalog, a Downloaded filter; catalog: OpenUncensored/Inkling-2-Mini-Uncensored
- Inkling-2 CUDA: prefill GEMMs on the weights' exact values (pp512 2900 -> 3560 tok/s), no ring upload for fresh sessions; docs: the CUDA engine, verification, speed against llama.cpp
- default-run = oxinfer, so `cargo run -- web` works beside the membench binary
- Inkling-2 drafter: checked against llama.cpp, the head product off the decode path, cost-aware draft length
- README, docs/cli.md: the install commands without trailing comments (they scrolled out of view)

## [0.0.15](https://github.com/KarpelesLab/oxinfer/compare/v0.0.14...v0.0.15) - 2026-11-22

### Other

- Merge remote-tracking branch 'origin/master'
- Merge inkling-cuda-rollback: CUDA rollback from the Metal side, joined with the CUDA engine's
- where a decode step's time goes now, what moved it and what did not (serial encoder, staging, fusions); Not done
- Inkling-2 CUDA: sessions go back, exact passes of up to 16 tokens, captured streams for drafts
- Inkling-2 Metal: the one-threadgroup kernels without latency chains; the message gates in fewer, wider kernels
- Inkling-2 on CUDA: sessions go back (prompt lookup on CUDA), convolution histories as rings
- Inkling-2 Metal: the one-threadgroup kernels without latency chains; the message gates in fewer, wider kernels
- torch_rng test: bit for bit on macOS, within 2 ulps elsewhere
- Inkling-2 CUDA: tiled prefill attention, gate projections per token, NaN-proof routing, sessions that outgrow the device go back to the host
- Inkling-2 Metal: the routed experts' gate / up LoRA updates on the GPU
- Inkling-2 Metal: a stage profiler, NVFP4 without table lookups, one expert output per SIMD-group
- Inkling-2 Metal: the LoRA factors through master's LoraDelta::links / factors and Lin::lora_delta
- Inkling-2-Mini's drafter on Metal: a whole draft pass in one submission
- Inkling-2 on Metal: whole decode and prefill passes on the GPU, rollback, speculative decoding, LoRA
- H.264 with CABAC, one encoder per second in parallel; WebM with its Duration and playable in Chrome; AAC from MP4 decodes
- CPU timings with the DMD2 and FameGrid LoRAs on a quieter machine
- speed after whole-pass Metal plans (before / after, CPU with LoRAs), verification rows, Base / SFT
- split float32 layers as one float32 convolution on Metal (conv_f32); moss_check single-branch reps
- whole transformer passes on Metal, ConvRot quants, single-file VAEs, Base / SFT, PyTorch noise
- LoRA merges into the GGUF block types, the ggml quantizers against llama.cpp
- ggml block quantizers; LoRA merges keep GGUF weights in their own type
- GPU side paths in the Metal / CUDA decode and prefill plans (Qwen, Gemma, per-expert MoE), MTP-layer adapters
- Inkling-2 CUDA: prefill GEMMs within ~5e-8 of f32, parallel convolutions, gates as GEMMs
- DevSession::as_any is used by the CUDA path only (no dead-code error elsewhere)
- Inkling-2 on CUDA: whole decode steps and whole prefill passes on the GPU
- Video as MP4 (H.264 + AAC) through oxideav by default; WebM (VP9 + Opus) and AVI on request
- Web GUI: MOSS-Image presets on the Image page and in the catalog; plan and download take a preset id
- ZONOS3 / Drama 3 listed as speech models before they load; a steadier video cancel test
- the page title without .localdomain; models without a probed config file are counted from their cache folder
- rebuilt with the accelerator panel, the update banner, voice chat and video
- Web GUI: tell when a newer oxinfer release is out, one-click update and restart
- Models page and --models: which accelerators this machine has and which one each model uses
- end-to-end video test on the tiny MiniMax-H3; Haru turns end on silence; LFM3-Omni codec fetch
- :tiny: tiny random MiniMax-H3 and MiniMax-H3-Image releases for end-to-end tests
- video jobs, spoken replies, SD3 / H3-Image, image progress, voices; GUI pages
- :tiny: run the test on the CPU, so the test process does not settle on Metal
- a tiny random SD 3.5 diffusers folder for end-to-end tests
- LoRA adapters (PEFT and GGUF), verification; catalog: AtomicChat/Inkling-2-Mini-GGUF with the Uncensored LoRA at run time
- llama.cpp GGUF adapters, user adapters on Qwen / Llama / Gemma / Inkling-2
- AtomicChat/Inkling-2-Mini-Uncensored-GGUF (AD-Q4_K, AD-IQ3_S, AD-IQ2_M)

## [0.0.14](https://github.com/KarpelesLab/oxinfer/compare/v0.0.13...v0.0.14) - 2026-11-21

### Other

- decode attention per key/value head, leaner KV caches, Mini docs and catalog
- the inkling2_loop decoder, GGUF (llama.cpp's and unsloth's layouts), mmproj vision, NVFP4 int8 kernels, quantized KV caches
- GGUF transformers (molbal/Kromoss-GGUF, arch moss_image), low-rank LoRAs as float32 matmuls beside the base one on the GPU (CPU: the low-rank kernel), oxinfer --presets, --help section; docs/moss-image.md (architecture, LoRAs, presets, verification against diffusers, speed), models table and README rows
- MOSS-Image (lodestones/Kromoss) text to image: Qwen3.5-9B conditioning, the 42-block SiTU DiT, FLUX.2 VAE decoding; LoRA (low-rank side paths, LyCORIS LoKr / LoHa merged), the release's schedule plus ComfyUI's er_sde / beta57; CLI (--preset), C API (oxi_moss_image_*), server; curated presets in the catalog (kromoss, kromoss-dmd2, kromoss-dmd, kromoss-famegrid)
- Thinking signatures: purecrypto's HMAC-SHA256 and OS RNG instead of a local SHA-256

## [0.0.13](https://github.com/KarpelesLab/oxinfer/compare/v0.0.12...v0.0.13) - 2026-11-21

### Other

- Anthropic /v1/messages: signed thinking blocks, Claude Code's request extras
- Browser build: no std::time on wasm32-unknown-unknown in a chat turn

## [0.0.12](https://github.com/KarpelesLab/oxinfer/compare/v0.0.11...v0.0.12) - 2026-11-21

### Other

- One-line installers (install.sh for macOS / Linux, install.ps1 for Windows) on GitHub Pages; oxinfer --version
- get started first, the model list moved to docs/models.md
- get started with oxinfer web
- Web GUI: native backend mode, model setup, every modality, Connect apps; docs/web.md
- oxinfer web backend: the GUI's management API, the model cache, Anthropic's /v1/messages
- MiniMax-H3 ref2va and audio VAE
- PixilabAI/Zawen-v0.1-35B-A3B and mradermacher's Q4_K_M GGUF of it
- a description that does not name models
- MiniMax-H3 ref2va: usage, what was checked, and a first run on the release's Ref2VA/ (an image reference and a melody copied as the score: the fox of the reference, the melody's pitches and envelope)
- MiniMax-H3 ref2va: video from references (the release's Ref2VA/): --ref FILE for images, videos with their soundtrack and audio clips, normalized as diffusers' setup step does; the presentation with <Picture i> / <Audio j> / <Video k> labels and timestamped Qwen3-VL video blocks (QwenVl::encode_video); the references through the video VAE's encoder and the audio VAE's; the [text | references | target audio | target video] layout (equal to diffusers' build_ref2va_packed_sequence on a mixed request: tools/minimax_h3_ref2va_layout.py, examples/h3_ref2va_check.rs); reference soundtracks held clean at t = 1 in the denoising loop
- MiniMax-H3 audio VAE: the encoder (DAC waveform encoder, causal attention projection, mean_proj) for ref2va's reference audio; within 7e-7 of diffusers on a tiny random VAE and 6.4e-5 on the release's weights over 5.2 s of stereo
- MiniMax-H3 FL2VA at the release's defaults on the CPU: 124 frames, 49 forwards on the bf16 transformer in 5001 s (attention 467 s with the blocked kernel), coherent frames, a quiet stereo soundtrack
- blocked CPU kernel with an online softmax
- MiniMax H3 video: the release's own FL2VA/ layout
- MiniMax-H3 DiT: read the AdaLN vectors from their table per row
- the MiniMax-H3 prompt conditioning against diffusers and transformers on the real Q8_0 text encoder: the dog prompt's token ids are the reference's and its embedding is transformers' hidden_states[50] on the same dequantized values within 3.6e-6 (1.8e-5 in the worst row; 24% against hidden_states[49] or [51]), so the blobs of the dog clip do not come from the text path; the speech prompt's <d> and </d> were split and are now the release's tokens, its 448x256 clip was redone and has blobs too; the release's bfloat16 arithmetic moves the embedding by 1.2%, Q4_K_M by 5%; the transformer takes no input the earlier torch runs did not build with diffusers' helpers, and the noise has the reference's layout and no scaling. Q8_0 against the bf16 weights stays unmeasured. tools/minimax_h3_text_ref.py run --dtype bf16.
- MiniMax H3 video: the prompt's tokenizer has the seven tokens MiniMax's tokenizer_config.json adds to Qwen's (<d>, </d>, <|cutoff|>, <|lyrics_start|>, <|lyrics_end|>, <|caption_start|>, <|caption_end|> at 151669 to 151675, as transformers and ComfyUI number them), whether the vocabulary comes from the release's tokenizer.json or from a GGUF: a line of dialogue's <d> and </d> were split into three tokens each. h3_text_check dump writes the pipeline's token ids and hidden states of a prompt on the real text encoder, and tools/minimax_h3_text_ref.py computes the same hidden states in transformers from the GGUF's dequantized tensors and compares.
- the MiniMax-H3 reference runs share oxinfer's prompt embedding, which they do not clear.
- the coloured blobs of the 4-step 768p LoRA at 448x256 are the weights' doing, not oxinfer's: torch on the same files agrees with every oxinfer forward within 3e-4 (each block within 2e-5) and gives the same blobs from the same noise, ComfyUI's model code agrees with diffusers within 7e-5 on equal weights and gives blobs too; the blobs first show in the third forward; ComfyUI rounds a merged LoRA back to FP8 on the CPU, which oxinfer does not; the cause in the weights is not found.
- MiniMax H3 video: OXINFER_H3_DUMP=DIR keeps every forward's rows and velocities (and one forward's sequence after every block), tools/minimax_h3_video_ref.py redoes a real request in torch from them (diffusers' transformer and scheduler on the decoded pruned FP8 file with the LoRA merged: block by block, one forward, or the whole sampling), tools/minimax_h3_video_comfy_ref.py does a forward or the sampling with ComfyUI's own code, and tools/minimax_h3_video_long.py with h3_video_check compares a tiny transformer with the real heads on a real request's 4682 rows.
- MiniMax-H3 video run on the real small files (pruned FP8 transformer, fp16 video VAE, fp32 audio VAE, 4- and 8-step LoRAs, Q8_0 and Q4_K_M text encoders): load times, memory, timings per forward at 448x256 and 1344x768, what the frames show, the soundtracks' statistics and Whisper transcripts, the coloured blobs of the 4-step 768p LoRA at 448x256, and what stays unverified.
- MiniMax H3 video: OXINFER_H3_LATENTS=FILE keeps the denoised video and audio rows (written when missing, decoded again without the transformer when present), and --profile prints the time by operation.
- MiniMax H3 video: --audio-vae DIR|FILE names the audio VAE (Sources.audio_vae), so a single-file --model run has a soundtrack: the release's audio_vae/ folder or ComfyUI's vae/minimax_h3_audio_vae_fp32.safetensors, its configuration read from the file's header; that file decodes within 3.7e-5 of the folder (h3_audio_check --write).
- MiniMax-H3 video from ComfyUI's quantized files (int8 ConvRot transformers and VAE, the W6A8 transformer, the text encoder as one file in bf16, int8 ConvRot or NVFP4): the formats as comfy_kitchen defines them, sampled rows of the real files against the bf16 ones, the agreement on tiny checkpoints, memory, what was not run and what stays unsupported.
- MiniMax H3 video: the text encoder from ComfyUI's single files (--text-encoder qwen3vl_32b_minimax_h3_{bf16,int8_convrot,nvfp4_awq}.safetensors, text-only prompts): the 50 stored layers with their configuration from the tensor shapes and no final norm; NVFP4 in comfy_kitchen's packing (high nibble first, block scales in cuBLAS's tiled order) repacked into ggml's blocks, pre_quant_scale on the input, the int8 embedding with a scale per row; tools/qwen3vl_comfy.py writes a tiny Qwen3-VL in the three forms and examples/h3_text_check.rs compares the conditioning with transformers.
- MiniMax H3 video: ComfyUI's W6A8 file (*_pruned_w6a8.safetensors): 6-bit codes with an FP8 scale per group and a scale per row, decoded at load to the int8 values comfy_kitchen computes with, on the ConvRot rotation; tools/minimax_h3_video_comfy.py writes the tiny transformer in that form (blocks and video head, the refiner as int8 ConvRot) and references through comfy_kitchen's int8-activation kernels beside the float32 ones.
- MiniMax H3 video: ComfyUI's int8 ConvRot files (the transformer's *_int8_convrot.safetensors, pruned or not, and the video VAE's): int8 rows with a scale per row used as stored, their input rotated by comfy_kitchen's grouped regular Hadamard matrix (src/comfy_quant.rs, from each module's comfy_quant entry); tools/minimax_h3_video_comfy.py writes a wide tiny transformer and the tiny VAE in that form with comfy_kitchen's quantizer, with references on the dequantized weights.
- The C API's request body pointer is cast with .cast::<u8>(), which clippy accepts where c_char is u8 (aarch64 Linux) as well as where it is i8.
- The `server` feature builds without `hf`: a model's load reports download progress only when downloads exist.
- The binary builds without the `media` feature: the Audio8 TTS command line needs none of it (WAV in and out), and CI's clippy covers --all-targets --no-default-features.
- MiniMax-H3 video from ComfyUI's single files, the turbo LoRAs and a GGUF text encoder: flags, what the files hold, what was checked on tiny checkpoints and on sampled rows, what was not run and what stays unsupported.
- MiniMax H3 video: tools/minimax_h3_video_comfy.py writes the tiny checkpoints as ComfyUI's single files (bf16, pruned, FP8 scaled, the fp16 VAE) and a LoRA in both namings with diffusers references; h3_video_check takes --transformer, --vae and --lora.
- MiniMax H3 video: the text encoder from --text-encoder, a Qwen3-VL-32B-Instruct folder or its language model as a GGUF file (llama.cpp's qwen3vl architecture, text-only prompts), or quantized at load with --quant q8.
- MiniMax H3 video: the video VAE from ComfyUI's fp16 single file (--vae FILE), its configuration read from the file's header and the encoder under MiniMax's tensor names.
- MiniMax H3 video: the transformer from ComfyUI's single files (--model FILE --vae DIR, --transformer): bf16, the pruned form with its AdaLN curve table, and pruned FP8 scaled weights used as stored.
- MiniMax H3 video: LoRA files on the transformer (--lora), the 4- and 8-step turbo LoRAs with the step count and sigma shifts their names stand for (--shift, --audio-shift).
- The NEON prefill GEMM takes float-stored matrices of any width.
- CI tests build with the `ci` profile (release without whole-program LTO), and every job has a time limit.
- MiniMax-H3's soundtrack, how it is written, what was checked of the audio VAE and that no generated soundtrack was decoded or listened to.
- MiniMax H3 video: the audio VAE's decoder checked against diffusers (tools/minimax_h3_audio_tiny.py, examples/h3_audio_check.rs), on a tiny random checkpoint (8.6e-8) and on the real audio_vae/ (8.8e-6 on the encoder's latents of a synthetic signal, 5.9e-5 on normal rows).
- MiniMax H3 video: the soundtrack is written from the command line, as a 16-bit PCM stream in the .avi (one chunk after each frame) and as a .wav next to an animated WebP or a PNG sequence.
- MiniMax H3 video: the audio VAE's decoder (BigVGAN with anti-aliased SnakeBeta activations), and the pipeline decodes the denoised audio rows into a 32 kHz stereo soundtrack.
- CPU prefill: llama.cpp's int8 rounding by default for language models' prompts; `exact` opts out
- Gemma 4 26B-A4B: Metal-vs-CPU prefill drift is routing amplification, not a bug
- Prefill plans with captured layers: prompts with drafts stay on the GPU
- second round of Metal prefill work, llama-bench matrix's Metal columns
- Metal matrix-unit GEMMs: Q8_0 and Q4_0 weights dequantized in half
- Metal prefill: residual adds in the GEMM stores
- Metal prefill: MoE router GEMMs on 8x the threadgroups
- Metal prefill GEMMs: 64-row tiles for narrow outputs
- Metal benches: K splits and kernel name patterns in mma_gemm_bench, moe_bench shapes
- matrix-unit GEMM kernel benchmark (mma_gemm_bench)
- Metal prefill: copy each layer's new keys and values while the GPU runs on
- Metal prefill GEMMs: a short last token tile in 32-row tiles
- count the BOS in the prompt length, as llama-bench does
- CPU prefill GEMMs and the re-measured CPU matrix; SME bit-equality test
- CPU prefill: bf16 weights on SME; `fast` hybrid of SME and int8
- CPU prefill: f32 GEMM on the SME tiles; int8 SDOT GEMM options
- SC117's GSQ-RCO abliterated Qwen3.8-Flash-Next GGUF; REPO:QUANT prefers the model over its draft sidecars
- allow clippy's const thread-local false positive on Android
- llama-bench matrix against llama.cpp master; whole-pass GPU prefill
- Metal decode: unpack Q4/Q4_0/Q4_K/Q5_K nibbles and Q5_K high bits as bytes
- Whole-pass GPU prefill for Llama, Qwen3, Qwen3.5 and Gemma 3/4
- Metal prefill plans: flash attention on the SIMD-group matrices
- MiniMax H3-Image: its VAE is no longer taken for the video VAE.
- MiniMax-H3 video generation, what was checked and what is not supported.
- The model catalog has a video kind with text-to-video and image-to-video capabilities, and MiniMax-H3 in it.
- MiniMax H3 video: text to video and first / last frame to video from the command line (--frames, --last-image), written as Motion JPEG AVI, animated WebP or PNG frames.
- MiniMax H3 video: the DiT's audio stream and the packed text / keyframe / audio / video sequence with its two schedules, checked against diffusers on tiny checkpoints.
- MiniMax H3 video: the video VAE, a causal 3D CNN encoder and the ViT decoder with 3-axis RoPE, temporal clips and tiling.
- LFM3-Omni in the model catalog, docs/models.md, docs/lfm3.md and the README index, with what was checked against the reference.
- tools/ref_lfm3_omni.py compares features, tokens, logits, greedy text, speech codes and the codec's waveform with the checkpoint's own code.
- LiquidAI/LFM3-Omni-2B reads audio and images through its own encoders, answers in text on the chat path, and speaks through the depth decoder and the Liquid codec (--speak, -o reply.wav).
- Streamed chat and completion responses say what they wait for before the first token.

## [0.0.11](https://github.com/KarpelesLab/oxinfer/compare/v0.0.10...v0.0.11) - 2026-11-14

### Other

- a newer release replaces the executable before any command runs, or is reported when it cannot be.
- The prompt-lookup test no longer asserts draft counts that depend on timing.
- Granite, LFM3 and EmbeddingGemma 2 notes say only what was checked.
- EmbeddingGemma 2: google/embeddinggemma-2 as an embedding backbone, with image and audio embeddings through Gemma 4's vision and audio towers.
- chat templates without a thinking block (taobao-mnn/MNN-Pocket-0.3B) get no <think> prefill after the assistant header.
- Liquid LFM3-1.7B runs on the Qwen path, its gated short convolutions as a mixer beside full and sliding-window attention, with its chat template and tool calls.
- Granite 4: IBM Granite 4.1 / 4.2 and fine-tunes such as TwIL-8B run on the Llama path, from safetensors and GGUF, with their chat template and tool calls.
- the catalog entry, the model table row, docs/haru.md with the measured agreement and its line in the README index.
- the comparison with the PyTorch reference (tools/ref_haru.py records a greedy conversation and every normal value it draws; examples/haru_check.rs replays them: the reference clip, the codec both ways, prompt tokens, and per frame the text token, the hidden state, the latent and the audio).
- the command line, in the shape of the reference's offline tool (--audio USER.wav -o REPLY.wav, --system for the card, --ref-audio for the voice; the model's text on standard output).
- Haru-4B (haru, higurashi-ai/Haru-4B), full-duplex speech to speech, from the PyTorch reference (oxinfer::haru): a frame's three streams summed into one position of the Qwen3.5-4B text model, the MeanFlow audio head and the projectors, VibeVoice's causal codec with a state per layer, the prompt (character card, voice reference), the text stream's sampling with the DRY penalty, and a session stepped one frame at a time.
- Audio8 TTS: the catalog entry, the model table row, docs/audio8-tts.md with the measured agreement, its line in the README index and the server's speech row.
- Audio8 TTS: the comparison with the checkpoint's own code (tools/ref_arktts.py, examples/arktts_check.rs: token ids, the RoPE table, the codes of a reference clip, prompt rows, greedy frames and logits, the decoder's waveform).
- Audio8 TTS: --speak on the command line (--ref-audio with --ref-text or a named clip of the voices folder, --speak-stream) and the server's /v1/audio/speech, streamed as it is generated.
- Audio8 TTS (arktts, Edge0/Audio8-TTS-Realistic-Air): the model from the checkpoint's own code (oxinfer::arktts): the chat prompt in rows of a token and ten codes, the slow and fast transformers with their bf16-rounded RoPE tables, the sampler with its redraw of repeated semantic tokens, and the codec read from codec.pth (causal DAC decoder and encoder, window transformers, residual quantizers), decoded in windows and piece by piece for streaming.
- documentation (docs/longcat.md with the sources of the architecture, the agreement on tiny checkpoints and the runs on LongCat-Flash-Lite-Sparse's real weights; the models table, the README index) and the --models catalog entries.
- longcat_check --nll (the likelihood a checkpoint gives a token sequence, for runs on real weights), and the n-gram mean taken as a division like the reference.
- the tiny checkpoint builder in the released layout (tools/longcat_tiny.py: LongCat-2.5's architecture, or LongCat-Flash-Lite's / LongCat-Flash-Chat's), the reference on transformers' LongcatFlash model and Meituan's NgramEmbedding with the indexer transcribed from SGLang-FluentLLM (tools/ref_longcat.py) and the comparison (examples/longcat_check.rs).
- LongCat from the CLI and the engine: streamed from storage when larger than memory, the chat format chosen from the checkpoint's template, prompt-lookup rounds.
- LongCat-2.5 (pro_v, LongcatCausalLM; also LongCat-2.0, LongCat-Flash-Lite-Sparse and the LongCat-Flash models they extend): the text model (two sublayers and a shortcut MoE per layer, MLA with LoRA scales and DeepSeek's YaRN, the sparse-attention indexer with keys always attended and a choice shared by the second attention, a softmax router with identity experts, the n-gram embedding read a few rows per token with the tokens left out of it) and the chat format with its tool calls; a config.json without model_type is recognized by its architecture.
- The GGUF matmul test no longer depends on which tests run beside it.
- Fish Audio Drama 3: the prompt cases of the reference check.
- Fish Audio Drama 3: the catalog entry, docs/fish-drama.md with the measured agreement (fish-speech's PyTorch reference on small checkpoints and on the released weights), speed and memory, the models table, the README index and the server's speech row.
- Fish Audio Drama 3: the server's memory estimate takes the checkpoint as stored whatever quant says (its FP8 experts are not requantized).
- Fish Audio Drama 3: the checks against fish-speech's PyTorch reference (examples/fish_check.rs dumps prompt rows, frames, logits and codec output; tools/ref_fish_drama.py compares; tools/fish_drama_tiny.py writes a small random checkpoint with bf16 or FP8 experts).
- Fish Audio Drama 3: --speak on the command line (--ref-audio with --ref-text, --voice as the speaker tag, --quant, --mmap; on the CPU) and the server's /v1/audio/speech.
- Fish Audio Drama 3: the model (fish_omni): the prompt rows, the backbone on the Qwen3.5 MoE code with its FP8 experts as stored, the codebook tower with its caches over the sequence, the reference's sampler and chunking of long texts, and the codec (the causal DAC decoder, and the encoder for reference clips).
- FP8 E4M3 rows are converted from their bits instead of through the table, in a form that vectorizes (AVX2 / AVX-512 where the CPU has them): the same values bit for bit, twice as fast on the conversion.
- catalog entries, the model table row, docs/ripple.md and its line in the README index.
- the model (sliding and full attention layers on Gemma's attention, gated attention, value residuals, a clamped MLP, an embedding scale) from Ricursive's modeling code, bf16 and NVFP4 checkpoints, CLI / engine dispatch, the check example.
- its chat format (untrimmed ChatML turns, JSON tool calls), held against the template's jinja rendering, and replies whose reasoning and tool calls are marked in plain text filtered in the text stream.
- catalog entry, the model table row, docs/raon.md and its line in the README index.
- the model (delta attention with a write gate, gated MLA, four Raon_mHC residual streams, per-layer embeddings, tied head) from KRAFTON's modeling code, CLI / engine dispatch and a check example.
- its chat format (Qwen3.5's ChatML template with its own no-thinking prompt), held against the template's jinja rendering.
- Kimi's KDA and gated MLA layers as functions other families can call, with what Raon-1's delta attention adds (a write gate, the decay's own rank, queries and keys normed as F.normalize does).
- the catalog entry, the model table row, docs/ocr.md and its line in the README index.
- the comparison with the checkpoint's own code (tools/ref_unlimited_ocr.py: token ids, pixel values, encoder features, packed embeddings, logits, the greedy reply).
- the command line (--image per page, the reply with its markup, the decoding rules as flags), the engine family and with it the OpenAI server's image input.
- the model from Baidu's own code: page views (global view and 640 tiles), the SAM ViT-B into CLIP ViT-L encoder and its projector, the MoE decoder with DASH block-sparse attention (per-layer router, entmax prior, page candidates), two-axis page RoPE, the page count guard and the no-repeat n-gram rule.
- Chat formats can ask for replies shown with their special tokens (ChatFormat::keeps_special_tokens), for models whose markup is made of them.
- documentation (docs/inkling.md, the models table, the README index) and the --models catalog entries.
- the tiny checkpoint builder with transformers' reference outputs (tools/inkling2_tiny.py: floats or packed NVFP4 / MXFP8 experts, chat prompts, an image), the reference for the vision tower alone (tools/ref_inkling2_vision.py) and the comparison (examples/inkling2_check.rs).
- Inkling-2 from the CLI and the engine: its chat format (typed messages, the thinking effort, tool declarations, calls and results, images as messages of their own), message headers left out of the text of a reply, and images through the vision tower.
- Inkling-2 (inkling2, thinkingmachines/Inkling-2-Small): the text decoder (attention with QK L2 clips, a relative-position bias, sinks and an output gate, short convolutions, sliding-window and full layers, the attention residual over block states, a soft-capped MoE and logits) and the vision tower (40-pixel tiles without resizing); experts read as the releases store them, interleaved gate / up rows in NVFP4, MXFP8 or floats.
- Nemotron 3.5: catalog entries, docs/nemotron.md with the measured agreement (transformers on tiny checkpoints and on the real Transcribe weights, llama.cpp on the real Lightning Q4_0), the models table and the README index.
- Nemotron 3.5 Transcribe: the audio side of the speech LM (NeMo's features, the speech recognition encoder, the Sortformer speaker encoder with its speaker cache, the projection), --transcribe and the server's transcription endpoints through whisper::Transcriber, with reference checks on a tiny checkpoint and on the real weights.
- Nemotron 3.5: the Mamba-2 time step follows the config's time_step_limit (no clamp, as NVIDIA's training and serving code and llama.cpp; transformers' scan floors it at time_step_min, which a session can reproduce); a GGUF block is MoE when it has a router; no prompt-lookup rounds on the command line; a tiny GGUF writer.
- Nemotron 3.5: the chat template (always a system turn, XML tool declarations, <think>, Qwen-style tool calls) and the command line and engine wiring.
- Nemotron 3.5: the Nemotron-H text model (nemotron_h: Mamba-2, attention without positions, MoE with relu^2 experts), the release's, NeMo's and GGUF tensor layouts, with a tiny-checkpoint reference check.
- Kimi K3.1: catalog entries, the model table row, docs/kimi.md and its line in the README index.
- Kimi K3.1: the text model (KDA, gated MLA with the K3.1 indexers, cross-layer residuals, latent MoE) from Moonshot's modeling code, safetensors and GGUF layouts, the tiktoken tokenizer, CLI / engine dispatch and a check example.
- Kimi K3.1: its chat format (encoding_k3.py: tagged messages, think / response / tools blocks, tool declarations and calls) and replies filtered by tag in the text stream.
- Kimi K3.1: the SiTU expert activation, compressed-tensors MXFP4 weights repacked into ggml blocks, tiktoken-ranked BPE merges, Mistral Large 4's MLA helpers shared.
- Fish Audio Drama 3: what the checkpoint holds and what each stage would need (docs/fish-drama.md); the model is not supported.
- the catalog entry and the documentation (docs/zonos.md).
- an example and a script that compare the tokenizer, the prompt, greedy generation, the decoder's waveform and the speaker embedding with Zyphra's PyTorch reference.
- --speak on the command line and the server's /v1/audio/speech.
- the language model, the ZAC 1 codec decoder and the speaker encoder of Zyphra/ZONOS3-Mini from its GGUF files (oxinfer::zonos).
- ERNIE 5.5: documentation (docs/ernie.md, the models table, the README index) and the --models catalog entries.
- ERNIE 5.5 from the CLI and the engine: the base model's prompt format (a text to continue, no system role), checkpoints whose shards have no index, and the tokenizer and the MTP layer in deepseek4_check --load.
- Tokenizer regex: a plus after an interval repeats it (Oniguruma's reading, which the Python tokenizers library has), so that ERNIE 5.5's tokenizer groups digits by three from the right.
- ERNIE 5.5: the reference in PyTorch (tools/ref_ernie55.py, from Baidu's vLLM plugin on DeepSeek's inference code), the tiny checkpoint builder, and the MTP layer in deepseek4_check.
- ERNIE 5.5 (ernie5_5_moe, baidu/ERNIE-5.5-Flash-Base) on the DeepSeek V4 model: HCA layers with the head mix and the output gate, MLA layers with the DSA indexer, the SiTU dense layer and latent MoE, MultiMax, the MTP layer and a greedy decoder with its drafts; bf16 and FP8 checkpoints in Baidu's naming.
- residency requested from eight sets in parallel; Flash-Next plans build concurrently
- Flash-Next prefill plans: QSA score rows in a 32 MB ring instead of 512 MB
- Metal MoE: grouped GEMM tiles of 128 / 32 / 16 rows; Q4_K and Q5_1 read as words
- allow clippy's const thread-local false positive on Android
- the fine-tunes run as smoke tests, and the count of types in the GSQ-RCO file.
- Catalog and docs: the GSQ-RCO and Ternary-Bonsai 2 GGUFs, what was checked on the trending Qwen and Gemma 4 derivatives, and what the Edge0 and E384-FP4 layouts would need.
- An example that dumps a Qwen model's greedy run in llama_logits_dump's format and compares two such runs.
- The AVX2 group products stop at the block for ggml blocks of 64 and 128 elements.
- AVX2 unpackers for the 1-bit and ternary GGUF types: 4 to 6 times fewer nanoseconds per weight than the portable decoder.
- DSpine drafts (tencent/Qwen3.8-27B-DSpine-b16) draft through their code bridge: a write after every draft layer, the last one redone per slot with the token drafted before it.
- A .gguf file without the GGUF magic is reported as damaged instead of as an unknown repo id.
- Qwen4-Exp GGUF files load again: the MiniMax lightning-indexer loader no longer claims their QSA indexer weights.
- Hadamard-folded GGUF checkpoints (prism.hadamard metadata, Ternary-Bonsai 2) run: folded weights get their input sign-flipped and rotated, the latent token embedding is rotated back.
- GGUF files with the 1-bit and ternary ggml types load: Q1_0, Q2_0, TQ1_0, TQ2_0 and the PrismML fork's PQ2_0 and PTQ1_0.
- catalog entries, docs/glm.md with the measured agreement (transformers on tiny checkpoints, llama.cpp on the real UD-Q2_K_XL), the models table and the README index; glm5_check --dump.
- no prompt-lookup rounds for GLM-5.3-Flash on the command line, whose recurrent state cannot roll a refuted draft back; the tiny GGUF writer copies a published file's tokenizer.
- the chat template (reasoning effort line, thinking, tool calls) and the command line, engine and streaming wiring.
- GGUF files keep -exp(A_log) in ssm_a, GLM-5.3's latent norms have their own epsilon; a tiny GGUF writer, a per-layer check and unit tests.
- the text model of GLM-5.3-Flash (glm5_next) and GLM-5.3 (glm_moe_dsa), with a tiny-checkpoint reference check.
- the transparent-picture run and the RGBA rule in the docs, a note in the C header.
- documentation (docs/qwen-image.md, models, the conditioning encoder) and the --models catalog entries.
- the session's DeepStack rows behind a box (the session stays under clippy's variant-size bound), an RGBA PNG only for a transparent region.
- editing with one or more input images (through the text encoder's vision tower and, as latent tokens, the transformer's sequence); an FP8 single file and edits in the tiny-checkpoint tool.
- Qwen3-VL DeepStack vision features for the conditioning encoder (qwen_vl): the tower's intermediate features through their mergers, added to the image tokens after the language model's first layers.
- Qwen-Image-2.1 and -2.1-Turbo: text to image through the Qwen-Image pipeline (Qwen3-VL text encoder, saved Turbo sigmas, GGUF transformers with or without metadata, LoRAs, RGBA output).
- the single-stream block-causal transformer, the prompt's keys and values computed once and reused by every step.
- the residual RGBA VAE (AutoencoderKLQwenImage21: 64 latent channels, 16x, a wider decoder, group shortcuts).
- DeepSeek V4.1 Flash: an engram table per layer (blk.N.engram_embd) is read too; where the two GGUF layouts' correspondence comes from.
- DeepSeek V4.1 Flash: catalog entries, the model table row and its section of the DeepSeek V4 page.
- DeepSeek V4.1 Flash: deepseek4_check --load opens a checkpoint without running it (the released GGUF layouts, headers only) and prints its tokenizer's ids.
- DeepSeek V4.1 Flash: a tiny GGUF writer in the layouts of the released conversions, integer GGUF tensors for the engram hash, image spans in the check example.
- DeepSeek V4.1 Flash: its chat format (a system token, the reasoning effort line, DSML tags with a leading space) and the CLI / engine dispatch.
- DeepSeek V4.1 Flash: the text model on the DeepSeek V4 code, from DeepSeek's naming and from deepseek41 GGUF files.

## [0.0.10](https://github.com/KarpelesLab/oxinfer/compare/v0.0.9...v0.0.10) - 2026-11-13

### Other

- httpsd 0.2.1; the static fullrust binary with the server again

## [0.0.9](https://github.com/KarpelesLab/oxinfer/compare/v0.0.8...v0.0.9) - 2026-11-13

### Other

- oxinfer --check-update / --update from the GitHub releases (compcol, fstool)
- capabilities (CAN column, --can filter), every model of the README table
- clippy (large enum variants boxed, redundant slicing) and rustfmt in the MiniMax code
- --models, what runs on this machine with an estimated decoding speed
- --control-image reaches both FLUX and Qwen-Image; README: transcription endpoint fields
- rustfmt
- Qwen-Image image to image, inpainting, ControlNet, Layered and FP8 files (flags, agreement)
- image to image, inpainting, ControlNet, Layered, FP8 scaled checkpoints
- response_format (json_object, json_schema), tool_choice required / function, logprobs
- Constrained decoding: JSON / JSON Schema grammars, forced tool calls and logprobs in the sampler
- reference checks for the new modes (tiny pipeline, real-weight steps) and README
- LoRAs, several Kontext references, image to image, inpainting, Fill, Canny / Depth, Redux, BFL single files
- OmniGen2 / BAGEL / Step1X-Edit step caches, DPM-Solver++ and posterior sampling as measured
- OmniGen2, BAGEL, Step1X-Edit: step caches, DPM-Solver++, VAE posterior sampling (all opt-in)
- JanusFlow, Janus image sizes and conversations, measured agreement
- JanusFlow (rectified flow, SDXL VAE), conversations, output size
- Stable Diffusion 3 / 3.5: tools/ref_sd3.py reference, README section and measured agreement
- Stable Diffusion 3 / 3.5: MMDiT transformer, CLIP-L / CLIP-G / T5 conditioning, text to image and image to image
- Stable Diffusion XL: diffusers reference script and measured agreement
- Stable Diffusion XL: CLI, C API and server defaults, library example
- Stable Diffusion XL: two text encoders, added conditioning, kohya LoRA names
- batched utterances
- batched decoding of several clips
- decode steps of independent sequences in one pass
- adapters on routed MoE experts, per selected expert
- document beam search and batches, reference check against openai-whisper
- beam search and batched transcription
- x86-64 AVX2 kernels executed and verified on real hardware
- x86-64 kernels: f16 scales through F16C inside the Q4 and GGUF kernels
- AVX2 score lanes and value accumulation on x86-64
- AVX2 decode kernels and dequantization, bit-identical to the portable path
- x86-64 kernels: direct AVX2 tests against the scalar fallbacks, OXINFER_SIMD=scalar
- README (platform row, section with the measured agreement, --device)
- prefill GEMMs on a Vulkan compute device (bf16, Q8, Q4, affine Q4)
- DeepSeek V4: runtime interventions (residual hooks on the hyper-connection copies, shared / routed expert neurons)
- residual hooks on a stream of several copies per token
- Qwen2-based embedders (gte-Qwen2, stella) with bidirectional attention and per-size heads
- fix the pool deadlock in the VAE decode; README row
- skip text encoder folders without weights
- text to image (DiT, PDD sampler, ViT VAE decoder)
- Qwen3-VL text models (`qwen3_vl`) on the Qwen path
- NVFP4 in ModelOpt's packaging (MiniMax-H3-Image-PE)
- MiniMax-VL-01 image input; MiniMax-01 reference fixes
- MiniMax M3 / M3.1 image input
- Qwen2 / Qwen2.5 text models (MiniMax SynLogic), GGUF `qwen2` too
- Always leave the system 20% of its memory (at most 2 GB)
- MiniMax M3 / M3.1 and Text-01 / M1 rows
- MiniMax-Text-01 / VL-01 (text) / M1: lightning attention, post-norm residuals
- MXFP8 and NVFP4 checkpoints (MiniMax-M3.1-Flash, M3-MXFP8)
- MiniMax M3 / M3.1 text: MSA sparse attention, swigluoai MoE, chat format
- MiniMax M2 / M2.1 / M2.5 / M2.7 on the Qwen path; M3 text groundwork
- the whole tokenizer.json format and encode API
- 2048-token prefill plan steps
- whole prefill passes on Metal as a prefill plan
- GPU decoding with images in the context
- Gated DeltaNet rule, attention and MoE router of prefill on Metal
- QSA sparse attention in the GPU decode plan past 2051 tokens of context
- Qwen3.8-Flash-Next image input: Qwen3-VL mmproj GGUF loading (qwen3vl_merger) for Qwen3.5-family towers
- GPU decoding with streamed weights for Gemma 4 / Gemma 3 (dense, E2B/E4B, 26B-A4B)
- CPU decode attention reading shared key/value rows once; DeepSeek V4 / Mistral Large 4 indexer skips
- Streamed MoE on the GPU: release kept expert wraps after every step; README measurements
- stream_gpu=auto in the engine options; README option row
- Streaming GPU decode: no page touching for device-read units, slower eviction retries, measured GPU/CPU choice for streamed MoE
- GPU decoding with streamed weights (dense and MoE Qwen), device memory defaults
- :ops: f32 matmuls are float-stored, so they never reach the decode kernels
- Prompt lookup and prompt cache: measured defaults, README results
- Prompt-lookup drafting with a verification-outcome memory; session snapshots and a prompt cache
- CPU multi-token column-group kernels and batched MoE experts (measurements)
- `old` mode (token pairs and per-expert MoE matmuls)
- multi-column decode kernels for blocks-of-32 formats, batched MoE experts
- Tokenizer in the crate: drop the tokenizers dependency
- cleanroom specs of llama-modchip's performance changes (quant kernels, CPU runtime, model/server, speculative)
- CUDA decoding (speeds, agreement, speculative verification, defaults)
- whole decode steps on the GPU (single- and multi-token plans), streamed MoE experts
- CUDA measurements for OmniGen2, BAGEL, Janus-Pro and device MoE experts
- routed MoE experts on the device (gather, per-expert GEMMs, activation, combination)

## [0.0.8](https://github.com/KarpelesLab/oxinfer/compare/v0.0.7...v0.0.8) - 2026-11-05

### Other

- Qwen3.8-Flash-Next (model table, section with agreement and speed numbers)
- multi-token GPU verification for MTP (shared expert wraps between plans), MTP from unsloth's GGUF heads
- GPU decode on the real model (per-pass expert wrapping for huge MoE), auto-streaming excludes the n-gram table, split-GGUF metadata fixes, tools
- Qwen4-Exp (Qwen3.8-Flash-Next): hyper-connections, n-gram embedding, QSA, MTP; CPU + Metal decode
- Fast verification by default; --exact-verify opts out; Q4 Gemma 4 31B / Qwen3.8-27B drafts default with it
- Gemma 4 26B-A4B (MoE) measured on CUDA
- Fast verification: README speed / drift tables, C header key, CLI banner, specialized Q5_K / Q6_K loaders
- Fast verification (opt-in): batched simdgroup-matrix matmuls for speculative verification
- FLUX Kontext, Qwen-Image and Step1X-Edit measured on CUDA
- httpsd 0.2.0; drop the 20 s late-response workaround, --max-body-mb sets httpsd's limit
- Build the binary without the `media` feature again (wasi)
- Start-up banner: only claim GPU decoding for a backend that has it
- Speculative decoding defaults: DSpark for Qwen3 4B / 8B / 14B and Gemma 4 12B, MTP for Qwen3.5-4B / 9B; README
- DSpark / MTP: faster exact GPU verification, MoE multi-token passes, cost-aware rounds, GGUF drafts, draft discovery
- OpenAI-compatible server: oxinfer serve, oxinfer::server (httpsd), Engine::chat, C API
- no-reassociation only where decoding plans need it
- DSpark drafts for Qwen and Gemma 4 (agreement, acceptance, speed, CLI)
- Qwen3 / Qwen3.5 / 3.8 and Gemma 4 drafts; multi-token GPU verification
- DeepSeek V4: README section (architecture, checkpoints, agreement, speed, DSpark) and engine docs
- DeepSeek V4: NEON MXFP4 kernel, draft mapped when the target streams, 3 drafts per round by default
- DeepSeek V4: DSpark draft (DeepSeek V4 stages behind dspark::Target), per-group wo_a on the decode kernels
- DeepSeek V4: GGUF tokenizer specials in the BPE vocabulary; streamed MoE prefill wraps only the routed experts
- DeepSeek V4: text model (mHC, CSA / HCA compressed attention, lightning indexer, hash and clamped-SwiGLU MoE), GGUF and DeepSeek-layout checkpoints, MXFP4
- Mistral Large 4: short-prompt prefill on Metal, bounded DSpark context, draft on the GPU
- Mistral Large 4: DSpark speculative decoding, on by default with the draft beside the GGUF
- record what was measured on the RTX 5080
- list the weight formats in the module docs
- FP8 E4M3 GEMMs (f32 and matrix units, dense and grouped MoE)
- Mistral Large 4: explicit mmproj option, tri-state weight streaming
- Mistral Large 4: runtime interventions (residual hooks, dense / shared / routed expert neurons)
- make the context current per call; direct tests of every kernel
- A device asked for by name no longer falls back to the CPU
- fix NVRTC kernel compilation, load native binaries; first run on NVIDIA

## [0.0.7](https://github.com/KarpelesLab/oxinfer/compare/v0.0.6...v0.0.7) - 2026-11-01

### Other

- Mistral Large 4: document streaming and image options in the library and C API
- Mistral Large 4: README (architecture, speed, agreement), reference tools
- Mistral Large 4: streaming by default when too big for RAM, GPU experts in prefill, Pixtral vision
- Mistral Large 4: MLA + DSA indexer + MoE text model, split GGUF, Mistral-native and FP8 checkpoints
- exact attention on the matrix units with f32 operands; Step1X half-attention cost was noise
- quantized Lightning runs lose far less (merge LoRA in f32, exact modulation channels)
- v1.1 template, padding-token masking and reference offsets; README
- Step1X-Edit (v1.2, v1.1): connector + FLUX MMDiT, Qwen2.5-VL prompt encoding, CLI, C API
- instruction-based image editing, text to image and image understanding
- README numbers, calcuis GGUF transformers, mapped text encoder; VAE mid-attention in query chunks
- instruction editing, in-context generation and text to image
- Qwen-Image-Edit (2511 / 2509 / Edit) and Qwen-Image text to image; Qwen2.5-VL
- FLUX CLI: --size is the output size, and the reference image is resized to it
- README section and model table rows; CLI shares --init-image with SD editing
- 128-wide-head attention on the matrix units; examples/flux_edit.rs
- public DoubleBlock / SingleBlock (load + forward) for reuse by other FLUX-style DiTs
- exact modulation vectors with quantized weights, F16 weights as stored, T5 GGUF encoders
- FLUX.1 Kontext: instruction-based image editing, FLUX.1 text to image
- Image editing: README section, sd_edit example, PyTorch .bin diffusers weights (MagicBrush)
- Stable Diffusion image editing: img2img, inpainting, InstructPix2Pix

## [0.0.6](https://github.com/KarpelesLab/oxinfer/compare/v0.0.5...v0.0.6) - 2026-10-28

### Other

- runtime interventions (spec, formats, CLI, library, C API, agreement, results, GPU coverage, limits)
- Runtime interventions: neuron edits, steering, directional ablation, SAE features
- decoding on the GPU (what runs where, speed vs CPU and llama.cpp, agreement), --gpu-decode
- GPU decode: Gemma 4 26B-A4B MoE layers
- whole single-token decode pass on the GPU
- Gemma 4 26B-A4B: MLX experts, GEMM prefill attention, GPU vision MLP, README
- Gemma 4 26B-A4B (MoE): routed experts, vision, GGUF, decision bundles
- voice cloning, SNAC encoder and 32/44 kHz codecs, multilingual info, loop guard
- Llama 3 GGUF and Orpheus GGUF, agreement and speed vs llama.cpp
- Llama GGUF: Llama 3.x and Orpheus TTS GGUF files

## [0.0.5](https://github.com/KarpelesLab/oxinfer/compare/v0.0.4...v0.0.5) - 2026-10-25

### Other

- streaming context 2/2 by default, faster SNAC, README section
- Orpheus text to speech: Llama 3 on the native engine, SNAC decoder, oxinfer::tts
- README section, model table rows, tools/ref_janus.py
- text to image and image understanding
- oxideav-core types in the API; `oxideav` feature; media without oxideav-meta
- Typed decisions over images (JEV-27B-VL, JEV-9B vl/)
- image, video and audio input (Qwen and Gemma 4), agreement and speed
- Vision towers: torchvision-exact resize, GPU matmuls, bf16 Qwen tower
- Qwen3.5/3.6/3.8 image and video input, Gemma 4 video input
- Whisper speed table (final measurements, 6-thread rows)
- --threads
- bf16 cross-attention keys and values below f16 decoder precision
- Whisper agreement for base, medium, large-v3 and tiny.en
- share the engine's run lock, en for English-only models, rule and segment tests
- Whisper speech to text (models, usage, agreement with transformers, speed); tools/ref_whisper.py
- fused attention kernel for 64-wide heads (Whisper encoder, SD 2.x)
- Whisper speech to text: transformers-exact features, encoder/decoder, long-form decoding, timestamps
- project goals (run everything everywhere, then make it fast)
- JEV-35B (merged, LM-head readout) measurements
- typed decisions (Jev models), measured agreement and latency
- Jev typed decisions: runtime LoRA, Engine::decide, C API oxi_decide, CLI --decide
- Core ML converter: block-shifting Gemma local cache, prefill and infer bit-identical
- Core ML runtime: shifting local caches (local_cache_length) are fed exactly once, never padded
- Core ML converter: Fisher-weighted LUTs, mixed precision, shifting Gemma local cache, Qwen3.5 MoE
- llama.cpp comparison rerun for oxinfer after the matrix-unit and multi-token kernel merges
- interleaved oxinfer vs llama.cpp benchmark; README comparison tables
- SD 1.5 with the LCM-LoRA as the default image model, LoRA files in the browser build

## [0.0.4](https://github.com/KarpelesLab/oxinfer/compare/v0.0.3...v0.0.4) - 2026-10-25

### Other

- multi-token GGUF kernels, MoE expert streaming, thread policy, measurements
- Multi-token kernels: SMMLA tiles only for per-16 integer-scale formats, pair kernels for 2 tokens
- optional read-ahead of guessed MoE experts; vector, parallel activation quantization
- evict on a background thread; fix the miss count of a file's last page
- Thread QoS policy, embedding rows read ahead when streaming
- read the routed MoE experts on demand instead of whole layers
- GGUF multi-token decode kernels: weights unpacked once per pass, SMMLA (i8mm) tiles
- Engine options gpu_exact and gpu_moe (process-wide switches in oxinfer::gpu)
- Metal matrix units (M5/A19): prefill and GEMM measurements, MoE on the GPU, decode assessment
- 32-token matrix-unit tiles for short inputs; fail loudly on a broken MMA library
- pad matrix-unit GEMM rows to whole tiles, 32-row tiles for MoE experts
- prefill GEMMs and MoE experts on the M5/A19 matrix units
- Core ML runtime: Qwen3.5 and Gemma 4 packages, Core ML models in the Engine API
- cargo fmt (stream trace lines)
- Core ML converter: docs for Gemma 4 and ANE findings, README section with measurements
- Core ML (ANE) converter in ANEMLL format for Qwen3 and Qwen3.5
- release Metal buffers of streamed weights, resident copies, measurements
- Weight streaming for models larger than RAM (--stream), cost-aware MTP drafts
- Fix the merge of the Core ML backend onto the engine API (Path import; clippy drop_non_drop)
- Core ML: library helpers, example, README section with measurements
- Core ML: auto prefill mode as chat.py, padded prefill option, more layouts
- Core ML backend: run ANEMLL models on the Apple Neural Engine
- Library engine API, C API (capi feature) and an iOS test app
- Qwen3.5/3.8 MTP speculative decoding (--mtp), greedy and sampled
- batch-invariant multi-token decode tiles, quantized passes in token pairs
- load ggml-quantized matrices, tables and experts in their own format
- GGUF NEON: centered nibbles through the integer activation sums; vector IQ4_XS scales
- GGUF decode: center the weights of formats with mins (Q4_K, Q5_K, Q2_K, Q4_1, Q5_1)
- NEON dequantization for the prefill path (K-quants, IQ2/IQ3, IQ4_XS)
- GGUF F32/F16/Q4_1/Q5_0/Q5_1 GEMM as well
- prefill GEMM over GGUF weights (K-quants, i-quants, Q8_0, Q4_0)
- GGUF weight formats: llama.cpp block types as a Weight / Table kind
- Zawen-v0.1-35B-A3B checked on its real weights; GGUF: expose FastMTP d2t maps
- GGUF reader: qwen35 / qwen35moe / qwen3 files with their tokenizer, config and chat template
- Qwen3.5 MoE (Qwen3.6-35B-A3B, Zawen): routed + shared experts, grouped prefill matmuls
- Library API: hf::set_hub, Embedding::truncate, Embedder::sparse_tokens; embed and sd_lora examples
- Stable Diffusion: LoRA / LoCon merging (--lora FILE[:SCALE]), checked against diffusers
- ModernBERT (modernbert-embed, gte-modernbert)
- NomicBERT (nomic-embed-text v1 and v1.5)
- MPNet (all-mpnet-base-v2)
- MiniLM, bge-small, e5 and mxbai embedders checked against sentence-transformers
- EmbeddingGemma, Qwen3-Embedding, BGE-M3 (dense, sparse, ColBERT)

## [0.0.3](https://github.com/KarpelesLab/oxinfer/compare/v0.0.2...v0.0.3) - 2026-10-24

### Other

- MLX checkpoints, the mirror, Gemma 3 and Qwen3; Qwen3 tool-call test
- convolution weights are channels-last; permute them for the Gemma 4 audio tower
- MLX models and the ATOnline mirror
- CUDA and WebGPU: affine Q4 GEMMs for MLX weights
- affine Q4 GEMM for MLX weights; vectorized bf16 dot fallback
- MLX checkpoints, the ATOnline mirror, Qwen3 and Gemma 3

## [0.0.2](https://github.com/KarpelesLab/oxinfer/compare/v0.0.1...v0.0.2) - 2026-10-24

### Other

- reasoning-effort instruction only for templates that have it
- attention and fused group/layer norms; GPU check summary
- attention and fused group/layer norms
- Diffusion on the GPU: attention, and group/layer norms fused into their consumers (Metal)

## [0.0.1](https://github.com/KarpelesLab/oxinfer/compare/v0.0.0...v0.0.1) - 2026-10-24

### Other

- rustfmt src/web/mod.rs
- Browser chat: tools (date/time, sandboxed JavaScript) and an environment-aware system prompt
- Tool calling and a default system prompt (CLI)
- Image generation in the browser: WebGPU convolution, SD exports, Image mode on the site
- implicit-GEMM convolution kernels (conv2d backend hook)
- Diffusion speed: token-split matmul driver, cache-blocked prefill tiles, Metal implicit-GEMM convolution
- Image generation: Stable Diffusion 1.x / 2.x and SD-Turbo from diffusers folders
