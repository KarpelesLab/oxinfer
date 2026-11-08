# llama-modchip: speculative drafting ("RouteCrack + Stockfish-PVS"), HARP-Cache, EAKV

Dirty-room analysis for a cleanroom reimplementation. Everything below is described in prose,
tables and pseudocode. No source code has been copied. File:line references point into the
upstream repo (`speculative/`, `kv_cache/`, `gateway/`) so a reviewer can check them; the
implementers should not open those files.

---

## 0. Executive summary

| Component | What it actually is | Wired into the server? | Lossless? |
|---|---|---|---|
| "RouteCrack + Stockfish-PVS v2" drafter (`speculative/cpp/ngram-map.cpp` + `rc_predictor.h` + `rc_trie_v2.h` + `rc_stockfish.h`) | A **draft-model-free, single-chain** token proposer. It does prompt-lookup (in-context suffix match) plus a stack of n-gram statistics (offline hash tables, online trie/PPM/skipgram, unigram "hot rules"), gated by hand-tuned confidence thresholds. A small session hash table remembers verification outcomes. **There is no tree, no PVS, no alpha-beta, no search.** The chess names are just labels for the heuristics. | **Yes.** `scripts/build_zen5_llama.sh:36-40` copies the files over llama.cpp's `common/ngram-map.cpp` (pinned commit d2f83055), replacing `common_ngram_simple_draft()`. The server turns it on with `--spec-type ngram-simple --spec-ngram-simple-size-m 7` (`systemd/nemotron-256k.service:66-69`, `install.sh:341-344`). It also runs as the fallback inside `common_ngram_map_draft` (ngram-map-k mode). | **Yes**, because llama-server's own verifier checks every draft. The drafter only affects speed. (The Fedora and Apple-Silicon scripts use `ngram-mod` instead, which does not reach this code.) |
| `rc_engine.cpp` | A standalone CLI benchmark harness built on the llama.cpp C API. It uses chain drafting and batch verification. | No (separate binary) | **Only with `--greedy`.** By default it verifies against plain argmax but samples the other tokens with a repetition penalty (and a temperature if one is set), so its output does not match baseline decoding. |
| `rc_predictor_lib.cpp` / `predictor.py` / `engine.py` | A ctypes .so plus a PyTorch/HF prototype. It has an "SGLang-compatible" tree batch API. | No. Nothing in the repo connects it to SGLang or llama-server. | engine.py is greedy-only. It has a KV double-feed bug and counts fallback tokens as accepted (§6). |
| Linear / logit probes (`tools/`, `rc_probe_extract*.cpp`) | Offline classifiers that predict token t+1 from the hidden state or logits at step t. That is exactly what the LM head already does, so they add no look-ahead. | No. `ngram-map.cpp` never calls them. | n/a |
| HARP-Cache (`kv_cache/harp_cache.py`) | A Python radix tree over 32-token blocks, backed by a sparse /dev/shm file. **It never stores KV tensors**: each page gets a 32-byte header, or "simulated" bytes in tests. | The gateway imports it, but **only as telemetry**. It is fed with Python `hash()` values of 4-character text chunks, not real tokens, and it reports a made-up "prefill saved" figure. The real prefix reuse comes from llama-server (`--cache-reuse 256`, `--slot-save-path /dev/shm`, the slot-0 snapshot restore). | n/a (it does nothing to inference) |
| EAKV (`kv_cache/eakv_quantizer.py`, `llama_kv_cell_rc.h`) | A NumPy simulator of mixed FP16/Q4 KV storage per token. Its "entropy" classifier is a pseudo-random rule: token id below 500, or a hash mod 10 below 7. | No. Production uses llama.cpp's stock `--cache-type-k q8_0 --cache-type-v q4_0`. The C header is not included anywhere. | Would be lossy (quantization) |
| `dram_kv_profiler.py` | An analytical model whose "speedup" includes a hard-coded ×2.4 speculative multiplier and a ×0.7 compute fudge. Its "conclusions" are fixed strings. | No | n/a |

---

## 1. Integration point (production path)

1. The build script checks out llama.cpp at a pinned commit, applies the project's kernel patch,
   then **overwrites** `common/ngram-map.cpp` and adds the three headers
   (`scripts/build_zen5_llama.sh:5-6, 36-40`).
2. Upstream llama.cpp has an `ngram-simple` speculative type. It calls
   `common_ngram_simple_draft(config, inp, sampled)` with:
   - `inp`: the slot's committed token history, **excluding** the token just sampled;
   - `sampled`: the token just sampled, not yet decoded;
   - `config.size_mgram`: the maximum draft length (`--spec-ngram-simple-size-m`, 7 in the shipped service);
   - `config.size_ngram`: `--spec-ngram-simple-size-n` (4). The replacement **ignores this value.**
   The function returns a vector of 0 to `size_mgram` draft tokens.
3. The upstream server verifies the draft as a batch `[sampled, d0..dk]`. It keeps the longest
   prefix that matches what the target's sampler produces, then adds the correction or bonus
   token. Mamba-2 rollback uses the server's own checkpoints (`--ctx-checkpoints 64`). This
   verification is unchanged, so output is identical to non-speculative decoding. The drafter
   only changes how many tokens are accepted per forward pass.
4. The replacement keeps upstream's `common_ngram_map_*` (the ngram-map-k implementation) and
   adds one change: if the map produces no draft, it calls the RouteCrack simple-draft
   (`ngram-map.cpp:982-986`). In that call it passes `size_key` as `size_ngram` and
   `size_value` as `size_mgram`.
5. Global state: a single process-wide engine object (a function-local static), guarded by a
   mutex. Tables are loaded by a static constructor when the library loads
   (`ngram-map.cpp:401-408`). The state is **not per slot**. The shipped config uses `-np 1`,
   but with several slots the incremental-learning bookkeeping (`last_inp_size`,
   `last_draft`) would mix up streams.
6. Live tuning: on Linux it optionally mmaps `$XDG_RUNTIME_DIR/llama_modchip.bin` or
   `/dev/shm/llama_modchip.bin`. Checks before use: magic number "MODC", same owner uid,
   regular file, and O_NOFOLLOW on open. The block holds an enable flag and three certainty
   gates (d0/d1/d2), plus several kernel knobs that are not used here. If the file is absent,
   the environment variables `LLAMA_RC_ENABLED` (default 1) and `LLAMA_RC_CERT_D0/D1/D2`
   (default 84/92/96) apply (`ngram-map.cpp:40-95, 425-454`).

---

## 2. Data structures (production drafter)

All hashing of token n-grams uses **FNV-1a 64-bit over the 32-bit token ids**. Each id is
XORed in as a whole 32-bit word, not byte by byte, and then multiplied by the FNV-64 prime.
The standard FNV-64 offset basis is used. Call this `H(tokens[a..b])`.

### 2.1 Offline hash cascade (`hc_top4_k{10,8,6,5,4,3,2,1}.bin`)
- One table per context length k in {10, 8, 6, 5, 4, 3, 2, 1}. Tables are kept sorted by k,
  longest first (`rc_predictor.h:1339-1353`).
- File v2 layout:
  - 16-byte header: entry count (u32), k (u32), number of candidates per entry (u32, value 4),
    reserved (u32).
  - Then one 40-byte entry per context: 64-bit hash, 4 predicted token ids (i32), 4
    confidences (i32, 0..100).
  - Entries are sorted by hash, and lookup is a **binary search on the full 64-bit hash**. No
    context tokens are stored, so a 64-bit collision would go undetected, which is negligible.
  - A legacy v1 layout also exists: an 8-byte header and 16-byte entries (hash, one prediction,
    one confidence). The loader picks the format from the file size (`rc_predictor.h:321-445`).
- Builder (`rc_build_tables.cpp:62-128`):
  1. For every position in the training corpus, count (hash of the previous k tokens) →
     next-token frequencies.
  2. Keep a context only if it has at least min_obs observations: 10 for k=1, 6 for k=2,
     4 for k≤4, otherwise 3.
  3. Confidence is a Laplace-damped percentage: c_i = floor(100·count_i / (total + prior)),
     with prior 3 for k≤2, 2 for k≤4, otherwise 1.
  4. Drop the context if the top-1 confidence is below 40. Store the top 4 candidates; any
     unused slot gets id -1 and confidence 0.
- At load time, n_preds is the number of leading candidates with confidence > 0.

### 2.2 Hot rules (`hot_rules.bin`)
- Sorted array of 12-byte records: predecessor id (i32), predicted id (i32), confidence (u16),
  padding (u16). It follows a 16-byte header.
- Builder (`rc_build_tables.cpp:130-175`):
  1. Find the 64 most frequent corpus tokens; these are the "hot" tokens.
  2. For each predecessor token, take its most frequent follower.
  3. Keep the rule if the predecessor has at least 10 occurrences, the follower is hot, and
     floor(100·best/(total+3)) ≥ 65.
- Lookup is a binary search by predecessor. (The docs say "300 rules" and "top-48". The code
  uses the top 64, and the rule count depends on the corpus.)

### 2.3 Macro codebook (`macro_codebook.bin`)
- Header: count (u32), max_cont (u32, written as 15).
- Each entry: hash (u64), cont_len (u16), conf (u16), then max_cont continuation ids (i32 each).
  With max_cont = 15 that is 72 bytes per entry.
- Builder (`rc_build_tables.cpp:177-244`):
  1. For context lengths k in {6, 5, 4}, map the hash of the previous k tokens to counts of
     the exact next-4-token sequence.
  2. Keep it if it has at least 5 observations and floor(100·best/(total+2)) ≥ 82. Store
     cont_len = 4.
  3. Hashes from the different k values share one sorted array. On a duplicate hash, the
     entry with the higher confidence wins.

### 2.4 Online learners (state is never reset across conversations)
- **Trie** (`rc_trie_v2.h`):
  - Node pool: a vector of 4,194,304 nodes, allocated up front (`RC_TRIE_CAPACITY`,
    `rc_predictor.h:38`). Each node holds a token, a frequency, a parent pointer and a hash map
    from child token to child pointer. That is roughly 70+ bytes per node, so about 300 MB
    resident before any child-map heap use.
  - Max depth constant: 18.
  - The production learner inserts only paths of depth 3..10 that end at the newest token
    (`ngram-map.cpp:229-246`).
  - Frequency rule: a newly created node starts at freq 1. An existing node is incremented only
    when it is the final token of the inserted path.
  - "Predict confident" (`rc_trie_v2.h:158-190`):
    1. For each suffix start, from up to 18 tokens back down to 1, walk the context suffix from
       the root.
    2. Keep the deepest fully matched node that has children.
    3. Return the child with the highest frequency, with confidence = best_freq / sum_freq,
       observations = sum_freq, and depth = suffix length.
- **PPM** (`rc_predictor.h:481-618`):
  - Five open-addressed tables, one per context order k = 2..6, each with 2^18 slots of about
    40 bytes (about 52 MB in total).
  - Entry: 64-bit context hash (0 means empty), up to 4 predictions with u16 counts kept sorted
    by insertion-bubble, total count (u16), n_preds.
  - Probing: linear, up to 8 slots.
  - Predict: for each order k from 6 down to 2, take the entry if its total is ≥ 2, and add
    2^k · (count/total) to each of its candidates. Candidates accumulate across orders, then are
    sorted by score.
- **Skipgram** (`rc_predictor.h:624-678`):
  - Key: (t-3, t-1). Table of 2^17 slots × 16 bytes (hash, prediction, hits u16, total u16),
    linear probing up to 8.
  - The prediction is the **first** token ever seen for that key. It is never replaced.
  - It is returned once total ≥ 2.
- **Session TT, the "Stockfish" part** (`rc_stockfish.h`):
  - Three direct-mapped tables, tt5, tt4 and tt3, with 4096 buckets each. Each bucket is
    16 bytes, aligned to 16:
    - key32 (u32): the upper half of the 64-bit hash;
    - best_move (i32);
    - refuted_move (i32);
    - history (i16, clamped to ±512);
    - wins (u8, saturating at 250);
    - total (u8, saturating at 250).
  - Size: 192 KB in total.
  - Context hash over the last k tokens (k = 5, 4, 3): XOR over positions i of
    SplitMix64-finalizer((token << 4) | i). The result is forced nonzero.
    Bucket index = hash & 4095. Tag = upper 32 bits.
  - `observe(context, actual, drafted)` updates each of tt5/tt4/tt3 for the context that
    precedes `actual`:
    - **Tag mismatch** (replace the bucket):
      - set best = actual;
      - if drafted is set and differs from actual, set refuted = drafted, otherwise refuted = -1;
      - history = -128 if it was refuted, otherwise +96;
      - wins = 1, total = 1.
    - **Tag match:**
      - total += 1.
      - If best == actual: wins += 1; clear refuted if it equals actual; history gets a
        "gravity" update with bonus +128.
      - Otherwise:
        - refuted becomes drafted if this was a refutation, else the old best;
        - best = actual;
        - wins = 1;
        - history gets a gravity update with bonus -192.
    - Gravity update: h ← clamp(h + b − h·|b|/512, −512, 512), using integer division.
  - Probes return the bucket only when the tag matches.

### 2.5 Warm start
- At init, the engine reads the **last 80,000 tokens** of `corpus_nemotron_warm.bin` (flat
  int32 tokens).
- For each position it runs the online learners on a 12-token window and calls the TT observe
  with no draft (`ngram-map.cpp:136-160`).
- `install.sh:194-213` builds the training corpus from **the repository's own source and docs
  concatenated 5 times** (plus an optional user code directory) and tokenizes that. The warm
  corpus is a **copy of the training corpus**. The repetition makes every n-gram appear at
  least 5 times, which trivially clears the min_obs thresholds and inflates confidences. The
  "3.7 M Nemotron tokens" in the comments is not what an install produces.

---

## 3. The drafting algorithm (`common_ngram_simple_draft`, `ngram-map.cpp:418-762`)

Notation:
- N = len(inp).
- p0 = sampled, p1 = inp[N-1], p2 = inp[N-2].
- `W` = a 28-slot scratch window. It starts as the last 11 tokens of inp followed by `sampled`
  (12 tokens), and each drafted token is appended to it.

### 3.0 Preconditions
Return an empty draft in any of these cases:
- size_mgram == 0;
- N < 12;
- the drafter is disabled.

### 3.1 Online feedback (before drafting)
- If N changed since the last call, set delta = N − last_N when 1 ≤ N − last_N ≤ 8.
  Otherwise delta = 1; this also covers context shrinkage and new conversations.
- For each of the last `delta` new tokens, oldest first, with index s = 0..delta-1:
  1. Feed a 12-token window ending at that token into the trie, PPM and skipgram.
  2. Call TT observe with drafted = last_draft[s].
  The newest token is the current `sampled`; its window is W.
- **Bug (off by one), assuming upstream's history convention.**
  - After a verify step, the new tokens are: the previous `sampled`, then the accepted drafts
    d0..d(a-1), then the new `sampled` (the correction, or the bonus token).
  - The code pairs new token s with last_draft[s]. Only the last pairing is correct: the
    correction is compared with d_a.
  - The earlier pairings compare the previous sampled token with d0, then d0 with d1, and so on.
    These almost always "mismatch", so new TT buckets for accepted tokens start with history
    -128 and a spurious refuted_move, not +96.
  - The offline harness (`test_synergy.cpp:510-519`) aligns this correctly. Its numbers
    therefore do not reflect this production behaviour.
  - A cleanroom version should pair new token s with draft index s-1, and skip the TT update
    for s = 0, the previous sampled token.

### 3.2 Stage 1: in-context suffix match (prompt lookup), lines 493-654

**Candidate positions.**
- Scan j downward from N-1 to `hot_start`, where hot_start = max(3, N−16384) (the "hot zone").
- A position is a candidate if inp[j] == p0, inp[j-1] == p1 and inp[j-2] == p2. The match is
  at least 3 tokens and ends at j.
- Stop the scan after 5 accepted hits.
- If the hot zone produced nothing and hot_start > 16, scan the "prefix zone": j from
  min(hot_start−1, 24576) down to 6. Stop after 3 hits.
- (In the prefix-zone loop, the `j == 6` stop sits after the check. That is fine.)

**Match evaluation for candidate j.**
1. Extend the match length k from 3 while k < 12 and inp[j-k] == inp[N-k].
   Note the comparison is against `inp`, not the window, so it compares the history before
   p0 with the history before inp[j]. Effectively k counts the matched tokens ending at
   p0/inp[j].
2. Reject the candidate if:
   - it is in the prefix zone and k < 5; or
   - it is in the hot zone, older than 256 tokens, and k < 4.
3. Otherwise hits += 1.
4. If this is the first accepted candidate: best = (j, k) and consensus_hits = 1.
5. Otherwise, compute `agree` = the number of matching continuation tokens between
   inp[j+1..] and inp[best_j+1..]. It is capped at consensus_len (initially 7) and at N.
   - If k ≥ best_k and agree > 0: best moves to (j, k), consensus_len = min(consensus_len,
     agree), and consensus_hits += 1.
   - Else, if k+1 ≥ best_k:
     - if agree == 0 and no runner-up is recorded yet, record the runner-up = (j, k);
     - if agree > 0, consensus_hits += 1;
     - consensus_len = min(consensus_len, agree). This can reach 0.

**Conflict resolution ("multi-branch disambiguation").** This applies when consensus_len == 0
and a runner-up exists.
- c1 = the token after best, c2 = the token after the runner-up.
- For each, count how many ensemble predictors back it (the "verify" mode in §3.4).
- Probe tt4 on W.
- If c1 has ≥ 3 supporters, c2 has 0, and tt4's best is c1 with wins ≥ 2: use best, with
  consensus_len = 1.
- If the mirror condition holds for c2: switch to the runner-up, with consensus_len = 1.

**Base draft length base_m.** "Cascade loaded" is the normal case:

| best_k | Cascade loaded | Online-only (no cascade files) |
|---|---|---|
| ≥ 8 | 5 | 4 if consensus_hits ≥ 2, else 3 |
| 7 | 5 if hits ≥ 2, else 4 | 3 if hits ≥ 2, else 2 (rows ≥6 combined) |
| 5–6 | 4 if hits ≥ 2, else 3 | k=6: as above; k=5: 2 if hits ≥ 2 or within the last 1024 tokens, else 1 |
| 4 | 2 if hits ≥ 2, else 1 | 2 if hits ≥ 2; 1 if within the last 512 tokens; else 0 |
| 3 | 1 if hits ≥ 2; otherwise 1 only if within the last 128 tokens **and** tt5's best is the candidate with wins ≥ 2 and history ≥ 128; else 0 | 1 if hits ≥ 2 and within the last 256 tokens |

**Extension cap ("singular extension").** This applies only when a cascade is loaded:
- cap = 7 if best_k ≥ 10, or if best_k ≥ 8 and consensus_hits ≥ 2;
- cap = 6 if best_k ≥ 7 and consensus_hits ≥ 2;
- otherwise cap = base_m.

Then max_m = min(cap, consensus_len, size_mgram). (`best_k ≥ 10` needs a 10–11-token match,
since k stops at 12.)

**Per-token emission loop.** For i in [0, max_m), with cand = inp[best_j+1+i]. All TT probes
use the current W.
1. Break if tt3 says cand is refuted (refuted == cand, best ≠ cand, history ≤ -64), and
   best_k ≤ 5 and hits == 1.
2. If a cascade is loaded, hits == 1 and best_k ≤ 5:
   - compute the supporter count `agree` for cand, and `tt_backed` (tt5's best == cand with
     wins ≥ 2, or tt4's best == cand with wins ≥ 3);
   - break if (best_k == 4, i == 0, agree == 0 and not tt_backed);
   - break if (i ≥ 1, agree == 0 and not tt_backed).
3. If i ≥ base_m (an extension token):
   - break if tt5 or tt4 lists cand as refuted (with best ≠ cand);
   - break if fewer than `req` predictors back cand and no TT confirms it (tt5 or tt4 best ==
     cand with wins ≥ 2). req = 1 if best_k ≥ 11, else 2.
4. Append cand to the draft and to W.

Note that the TT probes are always made on W, which grows as tokens are appended. That is
correct, because each step asks about the context ending just before cand.

### 3.3 Stage 2: statistical rollout, lines 656-754

**Macro latch.**
- If the macro book is loaded, look up H(last 6 of W), then H(last 5 of W).
- Accept the first hit that has conf ≥ 92 and cont_len ≥ 3. Remember macro_start = the current
  draft length.

**Stage-2 length limit.** limit = min(size_mgram, macro ? min(4, macro_start+4) : 3). Because
the loop starts at d = the current draft length, **Stage 2 runs only if Stage 1 produced fewer
than 3 tokens (4 with a macro)**. The total draft is therefore at most 7, and Stage 2 adds at
most 3 or 4 tokens.

**Loop.** For d from the current draft length up to limit:
1. Query the ensemble in "predict" mode (§3.4). This returns (chosen, cert, casc_k, agree).
2. Macro step m = d − macro_start:
   - if chosen equals the macro continuation[m], set cert = max(cert, macro.conf + (m ≥ 1 ? 4 : 0));
   - otherwise drop the macro.
   The macro **never proposes** a token. It only raises the certainty of a token the ensemble
   already chose.
3. Probe tt5, tt4 and tt3 on W.
4. If tt5 hit:
   - If chosen is refuted there (and history < 0): replace chosen with tt5's best and set
     cert = 88 + min(8, wins) when tt5 wins ≥ 3 and history ≥ 128 (a contradictory condition,
     see the bugs list); otherwise cert -= 12.
   - Else if chosen == tt5 best and wins ≥ 2: cert += 4.
   - Else if there is no chosen token, tt5 best exists, wins ≥ 3 and history ≥ 160: if at least
     1 predictor backs tt5's best, choose it with cert = 86 + min(8, wins).
5. Else if tt4 hit:
   - refuted (history ≤ -64): cert -= 10;
   - chosen == best with wins ≥ 3 and history ≥ 160: cert += 4;
   - no chosen, wins ≥ 4, history ≥ 200 and ≥ 2 supporters: choose best with cert 86.
6. If tt3 hit and casc_k ≤ 3:
   - refuted (history ≤ -64): cert -= 10;
   - best == chosen with wins ≥ 3 and history ≥ 192: cert += 3.
7. Threshold: req = cert_d0 for d = 0, cert_d1 for d = 1, cert_d2 for d ≥ 2 (defaults 84/92/96).
8. **"1-ply quiescence".** If req−2 ≤ cert < req, tentatively append chosen and query the
   ensemble one step further. Raise cert to req if the next prediction has: casc_k ≥ 6,
   cert ≥ 95, agree ≥ 3, and tt5 (on the extended window) has best == next with wins ≥ 2.
9. If cert ≥ req, append chosen and continue. Otherwise stop.

**After drafting.** Store the first ≤ 8 draft tokens as last_draft, for the next call's
feedback.

### 3.4 Ensemble query (`query_rc_nolock`, lines 256-397)

Inputs: a context (W and its current length), and optionally a candidate to verify.

**Signals collected.**
1. **Cascade.** Use the first table, from longest k, whose hash of the last k tokens is found.
   Take its top-1 token, confidence and k. In verify mode, also note whether the candidate is
   anywhere in its top 4.
2. **Trie.** "Predict confident" gives the token, confidence, observations and depth.
3. **PPM.** Top-1 token.
4. **Skipgram.** The token for (W[-3], W[-1]).
5. **Hot rule.** Rule for W[-1] gives a token and its confidence.

**Top-4 promotion ("Technique #2").** If the cascade entry has more than one candidate, and
either top-1 is refuted or the cascade confidence is below 80:
- "Refuted" means: tt5 has refuted == top1, best ≠ top1 and history < 0; or tt4 has the same
  with history ≤ -64.
- Scan candidates 2..4 that have confidence ≥ 20. Promote the first one that meets both:
  - (tt5 best == alt with wins ≥ 3 and history ≥ 160) or (tt4 best == alt with wins ≥ 4 and
    history ≥ 200); **and**
  - the trie top equals alt with depth ≥ 4 and confidence ≥ 0.80, **and** the PPM top equals alt.
- A promoted candidate gets cascade confidence 84.

**Verify mode.** Return agree = the number of predictors backing the candidate (0..5):
- cascade: top-1, or the candidate is anywhere in its top 4;
- trie top;
- PPM top;
- skipgram;
- hot rule.

**Predict mode.**
- If the cascade matched:
  - agree = 1, plus 1 each for trie (depth ≥ 2), PPM, skipgram and hot rule agreeing with it;
  - cert = cascade_conf + (4 if k ≥ 6) + (6 if agree ≥ 3, 3 if agree == 2);
  - force cert = 0 if k == 1 and conf < 90.
- Else if the trie has depth ≥ 4, observations ≥ 3 and confidence ≥ 0.88: use the trie token,
  with cert = 100·confidence.
- Else if the hot rule has confidence ≥ 90 and PPM or skipgram agrees: use the hot token, with
  cert = hot confidence.

**Trie rescue ("Technique #1").** Applies if cert < 84 and the trie has depth ≥ 4,
observations ≥ 4 and confidence ≥ 0.92.
- Override with the trie token when it is **not** refuted in tt5, tt4 or tt3 (tt3 also needs
  history ≤ -64), and one of these holds:
  - (tt5 best == token with wins ≥ 2, or tt4 best == token with wins ≥ 3) and t_agree ≥ 2; or
  - depth ≥ 6, observations ≥ 5 and t_agree ≥ 3.
- t_agree = 1 + PPM + skipgram + cascade (the token is the cascade top-1 or in its top 4).
- New cert = min(95, 92·confidence + 4 if TT-confirmed).
- The docs say this "fixes shadowing". In the code it is an extra override that runs after the
  normal branch.

### 3.5 What is not there
- No tree, no beam, no alternative branches sent for verification. The draft is one linear
  chain.
- No minimax, PVS, aspiration windows or late-move reductions. The doc's "countermove table",
  "continuation history (cont_hist)" and "LMR" (`ROUTECRACK_TECHNIQUES.md:114`, listed as
  "Active") **do not exist anywhere in the code**.
- No probability model and no expected-gain computation. Every gate is an integer threshold
  tuned by hand.
- The draft length does not adapt to measured acceptance in the server path. The only
  adaptation is the TT history.

---

## 4. Measured vs claimed acceptance

- **Offline "held-out" ablation** (`ROUTECRACK_TECHNIQUES.md` §4A; `test_synergy.cpp`).
  - Method: teacher-forced replay over a token file. A draft token counts as "accepted" if it
    equals the next corpus token (`test_synergy.cpp:498-507`). No model is involved.
  - Claimed: 12,230 drafted, 9,676 accepted (79%) over 77,456 tokens. That means drafts fire
    rarely. Simple accounting gives tokens per target step ≈ 77,456 / (77,456 − 9,676) ≈ 1.14
    before any verification overhead, so the corpus-level speedup potential is about 14%.
  - The "WallGain = A − 0.3·D" and "MambaGain = A − 0.5·rollbacks" metrics use **made-up
    constants**. They are not measured costs.
  - The +223 accepted tokens for "v2" is +0.29% of the tokens.
  - **The harness cannot be reproduced as shipped.** The file has no `main` (only `old_main`)
    (`test_synergy.cpp:532-545`).
  - The harness logic **differs from production**:
    - its hot zone is 1024 tokens, not 16384;
    - it has no prefix zone;
    - its base_m table is smaller (max 4, versus 5 in production);
    - it has no best_k ≥ 10 extension;
    - its TT feedback is aligned correctly, whereas production is off by one (§3.1).
    So the numbers in the docs are not numbers for the shipped drafter.
  - The test-corpus files themselves (`corpus_nemotron_test.bin`) are not in the repo.
- **Per-technique "oracle precision" claims** (92.19% for trie rescue, 91.8→96.4% macro chain,
  98.2–98.7% for extension steps 6–7). No code in the repo produces these breakdowns.
- **Live telemetry** (60.4 tok/s peak, 96.4% acceptance on in-context refactoring; 68% at
  27.7K context; 85% on Italian prose with mean accepted length 2.48).
  - These are llama-server `draft_n` / `draft_n_accepted` counters, quoted in the docs; the
    gateway UI also shows them (`sse_gateway.py:1329-1334`).
  - They cannot be verified from code. They are plausible for prompt-lookup on code-editing
    tasks, where the model copies the file it has just read, and that kind of workload drives
    the high numbers.
  - Note that `LLAMA_MOE_TOP_K=4` in the same service file (a kernel patch, out of scope here)
    reduces active experts. That change is **lossy** and confounds every tok/s comparison.
- "192 KB, 100% L2-resident, < 3 µs/step" refers to the TT only. Each draft call also touches:
  - the trie: a 4M-node pool, about 300 MB, with hash-map children;
  - PPM: about 52 MB;
  - skipgram: 2 MB;
  - cascade tables: tens of MB;
  - and up to 16K tokens of history in the hot-zone scan.

---

## 5. Bugs and risks in the production drafter

1. **Trie overflow crash** (`ngram-map.cpp:236-239`).
   - The production incremental learner allocates trie nodes but never evicts. The eviction
     logic lives only in the unused trie insert path.
   - It also does not check for a null allocation. Once the 4,194,304-node pool is full,
     allocation returns null and the code dereferences it.
   - Novel text creates roughly 10 nodes per token: about 2 for the depth-3 path plus 1 per
     deeper level. The pool therefore fills after a few hundred thousand novel tokens, counting
     the 80K warm-up. The state persists for the life of the process, so a long-running server
     should eventually crash.
2. **TT feedback off by one** (§3.1). The verification signal is partly corrupted.
3. **Contradictory condition in Stage 2.** The tt5 "refuted, switch to best_move" branch
   requires history < 0 and history ≥ 128 at the same time (`ngram-map.cpp:694-697`). The
   switch can never happen; only the cert -= 12 path runs.
4. **Shared global state across slots.** The engine is one process-wide object. Multi-slot use
   mixes the history-length bookkeeping and the last draft.
5. **Ignored parameter.** `size_ngram` (`--spec-ngram-simple-size-n 4`) has no effect. The
   minimum match is hard-coded to 3/4/5 tokens.
6. **Skipgram never updates its prediction.** It returns the first follower ever seen once that
   key has 2 or more observations, regardless of hit rate.
7. **Inflated confidences from the default corpus.** The training corpus is the repo text
   repeated 5×, and the warm corpus equals the training corpus.

None of these affect output correctness, because the server verifies every draft. They affect
speed, and bug 1 affects stability.

---

## 6. Offline / experimental pieces

### 6.1 `rc_engine.cpp` (standalone CLI; header at lines 1-14 admits it is a harness)
- **Loop per step:**
  1. **"Bypass" phase.** Chain up to 20 tokens from trie predictions with confidence ≥ 0.95,
     observations ≥ 5 and depth ≥ 3. Despite the name, these tokens **are** verified, in the
     same batch.
  2. **Macro phase.** If the bypass phase produced nothing, take the macro continuation (orders
     6..3, conf ≥ 50) when it is at least 2 tokens.
  3. **Extension up to depth D.** Each step tries, in order:
     - a 3-token match in the last 100 tokens;
     - a k=1 cascade entry with confidence ≥ 95;
     - otherwise the top-1 of the combined top-K scorer.
  4. **Verify.** One batch [last, drafts…]. Compare each position's **plain argmax** with the
     draft and stop at the first mismatch. The correction comes from the configured sampler
     chain. Then remove the rejected KV range.
  (`rc_engine.cpp:377-584`)
- **Adaptive D:**
  - Keep running counts of accepted and drafted tokens, halved every time drafted passes 50.
  - Acceptance > 0.70 sets D = 8; acceptance > 0.30 sets D = 1; otherwise D = 4. That
    ordering is odd: low acceptance gets a longer draft than medium acceptance.
  - The counters are function-statics, so they leak across prompts.
- **Not lossless by default.**
  - The default sampler has repetition penalty 1.1, and temperature can be set.
  - Verification uses argmax without the penalty, while non-draft tokens use the penalized or
    stochastic sampler. The output distribution therefore differs from baseline.
  - It is lossless only with `--greedy`.
- **The BFS tree builder is never called** (`rc_build_tree_bfs` is unused in rc_engine). The
  claim "Multi-Branch BFS Speculative Tree (rc_engine.cpp) Active" is false.
- Expert pruning (`--expert-used N`) overrides the expert count in GGUF metadata. That is
  **lossy**.

### 6.2 `rc_predictor_lib.cpp` + `predictor.py` + `engine.py`
- C ABI for create/load/learn/predict, plus a tree builder.
- `rc_batch_get` (`rc_predictor_lib.cpp:164-351`) returns, per sequence, a D-slot tree and a
  D×D ancestor mask. The tree is built like this:
  1. Depth-1 candidates are collected, deduplicated, from: a structural repeat; the cascade
     top 4; the trie top 4; the hot rule; the PPM top 2. At most K are kept.
  2. Each branch is then extended round-robin with the combined scorer's top-1.
- Fixed arrays of 64 nodes mean D > 64 overflows.
- **Nothing in the repo calls it from a server.**
- `engine.py` is an HF/PyTorch prototype (Qwen2.5-7B) with greedy chain verification.
  - Its "fallback" path counts the model's own correction token as an "accepted" draft when that
    token happens to be among the K candidates. This inflates acceptance.
  - When the follow-up extension is accepted, it keeps a KV that already contains the extension
    token and then feeds that token again on the next step. That is a duplicate KV entry and a
    **correctness bug** (`engine.py:206-221`).

### 6.3 Probes (`tools/train_*probe*.py`, `rc_probe_extract*.cpp`, `RCLogitProbe`/`RCOnlineProbe` in `rc_predictor.h:915-1223`)
- Training pairs are (hidden state or hot-token logits at position i, corpus token i+1). See
  `rc_probe_extract_logits.cpp:128-131` and `rc_probe_extract.cpp:112-115`.
- That is the same target the model's LM head predicts from the same inputs. The probe is a
  weaker copy of argmax over 48–200 tokens and offers **no look-ahead** for drafting.
- `linearize_decision.py` (comments in Italian) measures mutual information.
- `probe_to_cascade.py` distils probe outputs into extra hash entries.
- None of this is loaded by the production drafter.

### 6.4 `corpus_builder.py`
Generates corpora by prompting Ollama or SGLang with fixed coding prompts, tokenizes them with
an HF tokenizer and writes cascade tables. It is offline only.

---

## 7. Gateway caching / prefix reuse (`gateway/sse_gateway.py`)

What actually gives "zero prefill":
1. **llama-server native** features:
   - `--cache-reuse 256` (KV shifting to reuse a matching prefix);
   - `--slot-save-path /dev/shm` with slot save/restore;
   - `--ctx-checkpoints 64 --checkpoint-min-step 128` (Mamba state checkpoints).
2. **Slot-0 auto-restore** (`sse_gateway.py:66-122, 1765-1768`).
   - A pre-built snapshot file (default `claude_40tools_slot0.bin`, about 22.2K tokens of
     system prompt plus tool schemas) is copied from disk to /dev/shm if needed.
   - The gateway queries `GET /slots`. If slot 0 is idle and holds fewer than 21,500 tokens
     (2,500 for "dsv4"), it calls `POST /slots/0?action=restore`.
   - It triggers on short conversations, after a non-agent chat request, or when ≥ 20 tools are
     sent.
   - Correctness is preserved, because llama-server compares the restored tokens with the new
     prompt and keeps only the common prefix.
3. **Prefix canonicalization** (`sse_gateway.py:1650-1680`).
   - The gateway rewrites the system block so the token prefix stays byte-identical across
     requests, which keeps the snapshot valid.
   - It may substitute its own master system prompt when `MODCHIP_OVERRIDE_SYSTEM_PROMPT=1` or
     no system prompt is given.
   - It inserts an extra system message, the "master prompt" plus extracted MEMORY.md /
     CLAUDE.md blocks, before the first turn.
   - It also strips line-number prefixes from Read tool output, so files appear verbatim and the
     in-context matcher finds more matches.
   - This changes the model's input, so it is a behaviour change, though not part of the
     inference math.

What HARP does in the gateway (`sse_gateway.py:45-55, 1772-1797, 1845-1848`):
- It builds "sim_tokens" by splitting the concatenated system, tools and messages JSON text into
  **4-character chunks** and applying Python's built-in `hash()` (masked to 31 bits).
  - That hash is randomized per process unless PYTHONHASHSEED is set, so nothing persists across
    restarts.
  - Any one-character insertion shifts every later chunk.
- It runs prefix resolve and store, logs a "ZERO-PREFILL HIT", and sets headers `X-HARP-Cache`,
  `X-HARP-Cached-Tokens` and `X-HARP-Prefill-Saved-Sec`. The "saved seconds" figure is
  cached_count/250, an invented constant.
- **It does not influence what llama-server computes.** It is pure telemetry.
- The gateway does no speculation of its own. It only shows the server's draft counters.

---

## 8. HARP-Cache algorithm (`kv_cache/harp_cache.py`), for completeness

- **Page:** 32 tokens. Key = FNV-1a 64 over the little-endian int32 byte encoding (byte-wise
  FNV, unlike the drafter's word-wise FNV).
- **Page metadata:**
  - id, hash, tokens (the full list is kept and compared on lookup, so collisions are checked);
  - "entropy" = the number of unique tokens divided by 32;
  - shm offset = id × 131,072 bytes (32 × 196,608 / 48);
  - access count, created time, last-access time.
- **Arena:** max_pages (default 4096) × 131,072 B = 512 MB, a sparse file in /dev/shm, mmapped
  shared. If that fails, it falls back to a heap bytearray.
- **Index:**
  - (a) a global dict from hash to page;
  - (b) a radix tree of nested dicts keyed by block hash, each node holding {page, children}.
- **Resolve(stream):**
  1. Walk 32-token blocks.
  2. Follow the radix child if its page tokens are equal. Otherwise **fall back to the global
     content-addressed dict, regardless of position or prefix**, and stop using the radix
     cursor.
  3. Each hit updates LRU info.
  4. Stop at the first miss and return (32·hits, pages, suffix).
  The position-independent fallback would be **wrong for real KV**, because KV depends on the
  whole prefix and on position. It is harmless only because no KV is stored.
- **Store(stream):**
  1. For each complete block: if the hash exists with equal tokens, touch it.
  2. If the hash exists with different tokens (a collision), evict that page.
  3. Otherwise allocate a page id: free list first, then the watermark, then LRU eviction by
     O(n) min over (last_access, access_count).
  4. Write either the supplied simulated bytes or a 32-byte header (hash, length, entropy×1000,
     id) at the page offset.
  5. Insert or update the radix node.
  Eviction prunes the hash from the tree: it keeps nodes that have children (with page set to
  None) and deletes empty leaves.
- A trailing partial block (fewer than 32 tokens) is never cached.

## 9. EAKV (`kv_cache/eakv_quantizer.py`, `kv_cache/llama_kv_cell_rc.h`), for completeness

- **Per-token cell, per layer and head (head_dim 128):**
  - FP16 K and V, or a "Q4": one scale per 128-vector = max|x|/7, values rounded and clamped to
    ±7, packed two per byte.
  - This is not llama.cpp's Q4_0, which uses 32-element blocks with a scale of max/−8.
  - K is stored **after RoPE**, at its absolute position (theta 1e6), which the code calls
    "VPI".
- **Classifier:** a token is "low entropy" (Q4) if its id is below 500, or if the FNV hash of
  the 5-token window mod 10 is below 7. Otherwise it is FP16. This is essentially pseudo-random,
  with about 70%+ Q4. There is no real entropy measurement.
- **Guard band:** at a Q4→FP16 transition, the 2 tokens before it (indices i−1 and i) are
  promoted to FP16.
- **Attention:** plain softmax attention over the dequantized cells, for testing cosine
  similarity against an all-FP16 baseline using random vectors.
- **Size accounting:** FP16 cell 520 B per head, Q4 cell 144 B per head. The C header has the
  same layout as a fixed-size union padded to 512 B. A packed C struct therefore saves
  **nothing**, since the Q4 variant is padded to the FP16 size.
- **The profiler's own assumption** of 60% Q4 / 40% FP16 gives about 108 KB per token, which is
  **larger than uniform Q8_0** (98 KB per token). The "3.2–4.2× speedup" table multiplies by
  2.4 for speculation and scales compute by 0.7 (`dram_kv_profiler.py:112-114`), and the printed
  conclusions are fixed text (`:133-138`).
- **Status:** a simulator. It is not used at runtime. It would be lossy if implemented.

---

## 10. Cleanroom recommendations (what is worth reimplementing)

- **Worth taking:**
  - In-context suffix lookup with a long-match requirement (≥ 3–5 tokens) and multi-occurrence
    consensus. Draft length should grow with match length and with the number of agreeing
    occurrences, up to about 7.
  - A tiny per-session "verified continuation / refuted continuation" hash table keyed by the
    last 3–5 tokens. It can veto drafts the target has already rejected in this context.
  This combination is what plausibly produces the high acceptance on code-editing workloads.
- **Optional:** offline n-gram tables (top-4 per context) as a fallback when there is no
  in-context match. In the authors' own offline numbers they contribute little.
- **Skip:**
  - the logit and hidden-state probes (no look-ahead);
  - HARP (use the engine's real prefix cache);
  - EAKV (use standard KV quantization if needed);
  - "PVS"/"LMR"/"countermove" (not implemented).
- **Do correctly:**
  - align the draft-vs-actual feedback;
  - bound or evict online structures;
  - keep state per sequence;
  - verify with the real sampler (greedy or exact-match sampling) so the result stays lossless.
