# llama-modchip: quantization formats and CPU dot-product kernels (dirty-room analysis)

Scope: `patches/deepseek-v4-flash-fractal-256k.patch` (called **DS** below) and `patches/zen5-nemotron-256k-optimizations.patch` (called **Z5**). This covers the files ggml-common.h, ggml-quants.c, ggml-cpu/quants.c, quants.h, arch/x86/quants.c, the parts of ggml-cpu.c that dispatch to the new kernels, and the conversion scripts `scripts/transcode_dsv4.c`, `scripts/transcode_dsv4_shard.py` and `scripts/build_fractal_moe_gguf.sh`.

The descriptions below are in my own words. No source is reproduced. "Lxxxx" means a line number inside the patch file.

Facts about the base:
- `dsv4-64gb-cpu.patch` and `llama-modchip-dsv4-10tps.patch` are byte-identical to DS (same md5). Only two distinct patches exist.
- Z5 targets upstream llama.cpp commit `d2f83055` (pinned in `scripts/build_zen5_llama.sh`). DS targets a fork that has `deepseek4.cpp` and `llama-kv-cache-dsv4.cpp`, but its ggml-common.h, x86/quants.c, quants.c and ggml-quants.c blob hashes are the same as upstream `d2f83055`. I checked them against upstream.
- In upstream at that commit, `GGML_TYPE_Q2_0 = 42` already exists. It is an **18-byte, 64-weight, 2.25 bpw** format: one fp16 scale plus 2-bit codes mapped to {-1,0,+1,+2}. It has CPU NEON, CUDA, Metal and Vulkan kernels. **The patch silently redefines this existing type ID with a different layout.** Everything below follows from that.

---

## 1. New/modified block format: "Fractal-64" (reuses `GGML_TYPE_Q2_0`, id 42)

Location: struct DS L1436-1444 (Z5 L9-17, same content). Quantizer DS L6810-6956. Dequantizer DS L6958-6990. ggml-quants.c is identical in DS and Z5.

### 1.1 Layout (30 bytes per 64 weights = 3.75 bpw)

| byte offset | size | field | meaning |
|---|---|---|---|
| 0 | 2 | `d` | IEEE fp16 block ("master") scale. It is **not** BF16, although the README says "BF16-scale". |
| 2 | 4 | `scales[4]` | eight 4-bit sub-scale codes, one per 8-weight sub-block |
| 6 | 16 | `qs[16]` | the low 2 bits of the 64 3-bit indices |
| 22 | 8 | `qh[8]` | the high (3rd) bit of the 64 indices |

The code statically checks that the struct is 30 bytes.

**Sub-scale nibbles.** For sub-block b in 0..3, the code is the low nibble of scales[b]. For sub-block b in 4..7, it is the high nibble of scales[b-4]. Sub-block b covers weights 8b..8b+7. The multiplier is `mult_b = 4 + code`, so it ranges over 4..19.

**Low bits (qs).** Weight j (0..63) has group g = j>>4 and lane t = j&15. Its low 2 bits are at bit position 2g of qs[t]. Each qs byte t therefore packs weights t, t+16, t+32 and t+48 at bit pairs 0-1, 2-3, 4-5 and 6-7.

**High bit (qh).** The high bit of weight j sits in qh[j & 7], at bit position 2*(j>>4) + ((j>>3)&1). Simplified, **bit b of qh byte u is the high bit of weight 8b+u**. If qh is read as a 64-bit little-endian word, it is an 8x8 bit-matrix transpose of the weight order. The SIMD kernel relies on this.

**Index to value.** idx = low2 | (high << 2), in 0..7. The value comes from a fixed 8-entry symmetric codebook with **no zero level**: idx 0..7 maps to -14, -8, -4, -1, +1, +4, +8, +14. The high bit is effectively the sign: 0 means negative, 1 means positive. The low 2 bits select a magnitude from (14, 8, 4, 1) when negative, or (1, 4, 8, 14) when positive. Both are monotonic in idx.

**Dequant.** `w_j = fp16_to_f32(d) * (4 + code_{j>>3}) * codebook[idx_j]`.

### 1.2 Quantizer algorithm (DS L6822-6956)

The quantizer works on each 64-weight block:

1. **Per-sub-block optimal scale** (8 sub-blocks of 8 weights):
   - Start with s = amax/14.
   - Run 2 rounds of fixed-assignment least squares. In each round, assign every weight to its nearest codebook entry after dividing by s, then set s = sum(x*q)/sum(q*q). Only accept the new s if both sums are positive.
   - Nearest-entry selection is a branch tree over the midpoints ±2.5, ±6, ±11 and 0 (DS L6810-6820).
2. **Master scale.** d = max_b(s_b) / 19. Round-trip it through fp16 and use the rounded value from then on. If d <= 0 the block is all-zero bytes. Note that the code is still not zero: it decodes to ±1 * d * 4 = 0 because d = 0.
3. **Sub-scale code search.**
   - Estimate code = round(s_b/d) - 4, clamped to 0..15.
   - Search exhaustively over code-2 .. code+2 (clamped).
   - For each candidate, re-assign all 8 weights to nearest entries and compute the squared error. Keep the best.
4. **Pack** the nibbles and indices as described in 1.1.

Properties and issues:
- **No importance weighting.** The imatrix is ignored. Upstream's `quantize_q2_0` wrapper calls the ref function unchanged, so `--imatrix` has no effect on these tensors, even though the build script accepts it.
- **No zero level.** Exact zeros and tiny weights become ±d*mult.
- **Limited sub-block range.** Sub-block scales can only differ by a factor of 19/4 = 4.75 within a block. A sub-block whose optimal scale is below 4d is clamped to mult = 4 and over-scaled.
- **Not lossless.** At 3.75 bpw this is a real quality trade-off. The bpw is close to IQ3_S (3.44) or Q3_K (3.44), but those use a 256-element super-block, which Nemotron's dims (multiples of 64 but not of 256) can't use. That is the stated motivation for a 64-block format.
- **"Fractal" is a marketing name.** The codebook is a hand-picked non-uniform 8-level set. Nothing about it is self-similar.

### 1.3 Compatibility consequences (serious)

**Type-ID collision with upstream Q2_0.** The patch keeps `GGML_TYPE_Q2_0 = 42` and the name "q2_0", but `sizeof(block_q2_0)` goes from 18 to 30. GGUFs from either build cannot be read by the other. The tensor byte sizes disagree, so you get a load failure rather than silent garbage.

The docs disagree with the code and with each other on the ID:
- README L109 says "type ID 40" and "BF16". Upstream 40 is NVFP4.
- `transcode_dsv4.c` says "type 41" in a comment. Upstream 41 is Q1_0.
- The python driver writes 42, which is correct.

**Non-x86 CPU backends are broken.** Upstream `arch/arm/quants.c` has its own NEON `ggml_vec_dot_q2_0_q8_0` written for the old 18-byte layout. Neither patch touches it. On ARM (Apple Silicon, Graviton), single-column Q2_0 dot products decode the new 30-byte blocks with the old layout and produce garbage. The README nonetheless says "Linux/macOS/Windows".

The upstream Metal, CUDA and Vulkan Q2_0 kernels are also untouched. They are old-layout too, so offloading a Q2_0 tensor to a GPU is also wrong.

**DS does not link on non-x86.** `ggml-cpu.c` references the iq1_s, iq2_xxs, iq1_m, Q5_K and Q6_K `_1xN` kernels unconditionally (dispatcher at DS L4309-4361). Those symbols are only defined in `arch/x86/quants.c`. The Q2_0 `_1xN` kernels are in the generic `quants.c`, so they do resolve.

---

## 2. Q2_0 (Fractal-64) dot-product kernels: activation type Q8_0

The activation (vec_dot_type) is unchanged from upstream: **Q8_0**. One Q2_0 block (64 weights) pairs with two consecutive Q8_0 blocks (2 x 32), each with its own fp16 scale.

The kernels live in `ggml-cpu/quants.c`, inside the function upstream calls `ggml_vec_dot_q2_0_q8_0_generic` (DS L6583-6735). On x86, upstream's arch-fallback header renames `_generic` to the real symbol, so this "generic" function becomes the x86 kernel. It contains three paths.

### 2.1 AVX-512F+BW single-column path (DS L6586-6637): the "vprorvq" kernel

All 64 weights of a block are handled in one 512-bit register. Byte p of the register holds weight p.

**Low bits.**
- Broadcast the 16 qs bytes into all four 128-bit lanes.
- Apply a per-dword variable right shift (`vpsrlvd`). Lane L shifts by 2L, giving 0, 2, 4, 6 for the four lanes.
- AND with 0x03 per byte.
- Result: byte p = low 2 bits of weight p, because lane L byte t is weight 16L+t. Bits that bleed in from the neighbouring byte land above bit 1 and are masked off.

**High bits: the rotate trick.**
- Broadcast the 8 qh bytes as one 64-bit value into all eight qwords.
- Apply a **per-qword variable rotate right** (`vprorvq`, `_mm512_rorv_epi64`). The rotate counts for qwords 0..7 are 62, 63, 0, 1, 2, 3, 4, 5, i.e. (q-2) mod 64.
- AND with 0x04 per byte.
- Qword q covers weights 8q..8q+7, so byte u of qword q needs bit q of qh[u]. The rotate moves bit (8u+q) to bit (8u+2) with no cross-byte contamination at that bit position.
- One rotate per qword therefore implements the bit transpose. A rotate is needed rather than a shift because qwords 0 and 1 need a left move and the rest need a right move.

**Index and magnitude.**
- idx = low | high gives 0..7 per byte.
- `vpshufb` against a 16-byte table replicated per lane gives |value|. The table entries are 14, 8, 4, 1, 1, 4, 8, 14 in positions 0..7.

**Sign handling.**
- Build a 64-bit mask with `vptestnmb`: set where the high bit (bit 2 of the index byte) is 0, which means the weight is negative.
- Negate the matching activation bytes with a masked subtract-from-zero.

**Multiply and accumulate.**
- `vpmaddubsw(|q|, ±q8)` produces 32 int16 values, each the sum of 2 weights.
- `vpmaddwd` with a per-int16 multiplier vector produces 16 int32 values, each covering 4 weights. This is where the 4-bit sub-scale is "fused into vpmaddwd".
- The multiplier vector is built from the nibbles:
  - Take the 4 scale bytes. Low nibbles + 4 go to bytes 0-3 and high nibbles + 4 go to bytes 4-7 of a 64-bit word. Byte s of that word is mult_s.
  - Broadcast the word and apply `vpshufb` so that int16 lanes 4s..4s+3 hold mult_s with a zero high byte. The 0x80 shuffle index produces the zero.
- Overflow is impossible: 2 * 14 * 127 = 3556 fits int16, and the product with mult <= 19 is formed in int32.

**Scaling to float.**
- Build one float vector whose low 8 lanes are d * d_y0 and high 8 lanes are d * d_y1, using a masked broadcast with mask 0xFF00.
- Convert the int32 sums to float and FMA them into one 512-bit accumulator.
- A single horizontal reduce happens at the end.

Correctness: I traced the bit and lane mapping and it agrees with the scalar reference and the dequantizer. The rotate counts, shift counts, LUT order and shuffle control are all consistent.

### 2.2 AVX2 path (DS L6638-6700; Z5 L867-930)

Same idea at 256 bits, with two halves of 32 weights each.
- The qs broadcast is shifted per 128-bit lane by 0/2 and then 4/6.
- **DS**: the high bits come from four scalar shifts of the qh word, packed into one vector: left 2, left 1, 0, right 1 for the first half, and right 2..5 for the second half.
- **Z5**: the high bits come from a `vpsrlvq` of the qh word pre-shifted left by 2 (counts 0..3), and a `vpsrlvq` of the raw qh word (counts 2..5). This is equivalent; plain shifts are enough because all counts are non-negative.
- A signed LUT holds the full values -14..14. The upstream sign trick then applies: unsigned |q| times the activation with q's sign applied (`vpsignb`).
- The multipliers come from one dword of 4 nibbles (+4) per half, spread to int16 lanes by `vpshufb`.
- Two accumulators, one per Q8_0 scale.
- Correct per my trace.

### 2.3 Scalar fallback (DS L6701-6735)

A straight implementation of the dequant formula, computed per Q8_0 half with integer accumulation. This is the reference.

### 2.4 Z5 differences

- Z5 contains a second AVX-512 variant under `#if 0` (Z5 L814-866), so it is **dead code**. It builds the high-bit vector with 16-bit shifts instead of the rotate trick. I traced it and believe it is also correct; it is simply disabled.
- With Z5, Q2_0 therefore runs **AVX2-only, single-column**.
- Z5 has **no** multi-column kernels at all.

### 2.5 Multi-column Q2_0 kernels (DS only, L6398-6582)

There are variants for 8, 4, 3 and 2 columns. Two macros contain the common work:
- An "unpack step" decodes one weight block into |q|, the negative mask and the multiplier vector, exactly as in 2.1.
- A "dot column" step applies that decoded block to one activation column.

The kernel loops over blocks. It unpacks once, then applies the result to K columns, with one 512-bit accumulator per column. When AVX-512BW is missing, it falls back to calling the single-column kernel K times.

Wired: yes, via the dispatcher (section 5). Lossless relative to the single-column kernel: the arithmetic is identical per column.

### 2.6 Why Q2_0 is "faster"

At decode, the matmul is DRAM-bandwidth-bound. The tensors converted to Q2_0 in DeepSeek-V4 were Q8_0 (8.5 bpw) and are now 3.75 bpw. That is 2.27x fewer bytes, and it is almost certainly the dominant effect. The rotate trick is a neat way to keep the unpack cheap, but compute is not the bottleneck for single-token decode.

The gain is real, but it comes from **lossy re-quantization**, not from a kernel trick. The docs credit `vprorvq` for the speedup (docs/DEEPSEEK_V4_OPTIMIZATIONS.md L26, L39). That attribution is misleading.

---

## 3. Changes to existing kernels in `arch/x86/quants.c` (DS L1448-4175; Z5 L21-203)

### 3.1 AVX-512 `quantize_row_q8_K` (DS L1452-1521; not in Z5)

Upstream x86 simply called the scalar reference. The new version handles one 256-element block at a time:
- Load the block as 16 zmm of 16 floats and take a running max of |x|.
- Reduce to amax. If amax is 0, write zero d, zero qs and zero bsums.
- Find the **first** element whose |x| equals amax: scan the 16 registers in order and take the lowest set bit of an equality mask. Use that element's signed value as max.
- Set iscale = -127/max.
- For each value, multiply by iscale, convert to int32 using the default round-to-nearest-even mode, clamp from above at 127, and narrow to int8 with `vpmovdb`.
- The 16 bsums (sums over 16 consecutive values) come from a tree: add the two 256-bit halves of each zmm, run 3 levels of `vphaddd` across 4 registers, add the 128-bit halves, then saturating-pack to int16.
- d = 1/iscale.

Bit-exactness: the reference uses a round-to-nearest-even magic-number trick, the same clamp and the same first-max tie rule, so the output should be **bit-exact**. NaN inputs would differ, which doesn't matter in practice.

This is the activation quantizer for every K-quant and IQ type. It is wired in and lossless.

### 3.2 "RouteCrack" non-temporal prefetch hints

These are added to the main loops of the q4_0, q5_0, mxfp4, q2_K, q3_K, q4_K, q5_K, q6_K and iq2_xxs dot products. Each issues `prefetchnta` on weight data a few blocks ahead.

| Kernel | Lines | What it prefetches |
|---|---|---|
| q4_0 | DS L1523; Z5 L28 | block ib+4. Blocks are 18 B, so about 3.5 redundant prefetches per cache line. |
| q5_0 | DS L1547, L1560; Z5 L54, L67 | block ib+4 |
| mxfp4 | DS L1536; Z5 L42 | weights 6 blocks ahead (NTA), activations 6 blocks ahead (T0) |
| K-quants | see note below | block i+2 at offsets +64, +128, +192 (DS) or +0..+192 (Z5) |
| q5_K (AVX2) | DS L1648 | +0, +64, +128 |
| iq2_xxs | DS L1775 | +64 only |

K-quant line references: q2_K DS L1573, q3_K DS L1588 / L1603, q4_K DS L1618 / L1633, q5_K DS L1682, q6_K DS L1697 / L1712. Z5 has the same set.

Assessment:
- All of these are lossless, since they are only hints.
- **Partly dead code.** In upstream, the second hunk of each q3_K, q4_K, q5_K and q6_K pair, and the second q5_0 hunk, sit in the `#elif __AVX__` (non-AVX2) branch. That applies to DS L1560, L1603, L1633, L1682, L1712 and the Z5 equivalents. The added prefetch is guarded by `AVX2 || AVX512F`, so it **can never compile** there.
- **Sloppy offsets.** The DS K-quant hunks skip offset +0 of the block. For Q2_K (84 B) and Q4_K (144 B), offsets +128 and +192 fall into later blocks. This is harmless. Comments say "Q6_K" in non-Q6_K kernels (copy-paste).
- **Doubtful benefit.** A distance of 2 super-blocks (about 0.2-0.4 KB) is far too short to hide DRAM latency, and hardware stream prefetchers already cover sequential weight reads. The only plausible benefit is the claimed reduction in L3 pollution, so that the KV cache stays resident. That is hardware-dependent and **not demonstrated** in the code.
- I would treat any speedup claim from these hints as unverified and probably marginal.

### 3.3 q5_K single-column AVX-512 widening (DS L1660-1680)

Inside the AVX2 loop of `ggml_vec_dot_q5_K_q8_K`, when AVX512BW+VL are available:
- Concatenate the two 256-bit 5-bit-weight vectors, the two 256-bit scale vectors and 64 activation bytes into zmm registers.
- Do one `vpmaddubsw` and one `vpmaddwd` at 512 bits, then fold the result back to 256 bits.

The integer results are identical, so it is lossless and wired in. It is a small instruction-count saving.

### 3.4 iq2_xxs AVX-512 single-column (DS L1727-1774)

Upstream AVX2 looked up the 8-sign pattern for each 8-weight group in a 128-entry table indexed by 7 sign bits. The 8th sign is implied, chosen to make the count of minus signs even. The new code computes the signs arithmetically.

It processes two 32-weight sub-blocks (64 weights) per iteration:
- Gather 8 grid entries (8 bytes each) by index into one zmm.
- Take the four 7-bit sign groups from each of the two sub-blocks. These are the low 28 bits of the 2nd and 4th 32-bit words of the sub-block data. Concatenate them into 56 bits.
- `pdep` the 56 bits into a 64-bit word with 7 bits per byte (mask 0x7f in every byte).
- Compute each byte's parity with a 3-step xor fold (shift 4, 2, 1) and insert it as bit 7. The result is a **64-bit sign mask**, one bit per weight in byte order.
- Negate the masked activation bytes by masked subtract-from-zero.
- `vpmaddubsw` with the unsigned grid bytes, then `vpmaddwd` with a per-half scale vector. The scale is 2*s+1, where s is the top 4 bits of the respective word.
- Final scale factor 1/8, as upstream.

I verified the parity fold: it only mixes bits within the same byte. Lossless and wired (requires BMI2).

### 3.5 iq1_s

**AVX-512 single-column (DS L1785-1833).**
- Upstream AVX2 builds two 256-bit grid vectors using 11-bit indices from the `pdep` pack. The new code joins them into one zmm.
- The grid values are in {-1, 0, +1}. Take |g| with `vpabsb`, and get the sign mask with `vpmovb2m`.
- Negate the masked activation bytes, then `vpmaddubsw(|g|, ±q8)` followed by `vpmaddwd` with the per-half scale 2*s+1.
- The delta term (the `bsums` x ±1 x scale correction) stays scalar.
- Lossless and wired.

**AVX2 micro-opts (DS L1834-1865).**
1. The 11-bit indices are extracted from the 64-bit `pdep` result with shifts and casts. Previously they were read through a `uint16_t*` alias of a stack variable, which forces a store and reload. This affects codegen only.
2. Upstream computes `maddubs(sign(x,x), sign(y,x))`. The new code uses a constant all-ones byte vector in place of |x|, because |grid| is 0 or 1. When the grid value is 0, `vpsignb` already zeroes the activation byte. This saves one `vpsignb` per 32 weights.
   - Lossless as long as activations are never -128. Q8_K clamps to ±127, and the AVX-512 quantizer can't produce -128, as noted in 3.1.

**iq1_m (DS L1866-1884).** Only the same index-extraction cleanup. It also leaves an **empty `if` statement** at DS L1870: a placeholder for a prefetch that was never written. Harmless dead code.

### 3.6 Multi-column ("register-stationary") kernels for IQ1_S, IQ2_XXS, IQ1_M, Q5_K and Q6_K (DS L1885-4175)

There are 8, 4, 3 and 2-column variants for each type:

| Type | 1x8 | 1x4 | 1x3 | 1x2 |
|---|---|---|---|---|
| iq1_s | L1896 | L2770 | L3320 | L3789 |
| iq2_xxs | L2055 | L2865 | L3399 | L3852 |
| iq1_m | L2194 | L2952 | L3473 | L3913 |
| q5_K | L2395 | L3081 | L3584 | L4006 |
| q6_K | L2604 | L3214 | L3698 | L4101 |

Requirements: AVX512F, BW and VL, plus BMI2 for the IQ types. Otherwise each variant falls back to looping the single-column kernel K times.

Pattern, the same for every type:
- Per weight super-block (256 weights), decode the weights once into zmm registers. That covers grid gathers, sign masks, scale vectors, and the 5/6-bit reconstruction with the high-bit masks.
- For each of the K activation columns (Q8_K), load 64 activation bytes. Apply the sign or mask, run `vpmaddubsw` / `vpmaddwd`, and accumulate into that column's own 512-bit integer accumulator.
- Per-column scalar terms are computed separately for each column:
  - iq1_s: the delta x bsums term
  - Q5_K: the mins x bsums term
  - Q6_K: the -32 offset, done as 32 x (bsums · scales)
  - iq1_m: the delta term goes into a second accumulator
- At the end of the super-block, fold to 256 bits, convert, FMA with d_x * d_y,col, and do a final horizontal sum per column.

Per-column arithmetic matches the single-column AVX-512 path in integer terms. Float accumulation order is the same per column, so these should be bit-identical to single-column results.

I spot-checked the iq1_m deltas (sign-extended from a pdep of the qh sign bits, then OR 1), the Q6_K and Q5_K overflow bounds, and the scale shuffles. I found no bugs.

Why it's faster: in prefill or batched verification, each weight block is decoded once per K tokens instead of once per token. The decode instructions (gathers, `pdep`, shuffles) and the weight loads are amortized. This is a genuine, lossless, **prefill-only** gain. It does nothing for single-token decode, where N = 1.

The README's "+77% prefill" figure is plausible in direction but not independently verifiable from code.

**No VNNI.** No kernel in either patch uses `vpdpbusd` or AVX-512-VNNI. Everything is `vpmaddubsw` + `vpmaddwd` (BW). The README hardware table mentions "VPDPBUSD ... VBMI ... BF16" (README L166). None of those appear in the patches.

There are no NEON kernels at all.

---

## 4. Conversion pipeline

### 4.1 `build_fractal_moe_gguf.sh`

Non-DeepSeek mode is just stock `llama-quantize` with `--tensor-type` overrides:
- **fractal64 profile.** Routed experts `ffn_up_exps` and `ffn_down_exps` go to q2_0 (3.75 bpw). Attention q/k/v/o, `ssm_out` and the output head go to q8_0. Token embedding goes to q4_0. The base type is Q4_0 with `--allow-requantize`. The default input is the NVIDIA **Q4_0** GGUF, so this is lossy-on-lossy: Q4_0 is decoded and re-quantized to 3.75 bpw. The README table says "BF16 Official" as the source.
  - Note: `ffn_gate_exps` is **not** overridden and stays at the base Q4_0. The README's "hybrid Q4_0 + Fractal-64" for the experts reflects this.
- **mxfp4 / mxfp4_q5 profiles.** These use **upstream MXFP4 unchanged**. "Dyadic Fractal MXFP4" is just MXFP4 (E8M0 shared exponent + E2M1). The only patch change for MXFP4 is the prefetch hint. Routed experts go to mxfp4. Shared experts and SSM go to q4_0 or q5_0. Attention goes to q8_0.
- `--imatrix` is passed through, but q2_0 ignores it (1.2).

DeepSeek mode builds `transcode_dsv4.c` as a shared library linked against ggml and runs the python driver.

### 4.2 `transcode_dsv4_shard.py` (per GGUF shard)

1. Parse the GGUF header itself. It records each tensor's type and offset field position and works out sizes from neighbouring offsets.
2. Choose a target type per tensor (L27-52):
   - **Attention.** Tensors that are Q8_0 *and* whose name contains `attn_q_b`, `attn_output_a`, `attn_output_b`, `attn_compressor_gate`, `attn_compressor_kv`, `attn_kv`, `indexer_compressor_gate` or `indexer_compressor_kv` go to Q2_0 (id 42, 3.75 bpw).
     - Shared experts (`shexp`) are **not** converted. The docs table (docs L26) calls `shexp` "q2_0", which contradicts the script.
   - **Routed gate/up experts.** If currently IQ2_XXS and the layer is not in {1, 4, 26, 28, 40}, they go to IQ1_S (1.5625 bpw). Otherwise they stay as they are.
   - **Routed down experts:**
     - layers {0, 1, 26, 28, 41, 42}: IQ2_XXS (2.0625 bpw)
     - layers {2, 4, 5, 8, 27, 30, 32, 40}: IQ1_M (1.75 bpw)
     - all others: IQ1_S (1.5625 bpw)
   - Everything else is copied as is.
3. Rewrite the header with new types and 32-byte-aligned offsets, then stream the tensors out.

### 4.3 `transcode_dsv4.c`

- **Q8_0 to Q2_0.** Per row: dequantize Q8_0 to f32, then run the Q2_0 reference quantizer. Uses 32 OpenMP threads.
- **MoE.** Per expert: dequantize the source slice to f32, compute per-column importance weights, then call upstream `quantize_iq1_s` / `quantize_iq1_m` / `quantize_iq2_xxs` with those weights as the "imatrix".
- **"SwiGLU-H2" importance.** Only for down_proj, and only when the shape is exactly 2048 x 4096. For intermediate neuron j, importance = ||W_gate row j||² x ||W_up row j||², normalized to mean 1.
  - This is a **data-free proxy** for the true down_proj imatrix E[h_j²]. It assumes isotropic inputs.
  - It is computed from already-1-2-bit gate/up weights.
- **Fallback importance** (gate/up, or other shapes): 0.5 + 0.5 x (column energy / mean).
- **Lossy-on-lossy throughout.** The source is unsloth's **UD-IQ1_S** GGUF, which is already ~1.6-2 bpw for experts. Experts are decoded from IQ2_XXS / IQ3_XXS and re-quantized to even fewer bits. Attention goes from Q8_0 to 3.75 bpw.
  - The "lossless / bit-exact / 1.00000 cosine" claims in the README and docs (README L95, L184; docs L76) compare only against the *same transcoded model* with different runtime settings, not against the unsloth original.
  - The unpatched reference "Top-6" row can't even load the Q2_0-containing file. So those claims say nothing about the cost of the conversion.

Bugs and risks in the transcoder:
- **Silent garbage on unhandled types.** `dequant_expert_slice` handles only IQ2_XXS, IQ3_XXS, IQ1_S and MXFP4. **IQ1_M (29) is not handled.** For any other source type, f32 stays as uninitialized `malloc` memory.
  - In addition, `expert_bytes` returns 0 for unknown types, so every expert would read from offset 0.
  - Example: a source down tensor already in IQ1_M (or Q2_K, IQ2_XS, ...) and targeted to IQ1_S or IQ2_XXS would be quantized from garbage with no error. Whether this triggers depends on the exact unsloth tensor mix.
- **Shape assumptions are unchecked.** ncol % 64 is not checked for Q2_0, and ncol % 256 is not checked for the IQ types. The SwiGLU-H2 path silently falls back unless the shape is exactly 2048 x 4096.
- **Hard-coded resources.** `num_threads(32)` is fixed, and type IDs are hard-coded integers that match upstream `d2f83055`.

---

## 5. Wiring of the multi-column kernels (ggml-cpu.c, DS only)

**Dispatcher** (DS L4309-4361). Maps the types Q2_0, IQ1_S, IQ2_XXS, IQ1_M, Q5_K and Q6_K to their 1x8, 1x4, 1x3 and 1x2 kernels. Every other type gets nothing.

**`mul_mat_one_chunk`** (DS L4398-4497). The new path is used when all of these hold:
- one weight row per vec_dot call
- more than one activation column in the chunk
- the SwiGLU-skip mode (below) is not active

The loop structure:
- Weight rows are tiled in groups of 128, the "L2 tile".
- Columns are walked in steps of 8 / 4 / 3 / 2. A remainder of 9 is split as 4 + 3 + 2, and a remainder of 5 as 3 + 2.
- Steps never cross a boundary where the broadcast weight matrix changes (segment width = ne11 x r2).
- Results are written straight into dst.

**`mul_mat_id_one_chunk`** (MoE, DS L4630-4740). The same scheme runs over all token rows routed to the same expert, so tokens are batched per expert.

Both paths are wired and active for prefill and multi-token verification on AVX-512 machines. On AVX2-only machines they fall back to repeated single-column calls, which is correct but gives no gain. Z5 has none of this.

**Adjacent, lossy features in the same code** (not quant formats; noted for other reviewers):
- **SwiGLU gate masking** (`ggml_swiglu_gate_mask_128`, DS L4237-4306, used at L4502-4510 and in mul_mat_id). For single-token matmuls, rows whose SiLU(gate)² falls below a quickselect threshold are skipped, and the kept outputs are rescaled by sqrt(total energy / kept energy). This is **lossy activation sparsity**.
- **MoE expert skip mask** (DS L4822 onward). Drops routed experts. Lossy.
- **8-slot Q8 activation cache** (DS L4228-4232, L5032-5220). Reuses a quantized src1 across matmuls that share the same input, which saves re-quantization, and skips barriers when the cache hits. The reuse is lossless if the invalidation is right: it invalidates on any write overlapping the base buffer. The barrier elision deserves a race review.
  - The buffer is a process-global static (8 x 64 KiB) shared by all graphs, so concurrent `ggml_graph_compute` calls in one process would corrupt each other.

---

## 6. Summary table

| Item | Patch lines | ISA | Activation | Lossless? | Wired? | Notes |
|---|---|---|---|---|---|---|
| Q2_0 "Fractal-64" format (30 B / 64 w, 3.75 bpw) | DS L1428-1447, L6810-6990 (Z5 same) | n/a | Q8_0 | **No** (new lossy codebook, no imatrix) | Yes | **Collides with upstream Q2_0 id 42.** Breaks ARM NEON, Metal, CUDA and Vulkan Q2_0. README ID/scale-type claims wrong. |
| Q2_0 AVX-512 (`vprorvq` bit-transpose) | DS L6586-6637 | AVX512F+BW | Q8_0 | vs scalar ref: yes | DS yes; Z5 `#if 0` | Correct. The speed comes from fewer bytes (8.5 to 3.75 bpw), not the trick. |
| Q2_0 AVX2 | DS L6638-6700; Z5 L867-930 | AVX2 | Q8_0 | yes | yes | Correct. |
| Q2_0 1x8/4/3/2 | DS L6398-6582 | AVX512BW | Q8_0 | yes | DS yes | Prefill only. |
| AVX-512 quantize_row_q8_K | DS L1452-1521 | AVX512F/BW/VL | produces Q8_K | bit-exact | yes | Real small win. |
| NTA prefetches | DS L1523-1726; Z5 L28-203 | SSE prefetchnta | n/a | yes | partly; the `__AVX__`-branch copies are dead | Benefit unproven; very short prefetch distance. |
| q5_K 512-bit widening | DS L1660-1680 | AVX512BW/VL | Q8_K | yes | yes | Minor. |
| iq2_xxs AVX-512 (pdep + parity signs) | DS L1727-1774 | AVX512BW/VL + BMI2 | Q8_K | yes | yes | Clever and correct. |
| iq1_s AVX-512 | DS L1785-1833 | AVX512BW/VL + BMI2 | Q8_K | yes | yes | |
| iq1_s AVX2 micro-opts | DS L1834-1865 | AVX2 (+BMI2) | Q8_K | yes (q8 ≠ -128) | yes | Drops one vpsignb; avoids aliasing reload. |
| iq1_m index cleanup + empty `if` | DS L1866-1884 | AVX2+BMI2 | Q8_K | yes | yes | Dead placeholder at L1870. |
| 1xN for iq1_s / iq2_xxs / iq1_m / q5_K / q6_K | DS L1885-4175 | AVX512BW/VL (+BMI2) | Q8_K | yes | DS yes, x86 only | **Undefined symbols on non-x86 builds.** |
| DSV4 transcoder | scripts | n/a | n/a | **No** (re-quantizes an already IQ1_S model) | n/a | Garbage on unhandled source types (e.g. IQ1_M). "Lossless" claims are relative to its own output. |
