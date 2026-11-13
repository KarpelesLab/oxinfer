// Reference dequantization of the 1-bit / ternary ggml types for `REF_HASHES` in src/ggml.rs:
// ggml's own `to_float` for TQ1_0, TQ2_0, Q1_0 and Q2_0, and for PQ2_0 / PTQ1_0 (types 142 / 143 of
// PrismML's llama.cpp fork) that fork's dequantize_row_pq2_0 / dequantize_row_ptq1_0, copied from
// its ggml/src/ggml-quants.c (release prism-b12128; MIT License, Copyright (c) 2023-2026 The ggml
// authors).
//
//   OXINFER_GGML_DUMP=dir cargo test --lib ggml::tests::dequantization_matches_ggml
//   gcc -O1 -o lowbit tools/ggml_ref_lowbit.c -I llama.cpp/ggml/include \
//       llama.cpp/build/ggml/src/libggml-base.a -lm -lstdc++ -lpthread -lgomp
//   ./lowbit dir      # compares TYPE.ours bit for bit and prints the hashes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include "ggml.h"
typedef uint16_t ggml_half;
#define QK_PQ2_0 128
typedef struct { ggml_half d; uint8_t qs[QK_PQ2_0 / 4]; } block_pq2_0;
#define QK_PTQ1_0 128
typedef struct { uint8_t qs[(QK_PTQ1_0 - 4*QK_PTQ1_0/64)/5]; uint8_t qh[QK_PTQ1_0/64]; ggml_half d; } block_ptq1_0;
#define GGML_FP16_TO_FP32(x) ggml_fp16_to_fp32(x)
#define GGML_RESTRICT restrict
void dequantize_row_pq2_0(const block_pq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_PQ2_0;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            const int byte_index = j / 4;
            const int bit_offset = (j % 4) * 2;
            const uint8_t q = (x[i].qs[byte_index] >> bit_offset) & 0x03;
            y[i*qk + j] = ((int)q - 1) * d;
        }
    }
}
static const size_t ptq1_0_stages[3] = {32, 16, 8};
void dequantize_row_ptq1_0(const block_ptq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PTQ1_0 == 0);
    const int64_t nb = k / QK_PTQ1_0;
    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    for (int64_t i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        size_t j = 0;
        for (size_t s = 0; s < 3; ++s) {
            const size_t c = ptq1_0_stages[s];
            for (; j + c <= sizeof(x->qs); j += c) {
                for (size_t n = 0; n < 5; ++n) {
                    for (size_t m = 0; m < c; ++m) {
                        uint8_t q = x[i].qs[j + m] * pow3[n];
                        int16_t xi = ((uint16_t) q * 3) >> 8;
                        *y++ = (float) (xi - 1) * d;
                    }
                }
            }
        }
        for (size_t n = 0; n < 4; ++n) {
            for (size_t h = 0; h < sizeof(x->qh); ++h) {
                uint8_t q = x[i].qh[h] * pow3[n];
                int16_t xi = ((uint16_t) q * 3) >> 8;
                *y++ = (float) (xi - 1) * d;
            }
        }
    }
}
static uint64_t fnv(const uint8_t *b, size_t n) { uint64_t h = 0xCBF29CE484222325ull; for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001B3ull; return h; }
static uint8_t *slurp(const char *p, size_t *n) { FILE *f = fopen(p, "rb"); if (!f) return NULL; fseek(f, 0, SEEK_END); *n = ftell(f); rewind(f); uint8_t *b = malloc(*n); if (fread(b, 1, *n, f) != *n) return NULL; fclose(f); return b; }
int main(int argc, char **argv) {
    const char *names[] = {"TQ1_0", "TQ2_0", "Q1_0", "Q2_0", "PQ2_0", "PTQ1_0"};
    const int types[] = {GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, -1, -2};
    int bad = 0;
    for (int t = 0; t < 6; t++) {
        char p[1024]; size_t n, no;
        snprintf(p, sizeof p, "%s/%s.bin", argv[1], names[t]);
        uint8_t *raw = slurp(p, &n); if (!raw) { printf("# %s missing\n", names[t]); continue; }
        snprintf(p, sizeof p, "%s/%s.ours", argv[1], names[t]);
        uint8_t *ours = slurp(p, &no);
        size_t k = no / 4; float *y = malloc(no);
        if (types[t] == -1) dequantize_row_pq2_0((const block_pq2_0 *) raw, y, k);
        else if (types[t] == -2) dequantize_row_ptq1_0((const block_ptq1_0 *) raw, y, k);
        else { assert(ggml_row_size(types[t], k) == n); ggml_get_type_traits(types[t])->to_float(raw, y, k); }
        int same = memcmp(y, ours, no) == 0; bad |= !same;
        printf("        (GgmlType::%s, 0x%016llx),%s\n", names[t], (unsigned long long) fnv((uint8_t *) y, no), same ? "" : "  // MISMATCH");
    }
    return bad;
}
