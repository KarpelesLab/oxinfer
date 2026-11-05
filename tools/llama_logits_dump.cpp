// Greedy reference run: lgdump MODEL NGL IDS_FILE N_GEN OUT_BIN [THREADS]
// Feeds the token ids (whitespace separated) as one batch, then generates N_GEN tokens greedily
// one at a time. Writes, per step (the prompt's last position, then each generated token), the
// full f32 output row; prints the chosen ids.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <chrono>

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s MODEL NGL IDS_FILE N_GEN OUT_BIN [THREADS] [CTX]\n", argv[0]);
        return 1;
    }
    const char * path = argv[1];
    int ngl = atoi(argv[2]);
    std::vector<llama_token> ids;
    {
        std::ifstream f(argv[3]);
        long v;
        while (f >> v) ids.push_back((llama_token) v);
    }
    int n_gen = atoi(argv[4]);
    FILE * out = fopen(argv[5], "wb");
    int threads = argc > 6 ? atoi(argv[6]) : 12;
    int n_ctx = argc > 7 ? atoi(argv[7]) : 4096;

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int nv = llama_vocab_n_tokens(vocab);

    auto t0 = std::chrono::steady_clock::now();
    llama_batch b = llama_batch_get_one(ids.data(), (int) ids.size());
    if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "prefill %zu tokens %.2fs\n", ids.size(), std::chrono::duration<double>(t1 - t0).count());
    for (int s = 0; s <= n_gen; s++) {
        float * lg = llama_get_logits_ith(ctx, -1);
        fwrite(lg, sizeof(float), nv, out);
        int best = 0;
        for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
        printf("%d\n", best);
        fflush(stdout);
        if (s == n_gen) break;
        llama_token tk = best;
        llama_batch b1 = llama_batch_get_one(&tk, 1);
        if (llama_decode(ctx, b1)) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    auto t2 = std::chrono::steady_clock::now();
    fprintf(stderr, "decode %d tokens %.2fs (%.2f tok/s)\n", n_gen, std::chrono::duration<double>(t2 - t1).count(),
            n_gen / std::chrono::duration<double>(t2 - t1).count());
    fclose(out);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
