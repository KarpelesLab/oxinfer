#!/usr/bin/env python3
# Reference outputs of a Raon-1 nano checkpoint (KRAFTON/Raon-1-nano, or a tiny random one from
# tools/raon_tiny.py) from KRAFTON's own modeling code (`modeling_raon1_nano.py`, unchanged), run
# in float32 on the CPU: the bf16 weights are widened, so every `.to(dtype)` of the modeling code
# is the identity and both sides compute on the same weights.
#
# usage: ref_raon.py MODEL_DIR CODE_DIR OUT.safetensors N_PROMPT N_DECODE [SEED] [--chunk N] [--text FILE]
#   CODE_DIR holds `modeling_raon1_nano.py` and `configuration_raon1_nano.py`.
#   The prompt is N_PROMPT random token ids, or with --text the first N_PROMPT tokens of FILE
#   (through MODEL_DIR/tokenizer.json). --chunk: the prompt as passes of N tokens (default: one).
#   Writes `ids`, `logits` [n, vocab] (every prompt position), `gen` (greedy tokens) and
#   `decode_logits` [len(gen), vocab] (after feeding each of them, one token per pass).
# Then: cargo run --release --example lm_check -- MODEL_DIR OUT.safetensors [--chunk N]
import importlib, sys, types

import torch


def reference(code_dir):
    """KRAFTON's modeling file as a module (it imports its configuration relatively)."""
    pkg = types.ModuleType("raon_ref")
    pkg.__path__ = [code_dir]
    sys.modules["raon_ref"] = pkg
    return importlib.import_module("raon_ref.modeling_raon1_nano")


def main():
    from safetensors.torch import save_file

    d, code, out, n, nd = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    seed = int(sys.argv[6]) if len(sys.argv) > 6 and not sys.argv[6].startswith("--") else 0
    opt = lambda name: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None
    torch.set_num_threads(8)
    M = reference(code)
    model = M.Raon1NanoForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    vocab = model.config.vocab_size
    if opt("--text"):
        from tokenizers import Tokenizer

        ids = Tokenizer.from_file(f"{d}/tokenizer.json").encode(open(opt("--text")).read(), add_special_tokens=False).ids
        ids = torch.tensor([ids[:n]])
        n = ids.shape[1]
    else:
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(0, min(vocab, 259000), (1, n), generator=g)
    chunk = int(opt("--chunk") or n)
    res = {"ids": ids[0].to(torch.int64)}
    with torch.no_grad():
        past, logits = None, []
        for p in range(0, n, chunk):
            o = model(input_ids=ids[:, p:p + chunk], past_key_values=past, use_cache=True)
            past = o.past_key_values
            logits.append(o.logits[0].float())
        res["logits"] = torch.cat(logits)
        nxt = res["logits"][-1].argmax()
        gen, steps = [], []
        for _ in range(nd):
            gen.append(int(nxt))
            o = model(input_ids=nxt.view(1, 1), past_key_values=past, use_cache=True)
            past = o.past_key_values
            steps.append(o.logits[0, -1].float())
            nxt = steps[-1].argmax()
    res["gen"] = torch.tensor(gen, dtype=torch.int64)
    if steps:
        res["decode_logits"] = torch.stack(steps)
    save_file({k: v.contiguous() for k, v in res.items()}, out)
    print("prompt", n, "gen", gen)
    print("top5 last", torch.topk(res["logits"][-1], 5))


if __name__ == "__main__":
    main()
