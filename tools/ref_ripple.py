#!/usr/bin/env python3
# Reference outputs of a Ripple checkpoint (ricursiveAI/Ripple-65B-Preview, or a tiny random one
# from tools/ripple_tiny.py) from Ricursive's own modeling code (`modeling_ricursive.py`,
# unchanged), run in float32 on the CPU with the cache the modeling code builds by default (on
# transformers 5: sliding layers keep their window only).
#
# usage: ref_ripple.py MODEL_DIR CODE_DIR OUT.safetensors N_PROMPT N_DECODE [SEED] [--chunk N] [--text FILE] [--layers K]
#   CODE_DIR holds `modeling_ricursive.py` and `configuration_ricursive.py`.
#   The prompt is N_PROMPT random token ids, or with --text the first N_PROMPT tokens of FILE
#   (through MODEL_DIR/tokenizer.json). --chunk: the prompt as passes of N tokens (default: one).
#   Writes `ids`, `logits` [n, vocab] (every prompt position), `gen` (greedy tokens) and
#   `decode_logits` [len(gen), vocab] (after feeding each of them, one token per pass).
#   --layers K: the model cut to its first K layers (with the final norm and the head), read
#   tensor by tensor from the shards: what fits in memory of a checkpoint that does not, on its
#   real weights. NVFP4 matrices (ModelOpt's layout: `.weight` as packed E2M1 pairs, a
#   `.weight_scale` per 16 values, a `.weight_scale_2` factor) are decoded to float32.
# Then: cargo run --release --example lm_check -- MODEL_DIR OUT.safetensors [--chunk N]
import importlib, sys, types

import torch


def reference(code_dir):
    """Ricursive's modeling file as a module (it imports its configuration relatively)."""
    pkg = types.ModuleType("ripple_ref")
    pkg.__path__ = [code_dir]
    sys.modules["ripple_ref"] = pkg
    return importlib.import_module("ripple_ref.modeling_ricursive")


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def nvfp4(packed, scale, factor):
    """ModelOpt NVFP4: byte i holds elements 2i (low nibble) and 2i + 1 (sign, then an E2M1
    magnitude), one E4M3 scale per 16 elements, one factor per tensor."""
    table = torch.tensor(E2M1 + [-x for x in E2M1])
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(packed.shape[0], -1).long()
    return table[codes] * scale.float().repeat_interleave(16, dim=1) * float(factor)


def load_cut(M, d, layers):
    """The first `layers` layers of the checkpoint in `d` as a float32 model."""
    import json
    from safetensors import safe_open

    C = sys.modules["ripple_ref.configuration_ricursive"]
    raw = json.load(open(f"{d}/config.json"))
    raw = {k: v for k, v in raw.items() if k not in ("architectures", "model_type", "auto_map", "quantization_config", "dtype")}
    raw["num_hidden_layers"] = layers
    if raw.get("layer_types"):
        raw["layer_types"] = raw["layer_types"][:layers]
    model = M.RicursiveForCausalLM(C.RicursiveConfig(**raw)).float().eval()
    try:
        index = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
    except FileNotFoundError:  # one file
        index = dict.fromkeys(safe_open(f"{d}/model.safetensors", "pt").keys(), "model.safetensors")
    files = {}
    get = lambda name: files.setdefault(index[name], safe_open(f"{d}/{index[name]}", "pt")).get_tensor(name)
    state = {}
    for name in model.state_dict():
        if name == "lm_head.weight" and name not in index:
            continue  # tied
        base = name[: -len(".weight")]
        if name.endswith(".weight") and base + ".weight_scale_2" in index:
            state[name] = nvfp4(get(name), get(base + ".weight_scale"), get(base + ".weight_scale_2"))
        else:
            state[name] = get(name).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
    return model


def main():
    from safetensors.torch import save_file

    d, code, out, n, nd = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    seed = int(sys.argv[6]) if len(sys.argv) > 6 and not sys.argv[6].startswith("--") else 0
    opt = lambda name: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None
    torch.set_num_threads(8)
    M = reference(code)
    if opt("--layers"):
        model = load_cut(M, d, int(opt("--layers")))
    else:
        model = M.RicursiveForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    vocab = model.config.vocab_size
    if opt("--text"):
        from tokenizers import Tokenizer

        ids = Tokenizer.from_file(f"{d}/tokenizer.json").encode(open(opt("--text")).read(), add_special_tokens=False).ids
        ids = torch.tensor([ids[:n]])
        n = ids.shape[1]
    else:
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(0, min(vocab, 100000), (1, n), generator=g)
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
    print("cache", type(past).__name__, [type(l).__name__ for l in getattr(past, "layers", [])][:6])
    print("prompt", n, "gen", gen)
    print("top5 last", torch.topk(res["logits"][-1], 5))


if __name__ == "__main__":
    main()
