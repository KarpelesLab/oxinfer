#!/usr/bin/env python3
# Reference outputs of a (tiny) Kimi K3 / K3.1 text checkpoint from Moonshot's own modeling code
# (`modeling_kimi_linear.py` of moonshotai/Kimi-K3: the K3.1 repository ships none), run in
# float32 on the CPU.
#
# What is not Moonshot's code here:
# - The fla-core kernels the modeling file calls are Triton (GPU only). They are replaced by
#   plain PyTorch: the delta rule and its gate by fla-core's own `naive_recurrent_kda` and
#   `naive_kda_lowerbound_gate` (queries and keys L2-normed, beta through a sigmoid, as the
#   kernels do with the flags the modeling file passes), the short convolution by a causal
#   depthwise `conv1d` + SiLU, the gated norm by `x * rsqrt(mean(x^2) + eps) * w * sigmoid(g)`.
# - `A_log`: the released checkpoints store it in a `[head_dim]` buffer whose first `num_heads`
#   entries are the per-head values (the rest zero); it is narrowed on load, as SGLang,
#   TokenSpeed and llama.cpp do.
# - K3.1's sparse attention ("gated DSA", `dsa_config`): no modeling code is published for it.
#   The indexer below follows TokenSpeed (an engine Moonshot's model card names) and llama.cpp,
#   which agree: for query t and key s <= t,
#       I(t, s) = d^-0.5 H^-0.5 sum_h weights_proj(x_t)_h relu(wq_b(c_q,t)_h . k_norm(wk(x_s)))
#   (x: the attention input, c_q: the normed MLA query latent, k_norm: a LayerNorm with bias),
#   and the MLA attends only to the `index_topk` best keys of each query.
#
# usage: ref_kimi_k31.py MODEL_DIR CODE_DIR OUT.safetensors N_PROMPT N_DECODE [SEED] [--chunk N]
#   CODE_DIR holds `modeling_kimi_linear.py` (moonshotai/Kimi-K3) and `configuration_kimi_k3.py`
#   (moonshotai/Kimi-K3.1-Pro). --chunk: the prompt as passes of N tokens (default: one pass).
#   Writes `ids`, `logits` [n, vocab] (every prompt position), `gen` (greedy tokens) and
#   `decode_logits` [len(gen), vocab] (after feeding each of them, one token per pass).
import importlib, json, os, sys, types

import torch
import torch.nn.functional as F
from torch import nn


# The epsilon of the queries' and keys' L2 norm: 1e-6 in fla-core's kernels. llama.cpp's Kimi
# graphs use the RMS norm epsilon there (1e-5), which a random tiny model amplifies into visibly
# different logits: KIMI_REF_L2_EPS=1e-5 gives the reference to compare llama.cpp with.
L2_EPS = float(os.environ.get("KIMI_REF_L2_EPS", "1e-6"))


def reference(code_dir):
    """Moonshot's `modeling_kimi_linear.py` as a module, its GPU kernels replaced by PyTorch."""
    from fla.ops.kda.gate import naive_kda_lowerbound_gate
    from fla.ops.kda.naive import naive_recurrent_kda

    pkg = types.ModuleType("kimi_ref")
    pkg.__path__ = [code_dir]
    sys.modules["kimi_ref"] = pkg
    M = importlib.import_module("kimi_ref.modeling_kimi_linear")

    class ShortConvolution(nn.Module):
        def __init__(self, hidden_size, kernel_size, activation="silu", **kw):
            super().__init__()
            assert activation == "silu"
            self.kernel_size = kernel_size
            self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))

        def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, **kw):
            assert cu_seqlens is None
            b, t, d = x.shape
            if cache is None:
                cache = x.new_zeros(b, self.kernel_size - 1, d)
            xx = torch.cat([cache, x], dim=1)
            y = F.conv1d(xx.transpose(1, 2), self.weight, groups=d).transpose(1, 2)
            return F.silu(y), xx[:, -(self.kernel_size - 1):]

    class FusedRMSNormGated(nn.Module):
        def __init__(self, hidden_size, eps=1e-5, activation="swish", **kw):
            super().__init__()
            assert activation == "sigmoid"
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(hidden_size))

        def forward(self, x, g):
            x = x.float()
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight * torch.sigmoid(g.float())

    def kda(q, k, v, g, beta, A_log, dt_bias, initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
            lower_bound=None, cu_seqlens=None, **kw):
        assert use_qk_l2norm_in_kernel and use_gate_in_kernel and use_beta_sigmoid_in_kernel
        assert lower_bound is not None and cu_seqlens is None
        g = naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound)
        q, k = (x.float() * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + L2_EPS) for x in (q, k))
        return naive_recurrent_kda(q, k, v.float(), g, torch.sigmoid(beta.float()),
                                   initial_state=initial_state, output_final_state=True)

    M.ShortConvolution, M.FusedRMSNormGated = ShortConvolution, FusedRMSNormGated
    M.chunk_kda = M.fused_recurrent_kda = kda

    class Indexer(nn.Module):
        """K3.1's lightning indexer (see the header)."""

        def __init__(self, config):
            super().__init__()
            d = config.dsa_config
            assert d["index_rope_dim"] == 0 and d["index_k_norm_type"] == "layernorm" and d["index_topk_freq"] == 1
            self.heads, self.dim, self.topk = d["index_num_heads"], d["index_head_dim"], d["index_topk"]
            self.wq_b = nn.Linear(config.q_lora_rank, self.heads * self.dim, bias=False)
            self.wk = nn.Linear(config.hidden_size, self.dim, bias=False)
            self.k_norm = nn.LayerNorm(self.dim, eps=d.get("index_norm_eps", 1e-6))
            self.weights_proj = nn.Linear(config.hidden_size, self.heads, bias=False)

    mla_forward = M.KimiMLAAttention.forward

    def sparse_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, **kw):
        ix = getattr(self, "indexer", None)
        if ix is None:
            return mla_forward(self, hidden_states, attention_mask, position_ids, past_key_values, **kw)
        b, t, _ = hidden_states.shape
        cq = self.q_a_layernorm(self.q_a_proj(hidden_states))
        qi = ix.wq_b(cq).view(b, t, ix.heads, ix.dim)
        keys = ix.k_norm(ix.wk(hidden_states))
        store = past_key_values.__dict__.setdefault("index_keys", {})
        if self.layer_idx in store:
            keys = torch.cat([store[self.layer_idx], keys], dim=1)
        store[self.layer_idx] = keys
        n = keys.shape[1]
        w = ix.weights_proj(hidden_states) * (ix.dim ** -0.5 * ix.heads ** -0.5)
        score = (torch.einsum("bthd,bsd->bths", qi, keys).relu() * w.unsqueeze(-1)).sum(2)
        s = torch.arange(n).view(1, 1, n)
        visible = s <= (n - t + torch.arange(t)).view(1, t, 1)
        score = score.masked_fill(~visible, float("-inf"))
        # Equal scores (every head's product negative gives exactly 0): the earlier key first.
        top = score.sort(dim=-1, descending=True, stable=True).indices[..., :min(ix.topk, n)]
        chosen = torch.zeros(b, t, n, dtype=torch.bool).scatter_(2, top, True) & visible
        mask = torch.zeros(b, 1, t, n).masked_fill(~chosen.unsqueeze(1), float("-inf"))
        return mla_forward(self, hidden_states, mask, position_ids, past_key_values, **kw)

    M.KimiMLAAttention.forward = sparse_forward
    M.Indexer = Indexer
    return M


def build(M, cfg_json):
    """The text model of config.json `cfg_json` (float32), with an indexer on K3.1's DSA layers."""
    C = importlib.import_module("kimi_ref.configuration_kimi_k3")
    text = dict(cfg_json["text_config"])
    text.pop("quantization_config", None)
    cfg = C.KimiLinearConfig(**text)
    model = M.KimiLinearForCausalLM(cfg)
    cfg._attn_implementation = "eager"
    dsa = getattr(cfg, "dsa_config", None)
    if dsa:
        for l in dsa["dsa_layers"]:
            model.model.layers[l - 1].self_attn.indexer = M.Indexer(cfg)
    return model.float().eval()


def unpack_mxfp4(packed, scale):
    """compressed-tensors `mxfp4-pack-quantized`: element 2i in the low nibble of byte i, 2i + 1
    in the high one (E2M1: sign, two exponent bits, one mantissa bit), one E8M0 exponent per 32."""
    rows = packed.shape[0]
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(rows, -1).long()
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    s = torch.pow(2.0, scale.float() - 127.0).repeat_interleave(32, dim=1)
    return table[codes] * s


def load(model, d):
    from safetensors.torch import load_file

    sd = load_file(f"{d}/model.safetensors")
    out = {}
    for k, v in sd.items():
        if not k.startswith("language_model.") or k.endswith(".weight_scale"):
            continue
        k = k[len("language_model."):]
        if k.endswith(".weight_packed"):
            k = k[:-len("_packed")]
            v = unpack_mxfp4(v, sd["language_model." + k + "_scale"])
        elif k.endswith(".A_log"):
            assert not v[model.config.linear_attn_config["num_heads"]:].any(), "A_log tail"
            v = v[:model.config.linear_attn_config["num_heads"]]
        out[k] = v.float()
    missing, unexpected = model.load_state_dict(out, strict=False)
    assert not missing and not unexpected, (missing, unexpected)


def main():
    from safetensors.torch import save_file

    d, code, out, n, nd = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    seed = int(sys.argv[6]) if len(sys.argv) > 6 and not sys.argv[6].startswith("--") else 0
    chunk = int(sys.argv[sys.argv.index("--chunk") + 1]) if "--chunk" in sys.argv else n
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float32)
    M = reference(code)
    cfg = json.load(open(f"{d}/config.json"))
    model = build(M, cfg)
    load(model, d)
    vocab = cfg["text_config"]["vocab_size"]
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, min(vocab, 160000), (1, n), generator=g)
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
