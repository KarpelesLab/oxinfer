#!/usr/bin/env python3
# Reference outputs of a (tiny) LongCat text checkpoint (LongCat-2.5 / 2.0 / Flash-Lite-Sparse:
# `LongcatCausalLM`; LongCat-Flash-Chat / Flash-Lite: `LongcatFlash[Ngram]ForCausalLM`), run in
# float32 on the CPU. No single published model file runs LongCat-2.5: the repository's
# `auto_map` names `modeling_longcat_oe.py`, which is not in it. What runs here:
#
# - transformers' own `LongcatFlashForCausalLM` (transformers >= 5; `models/longcat_flash`): the
#   layer of two sublayers with the shortcut MoE, MLA with the LoRA scales, the softmax router
#   with identity ("zero") experts, YaRN RoPE with the `mscale^2` softmax scale. Unmodified; the
#   released tensors are renamed onto its modules (its experts are stacked).
# - Meituan's `NgramEmbedding` (`modeling_longcat_ngram.py` of meituan-longcat/LongCat-Flash-Lite,
#   CODE_DIR), for the n-gram ("over-encoding") embedding. Unmodified for the LongCat-Flash
#   architectures.
#
# What is transcribed here from Meituan's SGLang-FluentLLM (branch `longcat-2.5-npu`, the
# deployment LongCat-2.5's model card names), for `LongcatCausalLM` only:
#
# - The sparse-attention indexer (`layers/attention/dsa/nsa_indexer.py`, `IndexerBf16`, and
#   `topk_transform` of `layers/attention/dsa_backend.py`): for query t and key s <= t,
#       I(t, s) = d^-0.5 H^-0.5 sum_h weights_proj(x_t)_h relu(wq_b(c_t)_h . k_norm(wk(x_s)))
#   (x: the attention input, c_t: the normed query latent before the LoRA scale, k_norm: an RMS
#   norm with eps 1e-6, the first `qk_rope_head_dim` lanes of queries and keys rotated as
#   interleaved pairs), the first `index_init_tokens` keys and the query's last
#   `index_local_tokens` set to +inf, and the `index_topk` best kept. Attentions whose index is
#   not a multiple of `cli_factor` reuse the last choice (`models/deepseek_v32.py`,
#   `dsa_layer_has_indexer`). SGLang's top-k is a CUDA kernel whose order among equal scores is
#   not documented: the earlier key wins here.
# - `oe_ignore_tokens` (`layers/embedding/over_embedding.py`, and the kernels
#   `update_oe_token_table` / `compute_n_gram_ids` of flash-npu-kernel): such a token keeps its
#   word embedding alone, and ends the n-grams of the tokens after it as an end-of-sequence
#   token does. `NgramEmbedding._shift_right_ignore_eos` is given those extra boundaries.
# - The epsilon of `q_a_layernorm` / `kv_a_layernorm`: `rms_norm_eps` in SGLang
#   (`models/deepseek_v32.py`), the module default 1e-6 in transformers.
# - `rope_type: deepseek_yarn` is run as transformers' `yarn`; `sglang_inv_freq` below is
#   SGLang's `DeepseekScalingRotaryEmbedding._compute_inv_freq`, asserted equal.
#
# usage: ref_longcat.py MODEL_DIR CODE_DIR OUT.safetensors N_PROMPT N_DECODE [SEED] [--chunk N] [--ids A,B,..]
#   --chunk: the prompt as passes of N tokens (default: one pass). --ids: these prompt tokens
#   (N_PROMPT is then ignored) instead of random ones, as `oxinfer --show-tokens` prints them.
#   Writes `ids`, `logits` [n, vocab] (every prompt position), `gen` (greedy tokens) and
#   `decode_logits` [len(gen), vocab] (after feeding each of them, one token per pass).
import importlib, json, math, os, sys, types

import torch
from torch import nn


def sglang_inv_freq(dim, base, factor, original_max, beta_fast=32, beta_slow=1):
    """`DeepseekScalingRotaryEmbedding._compute_inv_freq` (SGLang-FluentLLM)."""
    def find_dim(num_rotations):
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low, high = max(math.floor(find_dim(beta_fast)), 0), min(math.ceil(find_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    return (1.0 / (factor * pos_freqs)) * (1 - mask) + (1.0 / pos_freqs) * mask


def ignore_tokens(spec):
    """`FLASHConfig.build_oe_ignore_tokens`."""
    out = set()
    if isinstance(spec, str):
        for part in filter(None, (p.strip() for p in spec.split(","))):
            a, sep, b = part.partition(":")
            out.update(range(int(a), int(b)) if sep else [int(a)])
    elif spec:
        out.update(int(t) for t in spec)
    return sorted(out)


def oe_settings(cfg):
    """(ratio, neighbors, split) under LongCat-2.x's names or LongCat-Flash-Lite's."""
    if cfg.get("oe_vocab_size_ratio") is not None:
        return cfg["oe_vocab_size_ratio"], cfg["oe_neighbor_num"], cfg["oe_split_num"]
    if cfg.get("ngram_vocab_size_ratio") is not None:
        return cfg["ngram_vocab_size_ratio"], cfg["emb_neighbor_num"], cfg["emb_split_num"]
    return None


def ngram_module(code_dir):
    pkg = types.ModuleType("longcat_ref")
    pkg.__path__ = [code_dir]
    sys.modules["longcat_ref"] = pkg
    return importlib.import_module("longcat_ref.modeling_longcat_ngram")


class Indexer(nn.Module):
    """LongCat-2.x's lightning indexer (see the header)."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.get("index_k_norm_type", "layer") == "rms" and cfg.get("indexer_rope_interleave", True)
        self.heads, self.dim, self.topk = cfg["index_n_heads"], cfg["index_head_dim"], cfg["index_topk"]
        self.init, self.local = cfg.get("index_init_tokens", 0), cfg.get("index_local_tokens", 0)
        self.rope = cfg["qk_rope_head_dim"]
        self.wq_b = nn.Linear(cfg["q_lora_rank"], self.heads * self.dim, bias=False)
        self.wk = nn.Linear(cfg["hidden_size"], self.dim, bias=False)
        self.k_norm = nn.Parameter(torch.ones(self.dim))
        self.weights_proj = nn.Linear(cfg["hidden_size"], self.heads, bias=False)

    def rotate(self, x, cos, sin):
        """The first `rope` lanes of `x` [..., t, (heads,) dim] as interleaved pairs, in place."""
        c, s = cos[..., : self.rope // 2], sin[..., : self.rope // 2]
        if x.dim() == 4:
            c, s = c.unsqueeze(2), s.unsqueeze(2)
        x1, x2 = x[..., 0:self.rope:2].clone(), x[..., 1:self.rope:2].clone()
        x[..., 0:self.rope:2] = x1 * c - x2 * s
        x[..., 1:self.rope:2] = x2 * c + x1 * s
        return x

    def choose(self, attn, x, cos, sin, store):
        """The keys each of the `t` new queries attends to: bool [b, t, n] (n keys so far)."""
        b, t, _ = x.shape
        cq = attn.q_a_layernorm(attn.q_a_proj(x))
        qi = self.rotate(self.wq_b(cq).view(b, t, self.heads, self.dim), cos, sin)
        k = self.wk(x)
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6) * self.k_norm
        keys = self.rotate(k, cos, sin)
        if attn.layer_idx in store:
            keys = torch.cat([store[attn.layer_idx], keys], dim=1)
        store[attn.layer_idx] = keys
        n = keys.shape[1]
        w = self.weights_proj(x) * (self.heads ** -0.5) * (self.dim ** -0.5)
        score = (torch.einsum("bthd,bsd->bths", qi, keys).relu() * w.unsqueeze(-1)).sum(2)
        last = (n - t + torch.arange(t)).view(1, t, 1)  # each query's own position
        s = torch.arange(n).view(1, 1, n)
        visible = s <= last
        score = score.masked_fill(~visible, float("-inf"))
        forced = ((s < self.init) | (s > last - self.local)) & visible
        score = score.masked_fill(forced, float("inf"))
        top = score.sort(dim=-1, descending=True, stable=True).indices[..., : min(self.topk, n)]
        return torch.zeros(b, t, n, dtype=torch.bool).scatter_(2, top, True) & visible


def build(cfg, code_dir):
    """The model of config.json `cfg` in float32: transformers' LongcatFlash model, with the
    n-gram embedding and the indexers attached as `model.model.ngram` / `self_attn[i].indexer`."""
    from transformers import LongcatFlashConfig, LongcatFlashForCausalLM
    from transformers.models.longcat_flash import modeling_longcat_flash as M

    flash = cfg["architectures"][0].startswith("LongcatFlash")
    rs = cfg.get("rope_scaling")
    rope = {"rope_type": "default", "rope_theta": cfg["rope_theta"]}
    if rs:
        assert rs.get("mscale", 1) == rs.get("mscale_all_dim", 1) == 1
        rope = {"rope_type": "yarn", "rope_theta": cfg["rope_theta"], "factor": rs["factor"],
                "original_max_position_embeddings": rs["original_max_position_embeddings"],
                "beta_fast": rs.get("beta_fast", 32), "beta_slow": rs.get("beta_slow", 1),
                "mscale": rs.get("mscale", 1), "mscale_all_dim": rs.get("mscale_all_dim", 1)}
    hc = LongcatFlashConfig(
        vocab_size=cfg["vocab_size"], hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
        num_attention_heads=cfg["num_attention_heads"], ffn_hidden_size=cfg["ffn_hidden_size"],
        q_lora_rank=cfg["q_lora_rank"], kv_lora_rank=cfg["kv_lora_rank"],
        qk_nope_head_dim=cfg["qk_nope_head_dim"], qk_rope_head_dim=cfg["qk_rope_head_dim"],
        head_dim=cfg["qk_rope_head_dim"], v_head_dim=cfg["v_head_dim"], moe_topk=cfg["moe_topk"],
        n_routed_experts=cfg["n_routed_experts"], zero_expert_num=cfg.get("zero_expert_num", 0),
        expert_ffn_hidden_size=cfg["expert_ffn_hidden_size"],
        routed_scaling_factor=float(cfg["routed_scaling_factor"]), rms_norm_eps=cfg["rms_norm_eps"],
        max_position_embeddings=cfg["max_position_embeddings"], rope_parameters=rope,
        bos_token_id=cfg.get("bos_token_id", 1), eos_token_id=cfg.get("eos_token_id", 2),
        tie_word_embeddings=False,
    )
    model = LongcatFlashForCausalLM(hc)
    hc._attn_implementation = "eager"
    if rs:
        got = model.model.rotary_emb.inv_freq
        want = sglang_inv_freq(cfg["qk_rope_head_dim"], cfg["rope_theta"], rs["factor"], rs["original_max_position_embeddings"],
                               rs.get("beta_fast", 32), rs.get("beta_slow", 1))
        assert torch.allclose(got, want, rtol=1e-6, atol=0), (got, want)
        assert model.model.rotary_emb.attention_scaling == 1.0
    attns = [a for layer in model.model.layers for a in layer.self_attn]
    mscale = 0.1 * math.log(rs["factor"]) + 1.0 if rs else 1.0
    for a in attns:
        assert abs(a.scaling - (cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]) ** -0.5 * mscale * mscale) < 1e-12
        if not cfg.get("mla_scale_q_lora", flash):
            a.mla_scale_q_lora = 1.0
        if not cfg.get("mla_scale_kv_lora", flash):
            a.mla_scale_kv_lora = 1.0
        if not flash:
            # SGLang-FluentLLM: `RMSNorm(q_lora_rank, eps=config.rms_norm_eps)`.
            a.q_a_layernorm.variance_epsilon = a.kv_a_layernorm.variance_epsilon = cfg["rms_norm_eps"]

    oe = oe_settings(cfg)
    ignore = ignore_tokens(cfg.get("oe_ignore_tokens")) if not flash else []
    if oe:
        N = ngram_module(code_dir)
        ncfg = types.SimpleNamespace(
            ngram_vocab_size_ratio=oe[0], emb_neighbor_num=oe[1], emb_split_num=oe[2], vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"], pad_token_id=None, eos_token_id=cfg.get("eos_token_id", 2))

        class NgramEmbedding(N.NgramEmbedding):
            # `oe_ignore_tokens`: an ignored token ends the n-grams of what follows, as an
            # end-of-sequence token does (it is negative in SGLang's token table, where the hash
            # stops), and keeps its word embedding alone.
            def _shift_right_ignore_eos(self, tensor, n, eos_token_id=2):
                marked = tensor.clone()
                for t in ignore:
                    marked[tensor == t] = eos_token_id
                out = super()._shift_right_ignore_eos(marked, n, eos_token_id=eos_token_id)
                # (The tokens shifted in are the original ones; an ignored one is never shifted
                # in past itself, and its own row is replaced below.)
                return out

            def forward(self, input_ids, ngram_context=None):
                x = super().forward(input_ids, ngram_context=ngram_context)
                if ignore:
                    flags = torch.isin(input_ids, torch.tensor(ignore))
                    x = torch.where(flags.unsqueeze(-1), self.word_embeddings(input_ids), x)
                return x

        model.model.ngram = (NgramEmbedding if ignore else N.NgramEmbedding)(ncfg, model.model.embed_tokens)

    if cfg.get("index_topk") is not None:
        assert not flash
        every = cfg.get("cli_factor", 1)
        for a in attns:
            if a.layer_idx % every == 0:
                a.indexer = Indexer(cfg)
        mla_forward = M.LongcatFlashMLA.forward
        shared = {}

        def sparse_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
            b, t, _ = hidden_states.shape
            ix = getattr(self, "indexer", None)
            if ix is not None:
                cos, sin = position_embeddings
                store = past_key_values.__dict__.setdefault("index_keys", {})
                shared["chosen"] = ix.choose(self, hidden_states, cos, sin, store)
            chosen = shared["chosen"]
            mask = torch.zeros(b, 1, t, chosen.shape[-1]).masked_fill(~chosen.unsqueeze(1), float("-inf"))
            return mla_forward(self, hidden_states, position_embeddings, mask, past_key_values, **kw)

        M.LongcatFlashMLA.forward = sparse_forward
    return model.float().eval()


def load(model, cfg, d):
    """Loads the released layout (one module per expert, `oe_embed_*` / `ngram_embeddings.*`)."""
    from safetensors.torch import load_file

    sd = {k: v.float() for k, v in load_file(f"{d}/model.safetensors").items() if not k.startswith("model.mtp.")}
    ne, nz = cfg["n_routed_experts"], cfg.get("zero_expert_num", 0)
    out = {}
    for l in range(cfg["num_layers"]):
        p = f"model.layers.{l}.mlp.experts."
        gu = [torch.cat([sd.pop(f"{p}{e}.gate_proj.weight"), sd.pop(f"{p}{e}.up_proj.weight")]) for e in range(ne)]
        # (transformers allocates gate / up rows for the identity experts too; they are not read.)
        gu += [torch.zeros_like(gu[0])] * nz
        out[f"{p}gate_up_proj"] = torch.stack(gu)
        out[f"{p}down_proj"] = torch.stack([sd.pop(f"{p}{e}.down_proj.weight") for e in range(ne)])
    for k, v in sd.items():
        if k.startswith("model.oe_embed_tokens"):
            k = f"model.ngram.embedders.{k[len('model.oe_embed_tokens'):-len('.weight')]}.weight"
        elif k.startswith("model.oe_embed_proj"):
            k = f"model.ngram.post_projs.{k[len('model.oe_embed_proj'):-len('.weight')]}.weight"
        elif k.startswith("model.ngram_embeddings."):
            k = k.replace("model.ngram_embeddings.", "model.ngram.")
        elif k.endswith("indexer.k_norm.weight"):
            k = k[:-len(".weight")]
        out[k] = v
    missing, unexpected = model.load_state_dict(out, strict=False)
    # (`model.ngram.word_embeddings` is `model.embed_tokens`.)
    missing = [m for m in missing if not m.startswith("model.ngram.word_embeddings")]
    assert not missing and not unexpected, (missing, unexpected)


def embed(model, cfg, ids, seen):
    """Input embeddings of `ids` [1, t] after the tokens `seen` [1, s]."""
    ng = getattr(model.model, "ngram", None)
    if ng is None:
        return model.model.embed_tokens(ids)
    n = oe_settings(cfg)[1]
    return ng(ids, ngram_context=seen[:, -(n - 1):] if seen.shape[1] else None)


def prompt_ids(cfg, n, seed):
    """Random tokens with end-of-sequence tokens and `oe_ignore_tokens` among them."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(4, cfg["vocab_size"], (1, n), generator=g)
    eos = cfg.get("eos_token_id", 2)
    for p in (n // 6, n // 6 + 1, n // 2 + 3):
        ids[0, p] = eos
    ign = ignore_tokens(cfg.get("oe_ignore_tokens"))
    if ign:
        for i, p in enumerate((n // 4, n // 4 + 1, n // 4 + 5, 2 * n // 3)):
            ids[0, p] = ign[(7 * i) % len(ign)]
    return ids


def main():
    from safetensors.torch import save_file

    d, code, out, n, nd = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    seed = int(sys.argv[6]) if len(sys.argv) > 6 and not sys.argv[6].startswith("--") else 0
    chunk = int(sys.argv[sys.argv.index("--chunk") + 1]) if "--chunk" in sys.argv else 1 << 30
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float32)
    cfg = json.load(open(f"{d}/config.json"))
    model = build(cfg, code)
    load(model, cfg, d)
    if "--ids" in sys.argv:
        ids = torch.tensor([[int(x) for x in sys.argv[sys.argv.index("--ids") + 1].replace(" ", "").split(",")]])
        n = ids.shape[1]
    else:
        ids = prompt_ids(cfg, n, seed)
    res = {"ids": ids[0].to(torch.int64)}
    with torch.no_grad():
        past, logits = None, []
        for p in range(0, n, chunk):
            x = embed(model, cfg, ids[:, p:p + chunk], ids[:, :p])
            o = model(inputs_embeds=x, past_key_values=past, use_cache=True)
            past = o.past_key_values
            logits.append(o.logits[0].float())
        res["logits"] = torch.cat(logits)
        nxt = res["logits"][-1].argmax()
        gen, steps, seen = [], [], ids
        for _ in range(nd):
            gen.append(int(nxt))
            x = embed(model, cfg, nxt.view(1, 1), seen)
            seen = torch.cat([seen, nxt.view(1, 1)], dim=1)
            o = model(inputs_embeds=x, past_key_values=past, use_cache=True)
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
