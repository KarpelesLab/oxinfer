#!/usr/bin/env python3
"""Tiny random MiniMax checkpoints + transformers reference outputs (float32, CPU, eager).

usage: minimax_tiny.py OUT_DIR [--arch m2|m3|01|qwen2] [--release] [--gen N] [--seed S] [--fp8]

--arch m2: MiniMax-M2 / M2.1 / M2.5 / M2.7 (`minimax_m2`): full attention with query / key norms
  over all heads, partial RoPE, sigmoid-routed experts with a selection bias, no shared expert.
--arch m3: MiniMax-M3 / M3.1 (text): a dense leading layer, then MoE layers with a shared expert,
  clamped swigluoai and the MSA lightning indexer (blocks of 4, top 3: sparse past 12 positions).
--release (m3): the tensors and config.json as MiniMax's release lays them out.
--fp8: M2: the projections as float8_e4m3fn with one `weight_scale_inv` per 32 x 32 block; M3
  --release: MXFP8 projections (E8M0 per 32) and NVFP4 experts (compressed-tensors), as
  MiniMax-M3.1-Flash (NVFP4=modelopt: ModelOpt's packaging, as MiniMax-H3-Image-PE). The
  reference runs on the dequantized values.

Writes OUT_DIR/model, OUT_DIR/prompt.txt, OUT_DIR/ref.bin ([1+N, vocab] f32: the prompt's last
position, then after each greedy token via the KV cache), OUT_DIR/gen.txt and OUT_DIR/prefill.bin
([len(prompt), vocab]: one uncached forward over the prompt). Then:
  cargo run --release --example qwen4exp_check -- OUT_DIR --cpu
"""
import json, os, sys

import torch
from safetensors.torch import load_file, save_file

out = sys.argv[1]
args = sys.argv[2:]


def opt(name, default):
    return type(default)(args[args.index(name) + 1]) if name in args else default


arch = opt("--arch", "m2")
gen = opt("--gen", 12)
seed = opt("--seed", 0)
fp8 = "--fp8" in args
torch.manual_seed(seed)
V = 512
EOS = 7
FP8_BLOCK = 32

if arch == "m2":
    from transformers import MiniMaxM2Config, MiniMaxM2ForCausalLM

    cfg = MiniMaxM2Config(
        vocab_size=V,
        hidden_size=128,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_local_experts=8,
        num_experts_per_tok=int(os.environ.get("TOPK", "3")),
        rotary_dim=32,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        initializer_range=0.08,
        eos_token_id=EOS,
        bos_token_id=EOS,
        max_position_embeddings=4096,
    )
    Model = MiniMaxM2ForCausalLM
elif arch == "m3":
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLForCausalLM

    # A dense leading layer (no indexer, as the release), then sparse MoE layers whose indexer
    # keeps 3 blocks of 4 positions: a 40-token prompt is well past the dense range.
    n = 4
    cfg = MiniMaxM3VLTextConfig(
        vocab_size=V,
        hidden_size=128,
        intermediate_size=64,
        dense_intermediate_size=192,
        shared_intermediate_size=64,
        num_hidden_layers=n,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_local_experts=8,
        num_experts_per_tok=int(os.environ.get("TOPK", "3")),
        # transformers rotates `partial_rotary_factor` of each head (the release gives both).
        rotary_dim=32,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        routed_scaling_factor=2.0,
        swiglu_alpha=1.702,
        # Low enough that random activations hit the clamp.
        swiglu_limit=float(os.environ.get("LIMIT", "1.0")),
        mlp_layer_types=["dense"] + ["sparse"] * (n - 1),
        layer_types=["full_attention"] + ["minimax_m3_sparse"] * (n - 1),
        index_n_heads=2,
        index_head_dim=64,
        index_block_size=int(os.environ.get("BLOCK", "4")),
        index_topk_blocks=3,
        index_local_blocks=int(os.environ.get("LOCAL", "1")),
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        initializer_range=0.08,
        eos_token_id=EOS,
        bos_token_id=EOS,
    )
    Model = MiniMaxM3VLForCausalLM
elif arch == "01":
    from transformers import MiniMaxConfig, MiniMaxForCausalLM

    # Lightning attention with a softmax attention layer every 3rd, post-norm residuals scaled as
    # the releases (alpha 3.56); the reference's prefill runs in blocks of 4 (its chunked form).
    n = 6
    cfg = MiniMaxConfig(
        vocab_size=V,
        hidden_size=128,
        intermediate_size=64,
        num_hidden_layers=n,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_local_experts=8,
        num_experts_per_tok=2,
        layer_types=[
            {"f": "full_attention", "l": "linear_attention"}[c]
            for c in os.environ.get("LAYERS", "llfllf")[:n]
        ],
        block_size=int(os.environ.get("BLOCK", "4")),
        # MiniMax's code rotates `rotary_dim` of each head; transformers needs the factor for it.
        rotary_dim=16,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        full_attn_alpha_factor=3.5565588200778455,
        linear_attn_alpha_factor=3.5565588200778455,
        mlp_alpha_factor=3.5565588200778455,
        postnorm=True,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        initializer_range=0.08,
        eos_token_id=EOS,
        bos_token_id=EOS,
    )
    Model = MiniMaxForCausalLM
elif arch == "qwen2":
    # MiniMax's SynLogic models are Qwen2.5 fine-tunes: biased q / k / v, no query / key norms.
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=V, hidden_size=128, intermediate_size=192, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, rope_theta=1000000.0, rms_norm_eps=1e-6,
        tie_word_embeddings=False, initializer_range=0.08, eos_token_id=EOS, bos_token_id=EOS,
    )
    Model = Qwen2ForCausalLM
else:
    sys.exit(f"unknown --arch {arch}")

cfg._attn_implementation = "eager"
m = Model(cfg).float().eval()
def minimax_slopes(model):
    """MiniMax's lightning slopes: `slope * (1 - l / (L - 1) + 1e-5)`; transformers' `minimax` puts
    the 1e-5 in the denominator too, which changes the last layer's decay by half (the factor is
    about 1e-5 there), visible over long contexts."""
    import math
    for l, layer in enumerate(model.model.layers):
        a = layer.self_attn
        if not hasattr(a, "slope_rate"):
            continue
        n, L = a.num_attention_heads, model.config.num_hidden_layers
        base = 1 / (2 ** (8 / n))
        rate = torch.tensor([base ** (h + 1) for h in range(n)], dtype=torch.float32)[:, None, None]
        rate = rate * (1 - l / (L - 1) + 1e-5)
        a.slope_rate = rate
        a.query_decay, a.key_decay, a.diagonal_decay = a.decay_factors(rate)


if arch == "01":
    # transformers' `minimax` rotates whole heads; MiniMax's own code (modeling_minimax_m1.py,
    # which llama.cpp and vLLM follow) rotates the first `rotary_dim` dimensions with frequencies
    # over `rotary_dim`. The reference here does as MiniMax's code.
    from transformers.models.minimax import modeling_minimax as mmod

    rd = cfg.rotary_dim
    m.model.rotary_emb.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rd, 2).float() / rd))
    m.model.rotary_emb.original_inv_freq = m.model.rotary_emb.inv_freq

    def partial_rope(q, k, cos, sin, unsqueeze_dim=1, **_):
        cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
        r = cos.shape[-1]
        rot = lambda x: torch.cat((-x[..., r // 2 : r], x[..., : r // 2]), dim=-1)
        q = torch.cat([q[..., :r] * cos + rot(q) * sin, q[..., r:]], dim=-1)
        k = torch.cat([k[..., :r] * cos + rot(k) * sin, k[..., r:]], dim=-1)
        return q, k

    mmod.apply_rotary_pos_emb = partial_rope
    minimax_slopes(m)


def fp8_quant(w):
    """float8_e4m3fn values and per-block inverse scales (w ~= q * s) of a 2D weight."""
    r, c = w.shape
    br, bc = -(-r // FP8_BLOCK), -(-c // FP8_BLOCK)
    s = torch.zeros(br, bc)
    q = torch.zeros(r, c, dtype=torch.float8_e4m3fn)
    for i in range(br):
        for j in range(bc):
            blk = w[i * FP8_BLOCK:(i + 1) * FP8_BLOCK, j * FP8_BLOCK:(j + 1) * FP8_BLOCK]
            sc = blk.abs().max().clamp(min=1e-8) / 448.0
            s[i, j] = sc
            q[i * FP8_BLOCK:(i + 1) * FP8_BLOCK, j * FP8_BLOCK:(j + 1) * FP8_BLOCK] = (blk / sc).to(torch.float8_e4m3fn)
    return q, s


def dequant(q, s):
    w = q.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i * FP8_BLOCK:(i + 1) * FP8_BLOCK, j * FP8_BLOCK:(j + 1) * FP8_BLOCK] *= s[i, j]
    return w


def quantized(name):
    return fp8 and name.endswith(".weight") and ("_proj." in name or ".experts." in name)


with torch.no_grad():
    for n, p in m.named_parameters():
        if n.endswith(".bias"):
            p.copy_(torch.randn_like(p) * 0.3)
        if "norm" in n:
            # M3's norms scale by 1 + w (Gemma style).
            p.copy_((0.0 if arch == "m3" else 1.0) + torch.randn_like(p) * 0.2)
    for n, b in m.named_buffers():
        if n.endswith("e_score_correction_bias"):
            # Large enough to change the choice from the plain top-k of the scores.
            b.copy_(torch.randn_like(b) * 0.3)
    # bf16-representable weights: oxinfer's bf16 matmuls are then exact (FP8 ones: the dequantized
    # values, which the reference then uses).
    for n, p in m.named_parameters():
        p.copy_(p.to(torch.bfloat16).float())
os.makedirs(out, exist_ok=True)
mdir = os.path.join(out, "model")
release = "--release" in args
if release and arch != "m3":
    sys.exit("--release is for --arch m3")
if not release:
    m.save_pretrained(mdir)
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def mxfp8(w):
    """MXFP8 as MiniMax M3.1 stores it: float8_e4m3fn values, an E8M0 exponent byte per 32 along
    each row. Returns (values, exponents, dequantized)."""
    r, c = w.shape
    b = w.reshape(r, c // 32, 32)
    e = torch.ceil(torch.log2(b.abs().amax(-1).clamp(min=1e-30) / 448.0)).clamp(-127, 127)
    q = (b / torch.exp2(e)[..., None]).to(torch.float8_e4m3fn)
    deq = (q.float() * torch.exp2(e)[..., None]).reshape(r, c)
    return q.reshape(r, c), (e + 127).to(torch.uint8), deq


def nvfp4(w):
    """NVFP4 as compressed-tensors stores it: E2M1 codes (two per byte, low nibble first), an
    E4M3 scale per 16, the tensor's global scale (a divisor). Returns (packed, scales, global,
    dequantized)."""
    r, c = w.shape
    g = 448.0 * 6.0 / w.abs().max().clamp(min=1e-30)
    b = w.reshape(r, c // 16, 16)
    s = ((b.abs().amax(-1) / 6.0) * g).to(torch.float8_e4m3fn)
    sf = s.float().clamp(min=1e-30)
    x = b * g / sf[..., None]
    idx = (x.abs()[..., None] - E2M1).abs().argmin(-1)
    codes = idx + 8 * (x < 0).to(idx.dtype)
    deq = (E2M1[idx] * torch.where(x < 0, -1.0, 1.0) * sf[..., None] / g).reshape(r, c)
    codes = codes.reshape(r, c)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed, s, torch.tensor(float(g)), deq


if release:
    # The tensors of MiniMax's release (`language_model.` prefix, `block_sparse_moe`, per-expert
    # w1 / w3 / w2, `index_*`), from the in-memory model; with --fp8 the attention, dense,
    # shared and indexer projections as MXFP8 and the experts as NVFP4, the reference then
    # running on what they hold.
    os.makedirs(mdir, exist_ok=True)
    P = dict(m.named_parameters())
    B = dict(m.named_buffers())
    out_t = {}

    def put(name, view, kind):
        """Stores `view` (a slice of a reference parameter) under `name`, quantized as `kind`."""
        w = view.detach().float()
        if fp8 and kind == "mx" and w.shape[1] % 32 == 0:
            q, e, deq = mxfp8(w)
            out_t[name] = q
            out_t[name + "_scale_inv"] = e
            view.data.copy_(deq)
        elif fp8 and kind == "nv":
            pk, sc, gs, deq = nvfp4(w)
            base = name[: -len(".weight")]
            if os.environ.get("NVFP4") == "modelopt":
                # ModelOpt's packaging: the pairs as `.weight`, the tensor scale as a factor.
                out_t[base + ".weight"] = pk
                out_t[base + ".weight_scale"] = sc
                out_t[base + ".weight_scale_2"] = (1.0 / gs).reshape(())
                out_t[base + ".input_scale"] = torch.tensor(1.0)
            else:
                out_t[base + ".weight_packed"] = pk
                out_t[base + ".weight_scale"] = sc
                out_t[base + ".weight_global_scale"] = gs
                out_t[base + ".input_global_scale"] = torch.tensor(1.0)
            view.data.copy_(deq)
        else:
            out_t[name] = w.to(torch.bfloat16)

    L = "language_model.model."
    put("language_model.lm_head.weight", P["lm_head.weight"], "plain")
    put(L + "embed_tokens.weight", P["model.embed_tokens.weight"], "plain")
    put(L + "norm.weight", P["model.norm.weight"], "plain")
    mi, si, di = cfg.intermediate_size, cfg.shared_intermediate_size, cfg.dense_intermediate_size
    for l in range(cfg.num_hidden_layers):
        hp, rp = f"model.layers.{l}.", f"{L}layers.{l}."
        for n in ["input_layernorm", "post_attention_layernorm"]:
            put(f"{rp}{n}.weight", P[f"{hp}{n}.weight"], "plain")
        for n in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            put(f"{rp}self_attn.{n}.weight", P[f"{hp}self_attn.{n}.weight"], "mx")
        for n in ["q_norm", "k_norm"]:
            put(f"{rp}self_attn.{n}.weight", P[f"{hp}self_attn.{n}.weight"], "plain")
        if f"{hp}self_attn.indexer.q_proj.weight" in P:
            for n in ["q_proj", "k_proj"]:
                put(f"{rp}self_attn.index_{n}.weight", P[f"{hp}self_attn.indexer.{n}.weight"], "mx")
            for n in ["q_norm", "k_norm"]:
                put(f"{rp}self_attn.index_{n}.weight", P[f"{hp}self_attn.indexer.{n}.weight"], "plain")
        if cfg.mlp_layer_types[l] == "dense":
            gu = P[f"{hp}mlp.gate_up_proj.weight"]
            put(f"{rp}mlp.gate_proj.weight", gu[:di], "mx")
            put(f"{rp}mlp.up_proj.weight", gu[di:], "mx")
            put(f"{rp}mlp.down_proj.weight", P[f"{hp}mlp.down_proj.weight"], "mx")
        else:
            mp = f"{rp}block_sparse_moe."
            put(mp + "gate.weight", P[f"{hp}mlp.gate.weight"], "plain")
            out_t[mp + "e_score_correction_bias"] = B[f"{hp}mlp.gate.e_score_correction_bias"].float()
            gu, dn = P[f"{hp}mlp.experts.gate_up_proj"], P[f"{hp}mlp.experts.down_proj"]
            for e in range(cfg.num_local_experts):
                put(f"{mp}experts.{e}.w1.weight", gu[e, :mi], "nv")
                put(f"{mp}experts.{e}.w3.weight", gu[e, mi:], "nv")
                put(f"{mp}experts.{e}.w2.weight", dn[e], "nv")
            sgu = P[f"{hp}mlp.shared_experts.gate_up_proj.weight"]
            put(f"{mp}shared_experts.gate_proj.weight", sgu[:si], "mx")
            put(f"{mp}shared_experts.up_proj.weight", sgu[si:], "mx")
            put(f"{mp}shared_experts.down_proj.weight", P[f"{hp}mlp.shared_experts.down_proj.weight"], "mx")
    save_file({k: v.contiguous() for k, v in out_t.items()}, os.path.join(mdir, "model.safetensors"), metadata={"format": "pt"})
    tc = {
        "hidden_size": cfg.hidden_size, "intermediate_size": mi, "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads, "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim, "vocab_size": V, "rms_norm_eps": cfg.rms_norm_eps, "use_gemma_norm": True,
        "attention_output_gate": False, "rope_theta": cfg.rope_parameters["rope_theta"],
        "rotary_dim": cfg.rotary_dim, "partial_rotary_factor": cfg.rope_parameters["partial_rotary_factor"],
        "hidden_act": "swigluoai", "use_qk_norm": True, "tie_word_embeddings": False,
        "dense_intermediate_size": di, "shared_intermediate_size": si,
        "num_local_experts": cfg.num_local_experts, "num_experts_per_tok": cfg.num_experts_per_tok,
        "n_shared_experts": 1, "scoring_func": "sigmoid", "use_routing_bias": True,
        "moe_layer_freq": [int(t == "sparse") for t in cfg.mlp_layer_types], "qk_norm_type": "per_head",
        "swiglu_alpha": cfg.swiglu_alpha, "swiglu_limit": cfg.swiglu_limit,
        "routed_scaling_factor": cfg.routed_scaling_factor,
        "sparse_attention_config": {
            "use_sparse_attention": True, "sparse_index_dim": cfg.index_head_dim,
            "sparse_num_index_heads": cfg.index_n_heads, "sparse_topk_blocks": cfg.index_topk_blocks,
            "sparse_block_size": cfg.index_block_size, "sparse_score_type": "max", "sparse_init_block": 0,
            "sparse_local_block": cfg.index_local_blocks,
            "sparse_attention_freq": [1] * cfg.num_hidden_layers,
        },
        "eos_token_id": EOS, "bos_token_id": EOS,
    }
    hub = {"architectures": ["MiniMaxM3SparseForConditionalGeneration"], "model_type": "minimax_m3_vl", "text_config": tc}
    json.dump(hub, open(os.path.join(mdir, "config.json"), "w"), indent=2)
if fp8 and not release:
    # Rewrite the projections as FP8 + block scales, and run the reference on what they hold.
    f = os.path.join(mdir, "model.safetensors")
    sd = load_file(f)
    new = {}
    params = dict(m.named_parameters())

    def target(k):
        """The in-memory tensor (a slice of the stacked experts) a saved weight loads into."""
        if k in params:
            return params[k]
        # `...block_sparse_moe.experts.E.wN.weight` <- `...mlp.experts.{gate_up,down}_proj[E]`
        pre, rest = k.split(".block_sparse_moe.experts.")
        e, w, _ = rest.split(".")
        e = int(e)
        mi = cfg.intermediate_size
        if w == "w2":
            return params[f"{pre}.mlp.experts.down_proj"][e]
        gu = params[f"{pre}.mlp.experts.gate_up_proj"][e]
        return gu[:mi] if w == "w1" else gu[mi:]

    with torch.no_grad():
        for k, v in sd.items():
            if quantized(k) and v.dim() == 2:
                q, s = fp8_quant(v.float())
                new[k] = q
                new[k + "_scale_inv"] = s
                target(k).copy_(dequant(q, s))
            else:
                new[k] = v
    save_file(new, f, metadata={"format": "pt"})
    c = json.load(open(os.path.join(mdir, "config.json")))
    c["quantization_config"] = {
        "quant_method": "fp8", "fmt": "float8_e4m3fn", "activation_scheme": "dynamic",
        "weight_block_size": [FP8_BLOCK, FP8_BLOCK],
    }
    json.dump(c, open(os.path.join(mdir, "config.json"), "w"), indent=2)

g = torch.Generator().manual_seed(seed + 1)
prompt = torch.randint(8, V, (int(os.environ.get("PLEN", "40")),), generator=g)
prompt[5] = EOS
prompt[17] = EOS
ids = prompt.unsqueeze(0)
with torch.no_grad():
    full = m(ids, use_cache=False).logits[0].float()
    full.numpy().tofile(os.path.join(out, "prefill.bin"))
    o = m(ids, use_cache=True)
    cache = o.past_key_values
    rows = [o.logits[0, -1].float()]
    toks = []
    for i in range(gen):
        t = int(rows[-1].argmax())
        toks.append(t)
        # Explicit positions: transformers' MiniMax-01 cache reports length 0 when layer 0 is a
        # lightning layer (as in every release), which would restart decode positions at 0.
        p = torch.tensor([[len(prompt) + i]])
        o = m(torch.tensor([[t]]), past_key_values=cache, use_cache=True, position_ids=p, cache_position=p[0])
        cache = o.past_key_values
        rows.append(o.logits[0, -1].float())
    torch.stack(rows).numpy().tofile(os.path.join(out, "ref.bin"))
open(os.path.join(out, "prompt.txt"), "w").write(" ".join(str(int(x)) for x in prompt))
open(os.path.join(out, "gen.txt"), "w").write(" ".join(str(x) for x in toks))
print("saved tensors:", sorted(load_file(os.path.join(mdir, "model.safetensors")).keys())[:12], "...")
print("prompt", prompt.tolist())
print("gen", toks)
print("cached vs uncached last logit diff", float((full[-1] - rows[0]).abs().max()))
