#!/usr/bin/env python3
"""Tiny random MiniMax checkpoints + transformers reference outputs (float32, CPU, eager).

usage: minimax_tiny.py OUT_DIR [--arch m2] [--gen N] [--seed S] [--fp8]

--arch m2: MiniMax-M2 / M2.1 / M2.5 / M2.7 (`minimax_m2`): full attention with query / key norms
  over all heads, partial RoPE, sigmoid-routed experts with a selection bias, no shared expert.
--fp8: the attention and expert projections stored as the releases store them (float8_e4m3fn with
  one `weight_scale_inv` per 32 x 32 block); the reference runs on the dequantized values.

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
else:
    sys.exit(f"unknown --arch {arch}")

cfg._attn_implementation = "eager"
m = Model(cfg).float().eval()


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
m.save_pretrained(mdir)
if fp8:
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
        o = m(torch.tensor([[t]]), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        rows.append(o.logits[0, -1].float())
    torch.stack(rows).numpy().tofile(os.path.join(out, "ref.bin"))
open(os.path.join(out, "prompt.txt"), "w").write(" ".join(str(int(x)) for x in prompt))
open(os.path.join(out, "gen.txt"), "w").write(" ".join(str(x) for x in toks))
print("saved tensors:", sorted(load_file(os.path.join(mdir, "model.safetensors")).keys())[:12], "...")
print("prompt", prompt.tolist())
print("gen", toks)
print("cached vs uncached last logit diff", float((full[-1] - rows[0]).abs().max()))
