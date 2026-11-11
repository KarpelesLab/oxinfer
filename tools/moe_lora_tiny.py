"""Tiny random MoE checkpoint + a random PEFT LoRA on its routed experts + reference logits.

usage: moe_lora_tiny.py OUT_DIR [--family qwen|gemma] [--hidden H] [--inter M] [--experts E]
                        [--top-k K] [--layers L] [--tokens T] [--rank R] [--seed S]

The model is transformers' Qwen3_5MoeForCausalLM (routed SiLU experts + shared expert) or
Gemma4ForCausalLM with the MoE block (routed GELU experts beside the dense MLP), whose experts are
stacked 3-D parameters; the adapter is made by peft (`target_parameters` on the experts'
`gate_up_proj` / `down_proj`, `target_modules` on attention and MLP projections) and the reference
is peft's own unmerged forward. Writes:

  OUT_DIR/model          the checkpoint (stacked experts)
  OUT_DIR/model_split    the same weights with one gate_proj / up_proj / down_proj per expert
  OUT_DIR/adapter        the adapter as peft saves it (stacked expert parameters)
  OUT_DIR/adapter_split  the same adapter with one module per expert (experts.E.gate_proj, ...)
  OUT_DIR/prompt.txt     token ids
  OUT_DIR/base.bin       [T, vocab] f32 logits without the adapter
  OUT_DIR/ref.bin        [T, vocab] f32 logits with it

Check with `cargo run --release --example moe_lora_check -- OUT_DIR`.
"""
import sys, os, json, shutil
import torch
from safetensors.torch import load_file, save_file
from peft import LoraConfig, get_peft_model

out = sys.argv[1]
opt = {"family": "qwen", "hidden": 128, "inter": 64, "experts": 8, "top-k": 3, "layers": 4, "tokens": 40, "rank": 4, "seed": 0}
args = sys.argv[2:]
for i in range(0, len(args), 2):
    k = args[i].lstrip("-")
    opt[k] = args[i + 1] if k == "family" else int(args[i + 1])
H, MI, E, K, L, T, R = (opt[k] for k in ("hidden", "inter", "experts", "top-k", "layers", "tokens", "rank"))
torch.manual_seed(opt["seed"])
V = 512
hd = H // 4
STD = 0.08 * (128 / H) ** 0.5

if opt["family"] == "qwen":
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=V, hidden_size=H, num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=2, head_dim=hd,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=hd, linear_value_head_dim=hd,
        moe_intermediate_size=MI, shared_expert_intermediate_size=MI, num_experts=E, num_experts_per_tok=K,
        initializer_range=STD, eos_token_id=7, bos_token_id=7)
    m = Qwen3_5MoeForCausalLM(cfg)
    experts = "mlp.experts"
    modules = ["q_proj", "v_proj", "in_proj_qkv", "shared_expert.down_proj"]
else:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    cfg = Gemma4TextConfig(
        vocab_size=V, hidden_size=H, intermediate_size=2 * H, num_hidden_layers=L, num_attention_heads=4,
        num_key_value_heads=2, head_dim=hd, vocab_size_per_layer_input=V, hidden_size_per_layer_input=0,
        sliding_window=16, final_logit_softcapping=30.0, enable_moe_block=True, num_experts=E, top_k_experts=K,
        moe_intermediate_size=MI, initializer_range=STD)
    m = Gemma4ForCausalLM(cfg)
    experts = "experts"
    modules = ["q_proj", "v_proj", "mlp.down_proj"]
m = m.float().eval()
with torch.no_grad():
    for n, p in m.named_parameters():
        if "norm" in n and p.dim() == 1:
            p.copy_(1 + torch.randn_like(p) * 0.2)
        elif n.endswith("router.scale") or n.endswith("per_expert_scale"):
            p.copy_(1 + torch.randn_like(p) * 0.3)
        elif "conv1d" in n:
            p.copy_(torch.randn_like(p) * 0.3)
        elif "A_log" in n:
            p.copy_(torch.rand_like(p) * 2 - 1)
        elif "dt_bias" in n:
            p.copy_(torch.randn_like(p) * 0.5)
    # bf16-representable weights: oxinfer's bf16 matmuls are then exact.
    for n, p in m.named_parameters():
        p.copy_(p.to(torch.bfloat16).float())
os.makedirs(out, exist_ok=True)
m.save_pretrained(os.path.join(out, "model"))

# The same checkpoint with one module per expert.
sd = load_file(os.path.join(out, "model", "model.safetensors"))
split = {}
for k, v in sd.items():
    if k.endswith("experts.gate_up_proj"):
        p = k[: -len("gate_up_proj")]
        for e in range(E):
            split[f"{p}{e}.gate_proj.weight"] = v[e, :MI].contiguous()
            split[f"{p}{e}.up_proj.weight"] = v[e, MI:].contiguous()
    elif k.endswith("experts.down_proj"):
        p = k[: -len("down_proj")]
        for e in range(E):
            split[f"{p}{e}.down_proj.weight"] = v[e].contiguous()
    else:
        split[k] = v
os.makedirs(os.path.join(out, "model_split"), exist_ok=True)
save_file(split, os.path.join(out, "model_split", "model.safetensors"))
shutil.copy(os.path.join(out, "model", "config.json"), os.path.join(out, "model_split", "config.json"))

g = torch.Generator().manual_seed(opt["seed"] + 1)
ids = torch.randint(8, V, (1, T), generator=g)
with torch.no_grad():
    base = m(ids, use_cache=False).logits[0].float()

lc = LoraConfig(r=R, lora_alpha=2 * R, target_modules=modules,
                target_parameters=[f"{experts}.gate_up_proj", f"{experts}.down_proj"], init_lora_weights=False)
pm = get_peft_model(m, lc).eval()
with torch.no_grad():
    for n, p in pm.named_parameters():
        if "lora_" in n:
            p.copy_(torch.randn_like(p) * (0.5 / p.shape[1] ** 0.5))
    ref = pm(ids, use_cache=False).logits[0].float()
pm.save_pretrained(os.path.join(out, "adapter"))

# The same adapter with one module per expert: peft stacks the experts of a 3-D parameter in
# lora_A [E * r, in] (expert e: rows e * r ...) and lora_B [out, r * E] (expert e, component k:
# column k * E + e), in nested wrappers (`experts`, `experts.base_layer`) told apart by shape.
ad = load_file(os.path.join(out, "adapter", "adapter_model.safetensors"))
per = {}
n_stacked = 0
for k, a in ad.items():
    if not k.endswith(".lora_A.weight"):
        continue
    mod = k[: -len(".lora_A.weight")]
    b = ad[mod + ".lora_B.weight"]
    stem = mod
    while stem.endswith(".base_layer"):
        stem = stem[: -len(".base_layer")]
    if not stem.endswith("experts"):
        per[k] = a
        per[mod + ".lora_B.weight"] = b
        continue
    n_stacked += 1
    r = a.shape[0] // E
    for e in range(E):
        ae = a[e * r:(e + 1) * r].contiguous()
        be = b.reshape(b.shape[0], r, E)[:, :, e].contiguous()
        if tuple(b.shape[:1]) == (2 * MI,) and a.shape[1] == H:
            parts = [("gate_proj", be[:MI]), ("up_proj", be[MI:])]
        else:
            parts = [("down_proj", be)]
        for name, bp in parts:
            per[f"{stem}.{e}.{name}.lora_A.weight"] = ae.clone()
            per[f"{stem}.{e}.{name}.lora_B.weight"] = bp.contiguous()
os.makedirs(os.path.join(out, "adapter_split"), exist_ok=True)
save_file(per, os.path.join(out, "adapter_split", "adapter_model.safetensors"))
shutil.copy(os.path.join(out, "adapter", "adapter_config.json"), os.path.join(out, "adapter_split", "adapter_config.json"))

base.numpy().tofile(os.path.join(out, "base.bin"))
ref.numpy().tofile(os.path.join(out, "ref.bin"))
open(os.path.join(out, "prompt.txt"), "w").write(" ".join(str(int(x)) for x in ids[0]))
print("adapter tensors:", len(ad), "of which stacked expert parameters:", 2 * n_stacked)
print("logits: max |adapter - base|", float((ref - base).abs().max()), " max |base|", float(base.abs().max()))
print("top-1 changed by the adapter at", int((ref.argmax(-1) != base.argmax(-1)).sum()), "of", T, "positions")
