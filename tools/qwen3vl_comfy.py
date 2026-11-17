#!/usr/bin/env python3
"""A tiny random Qwen3-VL as ComfyUI's single-file text encoders of Comfy-Org/MiniMax-H3
(`text_encoders/qwen3vl_32b_minimax_h3_*.safetensors`), with transformers references, for
examples/h3_text_check.rs.

usage: qwen3vl_comfy.py OUT_DIR [--layers L] [--keep K] [--tokenizer TOKENIZER.json --prompt TEXT]
(needs the `comfy_kitchen` package: its quantizers)

MiniMax-H3 conditions on the residual stream after the 50th of the 64 layers, and ComfyUI's files
hold only those 50 layers under `model.layers.N` (no final norm, no head, no configuration). Here:
L layers (default 6), the files hold the first K (default 4) and the reference is transformers'
`hidden_states[K]`, for 14 random token ids or, with `--tokenizer` (the release's
`tokenizer/tokenizer.json`; the vocabulary is then Qwen's 151936), for the ids of TEXT, which is
also written to OUT_DIR/prompt.txt: with `--layers 52 --keep 50` the files are read by the video
pipeline's own text path (`h3_text_check FILE --prompt-file ... --tokenizer-in DIR`).

Writes OUT_DIR/model (the folder, `save_pretrained`), OUT_DIR/ids.txt and
  te_bf16.safetensors          the 4 layers and the embedding in bf16
  te_int8_convrot.safetensors  the seven matrices of each layer as `int8_tensorwise` with ConvRot
                               (groups of 64), the embedding in bf16
  te_nvfp4_awq.safetensors     the seven matrices as NVFP4 in comfy_kitchen's packing, AWQ scales
                               folded into the norm weights (q / k / v, gate / up) or stored as
                               `pre_quant_scale` (o_proj, down_proj), the embedding as int8 rows
                               with a scale per row
and the references (f32, `[tokens, hidden]`): ref_bf16.bin, ref_int8.bin, ref_nvfp4.bin, the two
quantized ones computed by transformers on the weights comfy_kitchen dequantizes. Then:
  h3_text_check OUT_DIR/model OUT_DIR/ref_bf16.bin --ids OUT_DIR/ids.txt --layer 3
  h3_text_check OUT_DIR/te_nvfp4_awq.safetensors OUT_DIR/ref_nvfp4.bin --ids OUT_DIR/ids.txt --layer 3
"""
import copy, json, os, sys

import numpy as np
import torch
from comfy_kitchen.tensor import TensorCoreNVFP4Layout, TensorWiseINT8Layout
from safetensors.torch import save_file
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

out, args = sys.argv[1], sys.argv[2:]
opt = lambda name, default: type(default)(args[args.index(name) + 1]) if name in args else default
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
tok = opt("--tokenizer", "")
V, H, L, KEEP = (151936 if tok else 512), 128, opt("--layers", 6), opt("--keep", 4)
cfg = Qwen3VLConfig(
    text_config=dict(vocab_size=V, hidden_size=H, intermediate_size=192, num_hidden_layers=L,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6,
                     rope_parameters={"rope_type": "default", "rope_theta": 5000000.0,
                                      "mrope_section": [6, 5, 5], "mrope_interleaved": True},
                     tie_word_embeddings=False),
    vision_config=dict(depth=1, hidden_size=64, intermediate_size=64, num_heads=2, out_hidden_size=H,
                       deepstack_visual_indexes=[]),
    tie_word_embeddings=False,
)
m = Qwen3VLForConditionalGeneration(cfg).float().eval()
with torch.no_grad():
    for n, p in m.named_parameters():
        if "norm" in n:
            p.copy_(1.0 + torch.randn_like(p) * 0.2)
        elif "language_model" in n:
            p.copy_(torch.randn_like(p) * 0.08)
        p.copy_(p.to(torch.bfloat16).float())
m.save_pretrained(os.path.join(out, "model"))
if tok:
    from tokenizers import Tokenizer

    prompt = opt("--prompt", "integrated_multimodal_description: [Shot 1] A red fox trots across a snowy clearing.")
    open(os.path.join(out, "prompt.txt"), "w").write(prompt)
    ids = torch.tensor([Tokenizer.from_file(tok).encode(prompt, add_special_tokens=False).ids])
else:
    ids = torch.randint(0, V, (1, 14))
open(os.path.join(out, "ids.txt"), "w").write(" ".join(str(int(i)) for i in ids[0]))
sd = {k: v.clone() for k, v in m.state_dict().items()}
P = "model.language_model."
MATS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def reference(name, state):
    r = copy.deepcopy(m)
    r.load_state_dict(state)
    with torch.no_grad():
        hs = r.model.language_model(input_ids=ids, output_hidden_states=True).hidden_states[KEEP][0]
    np.asarray(hs, dtype=np.float32).tofile(os.path.join(out, name))
    return hs


def comfy_names(state):
    """The language model's first KEEP layers and embedding under ComfyUI's names, and the tower."""
    o = {}
    for k, v in state.items():
        if k.startswith(P + "layers."):
            if int(k[len(P) + 7:].split(".")[0]) < KEEP:
                o["model." + k[len(P):]] = v
        elif k == P + "embed_tokens.weight":
            o["model.embed_tokens.weight"] = v
        elif k.startswith("model.visual."):
            o[k[len("model."):]] = v
    return o


def conf(**kw):
    return torch.tensor(list(json.dumps(kw).encode()), dtype=torch.uint8)


META = {"minimax_h3_te": json.dumps({"num_hidden_layers": KEEP, "output": f"unnormalized_hidden_after_layer_{KEEP}"})}
bf16 = lambda o: {k: (v if v.dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn) or k.endswith(("weight_scale", "weight_scale_2")) else v.to(torch.bfloat16)).contiguous() for k, v in o.items()}

base = reference("ref_bf16.bin", sd)
save_file(bf16(comfy_names(sd)), os.path.join(out, "te_bf16.safetensors"), META)

# ---- int8 ConvRot ----------------------------------------------------------------------------------
o, dec = comfy_names(sd), dict(sd)
for l in range(KEEP):
    for mat in MATS:
        hf, ck_name = f"{P}layers.{l}.{mat}", f"model.layers.{l}.{mat}"
        q, p = TensorWiseINT8Layout.quantize(sd[hf + ".weight"], per_channel=True, convrot=True, convrot_groupsize=64)
        o[ck_name + ".weight"], o[ck_name + ".weight_scale"] = q, p.scale
        o[ck_name + ".comfy_quant"] = conf(format="int8_tensorwise", convrot=True, convrot_groupsize=64)
        dec[hf + ".weight"] = TensorWiseINT8Layout.dequantize(q, p).float()
save_file(bf16(o), os.path.join(out, "te_int8_convrot.safetensors"), META)
r = reference("ref_int8.bin", dec)
print(f"int8 ConvRot vs bf16: max abs diff {float((r - base).abs().max()):.3e} (values up to {float(base.abs().max()):.2f})")

# ---- NVFP4 with AWQ scales, an int8 embedding ------------------------------------------------------
o, dec = comfy_names(sd), dict(sd)
bf = lambda t: t.to(torch.bfloat16).float()
for l in range(KEEP):
    hp, cp = f"{P}layers.{l}.", f"model.layers.{l}."
    # The smoothing scale of a norm's consumers divides the norm's weight (what the real file's
    # input_layernorm / post_attention_layernorm hold).
    for norm, mats in (("input_layernorm", MATS[:3]), ("post_attention_layernorm", MATS[4:6])):
        s = torch.exp(torch.randn(H) * 0.5)
        nw = bf(sd[f"{hp}{norm}.weight"] / s)
        o[f"{cp}{norm}.weight"] = dec[f"{hp}{norm}.weight"] = nw
        for mat in mats:
            dec[f"{hp}{mat}.weight"] = sd[f"{hp}{mat}.weight"] * s
    for mat in (MATS[3], MATS[6]):
        w = sd[f"{hp}{mat}.weight"]
        pqs = bf(torch.exp(torch.randn(w.shape[1]) * 0.5))
        o[f"{cp}{mat}.pre_quant_scale"] = pqs
        dec[f"{hp}{mat}.weight"] = w / pqs
    for mat in MATS:
        w = dec[f"{hp}{mat}.weight"]
        q, p = TensorCoreNVFP4Layout.quantize(w)
        o[f"{cp}{mat}.weight"], o[f"{cp}{mat}.weight_scale"], o[f"{cp}{mat}.weight_scale_2"] = q, p.block_scale, p.scale
        o[f"{cp}{mat}.comfy_quant"] = conf(format="nvfp4", full_precision_matrix_mult=True)
        d = TensorCoreNVFP4Layout.dequantize(q, p).float()[: w.shape[0], : w.shape[1]]
        pqs = o.get(f"{cp}{mat}.pre_quant_scale")
        dec[f"{hp}{mat}.weight"] = d if pqs is None else d * pqs
q, p = TensorWiseINT8Layout.quantize(sd[P + "embed_tokens.weight"], per_channel=True)
o["model.embed_tokens.weight"], o["model.embed_tokens.weight_scale"] = q, p.scale
o["model.embed_tokens.comfy_quant"] = conf(format="int8_tensorwise")
dec[P + "embed_tokens.weight"] = TensorWiseINT8Layout.dequantize(q, p).float()
save_file(bf16(o), os.path.join(out, "te_nvfp4_awq.safetensors"))
r = reference("ref_nvfp4.bin", dec)
print(f"NVFP4 vs bf16: max abs diff {float((r - base).abs().max()):.3e}")
print("shapes:", {k: tuple(v.shape) for k, v in o.items() if k.startswith("model.layers.0.") and "proj.weight" in k})
print("wrote", out)
