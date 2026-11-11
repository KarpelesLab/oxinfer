#!/usr/bin/env python3
"""MiniMax-VL-01 image input against MiniMax's own code, on a tiny random checkpoint in the
release's layout (config.json of `minimax_vl_01`, `language_model.*`, `vision_tower.*`,
`multi_modal_projector.*`, `image_newline`) with the real tokenizer.

The reference: MiniMax's `image_processor.py` (anyres views, Pillow), a stock CLIP vision model
(MiniMax's `modeling_clip.py` encodes each view on its own the same way), the projector, and
`pack_image_features` / `unpad_image` / `get_anyres_image_grid_shape` as in
`modeling_minimax_vl_01.py`; the text model is transformers' `minimax` with MiniMax's partial RoPE
(see minimax_tiny.py --arch 01).

usage: minimax_vl01_tiny.py build OUT_DIR VL01_DIR      (VL01_DIR: tokenizer, preprocessor config)
       minimax_vl01_tiny.py check OUT_DIR VL01_CODE_DIR IMAGE PROMPT_FILE OURS_PREFIX OURS_LOGITS
         VL01_CODE_DIR holds MiniMax's image_processor.py.
"""
import json, os, shutil, sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import CLIPVisionConfig, CLIPVisionModel, MiniMaxConfig, MiniMaxForCausalLM
from transformers.models.minimax import modeling_minimax as mmod

mode, out = sys.argv[1], sys.argv[2]
V = 200064


def partial_rope(rd):
    def f(q, k, cos, sin, unsqueeze_dim=1, **_):
        cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
        r = cos.shape[-1]
        rot = lambda x: torch.cat((-x[..., r // 2 : r], x[..., : r // 2]), dim=-1)
        return (torch.cat([q[..., :r] * cos + rot(q) * sin, q[..., r:]], dim=-1),
                torch.cat([k[..., :r] * cos + rot(k) * sin, k[..., r:]], dim=-1))
    return f


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


def text_model(cfg_dict):
    c = MiniMaxConfig(**cfg_dict)
    c._attn_implementation = "eager"
    m = MiniMaxForCausalLM(c).float().eval()
    rd = cfg_dict["rotary_dim"]
    m.model.rotary_emb.inv_freq = 1.0 / (cfg_dict["rope_theta"] ** (torch.arange(0, rd, 2).float() / rd))
    mmod.apply_rotary_pos_emb = partial_rope(rd)
    return m


TEXT = dict(vocab_size=V, hidden_size=128, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, num_local_experts=4, num_experts_per_tok=2,
            layer_types=["linear_attention", "full_attention", "linear_attention"], block_size=16,
            rotary_dim=16, rope_theta=10000.0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
            full_attn_alpha_factor=3.56, linear_attn_alpha_factor=3.56, mlp_alpha_factor=3.56, postnorm=True,
            rms_norm_eps=1e-5, tie_word_embeddings=False, initializer_range=0.08)
VISION = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, image_size=336,
              patch_size=14, hidden_act="gelu", projection_dim=128)

if mode == "build":
    src = sys.argv[3]
    torch.manual_seed(0)
    os.makedirs(out, exist_ok=True)
    tm = text_model(TEXT)
    with torch.no_grad():
        for n, p in tm.named_parameters():
            if "norm" in n:
                p.copy_(1.0 + torch.randn_like(p) * 0.2)
        for p in tm.parameters():
            p.copy_(p.to(torch.bfloat16).float())
    # The reference reloads the text model from here (bf16-representable, as the checkpoint).
    tm.save_pretrained(out + "/_ref_text")
    vm = CLIPVisionModel(CLIPVisionConfig(**VISION)).float().eval()
    w = {}
    for k, v in load_file(out + "/_ref_text/model.safetensors").items():
        w["language_model." + k] = v
    for k, v in vm.state_dict().items():
        if "post_layernorm" not in k and "position_ids" not in k:
            # The release's names (transformers 5's CLIPVisionModel has no `vision_model.` level).
            w["vision_tower.vision_model." + k.removeprefix("vision_model.")] = v
    g = torch.Generator().manual_seed(1)
    for name, shape in [("multi_modal_projector.linear_1", (128, 64)), ("multi_modal_projector.linear_2", (128, 128))]:
        w[name + ".weight"] = torch.randn(shape, generator=g) * 0.1
        w[name + ".bias"] = torch.randn(shape[0], generator=g) * 0.05
    w["image_newline"] = torch.randn(128, generator=g)
    w = {k: v.to(torch.bfloat16).float().contiguous() for k, v in w.items()}
    save_file(w, out + "/model.safetensors", metadata={"format": "pt"})
    tc = {k: v for k, v in TEXT.items() if k not in ("layer_types", "rope_parameters")}
    tc.update(model_type="minimax_text_01", attn_type_list=[int(t == "full_attention") for t in TEXT["layer_types"]],
              layernorm_full_attention_alpha=3.56, layernorm_linear_attention_alpha=3.56, layernorm_mlp_alpha=3.56,
              layernorm_full_attention_beta=1.0, layernorm_linear_attention_beta=1.0, layernorm_mlp_beta=1.0)
    cfg = {"architectures": ["MiniMaxVL01ForConditionalGeneration"], "model_type": "minimax_vl_01",
           "image_token_index": 200025, "projector_hidden_act": "gelu", "vision_feature_layer": -1,
           "vision_feature_select_strategy": "default",
           "image_grid_pinpoints": json.load(open(src + "/config.json"))["image_grid_pinpoints"],
           "text_config": tc, "vision_config": dict(VISION, model_type="clip_vision_model")}
    json.dump(cfg, open(out + "/config.json", "w"), indent=2)
    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.json", "preprocessor_config.json"]:
        shutil.copy(os.path.join(src, f), out)
    print("saved", out)
    sys.exit(0)

code, image, prompt_file, ours_prefix, ours_logits = sys.argv[3:8]
sys.path.insert(0, code)
import image_processor as mip
from PIL import Image

cfg = json.load(open(out + "/config.json"))
w = load_file(out + "/model.safetensors")
pc = json.load(open(out + "/preprocessor_config.json"))
proc = mip.ImageProcessor(**{k: v for k, v in pc.items() if k not in ("auto_map", "processor_class")})
img = Image.open(image).convert("RGB")
pp = proc.preprocess([img], return_tensors=None)
views = torch.tensor(np.stack(pp["pixel_values"])).float()
H, W = pp["image_sizes"][0]
vm = CLIPVisionModel(CLIPVisionConfig(**VISION)).float().eval()
r = vm.load_state_dict({k[len("vision_tower.vision_model."):]: v for k, v in w.items() if k.startswith("vision_tower.")}, strict=False)
assert all("post_layernorm" in k for k in r.missing_keys) and not r.unexpected_keys, r
with torch.no_grad():
    f = vm(views, output_hidden_states=True).hidden_states[-1][:, 1:]
    f = torch.nn.functional.gelu(f @ w["multi_modal_projector.linear_1.weight"].T + w["multi_modal_projector.linear_1.bias"])
    f = f @ w["multi_modal_projector.linear_2.weight"].T + w["multi_modal_projector.linear_2.bias"]


# modeling_minimax_vl_01.py: unpad_image and pack_image_features (one image), verbatim in substance.
def unpad_image(tensor, original_size):
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]
    if original_width / original_height > current_width / current_height:
        new_height = int(original_height * current_width) // original_width
        padding = (current_height - new_height) // 2
        return tensor[:, padding : current_height - padding, :]
    new_width = int(original_width * current_height) // original_height
    padding = (current_width - new_width) // 2
    return tensor[:, :, padding : current_width - padding]


best_w, best_h = mip.select_best_resolution((W, H), cfg["image_grid_pinpoints"])
nph, npw = best_h // 336, best_w // 336
base, tiles = f[0], f[1:]
g = 24 if VISION["image_size"] == 336 else None
t = tiles.view(nph, npw, g, g, -1).permute(4, 0, 2, 1, 3).contiguous().flatten(1, 2).flatten(2, 3)
t = unpad_image(t, (H, W))
nl = w["image_newline"]
t = torch.cat((t, nl[:, None, None].expand(*t.shape[:-1], 1)), dim=-1).flatten(1, 2).transpose(0, 1)
feats = torch.cat((base, t), dim=0)


def cmp(name, a, b):
    a, b = np.asarray(a, np.float32).ravel(), np.asarray(b, np.float32).ravel()
    if a.size != b.size:
        print(f"{name}: size ours {a.size} vs ref {b.size}")
        return
    d = np.abs(a - b)
    print(f"{name}: max abs {d.max():.3e}, mean abs {d.mean():.3e}, ref scale {np.abs(b).mean():.3e}")


cmp("pixels", np.fromfile(f"{ours_prefix}_pixels.bin", "<f4"), views.numpy())
cmp("image features", np.fromfile(f"{ours_prefix}_image.bin", "<f4"), feats.numpy())
# The tokenizer.json itself: transformers 5 loads this checkpoint's `GPT2Tokenizer` class with
# GPT-2's split pattern instead of the file's (MiniMax's) one ("name=user" -> "=", "user").
from tokenizers import Tokenizer

ids = torch.tensor([Tokenizer.from_file(out + "/tokenizer.json").encode(open(prompt_file).read(), add_special_tokens=False).ids])
mask = ids[0] == cfg["image_token_index"]
print("prompt tokens", ids.shape[1], "image tokens", int(mask.sum()), "features", feats.shape[0])
tm = MiniMaxForCausalLM.from_pretrained(out + "/_ref_text", experts_implementation="eager").float().eval()
minimax_slopes(tm)
rd = TEXT["rotary_dim"]
tm.model.rotary_emb.inv_freq = 1.0 / (TEXT["rope_theta"] ** (torch.arange(0, rd, 2).float() / rd))
mmod.apply_rotary_pos_emb = partial_rope(rd)
with torch.no_grad():
    emb = tm.get_input_embeddings()(ids.clamp(max=V - 1))
    emb[0, mask] = feats
    logits = tm(inputs_embeds=emb).logits[0, -1]
    # The text model alone: the reference on oxinfer's image features.
    emb[0, mask] = torch.tensor(np.fromfile(f"{ours_prefix}_image.bin", "<f4")).view(-1, feats.shape[1])
    logits_ours_feats = tm(inputs_embeds=emb).logits[0, -1]
    # Float64 ground truth (summation order then matters far less).
    tm64 = tm.double()
    emb64 = emb.double()
    emb64[0, mask] = feats.double()
    logits64 = tm64(inputs_embeds=emb64).logits[0, -1].float()
ours = np.fromfile(ours_logits, "<f4")
cmp("reference f32 vs float64", logits.numpy(), logits64.numpy())
cmp("oxinfer vs float64", ours, logits64.numpy())
cmp("last logits", ours, logits.numpy())
cmp("last logits (reference text model on our image features)", ours, logits_ours_feats.numpy())
print("argmax ours", int(ours.argmax()), "ref", int(logits.argmax()))
