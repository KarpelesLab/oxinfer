#!/usr/bin/env python3
# Reference for Hugging Face DSpark drafts (Qwen3 / Gemma 4 backbones) in float32 torch: runs the
# draft with DeepSeek's own modeling code (DeepSpec `Qwen3DSparkModel` / `Gemma4DSparkModel`
# layers and Markov head) on the target features oxinfer dumped, to check oxinfer's draft logits.
# Per-layer attention masks follow the draft's config as vLLM / speculators train and serve it:
# context positions before the anchor (within the window on sliding layers), and the block (rows
# up to their own on causal layers).
#
# usage: ref_dspark_hf.py DRAFT_DIR TARGET_DIR DUMP_DIR DEEPSPEC_DIR [--own-head]
#   DUMP_DIR: written by `cargo run --release --example dspark_check -- MODEL DRAFT PROMPT DUMP_DIR`:
#     meta.json {"n": context positions, "anchor": token, "k": drafts, "drafts": [...]},
#     feats.f32 [n, n_feat] (target features of positions 0..n), logits.f32 [k, vocab].
#   --own-head: use the draft's embed_tokens / lm_head instead of the target's (as oxinfer with
#     OXINFER_DSPARK_OWN_HEAD=1).
# Prints per slot: max |diff|, relative error, argmax agreement, KL.
import inspect
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

draft_dir, target_dir, dump, deepspec_dir = sys.argv[1:5]
own_head = "--own-head" in sys.argv
sys.path.insert(0, deepspec_dir)
torch.set_num_threads(os.cpu_count())
torch.set_grad_enabled(False)

raw = json.load(open(os.path.join(draft_dir, "config.json")))
cfg = dict(raw)
speculators = "transformer_layer_config" in raw
if speculators:
    cfg.update(raw["transformer_layer_config"])
dfl = cfg.get("dflash_config") or cfg.get("dspark_config") or {}
archs = cfg.get("architectures") or []
gemma = any(a.startswith("Gemma4") for a in archs) or str(cfg.get("model_type", "")).startswith("gemma4")
for k in ("target_layer_ids", "mask_token_id"):
    if k not in cfg and k in dfl:
        cfg[k] = dfl[k]
if speculators:
    cfg["target_layer_ids"] = [i - 1 for i in raw["aux_hidden_state_layer_ids"]]
cfg.setdefault("num_anchors", 512)
cfg.setdefault("markov_rank", 256)
cfg.setdefault("markov_head_type", "vanilla")
cfg.setdefault("enable_confidence_head", False)
cfg.setdefault("confidence_head_with_markov", True)
sample_from_anchor = raw.get("sample_from_anchor", "DFlashDraftModel" not in archs)

if gemma:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    import deepspec.modeling.dspark.gemma4.modeling as gm
    from deepspec.modeling.dspark.gemma4.modeling import Gemma4DSparkModel as Cls

    # Newer transformers take the layer type at call time rather than construction.
    if "layer_type" not in inspect.signature(gm.Gemma4TextRotaryEmbedding.__init__).parameters:
        Orig = gm.Gemma4TextRotaryEmbedding

        class Rotary(Orig):
            def __init__(self, config, layer_type="full_attention", device=None):
                super().__init__(config, device)
                self.lt = layer_type

            def forward(self, x, position_ids, layer_type=None):
                return super().forward(x, position_ids, layer_type or self.lt)

        gm.Gemma4TextRotaryEmbedding = Rotary

    conf = Gemma4TextConfig(**cfg)
else:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel as Cls

    conf = Qwen3Config(**cfg)
for k, v in cfg.items():
    try:
        if not hasattr(conf, k):
            setattr(conf, k, v)
    except Exception:  # per-layer attributes (Gemma 4) cannot be read globally
        pass
conf._attn_implementation = "eager" if not gemma else "sdpa"
model = Cls(conf).float().eval()

sd = {}
for f in sorted(os.listdir(draft_dir)):
    if f.endswith(".safetensors"):
        with safe_open(os.path.join(draft_dir, f), "pt") as st:
            for name in st.keys():
                sd[name.removeprefix("model.")] = st.get_tensor(name).float()
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", [m for m in missing if "embed_tokens" not in m and "lm_head" not in m], "unexpected:", unexpected)


def target_tensor(suffix, avoid=("vision", "audio", "mtp")):
    idx = os.path.join(target_dir, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    for f in files:
        with safe_open(os.path.join(target_dir, f), "pt") as st:
            for name in st.keys():
                if name.endswith(suffix) and not any(a in name for a in avoid):
                    return st.get_tensor(name).float()
    return None


if own_head and "embed_tokens.weight" in sd:
    emb = sd["embed_tokens.weight"]
    head = sd.get("lm_head.weight", emb)
else:
    emb = target_tensor("embed_tokens.weight")
    head = target_tensor("lm_head.weight")
    if head is None:
        head = emb
    for nm, t in (("embed_tokens.weight", emb), ("lm_head.weight", head)):
        if nm in sd:
            d = (sd[nm] - t).abs().max().item()
            print(f"draft {nm} vs target's: max |diff| {d:.3g}")

meta = json.load(open(os.path.join(dump, "meta.json")))
n, anchor, k = meta["n"], meta["anchor"], meta["k"]
feats = torch.from_numpy(np.fromfile(os.path.join(dump, "feats.f32"), dtype=np.float32)).reshape(n, -1)
V = head.shape[0]
ox = torch.from_numpy(np.fromfile(os.path.join(dump, "logits.f32"), dtype=np.float32)).reshape(-1, V)[:k]

block = cfg["block_size"]
rows = block if sample_from_anchor else block
first = 0 if sample_from_anchor else 1
mask_id = cfg["mask_token_id"]
ids = torch.tensor([anchor] + [mask_id] * (rows - 1))
H = cfg["hidden_size"]
# Embedding scale: Gemma 4 targets scale by sqrt(hidden) (DeepSpec's own scaled table, z-lab
# DFlash on Gemma); speculators use the raw table.
target_cfg = json.load(open(os.path.join(target_dir, "config.json")))
target_gemma = str(target_cfg.get("model_type", "")).startswith("gemma")
scale = 1.0 if speculators or not target_gemma else float(target_cfg.get("text_config", target_cfg)["hidden_size"]) ** 0.5
noise = (emb[ids] * scale)[None]

# Per-layer windows / causality (vLLM `_resolve_layer_attention`).
NL = cfg["num_hidden_layers"]
lt = cfg.get("layer_types") or []
use_swa = dfl.get("use_swa", False)
any_sl = any(t == "sliding_attention" for t in lt)
win = dfl.get("swa_window_size") or cfg.get("sliding_window") or 0
causal_all = cfg.get("is_causal", dfl.get("causal"))
masks = []
for li in range(NL):
    sliding = use_swa if (not lt or (use_swa and not any_sl)) else lt[li] == "sliding_attention"
    if causal_all is not None:
        causal = bool(causal_all)
    else:
        causal = (lt[li] == "sliding_attention" if lt else False) and not (speculators and raw.get("sliding_window_non_causal", False))
    m = torch.full((rows, n + rows), float("-inf"))
    lo = max(0, n - win) if sliding else 0
    m[:, lo:n] = 0
    for i in range(rows):
        m[i, n : n + (i + 1 if causal else rows)] = 0
    masks.append(m[None, None])
    print(f"layer {li}: window {win if sliding else 0}, causal {causal}")

pos = torch.arange(n + rows)[None]
th = model.hidden_norm(model.fc(feats[None]))
pe = model.rotary_emb(noise, pos)
h = noise
for li, layer in enumerate(model.layers):
    h = layer(target_hidden_states=th, hidden_states=h, attention_mask=masks[li], position_ids=pos, position_embeddings=pe)
h = model.norm(h)[0]
sel = h[first : first + k]
base = sel @ head.T
cap = cfg.get("final_logit_softcapping")
if cap:
    base = torch.tanh(base / cap) * cap
toks, ref = model.markov_head.sample_block_tokens(
    base[None], first_prev_token_ids=torch.tensor([anchor]), hidden_states=sel[None], temperature=0.0
)
ref, toks = ref[0], toks[0].tolist()
print("ref drafts", toks, "oxinfer drafts", meta["drafts"])
for i in range(k):
    a, b = ox[i].double(), ref[i].double()
    diff = (a - b).abs().max().item()
    rel = ((a - b).norm() / b.norm()).item()
    p = torch.log_softmax(b, -1)
    q = torch.log_softmax(a, -1)
    kl = (p.exp() * (p - q)).sum().item()
    print(f"slot {i}: max|diff| {diff:.3g} rel {rel:.3g} argmax {int(a.argmax())} vs {int(b.argmax())} KL {kl:.3g}")
if model.confidence_head is not None:
    prev = torch.tensor([anchor] + toks[:-1])
    conf = model.predict_confidence_step(sel[None], prev_token_ids=prev[None])
    print("confidence", torch.sigmoid(conf[0]).tolist())
