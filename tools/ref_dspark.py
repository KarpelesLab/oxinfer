#!/usr/bin/env python3
# Reference DSpark draft (Mistral Large 4 MLA backbone) in float32 torch, written from llama.cpp's
# `graph_mistral_large4` / `build_dspark_markov_head` and ggml's YaRN RoPE, to check oxinfer's
# draft logits for the same target features.
#
# usage: ref_dspark.py DRAFT.gguf TARGET_PART.gguf DUMP_DIR GGUF_PY_DIR
#   TARGET_PART.gguf: the target GGUF file holding token_embd.weight and output.weight.
#   DUMP_DIR: written by `cargo run --release --example dspark_check -- MODEL DRAFT PROMPT DUMP_DIR`:
#     meta.json {"n": context positions, "anchor": token, "k": drafts, "drafts": [...]},
#     feats.f32 [n, n_feat] (target features of positions 0..n), logits.f32 [k, vocab].
# Prints per slot: max |diff|, relative error, argmax agreement, KL.
import json, math, os, sys

import numpy as np
import torch

sys.path.insert(0, sys.argv[4])
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

draft_path, target_path, dump = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(os.cpu_count())

r = GGUFReader(draft_path)
meta = {k: f.contents() for k, f in r.fields.items() if not k.startswith("tokenizer.")}
T = {t.name: t for t in r.tensors}


def W(name):
    t = T[name]
    a = dequantize(t.data, t.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).reshape([int(x) for x in reversed(t.shape)])


g = lambda k: meta[f"dflash.{k}"]  # noqa: E731
H, NL, NH = g("embedding_length"), g("block_count"), g("attention.head_count")
LAT, DR = g("attention.kv_lora_rank"), g("rope.dimension_count")
QK, VD = g("attention.key_length_mla"), g("attention.value_length_mla")
DN = QK - DR
EPS = g("attention.layer_norm_rms_epsilon")
WIN = g("attention.sliding_window")
CAP = g("final_logit_softcapping")
BLOCK = g("block_size")
assert g("sample_from_anchor")

# ggml YaRN (rope_yarn + ggml_rope_yarn_corr_dims), llama.cpp's attn_factor for a draft with
# rope.scaling.yarn_log_multiplier = 1: the magnitude cancels to mscale(f, 1) / mscale(f, 1) = 1.
base = g("rope.freq_base")
factor = g("rope.scaling.factor")
n_orig = g("rope.scaling.original_context_length")
beta_fast, beta_slow = g("rope.scaling.yarn_beta_fast"), g("rope.scaling.yarn_beta_slow")
log_mul = meta.get("dflash.rope.scaling.yarn_log_multiplier", 0.0)
freq_scale = 1.0 / factor


def corr_dim(n_rot):
    return DR * math.log(n_orig / (n_rot * 2 * math.pi)) / (2 * math.log(base))


lo = max(0.0, math.floor(corr_dim(beta_fast)))
hi = min(DR - 1.0, math.ceil(corr_dim(beta_slow)))
mscale = lambda s, m: 1.0 if s <= 1 else 0.1 * m * math.log(s) + 1.0  # noqa: E731
attn_factor = (mscale(factor, 1.0) / mscale(factor, log_mul)) if log_mul else mscale(factor, 1.0)
attn_factor *= 1.0 / (1.0 + 0.1 * math.log(factor))
mag = attn_factor * (1.0 + 0.1 * math.log(1.0 / freq_scale))


def rope(x, pos):
    # x [..., DR] at position pos: adjacent pairs (ggml GGML_ROPE_TYPE_NORM)
    i = torch.arange(DR // 2, dtype=torch.float64)
    theta_extrap = pos * base ** (-2.0 * i / DR)
    theta_interp = freq_scale * theta_extrap
    y = (i - lo) / max(0.001, hi - lo)
    ramp = 1.0 - torch.clamp(y, 0.0, 1.0)
    theta = theta_interp * (1 - ramp) + theta_extrap * ramp
    c, s = (torch.cos(theta) * mag).float(), (torch.sin(theta) * mag).float()
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


def rms(x, w):
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS) * w


m = json.load(open(os.path.join(dump, "meta.json")))
n, anchor, k = m["n"], m["anchor"], m["k"]
feats = torch.from_numpy(np.fromfile(os.path.join(dump, "feats.f32"), dtype=np.float32)).reshape(n, -1)
ox = np.fromfile(os.path.join(dump, "logits.f32"), dtype=np.float32).reshape(k, -1)
V = ox.shape[1]

# Target embeddings / head (dequantized row blocks).
tr = GGUFReader(target_path)
TT = {t.name: t for t in tr.tensors}
emb_t = TT["token_embd.weight"]
emb_row = torch.from_numpy(dequantize(emb_t.data[anchor : anchor + 1], emb_t.tensor_type).astype(np.float32)).reshape(H)
out_t = TT["output.weight"]

layers = []
for l in range(NL):
    p = f"blk.{l}."
    layers.append(
        dict(
            attn_norm=W(p + "attn_norm.weight"),
            q=W(p + "attn_q.weight"),
            kv_a=W(p + "attn_kv_a_mqa.weight"),
            kv_norm=W(p + "attn_kv_a_norm.weight"),
            k_b=W(p + "attn_k_b.weight"),  # [NH, LAT, DN]
            v_b=W(p + "attn_v_b.weight"),  # [NH, VD, LAT]
            o=W(p + "attn_output.weight"),
            ffn_norm=W(p + "ffn_norm.weight"),
            gate=W(p + "ffn_gate.weight"),
            up=W(p + "ffn_up.weight"),
            down=W(p + "ffn_down.weight"),
        )
    )
fc, enc_norm, out_norm = W("fc.weight"), W("enc.output_norm.weight"), W("output_norm.weight")
mask = W("mask_embd.weight")
w1, w2 = W("markov_w1.weight"), W("markov_w2.weight")
jp, jb = W("joint_proj.weight"), W("joint_proj.bias")
R = w1.shape[1]

with torch.no_grad():
    # Context cache from the target features.
    xg = rms(feats @ fc.T, enc_norm)
    cache = []
    for L in layers:
        kv = xg @ L["kv_a"].T
        c = rms(kv[:, :LAT], L["kv_norm"])
        kr = torch.stack([rope(kv[j, LAT:], j) for j in range(n)])
        cache.append((c, kr))
    # Block [anchor, mask_0, ..] at positions n.., non-causal, window of the anchor.
    x = torch.cat([emb_row[None], mask[: k - 1]], 0)
    lo_ctx = max(0, n - WIN)
    for L, (c, kr) in zip(layers, cache):
        hn = rms(x, L["attn_norm"])
        q = (hn @ L["q"].T).reshape(k, NH, QK)
        kv = hn @ L["kv_a"].T
        bc = rms(kv[:, :LAT], L["kv_norm"])
        bkr = torch.stack([rope(kv[i, LAT:], n + i) for i in range(k)])
        qn = q[..., :DN]
        qp = torch.stack([rope(q[i, :, DN:], n + i) for i in range(k)])
        ql = torch.einsum("hld,khd->khl", L["k_b"], qn)
        keys_c = torch.cat([c[lo_ctx:], bc], 0)
        keys_r = torch.cat([kr[lo_ctx:], bkr], 0)
        s = (torch.einsum("khl,jl->khj", ql, keys_c) + torch.einsum("khr,jr->khj", qp, keys_r)) / math.sqrt(QK)
        pr = torch.softmax(s, -1)
        ol = torch.einsum("khj,jl->khl", pr, keys_c)
        vo = torch.einsum("hvl,khl->khv", L["v_b"], ol).reshape(k, NH * VD)
        x = x + vo @ L["o"].T
        hn = rms(x, L["ffn_norm"])
        a = torch.nn.functional.silu(hn @ L["gate"].T) * (hn @ L["up"].T)
        x = x + a @ L["down"].T
    hid = rms(x, out_norm)
    # Target LM head over row blocks of output.weight.
    logits = torch.zeros(k, V)
    rows = 16384
    for r0 in range(0, V, rows):
        blk = out_t.data[r0 : r0 + rows]
        w = torch.from_numpy(dequantize(blk, out_t.tensor_type).astype(np.float32)).reshape(-1, H)
        logits[:, r0 : r0 + w.shape[0]] = hid @ w.T
    logits = torch.tanh(logits / CAP) * CAP
    # Markov rnn head, chained on the argmax.
    state = torch.zeros(R)
    prev = anchor
    ref = []
    for i in range(k):
        j = jp @ torch.cat([state, w1[prev], hid[i]]) + jb
        gate, cand = torch.sigmoid(j[:R]), torch.tanh(j[R : 2 * R])
        state = cand + gate * (state - cand)
        mk = torch.tanh(j[2 * R :])
        li = logits[i] + w2 @ mk
        ref.append(li)
        prev = int(torch.argmax(li))
    ref = torch.stack(ref).numpy()

for i in range(k):
    a, b = ox[i].astype(np.float64), ref[i].astype(np.float64)
    pa, pb = np.exp(a - a.max()), np.exp(b - b.max())
    pa, pb = pa / pa.sum(), pb / pb.sum()
    kl = float((pb * (np.log(pb + 1e-30) - np.log(pa + 1e-30))).sum())
    top_r = np.argsort(-b)[:5]
    print(
        f"slot {i}: max|diff| {np.abs(a - b).max():.4g} rel {np.linalg.norm(a - b) / np.linalg.norm(b):.3g} "
        f"argmax ox {int(a.argmax())} ref {int(b.argmax())} KL {kl:.3g} ref top5 {top_r.tolist()} "
        f"p {[round(float(pb[t]), 3) for t in top_r]}"
    )
print("oxinfer drafts", m["drafts"])
