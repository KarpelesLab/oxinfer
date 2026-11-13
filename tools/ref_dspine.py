#!/usr/bin/env python3
# Reference for DSpine drafts (Tencent, `DFlashDraftModel` with a selected-prefix code bridge, e.g.
# tencent/Qwen3.8-27B-DSpine-b16) in float32 numpy, written from the formulas of the reference
# implementation (github.com/Tencent/DSpine: sglang_dspine/models/dspine.py `_forward_eager`,
# dspine_mlx/dspine_mlx/{model,bridge,propose}.py), without its bf16 rounding points: the context
# projection, the causal block with a bridge write after every layer, and the serial last-write
# refinement of each slot given the token drafted before it. Checks oxinfer's refined scores.
#
# usage: ref_dspine.py DRAFT_DIR DUMP_DIR
#   DUMP_DIR: written by `cargo run --release --example dspark_check -- MODEL DRAFT PROMPT DUMP_DIR`:
#     meta.json {"n": context positions, "anchor": token, "k": drafts, "drafts": [...]},
#     feats.f32 [n, n_feat] (target features of positions 0..n),
#     logits.f32 [k, vocab] (refined scores of each slot's candidates, -inf elsewhere),
#     embed_rows.f32 [2, H] (target embeddings of the anchor and of the mask token),
#     head_rows.f32 [k, top_k, H] (target LM head rows of each slot's candidates, by token id).
# Prints per slot: the largest |difference| of the candidates' scores, of their log-probabilities
# among the candidates, and whether both pick the same token.
import json
import os
import struct
import sys

import numpy as np

draft_dir, dump = sys.argv[1:3]
cfg = json.load(open(os.path.join(draft_dir, "config.json")))
dfl = cfg["dflash_config"]
H, L = cfg["hidden_size"], cfg["num_hidden_layers"]
NH, NKV, HD = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
EPS = cfg.get("rms_norm_eps", 1e-6)
THETA = (cfg.get("rope_parameters") or {}).get("rope_theta") or cfg["rope_theta"]
T = cfg["block_size"]
br = dfl["selected_prefix_code_bridge"]
C, R, BEPS = br["code_size"], br["rank"], br["eps"]
second = cfg.get("code_bridge_second_pass") or dfl.get("selected_prefix_last_write_refine")
K = second["head_topk"]
window = cfg.get("sliding_window") if cfg.get("use_sliding_window") else None

path = os.path.join(draft_dir, "model.safetensors")
with open(path, "rb") as f:
    (n,) = struct.unpack("<Q", f.read(8))
    head = json.loads(f.read(n))
mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)


def tensor(name, rows=None):
    """Tensor `name` as float32 (its leading index `rows` only, when given)."""
    t = head[name]
    assert t["dtype"] == "BF16", (name, t["dtype"])
    a, b = t["data_offsets"]
    u = np.frombuffer(mm[a:b], dtype=np.uint16).reshape(t["shape"])
    if rows is not None:
        u = u[rows]
    return (u.astype(np.uint32) << 16).view(np.float32)


def rms_norm(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * w


def rope(x, pos):
    """Rotate-half RoPE on [t, heads, HD] at positions `pos`."""
    half = HD // 2
    inv = THETA ** (-np.arange(half, dtype=np.float64) / half)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang).astype(np.float32)[:, None, :], np.sin(ang).astype(np.float32)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


meta = json.load(open(os.path.join(dump, "meta.json")))
n_ctx, anchor, k, drafts = meta["n"], meta["anchor"], meta["k"], meta["drafts"]
feats = np.fromfile(os.path.join(dump, "feats.f32"), dtype=np.float32).reshape(n_ctx, -1)
emb = np.fromfile(os.path.join(dump, "embed_rows.f32"), dtype=np.float32).reshape(2, H)
ours = np.fromfile(os.path.join(dump, "logits.f32"), dtype=np.float32).reshape(k, -1)
head_rows = np.fromfile(os.path.join(dump, "head_rows.f32"), dtype=np.float32).reshape(k, K, H)
assert window is None or n_ctx + T <= window, "context beyond the sliding window is not modelled here"

# Context: the target features projected and normed, shared by every layer.
ctx = rms_norm(feats @ tensor("fc.weight").T, tensor("hidden_norm.weight"))
x = np.concatenate([emb[:1], np.repeat(emb[1:], T - 1, axis=0)], axis=0)  # [T, H]
codebook_rows = {t: tensor("selected_prefix_bridge.codebook", t) for t in set([anchor] + drafts)}
gate_w = tensor("selected_prefix_bridge.gate.weight")
gate_b = tensor("selected_prefix_bridge.gate.bias")
pred_w = tensor("selected_prefix_bridge.predecessor_gate.weight")
prev_w = tensor("selected_prefix_bridge.prev.weight")
sqrt_c = np.float32(np.sqrt(C))
pos_ctx, pos_blk = np.arange(n_ctx), np.arange(n_ctx, n_ctx + T)
causal = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
mask = np.concatenate([np.zeros((T, n_ctx), dtype=np.float32), causal], axis=1)  # [T, n_ctx + T]
stash = None
for li in range(L):
    p = f"layers.{li}."
    qw, kw, vw = (tensor(p + f"self_attn.{n}_proj.weight") for n in "qkv")
    qn, kn = tensor(p + "self_attn.q_norm.weight"), tensor(p + "self_attn.k_norm.weight")
    hn = rms_norm(x, tensor(p + "input_layernorm.weight"))
    q = rope(rms_norm((hn @ qw.T).reshape(T, NH, HD), qn), pos_blk)
    keys = np.concatenate(
        [
            rope(rms_norm((ctx @ kw.T).reshape(n_ctx, NKV, HD), kn), pos_ctx),
            rope(rms_norm((hn @ kw.T).reshape(T, NKV, HD), kn), pos_blk),
        ]
    )
    vals = np.concatenate([(ctx @ vw.T).reshape(n_ctx, NKV, HD), (hn @ vw.T).reshape(T, NKV, HD)])
    g = NH // NKV
    att = np.zeros((T, NH, HD), dtype=np.float32)
    for hh in range(NH):
        s = q[:, hh] @ keys[:, hh // g].T * np.float32(HD**-0.5) + mask
        s = np.exp(s - s.max(axis=-1, keepdims=True))
        att[:, hh] = (s / s.sum(axis=-1, keepdims=True)) @ vals[:, hh // g]
    x = x + att.reshape(T, NH * HD) @ tensor(p + "self_attn.o_proj.weight").T
    hn = rms_norm(x, tensor(p + "post_attention_layernorm.weight"))
    gate = hn @ tensor(p + "mlp.gate_proj.weight").T
    x = x + (gate * sigmoid(gate) * (hn @ tensor(p + "mlp.up_proj.weight").T)) @ tensor(p + "mlp.down_proj.weight").T

    # The bridge write after this layer.
    read = (
        tensor("selected_prefix_bridge.read_weight")
        if li == L - 1
        else tensor("selected_prefix_bridge.early_read_weights", li)
    )
    rms = np.sqrt(np.maximum(np.mean(x * x, axis=-1), np.finfo(np.float32).tiny))
    proj = (x / (rms[:, None] + BEPS)) @ read.T
    codes = sqrt_c * proj / (np.linalg.norm(proj, axis=-1, keepdims=True) + BEPS)
    codes[0] = sqrt_c * codebook_rows[anchor]
    qq, ss, vv = codes @ gate_w.T, codes @ pred_w.T, codes @ prev_w.T
    msg = np.zeros((T, R), dtype=np.float32)
    msg[1:] = sigmoid(qq[1:] + ss[:-1] + gate_b) * vv[:-1]
    write = tensor("selected_prefix_bridge.feature_write_weights", li)
    if li == L - 1:
        stash = (x.copy(), rms, qq)
    delta = msg @ write.T
    delta[0] = 0
    x = x + rms[:, None] * delta

norm_w = tensor("norm.weight")
raw, rms, qq = stash
write = tensor("selected_prefix_bridge.feature_write_weights", L - 1)
ok = True
sender = anchor
for i in range(k):
    row = i + 1
    cand = np.nonzero(np.isfinite(ours[i]))[0]
    assert len(cand) == K, (i, len(cand))
    e = sqrt_c * codebook_rows[sender]
    m = sigmoid(qq[row] + pred_w @ e + gate_b) * (prev_w @ e)
    refined = rms_norm(raw[row] + rms[row] * (write @ m), norm_w)
    ref = head_rows[i] @ refined
    got = ours[i][cand]
    lp = lambda v: v - (v.max() + np.log(np.exp(v - v.max()).sum()))
    pick_ref, pick_got = int(cand[np.argmax(ref)]), int(cand[np.argmax(got)])
    # The parallel pass (the block's own last write) on the same candidates, for comparison.
    par = head_rows[i] @ rms_norm(x[row], norm_w)
    same = pick_ref == pick_got == drafts[i]
    ok &= same
    print(
        f"slot {i}: max |diff| {np.abs(ref - got).max():.4f} (scores up to {np.abs(ref).max():.1f}), "
        f"log-prob diff {np.abs(lp(ref) - lp(got)).max():.4f}, pick {pick_ref} vs {pick_got} "
        f"{'same' if same else 'DIFFERENT'}; parallel pass would pick {int(cand[np.argmax(par)])}"
    )
    sender = drafts[i]
print("all slots agree" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
