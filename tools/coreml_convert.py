#!/usr/bin/env python3
"""Convert a Hugging Face checkpoint into an ANEMLL-format Core ML package for the Apple Neural Engine.

usage: coreml_convert.py --model DIR --output DIR [options]

The output directory follows ANEMLL's layout (https://github.com/Anemll/Anemll, 0.3.5) so that
ANEMLL-compatible runtimes (ANEMLL's tests/chat.py and Swift CLI, oxinfer's Core ML runtime) can
load it:

  meta.yaml                                    model description and runtime parameters
  <prefix>_embeddings[_lutN].mlmodelc          input_ids [1,1] or [1,batch] -> hidden_states
  <prefix>_FFN_PF[_lutN]_chunk_XXofYY.mlmodelc decoder layers; functions `infer` (one token) and
                                               `prefill` (batch tokens) sharing weights and the
                                               Core ML state (KV cache, recurrent state)
  <prefix>_lm_head[_lutN].mlmodelc             final hidden state -> logits1..logits16 (vocab split)
  tokenizer.json, tokenizer_config.json, ...   copied from the checkpoint

Supported architectures (docs/coreml-format.md has the exact runtime contract):
  qwen3    Qwen3 dense: the same graph and I/O as ANEMLL's Qwen3 converter
  qwen3_5  Qwen3.5 / 3.6 / 3.8 dense text models: the Gated DeltaNet layers keep their recurrent
           and convolution state as Core ML states next to the KV cache of the gated full
           attention layers; prefill runs the chunked delta rule on the Neural Engine
  gemma4   Gemma 4 E2B / E4B text: per-layer embeddings (an extra embeddings output fed to every
           chunk), KV-shared layers, sliding-window and global caches, logit soft cap

The graph is written for the ANE the way ANEMLL does it: fp16 everywhere, linear layers as 1x1
convolutions, RMSNorm through layer_norm on [x, -x], 4-D tensors, KV cache updated in place with a
dynamic slice at `current_pos`, LM head split 16 ways. Weights are palettized (LUT per group of 8
output channels, as in ANEMLL's released models: 6 bits for the decoder layers and the LM head,
embeddings unquantized). By default the LUTs are data-aware (--lut-method fisher, see
tools/coreml_quant.py): the float32 transformers model answers a few built-in prompts, the
Fisher information of every weight weights its k-means, and --lut-mix gives the most sensitive
weights a wider LUT. `infer` and `prefill` see byte-identical weights, so the merged package
stores them once. Decoder chunks are cut at about equal weight size, each below --max-chunk-mb.
Gemma 4's sliding-window layers keep a cache of window + batch positions once the context is
longer than that (docs/coreml-format.md), so 4K-8K contexts cost little memory.

Requires macOS with the Xcode command line tools (`xcrun coremlcompiler`), coremltools >= 8
(tested: 9.0 with torch 2.14 and numpy 2.2; coremltools 9.0 fails with numpy >= 2.3),
safetensors, transformers (calibration; --lut-method kmeans does without), scikit-learn
(Core ML's k-means) and pyyaml for coreml_check.py. Calibration loads the float32 model plus one
gradient per linear weight: about 3x the fp16 checkpoint size in RAM.

quantization (--ffn-quant / --lmhead-quant / --embed-quant):
  none          fp16
  lutN[,G]      palettization, N bits (1, 2, 3, 4, 6 or 8: Core ML has no 5-bit LUT), G output
                channels per LUT (default 8; smaller groups are slower on the ANE: G=1 halves
                decode speed)
  int8[,tensor] linear int8 per channel (or per tensor)  -- wrong results on the ANE, GPU only
  int4[,B]      linear int4 per block of B (default 32)   -- does not compile for the ANE

examples:
  coreml_convert.py --model ~/.cache/oxinfer/hub/Qwen--Qwen3-0.6B --output out/qwen3-0.6b
  coreml_convert.py --model ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B --output out/q35 \\
      --ffn-quant lut4 --lut-mix 4.5 --calib-cache out/q35-calib.pt   # ~30% smaller decoder
  coreml_convert.py --model models/gemma-4-E2B-it --output out/g4 --context-length 8192
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float16
FORMAT_VERSION = "0.3.5"  # ANEMLL meta.yaml version this layout follows
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
    "tokenizer.model",
]


# ---------------------------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------------------------


@dataclass
class Spec:
    """Architecture hyper-parameters, normalized across config.json variants."""

    arch: str
    model_type: str
    hidden: int
    inter: int
    n_layers: int
    n_heads: int
    n_kv: int
    head_dim: int
    eps: float
    theta: float
    rot_dim: int
    vocab: int
    tied: bool
    zero_centered: bool = False  # norm weights stored as w, applied as 1 + w
    attn_gate: bool = False  # q_proj also produces a per-head sigmoid output gate
    layer_types: list = field(default_factory=list)  # "full" | "linear"
    lin_nk: int = 0
    lin_nv: int = 0
    lin_dk: int = 0
    lin_dv: int = 0
    conv_k: int = 0
    n_experts: int = 0  # MoE (Qwen3.5 MoE): routed experts, top-k, expert width
    top_k: int = 0
    moe_inter: int = 0
    # Gemma 4
    global_head_dim: int = 0
    sliding_window: int = 0
    first_shared: int = 0  # first layer reusing the KV of an earlier layer (n_layers: none)
    ple_dim: int = 0  # per-layer embedding width (0: none)
    softcap: float = 0.0  # final logit soft cap
    theta_local: float = 0.0
    global_rot_frac: float = 1.0

    @property
    def full_layers(self):
        return [i for i, t in enumerate(self.layer_types) if t == "full"]

    @property
    def linear_layers(self):
        return [i for i, t in enumerate(self.layer_types) if t == "linear"]

    @property
    def conv_dim(self):
        return 2 * self.lin_nk * self.lin_dk + self.lin_nv * self.lin_dv


def load_spec(cfg: dict) -> Spec:
    top_type = cfg.get("model_type", "")
    tc = cfg.get("text_config") or cfg
    mt = tc.get("model_type", top_type)
    tied = bool(cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", False)))
    n_heads = tc["num_attention_heads"]
    head_dim = tc.get("head_dim") or tc["hidden_size"] // n_heads
    common = dict(
        model_type=mt,
        hidden=tc["hidden_size"],
        inter=tc.get("intermediate_size") or tc.get("moe_intermediate_size", 0),
        n_layers=tc["num_hidden_layers"],
        n_heads=n_heads,
        n_kv=tc.get("num_key_value_heads", n_heads),
        head_dim=head_dim,
        eps=float(tc.get("rms_norm_eps", 1e-6)),
        vocab=tc["vocab_size"],
        tied=tied,
    )
    rp = tc.get("rope_parameters") or {}
    theta = float(rp.get("rope_theta", tc.get("rope_theta", 10000.0)))
    if mt in ("qwen3",):
        if tc.get("rope_scaling"):
            raise SystemExit("rope_scaling is not supported")
        return Spec(arch="qwen3", theta=theta, rot_dim=head_dim, layer_types=["full"] * common["n_layers"], **common)
    if mt in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"):
        n = common["n_layers"]
        if tc.get("num_experts"):
            if tc.get("mlp_only_layers") or tc.get("decoder_sparse_step", 1) != 1:
                raise SystemExit("Qwen3.5 MoE with dense layers (mlp_only_layers) is not supported")
            common.update(
                inter=tc.get("moe_intermediate_size", 0),
                n_experts=tc["num_experts"],
                top_k=tc["num_experts_per_tok"],
                moe_inter=tc["moe_intermediate_size"],
            )
        lt = tc.get("layer_types")
        if lt is None:
            k = tc.get("full_attention_interval", 4)
            lt = ["full_attention" if (i + 1) % k == 0 else "linear_attention" for i in range(n)]
        frac = float(rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0)))
        return Spec(
            arch="qwen3_5",
            theta=theta,
            rot_dim=int(head_dim * frac),
            zero_centered=True,
            attn_gate=bool(tc.get("attn_output_gate", True)),
            layer_types=["full" if t == "full_attention" else "linear" for t in lt],
            lin_nk=tc["linear_num_key_heads"],
            lin_nv=tc["linear_num_value_heads"],
            lin_dk=tc["linear_key_head_dim"],
            lin_dv=tc["linear_value_head_dim"],
            conv_k=tc["linear_conv_kernel_dim"],
            **common,
        )
    if mt in ("gemma4", "gemma4_text"):
        if tc.get("enable_moe_block") or tc.get("num_experts"):
            raise SystemExit("Gemma 4 MoE checkpoints are not supported yet")
        if tc.get("attention_k_eq_v"):
            raise SystemExit("Gemma 4 attention_k_eq_v (12B/31B) is not supported yet")
        rpl = tc["rope_parameters"]
        n = common["n_layers"]
        return Spec(
            arch="gemma4",
            theta=float(rpl["full_attention"]["rope_theta"]),
            theta_local=float(rpl["sliding_attention"]["rope_theta"]),
            global_rot_frac=float(rpl["full_attention"].get("partial_rotary_factor", 1.0)),
            rot_dim=head_dim,
            layer_types=["full" if t == "full_attention" else "sliding" for t in tc["layer_types"]],
            global_head_dim=tc.get("global_head_dim") or head_dim,
            sliding_window=tc["sliding_window"],
            first_shared=n - (tc.get("num_kv_shared_layers") or 0),
            ple_dim=tc.get("hidden_size_per_layer_input") or 0,
            softcap=float(tc.get("final_logit_softcapping") or 0.0),
            **common,
        )
    raise SystemExit(f"unsupported model_type {mt!r} (supported: qwen3, qwen3_5, gemma4)")


class Checkpoint:
    """Lazy access to safetensors tensors by their name without the model prefix."""

    def __init__(self, path: str):
        from safetensors import safe_open

        self.files = {}
        names = {}
        for f in sorted(os.listdir(path)):
            if f.endswith(".safetensors"):
                h = safe_open(os.path.join(path, f), framework="pt")
                self.files[f] = h
                for k in h.keys():
                    names[k] = f
        if not names:
            raise SystemExit(f"no .safetensors files in {path}")
        for prefix in ("model.language_model.", "model.", ""):
            if prefix + "embed_tokens.weight" in names:
                self.prefix = prefix
                break
        else:
            raise SystemExit("cannot find embed_tokens.weight in the checkpoint")
        self.names = names

    def has(self, name: str, raw=False) -> bool:
        return (name if raw else self.prefix + name) in self.names

    def get(self, name: str, raw=False) -> torch.Tensor:
        full = name if raw else self.prefix + name
        return self.files[self.names[full]].get_tensor(full).float()


# ---------------------------------------------------------------------------------------------
# ANE building blocks
# ---------------------------------------------------------------------------------------------


def conv1x1(w: torch.Tensor, src: str | None = None, rows=None, gain: float = 1.0) -> nn.Conv2d:
    """A linear layer `y = W x` as a 1x1 convolution over [1, in, 1, T]. `src` names the
    checkpoint linear it comes from, `rows` the source rows it holds (default all) and `gain`
    the factor applied to them, for the calibration statistics of --lut-method."""
    m = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=False, dtype=DTYPE)
    m.src, m.src_rows, m.src_gain = src, rows, gain
    m.weight.data = w.to(DTYPE).view(w.shape[0], w.shape[1], 1, 1).contiguous()
    m.weight.requires_grad_(False)
    return m


def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS normalization over the last axis via layer_norm on [x, -x] (zero mean by
    construction), which the ANE evaluates more accurately than a hand-written rsqrt."""
    d = x.shape[-1]
    return F.layer_norm(torch.cat([x, -x], dim=-1), (2 * d,), eps=eps)[..., :d]


def sigmoid(x):
    """sigmoid through tanh: the ANE's own sigmoid / SiLU is a coarse approximation (errors up
    to ~0.015 on the Gated DeltaNet conv outputs), its tanh is accurate."""
    return 0.5 + 0.5 * torch.tanh(x * 0.5)


def silu(x):
    return x * sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, w: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(w.to(DTYPE), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        return rms(x, self.eps) * self.weight


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class Attention(nn.Module):
    """Full causal attention over the KV cache; optionally Qwen3.5's gated variant (sigmoid
    output gate from q_proj, partial RoPE)."""

    def __init__(self, sp: Spec, ck: Checkpoint, layer: int, slot: int):
        super().__init__()
        p = f"layers.{layer}.self_attn."
        nh, hd = sp.n_heads, sp.head_dim
        self.sp, self.slot = sp, slot
        wq = ck.get(p + "q_proj.weight")
        if sp.attn_gate:
            # q_proj rows are [query_h, gate_h] per head: split into two projections.
            wq = wq.view(nh, 2, hd, sp.hidden)
            r = torch.arange(nh * 2 * hd).view(nh, 2, hd)
            # holds W / 2 (sigmoid(y) = 0.5 + 0.5 tanh(y / 2)): Core ML would fold the 1/2 into the
            # weights itself, which changes them after palettization (--lut-method)
            self.g_proj = conv1x1(wq[:, 1].reshape(nh * hd, sp.hidden) * 0.5, p + "q_proj", r[:, 1].reshape(-1), 0.5)
            wq, rq = wq[:, 0].reshape(nh * hd, sp.hidden), r[:, 0].reshape(-1)
        else:
            rq = None
        self.q_proj = conv1x1(wq, p + "q_proj", rq)
        self.k_proj = conv1x1(ck.get(p + "k_proj.weight"), p + "k_proj")
        self.v_proj = conv1x1(ck.get(p + "v_proj.weight"), p + "v_proj")
        self.o_proj = conv1x1(ck.get(p + "o_proj.weight"), p + "o_proj")
        off = 1.0 if sp.zero_centered else 0.0
        self.q_norm = RMSNorm(ck.get(p + "q_norm.weight") + off, sp.eps)
        self.k_norm = RMSNorm(ck.get(p + "k_norm.weight") + off, sp.eps)

    def rope(self, x, cos, sin):
        r = self.sp.rot_dim
        if r == x.shape[-1]:
            return x * cos + rotate_half(x) * sin
        xr, xp = x[..., :r], x[..., r:]
        return torch.cat([xr * cos + rotate_half(xr) * sin, xp], dim=-1)

    def forward(self, x, cos, sin, causal_mask, current_pos, kv_cache, n_full):
        """x: normed hidden [1, T, H]; cos/sin [1, 1, T, rot]; causal_mask [1, 1, T, ctx]."""
        sp = self.sp
        nh, nkv, hd = sp.n_heads, sp.n_kv, sp.head_dim
        t = x.shape[1]
        hs = x.permute(0, 2, 1).unsqueeze(2)  # [1, H, 1, T]
        if t == 1:
            q = self.q_proj(hs).view(1, nh, 1, hd)
            k = self.k_proj(hs).view(1, nkv, 1, hd)
            v = self.v_proj(hs).view(1, nkv, 1, hd)
        else:
            q = self.q_proj(hs).view(1, nh, hd, t).permute(0, 1, 3, 2)
            k = self.k_proj(hs).view(1, nkv, hd, t).permute(0, 1, 3, 2)
            v = self.v_proj(hs).view(1, nkv, hd, t).permute(0, 1, 3, 2)
        q = self.rope(self.q_norm(q), cos, sin)
        k = self.rope(self.k_norm(k), cos, sin)

        s = self.slot
        kv_cache[s : s + 1, :, current_pos : current_pos + t, :] = k
        kv_cache[n_full + s : n_full + s + 1, :, current_pos : current_pos + t, :] = v
        kc = kv_cache[s : s + 1].squeeze(0)  # [nkv, ctx, hd]
        vc = kv_cache[n_full + s : n_full + s + 1].squeeze(0)
        rep = nh // nkv
        if rep > 1:
            kc = kc.unsqueeze(1).repeat(1, rep, 1, 1)
            vc = vc.unsqueeze(1).repeat(1, rep, 1, 1)
        kc = kc.view(1, nh, -1, hd)
        vc = vc.view(1, nh, -1, hd)

        att = torch.matmul(q, kc.transpose(-1, -2)) * (1.0 / math.sqrt(hd)) + causal_mask
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, vc)  # [1, nh, T, hd]
        o = o.permute(0, 1, 3, 2).reshape(1, nh * hd, 1, t)  # [1, nh*hd, 1, T]
        if sp.attn_gate:
            o = o * (0.5 + 0.5 * torch.tanh(self.g_proj(hs)))
        return self.o_proj(o).squeeze(2).permute(0, 2, 1)  # [1, T, H]


class GatedDeltaNet(nn.Module):
    """Qwen3.5 linear attention: causal depthwise conv + SiLU over [q, k, v], L2-normed q/k and
    the gated delta rule S <- a S + beta k (v - a S^T k)^T, o = S^T q, then a per-head RMSNorm
    gated by SiLU(z).

    States (shared by all layers of the model, indexed by this layer's slot):
      gdn_state  [n_lin * n_v, d_k, d_v]  sqrt(d_k) * S for every value head
      conv_state [n_lin, K - 1, conv_dim] the last K-1 conv inputs (pre-activation), oldest first
    One token runs the recurrence directly; a prefill batch uses the chunked (WY) form with a
    block-doubling triangular inverse, so everything is a matmul.

    fp16 scaling: with unit q, k and the 1/sqrt(d_k) attention scale folded into q, the products
    in q^T S and k^T S fall into fp16 subnormals (~1e-5), which the ANE flushes (the layer output
    was off by cosine 0.97). The state is therefore stored as S' = sqrt(d_k) S and the outer
    product uses k' = sqrt(d_k) k (= rms(k)); the output o' = d_k o only feeds the per-head
    RMSNorm, so it is used as is with eps * d_k^2.
    """

    def __init__(self, sp: Spec, ck: Checkpoint, layer: int, slot: int, chunk: int):
        super().__init__()
        p = f"layers.{layer}.linear_attn."
        self.sp, self.slot, self.chunk = sp, slot, chunk
        self.in_qkv = conv1x1(ck.get(p + "in_proj_qkv.weight"), p + "in_proj_qkv")
        self.in_z = conv1x1(ck.get(p + "in_proj_z.weight"), p + "in_proj_z")
        self.in_b = conv1x1(ck.get(p + "in_proj_b.weight"), p + "in_proj_b")
        self.in_a = conv1x1(ck.get(p + "in_proj_a.weight"), p + "in_proj_a")
        self.out_proj = conv1x1(ck.get(p + "out_proj.weight"), p + "out_proj")
        cw = ck.get(p + "conv1d.weight")  # [conv_dim, 1, K]
        self.conv = nn.Conv2d(sp.conv_dim, sp.conv_dim, (1, sp.conv_k), groups=sp.conv_dim, bias=False, dtype=DTYPE)
        self.conv.weight.data = cw.to(DTYPE).view(sp.conv_dim, 1, 1, sp.conv_k)
        self.conv.weight.requires_grad_(False)
        nv = sp.lin_nv
        self.register_buffer("a_neg", (-ck.get(p + "A_log").exp()).to(DTYPE).view(nv, 1))
        self.register_buffer("dt_bias", ck.get(p + "dt_bias").to(DTYPE).view(nv, 1))
        self.norm = RMSNorm(ck.get(p + "norm.weight"), sp.eps * sp.lin_dk**2)  # see o' above
        c = chunk
        i = torch.arange(c)
        lower = (i[:, None] >= i[None, :]).to(DTYPE)  # j <= i
        self.register_buffer("upper_neg", (1.0 - lower) * -30000.0)
        self.levels = int(math.log2(c))
        for lvl in range(self.levels):
            b = 1 << lvl  # lower-left b x b block of each 2b x 2b diagonal block
            m = (i[:, None] // (2 * b) == i[None, :] // (2 * b)) & (i[:, None] // b > i[None, :] // b)
            self.register_buffer(f"blk{lvl}", m.to(DTYPE))
        self.register_buffer("eye", torch.eye(c, dtype=DTYPE))
        # cum[t, i] = 1 if t <= i: (g @ cum)[i] = sum_{t<=i} g_t
        self.register_buffer("cum", lower.t().contiguous())
        # seg[t, i*c + j] = 1 if j < t <= i: (g @ seg)[i, j] = sum_{j<t<=i} g_t (exact log decay
        # from token j to token i, summed term by term instead of differencing a long cumsum)
        tt, ii, jj = torch.meshgrid(i, i, i, indexing="ij")
        self.register_buffer("seg", ((jj < tt) & (tt <= ii)).to(DTYPE).reshape(c, c * c))
        # rest[t, i] = 1 if t > i: (g @ rest)[i] = sum_{t>i} g_t (decay from i to chunk end)
        self.register_buffer("rest", (i[:, None] > i[None, :]).to(DTYPE))

    def forward(self, x, gdn_state, conv_state):
        sp = self.sp
        nk, nv, dk, dv, kd = sp.lin_nk, sp.lin_nv, sp.lin_dk, sp.lin_dv, sp.lin_nk * sp.lin_dk
        t = x.shape[1]
        hs = x.permute(0, 2, 1).unsqueeze(2)  # [1, H, 1, T]
        qkv = self.in_qkv(hs)  # [1, conv_dim, 1, T]
        s = self.slot
        cs = conv_state[s : s + 1].permute(0, 2, 1).unsqueeze(2)  # [1, cd, 1, K-1]
        win = torch.cat([cs, qkv], dim=3)  # [1, cd, 1, K-1+T]
        conv_state[s : s + 1] = win[:, :, 0, t:].permute(0, 2, 1)
        c = silu(self.conv(win))  # [1, cd, 1, T]

        q = c[:, :kd].view(nk, dk, t).permute(0, 2, 1)  # [nk, T, dk]
        k = c[:, kd : 2 * kd].view(nk, dk, t).permute(0, 2, 1)
        v = c[:, 2 * kd :].view(nv, dv, t).permute(0, 2, 1)  # [nv, T, dv]
        # l2norm(x) = rms(x) / sqrt(d): q, k unit vectors, ks = sqrt(d_k) k (see class doc).
        rd = 1.0 / math.sqrt(dk)
        # transformers: x / sqrt(sum x^2 + 1e-6), i.e. eps 1e-6 / d on the mean
        q = rms(q, 1e-6 / dk) * rd
        ks = rms(k, 1e-6 / dk)
        if nv != nk:
            rep = nv // nk
            q = q.unsqueeze(1).repeat(1, rep, 1, 1).view(nv, t, dk)
            ks = ks.unsqueeze(1).repeat(1, rep, 1, 1).view(nv, t, dk)
        k = ks * rd
        beta = sigmoid(self.in_b(hs).view(nv, t))
        g = self.a_neg * F.softplus(self.in_a(hs).view(nv, t) + self.dt_bias)  # log decay <= 0
        z = self.in_z(hs).view(nv, dv, t).permute(0, 2, 1)  # [nv, T, dv]

        st = gdn_state[s * nv : (s + 1) * nv]  # [nv, dk, dv] = sqrt(dk) S
        outs = []
        for c0 in range(0, t, self.chunk if t > 1 else 1):
            n = 1 if (t == 1 or self.chunk == 1) else self.chunk
            sl = slice(c0, c0 + n)
            step = self.token_step if n == 1 else self.chunk_step
            o_c, st = step(q[:, sl], k[:, sl], ks[:, sl], v[:, sl], g[:, sl], beta[:, sl], st)
            outs.append(o_c)
        o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        gdn_state[s * nv : (s + 1) * nv] = st

        o = self.norm(o) * silu(z)  # [nv, T, dv]
        o = o.permute(0, 2, 1).reshape(1, nv * dv, 1, t)
        return self.out_proj(o).squeeze(2).permute(0, 2, 1)  # [1, T, H]

    def token_step(self, q, k, ks, v, g, beta, st):
        """The recurrence for one token (q, k, ks [nv, 1, dk], v [nv, 1, dv], g, beta [nv, 1])."""
        nv = g.shape[0]
        st = st * torch.exp(g).view(nv, 1, 1)
        mem = torch.matmul(k, st) * (1.0 / math.sqrt(k.shape[-1]))  # S^T k, [nv, 1, dv]
        delta = (v - mem) * beta.view(nv, 1, 1)
        st = st + torch.matmul(ks.transpose(1, 2), delta)
        return torch.matmul(q, st), st  # d_k S^T q_scaled

    def chunk_step(self, q, k, ks, v, g, beta, st):
        """Chunked gated delta rule over C tokens (unit q, k and ks = sqrt(d_k) k [nv, C, dk],
        v [nv, C, dv], g, beta [nv, C], scaled state [nv, dk, dv]); returns (o', new state)."""
        nv, c = g.shape
        rd = 1.0 / math.sqrt(k.shape[-1])
        gc = torch.matmul(g, self.cum)  # [nv, C] cumulative log decay up to i
        seg = torch.matmul(g, self.seg).view(nv, c, c)  # log decay j -> i (0 above diagonal)
        dec = torch.exp(seg + self.upper_neg)  # [nv, C, C], zero above the diagonal
        kb = k * beta.unsqueeze(-1)
        n = torch.matmul(kb, k.transpose(1, 2)) * dec  # (I + strictly lower part)^-1 below
        # Unit lower-triangular inverse by block doubling: with X the inverse of the b-sized
        # diagonal blocks, X - X L X (L restricted to the lower-left b x b block of every 2b
        # block) inverts the 2b blocks. log2(C) levels of two matmuls, stable unlike the
        # power series (I - N)(I + N^2)(I + N^4)... which overflows for correlated keys.
        tinv = self.eye
        for lvl in range(self.levels):
            off = n * getattr(self, f"blk{lvl}")
            tinv = tinv - torch.matmul(torch.matmul(tinv, off), tinv)
        u = torch.matmul(tinv, v * beta.unsqueeze(-1))  # [nv, C, dv]
        w = torch.matmul(tinv, kb * torch.exp(gc).unsqueeze(-1))  # [nv, C, dk]
        v_new = u - torch.matmul(w, st) * rd
        att = torch.matmul(q, ks.transpose(1, 2)) * dec  # causal (incl. diagonal)
        o = torch.matmul(q * torch.exp(gc).unsqueeze(-1), st) + torch.matmul(att, v_new)
        g_last = gc[:, c - 1 :]  # [nv, 1]
        k_dec = ks * torch.exp(torch.matmul(g, self.rest)).unsqueeze(-1)  # decay i -> chunk end
        st = st * torch.exp(g_last).unsqueeze(-1) + torch.matmul(k_dec.transpose(1, 2), v_new)
        return o, st


class MLP(nn.Module):
    def __init__(self, ck: Checkpoint, layer: int, sub: str = "mlp."):
        super().__init__()
        p = f"layers.{layer}.{sub}"
        self.gate_proj = conv1x1(ck.get(p + "gate_proj.weight"), p + "gate_proj")
        self.up_proj = conv1x1(ck.get(p + "up_proj.weight"), p + "up_proj")
        self.down_proj = conv1x1(ck.get(p + "down_proj.weight"), p + "down_proj")

    def project(self, hs):
        return self.down_proj(silu(self.gate_proj(hs)) * self.up_proj(hs))

    def forward(self, x):
        return self.project(x.permute(0, 2, 1).unsqueeze(2)).squeeze(2).permute(0, 2, 1)


class MoE(nn.Module):
    """Qwen3.5 sparse MoE block, evaluated densely: every expert runs (the gate, up and down
    projections of all experts are three wide 1x1 convolutions) and the outputs are weighted by
    the router's top-k softmax weights (0 for the unselected experts), plus the shared expert
    scaled by sigmoid(shared_expert_gate). The top-k threshold is found by k-1 rounds of
    "max, then mask it out" (reductions and compares only, which the ANE runs; no topk op).
    Compute grows with the total expert count, so this suits small MoE models (and proves the
    graph); see docs/coreml-format.md for large ones."""

    def __init__(self, sp: Spec, ck: Checkpoint, layer: int):
        super().__init__()
        p = f"layers.{layer}.mlp."
        e, inter = sp.n_experts, sp.moe_inter
        self.e, self.k, self.inter = e, sp.top_k, inter
        if ck.has(p + "experts.gate_up_proj"):  # fused [E, 2I, H] / [E, H, I] layout
            gu = ck.get(p + "experts.gate_up_proj")
            gate, up = gu[:, :inter], gu[:, inter:]
            down = ck.get(p + "experts.down_proj")
        else:
            gate = torch.stack([ck.get(p + f"experts.{j}.gate_proj.weight") for j in range(e)])
            up = torch.stack([ck.get(p + f"experts.{j}.up_proj.weight") for j in range(e)])
            down = torch.stack([ck.get(p + f"experts.{j}.down_proj.weight") for j in range(e)])
        self.router = conv1x1(ck.get(p + "gate.weight"), p + "gate")
        self.router.keep_fp16 = True  # a palettized router would flip expert choices
        self.gate_proj = conv1x1(gate.reshape(e * inter, -1))
        self.up_proj = conv1x1(up.reshape(e * inter, -1))
        self.down_proj = conv1x1(down.permute(1, 0, 2).reshape(down.shape[1], e * inter))  # [H, E*I]
        self.shared = MLP(ck, layer, "mlp.shared_expert.")
        self.shared_gate = conv1x1(ck.get(p + "shared_expert_gate.weight"), p + "shared_expert_gate")

    def forward(self, x):
        t = x.shape[1]
        hs = x.permute(0, 2, 1).unsqueeze(2)  # [1, H, 1, T]
        logits = self.router(hs)  # [1, E, 1, T]
        # top-k on the logits (fp16 probabilities of the tail flush to zero and tie), then the
        # softmax over the selected experts = transformers' renormalized top-k probabilities
        top = torch.amax(logits, dim=1, keepdim=True)
        rest = logits
        for _ in range(self.k - 1):
            mx = torch.amax(rest, dim=1, keepdim=True)
            rest = rest - (rest >= mx).to(DTYPE) * 30000.0
        kth = torch.amax(rest, dim=1, keepdim=True)
        wsel = torch.exp(logits - top) * (logits >= kth).to(DTYPE)
        wsel = wsel / torch.sum(wsel, dim=1, keepdim=True)
        h = silu(self.gate_proj(hs)) * self.up_proj(hs)  # [1, E*I, 1, T]
        h = (h.view(1, self.e, self.inter, t) * wsel.view(1, self.e, 1, t)).view(1, self.e * self.inter, 1, t)
        y = self.down_proj(h) + self.shared.project(hs) * sigmoid(self.shared_gate(hs))
        return y.squeeze(2).permute(0, 2, 1)


class Layer(nn.Module):
    def __init__(self, sp: Spec, ck: Checkpoint, i: int, gdn_chunk: int):
        super().__init__()
        off = 1.0 if sp.zero_centered else 0.0
        self.kind = sp.layer_types[i]
        self.input_layernorm = RMSNorm(ck.get(f"layers.{i}.input_layernorm.weight") + off, sp.eps)
        self.post_attention_layernorm = RMSNorm(ck.get(f"layers.{i}.post_attention_layernorm.weight") + off, sp.eps)
        if self.kind == "full":
            self.self_attn = Attention(sp, ck, i, sp.full_layers.index(i))
        else:
            self.linear_attn = GatedDeltaNet(sp, ck, i, sp.linear_layers.index(i), gdn_chunk)
        self.mlp = MoE(sp, ck, i) if sp.n_experts else MLP(ck, i)


class Decoder(nn.Module):
    """The decoder layers of one chunk plus every Core ML state of the model (all chunks declare
    the full state so a single state object is shared by every chunk)."""

    def __init__(self, sp: Spec, ck: Checkpoint, start: int, end: int, ctx: int, gdn_chunk: int):
        super().__init__()
        self.sp, self.start, self.end, self.ctx = sp, start, end, ctx
        self.layers = nn.ModuleList([Layer(sp, ck, i, gdn_chunk) for i in range(start, end)])
        self.last = end == sp.n_layers
        if self.last:
            off = 1.0 if sp.zero_centered else 0.0
            self.norm = RMSNorm(ck.get("norm.weight") + off, sp.eps)
        inv = 1.0 / (sp.theta ** (torch.arange(0, sp.rot_dim, 2).float() / sp.rot_dim))
        freqs = torch.outer(torch.arange(ctx).float(), inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_t", emb.cos().to(DTYPE))
        self.register_buffer("sin_t", emb.sin().to(DTYPE))
        for name, shape in state_shapes(sp, ctx).items():
            self.register_buffer(name, torch.zeros(shape, dtype=DTYPE))

    def forward(self, x, position_ids, causal_mask, current_pos, prefill: bool):
        t = x.shape[1]
        cos = self.cos_t[position_ids].view(1, 1, t, -1)
        sin = self.sin_t[position_ids].view(1, 1, t, -1)
        n_full = len(self.sp.full_layers)
        for layer in self.layers:
            h = layer.input_layernorm(x)
            if layer.kind == "full":
                h = layer.self_attn(h, cos, sin, causal_mask, current_pos, self.kv_cache_0, n_full)
            else:
                h = layer.linear_attn(h, self.gdn_state_0, self.conv_state_0)
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        # Core ML rejects a declared state the graph never reads: touch the ones this chunk does
        # not use (e.g. the KV cache in a chunk of linear attention layers only).
        kinds = {layer.kind for layer in self.layers}
        unused = [n for n in state_shapes(self.sp, self.ctx) if ("kv" in n and "full" not in kinds) or ("kv" not in n and "linear" not in kinds)]
        for n in unused:
            st = getattr(self, n)
            x = x + st[(0,) * (st.dim() - 1)][0:1].view(1, 1, 1) * 0.0
        if self.last:
            # Final norm of the last token only: the prefill output gives the logits of the next
            # token when a prompt ends on a batch boundary.
            x = self.norm(x[:, t - 1 :, :])
        return x


def local_cache_len(sp: Spec, ctx: int, batch: int) -> int:
    """Positions kept by the sliding-window layers' cache: the whole context, or when the
    context is longer than window + batch a shifting window of window + batch positions (the
    first query of a prefill batch still sees `window` keys after the batch is appended)."""
    n = sp.sliding_window + batch
    return n if sp.arch == "gemma4" and SHIFT_LOCAL_CACHE and ctx > n else ctx


SHIFT_LOCAL_CACHE = True  # --full-local-cache turns it off


def state_shapes(sp: Spec, ctx: int, batch: int = 0) -> dict:
    """Core ML states of the model (buffer name -> shape), all fp16."""
    out = {}
    if sp.arch == "gemma4":
        loc, glob = gemma_slots(sp)
        # [slot, K heads then V heads, position, head_dim]: one slice update per layer (two
        # separate K and V updates with a single KV head fail to compile for the ANE)
        out["kv_cache_local"] = (len(loc), 2 * sp.n_kv, local_cache_len(sp, ctx, batch), sp.head_dim)
        out["kv_cache_global"] = (len(glob), 2 * sp.n_kv, ctx, sp.global_head_dim)
        return out
    nf = len(sp.full_layers)
    if nf:
        out["kv_cache_0"] = (2 * nf, sp.n_kv, ctx, sp.head_dim)
    nl = len(sp.linear_layers)
    if nl:
        out["gdn_state_0"] = (nl * sp.lin_nv, sp.lin_dk, sp.lin_dv)
        # time-major: a last axis of K-1 = 3 elements makes the ANE fail at run time
        out["conv_state_0"] = (nl, sp.conv_k - 1, sp.conv_dim)
    return out


class Root(nn.Module):
    """Holds the decoder at `model.model` so the Core ML state names match ANEMLL's
    (`model_model_kv_cache_0` once Core ML sanitizes `model.model.kv_cache_0`)."""

    def __init__(self, dec: Decoder):
        super().__init__()
        self.model = dec


class ChunkFn(nn.Module):
    def __init__(self, dec: nn.Module, prefill: bool):
        super().__init__()
        self.model = Root(dec)
        self.prefill = prefill

    def forward(self, hidden_states, position_ids, causal_mask, current_pos, per_layer_inputs=None):
        if per_layer_inputs is None:
            return self.model.model(hidden_states, position_ids, causal_mask, current_pos, self.prefill)
        return self.model.model(hidden_states, position_ids, causal_mask, current_pos, self.prefill, per_layer_inputs)


class Embeddings(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.embed_tokens = nn.Embedding(w.shape[0], w.shape[1], dtype=DTYPE)
        self.embed_tokens.weight.data = w.to(DTYPE)
        self.embed_tokens.weight.requires_grad_(False)

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LMHead(nn.Module):
    """Vocabulary split into `n` 1x1 convolutions (ANE output width limit), one output each."""

    def __init__(self, w, n: int, softcap: float = 0.0):
        super().__init__()
        v = w.shape[0]
        self.sizes = [v // n + (1 if i < v % n else 0) for i in range(n)]
        self.softcap = softcap
        if softcap:  # cap * tanh(logits / cap): fold 1/cap into the weights
            w = w / softcap
        starts = [sum(self.sizes[:i]) for i in range(n)]
        self.heads = nn.ModuleList(
            [conv1x1(p, "lm_head", torch.arange(s0, s0 + len(p)), 1.0 / softcap if softcap else 1.0) for p, s0 in zip(torch.split(w, self.sizes), starts)]
        )

    def forward(self, hidden_states):
        hs = hidden_states.permute(0, 2, 1).unsqueeze(2)
        out = [h(hs).squeeze(2).transpose(1, 2) for h in self.heads]
        if self.softcap:
            out = [torch.tanh(o) * self.softcap for o in out]
        return tuple(out)


# ---------------------------------------------------------------------------------------------
# Gemma 4
# ---------------------------------------------------------------------------------------------


def gelu_tanh(x):
    """gelu_pytorch_tanh, with x (1 + c x^2) inside so that it does not overflow fp16."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x * (1.0 + 0.044715 * x * x))))


def gemma_slots(sp: Spec):
    """KV cache slots: the non-shared sliding / full attention layers, in order."""
    loc = [i for i in range(sp.first_shared) if sp.layer_types[i] == "sliding"]
    glob = [i for i in range(sp.first_shared) if sp.layer_types[i] == "full"]
    return loc, glob


def gemma_kv_source(sp: Spec, i: int) -> int:
    """The layer whose K/V layer `i` attends to: itself, or for the KV-shared layers the last
    non-shared layer of the same type."""
    if i < sp.first_shared:
        return i
    return max(j for j in range(sp.first_shared) if sp.layer_types[j] == sp.layer_types[i])


class GemmaAttention(nn.Module):
    """Gemma 4 attention: q/k RMSNorm with weight, v RMSNorm without, RoPE, no 1/sqrt(d)
    scaling; KV-shared layers only project q and read the cache of their source layer."""

    def __init__(self, sp: Spec, ck: Checkpoint, i: int):
        super().__init__()
        p = f"layers.{i}.self_attn."
        self.kind = sp.layer_types[i]
        self.hd = sp.head_dim if self.kind == "sliding" else sp.global_head_dim
        self.nh, self.nkv, self.eps = sp.n_heads, sp.n_kv, sp.eps
        slots = gemma_slots(sp)[0 if self.kind == "sliding" else 1]
        src = gemma_kv_source(sp, i)
        self.shared = src != i
        self.slot = slots.index(src)
        self.q_proj = conv1x1(ck.get(p + "q_proj.weight"), p + "q_proj")
        self.o_proj = conv1x1(ck.get(p + "o_proj.weight"), p + "o_proj")
        self.q_norm = RMSNorm(ck.get(p + "q_norm.weight"), sp.eps)
        if not self.shared:
            self.k_proj = conv1x1(ck.get(p + "k_proj.weight"), p + "k_proj")
            self.v_proj = conv1x1(ck.get(p + "v_proj.weight"), p + "v_proj")
            self.k_norm = RMSNorm(ck.get(p + "k_norm.weight"), sp.eps)

    def heads(self, y, n, t):
        if t == 1:
            return y.view(1, n, 1, self.hd)
        return y.view(1, n, self.hd, t).permute(0, 1, 3, 2)

    def forward(self, x, cos, sin, mask, current_pos, cache, shift=False):
        t = x.shape[1]
        hs = x.permute(0, 2, 1).unsqueeze(2)
        q = self.q_norm(self.heads(self.q_proj(hs), self.nh, t))
        q = q * cos + rotate_half(q) * sin
        s, nkv = self.slot, self.nkv
        if not self.shared:
            k = self.k_norm(self.heads(self.k_proj(hs), nkv, t))
            k = k * cos + rotate_half(k) * sin
            v = rms(self.heads(self.v_proj(hs), nkv, t), self.eps)
            kv = torch.cat([k, v], dim=1)
            if shift:
                # shifting window cache: drop the t oldest positions, append the new ones
                cache[s : s + 1] = torch.cat([cache[s : s + 1, :, t:, :], kv], dim=2)
            else:
                cache[s : s + 1, :, current_pos : current_pos + t, :] = kv
        kc = cache[s, :nkv]  # [nkv, ctx, hd]
        vc = cache[s, nkv:]
        rep = self.nh // self.nkv
        if rep > 1:
            kc = kc.unsqueeze(1).repeat(1, rep, 1, 1)
            vc = vc.unsqueeze(1).repeat(1, rep, 1, 1)
        kc = kc.view(1, self.nh, -1, self.hd)
        vc = vc.view(1, self.nh, -1, self.hd)
        att = torch.softmax(torch.matmul(q, kc.transpose(-1, -2)) + mask, dim=-1)
        o = torch.matmul(att, vc).permute(0, 1, 3, 2).reshape(1, self.nh * self.hd, 1, t)
        return self.o_proj(o).squeeze(2).permute(0, 2, 1)


class GemmaLayer(nn.Module):
    def __init__(self, sp: Spec, ck: Checkpoint, i: int):
        super().__init__()
        p = f"layers.{i}."
        self.idx, self.kind = i, sp.layer_types[i]

        def norm(n):
            return RMSNorm(ck.get(p + n + ".weight"), sp.eps)

        self.input_layernorm = norm("input_layernorm")
        self.post_attention_layernorm = norm("post_attention_layernorm")
        self.pre_feedforward_layernorm = norm("pre_feedforward_layernorm")
        self.post_feedforward_layernorm = norm("post_feedforward_layernorm")
        self.self_attn = GemmaAttention(sp, ck, i)
        self.gate_proj = conv1x1(ck.get(p + "mlp.gate_proj.weight"), p + "mlp.gate_proj")
        self.up_proj = conv1x1(ck.get(p + "mlp.up_proj.weight"), p + "mlp.up_proj")
        self.down_proj = conv1x1(ck.get(p + "mlp.down_proj.weight"), p + "mlp.down_proj")
        self.ple = sp.ple_dim
        if self.ple:
            self.per_layer_input_gate = conv1x1(ck.get(p + "per_layer_input_gate.weight"), p + "per_layer_input_gate")
            self.per_layer_projection = conv1x1(ck.get(p + "per_layer_projection.weight"), p + "per_layer_projection")
            self.post_per_layer_input_norm = norm("post_per_layer_input_norm")
        self.scalar = float(ck.get(p + "layer_scalar").item()) if ck.has(p + "layer_scalar") else 1.0

    def forward(self, x, cos, sin, mask, current_pos, cache, pli, shift=False):
        h = self.self_attn(self.input_layernorm(x), cos, sin, mask, current_pos, cache, shift)
        x = x + self.post_attention_layernorm(h)
        hs = self.pre_feedforward_layernorm(x).permute(0, 2, 1).unsqueeze(2)
        h = self.down_proj(gelu_tanh(self.gate_proj(hs)) * self.up_proj(hs)).squeeze(2).permute(0, 2, 1)
        x = x + self.post_feedforward_layernorm(h)
        if self.ple:
            hs = x.permute(0, 2, 1).unsqueeze(2)
            g = gelu_tanh(self.per_layer_input_gate(hs)) * pli.permute(0, 2, 1).unsqueeze(2)
            h = self.per_layer_projection(g).squeeze(2).permute(0, 2, 1)
            x = x + self.post_per_layer_input_norm(h)
        if self.scalar != 1.0:
            x = x * self.scalar
        return x


def rope_table(inv: torch.Tensor, ctx: int):
    freqs = torch.outer(torch.arange(ctx).float(), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(DTYPE), emb.sin().to(DTYPE)


class GemmaDecoder(nn.Module):
    """Gemma 4 decoder chunk. Extra input: per_layer_inputs [1, T, n_layers * ple_dim] (the
    combined per-layer embeddings, an output of the embeddings model). States: the KV caches of
    the sliding (local) and full (global) attention layers, which have different head dims."""

    def __init__(self, sp: Spec, ck: Checkpoint, start: int, end: int, ctx: int, batch: int):
        super().__init__()
        self.sp, self.start, self.end, self.ctx, self.batch = sp, start, end, ctx, batch
        self.layers = nn.ModuleList([GemmaLayer(sp, ck, i) for i in range(start, end)])
        self.last = end == sp.n_layers
        if self.last:
            self.norm = RMSNorm(ck.get("norm.weight"), sp.eps)
        hd, ghd = sp.head_dim, sp.global_head_dim
        c, s_ = rope_table(1.0 / (sp.theta_local ** (torch.arange(0, hd, 2).float() / hd)), ctx)
        self.register_buffer("cos_l", c)
        self.register_buffer("sin_l", s_)
        # proportional RoPE: the first partial_rotary_factor of the frequency pairs rotate
        n_rot = int(sp.global_rot_frac * ghd // 2)
        inv = 1.0 / (sp.theta ** (torch.arange(0, 2 * n_rot, 2).float() / ghd))
        c, s_ = rope_table(torch.cat([inv, torch.zeros(ghd // 2 - n_rot)]), ctx)
        self.register_buffer("cos_g", c)
        self.register_buffer("sin_g", s_)
        self.local_len = local_cache_len(sp, ctx, batch)
        self.shift = self.local_len < ctx
        self.sliding = ctx > sp.sliding_window and not self.shift
        if self.sliding:
            self.register_buffer("cols", torch.arange(ctx, dtype=torch.int32))
        for name, shape in state_shapes(sp, ctx, batch).items():
            self.register_buffer(name, torch.zeros(shape, dtype=DTYPE))

    def forward(self, x, position_ids, causal_mask, current_pos, prefill: bool, per_layer_inputs):
        sp, t = self.sp, x.shape[1]
        rope = {
            "sliding": (self.cos_l[position_ids].view(1, 1, t, -1), self.sin_l[position_ids].view(1, 1, t, -1)),
            "full": (self.cos_g[position_ids].view(1, 1, t, -1), self.sin_g[position_ids].view(1, 1, t, -1)),
        }
        masks = {"full": causal_mask, "sliding": causal_mask}
        if self.shift:
            # Slot j of the local cache (after this call's write) holds position
            # p + t - L + j. Query p + i sees slots L - t + i - W < j <= L - t + i (window band,
            # a constant per function) with a position >= 0: L - t - j <= p, read from row p of
            # the causal mask (column c is 0 iff c <= p), reversed.
            n = self.local_len
            i = torch.arange(t).view(-1, 1)
            j = torch.arange(n).view(1, -1)
            band = ((j > n - t + i - sp.sliding_window) & (j <= n - t + i)).to(DTYPE)
            band = ((1.0 - band) * -30000.0).view(1, 1, t, n)
            born = torch.flip(causal_mask[:, :, 0:1, 0 : n - t + 1], dims=[3])
            if t > 1:
                born = torch.cat([born, torch.zeros((1, 1, 1, t - 1), dtype=DTYPE)], dim=3)
            masks["sliding"] = band + born
        if self.sliding:
            # keys at positions <= p - window are outside the sliding window of query p
            out = self.cols.view(1, -1) <= (position_ids.view(-1, 1) - sp.sliding_window)
            masks["sliding"] = causal_mask + out.to(DTYPE).view(1, 1, t, -1) * -30000.0
        caches = {"sliding": self.kv_cache_local, "full": self.kv_cache_global}
        used = set()
        d = sp.ple_dim
        for layer in self.layers:
            cos, sin = rope[layer.kind]
            pli = per_layer_inputs[:, :, layer.idx * d : (layer.idx + 1) * d] if d else None
            x = layer(x, cos, sin, masks[layer.kind], current_pos, caches[layer.kind], pli, self.shift and layer.kind == "sliding")
            used.add(layer.kind)
        for kind, cache in caches.items():  # Core ML rejects declared states the graph never reads
            if kind not in used:
                x = x + cache[0, 0, 0, 0:1].view(1, 1, 1) * 0.0
        if self.last:
            x = self.norm(x[:, t - 1 :, :])
        return x


class GemmaEmbeddings(nn.Module):
    """input_ids -> hidden_states = E[ids] * sqrt(H) and per_layer_inputs, the per-layer
    embeddings combined with the projection of the token embedding (transformers'
    get_per_layer_inputs + project_per_layer_inputs), [1, T, n_layers * ple_dim]."""

    def __init__(self, sp: Spec, ck: Checkpoint):
        super().__init__()
        h, n, d = sp.hidden, sp.n_layers, sp.ple_dim
        self.n, self.d = n, d
        self.embed_tokens = nn.Embedding(sp.vocab, h, dtype=DTYPE)
        self.embed_tokens.weight.data = (ck.get("embed_tokens.weight") * h**0.5).to(DTYPE)
        if d:
            self.embed_tokens_per_layer = nn.Embedding(sp.vocab, n * d, dtype=DTYPE)
            self.embed_tokens_per_layer.weight.data = (ck.get("embed_tokens_per_layer.weight") * d**0.5).to(DTYPE)
            self.per_layer_model_projection = conv1x1(ck.get("per_layer_model_projection.weight") * h**-0.5)
            self.per_layer_projection_norm = RMSNorm(ck.get("per_layer_projection_norm.weight"), sp.eps)
        for p_ in self.parameters():
            p_.requires_grad_(False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)  # [1, T, H]
        if not self.d:
            return x
        t = x.shape[1]
        proj = self.per_layer_model_projection(x.permute(0, 2, 1).unsqueeze(2))  # [1, n*d, 1, T]
        proj = self.per_layer_projection_norm(proj.view(self.n, self.d, t).permute(2, 0, 1))  # [T, n, d]
        ple = self.embed_tokens_per_layer(input_ids).view(t, self.n, self.d)
        return x, ((proj + ple) * (2.0**-0.5)).reshape(1, t, self.n * self.d)


# ---------------------------------------------------------------------------------------------
# Core ML conversion
# ---------------------------------------------------------------------------------------------


@dataclass
class Quant:
    """Weight compression: none, lutN[,G] (k-means palettization with N in 1,2,3,4,6,8 bits, G
    output channels per LUT, 0 = per tensor), int8[,tensor] (per-channel or per-tensor linear) or
    int4[,B] (per-block linear, block B).

    Only LUTs run correctly on the Neural Engine. Measured on an M5 Max / macOS 26 (Qwen3-0.6B):
    int8 weights compile for the ANE but give wrong results there (logit cosine 0.75 against
    0.9992 on the GPU, as if activations were quantized too), int4 per-block does not compile for
    the ANE. They are kept for packages meant for the GPU (CPU_AND_GPU)."""

    kind: str = "none"
    bits: int = 0
    group: int = 0

    @staticmethod
    def parse(s: str) -> "Quant":
        s = s.strip().lower()
        if s in ("none", "fp16", ""):
            return Quant()
        if s.startswith("lut"):
            b, _, g = s[3:].partition(",")
            return Quant("lut", int(b), int(g) if g else 8)
        if s.startswith("int8"):
            return Quant("int8", 8, -1 if s.endswith(",tensor") else 0)
        if s.startswith("int4"):
            _, _, b = s.partition(",")
            return Quant("int4", 4, int(b) if b else 32)
        raise argparse.ArgumentTypeError(f"bad quantization {s!r}")

    @property
    def suffix(self):
        return {"none": "", "lut": f"_lut{self.bits}", "int8": "_int8", "int4": "_int4"}[self.kind]

    @property
    def meta(self):
        return self.bits if self.kind == "lut" else "none"

    def __str__(self):
        if self.kind == "lut":
            return f"lut{self.bits},{self.group}"
        if self.kind == "int4":
            return f"int4,{self.group}"
        return self.kind


class Palettizer:
    """Data-aware palettization (--lut-method fisher / gptq / wkmeans, tools/coreml_quant.py):
    the torch modules' 1x1 conv weights are replaced by their palettized values before tracing,
    and the (LUT, indices, scale) of each is remembered by weight content so that `compress`
    writes exactly these as constexpr_lut_to_dense ops into the converted program.

    With --lut-mix B the decoder weights get mixed precision: all are palettized at the
    --ffn-quant bits first, then the ones with the largest estimated loss increase per weight
    (0.5 sum F (w - q)^2, Fisher F) move to the next width up (4 -> 6) until the average
    reaches B bits."""

    def __init__(self, args, stats: tuple | None):
        # per-channel scales on top of the LUT (coreml_quant scale_mode "rms") did not improve
        # quality in tests, so they are not exposed
        self.method, self.scale, self.rounds = args.lut_method, "none", args.lut_rounds
        self.hs, self.fisher = stats or ({}, {})
        self.table = {}
        self.keep = set()  # weights left fp16 on purpose
        self.bits = {}  # src -> bits (mixed precision)
        self.cache = {}  # (src, bits) -> (lut, idx uint8, scale) of the whole source weight
        self.mix = None

    def active(self, q: "Quant") -> bool:
        return q.kind == "lut" and self.method != "kmeans"

    @staticmethod
    def wider(bits: int) -> int:
        return min(b for b in (2, 3, 4, 6, 8) if b >= min(bits + 2, 8))

    def plan(self, ck: Checkpoint, q: "Quant", budget: float):
        """Choose the bits of every decoder weight for an average of `budget` bits."""
        import coreml_quant as cq

        if not self.fisher:
            raise SystemExit("--lut-mix needs --lut-method fisher")
        t0 = time.time()
        cand, total = [], 0
        for src, imp in self.fisher.items():
            if not src.startswith("layers.") or not ck.has(src + ".weight"):
                continue
            w = ck.get(src + ".weight").to(DTYPE).float()
            o = w.shape[0]
            group = q.group if q.group > 0 else o
            if o % group or w.numel() <= 2048:
                continue
            lut, idx, sc, wh = cq.palettize(w, None, q.bits, group, "fisher", self.scale, imp=imp)
            self.cache[src, q.bits] = (lut, idx.to(torch.uint8), sc)
            loss = float(0.5 * (imp * (w - wh) ** 2).sum())
            cand.append((loss / w.numel(), src, w.numel()))
            total += w.numel()
        hi = self.wider(q.bits)
        extra = (budget - q.bits) / (hi - q.bits) * total
        used = 0
        for _, src, n in sorted(cand, reverse=True):
            if used + n <= extra:
                self.bits[src] = hi
                used += n
        self.mix = q.bits + (hi - q.bits) * used / max(total, 1)
        print(f"  mixed precision: {len(self.bits)} of {len(cand)} decoder weights at {hi} bits, "
              f"average {self.mix:.2f} bits ({time.time() - t0:.0f}s)")

    def prepare(self, module: nn.Module, q: "Quant"):
        """Palettize the 1x1 convolutions of `module` in place (fp16 weights take the values
        of their LUT), recording their LUT form."""
        import coreml_quant as cq

        if not self.active(q):
            return
        t0, n, err = time.time(), 0, []
        for _name, m in module.named_modules():
            if not isinstance(m, nn.Conv2d):
                continue
            if m.kernel_size != (1, 1) or m.groups != 1:
                self.keep.add(cq.weight_key(m.weight.data.numpy()))  # e.g. the GDN depthwise conv: fp16
                continue
            o = m.weight.shape[0]
            group = q.group if q.group > 0 else o
            if o % group or m.weight.numel() <= 2048 or getattr(m, "keep_fp16", False):
                self.keep.add(cq.weight_key(m.weight.data.numpy()))
                continue  # left fp16 (e.g. the MoE router, the 1-row shared-expert gate)
            w = m.weight.data.float().view(o, -1)
            src = getattr(m, "src", None)
            bits = self.bits.get(src, q.bits)
            whole = m.src_rows is None and m.src_gain == 1.0
            if whole and (src, bits) in self.cache:
                lut, idx, sc = self.cache.pop((src, bits))
                idx = idx.long()
                lr = lut.repeat_interleave(group, 0)
                wh = torch.gather(lr * sc.view(-1, 1) if sc is not None else lr, 1, idx)
            else:
                h = self.hs.get(src)
                imp = self.fisher.get(src)
                if imp is not None and m.src_rows is not None:
                    imp = imp[m.src_rows]
                if imp is not None:
                    imp = imp / (m.src_gain**2)  # importance of the scaled weight
                method = self.method
                if (method == "fisher" and imp is None) or (method in ("gptq", "wkmeans") and h is None):
                    method = "kmeans"  # no statistics for this weight (e.g. fused MoE experts)
                lut, idx, sc, wh = cq.palettize(w, h, bits, group, method, self.scale, self.rounds, imp=imp)
                if h is not None:
                    err.append(cq.output_error(w, wh, h))
            m.weight.data = wh.to(DTYPE).view_as(m.weight).contiguous()
            self.table[cq.weight_key(m.weight.data.numpy())] = cq.lut_entry(lut, idx, sc, group, bits, tuple(m.weight.shape))
            n += 1
        e = f", mean relative output error {np.mean(err):.2e}" if err else ""
        print(f"  palettized {n} weights ({self.method}, {q}, scale {self.scale}) in {time.time() - t0:.0f}s{e}")

    def apply(self, mlmodel):
        import coreml_quant as cq

        m, replaced, unmatched = cq.apply_luts(mlmodel, self.table, self.keep)
        if unmatched:
            print(f"  warning: {unmatched} large fp16 conv weights were not palettized (changed by the converter?)", file=sys.stderr)
        return m


PALETTIZER: Palettizer | None = None


def calibrate(args) -> dict:
    """Input statistics of every linear layer of the transformers model (float32) over the
    calibration sequences, keyed by the checkpoint-relative name used by conv1x1(src=...)."""
    import coreml_quant as cq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    print(f"calibration ({args.lut_method}): loading {args.model} in float32 ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    model.to(cq.model_device())
    seqs = cq.calibration_sequences(
        model, tok, n_new=args.calib_tokens, text_file=args.calib_file, cache=args.calib_cache, samples=args.calib_samples
    )

    def select(n):
        return (".layers." in n and "vision" not in n and "audio" not in n) or n.endswith("lm_head")

    def short(name):
        return name[name.index("layers.") :] if "layers." in name else "lm_head"

    hs, fis = {}, {}
    if args.lut_method in ("gptq", "wkmeans"):
        keys, h = cq.collect_hessians(model, seqs, select)
        hs = {short(n): h[k] for n, k in keys.items()}
    if args.lut_method == "fisher":
        fis = {short(n): f for n, f in cq.collect_fisher(model, seqs, select).items()}
    del model
    print(f"calibration done in {time.time() - t0:.0f}s")
    return hs, fis


def compress(mlmodel, q: Quant, workers: int, op_type: str = "conv"):
    """Compress the weights consumed by `op_type` ops only (projections, or the embedding
    gather): RoPE tables and the constant masks of the delta rule stay fp16. Linear int8
    quantization of every constant also ran incorrectly on the ANE (logit cosine 0.75)."""
    import coremltools.optimize as cto

    if q.kind == "none":
        return mlmodel
    if op_type == "conv" and PALETTIZER is not None and PALETTIZER.active(q):
        return PALETTIZER.apply(mlmodel)
    if q.kind == "lut":
        # coremltools 9.0 closes its shared k-means pool after a call without clearing it.
        from coremltools.optimize.coreml import _quantization_passes as qp

        if qp.palettize_weights._compress_pool is not None:
            qp.palettize_weights._compress_pool.terminate()
            qp.palettize_weights._compress_pool = None
        if q.group > 0:
            op = cto.coreml.OpPalettizerConfig(
                mode="kmeans", nbits=q.bits, granularity="per_grouped_channel", group_size=q.group, num_kmeans_workers=workers
            )
        else:
            op = cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=q.bits, granularity="per_tensor", num_kmeans_workers=workers)
        cfg = cto.coreml.OptimizationConfig(op_type_configs={op_type: op})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cto.coreml.palettize_weights(mlmodel, cfg)
    if q.kind == "int8":
        gran = "per_tensor" if q.group < 0 else "per_channel"
        op = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity=gran)
    else:
        op = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=q.group)
    return cto.coreml.linear_quantize_weights(mlmodel, cto.coreml.OptimizationConfig(op_type_configs={op_type: op}))


def convert(module, example, inputs, outputs, states):
    import coremltools as ct

    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example, check_trace=False)
    return ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        states=states or None,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
        skip_model_load=True,
    )


def reset_states(dec: Decoder):
    with torch.no_grad():
        for name in state_shapes(dec.sp, dec.ctx, getattr(dec, "batch", 0)):
            getattr(dec, name).zero_()


def ct_states(sp: Spec, ctx: int, batch: int):
    import coremltools as ct

    return [
        ct.StateType(wrapped_type=ct.TensorType(shape=shape, dtype=np.float16), name=f"model.model.{name}")
        for name, shape in state_shapes(sp, ctx, batch).items()
    ]


def set_meta(mlmodel, desc: str, extra: dict):
    mlmodel.author = "oxinfer coreml_convert.py (ANEMLL format)"
    mlmodel.short_description = desc
    for k, v in extra.items():
        mlmodel.user_defined_metadata[k] = str(v)


def compile_package(pkg: str, out_dir: str) -> str:
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, out_dir], check=True, stdout=subprocess.DEVNULL)
    return os.path.join(out_dir, os.path.basename(pkg)[: -len(".mlpackage")] + ".mlmodelc")


def layer_sizes(ck: Checkpoint, n_layers: int):
    """Weight count of every decoder layer (from the checkpoint headers)."""
    sizes = [0] * n_layers
    for name, f in ck.names.items():
        if name.startswith(ck.prefix + "layers."):
            i = int(name[len(ck.prefix) + 7 :].split(".")[0])
            if i < n_layers:
                sizes[i] += math.prod(ck.files[f].get_slice(name).get_shape())
    return sizes


def chunk_ranges(sizes: list, chunks: int):
    """Split the layers into `chunks` consecutive ranges of about equal weight size."""
    n, total = len(sizes), sum(sizes)
    out, s, acc = [], 0, 0
    for c in range(chunks - 1):
        target = total * (c + 1) / chunks
        e = s + 1
        acc += sizes[s]
        # extend while the next layer brings the cumulative size closer to the target
        while e < n - (chunks - 1 - c) and abs(acc + sizes[e] - target) < abs(acc - target):
            acc += sizes[e]
            e += 1
        out.append((s, e))
        s = e
    out.append((s, n))
    return out


def bits_per_weight(q: Quant) -> float:
    return {"none": 16, "lut": q.bits, "int8": 8, "int4": 4}[q.kind]


def auto_chunks(ck: Checkpoint, q: Quant, max_mb: float) -> int:
    """Chunks needed to keep each decoder chunk under max_mb (decoder weights + 10%)."""
    n = 0
    for name, f in ck.names.items():
        if name.startswith(ck.prefix + "layers."):
            n += math.prod(ck.files[f].get_slice(name).get_shape())
    total_mb = n * bits_per_weight(q) / 8 / 1e6 * 1.1
    return max(1, math.ceil(total_mb / max_mb))


def write_meta(path, args, sp: Spec, names: dict, chunks: int, lm_sizes, gen_cfg: dict):
    a = args
    lines = [
        "model_info:",
        f"  name: anemll-{a.name}-ctx{a.context_length}",
        f"  version: {FORMAT_VERSION}",
        "  description: |",
        f"    {a.name} on the Apple Neural Engine (converted by oxinfer tools/coreml_convert.py)",
        f"    Context length: {a.context_length}",
        f"    Batch size: {a.batch_size}",
        f"    Chunks: {chunks}",
        "  license: MIT",
        "  author: oxinfer",
        "  framework: Core ML",
        "  language: Python",
        f"  architecture: {sp.arch}",
        "  parameters:",
        f"    context_length: {a.context_length}",
        f"    batch_size: {a.batch_size}",
        f"    lut_embeddings: {a.embed_quant.meta}",
    ]
    for key, q in (("embeddings", a.embed_quant), ("ffn", a.ffn_quant), ("lmhead", a.lmhead_quant)):
        if key != "embeddings":
            lines.append(f"    lut_{key}: {q.meta}")
        if q.kind == "lut":
            lines.append(f"    lut_{key}_per_channel: {q.group}")
            if key == "ffn" and PALETTIZER is not None and PALETTIZER.mix:
                lines.append(f"    lut_ffn_mixed: {PALETTIZER.mix:.2f}     # average bits (some weights at {Palettizer.wider(q.bits)})")
        elif q.kind != "none":
            lines.append(f"    quant_{key}: {q}")
    lines += [
        f"    num_chunks: {chunks}",
        f"    model_prefix: {a.prefix}",
        f"    embeddings: {names['embeddings']}",
        f"    lm_head: {names['lm_head']}",
        f"    ffn: {names['ffn']}",
        f"    split_lm_head: {len(lm_sizes)}",
        f"    vocab_size: {sp.vocab}",
        f"    lm_head_chunk_sizes: [{', '.join(str(s) for s in lm_sizes)}]",
        "    prefill_dynamic_slice: true",
        "    prefill_last_token_output: true",
        f"    hidden_size: {sp.hidden}",
        f"    num_layers: {sp.n_layers}",
    ]
    if gen_cfg.get("temperature") is not None:
        lines += [
            "    recommended_sampling:",
            f"      do_sample: {str(bool(gen_cfg.get('do_sample', True))).lower()}",
            f"      temperature: {gen_cfg.get('temperature')}",
            f"      top_p: {gen_cfg.get('top_p', 1.0)}",
            f"      top_k: {gen_cfg.get('top_k', 0)}",
        ]
    # oxinfer extensions (docs/coreml-format.md)
    if sp.arch == "qwen3_5":
        lines += [
            "    recurrent_state: true",
            "    prefill_full_batches_only: true",
            f"    gdn_chunk_size: {a.gdn_chunk}",
        ]
        if sp.n_experts:
            lines.append(f"    moe: {{experts: {sp.n_experts}, experts_per_token: {sp.top_k}, evaluation: dense}}   # informational")
    if sp.arch == "gemma4":
        lines += [f"    local_attention_window: {sp.sliding_window}", f"    final_logit_softcap: {sp.softcap}"]
        n_local = local_cache_len(sp, a.context_length, a.batch_size)
        if n_local < a.context_length:
            lines.append(f"    local_cache_length: {n_local}       # shifting window cache (docs/coreml-format.md)")
        if sp.ple_dim:
            lines.append(f"    per_layer_inputs: {sp.n_layers * sp.ple_dim}")
        if sp.first_shared < sp.n_layers:
            src = ", ".join(f"{i}: {gemma_kv_source(sp, i)}" for i in range(sp.first_shared, sp.n_layers))
            lines.append(f"    kv_shared_layers: {{{src}}}")
    if sp.arch != "qwen3":
        lines += [f"    layer_types: [{', '.join(sp.layer_types)}]", "    states:"]
        for name, shape in state_shapes(sp, a.context_length, a.batch_size).items():
            if name == "kv_cache_0":
                layers = sp.full_layers
            elif name.startswith("kv_cache_"):
                layers = gemma_slots(sp)[0 if name.endswith("local") else 1]
            else:
                layers = sp.linear_layers
            lines.append(f"      - name: model_model_{name}")
            lines.append(f"        shape: [{', '.join(str(s) for s in shape)}]")
            lines.append("        dtype: float16")
            lines.append(f"        layers: [{', '.join(str(i) for i in layers)}]")
    lines += [
        "",
        "# Conversion parameters",
        f"# source: {a.model}",
        f"# command: {' '.join(sys.argv)}",
        f"# ffn: {a.ffn_quant}  lm_head: {a.lmhead_quant}  embeddings: {a.embed_quant}  LUT method: {a.lut_method}",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def copy_tokenizer(src: str, dst: str, cfg: dict):
    for f in TOKENIZER_FILES:
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    # config.json: the source config plus tokenizer_class (swift-transformers reads it).
    out = dict(cfg)
    tc_path = os.path.join(src, "tokenizer_config.json")
    if os.path.exists(tc_path):
        tcls = json.load(open(tc_path)).get("tokenizer_class")
        if tcls:
            out["tokenizer_class"] = tcls
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 2)[2], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="Hugging Face checkpoint directory (config.json + safetensors)")
    ap.add_argument("--output", required=True, help="output directory (created)")
    ap.add_argument("--name", help="model name for meta.yaml (default: checkpoint directory name)")
    ap.add_argument("--prefix", help="file name prefix (default: qwen for qwen3, qwen35 for qwen3_5)")
    ap.add_argument("--context-length", type=int, default=512, help="KV cache length / max positions (default 512)")
    ap.add_argument("--batch-size", type=int, default=64, help="prefill batch (tokens per prefill call, default 64)")
    ap.add_argument("--chunks", default="auto", help="number of FFN chunks or 'auto' (<= --max-chunk-mb each)")
    ap.add_argument("--max-chunk-mb", type=float, default=950, help="chunk size target for --chunks auto (default 950)")
    ap.add_argument(
        "--ffn-quant",
        type=Quant.parse,
        default=Quant.parse("lut6"),
        help="decoder layers: none|lutN[,G]|int8|int4[,B] (default lut6 = lut6,8, what ANEMLL recommends and ships; lut4 is "
        "1.5x smaller and loses 12-14 points of top-1 agreement (vs lut6) on 0.6-2B models with the Fisher LUTs, 7-9 with "
        "--lut-mix 4.5)",
    )
    ap.add_argument("--lmhead-quant", type=Quant.parse, default=Quant.parse("lut6"), help="LM head (default lut6). int8/int4 are GPU-only, see below")
    ap.add_argument(
        "--embed-quant",
        type=Quant.parse,
        default=Quant.parse("none"),
        help="embeddings (default none: Core ML memory-maps an fp16 table and pages in the rows it reads, while a "
        "palettized gather table is decompressed at load time; Gemma 4 E2B with lut4: 87 s and +16.7 GB RSS)",
    )
    ap.add_argument(
        "--lut-method",
        choices=("fisher", "kmeans", "wkmeans", "gptq"),
        default="fisher",
        help="how LUT weights are computed: fisher (default: k-means weighted by the Fisher information of every "
        "weight, measured on a calibration set with the float32 transformers model; needs transformers and RAM for "
        "the float32 model plus one gradient per linear weight), kmeans (Core ML's plain k-means, no calibration), "
        "gptq / wkmeans (GPTQ rounding over the LUT / k-means weighted by input second moments: experimental, "
        "they overfit the small calibration set)",
    )
    ap.add_argument(
        "--lut-mix",
        type=float,
        help="fisher: mixed precision for the decoder layers, average bits B between the --ffn-quant bits and the "
        "next width (4 -> 6): the weights whose palettization costs the most (Fisher estimate) get the wider LUT, "
        "e.g. --ffn-quant lut4 --lut-mix 4.5",
    )
    ap.add_argument("--lut-rounds", type=int, default=1, help="gptq: GPTQ + LUT refit rounds before the final assignment (default 1)")
    ap.add_argument("--calib-file", help="calibration text file (default: the model's own answers to built-in chat prompts)")
    ap.add_argument("--calib-tokens", type=int, default=192, help="tokens generated per built-in calibration prompt (default 192)")
    ap.add_argument("--calib-samples", type=int, default=1, help="answers per built-in prompt: greedy, then sampled ones (default 1)")
    ap.add_argument("--calib-cache", help="file caching the calibration token sequences (reused when it exists)")
    ap.add_argument("--no-quant", action="store_true", help="fp16 everywhere (same as all three = none)")
    ap.add_argument(
        "--full-local-cache",
        action="store_true",
        help="gemma4: keep context-length caches for the sliding-window layers too (default: a shifting cache of "
        "window + batch positions once the context is longer than that)",
    )
    ap.add_argument("--split-lm-head", type=int, default=16, help="LM head output splits (default 16)")
    ap.add_argument(
        "--gdn-chunk",
        type=int,
        default=0,
        help="Gated DeltaNet prefill: chunk of the chunked delta rule (power of 2 dividing the batch; default = batch size, "
        "the fastest on the ANE); 1 = sequential scan over the batch",
    )
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="k-means worker processes")
    ap.add_argument("--keep-packages", action="store_true", help="also keep the .mlpackage files")
    ap.add_argument("--parts", default="embeddings,ffn,lm_head", help="comma list of parts to (re)build")
    ap.add_argument("--only-chunks", help="comma list of decoder chunk numbers (1-based) to build, e.g. to resume or split a long conversion")
    args = ap.parse_args()
    os.environ.setdefault("PYTHONWARNINGS", "ignore")  # k-means worker processes
    global SHIFT_LOCAL_CACHE
    SHIFT_LOCAL_CACHE = not args.full_local_cache
    if args.no_quant:
        args.ffn_quant = args.lmhead_quant = args.embed_quant = Quant()
    for q in (args.ffn_quant, args.lmhead_quant, args.embed_quant):
        if q.kind in ("int8", "int4"):
            print(f"warning: {q} weights do not run correctly on the Neural Engine (GPU only), see Quant", file=sys.stderr)

    import coremltools as ct

    src = os.path.abspath(os.path.expanduser(args.model))
    cfg = json.load(open(os.path.join(src, "config.json")))
    sp = load_spec(cfg)
    args.model = src
    args.name = args.name or os.path.basename(src.rstrip("/")).replace("--", "-")
    args.prefix = args.prefix or {"qwen3": "qwen", "qwen3_5": "qwen35", "gemma4": "gemma4"}[sp.arch]
    ck = Checkpoint(src)
    global PALETTIZER
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # coreml_quant.py
    parts_ = set(args.parts.split(","))
    needs = [q for q, part in ((args.ffn_quant, "ffn"), (args.lmhead_quant, "lm_head")) if part in parts_]
    stats = None
    if args.lut_method != "kmeans" and any(q.kind == "lut" for q in needs):
        stats = calibrate(args)
    PALETTIZER = Palettizer(args, stats)
    if args.lut_mix and "ffn" in parts_:
        if args.ffn_quant.kind != "lut":
            raise SystemExit("--lut-mix needs a LUT --ffn-quant")
        PALETTIZER.plan(ck, args.ffn_quant, args.lut_mix)
    args.gdn_chunk = args.gdn_chunk or args.batch_size
    if args.gdn_chunk & (args.gdn_chunk - 1) or args.batch_size % args.gdn_chunk:
        raise SystemExit("--gdn-chunk must be a power of two dividing --batch-size")
    ctx, bs = args.context_length, args.batch_size
    chunks = auto_chunks(ck, args.ffn_quant, args.max_chunk_mb) if args.chunks == "auto" else int(args.chunks)
    parts = set(args.parts.split(","))
    out = os.path.abspath(args.output)
    os.makedirs(out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="coreml_convert_", dir=out)
    print(f"{sp.arch}: {sp.n_layers} layers ({len(sp.full_layers)} full attention), hidden {sp.hidden}, vocab {sp.vocab}; "
          f"ctx {ctx}, batch {bs}, {chunks} chunk(s); ffn {args.ffn_quant}, lm_head {args.lmhead_quant}, embeddings {args.embed_quant}")
    common_meta = {
        "com.anemll.context_length": ctx,
        "com.anemll.batch_size": bs,
        "com.anemll.num_chunks": chunks,
        "com.anemll.architecture": sp.arch,
        "com.anemll.info": f"oxinfer coreml_convert.py from {os.path.basename(src)}",
    }

    def finish(mlmodel, base: str, q: Quant, desc: str, extra: dict):
        set_meta(mlmodel, desc, {**common_meta, **extra, "com.anemll.lut_bits": q.meta})
        pkg = os.path.join(tmp, base + ".mlpackage")
        mlmodel.save(pkg)
        dst = os.path.join(out, base + ".mlmodelc")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        compile_package(pkg, out)
        if args.keep_packages:
            final = os.path.join(out, base + ".mlpackage")
            if os.path.exists(final):
                shutil.rmtree(final)
            shutil.move(pkg, final)
        return base + ".mlmodelc"

    names = {}
    t0 = time.time()
    # Embeddings
    emb_w = ck.get("embed_tokens.weight")
    base = f"{args.prefix}_embeddings{args.embed_quant.suffix}"
    names["embeddings"] = base + ".mlmodelc"
    if "embeddings" in parts:
        print("embeddings ...")
        shape = ct.EnumeratedShapes(shapes=[[1, 1], [1, bs]], default=[1, 1])
        outputs = [ct.TensorType(name="hidden_states", dtype=np.float16)]
        if sp.arch == "gemma4":
            module = GemmaEmbeddings(sp, ck)
            if sp.ple_dim:
                outputs.append(ct.TensorType(name="per_layer_inputs", dtype=np.float16))
        else:
            module = Embeddings(emb_w)
        m = convert(
            module,
            (torch.zeros((1, 1), dtype=torch.int32),),
            [ct.TensorType(name="input_ids", shape=shape, dtype=np.int32)],
            outputs,
            None,
        )
        del module
        m = compress(m, args.embed_quant, args.workers, "gather")
        finish(m, base, args.embed_quant, "token embeddings", {})

    # LM head
    if sp.tied or not ck.has("lm_head.weight", raw=True):
        head_w = emb_w
    else:
        head_w = ck.get("lm_head.weight", raw=True)
    head = LMHead(head_w, args.split_lm_head, sp.softcap)
    base = f"{args.prefix}_lm_head{args.lmhead_quant.suffix}"
    names["lm_head"] = base + ".mlmodelc"
    if "lm_head" in parts:
        print("lm_head ...")
        PALETTIZER.prepare(head, args.lmhead_quant)
        m = convert(
            head,
            (torch.zeros((1, 1, sp.hidden), dtype=DTYPE),),
            [ct.TensorType(name="hidden_states", shape=(1, 1, sp.hidden), dtype=np.float16)],
            [ct.TensorType(name=f"logits{i + 1}", dtype=np.float16) for i in range(args.split_lm_head)],
            None,
        )
        m = compress(m, args.lmhead_quant, args.workers)
        finish(m, base, args.lmhead_quant, "LM head (vocabulary split)", {"com.anemll.split_lm_head": args.split_lm_head})
    del emb_w, head_w

    # Decoder chunks: infer and prefill are converted and compressed separately, then merged into
    # one multifunction package. Both see byte-identical fp16 weights and k-means is seeded
    # (random_state=0), so the compressed weights are identical too and stored once.
    for ci, (s, e) in enumerate(chunk_ranges(layer_sizes(ck, sp.n_layers), chunks)):
        base = f"{args.prefix}_FFN_PF{args.ffn_quant.suffix}_chunk_{ci + 1:02d}of{chunks:02d}"
        if ci == 0:
            names["ffn"] = base + ".mlmodelc"
        if "ffn" not in parts or (args.only_chunks and str(ci + 1) not in args.only_chunks.split(",")):
            continue
        print(f"chunk {ci + 1}/{chunks}: layers {s}..{e - 1}")
        dec = GemmaDecoder(sp, ck, s, e, ctx, bs) if sp.arch == "gemma4" else Decoder(sp, ck, s, e, ctx, args.gdn_chunk)
        PALETTIZER.prepare(dec, args.ffn_quant)
        fn_pkgs = {}
        for fn, t in (("infer", 1), ("prefill", bs)):
            ex = (
                torch.zeros((1, t, sp.hidden), dtype=DTYPE),
                torch.arange(t, dtype=torch.int32),
                torch.zeros((1, 1, t, ctx), dtype=DTYPE),
                torch.zeros((1,), dtype=torch.int32),
            )
            inputs = [
                ct.TensorType(name="hidden_states", shape=(1, t, sp.hidden), dtype=np.float16),
                ct.TensorType(name="position_ids", shape=(t,), dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=(1, 1, t, ctx), dtype=np.float16),
                ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
            ]
            if sp.ple_dim:
                ex += (torch.zeros((1, t, sp.n_layers * sp.ple_dim), dtype=DTYPE),)
                inputs.append(ct.TensorType(name="per_layer_inputs", shape=(1, t, sp.n_layers * sp.ple_dim), dtype=np.float16))
            reset_states(dec)
            m = convert(
                ChunkFn(dec, fn == "prefill"),
                ex,
                inputs,
                [ct.TensorType(name="output_hidden_states", dtype=np.float16)],
                ct_states(sp, ctx, bs),
            )
            reset_states(dec)
            if args.ffn_quant.kind != "none":
                print(f"  {fn}: compressing ({args.ffn_quant}) ...")
                m = compress(m, args.ffn_quant, args.workers)
            fn_pkgs[fn] = os.path.join(tmp, f"{base}_{fn}.mlpackage")
            m.save(fn_pkgs[fn])
            del m
        desc = ct.utils.MultiFunctionDescriptor()
        desc.add_function(fn_pkgs["infer"], "main", "infer")
        desc.add_function(fn_pkgs["prefill"], "main", "prefill")
        desc.default_function_name = "infer"
        merged = os.path.join(tmp, f"{base}_merged.mlpackage")
        ct.utils.save_multifunction(desc, merged)
        m = ct.models.MLModel(merged, skip_model_load=True)
        finish(
            m,
            base,
            args.ffn_quant,
            f"decoder layers {s}..{e - 1} (functions infer, prefill)",
            {"com.anemll.chunk_no": ci + 1, "com.anemll.layers": f"{s}-{e - 1}"},
        )
        del dec

    lm_sizes = head.sizes
    gen_cfg = {}
    if os.path.exists(os.path.join(src, "generation_config.json")):
        gen_cfg = json.load(open(os.path.join(src, "generation_config.json")))
    write_meta(os.path.join(out, "meta.yaml"), args, sp, names, chunks, lm_sizes, gen_cfg)
    copy_tokenizer(src, out, cfg)
    shutil.rmtree(tmp)
    print(f"done in {time.time() - t0:.0f}s: {out}")


if __name__ == "__main__":
    main()
