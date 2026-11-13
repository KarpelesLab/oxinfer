#!/usr/bin/env python3
# Reference outputs of a (tiny) ERNIE 5.5 checkpoint (`ernie5_5_moe`, baidu/ERNIE-5.5-Flash-Base) in
# float32 on the CPU. Baidu publishes the model code as a vLLM plugin (github.com/baidu/vLLM-ERNIE,
# `src/vllm_ernie/models/ernie5_5/`), which subclasses vLLM's DeepSeek V4 attention and runs on
# NVIDIA kernels only (FlashMLA, DeepGEMM, TileLang); it cannot run a tiny checkpoint on a CPU.
# This script is that plugin's forward pass in plain PyTorch:
#   - the HCA layers are DeepSeek's own `inference/model.py` `Attention` (deepseek-ai/DeepSeek-V4-Flash,
#     the code vLLM's `DeepseekV4Attention` implements; kernels replaced as in ref_deepseek4.py) with
#     the plugin's two additions (`Ernie5_5HCAAttentionMixin`): the head mix `I + V U^T` (VHA) on the
#     attention output and the sigmoid gate `o_gate(qr)` on the grouped low-rank output;
#   - the MLA layers with the DSA indexer (`Ernie5_5DSAAttention`, `Ernie5_5Indexer`,
#     `merge_swa_topk`), the hyper-connections with the plugin's RMS factor 1 / (sqrt(mean) + eps)
#     (`mhc.py`, its tests' `_mhc_pre_ref`), the SiTU MLP, the latent MoE with two routers
#     (`moe.py`), the MultiMax head (`model.py`) and the MTP layer (`mtp.py`) are written out here
#     from the plugin's code, with the tensor names of the checkpoint.
# Left out, as for DeepSeek V4: the quantization of cached keys and of the indexer (FP8 / MXFP4)
# and the Hadamard rotation before it (orthonormal: no score changes).
#
#   prefill: logits at every position; decode: greedy steps, one token at a time;
#   MTP: the draft layer teacher-forced over the prompt (`mtp_logits[i]`: from the target's
#   residual at position i and token i + 1, the prediction of token i + 2), then a chain of
#   drafts after the prompt (`draft`, `draft_logits`).
# usage: ref_ernie55.py MODEL_DIR MODEL_PY OUT.safetensors N_PROMPT N_DECODE [SEED] [--chunk N]
#            [--drafts K]
#   MODEL_PY: inference/model.py of deepseek-ai/DeepSeek-V4-Flash. --chunk: the prompt as one
#   prefill of N tokens followed by one-token steps (model.py only prefills from position 0).
import copy, importlib.util, json, math, os, sys, types

import torch
import torch.nn.functional as F

VHA_RANK = 16  # `attention.py`


def ds4(model_py):
    """DeepSeek's `inference/model.py` as a module, its kernels replaced by PyTorch."""
    kernel = types.ModuleType("kernel")

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        assert inplace
        return x

    def fp4_act_quant(x, block_size=32, inplace=False):
        assert inplace
        return x

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        b, m, h, dd = q.shape
        idx = topk_idxs.long()
        valid = idx >= 0
        g = torch.gather(kv.unsqueeze(1).expand(b, m, -1, dd), 2, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, dd))
        s = torch.einsum("bmhd,bmkd->bmhk", q.float(), g.float()) * softmax_scale
        s = s.masked_fill(~valid.unsqueeze(2), float("-inf"))
        sink = attn_sink.float().view(1, 1, h, 1).expand(b, m, h, 1)
        p = torch.softmax(torch.cat([s, sink], dim=-1), dim=-1)[..., :-1]
        return torch.einsum("bmhk,bmkd->bmhd", p, g.float()).to(q.dtype)

    def unused(*a, **k):
        raise RuntimeError("not reached")

    kernel.act_quant, kernel.fp4_act_quant, kernel.sparse_attn = act_quant, fp4_act_quant, sparse_attn
    kernel.hc_split_sinkhorn, kernel.fp8_gemm, kernel.fp4_gemm = unused, unused, unused
    sys.modules["kernel"] = kernel
    spec = importlib.util.spec_from_file_location("dsv4_model", model_py)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    M.rotate_activation = lambda x: x
    # What `Transformer.__init__` sets.
    M.world_size, M.rank, M.default_dtype, M.scale_fmt, M.scale_dtype = 1, 0, torch.bfloat16, None, torch.float32
    return M


def hca_class(M):
    class ErnieHCA(M.Attention):
        """`Attention.forward` with the plugin's gate and head mix (the two marked lines)."""

        def forward(self, x, start_pos):
            bsz, seqlen, _ = x.size()
            freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
            win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
            if self.compress_ratio and self.compressor.kv_cache is None:
                self.compressor.kv_cache = self.kv_cache[:, win:]
                self.compressor.freqs_cis = self.freqs_cis
            qr = q = self.q_norm(self.wq_a(x))
            gate = F.linear(qr, self.o_gate)  # ERNIE: `gate = self.o_gate(qr)`
            q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
            q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
            M.apply_rotary_emb(q[..., -rd:], freqs_cis)
            kv = self.kv_norm(self.wkv(x))
            M.apply_rotary_emb(kv[..., -rd:], freqs_cis)
            topk_idxs = M.get_window_topk_idxs(win, bsz, seqlen, start_pos)
            if self.compress_ratio:
                offset = kv.size(1) if start_pos == 0 else win
                topk_idxs = torch.cat([topk_idxs, M.get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)], dim=-1)
            topk_idxs = topk_idxs.int()
            sparse_attn = sys.modules["kernel"].sparse_attn
            if start_pos == 0:
                if seqlen <= win:
                    self.kv_cache[:bsz, :seqlen] = kv
                else:
                    cutoff = seqlen % win
                    self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
                if self.compress_ratio:
                    if (kv_compress := self.compressor(x, start_pos)) is not None:
                        kv = torch.cat([kv, kv_compress], dim=1)
                o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
            else:
                self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
                if self.compress_ratio:
                    self.compressor(x, start_pos)
                o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
            o = torch.einsum("hg,bsgd->bshd", self.vha_m, o)  # ERNIE: `o = self.vha(o)`, before the inverse RoPE
            M.apply_rotary_emb(o[..., -rd:], freqs_cis, True)
            o = o.reshape(bsz, seqlen, self.n_local_groups, -1)
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
            return self.wo_b(o.flatten(2) * torch.sigmoid(gate))  # ERNIE: `wo_b(z * sigmoid(gate))`

    return ErnieHCA


def rms(x, w, eps):
    return w * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps))


def rope(x, pos, theta, neox):
    """vLLM `RotaryEmbedding`: the whole last dimension rotated (plain RoPE, no scaling)."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    ang = pos.float()[:, None] * inv[None]
    cos, sin = ang.cos(), ang.sin()
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    if neox:
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2)


def situ(g, u, beta, lin_beta):
    """vLLM `SituAndMul.forward_native`."""
    return beta * torch.tanh(g / beta) * torch.sigmoid(g) * (lin_beta * torch.tanh(u / lin_beta))


class Ref:
    def __init__(self, d, model_py, max_seq):
        from safetensors.torch import load_file

        self.cfg = c = json.load(open(os.path.join(d, "config.json")))
        sd = {}
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".safetensors"):
                sd.update(load_file(os.path.join(d, fn)))
        self.sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
        # FP8 matrices: times their block scales (`dequantize_block_fp8`).
        for k in [k for k in self.sd if k.endswith("_scale_inv")]:
            s = self.sd.pop(k)
            wn = k.removesuffix("_scale_inv")
            w = self.sd[wn]
            bo, bi = (c.get("quantization_config") or {}).get("weight_block_size", [128, 128])
            self.sd[wn] = w * s.repeat_interleave(bo, 0)[: w.shape[0]].repeat_interleave(bi, 1)[:, : w.shape[1]]
        self.M = M = ds4(model_py)
        self.n = c["num_hidden_layers"]
        self.eps, self.hc, self.D = c["rms_norm_eps"], c["hc_mult"], c["hidden_size"]
        rs = c.get("rope_scaling") or {}
        args = M.ModelArgs(
            max_batch_size=1, max_seq_len=max_seq, dim=c["hidden_size"], n_heads=c["num_attention_heads"],
            q_lora_rank=c["q_lora_rank"], head_dim=c["head_dim"], rope_head_dim=c["qk_rope_head_dim"],
            norm_eps=c["rms_norm_eps"], o_groups=c["o_groups"], o_lora_rank=c["o_lora_rank"],
            window_size=c["sliding_window"], compress_ratios=tuple(max(r, 0) for r in c["compress_ratios"]),
            compress_rope_theta=c["compress_rope_theta"], original_seq_len=rs.get("original_max_position_embeddings", 0),
            rope_theta=c["rope_theta"], rope_factor=rs.get("factor", 1), beta_fast=rs.get("beta_fast", 32),
            beta_slow=rs.get("beta_slow", 1),
        )
        HCA = hca_class(M)
        self.hca, self.dsa = {}, {}
        for l in range(self.n + c.get("num_nextn_predict_layers", 0)):
            if c["compress_ratios"][l] <= 0:
                self.dsa[l] = {"kvc": [], "kpe": [], "ik": []}
                continue
            assert l < self.n, "the MTP layer is an MLA layer"
            a = HCA(l, args).float()
            p = f"model.layers.{l}.self_attn."
            # `ernie5_5_mapper` (model.py).
            names = {
                "wq_a.weight": "linear_q_down_proj.weight", "q_norm.weight": "q_layernorm.weight",
                "wq_b.weight": "linear_q_up_proj.weight", "wkv.weight": "linear_kv_proj.weight",
                "kv_norm.weight": "kv_layernorm.weight", "wo_a.weight": "linear_o_group_proj",
                "wo_b.weight": "o_proj.weight", "attn_sink": "core_attention.attn_sink",
                "compressor.ape": "core_attention.compressor.ape",
                "compressor.wkv.weight": "core_attention.compressor.linear_wkv.weight",
                "compressor.wgate.weight": "core_attention.compressor.linear_wgate.weight",
                "compressor.norm.weight": "core_attention.compressor.norm.weight",
            }
            missing, unexpected = a.load_state_dict({k: self.sd[p + v] for k, v in names.items()}, strict=False)
            assert not missing and not unexpected, (missing, unexpected)
            a.o_gate = self.sd[p + "gate_proj.weight"]
            a.vha_m = self.vha(p)
            self.hca[l] = a

    def vha(self, p):
        u, v = self.sd[p + "vha_postmix_U"], self.sd[p + "vha_postmix_V"]
        return torch.eye(u.shape[0]) + v @ u.t()  # `VHAPostMix.fold`

    # --- hyper-connections (`mhc.py`, `_mhc_pre_ref` of the plugin's tests) ---
    def hc_pre(self, x, p):
        hc = self.hc
        fn = self.sd[p + "mapping_proj.weight"].t()  # stored [in, out]
        scale = torch.cat([self.sd[p + f"alpha_{a}"] for a in ("pre", "post", "res")])
        base = self.sd[p + "bias"]
        flat = x.flatten(1)
        mixes = (flat @ fn.t()) / (flat.square().mean(-1, keepdim=True).sqrt() + self.eps)
        he = self.cfg["hc_eps"]
        pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + he
        post = torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc]) * 2.0  # `hc_post_alpha`
        comb = mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]
        comb = torch.softmax(comb.view(-1, hc, hc), dim=-1) + he
        comb = comb / (comb.sum(dim=-2, keepdim=True) + he)
        for _ in range(self.cfg["hc_sinkhorn_iters"] - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + he)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + he)
        return (pre.unsqueeze(-1) * x).sum(dim=1), post, comb

    @staticmethod
    def hc_post(o, residual, post, comb):
        # vLLM `mhc_post_torch`.
        return torch.einsum("tij,tih->tjh", comb, residual) + post.unsqueeze(-1) * o.unsqueeze(-2)

    def hc_head(self, x, fn, scale, base):
        # vLLM `hc_head_fused_kernel_tilelang` (DeepSeek's `ParallelHead.hc_head`).
        flat = x.flatten(1)
        mixes = (flat @ fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        pre = torch.sigmoid(mixes * scale + base) + self.cfg["hc_eps"]
        return (pre.unsqueeze(-1) * x).sum(dim=1)

    # --- MLA with the DSA indexer (`Ernie5_5DSAAttention`, `Ernie5_5Indexer`, `merge_swa_topk`) ---
    def dsa_attn(self, l, x, start):
        c, sd, st = self.cfg, self.sd, self.dsa[l]
        p = f"model.layers.{l}.self_attn."
        nh, ql, r = c["hybrid_mla_num_attention_heads"], c["hybrid_mla_q_lora_rank"], c["hybrid_mla_kv_lora_rank"]
        nope, rd, vd = c["hybrid_mla_qk_nope_head_dim"], c["hybrid_mla_qk_rope_head_dim"], c["hybrid_mla_v_head_dim"]
        ni, di, topk, win = c["index_n_heads"], c["index_head_dim"], c["index_topk"], c["sliding_window"]
        theta = c["rope_theta"]
        T = x.shape[0]
        pos = torch.arange(start, start + T)
        q_c = rms(x @ sd[p + "q_a_proj.weight"].t(), sd[p + "q_a_layernorm.weight"], self.eps)
        kv = x @ sd[p + "kv_a_proj_with_mqa.weight"].t()
        gate = q_c @ sd[p + "gate_proj.weight"].t()
        q = (q_c @ sd[p + "q_b_proj.weight"].t()).view(T, nh, nope + rd)
        q_nope, q_pe = q[..., :nope], rope(q[..., nope:], pos, theta, False)
        st["kvc"].append(rms(kv[:, :r], sd[p + "kv_a_layernorm.weight"], self.eps))
        st["kpe"].append(rope(kv[:, r:], pos, theta, False))
        # Indexer: queries from the q latent, keys `k_norm(wk(x))`, NEOX RoPE on the first dims.
        ip = p + "core_attention.indexer."
        iq = (q_c @ sd[ip + "wq_b.weight"].t()).view(T, ni, di)
        ik = F.layer_norm(x @ sd[ip + "wk.weight"].t(), (di,), sd[ip + "k_norm.weight"], sd[ip + "k_norm.bias"], 1e-6)
        iw = (x @ sd[ip + "weights_proj.weight"].t()) * di ** -0.5 * ni ** -0.5
        iq = torch.cat([rope(iq[..., :rd], pos, theta, True), iq[..., rd:]], dim=-1)
        st["ik"].append(torch.cat([rope(ik[..., :rd], pos, theta, True), ik[..., rd:]], dim=-1))
        kvc, kpe, iks = torch.cat(st["kvc"]), torch.cat(st["kpe"]), torch.cat(st["ik"])
        S = kvc.shape[0]
        kb = sd[p + "kv_b_proj.weight"].view(nh, nope + vd, r)
        k_nope = torch.einsum("sr,hnr->shn", kvc, kb[:, :nope])
        val = torch.einsum("sr,hvr->shv", kvc, kb[:, nope:])
        scores = (torch.einsum("thn,shn->ths", q_nope, k_nope) + torch.einsum("thd,sd->ths", q_pe, kpe)) * (nope + rd) ** -0.5
        allowed = torch.zeros(T, S, dtype=torch.bool)
        chosen = []
        for t in range(T):
            seq = start + t + 1
            allowed[t, max(seq - win, 0):seq] = True  # the window, the query included
            older = max(seq - win, 0)  # the indexer scores only the tokens older than the window
            if older <= topk:
                allowed[t, :older] = True
            else:
                s = (torch.relu(torch.einsum("hd,sd->hs", iq[t], iks[:older])) * iw[t][:, None]).sum(0)
                # Ties (keys no head scores above zero) to the older token, as a stable top-k.
                top = torch.sort(s, descending=True, stable=True).indices[:topk]
                allowed[t, top] = True
                chosen.append((start + t, torch.sort(top).values))
        self.last_topk = chosen
        scores = scores.masked_fill(~allowed[:, None, :], float("-inf"))
        sink = sd[p + "core_attention.softmax_offset"].view(1, nh, 1).expand(T, nh, 1)
        pr = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
        o = torch.einsum("ths,shv->thv", pr, val)
        o = torch.einsum("hg,tgv->thv", self.vha(p), o)
        return (o.flatten(1) * torch.sigmoid(gate)) @ sd[p + "o_proj.weight"].t()

    # --- FFN (`moe.py`) ---
    def mlp(self, x, p, names):
        c, sd = self.cfg, self.sd
        g, u = x @ sd[p + names[0]].t(), x @ sd[p + names[1]].t()
        return situ(g, u, c["activation_situ_beta"], c["activation_situ_linear_beta"]) @ sd[p + names[2]].t()

    def moe(self, x, p):
        c, sd = self.cfg, self.sd
        ne, k = c["n_routed_experts"], c["num_experts_per_tok"]
        scores = torch.sigmoid(x @ sd[p + "gate.weight"].t()) + torch.sigmoid(x @ sd[p + "gate.weight_1"].t())
        ids = torch.topk(scores + sd[p + "e_score_correction_bias"], k, dim=-1).indices  # `ernie5_5_routing`
        w = scores.gather(1, ids)
        if c.get("norm_topk_prob", True):
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-20)
        w = w * sd[p + "gate.routed_scaling_factor_param"][ids]
        lat = x @ sd[p + "fc1_latent_proj.weight"].t()
        out = torch.zeros_like(lat)
        for t in range(x.shape[0]):
            for j in range(k):
                e = int(ids[t, j])
                out[t] += w[t, j] * self.mlp(lat[t:t + 1], f"{p}experts.{e}.", ("w1.weight", "w3.weight", "w2.weight"))[0]
        routed = rms(out, sd[p + "latent_norm.weight"], self.eps) @ sd[p + "fc2_latent_proj.weight"].t()
        return routed + self.mlp(x, p + "shared_experts.", ("w1.weight", "w3.weight", "w2.weight"))

    def block(self, l, x, start):
        """`Ernie5_5DecoderLayer.forward` plus the closing `mhc_post`: x [T, hc, D]."""
        sd, p = self.sd, f"model.layers.{l}."
        hp = p + ("transformer_layer." if l >= self.n else "")
        xin, post, comb = self.hc_pre(x, hp + "self_attention_hyper_connection.")
        xin = rms(xin, sd[p + "input_layernorm.weight"], self.eps)
        if l in self.hca:
            a = self.hca[l](xin.unsqueeze(0), start)[0]
        else:
            a = self.dsa_attn(l, xin, start)
        x = self.hc_post(a, x, post, comb)
        xin, post, comb = self.hc_pre(x, hp + "mlp_hyper_connection.")
        xin = rms(xin, sd[p + "post_attention_layernorm.weight"], self.eps)
        if l < self.cfg["first_k_dense_replace"]:
            f = self.mlp(xin, p + "mlp.", ("gate_proj.weight", "up_proj.weight", "down_proj.weight"))
        else:
            f = self.moe(xin, p + "block_sparse_moe.")
        return self.hc_post(f, x, post, comb)

    def head(self, x, fn, scale, base, norm):
        sd = self.sd
        h = rms(self.hc_head(x, fn, scale, base), norm, self.eps)
        logits = h @ sd["lm_head.weight"].t()
        r, t = sd["model.lm_head.multimax_ranges"], sd["model.lm_head.multimax_ts"]
        return (logits + t[0] * (r[0] - logits).relu() + t[1] * (logits - r[1]).relu()
                + t[2] * (r[2] - logits).relu().square() + t[3] * (logits - r[3]).relu().square())

    def forward(self, ids, start):
        """Logits [T, V] and the residual before the mHC head [T, hc, D] (what the MTP layer reads)."""
        sd = self.sd
        x = sd["model.embed_tokens.weight"][ids].unsqueeze(1).repeat(1, self.hc, 1)
        for l in range(self.n):
            x = self.block(l, x, start)
        p = "model.mhc_contract."
        return self.head(x, sd[p + "hc_head_fn"], sd[p + "hc_head_scale"], sd[p + "hc_head_base"], sd["model.norm.weight"]), x

    def mtp(self, h, ids, start):
        """`Ernie5_5MTPLayer.forward` and `compute_logits`: h [T, hc, D] at positions start.., ids the
        tokens after them."""
        sd, l = self.sd, self.n
        p = f"model.layers.{l}."
        e = rms(sd["model.embed_tokens.weight"][ids], sd[p + "enorm.weight"], self.eps) @ sd[p + "e_proj.weight"].t()
        x = rms(h, sd[p + "hnorm.weight"], self.eps) @ sd[p + "h_proj.weight"].t() + e.unsqueeze(1)
        x = self.block(l, x, start)
        return self.head(x, sd[p + "hc_head_fn"], sd[p + "hc_head_scale"], sd[p + "hc_head_base"], sd[p + "shared_head.norm.weight"]), x


def main():
    from safetensors.torch import save_file

    a = sys.argv
    d, model_py, out, n, nd = a[1], a[2], a[3], int(a[4]), int(a[5])
    seed = int(a[6]) if len(a) > 6 and not a[6].startswith("--") else 0
    chunk = int(a[a.index("--chunk") + 1]) if "--chunk" in a else n
    drafts = int(a[a.index("--drafts") + 1]) if "--drafts" in a else 3
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float32)
    ref = Ref(d, model_py, max(1024, n + nd + 8))
    g = torch.Generator().manual_seed(seed)
    ids = torch.cat([torch.tensor([2]), torch.randint(1300, 200000, (n - 1,), generator=g)])
    res = {"ids": ids.to(torch.int64)}
    with torch.no_grad():
        lg, hs = ref.forward(ids[:chunk], 0)
        logits, hidden = [lg], [hs]
        for p in range(chunk, n):
            lg, hs = ref.forward(ids[p:p + 1], p)
            logits.append(lg)
            hidden.append(hs)
        logits, hidden = torch.cat(logits), torch.cat(hidden)
        res["logits"] = logits
        mtp = ref.cfg.get("num_nextn_predict_layers", 0) > 0
        if mtp:
            res["mtp_logits"], _ = ref.mtp(hidden[:-1], ids[1:], 0)
            # A round of drafts after the prompt: the layer on its own output and its own drafts.
            keep = copy.deepcopy(ref.dsa[ref.n])
            tok, h, dl, dr = logits[-1].argmax().view(1), hidden[-1:], [], []
            for j in range(drafts):
                lgj, h = ref.mtp(h, tok, n - 1 + j)
                tok = lgj[0].argmax().view(1)
                dl.append(lgj[0])
                dr.append(int(tok))
            ref.dsa[ref.n] = keep
            res["draft"], res["draft_logits"] = torch.tensor(dr, dtype=torch.int64), torch.stack(dl)
        nxt, gen, steps = logits[-1].argmax(), [], []
        for s in range(nd):
            gen.append(int(nxt))
            lg, _ = ref.forward(nxt.view(1), n + s)
            steps.append(lg[0])
            nxt = lg[0].argmax()
        res["gen"] = torch.tensor(gen, dtype=torch.int64)
        if steps:
            res["decode_logits"] = torch.stack(steps)
    save_file({k: v.contiguous() for k, v in res.items()}, out)
    print("prompt", n, "gen", gen, "drafts", res.get("draft"))
    print("top5 last", torch.topk(logits[-1], 5))
    print("logit std", float(logits.std()), "distinct top-1 over the prompt", len(set(logits.argmax(-1).tolist())))


if __name__ == "__main__":
    main()
