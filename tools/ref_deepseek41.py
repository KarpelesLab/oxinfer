#!/usr/bin/env python3
# Reference outputs of a (tiny) DeepSeek V4.1 checkpoint from DeepSeek's own inference code
# (`inference/model.py`, `engram.py`, `vision.py`, `image_processor.py` of
# deepseek-ai/DeepSeek-V4.1-Flash), run in float32 on the CPU: its TileLang kernels are replaced
# by plain PyTorch equivalents (sparse attention with sinks, the Sinkhorn split of the
# hyper-connections), the quantization simulations (FP8 of the window keys and of the attention
# output, FP4 of the compressed keys and of the indexer) are left out, and the engram tables are
# read as floats instead of FP8 rows with their scales.
#   prefill: logits at every position; decode: greedy steps (model.py's own decode path, one
#   token at a time), logits at each step.
# usage: ref_deepseek41.py MODEL_DIR INFERENCE_DIR OUT.safetensors N_PROMPT N_DECODE [SEED]
#            [--chunk N] [--image H W AT]
#   INFERENCE_DIR: the checkpoint's inference/ folder. --chunk: the prompt as one prefill of N
#   tokens followed by one-token steps (model.py only prefills from position 0).
#   --image: an image span of a random H x W picture (pixels) starting at prompt position AT, in
#   the prefill chunk; its pixels (`image`, [H, W, 3] in 0..1) and the span's embeddings
#   (`image_embeds`) are written too.
import importlib.util, json, os, sys, types

import torch
import torch.nn.functional as F
from torch import nn


def reference(inference_dir):
    """DeepSeek's `inference/model.py` as a module, its kernels replaced by PyTorch."""
    kernel = types.ModuleType("kernel")

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        assert inplace, "only the in-place (simulation) form is reached with float weights"
        return x

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
        assert inplace
        return x

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        # q [b, m, h, d], kv [b, n, d], topk_idxs [b, m, k] (-1 = none) -> [b, m, h, d]
        b, m, h, dd = q.shape
        idx = topk_idxs.long()
        valid = idx >= 0
        g = torch.gather(kv.unsqueeze(1).expand(b, m, -1, dd), 2, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, dd))
        s = torch.einsum("bmhd,bmkd->bmhk", q.float(), g.float()) * softmax_scale
        s = s.masked_fill(~valid.unsqueeze(2), float("-inf"))
        sink = attn_sink.float().view(1, 1, h, 1).expand(b, m, h, 1)
        p = torch.softmax(torch.cat([s, sink], dim=-1), dim=-1)[..., :-1]
        return torch.einsum("bmhk,bmkd->bmhd", p, g.float()).to(q.dtype)

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        hc = hc_mult
        pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
        comb = mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
        comb = comb.unflatten(-1, (hc, hc))
        comb = torch.softmax(comb, dim=-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    def unused(*a, **k):
        raise RuntimeError("quantized GEMM reached with float weights")

    kernel.act_quant, kernel.fp4_act_quant, kernel.sparse_attn = act_quant, fp4_act_quant, sparse_attn
    kernel.hc_split_sinkhorn, kernel.fp8_gemm, kernel.fp4_gemm = hc_split_sinkhorn, unused, unused
    sys.modules["kernel"] = kernel
    # The dtypes model.py names at import (their weights are never built here).
    for name in ("float8_e4m3fn", "float8_e8m0fnu", "float4_e2m1fn_x2"):
        if not hasattr(torch, name):
            setattr(torch, name, name)
    sys.path.insert(0, inference_dir)  # engram.py, vision.py, image_processor.py
    spec = importlib.util.spec_from_file_location("dsv41_model", os.path.join(inference_dir, "model.py"))
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)

    class FloatEngramEmbedding(nn.Module):
        """The n-gram table as floats (the release keeps FP8 rows and dequantizes on lookup)."""

        def __init__(self, num_embeddings, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(num_embeddings, dim))

        def forward(self, indices):
            return F.embedding(indices, self.weight)

    M.ParallelEngramEmbedding = FloatEngramEmbedding
    return M


def model_args(M, cfg, max_seq_len):
    """`ModelArgs` from the official (nested) config.json."""
    t = cfg["text_config"]
    rs = t.get("rope_scaling") or {}
    v = cfg.get("vision_config") or {}
    return M.ModelArgs(
        max_batch_size=1, max_seq_len=max_seq_len, temperature=0, dtype="bf16", expert_dtype=None,
        vocab_size=t["vocab_size"], dim=t["hidden_size"], moe_inter_dim=t["moe_intermediate_size"],
        n_layers=t["num_hidden_layers"], n_mtp_layers=0, n_heads=t["num_attention_heads"],
        n_routed_experts=t["n_routed_experts"], n_shared_experts=t["n_shared_experts"],
        n_activated_experts=t["num_experts_per_tok"], score_func=t["scoring_func"],
        norm_topk_prob=t["norm_topk_prob"], route_scale=t["routed_scaling_factor"], swiglu_limit=t["swiglu_limit"],
        q_lora_rank=t["q_lora_rank"], head_dim=t["head_dim"], rope_head_dim=t["qk_rope_head_dim"],
        norm_eps=t["rms_norm_eps"], o_groups=t["o_groups"], o_lora_rank=t["o_lora_rank"],
        window_size=t["sliding_window"], compress_ratios=tuple(t["compress_ratios"]),
        kv_source_layers=tuple(t["kv_source_layer_ids"]), index_source_layers=tuple(t["index_source_layer_ids"]),
        compress_rope_theta=t["compress_rope_theta"], original_seq_len=rs.get("original_max_position_embeddings", 0),
        rope_theta=t["rope_theta"], rope_factor=rs.get("factor", 1), beta_fast=rs.get("beta_fast", 32),
        beta_slow=rs.get("beta_slow", 1), index_n_heads=t["index_n_heads"], index_head_dim=t["index_head_dim"],
        index_topk=t["index_topk"], candidate_source_layer=t.get("candidate_source_layer_id", -1),
        candidate_topk_blocks=t.get("candidate_topk_blocks", 0), candidate_block_size=t.get("candidate_block_size", 0),
        hc_mult=t["hc_mult"], hc_sinkhorn_iters=t["hc_sinkhorn_iters"], hc_eps=t["hc_eps"],
        engram_layer_ids=tuple(t.get("engram_layer_ids", ())),
        engram_num_embeddings=tuple(t.get("engram_num_embeddings", ())),
        engram_max_ngram_size=t.get("engram_max_ngram_size", 1), engram_vocab_size=t.get("engram_vocab_size", 0),
        engram_n_heads=t.get("engram_n_heads", 0), engram_head_dim=t.get("engram_head_dim", 0),
        engram_pad_id=t.get("engram_pad_token_id", 2),
        engram_compressed_vocab_size=t.get("engram_compressed_vocab_size", 0),
        vision_n_layers=v.get("num_hidden_layers", 0), vision_dim=v.get("hidden_size", 1024),
        vision_n_heads=v.get("num_attention_heads", 16), vision_inter_dim=v.get("intermediate_size", 2816),
        vision_patch_size=v.get("patch_size", 14), vision_rope_theta=v.get("rope_theta", 10000.0),
        vision_downsample_ratio=v.get("downsample_ratio", 3), vision_max_n_token=v.get("max_image_tokens", 1024),
        vision_min_pixels=v.get("min_pixels", 0), vision_max_wh_ratio=v.get("max_wh_ratio"),
        image_token_id=cfg.get("image_token_id", 129264),
    )


def tokenizer(d):
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=os.path.join(d, "tokenizer.json"))


def main():
    from safetensors.torch import load_file, save_file

    d, inference, out, n, nd = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    seed = int(sys.argv[6]) if len(sys.argv) > 6 and not sys.argv[6].startswith("--") else 0
    chunk = int(sys.argv[sys.argv.index("--chunk") + 1]) if "--chunk" in sys.argv else n
    image = [int(x) for x in sys.argv[sys.argv.index("--image") + 1:][:3]] if "--image" in sys.argv else None
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float32)
    M = reference(inference)
    cfg = json.load(open(f"{d}/config.json"))
    args = model_args(M, cfg, max(1024, n + nd + 8))
    model = M.Transformer(args, tokenizer(d))
    model.float()
    sd = load_file(f"{d}/model.safetensors")
    # (The hash inputs oxinfer reads from the checkpoint are derived here by engram.py.)
    extra = {k: sd.pop(k) for k in list(sd) if k.startswith("engram.")}
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    if model.engram_hash is not None:
        assert torch.equal(extra["engram.token_map"].long(), model.engram_hash.token_map), "token map"
        assert torch.equal(extra["engram.multipliers"], model.engram_hash.multipliers), "multipliers"

    g = torch.Generator().manual_seed(seed)
    ids = torch.cat([torch.tensor([0]), torch.randint(1000, 128000, (n - 1,), generator=g)]).unsqueeze(0)
    res = {}
    images = types_ = None
    if image is not None:
        import image_processor as IP

        hh, ww, at = image
        px = torch.rand(hh, ww, 3, generator=g)
        p = args.vision_patch_size
        n_llm_h, n_llm_w, bh, bw = IP.plan_image_grid(ww, hh, args)
        assert (bh, bw) == (hh, ww), f"pick a size the processor keeps: {hh}x{ww} -> {bh}x{bw}"
        n_vit_h, n_vit_w = bh // p, bw // p
        x = ((px.permute(2, 0, 1) - 0.5) / 0.5).float()
        patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
        tt = IP.image_token_types(n_llm_h, n_llm_w)
        assert at + tt.numel() <= chunk, "the image span must be in the prefill chunk"
        ids[0, at:at + tt.numel()] = args.image_token_id
        types_ = torch.full((1, n), IP.TEXT, dtype=torch.long)
        types_[0, at:at + tt.numel()] = tt
        images = [[IP.ImageInput(at, patches, n_vit_h, n_vit_w, tt)]]
        res["image"] = px.contiguous()
        res["image_at"] = torch.tensor([at, tt.numel(), n_llm_h, n_llm_w], dtype=torch.int64)
        with torch.no_grad():
            h = torch.zeros(1, n, args.dim)
            model.merge_image_embeddings(images, h)
            res["image_embeds"] = h[0, at:at + tt.numel()].float().contiguous()

    # model.py returns the last position's logits: run the head on every position of the prefill.
    head_all = {}

    def head_hook(mod, inp, outp):
        head_all["x"] = F.linear(inp[0].float(), mod.weight)

    model.head.register_forward_hook(head_hook)
    with torch.no_grad():
        logits = []
        model(ids[:, :chunk], 0, images=images, token_types=None if types_ is None else types_[:, :chunk])
        logits.append(head_all["x"][0])
        for p in range(chunk, n):
            model(ids[:, p:p + 1], p)
            logits.append(head_all["x"][0])
    res.update({"ids": ids[0].to(torch.int64), "logits": torch.cat(logits).float()})

    with torch.no_grad():
        nxt = res["logits"][-1].argmax()
        gen, steps = [], []
        for s in range(nd):
            gen.append(int(nxt))
            lg = model(nxt.view(1, 1), n + s)[1][0]
            steps.append(lg.float())
            nxt = lg.argmax()
    res["gen"] = torch.tensor(gen, dtype=torch.int64)
    if steps:
        res["decode_logits"] = torch.stack(steps)
    save_file({k: v.contiguous() for k, v in res.items()}, out)
    print("prompt", ids.shape[1], "gen", gen)
    print("top5 last", torch.topk(res["logits"][-1], 5))


if __name__ == "__main__":
    main()
