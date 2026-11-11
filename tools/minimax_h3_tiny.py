#!/usr/bin/env python3
"""Tiny random MiniMax-H3-Image components (diffusers' MiniMaxH3ImageTransformer2DModel and
AutoencoderKLMiniMaxH3Image) and their outputs, for examples/h3_check.rs.

usage: minimax_h3_tiny.py OUT_DIR
Writes OUT_DIR/dit, OUT_DIR/vae (save_pretrained), the inputs and the reference outputs (*.bin,
little-endian f32) and OUT_DIR/case.json. Then: cargo run --release --example h3_check -- OUT_DIR

The DiT case packs `[text | reference rows | generated rows]` with two timesteps (the reference
rows at 0.999, as an edit holds them), so every AdaLN set and the RoPE grid are exercised; one
forward per output head. The VAE case decodes latents large enough for 2 x 2 tiles.
"""
import json, os, sys

import numpy as np
import torch
from diffusers import AutoencoderKLMiniMaxH3Image
from diffusers.models.transformers.transformer_minimax_h3_image import MiniMaxH3ImageTransformer2DModel
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import build_image_packed_sequence, patchify_video_latents

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)


def randomize(m, scale=0.1):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n and n.endswith("weight"):
                p.copy_(1.0 + torch.randn_like(p) * 0.2)
            else:
                p.copy_(torch.randn_like(p) * scale)
            p.copy_(p.to(torch.bfloat16).float())


def save(name, t):
    np.asarray(t, dtype=np.float32).tofile(os.path.join(out, name))


# ---- DiT -------------------------------------------------------------------------------------
heads = 3
dit = MiniMaxH3ImageTransformer2DModel(
    num_attention_heads=2, attention_head_dim=32, hidden_size=48, num_layers=2, num_refiner_layers=1,
    ffn_dim=64, in_channels=4, patch_size=(1, 2, 2), text_dim=40, freq_dim=32, time_embed_hidden_dim=48,
    time_embed_dim=24, rope_freq_dim=4, num_pdd_heads=heads,
).float().eval()
randomize(dit)
dit.save_pretrained(os.path.join(out, "dit"))
n_text = 7
lh, lw = 8, 12
text = torch.randn(n_text, 40)
tags_text = torch.ones(n_text, dtype=torch.long)
ref_grid = (4, 6)
pos, tags, img_idx, txt_idx, n_ref = build_image_packed_sequence(tags_text, [ref_grid], lh, lw, (1, 2, 2), 0)
ref_lat = torch.randn(1, 4, 1, *ref_grid)
gen_lat = torch.randn(1, 4, 1, lh, lw)
rows = torch.cat([patchify_video_latents(ref_lat, (1, 2, 2)), patchify_video_latents(gen_lat, (1, 2, 2))])
row_t = torch.full((pos.shape[0],), 0.3)
row_t[img_idx[:n_ref]] = 0.999
uniq, tidx = torch.unique(row_t, sorted=True, return_inverse=True)
with torch.no_grad():
    for h in range(heads):
        o = dit(hidden_states=rows[None], encoder_hidden_states=text[None], timestep=uniq, timestep_indices=tidx,
                token_tags=tags, position_ids=pos.float(), image_indices=img_idx, text_indices=txt_idx,
                pdd_step_index=h, return_dict=False)[0]
        save(f"dit_out_{h}.bin", o[0].numpy())
save("text.bin", text.numpy())
save("image.bin", rows.numpy())
save("positions.bin", pos.float().numpy())
save("tags.bin", tags.float().numpy())
save("tidx.bin", tidx.float().numpy())
save("timesteps.bin", uniq.numpy())

# ---- VAE decoder ------------------------------------------------------------------------------
C = 4
vae = AutoencoderKLMiniMaxH3Image(
    latent_channels=C, block_out_channels=(8, 8, 8, 8, 8, 8), norm_num_groups=4, decoder_num_layers=2,
    decoder_num_attention_heads=2, decoder_attention_head_dim=16, decoder_num_register_tokens=2,
    latents_mean=tuple(float(v) for v in torch.randn(C) * 0.3), latents_std=tuple(float(v) for v in torch.rand(C) + 0.5),
).float().eval()
randomize(vae, 0.05)
vae.save_pretrained(os.path.join(out, "vae"))
vh, vw = 24, 20  # 384 x 320 pixels: 2 x 2 tiles of 256 overlapping by at least 64
lat = torch.randn(C, vh, vw)
mean = torch.tensor(vae.config.latents_mean).view(-1, 1, 1)
std = torch.tensor(vae.config.latents_std).view(-1, 1, 1)
with torch.no_grad():
    img = vae.decode((lat * std + mean)[None], return_dict=False)[0][0]
save("latents.bin", lat.numpy())
save("vae_out.bin", img.numpy())
json.dump({"heads": heads, "lh": vh, "lw": vw}, open(os.path.join(out, "case.json"), "w"))
print("dit out", o.shape, "vae out", img.shape, "tiling", vae.use_tiling)

# ---- the release's layout (`--original`): the inverse of diffusers' conversion ------------------
if "--original" in sys.argv:
    from safetensors.torch import load_file, save_file

    def interleave(q, k, v, heads):
        """`[q_all; k_all; v_all]` -> per head `[q k v]` rows (the release's fused layout)."""
        hd = q.shape[0] // heads
        parts = []
        for h in range(heads):
            parts += [q[h * hd:(h + 1) * hd], k[h * hd:(h + 1) * hd], v[h * hd:(h + 1) * hd]]
        return torch.cat(parts)

    def swap_halves(w):
        a, b = w.chunk(2, dim=0)
        return torch.cat([b, a])

    # DiT
    sd = load_file(os.path.join(out, "dit", "diffusion_pytorch_model.safetensors"))
    o = {}
    nh = dit.config.num_attention_heads
    for prefix_d, prefix_o, n in [("token_refiner.refiner_blocks", "token_refiner.blocks", 1), ("transformer_blocks", "blocks", 2)]:
        for l in range(n):
            d, r = f"{prefix_d}.{l}.", f"{prefix_o}.{l}."
            o[r + "attn.qkv_proj.weight"] = interleave(sd[d + "attn.to_q.weight"], sd[d + "attn.to_k.weight"], sd[d + "attn.to_v.weight"], nh)
            o[r + "attn.out_proj.weight"] = sd[d + "attn.to_out.0.weight"]
            o[r + "attn.q_norm.weight"] = sd[d + "attn.norm_q.weight"]
            o[r + "attn.k_norm.weight"] = sd[d + "attn.norm_k.weight"]
            o[r + "mlp.fc1.weight"] = swap_halves(sd[d + "ff.net.0.proj.weight"])
            o[r + "mlp.fc2.weight"] = sd[d + "ff.net.2.weight"]
            o[r + "norm1.weight"], o[r + "norm2.weight"] = sd[d + "norm1.weight"], sd[d + "norm2.weight"]
            if d + "adaln_proj.linear.weight" in sd:
                o[r + "adaln_proj.linear.weight"] = sd[d + "adaln_proj.linear.weight"]
                o[r + "adaln_proj.linear.bias"] = sd[d + "adaln_proj.linear.bias"]
    ren = {"proj_in": "video_patch_proj", "context_embedder": "condition_proj", "time_embedder.linear_1": "time_embedder.proj_in",
           "time_embedder.linear_2": "time_embedder.proj_out", "norm_out.linear": "final_layer.adaln_proj.linear"}
    for a, b in ren.items():
        o[b + ".weight"], o[b + ".bias"] = sd[a + ".weight"], sd[a + ".bias"]
    o["final_layer.norm.weight"] = sd["norm_out.norm.weight"]
    o["token_refiner.final_norm.weight"] = sd["token_refiner.final_norm.weight"]
    os.makedirs(os.path.join(out, "dit_orig"), exist_ok=True)
    save_file({k: v.contiguous() for k, v in o.items()}, os.path.join(out, "dit_orig", "model.safetensors"))
    c = dict(dit.config)
    json.dump({"hidden_size": c["hidden_size"], "num_layers": c["num_layers"], "token_refiner_num_layers": c["num_refiner_layers"],
               "num_attention_heads": nh, "attention_head_dim": c["attention_head_dim"], "ffn_hidden_size": c["ffn_dim"],
               "latents_dim": c["in_channels"], "patch_size": list(c["patch_size"]), "text_dim": c["text_dim"],
               "timestep_input_dim": c["freq_dim"], "time_embed_hidden_size": c["time_embed_hidden_dim"],
               "time_embed_dim": c["time_embed_dim"], "rope_inv_freq_len": c["rope_freq_dim"]},
              open(os.path.join(out, "dit_orig", "config.json"), "w"))
    # The PDD heads as the release keeps them.
    w = torch.stack([sd[f"pdd_proj_out.{h}.weight"] for h in range(heads)])
    b = torch.stack([sd[f"pdd_proj_out.{h}.bias"] for h in range(heads)])
    save_file({"video_out.weight": w.contiguous(), "video_out.bias": b.contiguous()}, os.path.join(out, "pdd_fused_heads.safetensors"))
    # VAE decoder
    sd = load_file(os.path.join(out, "vae", "diffusion_pytorch_model.safetensors"))
    o = {k: v for k, v in sd.items() if k.startswith(("post_quant_conv", "decoder.register_tokens", "decoder.norm_out", "decoder.proj_out"))}
    o["decoder.x_embedder.weight"], o["decoder.x_embedder.bias"] = sd["decoder.proj_in.weight"], sd["decoder.proj_in.bias"]
    vh = vae.config.decoder_num_attention_heads
    for l in range(vae.config.decoder_num_layers):
        d = f"decoder.transformer_blocks.{l}."
        for s in ("weight", "bias"):
            o[d + f"attn.to_qkv.{s}"] = interleave(sd[d + f"attn.to_q.{s}"], sd[d + f"attn.to_k.{s}"], sd[d + f"attn.to_v.{s}"], vh)
            o[d + f"attn.to_out.{s}"] = sd[d + f"attn.to_out.0.{s}"]
            o[d + f"ff.w1.{s}"] = swap_halves(sd[d + f"ff.net.0.proj.{s}"])
            o[d + f"ff.w2.{s}"] = sd[d + f"ff.net.2.{s}"]
        for n in ("norm1.weight", "norm2.weight", "scale1", "scale2"):
            o[d + n] = sd[d + n]
    os.makedirs(os.path.join(out, "vae_orig"), exist_ok=True)
    save_file({k: v.contiguous() for k, v in o.items()}, os.path.join(out, "vae_orig", "diffusion_pytorch_model.safetensors"))
    vc = dict(vae.config)
    json.dump({"latent_channels": C, "latents_mean": list(vc["latents_mean"]), "latents_std": list(vc["latents_std"]),
               "vae_tile_size": 256, "vae_tile_overlap_min": 64,
               "vit_decoder_kwargs": {"num_layers": vc["decoder_num_layers"], "heads": vh, "dim_head": vc["decoder_attention_head_dim"],
                                      "rope_theta": 100.0, "rope_dim_ratio": 0.5}},
              open(os.path.join(out, "vae_orig", "config.json"), "w"))
    print("wrote the release's layout: dit_orig, vae_orig, pdd_fused_heads.safetensors")
