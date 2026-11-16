#!/usr/bin/env python3
"""The tiny MiniMax-H3 checkpoints of tools/minimax_h3_video_tiny.py in the forms of
Comfy-Org/MiniMax-H3 and lightx2v/Minimax-h3-Turbo, with diffusers references, for
examples/h3_video_check.rs.

usage: minimax_h3_video_comfy.py DIR        (DIR: the output of minimax_h3_video_tiny.py)

Writes into DIR/comfy:
  h3_bf16.safetensors               the transformer as ComfyUI's single file: MiniMax's tensor names,
                                    `qkv_proj` as [q; k; v], `fc1` as [gate; value], bf16 with the
                                    patch / time / output layers float32, `rope.inv_freq`, the
                                    configuration in the header
  release/                          the same in MiniMax's own layout (`qkv_proj` interleaved per head,
                                    config.json), which shares every tensor name with ComfyUI's
  h3_pruned_bf16.safetensors        ComfyUI's "pruned" form: no time embedder, `adaln_t_table`
                                    [1025, 8] (the principal components of silu(time_embedder(t)) at
                                    t = i / 1024) and float16 [.., 8] AdaLN projections, the curve's
                                    mean folded into their biases
  h3_pruned_fp8_scaled.safetensors  the same with the blocks' four matrices as float8_e4m3fn times a
                                    scalar `weight_scale` (`comfy_quant`, `input_scale` beside them)
  lora_comfyui.safetensors          a rank-4 LoRA in the turbo LoRAs' ComfyUI naming (fused qkv with a
                                    block-diagonal B, alpha x 3; fc1 over [gate; value]; `.alpha`)
  lora_diffusers.safetensors        the same LoRA as PEFT keys (alpha in the header)
  lora_adaln_comfyui.safetensors    with modules on the AdaLN projections too (turbo 8-step v1.2)
  video_vae_fp16.safetensors        the video VAE as ComfyUI's single file: float16, MiniMax's names,
                                    the configuration in the `minimax_h3_video_vae` header entry

and beside the tiny tool's references (`dit_v0.bin`, ...) the same outputs of diffusers for:
  pruned_*      the curve form (diffusers' model with the table and the [.., 8] projections)
  fp8_*         the curve form with the decoded FP8 weights
  lora_*        the LoRA merged into the weights (W += alpha / rank * B A)
  loraadaln_*   the LoRA with the AdaLN modules merged
  fp8lora_*     the LoRA merged into the decoded FP8 weights, curve form
  fp16_*        the VAE with its weights rounded to float16 (moments, decoded clips)
"""
import copy, json, os, sys

import numpy as np
import torch
from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep
from safetensors.torch import load_file, save_file

src = sys.argv[1]
out = os.path.join(src, "comfy")
os.makedirs(out, exist_ok=True)
torch.manual_seed(1)
case = json.load(open(os.path.join(src, "case.json")))


def load(name, shape):
    return torch.from_numpy(np.fromfile(os.path.join(src, name), dtype=np.float32).reshape(shape).copy())


def save(name, t):
    np.asarray(t, dtype=np.float32).tofile(os.path.join(src, name))


def bf16r(t):
    return t.to(torch.bfloat16).float()


# ---- the transformer in ComfyUI's layout ---------------------------------------------------------


def pretrained(cls, sub):
    """`from_pretrained` without accelerate: the configuration, then the weights."""
    m = cls.from_config(cls.load_config(os.path.join(src, sub)))
    m.load_state_dict(load_file(os.path.join(src, sub, "diffusion_pytorch_model.safetensors")))
    return m.float().eval()


dit = pretrained(MiniMaxH3Transformer3DModel, "transformer")
cfg = dit.config
inner, hd, ffn = cfg.num_attention_heads * cfg.attention_head_dim, cfg.attention_head_dim, cfg.ffn_dim
F32 = ("video_patch_proj.", "audio_patch_proj.", "time_embedder.", "final_layer.video_out.", "final_layer.audio_out.", "rope.")
TOP = {
    "proj_in": "video_patch_proj", "audio_proj_in": "audio_patch_proj", "context_embedder": "condition_proj",
    "time_embedder.linear_1": "time_embedder.proj_in", "time_embedder.linear_2": "time_embedder.proj_out",
    "norm_out.norm": "final_layer.norm", "norm_out.linear": "final_layer.adaln_proj.linear",
    "proj_out": "final_layer.video_out", "audio_proj_out": "final_layer.audio_out",
    "token_refiner.final_norm": "token_refiner.final_norm",
}
BLOCK = {
    "norm1.weight": "norm1.weight", "norm2.weight": "norm2.weight", "attn.norm_q.weight": "attn.q_norm.weight",
    "attn.norm_k.weight": "attn.k_norm.weight", "attn.to_out.0.weight": "attn.out_proj.weight",
    "ff.net.2.weight": "mlp.fc2.weight", "adaln_proj.linear.weight": "adaln_proj.linear.weight",
    "adaln_proj.linear.bias": "adaln_proj.linear.bias",
}


def block_prefixes(sd):
    """(diffusers prefix, ComfyUI prefix) of every block."""
    ps = []
    for dp, cp in (("transformer_blocks.", "blocks."), ("token_refiner.refiner_blocks.", "token_refiner.blocks.")):
        n = len({k[len(dp):].split(".")[0] for k in sd if k.startswith(dp)})
        ps += [(f"{dp}{i}.", f"{cp}{i}.") for i in range(n)]
    return ps


def interleave(w):
    """[q; k; v] rows -> [q k v] per head."""
    heads = w.shape[0] // (3 * hd)
    return w.reshape(3, heads, hd, *w.shape[1:]).transpose(0, 1).reshape(w.shape)


def to_comfy(sd, interleaved=False):
    """diffusers' state dict in ComfyUI's names and row orders (float32 tensors)."""
    o, used = {}, set()
    for dp, cp in block_prefixes(sd):
        qkv = torch.cat([sd[f"{dp}attn.to_{x}.weight"] for x in "qkv"])
        o[f"{cp}attn.qkv_proj.weight"] = interleave(qkv) if interleaved else qkv
        value, gate = sd[f"{dp}ff.net.0.proj.weight"].chunk(2)
        o[f"{cp}mlp.fc1.weight"] = torch.cat([gate, value])
        used |= {f"{dp}attn.to_{x}.weight" for x in "qkv"} | {f"{dp}ff.net.0.proj.weight"}
        for a, b in BLOCK.items():
            if dp + a in sd:
                o[cp + b] = sd[dp + a]
                used.add(dp + a)
    for a, b in TOP.items():
        for s in (".weight", ".bias"):
            if a + s in sd:
                o[b + s] = sd[a + s]
                used.add(a + s)
    assert used == set(sd), sorted(set(sd) - used)
    rd = cfg.rope_freq_dim
    o["rope.inv_freq"] = 1.0 / cfg.rope_theta ** (torch.arange(rd, dtype=torch.float64) / rd)
    return o


def from_comfy(o):
    """The inverse of `to_comfy` (ComfyUI row orders), for references on changed weights."""
    sd = {}
    inv_top = {b: a for a, b in TOP.items()}
    inv_block = {b: a for a, b in BLOCK.items()}
    for k, v in o.items():
        if k == "rope.inv_freq":
            continue
        cp = next((c for c in ("token_refiner.blocks.", "blocks.") if k.startswith(c)), None)
        if cp is None:
            base, s = k.rsplit(".", 1)
            sd[f"{inv_top[base]}.{s}"] = v
            continue
        i, tail = k[len(cp):].split(".", 1)
        dp = ("token_refiner.refiner_blocks." if cp.startswith("token") else "transformer_blocks.") + i + "."
        if tail == "attn.qkv_proj.weight":
            for x, w in zip("qkv", v.chunk(3)):
                sd[f"{dp}attn.to_{x}.weight"] = w
        elif tail == "mlp.fc1.weight":
            gate, value = v.chunk(2)
            sd[f"{dp}ff.net.0.proj.weight"] = torch.cat([value, gate])
        else:
            sd[dp + inv_block[tail]] = v
    return sd


def typed(o):
    """ComfyUI's dtypes: bf16, the patch / time / output layers float32."""
    return {k: (v.float() if k.startswith(F32) else v.to(torch.bfloat16)).contiguous() for k, v in o.items()}


sd = {k: v.clone() for k, v in dit.state_dict().items()}
comfy = to_comfy(sd)
release_cfg = {
    "hidden_size": cfg.hidden_size, "num_layers": cfg.num_layers, "token_refiner_num_layers": cfg.num_refiner_layers,
    "num_attention_heads": cfg.num_attention_heads, "attention_head_dim": hd, "ffn_hidden_size": ffn,
    "latents_dim": cfg.in_channels, "audio_latents_dim": cfg.audio_in_channels, "patch_size": list(cfg.patch_size),
    "text_dim": cfg.text_dim, "timestep_input_dim": cfg.freq_dim, "time_embed_hidden_size": cfg.time_embed_hidden_dim,
    "time_embed_dim": cfg.time_embed_dim, "rope_inv_freq_len": cfg.rope_freq_dim, "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5,
}
save_file(typed(comfy), os.path.join(out, "h3_bf16.safetensors"),
          {"config": json.dumps({"transformer": {**release_cfg, "image_model": "minimax_h3"}})})
os.makedirs(os.path.join(out, "release"), exist_ok=True)
save_file(typed(to_comfy(sd, interleaved=True)), os.path.join(out, "release", "model.safetensors"))
json.dump({"_class_name": "MiniMaxH3DiTModel", **release_cfg}, open(os.path.join(out, "release", "config.json"), "w"))

# ---- the "pruned" (curve) form -------------------------------------------------------------------
GRID, K = 1025, 8
with torch.no_grad():
    t = torch.arange(GRID, dtype=torch.float32) / (GRID - 1)
    curve = torch.nn.functional.silu(dit.time_embedder(dit.time_proj(t))).double()
mean = curve.mean(0)
U, S, Vt = torch.linalg.svd(curve - mean, full_matrices=False)
table = (U[:, :K] * S[:K]).float()
pruned = {k: v for k, v in comfy.items() if not k.startswith("time_embedder.")}
for k in [k for k in pruned if k.endswith("adaln_proj.linear.weight")]:
    w, b = pruned[k].double(), pruned[k[:-6] + "bias"].double()
    pruned[k] = (w @ Vt[:K].T).half()
    pruned[k[:-6] + "bias"] = (b + w @ mean).half()
pruned["adaln_t_table"] = table


def typed_pruned(o):
    return {k: (v if "adaln" in k else typed({k: v})[k]).contiguous() for k, v in o.items()}


save_file(typed_pruned(pruned), os.path.join(out, "h3_pruned_bf16.safetensors"))
print(f"curve: the first {K} of {len(S)} singular values hold all but {float(S[K:].square().sum() / S.square().sum()):.2e} of the centred curve")

# ---- FP8 scaled ------------------------------------------------------------------------------------
fp8, decoded = dict(typed_pruned(pruned)), dict(pruned)
for k in [k for k in pruned if k.startswith("blocks.") and k.endswith((".qkv_proj.weight", ".out_proj.weight", ".fc1.weight", ".fc2.weight"))]:
    w = pruned[k].float()
    scale = (w.abs().max() / 448.0).float()  # the largest float8_e4m3fn value
    q = (w / scale).to(torch.float8_e4m3fn)
    base = k[:-len(".weight")]
    fp8[k] = q
    fp8[base + ".weight_scale"] = scale
    quant = {"format": "float8_e4m3fn"}
    if k.endswith(".fc2.weight"):
        quant["full_precision_matrix_mult"] = True
    else:
        fp8[base + ".input_scale"] = torch.tensor(0.05)
    fp8[base + ".comfy_quant"] = torch.tensor(list(json.dumps(quant).encode()), dtype=torch.uint8)
    decoded[k] = q.float() * scale
save_file(fp8, os.path.join(out, "h3_pruned_fp8_scaled.safetensors"))

# ---- a LoRA in the turbo LoRAs' namings ----------------------------------------------------------
R, ALPHA = 4, 2.0
peft, peft_adaln = {}, {}
for dp, _ in block_prefixes(sd):
    for m in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"):
        w = sd[f"{dp}{m}.weight"]
        peft[dp + m] = (bf16r(torch.randn(R, w.shape[1]) * 0.1), bf16r(torch.randn(w.shape[0], R) * 0.1))
for m in [k[:-len(".weight")] for k in sd if k.endswith("adaln_proj.linear.weight")] + ["norm_out.linear"]:
    w = sd[m + ".weight"]
    peft_adaln[m] = (bf16r(torch.randn(2, w.shape[1]) * 0.1), bf16r(torch.randn(w.shape[0], 2) * 0.1))


def merged(base, lora, alphas):
    """diffusers' state dict with the PEFT modules merged."""
    m = dict(base)
    for name, (a, b) in lora.items():
        m[name + ".weight"] = base[name + ".weight"] + alphas(name) / a.shape[0] * (b @ a)
    return m


def comfy_lora(lora, alphas):
    o = {}
    names = {dp: cp for dp, cp in block_prefixes(sd)}
    for name, (a, b) in lora.items():
        dp = next((d for d in names if name.startswith(d)), None)
        if dp is None:
            key, alpha = TOP[name], alphas(name)
        else:
            tail = name[len(dp):]
            if tail in ("attn.to_k", "attn.to_v"):
                continue
            if tail == "attn.to_q":
                parts = [lora[f"{dp}attn.to_{x}"] for x in "qkv"]
                a = torch.cat([p[0] for p in parts])
                b = torch.block_diag(*[p[1] for p in parts])
                key, alpha = names[dp] + "attn.qkv_proj", 3 * alphas(name)
            elif tail == "ff.net.0.proj":
                value, gate = b.chunk(2)
                b = torch.cat([gate, value])
                key, alpha = names[dp] + "mlp.fc1", alphas(name)
            else:
                key = names[dp] + {"attn.to_out.0": "attn.out_proj", "ff.net.2": "mlp.fc2", "adaln_proj.linear": "adaln_proj.linear"}[tail]
                alpha = alphas(name)
        o[f"diffusion_model.{key}.lora_A.weight"] = a.to(torch.bfloat16)
        o[f"diffusion_model.{key}.lora_B.weight"] = b.to(torch.bfloat16)
        o[f"diffusion_model.{key}.alpha"] = torch.tensor(float(alpha))
    return o


meta = {"qkv_fusion": "block diagonal B; concat A; alpha multiplied by 3", "swi_glu_mapping": "Diffusers [value;gate] -> ComfyUI [gate;value]"}
save_file(comfy_lora(peft, lambda n: ALPHA), os.path.join(out, "lora_comfyui.safetensors"), meta)
save_file({f"{n}.lora_{x}.default.weight": w.to(torch.bfloat16) for n, ab in peft.items() for x, w in zip("AB", ab)},
          os.path.join(out, "lora_diffusers.safetensors"), {"format": "pt", "alpha": str(int(ALPHA))})
both = {**peft, **peft_adaln}
adaln_alpha = lambda n: 1.0 if "adaln" in n or n == "norm_out.linear" else ALPHA
save_file(comfy_lora(both, adaln_alpha), os.path.join(out, "lora_adaln_comfyui.safetensors"), meta)

# ---- references ----------------------------------------------------------------------------------
n_text, lt, dh, dw, n_audio, steps = 9, case["lt"], case["lh"], case["lw"], case["n_audio"], case["steps"]
patch = tuple(cfg.patch_size)
text = load("dit_text.bin", (n_text, cfg.text_dim))
text_tags = load("dit_text_tags.bin", (n_text,)).long()
rows0 = load("dit_video.bin", (-1, cfg.in_channels * 4))
audio0 = load("dit_audio.bin", (n_audio * 2, cfg.audio_in_channels))
pos, tags, vidx, aidx, tidx, n_cond, n_cond_audio = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
    text_tags, lt, dh, dw, n_audio, patch, 2, 2, 0, ("first", "last")
)


class Curve(torch.nn.Module):
    """ComfyUI's `_time_embedding` of the curve form: the table's row at t, interpolated."""

    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, t):
        p = t.float().clamp(0.0, 1.0) * (self.table.shape[0] - 1)
        i0 = p.floor().long().clamp(max=self.table.shape[0] - 2)
        return torch.lerp(self.table[i0], self.table[i0 + 1], (p - i0).unsqueeze(1))


def curve_model(o):
    """diffusers' model computing the curve form of ComfyUI state dict `o` (float32)."""
    m = copy.deepcopy(dit)
    full = {k: v.float() for k, v in o.items() if k != "adaln_t_table"}
    for k in [k for k in full if "adaln_proj.linear" in k]:
        full.pop(k)
    missing, unexpected = m.load_state_dict(from_comfy(full), strict=False)
    assert not unexpected and all("adaln_proj" in k or k.startswith(("time_embedder", "norm_out.linear")) for k in missing), missing
    m.time_proj = torch.nn.Identity()
    m.time_embedder = Curve(o["adaln_t_table"].float())

    def lin(key):
        w, b = o[key + ".weight"].float(), o[key + ".bias"].float()
        l = torch.nn.Linear(w.shape[1], w.shape[0])
        l.weight.data, l.bias.data = w, b
        return l

    for i, blk in enumerate(m.transformer_blocks):
        blk.adaln_proj.linear = lin(f"blocks.{i}.adaln_proj.linear")
        hs = blk.adaln_proj.hidden_size
        # diffusers' forward without the SiLU.
        blk.adaln_proj.forward = (lambda mod, hs: lambda temb: mod.linear(temb).view(-1, 6 * hs).chunk(6, dim=-1))(blk.adaln_proj, hs)
    no = m.norm_out
    no.linear = lin("final_layer.adaln_proj.linear")

    def out_forward(hidden_states, temb, timestep_indices):
        shift, scale = no.linear(temb).chunk(2, dim=-1)
        hidden_states = no.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)

    no.forward = out_forward
    return m.eval()


def run(model, prefix):
    rows, audio = rows0.clone(), audio0.clone()
    sv, sa = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    sv.set_timesteps(steps)
    sa.set_timesteps(steps)
    with torch.no_grad():
        for i, (t, ta) in enumerate(zip(sv.timesteps, sa.timesteps)):
            uniq, ridx = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                vidx, aidx, n_cond, n_cond_audio, n_text, float(t), float(ta), max(float(t), 0.999), 1.0
            )
            v, a = model(hidden_states=rows[None], audio_hidden_states=audio[None], encoder_hidden_states=text[None],
                         timestep=uniq, timestep_indices=ridx, token_tags=tags, position_ids=pos.float(),
                         video_indices=vidx, audio_indices=aidx, text_indices=tidx, return_dict=False)
            if i == 0:
                save(f"{prefix}dit_v0.bin", v[0].numpy())
                save(f"{prefix}dit_a0.bin", a[0].numpy())
            rows[n_cond:] = sv.step(v[0, n_cond:].float(), t, rows[n_cond:], return_dict=False)[0]
            audio = sa.step(a[0].float(), ta, audio, return_dict=False)[0]
    save(f"{prefix}dit_video_final.bin", rows.numpy())
    save(f"{prefix}dit_audio_final.bin", audio.numpy())
    return rows


def with_weights(state):
    m = copy.deepcopy(dit)
    m.load_state_dict(state)
    return m.eval()


base_final = load("dit_video_final.bin", rows0.shape)
check = run(dit, "check_")
assert torch.equal(check, base_final), "the tiny tool's reference is not reproduced"
for f in ("dit_v0", "dit_a0", "dit_video_final", "dit_audio_final"):
    os.remove(os.path.join(src, f"check_{f}.bin"))
pr = run(curve_model(pruned), "pruned_")
print(f"pruned vs full (rank {K} of a tiny random curve): max abs diff of the final rows {float((pr - base_final).abs().max()):.3e}")
run(curve_model(decoded), "fp8_")
lora_alpha = lambda n: ALPHA
run(with_weights(merged(sd, peft, lora_alpha)), "lora_")
run(with_weights(merged(sd, both, adaln_alpha)), "loraadaln_")
dec_sd = from_comfy({k: v.float() for k, v in decoded.items() if k != "adaln_t_table" and "adaln_proj.linear" not in k})
dec_merged = merged(dec_sd, peft, lora_alpha)
fp8_lora = dict(decoded)
fp8_lora.update({k: v for k, v in to_comfy({**sd, **dec_merged}).items() if "adaln" not in k and not k.startswith("time_embedder.")})
run(curve_model(fp8_lora), "fp8lora_")

# ---- the video VAE as ComfyUI's single file --------------------------------------------------------
vae = pretrained(AutoencoderKLMiniMaxH3, "vae")
vc = vae.config
vsd = {k: v.half() for k, v in vae.state_dict().items()}
vae.load_state_dict({k: v.float() for k, v in vsd.items()})
vhd = vc.decoder_attention_head_dim


def vae_interleave(w):
    heads = w.shape[0] // (3 * vhd)
    return w.reshape(3, heads, vhd, *w.shape[1:]).transpose(0, 1).reshape(w.shape)


cv, used = {}, set()
for i in range(vc.decoder_num_layers):
    p = f"decoder.transformer_blocks.{i}."
    for s in ("weight", "bias"):
        cv[f"{p}attn.to_qkv.{s}"] = vae_interleave(torch.cat([vsd[f"{p}attn.to_{x}.{s}"] for x in "qkv"]))
        value, gate = vsd[f"{p}ff.net.0.proj.{s}"].chunk(2)
        cv[f"{p}ff.w1.{s}"] = torch.cat([gate, value])
        cv[f"{p}attn.to_out.{s}"] = vsd[f"{p}attn.to_out.0.{s}"]
        cv[f"{p}ff.w2.{s}"] = vsd[f"{p}ff.net.2.{s}"]
        used |= {f"{p}attn.to_{x}.{s}" for x in "qkv"} | {f"{p}ff.net.0.proj.{s}", f"{p}attn.to_out.0.{s}", f"{p}ff.net.2.{s}"}
for k, v in vsd.items():
    if k in used:
        continue
    n = k.replace("decoder.proj_in.", "decoder.x_embedder.").replace("encoder.down_blocks.", "encoder.down.")
    n = n.replace(".resnets.", ".block.").replace(".conv_shortcut.", ".nin_shortcut.").replace(".downsamplers.0.conv.", ".downsample.conv.")
    cv[n] = v
cv["latents_mean"] = torch.tensor(vc.latents_mean).half()
cv["latents_std"] = torch.tensor(vc.latents_std).half()
cv["decoder.mask_token"] = torch.zeros(1, 1, vsd["decoder.proj_in.weight"].shape[0]).half()
ch = vc.block_out_channels[0]
source = {
    "_class_name": "AutoencoderKLLegacy", "ch": ch, "ch_mult": [c // ch for c in vc.block_out_channels],
    "embed_dim": vc.latent_channels, "z_channels": vc.latent_channels, "num_res_blocks": vc.layers_per_block,
    "padding_mode": vc.spatial_padding_mode, "space_down": list(vc.spatial_downsample_factors),
    "time_down": list(vc.temporal_downsample_factors), "use_vit_decoder": True,
    # Not in the release's header (its groups are the default 32): the tiny encoder's widths need fewer.
    "norm_num_groups": vc.norm_num_groups,
    "vit_decoder_kwargs": {"dim_head": vhd, "heads": vc.decoder_num_attention_heads, "num_layers": vc.decoder_num_layers,
                           "rope_dim_ratio": vc.decoder_rope_dim_ratio, "rope_theta": vc.decoder_rope_theta},
}
vmeta = {"latents_mean": list(vc.latents_mean), "latents_std": list(vc.latents_std), "vae_clip_length": vc.clip_length,
         "vae_token_drop": vc.token_drop, "source_config": source}
save_file({k: v.contiguous() for k, v in cv.items()}, os.path.join(out, "video_vae_fp16.safetensors"),
          {"minimax_h3_video_vae": json.dumps(vmeta)})
H, W, frames = case["h"], case["w"], case["frames"]
video = load("video.bin", (1, 3, frames, H, W))
C, lh, lw = vc.latent_channels, H // 16, W // 16
mean = torch.tensor(vc.latents_mean).view(-1, 1, 1, 1)
std = torch.tensor(vc.latents_std).view(-1, 1, 1, 1)
with torch.no_grad():
    save("fp16_moments.bin", vae.encode(video, return_dict=False)[0].parameters[0].numpy())
    save("fp16_moments1.bin", vae.encode(video[:, :, :1], return_dict=False)[0].parameters[0].numpy())
    for t in case["decode"]:
        lat = load(f"latents_{t}.bin", (C, int(t), lh, lw))
        save(f"fp16_decoded_{t}.bin", vae.decode((lat * std + mean)[None], return_dict=False)[0][0].numpy())
print("wrote", out)
