#!/usr/bin/env python3
"""One MiniMax-H3 forward through ComfyUI's own code on the real small files (CPU, float32), against
what oxinfer computed on the same inputs (its OXINFER_H3_DUMP files, see minimax_h3_video_ref.py).

usage: minimax_h3_video_comfy_ref.py COMFYUI_DIR DUMP EMBEDS.bin TRANSFORMER.safetensors LORA.safetensors \\
           CALL SIGMA_V LT LH LW N_AUDIO [--shifts 6 3] [--diffusers] [--sample OUT_DIR [--points 5]]
  COMFYUI_DIR: a checkout of ComfyUI (its requirements installed: comfy-kitchen, comfy-aimdo, ...)
  CALL, SIGMA_V: the forward of the dump to redo and its video sigma (448x256, 124 frames, the
  4-step 768p LoRA: 0 1.0, 1 0.9473684430122375, 2 0.8571428656578064, 3 0.6666666865348816)

ComfyUI loads the transformer (`comfy.sd.load_diffusion_model`: its reading of the FP8 scaled
weights and of the curve form), patches the LoRA into it (`comfy.sd.load_lora_for_models`, then
`load_models_gpu`: on the CPU every patched FP8 weight is rounded back to FP8, stochastically) and
runs `MiniMaxH3Model.forward` with float activations in every quantized layer
(`full_precision_matrix_mult`). Printed:
  - ComfyUI's patched weights of the first block against the file's decoded weights with the
    LoRA merged in float32 (what oxinfer and minimax_h3_video_ref.py compute with);
  - ComfyUI's velocities against oxinfer's;
  - with --diffusers, diffusers' transformer on ComfyUI's patched weights against both: the same
    weights through the two implementations, and the same implementation as
    minimax_h3_video_ref.py on the two sets of weights;
  - with --sample OUT_DIR (CALL 0, SIGMA_V 1), the whole sampling with ComfyUI's forwards from
    oxinfer's noise: Euler steps on the grid of --points sigma points shifted by --shifts (the
    audio sigma from the video's, as ComfyUI derives it), each step's distance from oxinfer's,
    and OUT_DIR/comfy.latents in the format of OXINFER_H3_LATENTS for oxinfer to decode.
"""
import argparse, os, struct, sys, time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
for n in ("comfyui", "dump", "embeds", "transformer", "lora"):
    ap.add_argument(n)
ap.add_argument("call", type=int)
ap.add_argument("sigma", type=float)
for n in ("lt", "lh", "lw", "n_audio"):
    ap.add_argument(n, type=int)
ap.add_argument("--shifts", type=float, nargs=2, default=[6.0, 3.0])
ap.add_argument("--diffusers", action="store_true")
ap.add_argument("--sample")
ap.add_argument("--points", type=int, default=5)
a = ap.parse_args()
sys.argv = sys.argv[:1]  # ComfyUI reads the command line
sys.path.insert(0, a.comfyui)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy.cli_args import args as comfy_args

comfy_args.cpu = True
import numpy as np
import torch
import comfy.ldm.minimax.model as M
import comfy.model_management
import comfy.sd
import comfy.utils
from comfy.quant_ops import QuantizedTensor

import minimax_h3_video_ref as ref

torch.set_grad_enabled(False)
mp = comfy.sd.load_diffusion_model(a.transformer)
mp, _ = comfy.sd.load_lora_for_models(mp, None, comfy.utils.load_torch_file(a.lora, safe_load=True), 1.0, 0.0)
t0 = time.time()
comfy.model_management.load_models_gpu([mp])
dm = mp.model.diffusion_model
print(f"ComfyUI: {type(mp.model).__name__}, {len(mp.patches)} LoRA patches, loaded in {time.time() - t0:.0f}s", flush=True)
for m in dm.modules():
    if hasattr(m, "_full_precision_mm"):
        m._full_precision_mm = True


def f32(t):
    return (t.dequantize() if isinstance(t, QuantizedTensor) else t).float()


class ComfyWeights(ref.Weights):
    """The tensors of ComfyUI's loaded, patched model."""

    def __init__(self):
        self.k = dm.adaln_t_table.shape[1]

    def get(self, name):
        if name == "adaln_t_table":
            return dm.adaln_t_table.float()
        mod, attr = name.rsplit(".", 1)
        return f32(getattr(dm.get_submodule(mod), attr)).clone()

    def weight(self, name):
        return self.get(name + ".weight")


cw, fw = ComfyWeights(), ref.Weights(a.transformer, a.lora)
for name in ("blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj", "blocks.0.mlp.fc1", "blocks.0.mlp.fc2"):
    c, merged, base = cw.weight(name), fw.weight(name), fw.get(name + ".weight") * fw.get(name + ".weight_scale")
    d = c - merged
    print(f"{name}: ComfyUI's patched weight against the file's with the LoRA merged in float32: relative {float(d.norm() / merged.norm()):.3e}, "
          f"mean {float(d.mean()):+.2e}, rms {float(d.pow(2).mean().sqrt()):.3e}; the LoRA's own change: relative "
          f"{float((merged - base).norm() / base.norm()):.3e}, ComfyUI's change against it: cosine "
          f"{float(torch.nn.functional.cosine_similarity((c - base).flatten(), (merged - base).flatten(), dim=0)):.4f}", flush=True)
for name in ("condition_proj.weight", "blocks.0.norm1.weight", "blocks.0.adaln_proj.linear.weight", "final_layer.video_out.weight"):
    print(f"{name}: ComfyUI's against the file's, max abs diff {float((cw.get(name) - fw.get(name)).abs().max()):.3e}", flush=True)


def rd(name, cols):
    return ref.rd(a.dump, name, cols)


b = open(a.embeds, "rb").read()
n_text = struct.unpack("<I", b[:4])[0]
text_tags = torch.tensor(list(b[4:4 + n_text]), dtype=torch.long)
text = torch.from_numpy(np.frombuffer(b[4 + n_text:], dtype="<f4").reshape(1, n_text, -1).copy())
i = a.call
rows, audio = rd(f"s{i}_video_in.bin", 96), rd(f"s{i}_audio_in.bin", 32)
video, aud = M.unpatchify_video(rows, a.lt, a.lh // 2, a.lw // 2), M.unpack_audio(audio)
assert torch.equal(M.patchify_video(video), rows) and torch.equal(M.pack_audio(aud), audio)
to = {"minimax_h3_sigma_shift_video": a.shifts[0], "minimax_h3_sigma_shift_audio": a.shifts[1]}
t0 = time.time()
out = dm([video, aud], torch.tensor([a.sigma * 1000.0]), text, transformer_options=to)
print(f"ComfyUI forward in {time.time() - t0:.0f}s", flush=True)
# ComfyUI returns the negated velocities.
cv, ca = M.patchify_video(-out[0].float()), M.pack_audio(-out[1].float())
ov, oa = rd(f"s{i}_v.bin", 96), rd(f"s{i}_a.bin", 32)
ref.cmp(f"step {i}, video velocity, ComfyUI against oxinfer", cv, ov)
ref.cmp(f"step {i}, audio velocity, ComfyUI against oxinfer", ca, oa)
if a.diffusers:
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep

    model = ref.build_model(cw)
    pos, tags, vidx, aidx, tidx, n_cond, n_cond_audio = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_tags, a.lt, a.lh, a.lw, a.n_audio, (1, 2, 2), 2, 2, 0, ())
    sigma_v = torch.tensor(a.sigma, dtype=torch.float32)
    t, ta = float(1.0 - sigma_v), float(1.0 - M.time_shift_sigma(sigma_v, a.shifts[0], a.shifts[1]))
    uniq, ridx = MiniMaxH3SetTimestepsStep.build_row_timesteps(vidx, aidx, n_cond, n_cond_audio, n_text, t, ta, max(t, 0.999), 1.0)
    t0 = time.time()
    dv, da = model(hidden_states=rows[None], audio_hidden_states=audio[None], encoder_hidden_states=text,
                   timestep=uniq, timestep_indices=ridx, token_tags=tags, position_ids=pos.float(),
                   video_indices=vidx, audio_indices=aidx, text_indices=tidx, return_dict=False)
    print(f"diffusers forward on ComfyUI's weights in {time.time() - t0:.0f}s (t = {t:.6f}, audio {ta:.6f})", flush=True)
    ref.cmp(f"step {i}, video velocity, ComfyUI against diffusers on ComfyUI's weights", cv, dv[0])
    ref.cmp(f"step {i}, audio velocity, ComfyUI against diffusers on ComfyUI's weights", ca, da[0])
    ref.cmp(f"step {i}, video velocity, diffusers on ComfyUI's weights against oxinfer", dv[0], ov)
    ref.cmp(f"step {i}, audio velocity, diffusers on ComfyUI's weights against oxinfer", da[0], oa)
if a.sample:
    n = a.points - 1
    base = torch.linspace(1.0, 0.0, a.points)
    sv = a.shifts[0] * base / (1.0 + (a.shifts[0] - 1.0) * base)
    sa = M.time_shift_sigma(sv, a.shifts[0], a.shifts[1])
    print("sigmas", sv.tolist(), sa.tolist(), flush=True)
    for j in range(n):
        if j > 0:
            ref.cmp(f"step {j}, video rows before, ComfyUI against oxinfer", rows, rd(f"s{j}_video_in.bin", 96))
            video, aud = M.unpatchify_video(rows, a.lt, a.lh // 2, a.lw // 2), M.unpack_audio(audio)
            out = dm([video, aud], sv[j:j + 1] * 1000.0, text, transformer_options=to)
            cv, ca = M.patchify_video(-out[0].float()), M.pack_audio(-out[1].float())
            ref.cmp(f"step {j}, video velocity, ComfyUI against oxinfer", cv, rd(f"s{j}_v.bin", 96))
            ref.cmp(f"step {j}, audio velocity, ComfyUI against oxinfer", ca, rd(f"s{j}_a.bin", 32))
        rows = rows + (sv[j] - sv[j + 1]) * cv
        audio = audio + (sa[j] - sa[j + 1]) * ca
    with open(f"{a.sample}/comfy.latents", "wb") as f:
        f.write(b"H3LT" + struct.pack("<6I", a.lt, a.lh, a.lw, a.n_audio, rows.numel(), audio.numel()))
        f.write(rows.numpy().astype("<f4").tobytes())
        f.write(audio.numpy().astype("<f4").tobytes())
    print(f"wrote {a.sample}/comfy.latents", flush=True)
