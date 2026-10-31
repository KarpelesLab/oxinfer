#!/usr/bin/env python3
"""Reference check for oxinfer's Qwen-Image pipelines against diffusers / transformers, stage by
stage, on the same inputs and initial noise.

usage:
  ref_qwen_image.py MODEL_DIR DUMP_DIR --make-noise --size WxH          write DUMP_DIR/noise.bin
  OXINFER_QI_DUMP=DUMP_DIR oxinfer --model MODEL_DIR --init-image IMG --prompt P --size WxH ...
  ref_qwen_image.py MODEL_DIR DUMP_DIR --stage text  --image IMG --prompt P [--negative N]
  ref_qwen_image.py MODEL_DIR DUMP_DIR --stage vae   --image IMG
  ref_qwen_image.py MODEL_DIR DUMP_DIR --stage step  --image IMG --size WxH [--lora F] [--lightning] [--steps N]
  ref_qwen_image.py MODEL_DIR DUMP_DIR --stage full  --image IMG --size WxH [--steps N] [--cfg F] [--lora F] [--lightning]

Each stage loads only the components it needs (run them one at a time: the text encoder is 16 GB
and the transformer 41 GB in bf16), writes its reference outputs as DUMP_DIR/ref_*.npy and
compares them with oxinfer's dumps (prompt_embeds.bin, negative_embeds.bin, ref_latents0.bin,
v0.bin, latents_out.bin, image.bin). Later stages read the earlier stages' reference outputs,
so each stage is checked on reference inputs.

Weights stay bf16 but every matmul runs in float32 (the bf16 weights upcast per call), as
oxinfer computes: the comparison then measures implementation error, not bf16 activation
rounding. The text encoder is called with `mm_token_type_ids`, which diffusers 0.42 does not pass
(transformers 5 then falls back to plain 1D positions for the image tokens instead of M-RoPE,
the positions the model was trained with).
"""
import argparse, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("dump")
ap.add_argument("--stage", default="")
ap.add_argument("--make-noise", action="store_true")
ap.add_argument("--image", action="append", default=[])
ap.add_argument("--prompt", default="")
ap.add_argument("--negative")
ap.add_argument("--size")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--lora")
ap.add_argument("--lightning", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

D = a.dump
os.makedirs(D, exist_ok=True)
torch.set_grad_enabled(False)


def ours(name, cols=None):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return None
    v = np.fromfile(p, dtype="<f4")
    return v.reshape(-1, cols) if cols else v


def report(name, o, r):
    if o is None:
        print(f"{name}: no oxinfer dump")
        return
    o, r = np.asarray(o, np.float64).ravel(), np.asarray(r, np.float64).ravel()
    if o.size != r.size:
        print(f"{name}: size ours {o.size} vs ref {r.size}")
        n = min(o.size, r.size)
        o, r = o[:n], r[:n]
    rel = np.linalg.norm(o - r) / max(np.linalg.norm(r), 1e-30)
    print(f"{name}: rel L2 {rel:.3e}, max abs {np.abs(o - r).max():.4e} (ref max |x| {np.abs(r).max():.3f})")


def f32_linears(model):
    """bf16 weights, float32 compute: Linear upcasts its weights per call; other params -> f32."""
    lin = set()
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            lin.add(id(m.weight))
            if m.bias is not None:
                lin.add(id(m.bias))
    for p in model.parameters():
        if id(p) not in lin:
            p.data = p.data.float()
    for b_name, b in list(model.named_buffers()):
        if b.is_floating_point():
            b.data = b.data.float()


_lin_fwd = torch.nn.Linear.forward


def _f32_linear(self, x):
    w = self.weight
    if w.dtype == torch.float32:
        y = _lin_fwd(self, x)
    else:
        y = F.linear(x.float(), w.float(), None if self.bias is None else self.bias.float())
    lo = getattr(self, "_lora", None)
    if lo is not None:
        dn, up, scale = lo
        y = y + scale * F.linear(F.linear(x.float(), dn), up)
    return y


torch.nn.Linear.forward = _f32_linear


def size():
    w, h = map(int, a.size.lower().split("x"))
    return w, h


if a.make_noise:
    w, h = size()
    lh, lw = 2 * (h // 16), 2 * (w // 16)
    g = torch.Generator().manual_seed(a.seed)
    n = torch.randn((16, lh, lw), generator=g, dtype=torch.float32)
    n.numpy().astype("<f4").tofile(os.path.join(D, "noise.bin"))
    print(f"noise [16, {lh}, {lw}] -> {D}/noise.bin")
    sys.exit(0)

from PIL import Image
from diffusers import QwenImageEditPlusPipeline, AutoencoderKLQwenImage, QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions, calculate_shift, CONDITION_IMAGE_SIZE, VAE_IMAGE_SIZE

imgs = [Image.open(p).convert("RGB") for p in a.image]
M = a.model


def pipe_with(**kw):
    """A pipeline object holding only the given components (the others None)."""
    comps = dict(scheduler=None, vae=None, text_encoder=None, tokenizer=None, processor=None, transformer=None)
    comps.update(kw)
    return QwenImageEditPlusPipeline(**comps)


if a.stage == "text":
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor, Qwen2Tokenizer
    t0 = time.time()
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(os.path.join(M, "text_encoder"), dtype=torch.bfloat16).eval()
    f32_linears(te)
    proc = Qwen2VLProcessor.from_pretrained(os.path.join(M, "processor"))
    tok = Qwen2Tokenizer.from_pretrained(os.path.join(M, "tokenizer"))
    print(f"loaded text encoder in {time.time() - t0:.1f}s")
    img_tok = te.config.image_token_id
    fwd = te.forward

    def forward(*args, **kw):
        if kw.get("mm_token_type_ids") is None and kw.get("input_ids") is not None:
            kw["mm_token_type_ids"] = (kw["input_ids"] == img_tok).long()
        return fwd(*args, **kw)

    te.forward = forward
    p = pipe_with(text_encoder=te, tokenizer=tok, processor=proc)
    cond = []
    for img in imgs:
        cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, img.size[0] / img.size[1])
        cond.append(p.image_processor.resize(img, ch, cw))
    if cond:
        pi = proc.image_processor(images=cond[:1], return_tensors="pt")
        f = te.model.get_image_features(pi["pixel_values"], pi["image_grid_thw"])
        emb = torch.cat(list(f.pooler_output), 0).float().numpy()
        print("condition image 0:", tuple(pi["image_grid_thw"][0].tolist()), "->", emb.shape)
        report("vision tower (condition image 0)", ours("cond_image0.bin"), emb)
    for name, text in [("prompt_embeds", a.prompt), ("negative_embeds", a.negative)]:
        if text is None:
            continue
        t0 = time.time()
        e, _ = p.encode_prompt(prompt=text, image=cond, device=torch.device("cpu"))
        e = e[0].float().numpy()
        print(f"{name}: {e.shape} in {time.time() - t0:.1f}s")
        np.save(os.path.join(D, f"ref_{name}.npy"), e)
        report(name, ours(f"{name}.bin"), e)

elif a.stage == "vae":
    vae = AutoencoderKLQwenImage.from_pretrained(os.path.join(M, "vae"), dtype=torch.float32).eval()
    p = pipe_with(vae=vae)
    lats = []
    for i, img in enumerate(imgs):
        vw, vh = calculate_dimensions(VAE_IMAGE_SIZE, img.size[0] / img.size[1])
        x = p.image_processor.preprocess(img, vh, vw).unsqueeze(2)
        t0 = time.time()
        lat = p._encode_vae_image(x, None)  # [1, 16, 1, lh, lw], normalized
        print(f"vae encode {tuple(x.shape)} -> {tuple(lat.shape)} in {time.time() - t0:.1f}s")
        packed = p._pack_latents(lat[:, :, 0], 1, 16, lat.shape[3], lat.shape[4])[0].numpy()
        lats.append(packed)
        if i == 0:
            report("ref_latents0", ours("ref_latents0.bin"), packed)
    np.save(os.path.join(D, "ref_image_latents.npy"), np.concatenate(lats, 0))
    # Decode oxinfer's final latents if present, else the reference image latents.
    lo = ours("latents_out.bin")
    if lo is not None and a.size:
        w, h = size()
        lt = torch.from_numpy(lo.reshape(1, -1, 64))
        z = p._unpack_latents(lt, h, w, 8)
        mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
        img = vae.decode(z * std + mean, return_dict=False)[0][:, :, 0]
        ref = img[0].permute(1, 2, 0).numpy()
        report("vae decode (oxinfer latents)", ours("image.bin"), ref)

elif a.stage in ("step", "full"):
    w, h = size()
    tr = QwenImageTransformer2DModel.from_pretrained(os.path.join(M, "transformer"), dtype=torch.bfloat16).eval()
    f32_linears(tr)
    if a.lora:
        from safetensors.torch import load_file
        sd = load_file(a.lora)
        mods = {}
        for k, v in sd.items():
            for suf, r in ((".lora_down.weight", 0), (".lora_up.weight", 1), (".alpha", 2), (".lora_A.weight", 0), (".lora_B.weight", 1)):
                if k.endswith(suf):
                    mods.setdefault(k[: -len(suf)], [None, None, None])[r] = v
        named = dict(tr.named_modules())
        for m, (dn, up, al) in mods.items():
            mm = m.removeprefix("transformer.").removeprefix("diffusion_model.")
            lin = named[mm]
            r = dn.shape[0]
            scale = (float(al) if al is not None else r) / r
            # Unmerged, in f32 (as oxinfer applies a LoRA to bf16 weights at run time).
            lin._lora = (dn.float(), up.float(), scale)
        print(f"attached {len(mods)} LoRA modules")
    cfg = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(M, "scheduler")).config
    if a.lightning:
        cfg = dict(cfg)
        cfg.update(base_shift=math.log(3), max_shift=math.log(3), shift_terminal=None)
    sched = FlowMatchEulerDiscreteScheduler.from_config(cfg)
    pe = torch.from_numpy(np.load(os.path.join(D, "ref_prompt_embeds.npy")))[None]
    ne = os.path.join(D, "ref_negative_embeds.npy")
    ne = torch.from_numpy(np.load(ne))[None] if os.path.exists(ne) and a.cfg > 1 else None
    il = torch.from_numpy(np.load(os.path.join(D, "ref_image_latents.npy")))[None]
    lh, lw = 2 * (h // 16), 2 * (w // 16)
    noise = torch.from_numpy(np.fromfile(os.path.join(D, "noise.bin"), dtype="<f4").reshape(1, 16, lh, lw))
    lat = QwenImageEditPlusPipeline._pack_latents(noise, 1, 16, lh, lw)
    shapes = [[(1, lh // 2, lw // 2)] + [(1, d[0], d[1]) for d in
               [(calculate_dimensions(VAE_IMAGE_SIZE, im.size[0] / im.size[1])[1] // 16,
                 calculate_dimensions(VAE_IMAGE_SIZE, im.size[0] / im.size[1])[0] // 16) for im in imgs]]]
    sigmas = np.linspace(1.0, 1 / a.steps, a.steps)
    mu = calculate_shift(lat.shape[1], cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                         cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
    sched.set_timesteps(sigmas=sigmas, mu=mu)
    print("sigmas", sched.sigmas.tolist())
    n_steps = 1 if a.stage == "step" else len(sched.timesteps)
    sched.set_begin_index(0)
    for i, t in enumerate(sched.timesteps[:n_steps]):
        t0 = time.time()
        inp = torch.cat([lat, il], 1)
        ts = t.expand(1).float() / 1000
        v = tr(hidden_states=inp, timestep=ts, encoder_hidden_states=pe, img_shapes=shapes, return_dict=False)[0][:, : lat.size(1)]
        if ne is not None:
            vn = tr(hidden_states=inp, timestep=ts, encoder_hidden_states=ne, img_shapes=shapes, return_dict=False)[0][:, : lat.size(1)]
            comb = vn + a.cfg * (v - vn)
            v = comb * (torch.norm(v, dim=-1, keepdim=True) / torch.norm(comb, dim=-1, keepdim=True))
        print(f"step {i}: t {float(t):.2f} in {time.time() - t0:.1f}s")
        if i == 0:
            np.save(os.path.join(D, "ref_v0.npy"), v[0].float().numpy())
            report("v0 (first velocity)", ours("v0.bin"), v[0].float().numpy())
        lat = sched.step(v, t, lat, return_dict=False)[0]
    if a.stage == "full":
        np.save(os.path.join(D, "ref_latents_out.npy"), lat[0].float().numpy())
        report("final latents", ours("latents_out.bin"), lat[0].float().numpy())
        vae = AutoencoderKLQwenImage.from_pretrained(os.path.join(M, "vae"), dtype=torch.float32).eval()
        z = QwenImageEditPlusPipeline._unpack_latents(lat.float(), h, w, 8)
        mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
        img = vae.decode(z * std + mean, return_dict=False)[0][:, :, 0][0].permute(1, 2, 0).numpy()
        ref8 = np.round(np.clip(img * 0.5 + 0.5, 0, 1) * 255)
        Image.fromarray(ref8.astype(np.uint8)).save(os.path.join(D, "ref_image.png"))
        o = ours("image.bin")
        if o is not None:
            o8 = np.round(np.clip(o.reshape(ref8.shape) * 0.5 + 0.5, 0, 1) * 255)
            mse = np.mean((o8 - ref8) ** 2)
            print(f"final image: PSNR {10 * np.log10(255 ** 2 / max(mse, 1e-12)):.2f} dB vs the reference pipeline")
else:
    print("give --stage text|vae|step|full or --make-noise")
