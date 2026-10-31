#!/usr/bin/env python3
"""Reference outputs for the Stable Diffusion pipeline, to compare with `OXINFER_SD_DUMP`.

Runs the diffusers components in float32 (fp16 weights upcast) step by step, exactly as
oxinfer does: CLIP text encoding, Euler or Euler-ancestral denoising from saved noise, VAE
decoding. Writes latents_in.bin (standard normal noise, [4, h, w] f32), which oxinfer reads
from the same directory, and the reference stages as ref_*.bin.

  python tools/ref_sd.py models/sd-turbo OUT "a red fox" --steps 1 --guidance 0
  OXINFER_SD_DUMP=OUT oxinfer --model models/sd-turbo -p "a red fox" --steps 1 --guidance 0 --sampler euler
  python tools/ref_sd.py --compare OUT

With --lora FILE[:SCALE] (repeatable), the LoRA files are loaded with diffusers'
load_lora_weights (needs peft) and fused into the weights before any bf16 rounding, as oxinfer
merges them at load.

Image editing (--mode img2img | inpaint | pix2pix, with --image, --mask, --strength,
--image-guidance) runs diffusers' own pipelines (StableDiffusionImg2ImgPipeline,
StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline) with the Euler scheduler
and records every random draw by role (latents_in.bin for the initial noise, enc_image.bin and
enc_masked.bin for the VAE samples), which oxinfer reads instead of its own noise. Give images
already at the working size (no resampling on either side):

  python tools/ref_sd.py MODEL OUT "make it snowy" --mode pix2pix --image in.png --steps 10 --guidance 7.5
  OXINFER_SD_DUMP=OUT oxinfer --model MODEL --init-image in.png -p "make it snowy" --steps 10 --sampler euler
"""
import argparse, os, sys
import numpy as np


def save(path, t):
    np.asarray(t, dtype=np.float32).tofile(path)


def compare(out):
    pairs = [("text_cond.bin", "ref_text_cond.bin"), ("enc_mean.bin", "ref_enc_mean.bin"), ("unet_out0.bin", "ref_unet_out0.bin"),
             ("latents_out.bin", "ref_latents_out.bin"), ("image.bin", "ref_image.bin")]
    for ours, ref in pairs:
        if not os.path.exists(os.path.join(out, ref)):
            continue
        a = np.fromfile(os.path.join(out, ours), dtype=np.float32)
        b = np.fromfile(os.path.join(out, ref), dtype=np.float32)
        if a.shape != b.shape:
            print(f"{ours}: shape {a.shape} vs {b.shape}")
            continue
        d = np.abs(a - b)
        rel = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)
        line = f"{ours:16s} max|diff| {d.max():.4g}  rel L2 {rel:.3e}"
        if ours == "image.bin":
            pa = np.clip((a + 1) / 2, 0, 1) * 255
            pb = np.clip((b + 1) / 2, 0, 1) * 255
            mse = np.mean((np.round(pa) - np.round(pb)) ** 2)
            line += f"  PSNR {10 * np.log10(255 ** 2 / max(mse, 1e-12)):.1f} dB (8-bit)"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--negative", default="")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", metavar="DIR")
    ap.add_argument("--lora", action="append", default=[], metavar="FILE[:SCALE]", help="LoRA to fuse (repeatable)")
    ap.add_argument("--mode", default="txt2img", choices=["txt2img", "img2img", "inpaint", "pix2pix"])
    ap.add_argument("--image", help="input image (img2img / inpaint / pix2pix), at the working size")
    ap.add_argument("--mask", help="inpainting mask (white = repaint)")
    ap.add_argument("--strength", type=float, default=None)
    ap.add_argument("--image-guidance", type=float, default=1.5)
    ap.add_argument("--bf16-weights", action="store_true", help="round matmul/conv weights to bf16 as oxinfer stores them (isolates implementation error)")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)

    import torch
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    os.makedirs(a.out, exist_ok=True)
    variant = "fp16" if os.path.exists(os.path.join(a.model, "unet", "diffusion_pytorch_model.fp16.safetensors")) else None
    kw = dict(torch_dtype=torch.float32, variant=variant)
    tok = CLIPTokenizer.from_pretrained(a.model, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(a.model, subfolder="text_encoder", **kw).eval()
    unet = UNet2DConditionModel.from_pretrained(a.model, subfolder="unet", **kw).eval()
    vae = AutoencoderKL.from_pretrained(a.model, subfolder="vae", **kw).eval()
    sched = EulerDiscreteScheduler.from_pretrained(a.model, subfolder="scheduler")
    if a.lora:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline(vae=vae, text_encoder=te, tokenizer=tok, unet=unet, scheduler=sched,
                                       safety_checker=None, feature_extractor=None, requires_safety_checker=False)
        names, scales = [], []
        for i, spec in enumerate(a.lora):
            path, scale = spec, 1.0
            head, _, tail = spec.rpartition(":")
            if head:
                try:
                    path, scale = head, float(tail)
                except ValueError:
                    pass
            pipe.load_lora_weights(os.path.dirname(path) or ".", weight_name=os.path.basename(path), adapter_name=f"l{i}")
            names.append(f"l{i}")
            scales.append(scale)
        pipe.set_adapters(names, adapter_weights=scales)
        pipe.fuse_lora(adapter_names=names, lora_scale=1.0)
        pipe.unload_lora_weights()
        te, unet = pipe.text_encoder.eval(), pipe.unet.eval()
    if a.bf16_weights:
        # oxinfer keeps Linear/Conv weights in bf16 (biases, norms and embeddings stay exact).
        for m in (te, unet, vae):
            for mod in m.modules():
                if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
                    mod.weight.data = mod.weight.data.to(torch.bfloat16).to(torch.float32)

    def encode(text):
        ids = tok(text, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
        return ids, te(ids)[0]

    if a.mode != "txt2img":
        return run_edit(a, tok, te, unet, vae, sched)

    with torch.no_grad():
        ids, cond = encode(a.prompt)
        print("ids", ids[0, :24].tolist())
        save(os.path.join(a.out, "ref_text_cond.bin"), cond[0].numpy())
        uncond = encode(a.negative)[1] if a.guidance > 1 else None

        lh = a.size // 8
        g = torch.Generator().manual_seed(a.seed)
        noise = torch.randn((1, unet.config.in_channels, lh, lh), generator=g)
        save(os.path.join(a.out, "latents_in.bin"), noise[0].numpy())
        sched.set_timesteps(a.steps)
        x = noise * sched.init_noise_sigma
        print("timesteps", sched.timesteps.tolist(), "init sigma", float(sched.init_noise_sigma))
        for i, t in enumerate(sched.timesteps):
            xin = sched.scale_model_input(x, t)
            out = unet(xin, t, encoder_hidden_states=cond).sample
            if uncond is not None:
                un = unet(xin, t, encoder_hidden_states=uncond).sample
                out = un + a.guidance * (out - un)
            if i == 0:
                save(os.path.join(a.out, "ref_unet_out0.bin"), out[0].numpy())
            x = sched.step(out, t, x).prev_sample
        save(os.path.join(a.out, "ref_latents_out.bin"), x[0].numpy())
        img = vae.decode(x / vae.config.scaling_factor).sample
        save(os.path.join(a.out, "ref_image.bin"), img[0].numpy())
        try:
            from PIL import Image
            arr = ((img[0].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
            Image.fromarray((arr * 255).round().astype("uint8")).save(os.path.join(a.out, "ref.png"))
        except ImportError:
            pass
    print("wrote reference to", a.out)


def run_edit(a, tok, te, unet, vae, sched):
    """Runs diffusers' editing pipeline, recording the random draws and the stages oxinfer dumps."""
    import torch
    import diffusers
    from PIL import Image
    from diffusers.utils import torch_utils
    import diffusers.models.autoencoders.vae as vae_mod
    from diffusers.pipelines.stable_diffusion import (pipeline_stable_diffusion_img2img as m_i2i,
                                                      pipeline_stable_diffusion_inpaint as m_inp,
                                                      pipeline_stable_diffusion_instruct_pix2pix as m_p2p)

    draws = []
    orig = torch_utils.randn_tensor

    def rec(*args, **kw):
        t = orig(*args, **kw)
        draws.append(t.clone())
        return t
    for m in (vae_mod, m_i2i, m_inp, m_p2p):
        m.randn_tensor = rec

    first = {}
    enc = vae.encode

    def enc_hook(x, *args, **kw):
        out = enc(x, *args, **kw)
        first.setdefault("enc_mean", out.latent_dist.mean[0].detach().clone())
        return out
    vae.encode = enc_hook
    step = sched.step

    def step_hook(model_output, *args, **kw):
        first.setdefault("unet_out0", model_output[0].detach().clone())
        return step(model_output, *args, **kw)
    sched.step = step_hook

    common = dict(vae=vae, text_encoder=te, tokenizer=tok, unet=unet, scheduler=sched, safety_checker=None,
                  feature_extractor=None, requires_safety_checker=False)
    image = Image.open(a.image).convert("RGB")
    kw = dict(prompt=a.prompt, image=image, num_inference_steps=a.steps, guidance_scale=a.guidance,
              generator=torch.Generator().manual_seed(a.seed), output_type="latent")
    if a.negative:
        kw["negative_prompt"] = a.negative
    if a.mode == "img2img":
        pipe = diffusers.StableDiffusionImg2ImgPipeline(**common)
        kw["strength"] = 0.8 if a.strength is None else a.strength
        roles = ["enc_image", "latents_in"]
    elif a.mode == "inpaint":
        pipe = diffusers.StableDiffusionInpaintPipeline(**common)
        kw["mask_image"] = Image.open(a.mask).convert("RGB")
        kw["strength"] = 1.0 if a.strength is None else a.strength
        kw["height"], kw["width"] = image.height, image.width
        if unet.config.in_channels == 9 and kw["strength"] == 1.0:
            roles = ["latents_in", "enc_masked"]
        else:
            roles = ["enc_image", "latents_in", "enc_masked"]
    else:
        pipe = diffusers.StableDiffusionInstructPix2PixPipeline(**common)
        kw["image_guidance_scale"] = a.image_guidance
        roles = ["latents_in"]
    with torch.no_grad():
        ids = tok(a.prompt, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
        save(os.path.join(a.out, "ref_text_cond.bin"), te(ids)[0][0].numpy())
        lat = pipe(**kw).images
        img = vae.decode(lat / vae.config.scaling_factor).sample
    if len(draws) != len(roles):
        raise SystemExit(f"expected {len(roles)} random draws ({roles}), got {len(draws)}")
    for r, d in zip(roles, draws):
        save(os.path.join(a.out, f"{r}.bin"), d[0].numpy())
    save(os.path.join(a.out, "ref_enc_mean.bin"), first["enc_mean"].numpy())
    save(os.path.join(a.out, "ref_unet_out0.bin"), first["unet_out0"].numpy())
    save(os.path.join(a.out, "ref_latents_out.bin"), lat[0].numpy())
    save(os.path.join(a.out, "ref_image.bin"), img[0].numpy())
    arr = ((img[0].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    Image.fromarray((arr * 255).round().astype("uint8")).save(os.path.join(a.out, "ref.png"))
    print("timesteps", [float(t) for t in sched.timesteps], "draws", roles)
    print("wrote reference to", a.out)


if __name__ == "__main__":
    sys.exit(main())
