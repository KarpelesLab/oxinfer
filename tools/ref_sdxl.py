#!/usr/bin/env python3
"""Reference outputs for the Stable Diffusion XL pipeline, to compare with `OXINFER_SD_DUMP`.

The SDXL counterpart of tools/ref_sd.py: runs diffusers' own pipelines (StableDiffusionXLPipeline,
StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline) in float32 (fp16 weights
upcast) with the Euler scheduler built from the model's scheduler config, and records the stages
oxinfer dumps: the text context (penultimate hidden states of both encoders, [77, 2048]), the
pooled embedding of the second encoder, the first (guided) UNet output, the final latents and the
decoded image. Every random draw is saved by role (latents_in.bin for the initial noise,
enc_image.bin / enc_masked.bin for the VAE samples), which oxinfer reads from the same directory
instead of drawing its own.

  python tools/ref_sdxl.py MODEL OUT "a red fox" --steps 4 --guidance 5 --size 512 --bf16-weights
  OXINFER_SD_DUMP=OUT oxinfer --model MODEL -p "a red fox" --steps 4 --guidance 5 --size 512x512 \
      --sampler euler --device cpu
  python tools/ref_sdxl.py --compare OUT

--mode img2img | inpaint take --image (and --mask) already at the working size; --lora
FILE[:SCALE] (repeatable) fuses LoRA files with diffusers' load_lora_weights (needs peft) before
any bf16 rounding, as oxinfer merges them at load. --bf16-weights rounds the matmul / convolution
weights to bf16 as oxinfer stores them, which isolates the implementation error.
"""
import argparse, os, sys
import numpy as np


def save(path, t):
    np.asarray(t, dtype=np.float32).tofile(path)


def compare(out):
    pairs = [("text_cond.bin", "ref_text_cond.bin"), ("text_pooled.bin", "ref_text_pooled.bin"), ("enc_mean.bin", "ref_enc_mean.bin"),
             ("unet_out0.bin", "ref_unet_out0.bin"), ("latents_out.bin", "ref_latents_out.bin"), ("image.bin", "ref_image.bin")]
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
    ap.add_argument("--size", type=int, default=1024, help="image side (txt2img); edits run at the input image's size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", metavar="DIR")
    ap.add_argument("--lora", action="append", default=[], metavar="FILE[:SCALE]", help="LoRA to fuse (repeatable)")
    ap.add_argument("--mode", default="txt2img", choices=["txt2img", "img2img", "inpaint"])
    ap.add_argument("--image", help="input image (img2img / inpaint), at the working size")
    ap.add_argument("--mask", help="inpainting mask (white = repaint)")
    ap.add_argument("--strength", type=float, default=None)
    ap.add_argument("--spacing", help="override the scheduler's timestep_spacing (e.g. trailing for SDXL-Lightning)")
    ap.add_argument("--bf16-weights", action="store_true", help="round matmul/conv weights to bf16 as oxinfer stores them")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)

    import torch
    import diffusers
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    from diffusers.utils import torch_utils
    import diffusers.models.autoencoders.vae as vae_mod
    from diffusers.pipelines.stable_diffusion_xl import (pipeline_stable_diffusion_xl as m_t2i,
                                                         pipeline_stable_diffusion_xl_img2img as m_i2i,
                                                         pipeline_stable_diffusion_xl_inpaint as m_inp)
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from PIL import Image

    os.makedirs(a.out, exist_ok=True)
    variant = "fp16" if os.path.exists(os.path.join(a.model, "unet", "diffusion_pytorch_model.fp16.safetensors")) else None
    kw = dict(torch_dtype=torch.float32, variant=variant)
    tok = CLIPTokenizer.from_pretrained(a.model, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(a.model, subfolder="tokenizer_2")
    te = CLIPTextModel.from_pretrained(a.model, subfolder="text_encoder", **kw).eval()
    te2 = CLIPTextModelWithProjection.from_pretrained(a.model, subfolder="text_encoder_2", **kw).eval()
    unet = UNet2DConditionModel.from_pretrained(a.model, subfolder="unet", **kw).eval()
    vae = AutoencoderKL.from_pretrained(a.model, subfolder="vae", **kw).eval()
    sched = EulerDiscreteScheduler.from_pretrained(a.model, subfolder="scheduler")
    if a.spacing:
        sched = EulerDiscreteScheduler.from_config(sched.config, timestep_spacing=a.spacing)
    import json
    with open(os.path.join(a.model, "model_index.json")) as f:
        force_zeros = json.load(f).get("force_zeros_for_empty_prompt", True)
    common = dict(vae=vae, text_encoder=te, text_encoder_2=te2, tokenizer=tok, tokenizer_2=tok2, unet=unet, scheduler=sched,
                  force_zeros_for_empty_prompt=force_zeros, add_watermarker=False)
    if a.lora:
        pipe = diffusers.StableDiffusionXLPipeline(**common)
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
        common.update(text_encoder=pipe.text_encoder.eval(), text_encoder_2=pipe.text_encoder_2.eval(), unet=pipe.unet.eval())
        te, te2, unet = common["text_encoder"], common["text_encoder_2"], common["unet"]
    if a.bf16_weights:
        # oxinfer keeps Linear/Conv weights in bf16 (biases, norms and embeddings stay exact).
        for m in (te, te2, unet, vae):
            for mod in m.modules():
                if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
                    mod.weight.data = mod.weight.data.to(torch.bfloat16).to(torch.float32)

    # Record every random draw and the first model output.
    draws = []
    orig = torch_utils.randn_tensor

    def rec(*args, **kw):
        t = orig(*args, **kw)
        draws.append(t.clone())
        return t
    for m in (vae_mod, m_t2i, m_i2i, m_inp):
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

    g = torch.Generator().manual_seed(a.seed)
    kw = dict(prompt=a.prompt, num_inference_steps=a.steps, guidance_scale=a.guidance, generator=g, output_type="latent")
    if a.negative:
        kw["negative_prompt"] = a.negative
    if a.mode == "txt2img":
        pipe = diffusers.StableDiffusionXLPipeline(**common)
        lh = a.size // 8
        noise = torch.randn((1, unet.config.in_channels, lh, lh), generator=g)
        kw.update(latents=noise, height=a.size, width=a.size)
        roles = []
    else:
        image = Image.open(a.image).convert("RGB")
        kw["image"] = image
        if a.mode == "img2img":
            pipe = diffusers.StableDiffusionXLImg2ImgPipeline(**common)
            kw["strength"] = 0.8 if a.strength is None else a.strength
            roles = ["enc_image", "latents_in"]
        else:
            pipe = diffusers.StableDiffusionXLInpaintPipeline(**common)
            kw["mask_image"] = Image.open(a.mask).convert("RGB")
            kw["strength"] = 1.0 if a.strength is None else a.strength
            kw["height"], kw["width"] = image.height, image.width
            roles = None  # checked against what the pipeline drew, below
    with torch.no_grad():
        pe, _, pooled, _ = pipe.encode_prompt(a.prompt, do_classifier_free_guidance=False)
        save(os.path.join(a.out, "ref_text_cond.bin"), pe[0].numpy())
        save(os.path.join(a.out, "ref_text_pooled.bin"), pooled[0].numpy())
        lat = pipe(**kw).images
        img = vae.decode(lat / vae.config.scaling_factor).sample
    if a.mode == "txt2img":
        save(os.path.join(a.out, "latents_in.bin"), noise[0].numpy())
    else:
        if roles is None:
            # StableDiffusionXLInpaintPipeline: the image latents, the noise, the masked image's.
            # A text-to-image UNet blends instead: its masked-image draw is not used.
            if unet.config.in_channels == 4:
                roles = ["enc_image", "latents_in", None]
            elif kw["strength"] == 1.0:
                roles = ["latents_in", "enc_masked"]
            else:
                roles = ["enc_image", "latents_in", "enc_masked"]
        if len(draws) != len(roles):
            raise SystemExit(f"expected {len(roles)} random draws ({roles}), got {len(draws)}: {[tuple(d.shape) for d in draws]}")
        for r, d in zip(roles, draws):
            if r:
                save(os.path.join(a.out, f"{r}.bin"), d[0].numpy())
        save(os.path.join(a.out, "ref_enc_mean.bin"), first["enc_mean"].numpy())
    save(os.path.join(a.out, "ref_unet_out0.bin"), first["unet_out0"].numpy())
    save(os.path.join(a.out, "ref_latents_out.bin"), lat[0].numpy())
    save(os.path.join(a.out, "ref_image.bin"), img[0].numpy())
    arr = ((img[0].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    Image.fromarray((arr * 255).round().astype("uint8")).save(os.path.join(a.out, "ref.png"))
    print("timesteps", [float(t) for t in sched.timesteps], "init sigma", float(sched.init_noise_sigma), "draws", roles)
    print("wrote reference to", a.out)


if __name__ == "__main__":
    sys.exit(main())
