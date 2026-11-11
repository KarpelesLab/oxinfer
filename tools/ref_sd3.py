#!/usr/bin/env python3
"""Reference outputs for the Stable Diffusion 3 / 3.5 pipelines, to compare with `OXINFER_SD3_DUMP`.

Runs diffusers (StableDiffusion3Pipeline / StableDiffusion3Img2ImgPipeline components) stage by
stage, so each part of oxinfer can be checked on identical inputs. Typical session (OUT is a
scratch directory, MODEL a diffusers SD 3 folder):

  # 1. the initial noise (writes latents_in.bin, which oxinfer reads)
  python tools/ref_sd3.py MODEL OUT --stage inputs --size 512x512
  # 2. oxinfer, dumping its intermediates into OUT
  OXINFER_SD3_DUMP=OUT oxinfer --model MODEL -p "a red fox in the snow" --size 512x512 --steps 4 -o out.png
  # 3. reference stages (each loads only what it needs)
  python tools/ref_sd3.py MODEL OUT --stage text -p "a red fox in the snow"     # f32 CLIP-L, CLIP-G, T5
  python tools/ref_sd3.py MODEL OUT --stage transformer --size 512x512          # one f32 pass
  python tools/ref_sd3.py MODEL OUT --stage vae --size 512x512                  # f32 VAE decode
  python tools/ref_sd3.py MODEL OUT --stage pipeline -p "a red fox in the snow" --size 512x512 --steps 4
  python tools/ref_sd3.py --compare OUT

The transformer stage reads oxinfer's own text conditioning (prompt_embeds.bin, pooled.bin) and
first timestep (sigmas.bin), so it isolates the transformer; the vae stage decodes oxinfer's
final latents. `--no-t5` drops the T5 encoder on both sides (oxinfer: `--no-t5`). Image to image:
pass `--image FILE --strength F` to the inputs and pipeline stages (the image must already have
the output size; inputs also writes posterior_noise_in.bin, the noise of the VAE sample).
`--transformer FILE.gguf` runs the pipeline stage with a GGUF transformer through diffusers'
GGUF loader.
"""
import argparse, os
import numpy as np


def save(path, t):
    np.asarray(t, dtype=np.float32).tofile(path)


def load(path):
    return np.fromfile(path, dtype=np.float32)


def stats(name, a, b):
    if a.shape != b.shape:
        return f"{name:34s} shape {a.shape} vs {b.shape}"
    d = np.abs(a - b)
    rel = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)
    cos = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    return f"{name:34s} max|diff| {d.max():.4g}  rel L2 {rel:.3e}  cos {cos:.7f}"


def psnr(a, b):
    pa = np.round(np.clip((a + 1) / 2, 0, 1) * 255)
    pb = np.round(np.clip((b + 1) / 2, 0, 1) * 255)
    mse = np.mean((pa - pb) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-12))


def compare(out):
    pairs = [("prompt_embeds.bin", "ref_prompt_embeds.bin"), ("pooled.bin", "ref_pooled.bin"),
             ("neg_prompt_embeds.bin", "ref_neg_prompt_embeds.bin"), ("neg_pooled.bin", "ref_neg_pooled.bin"),
             ("sigmas.bin", "ref_sigmas.bin"), ("init_latents.bin", "ref_init_latents.bin"),
             ("noise_pred0.bin", "ref_noise_pred0.bin"), ("image.bin", "ref_decode_ours.bin"),
             ("latents_out.bin", "ref_latents_out.bin"), ("image.bin", "ref_image.bin")]
    for ours, ref in pairs:
        pa, pb = os.path.join(out, ours), os.path.join(out, ref)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        a, b = load(pa), load(pb)
        line = stats(f"{ours} / {ref[4:]}", a, b)
        if ours == "image.bin" and a.shape == b.shape:
            line += f"  PSNR {psnr(a, b):.1f} dB (8-bit)"
        print(line)
        if ours == "prompt_embeds.bin" and a.shape == b.shape and a.size % 4096 == 0:
            a2, b2 = a.reshape(-1, 4096), b.reshape(-1, 4096)
            print(stats("  CLIP-L penultimate", a2[:77, :768].ravel(), b2[:77, :768].ravel()))
            print(stats("  CLIP-G penultimate", a2[:77, 768:2048].ravel(), b2[:77, 768:2048].ravel()))
            if np.abs(b2[77:]).max() > 0:
                print(stats("  T5", a2[77:].ravel(), b2[77:].ravel()))
        if ours == "pooled.bin" and a.shape == b.shape and a.size == 2048:
            print(stats("  CLIP-L pooled", a[:768], b[:768]))
            print(stats("  CLIP-G pooled", a[768:], b[768:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--stage", choices=["inputs", "text", "vae", "transformer", "pipeline"])
    ap.add_argument("-p", "--prompt", default="")
    ap.add_argument("--negative", default="")
    ap.add_argument("--image", help="image to image: the input, already at the output size")
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--size", type=lambda s: tuple(int(x) for x in s.split("x")), default=(1024, 1024))
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seq", type=int, default=256, help="T5 max_sequence_length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-t5", action="store_true")
    ap.add_argument("--dtype", choices=["f32", "bf16"], default="f32")
    ap.add_argument("--transformer", help="a GGUF transformer (transformer / pipeline stages)")
    ap.add_argument("--compare", metavar="DIR")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)

    import torch
    torch.set_grad_enabled(False)
    dt = torch.float32 if a.dtype == "f32" else torch.bfloat16
    os.makedirs(a.out, exist_ok=True)
    P = lambda n: os.path.join(a.out, n)
    w, h = (x // 16 * 16 for x in a.size)
    shape = (1, 16, h // 8, w // 8)

    def pixels():
        from PIL import Image
        from diffusers.image_processor import VaeImageProcessor
        img = Image.open(a.image).convert("RGB")
        assert img.size == (w, h), f"--image must be {w}x{h} (is {img.size})"
        return VaeImageProcessor(vae_scale_factor=16).preprocess(img, height=h, width=w)

    def transformer():
        from diffusers import SD3Transformer2DModel
        if a.transformer:
            from diffusers import GGUFQuantizationConfig
            return SD3Transformer2DModel.from_single_file(
                a.transformer, quantization_config=GGUFQuantizationConfig(compute_dtype=dt), torch_dtype=dt,
                config=a.model, subfolder="transformer").eval()
        return SD3Transformer2DModel.from_pretrained(a.model, subfolder="transformer", torch_dtype=dt).eval()

    def pipe_kwargs():
        kw = dict(torch_dtype=torch.float32)
        if a.no_t5:
            kw.update(text_encoder_3=None, tokenizer_3=None)
        return kw

    if a.stage == "inputs":
        g = torch.Generator().manual_seed(a.seed)
        if a.image:
            # The img2img pipeline samples the VAE posterior before drawing the noise.
            save(P("posterior_noise_in.bin"), torch.randn(shape, generator=g)[0].numpy())
        save(P("latents_in.bin"), torch.randn(shape, generator=g)[0].numpy())
        print(f"output {w}x{h}, noise {shape}")
        return

    if a.stage == "text":
        from diffusers import StableDiffusion3Pipeline
        kw = pipe_kwargs()
        if not a.no_t5:  # without T5 the pipeline reads the transformer's config for the zeros
            kw["transformer"] = None
        pipe = StableDiffusion3Pipeline.from_pretrained(a.model, vae=None, **kw)
        pe, ne, pp, npp = pipe.encode_prompt(a.prompt, a.prompt, a.prompt, negative_prompt=a.negative,
                                             do_classifier_free_guidance=True, max_sequence_length=a.seq)
        save(P("ref_prompt_embeds.bin"), pe[0].float().numpy())
        save(P("ref_pooled.bin"), pp[0].float().numpy())
        save(P("ref_neg_prompt_embeds.bin"), ne[0].float().numpy())
        save(P("ref_neg_pooled.bin"), npp[0].float().numpy())
        print("prompt_embeds", tuple(pe.shape), "pooled", tuple(pp.shape))
        return

    if a.stage == "transformer":
        tr = transformer()
        x = torch.from_numpy(load(P("latents_in.bin")).reshape(shape)).to(dt)
        pe = torch.from_numpy(load(P("prompt_embeds.bin")).reshape(1, -1, tr.config.joint_attention_dim)).to(dt)
        pp = torch.from_numpy(load(P("pooled.bin")).reshape(1, -1)).to(dt)
        t = torch.tensor([load(P("sigmas.bin"))[0] * 1000.0]).to(dt)
        y = tr(hidden_states=x, timestep=t, encoder_hidden_states=pe, pooled_projections=pp, return_dict=False)[0]
        save(P("ref_noise_pred0.bin"), y[0].float().numpy())
        print("timestep", float(t[0]), "tokens", pe.shape[1])
        return

    if a.stage == "vae":
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.float32).eval()
        z = torch.from_numpy(load(P("latents_out.bin")).reshape(shape))
        img = vae.decode(z / vae.config.scaling_factor + vae.config.shift_factor, return_dict=False)[0]
        save(P("ref_decode_ours.bin"), img[0].numpy())
        return

    if a.stage == "pipeline":
        from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline
        kw = pipe_kwargs()
        if a.transformer:
            kw["transformer"] = transformer()
        cls = StableDiffusion3Img2ImgPipeline if a.image else StableDiffusion3Pipeline
        pipe = cls.from_pretrained(a.model, **kw)
        call = dict(prompt=a.prompt, negative_prompt=a.negative, num_inference_steps=a.steps,
                    guidance_scale=a.guidance, max_sequence_length=a.seq, output_type="latent")
        if a.image:
            # The same draws as the inputs stage: posterior sample, then the noise.
            vae = pipe.vae
            z0 = vae.encode(pixels()).latent_dist.sample(torch.Generator().manual_seed(a.seed))
            save(P("ref_init_latents.bin"), ((z0 - vae.config.shift_factor) * vae.config.scaling_factor)[0].numpy())
            # (diffusers resizes the input to 1024x1024 unless told the size.)
            call.update(image=pixels(), height=h, width=w, strength=a.strength,
                        generator=torch.Generator().manual_seed(a.seed))
        else:
            call.update(height=h, width=w, latents=torch.from_numpy(load(P("latents_in.bin")).reshape(shape)))
        z = pipe(**call).images
        save(P("ref_sigmas.bin"), pipe.scheduler.sigmas.numpy())
        save(P("ref_latents_out.bin"), z[0].numpy())
        vae = pipe.vae
        img = vae.decode(z / vae.config.scaling_factor + vae.config.shift_factor, return_dict=False)[0]
        save(P("ref_image.bin"), img[0].numpy())
        from PIL import Image
        Image.fromarray(np.round(np.clip((img[0].permute(1, 2, 0).numpy() + 1) / 2, 0, 1) * 255).astype(np.uint8)).save(P("ref.png"))
        return

    ap.error("--stage or --compare is required")


if __name__ == "__main__":
    main()
