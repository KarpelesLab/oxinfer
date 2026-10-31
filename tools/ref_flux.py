#!/usr/bin/env python3
"""Reference outputs for the FLUX.1 / FLUX.1 Kontext pipelines, to compare with `OXINFER_FLUX_DUMP`.

Runs diffusers (FluxKontextPipeline / FluxPipeline components) stage by stage, so each part of
oxinfer can be checked on identical inputs. Typical session (OUT is a scratch directory):

  # 1. noise and the preprocessed reference image (writes latents_in.bin, which oxinfer reads)
  python tools/ref_flux.py MODEL OUT --stage inputs --image in.png --size 512x512 --max-area 262144 --no-auto-resize
  # 2. oxinfer, dumping its intermediates into OUT
  OXINFER_FLUX_DUMP=OUT oxinfer --model MODEL --init-image in.png -p "make it night" --size 512x512 \\
      --max-area 262144 --no-auto-resize --steps 1 -o out.png
  # 3. reference stages (each loads only what it needs; run one at a time on small machines)
  python tools/ref_flux.py MODEL OUT --stage text -p "make it night"            # f32 CLIP + T5
  python tools/ref_flux.py MODEL OUT --stage vae                                # f32 VAE encode / decode
  python tools/ref_flux.py MODEL OUT --stage transformer --layers 2,2           # f32, first blocks only
  python tools/ref_flux.py MODEL OUT --stage transformer --dtype bf16           # the whole model
  python tools/ref_flux.py MODEL OUT --stage pipeline -p "make it night" --image in.png ...  # end to end
  python tools/ref_flux.py --compare OUT

The transformer stage reads oxinfer's own text embeddings and reference latents (t5.bin,
pooled.bin, ref_latents.bin) so it isolates the transformer; run oxinfer with
OXINFER_FLUX_LAYERS=2,2 to compare against --layers 2,2.
"""
import argparse, json, math, os, sys
import numpy as np

PREFERRED = [(672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328), (832, 1248), (880, 1184), (944, 1104),
             (1024, 1024), (1104, 944), (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720), (1504, 688),
             (1568, 672)]


def save(path, t):
    np.asarray(t, dtype=np.float32).tofile(path)


def load(path):
    return np.fromfile(path, dtype=np.float32)


def stats(name, a, b):
    if a.shape != b.shape:
        return f"{name:22s} shape {a.shape} vs {b.shape}"
    d = np.abs(a - b)
    rel = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)
    cos = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    return f"{name:22s} max|diff| {d.max():.4g}  rel L2 {rel:.3e}  cos {cos:.7f}"


def psnr(a, b):
    pa = np.round(np.clip((a + 1) / 2, 0, 1) * 255)
    pb = np.round(np.clip((b + 1) / 2, 0, 1) * 255)
    mse = np.mean((pa - pb) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-12))


def compare(out):
    pairs = [("t5.bin", "ref_t5.bin"), ("pooled.bin", "ref_pooled.bin"), ("ref_latents.bin", "ref_ref_latents.bin"),
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


def sizes(a, img_wh):
    """The output (w, h) and reference (w, h) FluxKontextPipeline uses."""
    w, h = a.size or (1024, 1024)
    if img_wh is None:  # text to image (FluxPipeline): no max_area rescaling
        return (w // 16 * 16, h // 16 * 16), None
    ar = w / h
    w, h = round((a.max_area * ar) ** 0.5) // 16 * 16, round((a.max_area / ar) ** 0.5) // 16 * 16
    ref = None
    if img_wh is not None:
        iw, ih = (x - x % 16 for x in img_wh)
        if not a.no_auto_resize:
            ar = iw / ih
            _, iw, ih = min((abs(ar - pw / ph), pw, ph) for pw, ph in PREFERRED)
        ref = (iw // 16 * 16, ih // 16 * 16)
    return (w, h), ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--stage", choices=["inputs", "text", "vae", "transformer", "pipeline"])
    ap.add_argument("-p", "--prompt", default="")
    ap.add_argument("--image")
    ap.add_argument("--size", type=lambda s: tuple(int(x) for x in s.split("x")))
    ap.add_argument("--max-area", type=int, default=1024 ** 2)
    ap.add_argument("--no-auto-resize", action="store_true")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--seq", type=int, default=512, help="T5 max_sequence_length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["f32", "bf16"], default="f32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layers", help="D,S: only the first D double and S single blocks (transformer stage)")
    ap.add_argument("--transformer", help="a GGUF transformer (pipeline stage), loaded with diffusers' GGUF support")
    ap.add_argument("--compare", metavar="DIR")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)

    import torch
    torch.set_grad_enabled(False)
    dt = torch.float32 if a.dtype == "f32" else torch.bfloat16
    os.makedirs(a.out, exist_ok=True)
    P = lambda n: os.path.join(a.out, n)
    from PIL import Image

    img = Image.open(a.image).convert("RGB") if a.image else None
    (w, h), ref_wh = sizes(a, img.size if img else None)

    if a.stage == "inputs":
        g = torch.Generator().manual_seed(a.seed)
        noise = torch.randn((1, 16, 2 * (h // 16), 2 * (w // 16)), generator=g)
        save(P("latents_in.bin"), noise[0].numpy())
        print(f"output {w}x{h}, noise {tuple(noise.shape)}")
        if img is not None:
            from diffusers.image_processor import VaeImageProcessor
            proc = VaeImageProcessor(vae_scale_factor=16)
            r = proc.resize(img, ref_wh[1], ref_wh[0])
            r.save(P("ref_resized.png"))
            x = proc.preprocess(r, ref_wh[1], ref_wh[0])
            save(P("ref_pixels.bin"), x[0].numpy())
            print(f"reference {img.size} -> {ref_wh}")
        return

    if a.stage == "text":
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        tok = CLIPTokenizer.from_pretrained(a.model, subfolder="tokenizer")
        te = CLIPTextModel.from_pretrained(a.model, subfolder="text_encoder", torch_dtype=torch.float32).eval()
        ids = tok([a.prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        save(P("ref_pooled.bin"), te(ids).pooler_output[0].numpy())
        del te
        tok2 = T5TokenizerFast.from_pretrained(a.model, subfolder="tokenizer_2")
        ids2 = tok2([a.prompt], padding="max_length", max_length=a.seq, truncation=True, return_tensors="pt").input_ids
        print("t5 ids", ids2[0, :16].tolist())
        t5 = T5EncoderModel.from_pretrained(a.model, subfolder="text_encoder_2", torch_dtype=dt).eval()
        save(P("ref_t5.bin"), t5(ids2)[0][0].float().numpy())
        return

    if a.stage == "vae":
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.float32).eval()
        sf, sh = vae.config.scaling_factor, vae.config.shift_factor
        if os.path.exists(P("ref_pixels.bin")) and ref_wh is not None:
            x = torch.from_numpy(load(P("ref_pixels.bin")).reshape(1, 3, ref_wh[1], ref_wh[0]))
            z = vae.encode(x).latent_dist.mode()
            save(P("ref_ref_latents.bin"), ((z - sh) * sf)[0].numpy())
        if os.path.exists(P("latents_out.bin")):
            z = torch.from_numpy(load(P("latents_out.bin")).reshape(1, 16, 2 * (h // 16), 2 * (w // 16)))
            save(P("ref_decode_ours.bin"), vae.decode(z / sf + sh).sample[0].numpy())
        return

    if a.stage == "transformer":
        from diffusers import FluxTransformer2DModel
        from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline as KP
        from safetensors import safe_open
        sub = os.path.join(a.model, "transformer")
        cfg = FluxTransformer2DModel.load_config(sub)
        if a.layers:
            d, s = (int(x) for x in a.layers.split(","))
            cfg["num_layers"], cfg["num_single_layers"] = d, s
        if a.transformer and a.layers:
            # A GGUF file, dequantized exactly in f32 with the gguf package (only the blocks kept),
            # renamed by diffusers' single-file converter.
            from gguf import GGUFReader
            from gguf.quants import dequantize
            from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
            d, s = (int(x) for x in a.layers.split(","))
            keep = lambda n: not (n.startswith("double_blocks.") and int(n.split(".")[1]) >= d) and \
                not (n.startswith("single_blocks.") and int(n.split(".")[1]) >= s)
            ck = {t.name: torch.from_numpy(np.ascontiguousarray(
                      dequantize(t.data, t.tensor_type).reshape([int(x) for x in reversed(t.shape)]))).float()
                  for t in GGUFReader(a.transformer).tensors if keep(t.name)}
            with torch.device("meta"):
                m = FluxTransformer2DModel.from_config(cfg)
            m.load_state_dict({k: v.to(dt) for k, v in convert_flux_transformer_checkpoint_to_diffusers(ck).items()},
                              assign=True, strict=True)
        elif a.transformer:
            # A GGUF file (BFL layout): diffusers converts the names and dequantizes on the fly.
            from diffusers import GGUFQuantizationConfig
            m = FluxTransformer2DModel.from_single_file(
                a.transformer, quantization_config=GGUFQuantizationConfig(compute_dtype=dt), torch_dtype=dt,
                config=a.model, subfolder="transformer")
        else:
            with torch.device("meta"):
                m = FluxTransformer2DModel.from_config(cfg)
            idx = json.load(open(os.path.join(sub, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
            want = set(m.state_dict().keys())
            sd = {}
            for shard in sorted(set(idx[k] for k in want)):
                with safe_open(os.path.join(sub, shard), "pt") as f:
                    for k in f.keys():
                        if k in want:
                            sd[k] = f.get_tensor(k).to(dt)
            m.load_state_dict(sd, assign=True, strict=True)
        m = m.eval().to(a.device)
        seq = a.seq
        txt = torch.from_numpy(load(P("t5.bin")).reshape(1, seq, 4096)).to(dt)
        pooled = torch.from_numpy(load(P("pooled.bin")).reshape(1, 768)).to(dt)
        lh, lw = 2 * (h // 16), 2 * (w // 16)
        noise = torch.from_numpy(load(P("latents_in.bin")).reshape(1, 16, lh, lw))
        lat = KP._pack_latents(noise, 1, 16, lh, lw)
        ids = KP._prepare_latent_image_ids(1, lh // 2, lw // 2, "cpu", torch.float32)
        hs = lat
        if ref_wh is not None and os.path.exists(P("ref_latents.bin")):
            rh, rw = ref_wh[1] // 8, ref_wh[0] // 8
            rl = torch.from_numpy(load(P("ref_latents.bin")).reshape(1, 16, rh, rw))
            hs = torch.cat([lat, KP._pack_latents(rl, 1, 16, rh, rw)], dim=1)
            rid = KP._prepare_latent_image_ids(1, rh // 2, rw // 2, "cpu", torch.float32)
            rid[..., 0] = 1
            ids = torch.cat([ids, rid], dim=0)
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift
        sch = FlowMatchEulerDiscreteScheduler.from_pretrained(a.model, subfolder="scheduler")
        mu = calculate_shift(lat.shape[1], sch.config.base_image_seq_len, sch.config.max_image_seq_len,
                             sch.config.base_shift, sch.config.max_shift)
        sch.set_timesteps(sigmas=np.linspace(1.0, 1 / a.steps, a.steps), mu=mu)
        t = sch.timesteps[0]
        guidance = torch.full([1], a.guidance, dtype=torch.float32) if cfg.get("guidance_embeds") else None
        dev = a.device
        out = m(hidden_states=hs.to(dt).to(dev), timestep=(t / 1000).reshape(1).to(dev), guidance=None if guidance is None else guidance.to(dev),
                pooled_projections=pooled.to(dev), encoder_hidden_states=txt.to(dev), txt_ids=torch.zeros(seq, 3).to(dev),
                img_ids=ids.to(dev), return_dict=False)[0]
        save(P("ref_noise_pred0.bin"), out[0, :lat.shape[1]].float().cpu().numpy())
        print("timestep", float(t), "tokens", hs.shape[1] + seq)
        return

    if a.stage == "pipeline":
        from diffusers import FluxKontextPipeline, FluxPipeline
        kw = dict(torch_dtype=dt)
        if a.transformer:
            from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
            kw["transformer"] = FluxTransformer2DModel.from_single_file(
                a.transformer, quantization_config=GGUFQuantizationConfig(compute_dtype=dt), torch_dtype=dt,
                config=a.model, subfolder="transformer")
        cls = FluxKontextPipeline if img is not None else FluxPipeline
        pipe = cls.from_pretrained(a.model, **kw).to(a.device)
        lh, lw = 2 * (h // 16), 2 * (w // 16)
        noise = torch.from_numpy(load(P("latents_in.bin")).reshape(1, 16, lh, lw))
        lat = pipe._pack_latents(noise, 1, 16, lh, lw).to(dt).to(a.device)
        args = dict(prompt=a.prompt, num_inference_steps=a.steps, guidance_scale=a.guidance, latents=lat,
                    max_sequence_length=a.seq, output_type="latent")
        if a.size:
            args.update(width=a.size[0], height=a.size[1])
        if img is not None:
            args.update(image=img, max_area=a.max_area, _auto_resize=not a.no_auto_resize)
        z = pipe(**args).images
        z = pipe._unpack_latents(z, h, w, pipe.vae_scale_factor)
        save(P("ref_latents_out.bin"), z[0].float().cpu().numpy())
        x = pipe.vae.decode(z / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor, return_dict=False)[0]
        x = x[0].float().cpu()
        save(P("ref_image.bin"), x.numpy())
        arr = ((x.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        Image.fromarray((arr * 255).round().astype("uint8")).save(P("ref.png"))
        print("wrote", P("ref.png"))
        return


if __name__ == "__main__":
    sys.exit(main())
