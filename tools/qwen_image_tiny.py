#!/usr/bin/env python3
"""Tiny random-weight Qwen-Image pipelines (diffusers), and the reference runs of the pipelines
oxinfer follows beyond plain editing: image to image, inpainting, ControlNet, layer decomposition
and FP8 "scaled" single-file transformers. The plumbing these add (noising from an image, mask
blending per step, control residuals, layered position ids, FP8 scales) is checked exactly on
small models; `ref` also runs on real folders when memory allows.

usage:
  qwen_image_tiny.py make SRC OUT [--seed S]
      SRC is a real Qwen-Image folder (tokenizer and scheduler config are copied from it). Writes
      OUT/t2i (QwenImagePipeline), OUT/edit (QwenImageEditPipeline), OUT/layered
      (QwenImageLayeredPipeline, RGBA VAE), OUT/controlnet, OUT/controlnet_inpaint
      (QwenImageControlNetModel), and OUT/t2i's transformer as FP8 single files:
      OUT/fp8/tiny_fp8_e4m3fn_scaled_comfyui.safetensors (ComfyUI layout: scalar `scale_weight`,
      `scaled_fp8` marker, every linear layer), OUT/fp8/tiny_fp8_e4m3fn_scaled.safetensors
      (lightx2v layout: `weight_scale` per output row, block weights only), and the weights
      they decode to as diffusers folders (OUT/fp8/comfyui, OUT/fp8/rows) for the reference.
  qwen_image_tiny.py ref MODEL DUMP --mode MODE --prompt P [--negative N --cfg F] [--size WxH]
      [--steps N] [--image IMG] [--mask IMG] [--strength F] [--controlnet DIR --control-image IMG
      --control-scale F --control-start F --control-end F] [--layers N --resolution R]
      [--transformer DIR] [--seed S]
      MODE: t2i | img2img | inpaint | edit-inpaint | controlnet | controlnet-inpaint | layered,
      or caption: the caption the layered pipeline's text encoder writes for --image when it gets
      no prompt (greedy with the repetition penalty 1.05 of Qwen2.5-VL's generation config), to
      compare with the `[caption: ...]` line of `oxinfer --model MODEL --init-image IMG` (no -p).
      Writes the noise the pipeline will draw (DUMP/noise.bin, posterior_eps.bin, control_eps.bin)
      and the reference's prompt embeddings (DUMP/prompt_embeds_in.bin, negative_embeds_in.bin,
      which oxinfer then uses instead of its own), runs the diffusers pipeline (float32), saves
      DUMP/ref_*.npy and compares with oxinfer's dumps when they are there. So: run it, then
        OXINFER_QI_DUMP=DUMP oxinfer --model MODEL ... (the same settings)
      then run it again (or with --compare-only).
"""
import argparse, functools, json, math, os, shutil, sys
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["make", "ref"])
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mode", default="t2i")
ap.add_argument("--prompt", default="a red fox on a bench")
ap.add_argument("--negative")
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--size")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--image")
ap.add_argument("--mask")
ap.add_argument("--strength", type=float)
ap.add_argument("--controlnet")
ap.add_argument("--control-image", action="append", default=[])
ap.add_argument("--control-scale", type=float, default=1.0)
ap.add_argument("--control-start", type=float, default=0.0)
ap.add_argument("--control-end", type=float, default=1.0)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--resolution", type=int, default=640)
ap.add_argument("--no-cfg-normalize", action="store_true")
ap.add_argument("--transformer")
ap.add_argument("--compare-only", action="store_true")
ap.add_argument("--own-embeds", action="store_true", help="do not hand oxinfer the reference's prompt embeddings")
a = ap.parse_args()
torch.set_grad_enabled(False)


def make(src, out, seed):
    from diffusers import (AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImageControlNetModel,
                           QwenImageEditPipeline, QwenImageLayeredPipeline, QwenImagePipeline,
                           QwenImageTransformer2DModel)
    from safetensors.torch import save_file
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
    from transformers import Qwen2VLImageProcessor, Qwen2VLVideoProcessor

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    H = 64  # text hidden size = joint_attention_dim

    def bf16(*models):
        """Weights that bf16 holds exactly: oxinfer keeps matmul weights in bf16, the reference in
        float32, and the comparison is about the pipelines, not that rounding."""
        for m in models:
            for p in m.parameters():
                p.copy_(p.to(torch.bfloat16).float())
        return models[0]

    def randomize(m, std=0.5, only=None):
        """Diffusers zero-initializes some layers (ControlNet outputs); make every weight count."""
        for n, p in m.named_parameters():
            if only is None or any(o in n for o in only):
                p.copy_(torch.randn(p.shape, generator=g) * (std / math.sqrt(max(p.shape[-1], 1)) if p.ndim > 1 else 0.1))

    rc = json.load(open(os.path.join(src, "text_encoder", "config.json")))
    text = dict(rc["text_config"])
    text.pop("layer_types", None)
    text.update(hidden_size=H, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                num_hidden_layers=2, max_window_layers=2, rope_scaling={"mrope_section": [2, 3, 3], "rope_type": "default", "type": "default"})
    vision = dict(rc["vision_config"])
    vision.update(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=H, fullatt_block_indexes=[1])
    top = {k: v for k, v in rc.items() if k in ("image_token_id", "video_token_id", "vision_start_token_id",
                                               "vision_end_token_id", "vision_token_id", "bos_token_id", "eos_token_id")}
    cfg = Qwen2_5_VLConfig(text_config=text, vision_config=vision, **top)
    te = bf16(Qwen2_5_VLForConditionalGeneration(cfg).float().eval())
    tok = Qwen2Tokenizer.from_pretrained(os.path.join(src, "tokenizer"))
    proc = Qwen2VLProcessor(image_processor=Qwen2VLImageProcessor(), tokenizer=tok, video_processor=Qwen2VLVideoProcessor())
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(src, "scheduler"))

    tkw = dict(patch_size=2, in_channels=64, out_channels=16, num_layers=3, attention_head_dim=32,
               num_attention_heads=2, joint_attention_dim=H, axes_dims_rope=(8, 12, 12))
    mean = [round(float(v), 4) for v in torch.randn(16, generator=g) * 0.3]
    std = [round(float(v), 4) for v in 1.0 + torch.rand(16, generator=g)]
    vkw = dict(base_dim=16, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1, latents_mean=mean, latents_std=std)

    def vae(ch):
        return bf16(AutoencoderKLQwenImage(input_channels=ch, **vkw).float().eval())

    tr = QwenImageTransformer2DModel(**tkw).float().eval()
    randomize(tr, only=["proj_out", "norm_out"])
    bf16(tr)
    QwenImagePipeline(scheduler=sched, vae=vae(3), text_encoder=te, tokenizer=tok, transformer=tr).save_pretrained(os.path.join(out, "t2i"))

    tre = bf16(QwenImageTransformer2DModel(**tkw).float().eval())
    QwenImageEditPipeline(scheduler=sched, vae=vae(3), text_encoder=te, tokenizer=tok, processor=proc,
                          transformer=tre).save_pretrained(os.path.join(out, "edit"))

    trl = bf16(QwenImageTransformer2DModel(use_additional_t_cond=True, use_layer3d_rope=True, **tkw).float().eval())
    lcfg = dict(sched.config)
    lcfg.update(time_shift_type="linear", shift_terminal=None)
    QwenImageLayeredPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_config(lcfg), vae=vae(4), text_encoder=te,
                             tokenizer=tok, processor=proc, transformer=trl).save_pretrained(os.path.join(out, "layered"))

    for name, extra in (("controlnet", 0), ("controlnet_inpaint", 4)):
        ckw = dict(tkw)
        ckw.update(num_layers=2, extra_condition_channels=extra)
        cn = QwenImageControlNetModel(**ckw).float().eval()
        randomize(cn, only=["controlnet_blocks", "controlnet_x_embedder"])
        bf16(cn).save_pretrained(os.path.join(out, name))

    # FP8 single files of the t2i transformer, and what they decode to.
    fdir = os.path.join(out, "fp8")
    os.makedirs(fdir, exist_ok=True)
    sd = {k: v.float() for k, v in tr.state_dict().items()}
    f8 = torch.float8_e4m3fn
    fmax = torch.finfo(f8).max
    comfy, rows, dec_comfy, dec_rows = {}, {}, {}, {}
    for k, v in sd.items():
        lin = k.endswith(".weight") and v.ndim == 2
        if lin:
            s = (v.abs().max() / fmax).to(torch.bfloat16)
            q = (v / s.float()).to(f8)
            comfy[k] = q
            comfy[k[: -len("weight")] + "scale_weight"] = s
            dec_comfy[k] = q.float() * s.float()
        else:
            comfy[k] = v.to(torch.bfloat16)
            dec_comfy[k] = comfy[k].float()
        if lin and k.startswith("transformer_blocks."):
            s = (v.abs().amax(dim=1, keepdim=True) / fmax).to(torch.bfloat16)
            q = (v / s.float()).to(f8)
            rows[k] = q
            rows[k + "_scale"] = s
            dec_rows[k] = q.float() * s.float()
        else:
            rows[k] = v.to(torch.bfloat16)
            dec_rows[k] = rows[k].float()
    comfy["scaled_fp8"] = torch.zeros(2, dtype=f8)
    save_file({"model.diffusion_model." + k: v for k, v in comfy.items()}, os.path.join(fdir, "tiny_fp8_e4m3fn_scaled_comfyui.safetensors"))
    save_file(rows, os.path.join(fdir, "tiny_fp8_e4m3fn_scaled.safetensors"))
    for name, dec in (("comfyui", dec_comfy), ("rows", dec_rows)):
        m = QwenImageTransformer2DModel(**tkw).float().eval()
        m.load_state_dict(dec)
        m.save_pretrained(os.path.join(fdir, name))
    print("wrote", out)


def wbin(d, name, t):
    t.detach().float().numpy().astype("<f4").tofile(os.path.join(d, name))


def report(d, name, ours, ref):
    p = os.path.join(d, ours)
    if not os.path.exists(p):
        print(f"{name}: no oxinfer dump ({ours})")
        return
    o = np.fromfile(p, dtype="<f4").astype(np.float64)
    r = np.asarray(ref, np.float64).ravel()
    if o.size != r.size:
        print(f"{name}: size ours {o.size} vs ref {r.size}")
        return
    rel = np.linalg.norm(o - r) / max(np.linalg.norm(r), 1e-30)
    print(f"{name}: rel L2 {rel:.3e}, max abs {np.abs(o - r).max():.4e} (ref max |x| {np.abs(r).max():.3f})")


def ref(M, D):
    import diffusers
    from diffusers import (QwenImageControlNetInpaintPipeline, QwenImageControlNetModel, QwenImageControlNetPipeline,
                           QwenImageEditInpaintPipeline, QwenImageImg2ImgPipeline, QwenImageInpaintPipeline,
                           QwenImageLayeredPipeline, QwenImagePipeline, QwenImageTransformer2DModel)
    from PIL import Image

    os.makedirs(D, exist_ok=True)
    names = ["prompt_embeds", "v0", "latents_out", "image", "control_res", "x0"]
    if a.compare_only:
        for n in names:
            f = os.path.join(D, f"ref_{n}.npy")
            if os.path.exists(f):
                report(D, n, f"{n}.bin", np.load(f))
        return
    if a.mode == "caption":
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_layered import calculate_dimensions
        pipe = QwenImageLayeredPipeline.from_pretrained(M, torch_dtype=torch.float32)
        te, img_tok = pipe.text_encoder, pipe.text_encoder.config.image_token_id
        fwd = te.forward

        @functools.wraps(fwd)
        def forward(*args, **kws):
            if kws.get("mm_token_type_ids") is None and kws.get("input_ids") is not None:
                kws["mm_token_type_ids"] = (kws["input_ids"] == img_tok).long()
            return fwd(*args, **kws)

        te.forward = forward
        te.generation_config.update(do_sample=False, repetition_penalty=1.05, temperature=None, top_k=None, top_p=None,
                                    eos_token_id=[151645, 151643], pad_token_id=151643)
        img = Image.open(a.image).convert("RGB")
        w, h = calculate_dimensions(a.resolution * a.resolution, img.size[0] / img.size[1])
        cap = pipe.get_image_caption(pipe.image_processor.resize(img, h, w), use_en_prompt=True, device=torch.device("cpu"))
        print("caption:", json.dumps(cap, ensure_ascii=False))
        return
    cls = {"t2i": QwenImagePipeline, "img2img": QwenImageImg2ImgPipeline, "inpaint": QwenImageInpaintPipeline,
           "edit-inpaint": QwenImageEditInpaintPipeline, "controlnet": QwenImageControlNetPipeline,
           "controlnet-inpaint": QwenImageControlNetInpaintPipeline, "layered": QwenImageLayeredPipeline}[a.mode]
    kw = {}
    if a.mode.startswith("controlnet"):
        kw["controlnet"] = QwenImageControlNetModel.from_pretrained(a.controlnet, torch_dtype=torch.float32)
    if a.transformer:
        kw["transformer"] = QwenImageTransformer2DModel.from_pretrained(a.transformer, torch_dtype=torch.float32)
    pipe = cls.from_pretrained(M, torch_dtype=torch.float32, **kw)
    pipe.set_progress_bar_config(disable=True)
    te = pipe.text_encoder
    img_tok = te.config.image_token_id
    fwd = te.forward

    def forward(*args, **kws):
        # transformers 5 needs the token types for M-RoPE over image tokens (see ref_qwen_image.py).
        if kws.get("mm_token_type_ids") is None and kws.get("input_ids") is not None:
            kws["mm_token_type_ids"] = (kws["input_ids"] == img_tok).long()
        return fwd(*args, **kws)

    te.forward = forward
    img = Image.open(a.image).convert("RGB") if a.image else None
    mask = Image.open(a.mask).convert("L") if a.mask else None
    ctrl = [Image.open(p).convert("RGB") for p in a.control_image]
    if a.size:
        w, h = map(int, a.size.lower().split("x"))
    elif a.mode == "layered":
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_layered import calculate_dimensions
        w, h = calculate_dimensions(a.resolution * a.resolution, img.size[0] / img.size[1])
    else:
        w, h = (img or ctrl[0]).size
    w, h = w // 16 * 16, h // 16 * 16
    lh, lw = h // 8, w // 8

    # The noise the pipeline will draw from its generator, in its order.
    g = torch.Generator().manual_seed(a.seed)
    draw = lambda *shape: torch.randn(shape, generator=g, dtype=torch.float32)
    if a.mode in ("img2img", "inpaint", "edit-inpaint"):
        wbin(D, "posterior_eps.bin", draw(1, 16, 1, lh, lw))
        wbin(D, "noise.bin", draw(1, 1, 16, lh, lw))
    elif a.mode == "controlnet":
        for i in range(len(ctrl)):
            wbin(D, "control_eps.bin" if i == 0 else f"control_eps{i}.bin", draw(1, 16, 1, lh, lw))
        wbin(D, "noise.bin", draw(1, 1, 16, lh, lw))
    elif a.mode == "controlnet-inpaint":
        torch.manual_seed(a.seed + 1000)  # `latent_dist.sample()` without a generator: the global one
        wbin(D, "control_eps.bin", torch.randn((1, 16, 1, lh, lw), dtype=torch.float32))
        wbin(D, "noise.bin", draw(1, 1, 16, lh, lw))
    elif a.mode == "layered":
        wbin(D, "noise.bin", draw(1, a.layers + 1, 16, lh, lw))
    else:
        wbin(D, "noise.bin", draw(1, 1, 16, lh, lw))

    cap = {}

    def first(mod, key, pick):
        def hook(m, args, kwargs, out):
            if key not in cap:
                cap[key] = pick(out, kwargs)
        mod.register_forward_hook(hook, with_kwargs=True)

    first(pipe.transformer, "v0", lambda out, kws: out[0][0])
    first(pipe.transformer, "x0", lambda out, kws: kws["hidden_states"][0])
    if "controlnet" in kw:
        first(pipe.controlnet, "control_res", lambda out, kws: torch.cat([o[0].reshape(-1) for o in out]))
    emb = {}
    enc = pipe.encode_prompt

    def encode_prompt(*args, **kws):
        e, m = enc(*args, **kws)
        emb.setdefault("list", []).append(e[0])
        return e, m

    pipe.encode_prompt = encode_prompt
    call = dict(prompt=a.prompt, num_inference_steps=a.steps, true_cfg_scale=a.cfg, output_type="latent",
                generator=torch.Generator().manual_seed(a.seed))
    if a.negative is not None:
        call["negative_prompt"] = a.negative
    if a.mode != "layered":
        call.update(width=w, height=h)
    if a.mode in ("img2img", "inpaint", "edit-inpaint"):
        call.update(image=img, strength=a.strength if a.strength is not None else 0.6)
    if a.mode in ("inpaint", "edit-inpaint"):
        call.update(mask_image=mask)
    if a.mode == "controlnet":
        call.update(control_image=ctrl if len(ctrl) > 1 else ctrl[0],
                    controlnet_conditioning_scale=[a.control_scale] * len(ctrl) if len(ctrl) > 1 else a.control_scale,
                    control_guidance_start=a.control_start, control_guidance_end=a.control_end)
        if len(ctrl) > 1:
            from diffusers import QwenImageMultiControlNetModel
            pipe.controlnet = QwenImageMultiControlNetModel([pipe.controlnet])
    if a.mode == "controlnet-inpaint":
        call.update(control_image=ctrl[0], control_mask=mask, controlnet_conditioning_scale=a.control_scale,
                    control_guidance_start=a.control_start, control_guidance_end=a.control_end)
        torch.manual_seed(a.seed + 1000)
    if a.mode == "layered":
        call.update(image=img.convert("RGBA"), layers=a.layers, resolution=a.resolution,
                    cfg_normalize=not a.no_cfg_normalize)
        # (diffusers 0.42's layered pipeline fails with output_type="latent": take the latents
        # from the step callback and let it decode.)
        last = {}

        def on_step(p, i, t, kws):
            last["latents"] = kws["latents"]
            return {}

        call.update(output_type="pt", callback_on_step_end=on_step)
        pipe(**call)
        lat = last["latents"]
    else:
        lat = pipe(**call).images
    # v0 as oxinfer dumps it: the guided velocity is not captured by the hook, so only without CFG.
    out = {"prompt_embeds": emb["list"][0], "latents_out": lat[0], "x0": cap["x0"][: lat.shape[1]]}
    if a.cfg <= 1 or a.negative is None:
        out["v0"] = cap["v0"][: lat.shape[1]]
    if "control_res" in cap and len(ctrl) == 1:
        out["control_res"] = cap["control_res"]
    if not a.own_embeds:
        wbin(D, "prompt_embeds_in.bin", emb["list"][0])
        if len(emb["list"]) > 1:
            wbin(D, "negative_embeds_in.bin", emb["list"][1])
    # Decode as the pipeline does.
    vae = pipe.vae
    mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
    if a.mode == "layered":
        z = pipe._unpack_latents(lat, h, w, a.layers, 8) * std + mean
        z = z[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(-1, 16, 1, lh, lw)
        px = vae.decode(z, return_dict=False)[0][:, :, 0]  # [layers, 4, h, w]
        out["image"] = px.permute(0, 2, 3, 1)
        u8 = np.round(np.clip(px.permute(0, 2, 3, 1).numpy() * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
        for i, l in enumerate(u8):
            Image.fromarray(l, "RGBA").save(os.path.join(D, f"ref_layer_{i}.png"))
    else:
        z = pipe._unpack_latents(lat, h, w, 8) * std + mean
        px = vae.decode(z, return_dict=False)[0][:, :, 0][0].permute(1, 2, 0)
        out["image"] = px
        u8 = np.round(np.clip(px.numpy() * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(u8).save(os.path.join(D, "ref_image.png"))
    print(f"{a.mode}: {w}x{h}, {a.steps} steps, latents {tuple(lat.shape)}")
    for n in names:
        if n in out:
            v = out[n].float().numpy()
            np.save(os.path.join(D, f"ref_{n}.npy"), v)
            report(D, n, f"{n}.bin", v)
    o = os.path.join(D, "image.bin")
    if os.path.exists(o):
        o8 = np.round(np.clip(np.fromfile(o, dtype="<f4").reshape(u8.shape) * 0.5 + 0.5, 0, 1) * 255)
        mse = np.mean((o8 - u8.astype(np.float64)) ** 2)
        print(f"image: PSNR {10 * np.log10(255 ** 2 / max(mse, 1e-12)):.2f} dB, max pixel difference {int(np.abs(o8 - u8).max())}")


if a.cmd == "make":
    make(a.a, a.b, a.seed)
else:
    ref(a.a, a.b)
