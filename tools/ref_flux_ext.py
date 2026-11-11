#!/usr/bin/env python3
"""Checks oxinfer's FLUX.1 modes beyond plain text to image / single-image Kontext against
diffusers, end to end on the same noise: several Kontext references, image to image, inpainting,
FLUX.1 Fill, FLUX.1 Canny / Depth (native and as a BFL control LoRA), LoRA files in every key
naming, Redux image prompts and BFL single-file checkpoints.

It runs both sides itself: writes the noise (`latents_in.bin`, which oxinfer reads through
OXINFER_FLUX_DUMP), runs the oxinfer binary, runs the matching diffusers pipeline in float32 and
prints the relative L2 distance of the final latents and the PSNR of the decoded images.

  python tools/ref_flux_ext.py MODEL OUT --oxinfer target/release/oxinfer --mode MODE [options]

MODEL is a diffusers FLUX folder: the real one, or a tiny random-weight one from
`tools/flux_tiny.py` (exact plumbing checks in seconds; `--mode all` runs every check a folder's
transformer allows, generating its own input images and LoRA files).

Modes:
  t2i       FluxPipeline
  multi     Kontext with --image given several times; --layout diffusers compares with diffusers'
            community pipeline's token ids (examples/community/
            pipeline_flux_kontext_multiple_images.py; --community FILE runs that file itself),
            offset / index with ComfyUI's (transcribed in `comfy_positions` below); the reference
            is a Kontext denoising loop over those ids
  img2img   FluxImg2ImgPipeline (--image, --strength)
  inpaint   FluxInpaintPipeline (--image, --mask, --strength)
  fill      FluxFillPipeline (--image, --mask), MODEL's transformer taking 384 channels
  control   FluxControlPipeline (--control), MODEL's transformer taking 128 channels, or a
            64-channel one with a BFL control LoRA (--lora)
  lora      FluxPipeline with --lora FILE[:SCALE] (repeatable)
  redux     FluxPriorReduxPipeline (--redux DIR, --image) into FluxPipeline
  bfl       oxinfer loads --bfl FILE (single-file transformer) and --ae FILE over MODEL's other
            parts; diffusers runs MODEL
  step      real weights within a small memory budget: one transformer call of a GGUF
            transformer (--gguf FILE, MODEL gives the other parts) with --lora files (and
            --control for a control LoRA), compared with diffusers' GGUF loader + load_lora_weights
            on oxinfer's own text embeddings and control latents (only the transformer is loaded
            on the Python side)
  redux-tokens  the Redux tokens of --image (SigLIP + projector of --redux) against transformers,
            from a one-step oxinfer run with --gguf
Where diffusers samples the VAE posterior (image to image, Fill, control), the reference takes
its mode, as oxinfer does.
"""
import argparse, os, subprocess, sys

import numpy as np


def save(path, t):
    np.asarray(t, dtype=np.float32).tofile(path)


def load(path):
    return np.fromfile(path, dtype=np.float32)


def psnr(a, b):
    pa = np.round(np.clip((a + 1) / 2, 0, 1) * 255)
    pb = np.round(np.clip((b + 1) / 2, 0, 1) * 255)
    return 10 * np.log10(255 ** 2 / max(np.mean((pa - pb) ** 2), 1e-12))


def comfy_positions(method, sizes):
    """ComfyUI's comfy/ldm/flux/model.py (Flux._forward, ref_latents): the (index, h_offset,
    w_offset) of each reference latent of size (h, w) in latent pixels; token offsets are
    (offset + 1) // 2 (process_img with patch size 2)."""
    out, h, w, index = [], 0, 0, 0
    for (rh, rw) in sizes:
        if method == "index":
            index += 1
            ho = wo = 0
        else:  # "offset", the default
            index = 1
            ho = wo = 0
            if rh + h > rw + w:
                wo = w
            else:
                ho = h
            h = max(h, rh + ho)
            w = max(w, rw + wo)
        out.append((index, (ho + 1) // 2, (wo + 1) // 2))
    return out


def test_images(out):
    """Deterministic inputs: two photos-like gradients with shapes, a mask, an edge map."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(0)
    paths = {}
    for name, (w, h) in {"a": (200, 152), "b": (120, 168), "c": (96, 96)}.items():
        y, x = np.mgrid[0:h, 0:w]
        arr = np.stack([x * 255 / w, y * 255 / h, (x + y) * 255 / (w + h)], -1) + rng.normal(0, 12, (h, w, 3))
        img = Image.fromarray(arr.clip(0, 255).astype("uint8"))
        d = ImageDraw.Draw(img)
        d.ellipse([w // 4, h // 4, w // 2, h * 3 // 4], fill=(240, 60, 30))
        d.rectangle([w // 2, h // 3, w * 7 // 8, h // 2], fill=(20, 90, 220))
        paths[name] = os.path.join(out, f"in_{name}.png")
        img.save(paths[name])
    m = Image.new("RGB", (200, 152), (0, 0, 0))
    ImageDraw.Draw(m).ellipse([50, 30, 150, 120], fill=(255, 255, 255))
    # Soft edge, so that the threshold and the resize order matter.
    from PIL import ImageFilter
    m = m.filter(ImageFilter.GaussianBlur(3))
    paths["mask"] = os.path.join(out, "in_mask.png")
    m.save(paths["mask"])
    e = Image.new("RGB", (200, 152), (0, 0, 0))
    d = ImageDraw.Draw(e)
    d.line([10, 10, 190, 140], fill=(255, 255, 255), width=2)
    d.ellipse([60, 40, 140, 110], outline=(255, 255, 255), width=2)
    paths["edges"] = os.path.join(out, "in_edges.png")
    e.save(paths["edges"])
    return paths


def make_loras(model, out):
    """Random LoRA files over the transformer of `model`, one per key naming. Returns
    {name: path}; `control` is a BFL control LoRA widening img_in to 128 channels."""
    import torch
    from diffusers import FluxPipeline, FluxTransformer2DModel
    from safetensors.torch import save_file
    tr = FluxTransformer2DModel.from_pretrained(model, subfolder="transformer")
    sd = tr.state_dict()
    d = tr.inner_dim
    nd, ns = len(tr.transformer_blocks), len(tr.single_transformer_blocks)
    g = torch.Generator().manual_seed(7)
    rnd = lambda *s: torch.randn(*s, generator=g) * 0.2
    paths = {}
    # diffusers / PEFT names, alpha (12) from the metadata header, rank 6.
    r = 6
    peft = {}
    targets = ["x_embedder", "context_embedder", "proj_out", "norm_out.linear",
               "time_text_embed.timestep_embedder.linear_1", "time_text_embed.text_embedder.linear_2"]
    for i in range(nd):
        targets += [f"transformer_blocks.{i}.{m}" for m in (
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
            "attn.to_out.0", "attn.to_add_out", "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj",
            "ff_context.net.2", "norm1.linear", "norm1_context.linear")]
    for i in range(ns):
        targets += [f"single_transformer_blocks.{i}.{m}" for m in (
            "attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp", "proj_out", "norm.linear")]
    for t in targets:
        o, n = sd[f"{t}.weight"].shape
        peft[f"{t}.lora_A.weight"] = rnd(r, n)
        peft[f"{t}.lora_B.weight"] = rnd(o, r)
    from peft import LoraConfig
    meta = LoraConfig(r=r, lora_alpha=12, target_modules=targets).to_dict()
    os.makedirs(os.path.join(out, "lora_peft"), exist_ok=True)
    FluxPipeline.save_lora_weights(os.path.join(out, "lora_peft"), transformer_lora_layers=peft,
                                   transformer_lora_adapter_metadata=meta)
    paths["peft"] = os.path.join(out, "lora_peft", "pytorch_lora_weights.safetensors")
    # kohya names over the fused BFL modules, each with its own alpha, rank 4.
    r = 4
    ko = {}
    def kmod(name, rows, cols, alpha):
        k = "lora_unet_" + name.replace(".", "_")
        ko[f"{k}.lora_down.weight"] = rnd(r, cols)
        ko[f"{k}.lora_up.weight"] = rnd(rows, r)
        ko[f"{k}.alpha"] = torch.tensor(float(alpha))
    mlp = sd["single_transformer_blocks.0.proj_mlp.weight"].shape[0]
    for i in range(nd):
        for s in ("img", "txt"):
            kmod(f"double_blocks.{i}.{s}_attn.qkv", 3 * d, d, 2)
            kmod(f"double_blocks.{i}.{s}_attn.proj", d, d, 4)
            kmod(f"double_blocks.{i}.{s}_mlp.0", 4 * d, d, 8)
            kmod(f"double_blocks.{i}.{s}_mlp.2", d, 4 * d, 1)
            kmod(f"double_blocks.{i}.{s}_mod.lin", 6 * d, d, 4)
    for i in range(ns):
        kmod(f"single_blocks.{i}.linear1", 3 * d + mlp, d, 3)
        kmod(f"single_blocks.{i}.linear2", d, d + mlp, 4)
        kmod(f"single_blocks.{i}.modulation.lin", 3 * d, d, 4)
    paths["kohya"] = os.path.join(out, "lora_kohya.safetensors")
    save_file(ko, paths["kohya"])
    # XLabs attention-processor names (no alpha: scale 1), rank 3.
    r = 3
    xl = {}
    for i in range(nd):
        for p, rows in (("qkv_lora1", 3 * d), ("qkv_lora2", 3 * d), ("proj_lora1", d), ("proj_lora2", d)):
            xl[f"double_blocks.{i}.processor.{p}.down.weight"] = rnd(r, d)
            xl[f"double_blocks.{i}.processor.{p}.up.weight"] = rnd(rows, r)
    paths["xlabs"] = os.path.join(out, "lora_xlabs.safetensors")
    save_file(xl, paths["xlabs"])
    # A BFL control LoRA (flux1-canny-dev-lora layout): every module as lora_A / lora_B with a
    # bias on lora_B, img_in over 2x the input channels, the QK norm scales replaced.
    if tr.config.in_channels == 64 and tr.config.guidance_embeds:
        r = 5
        cl = {}
        def cmod(name, rows, cols):
            cl[f"{name}.lora_A.weight"] = rnd(r, cols)
            cl[f"{name}.lora_B.weight"] = rnd(rows, r)
            cl[f"{name}.lora_B.bias"] = rnd(rows) * 0.1
        jd, pd = tr.config.joint_attention_dim, tr.config.pooled_projection_dim
        cmod("img_in", d, 128)
        cmod("txt_in", d, jd)
        for n, c in (("time_in", 256), ("guidance_in", 256), ("vector_in", pd)):
            cmod(f"{n}.in_layer", d, c)
            cmod(f"{n}.out_layer", d, d)
        cmod("final_layer.linear", 64, d)
        cmod("final_layer.adaLN_modulation.1", 2 * d, d)
        hd = tr.config.attention_head_dim
        for i in range(nd):
            for s in ("img", "txt"):
                cmod(f"double_blocks.{i}.{s}_mod.lin", 6 * d, d)
                cmod(f"double_blocks.{i}.{s}_attn.qkv", 3 * d, d)
                cmod(f"double_blocks.{i}.{s}_attn.proj", d, d)
                cmod(f"double_blocks.{i}.{s}_mlp.0", 4 * d, d)
                cmod(f"double_blocks.{i}.{s}_mlp.2", d, 4 * d)
                for q in ("query_norm", "key_norm"):
                    cl[f"double_blocks.{i}.{s}_attn.norm.{q}.scale"] = 1 + rnd(hd)
        for i in range(ns):
            cmod(f"single_blocks.{i}.modulation.lin", 3 * d, d)
            cmod(f"single_blocks.{i}.linear1", 3 * d + mlp, d)
            cmod(f"single_blocks.{i}.linear2", d, d + mlp)
            for q in ("query_norm", "key_norm"):
                cl[f"single_blocks.{i}.norm.{q}.scale"] = 1 + rnd(hd)
        paths["control"] = os.path.join(out, "lora_control.safetensors")
        save_file(cl, paths["control"])
    return paths


def merged_transformer(model, specs):
    """The transformer of `model` with LoRA files in BFL-layout namings (kohya `lora_unet_*`,
    XLabs `processor.*_lora*`, BFL `*.lora_A` control LoRAs) merged into its weights by hand, in
    the BFL layout the keys name: W += scale * alpha / rank * up @ down (bias and QK norm scales
    of control LoRAs included, img_in widened with zeros). An independent reference for the
    plumbing: diffusers' own converters for these namings (and for single-file checkpoints)
    hardcode FLUX.1's sizes, so they cannot read a tiny model's files."""
    import torch
    from diffusers import FluxTransformer2DModel
    from safetensors.torch import load_file
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from flux_tiny import bfl_to_transformer, transformer_to_bfl
    tr = FluxTransformer2DModel.from_pretrained(model, subfolder="transformer", torch_dtype=torch.float32)
    cfg = dict(tr.config)
    sd = {k: v.clone().double() for k, v in transformer_to_bfl(tr.state_dict(), tr.inner_dim).items()}
    under = {k[:-len(".weight")].replace(".", "_"): k[:-len(".weight")] for k in sd if k.endswith(".weight")}
    xl = {"qkv_lora1": "img_attn.qkv", "qkv_lora2": "txt_attn.qkv", "proj_lora1": "img_attn.proj", "proj_lora2": "txt_attn.proj"}
    for spec in specs:
        path, _, sc = spec.partition(":")
        scale = float(sc or 1)
        lo = {k: v.double() for k, v in load_file(path).items()}
        for k in list(lo):
            if k.endswith(".scale"):
                sd[k] = lo[k]
                continue
            for dn, un in ((".lora_down.weight", ".lora_up.weight"), (".lora_A.weight", ".lora_B.weight"), (".down.weight", ".up.weight")):
                if not k.endswith(dn):
                    continue
                m = k[:-len(dn)]
                down, up = lo[k], lo[m + un]
                alpha = float(lo[m + ".alpha"]) if m + ".alpha" in lo else down.shape[0]
                if m.startswith("lora_unet_"):
                    name = under[m[len("lora_unet_"):]]
                elif ".processor." in m:
                    block, proc = m.split(".processor.")
                    name = f"{block}.{xl[proc]}"
                else:
                    name = m
                delta = scale * alpha / down.shape[0] * (up @ down)
                w = sd[f"{name}.weight"]
                if delta.shape[1] > w.shape[1]:  # control LoRA: more input channels
                    w = torch.cat([w, torch.zeros(w.shape[0], delta.shape[1] - w.shape[1], dtype=w.dtype)], dim=1)
                    cfg["in_channels"] = delta.shape[1]
                sd[f"{name}.weight"] = w + delta
                bk = m + un[:-len("weight")] + "bias"
                if bk in lo:
                    sd[f"{name}.bias"] = sd[f"{name}.bias"] + scale * alpha / down.shape[0] * lo[bk]
                break
    new = FluxTransformer2DModel.from_config(cfg)
    new.load_state_dict(bfl_to_transformer({k: v.float() for k, v in sd.items()}, tr.inner_dim), strict=True)
    return new.eval()


def diffusers_reads(path):
    """Whether the file is in diffusers / PEFT naming (loaded with load_lora_weights)."""
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        return all(k.startswith("transformer.") for k in f.keys())


def step(a):
    """One transformer call on real weights (see the module docs)."""
    import torch
    import diffusers
    from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline as FP, calculate_shift
    torch.set_grad_enabled(False)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    os.makedirs(a.out, exist_ok=True)
    P = lambda n: os.path.join(a.out, n)
    w, h = a.size or (256, 256)
    lh, lw = 2 * (h // 16), 2 * (w // 16)
    noise = torch.randn((1, 16, lh, lw), generator=torch.Generator().manual_seed(a.seed))
    save(P("latents_in.bin"), noise[0].numpy())
    ox = ["--size", f"{w}x{h}", "--base", a.model]
    for spec in a.lora or []:
        ox += ["--lora", spec]
    if a.control:
        ox += ["--control-image", a.control]
    a.steps = a.steps or 1
    if a.layers:
        os.environ["OXINFER_FLUX_LAYERS"] = a.layers
    run_oxinfer(a, a.out, ox, a.gguf)
    dt = torch.float32
    keep_lora = lambda k: True
    if a.layers:
        # Only the first D double and S single blocks, dequantized exactly to f32 with the gguf
        # package (as tools/ref_flux.py --layers); LoRA modules of the cut blocks are dropped.
        import re
        from gguf import GGUFReader
        from gguf.quants import dequantize
        from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
        d, s_ = (int(x) for x in a.layers.split(","))
        keep = lambda n: not (n.startswith("double_blocks.") and int(n.split(".")[1]) >= d) and \
            not (n.startswith("single_blocks.") and int(n.split(".")[1]) >= s_)
        ck = {t.name: torch.from_numpy(np.ascontiguousarray(
                  dequantize(t.data, t.tensor_type).reshape([int(x) for x in reversed(t.shape)]))).float()
              for t in GGUFReader(a.gguf).tensors if keep(t.name)}
        cfg = dict(FluxTransformer2DModel.load_config(os.path.join(a.model, "transformer")))
        cfg["num_layers"], cfg["num_single_layers"] = d, s_
        tr = FluxTransformer2DModel.from_config(cfg)
        tr.load_state_dict(convert_flux_transformer_checkpoint_to_diffusers(ck), strict=True)
        del ck

        def keep_lora(k):
            m = re.search(r"(single_)?transformer_blocks\.(\d+)\.", k)
            return m is None or int(m.group(2)) < (s_ if m.group(1) else d)
    else:
        tr = FluxTransformer2DModel.from_single_file(a.gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dt),
                                                     torch_dtype=dt, config=a.model, subfolder="transformer")
    cls = diffusers.FluxControlPipeline if a.control else FP
    pipe = cls(scheduler=None, vae=None, text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None, transformer=tr)
    names, scales = [], []
    for i, spec in enumerate(a.lora or []):
        path, _, sc = spec.partition(":")
        sd = pipe.lora_state_dict(path)
        if a.fix_norm_out:
            # diffusers' BFL control-LoRA converter swaps the (shift, scale) halves of
            # final_layer.adaLN_modulation.1's lora_A as well as lora_B's; only lora_B's rows are
            # output channels, so swap lora_A's rank rows back.
            for k in [k for k in sd if k.endswith("norm_out.linear.lora_A.weight")]:
                x, y = sd[k].chunk(2, dim=0)
                sd[k] = torch.cat([y, x], dim=0)
        pipe.load_lora_weights({k: v for k, v in sd.items() if keep_lora(k)}, adapter_name=f"l{i}")
        names.append(f"l{i}")
        scales.append(float(sc or 1))
    if names:
        pipe.set_adapters(names, scales)
    tr = pipe.transformer
    lat = FP._pack_latents(noise, 1, 16, lh, lw)
    hs = lat
    if a.control:
        cl = torch.from_numpy(load(P("control_latents.bin")).reshape(1, 16, lh, lw))
        hs = torch.cat([lat, FP._pack_latents(cl, 1, 16, lh, lw)], dim=2)
    txt = torch.from_numpy(load(P("t5.bin")).reshape(1, -1, 4096))
    pooled = torch.from_numpy(load(P("pooled.bin")).reshape(1, 768))
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(a.model, subfolder="scheduler")
    mu = calculate_shift(lat.shape[1], sch.config.base_image_seq_len, sch.config.max_image_seq_len,
                         sch.config.base_shift, sch.config.max_shift)
    sch.set_timesteps(sigmas=np.linspace(1.0, 1 / a.steps, a.steps), mu=mu)
    t = sch.timesteps[0]
    gd = torch.full([1], a.guidance) if tr.config.guidance_embeds else None
    ids = FP._prepare_latent_image_ids(1, lh // 2, lw // 2, "cpu", dt)
    out = tr(hidden_states=hs, timestep=(t / 1000).reshape(1), guidance=gd, pooled_projections=pooled,
             encoder_hidden_states=txt, txt_ids=torch.zeros(txt.shape[1], 3), img_ids=ids, return_dict=False)[0]
    ours, ref = load(P("noise_pred0.bin")), out[0].numpy().ravel()
    rel = np.linalg.norm(ours - ref) / np.linalg.norm(ref)
    print(f"step ({os.path.basename(a.gguf)}, {len(a.lora or [])} lora) {w}x{h}  noise prediction rel L2 {rel:.2e}", flush=True)
    if a.lora:
        # How much the LoRA moves the prediction, for scale: the same call without it.
        pipe.unload_lora_weights()
        if a.control:
            return
        base = pipe.transformer(hidden_states=hs, timestep=(t / 1000).reshape(1), guidance=gd, pooled_projections=pooled,
                                encoder_hidden_states=txt, txt_ids=torch.zeros(txt.shape[1], 3), img_ids=ids,
                                return_dict=False)[0][0].numpy().ravel()
        print(f"    (the LoRA moves the prediction by rel L2 {np.linalg.norm(ref - base) / np.linalg.norm(base):.2e})")


def redux_tokens(a):
    import torch
    from PIL import Image
    from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
    from transformers import SiglipImageProcessor, SiglipVisionModel
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    enc = SiglipVisionModel.from_pretrained(os.path.join(a.redux, "image_encoder"), torch_dtype=torch.float32)
    emb = ReduxImageEncoder.from_pretrained(os.path.join(a.redux, "image_embedder"), torch_dtype=torch.float32)
    fe = SiglipImageProcessor.from_pretrained(os.path.join(a.redux, "feature_extractor"))
    px = fe.preprocess(images=Image.open(a.image[0]).convert("RGB"), do_resize=True, return_tensors="pt", do_convert_rgb=True)
    ref = emb(enc(**px).last_hidden_state).image_embeds[0].numpy().ravel()
    # oxinfer dumps the tokens during a one-step run (--gguf FILE over MODEL's other parts).
    a.steps = 1
    run_oxinfer(a, a.out, ["--size", "256x256", "--base", a.model, "--redux", a.redux, "--image-prompt", a.image[0]], a.gguf)
    ours = load(os.path.join(a.out, "redux.bin"))
    print(f"redux tokens {ours.size // 4096} x 4096  rel L2 {np.linalg.norm(ours - ref) / np.linalg.norm(ref):.2e}")


def run_oxinfer(a, out, extra, model=None):
    env = dict(os.environ, OXINFER_FLUX_DUMP=out)
    for f in ("latents_out.bin", "image.bin"):
        if os.path.exists(os.path.join(out, f)):
            os.remove(os.path.join(out, f))
    cmd = [a.oxinfer, "--model", model or a.model, "--device", "cpu", "-p", a.prompt, "--steps", str(a.steps),
           "--guidance", str(a.guidance), "-o", os.path.join(out, "ours.png")] + extra
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(os.path.join(out, "latents_out.bin")):
        print(" ".join(cmd))
        print(r.stderr[-3000:])
        raise SystemExit("oxinfer failed")
    for line in r.stderr.splitlines():
        if line.startswith("[lora"):
            print("   ", line)


def one(a, mode, out, tag=None, **kw):
    """Runs one check; kw overrides the arguments."""
    import torch
    from PIL import Image
    import diffusers
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    # The posterior's mode wherever a pipeline would sample it.
    DiagonalGaussianDistribution.sample = lambda self, generator=None: self.mode()
    torch.set_grad_enabled(False)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    ns = argparse.Namespace(**{**vars(a), **kw})
    os.makedirs(out, exist_ok=True)
    P = lambda n: os.path.join(out, n)
    dt = torch.float32
    imgs = [Image.open(p).convert("RGB") for p in (ns.image or [])]
    w, h = ns.size if ns.size else (imgs[0].size if imgs and mode != "multi" else (256, 256))
    if mode == "control" and not ns.size:
        w, h = Image.open(ns.control).size
    if mode == "multi":
        given = ns.size
        if ns.size:
            ar = ns.size[0] / ns.size[1]
        else:
            ar = imgs[0].size[0] / imgs[0].size[1]
            ns.size = imgs[0].size
        w, h = round((ns.max_area * ar) ** 0.5) // 16 * 16, round((ns.max_area / ar) ** 0.5) // 16 * 16
    else:
        w, h = w // 16 * 16, h // 16 * 16
    lh, lw = 2 * (h // 16), 2 * (w // 16)
    gen = torch.Generator().manual_seed(ns.seed)
    noise = torch.randn((1, 16, lh, lw), generator=gen)
    save(P("latents_in.bin"), noise[0].numpy())
    common = dict(prompt=ns.prompt, num_inference_steps=ns.steps, guidance_scale=ns.guidance, output_type="latent",
                  max_sequence_length=512)
    ox = ["--size", f"{w}x{h}"]
    g = lambda: torch.Generator().manual_seed(ns.seed)  # draws the same noise first
    model = None
    FP = diffusers.FluxPipeline
    packed = lambda p: p._pack_latents(noise, 1, 16, lh, lw)
    if mode == "t2i":
        pipe = FP.from_pretrained(ns.model, torch_dtype=dt)
        z = pipe(**common, latents=packed(pipe), width=w, height=h).images
    elif mode == "bfl":
        pipe = FP.from_pretrained(ns.model, torch_dtype=dt)
        z = pipe(**common, latents=packed(pipe), width=w, height=h).images
        model = ns.bfl
        ox += ["--base", ns.model] + (["--vae", ns.ae] if ns.ae else [])
    elif mode == "lora":
        byhand = [] if ns.diffusers_loader else [s for s in ns.lora if not diffusers_reads(s.partition(":")[0])]
        pipe = FP.from_pretrained(ns.model, torch_dtype=dt)
        if byhand:
            pipe.transformer = merged_transformer(ns.model, byhand)
        names, scales = [], []
        for i, spec in enumerate(ns.lora):
            path, _, sc = spec.partition(":")
            ox += ["--lora", spec]
            if spec in byhand:
                continue
            pipe.load_lora_weights(path, adapter_name=f"l{i}")
            names.append(f"l{i}")
            scales.append(float(sc or 1))
        if getattr(ns, "bfl", None):  # the LoRA on the single-file checkpoint
            model = ns.bfl
            ox += ["--base", ns.model]
        if names:
            pipe.set_adapters(names, scales)
        z = pipe(**common, latents=packed(pipe), width=w, height=h).images
    elif mode == "img2img":
        pipe = diffusers.FluxImg2ImgPipeline.from_pretrained(ns.model, torch_dtype=dt)
        z = pipe(**common, image=imgs[0], width=w, height=h, strength=ns.strength, generator=g()).images
        ox += ["--init-image", ns.image[0], "--strength", str(ns.strength)]
    elif mode == "inpaint":
        pipe = diffusers.FluxInpaintPipeline.from_pretrained(ns.model, torch_dtype=dt)
        z = pipe(**common, image=imgs[0], mask_image=Image.open(ns.mask), width=w, height=h, strength=ns.strength,
                 generator=g()).images
        ox += ["--init-image", ns.image[0], "--mask", ns.mask, "--strength", str(ns.strength)]
    elif mode == "fill":
        pipe = diffusers.FluxFillPipeline.from_pretrained(ns.model, torch_dtype=dt)
        st = 1.0 if ns.strength is None else ns.strength
        z = pipe(**common, image=imgs[0], mask_image=Image.open(ns.mask), width=w, height=h, strength=st,
                 generator=g()).images
        ox += ["--init-image", ns.image[0], "--mask", ns.mask] + (["--strength", str(st)] if ns.strength is not None else [])
    elif mode == "control":
        pipe = diffusers.FluxControlPipeline.from_pretrained(ns.model, torch_dtype=dt)
        if ns.lora and not ns.diffusers_loader:
            pipe.transformer = merged_transformer(ns.model, ns.lora)
        for i, spec in enumerate(ns.lora or []):
            path, _, sc = spec.partition(":")
            ox += ["--lora", spec]
            if ns.diffusers_loader:
                pipe.load_lora_weights(path, adapter_name=f"l{i}")
                pipe.set_adapters([f"l{i}"], [float(sc or 1)])
        z = pipe(**common, control_image=Image.open(ns.control).convert("RGB"), width=w, height=h, latents=packed(pipe)).images
        ox += ["--control-image", ns.control]
    elif mode == "redux":
        from diffusers import FluxPriorReduxPipeline
        from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
        from transformers import SiglipImageProcessor, SiglipVisionModel
        pipe = FP.from_pretrained(ns.model, torch_dtype=dt)
        enc = SiglipVisionModel.from_pretrained(os.path.join(ns.redux, "image_encoder"), torch_dtype=dt)
        emb = ReduxImageEncoder.from_pretrained(os.path.join(ns.redux, "image_embedder"), torch_dtype=dt)
        s = enc.config.image_size
        fe = SiglipImageProcessor(size={"height": s, "width": s}, image_mean=[0.5] * 3, image_std=[0.5] * 3, resample=3)
        prior = FluxPriorReduxPipeline(image_encoder=enc, feature_extractor=fe, image_embedder=emb,
                                       text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer,
                                       text_encoder_2=pipe.text_encoder_2, tokenizer_2=pipe.tokenizer_2)
        # (The prior pipeline encodes the prompt with 512 T5 tokens; the same here.)
        pe, pp = prior(image=imgs[0], prompt=ns.prompt, return_dict=False)
        px = fe.preprocess(images=imgs[0], do_resize=True, return_tensors="pt", do_convert_rgb=True)
        save(P("ref_redux.bin"), emb(enc(**px).last_hidden_state).image_embeds[0].numpy())
        c2 = {k: v for k, v in common.items() if k not in ("prompt", "max_sequence_length")}
        z = pipe(**c2, prompt_embeds=pe, pooled_prompt_embeds=pp, latents=packed(pipe), width=w, height=h).images
        ox += ["--redux", ns.redux, "--image-prompt", ns.image[0]]
    elif mode == "multi":
        # (With --size and auto-resize the CLI resizes the references to the output instead.)
        ox = ["--max-area", str(ns.max_area), "--ref-layout", ns.layout] + (["--size", f"{given[0]}x{given[1]}"] if given else [])
        if ns.no_auto_resize:
            ox.append("--no-auto-resize")
        for p in ns.image:
            ox += ["--init-image", p]
        KP = diffusers.FluxKontextPipeline
        if ns.layout == "diffusers" and ns.community:
            pipe = diffusers.DiffusionPipeline.from_pretrained(ns.model, torch_dtype=dt, custom_pipeline=ns.community,
                                                               trust_remote_code=True)
            z = pipe(**common, multiple_images=tuple(imgs), width=ns.size[0], height=ns.size[1], max_area=ns.max_area,
                     _auto_resize=not ns.no_auto_resize, latents=packed(pipe)).images
        else:
            # A Kontext loop over explicit token ids (ComfyUI's, or the community pipeline's).
            from diffusers.pipelines.flux.pipeline_flux_kontext import PREFERRED_KONTEXT_RESOLUTIONS, calculate_shift
            pipe = KP.from_pretrained(ns.model, torch_dtype=dt)
            vae, proc = pipe.vae, pipe.image_processor
            toks, ids, sizes = [], [], []
            for i, im in enumerate(imgs):
                src = imgs[0] if ns.layout == "diffusers" else im
                iw, ih = (x - x % 16 for x in src.size)
                if not ns.no_auto_resize:
                    _, iw, ih = min((abs(iw / ih - pw / ph), pw, ph) for pw, ph in PREFERRED_KONTEXT_RESOLUTIONS)
                iw, ih = iw // 16 * 16, ih // 16 * 16
                x = proc.preprocess(proc.resize(im, ih, iw), ih, iw)
                zr = (vae.encode(x).latent_dist.mode() - vae.config.shift_factor) * vae.config.scaling_factor
                toks.append(KP._pack_latents(zr, 1, 16, ih // 8, iw // 8))
                sizes.append((ih // 8, iw // 8))
            if ns.layout == "diffusers":
                pos = [(1, 0, i * (sizes[0][0] // 2)) for i in range(len(imgs))]
            else:
                pos = comfy_positions(ns.layout, sizes)
            for (rh, rw), (index, y0, x0) in zip(sizes, pos):
                rid = KP._prepare_latent_image_ids(1, rh // 2, rw // 2, "cpu", dt)
                rid[..., 0] = index
                rid[..., 1] += y0
                rid[..., 2] += x0
                ids.append(rid)
            lat = packed(pipe)
            n = lat.shape[1]
            all_ids = torch.cat([KP._prepare_latent_image_ids(1, lh // 2, lw // 2, "cpu", dt)] + ids, dim=0)
            pe, pp, tid = pipe.encode_prompt(prompt=ns.prompt, prompt_2=None, max_sequence_length=512)
            sch = pipe.scheduler
            mu = calculate_shift(n, sch.config.get("base_image_seq_len", 256), sch.config.get("max_image_seq_len", 4096),
                                 sch.config.get("base_shift", 0.5), sch.config.get("max_shift", 1.15))
            sch.set_timesteps(sigmas=np.linspace(1.0, 1 / ns.steps, ns.steps), mu=mu)
            gd = torch.full([1], ns.guidance) if pipe.transformer.config.guidance_embeds else None
            for t in sch.timesteps:
                v = pipe.transformer(hidden_states=torch.cat([lat] + toks, dim=1), timestep=t.expand(1) / 1000, guidance=gd,
                                     pooled_projections=pp, encoder_hidden_states=pe, txt_ids=tid, img_ids=all_ids,
                                     return_dict=False)[0][:, :n]
                lat = sch.step(v, t, lat, return_dict=False)[0]
            z = lat
    else:
        raise SystemExit(f"unknown mode {mode}")
    z = pipe._unpack_latents(z, h, w, pipe.vae_scale_factor)
    x = pipe.vae.decode(z / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor, return_dict=False)[0][0]
    run_oxinfer(ns, out, ox, model)
    ours, ref = load(P("latents_out.bin")), z[0].numpy().ravel()
    rel = np.linalg.norm(ours - ref) / max(np.linalg.norm(ref), 1e-12)
    line = f"{tag or mode:28s} {w}x{h}  latents rel L2 {rel:.2e}  image PSNR {psnr(load(P('image.bin')), x.numpy().ravel()):.1f} dB"
    if mode == "redux" and os.path.exists(P("redux.bin")):
        r0, r1 = load(P("redux.bin")), load(P("ref_redux.bin"))
        line += f"  redux tokens rel L2 {np.linalg.norm(r0 - r1) / np.linalg.norm(r1):.2e}"
    print(line, flush=True)
    return rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--oxinfer", default="target/release/oxinfer")
    ap.add_argument("--mode", required=True)
    ap.add_argument("-p", "--prompt", default="a red fox reading a newspaper in the snow")
    ap.add_argument("--image", action="append")
    ap.add_argument("--mask")
    ap.add_argument("--control")
    ap.add_argument("--lora", action="append")
    ap.add_argument("--redux")
    ap.add_argument("--gguf")
    ap.add_argument("--fix-norm-out", action="store_true",
                    help="step, BFL control LoRAs: undo diffusers' swap of norm_out's lora_A halves (see the code)")
    ap.add_argument("--layers", help="step: D,S, only the first D double and S single blocks (f32 on the Python side)")
    ap.add_argument("--bfl")
    ap.add_argument("--ae")
    ap.add_argument("--diffusers-loader", action="store_true",
                    help="load every --lora with diffusers' load_lora_weights (real FLUX.1 files in any naming); "
                         "default: files not in diffusers naming are merged by hand (see merged_transformer)")
    ap.add_argument("--community", help="diffusers' examples/community/pipeline_flux_kontext_multiple_images.py")
    ap.add_argument("--layout", default="offset", choices=["offset", "index", "diffusers"])
    ap.add_argument("--size", type=lambda s: tuple(int(x) for x in s.split("x")))
    ap.add_argument("--max-area", type=int, default=1024 ** 2)
    ap.add_argument("--no-auto-resize", action="store_true")
    ap.add_argument("--strength", type=float)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "step":
        return step(a)
    if a.mode == "redux-tokens":
        return redux_tokens(a)
    if a.mode != "all":
        if a.mode in ("img2img", "inpaint") and a.strength is None:
            a.strength = 0.6
        one(a, a.mode, a.out)
        return
    # Every check the folder allows, on generated inputs (tiny models).
    import json
    os.makedirs(a.out, exist_ok=True)
    im = test_images(a.out)
    cfg = json.load(open(os.path.join(a.model, "transformer", "config.json")))
    ch = cfg["in_channels"]
    sub = lambda n: os.path.join(a.out, n)
    if ch == 384:
        one(a, "fill", sub("fill"), image=[im["a"]], mask=im["mask"])
        one(a, "fill", sub("fill_s"), "fill, strength 0.7", image=[im["a"]], mask=im["mask"], strength=0.7, steps=6)
        return
    if ch == 128:
        one(a, "control", sub("control"), control=im["edges"])
        return
    one(a, "t2i", sub("t2i"), size=(160, 128))
    one(a, "img2img", sub("img2img"), image=[im["a"]], strength=0.6, steps=6)
    one(a, "img2img", sub("img2img_r"), "img2img, resized, 0.35", image=[im["b"]], strength=0.35, steps=10, size=(144, 176))
    one(a, "inpaint", sub("inpaint"), image=[im["a"]], mask=im["mask"], strength=0.8, steps=6)
    one(a, "inpaint", sub("inpaint_1"), "inpaint, strength 1", image=[im["b"]], mask=im["mask"], strength=1.0, steps=4)
    multi = dict(image=[im["a"], im["b"], im["c"]], no_auto_resize=True, max_area=160 * 128, size=(160, 128))
    for layout in ("offset", "index", "diffusers"):
        one(a, "multi", sub(f"multi_{layout}"), f"multi ({layout} ids)", layout=layout, community=None, **multi)
    if a.community:
        one(a, "multi", sub("multi_community"), "multi (community pipeline)", layout="diffusers", **multi)
    one(a, "multi", sub("multi_auto"), "multi (offset, auto-resize)", layout="offset", image=[im["a"], im["b"]],
        max_area=128 * 128, steps=1)
    loras = make_loras(a.model, a.out)
    for n in ("peft", "kohya", "xlabs"):
        one(a, "lora", sub(f"lora_{n}"), f"lora ({n} keys)", lora=[loras[n]], size=(160, 128))
    one(a, "lora", sub("lora_mix"), "lora (peft:0.7 + kohya:-0.5)", lora=[loras["peft"] + ":0.7", loras["kohya"] + ":-0.5"],
        size=(160, 128))
    if "control" in loras:
        one(a, "control", sub("control_lora"), "control (BFL control LoRA)", control=im["edges"], lora=[loras["control"]])
    bfl = os.path.join(a.model, "bfl", "flux1-tiny.safetensors")
    if os.path.exists(bfl):
        one(a, "bfl", sub("bfl"), "bfl single file + ae", bfl=bfl, ae=os.path.join(a.model, "bfl", "ae.safetensors"), size=(160, 128))
        one(a, "lora", sub("bfl_lora"), "lora (kohya) on the bfl file", lora=[loras["kohya"]], size=(160, 128), bfl=bfl)
    redux = os.path.join(a.model, "redux")
    if os.path.isdir(redux):
        one(a, "redux", sub("redux"), image=[im["a"]], redux=redux, size=(160, 128))


if __name__ == "__main__":
    sys.exit(main())
