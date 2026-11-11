#!/usr/bin/env python3
"""A tiny random-weight OmniGen2 and the official pipeline's results on it, for exact checks of
oxinfer's OmniGen2 options that need no real weights: the TeaCache and TaylorSeer step caches,
the DPM-Solver++ scheduler and VAE posterior sampling.

    python tools/omnigen2_tiny.py REPO OUT [--steps 12] [--size 64] [--images 1]
        [--teacache THRESH | --taylorseer] [--scheduler euler|dpmsolver++] [--sample-posterior]

REPO is a checkout of github.com/VectorSpaceLab/OmniGen2. Writes into OUT:
  model/           an OmniGen2 diffusers folder (4-layer 96-wide transformer, 2-level VAE with
                   16 latent channels; every weight random and rounded to bf16, float32 files)
  in0.png ..       the input images
  dump/            oxinfer's inputs (OXINFER_OG2_DUMP): prompt_embeds_in.bin,
                   negative_embeds_in.bin, noise.bin, ref_noise{i}_in.bin (posterior noise)
  ref/             the official pipeline's results: latents_out.bin, v0.bin, ref_latents{i}.bin,
                   image.bin, cache.txt (one line per transformer pass: step, branch, 1 = blocks ran)

Then:
  OXINFER_OG2_DUMP=OUT/dump oxinfer --model OUT/model --device cpu --init-image OUT/in0.png \
      --steps 12 -p x -o OUT/out.png [--teacache 0.2 | --taylorseer] [--scheduler dpmsolver++] \
      [--sample-posterior]
and compare OUT/dump/latents_out.bin and OUT/dump/cache.txt with OUT/ref/ (--compare does it).

The official pipeline runs as it is (OmniGen2Pipeline.__call__ in float32 on the CPU) with
prompt embeddings in place of the Qwen2.5-VL encoder; only the posterior sampling of the input
images is replaced by a draw from a seeded generator (or the mean), to pass the same noise to
oxinfer. Needs torch, diffusers, transformers, einops, pillow.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("out")
ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--size", type=int, default=64)
ap.add_argument("--images", type=int, default=1)
ap.add_argument("--teacache", type=float)
ap.add_argument("--taylorseer", action="store_true")
ap.add_argument("--scheduler", default="euler")
ap.add_argument("--sample-posterior", action="store_true")
ap.add_argument("--text-guidance", type=float, default=5.0)
ap.add_argument("--image-guidance", type=float, default=2.0)
ap.add_argument("--compare", action="store_true", help="compare OUT/dump (oxinfer) with OUT/ref")
args = ap.parse_args()

out = args.out
ref_dir, dump_dir, model_dir = f"{out}/ref", f"{out}/dump", f"{out}/model"


def rd(p):
    return np.fromfile(p, dtype="<f4").astype(np.float64)


def tokens(chw):
    """`c (h p1) (w p2) -> (h w) (p1 p2 c)`: the layout of oxinfer's token dumps."""
    c = 16
    side = int(round((chw.size // c) ** 0.5))
    x = chw.reshape(c, side // 2, 2, side // 2, 2)
    return x.transpose(1, 3, 2, 4, 0).reshape(-1)


if args.compare:
    ok = True
    pairs = [(f"ref_tokens{i}.bin", f"ref_latents{i}.bin", True) for i in range(5)]
    pairs += [("v0.bin", "v0.bin", True), ("latents_out.bin", "latents_out.bin", False)]
    for ours, theirs, patch in pairs:
        a, b = f"{dump_dir}/{ours}", f"{ref_dir}/{theirs}"
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        x, y = rd(a), rd(b)
        if patch:
            y = tokens(y)
        rel = np.linalg.norm(x - y) / np.linalg.norm(y)
        print(f"{ours}: relative L2 error {rel:.3e} ({x.size} values)")
        ok &= rel < 1e-3
    a, b = f"{dump_dir}/cache.txt", f"{ref_dir}/cache.txt"
    if os.path.exists(b):
        x = open(a).read().split("\n") if os.path.exists(a) else []
        y = open(b).read().split("\n")
        same = x == y
        ran = sum(l.endswith(" 1") for l in y)
        print(f"cache decisions: {'identical' if same else 'DIFFERENT'} ({len(y) - 1} passes, {ran} ran the blocks)")
        ok &= same
    sys.exit(0 if ok else 1)

sys.path.insert(0, args.repo)
from diffusers.models.autoencoders import AutoencoderKL  # noqa: E402
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from omnigen2.models.transformers import OmniGen2Transformer2DModel  # noqa: E402
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline  # noqa: E402
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler  # noqa: E402

for d in [ref_dir, dump_dir, f"{model_dir}/transformer", f"{model_dir}/vae"]:
    os.makedirs(d, exist_ok=True)
for f in os.listdir(dump_dir):
    os.remove(f"{dump_dir}/{f}")

torch.manual_seed(0)
TEXT = 64
tcfg = dict(
    patch_size=2, in_channels=16, hidden_size=96, num_layers=4, num_refiner_layers=2,
    num_attention_heads=3, num_kv_heads=1, multiple_of=32, norm_eps=1e-5,
    axes_dim_rope=(8, 12, 12), axes_lens=(128, 64, 64), text_feat_dim=TEXT, timestep_scale=1000.0,
)
transformer = OmniGen2Transformer2DModel(**tcfg)
vae = AutoencoderKL(
    in_channels=3, out_channels=3, latent_channels=16, block_out_channels=(32, 32, 32, 32),
    down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
    layers_per_block=1, norm_num_groups=32, scaling_factor=0.3611, shift_factor=0.1159,
    use_quant_conv=False, use_post_quant_conv=False,
)


def randomize(m, std):
    """Random weights (the official init zeroes the modulation and output layers), bf16-exact."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 1 and ("norm" in n and "linear" not in n or n.endswith("caption_embedder.0.weight")):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif p.ndim == 1:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(std * torch.randn(p.shape, generator=g))
            p.copy_(p.to(torch.bfloat16).float())


randomize(transformer, 0.08)
# The VAE keeps its own init (a random one saturates), rounded to bf16.
with torch.no_grad():
    for p in vae.parameters():
        p.copy_(p.to(torch.bfloat16).float())
transformer.eval()
vae.eval()

save_file({k: v.contiguous() for k, v in transformer.state_dict().items()}, f"{model_dir}/transformer/diffusion_pytorch_model.safetensors")
json.dump({"_class_name": "OmniGen2Transformer2DModel", **{k: (list(v) if isinstance(v, tuple) else v) for k, v in tcfg.items()}},
          open(f"{model_dir}/transformer/config.json", "w"), indent=1)
save_file({k: v.contiguous() for k, v in vae.state_dict().items()}, f"{model_dir}/vae/diffusion_pytorch_model.safetensors")
json.dump(dict(vae.config), open(f"{model_dir}/vae/config.json", "w"), indent=1, default=list)
json.dump({"_class_name": "OmniGen2Pipeline"}, open(f"{model_dir}/model_index.json", "w"))


def wr(path, t):
    t.detach().to(torch.float32).cpu().contiguous().numpy().astype("<f4").tofile(path)


g = torch.Generator().manual_seed(2)
size = args.size
pos = torch.randn(1, 9, TEXT, generator=g)
neg = torch.randn(1, 7, TEXT, generator=g)
noise = torch.randn(1, 16, size // 8, size // 8, generator=g)
wr(f"{dump_dir}/prompt_embeds_in.bin", pos)
wr(f"{dump_dir}/negative_embeds_in.bin", neg)
wr(f"{dump_dir}/noise.bin", noise)
images = []
for i in range(args.images):
    # Smooth random images (sides multiples of 16: the processor leaves them as they are).
    s = size if i == 0 else size // 2
    low = torch.rand(1, 3, 4, 4, generator=g)
    im = torch.nn.functional.interpolate(low, size=(s, s), mode="bicubic").clamp(0, 1)
    im = Image.fromarray((im[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    im.save(f"{out}/in{i}.png")
    images.append(im)

# The posterior of the input images: the mean, or a sample with noise oxinfer is given too.
count = [0]


def sample(self, generator=None):
    i = count[0]
    count[0] += 1
    if not args.sample_posterior:
        return self.mean
    eps = torch.randn(self.mean.shape, generator=g)
    wr(f"{dump_dir}/ref_noise{i}_in.bin", eps)
    return self.mean + self.std * eps


DiagonalGaussianDistribution.sample = sample

if args.scheduler == "euler":
    scheduler = FlowMatchEulerDiscreteScheduler()
else:
    from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

    scheduler = DPMSolverMultistepScheduler(
        algorithm_type="dpmsolver++", solver_type="midpoint", solver_order=2, prediction_type="flow_prediction"
    )
pipe = OmniGen2Pipeline(transformer=transformer, vae=vae, scheduler=scheduler, mllm=None, processor=None)
if args.taylorseer:
    pipe.enable_taylorseer = True
elif args.teacache is not None:
    transformer.enable_teacache = True
    transformer.teacache_rel_l1_thresh = args.teacache

# Records: reference latents, the first guided velocity, the final latents, and per pass whether
# the joint blocks ran (the first of them, for the cache decisions).
enc0 = pipe.encode_vae
enc_n = [0]


def encode_vae(img):
    z = enc0(img)
    wr(f"{ref_dir}/ref_latents{enc_n[0]}.bin", z)
    enc_n[0] += 1
    return z


pipe.encode_vae = encode_vae
# encode_prompt applies the chat template (the processor's) even when embeddings are given.
pipe._apply_chat_template = lambda p: p
dec0 = vae.decode


def decode(z, *a, **k):
    wr(f"{ref_dir}/latents_out.bin", (z - vae.config.shift_factor) * vae.config.scaling_factor)
    return dec0(z, *a, **k)


vae.decode = decode
passes = []
ran = [False]
transformer.layers[0].register_forward_hook(lambda *a: ran.__setitem__(0, True))
step_no = [0]
branch_no = [0]


def before(*a, **k):
    ran[0] = False


def after(*a, **k):
    # Under TaylorSeer the first block is still called on a skipped step (it returns its
    # prediction): the reference's own flag says what the pass did.
    did = ran[0]
    if args.taylorseer:
        did = transformer.current["type"] == "full"
    passes.append((step_no[0], branch_no[0], int(did)))
    branch_no[0] += 1


# Hooks, not a wrapper of forward: the pipeline inspects forward's signature.
transformer.register_forward_pre_hook(before)
transformer.register_forward_hook(after)
step0 = scheduler.step
first = [True]


def sched_step(model_output, t, sample, **k):
    if first[0]:
        wr(f"{ref_dir}/v0.bin", model_output)
        first[0] = False
    step_no[0] += 1
    branch_no[0] = 0
    return step0(model_output, t, sample, **k)


scheduler.step = sched_step

with torch.no_grad():
    res = pipe(
        prompt="x",
        prompt_embeds=pos,
        negative_prompt_embeds=neg,
        prompt_attention_mask=torch.ones(1, pos.shape[1], dtype=torch.long),
        negative_prompt_attention_mask=torch.ones(1, neg.shape[1], dtype=torch.long),
        input_images=images if images else None,
        width=size, height=size,
        num_inference_steps=args.steps,
        text_guidance_scale=args.text_guidance,
        image_guidance_scale=args.image_guidance,
        latents=noise.clone(),
        output_type="pt",
    )
wr(f"{ref_dir}/image.bin", res.images[0])
open(f"{ref_dir}/cache.txt", "w").write("".join(f"{s} {b} {r}\n" for s, b, r in passes))
n_ran = sum(r for _, _, r in passes)
print(f"{len(passes)} transformer passes, {n_ran} ran the joint blocks; results in {ref_dir}")
