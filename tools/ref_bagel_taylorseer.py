#!/usr/bin/env python3
"""The official BAGEL code (github.com/ByteDance-Seed/Bagel), truncated to a few layers and run
in float32 on the CPU, for checks of oxinfer's BAGEL text-to-image path: with --taylorseer, the
official TaylorSeer step cache.

    python tools/ref_bagel_taylorseer.py REPO MODEL OUT [--layers 2] [--steps 30] [--size 256]
        [--taylorseer] [--cfg-img 1.5] [--prompt "a red fox"]

REPO is a checkout of the official repository, MODEL the BAGEL-7B-MoT folder. The first
`--layers` layers of the language model are loaded (the rest of a 7B model in float32 does not
fit a quick CPU check); oxinfer runs the same truncation with OXINFER_BAGEL_LAYERS. Writes
OUT/dump/noise_in.bin (the initial latent tokens, read by oxinfer through OXINFER_BAGEL_DUMP) and
OUT/ref/latents_out.bin, OUT/ref/cache.txt ("step branch ran" per language-model pass; branches:
0 conditional, 1 without the text, 2 without images).

Then:
  OXINFER_BAGEL_LAYERS=2 OXINFER_BAGEL_DUMP=OUT/dump oxinfer --model MODEL --mmap --device cpu \
      --size 256x256 --steps 30 -p "a red fox" -o OUT/out.png [--taylorseer] [--cfg-img 1.5]
and compare OUT/dump/latents_out.bin and OUT/dump/cache.txt with OUT/ref/ (--compare does it).

The repository's files are used as they are but for two things a CPU float32 run needs: the
`flash_attn_varlen_func` of modeling/bagel/qwen2_navit.py is replaced by the same attention on
torch's scaled_dot_product_attention, and the file's hard-coded `.to(torch.bfloat16)` casts are
removed (a patched copy is written into OUT/src). Needs torch, transformers 4.49 (what the
repository targets; newer ones changed the Qwen2 config and RoPE helpers), safetensors, pillow.
Memory: 20 GB with 2 layers (the float32 embedding table and LM head are most of it).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--taylorseer", action="store_true")
ap.add_argument("--cfg-text", type=float, default=4.0)
ap.add_argument("--cfg-img", type=float, default=1.0)
ap.add_argument("--prompt", default="a red fox")
ap.add_argument("--compare", action="store_true")
args = ap.parse_args()
out = args.out
ref_dir, dump_dir, src_dir = f"{out}/ref", f"{out}/dump", f"{out}/src"

if args.compare:
    x = np.fromfile(f"{dump_dir}/latents_out.bin", dtype="<f4").astype(np.float64)
    y = np.fromfile(f"{ref_dir}/latents_out.bin", dtype="<f4").astype(np.float64)
    rel = np.linalg.norm(x - y) / np.linalg.norm(y)
    print(f"latents_out.bin: relative L2 error {rel:.3e} ({x.size} values)")
    ok = rel < 1e-3
    b = f"{ref_dir}/cache.txt"
    if os.path.exists(b):
        a = f"{dump_dir}/cache.txt"
        xs = open(a).read() if os.path.exists(a) else ""
        ys = open(b).read()
        ran = sum(l.endswith(" 1") for l in ys.split("\n"))
        n = ys.count("\n")
        print(f"cache decisions: {'identical' if xs == ys else 'DIFFERENT'} ({n} passes, {ran} ran the model)")
        ok &= xs == ys
    sys.exit(0 if ok else 1)

# flash_attn_varlen_func for one packed sequence (what inference passes): grouped key-value
# heads, and flash-attn's causal mask aligned at the bottom right when there are cached keys.
SDPA = '''
def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False):
    assert len(cu_seqlens_q) == 2 and len(cu_seqlens_k) == 2
    (tq, h, _), (tk, hk, _) = q.shape, k.shape
    kk = k.transpose(0, 1).repeat_interleave(h // hk, dim=0)
    vv = v.transpose(0, 1).repeat_interleave(h // hk, dim=0)
    mask = torch.ones(tq, tk, dtype=torch.bool, device=q.device).tril(diagonal=tk - tq) if causal else None
    return scaled_dot_product_attention(q.transpose(0, 1), kk, vv, attn_mask=mask).transpose(0, 1)
'''

for d in [ref_dir, dump_dir]:
    os.makedirs(d, exist_ok=True)
for f in os.listdir(dump_dir):
    os.remove(f"{dump_dir}/{f}")
# A copy of the repository's Python packages with the bf16 casts of the packed attention removed.
if os.path.exists(src_dir):
    shutil.rmtree(src_dir)
for pkg in ["modeling", "data"]:
    shutil.copytree(f"{args.repo}/{pkg}", f"{src_dir}/{pkg}", ignore=shutil.ignore_patterns("*.pyc", "__pycache__"))
shutil.copy(f"{args.repo}/inferencer.py", f"{src_dir}/inferencer.py")
p = f"{src_dir}/modeling/bagel/qwen2_navit.py"
s = open(p).read().replace(".to(torch.bfloat16)", "")
s = s.replace("from flash_attn import flash_attn_varlen_func", SDPA)
open(p, "w").write(s)
# Only the language model is needed: keep the package import from pulling SigLIP's dependencies.
open(f"{src_dir}/modeling/__init__.py", "w").write("")
open(f"{src_dir}/modeling/bagel/__init__.py", "w").write("")
sys.path.insert(0, src_dir)

from safetensors import safe_open  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from data.data_utils import add_special_tokens  # noqa: E402
from inferencer import InterleaveInferencer  # noqa: E402
from modeling.autoencoder import AutoEncoderParams  # noqa: E402
from modeling.bagel.bagel import Bagel, BagelConfig  # noqa: E402
from modeling.bagel.qwen2_navit import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

torch.manual_seed(0)
if os.path.exists(f"{args.model}/llm_config.json"):
    llm_config = Qwen2Config.from_json_file(f"{args.model}/llm_config.json")
else:
    llm_config = Qwen2Config(**json.load(open(f"{args.model}/config.json"))["llm_config"])
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"
llm_config.num_hidden_layers = args.layers
if not hasattr(llm_config, "pad_token_id"):  # newer transformers no longer default it
    llm_config.pad_token_id = None
# The values of the repository's load_ae (the autoencoder itself is not needed here).
vae_config = AutoEncoderParams(
    resolution=256, in_channels=3, downsample=8, ch=128, out_ch=3, ch_mult=[1, 2, 4, 4],
    num_res_blocks=2, z_channels=16, scale_factor=0.3611, shift_factor=0.1159,
)
config = BagelConfig(
    visual_gen=True, visual_und=False, llm_config=llm_config, vit_config=None, vae_config=vae_config,
    vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh", latent_patch_size=2, max_latent_size=64,
)
language_model = Qwen2ForCausalLM(llm_config)
model = Bagel(language_model, None, config)
sd = model.state_dict()
with safe_open(f"{args.model}/ema.safetensors", "pt") as f:
    have = set(f.keys())
    missing = [k for k in sd if k not in have]
    if missing:
        raise SystemExit(f"tensors missing from the checkpoint: {missing[:5]}")
    for k in sd:
        sd[k] = f.get_tensor(k).to(torch.float32)
model.load_state_dict(sd)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
inferencer = InterleaveInferencer(model, None, tokenizer, None, None, new_token_ids)


def wr(path, t):
    t.detach().to(torch.float32).cpu().contiguous().numpy().astype("<f4").tofile(path)


# The initial noise goes to oxinfer; the final latents and the cache decisions are recorded.
prep0 = model.prepare_vae_latent


def prepare_vae_latent(*a, **k):
    r = prep0(*a, **k)
    wr(f"{dump_dir}/noise_in.bin", r["packed_init_noises"])
    return r


model.prepare_vae_latent = prepare_vae_latent
inferencer.decode_image = lambda latent, shape: wr(f"{ref_dir}/latents_out.bin", latent)

lm = language_model.model
passes = []
step_of = {}


def after(module, a, k, r):
    if not getattr(module, "enable_taylorseer", False):
        return
    # The branch is known by its cache dictionary (one per branch, created in this order).
    cur = module.current
    branch = step_of.setdefault(id(cur), len(step_of))
    # `step` was already advanced by the forward.
    passes.append((cur["step"] - 1, branch, int(cur["type"] == "full")))


lm.register_forward_hook(after, with_kwargs=True)
# The order the passes of a step run in is conditional, text, image; the step printed is the
# denoising step, which for every branch is its own pass count (they all start at step 0).
with torch.no_grad():
    inferencer.interleave_inference(
        [args.prompt],
        cfg_text_scale=args.cfg_text,
        cfg_img_scale=args.cfg_img,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=args.steps,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(args.size, args.size),
        enable_taylorseer=args.taylorseer,
    )
if args.taylorseer:
    passes.sort(key=lambda p: (p[0], p[1]))
    open(f"{ref_dir}/cache.txt", "w").write("".join(f"{s} {b} {r}\n" for s, b, r in passes))
    print(f"{len(passes)} language-model passes, {sum(r for _, _, r in passes)} ran the model")
print(f"results in {ref_dir}")
