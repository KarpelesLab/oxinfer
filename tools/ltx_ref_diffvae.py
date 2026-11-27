"""LTX-2.5 DiffVAE decoder reference (float32, CPU) for `examples/ltx_check.rs diffvae`: decodes
random latents with the release's code and records the noise it draws.

usage: LTX_CORE=.../LTX-2/packages/ltx-core/src python ltx_ref_diffvae.py VAE_FILE OUT_DIR [T H W] [MODE]
MODE: chunked (the pipelines' default, DiffVAEMode.CHUNKED_EAGER) or combined.
Writes lat.f32 ([128, T, H, W]), noise.f32 (every torch.randn draw of the decode, in order),
px.f32 ([3, F, 32 H, 32 W] in [-1, 1]) and meta.json. Needs torch, einops, safetensors, numpy
(github.com/Lightricks/LTX-2's ltx-core on LTX_CORE; natten not needed: the eager fallback runs).
"""
import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.environ["LTX_CORE"])


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return None

    def __getattr__(self, n):
        return _Dummy()


def _stub_attr(n):
    if n.startswith("__"):
        raise AttributeError(n)
    return _Dummy


# ltx_core.color needs colour-science; the decoder does not use it.
for _n in ["ltx_core.color", "ltx_core.color.hlg", "ltx_core.color.primaries", "ltx_core.color.yuv"]:
    _m = types.ModuleType(_n)
    _m.__getattr__ = _stub_attr
    _m.__path__ = []
    sys.modules[_n] = _m

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator  # noqa: E402
from ltx_core.model.video_vae.transformer import apply_diffvae_config  # noqa: E402
from ltx_core.model.video_vae.transformer.config import (  # noqa: E402
    DiffVAEBlockKind,
    DiffVAEConfig,
    NAttentionKind,
)

path, out = sys.argv[1], sys.argv[2]
t, h, w = (int(v) for v in sys.argv[3:6]) if len(sys.argv) >= 6 else (2, 4, 6)
mode = sys.argv[6] if len(sys.argv) > 6 else "chunked"
os.makedirs(out, exist_ok=True)
torch.set_num_threads(16)
with safe_open(path, "pt") as f:
    meta = f.metadata()
    sd = {}
    for k in f.keys():
        if not (k.startswith("decoder.") or k.startswith("per_channel_statistics.")):
            continue
        v = f.get_tensor(k).float()
        k = k.removeprefix("decoder.")
        k = k.replace("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1.")
        k = k.replace("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2.")
        if k.endswith(".qkv.weight") or k.endswith(".qkv.bias"):
            leaf = k.rsplit(".", 1)[1]
            pre = k[: -len("qkv." + leaf)]
            for part, c in zip("qkv", v.chunk(3, 0)):
                sd[f"{pre}qkv.to_{part}.{leaf}"] = c.clone()
            continue
        sd[k] = v
cfg = json.loads(meta["config"])
dec = VideoDecoderConfigurator.from_metadata({"config": cfg})
missing, unexpected = dec.load_state_dict(sd, strict=False, assign=True)
print("missing", [m for m in missing if "rope_inv" not in m][:10], "unexpected", unexpected[:10])
if mode == "chunked":
    dcfg = DiffVAEConfig(
        block=DiffVAEBlockKind.CHUNKED,
        w_chunks=4,
        natten_backend=None,
        attention=NAttentionKind.EAGER_SDPA,
        compile_blocks=False,
        compile_det_stages=False,
    )
else:
    dcfg = DiffVAEConfig(
        block=DiffVAEBlockKind.COMBINED,
        w_chunks=1,
        natten_backend=None,
        attention=NAttentionKind.EAGER_SDPA,
        compile_blocks=False,
        compile_det_stages=False,
    )
apply_diffvae_config(dec, dcfg)
dec = dec.float().eval()
print(type(dec).__name__, type(dec.diff_blocks[0]).__name__, dec.deferred_stage4_upsample)

rng = np.random.default_rng(3)
lat = rng.standard_normal((128, t, h, w)).astype(np.float32)
draws = []
_randn = torch.randn


def rec_randn(*a, **k):
    x = _randn(*a, **k)
    draws.append(x.detach().float().numpy().ravel().copy())
    return x


torch.randn = rec_randn
g = torch.Generator("cpu").manual_seed(7)
with torch.no_grad():
    px = dec(torch.from_numpy(lat)[None], generator=g)[0].numpy()
torch.randn = _randn
noise = np.concatenate(draws) if draws else np.zeros(0, np.float32)
for n, a in [("lat", lat), ("px", px), ("noise", noise)]:
    np.ascontiguousarray(a, dtype=np.float32).tofile(os.path.join(out, n + ".f32"))
json.dump({"t": t, "h": h, "w": w, "draws": [int(d.size) for d in draws], "px": list(px.shape), "mode": mode}, open(os.path.join(out, "meta.json"), "w"))
print("px", px.shape, float(np.abs(px).mean()), "draws", [d.size for d in draws])
