"""A tiny random H3 corrector and its outputs, for examples/h3_corrector_check.rs.

    python tools/minimax_h3_corrector_tiny.py PACK_DIR OUT_DIR

PACK_DIR is the `ComfyUI-H3-Corrector` folder of SulphurAI/Sulphur-3's `S3_ComfyUI.zip` (its
`corrector.py` defines the network; nothing else of ComfyUI is needed). Writes OUT_DIR/corrector.safetensors
(random float32 weights, the architecture in the `h3_corrector` metadata entry, as the release's
files) and OUT_DIR/io.safetensors: a latent `z` and a velocity `v` ([1, C, T, H, W]), `text`
([1, L, text_dim]), `sigma` / `sigma_next`, the network's raw residual `raw` and the capped one
`applied` (`bound_residual` at 0.5 / 0.001), for two latent shapes.
"""
import importlib.util
import os
import sys

import torch
from safetensors.torch import save_file

pack, out = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("h3_corrector_ref", os.path.join(pack, "corrector.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["h3_corrector_ref"] = mod
spec.loader.exec_module(mod)

torch.manual_seed(0)
cfg = mod.CorrectorConfig(video_channels=6, text_dim=40, width=64, depth=3, heads=4, head_dim=16, mlp_ratio=4,
                          time_embed_dim=32, patch=(1, 2, 2))
net = mod.H3Corrector(cfg).float().eval()
with torch.no_grad():
    for name, p in net.named_parameters():
        # Larger than the default init so every part (modulation, gates, rotary) matters.
        p.copy_(torch.randn_like(p) * (0.15 if p.dim() > 1 else 0.3))
os.makedirs(out, exist_ok=True)
save_file({k: v.contiguous() for k, v in net.state_dict().items()}, os.path.join(out, "corrector.safetensors"),
          metadata=cfg.to_metadata())

io = {}
for i, (t, h, w, l, s, sn) in enumerate([(3, 4, 6, 5, 0.9, 0.7), (2, 6, 4, 7, 0.3, 0.0)]):
    z = torch.randn(1, cfg.video_channels, t, h, w)
    v = torch.randn(1, cfg.video_channels, t, h, w) * 0.8
    text = torch.randn(1, l, cfg.text_dim) * 2.0
    with torch.no_grad():
        raw = net(z, v, s, sn, text)
    applied, ratio = mod.bound_residual(raw, v, 0.5, 0.001)
    io.update({f"{i}.z": z[0], f"{i}.v": v[0], f"{i}.text": text[0], f"{i}.raw": raw[0], f"{i}.applied": applied[0],
               f"{i}.sigma": torch.tensor([s, sn]), f"{i}.ratio": ratio.reshape(1)})
    print(f"case {i}: [{cfg.video_channels}, {t}, {h}, {w}], {l} text rows, raw rms {raw.square().mean().sqrt():.4f}, "
          f"ratio {float(ratio):.4f}")
save_file(io, os.path.join(out, "io.safetensors"))
