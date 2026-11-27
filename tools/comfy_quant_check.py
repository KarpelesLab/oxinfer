"""Compares `examples/comfy_quant_check.rs`'s `W x` with comfy_kitchen's dequantized weight.

    python tools/comfy_quant_check.py COMFY_KITCHEN_SRC FILE.safetensors TENSOR.weight DIR [BF16_FILE]

COMFY_KITCHEN_SRC is a checkout of github.com/Comfy-Org/comfy-kitchen (its eager backend is the
reference; no build needed). DIR holds the example's x.bin / y.bin. With BF16_FILE (the same
model unquantized, e.g. Sulphur-3's s3.safetensors beside quantized/s3_w4a8.safetensors) the
quantization error of the weight itself is printed too.
"""
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

src, path, name, d = sys.argv[1:5]
sys.path.insert(0, src)
from comfy_kitchen.backends.eager.quantization import rotate_int8_convrot_weight  # noqa: E402
from comfy_kitchen.backends.eager.w4a8_int8 import dequantize_w4a8_int8_weight  # noqa: E402

base = name[: -len(".weight")]
with safe_open(path, "pt") as f:
    keys = set(f.keys())
    if base + ".comfy_quant" in keys:
        conf = json.loads(bytes(f.get_tensor(base + ".comfy_quant").tolist()).decode())
    else:  # the entries in the header (`_quantization_metadata`)
        conf = json.loads(f.metadata()["_quantization_metadata"])["layers"][base]
    q = f.get_tensor(name)
    if conf["format"] in ("asym_w4a8_int8", "w6a8_int8"):
        s_rel = f.get_tensor(base + ".weight_s_rel")
        if s_rel.dtype == torch.uint8:
            s_rel = s_rel.view(torch.float8_e4m3fn)
        book = f.get_tensor(base + ".weight_codebook") if base + ".weight_codebook" in keys else None
        w = dequantize_w4a8_int8_weight(q, s_rel, f.get_tensor(base + ".weight_s_channel"), codebook=book,
                                        group_size=conf.get("group_size", 16),
                                        convrot_groupsize=conf.get("convrot_groupsize", 256), output_dtype=torch.float32)
    else:
        w = q.float() * f.get_tensor(base + ".weight_scale").float()
        if conf.get("convrot"):
            w = rotate_int8_convrot_weight(w, conf.get("convrot_groupsize", 256)).float()
n_out, n_in = w.shape
x = torch.from_numpy(np.fromfile(f"{d}/x.bin", dtype=np.float32)).reshape(-1, n_in)
y = torch.from_numpy(np.fromfile(f"{d}/y.bin", dtype=np.float32)).reshape(-1, n_out)
ref = (x.double() @ w.double().T).float()
rel = float((y - ref).norm() / ref.norm())
print(f"{name} ({conf['format']}): [{n_out}, {n_in}], oxinfer against comfy_kitchen: relative L2 {rel:.2e}, "
      f"max abs {float((y - ref).abs().max()):.2e} (values up to {float(ref.abs().max()):.2f})")
if len(sys.argv) > 5:
    with safe_open(sys.argv[5], "pt") as f:
        wb = f.get_tensor(name).float()
    e = float((w - wb).norm() / wb.norm())
    c = float(torch.nn.functional.cosine_similarity(w.flatten(), wb.flatten(), dim=0))
    yb = (x.double() @ wb.double().T).float()
    print(f"  against {sys.argv[5]}: weight relative L2 {e:.4f}, cosine {c:.5f}; "
          f"oxinfer's products against the bf16 weight's: relative L2 {float((y - yb).norm() / yb.norm()):.4f}")
