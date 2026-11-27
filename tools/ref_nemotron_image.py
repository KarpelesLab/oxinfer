#!/usr/bin/env python3
"""Nemotron-Labs-Image tokenizer reference dumps (NVIDIA's own `modeling_l24sq_vit.py` from the
checkpoint's `Nemotron-Labs-Image-Tokenizer/` folder, CPU), in the formats of `cargo run --release
--example nemotron_image_check`:

  python tools/ref_nemotron_image.py TOKDIR encode IMAGE.png OUTDIR [--bf16-weights]
      pixels.bin    the image as the encoder reads it: [3, H, W] f32 in [-1, 1]
      codes.txt     the codebook indices, row by row, after a line "GH GW"
      decoded.bin   the decoder's output for those indices: [3, H, W] f32 (clamped)
  python tools/ref_nemotron_image.py TOKDIR decode CODES.txt OUT.bin [--bf16-weights]
      the decoder's output for the indices of CODES.txt (first line "GH GW")

`--bf16-weights` rounds every weight to bf16 (kept in float32 arithmetic), which is what oxinfer
holds by default: the comparison then measures the implementation, not the rounding.

Needs torch, einops, safetensors, numpy, pillow. transformers is not needed: a two-class stand-in
for `PreTrainedModel` / `PretrainedConfig` is installed when it is missing.
"""
import importlib.util
import os
import sys
import types

import numpy as np
import torch

torch.set_grad_enabled(False)


def stub_transformers():
    try:
        import transformers  # noqa: F401
        return
    except ImportError:
        pass

    class PretrainedConfig:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class PreTrainedModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def post_init(self):
            pass

    m = types.ModuleType("transformers")
    m.PretrainedConfig = PretrainedConfig
    m.PreTrainedModel = PreTrainedModel
    sys.modules["transformers"] = m


def load(tokdir, bf16):
    import json

    from safetensors.torch import load_file

    stub_transformers()
    pkg = types.ModuleType("l24sq")
    pkg.__path__ = [tokdir]
    sys.modules["l24sq"] = pkg
    for name in ["configuration_l24sq_vit", "modeling_l24sq_vit"]:
        spec = importlib.util.spec_from_file_location(f"l24sq.{name}", os.path.join(tokdir, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"l24sq.{name}"] = mod
        spec.loader.exec_module(mod)
    cfgd = json.load(open(os.path.join(tokdir, "config.json")))
    cfgd = {k: v for k, v in cfgd.items() if k not in ("auto_map", "model_type", "transformers_version")}
    cfg = sys.modules["l24sq.configuration_l24sq_vit"].L24SQViTConfig(**cfgd)
    model = sys.modules["l24sq.modeling_l24sq_vit"].L24SQViTModel(cfg).eval()
    sd = load_file(os.path.join(tokdir, "model.safetensors"))
    if bf16:
        sd = {k: v.to(torch.bfloat16).float() for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit(f"missing weights: {missing}")
    return model


def write_codes(path, codes, grid):
    with open(path, "w") as f:
        f.write(f"{grid[0]} {grid[1]}\n")
        f.write(" ".join(str(int(c)) for c in codes) + "\n")


def read_codes(path):
    lines = open(path).read().split("\n", 1)
    gh, gw = map(int, lines[0].split())
    codes = [int(c) for c in lines[1].split()]
    return codes, (gh, gw)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bf16 = "--bf16-weights" in sys.argv
    tokdir, mode = args[0], args[1]
    model = load(tokdir, bf16)
    if mode == "encode":
        from PIL import Image

        img, out = args[2], args[3]
        os.makedirs(out, exist_ok=True)
        im = np.asarray(Image.open(img).convert("RGB"), dtype=np.float32) / 255.0
        x = model.preprocess_image(torch.from_numpy(im).permute(2, 0, 1).contiguous())
        x.numpy().astype(np.float32).tofile(os.path.join(out, "pixels.bin"))
        enc = model.encode(x[None])
        codes = enc["indices"][0].tolist()
        grid = enc["grid_size"]
        write_codes(os.path.join(out, "codes.txt"), codes, grid)
        dec = model.decode(indices=enc["indices"], grid_size=grid)[0]
        dec.numpy().astype(np.float32).tofile(os.path.join(out, "decoded.bin"))
        print(f"grid {grid}, {len(codes)} codes; decoded {tuple(dec.shape)}")
    elif mode == "decode":
        codes, grid = read_codes(args[2])
        idx = torch.tensor(codes, dtype=torch.long)[None]
        dec = model.decode(indices=idx, grid_size=grid)[0]
        dec.numpy().astype(np.float32).tofile(args[3])
        print(f"decoded {tuple(dec.shape)}")
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
