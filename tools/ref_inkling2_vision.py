"""Reference features of Inkling-2's vision tower alone (transformers' `Inkling2VisionModel` and
`Inkling2ImageProcessor`, float32, CPU), for a checkpoint that holds only the tower's tensors
under the release's names (`model.visual.*`): the real tower is 0.66 GB at the end of the last
shard of thinkingmachines/Inkling-2-Small and can be fetched by HTTP range request.

usage: ref_inkling2_vision.py MODEL_DIR IMAGE OUT.safetensors

MODEL_DIR holds `config.json` (the release's) and `model.safetensors`. Writes `pixel_values`,
`image_grid_thw` and `image_features` (`[tokens, text hidden]`, what replaces the image
placeholders), to compare with
`cargo run --release --example inkling2_check -- MODEL_DIR OUT.safetensors --image IMAGE --vision-only`.
"""
import json, os, re, sys
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor
from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel

mdir, image, out = sys.argv[1:4]
raw = json.load(open(os.path.join(mdir, "config.json")))
v = dict(raw["vision_config"])
# The release's spelling of the settings transformers names otherwise.
for old, new in [("native_patch_size", "patch_size"), ("adapter_merge_size", "spatial_merge_size")]:
    if old in v:
        v[new] = v.pop(old)
v.pop("temporal_patch_size", None)
v.setdefault("text_hidden_size", raw["text_config"]["hidden_size"])
cfg = Inkling2VisionConfig(**v)
cfg._attn_implementation = "eager"
m = Inkling2VisionModel(cfg).float().eval()

RENAMES = [
    (r"^model\.visual\.backbone\.embeddings\.position_embedding$", "embeddings.position_embedding.weight"),
    (r"^model\.visual\.backbone\.embeddings\.", "embeddings."),
    (r"^model\.visual\.backbone\.ln_pre\.", "pre_layernorm."),
    (r"^model\.visual\.backbone\.layers\.", "layers."),
    (r"^model\.visual\.proj_in\.", "merger.linear_1."),
    (r"^model\.visual\.proj_out\.", "merger.linear_2."),
    (r"\.ln_1\.", ".layernorm_before."),
    (r"\.ln_2\.", ".layernorm_after."),
    (r"\.mlp\.c_fc\.", ".mlp.fc1."),
    (r"\.mlp\.c_proj\.", ".mlp.fc2."),
    (r"\.ls_1\.gamma$", ".lambda_1"),
    (r"\.ls_2\.gamma$", ".lambda_2"),
]
state = {}
for name, w in load_file(os.path.join(mdir, "model.safetensors")).items():
    if not name.startswith("model.visual."):
        continue
    for a, b in RENAMES:
        name = re.sub(a, b, name)
    state[name] = w.float()
m.load_state_dict(state, strict=True)

inp = Inkling2ImageProcessor()(images=[Image.open(image).convert("RGB")], return_tensors="pt")
with torch.no_grad():
    feats = m(pixel_values=inp["pixel_values"].float(), image_grid_thw=inp["image_grid_thw"]).pooler_output
print("grid", inp["image_grid_thw"].tolist(), "features", tuple(feats.shape), "rms", float(feats.pow(2).mean().sqrt()))
save_file(
    {
        "pixel_values": inp["pixel_values"].float().contiguous(),
        "image_grid_thw": inp["image_grid_thw"].to(torch.int64),
        "image_features": feats.float().contiguous(),
    },
    out,
)
