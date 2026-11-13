"""Tiny random Fish Audio Drama 3 (`fish_omni`) checkpoint for tools/ref_fish_drama.py and
examples/fish_check.rs: the backbone and the codebook tower of fish-speech's `fish_omni` with
random weights, written under the release's tensor names.

usage: fish_drama_tiny.py FISH_SPEECH_DIR RELEASE_DIR OUT_DIR [--fp8] [--seed S]

RELEASE_DIR is a folder of fishaudio/drama-3 holding at least tokenizer.json and codec.pth: the
tiny model takes the release's vocabulary (so the prompt is the real one) and its codec, both
linked into OUT_DIR. The weights are random values that bf16 holds exactly, stored as bf16, so
that a float32 reference and oxinfer compute on the same numbers. With --fp8 the experts are
written as the release stores them (`float8_e4m3fn` with one float32 `weight_scale_inv` per
128 x 128 block; the reference keeps them in FP8 and dequantizes per forward on the CPU).

Needs torch, safetensors and fish-speech's sources (FISH_SPEECH_DIR).
"""
import json, os, sys

import torch
from safetensors.torch import save_file

fish, release, out = sys.argv[1:4]
fp8 = "--fp8" in sys.argv
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
sys.path.insert(0, fish)

from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder

real = json.load(open(os.path.join(release, "config.json")))
H = 128
text = dict(
    model_type="fish_omni_text",
    vocab_size=real["text_config"]["vocab_size"],
    hidden_size=H,
    num_hidden_layers=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    mlp_layer_types=["moe"] * 4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    hidden_act="silu",
    linear_conv_kernel_dim=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=128,
    shared_expert_intermediate_size=128,
    intermediate_size=128,
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    rope_parameters=dict(rope_type="default", rope_theta=10000000.0, partial_rotary_factor=0.25),
    attn_output_gate=True,
    use_qk_norm=True,
    layernorm_zero_centered_gamma=True,
    shared_expert_gate=True,
    rotary_interleaved=False,
    tie_word_embeddings=False,
    eos_token_id=real["eos_token_id"],
)
tower = dict(real["audio_decoder_config"], dim=H, text_dim=H, intermediate_size=256, n_head=4, n_local_heads=2, head_dim=32)
cfg = {k: real[k] for k in ("semantic_start_token_id", "semantic_end_token_id", "eos_token_id", "speech_template_version")}
cfg.update(model_type="fish_omni", architectures=["FishOmniModel"], text_config=text, audio_decoder_config=tower, audio_encoder_config=None)
if fp8:
    cfg["quantization_config"] = dict(quant_method="bsgemm-moe", weight_block_size=[128, 128], activation_scheme="dynamic")

torch.manual_seed(seed)
config = FishOmniConfig.from_dict(cfg)
backbone = FishOmniBackbone(config.text, fp8_experts=fp8, dtype=torch.float32, device="cpu")
decoder = FishOmniAudioDecoder(config, dtype=torch.float32, device="cpu")
bf16 = lambda t: t.to(torch.bfloat16)
with torch.no_grad():
    for module in (backbone, decoder):
        for p in module.parameters():
            p.normal_(0.0, 0.1 if p.ndim >= 2 else 0.3)
            p.copy_(bf16(p).float())
    for name, b in backbone.named_buffers():
        if name.endswith("_weight"):
            b.copy_(torch.randn(b.shape).to(b.dtype))
        elif name.endswith("_scale_inv"):
            b.copy_(torch.rand(b.shape) * 0.01)

tensors = {}
state = dict(backbone.named_parameters())
state.update(backbone.named_buffers())
for name, t in state.items():
    t = t.detach()
    if ".mlp.experts." in name:
        prefix, leaf = name.rsplit(".", 1)
        which, kind = leaf.split("_", 1)  # w13 / w2, weight / scale_inv
        suffix = "weight" if kind == "weight" else "weight_scale_inv"
        prefix = "model.language_model." + prefix[len("model.") :]
        for e in range(t.shape[0]):
            parts = [("down_proj", t[e])] if which == "w2" else list(zip(("gate_proj", "up_proj"), t[e].chunk(2)))
            for proj, m in parts:
                m = m.contiguous()
                tensors[f"{prefix}.{e}.{proj}.{suffix}"] = m if (fp8 and kind == "weight") or kind != "weight" else bf16(m)
    elif name == "lm_head.weight":
        tensors[name] = bf16(t)
    else:
        tensors["model.language_model." + name[len("model.") :]] = bf16(t)
names = {
    "codebook_embeddings.": "audio_decoder.codebook_embeddings.",
    "shared_layers.": "audio_decoder_mtp_shared.layers.",
    "residual_layers.": "audio_decoder_mtp.layers.",
    "codebook_heads.": "audio_decoder_mtp.heads.",
}
for name, t in decoder.named_parameters():
    old = next(o for o in names if name.startswith(o))
    tensors[names[old] + name[len(old) :]] = bf16(t.detach())

os.makedirs(out, exist_ok=True)
save_file(tensors, os.path.join(out, "model.safetensors"))
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
for f in ("tokenizer.json", "codec.pth"):
    dst = os.path.join(out, f)
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(os.path.join(release, f)), dst)
print(f"{out}: {len(tensors)} tensors, {sum(t.numel() for t in tensors.values()) / 1e6:.1f}M values, experts {'FP8' if fp8 else 'bf16'}")
