"""Tiny random Inkling-2 (`inkling2`) checkpoint + transformers reference outputs, every text
feature on: sliding-window layers whose heads differ from the full layers', a window and a
relative-position extent shorter than the prompt, the QK and relative L2 clips biting on part of
the heads, log scaling of the queries from a small position on, static and dynamic short
convolutions, attention sinks and output gates, the attention residual over the block states
(one and three queries), the MoE with a selection bias and soft caps that bite, the final soft
cap and an unembedding wider than the vocabulary.

usage: inkling2_tiny.py OUT_DIR [--len N] [--gen N] [--seed S] [--packed]
                        [--tokenizer RELEASE_DIR [--chat CONVERSATION.json]]
                        [--image FILE.png]
(environment: WINDOW, EXTENT, FLOOR, QK_CLIP, REL_CLIP, KERNEL, BLOCK override
`sliding_window_size`, `rel_extent`, `log_scaling_n_floor`, the two clip thresholds,
`conv_kernel_size` and `n_layers_per_attn_res_block`)

With --tokenizer the model takes the release's vocabulary and its tokenizer files (a folder with
tokenizer.json, tokenizer_config.json, special_tokens_map.json, chat_template.jinja and
processor_config.json), and with --chat the prompt is that conversation ({"messages": [...],
"tools": [...], "reasoning_effort": ...}) rendered by the chat template (written to
OUT_DIR/prompt.txt). With --image (and --tokenizer) the prompt is one user message with that
image and a text, through the processor; `pixel_values` and `image_grid_thw` go to the
reference file, and `image_features` (the projected features, one row per image token).

With --packed the routed experts are stored as the releases store them: the layers of the first
half in NVFP4 (E2M1 pairs, an E4M3 scale per 16, `amax` per expert over one denominator), the
others in MXFP8 (E4M3 values, an E8M0 exponent per 32). The reference then runs the dequantized
weights, so both sides compute the same model.

Writes OUT_DIR/model (the release's tensor names: `model.llm.*`, `attn_res.*`, the experts'
`w13_weight` with gate and up rows interleaved) and OUT_DIR/ref.safetensors: `ids` (I64),
`logits` [len, vocab] (one uncached forward), `gen` (I64, greedy tokens) and `decode_logits`
[gen, vocab] (after feeding each of them through the cache). Needs a transformers that has
`inkling2` (5.24; float32, CPU, eager attention).
"""
import json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

out = sys.argv[1]
plen, gen, seed, tokdir, chat, image, packed = 150, 16, 0, None, None, None, False
a = sys.argv[2:]
for i, x in enumerate(a):
    if x == "--len":
        plen = int(a[i + 1])
    if x == "--gen":
        gen = int(a[i + 1])
    if x == "--seed":
        seed = int(a[i + 1])
    if x == "--tokenizer":
        tokdir = a[i + 1]
    if x == "--chat":
        chat = a[i + 1]
    if x == "--image":
        image = a[i + 1]
    if x == "--packed":
        packed = True
torch.manual_seed(seed)
# With the release's tokenizer: its vocabulary (the unembedding is padded) and its stop token.
V, UNPADDED, EOS, IMAGE = (201024, 200058, 200006, 200054) if tokdir else (512, 500, 1, 499)

from transformers.models.inkling2.configuration_inkling2 import Inkling2Config
from transformers.models.inkling2.modeling_inkling2 import Inkling2ForConditionalGeneration

n = 12
text = dict(
    vocab_size=V,
    unpadded_vocab_size=UNPADDED,
    hidden_size=64,
    num_hidden_layers=n,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=16,
    swa_num_attention_heads=4,
    swa_num_key_value_heads=2,
    swa_head_dim=8,
    sliding_window_size=int(os.environ.get("WINDOW", "16")),
    d_rel=8,
    rel_extent=int(os.environ.get("EXTENT", "24")),
    log_scaling_n_floor=int(os.environ.get("FLOOR", "32")),
    log_scaling_alpha=0.3,
    conv_kernel_size=int(os.environ.get("KERNEL", "4")),
    moe_intermediate_size=64,
    n_routed_experts=16,
    num_experts_per_tok=4,
    route_scale=8.0,
    qk_l2_clip_threshold=float(os.environ.get("QK_CLIP", "2.0")),
    rel_l2_clip_threshold=float(os.environ.get("REL_CLIP", "2.0")),
    n_layers_per_attn_res_block=int(os.environ.get("BLOCK", "6")),
    mlp_gate_softcap=0.8,
    mlp_up_softcap=1.5,
    final_logit_softcapping=3.0,
    rms_norm_eps=1e-6,
    eos_token_id=EOS,
    initializer_range=0.08,
    max_position_embeddings=1 << 20,
)
vision = dict(
    hidden_size=32,
    num_hidden_layers=3,
    num_attention_heads=2,
    patch_size=20,
    spatial_merge_size=2,
    posemb_grid_size=8,
    projection_hidden_dim=48,
    text_hidden_size=64,
    rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
)
cfg = Inkling2Config(text_config=text, vision_config=vision, image_token_id=IMAGE)
cfg._attn_implementation = "eager"
m = Inkling2ForConditionalGeneration(cfg).float().eval()

with torch.no_grad():
    for name, p in m.named_parameters():
        if "vision_tower" in name:
            if "layernorm" in name and name.endswith("weight"):
                p.copy_(1.0 + torch.randn_like(p) * 0.2)
            elif name.endswith("bias"):
                p.copy_(torch.randn_like(p) * 0.1)
            elif "lambda_" in name:
                p.copy_(0.5 + torch.randn_like(p) * 0.2)
            elif name.endswith("embedding"):
                p.copy_(torch.randn_like(p) * 0.3)
            continue
        if name.endswith("norm.weight") or "layernorm" in name:
            p.copy_(1.0 + torch.randn_like(p) * 0.2)
        elif name.endswith(".query"):
            p.copy_(torch.randn_like(p) * 2.0)
        elif name.endswith("conv1d.weight") or name.endswith("static_weight"):
            p.copy_(torch.randn_like(p) * 0.4)
        elif name.endswith("l2_clip.scale"):
            p.copy_(1.0 + torch.randn_like(p) * 0.4)
        elif name.endswith("rel_logits_proj.proj"):
            p.copy_(torch.randn_like(p) * 0.6)
        elif name.endswith("sink_proj.weight") or name.endswith("gate_proj.weight"):
            p.copy_(torch.randn_like(p) * 0.3)
        elif name.endswith("weight_proj.weight"):
            p.copy_(torch.randn_like(p) * 0.2)
        elif name.endswith("mlp.gate.weight"):
            p.copy_(torch.randn_like(p) * 0.3)
        elif "mlp.experts" in name:
            p.copy_(torch.randn_like(p) * 0.3)
        elif name == "lm_head.weight":
            p.copy_(torch.randn_like(p) * 0.3)
    for name, b in m.named_buffers():
        if name.endswith("e_score_correction_bias"):
            b.copy_(torch.randn_like(b) * 0.1)
    # bf16-representable weights: oxinfer's bf16 matmuls are then exact.
    for name, p in m.state_dict().items():
        if p.is_floating_point():
            p.copy_(p.to(torch.bfloat16).float())

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)


def e4m3(x):
    """Rounds to float8_e4m3fn; returns (values, bytes)."""
    q = x.to(torch.float8_e4m3fn)
    return q.float(), q.view(torch.uint8)


def nvfp4(w):
    """[E, rows, cols] -> (pairs U8 [E, rows, cols / 2], scales E4M3 bytes [E, rows, cols / 16],
    amax [E], denominator [1], the values the codes stand for)."""
    e, rows, cols = w.shape
    amax = w.abs().amax(dim=(1, 2))
    den = torch.tensor([6.0 * 448.0])
    outer = (amax / den).reshape(-1, 1, 1)
    blocks = w.reshape(e, rows, cols // 16, 16)
    sv, sb = e4m3(blocks.abs().amax(-1) / 6.0 / outer)
    sv = sv.clamp_min(1e-30)
    x = blocks / (sv.unsqueeze(-1) * outer.unsqueeze(-1))
    idx = (x.abs().unsqueeze(-1) - E2M1).abs().argmin(-1)
    codes = (idx | ((x < 0) & (idx > 0)).long() << 3).to(torch.uint8).reshape(e, rows, cols)
    vals = E2M1[idx] * torch.where((x < 0) & (idx > 0), -1.0, 1.0) * sv.unsqueeze(-1) * outer.unsqueeze(-1)
    pairs = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return pairs, sb.reshape(e, rows, cols // 16), amax, den, vals.reshape(e, rows, cols)


def mxfp8(w):
    """[E, rows, cols] -> (E4M3 bytes, E8M0 exponents U8 [E, rows, cols / 32], values)."""
    e, rows, cols = w.shape
    blocks = w.reshape(e, rows, cols // 32, 32)
    ex = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-30) / 448.0)).clamp(-127, 127)
    scale = torch.pow(2.0, ex).unsqueeze(-1)
    qv, qb = e4m3(blocks / scale)
    return qb.reshape(e, rows, cols), (ex + 127).to(torch.uint8), (qv * scale).reshape(e, rows, cols)


os.makedirs(out, exist_ok=True)
mdir = os.path.join(out, "model")
m.save_pretrained(mdir)
if packed:
    # Rewrite the experts in the releases' packed formats, and give the reference their values.
    path = os.path.join(mdir, "model.safetensors")
    sd = load_file(path)
    lm = m.model.language_model
    for l in range(n):
        for leaf, attr in [("w13_weight", "gate_up_proj"), ("w2_weight", "down_proj")]:
            key = f"model.llm.layers.{l}.mlp.experts.{leaf}"
            w = sd.pop(key).float()
            if l < n // 2:
                pairs, scales, amax, den, vals = nvfp4(w)
                sd[key] = pairs
                sd[key + "_scale"] = scales.view(torch.float8_e4m3fn)
                sd[key + "_amax"] = amax
                sd[key + "_outer_scale_denominator"] = den
            else:
                data, scales, vals = mxfp8(w)
                sd[key] = data.view(torch.float8_e4m3fn)
                sd[key + "_scale"] = scales
            if leaf == "w13_weight":
                # The file interleaves gate and up rows; the module has gate rows, then up rows.
                e, rows, cols = vals.shape
                vals = vals.reshape(e, rows // 2, 2, cols).transpose(1, 2).reshape(e, rows, cols)
            with torch.no_grad():
                getattr(lm.layers[l].mlp.experts, attr).copy_(vals)
    save_file(sd, path, metadata={"format": "pt"})

g = torch.Generator().manual_seed(seed + 1)
ids = torch.randint(8, UNPADDED - 8, (1, plen), generator=g)
extra, kw = {}, {}
if tokdir:
    from transformers import AutoTokenizer

    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "processor_config.json"]:
        if os.path.exists(os.path.join(tokdir, f)):
            shutil.copy(os.path.join(tokdir, f), mdir)
    tk = AutoTokenizer.from_pretrained(mdir)
    if image:
        from PIL import Image
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(mdir)
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(image).convert("RGB")},
                                             {"type": "text", "text": "Describe this image."}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt")
        ids = inp["input_ids"]
        kw = dict(pixel_values=inp["pixel_values"].float(), image_grid_thw=inp["image_grid_thw"])
        extra = {"pixel_values": kw["pixel_values"].contiguous(), "image_grid_thw": kw["image_grid_thw"].to(torch.int64)}
        with torch.no_grad():
            feats = m.get_image_features(**kw, return_dict=True).pooler_output
        extra["image_features"] = torch.cat(list(feats)).float().contiguous()
        open(os.path.join(out, "prompt.txt"), "w").write(tk.decode(ids[0]))
    elif chat:
        conv = json.load(open(chat))
        more = {k: v for k, v in conv.items() if k not in ("messages", "tools")}
        text = tk.apply_chat_template(conv["messages"], tools=conv.get("tools"), add_generation_prompt=True, tokenize=False, **more)
        ids = torch.tensor([tk(text, add_special_tokens=False)["input_ids"]])
        open(os.path.join(out, "prompt.txt"), "w").write(text)
with torch.no_grad():
    full = m(ids, use_cache=False, **kw).logits[0].float()
    o = m(ids, use_cache=True, **kw)
    cache = o.cache
    last = o.logits[0, -1].float()
    toks, rows = [], []
    for i in range(gen):
        t = int(last.argmax())
        toks.append(t)
        o = m(torch.tensor([[t]]), cache=cache, use_cache=True)
        cache = o.cache
        last = o.logits[0, -1].float()
        rows.append(last)
save_file(
    {
        "ids": ids[0].to(torch.int64),
        "logits": full.contiguous(),
        "gen": torch.tensor(toks, dtype=torch.int64),
        "decode_logits": torch.stack(rows).contiguous(),
        **extra,
    },
    os.path.join(out, "ref.safetensors"),
)
print("prompt", ids.shape[1], "tokens; gen", toks)
if tokdir:
    open(os.path.join(out, "gen.txt"), "w").write(tk.decode(toks))
