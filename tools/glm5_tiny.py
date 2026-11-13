"""Tiny random GLM-5.3-Flash (`glm5_next`) or GLM-5.3 (`glm_moe_dsa`) checkpoint + transformers
reference outputs, every feature on: hyper-connections, KDA linear attention, NoPE MLA with the
k-pool DSA indexer (a small `index_topk`, so most queries are sparse), a "shared" indexer layer,
dense and MoE layers with clamped SwiGLU (Flash); MLA with RoPE and the per-token DSA indexer
(GLM-5.3).

usage: glm5_tiny.py OUT_DIR [--arch flash|dsa] [--len N] [--gen N] [--seed S]
                    [--tokenizer RELEASE_DIR [--chat CONVERSATION.json]]
(environment: TOPK, KPOOL, LIMIT override `index_topk`, `index_kpool`, `swiglu_limit`)

With --tokenizer the model takes the release's vocabulary and its tokenizer files (a folder with
tokenizer.json, tokenizer_config.json and chat_template.jinja), and with --chat the prompt is
that conversation rendered by the chat template (written to OUT_DIR/prompt.txt).

Writes OUT_DIR/model (save_pretrained, which writes the tensor names of the zai-org release:
`hc_attn_fn`, `self_attn.f_a_proj`, `q_conv1d`, one module per expert) and
OUT_DIR/ref.safetensors: `ids` (I64), `logits` [len, vocab] (one uncached forward),
`gen` (I64, greedy tokens) and `decode_logits` [gen, vocab] (after feeding each of them through
the cache). Needs transformers >= 5.23 (float32, CPU, eager attention).
"""
import os, sys
import torch
from safetensors.torch import save_file

out = sys.argv[1]
arch, plen, gen, seed, tokdir, chat = "flash", 150, 16, 0, None, None
a = sys.argv[2:]
for i, x in enumerate(a):
    if x == "--arch":
        arch = a[i + 1]
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
torch.manual_seed(seed)
# With the release's tokenizer: its vocabulary and special tokens.
V, PAD, EOS = (154880, 154820, [154820, 154827, 154829]) if tokdir else (512, 0, [1, 2])

if arch == "flash":
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration

    # Layers 3, 7, 8, 9 are MLA (8 shares 7's selection); 0..2 have dense MLPs.
    kda = [0, 1, 2, 4, 5, 6, 10]
    n = 11
    text = dict(
        vocab_size=V,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        num_hidden_layers=n,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=3,
        routed_scaling_factor=2.5,
        kv_lora_rank=24,
        q_lora_rank=32,
        qk_rope_head_dim=0,
        qk_nope_head_dim=16,
        v_head_dim=16,
        index_topk=int(os.environ.get("TOPK", "8")),
        index_head_dim=16,
        index_n_heads=16,
        index_kpool=int(os.environ.get("KPOOL", "4")),
        # The release's spelling of the MLA layers (transformers' own is "indexed_attention").
        layer_types=["linear_attention" if i in kda else "deepseek_sparse_attention" for i in range(n)],
        mlp_layer_types=["dense"] * 3 + ["sparse"] * (n - 3),
        indexer_types=["shared" if i == 8 else "full" for i in range(n)],
        linear_attn_config=dict(num_heads=4, head_dim=16, short_conv_kernel_size=4, gate_lower_bound=-5.0),
        swiglu_limit=float(os.environ.get("LIMIT", "0.6")),
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
        pad_token_id=PAD,
        eos_token_id=EOS,
        initializer_range=0.08,
        max_position_embeddings=4096,
    )
    vision = dict(depth=2, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=64,
                  projection_intermediate_size=96, patch_size=14, image_size=56)
    cfg = Glm5NextConfig(text_config=text, vision_config=vision, image_token_id=V - 2, video_token_id=V - 1,
                         image_start_token_id=V - 4, image_end_token_id=V - 3, video_start_token_id=V - 6,
                         video_end_token_id=V - 5)
    cfg._attn_implementation = "eager"
    m = Glm5NextForConditionalGeneration(cfg).float().eval()
else:
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

    n = 6
    cfg = GlmMoeDsaConfig(
        vocab_size=V,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        num_hidden_layers=n,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=3,
        routed_scaling_factor=2.5,
        kv_lora_rank=24,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=16,
        index_topk=int(os.environ.get("TOPK", "24")),
        index_head_dim=16,
        index_n_heads=16,
        mlp_layer_types=["dense"] * 2 + ["sparse"] * (n - 2),
        indexer_types=["full", "full", "shared", "full", "shared", "full"],
        rms_norm_eps=1e-5,
        pad_token_id=PAD,
        eos_token_id=EOS,
        initializer_range=0.08,
        max_position_embeddings=4096,
    )
    cfg._attn_implementation = "eager"
    m = GlmMoeDsaForCausalLM(cfg).float().eval()

with torch.no_grad():
    for name, p in m.named_parameters():
        if name.endswith("norm.weight") or "layernorm" in name or name.endswith("k_norm.weight"):
            p.copy_(1.0 + torch.randn_like(p) * 0.2)
        elif name.endswith("k_norm.bias"):
            p.copy_(torch.randn_like(p) * 0.2)
        elif "conv1d" in name:
            p.copy_(torch.randn_like(p) * 0.3)
        elif name.endswith("A_log"):
            p.copy_(torch.rand_like(p) * 2 - 1)
        elif name.endswith("dt_bias"):
            p.copy_(torch.randn_like(p) * 0.5)
        elif name.endswith("_hc.base"):
            p.copy_(torch.randn_like(p) * 0.5)
        elif name.endswith("_hc.scale"):
            p.copy_(1.0 + torch.randn_like(p) * 0.3)
        elif name.endswith("_hc.fn"):
            p.copy_(torch.randn_like(p) * 0.2)
        elif name.endswith("index_kpool_compress_ape"):
            p.copy_(torch.randn_like(p) * 0.5)
        elif name.endswith("index_kpool_compress_gate"):
            p.copy_(torch.randn_like(p) * 0.3)
    for name, b in m.named_buffers():
        if name.endswith("e_score_correction_bias"):
            b.copy_(torch.randn_like(b) * 0.1)
    # bf16-representable weights: oxinfer's bf16 matmuls are then exact. (Only what the
    # checkpoint stores: the rotary frequencies are a buffer computed from the config.)
    for name, p in m.state_dict().items():
        if p.is_floating_point():
            p.copy_(p.to(torch.bfloat16).float())

os.makedirs(out, exist_ok=True)
mdir = os.path.join(out, "model")
m.save_pretrained(mdir)

g = torch.Generator().manual_seed(seed + 1)
ids = torch.randint(8, V - 8, (1, plen), generator=g)
if tokdir:
    import json, shutil
    from transformers import AutoTokenizer

    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        shutil.copy(os.path.join(tokdir, f), mdir)
    json.dump({"eos_token_id": EOS, "pad_token_id": PAD}, open(os.path.join(mdir, "generation_config.json"), "w"))
    if chat:
        # The prompt: a conversation ({"messages": [...], "tools": [...]}) through the chat template.
        tk = AutoTokenizer.from_pretrained(mdir)
        conv = json.load(open(chat))
        text = tk.apply_chat_template(conv["messages"], tools=conv.get("tools"), add_generation_prompt=True, tokenize=False)
        ids = torch.tensor([tk(text, add_special_tokens=False)["input_ids"]])
        open(os.path.join(out, "prompt.txt"), "w").write(text)
with torch.no_grad():
    o = m(ids, use_cache=False, output_hidden_states=True)
    full = o.logits[0].float()
    # The residual stream after each layer (the mean of its copies with hyper-connections).
    hidden = torch.stack([x[0].mean(dim=-2) if x.dim() == 4 else x[0] for x in o.hidden_states[-n:]]).float()
    o = m(ids, use_cache=True)
    cache = o.past_key_values
    last = o.logits[0, -1].float()
    toks, rows = [], []
    for i in range(gen):
        t = int(last.argmax())
        toks.append(t)
        o = m(torch.tensor([[t]]), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        last = o.logits[0, -1].float()
        rows.append(last)
save_file(
    {
        "ids": ids[0].to(torch.int64),
        "logits": full.contiguous(),
        "hidden": hidden.contiguous(),
        "gen": torch.tensor(toks, dtype=torch.int64),
        "decode_logits": torch.stack(rows).contiguous(),
    },
    os.path.join(out, "ref.safetensors"),
)
print("gen", toks)
if tokdir:
    open(os.path.join(out, "gen.txt"), "w").write(AutoTokenizer.from_pretrained(mdir).decode(toks))
