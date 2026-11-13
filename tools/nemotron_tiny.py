"""Tiny random Nemotron-H (`nemotron_h`) checkpoint + transformers reference outputs, every block
type on: Mamba-2 (several scan chunks, some heads below the time step floor), attention without
positions (grouped queries), MoE with relu^2 experts and a shared expert, a dense MLP.

usage: nemotron_tiny.py OUT_DIR [--layout release|stacked|nemo] [--len N] [--gen N] [--seed S]
                        [--blocks PATTERN] [--tokenizer RELEASE_DIR [--chat CONVERSATION.json]]

--blocks: the mixers as a `hybrid_override_pattern` (M Mamba-2, * attention, E MoE, - MLP;
default MEM*EM-*EME). llama.cpp's `nemotron_h_moe` has no dense MLP block: MEM*EME*EME.

--layout: the tensor names of OUT_DIR/model/model.safetensors: the release's, which
save_pretrained writes (`backbone.layers.*`, one module per expert; the default), the module's
own (`model.layers.*`, experts stacked `[n, out, in]`, the names oxinfer gives a GGUF file's
tensors) or NeMo's speech LM export (`llm.model.layers.*`, `_fp32_params`, experts stacked with
the matrix axes swapped, as in nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B).

With --tokenizer the model takes the release's vocabulary and its tokenizer files (a folder with
tokenizer.json, tokenizer_config.json and chat_template.jinja), and with --chat the prompt is
that conversation rendered by the chat template (written to OUT_DIR/prompt.txt).

Writes OUT_DIR/model and OUT_DIR/ref.safetensors: `ids` (I64), `logits` [len, vocab] (one
uncached forward), `gen` (I64, greedy tokens), `decode_logits` [gen, vocab] (after feeding each
of them through the cache) and `decode_logits_full` [gen, vocab] (the same positions of one
uncached forward over the prompt and the generated tokens). The two differ: transformers'
multi-token scan floors the Mamba time step at `time_step_min`, its cached one-token step does
not (NVIDIA's own code floors in neither; see docs/nemotron.md).
Needs transformers >= 5.24 (float32, CPU, eager attention).
"""
import json, os, sys
import torch
from safetensors.torch import save_file
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

out = sys.argv[1]
layout, plen, gen, seed, tokdir, chat, pattern = "release", 150, 16, 0, None, None, "MEM*EM-*EME"
a = sys.argv[2:]
for i, x in enumerate(a):
    if x == "--layout":
        layout = a[i + 1]
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
    if x == "--blocks":
        pattern = a[i + 1]
torch.manual_seed(seed)
# With the release's tokenizer: its vocabulary and special tokens.
V, PAD, EOS = (131072, 0, [2, 11]) if tokdir else (512, 0, [1, 2])

KINDS = {"M": "linear_attention", "*": "full_attention", "E": "moe", "-": "mlp"}
cfg = NemotronHConfig(
    vocab_size=V,
    hidden_size=64,
    layers_block_type=[KINDS[c] for c in pattern],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=96,
    mamba_num_heads=8,
    mamba_head_dim=16,
    n_groups=4,
    ssm_state_size=16,
    conv_kernel=4,
    chunk_size=16,
    n_routed_experts=8,
    num_experts_per_tok=3,
    moe_intermediate_size=32,
    moe_shared_expert_intermediate_size=48,
    routed_scaling_factor=2.5,
    pad_token_id=PAD,
    eos_token_id=EOS,
    initializer_range=0.08,
    max_position_embeddings=4096,
)
cfg._attn_implementation = "eager"
m = NemotronHForCausalLM(cfg).float().eval()


def randomize(model):
    """Weights away from their initial values, bf16-representable (oxinfer's bf16 matmuls are then exact)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("norm_f.weight") or name.endswith("norm1.weight"):
                p.copy_(1.0 + torch.randn_like(p) * 0.2)
            elif "conv1d" in name:
                p.copy_(torch.randn_like(p) * 0.3)
            elif name.endswith("A_log"):
                p.copy_(torch.rand_like(p) * 2 - 1)
            elif name.endswith("dt_bias"):
                # softplus(-7) is below transformers' floor of 0.001
                p.copy_(torch.randn_like(p) * 2 - 4)
            elif name.endswith("mixer.D"):
                p.copy_(1.0 + torch.randn_like(p) * 0.3)
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(torch.randn_like(b) * 0.1)
        for name, p in model.state_dict().items():
            if p.is_floating_point():
                p.copy_(p.to(torch.bfloat16).float())


def nemo_layout(sd):
    """The language model's tensors of a module's state dict under NeMo's names."""
    new = {}
    for k, v in sd.items():
        k = k.replace("model.embeddings.", "llm.model.embed_tokens.")
        k = k.replace("model.norm_f.", "llm.model.norm.")
        k = k.replace("model.layers.", "llm.model.layers.")
        k = "llm." + k if k.startswith("lm_head.") else k
        for n in ("A_log", "D", "dt_bias"):
            if k.endswith(f".mixer.{n}"):
                k = k[: -len(n)] + "_fp32_params." + n
        if k.endswith("experts.up_proj"):
            k, v = k[: -len("up_proj")] + "gate_and_up_projs", v.transpose(1, 2)
        elif k.endswith("experts.down_proj"):
            k, v = k[: -len("down_proj")] + "down_projs", v.transpose(1, 2)
        new[k] = v.contiguous()
    return new


if __name__ == "__main__":
    randomize(m)
    os.makedirs(out, exist_ok=True)
    mdir = os.path.join(out, "model")
    m.save_pretrained(mdir)
    if layout != "release":
        sd = {k: v.contiguous() for k, v in m.state_dict().items()}
        save_file(nemo_layout(sd) if layout == "nemo" else sd, os.path.join(mdir, "model.safetensors"))

    g = torch.Generator().manual_seed(seed + 1)
    ids = torch.randint(8, V - 8, (1, plen), generator=g)
    if tokdir:
        import shutil
        from transformers import AutoTokenizer

        for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
            shutil.copy(os.path.join(tokdir, f), mdir)
        json.dump({"eos_token_id": EOS, "pad_token_id": PAD}, open(os.path.join(mdir, "generation_config.json"), "w"))
        if chat:
            # The prompt: a conversation ({"messages": [...], "tools": [...]}) through the chat template.
            tk = AutoTokenizer.from_pretrained(mdir)
            conv = json.load(open(chat))
            text = tk.apply_chat_template(
                conv["messages"], tools=conv.get("tools"), add_generation_prompt=True, tokenize=False,
                **conv.get("kwargs", {}),
            )
            ids = torch.tensor([tk(text, add_special_tokens=False)["input_ids"]])
            open(os.path.join(out, "prompt.txt"), "w").write(text)

    with torch.no_grad():
        logits = m(ids, logits_to_keep=0).logits[0]
        # Greedy decoding through the cache, one token at a time.
        o = m(ids, use_cache=True)
        cache, nxt = o.cache, o.logits[0, -1].argmax()
        toks, dec = [], []
        for _ in range(gen):
            toks.append(int(nxt))
            o = m(nxt.view(1, 1), cache=cache, use_cache=True)
            cache = o.cache
            dec.append(o.logits[0, -1])
            nxt = o.logits[0, -1].argmax()
        full = m(torch.cat([ids, torch.tensor([toks])], dim=1), logits_to_keep=0).logits[0, ids.shape[1] :]
    dec = torch.stack(dec)
    save_file(
        {
            "ids": ids[0].contiguous(),
            "logits": logits.float().contiguous(),
            "gen": torch.tensor(toks),
            "decode_logits": dec.float().contiguous(),
            "decode_logits_full": full.float().contiguous(),
        },
        os.path.join(out, "ref.safetensors"),
    )
    print(f"{layout}: {ids.shape[1]} prompt tokens, greedy {toks}")
    print(f"cached steps against the uncached forward (transformers' time step floor): max logit diff {(dec - full).abs().max():.3e}")
