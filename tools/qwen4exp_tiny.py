"""Tiny random Qwen4-Exp checkpoint + transformers reference outputs.

usage: tiny.py OUT_DIR [--gen N] [--seed S]
Writes OUT_DIR/model (save_pretrained), OUT_DIR/prompt.txt, OUT_DIR/ref.bin ([1+N, vocab] f32: the
prompt's last position, then after each generated token, via the KV cache), OUT_DIR/gen.txt and
OUT_DIR/prefill.bin ([len(prompt), vocab]: one uncached forward over the prompt).
"""
import sys, os, json
import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

out = sys.argv[1]
gen = 12
seed = 0
args = sys.argv[2:]
for i, a in enumerate(args):
    if a == '--gen':
        gen = int(args[i + 1])
    if a == '--seed':
        seed = int(args[i + 1])
torch.manual_seed(seed)
V = 512
EOS = 7
cfg = Qwen4ExpTextConfig(
    vocab_size=V,
    hidden_size=128,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    num_experts=8,
    num_experts_per_tok=int(os.environ.get("TOPK", "3")),
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    hc_count=4,
    hc_lowrank=32,
    indexer_budget=int(os.environ.get("BUDGET", "16")),
    indexer_compress_ratio=4,
    indexer_head_dim=32,
    indexer_n_heads=4,
    indexer_kv_heads=1,
    ple_layer_ids=[2],
    ple_embed_dim=128,
    ple_conv_kernel_size=4,
    ngram_size=3,
    ngram_vocab_size_base=1000,
    heads_per_ngram=8,
    split_ngram_parts=4,
    make_ngram_vocab_size_divisible_by=128,
    eos_token_id=EOS,
    bos_token_id=EOS,
    output_gate_type="sigmoid",
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                     "mrope_section": [3, 3, 2], "mrope_interleaved": True},
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    initializer_range=0.08,
)
cfg._attn_implementation = "eager"
m = Qwen4ExpForCausalLM(cfg).float().eval()
with torch.no_grad():
    for n, p in m.named_parameters():
        if n.endswith('norm.weight') or 'layernorm' in n or 'norm_' in n or n.endswith('hc_norm.weight'):
            p.copy_(torch.randn_like(p) * 0.2)
        elif 'conv1d' in n:
            p.copy_(torch.randn_like(p) * 0.3)
        elif 'A_log' in n:
            p.copy_(torch.rand_like(p) * 2 - 1)
        elif 'dt_bias' in n:
            p.copy_(torch.randn_like(p) * 0.5)
        elif 'ngram_embedding' in n:
            p.copy_(torch.randn_like(p) * 1.0)
    # bf16-representable weights: oxinfer's bf16 matmuls are then exact.
    for n, p in m.named_parameters():
        p.copy_(p.to(torch.bfloat16).float())
os.makedirs(out, exist_ok=True)
m.save_pretrained(os.path.join(out, 'model'))
g = torch.Generator().manual_seed(seed + 1)
prompt = torch.randint(8, V, (int(os.environ.get("PLEN", "40")),), generator=g)
prompt[5] = EOS
prompt[17] = EOS
ids = prompt.unsqueeze(0)
with torch.no_grad():
    full = m(ids, use_cache=False).logits[0].float()
    full.numpy().tofile(os.path.join(out, 'prefill.bin'))
    o = m(ids, use_cache=True)
    cache = o.past_key_values
    rows = [o.logits[0, -1].float()]
    toks = []
    for i in range(gen):
        t = int(rows[-1].argmax())
        toks.append(t)
        o = m(torch.tensor([[t]]), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        rows.append(o.logits[0, -1].float())
    torch.stack(rows).numpy().tofile(os.path.join(out, 'ref.bin'))
open(os.path.join(out, 'prompt.txt'), 'w').write(' '.join(str(int(x)) for x in prompt))
open(os.path.join(out, 'gen.txt'), 'w').write(' '.join(str(x) for x in toks))
print('prompt', prompt.tolist())
print('gen', toks)
print('cached vs uncached last logit diff', float((full[-1] - rows[0]).abs().max()))
