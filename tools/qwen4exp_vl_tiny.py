"""Tiny random Qwen4-Exp vision-language checkpoint (`Qwen4ExpForConditionalGeneration`: the
Qwen3.8-Flash-Next layout, scaled down) for checking image input against transformers.

usage: qwen4exp_vl_tiny.py SRC OUT_DIR [--seed S]
SRC is the real checkpoint folder (Qwen/Qwen3.8-Flash-Next: config, tokenizer, chat template and
processor files are copied from it, so prompts tokenize as with the real model). Writes OUT_DIR
(save_pretrained + those files). Compare with tools/ref_vl.py OUT_DIR --image X.png ... against
oxinfer's --dump-mm / --dump-logits.
"""
import sys, os, shutil
import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration

src, out = sys.argv[1], sys.argv[2]
seed = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 0
torch.manual_seed(seed)
text = dict(
    vocab_size=248320,
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
    num_experts_per_tok=3,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    hc_count=4,
    hc_lowrank=32,
    indexer_budget=16,
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
    eos_token_id=248044,
    bos_token_id=248044,
    output_gate_type="sigmoid",
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                     "mrope_section": [3, 3, 2], "mrope_interleaved": True},
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    initializer_range=0.08,
)
# The real tower's shape scaled down; the MLP width (100) is not a multiple of 64 like the
# real one's 4304, and the position grid (8x8) is resampled to every image's patch grid.
vision = dict(
    depth=2,
    hidden_size=64,
    hidden_act="gelu_pytorch_tanh",
    intermediate_size=100,
    num_heads=4,
    in_channels=3,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    out_hidden_size=128,
    num_position_embeddings=64,
    deepstack_visual_indexes=[],
)
cfg = Qwen4ExpConfig(text_config=text, vision_config=vision, image_token_id=248056, video_token_id=248057,
                     vision_start_token_id=248053, vision_end_token_id=248054, tie_word_embeddings=False)
cfg._attn_implementation = "eager"
cfg.text_config._attn_implementation = "eager"
cfg.vision_config._attn_implementation = "eager"
m = Qwen4ExpForConditionalGeneration(cfg).float().eval()
with torch.no_grad():
    for n, p in m.named_parameters():
        if 'visual' in n:
            if n.endswith('norm1.weight') or n.endswith('norm2.weight') or n.endswith('merger.norm.weight'):
                p.copy_(1 + torch.randn_like(p) * 0.2)
            elif n.endswith('.bias'):
                p.copy_(torch.randn_like(p) * 0.1)
            elif 'pos_embed' in n:
                p.copy_(torch.randn_like(p) * 0.5)
            else:
                p.copy_(torch.randn_like(p) / max(p.shape[-1] if p.dim() == 2 else p[0].numel(), 1) ** 0.5)
        elif n.endswith('norm.weight') or 'layernorm' in n or 'norm_' in n or n.endswith('hc_norm.weight'):
            p.copy_(torch.randn_like(p) * 0.2)
        elif 'conv1d' in n:
            p.copy_(torch.randn_like(p) * 0.3)
        elif 'A_log' in n:
            p.copy_(torch.rand_like(p) * 2 - 1)
        elif 'dt_bias' in n:
            p.copy_(torch.randn_like(p) * 0.5)
        elif 'ngram_embedding' in n:
            p.copy_(torch.randn_like(p) * 1.0)
        elif 'embed_tokens' in n:
            p.copy_(torch.randn_like(p) * 0.5)
    # bf16-representable weights: oxinfer's bf16 matmuls are then exact.
    for n, p in m.named_parameters():
        p.copy_(p.to(torch.bfloat16).float())
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
for f in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'preprocessor_config.json',
          'video_preprocessor_config.json', 'generation_config.json', 'vocab.json', 'merges.txt']:
    if os.path.exists(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
print('saved', out, sum(p.numel() for p in m.parameters()), 'parameters')
