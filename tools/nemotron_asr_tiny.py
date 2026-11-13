"""Tiny random Nemotron 3.5 Transcribe (`nemo_speechlm`) checkpoint + transformers reference
outputs on a synthetic recording: the real feature extractor (128 mel bins), a speech
recognition encoder and a speaker encoder of two layers each with small chunks (so that a few
seconds of audio take several chunks with context, fill the FIFO and compress the speaker
cache), the projection and a tiny Nemotron-H language model.

usage: nemotron_asr_tiny.py OUT_DIR [--secs S] [--gen N] [--seed S]

Writes OUT_DIR/model (config.json and model.safetensors in the names of NVIDIA's NeMo export,
which save_pretrained writes) and OUT_DIR/ref.safetensors (see tools/ref_nemotron_asr.py; the
logits of every prompt position). Needs transformers >= 5.24 (float32, CPU, eager attention).
"""
import os, sys
import torch
from safetensors.torch import save_file
from transformers import (
    Nemotron3_5TranscribeConfig,
    Nemotron3_5TranscribeForConditionalGeneration,
    NemotronAsrStreamingFeatureExtractor,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_nemotron_asr import reference  # noqa: E402

out = sys.argv[1]
secs, gen, seed = 9.3, 12, 0
a = sys.argv[2:]
for i, x in enumerate(a):
    if x == "--secs":
        secs = float(a[i + 1])
    if x == "--gen":
        gen = int(a[i + 1])
    if x == "--seed":
        seed = int(a[i + 1])
torch.manual_seed(seed)
V = 512
M, A, E = "linear_attention", "full_attention", "moe"
text = dict(
    vocab_size=V, hidden_size=64, layers_block_type=[M, E, M, A, E, M, E], num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, mamba_num_heads=8, mamba_head_dim=16, n_groups=4, ssm_state_size=16,
    conv_kernel=4, chunk_size=16, n_routed_experts=8, num_experts_per_tok=3, moe_intermediate_size=32,
    moe_shared_expert_intermediate_size=48, routed_scaling_factor=2.5, pad_token_id=0, eos_token_id=[1, 2],
    initializer_range=0.08, max_position_embeddings=4096, intermediate_size=96,
)
audio = dict(
    asr_config=dict(hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4),
    diarization_config=dict(
        audio_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4),
        head_config=dict(hidden_size=16, num_speakers=4, audio_hidden_size=32),
        streaming_config=dict(speaker_cache_length=24, num_speakers=4),
        chunk_length=20, chunk_left_context=3, chunk_right_context=3, fifo_length=4, speaker_cache_update_period=8,
    ),
)
cfg = Nemotron3_5TranscribeConfig(
    audio_config=audio, text_config=text, audio_token_id=V,
    word_timestamp_config=dict(hidden_size=48, intermediate_size=24, num_hidden_layers=1, num_attention_heads=4, vocab_size=40, blank_token_id=39),
)
cfg._attn_implementation = "eager"
m = Nemotron3_5TranscribeForConditionalGeneration(cfg).float().eval()
with torch.no_grad():
    for name, p in m.named_parameters():
        if "norm" in name and name.endswith("weight"):
            p.copy_(1.0 + torch.randn_like(p) * 0.2)
        elif "norm" in name and name.endswith("bias"):
            p.copy_(torch.randn_like(p) * 0.2)
        elif name.endswith(".bias"):
            p.copy_(torch.randn_like(p) * 0.1)
        elif "conv1d" in name:
            p.copy_(torch.randn_like(p) * 0.3)
        elif name.endswith("A_log"):
            p.copy_(torch.rand_like(p) * 2 - 1)
        elif name.endswith("dt_bias"):
            p.copy_(torch.randn_like(p) * 2 - 4)
        elif name.endswith("mixer.D"):
            p.copy_(1.0 + torch.randn_like(p) * 0.3)
        elif name.endswith("silence_embeds"):
            p.copy_(torch.randn_like(p) * 0.5)
        elif "audio_tower" in name or "multi_modal_projector" in name:
            # Wide enough that the speaker probabilities spread over (0, 1).
            p.copy_(torch.randn_like(p) * (2.0 / p.shape[-1] ** 0.5))
    for name, b in m.named_buffers():
        if name.endswith("e_score_correction_bias"):
            b.copy_(torch.randn_like(b) * 0.1)
    for name, p in m.state_dict().items():
        if p.is_floating_point() and "speaker_embedding" not in name:
            p.copy_(p.to(torch.bfloat16).float())

os.makedirs(out, exist_ok=True)
mdir = os.path.join(out, "model")
m.save_pretrained(mdir)

# A synthetic recording: tones that come and go over noise.
fe = NemotronAsrStreamingFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, n_fft=512, win_length=400, mel_filters_dtype="bfloat16")
g = torch.Generator().manual_seed(seed + 1)
n = int(secs * 16000)
t = torch.arange(n) / 16000
pcm = 0.02 * torch.randn(n, generator=g)
for f0, period in [(220.0, 1.3), (910.0, 2.1), (3100.0, 0.7)]:
    pcm += 0.2 * torch.sin(2 * torch.pi * f0 * t) * (torch.sin(2 * torch.pi * t / period) > 0)


def ids_of(k):
    head = torch.randint(8, V - 8, (11,), generator=g)
    tail = torch.randint(8, V - 8, (6,), generator=g)
    return torch.cat([head, torch.full((k,), V), tail])[None]


ref, toks, _ = reference(m, fe, pcm, ids_of, gen, 0)
save_file(ref, os.path.join(out, "ref.safetensors"))
probs = ref["speaker_logits"].sigmoid()
print(f"{secs}s: {ref['features'].shape[0]} frames ({int(ref['valid'])} valid), {ref['audio_embeds'].shape[0]} audio tokens, "
      f"{ref['ids'].shape[0]} prompt tokens, greedy {toks}")
print(f"speaker probabilities: mean {probs.mean():.3f}, above 0.5: {(probs > 0.5).float().mean():.3f}")
