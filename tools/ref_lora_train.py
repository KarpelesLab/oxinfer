#!/usr/bin/env python3
"""PyTorch + peft reference for oxinfer's LoRA trainer (src/lora_train.rs).

Builds a tiny random Qwen3 (or, with --llama, Llama 3) checkpoint with transformers, puts a peft
LoRA on every projection, and records what the trainer must reproduce on three episodes (one
rewarded, two penalized): the losses, the gradients of every LoRA matrix from autograd, and the
adapter after a few clipped AdamW steps.

    python tools/ref_lora_train.py DIR [--llama]
    OXINFER_LORA_TRAIN_REF=DIR cargo test --profile ci --lib lora_train::tests::reference -- --ignored --nocapture

Written into DIR: model/ (the checkpoint, weights rounded to bf16 values so both sides hold the
same numbers), episodes.jsonl, outcomes.jsonl, init/ (the starting adapter, B random so that
every matrix has a gradient), grads/ (an adapter-shaped folder whose lora_A / lora_B hold the
gradients), stepped/ (the adapter after the steps) and ref.json.
"""
import argparse
import json
import math
import os
import shutil

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--llama", action="store_true")
ap.add_argument("--steps", type=int, default=3)
args = ap.parse_args()

torch.manual_seed(0)
H, HEADS, KV, HD, INTER, LAYERS, VOCAB = 64, 4, 2, 16, 128, 3, 300
RANK, ALPHA, NEG_CLIP, LR, WD, CLIP = 4, 8.0, 0.012, 1e-3, 0.01, 0.05
common = dict(
    hidden_size=H, intermediate_size=INTER, num_hidden_layers=LAYERS, num_attention_heads=HEADS,
    num_key_value_heads=KV, vocab_size=VOCAB, rms_norm_eps=1e-6, max_position_embeddings=512,
    hidden_act="silu", attention_bias=False, attention_dropout=0.0,
)
if args.llama:
    from transformers import LlamaConfig, LlamaForCausalLM
    model = LlamaForCausalLM(LlamaConfig(rope_theta=10000.0, tie_word_embeddings=False, **common))
else:
    from transformers import Qwen3Config, Qwen3ForCausalLM
    model = Qwen3ForCausalLM(Qwen3Config(head_dim=HD, rope_theta=10000.0, tie_word_embeddings=False, **common))

# Weights large enough for every part of the network to matter, at bf16 precision.
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("norm.weight"):
            p.copy_(1.0 + 0.4 * (torch.rand_like(p) - 0.5))
        elif "embed_tokens" in name:
            p.copy_(0.3 * torch.randn_like(p))
        else:
            p.copy_((torch.rand_like(p) - 0.5) * 2.0 / math.sqrt(p.shape[1]))
        p.copy_(p.to(torch.bfloat16).to(torch.float32))
model.eval()
os.makedirs(args.dir, exist_ok=True)
model.save_pretrained(os.path.join(args.dir, "model"))

targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
peft_model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=targets, lora_dropout=0.0))
with torch.no_grad():
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            p.copy_(0.1 * (torch.rand_like(p) - 0.5))
peft_model.save_pretrained(os.path.join(args.dir, "init"))

# A rewarded episode with a random completion, and two penalized ones whose completion is what
# the model writes itself (greedily): tokens likely enough for the push-down to matter.
g = torch.Generator().manual_seed(1)
episodes = []
for i, (p_len, c_len, reward) in enumerate([(9, 14, 1.0), (5, 11, -0.7), (12, 6, -1.0)]):
    prompt = torch.randint(0, VOCAB, (p_len,), generator=g).tolist()
    completion = torch.randint(0, VOCAB, (c_len,), generator=g).tolist()
    if reward < 0:
        ids = list(prompt)
        with torch.no_grad():
            for _ in range(c_len):
                ids.append(int(peft_model(input_ids=torch.tensor([ids])).logits[0, -1].argmax()))
        completion = ids[p_len:]
    episodes.append({"id": f"ep{i}", "prompt_tokens": prompt, "completion_tokens": completion, "reward": reward})
with open(os.path.join(args.dir, "episodes.jsonl"), "w") as f:
    for e in episodes:
        f.write(json.dumps({k: e[k] for k in ("id", "prompt_tokens", "completion_tokens")}) + "\n")
with open(os.path.join(args.dir, "outcomes.jsonl"), "w") as f:
    for e in episodes:
        f.write(json.dumps({"id": e["id"], "reward": e["reward"], "kind": "tests"}) + "\n")


def episode_loss(e):
    """The mean loss of the completion tokens: -ln p when rewarded; when penalized -ln(1 - p),
    continued by its tangent in ln p past odds p / (1 - p) = NEG_CLIP."""
    tokens = e["prompt_tokens"] + e["completion_tokens"]
    ids = torch.tensor([tokens[:-1]])
    first = len(e["prompt_tokens"]) - 1
    logits = peft_model(input_ids=ids).logits[0, first:].double()
    targets = torch.tensor(tokens[first + 1:])
    logp = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None])[:, 0]
    if e["reward"] > 0:
        return (-logp).mean(), 0
    p = logp.exp()
    p0 = NEG_CLIP / (1.0 + NEG_CLIP)
    under = p < NEG_CLIP * (1.0 - p)
    capped = -math.log1p(-p0) + NEG_CLIP * (logp - math.log(p0))
    return torch.where(under, -torch.log1p(-p.clamp(max=0.999999)), capped).mean(), int((~under).sum())


def batch_loss():
    den = sum(abs(e["reward"]) for e in episodes)
    parts = [episode_loss(e) for e in episodes]
    total = sum(abs(e["reward"]) / den * l for e, (l, _) in zip(episodes, parts))
    return total, [float(l.detach()) for l, _ in parts], sum(c for _, c in parts)


params = [p for p in peft_model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
loss, episode_losses, capped = batch_loss()
opt.zero_grad()
loss.backward()
grads = {}
for name, p in peft_model.named_parameters():
    if p.requires_grad:
        grads[name.replace(".default", "")] = p.grad.detach().clone().contiguous()
os.makedirs(os.path.join(args.dir, "grads"), exist_ok=True)
save_file(grads, os.path.join(args.dir, "grads", "adapter_model.safetensors"))
shutil.copy(os.path.join(args.dir, "init", "adapter_config.json"), os.path.join(args.dir, "grads"))
norm0 = float(torch.nn.utils.clip_grad_norm_(params, CLIP))
opt.step()
for _ in range(args.steps - 1):
    opt.zero_grad()
    batch_loss()[0].backward()
    torch.nn.utils.clip_grad_norm_(params, CLIP)
    opt.step()
peft_model.save_pretrained(os.path.join(args.dir, "stepped"))
with torch.no_grad():
    loss_after = float(batch_loss()[0])
ref = {
    "loss": float(loss), "episode_losses": episode_losses, "loss_after": loss_after, "neg_clip": NEG_CLIP,
    "lr": LR, "weight_decay": WD, "clip": CLIP, "steps": args.steps, "grad_norm": norm0, "capped_tokens": capped,
}
with open(os.path.join(args.dir, "ref.json"), "w") as f:
    json.dump(ref, f, indent=1)
print(json.dumps(ref))
