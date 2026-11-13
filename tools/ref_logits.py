#!/usr/bin/env python3
"""Reference check against HuggingFace transformers (float32, CPU).

usage: ref_logits.py MODEL_DIR PROMPT_TEXT_FILE OURS_LOGITS.bin [--generate N] [--bf16] [--trust-remote-code]

PROMPT_TEXT_FILE holds the exact templated text (including <bos>) that oxinfer tokenized.
Prints token ids, compares our logits with the reference and optionally greedy-generates.
--trust-remote-code runs the model repo's own modeling code (LiquidAI/LFM3-1.7B: the class of
that name in transformers names the attention gate differently and leaves it uninitialized).
"""
import sys, struct, time
import numpy as np
import torch

def main():
    model_dir, text_file, ours_file = sys.argv[1:4]
    gen = 0
    if "--generate" in sys.argv:
        gen = int(sys.argv[sys.argv.index("--generate") + 1])
    dtype = torch.bfloat16 if "--bf16" in sys.argv else torch.float32
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    remote = "--trust-remote-code" in sys.argv
    tok = AutoTokenizer.from_pretrained(model_dir)
    text = open(text_file).read()
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    print("ref ids:", ids[0].tolist())
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=remote)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, trust_remote_code=remote)
    model.eval()
    print(f"loaded {type(model).__name__} in {time.time()-t0:.1f}s")
    with torch.no_grad():
        out = model(input_ids=ids)
    ref = out.logits[0, -1].float().numpy()
    ours = np.fromfile(ours_file, dtype="<f4")
    print("vocab", ref.shape, ours.shape)
    diff = np.abs(ref - ours)
    print(f"max abs diff {diff.max():.5f}  mean abs diff {diff.mean():.6f}")
    rt = np.argsort(-ref)[:10]; ot = np.argsort(-ours)[:10]
    print("ref  top10:", [(int(i), round(float(ref[i]), 3)) for i in rt])
    print("ours top10:", [(int(i), round(float(ours[i]), 3)) for i in ot])
    print("argmax match:", int(rt[0]) == int(ot[0]))
    if gen:
        with torch.no_grad():
            # Greedy on the raw logits, as oxinfer's default: a checkpoint's generation_config may
            # set a repetition penalty (LFM3: 1.1, which covers the prompt's tokens too).
            g = model.generate(ids, max_new_tokens=gen, do_sample=False, repetition_penalty=1.0)
        new = g[0, ids.shape[1]:].tolist()
        print("ref greedy ids:", new)
        print("ref greedy text:", repr(tok.decode(new)))

if __name__ == "__main__":
    main()
