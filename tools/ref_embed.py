#!/usr/bin/env python3
"""Reference check for embeddings against sentence-transformers (float32, CPU).

usage: ref_embed.py MODEL_DIR TEXTS_FILE OURS.jsonl [--prompt-name NAME] [--bf16-weights] [--trust-remote-code]
                    [--sparse | --colbert] [--dense MODULE_DIR] [--batch N]

TEXTS_FILE has one text per line; OURS.jsonl is `oxinfer --embed` output for the same lines (a
JSON array per line; with --sparse a {token: weight} object, with --colbert an array of arrays).
Prints the cosine between our vector and the reference per text and compares the similarity
matrices. --sparse / --colbert check BGE-M3's heads, computed here as FlagEmbedding does
(relu(sparse_linear(h)) max per token without special tokens; normalize(colbert_linear(h[1:]))).
--trust-remote-code runs the model repo's own modeling code (NomicBERT, gte-Qwen2, stella).
The Qwen2 ones need transformers 4.4x (4.44.2 with sentence-transformers 3.0.1 was used), and
gte-Qwen2's file imports flash_attn at parse time: an empty `flash_attn/__init__.py` in
site-packages is enough on a CPU.
--dense swaps the model's Dense module for another of its folders (stella: 2_Dense_256, ...; the
vectors to compare are then `oxinfer --dim 256`). --batch encodes N texts at a time, so the
shorter ones are padded (the default, 1, pads nothing).
"""
import json, sys
import numpy as np
import torch


def main():
    model_dir, texts_file, ours_file = sys.argv[1:4]
    prompt_name = sys.argv[sys.argv.index("--prompt-name") + 1] if "--prompt-name" in sys.argv else None
    texts = [l.rstrip("\n") for l in open(texts_file) if l.strip()]
    ours = [json.loads(l) for l in open(ours_file) if l.strip()]
    assert len(ours) == len(texts), f"{len(ours)} outputs for {len(texts)} texts"
    if "--sparse" in sys.argv or "--colbert" in sys.argv:
        return heads(model_dir, texts, ours, "--sparse" in sys.argv)
    import transformers
    from sentence_transformers import SentenceTransformer

    # The remote code of the Qwen2 embedders needs transformers 4.x, which calls it torch_dtype.
    dtype = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    m = SentenceTransformer(
        model_dir, device="cpu", model_kwargs={dtype: torch.float32}, trust_remote_code="--trust-remote-code" in sys.argv
    )
    if "--dense" in sys.argv:
        from sentence_transformers.models import Dense

        i = [k for k, mod in enumerate(m) if isinstance(mod, Dense)][-1]
        m[i] = Dense.load(f"{model_dir}/{sys.argv[sys.argv.index('--dense') + 1]}")
    for mod in m.modules():
        # transformers 5 dropped this helper; NomicBERT's remote code still calls it.
        if mod.__class__.__name__ == "NomicBertModel" and not hasattr(mod, "get_extended_attention_mask"):
            ext = lambda mask, shape: (1.0 - mask[:, None, None, :].float()) * torch.finfo(torch.float32).min
            mod.get_extended_attention_mask = ext
    if "--bf16-weights" in sys.argv:
        # Isolate implementation error: the same bf16-rounded weights oxinfer computes with.
        with torch.no_grad():
            for p in m.parameters():
                if p.dim() >= 2:  # matrices and embedding tables; norms and biases stay f32
                    p.copy_(p.to(torch.bfloat16).to(torch.float32))
    batch = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 1
    ref = m.encode(texts, prompt_name=prompt_name, convert_to_numpy=True, batch_size=batch)
    ours = np.array(ours, dtype=np.float64)
    ref = ref.astype(np.float64)
    cos = [float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) for a, b in zip(ours, ref)]
    print(f"dim {ref.shape[1]} | cosine(ours, ref) per text: min {min(cos):.6f} mean {np.mean(cos):.6f}")
    print("  " + " ".join(f"{c:.6f}" for c in cos))
    print(f"max abs diff {np.abs(ours - ref).max():.2e}")
    n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    so, sr = n(ours) @ n(ours).T, n(ref) @ n(ref).T
    print(f"similarity matrix max abs diff {np.abs(so - sr).max():.2e}")
    print("ref similarities:\n" + "\n".join(" ".join(f"{v:7.4f}" for v in r) for r in sr))


def heads(model_dir, texts, ours, sparse):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32).eval()
    if "--bf16-weights" in sys.argv:
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() >= 2:
                    p.copy_(p.to(torch.bfloat16).to(torch.float32))
    sd = torch.load(f"{model_dir}/{'sparse' if sparse else 'colbert'}_linear.pt", map_location="cpu")
    lin = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0])
    lin.load_state_dict(sd)
    unused = {tok.cls_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id}
    worst = 0.0
    for text, o in zip(texts, ours):
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state[0]
            if sparse:
                w = torch.relu(lin(h)).squeeze(-1)
                ref = {}
                for i, t in enumerate(enc.input_ids[0].tolist()):
                    if t in unused:
                        continue
                    k = tok.convert_ids_to_tokens(t)
                    ref[k] = max(ref.get(k, 0.0), float(w[i]))
                ref = {k: v for k, v in ref.items() if v > 0}
                assert set(ref) == set(o), f"token sets differ: {set(ref) ^ set(o)}"
                d = max(abs(ref[k] - o[k]) for k in ref)
            else:
                ref = torch.nn.functional.normalize(lin(h[1:]), dim=-1).numpy()
                o = np.array(o)
                assert o.shape == ref.shape, f"{o.shape} vs {ref.shape}"
                d = float(np.abs(o - ref).max())
                cos = (o * ref).sum(-1) / np.linalg.norm(o, axis=-1) / np.linalg.norm(ref, axis=-1)
                print(f"  per-token cosine: min {cos.min():.6f} mean {cos.mean():.6f}")
        print(f"  {len(enc.input_ids[0])} tokens: max abs diff {d:.2e}")
        worst = max(worst, d)
    print(f"{'sparse weights' if sparse else 'colbert vectors'}: max abs diff {worst:.2e} over {len(texts)} texts")


if __name__ == "__main__":
    main()
