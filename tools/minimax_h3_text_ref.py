#!/usr/bin/env python3
"""MiniMax-H3's prompt conditioning on a real GGUF text encoder, in transformers: the reference
for the embedding oxinfer computes from `--text-encoder Qwen3-VL-32B-Instruct-Q8_0.gguf`.

  minimax_h3_text_ref.py ids TOKENIZER_DIR PROMPT.txt [--out IDS.txt]
      the token ids diffusers' MiniMaxH3 text step feeds the conditioner
      (`tokenizer(prompt, add_special_tokens=False)`: no chat template, no special tokens), with
      the release's `tokenizer/` (tokenizer.json and tokenizer_config.json, whose
      `additional_special_tokens` adds `<d>`, `</d>`, ... after the last id of tokenizer.json).

  minimax_h3_text_ref.py run TEXT_ENCODER.gguf IDS.txt OUT_DIR [--layers 1 2 25 50] [--threads N] [--dtype bf16]
      `hidden_states[i]` of the language model for the ids (whitespace separated), as
      diffusers' `get_qwen3vl_prompt_embeds` reads them (`text_encoder.model(...,
      output_hidden_states=True).hidden_states[50]`: the residual stream after the 50th decoder
      layer, no final norm), written as OUT_DIR/hs_<i>.bin (float32 `[tokens, width]`).

  minimax_h3_text_ref.py cmp OUT_DIR/hs_50.bin EMBEDS.bin
      against oxinfer's embedding: an OXINFER_H3_PROMPT_EMBEDS file, or raw float32 rows
      (`h3_text_check --dump`).

The weights are the GGUF's own: every tensor is read with the `gguf` package and dequantized
(`gguf.quants.dequantize`) to float32, so the reference and oxinfer hold the same values and what
differs is the arithmetic (oxinfer multiplies the stored Q8_0 blocks). The model is transformers'
`Qwen3VLTextModel` with the configuration of the GGUF's metadata and one decoder layer, refilled
for each of the layers in turn (1.45 GB of float32 at a time instead of 72 GB for 50 layers):
that layer's output on the previous layer's output is `hidden_states[i]` of the whole model (the
positions are the same in every call). It is taken by a forward hook: the last entry of a
model's `hidden_states` is after the final norm, so the one-layer model's `hidden_states[1]` is
not it (which is why diffusers refuses a conditioner cut to exactly 50 layers). Text only: positions `(i, i, i)`, which is what `Qwen3VLModel` computes
without vision inputs. Float32 on the CPU, about a second per layer for 124 tokens. `--dtype
bf16` rounds the dequantized weights to bfloat16 and computes in bfloat16, the dtype the release's
`text_encoder/` is stored and loaded in: what that arithmetic alone moves `hidden_states[50]` by.
"""
import argparse, os, struct, sys, time

import numpy as np

NAMES = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def cmd_ids(a):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    text = open(a.prompt, encoding="utf-8").read()
    if a.strip:
        text = text.rstrip("\n")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    print(f"{len(ids)} tokens ({type(tok).__name__}, {len(tok)} entries)", file=sys.stderr)
    out = " ".join(str(i) for i in ids)
    if a.out:
        open(a.out, "w").write(out + "\n")
    else:
        print(out)


def cmd_run(a):
    import torch
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from transformers import Qwen3VLTextConfig, Qwen3VLTextModel

    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    ids = [int(v) for v in open(a.ids).read().split()]
    r = GGUFReader(a.gguf)
    arch = r.fields["general.architecture"].contents()
    md = lambda k: r.fields[f"{arch}.{k}"].contents()
    cfg = Qwen3VLTextConfig(
        vocab_size=8,
        hidden_size=md("embedding_length"),
        intermediate_size=md("feed_forward_length"),
        num_hidden_layers=1,
        num_attention_heads=md("attention.head_count"),
        num_key_value_heads=md("attention.head_count_kv"),
        head_dim=md("attention.key_length"),
        rms_norm_eps=md("attention.layer_norm_rms_epsilon"),
        rope_theta=md("rope.freq_base"),
        rope_scaling={"rope_type": "default", "mrope_section": list(md("rope.dimension_sections"))[:3], "mrope_interleaved": True},
        max_position_embeddings=md("context_length"),
        attention_dropout=0.0,
    )
    n_layers = md("block_count")
    print(
        f"{arch}: {n_layers} layers of width {cfg.hidden_size}, {cfg.num_attention_heads} heads of {cfg.head_dim} "
        f"({cfg.num_key_value_heads} for keys and values), MLP {cfg.intermediate_size}, theta {cfg.rope_parameters['rope_theta']}, "
        f"sections {cfg.rope_parameters['mrope_section']}, eps {cfg.rms_norm_eps}"
    )
    tensors = {t.name: t for t in r.tensors}
    dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    f32 = lambda t: torch.from_numpy(np.array(dequantize(t.data, t.tensor_type), dtype=np.float32)).to(dtype)
    emb = tensors["token_embd.weight"]
    h = torch.from_numpy(np.array(dequantize(np.asarray(emb.data[ids]), emb.tensor_type), dtype=np.float32))[None].to(dtype)
    print(f"{len(ids)} tokens, embeddings from {emb.tensor_type.name} rows: rms {h.float().pow(2).mean().sqrt():.4f}, {a.dtype}")
    model = Qwen3VLTextModel(cfg).to(dtype).eval()
    taken = []
    model.layers[0].register_forward_hook(lambda _m, _i, o: taken.append(o[0] if isinstance(o, tuple) else o))
    n = len(ids)
    # What Qwen3VLModel.compute_3d_position_ids gives without vision inputs.
    pos = torch.arange(n).view(1, 1, -1).expand(3, 1, -1)
    mask = torch.ones(1, n, dtype=torch.long)
    os.makedirs(a.out, exist_ok=True)
    last = max(a.layers)
    if last > n_layers:
        sys.exit(f"hidden_states[{last}] asked of {n_layers} layers")
    for i in range(last):
        t0 = time.time()
        sd = {}
        for g, name in NAMES.items():
            t = tensors[f"blk.{i}.{g}"]
            sd[name] = f32(t)
        model.layers[0].load_state_dict(sd, strict=True)
        out = model(inputs_embeds=h, attention_mask=mask, position_ids=pos, use_cache=False, output_hidden_states=True)
        # The last entry of a model's hidden_states is after the final norm: the layer's own
        # output is taken by the hook, and checked to be that entry before the norm.
        assert len(out.hidden_states) == 2 and torch.equal(out.hidden_states[0], h)
        (y,) = taken
        taken.clear()
        assert torch.equal(model.norm(y), out.hidden_states[1])
        h = y
        norms = h[0].float().norm(dim=-1)
        print(
            f"hidden_states[{i + 1}]: rms {h.float().pow(2).mean().sqrt():.4f}, row norms first {norms[0]:.1f} median {norms.median():.1f} "
            f"max of the others {norms[1:].max() if n > 1 else 0:.1f} ({time.time() - t0:.1f}s)",
            flush=True,
        )
        if i + 1 in a.layers:
            h[0].float().numpy().astype("<f4").tofile(os.path.join(a.out, f"hs_{i + 1}.bin"))


def load_rows(path, width):
    """An OXINFER_H3_PROMPT_EMBEDS file (u32 n, n tag bytes, float32 [n, width]) or raw float32 rows."""
    b = open(path, "rb").read()
    if len(b) >= 4:
        n = struct.unpack("<I", b[:4])[0]
        if n and len(b) == 4 + n + 4 * n * width:
            return np.frombuffer(b[4 + n :], dtype="<f4").reshape(n, width).astype(np.float64)
    return np.frombuffer(b, dtype="<f4").reshape(-1, width).astype(np.float64)


def cmd_cmp(a):
    ref, got = load_rows(a.ref, a.width), load_rows(a.got, a.width)
    if ref.shape != got.shape:
        sys.exit(f"reference {ref.shape}, oxinfer {got.shape}")
    d = got - ref
    nr = np.linalg.norm(ref, axis=1)
    rel = np.linalg.norm(d, axis=1) / nr
    cos = (ref * got).sum(1) / nr / np.linalg.norm(got, axis=1)
    print(f"{ref.shape[0]} rows of {ref.shape[1]}: reference rms {np.sqrt((ref**2).mean()):.4f}, values up to {abs(ref).max():.1f}")
    print(f"  max abs diff {abs(d).max():.4e} (row {abs(d).max(1).argmax()}), relative (Frobenius) {np.linalg.norm(d) / np.linalg.norm(ref):.4e}")
    print(f"  per row, relative: median {np.median(rel):.4e}, max {rel.max():.4e} (row {rel.argmax()}), min {rel.min():.4e}")
    print(f"  per row, cosine: median {np.median(cos):.7f}, min {cos.min():.7f} (row {cos.argmin()})")
    print(f"  row norms: first {nr[0]:.1f}, median {np.median(nr):.1f}, max of the others {nr[1:].max() if len(nr) > 1 else 0:.1f}")
    if len(nr) > 1:
        print(f"  without row 0: relative {np.linalg.norm(d[1:]) / np.linalg.norm(ref[1:]):.4e}, max abs diff {abs(d[1:]).max():.4e}")
    if a.shift:
        # Controls: the same rows against a neighbour (a wrong crop or an off-by-one in the rows).
        for s in (1, -1):
            g = np.roll(got, s, axis=0)[1:-1]
            print(f"  control, oxinfer's rows shifted by {s}: relative {np.linalg.norm(g - ref[1:-1]) / np.linalg.norm(ref[1:-1]):.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ids")
    s.add_argument("tokenizer")
    s.add_argument("prompt")
    s.add_argument("--out")
    s.add_argument("--strip", action="store_true", help="drop the file's trailing newlines (a shell's $(cat FILE))")
    s = sub.add_parser("run")
    s.add_argument("gguf")
    s.add_argument("ids")
    s.add_argument("out")
    s.add_argument("--layers", type=int, nargs="+", default=[1, 2, 25, 50])
    s.add_argument("--threads", type=int, default=16)
    s.add_argument("--dtype", choices=["f32", "bf16"], default="f32", help="bf16: the weights and the arithmetic of the release's text encoder (torch_dtype bfloat16)")
    s = sub.add_parser("cmp")
    s.add_argument("ref")
    s.add_argument("got")
    s.add_argument("--width", type=int, default=5120)
    s.add_argument("--shift", action="store_true")
    a = p.parse_args()
    {"ids": cmd_ids, "run": cmd_run, "cmp": cmd_cmp}[a.cmd](a)


if __name__ == "__main__":
    main()
