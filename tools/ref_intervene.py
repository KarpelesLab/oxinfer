#!/usr/bin/env python3
"""Reference for oxinfer's runtime interventions (src/intervene.rs) with PyTorch forward hooks on
a transformers model (dense Qwen3.5 / Qwen3 / Llama / Gemma-style `model.layers`).

usage: ref_intervene.py MODEL_DIR PROMPT_TEXT_FILE SPEC.json [OURS_LOGITS.bin] [--bf16]
                        [--generate N] [--ours-ids IDS.json]

SPEC.json is the same JSON as `oxinfer --interventions` (neurons, steer, ablate, sae). It is
applied independently of oxinfer's implementation:
- neuron edits: a forward pre-hook on `layers[L].mlp.down_proj` (its input is act(gate) * up);
  unit resid_median: dose * median_{t>=1} |hidden_states[L+1][t]| / |W_down[:, n]| measured on the
  unedited model over the same prompt (numpy median);
- steering: `post` adds alpha * v to the output of `layers[L]`, `mid` to the attention block's
  output (the residual after attention), `pre` to the input of `layers[L]`;
- ablation: Arditi et al.: r_hat projected out of every layer's input (pre-hook) and of every
  attention and MLP block's output;
- SAE features (Qwen-Scope layerN.sae.pt, topk(relu(x W_enc^T + b_enc)) or SAELens): on the output
  of layers[L], x += (a' - a) W_dec[f] (error term kept).
Positions: `generated` = positions >= len(prompt) - 1, `prompt` = the others.
Prints the logits difference with OURS_LOGITS.bin (last prompt position), and with --generate the
greedy continuation under the same hooks (compared with --ours-ids, a JSON list, when given).
"""
import sys, json, struct
import numpy as np
import torch


def load_vectors(path, n_layers):
    if path.endswith(".gguf"):
        raise SystemExit("GGUF vectors: use the safetensors form for the reference")
    from safetensors.numpy import load_file
    t = load_file(path)
    out = {}
    for k, v in t.items():
        if k.startswith("layers.") or k.startswith("direction."):
            out[int(k.split(".")[1])] = v.astype(np.float32)
    if not out and len(t) == 1:
        out[None] = next(iter(t.values())).astype(np.float32).reshape(-1)
    return out


def load_sae(path):
    import os
    if path.endswith(".pt"):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        cfg = json.load(open(os.path.join(os.path.dirname(path), "config.json")))
        W_enc = sd["W_enc"].float()  # [d_sae, d_in]
        W_dec = sd["W_dec"].float()  # [d_in, d_sae]
        return dict(W_enc=W_enc, b_enc=sd["b_enc"].float(), W_dec=W_dec.T.contiguous(), b_dec=sd["b_dec"].float(), k=cfg["k"],
                    layer=int(os.path.basename(path)[5:].split(".")[0]))
    raise SystemExit("only Qwen-Scope .pt SAEs in the reference")


def sae_encode(s, x):
    pre = x @ s["W_enc"].T + s["b_enc"]
    r = torch.relu(pre)
    v, i = torch.topk(r, s["k"], dim=-1)
    a = torch.zeros_like(r)
    a.scatter_(-1, i, v)
    return a


def main():
    args, it = [], iter(sys.argv[1:])
    for a in it:
        if a in ("--generate", "--ours-ids"):
            next(it)
        elif not a.startswith("--"):
            args.append(a)
    model_dir, text_file, spec_file = args[:3]
    ours_file = args[3] if len(args) > 3 else None
    gen = int(sys.argv[sys.argv.index("--generate") + 1]) if "--generate" in sys.argv else 0
    ours_ids = json.load(open(sys.argv[sys.argv.index("--ours-ids") + 1])) if "--ours-ids" in sys.argv else None
    dtype = torch.bfloat16 if "--bf16" in sys.argv else torch.float32
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_dir)
    text = open(text_file).read()
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    T = ids.shape[1]
    spec = json.load(open(spec_file)) if not spec_file.startswith("{") else json.loads(spec_file)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    model.eval()
    base = model.model
    if hasattr(base, "language_model"):
        base = base.language_model
    layers = base.layers
    nl = len(layers)
    gen_from = T - 1
    state = {"start": 0}

    def pos_mask(positions, t, device):
        p = torch.arange(state["start"], state["start"] + t, device=device)
        if positions in (None, "all"):
            return torch.ones(t, dtype=torch.bool, device=device)
        if positions == "generated":
            return p >= gen_from
        return p < gen_from

    def cur_t(x):
        return x.shape[1]

    # Calibration (unedited model, numpy median over positions 1..).
    med = None
    if any(n.get("unit") == "resid_median" and "resid_median" not in n for n in spec.get("neurons", [])):
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        med = [float(np.median(hs[l + 1][0, 1:].float().norm(dim=-1).numpy())) for l in range(nl)]
        print("calibration medians:", {n["layer"]: round(med[n["layer"]], 4) for n in spec["neurons"]})

    hooks = []
    # Neuron edits.
    by_layer = {}
    for n in spec.get("neurons", []):
        if n.get("expert") is not None:
            raise SystemExit("MoE neuron edits are not in the reference")
        op = n.get("op") or ("add" if "dose" in n else "scale")
        val = n.get("value", n.get("dose"))
        L = n["layer"]
        if n.get("unit") == "resid_median":
            col = layers[L].mlp.down_proj.weight[:, n["neuron"]].float()
            m = n.get("resid_median", med[L] if med else None)
            val = val * m / float(col.norm())
            print(f"neuron {L}:{n['neuron']} add {val:.5f} (median {m:.4f}, |col| {float(col.norm()):.5f})")
        by_layer.setdefault(L, []).append((n["neuron"], op, val, n.get("min"), n.get("max"), n.get("positions")))
    for L, edits in by_layer.items():
        def pre(mod, inp, edits=edits):
            x = inp[0].clone()
            t = x.shape[1]
            for idx, op, val, lo, hi, ps in edits:
                m = pos_mask(ps, t, x.device)
                a = x[0, :, idx]
                if op == "scale":
                    b = a * val
                elif op == "zero":
                    b = a * 0
                elif op == "add":
                    b = a + val
                elif op == "set":
                    b = torch.full_like(a, val)
                elif op == "clamp":
                    b = a.clamp(min=lo if lo is not None else -float("inf"), max=hi if hi is not None else float("inf"))
                x[0, :, idx] = torch.where(m, b, a)
            return (x,) + tuple(inp[1:])
        hooks.append(layers[L].mlp.down_proj.register_forward_pre_hook(pre))

    # Gemma's sandwich norms: what the attention block adds to the residual is the output of
    # post_attention_layernorm, and the MLP's is followed by more (MoE, per-layer inputs, layer
    # scalar), so the layer output stands for it.
    gemma = hasattr(layers[0], "pre_feedforward_layernorm")

    def attn_mod(layer):
        if gemma:
            return layer.post_attention_layernorm
        return getattr(layer, "self_attn", None) or getattr(layer, "linear_attn")

    def add_to_output(mod, fn):
        def hook(m, inp, out):
            if isinstance(out, tuple):
                return (fn(out[0]),) + tuple(out[1:])
            return fn(out)
        return mod.register_forward_hook(hook)

    def add_to_input(mod, fn):
        def hook(m, args, kwargs):
            if args:
                return (fn(args[0]),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = fn(kwargs["hidden_states"])
            return args, kwargs
        return mod.register_forward_pre_hook(hook, with_kwargs=True)

    # Steering.
    for s in spec.get("steer", []):
        if "vector" in s:
            vecs = {s["layer"]: np.array(s["vector"], dtype=np.float32)}
        else:
            vs = load_vectors(s["file"], nl)
            want = s.get("layers") or ([s["layer"]] if "layer" in s else None)
            if None in vs:
                vecs = {l: vs[None] for l in want}
            else:
                vecs = {l: v for l, v in vs.items() if want is None or l in want}
        for L, v in vecs.items():
            v = torch.tensor(v)
            if s.get("normalize"):
                v = v / v.norm()
            v = v * s.get("alpha", 1.0)
            ps = s.get("positions")

            def fn(x, v=v, ps=ps):
                m = pos_mask(ps, x.shape[1], x.device).to(x.dtype)[None, :, None]
                return x + m * v.to(x.dtype)
            point = s.get("point", "post")
            if point == "post":
                hooks.append(add_to_output(layers[L], fn))
            elif point == "mid":
                hooks.append(add_to_output(attn_mod(layers[L]), fn))
            else:
                hooks.append(add_to_input(layers[L], fn))

    # Ablation (Arditi et al.).
    for a in spec.get("ablate", []):
        if "vector" in a:
            r = np.array(a["vector"], dtype=np.float32)
        else:
            vs = load_vectors(a["file"], nl)
            r = vs[None] if None in vs else vs[a["direction_layer"]] if "direction_layer" in a else next(iter(vs.values()))
        r = torch.tensor(r).double()
        r = (r / r.norm()).float()
        ps = a.get("positions")

        def proj(x, r=r, ps=ps):
            m = pos_mask(ps, x.shape[1], x.device).to(x.dtype)[None, :, None]
            rr = r.to(x.dtype)
            return x - m * (x @ rr)[..., None] * rr
        for L in a.get("layers") or range(nl):
            hooks.append(add_to_input(layers[L], proj))
            hooks.append(add_to_output(attn_mod(layers[L]), proj))
            hooks.append(add_to_output(layers[L] if gemma else layers[L].mlp, proj))

    # SAE feature steering.
    for s in spec.get("sae", []):
        sae = load_sae(s["sae"])
        L = s.get("layer", sae["layer"])
        feats = s["features"]
        ps = s.get("positions")

        def fn(x, sae=sae, feats=feats, ps=ps):
            xf = x.float()
            a = sae_encode(sae, xf)
            delta = torch.zeros_like(xf)
            for f in feats:
                cur = a[..., f["id"]]
                op = f.get("op", "clamp")
                if op == "add":
                    d = torch.full_like(cur, f["value"])
                elif op == "scale":
                    d = cur * f["value"] - cur
                elif op == "zero":
                    d = -cur
                else:
                    d = f["value"] - cur
                delta += d[..., None] * sae["W_dec"][f["id"]]
            m = pos_mask(ps, x.shape[1], x.device).to(xf.dtype)[None, :, None]
            return (xf + m * delta).to(x.dtype)
        hooks.append(add_to_output(layers[L], fn))

    def track(mod, args, kwargs):
        x = kwargs.get("input_ids", args[0] if args else None)
        if x is None:
            x = kwargs["inputs_embeds"]
        state["t"] = x.shape[1]
    def advance(mod, args, kwargs, out):
        state["start"] += state["t"]
    model.register_forward_pre_hook(track, with_kwargs=True)
    model.register_forward_hook(advance, with_kwargs=True)

    with torch.no_grad():
        out = model(input_ids=ids)
    ref = out.logits[0, -1].float().numpy()
    rt = np.argsort(-ref)[:10]
    print("ref  top10:", [(int(i), round(float(ref[i]), 3)) for i in rt])
    if ours_file:
        ours = np.fromfile(ours_file, dtype="<f4")
        d = np.abs(ref - ours)
        lr = ref - ref.max(); lr = lr - np.log(np.exp(lr).sum())
        lo = ours - ours.max(); lo = lo - np.log(np.exp(lo).sum())
        kl = float((np.exp(lr) * (lr - lo)).sum())
        ot = np.argsort(-ours)[:10]
        print("ours top10:", [(int(i), round(float(ours[i]), 3)) for i in ot])
        print(f"max abs diff {d.max():.5f}  mean abs diff {d.mean():.6f}  KL(ref||ours) {kl:.3e}  argmax match {int(rt[0]) == int(ot[0])}")
    if gen:
        state["start"] = 0
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=gen, do_sample=False)
        new = g[0, T:].tolist()
        print("ref greedy ids:", new)
        print("ref greedy text:", repr(tok.decode(new)))
        if ours_ids is not None:
            n = 0
            while n < min(len(new), len(ours_ids)) and new[n] == ours_ids[n]:
                n += 1
            print(f"greedy agreement: first {n} of {min(len(new), len(ours_ids))} tokens identical")


if __name__ == "__main__":
    main()
