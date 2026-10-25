"""Data-aware LUT palettization for coreml_convert.py.

Core ML's palettizer runs plain k-means on the weights of each group of output channels; at 4
bits that loses 15-20 points of top-1 agreement on sub-1B models. This module computes the LUTs
from a small calibration set instead (method `fisher`, the default of coreml_convert.py):

1. calibration: the float32 transformers model answers built-in chat prompts (greedy, in
   distribution for an instruction model, no dataset needed) or reads a text file;
2. sensitivity: the diagonal Fisher information of every linear weight, the squared gradient
   of the next-token loss summed over the calibration sequences (SqueezeLLM, Kim et al. 2023);
3. LUT: weighted 1-D k-means (greedy k-means++ init, Lloyd steps) of each group of output
   channels, every weight weighted by its Fisher information, so the levels go where the loss
   is sensitive instead of where the weights are dense;
4. mixed precision (optional): the estimated loss increase 0.5 sum F (w - q)^2 ranks the
   weights; the costliest move to a wider LUT under an average-bits budget.

Also here, experimental: `gptq` (GPTQ rounding over the LUT grid with input second moments,
Frantar et al. 2022, plus a least-squares LUT refit) and `wkmeans` (k-means weighted by diag H).
With a few thousand calibration tokens they overfit: lower error on the calibration data,
higher on other text. A per-output-channel scale on top of the LUT (`rms`) is supported too.

The result is (lut [groups, K], indices [out, in], scale [out] or None), which coreml_convert.py
writes as constexpr_lut_to_dense (+ constexpr_blockwise_shift_scale) ops.
"""
from __future__ import annotations

import math
import os
import time

import torch

# Calibration prompts (none of them is the validation prompt of coreml_check.py): a mix of
# prose, code, math, structured output and other languages.
CALIB_PROMPTS = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the difference between a process and a thread, with a C example using pthreads.",
    "Solve step by step: a train leaves at 9:40 at 72 km/h, a second one at 10:15 at 96 km/h on the same track. When does the second catch up?",
    "Write a Python function that parses an ISO 8601 date string without using datetime, with unit tests.",
    "Résume en français les causes principales de la Révolution française.",
    "用中文介绍一下长城的历史，以及它在今天的意义。",
    "日本の四季について、それぞれの季節の特徴を説明してください。",
    "Return a JSON object describing three fictional books (title, author, year, genres, rating). Only JSON.",
    "What are the pros and cons of nuclear energy? Answer as a balanced bullet list.",
    "Translate into German and Spanish: 'The meeting has been moved to Thursday afternoon because the client is travelling.'",
    "Write a SQL query that finds, for every customer, their most recent order and its total, then explain how it works.",
    "Give me a recipe for a vegetarian lasagna with a shopping list and timings.",
    "Explain in simple terms how public-key cryptography lets two strangers agree on a secret.",
    "Write a Rust function that merges two sorted vectors into one sorted vector, and explain its complexity.",
    "Describe the water cycle for a ten-year-old, then give a more technical version for a geology student.",
    "Compose a haiku sequence about a city waking up, then explain the imagery you used.",
    "My laptop gets very hot and the fan is loud when I watch videos. What can I check and fix myself?",
    "Write a cover letter for a junior data analyst position at a logistics company.",
    "Explain how vaccines train the immune system, and why booster doses are sometimes needed.",
    "Implement a small LRU cache class in JavaScript and show how it would be used in a web server.",
    "¿Cuáles son los lugares más interesantes para visitar en México? Da una lista con explicaciones.",
    "Erkläre den Unterschied zwischen Wetter und Klima und warum das wichtig ist.",
    "Prove that the square root of 2 is irrational, then explain the proof in plain words.",
    "Plan a three-day trip to Kyoto for a family with two children, with a budget estimate.",
    "What happened during the fall of the Berlin Wall, and what were its consequences for Europe?",
    "Write a bash script that backs up a directory to a timestamped tar.gz and deletes backups older than 7 days.",
    "Give feedback on this sentence and rewrite it: 'Me and him goes to the store yesterday for buying some foods.'",
    "Explain gradient descent with a numeric example of fitting a line to three points.",
    "Write a dialogue between a detective and a suspect who has a surprising alibi.",
    "Как работает фотосинтез? Объясни простыми словами.",
    "List ten common Linux commands with a one-line explanation and an example each.",
    "Describe the rules of chess for a complete beginner, including castling and en passant.",
]


def device():
    """Device of the palettization math: the CPU (on MPS, torch 2.14, the GPTQ loop computes
    wrong results, and it is not faster)."""
    return torch.device("cpu")


def model_device():
    """Device for the calibration model (generation, Fisher): MPS when available. transformers'
    CPU fallback for the Gated DeltaNet layers decodes Qwen3.5-0.8B at ~4 s per token on the CPU,
    100x slower than on MPS."""
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# ---------------------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------------------


def calibration_sequences(model, tok, n_new=192, text_file=None, max_len=512, cache=None, samples=1):
    """Token sequences for calibration: chat prompts with the model's own answers (greedy, then
    `samples - 1` sampled ones at temperature 0.8; in distribution for a chat model, no dataset
    needed), or chunks of a text file."""
    if cache and os.path.exists(cache):
        return [list(map(int, s)) for s in torch.load(cache)]
    seqs = []
    if text_file:
        ids = tok(open(text_file).read(), add_special_tokens=False).input_ids
        seqs = [ids[i : i + max_len] for i in range(0, len(ids), max_len) if len(ids[i : i + max_len]) > 16]
    else:
        prompts = []
        for p in CALIB_PROMPTS:
            if tok.chat_template:
                s = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
            else:  # Gemma 4 checkpoints without a template
                s = f"<bos><|turn>user\n{p}<turn|>\n<|turn>model\n"
            prompts.append(tok(s, add_special_tokens=False).input_ids)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
        t0 = time.time()
        bs = 8
        torch.manual_seed(0)
        for sample in range(samples):
            for b in range(0, len(prompts), bs):
                batch = prompts[b : b + bs]
                n = max(len(p) for p in batch)
                dev = next(model.parameters()).device
                x = torch.tensor([[pad] * (n - len(p)) + p for p in batch], device=dev)
                att = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in batch], device=dev)
                kw = dict(do_sample=True, temperature=0.8, top_p=0.95, top_k=0) if sample else dict(do_sample=False)
                with torch.no_grad():
                    g = model.generate(input_ids=x, attention_mask=att, max_new_tokens=n_new, pad_token_id=pad, **kw)
                for p, row in zip(batch, g[:, n:].tolist()):
                    end = next((i + 1 for i, t in enumerate(row) if t in eos), len(row))  # drop the padding
                    seqs.append(p + row[:end])
        print(f"  calibration: {len(seqs)} sequences, {sum(map(len, seqs))} tokens generated in {time.time() - t0:.0f}s")
    if cache:
        torch.save(seqs, cache)
    return seqs


def collect_fisher(model, seqs, select):
    """Diagonal Fisher information of the weights of every nn.Linear accepted by `select`: the
    sum over calibration sequences of the squared gradient of the sequence's next-token loss
    (SqueezeLLM's sensitivity). Returns {name: F [out, in] float32}."""
    mods = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and select(n)}
    for p in model.parameters():
        p.requires_grad_(False)
    # untie weights shared with other modules (a tied LM head would also collect the
    # gradients of the embedding lookup)
    mine = {id(m.weight) for m in mods.values()}
    others = {id(p) for n, mod in model.named_modules() if n not in mods for p in mod.parameters(recurse=False)}
    for m in mods.values():
        if id(m.weight) in others and id(m.weight) in mine:
            m.weight = torch.nn.Parameter(m.weight.detach().clone())
    for m in mods.values():
        m.weight.requires_grad_(True)
    fis = {n: torch.zeros_like(m.weight, dtype=torch.float32) for n, m in mods.items()}
    t0 = time.time()
    for s in seqs:
        x = torch.tensor([s], device=next(model.parameters()).device)
        loss = model(input_ids=x, labels=x).loss
        model.zero_grad(set_to_none=True)
        loss.backward()
        for n, m in mods.items():
            if m.weight.grad is not None:
                fis[n] += m.weight.grad.float() ** 2
    for m in mods.values():
        m.weight.requires_grad_(False)
        m.weight.grad = None
    print(f"  Fisher information: {len(fis)} weights over {len(seqs)} sequences in {time.time() - t0:.0f}s")
    return {n: f.cpu() for n, f in fis.items()}


def collect_hessians(model, seqs, select):
    """H = sum x x^T over the calibration tokens for the input of every nn.Linear whose name
    `select(name)` accepts. Linears fed by the same tensor (q/k/v, gate/up) share one H.
    Returns ({name: key}, {key: H (float32, cpu)})."""
    keys, hs, hooks = {}, {}, []
    seen = {}  # data_ptr of the input in the current forward -> key

    def hook(name):
        def f(mod, inp, out):
            x = inp[0].detach()
            ptr = (x.data_ptr(), tuple(x.shape))
            if name not in keys:
                keys[name] = seen.get(ptr, name)
            k = keys[name]
            seen[ptr] = k
            if k != name:
                return
            x = x.reshape(-1, x.shape[-1]).float()
            h = (x.t() @ x).cpu().double()
            hs[k] = hs[k] + h if k in hs else h

        return f

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and select(name):
            hooks.append(m.register_forward_hook(hook(name)))
    t0 = time.time()
    with torch.no_grad():
        for s in seqs:
            seen.clear()
            model(input_ids=torch.tensor([s], device=next(model.parameters()).device))
    for h in hooks:
        h.remove()
    n = sum(map(len, seqs))
    print(f"  calibration statistics: {len(hs)} inputs over {n} tokens in {time.time() - t0:.0f}s")
    return keys, {k: (v / n).float() for k, v in hs.items()}


# ---------------------------------------------------------------------------------------------
# Palettization
# ---------------------------------------------------------------------------------------------


def weighted_kmeans_1d(v, w, k, iters=100, seed=0):
    """Weighted 1-D k-means of every row of v [G, n] (weights w [G, n]); centroids [G, k] sorted.

    Greedy k-means++ init (as scikit-learn: 2 + log k candidates per step, keep the one that
    lowers the potential most; quantile inits leave the tails without levels and end up to 3x
    worse at 6 bits), then Lloyd steps with sorted-centroid bucketing."""
    g, n = v.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    trials = 2 + int(math.log(k))
    rows = torch.arange(g)
    vi, wi = v, w
    if n > 192 * k:  # init on a weighted subsample (keeps the extremes, which k-means++ needs)
        sub = torch.multinomial(w.clamp_min(1e-30), 128 * k, replacement=False, generator=gen)
        ext = torch.cat([v.argmin(1, keepdim=True), v.argmax(1, keepdim=True)], 1)
        sub = torch.cat([sub, ext], 1)
        v, w = torch.gather(v, 1, sub), torch.gather(w, 1, sub)
    first = torch.multinomial(w.clamp_min(1e-30), 1, generator=gen)[:, 0]
    c = [v[rows, first]]
    d = w * (v - c[0].view(-1, 1)) ** 2
    for _ in range(1, k):
        cand = torch.multinomial(d.clamp_min(0) + 1e-30, trials, generator=gen)  # [G, trials]
        cv = torch.gather(v, 1, cand)  # [G, trials]
        best, best_pot, best_d = None, None, None
        for t in range(trials):
            nd = torch.minimum(d, w * (v - cv[:, t : t + 1]) ** 2)
            pot = nd.sum(1)
            if best is None:
                best, best_pot, best_d = cv[:, t].clone(), pot, nd
            else:
                better = pot < best_pot
                best = torch.where(better, cv[:, t], best)
                best_pot = torch.where(better, pot, best_pot)
                best_d = torch.where(better.view(-1, 1), nd, best_d)
        c.append(best)
        d = best_d
    c = torch.stack(c, 1)
    v, w = vi, wi
    vs, order = v.sort(dim=1)
    ws = torch.gather(w, 1, order).double()
    zero = torch.zeros(g, 1, dtype=torch.float64, device=v.device)
    cw = torch.cat([zero, ws.cumsum(1)], 1)  # prefix sums: a cluster is a run of sorted values
    cwv = torch.cat([zero, (ws * vs.double()).cumsum(1)], 1)
    ends = torch.full((g, 1), n, dtype=torch.long, device=v.device)
    for _ in range(iters):
        c = c.sort(1)[0]
        cut = torch.searchsorted(vs, ((c[:, 1:] + c[:, :-1]) * 0.5).contiguous())  # [G, k-1]
        b = torch.cat([torch.zeros_like(ends), cut, ends], 1)
        den = torch.gather(cw, 1, b[:, 1:]) - torch.gather(cw, 1, b[:, :-1])
        num = torch.gather(cwv, 1, b[:, 1:]) - torch.gather(cwv, 1, b[:, :-1])
        nc = torch.where(den > 0, (num / den.clamp_min(1e-300)).to(c.dtype), c)
        if torch.equal(nc, c):
            break
        c = nc
    return c.sort(1)[0]


def nearest(lut_rows, w):
    """Index of the nearest entry of each row's sorted LUT (lut_rows [out, K]) for w [out, m]."""
    mids = ((lut_rows[:, 1:] + lut_rows[:, :-1]) * 0.5).contiguous()
    return torch.searchsorted(mids, w.contiguous())


DAMP = 0.01


def gptq_lut(w, h, lut_rows, damp=None, block=128):
    """GPTQ with a fixed per-row grid: returns indices [out, in] into lut_rows [out, K]."""
    damp = DAMP if damp is None else damp
    w = w.clone()
    n = w.shape[1]
    diag = torch.diag(h)
    perm = torch.argsort(diag, descending=True)
    w = w[:, perm]
    h = h[perm][:, perm].clone()
    dead = torch.diag(h) <= 0
    h[dead, dead] = 1.0
    h += torch.eye(n, device=h.device) * (damp * torch.diag(h).mean())
    # upper Cholesky factor of H^-1 (on the CPU: MPS lacks some linalg kernels)
    hc = h.cpu().double()
    u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hc)), upper=True).float().to(w.device)
    idx = torch.empty_like(w, dtype=torch.long)
    mids = ((lut_rows[:, 1:] + lut_rows[:, :-1]) * 0.5).contiguous()
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        w1 = w[:, i1:i2].clone()
        e1 = torch.empty_like(w1)
        u1 = u[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = w1[:, i : i + 1]
            j = torch.searchsorted(mids, col.contiguous())
            idx[:, i1 + i : i1 + i + 1] = j
            err = (col - torch.gather(lut_rows, 1, j)) / u1[i, i]
            w1[:, i:] -= err * u1[i : i + 1, i:]
            e1[:, i : i + 1] = err
        w[:, i2:] -= e1 @ u[i1:i2, i2:]
    out = torch.empty_like(idx)
    out[:, perm] = idx
    return out


def refit_lut(w, h, idx, scale, group, k, lut):
    """Least-squares LUT entries for fixed indices: minimizes sum_r (w_r - s_r O_r c_g)^T H (...)
    per group g; returns lut [G, K]."""
    out, n = w.shape
    ng = out // group
    a = torch.zeros(ng, k, k, device=w.device)
    b = torch.zeros(ng, k, device=w.device)
    hw = w @ h  # rows of (H w_r)^T, [out, in]
    eye = torch.eye(k, device=w.device)
    batch = max(1, int(5e7 // (n * k)))
    for r0 in range(0, out, batch):
        r1 = min(r0 + batch, out)
        o = eye[idx[r0:r1]]  # [rows, in, K] one-hot
        s = scale[r0:r1] if scale is not None else torch.ones(r1 - r0, device=w.device)
        ho = torch.matmul(h, o)  # [rows, in, K]
        ar = torch.matmul(o.transpose(1, 2), ho) * (s * s).view(-1, 1, 1)  # [rows, K, K]
        br = torch.matmul(hw[r0:r1].unsqueeze(1), o).squeeze(1) * s.view(-1, 1)  # [rows, K]
        g = torch.arange(r0, r1, device=w.device) // group
        a.index_add_(0, g, ar)
        b.index_add_(0, g, br)
    # unused entries keep their value (ridge towards the previous LUT)
    reg = 1e-6 * a.diagonal(dim1=1, dim2=2).mean(1).view(-1, 1, 1) + 1e-12
    a = a + reg * eye
    b = b + reg.view(-1, 1) * lut
    new = torch.linalg.solve(a.cpu(), b.cpu().unsqueeze(-1)).squeeze(-1).to(w.device)
    return new


def palettize(w, h, bits, group, method="fisher", scale_mode="none", rounds=1, dev=None, imp=None):
    """Palettize w [out, in] (float32), in groups of `group` output channels sharing a LUT.

    method: 'fisher' (k-means weighted by imp [out, in], the Fisher information), 'kmeans'
    (unweighted), 'wkmeans' (weighted by diag of the input statistics h [in, in]), 'gptq'
    (wkmeans or fisher init + GPTQ assignment + least-squares LUT refit, `rounds` times).
    scale_mode: 'none' or 'rms' (per output channel scale, see module doc).
    Returns (lut [G, K] sorted, idx [out, in] int64, scale [out] or None, w_hat [out, in])
    with fp16-exact LUT values and scales."""
    dev = dev or device()
    out, n = w.shape
    k = 1 << bits
    if out % group:
        raise ValueError(f"{out} rows not divisible by group {group}")
    w = w.to(dev, torch.float32)
    if h is None or method == "kmeans":
        diag = torch.ones(n, device=dev)
    else:
        h = h.to(dev, torch.float32)
        diag = torch.diag(h).clamp_min(1e-12)
    scale = None
    if scale_mode == "rms":
        if h is not None and method != "kmeans":
            scale = torch.sqrt((w * w * diag).sum(1) / diag.sum())  # importance-weighted RMS
        else:
            scale = w.pow(2).mean(1).sqrt()
        scale = scale.clamp_min(1e-8).half().float()
    wn = w / scale.view(-1, 1) if scale is not None else w
    if imp is not None and method in ("fisher", "gptq"):  # per-weight importance (Fisher)
        wt = imp.to(dev, torch.float32).clamp_min(0) + 1e-12 * float(imp.max())
        wt = wt / wt.mean().clamp_min(1e-30)
    else:
        wt = diag.view(1, -1).expand(out, n)
    if scale is not None:
        wt = wt * (scale * scale).view(-1, 1)
    lut = torch.cat(
        [weighted_kmeans_1d(wn[g0 : g0 + 4096 * group].reshape(-1, group * n), wt[g0 : g0 + 4096 * group].reshape(-1, group * n), k)
         for g0 in range(0, out, 4096 * group)]
    )
    lut = lut.half().float().sort(1)[0]

    def rows_lut(lut):
        lr = lut.repeat_interleave(group, 0)
        return lr * scale.view(-1, 1) if scale is not None else lr

    if method == "gptq" and h is not None:
        for _ in range(rounds):
            idx = gptq_lut(w, h, rows_lut(lut))
            lut = refit_lut(w, h, idx, scale, group, k, lut).half().float()
            # keep sorted (the GPTQ search needs it): permute the indices with the entries
            lut, order = lut.sort(1)
            inv = torch.argsort(order, 1)
            idx = torch.gather(inv.repeat_interleave(group, 0), 1, idx)
        # final assignment with the refitted LUT
        idx = gptq_lut(w, h, rows_lut(lut))
    else:
        idx = torch.cat([nearest(rows_lut(lut)[r : r + 1024], w[r : r + 1024]) for r in range(0, out, 1024)])
    w_hat = torch.gather(rows_lut(lut), 1, idx)
    return lut.cpu(), idx.cpu(), None if scale is None else scale.cpu(), w_hat.cpu()


def output_error(w, w_hat, h):
    """Relative output error tr((W - Q) H (W - Q)^T) / tr(W H W^T)."""
    d = (w - w_hat).to(h.device)
    w = w.to(h.device)
    return float(((d @ h) * d).sum() / ((w @ h) * w).sum())


# ---------------------------------------------------------------------------------------------
# Core ML side
# ---------------------------------------------------------------------------------------------


def weight_key(w16) -> bytes:
    """Identity of an fp16 weight array (the torch module weight before tracing equals the MIL
    const value after conversion)."""
    import hashlib

    import numpy as np

    a = np.ascontiguousarray(w16)
    return hashlib.sha1(a.tobytes() + str(a.shape).encode()).digest()


def lut_entry(lut, idx, scale, group, bits, shape):
    """constexpr_lut_to_dense (iOS 18) operands for a conv weight of `shape` [out, in, 1, 1]:
    indices uint<bits> of the weight's shape, lut [out/group, 1, 1, 1, 2^bits, 1] fp16, and the
    per-output-channel scale [out, 1, 1, 1] fp16 (or None)."""
    import numpy as np

    from coremltools.converters.mil.mil import types

    k = 1 << bits
    dt = types.nptype_from_builtin(types.string_to_builtin(f"uint{bits}"))
    ind = idx.numpy().astype(np.uint8).reshape(shape).astype(dt)
    lt = lut.numpy().astype(np.float16).reshape(shape[0] // group, 1, 1, 1, k, 1)
    sc = None if scale is None else scale.numpy().astype(np.float16).reshape(shape[0], 1, 1, 1)
    return ind, lt, sc


def apply_luts(mlmodel, table: dict, keep=frozenset()):
    """Replace every const whose value is in `table` (weight_key -> lut_entry) by its palettized
    form. Returns (model, replaced, unmatched) with `unmatched` the 4-D fp16 consts of more
    than 2048 elements left as is."""
    import numpy as np

    import coremltools as ct
    from coremltools.converters.mil.frontend import _utils as fe
    from coremltools.converters.mil.mil import Builder as mb
    from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
    from coremltools.converters.mil.mil.passes.helper import block_context_manager
    from coremltools.models import utils as mu

    stats = {"replaced": 0, "unmatched": 0}

    class LutPass(AbstractGraphPass):
        def apply(self, prog):
            @block_context_manager
            def run(block):
                for op in list(block.operations):
                    for b in op.blocks:
                        run(b)
                    if op.op_type != "const":
                        continue
                    v = op.outputs[0].val
                    if not isinstance(v, np.ndarray) or v.ndim != 4 or v.dtype != np.float16:
                        continue
                    key = weight_key(v)
                    ent = table.get(key)
                    if ent is None:
                        stats["unmatched"] += v.size > 2048 and key not in keep
                        continue
                    ind, lt, sc = ent
                    new = fe._construct_constexpr_lut_op(ind, lt, None, name=op.name + "_palettized", before_op=op)
                    if sc is not None:
                        new = mb.constexpr_blockwise_shift_scale(
                            data=new, scale=sc, offset=None, before_op=op, name=op.name + "_palettized_pcs"
                        )
                    op.enclosing_block.replace_uses_of_var_after_op(
                        anchor_op=op, old_var=op.outputs[0], new_var=new, no_check_var_types=True
                    )
                    op.enclosing_block.remove_ops([op])
                    stats["replaced"] += 1

            for f in prog.functions.values():
                run(f)

    m = mu._apply_graph_pass(mlmodel, LutPass(), spec_version=ct._SPECIFICATION_VERSION_IOS_18)
    return m, stats["replaced"], stats["unmatched"]
