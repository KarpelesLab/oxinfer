#!/usr/bin/env python3
"""Run an ANEMLL-format Core ML package with coremltools and check it against transformers.

usage: coreml_check.py DIR [--hf CHECKPOINT] [--tokens N] [--bench] [--plan] [options]

DIR is a package written by tools/coreml_convert.py (or by ANEMLL): meta.yaml, the .mlmodelc
files and the tokenizer. The runner follows docs/coreml-format.md: full prefill batches, then
one `infer` call per remaining token; packages without `prefill_last_token_output` (ANEMLL's)
re-run the last prompt token through `infer` as ANEMLL's chat.py does.

--hf      greedy-generate --tokens tokens with transformers (float32, CPU), then feed the same
          sequence through the Core ML model (teacher forcing) and compare the logits of every
          generated position: cosine similarity, top-1 agreement, KL(ref || coreml). Also
          reports where free-running Core ML greedy decoding first diverges.
--bench   prefill and decode speed (best of --runs) for each of --units.
--plan    which compute device (ANE / GPU / CPU) Core ML picks for every op of every model
          (MLComputePlan), with the op types that do not run on the Neural Engine.

Note: on macOS 26 Core ML's CPU_ONLY path returns NaN logits for these stateful fp16 graphs
(ANEMLL's own Qwen3 release too); its timings are still indicative. CPU_AND_GPU is correct.

examples:
  coreml_check.py out/qwen3 --hf ~/.cache/oxinfer/hub/Qwen--Qwen3-0.6B --tokens 300
  coreml_check.py out/qwen3 --bench --units cpu_and_ne,cpu_only --plan
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

DEFAULT_PROMPT = (
    "I am preparing a short lecture for first-year students about how computers understand language. "
    "Please explain, step by step, how a transformer language model turns a prompt into generated text: "
    "tokenization, embeddings, attention layers, the feed-forward blocks, the final projection to the "
    "vocabulary and the sampling loop. For each step give one concrete example with real words, mention "
    "what can go wrong, and keep the tone friendly. After that, compare the transformer with older "
    "recurrent networks, and finish with three questions the students could discuss in small groups, "
    "plus a one-paragraph summary that I can put on the final slide of the presentation."
)

UNITS = {
    "cpu_and_ne": "CPU_AND_NE",
    "cpu_only": "CPU_ONLY",
    "cpu_and_gpu": "CPU_AND_GPU",
    "all": "ALL",
}


def load_meta(d):
    import yaml

    meta = yaml.safe_load(open(os.path.join(d, "meta.yaml")))
    info = meta["model_info"]
    p = dict(info["parameters"])
    p["architecture"] = info.get("architecture", "")
    return p


def chunk_paths(d, p):
    first = p["ffn"]
    n = int(p.get("num_chunks", 1))
    if "_chunk_01of" not in first:  # ANEMLL releases sometimes list the name without the suffix
        first = first.replace(".mlmodelc", f"_chunk_01of{n:02d}.mlmodelc")
    return [os.path.join(d, first.replace("_chunk_01of", f"_chunk_{i + 1:02d}of")) for i in range(n)]


class Runner:
    def __init__(self, d, units="cpu_and_ne"):
        import coremltools as ct

        self.p = p = load_meta(d)
        cu = getattr(ct.ComputeUnit, UNITS[units])
        self.ctx = int(p["context_length"])
        self.bs = int(p["batch_size"])
        self.recurrent = bool(p.get("recurrent_state", False))
        self.last_out = bool(p.get("prefill_last_token_output", False))
        self.emb = ct.models.CompiledMLModel(os.path.join(d, p["embeddings"]), cu)
        self.head = ct.models.CompiledMLModel(os.path.join(d, p["lm_head"]), cu)
        self.chunks = []
        for path in chunk_paths(d, p):
            self.chunks.append(
                {fn: ct.models.CompiledMLModel(path, cu, function_name=fn) for fn in ("infer", "prefill")}
            )
        self.n_split = int(p.get("split_lm_head", 16))
        mask = np.full((self.ctx, self.ctx), -np.inf, dtype=np.float16)
        self.mask = np.triu(mask, 1)[None, None]
        self.reset()

    def reset(self):
        self.state = self.chunks[0]["prefill"].make_state()
        self.pos = 0

    def _run(self, fn, emb, pos, t):
        hidden, pli = emb
        inp = {
            "position_ids": np.arange(pos, pos + t, dtype=np.int32),
            "causal_mask": np.ascontiguousarray(self.mask[:, :, pos : pos + t]),
            "current_pos": np.array([pos], dtype=np.int32),
        }
        if pli is not None:  # Gemma 4 per-layer embeddings, fed to every chunk
            inp["per_layer_inputs"] = pli
        for c in self.chunks:
            inp["hidden_states"] = hidden
            hidden = c[fn].predict(inp, self.state)["output_hidden_states"]
        return hidden

    def logits(self, hidden):
        out = self.head.predict({"hidden_states": hidden.astype(np.float16)})
        return np.concatenate([out[f"logits{i + 1}"] for i in range(self.n_split)], axis=-1)[0, -1].astype(np.float32)

    def embed(self, ids):
        out = self.emb.predict({"input_ids": np.asarray(ids, dtype=np.int32)[None]})
        return out["hidden_states"], out.get("per_layer_inputs")

    def prefill(self, ids):
        """Feed a prompt; returns the logits for the next token."""
        n, bs = len(ids), self.bs
        hidden = None
        while self.pos + bs <= n:
            hidden = self._run("prefill", self.embed(ids[self.pos : self.pos + bs]), self.pos, bs)
            self.pos += bs
        if self.pos == n:
            if self.last_out:
                return self.logits(hidden)
            if self.recurrent:
                raise SystemExit("recurrent package without prefill_last_token_output")
            # ANEMLL contract: re-run the last token (idempotent for a KV cache).
            self.pos -= 1
        while self.pos < n:
            hidden = self._run("infer", self.embed(ids[self.pos : self.pos + 1]), self.pos, 1)
            self.pos += 1
        return self.logits(hidden)

    def step(self, tok):
        hidden = self._run("infer", self.embed([tok]), self.pos, 1)
        self.pos += 1
        return self.logits(hidden)


def reference(hf, ids, n_new):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()
    x = torch.tensor([ids])
    with torch.no_grad():
        g = model.generate(x, max_new_tokens=n_new, do_sample=False, min_new_tokens=n_new)
        seq = g[0].tolist()
        logits = model(input_ids=g).logits[0, len(ids) - 1 : -1].float().numpy()
    return seq[len(ids) :], logits


def log_softmax(x):
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def compare(r, ids, gen, ref_logits):
    r.reset()
    cos, agree, kl = [], 0, []
    lg = r.prefill(ids)
    for i in range(len(gen)):
        a, b = lg.astype(np.float64), ref_logits[i].astype(np.float64)
        with np.errstate(all="ignore"):  # macOS Accelerate raises spurious FP warnings
            cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        agree += int(a.argmax() == b.argmax())
        pb, pa = log_softmax(b.astype(np.float64)), log_softmax(a.astype(np.float64))
        kl.append(float((np.exp(pb) * (pb - pa)).sum()))
        if i + 1 < len(gen):
            lg = r.step(gen[i])
    return np.array(cos), agree, np.array(kl)


def free_run(r, ids, n):
    r.reset()
    lg = r.prefill(ids)
    out = []
    for _ in range(n):
        t = int(lg.argmax())
        out.append(t)
        if len(out) < n:
            lg = r.step(t)
    return out


def bench(d, units, runs, n_prefill, n_decode):
    r = Runner(d, units)
    ids = [100 + (i * 7919) % 5000 for i in range(n_prefill)]
    pf, dc = [], []
    for _ in range(runs):
        r.reset()
        t0 = time.perf_counter()
        r.prefill(ids)
        pf.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        for i in range(n_decode):
            r.step(ids[i])
        dc.append(time.perf_counter() - t0)
    return n_prefill / min(pf), n_decode / min(dc)


def _plan_copy(path, fn, default, tmp):
    """A copy of a compiled multifunction model whose default function is `fn`.

    MLComputePlan only plans the default function; swapping the function names in model.mil
    (weights symlinked) makes it plan another one."""
    dst = os.path.join(tmp, f"{os.path.basename(path)[:-9]}_{fn}.mlmodelc")
    os.makedirs(dst)
    for f in os.listdir(path):
        if f != "model.mil":
            os.symlink(os.path.join(path, f), os.path.join(dst, f))
    mil = open(os.path.join(path, "model.mil")).read()
    mil = mil.replace(f"func {default}<", "func __swap<").replace(f"func {fn}<", f"func {default}<")
    mil = mil.replace("func __swap<", f"func {fn}<")
    open(os.path.join(dst, "model.mil"), "w").write(mil)
    return dst


SKIP_OPS = ("const", "constexpr_lut_to_dense", "constexpr_blockwise_shift_scale", "constexpr_affine_dequantize")


def plan(d):
    import re
    import shutil
    import tempfile

    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    p = load_meta(d)
    paths = [os.path.join(d, p["embeddings"])] + chunk_paths(d, p) + [os.path.join(d, p["lm_head"])]
    tmp = tempfile.mkdtemp()
    try:
        for path in paths:
            mil = open(os.path.join(path, "model.mil")).read()
            fns = re.findall(r"func (\w+)<", mil)
            for fn in fns:
                target = path if fn == fns[0] or len(fns) == 1 else _plan_copy(path, fn, fns[0], tmp)
                cp = MLComputePlan.load_from_path(path=target, compute_units=ct.ComputeUnit.CPU_AND_NE)
                prog = cp.model_structure.program
                func = prog.functions[fns[0]]
                counts, off, cost = {}, {}, {}
                for op in func.block.operations:
                    name = op.operator_name.split(".")[-1]
                    if name in SKIP_OPS:
                        continue
                    usage = cp.get_compute_device_usage_for_mlprogram_operation(op)
                    dev = type(usage.preferred_compute_device).__name__ if usage else "unknown"
                    dev = dev.replace("ML", "").replace("ComputeDevice", "")
                    counts[dev] = counts.get(dev, 0) + 1
                    est = cp.get_estimated_cost_for_mlprogram_operation(op)
                    cost[dev] = cost.get(dev, 0.0) + (est.weight if est else 0.0)
                    if dev != "NeuralEngine":
                        off[name] = off.get(name, 0) + 1
                total = sum(cost.values()) or 1.0
                share = ", ".join(f"{k} {v} ops ({100 * cost[k] / total:.1f}% of estimated cost)" for k, v in sorted(counts.items()))
                print(f"{os.path.basename(path)} [{fn}]: {share}")
                if off:
                    print("    not on the ANE: " + ", ".join(f"{k} x{v}" for k, v in sorted(off.items())))
    finally:
        shutil.rmtree(tmp)


def dir_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(path) for f in fs)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 2)[2], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("package", help="directory with meta.yaml")
    ap.add_argument("--hf", help="Hugging Face checkpoint for the float32 reference")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="user message (default: a ~170-token request, long enough for two prefill batches)")
    ap.add_argument("--prompt-file", help="read the user message from this file (long-context checks)")
    ap.add_argument("--no-template", action="store_true", help="tokenize the prompt as is (no chat template)")
    ap.add_argument("--tokens", type=int, default=200, help="generated tokens to compare (default 200)")
    ap.add_argument("--ref-cache", help="npz file caching the reference sequence and logits")
    ap.add_argument("--single-token", action="store_true", help="feed the prompt through `infer` only (no `prefill` calls)")
    ap.add_argument("--bench", action="store_true", help="measure prefill and decode speed")
    ap.add_argument("--plan", action="store_true", help="print the compute device of every op (MLComputePlan)")
    ap.add_argument("--units", default="cpu_and_ne", help="compute units for --hf / --bench (comma list): " + ", ".join(UNITS))
    ap.add_argument("--runs", type=int, default=3, help="benchmark repetitions (best is reported)")
    ap.add_argument("--bench-prefill", type=int, default=256, help="prompt tokens for the prefill benchmark")
    ap.add_argument("--bench-decode", type=int, default=64, help="tokens for the decode benchmark")
    args = ap.parse_args()
    if args.prompt_file:
        args.prompt = open(args.prompt_file).read()
    d = os.path.abspath(os.path.expanduser(args.package))
    p = load_meta(d)
    files = [p["embeddings"], p["lm_head"]] + [os.path.basename(c) for c in chunk_paths(d, p)]
    print(f"{d}: {p['architecture']}, ctx {p['context_length']}, batch {p['batch_size']}, lut ffn {p.get('lut_ffn')} lm_head {p.get('lut_lmhead')} embeddings {p.get('lut_embeddings')}")
    for f in files:
        print(f"  {f}: {dir_size(os.path.join(d, f)) / 1e6:.0f} MB")

    if args.hf:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(d)
        if args.no_template:
            ids = tok(args.prompt, add_special_tokens=False).input_ids
        elif p["architecture"] == "gemma4" and not tok.chat_template:
            ids = tok(f"<bos><|turn>user\n{args.prompt}<turn|>\n<|turn>model\n", add_special_tokens=False).input_ids
        else:
            ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True, tokenize=False)
            ids = tok(ids, add_special_tokens=False).input_ids
        if args.ref_cache and os.path.exists(args.ref_cache):
            z = np.load(args.ref_cache)
            assert list(z["prompt"]) == list(ids), "reference cache was made for another prompt"
            gen, ref_logits = [int(x) for x in z["gen"]], z["logits"]
        else:
            t0 = time.time()
            gen, ref_logits = reference(os.path.expanduser(args.hf), ids, args.tokens)
            print(f"reference: {len(gen)} tokens in {time.time() - t0:.0f}s")
            if args.ref_cache:
                np.savez(args.ref_cache, prompt=np.array(ids), gen=np.array(gen), logits=ref_logits)
        gen, ref_logits = gen[: args.tokens], ref_logits[: args.tokens]
        for units in args.units.split(","):
            r = Runner(d, units)
            if args.single_token:
                r.bs = 1 << 30
            t0 = time.time()
            cos, agree, kl = compare(r, ids, gen, ref_logits)
            print(f"[{units}] prompt {len(ids)} tokens, {len(gen)} teacher-forced positions ({time.time() - t0:.0f}s):")
            print(f"  logits cosine mean {cos.mean():.5f} min {cos.min():.5f}; top-1 agreement {agree}/{len(gen)} ({100 * agree / len(gen):.1f}%); KL mean {kl.mean():.4f} max {kl.max():.3f}")
            fr = free_run(r, ids, len(gen))
            div = next((i for i, (a, b) in enumerate(zip(fr, gen)) if a != b), None)
            print(f"  free-running greedy: {'identical to the reference' if div is None else f'first divergence at token {div}'}")
            print("  coreml:", repr(tok.decode(fr[:60])))
            if div is not None:
                print("  ref:   ", repr(tok.decode(gen[:60])))

    if args.bench:
        for units in args.units.split(","):
            pf, dc = bench(d, units, args.runs, args.bench_prefill, args.bench_decode)
            print(f"[{units}] prefill {pf:.1f} tok/s ({args.bench_prefill} tokens), decode {dc:.1f} tok/s (best of {args.runs})")

    if args.plan:
        plan(d)


if __name__ == "__main__":
    main()
