#!/usr/bin/env python3
"""Interleaved A/B benchmark of oxinfer against llama.cpp on the same GGUF file.

Every repetition runs each variant once, alternating engines (llama.cpp, oxinfer, llama.cpp,
...), and reverses the order on every other repetition so slow drift in machine load does not
favour one side. Each run records the load average before it starts. Results are appended to a
JSONL file; `summarize.py` turns them into tables (median and min/max) and checks greedy output
agreement between the engines.

llama.cpp runs through `llama-server` (one server start per variant and repetition; the
/completion endpoint reports prompt and generation timings and draft acceptance). The prompt
sent to it is the exact text oxinfer builds with its chat template (`--print-prompt`), so both
engines process the same tokens. Each server gets one warm-up request (Metal compiles batched
kernels on first use); oxinfer runs once per workload as a fresh process (it compiles its
shaders before the timers start).

Workloads (all greedy):
  pp512, pp2048   prompt processing of a ~512 / ~2048 token prompt (text from corpus.txt, a
                  frozen excerpt of this repository's README, cut to length with llama.cpp's tokenizer),
                  one generated token
  code, prose     256 generated tokens after a short prompt (stops early at end of turn)

Variants (pick with --variants; spec variants only run the code/prose workloads):
  llama-metal     llama.cpp, all layers on the GPU
  llama-cpu       llama.cpp, --device none -ngl 0 (CPU threads; Accelerate BLAS for batches)
  llama-mtp2/3    llama.cpp, embedded MTP head, --spec-type draft-mtp, depth 2 / 3
  llama-fast3     llama.cpp, FastMTP sidecar (--sidecar, needs the HauhauCS patch), depth 3
  ox              oxinfer defaults (Metal prefill when available)
  ox-cpu          oxinfer --device cpu
  ox-mtp          oxinfer --mtp
  ox-mtpfrom      oxinfer --mtp-from SIDECAR

Example:
  tools/bench/bench.py --model M.gguf --llama-bin ~/llama.cpp/build/bin \\
      --oxinfer target/release/oxinfer --reps 3 --out results.jsonl
  tools/bench/summarize.py results.jsonl
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPTS = {
    "code": "Write a Python function that parses a CSV file with a header row and returns a list of "
    "dictionaries, with type hints and a docstring.",
    "prose": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
}
GEN_TOKENS = 256
SPEC_VARIANTS = {"llama-mtp2", "llama-mtp3", "llama-fast3", "ox-mtp", "ox-mtpfrom"}
DEFAULT_VARIANTS = "llama-metal,ox,llama-cpu,ox-cpu,llama-mtp2,ox-mtp,llama-mtp3,ox-mtpfrom,llama-fast3"
PORT = 18093


def loadavg():
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return None


def llama_args(variant, a):
    base = [os.path.join(a.llama_bin, "llama-server"), "-m", a.model, "--port", str(PORT), "-c", "4096", "-np", "1"]
    base += ["-b", "2048", "-ub", "512"]
    if variant == "llama-cpu":
        return base + ["--device", "none", "-ngl", "0"] + (["-t", str(a.threads)] if a.threads else [])
    base += ["-ngl", "999"]
    if variant == "llama-metal":
        return base
    if variant in ("llama-mtp2", "llama-mtp3"):
        return base + ["--spec-type", "draft-mtp", "--spec-draft-n-max", variant[-1], "--spec-draft-p-min", "0"]
    if variant == "llama-fast3":
        if not a.sidecar:
            raise SystemExit("llama-fast3 needs --sidecar")
        return base + [
            "--spec-draft-model", a.sidecar, "--spec-draft-ngl", "all", "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "3", "--spec-draft-p-min", "0",
        ]  # fmt: skip
    raise SystemExit(f"unknown variant {variant}")


def ox_args(variant, a):
    base = [a.oxinfer, "--model", a.model, "--no-system", "--show-tokens"] + a.ox_extra.split()
    if a.threads and variant == "ox-cpu":
        base += ["--threads", str(a.threads)]
    return base + {
        "ox": [],
        "ox-cpu": ["--device", "cpu"],
        "ox-mtp": ["--mtp"],
        "ox-mtpfrom": ["--mtp-from", a.sidecar or ""],
    }[variant]


def http(path, body=None, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}")
    if body is not None:
        req.data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Server:
    def __init__(self, args, log):
        self.log = open(log, "a")
        self.p = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            if self.p.poll() is not None:
                raise SystemExit(f"llama-server exited (see {log})")
            try:
                if http("/health", timeout=2).get("status") == "ok":
                    return
            except Exception:
                pass
            if time.time() - t0 > 600:
                raise SystemExit("llama-server did not start")
            time.sleep(0.5)

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.log.close()


def template(a):
    """The chat template oxinfer applies, as (prefix, suffix) around the user message."""
    marker = "@@MSG@@"
    out = subprocess.run(
        [a.oxinfer, "--model", a.model, "--no-system", "--print-prompt", "--mmap", "-p", marker],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    pre, suf = out.split(marker)
    return pre, suf


def long_prompts(a, pre, suf):
    """User messages of ~512 / ~2048 template tokens cut from corpus.txt (token counts from llama.cpp)."""
    cache = a.out + ".prompts.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    words = open(os.path.join(HERE, "corpus.txt")).read().split(" ")
    srv = Server(llama_args("llama-cpu", a), a.out + ".server.log")
    try:

        def ntok(k):
            msg = " ".join(words[:k]) + "\n\nSummarize the text above in one sentence."
            return len(http("/tokenize", {"content": pre + msg + suf, "parse_special": True})["tokens"]), msg

        res = {}
        for target in (512, 2048):
            lo, hi = 1, len(words)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if ntok(mid)[0] <= target:
                    lo = mid
                else:
                    hi = mid - 1
            n, msg = ntok(lo)
            res[f"pp{target}"] = msg
            print(f"pp{target}: {n} tokens", file=sys.stderr)
    finally:
        srv.stop()
    json.dump(res, open(cache, "w"))
    return res


def run_llama(srv, prompt, n):
    d = http("/completion", {
        "prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "cache_prompt": False,
        "return_tokens": True, "samplers": ["top_k"],
    })  # fmt: skip
    t = d["timings"]
    r = {
        "prompt_tokens": t["prompt_n"], "pp_tps": t["prompt_per_second"],
        "gen_tokens": t["predicted_n"], "tg_tps": t["predicted_per_second"], "tokens": d.get("tokens", [])[:128],
    }  # fmt: skip
    if "draft_n" in t:
        r["draft_n"], r["draft_acc"] = t["draft_n"], t["draft_n_accepted"]
    return r


RE_TIMES = re.compile(r"\[prefill (\d+) tok in ([\d.]+)s = [\d.]+ tok/s \| decode (\d+) tok in ([\d.]+)s = [\d.]+ tok/s\]")
RE_MTP = re.compile(r"\[MTP: (\d+) main forwards for (\d+) tokens = ([\d.]+) tok/forward; drafts accepted (\d+)/(\d+)")
RE_OUT = re.compile(r"\[output ids: \[([\d, ]*)\]\]")


def run_ox(args, msg, n, log):
    p = subprocess.run(args + ["--max-tokens", str(n), "-p", msg], capture_output=True, text=True)
    with open(log, "a") as f:
        f.write(" ".join(args[:6]) + " ...\n" + p.stderr + "\n")
    m = RE_TIMES.search(p.stderr)
    if p.returncode or not m:
        raise SystemExit(f"oxinfer failed:\n{p.stderr[-3000:]}")
    pn, ps, dn, ds = int(m[1]), float(m[2]), int(m[3]), float(m[4])
    r = {"prompt_tokens": pn, "pp_tps": pn / ps, "gen_tokens": dn, "tg_tps": dn / ds if dn else None}
    o = RE_OUT.search(p.stderr)
    r["tokens"] = [int(x) for x in o[1].split(",") if x.strip()][:128] if o else []
    if mm := RE_MTP.search(p.stderr):
        r["forwards"], r["tok_per_fwd"] = int(mm[1]), float(mm[3])
        r["draft_acc"], r["draft_n"] = int(mm[4]), int(mm[5])
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="GGUF file (both engines load it)")
    ap.add_argument("--sidecar", help="FastMTP sidecar GGUF (llama-fast3, ox-mtpfrom)")
    ap.add_argument("--llama-bin", required=True, help="llama.cpp build/bin directory")
    ap.add_argument("--oxinfer", default=os.path.join(HERE, "../../target/release/oxinfer"))
    ap.add_argument("--ox-extra", default="", help="extra oxinfer flags for every run (e.g. --mmap)")
    ap.add_argument("--threads", type=int, help="CPU threads for the CPU-only variants (default: each engine's)")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--workloads", default="pp512,pp2048,code,prose")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-load", type=float, help="before each run, wait until the 1-minute load average is below this")
    ap.add_argument("--tag", default="", help="label stored with every record (e.g. the model name)")
    ap.add_argument("--out", required=True, help="JSONL file to append to")
    a = ap.parse_args()
    a.oxinfer = os.path.abspath(a.oxinfer)

    pre, suf = template(a)
    workloads = a.workloads.split(",")
    msgs = dict(PROMPTS)
    if any(w.startswith("pp") for w in workloads):
        msgs.update(long_prompts(a, pre, suf))
    variants = a.variants.split(",")
    log = a.out + ".log"
    for rep in range(a.reps):
        order = variants if rep % 2 == 0 else variants[::-1]
        for v in order:
            wl = [w for w in workloads if not (v in SPEC_VARIANTS and w.startswith("pp"))]
            srv = Server(llama_args(v, a), a.out + ".server.log") if v.startswith("llama") else None
            if srv:
                # Warm-up: Metal pipelines for batched kernels compile on first use.
                run_llama(srv, pre + msgs["code"] * 8 + suf, 4)
            try:
                for w in wl:
                    n = 1 if w.startswith("pp") else GEN_TOKENS
                    while a.max_load and loadavg() is not None and loadavg() > a.max_load:
                        time.sleep(5)
                    la, t0 = loadavg(), time.time()
                    if srv:
                        r = run_llama(srv, pre + msgs[w] + suf, n)
                    else:
                        r = run_ox(ox_args(v, a), msgs[w], n, log)
                    rec = {"tag": a.tag, "rep": rep, "variant": v, "workload": w, "load": la, "wall": round(time.time() - t0, 2)}
                    rec.update(r)
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    acc = f" acc {r['draft_acc']}/{r['draft_n']}" if r.get("draft_n") else ""
                    tg = f"{r['tg_tps']:.1f}" if r.get("tg_tps") else "-"
                    print(
                        f"[rep {rep} {v:12s} {w:7s}] load {la} pp {r['prompt_tokens']} @ {r['pp_tps']:.1f} | "
                        f"tg {r['gen_tokens']} @ {tg}{acc}",
                        flush=True,
                    )
            finally:
                if srv:
                    srv.stop()


if __name__ == "__main__":
    main()
