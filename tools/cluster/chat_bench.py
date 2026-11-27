#!/usr/bin/env python3
"""Concurrent chat requests against an OpenAI-compatible endpoint.

  chat_bench.py tput URL KEY MODEL N [MAX_TOKENS] [ROUND]   aggregate tokens/s of N requests at once
  chat_bench.py ttft URL KEY MODEL REPEATS [PAUSE]          time to the first streamed token
"""
import http.client, json, sys, threading, time, statistics
from urllib.parse import urlparse

TOPICS = ["a lighthouse keeper", "a river in winter", "a broken clock", "a night train", "a small bakery",
          "a mountain pass", "an old library", "a harbor at dawn", "a desert well", "a city garden",
          "a paper kite", "a stone bridge", "a ferry crossing", "a market square", "a snowed-in cabin",
          "a radio operator"]


def post(url, key, body, stream=False):
    u = urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    c.request("POST", "/v1/chat/completions", json.dumps(body),
              {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    return c, c.getresponse()


def tput(url, key, model, n, max_tokens, rnd):
    out = [None] * n

    def one(i):
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "user", "content": f"(run {rnd}.{i}) Write a long, detailed story about {TOPICS[i % len(TOPICS)]}. At least 600 words."}]}
        t0 = time.time()
        c, r = post(url, key, body)
        data = r.read()
        c.close()
        try:
            v = json.loads(data)
            out[i] = (v["usage"]["completion_tokens"], v["usage"]["prompt_tokens"], time.time() - t0)
        except Exception as e:
            out[i] = (0, 0, time.time() - t0)
            print("error:", r.status, data[:300], file=sys.stderr)

    t0 = time.time()
    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.time() - t0
    toks = sum(o[0] for o in out)
    print(f"N={n:2d} tokens={toks} wall={wall:.2f}s aggregate={toks / wall:.1f} tok/s per-request={toks / wall / n:.1f} tok/s")


def ttft(url, key, model, reps, pause=0.0):
    v = []
    for i in range(reps):
        time.sleep(pause)
        body = {"model": model, "max_tokens": 4, "temperature": 0, "stream": True,
                "messages": [{"role": "user", "content": f"Say the number {i}."}]}
        t0 = time.perf_counter()
        c, r = post(url, key, body, True)
        first = None
        while True:
            line = r.readline()
            if not line:
                break
            if line.startswith(b"data: ") and b'"content"' in line or b'"reasoning_content"' in line:
                first = time.perf_counter() - t0
                break
        r.read()
        c.close()
        if first is not None:
            v.append(first * 1e3)
    v.sort()
    print(f"ttft {url} n={len(v)} min={v[0]:.1f}ms p50={statistics.median(v):.1f}ms p90={v[int(len(v) * 0.9)]:.1f}ms mean={statistics.mean(v):.1f}ms")


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "tput":
        tput(a[2], a[3], a[4], int(a[5]), int(a[6]) if len(a) > 6 else 128, a[7] if len(a) > 7 else "0")
    else:
        ttft(a[2], a[3], a[4], int(a[5]), float(a[6]) if len(a) > 6 else 0.0)
