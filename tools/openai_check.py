#!/usr/bin/env python3
"""Checks `oxinfer serve` with the official `openai` Python client (pip install openai).

usage: openai_check.py BASE_URL [--chat ID] [--vision ID] [--embed ID] [--image ID]
                       [--whisper ID] [--tts ID] [--api-key KEY]
e.g.   oxinfer serve --listen 127.0.0.1:8080 --model qwen=Qwen/Qwen3-0.6B \
           --model vl=Qwen/Qwen3.5-0.8B --model bge=BAAI/bge-small-en-v1.5 \
           --model sd=stabilityai/sd-turbo --model whisper=openai/whisper-tiny \
           --model tts=isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF:Q4_K_M
       openai_check.py http://127.0.0.1:8080/v1 --chat qwen --vision vl --embed bge \
           --image sd --whisper whisper --tts tts
Each check prints a line; any failure raises.
"""
import argparse, base64, io, json, struct, wave

import openai

p = argparse.ArgumentParser()
p.add_argument("base_url")
for k in ("chat", "vision", "embed", "image", "whisper", "tts"):
    p.add_argument(f"--{k}")
p.add_argument("--api-key", default="none")
a = p.parse_args()
c = openai.OpenAI(base_url=a.base_url, api_key=a.api_key)


def ok(name, detail=""):
    print(f"ok  {name}  {detail}")


ids = [m.id for m in c.models.list().data]
ok("models.list", ids)

WEATHER = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]

if a.chat:
    m = a.chat
    r = c.chat.completions.create(model=m, temperature=0, max_tokens=30, messages=[
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "What is the capital of France?"}])
    t = r.choices[0].message.content
    assert "Paris" in t, t
    assert r.usage.prompt_tokens > 0 and r.usage.completion_tokens > 0
    ok("chat", repr(t))

    # Multi-turn: the second request extends the first one's conversation (cache reuse).
    first = [{"role": "user", "content": "My name is Ada. Reply with just OK."}]
    r1 = c.chat.completions.create(model=m, temperature=0, max_tokens=10, messages=first)
    convo = first + [{"role": "assistant", "content": r1.choices[0].message.content},
                     {"role": "user", "content": "What is my name?"}]
    r2 = c.chat.completions.create(model=m, temperature=0, max_tokens=20, messages=convo)
    cached = r2.usage.prompt_tokens_details.cached_tokens
    assert "Ada" in r2.choices[0].message.content, r2.choices[0].message.content
    ok("chat multi-turn", f"{r2.choices[0].message.content!r}, {cached} of {r2.usage.prompt_tokens} prompt tokens cached")

    # Streaming, with usage at the end.
    parts, finish, usage = [], None, None
    for ch in c.chat.completions.create(model=m, temperature=0, max_tokens=40, stream=True,
                                        stream_options={"include_usage": True},
                                        messages=[{"role": "user", "content": "Count from 1 to 5, comma separated."}]):
        if ch.usage:
            usage = ch.usage
        for choice in ch.choices:
            parts.append(choice.delta.content or "")
            finish = choice.finish_reason or finish
    text = "".join(parts)
    assert "1" in text and "5" in text and finish in ("stop", "length") and usage, (text, finish, usage)
    ok("chat stream", f"{len(parts)} chunks {text!r} finish={finish} usage={usage.completion_tokens}")

    # Stop sequences.
    r = c.chat.completions.create(model=m, temperature=0, max_tokens=40, stop=["3"],
                                  messages=[{"role": "user", "content": "Count from 1 to 5, comma separated."}])
    t = r.choices[0].message.content
    assert "3" not in t and "4" not in t and r.choices[0].finish_reason == "stop", t
    ok("chat stop", repr(t))

    # Tool calls, the result, then the answer.
    msgs = [{"role": "user", "content": "What is the weather in Paris?"}]
    r = c.chat.completions.create(model=m, temperature=0, max_tokens=200, tools=WEATHER, messages=msgs)
    msg = r.choices[0].message
    assert r.choices[0].finish_reason == "tool_calls" and msg.tool_calls, msg
    call = msg.tool_calls[0]
    assert call.function.name == "get_weather" and json.loads(call.function.arguments)["city"] == "Paris", call
    ok("tool call", f"{call.function.name}({call.function.arguments})")
    msgs += [msg.model_dump(exclude_none=True),
             {"role": "tool", "tool_call_id": call.id, "content": json.dumps({"temp_c": 18, "sky": "cloudy"})}]
    r = c.chat.completions.create(model=m, temperature=0, max_tokens=100, tools=WEATHER, messages=msgs)
    t = r.choices[0].message.content
    assert "18" in t, t
    ok("tool result", repr(t))

    # Streamed tool calls.
    calls = []
    for ch in c.chat.completions.create(model=m, temperature=0, max_tokens=200, tools=WEATHER, stream=True,
                                        messages=[{"role": "user", "content": "Weather in Tokyo?"}]):
        for choice in ch.choices:
            calls += choice.delta.tool_calls or []
    assert calls and calls[0].function.name == "get_weather", calls
    ok("tool call stream", calls[0].function.arguments)

    # Thinking (models that think): reasoning_content apart from the answer.
    r = c.chat.completions.create(model=m, temperature=0, max_tokens=2000, reasoning_effort="high",
                                  messages=[{"role": "user", "content": "What is 12 * 13?"}])
    msg = r.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None) or (msg.model_extra or {}).get("reasoning_content")
    assert "156" in (msg.content or "") and reasoning, msg
    ok("reasoning", f"{len(reasoning)} chars of reasoning, answer {msg.content.strip()[-40:]!r}")

    # Structured outputs: the reply is constrained to the schema while decoding.
    schema = {"type": "object",
              "properties": {"name": {"type": "string"}, "population": {"type": "integer"},
                             "continent": {"enum": ["Africa", "America", "Asia", "Europe", "Oceania"]},
                             "landmarks": {"type": "array", "items": {"$ref": "#/$defs/landmark"}}},
              "required": ["name", "population", "continent", "landmarks"], "additionalProperties": False,
              "$defs": {"landmark": {"type": "object", "properties": {"name": {"type": "string"}, "year": {"type": ["integer", "null"]}},
                                     "required": ["name", "year"], "additionalProperties": False}}}
    fmt = {"type": "json_schema", "json_schema": {"name": "city", "strict": True, "schema": schema}}
    def city(text):
        d = json.loads(text)
        assert list(d) == schema["required"] and isinstance(d["name"], str) and type(d["population"]) is int, d
        assert d["continent"] in schema["properties"]["continent"]["enum"], d
        for l in d["landmarks"]:
            assert list(l) == ["name", "year"] and isinstance(l["name"], str) and (l["year"] is None or type(l["year"]) is int), d
        return d
    n = 0
    for i, q in enumerate(["Tell me about Paris.", "Describe Tokyo.", "What is 2 + 2?", "Sing a song.", "Cairo?"]):
        for t in (0, 1):
            r = c.chat.completions.create(model=m, temperature=t, seed=i, max_tokens=400, response_format=fmt,
                                          messages=[{"role": "user", "content": q}])
            if r.choices[0].finish_reason != "length":
                city(r.choices[0].message.content)
                n += 1
    assert n >= 8, n
    ok("response_format json_schema", f"{n} of 10 replies complete, all valid")
    r = c.chat.completions.create(model=m, temperature=0, max_tokens=200, response_format={"type": "json_object"},
                                  messages=[{"role": "user", "content": "Say hello."}])
    assert isinstance(json.loads(r.choices[0].message.content), dict)
    ok("response_format json_object", r.choices[0].message.content)
    text = "".join(ch.choices[0].delta.content or "" for ch in c.chat.completions.create(
        model=m, temperature=0, max_tokens=400, response_format=fmt, stream=True,
        messages=[{"role": "user", "content": "Tell me about Rome."}]) if ch.choices)
    ok("response_format stream", city(text)["name"])
    try:
        bad = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "string", "pattern": "^a"}}}
        c.chat.completions.create(model=m, response_format=bad, messages=[{"role": "user", "content": "hi"}])
        raise AssertionError("expected an error")
    except openai.BadRequestError as e:
        ok("unsupported schema -> 400", e.body["message"] if isinstance(e.body, dict) else e.body)

    # Forced tool calls.
    for choice in ("required", {"type": "function", "function": {"name": "get_weather"}}):
        r = c.chat.completions.create(model=m, temperature=0, max_tokens=200, tools=WEATHER, tool_choice=choice,
                                      messages=[{"role": "user", "content": "Hello, how are you?"}])
        call = r.choices[0].message.tool_calls[0]
        args = json.loads(call.function.arguments)
        assert call.function.name == "get_weather" and list(args) == ["city"] and isinstance(args["city"], str), call
        ok(f"tool_choice {choice if isinstance(choice, str) else 'function'}", call.function.arguments)

    # Log probabilities.
    r = c.chat.completions.create(model=m, temperature=0, max_tokens=20, logprobs=True, top_logprobs=3,
                                  messages=[{"role": "user", "content": "Name three colors."}])
    lp = r.choices[0].logprobs.content
    assert len(lp) == r.usage.completion_tokens and all(len(e.top_logprobs) == 3 and e.logprob <= 0 for e in lp), lp
    assert bytes(b for e in lp for b in e.bytes).decode().strip() == r.choices[0].message.content.strip()
    ok("logprobs", f"{len(lp)} tokens, first {lp[0].token!r} {lp[0].logprob:.3f}")
    got = [e for ch in c.chat.completions.create(model=m, temperature=0, max_tokens=20, logprobs=True, top_logprobs=3, stream=True,
                                                 messages=[{"role": "user", "content": "Name three colors."}])
           for choice in ch.choices if choice.logprobs for e in choice.logprobs.content or []]
    # The same tokens. The values are equal only when the repeat reuses nothing of the first
    # request's cache (see the README): with a cached prefix the last prompt token runs alone.
    assert [(e.token, e.bytes) for e in got] == [(e.token, e.bytes) for e in lp], got
    worst = max(abs(x.logprob - y.logprob) for x, y in zip(got, lp))
    assert worst < 1.0, worst
    ok("logprobs stream", f"{len(got)} tokens, max difference to the first request {worst:.2e}")

    r = c.completions.create(model=m, prompt="The capital of France is", max_tokens=5, temperature=0, logprobs=2)
    assert len(r.choices[0].logprobs.tokens) == 5 and "".join(r.choices[0].logprobs.tokens) == r.choices[0].text
    ok("completions", repr(r.choices[0].text))

    try:
        c.chat.completions.create(model="no-such-model", messages=[{"role": "user", "content": "hi"}])
        raise AssertionError("expected an error")
    except openai.NotFoundError as e:
        ok("unknown model -> 404", e.body["code"] if isinstance(e.body, dict) else e.body)
    try:
        c.chat.completions.create(model=m, temperature=5, messages=[{"role": "user", "content": "hi"}])
        raise AssertionError("expected an error")
    except openai.BadRequestError as e:
        ok("bad parameter -> 400", e.body["param"] if isinstance(e.body, dict) else e.body)

if a.vision:
    # A 64x64 PNG, red left half and blue right half, as a data URL.
    def png(w, h, px):
        import zlib
        raw = b"".join(b"\0" + bytes(c for x in range(w) for c in px(x, y)) for y in range(h))
        chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    img = png(64, 64, lambda x, y: (220, 20, 20) if x < 32 else (20, 20, 220))
    url = "data:image/png;base64," + base64.b64encode(img).decode()
    r = c.chat.completions.create(model=a.vision, temperature=0, max_tokens=30, messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "What two colors does this image have? Answer with two words."}]}])
    t = r.choices[0].message.content.lower()
    assert "red" in t and "blue" in t, t
    ok("vision", repr(t))

if a.embed:
    r = c.embeddings.create(model=a.embed, input=["The cat sat on the mat.", "A cat is sitting on a rug.", "Stock prices fell."])
    v = [d.embedding for d in r.data]
    cos = lambda x, y: sum(p * q for p, q in zip(x, y)) / (sum(p * p for p in x) * sum(q * q for q in y)) ** 0.5
    assert cos(v[0], v[1]) > cos(v[0], v[2]), (cos(v[0], v[1]), cos(v[0], v[2]))
    ok("embeddings", f"dim {len(v[0])}, cos similar {cos(v[0], v[1]):.3f} vs unrelated {cos(v[0], v[2]):.3f}")
    r = c.embeddings.create(model=a.embed, input="hello", dimensions=64, encoding_format="float")
    assert len(r.data[0].embedding) == 64
    ok("embeddings dimensions", 64)

if a.image:
    r = c.images.generate(model=a.image, prompt="a red apple on a wooden table", size="512x512", n=1,
                          response_format="b64_json", extra_body={"seed": 3})
    first = base64.b64decode(r.data[0].b64_json)
    assert first.startswith(b"\x89PNG")
    ok("images.generate", f"{len(first)} bytes of PNG")
    r = c.images.edit(model=a.image, image=("apple.png", first, "image/png"), prompt="a green apple on a wooden table",
                      extra_body={"strength": 0.6, "steps": 2})
    assert base64.b64decode(r.data[0].b64_json).startswith(b"\x89PNG")
    ok("images.edit")

speech = None
if a.tts:
    r = c.audio.speech.create(model=a.tts, voice="tara", input="Hello there, this is a test of the speech server.",
                              response_format="wav", extra_body={"seed": 1})
    speech = r.content
    with wave.open(io.BytesIO(speech)) as w:
        secs = w.getnframes() / w.getframerate()
    assert secs > 1, secs
    ok("audio.speech", f"{secs:.2f}s of WAV")
    with c.audio.speech.with_streaming_response.create(model=a.tts, voice="leo", input="Streaming works.",
                                                       response_format="pcm", extra_body={"seed": 2}) as resp:
        chunks = [b for b in resp.iter_bytes()]
    ok("audio.speech pcm stream", f"{len(chunks)} chunks, {sum(map(len, chunks)) / 48000:.2f}s")

if a.whisper:
    if speech is None:
        raise SystemExit("--whisper needs --tts (it transcribes the generated speech)")
    t = c.audio.transcriptions.create(model=a.whisper, file=("speech.wav", speech, "audio/wav"))
    assert "test" in t.text.lower(), t.text
    ok("audio.transcriptions", repr(t.text))
    t = c.audio.transcriptions.create(model=a.whisper, file=("speech.wav", speech, "audio/wav"),
                                      response_format="verbose_json", timestamp_granularities=["word", "segment"])
    ok("audio.transcriptions verbose_json", f"{len(t.segments)} segments, {len(t.words or [])} words")
    t = c.audio.transcriptions.create(model=a.whisper, file=("speech.wav", speech, "audio/wav"), response_format="srt")
    assert "-->" in t, t
    ok("audio.transcriptions srt")

print("all checks passed")
