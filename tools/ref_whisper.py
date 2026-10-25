#!/usr/bin/env python3
"""Reference check of oxinfer's Whisper against transformers (WhisperForConditionalGeneration +
WhisperProcessor, float32), on the same 16 kHz samples:

  cargo run --release --example transcribe -- openai/whisper-small clip.flac \
      --precision f32 --device cpu --json --dump OUT > OUT/ours.json
  python tools/ref_whisper.py ~/.cache/oxinfer/hub/openai--whisper-small OUT --ours OUT/ours.json

OUT holds pcm.f32 (the decoded samples), mel.f32 (log-mel features), enc.f32 (first window's
encoder output) and logits.f32 (first-step logits after <|startoftranscript|><|en|><|transcribe|>
<|notimestamps|>). Prints the max differences of each, then runs transformers' generate (long-form
past 30 s) and compares the transcripts token for token, timestamps and segment times included.
Pass the same decoding options to both sides: --language, --translate, --timestamps, --prompt,
--condition (condition_on_prev_tokens), --fallback (temperature fallback, thresholds 2.4 / -1 /
0.6).
"""
import sys, json, argparse, time
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("dump")
ap.add_argument("--language")
ap.add_argument("--timestamps", action="store_true")
ap.add_argument("--translate", action="store_true")
ap.add_argument("--ours")
ap.add_argument("--first-lang", default="en")
ap.add_argument("--skip-gen", action="store_true")
ap.add_argument("--prompt")
ap.add_argument("--condition", action="store_true")
ap.add_argument("--fallback", action="store_true")
a = ap.parse_args()

torch.set_num_threads(8)
pcm = np.fromfile(f"{a.dump}/pcm.f32", dtype=np.float32)
proc = WhisperProcessor.from_pretrained(a.model)
model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float32).eval()
n_mels = model.config.num_mel_bins
long = len(pcm) > 480000
if long:
    inp = proc(pcm, sampling_rate=16000, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True)
else:
    inp = proc(pcm, sampling_rate=16000, return_tensors="pt")
feats = inp.input_features  # [1, mels, frames]
ours_mel = np.fromfile(f"{a.dump}/mel.f32", dtype=np.float32).reshape(-1, n_mels).T
ref_mel = feats[0].numpy()
print(f"mel: ours {ours_mel.shape} ref {ref_mel.shape}  max|diff| {np.abs(ours_mel - ref_mel).max():.3e}  (ref range {ref_mel.min():.3f}..{ref_mel.max():.3f})")

with torch.no_grad():
    win = feats[:, :, :3000]
    enc = model.model.encoder(win).last_hidden_state[0].numpy()
    ours_enc = np.fromfile(f"{a.dump}/enc.f32", dtype=np.float32).reshape(enc.shape)
    print(f"encoder: max|diff| {np.abs(ours_enc - enc).max():.3e}  max|ref| {np.abs(enc).max():.3f}  rel L2 {np.linalg.norm(ours_enc - enc) / np.linalg.norm(enc):.3e}")
    # Encoder from our features, to separate the front end from the network.
    enc2 = model.model.encoder(torch.from_numpy(ours_mel[None, :, :3000].copy())).last_hidden_state[0].numpy()
    print(f"encoder (our mel as input): max|diff| {np.abs(ours_enc - enc2).max():.3e}")
    gc = model.generation_config
    ids = [gc.decoder_start_token_id]
    if getattr(gc, "is_multilingual", True) and hasattr(gc, "lang_to_id"):
        ids += [gc.lang_to_id[f"<|{a.first_lang}|>"], gc.task_to_id["transcribe"]]
    ids.append(gc.no_timestamps_token_id)
    out = model(encoder_outputs=(torch.from_numpy(enc[None]),), decoder_input_ids=torch.tensor([ids]))
    lg = out.logits[0, -1].numpy()
    ours_lg = np.fromfile(f"{a.dump}/logits.f32", dtype=np.float32)
    print(f"first-step logits: max|diff| {np.abs(ours_lg - lg).max():.3e}  max|ref| {np.abs(lg).max():.2f}  argmax ours {ours_lg.argmax()} ref {lg.argmax()}  top5 ref {np.argsort(-lg)[:5].tolist()} ours {np.argsort(-ours_lg)[:5].tolist()}")

if a.skip_gen:
    sys.exit(0)
kw = dict(return_timestamps=a.timestamps or long, return_segments=True)
if a.language:
    kw["language"] = a.language
if a.translate:
    kw["task"] = "translate"
if long:
    kw["attention_mask"] = inp.attention_mask
if a.prompt:
    kw["prompt_ids"] = proc.get_prompt_ids(a.prompt, return_tensors="pt")
if a.condition:
    kw["condition_on_prev_tokens"] = True
if a.fallback:
    kw.update(temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=2.4, logprob_threshold=-1.0, no_speech_threshold=0.6)
t0 = time.time()
with torch.no_grad():
    g = model.generate(feats, **kw)
dt = time.time() - t0
seq = g["sequences"][0].tolist() if isinstance(g, dict) else g[0].tolist()
segs = g.get("segments", [[]])[0] if isinstance(g, dict) else []
text = proc.batch_decode([seq], skip_special_tokens=True)[0]
print(f"transformers generate: {dt:.2f}s")
print("ref text:", text)
ref_segs = [{"start": float(s["start"]), "end": float(s["end"]), "tokens": [int(t) for t in s["tokens"]]} for s in segs]
for s in ref_segs:
    print(f"  ref [{s['start']:7.2f} -> {s['end']:7.2f}] {proc.tokenizer.decode(s['tokens'], skip_special_tokens=True)}")
if a.ours:
    ours = json.load(open(a.ours))
    eot = gc.eos_token_id
    ref_tok = [t for s in ref_segs for t in s["tokens"]] if ref_segs else [t for t in seq if t < eot]
    our_tok = [t for s in ours["segments"] for t in s["tokens"]]
    # Text tokens only, and with timestamps.
    rt = [t for t in ref_tok if t < eot]
    ot = [t for t in our_tok if t < eot]
    same = sum(1 for x, y in zip(rt, ot) if x == y)
    first = next((i for i, (x, y) in enumerate(zip(rt, ot)) if x != y), None)
    print(f"tokens: ref {len(rt)} ours {len(ot)} text tokens; identical: {rt == ot}; first difference at {first}")
    if ref_segs:
        print(f"all tokens incl. timestamps identical: {ref_tok == our_tok}  (ref {len(ref_tok)}, ours {len(our_tok)})")
        rs = [(round(s['start'], 2), round(s['end'], 2)) for s in ref_segs]
        os_ = [(round(s['start'], 2), round(s['end'], 2)) for s in ours['segments']]
        print(f"segment times identical: {rs == os_}  ({len(rs)} vs {len(os_)} segments)")
    print("ours text:", ours["text"])
