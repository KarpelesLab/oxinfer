#!/usr/bin/env python3
"""LFM3-Omni reference (LiquidAI/LFM3-Omni-2B): runs the checkpoint's own processor and model
(trust_remote_code, float32, CPU) on the same inputs as oxinfer and compares with its dumps.

usage: ref_lfm3_omni.py MODEL_DIR [--audio F.wav]... [--image F]... [--prompt TEXT] [--system TEXT]
           [--ours PREFIX] [--ours-logits FILE] [--ours-prompt FILE] [--generate N]
           [--speech audio|interleaved --ours-codes FILE --ours-pcm FILE --dump-codes FILE --dump-wav FILE]
           [--sample]

Audio files must be 16 kHz mono WAV (both sides then read the same samples).
--ours PREFIX compares oxinfer's --dump-mm files: PREFIX_mel / _audio_features / _audio (log-mel
  features, encoder output, projected tokens) and PREFIX_pixels / _features / _image.
--ours-logits compares the logits of the prompt's last token (oxinfer --dump-logits; the tokens
  a text reply never holds are -inf there and skipped).
--ours-prompt compares the token ids of oxinfer's expanded prompt (--print-prompt).
--generate N greedy text tokens (generation_mode="text").
--speech generates a spoken reply with the most likely codes, no repetition penalty, and the
  codebooks revealed in index order (torch.randperm replaced), which is what oxinfer's
  --speak-json '{"fixed_order": true}' --temp 0 --repetition-penalty 1 does; --ours-codes (JSON
  rows of 12) and --ours-pcm (f32 samples) compare the codes and the codec's waveform.
  With --sample it uses the checkpoint's sampling settings instead (no comparison).
"""
import argparse, json, sys
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--audio", action="append", default=[])
ap.add_argument("--image", action="append", default=[])
ap.add_argument("--prompt", default="")
ap.add_argument("--system")
ap.add_argument("--ours")
ap.add_argument("--ours-logits")
ap.add_argument("--ours-prompt")
ap.add_argument("--generate", type=int, default=0)
ap.add_argument("--speech")
ap.add_argument("--max-new", type=int, default=256)
ap.add_argument("--ours-codes")
ap.add_argument("--ours-pcm")
ap.add_argument("--dump-codes")
ap.add_argument("--dump-wav")
ap.add_argument("--sample", action="store_true")
a = ap.parse_args()

import soundfile as sf
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

torch.manual_seed(0)
model = AutoModelForMultimodalLM.from_pretrained(a.model, dtype=torch.float32, trust_remote_code=True).eval()
proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
cfg = model.config


def stats(name, ref, ours):
    ref, ours = np.asarray(ref, dtype=np.float32).ravel(), np.asarray(ours, dtype=np.float32).ravel()
    if ref.shape != ours.shape:
        print(f"{name}: SHAPE MISMATCH ref {ref.shape} ours {ours.shape}")
        return
    d = np.abs(ref - ours)
    rel = float(np.linalg.norm(ref - ours) / max(np.linalg.norm(ref), 1e-30))
    print(f"{name}: {ref.size} values, max abs diff {d.max():.3e}, rel {rel:.3e} (ref max abs {np.abs(ref).max():.3f})")


def ours(suffix):
    return np.fromfile(f"{a.ours}_{suffix}.bin", dtype=np.float32)


content, call = [], {}
if a.image:
    content += [{"type": "image"} for _ in a.image]
    call["images"] = [Image.open(p).convert("RGB") for p in a.image]
clips = []
for p in a.audio:
    pcm, sr = sf.read(p, dtype="float32")
    assert sr == 16000 and pcm.ndim == 1, "16 kHz mono WAV expected"
    clips.append(pcm)
    content.append({"type": "audio"})
if clips:
    call["audio"] = clips
    call["sampling_rate"] = 16000
if a.prompt:
    content.append({"type": "text", "text": a.prompt})
conv = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": content}]
text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
print("template:", repr(text))
inputs = proc(text=text, return_tensors="pt", **call)
ids = inputs["input_ids"]
print("prompt tokens:", ids.shape[1])

if a.ours_prompt:
    mine = proc.tokenizer(open(a.ours_prompt).read(), add_special_tokens=False)["input_ids"]
    same = mine == ids[0].tolist()
    print("prompt ids identical:", same)
    if not same:
        print(" ref :", proc.tokenizer.decode(ids[0])[:600])
        print(" ours:", proc.tokenizer.decode(mine)[:600])

with torch.no_grad():
    if a.ours and clips:
        feats, mask = inputs["input_features"], inputs["feature_attention_mask"]
        stats("mel features", feats[0].T, ours("mel"))
        enc, _ = model.model.audio_encoder(feats, mask.sum(-1))
        stats("audio encoder output", enc[0], ours("audio_features"))
        stats("audio tokens", model.get_audio_features(feats[:1], mask[:1]), ours("audio"))
    if a.ours and a.image:
        px, grid = inputs["pixel_values"], inputs["image_grid_thw"]
        n = int(grid[0].prod())
        mine = ours("pixels")
        print("pixels identical:", bool(np.array_equal(px[:n].numpy().ravel(), mine)), "grid", grid[0].tolist())
        tower = model.model.vision_tower(px[:n], grid_thw=grid[:1], return_dict=True).last_hidden_state
        stats("vision tower output", tower, ours("features"))
        stats("image tokens", model.get_image_features(px[:n], grid[:1]), ours("image"))

    if a.ours_logits:
        logits = model(**inputs).logits[0, -1].numpy()
        mine = np.fromfile(a.ours_logits, dtype=np.float32)
        keep = np.isfinite(mine)
        print("suppressed ids on our side:", np.nonzero(~keep)[0].tolist())
        d = np.abs(logits[keep] - mine[keep])
        print(f"last-token logits: max abs diff {d.max():.3e}; argmax ref {int(np.where(keep, logits, -np.inf).argmax())} "
              f"ours {int(mine.argmax())}; top5 ref {np.argsort(-np.where(keep, logits, -np.inf))[:5].tolist()} "
              f"ours {np.argsort(-mine)[:5].tolist()}")

    if a.generate:
        out = model.generate(**inputs, max_new_tokens=a.generate, generation_mode="text", do_sample=False)
        new = out.sequences[0, ids.shape[1]:out.sequence_lengths[0]].tolist()
        print("greedy ids:", new)
        print("greedy text:", repr(proc.tokenizer.decode(new, skip_special_tokens=True)))

    if a.speech:
        kw = {}
        if not a.sample:
            # Reveal the codebooks in index order: the one random choice left under greedy codes.
            torch.randperm = lambda n, **k: torch.arange(n, **{x: y for x, y in k.items() if x == "device"})
            kw = dict(audio_temperature=0.0, audio_repetition_penalty=1.0)
        out = model.generate(**inputs, max_new_tokens=a.max_new, generation_mode=a.speech, do_sample=False, **kw)
        n = int(out.audio_lengths[0])
        codes = out.audio_codes[0, :, :n]
        new = out.sequences[0, ids.shape[1]:out.sequence_lengths[0]].tolist()
        print("speech frames:", n, "| slots:", len(new))
        print("text:", repr(proc.tokenizer.decode([t for t in new if t != cfg.audio_out_token_id], skip_special_tokens=True)))
        rows = codes.T.tolist()
        if a.dump_codes:
            json.dump(rows, open(a.dump_codes, "w"))
        if a.ours_codes:
            mine = json.load(open(a.ours_codes))
            same = mine == rows
            first = next((i for i, (x, y) in enumerate(zip(mine, rows)) if x != y), min(len(mine), len(rows)))
            print(f"codes identical: {same} (ours {len(mine)} frames, ref {len(rows)}; first difference at frame {first})")
        codec = proc.audio_tokenizer.float().eval()
        wav = codec.decode(codes[None]).audio_values[0].float().numpy().ravel()
        if a.dump_wav:
            sf.write(a.dump_wav, wav, 24000)
        if a.ours_pcm:
            mine = np.fromfile(a.ours_pcm, dtype=np.float32)
            if mine.shape == wav.shape:
                err = float(np.sum((wav - mine) ** 2))
                snr = 10 * np.log10(float(np.sum(wav ** 2)) / max(err, 1e-30))
                print(f"waveform: {wav.size} samples, max abs diff {np.abs(wav - mine).max():.3e}, SNR {snr:.1f} dB")
            else:
                print(f"waveform: SHAPE MISMATCH ref {wav.shape} ours {mine.shape}")
