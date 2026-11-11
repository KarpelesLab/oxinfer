#!/usr/bin/env python3
"""Reference check of oxinfer's Whisper beam search against openai-whisper (its BeamSearchDecoder
and MaximumLikelihoodRanker, float32 on the CPU), on the same 16 kHz WAV clips:

  cargo run --release --example transcribe -- openai/whisper-tiny a.wav --also b.wav --also c.wav \
      --precision f32 --device cpu --language en --beam 5 --json > ours.json
  python tools/ref_whisper_beam.py tiny ours.json a.wav b.wav c.wav --beam 5

(pip install openai-whisper soundfile; the first argument is an openai-whisper model name or a
.pt file, the same weights as the Hugging Face checkpoint oxinfer loads). Each clip of up to 30 s
is decoded with whisper.decode on its log-mel features; --long runs whisper's transcribe()
instead (sequential 30 s windows, no conditioning on the previous text; text tokens and segment
times are compared). Pass the same options to
both sides: --beam, --patience, --length-penalty, --timestamps, --language. Prints, per clip,
whether the tokens are identical (timestamp tokens included with --timestamps) and how far the
average log-probability and the no-speech probability are from the reference; exits 1 if a clip
differs. A window that ends on a timestamp pair before the end of the clip is compared alone
(oxinfer decodes the rest in a second window). ours.json holds one transcript or an array of them (--also), in the clips' order.
"""
import sys, json, argparse
import numpy as np
import soundfile as sf
import torch
import whisper

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("ours")
ap.add_argument("clips", nargs="+")
ap.add_argument("--beam", type=int, default=5)
ap.add_argument("--patience", type=float)
ap.add_argument("--length-penalty", type=float)
ap.add_argument("--timestamps", action="store_true")
ap.add_argument("--language", default="en")
ap.add_argument("--long", action="store_true")
ap.add_argument("--download-root")
a = ap.parse_args()

torch.set_num_threads(8)
model = whisper.load_model(a.model, device="cpu", download_root=a.download_root).eval()
ours = json.load(open(a.ours))
if isinstance(ours, dict):
    ours = [ours]
assert len(ours) == len(a.clips), f"{len(ours)} transcripts for {len(a.clips)} clips"
tok = whisper.tokenizer.get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language=a.language, task="transcribe")
language = a.language if model.is_multilingual else None

bad = 0
total = 0
worst_lp = 0.0
worst_ns = 0.0
for path, o in zip(a.clips, ours):
    pcm, sr = sf.read(path, dtype="float32")
    assert sr == 16000 and pcm.ndim == 1, f"{path}: expected 16 kHz mono"
    segs = o["segments"]
    our_tok = [t for s in segs for t in s["tokens"]]
    if a.long:
        r = model.transcribe(
            pcm, language=language, beam_size=a.beam, patience=a.patience, length_penalty=a.length_penalty,
            temperature=0.0, condition_on_previous_text=False, fp16=False, without_timestamps=False,
        )
        # Text tokens and segment times: the two keep a window's closing timestamps differently.
        ref_tok = [t for s in r["segments"] for t in s["tokens"] if t < tok.eot]
        our_tok = [t for t in our_tok if t < tok.eot]
        ref_times = [(round(s["start"], 2), round(s["end"], 2)) for s in r["segments"]]
        our_times = [(round(s["start"], 2), round(s["end"], 2)) for s in segs]
        times = f"; segment times {'identical' if ref_times == our_times else 'DIFFERENT'} ({len(ref_times)} vs {len(our_times)} segments)"
        lp = ns = None
    else:
        assert len(pcm) <= 480000, f"{path}: over 30 s, use --long"
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(pcm), n_mels=model.dims.n_mels)
        r = whisper.decode(model, mel, whisper.DecodingOptions(
            language=language, beam_size=a.beam, patience=a.patience, length_penalty=a.length_penalty,
            without_timestamps=not a.timestamps, fp16=False,
        ))
        ref_tok = list(r.tokens)
        lp, ns = r.avg_logprob, r.no_speech_prob
    # A window that ends on a timestamp pair before the clip's end leaves the rest to a second
    # window in oxinfer (as transcribe() does); whisper.decode stops at the first.
    more = ""
    if not a.long and len(our_tok) > len(ref_tok) >= 2 and ref_tok[-1] == ref_tok[-2] >= tok.timestamp_begin:
        more = f" (first window; ours goes on with {len(our_tok) - len(ref_tok)} tokens of a second one)"
        our_tok = our_tok[:len(ref_tok)]
    same = ref_tok == our_tok
    first = next((i for i, (x, y) in enumerate(zip(ref_tok, our_tok)) if x != y), min(len(ref_tok), len(our_tok)))
    total += len(ref_tok)
    line = f"{path}: {len(pcm) / 16000:.1f}s, ref {len(ref_tok)} tokens, ours {len(our_tok)}: "
    line += "identical" + more if same else f"DIFFERENT from token {first}"
    if a.long:
        line += times
    if lp is not None and segs:
        d_lp = abs(segs[0]["avg_logprob"] - lp)
        d_ns = abs(segs[0]["no_speech_prob"] - ns)
        line += f"; avg_logprob {lp:.4f} (ours off by {d_lp:.1e}), no_speech_prob off by {d_ns:.1e}"
        if same:
            worst_lp, worst_ns = max(worst_lp, d_lp), max(worst_ns, d_ns)
    print(line)
    if not same:
        bad += 1
        print("  ref :", tok.decode_with_timestamps(ref_tok))
        print("  ours:", tok.decode_with_timestamps(our_tok))
n = len(a.clips)
print(f"beam {a.beam}, patience {a.patience}, length penalty {a.length_penalty}, timestamps {a.timestamps or a.long}: "
      f"{n - bad} of {n} clips identical ({total} reference tokens); max |avg_logprob diff| {worst_lp:.1e}, "
      f"max |no_speech_prob diff| {worst_ns:.1e} on the identical ones")
sys.exit(1 if bad else 0)
