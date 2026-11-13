#!/usr/bin/env python3
"""Reference outputs of an Audio8 TTS checkpoint (model_type arktts) from the Python code it
ships (processing_arktts.py, modeling_arktts.py, modeling_arktts_codec.py), in float32 on the CPU,
for examples/arktts_check.rs to compare with.

    ref_arktts.py MODEL_DIR OUT_DIR --text TEXT [--ref-audio CLIP.wav --ref-text TRANSCRIPT]
                  [--max-new-tokens N]
    ref_arktts.py MODEL_DIR OUT_DIR --ids TEXTS.txt

The second form writes OUT_DIR/ids.json: for every line of TEXTS.txt, the ids of the prompt that
speaks it and of the prompt half that holds it as a reference transcript.

Writes to OUT_DIR:
  manifest.json   the text, the ids of the prompt's two halves, counts and timings
  rope.f32        the slow transformer's RoPE table [max_seq_len, head_dim / 2, 2]
  ref_audio.f32   the reference clip as the processor loaded it (mono, 44.1 kHz)
  ref_codes.i32   its codes from the codec's encoder [frames, 10]
  prompt.i32      the prompt [rows, 11] (a token and ten codes per row)
  codes.i32       the greedy generation [frames, 10]
  slow.f32        per step, the logits of the 4096 semantic tokens and of the end token
  fast.f32        per frame, the logits of codebooks 1..9 [9, 4096]
  audio.f32       the codec's waveform for codes.i32

Needs torch, transformers (4.57), soundfile and numpy.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--text")
    ap.add_argument("--ids", help="a file of texts, one per line: write only ids.json")
    ap.add_argument("--ref-audio")
    ap.add_argument("--ref-text")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    if a.ids:
        rows = []
        for line in Path(a.ids).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            full, _ = processor._prompt_segments(line, None, False)
            prefix, _ = processor._prompt_segments("x", line, True)
            rows.append({"text": line, "ids": full.tolist(), "reference_prefix": prefix.tolist()})
        (out / "ids.json").write_text(json.dumps(rows, ensure_ascii=False))
        print(f"{len(rows)} texts")
        return
    if not a.text:
        ap.error("--text or --ids")
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True, dtype=torch.float32).eval()
    cfg = model.config
    model.freqs_cis.float().numpy().astype("<f4").tofile(out / "rope.f32")

    kwargs = {}
    if a.ref_audio:
        kwargs = dict(reference_audio=[a.ref_audio], reference_text=[a.ref_text])
    inputs = dict(processor(text=[a.text], return_tensors="pt", **kwargs))
    manifest = {
        "text": a.text,
        "ref_text": a.ref_text,
        "prefix_ids": inputs["prefix_input_ids"][0].tolist(),
        "suffix_ids": inputs["suffix_input_ids"][0].tolist(),
        "sample_rate": cfg.codec_sample_rate,
        "torch": torch.__version__,
        "threads": a.threads,
    }
    if a.ref_audio:
        values = inputs.pop("reference_audio_values")
        lengths = inputs.pop("reference_audio_lengths")
        values[0, 0, : int(lengths[0])].numpy().astype("<f4").tofile(out / "ref_audio.f32")
        t0 = time.time()
        with torch.inference_mode():
            codes, code_lengths = model.encode_audio(values, lengths)
        manifest["encode_secs"] = time.time() - t0
        n = int(code_lengths[0])
        codes[0, :, :n].T.contiguous().numpy().astype("<i4").tofile(out / "ref_codes.i32")
        manifest["ref_frames"] = n
        inputs["reference_codes"] = codes
        inputs["reference_code_lengths"] = code_lengths

    with torch.inference_mode():
        prompt, _ = model._prepare_prompt(**inputs)
    prompt[0].T.contiguous().numpy().astype("<i4").tofile(out / "prompt.i32")
    manifest["prompt_rows"] = int(prompt.shape[-1])

    slow, fast = [], []
    slow_step, fast_step = model._slow_step, model._fast_step
    b, e, eos = cfg.semantic_begin_id, cfg.semantic_end_id, cfg.eos_token_id

    def rec_slow(*args, **kw):
        logits, hidden = slow_step(*args, **kw)
        slow.append(torch.cat((logits[0, b : e + 1], logits[0, eos : eos + 1])).float().numpy().copy())
        return logits, hidden

    def rec_fast(hidden, position):
        scores = fast_step(hidden, position)
        if position > 0:
            fast.append(scores[0].float().numpy().copy())
        return scores

    model._slow_step, model._fast_step = rec_slow, rec_fast
    t0 = time.time()
    with torch.inference_mode():
        res = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False, return_dict_in_generate=True)
    manifest["generate_secs"] = time.time() - t0
    model._slow_step, model._fast_step = slow_step, fast_step
    frames = int(res.code_lengths[0])
    codes = res.codes[0, :, :frames]
    manifest["frames"] = frames
    manifest["finished"] = bool(res.finished[0])
    codes.T.contiguous().numpy().astype("<i4").tofile(out / "codes.i32")
    # One slow step per frame, and one more when the end token stopped the speech.
    np.stack(slow).astype("<f4").tofile(out / "slow.f32")
    manifest["slow_steps"] = len(slow)
    # The reference runs the fast transformer on the stopping step too; those logits are dropped.
    np.stack(fast)[: frames * (cfg.num_codebooks - 1)].astype("<f4").tofile(out / "fast.f32")

    t0 = time.time()
    with torch.inference_mode():
        wave, lengths = model.decode_audio(res.codes[:, :, :frames])
    manifest["decode_secs"] = time.time() - t0
    wave[0, : int(lengths[0])].numpy().astype("<f4").tofile(out / "audio.f32")
    manifest["samples"] = int(lengths[0])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("_ids")}))


if __name__ == "__main__":
    main()
