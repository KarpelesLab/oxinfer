#!/usr/bin/env python3
"""Compares oxinfer's ZONOS3-Mini with Zyphra's PyTorch reference (github.com/Zyphra/ZONOS3-Mini).

REF is a checkout of that repository; HF a folder with the files of huggingface.co/Zyphra/ZONOS3-Mini
at their paths (lm/zonos3_mini/{params.json,model.pth,tokenizer/}, autoencoder/generator.pt,
autoencoder/heads/<head>/, gguf/). The reference needs torch, torchaudio, tokenizers, gguf,
descript-audio-codec and zyphra-textnorm.

  zonos_ref.py lm REF HF OURS [--gguf HF/gguf/backbone-ptq.gguf] [--codes VOICE.codes.npy] [--norm]
      OURS.json / OURS.logits come from `cargo run --release --example zonos_check -- gen ...`.
      Checks the prompt rows (tokenizer, speaker and condition rows, silence prefill) against
      zonos3.tts.plan, then generates greedily with zonos3.reference (float32, CPU) and compares the
      rows and every step's logits. --gguf loads the weights of that GGUF (dequantized) into the
      reference instead of model.pth, so both sides compute on the same numbers.

  zonos_ref.py codec REF HF CODES.npy OURS.f32 [--head keep2_plain|full_mr] [--f16] [--wav REF.wav]
      Decodes the frames with ISTFTCodec (generator.pt + the head checkpoint) and compares the
      waveform with oxinfer's (`zonos_check decode`). --f16 rounds the reference's matrices to
      half precision first, as the GGUF stores them.

  zonos_ref.py ids REF HF TEXTS.txt OURS.json
      Token ids of every line (`zonos_check ids`) against the tokenizers library.

  zonos_ref.py embed REF HF CLIP OURS.f32 [--npy VOICE.embedding.npy]
      The speaker embedding of a clip with MECTFbankSpeakerEmbedding (speaker_encoders/ownmect_s2)
      against oxinfer's (`zonos_check embed`), and against the embedding the repository ships.
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch


def load_gguf_weights(lm, path):
    """The reference LM's parameters from a zonos3 GGUF (tools/convert_gguf.py LM_NAMES, inverted)."""
    from convert_gguf import lm_entry
    from gguf import GGUFReader
    from gguf.quants import dequantize

    tensors = {}
    for t in GGUFReader(path).tensors:
        shape = [int(v) for v in reversed(t.shape.tolist())]
        tensors[t.name] = torch.from_numpy(np.array(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(shape))
    state = {}
    for name, param in lm.state_dict().items():
        _, _, _, names, transform = lm_entry(name, lm.config.n_codebooks)
        parts = [tensors[n] for n in names]
        kind = getattr(transform, "__name__", None)
        if kind is None:
            value = parts[0]
        elif kind == "_unbind0":
            value = torch.stack(parts, 0)
        elif kind == "_halves1":
            value = torch.cat(parts, 1)
        elif kind == "_squeeze1":
            value = parts[0].unsqueeze(1)
        elif kind == "_abs_flat":
            value = parts[0].reshape(param.shape)
        else:
            raise SystemExit(f"unknown transform {kind}")
        assert tuple(value.shape) == tuple(param.shape), (name, value.shape, param.shape)
        state[name] = value
    lm.load_state_dict(state, strict=True)
    lm.cast(torch.float32)


def cmd_lm(a):
    from zonos3.reference import synth
    from zonos3.reference.lm import Zonos3LM

    ours = json.loads(Path(a.ours + ".json").read_text())
    n_cb = ours["n_codebooks"]
    w = n_cb + 1
    lm = Zonos3LM.load(str(Path(a.hf) / "lm/zonos3_mini"))
    if a.gguf:
        load_gguf_weights(lm, a.gguf)
    c = lm.config
    ours_rows = np.array(ours["rows"], dtype=np.int64).reshape(-1, n_cb)
    request = {
        "text": ours["text"],
        "text_normalization": bool(a.norm),
        "temperature": 0.0,
        "seed": 0,
        "stream": False,
        "max_tokens": len(ours_rows),
        "silence_prefill_frames": ours["prefill_rows"],
    }
    if ours["speaker"] is not None:
        buf = io.BytesIO()
        np.save(buf, np.array(ours["speaker"], dtype=np.float32))
        request |= {"speaker_mode": "embedding", "speaker_embedding_base64": base64.b64encode(buf.getvalue()).decode()}
    elif ours["prefix_rows"]:
        request |= {"speaker_mode": "reference", "speaker_reference_codes": np.load(a.codes).tolist()}
    if ours["twin"] is not None:
        request["speaker_cfg_scale"] = a.cfg
    plan = synth.plan_prompt(lm, request, assets=synth.ReferenceAssets())
    ok = True

    def same(what, mine, theirs):
        nonlocal ok
        good = mine.shape == theirs.shape and bool((mine == theirs).all())
        ok &= good
        print(f"{what}: {'identical' if good else 'DIFFERENT'} ({theirs.shape[0]} rows)")
        if not good:
            print("  ours", mine.shape, "reference", theirs.shape)

    same("prompt rows", np.array(ours["prompt"], dtype=np.int64).reshape(-1, w), plan.cond.input_ids.numpy().astype(np.int64))
    if ours["twin"] is not None:
        same("twin rows", np.array(ours["twin"], dtype=np.int64).reshape(-1, w), plan.twin.input_ids.numpy().astype(np.int64))
    cond = plan.cond.condition_rows() or {}
    mine = {int(r): int(b) for r, b in ours["cond"]}
    print("condition rows:", "identical" if cond == mine else f"DIFFERENT ours {mine} reference {cond}")
    ok &= cond == mine
    print(f"normalized text: {plan.text!r}")

    logs = []
    lm.register_forward_hook(lambda m, i, out: logs.append(out.logits[-1].float().numpy().copy()))
    attempt = synth.generate_rows(
        lm,
        plan.cond,
        sampling=synth.Sampling(temperature=0.0),
        seed=0,
        max_tokens=plan.max_tokens,
        stop_id=c.eoa_id,
        twin=plan.twin,
        cfg_scale=plan.cfg_scale,
    )
    ref_rows = torch.stack(attempt.rows)[:, :n_cb].numpy().astype(np.int64)
    if plan.twin is not None:
        s = plan.cfg_scale
        logs = [u + s * (k - u) for k, u in zip(logs[0::2], logs[1::2])]
    mine = np.fromfile(a.ours + ".logits", dtype=np.float32).reshape(len(ours_rows), n_cb, -1)
    n = min(len(ref_rows), len(ours_rows))
    equal = (ref_rows[:n] == ours_rows[:n]).all(1)
    first = int(np.argmin(equal)) if not equal.all() else n
    print(f"generated rows: ours {len(ours_rows)}, reference {len(ref_rows)}; identical for the first {first} of {n}")
    print(f"end frame: ours {ours['eos_frame']}, reference {attempt.eos_frame}")
    steps = min(first + 1, n, len(logs))
    diff = [float(np.abs(mine[i] - logs[i]).max()) for i in range(steps)]
    top = sum(int((mine[i].argmax(-1) == logs[i].argmax(-1)).sum()) for i in range(steps))
    print(
        f"logits over the {steps} steps before any difference: max |diff| {max(diff):.3e}, "
        f"mean of the per-step max {np.mean(diff):.3e}; raw top-1 agreement {top}/{steps * n_cb}"
    )
    codes = int((ref_rows[:n] == ours_rows[:n]).sum())
    print(f"codes equal: {codes}/{n * n_cb}")
    return 0 if ok and first == n and len(ref_rows) == len(ours_rows) else 1


def cmd_codec(a):
    from zonos3.codec.istft_head_a_20261002.model import ISTFTCodec

    hf = Path(a.hf)
    head = hf / "autoencoder/heads" / a.head
    (cfg_path,) = head.glob("config_*.json")
    cfg = json.loads(cfg_path.read_text())
    generator = torch.load(hf / "autoencoder/generator.pt", map_location="cpu", mmap=True, weights_only=True)
    model = ISTFTCodec(cfg["model"], cfg["head"])
    model.load_pretrained(generator["generator"])
    state = torch.load(head / "step_0050000.pt", map_location="cpu", mmap=True, weights_only=False)
    model.load_trainable_state_dict(state["trainable"])
    model.eval().requires_grad_(False)
    if a.f16:
        # The GGUF stores the transformer, ConvNeXt and head matrices as F16.
        for name, param in model.named_parameters():
            if param.ndim == 2 and name.startswith(("stages.", "head.", "semantic_head.")):
                param.copy_(param.half().float())
    codes = torch.from_numpy(np.load(a.codes).astype(np.int64))
    if a.frames:
        codes = codes[: a.frames]
    with torch.inference_mode():
        audio = model(codes.T.unsqueeze(0)).float().reshape(-1).numpy()
    mine = np.fromfile(a.ours, dtype=np.float32)
    print(f"{len(codes)} frames; samples: ours {len(mine)}, reference {len(audio)}")
    if len(mine) != len(audio):
        return 1
    err = mine - audio
    snr = 10 * np.log10(float((audio.astype(np.float64) ** 2).sum()) / max(float((err.astype(np.float64) ** 2).sum()), 1e-30))
    print(f"peak of the reference {np.abs(audio).max():.4f}; max |diff| {np.abs(err).max():.3e}; SNR {snr:.1f} dB")
    if a.wav:
        import soundfile

        soundfile.write(a.wav, audio, 44100)
    return 0


def cmd_embed(a):
    import soundfile
    from zonos3.models.mect_speaker import MECTFbankSpeakerEmbedding

    entry = json.loads((Path(a.ref) / "model/src/zonos3/registries/speaker_encoder_registry.json").read_text())["ownmect_s2"]
    entry = dict(entry, codec="ownmect_s2", checkpoint=str(Path(a.hf) / "speaker_encoders/ownmect_s2/ckpt_0037.pth"))
    model = MECTFbankSpeakerEmbedding(entry, device="cpu")
    wav, rate = soundfile.read(a.clip, dtype="float32", always_2d=True)
    ref = model(torch.from_numpy(wav.T.copy()), rate).reshape(-1).numpy()
    mine = np.fromfile(a.ours, dtype=np.float32)

    def cos(x, y):
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    print(f"{wav.shape[0] / rate:.2f} s at {rate} Hz; norm of the reference {np.linalg.norm(ref):.3f}")
    print(f"ours vs reference: max |diff| {np.abs(mine - ref).max():.3e}, cosine {cos(mine, ref):.7f}")
    if a.npy:
        shipped = np.load(a.npy).reshape(-1)
        print(f"ours vs the shipped embedding: max |diff| {np.abs(mine - shipped).max():.3e}, cosine {cos(mine, shipped):.7f}")
        print(f"reference vs the shipped embedding: cosine {cos(ref, shipped):.7f}")
    return 0


def cmd_ids(a):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(a.hf) / "lm/zonos3_mini/tokenizer/superbpe8192.json"))
    ours = json.loads(Path(a.ours).read_text())
    lines = [line for line in Path(a.texts).read_text().split("\n") if line.strip()]
    bad = 0
    for line, mine in zip(lines, ours, strict=True):
        ref = [2] + [i + 4 for i in tok.encode(line, add_special_tokens=False).ids] + [3]
        if ref != mine:
            bad += 1
            print("DIFFERENT:", repr(line), "\n  ours     ", mine, "\n  reference", ref)
    print(f"{len(lines) - bad}/{len(lines)} lines tokenized identically ({sum(len(m) for m in ours)} ids)")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("lm")
    s.add_argument("ref")
    s.add_argument("hf")
    s.add_argument("ours")
    s.add_argument("--gguf")
    s.add_argument("--codes")
    s.add_argument("--cfg", type=float, default=1.0)
    s.add_argument("--norm", action="store_true")
    s = sub.add_parser("codec")
    s.add_argument("ref")
    s.add_argument("hf")
    s.add_argument("codes")
    s.add_argument("ours")
    s.add_argument("--head", default="keep2_plain")
    s.add_argument("--frames", type=int, default=0)
    s.add_argument("--f16", action="store_true")
    s.add_argument("--wav")
    s = sub.add_parser("ids")
    s.add_argument("ref")
    s.add_argument("hf")
    s.add_argument("texts")
    s.add_argument("ours")
    s = sub.add_parser("embed")
    s.add_argument("ref")
    s.add_argument("hf")
    s.add_argument("clip")
    s.add_argument("ours")
    s.add_argument("--npy")
    a = p.parse_args()
    sys.path[:0] = [str(Path(a.ref) / "model/src"), str(Path(a.ref) / "tools")]
    torch.set_grad_enabled(False)
    return {"lm": cmd_lm, "codec": cmd_codec, "ids": cmd_ids, "embed": cmd_embed}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
