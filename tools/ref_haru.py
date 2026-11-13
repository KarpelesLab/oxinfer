#!/usr/bin/env python3
"""Reference outputs of a Haru checkpoint (model_type haru) from the PyTorch code of
https://github.com/higurashi-ai/Haru, in float32 on the CPU, for examples/haru_check.rs to
compare with. The text stream is greedy and every normal value the session draws (the noise of
the latents and of the audio head) is recorded, so that the other side can replay it.

    ref_haru.py HARU_CHECKOUT MODEL_DIR OUT_DIR --user USER.wav [--reference VOICE.wav]
                [--system-prompt TEXT] [--frames N] [--speak-first]

Writes to OUT_DIR:
  manifest.json      the prompt's tokens, the token and text of every frame, timings
  noise.f32          every value of torch.randn, in the order drawn
  user.f32           the user's audio as fed (24 kHz, whole frames)
  reference_raw.f32  the voice clip as loaded, reference.f32 after prepare_reference
  ref_mean.f32       the codec's latent means of the reference [frames, 64]
  user_mean.f32      the codec's latent means of the user's frames, encoded as a stream
  hidden.f32         the language model's hidden state each frame came from [frames, hidden]
  latents.f32        the audio latent of every frame [frames, 64]
  audio.f32          the model's audio [frames * 3200]

Needs torch, transformers >= 5.16, tokenizers, safetensors, scipy, soundfile and numpy.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkout")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--user", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--system-prompt", default="")
    ap.add_argument("--frames", type=int, help="at most this many frames of the user")
    ap.add_argument("--speak-first", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(Path(a.checkout) / "haru"))
    import haru.session as session_module
    from haru import DuplexConfig, DuplexSession
    from haru.audio import load_audio, prepare_reference
    from haru.models import load_model

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(name, array, dtype="<f4"):
        np.asarray(array).astype(dtype).tofile(out / name)

    model = load_model(a.model, device="cpu", dtype=torch.float32)
    frame = model.config.frame_samples
    user = load_audio(a.user)
    count = max(1, -(-len(user) // frame))
    if a.frames:
        count = min(count, a.frames)
    user = np.pad(user, (0, max(0, count * frame - len(user))))[: count * frame]
    save("user.f32", user)
    reference = None
    if a.reference:
        raw = load_audio(a.reference)
        reference = prepare_reference(raw)
        save("reference_raw.f32", raw)
        save("reference.f32", reference)
        save("ref_mean.f32", model.codec.encode(torch.from_numpy(reference))[0].float().numpy())

    draws = []
    randn = torch.randn

    def recorded_randn(*args, **kwargs):
        value = randn(*args, **kwargs)
        draws.append(value.detach().float().reshape(-1).clone())
        return value

    torch.randn = recorded_randn
    prompts = []
    build_prompt = session_module.build_prompt

    def recorded_prompt(*args, **kwargs):
        prompt = build_prompt(*args, **kwargs)
        prompts.append(prompt.tokens.tolist())
        return prompt

    session_module.build_prompt = recorded_prompt

    config = DuplexConfig(temperature=0.0, seed=0, speak_first=a.speak_first, max_frames=count + 1)
    session = DuplexSession(model, a.system_prompt, reference, config)
    hidden, user_means = [], []
    emit = session._emit

    def recorded_emit(h, started):
        hidden.append(h[0].float().numpy().copy())
        return emit(h, started)

    session._emit = recorded_emit
    encode = session._codec.encode

    def recorded_encode(audio):
        mean = encode(audio)
        user_means.append(mean[0].float().numpy().copy())
        return mean

    session._codec.encode = recorded_encode

    t0 = time.time()
    frames = [session.start()]
    latents = [session._latent[0].float().numpy().copy()]
    start_secs = time.time() - t0
    t0 = time.time()
    for index in range(count - 1):
        frames.append(session.step(user[index * frame : (index + 1) * frame]))
        latents.append(session._latent[0].float().numpy().copy())
    step_secs = time.time() - t0
    torch.randn = randn

    save("noise.f32", torch.cat(draws).numpy())
    save("hidden.f32", np.stack(hidden))
    save("latents.f32", np.stack(latents))
    save("audio.f32", np.concatenate([f.audio for f in frames]))
    save("user_mean.f32", np.concatenate(user_means) if user_means else np.zeros((0, 64)))
    manifest = {
        "system_prompt": a.system_prompt,
        "speak_first": a.speak_first,
        "prompt_tokens": prompts[0],
        "tokens": [f.token for f in frames],
        "text": "".join(f.text for f in frames),
        "frames": count,
        "noise_values": int(sum(d.numel() for d in draws)),
        "start_secs": start_secs,
        "step_secs": step_secs,
        "torch": torch.__version__,
        "threads": a.threads,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("prompt_tokens", "tokens")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
