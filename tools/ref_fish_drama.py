"""Fish Audio's reference for Drama 3 (fish-speech's `fish_omni` and `dac` modules, on the CPU in
float32) beside what examples/fish_check.rs dumps.

  ref_fish_drama.py FISH_SPEECH_DIR prompt MODEL_DIR CASES.json OURS.json
  ref_fish_drama.py FISH_SPEECH_DIR gen MODEL_DIR OURS_PREFIX [--steps N]
  ref_fish_drama.py FISH_SPEECH_DIR decode CODEC.pth CODES.npy OURS.f32 [--wav OUT.wav]
  ref_fish_drama.py FISH_SPEECH_DIR encode CODEC.pth CLIP.wav OURS.npy

`prompt`: every case of CASES.json (`text`, optional `references` as `[transcript, frames]`
and `history` as `[text, frames]`, their codes a fixed function of the case's `seed`) through
`build_drama3_prompt`, compared with the rows `fish_check prompt` wrote.

`gen`: the greedy generation of `fish_check gen` (OURS_PREFIX.json: text, prompt rows, frames;
OURS_PREFIX.logits: per step the logits over the end token and the semantic band, then the nine
codebook heads) run again by `generate_drama3` on the same prompt: the experts stay FP8 where
the checkpoint has them so (dequantized per forward, without the activation quantization of the
SM89 kernel), everything else is float32. Greedy: every draw of the reference's sampler is
replaced by the argmax; when `fish_check gen` sampled (`--temp`), every draw returns the choice
oxinfer made, so both follow the same frames. Reports the identical frames, the largest logit difference and the mean
KL divergence per step until the first different frame, and the full-vocabulary logits of the
last prompt row against OURS_PREFIX.full when it exists.

`decode` / `encode`: the codec's `from_indices` and `encode` on the same input.

Needs torch, safetensors, tokenizers, numpy, soundfile, descript-audio-codec and fish-speech's
sources (FISH_SPEECH_DIR).
"""
import json, sys
from functools import partial

import numpy as np
import torch

fish, cmd = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
sys.path.insert(0, fish)
torch.set_grad_enabled(False)


def flag(name, default=None):
    return args[args.index(name) + 1] if name in args else default


def load_model(path):
    from pathlib import Path
    from tokenizers import Tokenizer
    from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone, load_backbone_weights
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
    from fish_speech.models.text2semantic.fish_omni.model import AUDIO_DECODER_PREFIXES, FishOmniModel, iter_checkpoint_tensors
    from fish_speech.models.text2semantic.fish_omni.backbone import CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY
    from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder, load_audio_decoder_weights

    path = Path(path)
    config = FishOmniConfig.from_dict(json.load(open(path / "config.json")))
    backbone = FishOmniBackbone(config.text, fp8_experts=config.fp8_experts, dtype=torch.float32, device="cpu")
    tower = FishOmniAudioDecoder(config, dtype=torch.float32, device="cpu")
    load_backbone_weights(backbone, iter_checkpoint_tensors(path, skip_prefixes=AUDIO_DECODER_PREFIXES))
    load_audio_decoder_weights(tower, iter_checkpoint_tensors(path, skip_prefixes=(CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY)))
    return FishOmniModel(config, backbone, tower, Tokenizer.from_file(str(path / "tokenizer.json"))).eval()


def load_codec(path):
    from fish_speech.models.dac.modded_dac import DAC, ModelArgs, WindowLimitedTransformer
    from fish_speech.models.dac.rvq import DownsampleResidualVectorQuantize

    # fish_speech/configs/modded_dac_vq.yaml.
    common = dict(n_local_heads=-1, head_dim=64, rope_base=10000, norm_eps=1e-5, dropout_rate=0.1, attn_dropout_rate=0.1, channels_first=True)
    module = lambda: WindowLimitedTransformer(
        causal=True, window_size=128, input_dim=1024, config=ModelArgs(block_size=2048, n_layer=8, n_head=16, dim=1024, intermediate_size=3072, **common)
    )
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=1024, n_codebooks=9, codebook_size=1024, codebook_dim=8, quantizer_dropout=0.5, downsample_factor=[2, 2],
        post_module=module(), pre_module=module(), semantic_codebook_size=4096,
    )
    model = DAC(
        sample_rate=44100, encoder_dim=64, encoder_rates=[2, 4, 8, 8], decoder_dim=1536, decoder_rates=[8, 8, 4, 2],
        encoder_transformer_layers=[0, 0, 0, 4], decoder_transformer_layers=[4, 0, 0, 0],
        transformer_general_config=partial(ModelArgs, block_size=8192, **common), quantizer=quantizer,
    )
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    state = state.get("state_dict", state)
    result = model.load_state_dict(state, strict=False, assign=True)
    assert not result.missing_keys, result.missing_keys
    return model.eval()


def snr(ref, ours):
    n = min(len(ref), len(ours))
    ref, ours = ref[:n].astype(np.float64), ours[:n].astype(np.float64)
    return 10 * np.log10((ref**2).sum() / max(((ref - ours) ** 2).sum(), 1e-30))


if cmd == "prompt":
    from tokenizers import Tokenizer
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
    from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

    config = FishOmniConfig.from_dict(json.load(open(f"{args[0]}/config.json")))
    tok = Tokenizer.from_file(f"{args[0]}/tokenizer.json")
    cases, ours = json.load(open(args[1])), json.load(open(args[2]))
    same = 0
    n_cb, size = config.audio_decoder.num_codebooks, config.audio_decoder.vocab_size
    for i, (case, got) in enumerate(zip(cases, ours)):
        seed = case.get("seed", 0)
        # The codes of clip k of a case, as fish_check derives them.
        codes = lambda k, frames: torch.tensor([[(seed * 7919 + k * 104729 + f * 31 + c * 17 + f * c) % size for f in range(frames)] for c in range(n_cb)])
        refs = [(t, codes(k, f)) for k, (t, f) in enumerate(case.get("references", []))]
        hist = [(t, codes(len(refs) + k, f)) for k, (t, f) in enumerate(case.get("history", []))]
        rows = build_drama3_prompt(case["text"], tok, config, [t for t, _ in refs], [c for _, c in refs], hist)
        want = rows.flatten().tolist()
        ok = want == got
        same += ok
        if not ok:
            first = next((j for j, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
            print(f"case {i} ({case['text'][:40]!r}): {len(want) // (1 + n_cb)} rows, ours {len(got) // (1 + n_cb)}, first difference in row {first // (1 + n_cb)}")
    print(f"prompt rows identical for {same} of {len(cases)} cases ({sum(len(o) for o in ours) // (1 + n_cb)} rows)")

elif cmd == "gen":
    from fish_speech.models.text2semantic.fish_omni import tower as tower_module
    from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt, generate_drama3

    ours = json.load(open(args[1] + ".json"))
    model = load_model(args[0])
    cfg = model.config
    n_cb, band = cfg.audio_decoder.num_codebooks, cfg.audio_decoder.vocab_size
    w = 1 + n_cb
    steps = int(flag("--steps", len(ours["frames"]) // w + 1))
    refs = [(t, torch.tensor(c).reshape(-1, n_cb).T) for t, c in ours.get("references", [])]
    prompt = build_drama3_prompt(ours["text"], model.tokenizer, cfg, [t for t, _ in refs], [c for _, c in refs])
    print(f"prompt: {len(prompt)} rows, identical to ours: {prompt.flatten().tolist() == ours['prompt']}")
    prompt = torch.tensor(ours["prompt"]).reshape(-1, w)

    got = ours["frames"]
    if ours.get("sampled"):
        # Led along our sampled frames: every draw returns our choice (the main token twice, the
        # second being the redraw), so the logits of every step are compared on one trajectory.
        steps = len(got) // w + bool(ours.get("ended"))
        forced = []
        for i in range(0, len(got), w):
            main = 1 + got[i] - cfg.semantic_start_token_id
            forced += [main, main] + got[i + 2 : i + w]
        forced += [0, 0] + [0] * (n_cb - 1)
        draws = iter(forced)
        tower_module.sample_top_k_top_p = lambda logits, *a, **k: torch.tensor([next(draws)])
    else:
        # Greedy: every draw is the argmax.
        tower_module.sample_top_k_top_p = lambda logits, *a, **k: logits.float().argmax(-1)
    log, full = [], []
    candidates = torch.tensor([cfg.eos_token_id] + list(range(cfg.semantic_start_token_id, cfg.semantic_end_token_id + 1)))
    compute_logits = model.backbone.compute_logits

    def recording(hidden):
        logits = compute_logits(hidden)
        full.append(logits[0].float().clone())
        log.append([logits[0, candidates].float().clone()])
        return logits

    model.backbone.compute_logits = recording
    for head in model.audio_decoder.codebook_heads:
        head.register_forward_hook(lambda m, i, o: log[-1].append(o[0].float().clone()))
    import time

    t0 = time.time()
    frames, finish = generate_drama3(model, prompt, max_new_tokens=steps, temperature=0.0, top_k=30, top_p=1.0)
    print(f"reference: {len(frames)} frames in {time.time() - t0:.1f}s, finish {finish}")
    want = frames.flatten().tolist()
    n = min(len(want), len(got)) // w
    same = next((i for i in range(n) if want[i * w : (i + 1) * w] != got[i * w : (i + 1) * w]), n)
    print(f"frames: reference {len(want) // w}, ours {len(got) // w}, identical: {same} of {n}" + ("" if same == n else f" (first difference at frame {same})"))
    codes_same = sum(a == b for a, b in zip(want[: n * w], got[: n * w]))
    print(f"codes equal: {codes_same} of {n * w}")
    ol = np.fromfile(args[1] + ".logits", dtype=np.float32)
    per = 1 + band + (n_cb - 1) * band
    ol = ol.reshape(-1, per)
    upto = min(same + 1, len(log), len(ol))
    dmax, kls = 0.0, []
    agree = total = 0
    for s in range(upto):
        ref = torch.cat(log[s]).numpy()
        dmax = max(dmax, float(np.abs(ref - ol[s]).max()))
        # The choices each side would make greedily at this step (the main token and 9 codes).
        for a in range(0, len(ref) - 1, band):
            a += a > 0
            b = a + band + (a == 0)
            agree += int(ref[a:b].argmax() == ol[s, a:b].argmax())
            total += 1
        for a, b in [(0, 1 + band)] + [(1 + band + i * band, 1 + band + (i + 1) * band) for i in range(n_cb - 1)]:
            p = torch.log_softmax(torch.tensor(ref[a:b]).double(), -1)
            q = torch.log_softmax(torch.tensor(ol[s, a:b]).double(), -1)
            kls.append(float((p.exp() * (p - q)).sum()))
    print(f"logits of the first {upto} steps (main token and 9 heads): max |difference| {dmax:.3e}, mean KL {np.mean(kls):.3e}, max KL {np.max(kls):.3e}, same argmax {agree} of {total}")
    try:
        of = np.fromfile(args[1] + ".full", dtype=np.float32)
        ref = full[0].numpy()
        p = torch.log_softmax(torch.tensor(ref).double(), -1)
        q = torch.log_softmax(torch.tensor(of).double(), -1)
        print(f"backbone logits of the last prompt row ({len(ref)} ids): max |difference| {np.abs(ref - of).max():.3e}, KL {float((p.exp() * (p - q)).sum()):.3e}, argmax {int(ref.argmax())} / {int(of.argmax())}")
    except FileNotFoundError:
        pass

elif cmd == "decode":
    codec = load_codec(args[0])
    codes = torch.from_numpy(np.load(args[1])).long()  # [frames, codebooks]
    audio = codec.from_indices(codes.T[None].clone())[0, 0].numpy()
    ours = np.fromfile(args[2], dtype=np.float32)
    print(f"{codes.shape[0]} frames: reference {len(audio)} samples, ours {len(ours)}, SNR {snr(audio, ours):.1f} dB, max |difference| {np.abs(audio[: len(ours)] - ours[: len(audio)]).max():.2e}")
    if flag("--wav"):
        import soundfile as sf

        sf.write(flag("--wav"), audio, codec.sample_rate)

elif cmd == "encode":
    import soundfile as sf

    codec = load_codec(args[0])
    audio, rate = sf.read(args[1], dtype="float32", always_2d=True)
    assert rate == codec.sample_rate, f"{args[1]}: {rate} Hz, the codec takes {codec.sample_rate}"
    audio = torch.from_numpy(audio.mean(1))[None, None]
    codes = codec.encode(audio, torch.tensor([audio.shape[2]]))[0][0].T.numpy()  # [frames, codebooks]
    ours = np.load(args[2])
    n = min(len(codes), len(ours))
    print(f"reference {codes.shape} codes, ours {ours.shape}; equal: {(codes[:n] == ours[:n]).mean() * 100:.2f}% of all, per codebook {[round(float((codes[:n, i] == ours[:n, i]).mean()) * 100, 1) for i in range(codes.shape[1])]}")
    if flag("--save"):
        np.save(flag("--save"), np.ascontiguousarray(codes))
else:
    sys.exit(__doc__)
