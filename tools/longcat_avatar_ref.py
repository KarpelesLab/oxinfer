#!/usr/bin/env python3
"""Reference outputs of LongCat-Video-Avatar 1.6 from Meituan's own code
(github.com/meituan-longcat/LongCat-Video) on the released weights, stage by stage, for
examples/longcat_avatar_check.rs.

usage: longcat_avatar_ref.py REF_REPO AVATAR_DIR BASE_DIR OUT_DIR [--int8] [--frames N]
           [--stages text,audio,vae,dit,e2e,avc,e2e_f32] [--f32 text,vae,dit]
           [--seconds S]

REF_REPO is a checkout of the reference repository (its `assets/avatar/single` example is the
input), AVATAR_DIR the release (`base_model/`, `base_model_int8/`, `loraA/`, `loraB/`,
`scheduler/`, `whisper-large-v3/`), BASE_DIR meituan-longcat/LongCat-Video (`text_encoder/`,
`tokenizer/`, `vae/`). Writes inputs and outputs (*.bin, little-endian f32), `ref.png` and
OUT_DIR/case.json; then

    cargo run --release --example longcat_avatar_check -- OUT_DIR --avatar=AVATAR_DIR --base=BASE_DIR --device=cuda

What differs from `run_demo_avatar_single_audio_to_video.py --model_type avatar-v1.6
--use_distill --mix_steps=4 --mix_alpha=1.0 [--use_int8]`, and why:

- one process on one GPU, no torch.distributed (context-parallel size 1);
- flash-attn replaced by torch's scaled_dot_product_attention with the same scale (flash-attn
  has no wheel for every platform; the released config asks for it);
- no vocal separation (ONNX Runtime): the speech is the example's audio decoded by librosa at
  16 kHz, the samples oxinfer is then given too (`speech.bin`);
- the noise comes from a CPU generator and is written out, so that both sides start from the
  same latents (the demo draws it on the GPU).

The models run in bfloat16 as the demo loads them; `--f32 text,vae` adds float32 runs of those
stages (`*_f32` cases), which separate bf16 rounding from a real difference: the text encoder
loaded again in float32, the VAE on the CPU with float32 activations on the weights its bfloat16
load rounded.
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("avatar")
ap.add_argument("base")
ap.add_argument("out")
ap.add_argument("--int8", action="store_true")
ap.add_argument("--frames", type=int, default=93)
ap.add_argument("--stages", default="text,audio,vae,dit,e2e")
ap.add_argument("--f32", default="")
ap.add_argument("--seconds", type=float, default=0.0, help="speech length (default: what the clips need plus one second)")
ap.add_argument("--segments", type=int, default=1)
ap.add_argument("--dit-frames", type=int, default=0, help="frames of the single-forward cases (default: --frames)")
args = ap.parse_args()
stages = set(args.stages.split(","))
f32 = set(x for x in args.f32.split(",") if x)
out = args.out
os.makedirs(out, exist_ok=True)

# ---- flash-attn shim (a package on disk, so that importlib finds a spec for it) ----------------
SHIM = '''
import torch, torch.nn.functional as F


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, **_):
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale)
    return o.transpose(1, 2)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **_):
    outs = []
    for i in range(len(cu_seqlens_q) - 1):
        qi = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].transpose(0, 1)
        ki = k[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].transpose(0, 1)
        vi = v[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].transpose(0, 1)
        outs.append(F.scaled_dot_product_attention(qi[None], ki[None], vi[None])[0].transpose(0, 1))
    return torch.cat(outs)
'''
try:
    import flash_attn  # noqa: F401

    case_attn = "flash-attn " + flash_attn.__version__
except Exception:
    os.makedirs(os.path.join(out, "_shim", "flash_attn"), exist_ok=True)
    with open(os.path.join(out, "_shim", "flash_attn", "__init__.py"), "w") as f:
        f.write(SHIM)
    sys.path.insert(0, os.path.join(out, "_shim"))
    case_attn = "torch scaled_dot_product_attention"
import types  # noqa: E402

try:
    import audio_separator.separator.uvr_lib_v5  # noqa: F401
except Exception:
    import importlib.machinery

    for name in ("audio_separator", "audio_separator.separator", "audio_separator.separator.uvr_lib_v5"):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        m.__path__ = []
        m.spec_utils = None
        m.Separator = None
        sys.modules[name] = m
sys.path.insert(0, args.ref)
try:
    import triton  # noqa: F401
except Exception:
    m = types.ModuleType("longcat_video.block_sparse_attention.bsa_interface")
    m.flash_attn_bsa_3d = None
    sys.modules["longcat_video.block_sparse_attention.bsa_interface"] = m
torch.compile = lambda f=None, **k: f if f is not None else (lambda g: g)

import librosa  # noqa: E402
from diffusers.utils import load_image  # noqa: E402
from transformers import AutoTokenizer, UMT5EncoderModel  # noqa: E402

from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor  # noqa: E402
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan  # noqa: E402
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel  # noqa: E402
from longcat_video.modules.quantization import load_quantized_dit  # noqa: E402
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler  # noqa: E402
from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline, prompt_clean  # noqa: E402

dev = "cuda"
case_path = os.path.join(out, "case.json")
case = json.load(open(case_path)) if os.path.exists(case_path) else {}
case["attention"] = case_attn
case["torch"] = torch.__version__
case["device"] = torch.cuda.get_device_name(0)
tag = "_int8" if args.int8 else ""


def save(name, t):
    a = t.detach().float().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    a.astype(np.float32).tofile(os.path.join(out, name))
    return list(a.shape)


def flush():
    with open(case_path, "w") as f:
        json.dump(case, f, indent=1)


with open(os.path.join(args.ref, "assets/avatar/single_example_1.json")) as f:
    example = json.load(f)
prompt = example["prompt"]
case["prompts"] = [prompt]
NF, NC, FPS = args.frames, 13, 25
bf16 = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(args.base, subfolder="tokenizer")
vae = AutoencoderKLWan.from_pretrained(args.base, subfolder="vae", torch_dtype=bf16).to(dev)
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(args.avatar, subfolder="scheduler", torch_dtype=bf16)
pipe = LongCatVideoAvatarPipeline(tokenizer=tokenizer, text_encoder=None, vae=vae, scheduler=sched, dit=None, model_type="avatar-v1.5")
pipe.device = dev

# ---- text -----------------------------------------------------------------------------------------
if "text" in stages:
    for dt, sfx in [(bf16, "")] + ([(torch.float32, "_f32")] if "text" in f32 else []):
        te = UMT5EncoderModel.from_pretrained(args.base, subfolder="text_encoder", torch_dtype=dt).to(dev)
        pipe.text_encoder = te
        t0 = time.time()
        with torch.no_grad():
            e, m = pipe._get_t5_prompt_embeds(prompt, max_sequence_length=512, device=dev, dtype=dt)
        n = int(m.sum())
        print(f"text encoder ({dt}): {n} tokens in {time.time() - t0:.1f}s")
        ids = tokenizer([prompt_clean(prompt)], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_tensors="pt").input_ids
        case["prompt0_ids"] = [int(v) for v in ids[0][:n]]
        case["prompt0_embeds" + sfx] = save("prompt0_embeds" + sfx + ".bin", e[0, 0, :n])
        if not sfx:
            torch.save((e.cpu(), m.cpu()), os.path.join(out, "_text.pt"))
        del te
        pipe.text_encoder = None
        torch.cuda.empty_cache()
    flush()

# ---- audio ----------------------------------------------------------------------------------------
seconds = args.seconds or (NF / FPS + (args.segments - 1) * (NF - NC) / FPS + 1.0)
if "audio" in stages:
    speech, _ = librosa.load(os.path.join(args.ref, example["cond_audio"]["person1"]), sr=16000)
    n = math.ceil(seconds * 16000)
    speech = speech[:n]
    if len(speech) < n:
        speech = np.append(speech, [0.0] * (n - len(speech)))
    speech = speech.astype(np.float32)
    case["speech"] = save("speech.bin", speech)
    wdir = os.path.join(args.avatar, "whisper-large-v3")
    pipe.audio_encoder = get_audio_encoder(wdir, "avatar-v1.5").to(dev)
    pipe.audio_feature_extractor = get_audio_feature_extractor(wdir, "avatar-v1.5")
    t0 = time.time()
    emb = pipe.get_audio_embedding(speech, fps=FPS, device=dev, sample_rate=16000, model_type="avatar-v1.5", use_loudness_norm=False)
    print(f"audio features: {tuple(emb.shape)} in {time.time() - t0:.1f}s")
    case["audio_emb"] = save("audio_emb.bin", emb)
    torch.save(emb.cpu(), os.path.join(out, "_audio.pt"))
    pipe.audio_encoder = None
    torch.cuda.empty_cache()
    flush()

# ---- VAE ------------------------------------------------------------------------------------------
image = load_image(os.path.join(args.ref, example["cond_image"]))
image.save(os.path.join(out, "ref.png"))
height, width = pipe.get_condition_shape(image, "480p", scale_factor_spatial=16)
case["size"] = [height, width]
pixels = pipe.video_processor.preprocess(image, height=height, width=width, resize_mode="crop")  # [1, 3, H, W] in [-1, 1]
if "vae" in stages:
    case["image_pixels"] = save("image_pixels.bin", pixels)
    x = pixels.unsqueeze(2)
    for dt, sfx in [(bf16, "")] + ([(torch.float32, "_f32")] if "vae" in f32 else []):
        # The float32 case is meant as the exact one: on the CPU (cuDNN picks float32 convolution
        # algorithms, TF32 among them, whose results differ by more than rounding).
        vdev = dev if sfx == "" else "cpu"
        v = vae.to(vdev, dt)
        with torch.no_grad():
            mode = v.encode(x.to(vdev, dt)).latent_dist.mode()
            case["vae_enc_image"] = save("vae_enc_image.bin", x)
            case["vae_enc_image_mode" + sfx] = save("vae_enc_image_mode" + sfx + ".bin", mode)
            g = torch.Generator().manual_seed(7)
            z = mode.float().cpu().repeat(1, 1, 3, 1, 1) + 0.3 * torch.randn(1, 16, 3, height // 8, width // 8, generator=g)
            case["vae_dec_latent"] = save("vae_dec_latent.bin", z)
            t0 = time.time()
            dec = v.decode(z.to(vdev, dt), return_dict=False)[0]
            print(f"VAE decode ({dt}) of 3 latent frames: {time.time() - t0:.1f}s")
            case["vae_dec_video" + sfx] = save("vae_dec_video" + sfx + ".bin", dec)
    vae.to(dev, bf16)
    flush()

# ---- transformer ------------------------------------------------------------------------------------
if stages & {"dit", "e2e", "avc", "e2e_f32"}:
    t0 = time.time()
    if args.int8:
        dit = load_quantized_dit(args.avatar, subfolder="base_model_int8", cp_split_hw=[1, 1])
    else:
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(args.avatar, subfolder="base_model", cp_split_hw=[1, 1], torch_dtype=bf16)
    dit = dit.to(dev).eval()
    print(f"transformer loaded in {time.time() - t0:.1f}s")
    pipe.dit = dit
    emb = torch.load(os.path.join(out, "_audio.pt"))
    te, tm = torch.load(os.path.join(out, "_text.pt"))
    idx = torch.arange(5) - 2

    def clip_audio(start, frames):
        ci = torch.arange(start, start + frames).unsqueeze(1) + idx.unsqueeze(0)
        return emb[torch.clamp(ci, 0, emb.shape[0] - 1)][None].to(dev)

    with torch.no_grad():
        cond = pipe.normalize_latents(vae.encode(pixels.unsqueeze(2).to(dev, bf16)).latent_dist.mode().float())

    def lora_reset():
        dit.disable_all_loras()
        dit.lora_dict.clear()
        dit.lora_mix_config = None
        dit._lora_mix_active = None

    def lora_load():
        dit.load_lora(os.path.join(args.avatar, "loraA", "lora.safetensors"), "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
        dit.load_lora(os.path.join(args.avatar, "loraB", "lora.safetensors"), "opsd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
        dit.enable_lora_mix("dmd", "opsd", 1.0, 4)

    def dit_f32():
        # Float32 activations on the weights as stored: the LoRAs dropped (enabling them cast their
        # factors to bfloat16 in place; they are read again when next loaded) and the transformer's
        # float tensors read again from the files (the loaders cast them to bfloat16).
        from safetensors.torch import load_file

        lora_reset()
        dit.float()
        sub = os.path.join(args.avatar, "base_model_int8" if args.int8 else "base_model")
        # One file at a time: the bf16 transformer is 63 GB in float32, beside which the whole
        # state dict and its float32 copy do not fit in a DGX Spark's memory.
        seen = set()
        for fn in sorted(os.listdir(sub)):
            if fn.endswith(".safetensors"):
                sd = load_file(os.path.join(sub, fn), device="cpu")
                bad = dit.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}, strict=False)
                assert not bad.unexpected_keys, bad.unexpected_keys
                seen.update(sd.keys())
                del sd
        missing = set(dit.state_dict().keys()) - seen
        assert not missing, sorted(missing)[:8]

    def dit_bf16():
        lora_reset()
        for n_, p_ in dit.named_parameters():
            if p_.dtype == torch.float32:
                p_.data = p_.data.to(bf16)
        for m_ in dit.modules():
            if m_.__class__.__name__ == "QuantizedLinear" and m_.bias is not None:
                m_.bias = m_.bias.to(bf16)

if "dit" in stages:
    DF = args.dit_frames or NF
    tl = 1 + (DF - 1) // 4
    g = torch.Generator().manual_seed(11)
    x = torch.randn(1, 16, tl, height // 8, width // 8, generator=g).to(dev)
    x[:, :, :1] = cond
    audio = clip_audio(0, DF)

    def one(name, sigma, dt=bf16):
        ts = torch.full((1, tl), sigma * 1000.0, device=dev, dtype=dt)
        ts[:, :1] = 0
        t0 = time.time()
        with torch.no_grad():
            y = dit(hidden_states=x.to(dt), timestep=ts, encoder_hidden_states=te.to(dev, dt), encoder_attention_mask=tm.to(dev), num_cond_latents=1, audio_embs=audio)
        torch.cuda.synchronize()
        print(f"{name} ({dt}): forward of {tl} latent frames in {time.time() - t0:.1f}s")
        if dt != bf16:
            # Same inputs as the bfloat16 case: only another output.
            case[name]["out_f32"] = save(name + "_out_f32.bin", y)
        else:
            case[name] = {
                "x": save(name + "_x.bin", x), "text": save(name + "_text.bin", te), "mask": [int(v) for v in tm[0]],
                "audio": save(name + "_audio.bin", audio), "t": [float(v) for v in ts[0].float()], "out": save(name + "_out.bin", y),
                "kw": {"num_cond_latents": 1},
            }
        flush()

    def three(dt):
        one("dit_i2v" + tag, 0.9, dt)
        lora_load()
        dit.set_lora_mix_step(0)
        one("dit_lora_a" + tag, 0.9, dt)
        dit.set_lora_mix_step(4)
        one("dit_lora_b" + tag, 0.5, dt)
        dit.set_lora_mix_step(0)

    three(bf16)
    if "dit" in f32:
        # The same three forwards with float32 activations on the weights as stored.
        dit_f32()
        three(torch.float32)
        # Back to what the demo runs for the stages after this one.
        dit_bf16()

if stages & {"e2e", "avc"}:
    if not dit.lora_dict:
        lora_load()
    pipe.text_encoder = types.SimpleNamespace(dtype=bf16, config=types.SimpleNamespace(d_model=4096))
    # The prompt's rows from the text stage (the 11 GB encoder need not stay beside the transformer).
    pipe.encode_prompt = lambda **kw: (te.to(dev, bf16), tm.to(dev), None, None)
    tl = 1 + (NF - 1) // 4
    noise = torch.randn(1, 16, tl, height // 8, width // 8, generator=torch.Generator().manual_seed(5))
    case["e2e_noise" + tag] = save("e2e_noise" + tag + ".bin", noise)
    t0 = time.time()
    with torch.no_grad():
        frames, lat = pipe.generate_ai2v(
            image=image, prompt=prompt, resolution="480p", num_frames=NF, num_inference_steps=8, use_distill=True, text_guidance_scale=1.0,
            audio_guidance_scale=1.0, latents=noise.clone().to(dev), output_type="both", audio_emb=clip_audio(0, NF),
        )
    print(f"generate_ai2v: {NF} frames in {time.time() - t0:.1f}s")
    case["e2e_latent" + tag] = save("e2e_latent" + tag + ".bin", lat)
    case["e2e_frames" + tag] = save("e2e_frames" + tag + ".bin", frames[0])
    flush()

    def write_frames(name, fr):
        # PNGs of a few frames, to look at (no video encoder is assumed).
        from PIL import Image

        for i in sorted({0, len(fr) // 2, len(fr) - 1}):
            Image.fromarray((fr[i] * 255).astype(np.uint8)).save(os.path.join(out, f"{name}_{i:03d}.png"))

    write_frames("e2e" + tag, frames[0])
    if "avc" in stages:
        from PIL import Image  # noqa: E402

        video = [Image.fromarray(f) for f in (frames[0] * 255).astype(np.uint8)]
        noise2 = torch.randn(1, 16, tl, height // 8, width // 8, generator=torch.Generator().manual_seed(6))
        case["avc_noise" + tag] = save("avc_noise" + tag + ".bin", noise2)
        t0 = time.time()
        with torch.no_grad():
            frames2, lat2 = pipe.generate_avc(
                video=video, video_latent=lat, prompt=prompt, height=height, width=width, num_frames=NF, num_cond_frames=NC, num_inference_steps=8,
                use_distill=True, text_guidance_scale=1.0, audio_guidance_scale=1.0, latents=noise2.clone().to(dev), output_type="both",
                use_kv_cache=False, enhance_hf=False, audio_emb=clip_audio(NF - NC, NF), ref_latent=lat[:, :, :1].clone(), ref_img_index=10,
                mask_frame_range=0,
            )
        print(f"generate_avc: {NF} frames in {time.time() - t0:.1f}s")
        case["avc_latent" + tag] = save("avc_latent" + tag + ".bin", lat2)
        case["avc_frames" + tag] = save("avc_frames" + tag + ".bin", frames2[0])
        write_frames("avc" + tag, frames2[0])
    flush()

if "e2e_f32" in stages:
    # The first clip again from the same noise with float32 activations everywhere: the
    # transformer and the LoRAs on the weights as stored, the float32 text encoder's rows (the
    # `--f32 text` case of the text stage), the VAE on the weights its bfloat16 load rounded.
    dit_f32()
    lora_load()
    vae.float()
    n = case["prompt0_embeds_f32"][0]
    te32 = te.float().clone()
    te32[0, 0, :n] = torch.from_numpy(np.fromfile(os.path.join(out, "prompt0_embeds_f32.bin"), dtype=np.float32).reshape(n, -1))
    pipe.text_encoder = types.SimpleNamespace(dtype=torch.float32, config=types.SimpleNamespace(d_model=4096))
    pipe.encode_prompt = lambda **kw: (te32.to(dev), tm.to(dev), None, None)
    tl = 1 + (NF - 1) // 4
    noise = torch.from_numpy(np.fromfile(os.path.join(out, "e2e_noise" + tag + ".bin"), dtype=np.float32).reshape(1, 16, tl, height // 8, width // 8))
    t0 = time.time()
    with torch.no_grad():
        frames, lat = pipe.generate_ai2v(
            image=image, prompt=prompt, resolution="480p", num_frames=NF, num_inference_steps=8, use_distill=True, text_guidance_scale=1.0,
            audio_guidance_scale=1.0, latents=noise.clone().to(dev), output_type="both", audio_emb=clip_audio(0, NF),
        )
    print(f"generate_ai2v (float32): {NF} frames in {time.time() - t0:.1f}s")
    case["e2e_latent" + tag + "_f32"] = save("e2e_latent" + tag + "_f32.bin", lat)
    case["e2e_frames" + tag + "_f32"] = save("e2e_frames" + tag + "_f32.bin", frames[0])
    flush()
print("wrote", out)
