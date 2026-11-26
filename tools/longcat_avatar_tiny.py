#!/usr/bin/env python3
"""Tiny random LongCat-Video-Avatar components built with Meituan's own modules
(github.com/meituan-longcat/LongCat-Video) and their outputs, for examples/longcat_avatar_check.rs.

usage: longcat_avatar_tiny.py REF_REPO OUT_DIR [TOKENIZER_DIR]

REF_REPO is a checkout of the reference repository, OUT_DIR receives a model folder in the layout
of the two hub repositories (OUT_DIR/LongCat-Video-Avatar-1.6/{base_model,base_model_int8,loraA,
loraB,scheduler,whisper-large-v3} and OUT_DIR/LongCat-Video/{vae,text_encoder,tokenizer}), inputs
and reference outputs (*.bin, little-endian f32) and OUT_DIR/case.json. TOKENIZER_DIR is the
`tokenizer/` folder of meituan-longcat/LongCat-Video (umT5's SentencePiece vocabulary); without
it the text encoder and the whole-pipeline cases are skipped. Then:

    cargo run --release --example longcat_avatar_check -- OUT_DIR

flash-attn is replaced by torch's scaled_dot_product_attention (same softmax scale), everything
runs in float32 on the CPU, and the weights are rounded to bfloat16 first so that the bf16 files
hold them exactly.
"""
import json, os, shutil, sys, types

import numpy as np
import torch
import torch.nn.functional as F

ref, out = sys.argv[1], sys.argv[2]
tok_dir = sys.argv[3] if len(sys.argv) > 3 else None

# ---- flash-attn shim (a package on disk, so that importlib finds a spec for it) ----------------
SHIM = '''
import torch, torch.nn.functional as F


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, **_):
    # [B, S, H, D]
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale)
    return o.transpose(1, 2)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **_):
    # [total, H, D]
    outs = []
    for i in range(len(cu_seqlens_q) - 1):
        qi = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].transpose(0, 1)
        ki = k[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].transpose(0, 1)
        vi = v[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].transpose(0, 1)
        outs.append(F.scaled_dot_product_attention(qi[None], ki[None], vi[None])[0].transpose(0, 1))
    return torch.cat(outs)
'''
os.makedirs(os.path.join(out, "_shim", "flash_attn"), exist_ok=True)
with open(os.path.join(out, "_shim", "flash_attn", "__init__.py"), "w") as f:
    f.write(SHIM)
sys.path.insert(0, os.path.join(out, "_shim"))
# The pipeline imports the vocal separator's package (ONNX Runtime) at module level; nothing here calls it.
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
if not torch.cuda.is_available():
    torch.cuda.ipc_collect = lambda: None
    torch.cuda.empty_cache = lambda: None
torch.compile = lambda f=None, **k: f if f is not None else (lambda g: g)

sys.path.insert(0, ref)
# Block-sparse attention (disabled in the released config) is written in triton; without triton
# the module that holds it is replaced.
try:
    import triton  # noqa: F401
except Exception:
    m = types.ModuleType("longcat_video.block_sparse_attention.bsa_interface")
    m.flash_attn_bsa_3d = None
    sys.modules["longcat_video.block_sparse_attention.bsa_interface"] = m
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan  # noqa: E402
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel  # noqa: E402
from longcat_video.modules.quantization import load_quantized_dit, quantize_model, save_quantized_state_dict  # noqa: E402
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler  # noqa: E402
from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

A = os.path.join(out, "LongCat-Video-Avatar-1.6")
B = os.path.join(out, "LongCat-Video")
os.makedirs(A, exist_ok=True)
os.makedirs(B, exist_ok=True)
torch.manual_seed(0)
case = {}


def randomize(m, scale=0.1):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if ("norm" in n or n.endswith("gamma")) and (n.endswith("weight") or n.endswith("gamma")):
                p.copy_(1.0 + torch.randn_like(p) * 0.2)
            else:
                p.copy_(torch.randn_like(p) * scale)
            p.copy_(p.to(torch.bfloat16).float())


def save(name, t):
    a = t.detach().float().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    a.astype(np.float32).tofile(os.path.join(out, name))
    return list(a.shape)


# ---- VAE (Wan 2.1 layout) ---------------------------------------------------------------------
Z = 4
vae = AutoencoderKLWan(
    base_dim=8, z_dim=Z, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True],
    latents_mean=[float(v) for v in torch.randn(Z) * 0.3], latents_std=[float(v) for v in torch.rand(Z) + 0.5],
).float().eval()
randomize(vae, 0.08)
vae.save_pretrained(os.path.join(B, "vae"))
with torch.no_grad():
    video = torch.randn(1, 3, 13, 48, 64).clamp(-1, 1)
    case["vae_enc_video"] = save("vae_enc_video.bin", video)
    case["vae_enc_video_mode"] = save("vae_enc_video_mode.bin", vae.encode(video).latent_dist.mode())
    image = torch.randn(1, 3, 1, 48, 64).clamp(-1, 1)
    case["vae_enc_image"] = save("vae_enc_image.bin", image)
    case["vae_enc_image_mode"] = save("vae_enc_image_mode.bin", vae.encode(image).latent_dist.mode())
    z = torch.randn(1, Z, 4, 6, 8)
    case["vae_dec_latent"] = save("vae_dec_latent.bin", z)
    case["vae_dec_video"] = save("vae_dec_video.bin", vae.decode(z, return_dict=False)[0])

# ---- transformer --------------------------------------------------------------------------------
cfg = dict(
    in_channels=Z, out_channels=Z, hidden_size=128, depth=3, num_heads=2, caption_channels=32, mlp_ratio=4,
    adaln_tembed_dim=32, frequency_embedding_size=32, patch_size=(1, 2, 2), enable_flashattn2=True, text_tokens_zero_pad=True,
    audio_window=5, audio_block=5, audio_channel=24, intermediate_dim=16, output_dim=64, context_tokens=4, vae_scale=4,
    audio_prenorm=False, class_range=24, class_interval=4,
)
dit = LongCatVideoAvatarTransformer3DModel(cp_split_hw=[1, 1], **cfg).float().eval()
randomize(dit, 0.05)
dit.to(torch.bfloat16).save_pretrained(os.path.join(A, "base_model"))
dit.float()
with open(os.path.join(A, "base_model", "config.json")) as f:
    c = json.load(f)
c["cp_split_hw"] = None
c["model_max_length"] = 512
with open(os.path.join(A, "base_model", "config.json"), "w") as f:
    json.dump(c, f, indent=2)
with open(os.path.join(A, "config.json"), "w") as f:
    json.dump({"model_name": "LongCat-Video-Avatar-1.6"}, f)
os.makedirs(os.path.join(A, "scheduler"), exist_ok=True)
with open(os.path.join(A, "scheduler", "scheduler_config.json"), "w") as f:
    json.dump({"_class_name": "FlowMatchEulerDiscreteScheduler", "shift": 7.0, "use_dynamic_shifting": False, "invert_sigmas": False, "time_shift_type": "linear"}, f)

# int8 copy, written by the reference's own quantizer
import copy  # noqa: E402

dq = quantize_model(copy.deepcopy(dit))
save_quantized_state_dict(dq, os.path.join(A, "base_model_int8"), config_source_dir=os.path.join(A, "base_model"))
dit8 = load_quantized_dit(A, subfolder="base_model_int8", cp_split_hw=[1, 1])
for p in dit8.parameters():
    p.data = p.data.float()
for b_ in dit8.buffers():
    if b_.dtype == torch.bfloat16:
        b_.data = b_.data.float()


# LoRAs in the released files' layout (rank 4 here; `alpha_scale` 0.5 as 64 / 128)
def make_lora(seed, path):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    R = 4
    for i in range(cfg["depth"]):
        blk = dit.blocks[i]
        for name, mod, sep in (
            ("attn.qkv", blk.attn.qkv, 3), ("attn.proj", blk.attn.proj, 1), ("cross_attn.q_linear", blk.cross_attn.q_linear, 1),
            ("cross_attn.kv_linear", blk.cross_attn.kv_linear, 2), ("ffn.w1", blk.ffn.w1, 1), ("ffn.w2", blk.ffn.w2, 1), ("ffn.w3", blk.ffn.w3, 1),
        ):
            key = "lora___lorahyphen___" + f"blocks.{i}.{name}".replace(".", "___lorahyphen___")
            sd[key + ".alpha_scale"] = torch.tensor(0.5)
            sd[key + ".lora_down.weight"] = torch.randn(sep * R, mod.in_features, generator=g) * 0.1
            if sep > 1:
                for j in range(sep):
                    sd[key + f".lora_up.blocks.{j}.weight"] = torch.randn(mod.out_features // sep, R, generator=g) * 0.1
            else:
                sd[key + ".lora_up.weight"] = torch.randn(mod.out_features, R, generator=g) * 0.1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(sd, path)


make_lora(1, os.path.join(A, "loraA", "lora.safetensors"))
make_lora(2, os.path.join(A, "loraB", "lora.safetensors"))
case["lora_rank"] = 4

T, LH, LW = 6, 8, 12  # latent frames, height, width
NA = 1 + 4 * (T - 1)  # audio rows of a clip (one per frame)


def dit_inputs(seed, t_lat=T, humans=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, Z, t_lat, LH, LW, generator=g)
    text = torch.randn(1, 1, 12, cfg["caption_channels"], generator=g)
    mask = torch.ones(1, 12, dtype=torch.int64)
    mask[:, 9:] = 0
    audio = torch.randn(humans, NA, cfg["audio_window"], cfg["audio_block"], cfg["audio_channel"], generator=g)
    return x, text, mask, audio


def run_dit(model, name, x, text, mask, audio, ts, **kw):
    with torch.no_grad():
        y = model(hidden_states=x, timestep=ts, encoder_hidden_states=text, encoder_attention_mask=mask, audio_embs=audio, **kw)
    case[name] = {
        "x": save(name + "_x.bin", x), "text": save(name + "_text.bin", text), "mask": [int(v) for v in mask[0]],
        "audio": save(name + "_audio.bin", audio), "t": [float(v) for v in ts[0]], "out": save(name + "_out.bin", y),
        "kw": {k: (v if not isinstance(v, torch.Tensor) else None) for k, v in kw.items()},
    }
    return y


def ts_of(t, t_lat, cond):
    ts = torch.full((1, t_lat), float(t))
    ts[:, :cond] = 0
    return ts


x, text, mask, audio = dit_inputs(10)
run_dit(dit, "dit_t2v", x, text, mask, audio, ts_of(875.0, T, 0), num_cond_latents=0)
run_dit(dit, "dit_i2v", x, text, mask, audio, ts_of(613.5, T, 1), num_cond_latents=1)
run_dit(dit8, "dit_i2v_int8", x, text, mask, audio, ts_of(613.5, T, 1), num_cond_latents=1)
# continuation: [reference | 2 conditioning frames | noise], the reference at index 10
x, text, mask, audio = dit_inputs(11, t_lat=T + 1)
run_dit(dit, "dit_avc", x, text, mask, audio, ts_of(301.0, T + 1, 3), num_cond_latents=3, num_ref_latents=1, ref_img_index=10, mask_frame_range=0)
run_dit(dit, "dit_avc_mask", x, text, mask, audio, ts_of(301.0, T + 1, 3), num_cond_latents=3, num_ref_latents=1, ref_img_index=4, mask_frame_range=1)
# two speakers
x, text, mask, audio = dit_inputs(12, humans=2)
masks = torch.zeros(3, LH * 8, LW * 8)
masks[0, 8:40, 8:40] = 1
masks[1, 16:56, 56:88] = 1
masks[2] = 1 - masks[0] - masks[1]
case["dit_multi_masks"] = save("dit_multi_masks.bin", masks)
run_dit(dit, "dit_multi", x, text, mask, audio, ts_of(613.5, T, 1), num_cond_latents=1, ref_target_masks=masks)
# LoRAs: one, then the mix (0.3 A + 0.7 B)
x, text, mask, audio = dit_inputs(13)
for model, tag in ((dit, ""), (dit8, "_int8")):
    model.load_lora(os.path.join(A, "loraA", "lora.safetensors"), "dmd", multiplier=1.0, lora_network_dim=4, lora_network_alpha=2)
    model.load_lora(os.path.join(A, "loraB", "lora.safetensors"), "opsd", multiplier=1.0, lora_network_dim=4, lora_network_alpha=2)
    for l in model.lora_dict.values():
        l.float()
    model.enable_lora_mix("dmd", "opsd", 0.3, 4)
    for l in model.lora_dict.values():
        l.float()
    model.set_lora_mix_step(0)
    run_dit(model, "dit_lora_mix" + tag, x, text, mask, audio, ts_of(613.5, T, 1), num_cond_latents=1)
    model.set_lora_mix_step(4)
    run_dit(model, "dit_lora_b" + tag, x, text, mask, audio, ts_of(613.5, T, 1), num_cond_latents=1)
    model.set_lora_mix_step(0)

# ---- Whisper (32 encoder layers, as large-v3: the pipeline groups the 33 hidden states) --------
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration  # noqa: E402

wcfg = WhisperConfig(
    d_model=24, encoder_layers=32, encoder_attention_heads=2, encoder_ffn_dim=48, decoder_layers=1, decoder_attention_heads=2,
    decoder_ffn_dim=48, num_mel_bins=128, max_source_positions=1500, vocab_size=64, max_target_positions=8, pad_token_id=0,
    bos_token_id=1, eos_token_id=2, decoder_start_token_id=1,
)
# The release's file is a WhisperForConditionalGeneration (tensor names under `model.`); the
# pipeline uses its WhisperModel.
whisper_full = WhisperForConditionalGeneration(wcfg).float().eval()
randomize(whisper_full, 0.08)
whisper_full.save_pretrained(os.path.join(A, "whisper-large-v3"))
whisper = whisper_full.model
fe = WhisperFeatureExtractor(feature_size=128)
fe.save_pretrained(os.path.join(A, "whisper-large-v3"))
sr = 16000
tt = np.arange(int(sr * 2.6)) / sr
speech = (0.3 * np.sin(2 * np.pi * 220 * tt) * (1 + 0.5 * np.sin(2 * np.pi * 3 * tt)) + 0.05 * np.random.RandomState(0).randn(len(tt))).astype(np.float32)
case["speech"] = save("speech.bin", speech)

# ---- text encoder ---------------------------------------------------------------------------------
text_encoder = tokenizer = None
if tok_dir:
    from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    ucfg = UMT5Config(
        vocab_size=256384, d_model=cfg["caption_channels"], d_kv=8, d_ff=64, num_layers=2, num_heads=4, relative_attention_num_buckets=32,
        relative_attention_max_distance=128, feed_forward_proj="gated-gelu", dense_act_fn="gelu_new", is_gated_act=True,
    )
    text_encoder = UMT5EncoderModel(ucfg).float().eval()
    randomize(text_encoder, 0.2)
    text_encoder.save_pretrained(os.path.join(B, "text_encoder"))
    shutil.copytree(tok_dir, os.path.join(B, "tokenizer"), dirs_exist_ok=True)

sched = FlowMatchEulerDiscreteScheduler.from_pretrained(A, subfolder="scheduler")
pipe = LongCatVideoAvatarPipeline(
    tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=sched, dit=dit, audio_encoder=whisper,
    audio_feature_extractor=fe, model_type="avatar-v1.5",
)
pipe.device = "cpu"
emb = pipe.get_audio_embedding(speech, fps=25, device="cpu", sample_rate=sr, model_type="avatar-v1.5", use_loudness_norm=False)
case["audio_emb"] = save("audio_emb.bin", emb)
case["sigmas_distill"] = [float(v) for v in pipe.get_timesteps_sigmas(8, use_distill=True)]
sched.set_timesteps(8, sigmas=pipe.get_timesteps_sigmas(8, use_distill=True))
case["timesteps_distill"] = [float(v) for v in sched.timesteps]
case["sched_sigmas"] = [float(v) for v in sched.sigmas]

if tok_dir:
    prompts = [
        "A western man stands on stage under dramatic lighting, holding a microphone close to their mouth.",
        "  Two people  are speaking.\n一个女人在说话，镜头静止。 Café &amp; naïve — 3.5% “quoted” ﬁne ｶﾀｶﾅ　ＡＢＣ１２３！",
    ]
    case["prompts"] = prompts
    from longcat_video.pipeline_longcat_video_avatar import prompt_clean  # noqa: E402

    for i, p in enumerate(prompts):
        e, m = pipe._get_t5_prompt_embeds(p, max_sequence_length=512, device="cpu", dtype=torch.float32)
        ids = tokenizer([prompt_clean(p)], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_tensors="pt").input_ids
        case[f"prompt{i}_ids"] = [int(v) for v in ids[0][: int(m.sum())]]
        case[f"prompt{i}_embeds"] = save(f"prompt{i}_embeds.bin", e[0, 0])

    # the whole pipeline: image + audio + text -> 13 frames, then one continuation of 29 frames
    from PIL import Image  # noqa: E402

    rs = np.random.RandomState(3)
    img = (np.clip(rs.randn(60, 100, 3) * 40 + 128, 0, 255)).astype(np.uint8)
    img = np.asarray(Image.fromarray(img).resize((500, 300), Image.BICUBIC))
    Image.fromarray(img).save(os.path.join(out, "ref.png"))
    image = Image.open(os.path.join(out, "ref.png")).convert("RGB")
    NF, NC = 29, 13
    idx = torch.arange(5) - 2

    def clip_audio(start):
        ci = torch.arange(start, start + NF).unsqueeze(1) + idx.unsqueeze(0)
        return emb[torch.clamp(ci, 0, emb.shape[0] - 1)][None]

    for model, tag in ((dit, ""), (dit8, "_int8")):
        pipe.dit = model
        model.enable_lora_mix("dmd", "opsd", 1.0, 4)
        for l in model.lora_dict.values():
            l.float()
        tl = 1 + (NF - 1) // 4
        noise = torch.randn(1, Z, tl, 60, 104, generator=torch.Generator().manual_seed(5))
        case["e2e_noise" + tag] = save("e2e_noise" + tag + ".bin", noise)
        with torch.no_grad():
            frames, lat = pipe.generate_ai2v(
                image=image, prompt=prompts[0], resolution="480p", num_frames=NF, num_inference_steps=8, use_distill=True,
                text_guidance_scale=1.0, audio_guidance_scale=1.0, latents=noise.clone(), output_type="both", audio_emb=clip_audio(0),
            )
        case["e2e_latent" + tag] = save("e2e_latent" + tag + ".bin", lat)
        case["e2e_frames" + tag] = save("e2e_frames" + tag + ".bin", frames[0])
        if tag:
            continue
        # continuation as the demo does it (no KV cache with the mixed LoRAs, no `enhance_hf`)
        video = [Image.fromarray(f) for f in (frames[0] * 255).astype(np.uint8)]
        noise2 = torch.randn(1, Z, tl, 60, 104, generator=torch.Generator().manual_seed(6))
        case["avc_noise"] = save("avc_noise.bin", noise2)
        with torch.no_grad():
            frames2, lat2 = pipe.generate_avc(
                video=video, video_latent=lat, prompt=prompts[0], height=480, width=832, num_frames=NF, num_cond_frames=NC,
                num_inference_steps=8, use_distill=True, text_guidance_scale=1.0, audio_guidance_scale=1.0, latents=noise2.clone(),
                output_type="both", use_kv_cache=False, enhance_hf=False, audio_emb=clip_audio(NF - NC), ref_latent=lat[:, :, :1].clone(),
                ref_img_index=10, mask_frame_range=0,
            )
        case["avc_latent"] = save("avc_latent.bin", lat2)
        case["avc_frames"] = save("avc_frames.bin", frames2[0])
        # classifier-free guidance (the undistilled schedule, 3 steps)
        model.disable_all_loras()
        model.lora_mix_config = None
        with torch.no_grad():
            lat3 = pipe.generate_ai2v(
                image=image, prompt=prompts[0], negative_prompt="blurry", resolution="480p", num_frames=NF, num_inference_steps=3,
                use_distill=False, text_guidance_scale=4.0, audio_guidance_scale=4.0, latents=noise.clone(), output_type="latent",
                audio_emb=clip_audio(0),
            )
        case["cfg_latent"] = save("cfg_latent.bin", lat3)

case["dit_config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
with open(os.path.join(out, "case.json"), "w") as f:
    json.dump(case, f, indent=1)
print("wrote", out)
