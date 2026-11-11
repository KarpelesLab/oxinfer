#!/usr/bin/env python3
"""Builds a tiny random-weight FLUX.1 pipeline folder (diffusers layout, float32) with every part
of the real one: a 2 + 2 block FluxTransformer2DModel, a 4-level 16-channel AutoencoderKL, a CLIP
text model and a T5 encoder, with the real tokenizers and scheduler config. `tools/ref_flux_ext.py`
runs diffusers on it and compares oxinfer's dumps, which checks the pipeline plumbing exactly
(token ids, mask blending, LoRA keys, BFL key remapping, ...) without the 12B checkpoint.

usage: flux_tiny.py OUT_DIR REAL_DIR [--kind kontext|dev|schnell|fill|control|redux]
  REAL_DIR: a FLUX.1 diffusers folder to copy tokenizer/, tokenizer_2/ and scheduler/ from
            (e.g. ~/.cache/oxinfer/hub/unsloth--FLUX.1-Kontext-dev).
  --kind:   kontext (FluxKontextPipeline), dev (FluxPipeline), schnell (no guidance embedding),
            fill (384 input channels, FluxFillPipeline), control (128, FluxControlPipeline),
            redux (dev plus a tiny SigLIP tower and Redux projector in OUT_DIR/redux)
  --bfl:    also write OUT_DIR/bfl/flux1-tiny.safetensors (the transformer in Black Forest Labs'
            single-file layout) and OUT_DIR/bfl/ae.safetensors (the autoencoder likewise)
"""
import argparse, json, os, shutil

import torch


def randomize(m, std=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    for n, p in m.named_parameters():
        with torch.no_grad():
            if p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * (1.0 / max(p.shape[1], 1)) ** 0.5)
            elif "norm" in n and n.endswith("weight") or n.endswith("scale"):
                p.copy_(1 + torch.randn(p.shape, generator=g) * std)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * std)
            # Exactly representable in bf16: loaders that keep weights as bf16 (CLIP, the VAE
            # decoder) then hold the same values as the float32 reference.
            p.copy_(p.to(torch.bfloat16).to(torch.float32))


def transformer_to_bfl(sd, d):
    """A diffusers FluxTransformer2DModel state dict in the original BFL names (the inverse of
    diffusers' convert_flux_transformer_checkpoint_to_diffusers)."""
    out = {}
    top = {"x_embedder": "img_in", "context_embedder": "txt_in", "proj_out": "final_layer.linear",
           "time_text_embed.timestep_embedder.linear_1": "time_in.in_layer",
           "time_text_embed.timestep_embedder.linear_2": "time_in.out_layer",
           "time_text_embed.guidance_embedder.linear_1": "guidance_in.in_layer",
           "time_text_embed.guidance_embedder.linear_2": "guidance_in.out_layer",
           "time_text_embed.text_embedder.linear_1": "vector_in.in_layer",
           "time_text_embed.text_embedder.linear_2": "vector_in.out_layer"}
    dbl = {"norm1.linear": "img_mod.lin", "norm1_context.linear": "txt_mod.lin", "attn.to_out.0": "img_attn.proj",
           "attn.to_add_out": "txt_attn.proj", "ff.net.0.proj": "img_mlp.0", "ff.net.2": "img_mlp.2",
           "ff_context.net.0.proj": "txt_mlp.0", "ff_context.net.2": "txt_mlp.2"}
    norms = {"attn.norm_q": "img_attn.norm.query_norm", "attn.norm_k": "img_attn.norm.key_norm",
             "attn.norm_added_q": "txt_attn.norm.query_norm", "attn.norm_added_k": "txt_attn.norm.key_norm"}
    for a, b in top.items():
        for s in ("weight", "bias"):
            if f"{a}.{s}" in sd:
                out[f"{b}.{s}"] = sd[f"{a}.{s}"]
    for s in ("weight", "bias"):
        scale, shift = sd[f"norm_out.linear.{s}"].chunk(2, dim=0)  # diffusers (scale, shift), BFL (shift, scale)
        out[f"final_layer.adaLN_modulation.1.{s}"] = torch.cat([shift, scale], dim=0)
    i = 0
    while f"transformer_blocks.{i}.norm1.linear.weight" in sd:
        p, q = f"transformer_blocks.{i}", f"double_blocks.{i}"
        for a, b in dbl.items():
            for s in ("weight", "bias"):
                out[f"{q}.{b}.{s}"] = sd[f"{p}.{a}.{s}"]
        for a, b in norms.items():
            out[f"{q}.{b}.scale"] = sd[f"{p}.{a}.weight"]
        for s in ("weight", "bias"):
            out[f"{q}.img_attn.qkv.{s}"] = torch.cat([sd[f"{p}.attn.{x}.{s}"] for x in ("to_q", "to_k", "to_v")])
            out[f"{q}.txt_attn.qkv.{s}"] = torch.cat(
                [sd[f"{p}.attn.{x}.{s}"] for x in ("add_q_proj", "add_k_proj", "add_v_proj")])
        i += 1
    i = 0
    while f"single_transformer_blocks.{i}.norm.linear.weight" in sd:
        p, q = f"single_transformer_blocks.{i}", f"single_blocks.{i}"
        for s in ("weight", "bias"):
            out[f"{q}.modulation.lin.{s}"] = sd[f"{p}.norm.linear.{s}"]
            out[f"{q}.linear2.{s}"] = sd[f"{p}.proj_out.{s}"]
            out[f"{q}.linear1.{s}"] = torch.cat(
                [sd[f"{p}.attn.to_q.{s}"], sd[f"{p}.attn.to_k.{s}"], sd[f"{p}.attn.to_v.{s}"], sd[f"{p}.proj_mlp.{s}"]])
        out[f"{q}.norm.query_norm.scale"] = sd[f"{p}.attn.norm_q.weight"]
        out[f"{q}.norm.key_norm.scale"] = sd[f"{p}.attn.norm_k.weight"]
        i += 1
    return {k: v.contiguous() for k, v in out.items()}


def bfl_to_transformer(sd, d):
    """The inverse of `transformer_to_bfl` (diffusers' own converter assumes FLUX.1's 3072-wide
    layers): BFL names back to a FluxTransformer2DModel state dict of hidden size `d`."""
    out = {}
    def put(k, v):
        out[k] = v.contiguous()
    top = {"img_in": "x_embedder", "txt_in": "context_embedder", "final_layer.linear": "proj_out",
           "time_in.in_layer": "time_text_embed.timestep_embedder.linear_1",
           "time_in.out_layer": "time_text_embed.timestep_embedder.linear_2",
           "guidance_in.in_layer": "time_text_embed.guidance_embedder.linear_1",
           "guidance_in.out_layer": "time_text_embed.guidance_embedder.linear_2",
           "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
           "vector_in.out_layer": "time_text_embed.text_embedder.linear_2"}
    dbl = {"img_mod.lin": "norm1.linear", "txt_mod.lin": "norm1_context.linear", "img_attn.proj": "attn.to_out.0",
           "txt_attn.proj": "attn.to_add_out", "img_mlp.0": "ff.net.0.proj", "img_mlp.2": "ff.net.2",
           "txt_mlp.0": "ff_context.net.0.proj", "txt_mlp.2": "ff_context.net.2"}
    norms = {"img_attn.norm.query_norm": "attn.norm_q", "img_attn.norm.key_norm": "attn.norm_k",
             "txt_attn.norm.query_norm": "attn.norm_added_q", "txt_attn.norm.key_norm": "attn.norm_added_k"}
    for b, a in top.items():
        for s in ("weight", "bias"):
            if f"{b}.{s}" in sd:
                put(f"{a}.{s}", sd[f"{b}.{s}"])
    for s in ("weight", "bias"):
        shift, scale = sd[f"final_layer.adaLN_modulation.1.{s}"].chunk(2, dim=0)
        put(f"norm_out.linear.{s}", torch.cat([scale, shift], dim=0))
    i = 0
    while f"double_blocks.{i}.img_mod.lin.weight" in sd:
        p, q = f"transformer_blocks.{i}", f"double_blocks.{i}"
        for b, a in dbl.items():
            for s in ("weight", "bias"):
                put(f"{p}.{a}.{s}", sd[f"{q}.{b}.{s}"])
        for b, a in norms.items():
            put(f"{p}.{a}.weight", sd[f"{q}.{b}.scale"])
        for s in ("weight", "bias"):
            for x, t in zip(("to_q", "to_k", "to_v"), sd[f"{q}.img_attn.qkv.{s}"].chunk(3, dim=0)):
                put(f"{p}.attn.{x}.{s}", t)
            for x, t in zip(("add_q_proj", "add_k_proj", "add_v_proj"), sd[f"{q}.txt_attn.qkv.{s}"].chunk(3, dim=0)):
                put(f"{p}.attn.{x}.{s}", t)
        i += 1
    i = 0
    while f"single_blocks.{i}.modulation.lin.weight" in sd:
        p, q = f"single_transformer_blocks.{i}", f"single_blocks.{i}"
        for s in ("weight", "bias"):
            put(f"{p}.norm.linear.{s}", sd[f"{q}.modulation.lin.{s}"])
            put(f"{p}.proj_out.{s}", sd[f"{q}.linear2.{s}"])
            l1 = sd[f"{q}.linear1.{s}"]
            for x, t in zip(("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"), torch.split(l1, [d, d, d, l1.shape[0] - 3 * d])):
                put(f"{p}.{x}.{s}", t)
        put(f"{p}.attn.norm_q.weight", sd[f"{q}.norm.query_norm.scale"])
        put(f"{p}.attn.norm_k.weight", sd[f"{q}.norm.key_norm.scale"])
        i += 1
    return out


def vae_to_bfl(sd, levels):
    """A diffusers AutoencoderKL state dict in the BFL `ae.safetensors` names."""
    out = {}
    for k, v in sd.items():
        part, rest = k.split(".", 1)
        r = rest
        if rest.startswith("down_blocks."):
            r = "down." + rest[len("down_blocks."):]
        elif rest.startswith("up_blocks."):
            i, tail = rest[len("up_blocks."):].split(".", 1)
            r = f"up.{levels - 1 - int(i)}.{tail}"
        elif rest.startswith("mid_block.resnets."):
            i, tail = rest[len("mid_block.resnets."):].split(".", 1)
            r = f"mid.block_{int(i) + 1}.{tail}"
        elif rest.startswith("mid_block.attentions.0."):
            l, tail = rest[len("mid_block.attentions.0."):].rsplit(".", 1)
            l = {"group_norm": "norm", "to_q": "q", "to_k": "k", "to_v": "v", "to_out.0": "proj_out"}[l]
            r = f"mid.attn_1.{l}.{tail}"
            if l != "norm" and tail == "weight":
                v = v[:, :, None, None]
        elif rest.startswith("conv_norm_out."):
            r = "norm_out." + rest[len("conv_norm_out."):]
        r = r.replace("resnets.", "block.").replace("downsamplers.0.", "downsample.").replace("upsamplers.0.", "upsample.")
        r = r.replace("conv_shortcut", "nin_shortcut")
        out[f"{part}.{r}"] = v.contiguous()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("real")
    ap.add_argument("--kind", default="kontext", choices=["kontext", "dev", "schnell", "fill", "control", "redux"])
    ap.add_argument("--bfl", action="store_true")
    a = ap.parse_args()
    from diffusers import (AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxControlPipeline, FluxFillPipeline,
                           FluxKontextPipeline, FluxPipeline, FluxTransformer2DModel)
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5Config, T5EncoderModel, T5TokenizerFast

    in_ch = {"fill": 384, "control": 128}.get(a.kind, 64)
    tr = FluxTransformer2DModel(
        patch_size=1, in_channels=in_ch, out_channels=64, num_layers=2, num_single_layers=2, attention_head_dim=32,
        num_attention_heads=2, joint_attention_dim=64, pooled_projection_dim=32, guidance_embeds=a.kind != "schnell",
        axes_dims_rope=(4, 14, 14))
    randomize(tr, seed=1)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(32, 32, 64, 64), layers_per_block=1,
        latent_channels=16, norm_num_groups=32, use_quant_conv=False, use_post_quant_conv=False,
        shift_factor=0.1159, scaling_factor=0.3611, sample_size=64)
    randomize(vae, seed=2)
    clip = CLIPTextModel(CLIPTextConfig(
        vocab_size=49408, hidden_size=32, intermediate_size=64, projection_dim=32, num_hidden_layers=2,
        num_attention_heads=2, max_position_embeddings=77, hidden_act="quick_gelu", bos_token_id=49406,
        eos_token_id=49407, pad_token_id=49407))
    randomize(clip, seed=3)
    t5 = T5EncoderModel(T5Config(
        vocab_size=32128, d_model=64, d_kv=16, d_ff=128, num_layers=2, num_heads=4, feed_forward_proj="gated-gelu",
        dense_act_fn="gelu_new", is_gated_act=True, relative_attention_num_buckets=32,
        relative_attention_max_distance=128))
    randomize(t5, seed=4)
    cls = {"kontext": FluxKontextPipeline, "fill": FluxFillPipeline, "control": FluxControlPipeline}.get(a.kind, FluxPipeline)
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(a.real, subfolder="scheduler")
    if a.kind == "schnell":
        sch = FlowMatchEulerDiscreteScheduler(shift=1.0, use_dynamic_shifting=False)
    pipe = cls(scheduler=sch, vae=vae, text_encoder=clip, tokenizer=CLIPTokenizer.from_pretrained(a.real, subfolder="tokenizer"),
               text_encoder_2=t5, tokenizer_2=T5TokenizerFast.from_pretrained(a.real, subfolder="tokenizer_2"),
               transformer=tr)
    pipe.save_pretrained(a.out, safe_serialization=True)
    # The tokenizers exactly as the real repo has them (oxinfer reads tokenizer.json / vocab.json).
    for t in ("tokenizer", "tokenizer_2"):
        shutil.rmtree(os.path.join(a.out, t), ignore_errors=True)
        shutil.copytree(os.path.join(a.real, t), os.path.join(a.out, t))
    # The CLIP weights under the `text_model.` prefix, as the released checkpoints store them
    # (newer transformers save them without it and read both).
    from safetensors.torch import load_file, save_file
    te = os.path.join(a.out, "text_encoder", "model.safetensors")
    sd = load_file(te)
    save_file({k if k.startswith("text_model.") else f"text_model.{k}": v for k, v in sd.items()}, te, metadata={"format": "pt"})
    if a.bfl:
        os.makedirs(os.path.join(a.out, "bfl"), exist_ok=True)
        save_file(transformer_to_bfl(tr.state_dict(), 64), os.path.join(a.out, "bfl", "flux1-tiny.safetensors"))
        save_file(vae_to_bfl(vae.state_dict(), 4), os.path.join(a.out, "bfl", "ae.safetensors"))
    if a.kind == "redux":
        from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
        from transformers import SiglipVisionConfig, SiglipVisionModel
        sig = SiglipVisionModel(SiglipVisionConfig(
            hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, image_size=56,
            patch_size=14, num_channels=3))
        randomize(sig, seed=5)
        red = ReduxImageEncoder(redux_dim=48, txt_in_features=64)
        randomize(red, seed=6)
        sig.save_pretrained(os.path.join(a.out, "redux", "image_encoder"))
        red.save_pretrained(os.path.join(a.out, "redux", "image_embedder"))
        json.dump({"_class_name": "FluxPriorReduxPipeline", "image_encoder": ["transformers", "SiglipVisionModel"],
                   "image_embedder": ["flux", "ReduxImageEncoder"], "feature_extractor": ["transformers", "SiglipImageProcessor"]},
                  open(os.path.join(a.out, "redux", "model_index.json"), "w"), indent=2)
        json.dump({"do_convert_rgb": None, "do_normalize": True, "do_rescale": True, "do_resize": True,
                   "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "image_processor_type": "SiglipImageProcessor",
                   "resample": 3, "rescale_factor": 1 / 255, "size": {"height": 56, "width": 56}},
                  open(os.path.join(a.out, "redux", "feature_extractor_config.json"), "w"), indent=2)
        os.makedirs(os.path.join(a.out, "redux", "feature_extractor"), exist_ok=True)
        shutil.move(os.path.join(a.out, "redux", "feature_extractor_config.json"),
                    os.path.join(a.out, "redux", "feature_extractor", "preprocessor_config.json"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
