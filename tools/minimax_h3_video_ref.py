#!/usr/bin/env python3
"""MiniMax-H3 video on the real small files in torch, against what oxinfer computed on the same
inputs (its OXINFER_H3_DUMP files): the reference for runs no tiny checkpoint stands for (the
real weights' ranges, 4000 rows, ComfyUI's pruned FP8 transformer with a turbo LoRA).

  oxinfer run:   OXINFER_H3_DUMP=DUMP OXINFER_H3_DUMP_BLOCKS=0 OXINFER_H3_PROMPT_EMBEDS=EMBEDS.bin \\
                 oxinfer --device cpu --model TRANSFORMER --vae ... --lora LORA -p ... -o out.avi

  minimax_h3_video_ref.py blocks DUMP TRANSFORMER.safetensors LORA.safetensors [--call I] [--first A] [--last B]
      every block of forward I (the one OXINFER_H3_DUMP_BLOCKS chose) on oxinfer's input of that
      block (teacher forced: the block's own difference) and on the reference's own previous
      output (free running: the accumulated difference).

  minimax_h3_video_ref.py sample DUMP EMBEDS.bin TRANSFORMER.safetensors LORA.safetensors OUT_DIR \\
          LT LH LW N_AUDIO [--points 5] [--shifts 6 3] [--start I] [--stop J] [--layers N]
      the whole sampling from oxinfer's noise (s0_video_in.bin, s0_audio_in.bin) and its cached
      prompt embedding: diffusers' MiniMaxH3Transformer3DModel.forward, its layout and timestep
      helpers and MiniMaxH3Scheduler, unchanged. Prints each step's distance from oxinfer's rows
      and velocities and writes OUT_DIR/ref.latents in the format of OXINFER_H3_LATENTS, which
      oxinfer decodes (OXINFER_H3_LATENTS=OUT_DIR/ref.latents with the same size and frames).
      LT LH LW N_AUDIO: latent frames, latent height and width, audio latents per channel
      (448x256, 124 frames: 37 16 28 207). --start I: from oxinfer's rows before its step I
      instead of the noise, --stop J: stops before step J, without a latents file (one forward
      on oxinfer's own input tells an implementation's difference from a trajectory's drift). --layers N: only
      the first N blocks (a smoke test).

The transformer: ComfyUI's single file (`minimax_h3_fl2va_pruned_fp8_scaled.safetensors`), read
here without ComfyUI: float8_e4m3fn times `weight_scale`, bf16, float16 decoded to float32, the
LoRA merged (W += alpha / rank * B A), `qkv_proj` split into q / k / v, `fc1` as [value; gate],
the curve form of the timestep embedding as ComfyUI's `_time_embedding` computes it
(`adaln_t_table` read by linear interpolation, AdaLN projections without the SiLU). The 50 blocks
are one diffusers MiniMaxH3TransformerBlock refilled per block (1.5 GB of float32 at a time), in
float32 on the CPU. A forward of 4682 rows takes 4 to 6 minutes on 64 shared cores, the blocks
mode 15 to 25 s per block. tools/minimax_h3_video_comfy_ref.py does one forward with ComfyUI's
own code.
"""
import argparse, json, struct, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from diffusers import MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed, MiniMaxH3TransformerBlock
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep

H, HEADS, HD, FFN, LAYERS = 5376, 56, 128, 14336, 50
torch.set_grad_enabled(False)


def fp8_table():
    """float8_e4m3fn: sign, 4 exponent bits (bias 7), 3 mantissa bits; no infinities."""
    o = np.zeros(256, dtype=np.float32)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 15, c & 7
        v = float("nan") if (e == 15 and m == 7) else (m / 8.0 * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7))
        o[c] = -v if s else v
    return o


FP8 = fp8_table()
_torch_fp8 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
assert np.array_equal(np.nan_to_num(_torch_fp8, nan=7e7), np.nan_to_num(FP8, nan=7e7)), "the FP8 table is not torch's"


class SafeTensors:
    """A safetensors file, mapped; `get` decodes a tensor to float32."""

    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.h = json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, mode="r")

    def get(self, k):
        i = self.h[k]
        a, b = i["data_offsets"]
        b = np.ascontiguousarray(self.mm[self.base + a:self.base + b])
        dt = i["dtype"]
        if dt == "F32":
            t = b.view("<f4").copy()
        elif dt == "F16":
            t = b.view("<f2").astype(np.float32)
        elif dt == "BF16":
            t = (b.view("<u2").astype(np.uint32) << 16).view("<f4")
        elif dt == "F8_E4M3":
            t = FP8[b]
        else:
            raise ValueError(f"{k}: dtype {dt}")
        return torch.from_numpy(t).reshape(i["shape"])


class Weights:
    """The transformer's tensors by ComfyUI's names, decoded to float32."""

    def __init__(self, transformer, lora):
        self.tfm, self.lora = SafeTensors(transformer), SafeTensors(lora)
        self.k = self.tfm.h["adaln_t_table"]["shape"][1]

    def get(self, name):
        return self.tfm.get(name)

    def weight(self, name):
        """The decoded weight of module `name` with the LoRA merged."""
        w = self.get(name + ".weight")
        if name + ".weight_scale" in self.tfm.h:
            w = w * self.get(name + ".weight_scale")
        ka = f"diffusion_model.{name}.lora_A.weight"
        if ka in self.lora.h:
            a, b = self.lora.get(ka), self.lora.get(f"diffusion_model.{name}.lora_B.weight")
            w = w + (float(self.lora.get(f"diffusion_model.{name}.alpha")) / a.shape[0]) * (b @ a)
        return w

    def block(self, p, adaln=True):
        """diffusers' state dict of the block under ComfyUI's prefix `p`."""
        q, k, v = self.weight(p + "attn.qkv_proj").chunk(3)
        gate, value = self.weight(p + "mlp.fc1").chunk(2)
        sd = {
            "norm1.weight": self.get(p + "norm1.weight"), "norm2.weight": self.get(p + "norm2.weight"),
            "attn.to_q.weight": q, "attn.to_k.weight": k, "attn.to_v.weight": v,
            "attn.norm_q.weight": self.get(p + "attn.q_norm.weight"),
            "attn.norm_k.weight": self.get(p + "attn.k_norm.weight"),
            "attn.to_out.0.weight": self.weight(p + "attn.out_proj"),
            "ff.net.0.proj.weight": torch.cat([value, gate]), "ff.net.2.weight": self.weight(p + "mlp.fc2"),
        }
        if adaln:
            sd["adaln_proj.linear.weight"] = self.get(p + "adaln_proj.linear.weight")
            sd["adaln_proj.linear.bias"] = self.get(p + "adaln_proj.linear.bias")
        return sd


def curve_block(blk):
    """The curve form: the AdaLN projection reads the coordinates as they are (no SiLU)."""
    blk.adaln_proj.forward = lambda t: blk.adaln_proj.linear(t).view(-1, 6 * H).chunk(6, dim=-1)
    return blk


def rd(dump, name, cols):
    return torch.from_numpy(np.fromfile(f"{dump}/{name}", dtype="<f4").reshape(-1, cols).copy())


def cmp(name, ref, ox):
    d = ref - ox
    rows = d.norm(dim=1) / ref.norm(dim=1).clamp(min=1e-9)
    print(f"  {name}: relative {float(d.norm() / ref.norm()):.3e}, max abs {float(d.abs().max()):.3e} (reference max "
          f"{float(ref.abs().max()):.2e}), per row median {float(rows.median()):.2e} max {float(rows.max()):.2e}", flush=True)


def blocks(a):
    w = Weights(a.transformer, a.lora)
    pos, tags = rd(a.dump, "positions.bin", 3), rd(a.dump, "tags.bin", 1).long().flatten()
    tidx, temb = rd(a.dump, f"s{a.call}_tidx.bin", 1).long().flatten(), rd(a.dump, f"s{a.call}_temb.bin", w.k)
    print(f"{pos.shape[0]} rows, {temb.shape[0]} timesteps, rows per tag {torch.bincount(tags).tolist()}", flush=True)
    adaln_idx, rot = tidx * 3 + tags, MiniMaxH3RotaryPosEmbed(16)(pos)
    blk = curve_block(MiniMaxH3TransformerBlock(H, HEADS, HD, FFN, w.k, 1e-5, 1e-5).float().eval())
    prev = rd(a.dump, "x_in.bin" if a.first == 0 else f"blk_{a.first - 1}.bin", H)
    free = prev.clone()
    for i in range(a.first, a.last):
        t0 = time.time()
        blk.load_state_dict(w.block(f"blocks.{i}."))
        ox = rd(a.dump, f"blk_{i}.bin", H)
        forced = blk(prev[None], temb, adaln_idx, rot)[0]
        print(f"block {i} ({time.time() - t0:.0f}s)", flush=True)
        # The block's own change of the sequence: the residual stream's largest values (1e5 to 7e6)
        # would hide a difference otherwise.
        cmp("teacher forced, the block's change", forced - prev, ox - prev)
        free = blk(free[None], temb, adaln_idx, rot)[0] if i > a.first else forced
        cmp("free running, the sequence", free, ox)
        prev = ox


class Curve(torch.nn.Module):
    """ComfyUI's `_time_embedding` of the curve form: the table's row at t, interpolated."""

    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, t):
        p = t.float().clamp(0.0, 1.0) * (self.table.shape[0] - 1)
        i0 = p.floor().long().clamp(max=self.table.shape[0] - 2)
        return torch.lerp(self.table[i0], self.table[i0 + 1], (p - i0).unsqueeze(1))


def build_model(w, layers=LAYERS):
    """diffusers' transformer on the weights `w`: the curve form, the blocks decoded one at a time."""
    k = w.k
    model = MiniMaxH3Transformer3DModel(num_layers=1, time_embed_dim=k, time_embed_hidden_dim=k, freq_dim=k).float().eval()
    model.time_proj = torch.nn.Identity()
    model.time_embedder = Curve(w.get("adaln_t_table"))
    for name, cname in (("proj_in", "video_patch_proj"), ("audio_proj_in", "audio_patch_proj"), ("context_embedder", "condition_proj"),
                        ("proj_out", "final_layer.video_out"), ("audio_proj_out", "final_layer.audio_out"),
                        ("norm_out.linear", "final_layer.adaln_proj.linear")):
        m = model.get_submodule(name)
        m.weight.data, m.bias.data = w.get(cname + ".weight"), w.get(cname + ".bias")
    model.norm_out.norm.weight.data = w.get("final_layer.norm.weight")
    model.token_refiner.final_norm.weight.data = w.get("token_refiner.final_norm.weight")
    for i, b in enumerate(model.token_refiner.refiner_blocks):
        b.load_state_dict(w.block(f"token_refiner.blocks.{i}.", adaln=False))
    no = model.norm_out

    def out_forward(hidden_states, temb, timestep_indices):
        shift, scale = no.linear(temb).chunk(2, dim=-1)
        hidden_states = no.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)

    no.forward = out_forward
    blk = curve_block(model.transformer_blocks[0])
    pool, pending = ThreadPoolExecutor(1), {}

    class Lazy(torch.nn.Module):
        """Block `i` of the file through the one resident block (the next one decoded meanwhile)."""

        def __init__(self, i):
            super().__init__()
            self.i = i

        def forward(self, *args):
            i = self.i
            blk.load_state_dict(pending.pop(i).result() if i in pending else w.block(f"blocks.{i}."))
            nxt = (i + 1) % layers
            pending[nxt] = pool.submit(w.block, f"blocks.{nxt}.")
            return blk(*args)

    model.transformer_blocks = torch.nn.ModuleList([Lazy(i) for i in range(layers)])
    return model


def sample(a):
    model = build_model(Weights(a.transformer, a.lora), a.layers or LAYERS)
    b = open(a.embeds, "rb").read()
    n_text = struct.unpack("<I", b[:4])[0]
    text_tags = torch.tensor(list(b[4:4 + n_text]), dtype=torch.long)
    text = torch.from_numpy(np.frombuffer(b[4 + n_text:], dtype="<f4").reshape(n_text, -1).copy())
    rows, audio = rd(a.dump, f"s{a.start}_video_in.bin", 96), rd(a.dump, f"s{a.start}_audio_in.bin", 32)
    pos, tags, vidx, aidx, tidx, n_cond, n_cond_audio = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_tags, a.lt, a.lh, a.lw, a.n_audio, (1, 2, 2), 2, 2, 0, ())
    print(f"{pos.shape[0]} rows: {n_text} text, {rows.shape[0]} video, {audio.shape[0]} audio", flush=True)
    print("positions against oxinfer's: max abs diff", float((pos.float() - rd(a.dump, "positions.bin", 3)).abs().max()), flush=True)
    sv, sa = MiniMaxH3Scheduler(shift=a.shifts[0]), MiniMaxH3Scheduler(shift=a.shifts[1])
    sv.set_timesteps(a.points)
    sa.set_timesteps(a.points)
    print("sigmas", sv.sigmas.tolist(), sa.sigmas.tolist(), flush=True)
    for i, (t, ta) in enumerate(zip(sv.timesteps, sa.timesteps)):
        if i < a.start:
            # The schedulers find their step by the timestep.
            continue
        if i >= a.stop:
            return
        t0 = time.time()
        uniq, ridx = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            vidx, aidx, n_cond, n_cond_audio, n_text, float(t), float(ta), max(float(t), 0.999), 1.0)
        cmp(f"step {i}, video rows before", rows, rd(a.dump, f"s{i}_video_in.bin", 96))
        cmp(f"step {i}, audio rows before", audio, rd(a.dump, f"s{i}_audio_in.bin", 32))
        v, au = model(hidden_states=rows[None], audio_hidden_states=audio[None], encoder_hidden_states=text[None],
                      timestep=uniq, timestep_indices=ridx, token_tags=tags, position_ids=pos.float(),
                      video_indices=vidx, audio_indices=aidx, text_indices=tidx, return_dict=False)
        print(f"step {i}: t = {float(t):.6f} (audio {float(ta):.6f}), {time.time() - t0:.0f}s", flush=True)
        v[0].numpy().tofile(f"{a.out}/ref_s{i}_v.bin")
        au[0].numpy().tofile(f"{a.out}/ref_s{i}_a.bin")
        cmp(f"step {i}, video velocity", v[0], rd(a.dump, f"s{i}_v.bin", 96))
        cmp(f"step {i}, audio velocity", au[0], rd(a.dump, f"s{i}_a.bin", 32))
        rows = sv.step(v[0].float(), t, rows, return_dict=False)[0]
        audio = sa.step(au[0].float(), ta, audio, return_dict=False)[0]
    with open(f"{a.out}/ref.latents", "wb") as f:
        f.write(b"H3LT" + struct.pack("<6I", a.lt, a.lh, a.lw, a.n_audio, rows.numel(), audio.numel()))
        f.write(rows.numpy().astype("<f4").tobytes())
        f.write(audio.numpy().astype("<f4").tobytes())
    print(f"wrote {a.out}/ref.latents", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    pb = sub.add_parser("blocks")
    pb.add_argument("dump")
    pb.add_argument("transformer")
    pb.add_argument("lora")
    pb.add_argument("--call", type=int, default=0)
    pb.add_argument("--first", type=int, default=0)
    pb.add_argument("--last", type=int, default=LAYERS)
    ps = sub.add_parser("sample")
    for n in ("dump", "embeds", "transformer", "lora", "out"):
        ps.add_argument(n)
    for n in ("lt", "lh", "lw", "n_audio"):
        ps.add_argument(n, type=int)
    ps.add_argument("--points", type=int, default=5)
    ps.add_argument("--shifts", type=float, nargs=2, default=[6.0, 3.0])
    ps.add_argument("--layers", type=int, default=0)
    ps.add_argument("--start", type=int, default=0)
    ps.add_argument("--stop", type=int, default=1 << 30)
    args = ap.parse_args()
    {"blocks": blocks, "sample": sample}[args.mode](args)


if __name__ == "__main__":
    main()
