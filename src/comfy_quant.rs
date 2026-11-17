//! ComfyUI's quantized safetensors weights: a module's `weight` beside a `comfy_quant` tensor (the
//! bytes of a JSON object, `{"format": ...}`) and its scales, as `comfy/ops.py` (the
//! mixed-precision operations) loads them into the layouts of the `comfy_kitchen` package. The
//! definitions below are those of comfy_kitchen's eager (reference) backend:
//!
//! - `int8_tensorwise` (`tensor/int8.py`, `backends/eager/quantization.py`): int8 `[N, K]` times
//!   `weight_scale`, a scalar or one value per row (`[N, 1]`). With `"convrot": true` the rows are
//!   stored rotated, `W_rot = W H^T` over consecutive groups of `convrot_groupsize` inputs (256
//!   unless the entry says otherwise; MiniMax-H3's AdaLN projections use 64), `H` the normalized
//!   regular Hadamard matrix ([`crate::hadamard::fwht_regular`]): the product is
//!   `W x = W_rot (H x)`, the input rotated first. An embedding table is the same, its rows
//!   rotated back after the lookup.
//! - `w6a8_int8` (`tensor/w4a8_int8.py`, `backends/eager/w4a8_int8.py`): 6-bit codes packed into
//!   int8 `[N, 3K/4]` (a `K/2`-byte plane of low nibbles, column `2i` in the low nibble of byte
//!   `i`, then a `K/4`-byte plane of the top two bits, column `c` in byte `c / 4` at bit
//!   `2 (c % 4)`), an FP8 E4M3 scale per group of `group_size` columns (`weight_s_rel`) and a
//!   float32 scale per row (`weight_s_channel`). A weight is
//!   `clamp(round((code - 32) s_rel), -127, 127) s_channel`: the int8 value the reference's
//!   kernels compute with, on the ConvRot rotation above. Decoded to that int8 matrix at load
//!   (8 bits per weight in memory for 6 in the file).
//! - `nvfp4` (`tensor/nvfp4.py`): E2M1 pairs `[N, K/2]`, the first value of a pair in the high
//!   nibble (ModelOpt and compressed-tensors put it in the low one), an E4M3 scale per 16 values
//!   in cuBLAS's tiled order (`float_utils.to_blocked`: tiles of 128 rows by 4 blocks, stored as
//!   32 rows of 16 with the tile's four 32-row bands side by side) and a tensor-wide
//!   `weight_scale_2`; rows and columns padded to multiples of 16. `pre_quant_scale` (ModelOpt's
//!   AWQ smoothing) multiplies the input. Repacked into ggml's NVFP4 blocks, as
//!   [`crate::model::nvfp4_matrix`] does for the other packings.
//!
//! Every product runs in float32 on the decoded weights, which is what ComfyUI computes where it
//! has no kernel for a format (its "emulated ops", `full_precision_matrix_mult`; the NVFP4 text
//! encoder of MiniMax-H3 asks for it in its `comfy_quant`). ComfyUI's int8 kernels also round the
//! rotated input to int8 with one scale per row (the "A8" of W6A8): that is not reproduced, and
//! `input_scale` (a calibrated activation scale for its FP8 / NVFP4 kernels) is not read.
//! `float8_e4m3fn` entries are left to [`crate::model::fp8_scaled`], `float8_e5m2` and `mxfp8`
//! ones to whatever read or refused them before; `asym_w4a8_int8` and `convrot_w4a4` are refused.

use crate::ggml::{GgufMatrix, e4m3_to_f32};
use crate::hadamard::Fold;
use crate::model::{Lin, Table};
use crate::quant::{PageVec, QBuf};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use std::sync::Arc;

/// A module's `comfy_quant` entry.
struct Conf {
    format: String,
    /// The ConvRot group size when the weight is stored rotated.
    convrot: Option<usize>,
    /// Columns per `weight_s_rel` scale (W6A8).
    group_size: usize,
}

fn conf(files: &ModelFiles, base: &str) -> Result<Option<Conf>> {
    let name = format!("{base}.comfy_quant");
    if !files.has(&name) {
        return Ok(None);
    }
    let v: serde_json::Value = serde_json::from_slice(files.bytes(&name)?).with_context(|| format!("{name}: not JSON"))?;
    let format = v
        .get("format")
        .and_then(|f| f.as_str())
        .with_context(|| format!("{name}: no format"))?
        .to_string();
    // `comfy/ops.py` reads each key from the entry, then from its `params` object.
    let key = |k: &str| v.get(k).or_else(|| v.get("params").and_then(|p| p.get(k)));
    let size = |k: &str, d: usize| key(k).and_then(|x| x.as_u64()).map_or(d, |x| x as usize);
    let grouped = matches!(format.as_str(), "w6a8_int8" | "asym_w4a8_int8" | "convrot_w4a4");
    let convrot = (grouped || key("convrot").and_then(|x| x.as_bool()).unwrap_or(false)).then(|| size("convrot_groupsize", 256));
    Ok(Some(Conf {
        format,
        convrot,
        group_size: size("group_size", 16),
    }))
}

/// The input transform of a weight stored ConvRot-rotated over groups of `block` inputs.
fn rotation(name: &str, block: usize, n_in: usize) -> Result<Arc<Fold>> {
    if block < 4 || !block.is_power_of_two() || !block.trailing_zeros().is_multiple_of(2) || !n_in.is_multiple_of(block) {
        bail!("{name}: ConvRot groups of {block} over {n_in} inputs (a power of 4 dividing the width is expected)");
    }
    Ok(Arc::new(Fold {
        block,
        signs: Vec::new(),
        perm: None,
        regular: true,
    }))
}

fn scales(v: &[f32]) -> QBuf<f32> {
    let mut p = PageVec::<f32>::zeroed(v.len());
    p.copy_from_slice(v);
    p.into()
}

/// The int8 matrix `full` (`[n_out, n_in]`, borrowed) with its `weight_scale`.
fn int8_matrix(files: &ModelFiles, full: &str, base: &str, n_out: usize, n_in: usize) -> Result<GgufMatrix> {
    let info = files.info(full).with_context(|| format!("missing tensor {full}"))?;
    if info.dtype != "I8" || info.shape != [n_out, n_in] {
        bail!("{full}: {} {:?} for an int8 [{n_out}, {n_in}] weight", info.dtype, info.shape);
    }
    let sn = format!("{base}.weight_scale");
    let s = files.f32(&sn)?;
    // SAFETY: the matrix borrows `files`, which outlives every model holding it.
    GgufMatrix::i8(unsafe { QBuf::borrowed(files.bytes(full)?) }, scales(&s), n_out, n_in).with_context(|| sn)
}

/// The W6A8 matrix `full` decoded to the int8 values the reference computes with, times its row
/// scales (see the module docs).
fn w6a8_matrix(files: &ModelFiles, full: &str, base: &str, c: &Conf, n_out: usize, n_in: usize) -> Result<GgufMatrix> {
    let g = c.group_size;
    if g < 16 || !g.is_multiple_of(16) || !n_in.is_multiple_of(32) || !n_in.is_multiple_of(g) {
        bail!("{full}: 6-bit rows of {n_in} in groups of {g} (K divisible by 32, groups a multiple of 16)");
    }
    let (rn, cn) = (format!("{base}.weight_s_rel"), format!("{base}.weight_s_channel"));
    let (wi, ri) = (
        files.info(full).with_context(|| format!("missing tensor {full}"))?,
        files.info(&rn).with_context(|| format!("missing tensor {rn}"))?,
    );
    let ng = n_in / g;
    if wi.dtype != "I8" || wi.shape != [n_out, n_in * 3 / 4] || ri.shape != [n_out, ng] || !matches!(ri.dtype.as_str(), "F8_E4M3" | "U8") {
        bail!(
            "{full}: {} {:?} with {} {:?} group scales for a 6-bit [{n_out}, {n_in}] weight",
            wi.dtype,
            wi.shape,
            ri.dtype,
            ri.shape
        );
    }
    if files.has(&format!("{base}.weight_codebook")) || files.has(&format!("{base}.weight_correction")) {
        bail!("{full}: a 6-bit weight with a codebook or a correction");
    }
    let s_channel = files.f32(&cn)?;
    if s_channel.len() != n_out {
        bail!("{cn}: {} scales for {n_out} rows", s_channel.len());
    }
    let (packed, s_rel) = (files.bytes(full)?, files.bytes(&rn)?);
    let rb = n_in * 3 / 4;
    let mut out = PageVec::<u8>::zeroed(n_out * n_in);
    let p = SendPtr(out.as_mut_ptr());
    crate::pool::run(n_out, &|r| {
        // SAFETY: rows are disjoint and `out` outlives the call.
        let o = unsafe { std::slice::from_raw_parts_mut(p.get().add(r * n_in), n_in) };
        let row = &packed[r * rb..(r + 1) * rb];
        let (low, high) = row.split_at(n_in / 2);
        for (gi, og) in o.chunks_mut(g).enumerate() {
            let s = e4m3_to_f32(s_rel[r * ng + gi]);
            for (j, v) in og.iter_mut().enumerate() {
                let col = gi * g + j;
                let code = ((low[col / 2] >> (4 * (col % 2))) & 0xF) | (((high[col / 4] >> (2 * (col % 4))) & 3) << 4);
                // torch.round: halves to even.
                *v = ((code as f32 - 32.0) * s).round_ties_even().clamp(-127.0, 127.0) as i8 as u8;
            }
        }
    });
    for n in [full, &rn, &cn] {
        files.note_use(n, true, false);
    }
    GgufMatrix::i8(out.into(), scales(&s_channel), n_out, n_in)
}

#[derive(Clone, Copy)]
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(self) -> *mut u8 {
        self.0
    }
}

/// The NVFP4 matrix `full` in comfy_kitchen's packing, repacked into ggml's NVFP4 blocks.
fn nvfp4_matrix(files: &ModelFiles, full: &str, base: &str, n_out: usize, n_in: usize) -> Result<GgufMatrix> {
    let (sn, gn) = (format!("{base}.weight_scale"), format!("{base}.weight_scale_2"));
    let (wi, si) = (
        files.info(full).with_context(|| format!("missing tensor {full}"))?,
        files.info(&sn).with_context(|| format!("missing tensor {sn}"))?,
    );
    // Stored padded: values to multiples of 16 each way, scales to tiles of 128 rows by 4 blocks.
    let (pr, pc) = (n_out.next_multiple_of(16), n_in.next_multiple_of(16));
    let (nb, tr, tc) = (n_in / 16, pr.div_ceil(128), (pc / 16).div_ceil(4));
    if wi.dtype != "U8" || wi.shape != [pr, pc / 2] || si.shape.iter().product::<usize>() != tr * tc * 512 || !n_in.is_multiple_of(64) {
        bail!(
            "{full}: {} {:?} with scales {:?} for an NVFP4 [{n_out}, {n_in}] weight (rows of whole blocks of 64)",
            wi.dtype,
            wi.shape,
            si.shape
        );
    }
    let g = files.f32(&gn)?.first().copied().with_context(|| format!("{gn}: empty"))?;
    let (packed, tiled) = (files.bytes(full)?, files.bytes(&sn)?);
    let mut vals = vec![0u8; n_out * n_in / 2];
    let mut sc = vec![0u8; n_out * nb];
    for r in 0..n_out {
        let src = &packed[r * pc / 2..r * pc / 2 + n_in / 2];
        for (d, &b) in vals[r * n_in / 2..(r + 1) * n_in / 2].iter_mut().zip(src) {
            *d = b.rotate_left(4);
        }
        for (c, d) in sc[r * nb..(r + 1) * nb].iter_mut().enumerate() {
            *d = tiled[((r / 128) * tc + c / 4) * 512 + (r % 32) * 16 + (r / 32 % 4) * 4 + c % 4];
        }
    }
    for n in [full, &sn, &gn] {
        files.note_use(n, true, false);
    }
    crate::model::nvfp4_blocks(&vals, &sc, g, n_out, n_in)
}

/// The linear weight `full` (`[n_out, n_in]`) when its module has a `comfy_quant` entry in one of
/// the formats of the module docs; None without an entry, or for one another loader reads.
pub(crate) fn lin<'a>(files: &'a ModelFiles, full: &str, n_out: usize, n_in: usize) -> Result<Option<Lin<'a>>> {
    let Some(base) = full.strip_suffix(".weight") else {
        return Ok(None);
    };
    let Some(c) = conf(files, base)? else { return Ok(None) };
    let m = match c.format.as_str() {
        "int8_tensorwise" => int8_matrix(files, full, base, n_out, n_in)?,
        "w6a8_int8" => w6a8_matrix(files, full, base, &c, n_out, n_in)?,
        "nvfp4" => nvfp4_matrix(files, full, base, n_out, n_in)?,
        "float8_e4m3fn" | "float8_e5m2" | "mxfp8" => return Ok(None),
        f => bail!("{full}: ComfyUI's {f} weights are not supported"),
    };
    let mut l = Lin::gguf(m);
    let pn = format!("{base}.pre_quant_scale");
    l.fold = match (c.convrot, files.has(&pn)) {
        (Some(_), true) => bail!("{full}: a rotated weight with a pre_quant_scale"),
        (Some(block), false) => Some(rotation(full, block, n_in)?),
        (None, true) => {
            let signs = files.f32(&pn)?;
            if signs.len() != n_in {
                bail!("{pn}: {} values for {n_in} inputs", signs.len());
            }
            // No rotation (groups of one): the input times the scale.
            Some(Arc::new(Fold {
                block: 1,
                signs,
                perm: None,
                regular: true,
            }))
        }
        (None, false) => None,
    };
    Ok(Some(l))
}

/// The lookup table `full` (`[rows, cols]`) when it has an `int8_tensorwise` entry: int8 rows
/// times their scale, rotated back when stored rotated.
pub(crate) fn table<'a>(files: &'a ModelFiles, full: &str, rows: usize, cols: usize) -> Result<Option<Table<'a>>> {
    let Some(base) = full.strip_suffix(".weight") else {
        return Ok(None);
    };
    let Some(c) = conf(files, base)? else { return Ok(None) };
    if c.format != "int8_tensorwise" {
        bail!("{full}: a lookup table in ComfyUI's {} format is not supported", c.format);
    }
    let t = Table::Gguf(int8_matrix(files, full, base, rows, cols)?);
    Ok(Some(match c.convrot {
        Some(block) => Table::Folded(Box::new(t), rotation(full, block, cols)?),
        None => t,
    }))
}

/// The configuration of the language model in one of ComfyUI's single-file Qwen3-VL text
/// encoders (`text_encoders/qwen3vl_32b_minimax_h3_*.safetensors`: `model.layers.N.*`,
/// `model.embed_tokens`, `visual.*`, no `config.json`), as `QwenConfig::load` reads it from a
/// folder: the sizes from the tensor shapes, the rest Qwen3-VL's constants
/// (`comfy/text_encoders/llama.py`, `Qwen3VL_8BConfig`: RoPE base 5e6, epsilon 1e-6, M-RoPE
/// sections 24 / 20 / 20 of 64 frequencies). None when `files` is not such a file.
pub(crate) fn qwen3vl_config(files: &ModelFiles) -> Result<Option<serde_json::Value>> {
    let p = "model.layers.0.";
    let (embed, qn) = ("model.embed_tokens.weight", format!("{p}self_attn.q_norm.weight"));
    if !files.has(embed) || !files.has(&qn) || files.has("lm_head.weight") {
        return Ok(None);
    }
    let dim = |n: &str, i: usize| -> Result<usize> {
        files
            .info(n)
            .and_then(|t| t.shape.get(i).copied())
            .with_context(|| format!("missing tensor {n}"))
    };
    let head_dim = dim(&qn, 0)?;
    let layers = (0..)
        .take_while(|l| files.has(&format!("model.layers.{l}.input_layernorm.weight")))
        .count();
    let half = head_dim / 2;
    let hw = half * 5 / 16;
    Ok(Some(serde_json::json!({
        "model_type": "qwen3_vl_text",
        "vocab_size": dim(embed, 0)?,
        "hidden_size": dim(embed, 1)?,
        "intermediate_size": dim(&format!("{p}mlp.gate_proj.weight"), 0)?,
        "num_hidden_layers": layers,
        "num_attention_heads": dim(&format!("{p}self_attn.q_proj.weight"), 0)? / head_dim,
        "num_key_value_heads": dim(&format!("{p}self_attn.k_proj.weight"), 0)? / head_dim,
        "head_dim": head_dim,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 5_000_000.0,
            "mrope_section": [half - 2 * hw, hw, hw],
            "mrope_interleaved": true,
        },
    })))
}

#[cfg(test)]
mod tests {
    use crate::hadamard::fwht_regular;

    /// The regular Hadamard matrix by Kronecker powers, as comfy_kitchen builds it.
    fn regular(n: usize) -> Vec<f32> {
        const H4: [[f32; 4]; 4] = [[1., 1., 1., -1.], [1., 1., -1., 1.], [1., -1., 1., 1.], [-1., 1., 1., 1.]];
        let (mut h, mut m) = (vec![1f32], 1);
        while m < n {
            let mut k = vec![0f32; 16 * m * m];
            for (i, hr) in h.chunks(m).enumerate() {
                for (j, &v) in hr.iter().enumerate() {
                    for (a, row) in H4.iter().enumerate() {
                        for (b, &w) in row.iter().enumerate() {
                            k[(4 * i + a) * 4 * m + 4 * j + b] = v * w;
                        }
                    }
                }
            }
            (h, m) = (k, 4 * m);
        }
        h.iter().map(|v| v / (n as f32).sqrt()).collect()
    }

    #[test]
    fn regular_hadamard_is_the_kronecker_power() {
        for n in [1usize, 4, 16, 64, 256] {
            let h = regular(n);
            let x: Vec<f32> = (0..n).map(|i| ((i * 37 + 11) % 23) as f32 - 9.5).collect();
            let want: Vec<f32> = h.chunks(n).map(|r| r.iter().zip(&x).map(|(a, b)| a * b).sum()).collect();
            let mut got = x.clone();
            fwht_regular(&mut got);
            assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-4), "n = {n}");
            // Its own inverse.
            fwht_regular(&mut got);
            assert!(got.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-4), "n = {n}");
        }
    }
}
