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
//! - `asym_w4a8_int8` (the same files; `*_w4a8.safetensors`, e.g. SulphurAI's Sulphur-3): the
//!   same storage at the width the packed row implies (`K bits / 8` bytes for `K / group_size`
//!   scales; comfy_kitchen's `_w4a8_geometry`). 4 bits: one nibble plane, the code indexing
//!   `weight_codebook` (16 float32 levels; comfy_kitchen's frozen Lloyd-Max table when absent);
//!   3 bits: a 2-bit plane then a 1-bit plane into an 8-level codebook; 5 bits: the nibble plane
//!   then a 1-bit plane, uniform `code - 16`. A weight is `clamp(round(level s_rel), -127, 127)
//!   s_channel` in the ConvRot basis, decoded to that int8 matrix at load like W6A8. A
//!   `weight_correction` (an asymmetric zero point per group) is refused.
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
//! ones to whatever read or refused them before; `convrot_w4a4` is refused.

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
    let v: serde_json::Value = if files.has(&name) {
        serde_json::from_slice(files.bytes(&name)?).with_context(|| format!("{name}: not JSON"))?
    } else {
        // Files written without the per-module tensors keep the entries in the header
        // (`_quantization_metadata`: `{"layers": {module: entry}}`), which ComfyUI's loader turns
        // into `comfy_quant` tensors (Sulphur-3's `*_w4a8.safetensors`).
        let Some(m) = files.metadata("_quantization_metadata") else {
            return Ok(None);
        };
        let all: serde_json::Value = serde_json::from_str(m).context("_quantization_metadata: not JSON")?;
        match all.get("layers").and_then(|l| l.get(base)) {
            Some(e) => e.clone(),
            None => return Ok(None),
        }
    };
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

/// comfy_kitchen's frozen 16-level Lloyd-Max table (`tensor/w4a8_packed.py`, `_FIXED_LUT`), the
/// levels of a 4-bit weight that stores no `weight_codebook` of its own.
const FIXED_LUT16: [f32; 16] = [
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720, 0.052541, 0.156985, 0.265284, 0.379533,
    0.502636, 0.638953, 0.794876, 0.980671,
];

/// The code of column `col` of a packed row of `k` columns at `bits` bits (the bit planes of the
/// module docs: comfy_kitchen's `_unpack_codes`).
fn grouped_code(row: &[u8], k: usize, bits: usize, col: usize) -> usize {
    let plane = |off: usize, width: usize| {
        let per = 8 / width;
        ((row[off + col / per] >> (width * (col % per))) as usize) & ((1 << width) - 1)
    };
    match bits {
        3 => plane(0, 2) | (plane(k / 4, 1) << 2),
        4 => plane(0, 4),
        5 => plane(0, 4) | (plane(k / 2, 1) << 4),
        _ => plane(0, 4) | (plane(k / 2, 2) << 4),
    }
}

/// A grouped W3A8 / W4A8 / W5A8 / W6A8 matrix `full` (`w6a8_int8`, `asym_w4a8_int8`) decoded to
/// the int8 values the reference computes with, times its row scales (see the module docs). The
/// bit width is the packed row's, `[N, K bits / 8]`, as comfy_kitchen infers it.
fn grouped_matrix(files: &ModelFiles, full: &str, base: &str, c: &Conf, n_out: usize, n_in: usize) -> Result<GgufMatrix> {
    let g = c.group_size;
    let (rn, cn, bn) = (
        format!("{base}.weight_s_rel"),
        format!("{base}.weight_s_channel"),
        format!("{base}.weight_codebook"),
    );
    let (wi, ri) = (
        files.info(full).with_context(|| format!("missing tensor {full}"))?,
        files.info(&rn).with_context(|| format!("missing tensor {rn}"))?,
    );
    if g == 0 || !n_in.is_multiple_of(g) || wi.dtype != "I8" || wi.shape.len() != 2 || wi.shape[0] != n_out {
        bail!(
            "{full}: {} {:?} for a grouped [{n_out}, {n_in}] weight in groups of {g}",
            wi.dtype,
            wi.shape
        );
    }
    let rb = wi.shape[1];
    let bits = rb * 8 / n_in;
    let ng = n_in / g;
    if !(3..=6).contains(&bits) || bits * n_in != rb * 8 || ri.shape != [n_out, ng] || !matches!(ri.dtype.as_str(), "F8_E4M3" | "U8") {
        bail!(
            "{full}: {} {:?} with {} {:?} group scales for a [{n_out}, {n_in}] weight of 3 to 6 bits in groups of {g}",
            wi.dtype,
            wi.shape,
            ri.dtype,
            ri.shape
        );
    }
    // The multi-plane widths keep a group inside one decode vector (`_check_plane_layout`).
    if bits != 4 && (!g.is_multiple_of(16) || !n_in.is_multiple_of(if bits == 6 { 32 } else { 64 })) {
        bail!("{full}: {bits}-bit rows of {n_in} in groups of {g} (K divisible by 32 or 64, groups a multiple of 16)");
    }
    if files.has(&format!("{base}.weight_correction")) {
        bail!("{full}: a grouped weight with a weight_correction (an asymmetric zero point) is not supported");
    }
    // 3- and 4-bit codes index a table, 5- and 6-bit ones are uniform around the middle code.
    let has_book = files.has(&bn);
    let levels: Vec<f32> = match (bits, has_book) {
        (3 | 4, true) => {
            let t = files.f32(&bn)?;
            if t.len() != 1 << bits {
                bail!("{bn}: {} levels for {bits}-bit codes", t.len());
            }
            t
        }
        (4, false) => FIXED_LUT16.to_vec(),
        (3, false) => bail!("{full}: a 3-bit weight without its weight_codebook"),
        (_, true) => bail!("{full}: a {bits}-bit weight with a codebook"),
        (_, false) => (0..1usize << bits).map(|q| q as f32 - (1usize << (bits - 1)) as f32).collect(),
    };
    let s_channel = files.f32(&cn)?;
    if s_channel.len() != n_out {
        bail!("{cn}: {} scales for {n_out} rows", s_channel.len());
    }
    let (packed, s_rel) = (files.bytes(full)?, files.bytes(&rn)?);
    let mut out = PageVec::<u8>::zeroed(n_out * n_in);
    let p = SendPtr(out.as_mut_ptr());
    crate::pool::run(n_out, &|r| {
        // SAFETY: rows are disjoint and `out` outlives the call.
        let o = unsafe { std::slice::from_raw_parts_mut(p.get().add(r * n_in), n_in) };
        let row = &packed[r * rb..(r + 1) * rb];
        for (gi, og) in o.chunks_mut(g).enumerate() {
            let s = e4m3_to_f32(s_rel[r * ng + gi]);
            for (j, v) in og.iter_mut().enumerate() {
                // torch.round: halves to even.
                *v = (levels[grouped_code(row, n_in, bits, gi * g + j)] * s)
                    .round_ties_even()
                    .clamp(-127.0, 127.0) as i8 as u8;
            }
        }
    });
    for n in [full, &rn, &cn] {
        files.note_use(n, true, false);
    }
    if has_book {
        files.note_use(&bn, true, false);
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

/// `W x` for `t` seeded normal input rows `x` (`[t, n_in]`) of the quantized weight `full` of
/// the safetensors file `path`, as the loaders compute it: `(n_out, n_in, x, y)`
/// (`examples/comfy_quant_check.rs` compares it with comfy_kitchen's dequantized weight).
#[doc(hidden)]
pub fn check_product(path: &std::path::Path, full: &str, t: usize, seed: u64) -> Result<(usize, usize, Vec<f32>, Vec<f32>)> {
    let files = ModelFiles::open_file(path, false)?;
    let base = full.strip_suffix(".weight").context("a .weight tensor")?;
    let c = conf(&files, base)?.with_context(|| format!("{full}: no comfy_quant entry"))?;
    let info = files.info(full).with_context(|| format!("missing tensor {full}"))?;
    let n_out = info.shape[0];
    // The logical width: the scale grid's for grouped weights.
    let n_in = match files.info(&format!("{base}.weight_s_rel")) {
        Some(r) => r.shape[1] * c.group_size,
        None => info.shape[1],
    };
    let l = lin(&files, full, n_out, n_in)?.context("not a supported comfy_quant weight")?;
    let x = crate::sd::ops::Rng::new(seed).normals(t * n_in);
    let mut y = vec![0f32; t * n_out];
    crate::model::linear(&l, n_out, n_in, &x, t, &mut y);
    Ok((n_out, n_in, x, y))
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
        "w6a8_int8" | "asym_w4a8_int8" => grouped_matrix(files, full, base, &c, n_out, n_in)?,
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

    /// comfy_kitchen's `_pack_codes`: the planes of `grouped_code`, written the other way round.
    fn pack(codes: &[usize], bits: usize) -> Vec<u8> {
        let k = codes.len();
        let mut out = vec![0u8; k * bits / 8];
        let mut put = |off: usize, width: usize, shift: usize| {
            let per = 8 / width;
            for (c, &q) in codes.iter().enumerate() {
                out[off + c / per] |= (((q >> shift) & ((1 << width) - 1)) as u8) << (width * (c % per));
            }
        };
        match bits {
            3 => {
                put(0, 2, 0);
                put(k / 4, 1, 2);
            }
            4 => put(0, 4, 0),
            5 => {
                put(0, 4, 0);
                put(k / 2, 1, 4);
            }
            _ => {
                put(0, 4, 0);
                put(k / 2, 2, 4);
            }
        }
        out
    }

    #[test]
    fn grouped_codes_round_trip() {
        let k = 128;
        for bits in 3..=6 {
            let codes: Vec<usize> = (0..k).map(|c| (c * 29 + 7) % (1 << bits)).collect();
            let row = pack(&codes, bits);
            assert_eq!(row.len(), k * bits / 8);
            let got: Vec<usize> = (0..k).map(|c| super::grouped_code(&row, k, bits, c)).collect();
            assert_eq!(got, codes, "{bits} bits");
        }
    }
}
