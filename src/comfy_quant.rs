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
//!   `W x = W_rot (H x)`, the input rotated first.
//! - `w6a8_int8` (`tensor/w4a8_int8.py`, `backends/eager/w4a8_int8.py`): 6-bit codes packed into
//!   int8 `[N, 3K/4]` (a `K/2`-byte plane of low nibbles, column `2i` in the low nibble of byte
//!   `i`, then a `K/4`-byte plane of the top two bits, column `c` in byte `c / 4` at bit
//!   `2 (c % 4)`), an FP8 E4M3 scale per group of `group_size` columns (`weight_s_rel`) and a
//!   float32 scale per row (`weight_s_channel`). A weight is
//!   `clamp(round((code - 32) s_rel), -127, 127) s_channel`: the int8 value the reference's
//!   kernels compute with, on the ConvRot rotation above. Decoded to that int8 matrix at load
//!   (8 bits per weight in memory for 6 in the file).
//!
//! Every product runs in float32 on the decoded weights, which is what ComfyUI computes where it
//! has no kernel for a format (its "emulated ops", `full_precision_matrix_mult`). ComfyUI's int8
//! kernels also round the rotated input to int8 with one scale per row (the "A8" of W6A8): that is
//! not reproduced. `float8_e4m3fn` entries are left to [`crate::model::fp8_scaled`]; the other
//! formats are refused.

use crate::ggml::{GgufMatrix, e4m3_to_f32};
use crate::hadamard::Fold;
use crate::model::Lin;
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
        "float8_e4m3fn" | "float8_e5m2" | "mxfp8" => return Ok(None),
        f => bail!("{full}: ComfyUI's {f} weights are not supported"),
    };
    let mut l = Lin::gguf(m);
    l.fold = c.convrot.map(|block| rotation(full, block, n_in)).transpose()?;
    Ok(Some(l))
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
