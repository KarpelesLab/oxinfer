//! ComfyUI's ConvRot block weights (arXiv 2512.03673) in the MOSS-Image transformer: the
//! `kromoss_int8_convrot` single files (`lodestones/Kromoss/diffusion_models/`, made by the
//! release's `quantize_convrot.py`) and molbal's `Q8_CR` / `Q4_CR_L` GGUFs (`molbal/Kromoss-GGUF`).
//!
//! A module's entry, the bytes of a JSON object in tensor `{module}.comfy_quant` (safetensors) or
//! GGUF metadata `comfy.gguf.quant.{module}.weight`, says how `{module}.weight` is stored:
//!
//! - `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`: int8 `[n_out,
//!   n_in]` times `{module}.weight_scale` (one float32 per row, or one for the tensor);
//! - `{"format": "int4_cr", "convrot_groupsize": 256, "orig_shape": [n_out, n_in], ...}`: int4
//!   codes two per byte `[n_out, n_in / 2]` (column `2 i` in the low nibble of byte `i`, two's
//!   complement, -8..7), times a float16 scale per row (ComfyUI-GGUF's Q4_CR, the `w4a4` and
//!   `w4a8` backings store the same codes; `quant_group_size` only names the kernel's tiling).
//!
//! Both are rotated: the stored rows are `W H` over consecutive groups of `convrot_groupsize`
//! inputs, `H` the normalized regular Hadamard matrix ([`crate::hadamard::fwht_regular`],
//! symmetric and its own inverse), so `W x = (W H) (H x)`: the input is rotated, the weight
//! used as stored. int8 rows become [`crate::quant::QMatrix`] int8 blocks (borrowed from the
//! file, the row scale repeated per block of 32); int4 rows ggml `Q4_0` blocks (the row scale as
//! each block's float16 scale): exactly the stored values, on every backend.

use crate::flux::weights::Dense;
use crate::model::Lin;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use std::sync::Arc;

/// A module's storage.
pub struct Conf {
    int4: bool,
    /// Rotation group (None: stored as is).
    pub rot: Option<usize>,
    n_in: Option<usize>,
}

/// The entry of `module` (`layers.N.proj.qkv`), if it has one.
pub fn conf(files: &ModelFiles, module: &str) -> Result<Option<Conf>> {
    let ct = format!("{module}.comfy_quant");
    let v: serde_json::Value = if files.has(&ct) {
        serde_json::from_slice(files.bytes(&ct)?).with_context(|| format!("{ct}: not JSON"))?
    } else if let Some(s) = files.metadata(&format!("comfy.gguf.quant.{module}.weight")) {
        serde_json::from_str(s).with_context(|| format!("comfy.gguf.quant.{module}.weight: not JSON"))?
    } else {
        return Ok(None);
    };
    let key = |k: &str| v.get(k).or_else(|| v.get("params").and_then(|p| p.get(k)));
    let format = key("format").and_then(|f| f.as_str()).unwrap_or_default();
    let group = key("convrot_groupsize").and_then(|x| x.as_u64()).unwrap_or(256) as usize;
    let (int4, rotated) = match format {
        "int8_tensorwise" => (false, key("convrot").and_then(|x| x.as_bool()).unwrap_or(false)),
        "int4_cr" => (true, true),
        f => bail!("{module}: ComfyUI's {f:?} weights are not supported (int8_tensorwise and int4_cr are)"),
    };
    if rotated && (group < 4 || !group.is_power_of_two() || !group.trailing_zeros().is_multiple_of(2)) {
        bail!("{module}: ConvRot groups of {group} (a power of 4 is expected)");
    }
    let n_in = key("orig_shape")
        .and_then(|s| s.get(1))
        .and_then(|x| x.as_u64())
        .map(|x| x as usize);
    Ok(Some(Conf {
        int4,
        rot: rotated.then_some(group),
        n_in,
    }))
}

/// The input transform of a weight rotated over groups of `group`.
pub fn fold(group: usize) -> Arc<crate::hadamard::Fold> {
    Arc::new(crate::hadamard::Fold {
        block: group,
        signs: Vec::new(),
        perm: None,
        regular: true,
    })
}

/// Rotates every group of `group` values of `w` (rows of a multiple of it) in place; the
/// transform is its own inverse. Nothing without a group.
pub fn rotate_rows(w: &mut [f32], group: Option<usize>) {
    if let Some(g) = group {
        crate::kernels::for_chunks(w, g * 64, w.len() > g * 256, |_, c| {
            c.chunks_exact_mut(g).for_each(crate::hadamard::fwht_regular);
        });
    }
}

/// Rows `r0..r1` of `{module}.weight` as a layer (without the input transform; the caller sets
/// it).
pub fn rows<'f>(files: &'f ModelFiles, module: &str, c: &Conf, r0: usize, r1: usize) -> Result<Dense<'f>> {
    let wn = format!("{module}.weight");
    let info = files.info(&wn).with_context(|| format!("missing tensor {wn}"))?;
    let [rows, cols] = info.shape[..] else {
        bail!("{wn}: shape {:?}, expected a matrix", info.shape);
    };
    let n_in = if c.int4 { c.n_in.unwrap_or(2 * cols) } else { cols };
    let row_bytes = if c.int4 { n_in / 2 } else { n_in };
    let bytes = files.bytes(&wn)?;
    if bytes.len() != rows * row_bytes || cols != row_bytes || r1 > rows || r0 >= r1 {
        bail!(
            "{wn}: {} bytes {:?} for rows {r0}..{r1} of {n_in} {} values",
            bytes.len(),
            info.shape,
            if c.int4 { "int4" } else { "int8" }
        );
    }
    if !n_in.is_multiple_of(crate::quant::QBLOCK) || c.rot.is_some_and(|g| !n_in.is_multiple_of(g)) {
        bail!("{wn}: {n_in} inputs (a multiple of 32 and of the rotation group is expected)");
    }
    let sn = format!("{module}.weight_scale");
    let s = files.f32(&sn)?;
    let scale = |r: usize| if s.len() == 1 { s[0] } else { s[r] };
    if s.len() != 1 && s.len() != rows {
        bail!("{sn}: {} scales for {rows} rows", s.len());
    }
    let n = r1 - r0;
    let bpr = n_in / crate::quant::QBLOCK;
    let w = if c.int4 {
        // ggml Q4_0: per block of 32 a float16 scale, then 16 bytes: value j in the low nibble of
        // byte j, value j + 16 in the high one, both biased by 8.
        let src = &bytes[r0 * row_bytes..r1 * row_bytes];
        let mut out = crate::quant::PageVec::<u8>::zeroed(n * bpr * 18);
        let o = SendPtr(out.as_mut_ptr());
        crate::pool::run(n, &|r| {
            // SAFETY: rows are disjoint and `out` outlives the call.
            let ob = unsafe { std::slice::from_raw_parts_mut(o.get().add(r * bpr * 18), bpr * 18) };
            let row = &src[r * row_bytes..(r + 1) * row_bytes];
            let d = half::f16::from_f32(scale(r0 + r)).to_le_bytes();
            for (b, blk) in ob.as_chunks_mut::<18>().0.iter_mut().enumerate() {
                blk[..2].copy_from_slice(&d);
                let code = |j: usize| {
                    let c = b * 32 + j;
                    // Two's complement nibble to unsigned (v + 8).
                    ((row[c / 2] >> (4 * (c % 2))) & 15) ^ 8
                };
                for j in 0..16 {
                    blk[2 + j] = code(j) | (code(j + 16) << 4);
                }
            }
        });
        let m = crate::ggml::GgufMatrix::new(crate::ggml::GgmlType::Q4_0, out.into(), n, n_in)?;
        Lin::gguf(m)
    } else {
        let mut sc = crate::quant::PageVec::<f32>::zeroed(n * bpr);
        for (r, c) in sc.chunks_mut(bpr).enumerate() {
            c.fill(scale(r0 + r));
        }
        let q = &bytes[r0 * n_in..r1 * n_in];
        // SAFETY: i8 and u8 have the same layout; the matrix borrows `files` for 'f.
        let q: &'f [i8] = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const i8, q.len()) };
        Lin::q8(crate::quant::QMatrix {
            q: unsafe { crate::quant::QBuf::borrowed(q) },
            scales: sc.into(),
            n_out: n,
            n_in,
        })
    };
    files.note_use(&wn, true, false);
    Ok(Dense {
        w,
        b: None,
        n_out: n,
        n_in,
    })
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

#[cfg(test)]
mod tests {
    #[test]
    fn rotation_is_its_own_inverse() {
        let x: Vec<f32> = (0..512).map(|i| ((i * 37 + 5) % 19) as f32 - 9.0).collect();
        let mut y = x.clone();
        super::rotate_rows(&mut y, Some(256));
        assert!(y.iter().zip(&x).any(|(a, b)| (a - b).abs() > 1e-3));
        super::rotate_rows(&mut y, Some(256));
        assert!(y.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-4));
    }
}
