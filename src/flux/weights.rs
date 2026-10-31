//! Weight loading for the FLUX transformer and the T5 encoder: linear layers from a
//! [`ModelFiles`] checkpoint (diffusers safetensors shards, or a GGUF file such as the
//! city96 / QuantStack FLUX quants), borrowed in place where the stored format runs as is
//! (bf16, any GGUF block format) and quantized at load otherwise (`Quant::Q8` / `Q4`).
//! Fused matrices (BFL's `qkv`, `linear1`) are split by rows into separate layers without
//! copying: every storage format keeps each row contiguous.

use crate::kernels::for_chunks;
use crate::model::{Lin, Quant, linear_multi};
use crate::quant::{PageVec, QBLOCK, quantize_bf16, quantize_q4};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

/// A linear layer `[n_out, n_in]` with an optional f32 bias.
pub struct Dense<'a> {
    pub w: Lin<'a>,
    pub b: Option<Vec<f32>>,
    pub n_out: usize,
    pub n_in: usize,
}

impl Dense<'_> {
    /// `x` is `[rows, n_in]`; returns `[rows, n_out]`.
    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows * self.n_out];
        self.forward_into(x, rows, &mut y);
        y
    }

    pub fn forward_into(&self, x: &[f32], rows: usize, y: &mut [f32]) {
        debug_assert_eq!(x.len(), rows * self.n_in);
        linear_multi(&mut [(&self.w, self.n_out, &mut y[..])], self.n_in, x, rows);
        self.add_bias(y, rows);
    }

    fn add_bias(&self, y: &mut [f32], rows: usize) {
        if let Some(b) = &self.b {
            for_chunks(y, self.n_out, rows > 64, |_, r| {
                for (v, &bb) in r.iter_mut().zip(b) {
                    *v += bb;
                }
            });
        }
    }

    /// Bytes held by the weight (borrowed or owned).
    pub fn bytes(&self) -> usize {
        self.w.bytes() + self.b.as_ref().map_or(0, |b| b.len() * 4)
    }
}

/// Several layers over one shared input in one batch (one GPU submission when the backend takes
/// them): returns one output per layer.
pub fn forward_many(ls: &[&Dense], x: &[f32], rows: usize) -> Vec<Vec<f32>> {
    let n_in = ls[0].n_in;
    let mut ys: Vec<Vec<f32>> = ls.iter().map(|l| vec![0f32; rows * l.n_out]).collect();
    {
        let mut jobs: Vec<(&Lin, usize, &mut [f32])> = ls.iter().zip(ys.iter_mut()).map(|(l, y)| (&l.w, l.n_out, &mut y[..])).collect();
        linear_multi(&mut jobs, n_in, x, rows);
    }
    for (l, y) in ls.iter().zip(ys.iter_mut()) {
        l.add_bias(y, rows);
    }
    ys
}

/// Reads layers out of one checkpoint at precision `quant` (for weights stored as float).
pub struct Loader<'f> {
    pub files: &'f ModelFiles,
    pub quant: Quant,
}

impl<'f> Loader<'f> {
    pub fn has(&self, name: &str) -> bool {
        self.files.has(name)
    }

    /// The `[rows, cols]` shape of a 2-D tensor (GGUF shapes are given in PyTorch order).
    pub fn shape2(&self, name: &str) -> Result<(usize, usize)> {
        let info = self.files.info(name).with_context(|| format!("missing tensor {name}"))?;
        match info.shape[..] {
            [r, c] => Ok((r, c)),
            [r, c, 1, 1] => Ok((r, c)),
            ref s => bail!("tensor {name}: expected a matrix, got shape {s:?}"),
        }
    }

    /// A float tensor as f32 (norm weights, biases).
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        if let Some(g) = self.files.ggml(name) {
            // Quantized vectors (rare): dequantize the single row.
            let (kind, bytes) = g?;
            let n = self.files.info(name).map(|i| i.shape.iter().product()).unwrap_or(0);
            let mut out = vec![0f32; n];
            crate::ggml::dequantize_row(kind, bytes, &mut out);
            return Ok(out);
        }
        self.files.f32(name)
    }

    /// Rows `r0..r1` of the matrix `{p}.weight` (and of `{p}.bias` when present) as a layer,
    /// quantized to `self.quant` if `quantize` and the stored format is a float type.
    pub fn rows(&self, p: &str, r0: usize, r1: usize, quantize: bool) -> Result<Dense<'f>> {
        let wn = format!("{p}.weight");
        let (rows, n_in) = self.shape2(&wn)?;
        if r1 > rows || r0 >= r1 {
            bail!("{wn}: rows {r0}..{r1} out of {rows}");
        }
        let n = r1 - r0;
        let w = self.lin_rows(&wn, rows, n_in, r0, r1, quantize)?;
        let bn = format!("{p}.bias");
        let b = if self.files.has(&bn) {
            let b = self.f32(&bn)?;
            if b.len() != rows {
                bail!("{bn}: {} values for {rows} rows", b.len());
            }
            Some(b[r0..r1].to_vec())
        } else {
            None
        };
        Ok(Dense { w, b, n_out: n, n_in })
    }

    /// The whole matrix `{p}.weight` (+ bias) as a layer.
    pub fn dense(&self, p: &str, quantize: bool) -> Result<Dense<'f>> {
        let (rows, _) = self.shape2(&format!("{p}.weight"))?;
        self.rows(p, 0, rows, quantize)
    }

    /// `parts` equal row blocks of `{p}.weight` (a fused projection such as `qkv`).
    pub fn split(&self, p: &str, parts: &[usize], quantize: bool) -> Result<Vec<Dense<'f>>> {
        let mut out = Vec::with_capacity(parts.len());
        let mut r = 0;
        for &n in parts {
            out.push(self.rows(p, r, r + n, quantize)?);
            r += n;
        }
        Ok(out)
    }

    fn lin_rows(&self, name: &str, rows: usize, n_in: usize, r0: usize, r1: usize, quantize: bool) -> Result<Lin<'f>> {
        let files = self.files;
        let n = r1 - r0;
        if let Some(g) = files.ggml(name) {
            let (kind, bytes) = g?;
            let rb = kind.row_bytes(n_in);
            if bytes.len() != rows * rb {
                bail!("tensor {name}: {} bytes for {rows} rows of {rb}", bytes.len());
            }
            return Lin::from_gguf(kind, &bytes[r0 * rb..r1 * rb], n, n_in).with_context(|| format!("tensor {name}"));
        }
        let info = files.info(name).with_context(|| format!("missing tensor {name}"))?;
        let q = if quantize && n_in.is_multiple_of(QBLOCK) {
            self.quant
        } else {
            Quant::Bf16
        };
        let quantized = |w: &[u16]| match q {
            Quant::Bf16 => unreachable!(),
            Quant::Q8 => Lin::q8(quantize_bf16(w, n, n_in)),
            Quant::Q4 => Lin::q4(quantize_q4(w, n, n_in)),
        };
        match info.dtype.as_str() {
            "BF16" => {
                let w = &files.bf16(name)?[r0 * n_in..r1 * n_in];
                Ok(if q == Quant::Bf16 { Lin::bf16(w) } else { quantized(w) })
            }
            "F16" | "F32" => {
                // Converted rows only (ModelFiles::bf16 would keep a converted copy of the whole
                // tensor for the life of the checkpoint).
                let b = files.bytes(name)?;
                let mut v = PageVec::<u16>::zeroed(n * n_in);
                let out: &mut [u16] = &mut v;
                if info.dtype == "F16" {
                    let src = &b[r0 * n_in * 2..r1 * n_in * 2];
                    convert_par(out, |i| {
                        half::bf16::from_f32(half::f16::from_le_bytes([src[2 * i], src[2 * i + 1]]).to_f32()).to_bits()
                    });
                } else {
                    let src = &b[r0 * n_in * 4..r1 * n_in * 4];
                    convert_par(out, |i| {
                        half::bf16::from_f32(f32::from_le_bytes([src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3]])).to_bits()
                    });
                }
                Ok(if q == Quant::Bf16 { Lin::bf16_owned(v) } else { quantized(&v) })
            }
            d => bail!("tensor {name}: dtype {d} is not supported"),
        }
    }
}

/// `out[i] = f(i)` over the thread pool.
fn convert_par(out: &mut [u16], f: impl Fn(usize) -> u16 + Sync) {
    const CH: usize = 1 << 16;
    let n = out.len();
    let p = SendPtr(out.as_mut_ptr());
    crate::pool::run(n.div_ceil(CH), &|c| {
        let (a, b) = (c * CH, ((c + 1) * CH).min(n));
        // SAFETY: chunks are disjoint and `out` outlives the call.
        let o = unsafe { std::slice::from_raw_parts_mut(p.get().add(a), b - a) };
        for (k, v) in o.iter_mut().enumerate() {
            *v = f(a + k);
        }
    });
}

#[derive(Clone, Copy)]
struct SendPtr(*mut u16);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(self) -> *mut u16 {
        self.0
    }
}
