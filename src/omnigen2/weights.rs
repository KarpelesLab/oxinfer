//! Linear layers of the OmniGen2 transformer from a diffusers checkpoint (the published one is
//! float32) or a GGUF file, bf16 or quantized at load.
//!
//! The transformer is 2520 wide, a multiple neither of the 32-element blocks of the Q8 / Q4
//! formats nor of the 64-row tiles of the GPU matmuls: layers are stored with their input padded
//! by zero columns (to 2528 when quantized or on a GPU) and their output by zero rows (to a
//! multiple of 64 on a GPU); inputs are padded the same way at run time and the extra outputs
//! dropped, an exact product either way. Float32 weights are converted row by row into owned
//! buffers, so a mapped checkpoint never keeps a converted copy of its own.

use crate::model::{Lin, Quant};
use crate::quant::{PageVec, QBLOCK, quantize_bf16, quantize_q4};
use crate::qwen_image::weights::Dense;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

/// Output rows per GPU matmul tile.
const GPU_ROWS: usize = 64;

/// A linear layer `[n_out, n_in]` stored as `[d.n_out, d.n_in]` (zero padded, `>=` each).
pub struct Linear<'a> {
    pub d: Dense<'a>,
    pub n_in: usize,
    pub n_out: usize,
}

/// `x` (`[rows, n]`) with each row zero-padded to `k` columns.
fn pad_cols(x: &[f32], rows: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * k];
    crate::kernels::for_chunks(&mut y, k, rows > 64, |r, o| o[..n].copy_from_slice(&x[r * n..(r + 1) * n]));
    y
}

/// The first `n` of every `k` columns of `y` into `out` (`[rows, n]`).
fn strip_cols(y: &[f32], k: usize, n: usize, out: &mut [f32]) {
    let rows = out.len() / n;
    crate::kernels::for_chunks(out, n, rows > 64, |r, o| o.copy_from_slice(&y[r * k..r * k + n]));
}

impl Linear<'_> {
    /// `x` is `[rows, n_in]`; returns `[rows, n_out]`.
    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows * self.n_out];
        Self::forward_multi(&mut [(self, &mut y[..])], x, rows);
        y
    }

    /// For a few rows (modulation vectors): full-precision activations whatever the weights (see
    /// [`Dense::forward_small`]).
    pub fn forward_small(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let xp;
        let x = if self.d.n_in != self.n_in {
            xp = pad_cols(x, rows, self.n_in, self.d.n_in);
            &xp[..]
        } else {
            x
        };
        let y = self.d.forward_small(x, rows);
        if self.d.n_out == self.n_out {
            return y;
        }
        let mut o = vec![0f32; rows * self.n_out];
        strip_cols(&y, self.d.n_out, self.n_out, &mut o);
        o
    }

    /// Several layers over one input (all with the same `n_in` and input padding); each `y` is
    /// `[rows, n_out]`.
    pub fn forward_multi(jobs: &mut [(&Self, &mut [f32])], x: &[f32], rows: usize) {
        let l0 = jobs[0].0;
        let xp;
        let x = if l0.d.n_in != l0.n_in {
            xp = pad_cols(x, rows, l0.n_in, l0.d.n_in);
            &xp[..]
        } else {
            x
        };
        let mut tmp: Vec<Vec<f32>> = jobs
            .iter()
            .map(|(l, _)| {
                if l.d.n_out != l.n_out {
                    vec![0f32; rows * l.d.n_out]
                } else {
                    Vec::new()
                }
            })
            .collect();
        {
            let mut dj: Vec<(&Dense, &mut [f32])> = jobs
                .iter_mut()
                .zip(tmp.iter_mut())
                .map(|((l, y), t)| (&l.d, if t.is_empty() { &mut **y } else { &mut t[..] }))
                .collect();
            Dense::forward_multi(&mut dj, x, rows);
        }
        for ((l, y), t) in jobs.iter_mut().zip(&tmp) {
            if !t.is_empty() {
                strip_cols(t, l.d.n_out, l.n_out, y);
            }
        }
    }

    pub fn bytes(&self) -> usize {
        self.d.w.bytes() + self.d.b.as_ref().map_or(0, |b| b.len() * 4)
    }
}

/// Reads layers of one checkpoint at precision `quant` (GGUF tensors keep their own format).
pub struct Loader<'a> {
    pub files: &'a ModelFiles,
    pub quant: Quant,
    pub prefix: String,
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

impl<'a> Loader<'a> {
    fn full(&self, name: &str) -> String {
        format!("{}{name}", self.prefix)
    }

    pub fn has(&self, name: &str) -> bool {
        self.files.has(&self.full(name))
    }

    /// A float vector (norm weights, biases, embeddings) as f32, checked to hold `n` values.
    pub fn f32(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        let full = self.full(name);
        let v = if let Some(g) = self.files.ggml(&full) {
            let (kind, bytes) = g?;
            let mut out = vec![0f32; n];
            if bytes.len() != kind.row_bytes(n) {
                bail!("tensor {full}: {} bytes for {n} values of {kind:?}", bytes.len());
            }
            crate::ggml::dequantize_row(kind, bytes, &mut out);
            out
        } else {
            self.files.f32(&full)?
        };
        if v.len() != n {
            bail!("tensor {full} has {} values, expected {n}", v.len());
        }
        Ok(v)
    }

    /// `{name}.weight` `[n_out, n_in]` and `{name}.bias` when present.
    pub fn linear(&self, name: &str, n_out: usize, n_in: usize) -> Result<Linear<'a>> {
        let wn = self.full(&format!("{name}.weight"));
        let files = self.files;
        let gpu = crate::gpu::active().is_some();
        let (w, k_in, k_out) = if let Some(g) = files.ggml(&wn) {
            let (kind, bytes) = g?;
            if bytes.len() != n_out * kind.row_bytes(n_in) {
                bail!("tensor {wn}: {} bytes for [{n_out}, {n_in}] {kind:?}", bytes.len());
            }
            (
                Lin::from_gguf(kind, bytes, n_out, n_in).with_context(|| format!("tensor {wn}"))?,
                n_in,
                n_out,
            )
        } else {
            let info = files.info(&wn).with_context(|| format!("missing tensor {wn}"))?;
            if info.shape != [n_out, n_in] {
                bail!("tensor {wn} has shape {:?}, expected [{n_out}, {n_in}]", info.shape);
            }
            let k_in = if self.quant != Quant::Bf16 || gpu {
                n_in.next_multiple_of(QBLOCK)
            } else {
                n_in
            };
            let k_out = if gpu { n_out.next_multiple_of(GPU_ROWS) } else { n_out };
            let dtype = info.dtype.clone();
            if dtype == "BF16" && self.quant == Quant::Bf16 && (k_in, k_out) == (n_in, n_out) {
                (Lin::bf16(files.bf16(&wn)?), n_in, n_out)
            } else {
                let b = files.bytes(&wn)?;
                let mut v = PageVec::<u16>::zeroed(k_out * k_in);
                let get: Box<dyn Fn(usize) -> u16 + Sync> = match dtype.as_str() {
                    "BF16" => Box::new(|i| u16::from_le_bytes([b[2 * i], b[2 * i + 1]])),
                    "F16" => Box::new(|i| half::bf16::from_f32(half::f16::from_le_bytes([b[2 * i], b[2 * i + 1]]).to_f32()).to_bits()),
                    "F32" => {
                        Box::new(|i| crate::sd::ops::f32_to_bf16(f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])))
                    }
                    d => bail!("tensor {wn}: dtype {d} is not supported"),
                };
                convert_par(&mut v, |i| {
                    let (r, c) = (i / k_in, i % k_in);
                    if r < n_out && c < n_in { get(r * n_in + c) } else { 0 }
                });
                let w = match self.quant {
                    Quant::Bf16 => Lin::bf16_owned(v),
                    Quant::Q8 => Lin::q8(quantize_bf16(&v, k_out, k_in)),
                    Quant::Q4 => Lin::q4(quantize_q4(&v, k_out, k_in)),
                };
                (w, k_in, k_out)
            }
        };
        let bn = format!("{name}.bias");
        let b = if self.has(&bn) {
            let mut b = self.f32(&bn, n_out)?;
            b.resize(k_out, 0.0);
            Some(b)
        } else {
            None
        };
        Ok(Linear {
            d: Dense {
                w,
                b,
                n_out: k_out,
                n_in: k_in,
                lora: None,
                exact: None,
            },
            n_in,
            n_out,
        })
    }
}
