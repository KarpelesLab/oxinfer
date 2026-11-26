//! The transformer's weights as stored: the bf16 shards of `base_model/`, or `base_model_int8/`
//! (Meituan's `int8_per_channel_symmetric`: every `nn.Linear` but `final_layer.linear` as
//! `weight_int8` `[out, in]` and one float32 `weight_scale` per output row, the bias in bf16),
//! read in place from the mapped files.

use crate::kernels::for_chunks;
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::Path;

/// The safetensors files of one component folder.
pub struct Shards {
    files: Vec<SafeTensors>,
    at: HashMap<String, usize>,
}

impl Shards {
    /// Every `*.safetensors` of `dir` (a tensor lives in exactly one of them).
    pub fn open(dir: &Path) -> Result<Self> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            bail!("{}: no .safetensors file", dir.display());
        }
        let mut files = Vec::with_capacity(paths.len());
        let mut at = HashMap::new();
        for (i, p) in paths.iter().enumerate() {
            let st = SafeTensors::open(p, cfg!(target_arch = "wasm32"))?;
            for n in st.names() {
                at.insert(n.clone(), i);
            }
            files.push(st);
        }
        Ok(Self { files, at })
    }

    /// One file.
    pub fn open_file(path: &Path) -> Result<Self> {
        let st = SafeTensors::open(path, cfg!(target_arch = "wasm32"))?;
        let at = st.names().map(|n| (n.clone(), 0)).collect();
        Ok(Self { files: vec![st], at })
    }

    pub fn has(&self, name: &str) -> bool {
        self.at.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.at.keys()
    }

    /// Dtype, shape and bytes of a tensor.
    pub fn raw(&self, name: &str) -> Result<(&str, &[usize], &[u8])> {
        let i = *self.at.get(name).with_context(|| format!("missing tensor {name}"))?;
        let info = self.files[i].info(name).unwrap();
        Ok((info.dtype.as_str(), &info.shape, self.files[i].bytes(name)?))
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.raw(name)?.1.to_vec())
    }

    /// A floating-point tensor as float32.
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let (dtype, shape, b) = self.raw(name)?;
        let n: usize = shape.iter().product();
        let mut out = vec![0f32; n];
        decode(dtype, b, &mut out).with_context(|| name.to_string())?;
        Ok(out)
    }

    /// The total size of the files' tensors (bytes).
    pub fn bytes(&self) -> usize {
        self.at
            .iter()
            .map(|(n, &i)| self.files[i].info(n).map_or(0, |t| t.end - t.start))
            .sum()
    }
}

/// `b` (little-endian `dtype`) as float32.
pub(crate) fn decode(dtype: &str, b: &[u8], out: &mut [f32]) -> Result<()> {
    let per = match dtype {
        "F32" => 4,
        "BF16" | "F16" => 2,
        "F64" => 8,
        t => bail!("dtype {t} is not a float format this loader reads"),
    };
    if b.len() != out.len() * per {
        bail!("{} bytes for {} {dtype} values", b.len(), out.len());
    }
    let par = out.len() >= 1 << 16;
    let chunk = if par { 1 << 14 } else { out.len().max(1) };
    for_chunks(out, chunk, par, |i, o| {
        let src = &b[i * chunk * per..];
        match dtype {
            "F32" => {
                for (v, c) in o.iter_mut().zip(src.as_chunks::<4>().0) {
                    *v = f32::from_le_bytes(*c);
                }
            }
            "BF16" => {
                for (v, c) in o.iter_mut().zip(src.as_chunks::<2>().0) {
                    *v = f32::from_bits((u16::from_le_bytes(*c) as u32) << 16);
                }
            }
            "F16" => {
                for (v, c) in o.iter_mut().zip(src.as_chunks::<2>().0) {
                    *v = half::f16::from_le_bytes(*c).to_f32();
                }
            }
            _ => {
                for (v, c) in o.iter_mut().zip(src.as_chunks::<8>().0) {
                    *v = f64::from_le_bytes(*c) as f32;
                }
            }
        }
    });
    Ok(())
}

/// How a matrix is stored.
#[derive(Clone, Copy)]
pub enum Store<'a> {
    /// A float format (`F32`, `BF16`, `F16`), `[n_out, n_in]`.
    Float(&'static str, &'a [u8]),
    /// Signed bytes `[n_out, n_in]` times one scale per output row ([`Mat::scales`]).
    Int8(&'a [u8]),
}

/// One linear layer: `y = W x + b`.
pub struct Mat<'a> {
    pub store: Store<'a>,
    pub bias: Option<Vec<f32>>,
    pub n_out: usize,
    pub n_in: usize,
    /// The int8 rows' scales (owned: the file's bytes are not aligned).
    scales: Option<Box<[f32]>>,
}

fn static_dtype(d: &str) -> Result<&'static str> {
    Ok(match d {
        "F32" => "F32",
        "BF16" => "BF16",
        "F16" => "F16",
        t => bail!("matrix dtype {t} is not supported"),
    })
}

impl<'a> Mat<'a> {
    /// `{prefix}.weight` or `{prefix}.weight_int8` + `{prefix}.weight_scale`, and `{prefix}.bias`
    /// when there is one. A kernel of more than two dimensions is read as `[out, rest]`.
    pub fn load(s: &'a Shards, prefix: &str) -> Result<Self> {
        let bias = if s.has(&format!("{prefix}.bias")) {
            Some(s.f32(&format!("{prefix}.bias"))?)
        } else {
            None
        };
        let wn = format!("{prefix}.weight");
        let qn = format!("{prefix}.weight_int8");
        let mut m = if s.has(&wn) {
            let (dtype, shape, b) = s.raw(&wn)?;
            let n_out = *shape.first().with_context(|| format!("{wn}: no shape"))?;
            let n_in = shape[1..].iter().product();
            Mat {
                store: Store::Float(static_dtype(dtype).with_context(|| wn.clone())?, b),
                bias: None,
                n_out,
                n_in,
                scales: None,
            }
        } else if s.has(&qn) {
            let (dtype, shape, b) = s.raw(&qn)?;
            if dtype != "I8" || shape.len() != 2 {
                bail!("{qn}: expected an I8 matrix, got {dtype} {shape:?}");
            }
            let (n_out, n_in) = (shape[0], shape[1]);
            let scales = s.f32(&format!("{prefix}.weight_scale"))?;
            if scales.len() != n_out {
                bail!("{prefix}.weight_scale: {} values for {n_out} rows", scales.len());
            }
            Mat {
                store: Store::Int8(b),
                bias: None,
                n_out,
                n_in,
                scales: Some(scales.into_boxed_slice()),
            }
        } else {
            bail!("missing tensor {wn} (or {qn})");
        };
        if let Some(b) = &bias
            && b.len() != m.n_out
        {
            bail!("{prefix}.bias: {} values for {} outputs", b.len(), m.n_out);
        }
        m.bias = bias;
        Ok(m)
    }

    /// The scales of an int8 matrix's rows.
    pub fn scales(&self) -> Option<&[f32]> {
        self.scales.as_deref()
    }

    pub fn is_int8(&self) -> bool {
        matches!(self.store, Store::Int8(_))
    }

    /// Bytes of the stored weights.
    pub fn bytes(&self) -> usize {
        match self.store {
            Store::Float(_, b) | Store::Int8(b) => b.len(),
        }
    }

    /// Rows `r0..r1` as float32 (`[r1 - r0, n_in]`).
    pub fn rows(&self, r0: usize, r1: usize, out: &mut [f32]) {
        let n_in = self.n_in;
        debug_assert_eq!(out.len(), (r1 - r0) * n_in);
        match self.store {
            Store::Float(dtype, b) => {
                let per = if dtype == "F32" { 4 } else { 2 };
                decode(dtype, &b[r0 * n_in * per..r1 * n_in * per], out).expect("matrix bytes");
            }
            Store::Int8(b) => {
                let sc = self.scales.as_deref().unwrap();
                for_chunks(out, n_in, (r1 - r0) * n_in >= 1 << 16, |i, o| {
                    let r = r0 + i;
                    let s = sc[r];
                    for (v, &q) in o.iter_mut().zip(&b[r * n_in..(r + 1) * n_in]) {
                        *v = (q as i8) as f32 * s;
                    }
                });
            }
        }
    }

    /// The whole matrix as float32.
    pub fn dense(&self) -> Vec<f32> {
        let mut w = vec![0f32; self.n_out * self.n_in];
        self.rows(0, self.n_out, &mut w);
        w
    }

    /// `[rows, n_out]` from `x` (`[rows, n_in]`), in float32 on the dequantized weights.
    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), rows * self.n_in);
        let w = self.dense();
        let mut y = crate::sd::ops::matmul_f32(&w, self.n_out, self.n_in, x, rows);
        if let Some(b) = &self.bias {
            add_bias(&mut y, b);
        }
        y
    }
}

/// `y[r] += b` for every row.
pub(crate) fn add_bias(y: &mut [f32], b: &[f32]) {
    let n = b.len();
    for_chunks(y, n * 16, y.len() >= 1 << 16, |_, c| {
        for r in c.chunks_mut(n) {
            for (v, &bb) in r.iter_mut().zip(b) {
                *v += bb;
            }
        }
    });
}
