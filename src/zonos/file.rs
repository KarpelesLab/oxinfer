//! Tensors and metadata of one of ZONOS3's GGUF files, read by their own names (the backbone,
//! the codec decoder and the speaker encoder each have their own architecture string and tensor
//! names, none of them llama.cpp's), and the float matmul they share.

use crate::ggml::{GgmlType, GgufMatrix};
use crate::gguf::{Header, Tensor, Value};
use anyhow::{Context, Result, bail};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub(crate) struct Gg {
    pub path: PathBuf,
    header: Arc<Header>,
    file: Mutex<std::fs::File>,
    arch: String,
}

impl Gg {
    /// Opens `path`, whose `general.architecture` must be `arch`.
    pub fn open(path: &Path, arch: &str) -> Result<Self> {
        let header = crate::gguf::header(path)?;
        if header.arch() != arch {
            bail!("{}: GGUF architecture {:?}, expected {arch:?}", path.display(), header.arch());
        }
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            header,
            file: Mutex::new(file),
            arch: arch.to_string(),
        })
    }

    fn value(&self, key: &str) -> Result<&Value> {
        self.header
            .get(&format!("{}.{key}", self.arch))
            .with_context(|| format!("{}: no metadata {}.{key}", self.path.display(), self.arch))
    }
    pub fn has_meta(&self, key: &str) -> bool {
        self.header.get(&format!("{}.{key}", self.arch)).is_some()
    }
    pub fn usize(&self, key: &str) -> Result<usize> {
        self.value(key)?
            .as_u64()
            .map(|v| v as usize)
            .with_context(|| format!("{}.{key}: not an integer", self.arch))
    }
    pub fn f64(&self, key: &str) -> Result<f64> {
        self.value(key)?
            .as_f64()
            .with_context(|| format!("{}.{key}: not a number", self.arch))
    }
    pub fn str(&self, key: &str) -> Result<&str> {
        self.value(key)?
            .as_str()
            .with_context(|| format!("{}.{key}: not a string", self.arch))
    }
    pub fn usizes(&self, key: &str) -> Result<Vec<usize>> {
        let a = self
            .value(key)?
            .as_array()
            .with_context(|| format!("{}.{key}: not an array", self.arch))?;
        a.iter()
            .map(|v| {
                v.as_u64()
                    .map(|v| v as usize)
                    .with_context(|| format!("{}.{key}: not integers", self.arch))
            })
            .collect()
    }
    pub fn f64s(&self, key: &str) -> Result<Vec<f64>> {
        let a = self
            .value(key)?
            .as_array()
            .with_context(|| format!("{}.{key}: not an array", self.arch))?;
        a.iter()
            .map(|v| v.as_f64().with_context(|| format!("{}.{key}: not numbers", self.arch)))
            .collect()
    }
    pub fn strs(&self, key: &str) -> Result<Vec<String>> {
        let a = self
            .value(key)?
            .as_array()
            .with_context(|| format!("{}.{key}: not an array", self.arch))?;
        a.iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .with_context(|| format!("{}.{key}: not strings", self.arch))
            })
            .collect()
    }

    pub fn has(&self, name: &str) -> bool {
        self.header.tensors.iter().any(|t| t.name == name)
    }

    /// The shape of tensor `name`, outermost dimension first.
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        let t = self.header.tensors.iter().find(|t| t.name == name);
        let t = t.with_context(|| format!("{}: no tensor {name}", self.path.display()))?;
        Ok(t.ne.iter().rev().copied().collect())
    }

    /// The tensor `name`, whose shape (outermost dimension first, as PyTorch prints it) must be
    /// `shape`.
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<(&Tensor, GgmlType)> {
        let t = self
            .header
            .tensors
            .iter()
            .find(|t| t.name == name)
            .with_context(|| format!("{}: no tensor {name}", self.path.display()))?;
        let got: Vec<usize> = t.ne.iter().rev().copied().collect();
        if got != shape {
            bail!("{}: tensor {name} has shape {got:?}, expected {shape:?}", self.path.display());
        }
        let kind = GgmlType::from_id(t.ggml_type).with_context(|| format!("tensor {name}: ggml type {} is not supported", t.ggml_type))?;
        Ok((t, kind))
    }

    fn read(&self, t: &Tensor) -> Result<Vec<u8>> {
        let mut b = vec![0u8; t.end - t.start];
        let mut f = self.file.lock().unwrap();
        f.seek(SeekFrom::Start(t.start as u64))?;
        f.read_exact(&mut b).with_context(|| format!("read tensor {}", t.name))?;
        Ok(b)
    }

    /// The tensor as f32 values (row-major in the PyTorch shape).
    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (t, kind) = self.tensor(name, shape)?;
        let bytes = self.read(t)?;
        let n: usize = shape.iter().product();
        let row = *shape.last().unwrap_or(&1);
        let mut out = vec![0f32; n];
        let rb = kind.row_bytes(row);
        for (o, b) in out.chunks_mut(row).zip(bytes.chunks(rb)) {
            crate::ggml::dequantize_row(kind, b, o);
        }
        Ok(out)
    }

    /// A `[n_out, n_in]` matrix in the file's own storage.
    pub fn mat(&self, name: &str, n_out: usize, n_in: usize) -> Result<Mat> {
        let (t, kind) = self.tensor(name, &[n_out, n_in])?;
        Mat::new(kind, &self.read(t)?, n_out, n_in)
    }

    /// The `n` matrices of a `[n, n_out, n_in]` tensor.
    pub fn mats(&self, name: &str, n: usize, n_out: usize, n_in: usize) -> Result<Vec<Mat>> {
        let (t, kind) = self.tensor(name, &[n, n_out, n_in])?;
        let bytes = self.read(t)?;
        bytes
            .chunks(n_out * kind.row_bytes(n_in))
            .map(|b| Mat::new(kind, b, n_out, n_in))
            .collect()
    }
}

/// A weight matrix `[n_out, n_in]` applied on the CPU in f32: float storage (F32 / F16) as it is,
/// quantized storage (Q8_0 / Q4_0) either dequantized row by row inside the f32 product (`exact`,
/// the arithmetic of the PyTorch reference on the same weights) or through the int8 kernels with
/// the activations rounded to 8 bits per block of 32 (what ggml does with these files).
pub(crate) struct Mat {
    m: GgufMatrix,
    pub n_out: usize,
    pub n_in: usize,
}

impl Mat {
    pub fn new(kind: GgmlType, bytes: &[u8], n_out: usize, n_in: usize) -> Result<Self> {
        Ok(Self {
            m: GgufMatrix::from_bytes(kind, bytes, n_out, n_in)?,
            n_out,
            n_in,
        })
    }

    pub fn from_f32(w: &[f32], n_out: usize, n_in: usize) -> Result<Self> {
        let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::new(GgmlType::F32, &bytes, n_out, n_in)
    }

    pub fn bytes(&self) -> usize {
        self.m.bytes()
    }

    /// `y[t][n_out] = x[t][n_in] W^T`.
    pub fn mul(&self, x: &[f32], t: usize, exact: bool) -> Vec<f32> {
        let mut y = vec![0f32; t * self.n_out];
        mul_multi(&mut [(self, &mut y[..])], x, t, exact);
        y
    }

    pub fn row(&self, r: usize, out: &mut [f32]) {
        self.m.dequantize_rows(r, r + 1, out);
    }
}

/// Several matrices applied to the same input in one parallel region.
pub(crate) fn mul_multi(jobs: &mut [(&Mat, &mut [f32])], x: &[f32], t: usize, exact: bool) {
    let mut js: Vec<(&GgufMatrix, &mut [f32])> = jobs.iter_mut().map(|(m, y)| (&m.m, &mut **y)).collect();
    if exact {
        crate::kernels::matmul_quant_multi_at(&mut js, x, t, 0);
    } else {
        crate::kernels::matmul_quant_multi(&mut js, x, t);
    }
}

/// `x` (`[t, n]`) plus the bias `b` on every row.
pub(crate) fn add_bias(x: &mut [f32], b: &[f32]) {
    for r in x.chunks_mut(b.len()) {
        for (o, v) in r.iter_mut().zip(b) {
            *o += v;
        }
    }
}
