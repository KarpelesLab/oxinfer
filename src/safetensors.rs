//! Minimal zero-copy safetensors reader (single file or sharded directory).

use crate::pool;
use crate::quant::PageVec;
use crate::quant::{Q4Matrix, QBuf, QKind, QMatrix};
use anyhow::{Context, Result, bail};
#[cfg(not(target_arch = "wasm32"))]
use memmap2::Mmap;
use std::collections::{BTreeMap, HashMap};
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute byte range inside the file.
    pub start: usize,
    pub end: usize,
    /// An MLX affine-quantized matrix: `dtype`/`shape` then describe the logical bf16 matrix,
    /// while the packed values sit at `start..end` (scales and biases are sibling tensors).
    pub mlx: Option<MlxQuant>,
    /// A ggml-quantized tensor (GGUF): its ggml type id; `dtype` is then "GGML".
    pub ggml: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct MlxQuant {
    pub bits: usize,
    pub group: usize,
}

/// MLX checkpoints (mlx-lm / mlx-vlm) name multimodal weights `language_model.model.*`,
/// `vision_tower.*`, ...; the loaders use the transformers names `model.language_model.*`, ...
fn mlx_canonical(n: &str) -> String {
    if let Some(r) = n.strip_prefix("language_model.model.") {
        // MoE experts (Gemma 4 26B-A4B): `experts.switch_glu.gate_proj.weight` is the 3D
        // `experts.gate_proj` of transformers' unfused layout.
        if let Some((pre, rest)) = r.split_once(".experts.switch_glu.") {
            let rest = rest.strip_suffix(".weight").unwrap_or(rest);
            return format!("model.language_model.{pre}.experts.{rest}");
        }
        return format!("model.language_model.{r}");
    }
    if let Some(r) = n.strip_prefix("language_model.lm_head.") {
        return format!("lm_head.{r}");
    }
    for p in [
        "vision_tower.",
        "embed_vision.",
        "audio_tower.",
        "embed_audio.",
        "multi_modal_projector.",
    ] {
        if n.starts_with(p) {
            return format!("model.{n}");
        }
    }
    n.to_string()
}

/// Reinterprets little-endian bytes as `T` words, borrowing when aligned.
fn words<T: Copy>(b: &[u8]) -> std::borrow::Cow<'_, [T]>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    let n = b.len() / std::mem::size_of::<T>();
    if (b.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()) {
        std::borrow::Cow::Borrowed(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const T, n) })
    } else {
        let mut v = Vec::<T>::with_capacity(n);
        unsafe {
            std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr() as *mut u8, n * std::mem::size_of::<T>());
            v.set_len(n);
        }
        std::borrow::Cow::Owned(v)
    }
}

enum Storage {
    /// A mapped file, kept open for streaming reads (`crate::stream`).
    #[cfg(not(target_arch = "wasm32"))]
    Mmap(Mmap, File),
    /// Heap copy in a page-aligned, page-multiple allocation (so accelerators can map it in place):
    /// (buffer, file length, padding before the file's first byte). The padding puts the tensor
    /// data on a 16-byte boundary when the header length leaves it unaligned (as MLX writes it).
    Heap(PageVec<u64>, usize, usize),
    /// Kept by the host (browser), read by range; tensors in it are only usable as row tables.
    Host(crate::fsx::HostFile),
}

impl Storage {
    fn bytes(&self) -> &[u8] {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Storage::Mmap(m, _) => &m[..],
            Storage::Heap(v, n, pad) => unsafe { std::slice::from_raw_parts((v.as_ptr() as *const u8).add(*pad), *n) },
            Storage::Host(_) => &[],
        }
    }
}

impl std::ops::Deref for Storage {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.bytes()
    }
}

pub struct SafeTensors {
    mmap: Storage,
    tensors: HashMap<String, TensorInfo>,
    metadata: HashMap<String, String>,
}

/// Reads a whole file into memory using parallel positional reads (sequential where the
/// platform has no positional read).
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) fn read_file_parallel(file: &File, len: usize, pad: usize) -> Result<PageVec<u64>> {
    let mut buf = PageVec::<u64>::zeroed((pad + len).div_ceil(8));
    let bytes = unsafe { std::slice::from_raw_parts_mut((buf.as_mut_ptr() as *mut u8).add(pad), len) };
    #[cfg(unix)]
    const CHUNK: usize = 32 << 20;
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        use std::sync::Mutex;
        let err: Mutex<Option<std::io::Error>> = Mutex::new(None);
        let p = SendBytes(bytes.as_mut_ptr());
        let n = len.div_ceil(CHUNK);
        pool::run(n, &|i| {
            let start = i * CHUNK;
            let end = (start + CHUNK).min(len);
            // SAFETY: chunks are disjoint.
            let chunk = unsafe { std::slice::from_raw_parts_mut(p.get().add(start), end - start) };
            if let Err(e) = file.read_exact_at(chunk, start as u64) {
                *err.lock().unwrap() = Some(e);
            }
        });
        if let Some(e) = err.into_inner().unwrap() {
            return Err(e).context("read weights");
        }
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = file;
        f.seek(SeekFrom::Start(0))?;
        f.read_exact(bytes).context("read weights")?;
    }
    Ok(buf)
}

#[cfg(unix)]
#[derive(Clone, Copy)]
struct SendBytes(*mut u8);
#[cfg(unix)]
unsafe impl Send for SendBytes {}
#[cfg(unix)]
unsafe impl Sync for SendBytes {}
#[cfg(unix)]
impl SendBytes {
    fn get(self) -> *mut u8 {
        self.0
    }
}

impl SafeTensors {
    pub fn open(path: &Path, in_memory: bool) -> Result<Self> {
        if let Some(hf) = crate::fsx::host_file(path) {
            let mut n8 = [0u8; 8];
            crate::fsx::read_host(hf, 0, &mut n8)?;
            let n = u64::from_le_bytes(n8) as usize;
            if 8 + n > hf.len {
                bail!("{}: header length {} exceeds file", path.display(), n);
            }
            let mut head = vec![0u8; n];
            crate::fsx::read_host(hf, 8, &mut head)?;
            return Self::from_header(path, Storage::Host(hf), &head, 8 + n, hf.len);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let mmap = if in_memory {
            use std::io::Read;
            let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
            let len = file.metadata()?.len() as usize;
            let mut n8 = [0u8; 8];
            file.read_exact(&mut n8)
                .with_context(|| format!("{}: file too small", path.display()))?;
            let data_start = 8usize.saturating_add(u64::from_le_bytes(n8) as usize);
            // GGUF data is aligned within the file already (and its offsets are absolute).
            let pad = if &n8[..4] == b"GGUF" { 0 } else { (16 - data_start % 16) % 16 };
            Storage::Heap(read_file_parallel(&file, len, pad)?, len, pad)
        } else {
            let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
            let map = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
            Storage::Mmap(map, file)
        };
        #[cfg(target_arch = "wasm32")]
        let mmap = {
            let _ = in_memory;
            let (v, len) = crate::fsx::read_pages(path)?;
            Storage::Heap(v, len, 0)
        };
        if mmap.len() < 8 {
            bail!("{}: file too small", path.display());
        }
        if mmap.len() >= 4 && &mmap[..4] == b"GGUF" {
            return Self::from_gguf(path, mmap);
        }
        if crate::torch::is_torch_zip(&mmap) {
            let tensors = crate::torch::read_tensors(&mmap)
                .with_context(|| format!("{}: read PyTorch checkpoint", path.display()))?
                .into_iter()
                .map(|(name, t)| {
                    let info = TensorInfo {
                        dtype: t.dtype.to_string(),
                        shape: t.shape,
                        start: t.start,
                        end: t.end,
                        mlx: None,
                        ggml: None,
                    };
                    (name, info)
                })
                .collect();
            return Ok(Self {
                mmap,
                tensors,
                metadata: HashMap::new(),
            });
        }
        let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        if 8 + n > mmap.len() {
            bail!("{}: header length {} exceeds file", path.display(), n);
        }
        let len = mmap.len();
        let head = mmap[8..8 + n].to_vec();
        Self::from_header(path, mmap, &head, 8 + n, len)
    }

    fn from_gguf(path: &Path, mmap: Storage) -> Result<Self> {
        let h = crate::gguf::parse_header(&mmap[..], mmap.len()).with_context(|| format!("{}", path.display()))?;
        let mut tensors = HashMap::new();
        for crate::gguf::HfTensor {
            name,
            shape,
            ggml_type: t,
            start,
            end,
        } in crate::gguf::hf_tensors(&h)?
        {
            let dtype = match t {
                0 => "F32",
                1 => "F16",
                30 => "BF16",
                26 => "I32",
                _ => "GGML",
            };
            let info = TensorInfo {
                dtype: dtype.to_string(),
                shape,
                start,
                end,
                mlx: None,
                ggml: (dtype == "GGML").then_some(t),
            };
            tensors.insert(name, info);
        }
        Ok(Self {
            mmap,
            tensors,
            metadata: crate::gguf::scalar_metadata(&h).into_iter().collect(),
        })
    }

    fn from_header(path: &Path, mmap: Storage, head: &[u8], data_start: usize, file_len: usize) -> Result<Self> {
        let header: serde_json::Value = serde_json::from_slice(head).with_context(|| format!("{}: parse header", path.display()))?;
        let obj = header.as_object().context("header is not an object")?;
        let mut tensors = HashMap::with_capacity(obj.len());
        let mut metadata = HashMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                if let Some(m) = v.as_object() {
                    for (k, v) in m {
                        if let Some(v) = v.as_str() {
                            metadata.insert(k.clone(), v.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = v["dtype"].as_str().context("dtype")?.to_string();
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let offs = v["data_offsets"].as_array().context("data_offsets")?;
            let start = data_start + offs[0].as_u64().unwrap() as usize;
            let end = data_start + offs[1].as_u64().unwrap() as usize;
            if end > file_len || start > end {
                bail!("tensor {name} out of bounds");
            }
            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    start,
                    end,
                    mlx: None,
                    ggml: None,
                },
            );
        }
        if metadata.get("format").map(String::as_str) == Some("mlx") {
            tensors = tensors.into_iter().map(|(k, v)| (mlx_canonical(&k), v)).collect();
        }
        Ok(Self { mmap, tensors, metadata })
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// A `__metadata__` value of the header.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Raw bytes of a tensor.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
        if let Storage::Host(_) = self.mmap {
            bail!("tensor {name} is kept by the host and can only be read by rows");
        }
        Ok(&self.mmap[t.start..t.end])
    }

    /// Raw bytes of a tensor; from a host-kept file they are copied out (loaders that convert every
    /// weight anyway, like the diffusion models, can then leave whole files outside module memory).
    pub fn bytes_cow(&self, name: &str) -> Result<std::borrow::Cow<'_, [u8]>> {
        let t = self.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
        match &self.mmap {
            Storage::Host(h) => {
                let mut buf = vec![0u8; t.end - t.start];
                crate::fsx::read_host(*h, t.start as u64, &mut buf)?;
                Ok(std::borrow::Cow::Owned(buf))
            }
            m => Ok(std::borrow::Cow::Borrowed(&m[t.start..t.end])),
        }
    }

    /// Returns a bf16 tensor as a slice of raw u16 bit patterns.
    pub fn bf16(&self, name: &str) -> Result<&[u16]> {
        let t = self.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
        if let Storage::Host(_) = self.mmap {
            bail!("tensor {name} is kept by the host and can only be read by rows");
        }
        if t.dtype != "BF16" {
            bail!("tensor {name} has dtype {}, expected BF16", t.dtype);
        }
        let b = &self.mmap[t.start..t.end];
        if !(b.as_ptr() as usize).is_multiple_of(2) {
            bail!("tensor {name} is not 2-byte aligned");
        }
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) })
    }

    pub fn len(&self) -> usize {
        match &self.mmap {
            Storage::Host(h) => h.len,
            m => m.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The mapped file and its bytes, when this file is memory-mapped (`--mmap`).
    pub fn mapping(&self) -> Option<MappedFile<'_>> {
        match &self.mmap {
            #[cfg(not(target_arch = "wasm32"))]
            Storage::Mmap(m, f) => Some(MappedFile { bytes: &m[..], file: f }),
            _ => None,
        }
    }

    /// Touch every page so the weights are resident before the first forward pass.
    pub fn preload(&self) {
        #[cfg(unix)]
        if let Storage::Mmap(m, _) = &self.mmap {
            let _ = m.advise(memmap2::Advice::WillNeed);
        }
        const CHUNK: usize = 64 << 20;
        let bytes: &[u8] = &self.mmap;
        let n = bytes.len().div_ceil(CHUNK);
        let total = std::sync::atomic::AtomicU64::new(0);
        pool::run(n, &|i| {
            let c = &bytes[i * CHUNK..((i + 1) * CHUNK).min(bytes.len())];
            let s: u64 = c.iter().step_by(4096).map(|&b| b as u64).sum();
            total.fetch_add(s, std::sync::atomic::Ordering::Relaxed);
        });
        std::hint::black_box(total);
    }
}

/// A memory-mapped checkpoint file: its bytes as mapped, and the open file (for positional
/// reads and read-ahead hints on the same pages; see `crate::stream`).
#[derive(Clone, Copy)]
pub struct MappedFile<'a> {
    pub bytes: &'a [u8],
    pub file: &'a std::fs::File,
}

/// A model directory: either `model.safetensors` or an index plus shards.
pub struct ModelFiles {
    shards: Vec<SafeTensors>,
    index: HashMap<String, usize>,
    /// Tensors the model loader read, and the format it quantized each to (None: used as stored).
    /// `write_quantized` replays this to produce a pre-quantized checkpoint.
    touched: Mutex<BTreeMap<String, Option<QKind>>>,
    /// How each tensor was used: (as a matmul weight, as a lookup table).
    uses: Mutex<HashMap<String, (bool, bool)>>,
    /// bf16 copies of MLX-quantized tensors that were read as plain bf16 (never freed).
    dequantized: Mutex<HashMap<String, PageVec<u16>>>,
    /// Aligned copies of tensors from host-kept shards or unaligned mapped ones (never freed).
    copies: Mutex<HashMap<String, (PageVec<u64>, usize)>>,
}

/// Maximum shard size written by `write_quantized`.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
const SHARD: usize = 1 << 30;

/// One tensor of a checkpoint being written.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
struct Planned {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: usize,
    /// (source tensor, quantization, scales part); None for padding.
    src: Option<(String, Option<QKind>, bool)>,
}

/// Where a row table's bytes live: in module memory, or in a host-kept file at an offset.
#[derive(Clone, Copy)]
pub enum TablePart<'a> {
    Mem(&'a [u8]),
    Host(crate::fsx::HostFile, u64),
}

impl TablePart<'_> {
    /// Copies `out.len()` bytes starting at byte `off` of the tensor.
    pub fn read(&self, off: usize, out: &mut [u8]) -> Result<()> {
        match *self {
            TablePart::Mem(b) => {
                out.copy_from_slice(&b[off..off + out.len()]);
                Ok(())
            }
            TablePart::Host(h, start) => crate::fsx::read_host(h, start + off as u64, out),
        }
    }
}

/// A weight matrix stored pre-quantized in the checkpoint, borrowed in place.
pub enum Stored {
    Q8(QMatrix),
    Q4(Q4Matrix),
}

impl ModelFiles {
    /// No tensors (models whose weights are not safetensors, e.g. Core ML packages).
    pub fn empty() -> Self {
        Self {
            shards: Vec::new(),
            index: HashMap::new(),
            touched: Mutex::new(BTreeMap::new()),
            uses: Mutex::new(HashMap::new()),
            dequantized: Mutex::new(HashMap::new()),
            copies: Mutex::new(HashMap::new()),
        }
    }

    /// Opens the checkpoint in directory `dir` (safetensors, sharded or not, or
    /// `pytorch_model.bin`), or a single GGUF file.
    pub fn open(dir: &Path, in_memory: bool) -> Result<Self> {
        if crate::gguf::is_gguf(dir) {
            let shard = SafeTensors::open(dir, in_memory)?;
            let index = shard.names().map(|n| (n.clone(), 0)).collect();
            return Ok(Self {
                shards: vec![shard],
                index,
                touched: Mutex::new(BTreeMap::new()),
                uses: Mutex::new(HashMap::new()),
                dequantized: Mutex::new(HashMap::new()),
                copies: Mutex::new(HashMap::new()),
            });
        }
        let single = dir.join("model.safetensors");
        let idx = dir.join("model.safetensors.index.json");
        let mut files: Vec<PathBuf> = Vec::new();
        let torch = dir.join("pytorch_model.bin");
        let torch_idx = dir.join("pytorch_model.bin.index.json");
        // Sharded PyTorch checkpoints (`pytorch_model-0000N-of-0000M.bin`) share the index format.
        let idx = if !crate::fsx::exists(&idx) && !crate::fsx::exists(&single) && crate::fsx::exists(&torch_idx) {
            torch_idx
        } else {
            idx
        };
        if crate::fsx::exists(&single) {
            files.push(single);
        } else if !crate::fsx::exists(&idx) && crate::fsx::exists(&torch) {
            files.push(torch);
        } else if crate::fsx::exists(&idx) {
            let v: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&idx)?)?;
            let map = v["weight_map"].as_object().context("weight_map")?;
            let mut names: Vec<String> = map.values().map(|x| x.as_str().unwrap().to_string()).collect();
            names.sort();
            names.dedup();
            files.extend(names.into_iter().map(|n| dir.join(n)));
        } else {
            bail!("no model.safetensors, index or pytorch_model.bin found in {}", dir.display());
        }
        let mut shards = Vec::with_capacity(files.len());
        for p in &files {
            shards.push(SafeTensors::open(p, in_memory)?);
        }
        let mut index = HashMap::new();
        for (i, s) in shards.iter().enumerate() {
            for n in s.names() {
                index.insert(n.clone(), i);
            }
        }
        let mut files = Self {
            shards,
            index,
            touched: Mutex::new(BTreeMap::new()),
            uses: Mutex::new(HashMap::new()),
            dequantized: Mutex::new(HashMap::new()),
            copies: Mutex::new(HashMap::new()),
        };
        if files.is_mlx() {
            let cfg = crate::fsx::read(&dir.join("config.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .unwrap_or_default();
            files.annotate_mlx(&cfg)?;
        }
        Ok(files)
    }

    /// Whether this is an MLX checkpoint (affine-quantized `{name}.weight` + `.scales` + `.biases`).
    pub fn is_mlx(&self) -> bool {
        self.metadata("format") == Some("mlx")
    }

    /// Marks every quantized matrix with its bit width and group size (from config.json's
    /// `quantization`, which may override them per module) and its logical bf16 shape.
    fn annotate_mlx(&mut self, cfg: &serde_json::Value) -> Result<()> {
        let q = cfg.get("quantization").or_else(|| cfg.get("quantization_config"));
        let get = |v: Option<&serde_json::Value>, k: &str, d: usize| v.and_then(|v| v[k].as_u64()).map_or(d, |x| x as usize);
        let (bits, group) = (get(q, "bits", 4), get(q, "group_size", 64));
        // Per-module overrides are keyed by MLX module path, so compare canonical forms.
        let overrides: HashMap<String, (usize, usize)> = q
            .and_then(|q| q.as_object())
            .map(|o| {
                o.iter()
                    .filter(|(_, v)| v.is_object())
                    .map(|(k, v)| (mlx_canonical(k), (get(Some(v), "bits", bits), get(Some(v), "group_size", group))))
                    .collect()
            })
            .unwrap_or_default();
        // Scales and their matrix may sit in different shards.
        let scales: Vec<(String, usize)> = self
            .shards
            .iter()
            .flat_map(|sh| sh.tensors.iter())
            .filter(|(n, _)| n.ends_with(".scales"))
            .map(|(n, t)| (n.clone(), *t.shape.last().unwrap_or(&0)))
            .collect();
        for (sn, s_cols) in scales {
            let base = &sn[..sn.len() - ".scales".len()];
            // 3D expert tensors have no `.weight` suffix (see `mlx_canonical`).
            let wn = match self.index.contains_key(&format!("{base}.weight")) {
                true => format!("{base}.weight"),
                false => base.to_string(),
            };
            let (bits, group) = overrides.get(base).copied().unwrap_or((bits, group));
            let Some(&i) = self.index.get(&wn) else { continue };
            let Some(w) = self.shards[i].tensors.get_mut(&wn) else { continue };
            if w.dtype != "U32" || !(2..=3).contains(&w.shape.len()) {
                continue;
            }
            let last = w.shape.len() - 1;
            let n_in = w.shape[last] * 32 / bits;
            if s_cols * group != n_in {
                bail!(
                    "{wn}: {} packed columns at {bits} bits do not match {s_cols} groups of {group}",
                    w.shape[last]
                );
            }
            w.shape[last] = n_in;
            w.dtype = "BF16".into();
            w.mlx = Some(MlxQuant { bits, group });
        }
        Ok(())
    }

    /// Raw bytes of tensor `name`, copied out of a host-kept shard if need be.
    fn bytes_cow(&self, name: &str) -> Result<std::borrow::Cow<'_, [u8]>> {
        let i = *self.index.get(name).with_context(|| format!("missing tensor {name}"))?;
        self.shards[i].bytes_cow(name)
    }

    /// The MLX quantization of tensor `name`, if it is an MLX-quantized matrix.
    pub fn mlx(&self, name: &str) -> Option<MlxQuant> {
        self.info(name).and_then(|i| i.mlx)
    }

    /// An MLX 4-bit matrix `name` (`[n_out, n_in]`) repacked as Q4 with offsets; None when `name`
    /// is not stored that way.
    pub fn mlx_q4(&self, name: &str, n_out: usize, n_in: usize) -> Result<Option<Q4Matrix>> {
        let Some(info) = self.info(name) else { return Ok(None) };
        let Some(m) = info.mlx else { return Ok(None) };
        if m.bits != 4 || !m.group.is_multiple_of(crate::quant::QBLOCK) {
            return Ok(None);
        }
        if info.shape != [n_out, n_in] {
            bail!("tensor {name} has shape {:?}, expected [{n_out}, {n_in}]", info.shape);
        }
        self.touch(name);
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let (w, s, b) = (
            self.bytes_cow(name)?,
            self.bytes_cow(&format!("{base}.scales"))?,
            self.bytes_cow(&format!("{base}.biases"))?,
        );
        Ok(Some(crate::quant::from_mlx_q4(
            &words::<u32>(&w),
            &words::<u16>(&s),
            &words::<u16>(&b),
            n_out,
            n_in,
            m.group,
        )))
    }

    /// The experts of an MLX 4-bit `[n_exp, n_out, n_in]` tensor (MoE `switch_glu` projections),
    /// each repacked as Q4 with offsets; None when `name` is not stored that way.
    pub fn mlx_q4_experts(&self, name: &str, n_exp: usize, n_out: usize, n_in: usize) -> Result<Option<Vec<Q4Matrix>>> {
        let Some(info) = self.info(name) else { return Ok(None) };
        let Some(m) = info.mlx else { return Ok(None) };
        if m.bits != 4 || !m.group.is_multiple_of(crate::quant::QBLOCK) {
            return Ok(None);
        }
        if info.shape != [n_exp, n_out, n_in] {
            bail!("tensor {name} has shape {:?}, expected [{n_exp}, {n_out}, {n_in}]", info.shape);
        }
        self.touch(name);
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let (w, s, b) = (
            self.bytes_cow(name)?,
            self.bytes_cow(&format!("{base}.scales"))?,
            self.bytes_cow(&format!("{base}.biases"))?,
        );
        let (w, s, b) = (words::<u32>(&w), words::<u16>(&s), words::<u16>(&b));
        let (wn, gn) = (n_out * n_in / 8, n_out * n_in / m.group);
        if w.len() != n_exp * wn || s.len() != n_exp * gn || b.len() != n_exp * gn {
            bail!("tensor {name}: MLX parts do not match {n_exp} experts of [{n_out}, {n_in}]");
        }
        Ok(Some(
            (0..n_exp)
                .map(|e| {
                    crate::quant::from_mlx_q4(
                        &w[e * wn..(e + 1) * wn],
                        &s[e * gn..(e + 1) * gn],
                        &b[e * gn..(e + 1) * gn],
                        n_out,
                        n_in,
                        m.group,
                    )
                })
                .collect(),
        ))
    }

    /// MLX 4-bit gate and up expert tensors (`[n_exp, n_out, n_in]` each) fused per expert into
    /// one `[2 * n_out, n_in]` Q4 matrix, gate rows then up rows (the layout of transformers'
    /// `gate_up_proj`, which the accelerator's expert path takes); None unless both are stored so.
    pub fn mlx_q4_experts_fused(&self, gate: &str, up: &str, n_exp: usize, n_out: usize, n_in: usize) -> Result<Option<Vec<Q4Matrix>>> {
        let (Some(mg), Some(mu)) = (self.mlx(gate), self.mlx(up)) else {
            return Ok(None);
        };
        if mg.bits != 4 || mu.bits != 4 || mg.group != mu.group || !mg.group.is_multiple_of(crate::quant::QBLOCK) {
            return Ok(None);
        }
        for n in [gate, up] {
            let shape = &self.info(n).unwrap().shape;
            if shape[..] != [n_exp, n_out, n_in] {
                bail!("tensor {n} has shape {shape:?}, expected [{n_exp}, {n_out}, {n_in}]");
            }
            self.touch(n);
        }
        let parts = |n: &str| -> Result<_> {
            let base = n.strip_suffix(".weight").unwrap_or(n);
            Ok((
                self.bytes_cow(n)?,
                self.bytes_cow(&format!("{base}.scales"))?,
                self.bytes_cow(&format!("{base}.biases"))?,
            ))
        };
        let ((gw, gs, gb), (uw, us, ub)) = (parts(gate)?, parts(up)?);
        let (gw, gs, gb) = (words::<u32>(&gw), words::<u16>(&gs), words::<u16>(&gb));
        let (uw, us, ub) = (words::<u32>(&uw), words::<u16>(&us), words::<u16>(&ub));
        let (wn, gn) = (n_out * n_in / 8, n_out * n_in / mg.group);
        if gw.len() != n_exp * wn || uw.len() != n_exp * wn || gs.len() != n_exp * gn || us.len() != n_exp * gn {
            bail!("tensors {gate} / {up}: MLX parts do not match {n_exp} experts of [{n_out}, {n_in}]");
        }
        let cat = |a: &[u32], b: &[u32]| [a, b].concat();
        let cat16 = |a: &[u16], b: &[u16]| [a, b].concat();
        Ok(Some(
            (0..n_exp)
                .map(|e| {
                    let (w, s) = (e * wn..(e + 1) * wn, e * gn..(e + 1) * gn);
                    crate::quant::from_mlx_q4(
                        &cat(&gw[w.clone()], &uw[w]),
                        &cat16(&gs[s.clone()], &us[s.clone()]),
                        &cat16(&gb[s.clone()], &ub[s]),
                        2 * n_out,
                        n_in,
                        mg.group,
                    )
                })
                .collect(),
        ))
    }

    /// Reads row-table parts of MLX matrix `name`: (packed values, scales, biases) as byte sources.
    pub fn mlx_parts(&self, name: &str) -> Option<[TablePart<'_>; 3]> {
        self.mlx(name)?;
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let part = |n: &str| -> Option<TablePart<'_>> {
            let i = *self.index.get(n)?;
            let sh = &self.shards[i];
            let t = sh.tensors.get(n)?;
            Some(match &sh.mmap {
                Storage::Host(h) => TablePart::Host(*h, t.start as u64),
                m => TablePart::Mem(&m[t.start..t.end]),
            })
        };
        Some([part(name)?, part(&format!("{base}.scales"))?, part(&format!("{base}.biases"))?])
    }

    /// An MLX-quantized matrix dequantized to bf16 (kept for the lifetime of `self`).
    fn mlx_bf16(&self, name: &str, m: MlxQuant, rows: usize, cols: usize) -> Result<&[u16]> {
        let mut cache = self.dequantized.lock().unwrap();
        if !cache.contains_key(name) {
            let base = name.strip_suffix(".weight").unwrap_or(name);
            let (w, s, b) = (
                self.bytes_cow(name)?,
                self.bytes_cow(&format!("{base}.scales"))?,
                self.bytes_cow(&format!("{base}.biases"))?,
            );
            let (w, s, b) = (words::<u32>(&w), words::<u16>(&s), words::<u16>(&b));
            let (wpr, gpr) = (cols * m.bits / 32, cols / m.group);
            let mut out = PageVec::<u16>::zeroed(rows * cols);
            let mut row = vec![0f32; cols];
            for r in 0..rows {
                crate::quant::mlx_dequant_row(
                    &w[r * wpr..(r + 1) * wpr],
                    &s[r * gpr..(r + 1) * gpr],
                    &b[r * gpr..(r + 1) * gpr],
                    m.bits,
                    m.group,
                    &mut row,
                );
                for (o, &v) in out[r * cols..(r + 1) * cols].iter_mut().zip(&row) {
                    *o = half::bf16::from_f32(v).to_bits();
                }
            }
            cache.insert(name.to_string(), out);
        }
        let v = &cache[name];
        // SAFETY: entries are never removed or replaced, and the PageVec's heap block does not
        // move when the map rehashes, so the slice lives as long as `self`.
        Ok(unsafe { std::slice::from_raw_parts(v.as_ptr(), v.len()) })
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).and_then(|&i| self.shards[i].info(name))
    }

    pub fn bf16(&self, name: &str) -> Result<&[u16]> {
        let i = *self.index.get(name).with_context(|| format!("missing tensor {name}"))?;
        self.touch(name);
        let info = self.shards[i].info(name).unwrap();
        if let Some(m) = info.mlx {
            let cols = *info.shape.last().unwrap_or(&0);
            let rows = info.shape[..info.shape.len().saturating_sub(1)].iter().product();
            return self.mlx_bf16(name, m, rows, cols);
        }
        let sh = &self.shards[i];
        let host = matches!(sh.mmap, Storage::Host(_));
        if info.dtype == "F32" || info.dtype == "F16" {
            // fp32 / fp16 checkpoints (PyTorch files, sentence-transformers heads): a bf16 copy.
            let mut cache = self.dequantized.lock().unwrap();
            if !cache.contains_key(name) {
                let b = self.copy(name)?;
                let v: PageVec<u16> = match info.dtype.as_str() {
                    "F32" => {
                        let mut v = PageVec::<u16>::zeroed(b.len() / 4);
                        for (o, c) in v.iter_mut().zip(b.as_chunks::<4>().0) {
                            *o = half::bf16::from_f32(f32::from_le_bytes(*c)).to_bits();
                        }
                        v
                    }
                    _ => {
                        let mut v = PageVec::<u16>::zeroed(b.len() / 2);
                        for (o, c) in v.iter_mut().zip(b.as_chunks::<2>().0) {
                            *o = half::bf16::from_f32(half::f16::from_le_bytes(*c).to_f32()).to_bits();
                        }
                        v
                    }
                };
                cache.insert(name.to_string(), v);
            }
            let v = &cache[name];
            // SAFETY: as in `mlx_bf16`.
            return Ok(unsafe { std::slice::from_raw_parts(v.as_ptr(), v.len()) });
        }
        if info.dtype == "BF16" && (host || !(sh.mmap.as_ptr() as usize + info.start).is_multiple_of(2)) {
            // In a host-kept shard, or unaligned in a mapped file (MLX headers have any length):
            // read an aligned copy.
            let b = self.copy(name)?;
            return Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) });
        }
        sh.bf16(name)
    }

    /// A bf16 convolution weight in torch layout `shape` (`[out, in, spatial...]`). MLX stores
    /// convolutions channels-last (`[out, spatial..., in]`); those are permuted into a copy.
    pub fn bf16_conv(&self, name: &str, shape: &[usize]) -> Result<&[u16]> {
        let info = self.info(name).with_context(|| format!("missing tensor {name}"))?;
        if info.shape == shape {
            return self.bf16(name);
        }
        let mut cl = vec![shape[0]];
        cl.extend_from_slice(&shape[2..]);
        cl.push(shape[1]);
        if !self.is_mlx() || info.shape != cl {
            bail!("tensor {name} has shape {:?}, expected {shape:?}", info.shape);
        }
        let src = self.bf16(name)?;
        let key = format!("{name} (torch layout)");
        let mut cache = self.dequantized.lock().unwrap();
        if !cache.contains_key(&key) {
            let (o, i, sp) = (shape[0], shape[1], shape[2..].iter().product::<usize>());
            let mut v = PageVec::<u16>::zeroed(src.len());
            for a in 0..o {
                for b in 0..i {
                    for c in 0..sp {
                        v[(a * i + b) * sp + c] = src[(a * sp + c) * i + b];
                    }
                }
            }
            cache.insert(key.clone(), v);
        }
        let v = &cache[&key];
        // SAFETY: as in `mlx_bf16`.
        Ok(unsafe { std::slice::from_raw_parts(v.as_ptr(), v.len()) })
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.touch(name);
        let i = *self.index.get(name).with_context(|| format!("missing tensor {name}"))?;
        if matches!(self.shards[i].mmap, Storage::Host(_)) {
            return self.copy(name);
        }
        self.raw(name)
    }

    /// Tensor `name` as f32, whatever its float dtype (small vectors: norms, biases).
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.info(name).with_context(|| format!("missing tensor {name}"))?;
        if info.mlx.is_some() {
            return Ok(self.bf16(name)?.iter().map(|&v| crate::kernels::bf16_to_f32(v)).collect());
        }
        let b = self.bytes(name)?;
        Ok(match info.dtype.as_str() {
            "F32" => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
            "BF16" => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| crate::kernels::bf16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
            "F16" => b.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect(),
            d => bail!("tensor {name} has dtype {d}, expected a float type"),
        })
    }

    /// An 8-byte-aligned copy of tensor `name`'s bytes, kept for the lifetime of `self`.
    fn copy(&self, name: &str) -> Result<&[u8]> {
        let mut cache = self.copies.lock().unwrap();
        if !cache.contains_key(name) {
            let b = self.bytes_cow(name)?;
            let mut v = PageVec::<u64>::zeroed(b.len().div_ceil(8));
            unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr() as *mut u8, b.len()) };
            cache.insert(name.to_string(), (v, b.len()));
        }
        let (v, n) = &cache[name];
        // SAFETY: entries are never removed or replaced, and the PageVec's heap block does not
        // move when the map rehashes, so the slice lives as long as `self`.
        Ok(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, *n) })
    }

    fn raw(&self, name: &str) -> Result<&[u8]> {
        let i = *self.index.get(name).with_context(|| format!("missing tensor {name}"))?;
        self.shards[i].bytes(name)
    }

    fn touch(&self, name: &str) {
        self.touched.lock().unwrap().entry(name.to_string()).or_insert(None);
    }

    /// Records whether `name` was used as a matmul weight or as a lookup table (`write_quantized`
    /// puts lookup-only tensors in their own shard).
    pub fn note_use(&self, name: &str, lin: bool, table: bool) {
        let mut u = self.uses.lock().unwrap();
        let e = u.entry(name.to_string()).or_insert((false, false));
        e.0 |= lin;
        e.1 |= table;
    }

    /// Records that the loader quantized tensor `name` to `kind` (see `write_quantized`).
    pub fn mark_quant(&self, name: &str, kind: QKind) {
        self.touched.lock().unwrap().insert(name.to_string(), Some(kind));
    }

    /// For a tensor in a host-kept shard: the file and the tensor's absolute byte offset.
    pub fn host_tensor(&self, name: &str) -> Option<(crate::fsx::HostFile, u64)> {
        let i = *self.index.get(name)?;
        let sh = &self.shards[i];
        match &sh.mmap {
            Storage::Host(h) => sh.tensors.get(name).map(|t| (*h, t.start as u64)),
            _ => None,
        }
    }

    /// Lets the host drop every host-kept shard the model no longer reads: all but those holding
    /// lookup tables (with their MLX scales and biases), which are read row by row while running.
    /// Call once the model is built.
    pub fn release_unused_host_files(&self) {
        let uses = self.uses.lock().unwrap();
        let mut keep = std::collections::HashSet::new();
        for (name, &(_, table)) in uses.iter() {
            if !table {
                continue;
            }
            let base = name.strip_suffix(".weight").unwrap_or(name);
            for n in [name.clone(), format!("{base}.scales"), format!("{base}.biases")]
                .into_iter()
                .chain([".q8", ".q8s", ".q4", ".q4s"].iter().map(|s| format!("{name}{s}")))
            {
                if let Some(&i) = self.index.get(&n) {
                    keep.insert(i);
                }
            }
        }
        for (i, sh) in self.shards.iter().enumerate() {
            if let Storage::Host(h) = &sh.mmap
                && !keep.contains(&i)
            {
                crate::fsx::release_host(*h);
            }
        }
    }

    /// A header metadata value (from the first shard that has it).
    /// A ggml-quantized tensor (GGUF): its type and raw blocks.
    pub fn ggml(&self, name: &str) -> Option<Result<(crate::ggml::GgmlType, &[u8])>> {
        let t = self.info(name)?.ggml?;
        Some((|| {
            let kind = crate::ggml::GgmlType::from_id(t).with_context(|| format!("tensor {name}: ggml type {t} is not supported"))?;
            Ok((kind, self.bytes(name)?))
        })())
    }

    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.shards.iter().find_map(|s| s.metadata.get(key).map(|v| v.as_str()))
    }

    /// The pre-quantized form of matrix `name` (`[n_out, n_in]`, or a table of `n_out` rows), if
    /// the checkpoint has one: tensors `{name}.q8` + `{name}.q8s` or `{name}.q4` + `{name}.q4s`.
    ///
    /// # Safety
    /// The returned matrix borrows this checkpoint's memory without a lifetime; the caller must
    /// keep `self` alive (and not move its storage) for as long as the matrix is used. The model
    /// loaders guarantee this by holding a `&'a ModelFiles` for the model's lifetime.
    pub unsafe fn quantized(&self, name: &str, n_out: usize, n_in: usize) -> Result<Option<Stored>> {
        let check = |t: &str, dtype: &str, shape: &[usize]| -> Result<&[u8]> {
            let info = self.info(t).with_context(|| format!("missing tensor {t}"))?;
            if info.dtype != dtype || info.shape != shape {
                bail!("tensor {t} is {} {:?}, expected {dtype} {shape:?}", info.dtype, info.shape);
            }
            let b = self.raw(t)?;
            let align = match dtype {
                "F32" => 4,
                "F16" => 2,
                _ => 1,
            };
            if !(b.as_ptr() as usize).is_multiple_of(align) {
                bail!("tensor {t} is not {align}-byte aligned");
            }
            Ok(b)
        };
        if !n_in.is_multiple_of(crate::quant::QBLOCK) {
            return Ok(None);
        }
        let bpr = n_in / crate::quant::QBLOCK;
        let q8 = format!("{name}.q8");
        let q4 = format!("{name}.q4");
        if self.has(&q8) {
            let q = check(&q8, "I8", &[n_out, n_in])?;
            let sc = check(&format!("{name}.q8s"), "F32", &[n_out, bpr])?;
            self.touched.lock().unwrap().insert(name.to_string(), Some(QKind::Q8));
            let q = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const i8, q.len()) };
            let sc = unsafe { std::slice::from_raw_parts(sc.as_ptr() as *const f32, sc.len() / 4) };
            return Ok(Some(Stored::Q8(QMatrix {
                q: unsafe { QBuf::borrowed(q) },
                scales: unsafe { QBuf::borrowed(sc) },
                n_out,
                n_in,
            })));
        }
        if self.has(&q4) {
            let q = check(&q4, "U8", &[n_out, n_in / 2])?;
            let sc = check(&format!("{name}.q4s"), "F16", &[n_out, bpr])?;
            self.touched.lock().unwrap().insert(name.to_string(), Some(QKind::Q4));
            let sc = unsafe { std::slice::from_raw_parts(sc.as_ptr() as *const u16, sc.len() / 2) };
            return Ok(Some(Stored::Q4(Q4Matrix {
                q: unsafe { QBuf::borrowed(q) },
                scales: unsafe { QBuf::borrowed(sc) },
                n_out,
                n_in,
                offsets: None,
            })));
        }
        Ok(None)
    }

    /// Writes a checkpoint holding exactly the tensors the loader read: quantized ones as
    /// `{name}.q8|q4` + scales, the rest as stored. Tensors only ever read as lookup tables (never
    /// as a matmul weight) go to a second shard, `model-tables.safetensors`, which a browser host
    /// can keep outside wasm memory and read row by row. Large tensors start on 16 KiB boundaries
    /// so accelerators can map them in place. Returns the files written with their sizes.
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[allow(clippy::type_complexity)]
    pub fn write_quantized(&self, dir: &Path, metadata: &[(&str, &str)]) -> Result<Vec<(PathBuf, u64)>> {
        use crate::quant::QBLOCK;
        if self.metadata("format") == Some("gguf") {
            bail!("--save-quantized does not convert GGUF files; load them directly instead");
        }
        if self.is_mlx() {
            bail!("--save-quantized does not convert MLX checkpoints; load them directly instead");
        }
        let touched = self.touched.lock().unwrap().clone();
        let uses = self.uses.lock().unwrap().clone();
        let mut weights: Vec<Planned> = Vec::new();
        let mut tables: Vec<Planned> = Vec::new();
        for (name, kind) in &touched {
            let info = self.info(name).with_context(|| format!("missing tensor {name}"))?.clone();
            let (lin, table) = uses.get(name).copied().unwrap_or((false, false));
            let dst = if table && !lin { &mut tables } else { &mut weights };
            match kind {
                None => {
                    let dtype: &'static str = match info.dtype.as_str() {
                        "BF16" => "BF16",
                        "F32" => "F32",
                        "F16" => "F16",
                        "I8" => "I8",
                        "U8" => "U8",
                        other => bail!("tensor {name}: dtype {other} not supported by the writer"),
                    };
                    dst.push(Planned {
                        name: name.clone(),
                        dtype,
                        shape: info.shape.clone(),
                        bytes: info.end - info.start,
                        src: Some((name.clone(), None, false)),
                    });
                }
                Some(k) => {
                    if info.dtype != "BF16" {
                        bail!("tensor {name}: only BF16 tensors can be quantized (is {})", info.dtype);
                    }
                    let n_out = info.shape[0];
                    let n_in: usize = info.shape[1..].iter().product();
                    if !n_in.is_multiple_of(QBLOCK) {
                        bail!("tensor {name}: row length {n_in} is not a multiple of {QBLOCK}");
                    }
                    let bpr = n_in / QBLOCK;
                    let (qd, qshape, qbytes, sd, sbytes) = match k {
                        QKind::Q8 => ("I8", vec![n_out, n_in], n_out * n_in, "F32", n_out * bpr * 4),
                        QKind::Q4 => ("U8", vec![n_out, n_in / 2], n_out * n_in / 2, "F16", n_out * bpr * 2),
                    };
                    dst.push(Planned {
                        name: format!("{name}.{}", k.suffix()),
                        dtype: qd,
                        shape: qshape,
                        bytes: qbytes,
                        src: Some((name.clone(), Some(*k), false)),
                    });
                    dst.push(Planned {
                        name: format!("{name}.{}s", k.suffix()),
                        dtype: sd,
                        shape: vec![n_out, bpr],
                        bytes: sbytes,
                        src: Some((name.clone(), Some(*k), true)),
                    });
                }
            }
        }
        // Clear outputs of an earlier run so the loader does not pick a stale layout.
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            let n = e.file_name().to_string_lossy().to_string();
            if n == "model.safetensors"
                || n == "model.safetensors.index.json"
                || ((n.starts_with("model-weights") || n.starts_with("model-tables")) && n.ends_with(".safetensors"))
            {
                std::fs::remove_file(e.path()).with_context(|| format!("remove {}", e.path().display()))?;
            }
        }
        // Shards of at most SHARD bytes: one allocation per file must stay below 2 GiB in wasm32
        // (isize::MAX), and smaller files download and resume better.
        let split = |plan: Vec<Planned>| -> Vec<Vec<Planned>> {
            let mut out: Vec<Vec<Planned>> = vec![Vec::new()];
            let mut size = 0usize;
            for o in plan {
                let est = o.bytes.next_multiple_of(16384);
                if size > 0 && size + est > SHARD {
                    out.push(Vec::new());
                    size = 0;
                }
                size += est;
                out.last_mut().unwrap().push(o);
            }
            out.retain(|s| !s.is_empty());
            out
        };
        let mut files: Vec<(String, Vec<Planned>)> = Vec::new();
        let weight_shards = split(weights);
        let table_shards = split(tables);
        if table_shards.is_empty() && weight_shards.len() == 1 {
            files.push(("model.safetensors".into(), weight_shards.into_iter().next().unwrap()));
        } else {
            let (nw, nt) = (weight_shards.len(), table_shards.len());
            for (i, p) in weight_shards.into_iter().enumerate() {
                files.push((format!("model-weights-{:05}-of-{:05}.safetensors", i + 1, nw), p));
            }
            for (i, p) in table_shards.into_iter().enumerate() {
                files.push((format!("model-tables-{:05}-of-{:05}.safetensors", i + 1, nt), p));
            }
        }
        let mut written = Vec::new();
        let mut map = serde_json::Map::new();
        let mut total = 0u64;
        let n_files = files.len();
        for (file, plan) in files {
            let p = dir.join(&file);
            let (size, names) = self.write_file(&p, plan, metadata)?;
            for n in names {
                map.insert(n, serde_json::Value::String(file.clone()));
            }
            total += size;
            written.push((p, size));
        }
        if n_files > 1 {
            let index = serde_json::json!({ "metadata": { "total_size": total }, "weight_map": map });
            std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec_pretty(&index)?)?;
        }
        Ok(written)
    }

    /// Writes one safetensors file from a plan; returns its size and the tensor names (no pads).
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    fn write_file(&self, out: &Path, plan: Vec<Planned>, metadata: &[(&str, &str)]) -> Result<(u64, Vec<String>)> {
        use crate::quant::{quantize_bf16, quantize_q4};
        use std::io::Write;
        const BIG: usize = 1 << 20;
        const PAGE_ALIGN: usize = 16384;
        // Lay out with alignment padding (explicit U8 pad tensors keep the buffer hole-free).
        let mut layout: Vec<Planned> = Vec::new();
        let mut off = 0usize;
        for (i, o) in plan.into_iter().enumerate() {
            let align = if o.bytes >= BIG { PAGE_ALIGN } else { 64 };
            let pad = off.next_multiple_of(align) - off;
            if pad > 0 {
                layout.push(Planned {
                    name: format!("__pad.{i}"),
                    dtype: "U8",
                    shape: vec![pad],
                    bytes: pad,
                    src: None,
                });
                off += pad;
            }
            off += o.bytes;
            layout.push(o);
        }
        let mut header = serde_json::Map::new();
        let mut meta = serde_json::Map::new();
        for (k, v) in metadata {
            meta.insert(k.to_string(), serde_json::Value::String(v.to_string()));
        }
        header.insert("__metadata__".into(), serde_json::Value::Object(meta));
        let mut off = 0usize;
        for o in &layout {
            header.insert(
                o.name.clone(),
                serde_json::json!({ "dtype": o.dtype, "shape": o.shape, "data_offsets": [off, off + o.bytes] }),
            );
            off += o.bytes;
        }
        let mut hjson = serde_json::to_vec(&header)?;
        // Pad the header with spaces (allowed by the format) so the data starts page-aligned.
        let target = (8 + hjson.len()).next_multiple_of(PAGE_ALIGN) - 8;
        hjson.resize(target, b' ');

        let tmp = out.with_extension("safetensors.partial");
        let mut f = std::io::BufWriter::with_capacity(8 << 20, File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?);
        f.write_all(&(hjson.len() as u64).to_le_bytes())?;
        f.write_all(&hjson)?;
        let mut cache: Option<(String, Vec<u8>, Vec<u8>)> = None; // last quantized tensor: (name, q, scales)
        for o in &layout {
            match &o.src {
                None => f.write_all(&vec![0u8; o.bytes])?,
                Some((name, None, _)) => f.write_all(self.raw(name)?)?,
                Some((name, Some(k), scales)) => {
                    if cache.as_ref().map(|c| &c.0) != Some(name) {
                        let info = self.info(name).unwrap();
                        let n_out = info.shape[0];
                        let n_in: usize = info.shape[1..].iter().product();
                        let b = self.raw(name)?;
                        let w = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) };
                        let (q, s) = match k {
                            QKind::Q8 => {
                                let m = quantize_bf16(w, n_out, n_in);
                                (as_bytes(&m.q).to_vec(), as_bytes(&m.scales).to_vec())
                            }
                            QKind::Q4 => {
                                let m = quantize_q4(w, n_out, n_in);
                                (as_bytes(&m.q).to_vec(), as_bytes(&m.scales).to_vec())
                            }
                        };
                        cache = Some((name.clone(), q, s));
                    }
                    let c = cache.as_ref().unwrap();
                    let data = if *scales { &c.2 } else { &c.1 };
                    debug_assert_eq!(data.len(), o.bytes);
                    f.write_all(data)?;
                }
            }
        }
        f.flush()?;
        drop(f);
        std::fs::rename(&tmp, out)?;
        let names = layout.into_iter().filter(|o| o.src.is_some()).map(|o| o.name).collect();
        Ok((std::fs::metadata(out)?.len(), names))
    }

    /// Checkpoint size (u64: a browser module's usize is 32-bit, checkpoints can be larger).
    pub fn total_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.len() as u64).sum()
    }

    /// Every shard's mapping; None if any shard is not memory-mapped (read into RAM, or kept by
    /// a browser host).
    pub fn mappings(&self) -> Option<Vec<MappedFile<'_>>> {
        self.shards.iter().map(|s| s.mapping()).collect()
    }

    pub fn preload(&self) {
        for s in &self.shards {
            s.preload();
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::quant::{quantize_bf16, quantize_q4};

    /// Writes a minimal bf16 safetensors file with the given 2-D tensors.
    fn write_bf16(path: &Path, tensors: &[(&str, usize, usize, Vec<u16>)]) {
        let mut header = serde_json::Map::new();
        let mut off = 0;
        for (n, r, c, _) in tensors {
            header.insert(
                n.to_string(),
                serde_json::json!({"dtype": "BF16", "shape": [r, c], "data_offsets": [off, off + r * c * 2]}),
            );
            off += r * c * 2;
        }
        let mut h = serde_json::to_vec(&header).unwrap();
        h.resize(h.len().next_multiple_of(8), b' '); // as the format requires
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&h);
        for (.., d) in tensors {
            out.extend(d.iter().flat_map(|v| v.to_le_bytes()));
        }
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn pre_quantized_round_trip() {
        let dir = std::env::temp_dir().join(format!("oxinfer-qtest-{}", std::process::id()));
        let out = dir.join("out");
        std::fs::create_dir_all(&out).unwrap();
        let (r, c) = (48, 96);
        let mk = |seed: u32| -> Vec<u16> {
            (0..r * c)
                .map(|i| ((((i as u32).wrapping_mul(2654435761) ^ seed) >> 20) as f32 / 4096.0 - 0.5).to_bits())
                .map(|b| (b >> 16) as u16)
                .collect()
        };
        let (a, b, t, n) = (mk(1), mk(2), mk(3), mk(4));
        write_bf16(
            &dir.join("model.safetensors"),
            &[
                ("a", r, c, a.clone()),
                ("b", r, c, b.clone()),
                ("tab", r, c, t.clone()),
                ("norm", 1, c, n[..c].to_vec()),
            ],
        );
        let files = ModelFiles::open(&dir, true).unwrap();
        // What a loader does: two matmul weights (q8, q4), a lookup-only table (q4), a raw tensor.
        files.bf16("a").unwrap();
        files.mark_quant("a", QKind::Q8);
        files.note_use("a", true, false);
        files.bf16("b").unwrap();
        files.mark_quant("b", QKind::Q4);
        files.note_use("b", true, false);
        files.bf16("tab").unwrap();
        files.mark_quant("tab", QKind::Q4);
        files.note_use("tab", false, true);
        files.bf16("norm").unwrap();
        let written = files.write_quantized(&out, &[("oxinfer_quant", "q4")]).unwrap();
        assert_eq!(written.len(), 2, "weights and tables shards: {written:?}");

        let back = ModelFiles::open(&out, false).unwrap();
        assert_eq!(back.metadata("oxinfer_quant"), Some("q4"));
        assert_eq!(back.bf16("norm").unwrap(), &n[..c]);
        let (ea, eb, et) = (quantize_bf16(&a, r, c), quantize_q4(&b, r, c), quantize_q4(&t, r, c));
        match unsafe { back.quantized("a", r, c) }.unwrap() {
            Some(Stored::Q8(m)) => assert!(m.q[..] == ea.q[..] && m.scales[..] == ea.scales[..] && m.q.is_borrowed()),
            _ => panic!("a should be stored as q8"),
        }
        match unsafe { back.quantized("b", r, c) }.unwrap() {
            Some(Stored::Q4(m)) => assert!(m.q[..] == eb.q[..] && m.scales[..] == eb.scales[..]),
            _ => panic!("b should be stored as q4"),
        }
        match unsafe { back.quantized("tab", r, c) }.unwrap() {
            Some(Stored::Q4(m)) => assert!(m.q[..] == et.q[..] && m.scales[..] == et.scales[..]),
            _ => panic!("tab should be stored as q4"),
        }
        // Large tensors are page-aligned (none here); every tensor is at least 64-byte aligned.
        for name in ["a.q8", "a.q8s", "b.q4", "b.q4s", "tab.q4"] {
            assert_eq!(back.raw(name).unwrap().as_ptr() as usize % 64, 0, "{name} alignment");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
