//! Gemma 4 text decoder: embeddings, per-layer embeddings (PLE), sliding/global
//! attention with KV sharing, gated-GELU MLP, tied softcapped LM head.

use crate::config::TextConfig;
use crate::ggml::{GgmlType, GgufMatrix};
use crate::kernels::*;
use crate::prof::{self, Cat};
use crate::quant::{Q4Matrix, QKind, QMatrix, quantize_bf16, quantize_q4};
use crate::safetensors::{ModelFiles, Stored};
use anyhow::{Context, Result, bail};

/// Weight precision for the streamed (linear) weights.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Quant {
    Bf16,
    Q8,
    /// Q4_0 layer weights; the LM head and PLE projection stay Q8.
    Q4,
}

impl Quant {
    /// The stored quantized format, None for bf16.
    pub fn kind(self) -> Option<QKind> {
        match self {
            Quant::Bf16 => None,
            Quant::Q8 => Some(QKind::Q8),
            Quant::Q4 => Some(QKind::Q4),
        }
    }
    pub fn parse(s: &str) -> Option<Quant> {
        match s {
            "bf16" => Some(Quant::Bf16),
            "q8" => Some(Quant::Q8),
            "q4" => Some(Quant::Q4),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Quant::Bf16 => "bf16",
            Quant::Q8 => "q8",
            Quant::Q4 => "q4",
        }
    }
    /// The precision a pre-quantized checkpoint was written with (overrides `--quant`).
    /// MLX checkpoints count as Q4 (their 4-bit matrices load as affine Q4).
    pub fn stored(files: &ModelFiles) -> Option<Quant> {
        files
            .metadata("oxinfer_quant")
            .and_then(Quant::parse)
            .or_else(|| files.is_mlx().then_some(Quant::Q4))
    }
}

/// Precision of the embedding tables written by `--save-quantized` (rows are dequantized one
/// token at a time, so they are never streamed in full): `OXINFER_TABLE_QUANT=bf16|q8|q4`,
/// default the model precision.
pub fn table_quant(quant: Quant) -> Quant {
    std::env::var("OXINFER_TABLE_QUANT")
        .ok()
        .and_then(|v| Quant::parse(&v))
        .unwrap_or(quant)
}

/// An embedding table `[rows, cols]` read one row at a time: borrowed bf16 or pre-quantized.
pub enum Table<'a> {
    Bf16(&'a [u16]),
    Q8(QMatrix),
    Q4(Q4Matrix),
    /// A GGUF table in any `GgmlType`, dequantized row by row.
    Gguf(GgufMatrix),
    /// Pre-quantized rows kept by the browser host, fetched one row per lookup.
    /// Values and scales can sit in different shards, hence two files.
    Host {
        qf: crate::fsx::HostFile,
        sf: crate::fsx::HostFile,
        kind: QKind,
        q_off: u64,
        s_off: u64,
        cols: usize,
    },
    /// An MLX affine-quantized table, dequantized row by row from the checkpoint.
    Mlx {
        parts: [crate::safetensors::TablePart<'a>; 3],
        q: crate::safetensors::MlxQuant,
    },
}

impl<'a> Table<'a> {
    /// A GGUF table's raw bytes (`rows` rows of `cols` elements of `kind`), borrowed for 'a.
    pub fn from_gguf(kind: GgmlType, bytes: &'a [u8], rows: usize, cols: usize) -> Result<Self> {
        if kind == GgmlType::BF16 && bytes.len() == rows * cols * 2 && (bytes.as_ptr() as usize).is_multiple_of(2) {
            // SAFETY: length and alignment checked; any bit pattern is a valid u16.
            return Ok(Table::Bf16(unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const u16, rows * cols)
            }));
        }
        // SAFETY: Table<'a> cannot outlive `bytes`.
        Ok(Table::Gguf(unsafe { GgufMatrix::borrowed(kind, bytes, rows, cols)? }))
    }

    /// The memory row `r` (of `cols` values) is read from, for tables that borrow mapped
    /// memory (weight streaming reads these spans ahead of the lookup).
    pub fn row_ranges(&self, r: usize, cols: usize) -> Vec<(usize, usize)> {
        fn span<T>(v: &[T]) -> (usize, usize) {
            (v.as_ptr() as usize, std::mem::size_of_val(v))
        }
        let bpr = cols / crate::quant::QBLOCK;
        match self {
            Table::Bf16(w) => vec![span(&w[r * cols..(r + 1) * cols])],
            Table::Q8(m) => vec![span(&m.q[r * cols..(r + 1) * cols]), span(&m.scales[r * bpr..(r + 1) * bpr])],
            Table::Q4(m) => vec![
                span(&m.q[r * cols / 2..(r + 1) * cols / 2]),
                span(&m.scales[r * bpr..(r + 1) * bpr]),
            ],
            Table::Gguf(m) => vec![span(m.row(r))],
            Table::Host { .. } | Table::Mlx { .. } => Vec::new(),
        }
    }

    /// Row `r` (of length `out.len()`) as f32.
    pub fn row(&self, r: usize, out: &mut [f32]) {
        let n = out.len();
        match self {
            Table::Bf16(w) => {
                for (o, &v) in out.iter_mut().zip(&w[r * n..(r + 1) * n]) {
                    *o = bf16_to_f32(v);
                }
            }
            Table::Q8(m) => crate::quant::dequantize_rows(m, r, r + 1, out),
            Table::Q4(m) => crate::quant::dequantize_rows_q4(m, r, r + 1, out),
            Table::Gguf(m) => m.dequantize_rows(r, r + 1, out),
            Table::Mlx { parts, q } => {
                let (wb, gb) = (n * q.bits / 8, n / q.group * 2);
                let mut w = vec![0u32; wb / 4];
                let mut sb = vec![0u16; gb / 2];
                let mut bb = vec![0u16; gb / 2];
                let as_bytes = |v: &mut [u16]| unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 2) };
                let ok = parts[0]
                    .read(r * wb, unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr() as *mut u8, wb) })
                    .and_then(|_| parts[1].read(r * gb, as_bytes(&mut sb)))
                    .and_then(|_| parts[2].read(r * gb, as_bytes(&mut bb)));
                if let Err(e) = ok {
                    panic!("table row {r}: {e}");
                }
                crate::quant::mlx_dequant_row(&w, &sb, &bb, q.bits, q.group, out);
            }
            &Table::Host {
                qf,
                sf,
                kind,
                q_off,
                s_off,
                cols,
            } => {
                use crate::quant::{PageVec, QBLOCK};
                let bpr = cols / QBLOCK;
                let (qb, sb) = match kind {
                    QKind::Q8 => (cols, bpr * 4),
                    QKind::Q4 => (cols / 2, bpr * 2),
                };
                let mut buf = PageVec::<u64>::zeroed((qb + sb).div_ceil(8));
                let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, qb + sb) };
                let (qbuf, sbuf) = bytes.split_at_mut(qb);
                let ok = crate::fsx::read_host(qf, q_off + (r * qb) as u64, qbuf)
                    .and_then(|_| crate::fsx::read_host(sf, s_off + (r * sb) as u64, sbuf));
                if let Err(e) = ok {
                    panic!("table row {r}: {e}");
                }
                // One-row matrices over the scratch buffer (scales sit right after the values,
                // which keeps them aligned because `qb` is a multiple of 16).
                unsafe {
                    match kind {
                        QKind::Q8 => {
                            let q = std::slice::from_raw_parts(qbuf.as_ptr() as *const i8, qb);
                            let sc = std::slice::from_raw_parts(sbuf.as_ptr() as *const f32, bpr);
                            let m = QMatrix {
                                q: crate::quant::QBuf::borrowed(q),
                                scales: crate::quant::QBuf::borrowed(sc),
                                n_out: 1,
                                n_in: cols,
                            };
                            crate::quant::dequantize_rows(&m, 0, 1, out);
                        }
                        QKind::Q4 => {
                            let sc = std::slice::from_raw_parts(sbuf.as_ptr() as *const u16, bpr);
                            let m = Q4Matrix {
                                q: crate::quant::QBuf::borrowed(qbuf),
                                scales: crate::quant::QBuf::borrowed(sc),
                                n_out: 1,
                                n_in: cols,
                                offsets: None,
                            };
                            crate::quant::dequantize_rows_q4(&m, 0, 1, out);
                        }
                    }
                }
            }
        }
    }

    /// Bytes held in memory by this table (0 when the host keeps it).
    pub fn bytes(&self) -> usize {
        match self {
            Table::Bf16(w) => w.len() * 2,
            Table::Q8(m) => m.bytes(),
            Table::Q4(m) => m.bytes(),
            Table::Gguf(m) => m.bytes(),
            Table::Host { .. } => 0,
            Table::Mlx { parts, .. } => parts
                .iter()
                .map(|p| match p {
                    crate::safetensors::TablePart::Mem(b) => b.len(),
                    crate::safetensors::TablePart::Host(..) => 0,
                })
                .sum(),
        }
    }

    /// Loads table `full` (`[rows, cols]`): the pre-quantized form when the checkpoint has one,
    /// otherwise bf16, recording `store_as` for `--save-quantized`.
    pub fn load(files: &'a ModelFiles, full: &str, rows: usize, cols: usize, store_as: Quant) -> Result<Self> {
        files.note_use(full, false, true);
        if let Some(g) = files.ggml(full) {
            let (kind, bytes) = g?;
            return Table::from_gguf(kind, bytes, rows, cols).with_context(|| format!("tensor {full}"));
        }
        for kind in [QKind::Q8, QKind::Q4] {
            let (qn, sn) = (format!("{full}.{}", kind.suffix()), format!("{full}.{}s", kind.suffix()));
            if let (Some((qf, q_off)), Some((sf, s_off))) = (files.host_tensor(&qn), files.host_tensor(&sn)) {
                let bpr = cols / crate::quant::QBLOCK;
                let (qshape, sdt) = match kind {
                    QKind::Q8 => (vec![rows, cols], "F32"),
                    QKind::Q4 => (vec![rows, cols / 2], "F16"),
                };
                let (qi, si) = (files.info(&qn).unwrap(), files.info(&sn).unwrap());
                if qi.shape != qshape || si.shape != [rows, bpr] || si.dtype != sdt {
                    bail!(
                        "table {full}: stored shapes {:?} / {:?} do not match [{rows}, {cols}]",
                        qi.shape,
                        si.shape
                    );
                }
                files.mark_quant(full, kind);
                return Ok(Table::Host {
                    qf,
                    sf,
                    kind,
                    q_off,
                    s_off,
                    cols,
                });
            }
        }
        if let (Some(q), Some(parts)) = (files.mlx(full), files.mlx_parts(full)) {
            let info = files.info(full).unwrap();
            if info.shape != [rows, cols] {
                bail!("table {full} has shape {:?}, expected [{rows}, {cols}]", info.shape);
            }
            return Ok(Table::Mlx { parts, q });
        }
        // SAFETY: the table borrows `files` for 'a, which outlives every model holding it.
        if let Some(st) = unsafe { files.quantized(full, rows, cols)? } {
            return Ok(match st {
                Stored::Q8(m) => Table::Q8(m),
                Stored::Q4(m) => Table::Q4(m),
            });
        }
        let info = files.info(full).with_context(|| format!("missing tensor {full}"))?;
        if info.shape.iter().product::<usize>() != rows * cols {
            bail!("tensor {full} has shape {:?}, expected [{rows}, {cols}]", info.shape);
        }
        let w = files.bf16(full)?;
        if let Some(k) = store_as.kind() {
            files.mark_quant(full, k);
        }
        Ok(Table::Bf16(w))
    }
}

/// A streamed linear weight `[n_out, n_in]` named `full`: pre-quantized from the checkpoint when
/// present (zero-copy), otherwise bf16 quantized to `q` at load (and recorded for `--save-quantized`).
pub fn load_lin<'a>(files: &'a ModelFiles, full: &str, n_out: usize, n_in: usize, q: Quant) -> Result<Lin<'a>> {
    files.note_use(full, true, false);
    // GGUF: used in its own format whatever `q` asks for.
    if let Some(g) = files.ggml(full) {
        let (kind, bytes) = g?;
        return Lin::from_gguf(kind, bytes, n_out, n_in).with_context(|| format!("tensor {full}"));
    }
    // SAFETY: the matrix borrows `files` for 'a, which outlives every model holding it.
    if let Some(st) = unsafe { files.quantized(full, n_out, n_in)? } {
        return Ok(match st {
            Stored::Q8(m) => Lin::q8(m),
            Stored::Q4(m) => Lin::q4(m),
        });
    }
    // MLX 4-bit: repacked as affine Q4 whatever `q` asks for (other widths dequantize to bf16 below).
    if let Some(m) = files.mlx_q4(full, n_out, n_in)? {
        return Ok(Lin::q4(m));
    }
    let info = files.info(full).with_context(|| format!("missing tensor {full}"))?;
    if info.shape != [n_out, n_in] {
        bail!("tensor {full} has shape {:?}, expected [{n_out}, {n_in}]", info.shape);
    }
    let w = files.bf16(full)?;
    if let Some(k) = q.kind() {
        files.mark_quant(full, k);
    }
    Ok(match q {
        Quant::Bf16 => Lin::bf16(w),
        Quant::Q8 => Lin::q8(quantize_bf16(w, n_out, n_in)),
        Quant::Q4 => Lin::q4(quantize_q4(w, n_out, n_in)),
    })
}

/// The experts of a fused `[n_exp, n_out, n_in]` tensor (MoE `gate_up_proj` / `down_proj`), one
/// linear layer each. Every expert is a contiguous block of rows, so quantizing the experts one by
/// one gives the same bytes as quantizing the tensor whole (as `--save-quantized` writes it, with
/// shape `[n_exp, n_out * n_in]`); pre-quantized experts are borrowed views of the stored matrix.
pub fn load_lin_experts<'a>(files: &'a ModelFiles, full: &str, n_exp: usize, n_out: usize, n_in: usize, q: Quant) -> Result<Vec<Lin<'a>>> {
    files.note_use(full, true, false);
    if let Some(g) = files.ggml(full) {
        let (kind, bytes) = g?;
        if !bytes.len().is_multiple_of(n_exp) {
            bail!("tensor {full}: {} bytes for {n_exp} experts", bytes.len());
        }
        return bytes
            .chunks(bytes.len() / n_exp)
            .map(|b| Lin::from_gguf(kind, b, n_out, n_in).with_context(|| format!("tensor {full}")))
            .collect();
    }
    let (rows, bpr) = (n_out, n_in / crate::quant::QBLOCK);
    // SAFETY: the stored matrix borrows `files` for 'a (zero-copy), and so do its per-expert views.
    if let Some(st) = unsafe { files.quantized(full, n_exp, n_out * n_in)? } {
        fn view<T: Copy>(s: &[T], e: usize, n: usize) -> crate::quant::QBuf<T> {
            unsafe { crate::quant::QBuf::borrowed(&s[e * n..(e + 1) * n]) }
        }
        return Ok((0..n_exp)
            .map(|e| match &st {
                Stored::Q8(m) => Lin::q8(QMatrix {
                    q: view(&m.q, e, rows * n_in),
                    scales: view(&m.scales, e, rows * bpr),
                    n_out,
                    n_in,
                }),
                Stored::Q4(m) => Lin::q4(Q4Matrix {
                    q: view(&m.q, e, rows * n_in / 2),
                    scales: view(&m.scales, e, rows * bpr),
                    n_out,
                    n_in,
                    offsets: None,
                }),
            })
            .collect());
    }
    if files.is_mlx() && files.has(&format!("{full}.scales")) {
        bail!("{full}: MLX-quantized experts are not supported yet");
    }
    let info = files.info(full).with_context(|| format!("missing tensor {full}"))?;
    if info.shape != [n_exp, n_out, n_in] {
        bail!("tensor {full} has shape {:?}, expected [{n_exp}, {n_out}, {n_in}]", info.shape);
    }
    let w = files.bf16(full)?;
    if let Some(k) = q.kind() {
        files.mark_quant(full, k);
    }
    Ok(w.chunks(n_out * n_in)
        .map(|w| match q {
            Quant::Bf16 => Lin::bf16(w),
            Quant::Q8 => Lin::q8(quantize_bf16(w, n_out, n_in)),
            Quant::Q4 => Lin::q4(quantize_q4(w, n_out, n_in)),
        })
        .collect())
}

/// A linear layer's weight `[n_out, n_in]`: bf16 (borrowed from the checkpoint or owned) or
/// quantized, plus lazily created device-side state when an accelerator backend is active.
pub struct Lin<'a> {
    pub w: Weight<'a>,
    dev: std::sync::OnceLock<Option<crate::gpu::DeviceWeight>>,
    /// Device state is created per call and dropped right after (see `set_transient_device`).
    transient: std::sync::atomic::AtomicBool,
}

pub enum Weight<'a> {
    /// Borrowed for 'a (see `Lin::bf16`) or owned (`Lin::bf16_owned`).
    Bf16(crate::quant::QBuf<u16>, std::marker::PhantomData<&'a [u16]>),
    Q8(QMatrix),
    Q4(Q4Matrix),
    /// A llama.cpp block format (GGUF checkpoints), any `GgmlType`.
    Gguf(GgufMatrix),
}

impl<'a> Lin<'a> {
    pub fn bf16(w: &'a [u16]) -> Self {
        // SAFETY: the PhantomData ties the borrow to 'a, so `w` outlives this Lin.
        Self {
            w: Weight::Bf16(unsafe { crate::quant::QBuf::borrowed(w) }, std::marker::PhantomData),
            dev: std::sync::OnceLock::new(),
            transient: std::sync::atomic::AtomicBool::new(false),
        }
    }
    /// bf16 weights converted or reordered at load time (e.g. from fp16 checkpoints).
    pub fn bf16_owned(w: crate::quant::PageVec<u16>) -> Self {
        Self {
            w: Weight::Bf16(w.into(), std::marker::PhantomData),
            dev: std::sync::OnceLock::new(),
            transient: std::sync::atomic::AtomicBool::new(false),
        }
    }
    pub fn q8(m: QMatrix) -> Self {
        Self {
            w: Weight::Q8(m),
            dev: std::sync::OnceLock::new(),
            transient: std::sync::atomic::AtomicBool::new(false),
        }
    }
    pub fn q4(m: Q4Matrix) -> Self {
        Self {
            w: Weight::Q4(m),
            dev: std::sync::OnceLock::new(),
            transient: std::sync::atomic::AtomicBool::new(false),
        }
    }
    /// A GGUF weight (`GgufMatrix::borrowed` keeps it zero-copy over a mapped file).
    pub fn gguf(m: GgufMatrix) -> Self {
        Self {
            w: Weight::Gguf(m),
            dev: std::sync::OnceLock::new(),
            transient: std::sync::atomic::AtomicBool::new(false),
        }
    }
    /// A GGUF tensor's raw bytes (`n_out` rows of `n_in` elements of `kind`), borrowed for 'a.
    /// BF16 becomes a plain bf16 weight (GPU-capable, bf16 kernels); everything else a `GgufMatrix`.
    pub fn from_gguf(kind: GgmlType, bytes: &'a [u8], n_out: usize, n_in: usize) -> Result<Self> {
        if kind == GgmlType::BF16 && bytes.len() == n_out * n_in * 2 && (bytes.as_ptr() as usize).is_multiple_of(2) {
            // SAFETY: length and alignment checked; any bit pattern is a valid u16.
            return Ok(Self::bf16(unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const u16, n_out * n_in)
            }));
        }
        // SAFETY: Lin<'a> cannot outlive `bytes`.
        Ok(Self::gguf(unsafe { GgufMatrix::borrowed(kind, bytes, n_out, n_in)? }))
    }
    pub fn bytes(&self) -> usize {
        self.as_ref().bytes()
    }
    /// The memory this weight reads, as `(address, length)` spans (for weight streaming).
    pub fn mem_ranges(&self) -> Vec<(usize, usize)> {
        fn span<T>(v: &[T]) -> (usize, usize) {
            (v.as_ptr() as usize, std::mem::size_of_val(v))
        }
        match &self.w {
            Weight::Bf16(w, _) => vec![span(&w[..])],
            Weight::Q8(m) => vec![span(&m.q[..]), span(&m.scales[..])],
            Weight::Q4(m) => {
                let mut v = vec![span(&m.q[..]), span(&m.scales[..])];
                v.extend(m.offsets.as_ref().map(|o| span(&o.v[..])));
                v
            }
            Weight::Gguf(m) => vec![span(&m.data[..])],
        }
    }
    pub fn as_ref(&self) -> crate::gpu::WeightRef<'_> {
        match &self.w {
            Weight::Bf16(w, _) => crate::gpu::WeightRef::Bf16(w),
            Weight::Q8(m) => crate::gpu::WeightRef::Q8(m),
            Weight::Q4(m) => crate::gpu::WeightRef::Q4(m),
            Weight::Gguf(m) => crate::gpu::WeightRef::Gguf(m),
        }
    }
    /// Device-side state for this layer on the active accelerator, whatever the op size (callers
    /// that bypass `Backend::wants`, such as convolutions, decide for themselves).
    pub(crate) fn device_weight(&self, n_out: usize, n_in: usize) -> Option<&crate::gpu::DeviceWeight> {
        let be = crate::gpu::active()?;
        self.dev.get_or_init(|| be.upload(&self.as_ref(), n_out, n_in)).as_ref()
    }
    /// Device-side state for this layer if the active accelerator wants the op at this size.
    fn device_state(&self, n_out: usize, n_in: usize, t: usize) -> Option<&crate::gpu::DeviceWeight> {
        let be = crate::gpu::active()?;
        if !be.wants(n_out, n_in, t) {
            return None;
        }
        self.dev.get_or_init(|| be.upload(&self.as_ref(), n_out, n_in)).as_ref()
    }
    /// Don't keep device-side state between calls: an accelerator mapping of streamed weights
    /// (Metal wraps them in place) keeps their pages from leaving the page cache while it lives.
    pub fn set_transient_device(&self, on: bool) {
        self.transient.store(on, std::sync::atomic::Ordering::Relaxed);
    }
    pub(crate) fn is_transient(&self) -> bool {
        self.transient.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Fused matmuls over one shared input (any mix of storage formats).
pub(crate) fn linear_multi(jobs: &mut [(&Lin, usize, &mut [f32])], n_in: usize, x: &[f32], t: usize) {
    if jobs.is_empty() {
        return;
    }
    // Accelerator first (one submission for the whole batch); whatever it declines runs on the CPU.
    let mut rest: Vec<(&Lin, usize, &mut [f32])> = Vec::with_capacity(jobs.len());
    let be = crate::gpu::active();
    // Transient device state (streamed weights), released when this call returns.
    let owned: Vec<Option<crate::gpu::DeviceWeight>> = jobs
        .iter()
        .map(|(l, n, _)| match be {
            Some(b) if l.is_transient() && b.wants(*n, n_in, t) => b.upload(&l.as_ref(), *n, n_in),
            _ => None,
        })
        .collect();
    let mut gpu: Vec<crate::gpu::Job> = Vec::with_capacity(jobs.len());
    let mut gpu_meta: Vec<(&Lin, usize)> = Vec::with_capacity(jobs.len());
    for ((l, n, y), own) in jobs.iter_mut().zip(&owned) {
        let dev = match own {
            Some(d) => Some(d),
            None if l.is_transient() => None,
            None => be.and_then(|_| l.device_state(*n, n_in, t)),
        };
        match dev {
            Some(dev) => {
                gpu.push(crate::gpu::Job {
                    w: l.as_ref(),
                    dev,
                    n_out: *n,
                    y,
                });
                gpu_meta.push((l, *n));
            }
            None => rest.push((l, *n, &mut **y)),
        }
    }
    if !gpu.is_empty() {
        let _timer = prof::start(Cat::Matmul);
        if !be.unwrap().matmul(x, t, n_in, &mut gpu) {
            // Backend failed at run time: run these on the CPU too.
            for (job, (l, n)) in gpu.into_iter().zip(gpu_meta) {
                rest.push((l, n, job.y));
            }
        }
    }
    if !rest.is_empty() {
        linear_multi_cpu(&mut rest, n_in, x, t);
    }
}

/// The CPU part of [`linear_multi`]: one fused region per storage format (GGUF checkpoints mix
/// formats freely, so mixed batches are split; GGUF matrices of different types share one).
fn linear_multi_cpu(jobs: &mut [(&Lin, usize, &mut [f32])], n_in: usize, x: &[f32], t: usize) {
    let fmt = |l: &Lin| match l.w {
        Weight::Bf16(..) => 0,
        Weight::Q8(_) => 1,
        Weight::Q4(_) => 2,
        Weight::Gguf(_) => 3,
    };
    let f0 = fmt(jobs[0].0);
    if jobs.iter().any(|(l, _, _)| fmt(l) != f0) {
        for f in 0..4 {
            let mut sub: Vec<(&Lin, usize, &mut [f32])> = jobs
                .iter_mut()
                .filter(|(l, _, _)| fmt(l) == f)
                .map(|(l, n, y)| (*l, *n, &mut **y))
                .collect();
            if !sub.is_empty() {
                linear_multi_cpu(&mut sub, n_in, x, t);
            }
        }
        return;
    }
    fn mats<'m, M>(jobs: &'m mut [(&Lin, usize, &mut [f32])], get: impl Fn(&'m Weight) -> &'m M) -> Vec<(&'m M, &'m mut [f32])> {
        jobs.iter_mut().map(|(l, _, y)| (get(&l.w), &mut **y)).collect()
    }
    match f0 {
        0 => {
            let mut v: Vec<(&[u16], usize, &mut [f32])> = jobs
                .iter_mut()
                .map(|(l, n, y)| match &l.w {
                    Weight::Bf16(w, _) => (&w[..], *n, &mut **y),
                    _ => unreachable!(),
                })
                .collect();
            matmul_multi(&mut v, n_in, x, t);
        }
        1 => matmul_quant_multi(
            &mut mats(jobs, |w| match w {
                Weight::Q8(m) => m,
                _ => unreachable!(),
            }),
            x,
            t,
        ),
        2 => matmul_quant_multi(
            &mut mats(jobs, |w| match w {
                Weight::Q4(m) => m,
                _ => unreachable!(),
            }),
            x,
            t,
        ),
        _ => matmul_quant_multi(
            &mut mats(jobs, |w| match w {
                Weight::Gguf(m) => m,
                _ => unreachable!(),
            }),
            x,
            t,
        ),
    }
}

/// Grouped matmuls on the CPU (see [`matmul_grouped`]): `ls[j]` over rows `segs[j]` of `x`.
/// All weights must have the same shape and precision.
pub(crate) fn linear_grouped(ls: &[&Lin], segs: &[(usize, usize)], n_out: usize, n_in: usize, x: &[f32], y: &mut [f32]) {
    match &ls[0].w {
        Weight::Bf16(..) => {
            let ws: Vec<&[u16]> = ls
                .iter()
                .map(|l| match &l.w {
                    Weight::Bf16(w, _) => &w[..],
                    _ => panic!("mixed precision in grouped matmul"),
                })
                .collect();
            matmul_grouped(&ws, segs, n_out, n_in, x, y);
        }
        Weight::Q8(_) => {
            let ms: Vec<&QMatrix> = ls
                .iter()
                .map(|l| match &l.w {
                    Weight::Q8(m) => m,
                    _ => panic!("mixed precision in grouped matmul"),
                })
                .collect();
            matmul_quant_grouped(&ms, segs, x, y);
        }
        Weight::Q4(_) => {
            let ms: Vec<&Q4Matrix> = ls
                .iter()
                .map(|l| match &l.w {
                    Weight::Q4(m) => m,
                    _ => panic!("mixed precision in grouped matmul"),
                })
                .collect();
            matmul_quant_grouped(&ms, segs, x, y);
        }
        Weight::Gguf(_) => {
            let ms: Vec<&GgufMatrix> = ls
                .iter()
                .map(|l| match &l.w {
                    Weight::Gguf(m) => m,
                    _ => panic!("mixed precision in grouped matmul"),
                })
                .collect();
            matmul_quant_grouped(&ms, segs, x, y);
        }
    }
}

pub(crate) fn linear(l: &Lin, n_out: usize, n_in: usize, x: &[f32], t: usize, y: &mut [f32]) {
    linear_multi(&mut [(l, n_out, y)], n_in, x, t);
}

struct PleLayer<'a> {
    gate: Lin<'a>,   // [P, H]
    proj: Lin<'a>,   // [H, P]
    norm: &'a [u16], // [H]
}

struct PleGlobal<'a> {
    embed: Table<'a>, // [V, L*P]
    proj: Lin<'a>,    // [L*P, H]
    norm: &'a [u16],  // [P]
}

pub struct Layer<'a> {
    pub sliding: bool,
    pub hd: usize,
    pub n_kv: usize,
    pub kv_src: usize,
    pub own_kv: bool,
    pub k_eq_v: bool,
    pub inter: usize,
    q: Lin<'a>,
    k: Option<Lin<'a>>,
    v: Option<Lin<'a>>,
    o: Lin<'a>,
    // Norm weights in f32 (with Gemma 3's `1 +` folded in).
    q_norm: Vec<f32>,
    k_norm: Option<Vec<f32>>,
    ln_in: Vec<f32>,
    ln_post_attn: Vec<f32>,
    ln_pre_ffn: Vec<f32>,
    ln_post_ffn: Vec<f32>,
    gate: Lin<'a>,
    up: Lin<'a>,
    down: Lin<'a>,
    ple: Option<PleLayer<'a>>,
    layer_scalar: f32,
}

pub struct Model<'a> {
    pub cfg: TextConfig,
    pub quant: Quant,
    embed: Table<'a>, // [V, H] for lookups
    head: Lin<'a>,    // tied LM head (same values as `embed`)
    ple: Option<PleGlobal<'a>>,
    pub layers: Vec<Layer<'a>>,
    final_norm: Vec<f32>,
    inv_freq_sliding: Vec<f32>,
    inv_freq_full: Vec<f32>,
}

#[derive(Clone)]
pub struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
    pub n_kv: usize,
    pub hd: usize,
    pub len: usize,
}

impl LayerCache {
    pub(crate) fn new(n_kv: usize, hd: usize) -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            n_kv,
            hd,
            len: 0,
        }
    }
    pub(crate) fn push(&mut self, k: &[f32], v: &[f32], t: usize) {
        debug_assert_eq!(k.len(), t * self.n_kv * self.hd);
        self.k.extend_from_slice(k);
        self.v.extend_from_slice(v);
        self.len += t;
    }
    #[inline(always)]
    pub fn k_at(&self, pos: usize, kvh: usize) -> &[f32] {
        let o = (pos * self.n_kv + kvh) * self.hd;
        &self.k[o..o + self.hd]
    }
    #[inline(always)]
    pub fn v_at(&self, pos: usize, kvh: usize) -> &[f32] {
        let o = (pos * self.n_kv + kvh) * self.hd;
        &self.v[o..o + self.hd]
    }
    pub(crate) fn truncate(&mut self, len: usize) {
        if len < self.len {
            self.k.truncate(len * self.n_kv * self.hd);
            self.v.truncate(len * self.n_kv * self.hd);
            self.len = len;
        }
    }
}

/// Mutable decoding state: KV caches and the current position.
#[derive(Clone)]
pub struct Session {
    caches: Vec<Option<LayerCache>>,
    pub pos: usize,
}

impl Session {
    /// Drops cached positions at or beyond `pos` (rewinding after rejected speculative tokens).
    pub fn truncate(&mut self, pos: usize) {
        for c in self.caches.iter_mut().flatten() {
            c.truncate(pos);
        }
        self.pos = self.pos.min(pos);
    }

    /// The KV cache owned by `layer` (which must not be a KV-sharing layer).
    pub fn cache(&self, layer: usize) -> &LayerCache {
        self.caches[layer].as_ref().expect("layer has no own KV cache")
    }
}

impl<'a> Model<'a> {
    pub fn load(cfg: TextConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        let has = |n: &str| files.has(n) || files.has(&format!("{n}.q8")) || files.has(&format!("{n}.q4"));
        let prefix = if has("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else if has("model.embed_tokens.weight") {
            "model."
        } else if has("embed_tokens.weight") {
            "" // a bare text model (embedding checkpoints)
        } else {
            bail!("cannot find embed_tokens in checkpoint");
        };
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nl = cfg.num_hidden_layers;
        let p = cfg.hidden_size_per_layer_input;

        let get = |name: &str, shape: &[usize]| -> Result<&'a [u16]> {
            let full = format!("{prefix}{name}");
            let info = files.info(&full).with_context(|| format!("missing tensor {full}"))?;
            if info.shape != shape {
                bail!("tensor {full} has shape {:?}, expected {:?}", info.shape, shape);
            }
            files.bf16(&full)
        };
        // A streamed linear weight: borrowed bf16 or quantized on load.
        let lin_with = |name: &str, n_out: usize, n_in: usize, q: Quant| load_lin(files, &format!("{prefix}{name}"), n_out, n_in, q);
        let lin = |name: &str, n_out: usize, n_in: usize| lin_with(name, n_out, n_in, quant);
        // The PLE projection feeds every layer directly: keep it at Q8 under Q4. The LM head follows
        // OXINFER_HEAD_Q8=1 (Q8) or defaults to the model precision.
        let hi_quant = if quant == Quant::Q4 { Quant::Q8 } else { quant };
        let head_quant = if quant == Quant::Q4 && std::env::var("OXINFER_HEAD_Q8").is_ok() {
            Quant::Q8
        } else {
            quant
        };

        // Tied: lookups and the LM head read the same tensor (one copy in a pre-quantized file).
        let embed = Table::load(files, &format!("{prefix}embed_tokens.weight"), cfg.vocab_size, h, head_quant)?;
        let head = lin_with("embed_tokens.weight", cfg.vocab_size, h, head_quant)?;
        // Norm weights in f32 straight from the file (fp32 checkpoints keep their precision).
        let off = cfg.norm_offset();
        let nw = |name: &str, shape: &[usize]| -> Result<Vec<f32>> {
            let full = format!("{prefix}{name}");
            let info = files.info(&full).with_context(|| format!("missing tensor {full}"))?;
            if info.shape != shape {
                bail!("tensor {full} has shape {:?}, expected {:?}", info.shape, shape);
            }
            Ok(files.f32(&full)?.into_iter().map(|v| off + v).collect())
        };
        let final_norm = nw("norm.weight", &[h])?;
        let ple = if cfg.has_ple() {
            Some(PleGlobal {
                embed: Table::load(
                    files,
                    &format!("{prefix}embed_tokens_per_layer.weight"),
                    cfg.vocab_size_per_layer_input,
                    nl * p,
                    table_quant(quant),
                )?,
                proj: lin_with("per_layer_model_projection.weight", nl * p, h, hi_quant)?,
                norm: get("per_layer_projection_norm.weight", &[p])?,
            })
        } else {
            None
        };

        let mut layers = Vec::with_capacity(nl);
        for l in 0..nl {
            let lp = format!("layers.{l}.");
            let g = |n: &str, s: &[usize]| get(&format!("{lp}{n}"), s);
            let gl = |n: &str, n_out: usize, n_in: usize| lin(&format!("{lp}{n}"), n_out, n_in);
            let sliding = cfg.is_sliding(l);
            let hd = cfg.layer_head_dim(l);
            let n_kv = cfg.layer_n_kv(l);
            let own_kv = !cfg.is_kv_shared(l);
            let k_eq_v = cfg.layer_k_eq_v(l);
            let inter = cfg.layer_intermediate(l);
            let (k, v, k_norm) = if own_kv {
                (
                    Some(gl("self_attn.k_proj.weight", n_kv * hd, h)?),
                    if k_eq_v {
                        None
                    } else {
                        Some(gl("self_attn.v_proj.weight", n_kv * hd, h)?)
                    },
                    Some(nw(&format!("{lp}self_attn.k_norm.weight"), &[hd])?),
                )
            } else {
                (None, None, None)
            };
            let ple_l = if cfg.has_ple() {
                Some(PleLayer {
                    gate: gl("per_layer_input_gate.weight", p, h)?,
                    proj: gl("per_layer_projection.weight", h, p)?,
                    norm: g("post_per_layer_input_norm.weight", &[h])?,
                })
            } else {
                None
            };
            let layer_scalar = match g("layer_scalar", &[1]) {
                Ok(s) => bf16_to_f32(s[0]),
                Err(_) => 1.0,
            };
            layers.push(Layer {
                sliding,
                hd,
                n_kv,
                kv_src: cfg.kv_source(l),
                own_kv,
                k_eq_v,
                inter,
                q: gl("self_attn.q_proj.weight", nh * hd, h)?,
                k,
                v,
                o: gl("self_attn.o_proj.weight", h, nh * hd)?,
                q_norm: nw(&format!("{lp}self_attn.q_norm.weight"), &[hd])?,
                k_norm,
                ln_in: nw(&format!("{lp}input_layernorm.weight"), &[h])?,
                ln_post_attn: nw(&format!("{lp}post_attention_layernorm.weight"), &[h])?,
                ln_pre_ffn: nw(&format!("{lp}pre_feedforward_layernorm.weight"), &[h])?,
                ln_post_ffn: nw(&format!("{lp}post_feedforward_layernorm.weight"), &[h])?,
                gate: gl("mlp.gate_proj.weight", inter, h)?,
                up: gl("mlp.up_proj.weight", inter, h)?,
                down: gl("mlp.down_proj.weight", h, inter)?,
                ple: ple_l,
                layer_scalar,
            });
        }

        let rs = cfg.rope_spec(true);
        let rf = cfg.rope_spec(false);
        let scaled = |mut f: Vec<f32>, factor: f64| {
            f.iter_mut().for_each(|v| *v /= factor as f32);
            f
        };
        let inv_freq_sliding = scaled(rope_inv_freq(cfg.head_dim, rs.theta, rs.proportion), rs.factor);
        let inv_freq_full = scaled(rope_inv_freq(cfg.global_head_dim, rf.theta, rf.proportion), rf.factor);

        Ok(Self {
            cfg,
            quant,
            embed,
            head,
            ple,
            layers,
            final_norm,
            inv_freq_sliding,
            inv_freq_full,
        })
    }

    /// Bytes of linear weights streamed per decoded token.
    pub fn streamed_bytes(&self) -> usize {
        let mut n = self.head.bytes();
        if let Some(p) = &self.ple {
            n += p.proj.bytes();
        }
        for l in &self.layers {
            n += l.q.bytes() + l.o.bytes() + l.gate.bytes() + l.up.bytes() + l.down.bytes();
            n += l.k.as_ref().map_or(0, |x| x.bytes()) + l.v.as_ref().map_or(0, |x| x.bytes());
            if let Some(p) = &l.ple {
                n += p.gate.bytes() + p.proj.bytes();
            }
        }
        n
    }

    pub fn new_session(&self) -> Session {
        let caches = self
            .layers
            .iter()
            .map(|l| if l.own_kv { Some(LayerCache::new(l.n_kv, l.hd)) } else { None })
            .collect();
        Session { caches, pos: 0 }
    }

    /// The main model's input embedding of `tok`, scaled by sqrt(hidden) (as fed to the drafter).
    pub fn embed_scaled(&self, tok: u32) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let scale = (h as f32).sqrt();
        let mut e = vec![0f32; h];
        self.embed.row(tok as usize, &mut e);
        e.iter_mut().for_each(|v| *v *= scale);
        e
    }

    /// Softcapped logits `[t, vocab]` for final-normed hidden states `[t, hidden]`.
    pub fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let _timer = prof::start(Cat::Head);
        let mut logits = vec![0f32; t * cfg.vocab_size];
        linear(&self.head, cfg.vocab_size, cfg.hidden_size, hidden, t, &mut logits);
        if let Some(cap) = cfg.final_logit_softcapping {
            let cap = cap as f32;
            for_chunks(&mut logits, 8192, true, |_, c| {
                for v in c.iter_mut() {
                    *v = (*v / cap).tanh() * cap;
                }
            });
        }
        logits
    }

    /// Runs `tokens` through the model starting at `sess.pos`, appending to the KV
    /// cache, and returns the (softcapped) logits for the last token.
    pub fn forward(&self, sess: &mut Session, tokens: &[u32]) -> Vec<f32> {
        let t = tokens.len();
        let h = self.cfg.hidden_size;
        let hs = self.forward_hidden(sess, tokens);
        self.logits(&hs[(t - 1) * h..], 1)
    }

    /// Runs `tokens` through the decoder and returns the final-normed hidden states `[t, hidden]`.
    pub fn forward_hidden(&self, sess: &mut Session, tokens: &[u32]) -> Vec<f32> {
        self.forward_hidden_mm(sess, tokens, &[])
    }

    /// Like `forward_hidden`, but positions listed in `mm` (relative to this call, sorted) take
    /// the given embedding (a multimodal soft token in text space) instead of the token
    /// embedding; their per-layer-embedding lookup uses the pad token, as in the reference.
    pub fn forward_hidden_mm(&self, sess: &mut Session, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        let t = tokens.len();
        assert!(t > 0);
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let nl = cfg.num_hidden_layers;
        let start = sess.pos;
        let eps = cfg.rms_norm_eps as f32;

        let par = t > 1;
        let timer = prof::start(Cat::Embed);
        // Token embeddings, scaled by sqrt(hidden).
        let mut x = vec![0f32; t * h];
        let scale = (h as f32).sqrt();
        for_chunks(&mut x, h, par, |ti, row| {
            self.embed.row(tokens[ti] as usize, row);
            row.iter_mut().for_each(|v| *v *= scale);
        });
        // Multimodal soft tokens replace the placeholder embeddings; PLE uses the pad token there.
        let mut ple_tokens: Vec<u32> = tokens.to_vec();
        for &(i, emb) in mm {
            debug_assert!(i < t && emb.len() == h);
            x[i * h..(i + 1) * h].copy_from_slice(emb);
            ple_tokens[i] = 0;
        }
        let tokens = &ple_tokens[..];

        // Per-layer embeddings: [t, nl, p]
        let pli: Option<Vec<f32>> = self.ple.as_ref().map(|ple| {
            let p = cfg.hidden_size_per_layer_input;
            let mut proj = vec![0f32; t * nl * p];
            linear(&ple.proj, nl * p, h, &x, t, &mut proj);
            let s_proj = (h as f32).powf(-0.5);
            let s_emb = (p as f32).sqrt();
            let s_out = std::f32::consts::FRAC_1_SQRT_2;
            for_chunks(&mut proj, nl * p, par, |ti, row| {
                let mut emb = vec![0f32; nl * p];
                ple.embed.row(tokens[ti] as usize, &mut emb);
                for l in 0..nl {
                    let chunk = &mut row[l * p..(l + 1) * p];
                    scale_inplace(chunk, s_proj);
                    rmsnorm_inplace(chunk, Some(ple.norm), eps);
                    for (j, c) in chunk.iter_mut().enumerate() {
                        *c = (*c + emb[l * p + j] * s_emb) * s_out;
                    }
                }
            });
            proj
        });

        let positions: Vec<usize> = (start..start + t).collect();
        let rope_s = rope_tables(&positions, &self.inv_freq_sliding);
        let rope_f = rope_tables(&positions, &self.inv_freq_full);
        drop(timer);

        for (l, layer) in self.layers.iter().enumerate() {
            let rope = if layer.sliding { &rope_s } else { &rope_f };
            self.layer_forward(l, layer, &mut x, t, start, pli.as_deref(), sess, rope);
        }

        // Final norm.
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.final_norm, eps);
        sess.pos += t;
        x
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self,
        l: usize,
        layer: &Layer,
        x: &mut [f32],
        t: usize,
        start: usize,
        pli: Option<&[f32]>,
        sess: &mut Session,
        rope: &(Vec<f32>, Vec<f32>),
    ) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let hd = layer.hd;
        let n_kv = layer.n_kv;
        let eps = cfg.rms_norm_eps as f32;
        let half = hd / 2;
        let (cos, sin) = rope;
        let par = t > 1;

        // ---- Attention ----
        let mut hn = vec![0f32; t * h];
        rmsnorm_f32w_rows(x, h, &layer.ln_in, eps, &mut hn);

        // Q (and K, V when this layer owns its cache) share the input: one fused parallel region.
        let mut q = vec![0f32; t * nh * hd];
        let mut k = vec![0f32; if layer.own_kv { t * n_kv * hd } else { 0 }];
        let mut v = vec![0f32; if layer.own_kv && !layer.k_eq_v { t * n_kv * hd } else { 0 }];
        {
            let mut jobs: Vec<(&Lin, usize, &mut [f32])> = vec![(&layer.q, nh * hd, &mut q[..])];
            if layer.own_kv {
                jobs.push((layer.k.as_ref().unwrap(), n_kv * hd, &mut k[..]));
                if !layer.k_eq_v {
                    jobs.push((layer.v.as_ref().unwrap(), n_kv * hd, &mut v[..]));
                }
            }
            linear_multi(&mut jobs, h, &hn, t);
        }
        let qscale = cfg.query_scale();
        for_chunks(&mut q, hd, par, |i, qh| {
            let ti = i / nh;
            rmsnorm_f32w_inplace(qh, &layer.q_norm, eps);
            apply_rope(qh, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
            if qscale != 1.0 {
                scale_inplace(qh, qscale);
            }
        });

        if layer.own_kv {
            if layer.k_eq_v {
                v = k.clone();
            }
            if !cfg.gemma3 {
                for_chunks(&mut v, hd, par, |_, vh| rmsnorm_inplace(vh, None, eps));
            }
            let k_norm = layer.k_norm.as_deref().unwrap();
            for_chunks(&mut k, hd, par, |i, kh| {
                let ti = i / n_kv;
                rmsnorm_f32w_inplace(kh, k_norm, eps);
                apply_rope(kh, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
            });
            sess.caches[l].as_mut().unwrap().push(&k, &v, t);
        }

        let cache = sess.caches[layer.kv_src].as_ref().expect("kv source cache");
        let window = if layer.sliding { Some(cfg.sliding_window) } else { None };
        let mut att = vec![0f32; t * nh * hd];
        attention_masked(&q, cache, nh, t, start, window, !cfg.bidirectional(), &mut att);

        let mut o = vec![0f32; t * h];
        linear(&layer.o, h, nh * hd, &att, t, &mut o);
        rmsnorm_f32w_rows_inplace(&mut o, h, &layer.ln_post_attn, eps);
        add_inplace(x, &o);

        // ---- MLP ----
        rmsnorm_f32w_rows(x, h, &layer.ln_pre_ffn, eps, &mut hn);
        let inter = layer.inter;
        let mut g = vec![0f32; t * inter];
        let mut u = vec![0f32; t * inter];
        linear_multi(&mut [(&layer.gate, inter, &mut g[..]), (&layer.up, inter, &mut u[..])], h, &hn, t);
        gelu_mul(&mut g, &u);
        let mut d = o; // reuse buffer [t, h]
        linear(&layer.down, h, inter, &g, t, &mut d);
        rmsnorm_f32w_rows_inplace(&mut d, h, &layer.ln_post_ffn, eps);
        add_inplace(x, &d);

        // ---- Per-layer embedding injection ----
        if let (Some(pw), Some(pli)) = (&layer.ple, pli) {
            let p = cfg.hidden_size_per_layer_input;
            let nl = cfg.num_hidden_layers;
            let mut gp = vec![0f32; t * p];
            linear(&pw.gate, p, h, x, t, &mut gp);
            for_chunks(&mut gp, p, par, |ti, row| {
                let inp = &pli[(ti * nl + l) * p..(ti * nl + l + 1) * p];
                for (r, &i) in row.iter_mut().zip(inp) {
                    *r = gelu_tanh(*r) * i;
                }
            });
            let mut pr = d;
            linear(&pw.proj, h, p, &gp, t, &mut pr);
            rmsnorm_rows_inplace(&mut pr, h, Some(pw.norm), eps);
            add_inplace(x, &pr);
        }

        if layer.layer_scalar != 1.0 {
            let s = layer.layer_scalar;
            for_chunks(x, 4096, par, |_, c| scale_inplace(c, s));
        }
    }
}

/// Causal (optionally sliding-window) GQA attention over the cache.
/// `q`: [t, nh, hd] already normed and rotated. `out`: [t, nh, hd].
/// Attention scaling is 1.0 (Gemma 4 relies on QK-norm).
///
/// Work is split per (token, kv head) so the `rep` query heads sharing a KV head read
/// each key/value once. When that gives too few tasks (short decode), it splits per head.
pub(crate) fn attention(q: &[f32], cache: &LayerCache, nh: usize, t: usize, start: usize, window: Option<usize>, out: &mut [f32]) {
    attention_masked(q, cache, nh, t, start, window, true, out)
}

/// `attention` with the causal mask optional: bidirectional (encoders, embedding models) lets every
/// position see the whole cache, or, with a window, the keys less than `window` positions away.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_masked(
    q: &[f32],
    cache: &LayerCache,
    nh: usize,
    t: usize,
    start: usize,
    window: Option<usize>,
    causal: bool,
    out: &mut [f32],
) {
    let _timer = prof::start(Cat::Attention);
    let hd = cache.hd;
    let n_kv = cache.n_kv;
    let rep = nh / n_kv;
    let threads = crate::pool::threads();
    let par = t > 1 || cache.len > 32;
    let group = if t * n_kv >= threads { rep } else { 1 };
    let groups_per_kv = rep / group;
    let ntasks = t * n_kv * groups_per_kv;
    let op = SendPtr(out.as_mut_ptr());
    let run = |task: usize| {
        let ti = task / (n_kv * groups_per_kv);
        let rem = task % (n_kv * groups_per_kv);
        let kvh = rem / groups_per_kv;
        let h0 = kvh * rep + (rem % groups_per_kv) * group;
        let qpos = start + ti;
        let lo = window.map(|w| (qpos + 1).saturating_sub(w)).unwrap_or(0);
        let hi = match (causal, window) {
            (true, _) => qpos + 1,
            (false, Some(w)) => (qpos + w).min(cache.len),
            (false, None) => cache.len,
        };
        debug_assert!(hi <= cache.len);
        let n = hi - lo;
        let qbase = (ti * nh + h0) * hd;
        let mut scores = vec![0f32; group * n];
        for j in 0..n {
            let k = cache.k_at(lo + j, kvh);
            for h in 0..group {
                scores[h * n + j] = dot(&q[qbase + h * hd..qbase + (h + 1) * hd], k);
            }
        }
        for h in 0..group {
            softmax_inplace(&mut scores[h * n..(h + 1) * n]);
        }
        // SAFETY: tasks write disjoint [group*hd] ranges of `out`.
        let o = unsafe { std::slice::from_raw_parts_mut(op.get().add(qbase), group * hd) };
        o.fill(0.0);
        for j in 0..n {
            let v = cache.v_at(lo + j, kvh);
            for h in 0..group {
                axpy(&mut o[h * hd..(h + 1) * hd], scores[h * n + j], v);
            }
        }
    };
    if par {
        crate::pool::run(ntasks, &run);
    } else {
        for i in 0..ntasks {
            run(i);
        }
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}
