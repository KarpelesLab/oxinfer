//! Gemma 4 text decoder: embeddings, per-layer embeddings (PLE), sliding/global
//! attention with KV sharing, gated-GELU MLP (plus, in 26B-A4B, a router and 128 routed
//! GELU experts beside it), tied softcapped LM head.

use crate::config::TextConfig;
use crate::ggml::{GgmlType, GgufMatrix};
use crate::kernels::*;
use crate::prof::{self, Cat};
use crate::quant::{Q4Matrix, QKind, QMatrix, quantize_bf16, quantize_q4};
use crate::safetensors::{ModelFiles, Stored};
use anyhow::{Context, Result, bail};

#[path = "model_gpu.rs"]
mod gpu_dec;

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
    // MLX 4-bit: repacked as affine Q4 per expert whatever `q` asks for (other widths dequantize
    // to bf16 below).
    if let Some(v) = files.mlx_q4_experts(full, n_exp, n_out, n_in)? {
        return Ok(v.into_iter().map(Lin::q4).collect());
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
    /// A runtime LoRA update added to the output while its adapter is switched on.
    lora: Option<Box<crate::lora::LoraDelta>>,
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
            lora: None,
        }
    }
    /// bf16 weights converted or reordered at load time (e.g. from fp16 checkpoints).
    pub fn bf16_owned(w: crate::quant::PageVec<u16>) -> Self {
        Self {
            w: Weight::Bf16(w.into(), std::marker::PhantomData),
            dev: std::sync::OnceLock::new(),
            transient: std::sync::atomic::AtomicBool::new(false),
            lora: None,
        }
    }
    pub fn q8(m: QMatrix) -> Self {
        Self {
            w: Weight::Q8(m),
            dev: std::sync::OnceLock::new(),
            transient: std::sync::atomic::AtomicBool::new(false),
            lora: None,
        }
    }
    pub fn q4(m: Q4Matrix) -> Self {
        Self {
            w: Weight::Q4(m),
            dev: std::sync::OnceLock::new(),
            transient: std::sync::atomic::AtomicBool::new(false),
            lora: None,
        }
    }
    /// A GGUF weight (`GgufMatrix::borrowed` keeps it zero-copy over a mapped file).
    pub fn gguf(m: GgufMatrix) -> Self {
        Self {
            w: Weight::Gguf(m),
            dev: std::sync::OnceLock::new(),
            transient: std::sync::atomic::AtomicBool::new(false),
            lora: None,
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
    /// Attaches (or with None removes) a runtime LoRA update (see `crate::lora`).
    pub fn set_lora(&mut self, d: Option<Box<crate::lora::LoraDelta>>) {
        self.lora = d;
    }
    pub fn has_lora(&self) -> bool {
        self.lora.is_some()
    }
    /// The LoRA update when its adapter is switched on.
    #[inline]
    fn active_lora(&self) -> Option<&crate::lora::LoraDelta> {
        self.lora.as_deref().filter(|d| d.is_active())
    }
}

/// Fused matmuls over one shared input (any mix of storage formats), plus the update of any
/// active runtime LoRA (on the CPU, after the base matmul).
pub(crate) fn linear_multi(jobs: &mut [(&Lin, usize, &mut [f32])], n_in: usize, x: &[f32], t: usize) {
    if jobs.is_empty() {
        return;
    }
    linear_multi_base(jobs, n_in, x, t);
    for (l, _, y) in jobs.iter_mut() {
        if let Some(d) = l.active_lora() {
            d.apply(x, t, y);
        }
    }
}

fn linear_multi_base(jobs: &mut [(&Lin, usize, &mut [f32])], n_in: usize, x: &[f32], t: usize) {
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
    gate: Lin<'a>,  // [P, H]
    proj: Lin<'a>,  // [H, P]
    norm: Vec<f32>, // [H]
}

struct PleGlobal<'a> {
    embed: Table<'a>, // [V, L*P]
    proj: Lin<'a>,    // [L*P, H]
    norm: Vec<f32>,   // [P]
}

/// The MoE block of a Gemma 4 26B-A4B layer, run beside the dense MLP on the same residual:
/// the router reads the residual RMS-normed without weight, times `scale / sqrt(hidden)` (folded
/// into `router`), takes the softmax top k renormalized, times `per_expert_scale`; the experts
/// read the residual normed by `ln_pre`, and their sum, normed by `ln_post`, is added to the
/// dense MLP output normed by `ln_post_dense` (then both go through `post_feedforward_layernorm`).
pub(crate) struct GemmaMoe<'a> {
    pub(crate) router: Vec<f32>, // [n_exp, H]
    pub(crate) per_expert_scale: Vec<f32>,
    pub(crate) ln_post_dense: Vec<f32>,
    pub(crate) ln_pre: Vec<f32>,
    pub(crate) ln_post: Vec<f32>,
    pub(crate) experts: crate::moe::Experts<'a>,
}

impl GemmaMoe<'_> {
    /// The routes (expert, weight) of each of the `t` rows of the residual `x`.
    fn route(&self, x: &[f32], t: usize, h: usize, eps: f32) -> Vec<Vec<(usize, f32)>> {
        let ne = self.experts.n_experts();
        let k = self.experts.top_k;
        let mut logits = vec![0f32; t * ne];
        for_chunks(&mut logits, ne, t > 1, |ti, l| {
            let mut xn = x[ti * h..(ti + 1) * h].to_vec();
            rmsnorm_inplace(&mut xn, None, eps);
            for (v, w) in l.iter_mut().zip(self.router.chunks(h)) {
                *v = dot(w, &xn);
            }
        });
        logits
            .chunks(ne)
            .map(|l| {
                let mut r = crate::moe::route_top_k(l, k);
                for (e, w) in r.iter_mut() {
                    *w *= self.per_expert_scale[*e];
                }
                r
            })
            .collect()
    }
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
    moe: Option<GemmaMoe<'a>>,
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
    /// Weight streaming: unit 0 is the PLE projection when there is one, then the layers, then
    /// the LM head (see [`Model::stream_units`]).
    stream: Option<crate::stream::StreamHook>,
    /// Single-token decoding on the GPU (see `model_gpu.rs`), compiled on first use.
    gpu_plan: std::sync::OnceLock<Option<Box<dyn crate::gpu::decode::DecodePlan>>>,
}

#[derive(Clone)]
pub struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
    pub n_kv: usize,
    pub hd: usize,
    pub len: usize,
    /// Positions unchanged since a device copy of the cache last synced (truncations lower it;
    /// see `crate::gpu::decode`).
    pub(crate) kept: usize,
}

impl LayerCache {
    pub(crate) fn new(n_kv: usize, hd: usize) -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            n_kv,
            hd,
            len: 0,
            kept: 0,
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
    /// All cached keys and values (`[len, n_kv, hd]` each).
    pub fn data(&self) -> (&[f32], &[f32]) {
        (&self.k, &self.v)
    }
    pub(crate) fn truncate(&mut self, len: usize) {
        self.kept = self.kept.min(len);
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
    /// Device state of GPU decoding (KV cache copies).
    gpu: crate::gpu::decode::Slot,
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
                norm: nw("per_layer_projection_norm.weight", &[p])?,
            })
        } else {
            None
        };

        let mut layers = Vec::with_capacity(nl);
        for l in 0..nl {
            let lp = format!("layers.{l}.");
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
                    norm: nw(&format!("{lp}post_per_layer_input_norm.weight"), &[h])?,
                })
            } else {
                None
            };
            let ls = format!("{prefix}{lp}layer_scalar");
            let layer_scalar = if files.has(&ls) {
                files.f32(&ls)?.first().copied().unwrap_or(1.0)
            } else {
                1.0
            };
            let moe = match cfg.moe() {
                Some((n_exp, top_k, mi)) => {
                    let proj = nw(&format!("{lp}router.proj.weight"), &[n_exp, h])?;
                    let scale = nw(&format!("{lp}router.scale"), &[h])?;
                    let c = (h as f32).powf(-0.5);
                    let mut router = proj;
                    for row in router.chunks_mut(h) {
                        for (w, s) in row.iter_mut().zip(&scale) {
                            *w *= s * c;
                        }
                    }
                    Some(GemmaMoe {
                        router,
                        per_expert_scale: nw(&format!("{lp}router.per_expert_scale"), &[n_exp])?,
                        ln_post_dense: nw(&format!("{lp}post_feedforward_layernorm_1.weight"), &[h])?,
                        ln_pre: nw(&format!("{lp}pre_feedforward_layernorm_2.weight"), &[h])?,
                        ln_post: nw(&format!("{lp}post_feedforward_layernorm_2.weight"), &[h])?,
                        experts: crate::moe::Experts::load(
                            files,
                            &format!("{prefix}{lp}experts."),
                            n_exp,
                            top_k,
                            mi,
                            h,
                            quant,
                            crate::moe::Act::GeluTanh,
                        )?,
                    })
                }
                None => None,
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
                moe,
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
        let mut inv_freq_full = scaled(rope_inv_freq(cfg.global_head_dim, rf.theta, rf.proportion), rf.factor);
        if !cfg.gemma3 && files.has("rope_freqs") {
            // GGUF: llama.cpp rotates the whole global head, dividing each frequency by
            // `rope_freqs` (proportional RoPE: 1 for the rotated pairs, huge for the others).
            let ff = files.f32("rope_freqs")?;
            let base = rope_inv_freq(cfg.global_head_dim, rf.theta, 1.0);
            if ff.len() != base.len() {
                bail!("rope_freqs has {} entries, expected {}", ff.len(), base.len());
            }
            inv_freq_full = base.iter().zip(&ff).map(|(b, f)| if *f > 1e6 { 0.0 } else { b / f }).collect();
        }

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
            stream: None,
            gpu_plan: std::sync::OnceLock::new(),
        })
    }

    /// Attaches the modules of a runtime LoRA adapter (see `crate::lora`) to the attention and
    /// MLP projections and the LM head, by Hugging Face module name. Returns the number patched.
    pub fn attach_lora(&mut self, ad: &mut crate::lora::LoraAdapter) -> Result<usize> {
        let (h, nh) = (self.cfg.hidden_size, self.cfg.num_attention_heads);
        let mut set = |lin: &mut Lin<'a>, name: String, n_out: usize, n_in: usize| -> Result<()> {
            if let Some(d) = ad.take(&name, n_out, n_in, None, None)? {
                lin.set_lora(Some(d));
            }
            Ok(())
        };
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let p = format!("layers.{l}.");
            let (hd, kv, inter) = (layer.hd, layer.n_kv * layer.hd, layer.inter);
            set(&mut layer.q, format!("{p}self_attn.q_proj"), nh * hd, h)?;
            if let Some(k) = &mut layer.k {
                set(k, format!("{p}self_attn.k_proj"), kv, h)?;
            }
            if let Some(v) = &mut layer.v {
                set(v, format!("{p}self_attn.v_proj"), kv, h)?;
            }
            set(&mut layer.o, format!("{p}self_attn.o_proj"), h, nh * hd)?;
            set(&mut layer.gate, format!("{p}mlp.gate_proj"), inter, h)?;
            set(&mut layer.up, format!("{p}mlp.up_proj"), inter, h)?;
            set(&mut layer.down, format!("{p}mlp.down_proj"), h, inter)?;
        }
        set(&mut self.head, "lm_head".into(), self.cfg.vocab_size, h)?;
        ad.finish()
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
            if let Some(m) = &l.moe {
                n += m.experts.bytes_per_token();
            }
        }
        n
    }

    /// The weights of each streaming unit (all experts included), in execution order.
    fn unit_lins(&self) -> Vec<Vec<&Lin<'a>>> {
        let mut v: Vec<Vec<&Lin>> = Vec::new();
        if let Some(p) = &self.ple {
            v.push(vec![&p.proj]);
        }
        for l in &self.layers {
            v.push(l.dense_lins());
        }
        v.push(vec![&self.head]);
        v
    }

    /// The streaming units of this model in execution order (see `crate::stream`): the PLE
    /// projection (E2B / E4B), the layers, the LM head. The routed experts of MoE layers are
    /// on-demand parts of their layer.
    pub fn stream_units(&self) -> Vec<crate::stream::UnitSpec> {
        let base = self.ple.is_some() as usize;
        self.unit_lins()
            .into_iter()
            .enumerate()
            .map(|(i, lins)| {
                let layer = i.checked_sub(base).and_then(|l| self.layers.get(l));
                crate::stream::UnitSpec {
                    name: match layer {
                        Some(_) => format!("layer.{}", i - base),
                        None if i == 0 && base == 1 => "ple".into(),
                        None => "lm_head".into(),
                    },
                    ranges: lins.iter().flat_map(|l| l.mem_ranges()).collect(),
                    uses: 1.0,
                    parts: layer.and_then(|l| l.moe.as_ref()).map(|m| m.experts.ranges()).unwrap_or_default(),
                }
            })
            .collect()
    }

    /// Marks the weights of the units with `streamed[i]` set as streamed: no accelerator state
    /// is kept for them between calls (see `Lin::set_transient_device`). Routed experts are read
    /// on demand whatever their layer, so they never keep device state.
    pub fn mark_streamed(&self, streamed: &[bool]) {
        for (lins, &s) in self.unit_lins().iter().zip(streamed) {
            for l in lins {
                l.set_transient_device(s);
            }
        }
        for m in self.layers.iter().filter_map(|l| l.moe.as_ref()) {
            for l in m.experts.lins().into_iter().flatten() {
                l.set_transient_device(true);
            }
        }
    }

    /// Brackets the PLE projection, every layer and the LM head with `hook`'s units (None stops
    /// streaming).
    pub fn set_stream(&mut self, hook: Option<crate::stream::StreamHook>) {
        self.stream = hook;
    }

    /// The unit index of layer `l` (see [`Model::stream_units`]).
    fn layer_unit(&self, l: usize) -> usize {
        l + self.ple.is_some() as usize
    }

    pub fn new_session(&self) -> Session {
        let caches = self
            .layers
            .iter()
            .map(|l| if l.own_kv { Some(LayerCache::new(l.n_kv, l.hd)) } else { None })
            .collect();
        Session {
            caches,
            pos: 0,
            gpu: Default::default(),
        }
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
        let unit = crate::stream::enter(&self.stream, self.layer_unit(self.layers.len()));
        linear(&self.head, cfg.vocab_size, cfg.hidden_size, hidden, t, &mut logits);
        drop(unit);
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
        if t == 1 {
            // OXINFER_GPU_DECODE_CHECK=1: compare every GPU step with the CPU path.
            let before = crate::gpu::decode::check().then(|| sess.clone());
            if let Some((_, Some(l))) = self.gpu_step(sess, tokens[0], true) {
                if let Some(mut s2) = before {
                    let hs = self.forward_hidden_cpu(&mut s2, tokens, &[]);
                    crate::gpu::decode::report(s2.pos - 1, &self.logits(&hs, 1), &l);
                }
                return l;
            }
        }
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
        if tokens.len() == 1
            && mm.is_empty()
            && let Some((hs, _)) = self.gpu_step(sess, tokens[0], false)
        {
            return hs;
        }
        self.forward_hidden_cpu(sess, tokens, mm)
    }

    fn forward_hidden_cpu(&self, sess: &mut Session, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        let t = tokens.len();
        assert!(t > 0);
        if t > 1 {
            let _ = self.gpu_plan();
        }
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let nl = cfg.num_hidden_layers;
        let start = sess.pos;
        let eps = cfg.rms_norm_eps as f32;

        let par = t > 1;
        let timer = prof::start(Cat::Embed);
        if let Some(s) = &self.stream {
            // All rows at once, ahead of the read-ahead, rather than a page fault per lookup.
            let mut rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| self.embed.row_ranges(tk as usize, h)).collect();
            if let Some(ple) = &self.ple {
                let w = nl * cfg.hidden_size_per_layer_input;
                rows.extend(tokens.iter().flat_map(|&tk| ple.embed.row_ranges(tk as usize, w)));
            }
            s.need_ranges(&rows);
        }
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
            let unit = crate::stream::enter(&self.stream, 0);
            linear(&ple.proj, nl * p, h, &x, t, &mut proj);
            drop(unit);
            let s_proj = (h as f32).powf(-0.5);
            let s_emb = (p as f32).sqrt();
            let s_out = std::f32::consts::FRAC_1_SQRT_2;
            for_chunks(&mut proj, nl * p, par, |ti, row| {
                let mut emb = vec![0f32; nl * p];
                ple.embed.row(tokens[ti] as usize, &mut emb);
                for l in 0..nl {
                    let chunk = &mut row[l * p..(l + 1) * p];
                    scale_inplace(chunk, s_proj);
                    rmsnorm_f32w_inplace(chunk, &ple.norm, eps);
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
        // `vision_bidirectional`: each run of consecutive soft tokens (one image) is a block whose
        // tokens see the whole block; the others stay causal.
        let ext: Option<Vec<usize>> = (cfg.vision_bidirectional() && !mm.is_empty()).then(|| {
            let mut e: Vec<usize> = (0..t).map(|i| start + i + 1).collect();
            let mut pos: Vec<usize> = mm.iter().map(|m| m.0).collect();
            pos.sort_unstable();
            let mut i = 0;
            while i < pos.len() {
                let mut j = i;
                while j + 1 < pos.len() && pos[j + 1] == pos[j] + 1 {
                    j += 1;
                }
                for &p in &pos[i..=j] {
                    e[p] = start + pos[j] + 1;
                }
                i = j + 1;
            }
            e
        });
        drop(timer);

        for (l, layer) in self.layers.iter().enumerate() {
            let _unit = crate::stream::enter(&self.stream, self.layer_unit(l));
            let rope = if layer.sliding { &rope_s } else { &rope_f };
            self.layer_forward(l, layer, &mut x, t, start, pli.as_deref(), sess, rope, ext.as_deref());
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
        ext: Option<&[usize]>,
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
        // Image soft tokens see their whole image on sliding layers (`vision_bidirectional`).
        let ext = if layer.sliding { ext } else { None };
        if !cfg.bidirectional() && t >= PREFILL_ATTN_MIN {
            attention_prefill(&q, cache, nh, t, start, window, ext, &mut att);
        } else {
            attention_ext(&q, cache, nh, t, start, window, !cfg.bidirectional(), ext, &mut att);
        }

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
        if let Some(m) = &layer.moe {
            // The routed experts, beside the dense MLP, on the same residual.
            rmsnorm_f32w_rows_inplace(&mut d, h, &m.ln_post_dense, eps);
            let routes = m.route(x, t, h, eps);
            rmsnorm_f32w_rows(x, h, &m.ln_pre, eps, &mut hn);
            let mut e = vec![0f32; t * h];
            let unit = self.layer_unit(l);
            let need = |ids: &[usize]| {
                if let Some(s) = &self.stream {
                    s.need_parts(unit, ids)
                }
            };
            m.experts
                .forward(&hn, t, h, &routes, &mut e, self.stream.as_ref().map(|_| &need as &dyn Fn(&[usize])));
            rmsnorm_f32w_rows_inplace(&mut e, h, &m.ln_post, eps);
            add_inplace(&mut d, &e);
        }
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
            rmsnorm_f32w_rows_inplace(&mut pr, h, &pw.norm, eps);
            add_inplace(x, &pr);
        }

        if layer.layer_scalar != 1.0 {
            let s = layer.layer_scalar;
            for_chunks(x, 4096, par, |_, c| scale_inplace(c, s));
        }
    }
}

impl<'a> Layer<'a> {
    /// The weights every pass reads (all but the routed experts).
    fn dense_lins(&self) -> Vec<&Lin<'a>> {
        let mut v = vec![&self.q, &self.o, &self.gate, &self.up, &self.down];
        v.extend(self.k.iter().chain(self.v.iter()));
        if let Some(p) = &self.ple {
            v.extend([&p.gate, &p.proj]);
        }
        v
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
    attention_ext(q, cache, nh, t, start, window, causal, None, out)
}

/// `attention_masked` where query `i` also sees the keys before `ext[i]` (absolute positions; a
/// later bound than causal lets the soft tokens of one image see each other both ways).
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_ext(
    q: &[f32],
    cache: &LayerCache,
    nh: usize,
    t: usize,
    start: usize,
    window: Option<usize>,
    causal: bool,
    ext: Option<&[usize]>,
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
            (true, _) => ext.map_or(qpos + 1, |e| e[ti].max(qpos + 1)),
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

/// Chunks of at least this many tokens take [`attention_prefill`].
const PREFILL_ATTN_MIN: usize = 32;

/// Causal (optionally sliding-window, `ext`-extended: see [`attention_ext`]) attention for a
/// prefill chunk as GEMMs: per block of queries and KV head, the scores of the group's query
/// heads against every key one of them sees, masked per row, soft-maxed, times V. A chunk that
/// starts the cache with a plain causal mask goes to the accelerator when there is one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_prefill(
    q: &[f32],
    cache: &LayerCache,
    nh: usize,
    t: usize,
    start: usize,
    window: Option<usize>,
    ext: Option<&[usize]>,
    out: &mut [f32],
) {
    let _timer = prof::start(Cat::Attention);
    let (hd, n_kv) = (cache.hd, cache.n_kv);
    let rep = nh / n_kv;
    let lo_of = |ti: usize| window.map_or(0, |w| (start + ti + 1).saturating_sub(w));
    let hi_of = |ti: usize| ext.map_or(start + ti + 1, |e| e[ti].max(start + ti + 1));
    if ext.is_none()
        && start == 0
        && window.is_none_or(|w| t <= w)
        && let Some(be) = crate::gpu::active()
        && 4.0 * (t * t * nh * hd) as f64 >= 1e8
    {
        // The device's attention scales by 1/sqrt(hd) (Gemma's is 1) and has one KV head per
        // query head.
        let s = (hd as f32).sqrt();
        let qs: Vec<f32> = q.iter().map(|x| x * s).collect();
        let mut k = vec![0f32; t * nh * hd];
        let mut v = vec![0f32; t * nh * hd];
        for_chunks(&mut k, nh * hd, true, |j, r| {
            for (h, o) in r.chunks_mut(hd).enumerate() {
                o.copy_from_slice(cache.k_at(j, h / rep));
            }
        });
        for_chunks(&mut v, nh * hd, true, |j, r| {
            for (h, o) in r.chunks_mut(hd).enumerate() {
                o.copy_from_slice(cache.v_at(j, h / rep));
            }
        });
        if be.attention(&qs, &k, &v, t, t, nh, hd, true, out) {
            return;
        }
    }
    let dhp = hd.next_multiple_of(32);
    // Query blocks keep the score matrix to a few tens of MB.
    let max_keys = (0..t).map(hi_of).max().unwrap_or(0) - lo_of(0);
    let tb = ((1usize << 23) / (rep * max_keys.max(1))).clamp(8, 256);
    let mut b0 = 0;
    while b0 < t {
        let b1 = (b0 + tb).min(t);
        let k0 = lo_of(b0);
        let k1 = (b0..b1).map(hi_of).max().unwrap();
        let (n, np) = (k1 - k0, (k1 - k0).next_multiple_of(32));
        let rows = (b1 - b0) * rep;
        for g in 0..n_kv {
            let mut qh = vec![0f32; rows * dhp];
            for_chunks(&mut qh, dhp, rows > 64, |r, row| {
                let (ti, h) = (b0 + r / rep, g * rep + r % rep);
                row[..hd].copy_from_slice(&q[(ti * nh + h) * hd..(ti * nh + h + 1) * hd]);
            });
            let mut kh = vec![0f32; n * dhp];
            for_chunks(&mut kh, dhp, n > 64, |j, row| row[..hd].copy_from_slice(cache.k_at(k0 + j, g)));
            let mut vt = vec![0f32; hd * np];
            for_chunks(&mut vt, np, true, |c, row| {
                for (j, x) in row[..n].iter_mut().enumerate() {
                    *x = cache.v_at(k0 + j, g)[c];
                }
            });
            let sc = crate::sd::ops::matmul_f32(&kh, n, dhp, &qh, rows);
            let mut p = vec![0f32; rows * np];
            for_chunks(&mut p, np, true, |r, row| {
                let ti = b0 + r / rep;
                let (lo, hi) = (lo_of(ti) - k0, hi_of(ti) - k0);
                row[lo..hi].copy_from_slice(&sc[r * n + lo..r * n + hi]);
                softmax_inplace(&mut row[lo..hi]);
            });
            let o = crate::sd::ops::matmul_f32(&vt, hd, np, &p, rows);
            let op = SendPtr(out.as_mut_ptr());
            crate::pool::run(rows, &|r| {
                let (ti, h) = (b0 + r / rep, g * rep + r % rep);
                // SAFETY: each row writes its own [hd] slice of `out`.
                let dst = unsafe { std::slice::from_raw_parts_mut(op.get().add((ti * nh + h) * hd), hd) };
                dst.copy_from_slice(&o[r * hd..(r + 1) * hd]);
            });
        }
        b0 = b1;
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
