//! Building blocks for the diffusion models. Activations are channels-last (`[pixels, channels]`,
//! pixel = `y * width + x`), so a convolution unfolds into the shared matmul (`model::linear`),
//! which brings the SIMD kernels, quantization and the GPU backends along. Weights come from
//! fp16 / bf16 / f32 safetensors and are held as bf16; biases and norms stay f32.

use crate::kernels::{QMat, for_chunks};
use crate::model::{Lin, linear};
use crate::pool;
use crate::quant::{PageVec, QAct};
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result, bail};
use std::path::Path;

// ---- profiling ---------------------------------------------------------------------------------

/// Wall time per operation kind, reported by `--profile` in image mode.
pub mod timing {
    use crate::clock::Instant;
    use std::sync::atomic::{AtomicU64, Ordering};
    pub const NAMES: [&str; 7] = [
        "conv unfold",
        "conv matmul",
        "linear",
        "attention",
        "group norm",
        "layer norm",
        "other",
    ];
    static NS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
    pub struct T(usize, Instant);
    pub fn start(k: usize) -> T {
        T(k, Instant::now())
    }
    impl Drop for T {
        fn drop(&mut self) {
            NS[self.0].fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
    pub fn report() -> String {
        let mut s = String::new();
        for (i, n) in NAMES.iter().enumerate() {
            let ms = NS[i].load(Ordering::Relaxed) as f64 / 1e6;
            if ms > 0.5 {
                s.push_str(&format!("  {n:12} {ms:9.1} ms\n"));
            }
        }
        s
    }
    pub const UNFOLD: usize = 0;
    pub const CONV: usize = 1;
    pub const LINEAR: usize = 2;
    pub const ATTN: usize = 3;
    pub const GN: usize = 4;
    pub const LN: usize = 5;
}

/// One safetensors file of a diffusers component (`unet/`, `vae/`, `text_encoder/`), with the
/// LoRA updates to merge into its weights as they are read.
pub struct Tensors {
    st: SafeTensors,
    lora: Option<super::lora::Patches>,
}

impl Tensors {
    pub fn open(path: &Path) -> Result<Self> {
        // Mapped where the platform can: every weight is converted on load anyway.
        Ok(Self {
            st: SafeTensors::open(path, cfg!(target_arch = "wasm32"))?,
            lora: None,
        })
    }

    /// Merges these LoRA updates into the weights read from now on.
    pub fn set_lora(&mut self, p: Option<super::lora::Patches>) {
        self.lora = p;
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.st.names()
    }

    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.st.metadata(key)
    }

    pub fn has(&self, name: &str) -> bool {
        self.st.info(name).is_some()
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.st.info(name).with_context(|| format!("missing tensor {name}"))?.shape.clone())
    }

    /// A tensor as f32, whatever its stored float type, with any LoRA updates merged in.
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.st.info(name).with_context(|| format!("missing tensor {name}"))?;
        let b = self.st.bytes_cow(name)?;
        let mut v: Vec<f32> = match info.dtype.as_str() {
            "F16" => b.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect(),
            "BF16" => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
                .collect(),
            "F32" => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
            other => bail!("tensor {name}: unsupported dtype {other}"),
        };
        if let Some(p) = &self.lora {
            p.apply(name, info.shape.first().copied().unwrap_or(1), &mut v)?;
        }
        Ok(v)
    }

    /// Raw fp16 bits when the tensor is stored as F16 (lookup tables keep them exact).
    pub fn f16_bits(&self, name: &str) -> Result<Option<Vec<u16>>> {
        let info = self.st.info(name).with_context(|| format!("missing tensor {name}"))?;
        if info.dtype != "F16" || self.lora.as_ref().is_some_and(|p| p.targets(name)) {
            return Ok(None);
        }
        Ok(Some(
            self.st
                .bytes_cow(name)?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect(),
        ))
    }
}

/// f32 -> bf16 with round-to-nearest-even.
#[inline]
pub fn f32_to_bf16(f: f32) -> u16 {
    let b = f.to_bits();
    if f.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn to_bf16_pages(v: &[f32]) -> PageVec<u16> {
    let mut out = PageVec::<u16>::zeroed(v.len());
    for (o, &x) in out.iter_mut().zip(v) {
        *o = f32_to_bf16(x);
    }
    out
}

/// A linear layer (or 1x1 convolution) with an optional bias.
pub struct Linear {
    w: Lin<'static>,
    b: Option<Vec<f32>>,
    pub n_out: usize,
    pub n_in: usize,
}

impl Linear {
    /// `{prefix}.weight` (`[out, in]` or `[out, in, 1, 1]`) and `{prefix}.bias` if present.
    pub fn load(t: &Tensors, prefix: &str) -> Result<Self> {
        let wn = format!("{prefix}.weight");
        let shape = t.shape(&wn)?;
        let (n_out, n_in) = match shape.as_slice() {
            [o, i] | [o, i, 1, 1] => (*o, *i),
            s => bail!("{wn}: expected a matrix or 1x1 kernel, got {s:?}"),
        };
        let w = t.f32(&wn)?;
        Self::from_parts(&w, n_out, n_in, Self::bias(t, prefix, n_out)?)
    }

    fn bias(t: &Tensors, prefix: &str, n_out: usize) -> Result<Option<Vec<f32>>> {
        let bn = format!("{prefix}.bias");
        if !t.has(&bn) {
            return Ok(None);
        }
        let b = t.f32(&bn)?;
        if b.len() != n_out {
            bail!("{bn}: {} values for {n_out} outputs", b.len());
        }
        Ok(Some(b))
    }

    fn from_parts(w: &[f32], n_out: usize, n_in: usize, b: Option<Vec<f32>>) -> Result<Self> {
        if w.len() != n_out * n_in {
            bail!("weight has {} values, expected {n_out}x{n_in}", w.len());
        }
        Ok(Self {
            w: Lin::bf16_owned(to_bf16_pages(w)),
            b,
            n_out,
            n_in,
        })
    }

    /// `x` is `[rows, n_in]`; returns `[rows, n_out]`.
    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let _t = timing::start(timing::LINEAR);
        self.forward_untimed(x, rows)
    }

    fn forward_untimed(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows * self.n_out];
        linear(&self.w, self.n_out, self.n_in, x, rows, &mut y);
        if let Some(b) = &self.b {
            for_chunks(&mut y, self.n_out, rows > 1, |_, r| {
                for (v, &bb) in r.iter_mut().zip(b) {
                    *v += bb;
                }
            });
        }
        y
    }
}

/// A k x k convolution (zero padding k/2 on every side unless loaded with
/// [`Conv::load_padded`]), held as a `[out, k*k*in]` matmul over unfolded pixels.
pub struct Conv {
    lin: Linear,
    pub k: usize,
    pub cin: usize,
    pub cout: usize,
    /// Zero padding before (top / left) and after (bottom / right) the input.
    pad: (usize, usize),
}

/// Upper bound on the unfold buffer of one convolution chunk.
const UNFOLD_BYTES: usize = 64 << 20;

impl Conv {
    pub fn load(t: &Tensors, prefix: &str) -> Result<Self> {
        let wn = format!("{prefix}.weight");
        let shape = t.shape(&wn)?;
        let [cout, cin, kh, kw] = shape[..] else {
            bail!("{wn}: expected a 4-D kernel, got {shape:?}")
        };
        if kh != kw || kh % 2 == 0 {
            bail!("{wn}: only square odd kernels are supported, got {kh}x{kw}");
        }
        let k = kh;
        let w = t.f32(&wn)?;
        // PyTorch [out][in][ky][kx] -> [out][ky][kx][in], the order of the unfolded pixel rows.
        let mut r = vec![0f32; w.len()];
        for o in 0..cout {
            for i in 0..cin {
                for ky in 0..k {
                    for kx in 0..k {
                        r[((o * k + ky) * k + kx) * cin + i] = w[((o * cin + i) * k + ky) * k + kx];
                    }
                }
            }
        }
        let lin = Linear::from_parts(&r, cout, k * k * cin, Linear::bias(t, prefix, cout)?)?;
        Ok(Self {
            lin,
            k,
            cin,
            cout,
            pad: (k / 2, k / 2),
        })
    }

    /// A convolution with asymmetric zero padding: `before` rows / columns above and left of the
    /// input, `after` below and right (diffusers' `Downsample2D` with `padding=0` pads (0, 1)).
    pub fn load_padded(t: &Tensors, prefix: &str, before: usize, after: usize) -> Result<Self> {
        let mut c = Self::load(t, prefix)?;
        c.pad = (before, after);
        Ok(c)
    }

    /// `x` is `[h*w, cin]`; returns `([oh*ow, cout], oh, ow)` for the given stride.
    pub fn forward(&self, x: &[f32], h: usize, w: usize, stride: usize) -> (Vec<f32>, usize, usize) {
        debug_assert_eq!(x.len(), h * w * self.cin);
        let g = self.geom(h, w, stride);
        if self.k == 1 && stride == 1 && self.pad == (0, 0) {
            let _t = timing::start(timing::CONV);
            return (self.lin.forward_untimed(x, h * w), h, w);
        }
        if let Some(y) = self.lin.device_conv(x, g, None) {
            check_gpu("conv", &y, || self.forward_cpu(x, h, w, stride));
            return (y, g.oh, g.ow);
        }
        (self.forward_cpu(x, h, w, stride), g.oh, g.ow)
    }

    fn geom(&self, h: usize, w: usize, stride: usize) -> crate::gpu::ConvGeom {
        let (before, after) = self.pad;
        let oh = (h + before + after - self.k) / stride + 1;
        let ow = (w + before + after - self.k) / stride + 1;
        crate::gpu::ConvGeom {
            h,
            w,
            cin: self.cin,
            k: self.k,
            stride,
            pad: before,
            oh,
            ow,
        }
    }

    /// The convolution of `pre(x)`: on the device with the normalization fused into the input
    /// loads when possible, otherwise normalized once on the CPU.
    pub fn forward_pre(&self, x: &[f32], h: usize, w: usize, stride: usize, pre: Pre) -> (Vec<f32>, usize, usize) {
        let g = self.geom(h, w, stride);
        if let Some(y) = self.lin.device_conv(x, g, Some(pre)) {
            check_gpu("conv+norm", &y, || {
                let xn = pre.apply(x, h * w);
                self.forward_cpu(&xn, h, w, stride)
            });
            return (y, g.oh, g.ow);
        }
        let xn = pre.apply(x, h * w);
        self.forward(&xn, h, w, stride)
    }

    /// Unfold on the CPU, then the shared matmul.
    fn forward_cpu(&self, x: &[f32], h: usize, w: usize, stride: usize) -> Vec<f32> {
        let (k, cin, cout) = (self.k, self.cin, self.cout);
        let g = self.geom(h, w, stride);
        let (pad, oh, ow) = (g.pad, g.oh, g.ow);
        if k == 1 && stride == 1 && pad == 0 {
            return self.lin.forward_untimed(x, h * w);
        }
        let row = k * k * cin;
        let chunk = (UNFOLD_BYTES / (row * 4)).max(ow).min(oh * ow);
        let mut out = vec![0f32; oh * ow * cout];
        let mut cols = vec![0f32; chunk * row];
        let mut p0 = 0;
        while p0 < oh * ow {
            let n = chunk.min(oh * ow - p0);
            let tu = timing::start(timing::UNFOLD);
            for_chunks(&mut cols[..n * row], row, true, |i, r| {
                let p = p0 + i;
                let (oy, ox) = (p / ow, p % ow);
                for ky in 0..k {
                    let iy = (oy * stride + ky) as isize - pad as isize;
                    for kx in 0..k {
                        let ix = (ox * stride + kx) as isize - pad as isize;
                        let dst = &mut r[(ky * k + kx) * cin..(ky * k + kx + 1) * cin];
                        if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                            dst.fill(0.0);
                        } else {
                            let s = (iy as usize * w + ix as usize) * cin;
                            dst.copy_from_slice(&x[s..s + cin]);
                        }
                    }
                }
            });
            drop(tu);
            let _t = timing::start(timing::CONV);
            let y = self.lin.forward_untimed(&cols[..n * row], n);
            out[p0 * cout..(p0 + n) * cout].copy_from_slice(&y);
            p0 += n;
        }
        out
    }
}

/// The normalization in front of an op: group norm (with SiLU or not) or layer norm.
#[derive(Clone, Copy)]
pub enum Pre<'a> {
    Group(&'a GroupNorm, bool),
    Layer(&'a LayerNorm),
}

impl Pre<'_> {
    fn spec(&self) -> crate::gpu::NormSpec<'_> {
        match *self {
            Pre::Group(g, silu) => crate::gpu::NormSpec {
                kind: crate::gpu::NormKind::Group(g.groups),
                eps: g.eps,
                gamma: &g.w,
                beta: &g.b,
                silu,
            },
            Pre::Layer(l) => crate::gpu::NormSpec {
                kind: crate::gpu::NormKind::Layer,
                eps: l.eps,
                gamma: &l.w,
                beta: &l.b,
                silu: false,
            },
        }
    }

    /// The normalized copy of `x` (`rows` pixels or rows), on the CPU.
    pub fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        match *self {
            Pre::Group(g, silu) => {
                let mut y = x.to_vec();
                g.apply(&mut y, rows, silu);
                y
            }
            Pre::Layer(l) => l.forward(x, rows),
        }
    }
}

/// Device-vs-CPU checking (`OXINFER_GPU_CHECK=1`, or `gpu_check::enable()`): every device op is
/// recomputed on the CPU; the worst relative error per op kind is kept for `gpu_check::summary()`.
pub mod gpu_check {
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU8, Ordering};

    static ON: AtomicU8 = AtomicU8::new(0); // 0 unknown, 1 off, 2 on
    static WORST: Mutex<BTreeMap<&'static str, (usize, f64)>> = Mutex::new(BTreeMap::new());

    pub fn enable() {
        ON.store(2, Ordering::Relaxed);
    }

    pub fn on() -> bool {
        match ON.load(Ordering::Relaxed) {
            0 => {
                let v = std::env::var_os("OXINFER_GPU_CHECK").is_some();
                ON.store(if v { 2 } else { 1 }, Ordering::Relaxed);
                v
            }
            v => v == 2,
        }
    }

    pub(super) fn record(what: &'static str, rel: f64) {
        let mut w = WORST.lock().unwrap();
        let e = w.entry(what).or_insert((0, 0.0));
        e.0 += 1;
        e.1 = e.1.max(rel);
    }

    /// "attention: 31 ops, worst rel L2 4.2e-6; ..." (empty when nothing was checked).
    pub fn summary() -> String {
        WORST
            .lock()
            .unwrap()
            .iter()
            .map(|(k, (n, r))| format!("{k}: {n} ops, worst rel L2 {r:.1e}"))
            .collect::<Vec<_>>()
            .join("; ")
    }
}

fn check_gpu(what: &'static str, y: &[f32], cpu: impl FnOnce() -> Vec<f32>) {
    if !gpu_check::on() {
        return;
    }
    let r = cpu();
    let (mut num, mut den) = (0f64, 0f64);
    for (a, b) in y.iter().zip(&r) {
        num += ((a - b) as f64).powi(2);
        den += (*b as f64).powi(2);
    }
    gpu_check::record(what, (num / den.max(1e-30)).sqrt());
}

impl Linear {
    /// True when the active accelerator takes this layer for `rows` inputs.
    pub fn on_device(&self, rows: usize) -> bool {
        crate::gpu::active().is_some_and(|be| be.wants(self.n_out, self.n_in, rows))
    }

    /// The layer as a convolution with geometry `g` on the active accelerator (a 1x1 convolution
    /// over `rows` pixels is a linear layer), optionally normalizing the input on the way in.
    fn device_conv(&self, x: &[f32], g: crate::gpu::ConvGeom, pre: Option<Pre>) -> Option<Vec<f32>> {
        let be = crate::gpu::active()?;
        let t = g.oh * g.ow;
        if !g.cin.is_multiple_of(32) || !be.wants(self.n_out, self.n_in, t) {
            return None;
        }
        let _t = timing::start(timing::CONV);
        let dev = self.w.device_weight(self.n_out, self.n_in)?;
        let mut y = vec![0f32; t * self.n_out];
        let spec = pre.as_ref().map(|p| p.spec());
        let mut job = crate::gpu::Job {
            w: self.w.as_ref(),
            dev,
            n_out: self.n_out,
            y: &mut y,
        };
        if !be.conv2d(x, &g, spec.as_ref(), &mut job) {
            return None;
        }
        if let Some(b) = &self.b {
            for_chunks(&mut y, self.n_out, true, |_, r| {
                for (v, &bb) in r.iter_mut().zip(b) {
                    *v += bb;
                }
            });
        }
        Some(y)
    }

    /// The layer applied to `pre(x)` (`rows` rows): normalization fused on the device when the
    /// accelerator takes it, otherwise normalized on the CPU first.
    pub fn forward_pre(&self, x: &[f32], rows: usize, pre: Pre) -> Vec<f32> {
        let g = crate::gpu::ConvGeom {
            h: rows,
            w: 1,
            cin: self.n_in,
            k: 1,
            stride: 1,
            pad: 0,
            oh: rows,
            ow: 1,
        };
        if let Some(y) = self.device_conv(x, g, Some(pre)) {
            check_gpu("linear+norm", &y, || self.forward(&pre.apply(x, rows), rows));
            return y;
        }
        self.forward(&pre.apply(x, rows), rows)
    }
}

/// Group normalization over channels-last activations, optionally fused with SiLU.
pub struct GroupNorm {
    w: Vec<f32>,
    b: Vec<f32>,
    groups: usize,
    eps: f32,
}

impl GroupNorm {
    pub fn load(t: &Tensors, prefix: &str, groups: usize, eps: f32) -> Result<Self> {
        let w = t.f32(&format!("{prefix}.weight"))?;
        let b = t.f32(&format!("{prefix}.bias"))?;
        if !w.len().is_multiple_of(groups) || b.len() != w.len() {
            bail!("{prefix}: {} channels do not split into {groups} groups", w.len());
        }
        Ok(Self { w, b, groups, eps })
    }

    pub fn apply(&self, x: &mut [f32], n: usize, silu_after: bool) {
        let _t = timing::start(timing::GN);
        let c = self.w.len();
        let g = self.groups;
        let cpg = c / g;
        debug_assert_eq!(x.len(), n * c);
        // One pass: each pixel chunk accumulates per-group sums (f64), then reduce.
        let chunks = (pool::threads() * 4).min(n).max(1);
        let per = n.div_ceil(chunks);
        let mut part = vec![0f64; chunks * g * 2];
        let pp = SyncPtr(part.as_mut_ptr());
        let xs: &[f32] = x;
        pool::run(chunks, &|ci| {
            let (p0, p1) = (ci * per, ((ci + 1) * per).min(n));
            let mut acc = vec![0f64; g * 2];
            for p in p0..p1 {
                let px = &xs[p * c..(p + 1) * c];
                for gi in 0..g {
                    let (mut s, mut s2) = (0f32, 0f32);
                    for &v in &px[gi * cpg..(gi + 1) * cpg] {
                        s += v;
                        s2 += v * v;
                    }
                    acc[gi * 2] += s as f64;
                    acc[gi * 2 + 1] += s2 as f64;
                }
            }
            // SAFETY: each chunk writes its own disjoint slot range.
            unsafe { std::ptr::copy_nonoverlapping(acc.as_ptr(), pp.get().add(ci * g * 2), g * 2) };
        });
        let cnt = (n * cpg) as f64;
        let mut mean = vec![0f32; g];
        let mut inv = vec![0f32; g];
        for gi in 0..g {
            let (mut s, mut s2) = (0f64, 0f64);
            for ci in 0..chunks {
                s += part[(ci * g + gi) * 2];
                s2 += part[(ci * g + gi) * 2 + 1];
            }
            let m = s / cnt;
            let var = (s2 / cnt - m * m).max(0.0);
            mean[gi] = m as f32;
            inv[gi] = (1.0 / (var + self.eps as f64).sqrt()) as f32;
        }
        let (w, b) = (&self.w, &self.b);
        for_chunks(x, c, n > 1, |_, px| {
            for ch in 0..c {
                let gi = ch / cpg;
                let v = (px[ch] - mean[gi]) * inv[gi] * w[ch] + b[ch];
                px[ch] = if silu_after { silu(v) } else { v };
            }
        });
    }
}

struct SyncPtr<T>(*mut T);
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    fn get(&self) -> *mut T {
        self.0
    }
}

/// Layer normalization over each row, with weight and bias.
pub struct LayerNorm {
    w: Vec<f32>,
    b: Vec<f32>,
    eps: f32,
}

impl LayerNorm {
    pub fn load(t: &Tensors, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            w: t.f32(&format!("{prefix}.weight"))?,
            b: t.f32(&format!("{prefix}.bias"))?,
            eps,
        })
    }

    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let _t = timing::start(timing::LN);
        let c = self.w.len();
        let mut y = x.to_vec();
        let (w, b, eps) = (&self.w, &self.b, self.eps);
        for_chunks(&mut y, c, rows > 1, |_, r| {
            let mean = r.iter().map(|&v| v as f64).sum::<f64>() / c as f64;
            let var = r.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / c as f64;
            let inv = 1.0 / (var + eps as f64).sqrt();
            for (i, v) in r.iter_mut().enumerate() {
                *v = ((*v as f64 - mean) * inv) as f32 * w[i] + b[i];
            }
        });
        y
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU with the exact erf (PyTorch's default `F.gelu`).
#[inline]
pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// OpenAI CLIP's `quick_gelu`: x * sigmoid(1.702 x).
#[inline]
pub fn quick_gelu(x: f32) -> f32 {
    x / (1.0 + (-1.702 * x).exp())
}

/// erf from the Numerical Recipes Chebyshev fit of erfc (fractional error below 1.2e-7),
/// evaluated in f64.
pub fn erf(x: f32) -> f32 {
    let x = x as f64;
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let y = t
        * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    (if x >= 0.0 { 1.0 - y } else { y - 1.0 }) as f32
}

pub fn silu_inplace(x: &mut [f32]) {
    for_chunks(x, 4096, x.len() > 65536, |_, c| c.iter_mut().for_each(|v| *v = silu(*v)));
}

pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    crate::kernels::add_inplace(a, b);
}

/// Nearest-neighbour 2x upsampling of `[h*w, c]`.
pub fn upsample2x(x: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let (oh, ow) = (h * 2, w * 2);
    let mut y = vec![0f32; oh * ow * c];
    for_chunks(&mut y, ow * c, true, |oy, row| {
        let sy = oy / 2;
        for ox in 0..ow {
            let s = (sy * w + ox / 2) * c;
            row[ox * c..(ox + 1) * c].copy_from_slice(&x[s..s + c]);
        }
    });
    y
}

/// Per-pixel channel concatenation `[n, ca] ++ [n, cb] -> [n, ca + cb]`.
pub fn concat_channels(a: &[f32], ca: usize, b: &[f32], cb: usize, n: usize) -> Vec<f32> {
    let c = ca + cb;
    let mut y = vec![0f32; n * c];
    for_chunks(&mut y, c, n > 1, |p, r| {
        r[..ca].copy_from_slice(&a[p * ca..(p + 1) * ca]);
        r[ca..].copy_from_slice(&b[p * cb..(p + 1) * cb]);
    });
    y
}

/// Sinusoidal timestep embedding (diffusers `get_timestep_embedding`).
pub fn timestep_embedding(t: f32, dim: usize, flip_sin_to_cos: bool, freq_shift: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut e = vec![0f32; dim];
    for i in 0..half {
        let exponent = -(10000f64.ln()) * i as f64 / (half as f64 - freq_shift as f64);
        let a = t as f64 * exponent.exp();
        let (s, c) = (a.sin() as f32, a.cos() as f32);
        if flip_sin_to_cos {
            e[i] = c;
            e[half + i] = s;
        } else {
            e[i] = s;
            e[half + i] = c;
        }
    }
    e
}

// ---- attention -----------------------------------------------------------------------------------

/// A plain f32 matrix used as the "weights" side of the quantized-matmul driver, which then runs
/// its f32 GEMM tiles (attention scores and the probability-times-value product).
struct F32Mat<'a> {
    m: &'a [f32],
    n_out: usize,
    n_in: usize,
}

impl QMat for F32Mat<'_> {
    fn n_out(&self) -> usize {
        self.n_out
    }
    fn n_in(&self) -> usize {
        self.n_in
    }
    fn bytes(&self) -> usize {
        self.m.len() * 4
    }
    fn dequantize_rows(&self, r0: usize, r1: usize, out: &mut [f32]) {
        out[..(r1 - r0) * self.n_in].copy_from_slice(&self.m[r0 * self.n_in..r1 * self.n_in]);
    }
    unsafe fn rows_t1(&self, _: &QAct, _: usize, _: *mut f32, _: usize, _: usize) {
        unreachable!("f32 matmuls always run with at least 8 rows")
    }
    unsafe fn rows_tc(&self, _: &QAct, _: usize, _: usize, _: *mut f32, _: usize, _: usize) {
        unreachable!("f32 matmuls always run with at least 8 rows")
    }
}

/// `y[t][n_out] = x[t][n_in] * m[n_out][n_in]^T` in f32 (pads `t` up to 8).
pub(crate) fn matmul_f32(m: &[f32], n_out: usize, n_in: usize, x: &[f32], t: usize) -> Vec<f32> {
    let tp = t.max(8);
    let mut xp;
    let xs = if tp != t {
        xp = vec![0f32; tp * n_in];
        xp[..t * n_in].copy_from_slice(x);
        &xp[..]
    } else {
        x
    };
    let mut y = vec![0f32; tp * n_out];
    let mat = F32Mat { m, n_out, n_in };
    crate::kernels::matmul_quant_multi(&mut [(&mat, &mut y[..])], xs, tp);
    y.truncate(t * n_out);
    y
}

/// Multi-head attention. `q` is `[tq, heads*dh]`, `k` and `v` are `[tk, heads*dh]`; returns
/// `[tq, heads*dh]`. `causal` masks keys after each query (CLIP's text encoder).
#[allow(clippy::too_many_arguments)]
pub fn attention(q: &[f32], k: &[f32], v: &[f32], tq: usize, tk: usize, heads: usize, dh: usize, causal: bool) -> Vec<f32> {
    let _t = timing::start(timing::ATTN);
    // Big enough to beat a device round trip: on the accelerator.
    if let Some(be) = crate::gpu::active()
        && 4.0 * (tq * tk * heads * dh) as f64 >= 1e8
    {
        let mut out = vec![0f32; tq * heads * dh];
        if be.attention(q, k, v, tq, tk, heads, dh, causal, &mut out) {
            check_gpu("attention", &out, || attention_cpu(q, k, v, tq, tk, heads, dh, causal));
            return out;
        }
    }
    attention_cpu(q, k, v, tq, tk, heads, dh, causal)
}

#[allow(clippy::too_many_arguments)]
fn attention_cpu(q: &[f32], k: &[f32], v: &[f32], tq: usize, tk: usize, heads: usize, dh: usize, causal: bool) -> Vec<f32> {
    let d = heads * dh;
    // Zero padding to multiples of 32 keeps the SIMD GEMM tiles on (and changes no dot product).
    let dhp = dh.next_multiple_of(32);
    let tkp = tk.next_multiple_of(32);
    let scale = 1.0 / (dh as f32).sqrt();
    let mut out = vec![0f32; tq * d];
    let mut qh = vec![0f32; tq * dhp];
    let mut kh = vec![0f32; tk * dhp];
    let mut vt = vec![0f32; dh * tkp];
    for h in 0..heads {
        for_chunks(&mut qh, dhp, tq > 64, |i, r| {
            r[..dh].copy_from_slice(&q[i * d + h * dh..i * d + (h + 1) * dh])
        });
        for_chunks(&mut kh, dhp, tk > 64, |j, r| {
            r[..dh].copy_from_slice(&k[j * d + h * dh..j * d + (h + 1) * dh])
        });
        for_chunks(&mut vt, tkp, true, |c, r| {
            for j in 0..tk {
                r[j] = v[j * d + h * dh + c];
            }
        });
        // Scores [tq, tk], then softmax into probabilities padded to [tq, tkp].
        let s = matmul_f32(&kh, tk, dhp, &qh, tq);
        let mut p = vec![0f32; tq * tkp];
        for_chunks(&mut p, tkp, tq > 1, |i, r| {
            let row = &s[i * tk..(i + 1) * tk];
            let lim = if causal { i + 1 } else { tk };
            let mut mx = f32::NEG_INFINITY;
            for &x in &row[..lim] {
                mx = mx.max(x * scale);
            }
            let mut sum = 0f32;
            for j in 0..lim {
                let e = (row[j] * scale - mx).exp();
                r[j] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for x in &mut r[..lim] {
                *x *= inv;
            }
        });
        let o = matmul_f32(&vt, dh, tkp, &p, tq);
        for_chunks(&mut out, d, tq > 64, |i, r| {
            r[h * dh..(h + 1) * dh].copy_from_slice(&o[i * dh..(i + 1) * dh])
        });
    }
    out
}

// ---- random numbers ------------------------------------------------------------------------------

/// xoshiro256** seeded through splitmix64, with Box-Muller normals.
pub struct Rng {
    s: [u64; 4],
    spare: Option<f32>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Uniform in (0, 1].
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let a = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some((r * a.sin()) as f32);
        (r * a.cos()) as f32
    }

    pub fn normals(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.normal()).collect()
    }
}
