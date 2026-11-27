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
    pub const NAMES: [&str; 8] = [
        "conv unfold",
        "conv matmul",
        "linear",
        "attention",
        "group norm",
        "layer norm",
        "other",
        "device pass",
    ];
    static NS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
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
    /// Whole UNet passes on the accelerator (`crate::gpu::unet`).
    pub const PASS: usize = 7;
}

/// One safetensors file of a diffusers component (`unet/`, `vae/`, `text_encoder/`), with the
/// LoRA updates to merge into its weights as they are read.
pub struct Tensors {
    st: SafeTensors,
    lora: Option<super::lora::Patches>,
    /// Names the loaders ask for -> names stored in the file (checkpoints in another layout).
    alias: Option<std::collections::HashMap<String, String>>,
    /// Weights stored as F32 load as two bf16 parts (`w = hi + lo`, ~16 significant bits
    /// instead of 8) for layers whose output is sensitive to weight rounding (see
    /// [`Tensors::set_split_f32`]).
    split_f32: bool,
}

impl Tensors {
    pub fn open(path: &Path) -> Result<Self> {
        // Mapped where the platform can: every weight is converted on load anyway.
        Ok(Self {
            st: SafeTensors::open(path, cfg!(target_arch = "wasm32"))?,
            lora: None,
            alias: None,
            split_f32: false,
        })
    }

    /// Opens a checkpoint stored under other names: `rename` maps each stored name to the name
    /// the loaders use (None: keep it).
    pub fn open_renamed(path: &Path, rename: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let mut t = Self::open(path)?;
        let alias = t.st.names().filter_map(|n| rename(n).map(|a| (a, n.clone()))).collect();
        t.alias = Some(alias);
        Ok(t)
    }

    /// The stored name of `name`.
    fn real<'n>(&'n self, name: &'n str) -> &'n str {
        self.alias.as_ref().and_then(|a| a.get(name)).map_or(name, |s| s.as_str())
    }

    /// Linear and convolution weights stored as F32 are then kept as a bf16 pair (`hi + lo`,
    /// run as two matmuls) instead of rounded to bf16: e.g. Step1X-Edit's f32 FLUX VAE, whose
    /// encoder moves its latents by 1.5% when its weights are rounded.
    pub fn set_split_f32(&mut self, on: bool) {
        self.split_f32 = on;
    }

    fn split(&self, name: &str) -> bool {
        self.split_f32 && self.st.info(self.real(name)).is_some_and(|i| i.dtype == "F32")
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
        self.st.info(self.real(name)).is_some()
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self
            .st
            .info(self.real(name))
            .with_context(|| format!("missing tensor {name}"))?
            .shape
            .clone())
    }

    /// A tensor as f32, whatever its stored float type, with any LoRA updates merged in.
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.st.info(self.real(name)).with_context(|| format!("missing tensor {name}"))?;
        let b = self.st.bytes_cow(self.real(name))?;
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
        let info = self.st.info(self.real(name)).with_context(|| format!("missing tensor {name}"))?;
        if info.dtype != "F16" || self.lora.as_ref().is_some_and(|p| p.targets(name)) {
            return Ok(None);
        }
        Ok(Some(
            self.st
                .bytes_cow(self.real(name))?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect(),
        ))
    }
}

/// `y += x`.
fn add_into(y: &mut [f32], x: &[f32]) {
    for_chunks(y, 1 << 16, y.len() > 1 << 16, |c, r| {
        for (a, &b) in r.iter_mut().zip(&x[c << 16..]) {
            *a += b;
        }
    });
}

/// A PyTorch `[out][in][ky][kx]` kernel as `[out][ky][kx][in]` rows, the order of the unfolded
/// pixel rows.
fn kernel_rows(w: &[f32], cout: usize, cin: usize, k: usize) -> Vec<f32> {
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
    r
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

/// `w` (`[n_out, n_in]`) as a float32 "GGUF" weight.
fn f32_lin(w: &[f32], n_out: usize, n_in: usize) -> Lin<'static> {
    let mut v = PageVec::<u8>::zeroed(w.len() * 4);
    for (o, x) in v.as_chunks_mut::<4>().0.iter_mut().zip(w) {
        *o = x.to_le_bytes();
    }
    Lin::gguf(crate::ggml::GgufMatrix {
        kind: crate::ggml::GgmlType::F32,
        data: v.into(),
        n_out,
        n_in,
        block_scales: None,
        x4: false,
    })
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
    /// The rounding residual of `w` (f32 checkpoints with [`Tensors::set_split_f32`]).
    lo: Option<Box<Linear>>,
    /// For a device whose kernels take whole tiles of 64 outputs: this layer with zero rows up
    /// to the next multiple of 64 (made on first use; None when the rows cannot be read).
    padded: std::sync::OnceLock<Option<Box<Linear>>>,
    /// The float32 weights of a split layer (`lo`) as one float32 matrix, for an accelerator that
    /// convolves with float32 weights directly (one convolution instead of the bf16 pair).
    exact: Option<Lin<'static>>,
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
        let l = Self::from_parts(&w, n_out, n_in, Self::bias(t, prefix, n_out)?)?;
        Ok(if t.split(&wn) { l.with_residual(&w)? } else { l })
    }

    /// Adds the bf16 rounding residual of `w` (the weights `self` was built from) as a second
    /// part.
    fn with_residual(mut self, w: &[f32]) -> Result<Self> {
        let r: Vec<f32> = w.iter().map(|&v| v - f32::from_bits((f32_to_bf16(v) as u32) << 16)).collect();
        self.lo = Some(Box::new(Self::from_parts(&r, self.n_out, self.n_in, None)?));
        if crate::gpu::active().is_some() {
            self.exact = Some(f32_lin(w, self.n_out, self.n_in));
        }
        Ok(self)
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
            lo: None,
            padded: std::sync::OnceLock::new(),
            exact: None,
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
        if let Some(lo) = &self.lo {
            add_into(&mut y, &lo.forward_untimed(x, rows));
        }
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
    /// For whole-pass device plans: the matrix with zero columns up to a multiple of 64 inputs
    /// and zero rows up to a multiple of 64 outputs (made on first use; None when the rows cannot
    /// be read).
    plan_lin: std::sync::OnceLock<Option<Box<Linear>>>,
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
        let w = t.f32(&wn)?;
        let mut c = Self::from_kernel(&w, cout, cin, kh, Linear::bias(t, prefix, cout)?)?;
        if t.split(&wn) {
            c.lin = c.lin.with_residual(&kernel_rows(&w, cout, cin, kh))?;
        }
        Ok(c)
    }

    /// A k x k convolution (padding k/2) from a kernel in PyTorch layout `[cout, cin, k, k]`
    /// and its bias.
    pub fn from_kernel(w: &[f32], cout: usize, cin: usize, k: usize, bias: Option<Vec<f32>>) -> Result<Self> {
        if w.len() != cout * cin * k * k || k.is_multiple_of(2) {
            bail!("convolution kernel: {} values for {cout}x{cin}x{k}x{k}", w.len());
        }
        let lin = Linear::from_parts(&kernel_rows(w, cout, cin, k), cout, k * k * cin, bias)?;
        Ok(Self {
            lin,
            k,
            cin,
            cout,
            pad: (k / 2, k / 2),
            plan_lin: std::sync::OnceLock::new(),
        })
    }

    /// [`Conv::from_kernel`], exact in float32 when `exact` (the bf16 weights plus their rounding
    /// residual on the CPU, the float32 weights on an accelerator).
    pub fn from_kernel_exact(w: &[f32], cout: usize, cin: usize, k: usize, bias: Option<Vec<f32>>, exact: bool) -> Result<Self> {
        let mut c = Self::from_kernel(w, cout, cin, k, bias)?;
        if exact {
            c.lin = c.lin.with_residual(&kernel_rows(w, cout, cin, k))?;
        }
        Ok(c)
    }

    /// A convolution with asymmetric zero padding: `before` rows / columns above and left of the
    /// input, `after` below and right (diffusers' `Downsample2D` with `padding=0` pads (0, 1)).
    pub fn load_padded(t: &Tensors, prefix: &str, before: usize, after: usize) -> Result<Self> {
        let mut c = Self::load(t, prefix)?;
        c.pad = (before, after);
        Ok(c)
    }

    /// This convolution with zero padding `before` (top / left) and `after` (bottom / right).
    pub fn with_padding(mut self, before: usize, after: usize) -> Self {
        self.pad = (before, after);
        self
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
        if !self.n_out.is_multiple_of(64) && g.cin.is_multiple_of(32) && be.wants(self.n_out.next_multiple_of(64), self.n_in, t) {
            // Through the zero-padded layer, then the real outputs of each row.
            let p = self.padded.get_or_init(|| self.padded_copy().map(Box::new)).as_ref()?;
            let yp = p.device_conv(x, g, pre)?;
            let (n, np) = (self.n_out, p.n_out);
            let mut y = vec![0f32; t * n];
            for_chunks(&mut y, n * 256, t > 1024, |c, out| {
                for (j, o) in out.chunks_mut(n).enumerate() {
                    let r = c * 256 + j;
                    o.copy_from_slice(&yp[r * np..r * np + n]);
                }
            });
            return Some(y);
        }
        if !g.cin.is_multiple_of(32) || !be.wants(self.n_out, self.n_in, t) {
            return None;
        }
        let _t = timing::start(timing::CONV);
        let mut y = vec![0f32; t * self.n_out];
        let spec = pre.as_ref().map(|p| p.spec());
        // A split layer as one float32 convolution where the backend has one.
        if let Some(ex) = &self.exact
            && let Some(dev) = ex.device_weight(self.n_out, self.n_in)
        {
            let mut job = crate::gpu::Job {
                w: ex.as_ref(),
                dev,
                n_out: self.n_out,
                y: &mut y,
            };
            if be.conv2d(x, &g, spec.as_ref(), &mut job) {
                if let Some(b) = &self.b {
                    for_chunks(&mut y, self.n_out, true, |_, r| {
                        for (v, &bb) in r.iter_mut().zip(b) {
                            *v += bb;
                        }
                    });
                }
                return Some(y);
            }
        }
        let dev = self.w.device_weight(self.n_out, self.n_in)?;
        let mut job = crate::gpu::Job {
            w: self.w.as_ref(),
            dev,
            n_out: self.n_out,
            y: &mut y,
        };
        if !be.conv2d(x, &g, spec.as_ref(), &mut job) {
            return None;
        }
        if let Some(lo) = &self.lo {
            add_into(&mut y, &lo.device_conv(x, g, pre)?);
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

    /// This layer (and its residual part) with zero rows up to a multiple of 64 outputs.
    fn padded_copy(&self) -> Option<Linear> {
        let np = self.n_out.next_multiple_of(64);
        let mut w = vec![0f32; np * self.n_in];
        let idx: Vec<u32> = (0..self.n_out as u32).collect();
        if !self.w.rows(&idx, self.n_in, &mut w[..self.n_out * self.n_in]) {
            return None;
        }
        let b = self.b.as_ref().map(|b| {
            let mut v = b.clone();
            v.resize(np, 0.0);
            v
        });
        let mut l = Self::from_parts(&w, np, self.n_in, b).ok()?;
        if let Some(lo) = &self.lo {
            l.lo = Some(Box::new(lo.padded_copy()?));
        }
        if let Some(ex) = &self.exact {
            let mut we = vec![0f32; np * self.n_in];
            if !ex.rows(&idx, self.n_in, &mut we[..self.n_out * self.n_in]) {
                return None;
            }
            l.exact = Some(f32_lin(&we, np, self.n_in));
        }
        Some(l)
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

// ---- whole-pass device plans (crate::gpu::unet) ---------------------------------------------------

impl Linear {
    /// This layer as a whole-pass plan sees it (bf16 weights on the active accelerator; None for
    /// split float32 layers).
    pub fn plan_lin(&self) -> Option<crate::gpu::unet::Lin<'_>> {
        if self.lo.is_some() || !matches!(self.w.as_ref(), crate::gpu::WeightRef::Bf16(_)) {
            return None;
        }
        Some(crate::gpu::unet::Lin {
            w: self.w.as_ref(),
            dev: self.w.device_weight(self.n_out, self.n_in)?,
            n_out: self.n_out,
            n_in: self.n_in,
            bias: self.b.as_deref(),
        })
    }

    /// This layer with zero rows up to `n_out` outputs and `n_in` inputs per row: each row's
    /// `taps` groups of `cin` inputs spread `cin_p` apart, zeros after them.
    fn zero_padded(&self, n_out: usize, n_in: usize, taps: usize, cin: usize, cin_p: usize) -> Option<Linear> {
        let mut w = vec![0f32; self.n_out * self.n_in];
        let idx: Vec<u32> = (0..self.n_out as u32).collect();
        if self.lo.is_some() || !self.w.rows(&idx, self.n_in, &mut w) {
            return None;
        }
        let mut wp = vec![0f32; n_out * n_in];
        for (o, r) in w.chunks(self.n_in).enumerate() {
            for (t, g) in r.chunks(cin).take(taps).enumerate() {
                wp[o * n_in + t * cin_p..o * n_in + t * cin_p + cin].copy_from_slice(g);
            }
        }
        let b = self.b.as_ref().map(|b| {
            let mut v = b.clone();
            v.resize(n_out, 0.0);
            v
        });
        Self::from_parts(&wp, n_out, n_in, b).ok()
    }
}

impl Conv {
    /// The convolution's matrix for a whole-pass plan: as loaded when its sizes suit the device
    /// GEMMs, else a zero-padded copy (inputs to a multiple of 64 when `cin` is not a multiple of
    /// 32, outputs to a multiple of 64: the first `cout` are the real ones).
    pub fn plan_lin(&self) -> Option<crate::gpu::unet::Lin<'_>> {
        let l = &self.lin;
        if self.cin.is_multiple_of(32) && l.n_out.is_multiple_of(64) {
            return l.plan_lin();
        }
        let taps = self.k * self.k;
        let cin_p = self.plan_cin();
        let n_in = if self.cin.is_multiple_of(32) {
            l.n_in
        } else {
            (taps * cin_p).next_multiple_of(64)
        };
        let p = self
            .plan_lin
            .get_or_init(|| {
                l.zero_padded(l.n_out.next_multiple_of(64), n_in, taps, self.cin, cin_p)
                    .map(Box::new)
            })
            .as_ref()?;
        p.plan_lin()
    }

    /// The input channels a whole-pass plan feeds this convolution: `cin` rounded up to a
    /// multiple of 4 (zero channels after the real ones; [`Conv::plan_lin`] spreads its taps to
    /// match).
    pub fn plan_cin(&self) -> usize {
        self.cin.next_multiple_of(4)
    }

    /// Zero padding before the input (top / left).
    pub fn pad_before(&self) -> usize {
        self.pad.0
    }

    /// Zero padding before and after the input.
    pub(crate) fn pads(&self) -> (usize, usize) {
        self.pad
    }

    /// The kernel as float32 rows `[cout][ky][kx][cin]` and the bias, for an engine that keeps
    /// its own copy on a device (None for split float32 layers and rows that cannot be read).
    pub(crate) fn rows_f32(&self) -> Option<(Vec<f32>, Option<&[f32]>)> {
        let l = &self.lin;
        let mut w = vec![0f32; l.n_out * l.n_in];
        let idx: Vec<u32> = (0..l.n_out as u32).collect();
        if l.lo.is_some() || !l.w.rows(&idx, l.n_in, &mut w) {
            return None;
        }
        Some((w, l.b.as_deref()))
    }
}

impl GroupNorm {
    pub fn plan_norm(&self) -> crate::gpu::unet::Norm<'_> {
        crate::gpu::unet::Norm {
            gamma: &self.w,
            beta: &self.b,
            eps: self.eps,
            groups: self.groups,
        }
    }
}

impl LayerNorm {
    pub fn plan_norm(&self) -> crate::gpu::unet::Norm<'_> {
        crate::gpu::unet::Norm {
            gamma: &self.w,
            beta: &self.b,
            eps: self.eps,
            groups: 0,
        }
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
    /// Weight, bias and epsilon.
    pub(crate) fn params(&self) -> (&[f32], &[f32], f32) {
        (&self.w, &self.b, self.eps)
    }

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
    // Float-stored: the driver takes the f32 GEMM at any row count, never the int8 decode
    // kernels (whose row limit, `kernels::decode_max_tokens`, speculative decoding raises).
    fn dense(&self) -> bool {
        true
    }
    fn dequantize_rows(&self, r0: usize, r1: usize, out: &mut [f32]) {
        out[..(r1 - r0) * self.n_in].copy_from_slice(&self.m[r0 * self.n_in..r1 * self.n_in]);
    }
    unsafe fn rows_t1(&self, _: &QAct, _: usize, _: *mut f32, _: usize, _: usize) {
        unreachable!("f32 matmuls run on the GEMM (dense)")
    }
    unsafe fn rows_tc(&self, _: &QAct, _: usize, _: usize, _: *mut f32, _: usize, _: usize) {
        unreachable!("f32 matmuls run on the GEMM (dense)")
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
    #[cfg(has_simd)]
    if tq * tk >= 64 * 64 && std::env::var_os("OXINFER_ATTN_TWO_PASS").is_none() {
        return attention_blocked(q, k, v, tq, tk, heads, dh, causal);
    }
    attention_two_pass(q, k, v, tq, tk, heads, dh, causal)
}

/// Query rows per task of [`attention_blocked`] (a multiple of the GEMM tile's 8).
const ATT_QB: usize = 32;
/// Keys per block of [`attention_blocked`] (a multiple of 8).
const ATT_KB: usize = 64;

/// [`attention_two_pass`] as one task per (head, block of 32 queries), each streaming blocks of 64
/// keys with an online softmax ("flash" attention) on the f32 GEMM micro-kernel: no `[tq, tk]`
/// score matrix, and every core busy on its own tiles rather than all of them on one head's
/// small products (which spends most of the time in the pool at a few thousand rows). The
/// softmax is the same up to rounding (running maxima instead of one maximum per row).
#[cfg(has_simd)]
#[allow(clippy::too_many_arguments)]
fn attention_blocked(q: &[f32], k: &[f32], v: &[f32], tq: usize, tk: usize, heads: usize, dh: usize, causal: bool) -> Vec<f32> {
    use crate::kernels::simd::gemm_8x8;
    let d = heads * dh;
    let scale = 1.0 / (dh as f32).sqrt();
    let nkb = tk.div_ceil(ATT_KB);
    let tkp = nkb * ATT_KB;
    let dhp = dh.next_multiple_of(8);
    // Keys per head, row-major `[tkp, dh]` (zero rows past `tk`); values per head and key block,
    // transposed: `[nkb][dhp][KB]` (the PV product's "weights" are the value columns).
    let mut kp = vec![0f32; heads * tkp * dh];
    for_chunks(&mut kp, tkp * dh, true, |h, kh| {
        for j in 0..tk {
            kh[j * dh..(j + 1) * dh].copy_from_slice(&k[j * d + h * dh..j * d + (h + 1) * dh]);
        }
    });
    let mut vt = vec![0f32; heads * nkb * dhp * ATT_KB];
    for_chunks(&mut vt, nkb * dhp * ATT_KB, true, |h, vh| {
        for j in 0..tk {
            let (b, jj) = (j / ATT_KB, j % ATT_KB);
            let row = &v[j * d + h * dh..j * d + (h + 1) * dh];
            for (c, &x) in row.iter().enumerate() {
                vh[(b * dhp + c) * ATT_KB + jj] = x;
            }
        }
    });
    let nqb = tq.div_ceil(ATT_QB);
    let mut out = vec![0f32; tq * d];
    let op = SendPtr(out.as_mut_ptr());
    crate::pool::run(heads * nqb, &|task| {
        let (h, qb) = (task / nqb, task % nqb);
        let q0 = qb * ATT_QB;
        let nq = (tq - q0).min(ATT_QB);
        let tiles = nq.div_ceil(8);
        // The queries as GEMM panels `[tile][dh][8]`, scaled.
        let mut qt = vec![0f32; tiles * dh * 8];
        for i in 0..nq {
            let src = &q[(q0 + i) * d + h * dh..(q0 + i) * d + (h + 1) * dh];
            let (t, l) = (i / 8, i % 8);
            for (c, &x) in src.iter().enumerate() {
                qt[(t * dh + c) * 8 + l] = x * scale;
            }
        }
        let mut s = vec![0f32; tiles * 8 * ATT_KB];
        let mut pt = vec![0f32; tiles * ATT_KB * 8];
        let mut tmp = vec![0f32; tiles * 8 * dhp];
        let mut acc = vec![0f32; nq * dhp];
        let mut m = vec![f32::NEG_INFINITY; nq];
        let mut l = vec![0f32; nq];
        let kh = &kp[h * tkp * dh..(h + 1) * tkp * dh];
        let vh = &vt[h * nkb * dhp * ATT_KB..(h + 1) * nkb * dhp * ATT_KB];
        // Causal: the blocks up to the last query's key.
        let last_kb = if causal { ((q0 + nq - 1) / ATT_KB + 1).min(nkb) } else { nkb };
        for b in 0..last_kb {
            let j0 = b * ATT_KB;
            // Scores `[tiles * 8, KB]`.
            for g in 0..ATT_KB / 8 {
                for t in 0..tiles {
                    unsafe {
                        gemm_8x8::<f32>(
                            kh.as_ptr().add((j0 + g * 8) * dh),
                            dh,
                            qt.as_ptr().add(t * dh * 8),
                            s.as_mut_ptr(),
                            ATT_KB,
                            g * 8,
                            t * 8,
                            8,
                        );
                    }
                }
            }
            // Online softmax: rescale what was accumulated to the new maximum.
            for i in 0..nq {
                let row = &mut s[i * ATT_KB..(i + 1) * ATT_KB];
                let lim = if causal {
                    (q0 + i + 1).saturating_sub(j0).min(ATT_KB)
                } else {
                    ATT_KB
                };
                let lim = lim.min(tk.saturating_sub(j0));
                let mut mx = m[i];
                for &x in &row[..lim] {
                    mx = mx.max(x);
                }
                let (t, ln) = (i / 8, i % 8);
                if mx == f32::NEG_INFINITY {
                    for jj in 0..ATT_KB {
                        pt[(t * ATT_KB + jj) * 8 + ln] = 0.0;
                    }
                    continue;
                }
                let corr = (m[i] - mx).exp();
                let mut sum = 0f32;
                for (jj, &x) in row.iter().enumerate() {
                    let e = if jj < lim { (x - mx).exp() } else { 0.0 };
                    pt[(t * ATT_KB + jj) * 8 + ln] = e;
                    sum += e;
                }
                l[i] = l[i] * corr + sum;
                m[i] = mx;
                if corr != 1.0 {
                    acc[i * dhp..(i + 1) * dhp].iter_mut().for_each(|a| *a *= corr);
                }
            }
            // Rows of a partial last tile have zero probabilities: their products are zero.
            for i in nq..tiles * 8 {
                let (t, ln) = (i / 8, i % 8);
                for jj in 0..ATT_KB {
                    pt[(t * ATT_KB + jj) * 8 + ln] = 0.0;
                }
            }
            // `tmp = P V` for this block, `[tiles * 8, dhp]`, then accumulated.
            let vb = &vh[b * dhp * ATT_KB..(b + 1) * dhp * ATT_KB];
            for g in 0..dhp / 8 {
                for t in 0..tiles {
                    unsafe {
                        gemm_8x8::<f32>(
                            vb.as_ptr().add(g * 8 * ATT_KB),
                            ATT_KB,
                            pt.as_ptr().add(t * ATT_KB * 8),
                            tmp.as_mut_ptr(),
                            dhp,
                            g * 8,
                            t * 8,
                            8,
                        );
                    }
                }
            }
            for (a, &x) in acc.iter_mut().zip(&tmp[..nq * dhp]) {
                *a += x;
            }
        }
        for i in 0..nq {
            let inv = if l[i] > 0.0 { 1.0 / l[i] } else { 0.0 };
            // SAFETY: each task writes its own (rows, head) slices of `out`.
            let o = unsafe { std::slice::from_raw_parts_mut(op.get().add((q0 + i) * d + h * dh), dh) };
            for (o, &a) in o.iter_mut().zip(&acc[i * dhp..i * dhp + dh]) {
                *o = a * inv;
            }
        }
    });
    out
}

#[cfg(has_simd)]
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
#[cfg(has_simd)]
unsafe impl Send for SendPtr {}
#[cfg(has_simd)]
unsafe impl Sync for SendPtr {}
#[cfg(has_simd)]
impl SendPtr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// One head at a time: the scores of all queries against all keys, softmax, then the product with
/// the values, each product on the (parallel) f32 GEMM.
#[allow(clippy::too_many_arguments)]
fn attention_two_pass(q: &[f32], k: &[f32], v: &[f32], tq: usize, tk: usize, heads: usize, dh: usize, causal: bool) -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The blocked attention against the two-pass one: partial query tiles and key blocks, head
    /// widths that are not multiples of 8, causal masks, fewer keys than queries.
    #[cfg(has_simd)]
    #[test]
    fn blocked_attention_matches_two_pass() {
        let mut rng = Rng::new(7);
        for &(tq, tk, heads, dh, causal) in &[
            (100usize, 100usize, 3usize, 64usize, false),
            (77, 130, 2, 80, false),
            (64, 64, 1, 128, true),
            (129, 129, 2, 72, true),
            (40, 300, 4, 36, false),
            (300, 40, 2, 64, false),
        ] {
            let d = heads * dh;
            let q: Vec<f32> = rng.normals(tq * d).iter().map(|x| x * 2.0).collect();
            let k: Vec<f32> = rng.normals(tk * d).iter().map(|x| x * 2.0).collect();
            let v = rng.normals(tk * d);
            let a = attention_blocked(&q, &k, &v, tq, tk, heads, dh, causal);
            let b = attention_two_pass(&q, &k, &v, tq, tk, heads, dh, causal);
            let max = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            assert!(max < 2e-5, "tq {tq} tk {tk} heads {heads} dh {dh} causal {causal}: max diff {max}");
        }
    }

    /// f32 matmuls stay on the GEMM even when speculative decoding has raised the decode-kernel
    /// row limit above their (padded) row count.
    #[test]
    fn f32_matmul_ignores_decode_row_limit() {
        let (n_out, n_in, t) = (24, 64, 9);
        let m: Vec<f32> = (0..n_out * n_in).map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0).collect();
        let x: Vec<f32> = (0..t * n_in).map(|i| ((i * 5 % 11) as f32 - 5.0) / 4.0).collect();
        let mat = F32Mat { m: &m, n_out, n_in };
        let mut y = vec![0f32; t * n_out];
        crate::kernels::matmul_quant_multi_at(&mut [(&mat, &mut y[..])], &x, t, 16);
        for r in 0..t {
            for o in 0..n_out {
                let want: f64 = (0..n_in).map(|k| m[o * n_in + k] as f64 * x[r * n_in + k] as f64).sum();
                assert!((y[r * n_out + o] as f64 - want).abs() < 1e-3, "row {r} out {o}");
            }
        }
    }
}
