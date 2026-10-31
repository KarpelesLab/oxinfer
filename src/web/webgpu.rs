//! The WebGPU backend as seen from the module: weights are uploaded through `gpu_upload`, and a
//! batch of prefill matmuls over one input runs through `gpu_matmul`, both implemented by the host
//! in JavaScript (see `web/oxinfer.js`, which also holds the WGSL shaders). The host suspends the
//! module during the asynchronous readback (JSPI), so from here the calls look synchronous.

use crate::gpu::{Backend, DeviceWeight, Job, WeightRef};
use std::sync::atomic::{AtomicU64, Ordering};

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// Non-zero when the host has a WebGPU device and can suspend the module (JSPI).
    fn gpu_available() -> i32;
    /// Uploads a weight matrix. `kind`: 0 bf16, 1 q8, 2 q4, 4 affine q4 (`scales` holds the f16
    /// scales padded to 4 bytes, then the bf16 group offsets). `scales` is null for bf16.
    /// Returns a handle, or -1 when the host could not allocate it (the matrix stays on the CPU).
    fn gpu_upload(kind: u32, data: *const u8, data_len: usize, scales: *const u8, scales_len: usize, n_out: u32, n_in: u32) -> i32;
    /// Runs `n_jobs` matmuls over `x` ([t, n_in] f32); each job is a `JobDesc`. Returns 0 on success.
    fn gpu_matmul(x: *const f32, t: u32, n_in: u32, jobs: *const JobDesc, n_jobs: u32) -> i32;
    /// A k x k convolution of `x` (`[h*w, cin]` f32) with weight `handle` into `y`
    /// (`[oh*ow, n_out]`); see `ConvDesc`. Returns 0 on success.
    fn gpu_conv(x: *const f32, desc: *const ConvDesc) -> i32;
    /// Attention over per-head padded operands (see `AttnDesc`); writes `out`. 0 on success.
    fn gpu_attention(q: *const f32, k: *const f32, vt: *const f32, out: *mut f32, desc: *const AttnDesc) -> i32;
}

/// Attention for the host: `q` `[heads][tqp][dhp]`, `k` `[heads][tkp][dhp]`, `vt`
/// `[heads][dho][tkp]` (zero padded), `out` `[heads][tqp][dho]`.
#[repr(C)]
pub struct AttnDesc {
    heads: u32,
    tqp: u32,
    tkp: u32,
    dhp: u32,
    dho: u32,
    tk: u32,
    causal: u32,
    scale: f32,
}

/// Convolution geometry for the host (all u32; `y` is a module-memory pointer).
#[repr(C)]
pub struct ConvDesc {
    handle: u32,
    n_out: u32,
    n_in: u32,
    h: u32,
    w: u32,
    cin: u32,
    k: u32,
    stride: u32,
    oh: u32,
    ow: u32,
    y: u32,
    /// Input normalization: 0 none, 1 group, 2 layer; SiLU after it; channels per group;
    /// epsilon; gamma and beta pointers (`cin` floats each); number of statistics.
    mode: u32,
    silu: u32,
    cpg: u32,
    eps: f32,
    gamma: u32,
    beta: u32,
    n_stats: u32,
    /// Zero padding before the first row / column (k/2 for a "same" convolution).
    pad: u32,
    _pad: u32,
}

/// One matmul for the host: output `y` is [t, n_out] f32 in module memory.
#[repr(C)]
pub struct JobDesc {
    handle: u32,
    n_out: u32,
    y: u32,
    _pad: u32,
}

struct Handle(i32);

pub struct WebGpuBackend {
    calls: AtomicU64,
    flops: AtomicU64,
    micros: AtomicU64,
}

const MIN_T: usize = 16;
/// The round trip through the host (submit, wait for the readback promise, copy back) is around a
/// millisecond; the single-threaded CPU in the browser does ~40 GFLOP/s, so 1e8 FLOPs is the break-even.
const MIN_FLOPS: f64 = 1e8;

pub fn install() {
    if unsafe { gpu_available() } != 0 {
        crate::gpu::install(Some(Box::new(WebGpuBackend {
            calls: AtomicU64::new(0),
            flops: AtomicU64::new(0),
            micros: AtomicU64::new(0),
        })));
    } else {
        super::log("[webgpu: not available on this host (no adapter or no JSPI); using CPU]");
    }
}

impl Backend for WebGpuBackend {
    fn name(&self) -> &str {
        "webgpu"
    }
    fn wants(&self, n_out: usize, n_in: usize, t: usize) -> bool {
        t >= MIN_T && n_out.is_multiple_of(64) && n_in.is_multiple_of(32) && 2.0 * t as f64 * n_out as f64 * n_in as f64 >= MIN_FLOPS
    }
    fn upload(&self, w: &WeightRef, n_out: usize, n_in: usize) -> Option<DeviceWeight> {
        // Affine Q4: one auxiliary buffer, scales then offsets.
        let aux: Vec<u16> = match w {
            WeightRef::Q4(m) if m.offsets.is_some() => {
                let o = m.offsets.as_ref().unwrap();
                let mut v = Vec::with_capacity(m.scales.len() + 1 + o.v.len());
                v.extend_from_slice(&m.scales);
                if v.len() % 2 == 1 {
                    v.push(0);
                }
                v.extend_from_slice(&o.v);
                v
            }
            _ => Vec::new(),
        };
        let h = unsafe {
            match w {
                WeightRef::Gguf(_) => return None,
                WeightRef::Bf16(w) => gpu_upload(
                    0,
                    w.as_ptr() as *const u8,
                    w.len() * 2,
                    std::ptr::null(),
                    0,
                    n_out as u32,
                    n_in as u32,
                ),
                WeightRef::Q8(m) => gpu_upload(
                    1,
                    m.q.as_ptr() as *const u8,
                    m.q.len(),
                    m.scales.as_ptr() as *const u8,
                    m.scales.len() * 4,
                    n_out as u32,
                    n_in as u32,
                ),
                WeightRef::Q4(m) if !aux.is_empty() => gpu_upload(
                    4,
                    m.q.as_ptr(),
                    m.q.len(),
                    aux.as_ptr() as *const u8,
                    aux.len() * 2,
                    n_out as u32,
                    n_in as u32,
                ),
                WeightRef::Q4(m) => gpu_upload(
                    2,
                    m.q.as_ptr(),
                    m.q.len(),
                    m.scales.as_ptr() as *const u8,
                    m.scales.len() * 2,
                    n_out as u32,
                    n_in as u32,
                ),
            }
        };
        if h < 0 {
            return None;
        }
        Some(Box::new(Handle(h)))
    }
    fn conv2d(&self, x: &[f32], g: &crate::gpu::ConvGeom, norm: Option<&crate::gpu::NormSpec>, job: &mut Job) -> bool {
        use crate::gpu::NormKind;
        let Some(h) = job.dev.downcast_ref::<Handle>() else { return false };
        if matches!(job.w, WeightRef::Q4(m) if m.offsets.is_some()) {
            return false; // no affine convolution shader
        }
        if !g.cin.is_multiple_of(32) || !job.n_out.is_multiple_of(64) {
            return false;
        }
        if let Some(n) = norm
            && (n.gamma.len() != g.cin
                || n.beta.len() != g.cin
                || matches!(n.kind, NormKind::Group(gr) if gr == 0 || !g.cin.is_multiple_of(gr)))
        {
            return false;
        }
        let t0 = crate::clock::Instant::now();
        let d = ConvDesc {
            handle: h.0 as u32,
            n_out: job.n_out as u32,
            n_in: (g.k * g.k * g.cin) as u32,
            h: g.h as u32,
            w: g.w as u32,
            cin: g.cin as u32,
            k: g.k as u32,
            stride: g.stride as u32,
            oh: g.oh as u32,
            ow: g.ow as u32,
            y: job.y.as_mut_ptr() as u32,
            mode: match norm.map(|n| n.kind) {
                None => 0,
                Some(NormKind::Group(_)) => 1,
                Some(NormKind::Layer) => 2,
            },
            silu: norm.is_some_and(|n| n.silu) as u32,
            cpg: match norm.map(|n| n.kind) {
                Some(NormKind::Group(gr)) => (g.cin / gr) as u32,
                _ => 1,
            },
            eps: norm.map_or(0.0, |n| n.eps),
            gamma: norm.map_or(0, |n| n.gamma.as_ptr() as u32),
            beta: norm.map_or(0, |n| n.beta.as_ptr() as u32),
            n_stats: match norm.map(|n| n.kind) {
                Some(NormKind::Group(gr)) => gr as u32,
                Some(NormKind::Layer) => (g.h * g.w) as u32,
                None => 0,
            },
            pad: g.pad as u32,
            _pad: 0,
        };
        let rc = unsafe { gpu_conv(x.as_ptr(), &d) };
        if rc != 0 {
            super::log(&format!("[webgpu: conv failed ({rc}); falling back to CPU]"));
            return false;
        }
        let flops = 2.0 * (g.oh * g.ow) as f64 * job.n_out as f64 * (g.k * g.k * g.cin) as f64;
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.flops.fetch_add(flops as u64, Ordering::Relaxed);
        self.micros.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        true
    }
    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        tq: usize,
        tk: usize,
        heads: usize,
        dh: usize,
        causal: bool,
        out: &mut [f32],
    ) -> bool {
        let d = heads * dh;
        if q.len() < tq * d || k.len() < tk * d || v.len() < tk * d || out.len() < tq * d || tq == 0 || tk == 0 {
            return false;
        }
        let t0 = crate::clock::Instant::now();
        let (dhp, dho, tkp, tqp) = (
            dh.next_multiple_of(32),
            dh.next_multiple_of(64),
            tk.next_multiple_of(64),
            tq.next_multiple_of(64),
        );
        let mut qh = vec![0f32; heads * tqp * dhp];
        let mut kh = vec![0f32; heads * tkp * dhp];
        let mut vt = vec![0f32; heads * dho * tkp];
        for h in 0..heads {
            for i in 0..tq {
                qh[(h * tqp + i) * dhp..][..dh].copy_from_slice(&q[i * d + h * dh..][..dh]);
            }
            for j in 0..tk {
                kh[(h * tkp + j) * dhp..][..dh].copy_from_slice(&k[j * d + h * dh..][..dh]);
                for c in 0..dh {
                    vt[(h * dho + c) * tkp + j] = v[j * d + h * dh + c];
                }
            }
        }
        let mut o = vec![0f32; heads * tqp * dho];
        let desc = AttnDesc {
            heads: heads as u32,
            tqp: tqp as u32,
            tkp: tkp as u32,
            dhp: dhp as u32,
            dho: dho as u32,
            tk: tk as u32,
            causal: causal as u32,
            scale: 1.0 / (dh as f32).sqrt(),
        };
        let rc = unsafe { gpu_attention(qh.as_ptr(), kh.as_ptr(), vt.as_ptr(), o.as_mut_ptr(), &desc) };
        if rc != 0 {
            super::log(&format!("[webgpu: attention failed ({rc}); falling back to CPU]"));
            return false;
        }
        for h in 0..heads {
            for i in 0..tq {
                out[i * d + h * dh..][..dh].copy_from_slice(&o[(h * tqp + i) * dho..][..dh]);
            }
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.flops.fetch_add((4 * tq * tk * d) as u64, Ordering::Relaxed);
        self.micros.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        true
    }
    fn matmul(&self, x: &[f32], t: usize, n_in: usize, jobs: &mut [Job]) -> bool {
        let t0 = crate::clock::Instant::now();
        let descs: Vec<JobDesc> = jobs
            .iter_mut()
            .map(|j| JobDesc {
                handle: j.dev.downcast_ref::<Handle>().map(|h| h.0 as u32).unwrap_or(u32::MAX),
                n_out: j.n_out as u32,
                y: j.y.as_mut_ptr() as u32,
                _pad: 0,
            })
            .collect();
        if descs.iter().any(|d| d.handle == u32::MAX) {
            return false;
        }
        let rc = unsafe { gpu_matmul(x.as_ptr(), t as u32, n_in as u32, descs.as_ptr(), descs.len() as u32) };
        if rc != 0 {
            super::log(&format!("[webgpu: matmul failed ({rc}); falling back to CPU]"));
            return false;
        }
        let flops: f64 = jobs.iter().map(|j| 2.0 * t as f64 * j.n_out as f64 * n_in as f64).sum();
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.flops.fetch_add(flops as u64, Ordering::Relaxed);
        self.micros.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        true
    }
    fn stats(&self) -> Option<String> {
        let calls = self.calls.load(Ordering::Relaxed);
        if calls == 0 {
            return None;
        }
        let secs = self.micros.load(Ordering::Relaxed) as f64 / 1e6;
        Some(format!(
            "{calls} batches, {:.2} GFLOP in {:.3}s = {:.0} GFLOP/s incl. host round trips",
            self.flops.load(Ordering::Relaxed) as f64 / 1e9,
            secs,
            self.flops.load(Ordering::Relaxed) as f64 / 1e9 / secs
        ))
    }
}
