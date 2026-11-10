//! Decoding on CUDA ([`crate::gpu::decode`]): a graph compiles to a list of kernel launches over
//! one activation arena, run in order on the default stream; a step copies the tokens' inputs up,
//! launches the ops, and copies the hidden states, the logits and the new keys and values down.
//! The kernels (`cuda_decode.cu`) are ports of the Metal decode kernels.
//!
//! Multi-token plans (speculative verification, DSpark draft blocks) keep `tokens` rows of every
//! activation buffer. Their matvecs read each weight once for all the tokens, every token's sum
//! taken exactly as the single-token kernel takes it; the other ops run per token, op by op (so
//! recurrences stay sequential and a draft block's attention sees all its keys). A step of `t`
//! tokens therefore computes what `t` single-token steps would.
//!
//! The KV caches stay authoritative on the host: each session keeps device copies
//! ([`CudaSlot`]), appends what the host added since the last step and forgets what a rewind
//! dropped. Recurrent states (Gated DeltaNet) go up before and down after every step: the host
//! may reallocate them between steps (a session reset reuses the address), which pinning them in
//! place could not see. Multi-token plans keep the recurrent ops' inputs, so a rewind inside the
//! step replays them ([`DecodePlan::replay_recurrent`]).
//!
//! Declined (the model then decodes on the CPU): interventions, Qwen4-Exp's hyper-connection ops,
//! GGUF weights, and graphs whose weights do not all fit in VRAM (`OXINFER_GPU_PLAN_DEBUG=1`
//! says which op).

use super::{Cu, CudaBackend, CudaWeight, CudaWeightBox};
use crate::gpu::WeightRef;
use crate::gpu::decode::{DecodePlan, Ew, Graph, LinMode, Op, Slot, Step, View};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Most chunks a long context's attention splits into (`attn_split` in the kernels).
const ATTN_SPLIT: usize = 64;
/// Warps per matvec block, and rows per warp of the single-token matvec (`DMV_NR`).
const MV_ROWS: u32 = 4;
const MV_NR: u32 = 4;
/// Most tokens of a multi-token plan (the multi-token matvec's register tiles).
const MAX_TOKENS: usize = 16;

/// Pointers to the kernel arguments, for `cuLaunchKernel` (in one statement with the launch, so
/// temporaries outlive it).
macro_rules! args {
    ($($v:expr),* $(,)?) => { [$( &$v as *const _ as *mut c_void ),*] };
}

/// A weight matrix as the kernels take it (`DW`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Dw {
    q: u64,
    s: u64,
    o: u64,
    gpr: u32,
    gk: u32,
}

impl Dw {
    fn of(cw: &CudaWeight) -> Dw {
        Dw {
            q: cw.data,
            s: cw.scales,
            o: cw.offsets,
            gpr: cw.gpr,
            gk: cw.gk,
        }
    }
}

/// A device address and the bytes between consecutive tokens' rows (0: shared by all tokens).
/// While a plan is built, the address is an offset tagged with its region (see `A` and `C`).
#[derive(Clone, Copy, Default)]
struct P {
    a: u64,
    ts: u64,
}

impl P {
    fn at(self, j: usize) -> u64 {
        if self.a == 0 { 0 } else { self.a + self.ts * j as u64 }
    }
}

/// Tags of plan-relative offsets: the arena, the constants, and the replay inputs (by index);
/// device addresses (below 2^48) carry none.
const A: u64 = 1 << 62;
const C: u64 = 1 << 61;
const S: u64 = 1 << 60;

/// Kernel name suffix of a weight format.
fn fmt(w: &WeightRef) -> Option<&'static str> {
    Some(match w {
        WeightRef::Bf16(_) => "bf",
        WeightRef::Q8(_) => "q8",
        WeightRef::Q4(m) if m.offsets.is_some() => "q4a",
        WeightRef::Q4(_) => "q4",
        WeightRef::Gguf(_) => return None,
    })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NormP {
    n: u32,
    eps: f32,
    has_w: u32,
    add: u32,
    x_stride: u32,
    o_stride: u32,
    pre: f32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QkP {
    nh: u32,
    nkv: u32,
    hd: u32,
    half_rot: u32,
    q_stride: u32,
    k_off: u32,
    v_off: u32,
    interleaved: u32,
    has_qn: u32,
    has_kn: u32,
    v_norm: u32,
    v_from_k: u32,
    eps: f32,
    _p: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnP {
    nh: u32,
    nkv: u32,
    window: u32,
    has_gate: u32,
    g_stride: u32,
    block: u32,
    scale: f32,
    _q: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GdnP {
    nk: u32,
    nv: u32,
    key_dim: u32,
    flags: u32,
    eps: f32,
    qscale: f32,
    hdk: u32,
    hdv: u32,
}

type F = *mut c_void;

enum COp {
    Mv {
        f: F,
        /// The multi-token kernel (multi-token plans; up to 8 tokens per launch).
        fk: Option<F>,
        w: Dw,
        w2: Dw,
        x: P,
        y: P,
        n_out: u32,
        n_in: u32,
        mode: u32,
        gate: P,
    },
    Topk {
        f: F,
        logits: P,
        sel: P,
        n: u32,
        k: u32,
        scale: P,
    },
    Experts {
        pair: F,
        sum: F,
        tab_g: u64,
        tab_u: u64,
        tab_d: u64,
        x: P,
        act: P,
        y: P,
        sel: P,
        h: u32,
        inter: u32,
        k: u32,
        r2off: u32,
        gelu: u32,
        add: u32,
    },
    Norm {
        f: F,
        x: P,
        out: P,
        w: P,
        p: NormP,
        rows: u32,
        threads: u32,
    },
    AddNorm {
        f: F,
        y: P,
        x: P,
        out: P,
        wy: P,
        wn: P,
        n: u32,
        eps: f32,
        scale: f32,
        has_next: u32,
        threads: u32,
    },
    QkRope {
        f: F,
        qkv: P,
        qo: P,
        kv: usize,
        kout: P,
        vout: P,
        qn: P,
        kn: P,
        cs: P,
        p: QkP,
    },
    Attn {
        f: F,
        combine: F,
        q: P,
        out: P,
        kv: usize,
        gate: P,
        p: AttnP,
        grp: u32,
        hd: u32,
    },
    GdnConv {
        f: F,
        x: P,
        out: P,
        state: usize,
        w: P,
        dim: u32,
        kc: u32,
        /// Multi-token plans: where the inputs are kept for a replay (consecutive rows).
        save: P,
    },
    GdnRule {
        f: F,
        c: P,
        z: P,
        b: P,
        a: P,
        out: P,
        state: usize,
        cst: P,
        p: GdnP,
        threads: u32,
        /// Kept c, z, b, a rows (multi-token plans) and their lengths.
        save: [P; 4],
        lens: [u32; 4],
    },
    Glu {
        f: F,
        g: P,
        u: P,
        out: P,
        n: u32,
        gelu: u32,
    },
    Ew {
        f: F,
        a: P,
        b: P,
        out: P,
        n: u32,
        kind: u32,
        p: [f32; 3],
    },
}

impl COp {
    /// Every address of the op, to resolve the region tags once the arena exists.
    fn ptrs(&mut self) -> Vec<&mut u64> {
        match self {
            COp::Mv { w, x, y, gate, .. } => vec![&mut w.q, &mut x.a, &mut y.a, &mut gate.a],
            COp::Topk { logits, sel, scale, .. } => vec![&mut logits.a, &mut sel.a, &mut scale.a],
            COp::Experts { x, act, y, sel, .. } => vec![&mut x.a, &mut act.a, &mut y.a, &mut sel.a],
            COp::Norm { x, out, w, .. } => vec![&mut x.a, &mut out.a, &mut w.a],
            COp::AddNorm { y, x, out, wy, wn, .. } => vec![&mut y.a, &mut x.a, &mut out.a, &mut wy.a, &mut wn.a],
            COp::QkRope {
                qkv,
                qo,
                kout,
                vout,
                qn,
                kn,
                cs,
                ..
            } => vec![&mut qkv.a, &mut qo.a, &mut kout.a, &mut vout.a, &mut qn.a, &mut kn.a, &mut cs.a],
            COp::Attn { q, out, gate, .. } => vec![&mut q.a, &mut out.a, &mut gate.a],
            COp::GdnConv { x, out, w, save, .. } => vec![&mut x.a, &mut out.a, &mut w.a, &mut save.a],
            COp::GdnRule {
                c,
                z,
                b,
                a,
                out,
                cst,
                save,
                ..
            } => {
                let mut v = vec![&mut c.a, &mut z.a, &mut b.a, &mut a.a, &mut out.a, &mut cst.a];
                v.extend(save.iter_mut().map(|s| &mut s.a));
                v
            }
            COp::Glu { g, u, out, .. } => vec![&mut g.a, &mut u.a, &mut out.a],
            COp::Ew { a, b, out, .. } => vec![&mut a.a, &mut b.a, &mut out.a],
        }
    }
}

pub(super) struct CudaPlan<'b> {
    be: &'b CudaBackend,
    id: u64,
    /// The KV caches' and recurrent states' shapes: plans of one model share a session's device
    /// copies (switching between the single- and multi-token plans keeps them).
    layout: u64,
    /// Most tokens of a step.
    tokens: usize,
    /// Sessions share the arena: one step at a time.
    lock: Mutex<()>,
    /// Device allocations owned by the plan (arena, constants, expert tables).
    owned: Vec<u64>,
    /// In the arena: the steps' parameters (per token: position, first position, tokens, index),
    /// the RoPE rows (`tokens` rows reserved per table), the new keys and values (per cache:
    /// `tokens` key rows, then `tokens` value rows) and the split-attention partials.
    sp: u64,
    cs: u64,
    kvout: (u64, usize),
    part: u64,
    copy_rows: F,
    ops: Vec<COp>,
    head_from: usize,
    /// (address, floats per row, token stride in floats).
    input: (u64, usize, usize),
    input2: Option<(u64, usize, usize)>,
    hidden: (u64, usize, usize),
    logits: Option<(u64, usize, usize)>,
    capture: Option<(u64, usize, usize)>,
    rope_halves: Vec<usize>,
    kv: Vec<(usize, usize)>,
    states: Vec<usize>,
    /// Pinned host memory for the recurrent states on their way down (copies from device memory
    /// to pageable memory run at a fraction of the bus speed), and its size in floats.
    pinned: (*mut f32, usize),
    /// A stream for those copies and an event per state: each state goes down as soon as the
    /// last op writing it has run, under the remaining layers' kernels.
    copy_stream: *mut c_void,
    events: Vec<*mut c_void>,
    /// Per op: the states it writes last.
    last_writer: Vec<Vec<usize>>,
}
unsafe impl Send for CudaPlan<'_> {}
unsafe impl Sync for CudaPlan<'_> {}

impl Drop for CudaPlan<'_> {
    fn drop(&mut self) {
        if self.be.bind() {
            for &p in &self.owned {
                unsafe { self.be.cu.cuMemFree_v2(p) };
            }
            if !self.pinned.0.is_null() {
                unsafe { self.be.cu.cuMemFreeHost(self.pinned.0 as *mut c_void) };
            }
        }
    }
}

impl CudaPlan<'_> {
    /// Whether states go down through the pinned buffer on the copy stream.
    fn async_states(&self) -> bool {
        !self.pinned.0.is_null() && !self.copy_stream.is_null() && self.events.len() == self.states.len()
    }

    /// Queues state `i`'s copy into the pinned buffer, after what the default stream has queued.
    fn queue_state(&self, dev: &[(u64, usize)], i: usize) -> bool {
        let cu = &self.be.cu;
        let o: usize = dev[..i].iter().map(|d| d.1).sum();
        unsafe {
            cu.cuEventRecord(self.events[i], std::ptr::null_mut()) == 0
                && cu.cuStreamWaitEvent(self.copy_stream, self.events[i], 0) == 0
                && cu.cuMemcpyDtoHAsync_v2(self.pinned.0.add(o) as *mut c_void, dev[i].0, dev[i].1 * 4, self.copy_stream) == 0
        }
    }

    /// Copies the device states into the host's: through the pinned buffer (one transfer, then a
    /// parallel copy) when there is one; `queued`: the transfers are on the copy stream already.
    fn states_down(&self, dev: &[(u64, usize)], host: &mut [&mut [f32]], queued: bool) -> bool {
        let cu = &self.be.cu;
        let total: usize = dev.iter().map(|d| d.1).sum();
        // Copies queued earlier must land before the pinned buffer is used either way.
        if !self.copy_stream.is_null() && unsafe { cu.cuStreamSynchronize(self.copy_stream) } != 0 {
            return false;
        }
        if !queued && (self.pinned.0.is_null() || self.pinned.1 < total) {
            return dev
                .iter()
                .zip(host.iter_mut())
                .all(|(&(p, _), st)| unsafe { cu.cuMemcpyDtoH_v2(st.as_mut_ptr() as *mut c_void, p, st.len() * 4) == 0 });
        }
        let mut jobs: Vec<(usize, usize, usize)> = Vec::new();
        let mut o = 0;
        for (&(p, n), st) in dev.iter().zip(host.iter_mut()) {
            // SAFETY: the pinned buffer holds `total` floats.
            let dst = unsafe { self.pinned.0.add(o) };
            if !queued && unsafe { cu.cuMemcpyDtoH_v2(dst as *mut c_void, p, n * 4) } != 0 {
                return false;
            }
            // 1 MB pieces for the parallel copy.
            for s in (0..n).step_by(1 << 18) {
                let len = (1 << 18).min(n - s);
                jobs.push((dst as usize + s * 4, st.as_mut_ptr() as usize + s * 4, len));
            }
            o += n;
        }
        let t0 = std::time::Instant::now();
        crate::pool::run(jobs.len(), &|i| {
            let (s, d, n) = jobs[i];
            // SAFETY: disjoint pieces of the pinned buffer into disjoint pieces of the states.
            unsafe { std::ptr::copy_nonoverlapping(s as *const f32, d as *mut f32, n) };
        });
        if trace() {
            eprintln!(
                "[cuda decode: host copy of {} MB in {:.0} us]",
                (total * 4) >> 20,
                t0.elapsed().as_secs_f64() * 1e6
            );
        }
        true
    }
}

/// A session's device copy of one KV cache.
struct Mirror {
    k: u64,
    v: u64,
    cap: usize,
    synced: usize,
}

/// Per-session device state: the KV copies and the recurrent states' device buffers (valid: they
/// hold what the last step left, which went to the host too); and the plan and tokens of the last
/// step (for a replay).
struct CudaSlot {
    layout: u64,
    kv: Vec<Mirror>,
    states: Vec<(u64, usize)>,
    states_valid: bool,
    last: (u64, usize),
    cu: Arc<Cu>,
    ctx: *mut c_void,
}
unsafe impl Send for CudaSlot {}
unsafe impl Sync for CudaSlot {}

impl Drop for CudaSlot {
    fn drop(&mut self) {
        unsafe {
            if self.cu.cuCtxSetCurrent(self.ctx) != 0 {
                return;
            }
            for m in &self.kv {
                for p in [m.k, m.v] {
                    if p != 0 {
                        self.cu.cuMemFree_v2(p);
                    }
                }
            }
            for &(p, _) in &self.states {
                if p != 0 {
                    self.cu.cuMemFree_v2(p);
                }
            }
        }
    }
}

fn op_name(op: &Op) -> &'static str {
    match op {
        Op::Norm { .. } => "norm",
        Op::Linear { .. } => "linear",
        Op::QkRope { .. } => "qk-rope",
        Op::AddNorm { .. } => "add-norm",
        Op::Ew { .. } => "elementwise",
        Op::Attend { .. } => "attention",
        Op::GdnConv { .. } => "gdn-conv",
        Op::GdnRule(_) => "gdn-rule",
        Op::Route { .. } => "route",
        Op::Experts(_) => "experts (VRAM?)",
        Op::Steer { .. } => "steer",
        Op::NeuronEdit { .. } => "neuron-edit",
        Op::Glu { .. } => "glu",
        Op::HcPre { .. } => "hc-pre",
        Op::HcPost { .. } => "hc-post",
        Op::PleGate { .. } => "ple-gate",
        Op::QsaKey { .. } => "qsa-key",
        Op::QsaSelect { .. } => "qsa-select",
        Op::AttendSel { .. } => "qsa-attention",
    }
}

/// `OXINFER_CUDA_DECODE_TRACE=1`: times each step's parts (synchronizing after the kernels).
fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OXINFER_CUDA_DECODE_TRACE").is_some())
}

static PLAN_ID: AtomicU64 = AtomicU64::new(1);

/// `OXINFER_GPU_PLAN_DEBUG`: says why a plan or a step was declined.
fn debug(why: &str) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("OXINFER_GPU_PLAN_DEBUG").is_some()) {
        eprintln!("[cuda decode: {why}]");
    }
}

impl CudaBackend {
    /// Device allocation of `bytes` (zeroed), or None.
    fn alloc_zeroed(&self, bytes: usize) -> Option<u64> {
        let mut d = 0u64;
        if unsafe { self.cu.cuMemAlloc_v2(&mut d, bytes.max(256)) } != 0 {
            return None;
        }
        unsafe { self.cu.cuMemsetD8_v2(d, 0, bytes.max(256)) };
        Some(d)
    }

    /// Compiles `g` for steps of up to `tokens` tokens (1: the single-token plan).
    pub(super) fn build_plan(&self, g: &Graph, tokens: usize) -> Option<CudaPlan<'_>> {
        if !self.bind() || g.params.is_some() || tokens == 0 || tokens > MAX_TOKENS {
            return None;
        }
        let mt = tokens > 1;
        // The arena: every buffer 256-byte aligned; in a multi-token plan, `tokens` rows of `tsz`
        // floats (256-byte aligned too).
        let tsz: Vec<usize> = g.bufs.iter().map(|&n| if mt { n.next_multiple_of(64) } else { n }).collect();
        let mut total = 0usize;
        let mut region = |bytes: usize| {
            let o = total;
            total += bytes.next_multiple_of(256);
            o
        };
        let boff: Vec<usize> = tsz.iter().map(|&n| region(n * tokens * 4)).collect();
        let sp_off = region(16 * tokens);
        let cs_off = region(tokens * 2 * g.rope_halves.iter().sum::<usize>() * 4);
        let kv_rows: Vec<usize> = g.kv.iter().map(|&(n, hd)| n * hd).collect();
        let kvout_len = 2 * tokens * kv_rows.iter().sum::<usize>();
        let kvout_off = region(kvout_len * 4);
        let mut part_floats = 0usize;
        let mut consts: Vec<f32> = Vec::new();
        let mut cst = |v: &[f32]| -> P {
            consts.resize(consts.len().next_multiple_of(16), 0.0);
            let o = consts.len() * 4;
            consts.extend_from_slice(v);
            P { a: C | o as u64, ts: 0 }
        };
        let byte = |v: View| P {
            a: A | (boff[v.buf] + v.off * 4) as u64,
            ts: (tsz[v.buf] * 4) as u64,
        };
        // Replay inputs of the recurrent ops (multi-token plans): consecutive rows after the rest.
        let mut saves: Vec<usize> = Vec::new();
        let mut save = |n: usize| -> P {
            saves.push(n * tokens);
            P {
                a: S | (saves.len() - 1) as u64,
                ts: (n * 4) as u64,
            }
        };
        let mut kvout_at = Vec::with_capacity(kv_rows.len());
        let mut o = 0usize;
        for &r in &kv_rows {
            let ts = (r * 4) as u64;
            kvout_at.push((
                P {
                    a: A | (kvout_off + o * 4) as u64,
                    ts,
                },
                P {
                    a: A | (kvout_off + (o + tokens * r) * 4) as u64,
                    ts,
                },
            ));
            o += 2 * tokens * r;
        }
        let cs_table = |r: usize| P {
            a: A | (cs_off + 4 * tokens * 2 * g.rope_halves[..r].iter().sum::<usize>()) as u64,
            ts: (2 * g.rope_halves[r] * 4) as u64,
        };
        let mut owned: Vec<u64> = Vec::new();
        let free = |owned: &[u64]| {
            for &p in owned {
                unsafe { self.cu.cuMemFree_v2(p) };
            }
        };
        let mut ops: Vec<COp> = Vec::new();
        let mut head_from = usize::MAX;
        let mut failed = String::new();
        let built: Option<()> = (|| {
            for (i, op) in g.ops.iter().enumerate() {
                failed = format!("{i} ({})", op_name(op));
                if i == g.head_from {
                    head_from = ops.len();
                }
                match *op {
                    Op::Norm {
                        x,
                        out,
                        ref w,
                        pre,
                        n,
                        rows,
                        x_stride,
                        out_stride,
                        eps,
                        add,
                    } => ops.push(COp::Norm {
                        f: self.func("dnorm")?,
                        x: byte(x),
                        out: byte(out),
                        w: w.as_deref().map_or(P::default(), &mut cst),
                        p: NormP {
                            n: n as u32,
                            eps,
                            has_w: w.is_some() as u32,
                            add: add as u32,
                            x_stride: x_stride as u32,
                            o_stride: out_stride as u32,
                            pre,
                            _p: 0,
                        },
                        rows: rows as u32,
                        threads: n.next_multiple_of(32).min(1024) as u32,
                    }),
                    Op::Linear {
                        ref w,
                        dev,
                        ref w2,
                        n_out,
                        n_in,
                        x,
                        y,
                        mode,
                    } => {
                        let name = fmt(w)?;
                        if !n_in.is_multiple_of(32) {
                            return None;
                        }
                        let cw = &dev.downcast_ref::<CudaWeightBox>()?.0;
                        let cw2 = match w2 {
                            Some((w2, d2)) => {
                                if fmt(w2)? != name || !crate::gpu::decode::pairable(w, w2) {
                                    return None;
                                }
                                Some(Dw::of(&d2.downcast_ref::<CudaWeightBox>()?.0))
                            }
                            None => None,
                        };
                        let (mode, gate) = match mode {
                            LinMode::Store => (0, P::default()),
                            LinMode::Add => (1, P::default()),
                            LinMode::SiluPair => (2, P::default()),
                            LinMode::GeluPair => (3, P::default()),
                            LinMode::AddGated(v) => (4, byte(v)),
                        };
                        if cw2.is_some() != (mode == 2 || mode == 3) {
                            return None;
                        }
                        ops.push(COp::Mv {
                            f: self.func(&format!("dmv_{name}"))?,
                            fk: if mt { Some(self.func(&format!("dmvk_{name}"))?) } else { None },
                            w: Dw::of(cw),
                            w2: cw2.unwrap_or_default(),
                            x: byte(x),
                            y: byte(y),
                            n_out: n_out as u32,
                            n_in: n_in as u32,
                            mode,
                            gate,
                        });
                    }
                    Op::QkRope {
                        qkv,
                        q_out,
                        kv,
                        nh,
                        nkv,
                        hd,
                        half_rot,
                        q_stride,
                        k_off,
                        v_off,
                        interleaved,
                        q_norm,
                        k_norm,
                        eps,
                        rope,
                        v_norm,
                        v_from_k,
                    } => {
                        if hd > 512 || !hd.is_multiple_of(32) || 2 * half_rot > hd || g.rope_halves.get(rope) != Some(&half_rot) {
                            return None;
                        }
                        if nkv > 0 && g.kv.get(kv) != Some(&(nkv, hd)) {
                            return None;
                        }
                        let (kout, vout) = kvout_at.get(kv).copied().unwrap_or_default();
                        ops.push(COp::QkRope {
                            f: self.func("dqk_rope")?,
                            qkv: byte(qkv),
                            qo: byte(q_out),
                            kv,
                            kout,
                            vout,
                            qn: q_norm.map_or(P::default(), &mut cst),
                            kn: k_norm.map_or(P::default(), &mut cst),
                            cs: cs_table(rope),
                            p: QkP {
                                nh: nh as u32,
                                nkv: nkv as u32,
                                hd: hd as u32,
                                half_rot: half_rot as u32,
                                q_stride: q_stride as u32,
                                k_off: k_off as u32,
                                v_off: v_off as u32,
                                interleaved: interleaved as u32,
                                has_qn: q_norm.is_some() as u32,
                                has_kn: k_norm.is_some() as u32,
                                v_norm: v_norm as u32,
                                v_from_k: v_from_k as u32,
                                eps,
                                _p: [0; 3],
                            },
                        });
                    }
                    Op::AddNorm {
                        y,
                        x,
                        ref wy,
                        scale,
                        next,
                        n,
                        eps,
                    } => {
                        if wy.len() != n || next.is_some_and(|(_, w)| w.len() != n) {
                            return None;
                        }
                        ops.push(COp::AddNorm {
                            f: self.func("dadd_norm")?,
                            y: byte(y),
                            x: byte(x),
                            out: next.map_or(byte(x), |(o, _)| byte(o)),
                            wy: cst(wy),
                            wn: next.map_or(P::default(), |(_, w)| cst(w)),
                            n: n as u32,
                            eps,
                            scale,
                            has_next: next.is_some() as u32,
                            threads: n.next_multiple_of(32).min(1024) as u32,
                        });
                    }
                    Op::Ew { kind, a, b, out, n } => {
                        let (k, p) = match kind {
                            Ew::Scale(s) => (0, [s, 0.0, 0.0]),
                            Ew::Softcap(c) => (1, [c, 0.0, 0.0]),
                            Ew::Combine { sa, sb, s } => (2, [sa, sb, s]),
                            Ew::SiluScale(s) => (3, [s, 0.0, 0.0]),
                        };
                        if k == 2 && b.is_none() {
                            return None;
                        }
                        ops.push(COp::Ew {
                            f: self.func("dew")?,
                            a: byte(a),
                            b: byte(b.unwrap_or(a)),
                            out: byte(out),
                            n: n as u32,
                            kind: k,
                            p,
                        });
                    }
                    Op::Glu { g: gv, u, out, n, gelu } => ops.push(COp::Glu {
                        f: self.func("dglu")?,
                        g: byte(gv),
                        u: byte(u),
                        out: byte(out),
                        n: n as u32,
                        gelu: gelu as u32,
                    }),
                    Op::Attend {
                        q,
                        out,
                        kv,
                        nh,
                        nkv,
                        hd,
                        scale,
                        window,
                        gate,
                        block,
                    } => {
                        let d = hd / 32;
                        if (block.is_some() && !mt)
                            || !hd.is_multiple_of(32)
                            || !matches!(d, 2 | 4 | 8 | 16)
                            || nkv == 0
                            || !nh.is_multiple_of(nkv)
                        {
                            return None;
                        }
                        if g.kv.get(kv) != Some(&(nkv, hd)) {
                            return None;
                        }
                        part_floats = part_floats.max(nh * ATTN_SPLIT * (hd + 2));
                        let rep = nh / nkv;
                        let grp = [8, 4, 2, 1]
                            .into_iter()
                            .find(|&g| rep.is_multiple_of(g) && g * d <= 16)
                            .unwrap_or(1);
                        let (window, block) = match block {
                            Some((causal, w)) => (w, 1 | (causal as u32) << 1),
                            None => (window.unwrap_or(0), 0),
                        };
                        ops.push(COp::Attn {
                            f: self.func(&format!("dattn_{d}_{grp}"))?,
                            combine: self.func("dattn_combine")?,
                            q: byte(q),
                            out: byte(out),
                            kv,
                            gate: gate.map_or(P::default(), |(v, _)| byte(v)),
                            p: AttnP {
                                nh: nh as u32,
                                nkv: nkv as u32,
                                window: window as u32,
                                has_gate: gate.is_some() as u32,
                                g_stride: gate.map_or(0, |(_, s)| s as u32),
                                block,
                                scale,
                                _q: 0,
                            },
                            grp: grp as u32,
                            hd: hd as u32,
                        });
                    }
                    Op::GdnConv { x, out, state, w, dim, kc } => {
                        if kc < 2 || w.len() != dim * kc || g.states.get(state) != Some(&((kc - 1) * dim)) {
                            return None;
                        }
                        ops.push(COp::GdnConv {
                            f: self.func("dgdn_conv")?,
                            x: byte(x),
                            out: byte(out),
                            state,
                            w: cst(w),
                            dim: dim as u32,
                            kc: kc as u32,
                            save: if mt { save(dim) } else { P::default() },
                        });
                    }
                    Op::GdnRule(ref r) => {
                        let ok = r.hdk.is_multiple_of(32)
                            && r.hdk <= 256
                            && r.hdv.is_multiple_of(32)
                            && r.hdk / 32 * r.hdv <= 1024
                            && r.nk > 0
                            && r.nv.is_multiple_of(r.nk)
                            && g.states.get(r.state) == Some(&(r.nv * r.hdk * r.hdv))
                            && r.norm_w.len() == r.hdv;
                        if !ok {
                            return None;
                        }
                        let mut v = r.decay_coef.to_vec();
                        v.extend_from_slice(r.dt_bias);
                        v.extend_from_slice(r.norm_w);
                        let key_dim = r.nk * r.hdk;
                        let lens = [2 * key_dim + r.nv * r.hdv, r.nv * r.hdv, r.nv, r.nv];
                        ops.push(COp::GdnRule {
                            f: self.func("dgdn_rule")?,
                            c: byte(r.c),
                            z: byte(r.z),
                            b: byte(r.b),
                            a: byte(r.a),
                            out: byte(r.out),
                            state: r.state,
                            cst: cst(&v),
                            p: GdnP {
                                nk: r.nk as u32,
                                nv: r.nv as u32,
                                key_dim: key_dim as u32,
                                flags: r.tiled as u32 | (r.sigmoid_gate as u32) << 1,
                                eps: r.eps,
                                qscale: (r.hdk as f32).powf(-0.5),
                                hdk: r.hdk as u32,
                                hdv: r.hdv as u32,
                            },
                            threads: (r.hdk / 32 * r.hdv) as u32,
                            save: if mt { lens.map(&mut save) } else { [P::default(); 4] },
                            lens: lens.map(|n| n as u32),
                        });
                    }
                    Op::Route {
                        x,
                        router,
                        gate,
                        n_exp,
                        h,
                        k,
                        logits,
                        sel,
                        expert_scale,
                    } => {
                        if router.len() != n_exp * h
                            || gate.is_some_and(|g| g.len() != h)
                            || expert_scale.is_some_and(|s| s.len() != n_exp)
                            || n_exp > 1024
                            || k == 0
                            || k > 64
                            || k > n_exp
                            || !h.is_multiple_of(8)
                        {
                            return None;
                        }
                        let mut w = router.to_vec();
                        w.extend_from_slice(gate.unwrap_or(&[]));
                        let rows = n_exp + gate.is_some() as usize;
                        ops.push(COp::Mv {
                            f: self.func("dmv_f32")?,
                            fk: None,
                            w: Dw {
                                q: cst(&w).a,
                                ..Dw::default()
                            },
                            w2: Dw::default(),
                            x: byte(x),
                            y: byte(logits),
                            n_out: rows as u32,
                            n_in: h as u32,
                            mode: 0,
                            gate: P::default(),
                        });
                        ops.push(COp::Topk {
                            f: self.func("dtopk")?,
                            logits: byte(logits),
                            sel: byte(sel),
                            n: n_exp as u32,
                            k: k as u32,
                            scale: expert_scale.map_or(P::default(), &mut cst),
                        });
                    }
                    Op::Experts(ref ex) => {
                        if ex.edits.is_some() || !ex.h.is_multiple_of(32) || !ex.inter.is_multiple_of(32) || ex.gate.is_empty() {
                            return None;
                        }
                        let n = ex.gate.len();
                        if ex.down.len() != n || ex.up.as_ref().is_some_and(|u| u.len() != n) {
                            return None;
                        }
                        // One format per projection (the gate's and the up's alike: one kernel).
                        let one = |ws: &[WeightRef]| -> Option<&'static str> {
                            let f = fmt(&ws[0])?;
                            ws.iter()
                                .all(|w| fmt(w) == Some(f) && crate::gpu::decode::pairable(&ws[0], w))
                                .then_some(f)
                        };
                        let (fg, fd) = (one(&ex.gate)?, one(&ex.down)?);
                        if ex
                            .up
                            .as_ref()
                            .is_some_and(|u| one(u) != Some(fg) || !crate::gpu::decode::pairable(&ex.gate[0], &u[0]))
                        {
                            return None;
                        }
                        let mut groups: Vec<&[WeightRef]> = vec![&ex.gate, &ex.down];
                        if let Some(u) = &ex.up {
                            groups.push(u);
                        }
                        let all = self.upload_all(&groups)?;
                        let mut tab = |ws: &[CudaWeight]| -> Option<u64> {
                            let t: Vec<Dw> = ws.iter().map(Dw::of).collect();
                            let p = self.upload_bytes(t.as_ptr() as *const u8, std::mem::size_of_val(&t[..]))?;
                            owned.push(p);
                            Some(p)
                        };
                        let tab_g = tab(&all[0])?;
                        let tab_d = tab(&all[1])?;
                        let tab_u = match all.get(2) {
                            Some(u) => tab(u)?,
                            None => tab_g,
                        };
                        ops.push(COp::Experts {
                            pair: self.func(&format!("dmvx_{fg}_pair"))?,
                            sum: self.func(&format!("dmvx_{fd}_sum"))?,
                            tab_g,
                            tab_u,
                            tab_d,
                            x: byte(ex.x),
                            act: byte(ex.act),
                            y: byte(ex.y),
                            sel: byte(ex.sel),
                            h: ex.h as u32,
                            inter: ex.inter as u32,
                            k: ex.k as u32,
                            r2off: if ex.up.is_some() { 0 } else { ex.inter as u32 },
                            gelu: ex.gelu as u32,
                            add: ex.add as u32,
                        });
                    }
                    Op::Steer { .. } | Op::NeuronEdit { .. } | Op::HcPre { .. } | Op::HcPost { .. } | Op::PleGate { .. } => return None,
                    Op::QsaKey { .. } | Op::QsaSelect { .. } | Op::AttendSel { .. } => return None,
                }
            }
            Some(())
        })();
        if built.is_none() {
            debug(&format!(
                "plan of {tokens} token(s) declined at graph op {failed} of {}",
                g.ops.len()
            ));
            free(&owned);
            return None;
        }
        if head_from == usize::MAX {
            head_from = ops.len();
        }
        let part_off = region(part_floats.max(1) * 4);
        let save_off: Vec<usize> = saves.iter().map(|&n| region(n * 4)).collect();
        let Some(arena) = self.alloc_zeroed(total) else {
            debug("no VRAM for the activation arena");
            free(&owned);
            return None;
        };
        owned.push(arena);
        consts.push(0.0);
        let Some(cbuf) = self.upload_bytes(consts.as_ptr() as *const u8, consts.len() * 4) else {
            free(&owned);
            return None;
        };
        owned.push(cbuf);
        for op in ops.iter_mut() {
            for p in op.ptrs() {
                if *p & A != 0 {
                    *p = arena + (*p & !A);
                } else if *p & C != 0 {
                    *p = cbuf + (*p & !C);
                } else if *p & S != 0 {
                    *p = arena + save_off[(*p & !S) as usize] as u64;
                }
            }
        }
        let last_writer: Vec<Vec<usize>> = {
            let mut last = vec![usize::MAX; g.states.len()];
            for (i, op) in ops.iter().enumerate() {
                if let COp::GdnConv { state, .. } | COp::GdnRule { state, .. } = op {
                    last[*state] = i;
                }
            }
            let mut v = vec![Vec::new(); ops.len()];
            for (st, &i) in last.iter().enumerate() {
                if i != usize::MAX {
                    v[i].push(st);
                }
            }
            v
        };
        let at = |v: View| arena + (boff[v.buf] + v.off * 4) as u64;
        let layout =
            g.kv.iter()
                .flat_map(|&(n, hd)| [n, hd])
                .chain([usize::MAX])
                .chain(g.states.iter().copied())
                .fold(0xcbf29ce484222325u64, |h, v| (h ^ v as u64).wrapping_mul(0x100000001b3));
        Some(CudaPlan {
            be: self,
            id: PLAN_ID.fetch_add(1, Ordering::Relaxed),
            layout,
            tokens,
            lock: Mutex::new(()),
            owned,
            sp: arena + sp_off as u64,
            cs: arena + cs_off as u64,
            kvout: (arena + kvout_off as u64, kvout_len),
            part: arena + part_off as u64,
            copy_rows: self.func("dcopy_rows")?,
            ops,
            head_from,
            input: (at(g.input), if g.n_input == 0 { g.n_hidden } else { g.n_input }, tsz[g.input.buf]),
            input2: g.input2.map(|(v, n)| (at(v), n, tsz[v.buf])),
            hidden: (at(g.hidden), g.n_hidden, tsz[g.hidden.buf]),
            logits: g.logits.map(|(v, n)| (at(v), n, tsz[v.buf])),
            capture: g.capture.map(|(v, n)| (at(v), n, tsz[v.buf])),
            rope_halves: g.rope_halves.clone(),
            kv: g.kv.clone(),
            states: g.states.clone(),
            copy_stream: {
                let mut st: *mut c_void = std::ptr::null_mut();
                // CU_STREAM_NON_BLOCKING: no implicit ordering with the default stream.
                if !g.states.is_empty() && unsafe { self.cu.cuStreamCreate(&mut st, 1) } != 0 {
                    st = std::ptr::null_mut();
                }
                st
            },
            events: g
                .states
                .iter()
                .map_while(|_| {
                    let mut e: *mut c_void = std::ptr::null_mut();
                    // CU_EVENT_DISABLE_TIMING
                    (unsafe { self.cu.cuEventCreate(&mut e, 2) } == 0).then_some(e)
                })
                .collect(),
            last_writer,
            pinned: {
                let n: usize = g.states.iter().sum();
                let mut p: *mut c_void = std::ptr::null_mut();
                if n > 0 && unsafe { self.cu.cuMemAllocHost_v2(&mut p, n * 4) } != 0 {
                    p = std::ptr::null_mut();
                }
                (p as *mut f32, if p.is_null() { 0 } else { n })
            },
        })
    }
}

impl CudaPlan<'_> {
    /// Brings the device KV copies in line with the host caches (`pos` positions each) with room
    /// for `t` more.
    fn sync_kv(&self, slot: &mut CudaSlot, kv: &mut [&mut crate::model::LayerCache], pos: usize, t: usize) -> bool {
        let cu = &self.be.cu;
        for (i, (m, c)) in slot.kv.iter_mut().zip(kv.iter_mut()).enumerate() {
            let row = self.kv[i].0 * self.kv[i].1;
            if c.len != pos || c.n_kv * c.hd != row {
                return false;
            }
            // A truncation (rewind) since the last step invalidates the copy from there on.
            m.synced = m.synced.min(c.len).min(c.kept);
            if m.cap < pos + t {
                let cap = (pos + t).next_power_of_two().max(256);
                let bytes = cap * row * 4;
                let (Some(k), Some(v)) = (self.be.alloc_zeroed(bytes), self.be.alloc_zeroed(bytes)) else {
                    return false;
                };
                unsafe {
                    if m.synced > 0 {
                        cu.cuMemcpyDtoD_v2(k, m.k, m.synced * row * 4);
                        cu.cuMemcpyDtoD_v2(v, m.v, m.synced * row * 4);
                    }
                    for p in [m.k, m.v] {
                        if p != 0 {
                            cu.cuMemFree_v2(p);
                        }
                    }
                }
                (m.k, m.v, m.cap) = (k, v, cap);
            }
            if m.synced < c.len {
                let (hk, hv) = c.data();
                let r = m.synced * row..c.len * row;
                let off = (r.start * 4) as u64;
                if !self.be.htod(m.k + off, &hk[r.clone()]) || !self.be.htod(m.v + off, &hv[r]) {
                    return false;
                }
                m.synced = c.len;
            }
            c.kept = c.len;
        }
        true
    }

    /// The session's device state for this plan (made fresh for another plan's or none), with its
    /// recurrent states' buffers.
    fn slot<'s>(&self, slot: &'s mut Slot) -> Option<&'s mut CudaSlot> {
        let be = self.be;
        if !matches!(slot.0.as_ref().and_then(|b| b.downcast_ref::<CudaSlot>()), Some(m) if m.layout == self.layout) {
            slot.0 = Some(Box::new(CudaSlot {
                layout: self.layout,
                kv: (0..self.kv.len())
                    .map(|_| Mirror {
                        k: 0,
                        v: 0,
                        cap: 0,
                        synced: 0,
                    })
                    .collect(),
                states: Vec::new(),
                states_valid: false,
                last: (0, 0),
                cu: be.cu.clone(),
                ctx: be.ctx,
            }));
        }
        let ms = slot.0.as_mut()?.downcast_mut::<CudaSlot>()?;
        if ms.states.len() != self.states.len() {
            ms.states = self.states.iter().map(|&n| (be.alloc_zeroed(n * 4).unwrap_or(0), n)).collect();
        }
        if ms.states.iter().any(|&(p, _)| p == 0) {
            debug("no VRAM for the recurrent states");
            for &(p, _) in &ms.states {
                if p != 0 {
                    unsafe { be.cu.cuMemFree_v2(p) };
                }
            }
            ms.states.clear();
            ms.states_valid = false;
            return None;
        }
        Some(ms)
    }

    fn copy_rows(&self, src: P, dst: P, n: u32, t: usize) -> bool {
        let (s, d, ss, tt) = (src.a, dst.a, (src.ts / 4) as u32, t as u32);
        self.be
            .launch(self.copy_rows, (n.div_ceil(256), tt), 256, &mut args![s, ss, d, n, tt])
    }

    /// Launches `op` for token `j` of the step (`sp` its parameters).
    fn launch_op(&self, op: &COp, slot: &CudaSlot, j: usize) -> bool {
        let be = self.be;
        let sp = self.sp + 16 * j as u64;
        match *op {
            COp::Mv {
                f,
                w,
                w2,
                x,
                y,
                n_out,
                n_in,
                mode,
                gate,
                ..
            } => {
                let (x, y, gate) = (x.at(j), y.at(j), gate.at(j));
                be.launch(
                    f,
                    (n_out.div_ceil(MV_ROWS * MV_NR), 1),
                    32 * MV_ROWS,
                    &mut args![w, w2, x, y, n_out, n_in, mode, gate],
                )
            }
            COp::Topk {
                f,
                logits,
                sel,
                n,
                k,
                scale,
            } => {
                let (logits, sel, scale) = (logits.at(j), sel.at(j), scale.a);
                be.launch(
                    f,
                    (1, 1),
                    n.next_multiple_of(32),
                    &mut args![logits, sel, n, k, (scale != 0) as u32, scale],
                )
            }
            COp::Experts {
                pair,
                sum,
                tab_g,
                tab_u,
                tab_d,
                x,
                act,
                y,
                sel,
                h,
                inter,
                k,
                r2off,
                gelu,
                add,
            } => {
                let (x, act, y, sel) = (x.at(j), act.at(j), y.at(j), sel.at(j));
                be.launch(
                    pair,
                    (inter.div_ceil(MV_ROWS), k),
                    32 * MV_ROWS,
                    &mut args![tab_g, tab_u, x, act, inter, h, r2off, gelu, sel],
                ) && be.launch(
                    sum,
                    (h.div_ceil(MV_ROWS), 1),
                    32 * MV_ROWS,
                    &mut args![tab_d, act, y, h, inter, k, add, sel],
                )
            }
            COp::Norm {
                f,
                x,
                out,
                w,
                p,
                rows,
                threads,
            } => {
                let (x, out, w) = (x.at(j), out.at(j), w.a);
                be.launch(f, (rows, 1), threads, &mut args![x, out, w, p])
            }
            COp::AddNorm {
                f,
                y,
                x,
                out,
                wy,
                wn,
                n,
                eps,
                scale,
                has_next,
                threads,
            } => {
                let (y, x, out, wy, wn) = (y.at(j), x.at(j), out.at(j), wy.a, wn.a);
                be.launch(f, (1, 1), threads, &mut args![y, x, out, wy, wn, n, eps, scale, has_next])
            }
            COp::QkRope {
                f,
                qkv,
                qo,
                kv,
                kout,
                vout,
                qn,
                kn,
                cs,
                p,
            } => {
                let (kc, vc) = slot.kv.get(kv).map_or((0, 0), |m| (m.k, m.v));
                let (qkv, qo, kout, vout, qn, kn, cs) = (qkv.at(j), qo.at(j), kout.at(j), vout.at(j), qn.a, kn.a, cs.at(j));
                be.launch(
                    f,
                    (p.nh + 2 * p.nkv, 1),
                    p.hd,
                    &mut args![qkv, qo, kc, vc, kout, vout, qn, kn, cs, sp, p],
                )
            }
            COp::Attn {
                f,
                combine,
                q,
                out,
                kv,
                gate,
                p,
                grp,
                hd,
            } => {
                let m = &slot.kv[kv];
                let (q, out, gate, part) = (q.at(j), out.at(j), gate.at(j), self.part);
                be.launch(
                    f,
                    (p.nh / grp, ATTN_SPLIT as u32),
                    256,
                    &mut args![q, m.k, m.v, out, gate, part, sp, p],
                ) && be.launch(combine, (p.nh, 1), 256, &mut args![part, out, gate, sp, p, hd])
            }
            COp::GdnConv {
                f,
                x,
                out,
                state,
                w,
                dim,
                kc,
                ..
            } => {
                let (x, out, st, w) = (x.at(j), out.at(j), slot.states[state].0, w.a);
                be.launch(f, (dim.div_ceil(256), 1), 256, &mut args![x, out, st, w, dim, kc])
            }
            COp::GdnRule {
                f,
                c,
                z,
                b,
                a,
                out,
                state,
                cst,
                p,
                threads,
                ..
            } => {
                let (c, z, b, a, out, st, cst) = (c.at(j), z.at(j), b.at(j), a.at(j), out.at(j), slot.states[state].0, cst.a);
                be.launch(f, (p.nv, 1), threads, &mut args![c, z, b, a, out, st, cst, p])
            }
            COp::Glu { f, g, u, out, n, gelu } => {
                let (g, u, out) = (g.at(j), u.at(j), out.at(j));
                be.launch(f, (n.div_ceil(256), 1), 256, &mut args![g, u, out, n, gelu])
            }
            COp::Ew { f, a, b, out, n, kind, p } => {
                let (a, b, out) = (a.at(j), b.at(j), out.at(j));
                be.launch(f, (n.div_ceil(256), 1), 256, &mut args![a, b, out, n, kind, p[0], p[1], p[2]])
            }
        }
    }

    /// Launches `op` over the step's `t` tokens: matvecs at once, the rest token by token; the
    /// recurrent ops keep their inputs first (multi-token plans).
    fn launch_tokens(&self, op: &COp, slot: &CudaSlot, t: usize) -> bool {
        let be = self.be;
        match *op {
            COp::Mv {
                fk: Some(fk),
                w,
                w2,
                x,
                y,
                n_out,
                n_in,
                mode,
                gate,
                ..
            } if t > 1 => {
                // At most 8 tokens per launch (the kernel's shared tiles).
                let (xs, ys, gs) = ((x.ts / 4) as u32, (y.ts / 4) as u32, (gate.ts / 4) as u32);
                return (0..t).step_by(8).all(|j0| {
                    let (xa, ya, ga, nt) = (x.at(j0), y.at(j0), gate.at(j0), (t - j0).min(8) as u32);
                    be.launch(
                        fk,
                        (n_out.div_ceil(32), 1),
                        256,
                        &mut args![w, w2, xa, xs, ya, ys, n_out, n_in, mode, ga, gs, nt],
                    )
                });
            }
            COp::GdnConv { x, save, dim, .. } if save.a != 0 => {
                if !self.copy_rows(x, save, dim, t) {
                    return false;
                }
            }
            COp::GdnRule {
                c, z, b, a, save, lens, ..
            } if save[0].a != 0 => {
                for (i, src) in [c, z, b, a].into_iter().enumerate() {
                    if !self.copy_rows(src, save[i], lens[i], t) {
                        return false;
                    }
                }
            }
            _ => {}
        }
        (0..t).all(|j| self.launch_op(op, slot, j))
    }
}

impl DecodePlan for CudaPlan<'_> {
    fn tokens(&self) -> usize {
        self.tokens
    }

    fn step(&self, slot: &mut Slot, s: Step) -> bool {
        let be = self.be;
        let t = s.t;
        if t == 0
            || t > self.tokens
            || s.kv.len() != self.kv.len()
            || s.rope_cs.len() != 2 * t * self.rope_halves.iter().sum::<usize>()
            || s.x.len() != t * self.input.1
            || self.input2.is_some_and(|(_, n, _)| s.x2.len() != t * n)
            || s.hidden.len() < t * self.hidden.1
            || s.capture
                .as_ref()
                .is_some_and(|c| self.capture.is_none_or(|(_, n, _)| c.len() < t * n))
            || s.states.len() != self.states.len()
            || s.states.iter().zip(&self.states).any(|(a, &n)| a.len() != n)
            || s.params.is_some()
            || !be.bind()
        {
            debug("step: inputs do not match the plan");
            return false;
        }
        let _g = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let Some(ms) = self.slot(slot) else { return false };
        let pos = s.pos;
        if !self.sync_kv(ms, s.kv, pos, t) {
            debug("step: KV caches could not be synced (VRAM?)");
            return false;
        }
        let tr = trace();
        let t0 = std::time::Instant::now();
        if !(s.states_kept && ms.states_valid) {
            for (&(p, _), st) in ms.states.iter().zip(s.states.iter()) {
                if !be.htod(p, st) {
                    return false;
                }
            }
        }
        // Until this step has written its states back.
        ms.states_valid = false;
        let t1 = std::time::Instant::now();
        // Inputs: rows `ts` floats apart; per token (position, first position, tokens, index);
        // each RoPE table's rows.
        let rows = |src: &[f32], n: usize, ts: usize| -> Vec<f32> {
            let mut v = vec![0f32; (t - 1) * ts + n];
            for j in 0..t {
                v[j * ts..j * ts + n].copy_from_slice(&src[j * n..(j + 1) * n]);
            }
            v
        };
        let sp: Vec<u32> = (0..t).flat_map(|j| [(pos + j) as u32, pos as u32, t as u32, j as u32]).collect();
        if !be.htod(self.input.0, &rows(s.x, self.input.1, self.input.2))
            || self.input2.is_some_and(|(p, n, ts)| !be.htod(p, &rows(s.x2, n, ts)))
            || !be.htod(self.sp, &sp)
        {
            return false;
        }
        let (mut src, mut dst) = (0, self.cs);
        for &half in &self.rope_halves {
            let n = t * 2 * half;
            if n > 0 && !be.htod(dst, &s.rope_cs[src..src + n]) {
                return false;
            }
            src += n;
            dst += (self.tokens * 2 * half * 4) as u64;
        }
        let want_logits = s.logits.is_some() && self.logits.is_some();
        let end = if want_logits { self.ops.len() } else { self.head_from };
        let queue = self.async_states();
        let mut queued = 0;
        let per_op = tr && std::env::var("OXINFER_CUDA_DECODE_TRACE").is_ok_and(|v| v == "2");
        let mut kinds: std::collections::BTreeMap<&str, (usize, f64)> = Default::default();
        for (i, op) in self.ops[..end].iter().enumerate() {
            let ts = std::time::Instant::now();
            if !self.launch_tokens(op, ms, t) {
                return false;
            }
            if per_op {
                unsafe { be.cu.cuCtxSynchronize() };
                let k = match op {
                    COp::Mv { fk: Some(_), .. } => "matvec",
                    COp::Mv { .. } => "matvec-1",
                    COp::Topk { .. } => "topk",
                    COp::Experts { .. } => "experts",
                    COp::Norm { .. } => "norm",
                    COp::AddNorm { .. } => "add-norm",
                    COp::QkRope { .. } => "qk-rope",
                    COp::Attn { .. } => "attention",
                    COp::GdnConv { .. } => "gdn-conv",
                    COp::GdnRule { .. } => "gdn-rule",
                    COp::Glu { .. } => "glu",
                    COp::Ew { .. } => "ew",
                };
                let e = kinds.entry(k).or_default();
                e.0 += 1;
                e.1 += ts.elapsed().as_secs_f64() * 1e6;
            }
            if queue {
                for &st in &self.last_writer[i] {
                    if !self.queue_state(&ms.states, st) {
                        return false;
                    }
                    queued += 1;
                }
            }
        }
        // States no op wrote (none in practice) go down after everything.
        let queued = queue && queued == self.states.len();
        if per_op {
            eprintln!(
                "[cuda decode ops t={t}: {:?}]",
                kinds.iter().map(|(k, (n, us))| format!("{k} {n}x {us:.0}us")).collect::<Vec<_>>()
            );
        }
        if tr {
            unsafe { be.cu.cuCtxSynchronize() };
        }
        let t2 = std::time::Instant::now();
        // Results down (each copy waits for the kernels before it on the stream).
        let dtoh = |dst: &mut [f32], src: u64| unsafe { be.cu.cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, src, dst.len() * 4) == 0 };
        let down = |dst: &mut [f32], (p, n, ts): (u64, usize, usize)| -> bool {
            if t == 1 || ts == n {
                return dtoh(&mut dst[..t * n], p);
            }
            let mut v = vec![0f32; (t - 1) * ts + n];
            if !dtoh(&mut v, p) {
                return false;
            }
            for j in 0..t {
                dst[j * n..(j + 1) * n].copy_from_slice(&v[j * ts..j * ts + n]);
            }
            true
        };
        if !down(s.hidden, self.hidden) {
            return false;
        }
        if let (Some(l), Some((p, n, ts))) = (s.logits, self.logits)
            && want_logits
        {
            // Callers may take fewer logits per token than the head computes.
            let n2 = n.min(l.len() / t);
            if n2 == n {
                if !down(l, (p, n, ts)) {
                    return false;
                }
            } else {
                for j in 0..t {
                    if !dtoh(&mut l[j * n2..(j + 1) * n2], p + (j * ts * 4) as u64) {
                        return false;
                    }
                }
            }
        }
        if let (Some(c), Some(cap)) = (s.capture, self.capture)
            && !down(c, cap)
        {
            return false;
        }
        let t3 = std::time::Instant::now();
        if !self.states_down(&ms.states, s.states, queued) {
            return false;
        }
        if tr {
            eprintln!(
                "[cuda decode step t={t}: states up {:.0} us, ops {:.0} us, results {:.0} us, states down {:.0} us]",
                (t1 - t0).as_secs_f64() * 1e6,
                (t2 - t1).as_secs_f64() * 1e6,
                (t3 - t2).as_secs_f64() * 1e6,
                t3.elapsed().as_secs_f64() * 1e6
            );
        }
        ms.states_valid = true;
        ms.last = (self.id, t);
        if s.scratch {
            for m in ms.kv.iter_mut() {
                m.synced = m.synced.min(pos);
            }
            return true;
        }
        let mut kvo = vec![0f32; self.kvout.1];
        if !kvo.is_empty() && !dtoh(&mut kvo, self.kvout.0) {
            return false;
        }
        let mut o = 0;
        for (i, (m, c)) in ms.kv.iter_mut().zip(s.kv.iter_mut()).enumerate() {
            let row = self.kv[i].0 * self.kv[i].1;
            let tr = self.tokens * row;
            c.push(&kvo[o..o + t * row], &kvo[o + tr..o + tr + t * row], t);
            o += 2 * tr;
            m.synced = pos + t;
            c.kept = c.len;
        }
        true
    }

    fn replay_recurrent(&self, slot: &mut Slot, states: &mut [&mut [f32]], keep: usize) -> bool {
        let be = self.be;
        if self.tokens < 2 || states.len() != self.states.len() || states.iter().zip(&self.states).any(|(a, &n)| a.len() != n) || !be.bind()
        {
            return false;
        }
        let _g = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let Some(ms) = self.slot(slot) else { return false };
        if keep == 0 || ms.last.0 != self.id || keep > ms.last.1 {
            return false;
        }
        ms.states_valid = false;
        for (&(p, _), st) in ms.states.iter().zip(states.iter()) {
            if !be.htod(p, st) {
                return false;
            }
        }
        // The recurrent ops again over the first `keep` tokens, from the inputs they kept (the
        // outputs land in the arena, unused).
        for op in &self.ops {
            let ok = match *op {
                COp::GdnConv {
                    f,
                    out,
                    state,
                    w,
                    dim,
                    kc,
                    save,
                    ..
                } => (0..keep).all(|j| {
                    let (x, out, st, w) = (save.at(j), out.at(j), ms.states[state].0, w.a);
                    be.launch(f, (dim.div_ceil(256), 1), 256, &mut args![x, out, st, w, dim, kc])
                }),
                COp::GdnRule {
                    f,
                    out,
                    state,
                    cst,
                    p,
                    threads,
                    save,
                    ..
                } => (0..keep).all(|j| {
                    let (c, z, b, a) = (save[0].at(j), save[1].at(j), save[2].at(j), save[3].at(j));
                    let (out, st, cst) = (out.at(j), ms.states[state].0, cst.a);
                    be.launch(f, (p.nv, 1), threads, &mut args![c, z, b, a, out, st, cst, p])
                }),
                _ => true,
            };
            if !ok {
                return false;
            }
        }
        if !self.states_down(&ms.states, states, false) {
            return false;
        }
        ms.states_valid = true;
        true
    }
}
