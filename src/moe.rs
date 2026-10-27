//! Routed experts of a mixture-of-experts layer, shared by the model families that have them
//! (Qwen3.5 / 3.6 MoE: SiLU experts; Gemma 4 26B-A4B: GELU-tanh experts). The router stays with
//! each model (they differ); this runs the chosen experts for given routes: per token on the
//! decode kernels for short passes, as one grouped matmul for prefill, or on the accelerator.

use crate::kernels::*;
use crate::model::{Lin, Quant, linear, linear_grouped, linear_multi, load_lin, load_lin_experts};
use crate::prof::{self, Cat};
use crate::safetensors::ModelFiles;
use anyhow::Result;

/// The gated activation of the experts: `act(gate) * up`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Act {
    Silu,
    GeluTanh,
}

impl Act {
    fn apply(self, g: &mut [f32], u: &[f32]) {
        match self {
            Act::Silu => silu_mul(g, u),
            Act::GeluTanh => gelu_mul(g, u),
        }
    }
    #[inline]
    fn scalar(self, g: f32, u: f32) -> f32 {
        match self {
            Act::Silu => g / (1.0 + (-g).exp()) * u,
            Act::GeluTanh => gelu_tanh(g) * u,
        }
    }
}

/// Called with the routed experts of a MoE pass before they run (weight streaming reads them on
/// demand).
pub(crate) type NeedExperts<'f> = Option<&'f dyn Fn(&[usize])>;

/// An expert's input projections: fused (`experts.gate_up_proj`, `[2 * inter, H]`, gate rows then
/// up rows) or separate (`experts.{e}.gate_proj` / `up_proj`, as transformers saves them).
// One per expert, built once: boxing the fused case would gain nothing.
#[allow(clippy::large_enum_variant)]
pub(crate) enum GateUp<'a> {
    Fused(Lin<'a>),
    Split(Lin<'a>, Lin<'a>),
}

impl GateUp<'_> {
    fn bytes(&self) -> usize {
        match self {
            GateUp::Fused(l) => l.bytes(),
            GateUp::Split(g, u) => g.bytes() + u.bytes(),
        }
    }
}

/// The routed experts of one layer.
pub(crate) struct Experts<'a> {
    pub gate_up: Vec<GateUp<'a>>, // per expert
    pub down: Vec<Lin<'a>>,       // per expert [H, inter]
    pub top_k: usize,
    pub inter: usize,
    pub act: Act,
    /// Accelerator state (`Backend::upload_moe`), made on first use.
    dev: std::sync::OnceLock<Option<crate::gpu::DeviceWeight>>,
}

/// Indices and renormalized weights of the top-k experts of one token (softmax over all experts,
/// top k, divided by their sum: the same as a softmax over the k largest logits). Ties go to the
/// lower index.
pub(crate) fn route_top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    idx.truncate(k);
    let m = logits[idx[0]];
    let e: Vec<f32> = idx.iter().map(|&i| (logits[i] - m).exp()).collect();
    let s: f32 = e.iter().sum();
    idx.into_iter().zip(e).map(|(i, v)| (i, v / s)).collect()
}

/// `y += a * x`.
fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += a * x;
    }
}

impl<'a> Experts<'a> {
    /// Loads the `n_exp` experts under `ep` (`...experts.`): fused 3D `gate_up_proj` +
    /// `down_proj`, separate 3D `gate_proj` / `up_proj` / `down_proj` (GGUF), or one module per
    /// expert (`{e}.gate_proj.weight`, ...).
    #[allow(clippy::too_many_arguments)]
    pub fn load(files: &'a ModelFiles, ep: &str, n_exp: usize, top_k: usize, mi: usize, h: usize, quant: Quant, act: Act) -> Result<Self> {
        let has =
            |n: &str| files.has(n) || files.has(&format!("{n}.q8")) || files.has(&format!("{n}.q4")) || files.has(&format!("{n}.scales"));
        let fused = format!("{ep}gate_up_proj");
        let (gate_up, down) = if has(&fused) {
            (
                load_lin_experts(files, &fused, n_exp, 2 * mi, h, quant)?
                    .into_iter()
                    .map(GateUp::Fused)
                    .collect(),
                load_lin_experts(files, &format!("{ep}down_proj"), n_exp, h, mi, quant)?,
            )
        } else if has(&format!("{ep}gate_proj")) {
            let (gn, un) = (format!("{ep}gate_proj"), format!("{ep}up_proj"));
            let d = load_lin_experts(files, &format!("{ep}down_proj"), n_exp, h, mi, quant)?;
            // MLX 4-bit (`switch_glu`): gate and up fused per expert as they are repacked, so the
            // experts take one input matmul (and the accelerator path) like fused checkpoints.
            if let Some(gu) = files.mlx_q4_experts_fused(&gn, &un, n_exp, mi, h)? {
                files.note_use(&gn, true, false);
                files.note_use(&un, true, false);
                (gu.into_iter().map(|m| GateUp::Fused(Lin::q4(m))).collect(), d)
            } else {
                let g = load_lin_experts(files, &gn, n_exp, mi, h, quant)?;
                let u = load_lin_experts(files, &un, n_exp, mi, h, quant)?;
                (g.into_iter().zip(u).map(|(g, u)| GateUp::Split(g, u)).collect(), d)
            }
        } else {
            let (mut gu, mut d) = (Vec::with_capacity(n_exp), Vec::with_capacity(n_exp));
            for e in 0..n_exp {
                let p = format!("{ep}{e}.");
                gu.push(GateUp::Split(
                    load_lin(files, &format!("{p}gate_proj.weight"), mi, h, quant)?,
                    load_lin(files, &format!("{p}up_proj.weight"), mi, h, quant)?,
                ));
                d.push(load_lin(files, &format!("{p}down_proj.weight"), h, mi, quant)?);
            }
            (gu, d)
        };
        Ok(Self {
            gate_up,
            down,
            top_k,
            inter: mi,
            act,
            dev: std::sync::OnceLock::new(),
        })
    }

    pub fn n_experts(&self) -> usize {
        self.down.len()
    }

    /// Weight bytes one token reads (top_k experts).
    pub fn bytes_per_token(&self) -> usize {
        self.top_k * (self.gate_up[0].bytes() + self.down[0].bytes())
    }

    /// The weights of each expert.
    pub fn lins(&self) -> Vec<Vec<&Lin<'a>>> {
        self.gate_up
            .iter()
            .zip(&self.down)
            .map(|(g, d)| match g {
                GateUp::Fused(l) => vec![l, d],
                GateUp::Split(a, b) => vec![a, b, d],
            })
            .collect()
    }

    /// The memory of each expert (`Lin::mem_ranges`), for streaming them on demand.
    pub fn ranges(&self) -> Vec<Vec<(usize, usize)>> {
        self.lins()
            .iter()
            .map(|e| e.iter().flat_map(|l| l.mem_ranges()).collect())
            .collect()
    }

    /// The routed experts on the active accelerator (`Backend::moe`), if it takes them: fused
    /// gate/up weights that are not streamed, and a prefill-sized pass.
    fn forward_device(&self, x: &[f32], t: usize, h: usize, routes: &[Vec<(usize, f32)>], y: &mut [f32]) -> bool {
        let mi = self.inter;
        let Some(be) = crate::gpu::active() else {
            return false;
        };
        if !be.wants(2 * mi, h, t * self.top_k) {
            return false;
        }
        // Streamed experts keep no device state (see `Lin::set_transient_device`).
        if self.down.iter().any(|l| l.is_transient()) {
            return false;
        }
        let dev = self.dev.get_or_init(|| {
            let mut gu = Vec::with_capacity(self.gate_up.len());
            for g in &self.gate_up {
                match g {
                    GateUp::Fused(l) if !l.is_transient() => gu.push(l.as_ref()),
                    _ => return None,
                }
            }
            let down: Vec<_> = self.down.iter().map(|l| l.as_ref()).collect();
            be.upload_moe(&gu, &down, h, mi, self.act == Act::GeluTanh)
        });
        let Some(dev) = dev else {
            return false;
        };
        let _timer = prof::start(Cat::Matmul);
        be.moe(x, t, h, mi, routes, dev, y)
    }

    /// Adds the routed experts' output to `y` (`[t, H]`): token `i` of `x` goes through the
    /// experts `routes[i]` lists (expert, weight). `need` is told the experts used before they run
    /// (weight streaming reads them on demand).
    pub fn forward(&self, x: &[f32], t: usize, h: usize, routes: &[Vec<(usize, f32)>], y: &mut [f32], need: NeedExperts) {
        if let Some(need) = need {
            let mut used: Vec<usize> = routes.iter().flatten().map(|r| r.0).collect();
            used.sort_unstable();
            used.dedup();
            need(&used);
        }
        let mi = self.inter;
        let act = self.act;
        if t == 1 {
            // Decode: every chosen expert sees the same row, so the gate/up matmuls run as one batch.
            let r = &routes[0];
            let mut gu = vec![vec![0f32; 2 * mi]; r.len()];
            let mut jobs: Vec<(&Lin, usize, &mut [f32])> = Vec::with_capacity(2 * r.len());
            for (&(e, _), o) in r.iter().zip(gu.iter_mut()) {
                match &self.gate_up[e] {
                    GateUp::Fused(l) => jobs.push((l, 2 * mi, &mut o[..])),
                    GateUp::Split(g, u) => {
                        let (og, ou) = o.split_at_mut(mi);
                        jobs.push((g, mi, og));
                        jobs.push((u, mi, ou));
                    }
                }
            }
            linear_multi(&mut jobs, h, x, 1);
            let mut o = vec![0f32; h];
            for (&(e, w), gu) in r.iter().zip(gu.iter_mut()) {
                let (g, u) = gu.split_at_mut(mi);
                act.apply(g, u);
                linear(&self.down[e], h, mi, g, 1, &mut o);
                axpy(y, w, &o);
            }
            return;
        }
        if t < TN {
            // Short passes (speculative verification): each chosen expert runs over its own tokens
            // with the decode kernels, so every row matches a one-token forward bit for bit.
            let mut used: Vec<usize> = routes.iter().flatten().map(|r| r.0).collect();
            used.sort_unstable();
            used.dedup();
            let mut outs: Vec<(Vec<usize>, Vec<f32>)> = Vec::with_capacity(used.len());
            for &e in &used {
                let toks: Vec<usize> = (0..t).filter(|&ti| routes[ti].iter().any(|r| r.0 == e)).collect();
                let n = toks.len();
                let mut xs = vec![0f32; n * h];
                for (row, &ti) in xs.chunks_mut(h).zip(&toks) {
                    row.copy_from_slice(&x[ti * h..(ti + 1) * h]);
                }
                let mut a = vec![0f32; n * mi];
                match &self.gate_up[e] {
                    GateUp::Fused(l) => {
                        let mut gu = vec![0f32; n * 2 * mi];
                        linear(l, 2 * mi, h, &xs, n, &mut gu);
                        for (ar, r) in a.chunks_mut(mi).zip(gu.chunks_mut(2 * mi)) {
                            let (g, u) = r.split_at_mut(mi);
                            act.apply(g, u);
                            ar.copy_from_slice(g);
                        }
                    }
                    GateUp::Split(g, u) => {
                        let mut ub = vec![0f32; n * mi];
                        linear_multi(&mut [(g, mi, &mut a[..]), (u, mi, &mut ub[..])], h, &xs, n);
                        act.apply(&mut a, &ub);
                    }
                }
                let mut o = vec![0f32; n * h];
                linear(&self.down[e], h, mi, &a, n, &mut o);
                outs.push((toks, o));
            }
            for (ti, r) in routes.iter().enumerate() {
                let yr = &mut y[ti * h..(ti + 1) * h];
                for &(e, w) in r {
                    let (toks, o) = &outs[used.binary_search(&e).unwrap()];
                    let i = toks.iter().position(|&x| x == ti).unwrap();
                    axpy(yr, w, &o[i * h..(i + 1) * h]);
                }
            }
            return;
        }
        // Prefill on an accelerator: all routed experts in one submission (gather, grouped
        // gate/up GEMM, activation, grouped down GEMM, weighted combination).
        if self.forward_device(x, t, h, routes, y) {
            return;
        }
        // Prefill: gather each expert's tokens into one segment (padded to whole GEMM tiles) and
        // run all experts as one grouped matmul.
        let ne = self.n_experts();
        let mut count = vec![0usize; ne];
        for r in routes {
            for &(e, _) in r {
                count[e] += 1;
            }
        }
        let mut segs = Vec::with_capacity(ne);
        let mut rows = 0;
        for &c in &count {
            let len = c.next_multiple_of(TN);
            segs.push((rows, len));
            rows += len;
        }
        // Row of each (token, k) in the gathered input, and the token of each gathered row.
        let mut fill: Vec<usize> = segs.iter().map(|s| s.0).collect();
        let mut src = vec![usize::MAX; rows];
        let slot: Vec<Vec<(usize, f32)>> = routes
            .iter()
            .enumerate()
            .map(|(ti, r)| {
                r.iter()
                    .map(|&(e, w)| {
                        let i = fill[e];
                        fill[e] += 1;
                        src[i] = ti;
                        (i, w)
                    })
                    .collect()
            })
            .collect();
        let mut xs = vec![0f32; rows * h];
        for_chunks(&mut xs, h, true, |i, row| {
            if src[i] != usize::MAX {
                row.copy_from_slice(&x[src[i] * h..(src[i] + 1) * h]);
            }
        });
        let mut a = vec![0f32; rows * mi];
        match &self.gate_up[0] {
            GateUp::Fused(_) => {
                let ls: Vec<&Lin> = self
                    .gate_up
                    .iter()
                    .map(|g| if let GateUp::Fused(l) = g { l } else { unreachable!() })
                    .collect();
                let mut gu = vec![0f32; rows * 2 * mi];
                linear_grouped(&ls, &segs, 2 * mi, h, &xs, &mut gu);
                for_chunks(&mut a, mi, true, |i, ar| {
                    let r = &gu[i * 2 * mi..(i + 1) * 2 * mi];
                    for ((o, &g), &u) in ar.iter_mut().zip(&r[..mi]).zip(&r[mi..]) {
                        *o = act.scalar(g, u);
                    }
                });
            }
            GateUp::Split(..) => {
                let (gs, us): (Vec<&Lin>, Vec<&Lin>) = self
                    .gate_up
                    .iter()
                    .map(|g| if let GateUp::Split(g, u) = g { (g, u) } else { unreachable!() })
                    .unzip();
                let mut u = vec![0f32; rows * mi];
                linear_grouped(&gs, &segs, mi, h, &xs, &mut a);
                linear_grouped(&us, &segs, mi, h, &xs, &mut u);
                act.apply(&mut a, &u);
            }
        }
        let ds: Vec<&Lin> = self.down.iter().collect();
        let mut o = vec![0f32; rows * h];
        linear_grouped(&ds, &segs, h, mi, &a, &mut o);
        for_chunks(y, h, true, |ti, yr| {
            for &(i, w) in &slot[ti] {
                axpy(yr, w, &o[i * h..(i + 1) * h]);
            }
        });
    }
}
