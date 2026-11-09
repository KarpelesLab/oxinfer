//! Routed experts of a mixture-of-experts layer, shared by the model families that have them
//! (Qwen3.5 / 3.6 MoE: SiLU experts; Gemma 4 26B-A4B: GELU-tanh experts). The router stays with
//! each model (they differ); this runs the chosen experts (and any shared expert) for given
//! routes: on the decode kernels for short passes (one batch per stage for all experts), as one
//! grouped matmul for prefill, or on the accelerator.

use crate::kernels::*;
use crate::model::{Lin, Quant, linear, linear_grouped, linear_multi, load_lin, load_lin_experts};
use crate::prof::{self, Cat};
use crate::safetensors::ModelFiles;
use anyhow::Result;

/// The gated activation of the experts: `act(gate) * up`.
pub(crate) use crate::gpu::MoeAct as Act;

impl Act {
    fn apply(self, g: &mut [f32], u: &mut [f32]) {
        match self {
            Act::Silu => silu_mul(g, u),
            Act::GeluTanh => gelu_mul(g, u),
            Act::SiluClamp(l) => {
                for v in g.iter_mut() {
                    *v = v.min(l);
                }
                for v in u.iter_mut() {
                    *v = v.clamp(-l, l);
                }
                silu_mul(g, u)
            }
        }
    }
    #[inline]
    fn scalar(self, g: f32, u: f32) -> f32 {
        match self {
            Act::Silu => g / (1.0 + (-g).exp()) * u,
            Act::GeluTanh => gelu_tanh(g) * u,
            Act::SiluClamp(l) => {
                let g = g.min(l);
                g / (1.0 + (-g).exp()) * u.clamp(-l, l)
            }
        }
    }
}

/// Called with the routed experts of a MoE pass before they run (weight streaming reads them on
/// demand).
pub(crate) type NeedExperts<'f> = Option<&'f dyn Fn(&[usize])>;

/// A shared expert beside the routed ones, on the same input: `y = scale * down(act(gate x) *
/// up x)` per token (overwriting `y`), before the routed experts add theirs.
pub(crate) struct Shared<'s, 'a> {
    pub gate: &'s Lin<'a>,
    pub up: &'s Lin<'a>,
    pub down: &'s Lin<'a>,
    pub inter: usize,
    pub act: Act,
    /// Neuron edits of target `Shared`.
    pub hook: Option<&'s crate::intervene::NeuronHook<'s>>,
    /// Per-token output scale (Qwen's sigmoid gate).
    pub scale: Option<Vec<f32>>,
}

impl Shared<'_, '_> {
    /// The shared expert alone over `t` rows of `x` into `y`.
    fn forward(&self, x: &[f32], t: usize, h: usize, y: &mut [f32]) {
        let mi = self.inter;
        let mut g = vec![0f32; t * mi];
        let mut u = vec![0f32; t * mi];
        linear_multi(&mut [(self.gate, mi, &mut g[..]), (self.up, mi, &mut u[..])], h, x, t);
        self.act.apply(&mut g, &mut u);
        if let Some(k) = self.hook {
            k.apply(crate::intervene::Target::Shared, &mut g, mi);
        }
        linear(self.down, h, mi, &g, t, y);
        self.apply_scale(t, h, y);
    }
    fn apply_scale(&self, t: usize, h: usize, y: &mut [f32]) {
        if let Some(s) = &self.scale {
            for_chunks(y, h, t > 1, |ti, row| scale_inplace(row, s[ti]));
        }
    }
}

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
    /// Prefill wraps only the experts a pass routes to, for that pass (as for streamed experts),
    /// rather than keeping all of them wrapped: set for expert sets too large to be wrapped twice
    /// (prefill and the GPU decode plan each wrap what they use).
    pub(crate) wrap_per_pass: std::sync::atomic::AtomicBool,
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
            wrap_per_pass: std::sync::atomic::AtomicBool::new(false),
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
    /// or split gate/up weights that are not streamed, and a prefill-sized pass.
    fn forward_device(&self, x: &[f32], t: usize, h: usize, routes: &[Vec<(usize, f32)>], y: &mut [f32]) -> bool {
        let mi = self.inter;
        let Some(be) = crate::gpu::active() else {
            return false;
        };
        if !be.wants(2 * mi, h, t * self.top_k) {
            return false;
        }
        let act = self.act;
        // The weights of experts `ids`: fused or split gate/up (one form for all), and down.
        let refs = |ids: &mut dyn Iterator<Item = usize>| {
            let mut gu = Vec::with_capacity(self.gate_up.len());
            let mut up = Vec::with_capacity(self.gate_up.len());
            let mut down = Vec::with_capacity(self.gate_up.len());
            for e in ids {
                down.push(self.down[e].as_ref());
                match &self.gate_up[e] {
                    GateUp::Fused(l) => gu.push(l.as_ref()),
                    GateUp::Split(a, b) => {
                        gu.push(a.as_ref());
                        up.push(b.as_ref());
                    }
                }
            }
            (up.is_empty() || up.len() == gu.len()).then_some((gu, up, down))
        };
        // Streamed experts keep no device state between calls (see `Lin::set_transient_device`):
        // a prefill pass wraps the experts it routes to for its own submission, which reads them
        // in place. Only those: a wrapped page must be in memory while the submission runs, so
        // wrapping the others would read every expert of the layer from storage.
        if self.wrap_per_pass.load(std::sync::atomic::Ordering::Relaxed)
            || self.down.iter().any(|l| l.is_transient())
            || self.gate_up.iter().any(|g| match g {
                GateUp::Fused(l) => l.is_transient(),
                GateUp::Split(a, b) => a.is_transient() || b.is_transient(),
            })
        {
            let mut used: Vec<usize> = routes.iter().flatten().map(|r| r.0).collect();
            used.sort_unstable();
            used.dedup();
            let Some((gu, up, down)) = refs(&mut used.iter().copied()) else {
                return false;
            };
            let Some(dev) = be.upload_moe_transient(&gu, (!up.is_empty()).then_some(&up[..]), &down, h, mi, act) else {
                return false;
            };
            let local: Vec<Vec<(usize, f32)>> = routes
                .iter()
                .map(|r| r.iter().map(|&(e, w)| (used.binary_search(&e).unwrap(), w)).collect())
                .collect();
            let _timer = prof::start(Cat::Matmul);
            return be.moe(x, t, h, mi, &local, &dev, y);
        }
        let dev = self.dev.get_or_init(|| {
            let (gu, up, down) = refs(&mut (0..self.down.len()))?;
            be.upload_moe(&gu, (!up.is_empty()).then_some(&up[..]), &down, h, mi, act)
        });
        let Some(dev) = dev else {
            return false;
        };
        let _timer = prof::start(Cat::Matmul);
        be.moe(x, t, h, mi, routes, dev, y)
    }

    /// Decode and short passes (speculative verification) on the CPU decode kernels with each
    /// stage of every chosen expert in one parallel region: `x` is quantized once and each
    /// expert gets the rows of its tokens (the expert -> tokens grouping is built once for gate,
    /// up and down), then all gate/up matmuls run as one batch, then all down matmuls (their
    /// inputs quantized together), the shared expert's in the same batches. Every row is
    /// bit-identical to running the experts one at a time (activations are quantized per row,
    /// and the decode kernels are batch-invariant). False, with nothing done, when some expert
    /// would not take the CPU decode kernels (see [`crate::model::cpu_decode_kernels`]): an
    /// accelerator, a LoRA or float weights then get the per-expert path, so a matmul handled
    /// elsewhere never consumes these quantized inputs.
    #[allow(clippy::too_many_arguments)]
    fn forward_batched(
        &self,
        x: &[f32],
        t: usize,
        h: usize,
        routes: &[Vec<(usize, f32)>],
        y: &mut [f32],
        hook: Option<&crate::intervene::NeuronHook>,
        shared: Option<&Shared>,
    ) -> bool {
        use crate::intervene::Target;
        use crate::model::{cpu_decode_kernels, linear_batch_q};
        let mi = self.inter;
        let mut used: Vec<usize> = routes.iter().flatten().map(|r| r.0).collect();
        used.sort_unstable();
        used.dedup();
        let toks: Vec<Vec<usize>> = used
            .iter()
            .map(|&e| (0..t).filter(|&ti| routes[ti].iter().any(|r| r.0 == e)).collect())
            .collect();
        let mut ok = shared.is_none_or(|s| {
            let si = s.inter;
            cpu_decode_kernels(s.gate, si, h, t) && cpu_decode_kernels(s.up, si, h, t) && cpu_decode_kernels(s.down, h, si, t)
        });
        for (&e, tk) in used.iter().zip(&toks) {
            let n = tk.len();
            ok &= match &self.gate_up[e] {
                GateUp::Fused(l) => cpu_decode_kernels(l, 2 * mi, h, n),
                GateUp::Split(g, u) => cpu_decode_kernels(g, mi, h, n) && cpu_decode_kernels(u, mi, h, n),
            } && cpu_decode_kernels(&self.down[e], h, mi, n);
        }
        if !ok {
            return false;
        }
        // Inputs: all of x (experts that every token uses), then each other expert's rows.
        let xa = crate::quant::quantize_act(x, t, h);
        let mut xs = vec![];
        let input: Vec<usize> = toks
            .iter()
            .map(|tk| {
                if tk.len() == t {
                    0
                } else {
                    xs.push(xa.gather(tk));
                    xs.len()
                }
            })
            .collect();
        xs.insert(0, xa);
        // Gate/up: per expert [n, 2 mi] (fused: rows of gate then up) or gate [n, mi] then up [n, mi].
        let mut gu: Vec<Vec<f32>> = toks.iter().map(|tk| vec![0f32; tk.len() * 2 * mi]).collect();
        // The shared expert's gate then up ([t, inter] each).
        let mut sgu = vec![0f32; shared.map_or(0, |s| 2 * t * s.inter)];
        let mut jobs: Vec<(&Lin, usize, &mut [f32])> = Vec::with_capacity(2 * used.len() + 2);
        if let Some(s) = shared {
            let (g, u) = sgu.split_at_mut(t * s.inter);
            jobs.push((s.gate, 0, g));
            jobs.push((s.up, 0, u));
        }
        for ((&e, &i), o) in used.iter().zip(&input).zip(gu.iter_mut()) {
            match &self.gate_up[e] {
                GateUp::Fused(l) => jobs.push((l, i, &mut o[..])),
                GateUp::Split(g, u) => {
                    let half = o.len() / 2;
                    let (og, ou) = o.split_at_mut(half);
                    jobs.push((g, i, og));
                    jobs.push((u, i, ou));
                }
            }
        }
        linear_batch_q(&xs, &mut jobs);
        drop(jobs);
        // Activations of all experts, one after another ([sum n, mi]).
        let total: usize = toks.iter().map(|tk| tk.len()).sum();
        let mut a = vec![0f32; total * mi];
        let mut row = 0;
        for ((&e, tk), gu) in used.iter().zip(&toks).zip(gu.iter_mut()) {
            let n = tk.len();
            let ae = &mut a[row * mi..(row + n) * mi];
            match &self.gate_up[e] {
                GateUp::Fused(_) => {
                    for (ar, r) in ae.chunks_mut(mi).zip(gu.chunks_mut(2 * mi)) {
                        let (g, u) = r.split_at_mut(mi);
                        self.act.apply(g, u);
                        ar.copy_from_slice(g);
                    }
                }
                GateUp::Split(..) => {
                    let (g, u) = gu.split_at_mut(n * mi);
                    self.act.apply(g, u);
                    ae.copy_from_slice(g);
                }
            }
            if let Some(k) = hook {
                k.apply_rows(Target::Expert(e), ae, mi, tk);
            }
            row += n;
        }
        drop(gu);
        if let Some(s) = shared {
            let (g, u) = sgu.split_at_mut(t * s.inter);
            s.act.apply(g, u);
            if let Some(k) = s.hook {
                k.apply(Target::Shared, g, s.inter);
            }
            sgu.truncate(t * s.inter);
        }
        // Down: the activations quantized together (the shared expert's too when it has the
        // routed experts' width), then split per expert.
        let same = shared.is_some_and(|s| s.inter == mi);
        if same {
            a.extend_from_slice(&sgu);
        }
        let aq = crate::quant::quantize_act(&a, a.len() / mi, mi);
        let mut row = 0;
        let mut xs: Vec<crate::quant::QAct> = toks
            .iter()
            .map(|tk| {
                let rows: Vec<usize> = (row..row + tk.len()).collect();
                row += tk.len();
                aq.gather(&rows)
            })
            .collect();
        if let Some(s) = shared {
            xs.push(if same {
                aq.gather(&(total..total + t).collect::<Vec<_>>())
            } else {
                crate::quant::quantize_act(&sgu, t, s.inter)
            });
        }
        let mut outs: Vec<Vec<f32>> = toks.iter().map(|tk| vec![0f32; tk.len() * h]).collect();
        let mut jobs: Vec<(&Lin, usize, &mut [f32])> = used
            .iter()
            .zip(outs.iter_mut())
            .enumerate()
            .map(|(i, (&e, o))| (&self.down[e], i, &mut o[..]))
            .collect();
        if let Some(s) = shared {
            jobs.push((s.down, used.len(), &mut *y));
        }
        linear_batch_q(&xs, &mut jobs);
        drop(jobs);
        if let Some(s) = shared {
            s.apply_scale(t, h, y);
        }
        for (ti, r) in routes.iter().enumerate() {
            let yr = &mut y[ti * h..(ti + 1) * h];
            for &(e, w) in r {
                let j = used.binary_search(&e).unwrap();
                let i = toks[j].iter().position(|&x| x == ti).unwrap();
                axpy(yr, w, &outs[j][i * h..(i + 1) * h]);
            }
        }
        true
    }

    /// Adds the routed experts' output to `y` (`[t, H]`): token `i` of `x` goes through the
    /// experts `routes[i]` lists (expert, weight). `need` is told the experts used before they run
    /// (weight streaming reads them on demand).
    /// `hook`: neuron edits (routed experts' activations; they keep the experts on the CPU).
    /// `shared`: a shared expert, which then writes `y` first (the routed experts add to it).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &[f32],
        t: usize,
        h: usize,
        routes: &[Vec<(usize, f32)>],
        y: &mut [f32],
        need: NeedExperts,
        hook: Option<&crate::intervene::NeuronHook>,
        shared: Option<&Shared>,
    ) {
        use crate::intervene::Target;
        let hook = hook.filter(|k| k.experts());
        if let Some(need) = need {
            let mut used: Vec<usize> = routes.iter().flatten().map(|r| r.0).collect();
            used.sort_unstable();
            used.dedup();
            need(&used);
        }
        let mi = self.inter;
        let act = self.act;
        if t < TN && crate::kernels::moe_batched() && self.forward_batched(x, t, h, routes, y, hook, shared) {
            return;
        }
        if let Some(s) = shared {
            s.forward(x, t, h, y);
        }
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
                if let Some(k) = hook {
                    k.apply(Target::Expert(e), g, mi);
                }
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
                        act.apply(&mut a, &mut ub);
                    }
                }
                if let Some(k) = hook {
                    k.apply_rows(Target::Expert(e), &mut a, mi, &toks);
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
        if hook.is_none() && self.forward_device(x, t, h, routes, y) {
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
                act.apply(&mut a, &mut u);
            }
        }
        if let Some(k) = hook {
            for (e, &(s0, len)) in segs.iter().enumerate() {
                if k.edits(Target::Expert(e)) {
                    k.apply_rows(Target::Expert(e), &mut a[s0 * mi..(s0 + len) * mi], mi, &src[s0..s0 + len]);
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::{GgmlType, GgufMatrix, random_blocks};

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    /// Batched experts (one parallel region per stage, inputs quantized once) give bit-identical
    /// outputs to running each expert's matmuls on their own, for decode and short passes, fused
    /// and separate gate/up, in each quantized CPU format.
    #[test]
    fn batched_experts_match_one_at_a_time() {
        crate::pool::init(3);
        let (ne, k, mi, h) = (6usize, 3usize, 96usize, 256usize);
        let mut seed = 5u64;
        for fmt in 0..3 {
            for fused in [false, true] {
                let mut lin = |n_out: usize, n_in: usize| -> Lin<'static> {
                    let w: Vec<f32> = (0..n_out * n_in).map(|_| lcg(&mut seed)).collect();
                    match fmt {
                        0 => Lin::q8(crate::quant::quantize_f32(&w, n_out, n_in)),
                        1 => Lin::q4(crate::quant::quantize_q4_f32(&w, n_out, n_in)),
                        _ => {
                            let mut s = seed;
                            let b = random_blocks(GgmlType::Q8_0, n_out * n_in / 32, &mut s);
                            Lin::gguf(GgufMatrix::from_bytes(GgmlType::Q8_0, &b, n_out, n_in).unwrap())
                        }
                    }
                };
                let gate_up = (0..ne)
                    .map(|_| {
                        if fused {
                            GateUp::Fused(lin(2 * mi, h))
                        } else {
                            GateUp::Split(lin(mi, h), lin(mi, h))
                        }
                    })
                    .collect();
                let down = (0..ne).map(|_| lin(h, mi)).collect();
                // Shared experts as wide as the routed ones (activations quantized with theirs)
                // and wider.
                let sh: Vec<(Lin, Lin, Lin, usize)> = [mi, 2 * mi]
                    .into_iter()
                    .map(|si| (lin(si, h), lin(si, h), lin(h, si), si))
                    .collect();
                let ex = Experts {
                    gate_up,
                    down,
                    top_k: k,
                    inter: mi,
                    act: Act::Silu,
                    dev: std::sync::OnceLock::new(),
                    wrap_per_pass: std::sync::atomic::AtomicBool::new(false),
                };
                for t in 1..TN {
                    let x: Vec<f32> = (0..t * h).map(|_| lcg(&mut seed)).collect();
                    let routes: Vec<Vec<(usize, f32)>> = (0..t)
                        .map(|_| {
                            let l: Vec<f32> = (0..ne).map(|_| lcg(&mut seed)).collect();
                            route_top_k(&l, k)
                        })
                        .collect();
                    let scale: Vec<f32> = (0..t).map(|_| lcg(&mut seed)).collect();
                    for si in [None, Some(0), Some(1)] {
                        let shared = si.map(|i| {
                            let (g, u, d, n) = &sh[i];
                            Shared {
                                gate: g,
                                up: u,
                                down: d,
                                inter: *n,
                                act: Act::SiluClamp(0.7),
                                hook: None,
                                scale: (i == 0).then(|| scale.clone()),
                            }
                        });
                        let run = |batched: bool| {
                            let mut y = vec![0.5f32; t * h];
                            if batched {
                                assert!(ex.forward_batched(&x, t, h, &routes, &mut y, None, shared.as_ref()));
                            } else {
                                crate::kernels::set_moe_batched(false);
                                ex.forward(&x, t, h, &routes, &mut y, None, None, shared.as_ref());
                                crate::kernels::set_moe_batched(true);
                            }
                            y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        };
                        assert_eq!(run(true), run(false), "format {fmt} fused {fused} t={t} shared {si:?}");
                    }
                }
            }
        }
    }
}
