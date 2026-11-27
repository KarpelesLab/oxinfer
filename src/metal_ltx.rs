//! Whole LTX-2.5 transformer passes on Metal ([`LtxSession`], `crate::ltx::dit`): both streams'
//! rows stay in f32 device buffers through all 48 blocks; the projections run on the matrix-unit
//! GEMMs of the diffusion-transformer plans (scaled half inputs, ConvRot inputs rotated by
//! `dit_rot_half`), attention on `flash_attn128_mma` (video, 128-wide heads: half operands, or
//! `flash_attn128_mma_f32`) and `flash_attn64` (audio and the audio-video cross-attention,
//! 64-wide heads, f32), the norms, modulations, gates and residuals on the `ltx_*` row kernels of
//! `metal_ltx.metal`. One submission per forward (blocks spread over a few command buffers);
//! the rows after the last block come back for the output heads (run on the CPU).

use super::super::decode::{Fns, run_concurrent};
use super::super::*;
use super::{DBuf, Enc, U4, Xh, grow, release_all, runnable};
use crate::gpu::dit::{Lin, LtxAttn, LtxConnBlock, LtxIn, LtxModel, LtxPerturb, LtxSession};

/// Blocks per command buffer.
const BLOCKS_PER_CB: usize = 4;

struct Pipes {
    rms: Id,
    heads: Id,
    glog: Id,
    gapp: Id,
    gadd: Id,
    gelu: Id,
    rot: Id,
    flash64: Id,
}

pub(crate) struct MetalLtx {
    be: &'static MetalBackend,
    f: Fns,
    p: Pipes,
    bufs: Vec<DBuf>,
    /// The static constants (biases, norm and gate weights) uploaded for this model: its key and
    /// the offsets of every layer's (see [`Consts`]).
    stat: Option<(usize, Vec<BlockConsts>)>,
    /// The same for the text connector last run (its first query layer's address): per block its
    /// attention's and its feed-forward's biases.
    cstat: Option<(usize, Vec<ConnConsts>)>,
}

// SAFETY: as MetalDit.
unsafe impl Send for MetalLtx {}

#[derive(Clone, Copy)]
enum B {
    Hv = 0,
    Ha,
    Xf,
    Up,
    Xh,
    Rs,
    Xh2,
    Rs2,
    Q,
    K,
    V,
    Qa,
    Ka,
    Va,
    Att,
    Tmp,
    G,
    AvQv,
    AvKa,
    AvQa,
    AvKv,
    CtxV,
    CtxA,
    CtxM,
    Vcos,
    Vsin,
    Acos,
    Asin,
    VCcos,
    VCsin,
    ACcos,
    ACsin,
    ClsV,
    ClsA,
    ModV,
    ModA,
    AvV,
    AvA,
    GateV,
    GateA,
    PromptV,
    PromptA,
    Stat,
    Zero,
    U,
    Uh,
    Urs,
    /// A text connector's static constants, and ones (its residuals' unit gates).
    CStat,
    One,
}
const NBUF: usize = 49;

#[repr(C)]
struct Lrm {
    n: u32,
    ld_x: u32,
    norm: u32,
    out_half: u32,
    eps: f32,
    ld_tab: u32,
    sc_off: u32,
    sh_off: u32,
    uniform_cls: u32,
    _p: [u32; 3],
}
#[repr(C)]
struct Lhd {
    n: u32,
    ld_src: u32,
    hd: u32,
    norm: u32,
    rope: u32,
    out_half: u32,
    eps: f32,
    n_valid: u32,
}
#[repr(C)]
struct Lgd {
    n: u32,
    ld_y: u32,
    ld_tab: u32,
    off: u32,
    uniform_cls: u32,
    has_bias: u32,
    _p: [u32; 2],
}

/// Byte offsets into the static constants of one attention's biases, norms and gate.
#[derive(Clone, Copy, Default)]
struct AttnConsts {
    bq: usize,
    bk: usize,
    bv: usize,
    bo: Option<usize>,
    qn: usize,
    kn: usize,
    gw: usize,
    gb: usize,
}

#[derive(Clone, Copy, Default)]
struct StreamConsts {
    attn1: AttnConsts,
    attn2: AttnConsts,
    up_b: Option<usize>,
    down_b: Option<usize>,
}

/// A text connector block's attention constants and feed-forward biases.
#[derive(Clone, Copy, Default)]
struct ConnConsts {
    attn: AttnConsts,
    up_b: Option<usize>,
    down_b: Option<usize>,
}

#[derive(Clone, Copy, Default)]
struct BlockConsts {
    v: StreamConsts,
    a: StreamConsts,
    a2v: AttnConsts,
    v2a: AttnConsts,
}

impl MetalBackend {
    /// [`crate::gpu::Backend::ltx_session`].
    pub(crate) fn ltx_session_impl(&'static self) -> Option<Box<dyn LtxSession>> {
        self.mma().filter(|m| m.units)?;
        let pipe = |n: &str| self.named_pipeline(n);
        let p = Pipes {
            rms: pipe("ltx_rms_mod")?,
            heads: pipe("ltx_heads")?,
            glog: pipe("ltx_gate_logits")?,
            gapp: pipe("ltx_gate_apply")?,
            gadd: pipe("ltx_gated_add")?,
            gelu: pipe("ltx_gelu")?,
            rot: pipe("dit_rot_half")?,
            flash64: pipe("flash_attn64")?,
        };
        Some(Box::new(MetalLtx {
            be: self,
            f: Fns::new(self),
            p,
            bufs: (0..NBUF)
                .map(|_| DBuf {
                    id: std::ptr::null_mut(),
                    cap: 0,
                })
                .collect(),
            stat: None,
            cstat: None,
        }))
    }
}

impl Drop for MetalLtx {
    fn drop(&mut self) {
        release_all(self.be, &self.bufs);
    }
}

fn attn_layers<'l, 'a>(a: &'l LtxAttn<'a>) -> [&'l Lin<'a>; 4] {
    [&a.q, &a.k, &a.v, &a.o]
}

/// The static constants of `m`: (floats, per-block offsets).
fn statics(m: &LtxModel) -> (Vec<f32>, Vec<BlockConsts>) {
    let mut c: Vec<f32> = Vec::new();
    let mut put = |v: &[f32]| -> usize {
        let o = c.len();
        c.extend_from_slice(v);
        c.resize(c.len().next_multiple_of(64), 0.0);
        o * 4
    };
    let attn = |a: &LtxAttn, put: &mut dyn FnMut(&[f32]) -> usize| AttnConsts {
        bq: put(a.q.bias.unwrap_or(&[])),
        bk: put(a.k.bias.unwrap_or(&[])),
        bv: put(a.v.bias.unwrap_or(&[])),
        bo: a.o.bias.map(&mut *put),
        qn: put(a.qn),
        kn: put(a.kn),
        gw: put(a.gate_w),
        gb: put(a.gate_b),
    };
    let mut blocks = Vec::with_capacity(m.blocks.len());
    for b in &m.blocks {
        let stream = |s: &crate::gpu::dit::LtxStream, put: &mut dyn FnMut(&[f32]) -> usize| StreamConsts {
            attn1: attn(&s.attn1, put),
            attn2: attn(&s.attn2, put),
            up_b: s.ff.up.bias.map(&mut *put),
            down_b: s.ff.down.bias.map(&mut *put),
        };
        let v = stream(&b.v, &mut put);
        let a = stream(&b.a, &mut put);
        let a2v = attn(&b.a2v, &mut put);
        let v2a = attn(&b.v2a, &mut put);
        blocks.push(BlockConsts { v, a, a2v, v2a });
    }
    (c, blocks)
}

impl MetalLtx {
    fn ensure(&mut self, b: B, bytes: usize) -> bool {
        grow(self.be, &mut self.bufs[b as usize], bytes)
    }

    fn host(&self, b: B) -> *mut f32 {
        self.f.contents(self.bufs[b as usize].id) as *mut f32
    }

    /// Copies `v` to the start of buffer `b` (grown to hold it).
    fn upload(&mut self, b: B, v: &[f32]) -> bool {
        if !self.ensure(b, (v.len() * 4).max(256)) {
            return false;
        }
        // SAFETY: the buffer holds at least v.len() floats; the device is idle.
        unsafe { super::super::par_copy(self.host(b), v.as_ptr(), v.len()) };
        true
    }
}

impl LtxSession for MetalLtx {
    fn connector(
        &mut self,
        blocks: &[LtxConnBlock],
        x: &[f32],
        rows: usize,
        cos: &[f32],
        sin: &[f32],
        eps: f32,
        half_attention: bool,
        out: &mut Vec<f32>,
    ) -> bool {
        self.connector_impl(blocks, x, rows, cos, sin, eps, half_attention, out)
    }

    fn forward(&mut self, m: &LtxModel, v: &LtxIn, a: &LtxIn, ptb: &LtxPerturb, vout: &mut Vec<f32>, aout: &mut Vec<f32>) -> bool {
        let be = self.be;
        let (dv, da) = (m.dim, m.audio_dim);
        if m.blocks.is_empty() || v.rows == 0 || a.rows == 0 {
            return false;
        }
        // Every layer on the GEMMs, the inputs of an attention's q / k / v alike (rotated or not),
        // and the head widths the kernels take.
        for b in &m.blocks {
            let at = [&b.v.attn1, &b.v.attn2, &b.a.attn1, &b.a.attn2, &b.a2v, &b.v2a];
            for x in at {
                if !attn_layers(x).iter().all(|l| runnable(be, l)) || x.k.rot != x.v.rot || x.q.rot != x.k.rot || !matches!(x.hd, 64 | 128)
                {
                    return false;
                }
                if x.q.bias.is_none() || x.k.bias.is_none() || x.v.bias.is_none() || x.gate_w.len() != x.heads * x.q.n_in {
                    return false;
                }
            }
            for s in [&b.v, &b.a] {
                if !runnable(be, &s.ff.up) || !runnable(be, &s.ff.down) {
                    return false;
                }
            }
        }
        // The widest LoRA side path (its padded rank).
        let rank = m
            .blocks
            .iter()
            .flat_map(|b| {
                [&b.v.attn1, &b.v.attn2, &b.a.attn1, &b.a.attn2, &b.a2v, &b.v2a]
                    .into_iter()
                    .flat_map(attn_layers)
                    .chain([&b.v.ff.up, &b.v.ff.down, &b.a.ff.up, &b.a.ff.down])
                    .collect::<Vec<_>>()
            })
            .filter_map(|l| l.side.as_ref().map(|(d, _)| d.n_out))
            .max()
            .unwrap_or(64);
        let rows = |n: usize| n.next_multiple_of(MMA_TM);
        let (rv, ra, rc) = (rows(v.rows), rows(a.rows), rows(v.n_ctx.max(a.n_ctx)));
        let rmax = rv.max(ra).max(rc);
        let wide = 4 * dv.max(da);
        let dmax = dv.max(da);
        let ok = self.ensure(B::Hv, rv * dv * 4)
            && self.ensure(B::Ha, ra * da * 4)
            && self.ensure(B::Xf, rmax * dmax * 4)
            && self.ensure(B::Up, rmax * wide * 4)
            && self.ensure(B::Xh, rmax * wide * 2)
            && self.ensure(B::Rs, rmax * 4)
            && self.ensure(B::Xh2, rmax * dmax * 2)
            && self.ensure(B::Rs2, rmax * 4)
            && self.ensure(B::Q, rmax * dmax * 4)
            && self.ensure(B::K, rmax * dmax * 4)
            && self.ensure(B::V, rmax * dmax * 4)
            && self.ensure(B::Qa, rmax * dmax * 4)
            && self.ensure(B::Ka, rmax * dmax * 4)
            && self.ensure(B::Va, rmax * dmax * 4)
            && self.ensure(B::Att, rmax * dmax * 4)
            && self.ensure(B::Tmp, rmax * dmax * 4)
            && self.ensure(B::G, rmax * 64 * 4)
            && self.ensure(B::AvQv, rv * dv * 4)
            && self.ensure(B::AvKa, ra * da * 4)
            && self.ensure(B::AvQa, ra * da * 4)
            && self.ensure(B::AvKv, rv * dv * 4)
            && self.ensure(B::CtxM, rc * dmax * 4)
            && self.ensure(B::Zero, wide * 4)
            && self.ensure(B::U, rmax * rank * 4)
            && self.ensure(B::Uh, rmax * rank * 2)
            && self.ensure(B::Urs, rmax * 4);
        if !ok {
            return false;
        }
        // Static constants: once per model.
        let key = m.blocks[0].v.attn1.q.dev as *const _ as *const u8 as usize;
        if self.stat.as_ref().map(|s| s.0) != Some(key) {
            let (c, offs) = statics(m);
            if !self.upload(B::Stat, &c) {
                return false;
            }
            self.stat = Some((key, offs));
        }
        // Per-forward inputs.
        let pad = |x: &[f32], n: usize, r: usize| -> Vec<f32> {
            let mut p = x.to_vec();
            p.resize(r * n, 0.0);
            p
        };
        let clsf = |c: &[u32], r: usize| -> Vec<f32> {
            let mut p: Vec<f32> = c.iter().map(|&x| f32::from_bits(x)).collect();
            p.resize(r, 0.0);
            p
        };
        let ok = self.upload(B::Hv, &pad(v.x, dv, rv))
            && self.upload(B::Ha, &pad(a.x, da, ra))
            && self.upload(B::CtxV, &pad(v.ctx, dv, rc))
            && self.upload(B::CtxA, &pad(a.ctx, da, rc))
            && self.upload(B::Vcos, &pad(v.cos, dv / 2, rv))
            && self.upload(B::Vsin, &pad(v.sin, dv / 2, rv))
            && self.upload(B::Acos, &pad(a.cos, da / 2, ra))
            && self.upload(B::Asin, &pad(a.sin, da / 2, ra))
            && self.upload(B::VCcos, &pad(v.ccos, da / 2, rv))
            && self.upload(B::VCsin, &pad(v.csin, da / 2, rv))
            && self.upload(B::ACcos, &pad(a.ccos, da / 2, ra))
            && self.upload(B::ACsin, &pad(a.csin, da / 2, ra))
            && self.upload(B::ClsV, &clsf(v.cls, rv))
            && self.upload(B::ClsA, &clsf(a.cls, ra))
            && self.upload(B::ModV, v.mods)
            && self.upload(B::ModA, a.mods)
            && self.upload(B::AvV, v.av)
            && self.upload(B::AvA, a.av)
            && self.upload(B::GateV, v.av_gate)
            && self.upload(B::GateA, a.av_gate)
            && self.upload(B::PromptV, v.prompt)
            && self.upload(B::PromptA, a.prompt);
        if !ok {
            return false;
        }
        // SAFETY: Zero holds `wide` floats.
        unsafe { std::ptr::write_bytes(self.host(B::Zero), 0, wide) };
        let ids: [Id; NBUF] = std::array::from_fn(|i| self.bufs[i].id);
        let id = |b: B| ids[b as usize];
        let at = |b: B, bytes: usize| (id(b), bytes);
        let p = &self.p;
        let mma = be.mma.as_ref().unwrap();
        let stat = id(B::Stat);
        let offs = &self.stat.as_ref().unwrap().1;
        let eps = m.eps;
        let half_att = m.half_attention;

        // y = W x (+ the side path `up (down x)`), x from `input`.
        let gemm = |en: &Enc, l: &Lin, x: Xh, r: usize, y: (Id, usize), ldy: usize| {
            en.gemm(l, x, r, y, ldy, false);
            if let Some((dn, up)) = &l.side {
                let k = dn.n_out;
                let u = Xh {
                    xh: at(B::Uh, 0),
                    rs: at(B::Urs, 0),
                };
                en.gemm(dn, x, r, at(B::U, 0), k, false);
                en.to_half(at(B::U, 0), k, k, r, u);
                en.gemm(up, u, r, y, ldy, true);
            }
        };
        // Rows `src` (f32, n wide, `r` rows) as the input of layer `l` in (xh, rs): rotated in
        // place first when `l` is stored rotated.
        let input = |en: &Enc, l: &Lin, src: B, n: usize, r: usize, xh: B, rs: B| -> Xh {
            let e = en.e;
            let x = Xh {
                xh: at(xh, 0),
                rs: at(rs, 0),
            };
            if l.rot.is_some() {
                e.pipe(p.rot);
                e.buf(id(src), 0, 0);
                e.buf(id(xh), 0, 1);
                e.buf(id(rs), 0, 2);
                e.bytes(&U4([n as u32, 0, 0, 0]), 3);
                e.dispatch((r, 1), 256);
                e.barrier();
            } else {
                en.to_half(at(src, 0), n, n, r, x);
            }
            x
        };
        // `ltx_rms_mod`: rows of `src` (n wide, `r` rows) into f32 rows of `dst`.
        #[allow(clippy::too_many_arguments)]
        let rms = |en: &Enc,
                   src: B,
                   dst: B,
                   n: usize,
                   r: usize,
                   tab: (B, usize),
                   ld_tab: usize,
                   sc: usize,
                   sh: usize,
                   cls: Option<B>,
                   norm: bool| {
            let e = en.e;
            e.pipe(p.rms);
            e.buf(id(src), 0, 0);
            e.buf(id(tab.0), tab.1, 1);
            e.buf(id(cls.unwrap_or(B::ClsV)), 0, 2);
            e.buf(id(dst), 0, 3);
            e.buf(id(B::Rs), 0, 4);
            e.bytes(
                &Lrm {
                    n: n as u32,
                    ld_x: n as u32,
                    norm: norm as u32,
                    out_half: 0,
                    eps,
                    ld_tab: ld_tab as u32,
                    sc_off: sc as u32,
                    sh_off: sh as u32,
                    uniform_cls: cls.is_none() as u32,
                    _p: [0; 3],
                },
                5,
            );
            e.dispatch((r, 1), 256);
        };
        #[allow(clippy::too_many_arguments)]
        let heads = |en: &Enc,
                     src: B,
                     dst: B,
                     n: usize,
                     hd: usize,
                     bias: usize,
                     norm: Option<usize>,
                     rope: Option<(B, B)>,
                     half: bool,
                     valid: usize,
                     r: usize| {
            let e = en.e;
            e.pipe(p.heads);
            e.buf(id(src), 0, 0);
            e.buf(id(dst), 0, 1);
            e.buf(stat, bias, 2);
            e.buf(stat, norm.unwrap_or(bias), 3);
            let (c, s) = rope.unwrap_or((B::Vcos, B::Vsin));
            e.buf(id(c), 0, 4);
            e.buf(id(s), 0, 5);
            e.bytes(
                &Lhd {
                    n: n as u32,
                    ld_src: n as u32,
                    hd: hd as u32,
                    norm: norm.is_some() as u32,
                    rope: rope.is_some() as u32,
                    out_half: half as u32,
                    eps,
                    n_valid: valid as u32,
                },
                6,
            );
            e.dispatch((r, 1), 256);
        };
        let gate_logits = |en: &Enc, x: B, n: usize, c: &AttnConsts, heads: usize, r: usize| {
            let e = en.e;
            e.pipe(p.glog);
            e.buf(id(x), 0, 0);
            e.buf(stat, c.gw, 1);
            e.buf(stat, c.gb, 2);
            e.buf(id(B::G), 0, 3);
            e.bytes(&U4([n as u32, n as u32, heads as u32, 0]), 4);
            e.dispatch((r, 1), 256);
        };
        // h[0..n_valid] += (y + b) * gate.
        #[allow(clippy::too_many_arguments)]
        let gated =
            |en: &Enc, h: B, n: usize, valid: usize, bias: Option<usize>, tab: (B, usize), ld_tab: usize, off: usize, cls: Option<B>| {
                let e = en.e;
                e.pipe(p.gadd);
                e.buf(id(h), 0, 0);
                e.buf(id(B::Tmp), 0, 1);
                e.buf(stat, bias.unwrap_or(0), 2);
                e.buf(id(tab.0), tab.1, 3);
                e.buf(id(cls.unwrap_or(B::ClsV)), 0, 4);
                e.bytes(
                    &Lgd {
                        n: n as u32,
                        ld_y: n as u32,
                        ld_tab: ld_tab as u32,
                        off: off as u32,
                        uniform_cls: cls.is_none() as u32,
                        has_bias: bias.is_some() as u32,
                        _p: [0; 2],
                    },
                    5,
                );
                e.dispatch(((n / 4).div_ceil(256), valid), 256);
                e.barrier();
            };
        // The attention of queries `x` (f32 rows of q_dim in `qsrc`, `nq` valid of `rq`) over
        // key / value inputs `kv` (f32 rows of kv_dim, `nk` valid of `rk`; None: the queries'
        // own), the result (out projection, bias left for the residual) into Tmp.
        #[allow(clippy::too_many_arguments)]
        let attention = |en: &Enc,
                         at_: &LtxAttn,
                         c: &AttnConsts,
                         qsrc: B,
                         nq: usize,
                         rq: usize,
                         kv: Option<(B, usize, usize)>,
                         qrope: Option<(B, B)>,
                         krope: Option<(B, B)>| {
            let e = en.e;
            let inner = at_.heads * at_.hd;
            let q_dim = at_.q.n_in;
            gate_logits(en, qsrc, q_dim, c, at_.heads, nq);
            e.barrier();
            let xq = input(en, &at_.q, qsrc, q_dim, rq, B::Xh, B::Rs);
            gemm(en, &at_.q, xq, rq, at(B::Q, 0), inner);
            let (nk, rk) = match kv {
                None => {
                    gemm(en, &at_.k, xq, rq, at(B::K, 0), inner);
                    gemm(en, &at_.v, xq, rq, at(B::V, 0), inner);
                    (nq, rq)
                }
                Some((src, nk, rk)) => {
                    let xk = input(en, &at_.k, src, at_.k.n_in, rk, B::Xh2, B::Rs2);
                    gemm(en, &at_.k, xk, rk, at(B::K, 0), inner);
                    gemm(en, &at_.v, xk, rk, at(B::V, 0), inner);
                    (nk, rk)
                }
            };
            let half = at_.hd == 128 && half_att;
            heads(en, B::Q, B::Qa, inner, at_.hd, c.bq, Some(c.qn), qrope, half, nq, rq);
            heads(en, B::K, B::Ka, inner, at_.hd, c.bk, Some(c.kn), krope, half, nk, rk);
            heads(en, B::V, B::Va, inner, at_.hd, c.bv, None, None, half, nk, rk);
            e.barrier();
            let scale = (1.0 / (at_.hd as f32).sqrt()).to_bits();
            if at_.hd == 128 {
                e.pipe(if half { mma.flash128 } else { mma.flash128_f32 });
                e.buf(id(B::Qa), 0, 0);
                e.buf(id(B::Ka), 0, 1);
                e.buf(id(B::Va), 0, 2);
                e.buf(id(B::Att), 0, 3);
                e.bytes(&U4([nq as u32, nk as u32, inner as u32, scale]), 4);
                e.dispatch((nq.div_ceil(64), at_.heads), 128);
            } else {
                e.pipe(p.flash64);
                e.buf(id(B::Qa), 0, 0);
                e.buf(id(B::Ka), 0, 1);
                e.buf(id(B::Va), 0, 2);
                e.buf(id(B::Att), 0, 3);
                e.bytes(&U4([nq as u32, nk as u32, inner as u32, scale]), 4);
                e.dispatch((nq.div_ceil(32), at_.heads), 128);
            }
            e.barrier();
            e.pipe(p.gapp);
            e.buf(id(B::Att), 0, 0);
            e.buf(id(B::G), 0, 1);
            e.bytes(&U4([inner as u32, at_.hd as u32, at_.heads as u32, 0]), 2);
            e.dispatch((inner.div_ceil(256), nq), 256);
            e.barrier();
            let xo = input(en, &at_.o, B::Att, inner, rq, B::Xh, B::Rs);
            gemm(en, &at_.o, xo, rq, at(B::Tmp, 0), at_.o.n_out);
        };

        // A skipped self-attention (STG): out_proj(v_proj(x) + bias) into Tmp, no gate.
        let passthrough = |en: &Enc, at_: &LtxAttn, c: &AttnConsts, src: B, n: usize, r: usize| {
            let inner = at_.heads * at_.hd;
            let xv = input(en, &at_.v, src, at_.v.n_in, r, B::Xh, B::Rs);
            gemm(en, &at_.v, xv, r, at(B::V, 0), inner);
            heads(en, B::V, B::Att, inner, at_.hd, c.bv, None, None, false, n, r);
            en.e.barrier();
            let xo = input(en, &at_.o, B::Att, inner, r, B::Xh, B::Rs);
            gemm(en, &at_.o, xo, r, at(B::Tmp, 0), at_.o.n_out);
        };
        let nb = m.blocks.len();
        let skip_self = |vid: bool, i: usize| {
            let s = if vid { &ptb.video_self } else { &ptb.audio_self };
            s.get(i).copied().unwrap_or(false)
        };
        let parts: Vec<std::ops::Range<usize>> = (0..nb.div_ceil(BLOCKS_PER_CB))
            .map(|c| c * BLOCKS_PER_CB..((c + 1) * BLOCKS_PER_CB).min(nb))
            .collect();
        let (ok, _) = run_concurrent(be, &self.f, &parts, |e, range| {
            let en = Enc { e, be };
            for i in range {
                let b = &m.blocks[i];
                let bc = &offs[i];
                for (vid, s, sc, sin) in [(true, &b.v, &bc.v, v), (false, &b.a, &bc.a, a)] {
                    let (h, d, r, n) = if vid {
                        (B::Hv, dv, rv, sin.rows)
                    } else {
                        (B::Ha, da, ra, sin.rows)
                    };
                    let cls = Some(if vid { B::ClsV } else { B::ClsA });
                    let mods = if vid { B::ModV } else { B::ModA };
                    let ld = 9 * d;
                    let tab = (mods, i * sin.classes * ld * 4);
                    let rope = if vid { (B::Vcos, B::Vsin) } else { (B::Acos, B::Asin) };
                    // Self-attention.
                    rms(&en, h, B::Xf, d, r, tab, ld, d, 0, cls, true);
                    e.barrier();
                    if skip_self(vid, i) {
                        passthrough(&en, &s.attn1, &sc.attn1, B::Xf, n, r);
                    } else {
                        attention(&en, &s.attn1, &sc.attn1, B::Xf, n, r, None, Some(rope), Some(rope));
                    }
                    gated(&en, h, d, n, sc.attn1.bo, tab, ld, 2 * d, cls);
                    // Text cross-attention: the context's own shift and scale.
                    rms(&en, h, B::Xf, d, r, tab, ld, 7 * d, 6 * d, cls, true);
                    let (ctx, prompt) = if vid { (B::CtxV, B::PromptV) } else { (B::CtxA, B::PromptA) };
                    rms(
                        &en,
                        ctx,
                        B::CtxM,
                        d,
                        sin.n_ctx.next_multiple_of(MMA_TM),
                        (prompt, i * 2 * d * 4),
                        2 * d,
                        d,
                        0,
                        None,
                        false,
                    );
                    e.barrier();
                    attention(&en, &s.attn2, &sc.attn2, B::Xf, n, r, Some((B::CtxM, sin.n_ctx, rc)), None, None);
                    gated(&en, h, d, n, sc.attn2.bo, tab, ld, 8 * d, cls);
                }
                // Audio-video cross-attention over both streams as they are now.
                if !ptb.cross {
                    let (tv, ta) = ((B::AvV, i * v.classes * 4 * dv * 4), (B::AvA, i * a.classes * 4 * da * 4));
                    rms(&en, B::Hv, B::AvQv, dv, rv, tv, 4 * dv, 0, dv, Some(B::ClsV), true);
                    rms(&en, B::Ha, B::AvKa, da, ra, ta, 4 * da, 0, da, Some(B::ClsA), true);
                    rms(&en, B::Ha, B::AvQa, da, ra, ta, 4 * da, 2 * da, 3 * da, Some(B::ClsA), true);
                    rms(&en, B::Hv, B::AvKv, dv, rv, tv, 4 * dv, 2 * dv, 3 * dv, Some(B::ClsV), true);
                    e.barrier();
                    attention(
                        &en,
                        &b.a2v,
                        &bc.a2v,
                        B::AvQv,
                        v.rows,
                        rv,
                        Some((B::AvKa, a.rows, ra)),
                        Some((B::VCcos, B::VCsin)),
                        Some((B::ACcos, B::ACsin)),
                    );
                    gated(&en, B::Hv, dv, v.rows, bc.a2v.bo, (B::GateV, i * dv * 4), dv, 0, None);
                    attention(
                        &en,
                        &b.v2a,
                        &bc.v2a,
                        B::AvQa,
                        a.rows,
                        ra,
                        Some((B::AvKv, v.rows, rv)),
                        Some((B::ACcos, B::ACsin)),
                        Some((B::VCcos, B::VCsin)),
                    );
                    gated(&en, B::Ha, da, a.rows, bc.v2a.bo, (B::GateA, i * da * 4), da, 0, None);
                }
                // Feed-forwards.
                for (vid, s, sc, sin) in [(true, &b.v, &bc.v, v), (false, &b.a, &bc.a, a)] {
                    let (h, d, r, n) = if vid {
                        (B::Hv, dv, rv, sin.rows)
                    } else {
                        (B::Ha, da, ra, sin.rows)
                    };
                    let cls = Some(if vid { B::ClsV } else { B::ClsA });
                    let ld = 9 * d;
                    let tab = (if vid { B::ModV } else { B::ModA }, i * sin.classes * ld * 4);
                    rms(&en, h, B::Xf, d, r, tab, ld, 4 * d, 3 * d, cls, true);
                    e.barrier();
                    let xu = input(&en, &s.ff.up, B::Xf, d, r, B::Xh, B::Rs);
                    let f = s.ff.up.n_out;
                    gemm(&en, &s.ff.up, xu, r, at(B::Up, 0), f);
                    e.pipe(p.gelu);
                    e.buf(id(B::Up), 0, 0);
                    e.buf(stat, sc.up_b.unwrap_or(0), 1);
                    e.bytes(&U4([f as u32, sc.up_b.is_some() as u32, 0, 0]), 2);
                    e.dispatch(((f / 4).div_ceil(256), r), 256);
                    e.barrier();
                    let xd = input(&en, &s.ff.down, B::Up, f, r, B::Xh, B::Rs);
                    gemm(&en, &s.ff.down, xd, r, at(B::Tmp, 0), d);
                    gated(&en, h, d, n, sc.down_b, tab, ld, 5 * d, cls);
                }
            }
        });
        if !ok {
            return false;
        }
        vout.resize(v.rows * dv, 0.0);
        aout.resize(a.rows * da, 0.0);
        // SAFETY: the buffers hold at least these rows.
        unsafe {
            super::super::par_copy(vout.as_mut_ptr(), self.host(B::Hv), v.rows * dv);
            super::super::par_copy(aout.as_mut_ptr(), self.host(B::Ha), a.rows * da);
        }
        true
    }
}

/// The static constants of a text connector: (floats, per-block offsets).
fn conn_statics(blocks: &[LtxConnBlock]) -> (Vec<f32>, Vec<ConnConsts>) {
    let mut c: Vec<f32> = Vec::new();
    let mut put = |v: &[f32]| -> usize {
        let o = c.len();
        c.extend_from_slice(v);
        c.resize(c.len().next_multiple_of(64), 0.0);
        o * 4
    };
    let mut out = Vec::with_capacity(blocks.len());
    for b in blocks {
        let a = &b.attn;
        out.push(ConnConsts {
            attn: AttnConsts {
                bq: put(a.q.bias.unwrap_or(&[])),
                bk: put(a.k.bias.unwrap_or(&[])),
                bv: put(a.v.bias.unwrap_or(&[])),
                bo: a.o.bias.map(&mut put),
                qn: put(a.qn),
                kn: put(a.kn),
                gw: put(a.gate_w),
                gb: put(a.gate_b),
            },
            up_b: b.ff.up.bias.map(&mut put),
            down_b: b.ff.down.bias.map(&mut put),
        });
    }
    (c, out)
}

impl MetalLtx {
    /// [`LtxSession::connector`]: the blocks in one submission, the rows in `Hv`, the norms'
    /// modulation a zero table and the residuals' gates a table of ones (the transformer
    /// blocks' kernels with nothing to modulate).
    #[allow(clippy::too_many_arguments)]
    fn connector_impl(
        &mut self,
        blocks: &[LtxConnBlock],
        x: &[f32],
        rows: usize,
        cos: &[f32],
        sin: &[f32],
        eps: f32,
        half_att: bool,
        out: &mut Vec<f32>,
    ) -> bool {
        let be = self.be;
        let Some(b0) = blocks.first() else { return false };
        let d = b0.attn.q.n_in;
        let f = b0.ff.up.n_out;
        if rows == 0 || x.len() != rows * d || cos.len() != rows * d / 2 || sin.len() != cos.len() {
            return false;
        }
        for b in blocks {
            let a = &b.attn;
            if !attn_layers(a).iter().all(|l| runnable(be, l))
                || a.k.rot != a.v.rot
                || a.q.rot != a.k.rot
                || !matches!(a.hd, 64 | 128)
                || a.heads * a.hd != d
                || a.q.n_in != d
                || a.q.bias.is_none()
                || a.k.bias.is_none()
                || a.v.bias.is_none()
                || a.gate_w.len() != a.heads * d
                || !runnable(be, &b.ff.up)
                || !runnable(be, &b.ff.down)
                || b.ff.up.n_out != f
            {
                return false;
            }
        }
        let rank = blocks
            .iter()
            .flat_map(|b| attn_layers(&b.attn).into_iter().chain([&b.ff.up, &b.ff.down]))
            .filter_map(|l| l.side.as_ref().map(|(dn, _)| dn.n_out))
            .max()
            .unwrap_or(64);
        let r = rows.next_multiple_of(MMA_TM);
        let ok = self.ensure(B::Hv, r * d * 4)
            && self.ensure(B::Xf, r * d * 4)
            && self.ensure(B::Up, r * f * 4)
            && self.ensure(B::Xh, r * f.max(d) * 2)
            && self.ensure(B::Rs, r * 4)
            && self.ensure(B::Xh2, r * d * 2)
            && self.ensure(B::Rs2, r * 4)
            && self.ensure(B::Q, r * d * 4)
            && self.ensure(B::K, r * d * 4)
            && self.ensure(B::V, r * d * 4)
            && self.ensure(B::Qa, r * d * 4)
            && self.ensure(B::Ka, r * d * 4)
            && self.ensure(B::Va, r * d * 4)
            && self.ensure(B::Att, r * d * 4)
            && self.ensure(B::Tmp, r * d * 4)
            && self.ensure(B::G, r * 64 * 4)
            && self.ensure(B::Zero, f.max(d) * 4)
            && self.ensure(B::U, r * rank * 4)
            && self.ensure(B::Uh, r * rank * 2)
            && self.ensure(B::Urs, r * 4);
        if !ok {
            return false;
        }
        let key = b0.attn.q.dev as *const _ as *const u8 as usize;
        if self.cstat.as_ref().map(|s| s.0) != Some(key) {
            let (c, offs) = conn_statics(blocks);
            if !self.upload(B::CStat, &c) {
                return false;
            }
            self.cstat = Some((key, offs));
        }
        let pad = |v: &[f32], n: usize| -> Vec<f32> {
            let mut p = v.to_vec();
            p.resize(r * n, 0.0);
            p
        };
        let ok = self.upload(B::Hv, &pad(x, d))
            && self.upload(B::Vcos, &pad(cos, d / 2))
            && self.upload(B::Vsin, &pad(sin, d / 2))
            && self.upload(B::One, &vec![1.0; d]);
        if !ok {
            return false;
        }
        // SAFETY: Zero holds f.max(d) floats; the device is idle.
        unsafe { std::ptr::write_bytes(self.host(B::Zero), 0, f.max(d)) };
        let ids: [Id; NBUF] = std::array::from_fn(|i| self.bufs[i].id);
        let id = |b: B| ids[b as usize];
        let at = |b: B, bytes: usize| (id(b), bytes);
        let p = &self.p;
        let mma = be.mma.as_ref().unwrap();
        let stat = id(B::CStat);
        let offs = &self.cstat.as_ref().unwrap().1;
        let gemm = |en: &Enc, l: &Lin, x: Xh, r: usize, y: (Id, usize), ldy: usize| {
            en.gemm(l, x, r, y, ldy, false);
            if let Some((dn, up)) = &l.side {
                let k = dn.n_out;
                let u = Xh {
                    xh: at(B::Uh, 0),
                    rs: at(B::Urs, 0),
                };
                en.gemm(dn, x, r, at(B::U, 0), k, false);
                en.to_half(at(B::U, 0), k, k, r, u);
                en.gemm(up, u, r, y, ldy, true);
            }
        };
        let input = |en: &Enc, l: &Lin, src: B, n: usize, r: usize, xh: B, rs: B| -> Xh {
            let e = en.e;
            let x = Xh {
                xh: at(xh, 0),
                rs: at(rs, 0),
            };
            if l.rot.is_some() {
                e.pipe(p.rot);
                e.buf(id(src), 0, 0);
                e.buf(id(xh), 0, 1);
                e.buf(id(rs), 0, 2);
                e.bytes(&U4([n as u32, 0, 0, 0]), 3);
                e.dispatch((r, 1), 256);
                e.barrier();
            } else {
                en.to_half(at(src, 0), n, n, r, x);
            }
            x
        };
        // Rows of Hv RMS-normed (the zero table: no modulation) into Xf.
        let norm = |en: &Enc| {
            let e = en.e;
            e.pipe(p.rms);
            e.buf(id(B::Hv), 0, 0);
            e.buf(id(B::Zero), 0, 1);
            e.buf(id(B::Zero), 0, 2);
            e.buf(id(B::Xf), 0, 3);
            e.buf(id(B::Rs), 0, 4);
            e.bytes(
                &Lrm {
                    n: d as u32,
                    ld_x: d as u32,
                    norm: 1,
                    out_half: 0,
                    eps,
                    ld_tab: 0,
                    sc_off: 0,
                    sh_off: 0,
                    uniform_cls: 1,
                    _p: [0; 3],
                },
                5,
            );
            e.dispatch((r, 1), 256);
            e.barrier();
        };
        // Hv[0..rows] += Tmp + bias (gates of one).
        let residual = |en: &Enc, bias: Option<usize>| {
            let e = en.e;
            e.pipe(p.gadd);
            e.buf(id(B::Hv), 0, 0);
            e.buf(id(B::Tmp), 0, 1);
            e.buf(stat, bias.unwrap_or(0), 2);
            e.buf(id(B::One), 0, 3);
            e.buf(id(B::One), 0, 4);
            e.bytes(
                &Lgd {
                    n: d as u32,
                    ld_y: d as u32,
                    ld_tab: 0,
                    off: 0,
                    uniform_cls: 1,
                    has_bias: bias.is_some() as u32,
                    _p: [0; 2],
                },
                5,
            );
            e.dispatch(((d / 4).div_ceil(256), rows), 256);
            e.barrier();
        };
        #[allow(clippy::too_many_arguments)]
        let heads = |en: &Enc, src: B, dst: B, n: usize, hd: usize, bias: usize, nw: Option<usize>, rope: bool, half: bool| {
            let e = en.e;
            e.pipe(p.heads);
            e.buf(id(src), 0, 0);
            e.buf(id(dst), 0, 1);
            e.buf(stat, bias, 2);
            e.buf(stat, nw.unwrap_or(bias), 3);
            e.buf(id(B::Vcos), 0, 4);
            e.buf(id(B::Vsin), 0, 5);
            e.bytes(
                &Lhd {
                    n: n as u32,
                    ld_src: n as u32,
                    hd: hd as u32,
                    norm: nw.is_some() as u32,
                    rope: rope as u32,
                    out_half: half as u32,
                    eps,
                    n_valid: rows as u32,
                },
                6,
            );
            e.dispatch((r, 1), 256);
        };
        let (ok, _) = run_concurrent(be, &self.f, std::slice::from_ref(&(0..blocks.len())), |e, range| {
            let en = Enc { e, be };
            for i in range {
                let b = &blocks[i];
                let (a, c) = (&b.attn, &offs[i].attn);
                // Self-attention.
                norm(&en);
                let inner = a.heads * a.hd;
                e.pipe(p.glog);
                e.buf(id(B::Xf), 0, 0);
                e.buf(stat, c.gw, 1);
                e.buf(stat, c.gb, 2);
                e.buf(id(B::G), 0, 3);
                e.bytes(&U4([d as u32, d as u32, a.heads as u32, 0]), 4);
                e.dispatch((rows, 1), 256);
                e.barrier();
                let xq = input(&en, &a.q, B::Xf, d, r, B::Xh, B::Rs);
                gemm(&en, &a.q, xq, r, at(B::Q, 0), inner);
                gemm(&en, &a.k, xq, r, at(B::K, 0), inner);
                gemm(&en, &a.v, xq, r, at(B::V, 0), inner);
                let half = a.hd == 128 && half_att;
                heads(&en, B::Q, B::Qa, inner, a.hd, c.bq, Some(c.qn), true, half);
                heads(&en, B::K, B::Ka, inner, a.hd, c.bk, Some(c.kn), true, half);
                heads(&en, B::V, B::Va, inner, a.hd, c.bv, None, false, half);
                e.barrier();
                let scale = (1.0 / (a.hd as f32).sqrt()).to_bits();
                e.pipe(if a.hd == 128 {
                    if half { mma.flash128 } else { mma.flash128_f32 }
                } else {
                    p.flash64
                });
                e.buf(id(B::Qa), 0, 0);
                e.buf(id(B::Ka), 0, 1);
                e.buf(id(B::Va), 0, 2);
                e.buf(id(B::Att), 0, 3);
                e.bytes(&U4([rows as u32, rows as u32, inner as u32, scale]), 4);
                e.dispatch((rows.div_ceil(if a.hd == 128 { 64 } else { 32 }), a.heads), 128);
                e.barrier();
                e.pipe(p.gapp);
                e.buf(id(B::Att), 0, 0);
                e.buf(id(B::G), 0, 1);
                e.bytes(&U4([inner as u32, a.hd as u32, a.heads as u32, 0]), 2);
                e.dispatch((inner.div_ceil(256), rows), 256);
                e.barrier();
                let xo = input(&en, &a.o, B::Att, inner, r, B::Xh, B::Rs);
                gemm(&en, &a.o, xo, r, at(B::Tmp, 0), d);
                residual(&en, c.bo);
                // Feed-forward.
                norm(&en);
                let xu = input(&en, &b.ff.up, B::Xf, d, r, B::Xh, B::Rs);
                gemm(&en, &b.ff.up, xu, r, at(B::Up, 0), f);
                let ub = offs[i].up_b;
                e.pipe(p.gelu);
                e.buf(id(B::Up), 0, 0);
                e.buf(stat, ub.unwrap_or(0), 1);
                e.bytes(&U4([f as u32, ub.is_some() as u32, 0, 0]), 2);
                e.dispatch(((f / 4).div_ceil(256), r), 256);
                e.barrier();
                let xd = input(&en, &b.ff.down, B::Up, f, r, B::Xh, B::Rs);
                gemm(&en, &b.ff.down, xd, r, at(B::Tmp, 0), d);
                residual(&en, offs[i].down_b);
            }
        });
        if !ok {
            return false;
        }
        out.resize(rows * d, 0.0);
        // SAFETY: Hv holds at least these rows.
        unsafe { super::super::par_copy(out.as_mut_ptr(), self.host(B::Hv), rows * d) };
        true
    }
}
