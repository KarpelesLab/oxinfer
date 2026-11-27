//! Whole MiniMax-H3 block stacks on Metal ([`H3Session`], `crate::minimax_h3::dit`): the packed
//! sequence (text, video and audio rows) stays in f32 device buffers through every block; the
//! projections run on the matrix-unit GEMMs of the diffusion-transformer plans (scaled half
//! inputs, or hi + lo pairs of them on the `*_x2` GEMMs with `InputPrecision::Split`; unit-scale
//! weight twins whose scales `h3_col_scale` applies to the outputs; ConvRot inputs rotated by
//! `dit_rot_half`; LoRA side paths), attention on
//! `flash_attn128_mma` (half operands), `dit_flash128_hilo` (queries and keys as hi + lo halves)
//! or `flash_attn128_mma_f32`, the AdaLN norms, the head norms with the 3-axis RoPE and the
//! SwiGLU on the `h3_*` kernels of `metal_h3.metal`, the gated residuals on `ltx_gated_add`. One
//! submission per forward (blocks spread over a few command buffers).

use super::super::decode::{Fns, run_concurrent};
use super::super::*;
use super::{DBuf, Enc, U4, Xh, grow, release_all, runnable};
use crate::gpu::dit::{AttnPrecision, CorrIn, CorrModel, CorrSession, H3In, H3Model, H3Session, InputPrecision, Lin, SplitLin};

/// Blocks per command buffer.
const BLOCKS_PER_CB: usize = 4;

struct Pipes {
    rms: Id,
    heads: Id,
    swiglu: Id,
    gadd: Id,
    rot: Id,
    flash64: Id,
    hilo: Option<Id>,
    split: Id,
    col: Id,
    side_gain: Id,
}

/// `metal_ltx.metal`'s `LSG`: a LoRA side path's run-time strengths over its rank columns.
#[repr(C)]
struct Lsg {
    k: u32,
    n: u32,
    start: [u32; 4],
    gain: [f32; 4],
}

pub(crate) struct MetalH3 {
    be: &'static MetalBackend,
    f: Fns,
    p: Pipes,
    bufs: Vec<DBuf>,
    /// The constants uploaded: the model's key and each block's offsets.
    stat: Option<(usize, Vec<HConsts>)>,
}

// SAFETY: as MetalDit.
unsafe impl Send for MetalH3 {}

#[derive(Clone, Copy)]
enum B {
    H = 0,
    Xf,
    Xh,
    Rs,
    Qkv,
    Q,
    K,
    V,
    Ql,
    Kl,
    Att,
    Tmp,
    Up,
    Key,
    Tab,
    Cos,
    Sin,
    Stat,
    U,
    Uh,
    Urs,
    Xl,
    Ul,
}
const NBUF: usize = 23;

#[repr(C)]
struct Hrm {
    n: u32,
    out_half: u32,
    eps: f32,
    ld_tab: u32,
    sc_off: u32,
    sh_off: u32,
    has_mod: u32,
    _p: u32,
}
#[repr(C)]
struct Hhd {
    ld_src: u32,
    base: u32,
    hstride: u32,
    heads: u32,
    norm: u32,
    rope: u32,
    half: u32,
    mode: u32,
    eps: f32,
    n_valid: u32,
    hd: u32,
    has_bias: u32,
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

impl MetalBackend {
    /// [`crate::gpu::Backend::h3_session`].
    pub(crate) fn h3_session_impl(&'static self) -> Option<Box<dyn H3Session>> {
        self.mma().filter(|m| m.units)?;
        let pipe = |n: &str| self.named_pipeline(n);
        let p = Pipes {
            rms: pipe("h3_rms_mod")?,
            heads: pipe("h3_heads")?,
            swiglu: pipe("h3_swiglu")?,
            gadd: pipe("ltx_gated_add")?,
            rot: pipe("dit_rot_half")?,
            flash64: pipe("flash_attn64")?,
            hilo: pipe("dit_flash128_hilo"),
            split: pipe("h3_split_half")?,
            col: pipe("h3_col_scale")?,
            side_gain: pipe("ltx_side_gain")?,
        };
        Some(Box::new(MetalH3 {
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
        }))
    }
}

impl Drop for MetalH3 {
    fn drop(&mut self) {
        release_all(self.be, &self.bufs);
    }
}

impl MetalH3 {
    fn ensure(&mut self, b: B, bytes: usize) -> bool {
        grow(self.be, &mut self.bufs[b as usize], bytes)
    }

    fn host(&self, b: B) -> *mut f32 {
        self.f.contents(self.bufs[b as usize].id) as *mut f32
    }

    /// Copies `v` to the start of buffer `b` (grown to hold it), zeros up to `len` floats after.
    fn upload(&mut self, b: B, v: &[f32], len: usize) -> bool {
        let len = len.max(v.len());
        if !self.ensure(b, (len * 4).max(256)) {
            return false;
        }
        // SAFETY: the buffer holds at least `len` floats; the device is idle.
        unsafe {
            super::super::par_copy(self.host(b), v.as_ptr(), v.len());
            std::ptr::write_bytes(self.host(b).add(v.len()), 0, len - v.len());
        }
        true
    }
}

/// Byte offsets of a block's constants in the static buffer.
#[derive(Clone, Copy, Default)]
struct HConsts {
    norm1: usize,
    norm2: usize,
    qn: usize,
    kn: usize,
    /// Biases of qkv, out, fc1, fc2.
    b: [Option<usize>; 4],
    /// The residual scales.
    scale: Option<(usize, usize)>,
    /// The output scales of qkv, out, fc1, fc2.
    os: [Option<usize>; 4],
}

impl H3Session for MetalH3 {
    fn forward(&mut self, m: &H3Model, inp: &H3In, out: &mut Vec<f32>) -> bool {
        let be = self.be;
        let (d, nh, hd, f, half) = (m.hidden, m.heads, m.head_dim, m.ffn, m.rope_half);
        let inner = nh * hd;
        let n = inp.rows;
        let ada = m.blocks.first().is_some_and(|b| b.scale.is_none());
        if m.blocks.is_empty()
            || n == 0
            || !matches!(hd, 64 | 128)
            || 2 * half > hd
            || !d.is_multiple_of(256)
            || !f.is_multiple_of(4)
            || m.blocks.iter().any(|b| b.scale.is_none() != ada)
        {
            return false;
        }
        if inp.x.len() != n * d || inp.cos.len() != n * half || inp.sin.len() != n * half {
            return false;
        }
        if ada
            && (inp.key.len() != n
                || inp.tables.len() != m.blocks.len()
                || inp.tables.iter().any(|t| t.len() != inp.sets * 6 * d)
                || inp.key.iter().any(|&k| k as usize >= inp.sets))
        {
            return false;
        }
        let mut rank = 64;
        for b in &m.blocks {
            for l in [&b.qkv, &b.out, &b.fc1, &b.fc2] {
                if !runnable(be, l) {
                    return false;
                }
                if let Some((dn, _)) = &l.side {
                    rank = rank.max(dn.n_out);
                }
            }
            if b.qkv.n_out != 3 * inner
                || b.qkv.n_in != d
                || b.out.n_in != inner
                || b.fc1.n_out != 2 * f
                || b.fc2.n_in != f
                || b.q_norm.len() != hd
                || b.k_norm.len() != hd
            {
                return false;
            }
        }
        // 64-wide heads: flash_attn64 (float32 operands).
        let attention = match m.attention {
            _ if hd == 64 => AttnPrecision::F32,
            AttnPrecision::Split if self.p.hilo.is_none() => AttnPrecision::F32,
            a => a,
        };
        // The projections' inputs as hi + lo halves.
        let pair = m.inputs == InputPrecision::Split;
        let r = n.next_multiple_of(MMA_TM);
        let wide = (2 * f).max(3 * inner);
        let ok = self.ensure(B::H, r * d * 4)
            && self.ensure(B::Xf, r * f.max(inner).max(d) * 4)
            && self.ensure(B::Xh, r * f.max(inner).max(d) * 2)
            && self.ensure(B::Rs, r * 4)
            && self.ensure(B::Qkv, r * 3 * inner * 4)
            && self.ensure(B::Q, r * inner * 4)
            && self.ensure(B::K, r * inner * 4)
            && self.ensure(B::V, r * inner * 4)
            && (attention != AttnPrecision::Split || self.ensure(B::Ql, r * inner * 2) && self.ensure(B::Kl, r * inner * 2))
            && self.ensure(B::Att, r * inner * 4)
            && self.ensure(B::Tmp, r * d * 4)
            && self.ensure(B::Up, r * wide * 4)
            && self.ensure(B::U, r * rank * 4)
            && self.ensure(B::Uh, r * rank * 2)
            && self.ensure(B::Urs, r * 4)
            && (!pair || self.ensure(B::Xl, r * f.max(inner).max(d) * 2) && self.ensure(B::Ul, r * rank * 2));
        if !ok {
            return false;
        }
        // The norm weights, biases and residual scales: once per model.
        let mkey = m.blocks[0].qkv.dev as *const _ as *const u8 as usize;
        if self.stat.as_ref().map(|s| s.0) != Some(mkey) {
            let mut c: Vec<f32> = Vec::new();
            let mut put = |v: &[f32]| -> usize {
                let o = c.len();
                c.extend_from_slice(v);
                c.resize(c.len().next_multiple_of(64), 0.0);
                o * 4
            };
            let offs: Vec<HConsts> = m
                .blocks
                .iter()
                .map(|b| HConsts {
                    norm1: put(b.norm1),
                    norm2: put(b.norm2),
                    qn: put(b.q_norm),
                    kn: put(b.k_norm),
                    b: [&b.qkv, &b.out, &b.fc1, &b.fc2].map(|l| l.bias.map(&mut put)),
                    scale: b.scale.map(|(s1, s2)| (put(s1), put(s2))),
                    os: b.out_scale.map(|s| s.map(&mut put)),
                })
                .collect();
            if !self.upload(B::Stat, &c, 0) {
                return false;
            }
            self.stat = Some((mkey, offs));
        }
        // Per-forward inputs: the rows, each row's set (padding rows read set 0), the angles, the
        // AdaLN tables of every block.
        let keys: Vec<f32> = inp.key.iter().map(|&k| f32::from_bits(k)).collect();
        let ld_tab = 6 * d;
        let tab_block = inp.sets * ld_tab;
        let tabs: Vec<f32> = inp.tables.concat();
        let ok = self.upload(B::H, inp.x, r * d)
            && self.upload(B::Key, &keys, r)
            && self.upload(B::Cos, inp.cos, r * half)
            && self.upload(B::Sin, inp.sin, r * half)
            && self.upload(B::Tab, &tabs, 0);
        if !ok {
            return false;
        }
        let ids: [Id; NBUF] = std::array::from_fn(|i| self.bufs[i].id);
        let id = |b: B| ids[b as usize];
        let at = |b: B, bytes: usize| (id(b), bytes);
        let p = &self.p;
        let mma = be.mma.as_ref().unwrap();
        let stat = id(B::Stat);
        let offs = &self.stat.as_ref().unwrap().1;
        let main_x = Xh {
            xh: at(B::Xh, 0),
            rs: at(B::Rs, 0),
        };

        // f32 rows `src` (nw wide) to hi + lo halves `hi`, `lo` with their scales `rs`.
        let to_pair = |en: &Enc, src: B, nw: usize, hi: B, lo: B, rs: B| {
            let e = en.e;
            e.pipe(p.split);
            e.buf(id(src), 0, 0);
            e.buf(id(hi), 0, 1);
            e.buf(id(lo), 0, 2);
            e.buf(id(rs), 0, 3);
            e.bytes(&U4([nw as u32, 0, 0, 0]), 4);
            e.dispatch((r, 1), 256);
            e.barrier();
        };
        // y = W x (+ the side path `up (down x)`), x = hi + lo with `pair` (the side path's
        // inner rows too).
        let gemm = |en: &Enc, l: &Lin, os: Option<usize>, y: (Id, usize), ldy: usize| {
            if pair {
                en.gemm_pair(l, main_x, at(B::Xl, 0), r, y, ldy, false);
            } else {
                en.gemm(l, main_x, r, y, ldy, false);
            }
            if let Some(so) = os {
                let e = en.e;
                e.pipe(p.col);
                e.buf(y.0, y.1, 0);
                e.buf(stat, so, 1);
                e.bytes(&U4([l.n_out as u32, ldy as u32, 0, 0]), 2);
                e.dispatch(((l.n_out / 4).div_ceil(256), r), 256);
                e.barrier();
            }
            if let Some((dn, up)) = &l.side {
                let k = dn.n_out;
                let u = Xh {
                    xh: at(B::Uh, 0),
                    rs: at(B::Urs, 0),
                };
                if pair {
                    en.gemm_pair(dn, main_x, at(B::Xl, 0), r, at(B::U, 0), k, false);
                } else {
                    en.gemm(dn, main_x, r, at(B::U, 0), k, false);
                }
                // The updates' strengths now (a switched adapter's gain).
                if !l.side_gains.is_empty() {
                    let mut g = Lsg {
                        k: k as u32,
                        n: l.side_gains.len().min(4) as u32,
                        start: [0; 4],
                        gain: [1.0; 4],
                    };
                    for (i, &(s, v)) in l.side_gains.iter().take(4).enumerate() {
                        (g.start[i], g.gain[i]) = (s as u32, v);
                    }
                    let e = en.e;
                    e.pipe(p.side_gain);
                    e.buf(id(B::U), 0, 0);
                    e.bytes(&g, 1);
                    e.dispatch((k.div_ceil(256), r), 256);
                    e.barrier();
                }
                if pair {
                    to_pair(en, B::U, k, B::Uh, B::Ul, B::Urs);
                    en.gemm_pair(up, u, at(B::Ul, 0), r, y, ldy, true);
                } else {
                    en.to_half(at(B::U, 0), k, k, r, u);
                    en.gemm(up, u, r, y, ldy, true);
                }
            }
        };
        // The input of `l` from f32 rows `src` (n wide, contiguous): rotated in place first when
        // `l` is stored rotated; hi + lo halves with `pair`.
        let input = |en: &Enc, l: &Lin, src: B, nw: usize| {
            let e = en.e;
            if l.rot.is_some() {
                e.pipe(p.rot);
                e.buf(id(src), 0, 0);
                e.buf(id(B::Xh), 0, 1);
                e.buf(id(B::Rs), 0, 2);
                e.bytes(&U4([nw as u32, 0, 0, 0]), 3);
                e.dispatch((r, 1), 256);
                e.barrier();
            } else if !pair {
                en.to_half(at(src, 0), nw, nw, r, main_x);
            }
            if pair {
                to_pair(en, src, nw, B::Xh, B::Xl, B::Rs);
            }
        };
        // The norm of the rows (with the AdaLN shift / scale) into the input of `l` (half rows
        // directly, or f32 rows rotated or split after).
        let norm_in = |en: &Enc, l: &Lin, w: usize, tab: usize, sc: usize, sh: usize| {
            let e = en.e;
            let rot = l.rot.is_some() || pair;
            e.pipe(p.rms);
            e.buf(id(B::H), 0, 0);
            e.buf(stat, w, 1);
            e.buf(id(B::Tab), tab, 2);
            e.buf(id(B::Key), 0, 3);
            e.buf(id(if rot { B::Xf } else { B::Xh }), 0, 4);
            e.buf(id(B::Rs), 0, 5);
            e.bytes(
                &Hrm {
                    n: d as u32,
                    out_half: !rot as u32,
                    eps: m.eps,
                    ld_tab: ld_tab as u32,
                    sc_off: sc as u32,
                    sh_off: sh as u32,
                    has_mod: ada as u32,
                    _p: 0,
                },
                6,
            );
            e.dispatch((r, 1), 256);
            e.barrier();
            if rot {
                input(en, l, B::Xf, d);
            }
        };
        // h[0..n] += gate * (Tmp + bias): the AdaLN gate `off` of each row's set, or a scale.
        let gated = |en: &Enc, tab: usize, off: usize, scale: Option<usize>, bias: Option<usize>| {
            let e = en.e;
            e.pipe(p.gadd);
            e.buf(id(B::H), 0, 0);
            e.buf(id(B::Tmp), 0, 1);
            match bias {
                Some(o) => e.buf(stat, o, 2),
                None => e.buf(id(B::Tmp), 0, 2),
            }
            match scale {
                Some(o) => e.buf(stat, o, 3),
                None => e.buf(id(B::Tab), tab, 3),
            }
            e.buf(id(B::Key), 0, 4);
            e.bytes(
                &Lgd {
                    n: d as u32,
                    ld_y: d as u32,
                    ld_tab: ld_tab as u32,
                    off: if scale.is_some() { 0 } else { off as u32 },
                    uniform_cls: scale.is_some() as u32,
                    has_bias: bias.is_some() as u32,
                    _p: [0; 2],
                },
                5,
            );
            e.dispatch(((d / 4).div_ceil(256), n), 256);
            e.barrier();
        };
        let mode = match attention {
            AttnPrecision::F32 => 0,
            AttnPrecision::Half => 1,
            AttnPrecision::Split => 2,
        };
        // Operand `which` (0 q, 1 k, 2 v) of the fused projection's rows into `dst`.
        let heads = |en: &Enc, which: usize, dst: B, lo: B, norm: Option<usize>, bias: Option<usize>| {
            let e = en.e;
            let (base, hstride) = if m.stacked { (which * inner, hd) } else { (which * hd, 3 * hd) };
            e.pipe(p.heads);
            e.buf(id(B::Qkv), 0, 0);
            e.buf(id(dst), 0, 1);
            e.buf(stat, norm.unwrap_or(0), 2);
            e.buf(id(B::Cos), 0, 3);
            e.buf(id(B::Sin), 0, 4);
            e.bytes(
                &Hhd {
                    ld_src: (3 * inner) as u32,
                    base: base as u32,
                    hstride: hstride as u32,
                    heads: nh as u32,
                    norm: norm.is_some() as u32,
                    rope: norm.is_some() as u32,
                    half: half as u32,
                    // Values plain half with hi / lo queries and keys.
                    mode: if mode == 2 && which == 2 { 1 } else { mode },
                    eps: m.qk_eps,
                    n_valid: n as u32,
                    hd: hd as u32,
                    has_bias: bias.is_some() as u32,
                },
                5,
            );
            e.buf(id(if mode == 2 { lo } else { dst }), 0, 6);
            e.buf(stat, bias.unwrap_or(0), 7);
            e.dispatch((r, nh), 32);
        };

        let blocks: Vec<usize> = (0..m.blocks.len()).filter(|i| !inp.skip.contains(i)).collect();
        let parts: Vec<std::ops::Range<usize>> = (0..blocks.len().div_ceil(BLOCKS_PER_CB))
            .map(|c| c * BLOCKS_PER_CB..((c + 1) * BLOCKS_PER_CB).min(blocks.len()))
            .collect();
        let (ok, _) = run_concurrent(be, &self.f, &parts, |e, range| {
            let en = Enc { e, be };
            for &i in &blocks[range] {
                let b = &m.blocks[i];
                let o = offs[i];
                let tab = i * tab_block * 4;
                // Attention.
                norm_in(&en, &b.qkv, o.norm1, tab, d, 0);
                gemm(&en, &b.qkv, o.os[0], at(B::Qkv, 0), 3 * inner);
                heads(&en, 0, B::Q, B::Ql, Some(o.qn), o.b[0]);
                heads(&en, 1, B::K, B::Kl, Some(o.kn), o.b[0]);
                heads(&en, 2, B::V, B::V, None, o.b[0]);
                e.barrier();
                let scale = (1.0 / (hd as f32).sqrt()).to_bits();
                if hd == 64 {
                    e.pipe(p.flash64);
                } else {
                    e.pipe(match mode {
                        1 => mma.flash128,
                        2 => p.hilo.unwrap(),
                        _ => mma.flash128_f32,
                    });
                }
                e.buf(id(B::Q), 0, 0);
                e.buf(id(B::K), 0, 1);
                e.buf(id(B::V), 0, 2);
                e.buf(id(B::Att), 0, 3);
                e.bytes(&U4([n as u32, n as u32, inner as u32, scale]), 4);
                if mode == 2 {
                    e.buf(id(B::Ql), 0, 5);
                    e.buf(id(B::Kl), 0, 6);
                }
                if hd == 64 {
                    e.dispatch((n.div_ceil(32), nh), 128);
                } else {
                    e.dispatch((n.div_ceil(64), nh), 128);
                }
                e.barrier();
                input(&en, &b.out, B::Att, inner);
                gemm(&en, &b.out, o.os[1], at(B::Tmp, 0), d);
                gated(&en, tab, 2 * d, o.scale.map(|s| s.0), o.b[1]);
                // Feed-forward.
                norm_in(&en, &b.fc1, o.norm2, tab, 4 * d, 3 * d);
                gemm(&en, &b.fc1, o.os[2], at(B::Up, 0), 2 * f);
                let rot = b.fc2.rot.is_some() || pair;
                e.pipe(p.swiglu);
                e.buf(id(B::Up), 0, 0);
                e.buf(id(if rot { B::Xf } else { B::Xh }), 0, 1);
                e.buf(id(B::Rs), 0, 2);
                e.bytes(&U4([f as u32, m.gate_first as u32, !rot as u32, o.b[2].is_some() as u32]), 3);
                e.buf(stat, o.b[2].unwrap_or(0), 4);
                e.dispatch((r, 1), 256);
                e.barrier();
                if rot {
                    input(&en, &b.fc2, B::Xf, f);
                }
                gemm(&en, &b.fc2, o.os[3], at(B::Tmp, 0), d);
                gated(&en, tab, 5 * d, o.scale.map(|s| s.1), o.b[3]);
            }
        });
        if !ok {
            return false;
        }
        out.resize(n * d, 0.0);
        // SAFETY: the buffer holds at least these rows.
        unsafe { super::super::par_copy(out.as_mut_ptr(), self.host(B::H), n * d) };
        true
    }
}

/// The H3 corrector's passes ([`crate::gpu::dit::CorrSession`]): every block in one submission,
/// the weights as float16 pairs on the matrix-unit GEMMs, attention on `flash_attn64` (float32
/// operands).
pub(crate) struct MetalCorr {
    be: &'static MetalBackend,
    f: Fns,
    p: CPipes,
    bufs: Vec<DBuf>,
    /// The norms and biases uploaded: the model's key and their offsets per block.
    stat: Option<(usize, Vec<CConsts>)>,
}

// SAFETY: as MetalDit.
unsafe impl Send for MetalCorr {}

struct CPipes {
    ln: Id,
    heads: Id,
    gadd: Id,
    gelu: Id,
    flash64: Id,
}

/// Byte offsets of a block's constants: the three norms' weights and biases, then the biases
/// of qkv, out, cq, ck, cv, cout, fc1, fc2.
#[derive(Clone, Copy, Default)]
struct CConsts {
    ln: [[usize; 2]; 3],
    b: [usize; 8],
}

#[derive(Clone, Copy)]
enum C {
    H = 0,
    Xf,
    Xh,
    Rs,
    Qkv,
    Q,
    K,
    V,
    Att,
    Tmp,
    Up,
    Ctx,
    Ch,
    Crs,
    Ck,
    Cv,
    Mods,
    Cos,
    Sin,
    Stat,
}
const NCBUF: usize = 20;

#[repr(C)]
struct Clm {
    n: u32,
    out_half: u32,
    eps: f32,
    has_mod: u32,
}
#[repr(C)]
struct Chd {
    ld_src: u32,
    base: u32,
    heads: u32,
    rope: u32,
    n_valid: u32,
    _p: [u32; 3],
}

impl MetalBackend {
    /// [`crate::gpu::Backend::corrector_session`].
    pub(crate) fn corrector_session_impl(&'static self) -> Option<Box<dyn crate::gpu::dit::CorrSession>> {
        self.mma().filter(|m| m.units)?;
        let pipe = |n: &str| self.named_pipeline(n);
        let p = CPipes {
            ln: pipe("cor_ln_mod")?,
            heads: pipe("cor_heads64")?,
            gadd: pipe("ltx_gated_add")?,
            gelu: pipe("ltx_gelu")?,
            flash64: pipe("flash_attn64")?,
        };
        Some(Box::new(MetalCorr {
            be: self,
            f: Fns::new(self),
            p,
            bufs: (0..NCBUF)
                .map(|_| DBuf {
                    id: std::ptr::null_mut(),
                    cap: 0,
                })
                .collect(),
            stat: None,
        }))
    }
}

impl Drop for MetalCorr {
    fn drop(&mut self) {
        release_all(self.be, &self.bufs);
    }
}

impl MetalCorr {
    fn ensure(&mut self, b: C, bytes: usize) -> bool {
        grow(self.be, &mut self.bufs[b as usize], bytes)
    }

    fn host(&self, b: C) -> *mut f32 {
        self.f.contents(self.bufs[b as usize].id) as *mut f32
    }

    /// Copies `v` to the start of buffer `b` (grown to hold it), zeros up to `len` floats after.
    fn upload(&mut self, b: C, v: &[f32], len: usize) -> bool {
        let len = len.max(v.len());
        if !self.ensure(b, (len * 4).max(256)) {
            return false;
        }
        // SAFETY: the buffer holds at least `len` floats; the device is idle.
        unsafe {
            super::super::par_copy(self.host(b), v.as_ptr(), v.len());
            std::ptr::write_bytes(self.host(b).add(v.len()), 0, len - v.len());
        }
        true
    }
}

fn split_runnable(be: &MetalBackend, l: &SplitLin) -> bool {
    let plain = |x: &Lin| runnable(be, x) && x.side.is_none() && x.rot.is_none();
    plain(&l.hi) && l.lo.as_ref().is_none_or(plain)
}

impl CorrSession for MetalCorr {
    fn forward(&mut self, m: &CorrModel, inp: &CorrIn, out: &mut Vec<f32>) -> bool {
        let be = self.be;
        let (d, nh, hd, f) = (m.width, m.heads, m.head_dim, m.ffn);
        let (n, l) = (inp.rows, inp.n_ctx);
        if m.blocks.is_empty() || n == 0 || l == 0 || hd != 64 || nh * hd != d || !d.is_multiple_of(4) || !f.is_multiple_of(4) {
            return false;
        }
        if inp.x.len() != n * d
            || inp.ctx.len() != l * d
            || inp.mods.len() != m.blocks.len() * 6 * d
            || inp.cos.len() != n * hd / 2
            || inp.sin.len() != n * hd / 2
        {
            return false;
        }
        for b in &m.blocks {
            for x in [&b.qkv, &b.out, &b.cq, &b.ck, &b.cv, &b.cout, &b.fc1, &b.fc2] {
                if !split_runnable(be, x) {
                    return false;
                }
            }
            if b.qkv.hi.n_out != 3 * d || b.fc1.hi.n_out != f || b.fc2.hi.n_in != f {
                return false;
            }
        }
        let r = n.next_multiple_of(MMA_TM);
        let rc = l.next_multiple_of(MMA_TM);
        let rk = r.max(rc);
        let ok = self.ensure(C::H, r * d * 4)
            && self.ensure(C::Xf, r * f.max(d) * 4)
            && self.ensure(C::Xh, r * f.max(d) * 2)
            && self.ensure(C::Rs, r * 4)
            && self.ensure(C::Qkv, r * 3 * d * 4)
            && self.ensure(C::Q, r * d * 4)
            && self.ensure(C::K, rk * d * 4)
            && self.ensure(C::V, rk * d * 4)
            && self.ensure(C::Att, r * d * 4)
            && self.ensure(C::Tmp, rk * d * 4)
            && self.ensure(C::Up, r * f * 4)
            && self.ensure(C::Ch, rc * d * 2)
            && self.ensure(C::Crs, rc * 4)
            && self.ensure(C::Ck, rc * d * 4)
            && self.ensure(C::Cv, rc * d * 4);
        if !ok {
            return false;
        }
        let mkey = m.blocks[0].qkv.hi.dev as *const _ as *const u8 as usize;
        if self.stat.as_ref().map(|s| s.0) != Some(mkey) {
            let mut c: Vec<f32> = Vec::new();
            let mut put = |v: &[f32]| -> usize {
                let o = c.len();
                c.extend_from_slice(v);
                c.resize(c.len().next_multiple_of(64), 0.0);
                o * 4
            };
            let zeros = vec![0f32; f.max(3 * d)];
            let mut offs = Vec::with_capacity(m.blocks.len());
            for b in &m.blocks {
                let ln = b.ln.map(|[w, bb]| [put(w), put(bb)]);
                let lins = [&b.qkv, &b.out, &b.cq, &b.ck, &b.cv, &b.cout, &b.fc1, &b.fc2];
                let bo = lins.map(|x| put(x.hi.bias.unwrap_or(&zeros[..x.hi.n_out])));
                offs.push(CConsts { ln, b: bo });
            }
            if !self.upload(C::Stat, &c, 0) {
                return false;
            }
            self.stat = Some((mkey, offs));
        }
        // Per block: shift, scale, 1 + gate of the self-attention, the same of the MLP, ones.
        let mut mods = Vec::with_capacity(m.blocks.len() * 7 * d);
        for mb in inp.mods.chunks_exact(6 * d) {
            for (k, part) in mb.chunks_exact(d).enumerate() {
                if k == 2 || k == 5 {
                    mods.extend(part.iter().map(|g| 1.0 + g));
                } else {
                    mods.extend_from_slice(part);
                }
            }
            mods.extend(std::iter::repeat_n(1.0f32, d));
        }
        let ok = self.upload(C::H, inp.x, r * d)
            && self.upload(C::Ctx, inp.ctx, rc * d)
            && self.upload(C::Mods, &mods, 0)
            && self.upload(C::Cos, inp.cos, r * hd / 2)
            && self.upload(C::Sin, inp.sin, r * hd / 2);
        if !ok {
            return false;
        }
        let ids: [Id; NCBUF] = std::array::from_fn(|i| self.bufs[i].id);
        let id = |b: C| ids[b as usize];
        let at = |b: C, bytes: usize| (id(b), bytes);
        let p = &self.p;
        let stat = id(C::Stat);
        let offs = &self.stat.as_ref().unwrap().1;
        let main_x = Xh {
            xh: at(C::Xh, 0),
            rs: at(C::Rs, 0),
        };
        let ctx_x = Xh {
            xh: at(C::Ch, 0),
            rs: at(C::Crs, 0),
        };
        // y = (hi + lo) x.
        let gemm = |en: &Enc, w: &SplitLin, x: Xh, rows: usize, y: (Id, usize), ldy: usize| match &w.lo {
            Some(lo) => {
                en.gemm(&w.hi, x, rows, y, ldy, false);
                en.gemm(lo, x, rows, y, ldy, true);
            }
            None => en.gemm(&w.hi, x, rows, y, ldy, false),
        };
        // LayerNorm `which` of block `bi` over H into half rows (modulated by the shift and the
        // scale `mo` floats into the block's mods, when given).
        let ln = |en: &Enc, bi: usize, which: usize, mo: Option<usize>| {
            let e = en.e;
            let [w, b] = offs[bi].ln[which];
            let mb = bi * 7 * d * 4;
            e.pipe(p.ln);
            e.buf(id(C::H), 0, 0);
            e.buf(stat, w, 1);
            e.buf(stat, b, 2);
            e.buf(id(C::Mods), mb + mo.map_or(0, |o| (o + d) * 4), 3);
            e.buf(id(C::Mods), mb + mo.map_or(0, |o| o * 4), 4);
            e.buf(id(C::Xh), 0, 5);
            e.buf(id(C::Rs), 0, 6);
            e.bytes(
                &Clm {
                    n: d as u32,
                    out_half: 1,
                    eps: m.eps,
                    has_mod: mo.is_some() as u32,
                },
                7,
            );
            e.dispatch((r, 1), 256);
            e.barrier();
        };
        #[allow(clippy::too_many_arguments)]
        let heads = |en: &Enc, src: C, ld: usize, base: usize, bias: usize, dst: C, rope: bool, rows: usize, valid: usize| {
            let e = en.e;
            e.pipe(p.heads);
            e.buf(id(src), 0, 0);
            e.buf(id(dst), 0, 1);
            e.buf(stat, bias, 2);
            e.buf(id(C::Cos), 0, 3);
            e.buf(id(C::Sin), 0, 4);
            e.bytes(
                &Chd {
                    ld_src: ld as u32,
                    base: base as u32,
                    heads: nh as u32,
                    rope: rope as u32,
                    n_valid: valid as u32,
                    _p: [0; 3],
                },
                5,
            );
            e.dispatch((rows, nh), 32);
        };
        // H += (Tmp + bias) * the block's mods row `row`.
        let gated = |en: &Enc, bi: usize, bias: usize, row: usize| {
            let e = en.e;
            e.pipe(p.gadd);
            e.buf(id(C::H), 0, 0);
            e.buf(id(C::Tmp), 0, 1);
            e.buf(stat, bias, 2);
            e.buf(id(C::Mods), (bi * 7 + row) * d * 4, 3);
            e.buf(id(C::Mods), 0, 4);
            e.bytes(
                &Lgd {
                    n: d as u32,
                    ld_y: d as u32,
                    ld_tab: 0,
                    off: 0,
                    uniform_cls: 1,
                    has_bias: 1,
                    _p: [0; 2],
                },
                5,
            );
            e.dispatch(((d / 4).div_ceil(256), n), 256);
            e.barrier();
        };
        let flash = |en: &Enc, nk: usize| {
            let e = en.e;
            e.pipe(p.flash64);
            e.buf(id(C::Q), 0, 0);
            e.buf(id(C::K), 0, 1);
            e.buf(id(C::V), 0, 2);
            e.buf(id(C::Att), 0, 3);
            e.bytes(&U4([n as u32, nk as u32, d as u32, (1.0 / (hd as f32).sqrt()).to_bits()]), 4);
            e.dispatch((n.div_ceil(32), nh), 128);
            e.barrier();
        };

        let nb = m.blocks.len();
        let per = 2 * BLOCKS_PER_CB;
        let parts: Vec<std::ops::Range<usize>> = (0..nb.div_ceil(per)).map(|c| c * per..((c + 1) * per).min(nb)).collect();
        let (ok, _) = run_concurrent(be, &self.f, &parts, |e, range| {
            let en = Enc { e, be };
            if range.start == 0 {
                en.to_half(at(C::Ctx, 0), d, d, rc, ctx_x);
            }
            for bi in range {
                let b = &m.blocks[bi];
                let o = offs[bi];
                // Self-attention.
                ln(&en, bi, 0, Some(0));
                gemm(&en, &b.qkv, main_x, r, at(C::Qkv, 0), 3 * d);
                heads(&en, C::Qkv, 3 * d, 0, o.b[0], C::Q, true, r, n);
                heads(&en, C::Qkv, 3 * d, d, o.b[0] + d * 4, C::K, true, r, n);
                heads(&en, C::Qkv, 3 * d, 2 * d, o.b[0] + 2 * d * 4, C::V, false, r, n);
                e.barrier();
                flash(&en, n);
                en.to_half(at(C::Att, 0), d, d, r, main_x);
                gemm(&en, &b.out, main_x, r, at(C::Tmp, 0), d);
                gated(&en, bi, o.b[1], 2);
                // Cross-attention to the caption.
                ln(&en, bi, 1, None);
                gemm(&en, &b.cq, main_x, r, at(C::Tmp, 0), d);
                gemm(&en, &b.ck, ctx_x, rc, at(C::Ck, 0), d);
                gemm(&en, &b.cv, ctx_x, rc, at(C::Cv, 0), d);
                heads(&en, C::Tmp, d, 0, o.b[2], C::Q, false, r, n);
                heads(&en, C::Ck, d, 0, o.b[3], C::K, false, rc, l);
                heads(&en, C::Cv, d, 0, o.b[4], C::V, false, rc, l);
                e.barrier();
                flash(&en, l);
                en.to_half(at(C::Att, 0), d, d, r, main_x);
                gemm(&en, &b.cout, main_x, r, at(C::Tmp, 0), d);
                gated(&en, bi, o.b[5], 6);
                // MLP.
                ln(&en, bi, 2, Some(3 * d));
                gemm(&en, &b.fc1, main_x, r, at(C::Up, 0), f);
                e.pipe(p.gelu);
                e.buf(id(C::Up), 0, 0);
                e.buf(stat, o.b[6], 1);
                e.bytes(&U4([f as u32, 1, 0, 0]), 2);
                e.dispatch(((f / 4).div_ceil(256), r), 256);
                e.barrier();
                en.to_half(at(C::Up, 0), f, f, r, main_x);
                gemm(&en, &b.fc2, main_x, r, at(C::Tmp, 0), d);
                gated(&en, bi, o.b[7], 5);
            }
        });
        if !ok {
            return false;
        }
        out.resize(n * d, 0.0);
        // SAFETY: the buffer holds at least these rows.
        unsafe { super::super::par_copy(out.as_mut_ptr(), self.host(C::H), n * d) };
        true
    }
}
