//! Prefill ops beyond the matmuls on Metal: attention over a host KV cache (QSA-aware) and the
//! Gated DeltaNet rule over a chunk of tokens ([`crate::gpu::AttnJob`], [`crate::gpu::GdnJob`]),
//! on the decode library's kernels (`pf_attn_*` in metal_qsa.metal, `dec_gdn_rule_t`).

use super::decode::{Fns, run_concurrent};
use super::*;

/// A device buffer released when dropped.
struct Owned {
    id: Id,
    release: (MsgVoid, Sel),
}

impl Drop for Owned {
    fn drop(&mut self) {
        unsafe { (self.release.0)(self.id, self.release.1) };
    }
}

impl MetalBackend {
    /// A new shared buffer holding `parts` one after the other (each 256-byte aligned): the
    /// buffer and each part's offset.
    fn upload_parts(&self, f: &Fns, parts: &[&[u8]], extra: usize) -> Option<(Owned, Vec<usize>)> {
        let mut offs = Vec::with_capacity(parts.len() + 1);
        let mut total = 0;
        for p in parts {
            offs.push(total);
            total += p.len().next_multiple_of(256);
        }
        offs.push(total);
        total += extra;
        let new_len: MsgNewBufLen = self.o.msg_send();
        let id = unsafe { new_len(self.device, self.sels.new_buffer_len, total.max(256), 0) };
        if id.is_null() {
            return None;
        }
        let b = Owned {
            id,
            release: (self.o.msg_send(), self.sels.release),
        };
        let base = f.contents(id);
        for (p, &o) in parts.iter().zip(&offs) {
            // SAFETY: the buffer was sized for every part at its offset.
            unsafe { super::par_copy(base.add(o) as *mut f32, p.as_ptr() as *const f32, p.len() / 4) };
        }
        Some((b, offs))
    }

    pub(crate) fn attention_cached_impl(&self, j: &mut crate::gpu::AttnJob) -> bool {
        let (nh, nkv, hd, t) = (j.nh, j.nkv, j.hd, j.t);
        let d = hd / 32;
        if t == 0 || !hd.is_multiple_of(32) || !matches!(d, 2 | 4 | 8 | 16) || nkv == 0 || !nh.is_multiple_of(nkv) {
            return false;
        }
        let len = j.start + t;
        if j.k.len() < len * nkv * hd || j.v.len() < len * nkv * hd || j.q.len() < t * nh * hd || j.out.len() < t * nh * hd {
            return false;
        }
        let rep = nh / nkv;
        let grp = [8, 4, 2, 1]
            .into_iter()
            .find(|&g| rep.is_multiple_of(g) && g * d <= 16)
            .unwrap_or(1);
        let Some(pipe) = self.named_pipeline(&format!("pf_attn_{d}_{grp}")) else {
            return false;
        };
        // Per token: (offset of its positions, count) or (!0, 0) for all up to its own.
        let mut meta = Vec::with_capacity(2 * t);
        let mut idx: Vec<u32> = Vec::new();
        for i in 0..t {
            match j.sel.and_then(|s| s[i].as_deref()) {
                Some(l) => {
                    meta.push(idx.len() as u32);
                    meta.push(l.len() as u32);
                    idx.extend_from_slice(l);
                }
                None => {
                    meta.push(u32::MAX);
                    meta.push(0);
                }
            }
        }
        idx.push(0);
        let f = Fns::new(self);
        let bytes = |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let ubytes = |v: &[u32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let kv = len * nkv * hd;
        let Some((buf, offs)) = self.upload_parts(
            &f,
            &[
                bytes(&j.q[..t * nh * hd]),
                bytes(&j.k[..kv]),
                bytes(&j.v[..kv]),
                ubytes(&idx),
                ubytes(&meta),
            ],
            t * nh * hd * 4,
        ) else {
            return false;
        };
        #[repr(C)]
        struct Ap {
            nh: u32,
            nkv: u32,
            len: u32,
            lo: u32,
            scale: f32,
            has_gate: u32,
            g_stride: u32,
            g_off: u32,
            nsplit: u32,
            chunk: u32,
            _p: [u32; 2],
        }
        let p = Ap {
            nh: nh as u32,
            nkv: nkv as u32,
            len: 0,
            lo: 0,
            scale: 1.0,
            has_gate: 0,
            g_stride: 0,
            g_off: 0,
            nsplit: 1,
            chunk: 0,
            _p: [0; 2],
        };
        let start = j.start as u32;
        let (ok, _) = run_concurrent(self, &f, std::slice::from_ref(&(0..1)), |e, _| {
            e.pipe(pipe);
            e.buf(buf.id, offs[0], 0);
            e.buf(buf.id, offs[1], 1);
            e.buf(buf.id, offs[2], 2);
            e.buf(buf.id, offs[5], 3);
            e.bytes(&p, 4);
            e.buf(buf.id, offs[3], 5);
            e.buf(buf.id, offs[4], 6);
            e.bytes(&start, 7);
            e.dispatch((nh / grp, t), 256);
        });
        if ok {
            // SAFETY: the output region holds t * nh * hd floats.
            unsafe { super::par_copy(j.out.as_mut_ptr(), f.contents(buf.id).add(offs[5]) as *const f32, t * nh * hd) };
        }
        self.stats.calls.fetch_add(1, Ordering::Relaxed);
        ok
    }

    pub(crate) fn gdn_rule_impl(&self, j: &mut crate::gpu::GdnJob) -> bool {
        let (t, nk, nv, hdk, hdv) = (j.t, j.nk, j.nv, j.hdk, j.hdv);
        // pf_gdn_rule: one threadgroup per (value head, 32 state columns) of hdk threads.
        if t == 0 || nk == 0 || !nv.is_multiple_of(nk) || !hdk.is_multiple_of(32) || !hdv.is_multiple_of(32) || hdk > 256 {
            return false;
        }
        let (key_dim, vd) = (nk * hdk, nv * hdv);
        let (cw, zw) = (2 * key_dim + vd, 2 * nv);
        if j.c.len() < t * cw || j.ba.len() < t * zw || j.state.len() != nv * hdk * hdv || j.out.len() < t * vd {
            return false;
        }
        if j.decay_coef.len() != nv || j.dt_bias.len() != nv {
            return false;
        }
        let Some(pipe) = self.named_pipeline("pf_gdn_rule") else {
            return false;
        };
        let mut cst = j.decay_coef.to_vec();
        cst.extend_from_slice(j.dt_bias);
        let f = Fns::new(self);
        let bytes = |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let Some((buf, offs)) = self.upload_parts(
            &f,
            &[bytes(&j.c[..t * cw]), bytes(&j.ba[..t * zw]), bytes(j.state), bytes(&cst)],
            t * vd * 4,
        ) else {
            return false;
        };
        #[repr(C)]
        struct Gr {
            nk: u32,
            nv: u32,
            key_dim: u32,
            tiled: u32,
            eps: f32,
            qscale: f32,
            hdk: u32,
            hdv: u32,
        }
        let p = Gr {
            nk: nk as u32,
            nv: nv as u32,
            key_dim: key_dim as u32,
            tiled: j.tiled as u32,
            eps: 0.0,
            qscale: (hdk as f32).powf(-0.5),
            hdk: hdk as u32,
            hdv: hdv as u32,
        };
        let ts = [t as u32, cw as u32, zw as u32, vd as u32];
        let (ok, _) = run_concurrent(self, &f, std::slice::from_ref(&(0..1)), |e, _| {
            e.pipe(pipe);
            e.buf(buf.id, offs[0], 0);
            e.buf(buf.id, offs[1], 1);
            e.buf(buf.id, offs[4], 2);
            e.buf(buf.id, offs[2], 3);
            e.buf(buf.id, offs[3], 4);
            e.bytes(&p, 5);
            e.bytes(&ts, 6);
            e.dispatch((nv, hdv / 32), hdk);
        });
        if ok {
            let base = f.contents(buf.id);
            // SAFETY: the regions hold the state and t * vd outputs.
            unsafe {
                std::ptr::copy_nonoverlapping(base.add(offs[2]) as *const f32, j.state.as_mut_ptr(), j.state.len());
                super::par_copy(j.out.as_mut_ptr(), base.add(offs[4]) as *const f32, t * vd);
            }
        }
        self.stats.calls.fetch_add(1, Ordering::Relaxed);
        ok
    }
}
