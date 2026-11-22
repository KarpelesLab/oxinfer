//! DeepSeek V4 / V4.1 prefill attention on Metal ([`crate::deepseek4`]'s `attend_dims` for passes
//! of many queries): each query head attends to its own rows of a key table (its sliding window,
//! then the compressed keys it chose), the keys being the values, with the head's sink. The CPU
//! path computes the same per-query softmax as GEMMs over the union of a block's keys; this one
//! reads only each query's own keys, which at long contexts is most of the prefill's attention
//! time (`ds4_attn` in metal_ds4.metal; f32 throughout, the sums in another order than the CPU's).
//! `OXINFER_DS4_GPU_ATTN=0` keeps it on the CPU.

use super::decode::{Fns, run_concurrent};
use super::{Id, IntoInnerUnsafe, MetalBackend, Msg0, Msg1, MsgNewBufLen, MsgNewLib, MsgNewPipe, MsgVoid};

const SRC: &str = include_str!("metal_ds4.metal");

/// The compiled kernel's pipeline (None when it does not build on this device).
struct Pipe(Option<Id>);
unsafe impl Send for Pipe {}
unsafe impl Sync for Pipe {}

fn pipeline(be: &MetalBackend) -> Option<Id> {
    static P: std::sync::OnceLock<Pipe> = std::sync::OnceLock::new();
    P.get_or_init(|| Pipe(compile(be))).0
}

fn compile(be: &MetalBackend) -> Option<Id> {
    let o = &be.o;
    let get_class: unsafe extern "C" fn(*const std::ffi::c_char) -> Id = unsafe { o.objc.get(c"objc_getClass") }.ok()?.into_inner_unsafe();
    let msg_str: unsafe extern "C" fn(Id, super::Sel, *const std::ffi::c_char) -> Id = o.msg_send();
    let msg0: Msg0 = o.msg_send();
    let void: MsgVoid = o.msg_send();
    let ns = |s: &std::ffi::CStr| unsafe { msg_str(get_class(c"NSString".as_ptr()), o.sel(c"stringWithUTF8String:"), s.as_ptr()) };
    let src = std::ffi::CString::new(SRC).ok()?;
    let new_lib: MsgNewLib = o.msg_send();
    let mut err: Id = std::ptr::null_mut();
    let lib = unsafe {
        new_lib(
            be.device,
            o.sel(c"newLibraryWithSource:options:error:"),
            ns(&src),
            std::ptr::null_mut(),
            &mut err,
        )
    };
    if lib.is_null() {
        if std::env::var_os("OXINFER_DEBUG").is_some() && !err.is_null() {
            let text = unsafe { o.nsstring_to_string(msg0(err, o.sel(c"localizedDescription"))) };
            eprintln!("[deepseek metal attention: {text}]");
        }
        return None;
    }
    let msg_fn: Msg1 = o.msg_send();
    let f = unsafe { msg_fn(lib, o.sel(c"newFunctionWithName:"), ns(c"ds4_attn")) };
    if f.is_null() {
        return None;
    }
    let new_pipe: MsgNewPipe = o.msg_send();
    let p = unsafe { new_pipe(be.device, o.sel(c"newComputePipelineStateWithFunction:error:"), f, &mut err) };
    unsafe { void(f, be.sels.release) };
    (!p.is_null()).then_some(p)
}

/// Attention of `t` queries (`q` `[t, nh, 512]`, unscaled) to rows of `keys` (`[nk, 512]`): query
/// `i` takes rows `idx[meta[2 i]..meta[2 i] + meta[2 i + 1]]` in that order; scores scaled by
/// `scale`, a softmax with the sink logit `sink[h]` of each head; `out` `[t, nh, 512]`. False when
/// the GPU cannot take it (the caller then runs the CPU path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn attend(q: &[f32], keys: &[f32], idx: &[u32], meta: &[u32], sink: &[f32], nh: usize, scale: f32, out: &mut [f32]) -> bool {
    const HD: usize = 512;
    let t = meta.len() / 2;
    if std::env::var_os("OXINFER_DS4_GPU_ATTN").is_some_and(|v| v == "0") || t == 0 || !nh.is_multiple_of(8) {
        return false;
    }
    if q.len() < t * nh * HD || out.len() < t * nh * HD || sink.len() < nh || keys.is_empty() {
        return false;
    }
    let Some(be) = crate::gpu::active()
        .and_then(|b| b.as_any())
        .and_then(|a| a.downcast_ref::<MetalBackend>())
    else {
        return false;
    };
    let Some(pipe) = pipeline(be) else {
        return false;
    };
    let f = Fns::new(be);
    let bytes = |p: *const u8, n: usize| unsafe { std::slice::from_raw_parts(p, n) };
    let mut idx_p = idx.to_vec();
    idx_p.push(0);
    let parts: [&[u8]; 5] = [
        bytes(q.as_ptr() as *const u8, t * nh * HD * 4),
        bytes(keys.as_ptr() as *const u8, keys.len() * 4),
        bytes(idx_p.as_ptr() as *const u8, idx_p.len() * 4),
        bytes(meta.as_ptr() as *const u8, meta.len() * 4),
        bytes(sink.as_ptr() as *const u8, nh * 4),
    ];
    let mut offs = Vec::with_capacity(parts.len() + 1);
    let mut total = 0;
    for p in &parts {
        offs.push(total);
        total += p.len().next_multiple_of(256);
    }
    offs.push(total);
    total += t * nh * HD * 4;
    let new_len: MsgNewBufLen = be.o.msg_send();
    let buf = unsafe { new_len(be.device, be.sels.new_buffer_len, total, 0) };
    if buf.is_null() {
        return false;
    }
    let void: MsgVoid = be.o.msg_send();
    let base = f.contents(buf);
    for (p, &o) in parts.iter().zip(&offs) {
        // SAFETY: the buffer holds every part at its offset; each part is a whole number of words.
        unsafe { super::par_copy(base.add(o) as *mut f32, p.as_ptr() as *const f32, p.len() / 4) };
    }
    #[repr(C)]
    struct P {
        nh: u32,
        scale: f32,
        _p: [u32; 2],
    }
    let pr = P {
        nh: nh as u32,
        scale,
        _p: [0; 2],
    };
    let (ok, _) = run_concurrent(be, &f, std::slice::from_ref(&(0..1)), |e, _| {
        e.pipe(pipe);
        for k in 0..5 {
            e.buf(buf, offs[k], k);
        }
        e.buf(buf, offs[5], 5);
        e.bytes(&pr, 6);
        e.dispatch((nh / 8, t), 256);
    });
    if ok {
        // SAFETY: the output region holds t * nh * 512 floats.
        unsafe { super::par_copy(out.as_mut_ptr(), base.add(offs[5]) as *const f32, t * nh * HD) };
    }
    unsafe { void(buf, be.sels.release) };
    ok
}

#[cfg(test)]
mod tests {
    /// The kernel against the softmax with sink written out, on random queries choosing random
    /// rows (repeated rows and an empty list included).
    #[test]
    fn attend_matches_cpu() {
        if crate::gpu::init(&crate::platform::detect(), crate::platform::DevicePref::Auto).is_err() || crate::gpu::active().is_none() {
            return;
        }
        let (t, nh, hd, nk) = (5, 16, 512, 300);
        let mut seed = 0x0d54u64;
        let mut rng = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        let q: Vec<f32> = (0..t * nh * hd).map(|_| rng()).collect();
        let keys: Vec<f32> = (0..nk * hd).map(|_| rng()).collect();
        let sink: Vec<f32> = (0..nh).map(|_| 3.0 * rng()).collect();
        let lens = [0usize, 1, 37, 300, 160];
        let (mut idx, mut meta) = (Vec::new(), Vec::new());
        for &n in &lens {
            meta.extend([idx.len() as u32, n as u32]);
            idx.extend((0..n).map(|j| ((j * 7 + n) % nk) as u32));
        }
        let scale = (hd as f32).powf(-0.5);
        let mut out = vec![f32::NAN; t * nh * hd];
        assert!(
            super::attend(&q, &keys, &idx, &meta, &sink, nh, scale, &mut out),
            "the Metal kernel did not run"
        );
        let mut worst = 0f32;
        for i in 0..t {
            let rows = &idx[meta[2 * i] as usize..(meta[2 * i] + meta[2 * i + 1]) as usize];
            for h in 0..nh {
                let qh = &q[(i * nh + h) * hd..(i * nh + h + 1) * hd];
                let s: Vec<f64> = rows
                    .iter()
                    .map(|&r| (0..hd).map(|d| qh[d] as f64 * keys[r as usize * hd + d] as f64).sum::<f64>() * scale as f64)
                    .collect();
                let m = s.iter().copied().fold(sink[h] as f64, f64::max);
                let tot = (sink[h] as f64 - m).exp() + s.iter().map(|v| (v - m).exp()).sum::<f64>();
                for d in 0..hd {
                    let o: f64 = rows
                        .iter()
                        .zip(&s)
                        .map(|(&r, v)| (v - m).exp() * keys[r as usize * hd + d] as f64)
                        .sum::<f64>()
                        / tot;
                    worst = worst.max((o as f32 - out[(i * nh + h) * hd + d]).abs());
                }
            }
        }
        assert!(worst < 1e-5, "max difference {worst}");
    }
}
