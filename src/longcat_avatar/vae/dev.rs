//! The video VAE's passes on CUDA with the frames kept on the device ([`crate::cuda_vae`]): the
//! layers of [`super`] walked over device frames, with the causal convolutions' caches as device
//! frames too. The mid block's attention (one head over the pixels of a latent frame) runs on
//! the host.
//!
//! `OXINFER_AVATAR_VAE_GPU=0` keeps the per-layer path; a pass that cannot run on the device
//! returns `None` and the host path runs.

use super::{Attn, CConv, Frame, Layer, Res, Rms, Stack, VideoVae};
use crate::cuda_vae::{DFrame, Out, Pass};
use anyhow::{Result, bail};

pub(super) use crate::cuda_vae::Engine;

/// The variable that turns the device passes off.
pub(super) const OFF: &str = "OXINFER_AVATAR_VAE_GPU";

enum DCache<'p> {
    None,
    Rep,
    Frames(Vec<DFrame<'p>>),
}

struct DState<'p> {
    caches: Vec<DCache<'p>>,
    at: usize,
}

impl<'p> DState<'p> {
    /// The index of the next layer's cache.
    fn next(&mut self) -> usize {
        if self.at == self.caches.len() {
            self.caches.push(DCache::None);
        }
        self.at += 1;
        self.at - 1
    }

    fn frames(&self, i: usize) -> Option<Vec<DFrame<'p>>> {
        match &self.caches[i] {
            DCache::Frames(f) => Some(f.clone()),
            _ => None,
        }
    }
}

/// The last two frames of `x`, with the last cached frame in front when `x` has only one.
fn tail2<'p>(x: &[DFrame<'p>], prev: Option<&[DFrame<'p>]>) -> Vec<DFrame<'p>> {
    let mut keep: Vec<DFrame> = x[x.len().saturating_sub(2)..].to_vec();
    if keep.len() < 2
        && let Some(last) = prev.and_then(|p| p.last())
    {
        keep.insert(0, last.clone());
    }
    keep
}

/// The layers of [`super`] over a pass.
struct Wan<'a, 'p, 's>(&'a mut Pass<'p, 's>);

impl<'p, 's> std::ops::Deref for Wan<'_, 'p, 's> {
    type Target = Pass<'p, 's>;
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl std::ops::DerefMut for Wan<'_, '_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

impl<'p> Wan<'_, 'p, '_> {
    fn rms_all(&mut self, n: &Rms, x: &[DFrame<'p>], npix: usize) -> Result<Vec<DFrame<'p>>> {
        x.iter().map(|f| self.rms(&n.g, f, npix, true)).collect()
    }

    /// [`CConv::forward`]; `adds[i]` is added to output frame `i`.
    #[allow(clippy::too_many_arguments)]
    fn cconv(
        &mut self,
        cc: &CConv,
        x: &[DFrame<'p>],
        cache: Option<&[DFrame<'p>]>,
        h: usize,
        w: usize,
        adds: Option<&[DFrame<'p>]>,
        out: Out,
    ) -> Result<Vec<DFrame<'p>>> {
        if cc.kt == 1 {
            return x
                .iter()
                .enumerate()
                .map(|(i, f)| Ok(self.conv(&cc.conv, &[Some(f)], h, w, 1, false, adds.map(|a| &a[i]), out)?.0))
                .collect();
        }
        let cached = if cc.pad_t > 0 { cache.unwrap_or(&[]) } else { &[] };
        let pad = cc.pad_t.saturating_sub(cached.len());
        let seq: Vec<Option<&DFrame>> = (0..pad)
            .map(|_| None)
            .chain(cached.iter().map(Some))
            .chain(x.iter().map(Some))
            .collect();
        let n_out = if seq.len() >= cc.kt {
            (seq.len() - cc.kt) / cc.stride_t + 1
        } else {
            0
        };
        let mut ys = Vec::with_capacity(n_out);
        for o in 0..n_out {
            let taps = &seq[o * cc.stride_t..o * cc.stride_t + cc.kt];
            if taps.iter().all(|t| t.is_none()) {
                bail!("a convolution of zero frames only");
            }
            ys.push(self.conv(&cc.conv, taps, h, w, 1, false, adds.map(|a| &a[o]), out)?.0);
        }
        Ok(ys)
    }

    /// [`CConv::cached`].
    #[allow(clippy::too_many_arguments)]
    fn cached(
        &mut self,
        cc: &CConv,
        x: &[DFrame<'p>],
        st: &mut DState<'p>,
        h: usize,
        w: usize,
        adds: Option<&[DFrame<'p>]>,
        out: Out,
    ) -> Result<Vec<DFrame<'p>>> {
        let i = st.next();
        let prev = st.frames(i);
        let y = self.cconv(cc, x, prev.as_deref(), h, w, adds, out)?;
        st.caches[i] = DCache::Frames(tail2(x, prev.as_deref()));
        Ok(y)
    }

    fn res(&mut self, r: &Res, x: Vec<DFrame<'p>>, st: &mut DState<'p>, h: usize, w: usize) -> Result<Vec<DFrame<'p>>> {
        let sc = match &r.shortcut {
            Some(c) => Some(self.cconv(c, &x, None, h, w, None, Out::Frame)?),
            None => None,
        };
        let a = self.rms_all(&r.norm1, &x, h * w)?;
        let a = self.cached(&r.conv1, &a, st, h, w, None, Out::Frame)?;
        let a = self.rms_all(&r.norm2, &a, h * w)?;
        self.cached(&r.conv2, &a, st, h, w, Some(sc.as_deref().unwrap_or(&x)), Out::Frame)
    }

    /// The attention of a frame, on the host.
    fn attn(&mut self, a: &Attn, f: &DFrame<'p>, h: usize, w: usize) -> Result<DFrame<'p>> {
        let x = self.download(f, h * w, a.c)?;
        self.upload(&a.forward(&x, h, w), a.c)
    }

    fn layer(&mut self, l: &Layer, x: Vec<DFrame<'p>>, st: &mut DState<'p>, h: usize, w: usize) -> Result<(Vec<DFrame<'p>>, usize, usize)> {
        match l {
            Layer::Res(r) => Ok((self.res(r, x, st, h, w)?, h, w)),
            Layer::Attn(a) => {
                let y = x.iter().map(|f| self.attn(a, f, h, w)).collect::<Result<Vec<_>>>()?;
                Ok((y, h, w))
            }
            Layer::Down(c, time) => {
                let (mut oh, mut ow) = (h, w);
                let mut y = Vec::with_capacity(x.len());
                for f in &x {
                    let (g, a, b) = self.conv(c, &[Some(f)], h, w, 2, false, None, Out::Frame)?;
                    (oh, ow) = (a, b);
                    y.push(g);
                }
                let Some(tc) = time else { return Ok((y, oh, ow)) };
                let i = st.next();
                match st.frames(i) {
                    Some(prev) => {
                        let mut seq = vec![prev.last().unwrap().clone()];
                        seq.extend(y);
                        let keep = seq.last().unwrap().clone();
                        let out = self.cconv(tc, &seq, None, oh, ow, None, Out::Frame)?;
                        st.caches[i] = DCache::Frames(vec![keep]);
                        Ok((out, oh, ow))
                    }
                    None => {
                        st.caches[i] = DCache::Frames(y.clone());
                        Ok((y, oh, ow))
                    }
                }
            }
            Layer::Up(c, time) => {
                let mut x = x;
                if let Some(tc) = time {
                    let i = st.next();
                    if matches!(st.caches[i], DCache::None) {
                        st.caches[i] = DCache::Rep;
                    } else {
                        let prev = st.frames(i);
                        let mut keep = tail2(&x, prev.as_deref());
                        if keep.len() < 2 && prev.is_none() {
                            let ch = tc.cin;
                            keep.insert(0, self.upload(&vec![0f32; h * w * ch], ch)?);
                        }
                        let y = self.cconv(tc, &x, prev.as_deref(), h, w, None, Out::Halves)?;
                        st.caches[i] = DCache::Frames(keep);
                        // Channels `[0, C)` are the first of each pair of frames, `[C, 2C)` the second.
                        x = y.iter().flat_map(|f| (0..2).map(|half| f.half(half))).collect();
                    }
                }
                let mut y = Vec::with_capacity(x.len());
                for f in &x {
                    y.push(self.conv(c, &[Some(f)], h, w, 1, true, None, Out::Frame)?.0);
                }
                Ok((y, 2 * h, 2 * w))
            }
        }
    }

    /// [`Stack::forward`].
    fn stack(&mut self, s: &Stack, x: &[DFrame<'p>], st: &mut DState<'p>, h: usize, w: usize) -> Result<(Vec<DFrame<'p>>, usize, usize)> {
        st.at = 0;
        let (mut x, mut h, mut w) = (self.cached(&s.conv_in, x, st, h, w, None, Out::Frame)?, h, w);
        for l in &s.layers {
            (x, h, w) = self.layer(l, x, st, h, w)?;
        }
        let a = self.rms_all(&s.norm_out, &x, h * w)?;
        drop(x);
        Ok((self.cached(&s.conv_out, &a, st, h, w, None, Out::Dense)?, h, w))
    }
}

/// [`VideoVae::encode`]'s chunks on the device: the encoder's output frames (`[lh * lw][2 z]`)
/// and the latent size.
pub(super) fn encode(e: &Engine, vae: &VideoVae, frames: &[Frame], h: usize, w: usize) -> Option<(Vec<Frame>, usize, usize)> {
    let enc = vae.enc.as_ref()?;
    let st_t = vae.scale_t();
    let cout = enc.conv_out.conv.cout;
    e.pass(|p| {
        let mut p = Wan(p);
        let mut st = DState { caches: Vec::new(), at: 0 };
        let mut out = Vec::new();
        let (mut lh, mut lw) = (0, 0);
        let mut at = 0;
        while at < frames.len() {
            let n = if at == 0 { 1 } else { st_t };
            let x = frames[at..at + n].iter().map(|f| p.upload(f, 3)).collect::<Result<Vec<_>>>()?;
            let (y, a, b) = p.stack(enc, &x, &mut st, h, w)?;
            (lh, lw) = (a, b);
            for f in &y {
                out.push(p.download(f, a * b, cout)?);
            }
            at += n;
        }
        Ok((out, lh, lw))
    })
}

/// [`VideoVae::decode`]'s chunks on the device: `x` are the latent frames after `post_quant`
/// (`[h * w][z]`); the frames `[64 h w][3]`, not clamped.
pub(super) fn decode(
    e: &Engine,
    vae: &VideoVae,
    x: &[Frame],
    h: usize,
    w: usize,
    progress: &mut dyn FnMut(usize, usize),
) -> Option<Vec<Frame>> {
    let z = vae.cfg.z_dim;
    e.pass(|p| {
        let mut p = Wan(p);
        let mut st = DState { caches: Vec::new(), at: 0 };
        let mut out = Vec::new();
        for (i, f) in x.iter().enumerate() {
            let up = p.upload(f, z)?;
            let (y, oh, ow) = p.stack(&vae.dec, &[up], &mut st, h, w)?;
            for g in &y {
                out.push(p.download(g, oh * ow, 3)?);
            }
            progress(i + 1, x.len());
        }
        Ok(out)
    })
}

#[cfg(test)]
mod tests {
    use super::super::State;
    use super::*;
    use crate::cuda::CudaBackend;

    fn backend() -> Option<&'static CudaBackend> {
        static B: std::sync::OnceLock<Option<CudaBackend>> = std::sync::OnceLock::new();
        B.get_or_init(|| CudaBackend::new(0).ok()).as_ref()
    }

    fn close(what: &str, dev: &[Frame], host: &[Frame]) {
        assert_eq!(dev.len(), host.len(), "{what}: frames");
        for (i, (a, b)) in dev.iter().zip(host).enumerate() {
            assert_eq!(a.len(), b.len(), "{what} frame {i}: size");
            let scale = b.iter().map(|v| v.abs()).sum::<f32>() / b.len() as f32;
            let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            assert!(max < 1e-4 * scale.max(1.0), "{what} frame {i}: {max:.3e} (mean |v| {scale:.3e})");
        }
    }

    /// The device passes against the host's on the tiny release's VAE (skipped without an NVIDIA
    /// GPU): 9 frames encoded (three chunks), 3 latent frames decoded (9 frames).
    #[test]
    fn passes_match_host() {
        let Some(be) = backend() else { return };
        let dir = std::env::temp_dir().join(format!("oxinfer-longcat-avatar-vae-cuda-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (_, base) = super::super::super::tiny::write_tiny_release(&dir).unwrap();
        let vae = VideoVae::load(&base.join("vae"), true).unwrap();
        let e = Engine::build(be).unwrap();
        let mut seed = 5u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        let (h, w) = (32, 48);
        let frames: Vec<Frame> = (0..9).map(|_| (0..h * w * 3).map(|_| rnd()).collect()).collect();
        let (dev, lh, lw) = encode(&e, &vae, &frames, h, w).expect("the engine declined");
        let enc = vae.enc.as_ref().unwrap();
        let mut st = State { caches: Vec::new(), at: 0 };
        let mut host = Vec::new();
        let mut at = 0;
        while at < frames.len() {
            let n = if at == 0 { 1 } else { vae.scale_t() };
            host.extend(enc.forward(&frames[at..at + n], &mut st, h, w).0);
            at += n;
        }
        assert_eq!((lh, lw), (h / vae.scale(), w / vae.scale()));
        close("encode", &dev, &host);

        let z = vae.cfg.z_dim;
        let lat: Vec<Frame> = (0..3).map(|_| (0..lh * lw * z).map(|_| rnd()).collect()).collect();
        let dev = decode(&e, &vae, &lat, lh, lw, &mut |_, _| {}).expect("the engine declined");
        let mut st = State { caches: Vec::new(), at: 0 };
        let mut host = Vec::new();
        for f in &lat {
            host.extend(vae.dec.forward(std::slice::from_ref(f), &mut st, lh, lw).0);
        }
        close("decode", &dev, &host);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
