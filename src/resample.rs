//! Pillow-exact image resizing: `Image.resize(size, Image.BICUBIC)` on 8-bit RGB, as the
//! Hugging Face image and video processors run it (the PIL backend exactly; torchvision's uint8
//! antialiased bicubic differs from it by at most a level or two on under 1% of the pixels).
//!
//! Pillow resamples separably through an 8-bit intermediate image (horizontal pass first unless
//! the height shrinks much more than the width), with filter coefficients computed in f64 and
//! applied as 22-bit fixed point (`Resample.c`, Pillow 13).

const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Keys bicubic kernel with a = -0.5 (Pillow's `bicubic_filter`).
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Per-output-pixel `(first input index, fixed-point weights)` for resampling `n_in` samples
/// into `n_out` (Pillow's `precompute_coeffs` + `normalize_coeffs_8bpc`).
fn coeffs(n_in: usize, n_out: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = n_in as f64 / n_out as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ss = 1.0 / filterscale;
    (0..n_out)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            // C `(int)` truncation; negative values clamp to 0 anyway.
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = (((center + support + 0.5) as i64) as usize).min(n_in);
            let w: Vec<f64> = (xmin..xmax).map(|x| bicubic((x as f64 - center + 0.5) * ss)).collect();
            let ww: f64 = w.iter().sum();
            let one = (1u32 << PRECISION_BITS) as f64;
            let k = w
                .iter()
                .map(|&v| {
                    let v = if ww != 0.0 { v / ww } else { v };
                    if v < 0.0 { (-0.5 + v * one) as i32 } else { (0.5 + v * one) as i32 }
                })
                .collect();
            (xmin, k)
        })
        .collect()
}

#[inline]
fn clip8(ss: i32) -> u8 {
    (ss >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Resizes interleaved RGB8 `w`x`h` to `tw`x`th` like Pillow's BICUBIC (antialiased when
/// downscaling). Returns RGB8.
pub fn resize_bicubic_rgb8(rgb: &[u8], w: usize, h: usize, tw: usize, th: usize) -> Vec<u8> {
    assert_eq!(rgb.len(), w * h * 3, "image size mismatch");
    assert!(tw > 0 && th > 0);
    // Pillow runs the vertical pass first when the height shrinks by more than twice as many
    // pixels as the width (the rounding of the 8-bit intermediate depends on the order).
    let (dh, dw) = (h as i64 - th as i64, w as i64 - tw as i64);
    let vertical_first = dh > 0 && dh > dw * 2;
    let (mut img, mut cw, mut ch) = (std::borrow::Cow::Borrowed(rgb), w, h);
    for vertical in [vertical_first, !vertical_first] {
        if vertical && ch != th {
            img = pass(&img, cw, ch, th, true).into();
            ch = th;
        } else if !vertical && cw != tw {
            img = pass(&img, cw, ch, tw, false).into();
            cw = tw;
        }
    }
    img.into_owned()
}

/// One resampling pass over `w`x`h` RGB8: columns to `n` (horizontal) or rows to `n` (vertical).
fn pass(src: &[u8], w: usize, h: usize, n: usize, vertical: bool) -> Vec<u8> {
    let half = 1i32 << (PRECISION_BITS - 1);
    let (ow, oh) = if vertical { (w, n) } else { (n, h) };
    let k = coeffs(if vertical { h } else { w }, n);
    let mut out = vec![0u8; ow * oh * 3];
    let op = SendPtr(out.as_mut_ptr());
    crate::pool::run(oh, &|oy| {
        // SAFETY: each task writes its own output row.
        let o = unsafe { std::slice::from_raw_parts_mut(op.get().add(oy * ow * 3), ow * 3) };
        if vertical {
            let (y0, ky) = &k[oy];
            for (x, ov) in o.iter_mut().enumerate() {
                let mut ss = half;
                for (j, &kj) in ky.iter().enumerate() {
                    ss += src[(y0 + j) * w * 3 + x] as i32 * kj;
                }
                *ov = clip8(ss);
            }
        } else {
            let row = &src[oy * w * 3..(oy + 1) * w * 3];
            for (ox, (x0, kx)) in k.iter().enumerate() {
                let mut ss = [half; 3];
                for (j, &kj) in kx.iter().enumerate() {
                    let p = &row[(x0 + j) * 3..(x0 + j) * 3 + 3];
                    for c in 0..3 {
                        ss[c] += p[c] as i32 * kj;
                    }
                }
                for c in 0..3 {
                    o[ox * 3 + c] = clip8(ss[c]);
                }
            }
        }
    });
    out
}

#[derive(Clone, Copy)]
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline(always)]
    fn get(self) -> *mut u8 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_constant() {
        let img: Vec<u8> = (0..4 * 3 * 3).map(|i| (i * 7) as u8).collect();
        assert_eq!(resize_bicubic_rgb8(&img, 4, 3, 4, 3), img);
        // A constant image stays constant whatever the scale (weights sum to one).
        let flat = vec![137u8; 37 * 23 * 3];
        for (tw, th) in [(10, 7), (64, 48), (37, 5)] {
            assert!(resize_bicubic_rgb8(&flat, 37, 23, tw, th).iter().all(|&v| v == 137));
        }
    }

    #[test]
    fn matches_pillow() {
        // Pillow 13: Image.fromarray(a).resize((3, 2), Image.BICUBIC) for the 5x4 ramp below.
        let a: Vec<u8> = (0..4 * 5)
            .flat_map(|i| {
                let v = (i * 13 % 256) as u8;
                [v, 255 - v, v / 2]
            })
            .collect();
        let r = resize_bicubic_rgb8(&a, 5, 4, 3, 2);
        assert_eq!(r.len(), 3 * 2 * 3);
        assert_eq!(r, PILLOW_5X4_TO_3X2);
    }

    const PILLOW_5X4_TO_3X2: [u8; 18] = [43, 212, 21, 64, 191, 32, 85, 170, 42, 162, 93, 81, 183, 72, 91, 204, 51, 102];
}
