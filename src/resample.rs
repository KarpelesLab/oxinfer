//! Antialiased bicubic resizing of 8-bit RGB exactly as the Hugging Face processors run it:
//! Pillow's `Image.resize(size, Image.BICUBIC)` (the image processors' PIL backend) and
//! torchvision's uint8 `resize(..., BICUBIC, antialias=True)` (the video processors, and the
//! image processors' torchvision backend).
//!
//! Both resample separably through an 8-bit intermediate image with the same Keys (a = -0.5)
//! filter, coefficients computed in f64 and applied in fixed point. Pillow (`Resample.c`,
//! Pillow 13) uses 22 fractional bits and runs the vertical pass first when the height shrinks
//! by more than twice as many pixels as the width; torchvision (Pillow-SIMD's scheme) scales
//! each pass's coefficients to fit in int16 and always runs the horizontal pass first.

/// Pillow's fixed-point precision.
const PIL_BITS: u32 = 32 - 8 - 2;

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

/// Fixed-point resampling coefficients of one pass: per output pixel, the first input index and
/// the weights, scaled by `2^bits`.
struct Coeffs {
    taps: Vec<(usize, Vec<i32>)>,
    bits: u32,
}

/// Coefficients for resampling `n_in` samples into `n_out` (Pillow's `precompute_coeffs`), in
/// Pillow's 22-bit fixed point, or (`int16`) at the largest precision whose biggest weight fits
/// in an int16 (torchvision).
fn coeffs(n_in: usize, n_out: usize, int16: bool) -> Coeffs {
    let scale = n_in as f64 / n_out as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ss = 1.0 / filterscale;
    let taps: Vec<(usize, Vec<f64>)> = (0..n_out)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            // C `(int)` truncation; negative values clamp to 0 anyway.
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = (((center + support + 0.5) as i64) as usize).min(n_in);
            let w: Vec<f64> = (xmin..xmax).map(|x| bicubic((x as f64 - center + 0.5) * ss)).collect();
            let ww: f64 = w.iter().sum();
            (xmin, w.iter().map(|&v| if ww != 0.0 { v / ww } else { v }).collect())
        })
        .collect();
    let bits = if int16 {
        let wmax = taps.iter().flat_map(|t| t.1.iter().copied()).fold(0f64, f64::max);
        let mut b = 0;
        while b < 22 && ((0.5 + wmax * (1u32 << (b + 1)) as f64) as i32) < (1 << 15) {
            b += 1;
        }
        b
    } else {
        PIL_BITS
    };
    let one = (1u32 << bits) as f64;
    let taps = taps
        .into_iter()
        .map(|(x0, w)| {
            let k = w
                .iter()
                .map(|&v| if v < 0.0 { (-0.5 + v * one) as i32 } else { (0.5 + v * one) as i32 })
                .collect();
            (x0, k)
        })
        .collect();
    Coeffs { taps, bits }
}

/// Resizes interleaved RGB8 `w`x`h` to `tw`x`th` like Pillow's BICUBIC (antialiased when
/// downscaling). Returns RGB8.
pub fn resize_bicubic_rgb8(rgb: &[u8], w: usize, h: usize, tw: usize, th: usize) -> Vec<u8> {
    let (dh, dw) = (h as i64 - th as i64, w as i64 - tw as i64);
    resize(rgb, w, h, tw, th, dh > 0 && dh > dw * 2, false)
}

/// Resizes interleaved RGB8 `w`x`h` to `tw`x`th` like torchvision's uint8 antialiased bicubic.
pub fn resize_bicubic_rgb8_torchvision(rgb: &[u8], w: usize, h: usize, tw: usize, th: usize) -> Vec<u8> {
    resize(rgb, w, h, tw, th, false, true)
}

fn resize(rgb: &[u8], w: usize, h: usize, tw: usize, th: usize, vertical_first: bool, int16: bool) -> Vec<u8> {
    assert_eq!(rgb.len(), w * h * 3, "image size mismatch");
    assert!(tw > 0 && th > 0);
    let (mut img, mut cw, mut ch) = (std::borrow::Cow::Borrowed(rgb), w, h);
    for vertical in [vertical_first, !vertical_first] {
        if vertical && ch != th {
            img = pass(&img, cw, ch, &coeffs(ch, th, int16), true).into();
            ch = th;
        } else if !vertical && cw != tw {
            img = pass(&img, cw, ch, &coeffs(cw, tw, int16), false).into();
            cw = tw;
        }
    }
    img.into_owned()
}

/// One resampling pass over `w`x`h` RGB8: columns (horizontal) or rows (vertical) to
/// `k.taps.len()`.
fn pass(src: &[u8], w: usize, h: usize, k: &Coeffs, vertical: bool) -> Vec<u8> {
    let n = k.taps.len();
    let half = 1i32 << (k.bits - 1);
    let clip8 = |ss: i32| (ss >> k.bits).clamp(0, 255) as u8;
    let (ow, oh) = if vertical { (w, n) } else { (n, h) };
    let mut out = vec![0u8; ow * oh * 3];
    let op = SendPtr(out.as_mut_ptr());
    crate::pool::run(oh, &|oy| {
        // SAFETY: each task writes its own output row.
        let o = unsafe { std::slice::from_raw_parts_mut(op.get().add(oy * ow * 3), ow * 3) };
        if vertical {
            let (y0, ky) = &k.taps[oy];
            for (x, ov) in o.iter_mut().enumerate() {
                let mut ss = half;
                for (j, &kj) in ky.iter().enumerate() {
                    ss += src[(y0 + j) * w * 3 + x] as i32 * kj;
                }
                *ov = clip8(ss);
            }
        } else {
            let row = &src[oy * w * 3..(oy + 1) * w * 3];
            for (ox, (x0, kx)) in k.taps.iter().enumerate() {
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
        assert_eq!(resize_bicubic_rgb8_torchvision(&img, 4, 3, 4, 3), img);
        // A constant image stays constant whatever the scale (weights sum to one).
        let flat = vec![137u8; 37 * 23 * 3];
        for (tw, th) in [(10, 7), (64, 48), (37, 5)] {
            assert!(resize_bicubic_rgb8(&flat, 37, 23, tw, th).iter().all(|&v| v == 137));
            assert!(resize_bicubic_rgb8_torchvision(&flat, 37, 23, tw, th).iter().all(|&v| v == 137));
        }
    }

    /// A 13x9 test pattern, resized to 17x5 below (width up, height down: Pillow and
    /// torchvision run the passes in different orders).
    fn pattern() -> Vec<u8> {
        (0..13 * 9u32)
            .flat_map(|i| [(i * 37 % 256) as u8, ((i * 91 + 17) % 256) as u8, (i * i * 7 % 256) as u8])
            .collect()
    }

    /// Position-weighted checksum.
    fn checksum(v: &[u8]) -> u64 {
        v.iter().enumerate().map(|(i, &x)| (i as u64 + 1) * x as u64).sum()
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
        assert_eq!(resize_bicubic_rgb8(&a, 5, 4, 3, 2), PILLOW_5X4_TO_3X2);
        let r = resize_bicubic_rgb8(&pattern(), 13, 9, 17, 5);
        assert_eq!(&r[..24], &PILLOW_13X9_TO_17X5_HEAD);
        assert_eq!(checksum(&r), 4330021);
    }

    #[test]
    fn matches_torchvision() {
        // torchvision 0.29: resize(a, [5, 17], BICUBIC, antialias=True) on the uint8 tensor.
        let r = resize_bicubic_rgb8_torchvision(&pattern(), 13, 9, 17, 5);
        assert_eq!(&r[..24], &TV_13X9_TO_17X5_HEAD);
        assert_eq!(checksum(&r), 4331677);
    }

    const PILLOW_5X4_TO_3X2: [u8; 18] = [43, 212, 21, 64, 191, 32, 85, 170, 42, 162, 93, 81, 183, 72, 91, 204, 51, 102];
    const PILLOW_13X9_TO_17X5_HEAD: [u8; 24] = [
        111, 85, 74, 52, 67, 60, 27, 99, 39, 57, 146, 27, 95, 103, 39, 124, 81, 136, 152, 122, 183, 184, 164, 199,
    ];
    const TV_13X9_TO_17X5_HEAD: [u8; 24] = [
        112, 85, 74, 52, 68, 59, 27, 99, 39, 58, 146, 27, 95, 103, 42, 124, 81, 136, 152, 122, 183, 183, 163, 199,
    ];
}
