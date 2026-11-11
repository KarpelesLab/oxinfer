//! Multi-axis rotary position embeddings over token ids (diffusers `FluxPosEmbed`): each token
//! has one position per axis (FLUX: `[index, row, column]`), axis `a` rotates `axes[a]` of the head
//! dimensions (pairs `(2i, 2i + 1)`, interleaved), with frequencies `theta^(-2i / axes[a])`.
//! Text tokens sit at position 0 everywhere; image tokens at their latent-grid row and column;
//! FLUX Kontext gives its reference-image tokens index 1 on the first axis.

use crate::kernels::for_chunks;

/// Per-token cos / sin tables (`[rows, dim]`, each value repeated for its pair).
pub struct Rope {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub rows: usize,
    pub dim: usize,
}

/// A grid of image tokens appended after the output's: `h` x `w` tokens at ids
/// `[index, y0 + y, x0 + x]` (a Kontext reference image).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    pub index: f32,
    pub h: usize,
    pub w: usize,
    pub y0: usize,
    pub x0: usize,
}

impl Rope {
    /// `ids` is `[rows, axes.len()]` (positions, possibly fractional); `axes` are the rotary
    /// widths per axis (summing to the head dimension).
    pub fn new(ids: &[f32], axes: &[usize], theta: f64) -> Self {
        let na = axes.len();
        let rows = ids.len() / na;
        let dim: usize = axes.iter().sum();
        // Frequencies in f64 as diffusers (`freqs_dtype=torch.float64`), tables in f32.
        let freqs: Vec<Vec<f64>> = axes
            .iter()
            .map(|&d| (0..d / 2).map(|i| 1.0 / theta.powf((2 * i) as f64 / d as f64)).collect())
            .collect();
        let mut cos = vec![0f32; rows * dim];
        let mut sin = vec![0f32; rows * dim];
        for r in 0..rows {
            let mut o = r * dim;
            for (a, f) in freqs.iter().enumerate() {
                let p = ids[r * na + a] as f64;
                for &w in f {
                    let ang = p * w;
                    let (s, c) = (ang.sin() as f32, ang.cos() as f32);
                    cos[o] = c;
                    cos[o + 1] = c;
                    sin[o] = s;
                    sin[o + 1] = s;
                    o += 2;
                }
            }
        }
        Self { cos, sin, rows, dim }
    }

    /// FLUX token ids: `n_txt` zero rows, then an `h` x `w` grid `[0, y, x]`, then any extra
    /// grids `(index, h, w)` (`[index, y, x]`, e.g. Kontext's reference image at index 1).
    pub fn flux_ids(n_txt: usize, h: usize, w: usize, extra: &[(f32, usize, usize)]) -> Vec<f32> {
        let grids: Vec<Grid> = extra.iter().map(|&(index, h, w)| Grid { index, h, w, y0: 0, x0: 0 }).collect();
        Self::flux_ids_at(n_txt, h, w, &grids)
    }

    /// [`Rope::flux_ids`] with every extra grid at its own offset (several Kontext references).
    pub fn flux_ids_at(n_txt: usize, h: usize, w: usize, extra: &[Grid]) -> Vec<f32> {
        let mut ids = vec![0f32; n_txt * 3];
        let out = Grid {
            index: 0.0,
            h,
            w,
            y0: 0,
            x0: 0,
        };
        for g in std::iter::once(&out).chain(extra) {
            for y in 0..g.h {
                for x in 0..g.w {
                    ids.extend_from_slice(&[g.index, (g.y0 + y) as f32, (g.x0 + x) as f32]);
                }
            }
        }
        ids
    }

    /// Rotates `x` (`[rows, heads * dim]`) in place: `x * cos + rotate(x) * sin` with
    /// `rotate(x)[2i] = -x[2i + 1]`, `rotate(x)[2i + 1] = x[2i]`.
    pub fn apply(&self, x: &mut [f32], heads: usize) {
        let (dim, row) = (self.dim, self.dim * heads);
        debug_assert_eq!(x.len(), self.rows * row);
        for_chunks(x, row, self.rows > 16, |r, xr| {
            let (c, s) = (&self.cos[r * dim..(r + 1) * dim], &self.sin[r * dim..(r + 1) * dim]);
            for h in xr.chunks_mut(dim) {
                for i in (0..dim).step_by(2) {
                    let (a, b) = (h[i], h[i + 1]);
                    h[i] = a * c[i] - b * s[i];
                    h[i + 1] = b * c[i + 1] + a * s[i + 1];
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_by_position() {
        // One axis of width 4: pair 0 turns by p, pair 1 by p / 100 (theta 10^4).
        let r = Rope::new(&[0.0, 1.0], &[4], 10000.0);
        let mut x = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        r.apply(&mut x, 1);
        assert_eq!(&x[..4], &[1.0, 0.0, 1.0, 0.0]);
        assert!((x[4] - 1f32.cos()).abs() < 1e-6 && (x[5] - 1f32.sin()).abs() < 1e-6);
        assert!((x[6] - 0.01f32.cos()).abs() < 1e-6 && (x[7] - 0.01f32.sin()).abs() < 1e-6);
        let ids = Rope::flux_ids(2, 1, 2, &[(1.0, 1, 1)]);
        assert_eq!(ids, vec![0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.]);
        let g = Grid {
            index: 2.0,
            h: 1,
            w: 2,
            y0: 3,
            x0: 5,
        };
        assert_eq!(Rope::flux_ids_at(0, 1, 1, &[g]), vec![0., 0., 0., 2., 3., 5., 2., 3., 6.]);
    }
}
