//! Rows interleaved four at a time for the CPU decode kernels (llama.cpp's `block_q8_0x4` and
//! `block_q4_0x4`, the layouts of its ggml-cpu "repack" buffers with 4-byte pieces): a matrix of
//! a supported kind whose row count is a multiple of four is stored as groups of four rows, and
//! each group as its blocks in order, the four rows' block `b` together: the four f16 scales,
//! then the values' bytes in 4-byte pieces, `[row 0 bytes 0..4, row 1 bytes 0..4, row 2, row 3,
//! row 0 bytes 4..8, ...]` (Q8_0: 136 bytes, Q4_0: 72, whose byte `j` holds values `j` and
//! `j + 16`). One SDOT by element multiplies a 16-byte piece with four activations and leaves
//! each row's sum in its own lane, so the kernel computes four rows with the instructions plain
//! rows take for one or two (`kernels_ggml.rs`, `x4_kern`).
//!
//! The same bytes are moved, so an interleaved matrix takes no more memory than the plain one;
//! the loader interleaves a tensor in place in a checkpoint read into memory
//! ([`crate::safetensors::ModelFiles::ggml_cpu`]). Mapped checkpoints (too big for memory,
//! streamed, `mmap=1`) keep the file's layout.

use super::GgmlType;
use std::sync::atomic::{AtomicU8, Ordering};

/// The explicit setting: 0 none (`OXINFER_REPACK`, else on), 1 off, 2 on.
static MODE: AtomicU8 = AtomicU8::new(0);

/// Sets whether checkpoints read into memory have their matrices interleaved for the CPU decode
/// kernels (process-wide; applies to models loaded afterwards). None: the default (on, unless
/// `OXINFER_REPACK=0`).
pub fn set_repack(on: Option<bool>) {
    MODE.store(
        match on {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}

/// The setting in effect: explicit, else `OXINFER_REPACK` (`0` / `off`), else on.
pub fn repack_setting() -> bool {
    match MODE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => !std::env::var("OXINFER_REPACK").is_ok_and(|v| matches!(v.trim(), "0" | "off" | "false" | "no")),
    }
}

/// Whether matrices loaded now are to be interleaved: the setting is on, this target has the
/// interleaved kernels (aarch64), and no GPU backend is active (its kernels read the file's
/// layout).
pub fn repack_wanted() -> bool {
    cfg!(target_arch = "aarch64") && repack_setting() && crate::gpu::active().is_none()
}

/// Whether a `[n_out, n_in]` matrix of `kind` can be interleaved.
pub fn can_interleave(kind: GgmlType, n_out: usize, n_in: usize) -> bool {
    matches!(kind, GgmlType::Q8_0 | GgmlType::Q4_0) && n_out > 0 && n_out.is_multiple_of(4) && n_in.is_multiple_of(32)
}

/// One group of four plain rows (`src`, `4 * row_bytes`) interleaved into `dst` (as long).
fn interleave_group(kind: GgmlType, src: &[u8], dst: &mut [u8]) {
    let (rb, ts) = (src.len() / 4, kind.type_size());
    for b in 0..rb / ts {
        let o = &mut dst[4 * b * ts..4 * (b + 1) * ts];
        for i in 0..4 {
            let s = &src[i * rb + b * ts..i * rb + (b + 1) * ts];
            o[2 * i..2 * i + 2].copy_from_slice(&s[..2]);
            for j in 0..(ts - 2) / 4 {
                o[8 + 16 * j + 4 * i..12 + 16 * j + 4 * i].copy_from_slice(&s[2 + 4 * j..6 + 4 * j]);
            }
        }
    }
}

/// Interleaves `bytes` (a `[n_out, n_in]` matrix of `kind` in the file's layout; see
/// [`can_interleave`]) in place, across the thread pool.
pub fn interleave_in_place(kind: GgmlType, bytes: &mut [u8], n_in: usize) {
    let gb = 4 * kind.row_bytes(n_in);
    assert!(
        bytes.len().is_multiple_of(gb),
        "{kind}: {} bytes are not whole groups of 4 rows",
        bytes.len()
    );
    let groups = bytes.len() / gb;
    // About 256 KiB a task.
    let per = (256 * 1024 / gb).max(1);
    struct Base(*mut u8);
    unsafe impl Sync for Base {}
    impl Base {
        fn get(&self) -> *mut u8 {
            self.0
        }
    }
    let base = Base(bytes.as_mut_ptr());
    crate::pool::run(groups.div_ceil(per), &|t| {
        let mut tmp = vec![0u8; gb];
        for g in t * per..((t + 1) * per).min(groups) {
            // SAFETY: tasks take disjoint groups of `bytes`.
            let grp = unsafe { std::slice::from_raw_parts_mut(base.get().add(g * gb), gb) };
            tmp.copy_from_slice(grp);
            interleave_group(kind, &tmp, grp);
        }
    });
}

/// Block `b` of row `r` in the file's layout (`out`, one block of `kind`), read from the
/// interleaved matrix `data` with rows of `rb` bytes.
#[inline]
pub(super) fn plain_block(kind: GgmlType, data: &[u8], rb: usize, r: usize, b: usize, out: &mut [u8]) {
    let (g, i, ts) = (r / 4, r % 4, kind.type_size());
    let s = &data[g * 4 * rb + 4 * b * ts..g * 4 * rb + 4 * (b + 1) * ts];
    out[..2].copy_from_slice(&s[2 * i..2 * i + 2]);
    for j in 0..(ts - 2) / 4 {
        out[2 + 4 * j..6 + 4 * j].copy_from_slice(&s[8 + 16 * j + 4 * i..12 + 16 * j + 4 * i]);
    }
}

/// Blocks `b0..b1` of row `r` in the file's layout into `out` (`(b1 - b0)` blocks).
pub(super) fn plain_blocks(kind: GgmlType, data: &[u8], rb: usize, r: usize, b0: usize, b1: usize, out: &mut [u8]) {
    let ts = kind.type_size();
    for (k, b) in (b0..b1).enumerate() {
        plain_block(kind, data, rb, r, b, &mut out[k * ts..(k + 1) * ts]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Interleaving moves every byte once, and `plain_blocks` reads every row back.
    #[test]
    fn interleave_round_trip() {
        crate::pool::init(3);
        let (n_out, n_in) = (12usize, 5 * 32);
        for kind in [GgmlType::Q8_0, GgmlType::Q4_0] {
            let rb = kind.row_bytes(n_in);
            let mut seed = 5u64;
            let plain = crate::ggml::random_blocks(kind, n_out * n_in / 32, &mut seed);
            let mut x = plain.clone();
            interleave_in_place(kind, &mut x, n_in);
            assert_ne!(x, plain);
            let mut a = x.clone();
            let mut b = plain.clone();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b);
            let mut row = vec![0u8; rb];
            for r in 0..n_out {
                plain_blocks(kind, &x, rb, r, 0, n_in / 32, &mut row);
                assert_eq!(row, plain[r * rb..(r + 1) * rb], "{kind} row {r}");
            }
        }
    }
}
