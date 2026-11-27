//! The release's temporal upscaling (`ltx_pipelines.dfr_pipeline`, its "diffusion fidelity
//! rendering" rounds): the clip is generated on a keyframe canvas (its length padded to whole 24-
//! or 32-frame segments, a generated keyframe slot at every segment border:
//! `VideoGeneratedKeyframeSlots`, extra single-frame rows the transformer denoises with the clip),
//! then each round
//!
//! 1. doubles the latent frames with the temporal upsampler (and the frame rate; the transformer's
//!    rate snaps to 60 above 30: `conditioning_fps`);
//! 2. splits the canvas at the keyframe borders into `2^round` tiles (`TemporalTilePlan`), each
//!    denoised alone on `DISTILLED_SIGMAS[4..]` with the ancestral sampler at eta 0.5: the
//!    borders it holds as guide rows (strength 0.95), a new slot in the middle of each segment,
//!    the soundtrack of its window frozen; a tile after the first starts on the keyframe plane
//!    before its border followed by the previous tile's frames up to it, all pinned
//!    (`tile_prefix`, `lead_in_carryover`), so the tiles meet without blending;
//! 3. stitches the tiles and keeps every keyframe plane for the next round.
//!
//! The canvas' padding is cut at the end: `(frames - 1) 2^rounds + 1` frames. The release's DFR
//! pipeline also runs its second stage with a detailing IC-LoRA (not part of the release's
//! files); here the second stage is the distilled pipeline's.

use super::*;

/// Segment lengths of the keyframe canvas (`SEGMENT_CANDIDATES`).
pub const SEGMENTS: [usize; 2] = [24, 32];
/// The strength of the keyframes a tile holds at its borders (`ANCHOR_KEYFRAME_STRENGTH`).
pub const ANCHOR_STRENGTH: f32 = 0.95;
/// The temporal rounds' ancestral noise (`TEMPORAL_ANCESTRAL_ETA`).
pub const TEMPORAL_ETA: f32 = 0.5;
/// At most two rounds (`temporal_upscalings` 0 to 2).
pub const MAX_ROUNDS: usize = 2;
/// The rounds' schedule (`DISTILLED_SIGMAS[4:]`).
pub const TEMPORAL_SIGMAS: [f32; 5] = [0.975, 0.909375, 0.725, 0.421875, 0.0];

/// `conditioning_fps`: the rate the transformer's positions use (above 30, 60).
pub fn conditioning_fps(playback: f32) -> f32 {
    if playback > 30.0 { 60.0 } else { playback }
}

/// `resolve_canvas`: `frames - 1` padded to whole segments (24 or 32 frames, the one padding
/// least, ties to 32): the canvas' frames, the segment and the keyframe positions (every segment
/// border but frame 0).
pub fn resolve_canvas(frames: usize) -> (usize, usize, Vec<usize>) {
    let content = frames.max(2) - 1;
    let pad = |s: usize| (s - content % s) % s;
    let seg = SEGMENTS.into_iter().min_by_key(|&s| (pad(s), usize::MAX - s)).unwrap_or(24);
    let padded = content + pad(seg);
    (padded + 1, seg, (1..=padded / seg).map(|i| seg * i).collect())
}

/// One tile of a round (`TemporalTile`): its latent frames `start..end`, its pixel frames, the
/// keyframe borders and the new slots it holds (canvas pixel frames).
#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    pub start: usize,
    pub end: usize,
    pub pixel_start: usize,
    pub pixel_end: usize,
    pub anchors: Vec<usize>,
    pub slots: Vec<usize>,
}

/// `TemporalTilePlan`: the canvas of `frames` split at the keyframe borders `seams` (pixel
/// frames) into `tiles` tiles without overlap (leftover segments to the leading tiles), a slot in
/// the middle of every segment.
pub fn tile_plan(seams: &[usize], frames: usize, tiles: usize) -> Vec<Tile> {
    let ts = TIME_SCALE;
    let mut b = vec![0];
    b.extend(seams.iter().map(|p| p / ts));
    let nseg = b.len() - 1;
    let nt = tiles.clamp(1, nseg.max(1));
    let (base, left) = (nseg / nt, nseg % nt);
    let mut prev = 0;
    let global_slots: Vec<usize> = seams
        .iter()
        .map(|&r| {
            let m = (prev + r) / 2;
            prev = r;
            m
        })
        .collect();
    let _ = frames;
    let mut out = Vec::with_capacity(nt);
    let mut cursor = 0;
    for i in 0..nt {
        let resume = b[cursor] + 1;
        let start = if i == 0 { 0 } else { resume };
        cursor += base + (i < left) as usize;
        let end = b[cursor] + 1;
        let (ps, pe) = (start * ts, (end - 1) * ts);
        out.push(Tile {
            start,
            end,
            pixel_start: ps,
            pixel_end: pe,
            anchors: seams.iter().copied().filter(|&p| ps <= p && p <= pe).collect(),
            slots: global_slots.iter().copied().filter(|&p| ps <= p && p <= pe).collect(),
        });
    }
    out
}

/// `TilePrefix`: a tile after the first starts on the keyframe plane at `keyframe` (the last one
/// before its border), then the canvas from latent frame `video_start` on; its first `cells`
/// latent frames are pinned and it denoises from pixel frame `resume`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prefix {
    pub keyframe: usize,
    pub video_start: usize,
    pub cells: usize,
    pub resume: usize,
}

/// `tile_prefix`.
pub fn tile_prefix(seam: usize, planes: impl Iterator<Item = usize>) -> Result<Prefix> {
    let keyframe = planes
        .filter(|&p| p < seam)
        .max()
        .with_context(|| format!("no keyframe plane before frame {seam}"))?;
    if !keyframe.is_multiple_of(TIME_SCALE) || !seam.is_multiple_of(TIME_SCALE) {
        bail!("keyframe {keyframe} and border {seam} must both sit on the x{TIME_SCALE} grid");
    }
    Ok(Prefix {
        keyframe,
        video_start: keyframe / TIME_SCALE + 1,
        cells: 1 + (seam - keyframe) / TIME_SCALE,
        resume: seam + 1,
    })
}

/// `resample_audio_time` over rows: `out` rows sampled linearly over rows `[from, to)`.
pub fn resample_rows(rows: &[f32], width: usize, from: f32, to: f32, out: usize) -> Vec<f32> {
    let n = rows.len() / width;
    let step = (to - from) / out as f32;
    let mut o = Vec::with_capacity(out * width);
    for i in 0..out {
        let p = (from + step * i as f32).clamp(0.0, (n - 1) as f32);
        let lo = p.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let w = p - lo as f32;
        for c in 0..width {
            o.push(rows[lo * width + c] * (1.0 - w) + rows[hi * width + c] * w);
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canvas_and_tiles() {
        assert_eq!(resolve_canvas(121), (121, 24, vec![24, 48, 72, 96, 120]));
        // 96 frames of content: 32 pads nothing (and wins the tie).
        assert_eq!(resolve_canvas(97), (97, 32, vec![32, 64, 96]));
        assert_eq!(resolve_canvas(25).2, vec![24]);
        // Round 1 of a 121-frame canvas: 241 frames, borders at 48, 96, .., 240, two tiles.
        let seams: Vec<usize> = (1..=5).map(|i| 48 * i).collect();
        let t = tile_plan(&seams, 241, 2);
        assert_eq!(t.len(), 2);
        assert_eq!((t[0].start, t[0].end, t[1].start, t[1].end), (0, 19, 19, 31));
        assert_eq!(t[0].anchors, vec![48, 96, 144]);
        assert_eq!(t[0].slots, vec![24, 72, 120]);
        assert_eq!(t[1].slots, vec![168, 216]);
        let p = tile_prefix(96, [48usize, 96, 24, 72].into_iter()).unwrap();
        assert_eq!(
            p,
            Prefix {
                keyframe: 72,
                video_start: 10,
                cells: 4,
                resume: 97
            }
        );
        let r = resample_rows(&[0.0, 1.0, 2.0, 3.0], 1, 0.0, 4.0, 8);
        assert_eq!(r, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.0]);
    }
}
