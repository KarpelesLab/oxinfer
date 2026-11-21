//! MiniMax-H3's `ref2va` task (the release's `Ref2VA/` partition, diffusers' `transformer_ref`):
//! a video with its soundtrack generated from an ordered list of references — images, videos
//! (with their own soundtrack) and audio clips — and a prompt, following diffusers' modular
//! `minimax_h3` pipeline (`MiniMaxH3Ref2VA*` steps):
//!
//! - normalization ([`normalize`]): an image to a short edge of its own (2048, upscaling
//!   included, no area cap; sides multiples of 32; within 1:4 and 4:1) with Lanczos; a video onto
//!   24 fps by holding every frame until the next one's slot, truncated to the generated frame
//!   count, then (Lanczos) onto the canvas its own aspect ratio resolves to (the generated video's
//!   rule: 768 short edge, 1344 x 768 pixels at most); a soundtrack truncated at its own rate to
//!   the generated duration, a mono one doubled to stereo, then resampled to 32 kHz as torchaudio
//!   does. The references never bind the generated canvas (16:9 unless asked otherwise);
//! - the presentation ([`encode_presentation`]): per reference in order, numbered per modality,
//!   `"<Audio j>: "` for one that carries sound (a video's soundtrack is labelled before its
//!   video), `"<Picture i>: "` and a vision block for an image, `"<Video k>: "` then, for every
//!   pair of the frames sampled at 2 fps (every 12th of the 24 fps frames; the last repeated to a
//!   whole pair), `"<t seconds>"` (the pair's mean time, one decimal, ties to even) and a vision
//!   block through Qwen3-VL's video processor; then the prompt verbatim. Vision blocks are tagged
//!   as video, as for keyframes;
//! - the references through the VAEs: images and videos (a video's frames cut down to
//!   `17 n + 5`) through the video VAE's encoder, the posterior sampled under a generator of its
//!   own and rounded through float16 like a keyframe; soundtracks through the audio VAE's encoder
//!   (the posterior mean), normalized, channel-major;
//! - one packed sequence `[text | reference blocks | target audio | target video]`
//!   ([`layout`]): an image block is one latent frame on its own aspect-normalized grid at one
//!   integer slot of the shared audio / video clock; an audio block is pinned to the target
//!   grid's left and right ends and advances the clock by its latent count; a video block's
//!   soundtrack rows come first, pinned to the ends of the video's own grid, and share its origin,
//!   its frames spaced `5/3 * (1, 4, 4, 4, 4)`, and it advances the clock by the longer of the
//!   two; the generated rows start where the references leave the clock;
//! - the denoising loop of `fl2va` ([`super::video::denoise_layout`]): image and video
//!   reference rows noised to `t = 0.999` and held there, soundtrack rows clean at `t = 1`; only
//!   the generated rows are stepped. One generator: each visual reference's conditioning noise in
//!   packed order, then the video noise, then the audio noise.

use super::audio_vae::AudioEncoder;
use super::dit::{TAG_AUDIO, TAG_IMAGE, TAG_TEXT};
use super::video::{
    AUDIO_CHANNELS, Conditioning, FPS, FRAME_RESCALE, FRAMES_PER_LATENT, Geometry, Image, KEYFRAME_T, Layout, Video, VideoParams,
    VideoPipeline, denoise_layout, patchify_video, read_latents, resolve_canvas, sigmas, write_latents,
};
use crate::clock::Instant;
use crate::sd::ops::Rng;
use anyhow::{Context, Result, bail};

/// What the release documents for one request.
pub const MAX_IMAGES: usize = 9;
pub const MAX_VIDEOS: usize = 3;
pub const MAX_AUDIOS: usize = 3;
pub const MAX_REFERENCES: usize = 12;
/// The rate the text encoder reads a reference video at.
pub const VIDEO_SAMPLE_FPS: f64 = 2.0;
/// The short edge an image reference is encoded at.
pub const IMAGE_SHORT_EDGE: usize = 2048;

/// A reference as decoded from its file, at its own rates.
pub enum RefInput {
    Image(Image),
    /// Frames (all the same size) at `fps`, and the soundtrack (one waveform per channel) at its
    /// sample rate.
    Video {
        frames: Vec<Image>,
        fps: f64,
        audio: Option<(Vec<Vec<f32>>, u32)>,
    },
    Audio(Vec<Vec<f32>>, u32),
}

/// A reference normalized onto MiniMax-H3's rates and resolutions ([`normalize`]).
pub enum Reference {
    Image(Image),
    /// Frames at 24 fps on their canvas; the soundtrack at the audio VAE's rate, stereo.
    Video {
        frames: Vec<Image>,
        audio: Option<[Vec<f32>; 2]>,
    },
    Audio([Vec<f32>; 2]),
}

impl Reference {
    pub fn has_audio(&self) -> bool {
        matches!(self, Reference::Audio(_) | Reference::Video { audio: Some(_), .. })
    }

    fn audio(&self) -> Option<&[Vec<f32>; 2]> {
        match self {
            Reference::Audio(a) | Reference::Video { audio: Some(a), .. } => Some(a),
            _ => None,
        }
    }
}

/// The settings of [`normalize`] (the release's: [`RefOptions::default`]).
#[derive(Clone, Copy, Debug)]
pub struct RefOptions {
    /// The short edge an image reference is encoded at.
    pub image_short_edge: usize,
    /// The canvas rule a video reference is put on (the generated video's).
    pub short_edge: usize,
    pub max_pixels: usize,
    /// What both canvas axes round to.
    pub multiple: usize,
}

impl Default for RefOptions {
    fn default() -> Self {
        Self {
            image_short_edge: IMAGE_SHORT_EDGE,
            short_edge: 768,
            max_pixels: 768 * 1344,
            multiple: 32,
        }
    }
}

fn resize(im: &Image, w: usize, h: usize) -> Image {
    if (im.width, im.height) == (w, h) {
        return im.clone();
    }
    Image {
        width: w,
        height: h,
        rgb: crate::resample::resize_lanczos_rgb8(&im.rgb, im.width, im.height, w, h),
    }
}

/// The frames of a `fps` clip on the 24 fps grid: frame `i` held from slot
/// `floor(i * 24 / fps + 0.5)` until the next frame's (`ffmpeg`'s `fps` filter).
pub fn resample_frames<T: Clone>(frames: &[T], fps: f64) -> Vec<T> {
    if fps == FPS as f64 {
        return frames.to_vec();
    }
    let scale = FPS as f64 / fps;
    let slot = |i: usize| (i as f64 * scale + 0.5).floor() as usize;
    let mut out = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        out.extend(std::iter::repeat_n(f.clone(), slot(i + 1).saturating_sub(slot(i))));
    }
    out
}

/// A soundtrack onto `rate` Hz stereo, truncated at its own rate to `secs` seconds.
fn normalize_audio(channels: &[Vec<f32>], from: u32, rate: u32, secs: f64) -> Result<[Vec<f32>; 2]> {
    if channels.is_empty() || channels.len() > 2 || channels[0].is_empty() {
        bail!("a reference soundtrack must be mono or stereo, got {} channels", channels.len());
    }
    let keep = ((secs * from as f64) as usize).min(channels[0].len());
    let cut = |c: &Vec<f32>| crate::zonos::speaker::resample_sinc(&c[..keep], from as usize, rate as usize);
    let left = cut(&channels[0]);
    let right = if channels.len() == 2 { cut(&channels[1]) } else { left.clone() };
    Ok([left, right])
}

/// Normalizes `refs` (in the order the model reads them) for a request of `frames` frames whose
/// soundtrack is at `rate` Hz; checks the release's limits.
pub fn normalize(refs: Vec<RefInput>, frames: usize, rate: u32, o: &RefOptions) -> Result<Vec<Reference>> {
    if refs.is_empty() {
        bail!("ref2va needs at least one reference");
    }
    let count = |f: fn(&RefInput) -> bool| refs.iter().filter(|r| f(r)).count();
    let (ni, nv, na) = (
        count(|r| matches!(r, RefInput::Image(_))),
        count(|r| matches!(r, RefInput::Video { .. })),
        count(|r| matches!(r, RefInput::Audio(..))),
    );
    for (n, max, what) in [(ni, MAX_IMAGES, "image"), (nv, MAX_VIDEOS, "video"), (na, MAX_AUDIOS, "audio")] {
        if n > max {
            bail!("MiniMax-H3 accepts at most {max} {what} references, got {n}");
        }
    }
    if refs.len() > MAX_REFERENCES {
        bail!(
            "MiniMax-H3 accepts at most {MAX_REFERENCES} references in total, got {}",
            refs.len()
        );
    }
    if na == refs.len() {
        bail!("an audio reference has to be paired with at least one image or video reference");
    }
    let secs = frames as f64 / FPS as f64;
    refs.into_iter()
        .map(|r| {
            Ok(match r {
                RefInput::Image(im) => {
                    if im.width == 0 || im.height == 0 {
                        bail!("an empty reference image");
                    }
                    if im.width > 4 * im.height || im.height > 4 * im.width {
                        bail!("a reference image must be within 1:4 and 4:1, got {}x{}", im.width, im.height);
                    }
                    let scale = o.image_short_edge as f64 / im.width.min(im.height) as f64;
                    let side = |v: usize| {
                        o.multiple
                            .max((v as f64 * scale / o.multiple as f64).round_ties_even() as usize * o.multiple)
                    };
                    Reference::Image(resize(&im, side(im.width), side(im.height)))
                }
                RefInput::Video { frames: fr, fps, audio } => {
                    if fr.is_empty() || fps <= 0.0 {
                        bail!("a reference video needs frames and a positive frame rate");
                    }
                    let mut fr = resample_frames(&fr, fps);
                    fr.truncate(frames);
                    let (w0, h0) = (fr[0].width, fr[0].height);
                    let (h, w) = resolve_canvas(w0 as f64, h0 as f64, o.short_edge, o.max_pixels, o.multiple)?;
                    let frames_out = fr.iter().map(|f| resize(f, w, h)).collect();
                    let audio = audio.map(|(c, from)| normalize_audio(&c, from, rate, secs)).transpose()?;
                    Reference::Video { frames: frames_out, audio }
                }
                RefInput::Audio(c, from) => Reference::Audio(normalize_audio(&c, from, rate, secs)?),
            })
        })
        .collect()
}

/// The indices of the frames the text encoder reads from a 24 fps reference of `n` frames (every
/// 12th) and the timestamp of every pair of them (the mean time; the last frame repeated to a
/// whole pair).
pub fn sample_video_frames(n: usize, temporal_patch: usize) -> Result<(Vec<usize>, Vec<f64>)> {
    let stride = FPS as f64 / VIDEO_SAMPLE_FPS;
    let (mut idx, mut cursor) = (Vec::<usize>::new(), 0f64);
    while (cursor.round_ties_even() as usize) < n {
        let i = cursor.round_ties_even() as usize;
        if idx.last().is_none_or(|&l| i > l) {
            idx.push(i);
        }
        cursor += stride;
    }
    if idx.len() < temporal_patch {
        let min = ((temporal_patch - 1) as f64 * stride).round() as usize + 1;
        bail!(
            "a reference video is read at 2 fps in groups of {temporal_patch} frames, so it must run at least {min} frames at 24 fps, got {n}"
        );
    }
    let mut ts: Vec<f64> = (0..idx.len()).map(|i| i as f64 / VIDEO_SAMPLE_FPS).collect();
    while !ts.len().is_multiple_of(temporal_patch) {
        ts.push(*ts.last().unwrap());
    }
    let blocks = ts.chunks(temporal_patch).map(|c| (c[0] + c[temporal_patch - 1]) / 2.0).collect();
    Ok((idx, blocks))
}

/// A block's timestamp as the presentation writes it (`"<{t:.1f} seconds>"`, Python rounding
/// the binary value half to even, as Rust's formatting does).
pub fn timestamp_label(t: f64) -> String {
    format!("<{t:.1} seconds>")
}

/// The conditioning of a `ref2va` request: the presentation of `refs` (normalized) and `prompt`
/// after the 50th layer of the text encoder (loaded with its vision tower).
pub fn encode_presentation(vl: &crate::qwen_vl::QwenVl, prompt: &str, refs: &[Reference]) -> Result<Conditioning> {
    let (mut ids, mut tags, mut blocks) = (Vec::new(), Vec::new(), Vec::new());
    let text = |ids: &mut Vec<u32>, tags: &mut Vec<u8>, s: &str| -> Result<()> {
        let t = vl.tokenize(s)?;
        tags.extend(std::iter::repeat_n(TAG_TEXT, t.len()));
        ids.extend(t);
        Ok(())
    };
    let vision = |ids: &mut Vec<u32>, tags: &mut Vec<u8>, im: &crate::qwen_vl::VlImage| -> Result<()> {
        let t = vl.tokenize(&im.placeholder())?;
        tags.extend(std::iter::repeat_n(TAG_IMAGE, t.len()));
        ids.extend(t);
        Ok(())
    };
    let temporal = vl
        .mm
        .as_ref()
        .context("the text encoder has no vision configuration")?
        .video
        .temporal_patch_size;
    let (mut n_image, mut n_video, mut n_audio) = (0, 0, 0);
    for r in refs {
        if r.has_audio() {
            n_audio += 1;
            text(&mut ids, &mut tags, &format!("<Audio {n_audio}>: "))?;
        }
        match r {
            Reference::Image(im) => {
                n_image += 1;
                text(&mut ids, &mut tags, &format!("<Picture {n_image}>: "))?;
                let b = vl.encode_image(&im.rgb, im.width, im.height, 0)?;
                vision(&mut ids, &mut tags, &b)?;
                blocks.push(b);
            }
            Reference::Video { frames, .. } => {
                n_video += 1;
                text(&mut ids, &mut tags, &format!("<Video {n_video}>: "))?;
                let (idx, times) = sample_video_frames(frames.len(), temporal)?;
                let picked: Vec<&[u8]> = idx.iter().map(|&i| frames[i].rgb.as_slice()).collect();
                let vb = vl.encode_video(&picked, frames[0].width, frames[0].height)?;
                if vb.len() != times.len() {
                    bail!(
                        "the video processor made {} blocks of a reference, MiniMax-H3 labels {}",
                        vb.len(),
                        times.len()
                    );
                }
                for (b, &t) in vb.into_iter().zip(&times) {
                    text(&mut ids, &mut tags, &timestamp_label(t))?;
                    vision(&mut ids, &mut tags, &b)?;
                    blocks.push(b);
                }
            }
            Reference::Audio(_) => {}
        }
    }
    text(&mut ids, &mut tags, prompt)?;
    let mut hs = vl.hidden_states_tokens(ids, &blocks, None, &[super::video::TEXT_LAYER - 1])?;
    let (_, text) = hs.layers.pop().context("layer 50 not captured")?;
    Ok(Conditioning { text, tags })
}

/// The encoded conditioning of one image or video reference: normalized latents `[C, t, h, w]`.
pub struct CondLatents {
    pub lat: Vec<f32>,
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

/// One reference's place in the packed sequence: its visual latent geometry `(t, h, w)` and its
/// soundtrack's latents per channel.
#[derive(Clone, Copy, Debug)]
pub struct RefBlock {
    pub visual: Option<(usize, usize, usize)>,
    pub audio: usize,
    pub image: bool,
}

/// `build_ref2va_packed_sequence`: `[text | reference blocks | target audio | target video]` for
/// the target geometry `g` (see the module docs).
pub fn layout(text_tags: &[u8], refs: &[RefBlock], g: Geometry, patch: usize) -> Layout {
    let nt = text_tags.len();
    let grid = |lh: usize, lw: usize| {
        let sa = ((lh * lw) as f64).sqrt();
        (super::axis_grid(lh, patch, sa), super::axis_grid(lw, patch, sa))
    };
    let rows = |(t, h, w): (usize, usize, usize)| t * (h / patch) * (w / patch);
    let n_ref_video: usize = refs.iter().filter_map(|r| r.visual).map(rows).sum();
    let n_ref_audio: usize = refs.iter().map(|r| r.audio * AUDIO_CHANNELS).sum();
    let n_target_video = rows((g.lt, g.lh, g.lw));
    let n_target_audio = g.n_audio * AUDIO_CHANNELS;
    let n = nt + n_ref_video + n_ref_audio + n_target_audio + n_target_video;
    let mut pos = vec![[0f64; 3]; n];
    let mut tags = vec![TAG_IMAGE; n];
    tags[..nt].copy_from_slice(text_tags);
    for (i, p) in pos.iter_mut().take(nt).enumerate() {
        p[0] = i as f64;
    }
    let (_, tx) = grid(g.lh, g.lw);
    let (mut video_index, mut audio_index) = (Vec::new(), Vec::new());
    // Audio rows: per channel, `n` latents from `time`, pinned to the ends of `xs`.
    let audio = |pos: &mut [[f64; 3]], tags: &mut [u8], idx: &mut Vec<usize>, start: usize, n: usize, time: f64, xs: &[f64]| {
        for i in 0..n * AUDIO_CHANNELS {
            let x = if i < n { xs[0] } else { xs[xs.len() - 1] };
            pos[start + i] = [time + (i % n) as f64, 0.0, x];
            tags[start + i] = TAG_AUDIO;
            idx.push(start + i);
        }
    };
    // Video rows: `t` latent frames of `h x w` from `time`.
    let frames = |pos: &mut [[f64; 3]], idx: &mut Vec<usize>, start: usize, (t, h, w): (usize, usize, usize), time: f64| {
        let (ys, xs) = grid(h, w);
        let (gh, gw) = (h / patch, w / patch);
        let mut cum = 0f64;
        for f in 0..t {
            for k in 0..gh * gw {
                let r = start + f * gh * gw + k;
                pos[r] = [time + cum, ys[k / gw], xs[k % gw]];
                idx.push(r);
            }
            cum += FRAME_RESCALE * FRAMES_PER_LATENT[f % 5];
        }
    };
    let (mut cursor, mut clock) = (nt, nt as f64);
    for r in refs {
        match (r.visual, r.image) {
            (Some(v), true) => {
                frames(&mut pos, &mut video_index, cursor, v, clock);
                cursor += rows(v);
                // An image takes one integer slot of the clock.
                clock += 1.0;
            }
            (None, _) => {
                audio(&mut pos, &mut tags, &mut audio_index, cursor, r.audio, clock, &tx);
                cursor += r.audio * AUDIO_CHANNELS;
                clock += r.audio as f64;
            }
            (Some(v), false) => {
                let (_, xs) = grid(v.1, v.2);
                audio(&mut pos, &mut tags, &mut audio_index, cursor, r.audio, clock, &xs);
                cursor += r.audio * AUDIO_CHANNELS;
                frames(&mut pos, &mut video_index, cursor, v, clock);
                cursor += rows(v);
                // Summed in order, as the reference does at this call site.
                let span: f64 = (0..v.0).map(|f| FRAME_RESCALE * FRAMES_PER_LATENT[f % 5]).fold(0.0, |a, b| a + b);
                clock += (r.audio as f64).max(span);
            }
        }
    }
    let (n_cond, n_cond_audio) = (video_index.len(), audio_index.len());
    audio(&mut pos, &mut tags, &mut audio_index, cursor, g.n_audio, clock, &tx);
    frames(&mut pos, &mut video_index, cursor + n_target_audio, (g.lt, g.lh, g.lw), clock);
    Layout {
        positions: pos.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect(),
        tags,
        text_index: (0..nt).collect(),
        video_index,
        audio_index,
        n_cond,
        n_cond_audio,
    }
}

impl VideoPipeline {
    /// The references through the VAEs: the visual conditioning of every image and video
    /// reference and the normalized soundtrack rows of every reference that carries one, in
    /// packed order.
    pub fn encode_references(&self, refs: &[Reference]) -> Result<(Vec<CondLatents>, Vec<Vec<f32>>)> {
        let enc = self.encoder()?;
        let aenc: Option<AudioEncoder> = if refs.iter().any(Reference::has_audio) {
            Some(self.audio_encoder()?)
        } else {
            None
        };
        let (tcs, _, _, _) = self.vae.chunking();
        let clip = self.vae.cfg.clip_length;
        let (mut vis, mut aud) = (Vec::new(), Vec::new());
        for r in refs {
            let frames: Vec<&Image> = match r {
                Reference::Image(im) => vec![im],
                Reference::Video { frames, .. } => {
                    // Cut down to `17 n + 5` so the encoder does not pad.
                    let n = ((frames.len().saturating_sub(tcs) / clip).max(1) * clip + tcs).min(frames.len());
                    frames[..n].iter().collect()
                }
                Reference::Audio(_) => Vec::new(),
            };
            if let Some(f0) = frames.first() {
                let px: Vec<&[u8]> = frames.iter().map(|f| f.rgb.as_slice()).collect();
                let (lat, t, h, w) = self.encode_frames(&enc, &px, f0.width, f0.height)?;
                vis.push(CondLatents { lat, t, h, w });
            }
            if let (Some(a), Some(e)) = (r.audio(), &aenc) {
                aud.push(e.encode_rows(a)?);
            }
        }
        Ok((vis, aud))
    }

    /// References (normalized, [`normalize`]) and a prompt to video. `cond`: the presentation
    /// computed earlier ([`encode_presentation`]) instead of running the text encoder.
    pub fn generate_ref(
        &self,
        p: &VideoParams,
        refs: &[Reference],
        cond: Option<Conditioning>,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<Video> {
        if self.partition.as_deref() == Some("fl2va") {
            bail!("references need the release's Ref2VA/ transformer; FL2VA/ generates from text and keyframes");
        }
        if p.steps < 2 {
            bail!("--steps must be at least 2 (the sigma grid includes its end)");
        }
        if p.first.is_some() || p.last.is_some() {
            bail!("ref2va takes references, not keyframes");
        }
        let (h, w, frames, g) = self.geometry(p)?;
        if let Some(f) = p.latents.as_ref().filter(|f| f.exists()) {
            let (video, audio) = read_latents(f, g)?;
            let mut out = self.decode(&video, g)?;
            out.audio = self.decode_audio(&audio)?;
            eprintln!("[MiniMax-H3 ref2va: {w}x{h}, {frames} frames from the rows in {}]", f.display());
            return Ok(out);
        }
        let t0 = Instant::now();
        let cond = match cond {
            Some(c) => c,
            None => {
                let vl = super::video::load_text_encoder(&self.dir, &self.text_encoder_dir, self.text_quant, true)?;
                encode_presentation(&vl, &p.prompt, refs)?
            }
        };
        let t_enc = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let (vis, aud) = self.encode_references(refs)?;
        let t_ref = t1.elapsed().as_secs_f64();
        let cfg = &self.dit.cfg;
        let (c, patch) = (self.vae.cfg.latent_channels, cfg.patch[1]);
        let ac = cfg.audio_channels;
        let (mut vi, mut ai) = (vis.iter(), aud.iter());
        let blocks: Vec<RefBlock> = refs
            .iter()
            .map(|r| RefBlock {
                visual: (!matches!(r, Reference::Audio(_))).then(|| {
                    let v = vi.next().unwrap();
                    (v.t, v.h, v.w)
                }),
                audio: if r.has_audio() {
                    ai.next().unwrap().len() / ac / AUDIO_CHANNELS
                } else {
                    0
                },
                image: matches!(r, Reference::Image(_)),
            })
            .collect();
        let lay = layout(&cond.tags, &blocks, g, patch);
        // One generator: the conditioning noise of each visual reference, the video noise, the
        // audio noise.
        let mut rng = Rng::new(p.seed);
        let mut video = Vec::new();
        for v in &vis {
            let noise = rng.normals(v.lat.len());
            let noised: Vec<f32> = v
                .lat
                .iter()
                .zip(&noise)
                .map(|(&x, &n)| KEYFRAME_T * x + (1.0 - KEYFRAME_T) * n)
                .collect();
            video.extend(patchify_video(&noised, c, v.t, v.h, v.w, patch));
        }
        let n_cond_vals = video.len();
        debug_assert_eq!(n_cond_vals, lay.n_cond * cfg.patch_dim());
        let plane = g.lh * g.lw;
        video.extend(patchify_video(&rng.normals(c * g.lt * plane), c, g.lt, g.lh, g.lw, patch));
        let mut audio: Vec<f32> = aud.concat();
        let n_cond_audio_vals = audio.len();
        audio.extend(rng.normals(g.n_audio * AUDIO_CHANNELS * ac));
        eprintln!(
            "[MiniMax-H3 ref2va: {} references, {} text rows, {} reference video + {} reference audio rows, {} rows in all; text {t_enc:.1}s, references {t_ref:.1}s]",
            refs.len(),
            cond.tags.len(),
            lay.n_cond,
            lay.n_cond_audio,
            lay.positions.len()
        );
        let (sv, sa) = (sigmas(p.steps, self.shifts.0), sigmas(p.steps, self.shifts.1));
        let t2 = Instant::now();
        let (video, audio) = denoise_layout(&self.dit, &cond.text, &lay, video, audio, (&sv, &sa), progress)?;
        let t_dit = t2.elapsed().as_secs_f64();
        let (video, audio) = (&video[n_cond_vals..], &audio[n_cond_audio_vals..]);
        if let Some(f) = &p.latents {
            write_latents(f, g, video, audio)?;
        }
        let t3 = Instant::now();
        let mut out = self.decode(video, g)?;
        let t_vae = t3.elapsed().as_secs_f64();
        out.audio = self.decode_audio(audio)?;
        eprintln!(
            "\n[MiniMax-H3 ref2va: {w}x{h}, {frames} frames ({} latent), {} steps {t_dit:.1}s, VAE {t_vae:.1}s]",
            g.lt,
            sv.len().min(sa.len()) - 1
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_round_half_to_even() {
        // Python's "{:.1f}" of the 2 fps pairs' means.
        assert_eq!(timestamp_label(0.25), "<0.2 seconds>");
        assert_eq!(timestamp_label(1.25), "<1.2 seconds>");
        assert_eq!(timestamp_label(0.75), "<0.8 seconds>");
        assert_eq!(timestamp_label(2.0), "<2.0 seconds>");
    }

    #[test]
    fn video_sampling() {
        // 124 frames: indices 0, 12, ..., 120 (11 frames), the last repeated to 6 pairs.
        let (idx, ts) = sample_video_frames(124, 2).unwrap();
        assert_eq!(idx, (0..11).map(|i| i * 12).collect::<Vec<_>>());
        assert_eq!(ts, vec![0.25, 1.25, 2.25, 3.25, 4.25, 5.0]);
        assert!(sample_video_frames(12, 2).is_err());
        assert_eq!(sample_video_frames(13, 2).unwrap().0, vec![0, 12]);
    }

    #[test]
    fn frame_rate_resampling() {
        // 30 fps to 24: 5 frames fill 4 slots (frame 2 dropped), as ffmpeg's fps filter does.
        let f: Vec<usize> = (0..5).collect();
        assert_eq!(resample_frames(&f, 30.0), vec![0, 1, 3, 4]);
        // 12 fps to 24: every frame twice.
        assert_eq!(resample_frames(&[7, 8], 12.0), vec![7, 7, 8, 8]);
    }
}
