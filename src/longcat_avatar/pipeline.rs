//! The generation loop of `LongCatVideoAvatarPipeline` for the 1.5 / 1.6 checkpoints
//! (`run_demo_avatar_single_audio_to_video.py` / `run_demo_avatar_multi_audio_to_video.py` with
//! `--model_type avatar-v1.6`): see the module docs of [`super`].

use super::audio::{AudioEncoder, clip_window};
use super::dit::{Cond, Dit};
use super::lora::{Active, LoraNet};
use super::vae::VideoVae;
use crate::clock::Instant;
use crate::flux::t5::T5Encoder;
use crate::model::Quant;
use crate::sd::Image;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// Frames per second of the 1.5 / 1.6 checkpoints (one audio row per frame).
pub const FPS: usize = 25;
/// The driving audio's sample rate.
pub const SPEECH_RATE: usize = 16_000;
/// Frames of a clip the next one is conditioned on.
pub const COND_FRAMES: usize = 13;
/// The demos' negative prompt (used with classifier-free guidance only).
pub const NEGATIVE_PROMPT: &str = "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards";

/// Where the parts are.
#[derive(Clone, Debug)]
pub struct Sources {
    /// The avatar release: `base_model/` or `base_model_int8/`, `loraA/`, `loraB/`, `scheduler/`,
    /// `whisper-large-v3/`.
    pub avatar: PathBuf,
    /// `meituan-longcat/LongCat-Video`: `text_encoder/`, `tokenizer/`, `vae/`.
    pub base: PathBuf,
    /// The int8 transformer (`base_model_int8/`) instead of the bf16 one.
    pub int8: bool,
    pub text_quant: Quant,
}

/// One request.
#[derive(Clone)]
pub struct AvatarParams {
    pub prompt: String,
    pub negative_prompt: String,
    /// The reference image (None: speech and text to video).
    pub image: Option<Image>,
    /// The driving speech per speaker, 16 kHz mono (one, or two with `regions`).
    pub speech: Vec<Vec<f32>>,
    /// Two speakers: each one's box in the reference image as `[y0, x0, y1, x1]` pixels (None:
    /// the left and right halves, a tenth in from their edges).
    pub regions: Option<[[usize; 4]; 2]>,
    /// The 720p buckets (around 768x1280) instead of the 480p ones (around 480x832).
    pub hd: bool,
    /// Width and height without a reference image (default 832x480, or 1280x768 with `hd`).
    pub size: Option<(usize, usize)>,
    /// Frames per clip (`4 n + 1`; the release's 93).
    pub frames: usize,
    /// Clips: each one after the first continues the previous one's last 13 frames and adds
    /// `frames - 13`.
    pub segments: usize,
    pub steps: usize,
    /// The distilled schedule with the two LoRAs (else the base model alone).
    pub distill: bool,
    pub text_guidance: f32,
    pub audio_guidance: f32,
    /// The first `mix_steps` steps run `mix_alpha` DMD + `(1 - mix_alpha)` OPSD, the others OPSD.
    pub mix_steps: usize,
    pub mix_alpha: f32,
    /// The temporal position a continuation places the reference frame at.
    pub ref_img_index: usize,
    pub mask_frame_range: usize,
    pub seed: u64,
}

impl Default for AvatarParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: NEGATIVE_PROMPT.to_string(),
            image: None,
            speech: Vec::new(),
            regions: None,
            hd: false,
            size: None,
            frames: 93,
            segments: 1,
            steps: 8,
            distill: true,
            text_guidance: 1.0,
            audio_guidance: 1.0,
            mix_steps: 4,
            mix_alpha: 1.0,
            ref_img_index: 10,
            mask_frame_range: 0,
            seed: 42,
        }
    }
}

/// What a run is doing, for progress reports.
#[derive(Clone, Copy, Debug)]
pub enum Stage {
    Text,
    Audio {
        done: usize,
        total: usize,
    },
    Encode,
    Step {
        segment: usize,
        segments: usize,
        step: usize,
        steps: usize,
        secs: f64,
    },
    Decode {
        segment: usize,
        done: usize,
        total: usize,
    },
}

/// Generated frames (RGB8, row-major) at [`FPS`].
pub struct Clip {
    pub width: usize,
    pub height: usize,
    pub frames: Vec<Vec<u8>>,
    /// The normalized latents of the last clip (`[16, t, h / 8, w / 8]`).
    pub latents: Vec<f32>,
}

/// The aspect-ratio buckets of the reference (`ASPECT_RATIO_627` for 480p, `ASPECT_RATIO_960`
/// for 720p): ratio (height / width) as its key string, height, width.
const BUCKETS_480: &[(&str, usize, usize)] = &[
    ("0.26", 320, 1216),
    ("0.31", 352, 1120),
    ("0.38", 384, 1024),
    ("0.43", 416, 960),
    ("0.52", 448, 864),
    ("0.58", 480, 832),
    ("0.67", 512, 768),
    ("0.74", 544, 736),
    ("0.86", 576, 672),
    ("0.95", 608, 640),
    ("1.05", 640, 608),
    ("1.17", 672, 576),
    ("1.29", 704, 544),
    ("1.35", 736, 544),
    ("1.50", 768, 512),
    ("1.67", 800, 480),
    ("1.73", 832, 480),
    ("2.00", 896, 448),
    ("2.31", 960, 416),
    ("2.58", 992, 384),
    ("2.75", 1056, 384),
    ("3.09", 1088, 352),
    ("3.70", 1184, 320),
    ("3.80", 1216, 320),
    ("3.90", 1248, 320),
    ("4.00", 1280, 320),
];
const BUCKETS_720: &[(&str, usize, usize)] = &[
    ("0.25", 480, 1920),
    ("0.29", 512, 1792),
    ("0.32", 544, 1696),
    ("0.36", 576, 1600),
    ("0.40", 608, 1504),
    ("0.49", 672, 1376),
    ("0.54", 704, 1312),
    ("0.59", 736, 1248),
    ("0.69", 800, 1152),
    ("0.74", 832, 1120),
    ("0.82", 864, 1056),
    ("0.88", 896, 1024),
    ("0.94", 928, 992),
    ("1.00", 960, 960),
    ("1.07", 992, 928),
    ("1.14", 1024, 896),
    ("1.22", 1056, 864),
    ("1.31", 1088, 832),
    ("1.35", 1120, 832),
    ("1.44", 1152, 800),
    ("1.70", 1248, 736),
    ("2.00", 1344, 672),
    ("2.05", 1376, 672),
    ("2.47", 1504, 608),
    ("2.53", 1536, 608),
    ("2.83", 1632, 576),
    ("3.06", 1664, 544),
    ("3.12", 1696, 544),
    ("3.62", 1856, 512),
    ("3.93", 1888, 480),
    ("4.00", 1920, 480),
];

/// The bucket (height, width) closest to an image's aspect ratio (`get_condition_shape`).
pub fn bucket(height: usize, width: usize, hd: bool) -> (usize, usize) {
    let ratio = height as f64 / width as f64;
    let list = if hd { BUCKETS_720 } else { BUCKETS_480 };
    let mut best = list[0];
    let mut bd = f64::INFINITY;
    for &b in list {
        let d = (b.0.parse::<f64>().unwrap() - ratio).abs();
        if d < bd {
            (best, bd) = (b, d);
        }
    }
    (best.1, best.2)
}

/// The sigmas of a run: the distilled schedule (the 8 points 1, 874/999, ... of a 1000-point
/// grid, whatever `steps` says: the reference always runs all of them) or
/// `linspace(1, 0.001, steps)`, shifted (`shift s / (1 + (shift - 1) s)`), with the final 0.
pub fn sigmas(steps: usize, distill: bool, shift: f32) -> Vec<f32> {
    let raw: Vec<f32> = if distill {
        // `torch.linspace(0, 1, 1000)` flipped, at indices 0, 125, ..., 875.
        (0..8).map(|k| 1.0 - (125 * k) as f32 / 999.0).collect()
    } else if steps == 1 {
        vec![1.0]
    } else {
        // `torch.linspace(1, 0.001, steps)`: from the start in the first half, from the end in the second.
        let step = (0.001f32 - 1.0) / (steps - 1) as f32;
        (0..steps)
            .map(|i| {
                if i < steps / 2 {
                    1.0 + step * i as f32
                } else {
                    0.001 - step * (steps - 1 - i) as f32
                }
            })
            .collect()
    };
    let mut s: Vec<f32> = raw.iter().map(|&v| shift * v / (1.0 + (shift - 1.0) * v)).collect();
    s.push(0.0);
    s
}

/// `html.unescape` for the entities prompts carry.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let end = rest[..rest.len().min(12)].find(';');
        let rep = end.and_then(|e| {
            let name = &rest[1..e];
            let c = match name {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some('\u{a0}'),
                _ => name
                    .strip_prefix("#x")
                    .or_else(|| name.strip_prefix("#X"))
                    .map(|h| u32::from_str_radix(h, 16))
                    .or_else(|| name.strip_prefix('#').map(|d| d.parse::<u32>()))
                    .and_then(|r| r.ok())
                    .and_then(char::from_u32),
            };
            c.map(|c| (c, e + 1))
        });
        match rep {
            Some((c, used)) => {
                out.push(c);
                rest = &rest[used..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Halfwidth katakana U+FF61..U+FF9F as ftfy's width map (NFKC) gives them.
const HALF_KATAKANA: &str = "。「」、・ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン\u{3099}\u{309a}";

/// The deterministic character fixes of `ftfy.fix_text`'s defaults: fullwidth ASCII, the
/// ideographic space and halfwidth katakana to their usual forms (`fix_character_width`), curly
/// quotes to straight ones, Latin ligatures spelled out.
fn fix_chars(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\u{3000}' => out.push(' '),
            '\u{ff01}'..='\u{ff5e}' => out.push(char::from_u32(c as u32 - 0xfee0).unwrap()),
            '\u{ff61}'..='\u{ff9f}' => out.push(HALF_KATAKANA.chars().nth((c as u32 - 0xff61) as usize).unwrap()),
            '\u{2bc}' | '\u{2018}'..='\u{201b}' => out.push('\''),
            '\u{201c}'..='\u{201f}' => out.push('"'),
            '\u{132}' => out.push_str("IJ"),
            '\u{133}' => out.push_str("ij"),
            '\u{fb00}' => out.push_str("ff"),
            '\u{fb01}' => out.push_str("fi"),
            '\u{fb02}' => out.push_str("fl"),
            '\u{fb03}' => out.push_str("ffi"),
            '\u{fb04}' => out.push_str("ffl"),
            '\u{fb05}' => out.push_str("\u{17f}t"),
            '\u{fb06}' => out.push_str("st"),
            _ => out.push(c),
        }
    }
    out
}

/// The reference's `prompt_clean`: `ftfy.fix_text` reduced to [`fix_chars`] (no mojibake repair,
/// no Unicode normalization, no control-character removal), HTML entities unescaped twice, runs
/// of whitespace collapsed, trimmed.
pub fn clean_prompt(s: &str) -> String {
    let s = unescape(&unescape(&fix_chars(&unescape(s))));
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The text encoder's rows for `prompt`'s tokens (`[n, 4096]`: the positions after them, up to
/// the transformer's 512, are padding the transformer zeroes).
pub fn encode_text(enc: &T5Encoder, prompt: &str, max_len: usize) -> Result<Vec<f32>> {
    let ids = enc.tokenize(&clean_prompt(prompt), max_len)?;
    let n = ids.iter().rposition(|&i| i != enc.pad_id).map_or(0, |p| p + 1);
    enc.forward(&ids[..n])
}

/// `image` resized to cover `width` x `height` (Lanczos, as diffusers' `VaeImageProcessor` with
/// `resize_mode="crop"`) and center-cropped, as one frame in `[-1, 1]`, channels-last.
pub fn preprocess_image(image: &Image, width: usize, height: usize) -> Vec<f32> {
    let (iw, ih) = (image.width, image.height);
    let ratio = width as f64 / height as f64;
    let src_ratio = iw as f64 / ih as f64;
    let sw = if ratio > src_ratio { width } else { iw * height / ih };
    let sh = if ratio <= src_ratio { height } else { ih * width / iw };
    let resized = if (sw, sh) == (iw, ih) {
        image.rgb.clone()
    } else {
        crate::resample::resize_lanczos_rgb8(&image.rgb, iw, ih, sw, sh)
    };
    // PIL pastes at (width / 2 - sw / 2, height / 2 - sh / 2): zero or negative offsets.
    let (ox, oy) = ((sw / 2).saturating_sub(width / 2), (sh / 2).saturating_sub(height / 2));
    let mut out = vec![0f32; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let (sx, sy) = ((x + ox).min(sw - 1), (y + oy).min(sh - 1));
            for c in 0..3 {
                out[(y * width + x) * 3 + c] = resized[(sy * sw + sx) * 3 + c] as f32 / 255.0 * 2.0 - 1.0;
            }
        }
    }
    out
}

/// A decoded frame as RGB8: `floor(clamp(x / 2 + 0.5, 0, 1) * 255)`, as the demos convert it.
pub fn frame_u8(f: &[f32]) -> Vec<u8> {
    f.iter().map(|&v| ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0) as u8).collect()
}

fn frame_f32(f: &[u8]) -> Vec<f32> {
    f.iter().map(|&v| v as f32 / 255.0 * 2.0 - 1.0).collect()
}

/// The speakers' masks and the background's (`[3, h, w]` of 0 / 1) from their boxes in an image
/// of `iw` x `ih`, resized (bilinear, cover) and center-cropped to `width` x `height`.
pub fn speaker_masks(regions: Option<[[usize; 4]; 2]>, iw: usize, ih: usize, width: usize, height: usize) -> Vec<f32> {
    let boxes = regions.unwrap_or_else(|| {
        let s = 0.1;
        let half = iw / 2;
        let (y0, y1) = ((ih as f64 * s) as usize, (ih as f64 * (1.0 - s)) as usize);
        let (x0, x1) = ((half as f64 * s) as usize, (half as f64 * (1.0 - s)) as usize);
        [[y0, x0, y1, x1], [y0, x0 + half, y1, x1 + half]]
    });
    let mut m = vec![0f32; 3 * ih * iw];
    for (c, b) in boxes.iter().enumerate() {
        for y in b[0].min(ih)..b[2].min(ih) {
            for x in b[1].min(iw)..b[3].min(iw) {
                m[(c * ih + y) * iw + x] = 1.0;
            }
        }
    }
    for p in 0..ih * iw {
        m[2 * ih * iw + p] = if m[p] + m[ih * iw + p] > 0.0 { 0.0 } else { 1.0 };
    }
    // `_resize_and_centercrop_tensor`.
    let ratio = width as f64 / height as f64;
    let src_ratio = iw as f64 / ih as f64;
    let (nw, nh) = if ratio > src_ratio {
        (width, ih * width / iw)
    } else {
        (iw * height / ih, height)
    };
    let (top, left) = ((nh - height) / 2, (nw - width) / 2);
    let mut out = vec![0f32; 3 * height * width];
    for c in 0..3 {
        // Channels-last with one channel.
        let r = crate::omnigen2::resize_bilinear(&m[c * ih * iw..(c + 1) * ih * iw], iw, ih, 1, nw, nh);
        for y in 0..height {
            out[(c * height + y) * width..(c * height + y + 1) * width]
                .copy_from_slice(&r[(y + top) * nw + left..(y + top) * nw + left + width]);
        }
    }
    out
}

pub struct AvatarPipeline {
    pub dit: Dit,
    pub vae: VideoVae,
    /// DMD (`loraA`) and OPSD (`loraB`), when the release has them.
    pub loras: Option<[LoraNet; 2]>,
    pub shift: f32,
    pub src: Sources,
    pub load_secs: f64,
}

/// What one clip is conditioned on besides the text.
struct ClipIn<'a> {
    /// Normalized latents of the given frames (`[16, n, h, w]`) and how many latent frames.
    cond: Option<(&'a [f32], usize)>,
    /// The reference latent frame of a continuation.
    reference: Option<&'a [f32]>,
    audio: &'a [f32],
    humans: usize,
    masks: Option<(&'a [f32], usize, usize, usize)>,
    noise: Vec<f32>,
}

impl AvatarPipeline {
    /// Loads the transformer, the LoRAs and the VAE (the text and audio encoders are read when a
    /// request needs them and dropped afterwards).
    pub fn load(src: &Sources) -> Result<Self> {
        let t0 = Instant::now();
        let dir = src.avatar.join(if src.int8 { "base_model_int8" } else { "base_model" });
        if !dir.join("config.json").exists() {
            bail!(
                "{}: no {} (the release has base_model/ in bf16 and base_model_int8/)",
                src.avatar.display(),
                dir.file_name().unwrap().to_string_lossy()
            );
        }
        let dit = Dit::load(&dir)?;
        let (a, b) = (src.avatar.join("loraA/lora.safetensors"), src.avatar.join("loraB/lora.safetensors"));
        let loras = if a.exists() && b.exists() {
            Some([LoraNet::load(&a)?, LoraNet::load(&b)?])
        } else {
            None
        };
        let vae = VideoVae::load(&src.base.join("vae"), true)?;
        if vae.cfg.z_dim != dit.cfg.in_channels {
            bail!(
                "the VAE has {} latent channels, the transformer {}",
                vae.cfg.z_dim,
                dit.cfg.in_channels
            );
        }
        let shift = crate::fsx::read(&src.avatar.join("scheduler/scheduler_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| v["shift"].as_f64())
            .unwrap_or(7.0) as f32;
        Ok(Self {
            dit,
            vae,
            loras,
            shift,
            src: src.clone(),
            load_secs: t0.elapsed().as_secs_f64(),
        })
    }

    /// The text encoder (umT5-XXL of the base repository).
    pub fn text_encoder(&self) -> Result<T5Encoder> {
        T5Encoder::load(
            &self.src.base.join("text_encoder"),
            &self.src.base.join("tokenizer"),
            self.src.text_quant,
        )
        .with_context(|| format!("the text encoder of {}", self.src.base.display()))
    }

    /// The audio encoder (`whisper-large-v3/` of the release).
    pub fn audio_encoder(&self) -> Result<AudioEncoder> {
        AudioEncoder::load(&self.src.avatar.join("whisper-large-v3"))
    }

    /// The latent grid of a `width` x `height` clip of `frames` frames: (frames, height, width).
    pub fn latent_shape(&self, width: usize, height: usize, frames: usize) -> (usize, usize, usize) {
        (
            (frames - 1) / self.vae.scale_t() + 1,
            height / self.vae.scale(),
            width / self.vae.scale(),
        )
    }

    /// The LoRA multipliers of step `i` (DMD, OPSD).
    fn mix(p: &AvatarParams, i: usize) -> (f32, f32) {
        if i < p.mix_steps {
            (p.mix_alpha, 1.0 - p.mix_alpha)
        } else {
            (0.0, 1.0)
        }
    }

    /// Denoises one clip; returns its normalized latents `[16, t, h, w]` (without the reference
    /// frame).
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        p: &AvatarParams,
        text: &[f32],
        neg: Option<&[f32]>,
        shape: (usize, usize, usize),
        input: ClipIn,
        segment: usize,
        progress: &mut dyn FnMut(Stage),
    ) -> Result<Vec<f32>> {
        let (t, lh, lw) = shape;
        let c = self.dit.cfg.in_channels;
        let frame = lh * lw;
        // [reference] ++ clip, channel-first.
        let n_ref = usize::from(input.reference.is_some());
        let tt = t + n_ref;
        let mut lat = vec![0f32; c * tt * frame];
        let n_cond = input.cond.map_or(0, |c| c.1);
        for ch in 0..c {
            let dst = &mut lat[ch * tt * frame..(ch + 1) * tt * frame];
            if let Some(r) = input.reference {
                dst[..frame].copy_from_slice(&r[ch * frame..(ch + 1) * frame]);
            }
            dst[n_ref * frame..].copy_from_slice(&input.noise[ch * t * frame..(ch + 1) * t * frame]);
            if let Some((cl, n)) = input.cond {
                dst[n_ref * frame..(n_ref + n) * frame].copy_from_slice(&cl[ch * n * frame..(ch + 1) * n * frame]);
            }
        }
        let given = n_ref + n_cond;
        let sig = sigmas(p.steps, p.distill, self.shift);
        let guided = p.text_guidance > 1.0 || p.audio_guidance > 1.0;
        let zero_audio = if guided { vec![0f32; input.audio.len()] } else { Vec::new() };
        let max_len = self.dit.cfg.model_max_length;
        for i in 0..p.steps {
            let t0 = Instant::now();
            let nets: Vec<(&LoraNet, f32)> = match (&self.loras, p.distill) {
                (Some(l), true) => {
                    let (a, b) = Self::mix(p, i);
                    vec![(&l[0], a), (&l[1], b)]
                }
                _ => Vec::new(),
            };
            let ts: Vec<f32> = (0..tt).map(|f| if f < given { 0.0 } else { sig[i] * 1000.0 }).collect();
            let cond = |text: &[f32], audio: &[f32]| -> Result<Vec<f32>> {
                self.dit.forward(
                    &lat,
                    (tt, lh, lw),
                    &ts,
                    &Cond {
                        text,
                        text_len: max_len,
                        audio,
                        humans: input.humans,
                        num_cond: given,
                        num_ref: n_ref,
                        ref_img_index: p.ref_img_index,
                        mask_frame_range: p.mask_frame_range,
                        ref_masks: input.masks,
                        lora: Active { nets: &nets },
                    },
                )
            };
            let mut v = cond(text, input.audio)?;
            if guided {
                let neg = neg.context("guidance needs the negative prompt's rows")?;
                let ut = cond(neg, input.audio)?;
                let un = cond(neg, &zero_audio)?;
                for ((a, &b), &u) in v.iter_mut().zip(&ut).zip(&un) {
                    *a = u + p.text_guidance * (*a - b) + p.audio_guidance * (b - u);
                }
            }
            // Euler on the noisy frames: x += (sigma - sigma_next) v.
            let dt = sig[i] - sig[i + 1];
            for ch in 0..c {
                let r = ch * tt * frame + given * frame..(ch + 1) * tt * frame;
                for (x, &vv) in lat[r.clone()].iter_mut().zip(&v[r]) {
                    *x += dt * vv;
                }
            }
            progress(Stage::Step {
                segment,
                segments: p.segments,
                step: i + 1,
                steps: p.steps,
                secs: t0.elapsed().as_secs_f64(),
            });
        }
        // Drop the reference frame.
        let mut out = vec![0f32; c * t * frame];
        for ch in 0..c {
            out[ch * t * frame..(ch + 1) * t * frame].copy_from_slice(&lat[(ch * tt + n_ref) * frame..(ch + 1) * tt * frame]);
        }
        Ok(out)
    }

    /// The first clip from its parts (text rows, audio rows `[humans, frames, 5, 5, dim]`, the
    /// reference frame in `[-1, 1]` if any, the noise `[16, t, h, w]`): normalized latents.
    #[allow(clippy::too_many_arguments)]
    pub fn first_clip(
        &self,
        p: &AvatarParams,
        text: &[f32],
        neg: Option<&[f32]>,
        size: (usize, usize),
        image: Option<&[f32]>,
        audio: &[f32],
        masks: Option<(&[f32], usize, usize, usize)>,
        noise: Vec<f32>,
        progress: &mut dyn FnMut(Stage),
    ) -> Result<Vec<f32>> {
        let (width, height) = size;
        let shape = self.latent_shape(width, height, p.frames);
        let cond = match image {
            Some(img) => {
                let mut l = self.vae.encode(&[img.to_vec()], height, width)?;
                self.vae.normalize(&mut l);
                Some(l)
            }
            None => None,
        };
        let humans = p.speech.len().max(1);
        self.denoise(
            p,
            text,
            neg,
            shape,
            ClipIn {
                cond: cond.as_deref().map(|c| (c, 1)),
                reference: None,
                audio,
                humans,
                masks,
                noise,
            },
            1,
            progress,
        )
    }

    /// A continuation: `prev_frames` are the previous clip's frames (RGB8), `reference` the first
    /// clip's first latent frame (normalized, `[16, h, w]`).
    #[allow(clippy::too_many_arguments)]
    pub fn next_clip(
        &self,
        p: &AvatarParams,
        text: &[f32],
        neg: Option<&[f32]>,
        size: (usize, usize),
        prev_frames: &[Vec<u8>],
        reference: &[f32],
        audio: &[f32],
        masks: Option<(&[f32], usize, usize, usize)>,
        noise: Vec<f32>,
        segment: usize,
        progress: &mut dyn FnMut(Stage),
    ) -> Result<Vec<f32>> {
        let (width, height) = size;
        let shape = self.latent_shape(width, height, p.frames);
        if prev_frames.len() < COND_FRAMES {
            bail!("a continuation needs {COND_FRAMES} frames of the previous clip");
        }
        let tail: Vec<Vec<f32>> = prev_frames[prev_frames.len() - COND_FRAMES..]
            .iter()
            .map(|f| frame_f32(f))
            .collect();
        let mut cl = self.vae.encode(&tail, height, width)?;
        self.vae.normalize(&mut cl);
        let n = (COND_FRAMES - 1) / self.vae.scale_t() + 1;
        let mut lat = self.denoise(
            p,
            text,
            neg,
            shape,
            ClipIn {
                cond: Some((&cl, n)),
                reference: Some(reference),
                audio,
                humans: p.speech.len().max(1),
                masks,
                noise,
            },
            segment,
            progress,
        )?;
        // The conditioning frames are written back as encoded.
        let (t, lh, lw) = shape;
        let frame = lh * lw;
        for ch in 0..self.dit.cfg.in_channels {
            lat[ch * t * frame..ch * t * frame + n * frame].copy_from_slice(&cl[ch * n * frame..(ch + 1) * n * frame]);
        }
        Ok(lat)
    }

    /// Decodes normalized latents to RGB8 frames.
    pub fn decode(
        &self,
        lat: &[f32],
        shape: (usize, usize, usize),
        segment: usize,
        progress: &mut dyn FnMut(Stage),
    ) -> Result<Vec<Vec<u8>>> {
        let (t, lh, lw) = shape;
        let mut z = lat.to_vec();
        self.vae.denormalize(&mut z);
        let frames = self
            .vae
            .decode(&z, t, lh, lw, &mut |done, total| progress(Stage::Decode { segment, done, total }))?;
        Ok(frames.iter().map(|f| frame_u8(f)).collect())
    }

    /// The canvas of a request: (width, height).
    pub fn canvas(&self, p: &AvatarParams) -> Result<(usize, usize)> {
        let (w, h) = match (&p.image, p.size) {
            (Some(img), _) => {
                let (h, w) = bucket(img.height, img.width, p.hd);
                (w, h)
            }
            (None, Some(s)) => s,
            (None, None) => {
                if p.hd {
                    (1280, 768)
                } else {
                    (832, 480)
                }
            }
        };
        let m = self.vae.scale() * self.dit.cfg.patch_size[1].max(self.dit.cfg.patch_size[2]);
        if w == 0 || h == 0 || !w.is_multiple_of(m) || !h.is_multiple_of(m) {
            bail!("the size must be multiples of {m} (got {w}x{h})");
        }
        Ok((w, h))
    }

    /// Runs a request: text and audio encoders, every clip, the VAE.
    pub fn generate(&self, p: &AvatarParams, progress: &mut dyn FnMut(Stage)) -> Result<Clip> {
        let st = self.vae.scale_t();
        if p.frames == 0 || !(p.frames - 1).is_multiple_of(st) {
            bail!("frames must be {st} n + 1 (got {})", p.frames);
        }
        if p.segments > 1 && p.frames <= COND_FRAMES {
            bail!("a continuation needs clips of more than {COND_FRAMES} frames");
        }
        if p.steps == 0 || (p.distill && p.steps != 8) {
            bail!("the distilled schedule has 8 steps (got {})", p.steps);
        }
        if p.distill && self.loras.is_none() {
            bail!("{}: no loraA/ and loraB/ for the distilled schedule", self.src.avatar.display());
        }
        let humans = p.speech.len();
        if humans == 0 || humans > 2 || p.speech.iter().any(|s| s.is_empty()) {
            bail!("one or two speech tracks are needed (got {humans})");
        }
        if humans == 2 && (p.image.is_none() || p.speech[0].len() != p.speech[1].len()) {
            bail!("two speakers need a reference image and tracks of the same length");
        }
        let (width, height) = self.canvas(p)?;
        let shape = self.latent_shape(width, height, p.frames);
        let guided = p.text_guidance > 1.0 || p.audio_guidance > 1.0;

        // Text.
        progress(Stage::Text);
        let (text, neg) = {
            let enc = self.text_encoder()?;
            let max_len = self.dit.cfg.model_max_length;
            let text = encode_text(&enc, &p.prompt, max_len)?;
            let neg = if guided {
                Some(encode_text(&enc, &p.negative_prompt, max_len)?)
            } else {
                None
            };
            (text, neg)
        };

        // Audio: the speech cut (or zero-padded) to the video's length plus a second.
        let total_frames = p.frames + (p.segments.max(1) - 1) * (p.frames - COND_FRAMES);
        let samples = ((total_frames as f64 / FPS as f64 + 1.0) * SPEECH_RATE as f64).ceil() as usize;
        let (embs, rows) = {
            let enc = self.audio_encoder()?;
            if enc.dim() != self.dit.cfg.audio_channel {
                bail!(
                    "the audio encoder is {} wide, the transformer reads {}",
                    enc.dim(),
                    self.dit.cfg.audio_channel
                );
            }
            let mut embs = Vec::with_capacity(humans);
            let mut rows = 0;
            for (i, s) in p.speech.iter().enumerate() {
                let mut s = s.clone();
                s.resize(samples, 0.0);
                let (e, r) = enc.embed(&s, FPS, &mut |done, total| {
                    progress(Stage::Audio {
                        done: i * total + done,
                        total: humans * total,
                    })
                })?;
                rows = r;
                embs.push(e);
            }
            (embs, rows)
        };
        let window = self.dit.cfg.audio_window;
        let clip_audio =
            |start: usize| -> Vec<f32> { embs.iter().flat_map(|e| clip_window(e, rows, start, p.frames, 1, window)).collect() };

        // The reference image and the speakers' regions.
        progress(Stage::Encode);
        let image = p.image.as_ref().map(|img| preprocess_image(img, width, height));
        let masks = match (&p.image, humans) {
            (Some(img), 2) => Some(speaker_masks(p.regions, img.width, img.height, width, height)),
            _ => None,
        };
        let masks = masks.as_deref().map(|m| (m, 3, height, width));

        let mut rng = crate::torch_rng::TorchGenerator::new(p.seed);
        let n_noise = self.dit.cfg.in_channels * shape.0 * shape.1 * shape.2;
        let mut lat = self.first_clip(
            p,
            &text,
            neg.as_deref(),
            (width, height),
            image.as_deref(),
            &clip_audio(0),
            masks,
            rng.randn(n_noise),
            progress,
        )?;
        let frame = shape.1 * shape.2;
        let reference: Vec<f32> = (0..self.dit.cfg.in_channels)
            .flat_map(|ch| lat[ch * shape.0 * frame..ch * shape.0 * frame + frame].to_vec())
            .collect();
        self.dit.release_device();
        let mut frames = self.decode(&lat, shape, 1, progress)?;
        let mut last = frames.clone();
        for seg in 1..p.segments.max(1) {
            let start = seg * (p.frames - COND_FRAMES);
            lat = self.next_clip(
                p,
                &text,
                neg.as_deref(),
                (width, height),
                &last,
                &reference,
                &clip_audio(start),
                masks,
                rng.randn(n_noise),
                seg + 1,
                progress,
            )?;
            self.dit.release_device();
            last = self.decode(&lat, shape, seg + 1, progress)?;
            frames.extend(last[COND_FRAMES..].iter().cloned());
        }
        Ok(Clip {
            width,
            height,
            frames,
            latents: lat,
        })
    }
}

/// Whether `dir` is a LongCat-Video-Avatar release (`base_model*/config.json` naming the avatar
/// transformer).
pub fn is_avatar(dir: &Path) -> bool {
    ["base_model_int8", "base_model"].iter().any(|d| {
        crate::fsx::read(&dir.join(d).join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .is_some_and(|v| v["_class_name"].as_str() == Some("LongCatVideoAvatarTransformer3DModel"))
    })
}
