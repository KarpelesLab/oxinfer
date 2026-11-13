//! Qwen-Image-2.1 and Qwen-Image-2.1-Turbo, following diffusers' `QwenImage21Pipeline`:
//!
//! - the prompt in the pipeline's own template (a system turn, "Comprehend and analyze the
//!   provided prompt.", then the user turn) through the Qwen3-VL-8B text encoder; the
//!   conditioning is the last layer's output *before* the final norm, minus the system turn;
//! - an edit's input images, each resized to about 1024x1024, twice: through the text encoder's
//!   vision tower (with its DeepStack features) as `<imageN><|vision_start|>...<|vision_end|>`
//!   before the prompt, and through the VAE (opaque RGBA) into latent tokens that take the
//!   places of the image's placeholders in the transformer's sequence;
//! - the single-stream block-causal transformer (`transformer21`), the prompt's keys and values
//!   computed once and reused by every step;
//! - flow matching with Euler steps: `linspace(1, 1/steps, steps)` through the scheduler's
//!   resolution-dependent shift (Qwen-Image-2.1), or the eight sigmas a distilled checkpoint
//!   saves in `model_index.json` (`sample_sigmas`, Qwen-Image-2.1-Turbo) as they are; no guidance
//!   by default, and plain classifier-free guidance (no norm rescale) with a negative prompt;
//! - the residual RGBA VAE (`vae`, 64 latent channels, one token per 16x16 pixels).

use super::transformer21::{Config21, Prefix, Transformer21};
use super::*;

pub(super) struct LoadedTransformer21 {
    // Drop order: the transformer (borrowing `_files`), then the files.
    pub t: Transformer21<'static>,
    pub _files: Box<ModelFiles>,
}

/// The system turn of the pipeline's prompt template.
const SYSTEM: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";

/// `model_index.json`'s `sample_sigmas` (the schedule of a distilled checkpoint), if any.
pub(super) fn sample_sigmas(dir: &Path) -> Option<Vec<f32>> {
    let v: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("model_index.json")).ok()?).ok()?;
    let s: Vec<f32> = v["sample_sigmas"]
        .as_array()?
        .iter()
        .filter_map(|x| x.as_f64())
        .map(|x| x as f32)
        .collect();
    (!s.is_empty()).then_some(s)
}

/// The settings a Qwen-Image-2.1 transformer config stands for in the pipeline's shared fields.
pub(super) fn shared_config(c: &Config21) -> TransformerConfig {
    TransformerConfig {
        patch_size: c.patch_size,
        in_channels: c.in_channels,
        out_channels: Some(c.out()),
        num_layers: c.num_layers,
        attention_head_dim: c.attention_head_dim,
        num_attention_heads: c.num_attention_heads,
        joint_attention_dim: c.context_in_dim,
        guidance_embeds: false,
        axes_dims_rope: c.axes_dims_rope,
        zero_cond_t: false,
        use_additional_t_cond: false,
        use_layer3d_rope: false,
        extra_condition_channels: 0,
    }
}

/// One prompt's conditioning: the hidden states after the system turn (`[slots.len(),
/// context_in_dim]`) and which of them are image placeholders.
struct Cond {
    hidden: Vec<f32>,
    slots: Vec<bool>,
}

impl Pipeline {
    /// [`Params`] for Qwen-Image-2.1: no guidance, the checkpoint's saved schedule (Turbo: 8
    /// steps) or 40 steps.
    pub(super) fn params21(&self, prompt: &str) -> Params {
        Params {
            negative: None,
            true_cfg: 1.0,
            steps: self.sample_sigmas.as_ref().map_or(40, |s| s.len()),
            ..Params::new(prompt)
        }
    }

    /// The text encoder for Qwen-Image-2.1 (if not loaded): the language model, and the vision
    /// tower when there are condition images.
    fn load_text_encoder21(&mut self, vision: bool) -> Result<f64> {
        if self.text.as_ref().is_some_and(|t| t.has_vision() || !vision) {
            return Ok(0.0);
        }
        self.text = None;
        let t0 = Instant::now();
        let pdir = self.dir.join("processor");
        let has = |f: &str| crate::fsx::exists(&pdir.join(f));
        let vo = QwenVlOptions {
            quant: self.opts.text_quant,
            mmap: self.opts.mmap,
            tokenizer_dir: has("tokenizer.json").then(|| pdir.clone()),
            processor_dir: (has("processor_config.json") || has("preprocessor_config.json")).then(|| pdir.clone()),
            vision,
        };
        let vl = QwenVl::load(&self.dir.join("text_encoder"), &vo)?;
        if vl.hidden_size() != self.tcfg.joint_attention_dim {
            bail!(
                "text encoder width {} does not match the transformer's {}",
                vl.hidden_size(),
                self.tcfg.joint_attention_dim
            );
        }
        self.text = Some(vl);
        Ok(t0.elapsed().as_secs_f64())
    }

    /// The conditioning of `prompt` with the condition images `cond` (`<imageN>` and the image's
    /// placeholders before the prompt, as the pipeline's template writes them).
    fn encode21(&self, prompt: &str, cond: &[VlImage]) -> Result<Cond> {
        let vl = self.text.as_ref().context("text encoder not loaded")?;
        // Qwen has no BOS token: an empty prompt would leave the user turn empty.
        let prompt = if prompt.is_empty() { " " } else { prompt };
        let drop = vl.tokenize(SYSTEM)?.len();
        let pics: Vec<String> = cond
            .iter()
            .enumerate()
            .map(|(i, c)| format!("<image{}>{}", i + 1, c.placeholder()))
            .collect();
        let text = format!(
            "{SYSTEM}<|im_start|>user\n{}{prompt}<|im_end|>\n<|im_start|>assistant\n",
            pics.join(" ")
        );
        let last = vl.model().cfg.num_hidden_layers - 1;
        let mut h = vl.hidden_states(&text, cond, &[last])?;
        let d = h.hidden;
        let n = h.ids.len();
        if n <= drop {
            bail!("the prompt template is shorter than the {drop} tokens it drops");
        }
        let (_, x) = h.layers.pop().context("the text encoder's last layer was not captured")?;
        let pad = vl.mm.as_ref().map(|m| m.image_token_id);
        Ok(Cond {
            hidden: x[drop * d..].to_vec(),
            slots: h.ids[drop..].iter().map(|&i| Some(i) == pad).collect(),
        })
    }

    /// Loads a Qwen-Image-2.1 transformer from `fs` (tensor names under `prefix`).
    pub(super) fn load_transformer21(&mut self, files: Box<ModelFiles>, prefix: String) -> Result<()> {
        if self.opts.stream.is_some() {
            bail!("Qwen-Image-2.1: streaming the transformer from storage is not supported (use --mmap)");
        }
        let cfg = self.cfg21.clone().context("no Qwen-Image-2.1 transformer config")?;
        // SAFETY: the box's heap block does not move; `LoadedTransformer21` drops the transformer
        // (every borrow of it) before `_files`.
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let lora = if self.opts.loras.is_empty() {
            None
        } else {
            let names: Vec<String> = (0..cfg.num_layers)
                .flat_map(|l| transformer21::BLOCK_LINEARS.map(|m| format!("{prefix}transformer_blocks.{l}.{m}.weight")))
                .chain(transformer21::TOP_LINEARS.map(|m| format!("{prefix}{m}.weight")))
                .collect();
            Some(weights::LoraSet::open(&self.opts.loras, fs, &names, &prefix)?)
        };
        let ld = weights::Loader {
            files: fs,
            quant: Quant::stored(fs).unwrap_or(self.opts.quant),
            lora: lora.as_ref(),
            prefix,
        };
        let t = Transformer21::load(cfg, &ld)?;
        self.lora_report.clear();
        if let Some(l) = &lora {
            for (f, n) in &l.files {
                self.lora_report.push(format!("[lora {f}: {n} modules]"));
            }
            if l.applied() != l.modules() {
                bail!(
                    "LoRA: {} of {} modules matched no loaded weight",
                    l.modules() - l.applied(),
                    l.modules()
                );
            }
        }
        self.tr21 = Some(LoadedTransformer21 { t, _files: files });
        Ok(())
    }

    /// The Qwen-Image-2.1 transformer, once loaded.
    pub fn transformer21(&self) -> Option<&Transformer21<'static>> {
        self.tr21.as_ref().map(|t| &t.t)
    }

    /// Text to image with Qwen-Image-2.1, or with `images` an edit: the images condition the
    /// prompt (through the text encoder's vision tower) and the transformer (their latents, in
    /// the sequence where the prompt has their placeholders).
    pub(super) fn run21(&mut self, images: &[RgbImage], p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        self.cancel.store(false, std::sync::atomic::Ordering::Relaxed);
        self.alpha = None;
        if p.mask.is_some() || p.strength.is_some() || !p.control.is_empty() {
            bail!("Qwen-Image-2.1: --strength, --mask and ControlNets are not supported");
        }
        for img in images {
            if img.data.len() != img.width * img.height * 3 || img.data.is_empty() {
                bail!("input image: {} bytes for {}x{} RGB", img.data.len(), img.width, img.height);
            }
        }
        let mut tm = Timings::default();
        let sf = self.vae.scale();
        let mult = sf * 2;
        // Condition images at about 1024x1024 (one resize for the text encoder and the VAE); the
        // output takes the last one's aspect ratio.
        let cond_rgb: Vec<Rgb> = images
            .iter()
            .map(|i| resized(i, calculate_dimensions(1024.0 * 1024.0, i.width as f64 / i.height as f64)))
            .collect();
        let (cw, ch) = cond_rgb.last().map_or((1024, 1024), |c| (c.1, c.2));
        let width = p.width.unwrap_or(cw) / mult * mult;
        let height = p.height.unwrap_or(ch) / mult * mult;
        if width < mult || height < mult {
            bail!("output size {width}x{height} is too small");
        }
        let guided = p.true_cfg > 1.0 && p.negative.is_some();

        // Conditioning (exact GPU matmuls, as for the other Qwen-Image text encoders).
        tm.load_text = self.load_text_encoder21(!images.is_empty())?;
        let t0 = Instant::now();
        let was_exact = crate::gpu::exact();
        crate::gpu::set_exact(true);
        let enc = (|| -> Result<(Cond, Option<Cond>)> {
            let vl = self.text.as_ref().context("text encoder not loaded")?;
            let cond: Vec<VlImage> = cond_rgb
                .iter()
                .map(|(rgb, w, h)| {
                    let c = vl.encode_image(rgb, *w, *h, 0)?;
                    // One placeholder per 2x2 latent tokens: the processor must keep the size.
                    if c.size != (*w, *h) || c.n * 4 != (w / sf) * (h / sf) {
                        bail!(
                            "condition image {w}x{h}: the image processor made it {}x{} ({} tokens)",
                            c.size.0,
                            c.size.1,
                            c.n
                        );
                    }
                    Ok(c)
                })
                .collect::<Result<_>>()?;
            let pos = self.encode21(&p.prompt, &cond)?;
            let neg = match (&p.negative, guided) {
                (Some(n), true) => Some(self.encode21(n, &cond)?),
                _ => None,
            };
            Ok((pos, neg))
        })();
        crate::gpu::set_exact(was_exact);
        let (mut pos, mut neg) = enc?;
        // Validation: the reference's conditioning instead of ours.
        if let Some(d) = dump_dir() {
            for (name, dst) in [("prompt_embeds_in.bin", Some(&mut pos)), ("negative_embeds_in.bin", neg.as_mut())] {
                if let (Some(dst), true) = (dst, d.join(name).exists()) {
                    dst.hidden = read_f32(&d.join(name))?;
                    dst.slots.resize(dst.hidden.len() / self.tcfg.joint_attention_dim, false);
                }
            }
        }
        dump("prompt_embeds.bin", &pos.hidden);
        if let Some(n) = &neg {
            dump("negative_embeds.bin", &n.hidden);
        }
        tm.text = t0.elapsed().as_secs_f64();
        if !self.opts.keep_text_encoder {
            self.unload_text_encoder();
        }
        self.check_cancel()?;

        // The condition images' latents (the posterior's mode): opaque RGBA through the VAE.
        let t1 = Instant::now();
        let z = self.vae.cfg.z_dim;
        let mut cond_tokens = Vec::new();
        let mut cond_shapes = Vec::new();
        for (rgb, w, h) in &cond_rgb {
            let px: Vec<f32> = match self.vae.image_channels() {
                4 => to_unit(rgb)
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .flat_map(|p| [p[0], p[1], p[2], 1.0])
                    .collect(),
                _ => to_unit(rgb),
            };
            let (lat, rh, rw) = self.vae.encode(&px, *h, *w)?;
            cond_tokens.extend_from_slice(&lat);
            cond_shapes.push(ImageShape {
                frames: 1,
                rows: rh,
                cols: rw,
            });
        }
        if !cond_tokens.is_empty() {
            dump("cond_latents.bin", &cond_tokens);
        }
        tm.vae_encode = t1.elapsed().as_secs_f64();

        tm.load_transformer = self.load_transformer()?;
        let t2 = Instant::now();
        let tr = &self.tr21.as_ref().context("transformer not loaded")?.t;
        let (lh, lw) = (height / sf, width / sf);
        let target = ImageShape {
            frames: 1,
            rows: lh,
            cols: lw,
        };
        let n0 = target.tokens();
        let mut rng = crate::sd::ops::Rng::new(p.seed);
        // Initial noise, generated as [c, h, w] (or read from the dump folder); a token per pixel.
        let mut x = chw_to_hwc(&normals_or_dump("noise.bin", &mut rng, z * lh * lw)?, z);
        dump("x0.bin", &x);
        // The schedule: the checkpoint's own sigmas when the step count is theirs.
        let sched = p.scheduler.clone().unwrap_or_else(|| self.sched.clone());
        let base = match &self.sample_sigmas {
            Some(s) if s.len() == p.steps => s.clone(),
            _ => crate::flux::sched::flux_sigmas(p.steps),
        };
        let sigmas = sched.plan(&base, Some(sched.mu(n0)))?.sigmas;
        let steps = sigmas.len() - 1;
        let _attn = AttnGuard(crate::gpu::set_exact_attention(true));
        let mut pre: Prefix = tr.prefill(&pos.hidden, &pos.slots, &cond_tokens, &cond_shapes, target)?;
        let mut pre_neg = match &neg {
            Some(n) => Some(tr.prefill(&n.hidden, &n.slots, &cond_tokens, &cond_shapes, target)?),
            None => None,
        };
        drop((pos, neg, cond_tokens));
        for i in 0..steps {
            self.check_cancel()?;
            let (s, s_next) = (sigmas[i], sigmas[i + 1]);
            let mut v = tr.step(&mut pre, &x, s);
            tm.transformer_calls += 1;
            if let Some(pn) = &mut pre_neg {
                let vn = tr.step(pn, &x, s);
                tm.transformer_calls += 1;
                guide(&mut v, &vn, z, p.true_cfg, false);
            }
            if i == 0 {
                dump("v0.bin", &v);
            }
            let dt = s_next - s;
            for (a, &b) in x.iter_mut().zip(&v) {
                *a += dt * b;
            }
            progress(i + 1, steps);
        }
        dump("latents_out.bin", &x);
        tm.denoise = t2.elapsed().as_secs_f64();

        let t3 = Instant::now();
        let (px, h, w) = self.vae.decode(&x, lh, lw);
        dump("image.bin", &px);
        let c = self.vae.image_channels();
        let bytes = to_u8(&px);
        let (rgb, alpha) = match c {
            4 => (
                bytes.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect(),
                Some(bytes.as_chunks::<4>().0.iter().map(|p| p[3]).collect::<Vec<u8>>()),
            ),
            _ => (bytes, None),
        };
        self.alpha = alpha;
        tm.decode = t3.elapsed().as_secs_f64();
        Ok((Image { width: w, height: h, rgb }, tm))
    }
}
