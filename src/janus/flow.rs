//! JanusFlow's image generation (`deepseek-ai/JanusFlow-1.3B`, `janus/janusflow/` of the
//! reference): rectified flow in the latent space of the SDXL VAE, with the language model as
//! the velocity network.
//!
//! One Euler step at time t (0 = noise, 1 = image) of a 48x48x4 latent z:
//!
//! * the `ShallowUViTEncoder` embeds t (sinusoidal, two linears) and turns z into 24x24 tokens
//!   of 768 channels (a 2x2 patch convolution and two ConvNeXt blocks modulated by the time
//!   embedding), which `vision_gen_enc_aligner` projects to the language model's width;
//! * the language model reads the prompt, the time embedding (in place of the prompt's last
//!   token) and the 576 tokens, and its hidden states at the 576 tokens go through an RMSNorm
//!   and `vision_gen_dec_aligner` back to 768 channels;
//! * the `ShallowUViTDecoder` concatenates them with the encoder's tokens (the long skip
//!   connection), runs two more ConvNeXt blocks and unpatchifies to the velocity (1x1
//!   convolution and pixel shuffle).
//!
//! Classifier-free guidance, `v = w * v_cond - (w - 1) * v_uncond`, runs the conditional and
//! unconditional sequences as one batch; the unconditional one sees only BOS of the prompt (the
//! reference masks the rest). The prompt's keys and values are computed once and every step
//! appends its 577 positions to them, then drops them again.

use super::llm::BatchState;
use super::siglip::{Dense, gelu_inplace};
use super::{ImageStats, Janus, Progress};
use crate::clock::Instant;
use crate::generate::Flow;
use crate::kernels::{for_chunks, rmsnorm_rows};
use crate::safetensors::ModelFiles;
use crate::sd::ops::{Rng, silu_inplace, timestep_embedding};
use crate::sd::vae::VaeDecoder;
use anyhow::{Context, Result, bail};
use std::sync::atomic::Ordering;

/// The reference's defaults (`demo/app_janusflow.py`, the README's example).
pub const DEFAULT_STEPS: usize = 30;
pub const DEFAULT_CFG: f32 = 2.0;

/// The VAE the reference decodes with (`AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")`).
pub const DEFAULT_VAE: &str = "stabilityai/sdxl-vae";

const KERNEL: usize = 7;

/// `ConvNextBlock`: a 7x7 depthwise convolution, RMSNorm, an inverted bottleneck with global
/// response normalization, the residual, then a scale and shift from the time embedding.
struct ConvNext<'a> {
    c: usize,
    /// Depthwise kernel, tap-major: `[49, c]`.
    dw: Vec<f32>,
    dw_b: Vec<f32>,
    norm: &'a [u16],
    fc1: Dense<'a>,
    grn_w: Vec<f32>,
    grn_b: Vec<f32>,
    fc2: Dense<'a>,
    cond: Dense<'a>,
    eps: f32,
}

fn norm_weight<'a>(files: &'a ModelFiles, name: &str, n: usize) -> Result<&'a [u16]> {
    let w = files.bf16(name)?;
    if w.len() != n {
        bail!("{name}: {} values, expected {n}", w.len());
    }
    Ok(w)
}

fn vector(files: &ModelFiles, name: &str, n: usize) -> Result<Vec<f32>> {
    let v = files.f32(name)?;
    if v.len() != n {
        bail!("{name}: {} values, expected {n}", v.len());
    }
    Ok(v)
}

impl<'a> ConvNext<'a> {
    fn load(files: &'a ModelFiles, p: &str, c: usize, hidden: usize, eps: f32) -> Result<Self> {
        let taps = KERNEL * KERNEL;
        let w = vector(files, &format!("{p}.depthwise.weight"), c * taps)?;
        let mut dw = vec![0f32; c * taps];
        for ch in 0..c {
            for t in 0..taps {
                dw[t * c + ch] = w[ch * taps + t];
            }
        }
        Ok(Self {
            c,
            dw,
            dw_b: vector(files, &format!("{p}.depthwise.bias"), c)?,
            norm: norm_weight(files, &format!("{p}.norm.weight"), c)?,
            fc1: Dense::load(files, &format!("{p}.channelwise_linear_1"), 4 * c, c)?,
            grn_w: vector(files, &format!("{p}.channelwise_norm.weight"), 4 * c)?,
            grn_b: vector(files, &format!("{p}.channelwise_norm.bias"), 4 * c)?,
            fc2: Dense::load(files, &format!("{p}.channelwise_linear_2"), c, 4 * c)?,
            cond: Dense::load(files, &format!("{p}.cond_embeds_mapper"), 2 * c, hidden)?,
            eps,
        })
    }

    /// `x` is one image's `[h * w, c]`; `t_act` is `silu(t_emb)`.
    fn forward(&self, x: &mut [f32], h: usize, w: usize, t_act: &[f32]) {
        let (c, n) = (self.c, h * w);
        let pad = (KERNEL / 2) as isize;
        let mut y = vec![0f32; n * c];
        {
            let (x, dw, dw_b) = (&*x, &self.dw, &self.dw_b);
            for_chunks(&mut y, c, true, |p, o| {
                o.copy_from_slice(dw_b);
                let (py, px) = ((p / w) as isize, (p % w) as isize);
                for ky in 0..KERNEL {
                    let iy = py + ky as isize - pad;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..KERNEL {
                        let ix = px + kx as isize - pad;
                        if ix < 0 || ix >= w as isize {
                            continue;
                        }
                        let s = (iy as usize * w + ix as usize) * c;
                        let k = &dw[(ky * KERNEL + kx) * c..(ky * KERNEL + kx + 1) * c];
                        for ((acc, &k), &v) in o.iter_mut().zip(k).zip(&x[s..s + c]) {
                            *acc += k * v;
                        }
                    }
                }
            });
        }
        let mut yn = vec![0f32; n * c];
        rmsnorm_rows(&y, c, Some(self.norm), self.eps, &mut yn);
        let mut m = self.fc1.forward(&yn, n);
        gelu_inplace(&mut m);
        // GlobalResponseNorm: each channel's l2 norm over the image, relative to their mean.
        let c4 = 4 * c;
        let mut g = vec![0f64; c4];
        for r in m.chunks(c4) {
            for (g, &v) in g.iter_mut().zip(r) {
                *g += v as f64 * v as f64;
            }
        }
        g.iter_mut().for_each(|v| *v = v.sqrt());
        let mean = g.iter().sum::<f64>() / c4 as f64;
        let scale: Vec<f32> = g
            .iter()
            .zip(&self.grn_w)
            .map(|(&g, &w)| w * (g / (mean + 1e-6)) as f32 + 1.0)
            .collect();
        let grn_b = &self.grn_b;
        for_chunks(&mut m, c4, true, |_, r| {
            for ((v, &s), &b) in r.iter_mut().zip(&scale).zip(grn_b) {
                *v = b + s * *v;
            }
        });
        let o = self.fc2.forward(&m, n);
        let cs = self.cond.forward(t_act, 1);
        let (sc, sh) = cs.split_at(c);
        for (xr, or) in x.chunks_mut(c).zip(o.chunks(c)) {
            for j in 0..c {
                xr[j] = sh[j] + (xr[j] + or[j]) * (1.0 + sc[j]);
            }
        }
    }
}

fn blocks<'a>(files: &'a ModelFiles, p: &str, n: usize, c: usize, hidden: usize, eps: f32) -> Result<Vec<ConvNext<'a>>> {
    (0..n)
        .map(|i| ConvNext::load(files, &format!("{p}.mid_block.res_blocks.{i}"), c, hidden, eps))
        .collect()
}

fn param(v: &serde_json::Value, key: &str, default: usize) -> usize {
    v.get(key).and_then(|x| x.as_u64()).map_or(default, |x| x as usize)
}

/// `ShallowUViTEncoder` and `vision_gen_enc_aligner`.
struct Encoder<'a> {
    time1: Dense<'a>,
    time2: Dense<'a>,
    in_conv: Dense<'a>,
    blocks: Vec<ConvNext<'a>>,
    aligner: Dense<'a>,
    /// Latent channels (4), the patch size (2) and the tokens' channels (768).
    cin: usize,
    patch: usize,
    c: usize,
}

impl<'a> Encoder<'a> {
    fn load(files: &'a ModelFiles, cfg: &serde_json::Value, hidden: usize) -> Result<Self> {
        let p = "vision_gen_enc_model";
        let c = cfg["block_out_channels"][0].as_u64().unwrap_or(768) as usize;
        let cin = param(cfg, "input_channels", 4);
        let patch = param(cfg, "stride", 2);
        if param(cfg, "kernel_size", patch) != patch || param(cfg, "padding", 0) != 0 {
            bail!("vision_gen_enc_config: only a patch convolution (kernel = stride, no padding) is supported");
        }
        if cfg.get("use_mid_block").and_then(|v| v.as_bool()) == Some(false) {
            bail!("vision_gen_enc_config: use_mid_block false is not supported");
        }
        let eps = cfg["norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        Ok(Self {
            time1: Dense::load(files, &format!("{p}.time_embed.linear_1"), hidden, c)?,
            time2: Dense::load(files, &format!("{p}.time_embed.linear_2"), hidden, hidden)?,
            in_conv: Dense::load(files, &format!("{p}.in_conv"), c, cin * patch * patch)?,
            blocks: blocks(files, p, param(cfg, "layers_in_middle", 2), c, hidden, eps)?,
            aligner: Dense::load(files, "vision_gen_enc_aligner", hidden, c)?,
            cin,
            patch,
            c,
        })
    }

    /// The time embedding of timestep `t` (0..1000): `[hidden]`.
    fn time(&self, t: f32) -> Vec<f32> {
        let e = timestep_embedding(t, self.c, true, 0.0);
        let mut y = self.time1.forward(&e, 1);
        silu_inplace(&mut y);
        self.time2.forward(&y, 1)
    }

    /// A latent (`[lh * lw, cin]`) to its tokens (`[lh/2 * lw/2, c]`, the skip connection) and
    /// their language-model embeddings (`[.., hidden]`).
    fn encode(&self, z: &[f32], lh: usize, lw: usize, t_act: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (p, cin) = (self.patch, self.cin);
        let (gh, gw) = (lh / p, lw / p);
        // Patches as rows of (channel, y, x), the kernel's layout.
        let mut rows = vec![0f32; gh * gw * cin * p * p];
        for (i, r) in rows.chunks_mut(cin * p * p).enumerate() {
            let (gy, gx) = (i / gw, i % gw);
            for c in 0..cin {
                for y in 0..p {
                    for x in 0..p {
                        r[(c * p + y) * p + x] = z[((gy * p + y) * lw + gx * p + x) * cin + c];
                    }
                }
            }
        }
        let mut x = self.in_conv.forward(&rows, gh * gw);
        for b in &self.blocks {
            b.forward(&mut x, gh, gw, t_act);
        }
        let emb = self.aligner.forward(&x, gh * gw);
        (x, emb)
    }
}

/// `vision_gen_dec_aligner_norm`, `vision_gen_dec_aligner` and `ShallowUViTDecoder`.
struct Decoder<'a> {
    aligner_norm: &'a [u16],
    aligner_eps: f32,
    aligner: Dense<'a>,
    input_norm: &'a [u16],
    blocks: Vec<ConvNext<'a>>,
    out_norm: &'a [u16],
    unpatch: Dense<'a>,
    c: usize,
    skip: usize,
    cout: usize,
    eps: f32,
}

impl<'a> Decoder<'a> {
    fn load(files: &'a ModelFiles, cfg: &serde_json::Value, hidden: usize, skip: usize, llm_eps: f32) -> Result<Self> {
        let p = "vision_gen_dec_model";
        let c = param(cfg, "in_channels", 768);
        let cout = param(cfg, "out_channels", 4);
        if param(cfg, "upsamples", 1) != 1 {
            bail!("vision_gen_dec_config: only one upsampling stage is supported");
        }
        if cfg.get("use_mid_block").and_then(|v| v.as_bool()) == Some(false) {
            bail!("vision_gen_dec_config: use_mid_block false is not supported");
        }
        let eps = cfg["norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let wide = c + skip;
        Ok(Self {
            aligner_norm: norm_weight(files, "vision_gen_dec_aligner_norm.weight", hidden)?,
            aligner_eps: llm_eps,
            aligner: Dense::load(files, "vision_gen_dec_aligner", c, hidden)?,
            input_norm: norm_weight(files, &format!("{p}.input_norm.weight"), c)?,
            blocks: blocks(files, p, param(cfg, "layers_in_middle", 2), wide, hidden, eps)?,
            out_norm: norm_weight(files, &format!("{p}.out_convs.0.norm.weight"), wide)?,
            unpatch: Dense::load(files, &format!("{p}.out_convs.0.unpatch_conv"), cout * 4, wide)?,
            c,
            skip,
            cout,
            eps,
        })
    }

    /// The velocity (`[2gh * 2gw, cout]`) from the language model's final hidden states at the
    /// image tokens (`[gh * gw, hidden]`) and the encoder's tokens `hs`.
    fn velocity(&self, hidden: &[f32], hs: &[f32], gh: usize, gw: usize, t_act: &[f32]) -> Vec<f32> {
        let n = gh * gw;
        let h = self.aligner.n_in;
        let mut hn = vec![0f32; n * h];
        rmsnorm_rows(hidden, h, Some(self.aligner_norm), self.aligner_eps, &mut hn);
        let a = self.aligner.forward(&hn, n);
        let mut an = vec![0f32; n * self.c];
        rmsnorm_rows(&a, self.c, Some(self.input_norm), self.eps, &mut an);
        let wide = self.c + self.skip;
        let mut x = Vec::with_capacity(n * wide);
        for (a, s) in an.chunks(self.c).zip(hs.chunks(self.skip)) {
            x.extend_from_slice(a);
            x.extend_from_slice(s);
        }
        for b in &self.blocks {
            b.forward(&mut x, gh, gw, t_act);
        }
        let mut xn = vec![0f32; n * wide];
        rmsnorm_rows(&x, wide, Some(self.out_norm), self.eps, &mut xn);
        let u = self.unpatch.forward(&xn, n);
        // PixelShuffle(2): channel c * 4 + i * 2 + j of token (y, x) is channel c of pixel
        // (2y + i, 2x + j).
        let co = self.cout;
        let mut v = vec![0f32; 4 * n * co];
        for (p, r) in u.chunks(4 * co).enumerate() {
            let (y, x) = (p / gw, p % gw);
            for c in 0..co {
                for i in 0..2 {
                    for j in 0..2 {
                        v[((2 * y + i) * 2 * gw + 2 * x + j) * co + c] = r[c * 4 + i * 2 + j];
                    }
                }
            }
        }
        v
    }
}

/// JanusFlow's generation parts.
pub(crate) struct FlowParts<'a> {
    enc: Encoder<'a>,
    dec: Decoder<'a>,
    vae: VaeDecoder,
    /// The latent side the model was trained at (48: a 384x384 image).
    pub latent: usize,
}

impl<'a> FlowParts<'a> {
    pub fn load(files: &'a ModelFiles, cfg: &serde_json::Value, hidden: usize, llm_eps: f32, vae: VaeDecoder) -> Result<Self> {
        for (key, cls) in [
            ("vision_gen_enc_config", "ShallowUViTEncoder"),
            ("vision_gen_dec_config", "ShallowUViTDecoder"),
        ] {
            if cfg[key]["cls"] != cls {
                bail!("{key}: only {cls} is supported, got {}", cfg[key]["cls"]);
            }
        }
        let enc = Encoder::load(files, &cfg["vision_gen_enc_config"]["params"], hidden)?;
        let dec = Decoder::load(files, &cfg["vision_gen_dec_config"]["params"], hidden, enc.c, llm_eps)?;
        if dec.cout != enc.cin || vae.latent_channels != enc.cin {
            bail!(
                "JanusFlow: the encoder takes {} latent channels, the decoder gives {}, the VAE has {}",
                enc.cin,
                dec.cout,
                vae.latent_channels
            );
        }
        Ok(Self { enc, dec, vae, latent: 48 })
    }

    /// Values of one latent (`[latent * latent, channels]`).
    pub fn latent_len(&self) -> usize {
        self.latent * self.latent * self.enc.cin
    }

    /// The image's side in pixels.
    pub fn image_size(&self) -> usize {
        self.latent * 8
    }
}

const VAE_WEIGHTS: [&str; 3] = [
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.fp16.safetensors",
    "sdxl_vae.safetensors",
];

/// The folder of the VAE named by `spec`: a local folder, or a repo id whose `config.json` and
/// weights are fetched into the cache (with the `hf` feature) unless already there.
fn vae_dir(spec: &str, cache_dir: Option<&std::path::Path>) -> Result<std::path::PathBuf> {
    let p = std::path::Path::new(spec);
    if p.is_dir() {
        return Ok(p.to_path_buf());
    }
    #[cfg(feature = "hf")]
    {
        let root = cache_dir.map(std::path::Path::to_path_buf).unwrap_or_else(crate::hf::cache_root);
        let dir = root.join(spec.replace('/', "--"));
        if dir.join("config.json").is_file() && dir.join(VAE_WEIGHTS[0]).is_file() {
            return Ok(dir);
        }
        crate::hf::resolve_files(spec, &["config.json", VAE_WEIGHTS[0]], cache_dir)
    }
    #[cfg(not(feature = "hf"))]
    {
        let _ = cache_dir;
        bail!("{spec}: not a directory (built without the `hf` feature, so the VAE cannot be fetched)")
    }
}

/// JanusFlow's VAE decoder ([`super::JanusOptions::vae`]).
pub(crate) fn load_vae(model_dir: &std::path::Path, opts: &super::JanusOptions) -> Result<VaeDecoder> {
    let own = model_dir.join("vae");
    let mut dir = match &opts.vae {
        Some(spec) => vae_dir(spec, opts.cache_dir.as_deref())?,
        None if own.join("config.json").is_file() => own,
        None => vae_dir(DEFAULT_VAE, opts.cache_dir.as_deref())?,
    };
    // A whole diffusers pipeline: its `vae/`.
    if dir.join("vae").join("config.json").is_file() {
        dir = dir.join("vae");
    }
    let weights = VAE_WEIGHTS
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.is_file())
        .with_context(|| format!("{}: no VAE weights ({})", dir.display(), VAE_WEIGHTS.join(", ")))?;
    let vcfg: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
    let mut t = crate::sd::ops::Tensors::open(&weights)?;
    // The SDXL VAE is sensitive to weight rounding (it does not run in half precision): keep its
    // f32 weights whole (bf16 alone moves the image by 0.6%).
    t.set_split_f32(true);
    VaeDecoder::from_tensors(
        &t,
        vcfg["scaling_factor"].as_f64().unwrap_or(0.13025) as f32,
        vcfg["shift_factor"].as_f64().unwrap_or(0.0) as f32,
    )
    .with_context(|| format!("JanusFlow VAE {}", dir.display()))
}

impl Janus {
    fn flow(&self) -> Result<&FlowParts<'static>> {
        self.parts.flow.as_ref().context(if self.parts.gen_parts.is_some() {
            "this is a Janus-Pro checkpoint (image tokens, no rectified flow)"
        } else {
            "image generation was not loaded (JanusOptions::generation)"
        })
    }

    /// JanusFlow: integrates the rectified-flow ODE for `prompt` and returns the final latents of
    /// the `parallel_size` images (`[48 * 48, 4]` each, channels last, in the VAE's scaled
    /// space), calling `progress` after every Euler step.
    ///
    /// `noise` replaces the seeded starting noise (the images' latents one after the other);
    /// `trace` receives, for each step and image, the conditional velocity followed by the
    /// unconditional one (both `[48 * 48, 4]`). Both are for checks against the reference.
    pub fn flow_latents(
        &self,
        prompt: &str,
        opts: &super::ImageOptions,
        noise: Option<&[f32]>,
        mut trace: Option<&mut Vec<Vec<f32>>>,
        progress: &mut dyn FnMut(Progress) -> Flow,
    ) -> Result<(Vec<Vec<f32>>, ImageStats)> {
        self.cancel.store(false, Ordering::Relaxed);
        let f = self.flow()?;
        let llm = &self.parts.llm;
        let h = llm.cfg.hidden_size;
        // The generation tag that ends the prompt gives its place to the time embedding.
        let mut ids = self.encode(&self.generation_text(prompt))?;
        ids.pop();
        let l = ids.len();
        if l < 2 {
            bail!("prompt too short");
        }
        let n = opts.parallel_size.max(1);
        let (side, m) = (f.latent, f.latent_len());
        let mut z: Vec<Vec<f32>> = match noise {
            Some(v) if v.len() == n * m => v.chunks(m).map(<[f32]>::to_vec).collect(),
            Some(v) => bail!("noise: {} values, expected {n} x {m}", v.len()),
            None => {
                let mut rng = Rng::new(opts.seed);
                (0..n).map(|_| rng.normals(m)).collect()
            }
        };
        let cfg = opts.cfg_weight.unwrap_or(DEFAULT_CFG);
        // A weight of 1 is the conditional velocity alone.
        let guided = cfg != 1.0;
        let steps = opts.steps.max(1);
        let t0 = Instant::now();
        let mut cond = llm.new_state(1);
        llm.forward(&mut cond, llm.embed(&ids), l);
        let mut seqs: Vec<&BatchState> = vec![&cond; n];
        // The unconditional sequence: the prompt padded after BOS and masked, so only BOS is
        // attended to, at the same positions.
        let mut uncond = llm.new_state(1);
        if guided {
            llm.forward(&mut uncond, llm.embed(&ids[..1]), 1);
            seqs.extend(vec![&uncond; n]);
        }
        let mut st = BatchState::join(&seqs, l);
        let lens = st.lens();
        let mut stats = ImageStats {
            prompt_tokens: l,
            prefill: t0.elapsed().as_secs_f64(),
            steps,
            ..Default::default()
        };
        let t1 = Instant::now();
        let grid = side / f.enc.patch;
        let tok = grid * grid;
        let dt = 1.0 / steps as f32;
        for step in 0..steps {
            let t_emb = f.enc.time((step as f64 / steps as f64 * 1000.0) as f32);
            let mut t_act = t_emb.clone();
            silu_inplace(&mut t_act);
            let enc: Vec<(Vec<f32>, Vec<f32>)> = z.iter().map(|z| f.enc.encode(z, side, side, &t_act)).collect();
            let mut x = Vec::with_capacity(seqs.len() * (tok + 1) * h);
            for _ in 0..seqs.len() / n {
                for (_, emb) in &enc {
                    x.extend_from_slice(&t_emb);
                    x.extend_from_slice(emb);
                }
            }
            let hid = llm.forward(&mut st, x, tok + 1);
            st.rewind(&lens, l);
            for (i, z) in z.iter_mut().enumerate() {
                let rows = |b: usize| &hid[(b * (tok + 1) + 1) * h..(b + 1) * (tok + 1) * h];
                let vc = f.dec.velocity(rows(i), &enc[i].0, grid, grid, &t_act);
                let vu = if guided {
                    f.dec.velocity(rows(n + i), &enc[i].0, grid, grid, &t_act)
                } else {
                    vc.clone()
                };
                for ((z, &c), &u) in z.iter_mut().zip(&vc).zip(&vu) {
                    *z += dt * (cfg * c - (cfg - 1.0) * u);
                }
                if let Some(t) = trace.as_deref_mut() {
                    t.push([vc, vu].concat());
                }
            }
            if progress(Progress { step: step + 1, steps }) == Flow::Stop || self.cancel.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
        }
        stats.tokens = t1.elapsed().as_secs_f64();
        Ok((z, stats))
    }

    /// JanusFlow: decodes a latent of [`Janus::flow_latents`] with the SDXL VAE to RGB in about
    /// [-1, 1] (`[height * width, 3]`, unclipped); returns `(pixels, width, height)`.
    pub fn decode_latents_f32(&self, latents: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        let f = self.flow()?;
        if latents.len() != f.latent_len() {
            bail!("latents: {} values, expected {}", latents.len(), f.latent_len());
        }
        let (px, h, w) = f.vae.decode(latents, f.latent, f.latent);
        Ok((px, w, h))
    }
}
