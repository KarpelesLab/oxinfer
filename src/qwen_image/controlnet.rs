//! `QwenImageControlNetModel` (InstantX's Qwen-Image-ControlNet-Union and -Inpainting): a copy
//! of the transformer's first blocks (5 for the Union model, 6 for the inpainting one) with their
//! own input layers, run on the noisy latents plus the embedded control tokens
//! (`img_in(latents) + controlnet_x_embedder(control)`). The image stream after each of its
//! blocks goes through one linear layer (`controlnet_blocks.i`) and is scaled by the conditioning
//! scale; the main transformer adds residual `i / ceil(blocks / residuals)` to its image stream
//! after its block `i` ([`super::transformer::Extra::control`]).
//!
//! The control tokens are the packed VAE latents of the control image (canny, soft edge, depth,
//! pose: one model, no control-type input); the inpainting model takes the latents of the image
//! with the repainted region blanked plus the kept-region mask as a 17th latent channel
//! (`extra_condition_channels = 4` packed channels).

use super::transformer::{ImageShape, Transformer, TransformerConfig};
use super::weights::{Dense, Loader};
use anyhow::{Result, bail};
use std::path::Path;

/// Whether `dir` holds a Qwen-Image ControlNet (`config.json` with `QwenImageControlNetModel`).
pub fn is_controlnet(dir: &Path) -> bool {
    crate::fsx::read(&dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["_class_name"].as_str() == Some("QwenImageControlNetModel"))
}

pub struct ControlNet<'a> {
    t: Transformer<'a>,
    x_embedder: Dense<'a>,
    out: Vec<Dense<'a>>,
}

impl<'a> ControlNet<'a> {
    pub fn load(cfg: TransformerConfig, ld: &Loader<'a, '_>) -> Result<Self> {
        if cfg.zero_cond_t || cfg.use_additional_t_cond || cfg.use_layer3d_rope {
            bail!("ControlNet config: zero_cond_t, additional t conditions and layered RoPE are not supported");
        }
        let d = cfg.inner();
        let x_embedder = ld.dense("controlnet_x_embedder", d, cfg.in_channels + cfg.extra_condition_channels)?;
        let out = (0..cfg.num_layers)
            .map(|i| ld.dense(&format!("controlnet_blocks.{i}"), d, d))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            t: Transformer::load_with(cfg, ld, false)?,
            x_embedder,
            out,
        })
    }

    pub fn cfg(&self) -> &TransformerConfig {
        &self.t.cfg
    }

    /// Channels of a control token: the packed latents, plus the packed mask of an inpainting
    /// ControlNet.
    pub fn cond_channels(&self) -> usize {
        self.t.cfg.in_channels + self.t.cfg.extra_condition_channels
    }

    /// Whether this is an inpainting ControlNet (its control tokens carry a mask).
    pub fn inpaint(&self) -> bool {
        self.t.cfg.extra_condition_channels > 0
    }

    /// The text stream of this ControlNet (its own `txt_norm` / `txt_in`) for the text encoder's
    /// hidden states.
    pub fn embed_text(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        self.t.embed_text(hidden, t)
    }

    /// The residuals for the main transformer: `latents` (`[tokens, in_channels]`, the noisy
    /// target) and `cond` (`[tokens, cond_channels]`) of one `shape`, the text stream `txt`
    /// ([`ControlNet::embed_text`]) at noise level `sigma`; one `[tokens, inner]` per block,
    /// times `scale`.
    pub fn forward(&self, latents: &[f32], cond: &[f32], shape: ImageShape, txt: &[f32], sigma: f32, scale: f32) -> Vec<Vec<f32>> {
        let n = shape.tokens();
        assert_eq!(latents.len(), n * self.t.cfg.in_channels, "latents do not match the shape");
        assert_eq!(cond.len(), n * self.cond_channels(), "control tokens do not match the shape");
        let mut x = self.t.img_in.forward(latents, n);
        crate::kernels::add_inplace(&mut x, &self.x_embedder.forward(cond, n));
        let mut res = Vec::with_capacity(self.out.len());
        self.t.run_blocks(&mut x, &[shape], txt, sigma, None, &mut |i, x| {
            let mut r = self.out[i].forward(x, n);
            if scale != 1.0 {
                r.iter_mut().for_each(|v| *v *= scale);
            }
            res.push(r);
        });
        res
    }
}
