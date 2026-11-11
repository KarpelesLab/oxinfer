//! What Stable Diffusion XL adds to the SD 1.x / 2.x pipeline (diffusers
//! `StableDiffusionXLPipeline`, and its img2img and inpainting variants):
//!
//! - two text encoders, CLIP ViT-L (`text_encoder/`) and OpenCLIP bigG with its text projection
//!   (`text_encoder_2/`): the penultimate hidden states of both, concatenated per token
//!   (768 + 1280 = 2048), are the cross-attention context ([`TextXl`]);
//! - the UNet's added conditioning: the pooled embedding of the second encoder and six "time
//!   ids" (original size, crop offset, target size), each as a sinusoidal embedding, through a
//!   small MLP into the time embedding ([`AddEmbed`], [`SizeCond`]);
//! - LoRA files in kohya's SDXL naming, which follows the original (SGM) UNet layout
//!   (`lora_unet_input_blocks_4_1_...`) instead of diffusers' ([`sgm_names`]).
//!
//! The UNet itself (transformer depth per block, no attention at the top level, linear
//! projections), the VAE (scaling factor 0.13025 from its config) and the samplers are the ones
//! of `unet.rs`, `vae.rs` and `sched.rs`, which take their structure from the weights and configs.

use super::clip::{ClipText, ClipTokenizer};
use super::ops::{Linear, Tensors, silu_inplace, timestep_embedding};
use anyhow::{Context, Result, bail};
use std::path::Path;

/// The size conditioning of an SDXL image (`add_time_ids`), in pixels. The model was trained with
/// the size of the source picture, the top-left corner of the training crop and the size of the
/// training image; at inference diffusers passes the output size for both sizes and no crop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizeCond {
    /// `(height, width)` of the "original" picture.
    pub original: (usize, usize),
    /// `(top, left)` of the crop.
    pub crop: (usize, usize),
    /// `(height, width)` of the target.
    pub target: (usize, usize),
}

impl SizeCond {
    /// diffusers' defaults for a `width` x `height` image: both sizes are the image's, no crop.
    pub fn for_image(width: usize, height: usize) -> Self {
        Self {
            original: (height, width),
            crop: (0, 0),
            target: (height, width),
        }
    }

    /// The six ids in the order the UNet takes them.
    pub fn ids(&self) -> [f32; 6] {
        [
            self.original.0 as f32,
            self.original.1 as f32,
            self.crop.0 as f32,
            self.crop.1 as f32,
            self.target.0 as f32,
            self.target.1 as f32,
        ]
    }
}

/// The UNet's `add_embedding` (`addition_embed_type: "text_time"`): an MLP over the pooled text
/// embedding followed by the sinusoidal embedding of each time id.
pub struct AddEmbed {
    l1: Linear,
    l2: Linear,
    /// Width of one time id's sinusoidal embedding (`addition_time_embed_dim`, 256).
    time_dim: usize,
    flip_sin_to_cos: bool,
    freq_shift: f32,
}

impl AddEmbed {
    /// From the UNet's weights and `config.json`.
    pub fn load(t: &Tensors, cfg: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            l1: Linear::load(t, "add_embedding.linear_1")?,
            l2: Linear::load(t, "add_embedding.linear_2")?,
            time_dim: cfg["addition_time_embed_dim"].as_u64().context("addition_time_embed_dim")? as usize,
            flip_sin_to_cos: cfg["flip_sin_to_cos"].as_bool().unwrap_or(true),
            freq_shift: cfg["freq_shift"].as_f64().unwrap_or(0.0) as f32,
        })
    }

    /// Width of the pooled text embedding this UNet expects with `n_ids` time ids.
    pub fn pooled_dim(&self, n_ids: usize) -> Option<usize> {
        self.l1.n_in.checked_sub(n_ids * self.time_dim)
    }

    /// The embedding added to the UNet's time embedding (constant over the denoising steps).
    pub fn forward(&self, pooled: &[f32], ids: &[f32]) -> Result<Vec<f32>> {
        if pooled.len() + ids.len() * self.time_dim != self.l1.n_in {
            bail!(
                "this UNet's added conditioning takes {} values, not a {}-wide pooled embedding and {} time ids (the SDXL refiner is not supported)",
                self.l1.n_in,
                pooled.len(),
                ids.len()
            );
        }
        let mut x = pooled.to_vec();
        for &id in ids {
            x.extend(timestep_embedding(id, self.time_dim, self.flip_sin_to_cos, self.freq_shift));
        }
        let mut y = self.l1.forward(&x, 1);
        silu_inplace(&mut y);
        Ok(self.l2.forward(&y, 1))
    }
}

/// SDXL's second text encoder and how the two are combined.
pub struct TextXl {
    pub tokenizer: ClipTokenizer,
    pub text: ClipText,
    /// `force_zeros_for_empty_prompt` of `model_index.json`: an empty negative prompt conditions
    /// on zeros instead of the encoding of the empty string.
    pub force_zeros: bool,
}

/// The conditioning of one prompt: the cross-attention context and, for SDXL, the pooled
/// embedding of the second encoder.
pub struct TextCond {
    /// `[tokens, cross_dim]`.
    pub ctx: Vec<f32>,
    pub pooled: Option<Vec<f32>>,
}

impl TextXl {
    /// `dir` is the pipeline folder; `t` the opened `text_encoder_2` weights (with any LoRA
    /// updates attached).
    pub fn load(dir: &Path, index: &serde_json::Value, t: Tensors) -> Result<Self> {
        Ok(Self {
            tokenizer: ClipTokenizer::load(&dir.join("tokenizer_2")).context("tokenizer_2")?,
            text: ClipText::load(&dir.join("text_encoder_2"), t).context("text_encoder_2")?,
            force_zeros: index["force_zeros_for_empty_prompt"].as_bool().unwrap_or(true),
        })
    }

    /// Width of the concatenated context with first encoder `first`.
    pub fn cross_dim(&self, first: &ClipText) -> usize {
        first.hidden + self.text.hidden
    }

    /// Encodes `text` with both encoders (`first_tok` / `first` are the pipeline's first pair).
    /// `negative` marks the unconditional prompt, which is all zeros when it is empty and the
    /// pipeline says so.
    pub fn encode(&self, first_tok: &ClipTokenizer, first: &ClipText, text: &str, negative: bool) -> TextCond {
        let m = first_tok.max_len.min(self.tokenizer.max_len);
        let (d1, d2) = (first.hidden, self.text.hidden);
        if negative && text.is_empty() && self.force_zeros {
            return TextCond {
                ctx: vec![0.0; m * (d1 + d2)],
                pooled: Some(vec![0.0; self.pooled_dim()]),
            };
        }
        let (h1, _) = first.forward_penultimate(&first_tok.encode(text));
        // Both tokenizers frame and pad to the same length (77); they differ in the pad token.
        let (h2, pooled) = self.text.forward_penultimate(&self.tokenizer.encode(text));
        let mut ctx = Vec::with_capacity(m * (d1 + d2));
        for i in 0..m {
            ctx.extend_from_slice(&h1[i * d1..(i + 1) * d1]);
            ctx.extend_from_slice(&h2[i * d2..(i + 1) * d2]);
        }
        TextCond { ctx, pooled: Some(pooled) }
    }

    /// Width of the pooled embedding (the second encoder's projection).
    pub fn pooled_dim(&self) -> usize {
        self.text.pooled_dim()
    }
}

/// The original (SGM / LDM) names of a diffusers UNet's weights, in kohya's underscore form:
/// `(sgm module with _ for ., diffusers weight name)` for every `*.weight` of `base` that has
/// one. kohya's SDXL LoRAs name their targets this way
/// (`lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q`,
/// `lora_unet_middle_block_1_proj_in`, `lora_unet_output_blocks_0_0_in_layers_2`).
pub fn sgm_names(base: &Tensors) -> Vec<(String, String)> {
    // ResNets per down block (the up blocks have one more).
    let mut layers = 0;
    while base.has(&format!("down_blocks.0.resnets.{layers}.conv1.weight")) {
        layers += 1;
    }
    if layers == 0 {
        return Vec::new();
    }
    let up_has_attn = |i: usize| base.has(&format!("up_blocks.{i}.attentions.0.proj_in.weight"));
    base.names()
        .filter_map(|n| n.strip_suffix(".weight"))
        .filter_map(|m| sgm_module(m, layers, &up_has_attn).map(|s| (s.replace('.', "_"), format!("{m}.weight"))))
        .collect()
}

/// The SGM name of diffusers UNet module `m` (`layers` ResNets per down block).
fn sgm_module(m: &str, layers: usize, up_has_attn: &dyn Fn(usize) -> bool) -> Option<String> {
    let resnet = |rest: &str| -> Option<&'static str> {
        Some(match rest {
            "norm1" => "in_layers.0",
            "conv1" => "in_layers.2",
            "time_emb_proj" => "emb_layers.1",
            "norm2" => "out_layers.0",
            "conv2" => "out_layers.3",
            "conv_shortcut" => "skip_connection",
            _ => return None,
        })
    };
    match m {
        "conv_in" => return Some("input_blocks.0.0".into()),
        "time_embedding.linear_1" => return Some("time_embed.0".into()),
        "time_embedding.linear_2" => return Some("time_embed.2".into()),
        "add_embedding.linear_1" => return Some("label_emb.0.0".into()),
        "add_embedding.linear_2" => return Some("label_emb.0.2".into()),
        "conv_norm_out" => return Some("out.0".into()),
        "conv_out" => return Some("out.2".into()),
        _ => {}
    }
    let mut it = m.splitn(5, '.');
    let (block, a, kind, b) = (it.next()?, it.next()?, it.next()?, it.next()?);
    let rest = it.next().unwrap_or("");
    let per = layers + 1;
    match (block, kind) {
        ("mid_block", _) => {
            // `mid_block.resnets.N.REST` / `mid_block.attentions.0.REST`: one level less.
            let (kind, idx, rest) = (a, kind.parse::<usize>().ok()?, m.splitn(4, '.').nth(3)?);
            match kind {
                "resnets" => Some(format!("middle_block.{}.{}", idx * 2, resnet(rest)?)),
                "attentions" => Some(format!("middle_block.1.{rest}")),
                _ => None,
            }
        }
        ("down_blocks", _) => {
            let (i, j) = (a.parse::<usize>().ok()?, b.parse::<usize>().ok()?);
            match kind {
                "resnets" => Some(format!("input_blocks.{}.0.{}", 1 + i * per + j, resnet(rest)?)),
                "attentions" => Some(format!("input_blocks.{}.1.{rest}", 1 + i * per + j)),
                "downsamplers" if rest == "conv" => Some(format!("input_blocks.{}.0.op", (i + 1) * per)),
                _ => None,
            }
        }
        ("up_blocks", _) => {
            let (i, j) = (a.parse::<usize>().ok()?, b.parse::<usize>().ok()?);
            match kind {
                "resnets" => Some(format!("output_blocks.{}.0.{}", i * per + j, resnet(rest)?)),
                "attentions" => Some(format!("output_blocks.{}.1.{rest}", i * per + j)),
                "upsamplers" if rest == "conv" => Some(format!(
                    "output_blocks.{}.{}.conv",
                    i * per + layers,
                    if up_has_attn(i) { 2 } else { 1 }
                )),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_ids() {
        let s = SizeCond::for_image(1024, 768);
        assert_eq!(s.ids(), [768.0, 1024.0, 0.0, 0.0, 768.0, 1024.0]);
    }

    #[test]
    fn sgm_module_names() {
        // SDXL: two ResNets per down block, attention in up blocks 0 and 1.
        let f = |m: &str| sgm_module(m, 2, &|i| i < 2);
        let eq = |m: &str, want: &str| assert_eq!(f(m).as_deref(), Some(want), "{m}");
        eq("conv_in", "input_blocks.0.0");
        eq("down_blocks.0.resnets.1.conv1", "input_blocks.2.0.in_layers.2");
        eq("down_blocks.0.downsamplers.0.conv", "input_blocks.3.0.op");
        eq(
            "down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_out.0",
            "input_blocks.4.1.transformer_blocks.1.attn2.to_out.0",
        );
        eq("down_blocks.2.resnets.0.conv_shortcut", "input_blocks.7.0.skip_connection");
        eq("down_blocks.2.attentions.1.proj_in", "input_blocks.8.1.proj_in");
        eq("mid_block.resnets.1.time_emb_proj", "middle_block.2.emb_layers.1");
        eq(
            "mid_block.attentions.0.transformer_blocks.9.ff.net.0.proj",
            "middle_block.1.transformer_blocks.9.ff.net.0.proj",
        );
        eq("up_blocks.0.resnets.2.conv2", "output_blocks.2.0.out_layers.3");
        eq("up_blocks.0.upsamplers.0.conv", "output_blocks.2.2.conv");
        eq("up_blocks.1.attentions.2.proj_out", "output_blocks.5.1.proj_out");
        eq("up_blocks.2.resnets.0.conv1", "output_blocks.6.0.in_layers.2");
        eq("add_embedding.linear_1", "label_emb.0.0");
        eq("conv_out", "out.2");
        assert_eq!(f("up_blocks.0.resnets.0.other"), None);
        // SD 1.x: the first up block has no attention, so its upsampler is entry 1.
        assert_eq!(
            sgm_module("up_blocks.0.upsamplers.0.conv", 2, &|i| i > 0).as_deref(),
            Some("output_blocks.2.1.conv")
        );
    }
}
