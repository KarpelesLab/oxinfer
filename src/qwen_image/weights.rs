//! Weight loading for the Qwen-Image transformer: biased linear layers in any precision (bf16,
//! Q8 / Q4 quantized at load, pre-quantized, GGUF blocks), with LoRA files (Lightning step
//! distillation, style LoRAs).
//!
//! LoRA keys: `<module>.lora_down.weight` / `.lora_up.weight` / `.alpha` (lightx2v, kohya-style
//! suffixes on diffusers module paths), PEFT / diffusers `<module>.lora_A.weight` /
//! `.lora_B.weight`, under an optional `transformer.` / `diffusion_model.` / `base_model.model.`
//! prefix, or kohya's `lora_unet_<module path with _>`. A weight quantized at load becomes
//! `W + strength * alpha / rank * up @ down` in f32 and is quantized from that (rounding it to
//! bf16 first would erase most of a Lightning update, which is below half a bf16 ulp of most
//! weights); bf16 weights and GGUF blocks keep the update as a low-rank pair applied at run time.
//!
//! FP8 "scaled" single-file checkpoints (`*_fp8_e4m3fn_scaled*.safetensors`) keep their weights
//! as stored: `float8_e4m3fn` bytes times a scale, one per tensor (ComfyUI's layout: a scalar
//! `<module>.scale_weight` beside every weight and a `scaled_fp8` marker tensor) or one per
//! output row (lightx2v's: `<module>.weight_scale` of shape `[n_out, 1]`, on the blocks' weights
//! only), or none (plain `*_fp8_e4m3fn.safetensors`). The tensor names are diffusers' own, under
//! an optional `model.diffusion_model.` prefix.

use crate::kernels::for_chunks;
use crate::model::{Lin, Quant, linear, linear_multi, load_lin};
use crate::quant::{PageVec, quantize_f32, quantize_q4_f32};
use crate::safetensors::{ModelFiles, SafeTensors};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

/// One LoRA file and the strength it is merged with.
#[derive(Clone, Debug)]
pub struct LoraSpec {
    pub path: PathBuf,
    pub scale: f32,
}

impl LoraSpec {
    /// `FILE[:SCALE]`; the scale defaults to 1.
    pub fn parse(s: &str) -> Result<Self> {
        let sd = crate::sd::lora::LoraSpec::parse(s)?;
        Ok(Self {
            path: sd.path,
            scale: sd.scale,
        })
    }
}

/// A low-rank update: `down` `[r, in]`, `up` `[out, r]`, already scaled.
struct Pair {
    down: Vec<f32>,
    up: Vec<f32>,
    r: usize,
    scale: f32,
}

/// The LoRA updates of a set of files, by base weight name (`<module>.weight`).
pub struct LoraSet {
    by_weight: HashMap<String, Vec<Pair>>,
    applied: Mutex<usize>,
    pub files: Vec<(String, usize)>,
}

fn tensor_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let info = st.info(name).with_context(|| format!("missing tensor {name}"))?;
    let b = st.bytes_cow(name)?;
    Ok(match info.dtype.as_str() {
        "F32" => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        "BF16" => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| crate::kernels::bf16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        "F16" => b.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect(),
        d => bail!("LoRA tensor {name}: unsupported dtype {d}"),
    })
}

impl LoraSet {
    /// Parses the LoRA files and ties their modules to the weights of `base` (whose tensor names
    /// may carry `prefix`). Errors when a module matches no weight.
    pub fn open(specs: &[LoraSpec], base: &ModelFiles, base_names: &[String], prefix: &str) -> Result<Self> {
        // kohya's underscore paths -> base weight names.
        let kohya: HashMap<String, String> = base_names
            .iter()
            .filter_map(|n| {
                n.strip_prefix(prefix)
                    .and_then(|m| m.strip_suffix(".weight"))
                    .map(|m| (m.replace('.', "_"), n.clone()))
            })
            .collect();
        let mut by_weight: HashMap<String, Vec<Pair>> = HashMap::new();
        let mut files = Vec::new();
        for spec in specs {
            let st = SafeTensors::open(&spec.path, false).with_context(|| format!("LoRA {}", spec.path.display()))?;
            let mut modules: HashMap<String, [Option<String>; 3]> = HashMap::new();
            let mut odd = Vec::new();
            const SUF: [(&str, usize); 8] = [
                (".lora_down.weight", 0),
                (".lora_up.weight", 1),
                (".lora_A.default.weight", 0),
                (".lora_B.default.weight", 1),
                (".lora_A.weight", 0),
                (".lora_B.weight", 1),
                (".lora.down.weight", 0),
                (".alpha", 2),
            ];
            for k in st.names() {
                match SUF.iter().find_map(|(s, r)| k.strip_suffix(s).map(|p| (p, *r))) {
                    Some((p, r)) => modules.entry(p.to_string()).or_default()[r] = Some(k.clone()),
                    None => odd.push(k.clone()),
                }
            }
            if !odd.is_empty() {
                odd.sort();
                bail!(
                    "LoRA {}: {} keys are not low-rank pairs (e.g. {}); only LoRA (down / up) modules are supported",
                    spec.path.display(),
                    odd.len(),
                    odd[0]
                );
            }
            let mut n = 0;
            for (m, [down, up, alpha]) in modules {
                let (Some(down), Some(up)) = (down, up) else {
                    bail!("LoRA {}: module {m} lacks its down or up matrix", spec.path.display());
                };
                let path = ["transformer.", "diffusion_model.", "base_model.model.", "model.diffusion_model."]
                    .iter()
                    .find_map(|p| m.strip_prefix(p))
                    .unwrap_or(&m);
                let weight = if let Some(u) = path.strip_prefix("lora_unet_") {
                    kohya.get(u).cloned()
                } else {
                    let w = format!("{prefix}{path}.weight");
                    base.has(&w).then_some(w)
                };
                let Some(weight) = weight else {
                    bail!("LoRA {}: module {m} matches no transformer weight", spec.path.display());
                };
                let ds = st.info(&down).unwrap().shape.clone();
                let us = st.info(&up).unwrap().shape.clone();
                let (r, n_in, n_out) = match (ds.as_slice(), us.as_slice()) {
                    ([r, i], [o, r2]) if r == r2 => (*r, *i, *o),
                    _ => bail!("LoRA {}: module {m} has shapes {ds:?} / {us:?}", spec.path.display()),
                };
                let ws = base.info(&weight).unwrap().shape.clone();
                if ws.iter().product::<usize>() != n_in * n_out {
                    bail!("LoRA {}: module {m} is {n_out}x{n_in}, the weight is {ws:?}", spec.path.display());
                }
                let alpha = match alpha {
                    Some(a) => tensor_f32(&st, &a)?.first().copied().unwrap_or(r as f32),
                    None => r as f32,
                };
                by_weight.entry(weight).or_default().push(Pair {
                    down: tensor_f32(&st, &down)?,
                    up: tensor_f32(&st, &up)?,
                    r,
                    scale: spec.scale * alpha / r as f32,
                });
                n += 1;
            }
            files.push((spec.path.display().to_string(), n));
        }
        Ok(Self {
            by_weight,
            applied: Mutex::new(0),
            files,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.by_weight.is_empty()
    }

    /// Adds the updates of weight `name` (`[n_out, n_in]`, row-major f32) to `w`; false if none.
    fn apply(&self, name: &str, n_out: usize, n_in: usize, w: &mut [f32]) -> bool {
        let Some(pairs) = self.by_weight.get(name) else {
            return false;
        };
        for p in pairs {
            // up [out, r] @ down [r, in]: rows of up against the columns of down (down^T rows).
            let mut dt = vec![0f32; n_in * p.r];
            for i in 0..p.r {
                for j in 0..n_in {
                    dt[j * p.r + i] = p.down[i * n_in + j];
                }
            }
            let d = crate::sd::ops::matmul_f32(&dt, n_in, p.r, &p.up, n_out);
            let s = p.scale;
            for_chunks(w, n_in, true, |o, row| {
                for (a, &b) in row.iter_mut().zip(&d[o * n_in..(o + 1) * n_in]) {
                    *a += s * b;
                }
            });
        }
        *self.applied.lock().unwrap() += pairs.len();
        true
    }

    /// Modules merged so far.
    pub fn applied(&self) -> usize {
        *self.applied.lock().unwrap()
    }

    /// Modules in the files.
    pub fn modules(&self) -> usize {
        self.by_weight.values().map(|v| v.len()).sum()
    }

    /// The updates of weight `name` as one low-rank pair (several files' ranks stacked): `down`
    /// `[r, n_in]` and `up` `[n_out, r]` with the scales folded in; None if none.
    fn stacked(&self, name: &str, n_out: usize, n_in: usize) -> Option<(Vec<f32>, Vec<f32>, usize)> {
        let pairs = self.by_weight.get(name)?;
        let r: usize = pairs.iter().map(|p| p.r).sum();
        let mut down = Vec::with_capacity(r * n_in);
        let mut up = vec![0f32; n_out * r];
        let mut c0 = 0;
        for p in pairs {
            down.extend_from_slice(&p.down);
            for o in 0..n_out {
                for i in 0..p.r {
                    up[o * r + c0 + i] = p.up[o * p.r + i] * p.scale;
                }
            }
            c0 += p.r;
        }
        *self.applied.lock().unwrap() += pairs.len();
        Some((down, up, r))
    }
}

/// A LoRA update applied at run time: `y += up (down x)`, both small matmuls through the shared
/// matmul (so on the GPU when the base one is).
pub struct RuntimeLora {
    down: Lin<'static>,
    up: Lin<'static>,
    r: usize,
}

impl RuntimeLora {
    fn new(down: &[f32], up: &[f32], r: usize) -> Self {
        let bf = |v: &[f32]| {
            let mut p = PageVec::<u16>::zeroed(v.len());
            for (o, &x) in p.iter_mut().zip(v) {
                *o = crate::sd::ops::f32_to_bf16(x);
            }
            Lin::bf16_owned(p)
        };
        Self {
            down: bf(down),
            up: bf(up),
            r,
        }
    }
}

/// Input channels of a quantized layer kept exact: `w` holds their weight columns (`[n_out,
/// idx.len()]`, f32) and the quantized matrix has them zeroed.
pub struct ExactCols {
    idx: Vec<usize>,
    w: Vec<f32>,
}

/// A linear layer with an optional bias (and a LoRA update applied at run time).
pub struct Dense<'a> {
    pub w: Lin<'a>,
    pub b: Option<Vec<f32>>,
    pub n_out: usize,
    pub n_in: usize,
    pub lora: Option<RuntimeLora>,
    pub exact: Option<ExactCols>,
}

impl Dense<'_> {
    /// `x` is `[rows, n_in]`; returns `[rows, n_out]`.
    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows * self.n_out];
        Self::forward_multi(&mut [(self, &mut y[..])], x, rows);
        y
    }

    /// [`Dense::forward`] for a few rows (the timestep embedding and modulation layers), padded
    /// to 16 rows so quantized weights take the prefill path (dequantized weights against f32
    /// activations) instead of the decode kernels, which round the activations to int8 per block
    /// of 32: the timestep embeddings have a few channels far larger than the rest, which int8
    /// blocks would flatten.
    pub fn forward_small(&self, x: &[f32], rows: usize) -> Vec<f32> {
        const PAD: usize = 16;
        if rows >= PAD {
            return self.forward(x, rows);
        }
        let mut xp = vec![0f32; PAD * self.n_in];
        xp[..rows * self.n_in].copy_from_slice(&x[..rows * self.n_in]);
        let mut y = self.forward(&xp, PAD);
        y.truncate(rows * self.n_out);
        y
    }

    /// Several layers over the same input (one fused matmul region).
    pub fn forward_multi(jobs: &mut [(&Self, &mut [f32])], x: &[f32], rows: usize) {
        let _t = crate::sd::ops::timing::start(crate::sd::ops::timing::LINEAR);
        let n_in = jobs[0].0.n_in;
        let mut lj: Vec<(&Lin, usize, &mut [f32])> = jobs.iter_mut().map(|(d, y)| (&d.w, d.n_out, &mut **y)).collect();
        linear_multi(&mut lj, n_in, x, rows);
        // LoRA down projections share the input: one fused region, then each up projection.
        let mut hs: Vec<Vec<f32>> = jobs
            .iter()
            .map(|(d, _)| vec![0f32; d.lora.as_ref().map_or(0, |l| rows * l.r)])
            .collect();
        let mut dj: Vec<(&Lin, usize, &mut [f32])> = jobs
            .iter()
            .zip(hs.iter_mut())
            .filter_map(|((d, _), h)| d.lora.as_ref().map(|l| (&l.down, l.r, &mut h[..])))
            .collect();
        if !dj.is_empty() {
            linear_multi(&mut dj, n_in, x, rows);
        }
        for ((d, y), h) in jobs.iter_mut().zip(&hs) {
            if let Some(e) = &d.exact {
                let k = e.idx.len();
                for_chunks(y, d.n_out, rows > 1, |t, yr| {
                    let xs: Vec<f32> = e.idx.iter().map(|&c| x[t * n_in + c]).collect();
                    for (v, wr) in yr.iter_mut().zip(e.w.chunks_exact(k)) {
                        *v += wr.iter().zip(&xs).map(|(a, b)| a * b).sum::<f32>();
                    }
                });
            }
            if let Some(l) = &d.lora {
                let mut u = vec![0f32; rows * d.n_out];
                linear(&l.up, d.n_out, l.r, h, rows, &mut u);
                crate::kernels::add_inplace(y, &u);
            }
            if let Some(b) = &d.b {
                for_chunks(y, d.n_out, rows > 1, |_, r| {
                    for (v, &bb) in r.iter_mut().zip(b) {
                        *v += bb;
                    }
                });
            }
        }
    }
}

/// Loads the transformer's layers from `files` in precision `quant` with the LoRA updates of
/// `lora`: merged into the weights when they are quantized at load anyway (a copy is made
/// either way), applied at run time otherwise (bf16 weights borrowed from the checkpoint, GGUF
/// blocks), so no full-size copy is ever made for a LoRA.
pub struct Loader<'a, 'l> {
    pub files: &'a ModelFiles,
    pub quant: Quant,
    pub lora: Option<&'l LoraSet>,
    pub prefix: String,
}

impl<'a> Loader<'a, '_> {
    pub fn has(&self, name: &str) -> bool {
        self.files.has(&format!("{}{name}", self.prefix))
    }

    pub fn f32(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        let full = format!("{}{name}", self.prefix);
        let v = self.files.f32(&full)?;
        if v.len() != n {
            bail!("tensor {full} has {} values, expected {n}", v.len());
        }
        Ok(v)
    }

    /// `{name}.weight` `[n_out, n_in]` and `{name}.bias` if present.
    pub fn dense(&self, name: &str, n_out: usize, n_in: usize) -> Result<Dense<'a>> {
        let wn = format!("{}{name}.weight", self.prefix);
        if self.quant != Quant::Bf16
            && let Ok(v) = std::env::var("OXINFER_QI_BF16")
            && v.split(',').any(|p| !p.is_empty() && name.contains(p))
        {
            let ld = Loader {
                files: self.files,
                quant: Quant::Bf16,
                lora: self.lora,
                prefix: self.prefix.clone(),
            };
            return ld.dense(name, n_out, n_in);
        }
        let merge = self.quant != Quant::Bf16 && self.quantizes(&wn);
        let (w, lora) = match self.lora.filter(|l| l.by_weight.contains_key(&wn)) {
            Some(l) if merge => (self.merged(&wn, n_out, n_in, l)?, None),
            Some(l) => {
                let (d, u, r) = l.stacked(&wn, n_out, n_in).unwrap();
                (self.lin(&wn, n_out, n_in)?, Some(RuntimeLora::new(&d, &u, r)))
            }
            None => (self.lin(&wn, n_out, n_in)?, None),
        };
        let bias = if self.has(&format!("{name}.bias")) {
            Some(self.f32(&format!("{name}.bias"), n_out)?)
        } else {
            None
        };
        Ok(Dense {
            w,
            b: bias,
            n_out,
            n_in,
            lora,
            exact: None,
        })
    }

    /// Whether weight `wn` is quantized here from float weights (not GGUF, FP8 or pre-quantized).
    fn quantizes(&self, wn: &str) -> bool {
        self.files.ggml(wn).is_none() && Quant::stored(self.files).is_none() && !self.is_fp8(wn)
    }

    fn is_fp8(&self, wn: &str) -> bool {
        self.files.info(wn).is_some_and(|i| i.dtype == "F8_E4M3")
    }

    /// Weight `wn` in this loader's precision, or as stored (GGUF, pre-quantized, FP8).
    fn lin(&self, wn: &str, n_out: usize, n_in: usize) -> Result<Lin<'a>> {
        if self.is_fp8(wn) && !self.files.has(&wn.replace(".weight", ".weight_scale_inv")) {
            return fp8_scaled(self.files, wn, n_out, n_in);
        }
        load_lin(self.files, wn, n_out, n_in, self.quant)
    }

    /// [`Loader::dense`] with input channels `cols` kept exact when the weight is quantized here:
    /// for inputs whose energy sits in a few channels (the modulation layers read
    /// `silu(timestep embedding)`, where a handful of channels hold 99.9% of it), the block
    /// rounding of those channels' weights would otherwise dominate the error.
    pub fn dense_exact_cols(&self, name: &str, n_out: usize, n_in: usize, cols: &[usize]) -> Result<Dense<'a>> {
        let wn = format!("{}{name}.weight", self.prefix);
        if self.quant == Quant::Bf16 || cols.is_empty() || !self.quantizes(&wn) {
            return self.dense(name, n_out, n_in);
        }
        let mut w = self.files.f32(&wn)?;
        if w.len() != n_out * n_in {
            bail!("tensor {wn} has {} values, expected {n_out}x{n_in}", w.len());
        }
        if let Some(l) = self.lora {
            l.apply(&wn, n_out, n_in, &mut w);
        }
        let k = cols.len();
        let mut cw = vec![0f32; n_out * k];
        for (row, c) in w.chunks_exact_mut(n_in).zip(cw.chunks_exact_mut(k)) {
            for (dst, &i) in c.iter_mut().zip(cols) {
                *dst = std::mem::take(&mut row[i]);
            }
        }
        self.files.mark_quant(&wn, self.quant.kind().unwrap_or(crate::quant::QKind::Q8));
        let lin = match self.quant {
            Quant::Q4 => Lin::q4(quantize_q4_f32(&w, n_out, n_in)),
            _ => Lin::q8(quantize_f32(&w, n_out, n_in)),
        };
        let b = if self.has(&format!("{name}.bias")) {
            Some(self.f32(&format!("{name}.bias"), n_out)?)
        } else {
            None
        };
        Ok(Dense {
            w: lin,
            b,
            n_out,
            n_in,
            lora: None,
            exact: Some(ExactCols { idx: cols.to_vec(), w: cw }),
        })
    }

    /// Weight `wn` (bf16 / f16 / f32 safetensors) with its LoRA updates merged, quantized to `quant`.
    fn merged(&self, wn: &str, n_out: usize, n_in: usize, l: &LoraSet) -> Result<Lin<'a>> {
        let mut w = self.files.f32(wn)?;
        if w.len() != n_out * n_in {
            bail!("tensor {wn} has {} values, expected {n_out}x{n_in}", w.len());
        }
        l.apply(wn, n_out, n_in, &mut w);
        self.files.mark_quant(wn, self.quant.kind().unwrap_or(crate::quant::QKind::Q8));
        Ok(match self.quant {
            Quant::Bf16 => {
                let mut bf = PageVec::<u16>::zeroed(w.len());
                for (o, &v) in bf.iter_mut().zip(&w) {
                    *o = crate::sd::ops::f32_to_bf16(v);
                }
                Lin::bf16_owned(bf)
            }
            Quant::Q8 => Lin::q8(quantize_f32(&w, n_out, n_in)),
            Quant::Q4 => Lin::q4(quantize_q4_f32(&w, n_out, n_in)),
        })
    }
}

/// An FP8 E4M3 weight `wn` (`<module>.weight`, `[n_out, n_in]`) of a "scaled" checkpoint, used as
/// stored: the bytes borrowed from the file, times `<module>.scale_weight` (ComfyUI) or
/// `<module>.weight_scale` (lightx2v), a scalar or one value per output row; no scale tensor
/// means 1.
fn fp8_scaled<'a>(files: &'a ModelFiles, wn: &str, n_out: usize, n_in: usize) -> Result<Lin<'a>> {
    let info = files.info(wn).with_context(|| format!("missing tensor {wn}"))?;
    if info.shape != [n_out, n_in] {
        bail!("tensor {wn} has shape {:?}, expected [{n_out}, {n_in}]", info.shape);
    }
    let base = wn.strip_suffix(".weight").unwrap_or(wn);
    let sname = [format!("{base}.scale_weight"), format!("{base}.weight_scale")]
        .into_iter()
        .find(|n| files.has(n));
    let (scales, block) = match &sname {
        Some(n) => {
            let v = files.f32(n)?;
            match v.len() {
                1 => (v, (n_out, n_in)),
                l if l == n_out => (v, (1, n_in)),
                l => bail!("{n}: {l} scales for a [{n_out}, {n_in}] weight (expected 1 or {n_out})"),
            }
        }
        None => (vec![1.0], (n_out, n_in)),
    };
    let mut sp = PageVec::<f32>::zeroed(scales.len());
    sp.copy_from_slice(&scales);
    files.note_use(wn, true, false);
    let data = files.bytes(wn)?;
    // SAFETY: the weight borrows `files` for 'a, which outlives every model holding it.
    let m = crate::ggml::GgufMatrix::fp8(unsafe { crate::quant::QBuf::borrowed(data) }, sp.into(), n_out, n_in, block)
        .with_context(|| format!("tensor {wn}"))?;
    Ok(Lin::gguf(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Q8 layer with exact input channels: the kept columns contribute exactly, the rest as Q8.
    #[test]
    fn exact_cols_forward() {
        let (n_out, n_in, rows) = (8usize, 64usize, 3usize);
        let w: Vec<f32> = (0..n_out * n_in).map(|i| ((i * 37 % 101) as f32 - 50.0) / 400.0).collect();
        let mut x: Vec<f32> = (0..rows * n_in).map(|i| ((i * 13 % 29) as f32 - 14.0) / 10.0).collect();
        let cols = [5usize, 40];
        for r in 0..rows {
            x[r * n_in + 5] = 300.0;
            x[r * n_in + 40] = -150.0;
        }
        let mut wz = w.clone();
        let mut cw = Vec::new();
        for o in 0..n_out {
            for &c in &cols {
                cw.push(std::mem::take(&mut wz[o * n_in + c]));
            }
        }
        let q = quantize_f32(&wz, n_out, n_in);
        let mut deq = vec![0f32; n_out * n_in];
        crate::quant::dequantize_rows(&q, 0, n_out, &mut deq);
        let d = Dense {
            w: Lin::q8(q),
            b: Some(vec![0.5; n_out]),
            n_out,
            n_in,
            lora: None,
            exact: Some(ExactCols { idx: cols.to_vec(), w: cw }),
        };
        let y = d.forward_small(&x, rows);
        for r in 0..rows {
            for o in 0..n_out {
                let mut want = 0.5f64;
                for i in 0..n_in {
                    let wv = if cols.contains(&i) { w[o * n_in + i] } else { deq[o * n_in + i] };
                    want += wv as f64 * x[r * n_in + i] as f64;
                }
                let got = y[r * n_out + o] as f64;
                assert!((got - want).abs() < 1e-3 * (1.0 + want.abs()), "row {r} out {o}: {got} vs {want}");
            }
        }
    }
}
