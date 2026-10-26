//! Runtime (unmerged) LoRA adapters for the language models: a PEFT adapter directory
//! (`adapter_config.json` + `adapter_model.safetensors`) is attached to the linear layers it
//! targets, and while its switch is on every targeted layer adds `scale * B (A x)` to its output.
//! The base weights stay untouched, in whatever format they were loaded (bf16, Q8 / Q4,
//! pre-quantized, MLX, GGUF), so one loaded model serves the base model (switch off: the code path
//! is exactly the one without an adapter, bit for bit) and the adapted one (switch on).
//!
//! The low-rank products are tiny next to the base matmuls (rank 16 against thousands of rows),
//! so they always run on the CPU, in f32, after the base matmul (CPU or GPU) has filled `y`.
//!
//! Keys are PEFT's: `base_model.model.<module path>.lora_A.weight` (`[r, in]`) and `.lora_B.weight`
//! (`[out, r]`). Module paths are matched from `layers.N.` on (the `model.` / `model.language_model.`
//! prefix differs between the text-only and the multimodal classes) or as `lm_head`.

use crate::kernels::dot;
use crate::pool;
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The low-rank update of one linear layer, attached to its `Lin`.
pub struct LoraDelta {
    /// `[r, n_in]`
    a: Vec<f32>,
    /// `[r, n_out]` (B transposed, so that the update is `r` contiguous axpys per token).
    bt: Vec<f32>,
    r: usize,
    n_in: usize,
    n_out: usize,
    scale: f32,
    active: Arc<AtomicBool>,
}

impl LoraDelta {
    #[inline]
    pub(crate) fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Parameter bytes held (f32).
    pub fn bytes(&self) -> usize {
        (self.a.len() + self.bt.len()) * 4
    }

    /// `y += scale * B (A x)` for `t` rows of `x` (`[t, n_in]`) into `y` (`[t, n_out]`).
    pub(crate) fn apply(&self, x: &[f32], t: usize, y: &mut [f32]) {
        let (r, n_in, n_out) = (self.r, self.n_in, self.n_out);
        assert_eq!(x.len(), t * n_in, "LoRA input shape mismatch");
        assert_eq!(y.len(), t * n_out, "LoRA output shape mismatch");
        if r == 0 || t == 0 {
            return;
        }
        let _timer = crate::prof::start(crate::prof::Cat::Matmul);
        // Work items cover groups of G tokens, computed in blocks of TB: each slice of A or B
        // brought into L1 serves the whole group, each register load of it a block.
        const G: usize = 16;
        let groups = t.div_ceil(G);
        let blocks = |g: usize| (g * G..((g + 1) * G).min(t)).step_by(TB).map(move |i0| (i0, (i0 + TB).min(t)));
        // u = scale * A x: [t, r], over slices of the input that keep the group's activations
        // in L1 while every rank component reads them.
        const KC: usize = 512;
        let mut u = vec![0f32; t * r];
        let up = SendPtr(u.as_mut_ptr());
        pool::run(groups, &|g| {
            let (g0, g1) = (g * G, ((g + 1) * G).min(t));
            // SAFETY: each token group of u is written by one item.
            let ug = unsafe { std::slice::from_raw_parts_mut(up.get().add(g0 * r), (g1 - g0) * r) };
            for c0 in (0..n_in).step_by(KC) {
                let c1 = (c0 + KC).min(n_in);
                for k in 0..r {
                    let ak = &self.a[k * n_in + c0..k * n_in + c1];
                    for (i0, i1) in blocks(g) {
                        let mut d = [0f32; TB];
                        dot_rows(ak, &x[i0 * n_in + c0..], n_in, i1 - i0, &mut d);
                        for (ii, v) in d[..i1 - i0].iter().enumerate() {
                            ug[(i0 - g0 + ii) * r + k] += v;
                        }
                    }
                }
            }
            for v in ug.iter_mut() {
                *v *= self.scale;
            }
        });
        // y += u B^T over (token group, column block) items.
        const BLK: usize = 256;
        let nb = n_out.div_ceil(BLK);
        let yp = SendPtr(y.as_mut_ptr());
        pool::run(groups * nb, &|j| {
            let (g, cb) = (j / nb, j % nb);
            let (c0, c1) = (cb * BLK, ((cb + 1) * BLK).min(n_out));
            for (i0, i1) in blocks(g) {
                let w = c1 - c0;
                // SAFETY: each (token, column block) of y is written by one item.
                let row = |i: usize| unsafe { std::slice::from_raw_parts_mut(yp.get().add(i * n_out + c0), w) };
                let bt = |k: usize, c: usize| -> &[f32; 8] { self.bt[k * n_out + c0 + c..].first_chunk::<8>().unwrap() };
                let w8 = w / 8 * 8;
                if i1 - i0 == TB {
                    // Four tokens share every 8-column slice of B.
                    let (y0, y1, y2, y3) = (row(i0), row(i0 + 1), row(i0 + 2), row(i0 + 3));
                    let us = &u[i0 * r..i1 * r];
                    update4([y0, y1, y2, y3], us, r, &self.bt[c0..], n_out, w8);
                } else {
                    for i in i0..i1 {
                        let y = row(i);
                        let ur = &u[i * r..(i + 1) * r];
                        for c in (0..w8).step_by(8) {
                            let mut acc = [0f32; 8];
                            acc.copy_from_slice(&y[c..c + 8]);
                            for (k, &uk) in ur.iter().enumerate() {
                                let b = bt(k, c);
                                for l in 0..8 {
                                    acc[l] += uk * b[l];
                                }
                            }
                            y[c..c + 8].copy_from_slice(&acc);
                        }
                    }
                }
                for i in i0..i1 {
                    let y = row(i);
                    for c in w8..w {
                        for k in 0..r {
                            y[c] += u[i * r + k] * self.bt[k * n_out + c0 + c];
                        }
                    }
                }
            }
        });
    }
}

/// Tokens per block of the LoRA products.
const TB: usize = 4;

/// `d[i] = a . x[i * stride ..]` for `n <= TB` rows of `x`, reading `a` once.
#[inline]
fn dot_rows(a: &[f32], x: &[f32], stride: usize, n: usize, d: &mut [f32; TB]) {
    let m = a.len();
    let row = |i: usize| &x[i * stride..i * stride + m];
    if n < TB {
        for i in 0..n {
            d[i] = dot(a, row(i));
        }
        return;
    }
    let xs = [row(0), row(1), row(2), row(3)];
    let m8 = m / 8 * 8;
    *d = dot4(&a[..m8], xs.map(|r| &r[..m8]));
    for (i, xi) in xs.iter().enumerate() {
        for o in m8..m {
            d[i] += a[o] * xi[o];
        }
    }
}

/// `a . x[i]` for four rows as long as `a` (a multiple of 8).
#[cfg(target_arch = "aarch64")]
fn dot4(a: &[f32], x: [&[f32]; 4]) -> [f32; 4] {
    use std::arch::aarch64::*;
    let m = a.len();
    assert!(m.is_multiple_of(8) && x.iter().all(|r| r.len() == m));
    // SAFETY: every load is within `a` or a row of `x` (lengths checked); NEON is baseline on
    // aarch64. Two accumulators per row keep eight FMA chains in flight.
    unsafe {
        let z = vdupq_n_f32(0.0);
        let (mut s0, mut t0, mut s1, mut t1, mut s2, mut t2, mut s3, mut t3) = (z, z, z, z, z, z, z, z);
        let (pa, p0, p1, p2, p3) = (a.as_ptr(), x[0].as_ptr(), x[1].as_ptr(), x[2].as_ptr(), x[3].as_ptr());
        for o in (0..m).step_by(8) {
            let (a0, a1) = (vld1q_f32(pa.add(o)), vld1q_f32(pa.add(o + 4)));
            s0 = vfmaq_f32(s0, a0, vld1q_f32(p0.add(o)));
            t0 = vfmaq_f32(t0, a1, vld1q_f32(p0.add(o + 4)));
            s1 = vfmaq_f32(s1, a0, vld1q_f32(p1.add(o)));
            t1 = vfmaq_f32(t1, a1, vld1q_f32(p1.add(o + 4)));
            s2 = vfmaq_f32(s2, a0, vld1q_f32(p2.add(o)));
            t2 = vfmaq_f32(t2, a1, vld1q_f32(p2.add(o + 4)));
            s3 = vfmaq_f32(s3, a0, vld1q_f32(p3.add(o)));
            t3 = vfmaq_f32(t3, a1, vld1q_f32(p3.add(o + 4)));
        }
        [
            vaddvq_f32(vaddq_f32(s0, t0)),
            vaddvq_f32(vaddq_f32(s1, t1)),
            vaddvq_f32(vaddq_f32(s2, t2)),
            vaddvq_f32(vaddq_f32(s3, t3)),
        ]
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn dot4(a: &[f32], x: [&[f32]; 4]) -> [f32; 4] {
    // Named accumulators: arrays of them are not always kept in registers.
    let (mut s0, mut s1, mut s2, mut s3) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
    let it = a.as_chunks::<8>().0.iter().zip(x[0].as_chunks::<8>().0);
    let it = it
        .zip(x[1].as_chunks::<8>().0)
        .zip(x[2].as_chunks::<8>().0)
        .zip(x[3].as_chunks::<8>().0);
    for ((((a8, p0), p1), p2), p3) in it {
        for l in 0..8 {
            s0[l] += a8[l] * p0[l];
            s1[l] += a8[l] * p1[l];
            s2[l] += a8[l] * p2[l];
            s3[l] += a8[l] * p3[l];
        }
    }
    [s0, s1, s2, s3].map(|s| s.iter().sum())
}

/// `y[i][c] += sum_k u[i * r + k] * bt[k * stride + c]` for four rows `y` and the first `w8`
/// columns (a multiple of 8).
#[cfg(target_arch = "aarch64")]
fn update4(y: [&mut [f32]; 4], u: &[f32], r: usize, bt: &[f32], stride: usize, w8: usize) {
    use std::arch::aarch64::*;
    assert!(w8.is_multiple_of(8) && y.iter().all(|v| v.len() >= w8) && u.len() >= 4 * r);
    assert!(r == 0 || bt.len() >= (r - 1) * stride + w8);
    // SAFETY: every access is within the bounds checked above; NEON is baseline on aarch64.
    unsafe {
        let [y0, y1, y2, y3] = y.map(|v| v.as_mut_ptr());
        let (pu, pb) = (u.as_ptr(), bt.as_ptr());
        for c in (0..w8).step_by(8) {
            let (mut a0, mut b0) = (vld1q_f32(y0.add(c)), vld1q_f32(y0.add(c + 4)));
            let (mut a1, mut b1) = (vld1q_f32(y1.add(c)), vld1q_f32(y1.add(c + 4)));
            let (mut a2, mut b2) = (vld1q_f32(y2.add(c)), vld1q_f32(y2.add(c + 4)));
            let (mut a3, mut b3) = (vld1q_f32(y3.add(c)), vld1q_f32(y3.add(c + 4)));
            for k in 0..r {
                let (lo, hi) = (vld1q_f32(pb.add(k * stride + c)), vld1q_f32(pb.add(k * stride + c + 4)));
                let (u0, u1, u2, u3) = (*pu.add(k), *pu.add(r + k), *pu.add(2 * r + k), *pu.add(3 * r + k));
                a0 = vfmaq_n_f32(a0, lo, u0);
                b0 = vfmaq_n_f32(b0, hi, u0);
                a1 = vfmaq_n_f32(a1, lo, u1);
                b1 = vfmaq_n_f32(b1, hi, u1);
                a2 = vfmaq_n_f32(a2, lo, u2);
                b2 = vfmaq_n_f32(b2, hi, u2);
                a3 = vfmaq_n_f32(a3, lo, u3);
                b3 = vfmaq_n_f32(b3, hi, u3);
            }
            vst1q_f32(y0.add(c), a0);
            vst1q_f32(y0.add(c + 4), b0);
            vst1q_f32(y1.add(c), a1);
            vst1q_f32(y1.add(c + 4), b1);
            vst1q_f32(y2.add(c), a2);
            vst1q_f32(y2.add(c + 4), b2);
            vst1q_f32(y3.add(c), a3);
            vst1q_f32(y3.add(c + 4), b3);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn update4(y: [&mut [f32]; 4], u: &[f32], r: usize, bt: &[f32], stride: usize, w8: usize) {
    let [y0, y1, y2, y3] = y;
    for c in (0..w8).step_by(8) {
        let ld = |y: &[f32]| -> [f32; 8] { *y[c..].first_chunk::<8>().unwrap() };
        let (mut a0, mut a1, mut a2, mut a3) = (ld(y0), ld(y1), ld(y2), ld(y3));
        for k in 0..r {
            let b = bt[k * stride + c..].first_chunk::<8>().unwrap();
            let (u0, u1, u2, u3) = (u[k], u[r + k], u[2 * r + k], u[3 * r + k]);
            for l in 0..8 {
                a0[l] += u0 * b[l];
                a1[l] += u1 * b[l];
                a2[l] += u2 * b[l];
                a3[l] += u3 * b[l];
            }
        }
        y0[c..c + 8].copy_from_slice(&a0);
        y1[c..c + 8].copy_from_slice(&a1);
        y2[c..c + 8].copy_from_slice(&a2);
        y3[c..c + 8].copy_from_slice(&a3);
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// One module of the adapter file, before it is attached.
struct Pair {
    a: Vec<f32>, // [r, n_in]
    b: Vec<f32>, // [n_out, r]
    r: usize,
    n_in: usize,
    n_out: usize,
}

/// A PEFT LoRA adapter, loaded and waiting to be attached to a model ([`LoraAdapter::take`] by
/// module, then [`LoraAdapter::finish`]).
pub struct LoraAdapter {
    pub dir: PathBuf,
    /// Rank from the config (zero-padded components are dropped per module on load).
    pub rank: usize,
    pub alpha: f32,
    /// `alpha / r` (or `alpha / sqrt(r)` with rsLoRA).
    pub scale: f32,
    pub target_modules: Vec<String>,
    modules: BTreeMap<String, Pair>,
    active: Arc<AtomicBool>,
    attached: usize,
    bytes: usize,
}

/// Whether `dir` holds a PEFT adapter.
pub fn is_adapter_dir(dir: &Path) -> bool {
    dir.join("adapter_config.json").is_file() && dir.join("adapter_model.safetensors").is_file()
}

/// The module path of a PEFT key, from `layers.` on, or `lm_head`; None for keys of other parts
/// (vision towers, MTP layers, embeddings).
fn canonical(module: &str) -> Option<String> {
    let m = module.strip_prefix("base_model.model.").unwrap_or(module);
    if m == "lm_head" || m.ends_with(".lm_head") && !m.contains("layers.") {
        return Some("lm_head".into());
    }
    for p in ["model.language_model.", "language_model.model.", "language_model.", "model.", ""] {
        if let Some(rest) = m.strip_prefix(p)
            && rest.starts_with("layers.")
        {
            return Some(rest.to_string());
        }
    }
    None
}

/// A float tensor of the adapter file as f32.
fn tensor_f32(st: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let info = st.info(name).with_context(|| format!("missing tensor {name}"))?;
    let b = st.bytes(name)?;
    let v = match info.dtype.as_str() {
        "F32" => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        "BF16" => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| crate::kernels::bf16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        "F16" => b.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect(),
        d => bail!("LoRA tensor {name} has dtype {d}, expected a float type"),
    };
    Ok((v, info.shape.clone()))
}

impl LoraAdapter {
    /// Loads the adapter in `dir` (all tensors as f32).
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg_path = dir.join("adapter_config.json");
        let cfg: serde_json::Value =
            serde_json::from_slice(&crate::fsx::read(&cfg_path)?).with_context(|| format!("parse {}", cfg_path.display()))?;
        let get_str = |k: &str| cfg.get(k).and_then(|v| v.as_str()).unwrap_or_default().to_string();
        let peft_type = get_str("peft_type");
        if !peft_type.is_empty() && peft_type != "LORA" {
            bail!("{}: peft_type {peft_type} (only LORA adapters are supported)", cfg_path.display());
        }
        let flag = |k: &str| cfg.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
        for k in ["use_dora", "fan_in_fan_out", "use_qalora", "lora_bias"] {
            if flag(k) {
                bail!("{}: {k} is not supported", cfg_path.display());
            }
        }
        if !matches!(get_str("bias").as_str(), "" | "none") {
            bail!("{}: bias {:?} is not supported", cfg_path.display(), get_str("bias"));
        }
        for k in ["rank_pattern", "alpha_pattern"] {
            if cfg.get(k).and_then(|v| v.as_object()).is_some_and(|o| !o.is_empty()) {
                bail!("{}: per-module {k} is not supported", cfg_path.display());
            }
        }
        if cfg
            .get("modules_to_save")
            .is_some_and(|v| v.as_array().is_some_and(|a| !a.is_empty()))
        {
            bail!("{}: modules_to_save (fully trained modules) is not supported", cfg_path.display());
        }
        let rank = cfg.get("r").and_then(|v| v.as_u64()).context("adapter_config.json: missing r")? as usize;
        let alpha = cfg.get("lora_alpha").and_then(|v| v.as_f64()).unwrap_or(rank as f64) as f32;
        let scale = if flag("use_rslora") {
            alpha / (rank as f32).sqrt()
        } else {
            alpha / rank as f32
        };
        let target_modules = match cfg.get("target_modules") {
            Some(serde_json::Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            _ => Vec::new(),
        };

        let path = dir.join("adapter_model.safetensors");
        let st = SafeTensors::open(&path, true)?;
        let mut halves: BTreeMap<String, (Option<String>, Option<String>)> = BTreeMap::new();
        for name in st.names() {
            let (module, is_a) = if let Some(m) = name
                .strip_suffix(".lora_A.weight")
                .or_else(|| name.strip_suffix(".lora_A.default.weight"))
            {
                (m, true)
            } else if let Some(m) = name
                .strip_suffix(".lora_B.weight")
                .or_else(|| name.strip_suffix(".lora_B.default.weight"))
            {
                (m, false)
            } else {
                bail!("{}: unsupported tensor {name} (expected lora_A / lora_B weights)", path.display());
            };
            let e = halves.entry(module.to_string()).or_default();
            if is_a {
                e.0 = Some(name.clone());
            } else {
                e.1 = Some(name.clone());
            }
        }
        let mut modules = BTreeMap::new();
        let mut bytes = 0;
        for (module, (an, bn)) in halves {
            let (Some(an), Some(bn)) = (an, bn) else {
                bail!("{}: {module} lacks its lora_A or lora_B weight", path.display());
            };
            let key =
                canonical(&module).with_context(|| format!("{}: module {module} is not a decoder layer or lm_head", path.display()))?;
            let (a, ash) = tensor_f32(&st, &an)?;
            let (b, bsh) = tensor_f32(&st, &bn)?;
            if ash.len() != 2 || bsh.len() != 2 || ash[0] != bsh[1] {
                bail!("{module}: lora_A {ash:?} and lora_B {bsh:?} do not form a low-rank pair");
            }
            let (r0, n_in, n_out) = (ash[0], ash[1], bsh[0]);
            // Drop components that cannot contribute (adapters zero-padded to a common rank).
            let keep: Vec<usize> = (0..r0)
                .filter(|&k| a[k * n_in..(k + 1) * n_in].iter().any(|&v| v != 0.0) && (0..n_out).any(|o| b[o * r0 + k] != 0.0))
                .collect();
            let r = keep.len();
            let mut ak = Vec::with_capacity(r * n_in);
            for &k in &keep {
                ak.extend_from_slice(&a[k * n_in..(k + 1) * n_in]);
            }
            let mut bk = Vec::with_capacity(n_out * r);
            for o in 0..n_out {
                bk.extend(keep.iter().map(|&k| b[o * r0 + k]));
            }
            bytes += (ak.len() + bk.len()) * 4;
            if modules
                .insert(
                    key.clone(),
                    Pair {
                        a: ak,
                        b: bk,
                        r,
                        n_in,
                        n_out,
                    },
                )
                .is_some()
            {
                bail!("{}: module {key} appears twice", path.display());
            }
        }
        if modules.is_empty() {
            bail!("{}: no LoRA modules", path.display());
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            rank,
            alpha,
            scale,
            target_modules,
            modules,
            active: Arc::new(AtomicBool::new(false)),
            attached: 0,
            bytes,
        })
    }

    /// The switch shared by every attached module (off at load).
    pub fn switch(&self) -> Arc<AtomicBool> {
        self.active.clone()
    }

    /// The largest `layers.N` index the adapter touches, plus one.
    pub fn layer_count(&self) -> usize {
        self.modules
            .keys()
            .filter_map(|k| k.strip_prefix("layers.")?.split('.').next()?.parse::<usize>().ok())
            .max()
            .map_or(0, |l| l + 1)
    }

    /// Whether the adapter has a module `name` (`layers.N.mlp.down_proj`, `lm_head`).
    pub fn has(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    /// Takes the update of module `name` for a layer of `n_out x n_in`, or None when the adapter
    /// does not target it. `rows[p]` / `cols[p]`, when given, name the adapter's row / column
    /// that the model stores at `p` (checkpoints that reorder heads, e.g. GGUF).
    pub fn take(
        &mut self,
        name: &str,
        n_out: usize,
        n_in: usize,
        rows: Option<&[usize]>,
        cols: Option<&[usize]>,
    ) -> Result<Option<Box<LoraDelta>>> {
        let Some(p) = self.modules.remove(name) else {
            return Ok(None);
        };
        if p.n_out != n_out || p.n_in != n_in {
            bail!(
                "LoRA module {name} is [{} x {}] but the model's layer is [{n_out} x {n_in}]: the adapter is for another base model",
                p.n_out,
                p.n_in
            );
        }
        let r = p.r;
        let mut a = p.a;
        if let Some(cols) = cols {
            let src = a.clone();
            for k in 0..r {
                for (dst, &c) in cols.iter().enumerate() {
                    a[k * n_in + dst] = src[k * n_in + c];
                }
            }
        }
        let mut bt = vec![0f32; r * n_out];
        for o in 0..n_out {
            let src = rows.map_or(o, |rs| rs[o]);
            for k in 0..r {
                bt[k * n_out + o] = p.b[src * r + k];
            }
        }
        self.attached += 1;
        Ok(Some(Box::new(LoraDelta {
            a,
            bt,
            r,
            n_in,
            n_out,
            scale: self.scale,
            active: self.active.clone(),
        })))
    }

    /// Checks that every module found a layer; returns the number attached.
    pub fn finish(&self) -> Result<usize> {
        if let Some(k) = self.modules.keys().next() {
            bail!(
                "LoRA adapter {}: {} module(s) match no layer of the model (first: {k})",
                self.dir.display(),
                self.modules.len()
            );
        }
        Ok(self.attached)
    }

    /// Parameter bytes (f32) of the loaded modules.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Turns an adapter switch on for the guard's lifetime.
pub struct Active(Arc<AtomicBool>);

impl Active {
    pub fn new(switch: &Arc<AtomicBool>) -> Self {
        switch.store(true, Ordering::SeqCst);
        Self(switch.clone())
    }
}

impl Drop for Active {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Row order of a Gated DeltaNet projection whose value heads a GGUF file stores tiled: entry `p`
/// is the Hugging Face row stored at row `p`. The first `head_rows` rows (query and key heads of
/// `in_proj_qkv`) keep their place; then `nv` value heads of `hd` rows follow, stored head `p`
/// being HF head `(p % nk) * (nv / nk) + p / nk`.
pub fn tiled_v_rows(head_rows: usize, nk: usize, nv: usize, hd: usize) -> Vec<usize> {
    let rep = nv / nk;
    let mut v: Vec<usize> = (0..head_rows).collect();
    for p in 0..nv {
        let hf = (p % nk) * rep + p / nk;
        v.extend((0..hd).map(|d| head_rows + hf * hd + d));
    }
    v
}

/// Row order of a Llama query / key projection in a GGUF file (llama.cpp's `LlamaModel.permute`):
/// entry `p` is the Hugging Face row stored at row `p`; stored row `2i + j` of a head of `hd` rows
/// is row `j * hd/2 + i` of that head.
pub fn rope_pair_rows(n_heads: usize, hd: usize) -> Vec<usize> {
    (0..n_heads * hd)
        .map(|p| {
            let (head, r) = (p / hd, p % hd);
            head * hd + (r % 2) * (hd / 2) + r / 2
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// llama.cpp's `permute` (`reshape(n_head, 2, hd/2).swapaxes(1, 2)`), and the equivalence the
    /// Llama GGUF path relies on: adjacent-pair RoPE on the stored rows is rotate-half RoPE on the
    /// Hugging Face rows, in the stored order.
    #[test]
    fn rope_pair_rows_match_llama_cpp_permute() {
        use crate::kernels::{apply_rope, apply_rope_interleaved};
        let (nh, hd) = (3, 8);
        let rows = rope_pair_rows(nh, hd);
        for h in 0..nh {
            for i in 0..hd / 2 {
                for j in 0..2 {
                    assert_eq!(rows[h * hd + i * 2 + j], h * hd + j * hd / 2 + i);
                }
            }
        }
        let x: Vec<f32> = (0..nh * hd).map(|i| (i as f32 * 0.37).sin()).collect();
        let cos: Vec<f32> = (0..hd / 2).map(|i| (1.0 + i as f32).cos()).collect();
        let sin: Vec<f32> = (0..hd / 2).map(|i| (1.0 + i as f32).sin()).collect();
        let mut hf = x.clone();
        let mut stored: Vec<f32> = rows.iter().map(|&r| x[r]).collect();
        for h in 0..nh {
            apply_rope(&mut hf[h * hd..(h + 1) * hd], &cos, &sin);
            apply_rope_interleaved(&mut stored[h * hd..(h + 1) * hd], &cos, &sin);
        }
        for (p, &r) in rows.iter().enumerate() {
            assert_eq!(stored[p], hf[r]);
        }
    }

    #[test]
    fn canonical_names() {
        assert_eq!(
            canonical("base_model.model.model.layers.3.mlp.down_proj").as_deref(),
            Some("layers.3.mlp.down_proj")
        );
        assert_eq!(
            canonical("base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv").as_deref(),
            Some("layers.0.linear_attn.in_proj_qkv")
        );
        assert_eq!(canonical("base_model.model.lm_head").as_deref(), Some("lm_head"));
        assert_eq!(canonical("base_model.model.mtp.layers.0.mlp.up_proj"), None);
        assert_eq!(canonical("base_model.model.model.visual.blocks.0.attn.qkv"), None);
    }

    #[test]
    fn delta_matches_reference() {
        let (r, n_in, n_out, t) = (3, 37, 1100, 21);
        let mut s = 1u32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let a: Vec<f32> = (0..r * n_in).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..n_out * r).map(|_| rnd()).collect();
        let x: Vec<f32> = (0..t * n_in).map(|_| rnd()).collect();
        let mut ad = LoraAdapter {
            dir: PathBuf::new(),
            rank: r,
            alpha: 2.0 * r as f32,
            scale: 2.0,
            target_modules: Vec::new(),
            modules: BTreeMap::from([(
                "m".to_string(),
                Pair {
                    a: a.clone(),
                    b: b.clone(),
                    r,
                    n_in,
                    n_out,
                },
            )]),
            active: Arc::new(AtomicBool::new(true)),
            attached: 0,
            bytes: 0,
        };
        let d = ad.take("m", n_out, n_in, None, None).unwrap().unwrap();
        assert!(ad.finish().is_ok());
        let mut y = vec![1f32; t * n_out];
        d.apply(&x, t, &mut y);
        for i in 0..t {
            for o in 0..n_out {
                let mut acc = 0f64;
                for k in 0..r {
                    let ax: f64 = (0..n_in).map(|c| a[k * n_in + c] as f64 * x[i * n_in + c] as f64).sum();
                    acc += b[o * r + k] as f64 * ax;
                }
                let want = 1.0 + 2.0 * acc;
                assert!((y[i * n_out + o] as f64 - want).abs() < 1e-4, "{i} {o}");
            }
        }
    }

    /// `cargo test --release --lib lora::tests::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench() {
        for (r, n_in, n_out, t) in [
            (16, 5120, 17408, 1024),
            (16, 17408, 8, 1024),
            (16, 8, 17408, 1024),
            (16, 5120, 17408, 1),
        ] {
            let d = LoraDelta {
                a: vec![0.01; r * n_in],
                bt: vec![0.02; r * n_out],
                r,
                n_in,
                n_out,
                scale: 2.0,
                active: Arc::new(AtomicBool::new(true)),
            };
            let x = vec![0.5f32; t * n_in];
            let mut y = vec![0f32; t * n_out];
            d.apply(&x, t, &mut y);
            let t0 = std::time::Instant::now();
            for _ in 0..5 {
                d.apply(&x, t, &mut y);
            }
            let s = t0.elapsed().as_secs_f64() / 5.0;
            let flops = 2.0 * (r * (n_in + n_out) * t) as f64;
            eprintln!("{r} {n_in} {n_out} {t}: {:.3} ms, {:.0} GFLOP/s", s * 1e3, flops / s / 1e9);
        }
    }

    #[test]
    fn tiled_rows() {
        // 2 key heads, 4 value heads of 1 row: stored heads are HF heads 0, 2, 1, 3.
        assert_eq!(tiled_v_rows(2, 2, 4, 1), vec![0, 1, 2, 4, 3, 5]);
    }
}
