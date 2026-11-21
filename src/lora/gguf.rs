//! llama.cpp LoRA adapters: GGUF files with `general.type = adapter`, `adapter.type = lora` and
//! `adapter.lora.alpha`, as `convert_lora_to_gguf.py` writes them and `llama-cli --lora FILE` /
//! `--lora-scaled FILE SCALE` reads them (`src/llama-adapter.cpp`, `llm_graph_context::
//! build_lora_mm` / `build_lora_mm_id` / `build_inp_embd`).
//!
//! Tensors are named after the base file's (`blk.N.attn_output.weight.lora_a` / `.lora_b`), in
//! ggml's dimension order (`ne`, fastest first):
//! - a linear layer `W` (`ne [n_in, n_out]`): `lora_a` `ne [n_in, r]` and `lora_b` `ne [r, n_out]`,
//!   i.e. row-major PEFT `A` `[r, n_in]` and `B` `[n_out, r]`, with `y += s * B (A x)`;
//! - a stack of experts (`ne [n_in, n_out, n_expert]`, `ffn_down_exps`, `ffn_gate_up_exps`):
//!   `lora_a` `ne [n_in, r, n_expert]` and `lora_b` `ne [r, n_out, n_expert]`, one pair per expert
//!   applied to the tokens routed to it (`ggml_mul_mat_id` on both);
//! - the token embedding (`token_embd.weight`, `ne [hidden, vocab]`): `lora_a` `ne [r, vocab]` and
//!   `lora_b` `ne [r, hidden]`, A and B "flipped": token `t`'s row gets `s * B a_t` with `a_t`
//!   row `t` of `lora_a` (llama.cpp's `get_rows` of `lora_a`).
//!
//! The strength is `s = alpha / r * scale` with `r` each pair's `lora_b` `ne[0]` and `scale` the
//! user's (`--lora-scaled`), or just `scale` when `alpha` is 0 or absent; `alpha / r` is the
//! adapter's scale (folded into each `B` when the pairs have different ranks), the user's scale
//! its strength. Norm vectors (`*_norm.weight`,
//! which `mergekit-extract-lora` adds) are skipped as llama.cpp skips them; any other tensor is
//! an error, as are activated LoRAs (`adapter.alora.invocation_tokens`).
//!
//! The converter runs the adapter through the same tensor transforms as the base model (the
//! Llama query / key row permutation, Qwen3.5's tiled value heads, ...), so the pairs are already
//! in the base GGUF file's layout: [`LoraAdapter::take`] then leaves rows and columns where they
//! are. Names are mapped to the Hugging Face module paths the models attach by through the same
//! tables the GGUF loader uses ([`crate::gguf::hf_tensor`]); `output.weight` is `lm_head`.

use super::{EmbedPair, LoraAdapter, Pair, canonical};
use crate::ggml::{GgmlType, dequantize_row};
use crate::gguf::{Header, Tensor, Value};
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::Path;

/// A tensor of the adapter file as f32 (row-major in ggml's order: `ne[0]` fastest).
fn tensor_f32(bytes: &[u8], t: &Tensor) -> Result<Vec<f32>> {
    let kind = GgmlType::from_id(t.ggml_type).with_context(|| format!("{}: unknown ggml type {}", t.name, t.ggml_type))?;
    let row = t.ne.first().copied().unwrap_or(1);
    let n: usize = t.ne.iter().product();
    let rb = kind.row_bytes(row);
    let data = bytes
        .get(t.start..t.end)
        .with_context(|| format!("{}: data outside the file", t.name))?;
    if data.len() != n / row.max(1) * rb {
        bail!("{}: {} bytes for {:?} of {kind}", t.name, data.len(), t.ne);
    }
    let mut out = vec![0f32; n];
    for (o, src) in out.chunks_mut(row).zip(data.chunks(rb)) {
        dequantize_row(kind, src, o);
    }
    Ok(out)
}

/// The strength factor `alpha / r` of a pair of rank `r` (1 without an alpha).
fn pair_scale(alpha: f32, r: usize) -> f32 {
    if alpha != 0.0 { alpha / r as f32 } else { 1.0 }
}

/// The module path (`layers.N.self_attn.o_proj`, `layers.N.mlp.experts.down_proj`, `lm_head`)
/// of base tensor `name` of shape `ne` in a GGUF file of architecture `arch`.
fn module_of(arch: &str, name: &str, ne: &[usize], n_layers: usize) -> Option<String> {
    if name == "output.weight" {
        return Some("lm_head".into());
    }
    let t = Tensor {
        name: name.to_string(),
        ne: ne.to_vec(),
        ggml_type: 0,
        start: 0,
        end: 0,
    };
    let (hf, _) = crate::gguf::hf_tensor(arch, &t, n_layers)?;
    canonical(hf.strip_suffix(".weight").unwrap_or(&hf)).filter(|k| k != "lm_head" || name == "output.weight")
}

/// Loads the adapter at `path` for the base model of GGUF header `base` (see the module docs).
pub(super) fn load(path: &Path, base: &Header) -> Result<LoraAdapter> {
    let bytes = crate::fsx::read(path)?;
    let h = crate::gguf::parse_header(&bytes[..], bytes.len()).with_context(|| format!("{}", path.display()))?;
    let s = |k: &str| h.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    if s("general.type") != "adapter" {
        bail!(
            "{}: general.type is {:?}, not \"adapter\" (a model file, not a LoRA adapter?)",
            path.display(),
            s("general.type")
        );
    }
    if s("adapter.type") != "lora" {
        bail!(
            "{}: adapter.type is {:?}; only LoRA adapters are supported (not control vectors)",
            path.display(),
            s("adapter.type")
        );
    }
    if h.get("adapter.alora.invocation_tokens").is_some() {
        bail!(
            "{}: an activated LoRA (adapter.alora.invocation_tokens) is not supported",
            path.display()
        );
    }
    let (arch, base_arch) = (h.arch(), base.arch());
    if arch != base_arch {
        bail!(
            "{}: the adapter is for architecture {arch:?}, the model is {base_arch:?} (llama.cpp refuses this too)",
            path.display()
        );
    }
    // (llama.cpp reads `adapter.lora.alpha` only: a file that spells it `adapter.lora_alpha`, as
    // AtomicChat's Inkling-2 adapter does, runs without an alpha there, and here.)
    let alpha = h.get("adapter.lora.alpha").and_then(Value::as_f64).unwrap_or(0.0) as f32;
    let n_layers = base.n_layers();

    let mut halves: BTreeMap<&str, (Option<&Tensor>, Option<&Tensor>)> = BTreeMap::new();
    let mut skipped = Vec::new();
    for t in &h.tensors {
        if let Some(b) = t.name.strip_suffix(".lora_a") {
            halves.entry(b).or_default().0 = Some(t);
        } else if let Some(b) = t.name.strip_suffix(".lora_b") {
            halves.entry(b).or_default().1 = Some(t);
        } else if t.name.ends_with("_norm.weight") {
            skipped.push(t.name.clone());
        } else {
            bail!("{}: tensor {} is not a lora_a / lora_b tensor", path.display(), t.name);
        }
    }
    // One rank for every pair (the usual case): `alpha / r` is the adapter's scale, applied as
    // the PEFT path applies it (the same rounding). Mixed ranks: each pair's is folded into its B.
    let ranks: Vec<usize> = halves.values().filter_map(|(_, b)| b.and_then(|b| b.ne.first().copied())).collect();
    let uniform = ranks.windows(2).all(|w| w[0] == w[1]);
    let scale = match (uniform, ranks.first()) {
        (true, Some(&r)) => pair_scale(alpha, r),
        _ => 1.0,
    };
    let fold = |r: usize| if uniform { 1.0 } else { pair_scale(alpha, r) };
    let mut ad = LoraAdapter::empty(path, "gguf", 0, alpha, scale);
    ad.stored_order = true;
    ad.skipped = skipped;
    let mut bytes_held = 0;
    for (name, (ta, tb)) in halves {
        let (Some(ta), Some(tb)) = (ta, tb) else {
            bail!("{}: {name} lacks its lora_a or lora_b tensor", path.display());
        };
        let (a, mut b) = (tensor_f32(&bytes, ta)?, tensor_f32(&bytes, tb)?);
        let (an, bn) = (&ta.ne, &tb.ne);
        if name == "token_embd.weight" {
            if an.len() != 2 || bn.len() != 2 || an[0] != bn[0] {
                bail!(
                    "{}: {name}: lora_a {an:?} and lora_b {bn:?} do not form an embedding pair",
                    path.display()
                );
            }
            let (r, vocab, hidden) = (bn[0], an[1], bn[1]);
            let sc = fold(r);
            b.iter_mut().for_each(|v| *v *= sc);
            ad.rank = ad.rank.max(r);
            bytes_held += (a.len() + b.len()) * 4;
            ad.embed = Some(EmbedPair { a, b, r, vocab, hidden });
            continue;
        }
        match (an.len(), bn.len()) {
            (2, 2) if an[1] == bn[0] => {
                let (n_in, r, n_out) = (an[0], an[1], bn[1]);
                let key = module_of(arch, name, &[n_in, n_out], n_layers)
                    .with_context(|| format!("{}: {name} is not a layer oxinfer adapts", path.display()))?;
                let sc = fold(r);
                b.iter_mut().for_each(|v| *v *= sc);
                bytes_held += (a.len() + b.len()) * 4;
                ad.rank = ad.rank.max(r);
                if ad.modules.insert(key.clone(), Pair { a, b, r, n_in, n_out }).is_some() {
                    bail!("{}: module {key} appears twice", path.display());
                }
            }
            (3, 3) if an[1] == bn[0] && an[2] == bn[2] => {
                let (n_in, r, n_out, ne) = (an[0], an[1], bn[1], an[2]);
                let key = module_of(arch, name, &[n_in, n_out, ne], n_layers)
                    .with_context(|| format!("{}: {name} is not an expert stack oxinfer adapts", path.display()))?;
                let (stack, param) = key
                    .rsplit_once('.')
                    .with_context(|| format!("{}: {name} maps to {key}, not an expert stack", path.display()))?;
                let sc = fold(r);
                ad.rank = ad.rank.max(r);
                let (pa, pb) = (r * n_in, n_out * r);
                for e in 0..ne {
                    let ae = a[e * pa..(e + 1) * pa].to_vec();
                    let be: Vec<f32> = b[e * pb..(e + 1) * pb].iter().map(|v| v * sc).collect();
                    bytes_held += (ae.len() + be.len()) * 4;
                    let k = format!("{stack}.{e}.{param}");
                    if ad
                        .modules
                        .insert(
                            k.clone(),
                            Pair {
                                a: ae,
                                b: be,
                                r,
                                n_in,
                                n_out,
                            },
                        )
                        .is_some()
                    {
                        bail!("{}: module {k} appears twice", path.display());
                    }
                }
            }
            _ => bail!(
                "{}: {name}: lora_a {an:?} and lora_b {bn:?} do not form a low-rank pair (lora_a must be transposed: \
                 adapters of llama.cpp's old finetune example are not supported, as in llama.cpp)",
                path.display()
            ),
        }
    }
    if ad.modules.is_empty() && ad.embed.is_none() {
        bail!("{}: no LoRA tensors", path.display());
    }
    ad.bytes = bytes_held;
    Ok(ad)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GGUF file with string / f32 metadata and F32 tensors (`ne` order).
    fn gguf_file(meta: &[(&str, Value)], tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        fn put_str(b: &mut Vec<u8>, s: &str) {
            b.extend((s.len() as u64).to_le_bytes());
            b.extend(s.as_bytes());
        }
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend((tensors.len() as u64).to_le_bytes());
        b.extend((meta.len() as u64).to_le_bytes());
        for (k, v) in meta {
            put_str(&mut b, k);
            match v {
                Value::Str(s) => {
                    b.extend(8u32.to_le_bytes());
                    put_str(&mut b, s);
                }
                Value::F(f) => {
                    b.extend(6u32.to_le_bytes());
                    b.extend((*f as f32).to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
        let mut off = 0usize;
        for (name, ne, data) in tensors {
            put_str(&mut b, name);
            b.extend((ne.len() as u32).to_le_bytes());
            for &n in ne {
                b.extend((n as u64).to_le_bytes());
            }
            b.extend(0u32.to_le_bytes());
            b.extend((off as u64).to_le_bytes());
            off += (data.len() * 4).next_multiple_of(32);
        }
        b.resize(b.len().next_multiple_of(32), 0);
        for (_, _, data) in tensors {
            let start = b.len();
            b.extend(data.iter().flat_map(|v| v.to_le_bytes()));
            b.resize(start + (data.len() * 4).next_multiple_of(32), 0);
        }
        b
    }

    fn base(arch: &str) -> Header {
        Header {
            meta: BTreeMap::from([
                ("general.architecture".to_string(), Value::Str(arch.into())),
                (format!("{arch}.block_count"), Value::U(2)),
            ]),
            tensors: Vec::new(),
        }
    }

    fn seq(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| i as f32 * k + 0.25).collect()
    }

    /// Names, layouts (linear, expert stack, embedding), alpha and skipped norms of a llama.cpp
    /// adapter, read as `src/llama-adapter.cpp` reads them.
    #[test]
    fn reads_llama_cpp_layout() {
        let meta = [
            ("general.architecture", Value::Str("llama".into())),
            ("general.type", Value::Str("adapter".into())),
            ("adapter.type", Value::Str("lora".into())),
            ("adapter.lora.alpha", Value::F(3.0)),
        ];
        let tensors = [
            ("blk.1.attn_output.weight.lora_a", vec![4, 2], seq(8, 1.0)),
            ("blk.1.attn_output.weight.lora_b", vec![2, 3], seq(6, -1.0)),
            ("blk.0.ffn_down_exps.weight.lora_a", vec![4, 2, 2], seq(16, 0.5)),
            ("blk.0.ffn_down_exps.weight.lora_b", vec![2, 3, 2], seq(12, 2.0)),
            ("token_embd.weight.lora_a", vec![2, 5], seq(10, 0.1)),
            ("token_embd.weight.lora_b", vec![2, 3], seq(6, 0.2)),
            ("blk.0.ffn_norm.weight", vec![3], seq(3, 1.0)),
        ];
        let dir = std::env::temp_dir().join(format!("oxinfer-lora-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("adapter.gguf");
        std::fs::write(&path, gguf_file(&meta, &tensors)).unwrap();
        let ad = load(&path, &base("llama"));
        let other = load(&path, &base("qwen3"));
        std::fs::remove_dir_all(&dir).ok();
        let mut ad = ad.unwrap();
        assert!(other.is_err(), "another architecture");
        assert_eq!((ad.format, ad.rank, ad.alpha, ad.scale), ("gguf", 2, 3.0, 1.5));
        assert!(ad.stored_order);
        assert_eq!(ad.skipped, ["blk.0.ffn_norm.weight"]);
        let o = &ad.modules["layers.1.self_attn.o_proj"];
        assert_eq!((o.r, o.n_in, o.n_out), (2, 4, 3));
        assert_eq!((o.a.clone(), o.b.clone()), (seq(8, 1.0), seq(6, -1.0)));
        // Expert 1: the second [r, n_in] / [n_out, r] slices.
        let e = &ad.modules["layers.0.mlp.experts.1.down_proj"];
        assert_eq!((e.r, e.n_in, e.n_out), (2, 4, 3));
        assert_eq!(e.a, seq(16, 0.5)[8..]);
        assert_eq!(e.b, seq(12, 2.0)[6..]);
        assert!(ad.modules.contains_key("layers.0.mlp.experts.0.down_proj"));
        assert_eq!(ad.modules.len(), 3);
        // The embedding: row t of lora_a is token t's coefficients; B is [hidden, r].
        ad.set_strength(2.0);
        let d = ad.take_embed(5, 3).unwrap().unwrap();
        let mut row = vec![0f32; 3];
        d.add_row(4, &mut row);
        let (a, b) = (seq(10, 0.1), seq(6, 0.2));
        for (h, v) in row.iter().enumerate() {
            let want = 2.0 * 1.5 * (0..2).map(|k| b[h * 2 + k] * a[4 * 2 + k]).sum::<f32>();
            assert!((v - want).abs() < 1e-4 * (1.0 + want.abs()), "{h}: {v} vs {want}");
        }
    }

    /// MTP-layer tensors (past the main layers, `blk.N.nextn.*`) map to the MTP layer's modules.
    #[test]
    fn maps_mtp_tensors() {
        let mut h = base("qwen35");
        h.meta.insert("qwen35.block_count".into(), Value::U(3));
        h.meta.insert("qwen35.nextn_predict_layers".into(), Value::U(1));
        let n = h.n_layers();
        assert_eq!(n, 2);
        assert_eq!(
            module_of("qwen35", "blk.2.nextn.eh_proj.weight", &[64, 32], n).as_deref(),
            Some("mtp.fc")
        );
        assert_eq!(
            module_of("qwen35", "blk.2.attn_output.weight", &[32, 32], n).as_deref(),
            Some("mtp.layers.0.self_attn.o_proj")
        );
        assert_eq!(
            module_of("qwen35", "blk.1.attn_output.weight", &[32, 32], n).as_deref(),
            Some("layers.1.self_attn.o_proj")
        );
        let mut ad = LoraAdapter::empty(Path::new("x"), "gguf", 1, 0.0, 1.0);
        for k in ["mtp.fc", "layers.0.mlp.down_proj"] {
            let p = Pair {
                a: vec![1.0; 4],
                b: vec![1.0; 4],
                r: 1,
                n_in: 4,
                n_out: 4,
            };
            ad.modules.insert(k.into(), p);
        }
        let m = ad.split_mtp().unwrap();
        assert_eq!(m.modules.keys().collect::<Vec<_>>(), ["mtp.fc"]);
        assert_eq!(ad.modules.keys().collect::<Vec<_>>(), ["layers.0.mlp.down_proj"]);
    }

    /// `lora` options: repeated keys add, a JSON array adds them all, `none` clears.
    #[test]
    fn engine_options() {
        let mut o = crate::engine::EngineOptions::default();
        o.set("lora", "a.gguf:0.5").unwrap();
        o.set("lora", "org/name").unwrap();
        o.set("lora", r#"["b", "c:2"]"#).unwrap();
        let got: Vec<(String, f32)> = o.loras.iter().map(|l| (l.path.display().to_string(), l.scale)).collect();
        let want = [("a.gguf", 0.5), ("org/name", 1.0), ("b", 1.0), ("c", 2.0)].map(|(p, s)| (p.to_string(), s));
        assert_eq!(got, want);
        o.set("lora_merge", "1").unwrap();
        assert!(o.lora_merge);
        o.set("lora", "none").unwrap();
        assert!(o.loras.is_empty());
    }
}
