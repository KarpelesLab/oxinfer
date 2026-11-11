//! CLIP text side of Stable Diffusion: the byte-level BPE tokenizer (`tokenizer/vocab.json` +
//! `merges.txt`, as `transformers.CLIPTokenizer`) and the text transformer (`text_encoder/`,
//! `CLIPTextModel`: pre-LN blocks, causal self-attention, final layer norm). SD 1.x uses OpenAI
//! CLIP ViT-L/14 (768 wide, quick_gelu); SD 2.x and SD-Turbo use OpenCLIP ViT-H (1024, gelu); SDXL
//! uses both ViT-L and OpenCLIP bigG (1280, gelu, with its text projection: see `xl.rs`).

use super::ops::{LayerNorm, Linear, Tensors, attention, gelu_erf, quick_gelu};
use crate::kernels::{add_inplace, for_chunks};
use anyhow::{Context, Result, bail};
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;

pub struct ClipTokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    byte_enc: [char; 256],
    pat: Regex,
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    pub max_len: usize,
}

/// GPT-2's reversible byte -> printable-character map.
fn bytes_to_unicode() -> [char; 256] {
    let mut printable: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut chars: Vec<u32> = printable.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !printable.contains(&b) {
            printable.push(b);
            chars.push(256 + n);
            n += 1;
        }
    }
    let mut map = ['\0'; 256];
    for (b, c) in printable.iter().zip(chars) {
        map[*b as usize] = char::from_u32(c).unwrap();
    }
    map
}

impl ClipTokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let read = |n: &str| crate::fsx::read(&dir.join(n)).with_context(|| format!("tokenizer/{n}"));
        let vocab: HashMap<String, u32> = serde_json::from_slice(&read("vocab.json")?).context("vocab.json")?;
        let merges = String::from_utf8(read("merges.txt")?).context("merges.txt")?;
        let mut ranks = HashMap::new();
        for line in merges.lines().filter(|l| !l.starts_with("#version") && !l.trim().is_empty()) {
            let mut it = line.split_whitespace();
            if let (Some(a), Some(b)) = (it.next(), it.next()) {
                let r = ranks.len();
                ranks.entry((a.to_string(), b.to_string())).or_insert(r);
            }
        }
        let cfg: serde_json::Value = read("tokenizer_config.json")
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let tok_str = |v: &serde_json::Value| v.as_str().map(str::to_string).or_else(|| v["content"].as_str().map(str::to_string));
        let id = |s: &str| vocab.get(s).copied().with_context(|| format!("token {s:?} not in vocab"));
        let bos = id(&tok_str(&cfg["bos_token"]).unwrap_or_else(|| "<|startoftext|>".into()))?;
        let eos = id(&tok_str(&cfg["eos_token"]).unwrap_or_else(|| "<|endoftext|>".into()))?;
        // SD 1.x pads with <|endoftext|>, SD 2.x with "!".
        let pad = match tok_str(&cfg["pad_token"]) {
            Some(p) => id(&p)?,
            None => eos,
        };
        let max_len = cfg["model_max_length"].as_u64().filter(|&n| n <= 1024).unwrap_or(77) as usize;
        let pat = Regex::new(r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+")?;
        Ok(Self {
            vocab,
            ranks,
            byte_enc: bytes_to_unicode(),
            pat,
            bos,
            eos,
            pad,
            max_len,
        })
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        loop {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..word.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(word[i].clone(), word[i + 1].clone()))
                    && best.is_none_or(|(br, _)| r < br)
                {
                    best = Some((r, i));
                }
            }
            let Some((_, i0)) = best else { break };
            let (a, b) = (word[i0].clone(), word[i0 + 1].clone());
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
                    merged.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
        }
        word
    }

    /// Token ids for `text`, framed as `<bos> ... <eos>` and padded to `max_len`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let clean = text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let mut ids = vec![self.bos];
        for m in self.pat.find_iter(&clean) {
            let mapped: String = m.as_str().bytes().map(|b| self.byte_enc[b as usize]).collect();
            for piece in self.bpe(&mapped) {
                if let Some(&id) = self.vocab.get(&piece) {
                    ids.push(id);
                }
            }
        }
        ids.truncate(self.max_len - 1);
        ids.push(self.eos);
        ids.resize(self.max_len, self.pad);
        ids
    }
}

enum Act {
    Gelu,
    QuickGelu,
}

struct Layer {
    ln1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

/// Token embedding rows, kept in their stored precision (fp16 bits or f32).
enum Table {
    F16(Vec<u16>),
    F32(Vec<f32>),
}

pub struct ClipText {
    tok: Table,
    pos: Vec<f32>,
    layers: Vec<Layer>,
    final_ln: LayerNorm,
    act: Act,
    heads: usize,
    pub hidden: usize,
    /// `text_projection` of a `CLIPTextModelWithProjection` (SDXL's second encoder).
    proj: Option<Linear>,
}

impl ClipText {
    /// `dir` holds `config.json`; `t` is the opened weight file (with any LoRA updates attached).
    pub fn load(dir: &Path, t: Tensors) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        let hidden = cfg["hidden_size"].as_u64().context("hidden_size")? as usize;
        let heads = cfg["num_attention_heads"].as_u64().context("num_attention_heads")? as usize;
        let n_layers = cfg["num_hidden_layers"].as_u64().context("num_hidden_layers")? as usize;
        let eps = cfg["layer_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
        let act = match cfg["hidden_act"].as_str().unwrap_or("quick_gelu") {
            "quick_gelu" => Act::QuickGelu,
            "gelu" => Act::Gelu,
            other => bail!("text encoder activation {other} is not supported"),
        };
        let p = "text_model.";
        let te = format!("{p}embeddings.token_embedding.weight");
        let tok = match t.f16_bits(&te)? {
            Some(bits) => Table::F16(bits),
            None => Table::F32(t.f32(&te)?),
        };
        let pos = t.f32(&format!("{p}embeddings.position_embedding.weight"))?;
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let lp = format!("{p}encoder.layers.{l}.");
            layers.push(Layer {
                ln1: LayerNorm::load(&t, &format!("{lp}layer_norm1"), eps)?,
                q: Linear::load(&t, &format!("{lp}self_attn.q_proj"))?,
                k: Linear::load(&t, &format!("{lp}self_attn.k_proj"))?,
                v: Linear::load(&t, &format!("{lp}self_attn.v_proj"))?,
                o: Linear::load(&t, &format!("{lp}self_attn.out_proj"))?,
                ln2: LayerNorm::load(&t, &format!("{lp}layer_norm2"), eps)?,
                fc1: Linear::load(&t, &format!("{lp}mlp.fc1"))?,
                fc2: Linear::load(&t, &format!("{lp}mlp.fc2"))?,
            });
        }
        let final_ln = LayerNorm::load(&t, &format!("{p}final_layer_norm"), eps)?;
        Ok(Self {
            tok,
            pos,
            layers,
            final_ln,
            act,
            heads,
            hidden,
            proj: if t.has("text_projection.weight") {
                Some(Linear::load(&t, "text_projection")?)
            } else {
                None
            },
        })
    }

    /// Last hidden state (after the final layer norm) for `ids`: `[ids.len(), hidden]`.
    pub fn forward(&self, ids: &[u32]) -> Vec<f32> {
        let t = ids.len();
        let mut x = self.embed(ids);
        for l in &self.layers {
            self.layer(l, &mut x, t);
        }
        self.final_ln.forward(&x, t)
    }

    /// What SDXL conditions on: the penultimate hidden state (`hidden_states[-2]`, before the
    /// last layer and the final layer norm; `[ids.len(), hidden]`) and the pooled embedding (the
    /// final hidden state at the end-of-text token, through `text_projection` when the model
    /// has one, as `CLIPTextModelWithProjection.text_embeds`).
    pub fn forward_penultimate(&self, ids: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let (t, h) = (ids.len(), self.hidden);
        let mut x = self.embed(ids);
        let n = self.layers.len();
        for l in &self.layers[..n.saturating_sub(1)] {
            self.layer(l, &mut x, t);
        }
        let penultimate = x.clone();
        if let Some(l) = self.layers.last() {
            self.layer(l, &mut x, t);
        }
        // The end-of-text token has the highest id: its first position (transformers' argmax).
        let eos = (0..t).rev().max_by_key(|&i| ids[i]).unwrap_or(0);
        let pooled = self.final_ln.forward(&x[eos * h..(eos + 1) * h], 1);
        let pooled = match &self.proj {
            Some(p) => p.forward(&pooled, 1),
            None => pooled,
        };
        (penultimate, pooled)
    }

    /// Width of the pooled embedding of [`ClipText::forward_penultimate`].
    pub fn pooled_dim(&self) -> usize {
        self.proj.as_ref().map_or(self.hidden, |p| p.n_out)
    }

    /// Token plus position embeddings: `[ids.len(), hidden]`.
    fn embed(&self, ids: &[u32]) -> Vec<f32> {
        let (t, h) = (ids.len(), self.hidden);
        let mut x = vec![0f32; t * h];
        for_chunks(&mut x, h, false, |i, r| {
            let id = ids[i] as usize;
            match &self.tok {
                Table::F16(b) => {
                    for (o, &v) in r.iter_mut().zip(&b[id * h..(id + 1) * h]) {
                        *o = half::f16::from_bits(v).to_f32();
                    }
                }
                Table::F32(f) => r.copy_from_slice(&f[id * h..(id + 1) * h]),
            }
            for (o, &p) in r.iter_mut().zip(&self.pos[i * h..(i + 1) * h]) {
                *o += p;
            }
        });
        x
    }

    /// One pre-LN block, in place.
    fn layer(&self, l: &Layer, x: &mut [f32], t: usize) {
        let h = self.hidden;
        let y = l.ln1.forward(x, t);
        let (q, k, v) = (l.q.forward(&y, t), l.k.forward(&y, t), l.v.forward(&y, t));
        let a = attention(&q, &k, &v, t, t, self.heads, h / self.heads, true);
        add_inplace(x, &l.o.forward(&a, t));
        let y = l.ln2.forward(x, t);
        let mut m = l.fc1.forward(&y, t);
        match self.act {
            Act::Gelu => m.iter_mut().for_each(|v| *v = gelu_erf(*v)),
            Act::QuickGelu => m.iter_mut().for_each(|v| *v = quick_gelu(*v)),
        }
        add_inplace(x, &l.fc2.forward(&m, t));
    }
}
