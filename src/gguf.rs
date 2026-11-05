//! GGUF (llama.cpp) checkpoints: the header (metadata and tensor table), the mapping from
//! llama.cpp tensor names back to the Hugging Face names the loaders use, and the config,
//! tokenizer and chat template that GGUF files carry in their metadata.
//!
//! Supported architectures: `qwen35` (Qwen3.5 / 3.6 / 3.8 dense hybrids), `qwen35moe`, `qwen3`,
//! `llama` (Llama 3) and `gemma4` (Gemma 4 text models, dense and 26B-A4B MoE; the vision tower
//! of the `mmproj` files is not read).
//! llama.cpp's converter changes some tensors on the way; the loaders undo that where needed:
//! RMSNorm weights already include the `+1` of Qwen3.5's zero-centred norms, `ssm_a` holds
//! `-exp(A_log)`, and when there are fewer key heads than value heads the Gated DeltaNet value
//! heads are stored in tiled order (value head `h` reads key head `h % n_k` instead of
//! `h / (n_v / n_k)`). See `QwenConfig::v_heads_tiled`.
//!
//! Llama: the converter permutes the rows of `q_proj` / `k_proj` within each head for ggml's
//! adjacent-pair RoPE (`LlamaModel.permute`: rows `[n_head, 2, d/2]` become `[n_head, d/2, 2]`,
//! so stored row `2i + j` of a head is Hugging Face row `j * d/2 + i`). Rotating adjacent pairs of
//! the stored layout is the same arithmetic as rotate-half RoPE on the original layout, and the
//! query / key dot products do not depend on the order within a head, so the weights stay mapped
//! as they are and the attention uses the interleaved RoPE (`QwenConfig::rope_interleaved`). The
//! llama3 frequency scaling is not in the metadata: the converter stores the resulting divisor of
//! each frequency as the `rope_freqs.weight` tensor (loaded as `rope_freqs`).

use anyhow::{Context, Result, bail};
use serde_json::{Value as Json, json};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

/// A metadata value.
#[derive(Clone, Debug)]
pub enum Value {
    U(u64),
    I(i64),
    F(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U(v) => Some(v),
            Value::I(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F(v) => Some(v),
            Value::U(v) => Some(v as f64),
            Value::I(v) => Some(v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// One tensor of the file: llama.cpp name, dimensions in ggml order (`ne[0]` is the row length),
/// ggml type id and absolute byte range.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub ne: Vec<usize>,
    pub ggml_type: u32,
    pub start: usize,
    pub end: usize,
}

pub struct Header {
    pub meta: BTreeMap<String, Value>,
    pub tensors: Vec<Tensor>,
}

/// Elements per block and bytes per block of a ggml type.
pub fn type_size(t: u32) -> Option<(usize, usize)> {
    Some(match t {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M
        39 => (32, 17),   // MXFP4
        30 => (1, 2),     // BF16
        _ => return None,
    })
}

/// Whether `path` is a GGUF file (by its magic).
pub fn is_gguf(path: &Path) -> bool {
    let mut m = [0u8; 4];
    path.is_file() && std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut m)).is_ok() && &m == b"GGUF"
}

struct Reader<R> {
    r: R,
    pos: usize,
}

impl<R: Read> Reader<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b).context("GGUF header truncated")?;
        self.pos += N;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn len(&mut self) -> Result<usize> {
        let n = self.u64()?;
        // Bounds every allocation by something sane before trusting the file.
        if n > 1 << 32 {
            bail!("GGUF header: implausible length {n}");
        }
        Ok(n as usize)
    }
    fn string(&mut self) -> Result<String> {
        let n = self.len()?;
        let mut b = vec![0u8; n];
        self.r.read_exact(&mut b).context("GGUF header truncated")?;
        self.pos += n;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }
    fn value(&mut self, t: u32) -> Result<Value> {
        Ok(match t {
            0 => Value::U(self.bytes::<1>()?[0] as u64),
            1 => Value::I(self.bytes::<1>()?[0] as i8 as i64),
            2 => Value::U(u16::from_le_bytes(self.bytes()?) as u64),
            3 => Value::I(i16::from_le_bytes(self.bytes()?) as i64),
            4 => Value::U(self.u32()? as u64),
            5 => Value::I(i32::from_le_bytes(self.bytes()?) as i64),
            6 => Value::F(f32::from_le_bytes(self.bytes()?) as f64),
            7 => Value::Bool(self.bytes::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let (et, n) = (self.u32()?, self.len()?);
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Value::Array(v)
            }
            10 => Value::U(self.u64()?),
            11 => Value::I(i64::from_le_bytes(self.bytes()?)),
            12 => Value::F(f64::from_le_bytes(self.bytes()?)),
            t => bail!("GGUF metadata: unknown value type {t}"),
        })
    }
}

/// Parses the header of a GGUF file of `file_len` bytes.
pub fn parse_header(r: impl Read, file_len: usize) -> Result<Header> {
    let mut r = Reader { r, pos: 0 };
    if &r.bytes::<4>()? != b"GGUF" {
        bail!("not a GGUF file");
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        bail!("GGUF version {version} is not supported");
    }
    let (n_tensors, n_kv) = (r.len()?, r.len()?);
    let mut meta = BTreeMap::new();
    for _ in 0..n_kv {
        let k = r.string()?;
        let t = r.u32()?;
        meta.insert(k, r.value(t)?);
    }
    let mut raw = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = r.string()?;
        let nd = r.u32()? as usize;
        if nd > 4 {
            bail!("GGUF tensor {name}: {nd} dimensions");
        }
        let ne = (0..nd).map(|_| r.len()).collect::<Result<Vec<_>>>()?;
        let (t, off) = (r.u32()?, r.u64()? as usize);
        raw.push((name, ne, t, off));
    }
    let align = meta.get("general.alignment").and_then(Value::as_u64).unwrap_or(32) as usize;
    let data_start = r.pos.next_multiple_of(align.max(1));
    let mut tensors = Vec::with_capacity(raw.len());
    for (name, ne, t, off) in raw {
        let (blk, bytes) = type_size(t).with_context(|| format!("GGUF tensor {name}: unknown ggml type {t}"))?;
        let n: usize = ne.iter().product();
        if ne.first().is_some_and(|&r| r % blk != 0) {
            bail!("GGUF tensor {name}: row length {} is not a multiple of the block size {blk}", ne[0]);
        }
        let start = data_start + off;
        let end = start + n / blk * bytes;
        if end > file_len {
            bail!("GGUF tensor {name} extends past the end of the file");
        }
        tensors.push(Tensor {
            name,
            ne,
            ggml_type: t,
            start,
            end,
        });
    }
    Ok(Header { meta, tensors })
}

/// Reads just the header of `path` (metadata and tensor table, not the weights).
pub fn read_header(path: &Path) -> Result<Header> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = f.metadata()?.len() as usize;
    parse_header(std::io::BufReader::with_capacity(1 << 20, f), len).with_context(|| format!("{}", path.display()))
}

/// [`read_header`], cached per path for the process (the config, tokenizer, template and EOS
/// lookups all read the same header, vocabulary included).
pub fn header(path: &Path) -> Result<std::sync::Arc<Header>> {
    use std::sync::{Arc, Mutex};
    static CACHE: Mutex<Vec<(std::path::PathBuf, Arc<Header>)>> = Mutex::new(Vec::new());
    if let Some((_, h)) = CACHE.lock().unwrap().iter().find(|(p, _)| p == path) {
        return Ok(h.clone());
    }
    let h = Arc::new(read_header(path)?);
    CACHE.lock().unwrap().push((path.to_path_buf(), h.clone()));
    Ok(h)
}

impl Header {
    pub fn get(&self, k: &str) -> Option<&Value> {
        self.meta.get(k)
    }
    pub fn arch(&self) -> &str {
        self.get("general.architecture").and_then(Value::as_str).unwrap_or("")
    }
    fn arch_u64(&self, k: &str) -> Option<usize> {
        self.get(&format!("{}.{k}", self.arch()))
            .and_then(Value::as_u64)
            .map(|v| v as usize)
    }
    fn arch_f64(&self, k: &str) -> Option<f64> {
        self.get(&format!("{}.{k}", self.arch())).and_then(Value::as_f64)
    }
    /// A per-layer metadata value (an array, or one value for every layer).
    fn arch_per_layer(&self, k: &str) -> Option<Vec<Value>> {
        let n = self.n_layers();
        match self.get(&format!("{}.{k}", self.arch()))? {
            Value::Array(a) if a.len() == n => Some(a.clone()),
            Value::Array(_) => None,
            v => Some(vec![v.clone(); n]),
        }
    }
    fn has(&self, name: &str) -> bool {
        self.tensors.iter().any(|t| t.name == name)
    }

    /// The `model_type` of the equivalent Hugging Face checkpoint.
    pub fn model_type(&self) -> Result<&'static str> {
        Ok(match self.arch() {
            "qwen35" => "qwen3_5_text",
            "qwen35moe" => "qwen3_5_moe_text",
            "qwen4exp" => "qwen4_exp_text",
            "qwen3" => "qwen3",
            "llama" => "llama",
            "gemma4" => "gemma4_text",
            "mistral-large4" => "mistral_large4_text",
            "deepseek4" => "deepseek_v4",
            a => bail!(
                "GGUF architecture {a:?} is not supported (qwen35, qwen35moe, qwen4exp, qwen3, llama, gemma4, mistral-large4 and deepseek4 are)"
            ),
        })
    }

    /// Transformer layers, not counting the MTP (NextN) layers stored after them.
    pub fn n_layers(&self) -> usize {
        self.arch_u64("block_count").unwrap_or(0) - self.arch_u64("nextn_predict_layers").unwrap_or(0)
    }

    /// The Hugging Face `config.json` this file was converted from (the fields the loaders read).
    pub fn config_json(&self) -> Result<Json> {
        let mt = self.model_type()?;
        if mt == "gemma4_text" {
            return self.gemma4_config();
        }
        if mt == "mistral_large4_text" {
            return self.mistral4_config();
        }
        if mt == "deepseek_v4" {
            return self.deepseek4_config();
        }
        let llama = mt == "llama";
        if let Some(t) = self.get(&format!("{}.rope.scaling.type", self.arch())).and_then(Value::as_str)
            && !matches!(t, "none" | "llama3")
        {
            bail!("GGUF RoPE scaling {t:?} is not supported");
        }
        let need = |k: &str| {
            self.arch_u64(k)
                .with_context(|| format!("GGUF metadata has no {}.{k}", self.arch()))
        };
        let h = need("embedding_length")?;
        let head_dim = self.arch_u64("attention.key_length").unwrap_or(h / need("attention.head_count")?);
        // A split file's first part may hold only metadata: the vocabulary's size then.
        let vocab = self
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .map(|t| t.ne[1])
            .or_else(|| self.get("tokenizer.ggml.tokens").and_then(Value::as_array).map(|a| a.len()))
            .context("GGUF has no token_embd.weight")?;
        let mut c = json!({
            "model_type": mt,
            "hidden_size": h,
            "intermediate_size": self.arch_u64("feed_forward_length").unwrap_or(0),
            "num_hidden_layers": self.n_layers(),
            "num_attention_heads": need("attention.head_count")?,
            "num_key_value_heads": self.arch_u64("attention.head_count_kv").unwrap_or(need("attention.head_count")?),
            "head_dim": head_dim,
            "rms_norm_eps": self.arch_f64("attention.layer_norm_rms_epsilon").unwrap_or(if llama { 1e-5 } else { 1e-6 }),
            "vocab_size": vocab,
            "rope_theta": self.arch_f64("rope.freq_base").unwrap_or(if llama { 10_000.0 } else { 10_000_000.0 }),
            "tie_word_embeddings": self.has("token_embd.weight") && !self.has("output.weight"),
            "mtp_num_hidden_layers": self.arch_u64("nextn_predict_layers").unwrap_or(0),
        });
        if llama {
            let rot = self.arch_u64("rope.dimension_count").unwrap_or(head_dim);
            if rot != head_dim {
                bail!("GGUF Llama with partial RoPE ({rot} of {head_dim} dimensions) is not supported");
            }
        } else if mt != "qwen3" {
            let (nk, nv) = (need("ssm.group_count")?, need("ssm.time_step_rank")?);
            let o = c.as_object_mut().unwrap();
            o.insert(
                "full_attention_interval".into(),
                json!(self.arch_u64("full_attention_interval").unwrap_or(4)),
            );
            o.insert("linear_conv_kernel_dim".into(), json!(need("ssm.conv_kernel")?));
            o.insert("linear_key_head_dim".into(), json!(need("ssm.state_size")?));
            o.insert("linear_value_head_dim".into(), json!(need("ssm.inner_size")? / nv));
            o.insert("linear_num_key_heads".into(), json!(nk));
            o.insert("linear_num_value_heads".into(), json!(nv));
            o.insert("attn_output_gate".into(), json!(true));
            let rot = self.arch_u64("rope.dimension_count").unwrap_or(head_dim / 4);
            o.insert("partial_rotary_factor".into(), json!(rot as f64 / head_dim as f64));
        }
        if mt == "qwen3_5_moe_text" || mt == "qwen4_exp_text" {
            let o = c.as_object_mut().unwrap();
            o.insert("num_experts".into(), json!(need("expert_count")?));
            o.insert("num_experts_per_tok".into(), json!(need("expert_used_count")?));
            o.insert("moe_intermediate_size".into(), json!(need("expert_feed_forward_length")?));
            o.insert(
                "shared_expert_intermediate_size".into(),
                json!(self.arch_u64("expert_shared_feed_forward_length").unwrap_or(0)),
            );
        }
        if mt == "qwen4_exp_text" {
            self.qwen4exp_config(c.as_object_mut().unwrap())?;
        }
        Ok(c)
    }

    /// The Qwen4-Exp keys (hyper-connections, QSA indexer, n-gram embedding; see
    /// `crate::qwen4exp`) of the `config.json` a qwen4exp file was converted from. The n-gram hash
    /// constants, buffers of the safetensors checkpoint, come along as config keys.
    fn qwen4exp_config(&self, o: &mut serde_json::Map<String, Json>) -> Result<()> {
        let need = |k: &str| {
            self.arch_u64(k)
                .with_context(|| format!("GGUF metadata has no {}.{k}", self.arch()))
        };
        let arr = |k: &str| -> Option<Vec<u64>> {
            self.get(&format!("{}.{k}", self.arch()))
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
        };
        o.insert("hc_count".into(), json!(need("hyper_connection.count")?));
        o.insert("hc_lowrank".into(), json!(need("hyper_connection.low_rank")?));
        o.insert("output_gate_type".into(), json!("sigmoid"));
        o.insert("indexer_n_heads".into(), json!(need("attention.indexer.head_count")?));
        o.insert("indexer_head_dim".into(), json!(need("attention.indexer.key_length")?));
        o.insert("indexer_budget".into(), json!(need("attention.indexer.top_k")?));
        let ratio = arr("attention.compress_ratios")
            .and_then(|r| r.into_iter().find(|&v| v > 0))
            .unwrap_or(4);
        o.insert("indexer_compress_ratio".into(), json!(ratio));
        if let Some(ple) = arr("ple.layers") {
            o.insert("ple_layer_ids".into(), json!(ple.iter().map(|l| l + 1).collect::<Vec<_>>()));
            o.insert("ngram_size".into(), json!(need("ple.ngram_size")?));
            o.insert("heads_per_ngram".into(), json!(need("ple.heads_per_ngram")?));
            o.insert("ple_conv_kernel_size".into(), json!(need("ple.conv_kernel")?));
            o.insert("eos_token_id".into(), json!(need("ple.eos_token_id")?));
            if let Some(img) = self.arch_u64("ple.image_token_id") {
                o.insert("image_token_id".into(), json!(img));
            }
            let h = need("embedding_length")?;
            o.insert("ple_embed_dim".into(), json!(h));
            for (k, key) in [
                ("ple_layer_multipliers", "ple.layer_multipliers"),
                ("ple_head_offsets", "ple.head_offsets"),
                ("ple_head_vocab_sizes", "ple.head_vocab_sizes"),
            ] {
                o.insert(k.into(), json!(arr(key).with_context(|| format!("GGUF metadata has no {key}"))?));
            }
        }
        if let Some(sec) = arr("rope.dimension_sections") {
            o.insert(
                "rope_parameters".into(),
                json!({
                    "rope_theta": self.arch_f64("rope.freq_base").unwrap_or(10_000_000.0),
                    "mrope_section": sec.iter().take(3).collect::<Vec<_>>(),
                    "mrope_interleaved": true,
                    "partial_rotary_factor": o.get("partial_rotary_factor").cloned().unwrap_or(json!(0.25)),
                }),
            );
        }
        Ok(())
    }

    /// The Gemma 4 text config (`Gemma4TextConfig` fields; the attention shape of the global
    /// layers comes from the per-layer KV head counts and the `key_length` / `key_length_swa` pair).
    fn gemma4_config(&self) -> Result<Json> {
        let need = |k: &str| {
            self.arch_u64(k)
                .with_context(|| format!("GGUF metadata has no {}.{k}", self.arch()))
        };
        let n = self.n_layers();
        let vocab = self
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .map(|t| t.ne[1])
            .context("GGUF has no token_embd.weight")?;
        let sliding: Vec<bool> = match self.arch_per_layer("attention.sliding_window_pattern") {
            Some(v) => v.iter().map(|x| matches!(x, Value::Bool(true))).collect(),
            None => (0..n).map(|i| (i + 1) % 6 != 0).collect(),
        };
        let kv: Vec<usize> = self
            .arch_per_layer("attention.head_count_kv")
            .context("GGUF metadata has no head_count_kv")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let ff: Vec<usize> = self
            .arch_per_layer("feed_forward_length")
            .context("GGUF metadata has no feed_forward_length")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let first_full = sliding.iter().position(|s| !s).unwrap_or(n - 1);
        let first_swa = sliding.iter().position(|s| *s).unwrap_or(0);
        let inter = *ff.iter().min().unwrap_or(&0);
        let h = need("embedding_length")?;
        let per_layer = self.arch_u64("embedding_length_per_layer_input").unwrap_or(0);
        // Global layers without a value projection read their keys as values.
        let k_eq_v = (0..n).any(|l| !sliding[l] && !self.has(&format!("blk.{l}.attn_v.weight")));
        let mut c = json!({
            "model_type": "gemma4_text",
            "hidden_size": h,
            "intermediate_size": inter,
            "use_double_wide_mlp": ff.iter().any(|&f| f > inter),
            "num_hidden_layers": n,
            "num_attention_heads": need("attention.head_count")?,
            "num_key_value_heads": kv[first_swa],
            "num_global_key_value_heads": kv[first_full],
            "attention_k_eq_v": k_eq_v,
            "head_dim": self.arch_u64("attention.key_length_swa").unwrap_or(256),
            "global_head_dim": self.arch_u64("attention.key_length").unwrap_or(512),
            "rms_norm_eps": self.arch_f64("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            "vocab_size": vocab,
            "vocab_size_per_layer_input": vocab,
            "hidden_size_per_layer_input": per_layer,
            "num_kv_shared_layers": self.arch_u64("attention.shared_kv_layers").unwrap_or(0),
            "sliding_window": self.arch_u64("attention.sliding_window").unwrap_or(512),
            "final_logit_softcapping": self.arch_f64("final_logit_softcapping"),
            "layer_types": sliding.iter().map(|&s| if s { "sliding_attention" } else { "full_attention" }).collect::<Vec<_>>(),
            "rope_parameters": {
                "full_attention": {
                    "rope_theta": self.arch_f64("rope.freq_base").unwrap_or(1_000_000.0),
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                },
                "sliding_attention": {
                    "rope_theta": self.arch_f64("rope.freq_base_swa").unwrap_or(10_000.0),
                    "rope_type": "default",
                },
            },
            "tie_word_embeddings": !self.has("output.weight"),
        });
        if let Some(e) = self.arch_u64("expert_count").filter(|&e| e > 0) {
            let o = c.as_object_mut().unwrap();
            o.insert("enable_moe_block".into(), json!(true));
            o.insert("num_experts".into(), json!(e));
            o.insert("top_k_experts".into(), json!(need("expert_used_count")?));
            o.insert("moe_intermediate_size".into(), json!(need("expert_feed_forward_length")?));
        }
        Ok(c)
    }

    /// The Mistral Large 4 text config (`MistralLarge4TextConfig` fields). llama.cpp stores the MLA
    /// shapes as the absorbed attention sees them: `key_length` = latent + rotary part,
    /// `key_length_mla` / `value_length_mla` = the per-head query-key / value sizes.
    fn mistral4_config(&self) -> Result<Json> {
        let need = |k: &str| {
            self.arch_u64(k)
                .with_context(|| format!("GGUF metadata has no {}.{k}", self.arch()))
        };
        let n = self.n_layers();
        // The embedding may sit in a later part of a split model: the vocabulary size key, else
        // the token list.
        let vocab = self
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .map(|t| t.ne[1])
            .or_else(|| self.arch_u64("vocab_size"))
            .or_else(|| self.get("tokenizer.ggml.tokens").and_then(Value::as_array).map(|a| a.len()))
            .context("GGUF has no token_embd.weight")?;
        let split = self.get("split.count").and_then(Value::as_u64).unwrap_or(1) > 1;
        let flags = |k: &str| -> Vec<bool> {
            match self.arch_per_layer(k) {
                Some(v) => v.iter().map(|x| matches!(x, Value::Bool(true))).collect(),
                None => vec![true; n],
            }
        };
        let rope = flags("attention.rope_pattern");
        let full = flags("attention.indexer.types");
        let src: Vec<usize> = match self.arch_per_layer("attention.kv_source_map") {
            Some(v) => v.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect(),
            None => (0..n).collect(),
        };
        // 1 = softmax (llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX), this model's router.
        if let Some(g) = self.arch_u64("expert_gating_func")
            && g != 1
        {
            bail!("GGUF mistral-large4 with expert gating function {g} is not supported (softmax only)");
        }
        if matches!(self.get(&format!("{}.expert_weights_norm", self.arch())), Some(Value::Bool(false))) {
            bail!("GGUF mistral-large4 without normalized expert weights is not supported");
        }
        let rot = need("rope.dimension_count")?;
        let qk = need("attention.key_length_mla")?;
        let theta = self.arch_f64("rope.freq_base").unwrap_or(10_000.0);
        let rope_parameters = match self.get(&format!("{}.rope.scaling.type", self.arch())).and_then(Value::as_str) {
            Some("yarn") => json!({
                "rope_type": "yarn",
                "rope_theta": theta,
                "factor": self.arch_f64("rope.scaling.factor").unwrap_or(1.0),
                "original_max_position_embeddings": self.arch_u64("rope.scaling.original_context_length").unwrap_or(4096),
                "beta_fast": self.arch_f64("rope.scaling.yarn_beta_fast").unwrap_or(32.0),
                "beta_slow": self.arch_f64("rope.scaling.yarn_beta_slow").unwrap_or(1.0),
                // llama.cpp's converter only takes YaRN without magnitude scaling for this model.
                "attention_factor": 1.0,
            }),
            None | Some("none") => json!({"rope_type": "default", "rope_theta": theta}),
            Some(t) => bail!("GGUF RoPE scaling {t:?} is not supported for mistral-large4"),
        };
        Ok(json!({
            "model_type": "mistral_large4_text",
            "hidden_size": need("embedding_length")?,
            "intermediate_size": self.arch_u64("feed_forward_length").unwrap_or(0),
            "moe_intermediate_size": need("expert_feed_forward_length")?,
            "num_hidden_layers": n,
            "num_attention_heads": need("attention.head_count")?,
            "vocab_size": vocab,
            "rms_norm_eps": self.arch_f64("attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
            "kv_lora_rank": need("attention.kv_lora_rank")?,
            "qk_rope_head_dim": rot,
            "qk_nope_head_dim": qk - rot,
            "v_head_dim": need("attention.value_length_mla")?,
            "n_routed_experts": need("expert_count")?,
            "num_experts_per_tok": need("expert_used_count")?,
            "n_shared_experts": self.arch_u64("expert_shared_count").unwrap_or(1),
            "routed_scaling_factor": self.arch_f64("expert_weights_scale").unwrap_or(1.0),
            "n_group": self.arch_u64("expert_group_count").unwrap_or(1),
            "topk_group": self.arch_u64("expert_group_used_count").unwrap_or(1),
            "first_k_dense_replace": self.arch_u64("leading_dense_block_count").unwrap_or(1),
            "index_topk": need("attention.indexer.top_k")?,
            "index_head_dim": need("attention.indexer.key_length")?,
            "index_n_heads": need("attention.indexer.head_count")?,
            "indexer_types": full.iter().map(|&f| if f { "full" } else { "shared" }).collect::<Vec<_>>(),
            "no_rope_layers": rope.iter().map(|&r| r as u8).collect::<Vec<_>>(),
            "kv_source_layers": src,
            "final_logit_softcapping": self.arch_f64("final_logit_softcapping"),
            // (In a split model the head may be in another part; the loader looks for it.)
            "tie_word_embeddings": !split && !self.has("output.weight"),
            "rope_parameters": rope_parameters,
            "indexer_norm_eps": self.arch_f64("attention.layer_norm_epsilon").unwrap_or(1e-6),
        }))
    }

    /// The DeepSeek V4 config (`config.json` keys of deepseek-ai/DeepSeek-V4-Flash).
    pub(crate) fn deepseek4_config(&self) -> Result<Json> {
        let need = |k: &str| {
            self.arch_u64(k)
                .with_context(|| format!("GGUF metadata has no {}.{k}", self.arch()))
        };
        let vocab = self
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .map(|t| t.ne[1])
            .or_else(|| self.arch_u64("vocab_size"))
            .or_else(|| self.get("tokenizer.ggml.tokens").and_then(Value::as_array).map(|a| a.len()))
            .context("GGUF has no token_embd.weight")?;
        let split = self.get("split.count").and_then(Value::as_u64).unwrap_or(1) > 1;
        // llama.cpp's gating functions: 1 softmax, 2 sigmoid, 4 sqrt(softplus).
        let scoring = match self.arch_u64("expert_gating_func") {
            Some(1) => "softmax",
            Some(2) => "sigmoid",
            Some(4) | None => "sqrtsoftplus",
            Some(g) => bail!("GGUF deepseek4 with expert gating function {g} is not supported"),
        };
        let first = |k: &str| match self.get(&format!("{}.{k}", self.arch())) {
            Some(Value::Array(a)) => a.first().and_then(Value::as_f64),
            Some(v) => v.as_f64(),
            None => None,
        };
        let ratios: Vec<u64> = self
            .get(&format!("{}.attention.compress_ratios", self.arch()))
            .and_then(Value::as_array)
            .context("GGUF deepseek4 has no attention.compress_ratios")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0))
            .collect();
        let rope_scaling = match self.get(&format!("{}.rope.scaling.type", self.arch())).and_then(Value::as_str) {
            Some("yarn") => json!({
                "type": "yarn",
                "factor": self.arch_f64("rope.scaling.factor").unwrap_or(1.0),
                "original_max_position_embeddings": self.arch_u64("rope.scaling.original_context_length").unwrap_or(0),
                "beta_fast": self.arch_f64("rope.scaling.yarn_beta_fast").unwrap_or(32.0),
                "beta_slow": self.arch_f64("rope.scaling.yarn_beta_slow").unwrap_or(1.0),
            }),
            None | Some("none") => Json::Null,
            Some(t) => bail!("GGUF RoPE scaling {t:?} is not supported for deepseek4"),
        };
        Ok(json!({
            "model_type": "deepseek_v4",
            "hidden_size": need("embedding_length")?,
            "num_hidden_layers": self.n_layers(),
            "num_attention_heads": need("attention.head_count")?,
            "head_dim": need("attention.key_length")?,
            "qk_rope_head_dim": need("rope.dimension_count")?,
            "q_lora_rank": need("attention.q_lora_rank")?,
            "o_groups": need("attention.output_group_count")?,
            "o_lora_rank": need("attention.output_lora_rank")?,
            "vocab_size": vocab,
            "rms_norm_eps": self.arch_f64("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            "n_routed_experts": need("expert_count")?,
            "n_shared_experts": self.arch_u64("expert_shared_count").unwrap_or(1),
            "num_experts_per_tok": need("expert_used_count")?,
            "moe_intermediate_size": need("expert_feed_forward_length")?,
            "routed_scaling_factor": self.arch_f64("expert_weights_scale").unwrap_or(1.0),
            "scoring_func": scoring,
            "norm_topk_prob": !matches!(self.get(&format!("{}.expert_weights_norm", self.arch())), Some(Value::Bool(false))),
            "swiglu_limit": first("swiglu_clamp_exp").unwrap_or(0.0),
            "num_hash_layers": self.arch_u64("hash_layer_count").unwrap_or(0),
            "sliding_window": need("attention.sliding_window")?,
            "compress_ratios": ratios,
            "compress_rope_theta": self.arch_f64("attention.compress_rope_freq_base").unwrap_or(160_000.0),
            "rope_theta": self.arch_f64("rope.freq_base").unwrap_or(10_000.0),
            "rope_scaling": rope_scaling,
            "index_n_heads": need("attention.indexer.head_count")?,
            "index_head_dim": need("attention.indexer.key_length")?,
            "index_topk": need("attention.indexer.top_k")?,
            "hc_mult": self.arch_u64("hyper_connection.count").unwrap_or(4),
            "hc_sinkhorn_iters": self.arch_u64("hyper_connection.sinkhorn_iterations").unwrap_or(20),
            "hc_eps": self.arch_f64("hyper_connection.epsilon").unwrap_or(1e-6),
            // (In a split model the head may be in another part; the loader looks for it.)
            "tie_word_embeddings": !split && !self.has("output.weight"),
        }))
    }

    /// End-of-generation token ids.
    pub fn eos_ids(&self) -> Vec<u32> {
        let mut v: Vec<u32> = ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"]
            .iter()
            .filter_map(|k| self.get(k).and_then(Value::as_u64).map(|x| x as u32))
            .collect();
        v.dedup();
        v
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.get("tokenizer.chat_template").and_then(Value::as_str)
    }

    /// A Hugging Face `tokenizer.json` equivalent to the GGUF vocabulary (byte-level BPE).
    pub fn tokenizer_json(&self) -> Result<Json> {
        let model = self.get("tokenizer.ggml.model").and_then(Value::as_str).unwrap_or("");
        if model == "gemma4" {
            return self.gemma4_tokenizer_json();
        }
        if model != "gpt2" {
            bail!("GGUF tokenizer model {model:?} is not supported (only byte-level BPE, \"gpt2\")");
        }
        let pre = self.get("tokenizer.ggml.pre").and_then(Value::as_str).unwrap_or("");
        // Llama 3 and Mistral's tekken (as their tokenizer.json): no normalizer, and a word found
        // whole in the vocabulary is one token whatever the merges say (`ignore_merges`).
        let whole_words = matches!(pre, "llama-bpe" | "tekken" | "tekken-num3" | "pixtral");
        let tekken = matches!(pre, "tekken" | "tekken-num3" | "pixtral");
        // DeepSeek V3 / V4: digits in groups of up to three and CJK runs split off first, no
        // normalizer.
        let deepseek = matches!(pre, "deepseek-v3" | "joyai-llm");
        let regex = match pre {
            "deepseek-v3" | "joyai-llm" => {
                r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##
            }
            "tekken" | "pixtral" => {
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            }
            // Numbers in right-aligned groups of three digits (Mistral Large 4).
            "tekken-num3" => {
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            }
            "qwen35" => {
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            }
            "qwen2" => r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            "llama-bpe" => {
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            }
            p => bail!(
                "GGUF pre-tokenizer {p:?} is not supported (qwen35, qwen2, llama-bpe, tekken, tekken-num3, deepseek-v3 and joyai-llm are)"
            ),
        };
        let tokens = self
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .context("GGUF has no tokenizer.ggml.tokens")?;
        let types = self.get("tokenizer.ggml.token_type").and_then(Value::as_array);
        let merges: Vec<&str> = self
            .get("tokenizer.ggml.merges")
            .and_then(Value::as_array)
            .context("GGUF has no tokenizer.ggml.merges")?
            .iter()
            .filter_map(Value::as_str)
            .collect();
        let mut vocab = serde_json::Map::new();
        let mut added = Vec::new();
        for (id, t) in tokens.iter().enumerate() {
            let s = t.as_str().context("non-string token")?;
            // 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte.
            match types.and_then(|ty| ty.get(id)).and_then(Value::as_u64).unwrap_or(1) {
                3 | 4 => {
                    let special = types.and_then(|ty| ty[id].as_u64()) == Some(3);
                    added.push(json!({"id": id, "content": s, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": special}));
                    // Tekken numbers its specials first (0..1000), DeepSeek's BOS / EOS / pad are
                    // 0..3: the model vocabulary must hold them too, or the tokenizer renumbers
                    // the added tokens after it.
                    if tekken || deepseek {
                        vocab.insert(s.to_string(), json!(id));
                    }
                }
                5 => {}
                _ => {
                    vocab.insert(s.to_string(), json!(id));
                }
            }
        }
        let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
        let split = |r: &str| json!({"type": "Split", "pattern": {"Regex": r}, "behavior": "Isolated", "invert": false});
        let mut pretok = Vec::new();
        if deepseek {
            pretok.push(split(r"\p{N}{1,3}"));
            pretok.push(split("[一-龥぀-ゟ゠-ヿ]+"));
        }
        pretok.push(split(regex));
        pretok.push(byte_level.clone());
        Ok(json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": if whole_words || deepseek { Json::Null } else { json!({"type": "NFC"}) },
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": pretok},
            "post_processor": byte_level,
            "decoder": byte_level,
            "model": {
                "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
                "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": whole_words,
                "vocab": vocab, "merges": merges,
            },
        }))
    }

    /// Gemma 4's `tokenizer.json`: SentencePiece-style BPE over `▁`-for-space text with byte
    /// fallback, rebuilt from the GGUF vocabulary and merges (control and user-defined tokens are
    /// the added tokens).
    fn gemma4_tokenizer_json(&self) -> Result<Json> {
        let tokens = self
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .context("GGUF has no tokenizer.ggml.tokens")?;
        let types = self.get("tokenizer.ggml.token_type").and_then(Value::as_array);
        let mut vocab = serde_json::Map::new();
        let mut added = Vec::new();
        for (id, t) in tokens.iter().enumerate() {
            let s = t.as_str().context("non-string token")?;
            if let Some(3 | 4) = types.and_then(|ty| ty.get(id)).and_then(Value::as_u64) {
                added.push(json!({"id": id, "content": s, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}));
            }
            vocab.insert(s.to_string(), json!(id));
        }
        let merges: Vec<Json> = self
            .get("tokenizer.ggml.merges")
            .and_then(Value::as_array)
            .context("GGUF has no tokenizer.ggml.merges")?
            .iter()
            .filter_map(Value::as_str)
            .map(|m| {
                let (a, b) = m.split_once(' ').unwrap_or((m, ""));
                json!([a, b])
            })
            .collect();
        let unk = self
            .get("tokenizer.ggml.unknown_token_id")
            .and_then(Value::as_u64)
            .and_then(|i| tokens.get(i as usize))
            .and_then(Value::as_str)
            .unwrap_or("<unk>");
        Ok(json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
            "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false},
            "post_processor": null,
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
            ]},
            "model": {
                "type": "BPE", "dropout": null, "unk_token": unk, "continuing_subword_prefix": null,
                "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false,
                "vocab": vocab, "merges": merges,
            },
        }))
    }

    pub fn tokenizer(&self) -> Result<tokenizers::Tokenizer> {
        let j = serde_json::to_vec(&self.tokenizer_json()?)?;
        tokenizers::Tokenizer::from_bytes(j).map_err(|e| anyhow::anyhow!("GGUF tokenizer: {e}"))
    }
}

/// The Hugging Face name and shape of GGUF tensor `t` (None for tensors the loaders do not use).
/// Shapes are given in PyTorch order (the reverse of ggml's `ne`).
pub fn hf_tensor(arch: &str, t: &Tensor, n_layers: usize) -> Option<(String, Vec<usize>)> {
    // DeepSeek V4: read in DeepSeek's own naming (as its safetensors release).
    if arch == "deepseek4" {
        return crate::deepseek4::gguf_name(&t.name, &t.ne);
    }
    let mut shape: Vec<usize> = t.ne.iter().rev().copied().collect();
    let name = match t.name.as_str() {
        "token_embd.weight" => "model.embed_tokens.weight".to_string(),
        "output_norm.weight" => "model.norm.weight".to_string(),
        "output.weight" => "lm_head.weight".to_string(),
        // Llama 3: the divisor of each RoPE frequency (the llama3 scaling, see the module docs).
        "rope_freqs.weight" => "rope_freqs".to_string(),
        "per_layer_token_embd.weight" if arch == "qwen4exp" => "model.ngram_embedding.weight".to_string(),
        // Gemma 4 E2B / E4B per-layer embeddings.
        "per_layer_token_embd.weight" => "model.embed_tokens_per_layer.weight".to_string(),
        "per_layer_model_proj.weight" => "model.per_layer_model_projection.weight".to_string(),
        "per_layer_proj_norm.weight" => "model.per_layer_projection_norm.weight".to_string(),
        // FastMTP sidecars: draft-vocabulary index -> target token id (for a trimmed output head).
        "d2t" => "mtp.d2t".to_string(),
        // Qwen4-Exp: the final hyper-connection mixer (in place of the output norm) and the
        // n-gram hash embedding table.
        "output_hc_norm.weight" if arch == "qwen4exp" => "model.hyper_connection_mixer.hc_norm.weight".to_string(),
        "output_hc_down.weight" if arch == "qwen4exp" => "model.hyper_connection_mixer.input_mix_weight_down.weight".to_string(),
        "output_hc_up.weight" if arch == "qwen4exp" => "model.hyper_connection_mixer.input_mix_weight_up.weight".to_string(),
        n => {
            let rest = n.strip_prefix("blk.")?;
            let (l, rest) = rest.split_once('.')?;
            let l: usize = l.parse().ok()?;
            let mtp = l >= n_layers;
            let (base, suffix) = rest.rsplit_once('.').unwrap_or((rest, ""));
            if mtp {
                let top = match base {
                    "nextn.eh_proj" => Some("mtp.fc.weight"),
                    "nextn.enorm" => Some("mtp.pre_fc_norm_embedding.weight"),
                    "nextn.hnorm" => Some("mtp.pre_fc_norm_hidden.weight"),
                    "nextn.shared_head_norm" => Some("mtp.norm.weight"),
                    "nextn.embed_tokens" => Some("mtp.embed_tokens.weight"),
                    "nextn.shared_head_head" => Some("mtp.lm_head.weight"),
                    _ => None,
                };
                if let Some(top) = top {
                    return Some((top.to_string(), shape));
                }
            }
            if arch == "qwen4exp"
                && let Some(n) = qwen4exp_name(base, suffix, &mut shape, &t.ne)
            {
                return Some((
                    match n.strip_prefix('/') {
                        Some(top) => top.to_string(),
                        None if mtp => format!("mtp.layers.{}.{n}", l - n_layers),
                        None => format!("model.layers.{l}.{n}"),
                    },
                    shape,
                ));
            }
            let hf = match (arch, base) {
                ("gemma4", _) => {
                    let n = match (base, suffix) {
                        ("ffn_norm", _) => "pre_feedforward_layernorm.weight",
                        ("post_ffw_norm", _) => "post_feedforward_layernorm.weight",
                        ("post_ffw_norm_1", _) => "post_feedforward_layernorm_1.weight",
                        ("pre_ffw_norm_2", _) => "pre_feedforward_layernorm_2.weight",
                        ("post_ffw_norm_2", _) => "post_feedforward_layernorm_2.weight",
                        ("ffn_gate_inp", "scale") => "router.scale",
                        ("ffn_gate_inp", _) => "router.proj.weight",
                        ("ffn_down_exps", "scale") => "router.per_expert_scale",
                        ("ffn_gate_up_exps", _) => "experts.gate_up_proj",
                        ("ffn_gate_exps", _) => "experts.gate_proj",
                        ("ffn_up_exps", _) => "experts.up_proj",
                        ("ffn_down_exps", _) => "experts.down_proj",
                        ("layer_output_scale", _) => "layer_scalar",
                        ("inp_gate", _) => "per_layer_input_gate.weight",
                        ("proj", _) => "per_layer_projection.weight",
                        ("post_norm", _) => "post_per_layer_input_norm.weight",
                        _ => "",
                    };
                    if !n.is_empty() {
                        return Some((format!("model.layers.{l}.{n}"), shape));
                    }
                    match base {
                        "attn_norm" => "input_layernorm.weight",
                        "post_attention_norm" => "post_attention_layernorm.weight",
                        "attn_q" => "self_attn.q_proj.weight",
                        "attn_k" => "self_attn.k_proj.weight",
                        "attn_v" => "self_attn.v_proj.weight",
                        "attn_output" => "self_attn.o_proj.weight",
                        "attn_q_norm" => "self_attn.q_norm.weight",
                        "attn_k_norm" => "self_attn.k_norm.weight",
                        "ffn_gate" => "mlp.gate_proj.weight",
                        "ffn_up" => "mlp.up_proj.weight",
                        "ffn_down" => "mlp.down_proj.weight",
                        _ => return None,
                    }
                }
                // Mistral Large 4: MLA with kv_b split per head into `attn_k_b` (`[n_head, latent,
                // nope]`, transposed for the absorbed query) and `attn_v_b` (`[n_head, v, latent]`),
                // per-head query scaling (SSMax), the DSA indexer.
                ("mistral-large4", _) => {
                    let n = match (base, suffix) {
                        ("attn_norm", _) => "input_layernorm.weight",
                        ("ffn_norm", _) => "post_attention_layernorm.weight",
                        ("attn_q", _) => "self_attn.q_proj.weight",
                        ("attn_kv_a_mqa", _) => "self_attn.kv_a_proj_with_mqa.weight",
                        ("attn_kv_a_norm", _) => "self_attn.kv_a_layernorm.weight",
                        ("attn_k_b", _) => "self_attn.k_b",
                        ("attn_v_b", _) => "self_attn.v_b",
                        ("attn_output", _) => "self_attn.o_proj.weight",
                        ("attn_ssmax", "bias") => "self_attn.softmax_scaling.bias",
                        ("attn_ssmax", _) => "self_attn.softmax_scaling.scale",
                        ("indexer.q_norm", "bias") => "self_attn.indexer.q_input_norm.bias",
                        ("indexer.q_norm", _) => "self_attn.indexer.q_input_norm.weight",
                        ("indexer.attn_q_b", _) => "self_attn.indexer.wq_b.weight",
                        ("indexer.attn_k", _) => "self_attn.indexer.wk.weight",
                        ("indexer.k_norm", "bias") => "self_attn.indexer.k_norm.bias",
                        ("indexer.k_norm", _) => "self_attn.indexer.k_norm.weight",
                        ("indexer.proj", _) => "self_attn.indexer.weights_proj.weight",
                        ("indexer.ssmax", "bias") => "self_attn.indexer.softmax_scaling.bias",
                        ("indexer.ssmax", _) => "self_attn.indexer.softmax_scaling.scale",
                        ("ffn_gate", _) => "mlp.gate_proj.weight",
                        ("ffn_up", _) => "mlp.up_proj.weight",
                        ("ffn_down", _) => "mlp.down_proj.weight",
                        ("ffn_gate_inp", _) => "mlp.gate.weight",
                        ("ffn_gate_exps", _) => "mlp.experts.gate_proj",
                        ("ffn_up_exps", _) => "mlp.experts.up_proj",
                        ("ffn_down_exps", _) => "mlp.experts.down_proj",
                        ("ffn_gate_shexp", _) => "mlp.shared_experts.gate_proj.weight",
                        ("ffn_up_shexp", _) => "mlp.shared_experts.up_proj.weight",
                        ("ffn_down_shexp", _) => "mlp.shared_experts.down_proj.weight",
                        _ => return None,
                    };
                    return Some((format!("model.layers.{l}.{n}"), shape));
                }
                (_, "attn_norm") => "input_layernorm.weight",
                ("qwen3" | "llama", "ffn_norm") => "post_attention_layernorm.weight",
                (_, "post_attention_norm") => "post_attention_layernorm.weight",
                (_, "attn_q") => "self_attn.q_proj.weight",
                (_, "attn_k") => "self_attn.k_proj.weight",
                (_, "attn_v") => "self_attn.v_proj.weight",
                (_, "attn_output") => "self_attn.o_proj.weight",
                (_, "attn_q_norm") => "self_attn.q_norm.weight",
                (_, "attn_k_norm") => "self_attn.k_norm.weight",
                (_, "attn_qkv") => "linear_attn.in_proj_qkv.weight",
                (_, "attn_gate") => "linear_attn.in_proj_z.weight",
                (_, "ssm_alpha") => "linear_attn.in_proj_a.weight",
                (_, "ssm_beta") => "linear_attn.in_proj_b.weight",
                (_, "ssm_out") => "linear_attn.out_proj.weight",
                (_, "ssm_norm") => "linear_attn.norm.weight",
                (_, "ssm_a") => "linear_attn.ssm_a",
                (_, "ssm_dt") if suffix == "bias" => "linear_attn.dt_bias",
                (_, "ssm_conv1d") => {
                    // ggml {K, C} -> HF Conv1d [C, 1, K].
                    shape = vec![t.ne[1], 1, t.ne[0]];
                    "linear_attn.conv1d.weight"
                }
                (_, "ffn_gate") => "mlp.gate_proj.weight",
                (_, "ffn_up") => "mlp.up_proj.weight",
                (_, "ffn_down") => "mlp.down_proj.weight",
                (_, "ffn_gate_inp") => "mlp.gate.weight",
                (_, "ffn_gate_exps") => "mlp.experts.gate_proj",
                (_, "ffn_up_exps") => "mlp.experts.up_proj",
                (_, "ffn_down_exps") => "mlp.experts.down_proj",
                (_, "ffn_gate_up_exps") => "mlp.experts.gate_up_proj",
                (_, "ffn_gate_shexp") => "mlp.shared_expert.gate_proj.weight",
                (_, "ffn_up_shexp") => "mlp.shared_expert.up_proj.weight",
                (_, "ffn_down_shexp") => "mlp.shared_expert.down_proj.weight",
                (_, "ffn_gate_inp_shexp") => {
                    shape = vec![1, t.ne[0]];
                    "mlp.shared_expert_gate.weight"
                }
                _ => return None,
            };
            if mtp {
                format!("mtp.layers.{}.{hf}", l - n_layers)
            } else {
                format!("model.layers.{l}.{hf}")
            }
        }
    };
    Some((name, shape))
}

/// The transformers `T5EncoderModel` name of a llama.cpp `t5encoder` tensor (others unchanged).
fn t5_hf_name(n: &str) -> String {
    match n {
        "token_embd.weight" => return "shared.weight".into(),
        "enc.output_norm.weight" => return "encoder.final_layer_norm.weight".into(),
        _ => {}
    }
    let Some((l, rest)) = n.strip_prefix("enc.blk.").and_then(|r| r.split_once('.')) else {
        return n.to_string();
    };
    let hf = match rest {
        "attn_q.weight" => "layer.0.SelfAttention.q.weight",
        "attn_k.weight" => "layer.0.SelfAttention.k.weight",
        "attn_v.weight" => "layer.0.SelfAttention.v.weight",
        "attn_o.weight" => "layer.0.SelfAttention.o.weight",
        "attn_rel_b.weight" => "layer.0.SelfAttention.relative_attention_bias.weight",
        "attn_norm.weight" => "layer.0.layer_norm.weight",
        "ffn_gate.weight" => "layer.1.DenseReluDense.wi_0.weight",
        "ffn_up.weight" => "layer.1.DenseReluDense.wi_1.weight",
        "ffn_down.weight" => "layer.1.DenseReluDense.wo.weight",
        "ffn_norm.weight" => "layer.1.layer_norm.weight",
        _ => return n.to_string(),
    };
    format!("encoder.block.{l}.{hf}")
}

/// A tensor of a GGUF file under its Hugging Face name and shape.
pub struct HfTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: u32,
    pub start: usize,
    pub end: usize,
}

/// The tensors of a GGUF file the loaders use, under Hugging Face names.
pub fn hf_tensors(h: &Header) -> Result<Vec<HfTensor>> {
    // Diffusion transformers (city96-style FLUX quants, unsloth Qwen-Image, calcuis "pig" files such
    // as OmniGen2), llama.cpp vision `clip` files (mmproj) and DFlash / DSpark drafts (`dflash`,
    // or `dspark` from llama.cpp's DSpark branch; see `crate::dspark`) keep their original tensor
    // names; T5 encoders (`t5encoder`) get the transformers names.
    if matches!(
        h.arch(),
        "flux" | "qwen_image" | "pig" | "omnigen2" | "t5encoder" | "clip" | "dflash" | "dspark"
    ) {
        let t5 = h.arch() == "t5encoder";
        // DeepSeek V4 DSpark drafts: the stages (and their head) named as the target's layers.
        let ds4 = h.arch() == "dflash" && h.get("dflash.hyper_connection.count").is_some();
        return Ok(h
            .tensors
            .iter()
            .map(|t| match crate::deepseek4::gguf_name(&t.name, &t.ne).filter(|_| ds4) {
                Some((name, shape)) => HfTensor {
                    name,
                    shape,
                    ggml_type: t.ggml_type,
                    start: t.start,
                    end: t.end,
                },
                None => HfTensor {
                    name: if t5 { t5_hf_name(&t.name) } else { t.name.clone() },
                    shape: t.ne.iter().rev().copied().collect(),
                    ggml_type: t.ggml_type,
                    start: t.start,
                    end: t.end,
                },
            })
            .collect());
    }
    h.model_type()?;
    Ok(hf_tensors_as(h, h.arch(), h.n_layers()))
}

/// [`hf_tensors`] for one file of a split model, named with the first file's architecture and
/// layer count (the other files carry only the `split.*` metadata).
/// The Qwen4-Exp name (relative to the layer, or absolute after a `/`) of a block tensor that is
/// not in the generic Qwen mapping; adjusts `shape` where the layouts differ.
fn qwen4exp_name(base: &str, suffix: &str, shape: &mut Vec<usize>, ne: &[usize]) -> Option<&'static str> {
    if suffix == "bias" {
        return None;
    }
    Some(match base {
        "hc_attn_norm" => "attn_hyper_connection.hc_norm.weight",
        "hc_attn_down" => "attn_hyper_connection.input_mix_weight_down.weight",
        "hc_attn_up" => "attn_hyper_connection.input_mix_weight_up.weight",
        "hc_attn_inject" => "attn_hyper_connection.block_inject_weight.weight",
        "hc_ffn_norm" => "mlp_hyper_connection.hc_norm.weight",
        "hc_ffn_down" => "mlp_hyper_connection.input_mix_weight_down.weight",
        "hc_ffn_up" => "mlp_hyper_connection.input_mix_weight_up.weight",
        "hc_ffn_inject" => "mlp_hyper_connection.block_inject_weight.weight",
        // Merged weights of llama.cpp's converter: `[a | b]` rows (not supported by the loader).
        "hc_attn_down_inject" => "attn_hyper_connection.down_inject.weight",
        "hc_ffn_down_inject" => "mlp_hyper_connection.down_inject.weight",
        "attn_qkv_gate" => "linear_attn.in_proj_qkvz.weight",
        "ssm_beta_alpha" => "linear_attn.in_proj_ba.weight",
        "indexer.q_proj" => "self_attn.indexer.q_proj.weight",
        "indexer.k_proj" => "self_attn.indexer.k_proj.weight",
        "indexer.q_norm" => "self_attn.indexer.q_layernorm.weight",
        "indexer.k_norm" => "self_attn.indexer.k_layernorm.weight",
        "ple_key" => "ple.key_proj.weight",
        "ple_value" => "ple.value_proj.weight",
        "ple_norm_key" => "ple.norm_key.weight",
        "ple_norm_query" => "ple.norm_query.weight",
        "ple_norm_conv" => "ple.norm_conv.weight",
        "ple_conv1d" => {
            // ggml {K, C} -> HF Conv1d [C, 1, K].
            *shape = vec![ne[1], 1, ne[0]];
            "ple.conv1d.weight"
        }
        "nextn.hc_head_norm" => "/mtp.hyper_connection_mixer.hc_norm.weight",
        "nextn.hc_head_down" => "/mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        "nextn.hc_head_up" => "/mtp.hyper_connection_mixer.input_mix_weight_up.weight",
        _ => return None,
    })
}

pub fn hf_tensors_as(h: &Header, arch: &str, n_layers: usize) -> Vec<HfTensor> {
    h.tensors
        .iter()
        .filter_map(|t| {
            hf_tensor(arch, t, n_layers).map(|(name, shape)| HfTensor {
                name,
                shape,
                ggml_type: t.ggml_type,
                start: t.start,
                end: t.end,
            })
        })
        .collect()
}

/// The `-NNNNN-of-MMMMM.gguf` part of a split GGUF file name: (prefix, index, count, digits).
pub fn split_name(name: &str) -> Option<(&str, usize, usize, usize)> {
    let stem = name.strip_suffix(".gguf")?;
    let (head, count) = stem.rsplit_once("-of-")?;
    let (prefix, idx) = head.rsplit_once('-')?;
    if idx.is_empty() || idx.len() != count.len() || !idx.bytes().chain(count.bytes()).all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((prefix, idx.parse().ok()?, count.parse().ok()?, idx.len()))
}

/// All files of the GGUF model `path` belongs to, in order: `path` alone, or for a split model
/// (`split.count` > 1, files `name-00001-of-00009.gguf` ...) every part, given any of them.
pub fn split_files(path: &Path) -> Result<Vec<std::path::PathBuf>> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let Some((prefix, _, count, digits)) = split_name(name) else {
        return Ok(vec![path.to_path_buf()]);
    };
    if count <= 1 {
        return Ok(vec![path.to_path_buf()]);
    }
    let dir = path.parent().unwrap_or(Path::new("."));
    (1..=count)
        .map(|i| {
            let p = dir.join(format!("{prefix}-{i:0digits$}-of-{count:0digits$}.gguf"));
            if p.is_file() {
                Ok(p)
            } else {
                bail!("split GGUF part {} is missing", p.display())
            }
        })
        .collect()
}

/// The first file of the GGUF model in folder `dir` (a single `.gguf`, or part 1 of a split one;
/// `mmproj*` vision files and `dspark*` drafts are not models), when the folder holds no other
/// checkpoint.
pub fn model_in_dir(dir: &Path) -> Option<std::path::PathBuf> {
    if !dir.is_dir() || dir.join("config.json").is_file() {
        return None;
    }
    let mut ggufs: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            n.ends_with(".gguf") && !n.starts_with("mmproj") && !n.starts_with("dspark") && split_name(n).is_none_or(|s| s.1 == 1)
        })
        .collect();
    ggufs.sort();
    match ggufs.as_slice() {
        [one] => Some(one.clone()),
        _ => None,
    }
}

/// String metadata worth keeping on the opened file (scalars only; arrays such as the vocabulary
/// are read from the header directly).
pub fn scalar_metadata(h: &Header) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for (k, v) in &h.meta {
        let s = match v {
            Value::Str(s) if s.len() < 256 => s.clone(),
            Value::U(x) => x.to_string(),
            Value::I(x) => x.to_string(),
            Value::F(x) => x.to_string(),
            Value::Bool(b) => b.to_string(),
            _ => continue,
        };
        m.insert(k.clone(), s);
    }
    m.insert("format".into(), "gguf".into());
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_str(b: &mut Vec<u8>, s: &str) {
        b.extend((s.len() as u64).to_le_bytes());
        b.extend(s.as_bytes());
    }

    /// The tokenizer rebuilt from a GGUF file against the checkpoint's own `tokenizer.json`, on a
    /// text file: `OXINFER_GGUF_TOK=FILE.gguf:MODEL_DIR:TEXT cargo test --release --lib -- --ignored
    /// gguf_tokenizer_matches`.
    #[test]
    #[ignore]
    fn gguf_tokenizer_matches_tokenizer_json() {
        let spec = std::env::var("OXINFER_GGUF_TOK").expect("OXINFER_GGUF_TOK=FILE.gguf:MODEL_DIR:TEXT");
        let parts: Vec<&str> = spec.split(':').collect();
        let ours = read_header(Path::new(parts[0])).unwrap().tokenizer().unwrap();
        let theirs = tokenizers::Tokenizer::from_file(Path::new(parts[1]).join("tokenizer.json")).unwrap();
        let text = std::fs::read_to_string(parts[2]).unwrap();
        let (mut lines, mut tokens) = (0, 0);
        for line in text.split_inclusive('\n').chain([text.as_str()]) {
            let a = ours.encode(line, false).unwrap();
            let b = theirs.encode(line, false).unwrap();
            assert_eq!(a.get_ids(), b.get_ids(), "{line:?}");
            assert_eq!(ours.decode(a.get_ids(), false).unwrap(), line);
            lines += 1;
            tokens += a.get_ids().len();
        }
        eprintln!("{lines} texts, {tokens} tokens: identical");
    }

    #[test]
    fn llama_config_and_names() {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(3u64.to_le_bytes()); // tensors
        b.extend(6u64.to_le_bytes()); // kv
        put_str(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        put_str(&mut b, "llama");
        for (k, v) in [
            ("block_count", 1u32),
            ("embedding_length", 64),
            ("attention.head_count", 4),
            ("attention.head_count_kv", 2),
        ] {
            put_str(&mut b, &format!("llama.{k}"));
            b.extend(4u32.to_le_bytes());
            b.extend(v.to_le_bytes());
        }
        put_str(&mut b, "llama.rope.freq_base");
        b.extend(6u32.to_le_bytes());
        b.extend(500000f32.to_le_bytes());
        for (name, ne, off) in [
            ("rope_freqs.weight", vec![8u64], 0u64),
            ("token_embd.weight", vec![64, 10], 32),
            ("blk.0.ffn_norm.weight", vec![64], 32 + 640 * 4),
        ] {
            put_str(&mut b, name);
            b.extend((ne.len() as u32).to_le_bytes());
            for d in ne {
                b.extend(d.to_le_bytes());
            }
            b.extend(0u32.to_le_bytes());
            b.extend(off.to_le_bytes());
        }
        let len = b.len().next_multiple_of(32) + 32 + 640 * 4 + 64 * 4;
        b.resize(len, 0);
        let h = parse_header(&b[..], len).unwrap();
        let c = h.config_json().unwrap();
        assert_eq!(c["model_type"], "llama");
        assert_eq!((c["head_dim"].as_u64(), c["num_key_value_heads"].as_u64()), (Some(16), Some(2)));
        assert_eq!((c["rope_theta"].as_f64(), c["rms_norm_eps"].as_f64()), (Some(500000.0), Some(1e-5)));
        assert_eq!(c["tie_word_embeddings"], true);
        let names: Vec<String> = hf_tensors(&h).unwrap().into_iter().map(|t| t.name).collect();
        assert_eq!(
            names,
            [
                "rope_freqs",
                "model.embed_tokens.weight",
                "model.layers.0.post_attention_layernorm.weight"
            ]
        );
    }

    #[test]
    fn parses_a_small_file_and_maps_names() {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(2u64.to_le_bytes()); // tensors
        b.extend(3u64.to_le_bytes()); // kv
        put_str(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        put_str(&mut b, "qwen35");
        put_str(&mut b, "qwen35.block_count");
        b.extend(4u32.to_le_bytes());
        b.extend(3u32.to_le_bytes());
        put_str(&mut b, "qwen35.nextn_predict_layers");
        b.extend(4u32.to_le_bytes());
        b.extend(1u32.to_le_bytes());
        for (name, ne, t, off) in [
            ("blk.0.ssm_conv1d.weight", vec![4u64, 6], 0u32, 0u64),
            ("blk.2.nextn.eh_proj.weight", vec![64, 32], 8, 96),
        ] {
            put_str(&mut b, name);
            b.extend((ne.len() as u32).to_le_bytes());
            for d in ne {
                b.extend(d.to_le_bytes());
            }
            b.extend(t.to_le_bytes());
            b.extend(off.to_le_bytes());
        }
        let data = b.len().next_multiple_of(32);
        let len = data + 96 + 32 * 2 * 34;
        b.resize(len, 0);
        let h = parse_header(&b[..], len).unwrap();
        assert_eq!(h.n_layers(), 2);
        let ts = hf_tensors(&h).unwrap();
        assert_eq!(ts[0].name, "model.layers.0.linear_attn.conv1d.weight");
        assert_eq!(ts[0].shape, vec![6, 1, 4]);
        assert_eq!((ts[0].start, ts[0].end), (data, data + 96));
        assert_eq!(ts[1].name, "mtp.fc.weight");
        assert_eq!(ts[1].shape, vec![32, 64]);
        assert_eq!(ts[1].end - ts[1].start, 32 * 2 * 34);
    }
}
