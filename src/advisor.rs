//! What runs on this machine: the models oxinfer was tested with, checked against the detected
//! memory, GPU and free disk space, with an estimate of the decoding speed (`oxinfer --models`).
//!
//! The estimate is the cost model the measurements follow: a decoded token reads every dense
//! weight once, so its time is the bytes read per token over the rate at which the kernel of that
//! weight format streams them, plus a fixed cost per layer (thread dispatch, attention, norms):
//!
//! ```text
//! seconds per token = bytes per token / rate(format) + layers * cost per layer
//! ```
//!
//! - On the CPU the rates are measured on the spot ([`calibrate`]): the real decode kernels of
//!   each format run on a random matrix for a fraction of a second. The cost per layer and the
//!   ratio between that rate and a model's are constants fitted on one machine ([`CPU_LAYER_MS`],
//!   [`CPU_FIT`]), so the estimate is rougher on a very different CPU.
//! - A GPU cannot be timed without loading a model, so its rates are those measured on one
//!   reference device per backend ([`GPU_REFS`]); on another device of that backend the estimate
//!   is marked as coming from the reference and may be far off.
//! - A checkpoint over 60% of the memory is streamed from storage ([`crate::stream`]): it runs,
//!   at a speed set by the storage, which is not estimated.
//!
//! What a model needs (download size, bytes read per token, layers) is a table of this file,
//! filled from the loader's own report on the checkpoints and from the README's model table.

use crate::platform::{DevicePref, GpuKind, Platform};
use std::path::{Path, PathBuf};

/// What a model is for. Only chat models get a speed estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Chat,
    Embedding,
    Image,
    Speech,
    Tts,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Chat => "chat",
            Kind::Embedding => "embedding",
            Kind::Image => "image",
            Kind::Speech => "speech-to-text",
            Kind::Tts => "text-to-speech",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "chat" | "llm" | "text" => Kind::Chat,
            "embedding" | "embeddings" | "embed" => Kind::Embedding,
            "image" | "images" => Kind::Image,
            "speech" | "speech-to-text" | "stt" | "whisper" => Kind::Speech,
            "tts" | "text-to-speech" => Kind::Tts,
            _ => return None,
        })
    }
}

/// How the weights read at every token are stored, i.e. which kernel streams them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// bf16 safetensors as stored.
    Bf16,
    /// oxinfer's int8 blocks (`--quant q8`).
    Q8,
    /// oxinfer's 4-bit blocks (`--quant q4`) and MLX 4-bit checkpoints.
    Q4,
    /// GGUF Q8_0.
    GgufQ8,
    /// GGUF Q4_K / Q5_K / Q6_K mixes (`Q4_K_M` and the like).
    GgufQ4K,
    /// GGUF Q4_0.
    GgufQ4,
    /// GGUF Q2_K / Q3_K mixes (`UD-Q2_K_XL`).
    GgufLow,
}

impl Format {
    pub const ALL: [Format; 7] = [
        Format::Bf16,
        Format::Q8,
        Format::Q4,
        Format::GgufQ8,
        Format::GgufQ4K,
        Format::GgufQ4,
        Format::GgufLow,
    ];
    fn index(self) -> usize {
        Format::ALL.iter().position(|f| *f == self).unwrap()
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Format::Bf16 => "bf16",
            Format::Q8 => "q8",
            Format::Q4 => "q4",
            Format::GgufQ8 => "gguf-q8_0",
            Format::GgufQ4K => "gguf-q4_k",
            Format::GgufQ4 => "gguf-q4_0",
            Format::GgufLow => "gguf-q2_k",
        }
    }
    /// The GGUF block type timed for this format.
    fn ggml(self) -> Option<&'static str> {
        match self {
            Format::GgufQ8 => Some("Q8_0"),
            Format::GgufQ4K => Some("Q4_K"),
            Format::GgufQ4 => Some("Q4_0"),
            Format::GgufLow => Some("Q2_K"),
            _ => None,
        }
    }
    /// Higher is closer to the original weights.
    fn quality(self) -> u8 {
        match self {
            Format::Bf16 => 4,
            Format::Q8 | Format::GgufQ8 => 3,
            Format::Q4 | Format::GgufQ4K | Format::GgufQ4 => 2,
            Format::GgufLow => 1,
        }
    }
}

/// `--quant q8` and `--quant q4` of a bf16 checkpoint read this fraction of its bytes per token
/// (measured: 0.85 / 1.50 and 0.42 / 1.50 GB on Qwen3.5-0.8B, 8.93 / 15.87 and 4.46 / 15.87 on
/// Qwen3.5-9B).
const Q8_OF_BF16: f64 = 0.565;
const Q4_OF_BF16: f64 = 0.281;

/// How a catalog entry is stored.
#[derive(Clone, Copy, Debug)]
pub enum Source {
    /// A bf16 (or fp16 / fp32) safetensors checkpoint: runs as stored, or quantized at load with
    /// `--quant q8` / `--quant q4` (the download is the same).
    Safetensors,
    /// A checkpoint in one fixed format (a GGUF file, an MLX 4-bit folder).
    Fixed(Format, &'static str),
}

/// What a model can do, as `--can` takes it and the table prints it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cap {
    /// Text in, text out.
    Text,
    ImageInput,
    VideoInput,
    AudioInput,
    /// A thinking block before the answer (`--think`).
    Thinking,
    /// Tool calls in the model's own format (`--tools`, the server's `tools`).
    Tools,
    /// Typed decisions (`--decide`, `/v1/decide`).
    Decisions,
    Embeddings,
    TextToImage,
    ImageToImage,
    /// Editing an image from an instruction.
    ImageEdit,
    Inpainting,
    /// A control image (edges, depth) or a ControlNet.
    Control,
    SpeechToText,
    /// Speech in any language to English text.
    Translation,
    TextToSpeech,
    VoiceCloning,
}

impl Cap {
    pub const ALL: [Cap; 17] = [
        Cap::Text,
        Cap::ImageInput,
        Cap::VideoInput,
        Cap::AudioInput,
        Cap::Thinking,
        Cap::Tools,
        Cap::Decisions,
        Cap::Embeddings,
        Cap::TextToImage,
        Cap::ImageToImage,
        Cap::ImageEdit,
        Cap::Inpainting,
        Cap::Control,
        Cap::SpeechToText,
        Cap::Translation,
        Cap::TextToSpeech,
        Cap::VoiceCloning,
    ];
    pub fn as_str(self) -> &'static str {
        match self {
            Cap::Text => "text",
            Cap::ImageInput => "image-input",
            Cap::VideoInput => "video-input",
            Cap::AudioInput => "audio-input",
            Cap::Thinking => "thinking",
            Cap::Tools => "tools",
            Cap::Decisions => "decisions",
            Cap::Embeddings => "embeddings",
            Cap::TextToImage => "text-to-image",
            Cap::ImageToImage => "image-to-image",
            Cap::ImageEdit => "image-edit",
            Cap::Inpainting => "inpainting",
            Cap::Control => "control",
            Cap::SpeechToText => "speech-to-text",
            Cap::Translation => "translation",
            Cap::TextToSpeech => "text-to-speech",
            Cap::VoiceCloning => "voice-cloning",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        Cap::ALL.into_iter().find(|c| c.as_str() == s)
    }
}

/// One tested model.
#[derive(Clone, Copy, Debug)]
pub struct Model {
    /// What `--model` takes.
    pub repo: &'static str,
    pub kind: Kind,
    pub caps: &'static [Cap],
    /// Total parameters, in billions.
    pub params_b: f64,
    pub source: Source,
    /// Bytes to download (all the files the model needs), in GB.
    pub download_gb: f64,
    /// Bytes read per decoded token as stored, in GB (the loader's `streaming N GB/token`); 0
    /// where no speed is estimated (the kinds other than chat, and the chat models whose bytes
    /// per token were not measured).
    pub token_gb: f64,
    /// Transformer layers of the language model (0 where no speed is estimated).
    pub layers: u16,
    /// Mixture of experts: a decoded token reads only the routed experts, and the routing costs
    /// more per layer.
    pub moe: bool,
    pub note: &'static str,
}

impl Model {
    const fn new(repo: &'static str, kind: Kind, params_b: f64, download_gb: f64, caps: &'static [Cap]) -> Model {
        Model {
            repo,
            kind,
            caps,
            params_b,
            source: Source::Safetensors,
            download_gb,
            token_gb: 0.0,
            layers: 0,
            moe: false,
            note: "",
        }
    }
    /// Bytes read per token (GB) and layers: the model gets a speed estimate.
    const fn speed(mut self, token_gb: f64, layers: u16) -> Model {
        self.token_gb = token_gb;
        self.layers = layers;
        self
    }
    /// Stored in one fixed format (a GGUF file, an MLX 4-bit folder, FP8 / NVFP4 weights).
    const fn fixed(mut self, format: Format, label: &'static str) -> Model {
        self.source = Source::Fixed(format, label);
        self
    }
    const fn moe(mut self) -> Model {
        self.moe = true;
        self
    }
    const fn note(mut self, note: &'static str) -> Model {
        self.note = note;
        self
    }
    pub fn can(&self, cap: Cap) -> bool {
        self.caps.contains(&cap)
    }
}

use Cap::*;
const QWEN_VL: &[Cap] = &[Text, ImageInput, VideoInput, Thinking, Tools];
const GEMMA_E: &[Cap] = &[Text, ImageInput, VideoInput, AudioInput, Thinking, Tools];
const GEMMA_L: &[Cap] = &[Text, ImageInput, Thinking, Tools];
const AGENT: &[Cap] = &[Text, Thinking, Tools];
const EMBED: &[Cap] = &[Embeddings];
const STT: &[Cap] = &[SpeechToText, Translation];
const EDIT: &[Cap] = &[ImageEdit];
const JANUS: &[Cap] = &[TextToImage, ImageInput, Text];
const PARAMS: &str = "bytes per token and layers from the parameter count";
const GLM_TINY: &str = "text only (no vision tower, no MTP); only its header was loaded, not its weights";

/// The models oxinfer was run with (the README's model table), by kind and size. What each can
/// do is what the README states for it. Bytes per token: the loader's report where the
/// checkpoint was loaded for the measurements of this file (Qwen3.5-0.8B / 9B, Qwen3-4B 4-bit,
/// Gemma 4 26B-A4B 4-bit, Llama 3.2 1B), the README's performance table for Gemma 4 and
/// Qwen3.8-27B, parameters x bits where the note says so; the other chat models get no speed
/// estimate.
pub const CATALOG: &[Model] = &[
    Model::new("Qwen/Qwen3.5-0.8B", Kind::Chat, 0.8, 1.7, QWEN_VL).speed(1.50, 24),
    Model::new("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", Kind::Chat, 1.2, 0.8, &[Text])
        .fixed(Format::GgufQ4K, "GGUF Q4_K_M")
        .speed(0.80, 16),
    Model::new("bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0", Kind::Chat, 1.2, 1.3, &[Text])
        .fixed(Format::GgufQ8, "GGUF Q8_0")
        .speed(1.31, 16),
    Model::new("prism-ml/Ternary-Bonsai-2-2B-gguf:PTQ1_0", Kind::Chat, 1.9, 0.43, AGENT)
        .fixed(Format::GgufLow, "GGUF PTQ1_0 (ternary)")
        .note("Hadamard-folded ternary weights (also :PQ2_0, :Q2_0); decodes on the CPU"),
    Model::new("Qwen/Qwen2.5-VL-3B-Instruct", Kind::Chat, 3.8, 7.5, &[Text, ImageInput]),
    Model::new("mlx-community/Qwen3-4B-4bit", Kind::Chat, 4.0, 2.4, AGENT)
        .fixed(Format::Q4, "MLX 4-bit")
        .speed(2.39, 36),
    Model::new("mlx-community/gemma-3-4b-it-4bit", Kind::Chat, 4.3, 3.4, &[Text]).fixed(Format::Q4, "MLX 4-bit"),
    Model::new("google/gemma-4-E2B-it", Kind::Chat, 5.1, 10.2, GEMMA_E).speed(4.6, 35),
    Model::new(
        "mlx-community/gemma-4-e2b-it-4bit",
        Kind::Chat,
        5.1,
        3.6,
        &[Text, ImageInput, AudioInput],
    )
    .fixed(Format::Q4, "MLX 4-bit"),
    Model::new("MiniMaxAI/SynLogic-7B", Kind::Chat, 7.6, 15.0, &[Text]).note("a Qwen2.5 fine-tune"),
    Model::new("google/gemma-4-E4B-it", Kind::Chat, 8.0, 16.0, GEMMA_E).speed(9.3, 42),
    Model::new(
        "mlx-community/gemma-4-e4b-it-4bit",
        Kind::Chat,
        8.0,
        5.2,
        &[Text, ImageInput, AudioInput],
    )
    .fixed(Format::Q4, "MLX 4-bit"),
    Model::new("Qwen/Qwen2.5-VL-7B-Instruct", Kind::Chat, 8.3, 16.6, &[Text, ImageInput]),
    Model::new("Qwen/Qwen3.5-9B", Kind::Chat, 9.0, 19.3, QWEN_VL).speed(15.87, 32),
    Model::new("autotrust/JEV-9B", Kind::Chat, 9.0, 18.2, &[Decisions, Text]).note("typed decisions; image decisions with its vl/ folder"),
    Model::new("google/gemma-4-12B-it", Kind::Chat, 12.0, 24.0, GEMMA_L).note("supported, untested"),
    Model::new("google/gemma-4-26B-A4B-it", Kind::Chat, 25.8, 52.0, GEMMA_L)
        .speed(7.6, 30)
        .moe(),
    Model::new("mlx-community/gemma-4-26b-a4b-it-4bit", Kind::Chat, 25.8, 15.4, &[Text, ImageInput])
        .fixed(Format::Q4, "MLX 4-bit")
        .speed(2.26, 30)
        .moe(),
    Model::new("ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0", Kind::Chat, 25.8, 14.6, &[Text])
        .fixed(Format::GgufQ4, "GGUF Q4_0")
        .moe(),
    Model::new("autotrust/GEV-26B-Decide", Kind::Chat, 25.8, 52.2, &[Decisions, Text, ImageInput])
        .moe()
        .note("typed decisions over text and images"),
    Model::new("MiniMaxAI/MiniMax-H3-Image-PE", Kind::Chat, 27.4, 18.0, &[Text, Thinking])
        .fixed(Format::Q4, "NVFP4")
        .note("the prompt enhancer of MiniMax-H3-Image (Qwen3.5-27B)"),
    Model::new("prism-ml/Ternary-Bonsai-2-27B-gguf:PTQ1_0", Kind::Chat, 27.4, 5.9, AGENT)
        .fixed(Format::GgufLow, "GGUF PTQ1_0 (ternary)")
        .note("Qwen3.8-27B in Hadamard-folded ternary weights (also :PQ2_0, :Q2_0); decodes on the CPU"),
    Model::new("Qwen/Qwen3.8-27B", Kind::Chat, 27.8, 55.6, QWEN_VL).speed(51.0, 64),
    Model::new("ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ2_XS-mtp", Kind::Chat, 27.8, 8.8, AGENT)
        .fixed(Format::GgufLow, "GGUF GSQ-RCO IQ2_XS")
        .note("a ggml type per tensor (1 to 6 bits), with the MTP layer for --mtp; also :IQ2_S, :IQ3_XXS, :IQ3_S"),
    Model::new("autotrust/JEV-27B", Kind::Chat, 27.8, 54.4, &[Decisions, Text]).note("typed decisions"),
    Model::new("autotrust/JEV-27B-VL", Kind::Chat, 27.8, 56.1, &[Decisions, Text, ImageInput]).note("typed decisions over text and images"),
    Model::new("google/gemma-4-31B-it", Kind::Chat, 31.0, 62.0, GEMMA_L).note("supported, untested"),
    Model::new("MiniMaxAI/SynLogic-32B", Kind::Chat, 32.8, 66.0, &[Text]).note("a Qwen2.5 fine-tune"),
    Model::new("Qwen/Qwen3.6-35B-A3B", Kind::Chat, 35.0, 69.0, AGENT)
        .speed(6.0, 40)
        .moe()
        .note("bytes per token from the active parameter count"),
    Model::new("unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL", Kind::Chat, 125.0, 111.0, AGENT)
        .fixed(Format::GgufQ4K, "GGUF UD-Q4_K_XL")
        .speed(3.8, 48)
        .moe()
        .note("29 GB of it an n-gram table read a few rows a token; bytes per token and layers from the parameter count"),
    Model::new("unsloth/MiniMax-M3.1-Flash-GGUF:UD-Q2_K_XL", Kind::Chat, 220.0, 144.0, AGENT)
        .fixed(Format::GgufLow, "GGUF UD-Q2_K_XL")
        .moe(),
    Model::new("unsloth/MiniMax-M2.7-GGUF:UD-Q4_K_XL", Kind::Chat, 229.0, 141.0, AGENT)
        .fixed(Format::GgufQ4K, "GGUF UD-Q4_K_XL")
        .moe(),
    Model::new("MiniMaxAI/MiniMax-M2.7", Kind::Chat, 229.0, 230.0, AGENT)
        .fixed(Format::Q8, "FP8")
        .moe(),
    Model::new("unsloth/DeepSeek-V4-Flash-GGUF:UD-Q2_K_XL", Kind::Chat, 284.0, 97.0, AGENT)
        .fixed(Format::GgufLow, "GGUF UD-Q2_K_XL")
        .speed(4.4, 48)
        .moe()
        .note(PARAMS),
    Model::new("unsloth/GLM-5.3-Flash-GGUF:UD-Q2_K_XL", Kind::Chat, 313.0, 109.0, AGENT)
        .fixed(Format::GgufLow, "GGUF UD-Q2_K_XL")
        .speed(8.95, 45)
        .moe()
        .note("text only (no vision tower, no MTP)"),
    Model::new("ggml-org/GLM-5.3-Flash-GGUF:Q4_K", Kind::Chat, 313.0, 181.0, AGENT)
        .fixed(Format::GgufQ4K, "GGUF Q4_K")
        .speed(13.9, 45)
        .moe()
        .note(GLM_TINY),
    Model::new("zai-org/GLM-5.3-Flash", Kind::Chat, 313.0, 328.0, AGENT)
        .fixed(Format::Q8, "FP8")
        .moe()
        .note(GLM_TINY),
    Model::new("nex-agi/Nex-N2.6", Kind::Chat, 313.0, 328.0, AGENT)
        .fixed(Format::Q8, "FP8")
        .moe()
        .note("a GLM-5.3-Flash fine-tune; text only; not run (the same architecture and template)"),
    Model::new("Scantist/GLM-5.3-SLIM-GGUF:UD-IQ2_M-E192", Kind::Chat, 573.0, 180.0, AGENT)
        .fixed(Format::GgufLow, "GGUF UD-IQ2_M")
        .speed(21.2, 78)
        .moe()
        .note("GLM-5.3 pruned to 192 experts; only its header was loaded, not its weights"),
    Model::new("unsloth/GLM-5.3-GGUF:UD-IQ1_S", Kind::Chat, 754.0, 217.0, AGENT)
        .fixed(Format::GgufLow, "GGUF UD-IQ1_S")
        .moe()
        .note("not run (GLM-5.3 was checked on tiny checkpoints and on the SLIM header)"),
    // ERNIE 5.5 Flash Base: a base model (completions, no chat template), never run on the real
    // weights (the agreement with Baidu's code is measured on tiny checkpoints; bytes per token
    // from the loader on the layout of the release).
    Model::new("baidu/ERNIE-5.5-Flash-Base", Kind::Chat, 280.0, 288.0, &[Text])
        .fixed(Format::Q8, "FP8")
        .speed(14.2, 43)
        .moe()
        .note("supported, untested; a base model: it continues a text, there is no chat template"),
    Model::new("baidu/ERNIE-5.5-Flash-Base-BF16", Kind::Chat, 280.0, 573.0, &[Text])
        .speed(26.3, 43)
        .moe()
        .note("supported, untested; a base model: it continues a text, there is no chat template"),
    Model::new(
        "MiniMaxAI/MiniMax-M3",
        Kind::Chat,
        427.0,
        854.0,
        &[Text, ImageInput, Thinking, Tools],
    )
    .moe(),
    Model::new("MiniMaxAI/MiniMax-Text-01", Kind::Chat, 456.0, 912.0, &[Text, Tools]).moe(),
    Model::new("MiniMaxAI/MiniMax-VL-01", Kind::Chat, 456.0, 913.0, &[Text, ImageInput, Tools]).moe(),
    // DeepSeek V4.1 Flash: text only here (no vision tower), never run on the real weights (the
    // agreement with DeepSeek's code is measured on tiny checkpoints). Sizes with the 111 GB of
    // engram tables (IQ4_NL), which every conversion needs.
    Model::new("unsloth/DeepSeek-V4.1-Flash-1017-GGUF:UD-IQ1_S", Kind::Chat, 748.0, 233.0, AGENT)
        .fixed(Format::GgufLow, "GGUF UD-IQ1_S")
        .moe()
        .note("supported, untested; 111 GB of it engram tables read a few rows a token; text only"),
    Model::new(
        "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF:IQ2_XXS",
        Kind::Chat,
        748.0,
        183.0,
        AGENT,
    )
    .fixed(Format::GgufLow, "GGUF IQ2_XXS")
    .moe()
    .note("supported, untested; 192 of 384 experts kept; fetch DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf (111 GB, the repository's root) by hand"),
    Model::new(
        "AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-GGUF:IQ1_M",
        Kind::Chat,
        748.0,
        232.0,
        AGENT,
    )
    .fixed(Format::GgufLow, "GGUF IQ1_M")
    .moe()
    .note("supported, untested; needs DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf (111 GB) of ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF beside it"),
    Model::new(
        "unsloth/Mistral-Large-4-1T-A52B-GGUF:UD-Q2_K_XL",
        Kind::Chat,
        1000.0,
        378.0,
        &[Text, ImageInput, Tools],
    )
    .fixed(Format::GgufLow, "GGUF UD-Q2_K_XL")
    .speed(19.6, 48)
    .moe()
    .note(PARAMS),
    Model::new("sentence-transformers/all-MiniLM-L6-v2", Kind::Embedding, 0.023, 0.09, EMBED),
    Model::new("BAAI/bge-small-en-v1.5", Kind::Embedding, 0.033, 0.13, EMBED),
    Model::new(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        Kind::Embedding,
        0.118,
        0.47,
        EMBED,
    ),
    Model::new("intfloat/multilingual-e5-small", Kind::Embedding, 0.118, 0.47, EMBED).note("size from the parameter count"),
    Model::new("google/embeddinggemma-300m", Kind::Embedding, 0.3, 1.2, EMBED),
    Model::new("mixedbread-ai/mxbai-embed-large-v1", Kind::Embedding, 0.335, 1.3, EMBED),
    Model::new("Qwen/Qwen3-Embedding-0.6B", Kind::Embedding, 0.6, 1.2, EMBED),
    Model::new("BAAI/bge-m3", Kind::Embedding, 0.6, 2.3, EMBED).note("dense, sparse and ColBERT"),
    Model::new("NovaSearch/stella_en_1.5B_v5", Kind::Embedding, 1.5, 6.2, EMBED).note("a head per output size (--dim)"),
    Model::new("Alibaba-NLP/gte-Qwen2-1.5B-instruct", Kind::Embedding, 1.5, 7.1, EMBED),
    Model::new("openai/whisper-tiny", Kind::Speech, 0.039, 0.15, STT),
    Model::new("openai/whisper-base", Kind::Speech, 0.074, 0.29, STT),
    Model::new("openai/whisper-small", Kind::Speech, 0.24, 0.97, STT),
    Model::new("distil-whisper/distil-large-v3", Kind::Speech, 0.756, 1.5, &[SpeechToText]).note("English"),
    Model::new("openai/whisper-medium", Kind::Speech, 0.769, 3.1, STT),
    Model::new("openai/whisper-large-v3-turbo", Kind::Speech, 0.809, 1.6, &[SpeechToText]),
    Model::new("openai/whisper-large-v3", Kind::Speech, 1.55, 3.1, STT),
    Model::new("isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF", Kind::Tts, 3.3, 2.5, &[TextToSpeech]).note("with hubertsiuzdak/snac_24khz"),
    Model::new("lex-au/Orpheus-3b-FT-Q8_0.gguf", Kind::Tts, 3.3, 4.1, &[TextToSpeech]).note("with hubertsiuzdak/snac_24khz"),
    Model::new("unsloth/orpheus-3b-0.1-ft", Kind::Tts, 3.3, 6.7, &[TextToSpeech]).note("with hubertsiuzdak/snac_24khz"),
    Model::new(
        "unsloth/orpheus-3b-0.1-pretrained",
        Kind::Tts,
        3.3,
        6.7,
        &[TextToSpeech, VoiceCloning],
    )
    .note("voice cloning from a reference clip"),
    Model::new("Zyphra/ZONOS3-Mini", Kind::Tts, 0.8, 1.2, &[TextToSpeech, VoiceCloning])
        .fixed(Format::GgufQ4, "GGUF Q4_0 / Q8_0")
        .note("44.1 kHz, 43 languages, 27 voices; cloning from a clip's speaker embedding; on the CPU"),
    Model::new(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        Kind::Image,
        1.1,
        2.0,
        &[TextToImage, ImageToImage, Inpainting],
    ),
    Model::new(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        Kind::Image,
        1.1,
        2.0,
        &[Inpainting],
    ),
    Model::new("timbrooks/instruct-pix2pix", Kind::Image, 1.1, 2.0, EDIT),
    Model::new("vinesmsuic/magicbrush-jul7", Kind::Image, 1.1, 4.3, EDIT),
    Model::new("stabilityai/sd-turbo", Kind::Image, 1.3, 2.6, &[TextToImage, ImageToImage]).note("one step"),
    Model::new("deepseek-ai/JanusFlow-1.3B", Kind::Image, 2.0, 4.4, JANUS).note("384x384"),
    Model::new("deepseek-ai/Janus-Pro-1B", Kind::Image, 2.1, 4.2, JANUS).note("384x384"),
    Model::new("stabilityai/sdxl-turbo", Kind::Image, 3.5, 7.0, &[TextToImage, ImageToImage]).note("fp16 files"),
    Model::new(
        "stabilityai/stable-diffusion-xl-base-1.0",
        Kind::Image,
        3.5,
        7.0,
        &[TextToImage, ImageToImage, Inpainting],
    )
    .note("fp16 files"),
    Model::new("deepseek-ai/Janus-Pro-7B", Kind::Image, 7.4, 14.8, JANUS).note("384x384"),
    Model::new(
        "v2ray/stable-diffusion-3-medium-diffusers",
        Kind::Image,
        7.2,
        15.3,
        &[TextToImage, ImageToImage],
    )
    .note("with T5-XXL; --no-t5 saves 9.5 GB"),
    Model::new(
        "adamo1139/stable-diffusion-3.5-medium-ungated",
        Kind::Image,
        7.6,
        16.0,
        &[TextToImage, ImageToImage],
    )
    .note("with T5-XXL; --no-t5 saves 9.5 GB"),
    Model::new("city96/FLUX.1-schnell-gguf", Kind::Image, 12.0, 16.6, &[TextToImage])
        .note("Q4_0 transformer + T5 / CLIP / VAE of --base; 4 steps"),
    Model::new(
        "city96/FLUX.1-dev-gguf",
        Kind::Image,
        12.0,
        16.6,
        &[TextToImage, ImageToImage, Inpainting, Control],
    )
    .note("Q4_0 transformer + T5 / CLIP / VAE of --base; LoRAs, Canny / Depth control LoRAs, Redux"),
    Model::new(
        "QuantStack/FLUX.1-Kontext-dev-GGUF",
        Kind::Image,
        17.0,
        16.6,
        &[ImageEdit, TextToImage],
    )
    .note("Q4_0 transformer + T5 / CLIP / VAE of unsloth/FLUX.1-Kontext-dev"),
    Model::new(
        "ByteDance-Seed/BAGEL-7B-MoT",
        Kind::Image,
        14.0,
        29.5,
        &[ImageEdit, TextToImage, ImageInput, Text, Thinking],
    ),
    Model::new(
        "OmniGen2/OmniGen2",
        Kind::Image,
        7.8,
        31.2,
        &[ImageEdit, TextToImage, ImageInput, Text],
    )
    .note("up to 5 input images"),
    Model::new("unsloth/FLUX.1-Kontext-dev", Kind::Image, 17.0, 33.3, &[ImageEdit, TextToImage]),
    Model::new("stepfun-ai/Step1X-Edit-v1p1-diffusers", Kind::Image, 20.7, 41.8, EDIT),
    Model::new("stepfun-ai/Step1X-Edit-v1p2", Kind::Image, 20.7, 41.8, &[ImageEdit, Thinking]),
    Model::new(
        "Qwen/Qwen-Image-2512",
        Kind::Image,
        28.3,
        57.9,
        &[TextToImage, ImageToImage, Inpainting, Control],
    )
    .note("ControlNet: InstantX/Qwen-Image-ControlNet-Union"),
    Model::new("Qwen/Qwen-Image-Edit-2511", Kind::Image, 28.3, 57.9, &[ImageEdit, Inpainting]).note("one or more input images"),
    Model::new("Qwen/Qwen-Image-2.1-Turbo", Kind::Image, 16.2, 32.4, &[TextToImage, ImageEdit])
        .note("8 steps; GGUF transformers: abenzerps/Qwen-Image-2.1-Turbo-Quantized"),
    Model::new("Qwen/Qwen-Image-2.1", Kind::Image, 16.2, 33.1, &[TextToImage, ImageEdit]).note("40 steps; RGBA output"),
    Model::new("MiniMaxAI/MiniMax-H3-Image", Kind::Image, 63.0, 127.2, &[TextToImage])
        .note("with MiniMax-H3's Qwen3-VL text encoder; 2048x2048 and up"),
];

/// Fixed CPU cost per layer and decoded token, in milliseconds, on top of streaming the weights.
///
/// This and [`CPU_FIT`] were fitted on a Threadripper 9970X (32 cores, 4-channel DDR5) against
/// 200 greedy tokens of ten checkpoints. Measured and predicted tokens/s there, with the rates
/// [`calibrate`] gave (bf16 74, q8 80, q4 94, GGUF Q8_0 84, Q4_K 96 GB/s):
///
/// | checkpoint | measured | predicted |
/// |---|---|---|
/// | Qwen3.5-0.8B bf16 / q8 / q4 | 33.0 / 49.1 / 73.2 | 39.5 / 55 / 71.5 |
/// | Qwen3.5-9B bf16 / q8 / q4 | 5.9 / 9.7 / 14.6 | 5.3 / 8.9 / 14.2 |
/// | Qwen3-4B MLX 4-bit | 19.5 | 22.5 |
/// | Llama 3.2 1B GGUF Q4_K_M / Q8_0 | 60.7 / 39.4 | 60.3 / 39.5 |
/// | Gemma 4 26B-A4B MLX 4-bit (MoE) | 15.9 | 15.9 (fixes [`CPU_MOE_LAYER_MS`]) |
pub const CPU_LAYER_MS: f64 = 0.35;
/// The same for a mixture-of-experts layer (router, top-k, one small matmul group per expert):
/// fitted on Gemma 4 26B-A4B 4-bit alone.
pub const CPU_MOE_LAYER_MS: f64 = 1.1;
/// The rate a format's kernel reaches inside a model, over the rate [`calibrate`] measures on
/// its synthetic matrix, per [`Format::ALL`] entry (the fit above; GGUF Q4_0 and Q2_K were not
/// measured on a model and take the other GGUF formats' factor).
pub const CPU_FIT: [f64; 7] = [1.2, 1.1, 0.8, 0.78, 0.78, 0.78, 0.78];

/// The speed of GPU decoding as measured on one device of a backend.
#[derive(Clone, Copy, Debug)]
pub struct GpuRef {
    pub kind: GpuKind,
    /// The device measured (matched against the detected device's name).
    pub device: &'static str,
    /// Milliseconds per layer and token.
    pub layer_ms: f64,
    /// GB/s per [`Format::ALL`] entry; 0 where the backend's decode plans do not take the format
    /// (decoding then runs on the CPU).
    pub rates: [f64; 7],
    /// Whether mixture-of-experts models decode faster on it than on the CPU.
    pub moe: bool,
}

/// - CUDA: an RTX 5080, 200 greedy tokens of Qwen3.5-0.8B and 9B (bf16, q8, q4) and Qwen3-4B
///   4-bit; GGUF files and Gemma 4 26B-A4B decoded at the CPU's speed there.
/// - Metal: the README's M5 Max table (Llama 3.2 1B Q4_K_M / Q8_0, Qwen3.5-0.8B q8).
/// - Vulkan runs prefill matmuls only.
pub const GPU_REFS: &[GpuRef] = &[
    GpuRef {
        kind: GpuKind::Cuda,
        device: "RTX 5080",
        layer_ms: 0.06,
        rates: [850.0, 740.0, 540.0, 0.0, 0.0, 0.0, 0.0],
        moe: false,
    },
    GpuRef {
        kind: GpuKind::Metal,
        device: "M5 Max",
        layer_ms: 0.03,
        rates: [400.0; 7],
        moe: true,
    },
];

/// Weights over this fraction of the memory are streamed from storage (the engine's rule,
/// [`crate::stream::DeviceMemory::too_big`]).
const RAM_FRACTION: f64 = 0.6;
/// GPU decoding keeps the weights on the device: they must fit in this fraction of its memory
/// with [`GPU_HEADROOM_GB`] to spare (activations, the KV cache, the desktop).
const GPU_FRACTION: f64 = 0.92;
const GPU_HEADROOM_GB: f64 = 1.0;

/// The GPU the estimates are for.
#[derive(Clone, Debug)]
pub struct Gpu {
    pub kind: GpuKind,
    pub name: String,
    pub memory_gb: Option<f64>,
    pub unified: bool,
}

/// What was detected (or is assumed, for a what-if).
#[derive(Clone, Debug)]
pub struct Hardware {
    pub summary: String,
    pub cores: usize,
    pub ram_gb: Option<f64>,
    pub ram_available_gb: Option<f64>,
    pub gpu: Option<Gpu>,
    /// Where repo ids are downloaded, and the space left there.
    pub cache_dir: PathBuf,
    pub disk_free_gb: Option<f64>,
    /// CPU streaming rate per [`Format::ALL`] entry, GB/s ([`calibrate`]).
    pub cpu_rates: [f64; 7],
}

/// `$OXINFER_CACHE`, else `~/.cache/oxinfer/hub` (as [`crate::hf::cache_root`], without the
/// `hf` feature).
pub fn default_cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("OXINFER_CACHE") {
        return PathBuf::from(p);
    }
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()));
    if cfg!(target_os = "ios") {
        return home.join("Library").join("Application Support").join("oxinfer").join("hub");
    }
    home.join(".cache").join("oxinfer").join("hub")
}

/// Times the single-token kernel of every format on a random matrix of 268M weights (67M on
/// a machine with less than 8 GB), larger than the caches so that memory feeds the kernel as it
/// does in a model: about 0.1 s per format, the best run counts, with the current thread pool.
/// GB of weights streamed per second; the whole takes 2 to 3 seconds and up to 0.9 GB.
pub fn calibrate() -> [f64; 7] {
    use crate::clock::Instant;
    const N: usize = 8192;
    let small = crate::platform::total_memory().is_some_and(|m| m < 8 << 30);
    let rows: usize = if small { N } else { 4 * N };
    let w: Vec<u16> = (0..rows * N)
        .map(|i| ((((i as u32).wrapping_mul(2654435761) >> 8) as f32 / (1u32 << 24) as f32 - 0.5).to_bits() >> 16) as u16)
        .collect();
    let x: Vec<f32> = (0..N).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
    let mut y = vec![0f32; rows];
    let mut time = |bytes: usize, run: &mut dyn FnMut(&mut [f32])| -> f64 {
        let (mut best, start) = (f64::MAX, Instant::now());
        let mut reps = 0;
        while reps < 3 || (start.elapsed().as_secs_f64() < 0.1 && reps < 200) {
            let t = Instant::now();
            run(&mut y);
            best = best.min(t.elapsed().as_secs_f64());
            reps += 1;
        }
        std::hint::black_box(&y);
        bytes as f64 / best.max(1e-9) / 1e9
    };
    let mut rates = [0f64; 7];
    rates[Format::Bf16.index()] = time(w.len() * 2, &mut |y| crate::kernels::matmul(&w, rows, N, &x, 1, y));
    {
        let q8 = crate::quant::quantize_bf16(&w, rows, N);
        rates[Format::Q8.index()] = time(q8.bytes(), &mut |y| crate::kernels::matmul_quant_multi(&mut [(&q8, y)], &x, 1));
    }
    {
        let q4 = crate::quant::quantize_q4(&w, rows, N);
        rates[Format::Q4.index()] = time(q4.bytes(), &mut |y| crate::kernels::matmul_quant_multi(&mut [(&q4, y)], &x, 1));
    }
    drop(w);
    for f in Format::ALL {
        let Some(name) = f.ggml() else { continue };
        let Some(kind) = crate::ggml::GgmlType::ALL.into_iter().find(|t| t.name() == name) else {
            continue;
        };
        let raw = crate::ggml::random_blocks(kind, rows * N / kind.block_size(), &mut 1);
        let Ok(m) = crate::ggml::GgufMatrix::from_bytes(kind, &raw, rows, N) else {
            continue;
        };
        drop(raw);
        rates[f.index()] = time(m.bytes(), &mut |y| crate::kernels::matmul_quant_multi(&mut [(&m, y)], &x, 1));
    }
    rates
}

impl Hardware {
    /// Detects this machine and times its CPU kernels ([`calibrate`]). `device` restricts the
    /// estimates as `--device` restricts a run (`cpu`: no GPU; a backend by name: that one).
    pub fn detect(device: DevicePref, cache_dir: Option<&Path>) -> Self {
        let plat = crate::platform::detect();
        let mut hw = Self::from_platform(&plat, device, cache_dir);
        hw.cpu_rates = calibrate();
        hw
    }

    /// The same without timing anything (`cpu_rates` all zero: set them, or no CPU speed is
    /// estimated).
    pub fn from_platform(plat: &Platform, device: DevicePref, cache_dir: Option<&Path>) -> Self {
        let gpu = match device {
            DevicePref::Cpu => None,
            DevicePref::Auto => plat.auto_order().next(),
            DevicePref::Gpu(_) => plat.pick(device),
        }
        .map(|g| Gpu {
            kind: g.kind,
            name: g.name.clone(),
            memory_gb: g.memory_bytes.map(|b| b as f64 / 1e9),
            unified: g.unified_memory,
        });
        let cache_dir = cache_dir.map(Path::to_path_buf).unwrap_or_else(default_cache_dir);
        Hardware {
            summary: plat.summary(),
            cores: plat.cpu.cores,
            ram_gb: plat.cpu.memory_bytes.map(|b| b as f64 / 1e9),
            ram_available_gb: crate::platform::available_memory().map(|b| b as f64 / 1e9),
            gpu,
            disk_free_gb: crate::platform::free_disk(&cache_dir).map(|b| b as f64 / 1e9),
            cache_dir,
            cpu_rates: [0.0; 7],
        }
    }

    fn gpu_ref(&self) -> Option<(&'static GpuRef, bool)> {
        let g = self.gpu.as_ref()?;
        let r = GPU_REFS.iter().find(|r| r.kind == g.kind)?;
        Some((r, g.name.contains(r.device)))
    }
}

/// Where a variant decodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Runs {
    /// Whole decode steps on the GPU.
    Gpu,
    /// In memory, decoding on the CPU.
    Cpu,
    /// Larger than 60% of the memory: weights read from storage as they are needed.
    Streamed,
    /// Fits in memory; where it computes is not looked at (the kinds without a speed estimate:
    /// they use the GPU backend for their matmuls when there is one).
    InMemory,
}

impl Runs {
    pub fn as_str(self) -> &'static str {
        match self {
            Runs::Gpu => "gpu",
            Runs::Cpu => "cpu",
            Runs::Streamed => "streamed",
            Runs::InMemory => "memory",
        }
    }
}

/// What a speed estimate rests on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    /// CPU kernels timed on this machine.
    Calibrated,
    /// GPU rates measured on this very device model.
    Reference,
    /// GPU rates measured on another device of the backend (named in [`GpuRef::device`]).
    OtherDevice,
}

impl Basis {
    pub fn as_str(self) -> &'static str {
        match self {
            Basis::Calibrated => "cpu-calibration",
            Basis::Reference => "gpu-reference",
            Basis::OtherDevice => "gpu-reference-other-device",
        }
    }
}

/// One way to run a model on the hardware.
#[derive(Clone, Debug)]
pub struct Variant {
    /// `bf16`, `--quant q8`, `GGUF Q4_K_M`, ...
    pub label: String,
    pub format: Format,
    /// Extra command-line arguments after `--model REPO` (`--quant q8`).
    pub args: &'static str,
    /// Bytes in memory while it runs.
    pub memory_gb: f64,
    pub token_gb: f64,
    pub runs: Runs,
    /// Estimated decoding speed; None where it is not estimated (streamed models, the kinds
    /// other than chat, a CPU that was not calibrated).
    pub tok_s: Option<f64>,
    pub basis: Option<Basis>,
    /// The one to pick: the most precise variant decoding in memory at 10 tokens/s or more,
    /// else the fastest in memory, else the smallest.
    pub recommended: bool,
}

/// A catalog entry on the hardware.
#[derive(Clone, Debug)]
pub struct Row {
    pub model: &'static Model,
    /// Already in the cache directory.
    pub on_disk: bool,
    /// Not on disk and more to download than there is free space.
    pub disk_short_gb: Option<f64>,
    pub variants: Vec<Variant>,
}

fn variants_of(m: &Model) -> Vec<(String, Format, &'static str, f64, f64)> {
    // (label, format, args, memory, bytes per token)
    match m.source {
        Source::Fixed(f, label) => vec![(label.to_string(), f, "", m.download_gb, m.token_gb)],
        // Mixture of experts: every expert is quantized, so the whole checkpoint shrinks even
        // where the bytes per token (and so the speed) are not known.
        Source::Safetensors if m.token_gb == 0.0 && m.moe && m.kind == Kind::Chat => vec![
            ("bf16".to_string(), Format::Bf16, "", m.download_gb, 0.0),
            ("--quant q8".to_string(), Format::Q8, "--quant q8", m.download_gb * Q8_OF_BF16, 0.0),
            ("--quant q4".to_string(), Format::Q4, "--quant q4", m.download_gb * Q4_OF_BF16, 0.0),
        ],
        Source::Safetensors if m.token_gb == 0.0 => vec![("as stored".to_string(), Format::Bf16, "", m.download_gb, 0.0)],
        Source::Safetensors => {
            // Dense models: what is not read per token (embedding tables, vision and audio towers)
            // is counted as stored. Mixture of experts: the experts a token does not route to
            // are quantized like the others, so the whole checkpoint shrinks.
            let memory = |ratio: f64| {
                if m.moe {
                    m.download_gb * ratio
                } else {
                    (m.download_gb - m.token_gb).max(0.0) + m.token_gb * ratio
                }
            };
            vec![
                ("bf16".to_string(), Format::Bf16, "", m.download_gb, m.token_gb),
                (
                    "--quant q8".to_string(),
                    Format::Q8,
                    "--quant q8",
                    memory(Q8_OF_BF16),
                    m.token_gb * Q8_OF_BF16,
                ),
                (
                    "--quant q4".to_string(),
                    Format::Q4,
                    "--quant q4",
                    memory(Q4_OF_BF16),
                    m.token_gb * Q4_OF_BF16,
                ),
            ]
        }
    }
}

/// Tokens per second of a decode streaming `token_gb` at `rate` GB/s over `layers` layers.
fn speed(token_gb: f64, rate: f64, layers: u16, layer_ms: f64) -> Option<f64> {
    (rate > 0.0 && token_gb > 0.0).then(|| 1.0 / (token_gb / rate + layers as f64 * layer_ms / 1e3))
}

/// The repo folder of the cache (`org--name`), for a `repo` or `repo:FILE` spec.
fn cache_folder(repo: &str) -> String {
    repo.split(':').next().unwrap_or(repo).replace('/', "--")
}

impl Row {
    fn new(m: &'static Model, hw: &Hardware) -> Row {
        let on_disk = hw.cache_dir.join(cache_folder(m.repo)).is_dir();
        let disk_short_gb = match hw.disk_free_gb {
            Some(free) if !on_disk && m.download_gb > free => Some(m.download_gb - free),
            _ => None,
        };
        let mut variants: Vec<Variant> = variants_of(m)
            .into_iter()
            .map(|(label, format, args, memory_gb, token_gb)| {
                let in_ram = hw.ram_gb.is_none_or(|r| memory_gb <= RAM_FRACTION * r);
                let cpu = speed(
                    token_gb,
                    hw.cpu_rates[format.index()] * CPU_FIT[format.index()],
                    m.layers,
                    if m.moe { CPU_MOE_LAYER_MS } else { CPU_LAYER_MS },
                );
                let gpu = hw.gpu_ref().and_then(|(r, same)| {
                    let mem = hw.gpu.as_ref()?.memory_gb?;
                    let fits = memory_gb + GPU_HEADROOM_GB <= GPU_FRACTION * mem;
                    let s = speed(token_gb, r.rates[format.index()], m.layers, r.layer_ms)?;
                    (fits && (r.moe || !m.moe)).then_some((s, if same { Basis::Reference } else { Basis::OtherDevice }))
                });
                let (runs, tok_s, basis) = match (in_ram, gpu, cpu) {
                    (false, _, _) => (Runs::Streamed, None, None),
                    (true, _, _) if token_gb == 0.0 => (Runs::InMemory, None, None),
                    (true, Some((g, b)), c) if c.is_none_or(|c| g > c) => (Runs::Gpu, Some(g), Some(b)),
                    (true, _, c) => (Runs::Cpu, c, c.map(|_| Basis::Calibrated)),
                };
                Variant {
                    label,
                    format,
                    args,
                    memory_gb,
                    token_gb,
                    runs,
                    tok_s,
                    basis,
                    recommended: false,
                }
            })
            .collect();
        let in_memory = |v: &Variant| v.runs != Runs::Streamed;
        // (A model that only streams: its smallest variant.)
        let pick = variants
            .iter()
            .enumerate()
            .filter(|(_, v)| in_memory(v) && v.tok_s.is_none_or(|s| s >= 10.0))
            .max_by_key(|(_, v)| v.format.quality())
            .or_else(|| {
                variants
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| in_memory(v))
                    .max_by(|a, b| a.1.tok_s.unwrap_or(0.0).total_cmp(&b.1.tok_s.unwrap_or(0.0)))
            })
            .or_else(|| variants.iter().enumerate().min_by(|a, b| a.1.memory_gb.total_cmp(&b.1.memory_gb)))
            .map(|(i, _)| i);
        if let Some(i) = pick {
            variants[i].recommended = true;
        }
        Row {
            model: m,
            on_disk,
            disk_short_gb,
            variants,
        }
    }

    pub fn recommended(&self) -> &Variant {
        self.variants.iter().find(|v| v.recommended).unwrap_or(&self.variants[0])
    }
}

/// Every catalog entry on `hw`, in catalog order (by kind, then size).
pub fn survey(hw: &Hardware) -> Vec<Row> {
    CATALOG.iter().map(|m| Row::new(m, hw)).collect()
}

fn gb(v: f64) -> String {
    if v < 10.0 { format!("{v:.1} GB") } else { format!("{v:.0} GB") }
}

fn rate(v: f64) -> String {
    if v < 10.0 { format!("{v:.1}") } else { format!("{v:.0}") }
}

/// The table `oxinfer --models` prints: one line per model with its recommended variant, or one
/// per variant with `all`.
pub fn render(hw: &Hardware, rows: &[Row], all: bool) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let opt = |v: Option<f64>| v.map_or("unknown".to_string(), gb);
    let _ = writeln!(s, "{}", hw.summary);
    let _ = writeln!(
        s,
        "memory {} ({} available), {} free in {}",
        opt(hw.ram_gb),
        opt(hw.ram_available_gb),
        opt(hw.disk_free_gb),
        hw.cache_dir.display()
    );
    match (&hw.gpu, hw.gpu_ref()) {
        (Some(g), Some((r, same))) => {
            let _ = writeln!(
                s,
                "GPU estimates: {} {}, {}{}",
                g.kind.as_str(),
                g.name,
                opt(g.memory_gb),
                if same {
                    String::new()
                } else {
                    format!(" (rates measured on a {}: marked ?)", r.device)
                }
            );
        }
        (Some(g), None) => {
            let _ = writeln!(
                s,
                "GPU: {} {} runs prefill only: decoding is estimated on the CPU",
                g.kind.as_str(),
                g.name
            );
        }
        (None, _) => {}
    }
    let fmts: Vec<String> = Format::ALL
        .iter()
        .filter(|f| hw.cpu_rates[f.index()] > 0.0)
        .map(|f| format!("{} {}", f.as_str(), rate(hw.cpu_rates[f.index()])))
        .collect();
    if !fmts.is_empty() {
        let _ = writeln!(s, "CPU kernels, GB/s: {}", fmts.join(", "));
    }
    let mut lines: Vec<[String; 7]> = Vec::new();
    let mut kind = None;
    for r in rows {
        if kind != Some(r.model.kind) {
            kind = Some(r.model.kind);
            lines.push([
                format!("\n{}", r.model.kind.as_str().to_uppercase()),
                "".into(),
                "DISK".into(),
                "MEMORY".into(),
                "RUNS ON".into(),
                "DECODE".into(),
                "CAN".into(),
            ]);
        }
        for (i, v) in r.variants.iter().filter(|v| all || v.recommended).enumerate() {
            let disk = match (r.on_disk, r.disk_short_gb) {
                (true, _) => "on disk".to_string(),
                (false, Some(short)) => format!("{} ({} short)", gb(r.model.download_gb), gb(short)),
                (false, None) => gb(r.model.download_gb),
            };
            let speed = match (v.tok_s, v.basis, v.runs) {
                (Some(t), Some(Basis::OtherDevice), _) => format!("~{} tok/s ?", rate(t)),
                (Some(t), _, _) => format!("~{} tok/s", rate(t)),
                (None, _, Runs::Streamed) => "storage-bound".to_string(),
                (None, _, _) => "-".to_string(),
            };
            let first = i == 0;
            lines.push([
                if first { r.model.repo.to_string() } else { String::new() },
                format!("{}{}", v.label, if all && v.recommended { " *" } else { "" }),
                if first { disk } else { String::new() },
                gb(v.memory_gb),
                match v.runs {
                    Runs::Gpu => "GPU".to_string(),
                    Runs::Cpu => "CPU".to_string(),
                    Runs::Streamed => "storage (over 60% of memory)".to_string(),
                    Runs::InMemory => "fits in memory".to_string(),
                },
                speed,
                if first {
                    r.model.caps.iter().map(|c| c.as_str()).collect::<Vec<_>>().join(", ")
                } else {
                    String::new()
                },
            ]);
        }
    }
    let mut width = [0usize; 7];
    for l in &lines {
        for (w, c) in width.iter_mut().zip(l) {
            *w = (*w).max(c.trim_start_matches('\n').chars().count());
        }
    }
    for l in &lines {
        let mut out = String::new();
        for (i, c) in l.iter().enumerate() {
            let pad = width[i] + c.len() - c.trim_start_matches('\n').len();
            let _ = write!(out, "{c:<pad$}  ");
        }
        let _ = writeln!(s, "{}", out.trim_end());
    }
    let _ = writeln!(
        s,
        "\nDECODE is an estimate for plain decoding of a short context (no speculative decoding): the\nbytes read per token over the kernel rate, plus a fixed cost per layer. Expect it within about\n25% on an idle machine; run a model with --model REPO [--quant q8|q4] to see its real speed.\n{}",
        if all {
            "* marks the variant to pick."
        } else {
            "--all lists every variant."
        }
    );
    s
}

/// The same as JSON: `{"hardware": {...}, "models": [{..., "variants": [...]}]}`.
pub fn to_json(hw: &Hardware, rows: &[Row]) -> serde_json::Value {
    use serde_json::json;
    // Three significant decimals are more than the estimates carry.
    let r = |v: f64| (v * 1e3).round() / 1e3;
    let ro = |v: Option<f64>| v.map(r);
    let rates: serde_json::Map<String, serde_json::Value> = Format::ALL
        .iter()
        .filter(|f| hw.cpu_rates[f.index()] > 0.0)
        .map(|f| (f.as_str().to_string(), json!(r(hw.cpu_rates[f.index()]))))
        .collect();
    json!({
        "hardware": {
            "summary": hw.summary,
            "cores": hw.cores,
            "memory_gb": ro(hw.ram_gb),
            "memory_available_gb": ro(hw.ram_available_gb),
            "disk_free_gb": ro(hw.disk_free_gb),
            "cache_dir": hw.cache_dir.display().to_string(),
            "gpu": hw.gpu.as_ref().map(|g| json!({
                "backend": g.kind.as_str(),
                "name": g.name,
                "memory_gb": ro(g.memory_gb),
                "unified_memory": g.unified,
                "reference_device": hw.gpu_ref().map(|(r, _)| r.device),
            })),
            "cpu_kernel_gb_per_s": rates,
        },
        "models": rows.iter().map(|row| json!({
            "model": row.model.repo,
            "kind": row.model.kind.as_str(),
            "capabilities": row.model.caps.iter().map(|c| c.as_str()).collect::<Vec<_>>(),
            "params_b": row.model.params_b,
            "download_gb": row.model.download_gb,
            "on_disk": row.on_disk,
            "disk_short_gb": ro(row.disk_short_gb),
            "moe": row.model.moe,
            "note": row.model.note,
            "variants": row.variants.iter().map(|v| json!({
                "label": v.label,
                "format": v.format.as_str(),
                "args": v.args,
                "memory_gb": r(v.memory_gb),
                "token_gb": r(v.token_gb),
                "runs_on": v.runs.as_str(),
                "decode_tok_s": ro(v.tok_s),
                "basis": v.basis.map(Basis::as_str),
                "recommended": v.recommended,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hw(ram: f64, gpu: Option<(GpuKind, &str, f64)>, disk: f64) -> Hardware {
        Hardware {
            summary: "test".into(),
            cores: 8,
            ram_gb: Some(ram),
            ram_available_gb: Some(ram),
            gpu: gpu.map(|(kind, name, mem)| Gpu {
                kind,
                name: name.into(),
                memory_gb: Some(mem),
                unified: false,
            }),
            cache_dir: PathBuf::from("/nonexistent/oxinfer-advisor-test"),
            disk_free_gb: Some(disk),
            cpu_rates: [100.0, 100.0, 70.0, 60.0, 50.0, 50.0, 40.0],
        }
    }

    fn row<'a>(rows: &'a [Row], repo: &str) -> &'a Row {
        rows.iter().find(|r| r.model.repo == repo).unwrap()
    }

    #[test]
    fn catalog_is_consistent() {
        for m in CATALOG {
            assert!(m.download_gb > 0.0 && m.token_gb <= m.download_gb * 1.02, "{}", m.repo);
            assert_eq!(m.token_gb > 0.0, m.layers > 0, "{}", m.repo);
            assert!(m.kind == Kind::Chat || m.token_gb == 0.0, "{}", m.repo);
            assert!(!m.caps.is_empty(), "{}", m.repo);
            let has = |c: Cap| m.can(c);
            assert!(
                match m.kind {
                    Kind::Chat => has(Cap::Text),
                    Kind::Embedding => has(Cap::Embeddings),
                    Kind::Speech => has(Cap::SpeechToText),
                    Kind::Tts => has(Cap::TextToSpeech),
                    Kind::Image => has(Cap::TextToImage) || has(Cap::ImageEdit) || has(Cap::Inpainting),
                },
                "{}",
                m.repo
            );
            assert_eq!(CATALOG.iter().filter(|o| o.repo == m.repo).count(), 1, "{}", m.repo);
        }
        // Grouped by kind, as the table prints them.
        let kinds: Vec<Kind> = CATALOG.iter().map(|m| m.kind).collect();
        let mut seen = Vec::new();
        for k in kinds {
            if seen.last() != Some(&k) {
                assert!(!seen.contains(&k), "{k:?} appears twice");
                seen.push(k);
            }
        }
    }

    #[test]
    fn cost_model() {
        // 1 GB at 100 GB/s is 10 ms; 20 layers at 0.5 ms are 10 ms more.
        assert!((speed(1.0, 100.0, 20, 0.5).unwrap() - 50.0).abs() < 1e-9);
        assert_eq!(speed(1.0, 0.0, 20, 0.5), None);
        assert_eq!(speed(0.0, 100.0, 20, 0.5), None);
    }

    #[test]
    fn cpu_only_machine() {
        let rows = survey(&hw(16.0, None, 500.0));
        // 1.5 GB at 100 x 1.2 GB/s + 24 layers: about 48 tok/s as stored, so bf16 is the pick.
        let small = row(&rows, "Qwen/Qwen3.5-0.8B");
        let v = small.recommended();
        assert_eq!((v.format, v.runs, v.basis), (Format::Bf16, Runs::Cpu, Some(Basis::Calibrated)));
        assert!((v.tok_s.unwrap() - 1.0 / (1.5 / (100.0 * CPU_FIT[0]) + 24.0 * CPU_LAYER_MS / 1e3)).abs() < 1e-6);
        // 9B: bf16 (19.3 GB) and q8 (12.4 GB) are over 60% of 16 GB and stream; q4 (7.9 GB) fits.
        let mid = row(&rows, "Qwen/Qwen3.5-9B");
        assert_eq!((mid.variants[0].runs, mid.variants[1].runs), (Runs::Streamed, Runs::Streamed));
        assert_eq!(mid.variants[0].tok_s, None);
        assert_eq!((mid.recommended().format, mid.recommended().runs), (Format::Q4, Runs::Cpu));
        // 27B: nothing fits in 9.6 GB; the smallest is recommended and nothing is estimated.
        let big = row(&rows, "Qwen/Qwen3.8-27B");
        assert!(big.variants.iter().all(|v| v.runs == Runs::Streamed && v.tok_s.is_none()));
        assert_eq!(big.recommended().format, Format::Q4);
        // Kinds without an estimate: one variant, a verdict, no speed.
        let sd = row(&rows, "stabilityai/sd-turbo");
        assert_eq!(sd.variants.len(), 1);
        assert_eq!((sd.variants[0].runs, sd.variants[0].tok_s), (Runs::InMemory, None));
        assert_eq!(row(&rows, "Qwen/Qwen-Image-Edit-2511").variants[0].runs, Runs::Streamed);
    }

    #[test]
    fn gpu_fit_and_reference() {
        let rows = survey(&hw(128.0, Some((GpuKind::Cuda, "NVIDIA GeForce RTX 5080", 16.6)), 500.0));
        let mid = row(&rows, "Qwen/Qwen3.5-9B");
        // bf16 (19.3 GB) does not fit the card: CPU. q8 (12.4 GB) and q4 do: GPU.
        assert_eq!(mid.variants[0].runs, Runs::Cpu);
        assert_eq!((mid.variants[1].runs, mid.variants[1].basis), (Runs::Gpu, Some(Basis::Reference)));
        // The measured 71.6 and 101.4 tok/s, within 10%.
        assert!(
            (mid.variants[1].tok_s.unwrap() / 71.6 - 1.0).abs() < 0.1,
            "{:?}",
            mid.variants[1].tok_s
        );
        assert!(
            (mid.variants[2].tok_s.unwrap() / 101.4 - 1.0).abs() < 0.1,
            "{:?}",
            mid.variants[2].tok_s
        );
        assert_eq!(mid.recommended().format, Format::Q8);
        // CUDA decode plans do not take GGUF blocks or the MoE models: CPU.
        assert_eq!(row(&rows, "bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0").variants[0].runs, Runs::Cpu);
        assert_eq!(row(&rows, "mlx-community/gemma-4-26b-a4b-it-4bit").variants[0].runs, Runs::Cpu);
        // Another card of the backend: same rates, flagged.
        let rows = survey(&hw(128.0, Some((GpuKind::Cuda, "NVIDIA GeForce RTX 3060", 12.0)), 500.0));
        let v = &row(&rows, "Qwen/Qwen3.5-0.8B").variants[0];
        assert_eq!((v.runs, v.basis), (Runs::Gpu, Some(Basis::OtherDevice)));
        // Vulkan has no decode plans.
        let rows = survey(&hw(128.0, Some((GpuKind::Vulkan, "Some GPU", 24.0)), 500.0));
        assert!(rows.iter().all(|r| r.variants.iter().all(|v| v.runs != Runs::Gpu)));
    }

    #[test]
    fn disk_and_output() {
        let h = hw(128.0, None, 10.0);
        let rows = survey(&h);
        assert_eq!(row(&rows, "Qwen/Qwen3.5-0.8B").disk_short_gb, None);
        let short = row(&rows, "Qwen/Qwen3.8-27B").disk_short_gb.unwrap();
        assert!((short - 45.6).abs() < 1e-6);
        let table = render(&h, &rows, false);
        assert!(table.contains("Qwen/Qwen3.8-27B") && table.contains("short)") && table.contains("tok/s"));
        assert_eq!(table.matches("Qwen/Qwen3.5-9B").count(), 1);
        assert!(render(&h, &rows, true).contains("--quant q4"));
        let j = to_json(&h, &rows);
        assert_eq!(j["models"].as_array().unwrap().len(), CATALOG.len());
        assert_eq!(j["models"][0]["variants"].as_array().unwrap().len(), 3);
        assert_eq!(j["models"][0]["capabilities"][0], "text");
        assert!(table.contains("CAN") && table.contains("text, image-input, video-input, thinking, tools"));
        for c in Cap::ALL {
            assert_eq!(Cap::parse(c.as_str()), Some(c));
            assert!(CATALOG.iter().any(|m| m.can(c)), "{c:?} is used");
        }
        assert_eq!(j["hardware"]["gpu"], serde_json::Value::Null);
        assert_eq!(cache_folder("org/name:Q4_K_M"), "org--name");
    }
}
