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
    Video,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Chat => "chat",
            Kind::Embedding => "embedding",
            Kind::Image => "image",
            Kind::Speech => "speech-to-text",
            Kind::Tts => "text-to-speech",
            Kind::Video => "video",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "chat" | "llm" | "text" => Kind::Chat,
            "embedding" | "embeddings" | "embed" => Kind::Embedding,
            "image" | "images" => Kind::Image,
            "speech" | "speech-to-text" | "stt" | "whisper" => Kind::Speech,
            "tts" | "text-to-speech" => Kind::Tts,
            "video" | "videos" => Kind::Video,
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
    TextToVideo,
    /// A first and / or last frame to video.
    ImageToVideo,
}

impl Cap {
    pub const ALL: [Cap; 19] = [
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
        Cap::TextToVideo,
        Cap::ImageToVideo,
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
            Cap::TextToVideo => "text-to-video",
            Cap::ImageToVideo => "image-to-video",
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
    Model::new("taobao-mnn/MNN-Pocket-0.3B", Kind::Chat, 0.36, 0.72, &[Text])
        .speed(0.61, 28)
        .note("a small Qwen3 without a thinking block"),
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
    Model::new("KRAFTON/Raon-1-nano", Kind::Chat, 3.1, 7.1, AGENT)
        .speed(3.06, 24)
        .note("half of its parameters are a per-layer embedding table read one row a token"),
    Model::new("LiquidAI/LFM3-1.7B", Kind::Chat, 1.7, 3.5, AGENT).speed(3.46, 24),
    Model::new("LiquidAI/LFM3-Omni-2B", Kind::Chat, 2.2, 4.5, &[Text, ImageInput, AudioInput, SpeechToText, TextToSpeech])
        .speed(3.46, 24)
        .note("text, images and audio in; text or 24 kHz speech out (--speak TEXT -o out.wav, --audio Q.wav -o reply.wav); on the CPU"),
    Model::new("Qwen/Qwen2.5-VL-3B-Instruct", Kind::Chat, 3.8, 7.5, &[Text, ImageInput]),
    Model::new("ibm-granite/granite-4.2-3b", Kind::Chat, 3.7, 7.3, AGENT).speed(6.81, 40),
    Model::new("ibm-granite/granite-4.2-3b-GGUF:Q4_K_M", Kind::Chat, 3.7, 2.2, AGENT)
        .fixed(Format::GgufQ4K, "GGUF Q4_K_M")
        .speed(2.10, 40),
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
    Model::new("webAI-Official/TwIL-8B:Q4_K_M-balanced", Kind::Chat, 8.8, 5.3, AGENT)
        .fixed(Format::GgufQ4K, "GGUF Q4_K_M")
        .speed(5.11, 40)
        .note("a Granite 4.2 8B fine-tune that reasons at length (--think)"),
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
    Model::new("ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:Q4_0", Kind::Chat, 31.6, 18.9, AGENT)
        .fixed(Format::GgufQ4, "GGUF Q4_0")
        .speed(2.73, 52)
        .moe()
        .note("no MTP, no speculative decoding (Mamba-2 state)"),
    Model::new("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", Kind::Chat, 31.6, 65.8, AGENT)
        .speed(6.44, 52)
        .moe()
        .note("not run in this layout (its GGUF and the Transcribe fine-tune were); bytes per token from the parameter count"),
    Model::new("MiniMaxAI/SynLogic-32B", Kind::Chat, 32.8, 66.0, &[Text]).note("a Qwen2.5 fine-tune"),
    Model::new("Qwen/Qwen3.6-35B-A3B", Kind::Chat, 35.0, 69.0, AGENT)
        .speed(6.0, 40)
        .moe()
        .note("bytes per token from the active parameter count"),
    Model::new("PixilabAI/Zawen-v0.1-35B-A3B", Kind::Chat, 35.0, 69.3, AGENT)
        .speed(6.0, 40)
        .moe()
        .note("a Qwen3.6-35B-A3B fine-tune for roleplay and creative writing; text only (no vision tower)"),
    Model::new("mradermacher/Zawen-v0.1-35B-A3B-GGUF:Q4_K_M", Kind::Chat, 35.0, 21.2, AGENT)
        .fixed(Format::GgufQ4, "GGUF Q4_K_M")
        .speed(1.84, 40)
        .moe()
        .note("Zawen in GGUF; also :IQ4_XS, :Q5_K_M, :Q6_K, :Q8_0"),
    // Ripple-65B: the NVFP4 checkpoint ran (its bytes per token are the loader's); the bf16 one
    // was never downloaded (a token reads all of it but the embedding table).
    Model::new("ricursiveAI/Ripple-65B-Preview-NVFP4", Kind::Chat, 65.0, 57.2, AGENT)
        .fixed(Format::Q4, "NVFP4")
        .speed(56.18, 80)
        .note("MLPs in NVFP4, attention in bf16; needs 60 GB of memory"),
    Model::new("ricursiveAI/Ripple-65B-Preview", Kind::Chat, 65.0, 129.6, AGENT)
        .speed(128.5, 80)
        .note("supported, untested (only its NVFP4 copy was run)"),
    Model::new("unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL", Kind::Chat, 125.0, 111.0, AGENT)
        .fixed(Format::GgufQ4K, "GGUF UD-Q4_K_XL")
        .speed(3.8, 48)
        .moe()
        .note("29 GB of it an n-gram table read a few rows a token; bytes per token and layers from the parameter count"),
    Model::new("SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF:IQ2_XS", Kind::Chat, 125.0, 68.5, AGENT)
        .fixed(Format::GgufLow, "GGUF GSQ-RCO IQ2_XS")
        .moe()
        .note("abliterated Qwen3.8-Flash-Next, a ggml type per tensor; also :IQ3_XXS, :IQ3_S, :Q2_0 (ternary experts); supported, untested"),
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
    // Inkling-2 Mini: the loop runs layers 6..18 three times (48 layer applications per token,
    // which is what `speed` counts; bytes per token from the loader on each file).
    Model::new(
        "thinkingmachines/Inkling-2-Mini",
        Kind::Chat,
        26.4,
        19.5,
        &[Text, ImageInput, Thinking, Tools],
    )
    .fixed(Format::Q4, "NVFP4")
    .speed(5.34, 48)
    .moe()
    .note("matches transformers on the real weights; Metal: ~56 tok/s decode, ~1850 tok/s prefill (M5 Max); --mtp drafts with its MTP head"),
    Model::new(
        "OpenUncensored/Inkling-2-Mini-Uncensored",
        Kind::Chat,
        26.4,
        19.5,
        &[Text, ImageInput, Thinking, Tools],
    )
    .fixed(Format::Q4, "NVFP4")
    .speed(5.34, 48)
    .moe()
    .note("uncensored (abliterated: the 20 attention output projections edited, experts, vision tower and MTP head unchanged); same files as the release; the edit alone as a PEFT LoRA: --lora OpenUncensored/Inkling-2-Mini-Uncensored-LoRA"),
    Model::new(
        "ggml-org/Inkling-2-Mini-GGUF:Q8_0",
        Kind::Chat,
        26.4,
        27.8,
        &[Text, ImageInput, Thinking, Tools],
    )
    .fixed(Format::GgufQ8, "GGUF Q8_0")
    .speed(5.11, 48)
    .moe()
    .note("images with the mmproj beside it; Metal: ~59 tok/s decode, ~1600 tok/s prefill (M5 Max; llama.cpp 38 / 1000); --draft auto with unsloth's drafter"),
    Model::new(
        "unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS",
        Kind::Chat,
        26.4,
        11.7,
        &[Text, ImageInput, Thinking, Tools],
    )
    .fixed(Format::GgufLow, "GGUF UD-IQ3_XXS")
    .speed(2.92, 48)
    .moe()
    .note("unsloth's own layout (the root files); --kv-cache q8 for long contexts on 16-24 GB; Metal: ~66 tok/s decode"),
    Model::new(
        "unsloth/Inkling-2-Mini-GGUF:UD-IQ2_XXS",
        Kind::Chat,
        26.4,
        8.5,
        &[Text, ImageInput, Thinking, Tools],
    )
    .fixed(Format::GgufLow, "GGUF UD-IQ2_XXS")
    .speed(2.40, 48)
    .moe()
    .note("the phone file: in memory (GPU) on a 16 GB iPad, streamed from flash on 8-12 GB iPhones (docs/inkling.md, iOS); --kv-cache q8; Metal: ~73 tok/s decode (M5 Max); also :UD-IQ1_S (7.8 GB)"),
    Model::new(
        "AtomicChat/Inkling-2-Mini-Uncensored-GGUF:AD-Q4_K",
        Kind::Chat,
        26.4,
        16.5,
        &[Text, ImageInput, Thinking, Tools],
    )
    .fixed(Format::GgufQ4, "GGUF AD-Q4_K")
    .moe()
    .note("uncensored (a rank-1 refusal-direction LoRA baked in); llama.cpp's layout; Metal: ~62 tok/s decode (llama.cpp 41); also :AD-IQ3_S (13.3 GB), :AD-IQ2_M (8.7 GB, run); images with the mmproj beside it"),
    // The same quantizations without the LoRA, which AtomicChat ships separately: applied at
    // run time (`--lora`), switchable, closer to the baked files than the base (docs/lora.md).
    Model::new(
        "AtomicChat/Inkling-2-Mini-GGUF:AD-IQ2_M",
        Kind::Chat,
        26.4,
        8.7,
        &[Text, ImageInput, Thinking, Tools],
    )
    .fixed(Format::GgufLow, "GGUF AD-IQ2_M")
    .moe()
    .note("the base of the Uncensored files; + --lora AtomicChat/Inkling-2-Mini-Uncensored-GGUF/Inkling-2-Mini-Uncensored-LoRA.gguf (44 MB) applies their LoRA at run time (strength :SCALE, 0 = off) instead of a second 8.7 GB file; also :AD-Q4_K (16.5 GB), :AD-IQ3_S (13.3 GB)"),
    // Inkling-2 Small: never run on the real weights (neither release fits the download budget;
    // the agreement with transformers is measured on tiny checkpoints, the vision tower on the
    // real one; bytes per token from the layout of the releases).
    Model::new(
        "thinkingmachines/Inkling-2-Small-MXFP8",
        Kind::Chat,
        594.0,
        622.8,
        &[Text, ImageInput, Thinking, Tools],
    )
    .fixed(Format::Q8, "MXFP8")
    .speed(23.5, 48)
    .moe()
    .note("supported, untested on the real weights; 73 GB of memory for the experts' scales; no MTP"),
    Model::new(
        "thinkingmachines/Inkling-2-Small",
        Kind::Chat,
        594.0,
        393.5,
        &[Text, ImageInput, Thinking, Tools],
    )
    .fixed(Format::Q4, "NVFP4")
    .speed(19.0, 48)
    .moe()
    .note("supported, untested on the real weights; its NVFP4 experts are repacked in memory (290 GB): MXFP8 streams; no MTP"),
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
    // Document OCR: pages in, text with its layout out (docs/ocr.md). 2.4B of the parameters are
    // the routed experts of its 12 layers (6 of 64 per token), 0.4B its SAM + CLIP image encoder.
    Model::new("baidu/Unlimited-OCR-2", Kind::Chat, 3.3, 6.7, &[Text, ImageInput])
        .moe()
        .note("document OCR: page images to text with layout blocks; not a chat model"),
    // DeepSeek V4.1 Flash: text only here (no vision tower), never run on the real weights (the
    // agreement with DeepSeek's code is measured on tiny checkpoints). Sizes with the 111 GB of
    // engram tables (IQ4_NL), which every conversion needs.
    Model::new("unsloth/DeepSeek-V4.1-Flash-1017-GGUF:UD-IQ1_S", Kind::Chat, 748.0, 233.0, AGENT)
        .fixed(Format::GgufLow, "GGUF UD-IQ1_S")
        .moe()
        .note("supported, untested; 111 GB of it engram tables read a few rows a token; text only"),
    // The smallest complete V4.1 checkpoint (120.7 GB of backbone + the 110.6 GB engram file,
    // which `org/repo:IQ1_M` fetches as the build's sixth part): run on the real weights.
    Model::new(
        "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF:IQ1_M",
        Kind::Chat,
        748.0,
        231.3,
        AGENT,
    )
    .fixed(Format::GgufLow, "GGUF IQ1_M")
    .speed(7.18, 40)
    .moe()
    .note("tested: same greedy tokens and perplexity as llama.cpp on the same file; streamed (5.4 GB resident); text only"),
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
    // LongCat-2.5 and LongCat-2.0: never run (3.5 TB each); the agreement with the published code
    // is measured on tiny checkpoints, the loader on the releases' headers, and
    // LongCat-Flash-Lite-Sparse (the same architecture) was run on its real weights. Bytes per
    // token: the loader's report.
    Model::new("meituan-longcat/LongCat-2.5", Kind::Chat, 1600.0, 3562.0, AGENT)
        .fixed(Format::Bf16, "bf16")
        .speed(112.2, 38)
        .moe()
        .note("supported, untested; text only (no vision tower, no MTP)"),
    Model::new("meituan-longcat/LongCat-2.0", Kind::Chat, 1600.0, 3552.0, AGENT)
        .fixed(Format::Bf16, "bf16")
        .speed(112.2, 38)
        .moe()
        .note("supported, untested; the text model of LongCat-2.5"),
    Model::new("meituan-longcat/LongCat-Flash-Lite-Sparse", Kind::Chat, 69.0, 138.3, &[Text, Tools])
        .fixed(Format::Bf16, "bf16")
        .speed(9.1, 14)
        .moe()
        .note("63 GB of it n-gram tables read a few rows a token; no thinking mode"),
    // Kimi K3.1 Pro: text only here (no vision tower), never run on the real weights (the
    // agreement with Moonshot's code is measured on tiny checkpoints; the loader was tried on
    // these files' headers). Bytes per token: the dense weights of 93 layers and 16 experts each.
    Model::new("midagedev/Kimi-K3.1-Pro-GGUF:IQ1_S", Kind::Chat, 2800.0, 580.0, AGENT)
        .fixed(Format::GgufLow, "GGUF IQ1_S")
        .speed(41.7, 93)
        .moe()
        .note("supported, untested; text only"),
    Model::new("KarLexAI/Kimi-K3.1-Pro-GGUF:KLX-IQ1_S", Kind::Chat, 2800.0, 593.0, AGENT)
        .fixed(Format::GgufLow, "GGUF KLX-IQ1_S")
        .speed(55.5, 93)
        .moe()
        .note("supported, untested; text only"),
    Model::new("DevQuasar/moonshotai.Kimi-K3.1-Pro-GGUF:MXFP4_MOE", Kind::Chat, 2800.0, 1509.0, AGENT)
        .fixed(Format::GgufQ4, "GGUF MXFP4_MOE")
        .speed(84.6, 93)
        .moe()
        .note("supported, untested; the release's own 4-bit experts; text only"),
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
    Model::new("google/embeddinggemma-2", Kind::Embedding, 0.74, 1.5, EMBED).note("text, images and audio in one space (--image, --audio)"),
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
    Model::new("nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B", Kind::Speech, 34.0, 67.5, STT)
        .note("a speech LM (audio encoders in front of Nemotron-H); one segment per file, no timestamps"),
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
    Model::new("fishaudio/drama-3", Kind::Tts, 35.9, 41.5, &[TextToSpeech, VoiceCloning])
        .fixed(Format::Q8, "FP8")
        .moe()
        .note("44.1 kHz, up to 16 speakers per text; cloning from a clip and its transcript; on the CPU"),
    Model::new("Edge0/Audio8-TTS-Realistic-Air", Kind::Tts, 0.6, 2.6, &[TextToSpeech, VoiceCloning])
        .note("44.1 kHz, 23 languages; cloning from a clip and its transcript; on the CPU"),
    Model::new("higurashi-ai/Haru-4B", Kind::Tts, 5.3, 10.5, &[AudioInput, TextToSpeech, VoiceCloning])
        .note("speech to speech, full duplex at 24 kHz: --audio USER.wav -o REPLY.wav; a research base model; on the CPU"),
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
    Model::new("lodestones/Kromoss", Kind::Image, 13.5, 29.5, &[TextToImage])
        .note("MOSS-Image fine-tune with Qwen3.5-9B (the cached Qwen/Qwen3.5-9B is reused); 50 steps, CFG 3; presets kromoss-dmd2, kromoss-famegrid"),
    Model::new("molbal/Kromoss-GGUF:Q4_K_M", Kind::Image, 13.5, 21.7, &[TextToImage])
        .fixed(Format::GgufQ4K, "GGUF Q4_K_M transformer")
        .note("the transformer as GGUF (also Q6_K, Q8_0); text encoder and VAE from lodestones/Kromoss"),
    Model::new("molbal/Kromoss-GGUF:Q8_CR", Kind::Image, 13.5, 23.5, &[TextToImage])
        .fixed(Format::GgufQ8, "GGUF Q8_CR transformer (ComfyUI ConvRot int8)")
        .note("rotated int8 rows (5.4 GB), the int8 of kromoss_int8_convrot; text encoder and VAE from lodestones/Kromoss"),
    Model::new("molbal/Kromoss-GGUF:Q4_CR_L", Kind::Image, 13.5, 21.9, &[TextToImage])
        .fixed(Format::GgufQ4, "GGUF Q4_CR_L transformer (ComfyUI ConvRot int4 / int8)")
        .note("rotated int4 feed-forward, int8 attention (3.8 GB); text encoder and VAE from lodestones/Kromoss"),
    Model::new("OpenMOSS-Team/MOSS-Image-SFT", Kind::Image, 13.5, 29.5, &[TextToImage])
        .note("the release's instruction-tuned MOSS-Image (Qwen3.5-9B, the cached Qwen/Qwen3.5-9B is reused); 50 steps, CFG 3"),
    Model::new("OpenMOSS-Team/MOSS-Image-Base", Kind::Image, 13.5, 29.5, &[TextToImage])
        .note("the release's base model, Kromoss' starting point; 50 steps, CFG 3"),
    Model::new("MiniMaxAI/MiniMax-H3-Image", Kind::Image, 63.0, 127.2, &[TextToImage])
        .note("with MiniMax-H3's Qwen3-VL text encoder; 2048x2048 and up"),
    Model::new("MiniMaxAI/MiniMax-H3", Kind::Video, 67.4, 140.1, &[TextToVideo, ImageToVideo])
        .note(
            "transformer/, text_encoder/, vae/ and audio_vae/ of the 498 GB repository; 24 fps, 5 to 15 s with a stereo \
             soundtrack; on the CPU (CUDA: the blocks and the VAE decoder); Comfy-Org/MiniMax-H3's single files, the turbo \
             LoRAs and a GGUF text encoder (with its mmproj for keyframes and references) load too (48 GB, 4 or 8 forwards): \
             docs/video.md",
        ),
    // SulphurAI/Sulphur-3: a fine-tune of MiniMax-H3 as ComfyUI single files; the downloads
    // count the VAEs of Comfy-Org/MiniMax-H3 (5.8 GB) and the Q8_0 GGUF text encoder (34.8 GB).
    Model::new("SulphurAI/Sulphur-3/s3.safetensors", Kind::Video, 53.5, 88.8, &[TextToVideo, ImageToVideo])
        .note(
            "Sulphur-3 (a MiniMax-H3 fine-tune), the release's pruned bf16 file (40.2 GB) with its corrector \
             (--corrector SulphurAI/Sulphur-3/s3_corrector.safetensors, 8.0 GB); --vae / --audio-vae from \
             Comfy-Org/MiniMax-H3, --text-encoder unsloth/Qwen3-VL-32B-Instruct-GGUF (Q8_0); 20 Euler steps (--steps 21), \
             JSON captions; CPU: 448x256, 22 frames in 12 min (33 s a forward, corrector 1.7 s), 62 GB resident: docs/video.md#sulphur-3",
        ),
    Model::new("SulphurAI/Sulphur-3/quantized/s3_w4a8.safetensors", Kind::Video, 53.5, 52.0, &[TextToVideo, ImageToVideo])
        .note(
            "Sulphur-3 in comfy_kitchen's W4A8 (4-bit codebook codes, 11.4 GB; decoded to int8 at load), same parts and \
             settings as s3.safetensors; CPU: 448x256, 22 frames in 11 min (31 s a forward), 34 GB resident: \
             docs/video.md#sulphur-3",
        ),
    Model::new(
        "SulphurAI/Sulphur-3/experimental/Sulphur-3_Ref2VA-hybrid_mid_int8_convrot_keepB0-1-49.safetensors",
        Kind::Video,
        54.1,
        63.9,
        &[TextToVideo, ImageToVideo],
    )
    .note(
        "Sulphur-3's experimental Ref2VA transformer (int8 ConvRot blocks 2-48, 22.1 GB): references with --ref, the \
         text encoder's vision tower from mmproj-F16.gguf beside the GGUF (1.2 GB); CPU: one image reference, 448x256, 22 frames, 12 forwards in 7 min (32 s each), 37 GB resident: docs/video.md#sulphur-3",
    ),
    Model::new("meituan-longcat/LongCat-Video-Avatar-1.6", Kind::Video, 22.2, 47.3, &[TextToVideo, ImageToVideo])
        .note(
            "audio-driven avatar video (--audio, --init-image), 25 fps, 93 frames a clip; the int8 transformer, the two LoRAs \
             and Whisper of the repository with umT5 and the VAE of meituan-longcat/LongCat-Video; presets \
             longcat-avatar-int8, longcat-avatar; CUDA or the CPU; not in the server: docs/longcat-avatar.md",
        ),
    Model::new("Lightricks/LTX-2.5", Kind::Video, 34.6, 39.7, &[TextToVideo, ImageToVideo])
        .fixed(Format::Q8, "ComfyUI int8 ConvRot transformer and text encoder")
        .note(
            "22B audio-video DiT + Gemma 4 12B text encoder, the distilled two-stage pipeline (8 + 3 forwards) with a 48 kHz \
             stereo soundtrack; image to video, the duration head, the spatial / temporal upsamplers, LoRAs (the distilled \
             LoRA on the dev transformer); fetched from the public mirror comfyicu/LTX-2.5 (the release is gated); the \
             transformer in whole passes on Metal: 768x512, 121 frames (two stages) in 88 s on an M5 Max, \
             22 GB resident; 7.8 s a forward on the CPU at 384x256x25 (docs/ltx.md)",
        ),
];

/// A curated stack: a base model of the catalog with LoRA files at set strengths, a trigger word
/// and the generation settings the authors recommend. `oxinfer --preset ID` (or `--model ID`),
/// the server's model sources and the pipelines' `preset` option take the id.
#[derive(Clone, Copy, Debug)]
pub struct Preset {
    pub id: &'static str,
    /// The catalog model it runs on (what `--model` takes).
    pub base: &'static str,
    /// The pipeline that reads [`Preset::load`] and [`Preset::defaults`] (`moss-image`).
    pub pipeline: &'static str,
    /// Load options of that pipeline (`lora` = `org/name/file.safetensors:STRENGTH`, ...).
    pub load: &'static [(&'static str, &'static str)],
    /// Generation defaults (`steps`, `guidance`, `sampler`, `scheduler`, ...); requests override them.
    pub defaults: &'static [(&'static str, &'static str)],
    /// Put at the start of prompts that lack it.
    pub trigger: &'static str,
    /// Downloads on top of the base, in GB.
    pub extra_download_gb: f64,
    /// Where the LoRA files come from and where the settings are documented.
    pub source: &'static str,
    pub note: &'static str,
}

/// The curated presets.
pub const PRESETS: &[Preset] = &[
    Preset {
        id: "kromoss",
        base: "lodestones/Kromoss",
        pipeline: "moss-image",
        load: &[],
        defaults: &[("steps", "50"), ("guidance", "3"), ("uncond_skip_layers", "22")],
        trigger: "",
        extra_download_gb: 0.0,
        source: "https://huggingface.co/lodestones/Kromoss (README: 50 steps, CFG 3, shift 5, negative branch without block 22)",
        note: "the full model, as the release's pipeline",
    },
    Preset {
        id: "kromoss-dmd2",
        base: "lodestones/Kromoss",
        pipeline: "moss-image",
        load: &[("lora", "lodestones/Kromoss/8-steps-dmd2-lora.safetensors:1")],
        defaults: &[("steps", "8"), ("guidance", "1")],
        trigger: "",
        extra_download_gb: 1.2,
        source: "https://huggingface.co/lodestones/Kromoss (8-steps-dmd2-lora: 8 steps, CFG 1, no negative branch)",
        note: "8-step turbo LoRA (DMD2), about 12x faster than 50 steps with CFG",
    },
    Preset {
        id: "kromoss-dmd",
        base: "lodestones/Kromoss",
        pipeline: "moss-image",
        load: &[("lora", "lodestones/Kromoss/8-steps-dmd-lora.safetensors:1")],
        defaults: &[("steps", "8"), ("guidance", "1")],
        trigger: "",
        extra_download_gb: 1.2,
        source: "https://huggingface.co/lodestones/Kromoss (8-steps-dmd-lora: 8 steps, CFG 1)",
        note: "8-step turbo LoRA (data-space DMD)",
    },
    Preset {
        id: "kromoss-famegrid",
        base: "lodestones/Kromoss",
        pipeline: "moss-image",
        load: &[("lora", "Lnypuuuu/txt2image/FameGrid-MOSS-Preview.safetensors:1.4")],
        defaults: &[
            ("steps", "24"),
            ("guidance", "2.5"),
            ("sampler", "er_sde"),
            ("scheduler", "beta57"),
            ("uncond_skip_layers", "none"),
        ],
        trigger: "famegrid",
        extra_download_gb: 0.58,
        source: "FameGrid Moss Image Preview by UltraMuse (civitai.com/models/2088956, version 3478421: trigger famegrid, \
                 ER_SDE + beta57, 24 steps, CFG 2.5, weight 1.5-2; Kromoss bf16 recommended); ungated copy with the same \
                 SHA-256 at huggingface.co/Lnypuuuu/txt2image",
        note: "photographic social-media look (a LyCORIS LoKr, merged at load) at strength 1.4",
    },
    // LongCat-Video-Avatar: the transformer and the two LoRAs are folders of the base repository
    // (`extra_download_gb`: nothing beyond it), with the text encoder and VAE of
    // meituan-longcat/LongCat-Video.
    Preset {
        id: "longcat-avatar-int8",
        base: "meituan-longcat/LongCat-Video-Avatar-1.6",
        pipeline: "longcat-avatar",
        load: &[
            ("transformer", "base_model_int8"),
            ("lora", "loraA/lora.safetensors:1"),
            ("lora", "loraB/lora.safetensors:1"),
        ],
        defaults: &[("steps", "8"), ("guidance", "1"), ("mix_steps", "4"), ("mix_alpha", "1")],
        trigger: "",
        extra_download_gb: 0.0,
        source: "https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6 (Quick Inference: --use_distill --use_int8 \
                 --mix_steps=4 --mix_alpha=1.0: 8 steps without guidance, the DMD LoRA (loraA) for the first four and the \
                 OPSD LoRA (loraB) for the rest)",
        note: "audio-driven avatar video (--audio, --init-image): the int8 transformer (15.9 GB) with the two LoRAs",
    },
    Preset {
        id: "longcat-avatar",
        base: "meituan-longcat/LongCat-Video-Avatar-1.6",
        pipeline: "longcat-avatar",
        load: &[
            ("transformer", "base_model"),
            ("lora", "loraA/lora.safetensors:1"),
            ("lora", "loraB/lora.safetensors:1"),
        ],
        defaults: &[("steps", "8"), ("guidance", "1"), ("mix_steps", "4"), ("mix_alpha", "1")],
        trigger: "",
        extra_download_gb: 0.0,
        source: "https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6 (Quick Inference: --use_distill \
                 --mix_steps=4 --mix_alpha=1.0)",
        note: "audio-driven avatar video (--audio, --init-image): the bf16 transformer (31.7 GB) with the two LoRAs",
    },
];

/// The preset of that id.
pub fn preset(id: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.id == id)
}

/// The presets as JSON (`oxinfer --presets --format json`).
pub fn presets_json() -> serde_json::Value {
    serde_json::Value::Array(
        PRESETS
            .iter()
            .map(|p| {
                serde_json::json!({
                    "id": p.id,
                    "base": p.base,
                    "pipeline": p.pipeline,
                    "load": p.load.iter().map(|(k, v)| serde_json::json!([k, v])).collect::<Vec<_>>(),
                    "defaults": p.defaults.iter().map(|(k, v)| (k.to_string(), serde_json::json!(v))).collect::<serde_json::Map<_, _>>(),
                    "trigger": p.trigger,
                    "extra_download_gb": p.extra_download_gb,
                    "source": p.source,
                    "note": p.note,
                })
            })
            .collect(),
    )
}

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

/// Whether the device keeps its own copy of the weights in the memory it shares with the CPU: a
/// CUDA device with unified memory (DGX Spark, Jetson), where `cuMemAlloc` takes system memory
/// beside the host's copy (Qwen3.5-0.8B bf16 on a Spark: 1.76 GB resident and 1.67 GB on the
/// device). Metal wraps the host's pages.
fn copies_in_shared_memory(kind: GpuKind, unified: bool) -> bool {
    unified && kind == GpuKind::Cuda
}

/// Whether `memory_gb` of weights fit a device with `mem` GB for a decode plan: its own memory
/// with the headroom, and twice the weights where the device's copy shares the host's memory.
fn fits_device(kind: GpuKind, unified: bool, mem: f64, memory_gb: f64) -> bool {
    let copies = if copies_in_shared_memory(kind, unified) { 2.0 } else { 1.0 };
    copies * memory_gb + GPU_HEADROOM_GB <= GPU_FRACTION * mem
}
/// What an app keeps for itself where the OS limits apps ([`crate::stream::DeviceMemory::APP_RESERVE`]).
const APP_RESERVE_GB: f64 = 1.073741824;

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
    /// What an app may still allocate where the OS limits apps (iOS: `os_proc_available_memory`,
    /// raised by the increased-memory-limit entitlement); None elsewhere. The weights must then
    /// fit in it less 1 GB (the engine's rule, [`crate::stream::DeviceMemory::budget`]).
    pub app_memory_gb: Option<f64>,
    pub gpu: Option<Gpu>,
    /// Where repo ids are downloaded, and the space left there.
    pub cache_dir: PathBuf,
    pub disk_free_gb: Option<f64>,
    /// CPU streaming rate per [`Format::ALL`] entry, GB/s ([`calibrate`]).
    pub cpu_rates: [f64; 7],
    /// Core ML loads (Apple Silicon): models converted for the Neural Engine run.
    pub neural_engine: bool,
    /// Every accelerator found, one line each ([`accelerators`]).
    pub accelerators: Vec<Accelerator>,
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
        let accelerators = accelerators(plat, device);
        Hardware {
            neural_engine: accelerators.iter().any(|a| a.backend == "neural-engine" && a.available),
            accelerators,
            summary: plat.summary(),
            cores: plat.cpu.cores,
            ram_gb: plat.cpu.memory_bytes.map(|b| b as f64 / 1e9),
            ram_available_gb: crate::platform::available_memory().map(|b| b as f64 / 1e9),
            app_memory_gb: crate::platform::process_available_memory().map(|b| b as f64 / 1e9),
            gpu,
            disk_free_gb: crate::platform::free_disk(&cache_dir).map(|b| b as f64 / 1e9),
            cache_dir,
            cpu_rates: [0.0; 7],
        }
    }

    /// The weights that stay in memory, GB: 60% of the RAM, and where the OS limits apps, at most
    /// what the app may use less 1 GB (as [`crate::stream::DeviceMemory::budget`]).
    pub fn memory_budget_gb(&self) -> Option<f64> {
        let ram = self.ram_gb.map(|r| RAM_FRACTION * r);
        match self.app_memory_gb.map(|a| (a - APP_RESERVE_GB).max(1.0)) {
            Some(a) => Some(ram.map_or(a, |r| r.min(a))),
            None => ram,
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
    /// Which accelerator the engine uses for it here ([`Accel`]).
    pub accel: Accel,
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
    // `org/name`, `org/name:QUANT` or a file of a repository, `org/name/path/file.safetensors`.
    let r = repo.split(':').next().unwrap_or(repo);
    r.splitn(3, '/').take(2).collect::<Vec<_>>().join("--")
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
                let in_ram = hw.memory_budget_gb().is_none_or(|b| memory_gb <= b);
                let cpu = speed(
                    token_gb,
                    hw.cpu_rates[format.index()] * CPU_FIT[format.index()],
                    m.layers,
                    if m.moe { CPU_MOE_LAYER_MS } else { CPU_LAYER_MS },
                );
                let gpu = hw.gpu_ref().and_then(|(r, same)| {
                    let g = hw.gpu.as_ref()?;
                    let fits = fits_device(g.kind, g.unified, g.memory_gb?, memory_gb);
                    let s = speed(token_gb, r.rates[format.index()], m.layers, r.layer_ms)?;
                    (fits && (r.moe || !m.moe)).then_some((s, if same { Basis::Reference } else { Basis::OtherDevice }))
                });
                let (runs, tok_s, basis) = match (in_ram, gpu, cpu) {
                    (false, _, _) => (Runs::Streamed, None, None),
                    (true, _, _) if token_gb == 0.0 => (Runs::InMemory, None, None),
                    (true, Some((g, b)), c) if c.is_none_or(|c| g > c) => (Runs::Gpu, Some(g), Some(b)),
                    (true, _, c) => (Runs::Cpu, c, c.map(|_| Basis::Calibrated)),
                };
                let accel = accel_of(m, &label, format, memory_gb, runs == Runs::Streamed, hw);
                Variant {
                    accel,
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

/// Where one part of a model's work runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Cpu,
    Gpu(GpuKind),
    /// Apple's Neural Engine, through Core ML.
    NeuralEngine,
}

impl Unit {
    /// `cpu`, `metal`, `cuda`, `vulkan` or `neural-engine`.
    pub fn as_str(self) -> &'static str {
        match self {
            Unit::Cpu => "cpu",
            Unit::Gpu(k) => k.as_str(),
            Unit::NeuralEngine => "neural-engine",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Unit::Cpu => "CPU",
            Unit::Gpu(GpuKind::Metal) => "GPU (Metal)",
            Unit::Gpu(GpuKind::Cuda) => "GPU (CUDA)",
            Unit::Gpu(GpuKind::Vulkan) => "GPU (Vulkan)",
            Unit::NeuralEngine => "Neural Engine",
        }
    }
}

/// Which accelerator the engine uses for a variant on this machine, with the defaults
/// (`--device auto`, `--gpu-decode on`).
#[derive(Clone, Debug, PartialEq)]
pub struct Accel {
    /// The prompt (chat models), the audio encoder (speech to text), or the whole pipeline (the
    /// other kinds, which have one stage).
    pub prefill: Unit,
    /// Token generation; the same as `prefill` for the kinds with one stage.
    pub decode: Unit,
    /// Decoding times the GPU and the CPU on the first tokens and keeps the faster (a streamed
    /// mixture of experts on unified memory, `gpu::decode::DeviceChoice`).
    pub decode_measured: bool,
    /// The weights are read from storage as they are needed.
    pub streamed: bool,
    /// What the GUI and the table print: `GPU (Metal) prefill + decode`, `GPU (CUDA) prefill, CPU
    /// decode`, `CPU`, `CPU, streamed from disk`, `Neural Engine`, ...
    pub label: String,
}

/// What the engine has for a model, as far as the accelerators go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EnginePath {
    /// A whole-token decode graph (`gpu::decode`, `qwen.rs` / `model_gpu.rs`): Llama 3 (Orpheus
    /// TTS too), Qwen3 / 3.5 / 3.6 / 3.8 dense and MoE, Granite 4, Gemma 3 / 4. The GPU runs the
    /// prompt and every decoded token.
    Graph,
    /// Qwen4-Exp (Qwen3.8-Flash-Next): a decode graph Metal runs and CUDA declines.
    GraphMetal,
    /// The shared linear layers only (`model::linear_multi`): the GPU takes the prompt's large
    /// matmuls, decoding runs on the CPU (the families without a graph, and Qwen2.5 / LFM3 /
    /// MiniMax, whose layers the graph does not cover).
    Matmuls,
    /// Speech to text (Whisper): the encoder on the GPU, text decoding on the CPU.
    Encoder,
    /// One stage whose products (convolutions, attention, matmuls) run on the GPU: images,
    /// embeddings.
    Pipeline,
    /// The CPU alone (Hadamard-folded ternary weights, the pipelines whose docs say so).
    Cpu,
    /// A Core ML package (ANEMLL): the Neural Engine.
    CoreMl,
}

/// The engine path of a catalog entry, from its family (named in its repo id) and kind. An entry
/// not listed here (a family added later) is taken as [`EnginePath::Matmuls`] for chat, the path every
/// language model has.
fn path_of(m: &Model) -> EnginePath {
    let r = m.repo.to_ascii_lowercase();
    let has = |s: &[&str]| s.iter().any(|p| r.contains(p));
    if r.starts_with("anemll/") {
        return EnginePath::CoreMl;
    }
    match m.kind {
        Kind::Chat if has(&["ternary-bonsai"]) => EnginePath::Cpu,
        Kind::Chat if has(&["flash-next"]) => EnginePath::GraphMetal,
        Kind::Chat
            if has(&[
                "qwen3.5",
                "qwen3.6",
                "qwen3.8",
                "qwen3-",
                "mnn-pocket",
                "zawen",
                "jev-",
                "llama-3",
                "gemma-3",
                "gemma-4",
                "gev-",
                "granite-4",
                "twil-",
                "minimax-h3-image-pe",
            ]) =>
        {
            EnginePath::Graph
        }
        Kind::Chat => EnginePath::Matmuls,
        Kind::Speech if has(&["whisper"]) => EnginePath::Encoder,
        Kind::Speech => EnginePath::Matmuls,
        Kind::Tts if has(&["orpheus"]) => EnginePath::Graph,
        Kind::Video if has(&["ltx-2"]) => EnginePath::Pipeline,
        Kind::Tts | Kind::Video => EnginePath::Cpu,
        Kind::Image | Kind::Embedding => EnginePath::Pipeline,
    }
}

/// Whether the backend has kernels for the bulk of the weights: Metal takes every format but
/// NVFP4 and MXFP8 (FP8 with 128 x 128 block scales only); CUDA and Vulkan take bf16 and
/// oxinfer's / MLX's Q8 and Q4 blocks, no GGUF or FP8 (`cuda.rs` / `vulkan.rs` `upload_weight`).
/// A GGUF file is also recognized by its repo id (entries such as the Orpheus GGUFs are listed
/// without a fixed format).
fn gpu_takes(kind: GpuKind, format: Format, label: &str, repo: &str) -> bool {
    if label.contains("NVFP4") || label.contains("MXFP8") || label.contains("ternary") {
        return false;
    }
    let gguf = repo.to_ascii_lowercase().contains("gguf");
    match kind {
        GpuKind::Metal => true,
        GpuKind::Cuda | GpuKind::Vulkan => matches!(format, Format::Bf16 | Format::Q8 | Format::Q4) && !label.contains("FP8") && !gguf,
    }
}

fn accel_of(m: &Model, label: &str, format: Format, memory_gb: f64, streamed: bool, hw: &Hardware) -> Accel {
    let path = path_of(m);
    let gpu = hw.gpu.as_ref().filter(|g| gpu_takes(g.kind, format, label, m.repo));
    let single = !matches!(m.kind, Kind::Chat | Kind::Speech | Kind::Tts);
    let (prefill, decode, measured) = match path {
        EnginePath::CoreMl => (Unit::NeuralEngine, Unit::NeuralEngine, false),
        EnginePath::Cpu => (Unit::Cpu, Unit::Cpu, false),
        _ if single => {
            let u = gpu.map_or(Unit::Cpu, |g| Unit::Gpu(g.kind));
            (u, u, false)
        }
        _ => {
            let prefill = gpu.map_or(Unit::Cpu, |g| Unit::Gpu(g.kind));
            let graph = match path {
                EnginePath::Graph => true,
                EnginePath::GraphMetal => gpu.is_some_and(|g| g.kind == GpuKind::Metal),
                _ => false,
            };
            // A device decode plan keeps every weight on the device: on a discrete GPU they must
            // fit its memory; Metal's unified memory wraps the host's (streamed layers per pass),
            // CUDA's holds a second copy. FP8 has Metal GEMMs but no decode matvec.
            let on_gpu = gpu.filter(|g| {
                graph
                    && g.kind != GpuKind::Vulkan
                    && !label.contains("FP8")
                    && ((g.unified && !copies_in_shared_memory(g.kind, g.unified))
                        || g.memory_gb.is_some_and(|mem| fits_device(g.kind, g.unified, mem, memory_gb)))
            });
            let measured = streamed && m.moe && on_gpu.is_some_and(|g| g.unified);
            (prefill, on_gpu.map_or(Unit::Cpu, |g| Unit::Gpu(g.kind)), measured)
        }
    };
    let mut text = match (path, prefill, decode) {
        (EnginePath::CoreMl, ..) if hw.neural_engine => "Neural Engine (Core ML)".to_string(),
        (EnginePath::CoreMl, ..) => "Neural Engine (Core ML), not available here".to_string(),
        (_, Unit::Cpu, Unit::Cpu) => "CPU".to_string(),
        (_, Unit::Gpu(GpuKind::Vulkan), _) if single => "GPU (Vulkan) matmuls, the rest on the CPU".to_string(),
        (EnginePath::Encoder, p, _) => format!("{} encoder, CPU decode", p.label()),
        (_, p, _) if single => p.label().to_string(),
        (_, p, _) if measured => format!("{} prefill, GPU or CPU decode (the faster)", p.label()),
        (_, p, d) if p == d => format!("{} prefill + decode", p.label()),
        (_, p, d) => format!("{} prefill, {} decode", p.label(), d.label()),
    };
    if streamed {
        text.push_str(", streamed from disk");
    }
    Accel {
        prefill,
        decode,
        decode_measured: measured,
        streamed,
        label: text,
    }
}

/// One accelerator of the machine, as the GUI's machine panel and `--models` list them.
#[derive(Clone, Debug)]
pub struct Accelerator {
    /// `cpu`, `metal`, `cuda`, `vulkan`, `neural-engine`, or `gpu` for the line saying there is
    /// none.
    pub backend: &'static str,
    pub name: String,
    pub available: bool,
    /// What `--device auto` (or the device asked for) uses: the CPU always, one GPU at most.
    pub used: bool,
    /// The CPU's SIMD features (`neon`, `dotprod`, `i8mm`, `sme`, `avx2`, `avx512f`, `simd128`,
    /// ...).
    pub features: Vec<&'static str>,
    pub memory_gb: Option<f64>,
    /// What models may take on this path: the CPU keeps in memory weights up to 60% of RAM
    /// (larger ones stream from storage); a discrete GPU decodes models whose weights fit in
    /// this; a unified one shares the RAM, up to the system's working-set limit.
    pub budget_gb: Option<f64>,
    pub unified: bool,
    /// Runs the prompt's matmuls / whole decode steps.
    pub prefill: bool,
    pub decode: bool,
    /// One sentence for people.
    pub status: String,
}

fn feature_name(f: &str) -> &str {
    match f {
        "neon" => "NEON",
        "dotprod" => "SDOT",
        "i8mm" => "i8mm",
        "sme" => "SME",
        "sve" => "SVE",
        "sve2" => "SVE2",
        "avx" => "AVX",
        "avx2" => "AVX2",
        "fma" => "FMA",
        "f16c" => "F16C",
        "avx512f" => "AVX-512",
        "avx512bw" => "AVX-512BW",
        "avx512vnni" => "AVX-512 VNNI",
        "avxvnni" => "AVX-VNNI",
        "avx512bf16" => "AVX-512 BF16",
        "simd128" => "wasm simd128",
        "sse4.2" => "SSE4.2",
        f => f,
    }
}

/// Whether Core ML loads, and why not.
fn neural_engine() -> Result<(), String> {
    #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
    return crate::coreml::available().map_err(|e| e.to_string());
    #[allow(unreachable_code)]
    Err("needs Apple Silicon and a build with the gpu feature".to_string())
}

/// The CPU, every GPU found (or a line saying there is none) and, on Apple systems, the Neural
/// Engine, with what each does for oxinfer. `device` is the `--device` preference.
pub fn accelerators(plat: &Platform, device: DevicePref) -> Vec<Accelerator> {
    let ram = plat.cpu.memory_bytes.map(|b| b as f64 / 1e9);
    let cpu_budget = ram.map(|r| RAM_FRACTION * r);
    let mut features = plat.cpu.features.clone();
    if crate::kernels::sme_available() {
        features.push("sme");
    }
    #[cfg(target_arch = "x86_64")]
    let kernels = format!("; kernels: {}", crate::kernels::simd::level_name());
    #[cfg(not(target_arch = "x86_64"))]
    let kernels = "";
    let mut out = vec![Accelerator {
        backend: "cpu",
        name: format!("{} CPU, {} cores", plat.cpu.arch, plat.cpu.cores),
        available: true,
        used: true,
        status: format!(
            "{}{kernels}; runs every model{}",
            if features.is_empty() {
                "no SIMD extension detected (portable kernels)".to_string()
            } else {
                features.iter().map(|f| feature_name(f)).collect::<Vec<_>>().join(", ")
            },
            cpu_budget.map_or(String::new(), |b| format!(
                ": up to {} of weights in memory (60% of RAM), larger models stream from disk",
                gb(b)
            )),
        ),
        features,
        memory_gb: ram,
        budget_gb: cpu_budget,
        unified: false,
        prefill: true,
        decode: true,
    }];
    let used = plat.pick(device);
    for g in &plat.gpus {
        let is_used = used.is_some_and(|u| u.kind == g.kind && u.index == g.index);
        let mem = g.memory_bytes.map(|b| b as f64 / 1e9);
        let decode = g.kind != GpuKind::Vulkan;
        let copies = copies_in_shared_memory(g.kind, g.unified_memory);
        let budget = if copies {
            mem.map(|m| ((GPU_FRACTION * m - GPU_HEADROOM_GB) / 2.0).max(0.0))
        } else if g.unified_memory {
            mem.map(|m| ram.map_or(m, |r| m.min(r)))
        } else {
            mem.map(|m| (GPU_FRACTION * m - GPU_HEADROOM_GB).max(0.0))
        };
        let does = if decode {
            "prefill + decode (Llama, Qwen3 / 3.5 / 3.8, Granite 4, Gemma 3 / 4; prompt matmuls for the others)"
        } else {
            "prompt matmuls only (bf16, Q8, Q4): decoding stays on the CPU"
        };
        let memory = match (g.unified_memory, budget, ram) {
            (true, Some(b), _) if copies => format!(
                "; shares the system's {} of RAM and keeps its own copy of the weights there: models up to {} decode on it",
                gb(ram.or(mem).unwrap_or(0.0)),
                gb(b)
            ),
            (true, Some(b), Some(r)) => format!("; shares the system's {} of RAM, {} of it usable by the GPU", gb(r), gb(b)),
            (true, Some(b), None) => format!("; shares the system's RAM, {} usable by the GPU", gb(b)),
            (false, Some(b), _) => format!(
                "; {} of its own memory: models up to {} decode on it",
                gb(mem.unwrap_or(0.0)),
                gb(b)
            ),
            _ => String::new(),
        };
        let status = if is_used {
            format!("{does}{memory}")
        } else if g.is_software() {
            "a software rasterizer (slower than the CPU kernels): not used".to_string()
        } else if device == DevicePref::Cpu {
            "not used (--device cpu)".to_string()
        } else {
            format!(
                "not used ({} is picked first); --device {} selects it",
                used.map_or("another device", |u| u.kind.as_str()),
                g.kind.as_str()
            )
        };
        out.push(Accelerator {
            backend: g.kind.as_str(),
            name: g.name.clone(),
            available: !g.is_software(),
            used: is_used,
            features: Vec::new(),
            memory_gb: mem,
            budget_gb: budget,
            unified: g.unified_memory,
            prefill: true,
            decode,
            status,
        });
    }
    if plat.gpus.is_empty() {
        out.push(Accelerator {
            backend: "gpu",
            name: "GPU".to_string(),
            available: false,
            used: false,
            features: Vec::new(),
            memory_gb: None,
            budget_gb: None,
            unified: false,
            prefill: false,
            decode: false,
            status: if cfg!(feature = "gpu") {
                "No GPU detected: models run on the CPU"
            } else {
                "This build has no GPU support (built without the `gpu` feature): models run on the CPU"
            }
            .to_string(),
        });
    }
    if cfg!(target_vendor = "apple") {
        let ne = neural_engine();
        out.push(Accelerator {
            backend: "neural-engine",
            name: "Neural Engine (Core ML)".to_string(),
            available: ne.is_ok(),
            used: ne.is_ok(),
            features: Vec::new(),
            memory_gb: None,
            budget_gb: None,
            unified: true,
            prefill: ne.is_ok(),
            decode: ne.is_ok(),
            status: match ne {
                Ok(()) => "available: runs models converted for Core ML (ANEMLL, anemll/* repos)".to_string(),
                Err(e) => format!("not available: {e}"),
            },
        });
    }
    out
}

impl Accelerator {
    pub fn to_json(&self) -> serde_json::Value {
        let r = |v: Option<f64>| v.map(|v| (v * 1e3).round() / 1e3);
        serde_json::json!({
            "backend": self.backend,
            "name": self.name,
            "available": self.available,
            "used": self.used,
            "features": self.features,
            "memory_gb": r(self.memory_gb),
            "budget_gb": r(self.budget_gb),
            "unified_memory": self.unified,
            "prefill": self.prefill,
            "decode": self.decode,
            "status": self.status,
        })
    }
}

impl Accel {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "label": self.label,
            "prefill": self.prefill.as_str(),
            "decode": self.decode.as_str(),
            "decode_measured": self.decode_measured,
            "streamed": self.streamed,
        })
    }
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
    if let Some(a) = hw.app_memory_gb {
        let _ = writeln!(
            s,
            "this app may use {} (weights kept in memory up to {})",
            gb(a),
            opt(hw.memory_budget_gb())
        );
    }
    for a in &hw.accelerators {
        let name = match a.backend {
            "cpu" => format!("CPU ({})", a.name),
            "gpu" | "neural-engine" => a.name.clone(),
            b => format!("GPU {} ({b})", a.name),
        };
        let _ = writeln!(s, "  {name}: {}", a.status);
    }
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
                v.accel.label.clone(),
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
            "app_memory_gb": ro(hw.app_memory_gb),
            "memory_budget_gb": ro(hw.memory_budget_gb()),
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
            "accelerators": hw.accelerators.iter().map(Accelerator::to_json).collect::<Vec<_>>(),
            "neural_engine": hw.neural_engine,
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
            "accelerator": row.recommended().accel.label,
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
                "accelerator": v.accel.label,
                "accel": v.accel.to_json(),
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
            app_memory_gb: None,
            gpu: gpu.map(|(kind, name, mem)| Gpu {
                kind,
                name: name.into(),
                memory_gb: Some(mem),
                unified: false,
            }),
            cache_dir: PathBuf::from("/nonexistent/oxinfer-advisor-test"),
            disk_free_gb: Some(disk),
            cpu_rates: [100.0, 100.0, 70.0, 60.0, 50.0, 50.0, 40.0],
            neural_engine: false,
            accelerators: Vec::new(),
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
                    Kind::Video => has(Cap::TextToVideo),
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
    fn phone_app_limit() {
        // An 8 GB iPhone whose app may use 6 GB: weights up to 4.8 GB stay in memory (60% of
        // the RAM, under 6 - 1 GB); Inkling-2 Mini's 8.5 GB phone file streams from flash.
        let mut h = unified(8.0, 5.7);
        h.app_memory_gb = Some(6.0);
        assert!((h.memory_budget_gb().unwrap() - 4.8).abs() < 1e-9);
        let rows = survey(&h);
        assert_eq!(
            row(&rows, "unsloth/Inkling-2-Mini-GGUF:UD-IQ2_XXS").variants[0].runs,
            Runs::Streamed
        );
        assert_ne!(row(&rows, "Qwen/Qwen3.5-0.8B").recommended().runs, Runs::Streamed);
        // A low app limit binds before the RAM: 3 GB leaves 1.9 GB of weights.
        h.app_memory_gb = Some(3.0);
        assert!((h.memory_budget_gb().unwrap() - (3.0 - APP_RESERVE_GB)).abs() < 1e-9);
        // A 16 GB iPad whose app may use 12 GB: 9.6 GB of weights, the phone file in memory,
        // UD-IQ3_XXS (11.7 GB) streamed.
        let mut h = unified(16.0, 12.0);
        h.app_memory_gb = Some(12.0);
        let rows = survey(&h);
        assert_ne!(
            row(&rows, "unsloth/Inkling-2-Mini-GGUF:UD-IQ2_XXS").variants[0].runs,
            Runs::Streamed
        );
        assert_eq!(
            row(&rows, "unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS").variants[0].runs,
            Runs::Streamed
        );
        let j = to_json(&h, &rows);
        assert_eq!(j["hardware"]["app_memory_gb"], 12.0);
        assert!(render(&h, &rows, false).contains("this app may use 12 GB"));
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
        // The accelerator fields: per variant, the recommended one per model, the hardware list.
        assert_eq!(j["models"][0]["variants"][0]["accelerator"], "CPU");
        assert_eq!(j["models"][0]["variants"][0]["accel"]["prefill"], "cpu");
        assert_eq!(j["models"][0]["variants"][0]["accel"]["decode"], "cpu");
        assert_eq!(j["models"][0]["accelerator"], "CPU");
        assert!(j["hardware"]["accelerators"].is_array());
        assert_eq!(j["hardware"]["neural_engine"], false);
    }

    fn unified(ram: f64, gpu: f64) -> Hardware {
        let mut h = hw(ram, Some((GpuKind::Metal, "Apple M5 Max", gpu)), 500.0);
        h.gpu.as_mut().unwrap().unified = true;
        h
    }

    fn label<'a>(rows: &'a [Row], repo: &str, i: usize) -> &'a str {
        &row(rows, repo).variants[i].accel.label
    }

    #[test]
    fn accelerator_per_model() {
        // Metal, unified memory: the families with a decode graph run whole on the GPU, the
        // others only their prompt there.
        let rows = survey(&unified(128.0, 110.0));
        let a = &row(&rows, "Qwen/Qwen3.5-9B").variants[0].accel;
        assert_eq!((a.prefill, a.decode), (Unit::Gpu(GpuKind::Metal), Unit::Gpu(GpuKind::Metal)));
        assert_eq!(a.label, "GPU (Metal) prefill + decode");
        assert_eq!(
            label(&rows, "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", 0),
            "GPU (Metal) prefill + decode"
        );
        assert_eq!(
            label(&rows, "mlx-community/gemma-4-26b-a4b-it-4bit", 0),
            "GPU (Metal) prefill + decode"
        );
        assert_eq!(
            label(&rows, "unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL", 0),
            "GPU (Metal) prefill, GPU or CPU decode (the faster), streamed from disk"
        );
        // Qwen2.5 (attention biases), LFM3 (convolutions), DeepSeek V4 (no graph): CPU decode.
        assert_eq!(label(&rows, "MiniMaxAI/SynLogic-7B", 0), "GPU (Metal) prefill, CPU decode");
        assert_eq!(label(&rows, "LiquidAI/LFM3-1.7B", 0), "GPU (Metal) prefill, CPU decode");
        // FP8 has Metal GEMMs, no decode matvec; NVFP4 neither.
        assert_eq!(label(&rows, "MiniMaxAI/MiniMax-H3-Image-PE", 0), "CPU");
        // Folded ternary weights: the CPU alone.
        assert_eq!(label(&rows, "prism-ml/Ternary-Bonsai-2-2B-gguf:PTQ1_0", 0), "CPU");
        // A streamed mixture of experts on unified memory: both are timed.
        let small = survey(&unified(64.0, 50.0));
        let v = &row(&small, "Qwen/Qwen3.6-35B-A3B").variants[0];
        assert!(v.accel.decode_measured && v.accel.streamed, "{:?}", v.accel);
        assert_eq!(
            v.accel.label,
            "GPU (Metal) prefill, GPU or CPU decode (the faster), streamed from disk"
        );
        // The other kinds.
        assert_eq!(label(&rows, "openai/whisper-small", 0), "GPU (Metal) encoder, CPU decode");
        assert_eq!(label(&rows, "stabilityai/sd-turbo", 0), "GPU (Metal)");
        assert_eq!(label(&rows, "BAAI/bge-m3", 0), "GPU (Metal)");
        assert_eq!(label(&rows, "unsloth/orpheus-3b-0.1-ft", 0), "GPU (Metal) prefill + decode");
        assert_eq!(label(&rows, "Zyphra/ZONOS3-Mini", 0), "CPU");
        assert_eq!(label(&rows, "MiniMaxAI/MiniMax-H3", 0), "CPU, streamed from disk");

        // CUDA: no GGUF, no Qwen4-Exp; decoding needs the weights in VRAM.
        let rows = survey(&hw(128.0, Some((GpuKind::Cuda, "NVIDIA GeForce RTX 5080", 16.6)), 500.0));
        assert_eq!(label(&rows, "Qwen/Qwen3.5-9B", 0), "GPU (CUDA) prefill, CPU decode");
        assert_eq!(label(&rows, "Qwen/Qwen3.5-9B", 1), "GPU (CUDA) prefill + decode");
        assert_eq!(label(&rows, "bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0", 0), "CPU");
        assert_eq!(label(&rows, "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF", 0), "CPU");
        assert_eq!(label(&rows, "stabilityai/sd-turbo", 0), "GPU (CUDA)");
        // Vulkan: prompt matmuls only.
        let rows = survey(&hw(128.0, Some((GpuKind::Vulkan, "Some GPU", 24.0)), 500.0));
        assert_eq!(label(&rows, "Qwen/Qwen3.5-0.8B", 0), "GPU (Vulkan) prefill, CPU decode");
        assert_eq!(label(&rows, "stabilityai/sd-turbo", 0), "GPU (Vulkan) matmuls, the rest on the CPU");
        // No GPU.
        let rows = survey(&hw(128.0, None, 500.0));
        assert!(rows.iter().all(|r| {
            r.variants
                .iter()
                .all(|v| v.accel.prefill == Unit::Cpu && v.accel.decode == Unit::Cpu)
        }));

        // Core ML packages: the Neural Engine, or a note that it is missing.
        static ANE: Model = Model::new("anemll/anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4", Kind::Chat, 0.6, 1.2, &[Cap::Text]);
        let mut h = unified(128.0, 110.0);
        assert_eq!(
            Row::new(&ANE, &h).variants[0].accel.label,
            "Neural Engine (Core ML), not available here"
        );
        h.neural_engine = true;
        let a = &Row::new(&ANE, &h).variants[0].accel;
        assert_eq!(
            (a.prefill, a.decode, a.label.as_str()),
            (Unit::NeuralEngine, Unit::NeuralEngine, "Neural Engine (Core ML)")
        );
    }

    #[test]
    fn machine_accelerators() {
        use crate::platform::{CpuInfo, GpuInfo};
        let cpu = CpuInfo {
            arch: "aarch64",
            os: "macos",
            features: vec!["neon", "dotprod", "i8mm"],
            cores: 18,
            memory_bytes: Some(137_400_000_000),
        };
        let metal = GpuInfo {
            kind: GpuKind::Metal,
            index: 0,
            name: "Apple M5 Max".into(),
            memory_bytes: Some(115_400_000_000),
            driver: String::new(),
            unified_memory: true,
        };
        let plat = Platform {
            cpu: cpu.clone(),
            gpus: vec![metal],
            notes: Vec::new(),
        };
        let list = accelerators(&plat, DevicePref::Auto);
        assert_eq!(list[0].backend, "cpu");
        assert!(list[0].status.starts_with("NEON, SDOT, i8mm"), "{}", list[0].status);
        assert!((list[0].budget_gb.unwrap() - 0.6 * 137.4).abs() < 1e-6);
        let g = &list[1];
        assert_eq!(
            (g.backend, g.used, g.prefill, g.decode, g.unified),
            ("metal", true, true, true, true)
        );
        assert!(
            g.status
                .contains("shares the system's 137 GB of RAM, 115 GB of it usable by the GPU"),
            "{}",
            g.status
        );
        assert_eq!(list.iter().any(|a| a.backend == "neural-engine"), cfg!(target_vendor = "apple"));
        // --device cpu: the GPU is listed, not used.
        let list = accelerators(&plat, DevicePref::Cpu);
        assert!(!list[1].used && list[1].status.contains("--device cpu"));
        // CUDA beside Vulkan on one card: CUDA is used, its own memory bounds decoding.
        let card = |kind, mem| GpuInfo {
            kind,
            index: 0,
            name: "NVIDIA GeForce RTX 5080".into(),
            memory_bytes: Some(mem),
            driver: String::new(),
            unified_memory: false,
        };
        let plat = Platform {
            cpu: cpu.clone(),
            gpus: vec![card(GpuKind::Cuda, 16_600_000_000), card(GpuKind::Vulkan, 17_100_000_000)],
            notes: Vec::new(),
        };
        let list = accelerators(&plat, DevicePref::Auto);
        assert!(list[1].used && list[1].decode && !list[1].unified);
        assert!((list[1].budget_gb.unwrap() - (0.92 * 16.6 - 1.0)).abs() < 1e-6);
        // A CUDA device sharing the system's memory (DGX Spark): the device's copy of the
        // weights is a second one in it.
        let mut spark = card(GpuKind::Cuda, 130_660_000_000);
        spark.unified_memory = true;
        let spark = Platform {
            cpu: cpu.clone(),
            gpus: vec![spark],
            notes: Vec::new(),
        };
        let s = &accelerators(&spark, DevicePref::Auto)[1];
        assert!(s.used && s.decode && s.unified);
        assert!((s.budget_gb.unwrap() - (0.92 * 130.66 - 1.0) / 2.0).abs() < 1e-6);
        assert!(
            s.status
                .contains("keeps its own copy of the weights there: models up to 60 GB decode on it"),
            "{}",
            s.status
        );
        assert!(fits_device(GpuKind::Cuda, true, 130.66, 59.0) && !fits_device(GpuKind::Cuda, true, 130.66, 61.0));
        assert!(fits_device(GpuKind::Cuda, false, 130.66, 61.0) && fits_device(GpuKind::Metal, true, 130.66, 61.0));
        assert!(
            !list[2].used && !list[2].decode && list[2].status.contains("cuda is picked first"),
            "{}",
            list[2].status
        );
        // No GPU.
        let plat = Platform {
            cpu,
            gpus: Vec::new(),
            notes: Vec::new(),
        };
        let list = accelerators(&plat, DevicePref::Auto);
        assert!(list[1].status.ends_with("models run on the CPU"), "{}", list[1].status);
        assert_eq!(list[1].to_json()["available"], false);
    }
}
