use anyhow::{Context, Result, bail};
use oxinfer::chat::{self, ChatFormat};
use oxinfer::engine::{load_tokenizer, resolve_file, resolve_model, save_quantized};
use oxinfer::generate::{self, Flow, SpecStats, TextLm, TextStream};
use oxinfer::model::{Model, Quant, Session};
use oxinfer::sampler::Sampler;
use oxinfer::{config, drafter, platform, pool, prof, qwen, safetensors};
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

#[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
mod anemll_cli;
#[cfg(feature = "media")]
mod whisper_cli;

struct Args {
    model: PathBuf,
    prompt: Option<String>,
    system: Option<String>,
    no_system: bool,
    tools: bool,
    print_prompt: bool,
    think: bool,
    raw: bool,
    chat: bool,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    threads: Option<usize>,
    qos: pool::Qos,
    efficiency_threads: usize,
    preload: bool,
    mmap: bool,
    quant: Quant,
    profile: bool,
    dump_logits: Option<PathBuf>,
    show_tokens: bool,
    prefill_chunk: Option<usize>,
    draft: Option<PathBuf>,
    draft_tokens: Option<usize>,
    mtp: bool,
    mtp_fixed: bool,
    mtp_from: Option<PathBuf>,
    mtp_cost: Option<bool>,
    stream: bool,
    cache_budget_gb: Option<f64>,
    resident_mb: Option<u64>,
    prefetch_mb: Option<u64>,
    resident_mode: oxinfer::stream::ResidentMode,
    flash_mbps: Option<f64>,
    stream_io: Option<oxinfer::stream::IoMethod>,
    stream_trace: bool,
    stream_threads: Option<usize>,
    expert_cache_mb: Option<u64>,
    expert_predict: bool,
    images: Vec<PathBuf>,
    audios: Vec<PathBuf>,
    videos: Vec<PathBuf>,
    video: oxinfer::video::VideoOptions,
    dump_mm: Option<PathBuf>,
    max_soft_tokens: usize,
    cache_dir: Option<PathBuf>,
    download_only: bool,
    save_quantized: Option<PathBuf>,
    // Image generation (diffusers folders).
    output: Option<PathBuf>,
    steps: Option<usize>,
    guidance: Option<f32>,
    negative: String,
    size: Option<(usize, usize)>,
    sampler: Option<String>,
    loras: Vec<oxinfer::sd::lora::LoraSpec>,
    device: platform::DevicePref,
    devices: bool,
    embed: bool,
    prompt_name: Option<String>,
    similarity: bool,
    sparse: bool,
    colbert: bool,
    dim: Option<usize>,
    // ANEMLL Core ML models.
    coreml_units: Option<String>,
    coreml_prefill: Option<String>,
    // Typed decisions (Jev models).
    decide: Option<String>,
    state: Option<String>,
    question: Option<String>,
    options: Vec<String>,
    decide_json: Option<PathBuf>,
    decision_from: Option<String>,
    // Speech to text (Whisper models).
    transcribe: Option<PathBuf>,
    language: Option<String>,
    translate: bool,
    timestamps: bool,
    word_timestamps: bool,
    initial_prompt: Option<String>,
    prefix: Option<String>,
    format: Option<String>,
    precision: Option<String>,
    fallback: bool,
    condition_on_prev: bool,
}

fn usage() -> ! {
    eprintln!(
        "usage: oxinfer --model DIR|org/name [options]\n\
         \n\
         A HuggingFace repo id is fetched into the cache (~/.cache/oxinfer/hub or $OXINFER_CACHE)\n\
         on first use; downloads resume. Set HF_TOKEN for gated repos.\n\
         \n\
         options:\n\
         \x20 --prompt TEXT        single prompt (default: read from stdin)\n\
         \x20 --chat               interactive multi-turn chat\n\
         \x20 --system TEXT        system prompt (default: a short one with the local date and time)\n\
         \x20 --no-system          no system prompt\n\
         \x20 --tools              let the model call built-in tools (date and time, calculator)\n\
         \x20 --print-prompt       print the formatted prompt and exit\n\
         \x20 --think              enable thinking mode\n\
         \x20 --raw                no chat template (prompt is fed as-is, with <bos>)\n\
         \x20 --max-tokens N       max new tokens (default 1024)\n\
         \x20 --temp F             temperature, 0 = greedy (default 0)\n\
         \x20 --top-k N            top-k (default 64)\n\
         \x20 --top-p F            top-p (default 0.95)\n\
         \x20 --seed N             RNG seed\n\
         \x20 --threads N          worker threads (default: all cores)\n\
         \x20 --qos C              thread QoS on Apple platforms: interactive (default), initiated,\n\
         \x20                      utility, background (efficiency cores only)\n\
         \x20 --efficiency-threads N  of the threads, N at background QoS (efficiency cores)\n\
         \x20 --mmap               map weights from disk instead of copying into RAM\n\
         \x20 --quant bf16|q8|q4   weight precision for streamed matmuls (default bf16)\n\
         \x20 --preload            with --mmap, touch all weight pages before starting\n\
         \x20 --profile            print per-operation timing breakdown\n\
         \x20 --prefill-chunk N    tokens per prefill chunk (default 256 on CPU, 1024 with a GPU backend)\n\
         \x20 --draft DIR          speculative decoding with a Gemma 4 assistant (drafter) model; greedy only\n\
         \x20 --draft-tokens N     max draft tokens per round (default 6 with --draft, adaptive from 3;\n\
         \x20                      3 with --mtp, 8 with --mtp --stream)\n\
         \x20 --mtp                Qwen3.5/3.8: speculative decoding with the checkpoint's multi-token\n\
         \x20                      prediction layer (greedy and sampled; same output distribution)\n\
         \x20 --mtp-fixed          with --mtp, always draft --draft-tokens (default: adaptive 1..N)\n\
         \x20 --mtp-from DIR       implies --mtp, with the MTP weights from another checkpoint (e.g. the\n\
         \x20                      original one when the model is a converted copy without them)\n\
         \x20 --mtp-cost           with --mtp, pick each round's draft length from measured round times and\n\
         \x20                      acceptance (default with --stream; --no-mtp-cost turns it off)\n\
         \x20 --stream             Qwen: stream weights from the mapped file (models larger than RAM): a\n\
         \x20                      reader thread fetches the next layers while one computes\n\
         \x20 --cache-budget-gb F  with --stream, page cache the model may use: a resident set of\n\
         \x20                      F - prefetch stays cached, the other layers are dropped after use\n\
         \x20 --resident-mb N      with --stream, resident set size (instead of the budget's)\n\
         \x20 --prefetch-mb N      with --stream, read-ahead window (default 768)\n\
         \x20 --resident-mode M    cache (default) | lock (mlock) | copy (anonymous memory)\n\
         \x20 --flash-mbps N       with --stream, emulate storage reading N MB/s (or $OXINFER_FLASH_MBPS)\n\
         \x20 --stream-io M        advise (default on Apple) | pread | touch\n\
         \x20 --stream-trace       with --stream, one line of IO statistics per decoding step\n\
         \x20 --stream-threads N   with --stream, reader threads (default 2)\n\
         \x20 --expert-cache-mb N  with --stream, page cache kept for recently used MoE experts (which\n\
         \x20                      are read on demand; default: what the budget leaves)\n\
         \x20 --expert-predict     with --stream, read the next MoE layer's guessed experts ahead\n\
         \x20 --image FILE         attach an image (repeatable; PNG/JPEG/WebP/GIF/BMP via oxideav)\n\
         \x20 --audio FILE         attach an audio clip (repeatable; WAV/MP3/FLAC/AAC/OGG via oxideav)\n\
         \x20 --video FILE|DIR     attach a video (repeatable; MP4/MOV/MKV/WebM/AVI with H.264, H.265,\n\
         \x20                      AV1, VP8, VP9, MPEG-4 or MJPEG via oxideav), or a directory of frames\n\
         \x20 --video-fps F        frames sampled per second (Qwen default 2; Gemma 4: uniform frames)\n\
         \x20 --video-frames N     most frames sampled (Qwen default 768; Gemma 4: the count, default 32)\n\
         \x20 --frames-fps F       frame rate of a directory of frames (default 24)\n\
         \x20 --max-soft-tokens N  image token budget: Gemma 4: 70, 140, 280 (default), 560 or 1120;\n\
         \x20                      Qwen: merged tokens of 32x32 pixels (default 16384)\n\
         \x20 --dump-mm PREFIX     write multimodal soft tokens / patches as f32 files (for reference checks)\n\
         \x20 --cache-dir DIR      where fetched models are cached\n\
         \x20 --download-only      fetch the model (and drafter) and exit\n\
         \x20 --save-quantized DIR write a text-only checkpoint with the weights already quantized to\n\
         \x20                      --quant (loads zero-copy with no quantization pass; for browsers, mmap)\n\
         \n\
         image generation (a diffusers folder: Stable Diffusion 1.x / 2.x, SD-Turbo):\n\
         \x20 -o, --output FILE     PNG to write (default image.png)\n\
         \x20 --steps N            denoising steps (default 1 for turbo models, else 25)\n\
         \x20 --guidance F         classifier-free guidance (default 0 for turbo, else 7.5)\n\
         \x20 --negative TEXT      negative prompt (with guidance > 1)\n\
         \x20 --size WxH           image size, multiples of 64 (default the model's)\n\
         \x20 --sampler NAME       euler or euler_a (default euler_a)\n\
         \x20 --seed N             noise seed (default 42)\n\
         \x20 --lora FILE[:SCALE]  merge a LoRA (kohya or diffusers safetensors; repeatable; scale 1);\n\
         \x20                      FILE may be org/repo/file.safetensors on Hugging Face\n\
         \x20 --devices            list detected CPU features and GPUs, then exit\n\
         \x20 --device D           auto (default) | cpu | cuda | metal | vulkan\n\
         \x20 --dump-logits FILE   write last-prompt-token logits (f32 LE) and exit\n\
         \n\
         ANEMLL Core ML models (a meta.yaml folder; macOS 15 / iOS 18, Apple Neural Engine):\n\
         \x20 --coreml-units U     ane (CPU + Neural Engine, default) | all | gpu (CPU + GPU) | cpu\n\
         \x20 --coreml-prefill M   auto (default: batch if meta.yaml allows it, else single; pad for\n\
         \x20                      tools/coreml_convert.py packages without recurrent state) | batch |\n\
         \x20                      pad (batch + padded last batch: fastest, not bit-exact) | single\n\
         \n\
         embeddings (sentence-transformers models are detected; --embed forces it):\n\
         \x20 --embed              embed the prompt, or each stdin line: one JSON array per line\n\
         \x20 --prompt-name NAME   the model's prompt to prepend (e.g. query, document)\n\
         \x20 --dim N              truncate to N dimensions (Matryoshka) and renormalize\n\
         \x20 --similarity         print the cosine similarity matrix instead\n\
         \x20 --sparse, --colbert  BGE-M3 lexical weights / multi-vector embeddings instead\n\
         \x20 --show-tokens        print token ids\n\
         \n\
         typed decisions (Jev models: a bundle directory or repo id such as autotrust/JEV-9B, or any\n\
         base model with --decision-from):\n\
         \x20 --decide KIND        noul (yes/no), score (0-5) or choice; prints the distribution\n\
         \x20 --state TEXT         what the question is about (a JSON object is used as such)\n\
         \x20 --question TEXT      the question\n\
         \x20 --option TEXT        a choice option (repeatable, 2-256)\n\
         \x20 --decide-json FILE   /v1/decide requests (an object, an array or one object per line;\n\
         \x20                      - for stdin); prints one JSON response per line\n\
         \x20 --decision-from S    the decision assets (adapter, head, configs) of bundle S, a directory\n\
         \x20                      or repo id, on top of --model (fetches the small files only)\n\
         \n\
         speech to text (Whisper models: openai/whisper-*, distil-whisper/*; detected from config.json):\n\
         \x20 --transcribe FILE    audio to transcribe (WAV/MP3/FLAC/AAC/OGG via oxideav; any length)\n\
         \x20 --language XX        spoken language (code or name; default: detected from the first 30 s)\n\
         \x20 --translate          translate to English instead of transcribing\n\
         \x20 --timestamps         print timed segments (always on past 30 s of audio)\n\
         \x20 --word-timestamps    also word-level times (cross-attention alignment)\n\
         \x20 --format F           text (default), json, srt or vtt; --json for json\n\
         \x20 --initial-prompt T   context text for the decoder (names, spelling, style)\n\
         \x20 --prefix T           text the transcript starts with\n\
         \x20 --fallback           temperature fallback on repetitive or unlikely windows (openai defaults)\n\
         \x20 --condition-on-prev  feed each window the previous text\n\
         \x20 --precision P        weights: f32 (matches transformers token for token), f16, bf16 (default);\n\
         \x20                      --quant q8|q4 quantizes the decoder (--temp T samples at T)"
    );
    std::process::exit(2)
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        model: PathBuf::new(),
        prompt: None,
        system: None,
        no_system: false,
        tools: false,
        print_prompt: false,
        think: false,
        raw: false,
        chat: false,
        max_tokens: 1024,
        temperature: 0.0,
        top_k: 64,
        top_p: 0.95,
        seed: 42,
        threads: None,
        qos: pool::Qos::UserInteractive,
        efficiency_threads: 0,
        preload: false,
        mmap: false,
        quant: Quant::Bf16,
        profile: false,
        dump_logits: None,
        show_tokens: false,
        prefill_chunk: None,
        draft: None,
        draft_tokens: None,
        mtp: false,
        mtp_fixed: false,
        mtp_from: None,
        mtp_cost: None,
        stream: false,
        cache_budget_gb: None,
        resident_mb: None,
        prefetch_mb: None,
        resident_mode: oxinfer::stream::ResidentMode::Cache,
        flash_mbps: None,
        stream_io: None,
        stream_trace: false,
        stream_threads: None,
        expert_cache_mb: None,
        expert_predict: false,
        images: Vec::new(),
        audios: Vec::new(),
        videos: Vec::new(),
        video: Default::default(),
        dump_mm: None,
        max_soft_tokens: 0,
        cache_dir: None,
        download_only: false,
        save_quantized: None,
        output: None,
        steps: None,
        guidance: None,
        negative: String::new(),
        size: None,
        sampler: None,
        loras: Vec::new(),
        device: platform::DevicePref::Auto,
        devices: false,
        embed: false,
        prompt_name: None,
        similarity: false,
        sparse: false,
        colbert: false,
        dim: None,
        coreml_units: None,
        coreml_prefill: None,
        decide: None,
        state: None,
        question: None,
        options: Vec::new(),
        decide_json: None,
        decision_from: None,
        transcribe: None,
        language: None,
        translate: false,
        timestamps: false,
        word_timestamps: false,
        initial_prompt: None,
        prefix: None,
        format: None,
        precision: None,
        fallback: false,
        condition_on_prev: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut val = || it.next().with_context(|| format!("{k} needs a value"));
        match k.as_str() {
            "--model" | "-m" => a.model = PathBuf::from(val()?),
            "--prompt" | "-p" => a.prompt = Some(val()?),
            "--system" => a.system = Some(val()?),
            "--no-system" => a.no_system = true,
            "--tools" => a.tools = true,
            "--print-prompt" => a.print_prompt = true,
            "--think" => a.think = true,
            "--raw" => a.raw = true,
            "--chat" => a.chat = true,
            "--max-tokens" | "-n" => a.max_tokens = val()?.parse()?,
            "--temp" | "-t" => a.temperature = val()?.parse()?,
            "--top-k" => a.top_k = val()?.parse()?,
            "--top-p" => a.top_p = val()?.parse()?,
            "--seed" => a.seed = val()?.parse()?,
            "--threads" | "-j" => a.threads = Some(val()?.parse()?),
            "--qos" => {
                a.qos = match val()?.as_str() {
                    "interactive" => pool::Qos::UserInteractive,
                    "initiated" => pool::Qos::UserInitiated,
                    "utility" => pool::Qos::Utility,
                    "background" => pool::Qos::Background,
                    q => bail!("--qos: unknown class {q} (interactive, initiated, utility, background)"),
                }
            }
            "--efficiency-threads" => a.efficiency_threads = val()?.parse()?,
            "--preload" => a.preload = true,
            "--mmap" => a.mmap = true,
            "--quant" => {
                a.quant = match val()?.as_str() {
                    "bf16" => Quant::Bf16,
                    "q8" | "int8" => Quant::Q8,
                    "q4" | "q4_0" => Quant::Q4,
                    other => bail!("unknown quantization {other} (use bf16, q8 or q4)"),
                }
            }
            "--profile" => a.profile = true,
            "--prefill-chunk" => a.prefill_chunk = Some(val()?.parse()?),
            "--draft" => a.draft = Some(PathBuf::from(val()?)),
            "--draft-tokens" => a.draft_tokens = Some(val()?.parse()?),
            "--mtp" => a.mtp = true,
            "--mtp-fixed" => a.mtp_fixed = true,
            "--mtp-from" => {
                a.mtp_from = Some(PathBuf::from(val()?));
                a.mtp = true;
            }
            "--mtp-cost" => a.mtp_cost = Some(true),
            "--no-mtp-cost" => a.mtp_cost = Some(false),
            "--stream" => {
                a.stream = true;
                a.mmap = true;
            }
            "--cache-budget-gb" => a.cache_budget_gb = Some(val()?.parse()?),
            "--resident-mb" => a.resident_mb = Some(val()?.parse()?),
            "--prefetch-mb" => a.prefetch_mb = Some(val()?.parse()?),
            "--resident-mode" => {
                use oxinfer::stream::ResidentMode;
                a.resident_mode = match val()?.as_str() {
                    "cache" => ResidentMode::Cache,
                    "lock" => ResidentMode::Lock,
                    "copy" => ResidentMode::Copy,
                    other => bail!("unknown resident mode {other} (cache, lock or copy)"),
                }
            }
            "--flash-mbps" => a.flash_mbps = Some(val()?.parse()?),
            "--stream-io" => {
                use oxinfer::stream::IoMethod;
                a.stream_io = Some(match val()?.as_str() {
                    "advise" => IoMethod::Advise,
                    "pread" => IoMethod::Pread,
                    "touch" => IoMethod::Touch,
                    other => bail!("unknown streaming IO {other} (advise, pread or touch)"),
                })
            }
            "--stream-trace" => a.stream_trace = true,
            "--stream-threads" => a.stream_threads = Some(val()?.parse()?),
            "--expert-cache-mb" => a.expert_cache_mb = Some(val()?.parse()?),
            "--expert-predict" => a.expert_predict = true,
            "--image" => a.images.push(PathBuf::from(val()?)),
            "--audio" => a.audios.push(PathBuf::from(val()?)),
            "--video" => a.videos.push(PathBuf::from(val()?)),
            "--video-fps" => a.video.fps = Some(val()?.parse()?),
            "--video-frames" => a.video.max_frames = Some(val()?.parse()?),
            "--frames-fps" => a.video.frames_fps = Some(val()?.parse()?),
            "--dump-mm" => a.dump_mm = Some(PathBuf::from(val()?)),
            "--max-soft-tokens" => a.max_soft_tokens = val()?.parse()?,
            "--cache-dir" => a.cache_dir = Some(PathBuf::from(val()?)),
            "--download-only" => a.download_only = true,
            "--save-quantized" => {
                a.save_quantized = Some(PathBuf::from(val()?));
                a.mmap = true;
            }
            "--devices" => a.devices = true,
            "--output" | "-o" => a.output = Some(PathBuf::from(val()?)),
            "--steps" => a.steps = Some(val()?.parse()?),
            "--guidance" | "--cfg" => a.guidance = Some(val()?.parse()?),
            "--negative" => a.negative = val()?,
            "--sampler" => a.sampler = Some(val()?),
            "--lora" => a.loras.push(oxinfer::sd::lora::LoraSpec::parse(&val()?)?),
            "--size" => {
                let v = val()?;
                let (w, h) = v
                    .split_once(['x', 'X'])
                    .with_context(|| format!("--size expects WIDTHxHEIGHT, got {v}"))?;
                a.size = Some((w.parse()?, h.parse()?));
            }
            "--device" => {
                let v = val()?;
                a.device = platform::DevicePref::parse(&v).with_context(|| format!("unknown device {v} (auto|cpu|cuda|metal|vulkan)"))?;
            }
            "--dump-logits" => a.dump_logits = Some(PathBuf::from(val()?)),
            "--embed" => a.embed = true,
            "--prompt-name" => a.prompt_name = Some(val()?),
            "--similarity" => a.similarity = true,
            "--sparse" => a.sparse = true,
            "--colbert" => a.colbert = true,
            "--dim" => a.dim = Some(val()?.parse()?),
            "--coreml-units" => a.coreml_units = Some(val()?),
            "--coreml-prefill" => a.coreml_prefill = Some(val()?),
            "--decide" => a.decide = Some(val()?),
            "--state" => a.state = Some(val()?),
            "--question" => a.question = Some(val()?),
            "--option" => a.options.push(val()?),
            "--decide-json" => a.decide_json = Some(PathBuf::from(val()?)),
            "--decision-from" => a.decision_from = Some(val()?),
            "--show-tokens" => a.show_tokens = true,
            "--transcribe" => a.transcribe = Some(PathBuf::from(val()?)),
            "--language" => a.language = Some(val()?),
            "--translate" => a.translate = true,
            "--timestamps" => a.timestamps = true,
            "--word-timestamps" => a.word_timestamps = true,
            "--initial-prompt" => a.initial_prompt = Some(val()?),
            "--prefix" => a.prefix = Some(val()?),
            "--format" => a.format = Some(val()?),
            "--json" => a.format = Some("json".into()),
            "--precision" => a.precision = Some(val()?),
            "--fallback" => a.fallback = true,
            "--condition-on-prev" => a.condition_on_prev = true,
            "-h" | "--help" => usage(),
            _ => {
                eprintln!("unknown argument {k}");
                usage()
            }
        }
    }
    if a.model.as_os_str().is_empty() && !a.devices {
        usage();
    }
    Ok(a)
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// The attachments of the command line.
fn mm_request(args: &Args) -> oxinfer::mm::MmRequest<'_> {
    oxinfer::mm::MmRequest {
        model: &args.model,
        images: &args.images,
        audios: &args.audios,
        videos: &args.videos,
        video: args.video.clone(),
        max_soft_tokens: args.max_soft_tokens,
        quant: args.quant,
        dump_mm: args.dump_mm.as_deref(),
    }
}

/// Incremental stdout printer: prints the stable decoded prefix of the token list.
struct Printer {
    stream: TextStream,
}

impl Printer {
    fn new(fmt: &dyn ChatFormat, think: bool, tok: &Tokenizer) -> Self {
        if fmt.starts_in_thinking(think) {
            let mut lock = std::io::stdout().lock();
            let _ = lock.write_all(b"\x1b[2m");
            let _ = lock.flush();
        }
        Self {
            stream: TextStream::new(fmt, think, tok),
        }
    }

    fn on_token(&mut self, id: u32) {
        if self.stream.on_token(id) {
            let mut lock = std::io::stdout().lock();
            let _ = lock.write_all(if self.stream.thinking() { b"\x1b[2m" } else { b"\x1b[0m\n" });
            let _ = lock.flush();
        }
    }

    fn flush_text(&mut self, tok: &Tokenizer, out: &[u32], final_: bool) {
        let text = self.stream.delta(tok, out, final_);
        let mut lock = std::io::stdout().lock();
        if !text.is_empty() {
            let _ = lock.write_all(text.as_bytes());
            let _ = lock.flush();
        }
        if final_ {
            let _ = lock.write_all(b"\x1b[0m");
            let _ = lock.flush();
        }
    }

    /// The decoding-loop callback: prints each new token.
    fn print(&mut self, tok: &Tokenizer, out: &[u32]) -> Flow {
        self.on_token(*out.last().unwrap());
        self.flush_text(tok, out, false);
        Flow::Continue
    }

    /// Prints the rest of a finished reply (without its stop token) and ends the line.
    fn finish(&mut self, tok: &Tokenizer, out: &[u32], eos: &[u32]) {
        let shown = if out.last().is_some_and(|e| eos.contains(e)) {
            &out[..out.len() - 1]
        } else {
            out
        };
        self.flush_text(tok, shown, true);
        println!();
    }
}

/// Greedy speculative decoding with the Gemma 4 drafter, printed as it goes. Produces exactly the
/// plain greedy sequence. `hidden_last` is the final-normed hidden state of the last prompt token.
#[allow(clippy::too_many_arguments)]
fn generate_speculative(
    model: &Model,
    drafter: &drafter::Drafter,
    sess: &mut Session,
    tok: &Tokenizer,
    hidden_last: Vec<f32>,
    k: usize,
    eos: &[u32],
    max_tokens: usize,
) -> (Vec<u32>, SpecStats, f64) {
    let mut printer = Printer::new(&chat::GemmaChat, false, tok);
    let (d, stats) = generate::decode_speculative(model, drafter, sess, hidden_last, k, eos, max_tokens, &mut |out| {
        printer.print(tok, out)
    });
    printer.finish(tok, &d.tokens, eos);
    (d.tokens, stats, d.secs)
}

/// Streams the model's reply to stdout. Returns (generated ids including stop token, forwards, seconds).
#[allow(clippy::too_many_arguments)]
fn generate<M: TextLm>(
    model: &M,
    sess: &mut M::Sess,
    tok: &Tokenizer,
    logits: Vec<f32>,
    sampler: &mut Sampler,
    eos: &[u32],
    max_tokens: usize,
    fmt: &dyn ChatFormat,
    think: bool,
) -> (Vec<u32>, usize, f64) {
    let mut printer = Printer::new(fmt, think, tok);
    let stream = STREAM.lock().unwrap().clone();
    let trace = TRACE.load(std::sync::atomic::Ordering::Relaxed);
    let s_start = stream.as_ref().map(|s| s.stats());
    let (mut s_last, mut t_last) = (s_start, Instant::now());
    let d = generate::decode(model, sess, logits, sampler, eos, max_tokens, &mut |out| {
        // Every token after the first follows one forward pass.
        if let (true, Some(s), Some(s0)) = (trace && out.len() > 1, &stream, &s_last) {
            let now = s.stats();
            eprintln!("[token: {}]", stream_line(&(now - *s0), 1, t_last.elapsed().as_secs_f64()));
            (s_last, t_last) = (Some(now), Instant::now());
        }
        printer.print(tok, out)
    });
    printer.finish(tok, &d.tokens, eos);
    if let (Some(s), Some(s0)) = (&stream, s_start) {
        let st = s.stats() - s0;
        eprintln!(
            "[stream: {}; {:.2} GB read per forward]",
            stream_line(&st, d.forwards, d.secs),
            st.bytes_read as f64 / 1e9 / d.forwards.max(1) as f64
        );
    }
    (d.tokens, d.forwards, d.secs)
}

fn report(prefill_tokens: usize, prefill_secs: f64, decode_tokens: usize, decode_secs: f64) {
    eprintln!(
        "[prefill {} tok in {:.2}s = {:.1} tok/s | decode {} tok in {:.2}s = {:.1} tok/s]",
        prefill_tokens,
        prefill_secs,
        prefill_tokens as f64 / prefill_secs.max(1e-9),
        decode_tokens,
        decode_secs,
        decode_tokens as f64 / decode_secs.max(1e-9),
    );
}

fn main() -> Result<()> {
    let mut args = parse_args()?;
    if args.devices {
        print!("{}", platform::detect().report());
        return Ok(());
    }
    // Resolve repo ids to cached local directories.
    let spec = args.model.to_string_lossy().to_string();
    args.model = resolve_model(&spec, args.cache_dir.as_deref())?;
    if let Some(d) = &args.draft {
        let spec = d.to_string_lossy().to_string();
        args.draft = Some(resolve_model(&spec, args.cache_dir.as_deref())?);
    }
    if let Some(d) = &args.mtp_from {
        let spec = d.to_string_lossy().to_string();
        args.mtp_from = Some(resolve_model(&spec, args.cache_dir.as_deref())?);
    }
    for l in &mut args.loras {
        l.path = resolve_file(&l.path.to_string_lossy(), args.cache_dir.as_deref())?;
    }
    if args.download_only {
        eprintln!("[model at {}]", args.model.display());
        return Ok(());
    }
    let threads = args.threads.unwrap_or_else(default_threads);
    pool::init_with(pool::PoolOptions {
        threads,
        qos: args.qos,
        efficiency_threads: args.efficiency_threads,
    });
    if args.decide.is_some() || args.decide_json.is_some() {
        return run_decide(&args);
    }
    if args.decision_from.is_some() {
        bail!("--decision-from goes with --decide or --decide-json");
    }
    run(args)
}

/// Typed decisions through the engine: one distribution per request.
fn run_decide(args: &Args) -> Result<()> {
    use oxinfer::decide::{DecisionKind, DecisionRequest};
    use oxinfer::engine::{Engine, EngineOptions};
    let mut opts = EngineOptions::default();
    opts.quant = args.quant;
    opts.mmap = args.mmap;
    opts.preload = args.preload;
    opts.device = args.device;
    opts.prefill_chunk = args.prefill_chunk;
    opts.cache_dir = args.cache_dir.clone();
    opts.decision_from = args.decision_from.clone();
    if args.stream {
        opts.stream = Some(Default::default());
    }
    let mut e = Engine::load(&args.model.to_string_lossy(), &opts)?;
    let info = e.info().clone();
    let d = e
        .decision_assets()
        .context("this model has no decision assets: use a Jev bundle or --decision-from")?;
    eprintln!(
        "[{} {} ({}, {}) on {}, loaded in {:.1}s | decision: {} ({:?}{})]",
        info.model_type,
        info.path.display(),
        info.format,
        info.quant.as_str(),
        info.device,
        info.load_secs,
        d.model_name,
        d.readout,
        match &d.adapter {
            Some(a) => format!(", LoRA r={} on {} modules, {:.0} MB", a.rank, a.modules, a.bytes as f64 / 1e6),
            None => String::new(),
        }
    );
    let (model_name, protocol) = (d.model_name.clone(), d.protocol());
    if let Some(path) = &args.decide_json {
        let text = if path.as_os_str() == "-" {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s
        } else {
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?
        };
        let reqs: Vec<serde_json::Value> = match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(serde_json::Value::Array(a)) => a,
            Ok(v) => vec![v],
            Err(_) => text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(serde_json::from_str)
                .collect::<Result<_, _>>()
                .context("parse the requests (a JSON object, an array, or one object per line)")?,
        };
        for r in &reqs {
            let dec = e.decide(&DecisionRequest::from_json(r)?)?;
            println!("{}", dec.to_json(&model_name, &protocol));
            eprintln!(
                "[{} tokens, prefill {:.3}s ({:.0} tok/s)]",
                dec.stats.prompt_tokens,
                dec.stats.prefill_secs,
                dec.stats.prefill_tok_s()
            );
        }
        return Ok(());
    }
    let kind = args.decide.as_deref().unwrap_or_default();
    let kind = DecisionKind::parse(kind).with_context(|| format!("--decide {kind}: expected noul, score or choice"))?;
    let state = args.state.clone().unwrap_or_default();
    let state = match serde_json::from_str::<serde_json::Value>(&state) {
        Ok(v @ serde_json::Value::Object(_)) => v,
        _ => serde_json::Value::String(state),
    };
    let question = args.question.clone().context("--decide needs --question")?;
    let dec = e.decide(&DecisionRequest::new(kind, state, question, args.options.clone()))?;
    let w = dec.options.iter().map(|o| o.chars().count()).max().unwrap_or(0).min(60);
    for (i, (o, p)) in dec.options.iter().zip(&dec.probabilities).enumerate() {
        let bar = "#".repeat((p * 40.0).round() as usize);
        let mark = if i == dec.best_index { '*' } else { ' ' };
        println!("{mark} {o:<w$}  {p:.4}  {bar}");
    }
    if let Some(s) = dec.expected_score() {
        println!("  expected score {s:.3}");
    }
    eprintln!(
        "[{} {} | {} prompt tokens, prefill {:.3}s ({:.0} tok/s), total {:.3}s]",
        dec.kind.as_str(),
        dec.adaptation.as_str(),
        dec.stats.prompt_tokens,
        dec.stats.prefill_secs,
        dec.stats.prefill_tok_s(),
        dec.stats.total_secs
    );
    Ok(())
}

fn run(mut args: Args) -> Result<()> {
    // ANEMLL models (meta.yaml + compiled Core ML parts) run on Core ML, not on our kernels.
    #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
    if anemll_cli::detect(&args.model) {
        return anemll_cli::run(&args);
    }
    let mlmodelc = |d: &std::path::Path| {
        std::fs::read_dir(d).is_ok_and(|rd| rd.flatten().any(|e| e.file_name().to_string_lossy().ends_with(".mlmodelc")))
    };
    if args.model.join("meta.yaml").is_file() && mlmodelc(&args.model) {
        bail!(
            "{}: an ANEMLL Core ML model; these run only in Apple Silicon macOS / iOS builds with the `gpu` feature",
            args.model.display()
        );
    }
    let plat = platform::detect();
    eprintln!("[platform: {}]", plat.summary());
    let gpu = oxinfer::gpu::init(&plat, args.device);
    match gpu {
        Some(b) => eprintln!("[device: {} (prefill matmuls on the GPU, everything else on the CPU)]", b.name()),
        None => eprintln!("[device: cpu]"),
    }
    // Bigger prefill chunks keep the GPU busier (577 vs 431 tok/s on E4B q4 at 1024 vs 256);
    // on the CPU 256 keeps the activations in cache.
    args.prefill_chunk = Some(args.prefill_chunk.unwrap_or(if gpu.is_some() { 1024 } else { 256 }));
    if oxinfer::sd::is_diffusers(&args.model) {
        return run_image(&args);
    }
    if args.embed || oxinfer::embed::is_embedding_model(&args.model) {
        return run_embed(&args);
    }
    let model_type = config::model_type(&args.model)?;
    if model_type == "whisper" || args.transcribe.is_some() {
        #[cfg(feature = "media")]
        return whisper_cli::run(&args);
        #[cfg(not(feature = "media"))]
        bail!("speech to text from the command line needs the `media` feature (audio decoding)");
    }
    match model_type.as_str() {
        "gemma4" | "gemma4_text" | "gemma3" | "gemma3_text" => run_gemma(args),
        "qwen3" | "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_6" | "qwen3_8" => run_qwen(args),
        other => bail!("unsupported model_type {other}"),
    }
}

/// The system prompt in effect: `--system`, else the default one (unless `--no-system`, a raw
/// prompt, or a logits dump, which must match reference renderings without one).
fn system_prompt(args: &Args) -> Option<String> {
    if args.system.is_some() {
        return args.system.clone();
    }
    if args.no_system || args.raw || args.dump_logits.is_some() {
        return None;
    }
    Some(oxinfer::tools::default_system_prompt(args.tools))
}

fn tool_specs(args: &Args) -> Vec<oxinfer::tools::ToolSpec> {
    if args.tools { oxinfer::tools::builtin_specs() } else { Vec::new() }
}

/// After a reply, runs the built-in tools it called, feeds the results back and lets the model
/// continue, for up to 5 rounds. Returns the tokens generated after the last round.
#[allow(clippy::too_many_arguments)]
fn continue_with_tools<M: TextLm>(
    args: &Args,
    model: &M,
    sess: &mut M::Sess,
    tok: &Tokenizer,
    eos: &[u32],
    fmt: &dyn ChatFormat,
    sampler: &mut Sampler,
    mut out: Vec<u32>,
) -> Result<Vec<u32>> {
    if !args.tools {
        return Ok(out);
    }
    for _ in 0..5 {
        let raw = tok.decode(&out, false).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        let calls = fmt.parse_tool_calls(&raw);
        if calls.is_empty() {
            break;
        }
        let results: Vec<oxinfer::tools::ToolResult> = calls
            .iter()
            .map(|c| {
                let content = oxinfer::tools::run_builtin(c);
                eprintln!("[tool {}({}) -> {content}]", c.name, serde_json::Value::Object(c.arguments.clone()));
                oxinfer::tools::ToolResult {
                    name: c.name.clone(),
                    content,
                }
            })
            .collect();
        let text = fmt.tool_results(&results, args.think);
        let ids = tok
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        let t0 = Instant::now();
        let hs = generate::prefill(model, sess, &ids, args.prefill_chunk.unwrap_or(256), &[]);
        let h = model.hidden_size();
        let logits = model.logits(&hs[hs.len() - h..], 1);
        let prefill_secs = t0.elapsed().as_secs_f64();
        let (o, n, secs) = generate(model, sess, tok, logits, sampler, eos, args.max_tokens, fmt, args.think);
        report(ids.len(), prefill_secs, n, secs);
        out = o;
    }
    Ok(out)
}

/// Text-to-image with a diffusers Stable Diffusion folder.
fn run_image(args: &Args) -> Result<()> {
    use oxinfer::sd::{Params, Pipeline, sched::Sampler};
    let t0 = Instant::now();
    let pipe = Pipeline::load_with_loras(&args.model, &args.loras, &mut |line| eprintln!("{line}"))?;
    eprintln!(
        "[loaded diffusion pipeline in {:.2}s{}]",
        t0.elapsed().as_secs_f64(),
        if pipe.turbo { ", turbo" } else { "" }
    );
    let prompt = match &args.prompt {
        Some(p) => p.clone(),
        None => {
            let mut s = String::new();
            std::io::stdin().read_line(&mut s)?;
            s.trim().to_string()
        }
    };
    let side = pipe.default_size();
    let (width, height) = args.size.unwrap_or((side, side));
    let sampler = match args.sampler.as_deref().unwrap_or("euler_a") {
        "euler" => Sampler::Euler,
        "euler_a" | "euler_ancestral" => Sampler::EulerAncestral,
        other => bail!("unknown sampler {other} (use euler or euler_a)"),
    };
    let params = Params {
        prompt,
        negative: args.negative.clone(),
        steps: args.steps.unwrap_or(if pipe.turbo { 1 } else { 25 }),
        guidance: args.guidance.unwrap_or(if pipe.turbo { 0.0 } else { 7.5 }),
        seed: args.seed,
        width,
        height,
        sampler,
    };
    eprintln!(
        "[{}x{}, {} steps, guidance {}, {:?}, seed {}]",
        width, height, params.steps, params.guidance, sampler, params.seed
    );
    let (img, tm) = pipe.generate(&params, &mut |i, n| eprint!("\r[step {i}/{n}]"))?;
    eprintln!();
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from("image.png"));
    std::fs::write(&out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb)).with_context(|| format!("write {}", out.display()))?;
    eprintln!(
        "[text {:.2}s | denoise {:.2}s ({} UNet passes, {:.2}s each) | decode {:.2}s | wrote {}]",
        tm.text,
        tm.denoise,
        tm.unet_calls,
        tm.denoise / tm.unet_calls.max(1) as f64,
        tm.decode,
        out.display()
    );
    if args.profile {
        eprint!("[time by operation]\n{}", oxinfer::sd::ops::timing::report());
        if let Some(s) = oxinfer::gpu::active().and_then(|b| b.stats()) {
            eprintln!("  gpu: {s}");
        }
    }
    if oxinfer::sd::ops::gpu_check::on() {
        eprintln!("[gpu check: {}]", oxinfer::sd::ops::gpu_check::summary());
    }
    Ok(())
}

/// Plain generation (single prompt or chat) for any model family, without attachments or drafter.
/// Attachments prepared for the first user turn: placeholder text and soft tokens.
type Mm = (String, Vec<oxinfer::mm::MmFeature>);

fn run_text<M: TextLm>(args: &Args, model: &M, tok: &Tokenizer, eos: &[u32], fmt: &dyn ChatFormat, mm: Mm) -> Result<()> {
    let (mm_text, mut mm_feats) = mm;
    let encode = |text: &str| -> Result<Vec<u32>> {
        let enc = tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    };
    let mut sampler = Sampler::new(args.temperature, args.top_k, args.top_p, args.seed);
    let mut sess = model.new_session();
    let h = model.hidden_size();
    if args.chat {
        let mut first = true;
        let stdin = std::io::stdin();
        loop {
            eprint!("> ");
            let mut line = String::new();
            if stdin.lock().read_line(&mut line)? == 0 {
                break;
            }
            let line = line.trim_end_matches(['\n', '\r']);
            if line.trim().is_empty() {
                continue;
            }
            let text = if first {
                first = false;
                fmt.first_turn(
                    system_prompt(args).as_deref(),
                    args.think,
                    &tool_specs(args),
                    &format!("{mm_text}{line}"),
                )
            } else {
                fmt.end_of_model_turn() + &fmt.user_turn(line, args.think)
            };
            let ids = encode(&text)?;
            if args.show_tokens {
                eprintln!("[prompt ids: {ids:?}]");
            }
            // The attachments belong to the first turn.
            let overrides = oxinfer::mm::bind(model, &mut sess, &ids, &std::mem::take(&mut mm_feats))?;
            let t = Instant::now();
            let hs = generate::prefill(model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
            let logits = model.logits(&hs[hs.len() - h..], 1);
            let prefill_secs = t.elapsed().as_secs_f64();
            prof::reset();
            let (out, n, decode_secs) = generate(model, &mut sess, tok, logits, &mut sampler, eos, args.max_tokens, fmt, args.think);
            if args.show_tokens {
                eprintln!("[output ids: {out:?}]");
            }
            report(ids.len(), prefill_secs, n, decode_secs);
            if args.profile {
                prof::report(decode_secs, n);
            }
            continue_with_tools(args, model, &mut sess, tok, eos, fmt, &mut sampler, out)?;
        }
        return Ok(());
    }
    let prompt = match &args.prompt {
        Some(p) => p.clone(),
        None => {
            let mut s = String::new();
            std::io::stdin().lock().read_to_string(&mut s)?;
            s
        }
    };
    let prompt = format!("{mm_text}{prompt}");
    let text = if args.raw {
        prompt
    } else {
        fmt.first_turn(system_prompt(args).as_deref(), args.think, &tool_specs(args), &prompt)
    };
    if args.print_prompt {
        print!("{text}");
        return Ok(());
    }
    let ids = encode(&text)?;
    if args.show_tokens {
        eprintln!("[prompt ids ({}): {ids:?}]", ids.len());
    }
    if ids.is_empty() {
        bail!("empty prompt");
    }
    let overrides = oxinfer::mm::bind(model, &mut sess, &ids, &mm_feats)?;
    let t = Instant::now();
    let hs = generate::prefill(model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
    let logits = model.logits(&hs[hs.len() - h..], 1);
    let prefill_secs = t.elapsed().as_secs_f64();
    if args.profile {
        eprintln!("-- prefill --");
        prof::report(prefill_secs, ids.len());
    }
    if let Some(path) = &args.dump_logits {
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        let top = idx[..10]
            .iter()
            .map(|&i| format!("{i}:{:.3}", logits[i]))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("[wrote {} logits to {} | top10 {top}]", logits.len(), path.display());
        eprintln!("[prefill {} tok in {:.2}s]", ids.len(), prefill_secs);
        return Ok(());
    }
    prof::reset();
    let (out, n, decode_secs) = generate(model, &mut sess, tok, logits, &mut sampler, eos, args.max_tokens, fmt, args.think);
    if args.show_tokens {
        eprintln!("[output ids: {out:?}]");
    }
    report(ids.len(), prefill_secs, n, decode_secs);
    if args.profile {
        eprintln!("-- decode --");
        prof::report(decode_secs, n);
    }
    continue_with_tools(args, model, &mut sess, tok, eos, fmt, &mut sampler, out)?;
    Ok(())
}

/// Embeddings for the prompt or each stdin line.
fn run_embed(args: &Args) -> Result<()> {
    let t0 = Instant::now();
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let emb = oxinfer::embed::Embedder::load(&args.model, &files, args.quant)?;
    eprintln!(
        "[loaded embedding model: dim {}, {:?} pooling{}{}, max {} tokens, prompts [{}], {:.2}s]",
        emb.dim(),
        emb.pooling,
        if emb.normalize { ", normalized" } else { "" },
        match (emb.has_sparse(), emb.has_colbert()) {
            (true, true) => ", sparse + colbert heads",
            (true, false) => ", sparse head",
            (false, true) => ", colbert head",
            _ => "",
        },
        emb.max_len,
        emb.prompts.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>().join(", "),
        t0.elapsed().as_secs_f64()
    );
    if (args.sparse && !emb.has_sparse()) || (args.colbert && !emb.has_colbert()) {
        bail!("this model has no {} head", if args.sparse { "sparse" } else { "colbert" });
    }
    let texts: Vec<String> = match &args.prompt {
        Some(p) => vec![p.clone()],
        None => std::io::stdin()
            .lock()
            .lines()
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .filter(|l| !l.trim().is_empty())
            .collect(),
    };
    let fmt = |v: &[f32]| format!("[{}]", v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(","));
    let t1 = Instant::now();
    let mut vecs = Vec::with_capacity(texts.len());
    for text in &texts {
        let mut e = emb.embed(text, args.prompt_name.as_deref())?;
        if let Some(d) = args.dim {
            e.truncate(d).context("--dim")?;
        }
        if args.similarity {
            vecs.push(e.dense);
            continue;
        }
        let line = if args.sparse {
            let m: serde_json::Map<String, serde_json::Value> = emb
                .sparse_tokens(&e)
                .unwrap()
                .into_iter()
                .map(|(k, w)| (k, serde_json::Value::from(w as f64)))
                .collect();
            serde_json::Value::Object(m).to_string()
        } else if args.colbert {
            format!("[{}]", e.colbert.unwrap().iter().map(|v| fmt(v)).collect::<Vec<_>>().join(","))
        } else {
            fmt(&e.dense)
        };
        println!("{line}");
    }
    if args.similarity {
        for a in &vecs {
            println!(
                "{}",
                vecs.iter()
                    .map(|b| format!("{:7.4}", oxinfer::embed::cosine(a, b)))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    eprintln!("[{} texts in {:.2}s]", texts.len(), t1.elapsed().as_secs_f64());
    Ok(())
}

fn run_qwen(args: Args) -> Result<()> {
    let t0 = Instant::now();
    if args.draft.is_some() {
        bail!("--draft is not supported for Qwen models (see --mtp)");
    }
    let cfg = config::QwenConfig::load(&args.model)?;
    let eos = config::load_eos_ids(&args.model);
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let mut model = qwen::QwenModel::load(cfg, &files, args.quant)?;
    let tok = load_tokenizer(&args.model)?;
    let fmt = chat::QwenChat::for_model(&args.model);
    let eos = chat::with_turn_end(fmt, &tok, eos);
    eprintln!(
        "[loaded Qwen: {} layers, hidden {}, {:.1} GB checkpoint ({}), {} weights streaming {:.2} GB/token, {} threads, {:.2}s]",
        model.cfg.num_hidden_layers,
        model.cfg.hidden_size,
        files.total_bytes() as f64 / 1e9,
        if args.mmap { "mmap" } else { "in RAM" },
        if files.metadata("format") == Some("gguf") {
            "GGUF".to_string()
        } else {
            format!("{:?}", model.quant)
        },
        model.streamed_bytes() as f64 / 1e9,
        pool::threads(),
        t0.elapsed().as_secs_f64()
    );
    // Converted checkpoints keep the MTP layer (loading it marks its tensors for writing).
    let side = match &args.mtp_from {
        Some(d) => Some(safetensors::ModelFiles::open(d, false)?),
        None => None,
    };
    let mut mtp = if args.mtp || (args.save_quantized.is_some() && oxinfer::qwen_mtp::mtp_prefix(&files).is_some()) {
        let t1 = Instant::now();
        let mut m = model.load_mtp(side.as_ref().unwrap_or(&files))?;
        if let Some(side) = &side
            && m.attach_sidecar(&model.cfg, side, model.quant)?
        {
            eprintln!("[draft head from the sidecar: {} tokens]", m.draft_head().map_or(0, |d| d.vocab()));
        }
        eprintln!(
            "[MTP layer loaded: {:.2} GB read per draft step, {:.2}s]",
            m.step_bytes(&model) as f64 / 1e9,
            t1.elapsed().as_secs_f64()
        );
        Some(m)
    } else {
        None
    };
    if let Some(out) = &args.save_quantized {
        // Keep the vision tower (loading it marks its tensors for writing).
        if let Some(qc) = oxinfer::qwen_vision::QwenMmConfig::load(&args.model)? {
            oxinfer::qwen_vision::QwenVisionTower::load(qc.vision, &files, model.quant)?;
            eprintln!("[vision tower kept]");
        }
        return save_quantized(&args.model, out, &files, model.quant);
    }
    let mm = if mm_request(&args).is_empty() {
        Default::default()
    } else {
        oxinfer::mm::prepare(&mm_request(&args), &files, model.cfg.hidden_size)?
    };
    let streamer = if args.stream {
        let mut fs = vec![&files];
        fs.extend(side.as_ref());
        Some(start_stream(&args, &mut model, mtp.as_mut().filter(|_| args.mtp), &fs)?)
    } else {
        None
    };
    *STREAM.lock().unwrap() = streamer.clone();
    TRACE.store(args.stream_trace, std::sync::atomic::Ordering::Relaxed);
    let r = match &mtp {
        Some(m) if args.mtp => run_qwen_mtp(&args, &model, m, &tok, &eos, fmt, streamer.as_deref(), mm),
        _ => run_text(&args, &model, &tok, &eos, fmt, mm),
    };
    if let Some(s) = &streamer
        && let Some(c) = s.cached_bytes()
    {
        eprintln!("[stream: {:.2} GB of the model files in the page cache at exit]", c as f64 / 1e9);
    }
    // The streamer reads the mapped files: stop it before they are unmapped.
    *STREAM.lock().unwrap() = None;
    if let Some(m) = &mut mtp {
        m.set_stream(None);
    }
    model.set_stream(None);
    drop(streamer);
    r
}

static TRACE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The active weight streamer, for the per-token trace of plain decoding.
static STREAM: std::sync::Mutex<Option<std::sync::Arc<oxinfer::stream::Streamer>>> = std::sync::Mutex::new(None);

/// Draft tokens per round by default: more when streaming, where verifying 9 tokens costs about
/// as much as verifying 1.
fn default_draft_tokens(args: &Args) -> usize {
    args.draft_tokens.unwrap_or(if args.stream { 8 } else { 3 })
}

/// Starts weight streaming for `--stream` and prints the plan.
fn start_stream(
    args: &Args,
    model: &mut qwen::QwenModel,
    mtp: Option<&mut oxinfer::qwen_mtp::QwenMtp>,
    files: &[&safetensors::ModelFiles],
) -> Result<std::sync::Arc<oxinfer::stream::Streamer>> {
    use oxinfer::stream::StreamOptions;
    let mut opts = StreamOptions::default().with_env();
    opts.cache_budget = args.cache_budget_gb.map(|g| (g * 1e9) as u64);
    opts.resident_bytes = args.resident_mb.map(|m| m << 20);
    if let Some(p) = args.prefetch_mb {
        opts.prefetch_bytes = p << 20;
    }
    opts.resident_mode = args.resident_mode;
    if let Some(f) = args.flash_mbps {
        opts.flash_bytes_per_sec = Some(f * 1e6);
    }
    if let Some(io) = args.stream_io {
        opts.io = io;
    }
    if let Some(m) = args.expert_cache_mb {
        opts.part_cache_bytes = Some(m << 20);
    }
    opts.predict_experts = args.expert_predict;
    if let Some(n) = args.stream_threads {
        opts.io_threads = n;
    }
    let s = oxinfer::stream::attach_qwen(model, mtp, files, default_draft_tokens(args), opts)?;
    let plan = s.plan();
    let o = s.options();
    let resident: Vec<&str> = plan.units.iter().filter(|u| u.2).map(|u| u.0.as_str()).collect();
    eprintln!(
        "[stream: {} units, {:.2} GB resident ({}), {:.2} GB streamed per pass, prefetch {} MB, {}{}, io {:?}, set-up {:.2}s]",
        plan.units.len(),
        plan.resident_bytes as f64 / 1e9,
        if resident.len() > 6 {
            format!("{}, ... {} units", resident[..4].join(", "), resident.len())
        } else {
            resident.join(", ")
        },
        plan.streamed_bytes as f64 / 1e9,
        o.prefetch_bytes >> 20,
        match (o.cache_budget, o.resident_bytes) {
            (_, Some(_)) | (Some(_), None) => format!("{:?} resident, streamed units evicted after use", o.resident_mode),
            _ => "no budget (the OS page cache decides)".into(),
        },
        match o.flash_bytes_per_sec {
            Some(r) => format!(", flash emulated at {:.0} MB/s", r / 1e6),
            None => String::new(),
        },
        o.io,
        plan.setup.as_secs_f64()
    );
    if plan.part_bytes > 0 {
        eprintln!(
            "[stream: {:.2} GB of routed experts read on demand, {:.2} GB of page cache keeps the recently used ones]",
            plan.part_bytes as f64 / 1e9,
            plan.part_cache_bytes as f64 / 1e9
        );
    }
    if plan.unmapped_bytes > 0 {
        eprintln!(
            "[stream: warning: {:.2} GB of weights are copies in RAM, not read from the file]",
            plan.unmapped_bytes as f64 / 1e9
        );
    }
    if plan.resident_failures > 0 {
        eprintln!("[stream: {} resident units could not be locked or copied]", plan.resident_failures);
    }
    Ok(s)
}

/// One line of streaming statistics for `steps` decoding steps (`--stream-trace` and summaries).
fn stream_line(d: &oxinfer::stream::StreamStats, tokens: usize, secs: f64) -> String {
    let parts = if d.parts > 0 {
        let pre = if d.parts_prefetched > 0 {
            format!(", {} guessed ahead, {} of them used", d.parts_prefetched, d.parts_prefetch_used)
        } else {
            String::new()
        };
        format!(", experts {} ({} cached{pre})", d.parts, d.parts_cached)
    } else {
        String::new()
    };
    format!(
        "{tokens} tok in {:.2}s: read {:.2} GB from storage, {:.2} GB cache hits, stalled {:.2}s ({:.0}%), reader busy {:.2}s (throttled {:.2}s), evicting {:.2}s{parts}",
        secs,
        d.bytes_read as f64 / 1e9,
        d.bytes_cached as f64 / 1e9,
        d.stall.as_secs_f64(),
        100.0 * d.stall.as_secs_f64() / secs.max(1e-9),
        d.io_busy.as_secs_f64(),
        d.throttled.as_secs_f64(),
        d.evict_time.as_secs_f64()
    )
}

/// Generation (single prompt or chat) with MTP speculative decoding.
#[allow(clippy::too_many_arguments)]
fn run_qwen_mtp(
    args: &Args,
    model: &qwen::QwenModel,
    mtp: &oxinfer::qwen_mtp::QwenMtp,
    tok: &Tokenizer,
    eos: &[u32],
    fmt: &dyn ChatFormat,
    stream: Option<&oxinfer::stream::Streamer>,
    mm: Mm,
) -> Result<()> {
    use oxinfer::qwen_mtp::{MtpDecoder, MtpOptions};
    let (mm_text, mut mm_feats) = mm;
    if args.tools || args.dump_logits.is_some() || args.print_prompt {
        bail!("--mtp does not combine with --tools, --dump-logits or --print-prompt");
    }
    let encode = |text: &str| -> Result<Vec<u32>> {
        let enc = tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    };
    let opts = MtpOptions {
        draft_tokens: default_draft_tokens(args),
        adaptive: !args.mtp_fixed,
        stop: eos.to_vec(),
        cost_aware: !args.mtp_fixed && args.mtp_cost.unwrap_or(args.stream),
        // Long drafts stay exact (the decode kernels are also faster than the GEMM at 8-15 tokens).
        exact: true,
    };
    eprintln!(
        "[MTP speculative decoding: {} draft tokens{}, {}]",
        opts.draft_tokens,
        if opts.cost_aware {
            " max (cost-aware)"
        } else if opts.adaptive {
            " max (adaptive)"
        } else {
            ""
        },
        if args.temperature > 0.0 { "sampled" } else { "greedy" }
    );
    let mut sampler = Sampler::new(args.temperature, args.top_k, args.top_p, args.seed);
    let mut dec = MtpDecoder::new(model, mtp, opts);
    if dec.opts.draft_tokens > oxinfer::qwen_mtp::max_exact_draft() {
        eprintln!(
            "[note: verifying more than {} drafts runs the prefill GEMM kernels, whose rounding differs from decoding: \
             the output can drift from plain decoding at near-ties]",
            oxinfer::qwen_mtp::max_exact_draft()
        );
    }
    let chunk = args.prefill_chunk.unwrap_or(256);
    let mut turn = |dec: &mut MtpDecoder, text: &str| -> Result<()> {
        let ids = encode(text)?;
        if ids.is_empty() {
            bail!("empty prompt");
        }
        if args.show_tokens {
            eprintln!("[prompt ids ({}): {ids:?}]", ids.len());
        }
        // The attachments belong to the first turn.
        let overrides = oxinfer::mm::bind(model, &mut dec.sess, &ids, &std::mem::take(&mut mm_feats))?;
        let t = Instant::now();
        let logits = dec.prefill_mm(&ids, chunk, &overrides);
        let prefill_secs = t.elapsed().as_secs_f64();
        prof::reset();
        let before = dec.stats.clone();
        let s0 = stream.map(|s| s.stats());
        let mut printer = Printer::new(fmt, args.think, tok);
        let (mut s_last, mut t_last, mut n_last) = (s0, Instant::now(), 0usize);
        let d = generate::decode_mtp(dec, &logits, &mut sampler, eos, args.max_tokens, &mut |out| {
            // A round commits several tokens at once: report when its reads show up.
            if let (true, Some(s), Some(sl)) = (args.stream_trace, stream, &s_last) {
                let now = s.stats();
                if now.bytes_read != sl.bytes_read || now.stall != sl.stall {
                    let n = out.len() - n_last;
                    eprintln!("[step: +{n}: {}]", stream_line(&(now - *sl), n, t_last.elapsed().as_secs_f64()));
                    (s_last, t_last, n_last) = (Some(now), Instant::now(), out.len());
                }
            }
            printer.print(tok, out)
        });
        let out = d.tokens;
        let decode_secs = d.secs;
        printer.finish(tok, &out, eos);
        if args.show_tokens {
            eprintln!("[output ids: {out:?}]");
        }
        let st = &dec.stats;
        let (rounds, drafted, accepted) = (
            st.rounds - before.rounds,
            st.drafted - before.drafted,
            st.accepted - before.accepted,
        );
        report(ids.len(), prefill_secs, out.len().saturating_sub(1), decode_secs);
        eprintln!(
            "[MTP: {} main forwards for {} tokens = {:.2} tok/forward; drafts accepted {}/{} = {:.0}%; per-round histogram {:?}]",
            rounds,
            out.len().saturating_sub(1),
            out.len().saturating_sub(1) as f64 / rounds.max(1) as f64,
            accepted,
            drafted,
            100.0 * accepted as f64 / drafted.max(1) as f64,
            st.per_round
        );
        eprintln!(
            "[MTP time: drafting {:.2}s, verification {:.2}s ({:.0} ms per forward)]",
            st.draft_secs - before.draft_secs,
            st.verify_secs - before.verify_secs,
            1e3 * (st.verify_secs - before.verify_secs) / rounds.max(1) as f64
        );
        if let (Some(s), Some(s0)) = (stream, s0) {
            let d = s.stats() - s0;
            eprintln!(
                "[stream: {}; {:.2} GB read per forward]",
                stream_line(&d, out.len().saturating_sub(1), decode_secs),
                d.bytes_read as f64 / 1e9 / rounds.max(1) as f64
            );
        }
        if args.profile {
            prof::report(decode_secs, out.len());
        }
        Ok(())
    };
    if !args.chat {
        let prompt = match &args.prompt {
            Some(p) => p.clone(),
            None => {
                let mut s = String::new();
                std::io::stdin().lock().read_to_string(&mut s)?;
                s
            }
        };
        let prompt = format!("{mm_text}{prompt}");
        let text = if args.raw {
            prompt
        } else {
            fmt.first_turn(system_prompt(args).as_deref(), args.think, &[], &prompt)
        };
        return turn(&mut dec, &text);
    }
    let mut first = true;
    let stdin = std::io::stdin();
    loop {
        eprint!("> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim_end_matches(['\n', '\r']);
        if line.trim().is_empty() {
            continue;
        }
        let text = if first {
            first = false;
            fmt.first_turn(system_prompt(args).as_deref(), args.think, &[], &format!("{mm_text}{line}"))
        } else {
            fmt.end_of_model_turn() + &fmt.user_turn(line, args.think)
        };
        turn(&mut dec, &text)?;
    }
    Ok(())
}

fn run_gemma(args: Args) -> Result<()> {
    let t0 = Instant::now();
    let cfg = config::TextConfig::load(&args.model)?;
    let eos = config::load_eos_ids(&args.model);
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let model = Model::load(cfg, &files, args.quant)?;
    let tok = Tokenizer::from_file(args.model.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    if model.cfg.gemma3 {
        if args.draft.is_some() || !mm_request(&args).is_empty() {
            bail!("speculative decoding and attachments are not supported for Gemma 3 yet");
        }
        if let Some(out) = &args.save_quantized {
            return save_quantized(&args.model, out, &files, model.quant);
        }
        let eos = chat::with_turn_end(&chat::Gemma3Chat, &tok, eos);
        return run_text(&args, &model, &tok, &eos, &chat::Gemma3Chat, Default::default());
    }
    let eos = chat::with_turn_end(&chat::GemmaChat, &tok, eos);
    eprintln!(
        "[loaded {} layers, hidden {}, {:.1} GB checkpoint ({}), {:?} weights streaming {:.2} GB/token, {} threads, {:.2}s]",
        model.cfg.num_hidden_layers,
        model.cfg.hidden_size,
        files.total_bytes() as f64 / 1e9,
        if args.mmap { "mmap" } else { "in RAM" },
        model.quant,
        model.streamed_bytes() as f64 / 1e9,
        pool::threads(),
        t0.elapsed().as_secs_f64()
    );
    if let Some(out) = &args.save_quantized {
        return save_quantized(&args.model, out, &files, model.quant);
    }
    if args.preload && args.mmap {
        let t = Instant::now();
        files.preload();
        eprintln!("[preloaded weights in {:.2}s]", t.elapsed().as_secs_f64());
    }

    let encode = |text: &str| -> Result<Vec<u32>> {
        let enc = tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    };

    let draft_files = match &args.draft {
        Some(d) => Some(safetensors::ModelFiles::open(d, true)?),
        None => None,
    };
    let drafter = match (&args.draft, &draft_files) {
        (Some(d), Some(f)) => {
            let dr = drafter::Drafter::load(d, f)?;
            if dr.backbone_hidden != model.cfg.hidden_size {
                bail!(
                    "drafter backbone size {} does not match model hidden size {}",
                    dr.backbone_hidden,
                    model.cfg.hidden_size
                );
            }
            if args.temperature > 0.0 {
                eprintln!("[note: speculative decoding is greedy; --temp ignored]");
            }
            eprintln!(
                "[drafter loaded: {} layers, hidden {}, {} draft tokens/round]",
                dr.cfg.num_hidden_layers,
                dr.cfg.hidden_size,
                args.draft_tokens.unwrap_or(6)
            );
            Some(dr)
        }
        _ => None,
    };

    let mut sampler = Sampler::new(args.temperature, args.top_k, args.top_p, args.seed);
    let mut sess = model.new_session();

    if args.chat {
        let mut first = true;
        let (mm_text, mm_feats) = if !mm_request(&args).is_empty() {
            oxinfer::mm::prepare(&mm_request(&args), &files, model.cfg.hidden_size)?
        } else {
            (String::new(), Vec::new())
        };
        let stdin = std::io::stdin();
        loop {
            eprint!("> ");
            let mut line = String::new();
            if stdin.lock().read_line(&mut line)? == 0 {
                break;
            }
            let line = line.trim_end_matches(['\n', '\r']);
            if line.trim().is_empty() {
                continue;
            }
            let mut text = String::new();
            let mut overrides = Vec::new();
            if first {
                text.push_str(&chat::GemmaChat.conversation_start_tools(system_prompt(&args).as_deref(), args.think, &tool_specs(&args)));
                text.push_str(&chat::user_turn(&format!("{mm_text}{line}")));
                first = false;
            } else {
                // The previous stop token was sampled but never fed to the model.
                text.push_str(&chat::end_of_model_turn());
                text.push_str(&chat::user_turn(line));
            }
            let ids = encode(&text)?;
            if !mm_feats.is_empty() && ["<|image|>", "<|audio|>", "<|video|>"].iter().any(|p| text.contains(p)) {
                overrides = oxinfer::mm::overrides(&ids, &mm_feats, model.cfg.hidden_size)?;
            }
            if args.show_tokens {
                eprintln!("[prompt ids: {ids:?}]");
            }
            let t = Instant::now();
            if let Some(dr) = &drafter {
                let hs = generate::prefill(&model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
                let h = model.cfg.hidden_size;
                let hidden_last = hs[hs.len() - h..].to_vec();
                let prefill_secs = t.elapsed().as_secs_f64();
                let (out, st, secs) = generate_speculative(
                    &model,
                    dr,
                    &mut sess,
                    &tok,
                    hidden_last,
                    args.draft_tokens.unwrap_or(6),
                    &eos,
                    args.max_tokens,
                );
                if args.show_tokens {
                    eprintln!("[output ids: {out:?}]");
                }
                report(ids.len(), prefill_secs, out.len(), secs);
                eprintln!(
                    "[speculative: {:.2} tok/forward, drafts accepted {}%]",
                    out.len() as f64 / st.forwards.max(1) as f64,
                    100 * st.accepted / st.drafted.max(1)
                );
                continue;
            }
            let hs = generate::prefill(&model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
            let logits = model.logits(&hs[hs.len() - model.cfg.hidden_size..], 1);
            let prefill_secs = t.elapsed().as_secs_f64();
            prof::reset();
            let (out, n, decode_secs) = generate(
                &model,
                &mut sess,
                &tok,
                logits,
                &mut sampler,
                &eos,
                args.max_tokens,
                &chat::GemmaChat,
                args.think,
            );
            if args.show_tokens {
                eprintln!("[output ids: {out:?}]");
            }
            report(ids.len(), prefill_secs, n, decode_secs);
            if args.profile {
                prof::report(decode_secs, n);
            }
            continue_with_tools(&args, &model, &mut sess, &tok, &eos, &chat::GemmaChat, &mut sampler, out)?;
        }
        return Ok(());
    }

    let prompt = match &args.prompt {
        Some(p) => p.clone(),
        None => {
            let mut s = String::new();
            std::io::stdin().lock().read_to_string(&mut s)?;
            s
        }
    };
    let (mm_text, mm_feats) = if !mm_request(&args).is_empty() {
        oxinfer::mm::prepare(&mm_request(&args), &files, model.cfg.hidden_size)?
    } else {
        (String::new(), Vec::new())
    };
    let prompt = format!("{mm_text}{prompt}");
    let text = if args.raw {
        format!("<bos>{prompt}")
    } else {
        let mut t = chat::GemmaChat.conversation_start_tools(system_prompt(&args).as_deref(), args.think, &tool_specs(&args));
        t.push_str(&chat::user_turn(&prompt));
        t
    };
    if args.print_prompt {
        print!("{text}");
        return Ok(());
    }
    let ids = encode(&text)?;
    let overrides = oxinfer::mm::overrides(&ids, &mm_feats, model.cfg.hidden_size)?;
    if args.show_tokens {
        eprintln!("[prompt ids ({}): {ids:?}]", ids.len());
    }
    if ids.is_empty() {
        bail!("empty prompt");
    }

    if let Some(dr) = &drafter {
        let t = Instant::now();
        let hs = generate::prefill(&model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
        let h = model.cfg.hidden_size;
        let hidden_last = hs[hs.len() - h..].to_vec();
        let prefill_secs = t.elapsed().as_secs_f64();
        prof::reset();
        let (out, st, secs) = generate_speculative(
            &model,
            dr,
            &mut sess,
            &tok,
            hidden_last,
            args.draft_tokens.unwrap_or(6),
            &eos,
            args.max_tokens,
        );
        if args.show_tokens {
            eprintln!("[output ids: {out:?}]");
        }
        let n = out.len();
        report(ids.len(), prefill_secs, n, secs);
        eprintln!(
            "[speculative: {} main forwards for {} tokens = {:.2} tok/forward; drafts accepted {}/{} = {:.0}%]",
            st.forwards,
            n,
            n as f64 / st.forwards.max(1) as f64,
            st.accepted,
            st.drafted,
            100.0 * st.accepted as f64 / st.drafted.max(1) as f64
        );
        eprintln!(
            "[accepted-per-round histogram: {:?}]",
            &st.per_pos[..=args.draft_tokens.unwrap_or(6).min(15)]
        );
        if args.profile {
            eprintln!("-- decode --");
            prof::report(secs, st.forwards);
        }
        return Ok(());
    }

    let t = Instant::now();
    let hs = generate::prefill(&model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
    let logits = model.logits(&hs[hs.len() - model.cfg.hidden_size..], 1);
    let prefill_secs = t.elapsed().as_secs_f64();
    if args.profile {
        eprintln!("-- prefill --");
        prof::report(prefill_secs, ids.len());
    }

    if let Some(path) = &args.dump_logits {
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
        let top = {
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            idx.truncate(10);
            idx.iter().map(|&i| format!("{i}:{:.3}", logits[i])).collect::<Vec<_>>().join(" ")
        };
        eprintln!("[wrote {} logits to {} | top10 {top}]", logits.len(), path.display());
        eprintln!("[prefill {} tok in {:.2}s]", ids.len(), prefill_secs);
        return Ok(());
    }

    prof::reset();
    let (out, n, decode_secs) = generate(
        &model,
        &mut sess,
        &tok,
        logits,
        &mut sampler,
        &eos,
        args.max_tokens,
        &chat::GemmaChat,
        args.think,
    );
    if args.show_tokens {
        eprintln!("[output ids: {out:?}]");
    }
    report(ids.len(), prefill_secs, n, decode_secs);
    if args.profile {
        eprintln!("-- decode --");
        prof::report(decode_secs, n);
    }
    continue_with_tools(&args, &model, &mut sess, &tok, &eos, &chat::GemmaChat, &mut sampler, out)?;
    Ok(())
}
