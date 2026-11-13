use anyhow::{Context, Result, bail};
use oxinfer::chat::{self, ChatFormat};
use oxinfer::engine::{load_tokenizer, resolve_auto_draft, resolve_file, resolve_model, save_quantized};
use oxinfer::generate::{self, Flow, SpecStats, TextStream};
use oxinfer::model::{Model, Quant, Session};
use oxinfer::sampler::Sampler;
use oxinfer::tokenizer::Tokenizer;
use oxinfer::{config, drafter, platform, pool, prof, qwen, safetensors};
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
mod anemll_cli;
mod bagel_cli;
mod flux_cli;
mod intervene_cli;
mod janus_cli;
mod minimax_h3_cli;
mod omnigen2_cli;
mod qwen_image_cli;
mod sd3_cli;
#[cfg(feature = "server")]
mod server_cli;
mod step1x_cli;
#[cfg(feature = "media")]
mod tts_cli;
#[cfg(feature = "media")]
mod whisper_cli;
#[cfg(feature = "media")]
mod zonos_cli;

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
    no_draft: bool,
    draft_tokens: Option<usize>,
    draft_p_min: Option<f32>,
    draft_device: Option<bool>,
    draft_conf_min: Option<f32>,
    fast_verify: bool,
    /// Prompt lookup: on/off (None: the default, `oxinfer::lookup::BY_DEFAULT`).
    lookup: Option<bool>,
    lookup_tokens: Option<usize>,
    lookup_min_match: Option<usize>,
    lookup_memory: bool,
    lookup_fixed: bool,
    /// Session persistence (runs the conversation through `oxinfer::engine::Engine`).
    prompt_cache: Option<String>,
    prompt_cache_gb: Option<f64>,
    save_session: Option<PathBuf>,
    load_session: Option<PathBuf>,
    mtp: bool,
    mtp_fixed: bool,
    mtp_from: Option<PathBuf>,
    mtp_cost: Option<bool>,
    stream: bool,
    /// `--no-stream`: never stream (an oversized Mistral Large 4 is then demand-paged).
    no_stream: bool,
    mmproj: oxinfer::pixtral::Mmproj,
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
    /// Image editing: the input images (repeatable: Qwen-Image-Edit-2509 / 2511 take several;
    /// the others the first), a mask (white = repaint), strength, image guidance.
    init_images: Vec<PathBuf>,
    mask: Option<PathBuf>,
    strength: Option<f32>,
    image_guidance: Option<f32>,
    /// Qwen-Image: the transformer from a GGUF file or another folder.
    transformer: Option<PathBuf>,
    /// Qwen-Image: ControlNet and layer-decomposition flags.
    qwen_image: qwen_image_cli::Flags,
    device: platform::DevicePref,
    devices: bool,
    /// `--models`: what runs on this machine (`--all`: every variant; `--kind K`).
    models: bool,
    models_all: bool,
    models_kind: Option<oxinfer::advisor::Kind>,
    models_can: Vec<oxinfer::advisor::Cap>,
    /// `--check-update` / `--update` (`--force`: reinstall the latest release).
    check_update: bool,
    update: bool,
    force: bool,
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
    /// Further `--transcribe` files, transcribed with the first as a batch.
    transcribe_more: Vec<PathBuf>,
    beam: Option<usize>,
    patience: Option<f32>,
    length_penalty: Option<f32>,
    best_of: Option<usize>,
    batch: Option<usize>,
    // Janus-Pro text-to-image.
    generate_image: Option<String>,
    num_images: usize,
    temp_given: bool,
    // Text to speech (Orpheus models).
    speak: Option<String>,
    voice: Option<String>,
    snac: Option<String>,
    speak_stream: bool,
    speak_json: Option<String>,
    speak_more: Vec<String>,
    speak_file: Option<PathBuf>,
    speak_split: bool,
    no_noise: bool,
    unconstrained: bool,
    decode_tokens: Option<PathBuf>,
    dump_pcm: Option<PathBuf>,
    ref_audio: Vec<PathBuf>,
    ref_text: Vec<String>,
    snac_encode: Option<PathBuf>,
    decode_codes: Option<PathBuf>,
    repetition_penalty: Option<f32>,
    // Set explicitly (text to speech has its own defaults).
    temperature_set: Option<f32>,
    top_p_set: Option<f32>,
    max_tokens_set: Option<usize>,
    // Runtime interventions.
    iv: intervene_cli::IvArgs,
    // FLUX.1 / Kontext.
    flux: flux_cli::FluxArgs,
    // Stable Diffusion 3 / 3.5.
    sd3: sd3_cli::Sd3Args,
    // BAGEL.
    bagel: bagel_cli::BagelArgs,
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
         \x20 --draft DIR          speculative decoding with a DSpark draft (Qwen, Gemma 4, Mistral Large 4,\n\
         \x20                      DeepSeek V4: a Hugging Face draft folder / repo id or a dflash GGUF; greedy\n\
         \x20                      and sampled, same output distribution; `auto`: the one beside the model or\n\
         \x20                      in the download cache, else the model's known draft fetched; see the\n\
         \x20                      README) or a Gemma 4 assistant (drafter) model (greedy only)\n\
         \x20 --no-draft           plain decoding: no DSpark draft (one beside the model or in the download\n\
         \x20                      cache is used by default where it measured faster) nor default MTP\n\
         \x20 --draft-tokens N     max draft tokens per round (default 6 with a drafter, adaptive from 3;\n\
         \x20                      3 with --mtp, 8 with --mtp --stream; DSpark: Mistral Large 4 / DeepSeek V4\n\
         \x20                      3, Qwen / Gemma 4 the whole block)\n\
         \x20 --spec-draft-n-max N  the same as --draft-tokens\n\
         \x20 --spec-draft-p-min P  DSpark: cut a block before the first draft the draft model gives\n\
         \x20                      less than probability P (default 0.85 when the model streams, else 0)\n\
         \x20 --draft-device D     DSpark: auto (default: the GPU when there is one) | gpu | cpu, where the\n\
         \x20                      draft's matmuls run\n\
         \x20 --spec-draft-fixed   DSpark (Qwen, Gemma 4): always draft and verify --draft-tokens (default:\n\
         \x20                      a draft or a plain step, whichever measures faster, and the most\n\
         \x20                      profitable part of each block verified; fixed when streaming)\n\
         \x20 --spec-draft-conf-min C  DSpark: cut a block before the first draft whose confidence-head\n\
         \x20                      estimate of being accepted is below C (default 0: off)\n\
         \x20 --exact-verify       DSpark / MTP: verify drafts with the one-token kernels, so greedy\n\
         \x20                      output is the plain greedy output token for token (default: fast\n\
         \x20                      batched matrix kernels on the GPU, whose logits differ from plain\n\
         \x20                      decoding in rounding; --fast-verify asks for them explicitly)\n\
         \x20 --lookup / --no-lookup  prompt-lookup drafting (default on): drafts copied from the\n\
         \x20                      conversation where its last tokens occurred before (quoting, editing,\n\
         \x20                      summarizing), verified like other drafts (exact); rounds without a\n\
         \x20                      match are plain steps. With DSpark / MTP, a long match replaces the\n\
         \x20                      draft model's round when it promises more\n\
         \x20 --lookup-tokens N    prompt lookup: most draft tokens per round (default 10, at most 14)\n\
         \x20 --lookup-min-match N prompt lookup: shortest context match giving a draft (default 2)\n\
         \x20 --no-lookup-memory   prompt lookup: no veto of drafts the model refuted in the same context\n\
         \x20 --lookup-fixed       prompt lookup: verify every match in full (default: cost-aware length)\n\
         \x20 --prompt-cache DIR   keep conversation snapshots (KV caches, recurrent states, tokens) in DIR\n\
         \x20                      (`auto`: ~/.cache/oxinfer/prompt-cache, created private) and restore\n\
         \x20                      the longest one the prompt starts with instead of recomputing it\n\
         \x20 --prompt-cache-gb F  size cap of the prompt cache (default 20; least recently used go first)\n\
         \x20 --load-session FILE  start from a session saved with --save-session (same model and options)\n\
         \x20 --save-session FILE  save the session (KV caches, recurrent states, tokens) after the reply\n\
         \x20 --mtp                Qwen3.5/3.8: speculative decoding with the checkpoint's multi-token\n\
         \x20                      prediction layer (greedy and sampled; same output distribution); the\n\
         \x20                      default for Qwen3.5-4B / 9B (--no-draft: plain decoding)\n\
         \x20 --mtp-fixed         with --mtp, always draft --draft-tokens (default: adaptive 1..N)\n\
         \x20 --mtp-from DIR       implies --mtp, with the MTP weights from another checkpoint (e.g. the\n\
         \x20                      original one when the model is a converted copy without them)\n\
         \x20 --mtp-cost           with --mtp, pick each round's draft length from measured round times and\n\
         \x20                      acceptance (default with --stream; --no-mtp-cost turns it off)\n\
         \x20 --stream             stream weights from the mapped file (models larger than RAM; Qwen and\n\
         \x20                      Gemma): a reader thread fetches the next layers while one computes\n\
         \x20 --cache-budget-gb F  with --stream, page cache the model may use: a resident set of\n\
         \x20                      F - prefetch stays cached, the other layers are dropped after use\n\
         \x20 --resident-mb N      with --stream, resident set size (instead of the budget's)\n\
         \x20 --prefetch-mb N      with --stream, read-ahead window (default 768)\n\
         \x20 --resident-mode M    cache (default) | lock (mlock) | copy (anonymous memory)\n\
         \x20 --flash-mbps N       with --stream, emulate storage reading N MB/s (or $OXINFER_FLASH_MBPS)\n\
         \x20 --stream-io M        advise (default on Apple) | pread | touch\n\
         \x20 --stream-trace       with --stream, one line of IO statistics per decoding step and per prefill\n\
         \x20 --stream-threads N   with --stream, reader threads (default 2)\n\
         \x20 --stream-gpu on|off|auto  with --stream, decode on the GPU: resident layers stay wrapped,\n\
         \x20                      the others are wrapped per pass as they arrive (auto, the default: dense\n\
         \x20                      models on the GPU, MoE models measure both and keep the faster)\n\
         \x20 --expert-cache-mb N  with --stream, page cache kept for recently used MoE experts (which\n\
         \x20                      are read on demand; default: what the budget leaves)\n\
         \x20 --expert-predict     with --stream, read the next MoE layer's guessed experts ahead (Qwen)\n\
         \x20 --no-stream          never stream, not even a Mistral Large 4 checkpoint larger than 60% of\n\
         \x20                      RAM (streamed by default): it is then memory-mapped and demand-paged\n\
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
         \x20 --mmproj FILE|none   Mistral Large 4, Qwen3.5-family (Qwen3.8-Flash-Next): the vision tower\n\
         \x20                      (llama.cpp mmproj GGUF, path or org/repo/file; default: the checkpoint's,\n\
         \x20                      or the mmproj*.gguf beside a GGUF model), or none\n\
         \x20 --cache-dir DIR      where fetched models are cached\n\
         \x20 --download-only      fetch the model (and drafter) and exit\n\
         \x20 --save-quantized DIR write a checkpoint with the weights already quantized to --quant (loads\n\
         \x20                      zero-copy with no quantization pass; for browsers, mmap); Qwen keeps\n\
         \x20                      its vision tower (bf16), Gemma's towers are left out\n\
         \n\
         image generation (a diffusers folder: Stable Diffusion 1.x / 2.x, SD-Turbo, SDXL):\n\
         \x20 -o, --output FILE     PNG to write (default image.png)\n\
         \x20 --steps N            denoising steps (default 1 for turbo models, else 25)\n\
         \x20 --guidance F         classifier-free guidance (default 0 for turbo, 5 for SDXL, else 7.5)\n\
         \x20 --negative TEXT      negative prompt (with guidance > 1)\n\
         \x20 --size WxH           image size, multiples of 64 (32 for SDXL; default the model's)\n\
         \x20 --sampler NAME       euler or euler_a (default euler_a; SDXL: its scheduler's), or\n\
         \x20                      euler_trailing / euler_a_trailing (few-step LoRAs such as SDXL-Lightning)\n\
         \x20 --seed N             noise seed (default 42)\n\
         \x20 --lora FILE[:SCALE]  merge a LoRA (kohya or diffusers safetensors; repeatable; scale 1);\n\
         \x20                      FILE may be org/repo/file.safetensors on Hugging Face\n\
         \x20 --init-image FILE    edit this image (PNG/JPEG/WebP/...; --image works too): with an\n\
         \x20                      InstructPix2Pix model (timbrooks/instruct-pix2pix) the prompt is an\n\
         \x20                      instruction (\"make it snowy\"); otherwise image-to-image\n\
         \x20 --strength F         image-to-image / inpainting: how much is redrawn, 0..1 (default 0.8;\n\
         \x20                      inpainting 1)\n\
         \x20 --mask FILE          inpaint: white = repaint, black = keep (an inpainting model such as\n\
         \x20                      stable-diffusion-v1-5/stable-diffusion-inpainting, or any model)\n\
         \x20 --image-guidance F   InstructPix2Pix: how closely to follow the input image (default 1.5)\n\
         \n\
         Qwen-Image (Qwen/Qwen-Image-Edit-2511, -Edit-2509, -Edit, Qwen/Qwen-Image): image editing and\n\
         text to image (-o, --steps, --negative, --size, --seed, --lora as above):\n\
         \x20 --init-image FILE    image to edit (repeatable: Edit-2509 / 2511 combine several)\n\
         \x20 --guidance F         true CFG scale (default 4 with negative \" \"; 1 with a Lightning LoRA)\n\
         \x20 --steps N            default 40 (a Lightning LoRA: the steps in its name, 4 or 8)\n\
         \x20 --transformer FILE   the transformer from a GGUF file (e.g. unsloth/Qwen-Image-Edit-2511-GGUF)\n\
         \x20                      or a single safetensors file (FP8 scaled checkpoints)\n\
         \x20 --strength F         with --init-image: start from the image, 0..1 of the schedule (Qwen-Image:\n\
         \x20                      image to image, default 0.6; edit models: default 1 with --mask)\n\
         \x20 --mask FILE          inpaint the white region of the (first) --init-image\n\
         \x20 --controlnet DIR     a ControlNet (InstantX/Qwen-Image-ControlNet-Union, -Inpainting) with\n\
         \x20 --control-image FILE its condition image (repeatable; with --mask: the image to inpaint),\n\
         \x20 --control-scale F    conditioning scale (default 1; repeatable, one per image),\n\
         \x20 --control-start F, --control-end F   the part of the steps it acts on (default 0 and 1)\n\
         \x20 --layers N           Qwen-Image-Layered: RGBA layers to decompose --init-image into (default 4;\n\
         \x20                      -p is a caption of the image, written by the text encoder without -p;\n\
         \x20                      -o out.png writes out_0.png ...)\n\
         \x20 --resolution N       Qwen-Image-Layered: 640 (default) or 1024\n\
         \x20 --quant q8|q4        transformer and text encoder precision; --mmap, --stream as above\n\
         \n\
         OmniGen2 (OmniGen2/OmniGen2): image editing, in-context generation and text to image (-o,\n\
         --steps (default 50), --negative, --seed, --quant, --mmap as above):\n\
         \x20 --init-image FILE    image to edit (repeatable, up to 5: \"the cat from image 1 on the sofa\n\
         \x20                      of image 2\"); none: text to image\n\
         \x20 --guidance F         text guidance (default 5); --image-guidance F image guidance (default 2)\n\
         \x20 --size WxH           output size (default: the input image's, within 1024x1024; a single\n\
         \x20                      input image is resized to it); 1024x1024 without or with several images\n\
         \x20 --image FILE         without -o: ask the model's Qwen2.5-VL about the image (--prompt)\n\
         \x20 --scheduler S        euler (default) | dpmsolver++\n\
         \x20 --teacache F         TeaCache step cache with relative-L1 threshold F (e.g. 0.05); or\n\
         \x20 --taylorseer         TaylorSeer step cache (both skip transformer passes: a different image)\n\
         \x20 --sample-posterior   sample the VAE posterior of input images (default: its mean)\n\
         \n\
         Step1X-Edit (stepfun-ai/Step1X-Edit-v1p2, -v1p1-diffusers): image editing (-o, --negative,\n\
         --seed, --quant, --mmap, --stream as above):\n\
         \x20 --init-image FILE    image to edit (without: text to image)\n\
         \x20 --guidance F         true CFG scale (default 6, against the negative prompt, default \"\")\n\
         \x20 --steps N            default 28 (StepFun's examples use 50)\n\
         \x20 --size WxH           output size (default the input's), also the working area; the input is\n\
         \x20                      center-cropped to that aspect ratio (default: about 1024x1024)\n\
         \x20 --think              v1.2: the VL model rewrites the instruction from the image first\n\
         \x20 --sample-posterior   sample the VAE posterior of the input image (default: its mean)\n\
         \n\
         Janus-Pro (deepseek-ai/Janus-Pro-1B / -7B) and JanusFlow (deepseek-ai/JanusFlow-1.3B): text\n\
         to image, and questions about images:\n\
         \x20 --generate-image T   generate a 384x384 image of T (or --prompt T with -o FILE)\n\
         \x20 -o, --output FILE     PNG to write (default image.png; FILE-N.png for several images)\n\
         \x20 --cfg F              classifier-free guidance weight (default 5, JanusFlow 2; alias\n\
         \x20                      --guidance)\n\
         \x20 --temp F             Janus-Pro: image-token temperature (default 1; 0 = greedy)\n\
         \x20 --steps N            JanusFlow: Euler steps of the rectified flow (default 30)\n\
         \x20 --size WxH           resample the 384x384 image (the models' only size) to WxH\n\
         \x20 --num-images N       images generated together in one batch (default 1)\n\
         \x20 --seed N             sampling seed (default 42)\n\
         \x20 --image FILE         with --prompt: answer a question about the image (repeatable)\n\
         \x20 --chat               a conversation: more questions from standard input, one per line\n\
         \x20 --devices            list detected CPU features and GPUs, then exit\n\
         \x20 --check-update       compare this binary with the latest GitHub release, then exit\n\
         \x20 --update             download the latest release's binary for this platform and replace this\n\
         \x20                      executable with it (--force: also when already up to date)\n\
         \x20 --models             list the tested models with what this machine can run: download size against\n\
         \x20                      the free disk space, memory against RAM and GPU memory, where each decodes\n\
         \x20                      and an estimated tokens/s (CPU kernels are timed for about a second);\n\
         \x20                      --all lists every precision; --kind chat|embedding|image|speech|tts,\n\
         \x20                      --can CAP[,CAP] (image-input, tools, image-edit, speech-to-text, ...: the\n\
         \x20                      CAN column) and --model TEXT filter; --device cpu ignores the GPU; --json\n\
         \x20 --device D           auto (default) | cpu | cuda | metal | vulkan\n\
         \x20 --gpu-decode M       on (default) | off: generate a whole token per GPU submission\n\
         \x20                      (Metal; Llama 3, Orpheus, Qwen3, Qwen3.5 / 3.8 dense)\n\
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
         \x20 --dim N              N dimensions: truncate (Matryoshka) and renormalize, or stella's head of that size\n\
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
         \x20 --image FILE         an image placed before the state text (repeatable; Qwen3.5-family\n\
         \x20                      bases with their vision tower; --max-soft-tokens caps its tokens)\n\
         \x20 --decide-json FILE   /v1/decide requests (an object, an array or one object per line;\n\
         \x20                      - for stdin; images inline in the state); one JSON response per line\n\
         \x20 --decision-from S    the decision assets (adapter, head, configs) of bundle S, a directory\n\
         \x20                      or repo id (org/name/subdir for a subfolder, e.g. autotrust/JEV-9B/vl),\n\
         \x20                      on top of --model (fetches the small files only)\n\
         \n\
         speech to text (Whisper models: openai/whisper-*, distil-whisper/*, detected from config.json;\n\
         NVIDIA Nemotron 3.5 Transcribe with --transcribe: one segment per file, --initial-prompt is\n\
         the instruction given to the model, no timestamps, beam search or fallback):\n\
         \x20 --transcribe FILE    audio to transcribe (WAV/MP3/FLAC/AAC/OGG via oxideav; any length);\n\
         \x20                      repeat it for several files, decoded together (--batch N at a time,\n\
         \x20                      default 8) with the results each file gets alone\n\
         \x20 --language XX        spoken language (code or name; default: detected from the first 30 s)\n\
         \x20 --translate          translate to English instead of transcribing\n\
         \x20 --timestamps         print timed segments (always on past 30 s of audio)\n\
         \x20 --word-timestamps    also word-level times (cross-attention alignment)\n\
         \x20 --format F           text (default), json, srt or vtt; --json for json\n\
         \x20 --initial-prompt T   context text for the decoder (names, spelling, style)\n\
         \x20 --prefix T           text the transcript starts with\n\
         \x20 --fallback           temperature fallback on repetitive or unlikely windows (openai defaults)\n\
         \x20 --condition-on-prev  feed each window the previous text\n\
         \x20 --beam N             beam search with N beams (openai-whisper's beam_size) at temperature 0;\n\
         \x20                      --patience P ends it after round(N * P) finished hypotheses (default 1),\n\
         \x20                      --length-penalty A ranks them by logprob / ((5 + length) / 6)^A\n\
         \x20 --best-of N          at temperatures above 0 (--temp, --fallback) keep the best of N samples\n\
         \x20 --precision P        weights: f32 (matches transformers token for token), f16, bf16 (default);\n\
         \x20                      --quant q8|q4 quantizes the decoder (--temp T samples at T)\n\
         \n\
         text to speech (Orpheus TTS: unsloth/orpheus-3b-0.1-ft or canopylabs/orpheus-3b-0.1-ft):\n\
         \x20 --speak TEXT         text to speak (- for stdin); -o FILE writes the WAV (default speech.wav)\n\
         \x20 --voice NAME         tara (default), leah, jess, leo, dan, mia, zac, zoe; \"\" for none\n\
         \x20 --speak-stream       decode audio while generating (sliding windows), report latency\n\
         \x20 --speak-file FILE    a batch: one utterance per line (- for stdin), generated together; so are\n\
         \x20                      several --speak. -o out.wav writes out-1.wav, out-2.wav, ...;\n\
         \x20                      --voice a,b,c gives utterance i the i-th voice (cycling)\n\
         \x20 --speak-split        split the text into sentences, generate them as a batch, join them in -o\n\
         \x20 --snac DIR           SNAC decoder (default hubertsiuzdak/snac_24khz); --precision f32|bf16\n\
         \x20 --repetition-penalty F  default 1.1 (also applies to text generation, default 1 there)\n\
         \x20 --temp, --top-p, --max-tokens, --seed  sampling (TTS defaults 0.6, 0.8, 1200)\n\
         \x20 --speak-json JSON    all speak options as a JSON object (see oxinfer::tts::SpeakOptions)\n\
         \x20 --no-noise           no SNAC noise; --unconstrained samples from the whole vocabulary\n\
         \x20 --decode-tokens FILE decode audio token ids (JSON array) to -o FILE (--model may be a SNAC dir)\n\
         \x20 --dump-pcm FILE      also write the samples as raw f32\n\
         \x20 --ref-audio FILE     voice cloning: a reference clip (WAV/MP3/FLAC/..., raw .f32 at 24 kHz),\n\
         \x20 --ref-text TEXT      with its transcript; repeatable (more pairs: a steadier voice). Best\n\
         \x20                      with the pretrained model (unsloth/orpheus-3b-0.1-pretrained)\n\
         \x20 --snac-encode FILE   (--model a SNAC dir) audio to codes, a JSON array per codebook (-o FILE)\n\
         \x20 --decode-codes FILE  (--model a SNAC dir) such codes back to a WAV (-o FILE)\n\
         \n\
         text to speech (ZONOS3-Mini: Zyphra/ZONOS3-Mini, 44.1 kHz; see docs/zonos.md):\n\
         \x20 --speak, --speak-file, --speak-split, --speak-json, --dump-pcm as above\n\
         \x20 --voice NAME         a default voice (Savannah, Imogen, Rupert, ...), a speaker embedding or\n\
         \x20                      reference codes as .npy, \"\" for none; --ref-audio FILE.npy is the same\n\
         \x20 --snac NAME          the codec decoder: keep2 (default), full, or a zac GGUF file\n\
         \x20 --temp, --top-p, --repetition-penalty, --max-tokens, --seed  (defaults 0.75, 0.69, 2.4, 2000 rows)"
    );
    eprintln!("\n{}", intervene_cli::USAGE);
    eprintln!("\n{}", flux_cli::USAGE);
    eprintln!("{}", sd3_cli::USAGE);
    eprintln!("{}", bagel_cli::USAGE);
    #[cfg(feature = "server")]
    eprintln!("\n{}", server_cli::USAGE);
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
        no_draft: false,
        draft_tokens: None,
        draft_p_min: None,
        draft_device: None,
        draft_conf_min: None,
        fast_verify: true,
        lookup: None,
        lookup_tokens: None,
        lookup_min_match: None,
        lookup_memory: true,
        lookup_fixed: false,
        prompt_cache: None,
        prompt_cache_gb: None,
        save_session: None,
        load_session: None,
        mtp: false,
        mtp_fixed: false,
        mtp_from: None,
        mtp_cost: None,
        stream: false,
        no_stream: false,
        mmproj: Default::default(),
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
        init_images: Vec::new(),
        transformer: None,
        qwen_image: Default::default(),
        size: None,
        sampler: None,
        loras: Vec::new(),
        mask: None,
        strength: None,
        image_guidance: None,
        device: platform::DevicePref::Auto,
        devices: false,
        models: false,
        models_all: false,
        models_kind: None,
        models_can: Vec::new(),
        check_update: false,
        update: false,
        force: false,
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
        transcribe_more: Vec::new(),
        beam: None,
        patience: None,
        length_penalty: None,
        best_of: None,
        batch: None,
        generate_image: None,
        num_images: 1,
        temp_given: false,
        speak: None,
        voice: None,
        snac: None,
        speak_stream: false,
        speak_json: None,
        speak_more: Vec::new(),
        speak_file: None,
        speak_split: false,
        no_noise: false,
        unconstrained: false,
        decode_tokens: None,
        dump_pcm: None,
        ref_audio: Vec::new(),
        ref_text: Vec::new(),
        snac_encode: None,
        decode_codes: None,
        repetition_penalty: None,
        temperature_set: None,
        top_p_set: None,
        max_tokens_set: None,
        iv: Default::default(),
        flux: Default::default(),
        sd3: Default::default(),
        bagel: Default::default(),
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
            "--max-tokens" | "-n" => {
                a.max_tokens = val()?.parse()?;
                a.max_tokens_set = Some(a.max_tokens);
            }
            "--temp" | "-t" => {
                a.temperature = val()?.parse()?;
                a.temp_given = true;
                a.temperature_set = Some(a.temperature);
            }
            "--top-k" => a.top_k = val()?.parse()?,
            "--top-p" => {
                a.top_p = val()?.parse()?;
                a.top_p_set = Some(a.top_p);
            }
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
            "--draft-tokens" | "--spec-draft-n-max" => a.draft_tokens = Some(val()?.parse()?),
            "--no-draft" => a.no_draft = true,
            "--spec-draft-p-min" | "--draft-p-min" => a.draft_p_min = Some(val()?.parse()?),
            "--spec-draft-conf-min" | "--draft-conf-min" => a.draft_conf_min = Some(val()?.parse()?),
            "--spec-draft-fixed" => a.mtp_fixed = true,
            "--fast-verify" => a.fast_verify = true,
            "--exact-verify" => a.fast_verify = false,
            "--lookup" => a.lookup = Some(true),
            "--no-lookup" => a.lookup = Some(false),
            "--lookup-tokens" => a.lookup_tokens = Some(val()?.parse()?),
            "--lookup-min-match" => a.lookup_min_match = Some(val()?.parse()?),
            "--no-lookup-memory" => a.lookup_memory = false,
            "--lookup-fixed" => a.lookup_fixed = true,
            "--prompt-cache" => a.prompt_cache = Some(val()?),
            "--prompt-cache-gb" => a.prompt_cache_gb = Some(val()?.parse()?),
            "--save-session" => a.save_session = Some(PathBuf::from(val()?)),
            "--load-session" => a.load_session = Some(PathBuf::from(val()?)),
            "--draft-device" => {
                a.draft_device = match val()?.as_str() {
                    "auto" => None,
                    "gpu" => Some(true),
                    "cpu" => Some(false),
                    v => bail!("--draft-device: expected auto, gpu or cpu, got {v:?}"),
                }
            }
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
                a.no_stream = false;
                a.mmap = true;
            }
            "--no-stream" => {
                a.no_stream = true;
                a.stream = false;
            }
            "--mmproj" => a.mmproj = oxinfer::pixtral::Mmproj::parse(&val()?),
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
            "--models" => a.models = true,
            "--check-update" => a.check_update = true,
            "--update" => a.update = true,
            "--force" => a.force = true,
            "--all" => a.models_all = true,
            "--can" => {
                for v in val()?.split(',') {
                    a.models_can.push(oxinfer::advisor::Cap::parse(v).with_context(|| {
                        let all: Vec<&str> = oxinfer::advisor::Cap::ALL.iter().map(|c| c.as_str()).collect();
                        format!("unknown capability {v} (one of {})", all.join(", "))
                    })?);
                }
            }
            "--kind" => {
                let v = val()?;
                a.models_kind =
                    Some(oxinfer::advisor::Kind::parse(&v).with_context(|| format!("unknown kind {v} (chat|embedding|image|speech|tts)"))?);
            }
            "--output" | "-o" => a.output = Some(PathBuf::from(val()?)),
            "--steps" => a.steps = Some(val()?.parse()?),
            "--guidance" | "--cfg" => a.guidance = Some(val()?.parse()?),
            "--generate-image" => a.generate_image = Some(val()?),
            "--num-images" => a.num_images = val()?.parse()?,
            "--negative" => a.negative = val()?,
            "--sampler" => a.sampler = Some(val()?),
            "--lora" => a.loras.push(oxinfer::sd::lora::LoraSpec::parse(&val()?)?),
            "--init-image" => a.init_images.push(PathBuf::from(val()?)),
            "--mask" => a.mask = Some(PathBuf::from(val()?)),
            "--strength" => a.strength = Some(val()?.parse()?),
            "--image-guidance" => a.image_guidance = Some(val()?.parse()?),
            "--transformer" => a.transformer = Some(PathBuf::from(val()?)),
            // FLUX (Canny / Depth) and Qwen-Image (ControlNet) both take control images.
            "--control-image" => {
                let v = val()?;
                a.flux.control_image = Some(PathBuf::from(&v));
                a.qwen_image.set(&k, &v)?
            }
            "--controlnet" | "--control-scale" | "--control-start" | "--control-end" | "--layers" | "--resolution" => {
                a.qwen_image.set(&k, &val()?)?
            }
            "--size" => {
                let v = val()?;
                let (w, h) = v
                    .split_once(['x', 'X'])
                    .with_context(|| format!("--size expects WIDTHxHEIGHT, got {v}"))?;
                a.size = Some((w.parse()?, h.parse()?));
            }
            "--stream-gpu" => {
                let v = val()?;
                oxinfer::gpu::set_stream_decode(match v.as_str() {
                    "on" | "1" | "true" => Some(true),
                    "off" | "0" | "false" => Some(false),
                    "auto" => None,
                    _ => bail!("--stream-gpu expects on|off|auto, got {v}"),
                });
            }
            "--gpu-decode" => {
                let v = val()?;
                oxinfer::gpu::set_decode(match v.as_str() {
                    "on" | "1" | "true" => Some(true),
                    "off" | "0" | "false" => Some(false),
                    "auto" => None,
                    _ => bail!("--gpu-decode expects on|off|auto, got {v}"),
                });
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
            "--transcribe" if a.transcribe.is_some() => a.transcribe_more.push(PathBuf::from(val()?)),
            "--transcribe" => a.transcribe = Some(PathBuf::from(val()?)),
            "--beam" => a.beam = Some(val()?.parse()?),
            "--patience" => a.patience = Some(val()?.parse()?),
            "--length-penalty" => a.length_penalty = Some(val()?.parse()?),
            "--best-of" => a.best_of = Some(val()?.parse()?),
            "--batch" => a.batch = Some(val()?.parse()?),
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
            "--speak" if a.speak.is_some() => a.speak_more.push(val()?),
            "--speak" => a.speak = Some(val()?),
            "--speak-file" => a.speak_file = Some(PathBuf::from(val()?)),
            "--speak-split" => a.speak_split = true,
            "--voice" => a.voice = Some(val()?),
            "--snac" => a.snac = Some(val()?),
            "--speak-stream" => a.speak_stream = true,
            "--speak-json" => a.speak_json = Some(val()?),
            "--no-noise" => a.no_noise = true,
            "--unconstrained" => a.unconstrained = true,
            "--decode-tokens" => a.decode_tokens = Some(PathBuf::from(val()?)),
            "--dump-pcm" => a.dump_pcm = Some(PathBuf::from(val()?)),
            "--ref-audio" => a.ref_audio.push(PathBuf::from(val()?)),
            "--ref-text" => a.ref_text.push(val()?),
            "--snac-encode" => a.snac_encode = Some(PathBuf::from(val()?)),
            "--decode-codes" => a.decode_codes = Some(PathBuf::from(val()?)),
            "--repetition-penalty" | "--repeat-penalty" => a.repetition_penalty = Some(val()?.parse()?),
            "-h" | "--help" => usage(),
            _ if a.iv.parse(&k, &mut val)? => {}
            _ if a.flux.parse(&k, &mut val)? => {}
            _ if a.sd3.parse(&k, &mut val)? => {}
            _ if a.bagel.parse(&k, &mut val)? => {}
            _ => {
                eprintln!("unknown argument {k}");
                usage()
            }
        }
    }
    if a.model.as_os_str().is_empty() && !a.devices && !a.models && !a.check_update && !a.update {
        usage();
    }
    Ok(a)
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// The command line's attachments, prepared for a model of hidden size `hidden`.
fn mm_for(args: &Args, files: &safetensors::ModelFiles, hidden: usize) -> Result<Mm> {
    if mm_request(args).is_empty() {
        return Ok(Default::default());
    }
    oxinfer::mm::prepare(&mm_request(args), files, hidden)
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
        dump_mm: args.dump_mm.as_deref(),
        mmproj: args.mmproj.clone(),
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
/// `fed` are the tokens just prefilled (the prompt-lookup drafter's context grows with them).
#[allow(clippy::too_many_arguments)]
fn generate<M: oxinfer::dspark::Target>(
    model: &M,
    sess: &mut M::Sess,
    tok: &Tokenizer,
    logits: Vec<f32>,
    sampler: &mut Sampler,
    eos: &[u32],
    max_tokens: usize,
    fmt: &dyn ChatFormat,
    think: bool,
    fed: &[u32],
) -> (Vec<u32>, usize, f64) {
    let mut printer = Printer::new(fmt, think, tok);
    let stream = STREAM.lock().unwrap().clone();
    let trace = TRACE.load(std::sync::atomic::Ordering::Relaxed);
    let s_start = stream.as_ref().map(|s| s.stats());
    let (mut s_last, mut t_last) = (s_start, Instant::now());
    let mut lookup = LOOKUP.lock().unwrap();
    let lookup_on = lookup.is_some();
    let d = match lookup.as_mut() {
        Some(lk) => {
            lk.extend(fed);
            lk.opts.stop = eos.to_vec();
            let (d, st) = {
                let mut dec = oxinfer::lookup::LookupRounds::new(model, sess, lk);
                let d = generate::decode_rounds(&mut dec, &logits, sampler, eos, max_tokens, &mut |out| printer.print(tok, out));
                (d, dec.stats.clone())
            };
            // As plain decoding does, run the last token unless it is a stop token.
            if let Some(t) = d.unfed {
                model.forward(sess, &[t]);
                lk.extend(&[t]);
            }
            if d.forwards > 0 {
                eprintln!(
                    "[lookup: {} forwards for {} tokens = {:.2} tok/forward; drafts accepted {}/{} = {:.0}%; {} drafts vetoed so far]",
                    d.forwards,
                    d.tokens.len(),
                    d.tokens.len() as f64 / d.forwards as f64,
                    st.accepted,
                    st.drafted,
                    100.0 * st.accepted as f64 / st.drafted.max(1) as f64,
                    lk.stats.vetoed
                );
            }
            d
        }
        None => generate::decode(model, sess, logits, sampler, eos, max_tokens, &mut |out| {
            // Every token after the first follows one forward pass.
            if let (true, Some(s), Some(s0)) = (trace && out.len() > 1, &stream, &s_last) {
                let now = s.stats();
                eprintln!("[token: {}]", stream_line(&(now - *s0), 1, t_last.elapsed().as_secs_f64()));
                (s_last, t_last) = (Some(now), Instant::now());
            }
            printer.print(tok, out)
        }),
    };
    drop(lookup);
    printer.finish(tok, &d.tokens, eos);
    if let (Some(s), Some(s0)) = (&stream, s_start) {
        let st = s.stats() - s0;
        eprintln!(
            "[stream: {}; {:.2} GB read per forward]",
            stream_line(&st, d.forwards, d.secs),
            st.bytes_read as f64 / 1e9 / d.forwards.max(1) as f64
        );
    }
    // Tokens generated (plain decoding: one forward each; with prompt lookup, fewer forwards).
    let n = d.tokens.len() - usize::from(d.reason == generate::StopReason::Stop);
    (d.tokens, if lookup_on { n } else { d.forwards }, d.secs)
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

/// `--check-update` / `--update`: the latest GitHub release against this binary.
#[cfg(feature = "update")]
fn run_update(args: &Args) -> Result<()> {
    use oxinfer::update::{CURRENT, Outcome, latest, platform_suffix, update};
    if !args.update {
        let rel = latest()?;
        if rel.is_newer() {
            let binary = match (platform_suffix(), rel.asset()) {
                (_, Some(a)) => format!("{} ({:.1} MB); oxinfer --update installs it", a.name, a.size as f64 / 1e6),
                (Some(s), None) => format!("no {s} binary attached to it"),
                (None, None) => "no binary is published for this platform".to_string(),
            };
            println!("oxinfer {CURRENT}: {} is available ({}): {binary}", rel.version(), rel.html_url);
        } else {
            println!("oxinfer {CURRENT} is the latest release");
        }
        return Ok(());
    }
    match update(args.force, &mut |line| eprintln!("[{line}]"))? {
        Outcome::UpToDate(v) => println!("oxinfer {CURRENT} is up to date (latest release: {v}); --force reinstalls it"),
        Outcome::Updated(exe, v) => println!("updated {} from {CURRENT} to {v}", exe.display()),
    }
    Ok(())
}

#[cfg(not(feature = "update"))]
fn run_update(_args: &Args) -> Result<()> {
    anyhow::bail!("this build has no self-update (the `update` feature)")
}

/// `--models`: the catalog of tested models against this machine.
fn run_models(args: &Args) -> Result<()> {
    pool::init(args.threads.unwrap_or_else(default_threads));
    let hw = oxinfer::advisor::Hardware::detect(args.device, args.cache_dir.as_deref());
    let filter = args.model.to_string_lossy().to_lowercase();
    let rows: Vec<_> = oxinfer::advisor::survey(&hw)
        .into_iter()
        .filter(|r| {
            args.models_kind.is_none_or(|k| r.model.kind == k)
                && args.models_can.iter().all(|c| r.model.can(*c))
                && r.model.repo.to_lowercase().contains(&filter)
        })
        .collect();
    if args.format.as_deref() == Some("json") {
        println!("{}", serde_json::to_string_pretty(&oxinfer::advisor::to_json(&hw, &rows))?);
    } else {
        print!("{}", oxinfer::advisor::render(&hw, &rows, args.models_all));
    }
    Ok(())
}

fn main() -> Result<()> {
    #[cfg(feature = "server")]
    if std::env::args().nth(1).as_deref() == Some("serve") {
        let rest: Vec<String> = std::env::args().skip(2).collect();
        return server_cli::run(&rest);
    }
    let mut args = parse_args()?;
    if args.devices {
        print!("{}", platform::detect().report());
        return Ok(());
    }
    if args.models {
        return run_models(&args);
    }
    if args.check_update || args.update {
        return run_update(&args);
    }
    // Resolve repo ids to cached local directories.
    let spec = args.model.to_string_lossy().to_string();
    // (A single safetensors file: a BFL FLUX checkpoint, local or `org/name/file.safetensors`.)
    args.model = if spec.ends_with(".safetensors") {
        resolve_file(&spec, args.cache_dir.as_deref())?
    } else if !args.model.exists() && oxinfer::zonos::is_zonos(&args.model) {
        // ZONOS3's repository: its loader fetches the few files it reads.
        args.model.clone()
    } else {
        resolve_model(&spec, args.cache_dir.as_deref())?
    };
    if let Some(d) = &args.draft
        && d.as_os_str() != "auto"
    {
        let spec = d.to_string_lossy().to_string();
        let d = resolve_model(&spec, args.cache_dir.as_deref())?;
        // A draft repository holding several GGUF quantizations: its Q8_0 draft.
        args.draft = Some(oxinfer::dspark::draft_in(&d).unwrap_or(d));
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
    if args.prompt_cache.is_some() || args.save_session.is_some() || args.load_session.is_some() {
        return run_session(&args);
    }
    if args.decide.is_some() || args.decide_json.is_some() {
        return run_decide(&args);
    }
    if args.decision_from.is_some() {
        bail!("--decision-from goes with --decide or --decide-json");
    }
    run(args)
}

/// Text generation through the engine with session persistence (`--prompt-cache`,
/// `--load-session`, `--save-session`): a one-shot prompt or a `--chat` loop, the conversation
/// run by `Engine::chat` (which restores from the prompt cache) or, after `--load-session`,
/// continued with `Engine::send`.
fn run_session(args: &Args) -> Result<()> {
    use oxinfer::engine::{ChatConfig, Engine, EngineOptions, Message, SamplingParams, SystemPrompt};
    let mut opts = EngineOptions::default();
    opts.quant = args.quant;
    opts.mmap = args.mmap;
    opts.preload = args.preload;
    opts.device = args.device;
    opts.prefill_chunk = args.prefill_chunk;
    opts.cache_dir = args.cache_dir.clone();
    opts.draft = if args.no_draft {
        Some("none".into())
    } else {
        args.draft.as_ref().map(|d| d.to_string_lossy().into_owned())
    };
    opts.draft_tokens = args.draft_tokens;
    opts.draft_p_min = args.draft_p_min;
    opts.draft_device = args.draft_device;
    opts.draft_conf_min = args.draft_conf_min;
    opts.fast_verify = args.fast_verify;
    opts.mtp = args.mtp;
    opts.mtp_fixed = args.mtp_fixed;
    opts.mtp_from = args.mtp_from.as_ref().map(|d| d.to_string_lossy().into_owned());
    opts.lookup = args.lookup;
    opts.lookup_tokens = args.lookup_tokens;
    opts.lookup_min_match = args.lookup_min_match;
    opts.lookup_memory = args.lookup_memory;
    opts.lookup_fixed = args.lookup_fixed;
    if let Some(p) = &args.prompt_cache {
        opts.set("prompt_cache", p)?;
    }
    opts.prompt_cache_gb = args.prompt_cache_gb;
    let mut e = Engine::load(&args.model.to_string_lossy(), &opts)?;
    let info = e.info().clone();
    eprintln!(
        "[{} {} ({}, {}) on {}, loaded in {:.1}s, speculative {:?}]",
        info.model_type,
        info.path.display(),
        info.format,
        info.quant.as_str(),
        info.device,
        info.load_secs,
        info.speculative
    );
    if let Some((dir, n, bytes)) = e.prompt_cache_info() {
        eprintln!("[prompt cache {}: {n} snapshots, {:.2} GB]", dir.display(), bytes as f64 / 1e9);
    }
    let c: &mut ChatConfig = e.config_mut();
    c.system = match (&args.system, args.no_system) {
        (_, true) => SystemPrompt::None,
        (Some(s), _) => SystemPrompt::Text(s.clone()),
        _ => SystemPrompt::Default,
    };
    c.think = args.think;
    c.max_tokens = args.max_tokens;
    e.set_sampling(SamplingParams {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
        repetition_penalty: args.repetition_penalty.unwrap_or(1.0),
    });
    let loaded = match &args.load_session {
        Some(p) => {
            let t = Instant::now();
            let n = e.load_session(p)?;
            eprintln!("[restored {n} tokens from {} in {:.3}s]", p.display(), t.elapsed().as_secs_f64());
            true
        }
        None => false,
    };
    let mut messages: Vec<Message> = Vec::new();
    let turn = |e: &mut Engine, text: &str, messages: &mut Vec<Message>| -> Result<()> {
        let mut out = std::io::stdout();
        let mut print = |d: &oxinfer::engine::Delta| {
            let _ = out.write_all(d.text.as_bytes());
            let _ = out.flush();
            Flow::Continue
        };
        let reply = if loaded {
            e.send(text, &mut print)?
        } else {
            messages.push(Message::user(text));
            e.chat(messages, &mut print)?
        };
        println!();
        messages.push(Message::assistant(reply.text.clone()));
        let s = &reply.stats;
        report(s.prompt_tokens, s.prefill_secs, s.generated_tokens, s.decode_secs);
        if s.cached_tokens > 0 {
            eprintln!(
                "[{} prompt tokens reused ({} restored from the prompt cache in {:.3}s)]",
                s.cached_tokens, s.restored_tokens, s.restore_secs
            );
        }
        if s.drafted > 0 {
            eprintln!(
                "[speculative: {} forwards, {:.2} tok/forward, drafts accepted {}/{} = {:.0}%]",
                s.forwards,
                s.tokens_per_forward(),
                s.accepted,
                s.drafted,
                100.0 * s.acceptance()
            );
        }
        let t = Instant::now();
        if e.flush_prompt_cache()? {
            eprintln!("[stored the conversation in the prompt cache in {:.3}s]", t.elapsed().as_secs_f64());
        }
        Ok(())
    };
    if args.chat {
        let stdin = std::io::stdin();
        loop {
            eprint!("> ");
            let mut line = String::new();
            if stdin.lock().read_line(&mut line)? == 0 {
                break;
            }
            let line = line.trim_end_matches(['\n', '\r']);
            if !line.trim().is_empty() {
                turn(&mut e, line, &mut messages)?;
            }
        }
    } else {
        let prompt = match &args.prompt {
            Some(p) => p.clone(),
            None => {
                let mut s = String::new();
                std::io::stdin().lock().read_to_string(&mut s)?;
                s
            }
        };
        turn(&mut e, &prompt, &mut messages)?;
    }
    if let Some(p) = &args.save_session {
        let t = Instant::now();
        let n = e.save_session(p)?;
        eprintln!(
            "[saved {} tokens to {} ({:.1} MB) in {:.3}s]",
            e.session_tokens().len(),
            p.display(),
            n as f64 / 1e6,
            t.elapsed().as_secs_f64()
        );
    }
    Ok(())
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
    opts.stream_mode = match (args.stream, args.no_stream) {
        (true, _) => oxinfer::engine::StreamMode::On,
        (_, true) => oxinfer::engine::StreamMode::Off,
        _ => oxinfer::engine::StreamMode::Auto,
    };
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
    if !args.videos.is_empty() || !args.audios.is_empty() {
        bail!("decisions take images only (--image): the /v1/decide protocol has no video or audio parts");
    }
    let report = |dec: &oxinfer::decide::Decision| {
        let s = &dec.stats;
        let images = match s.images {
            0 => String::new(),
            n => format!(", {n} image(s) = {} tokens, vision {:.3}s", s.image_tokens, s.vision_secs),
        };
        eprintln!(
            "[{} {} | {} prompt tokens{images}, prefill {:.3}s ({:.0} tok/s), total {:.3}s]",
            dec.kind.as_str(),
            dec.adaptation.as_str(),
            s.prompt_tokens,
            s.prefill_secs,
            s.prefill_tok_s(),
            s.total_secs
        );
    };
    if let Some(path) = &args.decide_json {
        if !args.images.is_empty() {
            bail!("--image goes with --decide; with --decide-json, put the images in each request's state");
        }
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
            let mut req = DecisionRequest::from_json(r)?;
            if req.attachments.max_soft_tokens == 0 {
                req.attachments.max_soft_tokens = args.max_soft_tokens;
            }
            let dec = e.decide(&req)?;
            println!("{}", dec.to_json(&model_name, &protocol));
            report(&dec);
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
    let mut req = DecisionRequest::new(kind, state, question, args.options.clone()).with_images(args.images.iter().cloned());
    req.attachments.max_soft_tokens = args.max_soft_tokens;
    let dec = e.decide(&req)?;
    let w = dec.options.iter().map(|o| o.chars().count()).max().unwrap_or(0).min(60);
    for (i, (o, p)) in dec.options.iter().zip(&dec.probabilities).enumerate() {
        let bar = "#".repeat((p * 40.0).round() as usize);
        let mark = if i == dec.best_index { '*' } else { ' ' };
        println!("{mark} {o:<w$}  {p:.4}  {bar}");
    }
    if let Some(s) = dec.expected_score() {
        println!("  expected score {s:.3}");
    }
    report(&dec);
    Ok(())
}

fn run(mut args: Args) -> Result<()> {
    *LOOKUP.lock().unwrap() = make_lookup(&args, &[]);
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
    let gpu = oxinfer::gpu::init(&plat, args.device)?;
    match gpu {
        Some(b) if b.decodes() && oxinfer::gpu::decode_enabled() => eprintln!(
            "[device: {} (prefill matmuls and, where supported, whole decode steps on the GPU)]",
            b.name()
        ),
        Some(b) => eprintln!("[device: {} (prefill matmuls on the GPU, everything else on the CPU)]", b.name()),
        None => eprintln!("[device: cpu]"),
    }
    // Bigger prefill chunks keep the GPU busier (577 vs 431 tok/s on E4B q4 at 1024 vs 256);
    // on the CPU 256 keeps the activations in cache.
    args.prefill_chunk = Some(args.prefill_chunk.unwrap_or(if gpu.is_some() { 1024 } else { 256 }));
    if oxinfer::omnigen2::is_omnigen2(&args.model) {
        if !omnigen2_cli::is_understanding(&args) {
            return omnigen2_cli::run(&args);
        }
        // Questions about images: the model's Qwen2.5-VL as a chat model.
        args = omnigen2_cli::understanding_args(args);
    }
    if oxinfer::flux::is_flux_dir(&args.model)
        || oxinfer::flux::is_flux_gguf(&args.model)
        || oxinfer::flux::is_flux_safetensors(&args.model)
    {
        return flux_cli::run(&args);
    }
    if oxinfer::sd3::is_sd3_dir(&args.model) || oxinfer::sd3::is_sd3_gguf(&args.model) {
        return sd3_cli::run(&args);
    }
    if oxinfer::qwen_image::is_qwen_image(&args.model) {
        return qwen_image_cli::run(&args);
    }
    if oxinfer::minimax_h3::is_h3_image(&args.model) {
        return minimax_h3_cli::run(&args);
    }
    if oxinfer::bagel::is_bagel(&args.model) {
        return bagel_cli::run(&args);
    }
    if oxinfer::step1x::is_step1x(&args.model) {
        return step1x_cli::run(&args);
    }
    if oxinfer::sd::is_diffusers(&args.model) {
        return run_image(&args);
    }
    if args.embed || oxinfer::embed::is_embedding_model(&args.model) {
        return run_embed(&args);
    }
    if args.speak.is_some()
        || args.speak_file.is_some()
        || args.decode_tokens.is_some()
        || args.snac_encode.is_some()
        || args.decode_codes.is_some()
    {
        #[cfg(feature = "media")]
        if oxinfer::zonos::is_zonos(&args.model) {
            return zonos_cli::run(&args);
        }
        #[cfg(feature = "media")]
        return tts_cli::run(&args);
        #[cfg(not(feature = "media"))]
        bail!("speech synthesis from the command line needs the `media` feature (audio encoding)");
    }
    let model_type = config::model_type(&args.model)?;
    if model_type == "whisper" || args.transcribe.is_some() {
        #[cfg(feature = "media")]
        return whisper_cli::run(&args);
        #[cfg(not(feature = "media"))]
        bail!("speech to text from the command line needs the `media` feature (audio decoding)");
    }
    if model_type == "multi_modality" {
        return janus_cli::run(&args);
    }
    match model_type.as_str() {
        "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" | "gemma3" | "gemma3_text" => run_gemma(args),
        "qwen3" | "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_6" | "qwen3_8" | "qwen2" | "qwen2_5_vl"
        | "qwen3_vl" | "qwen4_exp" | "qwen4_exp_text" => run_qwen(args),
        // Llama 3 runs on the Qwen3 code path.
        "llama" => run_qwen(args),
        // MiniMax M2 / M2.1 / M2.5 / M2.7: the Qwen code path with MiniMax's attention norms and router.
        "minimax_m2" | "minimax_m3_vl" | "minimax_m3_vl_text" | "minimax" | "minimax_text_01" | "minimax_m1" | "minimax_vl_01" => {
            run_qwen(args)
        }
        "mistral_large4" | "mistral_large4_text" => run_mistral4(args),
        "deepseek_v4" | "deepseek_v41" | "ernie5_5_moe" => run_deepseek4(args),
        m if oxinfer::glm5::is_model_type(m) => run_glm5(args),
        "kimi_k3" | "kimi_k31" => run_kimi(args),
        m if oxinfer::nemotron::is_model_type(m) => run_nemotron(args),
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
fn continue_with_tools<M: oxinfer::dspark::Target>(
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
        let (o, n, secs) = generate(model, sess, tok, logits, sampler, eos, args.max_tokens, fmt, args.think, &ids);
        report(ids.len(), prefill_secs, n, secs);
        out = o;
    }
    Ok(out)
}

/// Loads an image file (PNG, JPEG, WebP, ...) for an edit.
fn load_edit_image(path: &std::path::Path) -> Result<oxinfer::video::RgbImage> {
    #[cfg(feature = "media")]
    return oxinfer::media::Media::new().load_image(path);
    #[cfg(not(feature = "media"))]
    bail!("reading {} needs the `media` feature (image decoding)", path.display())
}

/// Text-to-image and image editing with a diffusers Stable Diffusion folder.
fn run_image(args: &Args) -> Result<()> {
    use oxinfer::sd::{EditInput, EditKind, Params, Pipeline, sched::Sampler};
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
    // The input of an edit: --init-image, or the first --image.
    let edit = match args.init_images.first().or(args.images.first()) {
        Some(path) => {
            let mut e = EditInput::new(load_edit_image(path)?);
            if let Some(m) = &args.mask {
                e = e.with_mask(load_edit_image(m)?);
            }
            e.strength = args.strength;
            e.image_guidance = args.image_guidance;
            Some(e)
        }
        None if args.mask.is_some() => bail!("--mask needs the image to inpaint (--init-image FILE)"),
        None => None,
    };
    let kind = match &edit {
        Some(e) => Some(pipe.edit_kind(e.mask.is_some())?),
        None => None,
    };
    let side = pipe.default_size();
    let (width, height) = match (&edit, args.size) {
        (_, Some(s)) => s,
        (Some(e), None) => pipe.edit_size(e.image.width, e.image.height),
        (None, None) => (side, side),
    };
    let sampler = match args.sampler.as_deref() {
        None => pipe.default_sampler(),
        Some(name) => Sampler::parse(name).with_context(|| format!("unknown sampler {name} (use {})", Sampler::NAMES))?,
    };
    // Turbo models: steps * strength must reach one step (strength 0.5 -> 2 steps, 1 run).
    let strength = args.strength.unwrap_or(match kind {
        Some(EditKind::Img2Img) => oxinfer::sd::DEFAULT_STRENGTH,
        _ => 1.0,
    });
    let params = Params {
        prompt,
        negative: args.negative.clone(),
        steps: args.steps.unwrap_or(oxinfer::sd::default_steps(pipe.turbo, kind, strength)),
        guidance: args.guidance.unwrap_or(pipe.default_guidance()),
        seed: args.seed,
        width,
        height,
        sampler,
    };
    let what = match kind {
        None => String::new(),
        Some(EditKind::Instruct) => format!(
            ", instruction edit, image guidance {}",
            args.image_guidance.unwrap_or(oxinfer::sd::DEFAULT_IMAGE_GUIDANCE)
        ),
        Some(EditKind::Img2Img) => format!(", image to image, strength {strength}"),
        Some(EditKind::Inpaint) => format!(", inpainting (inpainting UNet), strength {strength}"),
        Some(EditKind::InpaintBlend) => format!(", inpainting (mask blending), strength {strength}"),
    };
    eprintln!(
        "[{}x{}, {} steps, guidance {}, {:?}, seed {}{what}]",
        width, height, params.steps, params.guidance, sampler, params.seed
    );
    let mut progress = |i, n| eprint!("\r[step {i}/{n}]");
    let (img, tm) = match &edit {
        Some(e) => pipe.generate_from(&params, e, &mut progress)?,
        None => pipe.generate(&params, &mut progress)?,
    };
    eprintln!();
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from("image.png"));
    std::fs::write(&out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb)).with_context(|| format!("write {}", out.display()))?;
    eprintln!(
        "[text {:.2}s |{} denoise {:.2}s ({} UNet passes, {:.2}s each) | decode {:.2}s | wrote {}]",
        tm.text,
        if edit.is_some() {
            format!(" encode {:.2}s |", tm.encode)
        } else {
            String::new()
        },
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

fn run_text<M: oxinfer::dspark::Target + oxinfer::intervene::Host>(
    args: &Args,
    model: &M,
    tok: &Tokenizer,
    eos: &[u32],
    fmt: &dyn ChatFormat,
    mm: Mm,
) -> Result<()> {
    let chunk = args.prefill_chunk.unwrap_or(256);
    if intervene_cli::run_capture(
        &args.iv,
        model,
        tok,
        fmt,
        args.raw,
        args.think,
        args.prompt.as_deref(),
        args.output.as_deref(),
        chunk,
    )? {
        return Ok(());
    }
    let (mm_text, mut mm_feats) = mm;
    let encode = |text: &str| -> Result<Vec<u32>> {
        let enc = tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    };
    let mut sampler =
        Sampler::new(args.temperature, args.top_k, args.top_p, args.seed).with_repetition_penalty(args.repetition_penalty.unwrap_or(1.0));
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
            if model.session_len(&sess) == 0
                && let Some(st) = model.iv_state(&mut sess)
            {
                intervene_cli::install(&args.iv, args.cache_dir.as_deref(), model, tok, st, &ids, chunk)?;
            }
            // The attachments belong to the first turn.
            let overrides = oxinfer::mm::bind(model, &mut sess, &ids, &std::mem::take(&mut mm_feats))?;
            let t = Instant::now();
            let hs = generate::prefill(model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
            let logits = model.logits(&hs[hs.len() - h..], 1);
            let prefill_secs = t.elapsed().as_secs_f64();
            prof::reset();
            let (out, n, decode_secs) = generate(
                model,
                &mut sess,
                tok,
                logits,
                &mut sampler,
                eos,
                args.max_tokens,
                fmt,
                args.think,
                &ids,
            );
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
    let sink = match model.iv_state(&mut sess) {
        Some(st) => intervene_cli::install(&args.iv, args.cache_dir.as_deref(), model, tok, st, &ids, chunk)?,
        None => None,
    };
    let overrides = oxinfer::mm::bind(model, &mut sess, &ids, &mm_feats)?;
    let t = Instant::now();
    let s0 = prefill_trace_start();
    let hs = generate::prefill(model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
    let logits = model.logits(&hs[hs.len() - h..], 1);
    let prefill_secs = t.elapsed().as_secs_f64();
    prefill_trace_end(s0, ids.len(), prefill_secs);
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
        if let Some(sink) = &sink {
            intervene_cli::print_features(sink, tok, &ids);
        }
        return Ok(());
    }
    prof::reset();
    let (out, n, decode_secs) = generate(
        model,
        &mut sess,
        tok,
        logits,
        &mut sampler,
        eos,
        args.max_tokens,
        fmt,
        args.think,
        &ids,
    );
    if args.show_tokens {
        eprintln!("[output ids: {out:?}]");
    }
    if let Some(sink) = &sink {
        let all: Vec<u32> = ids.iter().chain(&out).copied().collect();
        intervene_cli::print_features(sink, tok, &all);
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
        let e = emb.embed_dim(text, args.prompt_name.as_deref(), args.dim)?;
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

fn run_qwen(mut args: Args) -> Result<()> {
    let t0 = Instant::now();
    if args.mtp && args.draft.is_some() {
        bail!("choose --mtp or a DSpark --draft, not both");
    }
    if let Some(d) = &args.draft
        && d.as_os_str() != "auto"
        && !oxinfer::dspark::is_draft(d)
    {
        bail!("{}: not a DSpark draft", d.display());
    }
    let draft = if args.mtp { None } else { pick_dspark(&args)? };
    let cfg = config::QwenConfig::load(&args.model)?;
    if cfg.is_exp() {
        // Qwen4-Exp: the n-gram table (tens of GB) is read a few rows per token, so the files are
        // always mapped; a checkpoint larger than most of RAM streams as Mistral Large 4's does.
        let size: u64 = oxinfer::safetensors::checkpoint_bytes(&args.model);
        let too_big = oxinfer::qwen4exp::too_big(&args.model);
        if too_big && !args.stream && !args.mmap && !args.no_stream {
            let budget = oxinfer::mistral4::default_stream().cache_budget.unwrap_or(0) as f64 / 1e9;
            args.cache_budget_gb = Some(args.cache_budget_gb.unwrap_or(budget));
            eprintln!(
                "[the {:.0} GB checkpoint does not fit in memory: streaming it (--stream --cache-budget-gb {:.0})]",
                size as f64 / 1e9,
                args.cache_budget_gb.unwrap()
            );
            args.stream = true;
        }
        args.mmap = true;
    } else {
        auto_stream(&mut args);
    }
    let eos = config::load_eos_ids(&args.model);
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let mut model = qwen::QwenModel::load(cfg, &files, args.quant)?;
    let tok = load_tokenizer(&args.model)?;
    let qwen2_fmt;
    let fmt: &dyn ChatFormat = if config::model_type(&args.model)?.starts_with("minimax") {
        chat::minimax_for_model(&args.model)
    } else if model.cfg.qk_norm {
        chat::QwenChat::for_model(&args.model)
    } else if model.cfg.attention_bias {
        qwen2_fmt = chat::Qwen2Chat::for_model(&args.model);
        &qwen2_fmt
    } else {
        if !args.raw && !chat::Llama3Chat::matches_model(&args.model) {
            eprintln!("[warning: the model's chat template is not Llama 3's; prompts use the Llama 3 format anyway]");
        }
        &chat::Llama3Chat
    };
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
    // The MTP layer where it is the default (`qwen_mtp::BY_DEFAULT`), unless --no-draft, a draft,
    // or a mode plain decoding serves.
    if !args.mtp
        && draft.is_none()
        && args.draft.is_none()
        && !args.no_draft
        && args.save_quantized.is_none()
        && !(args.tools || args.dump_logits.is_some() || args.print_prompt || iv_requested(&args.iv))
        && (oxinfer::qwen_mtp::by_default(&args.model) || (args.stream && model.cfg.num_experts.unwrap_or(0) == 0))
        && oxinfer::qwen_mtp::mtp_prefix(&files).is_some()
    {
        eprintln!(
            "[MTP speculative decoding: {} (--no-draft for plain decoding)]",
            if args.stream {
                "the default when streaming, where a pass costs about the same for 1 or 9 tokens"
            } else {
                "this model's default"
            }
        );
        args.mtp = true;
    }
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
            oxinfer::qwen_vision::QwenVisionTower::load(qc.vision, &files)?;
            eprintln!("[vision tower kept (bf16)]");
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
        let mtp = mtp.as_mut().filter(|_| args.mtp);
        let dt = default_draft_tokens(&args);
        Some(start_stream(&args, |o| oxinfer::stream::attach_qwen(&mut model, mtp, &fs, dt, o))?)
    } else {
        None
    };
    *STREAM.lock().unwrap() = streamer.clone();
    TRACE.store(args.stream_trace, std::sync::atomic::Ordering::Relaxed);
    let draft_files = match &draft {
        Some(d) => Some((d, safetensors::ModelFiles::open(d, true)?)),
        None => None,
    };
    let draft = match &draft_files {
        Some((d, f)) => Some(load_dspark(d, f, model.cfg.hidden_size, model.cfg.vocab_size, model.quant)?),
        None => None,
    };
    let r = match (&mtp, &draft) {
        (Some(m), _) if args.mtp => run_qwen_mtp(&args, &model, m, &tok, &eos, fmt, streamer.as_deref(), mm),
        (_, Some(d)) => run_dspark(&args, &model, d, &tok, &eos, fmt, streamer.as_deref(), mm),
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

fn run_mistral4(mut args: Args) -> Result<()> {
    let t0 = Instant::now();
    if args.mtp {
        bail!("Mistral Large 4 has no MTP layer: speculative decoding uses a DSpark draft (--draft)");
    }
    // DSpark: the given draft, `auto` for the one beside a GGUF model; that one is also the
    // default (`dspark::BY_DEFAULT`) unless --no-draft or a mode plain decoding serves (tool
    // calls, logits dumps, prompt printing, interventions).
    let beside = oxinfer::dspark::draft_for(&args.model);
    let plain_only = args.tools || args.dump_logits.is_some() || args.print_prompt || iv_requested(&args.iv);
    let draft = match &args.draft {
        Some(d) if d.as_os_str() == "auto" => Some(resolve_auto_draft(&args.model, args.cache_dir.as_deref())?),
        Some(d) if !oxinfer::dspark::is_draft(d) => bail!("{}: not a DSpark (dflash) draft GGUF", d.display()),
        Some(d) => Some(d.clone()),
        None if oxinfer::dspark::BY_DEFAULT && !args.no_draft && !plain_only && !oxinfer::gpu::plain_decode_preferred() => beside.clone(),
        None => None,
    };
    if draft.is_none()
        && !args.no_draft
        && !plain_only
        && let Some(b) = &beside
    {
        eprintln!(
            "[a DSpark draft is beside the model ({}): --draft auto uses it]",
            b.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()
        );
    }
    let cfg = config::Mistral4Config::load(&args.model)?;
    let eos = config::load_eos_ids(&args.model);
    // A checkpoint larger than most of RAM cannot be read in: it is streamed with a page-cache
    // budget of 70% of RAM (the routed experts read as they are used), unless `--no-stream` (or
    // `--mmap`) asks for plain demand paging.
    let size: u64 = oxinfer::safetensors::checkpoint_bytes(&args.model);
    if let oxinfer::pixtral::Mmproj::File(p) = &args.mmproj {
        args.mmproj = oxinfer::pixtral::Mmproj::File(resolve_file(&p.to_string_lossy(), args.cache_dir.as_deref())?);
    }
    if oxinfer::mistral4::too_big(size) && args.no_stream {
        args.mmap = true;
    }
    if oxinfer::mistral4::too_big(size) && !args.stream && !args.mmap {
        let budget = oxinfer::mistral4::default_stream().cache_budget.unwrap_or(0) as f64 / 1e9;
        args.cache_budget_gb = Some(args.cache_budget_gb.unwrap_or(budget));
        eprintln!(
            "[the {:.0} GB checkpoint does not fit in memory: streaming it (--stream --cache-budget-gb {:.0})]",
            size as f64 / 1e9,
            args.cache_budget_gb.unwrap()
        );
        args.stream = true;
    }
    args.mmap |= args.stream;
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let mut model = oxinfer::mistral4::Mistral4Model::load(cfg, &files, args.quant)?;
    let tok = load_tokenizer(&args.model)?;
    let fmt: &dyn ChatFormat = &chat::MistralChat;
    let eos = chat::with_turn_end(fmt, &tok, eos);
    eprintln!(
        "[loaded Mistral Large 4: {} layers, hidden {}, {:.1} GB checkpoint ({}), {} weights streaming {:.2} GB/token, {} threads, {:.2}s]",
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
    if let Some(out) = &args.save_quantized {
        return save_quantized(&args.model, out, &files, model.quant);
    }
    let mm = if mm_request(&args).is_empty() {
        Default::default()
    } else {
        oxinfer::mm::prepare(&mm_request(&args), &files, model.cfg.hidden_size)?
    };
    let draft_files = match &draft {
        Some(d) => Some((d, safetensors::ModelFiles::open(d, true)?)),
        None => None,
    };
    let draft = match &draft_files {
        Some((d, f)) => {
            let t = Instant::now();
            let dr = oxinfer::dspark::Dspark::load(d, f, model.cfg.hidden_size, model.cfg.vocab_size, args.quant)?;
            eprintln!(
                "[DSpark draft {}: {} layers, block {}, {:.2} GB read per draft pass, {:.2}s]",
                d.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
                dr.cfg.n_layers,
                dr.cfg.block_size,
                dr.pass_bytes() as f64 / 1e9,
                t.elapsed().as_secs_f64()
            );
            Some(dr)
        }
        None => None,
    };
    let streamer = if args.stream {
        Some(start_stream(&args, |o| oxinfer::stream::attach_mistral4(&mut model, &files, o))?)
    } else {
        None
    };
    *STREAM.lock().unwrap() = streamer.clone();
    TRACE.store(args.stream_trace, std::sync::atomic::Ordering::Relaxed);
    let r = match &draft {
        Some(d) => run_dspark(&args, &model, d, &tok, &eos, fmt, streamer.as_deref(), mm),
        None => run_text(&args, &model, &tok, &eos, fmt, mm),
    };
    if let Some(s) = &streamer
        && let Some(c) = s.cached_bytes()
    {
        eprintln!("[stream: {:.2} GB of the model files in the page cache at exit]", c as f64 / 1e9);
    }
    *STREAM.lock().unwrap() = None;
    model.set_stream(None);
    drop(streamer);
    r
}

fn run_deepseek4(mut args: Args) -> Result<()> {
    let t0 = Instant::now();
    if args.mtp {
        bail!(
            "DeepSeek V4 speculative decoding uses a DSpark draft (--draft), not the MTP layer (ERNIE 5.5's MTP drafts are in the library only: deepseek4::ErnieMtpDecoder)"
        );
    }
    // DSpark: as for Mistral Large 4, the draft beside a GGUF model by default (not with
    // interventions, which plain decoding serves).
    let beside = oxinfer::dspark::draft_for(&args.model);
    let plain_only = args.tools || args.dump_logits.is_some() || args.print_prompt || iv_requested(&args.iv);
    let draft = match &args.draft {
        Some(d) if d.as_os_str() == "auto" => Some(resolve_auto_draft(&args.model, args.cache_dir.as_deref())?),
        Some(d) if !oxinfer::dspark::is_draft(d) => bail!("{}: not a DSpark (dflash) draft GGUF", d.display()),
        Some(d) => Some(d.clone()),
        None if oxinfer::dspark::BY_DEFAULT && !args.no_draft && !plain_only && !oxinfer::gpu::plain_decode_preferred() => beside.clone(),
        None => None,
    };
    let cfg = config::DeepSeek4Config::load(&args.model)?;
    // V4.1's draft stages (bias for image tokens, 128 experts, a confidence head) are not read yet.
    let draft = match draft {
        Some(_) if cfg.v41 && args.draft.is_some() => bail!("DSpark drafts are not supported for DeepSeek V4.1 yet"),
        Some(_) if cfg.v41 => None,
        Some(_) if cfg.ernie.is_some() && args.draft.is_some() => bail!("DSpark drafts are not supported for ERNIE 5.5"),
        Some(_) if cfg.ernie.is_some() => None,
        d => d,
    };
    let name = match (cfg.v41, cfg.ernie.is_some()) {
        (true, _) => "DeepSeek V4.1",
        (_, true) => "ERNIE 5.5",
        _ => "DeepSeek V4",
    };
    let eos = config::load_eos_ids(&args.model);
    // A checkpoint larger than most of RAM is streamed with a page-cache budget of 70% of RAM,
    // unless `--no-stream` (or `--mmap`) asks for plain demand paging.
    let size: u64 = oxinfer::safetensors::checkpoint_bytes(&args.model);
    if oxinfer::deepseek4::too_big(size) && args.no_stream {
        args.mmap = true;
    }
    if oxinfer::deepseek4::too_big(size) && !args.stream && !args.mmap {
        let budget = oxinfer::deepseek4::default_stream().cache_budget.unwrap_or(0) as f64 / 1e9;
        args.cache_budget_gb = Some(args.cache_budget_gb.unwrap_or(budget));
        eprintln!(
            "[the {:.0} GB checkpoint does not fit in memory: streaming it (--stream --cache-budget-gb {:.0})]",
            size as f64 / 1e9,
            args.cache_budget_gb.unwrap()
        );
        args.stream = true;
    }
    args.mmap |= args.stream;
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let mut model = oxinfer::deepseek4::DeepSeek4Model::load(cfg, &files, args.quant)?;
    let tok = load_tokenizer(&args.model)?;
    let fmt: &dyn ChatFormat = if model.cfg.v41 {
        &chat::DeepSeek41Chat
    } else if model.cfg.ernie.is_some() {
        &chat::ErnieBaseChat
    } else {
        &chat::DeepSeekChat
    };
    let eos = chat::with_turn_end(fmt, &tok, eos);
    eprintln!(
        "[loaded {name}: {} layers, hidden {}, {:.1} GB checkpoint ({}), {} weights streaming {:.2} GB/token, {} threads, {:.2}s]",
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
    if let Some(out) = &args.save_quantized {
        return save_quantized(&args.model, out, &files, model.quant);
    }
    if !mm_request(&args).is_empty() {
        bail!("{name} takes text only here");
    }
    // The draft (11 GB, mostly experts of which a pass reads a few) is mapped when the target
    // streams, so that its unused experts take no memory.
    let draft_files = match &draft {
        Some(d) => Some((d, safetensors::ModelFiles::open(d, !args.stream)?)),
        None => None,
    };
    let draft = match &draft_files {
        Some((d, f)) => {
            let t = Instant::now();
            let dr = oxinfer::dspark::Dspark::load(d, f, model.cfg.hidden_size, model.cfg.vocab_size, args.quant)?;
            eprintln!(
                "[DSpark draft {}: {} layers, block {}, {:.2} GB read per draft pass, {:.2}s]",
                d.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
                dr.cfg.n_layers,
                dr.cfg.block_size,
                dr.pass_bytes() as f64 / 1e9,
                t.elapsed().as_secs_f64()
            );
            Some(dr)
        }
        None => None,
    };
    let streamer = if args.stream {
        Some(start_stream(&args, |o| oxinfer::stream::attach_deepseek4(&mut model, &files, o))?)
    } else {
        None
    };
    *STREAM.lock().unwrap() = streamer.clone();
    TRACE.store(args.stream_trace, std::sync::atomic::Ordering::Relaxed);
    let r = match &draft {
        Some(d) => run_dspark(&args, &model, d, &tok, &eos, fmt, streamer.as_deref(), Default::default()),
        None => run_text(&args, &model, &tok, &eos, fmt, Default::default()),
    };
    if let Some(s) = &streamer
        && let Some(c) = s.cached_bytes()
    {
        eprintln!("[stream: {:.2} GB of the model files in the page cache at exit]", c as f64 / 1e9);
    }
    *STREAM.lock().unwrap() = None;
    model.set_stream(None);
    drop(streamer);
    r
}

fn run_kimi(mut args: Args) -> Result<()> {
    let t0 = Instant::now();
    if args.mtp || args.draft.is_some() {
        bail!("Kimi K3 has no speculative decoding here (no MTP layer, no DSpark draft)");
    }
    let cfg = oxinfer::kimi::KimiConfig::load(&args.model)?;
    let name = if cfg.dsa_config.is_some() { "Kimi K3.1" } else { "Kimi K3" };
    let eos = config::load_eos_ids(&args.model);
    // A checkpoint larger than most of RAM is streamed with a page-cache budget of 70% of RAM,
    // unless `--no-stream` (or `--mmap`) asks for plain demand paging.
    let size: u64 = oxinfer::safetensors::checkpoint_bytes(&args.model);
    if oxinfer::kimi::too_big(size) && args.no_stream {
        args.mmap = true;
    }
    if oxinfer::kimi::too_big(size) && !args.stream && !args.mmap {
        let budget = oxinfer::kimi::default_stream().cache_budget.unwrap_or(0) as f64 / 1e9;
        args.cache_budget_gb = Some(args.cache_budget_gb.unwrap_or(budget));
        eprintln!(
            "[the {:.0} GB checkpoint does not fit in memory: streaming it (--stream --cache-budget-gb {:.0})]",
            size as f64 / 1e9,
            args.cache_budget_gb.unwrap()
        );
        args.stream = true;
    }
    args.mmap |= args.stream;
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let mut model = oxinfer::kimi::KimiModel::load(cfg, &files, args.quant)?;
    let tok = load_tokenizer(&args.model)?;
    let fmt: &dyn ChatFormat = &chat::KimiChat;
    let eos = chat::with_turn_end(fmt, &tok, eos);
    eprintln!(
        "[loaded {name}: {} layers, hidden {}, {:.1} GB checkpoint ({}), {} weights streaming {:.2} GB/token, {} threads, {:.2}s]",
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
    if let Some(out) = &args.save_quantized {
        return save_quantized(&args.model, out, &files, model.quant);
    }
    if !mm_request(&args).is_empty() {
        bail!("{name} takes text only here (its vision tower is not read)");
    }
    let streamer = if args.stream {
        Some(start_stream(&args, |o| oxinfer::stream::attach_kimi(&mut model, &files, o))?)
    } else {
        None
    };
    *STREAM.lock().unwrap() = streamer.clone();
    TRACE.store(args.stream_trace, std::sync::atomic::Ordering::Relaxed);
    let r = run_text(&args, &model, &tok, &eos, fmt, Default::default());
    if let Some(s) = &streamer
        && let Some(c) = s.cached_bytes()
    {
        eprintln!("[stream: {:.2} GB of the model files in the page cache at exit]", c as f64 / 1e9);
    }
    *STREAM.lock().unwrap() = None;
    model.set_stream(None);
    drop(streamer);
    r
}

/// GLM-5.3 and GLM-5.3-Flash (text; see `oxinfer::glm5`).
fn run_glm5(mut args: Args) -> Result<()> {
    let t0 = Instant::now();
    if args.mtp || args.draft.is_some() {
        bail!("GLM-5.3: speculative decoding (--mtp, --draft) is not supported");
    }
    let cfg = oxinfer::glm5::Glm5Config::load(&args.model)?;
    let eos = config::load_eos_ids(&args.model);
    // A checkpoint larger than most of RAM is streamed with a page-cache budget of 70% of RAM,
    // unless `--no-stream` (or `--mmap`) asks for plain demand paging.
    let size: u64 = oxinfer::safetensors::checkpoint_bytes(&args.model);
    if oxinfer::glm5::too_big(size) && args.no_stream {
        args.mmap = true;
    }
    if oxinfer::glm5::too_big(size) && !args.stream && !args.mmap {
        let budget = oxinfer::glm5::default_stream().cache_budget.unwrap_or(0) as f64 / 1e9;
        args.cache_budget_gb = Some(args.cache_budget_gb.unwrap_or(budget));
        eprintln!(
            "[the {:.0} GB checkpoint does not fit in memory: streaming it (--stream --cache-budget-gb {:.0})]",
            size as f64 / 1e9,
            args.cache_budget_gb.unwrap()
        );
        args.stream = true;
    }
    args.mmap |= args.stream;
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let mut model = oxinfer::glm5::Glm5Model::load(cfg, &files, args.quant)?;
    // Prompt lookup verifies a draft in one pass and rolls back what the model refutes: the
    // recurrent (KDA) layers of GLM-5.3-Flash cannot go back to an earlier position.
    if model.cfg.kda.iter().any(|k| *k) {
        if args.lookup == Some(true) {
            eprintln!("[--lookup is not available for GLM-5.3-Flash (its linear-attention state cannot roll back): plain decoding]");
        }
        *LOOKUP.lock().unwrap() = None;
    }
    let tok = load_tokenizer(&args.model)?;
    let fmt: &dyn ChatFormat = &chat::GlmChat;
    let eos = chat::GlmChat::stop_ids(&tok, eos);
    let c = &model.cfg;
    eprintln!(
        "[loaded {}: {} layers ({} linear attention), hidden {}, {:.1} GB checkpoint ({}), {} weights streaming {:.2} GB/token, {} threads, {:.2}s]",
        if c.hc_mult > 1 { "GLM-5.3-Flash" } else { "GLM-5.3" },
        c.num_hidden_layers,
        c.kda.iter().filter(|k| **k).count(),
        c.hidden_size,
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
    if let Some(out) = &args.save_quantized {
        return save_quantized(&args.model, out, &files, model.quant);
    }
    if !mm_request(&args).is_empty() {
        bail!("GLM-5.3: images and video are not supported yet (the vision tower is not read; text only)");
    }
    let streamer = if args.stream {
        Some(start_stream(&args, |o| oxinfer::stream::attach_glm5(&mut model, &files, o))?)
    } else {
        None
    };
    *STREAM.lock().unwrap() = streamer.clone();
    TRACE.store(args.stream_trace, std::sync::atomic::Ordering::Relaxed);
    let r = run_text(&args, &model, &tok, &eos, fmt, Default::default());
    if let Some(s) = &streamer
        && let Some(c) = s.cached_bytes()
    {
        eprintln!("[stream: {:.2} GB of the model files in the page cache at exit]", c as f64 / 1e9);
    }
    *STREAM.lock().unwrap() = None;
    model.set_stream(None);
    drop(streamer);
    r
}

/// Nemotron-H (`nemotron_h`), and the language model of a NeMo speech LM as a text model (see
/// `oxinfer::nemotron`; `--transcribe` runs its audio side).
fn run_nemotron(args: Args) -> Result<()> {
    let t0 = Instant::now();
    if args.mtp || args.draft.is_some() {
        bail!("Nemotron-H: speculative decoding (--mtp, --draft) is not supported");
    }
    if args.stream {
        bail!("Nemotron-H: weight streaming (--stream) is not supported");
    }
    let cfg = oxinfer::nemotron::NemotronConfig::load(&args.model)?;
    let eos = config::load_eos_ids(&args.model);
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let model = oxinfer::nemotron::NemotronModel::load(cfg, &files, args.quant)?;
    // Prompt lookup verifies a draft in one pass and rolls back what the model refutes: the
    // Mamba-2 state cannot go back to an earlier position.
    if model.cfg.blocks.contains(&oxinfer::nemotron::Block::Mamba) {
        if args.lookup == Some(true) {
            eprintln!("[--lookup is not available for Nemotron-H (its Mamba-2 state cannot roll back): plain decoding]");
        }
        *LOOKUP.lock().unwrap() = None;
    }
    let tok = load_tokenizer(&args.model)?;
    let fmt: &dyn ChatFormat = &chat::NemotronChat;
    let eos = chat::with_turn_end(fmt, &tok, eos);
    let c = &model.cfg;
    eprintln!(
        "[loaded Nemotron-H: {} blocks ({}), hidden {}, {:.1} GB checkpoint ({}), {} weights reading {:.2} GB/token, {} threads, {:.2}s]",
        c.num_hidden_layers(),
        c.describe(),
        c.hidden_size,
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
    if let Some(out) = &args.save_quantized {
        return save_quantized(&args.model, out, &files, model.quant);
    }
    if !mm_request(&args).is_empty() {
        bail!("Nemotron-H: attachments are not supported in a chat (a speech model transcribes audio with --transcribe FILE)");
    }
    run_text(&args, &model, &tok, &eos, fmt, Default::default())
}

/// The DSpark draft of a Qwen or Gemma 4 model: the one given, `auto` for the one beside the
/// model, or the one beside it that is a default (`dspark::default_draft`) unless `--no-draft`
/// or a mode plain decoding serves (tool calls, logits dumps, prompt printing, interventions). A
/// `--draft` that is not a DSpark draft (a Gemma 4 assistant) gives None.
fn pick_dspark(args: &Args) -> Result<Option<PathBuf>> {
    let beside = oxinfer::dspark::draft_for(&args.model);
    let plain_only = args.tools || args.dump_logits.is_some() || args.print_prompt || iv_requested(&args.iv);
    let draft = match &args.draft {
        Some(d) if d.as_os_str() == "auto" => Some(resolve_auto_draft(&args.model, args.cache_dir.as_deref())?),
        Some(d) if oxinfer::dspark::is_draft(d) => Some(d.clone()),
        Some(_) => None,
        None if oxinfer::dspark::BY_DEFAULT && !args.no_draft && !plain_only && !oxinfer::gpu::plain_decode_preferred() => {
            oxinfer::dspark::default_draft_for(&args.model, args.fast_verify)
        }
        None => None,
    };
    if draft.is_none() && args.draft.is_none() && !args.no_draft && !plain_only {
        if let Some(b) = &beside {
            eprintln!(
                "[a DSpark draft is available ({}; not a default for this model): --draft auto uses it]",
                b.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()
            );
        } else if oxinfer::dspark::known_drafts(&args.model).iter().any(|d| d.1)
            && let Some(repo) = oxinfer::dspark::auto_repo(&args.model)
        {
            eprintln!("[a faster DSpark draft is known for this model: --draft auto fetches {repo}]");
        }
    }
    Ok(draft)
}

/// Loads a DSpark draft for a target of hidden size `hidden` and vocabulary `vocab`.
fn load_dspark<'a>(
    path: &std::path::Path,
    files: &'a safetensors::ModelFiles,
    hidden: usize,
    vocab: usize,
    quant: oxinfer::model::Quant,
) -> Result<oxinfer::dspark::Dspark<'a>> {
    let t = Instant::now();
    let dr =
        oxinfer::dspark::Dspark::load(path, files, hidden, vocab, quant).with_context(|| format!("DSpark draft {}", path.display()))?;
    eprintln!(
        "[DSpark draft {}: {} layers, block {} ({} drafts), {:.2} GB read per draft pass, {:.2}s]",
        path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
        dr.cfg.n_layers,
        dr.cfg.block_size,
        dr.cfg.max_draft(),
        dr.pass_bytes() as f64 / 1e9,
        t.elapsed().as_secs_f64()
    );
    Ok(dr)
}

static TRACE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The prompt-lookup drafter of plain decoding (`--lookup`), whose context is the session's
/// tokens.
static LOOKUP: std::sync::Mutex<Option<oxinfer::lookup::Lookup>> = std::sync::Mutex::new(None);

/// The prompt-lookup drafter the options ask for.
fn make_lookup(args: &Args, stop: &[u32]) -> Option<oxinfer::lookup::Lookup> {
    args.lookup.unwrap_or(oxinfer::lookup::BY_DEFAULT).then(|| {
        oxinfer::lookup::Lookup::new(oxinfer::lookup::LookupOptions {
            max_draft: args.lookup_tokens.unwrap_or(oxinfer::lookup::DEFAULT_MAX_DRAFT),
            min_match: args.lookup_min_match.unwrap_or(2),
            memory: args.lookup_memory,
            cost_aware: !args.lookup_fixed,
            fast_verify: args.fast_verify,
            stop: stop.to_vec(),
        })
    })
}

/// The active weight streamer, for the per-token trace of plain decoding.
static STREAM: std::sync::Mutex<Option<std::sync::Arc<oxinfer::stream::Streamer>>> = std::sync::Mutex::new(None);

/// Draft tokens per round by default: more when streaming, where verifying 9 tokens costs about
/// as much as verifying 1.
fn default_draft_tokens(args: &Args) -> usize {
    args.draft_tokens.unwrap_or(if args.stream { 8 } else { 3 })
}

/// Starts weight streaming for `--stream` (`attach` hooks the model up with the options) and
/// prints the plan.
fn start_stream(
    args: &Args,
    attach: impl FnOnce(oxinfer::stream::StreamOptions) -> Result<std::sync::Arc<oxinfer::stream::Streamer>>,
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
    let s = attach(opts)?;
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

/// The streaming statistics before a prefill, when `--stream-trace` prints them.
fn prefill_trace_start() -> Option<(std::sync::Arc<oxinfer::stream::Streamer>, oxinfer::stream::StreamStats)> {
    if !TRACE.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let s = STREAM.lock().unwrap().clone()?;
    let st = s.stats();
    Some((s, st))
}

/// `--stream-trace`: one line of streaming statistics for a prefill of `tokens` tokens.
fn prefill_trace_end(start: Option<(std::sync::Arc<oxinfer::stream::Streamer>, oxinfer::stream::StreamStats)>, tokens: usize, secs: f64) {
    if let Some((s, s0)) = start {
        eprintln!("[prefill: {}]", stream_line(&(s.stats() - s0), tokens, secs));
    }
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
        fast_verify: args.fast_verify,
    };
    eprintln!(
        "[MTP speculative decoding: {} draft tokens{}, {}{}]",
        opts.draft_tokens,
        if opts.cost_aware {
            " max (cost-aware)"
        } else if opts.adaptive {
            " max (adaptive)"
        } else {
            ""
        },
        if args.temperature > 0.0 { "sampled" } else { "greedy" },
        if opts.fast_verify {
            ", fast verification"
        } else {
            ", exact verification"
        }
    );
    let mut sampler =
        Sampler::new(args.temperature, args.top_k, args.top_p, args.seed).with_repetition_penalty(args.repetition_penalty.unwrap_or(1.0));
    let mut dec = MtpDecoder::new(model, mtp, opts);
    dec.lookup = make_lookup(args, eos);
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
        if dec.sess.pos == 0 {
            intervene_cli::install(&args.iv, args.cache_dir.as_deref(), model, tok, &mut dec.sess.iv, &ids, chunk)?;
        }
        dec.sess.iv.gen_from = dec.sess.pos + ids.len() - 1;
        // The attachments belong to the first turn.
        let overrides = oxinfer::mm::bind(model, &mut dec.sess, &ids, &std::mem::take(&mut mm_feats))?;
        let t = Instant::now();
        let ps0 = prefill_trace_start();
        let logits = dec.prefill_mm(&ids, chunk, &overrides);
        let prefill_secs = t.elapsed().as_secs_f64();
        prefill_trace_end(ps0, ids.len(), prefill_secs);
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
        if let Some(lk) = &dec.lookup {
            let l = &lk.stats;
            eprintln!(
                "[lookup (since start): {} rounds verified lookup drafts, accepted {}/{} = {:.0}%, {} vetoed]",
                l.rounds,
                l.accepted,
                l.drafted,
                100.0 * l.accepted as f64 / l.drafted.max(1) as f64,
                l.vetoed
            );
        }
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

/// Whether any intervention flag was given.
fn iv_requested(iv: &intervene_cli::IvArgs) -> bool {
    iv.json.is_some()
        || !iv.neurons.is_empty()
        || !iv.steer.is_empty()
        || !iv.ablate.is_empty()
        || iv.sae.is_some()
        || !iv.sae_features.is_empty()
        || iv.make_direction.is_some()
        || iv.capture.is_some()
}

/// Generation with a DSpark draft (Mistral Large 4, DeepSeek V4, Qwen, Gemma 4), single prompt or
/// chat.
#[allow(clippy::too_many_arguments)]
fn run_dspark<M: oxinfer::dspark::Target<Sess: Clone>>(
    args: &Args,
    model: &M,
    draft: &oxinfer::dspark::Dspark,
    tok: &Tokenizer,
    eos: &[u32],
    fmt: &dyn ChatFormat,
    stream: Option<&oxinfer::stream::Streamer>,
    mm: Mm,
) -> Result<()> {
    use oxinfer::dspark::{DsparkDecoder, DsparkOptions};
    let (mm_text, mut mm_feats) = mm;
    if args.tools || args.dump_logits.is_some() || args.print_prompt || iv_requested(&args.iv) {
        bail!("DSpark decoding does not combine with --tools, --dump-logits, --print-prompt or interventions");
    }
    let encode = |text: &str| -> Result<Vec<u32>> {
        let enc = tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    };
    // Qwen / Gemma 4 drafts: the whole block, cost-aware (a draft or a plain step per round, the
    // verified length) unless fixed or streaming (where the cutoff p_min decides, see the
    // README). Mistral Large 4 / DeepSeek V4: 3 drafts.
    let gqa = matches!(draft.cfg.backbone, oxinfer::dspark::Backbone::Gqa(_));
    let opts = DsparkOptions {
        draft_tokens: args.draft_tokens.unwrap_or(if gqa { usize::MAX } else { 3 }),
        stop: eos.to_vec(),
        cost_aware: if gqa {
            !args.mtp_fixed && args.mtp_cost.unwrap_or(!args.stream)
        } else {
            args.mtp_cost.unwrap_or(false)
        },
        p_min: args
            .draft_p_min
            .unwrap_or(if args.stream { oxinfer::dspark::STREAM_P_MIN } else { 0.0 }),
        device: args.draft_device,
        conf_min: args.draft_conf_min.unwrap_or(0.0),
        fast_verify: args.fast_verify,
    };
    let mut sampler =
        Sampler::new(args.temperature, args.top_k, args.top_p, args.seed).with_repetition_penalty(args.repetition_penalty.unwrap_or(1.0));
    let mut dec = DsparkDecoder::new(model, draft, opts);
    dec.lookup = make_lookup(args, eos);
    eprintln!(
        "[DSpark speculative decoding: {} draft tokens{}, {}{}]",
        dec.opts.draft_tokens,
        if dec.opts.cost_aware { " max (cost-aware)" } else { "" },
        if args.temperature > 0.0 { "sampled" } else { "greedy" },
        if dec.opts.fast_verify {
            ", fast verification"
        } else {
            ", exact verification"
        }
    );
    let chunk = args.prefill_chunk.unwrap_or(256);
    let mut unfed: Option<u32> = None;
    let mut turn = |dec: &mut DsparkDecoder<M>, text: &str| -> Result<()> {
        let mut ids = encode(text)?;
        if ids.is_empty() {
            bail!("empty prompt");
        }
        // A reply cut short left its last token unprocessed: it goes first.
        if let Some(t) = unfed.take() {
            ids.insert(0, t);
        }
        if args.show_tokens {
            eprintln!("[prompt ids ({}): {ids:?}]", ids.len());
        }
        // The attachments belong to the first turn.
        let overrides = oxinfer::mm::bind(model, &mut dec.sess, &ids, &std::mem::take(&mut mm_feats))?;
        let t = Instant::now();
        let ps0 = prefill_trace_start();
        let logits = dec.prefill_mm(&ids, chunk, &overrides);
        let prefill_secs = t.elapsed().as_secs_f64();
        prefill_trace_end(ps0, ids.len(), prefill_secs);
        prof::reset();
        let before = dec.stats.clone();
        let s0 = stream.map(|s| s.stats());
        let mut printer = Printer::new(fmt, args.think, tok);
        let d = generate::decode_rounds(dec, &logits, &mut sampler, eos, args.max_tokens, &mut |out| printer.print(tok, out));
        unfed = d.unfed;
        let out = d.tokens;
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
        let n = out.len().saturating_sub(1);
        report(ids.len(), prefill_secs, n, d.secs);
        eprintln!(
            "[DSpark: {rounds} target forwards for {n} tokens = {:.2} tok/forward; drafts accepted {accepted}/{drafted} = {:.0}%; \
             per-round histogram {:?}; drafting {:.2}s, verification {:.2}s ({:.0} ms per forward)]",
            n as f64 / rounds.max(1) as f64,
            100.0 * accepted as f64 / drafted.max(1) as f64,
            st.per_round,
            st.draft_secs - before.draft_secs,
            st.verify_secs - before.verify_secs,
            1e3 * (st.verify_secs - before.verify_secs) / rounds.max(1) as f64
        );
        if let Some(lk) = &dec.lookup {
            let l = &lk.stats;
            eprintln!(
                "[lookup (since start): {} rounds verified lookup drafts, accepted {}/{} = {:.0}%, {} vetoed]",
                l.rounds,
                l.accepted,
                l.drafted,
                100.0 * l.accepted as f64 / l.drafted.max(1) as f64,
                l.vetoed
            );
        }
        if let (Some(s), Some(s0)) = (stream, s0) {
            let sd = s.stats() - s0;
            eprintln!(
                "[stream: {}; {:.2} GB read per forward]",
                stream_line(&sd, n, d.secs),
                sd.bytes_read as f64 / 1e9 / rounds.max(1) as f64
            );
        }
        if args.profile {
            prof::report(d.secs, out.len());
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

/// Streams a checkpoint larger than this device holds (a 27B on a 12 GB phone) with the device's
/// page-cache budget (see `stream::DeviceMemory`), unless told how to load it.
fn auto_stream(args: &mut Args) {
    if args.stream || args.mmap || args.no_stream {
        return;
    }
    let size = oxinfer::safetensors::checkpoint_bytes(&args.model);
    if let Some(o) = oxinfer::stream::auto_stream(size) {
        let budget = o.cache_budget.unwrap_or(0) as f64 / 1e9;
        args.cache_budget_gb = Some(args.cache_budget_gb.unwrap_or(budget));
        eprintln!(
            "[the {:.1} GB checkpoint does not fit in this device's memory: streaming it (--stream --cache-budget-gb {:.1})]",
            size as f64 / 1e9,
            args.cache_budget_gb.unwrap()
        );
        args.stream = true;
        args.mmap = true;
    }
}

fn run_gemma(mut args: Args) -> Result<()> {
    let t0 = Instant::now();
    auto_stream(&mut args);
    let cfg = config::TextConfig::load(&args.model)?;
    let eos = config::load_eos_ids(&args.model);
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let mut model = Model::load(cfg, &files, args.quant)?;
    let streamer = if args.stream {
        Some(start_stream(&args, |o| oxinfer::stream::attach_gemma(&mut model, &files, o))?)
    } else {
        None
    };
    *STREAM.lock().unwrap() = streamer.clone();
    TRACE.store(args.stream_trace, std::sync::atomic::Ordering::Relaxed);
    let r = run_gemma_loaded(&args, &model, &files, eos, t0);
    // The streamer reads the mapped files: stop it before they are unmapped.
    *STREAM.lock().unwrap() = None;
    model.set_stream(None);
    drop(streamer);
    r
}

fn run_gemma_loaded(args: &Args, model: &Model, files: &safetensors::ModelFiles, eos: Vec<u32>, t0: Instant) -> Result<()> {
    let tok = load_tokenizer(&args.model)?;
    if model.cfg.gemma3 {
        if args.draft.is_some() || !mm_request(args).is_empty() {
            bail!("speculative decoding and attachments are not supported for Gemma 3 yet");
        }
        if let Some(out) = &args.save_quantized {
            return save_quantized(&args.model, out, files, model.quant);
        }
        let eos = chat::with_turn_end(&chat::Gemma3Chat, &tok, eos);
        return run_text(args, model, &tok, &eos, &chat::Gemma3Chat, Default::default());
    }
    let fmt = chat::GemmaChat::for_model(&args.model);
    let eos = chat::with_turn_end(fmt, &tok, eos);
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
        return save_quantized(&args.model, out, files, model.quant);
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

    if let Some(d) = pick_dspark(args)? {
        let f = safetensors::ModelFiles::open(&d, true)?;
        let dr = load_dspark(&d, &f, model.cfg.hidden_size, model.cfg.vocab_size, model.quant)?;
        return run_dspark(
            args,
            model,
            &dr,
            &tok,
            &eos,
            fmt,
            STREAM.lock().unwrap().clone().as_deref(),
            mm_for(args, files, model.cfg.hidden_size)?,
        );
    }
    let draft_files = match &args.draft {
        Some(d) if d.as_os_str() != "auto" => Some(safetensors::ModelFiles::open(d, true)?),
        _ => None,
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

    let mut sampler =
        Sampler::new(args.temperature, args.top_k, args.top_p, args.seed).with_repetition_penalty(args.repetition_penalty.unwrap_or(1.0));
    let mut sess = model.new_session();
    let chunk = args.prefill_chunk.unwrap_or(256);
    if intervene_cli::run_capture(
        &args.iv,
        model,
        &tok,
        fmt,
        args.raw,
        args.think,
        args.prompt.as_deref(),
        args.output.as_deref(),
        chunk,
    )? {
        return Ok(());
    }

    if args.chat {
        let mut first = true;
        let (mm_text, mm_feats) = if !mm_request(args).is_empty() {
            oxinfer::mm::prepare(&mm_request(args), files, model.cfg.hidden_size)?
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
                text.push_str(&chat::GemmaChat.conversation_start_tools(system_prompt(args).as_deref(), args.think, &tool_specs(args)));
                text.push_str(&fmt.user_turn(&format!("{mm_text}{line}"), args.think));
                first = false;
            } else {
                // The previous stop token was sampled but never fed to the model.
                text.push_str(&chat::end_of_model_turn());
                text.push_str(&fmt.user_turn(line, args.think));
            }
            let ids = encode(&text)?;
            if sess.pos == 0 {
                intervene_cli::install(&args.iv, args.cache_dir.as_deref(), model, &tok, &mut sess.iv, &ids, chunk)?;
            }
            if !mm_feats.is_empty() && ["<|image|>", "<|audio|>", "<|video|>"].iter().any(|p| text.contains(p)) {
                overrides = oxinfer::mm::overrides(&ids, &mm_feats, model.cfg.hidden_size)?;
            }
            if args.show_tokens {
                eprintln!("[prompt ids: {ids:?}]");
            }
            let t = Instant::now();
            if let Some(dr) = &drafter {
                let hs = generate::prefill(model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
                let h = model.cfg.hidden_size;
                let hidden_last = hs[hs.len() - h..].to_vec();
                let prefill_secs = t.elapsed().as_secs_f64();
                let (out, st, secs) = generate_speculative(
                    model,
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
            let hs = generate::prefill(model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
            let logits = model.logits(&hs[hs.len() - model.cfg.hidden_size..], 1);
            let prefill_secs = t.elapsed().as_secs_f64();
            prof::reset();
            let (out, n, decode_secs) = generate(
                model,
                &mut sess,
                &tok,
                logits,
                &mut sampler,
                &eos,
                args.max_tokens,
                fmt,
                args.think,
                &ids,
            );
            if args.show_tokens {
                eprintln!("[output ids: {out:?}]");
            }
            report(ids.len(), prefill_secs, n, decode_secs);
            if args.profile {
                prof::report(decode_secs, n);
            }
            continue_with_tools(args, model, &mut sess, &tok, &eos, fmt, &mut sampler, out)?;
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
    let (mm_text, mm_feats) = if !mm_request(args).is_empty() {
        oxinfer::mm::prepare(&mm_request(args), files, model.cfg.hidden_size)?
    } else {
        (String::new(), Vec::new())
    };
    let prompt = format!("{mm_text}{prompt}");
    let text = if args.raw {
        format!("<bos>{prompt}")
    } else {
        let mut t = chat::GemmaChat.conversation_start_tools(system_prompt(args).as_deref(), args.think, &tool_specs(args));
        t.push_str(&fmt.user_turn(&prompt, args.think));
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
    let sink = intervene_cli::install(&args.iv, args.cache_dir.as_deref(), model, &tok, &mut sess.iv, &ids, chunk)?;

    if let Some(dr) = &drafter {
        let t = Instant::now();
        let hs = generate::prefill(model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
        let h = model.cfg.hidden_size;
        let hidden_last = hs[hs.len() - h..].to_vec();
        let prefill_secs = t.elapsed().as_secs_f64();
        prof::reset();
        let (out, st, secs) = generate_speculative(
            model,
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
    let hs = generate::prefill(model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
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
        if let Some(sink) = &sink {
            intervene_cli::print_features(sink, &tok, &ids);
        }
        return Ok(());
    }

    prof::reset();
    let (out, n, decode_secs) = generate(
        model,
        &mut sess,
        &tok,
        logits,
        &mut sampler,
        &eos,
        args.max_tokens,
        fmt,
        args.think,
        &ids,
    );
    if args.show_tokens {
        eprintln!("[output ids: {out:?}]");
    }
    if let Some(sink) = &sink {
        let all: Vec<u32> = ids.iter().chain(&out).copied().collect();
        intervene_cli::print_features(sink, &tok, &all);
    }
    report(ids.len(), prefill_secs, n, decode_secs);
    if args.profile {
        eprintln!("-- decode --");
        prof::report(decode_secs, n);
    }
    continue_with_tools(args, model, &mut sess, &tok, &eos, fmt, &mut sampler, out)?;
    Ok(())
}
