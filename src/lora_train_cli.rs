//! `oxinfer lora-train`: trains a LoRA adapter from logged episodes and their outcomes
//! (`oxinfer::lora_train`, docs/learning.md).

use anyhow::{Context, Result, bail};
use oxinfer::lora_train::{Paths, TrainOptions, parse_kind_weights};
use oxinfer::model::Quant;
use oxinfer::{config, inkling, pool, qwen, safetensors};
use std::path::PathBuf;

pub const USAGE: &str = "usage: oxinfer lora-train --model DIR|org/name --profile DIR [options]\n\
\n\
Trains a LoRA adapter on the episodes of a learning profile and keeps it when it lowers the loss\n\
of held-out episodes (see docs/learning.md). Dense Qwen3 and Llama 3 checkpoints, and Inkling-2\n\
(inkling2, inkling2_loop), on the CPU.\n\
\n\
  --model M             the base model the episodes were logged with (as for oxinfer --model)\n\
  --profile DIR         a profile directory: reads DIR/episodes.jsonl and DIR/outcomes.jsonl, creates\n\
                        or updates DIR/adapter (the adapter it replaces is kept as DIR/adapter.prev)\n\
  --episodes DIR        instead of --profile: the directory holding the two files, with\n\
  --out DIR             the adapter folder to create or update\n\
  --rank N              LoRA rank (default 8)\n\
  --alpha F             LoRA alpha (default 2 x rank)\n\
  --targets LIST        adapted projections, of q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj\n\
                        (default q_proj,v_proj); Inkling-2: of wq_du,wk_dv,wv_dv,wr_du,gate_attn,\n\
                        attn_sink_proj,wo_ud (default wo_ud)\n\
  --grad-window N       Inkling-2: backpropagate through the completion and the N positions before\n\
                        it only (default: the whole sequence)\n\
  --layers L            adapted layers, A-B or a,b,c (default all); the backward stops at the lowest\n\
  --lr F                learning rate (default 2e-4)\n\
  --steps N             optimizer steps (default: --epochs passes over the training episodes)\n\
  --epochs N            passes over the training episodes when --steps is not given (default 3)\n\
  --batch N             episodes per step (default 4)\n\
  --weight-decay F      AdamW weight decay (default 0)\n\
  --clip F              gradient-norm clipping, 0 for none (default 1)\n\
  --neg-clip F          cap on the odds that scale a penalized token's push-down (default 4)\n\
  --kind-weight LIST    weights of the outcome kinds, multiplying the reward: tests=1,user_praise=0.3\n\
                        (default 1 for every kind)\n\
  --holdout F           fraction of the episodes held out for the gate (default 0.2)\n\
  --max-tokens N        leave out episodes of more tokens (default 2048)\n\
  --seed N              seed of the initialisation and the episode order (default 0)\n\
  --fresh               start from a new adapter even when one exists (still compared with it)\n\
  --force               write the adapter whatever the held-out loss says\n\
  --quant bf16|q8|q4    precision the base weights are loaded with (default bf16; as served)\n\
  --mmap                map the weights instead of reading them\n\
  --threads N           CPU threads\n\
  --cache-dir DIR       model cache\n\
\n\
The report is printed as JSON on standard output, the progress on standard error.\n";

pub fn run(args: &[String]) -> Result<()> {
    let mut o = TrainOptions::default();
    let (mut model, mut profile, mut episodes, mut out) = (None, None, None, None);
    let (mut alpha, mut kinds, mut threads, mut cache_dir) = (None, Vec::new(), None, None);
    let (mut quant, mut mmap) = (Quant::Bf16, false);
    let (mut targets_given, mut grad_window) = (false, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().with_context(|| format!("{a} expects a value"));
        match a.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            "--model" | "-m" => model = Some(val()?),
            "--profile" => profile = Some(PathBuf::from(val()?)),
            "--episodes" => episodes = Some(PathBuf::from(val()?)),
            "--out" | "-o" => out = Some(PathBuf::from(val()?)),
            "--rank" => o.rank = val()?.parse()?,
            "--alpha" => alpha = Some(val()?.parse()?),
            "--targets" => {
                o.targets = val()?.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                targets_given = true;
            }
            "--grad-window" => grad_window = Some(val()?.parse()?),
            "--layers" => o.layers = Some(parse_layers(&val()?)?),
            "--lr" => o.lr = val()?.parse()?,
            "--steps" => o.steps = val()?.parse()?,
            "--epochs" => o.epochs = val()?.parse()?,
            "--batch" => o.batch = val()?.parse()?,
            "--weight-decay" => o.weight_decay = val()?.parse()?,
            "--clip" => o.clip = val()?.parse()?,
            "--neg-clip" => o.neg_clip = val()?.parse()?,
            "--kind-weight" => kinds.extend(parse_kind_weights(&val()?)?),
            "--holdout" => o.holdout = val()?.parse()?,
            "--max-tokens" => o.max_tokens = val()?.parse()?,
            "--seed" => o.seed = val()?.parse()?,
            "--fresh" => o.fresh = true,
            "--force" => o.force = true,
            "--quant" => {
                let v = val()?;
                quant = Quant::parse(&v).with_context(|| format!("--quant {v}: expected bf16, q8 or q4"))?;
            }
            "--mmap" => mmap = true,
            "--threads" => threads = Some(val()?.parse()?),
            "--cache-dir" => cache_dir = Some(PathBuf::from(val()?)),
            other => bail!("lora-train: unknown option {other}\n\n{USAGE}"),
        }
    }
    o.alpha = alpha.unwrap_or(2.0 * o.rank as f32);
    if o.neg_clip.is_nan() || o.neg_clip <= 0.0 || !(0.0..1.0).contains(&o.holdout) {
        bail!("--neg-clip must be positive and --holdout in 0..1");
    }
    let spec = model.with_context(|| format!("lora-train: --model is required\n\n{USAGE}"))?;
    let paths = match (profile, episodes, out) {
        (Some(p), None, None) => Paths::profile(&p),
        (None, Some(e), Some(out)) => Paths {
            episodes: e.join("episodes.jsonl"),
            outcomes: e.join("outcomes.jsonl"),
            adapter: out,
        },
        _ => bail!("lora-train: give --profile DIR, or --episodes DIR with --out DIR\n\n{USAGE}"),
    };
    pool::init(threads.unwrap_or_else(pool::default_threads));
    let path = oxinfer::engine::resolve_model_with(&spec, cache_dir.as_deref(), &Default::default())?;
    let model_type = config::model_type(&path).unwrap_or_default();
    if let Ok(cfg) = inkling::Inkling2Config::load(&path) {
        if !targets_given {
            o.targets = vec!["wo_ud".into()];
        }
        let t0 = std::time::Instant::now();
        let files = safetensors::ModelFiles::open(&path, !mmap)?;
        let model = inkling::Inkling2Model::load(cfg, &files, quant)?;
        eprintln!(
            "[lora-train] {spec}: Inkling-2, {} layers of {} ({} applications), loaded in {:.1}s",
            model.cfg.num_hidden_layers,
            model.cfg.hidden_size,
            model.cfg.steps().len(),
            t0.elapsed().as_secs_f64()
        );
        let report = oxinfer::lora_train::run_with(
            &model,
            &|_| Ok(Box::new(inkling::train::Trainer::new(&model, grad_window)?) as Box<dyn oxinfer::lora_train::Passes + '_>),
            &spec,
            &paths,
            &kinds,
            &o,
            &mut |l| eprintln!("[lora-train] {l}"),
        )?;
        if let Some(kb) = peak_rss_kb() {
            eprintln!("[lora-train] peak resident memory {:.2} GB", kb as f64 / 1e6);
        }
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    if grad_window.is_some() {
        bail!("--grad-window is for Inkling-2 models");
    }
    let cfg = config::QwenConfig::load(&path)
        .with_context(|| format!("{spec} ({model_type}): lora-train covers dense Qwen3 and Llama 3 checkpoints"))?;
    let t0 = std::time::Instant::now();
    let files = safetensors::ModelFiles::open(&path, !mmap)?;
    let model = qwen::QwenModel::load(cfg, &files, quant)?;
    eprintln!(
        "[lora-train] {spec}: {} layers of {}, {} weights, loaded in {:.1}s",
        model.cfg.num_hidden_layers,
        model.cfg.hidden_size,
        oxinfer::lora_train::weight_format(&model),
        t0.elapsed().as_secs_f64()
    );
    let report = oxinfer::lora_train::run(&model, &spec, &paths, &kinds, &o, &mut |l| eprintln!("[lora-train] {l}"))?;
    if let Some(kb) = peak_rss_kb() {
        eprintln!("[lora-train] peak resident memory {:.2} GB", kb as f64 / 1e6);
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// `A-B` or `a,b,c`.
fn parse_layers(s: &str) -> Result<Vec<usize>> {
    if let Some((a, b)) = s.split_once('-') {
        let (a, b): (usize, usize) = (a.trim().parse()?, b.trim().parse()?);
        if a > b {
            bail!("--layers {s}: empty range");
        }
        return Ok((a..=b).collect());
    }
    s.split(',').map(|p| Ok(p.trim().parse()?)).collect()
}

/// The process's peak resident set (Linux: `VmHWM` of `/proc/self/status`).
fn peak_rss_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}
