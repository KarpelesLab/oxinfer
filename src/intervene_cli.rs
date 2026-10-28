//! Command-line side of runtime interventions (`oxinfer::intervene`): flags to a `Spec`, the
//! calibration pass, SAE feature printing, and activation capture / direction making.

use anyhow::{Context, Result, bail};
use oxinfer::chat::ChatFormat;
use oxinfer::generate::TextLm;
use oxinfer::intervene::{
    self, Ablate, FeatureEdit, FeatureOp, Host, Point, Pool, SaeInspect, SaeSteer, Spec, State, Steer, TokenFeatures,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

/// The intervention flags.
#[derive(Default)]
pub struct IvArgs {
    pub json: Option<String>,
    pub neurons: Vec<String>,
    pub steer: Vec<String>,
    pub steer_layers: Option<String>,
    pub steer_point: Option<String>,
    pub positions: Option<String>,
    pub ablate: Vec<String>,
    pub ablate_layers: Option<String>,
    pub sae: Option<String>,
    pub sae_layer: Option<usize>,
    pub sae_features: Vec<String>,
    pub sae_inspect: Option<usize>,
    pub calibrate_text: Option<String>,
    pub make_direction: Option<String>,
    pub capture: Option<PathBuf>,
    pub capture_pool: Option<String>,
}

pub const USAGE: &str = "\
runtime interventions (Qwen3.5 / 3.8, Qwen3, Llama 3, Gemma 4 / 3; weights untouched):\n\
\x20 --interventions F    a JSON spec (file, or inline when it starts with '{'; see the README)\n\
\x20 --neuron-edit E      edit an MLP neuron (down_proj input), repeatable: LAYER:NEURON[@EXPERT|@shared],\n\
\x20                      then scale=F | zero | add=F | set=F | clamp=MIN:MAX | dose=F[,unit=resid_median]\n\
\x20                      [,positions=all|prompt|generated]; e.g. 14:8949,dose=1,unit=resid_median\n\
\x20 --calibrate-text T   measure the resid_median unit on T (default: the prompt, unedited model)\n\
\x20 --steer FILE[:A]     add A x the file's vectors to the residual (safetensors layers.N, llama.cpp /\n\
\x20                      repeng GGUF direction.N, or torch .pt; repeatable; A default 1)\n\
\x20 --steer-layers L     only these layers of the --steer files (A-B or a,b,c)\n\
\x20 --steer-point P      post (layer output, default) | mid (after attention) | pre (layer input)\n\
\x20 --positions P        all (default) | prompt | generated: where --steer / --ablate / --sae-feature apply\n\
\x20 --ablate FILE[:L]    project the direction (the file's layer L vector) out of the residual stream\n\
\x20                      everywhere it is written (runtime abliteration; repeatable)\n\
\x20 --ablate-layers L    only in these layers (A-B or a,b,c)\n\
\x20 --sae S              a sparse autoencoder: Qwen-Scope layerN.sae.pt (path or org/repo/file), or a\n\
\x20                      SAELens directory; --sae-layer N overrides its layer\n\
\x20 --sae-feature F      ID:clamp=V | ID:scale=F | ID:add=V | ID:zero (repeatable; error term kept)\n\
\x20 --sae-inspect K      print the K most active SAE features of every position\n\
\x20 --make-direction P,N difference of means of the layer outputs of prompt files P and N (one prompt per\n\
\x20                      line, or .json / .jsonl / .csv), chat-wrapped unless --raw; -o FILE (safetensors)\n\
\x20 --capture-activations FILE  write the prompt's pooled layer outputs (layers.N) and exit\n\
\x20 --capture-pool P     last (default) | mean (positions 1..)\n";

impl IvArgs {
    /// Takes `flag` (with its value from `val`) if it is an intervention flag.
    pub fn parse(&mut self, flag: &str, val: &mut dyn FnMut() -> Result<String>) -> Result<bool> {
        match flag {
            "--interventions" => self.json = Some(val()?),
            "--neuron-edit" => self.neurons.push(val()?),
            "--steer" => self.steer.push(val()?),
            "--steer-layers" => self.steer_layers = Some(val()?),
            "--steer-point" => self.steer_point = Some(val()?),
            "--positions" => self.positions = Some(val()?),
            "--ablate" => self.ablate.push(val()?),
            "--ablate-layers" => self.ablate_layers = Some(val()?),
            "--sae" => self.sae = Some(val()?),
            "--sae-layer" => self.sae_layer = Some(val()?.parse()?),
            "--sae-feature" => self.sae_features.push(val()?),
            "--sae-inspect" => self.sae_inspect = Some(val()?.parse()?),
            "--calibrate-text" => self.calibrate_text = Some(val()?),
            "--make-direction" => self.make_direction = Some(val()?),
            "--capture-activations" => self.capture = Some(PathBuf::from(val()?)),
            "--capture-pool" => self.capture_pool = Some(val()?),
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The spec the flags describe (None without any).
    pub fn spec(&self) -> Result<Option<Spec>> {
        let mut spec = match &self.json {
            Some(j) if j.trim_start().starts_with('{') => Spec::from_json(j)?,
            Some(f) => Spec::from_json(&std::fs::read_to_string(f).with_context(|| f.clone())?)?,
            None => Spec::default(),
        };
        let positions = self
            .positions
            .as_deref()
            .map(intervene::parse_positions)
            .transpose()?
            .unwrap_or_default();
        for n in &self.neurons {
            spec.neurons.push(Spec::parse_neuron(n)?);
        }
        let point = match self.steer_point.as_deref() {
            None | Some("post") => Point::Post,
            Some("mid") => Point::Mid,
            Some("pre") => Point::Pre,
            Some(p) => bail!("--steer-point: post, mid or pre, got {p:?}"),
        };
        for s in &self.steer {
            let (file, alpha) = split_num(s);
            spec.steer.push(Steer {
                file: Some(file.to_string()),
                alpha: alpha.unwrap_or(1.0),
                layers: self.steer_layers.as_deref().map(parse_layers).transpose()?,
                point,
                positions,
                ..Default::default()
            });
        }
        for a in &self.ablate {
            let (file, layer) = split_num(a);
            spec.ablate.push(Ablate {
                file: Some(file.to_string()),
                direction_layer: layer.map(|l| l as usize),
                layers: self.ablate_layers.as_deref().map(parse_layers).transpose()?,
                positions,
                ..Default::default()
            });
        }
        if !self.sae_features.is_empty() || self.sae_inspect.is_some() {
            let sae = self.sae.clone().context("--sae-feature / --sae-inspect need --sae")?;
            if !self.sae_features.is_empty() {
                let mut features = Vec::new();
                for f in &self.sae_features {
                    let (id, op) = f.split_once(':').unwrap_or((f, "zero"));
                    let (op, v) = op.split_once('=').unwrap_or((op, "0"));
                    let op = match op {
                        "clamp" | "set" => FeatureOp::Clamp,
                        "scale" => FeatureOp::Scale,
                        "add" => FeatureOp::Add,
                        "zero" => FeatureOp::Zero,
                        o => bail!("--sae-feature {f:?}: unknown op {o:?} (clamp, scale, add, zero)"),
                    };
                    features.push(FeatureEdit {
                        id: id.trim().parse().with_context(|| format!("--sae-feature {f:?}"))?,
                        op,
                        value: v.trim().parse().with_context(|| format!("--sae-feature {f:?}"))?,
                    });
                }
                spec.sae.push(SaeSteer {
                    sae: sae.clone(),
                    layer: self.sae_layer,
                    features,
                    positions,
                });
            }
            if let Some(k) = self.sae_inspect {
                spec.sae_inspect.push(SaeInspect {
                    sae,
                    layer: self.sae_layer,
                    top_k: k,
                });
            }
        }
        Ok((!spec.is_empty()).then_some(spec))
    }
}

/// `FILE:NUMBER` -> (FILE, NUMBER), or the whole string when the tail is not a number.
fn split_num(s: &str) -> (&str, Option<f32>) {
    match s.rsplit_once(':') {
        Some((f, n)) if !f.is_empty() => match n.parse::<f32>() {
            Ok(v) => (f, Some(v)),
            Err(_) => (s, None),
        },
        _ => (s, None),
    }
}

/// `A-B` or `a,b,c`.
fn parse_layers(s: &str) -> Result<Vec<usize>> {
    let mut v = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        match part.split_once('-') {
            Some((a, b)) => v.extend(a.trim().parse::<usize>()?..=b.trim().parse::<usize>()?),
            None => v.push(part.parse()?),
        }
    }
    Ok(v)
}

/// Compiles the flags' spec for `model` into the session state `st` (measuring the calibration
/// on `--calibrate-text` or `prompt` when needed). Returns the SAE feature sink when inspecting.
pub fn install<M: TextLm + Host>(
    iv: &IvArgs,
    cache_dir: Option<&std::path::Path>,
    model: &M,
    tok: &Tokenizer,
    st: &mut State,
    prompt: &[u32],
    chunk: usize,
) -> Result<Option<Arc<Mutex<Vec<TokenFeatures>>>>> {
    let Some(spec) = iv.spec()? else { return Ok(None) };
    let cal = if spec.needs_calibration() {
        let ids = match &iv.calibrate_text {
            Some(t) => tok
                .encode(t.as_str(), false)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?
                .get_ids()
                .to_vec(),
            None => prompt.to_vec(),
        };
        let c = intervene::calibrate(model, &ids, chunk)?;
        let mut layers: Vec<usize> = spec.neurons.iter().map(|n| n.layer).collect();
        layers.sort_unstable();
        layers.dedup();
        let shown: Vec<String> = layers.iter().map(|&l| format!("L{l} {:.2}", c.resid_median[l])).collect();
        eprintln!(
            "[calibration: median |layer output| over positions 1..{} of the {}: {}]",
            ids.len() - 1,
            if iv.calibrate_text.is_some() {
                "calibration text"
            } else {
                "prompt"
            },
            shown.join(", ")
        );
        Some(c)
    } else {
        None
    };
    let cache = cache_dir.map(|p| p.to_path_buf());
    let resolve = move |s: &str| oxinfer::engine::resolve_file(s, cache.as_deref());
    let ctx = intervene::Ctx {
        resolve: &resolve,
        calibration: cal.as_ref(),
        saes: None,
    };
    let c = intervene::Compiled::new(&spec, model, &ctx)?;
    let gpu = c.gpu_data().is_some();
    eprintln!(
        "[interventions: {} neuron edits, {} steering, {} ablations, {} SAE steering, {} SAE inspection; {}]",
        spec.neurons.len(),
        spec.steer.len(),
        spec.ablate.len(),
        spec.sae.len(),
        spec.sae_inspect.len(),
        if gpu {
            "GPU decode keeps running on the GPU"
        } else {
            "decode steps run on the CPU (TopK SAE features / inspection)"
        }
    );
    let sink = c.inspects().then(|| Arc::new(Mutex::new(Vec::new())));
    st.iv = Some(Arc::new(c));
    st.features = sink.clone();
    Ok(sink)
}

/// Prints the recorded SAE features, one line per position (`ids`: the tokens from position 0).
pub fn print_features(sink: &Arc<Mutex<Vec<TokenFeatures>>>, tok: &Tokenizer, ids: &[u32]) {
    let v = sink.lock().unwrap();
    for f in v.iter() {
        let t = ids
            .get(f.pos)
            .and_then(|&i| tok.decode(&[i], false).ok())
            .unwrap_or_default()
            .replace('\n', "\\n");
        let top: Vec<String> = f.top.iter().map(|(i, a)| format!("{i}:{a:.2}")).collect();
        eprintln!("[sae L{} pos {:>4} {:<14} {}]", f.layer, f.pos, format!("{t:?}"), top.join(" "));
    }
}

/// `--make-direction` / `--capture-activations`: runs and returns true (the caller exits).
#[allow(clippy::too_many_arguments)]
pub fn run_capture<M: TextLm>(
    iv: &IvArgs,
    model: &M,
    tok: &Tokenizer,
    fmt: &dyn ChatFormat,
    raw: bool,
    think: bool,
    prompt: Option<&str>,
    output: Option<&std::path::Path>,
    chunk: usize,
) -> Result<bool> {
    if iv.make_direction.is_none() && iv.capture.is_none() {
        return Ok(false);
    }
    let pool = match iv.capture_pool.as_deref() {
        None | Some("last") => Pool::Last,
        Some("mean") => Pool::Mean,
        Some(p) => bail!("--capture-pool: last or mean, got {p:?}"),
    };
    let wrap = |p: &str| if raw { p.to_string() } else { fmt.first_turn(None, think, &[], p) };
    let encode = |ps: &[String]| -> Result<Vec<Vec<u32>>> {
        ps.iter()
            .map(|p| {
                Ok(tok
                    .encode(wrap(p), false)
                    .map_err(|e| anyhow::anyhow!("encode: {e}"))?
                    .get_ids()
                    .to_vec())
            })
            .collect()
    };
    if let Some(md) = &iv.make_direction {
        let (a, b) = md.split_once(',').context("--make-direction POSITIVE,NEGATIVE")?;
        let (pa, pb) = (intervene::read_prompts(a.as_ref())?, intervene::read_prompts(b.as_ref())?);
        let t = std::time::Instant::now();
        let ca = intervene::capture_layers(model, &encode(&pa)?, pool, chunk)?;
        let cb = intervene::capture_layers(model, &encode(&pb)?, pool, chunk)?;
        let d = intervene::diff_of_means(&ca, &cb);
        let out = output
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("direction.safetensors"));
        intervene::save_vectors(
            &out,
            &d,
            &[
                ("kind", "difference of means (positive - negative)".into()),
                ("pool", format!("{pool:?}").to_lowercase()),
                ("positive", format!("{a} ({} prompts)", pa.len())),
                ("negative", format!("{b} ({} prompts)", pb.len())),
                ("chat", (!raw).to_string()),
            ],
        )?;
        let norms: Vec<String> = d
            .iter()
            .map(|(l, v)| format!("{l}:{:.1}", v.iter().map(|x| x * x).sum::<f32>().sqrt()))
            .collect();
        eprintln!(
            "[direction: {} + {} prompts in {:.1}s, written to {}; |direction| per layer {}]",
            pa.len(),
            pb.len(),
            t.elapsed().as_secs_f64(),
            out.display(),
            norms.join(" ")
        );
        return Ok(true);
    }
    let out = iv.capture.as_ref().unwrap();
    let text = prompt.context("--capture-activations needs --prompt")?;
    let caps = intervene::capture_layers(model, &encode(&[text.to_string()])?, pool, chunk)?;
    intervene::save_vectors(out, &caps[0], &[("pool", format!("{pool:?}").to_lowercase())])?;
    eprintln!("[captured {} layer outputs into {}]", caps[0].len(), out.display());
    Ok(true)
}
