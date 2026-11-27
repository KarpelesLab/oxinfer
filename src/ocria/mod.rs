//! OCRia-nano-1 (`llamaindex/OCRia-nano-1`): LiteParse's OCR models, run natively on oxinfer's
//! own ONNX interpreter ([`crate::onnx`]): a text-line detector, a script router (Latin, CJK,
//! right-to-left, world) that also tells upside-down pages, and four CTC line recognizers.
//!
//! The pipeline is LiteParse's (`crates/liteparse/src/ocr/ocria`, Apache-2.0), ported: the
//! detector at a long side of 1600 (a second, larger pass on pages of small print), connected
//! components grown into line quads and merged along their lines, perspective crops with
//! OpenCV's fixed-point arithmetic, column-gutter splitting, the sideways-page retry, the router's
//! page-level vote, width-bucketed recognizer batches (a line's reading depends on the padded
//! width it is run at, so the batching is the reference's), CTC decoding, right-to-left readings
//! reordered to logical order, the low-confidence script rescue, lookalike folding, the junk
//! filter and faint short tokens. [`OcrRunOptions::plain`] turns every page-level heuristic off
//! (the models' own outputs, as the reference script `ref_ocr.py` gives them).
//!
//! The models run on the CPU, or on the GPU (Metal) for large inputs or when asked
//! ([`OcrOptions::device`]).
//!
//! `rec_latin` is quantized (QDQ, int8 weights, uint8 activations); it runs with ONNX Runtime's
//! int8 arithmetic ([`crate::onnx`]), so it reads what LiteParse's ONNX Runtime reads.

mod det;
mod img;
mod page;
mod rec;

use anyhow::{Context, Result, bail};
use det::Detector;
pub use img::Img;
use rec::{GROUPS, Recognizer, Router};
use serde::Serialize;
use std::path::{Path, PathBuf};

/// The repository of the models.
pub const REPO: &str = "llamaindex/OCRia-nano-1";

/// A line quad: [top-left, top-right, bottom-right, bottom-left] along the text.
pub type Quad = [[f32; 2]; 4];

/// The script groups, in the router's order.
pub const SCRIPTS: [&str; 4] = GROUPS;

/// A faint candidate is kept only at this confidence (LiteParse: rule fragments and specks read
/// as "|", "l" or a letter pair at 0.5-0.7; lone digits at 0.97+).
const FAINT_CONF: f32 = 0.9;
/// Detector thresholds for a model without `det.meta.json`.
const DET_THRESH: f32 = 0.5;
const DET_MERGE: f32 = 0.6;
const DET_ADAPTIVE: det::Adaptive = det::Adaptive {
    min_h: 14.0,
    target_h: 22.0,
    max_side: 3200,
    quantile: 0.25,
    min_cover: 0.003,
};
const DET_FAINT: det::Faint = det::Faint { thresh: 0.1, max: 24 };

/// The smallest inputs (values) `device=auto` runs on an active GPU, for the detector (about a
/// 700 x 700 RGB image: below it the GPU's fixed cost per run outweighs it), the router and the
/// recognizers (eight lines at 512 pixels).
pub const AUTO_MIN: [usize; 3] = [1_500_000, 131_072, 131_072];

/// The files a model folder holds for the scripts in `scripts`.
pub fn files_for(scripts: &[bool; 4]) -> Vec<String> {
    let mut v: Vec<String> = ["det.onnx", "det.meta.json", "router.onnx", "config.json"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    for (g, _) in GROUPS.iter().zip(scripts).filter(|(_, on)| **on) {
        v.push(format!("rec_{g}.onnx"));
        v.push(format!("rec_{g}.charset.json"));
        v.push(format!("rec_{g}.meta.json"));
    }
    v
}

/// Loader options. Keys for [`OcrOptions::set`]: `scripts` (comma-separated: latin, cjk, rtl,
/// world, or all), `det_long_side`, `threads`, `cache_dir`, `device`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct OcrOptions {
    /// Which recognizers to load (`SCRIPTS` order). Lines routed to a missing one are read with
    /// the Latin recognizer (or the first one loaded).
    pub scripts: [bool; 4],
    /// The detector's input long side (default 1600).
    pub det_long_side: usize,
    /// CPU worker threads (process-wide).
    pub threads: Option<usize>,
    /// Where the repo id is downloaded.
    pub cache_dir: Option<PathBuf>,
    /// Where the models run: `cpu`, a GPU (`metal`: the detector, the router and the
    /// recognizers on it, the page heuristics on the CPU), or `auto` (the GPU for the inputs it
    /// runs faster than the CPU, see [`AUTO_MIN`]).
    pub device: crate::platform::DevicePref,
}

impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            scripts: [true; 4],
            det_long_side: 1600,
            threads: None,
            cache_dir: None,
            device: crate::platform::DevicePref::Auto,
        }
    }
}

/// Parses `latin,cjk` (or `all`) into flags in [`SCRIPTS`] order.
pub fn parse_scripts(v: &str) -> Result<[bool; 4]> {
    let mut s = [false; 4];
    for part in v.split([',', ' ', '+']).map(str::trim).filter(|p| !p.is_empty()) {
        if part.eq_ignore_ascii_case("all") {
            return Ok([true; 4]);
        }
        let i = GROUPS
            .iter()
            .position(|g| g.eq_ignore_ascii_case(part))
            .with_context(|| format!("unknown script group {part:?} (latin, cjk, rtl, world or all)"))?;
        s[i] = true;
    }
    if !s.iter().any(|&b| b) {
        bail!("scripts: at least one of latin, cjk, rtl, world");
    }
    Ok(s)
}

fn parse_bool(key: &str, v: &str) -> Result<bool> {
    match v.to_ascii_lowercase().as_str() {
        "" | "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("{key}: expected a boolean, got {v:?}"),
    }
}

impl OcrOptions {
    pub const KEYS: &'static [&'static str] = &["scripts", "det_long_side", "threads", "cache_dir", "device"];

    /// Sets an option by key (see [`OcrOptions::KEYS`]).
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "scripts" => self.scripts = parse_scripts(value)?,
            "det_long_side" | "long_side" => self.det_long_side = parse_side(value)?,
            "threads" => self.threads = Some(value.parse().with_context(|| format!("threads: {value:?}"))?),
            "cache_dir" => self.cache_dir = Some(PathBuf::from(value)),
            "device" => {
                self.device = crate::platform::DevicePref::parse(value).with_context(|| format!("device: unknown device {value:?}"))?
            }
            _ => bail!("unknown OCR option {key:?} (known: {})", Self::KEYS.join(", ")),
        }
        Ok(())
    }
}

fn parse_side(v: &str) -> Result<usize> {
    let n: usize = v.parse().with_context(|| format!("long side: {v:?}"))?;
    if !(32..=16384).contains(&n) {
        bail!("long side {n}: expected 32 to 16384");
    }
    Ok(n)
}

/// Per-page options: LiteParse's pipeline switches. The defaults are its benchmarked
/// configuration; [`OcrRunOptions::plain`] turns the page-level heuristics off.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OcrRunOptions {
    /// The detector's input long side (default: the loader's, 1600).
    pub det_long_side: Option<usize>,
    /// Detector probability threshold (default `det.meta.json`'s, else 0.5).
    pub det_thresh: Option<f32>,
    /// Collinear box merge gap in line heights (default `det.meta.json`'s, else 0.6; 0 = off).
    pub det_merge: Option<f32>,
    /// A second, larger detection pass on pages of small print.
    pub adaptive: bool,
    /// Cut boxes running across a column gutter.
    pub gutter: bool,
    /// Retry pages that look sideways a quarter turn round, and read upside-down pages turned.
    pub orient: bool,
    /// Re-read low-confidence Latin pages / lines with the other recognizers, and lines routed
    /// away from the page's script with the page's recognizer.
    pub rescue: bool,
    /// Drop readings with no letter or digit, or one or two unsure characters.
    pub junk: bool,
    /// Faint short tokens next to detected lines, kept at confidence 0.9+.
    pub faint: bool,
    /// Fold other scripts' lookalikes and digits in Latin lines (LiteParse's `clean_latin`).
    pub clean: bool,
    /// Reorder right-to-left readings (written in visual order) to logical order.
    pub bidi: bool,
    /// Order the lines (and the text) by columns and rows (XY cut); off: the detector's
    /// row-band order.
    pub layout: bool,
    /// Turn encoded photos upright per their EXIF orientation (JPEG APP1, PNG `eXIf`) before
    /// reading them, as LiteParse does (on in [`Self::plain`] too: it is the image, not a
    /// heuristic). Pixels given as such are read as they are.
    pub exif: bool,
}

impl Default for OcrRunOptions {
    fn default() -> Self {
        Self {
            det_long_side: None,
            det_thresh: None,
            det_merge: None,
            adaptive: true,
            gutter: true,
            orient: true,
            rescue: true,
            junk: true,
            faint: true,
            clean: true,
            bidi: true,
            layout: true,
            exif: true,
        }
    }
}

impl OcrRunOptions {
    pub const KEYS: &'static [&'static str] = &[
        "det_long_side",
        "det_thresh",
        "det_merge",
        "adaptive",
        "gutter",
        "orient",
        "rescue",
        "junk",
        "faint",
        "clean",
        "bidi",
        "layout",
        "exif",
        "plain",
    ];

    /// The models' own outputs: one detector pass with the merge, the router (with its upside-down
    /// vote) and the recognizers, no page-level heuristics, no reordering.
    pub fn plain() -> Self {
        Self {
            adaptive: false,
            gutter: false,
            orient: false,
            rescue: false,
            junk: false,
            faint: false,
            clean: false,
            bidi: false,
            layout: false,
            ..Self::default()
        }
    }

    /// Sets an option by key (see [`OcrRunOptions::KEYS`]; `plain=1` resets to [`Self::plain`]).
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let b = |v: &str| parse_bool(key, v);
        match key {
            "det_long_side" | "long_side" => self.det_long_side = Some(parse_side(value)?),
            "det_thresh" | "thresh" => self.det_thresh = Some(value.parse().with_context(|| format!("{key}: {value:?}"))?),
            "det_merge" | "merge" => self.det_merge = Some(value.parse().with_context(|| format!("{key}: {value:?}"))?),
            "adaptive" => self.adaptive = b(value)?,
            "gutter" => self.gutter = b(value)?,
            "orient" => self.orient = b(value)?,
            "rescue" => self.rescue = b(value)?,
            "junk" => self.junk = b(value)?,
            "faint" => self.faint = b(value)?,
            "clean" => self.clean = b(value)?,
            "bidi" => self.bidi = b(value)?,
            "layout" => self.layout = b(value)?,
            "exif" => self.exif = b(value)?,
            "plain" => {
                if b(value)? {
                    *self = Self::plain();
                }
            }
            _ => bail!("unknown OCR option {key:?} (known: {})", Self::KEYS.join(", ")),
        }
        Ok(())
    }
}

/// A word of a line: its text and box (x0, y0, x1, y1) in image pixels.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct OcrWord {
    pub text: String,
    pub bbox: [f32; 4],
}

/// A recognized line.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct OcrLine {
    pub text: String,
    /// The mean confidence of the characters read.
    pub confidence: f32,
    /// The line's corners in image pixels, in reading order: top-left, top-right,
    /// bottom-right, bottom-left of the text.
    #[serde(rename = "box")]
    pub quad: Quad,
    /// The extent (x0, y0, x1, y1) of `quad`, truncated to whole pixels.
    pub bbox: [f32; 4],
    /// The recognizer group that read it (latin, cjk, rtl, world).
    pub script: String,
    /// Word boxes (upright left-to-right lines with one span per word; empty otherwise).
    pub words: Vec<OcrWord>,
    /// A faint short token found next to the detected lines.
    pub faint: bool,
}

/// One page's result.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct OcrPage {
    pub width: usize,
    pub height: usize,
    pub lines: Vec<OcrLine>,
    /// The lines' text, one per line, in reading order.
    pub text: String,
    /// The page's dominant script group.
    pub script: String,
    /// Read turned 180 degrees.
    pub upside_down: bool,
    /// Read a quarter turn round.
    pub sideways: bool,
    /// The EXIF orientation the image was turned upright by (2 to 8; None: none applied).
    pub exif_orientation: Option<u16>,
    /// Wall time of the page.
    pub seconds: f64,
}

/// What was loaded.
#[derive(Clone, Debug, Serialize)]
pub struct OcriaInfo {
    pub name: String,
    pub path: PathBuf,
    /// Loaded recognizers.
    pub scripts: Vec<String>,
    pub det_thresh: f32,
    pub det_merge: f32,
    pub det_long_side: usize,
    /// Parameters of the loaded models (constants, folded).
    pub params: usize,
    pub device: String,
    pub load_secs: f64,
}

/// The models of OCRia-nano-1.
pub struct Ocria {
    det: crate::onnx::Model,
    router: Router,
    recs: [Option<Recognizer>; 4],
    thresh: f32,
    merge: f32,
    long_side: usize,
    info: OcriaInfo,
}

/// Whether `path` is an OCRia model folder (`det.onnx` and `router.onnx`).
pub fn is_ocria(path: &Path) -> bool {
    path.is_dir() && path.join("det.onnx").is_file() && path.join("router.onnx").is_file()
}

/// Whether `spec` names the OCRia repository.
pub fn is_repo_id(spec: &str) -> bool {
    spec.trim_end_matches('/').eq_ignore_ascii_case(REPO)
}

/// The model folder of `spec`: a folder as is, the repo id fetched into the cache (only the
/// files of the scripts asked for).
pub fn resolve(spec: &str, opts: &OcrOptions) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.is_dir() {
        return Ok(p.to_path_buf());
    }
    if p.is_file() {
        bail!("{spec}: give the folder holding det.onnx, router.onnx and the rec_*.onnx files");
    }
    #[cfg(feature = "hf")]
    {
        let files = files_for(&opts.scripts);
        let names: Vec<&str> = files.iter().map(String::as_str).collect();
        crate::hf::resolve_files(spec, &names, opts.cache_dir.as_deref())
    }
    #[cfg(not(feature = "hf"))]
    {
        let _ = opts;
        bail!("{spec}: not a local folder (built without the `hf` feature, so repo ids cannot be fetched)")
    }
}

fn load_json(path: &Path) -> Result<serde_json::Value> {
    let b = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&b).with_context(|| path.display().to_string())
}

fn input(shape: [usize; 4], x: Vec<f32>) -> Vec<(&'static str, crate::onnx::Tensor)> {
    vec![("x", crate::onnx::Tensor::new(shape.to_vec(), x))]
}

fn output(model: &crate::onnx::Model, out: &mut [crate::onnx::Tensor], name: &str) -> Result<crate::onnx::Tensor> {
    let i = model
        .output_index(name)
        .with_context(|| format!("the model has no output {name}"))?;
    Ok(std::mem::take(&mut out[i]))
}

/// The detector's probability map: (shape, values).
fn infer(model: &crate::onnx::Model, shape: [usize; 4], x: Vec<f32>, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let mut out = model.run(input(shape, x))?;
    let t = output(model, &mut out, name)?;
    Ok((t.shape, t.data))
}

/// A recognizer's step ids: (shape, ids).
fn infer_ids(model: &crate::onnx::Model, shape: [usize; 4], x: Vec<f32>) -> Result<(Vec<usize>, Vec<i64>)> {
    let mut out = model.run(input(shape, x))?;
    let t = output(model, &mut out, "ids")?;
    let ids = t.ints();
    Ok((t.shape, ids))
}

/// A recognizer's step ids and confidences: (shape, ids, conf).
fn infer_rec(model: &crate::onnx::Model, shape: [usize; 4], x: Vec<f32>) -> Result<(Vec<usize>, Vec<i64>, Vec<f32>)> {
    let mut out = model.run(input(shape, x))?;
    let ids = output(model, &mut out, "ids")?;
    let conf = output(model, &mut out, "conf")?;
    if ids.shape.len() != 2 || conf.shape != ids.shape {
        bail!("recognizer outputs of shapes {:?} and {:?}", ids.shape, conf.shape);
    }
    let v = ids.ints();
    Ok((ids.shape, v, conf.data))
}

/// The router's group and flip logits.
fn infer_router(model: &crate::onnx::Model, shape: [usize; 4], x: Vec<f32>) -> Result<(Vec<f32>, Vec<f32>)> {
    let n = shape[0];
    let mut out = model.run(input(shape, x))?;
    let g = output(model, &mut out, "group")?;
    let f = output(model, &mut out, "flip")?;
    if g.data.len() != n * 4 || f.data.len() != n * 2 {
        bail!("router outputs of shapes {:?} and {:?}", g.shape, f.shape);
    }
    Ok((g.data, f.data))
}

/// `(text, confidence, word spans)` per box, whether the page was read upside down, the page's
/// script group, and each box's group.
type PageReads = (Vec<rec::Read>, bool, usize, Vec<usize>);

impl Ocria {
    /// Loads the models: a folder holding them, or the repo id `llamaindex/OCRia-nano-1`
    /// (fetched with the `hf` feature: only the files of `opts.scripts`).
    pub fn load(spec: &str, opts: &OcrOptions) -> Result<Self> {
        let dir = resolve(spec, opts)?;
        Self::load_dir(&dir, opts)
    }

    /// Loads the models of a folder.
    pub fn load_dir(dir: &Path, opts: &OcrOptions) -> Result<Self> {
        let t0 = std::time::Instant::now();
        if let Some(n) = opts.threads {
            let mut o = crate::pool::options();
            o.threads = n;
            crate::pool::configure_with(o);
        }
        let load = |f: &str| crate::onnx::Model::load(&dir.join(f));
        let det = load("det.onnx")?;
        let router = Router::new(load("router.onnx")?);
        // Thresholds the detector was tuned with ship next to it.
        let meta = load_json(&dir.join("det.meta.json")).ok();
        let from_meta = |k: &str| meta.as_ref().and_then(|m| m.get(k)).and_then(|v| v.as_f64()).map(|v| v as f32);
        let mut recs: [Option<Recognizer>; 4] = [None, None, None, None];
        for (i, g) in GROUPS.iter().enumerate() {
            if !opts.scripts[i] {
                continue;
            }
            let model = dir.join(format!("rec_{g}.onnx"));
            if !model.is_file() {
                bail!("{}: no rec_{g}.onnx (load fewer scripts with the `scripts` option)", dir.display());
            }
            let chars: Vec<String> = serde_json::from_value(load_json(&dir.join(format!("rec_{g}.charset.json")))?)
                .with_context(|| format!("rec_{g}.charset.json: an array of strings"))?;
            let visual_rtl = load_json(&dir.join(format!("rec_{g}.meta.json")))
                .ok()
                .and_then(|v| v.get("visual_rtl").and_then(|x| x.as_bool()))
                .unwrap_or(false);
            recs[i] = Some(Recognizer::new(load(&format!("rec_{g}.onnx"))?, chars, visual_rtl).with_context(|| format!("rec_{g}"))?);
        }
        if recs.iter().all(Option::is_none) {
            bail!("no recognizer to load");
        }
        let params = det.params() + router.model.params() + recs.iter().flatten().map(|r| r.model.params()).sum::<usize>();
        // The models on the GPU: all of them asked for by name, those it runs faster with `auto`.
        let gpu = crate::gpu::ensure(opts.device)?;
        let (on, min) = match opts.device {
            crate::platform::DevicePref::Cpu => (false, [0; 3]),
            crate::platform::DevicePref::Gpu(_) => (true, [0; 3]),
            crate::platform::DevicePref::Auto => (true, AUTO_MIN),
        };
        det.set_device(on);
        det.set_device_min(min[0]);
        router.model.set_device(on);
        router.model.set_device_min(min[1]);
        for r in recs.iter().flatten() {
            r.model.set_device(on);
            r.model.set_device_min(min[2]);
        }
        let actual = [
            det.on_device(),
            router.model.on_device(),
            recs.iter().flatten().any(|r| r.model.on_device()),
        ];
        let device = match gpu.filter(|_| actual.iter().any(|&b| b)) {
            Some(b) => {
                let parts: Vec<&str> = ["detector", "router", "recognizers"]
                    .iter()
                    .zip(actual)
                    .filter(|(_, o)| *o)
                    .map(|(n, _)| *n)
                    .collect();
                format!("{} ({})", b.name(), parts.join(", "))
            }
            None => "cpu".into(),
        };
        let thresh = from_meta("thresh").unwrap_or(DET_THRESH);
        let merge = from_meta("merge").unwrap_or(DET_MERGE);
        let info = OcriaInfo {
            name: "OCRia-nano-1".into(),
            path: dir.to_path_buf(),
            scripts: GROUPS
                .iter()
                .zip(&recs)
                .filter(|(_, r)| r.is_some())
                .map(|(g, _)| g.to_string())
                .collect(),
            det_thresh: thresh,
            det_merge: merge,
            det_long_side: opts.det_long_side,
            params,
            device,
            load_secs: t0.elapsed().as_secs_f64(),
        };
        Ok(Self {
            det,
            router,
            recs,
            thresh,
            merge,
            long_side: opts.det_long_side,
            info,
        })
    }

    pub fn info(&self) -> &OcriaInfo {
        &self.info
    }

    /// The recognizer of `group`, else Latin's, else the first one loaded.
    fn rec_for(&self, group: usize) -> &Recognizer {
        self.recs[group]
            .as_ref()
            .or(self.recs[0].as_ref())
            .or_else(|| self.recs.iter().flatten().next())
            .expect("a recognizer is loaded")
    }

    /// The group whose recognizer reads `group`'s lines.
    fn group_for(&self, group: usize) -> usize {
        if self.recs[group].is_some() {
            group
        } else if self.recs[0].is_some() {
            0
        } else {
            self.recs.iter().position(Option::is_some).unwrap_or(0)
        }
    }

    /// OCR of an encoded image (PNG, JPEG, WebP, GIF, BMP; the `media` feature), upright per
    /// its EXIF orientation unless `opts.exif` is off.
    pub fn ocr_image_bytes(&self, bytes: &[u8], opts: &OcrRunOptions) -> Result<OcrPage> {
        let img = decode_image(bytes, opts.exif)?;
        let mut page = self.ocr_rgb(&img.rgb, img.width, img.height, opts)?;
        page.exif_orientation = img.orientation;
        Ok(page)
    }

    /// OCR of an image file.
    pub fn ocr_file(&self, path: &Path, opts: &OcrRunOptions) -> Result<OcrPage> {
        let b = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        self.ocr_image_bytes(&b, opts).with_context(|| path.display().to_string())
    }

    /// OCR of one page: `data` is RGB (3 bytes a pixel) or grey (1), row-major, `w` x `h`.
    pub fn ocr_rgb(&self, data: &[u8], w: usize, h: usize, opts: &OcrRunOptions) -> Result<OcrPage> {
        let t0 = std::time::Instant::now();
        if w == 0 || h == 0 {
            bail!("empty image ({w} x {h})");
        }
        let img0 = if data.len() == w * h * 3 {
            Img {
                w,
                h,
                ch: 3,
                data: data.to_vec(),
            }
        } else if data.len() == w * h {
            Img {
                w,
                h,
                ch: 3,
                data: data.iter().flat_map(|&v| [v, v, v]).collect(),
            }
        } else {
            bail!("{} bytes for a {w} x {h} image (RGB or grey expected)", data.len());
        };
        let mut det = Detector::new(
            &self.det,
            opts.det_long_side.unwrap_or(self.long_side),
            opts.det_thresh.unwrap_or(self.thresh),
            opts.det_merge.unwrap_or(self.merge),
        );
        det.adaptive = opts.adaptive.then_some(DET_ADAPTIVE);
        det.faint = opts.faint.then_some(DET_FAINT);
        let (mut boxes, mut faint) = det.detect_with_faint(&img0)?;
        let mut sideways = false;
        let mut flip_hint: Option<bool> = None;
        let turned;
        let mut img = &img0;
        let tall = page::mostly_tall(&boxes);
        if opts.orient && (page::looks_sideways(&boxes) || tall) {
            // Maybe a page scanned on its side: detect again a quarter turn counter-clockwise and
            // keep that when it reads clearly better (the router's flip vote finishes 270 degrees).
            turned = img0.rot90(1);
            let tboxes = det.detect(&turned)?;
            let longer = page::long_lines(&tboxes) > 2 * page::long_lines(&boxes) + 5;
            let margin = if longer { 0.05 } else { 0.1 };
            if longer || tall {
                let (conf_t, flip_t) = self.sample_conf(&turned, &tboxes, opts)?;
                if conf_t > self.sample_conf(&img0, &boxes, opts)?.0 + margin {
                    img = &turned;
                    boxes = tboxes;
                    faint.clear();
                    sideways = true;
                    flip_hint = Some(flip_t);
                }
            }
        }
        if opts.gutter {
            boxes = page::split_gutters(img, &boxes);
        }
        let (mut reads, flip, dom, mut groups) = if boxes.is_empty() {
            (Vec::new(), false, 0, Vec::new())
        } else {
            self.read(img, &boxes, flip_hint, opts)?
        };
        let n_lines = boxes.len();
        if !faint.is_empty() && !flip {
            // With the page's recognizer: a lone glyph is too short for the router.
            let crops: Vec<Img> = faint.iter().map(|b| page::crop(img, b)).collect();
            let refs: Vec<&Img> = crops.iter().collect();
            let res = self.rec_for(dom).read(&refs, opts.bidi)?;
            for (quad, (text, conf, _)) in faint.into_iter().zip(res) {
                let text = if opts.clean { page::clean_latin(&text) } else { text };
                if conf >= FAINT_CONF && page::is_short_token(&text) {
                    boxes.push(quad);
                    reads.push((text, conf, Vec::new()));
                    groups.push(self.group_for(dom));
                }
            }
        }
        let mut lines = Vec::new();
        let mut frame = Vec::new();
        for (k, (quad, (text, conf, spans))) in boxes.iter().zip(reads).enumerate() {
            if text.trim().is_empty() || (opts.junk && page::is_junk(&text, conf)) {
                continue;
            }
            let pts = page::reading_frame(quad, flip, sideways, img0.w);
            let (mut x1, mut y1, mut x2, mut y2) = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
            for p in &pts {
                x1 = x1.min(p[0]);
                y1 = y1.min(p[1]);
                x2 = x2.max(p[0]);
                y2 = y2.max(p[1]);
            }
            // Upright lines only: a turned page's word boxes are in another frame.
            let upright = !flip && !sideways && pts[1][0] > pts[0][0] && (pts[1][1] - pts[0][1]).abs() < 0.2 * (pts[1][0] - pts[0][0]);
            let words = if upright { word_boxes(&text, &spans, &pts) } else { Vec::new() };
            // The box in the frame the page was read in (turned back if upside down), for the order.
            frame.push(if flip {
                quad.map(|p| [img.w as f32 - p[0], img.h as f32 - p[1]])
            } else {
                *quad
            });
            lines.push(OcrLine {
                text,
                confidence: conf,
                quad: pts,
                bbox: [x1.trunc(), y1.trunc(), x2.trunc(), y2.trunc()],
                script: GROUPS[groups.get(k).copied().unwrap_or(dom)].to_string(),
                words,
                faint: k >= n_lines,
            });
        }
        if opts.layout {
            let order = reading_order(&frame);
            let mut taken: Vec<Option<OcrLine>> = lines.into_iter().map(Some).collect();
            lines = order.iter().filter_map(|&i| taken[i].take()).collect();
        }
        let text = lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n");
        Ok(OcrPage {
            width: w,
            height: h,
            lines,
            text,
            script: GROUPS[self.group_for(dom)].to_string(),
            upside_down: flip,
            sideways,
            exif_orientation: None,
            seconds: t0.elapsed().as_secs_f64(),
        })
    }

    /// Mean reading confidence over the widest boxes (up to 24), weighted by text length: how
    /// well the page reads in this frame; and whether the router reads it upside down.
    fn sample_conf(&self, img: &Img, boxes: &[Quad], opts: &OcrRunOptions) -> Result<(f32, bool)> {
        let mut by_w: Vec<&Quad> = boxes.iter().collect();
        let width = |q: &Quad| {
            let xs = q.iter().map(|p| p[0]);
            xs.clone().fold(f32::NEG_INFINITY, f32::max) - xs.fold(f32::INFINITY, f32::min)
        };
        by_w.sort_by(|a, b| width(b).total_cmp(&width(a)));
        let sample: Vec<Quad> = by_w.into_iter().take(24).copied().collect();
        if sample.is_empty() {
            return Ok((0.0, false));
        }
        let (reads, flip, ..) = self.read(img, &sample, None, opts)?;
        let (mut num, mut den) = (0f32, 0f32);
        for (t, c, _) in &reads {
            let n = t.trim().chars().count().max(1) as f32;
            num += c * n;
            den += n;
        }
        Ok((if den > 0.0 { num / den } else { 0.0 }, flip))
    }

    /// Readings per box (`server.read`): the router, the recognizers by group, the rescue.
    fn read(&self, img: &Img, boxes: &[Quad], flip_hint: Option<bool>, opts: &OcrRunOptions) -> Result<PageReads> {
        let t0 = stage_start();
        let mut crops: Vec<Img> = crop_all(img, boxes);
        stage("crops", t0);
        let t0 = stage_start();
        let (groups, flip, share) = self.router.route(&crops)?;
        stage("router", t0);
        let t0 = stage_start();
        let mut groups: Vec<usize> = groups.into_iter().map(|g| self.group_for(g)).collect();
        let flip = flip_hint.unwrap_or(flip);
        if flip {
            crops = crops.iter().map(|c| c.rot90(2)).collect();
        }
        let bidi = opts.bidi;
        let n = crops.len();
        let mut out: Vec<rec::Read> = vec![(String::new(), 0f32, Vec::new()); n];
        let subset = |idx: &[usize]| -> Vec<&Img> { idx.iter().map(|&i| &crops[i]).collect() };
        for gi in 0..GROUPS.len() {
            let idx: Vec<usize> = (0..n).filter(|&i| groups[i] == gi).collect();
            if idx.is_empty() {
                continue;
            }
            for (&i, res) in idx.iter().zip(self.rec_for(gi).read(&subset(&idx), bidi)?) {
                out[i] = res;
            }
        }
        stage("recognizers", t0);
        let t0 = stage_start();
        let mut dom = self.group_for(page::majority(&groups));
        // A routed reading this confident is a real reading, not the 0.5-0.7 junk of the wrong recognizer.
        const KEEP_CONF: f32 = 0.75;
        if opts.rescue && dom == 0 {
            // A whole page of another script sent to the Latin recognizer reads as confident-looking
            // junk at 0.5-0.7; the right recognizer then scores 0.83-0.90. Only scripts the router saw
            // a fair share of are tried.
            const RESCUE_CONF: f64 = 0.75;
            const RESCUE_MARGIN: f64 = 0.15;
            const RESCUE_SHARE: f64 = 0.2;
            let lat: Vec<usize> = (0..n).filter(|&i| groups[i] == dom).collect();
            let loaded = |gi: usize| self.recs[gi].is_some();
            if !lat.is_empty() && (1..4).any(|g| share[g] >= RESCUE_SHARE && loaded(g)) {
                let lat_conf = mean(lat.iter().map(|&i| out[i].1));
                if lat_conf < RESCUE_CONF {
                    let sample: Vec<usize> = lat.iter().copied().take(40).collect();
                    let (mut best, mut best_conf) = (None, lat_conf + RESCUE_MARGIN);
                    for gi in (1..GROUPS.len()).filter(|&gi| share[gi] >= RESCUE_SHARE && loaded(gi)) {
                        let c = mean(self.rec_for(gi).read(&subset(&sample), bidi)?.into_iter().map(|r| r.1));
                        if c >= best_conf {
                            best = Some(gi);
                            best_conf = c;
                        }
                    }
                    if let Some(gi) = best {
                        for (&i, res) in lat.iter().zip(self.rec_for(gi).read(&subset(&lat), bidi)?) {
                            // Per line: a bilingual page keeps its confident Latin lines.
                            if !(out[i].1 >= KEEP_CONF && out[i].1 >= res.1) {
                                out[i] = res;
                                groups[i] = gi;
                            }
                        }
                        dom = gi;
                    }
                }
            }
            // Per-line rescue when the page stays Latin: each Latin-routed line on a page carrying
            // another script's share is re-read; a switch needs the margin and the script's own
            // characters, so English lines on a bilingual page stay Latin.
            if dom == 0 {
                const LINE_MARGIN: f32 = 0.1;
                const LINE_SHARE: f64 = 0.15;
                const LINE_CONF: f32 = 0.8;
                for gi in (1..GROUPS.len()).filter(|&gi| share[gi] >= LINE_SHARE && loaded(gi)) {
                    let weak: Vec<usize> = (0..n).filter(|&i| groups[i] == 0).collect();
                    if weak.is_empty() {
                        break;
                    }
                    for (&i, res) in weak.iter().zip(self.rec_for(gi).read(&subset(&weak), bidi)?) {
                        if res.1 >= out[i].1 + LINE_MARGIN
                            && res.1 >= LINE_CONF
                            && page::script_chars(gi, &res.0) >= 3
                            && page::script_chars(gi, &res.0) > page::latin_letters(&res.0)
                        {
                            out[i] = res;
                            groups[i] = gi;
                        }
                    }
                }
            }
        }
        if opts.rescue {
            // Lines routed away from the page's script are read with the page's recognizer too;
            // the off-script reading must win clearly, unless it is confident and in its script.
            let odd: Vec<usize> = (0..n).filter(|&i| groups[i] != dom).collect();
            if !odd.is_empty() {
                for (&i, res) in odd.iter().zip(self.rec_for(dom).read(&subset(&odd), bidi)?) {
                    let routed_ok = out[i].1 >= KEEP_CONF
                        && page::script_chars(groups[i], &out[i].0) >= 2
                        && page::script_chars(groups[i], &out[i].0) > page::latin_letters(&out[i].0);
                    let replace = !routed_ok && (out[i].1 < res.1 + 0.1 || page::script_chars(groups[i], &out[i].0) < 3);
                    if replace {
                        out[i] = res;
                        groups[i] = dom;
                    }
                }
            }
        }
        stage("rescue", t0);
        if opts.clean {
            for o in out.iter_mut() {
                o.0 = page::clean_latin(&o.0);
            }
        }
        Ok((out, flip, dom, groups))
    }
}

static PROFILE: std::sync::Mutex<Vec<(&'static str, usize, f64)>> = std::sync::Mutex::new(Vec::new());
static PROFILE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Times the pipeline's stages from now on (process-wide), for [`profile_report`].
pub fn set_profile(on: bool) {
    PROFILE_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn stage_start() -> Option<std::time::Instant> {
    PROFILE_ON.load(std::sync::atomic::Ordering::Relaxed).then(std::time::Instant::now)
}

fn stage(name: &'static str, t0: Option<std::time::Instant>) {
    if let Some(t0) = t0 {
        let mut p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
        let dt = t0.elapsed().as_secs_f64();
        match p.iter_mut().find(|e| e.0 == name) {
            Some(e) => {
                e.1 += 1;
                e.2 += dt;
            }
            None => p.push((name, 1, dt)),
        }
    }
}

/// The time per stage since the last report (while [`set_profile`] is on), as lines.
pub fn profile_report() -> String {
    let p = std::mem::take(&mut *PROFILE.lock().unwrap_or_else(|e| e.into_inner()));
    p.iter()
        .map(|(k, n, t)| format!("{k:>20}: {n:6} calls {:9.2} ms\n", t * 1e3))
        .collect()
}

/// A decoded image, upright.
pub struct UprightImage {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
    /// The EXIF orientation applied (2 to 8), if any.
    pub orientation: Option<u16>,
}

/// Decodes an encoded image (PNG, JPEG, WebP, GIF, BMP; the `media` feature) to RGB, turned
/// upright per its EXIF orientation when `exif` (PIL's `exif_transpose`, as LiteParse's image
/// loading does).
pub fn decode_image(bytes: &[u8], exif: bool) -> Result<UprightImage> {
    let img = crate::mm::decode_images(&[bytes.to_vec()])?.remove(0);
    let o = exif
        .then(|| crate::depth_pro::read_exif(bytes).and_then(|e| e.orientation))
        .flatten()
        .filter(|o| (2..=8).contains(o));
    Ok(match o {
        Some(o) => {
            let (rgb, width, height) = crate::depth_pro::apply_orientation(&img.data, img.width, img.height, o);
            UprightImage {
                rgb,
                width,
                height,
                orientation: Some(o),
            }
        }
        None => UprightImage {
            rgb: img.data,
            width: img.width,
            height: img.height,
            orientation: None,
        },
    })
}

/// `f` over `0..n` on the pool, the results in order.
pub(crate) fn par_map<U: Send>(n: usize, f: impl Fn(usize) -> U + Sync) -> Vec<U> {
    if n <= 1 {
        return (0..n).map(f).collect();
    }
    let slots: Vec<std::sync::Mutex<Option<U>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    crate::pool::run(n, &|i| *slots[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(f(i)));
    slots
        .into_iter()
        .map(|s| s.into_inner().unwrap_or_else(|e| e.into_inner()).expect("computed"))
        .collect()
}

/// Perspective crops of every box, in parallel.
fn crop_all(img: &Img, boxes: &[Quad]) -> Vec<Img> {
    par_map(boxes.len(), |i| page::crop(img, &boxes[i]))
}

/// Word boxes from the recognizer's word spans (shares of the line along its reading direction),
/// placed on the line's quad. Empty unless there is one span per word of the final text.
fn word_boxes(text: &str, spans: &[(f32, f32)], pts: &Quad) -> Vec<OcrWord> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 2 || words.len() != spans.len() {
        return Vec::new();
    }
    let at = |a: [f32; 2], b: [f32; 2], f: f32| [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
    words
        .iter()
        .zip(spans)
        .map(|(w, &(f0, f1))| {
            let q = [
                at(pts[0], pts[1], f0),
                at(pts[0], pts[1], f1),
                at(pts[3], pts[2], f1),
                at(pts[3], pts[2], f0),
            ];
            let xs = q.iter().map(|p| p[0]);
            let ys = q.iter().map(|p| p[1]);
            OcrWord {
                text: w.to_string(),
                bbox: [
                    xs.clone().fold(f32::INFINITY, f32::min),
                    ys.clone().fold(f32::INFINITY, f32::min),
                    xs.fold(f32::NEG_INFINITY, f32::max),
                    ys.fold(f32::NEG_INFINITY, f32::max),
                ],
            }
        })
        .collect()
}

/// np.mean over Python floats.
fn mean(v: impl Iterator<Item = f32>) -> f64 {
    let (s, n) = v.fold((0f64, 0usize), |(s, n), x| (s + x as f64, n + 1));
    if n == 0 { f64::NAN } else { s / n as f64 }
}

/// Reading order of line boxes by recursive XY cuts: a page splits at blank horizontal bands
/// at least half a line high (sections, paragraphs), else at blank vertical bands a line high
/// (columns, left to right), else at any blank horizontal band (lines); what does not split
/// any further is read in row bands of half a line, left to right.
pub fn reading_order(quads: &[Quad]) -> Vec<usize> {
    let ext: Vec<[f32; 4]> = quads
        .iter()
        .map(|q| {
            let xs = q.iter().map(|p| p[0]);
            let ys = q.iter().map(|p| p[1]);
            [
                xs.clone().fold(f32::INFINITY, f32::min),
                ys.clone().fold(f32::INFINITY, f32::min),
                xs.fold(f32::NEG_INFINITY, f32::max),
                ys.fold(f32::NEG_INFINITY, f32::max),
            ]
        })
        .collect();
    let mut hs: Vec<f32> = ext.iter().map(|e| e[3] - e[1]).collect();
    hs.sort_by(f32::total_cmp);
    let line_h = hs.get(hs.len() / 2).copied().unwrap_or(1.0).max(1.0);
    let mut out = Vec::with_capacity(quads.len());
    xy_cut(&ext, (0..quads.len()).collect(), line_h, &mut out, 0);
    out
}

/// Splits `idx` at gaps of at least `min_gap` along axis `ax` (0: x, 1: y), in order; `None`
/// when there is no such gap.
fn cut(ext: &[[f32; 4]], idx: &[usize], ax: usize, min_gap: f32) -> Option<Vec<Vec<usize>>> {
    let mut by: Vec<usize> = idx.to_vec();
    by.sort_by(|&a, &b| ext[a][ax].total_cmp(&ext[b][ax]).then(a.cmp(&b)));
    let mut groups = vec![vec![by[0]]];
    let mut end = ext[by[0]][ax + 2];
    for &i in &by[1..] {
        if ext[i][ax] - end >= min_gap && ext[i][ax] > end {
            groups.push(Vec::new());
        }
        end = end.max(ext[i][ax + 2]);
        groups.last_mut().unwrap().push(i);
    }
    (groups.len() > 1).then_some(groups)
}

fn xy_cut(ext: &[[f32; 4]], idx: Vec<usize>, line_h: f32, out: &mut Vec<usize>, depth: usize) {
    if idx.len() <= 1 || depth > 64 {
        let mut idx = idx;
        idx.sort_by(|&a, &b| ext[a][1].total_cmp(&ext[b][1]).then(ext[a][0].total_cmp(&ext[b][0])));
        out.extend(idx);
        return;
    }
    let parts = cut(ext, &idx, 1, 0.5 * line_h)
        .or_else(|| cut(ext, &idx, 0, line_h))
        .or_else(|| cut(ext, &idx, 1, 0.0));
    match parts {
        Some(parts) => {
            for p in parts {
                xy_cut(ext, p, line_h, out, depth + 1);
            }
        }
        None => {
            // Rows of half a line, left to right.
            let mut idx = idx;
            let band = 0.5 * line_h;
            idx.sort_by(|&a, &b| {
                let (ya, yb) = ((ext[a][1] / band).round(), (ext[b][1] / band).round());
                ya.total_cmp(&yb).then(ext[a][0].total_cmp(&ext[b][0]))
            });
            out.extend(idx);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
