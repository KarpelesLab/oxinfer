use super::*;
use crate::onnx::build::{A, model, node, tensor};

/// A folder with tiny hand-made models in OCRia's interfaces: a detector whose probability is high
/// on dark pixels (a 2 x 2 stride-2 conv of minus the mean, a sigmoid), a router that always says
/// Latin and upright, a Latin recognizer whose 32 x 4 stride-4 conv reads "x" on dark columns and
/// the blank on light ones (charset `["x", "y"]`).
pub(crate) fn tiny(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oxinfer-ocria-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let k = 12.0f32;
    let det = model(
        &[
            node(
                "Conv",
                &["x", "dw", "db"],
                &["c"],
                &[("strides", A::Is(&[2, 2])), ("kernel_shape", A::Is(&[2, 2]))],
            ),
            node("Sigmoid", &["c"], &["prob"], &[]),
        ],
        &[
            tensor("dw", &[1, 3, 2, 2], 1, &[-k / 12.0; 12], &[]),
            tensor("db", &[1], 1, &[-2.0], &[]),
        ],
        &["x"],
        &["prob"],
    );
    std::fs::write(dir.join("det.onnx"), det).unwrap();
    std::fs::write(dir.join("det.meta.json"), r#"{"thresh": 0.55}"#).unwrap();
    let router = model(
        &[
            node("ReduceMean", &["x"], &["m"], &[("axes", A::Is(&[1, 2, 3])), ("keepdims", A::I(0))]),
            node("Unsqueeze", &["m", "ax"], &["m2"], &[]),
            node("Gemm", &["m2", "gw", "gb"], &["group"], &[("transB", A::I(1))]),
            node("Gemm", &["m2", "fw", "fb"], &["flip"], &[("transB", A::I(1))]),
        ],
        &[
            tensor("ax", &[1], 7, &[], &[1]),
            tensor("gw", &[4, 1], 1, &[0.0; 4], &[]),
            tensor("gb", &[4], 1, &[6.0, 0.0, 0.0, 0.0], &[]),
            tensor("fw", &[2, 1], 1, &[0.0; 2], &[]),
            tensor("fb", &[2], 1, &[6.0, 0.0], &[]),
        ],
        &["x"],
        &["group", "flip"],
    );
    std::fs::write(dir.join("router.onnx"), router).unwrap();
    // Logits from the middle rows' mean m: blank = 8 m, "x" = -8 m, "y" = -10.
    let mid: Vec<f32> = (0..32 * 4)
        .map(|i| if (13..19).contains(&(i / 4)) { 8.0 / 24.0 } else { 0.0 })
        .collect();
    let mut w = mid.clone();
    w.extend(mid.iter().map(|v| -v));
    w.extend(vec![0.0; 32 * 4]);
    let rec = model(
        &[
            node(
                "Conv",
                &["x", "rw", "rb"],
                &["c"],
                &[("strides", A::Is(&[32, 4])), ("kernel_shape", A::Is(&[32, 4]))],
            ),
            node("Squeeze", &["c", "sq"], &["h"], &[]),
            node("ReduceMax", &["h"], &["mx"], &[("axes", A::Is(&[1])), ("keepdims", A::I(0))]),
            node("ArgMax", &["h"], &["ids"], &[("axis", A::I(1)), ("keepdims", A::I(0))]),
            node("ReduceLogSumExp", &["h"], &["lse"], &[("axes", A::Is(&[1])), ("keepdims", A::I(0))]),
            node("Sub", &["mx", "lse"], &["d"], &[]),
            node("Exp", &["d"], &["conf"], &[]),
        ],
        &[
            tensor("rw", &[3, 1, 32, 4], 1, &w, &[]),
            tensor("rb", &[3], 1, &[0.0, 0.0, -10.0], &[]),
            tensor("sq", &[1], 7, &[], &[2]),
        ],
        &["x"],
        &["ids", "conf"],
    );
    std::fs::write(dir.join("rec_latin.onnx"), rec).unwrap();
    std::fs::write(dir.join("rec_latin.charset.json"), r#"["x", "y"]"#).unwrap();
    dir
}

/// A white page with dark bars: (x0, y0, x1, y1) each.
pub(crate) fn page_with(w: usize, h: usize, bars: &[(usize, usize, usize, usize)]) -> Vec<u8> {
    let mut img = vec![255u8; w * h * 3];
    for &(x0, y0, x1, y1) in bars {
        for y in y0..y1 {
            for x in x0..x1 {
                img[(y * w + x) * 3..(y * w + x) * 3 + 3].fill(0);
            }
        }
    }
    img
}

fn latin_only() -> OcrOptions {
    OcrOptions {
        scripts: parse_scripts("latin").unwrap(),
        ..Default::default()
    }
}

#[test]
fn pipeline_on_tiny_models() {
    let dir = tiny("pipeline");
    let o = Ocria::load(dir.to_str().unwrap(), &latin_only()).unwrap();
    assert_eq!(o.info().scripts, ["latin"]);
    assert_eq!(o.info().det_thresh, 0.55);
    // Two lines: one long bar, then two bars (words) on one line further down.
    let (w, h) = (320, 160);
    let img = page_with(w, h, &[(40, 30, 260, 50), (40, 100, 120, 120), (150, 100, 280, 120)]);
    let p = o.ocr_rgb(&img, w, h, &OcrRunOptions::plain()).unwrap();
    assert_eq!(p.lines.len(), 2, "{p:?}");
    assert_eq!((p.width, p.height, p.upside_down, p.sideways), (w, h, false, false));
    let first = &p.lines[0];
    assert_eq!(first.text, "x");
    assert!(first.confidence > 0.9);
    assert_eq!(first.script, "latin");
    // The box covers the bar.
    assert!(
        first.bbox[0] <= 40.0 && first.bbox[2] >= 258.0 && first.bbox[1] <= 30.0 && first.bbox[3] >= 49.0,
        "{:?}",
        first.bbox
    );
    // The two bars of the second line merge into one line, read as two "x" with a blank between.
    assert_eq!(p.lines[1].text, "xx");
    assert_eq!(p.text, "x\nxx");
    // Default options: the same lines here (the junk filter keeps confident readings).
    let d = o.ocr_rgb(&img, w, h, &OcrRunOptions::default()).unwrap();
    assert_eq!(d.text, "x\nxx", "{d:?}");
    // Grey input, and a blank page.
    let grey: Vec<u8> = img.as_chunks::<3>().0.iter().map(|p| p[0]).collect();
    assert_eq!(o.ocr_rgb(&grey, w, h, &OcrRunOptions::plain()).unwrap().text, "x\nxx");
    assert!(
        o.ocr_rgb(&vec![255; w * h * 3], w, h, &OcrRunOptions::default())
            .unwrap()
            .lines
            .is_empty()
    );
    assert!(o.ocr_rgb(&img[..10], w, h, &OcrRunOptions::default()).is_err());
    // A missing recognizer is an error unless the scripts leave it out.
    assert!(Ocria::load(dir.to_str().unwrap(), &OcrOptions::default()).is_err());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn options_parse() {
    assert_eq!(parse_scripts("latin, cjk").unwrap(), [true, true, false, false]);
    assert_eq!(parse_scripts("all").unwrap(), [true; 4]);
    assert!(parse_scripts("klingon").is_err());
    assert!(parse_scripts("").is_err());
    let mut o = OcrOptions::default();
    o.set("scripts", "rtl").unwrap();
    o.set("det_long_side", "2048").unwrap();
    assert_eq!((o.scripts, o.det_long_side), ([false, false, true, false], 2048));
    assert!(o.set("det_long_side", "8").is_err());
    assert!(o.set("nope", "1").is_err());
    let mut r = OcrRunOptions::default();
    r.set("gutter", "0").unwrap();
    r.set("det_thresh", "0.4").unwrap();
    assert!(!r.gutter && r.det_thresh == Some(0.4));
    r.set("plain", "1").unwrap();
    assert_eq!(r, OcrRunOptions::plain());
    let j: OcrRunOptions = serde_json::from_str(r#"{"faint": false, "det_long_side": 1024}"#).unwrap();
    assert!(!j.faint && j.rescue && j.det_long_side == Some(1024));
    assert!(serde_json::from_str::<OcrRunOptions>(r#"{"nope": 1}"#).is_err());
    assert_eq!(files_for(&[true, false, false, false]).len(), 7);
}

#[test]
fn rtl_readings_reorder_to_logical() {
    // The RTL recognizer writes what it sees left to right: "שלום" appears as "םולש".
    assert_eq!(rec::visual_to_logical("םולש"), "שלום");
    // Digits keep their left-to-right order inside a right-to-left line.
    assert_eq!(rec::visual_to_logical("123 םולש"), "שלום 123");
    assert_eq!(rec::visual_to_logical("ملاعلاب ابحرم"), "مرحبا بالعالم");
}

#[test]
fn reading_order_follows_columns() {
    let r = |x0: f32, y0: f32, x1: f32, y1: f32| -> Quad { [[x0, y0], [x1, y0], [x1, y1], [x0, y1]] };
    // A title over two columns of three lines (the right column a little lower).
    let q = vec![
        r(10.0, 0.0, 400.0, 30.0),
        r(10.0, 60.0, 190.0, 80.0),
        r(210.0, 64.0, 400.0, 84.0),
        r(10.0, 82.0, 190.0, 102.0),
        r(210.0, 86.0, 400.0, 106.0),
        r(10.0, 104.0, 190.0, 124.0),
        r(210.0, 108.0, 400.0, 128.0),
    ];
    assert_eq!(reading_order(&q), [0, 1, 3, 5, 2, 4, 6]);
    // One row of words is read left to right.
    let row = vec![r(100.0, 0.0, 150.0, 20.0), r(0.0, 2.0, 80.0, 22.0)];
    assert_eq!(reading_order(&row), [1, 0]);
    assert!(reading_order(&[]).is_empty());
}
