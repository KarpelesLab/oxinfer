//! Checks of the layout port. The reference checks read dumps of LiteParse's own run on an image
//! (`OXINFER_LAYOUT_REF`: a folder of page folders, each with `ocr_results.json`,
//! `pages_pre_proj.json`, `model.json` and the image as `image.png`, written by a LiteParse build
//! with the dump hooks of `tools/ocria/liteparse_dump.patch`, and `OXINFER_LAYOUT_MODEL`: the
//! folder holding `tablenet.bin`); without them only the self-contained tests run.

use super::*;
#[cfg(feature = "media")]
use serde_json::Value;

#[cfg(feature = "media")]
fn f(v: &Value) -> f32 {
    v.as_f64().unwrap() as f32
}

#[cfg(feature = "media")]
fn results_from(j: &Value) -> Vec<merge::OcrResult> {
    j["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let b4 = |v: &Value| -> [f32; 4] { [f(&v[0]), f(&v[1]), f(&v[2]), f(&v[3])] };
            merge::OcrResult {
                text: r["text"].as_str().unwrap().to_string(),
                bbox: b4(&r["bbox"]),
                confidence: f(&r["confidence"]),
                polygon: r["polygon"].as_array().map(|p| {
                    let pt = |k: usize| [f(&p[k][0]), f(&p[k][1])];
                    [pt(0), pt(1), pt(2), pt(3)]
                }),
                words: r["words"].as_array().map_or(Vec::new(), |ws| {
                    ws.iter()
                        .map(|w| (w["text"].as_str().unwrap().to_string(), b4(&w["bbox"])))
                        .collect()
                }),
                faint: r["faint"].as_bool().unwrap_or(false),
            }
        })
        .collect()
}

/// The image of a reference page, decoded.
#[cfg(feature = "media")]
fn image(dir: &std::path::Path) -> (Vec<u8>, usize, usize) {
    let b = std::fs::read(dir.join("image.png")).unwrap();
    let im = crate::ocria::decode_image(&b, true).unwrap();
    (im.rgb, im.width, im.height)
}

/// Max |a - b| over two equal-length vectors.
#[cfg(feature = "media")]
fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Each reference page: the merge, the projection's signals, the features, the logits, the
/// tables and the blocks against LiteParse's.
#[cfg(feature = "media")]
#[test]
fn against_liteparse() {
    let (Some(root), Some(model)) = (std::env::var_os("OXINFER_LAYOUT_REF"), std::env::var_os("OXINFER_LAYOUT_MODEL")) else {
        return;
    };
    let net = LayoutNet::load(&std::path::Path::new(&model).join(FILE)).unwrap();
    let mut dirs: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("model.json").exists())
        .collect();
    dirs.sort();
    assert!(!dirs.is_empty());
    for dir in dirs {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let rd = |n: &str| -> Value { serde_json::from_slice(&std::fs::read(dir.join(n)).unwrap()).unwrap() };
        let ocr = rd("ocr_results.json");
        let pages: Vec<types::Page> = serde_json::from_value(rd("pages_pre_proj.json")).unwrap();
        let m = rd("model.json");
        let (rgb, w, h) = image(&dir);
        let dpi = f(&ocr["dpi"]);
        let (layout, st) = analyze_full(&net, &results_from(&ocr), &rgb, w, h, dpi);
        // the merge: the same items, boxes and font sizes
        let refp = &pages[0];
        assert_eq!(st.page.text_items.len(), refp.text_items.len(), "{name}: items");
        assert_eq!(
            (st.page.page_width, st.page.page_height),
            (refp.page_width, refp.page_height),
            "{name}: page size"
        );
        assert_eq!(st.page.ocr_deskew, refp.ocr_deskew, "{name}: deskew");
        for (a, b) in st.page.text_items.iter().zip(&refp.text_items) {
            assert_eq!(a.text, b.text, "{name}");
            assert_eq!(
                [a.x, a.y, a.width, a.height, a.rotation],
                [b.x, b.y, b.width, b.height, b.rotation],
                "{name}: {}",
                a.text
            );
            assert_eq!(a.font_size, b.font_size, "{name}: {}", a.text);
            assert_eq!(a.words.len(), b.words.len(), "{name}: {}", a.text);
        }
        // the projection's signals, from LiteParse's own page
        let pj = projection::project_page(refp);
        let sig = &m["sig"];
        let items = sig["items"].as_array().unwrap();
        assert_eq!(pj.signals.items.len(), items.len(), "{name}: signal items");
        for (a, b) in pj.signals.items.iter().zip(items) {
            assert_eq!(a.text, b["text"].as_str().unwrap(), "{name}");
            let bx = [f(&b["bx"][0]), f(&b["bx"][1]), f(&b["bx"][2]), f(&b["bx"][3])];
            assert_eq!(a.bx, bx, "{name}: {}", a.text);
            let got = (a.flow, a.lc, a.rc, a.cc, a.snap.to_string(), a.block);
            let want = (
                b["flow"].as_bool().unwrap(),
                b["lc"].as_u64().unwrap() as usize,
                b["rc"].as_u64().unwrap() as usize,
                b["cc"].as_u64().unwrap() as usize,
                b["snap"].as_str().unwrap().to_string(),
                b["block"].as_u64().unwrap() as usize,
            );
            assert_eq!(got, want, "{name}: {}", a.text);
        }
        let leaves = sig["leaves"].as_array().unwrap();
        assert_eq!(pj.signals.leaves.len(), leaves.len(), "{name}: leaves");
        for (a, b) in pj.signals.leaves.iter().zip(leaves) {
            let path: Vec<u16> = b["path"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u16).collect();
            assert_eq!(a.1, path, "{name}");
        }
        // our pipeline's own page agrees with LiteParse's, so its stages are comparable
        assert_eq!(st.projected.signals.items.len(), items.len(), "{name}");
        // the ruling lines and the image branch's input, from our render of the image: PDFium's
        // when the image is at the render's 150 dpi (its pixels as they are), else close to it
        // (PDFium's resampling and ours differ)
        let exact = (dpi - DEFAULT_DPI).abs() < 1e-3;
        if let (Some(r), Some(rr), true) = (&st.rules, m["rules"].as_object(), exact) {
            let flat = |v: &Value| -> Vec<f32> {
                v.as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|s| s.as_array().unwrap().iter().map(f))
                    .collect()
            };
            let h: Vec<f32> = r.h.iter().flatten().copied().collect();
            let v: Vec<f32> = r.v.iter().flatten().copied().collect();
            assert_eq!(h, flat(&rr["h"]), "{name}: horizontal rules");
            assert_eq!(v, flat(&rr["v"]), "{name}: vertical rules");
        }
        if let Some(im) = &st.image {
            let want = std::fs::read(dir.join("page_img.raw")).unwrap();
            let diff = im.px.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert_eq!(im.px.len(), want.len(), "{name}: image size");
            if exact {
                assert!(diff == 0, "{name}: {diff} image-branch pixels differ");
            } else {
                let mad = im.px.iter().zip(&want).map(|(a, b)| (*a as f64 - *b as f64).abs()).sum::<f64>() / want.len() as f64;
                eprintln!(
                    "{name}: {dpi} dpi image, {diff} of {} image-branch pixels differ (mean {mad:.3} levels)",
                    want.len()
                );
            }
        }
        let feats: Vec<f32> = m["feats"].as_array().unwrap().iter().map(f).collect();
        let n = st.model.units.len();
        if n >= 2 && !feats.is_empty() {
            let df = max_diff(&st.model.feats, &feats);
            assert!(df == 0.0 || !exact, "{name}: features differ by {df}");
            let node: Vec<f32> = m["node"].as_array().unwrap().iter().map(f).collect();
            let cls: Vec<f32> = m["cls"].as_array().unwrap().iter().map(f).collect();
            let dn = max_diff(&st.model.node, &node);
            let dc = max_diff(&st.model.cls, &cls);
            eprintln!("{name}: {n} units, node logits within {dn:.2e}, class logits within {dc:.2e}");
            let tol = if exact { 1e-3 } else { 0.25 };
            assert!(dn < tol && dc < tol, "{name}: logits {dn} {dc}");
        }
        // the tables, cell by cell, and the blocks
        let tables = m["tables"].as_array().unwrap();
        assert_eq!(st.model.tables.len(), tables.len(), "{name}: tables");
        for (a, b) in st.model.tables.iter().zip(tables) {
            assert_eq!(
                (a.grid.n_rows, a.grid.n_cols),
                (b["n_rows"].as_u64().unwrap() as usize, b["n_cols"].as_u64().unwrap() as usize),
                "{name}"
            );
            let hr: Vec<usize> = b["head_rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(a.grid.head_rows, hr, "{name}");
            let cells = b["cells"].as_array().unwrap();
            assert_eq!(a.grid.cells.len(), cells.len(), "{name}");
            for (c, d) in a.grid.cells.iter().zip(cells) {
                let u = |k: &str| d[k].as_u64().unwrap() as usize;
                assert_eq!(
                    (c.r0, c.c0, c.r1, c.c1, c.text.as_str()),
                    (u("r0"), u("c0"), u("r1"), u("c1"), d["text"].as_str().unwrap()),
                    "{name}"
                );
            }
        }
        let blocks = m["blocks"].as_array().unwrap();
        assert_eq!(st.model.blocks.len(), blocks.len(), "{name}: blocks");
        for (a, b) in st.model.blocks.iter().zip(blocks) {
            assert_eq!(table::BLOCK_CLASSES[a.class], b["class"].as_str().unwrap(), "{name}: {}", a.text);
            assert_eq!(a.text, b["text"].as_str().unwrap(), "{name}");
        }
        // the page's blocks as LiteParse's `--extract-blocks` (its tables come from its markdown
        // stage, ours from the model: the same tables here; our table-class and figure-text
        // leftovers are not among its blocks)
        let lp: Value = serde_json::from_slice(&std::fs::read(dir.join("blocks.json")).unwrap()).unwrap();
        let want: Vec<(String, Option<u64>, String)> = lp["pages"][0]["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                let k = b["kind"].as_str().unwrap();
                let k = if k == "merged_table" { "table" } else { k };
                (
                    k.to_string(),
                    b["level"].as_u64(),
                    if k == "table" {
                        String::new()
                    } else {
                        b["text"].as_str().unwrap_or("").to_string()
                    },
                )
            })
            .collect();
        let got: Vec<(String, Option<u64>, String)> = layout
            .blocks
            .iter()
            .filter(|b| b.kind != "table_text" && b.kind != "figure_text")
            .map(|b| (b.kind.clone(), b.level.map(u64::from), b.text.clone()))
            .collect();
        assert_eq!(got, want, "{name}: blocks");
        eprintln!(
            "{name}: {} tables, {} model blocks, {} page blocks as LiteParse",
            tables.len(),
            blocks.len(),
            got.len()
        );
    }
}

/// A tiny random LayoutNet (`tablenet.bin` format: `TNET`, the header's length, its JSON, the
/// tensors as little-endian f32): d 8, one layer, the layout heads, no image branch.
pub(crate) fn tiny_tablenet() -> Vec<u8> {
    let (d, ff, pd, fin) = (8usize, 16usize, 4usize, table::features::FEAT_DIM - table::net::N_IDS);
    let mut shapes: Vec<(String, Vec<usize>)> = Vec::new();
    let mut lin = |n: &str, o: usize, i: usize| {
        shapes.push((format!("{n}.weight"), vec![o, i]));
        shapes.push((format!("{n}.bias"), vec![o]));
    };
    lin("inp.0", d, fin);
    lin("inp.2", d, d);
    lin("node", 2, d);
    lin("qk", 2 * 5 * pd, d);
    lin("pair_geo", 5, 15);
    lin("cls", 13, d);
    lin("enc.layers.0.self_attn.out_proj", d, d);
    lin("enc.layers.0.linear1", ff, d);
    lin("enc.layers.0.linear2", d, ff);
    shapes.push(("enc.layers.0.self_attn.in_proj_weight".into(), vec![3 * d, d]));
    shapes.push(("enc.layers.0.self_attn.in_proj_bias".into(), vec![3 * d]));
    for n in ["norm1", "norm2"] {
        shapes.push((format!("enc.layers.0.{n}.weight"), vec![d]));
        shapes.push((format!("enc.layers.0.{n}.bias"), vec![d]));
    }
    let mut tensors = Vec::new();
    let mut data = Vec::new();
    let mut seed = 0x2545F491u32;
    let mut off = 0;
    for (name, shape) in &shapes {
        let n: usize = shape.iter().product();
        tensors.push(serde_json::json!({ "name": name, "shape": shape, "offset": off }));
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let v = if name.contains("norm") && name.ends_with("weight") {
                1.0
            } else {
                (seed as f32 / u32::MAX as f32 - 0.5) * 0.5
            };
            data.extend_from_slice(&v.to_le_bytes());
        }
        off += n;
    }
    let header = serde_json::json!({
        "cfg": { "d": d, "layers": 1, "heads": 2, "pair_d": pd, "stack_feats": true, "rule_feats": true, "layout": true, "n_cls": 13 },
        "feat_dim": table::features::FEAT_DIM,
        "tensors": tensors,
    })
    .to_string();
    let mut out = b"TNET".to_vec();
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

/// The whole layout pass on a made-up page with the tiny network: every unit lands in a block or
/// a table, the boxes in the image's pixels.
#[test]
fn tiny_network_page() {
    let net = LayoutNet::from_bytes(&tiny_tablenet()).unwrap();
    assert_eq!(net.params(), 1783);
    let line = |t: &str, x: f32, y: f32, w: f32| merge::OcrResult {
        text: t.into(),
        bbox: [x, y, x + w, y + 30.0],
        confidence: 0.95,
        polygon: Some([[x, y], [x + w, y], [x + w, y + 30.0], [x, y + 30.0]]),
        words: Vec::new(),
        faint: false,
    };
    let results = vec![
        line("Quarterly results", 100.0, 100.0, 400.0),
        line("Revenue grew in every region this year.", 100.0, 200.0, 900.0),
        line("Costs stayed flat across the company.", 100.0, 240.0, 880.0),
    ];
    let (w, h) = (1275, 1650);
    let rgb = vec![255u8; w * h * 3];
    let (l, st) = analyze_full(&net, &results, &rgb, w, h, 150.0);
    assert_eq!(st.model.units.len(), 3);
    assert_eq!(st.model.classes.len(), 3);
    let text: Vec<&str> = l.blocks.iter().map(|b| b.text.as_str()).collect();
    for r in &results {
        assert!(
            text.iter().any(|t| t.contains(&r.text)) || l.tables.iter().flat_map(|t| &t.cells).any(|c| c.text.contains(&r.text)),
            "{}",
            r.text
        );
    }
    for b in &l.blocks {
        assert!(
            b.bbox[0] >= 99.0 && b.bbox[2] <= 1001.0 && b.bbox[1] >= 99.0 && b.bbox[3] <= 271.0,
            "{:?}",
            b.bbox
        );
    }
    assert!(!l.markdown.is_empty());
    // a page the model skips (under 20 characters) keeps its lines as paragraphs
    let (l, _) = analyze_full(&net, &[line("Hi", 10.0, 10.0, 60.0)], &rgb, w, h, 150.0);
    assert_eq!(
        (l.blocks.len(), l.blocks[0].kind.as_str(), l.markdown.as_str()),
        (1, "paragraph", "Hi")
    );
}

#[test]
fn image_dpi_from_png_and_jpeg() {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&9u32.to_be_bytes());
    png.extend_from_slice(b"pHYs");
    png.extend_from_slice(&11811u32.to_be_bytes());
    png.extend_from_slice(&11811u32.to_be_bytes());
    png.push(1);
    png.extend_from_slice(&[0; 4]);
    png.extend_from_slice(&0u32.to_be_bytes());
    png.extend_from_slice(b"IEND");
    png.extend_from_slice(&[0; 4]);
    let (x, y) = image_dpi(&png).unwrap();
    assert!((x - 300.0).abs() < 0.1 && (y - 300.0).abs() < 0.1);
    let jpg = [
        0xFF, 0xD8, 0xFF, 0xE0, 0, 16, b'J', b'F', b'I', b'F', 0, 1, 1, 1, 0, 200, 0, 200, 0, 0, 0xFF, 0xD9,
    ];
    assert_eq!(image_dpi(&jpg), Some((200.0, 200.0)));
    assert_eq!(page_points(1700, 150.0), 816.0);
}

#[test]
fn markdown_lists_and_tables() {
    let t = Table {
        bbox: [0.0; 4],
        rows: 2,
        cols: 2,
        header_rows: vec![0],
        cells: vec![
            TableCell {
                row: 0,
                col: 0,
                row_span: 1,
                col_span: 1,
                text: "Item".into(),
                bbox: [0.0; 4],
            },
            TableCell {
                row: 0,
                col: 1,
                row_span: 1,
                col_span: 1,
                text: "Qty".into(),
                bbox: [0.0; 4],
            },
            TableCell {
                row: 1,
                col: 0,
                row_span: 1,
                col_span: 1,
                text: "a|b".into(),
                bbox: [0.0; 4],
            },
            TableCell {
                row: 1,
                col: 1,
                row_span: 1,
                col_span: 1,
                text: "3".into(),
                bbox: [0.0; 4],
            },
        ],
    };
    let b = |kind: &str, text: &str| Block {
        kind: kind.into(),
        level: (kind == "heading").then_some(1),
        bbox: [0.0; 4],
        text: text.into(),
        table: None,
    };
    let blocks = vec![
        b("heading", "Report"),
        b("list_item", "• first"),
        b("list_item", "2. second"),
        Block {
            kind: "table".into(),
            level: None,
            bbox: [0.0; 4],
            text: String::new(),
            table: Some(0),
        },
        b("text", "The end."),
    ];
    let md = markdown(&blocks, &[t]);
    assert_eq!(
        md,
        "# Report\n\n- first\n2. second\n\n| Item | Qty |\n| --- | --- |\n| a\\|b | 3 |\n\nThe end."
    );
}
