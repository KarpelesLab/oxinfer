//! Layout blocks and tables of an OCR'd image with OCRia-nano-1's LayoutNet (`tablenet.bin`), as
//! LiteParse parses an image: its OCR lines merged into the page's text items (LiteParse's OCR
//! merge, [`merge`]), the page projected on a character grid with an XY-cut reading order
//! ([`projection`]: the signals LayoutNet reads), the page rendered at 150 dpi for the ruling
//! lines and the network's image branch, then LayoutNet and its decoder ([`table`]): tables with
//! rows, columns, spanning cells and header rows, a block class for every text unit (13 classes:
//! text, heading, title, page header / footer, list, table, caption, footnote, formula, code,
//! figure text, key-value) and layout blocks.
//!
//! Ported from LiteParse (`crates/liteparse/src`: `projection.rs`, `table_model/`, parts of
//! `ocr_merge.rs` and `types.rs`; Apache-2.0, Copyright LlamaIndex), the parts an image page
//! takes; the PDF-only inputs (a text layer, vector graphics) are absent on an image.
//!
//! On top of the model's output this module builds oxinfer's own outputs: the blocks in reading
//! order with tables in place, Markdown (headings, lists, pipe or HTML tables) and JSON.

mod bidi;
pub mod merge;
pub mod projection;
pub mod table;
#[cfg(test)]
pub(crate) mod tests;
pub mod types;

use anyhow::{Context, Result};
use serde::Serialize;
use std::path::Path;

/// The weights file in the model repository.
pub const FILE: &str = "tablenet.bin";

/// The resolution an image is taken at when it carries none (LiteParse's, as ImageMagick's).
pub const DEFAULT_DPI: f32 = 150.0;

/// LayoutNet, loaded.
pub struct LayoutNet {
    net: table::net::TableNet,
}

impl LayoutNet {
    pub fn load(path: &Path) -> Result<Self> {
        let b = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_bytes(&b)
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        Ok(Self {
            net: table::load_bytes(b)?,
        })
    }

    /// Parameters.
    pub fn params(&self) -> usize {
        self.net.params()
    }
}

/// One table cell: its top-left slot, its span, text and box (image pixels: x0, y0, x1, y1).
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct TableCell {
    pub row: usize,
    pub col: usize,
    pub row_span: usize,
    pub col_span: usize,
    pub text: String,
    pub bbox: [f32; 4],
}

/// A table: its grid, the header rows, and its box (image pixels).
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Table {
    pub bbox: [f32; 4],
    pub rows: usize,
    pub cols: usize,
    pub header_rows: Vec<usize>,
    pub cells: Vec<TableCell>,
}

/// A layout block, as LiteParse's `--extract-blocks` gives an image page's: its kind (`heading`
/// with a `level`, 1 for a title, 2 for a heading; `paragraph`, `list_item`, `page_header`,
/// `page_footer`, `code`, `table`), box (image pixels), text, and the table it is (an index
/// into [`Layout::tables`]). LiteParse leaves LayoutNet's table-class and figure-text blocks
/// outside every table to its heuristic tables and figures; here they stay, as `table_text` and
/// `figure_text`.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Block {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<u8>,
    pub bbox: [f32; 4],
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table: Option<usize>,
}

/// The layout of a page.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Layout {
    /// The page's resolution (pixels per inch): page points are pixels * 72 / dpi.
    pub dpi: f32,
    /// The blocks in reading order (an XY cut over their boxes), the tables among them.
    pub blocks: Vec<Block>,
    pub tables: Vec<Table>,
    /// The page as Markdown.
    pub markdown: String,
    /// The page as LiteParse's projected text (a character grid keeping columns apart).
    pub grid_text: String,
}

/// The image resolution recorded in an encoded image (PNG `pHYs` in metres, JPEG JFIF density),
/// as LiteParse reads it to size an image's page; `None` when it records none.
pub fn image_dpi(data: &[u8]) -> Option<(f32, f32)> {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        let mut i = 8;
        while i + 12 <= data.len() {
            let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
            let ty = &data[i + 4..i + 8];
            let body = i + 8;
            if body + len + 4 > data.len() {
                return None;
            }
            if ty == b"pHYs" && len == 9 {
                let be = |k: usize| u32::from_be_bytes([data[k], data[k + 1], data[k + 2], data[k + 3]]);
                if data[body + 8] != 1 {
                    return None;
                }
                let (x, y) = (be(body) as f32 * 0.0254, be(body + 4) as f32 * 0.0254);
                return (x > 1.0 && y > 1.0).then_some((x, y));
            }
            if ty == b"IDAT" || ty == b"IEND" {
                return None;
            }
            i = body + len + 4;
        }
        return None;
    }
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    let mut i = 2;
    while i + 4 <= data.len() {
        if data[i] != 0xFF {
            return None;
        }
        let marker = data[i + 1];
        i += 2;
        if marker == 0xD8 || marker == 0xD9 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        let seg_len = u16::from_be_bytes([data[i], data[i + 1]]) as usize;
        if seg_len < 2 || i + seg_len > data.len() {
            return None;
        }
        if marker == 0xE0 && seg_len >= 16 && &data[i + 2..i + 7] == b"JFIF\0" {
            let units = data[i + 9];
            let x = u16::from_be_bytes([data[i + 10], data[i + 11]]) as f32;
            let y = u16::from_be_bytes([data[i + 12], data[i + 13]]) as f32;
            return match units {
                1 if x > 1.0 && y > 1.0 => Some((x, y)),
                2 if x > 1.0 && y > 1.0 => Some((x * 2.54, y * 2.54)),
                _ => None,
            };
        }
        i += seg_len;
    }
    None
}

/// LiteParse's page size in points for a `w` x `h` image at `dpi` (its PDF's MediaBox, written
/// with 4 decimals).
fn page_points(px: usize, dpi: f32) -> f32 {
    let v = px as f32 * 72.0 / dpi;
    format!("{v:.4}").parse().unwrap_or(v)
}

/// The image as PDFium renders its page at the rules' 150 dpi: the image itself when that is its
/// size (an image at 150 dpi), else resampled to the render's size (area average when shrinking,
/// bilinear when enlarging); BGR as PDFium's buffer.
fn render_raw(rgb: &[u8], w: usize, h: usize, page_w: f32, page_h: f32) -> table::features::RawRaster {
    let scale = table::features::RULES_DPI / 72.0;
    let rw = ((page_w * scale).round() as usize).max(1);
    let rh = ((page_h * scale).round() as usize).max(1);
    let px = if (rw, rh) == (w, h) {
        rgb.to_vec()
    } else {
        let im = super::img::Img {
            w,
            h,
            ch: 3,
            data: rgb.to_vec(),
        };
        if rw < w {
            super::img::resize_area(&im, rw, rh)
        } else {
            super::img::resize_linear(&im, rw, rh)
        }
        .data
    };
    let mut bgr = px;
    for p in bgr.as_chunks_mut::<3>().0 {
        p.swap(0, 2);
    }
    table::features::RawRaster::new(rw, rh, rw * 3, 3, bgr)
}

/// The OCR lines (in the order the OCR read them, before any reordering) as LiteParse's OCR
/// results.
pub fn ocr_results(lines: &[crate::ocria::OcrLine]) -> Vec<merge::OcrResult> {
    lines
        .iter()
        .map(|l| merge::OcrResult {
            text: l.text.clone(),
            bbox: l.bbox,
            confidence: l.confidence,
            polygon: Some(l.quad),
            words: l.words.iter().map(|w| (w.text.clone(), w.bbox)).collect(),
            faint: l.faint,
        })
        .collect()
}

/// What the layout stages produced, for checks against LiteParse's (kept by [`analyze_full`]).
pub struct Stages {
    pub page: types::Page,
    pub projected: projection::Projected,
    pub model: table::PageModel,
    pub rules: Option<table::features::Rules>,
    pub image: Option<table::features::PageImage>,
}

/// The layout of an image page from its OCR results (pixels of `rgb`, `w` x `h`).
pub fn analyze(net: &LayoutNet, results: &[merge::OcrResult], rgb: &[u8], w: usize, h: usize, dpi: f32) -> Layout {
    analyze_full(net, results, rgb, w, h, dpi).0
}

/// [`analyze`], with its stages.
pub fn analyze_full(net: &LayoutNet, results: &[merge::OcrResult], rgb: &[u8], w: usize, h: usize, dpi: f32) -> (Layout, Stages) {
    let (pw, ph) = (page_points(w, dpi), page_points(h, dpi));
    let page = merge::ocr_page(results, dpi, pw, ph);
    let t0 = super::stage_start();
    let projected = projection::project_page(&page);
    super::stage("layout: projection", t0);
    // the page render: its ruling lines, and the image branch's input
    let t0 = super::stage_start();
    let raw = render_raw(rgb, w, h, pw, ph);
    let (rules, image) = table::features::process_raster(&raw, true, net.net.has_img());
    super::stage("layout: render", t0);
    let t0 = super::stage_start();
    // the rendered page's size, as LiteParse's table input has it (on a page turned upright by
    // the merge, the units are scaled from the upright page to it)
    let wh = (pw as f64, ph as f64);
    let model = table::run_page(
        &net.net,
        wh,
        &projected.signals,
        rules.as_ref(),
        image.as_ref(),
        &table::decode::DecodeOptions::default(),
    );
    super::stage("layout: model", t0);
    let frame = Frame {
        dpi,
        wh: (pw, ph),
        sig: (projected.signals.width, projected.signals.height),
        turn: page.turned,
        deskew: page.ocr_deskew,
    };
    let layout = build(&page, &projected, &model, &frame);
    (
        layout,
        Stages {
            page,
            projected,
            model,
            rules,
            image,
        },
    )
}

/// How the model's boxes map to the image: its units are in the rendered page's points (`wh`),
/// scaled from the projected page (`sig`, the upright page when the merge turned a sideways scan
/// by `turn` quarters), which was deskewed by `deskew` degrees.
struct Frame {
    dpi: f32,
    wh: (f32, f32),
    sig: (f32, f32),
    turn: u8,
    deskew: Option<f32>,
}

impl Frame {
    /// A model box (points, top-left origin) to image pixels (x0, y0, x1, y1).
    fn to_pixels(&self, b: [f64; 4]) -> [f32; 4] {
        let [x0, y0, x1, y1] = b.map(|v| v as f32);
        // back to the projected page's points
        let (sx, sy) = (self.sig.0 / self.wh.0.max(1e-6), self.sig.1 / self.wh.1.max(1e-6));
        let (x0, y0, x1, y1) = (x0 * sx, y0 * sy, x1 * sx, y1 * sy);
        // the upright frame back to the image's
        let (w, h) = self.wh;
        let [mut x0, mut y0, mut x1, mut y1] = match self.turn {
            1 => [y0, h - x1, y1, h - x0],
            2 => [w - x1, h - y1, w - x0, h - y0],
            3 => [w - y1, x0, w - y0, x1],
            _ => [x0, y0, x1, y1],
        };
        if let Some(deg) = self.deskew {
            let (cx, cy) = (w / 2.0, h / 2.0);
            let pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)].map(|(x, y)| merge::rotate_about(x, y, cx, cy, deg));
            x0 = pts.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
            x1 = pts.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
            y0 = pts.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
            y1 = pts.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
        }
        let s = self.dpi / 72.0;
        [x0 * s, y0 * s, x1 * s, y1 * s]
    }
}

fn rect_box(r: &types::Rect) -> [f64; 4] {
    [r.x as f64, r.y as f64, (r.x + r.width) as f64, (r.y + r.height) as f64]
}

/// LiteParse's `with_model_blocks` (`layout.rs`) on an image page: the tables (the model's, where
/// LiteParse keeps its markdown stage's tables built from them), then every model block whose
/// centre falls outside them by class, clipped to the page, in XY-cut reading order.
fn build(page: &types::Page, projected: &projection::Projected, model: &table::PageModel, frame: &Frame) -> Layout {
    let rect = |r: &types::Rect| [r.x, r.y, r.x + r.width, r.y + r.height];
    let mut tables = Vec::new();
    // (kind, level, box in points, text, table)
    let mut out: Vec<PageBlock> = Vec::new();
    for (ti, t) in model.tables.iter().enumerate() {
        let g = &t.grid;
        tables.push(Table {
            bbox: frame.to_pixels(rect_box(&t.bbox)),
            rows: g.n_rows,
            cols: g.n_cols,
            header_rows: g.head_rows.clone(),
            cells: g
                .cells
                .iter()
                .map(|c| TableCell {
                    row: c.r0,
                    col: c.c0,
                    row_span: c.r1 - c.r0 + 1,
                    col_span: c.c1 - c.c0 + 1,
                    text: c.text.clone(),
                    bbox: frame.to_pixels(c.bx),
                })
                .collect(),
        });
        out.push(("table", None, rect(&t.bbox), String::new(), Some(ti)));
    }
    let inside = |b: &[f32; 4]| {
        let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
        model.tables.iter().any(|t| {
            let r = rect(&t.bbox);
            r[0] <= cx && cx <= r[2] && r[1] <= cy && cy <= r[3]
        })
    };
    for m in &model.blocks {
        let b = rect(&m.bbox);
        if inside(&b) || m.text.trim().is_empty() {
            continue;
        }
        let (kind, level) = match table::BLOCK_CLASSES[m.class] {
            "Title" => ("heading", Some(1)),
            "Heading" => ("heading", Some(2)),
            "PageHeader" => ("page_header", None),
            "PageFooter" => ("page_footer", None),
            "List" => ("list_item", None),
            "Code" => ("code", None),
            "Table" => ("table_text", None),
            "FigureText" => ("figure_text", None),
            _ => ("paragraph", None),
        };
        out.push((kind, level, b, m.text.clone(), None));
    }
    // A page the model skips (under 2 units or 20 characters, over 1500 units) has no model
    // blocks; where LiteParse classifies its lines heuristically, its projected lines stand here
    // as paragraphs.
    if model.blocks.is_empty() && model.tables.is_empty() {
        let (sx, sy) = (frame.wh.0 / frame.sig.0.max(1e-6), frame.wh.1 / frame.sig.1.max(1e-6));
        for l in projected.lines.iter().filter(|l| !l.text.trim().is_empty()) {
            let r = &l.bbox;
            out.push((
                "paragraph",
                None,
                [r.x * sx, r.y * sy, (r.x + r.width) * sx, (r.y + r.height) * sy],
                l.text.trim().to_string(),
                None,
            ));
        }
    }
    // clipped to the page
    let (pw, ph) = (page.page_width, page.page_height);
    for o in &mut out {
        let b = &mut o.2;
        if b.iter().all(|v| v.is_finite()) && pw > 0.0 && ph > 0.0 {
            let x1 = b[2].clamp(0.0, pw);
            let y1 = b[3].clamp(0.0, ph);
            *b = [b[0].clamp(0.0, x1), b[1].clamp(0.0, y1), x1, y1];
        }
    }
    let boxes: Vec<[f32; 4]> = out.iter().map(|o| o.2).collect();
    let order = xy_cut_order(&boxes, pw * ph);
    let mut slots: Vec<Option<_>> = out.into_iter().map(Some).collect();
    let blocks: Vec<Block> = order
        .into_iter()
        .filter_map(|i| slots[i].take())
        .map(|(kind, level, b, text, table)| Block {
            kind: kind.into(),
            level,
            bbox: frame.to_pixels(b.map(f64::from)),
            text,
            table,
        })
        .collect();
    let markdown = markdown(&blocks, &tables);
    Layout {
        dpi: frame.dpi,
        blocks,
        tables,
        markdown,
        grid_text: projected.text.clone(),
    }
}

/// A page block under construction: kind, heading level, box in the model's points, text, table.
type PageBlock = (&'static str, Option<u8>, [f32; 4], String, Option<usize>);

/// Block indices in reading order by recursive XY cut over their boxes (LiteParse's
/// `xy_cut_order`): a region splits at its widest vertical gutter clear of every block (columns,
/// left to right) when there is one, else at its topmost horizontal gap; a region with neither
/// reads top to bottom, then left to right. Blocks covering half the page or more come first.
fn xy_cut_order(boxes: &[[f32; 4]], page_area: f32) -> Vec<usize> {
    let area = |b: &[f32; 4]| (b[2] - b[0]) * (b[3] - b[1]);
    let (mut order, rest): (Vec<usize>, Vec<usize>) =
        (0..boxes.len()).partition(|&i| page_area > 0.0 && area(&boxes[i]) >= 0.5 * page_area);
    xy_cut(boxes, rest, &mut order);
    order
}

/// A gap of at least `min` between the intervals of `ids` along axis `a` (0: x, 1: y): the
/// widest, or with `first` the lowest; its cut position and width.
fn find_gap(boxes: &[[f32; 4]], ids: &[usize], a: usize, min: f32, first: bool) -> Option<(f32, f32)> {
    let mut iv: Vec<(f32, f32)> = ids.iter().map(|&i| (boxes[i][a], boxes[i][a + 2])).collect();
    iv.sort_by(|p, q| p.0.total_cmp(&q.0));
    let mut best: Option<(f32, f32)> = None;
    let mut reach = f32::NEG_INFINITY;
    for (lo, hi) in iv {
        if reach > f32::NEG_INFINITY && lo - reach >= min && best.is_none_or(|b| lo - reach > b.1) {
            best = Some(((lo + reach) / 2.0, lo - reach));
            if first {
                break;
            }
        }
        reach = reach.max(hi);
    }
    best
}

fn xy_cut(boxes: &[[f32; 4]], ids: Vec<usize>, out: &mut Vec<usize>) {
    if ids.len() <= 1 {
        out.extend(ids);
        return;
    }
    let (axis, cut) = if let Some((c, _)) = find_gap(boxes, &ids, 0, 4.0, false) {
        (0, c)
    } else if let Some((c, _)) = find_gap(boxes, &ids, 1, 1.0, true) {
        (1, c)
    } else {
        let mut ids = ids;
        ids.sort_by(|&a, &b| boxes[a][1].total_cmp(&boxes[b][1]).then(boxes[a][0].total_cmp(&boxes[b][0])));
        out.extend(ids);
        return;
    };
    let (first, second): (Vec<usize>, Vec<usize>) = ids.into_iter().partition(|&i| (boxes[i][axis] + boxes[i][axis + 2]) / 2.0 < cut);
    xy_cut(boxes, first, out);
    xy_cut(boxes, second, out);
}

/// A list item's marker: a bullet, or a number / letter with `.` or `)`.
fn list_marker(t: &str) -> Option<(&str, &str)> {
    let t = t.trim_start();
    let mut it = t.char_indices();
    let (_, c) = it.next()?;
    if matches!(c, '•' | '·' | '◦' | '▪' | '■' | '-' | '–' | '*' | '‣' | '○' | '●') {
        let rest = t[c.len_utf8()..].trim_start();
        return (!rest.is_empty()).then_some(("-", rest));
    }
    let end = t.find(['.', ')'])?;
    let head = &t[..end];
    let ok = (!head.is_empty() && head.len() <= 3 && head.chars().all(|c| c.is_ascii_digit()))
        || (head.len() == 1 && head.chars().all(|c| c.is_ascii_alphabetic()));
    let rest = t[end + 1..].trim_start();
    (ok && !rest.is_empty() && t[end + 1..].starts_with(' ')).then(|| (&t[..end + 1], rest))
}

fn md_escape_cell(t: &str) -> String {
    t.replace('|', "\\|").replace('\n', " ")
}

/// A table as Markdown: a pipe table when no cell spans, else HTML (Markdown has no spans).
pub fn table_markdown(t: &Table) -> String {
    if t.cells.iter().any(|c| c.row_span > 1 || c.col_span > 1) {
        let g = table::decode::Grid {
            cells: t
                .cells
                .iter()
                .map(|c| table::decode::GridCell {
                    r0: c.row,
                    c0: c.col,
                    r1: c.row + c.row_span - 1,
                    c1: c.col + c.col_span - 1,
                    text: c.text.clone(),
                    bx: [0.0; 4],
                })
                .collect(),
            head_rows: t.header_rows.clone(),
            n_rows: t.rows,
            n_cols: t.cols,
        };
        return g.to_html();
    }
    let mut grid = vec![vec![String::new(); t.cols]; t.rows];
    for c in &t.cells {
        if c.row < t.rows && c.col < t.cols {
            grid[c.row][c.col] = md_escape_cell(c.text.trim());
        }
    }
    let head = t.header_rows.iter().take_while(|&&r| r < t.rows).count().max(1).min(t.rows);
    let mut out = Vec::new();
    let row = |r: &[String]| format!("| {} |", r.join(" | "));
    if head > 1 {
        // several header rows: joined per column into the one Markdown allows
        let joined: Vec<String> = (0..t.cols)
            .map(|c| {
                (0..head)
                    .map(|r| grid[r][c].as_str())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        out.push(row(&joined));
    } else {
        out.push(row(&grid[0]));
    }
    out.push(format!("|{}|", vec![" --- "; t.cols].join("|")));
    for r in &grid[head..] {
        out.push(row(r));
    }
    out.join("\n")
}

/// The blocks as Markdown: titles and headings as `#` / `##`, list items as `-` or numbered
/// (consecutive ones one list), tables in place (pipe tables, HTML with spanning cells), code
/// fenced, the rest as paragraphs.
pub fn markdown(blocks: &[Block], tables: &[Table]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for b in blocks {
        let text = b.text.trim();
        let s = match (b.kind.as_str(), b.table) {
            (_, Some(ti)) => table_markdown(&tables[ti]),
            ("heading", _) => format!("{} {text}", "#".repeat(b.level.unwrap_or(2) as usize)),
            ("list_item", _) => match list_marker(text) {
                Some((m, rest)) => format!("{m} {rest}"),
                None => format!("- {text}"),
            },
            ("code", _) => format!("```\n{text}\n```"),
            _ if text.is_empty() => continue,
            _ => text.to_string(),
        };
        parts.push(s);
    }
    let mut md = String::new();
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            // consecutive list items stay one list
            let tight = |s: &str| list_marker(s).is_some() && !s.starts_with('#');
            md.push_str(if tight(p) && tight(&parts[i - 1]) { "\n" } else { "\n\n" });
        }
        md.push_str(p);
    }
    md
}
