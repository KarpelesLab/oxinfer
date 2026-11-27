//! OCR lines into a page's text items, as LiteParse merges an OCR outcome into a page with no text
//! layer (`crates/liteparse/src/ocr_merge.rs`, `merge_ocr_outcomes`, Apache-2.0): readings at or
//! under 0.1 dropped, short fragments in a script the page doesn't use dropped, table-border
//! characters stripped around numbers, boxes in points (`72 / dpi` a pixel), a skewed scan's items
//! turned upright about the page centre, the font size from the quad's height, word boxes, and a
//! page whose text all runs one way other than upright re-expressed upright.

use super::types::{Page, TextItem, WordBox};
use std::collections::{HashMap, HashSet};

/// One OCR reading in image pixels (LiteParse's `OcrResult` from ocria-nano).
#[derive(Clone, Debug)]
pub struct OcrResult {
    pub text: String,
    /// [x1, y1, x2, y2]
    pub bbox: [f32; 4],
    pub confidence: f32,
    /// top-left, top-right, bottom-right, bottom-left in the glyphs' upright reading frame
    pub polygon: Option<[[f32; 2]; 4]>,
    /// word boxes [x1, y1, x2, y2] with their text, in reading order
    pub words: Vec<(String, [f32; 4])>,
    pub faint: bool,
}

/// Tilt range a page is deskewed over: below it rows already group correctly, above it the
/// "tilt" is more likely rotated or curved text than a skewed scan.
const DESKEW_MIN_DEG: f32 = 0.3;
const DESKEW_MAX_DEG: f32 = 10.0;

/// Share of a page's OCR text (by characters) that must share one quarter-turn rotation for the
/// page to be read as a scan turned on its side or upside down.
const ROTATED_SCAN_MIN_SHARE: f32 = 0.7;

/// The page (`width` x `height` points) with `results` (pixels at `dpi`) merged in.
pub fn ocr_page(results: &[OcrResult], dpi: f32, width: f32, height: f32) -> Page {
    let mut page = Page {
        page_number: 1,
        ocr_deskew: None,
        page_width: width,
        page_height: height,
        text_items: Vec::new(),
        turned: 0,
    };
    if results.is_empty() {
        return page;
    }
    let scale_factor = 72.0 / dpi;
    // the page's scripts: any script its OCR reads repeatedly (in 3+ results or 20+ letters, at
    // any confidence). Icon and texture junk reads as one-off fragments in random scripts; real
    // text in a script, however weakly read, recurs
    let page_scripts: HashSet<u32> = {
        let (mut res, mut letters): (HashMap<u32, usize>, HashMap<u32, usize>) = (HashMap::new(), HashMap::new());
        for r in results {
            let mut seen: HashSet<u32> = HashSet::new();
            for c in r.text.chars().filter(|c| c.is_alphabetic()) {
                let sc = script_of(c);
                *letters.entry(sc).or_default() += 1;
                seen.insert(sc);
            }
            for sc in seen {
                *res.entry(sc).or_default() += 1;
            }
        }
        letters
            .iter()
            .filter(|&(sc, &k)| k >= 20 || res.get(sc).copied().unwrap_or(0) >= 3)
            .map(|(&sc, _)| sc)
            .collect()
    };
    // A skewed scan: rotate the items upright about the page centre (axis-aligned boxes of
    // tilted lines overlap the lines above and below). The output boxes are rotated back later.
    let deskew = ocr_page_skew(results);
    page.ocr_deskew = deskew;
    let (cx, cy) = (width / 2.0, height / 2.0);
    let upright = |x: f32, y: f32| -> (f32, f32) {
        match deskew {
            Some(deg) => rotate_about(x, y, cx, cy, -deg),
            None => (x, y),
        }
    };
    // An axis-aligned pixel box in page points. Under a deskew the box moves with its centre and
    // keeps its size.
    let to_page = |b: [f32; 4]| -> (f32, f32, f32, f32) {
        let (w, h) = ((b[2] - b[0]) * scale_factor, (b[3] - b[1]) * scale_factor);
        match deskew {
            Some(_) => {
                let (mx, my) = upright((b[0] + b[2]) / 2.0 * scale_factor, (b[1] + b[3]) / 2.0 * scale_factor);
                (mx - w / 2.0, my - h / 2.0, w, h)
            }
            None => (b[0] * scale_factor, b[1] * scale_factor, w, h),
        }
    };
    // Page-point bounds of a pixel box, or of its quad. Without a deskew the quad's points stay in
    // image pixels until the bounds are taken, as LiteParse's arithmetic.
    let page_bounds = |bbox: [f32; 4], poly: Option<[[f32; 2]; 4]>| {
        let Some(poly) = poly else {
            return to_page(bbox);
        };
        let (s, pts) = match deskew {
            Some(_) => (1.0, poly.map(|[x, y]| upright(x * scale_factor, y * scale_factor))),
            None => (scale_factor, poly.map(|[x, y]| (x, y))),
        };
        let x_min = pts.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
        let x_max = pts.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
        let y_min = pts.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let y_max = pts.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
        (x_min * s, y_min * s, (x_max - x_min) * s, (y_max - y_min) * s)
    };
    for r in results {
        if r.confidence <= 0.1 || is_foreign_fragment(&r.text, &page_scripts) {
            continue;
        }
        let (ocr_x, ocr_y, ocr_w, ocr_h) = page_bounds(r.bbox, r.polygon);
        let rotation = r.polygon.map_or(0.0, |p| polygon_rotation_deg(&p));
        let cleaned = clean_ocr_table_artifacts(&r.text);
        if cleaned.is_empty() {
            continue;
        }
        // A tilted line's axis-aligned box is taller than its glyphs; the polygon's own side is
        // the line height. Rotated text takes the perpendicular extent.
        let poly_h = r.polygon.map(|p| (p[3][0] - p[0][0]).hypot(p[3][1] - p[0][1]) * scale_factor);
        let font_size_hint = if rotation == 90.0 || rotation == 270.0 {
            ocr_w.max(1.0)
        } else {
            poly_h.filter(|h| *h > 0.0).map_or(ocr_h, |h| h.min(ocr_h))
        };
        let word = |text: &str, (x, y, width, height): (f32, f32, f32, f32)| WordBox {
            text: text.to_string(),
            x,
            y,
            width,
            height,
        };
        // A count mismatch after cleaning keeps the line's own box.
        let words = if cleaned.split_whitespace().count() == r.words.len() {
            cleaned
                .split_whitespace()
                .zip(&r.words)
                .map(|(text, measured)| word(text, to_page(measured.1)))
                .collect()
        } else {
            vec![word(&cleaned, to_page(r.bbox))]
        };
        page.text_items.push(TextItem {
            text: cleaned,
            x: ocr_x,
            y: ocr_y,
            width: ocr_w,
            height: ocr_h,
            rotation,
            font_name: Some("OCR".to_string()),
            font_size: Some(font_size_hint),
            confidence: Some((r.confidence * 1000.0).round() / 1000.0),
            words,
            ocr_faint: r.faint,
            ..Default::default()
        });
    }
    page.turned = upright_rotated_scan(&mut page);
    page
}

/// A scan whose text all runs one way other than upright: every item re-expressed in the upright
/// page's frame with its rotation cleared (item boxes, and for quarter turns the page size, are
/// then in the upright frame). Returns the quarter turns (0 when left as is).
fn upright_rotated_scan(page: &mut Page) -> u8 {
    let mut chars = [0usize; 4];
    for it in &page.text_items {
        chars[(canonical_quarter(it.rotation) / 90) as usize] += it.text.chars().count();
    }
    let total: usize = chars.iter().sum();
    let Some((q, &top)) = chars.iter().enumerate().max_by_key(|&(_, c)| *c) else {
        return 0;
    };
    if q == 0 || total < 20 || (top as f32) < total as f32 * ROTATED_SCAN_MIN_SHARE {
        return 0;
    }
    let (w, h) = (page.page_width, page.page_height);
    let turn = |(x, y, bw, bh): (f32, f32, f32, f32)| match q {
        // reading bottom-to-top: the page's top edge is on the left
        1 => (h - (y + bh), x, bh, bw),
        2 => (w - (x + bw), h - (y + bh), bw, bh),
        // reading top-to-bottom: the page's top edge is on the right
        _ => (y, w - (x + bw), bh, bw),
    };
    for it in &mut page.text_items {
        (it.x, it.y, it.width, it.height) = turn((it.x, it.y, it.width, it.height));
        it.rotation = 0.0;
        for wb in &mut it.words {
            (wb.x, wb.y, wb.width, wb.height) = turn((wb.x, wb.y, wb.width, wb.height));
        }
    }
    if q != 2 {
        (page.page_width, page.page_height) = (h, w);
    }
    q as u8
}

/// 0, 90, 180 or 270 for an item rotation in degrees.
fn canonical_quarter(rot: f32) -> u32 {
    ((rot.rem_euclid(360.0) / 90.0).round() as u32 % 4) * 90
}

/// The page's tilt in degrees (image frame, y down: positive = lines descend to the right), as
/// the length-weighted median angle of its long, near-horizontal OCR lines (length >= 4x height).
/// `None` below three such lines or outside the deskew range.
fn ocr_page_skew(results: &[OcrResult]) -> Option<f32> {
    let mut lines: Vec<(f32, f32)> = results
        .iter()
        .filter_map(|r| {
            let p = r.polygon?;
            let (dx, dy) = (p[1][0] - p[0][0], p[1][1] - p[0][1]);
            let len = dx.hypot(dy);
            let h = (p[3][0] - p[0][0]).hypot(p[3][1] - p[0][1]);
            let deg = dy.atan2(dx).to_degrees();
            (len >= 4.0 * h && len > 0.0 && deg.abs() <= 45.0).then_some((deg, len))
        })
        .collect();
    if lines.len() < 3 {
        return None;
    }
    lines.sort_by(|a, b| a.0.total_cmp(&b.0));
    let half = lines.iter().map(|l| l.1).sum::<f32>() / 2.0;
    let mut acc = 0.0;
    let mid = lines
        .iter()
        .find(|l| {
            acc += l.1;
            acc >= half
        })
        .map_or(0.0, |l| l.0);
    (DESKEW_MIN_DEG..=DESKEW_MAX_DEG).contains(&mid.abs()).then_some(mid)
}

/// Rotate (x, y) by `deg` degrees about (cx, cy), in the y-down page frame.
pub fn rotate_about(x: f32, y: f32, cx: f32, cy: f32, deg: f32) -> (f32, f32) {
    let (s, c) = deg.to_radians().sin_cos();
    let (dx, dy) = (x - cx, y - cy);
    (cx + dx * c - dy * s, cy + dx * s + dy * c)
}

/// A script bucket for a letter (Latin 0, CJK 1, other scripts by block, everything else 2).
fn script_of(c: char) -> u32 {
    let o = c as u32;
    match o {
        0..=0x24F | 0x1E00..=0x1EFF | 0x2C60..=0x2C7F | 0xA720..=0xA7FF => 0,
        0x1100..=0x11FF | 0x3000..=0x30FF | 0x3400..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF | 0xFF00..=0xFFEF => 1,
        // Scripts whose letters span several blocks are one script each: polytonic Greek and
        // Coptic, Cyrillic extensions, Hebrew / Arabic presentation forms, Ethiopic, Georgian.
        0x0370..=0x03FF | 0x1F00..=0x1FFF | 0x2C80..=0x2CFF => 0x370,
        0x0400..=0x052F | 0x1C80..=0x1C8F | 0x2DE0..=0x2DFF | 0xA640..=0xA69F => 0x400,
        0x0590..=0x05FF | 0xFB1D..=0xFB4F => 0x590,
        0x0600..=0x06FF | 0x0750..=0x077F | 0x0860..=0x08FF | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => 0x600,
        0x1200..=0x139F | 0x2D80..=0x2DDF | 0xAB00..=0xAB2F => 0x1200,
        0x10A0..=0x10FF | 0x1C90..=0x1CBF | 0x2D00..=0x2D2F => 0x10A0,
        0x1780..=0x17FF | 0x19E0..=0x19FF => 0x1780,
        0x1000..=0x109F | 0xA9E0..=0xA9FF | 0xAA60..=0xAA7F => 0x1000,
        0x0530..=0x058F | 0xFB13..=0xFB17 => 0x530,
        0x0700..=0x074F => 0x700,
        // Indic scripts sit in exact 128-codepoint blocks
        0x0900..=0x0DFF => o & !0x7F,
        0x0E00..=0x0E7F => 0xE00,
        0x0E80..=0x0EFF => 0xE80,
        0x0F00..=0x0FFF => 0xF00,
        // anything else is one bucket: never counted as a script mix
        _ => 2,
    }
}

/// OCR junk by script, for a short reading (up to 12 letters) mostly in a script the page doesn't
/// use (`page`): one with a token mixing letters of two or more non-Latin scripts, or one of up to
/// 8 letters. Latin is never foreign.
fn is_foreign_fragment(text: &str, page: &HashSet<u32>) -> bool {
    let letters: Vec<u32> = text.chars().filter(|c| c.is_alphabetic()).map(script_of).collect();
    if letters.is_empty() || letters.len() > 12 {
        return false;
    }
    let mut n: HashMap<u32, usize> = HashMap::new();
    for &s in &letters {
        *n.entry(s).or_default() += 1;
    }
    let Some((script, k)) = n.into_iter().max_by_key(|&(s, k)| (k, s)) else {
        return false;
    };
    // text in a script the page uses stays, misreads and all
    if script == 0 || script == 2 || page.contains(&script) {
        return false;
    }
    let mixed = text.split_whitespace().any(|tok| {
        let scripts: HashSet<u32> = tok
            .chars()
            .filter(|c| c.is_alphabetic())
            .map(script_of)
            .filter(|&s| s != 0 && s != 2)
            .collect();
        scripts.len() >= 2 && !(scripts.len() == 2 && scripts.contains(&1))
    });
    mixed || (letters.len() <= 8 && k * 2 >= letters.len() && !page.is_empty())
}

/// The quarter-turn rotation (counter-clockwise degrees) of a reading-frame quad: its longer side
/// is the reading axis; near-square quads are upright.
fn polygon_rotation_deg(poly: &[[f32; 2]; 4]) -> f32 {
    let e0 = [poly[1][0] - poly[0][0], poly[1][1] - poly[0][1]];
    let e1 = [poly[2][0] - poly[1][0], poly[2][1] - poly[1][1]];
    let len0 = (e0[0] * e0[0] + e0[1] * e0[1]).sqrt();
    let len1 = (e1[0] * e1[0] + e1[1] * e1[1]).sqrt();
    if len0.max(len1) < 1.0 {
        return 0.0;
    }
    let (longer, shorter) = if len0 >= len1 { (len0, len1) } else { (len1, len0) };
    if shorter > 0.0 && longer / shorter < 1.3 {
        return 0.0;
    }
    let reading = if len0 >= len1 { e0 } else { e1 };
    // atan2 with screen-down y; negate to get the conventional CCW angle.
    canonical_quarter(-reading[1].atan2(reading[0]).to_degrees()) as f32
}

/// Border characters (`| [ ] ( ) { }`) read around a numeric cell value are stripped; a negative
/// number's parentheses stay.
fn clean_ocr_table_artifacts(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let without_artifacts: &str = trimmed
        .trim_start_matches(['|', '[', ']', '(', ')', '{', '}'])
        .trim_end_matches(['|', '[', ']', '(', ')', '{', '}'])
        .trim();
    if without_artifacts.is_empty() {
        return trimmed.to_string();
    }
    let is_numeric_ish = without_artifacts
        .chars()
        .all(|c| c.is_ascii_digit() || matches!(c, ',' | '.' | ' ' | '%' | '-' | '+' | '*' | '/'))
        || without_artifacts == "N/A"
        || without_artifacts == "Z"
        || without_artifacts == "-";
    if is_numeric_ish {
        if let Some((before, after)) = trimmed.split_once(without_artifacts)
            && before.ends_with('(')
            && after.starts_with(')')
        {
            return format!("({without_artifacts})");
        }
        without_artifacts.to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_artifacts_and_rotation() {
        assert_eq!(clean_ocr_table_artifacts("|12.5|"), "12.5");
        assert_eq!(clean_ocr_table_artifacts("(78,939)|"), "(78,939)");
        assert_eq!(clean_ocr_table_artifacts("(note)"), "(note)");
        let up = [[0.0, 0.0], [100.0, 0.0], [100.0, 20.0], [0.0, 20.0]];
        assert_eq!(polygon_rotation_deg(&up), 0.0);
        let ccw = [[0.0, 100.0], [0.0, 0.0], [20.0, 0.0], [20.0, 100.0]];
        assert_eq!(polygon_rotation_deg(&ccw), 90.0);
        assert!(is_foreign_fragment("नम", &HashSet::from([0])));
        assert!(!is_foreign_fragment("hello", &HashSet::new()));
    }
}
