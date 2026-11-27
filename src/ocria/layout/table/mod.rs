//! LiteParse's learned table and layout model (LayoutNet, `tablenet.bin` of OCRia-nano-1) on an
//! OCR page: text units -> tables with rows, columns, cells, spans and header rows, every unit's
//! block class and the page's layout blocks. Ported from LiteParse's `table_model` (Apache-2.0,
//! `crates/liteparse/src/table_model`), the parts an image page takes: its units are the page's
//! projected OCR items, with the projection's signals ([`super::projection::ProjSignals`]).

pub mod decode;
pub mod features;
pub mod net;
pub mod units;

use super::types::Rect;

/// Pages with more units than this skip the model (attention and decoding grow with N^2; such
/// pages are dense prose or indexes).
pub const MAX_UNITS: usize = 1500;

/// An OCR page's units and their projection signals, straight from the projected OCR items
/// (units.py `_lit_ocr_units`): one unit per item, box in page points (`wh`), height un-padded
/// (/1.4), leaf ids numbered in item order.
fn ocr_units(wh: (f64, f64), sig: &super::projection::ProjSignals) -> (Vec<units::Unit>, Vec<Option<features::Proj>>) {
    let (sx, sy) = (wh.0 / sig.width.max(1.0) as f64, wh.1 / sig.height.max(1.0) as f64);
    let r1 = |v: f32| (v * 10.0).round() as i64;
    let mut ids: Vec<&Vec<u16>> = Vec::new();
    let (mut us, mut proj) = (Vec::new(), Vec::new());
    for (item, it) in sig.items.iter().enumerate() {
        let t = it.text.trim();
        let [x, y, w, h] = it.bx;
        if t.is_empty() || w <= 0.0 || h <= 0.0 {
            continue;
        }
        let bx = [x as f64 * sx, y as f64 * sy, (x + w) as f64 * sx, (y + h) as f64 * sy];
        let path = sig
            .leaves
            .iter()
            .find(|(b, _)| r1(b[0]) == r1(x) && r1(b[1]) == r1(y))
            .map(|l| &l.1);
        let leaf = match path {
            Some(p) => match ids.iter().position(|q| *q == p) {
                Some(i) => i as f32,
                None => {
                    ids.push(p);
                    (ids.len() - 1) as f32
                }
            },
            None => -1.0,
        };
        us.push(units::Unit {
            bx,
            pbox: bx,
            text: t.to_string(),
            h: (bx[3] - bx[1]) / 1.4,
            item,
            // OCR word boxes (when the engine measured them): a line read across a ruled cell
            // boundary is cut there by the lattice
            words: if it.words.len() >= 2 {
                it.words
                    .iter()
                    .map(|(x0, x1, w)| (*x0 as f64 * sx, *x1 as f64 * sx, w.clone()))
                    .collect()
            } else {
                Vec::new()
            },
        });
        proj.push(Some(features::Proj {
            feat_text: it.faint.then(|| faint_mask(t)),
            flow: it.flow as u8 as f32,
            lc: it.lc as f32,
            rc: it.rc as f32,
            cc: it.cc as f32,
            snap: it.snap,
            block: it.block as f32,
            leaf,
        }));
    }
    (us, proj)
}

/// The text the table model's features read for an OCR faint-token read. LayoutNet reads a lone
/// value-like token ("1", "€1", "1.0") recovered between two cells as a reason to fuse the table's
/// rows (issue #615): the same token as an opaque word leaves the table intact. Tuned to the
/// LayoutNet of OCRia-nano-1 v1.0.0; a retrained model that has seen such tokens may not need it.
fn faint_mask(t: &str) -> String {
    "x".repeat(t.chars().count())
}

/// One unit's block class probabilities (layout models): its box on the page (points, top-left
/// origin), character count, and softmax over `BLOCK_CLASSES`.
#[derive(Clone, Debug)]
pub struct UnitClass {
    pub bbox: Rect,
    pub chars: usize,
    pub probs: Vec<f32>,
}

/// Block classes of a layout model's class head (build_grounded.py BLOCK_CLASSES).
pub const BLOCK_CLASSES: [&str; 13] = [
    "Text",
    "Heading",
    "Title",
    "PageHeader",
    "PageFooter",
    "List",
    "Table",
    "Caption",
    "Footnote",
    "Formula",
    "Code",
    "FigureText",
    "KeyValue",
];

/// A layout block from a joint model: units grouped by the same_block
/// head, labelled by their summed class probabilities.
#[derive(Clone, Debug)]
pub struct ModelBlock {
    pub bbox: Rect,
    /// index into BLOCK_CLASSES
    pub class: usize,
    /// the units' text in reading order
    pub text: String,
    /// the units, in reading order
    pub units: Vec<usize>,
}

/// What the model found on a page: its tables and (layout models) every unit's block class and
/// its layout blocks.
#[derive(Clone, Debug, Default)]
pub struct PageModel {
    pub tables: Vec<ModelTable>,
    pub classes: Vec<UnitClass>,
    pub blocks: Vec<ModelBlock>,
    /// the units the model read (the page's projected OCR items)
    pub units: Vec<units::Unit>,
    /// the units' features [n, feat_dim], node logits [n, 2] and class logits [n, n_cls]
    pub feats: Vec<f32>,
    pub node: Vec<f32>,
    pub cls: Vec<f32>,
}

/// A block's units split into its lines' runs: a new part starts where the vertical gap to the
/// next line is more than the block's median line height (a paragraph break; leading alone is a
/// fraction of it), or where the lines change between heading-like (the class head's Heading +
/// Title share over half, by characters) and body text, since the same_block head often joins
/// a heading to the paragraph under it.
fn split_paragraphs(g: &[usize], units: &[units::Unit], cls: &[f32], c: usize) -> Vec<Vec<usize>> {
    if g.len() < 2 || c < 3 {
        return vec![g.to_vec()];
    }
    let mut by_y: Vec<usize> = g.to_vec();
    by_y.sort_by(|&a, &b| {
        let (pa, pb) = (units[a].pbox, units[b].pbox);
        ((pa[1] + pa[3]) / 2.0).total_cmp(&((pb[1] + pb[3]) / 2.0))
    });
    // lines: units overlapping the current line by more than half the smaller height
    let mut lines: Vec<(f64, f64, Vec<usize>)> = Vec::new();
    for i in by_y {
        let p = units[i].pbox;
        match lines.last_mut() {
            Some(l) if (l.1.min(p[3]) - l.0.max(p[1])) > 0.5 * (l.1 - l.0).min(p[3] - p[1]) => {
                l.0 = l.0.min(p[1]);
                l.1 = l.1.max(p[3]);
                l.2.push(i);
            }
            _ => lines.push((p[1], p[3], vec![i])),
        }
    }
    if lines.len() < 2 {
        return vec![g.to_vec()];
    }
    let mut hs: Vec<f64> = lines.iter().map(|l| l.1 - l.0).collect();
    hs.sort_by(f64::total_cmp);
    let line_h = hs[hs.len() / 2].max(1.0);
    let headingish = |l: &[usize]| {
        let (mut head, mut all) = (0f32, 0f32);
        for &i in l {
            let row = &cls[i * c..(i + 1) * c];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
            let z: f32 = e.iter().sum();
            let n = units[i].text.chars().filter(|ch| !ch.is_whitespace()).count().max(1) as f32;
            head += n * (e[1] + e[2]) / z;
            all += n;
        }
        head * 2.0 > all
    };
    let mut parts: Vec<Vec<usize>> = vec![lines[0].2.clone()];
    for k in 1..lines.len() {
        let gap = lines[k].0 - lines[k - 1].1;
        if gap > line_h || headingish(&lines[k].2) != headingish(&lines[k - 1].2) {
            parts.push(Vec::new());
        }
        parts.last_mut().unwrap().extend_from_slice(&lines[k].2);
    }
    parts
}

/// A block's units split where their horizontal extents leave a gap wider than the units' median
/// height (about three word spaces), keeping the original order inside each part.
fn split_at_gaps(g: &[usize], units: &[units::Unit]) -> Vec<Vec<usize>> {
    let mut hs: Vec<f64> = g.iter().map(|&i| units[i].pbox[3] - units[i].pbox[1]).collect();
    hs.sort_by(f64::total_cmp);
    let gap = hs.get(hs.len() / 2).copied().unwrap_or(0.0).max(1.0);
    let mut by_x: Vec<usize> = g.to_vec();
    by_x.sort_by(|&a, &b| units[a].pbox[0].total_cmp(&units[b].pbox[0]));
    let mut parts: Vec<Vec<usize>> = Vec::new();
    let mut right = f64::NEG_INFINITY;
    for i in by_x {
        let p = units[i].pbox;
        if parts.is_empty() || p[0] - right > gap {
            parts.push(Vec::new());
        }
        parts.last_mut().unwrap().push(i);
        right = right.max(p[2]);
    }
    parts
}

/// Units grouped into blocks from same_block logits: connected components of positive links,
/// then average linkage inside each component (average linkage over a whole dense page is
/// O(n^3); its components are small).
fn block_groups(l: &net::Logits, keep: &[usize]) -> Vec<Vec<usize>> {
    let n = l.n;
    let m = keep.len();
    let mut parent: Vec<usize> = (0..m).collect();
    fn root(p: &mut [usize], mut k: usize) -> usize {
        while p[k] != k {
            p[k] = p[p[k]];
            k = p[k];
        }
        k
    }
    for a in 0..m {
        for b in a + 1..m {
            if l.pair[(4 * n + keep[a]) * n + keep[b]] > 0.0 {
                let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }
    let mut comps: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for a in 0..m {
        let r = root(&mut parent, a);
        comps.entry(r).or_default().push(a);
    }
    let mut out = Vec::new();
    for (_, c) in comps {
        let k = c.len();
        let mut p = vec![0f64; k * k];
        for x in 0..k {
            for y in 0..k {
                p[x * k + y] = l.pair[(4 * n + keep[c[x]]) * n + keep[c[y]]] as f64;
            }
        }
        for g in decode::avg_link(&p, k, 0.0) {
            out.push(g.into_iter().map(|x| keep[c[x]]).collect());
        }
    }
    out
}

/// A decoded table: its grid and its box on the page (points, top-left origin).
#[derive(Clone, Debug)]
pub struct ModelTable {
    pub bbox: Rect,
    pub grid: decode::Grid,
    /// the units the table holds
    pub units: Vec<usize>,
}

/// The model on one OCR page (LiteParse's `run_page` on a page whose units come from OCR):
/// features, forward pass, table decoding, block classes and layout blocks. `wh` is the page size
/// in points, `rules` its ruling lines and `page_img` the image branch's input (both from the
/// page render, [`features::process_raster`]).
pub fn run_page(
    net: &net::TableNet,
    wh: (f64, f64),
    sig: &super::projection::ProjSignals,
    rules: Option<&features::Rules>,
    page_img: Option<&features::PageImage>,
    opt: &decode::DecodeOptions,
) -> PageModel {
    let (units, proj) = ocr_units(wh, sig);
    let n = units.len();
    if !(2..=MAX_UNITS).contains(&n) || units.iter().map(|u| u.text.chars().count()).sum::<usize>() < 20 {
        return PageModel {
            units,
            ..Default::default()
        };
    }
    let f32x4 = |b: [f64; 4]| [b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32];
    let fu: Vec<features::FUnit> = units
        .iter()
        .zip(proj)
        .map(|(u, p)| features::FUnit {
            bx: f32x4(u.bx),
            pbox: f32x4(u.pbox),
            text: u.text.clone(),
            h: u.h as f32,
            bold: None,
            proj: p,
        })
        .collect();
    let whf = (wh.0 as f32, wh.1 as f32);
    let feats = features::unit_features(&fu, whf, rules);
    let pairf = match rules {
        Some(r) => features::pair_rule_features(&fu.iter().map(|u| u.pbox).collect::<Vec<_>>(), r),
        None => vec![0f32; n * n * 2],
    };
    let img = page_img
        .filter(|_| net.has_img())
        .map(|im| im.input(&fu.iter().map(|u| u.pbox).collect::<Vec<_>>()));
    let t0 = crate::ocria::stage_start();
    let logits = net.forward(&feats, &pairf, n, img.as_ref());
    crate::ocria::stage("layout: network", t0);
    let du: Vec<decode::DUnit> = units
        .iter()
        .map(|u| decode::DUnit {
            bx: u.bx,
            text: u.text.clone(),
            h: u.h,
        })
        .collect();
    let mut out = Vec::new();
    for g in decode::decode_tables(&du, &logits, opt) {
        if g.len() < 2 {
            continue;
        }
        let mut grid = decode::decode_grid(&du, &g, &logits, opt);
        let mut ruled = false;
        if let Some(r) = rules {
            let words: Vec<&[(f64, f64, String)]> = units.iter().map(|u| u.words.as_slice()).collect();
            match grid.lattice(&r.h, &r.v, &du, &words, &g) {
                Some(l) => {
                    grid = l;
                    ruled = true;
                }
                None => grid.merge_unruled_rows(&r.h),
            }
        }
        // one column is a list or a stack of lines, not a table (a signature block's lines
        // grouped into a 1-wide grid); tables of contents, charts and multi-column prose read as
        // grids are no tables either (a grid drawn in ruling lines is no table of contents)
        if grid.n_cols < 2 || (!ruled && grid.is_toc()) || grid.is_chart() || grid.is_prose() {
            continue;
        }
        let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for &i in &g {
            let p = units[i].pbox;
            b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[2]), b[3].max(p[3])];
        }
        let bbox = Rect {
            x: b[0] as f32,
            y: b[1] as f32,
            width: (b[2] - b[0]) as f32,
            height: (b[3] - b[1]) as f32,
        };
        out.push(ModelTable { bbox, grid, units: g });
    }
    let cl = &logits;
    let mut classes = Vec::new();
    if cl.n_cls > 0 {
        let c = cl.n_cls;
        for (i, u) in units.iter().enumerate() {
            let z = &cl.cls[i * c..(i + 1) * c];
            let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
            let sum: f32 = e.iter().sum();
            let p = u.pbox;
            classes.push(UnitClass {
                bbox: Rect {
                    x: p[0] as f32,
                    y: p[1] as f32,
                    width: (p[2] - p[0]) as f32,
                    height: (p[3] - p[1]) as f32,
                },
                chars: u.text.chars().count(),
                probs: e.iter().map(|v| v / sum).collect(),
            });
        }
    }
    let mut blocks = Vec::new();
    if cl.n_pair == 5 && cl.n_cls > 0 {
        let keep: Vec<usize> = (0..n).collect();
        let c = cl.n_cls;
        for g in block_groups(cl, &keep) {
            let mut sum = vec![0f32; c];
            for &i in &g {
                for (k, v) in cl.cls[i * c..(i + 1) * c].iter().enumerate() {
                    sum[k] += v;
                }
            }
            let class = (0..c).max_by(|&a, &b| sum[a].total_cmp(&sum[b])).unwrap_or(0);
            // Groups split at wide horizontal gaps: running headers and footers are rows of
            // separate items (page number, title, section tabs) that the same_block head joins
            // into one page-wide block, and side-by-side text blocks get joined the same way;
            // also at paragraph breaks and heading / body changes.
            let parts: Vec<Vec<usize>> = split_at_gaps(&g, &units)
                .into_iter()
                .flat_map(|part| split_paragraphs(&part, &units, &cl.cls, c))
                .collect();
            for g in parts {
                // each part takes its own class
                let class = {
                    let mut s = vec![0f32; c];
                    for &i in &g {
                        for (k, v) in cl.cls[i * c..(i + 1) * c].iter().enumerate() {
                            s[k] += v;
                        }
                    }
                    (0..c).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap_or(class)
                };
                let boxes: Vec<[f64; 4]> = g.iter().map(|&i| units[i].bx).collect();
                let order: Vec<usize> = decode::reading_order(&boxes).into_iter().map(|k| g[k]).collect();
                let text = order.iter().map(|&k| units[k].text.as_str()).collect::<Vec<_>>().join(" ");
                let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
                for &i in &g {
                    let p = units[i].pbox;
                    b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[2]), b[3].max(p[3])];
                }
                blocks.push(ModelBlock {
                    bbox: Rect {
                        x: b[0] as f32,
                        y: b[1] as f32,
                        width: (b[2] - b[0]) as f32,
                        height: (b[3] - b[1]) as f32,
                    },
                    class,
                    text,
                    units: order,
                });
            }
        }
    }
    let (node, cls) = (logits.node.clone(), logits.cls.clone());
    PageModel {
        tables: out,
        classes,
        blocks,
        units,
        feats,
        node,
        cls,
    }
}

impl decode::Grid {
    /// A table of contents: at most three columns, and most rows end in a page number (arabic or
    /// roman) with the arabic ones non-decreasing down the table. Rendered as text, not a table.
    pub fn is_toc(&self) -> bool {
        if self.n_cols > 3 {
            return false;
        }
        let mut rows: Vec<Vec<(usize, &str)>> = vec![Vec::new(); self.n_rows];
        for c in &self.cells {
            let t = c.text.trim();
            if !t.is_empty() {
                rows[c.r0].push((c.c0, t));
            }
        }
        let rows: Vec<_> = rows.into_iter().filter(|r| !r.is_empty()).collect();
        let mut hits = 0;
        let mut arabic = Vec::new();
        for r in &rows {
            let last = r.iter().max_by_key(|(q, _)| *q).unwrap().1;
            let roman = last.len() <= 7 && last.chars().all(|ch| "ivxlcdmIVXLCDM".contains(ch));
            let num = last.len() <= 4 && last.chars().all(|ch| ch.is_ascii_digit());
            if r.len() >= 2 && (roman || num) {
                hits += 1;
                if num {
                    arabic.push(last.parse::<u32>().unwrap_or(0));
                }
            }
        }
        let rising = arabic.windows(2).filter(|w| w[1] < w[0]).count() <= arabic.len() / 10;
        hits >= 3 && hits * 5 >= rows.len() * 3 && rising
    }

    /// Multi-column prose read as a table: long, mostly non-numeric cells whose text runs on
    /// down each column, a cell that doesn't end a sentence followed by one that starts
    /// lowercase ("calendar 2024, more than 70%" / "will be from recycled content"). A real
    /// table's cells are independent, so its columns rarely flow even when one column is long.
    pub fn is_prose(&self) -> bool {
        let words = |t: &str| t.split_whitespace().count();
        let cells: Vec<&decode::GridCell> = self.cells.iter().filter(|c| !c.text.trim().is_empty()).collect();
        if cells.len() < 6 {
            return false;
        }
        let mut wc: Vec<usize> = cells.iter().map(|c| words(&c.text)).collect();
        wc.sort_unstable();
        if wc[wc.len() / 2] < 4 {
            return false;
        }
        let numeric = cells
            .iter()
            .filter(|c| words(&c.text) <= 3 && c.text.chars().any(|ch| ch.is_ascii_digit()))
            .count();
        if numeric * 10 >= cells.len() * 3 {
            return false;
        }
        let mut cols: std::collections::BTreeMap<usize, Vec<&decode::GridCell>> = Default::default();
        for c in &cells {
            if c.c0 == c.c1 {
                cols.entry(c.c0).or_default().push(c);
            }
        }
        let (mut pairs, mut flow) = (0usize, 0usize);
        for col in cols.values_mut() {
            col.sort_by_key(|c| c.r0);
            for w in col.windows(2) {
                if w[1].r0 > w[0].r1 + 2 {
                    continue; // not adjacent: more than one empty row between them
                }
                pairs += 1;
                let up = w[0].text.trim_end();
                let open = !up.ends_with(['.', '!', '?', ':', ';']);
                let lower = w[1].text.trim_start().chars().next().is_some_and(|ch| ch.is_lowercase());
                flow += (open && lower) as usize;
            }
        }
        pairs >= 3 && flow * 10 >= pairs * 4
    }

    /// A chart read as a table: a sparse, mostly numeric grid with an axis - some column holding
    /// at least three lone numbers in an evenly stepped, monotonic run (the tick labels).
    pub fn is_chart(&self) -> bool {
        let slots = self.n_rows * self.n_cols;
        if slots == 0 {
            return false;
        }
        let parse = |t: &str| -> Option<f64> {
            let t = t.trim().trim_end_matches('%').replace(',', "");
            if t.is_empty() || !t.chars().all(|ch| ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == '+') {
                return None;
            }
            t.parse::<f64>().ok()
        };
        let (mut filled, mut nonempty, mut numeric) = (0, 0, 0);
        let mut cols: std::collections::BTreeMap<usize, Vec<(usize, f64)>> = Default::default();
        let mut flow: Vec<((usize, usize), f64)> = Vec::new();
        for c in &self.cells {
            if c.text.trim().is_empty() {
                continue;
            }
            nonempty += 1;
            filled += (c.r1 - c.r0 + 1) * (c.c1 - c.c0 + 1);
            if let Some(v) = parse(&c.text) {
                numeric += 1;
                if c.c0 == c.c1 {
                    cols.entry(c.c0).or_default().push((c.r0, v));
                }
                flow.push(((c.r0, c.c0), v));
            }
        }
        if filled * 2 >= slots || numeric * 2 < nonempty {
            return false;
        }
        let stepped = |vs: &[f64]| {
            vs.len() >= 3 && {
                let d: Vec<f64> = vs.windows(2).map(|w| w[1] - w[0]).collect();
                d[0] != 0.0 && d.iter().all(|&x| (x - d[0]).abs() <= 1e-6 * d[0].abs().max(1.0))
            }
        };
        // a column of ticks (y axis), or every number on the grid in reading order (an x axis of
        // years strewn diagonally, one per row)
        flow.sort_by_key(|&(rc, _)| rc);
        let all: Vec<f64> = flow.iter().map(|&(_, v)| v).collect();
        stepped(&all)
            || cols.values_mut().any(|vs| {
                vs.sort_by_key(|&(r, _)| r);
                stepped(&vs.iter().map(|&(_, v)| v).collect::<Vec<_>>())
            })
    }
}

/// The model from its weights (`tablenet.bin`).
pub fn load_bytes(bytes: &[u8]) -> anyhow::Result<net::TableNet> {
    net::TableNet::from_bytes(bytes).map_err(|e| anyhow::anyhow!("tablenet.bin: {e}"))
}
