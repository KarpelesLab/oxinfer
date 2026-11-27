//! The page and text-item types LiteParse's projection works on (`crates/liteparse/src/types.rs`,
//! Apache-2.0), the fields an OCR page carries. They (de)serialize as LiteParse's do, so a page
//! dumped by LiteParse can be projected here as it is.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a single text item extracted from a PDF page,
/// including its content, position, size, rotation, and font metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextItem {
    pub text: String,
    /// Viewport-space coordinates (top-left origin, 72 DPI).
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    /// Rotation in degrees (counter-clockwise, adjusted for page rotation).
    pub rotation: f32,
    pub font_name: Option<String>,
    pub font_size: Option<f32>,
    /// Font size * scale_y from the text matrix — accounts for CTM scaling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_height: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_ascent: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_descent: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_weight: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_flags: Option<i32>,
    /// Sum of glyph widths (using charcode-based lookup when possible).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_width: Option<f32>,
    /// Whether the font has buggy encoding (private-use codepoints, TT subset, etc.)
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub font_is_buggy: bool,
    /// Whether most characters in this item could not be mapped to Unicode
    /// (e.g. a Type3 font with no ToUnicode map). The text content is
    /// PDFium's char-code fallback and does not reflect the rendered glyphs.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub has_unicode_map_error: bool,
    /// Marked content ID from the PDF structure tree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcid: Option<i32>,
    /// Fill color as ARGB hex string (e.g. "ff000000").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill_color: Option<String>,
    /// Stroke color as ARGB hex string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_color: Option<String>,
    /// Raw character codes from the PDF content stream. These correspond to
    /// source glyphs rather than Unicode scalar values, so ligature expansion
    /// can produce more text characters than entries in this array.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub char_codes: Vec<u32>,
    /// Whether the trailing source space was synthesized by PDFium rather than
    /// represented by a real space glyph in the PDF content stream.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub trailing_space_generated: bool,
    /// A space (a glyph, or one PDFium synthesized from the glyph gap) followed
    /// the last glyph of this left-to-right item. Projection keeps it when it
    /// rejoins the item with an abutting neighbour.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub followed_by_space: bool,
    /// OCR confidence score (0.0–1.0). None for native PDF text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Target URI when this item falls inside a hyperlink annotation's
    /// rectangle. Populated in `extract.rs`; consumed by the markdown emitter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<String>,
    /// Whether a thin horizontal stroke/rect crosses this item's vertical middle
    /// band (a strikethrough line). Populated in `extract.rs`; consumed by the
    /// markdown emitter to wrap the text in `~~…~~`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub strike: bool,
    /// Whether a thin horizontal stroke/rect runs just under this item's
    /// baseline and spans its text (an underline). Populated in `extract.rs`;
    /// the markdown emitter wraps the text in `<u>…</u>`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub underline: bool,
    /// Raised (superscript) or lowered (subscript) small text relative to its
    /// line, e.g. a footnote marker. Set in projection; the markdown emitter
    /// wraps it in `<sup>…</sup>` / `<sub>…</sub>`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub sup: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub sub: bool,
    /// The pieces an item was merged from, when their emphasis differs (bold / italic / struck /
    /// monospace / underline / script): the markdown renderer styles each piece, while lines,
    /// tables and blocks see the one merged item. Empty otherwise. Internal.
    #[serde(skip)]
    pub style_pieces: Vec<TextItem>,
    /// A super/subscript run split from its line at a size step: its own glyph box `(y, height)`.
    /// `y` / `height` then hold the host text's band, so lines and tables group the run with its
    /// host as before the split; script detection reads this box. Internal.
    #[serde(skip)]
    pub script_geom: Option<(f32, f32)>,
    /// Whether this item is LaTeX rebuilt from TeX math fonts (`$…$`).
    /// Populated in `extract.rs`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub math: bool,
    /// Per-word sub-boxes within this item, split on the inter-word spaces seen
    /// during segment building. A segment groups several words together (it only
    /// breaks at line/column boundaries), so this exposes the finer word-level
    /// geometry needed for bbox attribution. Empty for items that produced no
    /// word split (e.g. OCR-sourced or single-token items). Internal/attribution
    /// use only — the public JSON output (`output/json.rs`) builds its own
    /// item view without it, but it is marshalled across the napi boundary
    /// and must survive a stage boundary (see `stages`), so it serializes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<WordBox>,
    /// Read by ocria-nano's faint-token pass (see `OcrResult::faint`); internal only.
    #[serde(skip)]
    pub ocr_faint: bool,
}

/// One word's bounding box within a `TextItem`, in the same viewport space
/// (top-left origin, 72 DPI) as the parent item. `text` is the word's content
/// with inter-word spaces excluded.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WordBox {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// A page before projection: its size in points (top-left origin, 72 per inch) and text items.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Page {
    pub page_number: usize,
    /// Tilt (degrees) the OCR merge removed from this page's OCR items: on a skewed scan the
    /// items are rotated upright about the page centre so lines group into rows correctly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocr_deskew: Option<f32>,
    pub page_width: f32,
    pub page_height: f32,
    pub text_items: Vec<TextItem>,
    /// Quarter turns the OCR merge turned a sideways scan's items upright by (oxinfer's: maps
    /// boxes back to the image).
    #[serde(skip)]
    pub turned: u8,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    /// Smallest rect containing both inputs. Used by the block classifier to
    /// grow a block's bbox as source lines are appended to it (paragraph
    /// accumulation, wrapped headings, list-item continuations) and to merge
    /// blocks that are spliced together after construction.
    pub fn union(&self, other: &Rect) -> Rect {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = (self.x + self.width).max(other.x + other.width);
        let bottom = (self.y + self.height).max(other.y + other.height);
        Rect {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    }

    /// Fold `rect` into an optional accumulator. `None` adopts the rect; `Some`
    /// unions. Blocks whose sources carry no geometry stay `None` rather than
    /// reporting a bogus rect anchored at the origin.
    pub fn extend(acc: &mut Option<Rect>, rect: &Rect) {
        *acc = Some(match acc {
            Some(existing) => existing.union(rect),
            None => rect.clone(),
        });
    }
}

/// Per-line structural metadata derived during grid projection. Used by the
/// markdown emitter; not surfaced in JSON/text output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectedLine {
    pub text: String,
    pub bbox: Rect,
    pub anchor: Anchor,
    pub indent_x: f32,
    pub dominant_font_size: f32,
    /// True when `dominant_font_size` was derived from bbox height (PDFium
    /// reported the font size baked into the text matrix, ~1.0) rather than a
    /// real font-size value. Height-derived sizes jitter ±1pt line-to-line
    /// based on glyph content (descenders, parens, capitals), so heading
    /// detection must use a wider margin over body for these lines.
    pub font_size_is_estimated: bool,
    /// Precise matrix-derived size (`Tf_size × text_matrix_scale`) for
    /// matrix-baked-size lines, when a glyph exposed a text matrix. Used
    /// *only* by heading detection (body-size + heading map), where the
    /// jitter-free value beats the bbox-height estimate in `dominant_font_size`.
    /// Deliberately NOT consumed by table/paragraph grouping, which stay on
    /// `dominant_font_size` to avoid perturbing well-tuned line grouping.
    pub heading_font_size: Option<f32>,
    pub dominant_font_name: Option<String>,
    pub all_bold: bool,
    pub all_italic: bool,
    pub all_mono: bool,
    pub all_strike: bool,
    pub spans: Vec<TextItem>,
    /// Index of each span's source item in the page's `text_items`, aligned
    /// with `spans`. Anything that reorders or extends `spans` must do the
    /// same here.
    #[serde(default)]
    pub span_item_indices: Vec<usize>,
    /// True when the line's base direction is right-to-left (strong RTL
    /// characters outnumber strong LTR ones). `spans` is always in x-ascending
    /// order; consumers that rebuild the line's *text* must walk it in reverse
    /// when this is set, or labels detach from their values.
    pub rtl: bool,
    /// Path from the page's region-tree root to the leaf containing this line.
    /// Equality means "same leaf"; prefix relationship means "one contains the
    /// other". Replaces the prior flat `column_id` scheme so nested layouts
    /// (banded splits with sub-columns) survive paragraph/table grouping.
    pub region_path: Vec<u16>,
    pub mcid: Option<i32>,
    /// True when the line's original-coordinate bbox falls inside a detected
    /// figure/chart region. Chart text (axis labels, legends, category titles)
    /// is often set in a font larger than real headings; if it reached the
    /// heading-size histogram it would hijack the top heading levels and get
    /// promoted itself. Heading detection skips these lines.
    pub in_figure: bool,
}

/// XY-cut region tree node. A page's root region recursively splits along H or
/// V axes until each leaf holds a coherent block of items.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Region {
    pub bbox: Rect,
    pub kind: RegionKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegionKind {
    Leaf { item_indices: Vec<usize> },
    Split { axis: CutAxis, children: Vec<Region> },
}

impl Default for RegionKind {
    fn default() -> Self {
        RegionKind::Leaf { item_indices: Vec::new() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CutAxis {
    Horizontal,
    Vertical,
}

#[derive(Debug, Serialize)]
pub enum Snap {
    Left,
    Right,
    Center,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Anchor {
    Left,
    Right,
    Center,
    /// Inline span that does not snap to a column edge — used by lines whose
    /// dominant items couldn't be classified as Left/Right/Center.
    Floating,
}

#[derive(Debug, Serialize)]
pub struct ProjectedTextItem {
    pub item: TextItem,
    pub snap: Snap,
    pub anchor: Anchor,
    pub is_dup: bool,
    pub rendered: bool,
    pub num_spaces: usize,
    pub force_unsnapped: bool,
    pub is_margin_line_number: bool,
    pub rotated: bool,
    pub d: f32,
    pub orig_x: f32,
    pub orig_y: f32,
    pub orig_width: f32,
    pub orig_height: f32,
    pub orig_rotation: f32,
}

pub type AnchorMap = HashMap<i32, Vec<(usize, usize)>>;
