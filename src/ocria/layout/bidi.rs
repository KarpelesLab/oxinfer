//! Minimal bidirectional-text helpers.
//!
//! PDF content streams store right-to-left text in *logical* order already —
//! a correctly generated Hebrew/Arabic PDF places the first logical character
//! at the highest x and advances leftward, and PDFium hands the characters back
//! in that same stream order. So liteparse never needs to reverse characters;
//! what it does need is to stop assuming that "left to right on the page" means
//! "first to last in reading order".
//!
//! Two places care:
//!   * `extract` measures inter-character gaps along the writing direction
//!     (see `SegmentBuilder::gap_to`), so RTL runs are not sheared into
//!     fragments by the line-change heuristics.
//!   * line assembly concatenates a line's items in reading order, which for an
//!     RTL line runs right-to-left across the page.
//!
//! Ported from LiteParse (crates/liteparse/src/bidi.rs; Apache-2.0, (c) LlamaIndex).

/// Strong right-to-left character (Unicode bidi class R or AL). Covers Hebrew,
/// Arabic, Syriac, Thaana, N'Ko, Samaritan, Mandaic, Adlam and the Arabic
/// presentation-form blocks that subset fonts commonly map to.
pub(crate) fn is_rtl_char(c: char) -> bool {
    matches!(c as u32,
        0x0590..=0x05FF   // Hebrew
        | 0x0600..=0x07BF // Arabic, Syriac, Thaana, N'Ko
        | 0x0800..=0x085F // Samaritan, Mandaic
        | 0x08A0..=0x08FF // Arabic Extended-A
        | 0xFB1D..=0xFDFF // Hebrew/Arabic presentation forms A
        | 0xFE70..=0xFEFF // Arabic presentation forms B
        | 0x10800..=0x10FFF // Cypriot..Old Hungarian etc.
        | 0x1E800..=0x1EFFF // Mende Kikakui, Adlam, Arabic Math
    )
}

/// Strong left-to-right character. Deliberately *not* the complement of
/// [`is_rtl_char`]: digits, punctuation and whitespace are bidi-neutral and
/// must not vote on a line's base direction, or every "12.34 SAR" amount on an
/// Arabic invoice would drag its line back to LTR.
fn is_strong_ltr_char(c: char) -> bool {
    c.is_alphabetic() && !is_rtl_char(c)
}

/// Base direction of a run of characters, decided by majority of
/// strong-direction characters. Neutral-only text (pure numbers, punctuation)
/// is LTR, which keeps every left-to-right document on exactly the path it took
/// before.
fn is_rtl_chars(chars: impl Iterator<Item = char>) -> bool {
    let mut rtl = 0usize;
    let mut ltr = 0usize;
    for c in chars {
        if is_rtl_char(c) {
            rtl += 1;
        } else if is_strong_ltr_char(c) {
            ltr += 1;
        }
    }
    rtl > ltr
}

/// Base direction of a line still held as separate pieces, without joining
/// them first — line assembly needs the direction *before* it picks a join
/// order, and the pieces can be numerous.
///
/// Every RTL character is non-ASCII, so a line whose pieces are all ASCII has
/// an RTL count of zero and cannot be RTL whatever its LTR count. `is_ascii` is
/// a byte-wise scan rather than per-`char` classification, which keeps this off
/// the hot path for left-to-right documents — the overwhelmingly common case.
pub(crate) fn is_rtl_pieces<'a>(pieces: impl IntoIterator<Item = &'a str> + Clone) -> bool {
    if pieces.clone().into_iter().all(|p| p.is_ascii()) {
        return false;
    }
    is_rtl_chars(pieces.into_iter().flat_map(str::chars))
}
