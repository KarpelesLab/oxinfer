//! A text unit of the table model: on an OCR page, one projected OCR item (LiteParse's
//! `table_model/units.rs`; its units built from a PDF's characters are not needed for images).
//! Points, top-left origin.
//!
//! Ported from LiteParse (crates/liteparse/src/table_model/units.rs, the unit an OCR page takes; Apache-2.0, (c) LlamaIndex).

#[derive(Clone, Debug)]
pub struct Unit {
    /// reading-frame box (the page's: OCR pages are read upright)
    pub bx: [f64; 4],
    /// rendered-page box
    pub pbox: [f64; 4],
    pub text: String,
    pub h: f64,
    /// the projected item it is (its index in the projection signals' items)
    pub item: usize,
    /// words as (x0, x1, text), for the lattice to cut a line read across a ruled cell boundary
    pub words: Vec<(f64, f64, String)>,
}
