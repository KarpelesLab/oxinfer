//! Unicode character properties for the tokenizer (from the `intl` crate's UCD tables).

use intl::unicode::{GeneralCategory, general_category};

/// General-category bit masks (bit = the `GeneralCategory` discriminant).
pub struct Gc;

#[allow(dead_code)]
impl Gc {
    pub const LU: u64 = 1 << GeneralCategory::UppercaseLetter as u8;
    pub const LL: u64 = 1 << GeneralCategory::LowercaseLetter as u8;
    pub const LT: u64 = 1 << GeneralCategory::TitlecaseLetter as u8;
    pub const LM: u64 = 1 << GeneralCategory::ModifierLetter as u8;
    pub const LO: u64 = 1 << GeneralCategory::OtherLetter as u8;
    pub const L: u64 = Self::LU | Self::LL | Self::LT | Self::LM | Self::LO;
    pub const MN: u64 = 1 << GeneralCategory::NonspacingMark as u8;
    pub const MC: u64 = 1 << GeneralCategory::SpacingMark as u8;
    pub const ME: u64 = 1 << GeneralCategory::EnclosingMark as u8;
    pub const M: u64 = Self::MN | Self::MC | Self::ME;
    pub const ND: u64 = 1 << GeneralCategory::DecimalNumber as u8;
    pub const NL: u64 = 1 << GeneralCategory::LetterNumber as u8;
    pub const NO: u64 = 1 << GeneralCategory::OtherNumber as u8;
    pub const N: u64 = Self::ND | Self::NL | Self::NO;
    pub const PC: u64 = 1 << GeneralCategory::ConnectorPunctuation as u8;
    pub const PD: u64 = 1 << GeneralCategory::DashPunctuation as u8;
    pub const PS: u64 = 1 << GeneralCategory::OpenPunctuation as u8;
    pub const PE: u64 = 1 << GeneralCategory::ClosePunctuation as u8;
    pub const PI: u64 = 1 << GeneralCategory::InitialPunctuation as u8;
    pub const PF: u64 = 1 << GeneralCategory::FinalPunctuation as u8;
    pub const PO: u64 = 1 << GeneralCategory::OtherPunctuation as u8;
    pub const P: u64 = Self::PC | Self::PD | Self::PS | Self::PE | Self::PI | Self::PF | Self::PO;
    pub const SM: u64 = 1 << GeneralCategory::MathSymbol as u8;
    pub const SC: u64 = 1 << GeneralCategory::CurrencySymbol as u8;
    pub const SK: u64 = 1 << GeneralCategory::ModifierSymbol as u8;
    pub const SO: u64 = 1 << GeneralCategory::OtherSymbol as u8;
    pub const S: u64 = Self::SM | Self::SC | Self::SK | Self::SO;
    pub const ZS: u64 = 1 << GeneralCategory::SpaceSeparator as u8;
    pub const ZL: u64 = 1 << GeneralCategory::LineSeparator as u8;
    pub const ZP: u64 = 1 << GeneralCategory::ParagraphSeparator as u8;
    pub const Z: u64 = Self::ZS | Self::ZL | Self::ZP;
    pub const CC: u64 = 1 << GeneralCategory::Control as u8;
    pub const CF: u64 = 1 << GeneralCategory::Format as u8;
    pub const CS: u64 = 1 << GeneralCategory::Surrogate as u8;
    pub const CO: u64 = 1 << GeneralCategory::PrivateUse as u8;
    pub const CN: u64 = 1 << GeneralCategory::Unassigned as u8;
    pub const C: u64 = Self::CC | Self::CF | Self::CS | Self::CO | Self::CN;
}

/// The general-category bit of `c`.
#[inline]
pub fn gc_mask(c: char) -> u64 {
    1 << general_category(c) as u8
}

/// `\w` as the `regex` crate defines it: Alphabetic, marks, decimal digits, connector
/// punctuation and Join_Control.
pub fn is_word_char(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric() || c == '_';
    }
    intl::unicode::is_alphabetic(c) || gc_mask(c) & (Gc::M | Gc::ND | Gc::PC) != 0 || matches!(c, '\u{200c}' | '\u{200d}')
}

/// Any punctuation (`P*`), plus ASCII punctuation (the BERT and Punctuation pre-tokenizers).
pub fn is_punctuation(c: char) -> bool {
    c.is_ascii_punctuation() || gc_mask(c) & Gc::P != 0
}

/// `C*` other (control, format, surrogate, private use, unassigned).
pub fn is_other(c: char) -> bool {
    gc_mask(c) & Gc::C != 0
}

pub fn is_mark(c: char) -> bool {
    gc_mask(c) & Gc::M != 0
}

pub fn is_nonspacing_mark(c: char) -> bool {
    gc_mask(c) & Gc::MN != 0
}

pub fn nfc(s: &str) -> String {
    intl::unicode::nfc(s.chars()).collect()
}

pub fn nfd(s: &str) -> String {
    intl::unicode::nfd(s.chars()).collect()
}

pub fn nfkc(s: &str) -> String {
    intl::unicode::nfkc(s.chars()).collect()
}

pub fn nfkd(s: &str) -> String {
    intl::unicode::nfkd(s.chars()).collect()
}

pub fn graphemes(s: &str) -> impl Iterator<Item = &str> {
    intl::unicode::graphemes(s)
}
