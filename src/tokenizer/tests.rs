use super::*;

fn tok(json: &str) -> Tokenizer {
    Tokenizer::from_bytes(json).unwrap()
}

#[test]
fn byte_level_bpe_with_added_tokens() {
    let t = tok(r#"{
        "added_tokens": [{"id": 6, "content": "<|end|>", "special": true, "normalized": false}],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "Ġ": 2, "ab": 3, "Ġab": 4, "Ġa": 5},
                  "merges": [["Ġ", "a"], "a b", "Ġa b"]}
    }"#);
    let e = t.encode("ab ab<|end|>a", false).unwrap();
    assert_eq!(e.get_ids(), [3, 4, 6, 0]);
    assert_eq!(t.decode(e.get_ids(), false).unwrap(), "ab ab<|end|>a");
    assert_eq!(t.decode(e.get_ids(), true).unwrap(), "ab aba");
    assert_eq!(t.token_to_id("<|end|>"), Some(6));
    assert_eq!(t.get_vocab_size(true), 7);
}

#[test]
fn template_post_processor_and_truncation() {
    let mut t = tok(r#"{
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": {"type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
            "pair": [],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [0], "tokens": ["<s>"]}, "</s>": {"id": "</s>", "ids": [1], "tokens": ["</s>"]}}},
        "model": {"type": "WordLevel", "vocab": {"<s>": 0, "</s>": 1, "<unk>": 2, "x": 3, "y": 4}, "unk_token": "<unk>"}
    }"#);
    let e = t.encode("x y z", true).unwrap();
    assert_eq!(e.get_ids(), [0, 3, 4, 2, 1]);
    assert_eq!(e.get_special_tokens_mask(), [1, 0, 0, 0, 1]);
    t.with_truncation(Some(TruncationParams {
        max_length: 4,
        ..Default::default()
    }))
    .unwrap();
    assert_eq!(t.encode("x y z", true).unwrap().get_ids(), [0, 3, 4, 1]);
    assert_eq!(t.encode("x y z", false).unwrap().get_ids(), [3, 4, 2]);
}

#[test]
fn wordpiece_with_bert_components() {
    let t = tok(r###"{
        "normalizer": {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
        "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
                  "vocab": {"[UNK]": 0, "hello": 1, "wor": 2, "##ld": 3, "!": 4, "café": 5, "cafe": 6}}
    }"###);
    let e = t.encode("Héllo, World!", false).unwrap();
    // "héllo" loses its accent (strip_accents follows lowercase), "," is unknown.
    assert_eq!(e.get_ids(), [1, 0, 2, 3, 4]);
    assert_eq!(t.decode(&[1, 2, 3, 4], false).unwrap(), "hello world!");
}

#[test]
fn unigram_metaspace_and_byte_fallback() {
    let t = tok(r#"{
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
        "decoder": {"type": "Sequence", "decoders": [{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}, {"type": "ByteFallback"}]},
        "model": {"type": "Unigram", "unk_id": 0, "byte_fallback": true, "vocab": [
            ["<unk>", 0.0], ["▁he", -1.0], ["llo", -1.5], ["▁hello", -2.0], ["▁", -3.0], ["h", -4.0], ["<0xC3>", -5.0], ["<0xA9>", -5.0]]}
    }"#);
    let e = t.encode("hello hé", false).unwrap();
    assert_eq!(e.get_ids(), [3, 4, 5, 6, 7]);
    assert_eq!(t.decode(e.get_ids(), false).unwrap(), "hello hé");
}

#[test]
fn gemma_style_bpe_byte_fallback() {
    let t = tok(r#"{
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false},
        "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "byte_fallback": true, "fuse_unk": true, "unk_token": "<unk>",
                  "vocab": {"<unk>": 0, "a": 1, "▁": 2, "▁a": 3, "<0xE2>": 4, "<0x82>": 5, "<0xAC>": 6},
                  "merges": [["▁", "a"]]}
    }"#);
    let e = t.encode("a a€", false).unwrap();
    assert_eq!(e.get_ids(), [1, 3, 4, 5, 6]);
    assert_eq!(t.decode(e.get_ids(), false).unwrap(), "a a€");
}
