# Unlimited-OCR-2: document OCR

Baidu's [Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) (`model_type`
`unlimited-ocr`, 3.3B parameters, 6.7 GB of bf16 safetensors) reads page images and writes their
text with its layout: one or many pages per request. It is recognised from its `config.json`
(`src/unlimited_ocr.rs`, written from the checkpoint's own Python code, which is the reference).

```sh
./target/release/oxinfer --model baidu/Unlimited-OCR-2 --image page.png
./target/release/oxinfer --model baidu/Unlimited-OCR-2 --image p1.png --image p2.png --image p3.png > doc.txt
./target/release/oxinfer --model baidu/Unlimited-OCR-2 --image scan.png --no-repeat-ngram 35 --ngram-window 128
```

```
<PAGE><BLOCK>title<LEVEL1>[347, 134, 652, 154]<FORK>Unlimited OCR Works 2
<BLOCK>text[417, 212, 581, 227]<FORK>Baidu ERNIE Team
<BLOCK>title<LEVEL2>[448, 274, 551, 291]<FORK>Abstract
<BLOCK>text[112, 298, 886, 570]<FORK>Unlimited OCR ushers the era of long-horizon parsing, ...
<BLOCK>image[203, 574, 481, 799]<FORK>
<BLOCK>image_caption[112, 808, 886, 859]<FORK>Figure 1 | R-SWA vs SR-SWA. ...
```

The reply is the model's own markup, printed as it is (special tokens included, as the reference
returns it with `skip_special_tokens=False`). The release does not document it; what the runs
here show: `<PAGE>` opens each page, in the order of the images; every block is `<BLOCK>`, a
label (`title` with `<LEVEL1>` / `<LEVEL2>`, `text`, `header`, `image`, `image_caption`, `chart`,
`table`, `page_number`, `other`), a box `[x1, y1, x2, y2]` with values up to 1000, `<FORK>` and
the text; formulas are LaTeX in `\( \)`, and a table is HTML (`<table><tr><td>..`) whose `<tr>`
and `<td>` are special tokens of the vocabulary. Nothing is converted to Markdown here. (The
reference's `_uocr_save_results` looks for DeepSeek-OCR's `<|ref|>` / `<|det|>` tags, which the
checkpoint wrote in none of these runs.)

`-p TEXT` replaces the instruction. The only one the model's code and README use is `Multi page
parsing.` (the default); the prompt formats of DeepSeek-OCR, which this model descends from
(`<|grounding|>`, `Free OCR.`, ...), are not mentioned anywhere in this release and were not
tried. A prompt may place the images itself with one `<image>` per page (it then also writes the
`<PAGE>` markers, or not: without them the reference falls back to plain positions, see below).

| Flag | |
|---|---|
| `--image FILE` | a page; repeat it for a document, in page order |
| `--max-tokens N` | reply length (default 8000 per page, the README's serving budget) |
| `--temp`, `--top-k`, `--top-p` | sampling; greedy by default, as `infer()` |
| `--no-repeat-ngram N`, `--ngram-window W` | no n-gram of N generated tokens twice within the last W (off by default as in `infer()`; the README's serving recipe uses 35 with a window of 128, or 1024 for several pages) |
| `--max-page-tokens N` | tokens of one page before the next `<PAGE>` (or the end) is forced (default 20000) |
| `--no-page-guard` | drop the rule that the reply has exactly as many `<PAGE>` as the prompt |
| `--max-soft-tokens N` | an image token budget below a page's tokens with tiles keeps its global view only (273 tokens) |
| `--dump-mm PREFIX`, `--dump-logits FILE`, `--show-tokens`, `--print-prompt` | for comparisons with the reference |

## What happens to a page

1. **Views** (as Pillow does them). The page resized to fit 1024x1024 (bicubic), centered on
   gray: the global view, always. A single page larger than 640 pixels is also squashed (bicubic)
   onto the grid of 2 to 16 tiles of 640x640 whose aspect ratio is closest to its own ("gundam"
   mode): an A4 page at 150 dpi gives 2x3 tiles. Several pages in one request are global views
   only ("base" mode), which is `infer()`'s automatic choice. Pixels are `x / 127.5 - 1`, rounded
   to bfloat16 as the reference rounds them.
2. **Encoder** ("deepencoder"), per view: a SAM ViT-B (16x16 patches, 12 blocks of width 768,
   windows of 14 with decomposed relative positions and four blocks of global attention, the neck,
   then two stride-2 convolutions) whose 1024-channel output, 1/64 of the pixels, is the patch
   embedding of a CLIP ViT-L (class token, learned positions, 24 blocks, no final norm). The two
   outputs side by side (2048) go through one linear layer into the decoder's width (1280): 256
   features for the global view, 100 per tile. The position tables are resampled for tiles as
   PyTorch does (antialiased bicubic for the embeddings, linear for SAM's relative positions).
3. **Packing**: the tiles' features as one grid, a learned newline after each row, then the global
   view's 16 rows with their newlines, then a separator: 273 tokens for a page alone,
   `(10 w + 1) * 10 h` more with `w` x `h` tiles (903 for 2x3).
4. **Prompt**: BOS, then `<PAGE>` and the page's `<image>` placeholders for every page, then the
   instruction. No chat roles, no system prompt.
5. **Decoder**: 12 layers of width 1280 with 10 heads; a dense MLP in the first layer, then
   DeepSeek-V2 mixture-of-experts layers (softmax router, the 6 best of 64 experts with their
   probabilities as they are, plus a shared expert).

### DASH attention

No layer attends to its whole context. A query reads its last 128 tokens (4 blocks of 32) and at
most 12 earlier blocks, which a per-layer router picks (`dash_attention.py`,
`modeling_deepseekv2.py`):

- the layer input goes through two more projections (128 wide), Gemma-style RMS norms and the
  rotary embedding: the router's query and key of the token;
- a learned query `head_cls` pools the 32 router keys of a block into one summary key (a softmax
  over the block);
- a query scores the summary keys of the blocks that end before its local window. The candidates
  are the blocks holding image tokens of the query's own page (`<PAGE>` markers count the pages
  of the prompt and of the reply; image tokens look at their own page, the instruction and
  anything before the first marker at every block), or all the earlier blocks when there is none;
- the scores, times `dash_scale_per_layer / sqrt(128)`, of the best 16 candidates go through
  1.5-entmax; the first 12 are read, and `log(max(p, floor))` minus its mean over them is added to
  the attention scores of their keys. The local window has no bias and wins where a chosen block
  overlaps it.

The cost of a decoding step therefore does not grow with the document: at most 512 keys per head.
The session keeps the rotated keys and values of every position, the router's keys and one
summary key per full block. Summaries are computed once, when a block fills (the reference
recomputes them at every step; a block still filling is always inside the local window).

Positions are a rotary embedding over two axes: the running index of the token within its page
(63 frequency pairs, base 10000) and the parity of the page number (one pair), so that the text
of page 3 sits right after the image tokens of page 3 whatever came between. A prompt without
`<PAGE>` markers before its last image, or without images, gets plain positions instead (YaRN
over the decoder's `rope_theta`), as in the reference.

Because a token's page depends on where the last image of the prompt ends, the prompt must be
fed whole: the first forward pass of a session is the prompt (`prefill_chunk` is ignored), every
later token a decoding step. One request, one document: there is no second turn (`--chat` is
refused; through the engine a conversation's later messages would be fed as decoding steps).

### Decoding rules

Both are the reference's (`_uocr_page_count_guard`, `_uocr_no_repeat_ngram`), over the generated
tokens, applied as a sampler filter (`OcrGuard`):

- the page count guard (on by default): no end of sequence before the reply has as many `<PAGE>`
  markers as the prompt; after `--max-page-tokens` tokens of one page the next marker (or the end,
  on the last page) is forced; one marker too many and only the end is allowed;
- the optional no-repeat rule: a token that would complete an n-gram already present in the last
  `--ngram-window` generated tokens is not allowed.

## Server and library

`oxinfer serve` takes the pages as `image_url` parts of the user message
([server](server.md)); the text part is the instruction, and may be the README's own request text
(`"<PAGE><image>" * pages + "Multi page parsing."`: placeholders written by the client are
dropped, the attachments already stand for them). Send `"temperature": 0` and a `max_tokens`
that fits the document (the model's default reply length is the engine's 1024 tokens). The
reply's `content` keeps the special tokens. The page count guard applies; the n-gram rule and the
page cap are command-line flags only for now (in the library: `OcrGuard::with_ngram`). A request
of one page without text, one with the README's text (streamed), one of three pages and one
of a page with tables (240 `<td>` in the reply) gave the command line's replies.

```rust
use oxinfer::engine::{Attachments, Engine, EngineOptions, Flow};
let mut e = Engine::load("baidu/Unlimited-OCR-2", &EngineOptions::default())?;
e.config_mut().max_tokens = 16000;
let pages = Attachments { images: vec!["p1.png".into(), "p2.png".into()], ..Default::default() };
let reply = e.send_with_media("", &pages, &mut |d| { print!("{}", d.text); Flow::Continue })?;
```

`oxinfer::unlimited_ocr` has the stages: `preprocess` (views), `OcrTower` (`features`, `project`,
`encode`), `prompt_text`, `OcrModel` (`forward_hidden_mm`, `logits`; it implements `TextLm`) and
`OcrGuard`.

## Agreement with the reference

`tools/ref_unlimited_ocr.py` runs the checkpoint's own code (transformers 4.52.1, torch 2.15,
Pillow 13.0) in float32 on the CPU and compares oxinfer's dumps (bf16 weights as stored,
`--device cpu`). Three things differ from `model.infer()` as published, none of them in the math:
the `.cuda()` call on the image mask is a no-op, the pixel tensors (rounded to bfloat16 by the
reference's own input builder) are widened to float32, and attention runs the code's own dense
path (`DashAttention._forward_dense`, `_decode_attend`), which it takes by itself without a GPU,
in place of its Triton kernels.

| Input | Prompt | Pixel values | Encoder features | Packed embeddings | Logits after the prompt | Greedy reply |
|---|---|---|---|---|---|---|
| a 600x500 crop (global view only) | 279 tokens | identical | 5.5e-5 | 4.3e-5 | 2.9e-5, same top 10 | 48 of 48 tokens (stopped there) |
| an A4 page at 150 dpi, 1241x1754 (2x3 tiles) | 909 | identical (7 views) | 4.8e-5 | 4.6e-4 | 6.3e-5, same top 10 | 511 of 511, to the end |
| another page of the same PDF, with three tables (2x3 tiles) | 909 | identical (7 views) | 6.4e-4 | 3.3e-4 | 6.0e-5, same top 10 | 1990 of 1990, to the end |
| a chart, 1670x1253 (4x3 tiles) | 1509 | identical (13 views) | 1.2e-3 | 5.8e-4 | 4.7e-4, same top 10 | 363 of 363, to the end |
| three A4 pages in one request (global views) | 827 | identical | 4.8e-5 | 4.1e-5 | 6.0e-5, same top 10 | 1685 of 1685, to the end |
| the crop with `<image>` and no `<PAGE>` (plain positions) | 278 | | | | 3.2e-5, same top 10 | 64 of 64 (stopped there) |
| text only (plain positions) | 23 | | | | 2.4e-5, same top 10 | 40 of 40 (stopped there) |
| the crop with `--no-repeat-ngram 3 --ngram-window 64` | 279 | | | | | 96 of 96, to the end |

The prompt's token ids are the reference's in every case. Differences are the largest over all
values, for encoder features of magnitude 3 (the global view's, 256 x 2048), embeddings of
magnitude 4.5 and logits of magnitude 26; pixel values, encoder features and embeddings are
compared for the first page of a request. The three-page reply crosses two page changes of the
position state and of the candidate blocks; the longest context compared is 2899 tokens.

With a GPU active (`--device auto`, CUDA) the A4 page gives the same 511 tokens and logits within
2.3e-5 of the CPU run. With `--quant q8` its first 417 tokens are the same, then one box
coordinate differs (797 for 795); the text is the same.

Speed on 64 x86 cores (bf16 weights, CPU only): 3.5 s per 1024x1024 view through the encoder
(10.5 s for a page with 6 tiles, 17.5 s with 12), prefill at 760-810 tokens/s, decoding at 29-35
tokens/s.

## Not done, not verified

- **Prompts other than `Multi page parsing.`**: not tried (see above); no claim about plain OCR,
  Markdown or grounding prompts.
- **Post-processing**: no Markdown file, no boxes drawn, no figures cut out
  (`_uocr_save_results`); the reply is the raw markup.
- **The serving recipe of the README** (SGLang): page-window scheduling of very long documents
  (`SGLANG_UOCR_DOC_*_PAGE_WINDOW`: a window of pages at a time, the tail of the previous window's
  cache reused), `images_config.crop_mode` and `custom_params` in requests are not implemented.
  The server applies the page count guard with its defaults and nothing else; PDFs must be
  rendered to images by the caller.
- **Long documents**: compared with the reference up to three pages and up to 2899 tokens of
  context; nothing longer was run, here or there. The attention reads a bounded number of keys,
  but every position's keys and values are kept (about 130 KB per token over the 12 layers), as
  in the reference's own `infer()`.
- **Inputs**: PNG files only were compared. JPEG decoding is not Pillow's, so pixels may differ
  by a rounding step; images with an alpha channel or an EXIF orientation were not compared (the
  reference converts to RGB with Pillow and ignores the orientation).
- **Speed**: nothing was optimized. The encoder and the per-token attention run on the CPU even
  with a GPU active (the tiled page: 9.0 s of encoder instead of 10.5, prefill 1160 tokens/s,
  decoding unchanged).
- **Quantized weights**: `--quant q8|q4` go through the usual loaders (the decoder's matrices
  only; the encoder and the routers stay bf16); only the q8 run above was looked at. No
  `--save-quantized`, no GGUF.
- **Not available for this model**: a second turn, session snapshots and the prompt cache, prompt
  lookup and speculative decoding, runtime interventions, typed decisions.
- **Configurations**: only what this checkpoint uses is implemented; a config with another router
  stage (`dash_stage0`), layers without sparse attention, other rope modes, MLA or grouped
  key/value heads is refused at load. A prompt of a single token takes a path of the reference
  that is not the one implemented.
- `dynamic_preprocess` orders candidate grids of equal area by Python's set order; that order can
  only matter for two such grids exactly as close to the image's aspect ratio, which cannot be
  the closest pair (another grid lies between them).
