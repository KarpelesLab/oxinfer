# OCR with OCRia-nano-1

[`llamaindex/OCRia-nano-1`](https://huggingface.co/llamaindex/OCRia-nano-1) is the OCR stack of
[LiteParse](https://github.com/run-llama/liteparse), LlamaIndex's local document parser: small
ONNX models distilled from their 4B document VLM, made for the CPU. oxinfer runs its OCR models
natively, on its own ONNX interpreter (no ONNX Runtime), with LiteParse's pipeline around them:

| file | what | parameters |
|---|---|---|
| `det.onnx` (+ `det.meta.json`: threshold 0.55) | text-line detector: a probability map at half the input's size | 82K |
| `router.onnx` | the script of a line (Latin, CJK, right-to-left, world) and whether it is upside down | 38K |
| `rec_latin.onnx` | Latin, Greek and Cyrillic lines (791 characters), quantized (int8 weights, uint8 activations) | 0.82M |
| `rec_cjk.onnx` | Chinese, Japanese, Korean (6213 characters) | 1.27M |
| `rec_rtl.onnx` (+ `rec_rtl.meta.json`: visual order) | Arabic, Hebrew (323 characters) | 0.71M |
| `rec_world.onnx` | about 17 more scripts: Devanagari and the Indic scripts, Thai, Armenian, Georgian, ... (1711 characters) | 0.98M |
| `tablenet.bin` (LayoutNet, optional) | tables (rows, columns, spanning cells, header rows) and 13 layout-block classes over the page's text units: a 6-layer transformer with a CNN image branch | 4.27M |

The recognizers decode in the graph: they output the best class and its probability at each step
(one step per 4 pixels of a 32-pixel-high line), and the CTC collapse happens outside. LayoutNet
runs with LiteParse's text projection and table decoder around it, ported: a page's layout blocks
(headings, paragraphs, list items, page headers and footers, code) and its tables, as LiteParse
parses an image (see [Layout and tables](#layout-and-tables)). The models are under their own
license (`LICENSE` in the repository).

## Use

```sh
oxinfer ocr scan.png                 # the text, in reading order
oxinfer ocr page1.png page2.jpg --json
oxinfer ocr receipt.png --scripts latin --json   # only the Latin recognizer (3 model files)
oxinfer ocr report.png --markdown    # headings, lists and tables as Markdown (LayoutNet)
oxinfer ocr invoice.png --tables --json          # the lines plus "layout": blocks, tables, cells
```

`oxinfer ocr IMAGE...` takes PNG, JPEG, WebP, GIF or BMP files (the `media` feature) and fetches
the models on first use (only the files of `--scripts`; 14 MB for all four scripts). Options:
`--model` (the repo id, or a folder holding the files), `--scripts latin,cjk,rtl,world` (default
all), `--long-side N` (detector input, default 1600), `--json`, `--plain` (no page-level
heuristics, see below), `--no-adaptive`, `--no-gutter`, `--no-orient`, `--no-rescue`, `--no-junk`,
`--no-faint`, `--no-clean`, `--no-bidi`, `--no-layout`, `--no-exif`, `--thresh T`, `--merge G`,
`--repeat N` (timing), `--device auto|cpu|metal` (where the models run, see [Speed](#speed)),
`--threads`, `--cache-dir`, `-q`; for the layout `--tables` (load LayoutNet), `--markdown` (print
the page as Markdown; loads it), `--dpi N` (the image's resolution, default its own, else 150),
`--no-structure`.

`--json` prints the page: `lines` (each with `text`, `confidence`, `box`: the four corners in
reading order, `bbox`: its extent, `script`: the recognizer that read it, `words`: word boxes on
upright left-to-right lines, `faint`), `text` (the lines one per line, in reading order),
`script` (the page's), `upside_down`, `sideways`, `exif_orientation` (the EXIF orientation the
photo was turned upright by, or null), `seconds`, and with LayoutNet loaded `layout`: `blocks` in
reading order (`kind`: `heading` with its `level`, 1 for a title, `paragraph`, `list_item`,
`page_header`, `page_footer`, `code`, `table` with the index of its table, `table_text` /
`figure_text` for LayoutNet's table and figure-text blocks outside every table; `bbox` in image
pixels; `text`), `tables` (`bbox`, `rows`, `cols`, `header_rows`, `cells`: `row`, `col`,
`row_span`, `col_span`, `text`, `bbox`), `markdown`, `grid_text` (LiteParse's projected text,
columns kept apart on a character grid) and `dpi`.

Library:

```rust
use oxinfer::ocria::{Ocria, OcrOptions, OcrRunOptions};
let ocr = Ocria::load("llamaindex/OCRia-nano-1", &OcrOptions::default())?;
let page = ocr.ocr_file("scan.png".as_ref(), &OcrRunOptions::default())?;
println!("{}", page.text);
for l in &page.lines {
    println!("{:?} {:.3} {} {}", l.bbox, l.confidence, l.script, l.text);
}
// RGB (or grey) pixels already in memory:
let page = ocr.ocr_rgb(&rgb, width, height, &OcrRunOptions::default())?;
// with LayoutNet: tables and layout blocks
let mut o = OcrOptions::default();
o.layoutnet = true;
let ocr = Ocria::load("llamaindex/OCRia-nano-1", &o)?;
let page = ocr.ocr_file("invoice.png".as_ref(), &OcrRunOptions::default())?;
let layout = page.layout.unwrap();
println!("{}", layout.markdown);
for t in &layout.tables {
    for c in &t.cells {
        println!("{} {} {:?}", c.row, c.col, c.text);
    }
}
```

`Ocria` is `Sync`: one instance serves pages from several threads.

C: `oxi_ocria_load` (`scripts=latin,cjk`, `layoutnet=1`, `det_long_side`, `threads`, `cache_dir`, `device`),
`oxi_ocria_rgb` (pixels, 1 or 3 channels, a row stride) and `oxi_ocria_file` return the page as
JSON (`options_json`: the switches below), `oxi_ocria_info_json`, `oxi_ocria_free` (see
`include/oxinfer.h`).

Server: a model of kind `ocr` (a folder holding `det.onnx` and `router.onnx` is detected, the repo
id is recognized by name; `scripts=latin+cjk`, `layoutnet=1` in its options), `POST /v1/ocr` with
the image as a multipart `file` (or `image`) part or base64 in a JSON body, `response_format`
`json` (the page as above, the default), `text` or `markdown` (LayoutNet's), and the switches
below (with `structure` and `dpi`) as fields:

```sh
oxinfer serve --model ocr=llamaindex/OCRia-nano-1,layoutnet=1
curl -s localhost:8080/v1/ocr -F model=ocr -F file=@scan.png -F response_format=text
curl -s localhost:8080/v1/ocr -F model=ocr -F file=@report.png -F response_format=markdown
```

## The pipeline

Ported from LiteParse (`crates/liteparse/src/ocr/ocria`, Apache-2.0), with its defaults:

- `exif`: an encoded photo (JPEG APP1, PNG `eXIf`) is turned upright per its EXIF orientation
  before anything else, as LiteParse's image loading does (PIL's `exif_transpose`; pixels given
  as such, `oxi_ocria_rgb` and `Ocria::ocr_rgb`, are read as they are);
- detection at a long side of 1600 (never enlarged; sides rounded to multiples of 32) with
  OpenCV's `INTER_AREA` resize, the map thresholded at `det.meta.json`'s 0.55, 4-connected
  components (at least 2 cells, mean probability 0.6+) turned into rotated rectangles grown by the
  training shrink ratio, collinear boxes merged (gaps under 0.6 line heights), a row-band sort;
  `adaptive`: a second pass at up to 3200 pixels on pages of small print (lines under 14 pixels at
  the first pass's scale covering 0.3% of the page), its lines merged with the first pass's;
- `gutter`: boxes that run across a column gutter (blank columns lined up with other boxes'
  edges, judged by Otsu's threshold) are cut;
- `orient`: pages whose boxes look sideways are detected again a quarter turn round and kept turned
  when they read clearly better; upside-down pages are the router's vote below. That is all of
  LiteParse's page orientation for images: beyond it there are only the per-page rotations a
  caller gives for a PDF (`page_orientation_corrections`, `/Rotate` rewrites) and the small-angle
  deskew of tilted scans while merging OCR with a PDF's text layer, which have no image
  counterpart here;
- crops: a perspective warp of each quad (OpenCV's fixed-point bicubic, replicated borders), turned
  upright when tall;
- the router reads every line tiled to 512 pixels; the page's script is the one with most of the
  line width (Latin unless another has 65%+); a line keeps its own script when the router is 90%+
  sure and it is 4+ times wider than high; the page is read upside down when 3+ lines vote so with
  70%+ of the width;
- recognition: lines sorted by width, 32 a batch, padded with white to a multiple of 128 pixels
  (a line's reading depends on the padded width, so the batching is the reference's), CTC collapse,
  the mean confidence of the characters, word spans;
- `bidi`: right-to-left readings (written left to right as seen) reordered to logical order
  (UAX #9 with a right-to-left paragraph, intl's implementation; LiteParse uses unicode-bidi);
- `rescue`: a low-confidence Latin page is re-read with the scripts the router saw a fair share of,
  Latin lines one by one too, and lines routed away from the page's script are re-read with the
  page's recognizer; `clean`: lookalikes of other scripts folded in Latin words (Cyrillic "а" in an
  English word, Thai "๐" for "o", other scripts' digits in Latin lines);
- `junk`: readings with no letter or digit, or one or two unsure characters, are dropped; `faint`:
  short tokens the detector scored under its threshold (a lone "1" in a table cell) next to
  detected lines, kept when read at 0.9+;
- `layout` (oxinfer's): lines and text ordered by recursive XY cuts (sections, then columns left to
  right, then lines), where LiteParse leaves the row-band order to its PDF layout stage.

`--plain` (`OcrRunOptions::plain()`, `"plain": true`) keeps only the models: one detection pass
with the merge, the router with its upside-down vote, the routed recognizers, no reordering.

## Accuracy against the reference

Against onnxruntime 1.32 (CPU) on twelve images: ten 900-pixel images of two or three lines
(English, CJK, Arabic / Hebrew, Devanagari / Cyrillic, each also upside down), a page mixing
seven scripts and a 2000 x 2600 page (title, two columns, a table, footnotes; turned 1.5 degrees).
The reference is a Python pipeline on onnxruntime written from LiteParse's code (its OpenCV
operations as numpy; the detector pass, crops, router and recognizers, no page heuristics):

- **The models, on the reference's own inputs**: the detector's probability map within 1.2e-6 (no
  cell on the other side of the threshold), the router's logits within 4.3e-6, the float
  recognizers (CJK, right-to-left, world) the same characters at every step and confidences within
  6e-6. `rec_latin` runs onnxruntime's int8 arithmetic (below): against its default CPU kernels
  (`QLinearConv`, `QLinearAdd`) the same characters at all 12,224 steps and confidences within
  6e-8 (the float tail after the last quantized layer). (As float fake quantization it differed on 5 steps,
  confidences up to 0.18, as onnxruntime's own float and int8 runs of the graph differ: 1 step,
  0.22; `OXINFER_ONNX_INT8=0` still runs it so.)
- **rec_latin's integer arithmetic**, found by emulating onnxruntime 1.32's optimized graph
  (29 `QLinearConv`, 11 `QLinearAdd`) layer by layer on its own intermediate tensors (the
  emulation reproduces every uint8 tensor of the graph on the twelve images' batches): the input
  quantized as `clamp(round(x / s) + zp)` (a division, rounded half to even); a convolution as
  the int32 sum of `(q - zp)` times the int8 weights plus the int32 bias, then
  `round(float(sum) * f32(s_in * w_scale[c] / s_out)) + zp` saturated (the ReLUs are the
  saturation at a zero point of 0); an addition as `round(f32(f32(sa / sc) (qa - za)) +
  f32(f32(sb / sc) (qb - zb)) + zc)` saturated. The usual alternatives (MLAS's `zc - (ra za +
  rb zb)` folded constant, fused multiply-adds, dequantize-add-quantize) each miss one rounding
  case (26 of 89.5 million values): `109.49999` where onnxruntime has `109.5`. Here the convolutions sum the
  integers in f32 (exact: every partial sum stays below 2^24, checked per channel at load), so
  they run on the same GEMMs as the float models.
- **The pipeline** (`--plain`): 103 lines, the same boxes (within 2.4e-4 pixels) and the same
  recognizer for every line, the same text on 101. The other two are noise: an upside-down
  two-line page (too few lines to turn) read at 0.55-0.66. The line images differ by a few grey
  levels on ~8% of their pixels: the reference's numpy versions of OpenCV's warp and resizes round
  some sub-pixel steps differently from LiteParse's Rust ones (used here), and OpenCV 5.1 on this
  machine differs from both by as much. Confidences: within 1.2e-4 for the float recognizers; Latin
  lines within a few hundredths, except very short ones ("12" of the table: 0.69 here, 0.92 in the
  reference), which the junk filter (one or two characters under 0.8) then treats differently.
- **What LiteParse reads**: its crops (this code) fed to onnxruntime's int8 kernels, measured
  before rec_latin had their arithmetic: the same text on every line but one (the upside-down
  noise again: 'e įxə o əl eцouy' for 'e xə o l eщouy') and confidences within 0.011. With it,
  the Latin lines read as there to the confidence: the "12" is 0.6863 here as in LiteParse.
- **The GPU** (`device=metal`): the same text, boxes and scripts as the CPU on the twelve images,
  confidences within 2.1e-7; the models against onnxruntime as on the CPU (below).

## Speed

M5 Max (the machine shared with other work; `tools/bench/with-lock.sh`), best of 10 runs, image
decoding excluded (`oxinfer ocr --repeat 10`), on the CPU (`--device cpu`; the GPU [below](#the-gpu)):

| image | lines | default | `--plain` | `--plain`, one thread |
|---|---|---|---|---|
| 2000 x 2600 page (two columns, a table, footnotes) | 70 (71 plain: the junk filter drops a "12" read at 0.69) | 71.5 ms (14 pages/s) | 71.0 ms | 316 ms |
| 1200 x 900, seven scripts | 7-8 | 30.1 ms | 25.1 ms | 76 ms |
| 900 x 260, three lines | 3 | 7.2 ms | 7.1 ms | 20 ms |

The page's stages (`--profile`, `--plain`): detector resize 4.0 ms, detector 21.7 ms (1216 x 1600
input), boxes 0.6 ms, crops 2.7 ms, router 5.2 ms (two batches), recognizers 33.1 ms (three
batches of `rec_latin`). onnxruntime 1.32 on the same inputs (its default CPU provider, all cores)
runs the models in 50.8 ms (detector 13.7, router 2.4, recognizers 34.7), 185 ms on one thread;
LiteParse reports 426 ms per page on one core for its whole pipeline.

The convolutions run channels-last: pointwise ones on the shared f32 GEMM (SME on this machine),
depthwise and one-channel ones summing each pixel's taps across the channels, the rest as im2col +
GEMM; each operator's own loops on the thread pool. The detector at high resolution has few
channels (12 to 32), where onnxruntime is about twice as fast; the recognizers run at its speed.
One page uses every core; `Ocria` reads pages from several threads at once (the server runs one
request per model at a time).

### The GPU

`device=metal` (`--device metal`, the load option `device`) runs the detector, the router and
the recognizers on the GPU through the interpreter's Metal operators (`crate::gpu::onnx`,
`metal_onnx.metal`: a direct convolution kernel, pointwise convolutions on 8 x 8 SIMD-group f32
GEMMs, additions, nearest resizes, channel concatenations, the fused ReLU / fake-quantization /
integer-requantization steps; the models' constants uploaded once, a run's tensors in shared
buffers, every operator of a run in one command buffer; the operators it has none for, the
recognizers' CTC heads and the router's pooling, on the CPU in between). rec_latin keeps
onnxruntime's int8 arithmetic there too (its integer sums are exact on the GPU as well). Its
fixed cost per run (a submission, about a millisecond) makes it slower than the CPU on small
inputs, so `device=auto` (the default) takes it only for detector inputs of 1.5 M values and
more (about a 700 x 700 RGB image) and router / recognizer batches of 128 K values and more
(eight 512-pixel lines); `cpu` never.

Same machine and conditions, best of 10, `--plain` in brackets:

| image | `device=cpu` | `metal` | `auto` (default) |
|---|---|---|---|
| 2000 x 2600 page | 77.7 ms (69.6) | 50.4 ms (48.9) | 50.4 ms (50.5) |
| 1200 x 900, seven scripts | 31.8 ms (25.4) | 25.8 ms (22.2) | 29.2 ms (23.0) |
| 900 x 200, two lines | 5.8 ms (5.7) | 7.3 ms (7.2) | 5.8 ms (5.6) |

The page's models (`--plain --profile`): detector 25.3 ms on the CPU, 17.0 ms on the GPU;
recognizers 38.1 and 25.1 ms; router 6.1 and 5.4 ms. (The CPU figures here are a few
milliseconds above the table before: the machine was busier.) The first GPU load compiles the
Metal library (seconds, once per OS shader cache).

## Layout and tables

With LayoutNet loaded (`layoutnet`, the `tablenet.bin` of the repository), each page also gets
its layout (`structure`, on by default), as LiteParse gets it when it parses an image: the stages
are ported from LiteParse (`crates/liteparse/src`, Apache-2.0: `projection.rs`, `table_model/`,
the image page's part of `ocr_merge.rs` and `layout.rs`; `src/ocria/layout/`):

- **the OCR merge**: the lines in the detector's order become the page's text items in points
  (`72 / dpi` a pixel; the image's own resolution, PNG `pHYs` / JPEG JFIF, else 150 dpi, as
  LiteParse sizes an image's page): readings at 0.1 or under and short fragments in a script the
  page doesn't use dropped, table-border characters stripped around numbers, word boxes, the font
  size from the quad's height; a skewed scan (0.3 to 10 degrees, the length-weighted median of
  its long lines) turned upright about the page centre, and a page whose text all runs a quarter
  or half turn round re-expressed upright;
- **the projection**: LiteParse's grid projection, the text laid out on a character grid
  (anchors, snapping, flowing prose, floating items) with the per-item signals LayoutNet reads
  (left / right / centre anchor-group sizes, snap, line block, flowing prose), and the XY-cut
  region tree whose leaves are its last signal;
- **the render**: the page as PDFium renders it at 150 dpi (the image itself at 150 dpi, else
  resampled to that size) in grey with OpenCV's integer weights; its ruling lines (runs of dark
  pixels joined into horizontal and vertical segments) and, area-resampled to a long side of 1024,
  the image branch's canvas;
- **LayoutNet**: 66 features per unit (geometry, text shape, Fourier terms of the centre, the
  projection signals, ruling-line flags) and two ids (block, leaf), pair inputs from the ruling
  lines; the input MLP, the CNN image branch (per-unit samples and stride-32 patch tokens the top
  layers cross-attend to), 6 pre-norm encoder layers, node, class and five pair heads (same
  table, row, column, cell, block), on oxinfer's f32 GEMM;
- **the decoder**: tables from the in-table and same-table logits (average linkage, absorption,
  the header band), each table's cells, rows and columns (span-aware linkage, header rows,
  wrapped-row folds), its lattice when ruling lines enclose it; tables of contents, charts and
  prose read as grids dropped; the layout blocks (same-block components split at wide gaps,
  paragraph breaks and heading changes, each labelled by its units' summed class logits);
- **the page's blocks**: LiteParse's `with_model_blocks`: the tables, then each layout block
  outside them by class (title and heading as headings of level 1 and 2, list item, page header,
  page footer, code, else paragraph), in XY-cut reading order over the block boxes. LiteParse
  leaves LayoutNet's table and figure-text blocks to its own heuristic tables and figures; here
  they stay, as `table_text` and `figure_text`. A page the model skips (under 2 units or 20
  characters, over 1500 units) keeps its projected lines as paragraphs, where LiteParse
  classifies them heuristically.

Boxes come back in image pixels (turned back from a deskewed or upright frame). The Markdown is
oxinfer's, from these blocks: headings as `#` / `##`, list items (`-`, or their own numbers),
tables as pipe tables (several header rows joined) or HTML when a cell spans, code fenced;
LiteParse's own Markdown stage (its heuristic block classifier, 20,000 lines more) is not ported.

### Against LiteParse

LiteParse 2.21.3 (`lit parse IMAGE --format markdown --ocr-backend ocria-nano`, with its
`tablenet.bin` on the Rust path it takes on macOS and the page image given as its OCR input), with
the dump hooks of `tools/liteparse_layout_dump.patch`, on nine pages made by
`tools/ocria_layout_pages.py`: an invoice with a ruled table, a two-column report with headings,
bullet lists and an unruled table, a form with key-value lines, a numbered list and a table with a
spanning header, a financial statement, a ruled schedule with merged and wrapped cells, a
three-column newsletter with running header and footer, the 2000 x 2600 test page of the OCR checks
(turned 1.5 degrees: the deskew), the invoice scanned sideways, and the report as a 300 dpi JPEG:

- **each stage on LiteParse's own OCR results** (`src/ocria/layout/tests.rs`, with
  `OXINFER_LAYOUT_REF` and `OXINFER_LAYOUT_MODEL`): the merged text items identical (text, boxes,
  rotations, font sizes, the deskew angle), the projection's signals identical on every item
  (anchor counts, snap, block, flowing prose, leaf paths), the ruling lines and the image branch's
  canvas identical, the 68 features identical, LayoutNet's node and class logits within 7.7e-6
  (Accelerate's GEMM there, oxinfer's here), and the same tables cell by cell (spans, header
  rows, text), the same 142 layout blocks and the same page blocks (kind, level, text, order).
  On the 300 dpi JPEG, PDFium's resampling to 150 dpi and ours differ on 9% of the canvas pixels
  (0.09 grey levels on average) and the logits by up to 0.02: still the same tables and blocks;
- **the whole pipeline, oxinfer's OCR against LiteParse's** (`tools/ocria_layout_e2e.py`): 207 of
  207 table cells and 73 of 74 page blocks the same; the other one differs by a space the two OCR
  runs read differently on the JPEG ("• Ship" / "•Ship").

Speed (M5 Max, release build, `tools/bench/with-lock.sh`, best of 10, image decoding excluded,
`device=auto`): the layout adds 14 to 18 ms a page: the 2000 x 2600 test page 49.9 -> 63.5 ms,
the 1700 x 2200 report (82 lines) 44.1 -> 62.5 ms, the invoice (42 lines) 30.0 -> 44.9 ms. Of
it: the projection 0.2-0.3 ms, the render and its ruling lines 5-7 ms, LayoutNet 10-12 ms (the
image branch's CNN most of it). LiteParse reports its table model at 20 ms a page on the same
machine.

## Not implemented

- the rest of LiteParse's document parsing: PDF rendering, merging the OCR with a PDF's text
  layer, its heuristic block classifier and Markdown stage (`markdown_layout/`), the document-level
  header / footer dedup.
