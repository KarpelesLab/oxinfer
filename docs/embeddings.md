# Embeddings

sentence-transformers models are recognised by their `modules.json` and run in embedding mode:
the backbone (Gemma 3 with bidirectional attention, Qwen3, Qwen2, or a BERT / XLM-RoBERTa encoder in
`bert.rs`), then the listed modules: pooling (CLS, mean, max, last token), dense projections,
normalization. Named prompts come from `config_sentence_transformers.json`. Each input line
gives one JSON array (or `-p TEXT` one):

```sh
./target/release/oxinfer --model Qwen/Qwen3-Embedding-0.6B --prompt-name query -p "What is the capital of France?"
printf 'Which planet is the Red Planet?\nMars looks red.\nVenus is the hottest planet.\n' |
  ./target/release/oxinfer --model google/embeddinggemma-300m --prompt-name query --similarity
#  1.0000  0.8014  0.5470
#  ...
./target/release/oxinfer --model google/embeddinggemma-300m --dim 256 -p "..."   # Matryoshka
./target/release/oxinfer --model google/embeddinggemma-2 --image photo.jpg           # an image (EmbeddingGemma 2)
./target/release/oxinfer --model BAAI/bge-m3 --sparse -p "..."     # lexical weights {token: weight}
./target/release/oxinfer --model BAAI/bge-m3 --colbert -p "..."    # one vector per token
```

`google/embeddinggemma-300m` is gated on Hugging Face (set `HF_TOKEN`); `unsloth/embeddinggemma-300m`
is an ungated copy. BGE-M3 ships only a `pytorch_model.bin`: `torch.rs` reads `torch.save`
archives directly (a zip of raw storages plus a pickled state dict; only the handful of pickle
opcodes and constructors that state dicts use are understood, and nothing is executed), so the
same goes for its `sparse_linear.pt` / `colbert_linear.pt` heads.

Checked against sentence-transformers float32 (`tools/ref_embed.py`) on short texts in five
languages and a ~2000-token one: cosine between our vector and the reference at least 0.99999 for
EmbeddingGemma (query and document prompts, and at 256 dimensions) and 1 - 1e-6 for
Qwen3-Embedding; BGE-M3's dense vectors 0.99991 or better. fp32 checkpoints are computed with
bf16-rounded matrices; against a reference using the same rounding the vectors agree to about
1e-5, so that is the whole difference. One caveat: on the long text, BGE-M3's ColBERT vectors
have a mean cosine of 0.9995 to the fp32 reference, but one token drops to 0.40. That text is
unusually sensitive: the reference itself in fp16 already drops a token to 0.89. The sparse
weights select the same tokens, with weights within 0.013.

Any BERT, RoBERTa or XLM-RoBERTa sentence-transformers model should work the same way; besides
the three above, all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2, bge-small-en-v1.5,
multilingual-e5-small and mxbai-embed-large-v1 were checked (cosine >= 0.99999 to the reference).
E5 models expect a `query: ` / `passage: ` prefix in the text itself (they define no prompts).
MPNet (all-mpnet-base-v2: BERT plus a shared T5-style relative position bias) is supported too,
cosine >= 0.999991 to the reference (0.999996 against bf16-rounded reference weights).

NomicBERT (`nomic_bert.rs`: rotary positions, fused Wqkv, SwiGLU, dynamic NTK scaling past 2048
tokens) runs nomic-embed-text-v1 and v1.5: cosine >= 0.99998 to the reference (`--trust-remote-code`)
with and without the `search_query: ` / `search_document: ` prefixes, which go in the text (the
model defines no prompts), and on a 4000-token text; 0.999999 against bf16-rounded weights.
v1.5's Matryoshka recipe layer-norms the pooled vector before truncating, which is not a
sentence-transformers module, so `--dim` alone truncates without it.

ModernBERT (alternating global and sliding-window attention) is supported too:
modernbert-embed-base and gte-modernbert-base give cosines of 0.99993 and 0.99998 against the
fp32 reference, and 1 - 1e-6 against one with the same bf16-rounded matrices. modernbert-embed
expects a `search_query: ` / `search_document: ` prefix in the text itself.

Qwen2-based embedders run on the Qwen2 decoder of `qwen.rs` (biased q / k / v projections, no
query / key norms) without the causal mask: the modeling code these repositories ship makes
`Qwen2Model` bidirectional unless asked otherwise, and sentence-transformers never asks.

- Alibaba-NLP/gte-Qwen2-1.5B-instruct (and gte-Qwen2-7B-instruct, the same code and layout; not
  run here): last-token pooling, normalized, 1536 dimensions. Queries take the `query` prompt
  (`Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: `);
  for another task write that format into the text and name no prompt. Documents take none. The
  pooled token is the `<|endoftext|>` that the model's tokenizer class appends
  (`add_eos_token`; its `tokenizer.json` has no template for it, so it is appended here).
- NovaSearch/stella_en_1.5B_v5 (dunzhang's, renamed): mean pooling over every token including
  the appended `<|endoftext|>`, then a linear head, not normalized (as its `modules.json` says).
  Prompts `s2p_query` (retrieval queries) and `s2s_query` (similarity), none for documents;
  512 tokens. The repository has one head per output size, `2_Dense_256` ... `2_Dense_8192`,
  and `modules.json` names the 1024 one: `--dim N` (`dimensions` in the API) uses the head of
  that size rather than truncating, and any other size is an error. Downloading fetches all of
  them (140 MB).

```sh
./target/release/oxinfer --model Alibaba-NLP/gte-Qwen2-1.5B-instruct --prompt-name query -p "how much protein should a female eat"
./target/release/oxinfer --model NovaSearch/stella_en_1.5B_v5 --prompt-name s2p_query --dim 256 -p "..."
```

Against sentence-transformers float32 running that modeling code (`tools/ref_embed.py
--trust-remote-code`), on nine short texts in five languages: gte-Qwen2-1.5B cosine >= 0.99999
with and without the query prompt (largest element difference 6e-4 on unit vectors), the same on
a 1560-word text, and 1 - 1e-6 (1.5e-6 per element) against the reference with the same
bf16-rounded matrices; the reference encoding the nine texts as one left-padded batch gives the
same numbers. stella_en_1.5B_v5: cosine >= 0.999997 for documents and both prompts, in a
right-aligned (left-padded) batch as well, 0.999996 on a text cut at 512 tokens, and >= 0.999997
with each of the other six heads (`--dense 2_Dense_N` on the reference side). Inputs are embedded
one at a time, so there is no padding to mask. stella_en_400M_v5 is not a Qwen2 model (its base
is Alibaba's `new` encoder) and is not supported.

Two details that are easy to get wrong: transformers halves EmbeddingGemma's sliding window for
bidirectional attention (512 in the config means keys less than 257 positions away on either
side), and Qwen3-Embedding pools the `<|endoftext|>` token its tokenizer appends.

From the library (`examples/embed.rs`):

```rust
let dir = oxinfer::hf::resolve("BAAI/bge-m3", None)?;
let files = oxinfer::safetensors::ModelFiles::open(&dir, false)?;
let emb = oxinfer::embed::Embedder::load(&dir, &files, oxinfer::model::Quant::Bf16)?;
let mut e = emb.embed("what is a panda?", None)?;          // or Some("query") with named prompts
e.truncate(256)?;                                         // Matryoshka (--dim)
let e256 = emb.embed_dim("what is a panda?", None, Some(256))?;  // --dim: stella's head of that size, else truncation
let lexical = emb.sparse_tokens(&e);                      // BGE-M3 heads: e.sparse / e.colbert
```

## EmbeddingGemma 2

[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (`model_type:
embedding_gemma2`, 1.5 GB) embeds text, images, audio and video into one 768-dimensional space
(Matryoshka: `--dim 512`, `256`, `128`).

```sh
./target/release/oxinfer --model google/embeddinggemma-2 --prompt-name query -p "a red square next to a blue circle"
./target/release/oxinfer --model google/embeddinggemma-2 --image photo.jpg
./target/release/oxinfer --model google/embeddinggemma-2 --audio clip.wav
./target/release/oxinfer --model google/embeddinggemma-2 --image photo.jpg -p "taken in Lisbon" --prompt-name document
```

With attachments (`--image`, `--audio`, each repeatable) the command gives one
embedding of the attachments followed by the `-p` text, if any (`Embedder::embed_media` in the
library). The server's `/v1/embeddings` takes text.

The text encoder (`embed_gemma2.rs`) is a small bidirectional Gemma-4-style stack: 24 layers of
width 512, query / key norms (and an unscaled norm on the values), unscaled attention scores,
sandwich norms, a `layer_scalar` per layer. Five layers in six are `sliding_attention` (keys at
most 512 positions away, on either side: 4 heads of 256 over 2 key / value heads, RoPE base 1e4),
every sixth is `full_attention` with its own shape (`per_layer_config`: 4 heads of 512 over one
key / value head, RoPE base 1e6). After the MLP each layer adds a third block, gated by a
per-layer input: `ple = norm(per_layer_model_projection(x0) / sqrt(512))` is computed once from
the input embeddings `x0` (a projection only, with no per-layer token table as in Gemma 4), and
layer `i` adds `norm(per_layer_projection(gelu(per_layer_input_gate(x)) * ple[i]))`. The final
norm is followed by `embedding_projection` (512 to 768) on every token; the sentence-transformers
modules are then mean pooling over all tokens and normalization.

Images and audio go through Gemma 4's own towers (`vision_tower` / `embed_vision`, 16
layers; `audio_tower` / `embed_audio`, 12 layers; see
[Image, video and audio input](multimodal-input.md)), read here from the root of the checkpoint
instead of under `model.`. Their soft tokens take the place of the placeholders the processor
writes (`<|image>`, one `<|image|>` per soft token, `<image|>`; at most 280 soft tokens per image;
`<|audio>` ... `<audio|>`, one token per 40 ms), between `<bos>` and `<eos>`, and are pooled like
text tokens.

Checked against sentence-transformers 6.3 on transformers 5.23, float32, CPU
(`tools/ref_embed.py` for text):

| Input | Cosine to the reference | Max difference |
|---|---|---|
| 7 short texts in six languages and code, and a 2581-token one; no prompt, `query`, `document` | 1.000000 for each | 3.9e-7 |
| a 640x480 image (266 soft tokens) | 1.000000 | 8.4e-7 |
| the same on CUDA | 1.000000 | 7.2e-7 |
| a 3.2 s audio clip (80 soft tokens) | 1.000000 | 1.2e-6 |
| the image followed by a text, without and with the `query` prompt | 1.000000 | 1.3e-6 |

Video is not supported yet: the model's video processor lays the frames out its own way (up to 32
frames at 1 fps and 140 soft tokens each, as `<|video|>` soft tokens without timestamps), and
`--video` is refused rather than fed in Gemma 4's layout.
