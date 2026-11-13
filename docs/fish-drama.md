# Fish Audio Drama 3 (not supported yet)

[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) (`model_type: fish_omni`,
`FishOmniModel`; Fish Audio Community License: research and non-commercial use, commercial use
under a revenue threshold) is a text-to-speech model with up to 16 speakers per text, voice
cloning from reference clips and editing by continuation. **oxinfer does not load it.** This
page records what the checkpoint holds and what each stage would need, from the repository's
small files, the safetensors headers (read by HTTP range requests) and Fish Audio's reference
code. Nothing below was run: no weights were downloaded and no code was written for it.

## The checkpoint

| File | What |
|---|---|
| `model-000NN-of-00015.safetensors` (41.5 GB, 62245 tensors) | 32.2B expert weights as FP8 (`F8_E4M3`) with an F32 `weight_scale_inv` per block of 128 x 128, and 3.7B bf16 weights for everything else |
| `codec.pth` (1.87 GB) | the audio codec, a PyTorch pickle (not examined beyond its size) |
| `tokenizer.json` (20.8 MB) | 252288 ids: the text vocabulary, `<|semantic:0|>` .. `<|semantic:4095|>` at 248099 .. 252194, `<|speaker:0|>` .. `<|speaker:15|>`, `<|voice|>` |
| `config.json`, `chat_template.jinja` | `text_config`, `audio_decoder_config`, `quantization_config: bsgemm-moe` |

## The pipeline (from the reference code)

The reference is `fish_speech/models/text2semantic/fish_omni/` of
[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) (`backbone.py`, `layers.py`,
`moe.py`, `fp8_moe.py`, `tower.py`, `generate.py`, `chunking.py`, `model.py`), with tests on small
models (`tests/test_fish_omni_backbone.py`, `tests/test_fish_omni_generate.py`); the same model is
in `sglang_omni/models/fishaudio_drama3/` of SGLang-Omni and in vLLM-Omni. Every stage has public
code.

**Prompt** (`generate.build_drama3_prompt`). Rows of 11 ids, `[token, code 0 .. code 9]`: text
rows carry a token and zeros, audio rows `[semantic_start + code 0, code 0 .. code 9]`. A system
turn (`convert the provided text to speech`, or with references `... reference to the
following:` and per reference `[Reference i]`, `Text: <|speaker:k|>transcript`, `Speech:` and the
clip's codes), earlier takes as finished user / assistant turns, then
`<|im_start|>user`, the text behind a speaker tag, `<|im_start|>assistant` and `<|voice|>`. Every
literal is tokenized on its own.

**Backbone** (`model.language_model.*`, 40 layers, width 2048). The shape of Qwen3.5 / 3.6
35B-A3B, which oxinfer runs as a chat model: three layers of four are Gated DeltaNet (16 key
heads and 32 value heads of 128, convolution kernel 4), the fourth is attention (16 heads of 256
over 2 key / value heads, normalized q and k, a quarter of each head rotated, an output gate);
every layer has 256 experts of width 512 (8 per token) and a gated shared expert; zero-centred
RMS norms. On audio rows the input is `(embedding of the main token + embedding of code 1) /
sqrt(2)` (`vq_backbone_codebooks: 1`).

**Codebook tower** (`audio_decoder*`, `tower.py`). The backbone's logits over the end token and
the 4096 semantic tokens give the row's main token (code 0). Then 4 shared layers and 9 residual
layers run on the backbone's hidden state: each mixes a code's embedding into the state
(`eh_proj` on the two normalized halves), runs one transformer block of width 2048 that attends
over all earlier rows (its own KV cache), and normalizes. The shared layers take code 0 (the first
also the codes 2 .. 9 of the row just read, which the backbone did not see); residual layer `i`
takes code `i` and its head gives code `i + 1` (heads of 4096 logits; the residual codebooks hold
1024 codes).

**Sampling.** Temperature, top-k and top-p on the main token and on each code; a main token that
repeats one of the last 16 frames' is drawn again with fixed settings (temperature 1.0, top-k 30,
top-p 0.95). The model card's numbers use temperature 1.0, top-p 0.9, top-k 30.

**Codec** (`codec.pth`). Fish Audio's DAC variant (`fish_speech/models/dac/modded_dac.py` and
`rvq.py`: one semantic codebook of 4096 and nine residual ones of 1024). The decoder turns the
frames into audio; the encoder turns reference clips into the codes the prompt takes.

**Long text** is cut into chunks (`chunking.py`), each generated with the earlier ones as
history.

## What oxinfer has and what is missing

| Stage | Reusable | Missing |
|---|---|---|
| Backbone | the Qwen3.5 MoE hybrid (`src/qwen.rs`: Gated DeltaNet, attention, experts), FP8 E4M3 matrices with block scales (`ggml::GgufMatrix::fp8`) | a loader for `fish_omni`'s config keys and tensor names, the added code embedding on audio rows, hidden states handed to the tower |
| Codebook tower | | all of it: 13 layers with their caches, 9 heads |
| Sampling, prompt, chunking | the tokenizer | all of it |
| Codec | | decoder and encoder (a different codec from SNAC and from ZONOS3's ZAC) |
| Checks | the method of [Verifying against the reference](verification.md) | small random checkpoints through `fish_omni` for the backbone and the tower, `codec.pth` itself for the codec (1.9 GB), then the real 41.5 GB checkpoint |

The backbone is the part closest to existing code; the tower and the codec are new. The 35B
checkpoint in FP8 needs about 45 GB of memory.
