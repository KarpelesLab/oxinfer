# Liquid LFM3

[LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) (`model_type: lfm3`,
`Lfm3ForCausalLM`; 1.7B parameters, 3.5 GB of bf16 safetensors) is a hybrid of gated short
convolutions and attention, made for phones and laptops.

```sh
./target/release/oxinfer --model LiquidAI/LFM3-1.7B -p "Why is the sky blue?"
./target/release/oxinfer --model LiquidAI/LFM3-1.7B --chat --think --tools
./target/release/oxinfer --model LiquidAI/LFM3-1.7B --quant q8 -p "..."     # 8-bit weights at load
./target/release/oxinfer serve --model lfm=LiquidAI/LFM3-1.7B
```

## The model

24 layers, each named by `layer_types`: 14 `conv`, 5 `full_attention`, 5 `sliding_attention`
(full attention on every fourth layer from the third, sliding attention between them). It runs on
the Qwen code path (`crate::qwen`), the conv as one more mixer (`crate::lfm3`):

```text
layer:  x += post_operator_norm(op(operator_norm(x)))        # op: conv or attention
        x += post_ffn_norm(w2(silu(w1(n)) * w3(n))),  n = ffn_norm(x)
conv:   [B | C | u] = in_proj(n);  y = out_proj(C * conv7(B * u))
attn:   q, k RMS-normed per head, RoPE, softmax over the causal keys;
        each head's output times sigmoid(gate_proj(n)[head]); o_proj
```

- **Conv blocks**: a depthwise causal convolution of width 7 (`conv_L_cache`) over the product of
  two of the three projections, gated by the third. Its state is the last 6 inputs per channel
  (12288 floats per conv layer): recurrent state, like Qwen 3.5's Gated DeltaNet. A conversation
  whose earlier turns change starts over instead of being cut back (the prompt cache stores
  snapshots, as for Qwen 3.5).
- **Sandwich norm** (`use_sandwich_norm`): an RMSNorm before each operator and MLP, and another
  on its output before it joins the residual.
- **Attention**: 16 heads of 128 over 4 key / value heads, query / key norms, and a sigmoid gate
  per head (`gate_proj`, one row per head). `sliding_attention` layers see the last 256 keys
  (`sliding_window`), the query's own included; the whole KV cache is kept.
- **RoPE per layer type** (`rope_parameters`): base 1e6 with `partial_rotary_factor` 0.5 for full
  attention, base 1e4 for sliding attention. The partial rotation keeps the frequency progression
  of the whole head dimension and zeroes the second half of the frequencies: of the rotate-half
  pairs `(i, i + 64)`, those with `i < 32` rotate.
- The final norm is `embedding_norm`; the output head is tied to the embeddings.

Decoding runs on the CPU kernels (the GPU decode plans do not cover the conv and the gated heads;
the prefill matmuls still use a GPU when there is one). Speculative decoding (prompt lookup) is
on as for other models; MTP and DSpark do not apply.

`transformers` 5.23 has an `lfm3` class of its own that differs from the one the repository
ships: its attention gate is an optional `g_proj` (the checkpoint's is `gate_proj`) and its RoPE
has no zero-tail partial rotation. It was not run here; the reference is the repository's
modeling code (`modeling_lfm3.py`, `tools/ref_logits.py --trust-remote-code`).

## Chat format and tools

`<|startoftext|>`, then ChatML turns. The system turn holds the system prompt and, after it, the
tools as bare JSON functions between `<tools>` and `</tools>` (no instructions; without a system
prompt or tools there is no system turn). The reply opens with `<think>` when thinking is on and
with an empty `<think></think>` otherwise. Tool calls are Qwen 3.5's
`<tool_call><function=NAME><parameter=KEY>` blocks; each result goes back in a `tool` turn as
`<tool_response>`. `--tools`, the server's `tools`, forced calls and `response_format` work as for
Qwen 3.5 (see [Tool calling](tool-calling.md)).

## Agreement with the reference

Against the repository's modeling code on `transformers` 5.23, float32, CPU
(`tools/ref_logits.py --trust-remote-code`; the prompt as the chat template renders it, logits
of its last token, then 24 greedy tokens):

| Check | Result |
|---|---|
| "What is the capital of France? Answer in one sentence." (23 tokens) | max logit difference 6e-5, same greedy tokens to the end of the reply |
| the same with thinking on (prompt ends in `<think>`) | max logit difference 5e-5, same 24 greedy tokens |
| a 1301-token prompt (notes to summarise; past the 256-key window of the sliding layers) | max logit difference 1.1e-4, same 24 greedy tokens |
| the 1301-token prompt fed one token at a time, or in chunks of 100, vs in one pass (conv state, windows, both RoPEs) | max logit difference 1e-4, same top 5 |
| 300 greedy tokens after that prompt: prompt lookup off, on (128 of 210 drafts rejected: the conv states rewound), and on CUDA | identical tokens |
| the chat template's renderings (system or not, tools, a call and its result, thinking on / off) | identical text (unit test `lfm3_matches_template`) |
| `tools/openai_check.py --chat` | all checks pass (tool calls, streamed and forced calls, JSON schema, reasoning, log probabilities) |

The greedy comparison runs the reference without a repetition penalty. The checkpoint's
`generation_config.json` asks for temperature 0.1, top-k 50 and a repetition penalty of 1.1, which
`transformers` applies to the prompt's tokens too, and that changes what the model does with
thinking off: on the 1301-token prompt above the raw logits put `<think>` first (20.2, against
19.6 for the first word of the answer) although the prompt ends in an empty `<think></think>`, so
plain greedy decoding thinks anyway, in the reference as here; the penalty demotes `<think>`
(it is in the prompt) and the reference then answers directly. oxinfer's default is greedy without
a penalty, and `--repetition-penalty` covers the generated tokens only, so with thinking off the
model may still open a thinking block on such prompts (the command line shows it dimmed, as
thinking). Short prompts answer directly.

## LFM3-Omni

[LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B) (`model_type: lfm3_omni`,
`Lfm3OmniForConditionalGeneration`; 2.2B parameters, 4.3 GB of bf16 safetensors plus the 147 MB
codec in `audio_tokenizer/`) is the LFM3 language model above (without thinking) with an audio
encoder, a vision tower and a speech head. It reads text, images and audio, and answers in text,
in speech, or in both interleaved (`src/lfm3_omni/`).

```sh
M=LiquidAI/LFM3-Omni-2B
# speech recognition, translation, questions about a recording
./target/release/oxinfer --model $M --system "Perform ASR in English." --audio talk.wav -p ""
./target/release/oxinfer --model $M --audio clip.wav -p "What is happening in this recording?"
# questions about an image (typed, or spoken: --image photo.png --audio question.wav)
./target/release/oxinfer --model $M --image photo.png -p "What is written on the sign?"
# text to speech: --voice female (default) or male, --system for another language
./target/release/oxinfer --model $M --speak "The quick brown fox jumps over the lazy dog." -o out.wav
# voice chat: the answer as text on standard output and as speech in the WAV
./target/release/oxinfer --model $M --audio question.wav -o reply.wav
./target/release/oxinfer --model $M -p "Tell me about Mount Fuji." -o reply.wav
```

Text replies go through the normal chat path and the engine (`--chat`, `oxinfer serve`,
`Engine::send_with_media`): the placeholders and the two markers that open speech are taken out
of the output head, which is what the reference's `generation_mode="text"` does. A reply that
writes a WAV (`--speak`, or `-o FILE.wav` with a prompt) runs `src/lfm3_omni_cli.rs`, from the
command line only.

| Task | System prompt (the model card's) | Command line |
|---|---|---|
| speech recognition | `Perform ASR in <language>.` | `--system ... --audio F -p ""` |
| speech translation | `Please translate the following spoken <source> into <target> text.` | `--system ... --audio F -p ""` |
| audio or image questions | none needed | `--audio F` / `--image F` with `-p` |
| text to speech | `Perform TTS in <language> using a <female / male> voice.` (the default of `--speak`, in American English) | `--speak TEXT -o F.wav` |
| voice chat | `Respond with interleaved text and audio.` (the default with `-o F.wav`) | `--audio F -o F.wav`, `-p TEXT -o F.wav` |

`--system` replaces the default. Without `--system` (and without `--no-system`) the text path
uses oxinfer's own default system prompt, as for other chat models.

### How it works

```text
audio, 16 kHz -> log-mel (128 bins, 10 ms hop) -> three stride-2 convolutions (8 frames to one)
              -> 17 causal LFM2 layers (width 768, attention over the last 32 positions)
              -> projector -> one <|audio|> soft token per 80 ms
image         -> smart resize to multiples of 32 pixels (65536 to 2097152 pixels), 16x16 patches
              -> 14 layers (gated depthwise 3x3 convolutions over the patch grid, bidirectional
                 attention with axial 2D RoPE) -> projector folding 2x2 patches
              -> one <|image|> soft token per 32x32 pixels
reply         -> text tokens from the LM head; speech frames of 12 codes from the depth decoder
codes         -> Liquid codec decoder -> 24 kHz audio, 1920 samples (80 ms) per frame
```

- **Prompt**: the LFM3 chat format, each image or clip as a run of placeholders between
  `<|vision_start|>` and `<|vision_end|>` (audio too), attachments before the text of the turn.
  Positions are sequential (no M-RoPE).
- **Audio features** (`Lfm3OmniFeatureExtractor`): Whisper's Slaney filterbank and Hann window
  of 400 samples, but no centering (the clip is zero-padded to whole hops), `log10` floored at
  -12 with no clamp relative to the maximum, and no padding to 30 s.
- **Speech output**: after `<|audio_start|>` every slot is a speech frame; after
  `<|mixed_start|>` the reply alternates 6 text tokens and 12 frames until `<|text_end|>`, after
  which the rest of the speech follows. A frame is read off the language model's hidden state by
  the depth decoder, a bidirectional 3-layer LFM3 stack over 12 slots (one per codebook: a slice
  of a projection of the hidden state plus the embedding of the slot's code, or a mask
  embedding). The codebooks are revealed 4 at a time over 3 passes in a random order; a frame
  holding the end code (2016) ends the speech. The frame goes back into the language model as
  the sum of its 12 code embeddings.
- **Sampling**: the checkpoint's settings for the codes, temperature 0.4 and a repetition
  penalty of 1.2 over the last 64 frames per codebook (`--temp`, `--top-k`,
  `--repetition-penalty`, `--seed`); the text tokens of a spoken reply are greedy.
- **Liquid codec** ([LiquidAI/Liquid-Codec](https://huggingface.co/LiquidAI/Liquid-Codec),
  `src/lfm3_omni/codec.rs`): each code is 4 finite-scalar-quantization digits (levels 8, 7, 6,
  6); the 48 digits of a frame are projected to 512, repeated 6 times, run through 8 LFM3 layers
  (attention over the last 30 rows) and mapped to the log magnitude and phase of 6 STFT frames
  (FFT 1280, hop 320), which an inverse STFT turns into audio. Only the decoder is implemented.

Everything was run on the CPU (`--device cpu`). On 16 threads of an x86-64 desktop: 266 prompt
tokens/s, 17-19 text tokens/s; 10 s of audio encoded in 0.1 s; a 960x686 image (630 tokens) in
2.8 s; speech at 1.1-1.3 times real time (about 65 ms per frame of 80 ms: three depth-decoder
passes and one language-model step).

### Agreement with the reference

Against the code in the repository (`transformers` 5.23, float32, CPU; `tools/ref_lfm3_omni.py`
runs its processor and model on the same files and reads oxinfer's `--dump-mm`, `--dump-logits`,
`--print-prompt` and `--dump-pcm`):

| Check | Result |
|---|---|
| log-mel features, 10.4 s clip (1042 frames) | max difference 1e-4, relative 1.5e-6 |
| audio encoder output / projected tokens (131 tokens) | relative 4.4e-7 / 4.5e-6 |
| image pixels (960x686 PNG, 60x42 patches) | identical |
| vision tower output / image tokens (630 tokens) | relative 7.8e-6 / 1.1e-5 |
| expanded prompts (ASR, image question, TTS) | identical token ids |
| last-token logits, ASR prompt (155 tokens) | max difference 4.0e-5, same top 5 |
| last-token logits, image question (676 tokens) | max difference 5.7e-5, same top 5 |
| greedy text: three clips transcribed (12 to 36 tokens), a question about a clip (14 tokens), 60 tokens about the image | identical tokens |
| speech codes, most likely code, codebooks revealed in index order: TTS (48 frames), voice chat from text (9 text tokens, 32 frames) and from audio (14 text tokens, 40 frames) | identical codes and text |
| codec waveform from those codes | 111 to 114 dB SNR |
| 16 English sentences spoken with the default sampling (two voices), transcribed by whisper-small | 1 word error in 190 (0.5%) |

```sh
M=~/.cache/oxinfer/hub/LiquidAI--LFM3-Omni-2B
./target/release/oxinfer --model $M --device cpu --system "Perform ASR in English." --audio clip.wav -p "" \
    --print-prompt > prompt.txt
./target/release/oxinfer --model $M --device cpu --system "Perform ASR in English." --audio clip.wav -p "" \
    --dump-mm ours --dump-logits ours.bin
./venv/bin/python tools/ref_lfm3_omni.py $M --audio clip.wav --system "Perform ASR in English." \
    --ours ours --ours-logits ours.bin --ours-prompt prompt.txt --generate 60
./target/release/oxinfer --model $M --device cpu --speak "The quick brown fox." -o out.wav --temp 0 \
    --repetition-penalty 1 --speak-json '{"fixed_order": true}' --dump-pcm ours.f32
./venv/bin/python tools/ref_lfm3_omni.py $M --system "Perform TTS in American English using a female voice." \
    --prompt "The quick brown fox." --speech audio --ours-codes ours.f32.codes.json --ours-pcm ours.f32
```

The reference draws the order in which a frame's codebooks are revealed from torch's generator,
so the codes are compared with that order fixed on both sides (the tool replaces
`torch.randperm`; `--speak-json '{"fixed_order": true}'` here) and with the most likely code;
sampled speech is checked by the word error rate only. The reference needs 16 kHz WAV input for
the comparison (the tool does not resample). JPEG files decode to slightly different pixels than
Pillow's decoder gives (the comparison above uses a PNG): on the same picture as a JPEG the
last-token logits moved by 0.33 and the greedy reply's wording changed after a dozen tokens.

Not implemented: video input (and a video's soundtrack), speech frames in the prompt (a
conversation whose earlier turns were spoken; the codec's encoder), the streaming encoder and
live transcription (`prefill_audio`, `transcribe_live`), spoken replies through the engine and
the server, sampling of the text tokens of a spoken reply, batches, and the GGUF release. The
standalone LiquidAI/Liquid-Codec repository was not loaded (the codec is read from the
checkpoint's `audio_tokenizer/`). Tool calls use LFM3's format and were not tested on this
checkpoint; neither were GPU devices, `--quant` or languages other than English.
