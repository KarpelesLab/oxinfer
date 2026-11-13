# Nemotron 3.5: Lightning (text) and Transcribe (speech to text)

Two checkpoints of NVIDIA's Nemotron 3.5 family run, on the same language model code
(`src/nemotron.rs`):

| | Nemotron 3.5 Lightning (`nemotron_h`) | Nemotron 3.5 Transcribe (`nemo_speechlm`) |
|---|---|---|
| checkpoints | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16, its GGUFs (ggml-org/, bartowski/, llama.cpp `nemotron_h_moe`), and the earlier Nemotron-H / Nemotron 3 models of the same `model_type` | nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B (NeMo's export, one 67.5 GB safetensors file) |
| what it is | a chat model: 52 blocks of 2688, 31.6B parameters, about 3.5B read per token | the same language model, fine-tuned, behind a 730M-parameter audio side (`src/nemotron_asr.rs`) |
| use | `--prompt`, `--chat`, `--tools`, the server's chat endpoints | `--transcribe FILE`, the server's `/v1/audio/transcriptions` and `/v1/audio/translations`; also a text model (`--prompt`) |

```sh
./target/release/oxinfer --model ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:Q4_0 --prompt "Why is the sky blue?"
./target/release/oxinfer --model nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --transcribe talk.mp3
./target/release/oxinfer --model nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --quant q8 --transcribe talk.mp3 --format json
./target/release/oxinfer --model nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --transcribe talk.mp3 --translate
./target/release/oxinfer --model nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --transcribe meeting.mp3 \
    --initial-prompt "Transcribe this audio with punctuation and capitalization. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
./target/release/oxinfer serve --model nemotron=nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B,quant=q8
curl -s localhost:8080/v1/audio/transcriptions -F model=nemotron -F file=@talk.mp3
```

## The language model (Nemotron-H)

As transformers' `NemotronHForCausalLM`: a stack of single-mixer blocks, `x += mixer(rms_norm(x))`,
the mixer of each block given by the config's `layers_block_type` (Lightning: 23 Mamba-2, 6
attention and 23 MoE blocks, `MEMEM*EMEMEM*E...`):

- **Mamba-2**: one projection of the 2688-wide input gives a gate (4096), the convolution input
  `x | B | C` (4096 + 8 x 128 + 8 x 128) and a time step per head (64 heads of 64). A causal
  depthwise convolution (kernel 4, with bias) and SiLU; then per head a 64 x 128 state,
  `S <- exp(dt A) S + dt x B^T` with `dt = softplus(dt + bias)` and `A = -exp(A_log)`,
  read as `S C + D x`, the 8 heads of a group sharing `B` and `C`. The read times `silu(gate)` is
  RMS-normed per group and projected out. The state is 2 MB per block whatever the context; the
  reference computes a prompt through the chunked scan (128 tokens at a time) and oxinfer token
  by token, the same recurrence.
- **Attention**: grouped-query causal attention (32 heads of 128 over 2 key-value heads) with no
  position encoding at all: the Mamba blocks carry the order.
- **MoE**: a sigmoid router with a selection bias, the top 6 of 128 experts renormalized and
  scaled by 2.5, plus one shared expert. The experts are two-layer MLPs with a squared ReLU,
  `down(relu(up x)^2)`, with no gate projection: they do not run on the shared expert code of
  the other MoE models (`moe.rs`), which is built around `act(gate) * up`.
- **MLP**: the same two-layer MLP, dense. Lightning has none; the earlier Nemotron-H models
  that do were not loaded (the tiny checkpoints have one).

**The time step is not clamped**, and this is where the references disagree. The config carries
`time_step_limit: [0, inf]`, the clamp of the Mamba-2 scan, and `time_step_min: 0.001`, the lower
end of the range `dt_bias` is initialized in. NVIDIA's training code (Megatron-LM's Mamba mixer,
and NeMo Automodel's `nemotron_v3` layers, the backend the Transcribe checkpoint was trained
with), its serving code (vLLM's, which the model card's numbers come from) and llama.cpp give the
scan `dt_limit = (0, inf)`: no clamp, and that is what oxinfer does. transformers 5.24 instead
floors `dt` at `time_step_min` whenever it runs the scan over several tokens (a prompt), and
does not in its cached one-token steps, so its uncached forward and its cached decoding are two
different functions of the same weights. The floor is not small on the real weights: at the last
prompt positions of a transcription request it moves the logits by up to 17 (mean KL 0.28 between
the two; the top token stays). oxinfer reproduces transformers exactly when asked to
(`NemotronSession::dt_min`, which the check tools set), and that is how the agreement below is
measured; the default follows NVIDIA's own stack.

Not done: the multi-token prediction head (`mtp.*`, the `mtp-*` GGUF files: an attention block
and a MoE block behind `eh_proj(enorm(embedding) | hnorm(hidden))`; transformers does not run it,
NeMo Automodel's `nemotron_v3/mtp.py` and llama.cpp do, so there is a reference), expert groups
and the MoE latent projection (no released checkpoint uses them), weight streaming (`--stream`),
and speculative decoding of any kind: verifying a draft means going back to an earlier position
when it is refuted, which the Mamba state cannot do without snapshots, so prompt-lookup rounds
are off too. Decoding runs on the CPU kernels (an accelerator takes the prefill matmuls).

### Checkpoints

Told apart by their tensor names:

- the release (`backbone.layers.{l}.mixer.*`, one module per expert), which `save_pretrained`
  writes too;
- NeMo's speech LM export (`llm.model.layers.{l}.mixer.*`): Mamba's `A_log`, `D` and `dt_bias`
  under `_fp32_params`, and the experts stacked with the two matrix axes swapped
  (`gate_and_up_projs` `[128, 2688, 1856]`, `down_projs`). They are transposed into owned
  matrices at load, at the precision asked for: by their sizes 59 GB in bf16 on top of the
  mapped file (the test machine's 125 GB hold it), 30 GB with `--quant q8`, 15 GB with
  `--quant q4`; 53 to 69 s on 64 cores;
- llama.cpp GGUF files (`nemotron_h`, `nemotron_h_moe`), named onto transformers' tensors
  (`nemotron::gguf_name`): a block with key-value heads is attention, one with a feed-forward
  width and a router is MoE, the others Mamba-2; `ssm_a` holds `-exp(A_log)`. The `pixtral`
  pre-tokenizer is Mistral's tekken split. An MTP block stored after the model's is skipped.

`--quant q8|q4` quantizes the matrices of a bf16 checkpoint at load as for the other families.

### Chat format

The template (the same for Lightning and Transcribe) is ChatML with three particulars. It always
writes a system turn, empty when there is no system prompt and no tools
(`<|im_start|>system\n<|im_end|>\n`). Tools are declared in that turn as XML (`<function>`,
`<name>`, `<parameters>` with one `<parameter>` per property, the keys the template does not
know as elements of their own) and called as Qwen 3.5 does, `<tool_call><function=NAME>
<parameter=K>...`, the results returned as `<tool_response>` blocks of a user turn. The reply
opens with `<think>`: `--think` leaves it open, without it the block is closed at once
(`<think></think>`), which is also how the template writes a past reply. The renderings with and
without a system prompt, thinking, tools, a call, its results and a second turn are unit tests
against transformers' `apply_chat_template` (`chat::tests::nemotron_matches_template`).

[Runtime interventions](interventions.md#runtime-interventions-neuron-edits-steering-abliteration-sae-features):
the residual stream after the embeddings and after each block (there is no point inside a
block); neurons are the input of `down_proj` of a dense MLP, of the shared expert and of the
routed experts. Checked without a reference only.

## Speech to text (Nemotron 3.5 Transcribe)

A speech LM: the audio becomes soft tokens that take the place of `<|audio|>` tokens in the
prompt of the language model, which then writes the transcript. The audio side follows
transformers' `Nemotron3_5TranscribeForConditionalGeneration`, a port of NeMo's
`AudioPerceptionModule` with its `ParallelExpertEncoder`:

- **Features**: pre-emphasis 0.97, a centered STFT (512 points, a 25 ms Hann window every 10 ms),
  128 Slaney mel bins of the power spectrum, `ln(x + 2^-24)`. The window and the filter bank are
  the checkpoint's own bf16 tensors. `samples / 160` frames are valid; nothing is normalized
  here.
- **Speech recognition encoder** (600M): the features normalized per mel bin over the recording,
  eight frames stacked per position (80 ms, 12.5 positions per second) and projected to 1280; 32
  pre-norm transformer layers with LayerNorm on the queries and keys of each head, rotary
  positions on half of each head, and GELU MLPs. A recording longer than 500 positions (40 s) is
  encoded in chunks of 500 that see 50 positions of context on each side.
- **Speaker encoder** (Nemotron 3 Diarization, a streaming Sortformer, 99M): the raw features
  through 31 layers of 512 of the same kind, a projection to 192, a sub-pixel convolution back to
  one frame per 10 ms and a two-layer head give the activity of up to eight speakers in order of
  arrival. Chunks after the first also attend to positions kept from the earlier ones: a FIFO of
  the latest 40, and the arrival-order speaker cache of 188 the FIFO overflows into, compressed
  to its best-scored positions per speaker (log-odds of the speaker with the others silent,
  boosted for the strongest, a silence slot per speaker).
- Each position's speaker activities (mean probability over its eight frames, above 0.5) go
  through a LayerNorm and select fixed sinusoidal speaker embeddings, added to the LayerNorm'd
  encoder output; a linear projection to 2688 gives one soft token per position.

The prompt is NeMo's training prompt through the chat template: an empty system turn, the
instruction, a space and the audio tokens in the user turn, and a reply that opens with a closed
`<think></think>`. The default instruction is the model card's verbatim transcription prompt
(`Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions.
Formatting: punctuation and capitalization. Output: transcript only.`). Decoding is greedy up to
`<|im_end|>`, at most twice the audio tokens plus 64 tokens, and the model card's loop guardrail
is applied to the text (a character repeated more than 20 times, or a block repeated 20 times,
becomes one copy).

The same commands as [Whisper](whisper.md#speech-to-text-whisper), with these differences:

| | |
|---|---|
| `--transcribe FILE` | the whole recording is one prompt and one segment (the model card allows up to an hour; nothing longer than 97.6 s was tried); several files are transcribed one after the other |
| `--initial-prompt TEXT` (the server's `prompt`) | replaces the instruction: any prompt of the model card (omit fillers, context terms, speaker tags `<spk:N>`, a translation into another language, a question about the audio) |
| `--translate` (`/v1/audio/translations`) | `Please translate the given speech to English.` |
| `--language` | not used: the model transcribes the language as spoken and does not report one |
| `--timestamps`, `--format srt\|vtt`, `verbose_json` | one segment or cue from 0 to the end of the recording |
| `--word-timestamps` | refused: the word timestamp head is not read (below) |
| `--beam`, `--best-of`, `--fallback`, `--prefix`, `--condition-on-prev` | Whisper's; refused or without effect |
| `--temp T` | samples at that temperature instead of greedy decoding |
| `--quant q8\|q4` | the language model's matrices; the encoders stay bf16 |

Through the library, `whisper::Transcriber` loads either family (`info().family`), and
`Transcriber::speech_lm()` gives the `nemotron_asr::SpeechLm` for requests beyond transcription
(a system prompt, reasoning, a token callback).

Not done:

- **Word timestamps and diarization output.** The checkpoint has a CTC head for them
  (`ctc_timestamp.*`: a transposed convolution, two transformer layers and a classifier over a
  23,862-piece SentencePiece vocabulary, on the speech encoder's states), which transformers
  force-aligns the transcript on (`align_word_timestamps`, with the speaker activities). Neither
  the head nor the alignment is implemented, so there are no word or segment times, and the
  speaker activities are used inside the model only.
- **MTP speculative decoding** (the checkpoint's `llm.mtp.*`, a repeated layer trained for two
  predict depths) and the DFlash2 draft; see the language model's list above.
- **Batching**: several recordings are transcribed one after the other.
- **Audio in a chat** (`--audio` with `--prompt`, the server's chat endpoint with audio parts).
- The FP8 and NVFP4 variants of the checkpoint were not looked at.

## Agreement

transformers 5.24 (the commit that adds `nemotron3_5_transcribe`), float32 on the CPU, eager
attention.

### Tiny random checkpoints

`tools/nemotron_tiny.py` (the language model: 11 blocks of every type, Mamba-2 over several scan
chunks with heads below the time step floor, grouped-query attention, MoE and a dense MLP)
compared by `examples/nemotron_check.rs`; weights rounded to bf16 so that oxinfer's bf16 matmuls
are exact.

`nemotron_check` runs the three combinations: transformers' floor in every pass (its uncached
forward), the floor in the prompt only (its cached decoding), and no floor (oxinfer's default).

| check, with transformers' floor | prefill: max logit diff / mean KL | 16 greedy decode steps: cached, uncached |
|---|---|---|
| the release's layout, 150 tokens | 2.6e-6 / 1.0e-13 (150/150 top-1) | 2.6e-6, 1.8e-6 (16/16) |
| 150 tokens in passes of 7, of 1 | 3.1e-6, 3.1e-6 | 2.7e-6, 1.8e-6 |
| the module's own layout (stacked experts) | 2.6e-6 | 2.6e-6, 1.8e-6 |
| NeMo's layout (experts transposed at load) | 2.6e-6 | 2.2e-6, 2.2e-6 |
| as a GGUF (`tools/nemotron_gguf.py`) | 2.4e-6 | 2.6e-6, 1.9e-6 |
| the release's tokenizer and chat template, 33 tokens (`--tokenizer DIR --chat FILE`) | 3.3e-6 / 1.1e-13 (33/33) | 3.4e-6, 3.8e-6 (16/16) |

The greedy tokens are the reference's at every step. Without the floor the same checkpoint is
1.05 away from transformers over the prompt (145/150 top-1) and 0.38 over the decode steps: its
`dt_bias` is drawn so that some heads sit below the floor, which is what makes the rows above a
test of it.

The command line end to end on the last checkpoint: the prompt text is the template's, its 33
token ids are the reference tokenizer's, and the 16 greedy tokens are the reference's (this
prompt is too short for the floor to change a token).

llama.cpp (master, CPU) reads the tiny GGUF (without the dense MLP block, which its
`nemotron_h_moe` does not have), which makes `tools/nemotron_gguf.py`'s layout the published
one. Over the prompt's last position and 16 greedy steps its logits are within 2.7e-2 of
oxinfer's unclamped ones (mean KL 4.7e-6, 17/17 top-1) and within 8.9e-2 of transformers' (mean
KL 4.7e-5): it does not floor the time step either. The remaining 2.7e-2 was not traced.

`tools/nemotron_asr_tiny.py` (the whole speech model: the real feature extractor, both encoders
with two layers each and chunks of 20 positions, so that 9.3 s of synthetic audio take 6 chunks
with context, fill the FIFO and compress the speaker cache; the projection; a 7-block language
model), compared by `examples/nemotron_asr_check.rs`:

| | max difference (largest value) |
|---|---|
| log-mel features, 936 frames | 5.3e-4 (16.6); mean 1.7e-6 |
| speech encoder states | 5.3e-5 (4.4) |
| speaker logits | 2.3e-4 (114) |
| soft tokens | 6.8e-5 (9.0) |
| logits over the 134 prompt positions (117 audio), with transformers' floor | 2.1e-5, mean KL 5.1e-12, 134/134 top-1 |
| 12 greedy steps after it (cached, no floor) | 6.0e-6, 12/12 |
| the same without any floor (the default) | 5.4e-3 and 1.4e-2, 134/134 and 12/12 |

### The real weights

nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B (67.5 GB) against transformers on the CPU
(`tools/ref_nemotron_asr.py`: the audio tower in float32, the language model in bf16, activations
included, so its logits are bf16 numbers: steps of 0.5 at the size of the top logits), compared
by `examples/nemotron_asr_check.rs` with oxinfer's weights in bf16 and its arithmetic in float32.
Two recordings of the model card: `en-Alice_woman.wav` (9.3 s, one speaker, 116 audio tokens,
one encoder chunk, the default prompt) and `diarization_example.mp3` (97.6 s, seven speakers
active, 1220 audio tokens, three encoder chunks, the speaker cache filled and compressed, the
model card's speaker-tag prompt).

| | 9.3 s | 97.6 s |
|---|---|---|
| log-mel features: max difference (largest value 16.6) | 2.9e-4, mean 1.4e-6 | 2.3e-4, mean 1.0e-6 |
| speech encoder states (largest value 19) | 1.9e-5 | 4.0e-4 |
| speaker logits (largest value 39) | 5.0e-5 | 2.3e-4 |
| soft tokens (largest value 3.7) | 2.3e-6 | 1.7e-4 |
| with transformers' floor: logits of the last 4 prompt positions, mean KL | 8.2e-3 (4/4 top-1) | 2.5e-4 (4/4) |
| the reference's greedy tokens fed one by one: mean KL, same next token | 9.8e-7, 34/34 | 9.3e-3, 510/518 |
| without the floor (the default): prompt positions, mean KL | 0.28 (4/4) | 2.9e-2 (4/4) |
| the reference's tokens fed one by one: mean KL, same next token | 4.1e-5, 34/34 | 3.8e-2, 502/518 |

The audio side is the reference's to float32 rounding. The language model follows transformers
when given its floor: the 8 of 518 next tokens that differ on the long recording are all places
where the reference's two best logits are within 2.0 of each other (three of them exact bf16
ties). Without the floor 16 differ, with margins up to 3.5: the floor is a real difference
between transformers and NVIDIA's stack, not noise.

Transcripts, decoding freely on the CPU:

- The 9.3 s recording gives the reference's text exactly, in bf16, `--quant q8` and `--quant
  q4`, on the command line and through the server (json, text, verbose_json, srt):
  "So just to clarify, we've had nineteen to twenty year olds on our podcast so far, and if you
  don't mind me asking, how old are you?" With `prompt=Please translate the given speech to
  French.` the server answers "Juste pour préciser, nous avons déjà eu des jeunes de dix-neuf à
  vingt ans sur notre podcast. ..."; `/v1/audio/translations` gives the English text again.
- The 97.6 s recording (bf16, the default, so without transformers' floor) gives a
  speaker-tagged transcript of 513 tokens against the reference's 518. It differs from
  transformers' in capitalisation ("Nematron Three", "Voice"), in the spelling of two names
  nobody could know ("QTie" for "QT Py", "Raspie" for "Raspbi"), in one word ("knows" for
  "names") and in the passages where several voices talk over each other, which the two
  transcribe differently: 344 words against 350, 90% of them the same in sequence. This
  transcript was not compared with NeMo or vLLM, which could not be run here (they need a GPU
  that holds the model), so which of the two is closer to NVIDIA's own output is not
  established.

Speed on 64 x86 cores (CPU; the language model is not run on the 16 GB GPU): the checkpoint loads
in 53 to 69 s (the experts are transposed, and quantized with `--quant`); the 9.3 s recording
takes about 0.4 s in the encoders and 4.3 s (q8) or 2.8 s (q4) in the language model, 1.6 to 3
times real time; the 97.6 s one 6.6 s and 78 s in bf16 (1280 prompt tokens, then 513 tokens at about 8
per second), 1.2 times real time.

The Transcribe checkpoint also answers as a plain chat model (`--prompt "What is the capital of
France? One word."` gives "Paris"), on the same code as Lightning.

ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF Q4_0 (18.9 GB; Q4_0 experts, Q8_0 for the
other matrices, float32 norms and routers) against llama.cpp (master, CPU) on a 27-token chat
prompt, with `tools/llama_logits_dump.cpp` feeding llama.cpp's own greedy tokens to both:

| | max logit diff | mean KL | same top-1 |
|---|---|---|---|
| the prompt's last position | 0.41 | 2.1e-4 | 1/1 |
| 32 greedy steps | 1.24 | 1.9e-3 | 31/32 |

Both answer "The sky appears blue because sunlight ..." and continue with different, equivalent
wordings from the step that differs on. On the command line the same file answers in two
sentences about Rayleigh scattering, reasons with `--think` (91 = 7 x 13, not prime), and with
`--tools` calls the calculator (`17 * 23 + 5` -> 396) and states the result: 15 tokens per second
on 64 x86 cores, where llama.cpp decodes 17. The loader reports 2.73 GB read per decoded token.
The bf16 safetensors release of Lightning (65.8 GB) was not downloaded: its tensor layout is the
one `save_pretrained` writes, which the tiny checkpoints cover, and the same architecture runs
from the Transcribe checkpoint's weights.
