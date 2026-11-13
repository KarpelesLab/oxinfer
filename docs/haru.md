# Speech to speech (Haru-4B)

[Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B) (higurashi, Apache 2.0,
`model_type: haru`) is a full-duplex spoken dialogue model: it listens and speaks at the same
time, one frame of 133 ms (3200 samples at 24 kHz) in and one out, in nine languages. It is a
pretrained base model released for research: it holds a conversation, it is not an assistant.
oxinfer reads the checkpoint as it is and runs it on the CPU:

| File | What | Size |
|---|---|---|
| `model.safetensors` | everything, bf16: `lm.*` the language model (Qwen3.5-4B text, 4.2B), `codec.*` the audio codec (687M), `head.*` the audio head (363M), `model_audio_proj.*` / `user_audio_proj.*` | 10.5 GB |
| `tokenizer.json`, `config.json` | the Qwen tokenizer (248320 ids), the shapes | 20 MB |

There is no live audio in oxinfer, so the command line is the shape of the reference's offline
tool (`haru-inference user.wav reply.wav`): a recording of the user is played to the model frame
by frame, exactly as a live conversation would arrive, and what the model says meanwhile is
written as a WAV of the same length. The text of the model's speech goes to standard output.

```sh
./target/release/oxinfer --model higurashi-ai/Haru-4B --audio user.wav -o reply.wav
# a character card and the voice the model should speak with (3 s to 3 min of one speaker)
./target/release/oxinfer --model higurashi-ai/Haru-4B --audio user.wav -o reply.wav \
    --system "$(cat card.txt)" --ref-audio voice.wav
# the model opens the conversation; another take; a greedy text stream
./target/release/oxinfer --model higurashi-ai/Haru-4B --audio user.wav -o reply.wav \
    --speak-json '{"speak_first": true}' --seed 7 --temp 0
```

`--audio` is the user's side (WAV; MP3, FLAC and the rest through oxideav; resampled to 24 kHz
mono). `--guidance` is the audio guidance scale (1 to 2, default 1.5), `--temp`, `--top-k`,
`--top-p` and `--seed` sample the text stream (defaults 0.7, 25, 1.0), `--quant q8`
quantizes the language model at load, `--show-tokens` prints every frame's token. The reply's
frame `t` is what the model says while the user speaks frame `t`; it was computed from the
user's frames before it.

The card has a fixed layout (the reference's `render_card`; templates for the nine languages are
in its `assets/card_templates.json`). In English:

```text
You are the person described below, talking with the other person by voice.
[Your Persona]
Haru is a 29-year-old nurse in Denver. She is warm, chatty and laughs easily.
[Your Name]
Haru
[Your Gender]
female
[The Other Person's Name]
Sam
[Relationship]
Old friends from high school.
[Situation]
A phone call to catch up after a long time.
```

In Rust, a conversation is a session stepped one frame at a time, which is what a live front
end would drive:

```rust
use oxinfer::haru::{DuplexConfig, DuplexSession, Haru, HaruOptions};
let model = Haru::load("higurashi-ai/Haru-4B", &HaruOptions::default())?;
let voice = model.load_reference("voice.wav".as_ref())?; // 24 kHz mono, silence trimmed
let mut session = DuplexSession::new(&model, &card, Some(&voice), DuplexConfig::default());
let mut frame = session.start()?;          // the model's first 133 ms: frame.audio, frame.text
while !frame.done {
    frame = session.step(&user_audio)?;    // 3200 samples at 24 kHz in, 3200 out
}
// The codec on its own: 24 kHz audio to and from 64 values per 3200 samples.
let latents = model.codec().encode(&pcm);
let pcm = model.codec().decode(&latents);
```

## The pipeline

**One position per frame.** The language model's input at a frame is the sum of three streams:
the embedding of the model's text token, the model's own audio latent and the user's audio
latent, each latent normalized (`latent_scale`, `latent_bias`) and passed through its projector
(a linear layer, an RMS norm, a linear layer). The hidden state at frame `t` gives frame
`t + 1`: the text token through the tied embedding matrix, and the audio latent through the
MeanFlow head, which the codec decodes at once. So the model is always one frame ahead of the
user, and the frame length is the only structural latency.

**Text stream.** The model's words, in the tokens of Qwen's tokenizer. Frames without text hold
a padding token (248076), and an end-of-padding token (248071) sits on the frame before the model
starts to speak. It is sampled with a temperature and top-k, with the DRY penalty (a token that
would extend a sequence the stream already holds costs `0.8 * 1.75^(length - 1)`, at most 32;
the two padding tokens and the end token are exempt).

**Prompt** (`prompt.py`). The card as a ChatML system turn (`<|im_start|>system\n...<|im_end|>\n`,
text only), then the voice reference: its latents on the model's audio stream, under padding
tokens and one end-of-padding token on the last frame. With `speak_first`, two more positions
(padding, end of padding) carry the reference's quietest frame on the model's side and encoded
silence on the user's, and the first frame is forced to be text.

**Language model** (`lm.*`). The Qwen3.5-4B text model, which oxinfer already runs as a chat
model ([Qwen 3.5](qwen.md)): 32 layers of width 2560, three Gated DeltaNet layers for each
attention layer. It is fed embeddings instead of token ids and read at its last hidden state.

**Audio head** (`oxinfer::haru::head`, `meanflow_head.py`). A MeanFlow head: four MLP blocks of
width 2560 (RMS norm, shift, scale and gate from the condition, SwiGLU of 7680) and a final
projection to 64 values. The condition is the hidden state plus embeddings of the time, the
interval and the guidance scale, which is an input of the network, so one pass gives the guided
latent: `z = noise + u(noise)`. No audio transformer and no discrete audio tokens.

**Codec** (`oxinfer::haru::codec`, `vibevoice_codec.py`: VibeVoice's causal acoustic tokenizer).
24 kHz audio to and from one latent of 64 values per 3200 samples. Each direction is a stack of
causal layers, seven stages of blocks (a depthwise convolution over 7 steps, an MLP of four times
the width) from 32 channels at 24 kHz to 2048 channels at 7.5 Hz, joined by strided (or
transposed) convolutions by 2, 2, 4, 5, 5 and 8. Every layer keeps its left context as a state,
so the user's audio is encoded and the model's decoded one frame at a time. The latents fed to
the language model are drawn around the encoder's means (one normal scale per call times
`0.5 / 0.8`, times normal noise per value), as the model was trained.

## Agreement with the reference

The reference is the PyTorch implementation of
[higurashi-ai/Haru](https://github.com/higurashi-ai/Haru) (`haru/`), with transformers 5.23 and
PyTorch 2.15 in float32 on the CPU on the released weights. `tools/ref_haru.py` runs a
conversation with a greedy text stream and records every normal value the session draws;
`examples/haru_check.rs` replays them and compares:

```sh
M=~/.cache/oxinfer/hub/higurashi-ai--Haru-4B
python tools/ref_haru.py Haru $M ref --user user.wav --reference voice.wav --system-prompt "$(cat card.txt)"
cargo run --release --example haru_check -- $M ref
```

Two conversations of 120 frames (16 s) with the README's card, a 6.1 s voice reference and a
recording of the user (5 s of speech, then silence), 147 and 149 prompt positions:

| | The model stays silent (greedy) | `speak_first` (20 text tokens, 4 end-of-padding tokens) |
|---|---|---|
| reference clip after trimming | identical | identical |
| prompt tokens | identical | identical |
| text token of every frame | 120 of 120 | 120 of 120 |
| hidden state, largest difference (its peak) | 3.0e-3 (133) | 3.4e-2 (161) |
| audio latent, largest difference (its peak) | 1.4e-3 (17) | 1.2e-2 (24) |
| the conversation's audio against the reference's | 80.8 dB SNR | 67.9 dB SNR |

Every frame's latent goes back into the language model, so these are differences accumulated
over a closed loop of 120 steps; the text is the reference's in both (`"I don't know I'll shoot
the host  It's been a while  Yeah  I'm  I'm gonna"` in the second).

The codec on its own, on the same inputs as the reference: the encoder's means of the reference
clip (46 latents) within 4.8e-5 of values up to 20 (115.6 dB), those of the user's frames encoded
one at a time within 3.0e-4 (101.6 dB); the decoder on the reference's latents, one frame at a
time, 112.5 and 119.4 dB against the reference's audio, and 16 frames at a time the same samples
to 2e-6.

**End to end.** Four conversations of 30 s with the text stream sampled (seeds 1 to 4, the same
card and voice, `speak_first`): the model's text stream (240 words) against oxinfer's
whisper-small transcript of its audio, in lower case without punctuation, differs in 10.4% of the
words: fillers whisper leaves out ("um"), "gonna" heard as "going to", the last words of a
recording whose audio comes after its end, and a few real differences ("I've always been" heard
as "for me, he's been"). The replies are in the reference clip's voice and react to what the user
said; neither was measured.

## Speed

On a 64-thread x86 machine shared with other jobs (a load average of 35 to 55 while measuring):
a frame takes 202 to 211 ms with the weights in bf16, against 133 ms for real time, and 158 ms
with `--quant q8` (whose output was listened to, not compared with the reference); the prompt of
149 positions takes 2.1 s. On its own the codec decodes 16 s of audio in 3.8 s and encodes 6 s in
0.8 s. The process peaks at 10.7 GB of memory in bf16 (15.3 GB with `--quant q8`, which keeps the
checkpoint mapped beside the quantized copy). The PyTorch reference on the same machine (16
threads, float32) takes about 800 ms per frame.

## Not done

- **Live audio**: no microphone, no playback, no WebRTC or web demo; the command line plays a
  recording to the model. `DuplexSession` is the frame-by-frame interface a live front end
  needs.
- **The server**: no endpoint. The OpenAI speech and transcription endpoints are one-way, and a
  full-duplex stream needs a realtime protocol that oxinfer's server does not have.
- **GPU**: a frame's step runs on the CPU (oxinfer's GPU decoding takes token ids, not
  embeddings). Everything here was run with `--device cpu`; with a GPU backend active the
  prompt's pass may use it, which was not tried.
- **The reference's random numbers**: the text stream and the noise use oxinfer's own generator,
  so a seed does not give the reference's take.
- **bf16 arithmetic**: the reference runs in bf16 on a GPU (and computes the two padding logits
  in float32); oxinfer computes in float32 from the bf16 weights throughout.
- The card templates (the card is given as text), stereo output with the user on the other
  channel, the per-frame transcript file, several conversations in a batch, the Triton and Metal
  kernels, the MLX implementation and the vLLM-Omni plugin of the reference repository.
