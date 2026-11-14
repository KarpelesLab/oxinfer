# Verifying against the reference

```sh
python3 -m venv venv && ./venv/bin/pip install torch transformers safetensors numpy
printf '<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n' > prompt.txt
./target/release/oxinfer --model models/gemma-4-E2B-it --prompt "What is the capital of France?" --dump-logits ours.bin
./venv/bin/python tools/ref_logits.py models/gemma-4-E2B-it prompt.txt ours.bin --generate 12
```

For images and videos (Qwen3.5-family, and Gemma 4 videos; the reference needs `torchvision`),
dump oxinfer's side without a system prompt and let `tools/ref_vl.py` run the processor and model:

```sh
M=~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B
./target/release/oxinfer --model $M --image fox.png -p "Describe this image." --no-system --print-prompt > prompt.txt
./target/release/oxinfer --model $M --image fox.png -p "Describe this image." --device cpu --dump-mm ours --dump-logits ours.bin
./venv/bin/python tools/ref_vl.py $M --image fox.png --prompt "Describe this image." \
    --ours ours --ours-logits ours.bin --ours-prompt prompt.txt --generate 40
```

Unlimited-OCR-2 is compared with the code in its checkpoint (`tools/ref_unlimited_ocr.py`: token
ids, pixel values, encoder features, packed embeddings, logits and the greedy reply; see
[Unlimited-OCR-2](ocr.md#agreement-with-the-reference)):

```sh
M=~/.cache/oxinfer/hub/baidu--Unlimited-OCR-2
./target/release/oxinfer --model $M --image page.png --device cpu --dump-mm ours --dump-logits ours.bin
./target/release/oxinfer --model $M --image page.png --device cpu --show-tokens > ours.txt 2> ours.log
./venv/bin/python tools/ref_unlimited_ocr.py $M --image page.png --ours ours --ours-logits ours.bin \
    --ours-log ours.log --generate 600
```

LFM3-Omni is compared with the code in its checkpoint too (`tools/ref_lfm3_omni.py`: log-mel
features, audio and image tokens, prompt ids, logits, greedy text, and for spoken replies the
speech codes and the codec's waveform; see [LFM3-Omni](lfm3.md#agreement-with-the-reference)).

Videos are compared as directories of frames (`ffmpeg -i clip.mp4 frames/%05d.png`, then
`--video frames --frames-fps 24` on both sides), so both read the same pixels.

Runtime interventions take the same JSON spec on both sides (`tools/ref_intervene.py` applies it
with PyTorch forward hooks; `--generate N --ours-ids IDS.json` also compares greedy text):

```sh
M=~/.cache/oxinfer/hub/Qwen--Qwen3.5-4B
./target/release/oxinfer --model $M --no-system -p "Write a short poem about the sea." --print-prompt > prompt.txt
./target/release/oxinfer --model $M --no-system -p "Write a short poem about the sea." --device cpu \
    --interventions spec.json --dump-logits ours.bin
./venv/bin/python tools/ref_intervene.py $M prompt.txt spec.json ours.bin
```

GGUF files of the Qwen3.5 family are compared with llama.cpp on the same file, greedy token by
greedy token: `tools/llama_logits_dump.cpp` (built against libllama) and
`examples/qwen_greedy_dump.rs` write the output row of the prompt's last position and of every
generated token, and `--compare` gives the number of identical steps and the largest
log-probability difference over the top 10 until the first different token:

```sh
G=model.gguf
cargo run --release --example qwen_greedy_dump -- $G "The history of the Roman Empire" 32 ours.bin --cpu --ids-out ids.txt
./lgdump $G 0 ids.txt 32 llama.bin 16        # MODEL NGL IDS_FILE N_GEN OUT.bin THREADS
cargo run --release --example qwen_greedy_dump -- --compare ours.bin llama.bin 248320
```
