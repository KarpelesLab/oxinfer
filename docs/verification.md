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
