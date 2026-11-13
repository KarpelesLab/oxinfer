"""transformers reference outputs of Nemotron 3.5 Transcribe (`nemo_speechlm`) on one recording,
for `examples/nemotron_asr_check.rs`.

usage: ref_nemotron_asr.py MODEL_DIR AUDIO.wav OUT.safetensors [--prompt TEXT] [--gen N] [--rows N]
                           [--float32]

MODEL_DIR: nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B (the weights in bf16 on the CPU, the
audio tower in float32 as transformers keeps it; 68 GB of memory). AUDIO: 16 kHz mono
(other rates are resampled by transformers' loader, whose resampler is not oxinfer's).

Writes OUT.safetensors:
  pcm [samples] F32             the samples as fed
  features [frames, mels] F32   the log-mel features (padded to whole encoder positions)
  valid [1] I64                 the frames of the recording
  asr_states [n, d]             the speech recognition encoder's output
  speaker_logits [frames, s]    the diarization logits
  audio_embeds [tokens, H]      the soft tokens
  ids [len] I64                 the prompt (the audio token at the audio positions)
  logits [rows, vocab]          the logits of the last `rows` prompt positions
  gen [k] I64, decode_logits [k, vocab]   greedy tokens and the logits after feeding each
and prints the transcript (also written to OUT.txt).
Needs transformers >= 5.24 (the commit that has nemotron3_5_transcribe).
"""
import sys
import torch
from safetensors.torch import save_file


def reference(model, feature_extractor, pcm, ids_of, gen, rows, subsampling=8):
    """The outputs listed above for `pcm` (float32 samples); `ids_of(n)` gives the prompt for n audio tokens."""
    audio_dtype = model.model.audio_tower.dtype
    with torch.no_grad():
        fe = feature_extractor(pcm.numpy(), sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
        feats, mask = fe["input_features"], fe["attention_mask"]
        pad = -feats.shape[1] % subsampling
        feats = torch.nn.functional.pad(feats, (0, 0, 0, pad))
        mask = torch.nn.functional.pad(mask, (0, pad))
        valid = int(mask.sum())
        n_tokens = (valid + subsampling - 1) // subsampling
        audio = model.model.audio_tower(feats.to(audio_dtype), input_features_mask=mask)
        embeds = model.get_audio_features(feats.to(audio_dtype), mask).pooler_output
        ids = ids_of(n_tokens)
        o = model(input_ids=ids, input_features=feats.to(audio_dtype), input_features_mask=mask, use_cache=True, logits_to_keep=rows)
        logits = o.logits[0].float()
        cache, nxt = o.cache, o.logits[0, -1].argmax()
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        toks, dec = [], []
        for _ in range(gen):
            if int(nxt) in eos:
                break
            toks.append(int(nxt))
            o = model(input_ids=nxt.view(1, 1), cache=cache, use_cache=True)
            cache = o.cache
            dec.append(o.logits[0, -1].float())
            nxt = o.logits[0, -1].argmax()
    out = {
        "pcm": pcm.float().contiguous(),
        "features": feats[0].float().contiguous(),
        "valid": torch.tensor([valid]),
        "asr_states": audio.asr_last_hidden_state[0].float().contiguous(),
        "speaker_logits": audio.speaker_logits[0].float().contiguous(),
        "audio_embeds": embeds.float().contiguous(),
        "ids": ids[0].contiguous(),
        "logits": logits.contiguous(),
        "gen": torch.tensor(toks, dtype=torch.long),
    }
    if dec:
        out["decode_logits"] = torch.stack(dec).contiguous()
    return out, toks, int(nxt) in eos


if __name__ == "__main__":
    import time
    from transformers import AutoProcessor, Nemotron3_5TranscribeForConditionalGeneration
    from transformers.audio_utils import load_audio

    mdir, wav, outp = sys.argv[1:4]
    prompt, gen, rows, f32 = None, 256, 4, False
    a = sys.argv[4:]
    for i, x in enumerate(a):
        if x == "--prompt":
            prompt = a[i + 1]
        if x == "--gen":
            gen = int(a[i + 1])
        if x == "--rows":
            rows = int(a[i + 1])
        if x == "--float32":
            f32 = True
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(mdir)
    model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(
        mdir, dtype=torch.float32 if f32 else torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    print(f"loaded in {time.time() - t0:.0f}s, language model {model.model.language_model.dtype}, audio tower {model.model.audio_tower.dtype}", flush=True)
    pcm = torch.tensor(load_audio(wav, sampling_rate=processor.feature_extractor.sampling_rate))

    def ids_of(n):
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt or processor.default_transcription_prompt}, {"type": "audio", "path": wav}]}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        head, tail = text.rsplit(processor.audio_token, 1)
        enc = lambda s: processor.tokenizer(s, add_special_tokens=False)["input_ids"]
        return torch.tensor([enc(head) + [processor.audio_token_id] * n + enc(tail)])

    t0 = time.time()
    out, toks, ended = reference(model, processor.feature_extractor, pcm, ids_of, gen, rows, processor.subsampling_factor)
    # The processor's own prompt must be the one built here.
    inputs = processor.apply_transcription_request(wav, prompt=prompt)
    assert inputs["input_ids"][0].tolist() == out["ids"].tolist(), "prompt ids differ from apply_transcription_request"
    save_file(out, outp)
    text = processor.tokenizer.decode(toks, skip_special_tokens=False)
    open(outp.rsplit(".", 1)[0] + ".txt", "w").write(text)
    print(f"{len(pcm) / 16000:.2f}s of audio, {out['audio_embeds'].shape[0]} audio tokens, {out['ids'].shape[0]} prompt tokens, "
          f"{len(toks)} generated ({'ended' if ended else 'cut'}) in {time.time() - t0:.0f}s")
    print(text)
