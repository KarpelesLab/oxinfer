"""Adds random Qwen4-Exp MTP tensors (the real checkpoint's names and shapes, scaled down) to a
tiny model folder: add_mtp.py DIR (DIR/model/model.safetensors)."""
import sys, torch
from safetensors.torch import load_file, save_file

d = sys.argv[1] + '/model/model.safetensors'
t = load_file(d)
g = torch.Generator().manual_seed(5)
src = 'model.layers.3.'  # an attention layer to copy shapes from
new = {}
for k, v in t.items():
    if k.startswith(src):
        rest = k[len(src):]
        nv = torch.randn(v.shape, generator=g) * (v.float().std() + 1e-3) if v.is_floating_point() else v.clone()
        new['mtp.layers.0.' + rest] = nv.to(v.dtype)
h = t['model.embed_tokens.weight'].shape[1]
w = t['model.hyper_connection_mixer.hc_norm.weight'].shape[0]
new['mtp.fc_embedding.weight'] = torch.randn(h, h, generator=g) * 0.08
new['mtp.fc_hidden.weight'] = torch.randn(h, h, generator=g) * 0.08
new['mtp.pre_fc_norm_embedding.weight'] = torch.randn(h, generator=g) * 0.2
new['mtp.pre_fc_norm_hidden.weight'] = torch.randn(w, generator=g) * 0.2
for n in ['hc_norm.weight', 'input_mix_weight_down.weight', 'input_mix_weight_up.weight']:
    v = t['model.hyper_connection_mixer.' + n]
    new['mtp.hyper_connection_mixer.' + n] = torch.randn(v.shape, generator=g) * v.float().std()
for k in list(new):
    new[k] = new[k].to(torch.bfloat16).float() if new[k].is_floating_point() else new[k]
t.update(new)
save_file(t, d, metadata={'format': 'pt'})
print(len(new), 'MTP tensors added')
