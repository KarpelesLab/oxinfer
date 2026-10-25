// Fetches model files into the module: from the ATOnline mirror or Hugging Face by repo id, or
// any URL (kept in the browser's Cache Storage so a reload does not download gigabytes again), or
// from a folder the user picks.
import { isHostShard } from './oxinfer.js';

const CACHE = 'oxinfer-models-v1';
const ROOT = '/model';
/// Lists a repo's files with download URLs (presigned, so they change on every listing).
export const MIRROR_API = 'https://ws.atonline.com/_special/rest/AI/Model:get';
/// Host shards are at most this big (one JavaScript allocation each, and offsets that fit the
/// module's 32-bit address space).
const SHARD = 1 << 30;

/// "org/name" -> its Hugging Face file URL base; URLs and paths ("./x", "/x") are used as they are.
export function sourceBase(spec) {
  const s = spec.trim().replace(/\/+$/, '');
  if (/^[\w.-]+\/[\w.-]+$/.test(s)) return `https://huggingface.co/${s}/resolve/main`;
  return s;
}

async function fetchJson(base, name) {
  const r = await fetch(`${base}/${name}`);
  if (!r.ok) return null;
  try {
    return await r.json();
  } catch {
    return null;
  }
}

const SMALL = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'];

/// The mirror's listing of `repo` as [{ name, url, size, key }], or null when it does not carry it.
async function mirrorFiles(repo) {
  let r;
  try {
    r = await fetch(`${MIRROR_API}?key=${encodeURIComponent(repo)}`);
  } catch {
    return null;
  }
  if (!r.ok) return null;
  const data = (await r.json())?.data;
  if (!data?.Files) return null;
  const all = new Map(data.Files.map((f) => [f.Filename, f]));
  const names = SMALL.filter((n) => all.has(n));
  if (all.has('model.safetensors')) names.push('model.safetensors');
  else if (all.has('model.safetensors.index.json')) {
    const index = await (await fetch(all.get('model.safetensors.index.json').Url)).json();
    names.push('model.safetensors.index.json', ...new Set(Object.values(index.weight_map)));
  } else return null;
  // Stable cache keys: the download URLs are signed per listing.
  const rev = data.HF_Hash || data.Version || 'main';
  return names.map((name) => {
    const f = all.get(name);
    return { name, url: f.Url, size: Number(f.Size) || 0, key: `https://cache.oxinfer.invalid/${repo}@${rev}/${name}` };
  });
}

/// The files a model folder needs, discovered from its config and index.
export async function listFiles(base) {
  if (!(await fetchJson(base, 'config.json'))) {
    throw new Error(`No config.json at ${base}. Check the repo id or URL.`);
  }
  const files = ['config.json', 'tokenizer.json'];
  if (await fetchJson(base, 'generation_config.json')) files.push('generation_config.json');
  // The chat template decides prompt details (Qwen's reasoning-effort line, for one).
  if (await fetchJson(base, 'tokenizer_config.json')) files.push('tokenizer_config.json');
  if ((await fetch(`${base}/chat_template.jinja`, { method: 'HEAD' }).catch(() => null))?.ok) files.push('chat_template.jinja');
  const index = await fetchJson(base, 'model.safetensors.index.json');
  if (index && index.weight_map) {
    files.push('model.safetensors.index.json', ...new Set(Object.values(index.weight_map)));
  } else {
    files.push('model.safetensors');
  }
  return files;
}

async function openCache() {
  try {
    return typeof caches !== 'undefined' ? await caches.open(CACHE) : null;
  } catch {
    return null;
  }
}

/// A Response for `url`, from the cache when present; otherwise downloaded into the cache first
/// (written straight to disk, so a multi-GB file never sits in memory twice).
async function cachedResponse(url, onBytes, key = url) {
  const cache = await openCache();
  if (cache) {
    const hit = await cache.match(key);
    if (hit) return { response: hit, cached: true };
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  if (!cache || !r.body) return { response: r, cached: false };
  let got = 0;
  const size = Number(r.headers.get('content-length') || 0);
  const counted = r.body.pipeThrough(
    new TransformStream({
      transform(chunk, ctl) {
        got += chunk.byteLength;
        onBytes(got, size);
        ctl.enqueue(chunk);
      },
    }),
  );
  try {
    await cache.put(key, new Response(counted, { headers: r.headers }));
    const stored = await cache.match(key);
    if (stored) return { response: stored, cached: false };
  } catch {
    // Quota exceeded or private mode: fall back to a plain download.
  }
  const again = await fetch(url);
  if (!again.ok) throw new Error(`${url}: HTTP ${again.status}`);
  return { response: again, cached: false };
}

async function readInto(g, name, size, stream, onBytes, host = isHostShard(name)) {
  const path = `${ROOT}/${name}`;
  const reader = stream.getReader();
  if (host) {
    const bytes = new Uint8Array(size);
    let off = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes.set(value, off);
      off += value.byteLength;
      onBytes(off);
    }
    g.putHostFile(path, bytes);
    return;
  }
  const w = g.putFileStream(path, size);
  let off = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    w.write(value);
    off += value.byteLength;
    onBytes(off);
  }
  if (off !== size) throw new Error(`${name}: got ${off} of ${size} bytes`);
  w.done();
}

/// Reads one MLX safetensors file (any size) as host shards of at most SHARD bytes, split at tensor
/// boundaries, each with its own header (offsets rebased, data 16-byte aligned). Returns the
/// tensor -> shard map for the index that replaces the original file names.
async function readMlxSharded(g, name, stream, onBytes) {
  const reader = stream.getReader();
  let pending = new Uint8Array(0);
  let got = 0;
  const next = async () => {
    const { done, value } = await reader.read();
    if (done) return false;
    got += value.byteLength;
    onBytes(got);
    const p = new Uint8Array(pending.byteLength + value.byteLength);
    p.set(pending);
    p.set(value, pending.byteLength);
    pending = p;
    return true;
  };
  while (pending.byteLength < 8) if (!(await next())) throw new Error(`${name}: truncated`);
  const n = Number(new DataView(pending.buffer).getBigUint64(0, true));
  while (pending.byteLength < 8 + n) if (!(await next())) throw new Error(`${name}: truncated header`);
  const header = JSON.parse(new TextDecoder().decode(pending.subarray(8, 8 + n)));
  const meta = header.__metadata__;
  const tensors = Object.entries(header)
    .filter(([k]) => k !== '__metadata__')
    .sort((a, b) => a[1].data_offsets[0] - b[1].data_offsets[0]);
  // Plan: consecutive tensors per shard, [start, end) in the original data section.
  const shards = [];
  for (const [k, t] of tensors) {
    const [a, b] = t.data_offsets;
    const cur = shards[shards.length - 1];
    if (cur && b - cur.start <= SHARD) {
      cur.names.push(k);
      cur.end = b;
    } else shards.push({ start: a, end: b, names: [k] });
  }
  const enc = new TextEncoder();
  const stem = name.replace(/\.safetensors$/, '');
  for (const [i, sh] of shards.entries()) {
    const h = {};
    if (meta) h.__metadata__ = meta;
    for (const k of sh.names) {
      const t = header[k];
      h[k] = { dtype: t.dtype, shape: t.shape, data_offsets: [t.data_offsets[0] - sh.start, t.data_offsets[1] - sh.start] };
    }
    let json = JSON.stringify(h);
    json += ' '.repeat((16 - ((8 + enc.encode(json).byteLength) % 16)) % 16);
    const hb = enc.encode(json);
    sh.file = `${stem}-oxinfer-${String(i + 1).padStart(3, '0')}.safetensors`;
    sh.base = 8 + hb.byteLength;
    sh.bytes = new Uint8Array(sh.base + (sh.end - sh.start));
    new DataView(sh.bytes.buffer).setBigUint64(0, BigInt(hb.byteLength), true);
    sh.bytes.set(hb, 8);
  }
  // Route the data section into the shards as it streams in.
  let pos = 0; // offset in the original data section of pending[8 + n]
  let si = 0;
  const route = (chunk) => {
    let off = 0;
    while (off < chunk.byteLength && si < shards.length) {
      const sh = shards[si];
      if (pos + off >= sh.end) {
        si++;
        continue;
      }
      const from = Math.max(pos + off, sh.start);
      const len = Math.min(chunk.byteLength - (from - pos), sh.end - from);
      if (len <= 0) {
        off = chunk.byteLength;
        break;
      }
      sh.bytes.set(chunk.subarray(from - pos, from - pos + len), sh.base + (from - sh.start));
      off = from - pos + len;
    }
    pos += chunk.byteLength;
  };
  route(pending.subarray(8 + n));
  pending = null;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    got += value.byteLength;
    onBytes(got);
    route(value);
  }
  const weightMap = {};
  for (const sh of shards) {
    if (pos < sh.end) throw new Error(`${name}: got ${pos} of ${sh.end} data bytes`);
    g.putHostFile(`${ROOT}/${sh.file}`, sh.bytes);
    for (const k of sh.names) weightMap[k] = sh.file;
  }
  return weightMap;
}

/// Downloads (or reuses) every file of a model into the module: `spec` is a repo id (the mirror
/// when it carries it, else Hugging Face), a URL or a path ("./x", "/x").
/// `progress({ file, index, count, got, size, phase })`, phase: 'download' | 'read'.
export async function loadModel(g, spec, progress) {
  const s = spec.trim().replace(/\/+$/, '');
  let files = /^[\w.-]+\/[\w.-]+$/.test(s) ? await mirrorFiles(s) : null;
  if (!files) {
    const base = sourceBase(s);
    files = (await listFiles(base)).map((name) => ({ name, url: `${base}/${name}`, size: 0, key: `${base}/${name}` }));
  }
  let total = 0;
  let mlx = false;
  const weightMap = {};
  for (const [index, f] of files.entries()) {
    const { name } = f;
    const report = (phase, known) => (got, size = known) => progress({ file: name, index, count: files.length, got, size, phase });
    if (mlx && name === 'model.safetensors.index.json') continue; // replaced below
    let { response } = await cachedResponse(f.url, report('download', f.size), f.key);
    const size = Number(response.headers.get('content-length') || 0) || f.size;
    if (name === 'config.json') {
      const bytes = new Uint8Array(await response.clone().arrayBuffer());
      try {
        // MLX checkpoints say how they are quantized; their weights are resharded below.
        mlx = !!JSON.parse(new TextDecoder().decode(bytes)).quantization;
      } catch {
        /* not JSON: the loader reports it */
      }
    }
    if (mlx && name.endsWith('.safetensors')) {
      Object.assign(weightMap, await readMlxSharded(g, name, response.body, report('read', size)));
      total += size;
      continue;
    }
    if (!size || !response.body) {
      const bytes = new Uint8Array(await response.arrayBuffer());
      response = new Response(bytes);
      await readInto(g, name, bytes.byteLength, response.body, report('read', bytes.byteLength));
      total += bytes.byteLength;
      continue;
    }
    await readInto(g, name, size, response.body, report('read', size));
    total += size;
  }
  if (mlx) g.putFile(`${ROOT}/model.safetensors.index.json`, new TextEncoder().encode(JSON.stringify({ metadata: {}, weight_map: weightMap })));
  return total;
}

/// Downloads (or reuses) every file of the model at URL `base` into the module.
export async function loadFromUrl(g, base, progress) {
  return loadModel(g, base, progress);
}

const WANTED = /^(config\.json|generation_config\.json|tokenizer\.json|tokenizer_config\.json|chat_template\.jinja|model\.safetensors\.index\.json|.*\.safetensors)$/;

/// Reads a folder picked with <input webkitdirectory> into the module.
export async function loadFromFolder(g, fileList, progress) {
  const files = [...fileList].filter((f) => WANTED.test(f.name));
  const config = files.find((f) => f.name === 'config.json');
  if (!config) throw new Error('That folder has no config.json, so it is not a model folder.');
  let mlx = false;
  try {
    mlx = !!JSON.parse(await config.text()).quantization;
  } catch {
    /* the loader reports a bad config */
  }
  let total = 0;
  const weightMap = {};
  for (const [index, f] of files.entries()) {
    const report = (got) => progress({ file: f.name, index, count: files.length, got, size: f.size, phase: 'read' });
    if (mlx && f.name === 'model.safetensors.index.json') continue;
    if (mlx && f.name.endsWith('.safetensors')) Object.assign(weightMap, await readMlxSharded(g, f.name, f.stream(), report));
    else await readInto(g, f.name, f.size, f.stream(), report);
    total += f.size;
  }
  if (mlx) g.putFile(`${ROOT}/model.safetensors.index.json`, new TextEncoder().encode(JSON.stringify({ metadata: {}, weight_map: weightMap })));
  return total;
}

/// Bytes this site keeps in Cache Storage, and a way to drop them.
export async function cachedBytes() {
  try {
    const est = await navigator.storage?.estimate?.();
    return est?.usageDetails?.caches ?? est?.usage ?? 0;
  } catch {
    return 0;
  }
}

export async function clearCache() {
  if (typeof caches !== 'undefined') await caches.delete(CACHE);
}

// ---- diffusers pipelines (image generation) ------------------------------------------------------

const DIFFUSERS_SMALL = [
  'model_index.json',
  'scheduler/scheduler_config.json',
  'text_encoder/config.json',
  'tokenizer/vocab.json',
  'tokenizer/merges.txt',
  'tokenizer/tokenizer_config.json',
  'unet/config.json',
  'vae/config.json',
];
const DIFFUSERS_WEIGHTS = [
  ['text_encoder/model.fp16.safetensors', 'text_encoder/model.safetensors'],
  ['unet/diffusion_pytorch_model.fp16.safetensors', 'unet/diffusion_pytorch_model.safetensors'],
  ['vae/diffusion_pytorch_model.fp16.safetensors', 'vae/diffusion_pytorch_model.safetensors'],
];

/// True when `base` serves a diffusers pipeline (it has model_index.json).
export async function isDiffusers(base) {
  return !!(await fetchJson(base, 'model_index.json'));
}

/// Downloads (or reuses) a Stable Diffusion pipeline. Weights stay in JavaScript memory (the
/// module converts them tensor by tensor), configs and the tokenizer go into module memory.
export async function loadDiffusersFromUrl(g, base, progress) {
  const count = DIFFUSERS_SMALL.length + DIFFUSERS_WEIGHTS.length;
  let index = 0;
  for (const name of DIFFUSERS_SMALL) {
    const { response } = await cachedResponse(`${base}/${name}`, () => {});
    const bytes = new Uint8Array(await response.arrayBuffer());
    g.putFile(`${ROOT}/${name}`, bytes);
    progress({ file: name, index: index++, count, got: bytes.byteLength, size: bytes.byteLength, phase: 'read' });
  }
  for (const choices of DIFFUSERS_WEIGHTS) {
    let done = false;
    for (const name of choices) {
      let response;
      try {
        ({ response } = await cachedResponse(`${base}/${name}`, (got, size) => progress({ file: name, index, count, got, size, phase: 'download' })));
      } catch {
        continue; // try the other variant
      }
      const size = Number(response.headers.get('content-length') || 0);
      const report = (got) => progress({ file: name, index, count, got, size, phase: 'read' });
      if (size && response.body) await readInto(g, name, size, response.body, report, true);
      else {
        const bytes = new Uint8Array(await response.arrayBuffer());
        g.putHostFile(`${ROOT}/${name}`, bytes);
      }
      done = true;
      break;
    }
    if (!done) throw new Error(`${base}: none of ${choices.join(', ')} could be downloaded`);
    index++;
  }
}

/// "org/name/file.safetensors" (on Hugging Face) or a URL, with an optional ":SCALE" suffix ->
/// { url, scale, name }.
export function loraSource(spec) {
  let s = spec.trim();
  let scale = 1;
  const m = s.match(/^(.*\.safetensors):(-?\d*\.?\d+)$/);
  if (m) {
    s = m[1];
    scale = Number(m[2]);
  }
  if (!/\.safetensors$/.test(s)) throw new Error(`${spec}: a LoRA is a .safetensors file (org/name/file.safetensors or a URL)`);
  const hf = s.match(/^([\w.-]+\/[\w.-]+)\/(.+)$/);
  const url = /^(https?:|\.?\/)/.test(s) || !hf ? s : `https://huggingface.co/${hf[1]}/resolve/main/${hf[2]}`;
  return { url, scale, name: s.split('/').pop() };
}

/// Downloads (or reuses) LoRA files into the module's filesystem as host files; resolves to the
/// [{ path, scale }] list `sdLoad` takes.
export async function loadLoras(g, specs, progress) {
  const out = [];
  for (const [index, spec] of specs.entries()) {
    const { url, scale, name } = loraSource(spec);
    const file = `lora/${index}-${name}`;
    const { response } = await cachedResponse(url, (got, size) => progress({ file: name, index, count: specs.length, got, size, phase: 'download' }));
    const bytes = new Uint8Array(await response.arrayBuffer());
    g.putHostFile(`${ROOT}/${file}`, bytes);
    progress({ file: name, index, count: specs.length, got: bytes.byteLength, size: bytes.byteLength, phase: 'read' });
    out.push({ path: `${ROOT}/${file}`, scale });
  }
  return out;
}

/// Reads a diffusers folder picked with <input webkitdirectory>, keeping its subfolders.
export async function loadDiffusersFromFolder(g, fileList, progress) {
  const files = [...fileList];
  const rel = (f) => (f.webkitRelativePath || f.name).split('/').slice(1).join('/');
  if (!files.some((f) => rel(f) === 'model_index.json')) throw new Error('That folder has no model_index.json, so it is not a diffusers pipeline.');
  const wanted = files.filter((f) => /^(model_index\.json|scheduler\/.*\.json|tokenizer\/.*|(text_encoder|unet|vae)\/.*\.(json|safetensors))$/.test(rel(f)));
  for (const [index, f] of wanted.entries()) {
    const name = rel(f);
    if (name.endsWith('.safetensors')) {
      await readInto(g, name, f.size, f.stream(), (got) => progress({ file: name, index, count: wanted.length, got, size: f.size, phase: 'read' }), true);
    } else {
      g.putFile(`${ROOT}/${name}`, new Uint8Array(await f.arrayBuffer()));
    }
  }
}
