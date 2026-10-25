<script setup>
// Text-to-image with Stable Diffusion (SD 1.5 with the LCM-LoRA by default) in the same module as
// the chat.
import { computed, nextTick, onMounted, reactive, ref } from 'vue';
import { Oxinfer, WebGpuHost, hasJspi } from '../lib/oxinfer.js';
import { cachedBytes, loadDiffusersFromFolder, loadDiffusersFromUrl, loadLoras, sourceBase } from '../lib/loader.js';

const props = defineProps({ device: { type: Object, required: true }, auto: { type: Boolean, default: false }, log: { type: Function, required: true } });
const emit = defineEmits(['storage']);

const WASM_URL = `${import.meta.env.BASE_URL}oxinfer.wasm`;
// Presets carry their own sampling defaults: the LCM-LoRA makes SD 1.5 a 4-step model without
// guidance (with Euler ancestral: plain Euler leaves speckle noise, since LCM is trained to be
// re-noised between steps), SD-Turbo is a 1-step one.
const PRESETS = {
  sd15: {
    repo: 'stable-diffusion-v1-5/stable-diffusion-v1-5',
    name: 'SD 1.5',
    loras: ['latent-consistency/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors'],
    steps: 4,
    guidance: 1,
    ancestral: true,
  },
  turbo: { repo: 'stabilityai/sd-turbo', name: 'SD-Turbo', loras: [], steps: 1, guidance: 0, ancestral: true },
};
const params = new URLSearchParams(location.search);

const source = ref(params.get('model') ? 'custom' : PRESETS[params.get('preset')] ? params.get('preset') : 'sd15');
const custom = ref(params.get('model') || '');
const customLora = ref(params.get('lora') || '');
const folderFiles = ref(null);
const useGpu = ref(params.get('gpu') !== '0');
const phase = ref('idle'); // idle | loading | preparing | ready | generating
const error = ref('');
const progress = reactive({ file: '', index: 0, count: 0, got: 0, size: 0, phase: '' });
const modelName = ref('');
const prompt = ref(params.get('prompt') || 'a red fox in a snowy forest, golden hour, detailed fur');
const negative = ref('');
const steps = ref(4);
const guidance = ref(1);
const ancestral = ref(true);
const size = ref(512);
const seed = ref(Number(params.get('seed') ?? 42));
const randomSeed = ref(params.get('seed') === null);
const step = reactive({ i: 0, n: 0 });
const result = ref(null); // { width, height, url, secs, seed }
const canvas = ref(null);
let engine = null;
let gpuHost = null;

const busy = computed(() => ['loading', 'preparing', 'generating'].includes(phase.value));
const canLoad = computed(() => !busy.value && (PRESETS[source.value] || (source.value === 'custom' && custom.value.trim()) || (source.value === 'folder' && folderFiles.value?.length)));
const progressLine = computed(() => {
  const p = progress;
  if (!p.count) return 'Finding the model files…';
  const verb = p.phase === 'download' ? 'Downloading' : 'Reading';
  const amount = p.size ? `${fmt(p.got)} of ${fmt(p.size)}` : fmt(p.got);
  return `${verb} ${p.file} (file ${p.index + 1} of ${p.count}): ${amount}`;
});
const progressFraction = computed(() => (progress.size ? progress.got / progress.size : 0));

function fmt(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
  return `${(n / 1e3).toFixed(0)} kB`;
}

const pause = () => new Promise((r) => setTimeout(r, 0));

async function load() {
  error.value = '';
  phase.value = 'loading';
  result.value = null;
  Object.assign(progress, { file: '', index: 0, count: 0, got: 0, size: 0, phase: '' });
  try {
    if (gpuHost) gpuHost.device.destroy();
    // Ask for an adapter directly: the page's own probe may still be running.
    gpuHost = useGpu.value && hasJspi ? await WebGpuHost.create(props.log) : null;
    engine = await Oxinfer.create(WASM_URL, { gpu: gpuHost, log: props.log });
    const onProgress = (p) => Object.assign(progress, p);
    const preset = PRESETS[source.value];
    if (source.value === 'folder') {
      modelName.value = folderFiles.value[0]?.webkitRelativePath?.split('/')[0] || 'Local model';
      await loadDiffusersFromFolder(engine, folderFiles.value, onProgress);
    } else {
      const spec = preset ? preset.repo : custom.value;
      modelName.value = preset ? preset.name : spec.split('/').filter(Boolean).pop();
      await loadDiffusersFromUrl(engine, sourceBase(spec), onProgress);
    }
    const loraSpecs = preset ? preset.loras : customLora.value.split(/[\n,]/).map((l) => l.trim()).filter(Boolean);
    const loras = await loadLoras(engine, loraSpecs, onProgress);
    phase.value = 'preparing';
    await nextTick();
    await pause();
    engine.sdLoad('/model', loras);
    if (params.get('check') === '1') engine.gpuCheckEnable();
    const d = preset || engine.sdDefaults();
    steps.value = Number(params.get('steps') || d.steps);
    guidance.value = Number(params.get('guidance') ?? d.guidance);
    ancestral.value = params.has('sampler') ? params.get('sampler') !== 'euler' : (preset?.ancestral ?? true);
    size.value = engine.sdDefaults().size;
    phase.value = 'ready';
  } catch (e) {
    error.value = friendly(e);
    props.log(`[error] ${e.message || e}`);
    phase.value = 'idle';
    engine = null;
  }
  emit('storage', await cachedBytes());
}

function friendly(e) {
  const m = String(e?.message || e);
  if (/unreachable|out of memory|allocation/i.test(m)) return 'The model did not fit in this tab’s memory.';
  if (/Failed to fetch|NetworkError/i.test(m)) return 'The download failed. Check the connection and that the URL allows cross-origin requests.';
  return m;
}

async function generate() {
  if (!engine || phase.value !== 'ready' || !prompt.value.trim()) return;
  error.value = '';
  phase.value = 'generating';
  if (randomSeed.value) seed.value = Math.floor(Math.random() * 2 ** 31);
  Object.assign(step, { i: 0, n: steps.value });
  await nextTick();
  await pause();
  const t0 = performance.now();
  try {
    const img = await engine.sdGenerate(
      { prompt: prompt.value, negative: negative.value, steps: steps.value, guidance: guidance.value, seed: seed.value, width: size.value, height: size.value, ancestral: ancestral.value },
      (i, n) => Object.assign(step, { i, n }),
    );
    const secs = (performance.now() - t0) / 1e3;
    await nextTick();
    draw(img);
    result.value = { width: img.width, height: img.height, secs, seed: seed.value };
    props.log(`[image ${img.width}x${img.height} in ${secs.toFixed(1)}s, seed ${seed.value}]`);
  } catch (e) {
    error.value = friendly(e);
    props.log(`[error] ${e.message || e}`);
  }
  phase.value = 'ready';
}

function draw(img) {
  const c = canvas.value;
  if (!c) return;
  c.width = img.width;
  c.height = img.height;
  const rgba = new Uint8ClampedArray(img.width * img.height * 4);
  for (let i = 0, j = 0; i < img.rgb.length; i += 3, j += 4) {
    rgba[j] = img.rgb[i];
    rgba[j + 1] = img.rgb[i + 1];
    rgba[j + 2] = img.rgb[i + 2];
    rgba[j + 3] = 255;
  }
  c.getContext('2d').putImageData(new ImageData(rgba, img.width, img.height), 0, 0);
}

function save() {
  canvas.value?.toBlob((b) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b);
    a.download = `oxinfer-${result.value?.seed ?? 'image'}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }, 'image/png');
}

function onKey(e) {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    generate();
  }
}

onMounted(async () => {
  if (props.auto) {
    await load();
    if (engine) {
      for (let i = 0; i < Number(params.get('repeat') || 1); i++) await generate();
      if (params.get('dump') === '1' && result.value) props.log(`[png ${canvas.value.toDataURL('image/png')}]`);
      props.log(`[image-done ${result.value ? `${result.value.width}x${result.value.height} ${result.value.secs.toFixed(2)}s` : 'none'}]`);
      if (params.get('check') === '1') props.log(`[gpu check: ${engine.gpuCheckSummary()}]`);
    }
    fetch('/exit', { method: 'POST' }).catch(() => {});
  }
});
</script>

<template>
  <main class="layout">
    <aside class="side">
      <section aria-labelledby="img-model-h">
        <h2 id="img-model-h">Image model</h2>
        <fieldset class="choices" :disabled="busy">
          <legend class="sr-only">Where the model comes from</legend>
          <label class="choice">
            <input v-model="source" type="radio" value="sd15" />
            <span>
              <strong>SD 1.5</strong>
              <small>2.1 GB from Hugging Face, with the LCM-LoRA so a 512×512 image takes 4 steps. The lighter model.</small>
            </span>
          </label>
          <label class="choice">
            <input v-model="source" type="radio" value="turbo" />
            <span>
              <strong>SD-Turbo</strong>
              <small>2.6 GB from Hugging Face. Makes a 512×512 image in one step.</small>
            </span>
          </label>
          <label class="choice">
            <input v-model="source" type="radio" value="custom" />
            <span>
              <strong>Another Stable Diffusion model</strong>
              <small>A diffusers repo id or folder URL (SD 1.x or 2.x).</small>
            </span>
          </label>
          <template v-if="source === 'custom'">
            <input v-model="custom" class="field" type="text" placeholder="org/name, https://… or ./folder" aria-label="Diffusers repo id or folder URL" spellcheck="false" />
            <input v-model="customLora" class="field" type="text" placeholder="LoRA (optional): org/name/file.safetensors:0.8" aria-label="LoRA files, comma separated, each with an optional :scale" spellcheck="false" />
          </template>
          <label class="choice">
            <input v-model="source" type="radio" value="folder" />
            <span>
              <strong>A folder on this computer</strong>
              <small>A diffusers pipeline folder with model_index.json.</small>
            </span>
          </label>
          <input v-if="source === 'folder'" class="field" type="file" webkitdirectory directory multiple aria-label="Model folder" @change="(e) => (folderFiles = e.target.files)" />
        </fieldset>
        <label v-if="device.gpu && hasJspi" class="inline row"><input v-model="useGpu" type="checkbox" /> Use the GPU</label>
        <p v-else class="hint">Without WebGPU every step runs on one CPU thread: expect minutes per image.</p>
        <button class="primary" type="button" :disabled="!canLoad" @click="load">{{ phase === 'idle' ? 'Load model' : 'Load again' }}</button>
        <div v-if="phase === 'loading'" class="progress" role="status">
          <div class="bar"><div class="fill" :style="{ width: `${(progressFraction * 100).toFixed(1)}%` }"></div></div>
          <p>{{ progressLine }}</p>
        </div>
        <p v-else-if="phase === 'preparing'" class="status" role="status">Converting the weights. The page may not respond for a while.</p>
        <p v-else-if="phase === 'ready' || phase === 'generating'" class="status ready" role="status">{{ modelName }} is ready.</p>
        <p v-if="error" class="error" role="alert">{{ error }}</p>
      </section>

      <section v-if="phase === 'ready' || phase === 'generating'" aria-labelledby="img-set-h">
        <h2 id="img-set-h">Settings</h2>
        <label class="inline">Steps <input v-model.number="steps" class="num" type="number" min="1" max="100" /></label>
        <label class="inline">Guidance <input v-model.number="guidance" class="num" type="number" min="0" max="20" step="0.5" /></label>
        <label class="inline">Sampler
          <select v-model="ancestral">
            <option :value="false">Euler</option>
            <option :value="true">Euler ancestral</option>
          </select>
        </label>
        <label class="inline">Size
          <select v-model.number="size">
            <option :value="256">256 × 256</option>
            <option :value="384">384 × 384</option>
            <option :value="512">512 × 512</option>
            <option :value="768">768 × 768</option>
          </select>
        </label>
        <label class="inline">Seed <input v-model.number="seed" class="num" type="number" min="0" :disabled="randomSeed" /></label>
        <label class="inline"><input v-model="randomSeed" type="checkbox" /> New seed each time</label>
        <label class="block">Leave out
          <input v-model="negative" class="field" type="text" placeholder="blurry, text" aria-label="Negative prompt" />
        </label>
        <p class="hint">Leaving things out needs a guidance above 1.</p>
      </section>
    </aside>

    <section class="canvas-pane" aria-label="Image">
      <div class="stage">
        <canvas v-show="result" ref="canvas" class="picture" :aria-label="result ? `Generated image: ${prompt}` : ''"></canvas>
        <p v-if="!result && phase !== 'generating'" class="empty">
          {{ phase === 'ready' ? 'Describe an image, then press Generate.' : 'Load an image model, then describe a picture.' }}
        </p>
        <p v-if="phase === 'generating'" class="working" role="status">
          {{ step.i < step.n ? `Denoising, step ${step.i + 1} of ${step.n}…` : 'Decoding the image…' }}
        </p>
        <p v-if="result && phase !== 'generating'" class="meta">
          {{ result.width }}×{{ result.height }} in {{ result.secs.toFixed(1) }} s, seed {{ result.seed }}.
          <button class="link" type="button" @click="save">Save as PNG</button>
        </p>
      </div>
      <form class="composer" @submit.prevent="generate">
        <textarea v-model="prompt" rows="2" :placeholder="phase === 'idle' ? 'Load a model first' : 'Describe the image'" aria-label="Image description" :disabled="!engine" @keydown="onKey"></textarea>
        <div class="actions">
          <button type="submit" class="primary" :disabled="phase !== 'ready' || !prompt.trim()">Generate</button>
        </div>
      </form>
    </section>
  </main>
</template>

<style scoped>
.layout {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(17rem, 23rem) 1fr;
}
.side {
  overflow-y: auto;
  padding: 1.25rem clamp(1rem, 3vw, 2rem) 2rem;
  border-right: 1px solid var(--line);
}
.side section + section {
  margin-top: 1.5rem;
}
h2 {
  margin: 0 0 0.6rem;
  font-family: var(--display);
  font-size: 1.15rem;
  font-weight: 650;
}
.choices {
  border: 0;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.5rem;
}
.choice {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.6rem;
  cursor: pointer;
}
.choice input {
  margin-top: 0.35rem;
  accent-color: var(--verdigris);
}
.choice small {
  display: block;
  color: var(--muted);
  font-size: 0.85rem;
  line-height: 1.4;
}
.field {
  width: 100%;
  padding: 0.45rem 0.6rem;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--surface);
}
.row {
  margin-top: 1rem;
}
.inline {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  margin: 0.35rem 0;
}
.block {
  display: grid;
  gap: 0.3rem;
  margin: 0.6rem 0 0;
}
.inline select,
.num {
  padding: 0.25rem 0.4rem;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--surface);
}
.num {
  width: 6rem;
}
.hint {
  margin: 0.4rem 0 0;
  color: var(--muted);
  font-size: 0.85rem;
}
button {
  cursor: pointer;
}
button:disabled {
  cursor: default;
  opacity: 0.45;
}
.primary {
  margin-top: 1rem;
  padding: 0.55rem 1.2rem;
  border: 0;
  border-radius: 4px;
  background: var(--verdigris-deep);
  color: var(--paper);
  font-weight: 650;
}
.link {
  padding: 0;
  border: 0;
  background: none;
  color: var(--verdigris-deep);
  text-decoration: underline;
  text-underline-offset: 0.15em;
}
.progress p,
.status,
.error {
  margin: 0.6rem 0 0;
  font-size: 0.9rem;
}
.bar {
  height: 6px;
  margin-top: 1rem;
  background: var(--line);
  border-radius: 3px;
  overflow: hidden;
}
.fill {
  height: 100%;
  background: var(--verdigris);
  transition: width 0.2s linear;
}
.ready {
  color: var(--verdigris-deep);
}
.error {
  color: var(--error);
}
.canvas-pane {
  display: grid;
  grid-template-rows: 1fr auto;
  min-height: 0;
}
.stage {
  overflow: auto;
  padding: 1.5rem clamp(1rem, 4vw, 3rem);
}
.picture {
  display: block;
  max-width: min(100%, 768px);
  height: auto;
  border-radius: 4px;
  box-shadow: 0 0 0 1px var(--line);
}
.empty {
  margin: 3rem 0 0;
  color: var(--muted);
  font-family: var(--display);
  font-size: 1.4rem;
}
.working {
  margin: 1rem 0 0;
  color: var(--copper);
  font-family: var(--display);
  font-size: 1.2rem;
}
.meta {
  margin: 0.75rem 0 0;
  color: var(--muted);
  font-size: 0.9rem;
}
.composer {
  padding: 0.75rem clamp(1rem, 4vw, 3rem) 1rem;
  border-top: 1px solid var(--line);
  background: var(--surface);
}
.composer textarea {
  display: block;
  width: 100%;
  max-width: 68ch;
  min-height: 3.2rem;
  resize: vertical;
  padding: 0.6rem 0.75rem;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--paper);
}
.actions .primary {
  margin-top: 0.6rem;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
}
@media (max-width: 760px) {
  .layout {
    grid-template-columns: 1fr;
  }
  .side {
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
  .canvas-pane {
    min-height: 70dvh;
  }
}
</style>
