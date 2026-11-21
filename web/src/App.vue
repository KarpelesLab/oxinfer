<script setup>
// Two backends, one app. Hosted on its own (GitHub Pages, `npm run dev`) the models run in this
// tab as WebAssembly with WebGPU (lib/oxinfer.js). Served by `oxinfer web`, /api/web/info answers
// and the page drives that machine's engine over HTTP instead (lib/api.js), with views for every
// kind of model it serves and for setting models up.
import { computed, nextTick, onMounted, reactive, ref } from 'vue';
import { Oxinfer, WebGpuHost, hasJspi } from './lib/oxinfer.js';
import { cachedBytes, clearCache, loadFromFolder, loadModel } from './lib/loader.js';
import { WasmChat, HttpChat } from './lib/chat.js';
import { HttpApi, detectBackend } from './lib/api.js';
import { store, setup, refresh, startPolling, modelsOf } from './lib/native.js';
import { systemPrompt } from './lib/tools.js';
import ChatView from './components/ChatView.vue';
import ImageStudio from './components/ImageStudio.vue';
import ModelPicker from './components/ModelPicker.vue';
import ModelsView from './components/ModelsView.vue';
import TranscribeView from './components/TranscribeView.vue';
import SpeechView from './components/SpeechView.vue';
import VoiceView from './components/VoiceView.vue';
import VideoView from './components/VideoView.vue';
import EmbedView from './components/EmbedView.vue';
import ConnectView from './components/ConnectView.vue';

const WASM_URL = `${import.meta.env.BASE_URL}oxinfer.wasm`;
const REPO = 'https://github.com/KarpelesLab/oxinfer';
const PRESET = { repo: 'Qwen/Qwen3.5-0.8B', name: 'Qwen 3.5 0.8B' };
// MLX 4-bit checkpoints the mirror carries (they load as 4-bit whatever the precision setting).
const SUGGESTIONS = ['mlx-community/gemma-4-e2b-it-4bit', 'mlx-community/Qwen3-4B-4bit', 'mlx-community/gemma-3-4b-it-4bit', 'mlx-community/gemma-4-e4b-it-4bit'];

const params = new URLSearchParams(location.search);
const auto = params.get('auto') === '1';
const WASM_MODES = [
  ['chat', 'Chat'],
  ['image', 'Image'],
];
const NATIVE_MODES = [
  ['models', 'Models'],
  ['chat', 'Chat'],
  ['image', 'Image'],
  ['transcribe', 'Transcribe'],
  ['speak', 'Speak'],
  ['voice', 'Voice chat'],
  ['video', 'Video'],
  ['embed', 'Embed'],
  ['connect', 'Connect apps'],
];
const mode = ref(params.get('mode') || 'chat');
// null while the backend is being detected.
const backend = ref(null);
const native = computed(() => backend.value?.native === true);
const modes = computed(() => (native.value ? NATIVE_MODES : WASM_MODES));

function setMode(m) {
  mode.value = m;
  const u = new URL(location.href);
  u.searchParams.set('mode', m);
  history.replaceState(null, '', u);
}

// ---- device -------------------------------------------------------------------------------------
const device = reactive({ checked: false, gpuName: '', gpu: false });
const logs = ref([]);
function log(line) {
  logs.value.push(line);
  if (logs.value.length > 400) logs.value.shift();
  if (auto) {
    console.log(line);
    fetch('/log', { method: 'POST', body: line }).catch(() => {});
  }
}

async function probeDevice() {
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (a) {
        const i = a.info || {};
        device.gpu = true;
        // Browsers often hide the model name; the vendor alone still says which GPU it is.
        const vendor = i.vendor ? i.vendor.charAt(0).toUpperCase() + i.vendor.slice(1) : '';
        device.gpuName = i.description || i.device || (vendor ? `${/^[AEIOU]/.test(vendor) ? 'an' : 'a'} ${vendor} GPU` : 'a GPU');
      }
    } catch {
      /* no adapter */
    }
  }
  device.checked = true;
}

const graphicsLine = computed(() => {
  if (!device.checked) return 'Checking…';
  if (device.gpu && hasJspi) return `WebGPU on ${device.gpuName}. Prompts are processed on the GPU, replies on the CPU.`;
  if (device.gpu) return 'WebGPU is here, but this browser cannot pause WebAssembly while the GPU works (JSPI), so everything runs on the CPU. Chrome 137 or newer has both.';
  return 'No WebGPU in this browser, so everything runs on the CPU.';
});

// ---- wasm model loading -------------------------------------------------------------------------
const source = ref('preset'); // 'preset' | 'custom' | 'folder'
const custom = ref(params.get('model') || '');
const folderFiles = ref(null);
const precision = ref(params.get('quant') || 'q8');
const useGpu = ref(params.get('gpu') !== '0');
const phase = ref('idle'); // idle | loading | preparing | ready
const busy = ref(false);
const error = ref('');
const progress = reactive({ file: '', index: 0, count: 0, got: 0, size: 0, phase: '' });
const modelName = ref('');
const storage = ref(0);
const chatView = ref(null);
const wasmSession = ref(null);
const toolsOn = ref(params.get('tools') !== '0');
const systemText = ref('');

let gpuHost = null;

if (params.get('model')) source.value = 'custom';

const canLoad = computed(() => {
  if (phase.value === 'loading' || phase.value === 'preparing' || busy.value) return false;
  if (source.value === 'custom') return custom.value.trim().length > 0;
  if (source.value === 'folder') return !!folderFiles.value?.length;
  return true;
});

const progressLine = computed(() => {
  const p = progress;
  if (!p.count) return 'Finding the model files…';
  const verb = p.phase === 'download' ? 'Downloading' : 'Reading';
  const amount = p.size ? `${fmtBytes(p.got)} of ${fmtBytes(p.size)}` : fmtBytes(p.got);
  return `${verb} ${p.file} (file ${p.index + 1} of ${p.count}): ${amount}`;
});
const progressFraction = computed(() => (progress.size ? progress.got / progress.size : 0));

function fmtBytes(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} kB`;
  return `${n} B`;
}

function onFolder(e) {
  folderFiles.value = e.target.files;
}

const pause = () => new Promise((r) => setTimeout(r, 0));

function resetSystem() {
  systemText.value = systemPrompt({ tools: toolsOn.value, gpuName: gpuHost ? device.gpuName : '', modelName: modelName.value });
}

function onTools(on) {
  // Takes effect from the next conversation: the declarations live in its first system turn.
  toolsOn.value = on;
  resetSystem();
}

async function load() {
  error.value = '';
  phase.value = 'loading';
  Object.assign(progress, { file: '', index: 0, count: 0, got: 0, size: 0, phase: '' });
  wasmSession.value = null;
  chatView.value?.newConversation();
  try {
    if (gpuHost) gpuHost.device.destroy();
    gpuHost = useGpu.value && device.gpu && hasJspi ? await WebGpuHost.create(log) : null;
    const engine = await Oxinfer.create(WASM_URL, { gpu: gpuHost, log });
    const onProgress = (p) => Object.assign(progress, p);
    if (source.value === 'folder') {
      const first = folderFiles.value[0];
      modelName.value = first?.webkitRelativePath?.split('/')[0] || 'Local model';
      await loadFromFolder(engine, folderFiles.value, onProgress);
    } else {
      const spec = source.value === 'preset' ? PRESET.repo : custom.value;
      modelName.value = source.value === 'preset' ? PRESET.name : spec.split('/').filter(Boolean).pop();
      await loadModel(engine, spec, onProgress);
    }
    phase.value = 'preparing';
    await nextTick();
    await pause();
    engine.load('/model', precision.value);
    const s = new WasmChat(engine);
    s.setTools(toolsOn.value);
    wasmSession.value = s;
    resetSystem();
    phase.value = 'ready';
  } catch (e) {
    error.value = friendly(e);
    log(`[error] ${e.message || e}`);
    phase.value = 'idle';
  }
  storage.value = await cachedBytes();
}

function friendly(e) {
  const m = String(e?.message || e);
  if (/unreachable|out of memory|allocation/i.test(m)) {
    return 'The model did not fit in this tab’s memory. Try a smaller model, or a folder written by oxinfer --save-quantized.';
  }
  if (/Failed to fetch|NetworkError/i.test(m)) return 'The download failed. Check the connection and that the URL allows cross-origin requests.';
  return m;
}

async function forgetDownloads() {
  await clearCache();
  storage.value = await cachedBytes();
}

// ---- native ---------------------------------------------------------------------------------
const keyDraft = ref('');
const chatModel = computed({
  get: () => store.chosen.chat || '',
  set: (v) => (store.chosen.chat = v),
});
const chatModelInfo = computed(() => modelsOf('chat').find((m) => m.id === chatModel.value));
const httpSession = computed(() => (native.value && chatModel.value ? new HttpChat(store.api, chatModel.value) : null));
const nativeSystem = ref('');

function resetNativeSystem() {
  nativeSystem.value = systemPrompt({ tools: toolsOn.value, native: store.status?.platform || 'this computer', modelName: chatModelInfo.value?.id || '' });
}

function saveKey() {
  store.api.setKey(keyDraft.value.trim());
  keyDraft.value = '';
  refresh();
}

function onNativeTools(on) {
  toolsOn.value = on;
  resetNativeSystem();
}

onMounted(async () => {
  backend.value = await detectBackend();
  if (native.value) {
    const host = backend.value.info?.hostname;
    document.title = host ? `oxinfer on ${host}` : 'oxinfer: local models';
    const api = new HttpApi(backend.value.info);
    setup(api, backend.value.info);
    await refresh();
    startPolling();
    if (!params.get('mode')) mode.value = modelsOf('chat').some((m) => m.loaded) ? 'chat' : 'models';
    if (!NATIVE_MODES.some(([m]) => m === mode.value)) mode.value = 'models';
    resetNativeSystem();
    return;
  }
  if (!WASM_MODES.some(([m]) => m === mode.value)) mode.value = 'chat';
  await probeDevice();
  storage.value = await cachedBytes();
  if (auto) log(`[browser: ${navigator.userAgent}; jspi ${hasJspi}; webgpu ${device.gpu ? device.gpuName : 'none'}]`);
  if (auto && mode.value === 'chat') {
    await load();
    for (let i = 0; i < Number(params.get('repeat') || 1) && wasmSession.value; i++) {
      chatView.value.newConversation();
      chatView.value.draft = params.get('prompt') || 'Hello';
      await chatView.value.send();
    }
    fetch('/exit', { method: 'POST' }).catch(() => {});
  }
});
</script>

<template>
  <header class="masthead">
    <h1 class="wordmark">oxinfer</h1>
    <p v-if="native" class="lede">Every model runs on {{ store.status?.platform ? 'this computer' : 'the computer that serves this page' }}. Pick one in Models, then chat, make images and videos, transcribe, speak, talk or search.</p>
    <p v-else class="lede">Chat with a language model or make images, all in this tab. The weights download once and stay in this browser; nothing you type leaves the page.</p>
    <nav class="nav">
      <div class="modes" role="group" aria-label="Mode">
        <button v-for="[m, label] in modes" :key="m" type="button" :aria-pressed="mode === m" @click="setMode(m)">{{ label }}</button>
      </div>
      <a class="repo" :href="REPO">Source on GitHub</a>
    </nav>
  </header>

  <main v-if="!backend" class="layout single" aria-busy="true"><p class="empty pad">Starting…</p></main>

  <!-- ---------------------------------------------------------------- native backend -->
  <template v-else-if="native">
    <main v-if="store.needKey" class="layout single">
      <form class="keyform" @submit.prevent="saveKey">
        <h2>This server needs its API key</h2>
        <p class="hint">It was started with <code>--api-key</code>. The key stays in this browser.</p>
        <label class="block">API key <input v-model="keyDraft" class="field" type="password" autocomplete="current-password" autofocus /></label>
        <button class="primary" type="submit" :disabled="!keyDraft.trim()">Continue</button>
      </form>
    </main>
    <template v-else>
      <p v-if="store.error" class="banner error" role="alert">{{ store.error }}</p>
      <ModelsView v-if="mode === 'models'" @use="setMode" />
      <ChatView
        v-else-if="mode === 'chat'"
        :session="httpSession"
        :model-name="chatModel"
        :ready="!!chatModelInfo"
        :log="log"
        :system="nativeSystem"
        :initial-tools="toolsOn"
        native
        @reset-system="resetNativeSystem"
        @tools="onNativeTools"
      >
        <template #model>
          <ModelPicker v-model="chatModel" kind="chat" title="Chat model" @manage="setMode('models')" />
        </template>
        <template #after>
          <details class="more">
            <summary>Log</summary>
            <pre class="log">{{ logs.join('\n') || 'Nothing yet.' }}</pre>
          </details>
        </template>
      </ChatView>
      <ImageStudio v-else-if="mode === 'image'" :device="device" :log="log" native @manage="setMode('models')" />
      <TranscribeView v-else-if="mode === 'transcribe'" @manage="setMode('models')" />
      <SpeechView v-else-if="mode === 'speak'" @manage="setMode('models')" />
      <VoiceView v-else-if="mode === 'voice'" @manage="setMode('models')" />
      <VideoView v-else-if="mode === 'video'" @manage="setMode('models')" />
      <EmbedView v-else-if="mode === 'embed'" @manage="setMode('models')" />
      <ConnectView v-else-if="mode === 'connect'" @manage="setMode('models')" />
    </template>
  </template>

  <!-- ---------------------------------------------------------------- wasm in this tab -->
  <ImageStudio v-else-if="mode === 'image'" :device="device" :auto="auto" :log="log" @storage="(v) => (storage = v)" />
  <ChatView
    v-else
    ref="chatView"
    :session="wasmSession"
    :model-name="modelName"
    :ready="phase === 'ready'"
    :log="log"
    :auto="auto"
    :system="systemText"
    :initial-draft="params.get('prompt') || ''"
    :initial-temperature="Number(params.get('temp') ?? 0.7)"
    :initial-max-tokens="Number(params.get('max') || 512)"
    :initial-tools="toolsOn"
    @busy="(b) => (busy = b)"
    @reset-system="resetSystem"
    @tools="onTools"
  >
    <template #model>
      <section aria-labelledby="model-h">
        <h2 id="model-h">Model</h2>
        <fieldset class="choices" :disabled="phase === 'loading' || phase === 'preparing' || busy">
          <legend class="sr-only">Where the model comes from</legend>
          <label class="choice">
            <input v-model="source" type="radio" value="preset" />
            <span>
              <strong>Qwen 3.5 0.8B</strong>
              <small>1.8 GB from Hugging Face, made 8-bit in the browser. Works on any recent browser.</small>
            </span>
          </label>
          <label class="choice">
            <input v-model="source" type="radio" value="custom" />
            <span>
              <strong>Another model</strong>
              <small>A repo id, from the ATOnline mirror when it has the model and Hugging Face otherwise (MLX 4-bit repos work too), or a URL to a model folder.</small>
            </span>
          </label>
          <input
            v-if="source === 'custom'"
            v-model="custom"
            class="field"
            type="text"
            placeholder="org/name, https://… or ./folder"
            aria-label="Repo id or folder URL"
            list="model-suggestions"
            spellcheck="false"
          />
          <datalist id="model-suggestions">
            <option v-for="m in SUGGESTIONS" :key="m" :value="m"></option>
          </datalist>
          <label class="choice">
            <input v-model="source" type="radio" value="folder" />
            <span>
              <strong>A folder on this computer</strong>
              <small>A model folder, such as an MLX download or one written by <code>oxinfer --save-quantized</code>.</small>
            </span>
          </label>
          <input v-if="source === 'folder'" class="field" type="file" webkitdirectory directory multiple aria-label="Model folder" @change="onFolder" />
        </fieldset>

        <div class="row">
          <label class="inline">
            Precision
            <select v-model="precision" :disabled="phase === 'loading' || phase === 'preparing' || busy">
              <option value="q8">8-bit</option>
              <option value="q4">4-bit</option>
              <option value="bf16">16-bit</option>
            </select>
          </label>
          <label v-if="device.gpu && hasJspi" class="inline"><input v-model="useGpu" type="checkbox" /> Use the GPU</label>
        </div>
        <p class="hint">Precision applies to original checkpoints. MLX models and folders from <code>--save-quantized</code> keep their own.</p>

        <button class="primary" type="button" :disabled="!canLoad" @click="load">
          {{ phase === 'ready' ? 'Load again' : 'Load model' }}
        </button>

        <div v-if="phase === 'loading'" class="progress" role="status">
          <div class="bar"><div class="fill" :style="{ width: `${(progressFraction * 100).toFixed(1)}%` }"></div></div>
          <p>{{ progressLine }}</p>
        </div>
        <p v-else-if="phase === 'preparing'" class="status" role="status">Preparing the weights. The page may not respond for a few seconds.</p>
        <p v-else-if="phase === 'ready'" class="status ready" role="status">{{ modelName }} is ready.</p>
        <p v-if="error" class="error" role="alert">{{ error }}</p>
      </section>

      <section aria-labelledby="device-h">
        <h2 id="device-h">This device</h2>
        <p class="device">{{ graphicsLine }}</p>
        <p class="device">The browser build uses one CPU thread with SIMD, so replies are slower than the native app.</p>
      </section>
    </template>
    <template #after>
      <details class="more">
        <summary>Log</summary>
        <pre class="log">{{ logs.join('\n') || 'Nothing yet.' }}</pre>
      </details>

      <p v-if="storage > 1e6" class="hint">
        Downloaded models use {{ fmtBytes(storage) }} of this browser’s storage.
        <button class="link" type="button" @click="forgetDownloads">Delete them</button>
      </p>
    </template>
  </ChatView>
</template>

<style scoped>
.masthead {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: baseline;
  gap: 0 1.5rem;
  padding: 1.5rem clamp(1rem, 4vw, 2.5rem) 1rem;
  border-bottom: 1px solid var(--line);
}
.wordmark {
  margin: 0;
  font-family: var(--display);
  font-size: clamp(2.4rem, 6vw, 3.6rem);
  font-weight: 750;
  font-stretch: 78%;
  letter-spacing: -0.035em;
  line-height: 0.9;
  color: var(--verdigris-deep);
}
.lede {
  margin: 0;
  max-width: 52ch;
  color: var(--muted);
}
.nav {
  display: flex;
  align-items: center;
  gap: 1.25rem;
}
.repo {
  white-space: nowrap;
}
.modes {
  display: inline-flex;
  flex-wrap: wrap;
  border: 1px solid var(--line);
  border-radius: 4px;
  overflow: hidden;
}
.modes button {
  padding: 0.35rem 0.9rem;
  border: 0;
  background: transparent;
  font-family: var(--display);
  font-weight: 600;
}
.modes button + button {
  border-left: 1px solid var(--line);
}
.modes button[aria-pressed='true'] {
  background: var(--verdigris-deep);
  color: var(--paper);
}
.single {
  grid-template-columns: 1fr;
}
.pad {
  padding: 0 clamp(1rem, 4vw, 3rem);
}
.keyform {
  max-width: 28rem;
  padding: 2rem clamp(1rem, 4vw, 3rem);
}
.banner {
  margin: 0;
  padding: 0.5rem clamp(1rem, 4vw, 2.5rem);
  border-bottom: 1px solid var(--line);
  background: var(--surface);
}

@media (max-width: 1500px) {
  .masthead {
    grid-template-columns: 1fr auto;
  }
  .lede {
    grid-column: 1 / -1;
    grid-row: 2;
    margin-top: 0.5rem;
  }
  .nav {
    grid-column: 1 / -1;
    grid-row: 3;
    margin-top: 0.75rem;
    flex-wrap: wrap;
  }
}
</style>
