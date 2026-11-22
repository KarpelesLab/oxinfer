<script setup>
// Setting models up on the native backend: the tested models of the catalog (what fits this
// machine, what is downloaded), any other model by its spec, downloads with live progress, the
// models in use and the cache on disk.
import { computed, onMounted, onUnmounted, ref } from 'vue';
import { store, refresh, fmtBytes, fmtSecs, fmtAgo, loadLine } from '../lib/native.js';
import SetupDialog from './SetupDialog.vue';

const emit = defineEmits(['use']);

const GROUPS = [
  ['all', 'All'],
  ['downloaded', 'Downloaded'],
  ['chat', 'Chat'],
  ['vision', 'Vision-language'],
  ['image', 'Images'],
  ['stt', 'Speech to text'],
  ['tts', 'Text to speech'],
  ['embedding', 'Embeddings'],
  ['video', 'Video'],
];
const KIND_LABEL = { chat: 'chat', embedding: 'embeddings', image: 'images', transcription: 'speech to text', speech: 'text to speech' };

const catalog = ref(null);
const catalogError = ref('');
const cache = ref(null);
const cacheError = ref('');
const group = ref('all');
const query = ref('');
// The downloaded models, most recently used (then largest) first; the first 8 unless expanded.
const cacheAll = ref(false);
const cacheShown = computed(() => {
  const list = [...(cache.value?.entries || [])].sort((a, b) => (b.loaded - a.loaded) || ((b.last_used || 0) - (a.last_used || 0)) || (b.bytes - a.bytes));
  return cacheAll.value ? list : list.slice(0, 8);
});
const showAll = ref(false);
const manual = ref('');
const setup = ref(null); // { spec, model (catalog entry) }
const confirmDelete = ref(null); // a cache entry
const deleteError = ref('');
const deleting = ref(false);
const dialog = ref(null);
let timer = null;

function groupOf(m) {
  switch (m.kind) {
    case 'chat':
      return m.capabilities.includes('image-input') || m.capabilities.includes('video-input') ? 'vision' : 'chat';
    case 'speech-to-text':
      return 'stt';
    case 'text-to-speech':
      return 'tts';
    default:
      return m.kind;
  }
}

const counts = computed(() => {
  const c = { all: 0 };
  for (const m of catalog.value?.models || []) {
    c.all++;
    if (m.cached) c.downloaded = (c.downloaded || 0) + 1;
    const g = groupOf(m);
    c[g] = (c[g] || 0) + 1;
  }
  return c;
});

const shown = computed(() => {
  const q = query.value.trim().toLowerCase();
  const list = (catalog.value?.models || []).filter((m) => (group.value === 'all' || (group.value === 'downloaded' ? m.cached : groupOf(m) === group.value)) && (!q || m.model.toLowerCase().includes(q) || m.note.toLowerCase().includes(q) || m.capabilities.some((c) => c.includes(q))));
  return showAll.value || q || group.value !== 'all' ? list : list.slice(0, 24);
});

function recommended(m) {
  return m.variants.find((v) => v.recommended) || m.variants[0];
}

// Where the recommended variant runs here (`accel.label`: "GPU (Metal) prefill + decode",
// "GPU (CUDA) prefill, CPU decode", "CPU", ..., with ", streamed from disk").
function accelText(v) {
  return (v.accelerator || '').replace(/, streamed from disk$/, '');
}

function fit(m) {
  const v = recommended(m);
  const speed = v.decode_tok_s ? ` · ~${v.decode_tok_s >= 10 ? v.decode_tok_s.toFixed(0) : v.decode_tok_s.toFixed(1)} tokens/s` : '';
  if (!m.servable) return { cls: 'warn', text: 'Not served here (command line only)', accel: accelText(v) };
  if (m.disk_short_gb) return { cls: 'bad', text: `${m.disk_short_gb.toFixed(1)} GB short on disk`, accel: accelText(v) };
  if (v.runs_on === 'streamed') return { cls: 'warn', text: 'Larger than memory: streams from disk, slowly', accel: accelText(v) };
  if (v.accelerator) return { cls: 'good', text: `Fits · ${accelText(v)}${speed}`, accel: '' };
  const where = v.runs_on === 'gpu' ? 'on the GPU' : v.runs_on === 'cpu' ? 'on the CPU' : 'in memory';
  return { cls: 'good', text: `Fits ${where}${speed}`, accel: '' };
}

const ACCEL_KIND = { cpu: 'CPU', metal: 'GPU · Metal', cuda: 'GPU · CUDA', vulkan: 'GPU · Vulkan', gpu: 'GPU', 'neural-engine': 'Neural Engine' };

function accelState(a) {
  if (a.used && a.backend === 'neural-engine') return { cls: 'good', text: 'available' };
  if (a.used && (a.prefill || a.decode)) return { cls: 'good', text: a.backend === 'cpu' ? 'always' : a.decode ? 'prefill + decode' : 'prefill' };
  if (a.available) return { cls: 'muted', text: 'not used' };
  return { cls: 'muted', text: 'none' };
}

async function loadCatalog() {
  try {
    catalog.value = await store.api.catalog();
    catalogError.value = '';
    // The CPU speed estimates come a few seconds later.
    if (!catalog.value.calibrated) setTimeout(loadCatalog, 4000);
  } catch (e) {
    catalogError.value = e.message;
  }
}

async function loadCache() {
  try {
    cache.value = await store.api.cache();
    cacheError.value = '';
  } catch (e) {
    cacheError.value = e.message;
  }
}

function openSetup(spec, model = null) {
  setup.value = { spec, model };
}

function closeSetup() {
  setup.value = null;
  loadCache();
  loadCatalog();
}

function openManual() {
  const s = manual.value.trim();
  if (s) openSetup(s);
}

async function cancel(id) {
  try {
    await store.api.cancelDownload(id);
  } finally {
    refresh();
  }
}

async function unload(id) {
  try {
    await store.api.unload(id);
  } catch (e) {
    store.error = e.message;
  }
  refresh();
}

function askDelete(e) {
  deleteError.value = '';
  confirmDelete.value = e;
  requestAnimationFrame(() => dialog.value?.showModal());
}

function closeDelete() {
  dialog.value?.close();
  confirmDelete.value = null;
}

async function doDelete() {
  const e = confirmDelete.value;
  if (!e) return;
  deleting.value = true;
  deleteError.value = '';
  try {
    await store.api.removeCached(e.name);
    closeDelete();
    await Promise.all([loadCache(), refresh()]);
    loadCatalog();
  } catch (err) {
    deleteError.value = err.message;
  }
  deleting.value = false;
}

const downloads = computed(() => [...(store.status?.downloads || [])].reverse());
const activeDownloads = computed(() => downloads.value.filter((d) => d.status === 'planning' || d.status === 'downloading'));
const disk = computed(() => store.status?.cache);
const usedFraction = computed(() => (disk.value?.total ? 1 - disk.value.free / disk.value.total : 0));
const cacheTotal = computed(() => (cache.value?.entries || []).reduce((a, e) => a + e.bytes, 0));
const readOnly = computed(() => store.status && !store.status.admin);

function dlFraction(d) {
  return d.total ? Math.min(1, d.done / d.total) : 0;
}

function dlLine(d) {
  if (d.status === 'planning') return 'Listing the files…';
  if (d.status === 'done') return `Done: ${fmtBytes(d.done)}${d.elapsed ? ` in ${fmtSecs(d.elapsed)}` : ''}.`;
  if (d.status === 'cancelled') return `Cancelled at ${fmtBytes(d.done)}${d.total ? ` of ${fmtBytes(d.total)}` : ''}. Downloading again resumes it.`;
  if (d.status === 'error') return `Failed: ${d.error}`;
  const parts = [`${fmtBytes(d.done)}${d.total ? ` of ${fmtBytes(d.total)}` : ''}`];
  if (d.speed) parts.push(`${fmtBytes(d.speed)}/s`);
  if (d.eta) parts.push(`${fmtSecs(d.eta)} left`);
  return parts.join(' · ');
}

function useKind(kind) {
  const tab = { chat: 'chat', image: 'image', transcription: 'transcribe', speech: 'speak', embedding: 'embed' }[kind];
  if (tab) emit('use', tab);
}

function setupFromCache(e) {
  const spec = e.specs[0];
  const model = catalog.value?.models.find((m) => e.catalog.includes(m.model));
  openSetup(spec, model || null);
}

onMounted(() => {
  loadCatalog();
  loadCache();
  // Picks up downloads that finish while the page is open.
  let seen = activeDownloads.value.length;
  timer = setInterval(() => {
    const n = activeDownloads.value.length;
    if (n < seen) {
      loadCache();
      loadCatalog();
    }
    seen = n;
  }, 1000);
});
onUnmounted(() => clearInterval(timer));
</script>

<template>
  <main class="layout manage">
    <aside class="side">
      <section aria-labelledby="machine-h">
        <h2 id="machine-h">This computer</h2>
        <template v-if="store.status">
          <p class="device">{{ store.status.platform }}</p>
          <dl class="speed">
            <dt>oxinfer</dt>
            <dd>{{ store.status.version }}</dd>
            <dt>Memory</dt>
            <dd>{{ fmtBytes(store.status.memory) }}{{ store.status.memory_available ? `, ${fmtBytes(store.status.memory_available)} free` : '' }}</dd>
            <dt>For models</dt>
            <dd>{{ store.status.memory_budget ? fmtBytes(store.status.memory_budget) : 'no limit' }}</dd>
            <template v-if="!store.status.accelerators">
              <template v-for="g in store.status.gpus" :key="g.name">
                <dt>GPU</dt>
                <dd>{{ g.name }} ({{ g.backend }}{{ g.memory ? `, ${fmtBytes(g.memory)}` : '' }})</dd>
              </template>
            </template>
          </dl>
          <ul v-if="store.status.accelerators" class="plain accels" aria-label="Accelerators">
            <li v-for="a in store.status.accelerators" :key="`${a.backend}-${a.name}`" :class="['accel', { off: !a.used }]">
              <div class="accel-head">
                <span class="accel-kind">{{ ACCEL_KIND[a.backend] || a.backend }}</span>
                <span :class="['badge', accelState(a).cls]">{{ accelState(a).text }}</span>
              </div>
              <span v-if="a.backend !== 'gpu' && a.backend !== 'neural-engine'" class="accel-name">{{ a.name }}<template v-if="a.memory_gb && a.backend !== 'cpu'"> · {{ a.memory_gb.toFixed(0) }} GB{{ a.unified_memory ? ' unified' : '' }}</template></span>
              <span class="accel-status">{{ a.status }}</span>
            </li>
          </ul>
        </template>
      </section>

      <section aria-labelledby="disk-h">
        <h2 id="disk-h">Disk</h2>
        <template v-if="disk">
          <div v-if="disk.total" class="bar" role="img" :aria-label="`${fmtBytes(disk.free)} free of ${fmtBytes(disk.total)}`">
            <div class="fill used" :style="{ width: `${(usedFraction * 100).toFixed(1)}%` }"></div>
          </div>
          <p class="device">
            <template v-if="disk.total">{{ fmtBytes(disk.free) }} free of {{ fmtBytes(disk.total) }}. </template>
            Models take {{ fmtBytes(cacheTotal) }} in <code class="path">{{ disk.path }}</code>.
          </p>
        </template>
      </section>

      <section aria-labelledby="inuse-h">
        <h2 id="inuse-h">Models in use</h2>
        <p v-if="!store.status?.models?.length" class="hint">None yet. Loaded models also answer on <code>/v1</code> for other programs.</p>
        <ul class="plain">
          <li v-for="m in store.status?.models || []" :key="m.id" class="inuse">
            <strong class="mono">{{ m.id }}</strong>
            <span :class="['state', m.state]">{{ loadLine(m) }}</span>
            <span class="row-actions">
              <button v-if="m.kind" type="button" class="link" @click="useKind(m.kind)">Use for {{ KIND_LABEL[m.kind] || m.kind }}</button>
              <button v-if="m.loaded || ['queued', 'waiting', 'downloading', 'loading'].includes(m.state)" type="button" class="link" :disabled="m.busy && m.loaded" @click="unload(m.id)">Unload</button>
            </span>
          </li>
        </ul>
      </section>
      <p v-if="readOnly" class="hint warn-text">This server listens beyond this computer without an API key, so downloads, deletions and loads are turned off.</p>
    </aside>

    <section class="pane" aria-label="Models">
      <section v-if="downloads.length" class="block-section" aria-labelledby="dl-h">
        <h2 id="dl-h">Downloads</h2>
        <ul class="plain">
          <li v-for="d in downloads.slice(0, 6)" :key="d.id" class="download">
            <div class="dl-head">
              <strong class="mono">{{ d.model }}</strong>
              <button v-if="d.status === 'downloading' || d.status === 'planning'" type="button" class="quiet small" @click="cancel(d.id)">Cancel</button>
            </div>
            <div v-if="d.status === 'downloading' || d.status === 'planning'" class="bar"><div class="fill" :style="{ width: `${(dlFraction(d) * 100).toFixed(1)}%` }"></div></div>
            <p :class="['status', d.status === 'error' ? 'error' : d.status === 'done' ? 'ready' : '']">{{ dlLine(d) }}</p>
            <p v-if="d.status === 'downloading' && d.file" class="hint">File {{ d.index }} of {{ d.count }}: {{ d.file }}</p>
          </li>
        </ul>
      </section>

      <section class="block-section" aria-labelledby="cache-h">
        <h2 id="cache-h">Downloaded models</h2>
        <p v-if="cacheError" class="error" role="alert">{{ cacheError }}</p>
        <p v-else-if="cache && !cache.entries.length" class="empty-note">No models downloaded yet — pick one from the catalog.</p>
        <div v-else-if="cache" class="table-wrap">
          <table class="cache">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Kind</th>
                <th scope="col" class="num-col">Size</th>
                <th scope="col">Last used</th>
                <th scope="col"><span class="sr-only">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="e in cacheShown" :key="e.name">
                <td>
                  <span class="mono">{{ e.repo || e.name }}</span>
                  <span v-if="e.catalog.length" class="badge">catalog</span>
                  <span v-if="e.loaded" class="badge good">in use</span>
                  <span v-if="e.downloading" class="badge warn">downloading</span>
                  <span v-if="!e.usable" class="badge bad">incomplete</span>
                </td>
                <td>{{ e.kind || '—' }}</td>
                <td class="num-col">{{ fmtBytes(e.bytes) }}</td>
                <td>{{ fmtAgo(e.last_used) }}</td>
                <td class="row-actions">
                  <button v-if="e.usable && e.kind" type="button" class="link" @click="setupFromCache(e)">Load…</button>
                  <button type="button" class="link danger" :disabled="e.loaded || e.downloading || readOnly" :title="e.loaded ? 'Unload it first' : e.downloading ? 'Cancel the download first' : ''" @click="askDelete(e)">Delete</button>
                </td>
              </tr>
            </tbody>
          </table>
          <button v-if="!cacheAll && cache.entries.length > cacheShown.length" type="button" class="link" @click="cacheAll = true">
            Show all {{ cache.entries.length }} downloaded models
          </button>
        </div>
      </section>

      <section class="block-section" aria-labelledby="cat-h">
        <h2 id="cat-h">Catalog</h2>
        <p class="hint">The models oxinfer was tested with, checked against this computer’s memory, GPU and free disk space.</p>
        <div class="filters">
          <div class="modes tabs" role="group" aria-label="Kind of model">
            <button v-for="[g, label] in GROUPS.filter(([g]) => counts[g] || g === 'all')" :key="g" type="button" :aria-pressed="group === g" @click="group = g">
              {{ label }} <span class="count">{{ counts[g] || 0 }}</span>
            </button>
          </div>
          <input v-model="query" class="field search" type="search" placeholder="Filter by name or ability" aria-label="Filter the catalog" />
        </div>
        <p v-if="catalogError" class="error" role="alert">{{ catalogError }}</p>
        <p v-else-if="!catalog" class="hint" role="status">Checking this computer…</p>
        <p v-else-if="!shown.length" class="hint">No model matches.</p>
        <ul class="cards">
          <li v-for="m in shown" :key="m.model" class="card">
            <div class="card-head">
              <h3 class="mono">{{ m.model }}</h3>
              <span v-if="m.cached" class="badge good">Downloaded</span>
            </div>
            <p class="facts">
              {{ m.params_b >= 1 ? `${m.params_b} B` : `${Math.round(m.params_b * 1000)} M` }} parameters · {{ m.download_gb >= 10 ? m.download_gb.toFixed(0) : m.download_gb.toFixed(1) }} GB download ·
              {{ recommended(m).label }}, {{ recommended(m).memory_gb.toFixed(1) }} GB in memory
            </p>
            <p :class="['fit', fit(m).cls]">{{ fit(m).text }}</p>
            <p v-if="fit(m).accel" class="accel-line">Runs on: {{ fit(m).accel }}</p>
            <p class="caps">{{ m.capabilities.join(' · ') }}</p>
            <p v-if="m.note" class="hint note">{{ m.note }}</p>
            <button type="button" class="quiet small" :disabled="!m.servable" @click="openSetup(m.model, m)">{{ m.cached ? 'Load…' : 'Set up…' }}</button>
          </li>
        </ul>
        <button v-if="!showAll && group === 'all' && !query && (catalog?.models?.length || 0) > shown.length" type="button" class="link" @click="showAll = true">
          Show all {{ catalog.models.length }} models
        </button>
        <template v-if="catalog?.presets?.length && (group === 'all' || group === 'image') && !query">
          <h3 class="presets-h">Presets</h3>
          <p class="hint">A catalog model with LoRAs and generation settings, set up under the preset's name (its LoRAs download with the first load).</p>
          <ul class="cards">
            <li v-for="p in catalog.presets" :key="p.id" class="card" :data-preset="p.id">
              <div class="card-head">
                <h3 class="mono">{{ p.id }}</h3>
                <span v-if="p.listed" class="badge good">Set up</span>
                <span v-else-if="p.base_cached" class="badge">base downloaded</span>
              </div>
              <p class="facts">{{ p.base }}{{ p.extra_download_gb ? ` + ${p.extra_download_gb} GB of LoRA` : '' }} · {{ Object.entries(p.defaults).map(([k, v]) => `${k} ${v}`).join(', ') }}</p>
              <p class="hint note">{{ p.note }}</p>
              <button type="button" class="quiet small" @click="openSetup(p.id)">{{ p.base_cached ? 'Load…' : 'Set up…' }}</button>
            </li>
          </ul>
        </template>
      </section>

      <section class="block-section" aria-labelledby="any-h">
        <h2 id="any-h">Any other model</h2>
        <form class="manual" @submit.prevent="openManual">
          <input v-model="manual" class="field" type="text" placeholder="org/name, org/name:Q4_K_M or /path/to/model" aria-label="Model spec" spellcheck="false" />
          <button type="submit" class="primary" :disabled="!manual.trim()">Set up…</button>
        </form>
        <p class="hint">A Hugging Face repo id (fetched from the ATOnline mirror when it has it, else Hugging Face), <code>org/name:QUANT</code> for one GGUF file of a repo, or a folder or file on this computer.</p>
      </section>

    </section>

    <SetupDialog v-if="setup" :spec="setup.spec" :model="setup.model" @close="closeSetup" @use="(k) => (closeSetup(), useKind(k))" />

    <dialog ref="dialog" class="confirm" aria-labelledby="del-h" @close="confirmDelete = null">
      <template v-if="confirmDelete">
        <h2 id="del-h">Delete {{ confirmDelete.repo || confirmDelete.name }}?</h2>
        <p>{{ fmtBytes(confirmDelete.bytes) }} in <code class="path">{{ confirmDelete.path }}</code> will be removed from the disk. Downloading it again takes the whole download.</p>
        <p v-if="deleteError" class="error" role="alert">{{ deleteError }}</p>
        <div class="actions">
          <button type="button" class="primary danger-bg" :disabled="deleting" @click="doDelete">{{ deleting ? 'Deleting…' : 'Delete' }}</button>
          <button type="button" class="quiet" autofocus @click="closeDelete">Keep it</button>
        </div>
      </template>
    </dialog>
  </main>
</template>

<style scoped>
.manage {
  grid-template-columns: minmax(16rem, 21rem) 1fr;
}
.pane {
  overflow-y: auto;
  padding: 1.25rem clamp(1rem, 4vw, 3rem) 3rem;
}
.block-section + .block-section {
  margin-top: 2.25rem;
}
.filters {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
  margin: 0.75rem 0 1rem;
}
.tabs button {
  font-size: 0.9rem;
}
.count {
  color: var(--muted);
  font-weight: 500;
  font-size: 0.8rem;
}
.tabs button[aria-pressed='true'] .count {
  color: inherit;
  opacity: 0.8;
}
.search {
  max-width: 18rem;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 19rem), 1fr));
  gap: 0.9rem;
  margin: 0;
  padding: 0;
  list-style: none;
}
.card {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  padding: 0.8rem 0.9rem;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--surface);
}
.card-head {
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
  align-items: start;
}
.card h3 {
  margin: 0;
  font-size: 0.92rem;
  font-weight: 650;
  overflow-wrap: anywhere;
}
.facts,
.caps {
  margin: 0;
  font-size: 0.85rem;
}
.caps {
  color: var(--muted);
}
.fit {
  margin: 0;
  font-size: 0.88rem;
  font-weight: 600;
}
.note {
  margin-top: 0;
}
.accel-line {
  margin: 0;
  font-size: 0.85rem;
  color: var(--muted);
}
.accels {
  margin-top: 0.6rem;
}
.accel {
  display: grid;
  gap: 0.1rem;
  padding: 0.45rem 0;
  border-top: 1px solid var(--line);
  font-size: 0.86rem;
}
.accel.off .accel-kind,
.accel.off .accel-name {
  color: var(--muted);
}
.accel-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 0.5rem;
}
.accel-kind {
  font-family: var(--display);
  font-weight: 650;
}
.accel-name {
  overflow-wrap: anywhere;
}
.accel-status {
  color: var(--muted);
  font-size: 0.82rem;
}
.badge.muted {
  color: var(--muted);
}
.card .quiet {
  align-self: start;
  margin-top: auto;
}
.manual {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
  max-width: 44rem;
}
.manual .field {
  flex: 1 1 18rem;
}
.manual .primary {
  margin-top: 0;
}
.download {
  padding: 0.6rem 0;
  border-bottom: 1px solid var(--line);
  max-width: 44rem;
}
.dl-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.6rem;
}
.download .bar {
  margin-top: 0.5rem;
}
.inuse {
  display: grid;
  gap: 0.1rem;
  padding: 0.45rem 0;
  border-bottom: 1px solid var(--line);
  font-size: 0.9rem;
}
.state {
  color: var(--muted);
}
.state.ready,
.state.busy {
  color: var(--verdigris-deep);
}
.state.error {
  color: var(--error);
}
.used {
  background: var(--copper);
}
.path {
  overflow-wrap: anywhere;
}
.empty-note {
  margin: 0.5rem 0 0;
  color: var(--muted);
  font-family: var(--display);
  font-size: 1.15rem;
}
.table-wrap {
  overflow-x: auto;
}
.cache {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}
.cache th {
  text-align: left;
  font-family: var(--display);
  font-weight: 600;
  color: var(--muted);
  border-bottom: 1px solid var(--line);
  padding: 0.35rem 0.6rem 0.35rem 0;
}
.cache td {
  padding: 0.4rem 0.6rem 0.4rem 0;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}
.num-col {
  text-align: right;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.row-actions {
  display: flex;
  gap: 0.9rem;
  white-space: nowrap;
}
.confirm .actions .primary,
.confirm .actions .quiet {
  margin-top: 0.4rem;
}
.warn-text {
  color: var(--copper);
}
@media (max-width: 760px) {
  .manage {
    grid-template-columns: 1fr;
  }
}
.presets-h {
  margin: 1.5rem 0 0.25rem;
}
</style>
