<script setup>
// What the native backend is doing, live: the requests running and waiting for their model
// (with a cancel), the models with their memory and device (unload, free cached experts), the
// process's and the machine's resources, and the prompt caches (clear). One GET of
// /api/web/activity a second while the page is shown.
import { computed, onMounted, onUnmounted, ref } from 'vue';
import { store, refresh, fmtBytes, fmtSecs, fmtAgo } from '../lib/native.js';

const KIND_LABEL = { chat: 'chat', completion: 'completion', embedding: 'embeddings', image: 'image', transcription: 'speech to text', speech: 'text to speech', video: 'video' };

const st = ref(null);
const error = ref('');
const busy = ref({}); // action key -> true while it runs
const confirm = ref(null); // { title, text, action, label }
const confirmError = ref('');
const dialog = ref(null);
let timer = null;

async function load() {
  if (document.hidden) return;
  try {
    st.value = await store.api.activity();
    error.value = '';
  } catch (e) {
    error.value = `The status could not be read (${e.message}).`;
  }
}

const readOnly = computed(() => st.value && st.value.admin === false);
const requests = computed(() => st.value?.requests || []);
const running = computed(() => requests.value.filter((r) => r.stage !== 'queued'));
const queued = computed(() => requests.value.filter((r) => r.stage === 'queued'));
const videos = computed(() => st.value?.videos || []);
const models = computed(() => [...(st.value?.models || [])].sort((a, b) => b.loaded - a.loaded || b.memory - a.memory || a.id.localeCompare(b.id)));
const loadedModels = computed(() => models.value.filter((m) => m.loaded || m.busy));
const otherModels = computed(() => models.value.filter((m) => !m.loaded && !m.busy));
const showAllModels = ref(false);
const cache = computed(() => st.value?.prompt_cache);
const proc = computed(() => st.value?.process || {});
const machine = computed(() => st.value?.machine || {});
const gpu = computed(() => st.value?.gpu);

const budgetFraction = computed(() => {
  const s = st.value;
  return s?.memory_budget ? Math.min(1, s.models_memory / s.memory_budget) : 0;
});
const cacheFraction = computed(() => (cache.value?.cap ? Math.min(1, cache.value.bytes / cache.value.cap) : 0));

const PRESSURE = { normal: ['normal', 'good'], warning: ['warning', 'warn'], critical: ['critical', 'bad'] };
const pressure = computed(() => {
  const p = machine.value.pressure;
  if (!p) return null;
  if (p.level) return { text: `${PRESSURE[p.level]?.[0] || p.level}${p.free_percent != null ? `, ${p.free_percent}% free` : ''}`, cls: PRESSURE[p.level]?.[1] || '' };
  if (p.some_avg10 != null) return { text: `${p.some_avg10.toFixed(1)}% stalled (10 s)`, cls: p.some_avg10 > 10 ? 'bad' : p.some_avg10 > 1 ? 'warn' : 'good' };
  return null;
});

function stageLine(r) {
  switch (r.stage) {
    case 'starting':
      return 'Starting';
    case 'queued':
      return 'Waiting for the model';
    case 'downloading':
      return r.download?.total ? `Downloading the model: ${Math.floor((100 * r.download.done) / r.download.total)}%` : 'Downloading the model';
    case 'loading':
      return 'Loading the model';
    case 'prompt':
      return r.prompt ? `Reading the prompt: ${r.prompt.done} of ${r.prompt.total} tokens` : 'Reading the prompt';
    case 'generating':
      if (r.step) return r.step.total ? `Step ${r.step.done} of ${r.step.total}` : `Step ${r.step.done}`;
      return r.tokens ? 'Generating' : 'Working';
    default:
      return r.stage;
  }
}

function videoLine(v) {
  if (v.stage === 'generating' && v.steps) return `Step ${v.step} of ${v.steps}`;
  return v.stage.charAt(0).toUpperCase() + v.stage.slice(1);
}

function rate(r) {
  return r.tokens_per_sec ? `${r.tokens_per_sec.toFixed(1)}` : '—';
}

function client(r) {
  if (!r.client) return '—';
  return r.client.replace(/^(127\.0\.0\.1|\[::1\]):\d+$/, 'this computer');
}

function modelLabel(g) {
  if (g.ids?.length) return g.ids.join(', ');
  return g.model_type ? `${g.model_type} (not loaded since the start)` : 'unreadable snapshots';
}

const STATE = { ready: ['loaded', 'good'], busy: ['working', 'good'], loading: ['loading', 'warn'], unloaded: ['not loaded', ''] };

async function act(key, fn) {
  busy.value = { ...busy.value, [key]: true };
  try {
    await fn();
    error.value = '';
  } catch (e) {
    error.value = e.message;
  } finally {
    const b = { ...busy.value };
    delete b[key];
    busy.value = b;
    await load();
    refresh();
  }
}

function cancel(r) {
  act(`cancel:${r.id}`, () => store.api.cancelRequest(r.id));
}

function unload(m) {
  if (!m.busy) {
    act(`unload:${m.id}`, () => store.api.unload(m.id));
    return;
  }
  ask({
    title: `Stop ${m.id} and unload it?`,
    text: `${m.requests ? `${m.requests} request${m.requests === 1 ? ' is' : 's are'} using it. They are cancelled (their clients get an error)` : 'It is in use: its requests are cancelled'}, then the model is unloaded. It loads again on its next request.`,
    label: 'Cancel and unload',
    action: () => store.api.unload(m.id, { cancel: true }),
  });
}

function freeCache(m) {
  act(`free:${m.id}`, () => store.api.freeCaches(m.id));
}

function clearGroup(dir, g) {
  ask({
    title: `Clear the prompt cache of ${modelLabel(g)}?`,
    text: `${g.entries} snapshot${g.entries === 1 ? '' : 's'} (${fmtBytes(g.bytes)}) in ${dir.path} will be removed. Conversations that would have restored them read their prompt again.`,
    label: 'Clear',
    action: () => store.api.clearPromptCache({ fingerprint: g.fingerprint, dir: dir.path }),
  });
}

function clearAll() {
  ask({
    title: 'Clear the whole prompt cache?',
    text: `All ${cache.value.entries} snapshots (${fmtBytes(cache.value.bytes)}) of every model will be removed.`,
    label: 'Clear all',
    action: () => store.api.clearPromptCache({}),
  });
}

function ask(c) {
  confirmError.value = '';
  confirm.value = c;
  requestAnimationFrame(() => dialog.value?.showModal());
}

function closeConfirm() {
  dialog.value?.close();
  confirm.value = null;
}

async function doConfirm() {
  const c = confirm.value;
  if (!c) return;
  busy.value = { ...busy.value, confirm: true };
  confirmError.value = '';
  try {
    await c.action();
    closeConfirm();
  } catch (e) {
    confirmError.value = e.message;
  } finally {
    const b = { ...busy.value };
    delete b.confirm;
    busy.value = b;
    await load();
    refresh();
  }
}

function fmtUptime(s) {
  if (s == null) return '';
  const d = Math.floor(s / 86400);
  if (d >= 1) return `${d} day${d === 1 ? '' : 's'} ${Math.floor((s % 86400) / 3600)} h`;
  return fmtSecs(s);
}

function onVisible() {
  if (!document.hidden) load();
}

onMounted(() => {
  load();
  timer = setInterval(load, 1000);
  document.addEventListener('visibilitychange', onVisible);
});
onUnmounted(() => {
  clearInterval(timer);
  document.removeEventListener('visibilitychange', onVisible);
});
</script>

<template>
  <main class="layout manage">
    <aside class="side">
      <section aria-labelledby="server-h">
        <h2 id="server-h">Server</h2>
        <dl v-if="st" class="speed">
          <dt>oxinfer</dt>
          <dd>{{ st.version }}</dd>
          <dt>Up for</dt>
          <dd>{{ fmtUptime(st.uptime) }}</dd>
          <dt>Listens on</dt>
          <dd class="mono">{{ st.listen }}</dd>
          <dt>Requests</dt>
          <dd>{{ st.served }} served{{ st.cancelled ? `, ${st.cancelled} cancelled` : '' }}</dd>
        </dl>
      </section>

      <section aria-labelledby="proc-h">
        <h2 id="proc-h">This process</h2>
        <dl v-if="st" class="speed">
          <dt>CPU</dt>
          <dd>{{ proc.cpu_percent != null ? `${Math.round(proc.cpu_percent)}%` : '—' }}<span v-if="store.status?.cores" class="muted"> of {{ store.status.cores * 100 }}%</span></dd>
          <dt>Memory</dt>
          <dd>{{ fmtBytes(proc.footprint ?? proc.resident) }}<span v-if="proc.footprint != null && proc.resident != null" class="muted"> ({{ fmtBytes(proc.resident) }} resident)</span></dd>
          <template v-if="proc.peak_resident">
            <dt>Peak</dt>
            <dd>{{ fmtBytes(proc.peak_resident) }}</dd>
          </template>
          <template v-if="gpu">
            <dt>GPU</dt>
            <dd>
              {{ gpu.backend }}<template v-if="gpu.allocated != null">, {{ fmtBytes(gpu.allocated) }} allocated</template>
              <template v-if="gpu.device_total != null">, {{ fmtBytes(gpu.device_total - gpu.device_free) }} of {{ fmtBytes(gpu.device_total) }} used</template>
            </dd>
          </template>
        </dl>
      </section>

      <section aria-labelledby="machine-mem-h">
        <h2 id="machine-mem-h">Memory</h2>
        <template v-if="st">
          <dl class="speed">
            <dt>This computer</dt>
            <dd>{{ fmtBytes(machine.memory) }}{{ machine.memory_available != null ? `, ${fmtBytes(machine.memory_available)} free` : '' }}</dd>
            <template v-if="pressure">
              <dt>Pressure</dt>
              <dd><span :class="['badge', 'flat', pressure.cls]">{{ pressure.text }}</span></dd>
            </template>
            <dt>Models</dt>
            <dd>{{ fmtBytes(st.models_memory) }} of {{ st.memory_budget ? fmtBytes(st.memory_budget) : 'no limit' }}</dd>
          </dl>
          <div v-if="st.memory_budget" class="bar" role="img" :aria-label="`Models take ${fmtBytes(st.models_memory)} of ${fmtBytes(st.memory_budget)}`">
            <div class="fill" :style="{ width: `${(budgetFraction * 100).toFixed(1)}%` }"></div>
          </div>
          <p v-if="st.idle_unload" class="hint">Models unused for {{ fmtSecs(st.idle_unload) }} are unloaded.</p>
        </template>
      </section>
      <p v-if="readOnly" class="hint warn-text">This server listens beyond this computer without an API key, so cancelling, unloading and clearing are turned off.</p>
    </aside>

    <section class="pane" aria-label="Server status">
      <p v-if="error" class="error" role="alert">{{ error }}</p>
      <p v-if="!st && !error" class="empty-note">Reading the status…</p>

      <template v-if="st">
        <section class="block-section" aria-labelledby="running-h">
          <h2 id="running-h">Running <span class="count">{{ running.length + videos.length }}</span></h2>
          <p v-if="!running.length && !videos.length" class="empty-note">Nothing is running.</p>
          <div v-else class="table-wrap">
            <table class="cache live">
              <thead>
                <tr>
                  <th scope="col">Request</th>
                  <th scope="col">Model</th>
                  <th scope="col">Doing</th>
                  <th scope="col" class="num-col">Time</th>
                  <th scope="col" class="num-col">Tokens</th>
                  <th scope="col" class="num-col">tok/s</th>
                  <th scope="col">Client</th>
                  <th scope="col"><span class="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="r in running" :key="r.id">
                  <td>
                    <span class="mono">{{ r.id }}</span>
                    <span class="badge">{{ KIND_LABEL[r.kind] || r.kind }}</span>
                    <span v-if="r.stream" class="badge">stream</span>
                  </td>
                  <td class="mono">{{ r.model || '—' }}</td>
                  <td>
                    <span :class="['state', r.cancelling ? 'warn-text' : '']">{{ r.cancelling ? 'Stopping…' : stageLine(r) }}</span>
                    <div v-if="r.step?.total || r.prompt?.total" class="bar thin">
                      <div class="fill" :style="{ width: `${(100 * (r.step ? r.step.done / r.step.total : r.prompt.done / r.prompt.total)).toFixed(1)}%` }"></div>
                    </div>
                  </td>
                  <td class="num-col">{{ fmtSecs(r.elapsed) }}</td>
                  <td class="num-col">{{ r.tokens || '—' }}</td>
                  <td class="num-col">{{ rate(r) }}</td>
                  <td :title="r.user_agent || ''">{{ client(r) }}</td>
                  <td class="row-actions">
                    <button type="button" class="link danger" :disabled="readOnly || r.cancelling || busy[`cancel:${r.id}`]" @click="cancel(r)">Cancel</button>
                  </td>
                </tr>
                <tr v-for="v in videos" :key="v.id">
                  <td>
                    <span class="mono">{{ v.id }}</span>
                    <span class="badge">video</span>
                  </td>
                  <td class="mono">{{ v.model }}</td>
                  <td>
                    <span class="state">{{ videoLine(v) }}</span>
                    <div v-if="v.progress" class="bar thin"><div class="fill" :style="{ width: `${v.progress}%` }"></div></div>
                  </td>
                  <td class="num-col">{{ fmtSecs(v.elapsed) }}</td>
                  <td class="num-col">—</td>
                  <td class="num-col">—</td>
                  <td>—</td>
                  <td class="row-actions">
                    <button type="button" class="link danger" :disabled="readOnly || v.stage === 'cancelling' || busy[`cancel:${v.id}`]" @click="cancel(v)">Cancel</button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section v-if="queued.length" class="block-section" aria-labelledby="queued-h">
          <h2 id="queued-h">Waiting <span class="count">{{ queued.length }}</span></h2>
          <p class="hint">Each model runs one request at a time; these wait for theirs.</p>
          <div class="table-wrap">
            <table class="cache">
              <thead>
                <tr>
                  <th scope="col">Request</th>
                  <th scope="col">Model</th>
                  <th scope="col" class="num-col">Waiting</th>
                  <th scope="col">Client</th>
                  <th scope="col"><span class="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="r in queued" :key="r.id">
                  <td>
                    <span class="mono">{{ r.id }}</span>
                    <span class="badge">{{ KIND_LABEL[r.kind] || r.kind }}</span>
                  </td>
                  <td class="mono">{{ r.model || '—' }}</td>
                  <td class="num-col">{{ fmtSecs(r.elapsed) }}</td>
                  <td :title="r.user_agent || ''">{{ client(r) }}</td>
                  <td class="row-actions">
                    <button type="button" class="link danger" :disabled="readOnly || r.cancelling || busy[`cancel:${r.id}`]" @click="cancel(r)">Cancel</button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="block-section" aria-labelledby="models-h">
          <h2 id="models-h">Models <span class="count">{{ loadedModels.length }} loaded</span></h2>
          <p v-if="!loadedModels.length" class="empty-note">No model is loaded. Models load on their first request.</p>
          <div v-else class="table-wrap">
            <table class="cache">
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">State</th>
                  <th scope="col">Device</th>
                  <th scope="col" class="num-col">Memory</th>
                  <th scope="col">Streamed from disk</th>
                  <th scope="col">Loaded</th>
                  <th scope="col"><span class="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="m in loadedModels" :key="m.id">
                  <td>
                    <span class="mono">{{ m.id }}</span>
                    <span v-if="m.kind" class="badge">{{ KIND_LABEL[m.kind] || m.kind }}</span>
                    <div v-if="m.model_type" class="hint">{{ m.model_type }}{{ m.quant ? `, ${m.quant}` : '' }}{{ m.format ? `, ${m.format}` : '' }}</div>
                  </td>
                  <td>
                    <span :class="['badge', 'flat', STATE[m.state]?.[1]]">{{ STATE[m.state]?.[0] || m.state }}</span>
                    <div v-if="m.requests" class="hint">{{ m.requests }} request{{ m.requests === 1 ? '' : 's' }}</div>
                    <div v-else-if="m.idle_secs != null" class="hint">idle {{ fmtSecs(m.idle_secs) }}</div>
                  </td>
                  <td>{{ m.device || '—' }}</td>
                  <td class="num-col">
                    {{ m.memory ? fmtBytes(m.memory) : '—' }}
                    <div v-if="m.vram" class="hint">{{ fmtBytes(m.vram) }} on the GPU</div>
                  </td>
                  <td>
                    <template v-if="m.stream">
                      {{ fmtBytes(m.stream.resident_bytes) }} resident
                      <div v-if="m.stream.part_cache.cap" class="hint">
                        experts cached: {{ fmtBytes(m.stream.part_cache.used) }} of {{ fmtBytes(m.stream.part_cache.cap) }}<template v-if="m.stream.parts">, {{ Math.round((100 * m.stream.parts_cached) / m.stream.parts) }}% hits</template>
                      </div>
                      <div v-if="m.stream.budget" class="hint">page-cache budget {{ fmtBytes(m.stream.budget) }}</div>
                    </template>
                    <template v-else>no</template>
                  </td>
                  <td>{{ m.loaded_at ? fmtAgo(m.loaded_at) : '—' }}<div v-if="m.load_secs" class="hint">in {{ fmtSecs(m.load_secs) }}</div></td>
                  <td class="row-actions">
                    <button v-if="m.stream?.part_cache.used" type="button" class="link" :disabled="readOnly || busy[`free:${m.id}`]" title="Drop the cached experts from memory; they are read from disk again when needed" @click="freeCache(m)">Free cache</button>
                    <button v-if="m.loaded" type="button" class="link danger" :disabled="readOnly || busy[`unload:${m.id}`]" @click="unload(m)">{{ m.busy ? 'Stop and unload' : 'Unload' }}</button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <template v-if="otherModels.length">
            <button v-if="!showAllModels" type="button" class="link" @click="showAllModels = true">Show the {{ otherModels.length }} models not loaded</button>
            <ul v-else class="plain others">
              <li v-for="m in otherModels" :key="m.id"><span class="mono">{{ m.id }}</span> <span v-if="m.kind" class="badge">{{ KIND_LABEL[m.kind] || m.kind }}</span></li>
            </ul>
          </template>
        </section>

        <section class="block-section" aria-labelledby="pc-h">
          <div class="section-head">
            <h2 id="pc-h">Prompt cache</h2>
            <button v-if="cache?.entries" type="button" class="quiet small" :disabled="readOnly" @click="clearAll">Clear all</button>
          </div>
          <p v-if="!cache?.dirs.length" class="empty-note">Off. Start the server with <code>--prompt-cache auto</code> to keep conversations across requests and restarts.</p>
          <template v-else>
            <p class="hint">
              {{ cache.entries }} snapshot{{ cache.entries === 1 ? '' : 's' }}, {{ fmtBytes(cache.bytes) }} of {{ fmtBytes(cache.cap) }}. Conversations that start like a stored one restore it instead of reading their prompt again.
            </p>
            <div class="bar"><div class="fill" :style="{ width: `${(cacheFraction * 100).toFixed(1)}%` }"></div></div>
            <div v-for="d in cache.dirs" :key="d.path" class="pc-dir">
              <p class="device"><code class="path">{{ d.path }}</code><span v-if="!d.exists" class="badge">not created yet</span></p>
              <p v-if="!d.models.length" class="hint">Empty.</p>
              <div v-else class="table-wrap">
                <table class="cache">
                  <thead>
                    <tr>
                      <th scope="col">Model</th>
                      <th scope="col" class="num-col">Snapshots</th>
                      <th scope="col" class="num-col">Tokens</th>
                      <th scope="col" class="num-col">Size</th>
                      <th scope="col">Last used</th>
                      <th scope="col"><span class="sr-only">Actions</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="g in d.models" :key="g.fingerprint">
                      <td>
                        <span :class="g.ids?.length ? 'mono' : 'muted'">{{ modelLabel(g) }}</span>
                        <div class="hint mono" :title="g.fingerprint">{{ g.fingerprint ? g.fingerprint.slice(0, 16) : '' }}</div>
                      </td>
                      <td class="num-col">{{ g.entries }}</td>
                      <td class="num-col">{{ g.tokens.toLocaleString() }}</td>
                      <td class="num-col">{{ fmtBytes(g.bytes) }}</td>
                      <td>{{ fmtAgo(g.last_used) }}</td>
                      <td class="row-actions">
                        <button type="button" class="link danger" :disabled="readOnly" @click="clearGroup(d, g)">Clear</button>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </template>
        </section>

        <section v-if="st.recent?.length" class="block-section" aria-labelledby="recent-h">
          <h2 id="recent-h">Recently finished</h2>
          <div class="table-wrap">
            <table class="cache">
              <thead>
                <tr>
                  <th scope="col">Request</th>
                  <th scope="col">Model</th>
                  <th scope="col">Outcome</th>
                  <th scope="col" class="num-col">Time</th>
                  <th scope="col" class="num-col">Tokens</th>
                  <th scope="col" class="num-col">tok/s</th>
                  <th scope="col">Client</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="r in st.recent" :key="r.id">
                  <td>
                    <span class="mono">{{ r.id }}</span>
                    <span class="badge">{{ KIND_LABEL[r.kind] || r.kind }}</span>
                  </td>
                  <td class="mono">{{ r.model || '—' }}</td>
                  <td>
                    <span :class="['badge', 'flat', r.outcome === 'ok' ? 'good' : r.outcome === 'cancelled' ? 'warn' : 'bad']">{{ r.outcome === 'ok' ? 'done' : r.outcome }}</span>
                    <span v-if="r.outcome === 'error' && r.status" class="hint"> {{ r.status }}</span>
                  </td>
                  <td class="num-col">{{ fmtSecs(r.elapsed) }}</td>
                  <td class="num-col">{{ r.tokens || '—' }}</td>
                  <td class="num-col">{{ rate(r) }}</td>
                  <td :title="r.user_agent || ''">{{ client(r) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      </template>
    </section>

    <dialog ref="dialog" class="confirm" aria-labelledby="confirm-h" @close="confirm = null">
      <template v-if="confirm">
        <h2 id="confirm-h">{{ confirm.title }}</h2>
        <p>{{ confirm.text }}</p>
        <p v-if="confirmError" class="error" role="alert">{{ confirmError }}</p>
        <div class="actions">
          <button type="button" class="primary danger-bg" :disabled="busy.confirm" @click="doConfirm">{{ busy.confirm ? 'Working…' : confirm.label }}</button>
          <button type="button" class="quiet" autofocus @click="closeConfirm">Keep it</button>
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
.count {
  margin-left: 0.3rem;
  color: var(--muted);
  font-weight: 500;
  font-size: 0.85rem;
}
.section-head {
  display: flex;
  align-items: baseline;
  gap: 1rem;
}
.section-head .quiet {
  margin-top: 0;
}
.state {
  color: var(--muted);
}
.muted {
  color: var(--muted);
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
  white-space: nowrap;
}
.cache td {
  padding: 0.4rem 0.6rem 0.4rem 0;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}
.cache .hint {
  margin-top: 0.1rem;
}
.live td {
  font-variant-numeric: tabular-nums;
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
.bar.thin {
  height: 4px;
  margin-top: 0.3rem;
  min-width: 6rem;
}
.badge.flat {
  margin-left: 0;
}
.pc-dir {
  margin-top: 1rem;
}
.pc-dir .device {
  margin-bottom: 0.3rem;
}
.path {
  overflow-wrap: anywhere;
}
.others {
  margin-top: 0.5rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1rem;
  font-size: 0.9rem;
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
</style>
