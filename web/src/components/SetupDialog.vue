<script setup>
// The guided setup of one model: what it needs (the files, their size, the free disk space), the
// download with live progress (server-sent events; a download keeps running when the page is
// closed, and opening this again reattaches to it), then the load, and a way to use it.
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue';
import { store, refresh, fmtBytes, fmtSecs, loadLine } from '../lib/native.js';

const props = defineProps({
  spec: { type: String, required: true },
  model: { type: Object, default: null }, // its catalog entry, when it has one
});
const emit = defineEmits(['close', 'use']);

const dlg = ref(null);
const plan = ref(null);
const planError = ref('');
const planning = ref(false);
const job = ref(null);
const jobError = ref('');
const variant = ref(0);
const quant = ref('');
const alias = ref('');
const extra = ref('');
const loadError = ref('');
const loadedId = ref('');
let watching = null;

const isPath = computed(() => /^(\/|~|\.|[A-Za-z]:\\)/.test(props.spec));
const variants = computed(() => props.model?.variants || []);
const guessChat = computed(() => !props.model || props.model.kind === 'chat');
const options = computed(() => {
  const o = {};
  const v = variants.value[variant.value];
  const m = /--quant (\S+)/.exec(v?.args || '');
  if (m) o.quant = m[1];
  else if (!props.model && quant.value) o.quant = quant.value;
  for (const kv of extra.value.split(',').map((s) => s.trim()).filter(Boolean)) {
    const [k, ...rest] = kv.split('=');
    if (k && rest.length) o[k.trim()] = rest.join('=').trim();
  }
  return o;
});
const downloading = computed(() => job.value && (job.value.status === 'planning' || job.value.status === 'downloading'));
const needsDownload = computed(() => !isPath.value && plan.value && plan.value.missing > 0 && job.value?.status !== 'done');
const entry = computed(() => (store.status?.models || []).find((m) => m.id === loadedId.value));
const loadMoving = computed(() => ['queued', 'waiting', 'downloading', 'loading'].includes(entry.value?.state));
const fraction = computed(() => (job.value?.total ? Math.min(1, job.value.done / job.value.total) : 0));

async function check() {
  planError.value = '';
  if (isPath.value) {
    plan.value = { local: true, missing: 0, total: 0, files: [] };
    return;
  }
  planning.value = true;
  try {
    plan.value = await store.api.plan(props.spec);
  } catch (e) {
    planError.value = e.message;
  }
  planning.value = false;
}

async function follow(id) {
  watching?.abort();
  watching = new AbortController();
  try {
    await store.api.watchDownload(id, (j) => (job.value = j), watching.signal);
  } catch (e) {
    if (e.name !== 'AbortError') jobError.value = e.message;
  }
  if (job.value?.status === 'done') {
    await check();
    refresh();
  }
}

async function download() {
  jobError.value = '';
  try {
    job.value = await store.api.download(props.spec);
    refresh();
    follow(job.value.id);
  } catch (e) {
    jobError.value = e.message;
  }
}

async function cancel() {
  if (!job.value) return;
  try {
    job.value = await store.api.cancelDownload(job.value.id);
  } catch (e) {
    jobError.value = e.message;
  }
}

async function load() {
  loadError.value = '';
  try {
    const r = await store.api.load(props.spec, { id: alias.value.trim() || undefined, options: options.value });
    loadedId.value = r.id;
    await refresh();
  } catch (e) {
    loadError.value = e.message;
  }
}

function close() {
  dlg.value?.close();
}

watch(
  () => entry.value?.state,
  (s) => {
    if (s === 'error') loadError.value = entry.value.load?.error || 'the load failed';
  },
);

onMounted(async () => {
  const rec = variants.value.findIndex((v) => v.recommended);
  variant.value = rec >= 0 ? rec : 0;
  await nextTick();
  dlg.value?.showModal();
  // A download of this model already running (the page was reloaded): follow it.
  const running = (store.status?.downloads || []).find((d) => d.model === props.spec && (d.status === 'planning' || d.status === 'downloading'));
  if (running) {
    job.value = running;
    follow(running.id);
  }
  check();
});
onUnmounted(() => watching?.abort());
</script>

<template>
  <dialog ref="dlg" class="setup" aria-labelledby="setup-h" @close="emit('close')">
    <div class="setup-head">
      <h2 id="setup-h" class="mono">{{ spec }}</h2>
      <button type="button" class="link" aria-label="Close" @click="close">Close</button>
    </div>
    <p v-if="model?.note" class="hint">{{ model.note }}</p>

    <ol class="steps">
      <li :class="{ done: plan && !needsDownload }">
        <h3>What it needs</h3>
        <p v-if="planning" role="status">Listing its files…</p>
        <p v-else-if="planError" class="error" role="alert">{{ planError }}</p>
        <template v-else-if="plan">
          <p v-if="plan.local">A file or folder on this computer: nothing to download.</p>
          <template v-else>
            <p>
              {{ plan.files.length }} file{{ plan.files.length === 1 ? '' : 's' }}, {{ fmtBytes(plan.total) }}.
              <template v-if="plan.missing === 0">All downloaded.</template>
              <template v-else-if="plan.missing < plan.total">{{ fmtBytes(plan.missing) }} still to download.</template>
              <template v-else>{{ fmtBytes(plan.missing) }} to download.</template>
            </p>
            <p v-if="plan.free !== null && plan.missing > 0" :class="plan.fits ? 'hint' : 'error'">
              {{ fmtBytes(plan.free) }} free on the disk{{ plan.fits ? '.' : ': not enough room. Delete downloaded models in the list below first.' }}
            </p>
            <p v-if="model" class="hint">{{ model.variants.length > 1 ? 'Recommended here' : 'Runs as' }}: {{ model.variants.find((v) => v.recommended)?.label || model.variants[0].label }}, {{ (model.variants.find((v) => v.recommended) || model.variants[0]).memory_gb.toFixed(1) }} GB of memory. Some pipelines fetch a few more parts when they load.</p>
          </template>
        </template>
      </li>

      <li v-if="!isPath" :class="{ done: plan && !needsDownload && !downloading }">
        <h3>Download</h3>
        <template v-if="job">
          <div class="bar"><div class="fill" :style="{ width: `${(fraction * 100).toFixed(1)}%` }"></div></div>
          <p role="status">
            <template v-if="job.status === 'planning'">Listing the files…</template>
            <template v-else-if="job.status === 'downloading'">
              {{ fmtBytes(job.done) }}{{ job.total ? ` of ${fmtBytes(job.total)}` : '' }}
              <template v-if="job.speed"> · {{ fmtBytes(job.speed) }}/s</template>
              <template v-if="job.eta"> · {{ fmtSecs(job.eta) }} left</template>
            </template>
            <template v-else-if="job.status === 'done'">Downloaded{{ job.downloaded ? ` ${fmtBytes(job.downloaded)}` : '' }} in {{ fmtSecs(job.elapsed) }}.</template>
            <template v-else-if="job.status === 'cancelled'">Cancelled. The part downloaded is kept: downloading again resumes it.</template>
            <template v-else-if="job.status === 'error'"><span class="error">Failed: {{ job.error }}</span></template>
          </p>
          <details v-if="job.files?.length" class="files">
            <summary>Files ({{ job.files.length }})</summary>
            <ul class="plain">
              <li v-for="f in job.files" :key="f.name">
                <span class="mono">{{ f.name }}</span>
                <span class="num">{{ f.total ? `${Math.floor((100 * f.done) / f.total)}% of ${fmtBytes(f.total)}` : fmtBytes(f.done) }}</span>
              </li>
            </ul>
          </details>
        </template>
        <p v-else-if="plan && !needsDownload" class="hint">Already downloaded.</p>
        <p v-if="jobError" class="error" role="alert">{{ jobError }}</p>
        <div class="actions">
          <button v-if="downloading" type="button" class="quiet" @click="cancel">Cancel the download</button>
          <button v-else-if="needsDownload || job?.status === 'cancelled' || job?.status === 'error'" type="button" class="primary" :disabled="!plan || (plan.free !== null && !plan.fits)" @click="download">
            {{ job?.status === 'cancelled' || job?.status === 'error' ? 'Resume the download' : 'Download' }}
          </button>
        </div>
      </li>

      <li :class="{ done: entry && (entry.state === 'ready' || entry.state === 'busy') }">
        <h3>Load</h3>
        <fieldset v-if="variants.length > 1" class="choices">
          <legend class="sr-only">Precision</legend>
          <label v-for="(v, i) in variants" :key="v.label" class="choice">
            <input v-model="variant" type="radio" :value="i" />
            <span>
              <strong>{{ v.label === 'bf16' ? 'As stored (bf16)' : v.label.replace('--quant ', '') + ' (quantized at load)' }}</strong>
              <small>{{ v.memory_gb.toFixed(1) }} GB in memory, {{ v.runs_on === 'streamed' ? 'streamed from disk' : `on the ${v.runs_on === 'gpu' ? 'GPU' : v.runs_on === 'cpu' ? 'CPU' : 'machine'}` }}{{ v.decode_tok_s ? `, ~${v.decode_tok_s.toFixed(0)} tokens/s` : '' }}{{ v.recommended ? ' — recommended' : '' }}</small>
            </span>
          </label>
        </fieldset>
        <label v-else-if="!model && guessChat && !spec.includes(':') && !/\.gguf$/i.test(spec)" class="inline">
          Precision (chat models)
          <select v-model="quant">
            <option value="">As stored</option>
            <option value="q8">8-bit (q8)</option>
            <option value="q4">4-bit (q4)</option>
          </select>
        </label>
        <details class="more">
          <summary>Name and options</summary>
          <label class="block">Name clients use (default: the spec) <input v-model="alias" class="field" type="text" :placeholder="spec" spellcheck="false" /></label>
          <label class="block">Model options (KEY=VALUE, comma separated)<input v-model="extra" class="field" type="text" placeholder="think=1, max_context=8192, device=cpu" spellcheck="false" /></label>
          <p class="hint">The options of <code>oxinfer serve --model</code>: <code>kind</code>, <code>system</code>, <code>think</code>, <code>max_tokens</code>, the engine’s (<code>device</code>, <code>mmap</code>, <code>mtp</code>, …) or the loader’s.</p>
        </details>
        <p v-if="entry" :class="['status', entry.state === 'ready' || entry.state === 'busy' ? 'ready' : entry.state === 'error' ? 'error' : '']" role="status">{{ loadLine(entry) }}</p>
        <p v-if="loadError && entry?.state !== 'error'" class="error" role="alert">{{ loadError }}</p>
        <div class="actions">
          <button v-if="!entry || entry.state === 'error' || entry.state === 'unloaded' || entry.state === 'cancelled'" type="button" class="primary" :disabled="downloading || (!plan && !planError)" @click="load">
            {{ needsDownload ? 'Download and load' : 'Load' }}
          </button>
          <button v-if="entry && (entry.state === 'ready' || entry.state === 'busy') && entry.kind" type="button" class="primary" @click="emit('use', entry.kind)">Use it</button>
          <span v-if="loadMoving" class="hint">You can close this: it keeps loading.</span>
        </div>
      </li>
    </ol>
  </dialog>
</template>

<style scoped>
.setup {
  width: min(44rem, calc(100vw - 2rem));
}
.setup-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 1rem;
}
.setup-head h2 {
  margin: 0;
  font-size: 1.05rem;
  overflow-wrap: anywhere;
}
.steps {
  margin: 1rem 0 0;
  padding: 0;
  list-style: none;
  counter-reset: step;
}
.steps > li {
  position: relative;
  padding: 0 0 1.1rem 2.2rem;
  counter-increment: step;
}
.steps > li::before {
  content: counter(step);
  position: absolute;
  left: 0;
  top: 0.05rem;
  width: 1.5rem;
  height: 1.5rem;
  border-radius: 50%;
  border: 1px solid var(--line);
  display: grid;
  place-items: center;
  font-family: var(--display);
  font-size: 0.85rem;
  font-weight: 650;
}
.steps > li.done::before {
  content: '✓';
  background: var(--verdigris-deep);
  border-color: var(--verdigris-deep);
  color: var(--paper);
}
.steps h3 {
  margin: 0 0 0.3rem;
  font-family: var(--display);
  font-size: 1rem;
  font-weight: 650;
}
.steps p {
  margin: 0.3rem 0 0;
}
.files ul {
  max-height: 12rem;
  overflow: auto;
  font-size: 0.82rem;
}
.files li {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
}
.files .num {
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  align-items: center;
}
.actions .primary,
.actions .quiet {
  margin-top: 0.6rem;
}
.actions .hint {
  margin-top: 0.6rem;
}
</style>
