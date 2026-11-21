<script setup>
// The model a native view uses, for one kind (chat, image, transcription, speech, embedding): one
// of the server's models, or a downloaded one loaded here. Setting models up (the catalog,
// downloads, the cache) is the Models view (`manage`).
import { computed, onMounted, ref, watch } from 'vue';
import { store, modelsOf, loadLine, refresh, fmtBytes } from '../lib/native.js';

const props = defineProps({
  kind: { type: String, required: true },
  title: { type: String, default: 'Model' },
  modelValue: { type: String, default: '' },
});
const emit = defineEmits(['update:modelValue', 'manage']);

const KIND_WORD = { chat: 'chat', image: 'image', transcription: 'speech-to-text', speech: 'text-to-speech', embedding: 'embedding' };
const word = computed(() => KIND_WORD[props.kind] || props.kind);
const models = computed(() => modelsOf(props.kind));
const current = computed(() => models.value.find((m) => m.id === props.modelValue));
const cached = ref([]);
const pick = ref('');
const quant = ref('');
const error = ref('');
const working = ref(false);

// Downloaded models of this kind that are not listed yet.
const downloaded = computed(() => {
  const listed = new Set((store.status?.models || []).map((m) => m.source));
  const out = [];
  for (const e of cached.value) {
    if (e.kind !== props.kind || !e.usable) continue;
    for (const s of e.specs) if (!listed.has(s)) out.push({ spec: s, bytes: e.bytes, name: e.name });
  }
  return out;
});

async function loadCache() {
  try {
    cached.value = (await store.api.cache()).entries;
  } catch {
    cached.value = [];
  }
}

// Keep a valid choice: the current one, else a loaded model, else the first.
watch(
  models,
  (list) => {
    if (list.some((m) => m.id === props.modelValue)) return;
    const m = list.find((x) => x.loaded) || list[0];
    if (m) emit('update:modelValue', m.id);
  },
  { immediate: true },
);

async function loadSpec() {
  if (!pick.value) return;
  error.value = '';
  working.value = true;
  try {
    const options = quant.value ? { quant: quant.value } : {};
    const r = await store.api.load(pick.value, { options });
    emit('update:modelValue', r.id);
    pick.value = '';
    await refresh();
    await loadCache();
  } catch (e) {
    error.value = e.message;
  }
  working.value = false;
}

async function loadCurrent() {
  if (!current.value) return;
  error.value = '';
  try {
    await store.api.load(current.value.source, { id: current.value.id, options: Object.fromEntries(current.value.options.map((o) => o.split('='))) });
    await refresh();
  } catch (e) {
    error.value = e.message;
  }
}

async function unload() {
  error.value = '';
  try {
    await store.api.unload(props.modelValue);
    await refresh();
  } catch (e) {
    error.value = e.message;
  }
}

const moving = computed(() => ['queued', 'waiting', 'downloading', 'loading'].includes(current.value?.state));
const fraction = computed(() => {
  const l = current.value?.load;
  return l?.total ? l.done / l.total : 0;
});

onMounted(loadCache);
</script>

<template>
  <section :aria-labelledby="`pick-${kind}`">
    <h2 :id="`pick-${kind}`">{{ title }}</h2>
    <template v-if="models.length">
      <label class="block">
        <span class="sr-only">{{ title }}</span>
        <select class="field" :value="modelValue" @change="emit('update:modelValue', $event.target.value)">
          <option v-for="m in models" :key="m.id" :value="m.id">{{ m.id }}{{ m.loaded ? ' (loaded)' : '' }}</option>
        </select>
      </label>
      <p v-if="current" :class="['status', current.state === 'ready' || current.state === 'busy' ? 'ready' : current.state === 'error' ? 'error' : '']" role="status">
        {{ loadLine(current) }}<template v-if="current.loaded && current.memory">, {{ fmtBytes(current.memory) }} of memory</template>.
      </p>
      <div v-if="moving && current.load?.total" class="bar" aria-hidden="true"><div class="fill" :style="{ width: `${(fraction * 100).toFixed(1)}%` }"></div></div>
      <div class="actions">
        <button v-if="current && !current.loaded && !moving" type="button" class="quiet" @click="loadCurrent">Load now</button>
        <button v-if="current && (current.loaded || moving)" type="button" class="quiet" :disabled="current.busy && !moving" @click="unload">{{ moving ? 'Cancel' : 'Unload' }}</button>
      </div>
    </template>
    <p v-else class="hint">No {{ word }} model is set up yet. Load one you have downloaded below, or pick one from the catalog.</p>

    <template v-if="downloaded.length">
      <label class="block">Load a downloaded model
        <select v-model="pick" class="field">
          <option value="">Choose…</option>
          <option v-for="d in downloaded" :key="d.spec" :value="d.spec">{{ d.spec.length > 60 ? d.name : d.spec }} ({{ fmtBytes(d.bytes) }})</option>
        </select>
      </label>
      <div v-if="pick && kind === 'chat' && !pick.includes(':') && !/gguf$/i.test(pick)" class="inline">
        Precision
        <select v-model="quant" aria-label="Precision">
          <option value="">As stored</option>
          <option value="q8">8-bit (q8)</option>
          <option value="q4">4-bit (q4)</option>
        </select>
      </div>
      <button type="button" class="primary" :disabled="!pick || working" @click="loadSpec">Load</button>
    </template>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <p class="hint"><button type="button" class="link" @click="emit('manage')">Browse the catalog and set up models</button></p>
  </section>
</template>

<style scoped>
.actions {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.actions .quiet {
  margin-top: 0.6rem;
}
</style>
