<script setup>
// Embeddings on the native backend (/v1/embeddings): a query against a few texts, ranked by
// cosine similarity, to see what the model considers close.
import { computed, ref } from 'vue';
import { store } from '../lib/native.js';
import ModelPicker from './ModelPicker.vue';

const emit = defineEmits(['manage']);

const model = computed({ get: () => store.chosen.embedding || '', set: (v) => (store.chosen.embedding = v) });
const query = ref('How do I keep my bread fresh?');
const docs = ref(
  [
    'Store bread in a cloth bag at room temperature, or freeze slices for later.',
    'The stock market closed higher on Friday after a week of losses.',
    'A sourdough loaf stays good for three days wrapped in a towel.',
    'Water boils at 100 degrees Celsius at sea level.',
    'Refrigerating bread makes it go stale faster.',
  ].join('\n'),
);
const promptName = ref('');
const dimensions = ref('');
const working = ref(false);
const error = ref('');
const result = ref(null); // { ranked: [{ text, score }], dims, secs, tokens }

function cosine(a, b) {
  let d = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / Math.sqrt(na * nb || 1);
}

async function run() {
  const texts = docs.value.split('\n').map((s) => s.trim()).filter(Boolean);
  if (!model.value || !query.value.trim() || !texts.length) return;
  error.value = '';
  working.value = true;
  const t0 = performance.now();
  try {
    const body = { model: model.value, input: [query.value.trim(), ...texts] };
    if (promptName.value.trim()) body.prompt_name = promptName.value.trim();
    if (dimensions.value) body.dimensions = Number(dimensions.value);
    const v = await store.api.post('v1/embeddings', body);
    const vecs = v.data.sort((a, b) => a.index - b.index).map((d) => d.embedding);
    const q = vecs[0];
    const ranked = texts.map((text, i) => ({ text, score: cosine(q, vecs[i + 1]) })).sort((a, b) => b.score - a.score);
    result.value = { ranked, dims: q.length, secs: (performance.now() - t0) / 1e3, tokens: v.usage?.total_tokens };
  } catch (e) {
    error.value = e.message;
  }
  working.value = false;
}

const lo = computed(() => Math.min(0, ...(result.value?.ranked.map((r) => r.score) || [0])));
</script>

<template>
  <main class="layout">
    <aside class="side">
      <ModelPicker v-model="model" kind="embedding" title="Embedding model" @manage="emit('manage')" />
      <section aria-labelledby="emb-set-h">
        <h2 id="emb-set-h">Settings</h2>
        <label class="block">Prompt name
          <input v-model="promptName" class="field" type="text" placeholder="query, document, … (if the model has them)" spellcheck="false" />
        </label>
        <label class="inline">Dimensions <input v-model="dimensions" class="num" type="number" min="1" placeholder="all" /></label>
        <p class="hint">Fewer dimensions for Matryoshka models (nomic, Qwen3-Embedding, …).</p>
      </section>
    </aside>

    <section class="pane" aria-label="Embeddings">
      <div class="stack">
        <h2>Search</h2>
        <label class="block">Query <input v-model="query" class="field wide" type="text" @keydown.enter.prevent="run" /></label>
        <label class="block">Texts to rank, one per line
          <textarea v-model="docs" class="field wide" rows="7" spellcheck="false"></textarea>
        </label>
        <div>
          <button type="button" class="primary" :disabled="!model || working || !query.trim()" @click="run">{{ working ? 'Embedding…' : 'Rank by similarity' }}</button>
        </div>
        <p v-if="!model" class="hint">Set up an embedding model first, such as BAAI/bge-small-en-v1.5.</p>
        <p v-if="error" class="error" role="alert">{{ error }}</p>

        <section v-if="result" aria-labelledby="emb-out-h">
          <h2 id="emb-out-h">Closest first</h2>
          <p class="hint">{{ result.dims }} dimensions, {{ result.ranked.length + 1 }} texts{{ result.tokens ? `, ${result.tokens} tokens` : '' }} in {{ result.secs.toFixed(2) }} s. Cosine similarity to the query:</p>
          <ol class="plain ranked">
            <li v-for="r in result.ranked" :key="r.text">
              <span class="score">{{ r.score.toFixed(3) }}</span>
              <span class="meter" aria-hidden="true"><span :style="{ width: `${(((r.score - lo) / (1 - lo)) * 100).toFixed(1)}%` }"></span></span>
              <span class="text">{{ r.text }}</span>
            </li>
          </ol>
        </section>
      </div>
    </section>
  </main>
</template>

<style scoped>
.wide {
  max-width: 68ch;
}
.ranked li {
  display: grid;
  grid-template-columns: 3.5rem 6rem 1fr;
  gap: 0.75rem;
  align-items: center;
  padding: 0.35rem 0;
  border-bottom: 1px solid var(--line);
  max-width: 72ch;
}
.score {
  font-variant-numeric: tabular-nums;
  font-size: 0.9rem;
}
.meter {
  height: 6px;
  background: var(--line);
  border-radius: 3px;
  overflow: hidden;
}
.meter span {
  display: block;
  height: 100%;
  background: var(--verdigris);
}
@media (max-width: 520px) {
  .ranked li {
    grid-template-columns: 3.5rem 1fr;
  }
  .meter {
    display: none;
  }
}
</style>
