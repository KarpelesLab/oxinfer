<script setup>
// Text to speech on the native backend (/v1/audio/speech): text in, a WAV file out, played here
// and saved on demand. Each clip stays in the list until the page is closed.
import { computed, onUnmounted, ref } from 'vue';
import { store } from '../lib/native.js';
import ModelPicker from './ModelPicker.vue';

const emit = defineEmits(['manage']);

// Orpheus' voices; the other models name theirs in the error when one is wrong.
const VOICES = ['tara', 'leah', 'jess', 'leo', 'dan', 'mia', 'zac', 'zoe'];

const model = computed({ get: () => store.chosen.speech || '', set: (v) => (store.chosen.speech = v) });
const text = ref('Hello! This voice was made on your own computer.');
const voice = ref('');
const temperature = ref('');
const topP = ref('');
const repetition = ref('');
const seed = ref('');
const working = ref(false);
const error = ref('');
const clips = ref([]); // { url, text, voice, secs, bytes }

async function speak() {
  const input = text.value.trim();
  if (!input || !model.value) return;
  error.value = '';
  working.value = true;
  const t0 = performance.now();
  const num = (k, v) => (v === '' || v === null || Number.isNaN(Number(v)) ? {} : { [k]: Number(v) });
  try {
    const r = await store.api.raw('v1/audio/speech', {
      method: 'POST',
      json: {
        model: model.value,
        input,
        response_format: 'wav',
        ...(voice.value.trim() ? { voice: voice.value.trim() } : {}),
        ...num('temperature', temperature.value),
        ...num('top_p', topP.value),
        ...num('repetition_penalty', repetition.value),
        ...num('seed', seed.value),
      },
    });
    const blob = await r.blob();
    clips.value.unshift({ url: URL.createObjectURL(blob), text: input, voice: voice.value.trim() || 'default', secs: (performance.now() - t0) / 1e3, bytes: blob.size });
  } catch (e) {
    error.value = e.message;
  }
  working.value = false;
}

function name(c, i) {
  return `oxinfer-speech-${clips.value.length - i}.wav`;
}

function onKey(e) {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    speak();
  }
}

onUnmounted(() => clips.value.forEach((c) => URL.revokeObjectURL(c.url)));
</script>

<template>
  <main class="layout">
    <aside class="side">
      <ModelPicker v-model="model" kind="speech" title="Text-to-speech model" @manage="emit('manage')" />
      <section aria-labelledby="tts-set-h">
        <h2 id="tts-set-h">Voice and sampling</h2>
        <label class="block">Voice
          <input v-model="voice" class="field" type="text" list="voices" placeholder="the model’s default" spellcheck="false" />
        </label>
        <datalist id="voices"><option v-for="v in VOICES" :key="v" :value="v"></option></datalist>
        <p class="hint">Orpheus: tara, leah, jess, leo, dan, mia, zac, zoe. Other models list theirs if the name is wrong.</p>
        <label class="inline">Temperature <input v-model="temperature" class="num" type="number" min="0" max="2" step="0.1" placeholder="default" /></label>
        <label class="inline">Top-p <input v-model="topP" class="num" type="number" min="0.05" max="1" step="0.05" placeholder="default" /></label>
        <label class="inline">Repetition penalty <input v-model="repetition" class="num" type="number" min="1" max="2" step="0.05" placeholder="default" /></label>
        <label class="inline">Seed <input v-model="seed" class="num" type="number" min="0" placeholder="random" /></label>
      </section>
    </aside>

    <section class="pane" aria-label="Text to speech">
      <div class="stack">
        <h2>Text</h2>
        <textarea v-model="text" class="field big" rows="5" aria-label="Text to say" @keydown="onKey"></textarea>
        <div>
          <button type="button" class="primary" :disabled="!model || !text.trim() || working" @click="speak">{{ working ? 'Speaking…' : 'Speak' }}</button>
          <span class="hint key-hint">Ctrl+Enter</span>
        </div>
        <p v-if="!model" class="hint">Set up a text-to-speech model first, such as an Orpheus GGUF.</p>
        <p v-if="error" class="error" role="alert">{{ error }}</p>
        <p v-if="working" class="working" role="status">Generating the audio…</p>

        <p v-if="!clips.length && !working" class="empty">Write something, then press Speak.</p>
        <ul class="plain clips">
          <li v-for="(c, i) in clips" :key="c.url" class="clip">
            <p class="said">{{ c.text }}</p>
            <audio :src="c.url" controls class="player" :autoplay="i === 0"></audio>
            <p class="hint">
              {{ c.voice }} · {{ c.secs.toFixed(1) }} s to make ·
              <a :href="c.url" :download="name(c, i)">Download WAV</a>
            </p>
          </li>
        </ul>
      </div>
    </section>
  </main>
</template>

<style scoped>
.big {
  max-width: 68ch;
  resize: vertical;
}
.clip {
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--line);
  max-width: 68ch;
}
.said {
  margin: 0 0 0.4rem;
  padding-left: 0.8rem;
  border-left: 3px solid var(--verdigris);
  white-space: pre-wrap;
}
.key-hint {
  margin-left: 0.75rem;
}
</style>
