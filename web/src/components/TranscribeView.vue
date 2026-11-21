<script setup>
// Speech to text on the native backend (/v1/audio/transcriptions and /translations): a file or a
// recording from the microphone, in, the text (or subtitles) out.
import { computed, onUnmounted, ref } from 'vue';
import { store } from '../lib/native.js';
import { startRecording } from '../lib/audio.js';
import ModelPicker from './ModelPicker.vue';

const emit = defineEmits(['manage']);

const model = computed({ get: () => store.chosen.transcription || '', set: (v) => (store.chosen.transcription = v) });
const task = ref('transcribe');
const language = ref('');
const format = ref('verbose_json');
const words = ref(false);
const prompt = ref('');
const temperature = ref(0);
const beam = ref('');
const audio = ref(null); // { blob, url, name }
const recording = ref(null);
const recSecs = ref(0);
const working = ref(false);
const error = ref('');
const result = ref(null); // { text, segments, raw, secs, format }
let recTimer = null;

function pickFile(e) {
  const f = e.target.files?.[0];
  if (!f) return;
  setAudio(f, f.name);
  e.target.value = '';
}

function setAudio(blob, name) {
  if (audio.value) URL.revokeObjectURL(audio.value.url);
  audio.value = { blob, name, url: URL.createObjectURL(blob) };
  result.value = null;
}

async function record() {
  error.value = '';
  try {
    recording.value = await startRecording();
    recSecs.value = 0;
    const t0 = Date.now();
    recTimer = setInterval(() => (recSecs.value = (Date.now() - t0) / 1e3), 200);
  } catch (e) {
    error.value = e.message || String(e);
  }
}

async function stopRecording() {
  clearInterval(recTimer);
  const r = recording.value;
  recording.value = null;
  try {
    setAudio(await r.stop(), `recording-${new Date().toISOString().slice(11, 19).replace(/:/g, '')}.wav`);
  } catch (e) {
    error.value = `The recording could not be read (${e.message || e}).`;
  }
}

async function run() {
  if (!audio.value || !model.value) return;
  error.value = '';
  working.value = true;
  const t0 = performance.now();
  try {
    const f = new FormData();
    f.append('model', model.value);
    f.append('file', audio.value.blob, audio.value.name);
    f.append('response_format', format.value);
    if (language.value.trim() && task.value === 'transcribe') f.append('language', language.value.trim());
    if (prompt.value.trim()) f.append('prompt', prompt.value.trim());
    if (temperature.value) f.append('temperature', String(temperature.value));
    if (beam.value) f.append('beam_size', String(beam.value));
    if (format.value === 'verbose_json') {
      f.append('timestamp_granularities[]', 'segment');
      if (words.value) f.append('timestamp_granularities[]', 'word');
    }
    const r = await store.api.raw(`v1/audio/${task.value === 'translate' ? 'translations' : 'transcriptions'}`, { method: 'POST', body: f });
    const secs = (performance.now() - t0) / 1e3;
    if (format.value === 'json' || format.value === 'verbose_json') {
      const v = await r.json();
      result.value = { text: v.text, segments: v.segments || [], words: v.words || [], language: v.language, duration: v.duration, secs, format: format.value };
    } else {
      result.value = { text: await r.text(), segments: [], words: [], secs, format: format.value };
    }
  } catch (e) {
    error.value = e.message;
  }
  working.value = false;
}

function ts(t) {
  const m = Math.floor(t / 60);
  return `${m}:${(t - m * 60).toFixed(1).padStart(4, '0')}`;
}

async function copy() {
  try {
    await navigator.clipboard.writeText(result.value.text);
  } catch {
    /* not allowed */
  }
}

function download() {
  const ext = { srt: 'srt', vtt: 'vtt' }[result.value.format] || 'txt';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([result.value.text], { type: 'text/plain' }));
  a.download = `${(audio.value?.name || 'transcript').replace(/\.[^.]+$/, '')}.${ext}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

onUnmounted(() => {
  clearInterval(recTimer);
  recording.value?.cancel();
});
</script>

<template>
  <main class="layout">
    <aside class="side">
      <ModelPicker v-model="model" kind="transcription" title="Speech-to-text model" @manage="emit('manage')" />
      <section aria-labelledby="stt-set-h">
        <h2 id="stt-set-h">Settings</h2>
        <label class="inline">Task
          <select v-model="task">
            <option value="transcribe">Write down what is said</option>
            <option value="translate">Translate it to English</option>
          </select>
        </label>
        <label v-if="task === 'transcribe'" class="inline">Language <input v-model="language" class="num" type="text" placeholder="auto" spellcheck="false" aria-describedby="lang-hint" /></label>
        <p v-if="task === 'transcribe'" id="lang-hint" class="hint">A code such as en, fr, ja; empty to detect it.</p>
        <label class="inline">Output
          <select v-model="format">
            <option value="verbose_json">Text with timings</option>
            <option value="text">Plain text</option>
            <option value="srt">SRT subtitles</option>
            <option value="vtt">WebVTT subtitles</option>
          </select>
        </label>
        <label v-if="format === 'verbose_json'" class="inline"><input v-model="words" type="checkbox" /> Word timings</label>
        <label class="inline">Temperature <input v-model.number="temperature" class="num" type="number" min="0" max="1" step="0.1" /></label>
        <label class="inline">Beam size <input v-model.number="beam" class="num" type="number" min="1" max="10" placeholder="1" /></label>
        <label class="block">Prompt (names, spelling, style)
          <input v-model="prompt" class="field" type="text" spellcheck="false" />
        </label>
      </section>
    </aside>

    <section class="pane" aria-label="Speech to text">
      <div class="stack">
        <h2>Audio</h2>
        <div class="row-actions wrap">
          <label class="quiet as-button">Choose a file…<input class="sr-only" type="file" accept="audio/*,video/*" @change="pickFile" /></label>
          <button v-if="!recording" type="button" class="quiet" @click="record">Record</button>
          <button v-else type="button" class="primary rec" @click="stopRecording">Stop recording ({{ recSecs.toFixed(0) }} s)</button>
        </div>
        <p v-if="!audio && !recording" class="empty">Choose an audio or video file, or record from the microphone.</p>
        <template v-if="audio">
          <p class="hint">{{ audio.name }}</p>
          <audio :src="audio.url" controls class="player"></audio>
        </template>
        <div>
          <button type="button" class="primary" :disabled="!audio || !model || working || !!recording" @click="run">{{ working ? 'Working…' : task === 'translate' ? 'Translate' : 'Transcribe' }}</button>
        </div>
        <p v-if="!model" class="hint">Set up a speech-to-text model first, such as openai/whisper-small.</p>
        <p v-if="error" class="error" role="alert">{{ error }}</p>
        <p v-if="working" class="working" role="status">Listening… long recordings take a while.</p>

        <section v-if="result" aria-labelledby="stt-out-h">
          <h2 id="stt-out-h">Text</h2>
          <p class="hint">
            {{ result.secs.toFixed(1) }} s<template v-if="result.duration"> for {{ result.duration.toFixed(1) }} s of audio</template><template v-if="result.language">, language {{ result.language }}</template>.
            <button type="button" class="link" @click="copy">Copy</button> ·
            <button type="button" class="link" @click="download">Download</button>
          </p>
          <ol v-if="result.segments.length" class="segments">
            <li v-for="(s, i) in result.segments" :key="i">
              <span class="t">{{ ts(s.start) }}–{{ ts(s.end) }}</span>
              <span>{{ s.text.trim() }}</span>
            </li>
          </ol>
          <pre v-else class="out">{{ result.text }}</pre>
          <details v-if="result.words.length" class="more">
            <summary>Words ({{ result.words.length }})</summary>
            <p class="words"><span v-for="(w, i) in result.words" :key="i" :title="`${ts(w.start)}–${ts(w.end)}`">{{ w.word }} </span></p>
          </details>
        </section>
      </div>
    </section>
  </main>
</template>

<style scoped>
.segments {
  margin: 0.5rem 0 0;
  padding: 0;
  list-style: none;
  max-width: 72ch;
}
.segments li {
  display: grid;
  grid-template-columns: 7.5rem 1fr;
  gap: 0.75rem;
  padding: 0.25rem 0;
  border-bottom: 1px solid var(--line);
}
.t {
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  font-size: 0.88rem;
}
.rec {
  background: var(--error);
}
.words {
  max-width: 72ch;
}
</style>
