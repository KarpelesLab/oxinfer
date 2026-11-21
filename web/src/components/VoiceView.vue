<script setup>
// Voice chat on the native backend: a recording (or typed text, for LFM3-Omni) in, the model's
// spoken answer out, through /v1/chat/completions with modalities ["text", "audio"]. LFM3-Omni
// answers in text and speech; Haru listens to the recording as a live conversation and answers
// in speech (in the voice of a reference clip, when one is given).
import { computed, onUnmounted, ref } from 'vue';
import { store, modelsOf } from '../lib/native.js';
import { startRecording, toWav } from '../lib/audio.js';
import ModelPicker from './ModelPicker.vue';

const emit = defineEmits(['manage']);

const model = computed({ get: () => store.chosen.voice || '', set: (v) => (store.chosen.voice = v) });
const family = computed(() => modelsOf('voice').find((m) => m.id === model.value)?.audio_out || '');
const text = ref('');
const system = ref('');
const reference = ref(null); // { blob, name }
const maxReply = ref('');
const seed = ref('');
const recording = ref(null);
const recSecs = ref(0);
let recTimer = null;
const pending = ref(null); // { blob, url, name }: the recording to send
const working = ref(false);
const error = ref('');
const turns = ref([]); // { user: {text, url}, reply: {text, url, secs, duration} }

async function record() {
  error.value = '';
  try {
    recording.value = await startRecording();
    const t0 = performance.now();
    recSecs.value = 0;
    recTimer = setInterval(() => (recSecs.value = (performance.now() - t0) / 1e3), 200);
  } catch (e) {
    error.value = e.message || String(e);
  }
}

async function stopRecording() {
  clearInterval(recTimer);
  const r = recording.value;
  recording.value = null;
  try {
    setPending(await r.stop(), 'recording.wav');
  } catch (e) {
    error.value = `The recording could not be read (${e.message || e}).`;
  }
}

function setPending(blob, name) {
  if (pending.value) URL.revokeObjectURL(pending.value.url);
  pending.value = { blob, name, url: URL.createObjectURL(blob) };
}

async function onFile(e) {
  const f = e.target.files?.[0];
  e.target.value = '';
  if (!f) return;
  try {
    setPending(await toWav(f, 24000), f.name);
  } catch (err) {
    error.value = `${f.name} could not be decoded (${err.message || err}).`;
  }
}

async function onReference(e) {
  const f = e.target.files?.[0];
  e.target.value = '';
  if (!f) return;
  try {
    reference.value = { blob: await toWav(f, 24000), name: f.name };
  } catch (err) {
    error.value = `${f.name} could not be decoded (${err.message || err}).`;
  }
}

const b64 = (blob) =>
  new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(',')[1]);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });

const canSend = computed(() => model.value && !working.value && !recording.value && (pending.value || (family.value === 'lfm3_omni' && text.value.trim())));

async function send() {
  if (!canSend.value) return;
  error.value = '';
  working.value = true;
  const t0 = performance.now();
  const content = [];
  if (pending.value) content.push({ type: 'input_audio', input_audio: { data: await b64(pending.value.blob), format: 'wav' } });
  if (text.value.trim()) content.push({ type: 'text', text: text.value.trim() });
  const messages = [];
  if (system.value.trim()) messages.push({ role: 'system', content: system.value.trim() });
  messages.push({ role: 'user', content });
  const audio = { format: 'wav' };
  if (family.value === 'haru' && reference.value) audio.voice_reference = `data:audio/wav;base64,${await b64(reference.value.blob)}`;
  const num = (k, v) => (v === '' || v === null || Number.isNaN(Number(v)) ? {} : { [k]: Number(v) });
  const user = { text: text.value.trim(), url: pending.value?.url || '' };
  try {
    const r = await store.api.post('v1/chat/completions', {
      model: model.value,
      modalities: ['text', 'audio'],
      audio,
      messages,
      ...num('seed', seed.value),
      ...(family.value === 'haru' ? num('max_reply_seconds', maxReply.value) : {}),
    });
    const a = r.choices?.[0]?.message?.audio || {};
    const bytes = Uint8Array.from(atob(a.data || ''), (c) => c.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
    turns.value.unshift({ user, reply: { text: a.transcript || '', url, secs: (performance.now() - t0) / 1e3, duration: a.duration || 0 } });
    pending.value = null; // its URL stays with the turn
    text.value = '';
  } catch (e) {
    error.value = e.message;
  }
  working.value = false;
}

onUnmounted(() => {
  clearInterval(recTimer);
  recording.value?.cancel();
  for (const t of turns.value) {
    if (t.user.url) URL.revokeObjectURL(t.user.url);
    URL.revokeObjectURL(t.reply.url);
  }
});
</script>

<template>
  <main class="layout">
    <aside class="side">
      <ModelPicker v-model="model" kind="voice" title="Voice-chat model" @manage="emit('manage')" />
      <section aria-labelledby="voice-set-h">
        <h2 id="voice-set-h">Settings</h2>
        <label class="block">{{ family === 'haru' ? 'Character card' : 'System prompt' }}
          <textarea v-model="system" class="field" rows="3" :placeholder="family === 'haru' ? 'who the model is (optional)' : 'default: Respond with interleaved text and audio.'"></textarea>
        </label>
        <template v-if="family === 'haru'">
          <label class="block">Voice reference
            <input type="file" accept="audio/*" @change="onReference" />
          </label>
          <p class="hint">{{ reference ? `Speaking as ${reference.name}.` : 'A clip of the voice Haru should speak with (optional).' }}
            <button v-if="reference" type="button" class="link" @click="reference = null">Remove</button></p>
          <label class="inline">Longest answer (s) <input v-model="maxReply" class="num" type="number" min="1" max="60" placeholder="12" /></label>
        </template>
        <label class="inline">Seed <input v-model="seed" class="num" type="number" min="0" placeholder="random" /></label>
      </section>
    </aside>

    <section class="pane" aria-label="Voice chat">
      <div class="stack">
        <h2>Say something</h2>
        <div class="row">
          <button v-if="!recording" type="button" class="quiet" :disabled="working" @click="record">Record</button>
          <button v-else type="button" class="primary rec" @click="stopRecording">Stop recording ({{ recSecs.toFixed(0) }} s)</button>
          <label class="quiet file">Choose a file <input type="file" accept="audio/*,video/*" data-testid="voice-file" @change="onFile" /></label>
        </div>
        <div v-if="pending" class="pending">
          <audio :src="pending.url" controls class="player"></audio>
          <span class="hint">{{ pending.name }} <button type="button" class="link" @click="pending = null">Remove</button></span>
        </div>
        <textarea
          v-if="family !== 'haru'"
          v-model="text"
          class="field big"
          rows="2"
          :placeholder="pending ? 'optional: a question about the recording' : 'or type a question'"
          aria-label="Text"
        ></textarea>
        <div>
          <button type="button" class="primary" :disabled="!canSend" data-testid="voice-send" @click="send">{{ working ? 'Answering…' : 'Send' }}</button>
        </div>
        <p v-if="!model" class="hint">Set up a model that answers in speech first: LiquidAI/LFM3-Omni-2B or higurashi-ai/Haru-4B.</p>
        <p v-if="family === 'haru'" class="hint">Haru hears the recording as a live conversation, then answers in speech.</p>
        <p v-if="error" class="error" role="alert">{{ error }}</p>
        <p v-if="working" class="working" role="status">The model is listening and answering…</p>
        <p v-if="!turns.length && !working" class="empty">Record a question or choose a recording, then press Send.</p>

        <ul class="plain turns">
          <li v-for="(t, i) in turns" :key="t.reply.url" class="turn">
            <div class="you">
              <p class="who">You</p>
              <p v-if="t.user.text" class="said">{{ t.user.text }}</p>
              <audio v-if="t.user.url" :src="t.user.url" controls class="player"></audio>
            </div>
            <div class="model">
              <p class="who">{{ model }}</p>
              <p class="said" data-testid="voice-transcript">{{ t.reply.text || '(no text)' }}</p>
              <audio :src="t.reply.url" controls class="player" :autoplay="i === 0" data-testid="voice-reply"></audio>
              <p class="hint">{{ t.reply.duration.toFixed(1) }} s of audio · {{ t.reply.secs.toFixed(1) }} s to make · <a :href="t.reply.url" :download="`oxinfer-reply-${turns.length - i}.wav`">Download WAV</a></p>
            </div>
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
.row {
  display: flex;
  gap: 0.6rem;
  align-items: center;
  flex-wrap: wrap;
}
.file input {
  display: none;
}
.file {
  cursor: pointer;
  display: inline-block;
  padding: 0.4rem 0.9rem;
  border: 1px solid var(--line);
  border-radius: 4px;
}
.pending {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  flex-wrap: wrap;
}
.turn {
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--line);
  max-width: 68ch;
  display: grid;
  gap: 0.6rem;
}
.who {
  margin: 0;
  font-family: var(--display);
  font-weight: 600;
  font-size: 0.85rem;
  color: var(--muted);
}
.said {
  margin: 0.2rem 0 0.4rem;
  padding-left: 0.8rem;
  border-left: 3px solid var(--verdigris);
  white-space: pre-wrap;
}
.you .said {
  border-left-color: var(--line);
}
</style>
