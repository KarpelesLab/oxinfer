<script setup>
// Video generation on the native backend (MiniMax-H3, /v1/videos): a job per request, followed
// while it waits for the model, loads it and denoises; once it is done its MP4 (H.264 and AAC)
// plays in a <video> element and, with the soundtrack (the WAV variant), can be saved.
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { store, fmtBytes, fmtSecs } from '../lib/native.js';
import { readSse } from '../lib/api.js';
import ModelPicker from './ModelPicker.vue';

const emit = defineEmits(['manage']);

const model = computed({ get: () => store.chosen.video || '', set: (v) => (store.chosen.video = v) });
const prompt = ref('');
const seconds = ref('5');
const size = ref('');
const steps = ref('');
const seed = ref('');
const format = ref('mp4');
const first = ref(null);
const last = ref(null);
const refs = ref([]);
const working = ref(false);
const error = ref('');
const jobs = ref([]);
const selected = ref('');
let timer = null;
const watching = new Map(); // id -> AbortController

const job = computed(() => jobs.value.find((j) => j.id === selected.value));

const STAGE = {
  queued: 'Queued',
  waiting: 'Waiting for the model to be free',
  downloading: 'Downloading the model',
  loading: 'Loading the model',
  generating: 'Encoding the prompt and denoising',
  decoding: 'Decoding the frames and the soundtrack',
  encoding: 'Writing the video',
  cancelling: 'Cancelling',
  completed: 'Done',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

function stageLine(j) {
  let s = STAGE[j.stage] || j.stage;
  if (j.stage === 'generating' && j.steps) s = `Denoising: step ${j.step} of ${j.steps}`;
  if (j.stage === 'downloading' && j.download) s += `: ${j.download.file} ${fmtBytes(j.download.done)}${j.download.total ? ` of ${fmtBytes(j.download.total)}` : ''}`;
  if (j.elapsed) s += ` · ${fmtSecs(j.elapsed)}`;
  return s;
}

async function refreshJobs() {
  try {
    const r = await store.api.get('v1/videos?limit=50');
    // Keep the live copies of the jobs being followed.
    const live = new Map(jobs.value.map((j) => [j.id, j]));
    jobs.value = r.data.map((j) => (watching.has(j.id) && live.get(j.id)?.progress >= j.progress ? live.get(j.id) : j));
    for (const j of jobs.value) if (['queued', 'in_progress'].includes(j.status)) follow(j.id);
    if (!selected.value && jobs.value.length) selected.value = jobs.value[0].id;
  } catch (e) {
    error.value = e.message;
  }
}

/// Follows a job's events until it ends.
async function follow(id) {
  if (watching.has(id)) return;
  const ctl = new AbortController();
  watching.set(id, ctl);
  try {
    const r = await store.api.raw(`v1/videos/${encodeURIComponent(id)}/events`, { signal: ctl.signal });
    await readSse(r, (event, data) => {
      if (event !== 'video') return;
      const v = JSON.parse(data);
      const i = jobs.value.findIndex((j) => j.id === id);
      if (i >= 0) jobs.value[i] = v;
      else jobs.value.unshift(v);
    });
  } catch {
    /* closed */
  }
  watching.delete(id);
}

function form() {
  const f = new FormData();
  f.append('model', model.value);
  f.append('prompt', prompt.value.trim());
  if (seconds.value) f.append('seconds', seconds.value);
  if (size.value.trim()) f.append('size', size.value.trim());
  if (steps.value) f.append('steps', steps.value);
  if (seed.value) f.append('seed', seed.value);
  f.append('response_format', format.value);
  if (first.value) f.append('first_frame', first.value, first.value.name);
  if (last.value) f.append('last_frame', last.value, last.value.name);
  for (const r of refs.value) f.append('references[]', r, r.name);
  return f;
}

async function generate() {
  if (!model.value || !prompt.value.trim()) return;
  error.value = '';
  working.value = true;
  try {
    const r = await (await store.api.raw('v1/videos', { method: 'POST', body: form() })).json();
    jobs.value.unshift(r);
    selected.value = r.id;
    follow(r.id);
  } catch (e) {
    error.value = e.message;
  }
  working.value = false;
}

async function cancel(j) {
  try {
    await store.api.post(`v1/videos/${encodeURIComponent(j.id)}/cancel`, {});
  } catch (e) {
    error.value = e.message;
  }
}

async function remove(j) {
  try {
    await store.api.raw(`v1/videos/${encodeURIComponent(j.id)}`, { method: 'DELETE' });
    watching.get(j.id)?.abort();
    jobs.value = jobs.value.filter((x) => x.id !== j.id);
    if (selected.value === j.id) selected.value = jobs.value[0]?.id || '';
  } catch (e) {
    error.value = e.message;
  }
}

// ---- the player ------------------------------------------------------------------------------
// The video file (MP4: H.264 and AAC, by default) in a <video> element with its own soundtrack.
const player = ref(null); // { id, url, type, name, wavUrl, bytes }
const loadingPlayer = ref(false);
const EXT = { 'video/mp4': 'mp4', 'video/webm': 'webm', 'video/x-msvideo': 'avi' };

async function openPlayer(j) {
  if (player.value?.id === j.id || j.status !== 'completed') return;
  closePlayer();
  loadingPlayer.value = true;
  try {
    const base = `v1/videos/${encodeURIComponent(j.id)}/content?variant=`;
    const blob = await (await store.api.raw(`${base}video`)).blob();
    const type = blob.type || 'video/mp4';
    let wavUrl = '';
    if (j.has_audio) wavUrl = URL.createObjectURL(await (await store.api.raw(`${base}audio`)).blob());
    player.value = { id: j.id, url: URL.createObjectURL(blob), type, name: `oxinfer-video.${EXT[type] || 'mp4'}`, wavUrl, bytes: blob.size };
  } catch (e) {
    error.value = `The video could not be opened (${e.message}).`;
  }
  loadingPlayer.value = false;
}

function closePlayer() {
  const p = player.value;
  if (!p) return;
  URL.revokeObjectURL(p.url);
  if (p.wavUrl) URL.revokeObjectURL(p.wavUrl);
  player.value = null;
}

watch(job, (j) => {
  if (j?.status === 'completed' && player.value?.id !== j.id) openPlayer(j);
  if (j && j.status !== 'completed' && player.value) closePlayer();
});

onMounted(() => {
  refreshJobs();
  timer = setInterval(refreshJobs, 5000);
});
onUnmounted(() => {
  clearInterval(timer);
  for (const c of watching.values()) c.abort();
  closePlayer();
});
</script>

<template>
  <main class="layout">
    <aside class="side">
      <ModelPicker v-model="model" kind="video" title="Video model" @manage="emit('manage')" />
      <section aria-labelledby="video-set-h">
        <h2 id="video-set-h">Settings</h2>
        <label class="inline">Seconds <input v-model="seconds" class="num" type="number" min="0.5" max="15" step="0.5" /></label>
        <label class="inline">Size <input v-model="size" class="num wide" type="text" placeholder="1344x768" spellcheck="false" /></label>
        <label class="inline">Steps <input v-model="steps" class="num" type="number" min="2" max="200" placeholder="50" /></label>
        <label class="inline">File
          <select v-model="format" aria-label="Video file format">
            <option value="mp4">MP4 (H.264 + AAC)</option>
            <option value="webm">WebM (VP9 + Opus)</option>
            <option value="avi">AVI (Motion JPEG + PCM)</option>
          </select>
        </label>
        <label class="inline">Seed <input v-model="seed" class="num" type="number" min="0" placeholder="random" /></label>
        <p class="hint">At 24 frames a second, rounded up to 17 n + 5 frames; the size in multiples of 32. MiniMax-H3 is made for 5 to 15 seconds.</p>
        <label class="block">First frame <input type="file" accept="image/*" @change="first = $event.target.files[0] || null" /></label>
        <label class="block">Last frame <input type="file" accept="image/*" @change="last = $event.target.files[0] || null" /></label>
        <label class="block">References (Ref2VA) <input type="file" accept="image/*,audio/*,video/*" multiple @change="refs = [...$event.target.files]" /></label>
        <p class="hint">References are for a model set up from the release's Ref2VA/ folder, in the order the prompt names them.</p>
      </section>
    </aside>

    <section class="pane" aria-label="Video">
      <div class="stack">
        <h2>Prompt</h2>
        <textarea v-model="prompt" class="field big" rows="4" aria-label="Prompt" placeholder="A red fox trots across a snowy clearing at dawn, its breath steaming…"></textarea>
        <div>
          <button type="button" class="primary" :disabled="!model || !prompt.trim() || working" data-testid="video-generate" @click="generate">{{ working ? 'Starting…' : 'Generate video' }}</button>
        </div>
        <p v-if="!model" class="hint">Set up a video model first (MiniMax-H3, a release folder).</p>
        <p v-if="error" class="error" role="alert">{{ error }}</p>

        <div v-if="job" class="job" data-testid="video-job">
          <p class="title"><strong>{{ job.prompt }}</strong></p>
          <p class="hint">{{ job.model }} · {{ job.size || 'auto size' }} · {{ job.seconds }} s{{ job.frames ? ` (${job.frames} frames)` : '' }}</p>
          <template v-if="['queued', 'in_progress'].includes(job.status)">
            <div class="bar" role="progressbar" :aria-valuenow="job.progress" aria-valuemin="0" aria-valuemax="100"><div class="fill" :style="{ width: `${job.progress}%` }"></div></div>
            <p class="status" role="status" data-testid="video-stage">{{ job.progress }}% · {{ stageLine(job) }}</p>
            <button type="button" class="quiet" @click="cancel(job)">Cancel</button>
          </template>
          <p v-else-if="job.status !== 'completed'" :class="job.status === 'failed' ? 'error' : 'status'" role="status">{{ STAGE[job.status] }}{{ job.error ? `: ${job.error.message}` : '' }}</p>
          <template v-else>
            <p v-if="loadingPlayer" class="working" role="status">Opening the video…</p>
            <div v-if="player" class="player-box">
              <video :src="player.url" controls playsinline class="screen" data-testid="video-player"></video>
              <p class="hint">
                {{ job.size }}, {{ job.frames }} frames{{ job.has_audio ? ', with sound' : ', no soundtrack' }} · {{ (player.bytes / 1e6).toFixed(1) }} MB ·
                <a :href="player.url" :download="player.name">Download {{ player.name.split('.').pop().toUpperCase() }}</a>
                <template v-if="player.wavUrl"> · <a :href="player.wavUrl" download="oxinfer-video.wav">Download WAV</a></template>
              </p>
            </div>
          </template>
        </div>

        <template v-if="jobs.length">
          <h2>Videos</h2>
          <ul class="plain list">
            <li v-for="j in jobs" :key="j.id" :class="{ on: j.id === selected }">
              <button type="button" class="link" @click="selected = j.id">{{ j.prompt.length > 70 ? `${j.prompt.slice(0, 70)}…` : j.prompt }}</button>
              <span class="hint"> · {{ j.status === 'in_progress' || j.status === 'queued' ? `${j.progress}%` : STAGE[j.status] }}</span>
              <button type="button" class="link" @click="remove(j)">Delete</button>
            </li>
          </ul>
        </template>
        <p v-else class="empty">Write a prompt, then press Generate video. Generating takes a long time; you can leave the page and come back.</p>
      </div>
    </section>
  </main>
</template>

<style scoped>
.big {
  max-width: 68ch;
  resize: vertical;
}
.wide {
  width: 7rem;
}
.job {
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--line);
  max-width: 72ch;
}
.title {
  margin: 0 0 0.2rem;
}
.screen {
  display: block;
  /* Small videos scaled up for a look; large ones fit the column. */
  width: min(100%, 640px);
  height: auto;
  background: #000;
  border-radius: 4px;
}
.list li {
  padding: 0.25rem 0;
}
.list li.on > .link:first-child {
  font-weight: 700;
}
.list .link + .hint + .link {
  margin-left: 0.75rem;
}
</style>
