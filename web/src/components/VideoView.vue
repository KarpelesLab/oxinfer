<script setup>
// Video generation on the native backend (MiniMax-H3, /v1/videos): a job per request, followed
// while it waits for the model, loads it and denoises; once it is done its AVI (Motion JPEG and
// PCM, which browsers do not play) is taken apart here and played on a canvas in time with the
// soundtrack (the WAV variant), and both can be saved.
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue';
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
const canvas = ref(null);
const audioEl = ref(null);
const player = ref(null); // { id, frames: ImageBitmap[], fps, aviUrl, wavUrl, width, height }
const loadingPlayer = ref(false);
const playing = ref(false);
const position = ref(0); // frame index
let raf = 0;
let clock = { t0: 0, from: 0 };

/// The frames (JPEG) of a Motion JPEG AVI: the `00dc` chunks of its `movi` list.
function aviFrames(buf) {
  const b = new Uint8Array(buf);
  const dv = new DataView(buf);
  const tag = (o) => String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);
  const out = [];
  const walk = (start, end) => {
    let o = start;
    while (o + 8 <= end) {
      const id = tag(o);
      const n = dv.getUint32(o + 4, true);
      if (id === 'RIFF' || id === 'LIST') walk(o + 12, Math.min(end, o + 8 + n));
      else if (/^\d\ddc$|^\d\ddb$/.test(id)) out.push(b.subarray(o + 8, o + 8 + n));
      o += 8 + n + (n & 1);
    }
  };
  walk(0, b.length);
  return out;
}

async function openPlayer(j) {
  if (player.value?.id === j.id || j.status !== 'completed') return;
  closePlayer();
  loadingPlayer.value = true;
  try {
    const base = `v1/videos/${encodeURIComponent(j.id)}/content?variant=`;
    const avi = await (await store.api.raw(`${base}video`)).blob();
    const frames = await Promise.all(aviFrames(await avi.arrayBuffer()).map((f) => createImageBitmap(new Blob([f], { type: 'image/jpeg' }))));
    let wavUrl = '';
    if (j.has_audio) wavUrl = URL.createObjectURL(await (await store.api.raw(`${base}audio`)).blob());
    player.value = { id: j.id, frames, fps: j.fps || 24, aviUrl: URL.createObjectURL(avi), wavUrl, width: frames[0]?.width || 0, height: frames[0]?.height || 0 };
    position.value = 0;
    await nextTick();
    draw(0);
  } catch (e) {
    error.value = `The video could not be opened (${e.message}).`;
  }
  loadingPlayer.value = false;
}

function closePlayer() {
  pause();
  const p = player.value;
  if (!p) return;
  p.frames.forEach((f) => f.close?.());
  URL.revokeObjectURL(p.aviUrl);
  if (p.wavUrl) URL.revokeObjectURL(p.wavUrl);
  player.value = null;
}

function draw(i) {
  const p = player.value;
  const c = canvas.value;
  if (!p || !c || !p.frames.length) return;
  const f = p.frames[Math.max(0, Math.min(p.frames.length - 1, i))];
  c.getContext('2d').drawImage(f, 0, 0, c.width, c.height);
}

function now() {
  const a = audioEl.value;
  if (a && player.value?.wavUrl && !a.paused) return a.currentTime;
  return clock.from + (performance.now() - clock.t0) / 1e3;
}

function tick() {
  const p = player.value;
  if (!p || !playing.value) return;
  const i = Math.floor(now() * p.fps);
  if (i >= p.frames.length) {
    pause();
    position.value = p.frames.length - 1;
    draw(position.value);
    return;
  }
  if (i !== position.value) {
    position.value = i;
    draw(i);
  }
  raf = requestAnimationFrame(tick);
}

function play() {
  const p = player.value;
  if (!p) return;
  if (position.value >= p.frames.length - 1) position.value = 0;
  const from = position.value / p.fps;
  clock = { t0: performance.now(), from };
  playing.value = true;
  const a = audioEl.value;
  if (a && p.wavUrl) {
    a.currentTime = from;
    a.play().catch(() => {});
  }
  raf = requestAnimationFrame(tick);
}

function pause() {
  playing.value = false;
  cancelAnimationFrame(raf);
  audioEl.value?.pause();
}

function seek(e) {
  const i = Number(e.target.value);
  const was = playing.value;
  pause();
  position.value = i;
  draw(i);
  if (was) play();
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
              <canvas ref="canvas" :width="player.width" :height="player.height" class="screen" data-testid="video-canvas"></canvas>
              <audio v-if="player.wavUrl" ref="audioEl" :src="player.wavUrl" preload="auto"></audio>
              <div class="controls">
                <button type="button" class="primary" data-testid="video-play" @click="playing ? pause() : play()">{{ playing ? 'Pause' : 'Play' }}</button>
                <input type="range" min="0" :max="player.frames.length - 1" :value="position" aria-label="Position" @input="seek" />
                <span class="hint">{{ (position / player.fps).toFixed(1) }} / {{ (player.frames.length / player.fps).toFixed(1) }} s</span>
              </div>
              <p class="hint">
                {{ player.width }}x{{ player.height }}, {{ player.frames.length }} frames{{ player.wavUrl ? ', with sound' : ', no soundtrack' }} ·
                <a :href="player.aviUrl" download="oxinfer-video.avi">Download AVI</a>
                <template v-if="player.wavUrl"> · <a :href="player.wavUrl" download="oxinfer-video.wav">Download WAV</a></template>
              </p>
              <p class="hint">The AVI (Motion JPEG with PCM sound) plays in VLC and mpv; <code>ffmpeg -i oxinfer-video.avi out.mp4</code> converts it.</p>
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
  max-width: 100%;
  height: auto;
  background: #000;
  border-radius: 4px;
}
.controls {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  margin-top: 0.5rem;
}
.controls input[type='range'] {
  flex: 1;
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
