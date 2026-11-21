<script setup>
// A newer oxinfer release, when the server knows of one (GET /api/web/update, which checks at
// most every few hours in the background), with its release notes and, when this server may
// replace itself, an Update button: the server downloads the release, restarts into it on the
// same address, and the page waits for /api/web/info to answer with the new version.
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { store } from '../lib/native.js';

const DISMISS = 'oxinfer-update-dismissed';
const info = ref(null);
// idle | working | reconnecting | done | failed
const phase = ref('idle');
const message = ref('');
const dismissed = ref(readDismissed());
let timer = null;
let alive = true;

function readDismissed() {
  try {
    return sessionStorage.getItem(DISMISS) || '';
  } catch {
    return '';
  }
}

function dismiss() {
  dismissed.value = info.value?.latest || '';
  try {
    sessionStorage.setItem(DISMISS, dismissed.value);
  } catch {
    /* private mode */
  }
}

const shown = computed(() => {
  if (phase.value !== 'idle') return true;
  const i = info.value;
  if (!i) return false;
  if (i.job && ['updated', 'error'].includes(i.job.state)) return true;
  return i.state === 'available' && dismissed.value !== i.latest;
});

const later = (ms, f) => {
  clearTimeout(timer);
  if (alive) timer = setTimeout(f, ms);
};

async function check() {
  try {
    info.value = await store.api.updateStatus();
  } catch {
    // An older server without the endpoint, or a key not entered yet: try again later.
    return later(60_000, check);
  }
  const job = info.value.job?.state;
  if (['checking', 'downloading', 'restarting'].includes(job)) return follow();
  // While the server asks GitHub, look again soon; then once an hour.
  later(info.value.checking ? 3000 : 3_600_000, check);
}

function jobLine(job, latest) {
  switch (job?.state) {
    case 'checking':
      return 'Looking up the release…';
    case 'downloading':
      return `Downloading oxinfer ${job.version || latest}${job.size ? ` (${(job.size / 1e6).toFixed(0)} MB)` : ''}…`;
    case 'restarting':
      return `Restarting into oxinfer ${job.version || latest}…`;
    default:
      return job?.message || '';
  }
}

async function update() {
  phase.value = 'working';
  message.value = 'Starting the update…';
  try {
    await store.api.startUpdate();
  } catch (e) {
    phase.value = 'failed';
    message.value = e.message;
    return;
  }
  follow();
}

/// Polls the update job until it ends; the server going away during the restart is expected.
async function follow() {
  phase.value = 'working';
  let st;
  try {
    st = await store.api.updateStatus();
  } catch {
    return reconnect();
  }
  info.value = st;
  const job = st.job || {};
  message.value = jobLine(job, st.latest);
  if (job.state === 'restarting') return reconnect();
  if (job.state === 'updated') {
    phase.value = 'done';
    message.value = job.message;
    return;
  }
  if (job.state === 'error') {
    phase.value = 'failed';
    message.value = `The update failed: ${job.error}`;
    return;
  }
  if (job.state === 'up_to_date') {
    phase.value = 'idle';
    return check();
  }
  later(1000, follow);
}

/// Waits for the restarted server: /api/web/info answers with another version.
function reconnect() {
  phase.value = 'reconnecting';
  const from = info.value?.current;
  const target = info.value?.job?.version || info.value?.latest;
  const t0 = Date.now();
  message.value = `Restarting into oxinfer ${target}…`;
  const poll = async () => {
    try {
      const r = await fetch(new URL('api/web/info', document.baseURI), { cache: 'no-store' });
      const v = r.ok ? await r.json() : null;
      if (v?.version && v.version !== from) {
        phase.value = 'done';
        message.value = `oxinfer ${v.version} is running. Reloading the page…`;
        later(1500, () => location.reload());
        return;
      }
    } catch {
      /* not back yet */
    }
    if (Date.now() - t0 > 120_000) {
      phase.value = 'failed';
      message.value = 'The server did not come back after the update. Check its terminal, or start oxinfer web again.';
      return;
    }
    later(1000, poll);
  };
  later(1500, poll);
}

onMounted(check);
onBeforeUnmount(() => {
  alive = false;
  clearTimeout(timer);
});
</script>

<template>
  <aside v-if="shown && info" class="update" role="status" aria-live="polite">
    <template v-if="phase === 'idle' && info.job?.state === 'updated'">
      <span>{{ info.job.message }}</span>
    </template>
    <template v-else-if="phase === 'idle' && info.job?.state === 'error' && info.state !== 'available'">
      <span>The last update failed: {{ info.job.error }}</span>
    </template>
    <template v-else-if="phase === 'idle'">
      <span class="what">
        <strong>oxinfer {{ info.latest }} is available</strong>
        <span class="muted"> (this is {{ info.current }})</span>
      </span>
      <a v-if="info.url" :href="info.url" target="_blank" rel="noopener">Release notes</a>
      <button v-if="info.can_update" class="primary small" type="button" @click="update">Update and restart</button>
      <span v-else class="muted">
        To install it, run <code>{{ info.command }}</code> where oxinfer runs<template v-if="info.why_not"> ({{ info.why_not }})</template>.
      </span>
      <button class="link close" type="button" aria-label="Hide until the next visit" @click="dismiss">Hide</button>
    </template>
    <template v-else>
      <span v-if="phase === 'working' || phase === 'reconnecting'" class="spinner" aria-hidden="true"></span>
      <span :class="{ err: phase === 'failed' }">{{ message }}</span>
      <a v-if="phase === 'failed' && info.url" :href="info.url" target="_blank" rel="noopener">Release notes</a>
      <button v-if="phase === 'failed'" class="link" type="button" @click="phase = 'idle'">Close</button>
    </template>
  </aside>
</template>

<style scoped>
.update {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem 1rem;
  margin: 0;
  padding: 0.45rem clamp(1rem, 4vw, 2.5rem);
  border-bottom: 1px solid var(--line);
  background: var(--surface);
  font-size: 0.95rem;
}
.what strong {
  font-family: var(--display);
  font-weight: 650;
  color: var(--verdigris-deep);
}
.muted {
  color: var(--muted);
}
.err {
  color: var(--error);
}
.small {
  padding: 0.2rem 0.75rem;
  font-size: 0.9rem;
}
.close {
  margin-left: auto;
}
.spinner {
  width: 0.9rem;
  height: 0.9rem;
  border: 2px solid var(--line);
  border-top-color: var(--verdigris);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
@media (prefers-reduced-motion: reduce) {
  .spinner {
    animation: none;
  }
}
</style>
