// Shared state of the native mode: the backend's status (models, downloads, disk), refreshed by
// polling while the page is open, and helpers the views share.
import { reactive } from 'vue';
import { ApiError } from './api.js';

export const store = reactive({
  api: null,
  info: null,
  status: null,
  error: '',
  needKey: false,
  // Model chosen per kind (chat, embedding, image, transcription, speech), by id.
  chosen: {},
});

let timer = null;
let fast = false;

export function setup(api, info) {
  store.api = api;
  store.info = info;
}

export async function refresh() {
  if (!store.api) return;
  try {
    store.status = await store.api.status();
    store.error = '';
    store.needKey = false;
  } catch (e) {
    if (e instanceof ApiError && e.status === 401) store.needKey = true;
    else store.error = `The oxinfer server does not answer (${e.message}).`;
  }
  // Faster while something moves.
  const moving = store.status?.models?.some((m) => ['queued', 'waiting', 'downloading', 'loading'].includes(m.state)) || store.status?.downloads?.some((d) => ['planning', 'downloading'].includes(d.status));
  if (moving !== fast) {
    fast = moving;
    startPolling();
  }
}

export function startPolling() {
  clearInterval(timer);
  timer = setInterval(refresh, fast ? 700 : 3000);
}

/// Whether a model (or a cache entry) of `kind` serves the views of `want`: chat, embedding,
/// image, transcription, speech (text to speech: the speech models but Haru, and LFM3-Omni),
/// voice (spoken replies: LFM3-Omni and Haru), video.
export function serves(m, want) {
  const kind = m.kind || 'chat';
  if (want === 'voice') return !!m.audio_out;
  if (want === 'speech') return (kind === 'speech' && m.audio_out !== 'haru') || m.audio_out === 'lfm3_omni';
  return kind === want;
}

/// The models of the backend that serve `kind` (see `serves`).
export function modelsOf(kind) {
  return (store.status?.models || []).filter((m) => serves(m, kind));
}

export function fmtBytes(n) {
  if (n === null || n === undefined) return '?';
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} TB`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} kB`;
  return `${n} B`;
}

export function fmtSecs(s) {
  if (s === null || s === undefined || !Number.isFinite(s)) return '';
  s = Math.round(s);
  if (s >= 3600) return `${Math.floor(s / 3600)} h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')} min`;
  if (s >= 60) return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, '0')} s`;
  return `${s} s`;
}

export function fmtAgo(t) {
  if (!t) return 'never';
  const d = Date.now() / 1e3 - t;
  if (d < 60) return 'just now';
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  if (d < 86400 * 60) return `${Math.floor(d / 86400)} days ago`;
  return new Date(t * 1e3).toLocaleDateString();
}

/// opencode's configuration as `GET /api/web/opencode` reports it (null until asked, or when the
/// server does not offer it), with the last change's message and the actions in progress.
export const opencode = reactive({ status: null, message: '', error: '', busy: {} });
let ocAsked = 0;

/// Asks the server again (at most every 5 s unless `force`).
export async function refreshOpencode(force = false) {
  if (!store.api || (!force && Date.now() - ocAsked < 5000)) return;
  ocAsked = Date.now();
  try {
    opencode.status = await store.api.opencode();
  } catch {
    opencode.status = null;
  }
}

/// The opencode state of chat model `id`: null when opencode is not offered or the model is not
/// a listed chat model, else its row (`present`, `up_to_date`).
export function opencodeOf(id) {
  const s = opencode.status;
  if (!s?.available || !s.detected) return null;
  return s.models?.find((m) => m.id === id) || null;
}

/// Adds (or updates) or removes model `id` in opencode's configuration.
export async function opencodeChange(id, add) {
  opencode.busy[id] = true;
  opencode.error = '';
  try {
    const r = add ? await store.api.opencodeAdd(id) : await store.api.opencodeRemove(id);
    opencode.message = r.message || '';
  } catch (e) {
    opencode.error = e.message;
  }
  delete opencode.busy[id];
  await refreshOpencode(true);
}

/// What a model's load state reads as.
export function loadLine(m) {
  const l = m.load || {};
  switch (m.state) {
    case 'ready':
      return 'Loaded';
    case 'busy':
      return 'Loaded, working';
    case 'queued':
      return 'Starting…';
    case 'waiting':
      return 'Waiting for the model to be free…';
    case 'downloading':
      return `Downloading ${l.file || ''}${l.total ? `: ${fmtBytes(l.done)} of ${fmtBytes(l.total)}` : ''} (file ${l.index} of ${l.count})`;
    case 'loading':
      return 'Reading the weights into memory…';
    case 'error':
      return `Could not load: ${l.error || 'unknown error'}`;
    default:
      return 'Not loaded (loads on first use)';
  }
}
