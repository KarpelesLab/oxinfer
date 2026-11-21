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

/// The models of the backend that serve `kind` (chat, embedding, image, transcription, speech).
export function modelsOf(kind) {
  return (store.status?.models || []).filter((m) => m.kind === kind || (!m.kind && kind === 'chat'));
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
