// The native backend: `oxinfer web` serves this app and runs the models on its own machine. The
// page talks to it over HTTP: the management API under /api/web/ (catalog, downloads, cache,
// loading) and the OpenAI-compatible /v1 API for the models themselves. Hosted on its own (GitHub
// Pages, `npm run dev`), /api/web/info does not answer and the app runs the wasm engine instead.

const KEY_STORE = 'oxinfer-api-key';

function readKey() {
  try {
    return localStorage.getItem(KEY_STORE) || '';
  } catch {
    return '';
  }
}

/// The API key from the address (`#key=...`, as `oxinfer web --api-key` opens the page), kept
/// and removed from the address bar.
function keyFromHash() {
  const m = /(?:^#|&)key=([^&]+)/.exec(location.hash);
  if (!m) return '';
  const key = decodeURIComponent(m[1]);
  try {
    localStorage.setItem(KEY_STORE, key);
  } catch {
    /* private mode */
  }
  history.replaceState(null, '', location.pathname + location.search);
  return key;
}

export class ApiError extends Error {
  constructor(status, message, code) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

/// `{ native: true, info }` when a native backend serves this page, else `{ native: false }`.
export async function detectBackend() {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 3000);
  try {
    const r = await fetch(new URL('api/web/info', document.baseURI), { signal: ctl.signal, cache: 'no-store' });
    if (!r.ok) return { native: false };
    const info = await r.json();
    return info && info.oxinfer === true ? { native: true, info } : { native: false };
  } catch {
    return { native: false };
  } finally {
    clearTimeout(timer);
  }
}

/// Server-sent events from a fetch Response: calls `onEvent(event, data)` for each.
export async function readSse(resp, onEvent) {
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let event = 'message';
  let data = [];
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).replace(/\r$/, '');
      buf = buf.slice(nl + 1);
      if (line === '') {
        if (data.length) {
          const stop = onEvent(event, data.join('\n'));
          if (stop === false) {
            reader.cancel().catch(() => {});
            return;
          }
        }
        event = 'message';
        data = [];
      } else if (line.startsWith(':')) {
        onEvent('comment', line.slice(1).trim());
      } else if (line.startsWith('data:')) {
        data.push(line.slice(5).replace(/^ /, ''));
      } else if (line.startsWith('event:')) {
        event = line.slice(6).trim();
      }
    }
  }
}

export class HttpApi {
  constructor(info) {
    this.info = info;
    this.base = new URL('.', document.baseURI);
    this.key = keyFromHash() || readKey();
  }

  setKey(key) {
    this.key = key;
    try {
      if (key) localStorage.setItem(KEY_STORE, key);
      else localStorage.removeItem(KEY_STORE);
    } catch {
      /* private mode */
    }
  }

  url(path) {
    return new URL(path.replace(/^\//, ''), this.base);
  }

  headers(extra = {}) {
    const h = { ...extra };
    if (this.key) h.Authorization = `Bearer ${this.key}`;
    return h;
  }

  async raw(path, { method = 'GET', json, body, headers = {}, signal } = {}) {
    const opts = { method, headers: this.headers(headers), signal };
    if (json !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(json);
    } else if (body !== undefined) {
      opts.body = body;
    } else if (method !== 'GET') {
      // The management API wants JSON for every change (no cross-site forms).
      opts.headers['Content-Type'] = 'application/json';
    }
    const r = await fetch(this.url(path), opts);
    if (!r.ok) {
      let msg = `${r.status} ${r.statusText}`;
      let code;
      try {
        const v = await r.json();
        msg = v?.error?.message || msg;
        code = v?.error?.code;
      } catch {
        /* not JSON */
      }
      throw new ApiError(r.status, msg, code);
    }
    return r;
  }

  async get(path, opts) {
    return (await this.raw(path, opts)).json();
  }

  async post(path, json, opts = {}) {
    return (await this.raw(path, { ...opts, method: 'POST', json })).json();
  }

  // ---- management --------------------------------------------------------------------------
  status() {
    return this.get('api/web/status');
  }
  catalog() {
    return this.get('api/web/catalog');
  }
  cache() {
    return this.get('api/web/cache');
  }
  models() {
    return this.get('api/web/models');
  }
  plan(model) {
    return this.post('api/web/plan', { model });
  }
  download(model) {
    return this.post('api/web/downloads', { model });
  }
  downloads() {
    return this.get('api/web/downloads');
  }
  cancelDownload(id) {
    return this.post(`api/web/downloads/${encodeURIComponent(id)}/cancel`, {});
  }
  removeCached(name) {
    return this.post('api/web/cache/delete', { name });
  }
  load(model, { id, options } = {}) {
    return this.post('api/web/models/load', { model, id, options });
  }
  unload(id) {
    return this.post('api/web/models/unload', { id });
  }

  /// Follows a download job until it ends: `onUpdate(job)` for each report.
  async watchDownload(id, onUpdate, signal) {
    const r = await this.raw(`api/web/downloads/${encodeURIComponent(id)}/events`, { signal });
    await readSse(r, (event, data) => {
      if (event !== 'message') return;
      try {
        onUpdate(JSON.parse(data));
      } catch {
        /* partial line */
      }
    });
  }

  // ---- /v1 ---------------------------------------------------------------------------------
  /// A streamed chat completion: `on.delta({content, reasoning})`, `on.progress(event)`;
  /// resolves to { finish, toolCalls, usage }.
  async chatStream(body, on = {}, signal) {
    const r = await this.raw('v1/chat/completions', {
      method: 'POST',
      json: { ...body, stream: true, stream_options: { include_usage: true, progress: true } },
      signal,
    });
    const out = { finish: null, toolCalls: [], usage: null, error: null };
    await readSse(r, (event, data) => {
      if (event === 'comment') return;
      if (data === '[DONE]') return false;
      let v;
      try {
        v = JSON.parse(data);
      } catch {
        return;
      }
      if (event === 'progress') {
        on.progress?.(v);
        return;
      }
      if (v.error) {
        out.error = v.error.message || 'the model failed';
        return;
      }
      if (v.usage) out.usage = v.usage;
      const c = v.choices?.[0];
      if (!c) return;
      const d = c.delta || {};
      if (d.content || d.reasoning_content) on.delta?.({ content: d.content || '', reasoning: d.reasoning_content || '' });
      for (const t of d.tool_calls || []) {
        const i = t.index ?? out.toolCalls.length;
        const cur = (out.toolCalls[i] ||= { id: t.id, name: '', arguments: '' });
        if (t.id) cur.id = t.id;
        if (t.function?.name) cur.name += t.function.name;
        if (t.function?.arguments) cur.arguments += t.function.arguments;
      }
      if (c.finish_reason) out.finish = c.finish_reason;
    });
    if (out.error) throw new ApiError(500, out.error);
    return out;
  }
}
