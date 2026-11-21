// Tools the chat model can call in the browser, and the system prompt that tells it where it runs.

export const TOOL_SPECS = [
  {
    name: 'get_datetime',
    description: "Returns the current date and time in the user's time zone: ISO 8601 with offset, weekday, time zone name and Unix time in milliseconds.",
    parameters: { type: 'object', properties: {} },
  },
  {
    name: 'run_javascript',
    description:
      'Runs JavaScript in a sandboxed Web Worker (no network, no storage, no page access) and returns the returned value and anything printed with console.log. Use it for exact arithmetic, date calculations and text or data processing. The code is the body of an async function: give the answer back with `return`; `await` is allowed. Time limit 5 seconds.',
    parameters: {
      type: 'object',
      properties: { code: { type: 'string', description: 'Function body, for example: return [1, 2, 3].map((x) => x * 2);' } },
      required: ['code'],
    },
  },
];

const MAX_RESULT = 4000;

function clip(s) {
  return s.length > MAX_RESULT ? `${s.slice(0, MAX_RESULT)}… (${s.length - MAX_RESULT} more characters)` : s;
}

export function datetimeNow() {
  const d = new Date();
  const off = -d.getTimezoneOffset();
  const sign = off >= 0 ? '+' : '-';
  const pad = (n) => String(Math.floor(Math.abs(n))).padStart(2, '0');
  const local = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}${sign}${pad(off / 60)}:${pad(off % 60)}`;
  const weekday = d.toLocaleDateString('en-US', { weekday: 'long' });
  const zone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'unknown';
  return { local, weekday, zone, unixMs: d.getTime() };
}

// The worker removes network and storage APIs before running anything, and is terminated after
// each call (or on timeout), so code cannot keep state or reach the page. This is hardening, not a
// security boundary: the model writes the code, and the page holds nothing secret.
const WORKER_SRC = `
const logs = [];
const show = (v) => {
  if (typeof v === 'string') return v;
  try { const j = JSON.stringify(v, (k, x) => (typeof x === 'bigint' ? x.toString() + 'n' : x)); return j === undefined ? String(v) : j; } catch { return String(v); }
};
for (const k of ['log', 'info', 'warn', 'error', 'debug']) console[k] = (...a) => logs.push(a.map(show).join(' '));
// Some of these live on the global's prototypes (fetch is on WorkerGlobalScope.prototype), so
// remove them along the whole chain, not just on self.
const blocked = ['fetch', 'XMLHttpRequest', 'WebSocket', 'WebTransport', 'EventSource', 'importScripts', 'indexedDB', 'caches', 'BroadcastChannel', 'Worker', 'SharedWorker', 'RTCPeerConnection'];
for (let o = self; o; o = Object.getPrototypeOf(o)) {
  for (const k of blocked) {
    try { Object.defineProperty(o, k, { value: undefined, configurable: false, writable: false }); } catch {}
  }
}
self.onmessage = async (e) => {
  try {
    const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
    const r = await new AsyncFunction(e.data)();
    postMessage({ ok: true, result: r === undefined ? undefined : show(r), logs });
  } catch (err) {
    postMessage({ ok: false, error: String((err && err.stack) || err), logs });
  }
};`;

export function runJavascript(code, timeoutMs = 5000) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(new Blob([WORKER_SRC], { type: 'text/javascript' }));
    const w = new Worker(url);
    const done = (text) => {
      clearTimeout(timer);
      w.terminate();
      URL.revokeObjectURL(url);
      resolve(clip(text));
    };
    const timer = setTimeout(() => done(`error: the code did not finish within ${timeoutMs / 1000} s and was stopped`), timeoutMs);
    w.onmessage = (e) => {
      const m = e.data;
      const parts = [];
      if (m.ok) parts.push(m.result === undefined ? 'returned: undefined (use `return` to give back a value)' : `returned: ${m.result}`);
      else parts.push(`error: ${m.error}`);
      if (m.logs.length) parts.push(`console:\n${m.logs.join('\n')}`);
      done(parts.join('\n'));
    };
    w.onerror = (e) => done(`error: ${e.message || 'the worker failed'}`);
    w.postMessage(String(code ?? ''));
  });
}

/// Runs one tool call ({ name, arguments }) and returns its result text.
export async function runTool(call) {
  switch (call.name) {
    case 'get_datetime':
      return JSON.stringify(datetimeNow());
    case 'run_javascript':
      if (typeof call.arguments?.code !== 'string') return 'error: missing argument "code"';
      return runJavascript(call.arguments.code);
    default:
      return `error: unknown tool ${call.name}`;
  }
}

/// A system prompt describing where the model runs: this browser tab (wasm), or with `native`
/// (the backend's platform summary) the computer that serves the page.
export function systemPrompt({ tools, gpuName, modelName, native }) {
  const n = navigator;
  const now = datetimeNow();
  if (native) {
    const env = [`computer: ${native}`, `time zone: ${now.zone}`, `local date and time when this conversation started: ${now.weekday} ${now.local}`, `language: ${n.language}`];
    let s = `You are ${modelName ? `${modelName}, ` : ''}a helpful assistant running locally on the user's own computer with oxinfer. Nothing the user types leaves their machine, and you have no internet access.\n\nEnvironment:\n- ${env.join('\n- ')}`;
    if (tools) s += TOOLS_NOTE;
    return s;
  }
  const brand = n.userAgentData?.brands?.filter((b) => !/not.a.brand/i.test(b.brand)).map((b) => `${b.brand} ${b.version}`).join(', ');
  const platform = n.userAgentData?.platform || n.platform || 'unknown';
  const env = [
    `browser: ${brand || n.userAgent}`,
    `operating system: ${platform}`,
    `language: ${n.language}`,
    `time zone: ${now.zone}`,
    `local date and time when this conversation started: ${now.weekday} ${now.local}`,
    `screen: ${screen.width}x${screen.height}`,
    `${n.hardwareConcurrency || '?'} logical CPU cores`,
    gpuName ? `GPU (WebGPU): ${gpuName}` : 'no WebGPU, running on the CPU',
  ];
  let s = `You are ${modelName ? `${modelName}, ` : ''}a helpful assistant running entirely inside the user's web browser, in this tab, with oxinfer (WebAssembly${gpuName ? ' and WebGPU' : ''}). Nothing the user types leaves their device, and you have no internet access.\n\nEnvironment:\n- ${env.join('\n- ')}`;
  if (tools) s += TOOLS_NOTE;
  return s;
}

const TOOLS_NOTE =
  '\n\nYou can call tools. Use get_datetime when the current time matters (the time above may be stale), and run_javascript for exact calculations, dates or data processing instead of guessing. Answer directly when no tool is needed.';
