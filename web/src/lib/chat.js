// Chat sessions behind one interface, so that the same conversation view drives the wasm engine
// in this tab or a model of the native backend:
//
//   session.send(text, settings, hooks) -> { promptTokens, prefill, tokens, decode, stopped }
//   session.stop()     ends the reply being written
//   session.reset()    starts a new conversation
//
// settings: { images: [{ url, kind: 'image' | 'audio' | 'video', name }], system, think,
//             maxTokens, temperature, topP, tools }
// hooks:    { onText(text, thinking), onTool(call) -> Promise<result text>, onNextReply(),
//             onStage(text) }
import { TOOL_SPECS, runTool } from './tools.js';

const MAX_TOOL_ROUNDS = 5;
const pause = () => new Promise((r) => setTimeout(r, 0));

/// The engine of this tab (lib/oxinfer.js), whose conversation state lives in the module.
export class WasmChat {
  constructor(engine) {
    this.engine = engine;
    this.stopRequested = false;
    this.toolsOn = null;
    this.supports = { images: false, tools: true, think: true, stages: false };
  }

  setTools(on) {
    if (this.toolsOn === on) return;
    this.toolsOn = on;
    this.engine.setTools(on ? TOOL_SPECS : []);
  }

  async send(text, s, hooks) {
    const engine = this.engine;
    this.stopRequested = false;
    this.setTools(!!s.tools);
    engine.setSampling({ temperature: s.temperature, topK: 64, topP: s.topP ?? 0.95, seed: Date.now() % 1e9 });
    const t0 = performance.now();
    const promptTokens = await engine.chatTurn(text, { system: s.system || null, think: s.think, maxTokens: s.maxTokens });
    const t1 = performance.now();
    let tokens = 0;
    for (let round = 0; ; round++) {
      while (!this.stopRequested && (await engine.next()) >= 0) {
        tokens++;
        const t = engine.text();
        if (!t.endsWith('�')) hooks.onText(t, '');
        await pause(); // a real task boundary, so the browser paints between tokens
      }
      hooks.onText(engine.text(), '');
      if (this.stopRequested || !s.tools || round >= MAX_TOOL_ROUNDS) break;
      const calls = engine.toolCalls();
      if (!calls.length) break;
      const results = [];
      for (const c of calls) results.push({ name: c.name, content: await hooks.onTool(c) });
      hooks.onNextReply();
      await engine.toolResults(results);
    }
    const t2 = performance.now();
    return { promptTokens, prefill: promptTokens / ((t1 - t0) / 1e3), tokens, decode: tokens / Math.max(1e-3, (t2 - t1) / 1e3), stopped: this.stopRequested };
  }

  stop() {
    this.stopRequested = true;
  }

  reset() {
    this.engine.reset();
  }
}

function fmtStage(p) {
  const mb = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${(n / 1e6).toFixed(0)} MB`);
  switch (p.stage) {
    case 'waiting':
      return 'Waiting: the model is busy with another request…';
    case 'loading':
      return 'Loading the model…';
    case 'downloading':
      return `Downloading ${p.file}${p.total ? ` (${mb(p.done)} of ${mb(p.total)})` : ''}…`;
    case 'prompt':
      return `Reading the prompt: ${p.done} of ${p.total} tokens…`;
    default:
      return '';
  }
}

/// A model of the native backend through /v1/chat/completions. The conversation is kept here and
/// sent whole each turn; the server reuses its cache for the part it has seen.
export class HttpChat {
  constructor(api, model) {
    this.api = api;
    this.model = model;
    this.messages = [];
    this.ctl = null;
    this.stopped = false;
    this.supports = { images: true, tools: true, think: true, stages: true };
  }

  userMessage(text, media) {
    if (!media?.length) return { role: 'user', content: text };
    const parts = [];
    for (const m of media) {
      if (m.kind === 'image') parts.push({ type: 'image_url', image_url: { url: m.url } });
      else if (m.kind === 'video') parts.push({ type: 'video_url', video_url: { url: m.url } });
      else if (m.kind === 'audio') {
        const [head, data] = m.url.split(',');
        const format = /audio\/(?:x-)?(wav|wave|mpeg|mp3|flac|ogg|webm|mp4|aac)/.exec(head)?.[1]?.replace('mpeg', 'mp3').replace('wave', 'wav');
        parts.push({ type: 'input_audio', input_audio: { data, format: format || 'wav' } });
      }
    }
    if (text) parts.push({ type: 'text', text });
    return { role: 'user', content: parts };
  }

  async send(text, s, hooks) {
    this.stopped = false;
    const turn = [this.userMessage(text, s.images)];
    const tools = s.tools ? TOOL_SPECS.map((t) => ({ type: 'function', function: t })) : undefined;
    let promptTokens = 0;
    let tokens = 0;
    let usageTokens = 0;
    let firstAt = 0;
    let content = '';
    let pending = false;
    const t0 = performance.now();
    try {
      for (let round = 0; ; round++) {
        const messages = [...(s.system ? [{ role: 'system', content: s.system }] : []), ...this.messages, ...turn];
        content = '';
        let reasoning = '';
        pending = true;
        this.ctl = new AbortController();
        const r = await this.api.chatStream(
          {
            model: this.model,
            messages,
            max_tokens: s.maxTokens,
            temperature: s.temperature,
            top_p: s.topP,
            reasoning_effort: s.think ? 'medium' : 'none',
            tools,
          },
          {
            progress: (p) => hooks.onStage?.(fmtStage(p)),
            delta: (d) => {
              if (!firstAt) firstAt = performance.now();
              content += d.content;
              reasoning += d.reasoning;
              tokens++;
              hooks.onText(content, reasoning);
            },
          },
          this.ctl.signal,
        );
        pending = false;
        if (r.usage) {
          if (round === 0) promptTokens = r.usage.prompt_tokens;
          usageTokens += r.usage.completion_tokens || 0;
        }
        const assistant = { role: 'assistant', content };
        if (r.toolCalls.length) {
          assistant.tool_calls = r.toolCalls.map((c, i) => ({ id: c.id || `call_${round}_${i}`, type: 'function', function: { name: c.name, arguments: c.arguments } }));
        }
        turn.push(assistant);
        if (!r.toolCalls.length || !s.tools || this.stopped || round >= MAX_TOOL_ROUNDS) break;
        for (const [i, c] of r.toolCalls.entries()) {
          let args = {};
          try {
            args = JSON.parse(c.arguments || '{}');
          } catch {
            args = { _raw: c.arguments };
          }
          const result = await hooks.onTool({ name: c.name, arguments: args });
          turn.push({ role: 'tool', tool_call_id: assistant.tool_calls[i].id, content: result });
        }
        hooks.onNextReply();
      }
    } catch (e) {
      if (e.name !== 'AbortError') throw e;
      // Stopped: the reply as far as it went.
      if (pending) turn.push({ role: 'assistant', content });
    } finally {
      this.ctl = null;
      // What was said stays in the conversation, even when stopped.
      if (turn.length > 1) this.messages.push(...turn);
    }
    const t2 = performance.now();
    const t1 = firstAt || t2;
    return {
      promptTokens,
      prefill: promptTokens / Math.max(1e-3, (t1 - t0) / 1e3),
      tokens: usageTokens || tokens,
      decode: (usageTokens || tokens) / Math.max(1e-3, (t2 - t1) / 1e3),
      stopped: this.stopped,
    };
  }

  stop() {
    this.stopped = true;
    this.ctl?.abort();
  }

  reset() {
    this.messages = [];
  }
}

export { runTool };
