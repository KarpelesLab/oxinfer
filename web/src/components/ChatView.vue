<script setup>
// The conversation: transcript, composer, reply settings and the last reply's speed. It drives a
// chat session (lib/chat.js): the wasm engine of this tab or a model of the native backend. The
// model section of the side column comes from the parent (slot "model"), and more sections can
// follow (slot "after").
import { computed, nextTick, ref, watch } from 'vue';
import Reply from './Reply.vue';
import { runTool } from '../lib/tools.js';

const props = defineProps({
  session: { type: Object, default: null },
  modelName: { type: String, default: '' },
  // The session can take a message (a model is loaded and idle).
  ready: { type: Boolean, default: false },
  log: { type: Function, default: () => {} },
  auto: { type: Boolean, default: false },
  system: { type: String, default: '' },
  // Native models: attachments, top-p.
  native: { type: Boolean, default: false },
  initialDraft: { type: String, default: '' },
  initialTemperature: { type: Number, default: 0.7 },
  initialMaxTokens: { type: Number, default: 512 },
  initialTools: { type: Boolean, default: true },
});
const emit = defineEmits(['busy', 'error', 'reset-system', 'tools']);

const messages = ref([]); // { role: 'user' | 'model' | 'tool', text, thinking, media, name, args, result, running }
const draft = ref(props.initialDraft);
const temperature = ref(props.initialTemperature);
const topP = ref(0.95);
const maxTokens = ref(props.initialMaxTokens);
const think = ref(false);
const useTools = ref(props.initialTools);
const systemText = ref(props.system);
const stats = ref(null);
const generating = ref(false);
const stage = ref('');
const error = ref('');
const transcript = ref(null);
const attachments = ref([]); // { url, kind, name }
const fileInput = ref(null);

watch(
  () => props.system,
  (s) => (systemText.value = s),
);
watch(generating, (g) => emit('busy', g));
// Another model (or the same one loaded again): a new conversation.
watch(
  () => props.session,
  () => {
    messages.value = [];
    stats.value = null;
    error.value = '';
  },
);

const canSend = computed(() => props.ready && !generating.value && (draft.value.trim() || attachments.value.length));

function toolLabel(m) {
  if (m.name === 'get_datetime') return m.running ? 'Checking the date and time…' : 'Checked the date and time';
  if (m.name === 'run_javascript') return m.running ? 'Running JavaScript…' : 'Ran JavaScript';
  return m.running ? `Calling ${m.name}…` : `Called ${m.name}`;
}

async function scrollDown() {
  await nextTick();
  const el = transcript.value;
  if (el) el.scrollTop = el.scrollHeight;
}

function friendly(e) {
  const m = String(e?.message || e);
  if (/unreachable|out of memory|allocation/i.test(m)) {
    return 'The model did not fit in this tab’s memory. Try a smaller model, or a folder written by oxinfer --save-quantized.';
  }
  if (/Failed to fetch|NetworkError/i.test(m)) return props.native ? 'The oxinfer server could not be reached.' : 'The download failed. Check the connection and that the URL allows cross-origin requests.';
  return m;
}

async function send() {
  const text = draft.value.trim();
  if (!canSend.value || !props.session) return;
  draft.value = '';
  const media = attachments.value;
  attachments.value = [];
  error.value = '';
  stage.value = '';
  generating.value = true;
  messages.value.push({ role: 'user', text, media });
  messages.value.push({ role: 'model', text: '', thinking: '' });
  let reply = messages.value[messages.value.length - 1];
  await scrollDown();
  let n = 0;
  try {
    stats.value = await props.session.send(
      text,
      { images: media, system: systemText.value.trim(), think: think.value, maxTokens: maxTokens.value, temperature: temperature.value, topP: topP.value, tools: useTools.value },
      {
        onText(t, thinking) {
          reply.text = t;
          reply.thinking = thinking;
          stage.value = '';
          if (++n % 8 === 0) scrollDown();
        },
        onStage(s) {
          stage.value = s;
        },
        async onTool(c) {
          if (!reply.text.trim() && !reply.thinking) messages.value.splice(messages.value.indexOf(reply), 1);
          messages.value.push({ role: 'tool', name: c.name, args: c.arguments, result: '', running: true });
          const m = messages.value[messages.value.length - 1];
          await scrollDown();
          m.result = await runTool(c);
          m.running = false;
          props.log(`[tool ${c.name}(${JSON.stringify(c.arguments)}) -> ${m.result.replace(/\n/g, '\\n').slice(0, 300)}]`);
          return m.result;
        },
        onNextReply() {
          messages.value.push({ role: 'model', text: '', thinking: '' });
          reply = messages.value[messages.value.length - 1];
          scrollDown();
        },
      },
    );
    const st = stats.value;
    props.log(`[prefill ${st.promptTokens} tok at ${st.prefill.toFixed(1)} tok/s | decode ${st.tokens} tok at ${st.decode.toFixed(1)} tok/s]`);
    if (props.auto) props.log(`[output] ${reply.text.replace(/\n/g, '\\n')}`);
  } catch (e) {
    error.value = friendly(e);
    props.log(`[error] ${e.message || e}`);
    if (!reply.text && !reply.thinking) messages.value.splice(messages.value.indexOf(reply), 1);
    emit('error', e);
  }
  stage.value = '';
  generating.value = false;
  scrollDown();
}

function stop() {
  props.session?.stop();
}

function newConversation() {
  if (!props.session || generating.value) return;
  props.session.reset();
  messages.value = [];
  stats.value = null;
  error.value = '';
}

function onKey(e) {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
}

function readUrl(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

async function attach(e) {
  for (const f of e.target.files || []) {
    const kind = f.type.startsWith('image/') ? 'image' : f.type.startsWith('audio/') ? 'audio' : f.type.startsWith('video/') ? 'video' : null;
    if (!kind) {
      error.value = `${f.name}: attach images, audio or video.`;
      continue;
    }
    attachments.value.push({ url: await readUrl(f), kind, name: f.name });
  }
  e.target.value = '';
}

function onToolsChange() {
  emit('tools', useTools.value);
  newConversation();
}

// The parent's "new conversation" (a model change).
defineExpose({ newConversation, send, draft, messages, useTools });
</script>

<template>
  <main class="layout">
    <aside class="side">
      <slot name="model" />

      <section v-if="stats" aria-labelledby="speed-h">
        <h2 id="speed-h">Last reply</h2>
        <dl class="speed">
          <dt>Prompt</dt>
          <dd>{{ stats.promptTokens }} tokens at {{ stats.prefill.toFixed(0) }} per second</dd>
          <dt>Reply</dt>
          <dd>{{ stats.tokens }} tokens at {{ stats.decode.toFixed(1) }} per second{{ stats.stopped ? ', stopped' : '' }}</dd>
        </dl>
      </section>

      <details class="more" :open="native">
        <summary>Settings</summary>
        <label class="inline">Temperature <input v-model.number="temperature" class="num" type="number" min="0" max="2" step="0.1" /></label>
        <label v-if="native" class="inline">Top-p <input v-model.number="topP" class="num" type="number" min="0.05" max="1" step="0.05" /></label>
        <label class="inline">Longest reply <input v-model.number="maxTokens" class="num" type="number" min="1" max="32768" step="1" /> tokens</label>
        <label class="inline"><input v-model="think" type="checkbox" /> Let the model think first</label>
        <label class="inline"><input v-model="useTools" type="checkbox" @change="onToolsChange" /> Let the model use tools</label>
        <p class="hint">It can read the date and time, and run JavaScript in a sandboxed worker of this page with no network or storage access.</p>
        <label class="block">System prompt
          <textarea v-model="systemText" class="sys" rows="7" spellcheck="false" aria-label="System prompt"></textarea>
        </label>
        <p class="hint">
          {{ native ? 'Sent with every message.' : 'Applies from the next conversation.' }}
          <button class="link" type="button" :disabled="!session" @click="emit('reset-system')">Reset</button>
        </p>
      </details>

      <slot name="after" />
    </aside>

    <section class="chat" aria-label="Conversation">
      <div ref="transcript" class="transcript" aria-live="polite">
        <p v-if="!messages.length" class="empty">
          {{ ready ? 'Ask it something.' : native ? 'Load a chat model, then ask it something.' : 'Load a model, then ask it something.' }}
        </p>
        <template v-for="(m, i) in messages" :key="i">
          <details v-if="m.role === 'tool'" class="tool" :open="m.name === 'run_javascript' && !m.running">
            <summary>{{ toolLabel(m) }}</summary>
            <pre v-if="m.name === 'run_javascript'" class="code">{{ m.args.code }}</pre>
            <pre v-if="m.result" class="result">{{ m.result }}</pre>
          </details>
          <article v-else :class="['msg', m.role]">
            <h3 class="who">{{ m.role === 'user' ? 'You' : modelName }}</h3>
            <template v-if="m.role === 'user'">
              <div v-if="m.media?.length" class="media">
                <template v-for="(a, j) in m.media" :key="j">
                  <img v-if="a.kind === 'image'" :src="a.url" :alt="a.name" />
                  <audio v-else-if="a.kind === 'audio'" :src="a.url" controls></audio>
                  <video v-else :src="a.url" controls muted></video>
                </template>
              </div>
              <p v-if="m.text" class="said">{{ m.text }}</p>
            </template>
            <template v-else>
              <details v-if="m.thinking" class="thought" :open="!m.text">
                <summary>{{ m.text ? 'Thought' : 'Thinking…' }}</summary>
                <p>{{ m.thinking }}</p>
              </details>
              <Reply v-if="m.text" :text="m.text" />
              <p v-else-if="!m.thinking" class="thinking">
                {{ i === messages.length - 1 && stage ? stage : i > 0 && messages[i - 1].role === 'tool' ? 'Reading the result…' : 'Reading your message…' }}
              </p>
            </template>
          </article>
        </template>
        <p v-if="error" class="error" role="alert">{{ error }}</p>
      </div>

      <form class="composer" @submit.prevent="send">
        <div v-if="attachments.length" class="attached" aria-label="Attachments">
          <span v-for="(a, i) in attachments" :key="i" class="chip">
            {{ a.kind }}: {{ a.name }}
            <button type="button" class="link" :aria-label="`Remove ${a.name}`" @click="attachments.splice(i, 1)">×</button>
          </span>
        </div>
        <textarea
          v-model="draft"
          rows="2"
          :placeholder="ready || generating ? 'Message' : 'Load a model first'"
          aria-label="Message"
          :disabled="!session"
          @keydown="onKey"
        ></textarea>
        <div class="actions">
          <button v-if="generating" type="button" class="primary" @click="stop">Stop</button>
          <button v-else type="submit" class="primary" :disabled="!canSend">Send</button>
          <button type="button" class="quiet" :disabled="generating || !messages.length" @click="newConversation">New conversation</button>
          <template v-if="native && session?.supports?.images">
            <input ref="fileInput" class="sr-only" type="file" accept="image/*,audio/*,video/*" multiple tabindex="-1" aria-hidden="true" @change="attach" />
            <button type="button" class="quiet" :disabled="!session" title="Images for vision models, audio or video for the models that take them" @click="fileInput.click()">Attach…</button>
          </template>
        </div>
      </form>
    </section>
  </main>
</template>

<style scoped>
.media {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 0.2rem 0 0.4rem;
}
.media img,
.media video {
  max-width: min(100%, 18rem);
  max-height: 14rem;
  border-radius: 4px;
  box-shadow: 0 0 0 1px var(--line);
}
.thought {
  margin: 0 0 0.5rem;
  padding: 0.3rem 0.7rem;
  border-left: 3px solid var(--line);
  color: var(--muted);
  font-size: 0.9rem;
}
.thought summary {
  cursor: pointer;
  font-family: var(--display);
  font-weight: 600;
}
.thought p {
  margin: 0.3rem 0 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.attached {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-bottom: 0.4rem;
}
.chip {
  display: inline-flex;
  gap: 0.4rem;
  align-items: center;
  padding: 0.1rem 0.5rem;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--paper);
  font-size: 0.85rem;
}
.chat .error {
  max-width: 68ch;
}
</style>
