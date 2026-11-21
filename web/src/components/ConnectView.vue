<script setup>
// Connecting other programs to this server: coding agents (opencode, Claude Code) and any
// OpenAI-compatible client, with the server's address, API key and model ids filled in.
import { computed, ref } from 'vue';
import { store } from '../lib/native.js';

const emit = defineEmits(['manage']);

const origin = location.origin;
const hostname = location.hostname;
const base = `${origin}/v1`;
const reveal = ref(false);
const copied = ref('');

const key = computed(() => store.api?.key || '');
const needsKey = computed(() => !!store.status?.auth_required);
const shownKey = computed(() => (needsKey.value ? (reveal.value ? key.value || 'YOUR_API_KEY' : '••••••••') : ''));
// What goes into the snippets: the real key only when shown.
const keyIn = computed(() => (needsKey.value ? (reveal.value && key.value ? key.value : 'YOUR_API_KEY') : 'oxinfer'));
const models = computed(() => (store.status?.models || []).filter((m) => m.kind === 'chat' || !m.kind));
const chosen = ref('');
const model = computed(() => chosen.value || models.value.find((m) => m.loaded)?.id || models.value[0]?.id || 'MODEL_ID');
const loopback = computed(() => !!store.status?.loopback);
const localHost = /^(localhost|127\.|\[::1\])/.test(location.hostname);

const opencode = computed(() => {
  const ms = {};
  for (const m of models.value.length ? models.value : [{ id: model.value }]) ms[m.id] = { name: `${m.id} (oxinfer)` };
  const options = { baseURL: base };
  if (needsKey.value) options.apiKey = keyIn.value;
  return JSON.stringify(
    {
      $schema: 'https://opencode.ai/config.json',
      provider: { oxinfer: { npm: '@ai-sdk/openai-compatible', name: 'oxinfer (local)', options, models: ms } },
      model: `oxinfer/${model.value}`,
    },
    null,
    2,
  );
});

const claude = computed(
  () => `export ANTHROPIC_BASE_URL=${origin}
export ANTHROPIC_AUTH_TOKEN=${keyIn.value}
export ANTHROPIC_MODEL=${model.value}
export ANTHROPIC_SMALL_FAST_MODEL=${model.value}
export ANTHROPIC_DEFAULT_HAIKU_MODEL=${model.value}
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
claude`,
);

const curl = computed(
  () => `curl ${base}/chat/completions \\
  -H "Content-Type: application/json" \\${needsKey.value ? `\n  -H "Authorization: Bearer ${keyIn.value}" \\` : ''}
  -d '{"model": "${model.value}", "messages": [{"role": "user", "content": "Hello!"}]}'`,
);

const env = computed(() => `export OPENAI_BASE_URL=${base}\nexport OPENAI_API_KEY=${keyIn.value}`);

const python = computed(
  () => `from openai import OpenAI

client = OpenAI(base_url="${base}", api_key="${keyIn.value}")
r = client.chat.completions.create(
    model="${model.value}",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(r.choices[0].message.content)`,
);

const js = computed(
  () => `import OpenAI from 'openai';

const client = new OpenAI({ baseURL: '${base}', apiKey: '${keyIn.value}' });
const r = await client.chat.completions.create({
  model: '${model.value}',
  messages: [{ role: 'user', content: 'Hello!' }],
});
console.log(r.choices[0].message.content);`,
);

const anthropicCurl = computed(
  () => `curl ${origin}/v1/messages \\
  -H "Content-Type: application/json" -H "anthropic-version: 2023-06-01" \\${needsKey.value ? `\n  -H "x-api-key: ${keyIn.value}" \\` : ''}
  -d '{"model": "${model.value}", "max_tokens": 256, "messages": [{"role": "user", "content": "Hello!"}]}'`,
);

const aider = computed(() => `OPENAI_API_BASE=${base} OPENAI_API_KEY=${keyIn.value} aider --model openai/${model.value}`);

async function copy(name, text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // Not a secure page: the old way.
    const t = document.createElement('textarea');
    t.value = text;
    document.body.appendChild(t);
    t.select();
    document.execCommand('copy');
    t.remove();
  }
  copied.value = name;
  setTimeout(() => copied.value === name && (copied.value = ''), 1500);
}

const snippets = computed(() => [
  {
    id: 'opencode',
    title: 'opencode',
    lang: 'opencode.json',
    text: opencode.value,
    note: 'Save as opencode.json at the root of a project, or merge into ~/.config/opencode/opencode.json for every project. Then pick the model with /models, or run: opencode run -m "oxinfer/' + model.value + '" "…". The models block lists the server’s chat models.',
  },
  {
    id: 'claude',
    title: 'Claude Code',
    lang: 'shell',
    text: claude.value,
    note: 'Claude Code speaks Anthropic’s Messages API, which this server also answers at /v1/messages (text, images, tools, thinking, streaming). Set these in the shell that starts claude (or in the "env" block of ~/.claude/settings.json). The token can be anything when the server has no API key. A small local model makes a weak coding agent: use the largest chat model the machine runs well.',
  },
  { id: 'curl', title: 'curl (OpenAI API)', lang: 'shell', text: curl.value },
  { id: 'anthropic', title: 'curl (Anthropic API)', lang: 'shell', text: anthropicCurl.value },
  { id: 'env', title: 'Environment variables', lang: 'shell', text: env.value, note: 'Read by the openai SDKs and by many tools that take an OpenAI-compatible endpoint.' },
  { id: 'python', title: 'Python (openai)', lang: 'python', text: python.value },
  { id: 'js', title: 'JavaScript (openai)', lang: 'javascript', text: js.value },
  { id: 'aider', title: 'Aider', lang: 'shell', text: aider.value },
]);
</script>

<template>
  <main class="layout">
    <aside class="side">
      <section aria-labelledby="conn-h">
        <h2 id="conn-h">This server</h2>
        <dl class="speed">
          <dt>OpenAI API</dt>
          <dd class="mono">{{ base }}</dd>
          <dt>Anthropic API</dt>
          <dd class="mono">{{ origin }}/v1/messages</dd>
          <dt>Listening on</dt>
          <dd class="mono">{{ store.status?.listen }}</dd>
          <dt>API key</dt>
          <dd>
            <template v-if="needsKey">
              <span class="mono">{{ shownKey }}</span>
              <button type="button" class="link" @click="reveal = !reveal">{{ reveal ? 'Hide' : 'Show in the snippets' }}</button>
            </template>
            <template v-else>none (any value works)</template>
          </dd>
        </dl>
        <p v-if="loopback" class="hint warn-text">
          It listens on this computer only: other machines cannot reach it. To use it from another one, start it with <code>oxinfer web --host 0.0.0.0 --api-key KEY</code> and use this machine’s address in place of {{ hostname }}.
        </p>
        <p v-else-if="localHost" class="hint">It listens beyond this computer: from another machine, use this machine’s address in place of {{ hostname }}.</p>
      </section>

      <section aria-labelledby="conn-model-h">
        <h2 id="conn-model-h">Model id</h2>
        <template v-if="models.length">
          <label class="block">
            <span class="sr-only">Model for the snippets</span>
            <select v-model="chosen" class="field">
              <option value="">{{ model }}{{ models.find((m) => m.id === model)?.loaded ? ' (loaded)' : '' }}</option>
              <option v-for="m in models.filter((x) => x.id !== model)" :key="m.id" :value="m.id">{{ m.id }}{{ m.loaded ? ' (loaded)' : '' }}</option>
            </select>
          </label>
          <p class="hint">Clients name a model by this id. Only the models listed here answer (as in Models › Models in use); one that is not loaded loads on its first request, which then waits for it. The least recently used idle models are unloaded when memory runs short.</p>
        </template>
        <p v-else class="hint">
          No chat model is set up yet, so clients have nothing to use.
          <button type="button" class="link" @click="emit('manage')">Load one in Models</button>; it then joins this list.
        </p>
      </section>

      <section aria-labelledby="conn-other-h">
        <h2 id="conn-other-h">Other clients</h2>
        <p class="hint">
          <strong>Continue</strong>: a model with <code>provider: openai</code>, <code>apiBase: {{ base }}</code>, <code>model: {{ model }}</code>.
          <strong>Open WebUI</strong>: Settings › Connections › OpenAI API, URL <code>{{ base }}</code>.
          <strong>LM Studio-style and other clients</strong>: an “OpenAI-compatible” endpoint with base URL <code>{{ base }}</code>, any key unless one is set.
        </p>
      </section>
    </aside>

    <section class="pane" aria-label="Snippets">
      <div class="stack wide-stack">
        <h2>Connect apps</h2>
        <p class="hint">Copy a snippet into the program’s configuration. The address, key and model id are this server’s.</p>
        <article v-for="s in snippets" :key="s.id" class="snippet">
          <div class="snippet-head">
            <h3>{{ s.title }}</h3>
            <span class="hint">{{ s.lang }}</span>
            <button type="button" class="quiet small" :aria-label="`Copy the ${s.title} snippet`" @click="copy(s.id, s.text)">{{ copied === s.id ? 'Copied' : 'Copy' }}</button>
          </div>
          <pre class="code-block"><code>{{ s.text }}</code></pre>
          <p v-if="s.note" class="hint">{{ s.note }}</p>
        </article>
      </div>
    </section>
  </main>
</template>

<style scoped>
.wide-stack {
  max-width: 88ch;
}
.snippet {
  padding: 0.5rem 0 1rem;
  border-bottom: 1px solid var(--line);
}
.snippet-head {
  display: flex;
  align-items: baseline;
  gap: 0.75rem;
}
.snippet-head h3 {
  margin: 0;
  font-family: var(--display);
  font-size: 1rem;
  font-weight: 650;
}
.snippet-head .quiet {
  margin: 0 0 0 auto;
}
.code-block {
  margin: 0.5rem 0 0;
  padding: 0.7rem 0.8rem;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--surface);
  overflow-x: auto;
  font-size: 0.82rem;
  line-height: 1.45;
}
.code-block code {
  padding: 0;
  border: 0;
  background: none;
  font-size: inherit;
}
.speed dd {
  overflow-wrap: anywhere;
}
.warn-text {
  color: var(--copper);
}
</style>
