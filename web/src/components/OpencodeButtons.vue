<script setup>
// "Add to opencode" / "Update" / "Remove from opencode" for one chat model, when the server
// found opencode's configuration (GET /api/web/opencode); nothing otherwise. "with a profile…"
// adds the model as MODEL@PROFILE (docs/learning.md), with the plugin that reports outcomes.
import { computed, onMounted, watch } from 'vue';
import { store, opencode, opencodeOf, opencodeChange, refreshOpencode } from '../lib/native.js';

const props = defineProps({ id: { type: String, required: true } });

const row = computed(() => opencodeOf(props.id));
const busy = computed(() => !!opencode.busy[props.id]);
const writable = computed(() => !!opencode.status?.writable);
const why = computed(() => {
  const s = opencode.status;
  if (s?.error) return `opencode's configuration cannot be read: ${s.error}`;
  if (!writable.value) return 'The management API is read-only on this server';
  return '';
});

// What the entry says, for the button's tooltip.
const summary = computed(() => {
  const c = row.value?.capabilities;
  if (!c) return '';
  const what = [c.vision && 'images', c.thinking && 'thinking', c.tools && 'tools'].filter(Boolean);
  const ctx = row.value.entry?.limit?.context;
  return `${what.length ? what.join(', ') : 'text only'}${ctx ? `, ${ctx.toLocaleString()} tokens of context` : ''}${row.value.source === 'files' ? ' (from its files: load it for the exact figures)' : ''}`;
});

// The model under a profile: opencode's requests for it are logged and run with the profile's
// adapter; the server also writes the plugin that reports test, build and lint results.
function addProfile() {
  const name = window.prompt(
    `Profile to run ${props.id} under in opencode (letters, digits, _ . -), for example the project's name.\n` +
      'Its requests are logged on this server, and a plugin written to opencode\'s plugins directory reports test, build and lint results.',
  );
  if (name?.trim()) opencodeChange(props.id, true, name.trim());
}

// A load changes what the model is known to do (from its files to the engine's own figures).
const state = computed(() => store.status?.models?.find((m) => m.id === props.id)?.state);
watch(state, (s, old) => {
  if (s === 'ready' && old !== 'busy') refreshOpencode(true);
});
onMounted(() => refreshOpencode());
</script>

<template>
  <span v-if="row" class="opencode">
    <span v-if="row.up_to_date" class="badge good" :title="`In opencode as oxinfer/${id}: ${summary}`">In opencode</span>
    <span v-else-if="row.present" class="badge warn" title="opencode's entry differs from what oxinfer knows of the model now">opencode: outdated</span>
    <button v-if="!row.present" type="button" class="link" :disabled="busy || !writable" :title="why || `Adds oxinfer/${id} to ${opencode.status.path}: ${summary}`" @click="opencodeChange(id, true)">{{ busy ? 'Adding…' : 'Add to opencode' }}</button>
    <button v-else-if="!row.up_to_date" type="button" class="link" :disabled="busy || !writable" :title="why || `Rewrites its entry: ${summary}`" @click="opencodeChange(id, true)">{{ busy ? 'Updating…' : 'Update' }}</button>
    <button v-if="row.present" type="button" class="link danger" :disabled="busy || !writable" :title="why || `Removes oxinfer/${id} from ${opencode.status.path}`" @click="opencodeChange(id, false)">Remove from opencode</button>
    <template v-for="p in row.profiles || []" :key="p.id">
      <span class="badge" :class="p.up_to_date ? 'good' : 'warn'" :title="p.up_to_date ? `In opencode as oxinfer/${p.id}` : 'opencode\'s entry differs from what oxinfer knows of the model now'">{{ p.id }}</span>
      <button v-if="!p.up_to_date" type="button" class="link" :disabled="busy || !writable" :title="why || 'Rewrites its entry'" @click="opencodeChange(id, true, p.profile)">Update</button>
      <button type="button" class="link danger" :disabled="busy || !writable" :title="why || `Removes oxinfer/${p.id} from ${opencode.status.path}`" @click="opencodeChange(id, false, p.profile)">Remove</button>
    </template>
    <button v-if="opencode.status.profiles" type="button" class="link" :disabled="busy || !writable" :title="why || `Adds oxinfer/${id}@PROFILE to ${opencode.status.path}: requests logged under the profile, outcomes reported by a plugin`" @click="addProfile">with a profile…</button>
  </span>
</template>

<style scoped>
.opencode {
  display: inline-flex;
  flex-wrap: wrap;
  gap: 0.9rem;
  align-items: baseline;
}
</style>
