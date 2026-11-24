<script setup>
// "Add to opencode" / "Update" / "Remove from opencode" for one chat model, when the server
// found opencode's configuration (GET /api/web/opencode); nothing otherwise.
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
