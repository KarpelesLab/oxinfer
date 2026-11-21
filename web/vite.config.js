import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { createReadStream, readFileSync, statSync } from 'node:fs';
import { join, normalize, sep } from 'node:path';

// GitHub Pages serves a project site under /<repo>/. Override with PAGES_BASE ("/" for local use).
const base = process.env.PAGES_BASE || '/oxinfer/';

// Local testing: OXINFER_MODELS=~/.cache/oxinfer/hub npm run dev serves that directory at
// <base>models/, so "./models/<folder>" works as a model source without uploading anything.
function localModels() {
  const root = process.env.OXINFER_MODELS;
  const mount = `${base}models/`;
  const handler = (req, res, next) => {
    if (!root || !req.url.startsWith(mount)) return next();
    const rel = decodeURIComponent(req.url.slice(mount.length).split('?')[0]);
    const file = normalize(join(root, rel));
    if (!file.startsWith(normalize(root) + sep)) return next();
    let st;
    try {
      st = statSync(file);
    } catch {
      res.statusCode = 404;
      return res.end('not found');
    }
    if (!st.isFile()) return next();
    res.setHeader('content-length', String(st.size));
    res.setHeader('content-type', file.endsWith('.json') ? 'application/json' : 'application/octet-stream');
    createReadStream(file).pipe(res);
  };
  return {
    name: 'oxinfer-local-models',
    // Block bodies: a function returned from these hooks would be run as a post-middleware hook.
    configureServer(s) {
      s.middlewares.use(handler);
    },
    configurePreviewServer(s) {
      s.middlewares.use(handler);
    },
  };
}

// `npm run build:native` (`--mode native`): the GUI `oxinfer web` embeds, written to
// ../src/server/webui (committed, so that building oxinfer needs no node), with relative paths
// and without the .wasm (that page drives the native engine over HTTP).
function faviconOnly() {
  return {
    name: 'oxinfer-favicon',
    generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'favicon.svg', source: readFileSync(new URL('./public/favicon.svg', import.meta.url)) });
    },
  };
}

export default defineConfig(({ mode }) => {
  const native = mode === 'native';
  return {
    base: native ? './' : base,
    plugins: [vue(), localModels(), ...(native ? [faviconOnly()] : [])],
    build: {
      target: 'es2022',
      // The .wasm lives in public/ and is fetched at run time, not bundled.
      assetsInlineLimit: 0,
      ...(native ? { outDir: '../src/server/webui', emptyOutDir: true, copyPublicDir: false } : {}),
    },
  };
});
