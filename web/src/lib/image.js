// Image generation sessions behind one interface (see ImageStudio.vue): the Stable Diffusion
// pipeline of the wasm engine in this tab, or an image model of the native backend.
//
//   session.defaults() -> { size, steps, guidance, turbo, edit, native }
//   session.generate({ prompt, negative, steps, guidance, seed, width, height, ancestral,
//                      image, mask, strength, imageGuidance }, onStep(i, n))
//     -> { width, height, rgb } or { width, height, url }
// image / mask: { width, height, rgb, url } as ImageStudio decodes them (url: a PNG data URL).

export class WasmImage {
  constructor(engine) {
    this.engine = engine;
  }

  defaults() {
    return { ...this.engine.sdDefaults(), native: false };
  }

  async generate(p, onStep) {
    this.engine.sdSetInput(p.image, p.mask, { strength: p.strength || 0, imageGuidance: p.imageGuidance || 0 });
    return this.engine.sdGenerate(
      { prompt: p.prompt, negative: p.negative, steps: p.steps, guidance: p.guidance, seed: p.seed, width: p.width, height: p.height, ancestral: p.ancestral },
      onStep,
    );
  }
}

async function pngBlob(img) {
  return (await fetch(img.url)).blob();
}

function imageSize(url) {
  return new Promise((resolve, reject) => {
    const el = new Image();
    el.onload = () => resolve({ width: el.naturalWidth, height: el.naturalHeight });
    el.onerror = () => reject(new Error('the server returned an image the browser cannot show'));
    el.src = url;
  });
}

/// An image model of the native backend: /v1/images/generations, /v1/images/edits with an input.
export class HttpImage {
  constructor(api, model) {
    this.api = api;
    this.model = model;
    this.ctl = null;
  }

  defaults() {
    // The server applies each pipeline's own defaults when steps and guidance are left out.
    return { size: 512, steps: null, guidance: null, turbo: false, edit: 'any', native: true };
  }

  async generate(p) {
    this.ctl = new AbortController();
    const opt = (k, v) => (v === null || v === undefined || v === '' || Number.isNaN(v) ? {} : { [k]: v });
    let r;
    if (p.image) {
      const f = new FormData();
      f.append('model', this.model);
      f.append('prompt', p.prompt);
      f.append('image', await pngBlob(p.image), 'image.png');
      if (p.mask) f.append('mask', await pngBlob(p.mask), 'mask.png');
      f.append('response_format', 'b64_json');
      for (const [k, v] of Object.entries({ ...opt('negative_prompt', p.negative), ...opt('steps', p.steps), ...opt('guidance', p.guidance), ...opt('seed', p.seed), ...opt('strength', p.strength) })) {
        f.append(k, String(v));
      }
      r = await this.api.raw('v1/images/edits', { method: 'POST', body: f, signal: this.ctl.signal });
    } else {
      r = await this.api.raw('v1/images/generations', {
        method: 'POST',
        json: {
          model: this.model,
          prompt: p.prompt,
          size: `${p.width}x${p.height}`,
          n: 1,
          response_format: 'b64_json',
          ...opt('negative_prompt', p.negative),
          ...opt('steps', p.steps),
          ...opt('guidance', p.guidance),
          ...opt('seed', p.seed),
        },
        signal: this.ctl.signal,
      });
    }
    const v = await r.json();
    if (v.error) throw new Error(v.error.message);
    const b64 = v.data?.[0]?.b64_json;
    const url = b64 ? `data:image/png;base64,${b64}` : v.data?.[0]?.url;
    if (!url) throw new Error('the server returned no image');
    return { ...(await imageSize(url)), url };
  }

  stop() {
    this.ctl?.abort();
  }
}
