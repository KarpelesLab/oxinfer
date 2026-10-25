// Host side of the oxinfer WebAssembly module (browser and Deno, ES module, no build step).
//
// The module is plain C-ABI wasm (see src/web/mod.rs, built with `cargo build-web`). This file provides its imports: a log sink, a
// clock, randomness, and the WebGPU backend (`gpu_upload` / `gpu_matmul`). GPU readback is
// asynchronous while the module's call is synchronous, so `gpu_matmul` is wrapped with
// `WebAssembly.Suspending` and the exports that can reach it with `WebAssembly.promising` (JSPI).
// Without JSPI or WebGPU the module still runs, on the CPU.

const TILE = 64;

// ---- WGSL ----------------------------------------------------------------------------------------
// y[t][n] = sum_k x[t][k] * w[n][k]; one workgroup computes a 64 (n) x 64 (t) tile with 256 threads,
// each 4x4 outputs; weights are dequantized to f32 in workgroup memory, 32 k per step.
// Matmul shaders read x densely ([t][n_in]); convolution shaders gather each pixel's k x k
// neighbourhood from the channels-last image (implicit GEMM; cin % 32 == 0, so a 32-wide block of
// K is one tap's contiguous channel slice), with zeros outside the image.
const X_DENSE = `
    let xb = (t0 + lr) * p.n_in + k0 + lc * 8u;
    for (var j = 0u; j < 8u; j++) { put_x(lc * 8u + j, lr, x[xb + j]); }`;
const X_CONV = `
    let tok = t0 + lr;
    let tap = k0 / p.cin;
    let ci0 = k0 - tap * p.cin;
    let iy = i32(tok / p.ow) * i32(p.stride) + i32(tap / p.k) - i32(p.padk);
    let ix = i32(tok % p.ow) * i32(p.stride) + i32(tap % p.k) - i32(p.padk);
    if (tok < p.npix && iy >= 0 && ix >= 0 && iy < i32(p.h) && ix < i32(p.w)) {
      let pin = u32(iy) * p.w + u32(ix);
      let xb = pin * p.cin + ci0 + lc * 8u;
      for (var j = 0u; j < 8u; j++) { put_x(lc * 8u + j, lr, norm1(x[xb + j], ci0 + lc * 8u + j, pin)); }
    } else {
      for (var j = 0u; j < 8u; j++) { put_x(lc * 8u + j, lr, 0.0); }
    }`;
// gk: affine Q4 only, blocks of 32 per offset group.
const P_DENSE = 'struct P { n_out: u32, n_in: u32, t_pad: u32, gk: u32 }';
// Convolutions may normalize their input as they load it (mode 1 group norm, 2 layer norm; st holds
// (mean, 1/std) per group or input pixel, gb gamma then beta), then apply SiLU.
const P_CONV = `struct P { n_out: u32, n_in: u32, t_pad: u32, pad: u32, h: u32, w: u32, cin: u32, k: u32, stride: u32, ow: u32, npix: u32, padk: u32, mode: u32, silu: u32, cpg: u32, pad2: u32 }
@group(0) @binding(5) var<storage, read> st: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> gb: array<f32>;
fn norm1(v: f32, ch: u32, pin: u32) -> f32 {
  if (p.mode == 0u) { return v; }
  var m: vec2<f32>;
  if (p.mode == 1u) { m = st[ch / p.cpg]; } else { m = st[pin]; }
  var y = (v - m.x) * m.y * gb[ch] + gb[p.cin + ch];
  if (p.silu != 0u) { y = y / (1.0 + exp(-y)); }
  return y;
}`;

const WGSL_COMMON = `
__P__
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> s: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

// [k][n] and [k][t], scalar: assigning one component of a workgroup vec4 through a dynamic index
// is miscompiled by some implementations (only lane 0 lands), so the tiles stay scalar arrays and
// the inner loop assembles vec4s from four adjacent loads.
var<workgroup> Ws: array<f32, ${32 * 64}>;
var<workgroup> Xs: array<f32, ${32 * 64}>;

fn put_w(k: u32, n: u32, v: f32) { Ws[k * 64u + n] = v; }
fn put_x(k: u32, t: u32, v: f32) { Xs[k * 64u + t] = v; }

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n0 = wg.x * 64u;
  let t0 = wg.y * 64u;
  let tn = li % 16u;
  let tt = li / 16u;
  let lr = li / 4u;   // row (n or t) this thread loads
  let lc = li % 4u;   // which 8 of the 32 k
  var a0 = vec4<f32>(0.0);
  var a1 = vec4<f32>(0.0);
  var a2 = vec4<f32>(0.0);
  var a3 = vec4<f32>(0.0);
  for (var k0 = 0u; k0 < p.n_in; k0 += 32u) {
    load_w(n0 + lr, k0, lr, lc);
__X__
    workgroupBarrier();
    for (var k = 0u; k < 32u; k++) {
      let wi = k * 64u + tn * 4u;
      let xi = k * 64u + tt * 4u;
      let wv = vec4<f32>(Ws[wi], Ws[wi + 1u], Ws[wi + 2u], Ws[wi + 3u]);
      let xv = vec4<f32>(Xs[xi], Xs[xi + 1u], Xs[xi + 2u], Xs[xi + 3u]);
      a0 += wv.x * xv;
      a1 += wv.y * xv;
      a2 += wv.z * xv;
      a3 += wv.w * xv;
    }
    workgroupBarrier();
  }
  let nb = n0 + tn * 4u;
  let tb = t0 + tt * 4u;
  for (var j = 0u; j < 4u; j++) {
    let o = (tb + j) * p.n_out + nb;
    y[o] = a0[j];
    y[o + 1u] = a1[j];
    y[o + 2u] = a2[j];
    y[o + 3u] = a3[j];
  }
}
`;

// Statistics for fused normalization (two passes: mean, then mean squared deviation) and the
// attention softmax; reductions go through workgroup memory.
const WGSL_GROUP_STATS = `
struct GS { n: u32, c: u32, groups: u32, eps: f32 }
@group(0) @binding(0) var<uniform> p: GS;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> st: array<vec2<f32>>;
var<workgroup> red: array<f32, 256>;
fn reduce(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var o = 128u; o > 0u; o = o >> 1u) {
    if (li < o) { red[li] += red[li + o]; }
    workgroupBarrier();
  }
  let t = red[0];
  workgroupBarrier();
  return t;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let g = wg.x;
  let cpg = p.c / p.groups;
  let total = p.n * cpg;
  var s = 0.0;
  for (var i = li; i < total; i += 256u) { s += x[(i / cpg) * p.c + g * cpg + i % cpg]; }
  let mean = reduce(li, s) / f32(total);
  var v = 0.0;
  for (var i = li; i < total; i += 256u) { let d = x[(i / cpg) * p.c + g * cpg + i % cpg] - mean; v += d * d; }
  let vr = reduce(li, v) / f32(total);
  if (li == 0u) { st[g] = vec2<f32>(mean, inverseSqrt(vr + p.eps)); }
}`;
const WGSL_ROW_STATS = `
struct RS { c: u32, eps: f32, rows: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: RS;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> st: array<vec2<f32>>;
var<workgroup> red: array<f32, 64>;
fn reduce(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var o = 32u; o > 0u; o = o >> 1u) {
    if (li < o) { red[li] += red[li + o]; }
    workgroupBarrier();
  }
  let t = red[0];
  workgroupBarrier();
  return t;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wg.x + wg.y * 32768u;
  let base = min(r, p.rows - 1u) * p.c;
  var s = 0.0;
  for (var i = li; i < p.c; i += 64u) { s += x[base + i]; }
  let mean = reduce(li, s) / f32(p.c);
  var v = 0.0;
  for (var i = li; i < p.c; i += 64u) { let d = x[base + i] - mean; v += d * d; }
  let vr = reduce(li, v) / f32(p.c);
  if (li == 0u && r < p.rows) { st[r] = vec2<f32>(mean, inverseSqrt(vr + p.eps)); }
}`;
const WGSL_SOFTMAX = `
struct SM { cols: u32, ld: u32, scale: f32, causal: u32 }
@group(0) @binding(0) var<uniform> p: SM;
@group(0) @binding(1) var<storage, read_write> s: array<f32>;
var<workgroup> red: array<f32, 64>;
fn reduce(li: u32, v: f32, is_max: bool) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var o = 32u; o > 0u; o = o >> 1u) {
    if (li < o) { red[li] = select(red[li] + red[li + o], max(red[li], red[li + o]), is_max); }
    workgroupBarrier();
  }
  let t = red[0];
  workgroupBarrier();
  return t;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wg.x;
  let base = r * p.ld;
  let lim = select(p.cols, min(r + 1u, p.cols), p.causal != 0u);
  var m = -3.0e38;
  for (var i = li; i < lim; i += 64u) { m = max(m, s[base + i] * p.scale); }
  m = reduce(li, m, true);
  var sum = 0.0;
  for (var i = li; i < lim; i += 64u) { let e = exp(s[base + i] * p.scale - m); s[base + i] = e; sum += e; }
  let inv = 1.0 / reduce(li, sum, false);
  for (var i = li; i < p.ld; i += 64u) { s[base + i] = select(0.0, s[base + i] * inv, i < lim); }
}`;

const WGSL_LOAD = {
  // bf16: two per u32, low half first.
  0: `
fn load_w(row: u32, k0: u32, lr: u32, lc: u32) {
  let e = (row * p.n_in + k0 + lc * 8u) / 2u;
  for (var i = 0u; i < 4u; i++) {
    let u = w[e + i];
    put_w(lc * 8u + 2u * i, lr, bitcast<f32>(u << 16u));
    put_w(lc * 8u + 2u * i + 1u, lr, bitcast<f32>(u & 0xffff0000u));
  }
}`,
  // q8: int8 x4 per u32, one f32 scale per 32.
  1: `
fn load_w(row: u32, k0: u32, lr: u32, lc: u32) {
  let d = bitcast<f32>(s[row * (p.n_in / 32u) + k0 / 32u]);
  let e = (row * p.n_in + k0 + lc * 8u) / 4u;
  for (var i = 0u; i < 2u; i++) {
    let u = w[e + i];
    put_w(lc * 8u + 4u * i, lr, d * f32(i32(u << 24u) >> 24u));
    put_w(lc * 8u + 4u * i + 1u, lr, d * f32(i32(u << 16u) >> 24u));
    put_w(lc * 8u + 4u * i + 2u, lr, d * f32(i32(u << 8u) >> 24u));
    put_w(lc * 8u + 4u * i + 3u, lr, d * f32(i32(u) >> 24u));
  }
}`,
  // f32 rows (attention keys and transposed values), one value per u32.
  3: `
fn load_w(row: u32, k0: u32, lr: u32, lc: u32) {
  let e = row * p.n_in + k0 + lc * 8u;
  for (var j = 0u; j < 8u; j++) { put_w(lc * 8u + j, lr, bitcast<f32>(w[e + j])); }
}`,
  // q4_0: a block of 32 is 16 bytes, byte j holds elements j (low nibble) and j+16 (high); f16 scale.
  2: `
fn load_w(row: u32, k0: u32, lr: u32, lc: u32) {
  let b = row * (p.n_in / 32u) + k0 / 32u;
  let d = unpack2x16float(s[b / 2u])[b % 2u];
  let u = w[b * 4u + lc];
  for (var i = 0u; i < 4u; i++) {
    let byte = (u >> (8u * i)) & 0xffu;
    put_w(lc * 4u + i, lr, d * (f32(byte & 15u) - 8.0));
    put_w(16u + lc * 4u + i, lr, d * (f32(byte >> 4u) - 8.0));
  }
}`,
  // affine q4 (MLX): q4_0 plus a bf16 offset per group of p.gk blocks, stored after the f16
  // scales (which are padded to a whole u32).
  4: `
fn load_w(row: u32, k0: u32, lr: u32, lc: u32) {
  let bpr = p.n_in / 32u;
  let b = row * bpr + k0 / 32u;
  let d = unpack2x16float(s[b / 2u])[b % 2u];
  let oi = (p.n_out * bpr + 1u) / 2u * 2u + row * (bpr / p.gk) + (k0 / 32u) / p.gk;
  let m = bitcast<f32>(((s[oi / 2u] >> (16u * (oi % 2u))) & 0xffffu) << 16u);
  let u = w[b * 4u + lc];
  for (var i = 0u; i < 4u; i++) {
    let byte = (u >> (8u * i)) & 0xffu;
    put_w(lc * 4u + i, lr, d * (f32(byte & 15u) - 8.0) + m);
    put_w(16u + lc * 4u + i, lr, d * (f32(byte >> 4u) - 8.0) + m);
  }
}`,
};

// ---- WebGPU host ---------------------------------------------------------------------------------

export class WebGpuHost {
  constructor(device, adapter) {
    this.device = device;
    this.adapter = adapter;
    this.weights = [];
    this.pipelines = {};
    this.xBuf = null;
    this.yBufs = new Map();   // size -> buffer
    this.stage = [];          // readback buffers, by job slot
    this.params = new Map();  // key -> uniform buffer
    this.bytesUploaded = 0;
    this.maxBinding = device.limits.maxStorageBufferBindingSize;
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.dummy = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    // Convolutions also bind the normalization statistics and gamma/beta.
    this.convLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.bufs = {};
  }

  /// A growable storage buffer by name (contents are not preserved when it grows).
  buf(name, bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
    const b = this.bufs[name];
    const size = Math.max(256, Math.ceil(bytes / 256) * 256);
    if (b && b.size >= size) return b;
    if (b) b.destroy();
    return (this.bufs[name] = this.device.createBuffer({ size, usage }));
  }

  uniform(values, isFloat) {
    const b = this.device.createBuffer({ size: Math.max(16, values.length * 4), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const data = new ArrayBuffer(values.length * 4);
    const u = new Uint32Array(data);
    const f = new Float32Array(data);
    values.forEach((v, i) => (isFloat?.[i] ? (f[i] = v) : (u[i] = v)));
    this.device.queue.writeBuffer(b, 0, data);
    (this.temps ||= []).push(b);
    return b;
  }

  /// Destroys the per-call uniform buffers once the work using them has completed.
  releaseTemps() {
    for (const b of this.temps || []) b.destroy();
    this.temps = [];
  }

  auxPipeline(name, code) {
    if (!this.pipelines[name]) {
      this.pipelines[name] = this.device.createComputePipeline({ layout: 'auto', compute: { module: this.device.createShaderModule({ code }), entryPoint: 'main' } });
    }
    return this.pipelines[name];
  }

  static async create(log = () => {}) {
    if (typeof navigator === 'undefined' || !navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    const want = ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeWorkgroupStorageSize'];
    const requiredLimits = {};
    for (const k of want) requiredLimits[k] = adapter.limits[k];
    const device = await adapter.requestDevice({ requiredLimits });
    device.lost.then((info) => log(`[webgpu: device lost: ${info.message}]`));
    const info = adapter.info || {};
    const host = new WebGpuHost(device, adapter);
    host.name = info.description || info.device || [info.vendor, info.architecture].filter(Boolean).join(' ') || 'GPU';
    log(`[webgpu: ${host.name}; max binding ${(device.limits.maxStorageBufferBindingSize / 2 ** 30).toFixed(1)} GiB]`);
    return host;
  }

  pipeline(kind, conv = false) {
    const key = `${kind}${conv ? 'c' : ''}`;
    if (!this.pipelines[key]) {
      const code = WGSL_LOAD[kind] + WGSL_COMMON.replace('__P__', conv ? P_CONV : P_DENSE).replace('__X__', conv ? X_CONV : X_DENSE);
      const module = this.device.createShaderModule({ code });
      this.pipelines[key] = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [conv ? this.convLayout : this.layout] }),
        compute: { module, entryPoint: 'main' },
      });
    }
    return this.pipelines[key];
  }

  /// Attention over per-head padded operands (views of module memory, copied by writeBuffer):
  /// q [heads][tqp][dhp], k [heads][tkp][dhp], vt [heads][dho][tkp]; resolves after
  /// `writeO(Float32Array [heads][tqp][dho])`.
  async attention(q, k, vt, d, writeO) {
    const { heads, tqp, tkp, dhp, dho, tk, causal, scale } = d;
    const qb = this.buf('aq', q.byteLength);
    const kb = this.buf('ak', k.byteLength);
    const vb = this.buf('av', vt.byteLength);
    const sb = this.buf('asc', tqp * tkp * 4);
    const ob = this.buf('ao', heads * tqp * dho * 4);
    this.device.queue.writeBuffer(qb, 0, q.buffer, q.byteOffset, q.byteLength);
    this.device.queue.writeBuffer(kb, 0, k.buffer, k.byteOffset, k.byteLength);
    this.device.queue.writeBuffer(vb, 0, vt.buffer, vt.byteOffset, vt.byteLength);
    const gemm = this.pipeline(3, false);
    const smPipe = this.auxPipeline('softmax', WGSL_SOFTMAX);
    const smU = this.uniform([tk, tkp, scale, causal], [false, false, true, false]);
    const smBg = this.device.createBindGroup({ layout: smPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: smU } }, { binding: 1, resource: { buffer: sb, size: tqp * tkp * 4 } }] });
    const pS = this.paramsFor(tkp, dhp, tqp);
    const pO = this.paramsFor(dho, tkp, tqp);
    const enc = this.device.createCommandEncoder();
    for (let h = 0; h < heads; h++) {
      // Scores: keys as the "weights", queries as the input.
      let pass = enc.beginComputePass();
      pass.setPipeline(gemm);
      pass.setBindGroup(0, this.device.createBindGroup({ layout: this.layout, entries: [
        { binding: 0, resource: { buffer: pS } },
        { binding: 1, resource: { buffer: kb, offset: h * tkp * dhp * 4, size: tkp * dhp * 4 } },
        { binding: 2, resource: { buffer: this.dummy } },
        { binding: 3, resource: { buffer: qb, offset: h * tqp * dhp * 4, size: tqp * dhp * 4 } },
        { binding: 4, resource: { buffer: sb, size: tqp * tkp * 4 } },
      ] }));
      pass.dispatchWorkgroups(tkp / TILE, tqp / TILE, 1);
      pass.end();
      pass = enc.beginComputePass();
      pass.setPipeline(smPipe);
      pass.setBindGroup(0, smBg);
      pass.dispatchWorkgroups(tqp);
      pass.end();
      // Output: transposed values as the "weights", probabilities as the input.
      pass = enc.beginComputePass();
      pass.setPipeline(gemm);
      pass.setBindGroup(0, this.device.createBindGroup({ layout: this.layout, entries: [
        { binding: 0, resource: { buffer: pO } },
        { binding: 1, resource: { buffer: vb, offset: h * dho * tkp * 4, size: dho * tkp * 4 } },
        { binding: 2, resource: { buffer: this.dummy } },
        { binding: 3, resource: { buffer: sb, size: tqp * tkp * 4 } },
        { binding: 4, resource: { buffer: ob, offset: h * tqp * dho * 4, size: tqp * dho * 4 } },
      ] }));
      pass.dispatchWorkgroups(dho / TILE, tqp / TILE, 1);
      pass.end();
    }
    const bytes = heads * tqp * dho * 4;
    const stage = this.buf('ao_stage', bytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    enc.copyBufferToBuffer(ob, 0, stage, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ, 0, bytes);
    writeO(new Float32Array(stage.getMappedRange(0, bytes)));
    stage.unmap();
    this.releaseTemps();
  }

  /// k x k convolution: `x` is the channels-last image ([h*w, cin] f32 view, copied by
  /// writeBuffer at once), `d` its geometry; resolves after `writeY(Float32Array)`.
  async conv(x, d, writeY) {
    const wt = this.weights[d.handle];
    if (!wt || wt.nOut !== d.nOut || wt.nIn !== d.nIn) throw new Error(`bad conv: handle ${d.handle}`);
    const t = d.oh * d.ow;
    const tPad = Math.ceil(t / TILE) * TILE;
    const xBytes = Math.max(16, x.byteLength);
    if (!this.cxBuf || this.cxBuf.size < xBytes) {
      if (this.cxBuf) this.cxBuf.destroy();
      this.cxBuf = this.device.createBuffer({ size: xBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    this.device.queue.writeBuffer(this.cxBuf, 0, x.buffer, x.byteOffset, x.byteLength);
    if (!this.convParams) this.convParams = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.convParams, 0, new Uint32Array([d.nOut, d.nIn, tPad, 0, d.h, d.w, d.cin, d.k, d.stride, d.ow, t, (d.k / 2) | 0, d.mode || 0, d.silu || 0, d.cpg || 1, 0]));
    const ySize = tPad * d.nOut * 4;
    const y = this.yFor(ySize, 0);
    const st = this.stageFor(t * d.nOut * 4, 0);
    const statsBuf = this.buf('stats', Math.max(1, d.nStats || 1) * 8);
    const gbBuf = this.buf('gb', 2 * d.cin * 4);
    const enc = this.device.createCommandEncoder();
    if (d.mode) {
      // Statistics first, in the same submission (passes run in order).
      const gbArr = new Float32Array(2 * d.cin);
      gbArr.set(d.gamma, 0);
      gbArr.set(d.beta, d.cin);
      this.device.queue.writeBuffer(gbBuf, 0, gbArr);
      const sp = enc.beginComputePass();
      if (d.mode === 1) {
        const pipe = this.auxPipeline('group_stats', WGSL_GROUP_STATS);
        const u = this.uniform([d.h * d.w, d.cin, d.nStats, d.eps], [false, false, false, true]);
        sp.setPipeline(pipe);
        sp.setBindGroup(0, this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: u } }, { binding: 1, resource: { buffer: this.cxBuf, size: xBytes } }, { binding: 2, resource: { buffer: statsBuf } }] }));
        sp.dispatchWorkgroups(d.nStats);
      } else {
        const rows = d.h * d.w;
        const pipe = this.auxPipeline('row_stats', WGSL_ROW_STATS);
        const u = this.uniform([d.cin, d.eps, rows, 0], [false, true, false, false]);
        sp.setPipeline(pipe);
        sp.setBindGroup(0, this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: u } }, { binding: 1, resource: { buffer: this.cxBuf, size: xBytes } }, { binding: 2, resource: { buffer: statsBuf } }] }));
        sp.dispatchWorkgroups(Math.min(rows, 32768), Math.ceil(rows / 32768));
      }
      sp.end();
    }
    const bg = this.device.createBindGroup({
      layout: this.convLayout,
      entries: [
        { binding: 0, resource: { buffer: this.convParams } },
        { binding: 1, resource: { buffer: wt.w } },
        { binding: 2, resource: { buffer: wt.s } },
        { binding: 3, resource: { buffer: this.cxBuf, size: xBytes } },
        { binding: 4, resource: { buffer: y, size: ySize } },
        { binding: 5, resource: { buffer: statsBuf } },
        { binding: 6, resource: { buffer: gbBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline(wt.kind, true));
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(d.nOut / TILE, tPad / TILE, 1);
    pass.end();
    enc.copyBufferToBuffer(y, 0, st, 0, t * d.nOut * 4);
    this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ, 0, t * d.nOut * 4);
    writeY(new Float32Array(st.getMappedRange(0, t * d.nOut * 4)));
    st.unmap();
    this.releaseTemps();
  }

  storage(bytes, extra = 0) {
    const size = Math.max(16, Math.ceil(bytes.byteLength / 4) * 4);
    const buf = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra });
    this.device.queue.writeBuffer(buf, 0, bytes.buffer, bytes.byteOffset, bytes.byteLength);
    return buf;
  }

  /// Returns a handle, or -1 if the matrix cannot live on this device.
  upload(kind, data, scales, nOut, nIn) {
    if (data.byteLength > this.maxBinding || (scales && scales.byteLength > this.maxBinding)) return -1;
    try {
      const w = this.storage(data);
      const s = scales ? this.storage(scales) : this.dummy;
      // Affine q4: the offsets follow the f16 scales; their count gives the group size.
      let gk = 0;
      if (kind === 4) {
        const bpr = nIn / 32;
        const gpr = (scales.byteLength / 2 - Math.ceil((nOut * bpr) / 2) * 2) / nOut;
        gk = bpr / gpr;
        if (!Number.isInteger(gk) || gk < 1) return -1;
      }
      this.weights.push({ kind, w, s, nOut, nIn, gk });
      this.bytesUploaded += data.byteLength + (scales ? scales.byteLength : 0);
      return this.weights.length - 1;
    } catch (e) {
      return -1;
    }
  }

  paramsFor(nOut, nIn, tPad, gk = 0) {
    const key = `${nOut},${nIn},${tPad},${gk}`;
    let b = this.params.get(key);
    if (!b) {
      b = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(b, 0, new Uint32Array([nOut, nIn, tPad, gk]));
      this.params.set(key, b);
    }
    return b;
  }

  yFor(size, slot) {
    // One output buffer per slot (jobs in a batch run concurrently), grown as needed.
    const cur = this.yBufs.get(slot);
    if (cur && cur.size >= size) return cur;
    if (cur) cur.destroy();
    const b = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.yBufs.set(slot, b);
    return b;
  }

  stageFor(size, slot) {
    const cur = this.stage[slot];
    if (cur && cur.size >= size) return cur;
    if (cur) cur.destroy();
    const b = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.stage[slot] = b;
    return b;
  }

  /// x: Float32Array [t][nIn] (copied already); jobs: [{handle, nOut, dst(Float32Array writer)}].
  /// Resolves when every y has been written through `writeY(slot, Float32Array)`.
  async matmul(x, t, nIn, jobs, writeY) {
    const tPad = Math.ceil(t / TILE) * TILE;
    const xBytes = tPad * nIn * 4;
    if (!this.xBuf || this.xBuf.size < xBytes) {
      if (this.xBuf) this.xBuf.destroy();
      this.xBuf = this.device.createBuffer({ size: xBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    this.device.queue.writeBuffer(this.xBuf, 0, x.buffer, x.byteOffset, t * nIn * 4);
    if (tPad > t) this.device.queue.writeBuffer(this.xBuf, t * nIn * 4, new Float32Array((tPad - t) * nIn));
    const enc = this.device.createCommandEncoder();
    const outs = [];
    jobs.forEach((j, slot) => {
      const wt = this.weights[j.handle];
      if (!wt || wt.nIn !== nIn || wt.nOut !== j.nOut) throw new Error(`bad job: handle ${j.handle}`);
      const ySize = tPad * j.nOut * 4;
      const y = this.yFor(ySize, slot);
      const st = this.stageFor(t * j.nOut * 4, slot);
      const bg = this.device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: this.paramsFor(j.nOut, nIn, tPad, wt.gk) } },
          { binding: 1, resource: { buffer: wt.w } },
          { binding: 2, resource: { buffer: wt.s } },
          { binding: 3, resource: { buffer: this.xBuf, size: xBytes } },
          { binding: 4, resource: { buffer: y, size: ySize } },
        ],
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeline(wt.kind));
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(j.nOut / TILE, tPad / TILE, 1);
      pass.end();
      enc.copyBufferToBuffer(y, 0, st, 0, t * j.nOut * 4);
      outs.push({ st, bytes: t * j.nOut * 4, slot });
    });
    this.device.queue.submit([enc.finish()]);
    await Promise.all(outs.map((o) => o.st.mapAsync(GPUMapMode.READ, 0, o.bytes)));
    for (const o of outs) {
      writeY(o.slot, new Float32Array(o.st.getMappedRange(0, o.bytes)));
      o.st.unmap();
    }
  }
}

// ---- module wrapper ------------------------------------------------------------------------------

const enc = new TextEncoder();
const dec = new TextDecoder();

export const hasJspi = typeof WebAssembly.Suspending === 'function' && typeof WebAssembly.promising === 'function';

export class Oxinfer {
  /// `source`: the .wasm bytes, a Response, or a URL string. `gpu`: a WebGpuHost, null for CPU only.
  /// `jspi: false` forces the plain synchronous exports (CPU only), as on engines without JSPI.
  static async create(source, { gpu = null, log = (s) => console.error(s), jspi = hasJspi } = {}) {
    const g = new Oxinfer();
    g.log = log;
    jspi = jspi && hasJspi;
    g.gpu = jspi ? gpu : null;
    if (gpu && !jspi) log('[webgpu: no JSPI (WebAssembly.Suspending) on this engine; using CPU]');
    g.hostFiles = [];
    const env = {
      host_log: (p, n) => log(g.str(p, n)),
      host_read: (id, off, p, n) => {
        const src = g.hostFiles[id];
        if (!src || off + n > src.byteLength) return 1;
        g.u8(p, n).set(src.subarray(off, off + n));
        return 0;
      },
      host_release: (id) => {
        g.hostFiles[id] = null;
      },
      host_now_ms: () => performance.now(),
      host_random: (p, n) => {
        const v = g.u8(p, n);
        for (let i = 0; i < n; i += 65536) crypto.getRandomValues(v.subarray(i, Math.min(n, i + 65536)));
      },
      gpu_available: () => (g.gpu ? 1 : 0),
      gpu_upload: (kind, dp, dn, sp, sn, nOut, nIn) => {
        if (!g.gpu) return -1;
        return g.gpu.upload(kind, g.u8(dp, dn), sn ? g.u8(sp, sn) : null, nOut, nIn);
      },
      host_progress: (step, total) => g.onProgress?.(step, total),
      gpu_conv: async (xp, dp) => {
        if (!g.gpu) return 1;
        try {
          const v = new Uint32Array(g.memory.buffer, dp >>> 0, 20);
          const fv = new Float32Array(g.memory.buffer, dp >>> 0, 20);
          const d = { handle: v[0], nOut: v[1], nIn: v[2], h: v[3], w: v[4], cin: v[5], k: v[6], stride: v[7], oh: v[8], ow: v[9], y: v[10], mode: v[11], silu: v[12], cpg: v[13], eps: fv[14], nStats: v[17] };
          if (d.mode) {
            d.gamma = new Float32Array(g.memory.buffer, v[15] >>> 0, d.cin);
            d.beta = new Float32Array(g.memory.buffer, v[16] >>> 0, d.cin);
          }
          const x = new Float32Array(g.memory.buffer, xp >>> 0, d.h * d.w * d.cin);
          await g.gpu.conv(x, d, (out) => new Float32Array(g.memory.buffer, d.y >>> 0, out.length).set(out));
          return 0;
        } catch (e) {
          log(`[webgpu: ${e.message || e}]`);
          return 2;
        }
      },
      gpu_attention: async (qp, kp, vp, op, dp) => {
        if (!g.gpu) return 1;
        try {
          const v = new Uint32Array(g.memory.buffer, dp >>> 0, 8);
          const d = { heads: v[0], tqp: v[1], tkp: v[2], dhp: v[3], dho: v[4], tk: v[5], causal: v[6], scale: new Float32Array(g.memory.buffer, dp >>> 0, 8)[7] };
          const mem = g.memory.buffer;
          const q = new Float32Array(mem, qp >>> 0, d.heads * d.tqp * d.dhp);
          const k = new Float32Array(mem, kp >>> 0, d.heads * d.tkp * d.dhp);
          const vt = new Float32Array(mem, vp >>> 0, d.heads * d.dho * d.tkp);
          await g.gpu.attention(q, k, vt, d, (o) => new Float32Array(g.memory.buffer, op >>> 0, o.length).set(o));
          return 0;
        } catch (e) {
          log(`[webgpu: ${e.message || e}]`);
          return 2;
        }
      },
      gpu_matmul: async (xp, t, nIn, jp, nJobs) => {
        if (!g.gpu) return 1;
        try {
          const x = new Float32Array(g.memory.buffer, xp >>> 0, t * nIn).slice(); // copy: the module may move on
          const d = new Uint32Array(g.memory.buffer, jp >>> 0, nJobs * 4);
          const jobs = [];
          for (let i = 0; i < nJobs; i++) jobs.push({ handle: d[i * 4], nOut: d[i * 4 + 1], y: d[i * 4 + 2] });
          await g.gpu.matmul(x, t, nIn, jobs, (slot, out) => {
            // Re-read the memory after suspension: it may have grown.
            new Float32Array(g.memory.buffer, jobs[slot].y >>> 0, out.length).set(out);
          });
          return 0;
        } catch (e) {
          log(`[webgpu: ${e.message || e}]`);
          return 2;
        }
      },
    };
    if (jspi) {
      env.gpu_matmul = new WebAssembly.Suspending(env.gpu_matmul);
      env.gpu_conv = new WebAssembly.Suspending(env.gpu_conv);
      env.gpu_attention = new WebAssembly.Suspending(env.gpu_attention);
    } else {
      env.gpu_conv = () => 1;
      env.gpu_attention = () => 1;
    }
    let bytes = source;
    if (typeof source === 'string') source = await fetch(source);
    if (source instanceof Response) bytes = await source.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, { env });
    g.ex = instance.exports;
    g.memory = g.ex.memory;
    // Exports that can reach gpu_matmul must be promising (they return promises then).
    g.chatTurnRaw = jspi ? WebAssembly.promising(g.ex.web_chat_turn) : async (...a) => g.ex.web_chat_turn(...a);
    g.nextRaw = jspi ? WebAssembly.promising(g.ex.web_next) : async () => g.ex.web_next();
    g.sdGenerateRaw = jspi ? WebAssembly.promising(g.ex.web_sd_generate) : async (...a) => g.ex.web_sd_generate(...a);
    g.toolResultsRaw = jspi ? WebAssembly.promising(g.ex.web_tool_results) : async (...a) => g.ex.web_tool_results(...a);
    return g;
  }

  // Pointers cross the boundary as i32; `>>> 0` restores addresses above 2 GiB.
  u8(p, n) { return new Uint8Array(this.memory.buffer, p >>> 0, n >>> 0); }
  str(p, n) { return dec.decode(this.u8(p, n)); }

  /// Copies a string into module memory; returns [ptr, len] (free with web_free).
  strArg(s) {
    const b = enc.encode(s);
    const p = this.ex.web_alloc(b.length || 1) >>> 0;
    this.u8(p, b.length).set(b);
    return [p, b.length];
  }

  lastError() {
    const lenBuf = this.ex.web_alloc(8) >>> 0;
    const p = this.ex.web_last_error(lenBuf);
    const n = new Uint32Array(this.memory.buffer, lenBuf, 1)[0];
    const s = this.str(p, n);
    this.ex.web_free(lenBuf, 8);
    return s;
  }

  /// Registers `bytes` (Uint8Array) as `path` in the module's virtual filesystem.
  putFile(path, bytes) {
    const p = this.ex.web_alloc(bytes.byteLength || 1) >>> 0;
    this.u8(p, bytes.byteLength).set(bytes);
    const [pp, pn] = this.strArg(path);
    this.ex.web_put_file(pp, pn, p, bytes.byteLength);
    this.ex.web_free(pp, pn);
  }

  /// Allocates a file of `size` bytes and returns a function that receives chunks in order, so a
  /// download can stream straight into module memory; call `.done()` to register it.
  putFileStream(path, size) {
    const p = this.ex.web_alloc(size || 1) >>> 0;
    let off = 0;
    const self = this;
    return {
      write(chunk) { self.u8(p, size).set(chunk, off); off += chunk.byteLength; },
      done() {
        const [pp, pn] = self.strArg(path);
        self.ex.web_put_file(pp, pn, p, size);
        self.ex.web_free(pp, pn);
      },
    };
  }

  /// Registers `bytes` (Uint8Array, kept in JavaScript memory) as `path`; the module reads it by
  /// range. For lookup-table shards, which would not fit in wasm memory alongside the weights.
  putHostFile(path, bytes) {
    const id = this.hostFiles.length;
    this.hostFiles.push(bytes);
    const [pp, pn] = this.strArg(path);
    const rc = this.ex.web_put_host_file(pp, pn, id, bytes.byteLength);
    this.ex.web_free(pp, pn);
    if (rc < 0) throw new Error(this.lastError());
  }

  // ---- tools ----

  /// Declares tools ([{ name, description, parameters }]) for the next conversation.
  setTools(specs) {
    const [p, n] = this.strArg(JSON.stringify(specs || []));
    const rc = this.ex.web_set_tools(p, n);
    this.ex.web_free(p, n);
    if (rc < 0) throw new Error(this.lastError());
  }

  /// Tool calls in the reply that just ended: [{ name, arguments }].
  toolCalls() {
    const lenBuf = this.ex.web_alloc(8) >>> 0;
    const p = this.ex.web_tool_calls(lenBuf);
    const n = new Uint32Array(this.memory.buffer, lenBuf, 1)[0];
    const s = n ? this.str(p, n) : '[]';
    this.ex.web_free(lenBuf, 8);
    return JSON.parse(s);
  }

  /// Feeds results ([{ name, content }]); then keep calling next() for the continued reply.
  async toolResults(results) {
    const [p, n] = this.strArg(JSON.stringify(results));
    const rc = await this.toolResultsRaw(p, n);
    this.ex.web_free(p, n);
    if (rc < 0) throw new Error(this.lastError());
    return rc;
  }

  // ---- image generation ----

  /// Loads a diffusers Stable Diffusion folder registered under `dir`, merging the LoRA files
  /// `loras` ([{ path, scale }], paths in the module's filesystem) into its weights.
  sdLoad(dir, loras = []) {
    const [dp, dn] = this.strArg(dir);
    const spec = loras.map((l) => `${l.path}:${l.scale ?? 1}`).join('\n');
    const [lp, ln] = spec ? this.strArg(spec) : [0, 0];
    const rc = this.ex.web_sd_load(dp, dn, this.gpu ? 1 : 0, lp, ln);
    this.ex.web_free(dp, dn);
    if (ln) this.ex.web_free(lp, ln);
    if (rc < 0) throw new Error(this.lastError());
  }

  /// Recompute every GPU op on the CPU as well (slow) and keep the worst error per kind.
  gpuCheckEnable() {
    this.ex.web_gpu_check_enable();
  }

  gpuCheckSummary() {
    const lenBuf = this.ex.web_alloc(8) >>> 0;
    const p = this.ex.web_gpu_check_summary(lenBuf);
    const n = new Uint32Array(this.memory.buffer, lenBuf, 1)[0];
    const s = n ? this.str(p, n) : '';
    this.ex.web_free(lenBuf, 8);
    return s;
  }

  /// { size, steps, guidance, turbo } for the loaded pipeline.
  sdDefaults() {
    const p = this.ex.web_alloc(16) >>> 0;
    const rc = this.ex.web_sd_defaults(p);
    const v = new Float32Array(this.memory.buffer, p, 4).slice();
    this.ex.web_free(p, 16);
    if (rc < 0) throw new Error(this.lastError());
    return { size: v[0], steps: v[1], guidance: v[2], turbo: v[3] !== 0 };
  }

  /// Generates an image; resolves to { width, height, rgb: Uint8Array }.
  async sdGenerate({ prompt, negative = '', steps = 1, guidance = 0, seed = 0, width = 512, height = 512, ancestral = true }, onProgress) {
    const [pp, pn] = this.strArg(prompt);
    const [np, nn] = negative ? this.strArg(negative) : [0, 0];
    this.onProgress = onProgress;
    try {
      const rc = await this.sdGenerateRaw(pp, pn, np, nn, steps, guidance, seed, width, height, ancestral ? 1 : 0);
      if (rc < 0) throw new Error(this.lastError());
    } finally {
      this.onProgress = null;
      this.ex.web_free(pp, pn);
      if (nn) this.ex.web_free(np, nn);
    }
    const dims = this.ex.web_alloc(8) >>> 0;
    const ptr = this.ex.web_sd_image(dims, dims + 4) >>> 0;
    const [w, h] = new Uint32Array(this.memory.buffer, dims, 2);
    this.ex.web_free(dims, 8);
    return { width: w, height: h, rgb: new Uint8Array(this.memory.buffer, ptr, w * h * 3).slice() };
  }

  /// quant: 'bf16' | 'q8' | 'q4' (ignored for pre-quantized checkpoints, which carry their own).
  load(dir, quant = 'q8', { prefillChunk = 0 } = {}) {
    const q = { bf16: 0, q8: 1, q4: 2 }[quant];
    const [dp, dn] = this.strArg(dir);
    const rc = this.ex.web_load(dp, dn, q, this.gpu ? 1 : 0, prefillChunk);
    this.ex.web_free(dp, dn);
    if (rc < 0) throw new Error(this.lastError());
  }

  setSampling({ temperature = 0, topK = 0, topP = 1, seed = 0 } = {}) {
    this.ex.web_set_sampling(temperature, topK, topP, BigInt(seed));
  }

  reset() { this.ex.web_reset(); }

  /// Runs the prefill for a user turn; resolves to the number of prompt tokens.
  async chatTurn(user, { system = null, think = false, maxTokens = 512 } = {}) {
    const [up, un] = this.strArg(user);
    const [sp, sn] = system ? this.strArg(system) : [0, 0];
    const n = await this.chatTurnRaw(up, un, sp, sn, think ? 1 : 0, maxTokens);
    this.ex.web_free(up, un);
    if (sn) this.ex.web_free(sp, sn);
    if (n < 0) throw new Error(this.lastError());
    return n;
  }

  /// Next token id, or -1 when the reply is complete.
  async next() { return await this.nextRaw(); }

  text() {
    const lenBuf = this.ex.web_alloc(8) >>> 0;
    const p = this.ex.web_text(lenBuf);
    const n = new Uint32Array(this.memory.buffer, lenBuf, 1)[0];
    const s = this.str(p, n);
    this.ex.web_free(lenBuf, 8);
    return s;
  }

  gpuStats() {
    const lenBuf = this.ex.web_alloc(8) >>> 0;
    const p = this.ex.web_gpu_stats(lenBuf);
    const n = new Uint32Array(this.memory.buffer, lenBuf, 1)[0];
    const s = n ? this.str(p, n) : '';
    this.ex.web_free(lenBuf, 8);
    return s;
  }

  /// Streams a whole reply: calls `onText(fullTextSoFar)` after each token. Returns stats.
  async generate(user, opts = {}, onText = () => {}) {
    const t0 = performance.now();
    const nPrompt = await this.chatTurn(user, opts);
    const t1 = performance.now();
    let n = 0;
    while ((await this.next()) >= 0) {
      n++;
      onText(this.text());
    }
    const t2 = performance.now();
    return { promptTokens: nPrompt, prefillSecs: (t1 - t0) / 1e3, tokens: n, decodeSecs: (t2 - t1) / 1e3, text: this.text() };
  }
}

/// Model files to register for a directory listing: config, tokenizer, generation config, shards.
export function wantedFiles(names) {
  return names.filter((n) => n === 'config.json' || n === 'generation_config.json' || n === 'tokenizer.json' || n === 'tokenizer_config.json' || n === 'model.safetensors.index.json' || n.endsWith('.safetensors'));
}

/// Shards written by `--save-quantized` that only hold lookup tables; the hosts keep these in
/// JavaScript memory (`putHostFile`) instead of module memory.
export function isHostShard(name) {
  return /^model-tables(-\d+-of-\d+)?\.safetensors$/.test(name);
}
