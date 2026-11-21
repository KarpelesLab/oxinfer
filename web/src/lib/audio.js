// Microphone recording as a 16 kHz mono WAV file, which every speech-to-text model of the server
// reads (the browser's own recording formats vary: WebM / Opus, MP4 / AAC, Ogg).

/// Starts recording; resolves to { stop(): Promise<Blob WAV>, cancel() }.
export async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(window.isSecureContext ? 'This browser cannot record audio.' : 'Recording needs a secure page: open the GUI at http://localhost or over HTTPS.');
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  const rec = new MediaRecorder(stream);
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.start(250);
  const end = () => stream.getTracks().forEach((t) => t.stop());
  return {
    stop: () =>
      new Promise((resolve, reject) => {
        rec.onstop = async () => {
          end();
          try {
            resolve(await toWav(new Blob(chunks, { type: rec.mimeType })));
          } catch (e) {
            reject(e);
          }
        };
        rec.stop();
      }),
    cancel() {
      rec.onstop = end;
      rec.stop();
    },
  };
}

/// Any audio the browser decodes, as a 16 kHz mono 16-bit WAV.
export async function toWav(blob, rate = 16000) {
  const ctx = new AudioContext();
  let buf;
  try {
    buf = await ctx.decodeAudioData(await blob.arrayBuffer());
  } finally {
    ctx.close();
  }
  const frames = Math.max(1, Math.ceil(buf.duration * rate));
  const off = new OfflineAudioContext(1, frames, rate);
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start();
  const mono = (await off.startRendering()).getChannelData(0);
  return encodeWav(mono, rate);
}

export function encodeWav(samples, rate) {
  const out = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const str = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF');
  out.setUint32(4, 36 + samples.length * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  out.setUint32(16, 16, true);
  out.setUint16(20, 1, true);
  out.setUint16(22, 1, true);
  out.setUint32(24, rate, true);
  out.setUint32(28, rate * 2, true);
  out.setUint16(32, 2, true);
  out.setUint16(34, 16, true);
  str(36, 'data');
  out.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([out], { type: 'audio/wav' });
}
