//! Speech out beyond `/v1/audio/speech`'s text-to-speech models: spoken chat replies (OpenAI's
//! `modalities: ["text", "audio"]` with `audio: {voice, format}`) from LFM3-Omni (a chat model,
//! text and attachments in, its answer as text and speech) and Haru (a speech model, a recording
//! in, the spoken answer out); LFM3-Omni's text to speech through `/v1/audio/speech`; and the voices
//! of every speech model (`GET /v1/audio/voices?model=ID`).

use super::media::{OPENAI_VOICES, TempFiles, random_seed};
use super::registry::{Kind, Loaded};
use super::{Api, ApiError, ApiRequest, ApiResponse, ApiResult, Output, Sink, now};
use serde_json::{Value, json};
use std::sync::Arc;

/// The voices of a loaded model, as `/v1/audio/voices` lists them: `(voices, default, accepts
/// other names)`; each voice `{"id", "name", ...}`.
fn voices_of(m: &Loaded) -> Option<(Vec<Value>, Option<String>, bool, &'static str)> {
    let named = |v: &[String]| v.iter().map(|n| json!({"id": n, "name": n})).collect::<Vec<_>>();
    Some(match m {
        Loaded::Speech(sp) => {
            let info = sp.info();
            let default = if info.voices.iter().any(|v| v == "tara") {
                Some("tara".to_string())
            } else {
                info.voices.first().cloned()
            };
            // A pretrained Orpheus checkpoint has no named voices: any name is a prompt.
            (named(&info.voices), default, info.voices.is_empty(), "orpheus")
        }
        Loaded::Zonos(z) => (
            z.voices()
                .iter()
                .map(|v| json!({"id": v.name, "name": v.name, "label": v.label, "language": v.language}))
                .chain(std::iter::once(
                    json!({"id": "none", "name": "none", "description": "no speaker conditioning"}),
                ))
                .collect(),
            None,
            false,
            "zonos",
        ),
        Loaded::Drama(_) => (
            (0..16)
                .map(|i| json!({"id": i.to_string(), "name": format!("speaker {i}")}))
                .collect(),
            Some("0".into()),
            false,
            "drama",
        ),
        Loaded::Arktts(a) => (
            a.voices()
                .iter()
                .map(|v| json!({"id": v.name, "name": v.name}))
                .chain(std::iter::once(
                    json!({"id": "none", "name": "none", "description": "the model picks a voice"}),
                ))
                .collect(),
            None,
            false,
            "arktts",
        ),
        Loaded::Haru(_) => (Vec::new(), None, false, "haru"),
        _ => return None,
    })
}

fn omni_voices() -> (Vec<Value>, Option<String>, bool, &'static str) {
    (
        crate::lfm3_omni::voice::VOICES
            .iter()
            .map(|v| json!({"id": v, "name": v}))
            .collect(),
        Some("female".into()),
        false,
        "lfm3_omni",
    )
}

fn voices_json(id: &str, (data, default, any, family): (Vec<Value>, Option<String>, bool, &str)) -> Value {
    let note = match family {
        "haru" => Some("Haru speaks in the voice of a reference clip (audio.voice_reference: a data: URL), not by name"),
        "lfm3_omni" => Some("the voice of text to speech; a spoken chat reply's voice is the model's own"),
        _ if any => Some("a pretrained checkpoint: any name is passed to the prompt"),
        _ => None,
    };
    json!({
        "object": "list",
        "model": id,
        "family": family,
        "default": default,
        "data": data,
        // OpenAI's voice names, which take the default voice.
        "aliases": OPENAI_VOICES,
        "accepts_any": any,
        "note": note,
    })
}

/// Whether the chat model `id` is LFM3-Omni (its files are on disk).
fn is_omni(api: &Api, id: &str) -> bool {
    api.models.local_path(id).is_some_and(|p| crate::lfm3_omni::is_omni(&p))
}

/// `GET /v1/audio/voices?model=ID`: the voices of a speech model (loaded first when needed:
/// some read them from the model's files).
pub(super) fn voices(api: &Arc<Api>, req: &ApiRequest) -> ApiResult<Output> {
    let id = req
        .query_param("model")
        .filter(|m| !m.is_empty())
        .ok_or_else(|| ApiError::bad("model is required (?model=ID)").param("model"))?;
    let kind = api.models.kind_of(&id)?;
    let v = match kind {
        Kind::Chat if is_omni(api, &id) => omni_voices(),
        Kind::Speech => {
            let mut lease = api.models.acquire(&id, Kind::Speech)?;
            voices_of(lease.model()).ok_or_else(|| ApiError::bad(format!("{id} has no voices")))?
        }
        k => {
            return Err(ApiError::bad(format!("the model {id:?} is a {} model, not a speech model", k.as_str())).param("model"));
        }
    };
    Ok(Output::Full(ApiResponse::json(200, &voices_json(&id, v))))
}

/// The voices of `id` for `/v1/models/{id}` when it is loaded and idle (None otherwise).
pub(super) fn loaded_voices(api: &Api, id: &str) -> Option<Value> {
    if api.models.kind_of(id).ok()? == Kind::Chat {
        return is_omni(api, id).then(|| voices_json(id, omni_voices())["data"].clone());
    }
    let v = api.models.peek(id, |m| voices_of(m))??;
    Some(voices_json(id, v)["data"].clone())
}

/// The audio of a reply in the requested format: `wav` or `pcm16` (16-bit little-endian mono).
fn encode_audio(samples: &[f32], rate: u32, format: &str) -> Vec<u8> {
    match format {
        "pcm16" | "pcm" => crate::tts::pcm16_bytes(samples),
        _ => crate::tts::wav_bytes(samples, rate),
    }
}

/// What the request asks of the audio: its format and voice.
struct AudioOut {
    format: String,
    voice: Option<String>,
    /// Haru: a clip of the voice to speak with (a `data:` or `http(s)://` URL).
    reference: Option<String>,
}

fn audio_out(body: &Value, stream: bool) -> ApiResult<AudioOut> {
    let a = body.get("audio").cloned().unwrap_or(Value::Null);
    let format = a
        .get("format")
        .and_then(Value::as_str)
        .unwrap_or(if stream { "pcm16" } else { "wav" })
        .to_string();
    if !["wav", "pcm16"].contains(&format.as_str()) {
        return Err(ApiError::bad(format!("audio.format {format:?}: this server writes wav or pcm16")).param("audio"));
    }
    if stream && format != "pcm16" {
        return Err(ApiError::bad("audio.format: a streamed reply is pcm16").param("audio"));
    }
    Ok(AudioOut {
        format,
        voice: a.get("voice").and_then(Value::as_str).map(str::to_string),
        reference: a
            .get("voice_reference")
            .or_else(|| a.get("reference"))
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

/// A finished spoken reply.
struct Spoken {
    text: String,
    samples: Vec<f32>,
    rate: u32,
    prompt_tokens: usize,
    tokens: usize,
    finished: bool,
}

/// `POST /v1/chat/completions` with `modalities` holding `audio`: LFM3-Omni (a chat model) or
/// Haru (a speech model). Only the system prompt and the last user message are read; the reply
/// is `message.audio` (`data`: base64 in `audio.format`, `transcript`: its text), streamed as
/// `delta.audio` pieces (`transcript`, then `data` as base64 PCM16) with `stream: true`.
pub(super) fn spoken_chat(api: &Arc<Api>, body: &Value) -> ApiResult<Output> {
    let mut temp = TempFiles::default();
    let input = super::openai::spoken_input(api, body, &mut temp)?;
    let out = audio_out(body, input.stream)?;
    let kind = api.models.kind_of(&input.model)?;
    match kind {
        Kind::Chat if is_omni(api, &input.model) => {}
        Kind::Chat => {
            return Err(ApiError::bad(format!(
                "the model {:?} answers in text only; spoken replies are LFM3-Omni's and Haru's",
                input.model
            ))
            .param("modalities"));
        }
        Kind::Speech => {}
        k => return Err(ApiError::bad(format!("the model {:?} is a {} model", input.model, k.as_str())).param("model")),
    }
    // A spoken chat reply has the model's own voice (Haru: the reference clip's).
    if let Some(v) = out.voice.as_deref()
        && !OPENAI_VOICES.contains(&v)
        && !crate::lfm3_omni::voice::VOICES.contains(&v)
    {
        return Err(ApiError::bad(format!(
            "audio.voice {v:?}: a spoken chat reply has the model's own voice (Haru: give audio.voice_reference, a clip)"
        ))
        .param("audio"));
    }
    let reference = match &out.reference {
        Some(u) => Some(super::media::to_file(api, u, &mut temp)?),
        None => None,
    };
    // Haru: the longest answer after the recording, in seconds (an extension).
    let max_reply = body.get("max_reply_seconds").and_then(Value::as_f64).map(|v| v as f32);
    let end_silence = body.get("end_silence_seconds").and_then(Value::as_f64).map(|v| v as f32);
    let id = api.new_id("chatcmpl-");
    let audio_id = api.new_id("audio_");
    let created = now();
    let model = input.model.clone();
    let include_usage = input.include_usage;
    let stream = input.stream;
    let api2 = api.clone();
    let run = move |on_text: &mut dyn FnMut(&str), on_audio: &mut dyn FnMut(&[f32]) -> bool| -> ApiResult<Spoken> {
        let _temp = &temp;
        if kind == Kind::Chat {
            let lease = api2.models.acquire_chat(&input.model, None)?;
            // (The first session: the one whose speech head is loaded.)
            let mut engine = lease.pool.slot(0);
            let dir = api2.models.local_path(&input.model).unwrap_or_default();
            let mut sp = crate::lfm3_omni::SpeechOptions::load(&dir);
            if let Some(t) = input.temperature {
                sp.temperature = t;
            }
            if let Some(k) = input.top_k {
                sp.top_k = k;
            }
            sp.seed = input.seed.unwrap_or_else(random_seed);
            if let Some(m) = input.max_tokens {
                sp.max_new = m.max(1);
            }
            let mut o = crate::lfm3_omni::voice::SpokenOptions::new(crate::lfm3_omni::Mode::Interleaved);
            o.system = input.system.clone();
            o.speech = Some(sp);
            let r = engine
                .speak_omni(&input.text, &input.media, &o, on_text)
                .map_err(|e| ApiError::new(500, format!("{e:#}")))?;
            on_audio(&r.samples);
            Ok(Spoken {
                text: r.text,
                samples: r.samples,
                rate: r.sample_rate,
                prompt_tokens: r.prompt_tokens,
                tokens: r.tokens,
                finished: r.finished,
            })
        } else {
            let mut lease = api2.models.acquire(&input.model, Kind::Speech)?;
            let Loaded::Haru(h) = lease.model() else {
                return Err(ApiError::bad(format!(
                    "the model {:?} is a text-to-speech model: use /v1/audio/speech (spoken chat replies are LFM3-Omni's and Haru's)",
                    input.model
                ))
                .param("model"));
            };
            let rate = h.info().sample_rate;
            let clip =
                input.media.audios.first().ok_or_else(|| {
                    ApiError::bad("Haru answers a recording: put the user's speech in an input_audio part").param("messages")
                })?;
            let user = crate::tts::load_audio_file(clip, rate).map_err(|e| ApiError::bad(format!("decode audio: {e:#}")))?;
            let mut o = crate::haru::turn::TurnOptions {
                system: input.system.clone().unwrap_or_default(),
                ..Default::default()
            };
            if let Some(r) = &reference {
                o.reference = Some(
                    h.load_reference(r)
                        .map_err(|e| ApiError::bad(format!("audio.voice_reference: {e:#}")))?,
                );
            }
            if let Some(t) = input.temperature {
                o.config.temperature = t;
            }
            if let Some(k) = input.top_k {
                o.config.top_k = k;
            }
            if let Some(p) = input.top_p {
                o.config.top_p = p;
            }
            o.config.seed = input.seed.unwrap_or_else(random_seed);
            if let Some(s) = max_reply {
                o.max_reply_seconds = s;
            }
            if let Some(s) = end_silence {
                o.end_silence_seconds = s;
            }
            let t = h
                .respond(&user, &o, &mut |text, audio| {
                    if !text.is_empty() {
                        on_text(text);
                    }
                    on_audio(audio)
                })
                .map_err(|e| ApiError::new(500, format!("{e:#}")))?;
            Ok(Spoken {
                text: t.text,
                samples: t.samples,
                rate,
                prompt_tokens: t.user_frames,
                tokens: t.frames - t.user_frames.min(t.frames),
                finished: t.finished,
            })
        }
    };
    let usage = |s: &Spoken| {
        json!({
            "prompt_tokens": s.prompt_tokens, "completion_tokens": s.tokens, "total_tokens": s.prompt_tokens + s.tokens,
            "completion_tokens_details": {"audio_tokens": s.tokens},
        })
    };
    if !stream {
        let s = run(&mut |_| {}, &mut |_| true)?;
        let data = crate::b64::encode(&encode_audio(&s.samples, s.rate, &out.format));
        let msg = json!({
            "role": "assistant",
            "content": null,
            "refusal": null,
            "audio": {
                "id": audio_id, "data": data, "expires_at": created + 3600, "transcript": s.text,
                "format": out.format, "sample_rate": s.rate, "duration": s.samples.len() as f64 / s.rate.max(1) as f64,
            },
        });
        return Ok(Output::Full(ApiResponse::json(
            200,
            &json!({
                "id": id, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": msg, "finish_reason": if s.finished { "stop" } else { "length" }}],
                "usage": usage(&s), "system_fingerprint": null,
            }),
        )));
    }
    Ok(Output::Stream {
        content_type: "text/event-stream".into(),
        producer: Box::new(move |sink: Sink| {
            let chunk = |delta: Value, finish: Value| {
                json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                })
            };
            let cell = std::cell::RefCell::new(sink);
            let send = |v: &Value| {
                let mut g = cell.borrow_mut();
                (*g)(format!("data: {v}\n\n").as_bytes())
            };
            let mut alive = send(&chunk(json!({"role": "assistant", "content": null}), Value::Null));
            let r = run(
                &mut |t| {
                    if alive {
                        alive = send(&chunk(json!({"audio": {"id": audio_id, "transcript": t}}), Value::Null));
                    }
                },
                &mut |a| {
                    // In pieces of at most a second (24000 samples).
                    for piece in a.chunks(24_000) {
                        let data = crate::b64::encode(&crate::tts::pcm16_bytes(piece));
                        if !send(&chunk(json!({"audio": {"id": audio_id, "data": data}}), Value::Null)) {
                            return false;
                        }
                    }
                    true
                },
            );
            match r {
                Ok(s) => {
                    let mut last = chunk(
                        json!({"audio": {"id": audio_id, "expires_at": created + 3600}}),
                        json!(if s.finished { "stop" } else { "length" }),
                    );
                    if include_usage {
                        last["usage"] = usage(&s);
                    }
                    send(&last);
                    let mut g = cell.borrow_mut();
                    (*g)(b"data: [DONE]\n\n");
                }
                Err(e) => {
                    send(&e.to_json());
                }
            }
        }),
    })
}

/// `/v1/audio/speech` on LFM3-Omni (a chat model): its text to speech (`voice`: `female` or
/// `male`; OpenAI's names take `female`).
pub(super) fn omni_speech(api: &Arc<Api>, body: &Value, model: &str, input: &str, format: &str) -> ApiResult<Output> {
    let voice = body.get("voice").and_then(Value::as_str).filter(|v| !OPENAI_VOICES.contains(v));
    if let Some(v) = voice
        && !crate::lfm3_omni::voice::VOICES.contains(&v)
    {
        return Err(ApiError::bad(format!("voice {v:?}: this model's voices are female, male")).param("voice"));
    }
    if body.get("stream_format").and_then(Value::as_str).is_some_and(|f| f != "audio") {
        return Err(ApiError::bad("stream_format: LFM3-Omni's speech is not streamed").param("stream_format"));
    }
    let dir = api.models.local_path(model).unwrap_or_default();
    let mut sp = crate::lfm3_omni::SpeechOptions::load(&dir);
    let num = |k: &str| body.get(k).and_then(Value::as_f64);
    if let Some(t) = num("temperature") {
        sp.temperature = t as f32;
    }
    if let Some(t) = num("top_k") {
        sp.top_k = t as usize;
    }
    if let Some(t) = num("repetition_penalty") {
        sp.repetition_penalty = t as f32;
    }
    if let Some(t) = num("max_tokens") {
        sp.max_new = (t as usize).max(1);
    }
    sp.seed = num("seed").map(|v| v as u64).unwrap_or_else(random_seed);
    let mut o = crate::lfm3_omni::voice::SpokenOptions::new(crate::lfm3_omni::Mode::Audio);
    o.voice = voice.map(str::to_string);
    o.system = body
        .get("instructions")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    o.speech = Some(sp);
    let lease = api.models.acquire_chat(model, None)?;
    let mut engine = lease.pool.slot(0);
    let r = engine
        .speak_omni(input, &Default::default(), &o, &mut |_| {})
        .map_err(|e| ApiError::new(500, format!("{e:#}")))?;
    drop(engine);
    drop(lease);
    let (bytes, ct) = if format == "pcm" {
        (crate::tts::pcm16_bytes(&r.samples), "audio/pcm")
    } else {
        (r.to_wav(), "audio/wav")
    };
    Ok(Output::Full(ApiResponse {
        status: 200,
        content_type: ct.into(),
        body: bytes,
    }))
}

/// Whether `/v1/audio/speech` should go to [`omni_speech`] for `model`.
pub(super) fn speech_goes_to_omni(api: &Api, model: &str) -> bool {
    matches!(api.models.kind_of(model), Ok(Kind::Chat)) && is_omni(api, model)
}
