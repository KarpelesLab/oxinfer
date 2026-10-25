//! End-to-end checks of the oxideav-core based API against the path-based one: media decoded by
//! oxideav in the test (as an oxideav pipeline would hand it over) must give the same soft
//! tokens, replies and transcripts as the file inputs. They need models and media files, so they
//! are ignored by default:
//!
//! ```text
//! OXINFER_E2E_IMAGE=fox.png OXINFER_E2E_VIDEO=clip.mp4 OXINFER_E2E_AUDIO=speech.mp3 \
//!   cargo test --release --test oxideav_e2e -- --ignored --nocapture
//! ```
//!
//! Models (fetched or taken from the cache): `Qwen/Qwen3.5-0.8B` (vision tower),
//! `openai/whisper-tiny`, `stabilityai/sd-turbo`; override with `OXINFER_E2E_QWEN`,
//! `OXINFER_E2E_WHISPER`, `OXINFER_E2E_SD`.

#![cfg(all(feature = "media", feature = "hf"))]

use oxideav_core::{AudioFrame, CodecParameters, Frame, MediaType, RuntimeContext, TimeBase, VideoFrame};
use oxinfer::av::{AudioClip, Picture, VideoClip};
use oxinfer::engine::{Attachments, Engine, EngineOptions, Flow};
use oxinfer::mm::{self, MmFeature, MmFrames, MmRequest};
use oxinfer::safetensors::ModelFiles;
use std::path::{Path, PathBuf};

fn var(name: &str, default: Option<&str>) -> String {
    std::env::var(name)
        .ok()
        .or(default.map(str::to_string))
        .unwrap_or_else(|| panic!("set {name} (see tests/oxideav_e2e.rs)"))
}

fn context() -> RuntimeContext {
    let mut ctx = RuntimeContext::new();
    oxinfer::media::register(&mut ctx);
    ctx
}

/// Every decoded frame of the first stream of `kind` in a file, its parameters and time base:
/// what a demuxer + decoder stage of an oxideav pipeline produces.
fn decode(ctx: &RuntimeContext, path: &Path, kind: MediaType) -> (Vec<Frame>, CodecParameters, TimeBase) {
    let file = std::fs::File::open(path).unwrap();
    let mut input: Box<dyn oxideav_core::ReadSeek> = Box::new(file);
    let ext = path.extension().and_then(|e| e.to_str());
    let format = ctx.containers.probe_input(&mut *input, ext).unwrap();
    let mut demux = ctx.containers.open_demuxer(&format, input, &ctx.codecs).unwrap();
    let stream = demux.streams().iter().find(|s| s.params.media_type == kind).cloned().unwrap();
    let mut params = stream.params.clone();
    // MOV-family files can carry the codec configuration box with its header.
    if params.extradata.len() > 8 && matches!(&params.extradata[4..8], b"avcC" | b"hvcC" | b"av1C" | b"vpcC") {
        params.extradata.drain(..8);
    }
    let mut dec = oxideav_pipeline::make_decoder(&ctx.codecs, &params).unwrap();
    let mut frames = Vec::new();
    loop {
        match demux.next_packet() {
            Ok(p) if p.stream_index == stream.index => {
                dec.send_packet(&p).unwrap();
                while let Ok(f) = dec.receive_frame() {
                    frames.push(f);
                }
            }
            Ok(_) => {}
            Err(e) if e.is_eof() => break,
            Err(e) => panic!("demux: {e}"),
        }
    }
    let _ = dec.flush();
    while let Ok(f) = dec.receive_frame() {
        frames.push(f);
    }
    (frames, params, stream.time_base)
}

fn video_frames(frames: Vec<Frame>) -> Vec<VideoFrame> {
    frames
        .into_iter()
        .filter_map(|f| match f {
            Frame::Video(v) => Some(v),
            _ => None,
        })
        .collect()
}

fn audio_frames(frames: Vec<Frame>) -> Vec<AudioFrame> {
    frames
        .into_iter()
        .filter_map(|f| match f {
            Frame::Audio(a) => Some(a),
            _ => None,
        })
        .collect()
}

fn same_features(a: &(String, Vec<MmFeature>), b: &(String, Vec<MmFeature>)) {
    assert_eq!(a.0, b.0, "placeholder text");
    assert_eq!(a.1.len(), b.1.len());
    for (x, y) in a.1.iter().zip(&b.1) {
        assert_eq!((x.token_id, x.n, &x.grids), (y.token_id, y.n, &y.grids));
        assert!(x.emb == y.emb, "soft tokens differ");
    }
}

fn reply(engine: &mut Engine, prompt: &str, media: &Attachments) -> String {
    engine.reset();
    engine.send_with_media(prompt, media, &mut |_| Flow::Continue).unwrap().text
}

/// An image decoded by oxideav-image and handed over as a VideoFrame: the same soft tokens and
/// the same greedy reply as the file.
#[test]
#[ignore]
fn qwen_image_frame_matches_file() {
    let image = PathBuf::from(var("OXINFER_E2E_IMAGE", None));
    let mut engine = Engine::load(&var("OXINFER_E2E_QWEN", Some("Qwen/Qwen3.5-0.8B")), &EngineOptions::default()).unwrap();
    engine.config_mut().max_tokens = 48;
    let dir = engine.info().path.clone();
    let hidden = engine.info().hidden_size;
    let files = ModelFiles::open(&dir, false).unwrap();

    let ctx = context();
    let (frame, params) = oxideav_image::open(&ctx, &image).unwrap().into_primary().into_video_frame();
    eprintln!(
        "[{}: {:?} {}x{}]",
        image.display(),
        params.pixel_format,
        params.width.unwrap(),
        params.height.unwrap()
    );
    let pic = Picture::new(frame.clone(), params.clone());

    let from_file = mm::prepare(&MmRequest::new(&dir, std::slice::from_ref(&image), &[]), &files, hidden).unwrap();
    let frames = MmFrames {
        images: std::slice::from_ref(&pic),
        ..MmFrames::default()
    };
    let from_frame = mm::prepare_frames(&MmRequest::new(&dir, &[], &[]), frames, &files, hidden).unwrap();
    same_features(&from_file, &from_frame);

    let prompt = "Describe this image in one sentence.";
    let mut by_path = Attachments::default();
    by_path.images.push(image.clone());
    let a = reply(&mut engine, prompt, &by_path);
    let b = reply(&mut engine, prompt, &Attachments::default().with_image_frame(frame, params));
    eprintln!("[file]  {a}\n[frame] {b}");
    assert_eq!(a, b);
}

/// A video decoded by oxideav and handed over as VideoFrames with their pts: the same sampled
/// frames, timestamps and soft tokens as the file.
#[test]
#[ignore]
fn qwen_video_frames_match_file() {
    let video = PathBuf::from(var("OXINFER_E2E_VIDEO", None));
    let model = oxinfer::hf::resolve(&var("OXINFER_E2E_QWEN", Some("Qwen/Qwen3.5-0.8B")), None).unwrap();
    let files = ModelFiles::open(&model, false).unwrap();
    let hidden = oxinfer::qwen_vision::QwenMmConfig::load(&model)
        .unwrap()
        .unwrap()
        .vision
        .out_hidden_size;

    let ctx = context();
    let (frames, params, tb) = decode(&ctx, &video, MediaType::Video);
    let frames = video_frames(frames);
    eprintln!(
        "[{}: {} frames, {:?}, time base {tb:?}]",
        video.display(),
        frames.len(),
        params.pixel_format
    );
    let clip = VideoClip::new(frames, params, Some(tb));

    let mut req = MmRequest::new(&model, &[], &[]);
    let videos = [video.clone()];
    req.videos = &videos;
    let from_file = mm::prepare(&req, &files, hidden).unwrap();
    let req = MmRequest::new(&model, &[], &[]);
    let mmf = MmFrames {
        videos: std::slice::from_ref(&clip),
        ..MmFrames::default()
    };
    let from_frames = mm::prepare_frames(&req, mmf, &files, hidden).unwrap();
    let stamps = |t: &str| t.split("seconds>").count() - 1;
    eprintln!("[{} temporal patches]", stamps(&from_file.0));
    same_features(&from_file, &from_frames);
}

/// Audio decoded by oxideav and handed over as AudioFrames: the same transcript as the file.
#[test]
#[ignore]
fn whisper_frames_match_file() {
    use oxinfer::whisper::{TranscribeOptions, Transcriber, TranscriberOptions};
    let audio = PathBuf::from(var("OXINFER_E2E_AUDIO", None));
    let asr = Transcriber::load(
        &var("OXINFER_E2E_WHISPER", Some("openai/whisper-tiny")),
        &TranscriberOptions::default(),
    )
    .unwrap();
    let opts = TranscribeOptions::default();

    let ctx = context();
    let (frames, params, _) = decode(&ctx, &audio, MediaType::Audio);
    let frames = audio_frames(frames);
    eprintln!(
        "[{}: {} frames, {:?} Hz, {:?} channels, {:?}]",
        audio.display(),
        frames.len(),
        params.sample_rate,
        params.channels,
        params.sample_format
    );
    // The samples first: the same as the file decoder's.
    assert!(AudioClip::new(frames.clone(), params.clone()).to_mono(16000).unwrap() == Transcriber::load_audio(&audio).unwrap());

    let a = asr.transcribe_file(&audio, &opts).unwrap();
    let b = asr.transcribe_frames(&frames, &params, &opts).unwrap();
    let head = |s: &str| s.chars().take(100).collect::<String>();
    eprintln!("[file]   {} ...\n[frames] {} ...", head(&a.text), head(&b.text));
    assert_eq!(a.text, b.text);
    let seg = |t: &oxinfer::whisper::Transcript| t.segments.iter().map(|s| (s.start, s.end, s.text.clone())).collect::<Vec<_>>();
    assert_eq!(seg(&a), seg(&b));
}

/// A generated image as a VideoFrame, encoded to PNG by oxideav and decoded again: the same
/// pixels.
#[test]
#[ignore]
fn sd_frame_round_trips_through_png() {
    use oxinfer::sd::{Params, Pipeline, sched::Sampler};
    // A diffusers folder (the weights live in its component folders).
    let spec = var("OXINFER_E2E_SD", Some("stabilityai/sd-turbo"));
    let dir = match Path::new(&spec).is_dir() {
        true => PathBuf::from(&spec),
        false => oxinfer::hf::cache_root().join(spec.replace('/', "--")),
    };
    let pipe = Pipeline::load(&dir).unwrap();
    let p = Params {
        prompt: "a red fox in the snow".into(),
        negative: String::new(),
        steps: 1,
        guidance: 0.0,
        seed: 42,
        width: 256,
        height: 192,
        sampler: Sampler::Euler,
    };
    let (img, _) = pipe.generate(&p, &mut |_, _| {}).unwrap();
    let (frame, params) = img.to_video_frame();
    assert_eq!((params.width, params.height), (Some(256), Some(192)));

    let ctx = context();
    let image = oxideav_image::Image::from_video_frame(frame, &params).unwrap();
    let png = oxideav_image::encode(&ctx, &image, "png", &Default::default()).unwrap();
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    let back = oxideav_image::decode_bytes(&ctx, &png).unwrap().into_primary();
    assert_eq!((back.width(), back.height()), (256, 192));
    assert!(back.to_rgb8().unwrap() == img.rgb, "PNG round trip changed the pixels");
    eprintln!("[{}x{} image -> {} byte PNG -> same pixels]", img.width, img.height, png.len());
}
