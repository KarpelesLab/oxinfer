//! Inkling-2 sessions going back on CUDA (`DevSession::truncate` of `inkling_cuda`), on the real
//! weights: needs an NVIDIA GPU and `ggml-org/Inkling-2-Mini-GGUF:Q8_0` (downloaded on first use,
//! 27.8 GB; `OXINFER_INKLING_ROLLBACK_MODEL` names another checkpoint). Ignored by default:
//!
//! ```text
//! cargo test --release --test inkling_cuda_rollback -- --ignored --nocapture --test-threads 1
//! ```
//!
//! (a) A session that runs extra tokens and goes back 1, 3, 7 and 64 positions (from passes of one
//! token, of 3 and of 9) gives, token for token, the logits of an uninterrupted run. (b) Greedy
//! decoding with prompt lookup, and with a draft model when one is available, writes the plain
//! greedy text token for token. Its own test binary: it installs the process-wide GPU backend.

#![cfg(all(
    feature = "gpu",
    feature = "hf",
    not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))
))]

use oxinfer::engine::{Engine, EngineOptions, Flow};
use oxinfer::inkling::{Inkling2Config, Inkling2Model};
use oxinfer::safetensors::ModelFiles;

fn model_name() -> String {
    std::env::var("OXINFER_INKLING_ROLLBACK_MODEL").unwrap_or_else(|_| "ggml-org/Inkling-2-Mini-GGUF:Q8_0".into())
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
#[ignore = "needs an NVIDIA GPU and the Inkling-2-Mini weights"]
fn truncate_matches_an_uninterrupted_run() {
    oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Auto).unwrap();
    let be = oxinfer::gpu::active().expect("no GPU backend");
    eprintln!("[backend: {}]", be.name());
    let dir = oxinfer::engine::resolve_model(&model_name(), None).unwrap();
    let files = ModelFiles::open(&dir, true).unwrap();
    let m = Inkling2Model::load(Inkling2Config::load(&dir).unwrap(), &files, oxinfer::model::Quant::Bf16).unwrap();
    assert_eq!(m.rollback_limit(), 64, "the device's sessions should go back 64 positions");
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % 199_000) as u32 + 100
    };
    let prompt: Vec<u32> = (0..40).map(|_| rnd()).collect();
    let steps = 40;
    // The reference: the prompt, then greedy one token at a time.
    let mut sess = m.new_session();
    let mut logits = m.forward(&mut sess, &prompt);
    let mut toks = Vec::new();
    let mut want = Vec::new();
    for _ in 0..steps {
        let t = argmax(&logits) as u32;
        toks.push(t);
        logits = m.forward(&mut sess, &[t]);
        want.push(logits.clone());
    }
    // The same tokens with detours: every few tokens `back` junk ones (in passes of 1, 3 and 9
    // tokens: the matvec path and the GEMMs), then going back over them; real tokens in passes of
    // `per` (the last one's logits compared).
    for (back, junk_pass, per) in [(1usize, 1usize, 1usize), (3, 3, 1), (7, 9, 3), (64, 9, 1), (3, 1, 3)] {
        let mut s = m.new_session();
        m.forward(&mut s, &prompt);
        let mut i = 0;
        let mut worst = 0f32;
        while i < steps {
            if i % 5 == 2 {
                let junk: Vec<u32> = (0..back).map(|_| rnd()).collect();
                for c in junk.chunks(junk_pass) {
                    m.forward(&mut s, c);
                }
                let len = s.pos - back;
                s.truncate(len).unwrap();
            }
            let n = per.min(steps - i);
            let got = m.forward(&mut s, &toks[i..i + n]);
            let w = &want[i + n - 1];
            let d = max_diff(&got, w);
            worst = worst.max(d);
            assert_eq!(
                argmax(&got),
                argmax(w),
                "back {back} (junk passes of {junk_pass}, real ones of {per}): token {} parted (max diff {d})",
                i + n - 1
            );
            assert!(
                d < 1e-3,
                "back {back} (junk passes of {junk_pass}, real ones of {per}): token {}: max diff {d}",
                i + n - 1
            );
            i += n;
        }
        eprintln!("[back {back}, junk passes of {junk_pass}, real passes of {per}: max logit diff {worst:.3e}]");
    }
}

fn greedy(opts: &EngineOptions, prompt: &str, n: usize) -> Vec<u32> {
    run(&mut Engine::load(&model_name(), opts).unwrap(), prompt, n)
}

fn run(e: &mut Engine, prompt: &str, n: usize) -> Vec<u32> {
    e.config_mut().max_tokens = n;
    let r = e.complete(prompt, &mut |_| Flow::Continue).unwrap();
    eprintln!("[{} tokens, {:.1} tok/s]", r.tokens.len(), r.stats.decode_tok_s());
    r.tokens
}

#[test]
#[ignore = "needs an NVIDIA GPU and the Inkling-2-Mini weights"]
fn speculative_greedy_is_plain_greedy() {
    // A prompt that repeats itself (prompt lookup finds matches).
    let prompt = "fn add(a: u32, b: u32) -> u32 {\n    a + b\n}\n\nfn sub(a: u32, b: u32) -> u32 {\n    a - b\n}\n\n\
                  fn mul(a: u32, b: u32) -> u32 {\n    a * b\n}\n\nfn div(a: u32, b: u32) -> u32 {\n";
    let n = 64;
    let mut plain = EngineOptions::default();
    plain.set("lookup", "0").unwrap();
    let want = greedy(&plain, prompt, n);
    let mut lookup = EngineOptions::default();
    lookup.set("lookup", "1").unwrap();
    assert_eq!(greedy(&lookup, prompt, n), want, "prompt lookup changed the greedy text");
    // A draft model, when one is found (on CUDA drafts are off: the same plain text either way).
    let mut draft = EngineOptions::default();
    draft.set("lookup", "0").unwrap();
    draft.set("draft", "auto").unwrap();
    match Engine::load(&model_name(), &draft) {
        Ok(mut e) => assert_eq!(run(&mut e, prompt, n), want, "the draft model changed the greedy text"),
        Err(e) => eprintln!("[no draft model: {e}]"),
    }
}
