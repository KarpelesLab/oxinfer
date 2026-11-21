//! One turn of a conversation with Haru: a recording of the user in, the model's spoken answer
//! (and its text) out. Haru is full duplex: [`super::DuplexSession`] plays the recording to it frame
//! by frame, as live, then silence, during which the model answers; the turn ends once the model
//! has spoken and gone quiet again, or after a time limit. The server's chat completions with
//! `modalities: ["text", "audio"]` and the C API's `oxi_haru_respond` run this.

use super::{DuplexConfig, DuplexSession, Haru};
use anyhow::{Result, bail};

/// How a turn is run.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TurnOptions {
    /// The character card (system prompt); empty: none.
    pub system: String,
    /// A clip of the voice the model should speak with (mono, at the model's 24 kHz).
    pub reference: Option<Vec<f32>>,
    /// The session's sampling and guidance (its `max_frames` is set from the lengths below).
    pub config: DuplexConfig,
    /// The longest answer, in seconds of silence played after the recording (default 12).
    pub max_reply_seconds: f32,
    /// The turn ends once the model has spoken after the recording and then been silent (no
    /// text) for this long (default 1.2 s).
    pub end_silence_seconds: f32,
    /// Keep what the model said while the user spoke too (default false: the answer only, from
    /// the end of the recording).
    pub include_overlap: bool,
}

impl Default for TurnOptions {
    fn default() -> Self {
        Self {
            system: String::new(),
            reference: None,
            config: DuplexConfig::default(),
            max_reply_seconds: 12.0,
            end_silence_seconds: 1.2,
            include_overlap: false,
        }
    }
}

/// The model's side of a turn.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct Turn {
    /// What the model said (its text stream, trimmed).
    pub text: String,
    /// Mono samples in `[-1, 1]` at `sample_rate`.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Frames the model ran (the recording's and the answer's).
    pub frames: usize,
    /// Frames of the recording.
    pub user_frames: usize,
    /// The model answered and went quiet (not the time limit).
    pub finished: bool,
}

impl Turn {
    /// The audio as a 16-bit PCM WAV file.
    pub fn to_wav(&self) -> Vec<u8> {
        crate::tts::wav_bytes(&self.samples, self.sample_rate)
    }
}

impl Haru {
    /// Plays `user` (mono at the model's rate, [`super::HaruInfo::sample_rate`]) to the model,
    /// then silence until it has answered; `on_frame` receives each kept frame's text and audio
    /// as it is made (return false to stop).
    pub fn respond(&self, user: &[f32], o: &TurnOptions, on_frame: &mut dyn FnMut(&str, &[f32]) -> bool) -> Result<Turn> {
        let info = self.info();
        let fs = info.frame_samples;
        let rate = info.sample_rate;
        if user.is_empty() {
            bail!("an empty recording");
        }
        let user_frames = user.len().div_ceil(fs);
        let max_tail = ((o.max_reply_seconds.max(0.0) * rate as f32) as usize).div_ceil(fs).max(1);
        let quiet_frames = ((o.end_silence_seconds.max(0.0) * rate as f32) as usize).div_ceil(fs).max(1);
        let mut config = o.config.clone();
        config.max_frames = user_frames + max_tail + 1;
        let mut session = DuplexSession::new(self, &o.system, o.reference.as_deref(), config);
        let mut padded = user.to_vec();
        padded.resize(user_frames * fs, 0.0);
        let silence = vec![0f32; fs];
        let (mut text, mut samples) = (String::new(), Vec::new());
        let (mut spoke, mut quiet, mut finished) = (false, 0usize, false);
        // The model's frame t is heard while the user speaks frame t, computed from the user's
        // frames before it.
        let mut frame = session.start()?;
        let total = user_frames + max_tail;
        let mut ran = 0;
        for index in 0..total {
            ran = index + 1;
            let tail = index >= user_frames;
            let silent = frame.text.trim().is_empty();
            if tail || o.include_overlap {
                if !on_frame(&frame.text, &frame.audio) {
                    break;
                }
                text.push_str(&frame.text);
                samples.extend_from_slice(&frame.audio);
            }
            if tail {
                if silent {
                    quiet += 1;
                } else {
                    spoke = true;
                    quiet = 0;
                }
                if spoke && quiet >= quiet_frames {
                    finished = true;
                    break;
                }
            }
            if index + 1 < total {
                let next = if index < user_frames {
                    &padded[index * fs..(index + 1) * fs]
                } else {
                    &silence[..]
                };
                frame = session.step(next)?;
            }
        }
        // The trailing quiet frames of the answer are dropped, keeping 0.2 s.
        if finished && !o.include_overlap {
            let keep = (quiet_frames * fs).saturating_sub(rate as usize / 5);
            samples.truncate(samples.len().saturating_sub(keep));
        }
        Ok(Turn {
            text: text.split_whitespace().collect::<Vec<_>>().join(" "),
            samples,
            sample_rate: rate,
            frames: ran,
            user_frames,
            finished,
        })
    }
}
