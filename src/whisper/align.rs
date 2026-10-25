//! Word-level timestamps, as openai-whisper's `find_alignment`: the cross-attention of the
//! checkpoint's alignment heads over the text tokens (softmax over the window's audio frames),
//! normalized per frame across tokens, median-filtered (width 7) along time, averaged over the
//! heads, then aligned to the frames with dynamic time warping. Tokens are grouped into words
//! (spaces, punctuation, or Unicode boundaries for languages without spaces) and punctuation is
//! merged into the neighbouring words.

/// openai-whisper's `dtw` on a cost matrix `[n, m]`; returns the (text, time) index path.
pub fn dtw(x: &[f32], n: usize, m: usize) -> (Vec<usize>, Vec<usize>) {
    let w = m + 1;
    let mut cost = vec![f32::INFINITY; (n + 1) * w];
    let mut trace = vec![-1i8; (n + 1) * w];
    cost[0] = 0.0;
    for j in 1..=m {
        for i in 1..=n {
            let c0 = cost[(i - 1) * w + j - 1];
            let c1 = cost[(i - 1) * w + j];
            let c2 = cost[i * w + j - 1];
            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            cost[i * w + j] = x[(i - 1) * m + j - 1] + c;
            trace[i * w + j] = t;
        }
    }
    for j in 0..w {
        trace[j] = 2;
    }
    for i in 0..=n {
        trace[i * w] = 1;
    }
    let (mut i, mut j) = (n, m);
    let mut path = Vec::new();
    while i > 0 || j > 0 {
        path.push((i.wrapping_sub(1), j.wrapping_sub(1)));
        match trace[i * w + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    path.reverse();
    path.into_iter().filter(|p| p.0 != usize::MAX && p.1 != usize::MAX).unzip()
}

/// Median filter of width `w` (odd) along each row of `[rows, cols]`, reflect-padded.
pub fn median_filter(x: &mut [f32], rows: usize, cols: usize, w: usize) {
    let half = w / 2;
    if cols <= half {
        return;
    }
    let mut win = vec![0f32; w];
    let mut out = vec![0f32; cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        for c in 0..cols {
            for k in 0..w {
                let mut i = c as isize + k as isize - half as isize;
                if i < 0 {
                    i = -i;
                }
                if i >= cols as isize {
                    i = 2 * (cols as isize - 1) - i;
                }
                win[k] = row[i as usize];
            }
            win.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            out[c] = win[half];
        }
        x[r * cols..(r + 1) * cols].copy_from_slice(&out);
    }
}

/// Token start times from per-head cross-attention scores `[heads][n_tok][n_src]` over the
/// tokens `[sot sequence (sot_len), <|notimestamps|>, text..., <|endoftext|>]`: the softmax is
/// taken over the first `n_frames` positions (the window's audio, 20 ms each). Returns the DTW
/// jump frame of each text token and of the closing `<|endoftext|>` (`n_tok - sot_len - 1`).
pub fn token_times(scores: &[Vec<f32>], n_tok: usize, n_src: usize, n_frames: usize, sot_len: usize) -> Vec<f32> {
    let heads = scores.len();
    let mut mat = vec![0f32; n_tok * n_frames];
    for s in scores {
        let mut w = vec![0f32; n_tok * n_frames];
        for t in 0..n_tok {
            let row = &mut w[t * n_frames..(t + 1) * n_frames];
            row.copy_from_slice(&s[t * n_src..t * n_src + n_frames]);
            crate::kernels::softmax_inplace(row);
        }
        // Normalize each frame across tokens (population std), then the median filter.
        for f in 0..n_frames {
            let mean = (0..n_tok).map(|t| w[t * n_frames + f] as f64).sum::<f64>() / n_tok as f64;
            let var = (0..n_tok).map(|t| (w[t * n_frames + f] as f64 - mean).powi(2)).sum::<f64>() / n_tok as f64;
            let sd = var.sqrt().max(1e-12);
            for t in 0..n_tok {
                w[t * n_frames + f] = ((w[t * n_frames + f] as f64 - mean) / sd) as f32;
            }
        }
        median_filter(&mut w, n_tok, n_frames, 7);
        for (m, v) in mat.iter_mut().zip(&w) {
            *m += v / heads as f32;
        }
    }
    // Text tokens plus the closing <|endoftext|>.
    let rows = n_tok - sot_len - 1;
    let cost: Vec<f32> = mat[sot_len * n_frames..(sot_len + rows) * n_frames].iter().map(|v| -v).collect();
    let (ti, fi) = dtw(&cost, rows, n_frames);
    // A jump is where the text index changes: the time each token starts.
    let mut times = Vec::with_capacity(rows + 1);
    for k in 0..ti.len() {
        if k == 0 || ti[k] != ti[k - 1] {
            times.push(fi[k] as f32);
        }
    }
    times
}

/// Groups tokens into words: a new word starts at a token whose text begins with a space, is
/// punctuation, or is special. `texts[i]` is the decoded text of `groups[i]`.
pub fn split_on_spaces(texts: Vec<String>, groups: Vec<Vec<u32>>, eot: u32) -> (Vec<String>, Vec<Vec<u32>>) {
    let mut words: Vec<String> = Vec::new();
    let mut toks: Vec<Vec<u32>> = Vec::new();
    for (t, g) in texts.into_iter().zip(groups) {
        let special = g[0] >= eot;
        let with_space = t.starts_with(' ');
        let s = t.trim();
        let punct = !s.is_empty() && s.chars().all(|c| c.is_ascii_punctuation());
        if special || with_space || punct || words.is_empty() {
            words.push(t);
            toks.push(g);
        } else {
            words.last_mut().unwrap().push_str(&t);
            toks.last_mut().unwrap().extend(g);
        }
    }
    (words, toks)
}

/// One aligned word before punctuation merging.
#[derive(Clone, Debug)]
pub struct RawWord {
    pub word: String,
    pub tokens: Vec<u32>,
    pub start: f64,
    pub end: f64,
    pub probability: f32,
}

/// openai-whisper's `merge_punctuations` (default sets), dropping the emptied words.
pub fn merge_punctuations(words: &mut Vec<RawWord>) {
    const PREPEND: &str = "\"'\u{201c}\u{bf}([{-";
    const APPEND: &str = "\"'.\u{3002},\u{ff0c}!\u{ff01}?\u{ff1f}:\u{ff1a}\u{201d})]}\u{3001}";
    let n = words.len();
    if n >= 2 {
        let (mut i, mut j) = (n as isize - 2, n - 1);
        while i >= 0 {
            let iu = i as usize;
            let prev = words[iu].word.clone();
            let p = prev.trim();
            if prev.starts_with(' ') && !p.is_empty() && p.chars().count() == 1 && PREPEND.contains(p) {
                let moved = std::mem::take(&mut words[iu].tokens);
                words[j].word = prev + &words[j].word;
                let mut t = moved;
                t.extend(std::mem::take(&mut words[j].tokens));
                words[j].tokens = t;
                words[iu].word.clear();
            } else {
                j = iu;
            }
            i -= 1;
        }
        let (mut i, mut j) = (0usize, 1usize);
        while j < n {
            let f = words[j].word.clone();
            if !words[i].word.ends_with(' ') && !f.is_empty() && f.chars().count() == 1 && APPEND.contains(f.as_str()) {
                words[i].word.push_str(&f);
                let moved = std::mem::take(&mut words[j].tokens);
                words[i].tokens.extend(moved);
                words[j].word.clear();
            } else {
                i = j;
            }
            j += 1;
        }
    }
    words.retain(|w| !w.word.is_empty());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtw_matches_openai() {
        // Paths from openai-whisper's dtw_cpu on the same matrices.
        let x = [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0];
        assert_eq!(dtw(&x, 3, 3), (vec![0, 1, 1, 2, 2], vec![0, 0, 1, 1, 2]));
        let x = [-1.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, -1.0, -3.0];
        assert_eq!(dtw(&x, 3, 4), (vec![0, 1, 1, 2, 2, 2], vec![0, 0, 1, 1, 2, 3]));
    }

    #[test]
    fn punctuation_merges() {
        let w = |s: &str| RawWord {
            word: s.into(),
            tokens: vec![1],
            start: 0.0,
            end: 0.0,
            probability: 1.0,
        };
        let mut words = vec![w(" Hello"), w(","), w(" \""), w("world"), w(".")];
        merge_punctuations(&mut words);
        let s: Vec<&str> = words.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(s, vec![" Hello,", " \"world."]);
    }
}
