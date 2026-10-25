//! The compression ratio Whisper's temperature fallback uses to spot repetition loops:
//! `len(bytes) / len(zlib.compress(bytes))` over the window's tokens (2 bytes each, little
//! endian, as transformers computes it). There is no deflate implementation in the dependency
//! tree, so this sizes a DEFLATE stream with greedy LZ77 (hash chains, 32 KiB window) and the
//! fixed Huffman codes, plus zlib's 6 bytes of framing. zlib proper picks dynamic codes for
//! longer inputs, so ratios can differ from it by some percent; repetition loops, which the
//! threshold (2.4) is meant to catch, compress far beyond it either way.

const WINDOW: usize = 32 * 1024;
const MAX_CHAIN: usize = 128;

/// Bits of a literal byte with the fixed literal/length code.
fn literal_bits(b: u8) -> usize {
    if b < 144 { 8 } else { 9 }
}

/// Bits of a match of `len` (3..=258) at distance `dist` (1..=32768) with the fixed codes.
fn match_bits(len: usize, dist: usize) -> usize {
    // Length symbol 257..285 (7 bits up to 279, 8 after) plus extra bits.
    let (sym, extra) = if len == 258 {
        (285, 0)
    } else {
        let base = [3, 11, 19, 35, 67, 131];
        let k = base.iter().rposition(|&b| len >= b).unwrap();
        let extra = k;
        let first_sym = [257, 265, 269, 273, 277, 281][k];
        let per = 1 << extra;
        (first_sym + if k == 0 { len - 3 } else { (len - base[k]) / per }, extra)
    };
    let sym_bits = if sym <= 279 { 7 } else { 8 };
    // Distance code: 5 bits plus floor(log2(dist - 1)) - 1 extra bits from distance 5 on.
    let dextra = if dist <= 4 {
        0
    } else {
        (usize::BITS - 1 - (dist - 1).leading_zeros()) as usize - 1
    };
    sym_bits + extra + 5 + dextra
}

/// Estimated `zlib.compress(data)` length in bytes.
pub fn zlib_size(data: &[u8]) -> usize {
    let n = data.len();
    let mut bits = 3 + 7; // block header, end-of-block
    let hash = |i: usize| -> usize { ((data[i] as usize) << 10 ^ (data[i + 1] as usize) << 5 ^ data[i + 2] as usize) & 0x7fff };
    let mut head = vec![usize::MAX; 1 << 15];
    let mut prev = vec![usize::MAX; n];
    let insert = |i: usize, head: &mut Vec<usize>, prev: &mut Vec<usize>| {
        if i + 2 < n {
            let h = hash(i);
            prev[i] = head[h];
            head[h] = i;
        }
    };
    let mut i = 0;
    while i < n {
        let (mut best, mut best_d) = (0usize, 0usize);
        if i + 2 < n {
            let mut cand = head[hash(i)];
            let mut chain = 0;
            while cand != usize::MAX && i - cand <= WINDOW && chain < MAX_CHAIN {
                let max = (n - i).min(258);
                let mut l = 0;
                while l < max && data[cand + l] == data[i + l] {
                    l += 1;
                }
                if l > best {
                    best = l;
                    best_d = i - cand;
                    if l == max {
                        break;
                    }
                }
                cand = prev[cand];
                chain += 1;
            }
        }
        if best >= 3 {
            bits += match_bits(best, best_d);
            for j in i..i + best {
                insert(j, &mut head, &mut prev);
            }
            i += best;
        } else {
            bits += literal_bits(data[i]);
            insert(i, &mut head, &mut prev);
            i += 1;
        }
    }
    2 + bits.div_ceil(8) + 4
}

/// transformers' `_retrieve_compression_ratio` over token ids.
pub fn compression_ratio(tokens: &[u32]) -> f32 {
    if tokens.is_empty() {
        return 0.0;
    }
    let bytes: Vec<u8> = tokens.iter().flat_map(|&t| (t as u16).to_le_bytes()).collect();
    bytes.len() as f32 / zlib_size(&bytes) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_compresses() {
        let text: Vec<u32> = (0..60).map(|i| 1000 + (i * 37 % 211)).collect();
        let r = compression_ratio(&text);
        assert!(r < 1.3, "varied text ratio {r}");
        let mut looped = Vec::new();
        for _ in 0..30 {
            looped.extend_from_slice(&[440, 1029, 291, 13]);
        }
        let r = compression_ratio(&looped);
        assert!(r > 2.4, "loop ratio {r}");
    }

    #[test]
    fn match_bits_table() {
        assert_eq!(match_bits(3, 1), 7 + 5);
        assert_eq!(match_bits(10, 4), 7 + 5);
        assert_eq!(match_bits(11, 5), 7 + 1 + 5 + 1);
        assert_eq!(match_bits(258, 32768), 8 + 5 + 13);
    }
}
