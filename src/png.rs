//! A dependency-free PNG writer for 8-bit RGB images (zlib "stored" blocks: no compression, so
//! the file is about 3 bytes per pixel, but it works on every target, the browser included).

fn crc32(data: &[u8], mut c: u32) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut v = i as u32;
            for _ in 0..8 {
                v = if v & 1 != 0 { 0xEDB8_8320 ^ (v >> 1) } else { v >> 1 };
            }
            *e = v;
        }
        t
    });
    c = !c;
    for &b in data {
        c = t[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    !c
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let c = crc32(data, crc32(kind, 0));
    out.extend_from_slice(&c.to_be_bytes());
}

/// Encodes `rgb` (`width * height * 3` bytes, row-major) as a PNG file.
pub fn encode_rgb(width: usize, height: usize, rgb: &[u8]) -> Vec<u8> {
    encode(width, height, rgb, 3)
}

/// Encodes `rgba` (`width * height * 4` bytes, row-major, straight alpha) as a PNG file.
pub fn encode_rgba(width: usize, height: usize, rgba: &[u8]) -> Vec<u8> {
    encode(width, height, rgba, 4)
}

/// 8-bit truecolour (`ch` 3) or truecolour with alpha (`ch` 4).
fn encode(width: usize, height: usize, rgb: &[u8], ch: usize) -> Vec<u8> {
    assert_eq!(rgb.len(), width * height * ch, "pixel buffer does not match the size");
    // Scanlines with filter type 0.
    let mut raw = Vec::with_capacity(height * (width * ch + 1));
    for row in rgb.chunks_exact(width * ch) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut z = vec![0x78, 0x01];
    let mut blocks = raw.chunks(65535).peekable();
    while let Some(b) = blocks.next() {
        z.push(if blocks.peek().is_none() { 1 } else { 0 });
        z.extend_from_slice(&(b.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(b.len() as u16)).to_le_bytes());
        z.extend_from_slice(b);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());

    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    // 8-bit, truecolour (with alpha), deflate, adaptive filter, no interlace
    ihdr.extend_from_slice(&[8, if ch == 4 { 6 } else { 2 }, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn crc_matches_the_png_spec_example() {
        // CRC of the IEND chunk type is the well-known AE 42 60 82.
        assert_eq!(super::crc32(b"IEND", 0), 0xAE42_6082);
    }
}
