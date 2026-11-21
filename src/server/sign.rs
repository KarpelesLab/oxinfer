//! Signatures of the thinking blocks the Anthropic Messages API returns (`/v1/messages`): an
//! HMAC-SHA256 of the thinking text under the server's key ([`super::ServerOptions::thinking_key`],
//! random at start when none is set), so that the blocks a client sends back in later turns can be
//! told from edited or foreign ones.

/// SHA-256 (FIPS 180-4).
pub(crate) fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
        0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
        0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
        0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
        0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, c) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*c);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 32];
    for (o, x) in out.as_chunks_mut::<4>().0.iter_mut().zip(h) {
        *o = x.to_be_bytes();
    }
    out
}

/// HMAC-SHA256 (RFC 2104).
pub(crate) fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 64];
    if key.len() > 64 {
        k[..32].copy_from_slice(&sha256(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut inner: Vec<u8> = k.iter().map(|b| b ^ 0x36).collect();
    inner.extend_from_slice(msg);
    let mut outer: Vec<u8> = k.iter().map(|b| b ^ 0x5c).collect();
    outer.extend_from_slice(&sha256(&inner));
    sha256(&outer)
}

/// 32 random bytes: the operating system's generator where there is one (`/dev/urandom`), else
/// the standard library's randomly keyed hasher, mixed with the time and the process id.
pub(crate) fn random_key() -> Vec<u8> {
    use std::io::Read;
    let mut key = vec![0u8; 32];
    if std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut key)).is_ok() {
        return key;
    }
    use std::hash::{BuildHasher, Hasher};
    let mut seed = Vec::new();
    for i in 0..8u64 {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(i);
        seed.extend_from_slice(&h.finish().to_le_bytes());
    }
    seed.extend_from_slice(&crate::clock::unix_time().as_nanos().to_le_bytes());
    seed.extend_from_slice(&std::process::id().to_le_bytes());
    sha256(&seed).to_vec()
}

/// Signs and checks thinking blocks with one key.
pub(crate) struct ThinkingSigner {
    key: Vec<u8>,
}

/// The signature's format, first byte of the signed bytes (base64 of version + MAC).
const VERSION: u8 = 1;

impl ThinkingSigner {
    /// With `key` (any bytes), or a random one.
    pub fn new(key: Option<&str>) -> Self {
        Self {
            key: key.map_or_else(random_key, |k| k.as_bytes().to_vec()),
        }
    }

    fn mac(&self, thinking: &str) -> [u8; 32] {
        let mut msg = b"oxinfer thinking v1\0".to_vec();
        msg.extend_from_slice(thinking.as_bytes());
        hmac_sha256(&self.key, &msg)
    }

    /// The `signature` of a thinking block with this text.
    pub fn sign(&self, thinking: &str) -> String {
        let mut b = vec![VERSION];
        b.extend_from_slice(&self.mac(thinking));
        crate::b64::encode(&b)
    }

    /// Whether `signature` is this server's signature of `thinking` (compared in constant time).
    pub fn verify(&self, thinking: &str, signature: &str) -> bool {
        let Some(b) = crate::b64::decode(signature.trim()) else {
            return false;
        };
        if b.len() != 33 || b[0] != VERSION {
            return false;
        }
        let mac = self.mac(thinking);
        b[1..].iter().zip(mac).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn sha256_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(&sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            hex(&sha256(&vec![b'a'; 1_000_000])),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn hmac_vectors() {
        // RFC 4231, test cases 1, 2 and 6 (a key longer than the block).
        assert_eq!(
            hex(&hmac_sha256(&[0x0b; 20], b"Hi There")),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        assert_eq!(
            hex(&hmac_sha256(b"Jefe", b"what do ya want for nothing?")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert_eq!(
            hex(&hmac_sha256(
                &[0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            )),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn signatures() {
        let s = ThinkingSigner::new(Some("secret"));
        let sig = s.sign("Let me think.");
        assert!(s.verify("Let me think.", &sig));
        assert!(!s.verify("Let me think!", &sig));
        assert!(!s.verify("Let me think.", ""));
        assert!(!s.verify("Let me think.", "not base64 at all"));
        assert!(!ThinkingSigner::new(Some("other")).verify("Let me think.", &sig));
        assert_eq!(sig, ThinkingSigner::new(Some("secret")).sign("Let me think."));
        let (a, b) = (ThinkingSigner::new(None), ThinkingSigner::new(None));
        assert_ne!(a.sign("x"), b.sign("x"));
    }
}
