//! Signatures of the thinking blocks the Anthropic Messages API returns (`/v1/messages`): an
//! HMAC-SHA256 of the thinking text under the server's key ([`super::ServerOptions::thinking_key`],
//! random at start when none is set), so that the blocks a client sends back in later turns can be
//! told from edited or foreign ones. The HMAC, the digest and the random key are `purecrypto`'s.

use purecrypto::hash::HmacSha256;
use purecrypto::rng::{OsRng, RngCore};

/// Signs and checks thinking blocks with one key.
pub(crate) struct ThinkingSigner {
    key: Vec<u8>,
}

/// The signature's format, first byte of the signed bytes (base64 of version + MAC).
const VERSION: u8 = 1;

/// What the MAC covers in front of the thinking text, so it cannot be replayed for other data.
const DOMAIN: &[u8] = b"oxinfer thinking v1\0";

impl ThinkingSigner {
    /// With `key` (any bytes), or 32 random bytes from the operating system.
    pub fn new(key: Option<&str>) -> Self {
        let key = match key {
            Some(k) => k.as_bytes().to_vec(),
            None => {
                let mut k = vec![0u8; 32];
                OsRng.fill_bytes(&mut k);
                k
            }
        };
        Self { key }
    }

    fn mac(&self, thinking: &str) -> HmacSha256 {
        HmacSha256::new(&self.key).chain(DOMAIN).chain(thinking.as_bytes())
    }

    /// The `signature` of a thinking block with this text.
    pub fn sign(&self, thinking: &str) -> String {
        let mut b = vec![VERSION];
        b.extend_from_slice(&self.mac(thinking).finalize());
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
        self.mac(thinking).verify(&b[1..]).unwrap_u8() == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_vector() {
        // RFC 4231 test case 2: pins the construction used here.
        let mac = HmacSha256::new(b"Jefe").chain(b"what do ya want for nothing?").finalize();
        let hex: String = mac.iter().map(|x| format!("{x:02x}")).collect();
        assert_eq!(hex, "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
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
