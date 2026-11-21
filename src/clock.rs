//! Monotonic time. `std::time::Instant` is unavailable on `wasm32-unknown-unknown` (it panics), so
//! there the host supplies a millisecond clock through the `host_now_ms` import. The same goes
//! for `std::time::SystemTime`: [`unix_time`] is the wall clock on every target.

/// The time since the Unix epoch. On `wasm32-unknown-unknown` it is the host's clock
/// (`performance.timeOrigin + performance.now()` in `web/src/lib/oxinfer.js`).
pub fn unix_time() -> std::time::Duration {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
    }
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        web::unix_time()
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use std::time::Instant;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub use web::Instant;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod web {
    use std::time::Duration;

    #[link(wasm_import_module = "env")]
    unsafe extern "C" {
        fn host_now_ms() -> f64;
    }

    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct Instant(f64);

    pub fn unix_time() -> Duration {
        Duration::from_secs_f64((unsafe { host_now_ms() } / 1e3).max(0.0))
    }

    impl Instant {
        pub fn now() -> Self {
            Instant(unsafe { host_now_ms() })
        }
        pub fn elapsed(&self) -> Duration {
            Self::now().duration_since(*self)
        }
        pub fn duration_since(&self, earlier: Instant) -> Duration {
            Duration::from_secs_f64(((self.0 - earlier.0) / 1e3).max(0.0))
        }
    }
}
