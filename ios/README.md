# oxinfer on iOS

Rust apps use the crate directly (`oxinfer::engine`, see the crate docs): add `oxinfer` as a
dependency and build for `aarch64-apple-ios` / `aarch64-apple-ios-sim` as usual. Everything
below is for Swift / Objective-C hosts, through the C API (`include/oxinfer.h`, cargo feature
`capi`), and for `OxinferTest`, a minimal SwiftUI harness to try models on a phone or in the
Simulator.

## Build the library

Requirements: Xcode, rustup with the iOS targets:

```sh
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
ios/build-rust.sh   # -> ios/build/Oxinfer.xcframework (device + Apple-silicon Simulator)
```

The script runs `cargo build-capi --target T` (an alias for
`cargo rustc --release --lib --features capi --crate-type staticlib`) for both targets and
packages `target/T/release/liboxinfer.a` with the header and a module map (Swift: `import
Oxinfer`). The library needs no extra system frameworks or libraries (Metal and the
Objective-C runtime are opened at run time).

## Run the test app on an iPhone

1. `ios/build-rust.sh` (again after every Rust change).
2. Open `ios/OxinferTest/OxinferTest.xcodeproj` in Xcode.
3. Target OxinferTest > Signing & Capabilities: pick your Team, and change the bundle
   identifier (`dev.oxinfer.OxinferTest`) to one of yours if Xcode asks.
   The entitlements file requests `com.apple.developer.kernel.increased-memory-limit` and
   `com.apple.developer.kernel.extended-virtual-addressing` (needed to map an 11-18 GB GGUF
   file). Automatic signing adds both capabilities to the App ID; they need a paid developer
   account (with a free Personal Team, remove the entitlements file from the target's
   "Code Signing Entitlements" setting and stay with small models).
4. Select your iPhone and Run. The first run on a device asks you to trust the developer
   certificate (Settings > General > VPN & Device Management).

Use a Release build for speed (Product > Scheme > Edit Scheme > Run > Build Configuration;
the Rust library itself is always built in release mode).

### Models

- **Download in the app**: type a repo id and Load. `Qwen/Qwen3.5-0.8B` (1.7 GB, safetensors),
  or `org/repo:QUANT` for one GGUF file of a repo (e.g. `unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M`).
  Files go to `Documents/hub/` (resumable; excluded from iCloud backup).
- **Copy from a Mac**: Finder > your iPhone > Files > oxinfer, drop a model folder (with
  `config.json`, `tokenizer.json`, `*.safetensors`) or a `.gguf` file; or use the Files app
  (On My iPhone > oxinfer). Then pick it under "On device".

Options (one `key=value` per line, see `oxinfer.h`): `quant=q8` or `q4` for bf16 checkpoints
(GGUF and pre-quantized checkpoints keep their format), `mtp=1` for Qwen3.5/3.8 speculative
decoding, `mtp_from=PATH` for the MTP sidecar, `device=cpu|metal`, `threads=N`,
`max_context=N`. `mmap` defaults to on for iOS: mapped weights are clean file pages, which do
not count against the app's memory limit the way copied ones do.

### Models larger than the phone's memory (Qwen3.8-27B on a 12 GB iPhone)

Nothing to set: a Qwen or Gemma checkpoint larger than the device holds streams from flash by
default (see "Streaming weights from flash" in the main README). The budget comes from the
device: what `os_proc_available_memory` leaves the app minus 1 GB, at most 60% of the phone's
memory and 85% of Metal's working set (`recommendedMaxWorkingSetSize`). The resident layers
stay wrapped for the GPU, the others are read ahead and wrapped per pass as they arrive, and a
dense model with an MTP layer drafts up to 8 tokens per pass (cost-aware). `oxi_engine_info_json`
reports what was chosen (`stream`, `device_memory`). Override with options:
`cache_budget_gb=5` (page cache the weights may use), `stream=off`, `stream_gpu=0` (decode on
the CPU), `draft=none` (plain decoding), `threads=6`. With the increased-memory-limit
entitlement the app gets more, and the default budget follows.

## Simulator

```sh
ios/build-rust.sh
cd ios/OxinferTest
xcodebuild -project OxinferTest.xcodeproj -scheme OxinferTest -configuration Release \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro Max' -derivedDataPath build build
xcrun simctl boot 'iPhone 17 Pro Max'
xcrun simctl install booted build/Build/Products/Release-iphonesimulator/OxinferTest.app
# Unattended run: load a model (Simulator apps can read host paths) and send a prompt;
# the output and stats go to the console.
xcrun simctl launch --console booted dev.oxinfer.OxinferTest \
  -model ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B -options 'quant=q8' -prompt 'Write a haiku.'
xcrun simctl io booted screenshot shot.png
```

## C API smoke test (macOS)

```sh
cargo build-capi --target aarch64-apple-darwin
cc -O2 -Iinclude ios/capi_smoke.c target/aarch64-apple-darwin/release/liboxinfer.a -o /tmp/capi_smoke
/tmp/capi_smoke Qwen/Qwen3.5-0.8B 'quant=q8'
```
