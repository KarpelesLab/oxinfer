// A thin Swift wrapper over the oxinfer C API (include/oxinfer.h). Calls block: run them off the
// main thread. `cancel()` may be called from any thread.

import Foundation
import Oxinfer

struct OxinferError: Error, CustomStringConvertible {
    let description: String
    static func last() -> OxinferError { OxinferError(description: String(cString: oxi_last_error())) }
}

/// Boxes a Swift closure for a C callback's `user` pointer.
private final class Box<T> {
    let value: T
    init(_ value: T) { self.value = value }
}

private func takeString(_ p: UnsafeMutablePointer<CChar>?) -> String? {
    guard let p else { return nil }
    defer { oxi_string_free(p) }
    return String(cString: p)
}

/// Progress: file, bytes done, total (0 unknown), file index, file count; return false to cancel.
typealias ProgressHandler = (String, UInt64, UInt64, Int, Int) -> Bool

private let progressThunk: oxi_progress_fn = { user, file, done, total, index, count in
    let h = Unmanaged<Box<ProgressHandler>>.fromOpaque(user!).takeUnretainedValue().value
    return h(file.map { String(cString: $0) } ?? "", done, total, Int(index), Int(count)) ? 0 : 1
}

final class OxinferEngine: @unchecked Sendable {
    private let handle: OpaquePointer

    static var version: String { String(cString: oxi_version()) }
    static var devices: String { takeString(oxi_devices()) ?? "" }
    /// Degraded-mode warnings (a JSON array of strings, "[]" when none).
    static var warnings: String { takeString(oxi_warnings()) ?? "[]" }

    /// Downloads `spec` ("org/name" or "org/name:QUANT") into `cacheDir`; returns the local path.
    static func download(_ spec: String, cacheDir: String?, progress: @escaping ProgressHandler) throws -> String {
        let box = Box(progress)
        let user = Unmanaged.passUnretained(box).toOpaque()
        let p = withExtendedLifetime(box) { oxi_download(spec, cacheDir, progressThunk, user) }
        guard let path = takeString(p) else { throw OxinferError.last() }
        return path
    }

    /// Loads a model; `options` are "key=value" lines (see oxinfer.h).
    init(model: String, options: String, progress: @escaping ProgressHandler = { _, _, _, _, _ in true }) throws {
        let box = Box(progress)
        let user = Unmanaged.passUnretained(box).toOpaque()
        let h = withExtendedLifetime(box) { oxi_engine_load(model, options, progressThunk, user) }
        guard let h else { throw OxinferError.last() }
        handle = h
    }

    deinit { oxi_engine_free(handle) }

    func set(_ key: String, _ value: String?) throws {
        if oxi_engine_set(handle, key, value) < 0 { throw OxinferError.last() }
    }

    enum Finish: Int32 { case stop = 0, length = 1, cancelled = 2 }

    /// Sends a user message; `onText(text, thinking)` receives the streamed reply.
    @discardableResult
    func send(_ text: String, onText: @escaping (String, Bool) -> Void) throws -> Finish {
        try run(text, onText: onText) { oxi_engine_send($0, $1, $2, $3) }
    }

    /// Raw completion of `text` (no chat template).
    @discardableResult
    func complete(_ text: String, onText: @escaping (String, Bool) -> Void) throws -> Finish {
        try run(text, onText: onText) { oxi_engine_complete($0, $1, $2, $3) }
    }

    private func run(
        _ text: String, onText: @escaping (String, Bool) -> Void,
        _ call: (OpaquePointer, UnsafePointer<CChar>, oxi_token_fn, UnsafeMutableRawPointer) -> Int32
    ) throws -> Finish {
        typealias Handler = (String, Bool) -> Void
        let box = Box<Handler>(onText)
        let user = Unmanaged.passUnretained(box).toOpaque()
        let thunk: oxi_token_fn = { user, ptr, len, thinking in
            let h = Unmanaged<Box<Handler>>.fromOpaque(user!).takeUnretainedValue().value
            let buf = UnsafeRawBufferPointer(start: ptr, count: len)
            h(String(decoding: buf, as: UTF8.self), thinking != 0)
            return 0
        }
        let rc = withExtendedLifetime(box) { text.withCString { call(handle, $0, thunk, user) } }
        guard let f = Finish(rawValue: rc) else { throw OxinferError.last() }
        return f
    }

    func reset() { _ = oxi_engine_reset(handle) }
    func cancel() { oxi_engine_cancel(handle) }

    var infoJSON: String { takeString(oxi_engine_info_json(handle)) ?? "{}" }
    var statsJSON: String { takeString(oxi_engine_stats_json(handle)) ?? "{}" }

    var stats: [String: Any] {
        (try? JSONSerialization.jsonObject(with: Data(statsJSON.utf8))) as? [String: Any] ?? [:]
    }
}
