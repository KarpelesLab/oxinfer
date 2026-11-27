// Checks the command classification of the opencode plugin (src/opencode/oxinfer-outcomes.js):
//   node tools/opencode-oxinfer/test.mjs
// The hooks themselves need opencode; docs/learning.md says what was run with it.
import assert from "node:assert/strict";
import { OxinferOutcomes } from "../../src/opencode/oxinfer-outcomes.js";

const { classify, verdict, settings, DEFAULTS } = OxinferOutcomes;
const kind = (c, cfg) => classify(c, cfg)?.kind ?? null;

// What is a check, and of which kind.
for (const [cmd, want] of [
  ["cargo test", "tests"],
  ["cargo test --lib opencode -- --nocapture", "tests"],
  ["cargo +nightly test", "tests"],
  ["RUST_BACKTRACE=1 cargo test 2>&1", "tests"],
  ["cd crates/x && cargo test", "tests"],
  ["which make && make test 2>&1", "tests"],
  ["timeout 600 go test ./...", "tests"],
  ["python -m pytest -x tests/", "tests"],
  ["npm test", "tests"],
  ["npm run test:unit", "tests"],
  ["pnpm exec vitest run", "tests"],
  ["cargo test > /tmp/log.txt 2>&1", "tests"],
  ["cargo build --release", "build"],
  ["cargo check --all-targets", "build"],
  ["make", "build"],
  ["npx tsc --noEmit", "build"],
  ["cargo clippy --all-targets -- -D warnings", "lint"],
  ["cargo fmt --check", "lint"],
  ["cargo fmt --all -- --check", "lint"],
  ["cargo build && cargo test", "tests"],
  // Not checks, or a status that is not the check's.
  ["cargo test | tail -5", null],
  ["cargo test; echo done", null],
  ["cargo test || true", null],
  ["cargo test &", null],
  ["echo $(cargo test)", null],
  ["cargo test\ncargo build", null],
  ["cargo test --no-run", null],
  ["cargo test --help", null],
  ["pytest --collect-only", null],
  ["cargo fmt", null],
  ["cargo testx", null],
  ["make clean", null],
  ["make install", null],
  ["echo cargo test", null],
  ["grep -rn 'cargo test' .", null],
  ["ls", null],
  ["", null],
  [undefined, null],
]) {
  assert.equal(kind(cmd), want, String(cmd));
}

// A failure is the check's only when nothing else could have failed.
assert.equal(classify("cd x && cargo test").bad, true);
assert.equal(classify("FOO=1 cargo test 2>&1").bad, true);
assert.equal(classify("which make && make test").bad, false);
assert.equal(classify("cargo build && cargo test").bad, false);
assert.equal(classify("cargo test && git commit -m x").bad, false);

assert.deepEqual(verdict("cargo test", 0), { result: "good", kind: "tests", note: "exit 0: cargo test" });
assert.deepEqual(verdict("cd x && cargo  test", 101), { result: "bad", kind: "tests", note: "exit 101: cd x && cargo test" });
assert.equal(verdict("which make && make test", 0).result, "good");
for (const [cmd, exit] of [
  ["which make && make test", 1],
  ["cargo build && cargo test", 101],
  ["cargo test", undefined], // a timeout: no status
  ["cargo test", 127], // not found
  ["cargo test", 126],
  ["cargo test", 137], // killed
  ["cargo test", 1.5],
  ["ls", 0],
]) {
  assert.equal(verdict(cmd, exit), null, `${cmd} -> ${exit}`);
}
assert.equal(verdict("cargo test " + "x".repeat(400), 0).note.length, "exit 0: ".length + 200);

// Settings: a listed kind replaces its defaults, a new one is added, the rest stays.
const cfg = settings(null, { patterns: { tests: ["\\./run-tests\\.sh"], e2e: ["make e2e"], lint: [] }, tool: false }, undefined);
assert.equal(kind("./run-tests.sh --fast", cfg), "tests");
assert.equal(kind("cargo test", cfg), null);
assert.equal(kind("make e2e", cfg), "e2e");
assert.equal(kind("cargo clippy", cfg), null);
assert.equal(kind("cargo build", cfg), "build");
assert.equal(cfg.tool, false);
assert.deepEqual(cfg.providers, ["oxinfer"]);
assert.equal(kind("cargo test", settings({ skip: ["cargo"] })), null);
assert.deepEqual(settings().patterns, DEFAULTS.patterns);

console.log("ok");
