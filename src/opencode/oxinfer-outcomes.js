// oxinfer-outcomes: an opencode plugin that tells an oxinfer server how the replies of a
// MODEL@PROFILE session turned out (docs/learning.md, "opencode").
//
// It does nothing in a session whose model is not `MODEL@PROFILE` of one of the providers
// below. In one that is:
//   - when the bash tool finishes a command recognised as a test, build or lint run, the exit
//     status is reported as an outcome (`kind`: tests, build, lint; good for 0, bad otherwise)
//     of the session's replies since its previous outcome (not when the same command gave the
//     same status just before, with no other tool call in between);
//   - the model has a `report_outcome` tool for results no pattern recognises.
// The report is `POST <provider baseURL>/outcomes` with the provider's API key and
// `{"session": <opencode session id>, "result", "kind", "note"}`: opencode does not show a
// plugin the provider's response ids, so the server finds the replies by the session id
// opencode sends with each request (`X-Session-Id`).
//
// Installed by oxinfer (`POST /api/web/opencode/plugin`, or with a profile's entry) as
// ~/.config/opencode/plugins/oxinfer-outcomes.js, or copied there by hand. Written for the
// plugin API of opencode 1.x (`@opencode-ai/plugin`); checked with opencode 1.19.6.
//
// Settings, all optional: ~/.config/opencode/oxinfer-outcomes.json (beside opencode.json), or
// the plugin's options when it is listed in opencode.json's "plugin" with options:
//   {
//     "providers": ["oxinfer"],          // provider ids whose MODEL@PROFILE sessions report
//     "tool": true,                      // false: no report_outcome tool
//     "patterns": {                      // per kind, regular expressions matched at the start of
//       "tests": ["cargo test", "..."],  // a command; a kind listed here replaces its defaults
//       "e2e": ["make e2e"]              // ([] turns it off), a new name adds a kind
//     },
//     "skip": ["--help"]                 // regular expressions: a command matching one is ignored
//   }

// The plugin function is the file's only export: opencode calls every export of a plugin file as
// a plugin. The pure parts hang off it for tests (tools/opencode-oxinfer/test.mjs).

import { readFileSync } from "node:fs";

const DEFAULTS = {
  providers: ["oxinfer"],
  tool: true,
  patterns: {
    tests: [
      "cargo (?:\\+\\S+ )?(?:test|nextest run)",
      "go test",
      "(?:python3? -m )?pytest",
      "python3? -m unittest",
      "tox",
      "(?:npm|pnpm|yarn|bun) (?:run )?test(?::\\S+)?",
      "(?:npx |pnpm exec |bunx )?(?:jest|vitest run|mocha)",
      "deno test",
      "make (?:test|check)",
      "ctest",
      "mvn (?:test|verify)",
      "(?:\\./gradlew|gradle) (?:test|check)",
      "dotnet test",
      "swift test",
      "zig build test",
      "mix test",
      "(?:bundle exec )?rspec",
      "(?:\\./)?(?:vendor/bin/)?phpunit",
    ],
    lint: [
      "cargo (?:\\+\\S+ )?clippy",
      "cargo (?:\\+\\S+ )?fmt (?:.* )?--check",
      "go vet",
      "golangci-lint run",
      "(?:npx |pnpm exec |bunx )?eslint",
      "(?:npm|pnpm|yarn|bun) run lint",
      "ruff check",
      "flake8",
      "mypy",
      "shellcheck",
    ],
    build: [
      "cargo (?:\\+\\S+ )?(?:build|check)",
      "go build",
      "make(?: (?:all|build))?$",
      "(?:npm|pnpm|yarn|bun) run build",
      "(?:npx |pnpm exec |bunx )?tsc",
      "cmake --build",
      "ninja",
      "mvn (?:compile|package)",
      "(?:\\./gradlew|gradle) (?:build|assemble)",
      "dotnet build",
      "swift build",
      "zig build$",
    ],
  },
  skip: ["(?:^|\\s)(?:--help|-h|--version|--no-run|--list|--collect-only|--dry-run)(?:\\s|=|$)"],
};

/** The settings: `DEFAULTS` with the given objects over them (later ones win). */
function settings(...over) {
  const cfg = { ...DEFAULTS, patterns: { ...DEFAULTS.patterns } };
  for (const o of over) {
    if (!o || typeof o !== "object") continue;
    if (Array.isArray(o.providers)) cfg.providers = o.providers.map(String);
    if (typeof o.tool === "boolean") cfg.tool = o.tool;
    if (Array.isArray(o.skip)) cfg.skip = o.skip.map(String);
    if (o.patterns && typeof o.patterns === "object") {
      for (const [kind, list] of Object.entries(o.patterns)) {
        if (Array.isArray(list)) cfg.patterns[kind] = list.map(String);
      }
    }
  }
  return cfg;
}

// One simple command: environment assignments and a few wrappers in front of it do not count.
const PREFIX = /^(?:(?:[A-Za-z_][A-Za-z0-9_]*=\S*|time|nice|command|timeout\s+\S+)\s+)+/;
// opencode's own agents whose replies are not the session's work.
const SIDE = new Set(["title", "summary", "compaction"]);
// Commands that only set the scene for the one that is judged.
const SETUP = /^(?:cd|pushd|export|set)(?:\s|$)/;

/**
 * What a shell command's exit status says, or null when it says nothing reliable.
 * `{ kind, bad }`: the kind of check the command is, and whether a non-zero status can be
 * read as that check failing (false: only its success counts).
 *
 * Conservative on purpose: a command with a pipe, `;`, `||`, `&`, a subshell or several lines is
 * ignored (its status is not the check's), `2>&1` and output redirections are fine, and
 * `a && b` chains are followed: status 0 means every part passed; a failure is only blamed on
 * the check when it is the only part beside `cd` / `export`.
 */
function classify(command, cfg = DEFAULTS) {
  if (typeof command !== "string") return null;
  const text = command.trim().replace(/\s*\d?>&\d/g, "");
  if (!text || cfg.skip.some((s) => new RegExp(s).test(text))) return null;
  const parts = text.split("&&").map((p) => p.trim());
  if (parts.some((p) => !p || /[|;&`\n<]|\$\(/.test(p))) return null;
  let kind = null;
  let checks = 0;
  let others = 0;
  for (const part of parts) {
    const cmd = part.replace(PREFIX, "").replace(/[ \t]+/g, " ");
    const found = Object.entries(cfg.patterns).find(([, list]) =>
      list.some((p) => new RegExp(`^(?:${p})(?=\\s|$)`).test(cmd)),
    );
    if (found) {
      kind = found[0];
      checks += 1;
    } else if (cmd && !SETUP.test(cmd)) {
      others += 1;
    }
  }
  return kind ? { kind, bad: checks === 1 && others === 0 } : null;
}

/**
 * The outcome of a finished shell command: `{ result, kind, note }`, or null when there is
 * none to report (not a check, no exit status, killed, not found, or an ambiguous failure).
 */
function verdict(command, exit, cfg = DEFAULTS) {
  const c = classify(command, cfg);
  if (!c || !Number.isInteger(exit)) return null;
  if (exit !== 0 && (!c.bad || exit === 126 || exit === 127 || exit >= 128)) return null;
  const line = command.trim().replace(/\s+/g, " ");
  return {
    result: exit === 0 ? "good" : "bad",
    kind: c.kind,
    note: `exit ${exit}: ${line.length > 200 ? line.slice(0, 197) + "..." : line}`,
  };
}

function fileSettings() {
  try {
    return JSON.parse(readFileSync(new URL("../oxinfer-outcomes.json", import.meta.url), "utf8"));
  } catch {
    return null;
  }
}

export const OxinferOutcomes = async ({ client }, options) => {
  const cfg = settings(fileSettings(), options);
  // The profile sessions seen: where and with which key their outcomes are reported.
  const sessions = new Map();

  const log = async (level, message) => {
    try {
      await client.app.log({ body: { service: "oxinfer-outcomes", level, message } });
    } catch {}
  };

  // Whether the request is one of a profile of an oxinfer server.
  const profiled = (input) => {
    const model = input.model?.api?.id ?? input.model?.id;
    return (
      cfg.providers.includes(input.model?.providerID) &&
      typeof model === "string" &&
      model.lastIndexOf("@") > 0 &&
      typeof input.provider?.options?.baseURL === "string"
    );
  };

  // Notes where the session's outcomes go (or that they go nowhere: another model was picked).
  const see = (input) => {
    if (SIDE.has(input.agent)) return;
    if (!profiled(input)) return void sessions.delete(input.sessionID);
    const options = input.provider.options;
    const s = sessions.get(input.sessionID) ?? {};
    sessions.set(
      input.sessionID,
      Object.assign(s, {
        url: options.baseURL.replace(/\/+$/, ""),
        key: options.apiKey,
        model: input.model?.api?.id ?? input.model?.id,
      }),
    );
  };

  /** Reports `outcome` for the session's replies since its previous outcome; a line saying what happened. */
  const report = async (sessionID, outcome) => {
    const s = sessions.get(sessionID);
    if (!s) return "Nothing recorded: this session's model is not a MODEL@PROFILE of an oxinfer server.";
    try {
      const res = await fetch(`${s.url}/outcomes`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(s.key ? { Authorization: `Bearer ${s.key}` } : {}) },
        body: JSON.stringify({ session: sessionID, ...outcome }),
        signal: AbortSignal.timeout(10000),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        const why = body?.error?.message ?? `HTTP ${res.status}`;
        await log("warn", `${s.model}: outcome not recorded: ${why}`);
        return `Not recorded: ${why}`;
      }
      await log("info", `${s.model}: ${outcome.kind} ${outcome.result} recorded for ${body.recorded} replies`);
      return `Recorded ${outcome.result} (${outcome.kind}) for ${body.recorded} replies.`;
    } catch (e) {
      await log("warn", `${s.model}: outcome not sent to ${s.url}: ${e}`);
      return `Not recorded: ${e}`;
    }
  };

  const hooks = {
    "chat.params": async (input) => {
      see(input);
    },
    "chat.headers": async (input, output) => {
      see(input);
      if (!profiled(input)) return;
      // The session's name for the server (opencode 1.19 sends it itself; older ones do not).
      // The replies opencode asks for on the side (the session's title, summaries) are named
      // apart: a test run says nothing about them.
      output.headers["X-Session-Id"] = SIDE.has(input.agent) ? `${input.sessionID}.${input.agent}` : input.sessionID;
    },
    "tool.execute.after": async (input, output) => {
      const s = sessions.get(input.sessionID);
      if (!s) return;
      const command = input.tool === "bash" ? input.args?.command : undefined;
      const exit = output?.metadata?.exit;
      const v = verdict(command, exit, cfg);
      if (!v) {
        // Something else was done: the same check may say something new afterwards.
        s.last = undefined;
        return;
      }
      // The same check with the same result and nothing done in between judges nothing new (a
      // model running its tests in a loop is not rewarded once per run).
      if (s.last?.command === command && s.last?.exit === exit) return;
      s.last = { command, exit };
      await report(input.sessionID, v);
    },
  };

  if (cfg.tool) {
    const { tool } = await import("@opencode-ai/plugin");
    hooks.tool = {
      report_outcome: tool({
        description:
          "Record how the work of this session turned out since the last recorded result, for the model server's " +
          "learning log. Call it only when something outside you judged the work and no test, build or lint " +
          "command's exit status already said so: a check you ran by other means, or the user saying the " +
          "change is right or wrong. Do not report your own opinion of your work, and do not guess.",
        args: {
          result: tool.schema.enum(["good", "bad"]).describe("good: the check passed or the user accepted; bad: it failed or the user rejected"),
          kind: tool.schema
            .string()
            .optional()
            .describe("what judged it: tests, build, lint, user_accepted, user_rejected, or another short word"),
          note: tool.schema.string().optional().describe("one line saying what was checked and what it showed"),
        },
        async execute(args, context) {
          return report(context.sessionID, {
            result: args.result,
            kind: args.kind?.trim() || "reported",
            ...(args.note ? { note: args.note.slice(0, 500) } : {}),
          });
        },
      }),
    };
  }
  return hooks;
};

Object.assign(OxinferOutcomes, { DEFAULTS, settings, classify, verdict });
