/* Greedy run through the C API, for comparing devices (the same text on the Mac, the Simulator
 * and a phone): no system prompt, temperature 0, one message, then the info and stats JSON.
 *
 * macOS:
 *   cargo build-capi --target aarch64-apple-darwin
 *   cc -O2 -Iinclude ios/capi_greedy.c target/aarch64-apple-darwin/release/liboxinfer.a -o /tmp/capi_greedy
 * Simulator (Apple silicon; the binary runs inside a booted Simulator):
 *   IPHONEOS_DEPLOYMENT_TARGET=17.0 cargo build-capi --target aarch64-apple-ios-sim
 *   xcrun -sdk iphonesimulator clang -target arm64-apple-ios17.0-simulator -O2 -Iinclude \
 *     ios/capi_greedy.c target/aarch64-apple-ios-sim/release/liboxinfer.a -o /tmp/capi_greedy_sim
 *   xcrun simctl spawn booted /tmp/capi_greedy_sim MODEL OPTIONS PROMPT [MAX_TOKENS]
 * (Environment variables reach the spawned program with the SIMCTL_CHILD_ prefix, e.g.
 * SIMCTL_CHILD_OXINFER_APP_MEM_GB=6 to give it a phone's app limit.)
 *
 * Usage: capi_greedy MODEL OPTIONS PROMPT [MAX_TOKENS]   (OPTIONS: "key=value;..." or "") */
#include "oxinfer.h"
#include <stdio.h>
#include <stdlib.h>

static int on_text(void *user, const char *text, size_t len, int thinking) {
	(void)user;
	(void)thinking;
	fwrite(text, 1, len, stdout);
	fflush(stdout);
	return 0;
}

int main(int argc, char **argv) {
	if (argc < 4) {
		fprintf(stderr, "usage: %s MODEL OPTIONS PROMPT [MAX_TOKENS]\n", argv[0]);
		return 2;
	}
	OxiEngine *e = oxi_engine_load(argv[1], argv[2], NULL, NULL);
	if (!e) {
		fprintf(stderr, "load failed: %s\n", oxi_last_error());
		return 1;
	}
	char *info = oxi_engine_info_json(e);
	fprintf(stderr, "info: %s\n", info);
	oxi_string_free(info);
	if (oxi_engine_set(e, "system", "") < 0 || oxi_engine_set(e, "temperature", "0") < 0 ||
	    oxi_engine_set(e, "max_tokens", argc > 4 ? argv[4] : "32") < 0) {
		fprintf(stderr, "set failed: %s\n", oxi_last_error());
		return 1;
	}
	printf("=== output\n");
	int rc = oxi_engine_send(e, argv[3], on_text, NULL);
	printf("\n=== end\n");
	char *st = oxi_engine_stats_json(e);
	fprintf(stderr, "finish %d, stats: %s\n", rc, st);
	oxi_string_free(st);
	oxi_engine_free(e);
	return rc < 0;
}
