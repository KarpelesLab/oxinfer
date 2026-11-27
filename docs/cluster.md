# Two machines

Two machines joined by a fast link can run oxinfer together in two ways:

- **Peers**: one `oxinfer serve` is the endpoint clients talk to, and sends requests to the
  other machine's server when that one has the model, the conversation or the free capacity.
  Every model kind, every endpoint with a `model`.
- **A split model**: one chat model's layers on both machines, the first on one, the rest on the
  other, for a model that does not fit in one machine's memory. Dense Qwen-path models only
  (Qwen3, Qwen3.5 / 3.6 / 3.8, Llama 3), plain decoding only.

Everything here was measured on two DGX Sparks (GB10, 121 GB unified memory, CUDA 13, Ubuntu
24.04, aarch64) joined by their two 200 Gb/s ports with a direct cable, `10.100.80.1` and
`10.100.80.2` on the first port. Nothing else was tried: no other hardware, no more than two
machines, no switch in between.

## The link

Measured with `tools/cluster/linkbench.rs` (a TCP sender / echo, one thread per stream; the
numbers are the sender's) and perftest, both ends unprivileged, the interfaces as Ubuntu left
them (MTU 1500, adaptive interrupt coalescing):

```sh
rustc -O tools/cluster/linkbench.rs                        # on both machines
./linkbench server 0.0.0.0:47000                           # on 10.100.80.2
./linkbench tput 10.100.80.2:47000 8 5                     # 8 streams for 5 s
./linkbench rr 10.100.80.2:47000 65536 5000                # 64 KB there and back, 5000 times
ib_send_bw -d rocep1s0f1 -x 3 -s 65536 -D 4                # on 10.100.80.2, then with its address
ib_send_lat -d rocep1s0f1 -x 3 -s 4096 -n 20000 10.100.80.2
```

| | port 1 (`enp1s0f1np1`) | port 2 (`enP2p1s0f1np1`) |
|---|---|---|
| TCP, 1 stream | 10.6-17.0 Gb/s | 21.9 Gb/s |
| TCP, 2 / 4 / 8 streams | 22.6-39.0 / 52.9 / 98.9-107.6 Gb/s | 26.8 / 52.6 / 103.0 Gb/s |
| both ports, 4 streams each | 66.9 Gb/s | 54.5 Gb/s |
| request + reply of 4 KB, p50 / p99 | 80-170 us / 0.86-1.15 ms | 153 us / 0.87 ms |
| request + reply of 64 KB, p50 / p99 | 269-295 us / 1.1-1.3 ms | 276 us / 0.81 ms |
| request + reply of 1 MB, p50 / p99 | 0.99-1.09 ms / 2.7-2.9 ms | 0.96 ms / 2.4 ms |
| `ping` | 0.99 ms average | 1.28 ms average |

(Ranges: two runs half an hour apart. Loopback on the same machine, for scale: 120 Gb/s on one
stream, 18 us for 4 KB, 42 us for 64 KB.)

RDMA (RoCE v2, `rocep1s0f1`, GID index 3) works without privileges: `ib_send_bw` and
`ib_write_bw` 13.0 GB/s (104 Gb/s) on one queue pair at 64 KB and 1 MB messages; `ib_send_lat`
1.8 us at 64 bytes, 3.1 us at 4 KB, 9.1 us at 64 KB (99th percentile 12.8 us).

What follows from these numbers:

- A hop per token is nearly free. A decode step of a split 27B model sends one hidden state
  (20 KB) and reads the logits back (248,320 `f32`, 1 MB): a 1 MB round trip is 1 ms at the
  median, and the step itself takes 99 ms on this hardware. Measured: 2.0-2.5 ms more per
  token (below).
- A prompt chunk is cheap too: 2048 tokens of hidden states are 42 MB, 20-30 ms at the
  one-stream rates above, against 1.6-2.8 s of computation for the chunk.
- So plain TCP is enough, and oxinfer uses nothing else. RDMA would cut a hop from 0.2 ms to
  0.01 ms, which is worth having only for a design that crosses the link many times per token
  (routed experts on the other machine, say: tens of round trips per token and megabytes of
  activations; not built).
- The TCP round trips are slow for the link (0.1-0.3 ms where RDMA takes 0.01 ms). Busy
  polling takes the 4 KB one to 22-28 us (below): most of it is the wait for the interrupt
  and the wake-up of the reader, not the wire.

### Tuning the link (needs root)

Each change was applied on both machines, measured, and put back; the machines are as they were.

| | 1 stream | 2 streams | 8 streams | 4 KB p50 / p99 | 64 KB p50 / p99 | 1 MB p50 / p99 |
|---|---|---|---|---|---|---|
| as found (MTU 1500) | 17.0 Gb/s | 39.0 | 107.6 | 80 us / 1.15 ms | 295 us / 1.28 ms | 0.99 / 2.89 ms |
| MTU 9000 | 48.2 | 81.4 | 111.1 | 623 us / 1.31 ms | 276 us / 1.24 ms | 0.66 / 1.47 ms |
| MTU 9000, busy polling 50 us | 44.1 | 94.9 | 111.0 | 28 us / 51 us | 266 us / 0.95 ms | 0.86 / 2.00 ms |
| MTU 9000, busy polling, coalescing off | 55.4 | 93.5 | 110.2 | 22 us / 43 us | 174 us / 1.09 ms | 0.56 / 1.45 ms |
| MTU 1500, busy polling, coalescing off | 13.0 | 45.5 | 104.5 | 395 us / 1.69 ms | 174 us / 0.85 ms | 0.60 / 1.43 ms |

```sh
# on both machines, for both ports (runtime only: gone at the next boot)
sudo ip link set dev enp1s0f1np1 mtu 9000          # was 1500
sudo ip link set dev enP2p1s0f1np1 mtu 9000
ping -M do -s 8972 -c 3 10.100.80.2                # must pass before anything uses the link
sudo sysctl -w net.core.busy_read=50 net.core.busy_poll=50          # were 0
sudo ethtool -C enp1s0f1np1 adaptive-rx off adaptive-tx off rx-usecs 0 tx-usecs 0   # were on, on, 8, 8
```

MTU 9000 triples what one TCP stream carries (17 to 48-55 Gb/s), which is what copying weights
from one machine to the other and a split model's prompt chunks use; busy polling takes a
small request and its reply from 0.1-0.6 ms to 22-28 us. oxinfer was not measured with these
settings. What they could give it, from the figures here: a split model's decode step spends 2
of its 100 ms on the hop, most of it the megabyte of logits coming back, which MTU 9000 with
busy polling moved in 0.56 ms instead of 0.99; a prompt chunk's 20-30 ms of transfer would be
under 10 ms, of 1.6 s; forwarded requests would not change (their cost is the accept, below).
A percent or two: oxinfer needs none of it. To keep the settings: an `mtu: 9000` under each of
the two interfaces in the machine's netplan file, `net.core.busy_read = 50` and
`net.core.busy_poll = 50` in a file under `/etc/sysctl.d/`, and the `ethtool -C` line in a
networkd-dispatcher or systemd unit (ethtool settings have no netplan key). Busy polling spends
CPU time in every socket read of the machine, not only the link's.

## Peers: one endpoint, two machines

```sh
# machine B (10.100.80.2): an ordinary server, on the link's address only, with a key
oxinfer serve --listen 10.100.80.2:8091 --api-key "$KEY" --model q=/path/to/model.gguf

# machine A: the endpoint clients use
oxinfer serve --listen 127.0.0.1:8080 --api-key "$KEY" \
    --model q=/path/to/model.gguf --peer 10.100.80.2:8091
```

`--peer HOST:PORT[,key=KEY]` is repeatable (`"peers": ["HOST:PORT,key=KEY", ...]` or
`[{"addr": ..., "key": ...}]` in a `--config` file; `peer` in `ServerOptions::set`). `KEY` is
the peer's `--api-key`, by default the front's own.

**The link is not an open door.** A peer is used only if it refuses requests without its key:
the front checks, and a peer that answers without one is listed as not used, with the reason
(`GET /health`, the status page). A front with a peer and no key for it does not start. Bind
the peer to the link's address (`--listen 10.100.80.2:PORT`), not to `0.0.0.0`: its key then
only travels over the cable. There is no TLS between the machines (peers are reached over plain
HTTP; `https://` peers are refused): the key and the conversations cross the link in the clear,
which is fine for a cable between two machines and not for a network others are on.

What the front does:

- **`GET /v1/models`** lists its own models and the peers', each with `nodes`: where it is
  listed (`"local"`, the peers' addresses).
- **A model only a peer lists**: the request is forwarded as it came (any `POST` that names a
  `model`, in a JSON body or a form; tried with chat completions, token counts, image
  generation and embeddings, not with the form endpoints), and the response passed through as
  it arrives, server-sent events included. A cancel on the front
  (the status page, `Api::cancel`), or a client that goes away from a streamed response, closes
  the connection to the peer and tells it to stop the request (`POST /v1/cluster/cancel` with
  the name the front gave the request: a response that is not streamed is not written until it
  is done, so the peer would not notice the closed connection). The peer's errors come back as
  the peer worded them. The connection to the peer is kept for 2 s after a response and reused
  by the next request: a new one waits for the peer's listener to accept it, which looks for
  connections every 25 ms.
- **A model both list** goes to the machine that holds the conversation, else to the least
  loaded one: a machine where the model is loaded and has a free session (`--parallel`; one
  request at a time for the other kinds) first, then one where it is not loaded, then one where
  the request would wait; fewer requests in flight first; the front first when equal. The
  front counts its own requests as it sends them and reads each peer's other requests every
  2 s. `OXINFER_PEER_DEBUG=1` prints each choice.
- **Conversations stay where their cache is**: a chat request goes back to the machine its
  conversation last ran on, while that machine has a free session and no request of the same
  conversation is still running. A conversation is told by the `X-Session-Id` header when the
  client sends one (opencode does), else by how it starts (the system prompt and the first
  message, which every later request repeats). The front remembers the last 8192.
- **Models on request** (`auto_models`): a model no machine lists goes to the front when it
  fits in its free memory, else to the peer with the most free memory it fits in, else to a
  machine that would have to unload idle models. The size is the model's files on the front's
  disk; a model the front does not have is tried on the front first, and passed on when the
  front cannot take it (no memory, no disk, cannot fetch it).
- **Profiles** (`MODEL@PROFILE`, [learning.md](learning.md)): always the same machine, the
  first (the front, then the peers in order) that has the profile, else the first that lists
  the model. The learn directory is each machine's own: episodes are logged where the request
  ran, and `/v1/episodes`, `/v1/outcomes` and `/v1/profiles` are not forwarded, so outcomes are
  reported to, and adapters trained on, the machine that holds the profile.
- **A peer that is down** costs one warning line and no waiting: connections are given 2 s to
  open and a poll 3 s to be answered, the peer is asked again every 2 s and used again once it
  answers. Requests it was running end with an error (`502`, `peer_unavailable`; an error event
  on a stream) when nothing listens there any more, or after 30 s without an answer to the
  polls. A request for a model that is also on another machine runs there instead when the
  peer could not be reached at all.
- **Status**: `peers` in `GET /api/web/activity` (the status page's data) has each peer's
  state, the reason it is not used, its models, its free memory and the requests forwarded to
  it, and a running request has the `peer` it is on; the status page shows both (its Peers
  table was written for this, not looked at in a browser). `GET /v1/cluster` is what each
  server tells its fronts (its models with their requests in flight, its free memory, its own
  peers), and `GET /health` names the peers that are not used.

The tests of `tests/cluster.rs` run two servers of one process on loopback ports, with tiny
checkpoints on the CPU: the list of models, a chat model only the peer has (complete and
streamed replies, an error, a token count), a conversation that stays on its machine and one
that goes back to the peer it started on, two requests arriving together, a cancel on the
front and a client that leaves (streamed and not), a peer that is not there at start, comes,
and goes away again (with and without the model on the front), peers without a key or with
another one, two servers naming each other, a model on request with and without the memory
for it on each side, and a profile on the peer. Not covered by a test: the 30 s without an
answer after which a silent peer's requests are ended, and a model on request that the front
cannot fetch.

### Measured

`unsloth/Qwen3.8-27B-GGUF`'s `UD-Q4_K_M` (16.6 GB) loaded on both Sparks with `parallel=8`,
`N` chat requests at once, each its own conversation, 200 tokens each
(`tools/cluster/chat_bench.py tput URL KEY q N 200 LABEL`, run on the front's machine; tokens
generated by all of them over the time from the first request to the last reply):

| requests at once | one Spark (asked directly) | both, through the front | of them on the peer | |
|---|---|---|---|---|
| 1 | 10.9 tok/s | 10.3 tok/s | 0 | |
| 2 | 12.9 | 21.1 | 1 | 1.64x |
| 4 | 14.5 | 25.0 | 2 | 1.72x |
| 8 | 15.4 | 28.1 | 4 | 1.82x |
| 16 | 15.4 | 30.3 | 8 | 1.97x |

(One run each. The single request ran on the front's own machine, the other Spark: 10.3
against 10.9 tok/s is the two machines, not the front. At 16, half of the requests wait on one
machine and none on two.) Requests that repeat an earlier conversation's beginning go back to
its machine, as meant: a benchmark that sends the same prompts again measures that, not the
balancing (the first run of this table did, and put 2 of 2 and 3 of 4 on one machine).

The time to the first streamed token of a model only the peer has (Llama-3.2-1B `Q4_K_M`, a
one-line prompt, 40 requests one after the other, three rounds, from the front's machine;
`chat_bench.py ttft`):

| | median | mean | 90th percentile |
|---|---|---|---|
| the peer asked directly, over the link | 7.0-7.3 ms | 9.7-12.6 ms | 27-28 ms |
| through the front | 10.8-13.4 ms | 16.1-17.5 ms | 30-31 ms |
| directly, 3 s between requests (15 requests) | 17.4 ms | 16.3 ms | 29 ms |
| through the front, 3 s between requests | 33.8 ms | 30.3 ms | 41 ms |

So a forwarded request costs 4-6 ms at the median when the front has a connection to the peer
from the last 2 s, and about 15 ms when it has to open one: most of both figures, and of the
direct ones, is httpsd's listener, which accepts new connections every 25 ms (before the
connections were kept, every forwarded request paid it twice: 31 ms at the median against 7).

Limits:

- Video jobs (`/v1/videos`) are not forwarded (a job is polled by id on the machine that runs
  it), nor are the management API, the prompt-cache and status endpoints: those are each
  machine's own.
- Two servers may name each other as peers (a forwarded request is never forwarded again), but
  each then balances on its own.
- A peer whose every HTTP worker (`--workers`, default 32) is busy with a response that is not
  streamed has none left to answer the polls, and would look down for that long (by
  construction; not tried).
- The peer's prompt cache and sessions are the peer's: a conversation that moves to the other
  machine (its machine was full, or down) reads its prompt again there.

## A split model: one model, two machines

```sh
# machine B: the last layers (by default the second half), on the link's address
oxinfer split-worker /path/to/model.gguf --listen 10.100.80.2:8092 --key "$KEY"

# machine A: the first layers, and everything else (tokenizer, sampling, the API)
oxinfer serve --model big=/path/to/model.gguf,split=10.100.80.2:8092,split_key="$KEY"
```

Both machines hold the same model files (the worker checks the hidden size, the vocabulary, the
number of layers and the files' size). The worker runs layers `K..L`, the final norm and the LM
head (`--from K`, default half the layers); the front asks it for `K` and loads the embeddings
and layers `0..K`. The weights are mapped, so each side reads its own layers only. Each side
keeps the caches of its layers. `split_key` defaults to `$OXINFER_SPLIT_KEY`, as the worker's
`--key` does; a worker does not start without one. As for peers: plain TCP, the key and the
hidden states in the clear, for a cable and nothing wider.

Each pass, the front runs its layers and sends the residual stream after layer `K - 1` as raw
`f32` rows; the worker answers with the logits, or the final hidden states, that the caller
asked for. `f32`, not `f16`: nothing is rounded on the way, so the two halves compute exactly
what one machine does (see the checks below), and the payload is too small to matter (above).

- **Decoding** takes the two halves in turn: one stream is no faster than on one machine. Its
  worth is the memory: a model up to about twice one machine's.
- **Prompts are pipelined**: a prompt chunk nothing reads the hidden states of is not waited
  for, so the front starts chunk `n + 1` while the worker runs chunk `n`.
- **Concurrent requests** (`--parallel`): a decode step of several sessions sends each
  session's row as soon as the front has it, and the worker runs it while the front is on the
  next session.

Why a cut between layers and not something finer (estimates from the link's figures above, for
a model of 40 layers and hidden size 5120; none of this was built): keeping a
mixture-of-experts model's routed experts on the other machine crosses the link once per layer
instead of once per pass. Per decoded token that is 40 round trips of a hidden state, 8-12 ms
on TCP as found (1 ms with busy polling or over RDMA); per prompt chunk of 2048 tokens, 40
times 42 MB each way, 3.4 GB, which is 1.5-3 s on one TCP stream at MTU 1500 (0.3 s over RDMA
or eight streams). The memory it frees is the same as a cut between layers, which crosses
once, so the cut between layers is the one to extend to those models.

Not supported with `split` (refused at load, or off): MTP and DSpark drafts, prompt lookup, LoRA
adapters and profiles' adapters, interventions, weight streaming, the prompt cache and session
snapshots, mixture-of-experts layers, Qwen4-Exp, LFM3, MiniMax's lightning attention, Granite 3.
Images were not tried. If the worker stops, the requests in flight fail (the engine panics on
their threads with the worker's error) and the model has to be loaded again once the worker is
back: its half of every session is gone. A layer split of the automatic kind (when a model does
not fit) is not there: `split` is asked for.

### What was checked

The split model gives the bits of the whole one:

- On the CPU, in the tests of `src/split.rs` (a worker and a front in one process, over
  loopback), with tiny random Qwen3.5 (Gated DeltaNet layers on both sides, and attention on
  the worker alone), Qwen3 and Llama checkpoints: the logits after a prompt read in chunks,
  after decode steps, after a checkpoint and a cut back to it, from a copied session, from a
  batched step of two sessions, and the tokens of a three-request conversation through the
  engine with its cache reuse.
- On the two Sparks (CUDA on both), with `unsloth/Qwen3.8-27B-GGUF`'s `UD-Q4_K_M` (16.6 GB, 64
  layers, 32 on each machine): the logits after the prompt, after each of 64 greedy steps and
  after each of 64 steps of 4 sessions at once, 129 rows of 248,320, written to a file by the
  whole model on one machine and by the split one, and compared with `cmp`: identical, for a
  prompt of 8192 tokens and for one of 16384.

```sh
# on 10.100.80.2
oxinfer split-worker Qwen3.8-27B-UD-Q4_K_M.gguf --listen 10.100.80.2:8092 --key "$KEY"
# on 10.100.80.1
cargo run --release --example split_bench -- Qwen3.8-27B-UD-Q4_K_M.gguf --prompt 16384 --dump whole.f32
cargo run --release --example split_bench -- Qwen3.8-27B-UD-Q4_K_M.gguf --prompt 16384 --dump split.f32 \
    --split 10.100.80.2:8092 --key "$KEY"
cmp whole.f32 split.f32
```

| Qwen3.8-27B `UD-Q4_K_M`, CUDA | one Spark | split over two | |
|---|---|---|---|
| prompt of 8192 tokens (4 chunks of 2048) | 802 tok/s | 1178-1188 tok/s | 1.48x |
| prompt of 16384 tokens (8 chunks) | 738 tok/s | 1250-1256 tok/s | 1.70x |
| decoding, one stream | 98.7 / 107.1 ms per token | 100.7 / 109.6 ms | +2.0 / +2.5 ms |
| decoding, 4 sessions at once | 14.6 tok/s together | 16.6-16.7 tok/s | 1.14x |

(The two decode figures: after the 8192-token prompt and after the 16384-token one. The split
runs were made twice each; the whole-model runs once.) As estimated from the link: the prompt
gets `2n / (n + 1)` of one machine's speed with `n` chunks in the pipeline (1.6 and 1.78 here)
less the transfers, and a decode step pays one hop, 20 KB out and the 248,320 logits (1 MB)
back, which is the 1 MB round trip measured above twice over. Sending the logits as `f16`, or
sampling on the worker, would take most of those 2 ms back; neither is done. The whole model's
4 sessions run as one batched step on one GPU, the split model's as 4 steps that overlap across
the two machines.

Behind the server (`--model big=...,split=10.100.80.2:8092,split_key=...,parallel=8`, the same
model and worker), a question at temperature 0 got the same reply from the split model as from
the whole one on the front's machine, and `N` chat requests at once of 200 tokens each gave
10.6 tok/s for 1, 13.7 for 2, 15.9 for 4 and 17.8 for 8 together; the whole model on the same
server with the same plain decoding (`draft=none,lookup=0`) 10.6 for 1 and 14.0 for 4 (one run
each).

**A model larger than one machine was not run.** The nearest one the Qwen path loads is
Llama-3.3-70B at `f16` (`bartowski/Llama-3.3-70B-Instruct-GGUF`, four files, 141 GB); the two
Sparks downloaded at 6.1 and 2.7 MB/s at the same time (one uplink), which makes it four and a
half hours, and it was not started. What the 27B model forced to split does not show: the load
time and the page-cache behaviour when each half is most of a machine's memory.
