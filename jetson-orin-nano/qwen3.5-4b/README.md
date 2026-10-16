# llama.cpp deployment for Qwen/Qwen3.5-4B (NVIDIA Jetson Orin Nano, 8GB)

`start.sh` runs [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)
(GGUF Q4_K_M) under llama.cpp's `llama-server` with full CUDA offload and
MTP speculative decoding from the model's own MTP layer, text only by
default, on a Jetson Orin Nano running **JetPack 7 / L4T R39** (CUDA 13).
It serves an OpenAI-compatible `/v1` API.

Alternative to `../qwen3-4b-instruct-2507`,
`../gemma4-e2b` and `../lfm2.5-8b-a1b`.
All share the same `Dockerfile`, image tag
(`nixclaw/llama-server-jetson-orin:local`) and port 8000, so run only one at
a time; `start.sh` refuses to start if port 8000 is already serving. See the
Qwen3-4B deploy's README for why llama.cpp rather than Ollama, vLLM or
NVFP4 on this board.

## Why this model

- **Coding and tool use**: LiveCodeBench v6 55.8, τ²-Bench 79.9, BFCL v4
  50.3; clearly ahead of Gemma 4 E2B and Qwen3-4B-Instruct-2507.
- **MTP**: one built-in MTP layer, supported by llama.cpp's
  `--spec-type draft-mtp` since May 2026 (PR #22673). No separate drafter file.
- **Small KV cache**: only 8 of 32 layers use full attention (4 KV heads x
  256); the other 24 are Gated DeltaNet with a fixed-size state. ~19KB/token
  at q8_0 vs ~76KB for Qwen3-4B, so ~3x the context in less memory.
- **Thinks by default**: better answers, but on this board's decode speed a
  long thought adds tens of seconds per turn. `THINKING=0` turns it off.
- **Vision**: the model reads images, but the encoder is a separate file
  (`mmproj-F16.gguf`, 0.67GB) and off by default. It only runs for requests
  with images, so it doesn't affect text speed; off leaves the memory for
  context.

## Files

Downloaded once by `start.sh` into `./.models/` (resumable):

| File | Repo | Size | When |
|---|---|---|---|
| `Qwen3.5-4B-Q4_K_M.gguf` | `unsloth/Qwen3.5-4B-MTP-GGUF` | 2.84GB | always |
| `mmproj-F16.gguf` | same | 0.67GB | `ENABLE_MMPROJ=1` |

Use the `-MTP-GGUF` repo: its files contain the MTP layer
(`blk.32.nextn.*`, `nextn_predict_layers=1`). The files in unsloth's plain
`Qwen3.5-4B-GGUF` repo don't (checked in both GGUF headers), so MTP
wouldn't work with them.

## Sizing (8GB unified RAM/VRAM, ~6.8GB available)

Estimates, not measured on the board:

| | |
|---|---|
| Weights (Q4_K_M incl. MTP layer) | 2.84GB |
| KV cache, 98304 tokens @ q8_0 (~19KB/token incl. MTP layer) | ~1.9GB |
| DeltaNet state | ~50MB per slot |
| CUDA context + compute buffers | ~0.5-1GB |

Reference point: Qwen3-4B (2.5GB weights + 1.3GB KV) left 2.2GB free. This
default uses ~0.9GB more, leaving ~1.3GB. `CTX_SIZE=131072` (~2.5GB KV)
probably fits too, with less headroom; the model supports 262144. With
`ENABLE_MMPROJ=1`, subtract 0.67GB. If it OOMs, see the LFM2.5 README's
"Sizing" (`UBATCH_SIZE=128`, swapfile, headless boot).

## MTP

Decode is memory-bandwidth bound (68 GB/s, 102 GB/s in MAXN SUPER mode). A
target step reads ~2.7GB. One draft token runs the MTP layer and the tied
248K-vocab lm_head (~0.5GB), ~20-25% of a target step; verifying the drafts
costs about one step.

With per-position acceptance 85/59/49% (a Qwen3.5 report, see the Spark
deploy's README):

| Draft tokens | Tokens per step | Step cost | Speedup |
|---|---|---|---|
| 1 | 1.85 | ~1.23 | ~1.5x |
| 2 | 2.35 | ~1.46 | ~1.6x |
| 3 | 2.59 | ~1.69 | ~1.5x |

With higher acceptance (code, JSON) 3 can win. Default 2; measure 0-3. The
Orin GPU is small, so verifying several tokens may cost more than one step;
that is another reason to measure. Unsloth's own example uses 6 draft
tokens, but on faster GPUs.

The image is shared with the other Jetson deploys. If it predates the MTP
support, `start.sh` stops and asks for `REBUILD=1` (or
`NUM_SPECULATIVE_TOKENS=0`). MTP together with image input needs a
llama.cpp from 2026-10-10 or later (PR #30257).

## Usage

```bash
cd deploy/jetson-orin-nano/qwen3.5-4b
../qwen3-4b-instruct-2507/stop.sh   # or whichever Jetson deploy is running
./start.sh                                   # text only, MTP 2 draft tokens, thinking on
THINKING=0 ./start.sh                        # answer directly
NUM_SPECULATIVE_TOKENS=0 ./start.sh          # MTP off
ENABLE_MMPROJ=1 ./start.sh                   # with image input
REBUILD=1 ./start.sh                         # rebuild the shared image (newer llama.cpp)
./stop.sh
```

Measure decode speed and MTP acceptance per depth (acceptance comes from
llama-server's per-request `timings.draft_n` / `draft_n_accepted`):

```bash
for k in 0 1 2 3; do
  ./stop.sh; NUM_SPECULATIVE_TOKENS=$k ./start.sh
  python3 ../../dgx-spark/vllm-qwen3.5-4b/bench.py http://<jetson>:8000 qwen3.5-4b 1000,8000 code
done
```

Smoke tests:

```bash
# plain chat (German), no thinking for this request
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-4b","chat_template_kwargs":{"enable_thinking":false},
       "messages":[{"role":"user","content":"Sag in einem Satz hallo."}]}'

# tool call: expect choices[0].message.tool_calls, not text
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-4b",
       "messages":[{"role":"user","content":"Wie ist das Wetter in Berlin?"}],
       "tools":[{"type":"function","function":{"name":"get_weather",
         "description":"Get current weather for a city",
         "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'
```

Thinking comes back in `reasoning_content`, separate from `content`.

## Wiring into NixClaw

`config/agent.yaml`:

```yaml
backend: "vllm"            # llama-server speaks the same OpenAI /v1 API
vllm:
  host: "http://<jetson-ip>:8000"
  model: "qwen3.5-4b"
  options:
    context_window: 98304        # match CTX_SIZE / PARALLEL
    max_tokens: 8192             # thinking counts against this
    max_concurrent_requests: 1   # match PARALLEL
```

## Environment overrides

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_REPO` / `MODEL_FILE` | `unsloth/Qwen3.5-4B-MTP-GGUF` / `Qwen3.5-4B-Q4_K_M.gguf` | e.g. `Qwen3.5-4B-Q4_0.gguf` (2.67GB, slightly faster) |
| `ENABLE_MMPROJ` | `0` | `1` = image input, +0.67GB |
| `MMPROJ_REPO` / `MMPROJ_FILE` | `MODEL_REPO` / `mmproj-F16.gguf` | |
| `NUM_SPECULATIVE_TOKENS` | `2` | MTP draft tokens per step, `0` = off |
| `THINKING` | `1` | `0` = `--reasoning off`; requests can override via `chat_template_kwargs.enable_thinking` |
| `CTX_SIZE` | `98304` | total across slots; model maximum 262144 |
| `CACHE_TYPE` | `q8_0` | KV cache type (K and V) |
| `PARALLEL` | `1` | slots; each gets `CTX_SIZE / PARALLEL` |
| `BATCH_SIZE` / `UBATCH_SIZE` | `1024` / `256` | lower = smaller prefill memory spike |
| `TEMP` / `TOP_P` / `TOP_K` / `PRESENCE_PENALTY` | thinking `1.0`/`0.95`/`20`/`1.5`, else `0.7`/`0.8`/`20`/`1.5` | model card values; requests override |
| `SERVED_MODEL_NAME` | `qwen3.5-4b` | |
| `IMAGE` | `nixclaw/llama-server-jetson-orin:local` | shared with the other Jetson deploys |
| `LLAMA_CPP_REF` | `master` | llama.cpp git ref, resolved to a commit when building |
| `BUILD_JOBS` | `3` | parallel nvcc jobs; higher OOMs the 8GB board |
| `REBUILD` | `0` | `1` rebuilds the image |
| `PORT` | `8000` | |
| `CONTAINER_NAME` | `llama-qwen3.5-4b-jetson` | |
| `HF_TOKEN` | unset | not needed (ungated repos) |

## Not verified yet

Nothing has run on the board: memory, decode speed, MTP acceptance and
speedup, speed of llama.cpp's Gated DeltaNet CUDA kernels on sm_87, tool
calling through the GGUF's chat template with thinking on.
