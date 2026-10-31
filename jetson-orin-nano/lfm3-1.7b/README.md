# llama.cpp deployment for LiquidAI/LFM3-1.7B (NVIDIA Jetson Orin Nano, 8GB)

`start.sh` runs `LiquidAI/LFM3-1.7B` (GGUF **Q4_K_M**) under llama.cpp's
`llama-server` with full CUDA offload on a Jetson Orin Nano running
**JetPack 7 / L4T R39** (CUDA 13). It serves an OpenAI-compatible `/v1` API.

This is one of the one-model-at-a-time options on this board
(`../qwen3-4b-instruct-2507`, `../qwen3.5-4b`, `../gemma4-e2b`,
`../lfm2.5-8b-a1b`). The deploys share:

- the same `Dockerfile` and image tag (`nixclaw/llama-server-jetson-orin:local`),
  so the image is only built once. Keep the Dockerfiles identical, or set a
  different `IMAGE` here. **This model needs llama.cpp >= v0.8.0** (the
  `lfm3` architecture); if the shared image is older, `REBUILD=1 ./start.sh`
  (the rebuild affects the sibling deploys too).
- the same port (8000). Only run one at a time; `start.sh` refuses to start
  if port 8000 is already serving.

## Why llama.cpp (and not NVFP4 / vLLM / SGLang / TensorRT-LLM)

This deploy started as a hunt for an **NVFP4** recipe for this board. Short
answer: not possible today, and even at best it would gain nothing here.

1. **The NVFP4 checkpoint runs nowhere today.** `LiquidAI/LFM3-1.7B-NVFP4`
   exists (TensorRT-Model-Optimizer NVFP4, group size 16), but NVFP4
   safetensors are only consumable by vLLM / SGLang / TensorRT-LLM, and none
   of them implement the `Lfm3ForCausalLM` architecture (checked against the
   vLLM model registry, which only knows the `Lfm2*` family; likewise SGLang
   and TRT-LLM). Any of them fails at model registration, before quant
   support is even consulted.
2. **The Orin has no FP4 hardware.** The Orin GPU is Ampere (`sm_87`). FP4
   tensor cores exist from Ada (`sm_89`) and Blackwell only — on Orin, FP4
   weights would at best be unpacked to 16-bit in software, i.e. **no speedup
   over INT4** (same conclusion as
   [`../qwen3-4b-instruct-2507/README.md`](../qwen3-4b-instruct-2507/README.md)).
   The fast path on Ampere is 4-bit weights with fp16 compute, which is
   exactly llama.cpp's `Q4_K_M` CUDA kernels.
3. **vLLM/SGLang don't fit the 8GB budget anyway** (PyTorch + CUDA-graph
   memory), per the Qwen deploy's rationale.

Re-check the NVFP4 verdict on any board/runtime with (expect the arch error,
not a quant error):

```bash
docker run --rm --runtime nvidia -p 8000:8000 \
  -e HF_HOME=/tmp/hf vllm/vllm-openai:latest \
  --model LiquidAI/LFM3-1.7B-NVFP4 --max-model-len 4096 --enforce-eager
# observed: fails in vLLM's model registry (model architecture
# "Lfm3ForCausalLM" not supported) -- quantization never reached.
```

Alternatives compared as the 4-bit NVFP4 replacement (official
`LiquidAI/LFM3-1.7B-GGUF` unless noted):

| Quant | Size on disk | Notes |
|-------|--------------|-------|
| **Q4_K_M** (default) | 1.05GB (1.02GiB) | measured on the llmtest suite: coding 0.9875, agent/toolcalling 1.0, 100% native tool calls |
| Q4_K_S | 1.05GB | ~same size, slightly leaner quality |
| QAD-Q4_0 | 1.05GB | Liquid's quantization-aware-distilled Q4_0; size of Q4_0 with quality closer to K-quants; llama.cpp support needs a v0.8.0+ build like everything here |
| Q5_K_M | 1.26GB | quality first if RAM allows (plenty on this board) |
| Q8_0 | 1.85GB | reference; llmtest coding 0.9375 (i.e. Q4_K_M showed no measurable loss vs Q8 on our suite) |
| IQ4_XS (bartowski/LFM3-1.7B-GGUF) | 1.0GB | unofficial repack; smallest 4-bit, unbenchmarked by us |

## Model

- **Checkpoint**: [`LiquidAI/LFM3-1.7B`](https://huggingface.co/LiquidAI/LFM3-1.7B)
  (`Lfm3ForCausalLM`), hybrid-conv: of 24 layers, 13 are `conv` (fixed-size
  state), 5 `sliding_attention` (256-token window), only **6**
  `full_attention` (4 KV heads x 128). Dense, 1.7B, reasoning model (emits
  `reasoning_content` beside `content` with `--jinja`). Context max 262144.
- **GGUF**: `LiquidAI/LFM3-1.7B-GGUF`, file `LFM3-1.7B-Q4_K_M.gguf`
  (1.05GB). Downloaded once by `start.sh` into `./.models/` (resumable).
  Override with `MODEL_REPO` / `MODEL_FILE` (see the table above).
- Served as `lfm3-1.7b` (`SERVED_MODEL_NAME`).
- Sampling defaults follow the model card (`temperature 0.1`, `top_k 50`,
  `repetition_penalty 1.1`; math/science: temperature 1.0 / top_p 0.95 /
  top_k 20 per request). Server defaults only; nixclaw's own params win.
- **Tool calling** works natively with `--jinja` (100% native tool-call rate
  across the agent + toolcalling tracks in `llmtest/results/lfm3-1.7b-q4/`).
- German among the officially supported languages.
- License: LFM Open License 1.0 (check it for commercial use).

## Sizing (8GB unified RAM/VRAM)

| Part | Approx. |
|------|---------|
| Weights (Q4_K_M) | 1.02GiB |
| KV cache, 65536 tokens @ q8_0 (~6.5KB/token x 6 full-attn layers) | ~0.43GB |
| Sliding-window KV (5 layers x 256 tokens) + conv state | flat, few MB |
| CUDA context + compute buffers | ~0.5-1GB |

~3GB total - by far the most headroom of any model option on this board
(lfm2.5-8b-a1b: ~6GB, gemma4-e2b: full). Options:

- **OOM under load** (the usual llama.cpp failure on this board - prompt-
  processing compute-buffer spike, not steady state): lower
  `BATCH_SIZE`/`UBATCH_SIZE` (e.g. 512/128). See `../lfm2.5-8b-a1b/README.md`
  for the full diagnosis; defaults here are 1024/256.
- **No swap configured**: stock JetPack images often have none, turning any
  transient spike into a hard OOM-kill. Add a swapfile (see
  `../lfm2.5-8b-a1b/README.md`).
- **More context**: `CTX_SIZE=131072` costs just ~0.87GB (q8_0) thanks to
  the conv/sliding hybrid; the model allows up to 262144.
- `PARALLEL=2` splits `CTX_SIZE` into two slots at ~0.22GB KV each. Set
  `max_concurrent_requests: 2` in nixclaw to match.

## Usage

```bash
cd deploy/jetson-orin-nano/lfm3-1.7b
../<current-deploy>/stop.sh   # whichever sibling is running
./start.sh   # builds shared image if missing (20-60 min), downloads GGUF, waits for /health
./stop.sh
```

Smoke tests:

```bash
# plain chat (German). LFM3 is a reasoning model: expect `content` to hold
# the answer; `reasoning_content` (if present) is scratchpad, not user text.
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm3-1.7b","messages":[{"role":"user","content":"Sag in einem Satz hallo."}]}'

# tool call: expect choices[0].message.tool_calls, not text
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm3-1.7b",
       "messages":[{"role":"user","content":"Wie ist das Wetter in Berlin?"}],
       "tools":[{"type":"function","function":{"name":"get_weather",
         "description":"Get current weather for a city",
         "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'
```

If the second request returns tool-call markup as plain `content` instead of
`tool_calls`, the llama.cpp build is too old to parse the LFM3 chat template
-> `REBUILD=1 ./start.sh` (and re-download the GGUF if pinned files are
older than the tokenizer fix Liquid shipped shortly after release).

## Wiring into NixClaw

`config/agent.yaml`:

```yaml
backend: "vllm"            # llama-server speaks the same OpenAI /v1 API
vllm:
  host: "http://<jetson-ip>:8000"
  model: "lfm3-1.7b"
  options:
    context_window: 65536        # match CTX_SIZE / PARALLEL
    max_tokens: 4096
    max_concurrent_requests: 1   # match PARALLEL
```

## Environment overrides

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_REPO` | `LiquidAI/LFM3-1.7B-GGUF` | HF repo holding the GGUF |
| `MODEL_FILE` | `LFM3-1.7B-Q4_K_M.gguf` | see quant table; `LFM3-1.7B-QAD-Q4_0.gguf` / `Q5_K_M` are the sane swaps |
| `SERVED_MODEL_NAME` | `lfm3-1.7b` | `--alias`, the `model` name clients send |
| `IMAGE` | `nixclaw/llama-server-jetson-orin:local` | shared with the sibling deploys |
| `CUDA_VERSION` | `13.0.1` | `nvidia/cuda` base tag; must be <= the host's CUDA version |
| `LLAMA_CPP_REF` | `master` | must include lfm3 support (v0.8.0+); built commit is printed at start |
| `BUILD_JOBS` | `3` | Parallel nvcc jobs; higher OOMs the 8GB board |
| `REBUILD` | `0` | `1` rebuilds the shared image (also affects siblings) |
| `CONTAINER_NAME` | `llama-lfm3-1.7b-jetson` | |
| `PORT` | `8000` | shared across deploys, one at a time |
| `CTX_SIZE` | `65536` | Total KV context (shared across slots), max 262144 |
| `CACHE_TYPE` | `q8_0` | KV cache type (`f16`, `q8_0`, `q4_0`) |
| `PARALLEL` | `1` | Server slots |
| `BATCH_SIZE` | `1024` | Sizes prompt-processing compute buffers (see Sizing) |
| `UBATCH_SIZE` | `256` | Lower = smaller prefill peak, slower prompt processing |
| `TEMP` | `0.1` | Server default temperature (model card) |
| `TOP_K` | `50` | Server default top_k (model card) |
| `REPEAT_PENALTY` | `1.1` | Server default repetition penalty (model card) |
| `GPU_DOCKER_ARG` | `--runtime nvidia` | `--gpus all` fails in the runtime's csv mode |
| `HF_TOKEN` | unset | Not needed (repo is not gated) |

## Verification status

- **Measured ON the Orin Nano** (JetPack 7, built via `REBUILD=1 LLAMA_CPP_REF=v0.8.0
  ./start.sh`, board reachable at `jetson.lan:8000`, bench from the LAN host via
  `/v1/chat/completions`; generation t/s from server-side `timings`, TTFT
  client-side wall): **generation 34.6 tok/s** (3x 256-token runs,
  finish=length, includes reasoning tokens), **prefill ~1280 tok/s cold**
  (~925-token random prompts, no cache reuse), **TTFT ~740-760 ms** at 925
  cold prompt tokens, ~100 ms first-token on a short prompt, **~240 ms when
  the prefix cache hits** (llama.cpp auto-caches ~2/3 of a repeated prompt -
  this is what nixclaw's steady system prompt will ride on).
- **Measured locally on x86** (this repo's `llmtest` harness, same GGUF):
  coding **0.9875**/1.0 (Q4_K_M), agent **1.0**, toolcalling **1.0** with
  100% native tool calls. Q8_0 control: 0.9375 (no quant-related regression).
- Note: the shared image tag was rebuilt past llama.cpp v0.8.0 for lfm3;
  sibling deploys pick that up on their next start (re-smoke them once).

## Troubleshooting

Shared issues (CUDA visibility, csv-mode runtime, build OOM): see
`../qwen3-4b-instruct-2507/README.md`. Specific to this model:

- **`unknown model architecture: 'lfm3'`**: the shared image's llama.cpp
  predates v0.8.0. `REBUILD=1 ./start.sh`.
- **Tool calls come back as text**: see smoke test note (old build or old
  GGUF/tokenizer).
- **Long answers start empty / only thinking**: normal mid-generation
  (reasoning model); `content` fills in when the answer phase starts. If a
  caller only reads `content` and cuts off early, raise its `max_tokens`
  budget to cover the thinking phase (nixclaw `max_tokens: 4096` is generous;
  the llmtest runs used 3000 without issue).
- **Prefill slower than the dense Qwen options despite fewer FLOPs**:
  expected for hybrid conv/recurrent models - llama.cpp's prompt-cache reuse
  is limited there. Compare `prompt eval time` in `.llama.log`.
