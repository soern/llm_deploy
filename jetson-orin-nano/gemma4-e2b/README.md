# llama.cpp deployment for google/gemma-4-E2B-it (NVIDIA Jetson Orin Nano, 8GB)

`start.sh` runs Google's QAT `q4_0` GGUF of
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)
under llama.cpp's `llama-server` with full CUDA offload, MTP speculative
decoding and (optionally) image + audio input, on a Jetson Orin Nano running
**JetPack 7 / L4T R39** (CUDA 13). It serves an OpenAI-compatible `/v1` API.

Third alternative next to `../qwen3-4b-instruct-2507` and
`../lfm2.5-8b-a1b`. All three share the same `Dockerfile`
and image tag (`nixclaw/llama-server-jetson-orin:local`) and port 8000, so
run only one at a time; `start.sh` refuses to start if port 8000 is already
serving. See the Qwen deploy's README for why llama.cpp rather than Ollama,
vLLM or NVFP4 on this board.

## Why this model

- **Fastest of the three (estimate)**: 5.1B parameters, but 2.35B of them
  are the per-layer embedding (PLE) table, which is only a lookup. Decode
  reads ~1.3GB per token vs ~2.4GB for Qwen3-4B Q4_K_M, so roughly 30-50
  tok/s on the Orin's 68 GB/s (102 GB/s in Super mode) vs ~20-30. Not
  measured.
- **Image and audio input** (encoder `gemma-4-E2B-it-mmproj.gguf`, 0.99GB).
  Whether llama.cpp's multimodal support handles Gemma 4 audio (not just
  images) is not verified.
- **Weak at coding and agentic tool use**: LiveCodeBench v6 44 (Qwen3.5-4B:
  55.8), τ²-Bench lower than E4B's 42.2 (Qwen3.5-4B: 79.9). Use it as the
  small, fast, multimodal model, not as the coding agent.
- **QAT weights**: Google trained this checkpoint for 4-bit weights, so
  `q4_0` should be the closest 4-bit variant to BF16. NVFP4 makes no sense
  here: Ampere (sm_87) has no FP4/FP8 tensor cores, and the unsloth NVFP4
  checkpoint is a vLLM format, post-training quantized and 5.5GB.

## Files

All downloaded once by `start.sh` into `./.models/` (resumable):

| File | Repo | Size | When |
|---|---|---|---|
| `gemma-4-E2B_q4_0-it.gguf` | `google/gemma-4-E2B-it-qat-q4_0-gguf` | 3.35GB | always |
| `gemma-4-E2B-it-mmproj.gguf` | same | 0.99GB | `ENABLE_MMPROJ=1` (default) |
| `mtp-gemma-4-E2B-it-Q8_0.gguf` | `unsloth/gemma-4-E2B-it-GGUF` (`MTP/`) | 0.10GB | `NUM_SPECULATIVE_TOKENS` > 0 (default 3) |

## Sizing (8GB unified RAM/VRAM, ~6.8GB available)

Estimates, not measured on the board:

| | |
|---|---|
| Weights (QAT q4_0) | 3.35GB |
| Vision/audio encoder | 0.99GB (0 with `ENABLE_MMPROJ=0`) |
| MTP drafter | 0.10GB, shares the target's KV cache |
| KV cache, 131072 tokens @ q8_0 (~3.3KB/token: only 3 of 35 layers keep full-context KV) | ~0.4GB |
| CUDA context + compute buffers | ~0.5-1GB |

About 5.3-5.8GB with vision/audio, ~4.5GB text-only. The full 128K context
costs little here; the encoder is the big optional item. If it OOMs, the
LFM2.5 README's advice applies (prompt-processing compute buffers spike
during prefill; `UBATCH_SIZE=128`, a swapfile, headless mode):
`../lfm2.5-8b-a1b/README.md`, "Sizing". Then try
`ENABLE_MMPROJ=0` or a lower `CTX_SIZE`.

## MTP

Gemma 4 has no MTP layer inside the checkpoint; Google publishes a separate
drafter (`google/gemma-4-E2B-it-assistant`: 77M params, 4 layers, hidden
size 256) that reuses the target's KV cache. llama.cpp supports it since
June 2026 (PR #23398, E2B/E4B in PR #24282) via
`--spec-type draft-mtp --spec-draft-model <drafter.gguf> --spec-draft-n-max N`.
Only `llama-server` handles the E2B/E4B drafters; `llama-speculative` and
`llama-bench` reportedly fail with them.

**Older images**: the image is shared with the other Jetson deploys and
built from llama.cpp `master` at build time. If it predates the MTP support,
`start.sh` stops and asks for `REBUILD=1` (or `NUM_SPECULATIVE_TOKENS=0`).

**Drafter choice**: Google trained a second drafter for the QAT model
(`google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant`), which should match
this `q4_0` target better. Its only GGUF
(`msievers/gemma-4-E2B-it-qat-q4_0-assistant-GGUF`) is in ik_llama.cpp's
`gemma4_mtp` format; mainline llama.cpp expects `gemma4-assistant` (read
from both files' headers), so it won't load. The default is therefore
unsloth's GGUF of the standard drafter, trained against the BF16 model;
acceptance may be somewhat lower with the QAT target. Converting Google's
QAT drafter with llama.cpp's `convert_hf_to_gguf.py` and pointing
`DRAFT_REPO`/`DRAFT_FILE` (or a local file in `.models/`) at it is the
likely improvement.

**Depth**: default 3 draft tokens. Mainline llama.cpp computes the
drafter's logits over the full 262K vocabulary at hidden size 256 (~67M
weights; Google's centroid masking is loaded but not used), so one draft
token reads ~0.08GB vs ~1.3GB for a target step: cheap enough that a third
token can pay off if acceptance holds. vLLM's recipe uses 2 for E2B on
A100/H100, unsloth suggests starting at 2. The only E2B report found (a
Galaxy S26+ phone, 3 draft tokens) saw 48% acceptance and just 1-2 tok/s
gain; unsloth notes gains are smaller on low-bandwidth devices. Expect
anywhere from no gain to ~1.5x; measure.

## Usage

```bash
cd deploy/jetson-orin-nano/gemma4-e2b
../qwen3-4b-instruct-2507/stop.sh   # or the LFM2.5 deploy, if running
./start.sh                                   # vision/audio on, MTP 3 draft tokens
NUM_SPECULATIVE_TOKENS=0 ./start.sh          # MTP off
ENABLE_MMPROJ=0 ./start.sh                   # text only
REBUILD=1 ./start.sh                         # rebuild the shared image (newer llama.cpp)
./stop.sh
```

Measure decode speed and MTP acceptance per depth (acceptance comes from
llama-server's per-request `timings.draft_n` / `draft_n_accepted`):

```bash
for k in 0 1 2 3 4; do
  ./stop.sh; NUM_SPECULATIVE_TOKENS=$k ./start.sh
  python3 ../../dgx-spark/vllm-gemma4-e2b/bench.py http://<jetson>:8000 gemma-4-e2b 1000,8000 code
done
```

Smoke tests:

```bash
# plain chat (German)
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-e2b","messages":[{"role":"user","content":"Sag in einem Satz hallo."}]}'

# tool call: expect choices[0].message.tool_calls, not text
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-e2b",
       "messages":[{"role":"user","content":"Wie ist das Wetter in Berlin?"}],
       "tools":[{"type":"function","function":{"name":"get_weather",
         "description":"Get current weather for a city",
         "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'
```

Thinking is off unless the system prompt starts with `<|think|>` (Gemma 4
chat template).

## Wiring into NixClaw

`config/agent.yaml`:

```yaml
backend: "vllm"            # llama-server speaks the same OpenAI /v1 API
vllm:
  host: "http://<jetson-ip>:8000"
  model: "gemma-4-e2b"
  options:
    context_window: 131072       # match CTX_SIZE / PARALLEL
    max_tokens: 4096
    max_concurrent_requests: 1   # match PARALLEL
```

## Environment overrides

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_REPO` / `MODEL_FILE` | `google/gemma-4-E2B-it-qat-q4_0-gguf` / `gemma-4-E2B_q4_0-it.gguf` | e.g. `unsloth/gemma-4-E2B-it-GGUF` / `gemma-4-E2B-it-Q4_0.gguf` (3.04GB, post-training quantized) |
| `ENABLE_MMPROJ` | `1` | `0` = text only, saves ~1GB |
| `MMPROJ_REPO` / `MMPROJ_FILE` | `MODEL_REPO` / `gemma-4-E2B-it-mmproj.gguf` | |
| `NUM_SPECULATIVE_TOKENS` | `3` | MTP draft tokens per step, `0` = off |
| `DRAFT_REPO` / `DRAFT_FILE` | `unsloth/gemma-4-E2B-it-GGUF` / `MTP/mtp-gemma-4-E2B-it-Q8_0.gguf` | stored in `.models/` under the file's base name |
| `CTX_SIZE` | `131072` | model maximum; total across slots |
| `CACHE_TYPE` | `q8_0` | KV cache type (K and V) |
| `PARALLEL` | `1` | slots; each gets `CTX_SIZE / PARALLEL` |
| `BATCH_SIZE` / `UBATCH_SIZE` | `1024` / `256` | lower = smaller prefill memory spike |
| `TEMP` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `64` | model card values; requests override |
| `SERVED_MODEL_NAME` | `gemma-4-e2b` | |
| `IMAGE` | `nixclaw/llama-server-jetson-orin:local` | shared with the other Jetson deploys |
| `LLAMA_CPP_REF` | `master` | llama.cpp git ref; the built commit is printed at start |
| `BUILD_JOBS` | `3` | parallel nvcc jobs; higher OOMs the 8GB board |
| `REBUILD` | `0` | `1` rebuilds the image |
| `PORT` | `8000` | |
| `CONTAINER_NAME` | `llama-gemma4-e2b-jetson` | |
| `HF_TOKEN` | unset | not needed (ungated repos) |

## Not verified yet

Nothing has run on the board yet: build with current llama.cpp, memory,
decode speed, MTP acceptance and speedup, image and audio input, tool
calling through llama.cpp's Gemma 4 template.
