# llama.cpp deployment for LiquidAI/LFM2.5-8B-A1B (NVIDIA Jetson Orin Nano, 8GB)

`start.sh` runs `LiquidAI/LFM2.5-8B-A1B` (GGUF Q4_K_M) under llama.cpp's
`llama-server` with full CUDA offload on a Jetson Orin Nano running
**JetPack 7 / L4T R39** (CUDA 13). It serves an OpenAI-compatible `/v1` API.

This is an alternative to `../qwen3-4b-instruct-2507`, not
a replacement. The two deploys share:

- the same `Dockerfile` and image tag (`nixclaw/llama-server-jetson-orin:local`),
  so the image is only built once. Keep the two Dockerfiles identical, or set
  a different `IMAGE` here.
- the same port (8000). Only run one at a time; the 8GB board can't hold both
  models anyway. `start.sh` refuses to start if port 8000 is already serving.

See the Qwen deploy's README for why llama.cpp rather than Ollama, vLLM or
NVFP4 on this board.

## Why this model

- **Long context for little memory.** `Lfm2MoeForCausalLM` has 24 layers,
  and only 6 of them are attention layers (8 KV heads × 64 dims). The other
  18 are short causal-conv blocks with a small fixed-size state. The KV cache
  is about 12KB/token at f16, compared with ~144KB/token for Qwen3-4B. That
  makes a 64K context cheaper than Qwen's 16K.
- **Fast decode.** It's a MoE with 8B total and ~1B active parameters per
  token (32 experts, top-4). Decode speed on the Orin is bound by memory
  bandwidth, so reading ~1B parameters per token should be faster than a
  dense 4B model. All 8B weights still have to fit in memory.
- **Tool calling** is a stated design goal (on-device assistant, chained
  tool calls). German is an officially supported language.
- License: LFM Open License (check it for commercial use).

## Model

- **GGUF**: `LiquidAI/LFM2.5-8B-A1B-GGUF`, file `LFM2.5-8B-A1B-Q4_K_M.gguf`
  (5.16GB, about 4.8GiB). Downloaded once by `start.sh` into `./.models/`
  (resumable). Alternatives in the same repo: `Q4_0` (4.84GB) if memory is
  tight, `Q5_K_M` (6.03GB, probably too big for this board).
- Served as `lfm2.5-8b-a1b` (`SERVED_MODEL_NAME`).
- Sampling defaults follow the model card (`temperature 0.2`,
  `repetition_penalty 1.05`). They're set as llama-server defaults; if nixclaw
  sends its own temperature, that wins.

## Sizing (8GB unified RAM/VRAM)

| Part | Approx. |
|------|---------|
| Weights (Q4_K_M) | 4.8GiB |
| KV cache, 65536 tokens @ q8_0 (~6.5KB/token) | 0.4GB |
| Conv state (per slot, fixed) | negligible |
| CUDA context + compute buffers | ~0.5-1GB |

That's about 6GB total. The weights are larger than Qwen's (2.4GB), so there
is less headroom for the OS than the small KV cache suggests. Run headless
(`sudo systemctl set-default multi-user.target`) and watch `tegrastats` /
`free -h` on first start. Options:

- **OOM under load (not at startup)**: this is the common case, and it's not
  the steady-state weights+KV total above - llama.cpp sizes its prompt-
  processing compute buffers off `--batch-size`/`--ubatch-size` at load time,
  and that reservation spikes well past steady state during a real prefill
  (nixclaw's system prompt + history). `BATCH_SIZE=1024`/`UBATCH_SIZE=256`
  (defaults since the OOM was diagnosed - previously unset, i.e. llama.cpp's
  own defaults of 2048/512) trade slower prompt processing for a smaller
  peak. Lower further (e.g. `UBATCH_SIZE=128`) if it still OOMs.
- **OOM at load (fails before serving)**: use
  `MODEL_FILE=LFM2.5-8B-A1B-Q4_0.gguf`, or lower `CTX_SIZE`.
- **No swap configured**: check `swapon --show` on the Jetson. A stock
  JetPack image often has none, so any transient spike (this model's or
  otherwise) is a hard OOM-kill instead of a slowdown. Don't repartition for
  this - add a swapfile on the existing root filesystem instead:
  ```bash
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  # low swappiness: this is a safety net for transient spikes, not routine
  # paging (CUDA allocations aren't swappable anyway; only CPU-side/OS memory is)
  sudo sysctl vm.swappiness=10
  echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
  ```
- **More context**: `CTX_SIZE=128000` (the model's max) is ~0.8GB at q8_0.
- `PARALLEL=2` splits `CTX_SIZE` into two 32K slots, which still covers
  nixclaw's prompts. Set `max_concurrent_requests: 2` to match.

## Usage

```bash
cd deploy/jetson-orin-nano/lfm2.5-8b-a1b
../qwen3-4b-instruct-2507/stop.sh   # if Qwen is running
./start.sh      # builds image if missing (20-60 min), downloads GGUF, waits for /health
./stop.sh
```

Smoke tests:

```bash
# plain chat (German)
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm2.5-8b-a1b","messages":[{"role":"user","content":"Sag in einem Satz hallo."}]}'

# tool call: expect choices[0].message.tool_calls, not text
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm2.5-8b-a1b",
       "messages":[{"role":"user","content":"Wie ist das Wetter in Berlin?"}],
       "tools":[{"type":"function","function":{"name":"get_weather",
         "description":"Get current weather for a city",
         "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'
```

If the second request returns `<|tool_call_start|>...` as plain `content`
instead of `tool_calls`, llama.cpp didn't parse the LFM2 tool-call format.
Rebuild with a newer llama.cpp (`REBUILD=1 ./start.sh`). Liquid updated the
tokenizer shortly after release to fix tool calling in llama.cpp, so an older
build or GGUF can show this.

## Wiring into NixClaw

`config/agent.yaml`:

```yaml
backend: "vllm"            # llama-server speaks the same OpenAI /v1 API
vllm:
  host: "http://<jetson-ip>:8000"
  model: "lfm2.5-8b-a1b"
  options:
    context_window: 65536        # match CTX_SIZE / PARALLEL
    max_tokens: 4096
    max_concurrent_requests: 1   # match PARALLEL
```

## Environment overrides

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_REPO` | `LiquidAI/LFM2.5-8B-A1B-GGUF` | HF repo holding the GGUF |
| `MODEL_FILE` | `LFM2.5-8B-A1B-Q4_K_M.gguf` | `LFM2.5-8B-A1B-Q4_0.gguf` if memory is tight |
| `SERVED_MODEL_NAME` | `lfm2.5-8b-a1b` | `--alias`, the `model` name clients send |
| `IMAGE` | `nixclaw/llama-server-jetson-orin:local` | Shared with the Qwen deploy |
| `CUDA_VERSION` | `13.0.1` | `nvidia/cuda` base tag; must be <= the host's CUDA version |
| `LLAMA_CPP_REF` | `master` | llama.cpp git ref; the built commit is printed at start |
| `BUILD_JOBS` | `3` | Parallel nvcc jobs; higher OOMs the 8GB board |
| `REBUILD` | `0` | `1` rebuilds the image (also affects the Qwen deploy) |
| `CONTAINER_NAME` | `llama-lfm2.5-8b-a1b-jetson` | |
| `PORT` | `8000` | |
| `CTX_SIZE` | `65536` | Total KV context (shared across slots), max 128000 |
| `CACHE_TYPE` | `q8_0` | KV cache type (`f16`, `q8_0`, `q4_0`) |
| `PARALLEL` | `1` | Server slots |
| `BATCH_SIZE` | `1024` | Logical batch size; sizes prompt-processing compute buffers (see Troubleshooting) |
| `UBATCH_SIZE` | `256` | Physical micro-batch size; lower = smaller peak memory during prefill, slower prompt processing |
| `TEMP` | `0.2` | Server default temperature |
| `REPEAT_PENALTY` | `1.05` | Server default repetition penalty |
| `GPU_DOCKER_ARG` | `--runtime nvidia` | `--gpus all` fails in the runtime's csv mode (see Troubleshooting) |
| `HF_TOKEN` | unset | Not needed (repo is not gated) |

## Troubleshooting

The build and GPU-visibility issues are the same as for the Qwen deploy; see
its README. Specific to this model:

- **`unknown model architecture: 'lfm2moe'`**: the image's llama.cpp is too
  old. Run `REBUILD=1 ./start.sh`.
- **OOM / container killed**: see Sizing - and note load-time OOM vs.
  under-load OOM have different fixes (batch/ubatch size vs. quant/context).
- **Tool calls come back as text**: see the smoke test above.
- **Every request reprocesses the whole prompt**: llama.cpp's prompt cache
  reuse is limited for hybrid (conv/recurrent) models. Prefill of a long
  nixclaw prompt may be slower than with Qwen even though decode is faster.
  Compare `prompt eval time` in `.llama.log`.
