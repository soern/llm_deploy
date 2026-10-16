# vllm-dgx-spark

vLLM launch config for Qwen3.6-27B-NVFP4 **with DFlash speculative decoding**
on the DGX Spark (GB10) box that backs nixclaw's `vllm` LLM client
(`config/agent.yaml` -> `vllm.host`).

## Why this exists

The current deployment (`unsloth/Qwen3.6-27B-NVFP4`, plain `vllm serve`, no
speculative decoding) measured at **~5 tokens/sec** decode via the server's
own `/metrics` (`vllm:inter_token_latency_seconds`), a rate consistent with
running an *unquantized bf16* 27B model, not an NVFP4 one — the FP4 memory-bandwidth
win wasn't materializing. That's slow enough that nixclaw's client-side
30s stream-idle watchdog (`llm.py`) trips on ordinary tool-call-laden
replies, causing retries that hold the vLLM concurrency semaphore and,
under load, user-facing `"LLM temporarily unavailable"` errors.

This config mirrors a community recipe benchmarked on DGX Spark with the
same model, reporting **26.6-45.3 tok/s** decode depending on context length:
[MiaAI-Lab/Qwen3.6-27B-NVFP4-DFlash-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.6-27B-NVFP4-DFlash-DGX-Spark)
(`start.sh` verified byte-identical to the repo's prior name,
`MiaAI-Lab/Qwen3.6-27B-NVFP4-vLLM`, as of 2026-08-14).

## What's different from the current `docker run`, and why

| | Current | This config |
|---|---|---|
| Model | `unsloth/Qwen3.6-27B-NVFP4` | `nvidia/Qwen3.6-27B-NVFP4` |
| Speculative decoding | none | DFlash, `z-lab/Qwen3.6-27B-DFlash`, 10 draft tokens |
| `--quantization` | unset (auto-detect) | `compressed-tensors` (explicit) |
| `--attention-backend` | unset (default) | `flash_attn` (explicit) |
| `--moe-backend` | unset | `flashinfer_b12x` |
| `--kv-cache-dtype` | unset (default) | `bfloat16` (explicit) |
| `--max-num-seqs` / `--max-num-batched-tokens` | unset (defaults) | `4` / `8192` |
| `--gpu-memory-utilization` | unset (vLLM default, observed 0.92 live) | `0.84` |
| Container image | `nvcr.io/nvidia/vllm:26.06-py3` | `vllm/vllm-openai:nightly-aarch64` |
| Docker networking | `-p 8000:8000` | `--network host --ipc host --privileged` |
| Port / served name | `8000` / `qwen3.6` (auto) | `8000` / `qwen3.6` (unchanged — see below) |

**Model swap (unsloth -> nvidia checkpoint)**: the DFlash draft model
(`z-lab/Qwen3.6-27B-DFlash`) was published alongside NVIDIA's own
`nvidia/Qwen3.6-27B-NVFP4` checkpoint. Draft-model acceptance rates depend
on the draft closely matching the target model's output distribution, so
this config switches the base checkpoint to match rather than risk a
degraded speculative accept-rate against a differently-converted
(unsloth) checkpoint. This means a fresh ~15GB+ download on first start
(see Prerequisites).

**Port and served-model-name kept as-is on purpose**: nixclaw's
`config/agent.yaml` has `vllm.model` unset, so `VLLMClient` auto-selects
whichever model `/v1/models` reports first (`src/nixclaw_agent/llm.py:754`).
Keeping `--served-model-name qwen3.6` and port `8000` means **no nixclaw-side
config change is needed** — swapping the container is a drop-in replacement
from nixclaw's point of view.

**`--network host --ipc host --privileged`**: taken as-is from the
upstream recipe (needed for GB10 GPU/NVLink access per that repo). This is
more permissive than the current `-p 8000:8000` invocation — worth
confirming you're comfortable with `--privileged` on this box before
running, since it's a real change in container isolation, not just a perf
tweak.

**`stop.sh` fix**: the upstream `stop.sh` writes the *container ID* to
`.vllm.pid` but then tries to `kill` it as if it were a host process PID —
that's a no-op/bug upstream. This version's `stop.sh` just does
`docker stop && docker rm` by container name directly, which is correct
and simpler.

## Prerequisites

- `nvidia/Qwen3.6-27B-NVFP4` is likely a gated HuggingFace repo. Export
  `HF_TOKEN` before running `start.sh` if the download fails with an auth
  error:
  ```bash
  export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  ```
- ~50GB free disk for model weights + caches (downloaded into
  `.cache/huggingface` and `.cache/triton` next to these scripts).
- Docker with the NVIDIA Container Toolkit, `--gpus all` working.
- Port 8000 free on the target box (matches `config/agent.yaml`'s
  `vllm.host: http://<box>:8000`).

## Deploy

Copy this whole `vllm-dgx-spark/` folder to the DGX Spark box, then:

```bash
cd vllm-dgx-spark
./start.sh          # pulls the image, downloads the model + draft model, waits for readiness
```

`start.sh` streams container logs to `.vllm.log` and polls
`http://127.0.0.1:8000/v1/models` until ready. First start will take a
while (model + draft-model download).

## Verify before cutting nixclaw over

```bash
# Model list should show "qwen3.6", same as before:
curl -s http://127.0.0.1:8000/v1/models | jq

# Basic chat + tool-call smoke test:
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "stream": false
  }' | jq

# Decode throughput check (compare against the ~5 tok/s baseline):
curl -s http://127.0.0.1:8000/metrics | grep 'vllm:request_time_per_output_token_seconds_sum\|vllm:request_time_per_output_token_seconds_count'
# mean seconds/token = sum / count; tok/s = 1 / that
```

Since the served model name and port are unchanged, nixclaw will pick this
up automatically on its next request — no `config/agent.yaml` edit needed.
Restart the nixclaw container only if you want it to drop any
already-established connections immediately.

## Rollback

```bash
./stop.sh
```

Then relaunch the original `docker run ... nvcr.io/nvidia/vllm:26.06-py3 ...`
command to go back to the prior setup.

## Sources

- [MiaAI-Lab/Qwen3.6-27B-NVFP4-DFlash-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.6-27B-NVFP4-DFlash-DGX-Spark) — upstream recipe this is adapted from
- [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)
- [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) — draft model for speculative decoding
- [DFlash](https://github.com/nbasyl/DFlash)
