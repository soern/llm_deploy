# sglang-minicpm5-2b-dgx-spark

SGLang launch config for **`openbmb/MiniCPM5-2B`** (native BF16, no
quantization) on the DGX Spark (GB10) box that backs nixclaw's `vllm` LLM
client (`config/agent.yaml` -> `vllm.host`). It serves the same
OpenAI-compatible `/v1` API as the sibling `vllm-*` and
`sglang-qwen3.8-dgx-spark` deploys, so nixclaw's `VLLMClient` works against
it unchanged.

## Why SGLang instead of vLLM

MiniCPM5-2B emits XML-style tool calls — a `<function>` element wrapping
`<param>` elements. SGLang ships a built-in `minicpm5` detector that
converts them to OpenAI-compatible `tool_calls` (parallel calls included),
which is what nixclaw's tool loop consumes. The model card and SGLang's
official cookbook both recommend SGLang for this model's tool calling on
exactly this basis; vLLM has no equivalent `minicpm5` parser as of this
writing.

## Model

[`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) — dense
**2.5B**-parameter Transformer (`LlamaForCausalLM`, 42 layers, GQA 16
query / 2 KV heads, head_dim 128), Apache 2.0, **not gated** (no
`HF_TOKEN` needed). Native context **131,072** tokens. Standard Llama
architecture — no custom kernels or model-code fork — so it loads through
the stock Llama path. Text-only (no vision). Second model in OpenBMB's
MiniCPM5 series, post-trained SFT → RL → On-Policy Distillation; OpenBMB
reports 2B-class open-source SOTA in their comparison set (avg 53.9), with
the clearest advantages in code, math, long-context, tool use and agentic
tasks. Weights: ~5.0GiB BF16.

## What this config does, and why

| | Value | Why |
|---|---|---|
| Image | `lmsysorg/sglang:dev` | SGLang's official image; the `minicpm5` tool-call detector and the DSPARK draft worker ship in the dev image (per the cookbook's config source). Moving `:dev` tag — pull once and keep the local copy if you want reproducibility across restarts |
| `--model-path` | `openbmb/MiniCPM5-2B` | Bare repo id — the image's server args pass it straight to transformers, whose `validate_repo_id` rejects `@`, so a `repo@revision` string dies at startup with an `HFValidationError`. The revision `12a3808a…` (as of 2026-09-12) is recorded in `start.sh` for provenance only |
| `--tp-size 1` | | 2.5B fits one GPU on every card the cookbook supports; TP=1 is the verified shape |
| `--mem-fraction-static 0.30` | **Verified, required on GB10** — see "Memory budget" below | From the official cookbook's verified DGX Spark cell (sglang dev @ 30705c004c) |
| `--cuda-graph-max-bs-decode 128` | | Same verified cookbook cell |
| `--reasoning-parser qwen3` | | The chat template is Qwen-style (`system` + `think` tags); there is no `minicpm5` reasoning parser. Splits the reasoning segment into `message.reasoning_content` |
| `--tool-call-parser minicpm5` | | See "Why SGLang" above; without it `tool_calls` comes back null and the XML leaks into `content` |
| Chat template | checkpoint's own `chat_template.jinja` | No customization needed, so `start.sh` does not mount one (unlike the sibling qwen3.8 deploys) |
| Docker networking | `--network host --ipc host` + `--gpus all` (default) | Same as the sibling deploys; `GPU_DOCKER_ARG` overrides for CDI-only setups |

### Memory budget

GB10 has no discrete VRAM — SGLang's `mem_get_info()` reports the full
128GB of unified memory, and the **default** static fraction would claim
~89GB for the KV pool, leaving ~5GB for the OS. The cookbook's verified
DGX Spark cell documents that this kills the node during warmup with no
traceback and no OOMKilled event, and pins `--mem-fraction-static 0.30` as
*required, not tuning*. It still leaves a ~658k-token KV pool — far more
than a 2.5B model with 131K context needs at nixclaw's concurrency.

Do **not** raise this toward the sibling `sglang-qwen3.8-dgx-spark`
deploy's 0.80: that value was validated for a different model (hybrid
mamba) under an explicit cgroup memory cap, and the two memory-accounting
situations are not comparable.

### DSpark speculative decoding (opt-in)

OpenBMB publishes a BF16 draft checkpoint
(`openbmb/MiniCPM5-2B-DSpark`, ~648MiB) for SGLang's DSPARK speculative
algorithm. `start.sh` ships it as an opt-in overlay: `ENABLE_DSPARK=1`
adds `--speculative-algorithm DSPARK`, the draft model path,
`--speculative-dspark-block-size 7`, and `--trust-remote-code` (the base
checkpoint is plain Llama and does not need it; the draft's config does).

It is off by default because the cookbook publishes no DSpark speed
numbers on any card yet — the speedup tracks acceptance length, which moves
with the prompt distribution. If you enable it, compare tok/s against the
baseline numbers below before adopting it for the agent's main backend.

### Sampling params

Model card recommends: **temperature 1.0, top_p 0.95, min_p 0.0**
(repetition_penalty 1.05 if outputs repeat). SGLang reads the checkpoint's
`generation_config.json` (temperature 1.0, top_p 0.95) as server defaults,
but nixclaw's `VllmOptionsConfig` sends `temperature`/`top_p`/`top_k`
explicitly on every request, overriding the server defaults regardless. Its
defaults (`temperature=1.0, top_p=0.95, top_k=64`) are close enough to the
model's recommendation; `top_k` is not in the model card's recommended set,
so if you see quality issues try `vllm.options.top_k: 20` in
`config/agent.yaml`. This deploy does not make that change for you — same
policy as the sibling qwen3.8 deploys.

## Prerequisites

- **Not gated** — no `HF_TOKEN` required (pass one anyway via the env if
  you want it cached in the container).
- ~6GB free disk for model weights + caches (downloaded into
  `.cache/huggingface` next to these scripts; another ~650MiB if you use
  DSpark).
- Docker with the NVIDIA Container Toolkit, `--gpus all` working.
- Port 8000 free on the target box (matches `config/agent.yaml`'s
  `vllm.host`) — or override `PORT` when running alongside another variant.

## Deploy

Copy this whole `sglang-minicpm5-2b-dgx-spark/` folder to the DGX Spark
box, then:

    ./start.sh          # pulls the image, downloads the model, waits for readiness

`start.sh` streams container logs to `.sglang.log` and polls
`http://127.0.0.1:8000/health` until ready. First start will pull the dev
image and download ~5GiB of weights.

Useful overrides (env vars, see top of `start.sh`): `ENABLE_DSPARK`,
`DRAFT_MODEL_ID`, `DSPARK_BLOCK_SIZE`, `MEM_FRACTION_STATIC`,
`CUDA_GRAPH_MAX_BS_DECODE`, `PORT`, `CONTAINER_NAME`, `SERVED_MODEL_NAME`,
`MODEL_ID`, `MODEL_REV`, `DRAFT_MODEL_ID`, `DRAFT_REV` (provenance only —
see `--model-path` row above; the revs are not passed to the server),
`IMAGE`, `GPU_DOCKER_ARG`.

## Verify before cutting nixclaw over

    # Model list should show "minicpm5-2b":
    curl -s http://127.0.0.1:8000/v1/models | jq

    # Basic chat smoke test:
    curl -s http://127.0.0.1:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "minicpm5-2b",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 64,
        "stream": false
      }' | jq

    # Tool-call smoke test — tool_calls must be populated, content may be
    # empty on a tool-call turn:
    curl -s http://127.0.0.1:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "minicpm5-2b",
        "messages": [{"role": "user", "content": "What is the weather in Beijing today?"}],
        "tools": [{"type": "function", "function": {"name": "get_weather",
                   "description": "Get the current weather for a city.",
                   "parameters": {"type": "object",
                     "properties": {"city": {"type": "string"}},
                     "required": ["city"]}}}]
      }' | jq

Baseline DGX Spark throughput from the cookbook (same image, random
1024/1024 workload): ~72 tok/s at concurrency 1, ~2892 tok/s aggregate at
concurrency 64, TTFT 85ms / 1373ms (Mean). DSpark carries no published
speed numbers.

Since `vllm.model` is unset in `config/agent.yaml`, `VLLMClient`
auto-selects whichever model `/v1/models` reports first
(`src/nixclaw_agent/llm.py`) — swapping this container in for whatever was
previously running needs no nixclaw-side config change, same as the
existing qwen3.6/qwen3.8 deploys. Restart the nixclaw container only if
you want it to drop already-established connections immediately.

Note: thinking mode is **off** by default here — the model's chat template
only emits the thinking segment when `chat_template_kwargs.enable_thinking`
is sent per request, and nixclaw's `VLLMClient` does not send it. If you
want thinking on for nixclaw, that is a client-side change
(`extra_body={"chat_template_kwargs": {"enable_thinking": true}}`), outside
the scope of this deploy.

## Rollback

    ./stop.sh

Then relaunch whatever was previously running (e.g.
`../sglang-qwen3.8/start.sh` or
`../vllm-qwen3.8-native/start.sh`).

## Sources

- [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) — model card: architecture, context, sampling recommendations, SGLang/vLLM launch commands
- [SGLang cookbook: MiniCPM5-2B](https://docs.sglang.io/cookbook/autoregressive/OpenBMB/MiniCPM5-2B) — verified per-card launch cells incl. the DGX Spark cell this deploy is based on (verified at sglang dev @ 30705c004c)
- `docs/src/snippets/configs/openbmb/minicpm5-2b.jsx` in the [sgl-project/sglang](https://github.com/sgl-project/sglang) repo — machine-readable config source for the cookbook cells
- [openbmb/MiniCPM5-2B-DSpark](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark) — draft checkpoint for the opt-in DSpark overlay
- `../sglang-qwen3.8/README.md` — sibling deploy's structure/rationale this one follows
