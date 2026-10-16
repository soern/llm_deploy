# vllm-qwen3.8-nvfp4-dgx-spark

vLLM launch config for **`unsloth/Qwen3.8-27B-NVFP4`** on the DGX Spark
(GB10) box that backs nixclaw's `vllm` LLM client (`config/agent.yaml` ->
`vllm.host`).

This is one of two Qwen3.8 deploy variants — see the sibling
[`../vllm-qwen3.8-native`](../vllm-qwen3.8-native) for
the unquantized BF16 alternative. Run **one at a time** (both default to
port 8000; they're meant as alternatives, not a pair).

## Why NVFP4 vs. native

~4x smaller weights (22.5GiB + 0.81GiB MTP weights, vs. 52GiB for BF16),
leaving far more headroom on GB10's 121GiB usable unified memory for KV
cache / concurrency / vision buffers / speculative decoding, at the cost of
some quantization-induced quality loss. The unsloth NVFP4 build is
specifically called out by the source recipe below as the memory-sane
choice for this exact hardware, where BF16 "does not leave enough
unified-memory headroom." Pick this variant for throughput/concurrency;
pick native BF16 if quality matters more than latency.

## Model

[`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
— NVFP4 quantization of `Qwen/Qwen3.8-27B` via "Unsloth Dynamic V3.0
(preview)". Same dense 27B hybrid `Gated DeltaNet` / `Gated Attention`
architecture as the base model (hidden dim 5120, 64 layers), native
262,144-token context, native vision (image/video), thinking enabled by
default. Includes MTP (multi-token-prediction) weights for speculative
decoding — no separate draft-model download needed, unlike the older
Qwen3.6+DFlash deploy in `../vllm-qwen3.6`.

## What this config does, and why

Adapted directly from
[gitcommit90/qwen38-27b-dgx-spark](https://github.com/gitcommit90/qwen38-27b-dgx-spark),
a start.sh **specifically written and benchmarked for this exact
checkpoint on this exact hardware** (single GB10). Reported: ~21.4 tok/s
single-stream decode, ~66.5 tok/s aggregate throughput at concurrency 4
(~19.9 tok/s per-stream), ~1s TTFT, 81.1GiB KV cache observed at 8.73x
concurrency headroom.

| | Value | Why |
|---|---|---|
| Image | `eugr/spark-vllm:latest` | Same widely-used community GB10/sm_121a vLLM build as the source recipe ([eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)) |
| `--quantization` | `compressed-tensors` | Explicit NVFP4 quant format for this checkpoint |
| `--gpu-memory-utilization` | `0.90` | The source recipe's actual benchmarked value (see "Memory budget" below for a disputed alternative) |
| `--max-model-len` | `262144` | Native context |
| `--max-num-seqs` / `--max-num-batched-tokens` | `4` / `16384` | Source recipe's benchmarked values |
| `--speculative-config` (MTP, k=3) | on by default | Free speedup — draft head ships inside the checkpoint; toggle off with `ENABLE_MTP=0` |
| `--reasoning-parser qwen3` / `--tool-call-parser qwen3_coder` / `--enable-auto-tool-choice` | | Matches the parsers nixclaw's `tool_markup.py` and the bundled `chat_template.jinja` expect (same family as the current Qwen3.6 deploy) |
| `--generation-config vllm` | | Loads the model's recommended thinking-mode sampling defaults server-side — see "Sampling params" below, though, since nixclaw overrides these client-side |
| `--chat-template` | bundled `chat_template.jinja` | Official Qwen3.8 template with three small patches: default `reasoning_effort` `low` (`low`/`medium`/`xhigh` selectable per request via `chat_template_kwargs`), system/developer messages allowed mid-conversation, string tool-call arguments tolerated. Shared with the SGLang deploy; details in `../sglang-qwen3.8/README.md`, "Chat template" |
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | | From the source recipe — Marlin-kernel tuning for the FP4 GEMM path |
| Docker networking | `--network host --ipc host --gpus all` | Needed for GPU access; **no `--privileged`** — unlike the older Qwen3.6+DFlash deploy, this uses MTP (built into vLLM/the checkpoint) rather than a custom external draft-model repo, so nothing needs the extra device access DFlash's wiring required |

### Memory budget

`--gpu-memory-utilization 0.90` is the source recipe's actual benchmarked
value (see throughput numbers above — it ran clean at concurrency 4 with
81.1GiB observed KV cache headroom). A commenter on the
[unsloth NVFP4 discussion board](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/discussions/7)
argued for something much lower — ~0.39-0.42 — based on a back-of-envelope
calculation (weights ~23.4GiB + ~16GiB for full 262144-token context ≈
39.4GiB total), with the commenter noting themselves "I haven't tested
lower than 0.42 though." **This estimate was not reproduced against actual
throughput/OOM numbers on this hardware**, unlike 0.90, and the math is a
bit off vs. GB10's actual usable memory (121GiB × 0.42 ≈ 51GiB, well above
their own 39.4GiB estimate) — so treat it as an unverified lower bound, not
a recommendation. Use it only if you need more headroom for other
processes on the box and are willing to accept less KV-cache/concurrency
room, e.g. `GPU_MEM_UTIL=0.42 ./start.sh`.

### Sampling params for thinking mode

Same recommendation as the base model: thinking mode wants **temperature
1.0, top_p 0.95, top_k 20, min_p 0.0**, repetition_penalty 1.0,
presence_penalty 0.0 (the unsloth card notes higher `presence_penalty`
values "may occasionally result in language mixing and a slight decrease
in model performance" — another reason to leave it at 0.0).
`--generation-config vllm` applies these server-side, but **nixclaw's
`VllmOptionsConfig` overrides `temperature`/`top_p`/`top_k` on every
request** (`llm.py:1003-1005`) with its own defaults
(`temperature=1.0, top_p=0.95, top_k=64`). The first two already match;
`top_k=64` doesn't (recommended: `20`). To make nixclaw actually use it,
set in `config/agent.yaml`:

```yaml
vllm:
  options:
    top_k: 20
```

Not done automatically by these scripts — it's a nixclaw-wide config edit
outside a per-model deploy folder's scope, and affects whichever model is
currently being served.

## Prerequisites

- `unsloth/Qwen3.8-27B-NVFP4` may require `HF_TOKEN` if gated:
  ```bash
  export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  ```
- ~30GB free disk for model weights + caches (downloaded into
  `.cache/huggingface` and `.cache/vllm-runtime` next to these scripts).
- Docker with the NVIDIA Container Toolkit, `--gpus all` working.
- Port 8000 free on the target box (matches `config/agent.yaml`'s
  `vllm.host: http://<box>:8000`) — or override `PORT` if running alongside
  another variant for comparison.
- `ninja-build` on the host if the `eugr/spark-vllm` image needs to JIT
  compile FlashInfer sampling kernels on first run (noted as a prerequisite
  by the source recipe).

## Deploy

Copy this whole `vllm-qwen3.8-nvfp4-dgx-spark/` folder to the DGX Spark
box, then:

```bash
cd vllm-qwen3.8-nvfp4-dgx-spark
./start.sh          # pulls the image, downloads the model, waits for readiness
```

`start.sh` streams container logs to `.vllm.log` and polls
`http://127.0.0.1:8000/v1/models` until ready.

Useful overrides (env vars, see top of `start.sh`): `GPU_MEM_UTIL`,
`MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `ENABLE_MTP`,
`NUM_SPECULATIVE_TOKENS`, `LANGUAGE_ONLY`, `PORT`, `CONTAINER_NAME`,
`SERVED_MODEL_NAME`.

## Verify before cutting nixclaw over

```bash
# Model list should show "qwen3.8":
curl -s http://127.0.0.1:8000/v1/models | jq

# Basic chat + tool-call smoke test:
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "stream": false
  }' | jq

# Decode throughput check (compare against the ~21.4 tok/s single-stream
# baseline reported by the source recipe):
curl -s http://127.0.0.1:8000/metrics | grep 'vllm:request_time_per_output_token_seconds_sum\|vllm:request_time_per_output_token_seconds_count'
# mean seconds/token = sum / count; tok/s = 1 / that
```

Since `vllm.model` is unset in `config/agent.yaml`, `VLLMClient`
auto-selects whichever model `/v1/models` reports first
(`src/nixclaw_agent/llm.py:754`) — swapping this container in for whatever
was previously running needs no nixclaw-side config change, same as the
existing Qwen3.6 deploy. Restart the nixclaw container only if you want it
to drop already-established connections immediately.

## Rollback

```bash
./stop.sh
```

Then relaunch whatever was previously running (e.g.
`../vllm-qwen3.6/start.sh` for the Qwen3.6 config, or
`../vllm-qwen3.8-native/start.sh` for the BF16 variant).

## Sources

- [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) — model card: quantization, architecture, sampling recommendations
- [gitcommit90/qwen38-27b-dgx-spark](https://github.com/gitcommit90/qwen38-27b-dgx-spark) — this deploy's primary source: benchmarked start.sh for this exact checkpoint+hardware
- [Discussion #7 on the unsloth NVFP4 repo](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/discussions/7) — community-proposed (unverified) lower `--gpu-memory-utilization` bound, discussed above
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — GB10/sm_121a vLLM image
- [Qwen/Qwen3.8-27B | vLLM Recipes](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) — cross-check on quantization/reasoning/tool-call flags
- `../vllm-qwen3.6/README.md` — prior deployment's history/rationale (why the earlier Qwen3.6 deploy needed DFlash speculative decoding; MTP here serves the same throughput goal without a separate draft-model repo)
