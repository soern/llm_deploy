# vllm-qwen3.8-native-dgx-spark

vLLM launch config for the **native (unquantized BF16) `Qwen/Qwen3.8-27B`**
on the DGX Spark (GB10) box that backs nixclaw's `vllm` LLM client
(`config/agent.yaml` -> `vllm.host`).

This is one of two Qwen3.8 deploy variants — see the sibling
[`../vllm-qwen3.8-nvfp4`](../vllm-qwen3.8-nvfp4) for the
`unsloth/Qwen3.8-27B-NVFP4` quantized alternative. Run **one at a time**
(both default to port 8000 / container-name-scoped so they don't collide,
but they're meant as alternatives, not a pair).

## Why native BF16 vs. NVFP4

Full precision, zero quantization-induced quality loss, at the cost of
~2.3x the weight memory (52GiB vs. ~22.5GiB) and — per this deployment's own
history with the *previous* model generation (see
[`../vllm-qwen3.6/README.md`](../vllm-qwen3.6/README.md)) — meaningfully
slower decode. Pick this variant if quality matters more than latency for
your workload, or as a quality baseline to compare the NVFP4 variant
against; pick NVFP4 if you want more headroom for concurrency/context or
faster responses.

## Model

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) — dense 27B
(not MoE), hybrid architecture: `16 x (3 x (Gated DeltaNet -> FFN) -> 1 x
(Gated Attention -> FFN))` (same family/shape as the Qwen3.6/Qwen3-Next
hybrid-attention line already running on this box). Native 262,144-token
context (extensible to 1M via YaRN — not configured here, see model card if
needed), Apache 2.0 license, native vision (image/video), thinking enabled
by default, Apache-licensed. Training includes multi-token prediction (MTP)
weights, which this config uses for speculative decoding (see below) — no
separate draft-model download needed, unlike the older Qwen3.6+DFlash deploy.

## What this config does, and why

| | Value | Why |
|---|---|---|
| Image | `eugr/spark-vllm:latest` | Widely-used community vLLM build specifically for GB10/sm_121a ([eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker), referenced from NVIDIA's own DGX Spark developer forum) |
| `--dtype` / `--kv-cache-dtype` | `bfloat16` | Explicit, matches the checkpoint's native precision — no quality loss |
| `--gpu-memory-utilization` | `0.92` | See "Memory budget" below |
| `--max-model-len` | `262144` | Native context, no YaRN scaling |
| `--max-num-seqs` / `--max-num-batched-tokens` | `4` / `8192` | Matches a benchmarked bf16 config for the same-size Qwen3.6-27B dense model on this exact hardware (see Sources) |
| `--speculative-config` (MTP, k=3) | on by default | Free speedup — the draft head ships inside the checkpoint itself; toggle off with `ENABLE_MTP=0` if you hit instability |
| `--reasoning-parser qwen3` / `--tool-call-parser qwen3_coder` / `--enable-auto-tool-choice` | | Matches the parsers nixclaw's `tool_markup.py` and the bundled `chat_template.jinja` expect (same family as the current Qwen3.6 deploy) |
| `--generation-config vllm` | | Tells vLLM to load the model's own recommended sampling defaults (thinking-mode: temp 1.0, top_p 0.95, top_k 20, min_p 0.0) — see "Sampling params" below, though, since nixclaw overrides these client-side |
| `--chat-template` | bundled `chat_template.jinja` | Official Qwen3.8 template with three small patches: default `reasoning_effort` `low` (`low`/`medium`/`xhigh` selectable per request via `chat_template_kwargs`), system/developer messages allowed mid-conversation, string tool-call arguments tolerated. Shared with the SGLang deploy; details in `../sglang-qwen3.8/README.md`, "Chat template" |
| Docker networking | `--network host --ipc host --gpus all` | Needed for GPU access; **no `--privileged`** — unlike the older Qwen3.6+DFlash deploy, nothing here needs the extra device access DFlash's custom draft-model wiring required |

### Memory budget

The official BF16 checkpoint is ~52GiB. A community benchmark of the
same-size, same-architecture-family Qwen3.6-27B dense model in BF16 on a
single GB10 used `--gpu-memory-utilization 0.95` successfully (~85-90GiB
actual usage of 128GiB, ~38GiB headroom, 262144 context, max-num-seqs 4) —
see Sources. This config uses **0.92** instead of that benchmark's exact
0.95, since it additionally enables vision buffers and MTP speculative
decoding, which that benchmark did not exercise; treat 0.92 as a
conservative starting point, not a verified number for *this* exact flag
combination. If you hit OOM, lower it (or set `LANGUAGE_ONLY=1` /
`ENABLE_MTP=0` to free budget first); if it starts fine with room to spare,
raising it toward 0.95 is reasonable.

### Sampling params for thinking mode

Qwen3.8's model card recommends, for thinking mode: **temperature 1.0,
top_p 0.95, top_k 20, min_p 0.0**, repetition_penalty 1.0, presence_penalty
0.0. `--generation-config vllm` makes vLLM apply these as server-side
defaults automatically.

**However**, nixclaw's `VllmOptionsConfig` (`src/nixclaw_agent/config.py`)
sends `temperature`/`top_p`/`top_k` explicitly on every request
(`llm.py:1003-1005`), which overrides the server default regardless. Its
current defaults are `temperature=1.0, top_p=0.95, top_k=64` — the first
two already match Qwen3.8's recommendation, but **`top_k=64` does not**
(model card recommends `20`). If you want nixclaw to actually use the
model's recommended thinking-mode sampling, set in `config/agent.yaml`:

```yaml
vllm:
  options:
    top_k: 20
```

This repo's deploy scripts don't make that change for you — it's a
nixclaw-wide config edit outside the scope of a per-model deploy folder,
and affects whichever model is currently being served.

## Prerequisites

- `Qwen/Qwen3.8-27B` may require `HF_TOKEN` if gated:
  ```bash
  export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  ```
- ~60GB+ free disk for model weights + caches (downloaded into
  `.cache/huggingface` and `.cache/vllm-runtime` next to these scripts).
- Docker with the NVIDIA Container Toolkit, `--gpus all` working.
- Port 8000 free on the target box (matches `config/agent.yaml`'s
  `vllm.host: http://<box>:8000`) — or override `PORT` if running alongside
  another variant for comparison.

## Deploy

Copy this whole `vllm-qwen3.8-native-dgx-spark/` folder to the DGX Spark
box, then:

```bash
cd vllm-qwen3.8-native-dgx-spark
./start.sh          # pulls the image, downloads the model, waits for readiness
```

`start.sh` streams container logs to `.vllm.log` and polls
`http://127.0.0.1:8000/v1/models` until ready. First start will take a
while (52GiB model download).

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

# Decode throughput check:
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
`../vllm-qwen3.8-nvfp4/start.sh` for the NVFP4 variant).

## Sources

- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — model card: architecture, context, sampling recommendations, MTP
- Anduin Xue, [Setup DGX Spark with Qwen 3.6 27B Dense](https://anduin.aiursoft.com/post/2026/5/27/setup-dgx-spark-with-qwen-36-27b-dense) — benchmarked BF16-on-single-GB10 config this deploy's memory/batch settings are adapted from
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — GB10/sm_121a vLLM image
- [Qwen/Qwen3.8-27B | vLLM Recipes](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) — confirms MTP speculative decoding applies to non-quantized checkpoints too
- `../vllm-qwen3.6/README.md` — prior deployment's history/rationale (bf16 decode-speed problem that motivated quantization + speculative decoding for the previous model generation)
