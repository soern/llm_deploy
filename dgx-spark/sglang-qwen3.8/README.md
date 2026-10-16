# sglang-qwen3.8-dgx-spark

SGLang launch config for **`nvidia/Qwen3.8-27B-NVFP4`** + **DFlash2**
speculative decoding on the DGX Spark (GB10) box that backs nixclaw's
`vllm` LLM client (`config/agent.yaml` -> `vllm.host`; the client is
backend-agnostic — it just talks OpenAI-compatible `/v1/chat/completions`,
so "vllm" in that config key is historical naming, not a requirement that
the backend actually be vLLM).

**Optimized for 4 concurrent users** based on community validation (NVIDIA
Developer Forums #380257, post #95). Measured 31-36 tok/s mean throughput
at full concurrency.

This is an alternative engine to the `vllm-qwen3.8-*` deploys in this
directory — same model family, different serving stack. Run **one serving
container at a time** (GB10 doesn't have memory for two); relaunch whichever
you want with its own `start.sh`.

## Corrections vs. the earlier draft

This config was originally hand-derived from a third-party repo
(`hasso5703/dgx-spark-qwen38`) surfaced by a forum thread. We since found
**SGLang's own official cookbook page** for this exact model
([docs.sglang.io/cookbook/.../Qwen/Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)),
which documents DGX-Spark-specific tuning validated by the SGLang team
themselves — a materially more authoritative source. Two values changed as
a result:

| Flag | Third-party draft | Official cookbook (now used) | Why |
|---|---|---|---|
| `--mem-fraction-static` | `0.50` | `0.80` | The third-party number came from an incident report ("we learned this the hard way") with no stated validation methodology. The official page re-measured all 48 checkpoint/speculative/strategy/dtype combinations directly on GB10/aarch64: 0.80 served every combo; 0.85 got 15/48 killed by DGX OS's `earlyoom` (SIGTERM, not a hard freeze — see "Memory budget" below) |
| `--chunked-prefill-size` | `8192` | `2048` | Official guidance specific to this model's hybrid GDN architecture: "decode steps stall behind each prefill chunk on hybrid GDN models, and 8192-token chunks stall them ~600ms at a time. 2048 keeps decode inter-token latency smooth" |
| Image | `lmsysorg/sglang@sha256:febfb9…` (third-party-cited digest) | `lmsysorg/sglang:dev-qwen38-27b-dflash2` | Official SGLang-team dev image built and validated for this exact model, superseding a digest we could only trust via a third party's citation of it |

The parser flags (`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`),
DSpark flags, and mamba-cache flags matched between both sources and are
unchanged.

**Caveat on throughput numbers below:** the official cookbook validated DGX
Spark cells only as "boot and serve" at ISL 8192 / OSL 1024, concurrency 1 —
**no throughput or acceptance-length numbers taken**. The tok/s figures
quoted in this README are still the third-party recipe's own benchmarks, not
independently confirmed by SGLang. Run `bench.sh`-style probes yourself
after deploying (see "Verify" below) rather than trusting the numbers below
as-is.

## Why SGLang instead of vLLM for this model

A DGX-Spark-specific benchmark thread reported vLLM at ~21-25 tok/s
single-stream decode for Qwen3.8-27B-NVFP4 on this hardware, vs. SGLang +
speculative decoding at 28-50+ tok/s depending on the draft strategy —
credited to SGLang's FlashInfer kernel integration, CUDA-graph-verified
speculative decoding, and torch.compile support being more mature for this
model/hardware combo than vLLM's equivalent paths at the time. See "Sources"
below.

## Why DFlash2 (switched from DSpark)

This config previously used DSpark speculative decoding but has been updated
to DFlash2 based on community validation and supply-chain reassessment.

**Why the switch:**
1. **Official image support:** The SGLang cookbook ships DFlash2 support in
   the official image (`lmsysorg/sglang:dev-qwen38-27b-dflash2`) with no
   custom build step — no supply-chain risk. Different from the earlier
   DSpark path where DFlash2 required a third-party-built Docker overlay.

2. **Measured throughput:** Community testing (NVIDIA forums post #95)
   reported DFlash2 at 135-148 tok/s aggregate (8 concurrent) vs. DSpark's
   100-104 — a **35-50% throughput win**. This was the deciding factor for
   4-concurrent-user optimization.

3. **Official validation:** The official cookbook confirms "all 12 DGX Spark
   DFlash2 cells boot and serve" — same validation bar as DSpark.

**If you want to revert to DSpark** for reproducibility or other reasons,
swap the launch flags:
```bash
# Current (DFlash2):
--speculative-algorithm DFLASH
--speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2
--speculative-num-draft-tokens 8

# DSpark (older, slower):
--speculative-algorithm DSPARK
--speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark
--speculative-dspark-block-size 7
--speculative-draft-model-quantization unquant
```

## Measured on this box (2026-10-10)

Same SGLang image, flags and DFlash2 draft for both runs; only `MODEL_ID`
changed (`MAX_TOTAL_TOKENS=262144 MAX_MAMBA_CACHE_SIZE=32`). Test: 8
nixclaw-style tool-calling cases (bash, read, write, search, plain answer,
multi-step debugging, a long file write, a retry after a failed tool call
with a mid-conversation system message), 2 runs each per reasoning effort,
single stream, streaming, default sampling, `max_tokens` 6000. "Correct" =
the expected tool (or plain answer) with valid arguments.

| | `Qwen/Qwen3.8-27B-FP8` | `nvidia/Qwen3.8-27B-NVFP4` |
|---|---|---|
| Decode tok/s, answers >= 200 tokens | 29.3 | **39.6** (+35%) |
| Decode tok/s, long file write (~2K tokens) | 27-32 | 40-45 |
| TTFT, thinking on / off | 0.29 s / 0.53 s | 0.21 s / 0.42 s |
| Avg request time at `low` (default) | 10.9 s | 7.9 s |
| Correct: low / medium / off | 16/16 each | 16/16 each |
| Correct: xhigh | 15/16 | 14/16 |

- **Speed**: decode is memory-bandwidth bound; NVFP4 reads ~22GB of weights
  per step vs ~28GB. DFlash2 acceptance is high on both (avg accept length
  4.67 of 8 measured on FP8). The forum's mid-40s figure used greedy
  decoding; the long write here reaches 40-45 with default sampling.
- **Quality**: no difference visible at low, medium and off (same tools,
  same argument shapes, similar file sizes, similar thinking length: avg
  172 vs 175 chars at low). The test checks tool-call correctness, not
  answer quality, so a 4-bit vs 8-bit difference on hard reasoning or long
  code would not show up here.
- **xhigh**: every miss is the long file write: the model thinks for ~20K
  characters and hits `max_tokens` before calling the tool (FP8 1 of 2
  runs, NVFP4 2 of 2; too few runs to call it a quantization effect). Use
  xhigh only with a large output budget; low stays the default.
- First request after a restart had a 3.1 s TTFT (warm-up), excluded from
  the TTFT row.

## What this config does, and why

Flags below are **hand-derived** from the `v1.1` (DSpark) git tag of
[hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38)
— specifically its `run.sh` and `qwen38-sglang.service.template`, fetched
and read directly (not executed) via `raw.githubusercontent.com`. We did
**not** run that repo's `install.sh` (see "Trust notes" below).

| | Value | Why |
|---|---|---|
| Image | `lmsysorg/sglang:dev-qwen38-27b-dflash2` | Official SGLang-team image, validated for this model on GB10/aarch64. Supports both DSpark and DFlash2 without custom build. Moving `:dev-*` tag, not a pinned digest — SGLang doesn't publish a digest-pinned release of it |
| Model | `nvidia/Qwen3.8-27B-NVFP4` (`MODEL_ID`) | NVIDIA ModelOpt quant: NVFP4 MLPs, FP8 attention/linear-attention projections, vision tower unquantized; ~22GB. Its config declares no KV-cache quant, so `start.sh` sets `--kv-cache-dtype fp8_e4m3` explicitly (`KV_CACHE_DTYPE`). Chosen over `Qwen/Qwen3.8-27B-FP8` after measuring both, see "Measured on this box". `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` (the previous default) also works |
| Draft model (speculative) | `incoai/Qwen3.8-27B-DFlash2` | DFlash2 speculative decoding, validated for 4-concurrent workloads. Measured 135-148 tok/s aggregate (8 concurrent) vs. DSpark's 100-104 (community testing, NVIDIA forums #380257) |
| `--mem-fraction-static` | `0.60` | **Tuned for 4 concurrent users** (NVIDIA forums post #95). Community testing found 0.60 stable at full concurrency with headroom for prefill spikes. Official cookbook validated 0.80 as ceiling at concurrency 1, but dynamic workloads benefit from lower ceiling |
| `--attention-backend flashinfer` | | Official cookbook: required on SM120/SM121 (RTX PRO 6000 Blackwell, RTX 5090, DGX Spark) — `trtllm_mha` is SM100-only |
| `--chunked-prefill-size 8192` / `--disable-prefill-cuda-graph` / `--cuda-graph-max-bs 8` | | **Tuned for 4 concurrent users:** larger chunks (8192 vs. 2048 in cookbook) prioritize throughput over per-token latency variance. Community testing found 8192 gives higher mean tok/s at concurrency 4; set to 2048 if low-latency variance matters more than peak throughput |
| FlashInfer autotune | **off** by default (`FLASHINFER_AUTOTUNE=0`) | **Tuned for 4 concurrent users:** disabling gives deterministic, stable throughput (31-36 tok/s) at concurrency 4. Earlier testing on this box found +41-54% at concurrency 1, but community testing shows autotune introduces variance at higher concurrency. Set `FLASHINFER_AUTOTUNE=1` if your workload is mostly single-user |
| `--speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8` | | DFlash2 speculative decoding tuned for 4 concurrent users. 8 draft tokens per step |
| `--mamba-radix-cache-strategy extra_buffer_lazy --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 96` | | Cache tuning for the model's Gated-DeltaNet/Mamba-style linear-attention layers. `96` is the third-party recipe's pinned value, not independently re-derived — the official cookbook's "Mamba ratio calculator" can compute a value tailored to your actual target concurrency; reverify if you change `MAX_RUNNING_REQUESTS` |
| `--max-running-requests 8` | | Concurrency ceiling on the SGLang side — well above nixclaw's own `vllm.options.max_concurrent_requests` client-side gate (default 2), so this isn't the binding constraint |
| `--enable-torch-compile --torch-compile-max-bs 4` | | ~2 tok/s from compiled kernels |
| `--num-continuous-decode-steps 2` | | Minor scheduler throughput gain |
| `--reasoning-parser qwen3 --tool-call-parser qwen3_coder` | | **Same parser names as our vLLM deploys**, and confirmed directly by the official cookbook's "Agent Harnesses" section. This is what makes nixclaw's `tool_markup.py` (which expects server-side `<think>`/`<tool_call>` extraction) work unmodified against this backend |
| `--chat-template` | bundled `chat_template.jinja` | Official Qwen3.8 template with three small patches (see "Chat template" below) |
| Docker: `--memory 100g --memory-swap 100g --shm-size 16g --network host --ipc host` + `${GPU_DOCKER_ARG}` (default `--gpus all`) | | Hard memory cap (cgroup can't see CUDA unified-memory allocations, so this only catches host-RAM runaways, not the GPU-memory trap below); no `--privileged` needed. Override with `GPU_DOCKER_ARG` if this container can't see the GPU |

### Chat template — official Qwen3.8, three small patches

`chat_template.jinja` is the official `Qwen/Qwen3.8-27B` template
(downloaded 2026-10-10) with three changes, each marked `nixclaw:` in the
file. The same file is used by `../vllm-qwen3.8-native` and
`../vllm-qwen3.8-nvfp4`.

1. **Default `reasoning_effort` is `low`** instead of `xhigh`: the model
   overthinks. `low`, `medium` and `xhigh` all stay selectable per request
   via `chat_template_kwargs: {"reasoning_effort": "..."}`; thinking off
   is `{"enable_thinking": false}`. Other values raise the template's
   error. `medium` adds no instruction; `low` and `xhigh` add one sentence
   to the system prompt.
2. **System messages after the first are rendered as their own system
   turn** instead of raising "System message must be at the beginning."
   nixclaw sends them on most turns (history keywords and summary right
   after the system prompt, `llm.py`; compaction summary,
   `core.py`; context-exceeded notice, `tool_loop.py`), and the Pi coding
   agent does too. The official template rejects all of these with an
   HTTP 400.
3. **`developer` is treated like `system`** (Pi sends its system prompt as
   `developer` unless told otherwise), and tool-call arguments that are
   still a JSON string are emitted as-is instead of failing.

Everything else is Qwen's: tool-call instructions as trained,
`preserve_thinking` on by default, no extra warnings.

Before 2026-10-10 the deploys used the community template
`qwen3.8-froggeric-v20` (inherited from the Qwen3.6 deploy). It had no
`reasoning_effort` support (thinking on behaved like `medium`), defaulted
`preserve_thinking` to off, used its own tool-call instructions, and added
"SYSTEM WARNING" text after tool results that looked like errors, turning
thinking off after two of them. It is in git history if a comparison is
needed.

The source recipe's `patch-template.py` fixes the same system-message
problem for Claude Code and remaps `reasoning_effort: "max"` to `xhigh`;
here `max` is rejected (Pi maps its levels itself, see below).

### Using it from the Pi coding agent

Pi (`@earendil-works/pi-coding-agent`) registers a self-hosted
OpenAI-compatible server in `~/.pi/agent/models.json`. Pi assumes a
standard OpenAI server for unknown URLs, so set its compat options
explicitly:

```json
{
  "providers": {
    "sparky": {
      "baseUrl": "http://sparky.lan:8000/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "models": [{
        "id": "qwen3.8",
        "name": "Qwen3.8-27B (Spark)",
        "reasoning": true,
        "input": ["text"],
        "contextWindow": 131072,
        "maxTokens": 32768,
        "thinkingLevelMap": {
          "minimal": "low", "low": "low", "medium": "medium",
          "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"
        },
        "compat": {
          "thinkingFormat": "chat-template",
          "chatTemplateKwargs": {
            "enable_thinking": { "$var": "thinking.enabled" },
            "reasoning_effort": { "$var": "thinking.effort", "omitWhenOff": true },
            "preserve_thinking": true
          },
          "supportsDeveloperRole": false,
          "supportsMidConvoSystemMessages": true,
          "supportsStore": false,
          "maxTokensField": "max_tokens"
        }
      }]
    }
  }
}
```

- `thinkingFormat: "chat-template"` sends the level as
  `chat_template_kwargs`, which the template reads directly; Pi's default
  sends OpenAI's top-level `reasoning_effort`, which SGLang may not pass to
  the template. Pi's 7 levels map onto Qwen's 3 plus off.
- `supportsMidConvoSystemMessages: true`: otherwise Pi folds later system
  messages into the first one, which changes the prompt prefix and defeats
  the radix cache.
- `supportsStore: false` / `maxTokensField: "max_tokens"`: avoids fields
  SGLang may not accept (not tested).
- `contextWindow: 131072`: `MAX_TOTAL_TOKENS` is the KV pool shared by all
  requests, so a Pi session near 262K would block nixclaw.

### Memory budget — read before changing `MEM_FRACTION_STATIC`

`--mem-fraction-static 0.80` is **not** the same kind of knob as vLLM's
`--gpu-memory-utilization` (see the sibling `vllm-qwen3.8-nvfp4-dgx-spark`
README's disputed-value discussion) — different engine, different
accounting model. This is the value the **official SGLang cookbook**
validated directly on DGX Spark: DGX Spark's 128GB is unified memory shared
with the host CPU, and the cookbook re-measured all 48
checkpoint/speculative/strategy/dtype combinations for this model on
GB10/aarch64. At `0.80`, every combination served; at `0.85` (which leaves
~8GB for the OS — exactly DGX OS's `earlyoom` SIGTERM threshold), 15 of 48
cells got killed mid-boot ("the first long prefill or boot-time graph
capture dips under it and gets the scheduler killed with `exit code -15`,"
visible in `journalctl -u earlyoom`). That's a graceful kill, not the hard
freeze a third-party recipe warned about at a much lower threshold (0.50) —
we're not sure why the two sources' safe ceilings differ so much (possibly
different SGLang versions' memory accounting, or `earlyoom` mitigating what
used to be a harder failure), but we're following the official, directly-
measured number for this hardware over the older incident report. Treat
`0.80` as validated, not `0.85`+ — if you do experiment upward, watch
`journalctl -u earlyoom` and `MemAvailable` in `/proc/meminfo` (not
`nvidia-smi`, which reports "Not Supported" for memory on this unified-
memory hardware).

**Explicit cache sizes (2026-10-09).** Without `--max-total-tokens`, SGLang
sizes the KV pool from the memory free at launch, so its footprint depended on
whether qwen-image was already loaded (671k tokens / 26.9 GB on an empty box).
Sharing the box with qwen-image and qwen3-tts, the defaults now pin it:
`MAX_TOTAL_TOKENS=262144` (the model's max context; ~40 KB/token for main +
draft KV in fp8, about 10.5 GB) and `MAX_MAMBA_CACHE_SIZE=32` (~127 MB/slot,
about 4 GB). `MEM_FRACTION_STATIC` stays as the ceiling. A request that doesn't
fit waits in SGLang's queue (two 200k-token requests run one after the other);
none can be larger than the pool, because the pool equals the max context.
Measured use with nixclaw before the change: at most 10% of 671k tokens and
12 of 96 mamba slots.

## Known gaps — verify before cutting over

- **Vision/multimodal support is unconfirmed.** The source launch command
  has no multimodal flags (no equivalent of our vLLM deploys'
  `--limit-mm-per-prompt`/`--media-io-kwargs`), and the DSpark/Mamba-cache
  speculative path may not be wired up for image/video input at all. Test
  nixclaw's `analyze_image` tool against this backend before relying on it;
  if it doesn't work, keep `vllm.vision_model` pointed at a vLLM deploy that
  does vision while using this one for text.
- **`preserve_thinking`**: on by default in the template (Qwen's
  default), so prior-turn reasoning is kept without a server flag; the vLLM
  deploys also set it explicitly via `--default-chat-template-kwargs`.
- **No `--api-key` set**, matching nixclaw's existing posture (its
  `VLLMClient` sends no `Authorization` header) and the other `deploy/`
  scripts here — fine on a trusted internal network, same as before.

## Prerequisites

- The default weights (`nvidia/Qwen3.8-27B-NVFP4`) and the DFlash2 draft are
  not gated. Other `MODEL_ID`s may require `HF_TOKEN` if gated:
  ```bash
  export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  ```
- ~30GB free disk for checkpoints + caches (downloaded into
  `.cache/huggingface` and `.cache/sglang-runtime` next to these scripts).
- Docker with the NVIDIA Container Toolkit. `start.sh` defaults to
  `--gpus all`; if that can't see the GPU on your box, the official cookbook
  notes DGX Spark is CDI-only (no `nvidia` runtime registered) — retry with
  `GPU_DOCKER_ARG="--device nvidia.com/gpu=all" ./start.sh`.
- Port 8000 free on the target box (matches `config/agent.yaml`'s
  `vllm.host: http://<box>:8000` — note the upstream source recipe defaults
  to port 30000; we default to 8000 here instead, to match nixclaw's
  existing swap-in convention across all `deploy/*` folders) — or override
  `PORT` if running alongside another variant for comparison.
- ~121GB unified memory (GB10 spec); this config needs the machine to
  itself — no other inference server running concurrently.

## Deploy

Copy this whole `sglang-qwen3.8-dgx-spark/` folder to the DGX Spark box,
then:

```bash
cd sglang-qwen3.8-dgx-spark
./start.sh          # pulls the image, downloads both checkpoints, waits for readiness
```

`start.sh` streams container logs to `.sglang.log` and polls
`http://127.0.0.1:8000/health` until ready. First boot takes ~7-9 minutes
(torch.compile + CUDA graph capture); later boots reuse the compile caches
under `.cache/sglang-runtime` (Inductor in `inductor/`, Triton in `triton/`,
FlashInfer/DeepGEMM/other `~/.cache` JIT output in `home-cache/`) and are
faster. CUDA graph capture, weight loading and memory profiling still run on
every boot - they are per-process state and cannot be cached. Pulling a new
`IMAGE` or switching `MODEL_ID` invalidates the caches and costs one slow boot.

Useful overrides (env vars, see top of `start.sh`): `MEM_FRACTION_STATIC`,
`MAX_RUNNING_REQUESTS`, `MAX_TOTAL_TOKENS`, `MAX_MAMBA_CACHE_SIZE`, `PORT`, `CONTAINER_NAME`, `SERVED_MODEL_NAME`,
`MODEL_REV`, `DRAFT_REV`, `IMAGE`, `GPU_DOCKER_ARG`, `FLASHINFER_AUTOTUNE`,
`MAMBA_SSM_DTYPE`.

### Tuning — 4-concurrent optimization

This config was re-tuned based on community validation (NVIDIA Developer
Forums #380257, post #95) for 4-concurrent-user workloads (nixclaw's target).

**Single-user baseline (DSpark, autotune ON):** 23.2/29.7/10.8 tok/s
(code/reasoning/free-prose)

**4-concurrent optimized (DFlash2, autotune OFF, mem 0.60, prefill 8192):**
31-36 tok/s mean (community measurement at 4 concurrent)

**Key trade-offs:**
- Larger `--chunked-prefill-size` (8192 vs. 2048) increases mean throughput
  but adds latency variance during prefill
- Lower `--mem-fraction-static` (0.60 vs. 0.80) provides stability and
  headroom for concurrent spikes, measured as safe by community at 4
  concurrent
- `FLASHINFER_AUTOTUNE=0` trades the +41-54% single-user peak from autotune
  for deterministic performance at concurrency 4 (community found autotune
  introduces stalls at higher concurrency)
- DFlash2 draft model (+35-50% throughput vs. DSpark) now worth the switch,
  as official image support removes earlier supply-chain concerns

Earlier single-user benchmarking found:
- `FLASHINFER_AUTOTUNE=1` gave +41-54% on this box at concurrency 1
- `MAMBA_SSM_DTYPE=float32` was tested and rejected (15% worse on reasoning,
  2x memory cost)

Each restart costs ~7-9min (autotune OFF, so no recompile). To re-baseline:
restart with new flags and benchmark from a warm (already-serving) state.

## Auto-restart and OOM kill order

The LLM is the service on the Spark that matters most, so it gets two
protections the image and TTS servers don't.

**OOM kill order.** When memory runs out, Linux kills the process with the
highest `oom_score`. On unified memory, GPU allocations don't count toward
that score, so the `--oom-score-adj` each container gets decides the order:

| Container | `OOM_SCORE_ADJ` | Killed |
|---|---|---|
| `qwen3-tts` | +800 | first |
| `qwen-image-2.1` | +500 | second |
| `sglang-qwen3.8-dspark` | -900 | last, only if nothing else frees enough |

-900 rather than -1000: a runaway SGLang can still be killed instead of
freezing the box. DGX OS's `earlyoom` ranks by the same score.

**Watchdog** (`watchdog.sh`, run by `sglang-watchdog.service`). The container
itself has no Docker restart policy; the watchdog checks every 30 s and

- starts it again after a crash, OOM kill or reboot, but only once
  `MemAvailable` reaches 62 GB (`REQUIRED_AVAILABLE_MB`). It waits up to 2 min
  for a killed process's memory to come back, then stops `qwen3-tts` and then
  `qwen-image-2.1` until there is room. Without room it doesn't start
  SGLang, because it would only thrash; it tries again on the next check;
- restarts a hung server: `/health` failing 10 times in a row **and** no
  `Prefill batch`/`Decode batch` log line for 5 min. That was the 2026-10-09
  failure: the HTTP server answered, the scheduler had been silent for 3 h.
  A server that is merely saturated also fails `/health` (it queues a test
  generation), but keeps logging batches, so it is left alone;
- gives a fresh start 30 min to become healthy (boot with torch.compile and
  CUDA graph capture takes ~10-15 min);
- stops `qwen3-tts`, then `qwen-image-2.1` when `MemAvailable` falls below
  3 GB on two checks in a row while SGLang runs, before the box starts
  swapping;
- starts the containers it stopped again once SGLang is healthy, at least
  30 min later and only with 20 GB free, one per check;
- restarts SGLang at most 3 times per hour, then only logs.

Install (on the Spark, from this folder):

```bash
sed -e "s|@WORK_DIR@|$PWD|" -e "s|@USER@|$USER|" sglang-watchdog.service \
  | sudo tee /etc/systemd/system/sglang-watchdog.service >/dev/null
sudo systemctl daemon-reload && sudo systemctl enable --now sglang-watchdog
journalctl -u sglang-watchdog -f
```

Settings go in an optional `watchdog.env` next to the script (`VAR=value`
lines, see the top of `watchdog.sh`), e.g. `NOTIFY_CMD=/path/to/script` to
get a message for every action. To stop SGLang on purpose use `./stop.sh`:
it removes the container, which the watchdog leaves alone. A plain
`docker stop` looks like a crash and gets restarted. The watchdog restarts the
existing container (`docker start`), so it keeps the settings `start.sh`
created it with; after changing settings run `./stop.sh && ./start.sh`.

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

# Vision smoke test (known gap above — confirm one way or the other):
# send an image via nixclaw's analyze_image tool against this backend.
```

Since `vllm.model` is unset in `config/agent.yaml`, `VLLMClient`
auto-selects whichever model `/v1/models` reports first
(`src/nixclaw_agent/llm.py:754`) — swapping this container in needs no
nixclaw-side config change. Restart the nixclaw container only if you want
it to drop already-established connections immediately.

## Rollback

```bash
./stop.sh
```

Then relaunch whatever was previously running (e.g.
`../vllm-qwen3.8-nvfp4/start.sh`).

## Trust notes

We read (via `raw.githubusercontent.com`, not execution) the `v1.1` tag's
`install.sh`, `run.sh`, `qwen38-sglang.service.template`, and
`patch-template.py` from `hasso5703/dgx-spark-qwen38` to derive the flags
above. We deliberately did **not**:

- run their `install.sh`/`get.sh` (`curl | bash`, requires `sudo`, installs
  a systemd service, writes files under `~/.config`),
- use their systemd unit (none of our `deploy/*` folders touch systemd —
  plain `docker run`, matching this repo's existing pattern),
- build or run their DFlash2 overlay image (`dflash2/build-image.sh`,
  contents not reviewed),
- use their Claude-Code integration env file or `patch-template.py`
  (see "Chat template" above).

## Sources

- [NVIDIA Developer Forums thread #380257, post #95](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257/95) — **primary source for 4-concurrent tuning**: community validation of DFlash2, `mem-fraction 0.60`, `chunked-prefill 8192`, and `autotune OFF` for stable multi-user operation; measured 31.6 tok/s mean at 4 concurrent
- [SGLang official cookbook: Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) — SGLang-team-validated DGX Spark tuning (concurrency 1), official image, "Agent Harnesses" parser confirmation, Mamba-ratio calculator
- [hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38), `v1.1` tag — earlier draft's source for launch flags; values have been superseded by official sources
- [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) — model card (weights, default)
- [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) — previous default (`-BF16-LMHead` variant)
- [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) — model card (draft model); used for speculative decoding
- `../vllm-qwen3.8-nvfp4/README.md` — sibling vLLM deploy; shared chat template, parser-flag cross-check
