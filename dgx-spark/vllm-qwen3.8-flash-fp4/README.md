# vllm-qwen3.8-flash-fp4-dgx-spark

vLLM launch config for
[`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
on the DGX Spark (GB10), sized to run **alongside `../qwen-image-2.1`**
(~30 GB). Serves nixclaw's `vllm` client on port 8000, like the other
Qwen3.8 deploys. Run one LLM deploy at a time.

**Status: running on the Spark** (2026-10-20) with qwen-image-2.1
alongside, at `GPU_MEM_UTIL` 0.69 and full 262K context. Memory numbers
below are measured. Not yet verified: both models under peak load at once
(see "Remaining check" below).

## Model

NVIDIA's QAD-trained FP4 quant of Qwen3.8-Flash-Next: 125B-parameter MoE,
6B active per token, plus a 51B-parameter n-gram embedding table and a 4B
MTP draft head. Hybrid attention (Gated DeltaNet + sparse attention),
vision input, 262K native context.

Quality vs. the dense 27B (`../vllm-qwen3.8-nvfp4`), from Qwen's model
card (BF16): ahead on every shared benchmark except one tie. The gap is
small on single-turn reasoning (GPQA 91.7 vs 89.2) and large on long
agent tasks (DeepSWE 58.7 vs 42.2, JobBench 55.7 vs 33.4). NVIDIA's FP4
costs ~0.7 points (GPQA 92.2 → 91.5).

Checkpoint: 95 GiB on disk, measured from the safetensors headers:

| Part | Size | Where it lives here |
|---|---|---|
| Routed experts (NVFP4) | 59.9 GiB | memory |
| N-gram (PLE) table (NVFP4) | 26.9 GiB | **NVMe**, read per token |
| Attention, shared experts, norms | 4.1 GiB | memory |
| MTP head | 1.5 GiB | memory |
| Embeddings + LM head | 2.4 GiB | memory |
| Vision encoder | 0.3 GiB | memory |

The checkpoint's NVFP4 scale factors use a new compression format that
upstream vLLM and SGLang can't load yet. The card requires the community
image `eugr/spark-vllm-b12x`. (The SGLang cookbook's "doesn't fit one
Spark" note is about a different, 126 GiB checkpoint.)

## Memory budget

GB10 has ~119.6 GiB of unified memory (`free -m` total: 122,502 MiB).
CPU and GPU share it, so "CPU offload" saves nothing here; only moving
data to disk does. That's why the n-gram table is read from NVMe
(`VLLM_PLE_TABLE_MEMORY=disk`).

Measured on 2026-10-20 at `GPU_MEM_UTIL` 0.69, vision on:

| Item | GiB |
|---|---|
| Total | 119.6 |
| Linux + services (before vLLM) | 3.3 |
| **vLLM in total** | **83.7** |
| — inside its limit (0.69 × total) | 82.5 |
| —— KV cache pool | 297,194 tokens |
| — outside its limit (CPU side) | ~1.2 |
| **MemAvailable, vLLM only** | **32.6** |
| qwen-image-2.1, idle | 28.0 |
| **MemAvailable, both loaded and idle** | **4.6** |

4.6 GiB is thin: this box has frozen at 2–4 GB free, and qwen-image's
per-request peaks (1–2 GB for a tiled decode, 3–6 GB for an edit) come on
top of its idle 28 GiB. Its admission control and tiled VAE decode check
`MemAvailable`, so large requests should wait or get a 503 rather than
overrun memory. vLLM allocates everything at startup and doesn't grow.
Page cache from n-gram table reads counts as available.

For more margin, without going below NVFP4:

- **FP8 DiT in qwen-image-2.1** (frees ~7 GB, see its README).
- **`LANGUAGE_ONLY=1`**: no vision encoder or the memory reserved for it.
  Not measured yet; expected to free ~1–3 GiB.

| `GPU_MEM_UTIL` | KV cache | KV pool (tokens, measured) |
|---|---|---|
| 0.65 | 0.47 GiB | fails at 131072 |
| 0.68 | | 194,421 |
| **0.69 (default)** | | **297,194** (1.13 × 262K) |
| 0.70 | | 396,313; only 2.3 GiB available with qwen-image loaded |

Above 0.68, each +0.01 adds about 100K tokens.

### Defaults vs. the card's recipe

| | Here | Card | Why |
|---|---|---|---|
| `--gpu-memory-utilization` | 0.69 | 0.8 | Smallest value that fits full 262K context (measured) |
| `--max-num-seqs` | 2 | 8 | Less recurrent state reserved; nixclaw rarely runs more |
| `--max-num-batched-tokens` | 2048 | 4096 | Smaller prefill activations; prefill a bit slower |
| `--limit-mm-per-prompt` | image 2, video 0 | unlimited | vLLM reserves memory for the largest allowed input |
| `--mm-processor-cache-gb` | 0 | 4 (vLLM default) | That cache sits in the same unified memory |
| Chat template / parsers | repo's patched template, `qwen3_xml` | model's own | Same template as the other Qwen3.8 deploys (the model's own is byte-identical to the 27B's) |

Everything else (b12x kernels and env vars, FP8 KV cache, MTP with 4 draft
tokens, prefix caching, `seccomp=unconfined` for io_uring) is the card's.

## Measured speed

2026-10-20, defaults (`GPU_MEM_UTIL` 0.69, MTP with 4 draft tokens),
qwen-image-2.1 loaded but idle. One request at a time, thinking off,
256 output tokens, temperature 0.7. The prompt is Python source code
with a unique prefix so the prefix cache can't help, and the model is
asked to explain it in prose.

| Context | TTFT | Prefill | Decode |
|---|---|---|---|
| ~0 (15 tokens) | 0.17 s | — | 45.7 tok/s |
| 16K | 8.9 s | 1,845 tok/s | 45.4 tok/s |
| 32K | 17.7 s | 1,853 tok/s | 37.2 tok/s |
| 64K | 35.6 s | 1,838 tok/s | 43.6 tok/s |
| 128K | 72.7 s | 1,804 tok/s | 39.2 tok/s |
| 192K | 111.1 s | 1,769 tok/s | 43.8 tok/s |

- **Decode** doesn't slow down with context (sparse attention reads at most
  2,048 earlier tokens). The 37–46 tok/s spread comes from how many MTP
  draft tokens get accepted, which varies with the text.
- **TTFT** grows linearly, about 9 s per 16K tokens of new prompt. The
  prefix cache skips tokens already sent in earlier turns, so follow-up
  turns of a long conversation are much faster than these cold numbers.
- Rerun with `python3 bench.py http://127.0.0.1:8000 16384 32768 65536
  131072 196608` (standard library only). One run per size, so expect a
  few tok/s of noise.

## Prerequisites

- ~100 GB free on the Spark's **local NVMe** for the checkpoint
  (`.cache/huggingface` next to these scripts, or set `HF_HOME_DIR`). Don't
  put it on a network share: the n-gram table is read on every token.
- Docker with the NVIDIA Container Toolkit.
- Port 8000 free (stop the other LLM deploy first).
- `HF_TOKEN` optional; the repo is not gated.

## Deploy

Copy this folder to the Spark, then **start this before qwen-image-2.1**:

```bash
cd vllm-qwen3.8-flash-fp4
./start.sh                 # first run downloads ~96 GiB and compiles: 10–20+ min
../qwen-image-2.1/start.sh # only after vLLM reports ready
```

Order matters. While loading, vLLM uses ~10 GiB more than it does once
running, and it refuses to start if free memory is below its limit.
`start.sh` checks `MemAvailable` ≥ `GPU_MEM_UTIL` × MemTotal + 10 GiB and
exits otherwise (`FORCE=1` skips the check). **Don't restart vLLM while the
image model is running.** Stop the image model first.

Overrides (env vars): `GPU_MEM_UTIL`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`,
`MAX_NUM_BATCHED_TOKENS`, `ENABLE_MTP`, `NUM_SPECULATIVE_TOKENS`,
`LANGUAGE_ONLY` (1 = no vision, saves ~1–3 GiB), `LIMIT_MM_PER_PROMPT`,
`TOOL_CALL_PARSER`, `IMAGE`, `PORT`, `CONTAINER_NAME`, `SERVED_MODEL_NAME`,
`HF_HOME_DIR`, `CACHE_DIR`, `LOAD_HEADROOM_GIB`, `FORCE`.

## Remaining check: both models under load

Done so far: vLLM alone (297K-token KV pool at 0.69) and both models
loaded and idle (4.6 GiB available). Still to do:

1. With both running, start a long LLM request, then run a 2048²
   generation and an edit with a reference image in qwen-image-2.1.
2. Watch the lowest value of `MemAvailable`:
   `while sleep 1; do grep MemAvailable /proc/meminfo; done`.
   Above ~2 GiB, and image requests waiting or getting a 503 instead of
   the box stalling, is fine. If it goes lower, use `GPU_MEM_UTIL=0.68`
   with `MAX_MODEL_LEN=192000` (frees ~1.2 GiB), set `LANGUAGE_ONLY=1`, or
   switch the image model to an FP8 DiT.
3. Record the result here.

A memory watchdog like `../sglang-qwen3.8/watchdog.sh` is worth running
too: a unified-memory exhaustion can take the whole box down.

## Verify

```bash
curl -s http://127.0.0.1:8000/v1/models | jq

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-flash",
       "messages": [{"role": "user", "content": "Say hello in one sentence."}],
       "max_tokens": 64}' | jq
```

Since `vllm.model` is unset in nixclaw's `config/agent.yaml`, it picks up
the new served model name automatically.

## Rollback

```bash
./stop.sh
../vllm-qwen3.8-nvfp4/start.sh   # or whichever deploy ran before
```

## Sources

- [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local): recipe, container, FP4 accuracy
- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): benchmarks vs. the 27B
- [ai-muninn single-Spark benchmark](https://ai-muninn.com/en/blog/qwen38-flash-next-nvfp4-dgx-spark-vllm-recipe): decode/prefill numbers, load-time memory spike
- [SGLang cookbook, DGX Spark notes](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next#dgx-spark-notes-1x-gb10-with-the-n-gram-table-on-nvme-or-2x-gb10-tp-2): why host offload doesn't help on GB10
