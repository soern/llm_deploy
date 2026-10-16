# vllm-gemma4-e2b-dgx-spark

vLLM launch config for **`google/gemma-4-E2B-it`** on the DGX Spark (GB10),
meant to run **next to** the main Qwen3.8 deploy (port 8000) as a small,
fast vision model (`vllm.vision_model` in `config/agent.yaml`) and optionally
a second-opinion STT backend. Not a replacement for Qwen as the agent/coding
model: with FP8 / NVFP4 it decodes faster than Qwen3.8 + DFlash2 (69 / 81
vs ~45 tok/s, see "Measured speed") but is much weaker at code (LiveCodeBench v6 44%).

## Model

[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it):
5.1B params total / 2.3B effective (per-layer embeddings), 35 layers, 1 KV
head, 128K context, BF16. Text + image (~150M encoder, 70-1120 tokens per
image, default 280) + audio (~300M encoder, ASR / speech translation, max
30s per clip).

## Usage

```bash
./start.sh                                   # BF16, 128K, vision + audio on, MTP 2 draft tokens
NUM_SPECULATIVE_TOKENS=0 ./start.sh          # MTP off (all "Measured speed" numbers below)
WEIGHTS=fp8 ./start.sh                       # online FP8 quantization
WEIGHTS=w4a16 ./start.sh                     # Google's QAT int4 checkpoint
WEIGHTS=nvfp4 ./start.sh                     # unsloth's NVFP4 checkpoint
ENABLE_AUDIO=0 ./start.sh                    # vision only
ENABLE_VISION=0 ENABLE_AUDIO=0 ./start.sh    # text only (--language-model-only)
./stop.sh
```

| Variable | Default | Notes |
|---|---|---|
| `WEIGHTS` | `bf16` | `bf16`, `fp8` (vLLM `--quantization fp8` at load time, same checkpoint), `w4a16` or `nvfp4` (separate checkpoints, see "4-bit variants") |
| `MODEL_ID` | per `WEIGHTS` | override the checkpoint |
| `MAX_MODEL_LEN` | `131072` | model maximum |
| `NUM_SPECULATIVE_TOKENS` | `2` | MTP draft tokens per step, `0` = off; see "MTP drafter" |
| `DRAFT_MODEL_ID` | `google/gemma-4-E2B-it-assistant` | MTP drafter |
| `ENABLE_VISION` / `ENABLE_AUDIO` | `1` | `0` sets that modality's per-prompt limit to 0; both `0` = `--language-model-only` |
| `MAX_IMAGES_PER_PROMPT` / `MAX_AUDIO_PER_PROMPT` | `4` / `1` | |
| `GPU_MEM_UTIL` | `0.11` (bf16) / `0.095` (fp8) / `0.105` (w4a16) / `0.08` (nvfp4) | fraction of the **whole** 119.63 GiB unified memory (~13.2 / ~11.4 GiB); must not exceed free memory at launch — `start.sh` checks first |
| `PORT` | `8004` | |
| `MAX_NUM_SEQS` | `4` | |
| `OOM_SCORE_ADJ` | `300` | killed after qwen3-tts (+800) and qwen-image (+500), before Qwen3.8 (-900) |
| `IMAGE` | `eugr/spark-vllm:latest` | same image as the Qwen3.8 vLLM deploys |
| `TOOL_CALL_PARSER` | `gemma4` | passed with `--enable-auto-tool-choice`; empty disables it, and vLLM then rejects every request carrying `tools` with HTTP 400 |

## Memory

Estimates (memory not measured on this box):

| | BF16 | FP8 | W4A16 | NVFP4 |
|---|---|---|---|---|
| Weights | ~10.5 GB | ~8.5 GB | 8.3 GB | 5.5 GB |
| KV cache at 128K | ~0.8 GB (3 of 35 layers hold full-context KV) | same | same | same |
| Total incl. CUDA graphs / activations | ~15 GB | ~13 GB | didn't start, see below | 0.08 = 9.6 GiB budget, works (KV room for ~780K tokens) |

Why 4-bit saves so little: of the 5.1B parameters, 2.35B are the per-layer
embedding (PLE) table, 0.4B the embedding / tied lm_head, 0.48B the
vision + audio encoders, and only 1.85B the language model's Linear layers.
FP8 and W4A16 quantize only those Linear layers; the NVFP4 checkpoint also
stores the PLE table as int8.

Disabling vision and/or audio saves at most ~1.5-2 GB and does not change
text decode speed.

vLLM refuses to start if `GPU_MEM_UTIL` x total exceeds CUDA's free memory,
and on the Spark CUDA does not count the page cache as free (model downloads
and the weight load itself fill it). `start.sh` checks `MemFree` vs
`MemAvailable` first; if only the page cache is in the way it stops and
prints `sync; echo 1 | sudo tee /proc/sys/vm/drop_caches`. First BF16 launch
at 0.13 failed this way (15.55 GiB wanted, 15.3 GiB free).

The other failure comes after loading: vLLM needs ~0.94 GiB of KV cache for
one 128K request and stops ("estimated maximum model length is ...") if the
budget leaves less. w4a16 at 0.095 left 0.06 GiB, with 2.26 GiB of
non-torch memory (CUDA context, kernels' workspaces) on top of the weights;
at 0.105 it froze the box and nearly hit OOM next to Qwen3.8. w4a16 is not
usable on this box as deployed today; nvfp4 runs fine at 0.08. Fix for that error: raise `GPU_MEM_UTIL` by 0.01 or
lower `MAX_MODEL_LEN`.

## Measured speed (2026-10-09)

All without MTP (before the drafter was added; `NUM_SPECULATIVE_TOKENS=0`
reproduces them).

`bench.py` (streaming chat completions, unique prefix per request so the
prefix cache never hits, 256 output tokens forced via `min_tokens` +
`ignore_eos`, temperature 0, 2 runs per size; prompt sizes from the server's
usage report):

```bash
python3 bench.py http://sparky.lan:8004 gemma-4-e2b 1000,16000,100000
```

Single request:

| Prompt tokens | BF16 TTFT | BF16 decode | FP8 TTFT | FP8 decode | NVFP4 TTFT | NVFP4 decode |
|---|---|---|---|---|---|---|
| ~1.3K | 0.25-0.31 s | 16.5-17 tok/s | 0.09-0.13 s | 69 tok/s | 0.08 s | 80-81 tok/s |
| ~21K | 6.0-7.2 s | 15.3 tok/s | 2.4 s | 60 tok/s | 2.6 s | 73-74 tok/s |
| ~130K (full context) | 132-133 s | 12-12.5 tok/s | 56 s | 42 tok/s | 78-79 s (confirmed in a second pass) | 55.6-56.7 tok/s |

Prefill: BF16 ~5K / ~3.2K / ~975 tok/s, FP8 ~12K / ~8.7K / ~2.3K tok/s,
NVFP4 ~16K / ~8.1K / ~1.6K tok/s at the three sizes.

Concurrent requests (~1.3K-token prompts, decode tok/s per request /
aggregate):

| Concurrency | BF16 | FP8 | NVFP4 |
|---|---|---|---|
| 1 | 16.2 / 16 | 63 / 63 | 81 / 81 |
| 2 | 18.2 / 36 | 72 / 145 | 75 / 150 |
| 4 | 17.1 / 68 | 70 / 280 | 82 / 326 |

NVFP4 (2026-10-09, `unsloth/gemma-4-E2B-it-NVFP4`, vLLM
0.30.1rc1 in `eugr/spark-vllm`, `GPU_MEM_UTIL=0.08`) was measured with
Qwen3.8 idle (polled `/get_load` every 2 s during each request). The
first full-context NVFP4 pass overlapped a ~43K-token Qwen request: TTFT
111-169 s, decode 21-22 tok/s. One concurrency pass with Qwen idle but
something else on the GPU (FLUX on 8002 and the 8010 service were up) gave
~32 tok/s per request at 1/2/4 requests, the same fixed-step pattern as the
BF16 run. So any other GPU process, not only Qwen, slows this server down.

vLLM's own log line ("Avg prompt throughput: 12968 tokens/s") is not the
prefill speed: it counts a request's whole prompt in the 10 s interval in
which its prefill finishes (129.7K / 10 s = 12968), and shows 0 during the
78 s the prefill actually runs. Real full-context prefill is ~1.65K tok/s.

NVFP4 vs FP8: decode +17% at 1.3K, +22% at 21K, +35% at full context;
prefill faster on short prompts, slower at full context (TTFT 79 s vs 56 s).

**The BF16 and FP8 runs are not directly comparable.** During the BF16 run
Qwen3.8 (SGLang, port 8000) was serving one ~36K-token request; during the
FP8 run it was idle (`/get_load`). FP8 alone can't explain the 4x decode gain
(it only halves the Linear weights, so ~1.5x at best). The BF16 pattern - a
fixed ~60 ms per decode step and aggregate throughput scaling linearly with
concurrency - points to the two processes time-slicing the GPU rather than
to memory bandwidth. Expected consequence: whenever Qwen3.8 is generating,
this server runs far below the FP8 numbers above (and slows Qwen too).
Not yet verified: BF16 with Qwen idle, FP8 with Qwen busy, and whether
NVIDIA MPS (concurrent GPU sharing between processes) helps.

For comparison: Qwen3.8-27B + DFlash2 decodes ~45 tok/s on this box
(`../sglang-qwen3.8/BENCHMARK_REPORT.md`); community numbers for
Gemma 4 E2B on GB10 are 53 tok/s (Ollama) and 83 tok/s (llama.cpp).

vLLM logs `FA4 not available, forcing TRITON_ATTN backend` for this model
(heterogeneous head dims 256/512).

## MTP drafter

The Gemma checkpoints have no built-in MTP layer, but Google publishes a
separate drafter per Gemma 4 size,
[`google/gemma-4-E2B-it-assistant`](https://huggingface.co/google/gemma-4-E2B-it-assistant)
(158 MB BF16, 4 layers, hidden size 256), which vLLM loads via
`--speculative-config '{"model": ..., "num_speculative_tokens": 2}'`. It
reuses the target's KV cache (no extra KV memory) and computes logits for
~4K centroid-selected candidates instead of the full 262K vocabulary
("centroids masking", automatic for E2B/E4B), so a draft token costs a few
percent of a target step. That makes it a much better fit than Qwen3.5-4B's
MTP head, which re-reads a 1.27 GB lm_head per draft token.

2 draft tokens = vLLM's Gemma 4 recipe value for E2B (benchmarked on
A100/H100; E4B and larger use 4+, since fast targets gain less per accepted
token than the draft costs). Google claims up to 3x. Estimate for the Spark,
not measured: ~1.6-1.8x if acceptance is good, realistically 1.3-1.5x
given vLLM's fixed per-step speculative overhead at ~12 ms per token. Also
expect lower acceptance with `nvfp4` (the drafter was trained against the
BF16 model), and little gain while another process uses the GPU.

`bench.py` reports the acceptance rate per draft token (from `/metrics`;
mean accepted length = 1 + rate x k). The default `story` prompt is random
text and keeps the old numbers comparable; `code` is closer to agent work:

```bash
for k in 0 1 2 3; do
  ./stop.sh; WEIGHTS=nvfp4 NUM_SPECULATIVE_TOKENS=$k ./start.sh
  python3 bench.py http://sparky.lan:8004 gemma-4-e2b 1000,16000 code
done
```

Alternative (not wired in): the community DFlash2 drafter
[`rasyosef/gemma-4-E2B-it-dflash2`](https://github.com/vllm-project/speculators/issues/1169)
(0.32B, 7 tokens per block) reported 3.1-3.7x on an A100, but lower
acceptance than Google's drafter on 8 of 9 benchmark subsets.

## 4-bit variants

Of the many 4-bit checkpoints on Hugging Face, these two fit vLLM on the
Spark best (both compressed-tensors, both keep vision + audio in BF16):

| `WEIGHTS` | Checkpoint | Format | Why |
|---|---|---|---|
| `w4a16` | [`google/gemma-4-E2B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct) | int4 weights (group 32), BF16 activations | Official quantization-aware training (QAT): the model was trained for int4, so quality should stay closest to BF16. Should be the best 4-bit quality, but failed to start on the Spark (see Memory). |
| `nvfp4` | [`unsloth/gemma-4-E2B-it-NVFP4`](https://huggingface.co/unsloth/gemma-4-E2B-it-NVFP4) | mixed: NVFP4 W4A4 MLPs, FP8 attention + 12 sensitive MLP layers, int8 PLE | Smallest, and the only one using GB10's FP4 tensor cores (faster prefill). Post-training quantization of a small model with 4-bit activations, no published accuracy numbers. unsloth requires vLLM >= 0.25 and warns that the Marlin fallback is ~2x slower. |

Not chosen: other NVFP4 checkpoints (`bg-digitalservices/*` keep the PLE in
BF16 at the same 7.5 GB with no sensitive layers spared;
`yasu-oh/gemma-4-E2B-it-qat-NVFP4` puts the int4-trained QAT weights onto
the FP4 grid, which doesn't match what they were trained for; almost
nobody uses either), `cyankiwi/*-AWQ-INT4` (plain post-training quantization,
worse than QAT at the same format), `google/*-qat-mobile-ct` (2-bit/int8
mobile scheme), and GGUF (llama.cpp, a different server; community numbers
put it at 83 tok/s on GB10, see `google/gemma-4-E2B-it-qat-q4_0-gguf`).

Expected decode speed (estimate, not measured): each token reads the
quantized Linear layers + the BF16 lm_head (0.8 GB); the PLE table is only
a lookup. That's ~2.7 GB per token for FP8 and ~1.9 GB for W4A16 / all-FP4,
so at most ~1.4x faster than FP8 (~69 -> ~95 tok/s with Qwen idle). The
mixed NVFP4 checkpoint reads more (FP8 attention), so ~1.2x; measured
1.17x at 1.3K prompt tokens (81 vs 69 tok/s, see "Measured speed"). Prefill: NVFP4
should be fastest (FP4 tensor cores, if vLLM picks a native FP4 kernel on
sm_121 rather than Marlin; check the startup log), W4A16 likely slower than
FP8 because it computes in BF16.

## Unverified

- Audio input end-to-end (the server starts with audio enabled; no
  transcription request tested yet).
- Image requests: time for a real `analyze_image` call (expected from the
  numbers above: ~0.1 s TTFT at the default 280 image tokens, ~2-3 s for a
  150-token description with FP8 and Qwen idle).
- Whether `--limit-mm-per-prompt` 0 also skips loading that encoder's weights
  (it does skip its profiling buffers); `--language-model-only` does skip them.
- Actual memory footprint (estimates above).
- MTP drafter: startup with each `WEIGHTS` value, memory, acceptance and
  speedup on the Spark; whether `GPU_MEM_UTIL` defaults still leave room
  for one 128K request.
- `nvfp4`: answer quality vs BF16/FP8 (only speed measured), and which FP4
  kernel vLLM picked on sm_121 (check the startup log for Marlin).
