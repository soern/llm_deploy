# vllm-qwen3.5-4b-dgx-spark

vLLM launch config for **`Qwen/Qwen3.5-4B`** (text-only) with MTP
speculative decoding on the DGX Spark (GB10), meant to run **next to** the
main Qwen3.8 deploy (port 8000) as a small, fast coding / tool-calling
model. Alternative to `../vllm-gemma4-e2b` when images and audio
aren't needed: clearly stronger at code and tool use, similar memory, decode
speed expected in the same range only with MTP (not measured yet).

## Model

[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B): 4B dense,
Apache 2.0, 262K native context (served at 128K). 32 layers: 24 Gated
DeltaNet (linear attention, fixed-size state per request) + 8 full-attention
layers (4 KV heads x 256), plus one built-in MTP layer for speculative
decoding. 248K vocab with a tied embedding / lm_head (1.27 GB BF16). The
checkpoint is multimodal; `--language-model-only` skips the vision encoder.
Thinks before answering by default.

| | Gemma 4 E2B | Qwen3.5-4B |
|---|---|---|
| LiveCodeBench v6 | 44.0 | 55.8 |
| BFCL v4 (function calling) | - | 50.3 |
| τ²-Bench (agentic tool use) | lower than E4B's 42.2 | 79.9 |

## Usage

```bash
./start.sh                                   # FP8, MTP with 2 draft tokens, thinking on
WEIGHTS=nvfp4 ./start.sh                     # AxionML NVFP4 checkpoint
NUM_SPECULATIVE_TOKENS=1 ./start.sh          # MTP depth; 0 = MTP off
THINKING=0 ./start.sh                        # answer directly by default
./stop.sh
```

| Variable | Default | Notes |
|---|---|---|
| `WEIGHTS` | `fp8` | `fp8` (official BF16 checkpoint + vLLM `--quantization fp8` at load time) or `nvfp4` (`AxionML/Qwen3.5-4B-NVFP4`) |
| `MODEL_ID` | per `WEIGHTS` | override the checkpoint |
| `NUM_SPECULATIVE_TOKENS` | `2` | MTP draft tokens per step, `0` = off; see "MTP depth" |
| `THINKING` | `1` | `0` adds `--default-chat-template-kwargs '{"enable_thinking":false}'`; per-request `chat_template_kwargs` still wins |
| `MAX_MODEL_LEN` | `131072` | model maximum is 262144 |
| `KV_CACHE_DTYPE` | `auto` (BF16) | `fp8` halves the KV cache, untested here |
| `GPU_MEM_UTIL` | `0.12` (fp8) / `0.112` (nvfp4) | fraction of the **whole** 119.63 GiB unified memory (14.4 / 13.4 GiB); `start.sh` checks free memory first |
| `TOOL_CALL_PARSER` | `qwen3_coder` | empty = no tool calling |
| `PORT` | `8006` | |
| `MAX_NUM_SEQS` | `4` | |
| `OOM_SCORE_ADJ` | `300` | killed after qwen3-tts (+800) and qwen-image (+500), before Qwen3.8 (-900) |
| `IMAGE` | `eugr/spark-vllm:latest` | same image as the other vLLM deploys |

Sampling recommended by Qwen (set by the client, not the server):
thinking + coding `temperature=0.6, top_p=0.95, top_k=20,
presence_penalty=0`; non-thinking `temperature=0.7, top_p=0.8, top_k=20,
presence_penalty=1.5`.

## Checkpoints

Read from the safetensors headers (2026-10-10):

| Checkpoint | Linear layers | MTP head | lm_head | Size | Used |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-4B` + online FP8 | FP8 after loading | yes | tied, BF16 | 9.3 GB download | `fp8` |
| `AxionML/Qwen3.5-4B-NVFP4` (ModelOpt) | NVFP4 W4A4 (2.0 GB) | yes, BF16 | tied, BF16 | 4.2 GB | `nvfp4` |
| `cosmicproc/Qwen3.5-4B-NVFP4` | same as AxionML | **no** | tied, BF16 | 3.3 GB | no MTP |
| `Hob-forge/Qwen3.5-4B-FP8` | block FP8, official format | yes, FP8 | tied, BF16 | 5.7 GB | alternative via `MODEL_ID`, few users |
| `RedHatAI/Qwen3.5-4B-FP8-dynamic` | FP8 MLPs, **BF16 DeltaNet** | yes (separate file) | untied BF16 copy | 8.0 GB | reads ~1 GB more per token |
| `Shashwat42/Qwen3.5-4B-FP8` | FP8 | **no** | tied, BF16 | 5.5 GB | no MTP |

Qwen publishes official FP8 checkpoints only for 27B and larger.

## Memory

Estimates, not measured on this box:

| | FP8 | NVFP4 |
|---|---|---|
| Weights incl. MTP, without vision | ~5.3 GiB | ~3.5 GiB |
| KV cache, one 128K request (8 attention layers + MTP layer, BF16) | ~4.7 GiB | same |
| DeltaNet state | ~50 MB per request | same |
| Non-torch (CUDA context, workspaces) + activations | ~3.8 GiB | same |
| `GPU_MEM_UTIL` default | 0.12 = 14.4 GiB, starts | 0.112 = 13.4 GiB (0.105 left 4.31 of the 4.73 GiB KV needed) |

More than Gemma 4 E2B's 9.6 GiB (nvfp4) because its KV cache is ~5x
larger per token (Gemma keeps full-context KV in only 3 layers with 1 KV
head). `KV_CACHE_DTYPE=fp8` or a lower `MAX_MODEL_LEN` brings it closer.

Online FP8 loads the BF16 weights (~8.6 GB without vision) before
quantizing them, so the load needs that much headroom. The same two
failures as the Gemma deploy apply: vLLM refuses to start if
`GPU_MEM_UTIL` x total exceeds CUDA's free memory (page cache counts as
used; `start.sh` checks and prints the `drop_caches` command), and stops
after loading if one 128K request's KV cache doesn't fit ("estimated
maximum model length is ..."): raise `GPU_MEM_UTIL` by 0.01, lower
`MAX_MODEL_LEN`, or set `KV_CACHE_DTYPE=fp8`.

## MTP depth

There are no published MTP numbers for the 4B on the Spark, so the default
of 2 is an estimate from the data below.

Decode on the Spark is memory-bound; the Gemma FP8 run implies ~185 GB/s
effective (2.7 GB read per token at 69 tok/s). Per decode step:

- Full model: ~4.8 GB (FP8) / ~3.3 GB (NVFP4) -> ~38 / ~56 tok/s without MTP.
- Each draft token: the MTP layer (0.24 GB) + the shared lm_head (1.27 GB)
  = ~1.5 GB, i.e. ~30% of an FP8 step, ~45% of an NVFP4 step. The large
  vocabulary makes drafting expensive on a model this small.
- Verifying k draft tokens costs about the same as decoding one (memory-bound).

Acceptance data from other Qwen3.5 models (vLLM, greedy or near-greedy):

| Source | Per-position acceptance |
|---|---|
| Qwen3.5-35B-A3B resources repo, MTP=2 (GPU not stated) | 88-96% / 72-89% (higher on code) |
| Unattributed report found via web search, MTP=5 (source not verified) | 85% / 59% / 49% / 44% / 41% |
| vLLM issue #47277 (Qwen3.5 0.8B, H200) | MTP=1: 82-88% |

With the second (lower) curve, expected speedup = tokens per step / step
cost:

| k | Tokens per step | FP8 speedup | NVFP4 speedup |
|---|---|---|---|
| 1 | 1.85 | 1.41x (~54 tok/s) | 1.27x (~71) |
| 2 | 2.34 | 1.45x (~55) | 1.23x (~69) |
| 3 | 2.59 | 1.34x | 1.10x |

With the higher curve from the first row, k=2 wins clearly for both (~1.7-1.9x FP8).
Hence k=2 as the default, k=1 the likely alternative for NVFP4, and k>=3
not worth it.

Caveats:

- vLLM's speculative path has a fixed per-step overhead (rejection
  sampling, metadata). vLLM issue #47277 measured MTP=1 at 0.86x and MTP=2
  at 0.54x of no-MTP on a 0.8B Qwen3.5 at high batch on an H200; small
  models with fast steps are where this overhead hurts most. So also measure
  `NUM_SPECULATIVE_TOKENS=0`.
- Drafts cost KV cache and batch slots, so MTP lowers aggregate
  throughput under concurrent load (vLLM's Qwen3.5 recipe).
- Acceptance drops at higher temperature and on free prose; code and JSON
  (tool calls) accept best.

Measure with `bench.py`, which also reports the acceptance rate per draft
token (from `/metrics`; mean accepted length = 1 + rate x k):

```bash
for k in 0 1 2 3; do
  ./stop.sh; NUM_SPECULATIVE_TOKENS=$k ./start.sh
  python3 bench.py http://sparky.lan:8006 qwen3.5-4b 1000,16000 code
done
```

Measure with Qwen3.8 and every other GPU process idle: on this box any
second GPU user cuts decode to a fixed-step pattern (see the Gemma README).

## Not verified yet

- Startup in `eugr/spark-vllm` (Qwen3.5 needs vLLM >= 0.16; the image runs
  0.30.x), memory, speed, MTP acceptance: nothing has been run yet.
- Whether vLLM's online FP8 also quantizes the MTP layer.
- Which FP4 kernel vLLM picks on sm_121 for the ModelOpt checkpoint
  (check the startup log for Marlin).
- `KV_CACHE_DTYPE=fp8` quality.
- Tool calling end to end with nixclaw (`qwen3_coder` parser, thinking on).

## Sources

- [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (benchmarks, sampling, architecture)
- [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html) (parsers, MTP, `--language-model-only`)
- [AxionML/Qwen3.5-4B-NVFP4](https://huggingface.co/AxionML/Qwen3.5-4B-NVFP4)
- [MTP=2 on Qwen3.5-35B-A3B](https://github.com/patrickbdevaney/qwen-3.5-35b-a3b-vllm-resources/blob/main/MTP_SPECULATIVE.md)
- [vLLM issue #47277](https://github.com/vllm-project/vllm/issues/47277) (MTP slower than no-MTP on a small Qwen3.5)
