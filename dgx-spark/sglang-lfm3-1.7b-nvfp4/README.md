# SGLang deployment for LiquidAI/LFM3-1.7B-NVFP4 (NVIDIA DGX Spark)

`start.sh` serves `LiquidAI/LFM3-1.7B-NVFP4` (ModelOpt NVFP4, W4A4, group 16;
attention, conv, embeddings, lm_head and norms stay BF16) under SGLang on the
Spark's GB10, which has native FP4 tensor cores. OpenAI-compatible `/v1` API on
**port 8010**. NVFP4 is the default weight format for this model on the Spark;
the GGUF alternative is `../llamacpp-lfm3-1.7b`.

## Status: UNVERIFIED on the box

Written without access to the Spark. What was checked offline:

- `LiquidAI/LFM3-1.7B-NVFP4` exists (1.85GB, `hf_quant_config.json` NVFP4).
- Neither vLLM nor SGLang main implements `Lfm3ForCausalLM`. Liquid's model
  card points at [sgl-project/sglang#46858](https://github.com/sgl-project/sglang/pull/46858)
  (open, unmerged). The PR's `python/` diff (27 files, tests dropped) applies
  cleanly to a fresh SGLang `main` checkout; it was **not** tried against the
  `lmsysorg/sglang:dev-cu13` image itself.
- So the deploy builds a thin local image (`nixclaw/sglang-lfm3:local`):
  base image + PR overlay. The `docker build` fails loudly (`git apply --check`)
  if the PR does not apply to the image's SGLang.

## Usage

```bash
cd deploy/dgx-spark/sglang-lfm3-1.7b-nvfp4
./start.sh    # fetches PR diff (pinned in ./lfm3-sglang-pr.diff), builds image, serves
./stop.sh
```

First-run checklist: the build step succeeds; the log shows the model loading
as NVFP4 (if it loads as BF16, set `QUANTIZATION=modelopt_fp4`); then smoke
test chat + a tool call (`reasoning-parser`/`tool-call-parser` are `auto`, per
the model card, which relies on the PR's template detection).

## Port 8010

Requested to differ from the Qwen LLM deploys (8000). Note
`../qwen3-tts` also defaults to **8010** (and nixclaw's TTS config points at
`gx10-811e.lan:8010`). `start.sh` refuses to start when the port is bound.
Run only one of them, or set `PORT=` (e.g. 8012).

## Memory

A 1.7B NVFP4 model is ~2GB of weights. `MEM_FRACTION_STATIC` defaults to 0.15
(not the 0.80 of the big hybrid deploys) so it coexists with the other Spark
services; the container is capped at 32GB. Context 131072 (model max 262144).

## Environment overrides

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_ID` | `LiquidAI/LFM3-1.7B-NVFP4` | `LiquidAI/LFM3-1.7B` for BF16 (same image) |
| `SERVED_MODEL_NAME` | `lfm3-1.7b` | |
| `PORT` | `8010` | |
| `BASE_IMAGE` | `lmsysorg/sglang:dev-cu13` | patched by the PR overlay |
| `IMAGE` | `nixclaw/sglang-lfm3:local` | `REBUILD=1` to rebuild |
| `LFM3_PR` | `46858` | delete `lfm3-sglang-pr.diff` + `REBUILD=1` to refresh |
| `CONTEXT_LENGTH` | `131072` | |
| `MEM_FRACTION_STATIC` | `0.15` | |
| `QUANTIZATION` | empty | e.g. `modelopt_fp4` if auto-detect fails |
| `GPU_DOCKER_ARG` | `--gpus all` | `--device nvidia.com/gpu=all` for CDI-only |

## Retire this overlay

Once SGLang releases LFM3 support, delete `Dockerfile`/`filter_pr_diff.py`,
and run the stock image from the other SGLang deploys.
