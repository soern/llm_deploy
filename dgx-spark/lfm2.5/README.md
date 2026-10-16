# SGLang deployment for LiquidAI/LFM2.5-2.6B on NVIDIA DGX Spark (GB10, sm_121a)

`start.sh` runs `LiquidAI/LFM2.5-2.6B` under the official SGLang dev image,
same plain-`docker run` pattern as the sibling
`../sglang-qwen3.8` and `../sglang-minicpm5-2b` deploys.

## Model

- **Checkpoint**: `LiquidAI/LFM2.5-2.6B` (~5.4GB, native BF16). Revision
  `654f9463ce32b05d0429d76fe1f580b27d4c1ac0` is recorded in `start.sh` for
  provenance only — the image's `--model-path` does not accept
  `repo@revision`, so the live pull tracks HF main.
- **Architecture**: `Lfm2ForCausalLM`, `model_type: lfm2` — a **dense hybrid**:
  32 layers, mostly gated causal-conv blocks plus full-attention layers
  (`layer_types` = 24× `conv` + 8× `full_attention`), hidden 2048, 131k
  `max_position_embeddings`. SGLang's `lfm2.py` runs the conv blocks through
  its Mamba/conv state cache (MambaPool), so it is a hybrid-mamba-style model
  like qwen3.8 on this box.
  **Not** the `LiquidAI/LFM2.5-8B-A1B` MoE — that is a separate, larger
  checkpoint.
- **Chat template**: the checkpoint ships its own `chat_template.jinja`
  (Qwen-style `...` thinking, `<|tool_call_start|>[` / `<|tool_call_end|>]`
  tool calls). No template mount — SGLang loads it from the model config.
- **Gating**: none; `HF_TOKEN` is not required (still passed through if set).

## Flag sourcing — read this before trusting the numbers

**No SGLang cookbook cell verifies LFM2.5-2.6B on GB10.** The official
cookbook (docs.sglang.io/cookbook/autoregressive/LiquidAI/LFM2.5, source
`docs/src/snippets/configs/LiquidAI/lfm2.5.jsx`) covers h100/h200/b200 and
its variant list does not include 2.6B. The launch flags are therefore the
union of:

1. **Verified LFM2.5 dense-text cookbook cells** (h100/h200 rows):
   `--trust-remote-code`, `--tp 1`, `--reasoning-parser qwen3` (thinking
   variants), `--tool-call-parser lfm2` (all LFM2.5 variants). These are
   hardware-independent model-side flags.
2. **The machine-validated qwen3.8 recipe on this exact box** (sibling
   `../sglang-qwen3.8` deploy, which runs the same kind of hybrid
   conv/mamba model on the same GB10 unified memory):
   `--memory 100g --memory-swap 100g` cgroup cap,
   `--mem-fraction-static 0.80`, `--attention-backend flashinfer`.

   The cookbook pins no `--mem-fraction-static` for the LFM2.5 text
   variants on h100/h200 (dense cells rely on the default; the vl-450m
   cell uses 0.8), and its b200 dense cells use `trtllm_mha` — an
   unavailable backend on sm_121a — so 0.80/flashinfer come from the
   sibling recipe, not the cookbook.

Everything above is **UNVERIFIED for this model on this hardware** until a
first run. If boot OOMs, lower `MEM_FRACTION_STATIC` first (the 2.6B
weights + 655MB draft need only ~6GB of the static pool).

## Flags

| Flag | Value | Source |
|------|-------|--------|
| `--model-path` | `LiquidAI/LFM2.5-2.6B` | Bare repo id — `repo@revision` is rejected by the image (HFValidationError); revision `654f9463…` is provenance only |
| `--trust-remote-code` | (set) | cookbook: in every verified LFM2.5 cell |
| `--tp` | `1` | cookbook: every LFM2.5 variant verified at TP=1 |
| `--served-model-name` | `lfm2.5-2.6b` | local naming |
| `--mem-fraction-static` | `0.80` | qwen3.8 machine-validated value; do NOT raise to 0.85 (qwen3.8 hit earlyoom SIGTERMs there) |
| `--attention-backend` | `flashinfer` | qwen3.8 machine-validated choice on GB10 |
| `--reasoning-parser` | `qwen3` | template is Qwen-style `...`; matches cookbook reasoning variants |
| `--tool-call-parser` | `lfm2` | SGLang `FunctionCallParser.ToolCallParserEnum["lfm2"]`; all LFM2.5 cells |
| `--host` / `--port` | `0.0.0.0` / `8000` | local convention |

### DSpark (on by default)

`LiquidAI/LFM2.5-2.6B-DSpark` (revision `458cedab07d0f7b2b05700c77e1aa463d43d6f04`,
655MB, BF16) is LiquidAI's matching draft checkpoint — a 5-layer Qwen3-style
`Lfm2DSparkDraftModel` with `block_size: 9`. SGLang ships native support
(`lfm2_dspark.py`).

| Flag | Value | Note |
|------|-------|------|
| `--speculative-algorithm` | `DSPARK` | |
| `--speculative-draft-model-path` | `LiquidAI/LFM2.5-2.6B-DSpark` | bare repo id — `repo@revision` is rejected by the image (HFValidationError) |
| `--speculative-dspark-block-size` | `9` | **must match the draft config's `block_size`** (qwen3.8's draft used 7 — do not copy that) |
| `--speculative-draft-model-quantization` | `unquant` | draft is BF16 |

Set `ENABLE_DSPARK=0` to run the base model alone. No acceptance-rate or
speedup numbers are published for this draft; the default is on because
LFM2.5-DSpark is the model family's own recommended overlay.

## Docker

- **Image**: `lmsysorg/sglang:dev-cu13` — the cookbook-mandated image for
  LFM2.5 ("LFM2.5 support … ships in the SGLang dev image; not yet in a
  tagged release"). The sibling minicpm5 deploy uses plain `:dev`; override
  `IMAGE=` to switch.
- **GPU**: `--gpus all` (CDI-only override documented in `start.sh`)
- **cgroup cap**: `--memory 100g --memory-swap 100g` (qwen3.8 recipe; this
  cap is what gives the box its dedicated-VRAM-style memory accounting)
- **shm**: `--shm-size 16g`; **network**: host; **ipc**: host
- Caches live under `./.cache/` in this directory (HF model cache + runtime)

## Environment overrides

All knobs in `start.sh` are `VAR="${VAR:-default}"`:

`MODEL_ID`, `MODEL_REV`, `DRAFT_MODEL_ID`, `DRAFT_REV` (provenance only —
the revs are not passed to `--model-path`, which rejects `@`),
`DSPARK_BLOCK_SIZE`, `ENABLE_DSPARK`, `SERVED_MODEL_NAME`, `IMAGE`,
`CONTAINER_NAME`, `HOST`, `PORT`, `MEM_FRACTION_STATIC`, `GPU_DOCKER_ARG`,
`HF_TOKEN`.

## Usage

```bash
./start.sh                    # boot + wait for /health
docker logs -f sglang-lfm2.5-2.6b   # live logs (also mirrored to .sglang.log)
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm2.5-2.6b","messages":[{"role":"user","content":"Hello"}]}'
./stop.sh
```

nixclaw: set `config.backend` to `vllm` in `config/agent.yaml` and point
`vllm.base_url` at `http://localhost:8000` — SGLang serves the same
OpenAI-compatible `/v1` API (same as the sibling deploys).

## Files

- `start.sh` — launch (idempotent: reuses a running container)
- `stop.sh` — stop the container
