# flux2-klein-dgx-spark

Image-generation/editing HTTP service backed by
[FLUX.2 \[klein\] 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B),
running on the same DGX Spark (GB10, ARM64, sm_121a, 128 GB unified memory)
box that already hosts vLLM (`deploy/dgx-spark/vllm-qwen3.6/`). Used by nixclaw's
planned `generate_image` tool to turn a Signal prompt (or a photo + edit
instruction) into a returned image attachment.

Pipeline note: load the model with diffusers' `Flux2KleinPipeline` (Qwen
text encoder), **not** `Flux2Pipeline` (Mistral text encoder) — the wrong
class loads fine but fails generation with
`can only concatenate str (not "list") to str`. The diffusers install is
pinned to git main because FLUX.2 support is not yet in a release.

## Why FLUX.2 [klein] 9B

- Apache 2.0 licensed.
- Distilled: only ~4 inference steps needed for good output, keeping
  latency reasonable on a shared box.
- Native image-editing path (no separate inpainting/edit model needed) —
  covers both "generate" and "edit an attached photo" with one model.

## Endpoints

- `GET  /health` — liveness + model id / quant mode / resident VRAM
- `GET  /v1/models` — OpenAI-compatible model list (one entry)
- `POST /v1/images/generations` — OpenAI-shaped JSON body:
  `{"prompt": str, "size": "1024x1024", "n": 1, "response_format": "b64_json", "steps": 4, "seed": 42}`
  (`steps` and `seed` are non-standard extras; `response_format` must be `b64_json`)
- `POST /v1/images/edits` — OpenAI-shaped multipart: `image` file + `prompt`
  form fields, plus the same optional `size`/`n`/`steps`/`seed` extras

## Build & run

Copy this whole `flux2-klein-dgx-spark/` folder to the DGX Spark box, then either:

```bash
# Plain docker (mirrors deploy/dgx-spark/vllm-qwen3.6/start.sh):
cd flux2-klein-dgx-spark
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"   # if the model repo is gated
./start.sh          # builds the image, downloads weights, waits for readiness
```

```bash
# Or docker compose. QUANT_MODE=none until the torchao MXFP8 kernel gap
# in the Quantization section is resolved upstream (see note below):
HF_TOKEN=hf_xxx QUANT_MODE=none \
docker compose -f deploy/dgx-spark/flux2-klein/docker-compose.yml up -d --build
```

`start.sh` builds the image from the local Dockerfile (there is no public
image for this stack — it needs `nvcr.io/nvidia/pytorch:25.11-py3` as a
base for ARM64 + CUDA 13 support), streams container logs to `.flux.log`,
and polls `http://127.0.0.1:8002/health` until ready. First run downloads
~40 GB of weights and JIT-quantizes the transformer, so expect this to take
a while — the healthcheck `--start-period` and the poll loop are both sized
generously for that.

## Verify

```bash
curl -s localhost:8002/health | jq
curl -s localhost:8002/v1/models | jq

# Generation:
curl -s localhost:8002/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red bicycle leaning against a brick wall", "size": "1024x1024", "steps": 4}' \
  | jq -r '.data[0].b64_json' | base64 -d > /tmp/out.png

# Editing:
curl -s localhost:8002/v1/images/edits \
  -F image=@/tmp/out.png \
  -F prompt="make the bicycle blue" \
  | jq -r '.data[0].b64_json' | base64 -d > /tmp/edited.png
```

## Quantization

`QUANT_MODE` (env var, default `mxfp8`) picks how the transformer is
quantized **at load time** (not from a pre-quantized checkpoint —
pre-quantized FP4 checkpoints have known SM121A incompatibilities):

| `QUANT_MODE` | torchao config | Trade-off |
|---|---|---|
| `nvfp4` | `NVFP4DynamicActivationNVFP4WeightConfig` | Smallest memory footprint, but on FLUX.1-dev measured mean LPIPS 0.44 (visible quality drift) |
| `mxfp8` | `MXDynamicActivationMXWeightConfig` | ~5 GB more resident memory than nvfp4, but LPIPS 0.11 (near-indistinguishable) — **preferred if the memory budget allows** |
| `none` | no quantization | Escape hatch if the torchao nightly aarch64 wheels misbehave on this box; largest footprint |

The NVFP4/MXFP8 inference-workflow ops need nightly `torch`/`torchao`
cu130 aarch64 wheels (baked into the `nvcr.io/nvidia/pytorch:25.11-py3`
base plus the `torchao`/`mslk` pip installs in the Dockerfile) — aarch64
wheel availability for this exact combination is the main build risk, and
`QUANT_MODE=none` is there specifically so the service still comes up if
that toolchain doesn't cooperate.

Note: the NVCR base image bundles an older `torchao` (<0.15) that lacks
`MXDynamicActivationMXWeightConfig`, so the Dockerfile pins
`torchao==0.15.0` exactly (a bare `pip install torchao` is a no-op against
the bundled copy) and asserts the config classes import at build time.
Exact pin because the base image's torch (`2.10.0a0` nv25.11) is too old
for 0.16+: 0.16 imports `ScalingType`/`SwizzleType` (torch >= 2.11 APIs)
behind a `torch_version_at_least("2.10.0")` guard that the `2.10.0a0`
pre-release satisfies but the API still does not exist in; 0.17/0.18
import them unconditionally. The "Skipping import of cpp extensions due
to incompatible torch version" warning at startup is a soft torchao
fallback, not a failure.

**Known runtime gap (as of 2026-08): `QUANT_MODE=mxfp8` quantizes and loads
fine, but generation fails with `MXTensor dispatch: attempting to run
unimplemented operator ... aten.expand` — the transformer forward pass
expands quantized tensors, and `aten.expand` is not implemented for
MXTensor in torchao 0.15 through 0.18 (and 0.17+ can't be imported on
this torch at all). Run with `QUANT_MODE=none` until the NVCR base ships
a torch >= 2.11 carrying `ScalingType` and upstream torchao lands the
expand op; then re-test mxfp8 first, since it is the preferred mode.

## Known incident: OOM-killed vLLM after a large `/v1/images/edits` upload (2026-08-24)

Sending an oversized edit-source image (e.g. a 4080x2296 phone photo) with no
explicit `size` caused resident VRAM to balloon to ~65 GB and the kernel OOM
killer to kill the co-located vLLM (qwen3.8/sglang) process. Root cause: the
edit endpoint used the raw uploaded image's dimensions as the generation
target, bypassing the 2048x2048 `size` cap entirely — and klein's edit path
also VAE-encodes the input image as context at its own resolution regardless
of the requested output size, so attention/VAE memory scales with whatever
the caller uploads. Fixed in `server.py`: edit-source images are now
downscaled to `MAX_EDIT_LONG_EDGE` (env var, default `2048` — matching the
existing generation-endpoint size cap; lower it if resident VRAM at this
resolution still crowds vLLM's headroom) before touching the
pipeline, and `torch.cuda.empty_cache()` runs after every generate/edit call
so one large request's reserved-memory high-water mark doesn't persist for
the container's lifetime and starve vLLM. `/health`'s `resident_memory_mb`
now reports `memory_reserved()` (what `nvidia-smi` shows) instead of
`memory_allocated()` (live tensors only, which under-reported during the
incident).

## Memory budget — read this before starting alongside vLLM

vLLM on this box currently runs at `--gpu-memory-utilization 0.84`
(`deploy/dgx-spark/vllm-qwen3.6/start.sh:96`). FLUX.2-klein-9B's peak *pipeline*
memory (transformer + text encoder + VAE) is on the order of ~66 GB
unquantized; quantizing the transformer via `QUANT_MODE` reduces this but
the text encoder and VAE are currently left at full precision, so the
actual number needs measuring on this box, not assumed.

**Before running both services at once:**

1. Start `flux2-klein` alone, run a generation, and record resident VRAM
   from `GET /health` (`resident_memory_mb`) or `nvidia-smi`.
2. Lower vLLM's `--gpu-memory-utilization` in
   `deploy/dgx-spark/vllm-qwen3.6/start.sh` (or whichever vLLM deploy dir is
   currently live) by enough to leave headroom for that figure, then
   restart vLLM.
3. Start both, confirm `/health` is `ok` on both `:8000` (vLLM) and
   `:8002` (flux2-klein) simultaneously, and re-check vLLM decode
   throughput hasn't regressed.

This requires a vLLM restart (brief downtime) to take effect — do it in a
maintenance window, not while the agent is mid-conversation.

## Benchmark table (fill in after step 1 above)

| `QUANT_MODE` | Resident VRAM | Wall-clock @ 4 steps, 1024² |
|---|---|---|
| `none` | 33,116 MB (`/health` `memory_allocated`) | 5.2s generation / 9.5s edit (measured 2026-08-23) |
| `mxfp8` | 24,746 MB (loads; generation currently blocked by the MXTensor expand gap above) | — |
| `nvfp4` | — | — |

## Rollback

```bash
./stop.sh
```

## Notes

- A single `asyncio.Lock` around the pipeline call serializes requests, same
  pattern as `deploy/dgx-spark/faster-whisper/server.py` — one generation/edit
  at a time on this GPU.
- `n` (images per request) is capped at 4 to bound one request's memory/time
  cost on a shared box.
- `size` is capped at 2048×2048 for the same reason.
