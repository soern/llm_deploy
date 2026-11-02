# moss-image

Text-to-image HTTP service backed by
[MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)
(default) or [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base),
on the DGX Spark (GB10, ARM64, 128 GB unified memory). Same OpenAI-shaped
endpoints and host port (**8002**) as `qwen-image-2.1` / `flux2-klein`, so
the nixclaw client works unchanged — **except editing**: MOSS-Image is
text-to-image only, and `/v1/images/edits` answers 501.

Apache-2.0, not gated (`HF_TOKEN` optional).

## Model

| Component | Class | Precision | On disk |
|---|---|---|---|
| DiT (5B, 42 blocks) | `MOSSImageTransformer` (repo code) | bf16 | 10.2 GB |
| Text encoder (Qwen3.5-9B) | `Qwen3_5Model` | bf16 | 16.8 GB |
| VAE (FLUX.2 small) | `AutoencoderKLFlux2` | fp32 | 0.25 GB |

The pipeline (`pipeline_moss_image.py`) comes from the model repo via
`trust_remote_code`. Set `MODEL_REVISION=<commit>` to pin both the weights
and that code.

### SFT (default) vs Base

Same architecture, same size, same pipeline. SFT is Base fine-tuned on
curated data. Neither the model cards nor the tech report (arXiv 2610.23374)
publish Base-vs-SFT numbers; the cards show one example where SFT has
cleaner text grouping and layout.

- **Serving images → SFT.** It is the polished checkpoint, and it is what
  the community has picked (the SFT quantizations get ~10x the downloads of
  the Base ones).
- **Training LoRAs → Base.** A less opinionated starting point; most LoRA
  tooling (e.g. the NF4-for-LoRA-training repo) targets Base.

Switch with `IMAGE_MODEL_ID=OpenMOSS-Team/MOSS-Image-Base`.

### Memory and quantization

| `TEXT_ENCODER_QUANT` | DiT | Text encoder | VAE | Resident (est.) |
|---|---|---|---|---|
| `none` (default) | 10.2 GB bf16 | ~16.0 GB bf16 | 0.25 GB | **~27 GB**, ~29–30 GB peak at 1 MP |
| `fp8` | 10.2 GB bf16 | ~9 GB | 0.25 GB | **~19.5 GB**, ~22 GB peak |

The unused vision tower of the encoder (~0.8 GB) is dropped at load in
both modes (`DROP_VISION_TOWER=0` keeps it).

Full bf16 already lands at the ~30 GB target, so it is the default: no
quality risk at all. `fp8` is the next step if the pool gets tight. It
stores the encoder's Linear weights as `float8_e4m3fn` with one scale per
output channel and upcasts them per call. The encoder runs once per
request, so this costs nothing in speed, and of all the components it is
the one where quantization matters least. The embedding table (2 GB) stays
bf16.

The DiT is left alone on purpose. It is only 10 GB, its precision shows
up directly in text rendering (MOSS-Image's main strength), and on GB10
weight-only quantization saves memory but not time. The community GGUF/
SDNQ/INT8 builds target ComfyUI and low-VRAM GPUs, not this box.

All figures are from file sizes, not yet measured on the Spark: check
`/health` (`resident_memory_mb`) and the `Request peak` / `VAE decode peak`
log lines, then tune `DENOISE_MB_PER_MP` and `UNTILED_VAE_MB_PER_MP`.

### Sampling defaults

From the model card: flow-matching Euler, **50 steps**, **CFG 3**, shift 5,
and the negative branch skips DiT block 22 (`UNCOND_SKIP_LAYERS=22`; empty
disables it). The model is **not distilled**, and each step runs the DiT
twice (cond + uncond). Expect roughly a minute or more per 1 MP image on
GB10 (unmeasured). If that is too slow for the agent's 180 s
`image_gen.timeout`, try `NUM_INFERENCE_STEPS=30`.

Sizes are rounded to multiples of 16. The default is 1024×1024; the cap is
1456×1456 (2.1 MP, the largest the upstream example uses). The final
training stage capped the area at ~1 MP.

Prompts can be natural language, tag lists, or a JSON layout
(`summary.caption` + `items` with `bbox_format: "yxyx"`,
`bbox_coord_space: "abs_centralized"`). See the upstream
[user guide](https://github.com/OpenMOSS/MOSS-Image).

## Endpoints

- `GET  /health`: 503 while loading or after a failed load (with the reason), then model, memory and admission info
- `GET  /v1/models`
- `POST /v1/images/generations`:
  `{"prompt": str, "size": "1024x1024", "n": 1, "response_format": "b64_json", "steps": 50, "seed": 42, "negative_prompt": "", "guidance_scale": 3.0}`
  (everything after `response_format` is an optional extra)
- `POST /v1/images/edits`: 501, not supported by the model

## Build & run

```bash
# stop whatever holds port 8002 first, e.g. in ../qwen-image-2.1:
./stop.sh

./start.sh                            # SFT, bf16 (~27 GB)
TEXT_ENCODER_QUANT=fp8 ./start.sh     # ~19.5 GB
IMAGE_MODEL_ID=OpenMOSS-Team/MOSS-Image-Base ./start.sh

# or
docker compose up -d --build
```

The first run downloads ~27 GB into `.cache/huggingface/` next to the
script. `start.sh` refuses to start if port 8002 is already serving.

## Verify

```bash
curl -s localhost:8002/health | jq
curl -s localhost:8002/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A whiteboard with the handwritten words \"Ship it Friday\" and a small doodle of a rocket", "seed": 42}' \
  | jq -r '.data[0].b64_json' | base64 -d > /tmp/moss.png
```
