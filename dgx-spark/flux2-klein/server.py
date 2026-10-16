"""FLUX.2 [klein] image generation/editing server.

OpenAI-shaped /v1/images/generations and /v1/images/edits endpoints backed
by black-forest-labs/FLUX.2-klein-9B via diffusers' Flux2KleinPipeline (the
klein model is Qwen-text-encoder based; the Mistral-based Flux2Pipeline
crashes on it with "can only concatenate str (not 'list') to str" at
generation time). Designed to sit next to vLLM on the same DGX Spark (GB10, ARM64, sm_121a) box,
sharing the GPU, so requests are serialized through a single asyncio.Lock
exactly like deploy/dgx-spark/faster-whisper/server.py serializes STT calls.

Quantization is applied on the fly at load time (not from a pre-quantized
checkpoint - pre-quantized FP4 checkpoints have known SM121A
incompatibilities), controlled by QUANT_MODE:
  - "nvfp4": NVFP4DynamicActivationNVFP4WeightConfig (smallest, more drift)
  - "mxfp8": MXDynamicActivationMXWeightConfig (larger, near-lossless)
  - "none":  no quantization (escape hatch if the torchao nightly wheels
             misbehave on this box)

klein is a distilled model, so NUM_INFERENCE_STEPS defaults low (4).
"""

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

LOG = logging.getLogger("flux2-klein-server")

IMAGE_MODEL_ID = os.environ.get("IMAGE_MODEL_ID", "black-forest-labs/FLUX.2-klein-9B")
QUANT_MODE = os.environ.get("QUANT_MODE", "mxfp8").lower()  # mxfp8 | nvfp4 | none
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "4"))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "3.5"))
HF_TOKEN = os.environ.get("HF_TOKEN") or None
DEFAULT_SIZE = os.environ.get("DEFAULT_SIZE", "1024x1024")
# Edit-source images are VAE-encoded as context at their own resolution
# regardless of the requested output size, so an oversized upload (e.g. a
# 4080x2296 phone photo) inflates transformer attention memory independent
# of any width/height cap on the output - this is what took the box down
# via OOM-killing the co-located vLLM process. 2048 matches the existing
# generation-endpoint size cap (see _parse_size); lower via env var if
# resident VRAM (see /health) still gets too close to the vLLM process's
# headroom at this resolution.
MAX_EDIT_LONG_EDGE = int(os.environ.get("MAX_EDIT_LONG_EDGE", "2048"))

if QUANT_MODE not in ("mxfp8", "nvfp4", "none"):
    raise SystemExit(f"Invalid QUANT_MODE={QUANT_MODE!r}; expected mxfp8, nvfp4, or none")

_pipe = None
_lock = None  # type: ignore[var-annotated]
_load_error: Optional[str] = None


def _parse_size(size: Optional[str]) -> tuple[int, int]:
    raw = size or DEFAULT_SIZE
    try:
        w_str, h_str = raw.lower().split("x")
        width, height = int(w_str), int(h_str)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"invalid size: {raw!r}, expected e.g. '1024x1024'")
    if width <= 0 or height <= 0 or width > 2048 or height > 2048:
        raise HTTPException(status_code=400, detail=f"size out of range: {raw!r}")
    return width, height


def _build_quantization_config():
    """Return a diffusers PipelineQuantizationConfig for the transformer, or None."""
    if QUANT_MODE == "none":
        return None

    from diffusers import PipelineQuantizationConfig, TorchAoConfig

    if QUANT_MODE == "nvfp4":
        from torchao.prototype.mx_formats.inference_workflow import (
            NVFP4DynamicActivationNVFP4WeightConfig,
        )
        cfg = NVFP4DynamicActivationNVFP4WeightConfig()
    else:  # mxfp8
        from torchao.prototype.mx_formats.inference_workflow import (
            MXDynamicActivationMXWeightConfig,
        )
        cfg = MXDynamicActivationMXWeightConfig()

    return PipelineQuantizationConfig(quant_mapping={"transformer": TorchAoConfig(cfg)})


def _load_pipeline() -> None:
    """Load the Flux2KleinPipeline once at startup. Sets _load_error instead of
    raising so /health can report a clear "not loaded" status rather than
    the container just crash-looping on a slow/broken GPU box."""
    global _pipe, _lock, _load_error
    import asyncio

    import torch
    from diffusers import Flux2KleinPipeline

    try:
        LOG.info(f"Loading {IMAGE_MODEL_ID} (quant_mode={QUANT_MODE}, steps={NUM_INFERENCE_STEPS})")
        quant_config = _build_quantization_config()
        kwargs: dict = {"torch_dtype": torch.bfloat16, "token": HF_TOKEN}
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
        pipe = Flux2KleinPipeline.from_pretrained(IMAGE_MODEL_ID, **kwargs).to("cuda")
        pipe.transformer.compile_repeated_blocks(fullgraph=True)
        _pipe = pipe
        _lock = asyncio.Lock()
        LOG.info("Pipeline loaded successfully")
    except Exception as exc:
        LOG.exception(f"Pipeline load failed: {exc}")
        _load_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_pipeline()
    yield


app = FastAPI(title="flux2-klein-server", lifespan=lifespan)


def _resident_memory_mb() -> Optional[float]:
    """Reserved (not just allocated) VRAM - this is what nvidia-smi shows and
    what actually contends with the co-located vLLM process, since PyTorch's
    caching allocator holds reserved blocks even after their tensors are
    freed."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / (1024 * 1024)
    except Exception:
        pass
    return None


@app.get("/health")
async def health() -> dict:
    if _pipe is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error or 'loading'}")
    return {
        "status": "ok",
        "model": IMAGE_MODEL_ID,
        "quant_mode": QUANT_MODE,
        "steps": NUM_INFERENCE_STEPS,
        "resident_memory_mb": _resident_memory_mb(),
    }


@app.get("/v1/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": IMAGE_MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "black-forest-labs",
                "quant_mode": QUANT_MODE,
            }
        ],
    }


def _image_to_b64_png(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _clamp_to_long_edge(image, max_long_edge: int):
    """Downscale `image` (preserving aspect ratio) so its longest edge is
    <= max_long_edge. Dimensions are rounded to a multiple of 16 (diffusion
    U-Net/transformer latent stride). No-op if already within bounds."""
    from PIL import Image

    width, height = image.width, image.height
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    new_width = max(16, round(width * scale / 16) * 16)
    new_height = max(16, round(height * scale / 16) * 16)
    LOG.info(f"Downscaling oversized edit input {width}x{height} -> {new_width}x{new_height}")
    return image.resize((new_width, new_height), Image.LANCZOS)


@app.post("/v1/images/generations")
async def generate(payload: dict) -> JSONResponse:
    """OpenAI-shaped image generation.

    Body: {"prompt": str, "size": "WxH", "n": int, "response_format": "b64_json",
           "steps": int, "seed": int}  (steps/seed are non-standard extras)
    """
    if _pipe is None or _lock is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error or 'loading'}")

    prompt = payload.get("prompt")
    if not prompt or not isinstance(prompt, str):
        raise HTTPException(status_code=400, detail="prompt is required")

    response_format = payload.get("response_format", "b64_json")
    if response_format != "b64_json":
        raise HTTPException(status_code=400, detail=f"unsupported response_format: {response_format}")

    width, height = _parse_size(payload.get("size"))
    n = int(payload.get("n", 1) or 1)
    if n < 1 or n > 4:
        raise HTTPException(status_code=400, detail="n must be between 1 and 4")
    steps = int(payload.get("steps") or NUM_INFERENCE_STEPS)
    seed = payload.get("seed")

    import torch

    generator = torch.Generator(device="cuda").manual_seed(int(seed)) if seed is not None else None

    async with _lock:
        start = time.monotonic()
        LOG.info(f"Generating: prompt={prompt!r} size={width}x{height} n={n} steps={steps} seed={seed}")
        try:
            result = _pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_images_per_prompt=n,
                num_inference_steps=steps,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            )
        except Exception as exc:
            LOG.exception(f"Generation failed: {exc}")
            return JSONResponse({"error": str(exc)}, status_code=500)
        finally:
            # The caching allocator doesn't return reserved VRAM to the OS on
            # its own, so a large request's high-water mark otherwise persists
            # for the container's lifetime, starving the co-located vLLM
            # process on this shared unified-memory box.
            torch.cuda.empty_cache()
        elapsed = time.monotonic() - start
        LOG.info(f"Generated {len(result.images)} image(s) in {elapsed:.1f}s")

    return JSONResponse({
        "created": int(time.time()),
        "data": [{"b64_json": _image_to_b64_png(img)} for img in result.images],
    })


@app.post("/v1/images/edits")
async def edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    size: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    response_format: Optional[str] = Form("b64_json"),
    steps: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
) -> JSONResponse:
    """OpenAI-shaped image editing (multipart), routed to klein's native
    image-editing path (the pipeline accepts an input image alongside the
    prompt rather than a separate edit-specific model)."""
    if _pipe is None or _lock is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error or 'loading'}")
    if response_format != "b64_json":
        raise HTTPException(status_code=400, detail=f"unsupported response_format: {response_format}")

    from PIL import Image

    try:
        input_bytes = await image.read()
        input_image = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode input image: {exc}")

    input_image = _clamp_to_long_edge(input_image, MAX_EDIT_LONG_EDGE)
    width, height = _parse_size(size) if size else (input_image.width, input_image.height)
    n_val = int(n or 1)
    if n_val < 1 or n_val > 4:
        raise HTTPException(status_code=400, detail="n must be between 1 and 4")
    steps_val = int(steps or NUM_INFERENCE_STEPS)

    import torch

    generator = torch.Generator(device="cuda").manual_seed(int(seed)) if seed is not None else None

    async with _lock:
        start = time.monotonic()
        LOG.info(f"Editing: prompt={prompt!r} size={width}x{height} n={n_val} steps={steps_val} seed={seed}")
        try:
            result = _pipe(
                prompt=prompt,
                image=input_image,
                width=width,
                height=height,
                num_images_per_prompt=n_val,
                num_inference_steps=steps_val,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            )
        except Exception as exc:
            LOG.exception(f"Edit failed: {exc}")
            return JSONResponse({"error": str(exc)}, status_code=500)
        finally:
            torch.cuda.empty_cache()
        elapsed = time.monotonic() - start
        LOG.info(f"Edited -> {len(result.images)} image(s) in {elapsed:.1f}s")

    return JSONResponse({
        "created": int(time.time()),
        "data": [{"b64_json": _image_to_b64_png(img)} for img in result.images],
    })


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
