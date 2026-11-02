"""MOSS-Image text-to-image server.

OpenAI-shaped /v1/images/generations backed by OpenMOSS-Team/MOSS-Image-SFT
(default) or MOSS-Image-Base, loaded through the model repo's own diffusers
custom pipeline (pipeline_moss_image.py, trust_remote_code).

Same endpoint shape, port and `steps`/`seed`/`size`/`n` extras as
dgx-spark/qwen-image-2.1 and flux2-klein, so the nixclaw client can point at
it unchanged - except /v1/images/edits: MOSS-Image is text-to-image only, so
that endpoint answers 501.

Model notes (from the model card and pipeline_moss_image.py):
  - 5B DiT (bf16) + Qwen3.5-9B text encoder (bf16, only last_hidden_state is
    used) + FLUX.2 small VAE (fp32).
  - Not distilled: 50 steps, CFG 3, shift 5. The negative branch skips DiT
    block 22 (uncond_skip_layers=[22]). Two DiT forwards per step.
  - width/height must be multiples of 16; trained up to ~1 MP.
  - Output is RGB.

Requests are serialized through one asyncio.Lock (the box is shared with
vLLM/sglang) and run in a worker thread so /health stays responsive while an
image is generating.
"""

import asyncio
import base64
import io
import logging
import math
import os
import time
import types
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

LOG = logging.getLogger("moss-image-server")

IMAGE_MODEL_ID = os.environ.get("IMAGE_MODEL_ID", "OpenMOSS-Team/MOSS-Image-SFT")
# Optional commit/tag. The pipeline code is pulled from the model repo
# (trust_remote_code), so pinning a revision also pins that code.
MODEL_REVISION = os.environ.get("MODEL_REVISION") or None
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# Text encoder storage precision. The DiT and VAE always stay at the
# model card's precision (bf16 / fp32).
#   none (default): bf16, ~27 GB resident in total
#   fp8: Linear weights in the language model stored as float8_e4m3fn with a
#        per-output-channel scale, upcast to bf16 per call; ~19 GB in total.
#        The encoder runs once per request, so the upcast costs ~nothing.
TEXT_ENCODER_QUANT = os.environ.get("TEXT_ENCODER_QUANT", "none").lower()
# The encoder checkpoint is a full Qwen3.5 VL model, but the pipeline only
# feeds it text. Dropping the vision tower frees ~0.8 GB.
DROP_VISION_TOWER = os.environ.get("DROP_VISION_TOWER", "1").lower() not in ("0", "false", "no")

NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "50"))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "3.0"))
# Comma-separated, zero-based. Empty string disables negative-branch skipping.
UNCOND_SKIP_LAYERS = [int(x) for x in os.environ.get("UNCOND_SKIP_LAYERS", "22").split(",") if x.strip()]
MAX_STEPS = int(os.environ.get("MAX_STEPS", "100"))

# Final training stage capped the area at 1024x1024; the upstream Python
# example goes up to 1456x1456 (2.1 MP). Allow that, default to 1 MP.
DEFAULT_SIZE = os.environ.get("DEFAULT_SIZE", "1024x1024")
MAX_SIZE = int(os.environ.get("MAX_SIZE", "2048"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(1456 * 1456)))

# Admission control: this box shares one 128 GB unified pool with other
# tenants, so a request only starts when the pool has room for it:
#   n x output_MP x (DENOISE_MB_PER_MP) + VAE decode + ADMISSION_MARGIN_MB
# The per-MP rates are first estimates, NOT measurements - check the
# "Request peak" and "VAE decode peak" log lines after a few requests and
# tune them.
DENOISE_MB_PER_MP = float(os.environ.get("DENOISE_MB_PER_MP", "1500"))
TILED_VAE_MB = float(os.environ.get("TILED_VAE_MB", "1000"))
ADMISSION_MARGIN_MB = float(os.environ.get("ADMISSION_MARGIN_MB", "2000"))
ADMISSION_WAIT_SECONDS = float(os.environ.get("ADMISSION_WAIT_SECONDS", "60"))
ADMISSION_POLL_SECONDS = float(os.environ.get("ADMISSION_POLL_SECONDS", "5"))

# VAE decode (fp32) is the peak-memory moment of a request.
#   auto (default): untiled when the pool has room right before the decode
#   1/always: always tiled     0/never: never tiled
_vae_tiling_raw = os.environ.get("VAE_TILING", "auto").lower()
if _vae_tiling_raw in ("0", "false", "no", "never"):
    VAE_TILING = "never"
elif _vae_tiling_raw == "auto":
    VAE_TILING = "auto"
else:
    VAE_TILING = "always"
UNTILED_VAE_MB_PER_MP = float(os.environ.get("UNTILED_VAE_MB_PER_MP", "6000"))
UNTILED_VAE_MARGIN_MB = float(os.environ.get("UNTILED_VAE_MARGIN_MB", "2000"))

_pipe = None
_lock: Optional[asyncio.Lock] = None
_load_error: Optional[str] = None
_last_vae_decision: Optional[dict] = None
_quant_mode = "none"


# ---------------------------------------------------------------- sizing

def _parse_size(size: Optional[str]) -> tuple[int, int]:
    raw = size or DEFAULT_SIZE
    try:
        w_str, h_str = raw.lower().split("x")
        width, height = int(w_str), int(h_str)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"invalid size: {size!r}, expected e.g. '1024x1024'")
    # The pipeline rejects sizes that aren't multiples of 16; round instead.
    width, height = max(16, round(width / 16) * 16), max(16, round(height / 16) * 16)
    if width > MAX_SIZE or height > MAX_SIZE:
        raise HTTPException(status_code=400, detail=f"size {width}x{height} exceeds max edge {MAX_SIZE}")
    if width * height > MAX_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"size {width}x{height} = {width * height / 1e6:.2f} MP exceeds the {MAX_PIXELS / 1e6:.2f} MP cap",
        )
    return width, height


# ---------------------------------------------------------------- memory

def _host_available_mb() -> Optional[float]:
    """MemAvailable from /proc/meminfo. On this unified-memory box
    torch.cuda.mem_get_info() alone overstates headroom."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cuda_memory_info_mb() -> Optional[tuple[float, float]]:
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / 2**20, total / 2**20
    except Exception:
        pass
    return None


def _memory_info_mb() -> Optional[tuple[float, float]]:
    """(free, total) MB of the shared pool; free = min(CUDA view, host view)."""
    cuda = _cuda_memory_info_mb()
    if cuda is None:
        return None
    free, total = cuda
    host_free = _host_available_mb()
    if host_free is not None:
        free = min(free, host_free)
    return free, total


def _request_cost_mb(pixels: int, n: int) -> float:
    return n * pixels / 1e6 * DENOISE_MB_PER_MP + TILED_VAE_MB


def _request_need_mb(pixels: int, n: int) -> float:
    return _request_cost_mb(pixels, n) + ADMISSION_MARGIN_MB


def _largest_fitting_size(free_mb: float, n: int, aspect: float) -> Optional[tuple[int, int]]:
    pixels = min((free_mb - TILED_VAE_MB - ADMISSION_MARGIN_MB) / n / DENOISE_MB_PER_MP * 1e6, MAX_PIXELS)
    if pixels < 512 * 512:
        return None
    height = math.sqrt(pixels / aspect)
    width = min(height * aspect, MAX_SIZE)
    height = min(height, MAX_SIZE)
    return max(16, int(width) // 16 * 16), max(16, int(height) // 16 * 16)


async def _await_memory_headroom(pixels: int, n: int, aspect: float, deadline: float) -> None:
    """Wait until the pool has room for this request, or 503 at the deadline.
    Runs under _lock, so queued requests keep their order."""
    need_mb = _request_need_mb(pixels, n)
    waited_from = time.monotonic()
    while True:
        info = _memory_info_mb()
        if info is None:
            return
        free_mb, total_mb = info
        if free_mb >= need_mb:
            return
        if time.monotonic() >= deadline:
            break
        LOG.info(f"Waiting for memory: {free_mb:.0f} of {total_mb:.0f} MB free, need {need_mb:.0f} MB")
        await asyncio.sleep(ADMISSION_POLL_SECONDS)

    fit = _largest_fitting_size(free_mb, n, aspect)
    hint = f"The largest size that fits right now is about {fit[0]}x{fit[1]}." if fit else "Retry later."
    raise HTTPException(
        status_code=503,
        detail=(
            f"insufficient GPU memory: {free_mb:.0f} MB free, {need_mb:.0f} MB required "
            f"(waited {time.monotonic() - waited_from:.0f}s; another tenant is using the shared pool). {hint}"
        ),
    )


def _resident_memory_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 2**20
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- VAE

def _configure_vae(pipe) -> None:
    """Decode n>1 images one at a time, and pick tiled vs untiled per decode
    from the free pool right before it (the decode runs a minute after
    admission, other tenants may have grown since)."""
    global _last_vae_decision
    import torch

    vae = pipe.vae
    for method in ("enable_slicing",):
        try:
            getattr(vae, method)()
            LOG.info(f"vae.{method}() enabled")
        except Exception as exc:
            LOG.warning(f"vae.{method}() unavailable: {exc}")
    if VAE_TILING == "never":
        LOG.info("VAE tiling disabled (VAE_TILING=0)")
        return
    try:
        vae.enable_tiling()
    except Exception as exc:
        LOG.warning(f"VAE tiling unavailable, decoding untiled: {exc}")
        return
    LOG.info(f"VAE tiling: {VAE_TILING}")

    orig_decode = vae.decode

    def decode(z, *args, **kwargs):
        global _last_vae_decision
        # The pipeline unpacks 2x2 latent patches before decoding, so z is at
        # 1/8 of the output resolution.
        pixels = z.shape[-2] * z.shape[-1] * 64
        if VAE_TILING == "auto":
            torch.cuda.empty_cache()
            info = _memory_info_mb()
            free_mb = info[0] if info else 0.0
            need_mb = pixels / 1e6 * UNTILED_VAE_MB_PER_MP + UNTILED_VAE_MARGIN_MB
            vae.use_tiling = not (info is not None and free_mb >= need_mb)
        else:
            free_mb, need_mb = None, None
            vae.use_tiling = True
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_reserved()
        try:
            return orig_decode(z, *args, **kwargs)
        finally:
            peak_mb = (torch.cuda.max_memory_reserved() - base) / 2**20
            mp = pixels / 1e6
            _last_vae_decision = {
                "tiled": vae.use_tiling,
                "pixels": pixels,
                "free_mb": round(free_mb) if free_mb is not None else None,
                "untiled_need_mb": round(need_mb) if need_mb is not None else None,
                "peak_reserved_mb": round(peak_mb),
                "at": int(time.time()),
            }
            LOG.info(
                f"VAE decode peak: +{peak_mb:.0f} MB reserved "
                f"({'tiled' if vae.use_tiling else 'untiled'}, {mp:.2f} MP, {peak_mb / mp:.0f} MB/MP)"
            )
            vae.use_tiling = True

    vae.decode = decode


# ---------------------------------------------------------------- text encoder

def _fp8_linear_forward(self, x):
    weight = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype)
    bias = None if self.bias is None else self.bias.to(x.dtype)
    import torch.nn.functional as F

    return F.linear(x, weight, bias)


def _quantize_text_encoder_fp8(text_encoder) -> int:
    """Store every language-model Linear weight as float8_e4m3fn with one
    bf16 scale per output channel (absmax / 448). A plain cast without scales
    would flush the many small weights of a 9B LLM to zero. Embeddings and
    norms stay bf16. Done in place, layer by layer, on the GPU."""
    import torch

    root = getattr(text_encoder, "language_model", text_encoder)
    count = 0
    for module in root.modules():
        if not isinstance(module, torch.nn.Linear) or module.weight.dtype == torch.float8_e4m3fn:
            continue
        w = module.weight.data
        scale = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / 448.0
        q = (w.float() / scale).to(torch.float8_e4m3fn)
        module.weight = torch.nn.Parameter(q, requires_grad=False)
        module.register_buffer("weight_scale", scale.to(torch.bfloat16), persistent=False)
        module.forward = types.MethodType(_fp8_linear_forward, module)
        count += 1
        del w, scale, q
        if count % 32 == 0:
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return count


def _module_gb(module) -> float:
    return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers())) / 1e9


# ---------------------------------------------------------------- loading

def _load_pipeline() -> None:
    """Load once at startup. Sets _load_error instead of raising, so /health
    reports why instead of the container crash-looping."""
    global _pipe, _lock, _load_error, _quant_mode
    import torch
    from diffusers import DiffusionPipeline

    try:
        if TEXT_ENCODER_QUANT not in ("none", "fp8"):
            raise ValueError(f"TEXT_ENCODER_QUANT must be 'none' or 'fp8', got {TEXT_ENCODER_QUANT!r}")
        LOG.info(
            f"Loading {IMAGE_MODEL_ID} (revision={MODEL_REVISION or 'main'}, text_encoder={TEXT_ENCODER_QUANT}, "
            f"steps={NUM_INFERENCE_STEPS}, guidance={GUIDANCE_SCALE}, uncond_skip_layers={UNCOND_SKIP_LAYERS})"
        )
        kwargs = dict(
            custom_pipeline="pipeline_moss_image",
            trust_remote_code=True,
            torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
            token=HF_TOKEN,
            revision=MODEL_REVISION,
        )
        if MODEL_REVISION:
            kwargs["custom_revision"] = MODEL_REVISION
        # Load straight onto the GPU: on unified memory a CPU copy + .to()
        # briefly needs the weights twice in the same pool.
        try:
            pipe = DiffusionPipeline.from_pretrained(IMAGE_MODEL_ID, device_map="cuda", **kwargs)
            for name, component in pipe.components.items():
                if isinstance(component, torch.nn.Module) and any(
                    t.device.type == "cpu" for t in list(component.parameters()) + list(component.buffers())
                ):
                    LOG.warning(f"{name} had tensors on cpu after device_map load; moving to cuda")
                    component.to("cuda")
        except (ValueError, NotImplementedError, TypeError) as exc:
            LOG.warning(f"device_map='cuda' not supported ({exc}); loading on CPU and moving to cuda")
            pipe = DiffusionPipeline.from_pretrained(IMAGE_MODEL_ID, **kwargs).to("cuda")
        # As in the model card's quick start.
        pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)

        te = pipe.text_encoder
        if DROP_VISION_TOWER and getattr(te, "visual", None) is not None:
            freed = _module_gb(te.visual)
            te.visual = None
            torch.cuda.empty_cache()
            LOG.info(f"Dropped unused vision tower ({freed:.2f} GB)")
        if TEXT_ENCODER_QUANT == "fp8":
            before = _module_gb(te)
            n = _quantize_text_encoder_fp8(te)
            _quant_mode = "text_encoder_fp8"
            LOG.info(f"Text encoder FP8: {n} Linear layers, {before:.1f} GB -> {_module_gb(te):.1f} GB")
        te.requires_grad_(False).eval()

        for name in ("transformer", "text_encoder", "vae"):
            LOG.info(f"{name}: {_module_gb(getattr(pipe, name)):.2f} GB")

        # Fail at startup, not on the first request, if the encoder is broken
        # (vision tower removal, FP8 patching, transformers drift).
        t0 = time.monotonic()
        embeds, _ = pipe.encode_prompt("warm-up prompt")
        if not torch.isfinite(embeds).all():
            raise RuntimeError("text encoder produced non-finite embeddings on warm-up")
        LOG.info(f"Text encoder warm-up OK ({time.monotonic() - t0:.1f}s, embeds {tuple(embeds.shape)})")

        _configure_vae(pipe)
        torch.cuda.empty_cache()
        _pipe = pipe
        _lock = asyncio.Lock()
        LOG.info(f"Pipeline loaded; resident {(_resident_memory_mb() or 0) / 1024:.1f} GB")
    except Exception as exc:
        LOG.exception(f"Pipeline load failed: {exc}")
        _load_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load in a thread so the event loop serves /health (503 "loading")
    # during the download instead of refusing connections.
    asyncio.get_running_loop().run_in_executor(None, _load_pipeline)
    yield


app = FastAPI(title="moss-image-server", lifespan=lifespan)


# ---------------------------------------------------------------- endpoints

@app.get("/health")
async def health() -> dict:
    if _pipe is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error or 'loading'}")
    info = _memory_info_mb()
    return {
        "status": "ok",
        "model": IMAGE_MODEL_ID,
        "revision": MODEL_REVISION,
        "quant_mode": _quant_mode,
        "steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "uncond_skip_layers": UNCOND_SKIP_LAYERS,
        "busy": _lock.locked() if _lock else None,
        "resident_memory_mb": _resident_memory_mb(),
        "pool_free_mb": round(info[0]) if info else None,
        "pool_total_mb": round(info[1]) if info else None,
        "vae_tiling": VAE_TILING,
        "last_vae_decision": _last_vae_decision,
        "admission": {
            "denoise_mb_per_mp": DENOISE_MB_PER_MP,
            "tiled_vae_mb": TILED_VAE_MB,
            "margin_mb": ADMISSION_MARGIN_MB,
            "wait_seconds": ADMISSION_WAIT_SECONDS,
            "need_mb_1024x1024": round(_request_need_mb(1024 * 1024, 1)),
        },
    }


@app.get("/v1/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [{"id": IMAGE_MODEL_ID, "object": "model", "created": 0, "owned_by": "openmoss", "quant_mode": _quant_mode}],
    }


def _image_to_b64_png(image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/v1/images/generations")
async def generate(payload: dict) -> JSONResponse:
    """OpenAI-shaped generation.

    Body: {"prompt": str, "size": "WxH", "n": int, "response_format": "b64_json",
           "steps": int, "seed": int, "negative_prompt": str, "guidance_scale": float}
    (everything after response_format is a non-standard extra)
    """
    if _pipe is None or _lock is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error or 'loading'}")

    prompt = payload.get("prompt")
    if not prompt or not isinstance(prompt, str):
        raise HTTPException(status_code=400, detail="prompt is required")
    if payload.get("response_format", "b64_json") != "b64_json":
        raise HTTPException(status_code=400, detail="only response_format=b64_json is supported")
    width, height = _parse_size(payload.get("size"))
    n = int(payload.get("n") or 1)
    if not 1 <= n <= 4:
        raise HTTPException(status_code=400, detail="n must be between 1 and 4")
    steps = int(payload.get("steps") or NUM_INFERENCE_STEPS)
    if not 2 <= steps <= MAX_STEPS:
        raise HTTPException(status_code=400, detail=f"steps must be between 2 and {MAX_STEPS}")
    guidance = float(payload.get("guidance_scale") or GUIDANCE_SCALE)
    negative_prompt = payload.get("negative_prompt") or ""
    seed = payload.get("seed")
    deadline = time.monotonic() + ADMISSION_WAIT_SECONDS

    import torch

    # CPU generator, as in the model card: same seed -> same image on any device.
    generator = torch.Generator("cpu").manual_seed(int(seed)) if seed is not None else None

    def _run():
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_reserved()
        try:
            return _pipe(
                prompt,
                width=width,
                height=height,
                num_images_per_prompt=n,
                num_inference_steps=steps,
                guidance_scale=guidance,
                negative_prompt=negative_prompt,
                uncond_skip_layers=UNCOND_SKIP_LAYERS,
                generator=generator,
            )
        finally:
            peak_mb = (torch.cuda.max_memory_reserved() - base) / 2**20
            LOG.info(
                f"Request peak: +{peak_mb:.0f} MB reserved "
                f"(admission estimate {_request_cost_mb(width * height, n):.0f} MB without margin)"
            )
            # Return the high-water mark to the shared pool.
            torch.cuda.empty_cache()

    async with _lock:
        await _await_memory_headroom(width * height, n, width / height, deadline)
        LOG.info(f"Generating: prompt={prompt[:200]!r} size={width}x{height} n={n} steps={steps} cfg={guidance} seed={seed}")
        start = time.monotonic()
        try:
            result = await asyncio.to_thread(_run)
        except Exception as exc:
            LOG.exception(f"Generation failed: {exc}")
            return JSONResponse({"error": str(exc)}, status_code=500)
        LOG.info(f"Generated {len(result.images)} image(s) in {time.monotonic() - start:.1f}s")

    return JSONResponse({
        "created": int(time.time()),
        "data": [{"b64_json": _image_to_b64_png(img)} for img in result.images],
    })


@app.post("/v1/images/edits")
async def edit() -> JSONResponse:
    raise HTTPException(status_code=501, detail="MOSS-Image is text-to-image only; image editing is not supported")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
