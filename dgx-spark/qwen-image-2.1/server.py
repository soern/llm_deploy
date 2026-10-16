"""Qwen-Image-2.1 image generation/editing server.

OpenAI-shaped /v1/images/generations and /v1/images/edits endpoints backed
by Qwen/Qwen-Image-2.1-Turbo (default) or Qwen/Qwen-Image-2.1 via diffusers'
QwenImage21Pipeline. One 7.1B DiT
serves both text-to-image and image-conditioned generation (editing) —
the prompt and any reference image are encoded together by the
Qwen3-VL-8B text encoder; there is no separate edit model.

Drop-in replacement for deploy/dgx-spark/flux2-klein/: same endpoint
shape, same `steps`/`seed`/`size`/`n` parameter names, same /health and
/v1/models reporting. The DiT and VAE run at full precision (bf16); the
Qwen3-VL text encoder is FP8-stored by default (TEXT_ENCODER_QUANT, see
_load_fp8_text_encoder), on the DGX Spark (GB10, ARM64, sm_121a) box that already hosts vLLM,
sharing the GPU, so requests are serialized through a single
asyncio.Lock exactly like deploy/dgx-spark/faster-whisper/server.py.

Model signature notes (diffusers main, merged 2026-09-18 via PR #14804):
  - base model: num_inference_steps default 40 (NOT distilled — the old
    FLUX.2-klein default of 4 steps does NOT apply)
  - Turbo: ships a fixed 8-step schedule as `sample_sigmas` in its
    model_index.json (diffusers PR #14950, merged 2026-10-05); the pipeline
    then ignores num_inference_steps. Same DiT architecture, VAE and
    (byte-identical) text encoder as the base model
  - true_cfg_scale default 1.0: the model is meant to be sampled
    without guidance; cfg > 1 only engages with a negative_prompt and
    roughly doubles latency. We do not use guidance.
  - height/width must be divisible by 32 (vae_scale_factor * 2 = 32)
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

LOG = logging.getLogger("qwen-image-2.1-server")

IMAGE_MODEL_ID = os.environ.get("IMAGE_MODEL_ID", "Qwen/Qwen-Image-2.1-Turbo")
# Text encoder storage precision: "fp8" (default) or "none" (bf16 from
# IMAGE_MODEL_ID). The DiT and VAE always stay bf16. FP8 cuts the encoder from
# 17.5 GB to 9.4 GB of the shared 128 GB pool (OOM-kills on 2026-09-21/22);
# it runs once per request, so the per-layer upcast costs ~nothing.
TEXT_ENCODER_QUANT = os.environ.get("TEXT_ENCODER_QUANT", "fp8").lower()
TEXT_ENCODER_FP8_REPO = os.environ.get("TEXT_ENCODER_FP8_REPO", "unsloth/Qwen-Image-2.1-FP8")
TEXT_ENCODER_FP8_FILE = os.environ.get("TEXT_ENCODER_FP8_FILE", "Qwen-Image-2.1-text_encoder-FP8.safetensors")
QUANT_MODE = "text_encoder_fp8" if TEXT_ENCODER_QUANT == "fp8" else "none"
# Only used by checkpoints without a configured schedule (base model: 40).
# Turbo's sample_sigmas override it - see _fixed_schedule_steps.
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "8"))
TRUE_CFG_SCALE = float(os.environ.get("TRUE_CFG_SCALE", "1.0"))
HF_TOKEN = os.environ.get("HF_TOKEN") or None
# Qwen-Image-2.1 is a NATIVE 2K model: the official default is 2048x2048 and
# every recommended aspect ratio is ~4.2 MP (2400x1792, 2528x1696, 2752x1536,
# ...). Sampling it at FLUX-era 1024x1024 is off-distribution and produces
# mushy/garbled output - that is NOT a quantization or plumbing bug.
DEFAULT_SIZE = os.environ.get("DEFAULT_SIZE", "2048x2048")
# 2752 is the longest edge in the official aspect-ratio table (16:9).
MAX_SIZE = int(os.environ.get("MAX_SIZE", "2752"))
# Area cap, since a 2752x2752 request would be far past anything recommended.
# 4.4 MP clears every official ratio (largest is 2400x1792 = 4.30 MP).
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(4_400_000)))
# Official edit examples pass no width/height at all, so the pipeline derives
# them from the reference image's aspect ratio at `output_resolution` (1024).
# Editing therefore runs at ~1 MP by default, unlike 2K text-to-image.
EDIT_OUTPUT_RESOLUTION = int(os.environ.get("EDIT_OUTPUT_RESOLUTION", "1024"))
# The VAE is a 4-channel RGBA autoencoder, so the pipeline ALWAYS returns RGBA
# PIL images - even for ordinary opaque prompts. Chat clients that composite
# alpha badly render those as broken/black images. "auto" drops a
# fully-opaque alpha channel and keeps a genuinely transparent one;
# "flatten" always composites onto white; "keep" preserves RGBA verbatim.
# VAE decode is the peak-memory moment of a request (see
# _configure_vae_memory). Tiled decode leaves faint purple seams in smooth
# areas (diffusers #14928), untiled needs ~11 GB extra at 2048x2048.
#   auto (default): untiled when the pool has room for it right now, else tiled
#   1/always: always tiled (seams, smallest peak)
#   0/never: never tiled - froze the box on 2026-10-08, A/B testing only
_vae_tiling_raw = os.environ.get("VAE_TILING", "auto").lower()
if _vae_tiling_raw in ("0", "false", "no", "never"):
    VAE_TILING = "never"
elif _vae_tiling_raw == "auto":
    VAE_TILING = "auto"
else:
    VAE_TILING = "always"
# auto mode's estimate of an untiled VAE decode. The old 2700 came from host
# free memory sampled during one decode (2026-09-22) and was far too low: on
# 2026-10-10 an untiled 4.23 MP decode with 28.8 GB free (estimate 13.4 GB)
# drove the box into swap. Measured afterwards with the allocator ("VAE
# decode peak" log line, 2026-10-11): 7183 and 7193 MB/MP.
UNTILED_VAE_MB_PER_MP = float(os.environ.get("UNTILED_VAE_MB_PER_MP", "7500"))
# Untiled encode (edit reference images) is far cheaper than a decode:
# measured 2394 MB/MP twice (2026-10-11). Using the decode rate here tiled
# reference encodes needlessly.
UNTILED_VAE_ENCODE_MB_PER_MP = float(os.environ.get("UNTILED_VAE_ENCODE_MB_PER_MP", "2600"))
# ...plus this much free memory left over, for estimate error and for other
# tenants growing during the decode.
UNTILED_VAE_MARGIN_MB = float(os.environ.get("UNTILED_VAE_MARGIN_MB", "2000"))
ALPHA_MODE = os.environ.get("ALPHA_MODE", "auto").lower()
# Alpha values at or below this count as "could be a real transparent pixel".
ALPHA_LOW_THRESHOLD = int(os.environ.get("ALPHA_LOW_THRESHOLD", "200"))
# An image is only treated as genuinely transparent if at least this fraction
# of pixels are at/below ALPHA_LOW_THRESHOLD (measured noise floor on opaque
# prompts is 0.000% - see _normalize_alpha).
ALPHA_MIN_FRACTION = float(os.environ.get("ALPHA_MIN_FRACTION", "0.0005"))
# Steps 2..N take the pipeline's kv-cache "cached" branch, which attaches a
# broadcast [B,1,1,S] padding mask. A non-None attn_mask disqualifies the
# flash SDPA backend, and torch then falls to the efficient kernel: measured
# on this GB10 at seq=16384, 146.17 ms vs 51.93 ms under cuDNN (2.8x). cuDNN
# is also marginally faster unmasked (47.43 vs 48.73 ms), so this is a free
# win either way - same math, better kernel, no quality change.
ATTENTION_BACKEND = os.environ.get("ATTENTION_BACKEND", "_native_cudnn")
_attention_backend_active = ATTENTION_BACKEND


def _cudnn_attention_active() -> bool:
    return "cudnn" in (_attention_backend_active or "").lower()


def _disable_cudnn_attention() -> None:
    global _attention_backend_active
    _pipe.transformer.set_attention_backend("native")
    _attention_backend_active = "native"


# Admission control. This box shares ONE 128 GB unified pool between the GPU
# and the OS, alongside other tenants (sglang/vLLM). On 2026-09-21 a 2K run
# drove the host into a global OOM that killed dbus, systemd and the LLM
# before us, and on 2026-10-09 a request admitted with 3-4 GB really available
# ran on swap. VAE tiling (VAE_TILING=auto) shrinks only the decode/encode
# peak; the denoising loop's activations can't be tiled, so a request still
# needs room for those before it starts.
#
# The need is sized per request instead of one flat floor, so small requests
# get through when the pool is tight:
#   n x (output_MP x DENOISE_MB_PER_MP + ref_MP x REF_MB_PER_MP)
#     + TILED_VAE_MB + ADMISSION_MARGIN_MB
# ref_MP = ~EDIT_OUTPUT_RESOLUTION^2 per reference image, since the pipeline
# resizes every reference to that area. A request that
# doesn't fit waits (polling) up to ADMISSION_WAIT_SECONDS from arrival, and
# only then gets a 503 that names the largest size that would fit right now.
# Keep the wait well under the agent's HTTP timeout (image_gen.timeout, 180 s)
# minus the generation itself (~65 s at 4 MP).
#
# Measured 2026-10-11 with the allocator ("Request peak" log line):
# - the denoising loop itself is cheap: a whole tiled 2048x2048 generation
#   peaked at +548 MB, 526 MB of it the tiled decode, i.e. < 100 MB for
#   4.19 MP of denoising. 250 = ~2.5x headroom. (The "6 GB transient" from
#   2026-09-22 was most likely the untiled decode.)
DENOISE_MB_PER_MP = float(os.environ.get("DENOISE_MB_PER_MP", "250"))
# - reference images are what costs: the pipeline caches their keys/values
#   for every block for the whole request (32 blocks x K+V x 4096 x bf16 =
#   ~0.5 MB per token, ~2.15 GB per MP). Measured: ~2.1 GB still held at
#   decode time in a 1-reference edit vs 22 MB in a generation.
REF_MB_PER_MP = float(os.environ.get("REF_MB_PER_MP", "2400"))
# Peak of a tiled VAE decode/encode: measured 526-530 MB at 4.2 MP. The
# decode falls back to tiled whenever an untiled one wouldn't fit, so this is
# the guaranteed cost.
TILED_VAE_MB = float(os.environ.get("TILED_VAE_MB", "1000"))
ADMISSION_MARGIN_MB = float(os.environ.get("ADMISSION_MARGIN_MB", "2000"))
ADMISSION_WAIT_SECONDS = float(os.environ.get("ADMISSION_WAIT_SECONDS", "60"))
ADMISSION_POLL_SECONDS = float(os.environ.get("ADMISSION_POLL_SECONDS", "5"))
# Smallest size the 503 message will suggest.
MIN_SUGGESTED_PIXELS = 512 * 512
# One-shot diagnostic: how many attention calls actually carry a mask.
MASK_PROBE_CALLS = int(os.environ.get("MASK_PROBE_CALLS", "64"))
# /debug/profile is a diagnostic endpoint, off unless explicitly enabled: it
# runs a real generation under torch.profiler (several x slower) and holds the
# pipeline lock for the duration.
ENABLE_PROFILE_ENDPOINT = os.environ.get("ENABLE_PROFILE_ENDPOINT", "1").lower() not in ("0", "false", "no")
# Edit-source images are VAE-encoded as context at their own resolution
# regardless of the requested output size, so an oversized upload (e.g. a
# 4080x2296 phone photo) inflates attention/VAE memory independent of any
# output width/height cap - same failure mode that OOM-killed the
# co-located vLLM process on the old FLUX.2-klein stack (2026-08-24).
MAX_EDIT_LONG_EDGE = int(os.environ.get("MAX_EDIT_LONG_EDGE", "2048"))
# Max number of reference images per edit request. The agent-side cap
# (image_gen.max_input_images) is configurable via env IMAGEGEN_MAX_INPUT_IMAGES.
MAX_EDIT_IMAGES = int(os.environ.get("MAX_EDIT_IMAGES", "8"))

_pipe = None
_lock = None  # type: ignore[var-annotated]
# Step count of the checkpoint's own sample_sigmas schedule (Turbo: 8), or
# None when the request's `steps` is honoured (base model).
_fixed_schedule_steps: Optional[int] = None
_load_error: Optional[str] = None
# Last auto-mode tiling decision, reported by /health.
_last_vae_decision: Optional[dict] = None


def _round_to_32(x: int) -> int:
    return max(32, round(x / 32) * 32)


def _parse_size(size: Optional[str]) -> tuple[int, int]:
    raw = size or DEFAULT_SIZE
    try:
        w_str, h_str = raw.lower().split("x")
        width, height = int(w_str), int(h_str)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"invalid size: {size!r}, expected e.g. '1024x1024'")
    if width <= 0 or height <= 0 or width > MAX_SIZE or height > MAX_SIZE:
        raise HTTPException(status_code=400, detail=f"size out of range: {raw!r} (max edge {MAX_SIZE})")
    # QwenImage21Pipeline floors width/height to a multiple of 32 internally
    # (vae_scale_factor * 2); round here so the caller is told the real size.
    width, height = _round_to_32(width), _round_to_32(height)
    if width > MAX_SIZE or height > MAX_SIZE:
        raise HTTPException(status_code=400, detail=f"size {width}x{height} exceeds max edge {MAX_SIZE} after rounding to multiple of 32")
    if width * height > MAX_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"size {width}x{height} = {width * height / 1e6:.1f} MP exceeds the {MAX_PIXELS / 1e6:.1f} MP cap",
        )
    return width, height


def _install_mask_probe() -> None:
    """Log once whether attention is actually being called with a mask.

    This decides whether ATTENTION_BACKEND matters at all: if the prompt is
    unpadded, `encoder_hidden_states_mask` is None, no mask is built, and the
    default backend was already running flash at full speed. Samples only the
    first MASK_PROBE_CALLS calls, then costs one boolean compare per call."""
    try:
        from diffusers.models import attention_dispatch as ad
        from diffusers.models.transformers import transformer_qwenimage21 as t21
    except Exception as exc:
        LOG.info(f"mask probe not installed: {exc}")
        return

    original = ad.dispatch_attention_fn
    state = {"calls": 0, "masked": 0, "shapes": set(), "done": False}

    def probe(*args, **kwargs):
        if not state["done"]:
            mask = kwargs.get("attn_mask")
            state["calls"] += 1
            if mask is not None:
                state["masked"] += 1
                if hasattr(mask, "shape"):
                    state["shapes"].add(tuple(mask.shape))
            if state["calls"] >= MASK_PROBE_CALLS:
                state["done"] = True
                LOG.info(
                    f"ATTENTION MASK PROBE: {state['masked']}/{state['calls']} calls carried a mask; "
                    f"shapes={sorted(state['shapes']) or 'none'} "
                    f"(masked calls are the ones ATTENTION_BACKEND={ATTENTION_BACKEND} accelerates)"
                )
        return original(*args, **kwargs)

    # The transformer imported the symbol into its own namespace, so patching
    # only the dispatch module would miss every real call.
    ad.dispatch_attention_fn = probe
    t21.dispatch_attention_fn = probe


def _host_available_mb() -> Optional[float]:
    """MemAvailable from /proc/meminfo (the host's, containers see it too).

    On this unified-memory box torch.cuda.mem_get_info() overstates headroom:
    on 2026-10-09 it let a request through (>= 12 GB "free") while
    MemAvailable was 3.2-4.4 GB, i.e. the request ran on swap."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cuda_memory_info_mb() -> Optional[tuple[float, float]]:
    """(free, total) MB as the CUDA driver reports them."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / (1024 * 1024), total / (1024 * 1024)
    except Exception:
        pass
    return None


def _memory_info_mb() -> Optional[tuple[float, float]]:
    """(free, total) MB of the unified pool: free is the smaller of the CUDA
    driver's view and the host's MemAvailable, so neither can overstate it."""
    cuda = _cuda_memory_info_mb()
    host_free = _host_available_mb()
    if cuda is None:
        return None
    free, total = cuda
    if host_free is not None:
        free = min(free, host_free)
    return free, total


def _request_cost_mb(output_pixels: int, n: int, n_refs: int) -> float:
    """Estimated peak growth of one request: denoising the output, the
    references' KV cache, and a tiled VAE pass (no margin)."""
    ref_pixels = n_refs * EDIT_OUTPUT_RESOLUTION * EDIT_OUTPUT_RESOLUTION
    per_image = output_pixels / 1e6 * DENOISE_MB_PER_MP + ref_pixels / 1e6 * REF_MB_PER_MP
    return n * per_image + TILED_VAE_MB


def _request_need_mb(output_pixels: int, n: int, n_refs: int) -> float:
    """Free pool memory a request needs to start."""
    return _request_cost_mb(output_pixels, n, n_refs) + ADMISSION_MARGIN_MB


def _largest_fitting_size(free_mb: float, n: int, n_refs: int, aspect: float) -> Optional[tuple[int, int]]:
    """Largest WxH at `aspect` (w/h) whose need fits into free_mb, floored to
    multiples of 32, or None if not even MIN_SUGGESTED_PIXELS fits."""
    import math

    ref_mb = n_refs * EDIT_OUTPUT_RESOLUTION * EDIT_OUTPUT_RESOLUTION / 1e6 * REF_MB_PER_MP
    per_image_mb = (free_mb - TILED_VAE_MB - ADMISSION_MARGIN_MB) / n - ref_mb
    pixels = min(per_image_mb / DENOISE_MB_PER_MP * 1e6, MAX_PIXELS)
    if pixels < MIN_SUGGESTED_PIXELS:
        return None
    height = math.sqrt(pixels / aspect)
    width = min(height * aspect, MAX_SIZE)
    height = min(height, MAX_SIZE)
    return max(32, int(width) // 32 * 32), max(32, int(height) // 32 * 32)


async def _await_memory_headroom(
    output_pixels: int, n: int, n_refs: int, aspect: float, deadline: float
) -> None:
    """Wait until the shared pool has room for this request, or 503 at the
    deadline. Runs under _lock, so requests behind it queue in order."""
    import asyncio

    need_mb = _request_need_mb(output_pixels, n, n_refs)
    waited_from = time.monotonic()
    while True:
        info = _memory_info_mb()
        if info is None:
            return
        free_mb, total_mb = info
        if free_mb >= need_mb:
            if time.monotonic() - waited_from > 1:
                LOG.info(f"Memory freed up after {time.monotonic() - waited_from:.0f}s: {free_mb:.0f} MB free, need {need_mb:.0f} MB")
            return
        if time.monotonic() >= deadline:
            break
        if time.monotonic() - waited_from < ADMISSION_POLL_SECONDS:
            LOG.info(
                f"Waiting for memory: {free_mb:.0f} MB free of {total_mb:.0f} MB, need {need_mb:.0f} MB "
                f"({output_pixels / 1e6:.2f} MP output, n={n}, {n_refs} reference image(s)); "
                f"up to {max(0.0, deadline - time.monotonic()):.0f}s"
            )
        await asyncio.sleep(ADMISSION_POLL_SECONDS)

    fit = _largest_fitting_size(free_mb, n, n_refs, aspect)
    hint = (
        f"The largest size that fits right now is about {fit[0]}x{fit[1]} - retry with that size."
        if fit
        else "Not even 512x512 fits right now - retry later."
    )
    LOG.warning(f"Refusing request after {time.monotonic() - waited_from:.0f}s: {free_mb:.0f} MB free, need {need_mb:.0f} MB; suggested {fit}")
    raise HTTPException(
        status_code=503,
        detail=(
            f"insufficient GPU memory: {free_mb:.0f} MB free, {need_mb:.0f} MB required for this request "
            f"(waited {time.monotonic() - waited_from:.0f}s; another tenant on this shared box is using the pool). "
            f"{hint}"
        ),
    )


def _run_measured(label: str, estimate_mb: float, fn):
    """Run one pipeline call and log its real allocator peak next to the
    admission estimate (_request_cost_mb, without margin). Includes the VAE
    passes; when one ran untiled its own peak (see the "VAE ... peak" line)
    dominates, so compare against tiled runs."""
    import torch

    torch.cuda.reset_peak_memory_stats()
    base_reserved = torch.cuda.memory_reserved()
    try:
        return fn()
    finally:
        peak_mb = (torch.cuda.max_memory_reserved() - base_reserved) / (1024 * 1024)
        LOG.info(
            f"Request peak ({label}): +{peak_mb:.0f} MB reserved "
            f"(admission estimate {estimate_mb:.0f} MB without margin)"
        )


def _configure_vae_memory(pipe) -> None:
    """Enable VAE tiling/slicing - the single biggest peak-memory lever here.

    The DiT denoising loop is the *steady* cost (~6 GB of activations at
    2048x2048). The VAE decode that runs after the last step is the *peak*:
    Qwen-Image's VAE is a video VAE (3D convs) and decodes the whole frame at
    once, holding full-resolution intermediates (~1 GB per tensor at 2048x2048
    bf16, several live at a time plus conv workspace). Measured on the box
    2026-09-22: host free memory sat at 18 GB through the whole denoising
    loop, then fell to 6.7 GB in one steep drop at decode time - ~11 GB of
    peak that admission control never saw coming, and the proximate cause of
    the OOM kills.

    Tiling decodes in overlapping blended tiles instead (peak ~1-2 GB);
    slicing decodes n>1 images one at a time instead of as a batch.

    In VAE_TILING=auto mode tiling is enabled here (so the tile parameters
    are set) and _install_adaptive_tiling then switches it off per call when
    the pool has room for an untiled pass.
    """
    if VAE_TILING == "never":
        LOG.info("VAE tiling disabled by VAE_TILING=0 - expect a large decode-time memory peak")
        return
    for name in ("enable_vae_tiling", "enable_vae_slicing"):
        # Prefer the pipeline-level helpers; fall back to the VAE's own
        # methods across diffusers versions. Never fail startup over this.
        try:
            getattr(pipe, name)()
            LOG.info(f"{name}() enabled")
            continue
        except Exception:
            pass
        vae_method = name.replace("_vae", "")
        try:
            getattr(pipe.vae, vae_method)()
            LOG.info(f"vae.{vae_method}() enabled")
        except Exception as exc:
            LOG.warning(f"Could not enable {name}: {exc}")
    if VAE_TILING == "auto":
        _install_adaptive_tiling(pipe)


def _install_adaptive_tiling(pipe) -> None:
    """Decide tiled vs untiled per VAE call from the free pool right now.

    Admission control runs before the request, but the decode comes minutes
    later, after other tenants may have grown - so measure immediately before
    each decode/encode (edits encode their reference images too). Requests are
    serialized by _lock, so flipping vae.use_tiling here can't race. Falls
    back to always-tiled if this diffusers VAE has no use_tiling flag.
    """
    vae = pipe.vae
    if not getattr(vae, "use_tiling", False):
        LOG.warning("VAE has no active use_tiling flag - adaptive tiling unavailable, staying tiled")
        return
    scale = int(getattr(pipe, "vae_scale_factor", 16) or 16)
    orig_decode, orig_encode = vae.decode, vae.encode

    def _choose(op: str, pixels: int) -> None:
        global _last_vae_decision
        import torch

        # Return blocks the denoising loop left in the caching allocator to
        # the pool first; mem_get_info doesn't count them as free.
        torch.cuda.empty_cache()
        info = _memory_info_mb()
        rate = UNTILED_VAE_ENCODE_MB_PER_MP if op == "encode" else UNTILED_VAE_MB_PER_MP
        need_mb = pixels / 1e6 * rate + UNTILED_VAE_MARGIN_MB
        free_mb = info[0] if info else 0.0
        untiled = info is not None and free_mb >= need_mb
        vae.use_tiling = not untiled
        _last_vae_decision = {
            "op": op,
            "tiled": not untiled,
            "pixels": pixels,
            "free_mb": round(free_mb),
            "need_mb": round(need_mb),
            "at": int(time.time()),
        }
        LOG.info(
            f"VAE {op}: {'tiled' if not untiled else 'untiled'} "
            f"({pixels / 1e6:.2f} MP, {free_mb:.0f} MB free, untiled needs {need_mb:.0f} MB)"
        )

    def _measured(op: str, fn, *args, **kwargs):
        # UNTILED_VAE_MB_PER_MP was estimated from host free memory sampled
        # during one decode (2026-09-22) and turned out far too low: on
        # 2026-10-10 an untiled 4.23 MP decode with 28.8 GB free (estimate:
        # 13.4 GB) drove the box into swap. Record the allocator's real peak
        # so the estimate can be set from data.
        import torch

        torch.cuda.reset_peak_memory_stats()
        base_reserved = torch.cuda.memory_reserved()
        base_allocated = torch.cuda.memory_allocated()
        try:
            return fn(*args, **kwargs)
        finally:
            peak_reserved_mb = (torch.cuda.max_memory_reserved() - base_reserved) / (1024 * 1024)
            peak_allocated_mb = (torch.cuda.max_memory_allocated() - base_allocated) / (1024 * 1024)
            decision = _last_vae_decision or {}
            decision["peak_reserved_mb"] = round(peak_reserved_mb)
            decision["peak_allocated_mb"] = round(peak_allocated_mb)
            mp = decision.get("pixels", 0) / 1e6
            LOG.info(
                f"VAE {op} peak: +{peak_reserved_mb:.0f} MB reserved, +{peak_allocated_mb:.0f} MB allocated "
                f"({'tiled' if decision.get('tiled') else 'untiled'}, {mp:.2f} MP"
                + (f", {peak_reserved_mb / mp:.0f} MB/MP" if mp else "")
                + ")"
            )

    def decode(z, *args, **kwargs):
        # Slicing decodes one image at a time, so size the estimate per image.
        _choose("decode", z.shape[-2] * z.shape[-1] * scale * scale)
        try:
            return _measured("decode", orig_decode, z, *args, **kwargs)
        finally:
            vae.use_tiling = True

    def encode(x, *args, **kwargs):
        _choose("encode", x.shape[-2] * x.shape[-1])
        try:
            return _measured("encode", orig_encode, x, *args, **kwargs)
        finally:
            vae.use_tiling = True

    vae.decode = decode
    vae.encode = encode
    LOG.info(
        f"Adaptive VAE tiling: untiled when free >= {UNTILED_VAE_MB_PER_MP:.0f} MB/MP (decode), "
        f"{UNTILED_VAE_ENCODE_MB_PER_MP:.0f} MB/MP (encode) "
        f"+ {UNTILED_VAE_MARGIN_MB:.0f} MB margin"
    )


def _upcast_linear_forward(self, x):
    """nn.Linear forward for fp8-stored weights: upcast to the activation dtype
    (bf16) per call. Same compute as the bf16 encoder, half the resident bytes."""
    import torch.nn.functional as F

    bias = None if self.bias is None else self.bias.to(x.dtype)
    return F.linear(x, self.weight.to(x.dtype), bias)


def _load_fp8_text_encoder():
    """Build the Qwen3-VL text encoder from unsloth's FP8 file.

    The file is a plain cast (no scale tensors; its metadata says
    cast_backend=diffusers_layerwise): every nn.Linear weight+bias in both the
    language model and the vision tower is float8_e4m3fn, while norms,
    embed_tokens, patch_embed and pos_embed stay bf16. So: build the model on
    the meta device from IMAGE_MODEL_ID's config, assign the tensors as-is
    (the bf16 encoder is never materialized, so loading never peaks above the
    FP8 footprint), and patch only the fp8 Linears to upcast at compute time."""
    import types

    import torch
    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    path = hf_hub_download(TEXT_ENCODER_FP8_REPO, TEXT_ENCODER_FP8_FILE, token=HF_TOKEN)
    LOG.info(f"Loading FP8 text encoder from {TEXT_ENCODER_FP8_REPO}/{TEXT_ENCODER_FP8_FILE}")
    config = AutoConfig.from_pretrained(IMAGE_MODEL_ID, subfolder="text_encoder", token=HF_TOKEN)
    with init_empty_weights():
        text_encoder = Qwen3VLForConditionalGeneration._from_config(config, torch_dtype=torch.bfloat16)
    # strict: any key/layout drift between the file and this transformers
    # version fails startup (-> /health 503) instead of running half-random.
    text_encoder.load_state_dict(load_file(path, device="cuda"), strict=True, assign=True)
    # Buffers that aren't in the file (e.g. the rotary inv_freq) were created
    # on the CPU by init_empty_weights. The pipeline's device_map="cuda" load
    # doesn't move components passed in from outside, so move them here; the
    # weights are already on cuda, so this copies only those small buffers.
    text_encoder.to("cuda")
    patched = 0
    for module in text_encoder.modules():
        if isinstance(module, torch.nn.Linear) and module.weight.dtype == torch.float8_e4m3fn:
            module.forward = types.MethodType(_upcast_linear_forward, module)
            patched += 1
    text_encoder.requires_grad_(False).eval()
    LOG.info(f"FP8 text encoder loaded: {patched} Linear layers upcast to bf16 at compute time")
    return text_encoder


def _check_sample_sigmas(pipe) -> None:
    """Fail startup if the checkpoint ships a sampling schedule the installed
    diffusers can't read.

    Turbo's 8-step schedule lives in model_index.json as `sample_sigmas`.
    A diffusers build older than PR #14950 drops that key with a warning and
    samples a plain linspace at the request's step count instead - a distilled
    model run on the wrong schedule, which looks like a quality problem
    rather than an error. Docker's layer cache makes that easy to hit (the
    git-main pip install layer is reused), so check instead of trusting it."""
    global _fixed_schedule_steps
    import json

    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(IMAGE_MODEL_ID, "model_index.json", token=HF_TOKEN)) as f:
        wanted = json.load(f).get("sample_sigmas")
    loaded = pipe.config.get("sample_sigmas") if hasattr(pipe.config, "get") else None
    if wanted and not loaded:
        raise RuntimeError(
            f"{IMAGE_MODEL_ID} configures sample_sigmas but this diffusers ignores them "
            f"(needs diffusers PR #14950) - rebuild the image with --no-cache"
        )
    if loaded:
        _fixed_schedule_steps = len(loaded)
        LOG.info(f"Checkpoint schedule: {_fixed_schedule_steps} fixed steps (request `steps` is ignored)")


def _effective_steps(requested: int) -> int:
    return _fixed_schedule_steps or requested


def _load_pipeline() -> None:
    """Load the QwenImage21Pipeline once at startup. Sets _load_error instead
    of raising so /health can report a clear "not loaded" status rather than
    the container just crash-looping on a slow/broken GPU box."""
    global _pipe, _lock, _load_error
    import asyncio

    import torch
    from diffusers import QwenImage21Pipeline

    try:
        LOG.info(f"Loading {IMAGE_MODEL_ID} (quant_mode={QUANT_MODE}, steps={NUM_INFERENCE_STEPS}, true_cfg_scale={TRUE_CFG_SCALE})")
        if TEXT_ENCODER_QUANT not in ("fp8", "none"):
            raise ValueError(f"TEXT_ENCODER_QUANT must be 'fp8' or 'none', got {TEXT_ENCODER_QUANT!r}")
        components = {}
        if TEXT_ENCODER_QUANT == "fp8":
            components["text_encoder"] = _load_fp8_text_encoder()
        # Load straight onto the GPU. from_pretrained(...).to("cuda") builds a
        # CPU copy first and then copies it over; on unified memory both
        # copies live in the same pool, so startup briefly needed ~2x the
        # weights - worst exactly when restarting after an OOM kill. Older
        # diffusers without pipeline-level device_map="cuda" take the old path.
        try:
            pipe = QwenImage21Pipeline.from_pretrained(
                IMAGE_MODEL_ID, torch_dtype=torch.bfloat16, token=HF_TOKEN, device_map="cuda", **components
            )
            LOG.info("Pipeline loaded with device_map='cuda'")
            # Anything left on the CPU fails the first request with a
            # device-mismatch error; move it now instead.
            for name, component in pipe.components.items():
                if not isinstance(component, torch.nn.Module):
                    continue
                tensors = list(component.parameters()) + list(component.buffers())
                if any(t.device.type == "cpu" for t in tensors):
                    LOG.warning(f"{name} had tensors on cpu after device_map load; moving it to cuda")
                    component.to("cuda")
        except (ValueError, NotImplementedError, TypeError) as exc:
            LOG.warning(f"device_map='cuda' not supported ({exc}); loading on CPU and moving to cuda")
            pipe = QwenImage21Pipeline.from_pretrained(
                IMAGE_MODEL_ID, torch_dtype=torch.bfloat16, token=HF_TOKEN, **components
            ).to("cuda")
        if ATTENTION_BACKEND and ATTENTION_BACKEND.lower() != "default":
            try:
                pipe.transformer.set_attention_backend(ATTENTION_BACKEND)
                LOG.info(f"Attention backend set to {ATTENTION_BACKEND}")
            except Exception as exc:
                # Never fail startup over an optimization; the default backend
                # is correct, just slower on the masked path.
                LOG.warning(f"Could not set attention backend {ATTENTION_BACKEND!r}, using default: {exc}")
        if TEXT_ENCODER_QUANT == "fp8":
            # Guard against the pipeline loader re-casting the passed-in encoder
            # to torch_dtype, which would silently undo the memory saving.
            fp8_params = sum(p.numel() for p in pipe.text_encoder.parameters() if p.dtype == torch.float8_e4m3fn)
            LOG.info(f"Text encoder after pipeline load: {fp8_params / 1e9:.2f}B params stored as fp8")
        _check_sample_sigmas(pipe)
        _configure_vae_memory(pipe)
        _install_mask_probe()
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


app = FastAPI(title="qwen-image-2.1-server", lifespan=lifespan)


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
    info = _memory_info_mb()
    cuda_info = _cuda_memory_info_mb()
    host_available = _host_available_mb()
    return {
        "status": "ok",
        "model": IMAGE_MODEL_ID,
        "quant_mode": QUANT_MODE,
        "vae_tiling": VAE_TILING,
        "untiled_vae_mb_per_mp": UNTILED_VAE_MB_PER_MP,
        "untiled_vae_encode_mb_per_mp": UNTILED_VAE_ENCODE_MB_PER_MP,
        "untiled_vae_margin_mb": UNTILED_VAE_MARGIN_MB,
        "last_vae_decision": _last_vae_decision,
        "steps": _effective_steps(NUM_INFERENCE_STEPS),
        "fixed_schedule": _fixed_schedule_steps is not None,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "attention_backend": _attention_backend_active,
        "resident_memory_mb": _resident_memory_mb(),
        "pool_free_mb": round(info[0]) if info else None,
        "pool_total_mb": round(info[1]) if info else None,
        "cuda_free_mb": round(cuda_info[0]) if cuda_info else None,
        "host_available_mb": round(host_available) if host_available is not None else None,
        "admission": {
            "denoise_mb_per_mp": DENOISE_MB_PER_MP,
            "ref_mb_per_mp": REF_MB_PER_MP,
            "tiled_vae_mb": TILED_VAE_MB,
            "margin_mb": ADMISSION_MARGIN_MB,
            "wait_seconds": ADMISSION_WAIT_SECONDS,
            "need_mb_1024x1024": round(_request_need_mb(1024 * 1024, 1, 0)),
            "need_mb_2048x2048": round(_request_need_mb(2048 * 2048, 1, 0)),
            "need_mb_edit_1mp_1ref": round(_request_need_mb(EDIT_OUTPUT_RESOLUTION**2, 1, 1)),
        },
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
                "owned_by": "qwen",
                "quant_mode": QUANT_MODE,
            }
        ],
    }


def _normalize_alpha(image):
    """Handle the RGBA output this model always produces (see ALPHA_MODE).

    The 4-channel VAE means even "a photo of a cat" comes back as RGBA. An
    opaque alpha channel is pure overhead and a hazard for downstream chat
    clients, so drop it unless the model actually painted transparency."""
    from PIL import Image

    if image.mode != "RGBA" or ALPHA_MODE == "keep":
        return image
    if ALPHA_MODE == "flatten":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    # "auto": keep real transparency, drop an effectively-opaque alpha channel.
    #
    # A single-pixel min() check here is too fragile: measured 40-step
    # outputs on plain opaque prompts (bicycle, astronaut, macro dragonfly)
    # had a minimum alpha of 241-249 with ZERO pixels below 200 - that's
    # decode noise, not a painted transparent subject, but it was enough to
    # flip getextrema()[0] under the old >=250 threshold and keep the whole
    # image as needless RGBA. Require a real, non-trivial LOW-alpha area
    # instead of reacting to one outlier pixel.
    alpha = image.getchannel("A")
    hist = alpha.histogram()
    total = sum(hist)
    low_alpha_fraction = sum(hist[: ALPHA_LOW_THRESHOLD + 1]) / total if total else 0.0
    if low_alpha_fraction < ALPHA_MIN_FRACTION:
        return image.convert("RGB")
    LOG.info(
        f"Output has real transparency ({low_alpha_fraction*100:.2f}% of pixels "
        f"<= alpha {ALPHA_LOW_THRESHOLD}); returning RGBA PNG"
    )
    return image


def _image_to_b64_png(image) -> str:
    buf = io.BytesIO()
    _normalize_alpha(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _clamp_to_long_edge(image, max_long_edge: int):
    """Downscale `image` (preserving aspect ratio) so its longest edge is
    <= max_long_edge. Dimensions are rounded to a multiple of 32 (this
    pipeline's vae_scale_factor * 2). No-op if already within bounds."""
    from PIL import Image

    width, height = image.width, image.height
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    new_width = max(32, round(width * scale / 32) * 32)
    new_height = max(32, round(height * scale / 32) * 32)
    LOG.info(f"Downscaling oversized edit input {width}x{height} -> {new_width}x{new_height}")
    return image.resize((new_width, new_height), Image.LANCZOS)


@app.post("/v1/images/generations")
async def generate(payload: dict) -> JSONResponse:
    """OpenAI-shaped image generation.

    Body: {"prompt": str, "size": "WxH", "n": int, "response_format": "b64_json",
           "steps": int, "seed": int}  (steps/seed are non-standard extras,
           same names as the FLUX.2-klein server for drop-in compatibility)
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
    deadline = time.monotonic() + ADMISSION_WAIT_SECONDS

    import torch

    generator = torch.Generator(device="cuda").manual_seed(int(seed)) if seed is not None else None

    async with _lock:
        await _await_memory_headroom(width * height, n, 0, width / height, deadline)
        start = time.monotonic()
        LOG.info(f"Generating: prompt={prompt!r} size={width}x{height} n={n} steps={_effective_steps(steps)} seed={seed}")
        def _run():
            return _pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_images_per_prompt=n,
                num_inference_steps=steps,
                true_cfg_scale=TRUE_CFG_SCALE,
                generator=generator,
            )

        estimate_mb = _request_cost_mb(width * height, n, 0)
        try:
            try:
                result = _run_measured("generate", estimate_mb, _run)
            except RuntimeError as exc:
                if "mha_graph" not in str(exc) or not _cudnn_attention_active():
                    raise
                # cuDNN's SDPA graph can fail to execute at runtime (seen at
                # step 0 of a 1024x1024 request on 2026-09-30) even though it
                # built fine at startup. Fall back to torch's default backend
                # for the rest of the process life (correct, just slower on
                # masked calls) and retry this request once.
                LOG.warning(f"cuDNN attention failed ({exc}); falling back to default backend and retrying")
                _disable_cudnn_attention()
                torch.cuda.empty_cache()
                if generator is not None:
                    generator.manual_seed(int(seed))
                result = _run_measured("generate", estimate_mb, _run)
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


@app.post("/debug/profile")
async def debug_profile(payload: dict) -> JSONResponse:
    """Profile one generation and return the hottest CUDA kernels.

    Answers the open question: per-step cost is ~1.55x the block-benchmark
    floor at every resolution, which rules out attention scaling and points
    at per-block overhead. This names the actual kernels instead of guessing.

    Runs on the already-loaded pipeline, so it costs no extra VRAM. Keep
    `steps` small - profiling adds significant overhead per step.
    """
    if not ENABLE_PROFILE_ENDPOINT:
        raise HTTPException(status_code=403, detail="profiling endpoint disabled (ENABLE_PROFILE_ENDPOINT=0)")
    if _pipe is None:
        raise HTTPException(status_code=503, detail="pipeline not loaded")

    prompt = payload.get("prompt") or "a photograph of a red cube on a white table"
    width, height = _parse_size(payload.get("size"))
    steps = int(payload.get("steps") or 2)
    row_limit = int(payload.get("row_limit") or 25)

    import torch
    from torch.profiler import ProfilerActivity, profile

    async with _lock:
        # Diagnostic: no waiting, 503 straight away if it doesn't fit.
        await _await_memory_headroom(width * height, 1, 0, width / height, time.monotonic())
        LOG.info(f"Profiling: size={width}x{height} steps={steps}")
        try:
            # Warm up first: one-off autotune/allocation on the first call
            # would otherwise dominate the profile and mask the steady state.
            # num_inference_steps=1 is NOT safe here: the dynamic-shifting
            # scheduler's stretched_t = 1 - (one_minus_z / scale_factor)
            # divides by ~0 at a single step, producing NaNs (seen as
            # "invalid value encountered in divide" then "... in cast" in the
            # 2026-09-21 run). 2 is the minimum that keeps the schedule sane.
            _pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=2,
                true_cfg_scale=TRUE_CFG_SCALE,
            )
            torch.cuda.synchronize()

            start = time.monotonic()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                _pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    true_cfg_scale=TRUE_CFG_SCALE,
                )
                torch.cuda.synchronize()
            elapsed = time.monotonic() - start

            table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit)

            # Bucket kernels into attention / gemm / launch-overhead /
            # elementwise-other, so the unexplained overhead can be
            # attributed without reading every row. NOTE: the op-name row
            # (e.g. "aten::mm") and the underlying kernel-name row (e.g. the
            # nvjet GEMM kernel) both get self time in key_averages() - they
            # are DIFFERENT rows for the SAME work, not parent/child, so this
            # sum double-counts across that pair. Treat the percentages as
            # roughly-relative, not as an exact partition of total_cuda_us.
            buckets = {"attention": 0.0, "gemm": 0.0, "launch_overhead": 0.0, "elementwise/other": 0.0}
            total_cuda_us = 0.0
            for evt in prof.key_averages():
                cuda_us = float(getattr(evt, "self_device_time_total", 0) or 0)
                if cuda_us <= 0:
                    continue
                total_cuda_us += cuda_us
                name = evt.key.lower()
                if any(tag in name for tag in ("attn", "attention", "flash", "cudnn_infer", "fmha", "sdpa")):
                    buckets["attention"] += cuda_us
                elif "command buffer" in name or "launch" in name:
                    # CPU-side wait for the GPU command queue to drain -
                    # symptomatic of many small kernel launches (32 blocks x
                    # several ops/block), not compute time.
                    buckets["launch_overhead"] += cuda_us
                elif any(
                    tag in name
                    for tag in (
                        "gemm", "matmul", "aten::mm", "aten::bmm", "nvjet", "mma", "wgmma",
                        "cutlass", "sm90", "sm100", "sm120", "sm121", "addmm", "aten::linear", "conv",
                    )
                ):
                    buckets["gemm"] += cuda_us
                else:
                    buckets["elementwise/other"] += cuda_us
        except Exception as exc:
            LOG.exception(f"Profiling failed: {exc}")
            return JSONResponse({"error": str(exc)}, status_code=500)
        finally:
            torch.cuda.empty_cache()

    share = {
        k: (round(100 * v / total_cuda_us, 1) if total_cuda_us else None)
        for k, v in buckets.items()
    }
    return JSONResponse({
        "size": f"{width}x{height}",
        "steps": _effective_steps(steps),
        "elapsed_s": round(elapsed, 2),
        "per_step_s": round(elapsed / _effective_steps(steps), 2) if steps else None,
        "total_cuda_ms": round(total_cuda_us / 1000, 1),
        "bucket_share_pct": share,
        "table": table,
    })


@app.post("/v1/images/edits")
async def edit(
    image: list[UploadFile] = File(...),
    prompt: str = Form(...),
    size: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    response_format: Optional[str] = Form("b64_json"),
    steps: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
) -> JSONResponse:
    """OpenAI-shaped image editing (multipart), routed to Qwen-Image-2.1's
    unified image-conditioned generation path (the pipeline accepts input
    images alongside the prompt; there is no separate edit model).

    Accepts one or more reference images as repeated `image` form fields
    (the OpenAI multi-file convention). A single `image` part binds to a
    one-element list, so legacy single-image clients keep working."""
    if _pipe is None or _lock is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error or 'loading'}")
    if response_format != "b64_json":
        raise HTTPException(status_code=400, detail=f"unsupported response_format: {response_format}")

    from PIL import Image

    if not image:
        raise HTTPException(status_code=400, detail="at least one image is required")
    if len(image) > MAX_EDIT_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"too many images: {len(image)} (max {MAX_EDIT_IMAGES})",
        )

    input_pils = []
    for upload in image:
        input_bytes = await upload.read()
        try:
            pil = Image.open(io.BytesIO(input_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not decode input image: {exc}")
        input_pils.append(_clamp_to_long_edge(pil, MAX_EDIT_LONG_EDGE))
    # Only honour an explicit size. With width/height left as None the pipeline
    # derives them from the reference image's aspect ratio at
    # EDIT_OUTPUT_RESOLUTION - which is what the official edit examples do, and
    # it keeps the reference latents and the target grid at the same scale.
    # Forcing the upload's own (possibly 2K) dimensions here instead makes the
    # model paint a large target from a ~1 MP reference and degrades the edit.
    width, height = _parse_size(size) if size else (None, None)
    n_val = int(n or 1)
    if n_val < 1 or n_val > 4:
        raise HTTPException(status_code=400, detail="n must be between 1 and 4")
    steps_val = int(steps or NUM_INFERENCE_STEPS)
    deadline = time.monotonic() + ADMISSION_WAIT_SECONDS
    # Without an explicit size the pipeline renders ~EDIT_OUTPUT_RESOLUTION^2
    # at the last reference's aspect ratio.
    aspect = input_pils[-1].width / input_pils[-1].height
    out_pixels = width * height if width else EDIT_OUTPUT_RESOLUTION * EDIT_OUTPUT_RESOLUTION
    estimate_mb = _request_cost_mb(out_pixels, n_val, len(input_pils))

    import torch

    generator = torch.Generator(device="cuda").manual_seed(int(seed)) if seed is not None else None

    async with _lock:
        await _await_memory_headroom(out_pixels, n_val, len(input_pils), aspect, deadline)
        start = time.monotonic()
        size_desc = f"{width}x{height}" if width else f"auto@{EDIT_OUTPUT_RESOLUTION}"
        LOG.info(f"Editing: prompt={prompt!r} size={size_desc} n={n_val} n_images={len(input_pils)} steps={_effective_steps(steps_val)} seed={seed}")
        try:
            result = _run_measured("edit", estimate_mb, lambda: _pipe(
                prompt=prompt,
                image=input_pils,
                width=width,
                height=height,
                num_images_per_prompt=n_val,
                num_inference_steps=steps_val,
                true_cfg_scale=TRUE_CFG_SCALE,
                output_resolution=EDIT_OUTPUT_RESOLUTION,
                generator=generator,
            ))
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
