# nvcr.io/nvidia/pytorch is the only source of ARM64 + CUDA-13 PyTorch for
# GB10 (sm_121a) - CUDA 12.x toolchains cannot emit sm_121a. Pin to a tag
# known to carry a CUDA 13 / Blackwell-capable PyTorch build; bump deliberately.
FROM nvcr.io/nvidia/pytorch:25.11-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# QwenImage21Pipeline was merged into diffusers main on 2026-09-18 (PR
# #14804); the Turbo checkpoint's fixed sampling schedule (`sample_sigmas`)
# needs PR #14950 (2026-10-05). Pin to git main until it ships in a stable
# release, like the FLUX.2-klein stack does. Docker caches this layer, so an
# image built before 2026-10-05 keeps the old diffusers: rebuild with
# `NO_CACHE=1 ./start.sh` (or `docker build --no-cache`). Qwen3-VL-8B text encoder needs transformers >= 4.57.
RUN pip install --no-cache-dir \
        "git+https://github.com/huggingface/diffusers.git" \
        "transformers>=4.57" \
        "accelerate>=1.0" \
        # The NVCR base bundles an old torchao; diffusers main unconditionally
        # imports FqnToConfig from it. 0.15.0 is the newest release that
        # imports cleanly on the base image's torch (same pin as the
        # flux2-klein stack).
        "torchao==0.15.0" \
        "safetensors" \
        "pillow" \
        "fastapi==0.115.*" \
        "uvicorn[standard]==0.32.*" \
        "python-multipart==0.0.12" \
        "huggingface_hub>=0.20"

# Fail the build if the pipeline class or Turbo's sample_sigmas support is
# missing from this diffusers version. Without it Turbo silently samples the
# wrong schedule; server.py checks this again at load time.
RUN python3 -c "import inspect; from diffusers import QwenImage21Pipeline as P; \
assert 'sample_sigmas' in inspect.signature(P.__init__).parameters, \
'diffusers lacks PR #14950 (sample_sigmas) - rebuild with --no-cache'"

WORKDIR /app
COPY server.py /app/server.py

# Override at runtime:
#   IMAGE_MODEL_ID=Qwen/Qwen-Image-2.1-Turbo  (Qwen/Qwen-Image-2.1 = base model)
#   NUM_INFERENCE_STEPS=8    (ignored by Turbo's fixed 8-step schedule;
#                             set 40 for the base model, which is NOT distilled)
#   TRUE_CFG_SCALE=1.0       (model is sampled without guidance)
#   TEXT_ENCODER_QUANT=fp8   (FP8-stored Qwen3-VL encoder; none = bf16)
#   HF_TOKEN=hf_...         (optional; repo is not gated)
#   ATTENTION_BACKEND=_native_cudnn  (2.8x on the masked kv-cache path)
#   DENOISE_MB_PER_MP=250, REF_MB_PER_MP=2400, ADMISSION_WAIT_SECONDS=60  (size-aware admission)
# expandable_segments cuts allocator fragmentation across varying request
# sizes - this server sees 1024^2 through 2752x1536 in the same process, and
# the driver was throwing NV_ERR_NO_MEMORY for hours before the 2026-09-21
# host OOM.
ENV IMAGE_MODEL_ID=Qwen/Qwen-Image-2.1-Turbo \
    NUM_INFERENCE_STEPS=8 \
    TRUE_CFG_SCALE=1.0 \
    TEXT_ENCODER_QUANT=fp8 \
    ATTENTION_BACKEND=_native_cudnn \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PORT=8000

EXPOSE 8000

# Long start-period: first run downloads ~33 GB of weights before the model
# is ready to serve.
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python3", "server.py"]
