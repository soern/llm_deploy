# nvcr.io/nvidia/pytorch is the only source of ARM64 + CUDA-13 PyTorch for
# GB10 (sm_121a) - CUDA 12.x toolchains cannot emit sm_121a. Pin to a tag
# known to carry a CUDA 13 / Blackwell-capable PyTorch build; bump deliberately.
FROM nvcr.io/nvidia/pytorch:25.11-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# FLUX.2 support is not yet in a diffusers release; install from git main.
# torchao/mslk need nightly cu130 aarch64 wheels for the NVFP4/MXFP8 inference
# workflow ops used below - if those wheels aren't available for this box,
# set QUANT_MODE=none at runtime as a fallback (no torchao quant path taken).
RUN pip install --no-cache-dir \
        "git+https://github.com/huggingface/diffusers.git" \
        "transformers>=4.46" \
        "accelerate>=1.0" \
        # The NVCR base image bundles an older torchao (<0.15); a bare
        # "torchao" would be a no-op. 0.15.0 is pinned exactly: it is the
        # newest release whose MX-format sources import cleanly on the
        # base image's torch (2.10.0a0 nv25.11). 0.16+ pulls in
        # torch.nn.functional.ScalingType/SwizzleType (a torch >= 2.11 API)
        # - in 0.16 behind a version guard that the 2.10.0a0 build passes
        # yet still crashes on, in 0.17/0.18 unconditionally.
        "torchao==0.15.0" \
        "mslk" \
        "safetensors" \
        "pillow" \
        "fastapi==0.115.*" \
        "uvicorn[standard]==0.32.*" \
        "python-multipart==0.0.12" \
        "huggingface_hub>=0.20"

# Fail the build, not the first remote run, if the torchao quant configs
# the server imports are missing from this base image's toolchain.
RUN python3 -c "from torchao.prototype.mx_formats.inference_workflow import \
        MXDynamicActivationMXWeightConfig, NVFP4DynamicActivationNVFP4WeightConfig"

WORKDIR /app
COPY server.py /app/server.py

# Override at runtime:
#   IMAGE_MODEL_ID=black-forest-labs/FLUX.2-klein-9B
#   QUANT_MODE=mxfp8 | nvfp4 | none
#   NUM_INFERENCE_STEPS=4  (klein is distilled)
#   GUIDANCE_SCALE=3.5
#   HF_TOKEN=hf_...        (if the model repo is gated)
ENV IMAGE_MODEL_ID=black-forest-labs/FLUX.2-klein-9B \
    QUANT_MODE=mxfp8 \
    NUM_INFERENCE_STEPS=4 \
    GUIDANCE_SCALE=3.5 \
    PORT=8000

EXPOSE 8000

# Long start-period: first run downloads ~40GB of weights and JIT-quantizes
# the transformer before the model is ready to serve.
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python3", "server.py"]
