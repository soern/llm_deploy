# nvcr.io/nvidia/pytorch is the only source of ARM64 + CUDA-13 PyTorch for
# GB10 (sm_121a). Same base as dgx-spark/qwen-image-2.1; it ships
# torch >= 2.9, which MOSS-Image requires.
FROM nvcr.io/nvidia/pytorch:25.11-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Versions from the MOSS-Image model card. transformers >= 5.3 is needed for
# the Qwen3.5 text encoder (Qwen3_5Model); diffusers >= 0.38 for
# AutoencoderKLFlux2 (the FLUX.2 small VAE).
RUN pip install --no-cache-dir \
        "diffusers>=0.38" \
        "transformers>=5.3" \
        "accelerate>=1.10" \
        # The NVCR base bundles an old torchao that diffusers can't import;
        # same pin as qwen-image-2.1 and flux2-klein.
        "torchao==0.15.0" \
        "safetensors>=0.6" \
        "pillow>=10" \
        "fastapi==0.115.*" \
        "uvicorn[standard]==0.32.*" \
        "python-multipart==0.0.12" \
        "huggingface_hub"

# Fail the build, not the first request, if a dependency is too old.
RUN python3 -c "import torch, diffusers, transformers; \
from packaging.version import Version as V; \
assert V(torch.__version__.split('+')[0]) >= V('2.9'), torch.__version__; \
assert V(diffusers.__version__) >= V('0.38'), diffusers.__version__; \
assert V(transformers.__version__) >= V('5.3'), transformers.__version__; \
from diffusers import AutoencoderKLFlux2; from transformers import Qwen3_5Model"

WORKDIR /app
COPY server.py /app/server.py

# Override at runtime (see README):
#   IMAGE_MODEL_ID=OpenMOSS-Team/MOSS-Image-SFT   (…-Base = base model)
#   TEXT_ENCODER_QUANT=none   (bf16, ~27 GB resident; fp8 = ~19 GB)
#   NUM_INFERENCE_STEPS=50, GUIDANCE_SCALE=3.0, UNCOND_SKIP_LAYERS=22
ENV IMAGE_MODEL_ID=OpenMOSS-Team/MOSS-Image-SFT \
    TEXT_ENCODER_QUANT=none \
    NUM_INFERENCE_STEPS=50 \
    GUIDANCE_SCALE=3.0 \
    UNCOND_SKIP_LAYERS=22 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PORT=8000

EXPOSE 8000

# First run downloads ~27 GB of weights before the model can serve.
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=5 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["python3", "server.py"]
