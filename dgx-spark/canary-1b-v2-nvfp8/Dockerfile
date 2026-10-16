# canary-1b-v2 NVFP8 STT server — DGX Spark (Blackwell / GB10)
#
# Strategy:
#  - Base model (nvidia/canary-1b-v2, ~3.9 GB FP32 safetensors) is downloaded
#    from HF on first startup into the model-cache volume.
#  - At startup the Linear layers are quantized to NVFP8 (FP8 e4m3 weights +
#    per-1x128 block FP32 scales, ~0.7 GB) and cached under
#    /models/canary-1b-v2-nvfp8/ — subsequent starts skip the quantization.
#  - Compute runs in BF16 (quantized weights are dequantized on the fly per
#    layer; deterministic and version-robust). True FP8 GEMM kernels via
#    torch._scaled_mm can be added later as an optimization.
#  - Peak VRAM at inference (batch 1, chunked audio): ~1.5–2.5 GB.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev ffmpeg curl ca-certificates git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        "torch" \
        "torchaudio" \
        "fastapi==0.115.*" \
        "uvicorn[standard]==0.32.*" \
        "python-multipart==0.0.12" \
        "huggingface_hub>=0.20" \
        "numpy" \
        "librosa"

# Canary-1b-v2 (architectures: CanaryForConditionalGeneration) is only in
# recent transformers — install from git main so the model class ships.
RUN pip3 install --no-cache-dir --upgrade \
        "git+https://github.com/huggingface/transformers.git@main"

WORKDIR /app
COPY server.py quantize.py /app/

ENV CANARY_MODEL_ID=nvidia/canary-1b-v2 \
    CANARY_DEFAULT_LANG=de \
    CANARY_FALLBACK_LANG=en \
    CANARY_QUANT=1 \
    HF_HOME=/models/hf-cache \
    QUANT_CACHE_DIR=/models/canary-1b-v2-nvfp8 \
    PORT=8006

EXPOSE 8006

# First boot: ~4 GB download + quantization (~1–2 min). Give it plenty of room.
HEALTHCHECK --interval=30s --timeout=10s --start-period=1200s --retries=10 \
    CMD curl -fsS http://localhost:8006/health || exit 1

CMD ["python3", "server.py"]
