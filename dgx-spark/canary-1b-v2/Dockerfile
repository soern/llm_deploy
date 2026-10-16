# canary-1b-v2 (model-default precision) STT server — DGX Spark (GB10)
#
# Loads nvidia/canary-1b-v2 exactly as released on HF:
#   - safetensors ship as FP32 (config dtype: float32), ~3.9 GB
#   - no quantization step, no modified weights
# CANARY_DTYPE can be set to bf16/float16 to halve VRAM at build/run time
# (default: float32 = model default).
#
# Peak VRAM (batch 1, chunked audio): ~5-6 GB in FP32, ~3.5-4.5 GB in BF16.

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
COPY server.py /app/server.py

ENV CANARY_MODEL_ID=nvidia/canary-1b-v2 \
    CANARY_DEFAULT_LANG=de \
    CANARY_FALLBACK_LANG=en \
    CANARY_DTYPE=float32 \
    HF_HOME=/models/hf-cache \
    PORT=8007

EXPOSE 8007

# First boot: ~4 GB download from HF. Give it plenty of room.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=10 \
    CMD curl -fsS http://localhost:8007/health || exit 1

CMD ["python3", "server.py"]
