FROM python:3.11-slim

# CPU-only test image. GPU inference would need an nvidia/cuda base plus
# the CUDA build of torch (see README "GPU" section) - this Dockerfile
# intentionally installs the CPU wheel of torch first so nemo_toolkit's
# resolver doesn't pull the much larger CUDA wheel as a transitive dep.

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchaudio

RUN pip install --no-cache-dir \
        "nemo_toolkit[asr]" \
        "fastapi==0.115.*" \
        "uvicorn[standard]==0.32.*" \
        "python-multipart==0.0.12" \
        "huggingface_hub>=0.20"

WORKDIR /app
COPY server.py /app/server.py

# Model weights land in HF_HOME (a mounted volume) on first startup and
# are reused on restart - see README "Model assets".
ENV CANARY_MODEL_ID=nvidia/canary-1b-flash \
    CANARY_DEFAULT_LANG=en \
    HF_HOME=/models/hf-cache \
    PORT=8000

EXPOSE 8000

# Long start-period: first boot downloads ~3.5 GB of weights from HF.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python3", "server.py"]
