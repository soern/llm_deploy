FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# We use the CUDA runtime image even for CPU-only deployments so the same
# image can run on either host. The server detects CUDA at startup and
# falls back to device="cpu" + compute_type="int8" if no GPU is visible.

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg curl ca-certificates git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        "faster-whisper==1.0.3" \
        "fastapi==0.115.*" \
        "uvicorn[standard]==0.32.*" \
        "python-multipart==0.0.12" \
        "huggingface_hub>=0.20" \
        "requests>=2.31"

# Model assets live at /models. We either:
#  (a) COPY a pre-exported CTranslate2 directory at build time, OR
#  (b) download a pre-converted faster-whisper repo from HF at startup
#      via WHISPER_MODEL_ID + MODEL_DIR env vars (see README).
# For this image we keep /models empty and rely on (b); the docker-compose
# mount in repo can also inject a baked model directory.

WORKDIR /app
COPY server.py /app/server.py

# Override at build/run time:
#   WHISPER_MODEL_ID=Systran/faster-whisper-large-v3  (production, GPU)
#   WHISPER_MODEL_ID=Systran/faster-whisper-base      (CPU testing)
#   MODEL_DIR=/models/faster-whisper-large-v3
ENV MODEL_DIR=/models/faster-whisper-large-v3 \
    WHISPER_MODEL_ID=Systran/faster-whisper-large-v3 \
    COMPUTE_TYPE=float16 \
    PORT=8006

EXPOSE 8006

# Use the server module's own entrypoint so lifespan startup runs.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=5 \
    CMD curl -fsS http://localhost:8006/health || exit 1

CMD ["python3", "server.py"]