# CPU-first: an 82M-parameter ONNX model runs comfortably on CPU (~real-time
# or faster), which keeps the DGX Spark GPU free for vLLM + FLUX.2. This
# image ships CPU-only onnxruntime; ORT_PROVIDER=cuda is an opt-in switch
# but requires swapping "onnxruntime" for "onnxruntime-gpu" below (and a
# CUDA-capable base image) - not done here on purpose, see README "CPU vs
# GPU". server.py falls back to CPU with a warning if cuda is requested but
# unavailable, rather than crashing.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg espeak-ng espeak-ng-data libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# kokoro-onnx==0.5.0 pinned exactly: it's the version the model author's own
# onnx-docker/Dockerfile uses against this exact model export (confirmed by
# downloading the wheel and inspecting it - see README "Model recon
# findings"). It pulls in phonemizer-fork + espeakng-loader, which bundle
# their own espeak-ng shared library/data; the apt-installed
# espeak-ng/espeak-ng-data above is the system-wide fallback kokoro-onnx's
# own Tokenizer falls back to if the bundled copy fails to load (see
# kokoro_onnx/tokenizer.py in the pip package). libsndfile1 is needed by the
# `soundfile` pip package used to write the WAV before ffmpeg transcodes it.
RUN pip install --no-cache-dir \
        "onnxruntime>=1.20.1" \
        "kokoro-onnx==0.5.0" \
        "soundfile" \
        "numpy" \
        "fastapi==0.115.*" \
        "uvicorn[standard]==0.32.*" \
        "huggingface_hub>=0.20"

COPY server.py /app/server.py

# Override at build/run time:
#   TTS_MODEL_ID=Godelaune/Kokoro-82M-ONNX-German-Martin
#   MODEL_DIR=/models/kokoro-de
#   DEFAULT_LANG=de   DEFAULT_VOICE=martin
#   ORT_PROVIDER=cpu | cuda
ENV TTS_MODEL_ID=Godelaune/Kokoro-82M-ONNX-German-Martin \
    MODEL_DIR=/models/kokoro-de \
    DEFAULT_LANG=de \
    DEFAULT_VOICE=martin \
    ORT_PROVIDER=cpu \
    PORT=8000

EXPOSE 8000

# Model download is small (~80MB total: onnx + npz), so a much shorter
# start-period than the whisper/flux2 sidecars' (which pull GBs of weights)
# is enough.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python3", "server.py"]
