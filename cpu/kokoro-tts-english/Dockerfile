FROM python:3.11-slim

# Install system dependencies for audio processing and TTS
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    espeak-ng-data \
    curl \
    ca-certificates \
    build-essential \
    libsndfile1-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# 'kokoro' PyPI package is the REAL hexgrad/Kokoro-82M reference implementation
# (KModel + KPipeline). Uses misaki[en] for English G2P (grapheme-to-phoneme).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 && \
    pip install --no-cache-dir \
    kokoro==0.7.16 \
    "misaki[en]" \
    soundfile \
    numpy \
    scipy \
    fastapi \
    uvicorn[standard] \
    huggingface_hub

# misaki[en]'s English G2P lazily does `pip install en_core_web_sm` (a spacy
# model, ~13MB from GitHub) the first time an English KPipeline is built if
# it isn't already importable. Left to run at container startup, this makes
# every restart depend on GitHub reachability and can hang for minutes (or
# fail outright) with the server not yet listening. Install it here at
# build time instead, so startup never touches the network for it.
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

WORKDIR /app

# Copy server code
COPY server.py .

# Environment variables (overridable at runtime)
ENV TTS_MODEL_ID="hexgrad/Kokoro-82M"
ENV MODEL_DIR="/models/kokoro-en"
ENV DEFAULT_LANG="en"
ENV DEFAULT_VOICE="af_heart"
ENV PORT="8000"
ENV ORT_PROVIDER="cpu"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health > /dev/null || exit 1

CMD ["python3", "server.py"]
