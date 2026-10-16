FROM python:3.11-slim

# espeak-ng does the French grapheme-to-phoneme step (kokoro lang_code "f"
# -> misaki EspeakG2P("fr-fr")); ffmpeg encodes the output formats.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    espeak-ng-data \
    curl \
    ca-certificates \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Versions verified together on 2026-10-07. Do NOT copy the English deploy's
# pins (torch 2.5.1, kokoro 0.7.16), they break French:
#  - kokoro must be 0.9.x to match misaki 0.9.x: misaki 0.9 G2P returns a
#    (phonemes, tokens) tuple, which kokoro 0.7.16 feeds to the model as-is on
#    the non-English path -> ~1s of junk audio for any input.
#  - torch>=2.6: transformers 5.x (pulled in by kokoro) calls
#    torch.accelerator, which torch 2.5.1 lacks.
# The French pipeline needs no spacy model (English-only G2P), only
# phonemizer-fork + espeakng-loader from misaki[en]. num2words drives
# french_text.py.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 && \
    pip install --no-cache-dir \
    kokoro==0.9.4 \
    "misaki[en]==0.9.4" \
    transformers==5.19.0 \
    num2words \
    soundfile \
    numpy \
    fastapi \
    "uvicorn[standard]" \
    huggingface_hub

WORKDIR /app
COPY french_text.py server.py ./

ENV DEFAULT_LANG="fr"
ENV DEFAULT_VOICE="ff_siwis"
ENV PORT="8000"

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health > /dev/null || exit 1

CMD ["python3", "server.py"]
