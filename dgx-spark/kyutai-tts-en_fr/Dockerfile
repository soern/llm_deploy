# Kyutai TTS 1.6B (en+fr) server - DGX Spark (GB10, aarch64, sm_121).
#
# nvcr.io/nvidia/pytorch is the ARM64 + CUDA-13 PyTorch source for GB10 (same
# base as deploy/dgx-spark/flux2-klein/). For a CPU-only smoke test on an
# x86 box, build with
#   --build-arg BASE_IMAGE=python:3.12-slim --build-arg INSTALL_CPU_TORCH=1
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE}

ARG INSTALL_CPU_TORCH=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates build-essential pkg-config libopus-dev \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$INSTALL_CPU_TORCH" = "1" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.9.1"; \
    fi

# sphn (moshi's audio I/O, imported by moshi.models.tts) ships no aarch64
# wheels on PyPI, so it is built from source - which needs a Rust toolchain
# (and libopus-dev from the apt step above, for its audiopus_sys dependency).
# --no-binary forces the same source build on x86 so the CPU smoke test
# exercises the path the Spark takes.
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && pip install --no-cache-dir --no-binary sphn "sphn>=0.2.0,<0.3.0"

# moshi pins torch<2.10; the NVCR torch is a 2.10 pre-release build, and
# letting pip "fix" that would replace the only GB10-capable torch with a
# generic wheel. So moshi goes in with --no-deps and its runtime deps are
# listed here (minus sounddevice/bitsandbytes, unused by the TTS path).
RUN pip install --no-cache-dir --no-deps "moshi==0.2.13" \
    && pip install --no-cache-dir \
        "safetensors>=0.4.0,<0.8.0" \
        "huggingface-hub>=0.24,<1.0" \
        "einops>=0.7,<0.9" \
        "sentencepiece>=0.2.0,<0.3" \
        "aiohttp>=3.10.5,<3.12" \
        "tqdm>=4.48" \
        num2words \
        soundfile \
        fastapi \
        "uvicorn[standard]"

WORKDIR /app
COPY french_text.py server.py ./

ENV PORT=8009

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health > /dev/null || exit 1

CMD ["python3", "server.py"]
