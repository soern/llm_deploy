# Qwen3-ASR STT server - DGX Spark (GB10, aarch64, sm_121).
#
# nvcr.io/nvidia/pytorch is the ARM64 + CUDA-13 PyTorch source for GB10 (same
# base as deploy/dgx-spark/qwen3-tts/). For a CPU-only smoke test on any box,
# build with
#   --build-arg BASE_IMAGE=python:3.12-slim --build-arg INSTALL_CPU_TORCH=1
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE}

ARG INSTALL_CPU_TORCH=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$INSTALL_CPU_TORCH" = "1" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.9.1"; \
    fi

# qwen-asr goes in with --no-deps: its requirements pull in gradio + flask
# (demo UIs only), nagisa/soynlp (forced aligner only) and sox/qwen-omni-utils
# (never imported by the inference path). Its real runtime deps are listed
# here, with qwen-asr's own pins for transformers/accelerate, so NVCR's
# GB10-capable torch is never replaced.
RUN pip install --no-cache-dir --no-deps "qwen-asr==0.0.6" \
    && pip install --no-cache-dir \
        "transformers==4.57.6" \
        "accelerate==1.12.0" \
        librosa \
        soundfile \
        safetensors \
        huggingface_hub \
        fastapi \
        "uvicorn[standard]" \
        python-multipart

# qwen_asr eagerly imports its forced aligner, which imports nagisa. Install
# an import stub (see nagisa_stub/nagisa/__init__.py) unless nagisa works.
COPY nagisa_stub /opt/nagisa_stub
RUN python3 -c "import nagisa" 2>/dev/null \
    || cp -r /opt/nagisa_stub/nagisa "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/" \
    && python3 -c "from qwen_asr import Qwen3ASRModel; print('qwen_asr import ok')"

WORKDIR /app
COPY server.py ./

ENV PORT=8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health > /dev/null || exit 1

CMD ["python3", "server.py"]
