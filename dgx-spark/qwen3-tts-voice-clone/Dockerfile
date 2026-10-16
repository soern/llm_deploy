# Qwen3-TTS voice-clone service: turns an uploaded reference clip into a
# cloned-voice file for deploy/dgx-spark/qwen3-tts/. Runs on CPU or GPU.
#
# The layers up to the faster-whisper step are kept identical to
# deploy/dgx-spark/qwen3-tts/Dockerfile so the two images share them.
#
# nvcr.io/nvidia/pytorch is the ARM64 + CUDA-13 PyTorch source for GB10 (same
# base as deploy/dgx-spark/flux2-klein/). For a CPU-only smoke test on any
# box (cloning itself is cheap enough for CPU), build with
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

# qwen-tts goes in with --no-deps: its requirements would drag in gradio (demo
# UI only) and a torchaudio wheel that could replace NVCR's GB10-capable
# torch. Its real runtime deps are listed here, with qwen-tts's own pins for
# transformers/accelerate. sox + onnxruntime are only *imported* (by the 25Hz
# tokenizer module, which qwen_tts loads eagerly), never used by 12Hz models.
RUN pip install --no-cache-dir --no-deps "qwen-tts==0.1.1" \
    && pip install --no-cache-dir \
        "transformers==4.57.3" \
        "accelerate==1.12.0" \
        librosa \
        soundfile \
        sox \
        onnxruntime \
        einops \
        safetensors \
        huggingface_hub \
        fastapi \
        "uvicorn[standard]"

# Same reason: install a torchaudio import stub only if the base image has
# no working torchaudio (see torchaudio_stub/torchaudio/__init__.py).
COPY torchaudio_stub /opt/torchaudio_stub
RUN python3 -c "import torchaudio.compliance.kaldi" 2>/dev/null \
    || cp -r /opt/torchaudio_stub/torchaudio "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/" \
    && python3 -c "import qwen_tts; print('qwen_tts import ok')"

# Transcribes reference clips (and picks where to cut them). ctranslate2 is
# CPU-only on aarch64, which is fine for short clips.
RUN pip install --no-cache-dir "faster-whisper==1.2.1" python-multipart

WORKDIR /app
COPY qwen_voice_file.py server.py ./

ENV PORT=8011

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health > /dev/null || exit 1

CMD ["python3", "server.py"]
