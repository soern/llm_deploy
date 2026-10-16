"""Faster-Whisper STT server.

OpenAI-shaped /v1/audio/transcriptions endpoint backed by faster-whisper
(CTranslate2). Designed for both GPU and CPU inference — `device` and
`compute_type` are read from environment variables. CPU mode uses
`compute_type="int8"`; GPU mode uses `compute_type="float16"`.

The server accepts WAV / MP3 / M4A / FLAC / OGG / WEBM (anything ffmpeg
can decode) via multipart upload and returns `{"text": "..."}` JSON.

Model loading: if MODEL_DIR is empty, we download WHISPER_MODEL_ID
(Systran/faster-whisper-* by default) from HuggingFace into MODEL_DIR
on first startup. Subsequent restarts reuse the on-disk weights.
"""

import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

LOG = logging.getLogger("faster-whisper-server")

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/faster-whisper-large-v3")
WHISPER_MODEL_ID = os.environ.get("WHISPER_MODEL_ID", "Systran/faster-whisper-large-v3")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE")  # None → auto below
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "5"))


def _detect_device() -> str:
    """Pick CUDA if available, otherwise CPU."""
    try:
        import ctranslate2
        if ctranslate2.get_device_count("cuda") > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


DEVICE = os.environ.get("DEVICE") or _detect_device()
if not COMPUTE_TYPE:
    COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

_model = None
_model_lock = None  # type: ignore[var-annotated]


def _ensure_model_present() -> None:
    """If MODEL_DIR is empty, populate it from WHISPER_MODEL_ID."""
    target = Path(MODEL_DIR)
    needs_marker = target / "model.bin"
    if needs_marker.exists():
        LOG.info(f"Reusing existing model at {target}")
        return

    LOG.info(f"Model not found at {target}; downloading {WHISPER_MODEL_ID} from HuggingFace")
    target.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=WHISPER_MODEL_ID,
            local_dir=str(target),
            token=HF_TOKEN,
            allow_patterns=[
                "*.bin",
                "*.json",
                "tokenizer.*",
                "vocabulary.txt",
                "vocabulary.json",
                "preprocessor_config.json",
                "config.json",
            ],
        )
    except Exception as exc:
        # Clean up partial download so the next start retries cleanly.
        LOG.error(f"Download failed: {exc}")
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        raise


def _load_model() -> None:
    global _model, _model_lock
    import asyncio
    from faster_whisper import WhisperModel

    _ensure_model_present()
    LOG.info(f"Loading faster-whisper model from {MODEL_DIR} (device={DEVICE}, compute_type={COMPUTE_TYPE})")
    _model = WhisperModel(MODEL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)
    _model_lock = asyncio.Lock()
    LOG.info("Model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="faster-whisper-server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
    }


@app.get("/v1/models")
async def list_models() -> dict:
    """OpenAI-compatible model list. The model id is the basename of MODEL_DIR."""
    model_id = Path(MODEL_DIR).name
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "faster-whisper",
                "device": DEVICE,
                "compute_type": COMPUTE_TYPE,
            }
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
) -> JSONResponse:
    """Transcribe an uploaded audio file.

    Mirrors the OpenAI `POST /v1/audio/transcriptions` shape, with a
    simplified field set (no `timestamp_granularities`, `translate`, etc.).
    """
    if _model is None or _model_lock is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if response_format and response_format != "json":
        raise HTTPException(status_code=400, detail=f"unsupported response_format: {response_format}")

    tmp_path: Optional[Path] = None
    try:
        suffix = Path(file.filename or "audio").suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        async with _model_lock:
            kwargs: dict = {"beam_size": BEAM_SIZE, "temperature": temperature or 0.0}
            if language:
                kwargs["language"] = language
            LOG.info(
                f"Transcribing {tmp_path} ({tmp_path.stat().st_size} bytes) "
                f"model={model} language={language}"
            )
            segments, info = _model.transcribe(str(tmp_path), **kwargs)
            text = " ".join(segment.text.strip() for segment in segments).strip()

        LOG.info(
            f"Transcription done: language={info.language} "
            f"prob={info.language_probability:.2f} duration={info.duration:.1f}s "
            f"text_len={len(text)}"
        )
        return JSONResponse({"text": text})
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception(f"Transcription failed: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8006")))