"""Canary-1B-flash STT server.

OpenAI-shaped /v1/audio/transcriptions endpoint backed by NVIDIA NeMo's
Canary-1b-flash multitask ASR model (FastConformer encoder + Transformer
decoder). Mirrors the request/response contract of
deploy/dgx-spark/faster-whisper/server.py exactly, so nixclaw's stt.py
client (which only speaks that generic contract) works against either
backend unchanged - just repoint STT_BASE_URL.

Device is autodetected (CUDA if available, else CPU) for local testing
without a GPU; see README.md for expected latency on CPU.
"""

import logging
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

LOG = logging.getLogger("canary-server")

MODEL_ID = os.environ.get("CANARY_MODEL_ID", "nvidia/canary-1b-flash")
DEFAULT_LANG = os.environ.get("CANARY_DEFAULT_LANG", "en")
# Canary-1b-flash supports ASR + translation among these four languages.
SUPPORTED_LANGS = {"en", "de", "es", "fr"}

_model = None
_model_lock = None  # type: ignore[var-annotated]


def _detect_device() -> str:
    """Pick CUDA if available, otherwise CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


DEVICE = os.environ.get("DEVICE") or _detect_device()


def _load_model() -> None:
    global _model, _model_lock
    import asyncio
    from nemo.collections.asr.models import EncDecMultiTaskModel

    LOG.info(f"Loading {MODEL_ID} (device={DEVICE})")
    model = EncDecMultiTaskModel.from_pretrained(MODEL_ID, map_location=DEVICE)
    model.change_decoding_strategy(model.cfg.decoding)
    model.eval()
    _model = model
    _model_lock = asyncio.Lock()
    LOG.info("Model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="canary-1b-flash-server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_id": MODEL_ID, "device": DEVICE}


@app.get("/v1/models")
async def list_models() -> dict:
    """OpenAI-compatible model list (one entry, the baked Canary checkpoint)."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "nvidia-nemo",
                "device": DEVICE,
            }
        ],
    }


def _convert_to_wav(src: Path, dst: Path) -> None:
    """Downmix/resample to 16kHz mono WAV (FastConformer's trained rate)."""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore")[-500:])


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
    simplified field set (no `timestamp_granularities`, `translate`, etc.)
    - identical to deploy/dgx-spark/faster-whisper/server.py's contract.
    """
    if _model is None or _model_lock is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if response_format and response_format != "json":
        raise HTTPException(status_code=400, detail=f"unsupported response_format: {response_format}")

    lang = (language or DEFAULT_LANG).lower()
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported language: {lang!r} (Canary-1b-flash supports {sorted(SUPPORTED_LANGS)})",
        )

    tmp_src: Optional[Path] = None
    tmp_wav: Optional[Path] = None
    try:
        suffix = Path(file.filename or "audio").suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_src = Path(tmp.name)
        tmp_wav = tmp_src.with_suffix(".16k.wav")
        _convert_to_wav(tmp_src, tmp_wav)

        async with _model_lock:
            LOG.info(
                f"Transcribing {tmp_wav} ({tmp_wav.stat().st_size} bytes) "
                f"model={model} language={lang}"
            )
            # ASR (not translation): source == target language.
            results = _model.transcribe(
                audio=[str(tmp_wav)],
                batch_size=1,
                task="asr",
                source_lang=lang,
                target_lang=lang,
                pnc="yes",
            )
            hyp = results[0]
            text = (hyp.text if hasattr(hyp, "text") else str(hyp)).strip()

        LOG.info(f"Transcription done: text_len={len(text)}")
        return JSONResponse({"text": text})
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception(f"Transcription failed: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        for p in (tmp_src, tmp_wav):
            if p and p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
