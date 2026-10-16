"""Kokoro-82M French TTS server - real synthesis via the `kokoro` PyPI package.

Same model and endpoint contract as deploy/cpu/kokoro-tts-english/, but with the
French KPipeline (lang_code "f", espeak-ng fr-fr G2P) and Kokoro's only French
voice, `ff_siwis`. Input text first goes through french_text.normalize_french
(numbers, times, ordinals, currency, a few abbreviations).
"""

import asyncio
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from french_text import normalize_french

LOG = logging.getLogger("kokoro-tts-fr")

MODEL_ID = "hexgrad/Kokoro-82M"
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "fr")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "ff_siwis")
PORT = int(os.environ.get("PORT", "8000"))
NORMALIZE_TEXT = os.environ.get("FRENCH_NORMALIZE", "1") not in ("0", "false", "no")

SAMPLE_RATE = 24000
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "20000"))

DEFAULT_RESPONSE_FORMAT = "aac"
SUPPORTED_RESPONSE_FORMATS = ("aac", "m4a", "mp3", "ogg", "wav")
SUPPORTED_LANGUAGES = ("fr", "fr-fr")

_FFMPEG_CODEC_ARGS = {
    "aac": ["-c:a", "aac", "-b:a", "64k", "-f", "adts"],
    "m4a": ["-c:a", "aac", "-b:a", "64k", "-f", "mp4", "-movflags", "frag_keyframe+empty_moov"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"],
    "ogg": ["-c:a", "libopus", "-b:a", "32k", "-f", "ogg"],
}
_CONTENT_TYPE = {
    "aac": "audio/aac",
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "wav": "audio/wav",
}

# hexgrad/Kokoro-82M VOICES.md lists exactly one French voice.
ALL_VOICES = ["ff_siwis"]

_pipeline = None
_device: Optional[str] = None
_lock: Optional[asyncio.Lock] = None
_load_error: Optional[str] = None


def _load_model() -> None:
    global _pipeline, _device, _lock, _load_error

    try:
        from kokoro import KModel, KPipeline

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        LOG.info(f"Loading KModel ({MODEL_ID}) on device={_device}")
        model = KModel().to(_device).eval()
        _pipeline = KPipeline(lang_code="f", model=model, device=_device)
        # Fetch the voice pack now so the first request doesn't hit the network.
        _pipeline.load_voice(DEFAULT_VOICE)
        _lock = asyncio.Lock()
        LOG.info(f"Kokoro French TTS ready (voice={DEFAULT_VOICE}, normalize={NORMALIZE_TEXT})")
    except Exception as exc:
        LOG.exception(f"Model load failed: {exc}")
        _load_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="kokoro-tts-french", lifespan=lifespan)


def _require_loaded() -> None:
    if _pipeline is None or _lock is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error or 'loading'}")


@app.get("/health")
async def health() -> dict:
    _require_loaded()
    return {
        "status": "ok",
        "model": MODEL_ID,
        "languages": ["fr"],
        "voices": ALL_VOICES,
        "device": _device,
        "sample_rate": SAMPLE_RATE,
    }


@app.get("/v1/voices")
async def list_voices() -> dict:
    _require_loaded()
    return {
        "voices": ALL_VOICES,
        "default_voice": DEFAULT_VOICE,
        "default_language": DEFAULT_LANG,
    }


def _synthesize_blocking(text: str, voice: str, speed: float) -> np.ndarray:
    """Real Kokoro synthesis via KPipeline. Blocking - run via to_thread."""
    assert _pipeline is not None, "Pipeline not loaded"

    chunks = []
    # KPipeline splits on '\n+' and, for espeak languages, further chunks
    # long lines at sentence boundaries itself.
    for result in _pipeline(text, voice=voice, speed=speed):
        if result.audio is None:
            continue
        chunks.append(result.audio.detach().cpu().numpy().astype(np.float32))

    if not chunks:
        raise ValueError("no audio produced (empty or unsynthesizable text)")

    return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]


async def _encode_audio(samples: np.ndarray, response_format: str) -> bytes:
    """Encode 24kHz mono float32 PCM to requested format via ffmpeg."""
    wav_buf = io.BytesIO()
    sf.write(wav_buf, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_bytes = wav_buf.getvalue()

    if response_format == "wav":
        return wav_bytes

    codec_args = _FFMPEG_CODEC_ARGS[response_format]
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "wav", "-i", "pipe:0",
        *codec_args, "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=wav_bytes)
    if proc.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"ffmpeg failed: {detail}")
    return stdout


@app.post("/v1/audio/speech")
async def synthesize(payload: dict) -> Response:
    _require_loaded()
    assert _lock is not None

    text = payload.get("input")
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="input is required")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=400, detail=f"input too long: {len(text)} > {MAX_INPUT_CHARS}")

    language = str(payload.get("language") or DEFAULT_LANG).lower()
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"unsupported language {language!r}")

    voice = str(payload.get("voice") or DEFAULT_VOICE)
    if voice not in ALL_VOICES:
        raise HTTPException(status_code=400, detail=f"unsupported voice {voice!r}; available: {ALL_VOICES}")

    speed = float(payload.get("speed") or 1.0)
    if not (0.5 <= speed <= 2.0):
        raise HTTPException(status_code=400, detail="speed must be 0.5-2.0")

    response_format = str(payload.get("response_format") or DEFAULT_RESPONSE_FORMAT).lower()
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(status_code=400, detail=f"unsupported response_format {response_format!r}")

    if NORMALIZE_TEXT:
        text = normalize_french(text)

    async with _lock:
        start = time.monotonic()
        LOG.info(f"Synthesizing {len(text)} chars, voice={voice}, speed={speed}")
        try:
            samples = await asyncio.to_thread(_synthesize_blocking, text, voice, speed)
        except Exception as exc:
            LOG.exception(f"Synthesis failed: {exc}")
            return Response(status_code=500, content=str(exc)[:2000])
        elapsed = time.monotonic() - start
        LOG.info(f"Synthesized {len(samples) / SAMPLE_RATE:.2f}s of audio in {elapsed:.2f}s wall-clock")

    try:
        audio_bytes = await _encode_audio(samples, response_format)
    except Exception as exc:
        LOG.exception(f"Encoding failed: {exc}")
        return Response(status_code=500, content=str(exc)[:2000])

    return Response(content=audio_bytes, media_type=_CONTENT_TYPE[response_format])


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
