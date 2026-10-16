"""Kokoro-82M English TTS server - REAL synthesis via the `kokoro` PyPI package.

Uses the actual hexgrad/Kokoro-82M reference implementation:
  - KModel: the PyTorch model (StyleTTS2-based architecture)
  - KPipeline: language-aware G2P (via misaki) + voice pack loading + inference

This is NOT a stub. It performs real neural TTS synthesis.
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

LOG = logging.getLogger("kokoro-tts-en")

DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "en")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "af_heart")
PORT = int(os.environ.get("PORT", "8000"))
# 'a' = American English, 'b' = British English (kokoro.pipeline.LANG_CODES)
LANG_CODE = os.environ.get("LANG_CODE", "a")

SAMPLE_RATE = 24000
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "20000"))

DEFAULT_RESPONSE_FORMAT = "aac"
SUPPORTED_RESPONSE_FORMATS = ("aac", "m4a", "mp3", "ogg", "wav")

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

# All English voices from hexgrad/Kokoro-82M VOICES.md
AMERICAN_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
]
BRITISH_VOICES = [
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]
ALL_VOICES = AMERICAN_VOICES + BRITISH_VOICES

_pipeline_a = None  # American English pipeline
_pipeline_b = None  # British English pipeline
_device: Optional[str] = None
_lock: Optional[asyncio.Lock] = None
_load_error: Optional[str] = None


def _load_model() -> None:
    """Load KModel once, share across two KPipelines (American + British)."""
    global _pipeline_a, _pipeline_b, _device, _lock, _load_error

    try:
        from kokoro import KModel, KPipeline

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        LOG.info(f"Loading KModel (hexgrad/Kokoro-82M) on device={_device}")

        model = KModel().to(_device).eval()
        LOG.info("KModel loaded successfully")

        # One KPipeline per lang_code, sharing the same KModel instance
        # (per KPipeline docstring: "you should reuse one KModel instance
        # across all of them")
        _pipeline_a = KPipeline(lang_code="a", model=model, device=_device)
        _pipeline_b = KPipeline(lang_code="b", model=model, device=_device)
        LOG.info("KPipelines ready: American English (a), British English (b)")

        _lock = asyncio.Lock()
        LOG.info(f"Kokoro TTS ready. {len(ALL_VOICES)} voices available.")

    except Exception as exc:
        LOG.exception(f"Model load failed: {exc}")
        _load_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="kokoro-tts-english", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    if _pipeline_a is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error or 'loading'}")
    return {
        "status": "ok",
        "model": "hexgrad/Kokoro-82M",
        "languages": ["en"],
        "voices": ALL_VOICES,
        "device": _device,
        "sample_rate": SAMPLE_RATE,
    }


@app.get("/v1/voices")
async def list_voices() -> dict:
    if _pipeline_a is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error or 'loading'}")
    return {
        "voices": ALL_VOICES,
        "default_voice": DEFAULT_VOICE,
        "default_language": DEFAULT_LANG,
    }


def _pipeline_for_voice(voice: str):
    """British voices (bf_/bm_) use the British pipeline, else American."""
    if voice.startswith("b"):
        return _pipeline_b
    return _pipeline_a


def _synthesize_blocking(text: str, voice: str, speed: float) -> np.ndarray:
    """Real Kokoro synthesis via KPipeline. Blocking - run via to_thread."""
    pipeline = _pipeline_for_voice(voice)
    assert pipeline is not None, "Pipeline not loaded"

    chunks = []
    # KPipeline.__call__ splits on '\n+' by default and yields Result objects
    # with .audio (torch.FloatTensor) per chunk. It also does its own
    # sentence-level phoneme-length chunking internally (waterfall_last, 510
    # phoneme cap), so we don't need to pre-split sentences ourselves.
    for result in pipeline(text, voice=voice, speed=speed):
        if result.audio is None:
            continue
        audio_np = result.audio.detach().cpu().numpy().astype(np.float32)
        chunks.append(audio_np)

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
    """Real TTS synthesis."""
    if _pipeline_a is None or _lock is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error or 'loading'}")

    text = payload.get("input")
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="input is required")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=400, detail=f"input too long: {len(text)} > {MAX_INPUT_CHARS}")

    language = str(payload.get("language") or DEFAULT_LANG).lower()
    if language not in ("en", "en-us", "en-gb"):
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

    async with _lock:
        start = time.monotonic()
        LOG.info(f"Synthesizing {len(text)} chars, voice={voice}, speed={speed}")
        try:
            samples = await asyncio.to_thread(_synthesize_blocking, text, voice, speed)
        except Exception as exc:
            LOG.exception(f"Synthesis failed: {exc}")
            return Response(status_code=500, content=str(exc)[:2000])
        elapsed = time.monotonic() - start
        duration = len(samples) / SAMPLE_RATE
        LOG.info(f"Synthesized {duration:.2f}s of REAL audio in {elapsed:.2f}s wall-clock")

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
