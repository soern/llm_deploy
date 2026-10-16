"""Kyutai TTS 1.6B (English + French) server, OpenAI-shaped /v1/audio/speech.

Wraps moshi's PyTorch TTSModel (kyutai/tts-1.6b-en_fr) behind the same
endpoint contract as deploy/cpu/kokoro-tts-*/, so nixclaw's tts.py client works
unchanged. Voices are pre-computed speaker embeddings from the
kyutai/tts-voices HF repo, addressed by a short alias (see VOICE_ALIASES) or
by their path inside that repo (e.g. "cml-tts/fr/1406_1028_000009-0003_enhanced.wav").

Long input is split at sentence boundaries into chunks of <= CHUNK_CHARS,
which are generated as one batch (up to MAX_BATCH at a time) and joined.
"""

import asyncio
import io
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from french_text import normalize_french

LOG = logging.getLogger("kyutai-tts")

MODEL_REPO = os.environ.get("KYUTAI_TTS_REPO", "kyutai/tts-1.6b-en_fr")
VOICE_REPO = os.environ.get("KYUTAI_VOICE_REPO", "kyutai/tts-voices")
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "fr")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "developpeuse")
PORT = int(os.environ.get("PORT", "8009"))
DEVICE = os.environ.get("KYUTAI_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
# Audio codebooks generated per step: 32 = best quality, 8-16 = faster.
N_Q = int(os.environ.get("KYUTAI_N_Q", "32"))
# Must be one of the model's distilled CFG values (1.0-4.0 in 0.5 steps).
CFG_COEF = float(os.environ.get("KYUTAI_CFG_COEF", "2.0"))
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "500"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "8"))
CHUNK_PAUSE_SECONDS = float(os.environ.get("CHUNK_PAUSE_SECONDS", "0.25"))
NORMALIZE_TEXT = os.environ.get("FRENCH_NORMALIZE", "1") not in ("0", "false", "no")
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "20000"))

# The voices recommended for French. Both are Kyutai's own recordings from
# the unmute-prod-website/ folder, licensed CC0 (see the tts-voices README).
VOICE_ALIASES = {
    "developpeuse": "unmute-prod-website/developpeuse-3.wav",
    "fabien": "unmute-prod-website/fabieng-enhanced-v2.wav",
}
# Repo-relative voice paths: no "..", no URLs, no absolute paths.
_VOICE_PATH_RE = re.compile(r"^[A-Za-z0-9_\-]+(/[A-Za-z0-9_.\-]+)*\.(wav|mp3)$")

DEFAULT_RESPONSE_FORMAT = "aac"
SUPPORTED_RESPONSE_FORMATS = ("aac", "m4a", "mp3", "ogg", "wav")
SUPPORTED_LANGUAGES = ("fr", "fr-fr", "en", "en-us", "en-gb")

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

_tts_model = None
_sample_rate: int = 24000
_lock: Optional[asyncio.Lock] = None
_load_error: Optional[str] = None
_voice_paths: dict[str, str] = {}


def _resolve_voice_name(voice: str) -> str:
    return VOICE_ALIASES.get(voice, voice)


def _voice_path(voice: str) -> str:
    """Local path of a voice embedding, downloading it on first use. Blocking."""
    name = _resolve_voice_name(voice)
    if name not in _voice_paths:
        assert _tts_model is not None
        _voice_paths[name] = str(_tts_model.get_voice_path(name))
    return _voice_paths[name]


def _load_model() -> None:
    global _tts_model, _sample_rate, _lock, _load_error

    try:
        from moshi.models.loaders import CheckpointInfo
        from moshi.models.tts import TTSModel

        # bf16 is the model's native dtype; CPU smoke tests run in fp32.
        dtype = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
        LOG.info(f"Loading {MODEL_REPO} on device={DEVICE} dtype={dtype} n_q={N_Q}")
        checkpoint_info = CheckpointInfo.from_hf_repo(MODEL_REPO)
        _tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info, n_q=N_Q, temp=0.6, device=DEVICE, dtype=dtype, voice_repo=VOICE_REPO
        )
        _sample_rate = int(_tts_model.mimi.sample_rate)
        # Fetch the alias voices now so the first request doesn't hit the network.
        for alias in VOICE_ALIASES:
            _voice_path(alias)
        _voice_path(DEFAULT_VOICE)
        _lock = asyncio.Lock()
        LOG.info(f"Kyutai TTS ready (default voice={DEFAULT_VOICE}, sample_rate={_sample_rate})")
    except Exception as exc:
        LOG.exception(f"Model load failed: {exc}")
        _load_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="kyutai-tts-en_fr", lifespan=lifespan)


def _require_loaded() -> None:
    if _tts_model is None or _lock is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error or 'loading'}")


@app.get("/health")
async def health() -> dict:
    _require_loaded()
    return {
        "status": "ok",
        "model": MODEL_REPO,
        "languages": ["fr", "en"],
        "voices": sorted(VOICE_ALIASES),
        "device": DEVICE,
        "n_q": N_Q,
        "sample_rate": _sample_rate,
    }


@app.get("/v1/voices")
async def list_voices() -> dict:
    _require_loaded()
    return {
        "voices": sorted(VOICE_ALIASES),
        "voice_aliases": VOICE_ALIASES,
        "voice_repo": VOICE_REPO,
        "default_voice": DEFAULT_VOICE,
        "default_language": DEFAULT_LANG,
    }


def _split_text(text: str) -> list[str]:
    """Split into chunks of <= CHUNK_CHARS at paragraph/sentence boundaries."""
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n|\n", text):
        paragraph = " ".join(paragraph.split())
        if paragraph:
            sentences.extend(s for s in re.split(r"(?<=[.!?…])\s+", paragraph) if s)

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        # A single over-long sentence is cut at word boundaries.
        while len(sentence) > CHUNK_CHARS:
            cut = sentence.rfind(" ", 0, CHUNK_CHARS)
            cut = cut if cut > 0 else CHUNK_CHARS
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > CHUNK_CHARS:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def _synthesize_blocking(chunks: list[str], voice_path: str) -> np.ndarray:
    """Generate all chunks (batched) and join them. Blocking - run via to_thread."""
    assert _tts_model is not None
    pause = np.zeros(int(CHUNK_PAUSE_SECONDS * _sample_rate), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for start in range(0, len(chunks), MAX_BATCH):
        batch = chunks[start:start + MAX_BATCH]
        with torch.no_grad():
            pcms = _tts_model.simple_generate(batch, voice_path, cfg_coef=CFG_COEF, show_progress=False)
        for pcm in pcms:
            if pieces:
                pieces.append(pause)
            pieces.append(pcm.float().cpu().numpy())
    if not pieces:
        raise ValueError("no audio produced (empty or unsynthesizable text)")
    return np.concatenate(pieces)


async def _encode_audio(samples: np.ndarray, response_format: str, speed: float) -> bytes:
    """Encode mono float32 PCM to the requested format via ffmpeg."""
    wav_buf = io.BytesIO()
    sf.write(wav_buf, samples, _sample_rate, format="WAV", subtype="PCM_16")
    wav_bytes = wav_buf.getvalue()

    if response_format == "wav" and speed == 1.0:
        return wav_bytes

    # Kyutai TTS itself has no speed control; atempo changes tempo without
    # shifting pitch.
    filter_args = ["-filter:a", f"atempo={speed}"] if speed != 1.0 else []
    codec_args = _FFMPEG_CODEC_ARGS.get(response_format, ["-f", "wav"])
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "wav", "-i", "pipe:0",
        *filter_args, *codec_args, "pipe:1",
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
    if voice not in VOICE_ALIASES and not _VOICE_PATH_RE.match(voice):
        raise HTTPException(
            status_code=400,
            detail=f"unsupported voice {voice!r}; use one of {sorted(VOICE_ALIASES)} "
                   f"or a .wav/.mp3 path inside {VOICE_REPO}",
        )

    speed = float(payload.get("speed") or 1.0)
    if not (0.5 <= speed <= 2.0):
        raise HTTPException(status_code=400, detail="speed must be 0.5-2.0")

    response_format = str(payload.get("response_format") or DEFAULT_RESPONSE_FORMAT).lower()
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(status_code=400, detail=f"unsupported response_format {response_format!r}")

    if NORMALIZE_TEXT and language.startswith("fr"):
        text = normalize_french(text)
    chunks = _split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="input is empty after whitespace cleanup")

    async with _lock:
        try:
            voice_path = await asyncio.to_thread(_voice_path, voice)
        except Exception as exc:
            LOG.warning(f"Voice {voice!r} not found: {exc}")
            raise HTTPException(status_code=400, detail=f"voice {voice!r} not found in {VOICE_REPO}")

        start = time.monotonic()
        LOG.info(f"Synthesizing {len(text)} chars in {len(chunks)} chunk(s), voice={voice}, speed={speed}")
        try:
            samples = await asyncio.to_thread(_synthesize_blocking, chunks, voice_path)
        except Exception as exc:
            LOG.exception(f"Synthesis failed: {exc}")
            return Response(status_code=500, content=str(exc)[:2000])
        elapsed = time.monotonic() - start
        LOG.info(f"Synthesized {len(samples) / _sample_rate:.2f}s of audio in {elapsed:.2f}s wall-clock")

    try:
        audio_bytes = await _encode_audio(samples, response_format, speed)
    except Exception as exc:
        LOG.exception(f"Encoding failed: {exc}")
        return Response(status_code=500, content=str(exc)[:2000])

    return Response(content=audio_bytes, media_type=_CONTENT_TYPE[response_format])


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
