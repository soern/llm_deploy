"""Qwen3-TTS voice-clone service.

Upload a reference clip, get back a cloned-voice file
(`<name>.safetensors`, see qwen_voice_file.py) that
deploy/dgx-spark/qwen3-tts/ plays from its voices dir.

Qwen3-TTS cloning is zero-shot: the Base model encodes the clip into speech
codes + a speaker embedding, conditioned on the clip's transcript. No
training run, so this is cheap enough for CPU. What costs time is the
optional preview sentence, which is real generation.

Pipeline for POST /v1/voices:
  1. decode the upload with ffmpeg (anything ffmpeg reads: m4a, aac, mp3, wav...)
  2. transcript: the given `ref_text`, else faster-whisper. With whisper the
     reference is also trimmed at segment/word boundaries to
     <= max_ref_seconds: every generation re-processes the whole reference,
     so a long one makes every reply slower without sounding better.
  3. create_voice_clone_prompt -> <name>.safetensors (+ <name>.ref.wav, the
     exact audio used, for auditing)
  4. optional preview: one sentence in the cloned voice -> <name>.preview.wav
"""

import asyncio
import datetime
import io
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from qwen_voice_file import VOICE_NAME_RE, read_metadata, save_voice

LOG = logging.getLogger("qwen3-tts-clone")

BASE_MODEL = os.environ.get("QWEN_TTS_BASE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
VOICES_DIR = os.environ.get("VOICES_DIR", "/voices")
PORT = int(os.environ.get("PORT", "8011"))
DEVICE = os.environ.get("QWEN_TTS_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
# bf16 on GPU; fp32 on CPU unless QWEN_TTS_DTYPE=bfloat16 (halves RAM, ~8GB -> ~4GB per 1.7B model).
DTYPE = getattr(torch, os.environ.get("QWEN_TTS_DTYPE") or ("bfloat16" if DEVICE.startswith("cuda") else "float32"))
ATTN_IMPLEMENTATION = os.environ.get("QWEN_TTS_ATTN", "sdpa")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_SOURCE_SECONDS = float(os.environ.get("MAX_SOURCE_SECONDS", "300"))
# With an explicit transcript nothing is trimmed, so cap the reference length.
MAX_REF_SECONDS_WITH_TEXT = float(os.environ.get("MAX_REF_SECONDS_WITH_TEXT", "30"))
DEFAULT_MAX_REF_SECONDS = float(os.environ.get("DEFAULT_MAX_REF_SECONDS", "12"))
MIN_REF_SECONDS = 3.0
SAMPLE_RATE = 24000

LANGUAGES = {
    "de": "german", "en": "english", "fr": "french", "es": "spanish", "it": "italian",
    "pt": "portuguese", "ru": "russian", "ja": "japanese", "ko": "korean", "zh": "chinese",
}
PREVIEW_TEXTS = {
    "de": "Hallo! Das ist eine kurze Hörprobe meiner neuen Stimme. Klinge ich so, wie du es erwartet hast?",
    "en": "Hello! This is a short sample of my new voice. Do I sound the way you expected?",
    "fr": "Bonjour ! Voici un court extrait de ma nouvelle voix. Est-ce que je ressemble à ce que tu attendais ?",
    "es": "¡Hola! Esta es una breve muestra de mi nueva voz. ¿Sueno como esperabas?",
}

_base = None
_whisper = None
_lock: Optional[asyncio.Lock] = None
_load_error: Optional[str] = None


def _load_models() -> None:
    global _base, _whisper, _lock, _load_error
    try:
        from faster_whisper import WhisperModel
        from qwen_tts import Qwen3TTSModel

        LOG.info(f"Loading {BASE_MODEL} on {DEVICE} ({DTYPE}, attn={ATTN_IMPLEMENTATION})")
        _base = Qwen3TTSModel.from_pretrained(
            BASE_MODEL, device_map=DEVICE, dtype=DTYPE, attn_implementation=ATTN_IMPLEMENTATION
        )
        LOG.info(f"Loading faster-whisper {WHISPER_MODEL} (cpu, int8)")
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        os.makedirs(VOICES_DIR, exist_ok=True)
        _lock = asyncio.Lock()
        LOG.info(f"Voice-clone service ready, voices dir {VOICES_DIR}")
    except Exception as exc:
        LOG.exception(f"Model load failed: {exc}")
        _load_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(title="qwen3-tts-voice-clone", lifespan=lifespan)


def _require_loaded() -> None:
    if _lock is None:
        raise HTTPException(status_code=503, detail=f"not ready: {_load_error or 'loading'}")


def _paths(name: str) -> dict[str, str]:
    base = os.path.join(VOICES_DIR, name)
    return {"file": f"{base}.safetensors", "preview": f"{base}.preview.wav", "reference": f"{base}.ref.wav"}


def _check_name(name: str) -> str:
    name = name.strip().lower()
    if not VOICE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="name must match [a-z0-9][a-z0-9_-]{0,63}")
    return name


def _decode(data: bytes, sample_rate: int) -> np.ndarray:
    """Any ffmpeg-readable audio -> mono float32 at sample_rate. Blocking."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"],
        input=data, capture_output=True, timeout=120,
    )
    if proc.returncode != 0:
        raise ValueError(f"could not decode audio: {proc.stderr.decode(errors='replace')[:500]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _transcribe_and_trim(audio_16k: np.ndarray, language: str, max_ref_seconds: float) -> tuple[str, float, str]:
    """(transcript, cut_seconds, full_transcript) for the longest prefix that
    ends <= max_ref_seconds on a word followed by punctuation (a natural
    pause), else on any word boundary. Blocking."""
    assert _whisper is not None
    segments = list(_whisper.transcribe(audio_16k, language=language, word_timestamps=True, vad_filter=True)[0])
    if not segments:
        raise ValueError("no speech found in the clip")
    full = " ".join(s.text.strip() for s in segments)

    words = [w for seg in segments for w in (seg.words or [])]
    fitting = [i for i, w in enumerate(words) if w.end <= max_ref_seconds]
    if not fitting:
        raise ValueError("could not find a word boundary within max_ref_seconds")
    at_pause = [i for i in fitting if words[i].word.strip()[-1:] in ".,!?;:…"]
    last = at_pause[-1] if at_pause else fitting[-1]
    return "".join(w.word for w in words[: last + 1]).strip(), words[last].end, full


def _clone_blocking(data: bytes, name: str, lang_code: str, ref_text: Optional[str],
                    max_ref_seconds: float, preview_text: Optional[str], source_name: str) -> dict[str, Any]:
    assert _base is not None
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    audio = _decode(data, SAMPLE_RATE)
    source_seconds = len(audio) / SAMPLE_RATE
    if source_seconds > MAX_SOURCE_SECONDS:
        raise ValueError(f"clip is {source_seconds:.0f}s, max {MAX_SOURCE_SECONDS:.0f}s")

    full_transcript = None
    if ref_text:
        transcribed_by = "user"
        if source_seconds > MAX_REF_SECONDS_WITH_TEXT:
            raise ValueError(
                f"clip is {source_seconds:.1f}s; with an explicit transcript nothing is trimmed, so keep it "
                f"<= {MAX_REF_SECONDS_WITH_TEXT:.0f}s (or omit ref_text to auto-transcribe and trim)"
            )
        ref = audio
    else:
        transcribed_by = f"faster-whisper-{WHISPER_MODEL}"
        audio_16k = _decode(data, 16000)
        ref_text, cut, full_transcript = _transcribe_and_trim(audio_16k, lang_code, max_ref_seconds)
        # A little tail so the last word isn't clipped.
        ref = audio[: min(len(audio), int((cut + 0.15) * SAMPLE_RATE))]
        timings["transcribe"] = time.monotonic() - t0
    ref_seconds = len(ref) / SAMPLE_RATE
    if ref_seconds < MIN_REF_SECONDS:
        raise ValueError(f"reference is only {ref_seconds:.1f}s; need >= {MIN_REF_SECONDS:.0f}s of speech")

    t1 = time.monotonic()
    item = _base.create_voice_clone_prompt(ref_audio=(ref, SAMPLE_RATE), ref_text=ref_text)[0]
    timings["clone"] = time.monotonic() - t1

    paths = _paths(name)
    tmp = paths["file"] + ".tmp"
    save_voice(tmp, item, base_model=BASE_MODEL, language=lang_code, extra={
        "ref_seconds": f"{ref_seconds:.2f}",
        "source_seconds": f"{source_seconds:.2f}",
        "source_filename": os.path.basename(source_name)[:200],
        "transcribed_by": transcribed_by,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    })
    sf.write(paths["reference"], ref, SAMPLE_RATE, subtype="PCM_16")
    # safetensors writes 0600; this container runs as root, so without this the
    # host user couldn't read (or commit) the voice file.
    os.chmod(tmp, 0o644)
    # Rename last, so the TTS server (sharing this dir) never sees a half-written file.
    os.replace(tmp, paths["file"])

    result: dict[str, Any] = {
        "name": name,
        "language": lang_code,
        "base_model": BASE_MODEL,
        "ref_text": ref_text,
        "ref_seconds": round(ref_seconds, 2),
        "source_seconds": round(source_seconds, 2),
        "transcribed_by": transcribed_by,
    }
    if full_transcript is not None and full_transcript != ref_text:
        result["full_transcript"] = full_transcript

    if preview_text:
        t2 = time.monotonic()
        wavs, sr = _base.generate_voice_clone(
            text=preview_text, language=LANGUAGES[lang_code], voice_clone_prompt=[item]
        )
        sf.write(paths["preview"], np.asarray(wavs[0], dtype=np.float32), sr, subtype="PCM_16")
        timings["preview"] = time.monotonic() - t2
        result["preview_text"] = preview_text
        result["preview_seconds"] = round(len(wavs[0]) / sr, 2)
    elif os.path.exists(paths["preview"]):
        os.remove(paths["preview"])  # stale preview of an overwritten voice

    result["timings_seconds"] = {k: round(v, 1) for k, v in timings.items()}
    return result


@app.get("/health")
async def health() -> dict:
    _require_loaded()
    return {"status": "ok", "base_model": BASE_MODEL, "whisper": WHISPER_MODEL, "device": DEVICE,
            "voices_dir": VOICES_DIR}


@app.post("/v1/voices")
async def create_voice(
    file: UploadFile = File(...),
    name: str = Form(...),
    language: str = Form(...),
    ref_text: Optional[str] = Form(None),
    max_ref_seconds: float = Form(DEFAULT_MAX_REF_SECONDS),
    preview: bool = Form(True),
    preview_text: Optional[str] = Form(None),
    overwrite: bool = Form(False),
) -> dict:
    _require_loaded()
    assert _lock is not None
    name = _check_name(name)
    lang_code = language.strip().lower().split("-")[0]
    if lang_code not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"unsupported language {language!r}; supported: {sorted(LANGUAGES)}")
    if not (MIN_REF_SECONDS <= max_ref_seconds <= MAX_REF_SECONDS_WITH_TEXT):
        raise HTTPException(status_code=400, detail=f"max_ref_seconds must be {MIN_REF_SECONDS:.0f}-{MAX_REF_SECONDS_WITH_TEXT:.0f}")
    if os.path.exists(_paths(name)["file"]) and not overwrite:
        raise HTTPException(status_code=409, detail=f"voice {name!r} exists; pass overwrite=true to replace it")
    if preview and not preview_text:
        preview_text = PREVIEW_TEXTS.get(lang_code)
        if not preview_text:
            raise HTTPException(status_code=400, detail=f"no default preview sentence for {lang_code!r}; pass preview_text")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload larger than {MAX_UPLOAD_BYTES} bytes")
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")

    async with _lock:
        LOG.info(f"Cloning voice {name!r} ({lang_code}) from {file.filename!r}, {len(data)} bytes")
        try:
            result = await asyncio.to_thread(
                _clone_blocking, data, name, lang_code, (ref_text or "").strip() or None,
                max_ref_seconds, preview_text if preview else None, file.filename or "",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            LOG.exception(f"Cloning {name!r} failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)[:2000])
    LOG.info(f"Cloned {name!r}: {result['ref_seconds']}s reference, timings {result['timings_seconds']}")
    return result


@app.get("/v1/voices")
async def list_voices() -> dict:
    _require_loaded()
    voices = {}
    for f in sorted(os.listdir(VOICES_DIR)):
        if f.endswith(".safetensors") and VOICE_NAME_RE.match(f[: -len(".safetensors")]):
            name = f[: -len(".safetensors")]
            try:
                voices[name] = read_metadata(_paths(name)["file"])
            except Exception as exc:
                voices[name] = {"error": str(exc)}
    return {"voices": voices}


def _existing(name: str, kind: str) -> str:
    path = _paths(_check_name(name))[kind]
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"no {kind} for voice {name!r}")
    return path


@app.get("/v1/voices/{name}/file")
async def get_voice_file(name: str) -> FileResponse:
    return FileResponse(_existing(name, "file"), media_type="application/octet-stream",
                        filename=f"{name}.safetensors")


@app.get("/v1/voices/{name}/preview")
async def get_preview(name: str) -> FileResponse:
    return FileResponse(_existing(name, "preview"), media_type="audio/wav", filename=f"{name}.preview.wav")


@app.get("/v1/voices/{name}/reference")
async def get_reference(name: str) -> FileResponse:
    return FileResponse(_existing(name, "reference"), media_type="audio/wav", filename=f"{name}.ref.wav")


@app.delete("/v1/voices/{name}")
async def delete_voice(name: str) -> dict:
    _existing(name, "file")
    for path in _paths(_check_name(name)).values():
        if os.path.exists(path):
            os.remove(path)
    return {"deleted": name}


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
