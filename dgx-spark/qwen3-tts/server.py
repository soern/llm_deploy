"""Qwen3-TTS server, OpenAI-shaped /v1/audio/speech, one container for all languages.

Same endpoint contract as deploy/cpu/kokoro-tts-*/, so nixclaw's tts.py works
unchanged: point every `tts.engines.<lang>.base_url` at this server.

A `voice` is either
  - a preset speaker of the CustomVoice model (e.g. "aiden"), or
  - a cloned voice: `<VOICES_DIR>/<name>.safetensors`, made by
    deploy/dgx-spark/qwen3-tts-voice-clone/ and played by the Base model.
Each model is loaded only if some voice needs it. At most MAX_LOADED_MODELS
(default 2: 1.7B Base + 0.6B CustomVoice) stay in memory. With 1, a request
for the other model unloads the current one first, so switching between a
preset voice and a cloned voice costs a model load.
Startup preloads the model(s) of the first voices in DEFAULT_VOICES. Voice
files are picked up without a restart.

Long input is split at sentence boundaries into chunks of <= CHUNK_CHARS,
generated as one batch (up to MAX_BATCH at a time) and joined.

Optional sampling fields per request (qwen-tts defaults when absent:
temperature 0.9, top_k 50, top_p 1.0, repetition_penalty 1.05,
subtalker_temperature 0.9): lower values give a steadier delivery across
separately generated sentences (nixclaw's voice calls). `seed` makes a
request reproducible (A/B listening tests).
"""

import asyncio
import gc
import io
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from qwen_voice_file import VOICE_NAME_RE, load_voice, read_metadata

LOG = logging.getLogger("qwen3-tts")

BASE_MODEL = os.environ.get("QWEN_TTS_BASE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
CUSTOM_MODEL = os.environ.get("QWEN_TTS_CUSTOM_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
VOICES_DIR = os.environ.get("VOICES_DIR", "/voices")
PORT = int(os.environ.get("PORT", "8010"))
DEVICE = os.environ.get("QWEN_TTS_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
# bf16 on GPU; fp32 on CPU unless QWEN_TTS_DTYPE=bfloat16 (halves RAM, ~8GB -> ~4GB per 1.7B model).
DTYPE = getattr(torch, os.environ.get("QWEN_TTS_DTYPE") or ("bfloat16" if DEVICE.startswith("cuda") else "float32"))
# flash_attention_2 is faster but has no GB10/aarch64 wheel; sdpa is built in.
ATTN_IMPLEMENTATION = os.environ.get("QWEN_TTS_ATTN", "sdpa")
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "400"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "2"))
CHUNK_PAUSE_SECONDS = float(os.environ.get("CHUNK_PAUSE_SECONDS", "0.2"))
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "20000"))
# Models kept in memory at once (1 or 2). Each 1.7B model is ~4.5 GB in bf16.
MAX_LOADED_MODELS = max(1, int(os.environ.get("MAX_LOADED_MODELS", "2")))
# Voice used when a request names none, per language: "de=x,en=aiden,fr=y".
DEFAULT_VOICES = dict(
    pair.split("=", 1) for pair in
    os.environ.get("DEFAULT_VOICES", "de=de-voiceclip,en=aiden,fr=developpeuse,es=es-villarraza").split(",")
    if "=" in pair
)

# ISO code (what nixclaw sends) -> Qwen3-TTS language name.
LANGUAGES = {
    "de": "german", "en": "english", "fr": "french", "es": "spanish", "it": "italian",
    "pt": "portuguese", "ru": "russian", "ja": "japanese", "ko": "korean", "zh": "chinese",
}
# CustomVoice speakers (config.json spk_id); each can speak every language,
# best in its native one (aiden, ryan: English).
PRESET_SPEAKERS = ("aiden", "ryan", "vivian", "serena", "uncle_fu", "dylan", "eric", "ono_anna", "sohee")

DEFAULT_RESPONSE_FORMAT = "aac"
SUPPORTED_RESPONSE_FORMATS = ("aac", "m4a", "mp3", "ogg", "wav")

_FFMPEG_CODEC_ARGS = {
    "aac": ["-c:a", "aac", "-b:a", "64k", "-f", "adts"],
    "m4a": ["-c:a", "aac", "-b:a", "64k", "-f", "mp4", "-movflags", "frag_keyframe+empty_moov"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"],
    "ogg": ["-c:a", "libopus", "-b:a", "32k", "-f", "ogg"],
    "wav": ["-f", "wav"],
}
_CONTENT_TYPE = {
    "aac": "audio/aac",
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "wav": "audio/wav",
}

_models: dict[str, Any] = {}  # "base" / "custom" -> Qwen3TTSModel
_clone_voices: dict[str, tuple[float, Any]] = {}  # name -> (mtime, VoiceClonePromptItem)
_lock: Optional[asyncio.Lock] = None
_load_error: Optional[str] = None


def _free_memory() -> None:
    """Return freed tensors and the allocator's cached blocks to the system."""
    gc.collect()
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()


def _model(kind: str) -> Any:
    """Load (if needed) and return the Base or CustomVoice model, unloading
    another one first if MAX_LOADED_MODELS are already loaded. Blocking;
    only called under _lock or at startup."""
    if kind not in _models:
        from qwen_tts import Qwen3TTSModel

        while len(_models) >= MAX_LOADED_MODELS:
            evicted = next(iter(_models))
            LOG.info(f"Unloading {BASE_MODEL if evicted == 'base' else CUSTOM_MODEL} to make room")
            del _models[evicted]
            if evicted == "base":
                # cloned voice prompts hold tensors on the Base model's device
                _clone_voices.clear()
            _free_memory()
        model_id = BASE_MODEL if kind == "base" else CUSTOM_MODEL
        LOG.info(f"Loading {model_id} on {DEVICE} ({DTYPE}, attn={ATTN_IMPLEMENTATION})")
        start = time.monotonic()
        _models[kind] = Qwen3TTSModel.from_pretrained(
            model_id, device_map=DEVICE, dtype=DTYPE, attn_implementation=ATTN_IMPLEMENTATION
        )
        LOG.info(f"Loaded {model_id} in {time.monotonic() - start:.1f}s")
    return _models[kind]


def _voice_path(name: str) -> str:
    return os.path.join(VOICES_DIR, f"{name}.safetensors")


def _clone_voice(name: str) -> Any:
    """Cloned voice prompt, reloaded when its file changes. Blocking."""
    path = _voice_path(name)
    mtime = os.path.getmtime(path)
    cached = _clone_voices.get(name)
    if cached and cached[0] == mtime:
        return cached[1]
    base = _model("base")
    item, metadata = load_voice(path, device=base.device, dtype=DTYPE)
    if metadata.get("base_model") != BASE_MODEL:
        raise ValueError(
            f"voice {name!r} was cloned with {metadata.get('base_model')!r}, but this server "
            f"runs {BASE_MODEL!r}; re-clone it with the matching model"
        )
    _clone_voices[name] = (mtime, item)
    return item


def _resolve_voice(voice: str) -> tuple[str, str]:
    """('preset'|'clone', name) or HTTP 400."""
    name = voice.strip().lower()
    if name in PRESET_SPEAKERS:
        return "preset", name
    if VOICE_NAME_RE.match(name) and os.path.isfile(_voice_path(name)):
        return "clone", name
    raise HTTPException(
        status_code=400,
        detail=f"unknown voice {voice!r}; presets: {list(PRESET_SPEAKERS)}, cloned: {_list_clone_voices()}",
    )


def _list_clone_voices() -> list[str]:
    if not os.path.isdir(VOICES_DIR):
        return []
    return sorted(
        f[: -len(".safetensors")] for f in os.listdir(VOICES_DIR)
        if f.endswith(".safetensors") and VOICE_NAME_RE.match(f[: -len(".safetensors")])
    )


def _preload() -> None:
    global _lock, _load_error
    try:
        # Preload the models of the first default voices (DEFAULT_VOICES order),
        # up to MAX_LOADED_MODELS; the rest load on first use.
        for lang, voice in DEFAULT_VOICES.items():
            try:
                kind, name = _resolve_voice(voice)
            except HTTPException:
                LOG.warning(f"Default voice {voice!r} for {lang!r} not available (yet); skipping preload")
                continue
            model_kind = "custom" if kind == "preset" else "base"
            if model_kind not in _models and len(_models) >= MAX_LOADED_MODELS:
                continue
            if kind == "preset":
                _model("custom")
            else:
                _clone_voice(name)
        _lock = asyncio.Lock()
        LOG.info(f"Qwen3-TTS ready: models={sorted(_models)} default_voices={DEFAULT_VOICES} "
                 f"cloned={_list_clone_voices()}")
    except Exception as exc:
        LOG.exception(f"Startup failed: {exc}")
        _load_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _preload()
    yield


app = FastAPI(title="qwen3-tts", lifespan=lifespan)


def _require_loaded() -> None:
    if _lock is None:
        raise HTTPException(status_code=503, detail=f"not ready: {_load_error or 'loading'}")


@app.get("/health")
async def health() -> dict:
    _require_loaded()
    return {
        "status": "ok",
        "models": {kind: (BASE_MODEL if kind == "base" else CUSTOM_MODEL) for kind in _models},
        "languages": sorted(LANGUAGES),
        "default_voices": DEFAULT_VOICES,
        "max_loaded_models": MAX_LOADED_MODELS,
        "device": DEVICE,
    }


@app.get("/v1/voices")
async def list_voices() -> dict:
    _require_loaded()
    cloned = {}
    for name in _list_clone_voices():
        try:
            meta = read_metadata(_voice_path(name))
        except Exception as exc:
            cloned[name] = {"error": str(exc)}
            continue
        cloned[name] = {k: meta.get(k) for k in ("language", "base_model", "ref_text", "ref_seconds", "created")}
    return {"presets": list(PRESET_SPEAKERS), "cloned": cloned, "default_voices": DEFAULT_VOICES}


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


# Request field -> (type, min, max) of the optional sampling settings.
SAMPLING_FIELDS = {
    "temperature": (float, 0.05, 2.0),
    "top_k": (int, 1, 1000),
    "top_p": (float, 0.05, 1.0),
    "repetition_penalty": (float, 1.0, 2.0),
    "subtalker_temperature": (float, 0.05, 2.0),
}


def _sampling_kwargs(payload: dict) -> dict:
    """Validated optional sampling settings of a request (HTTP 400 if bad)."""
    out: dict = {}
    for name, (kind, lo, hi) in SAMPLING_FIELDS.items():
        if payload.get(name) is None:
            continue
        try:
            value = kind(payload[name])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{name} must be a number")
        if not (lo <= value <= hi):
            raise HTTPException(status_code=400, detail=f"{name} must be {lo}-{hi}")
        out[name] = value
    return out


def _synthesize_blocking(chunks: list[str], language: str, kind: str, voice: str,
                         sampling: Optional[dict] = None,
                         seed: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Generate all chunks (batched) and join them. Blocking - run via to_thread."""
    pieces: list[np.ndarray] = []
    sample_rate = 24000
    sampling = sampling or {}
    if seed is not None:
        torch.manual_seed(seed)
    for start in range(0, len(chunks), MAX_BATCH):
        batch = chunks[start:start + MAX_BATCH]
        languages = [language] * len(batch)
        if kind == "preset":
            wavs, sample_rate = _model("custom").generate_custom_voice(
                text=batch, speaker=[voice] * len(batch), language=languages, **sampling
            )
        else:
            wavs, sample_rate = _model("base").generate_voice_clone(
                text=batch, language=languages, voice_clone_prompt=[_clone_voice(voice)], **sampling
            )
        for wav in wavs:
            if pieces:
                pieces.append(np.zeros(int(CHUNK_PAUSE_SECONDS * sample_rate), dtype=np.float32))
            pieces.append(np.asarray(wav, dtype=np.float32))
    if not pieces:
        raise ValueError("no audio produced")
    return np.concatenate(pieces), sample_rate


async def _encode_audio(samples: np.ndarray, sample_rate: int, response_format: str, speed: float) -> bytes:
    """Encode mono float32 PCM to the requested format via ffmpeg."""
    wav_buf = io.BytesIO()
    sf.write(wav_buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    wav_bytes = wav_buf.getvalue()
    if response_format == "wav" and speed == 1.0:
        return wav_bytes

    # The model has no speed control; atempo changes tempo without pitch shift.
    filter_args = ["-filter:a", f"atempo={speed}"] if speed != 1.0 else []
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "wav", "-i", "pipe:0",
        *filter_args, *_FFMPEG_CODEC_ARGS[response_format], "pipe:1",
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

    lang_code = str(payload.get("language") or "").lower().split("-")[0]
    if lang_code not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"unsupported language {payload.get('language')!r}; "
                                                    f"supported: {sorted(LANGUAGES)}")

    voice = str(payload.get("voice") or DEFAULT_VOICES.get(lang_code) or "")
    if not voice:
        raise HTTPException(status_code=400, detail=f"no voice given and no default voice for {lang_code!r}")
    kind, voice_name = _resolve_voice(voice)

    speed = float(payload.get("speed") or 1.0)
    if not (0.5 <= speed <= 2.0):
        raise HTTPException(status_code=400, detail="speed must be 0.5-2.0")

    response_format = str(payload.get("response_format") or DEFAULT_RESPONSE_FORMAT).lower()
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(status_code=400, detail=f"unsupported response_format {response_format!r}")

    sampling = _sampling_kwargs(payload)
    seed = payload.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise HTTPException(status_code=400, detail="seed must be an integer >= 0")

    chunks = _split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="input is empty after whitespace cleanup")

    async with _lock:
        start = time.monotonic()
        LOG.info(f"Synthesizing {len(text)} chars in {len(chunks)} chunk(s), "
                 f"lang={lang_code}, voice={voice_name} ({kind}), speed={speed}"
                 + (f", sampling={sampling}" if sampling else "") + (f", seed={seed}" if seed is not None else ""))
        try:
            samples, sample_rate = await asyncio.to_thread(
                _synthesize_blocking, chunks, LANGUAGES[lang_code], kind, voice_name, sampling, seed
            )
        except ValueError as exc:
            # e.g. a voice file cloned with a different base model
            LOG.warning(f"Synthesis rejected: {exc}")
            raise HTTPException(status_code=400, detail=str(exc)[:2000])
        except Exception as exc:
            LOG.exception(f"Synthesis failed: {exc}")
            return Response(status_code=500, content=str(exc)[:2000])
        finally:
            # Release the generation's peak working memory; the GPU memory is
            # shared with the LLM server on the Spark.
            await asyncio.to_thread(_free_memory)
        elapsed = time.monotonic() - start
        LOG.info(f"Synthesized {len(samples) / sample_rate:.2f}s of audio in {elapsed:.2f}s wall-clock")

    try:
        audio_bytes = await _encode_audio(samples, sample_rate, response_format, speed)
    except Exception as exc:
        LOG.exception(f"Encoding failed: {exc}")
        return Response(status_code=500, content=str(exc)[:2000])

    return Response(content=audio_bytes, media_type=_CONTENT_TYPE[response_format])


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
