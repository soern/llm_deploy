"""Kokoro-82M ONNX German TTS server (voice "Martin").

OpenAI-shaped `/v1/audio/speech` endpoint backed by
`Godelaune/Kokoro-82M-ONNX-German-Martin` via the `kokoro-onnx` PyPI package
(ONNX Runtime under the hood). CPU-first, like
`deploy/dgx-spark/faster-whisper/server.py` - an 82M-parameter model runs
comfortably on CPU, keeping the DGX Spark GPU free for vLLM + FLUX.2.
`ORT_PROVIDER=cuda` is an opt-in escape hatch (see README "CPU vs GPU").

Model recon findings (verified live on 2026-08-25 against the real HF repo -
NOT guessed; see README.md "Model recon findings" for the full writeup and
how they were obtained):
  - exact ONNX filename: `kokoro-martin.onnx` (single file, no fp32/int8
    variants in this repo)
  - voices file: `voices-martin.npz` (numpy npz, one voice: "martin"),
    loaded via `kokoro_onnx.Kokoro.from_session(session, voices_path)`
  - phonemizer: `kokoro-onnx` uses `phonemizer-fork` + `espeakng-loader`
    (bundles its own espeak-ng lib/data) with a fallback to a system-wide
    espeak-ng shared library if the bundled one fails to load - see
    `kokoro_onnx/tokenizer.py`. The language code passed straight through is
    plain `"de"` (confirmed: the model author's own reference service sets
    `KOKORO_ONNX_LANG=de`).
  - sample rate: 24000 Hz mono float32 (confirmed in both `kokoro_onnx`'s
    `config.py` and the model author's own reference `onnx-docker/main.py`).
  - `kokoro-onnx==0.5.0` is pinned exactly because it's the version the
    model author's own `onnx-docker/Dockerfile` uses against this exact
    model export.

This file is intentionally the only place that knows these specifics, per
the plan's "Remaining risks" note - if the upstream repo layout ever
changes, only this file (plus the two *_FILENAME env var defaults) needs to
change.

Not ported from the upstream author's reference service (kept out of scope
for this deploy - see README "Known limitations"): the German text
normalization layer (`german_text_rules.py` / `tts_normalizer.py`, e.g.
"2,5 kWh" -> "zwei komma fünf Kilowattstunden") and the multi-worker
process-pool parallelism. This server does a single in-process model with a
serializing `asyncio.Lock`, matching the simpler pattern used by
`deploy/dgx-spark/faster-whisper/server.py` and
`deploy/dgx-spark/flux2-klein/server.py`.
"""

import asyncio
import io
import logging
import os
import re
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

LOG = logging.getLogger("kokoro-tts-de-server")

TTS_MODEL_ID = os.environ.get("TTS_MODEL_ID", "Godelaune/Kokoro-82M-ONNX-German-Martin")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/kokoro-de")
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "de")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "martin")
ORT_PROVIDER = os.environ.get("ORT_PROVIDER", "cpu").lower()  # cpu | cuda
HF_TOKEN = os.environ.get("HF_TOKEN") or None
PORT = int(os.environ.get("PORT", "8000"))

# Exact filenames inside the HF repo - confirmed via
# `huggingface_hub.list_repo_files("Godelaune/Kokoro-82M-ONNX-German-Martin")`,
# not guessed. Overridable in case TTS_MODEL_ID is later pointed at a
# differently-laid-out export.
ONNX_FILENAME = os.environ.get("ONNX_FILENAME", "kokoro-martin.onnx")
VOICES_FILENAME = os.environ.get("VOICES_FILENAME", "voices-martin.npz")

# Kokoro's native output rate (see module docstring for how this was
# confirmed). Not configurable - it's a property of the model, not a server
# option.
SAMPLE_RATE = 24000

# Matches TTSEngineConfig.max_input_chars's default on the agent side (see
# PLAN_TTS.md step 4) so the client-side check and this server-side
# rejection agree.
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "20000"))

# "aac" (raw ADTS AAC, no MP4 box) is the default, not "m4a" (MP4-boxed AAC):
# real Signal voice notes arrive as raw `audio/aac` ADTS (see
# `src/nixclaw_agent/stt.py`'s `_AUDIO_TYPES`), and a fragmented-MP4-boxed AAC
# file with no video track is ambiguous enough that magic-byte sniffing
# (which is what happens when the receiving Signal client doesn't trust the
# declared content-type) can misdetect it as a video. Raw ADTS AAC doesn't
# have that ambiguity. "m4a" is kept available for callers that specifically
# want an MP4 container.
DEFAULT_RESPONSE_FORMAT = "aac"
SUPPORTED_RESPONSE_FORMATS = ("aac", "m4a", "mp3", "ogg", "wav")

# ffmpeg args to transcode the WAV kokoro-onnx/soundfile produce into the
# requested container/codec. "wav" needs no ffmpeg pass at all.
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

_kokoro = None  # type: ignore[var-annotated]
_lock: Optional[asyncio.Lock] = None
_load_error: Optional[str] = None


def _ensure_model_present() -> None:
    """If MODEL_DIR doesn't already have the onnx model, download it from
    TTS_MODEL_ID via HF snapshot_download. Reuse on subsequent restarts.

    Mirrors deploy/dgx-spark/faster-whisper/server.py's `_ensure_model_present`
    idiom exactly, including the marker-file existence check so repeat
    starts don't re-download.
    """
    target = Path(MODEL_DIR)
    marker = target / ONNX_FILENAME
    if marker.exists():
        LOG.info(f"Reusing existing model at {target}")
        return

    LOG.info(f"Model not found at {target}; downloading {TTS_MODEL_ID} from HuggingFace")
    target.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=TTS_MODEL_ID,
            local_dir=str(target),
            token=HF_TOKEN,
            # The upstream repo also contains mp3 samples, docker-compose
            # files, and a second (Wyoming bridge) service's source - we
            # only need the two model asset files.
            allow_patterns=["*.onnx", "*.npz"],
        )
    except Exception as exc:
        # Clean up partial download so the next start retries cleanly.
        LOG.error(f"Download failed: {exc}")
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        raise

    if not marker.exists():
        raise RuntimeError(
            f"snapshot_download completed but {ONNX_FILENAME} is still missing from "
            f"{target} - the repo layout may have changed; check ONNX_FILENAME/VOICES_FILENAME"
        )


def _build_providers(available: list) -> list:
    """Pick ONNX Runtime execution providers for ORT_PROVIDER, falling back
    to CPU (with a warning, not a crash) if cuda was requested but isn't
    actually available - e.g. because this image was built from the
    CPU-only Dockerfile in this directory without swapping in
    onnxruntime-gpu (see README "CPU vs GPU")."""
    if ORT_PROVIDER == "cuda":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        LOG.warning(
            "ORT_PROVIDER=cuda requested but CUDAExecutionProvider is not "
            f"available (installed providers: {available}); this image ships "
            "the CPU-only onnxruntime build by default - see README's "
            "'CPU vs GPU' section. Falling back to CPU."
        )
        return ["CPUExecutionProvider"]
    if ORT_PROVIDER != "cpu":
        LOG.warning(f"Unknown ORT_PROVIDER={ORT_PROVIDER!r}; falling back to cpu")
    return ["CPUExecutionProvider"]


def _load_model() -> None:
    """Load the ORT session + voices once at startup. Sets _load_error
    instead of raising so /health can report a clear "not loaded" status
    rather than the container crash-looping, matching
    deploy/dgx-spark/flux2-klein/server.py's `_load_pipeline` pattern."""
    global _kokoro, _lock, _load_error
    try:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        _ensure_model_present()
        onnx_path = str(Path(MODEL_DIR) / ONNX_FILENAME)
        voices_path = str(Path(MODEL_DIR) / VOICES_FILENAME)
        providers = _build_providers(ort.get_available_providers())
        LOG.info(f"Loading Kokoro ONNX model from {onnx_path} (providers={providers})")
        session = ort.InferenceSession(onnx_path, providers=providers)
        _kokoro = Kokoro.from_session(session, voices_path)
        _lock = asyncio.Lock()
        LOG.info(f"Model loaded successfully; voices={_kokoro.get_voices()}")
    except Exception as exc:
        LOG.exception(f"Model load failed: {exc}")
        _load_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="kokoro-tts-german-server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    if _kokoro is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error or 'loading'}")
    return {
        "status": "ok",
        "model": TTS_MODEL_ID,
        "languages": [DEFAULT_LANG],
        "voices": _kokoro.get_voices(),
        "ort_provider": ORT_PROVIDER,
        "sample_rate": SAMPLE_RATE,
    }


@app.get("/v1/voices")
async def list_voices() -> dict:
    if _kokoro is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error or 'loading'}")
    return {
        "voices": _kokoro.get_voices(),
        "default_voice": DEFAULT_VOICE,
        "default_language": DEFAULT_LANG,
    }


# Rough sentence-boundary splitter for long-input chunking (Kokoro degrades
# on very long inputs; kokoro-onnx itself also internally batches phonemes
# at MAX_PHONEME_LENGTH=510, but that's a phoneme-count safety net, not a
# semantically-aware chunker). Not German-abbreviation-aware - see README
# "Known limitations" re: the upstream author's more sophisticated
# `german_text_rules.py` sentence segmentation, not ported here.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=\S)")


def _split_sentences(text: str) -> list:
    text = re.sub(r"\n\s*\n", "\n\n", text.strip())
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    sentences = []
    for part in parts:
        for line in part.split("\n"):
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences or [text]


def _synthesize_blocking(text: str, voice: str, lang: str, speed: float) -> np.ndarray:
    """Blocking Kokoro inference - run via asyncio.to_thread from the
    endpoint, ORT sessions are not async-native."""
    assert _kokoro is not None
    sentences = _split_sentences(text)
    if not sentences:
        raise ValueError("no synthesizable text after sentence splitting")
    chunks = []
    for sentence in sentences:
        samples, _sr = _kokoro.create(sentence, voice=voice, speed=speed, lang=lang)
        chunks.append(samples)
    return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]


async def _encode_audio(samples: np.ndarray, response_format: str) -> bytes:
    """Encode 24kHz mono float32 PCM to the requested container/codec.

    Runs ffmpeg via asyncio.create_subprocess_exec (NOT blocking
    subprocess.run) so it doesn't stack a second blocking call on top of the
    asyncio.to_thread-wrapped ORT inference that already ran above it.
    """
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
        raise RuntimeError(f"ffmpeg exited {proc.returncode}: {detail}")
    return stdout


@app.post("/v1/audio/speech")
async def synthesize(payload: dict) -> Response:
    """OpenAI-shaped TTS synthesis.

    Body: {"input": str, "voice": "martin", "language": "de",
           "response_format": "aac", "speed": 1.0}
    -> 200, binary audio body, Content-Type per response_format.
    """
    if _kokoro is None or _lock is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error or 'loading'}")

    text = payload.get("input")
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="input is required")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"input too long: {len(text)} chars > MAX_INPUT_CHARS={MAX_INPUT_CHARS}",
        )

    language = str(payload.get("language") or DEFAULT_LANG).lower()
    if language != DEFAULT_LANG:
        # German-only model (single fine-tuned voice) - reject rather than
        # silently mis-synthesize with the wrong phonemizer language code.
        raise HTTPException(
            status_code=400,
            detail=f"unsupported language {language!r}; this server only serves {DEFAULT_LANG!r}",
        )

    voice = str(payload.get("voice") or DEFAULT_VOICE)
    available_voices = _kokoro.get_voices()
    if voice not in available_voices:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported voice {voice!r}; available: {available_voices}",
        )

    speed = float(payload.get("speed") or 1.0)
    if not (0.5 <= speed <= 2.0):
        raise HTTPException(status_code=400, detail="speed must be between 0.5 and 2.0")

    response_format = str(payload.get("response_format") or DEFAULT_RESPONSE_FORMAT).lower()
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported response_format {response_format!r}; expected one of {SUPPORTED_RESPONSE_FORMATS}",
        )

    async with _lock:
        start = time.monotonic()
        LOG.info(
            f"Synthesizing {len(text)} chars, voice={voice}, lang={language}, "
            f"speed={speed}, format={response_format}"
        )
        try:
            samples = await asyncio.to_thread(_synthesize_blocking, text, voice, language, speed)
        except Exception as exc:
            LOG.exception(f"Synthesis failed: {exc}")
            return Response(status_code=500, content=str(exc)[:2000])
        elapsed = time.monotonic() - start
        LOG.info(f"Synthesized {len(samples) / SAMPLE_RATE:.1f}s of audio in {elapsed:.1f}s")

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
