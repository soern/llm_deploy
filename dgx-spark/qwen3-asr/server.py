"""Qwen3-ASR STT server.

OpenAI-shaped /v1/audio/transcriptions endpoint backed by Qwen3-ASR
(default Qwen/Qwen3-ASR-0.6B, 30 languages + 22 Chinese dialects, built-in
language identification), via the `qwen-asr` package: its transformers
backend (default) or, with QWEN_ASR_BACKEND=vllm, its vLLM backend
(Dockerfile.vllm; CUDA graphs and prefix caching, see README).

Mirrors the request/response contract of deploy/dgx-spark/faster-whisper so
nixclaw's stt.py client works unchanged - it listens on the same port (8001).

Language handling:
  - no/empty `language`        -> auto-detect (model LID)
  - ISO-639-1 code or name     -> forced ("de" / "de-DE" / "German")
  - unsupported `language`     -> auto-detect, flagged in the response via
    "language_fallback": true
`prompt` (OpenAI's field name) is passed to the model as context, e.g. names
or jargon to bias spelling.

Speech detection (`detect_speech`): forced to a language, the model always
outputs words, even for silence or clatter ("Ich.", "Die."). Without one it
answers "language None" for non-speech. With detect_speech the server first
asks the model for the language alone and reports P("None") as
`no_speech_prob`; at or above `no_speech_threshold` the text is "" and
`no_speech` is true. The question is a prefix of the transcription prompt
(unforced chat prompt + "language", the forced one goes on with " German
<asr_text>"), so the transformers backend runs the audio and that prefix
once and continues the transcription from the same KV cache; vLLM reuses it
through its prefix cache.

Streaming: WebSocket /v1/audio/transcriptions/stream (protocol in
stream_transcriptions). The transformers backend has no incremental decoder:
each step re-decodes all audio so far (cheap for utterance-length audio).
qwen-asr's vLLM-only streaming_transcribe() feeds the previous text minus a
few tokens as a prefix after two steps; that locks in early mishearings, so
here only partials beyond QWEN_ASR_STREAM_PREFIX_AFTER_SECONDS use one. When
the last step already covers everything but the trailing silence the client
reports with "finish", its result is the final one without another decode.
The client sends "flush" as soon as the caller goes quiet, so that decode
mostly happens during the client's end-of-speech wait; it can switch
partials off ("partials") while it has no use for them, which keeps the
shared GPU free.
"""

import asyncio
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

LOG = logging.getLogger("qwen3-asr")

MODEL_ID = os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
BACKEND = os.environ.get("QWEN_ASR_BACKEND", "transformers").strip().lower()
if BACKEND not in ("transformers", "vllm"):
    raise ValueError(f"QWEN_ASR_BACKEND must be transformers or vllm, got {BACKEND!r}")
# vLLM backend only. Share of the GPU memory vLLM takes (weights, KV cache,
# CUDA graphs); on the Spark it sits next to the LLM, FLUX.2 and Qwen3-TTS.
VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("QWEN_ASR_VLLM_GPU_MEMORY_UTILIZATION", "0.07"))
VLLM_MAX_MODEL_LEN = int(os.environ.get("QWEN_ASR_VLLM_MAX_MODEL_LEN", "4096"))
VLLM_ENFORCE_EAGER = os.environ.get("QWEN_ASR_VLLM_ENFORCE_EAGER", "0") == "1"
# Top-k logprobs of the speech probe: P("None") below the k-th best counts as 0.
VLLM_PROBE_LOGPROBS = 20
MAX_NEW_TOKENS = int(os.environ.get("QWEN_ASR_MAX_NEW_TOKENS", "512"))
NO_SPEECH_THRESHOLD = float(os.environ.get("QWEN_ASR_NO_SPEECH_THRESHOLD", "0.5"))
STREAM_MAX_SECONDS = float(os.environ.get("QWEN_ASR_STREAM_MAX_SECONDS", "60"))
STREAM_PREFIX_AFTER_SECONDS = float(os.environ.get("QWEN_ASR_STREAM_PREFIX_AFTER_SECONDS", "10"))
SAMPLE_RATE = 16000
# Streaming prefix: the previous text minus its last tokens (as qwen-asr's
# streaming_transcribe), to bound the generation for long utterances.
STREAM_ROLLBACK_TOKENS = 5

# ISO-639 codes -> Qwen3-ASR's canonical language names (qwen_asr's
# SUPPORTED_LANGUAGES). Full names are accepted too.
ISO_TO_NAME = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "tl": "Filipino", "fa": "Persian", "el": "Greek",
    "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
}


def _detect_device() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


DEVICE = os.environ.get("DEVICE") or _detect_device()
_DTYPES = {"bf16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
DTYPE_NAME = (os.environ.get("QWEN_ASR_DTYPE") or ("float32" if DEVICE == "cpu" else "bf16")).lower()
if DTYPE_NAME not in _DTYPES:
    raise ValueError(f"QWEN_ASR_DTYPE must be one of {sorted(_DTYPES)}, got {DTYPE_NAME!r}")

_model = None
_supported: set[str] = set()
_model_lock = asyncio.Lock()
# Token ids of "language" and of the "None" that follows it for non-speech
# (see _probe); None if the tokenizer splits them unexpectedly.
_lang_ids: list[int] = []
_none_id: Optional[int] = None


def _load_model() -> None:
    global _model, _supported
    from qwen_asr import Qwen3ASRModel

    LOG.info(f"Loading {MODEL_ID} (backend={BACKEND}, device={DEVICE}, dtype={DTYPE_NAME})")
    if BACKEND == "vllm":
        _model = Qwen3ASRModel.LLM(
            model=MODEL_ID,
            dtype={"bf16": "bfloat16"}.get(DTYPE_NAME, DTYPE_NAME),
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=VLLM_MAX_MODEL_LEN,
            enforce_eager=VLLM_ENFORCE_EAGER,
            enable_prefix_caching=True,
            limit_mm_per_prompt={"audio": 1},
            max_inference_batch_size=1,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    else:
        _model = Qwen3ASRModel.from_pretrained(
            MODEL_ID,
            dtype=_DTYPES[DTYPE_NAME],
            device_map=DEVICE,
            max_inference_batch_size=1,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    _supported = set(_model.get_supported_languages())
    global _lang_ids, _none_id
    tok = _model.processor.tokenizer
    _lang_ids = tok.encode("language", add_special_tokens=False)
    full = tok.encode("language None", add_special_tokens=False)
    if full[:len(_lang_ids)] == _lang_ids and len(full) > len(_lang_ids):
        _none_id = full[len(_lang_ids)]
    else:
        LOG.warning("Unexpected tokenization of 'language None': no_speech_prob unavailable")
    LOG.info("Model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(_load_model)
    yield


app = FastAPI(title="qwen3-asr-server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "backend": BACKEND,
        "device": DEVICE,
        "dtype": DTYPE_NAME,
        "languages": sorted(_supported),
        "stream_features": ["flush", "partials"],
    }


@app.get("/v1/models")
async def list_models() -> dict:
    """OpenAI-compatible model list (one entry, the loaded checkpoint)."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "qwen",
                "device": DEVICE,
                "dtype": DTYPE_NAME,
            }
        ],
    }


def _resolve_language(requested: Optional[str]) -> tuple[Optional[str], bool]:
    """Map an ISO code / name to Qwen's name; None = auto-detect.

    Returns (language name or None, fell_back).
    """
    raw = (requested or "").strip()
    if not raw:
        return None, False
    code = raw.lower().replace("_", "-").split("-")[0]
    name = ISO_TO_NAME.get(code) or raw.capitalize()
    if name in _supported:
        return name, False
    LOG.warning(f"Unsupported language {requested!r} - using auto-detect")
    return None, True


def _prompt(context: str, lang: Optional[str]) -> str:
    return _model._build_text_prompt(context=context, force_language=lang)


def _probe_prompt(context: str) -> str:
    """Unforced prompt + "language": the next token is the language, "None"
    for non-speech. The forced prompt starts with it too."""
    return _prompt(context, None) + "language"


def _decode_ids(ids: list[int]) -> str:
    return _model.processor.tokenizer.decode(ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)


# --- transformers backend -------------------------------------------------

_EOS_IDS = [151645, 151643]     # as Qwen3ASRForConditionalGeneration.generate


def _hf_inputs(prompt: str, wav: np.ndarray):
    inputs = _model.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
    return inputs.to(_model.model.device).to(_model.model.dtype)


def _hf_generate(inputs, cache=None) -> list[int]:
    """Greedy generation; with `cache` (prefill of a prefix of the prompt)
    only the rest of the prompt is processed - the audio isn't encoded again."""
    with torch.inference_mode():
        if cache is None:
            out = _model.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        else:
            out = _model.model.thinker.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                past_key_values=cache, max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=_EOS_IDS, pad_token_id=_EOS_IDS[0], return_dict_in_generate=True)
    return out.sequences[0, inputs["input_ids"].shape[1]:].tolist()


def _hf_prefill(inputs, n: int):
    """Forward pass over the first `n` prompt tokens (audio included):
    (P("None") at the last position, KV cache)."""
    from transformers import DynamicCache

    cache = DynamicCache()
    with torch.inference_mode():
        out = _model.model.thinker(
            input_ids=inputs["input_ids"][:, :n], attention_mask=inputs["attention_mask"][:, :n],
            input_features=inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
            past_key_values=cache, use_cache=True)
    p_none = None
    if _none_id is not None:
        p_none = float(torch.softmax(out.logits[0, -1].float(), dim=-1)[_none_id])
    return p_none, cache


def _hf_probe(wav: np.ndarray, context: str) -> Optional[float]:
    inputs = _hf_inputs(_probe_prompt(context), wav)
    return _hf_prefill(inputs, inputs["input_ids"].shape[1])[0]


def _hf_probe_and_decode(wav: np.ndarray, context: str, lang: Optional[str], prefix: str,
                         threshold: Optional[float]) -> tuple[Optional[float], Optional[str]]:
    # Unforced without a prefix the prompt is the probe prompt itself: the
    # model's answer starts with "language" anyway.
    head = prefix or ("language" if lang is None else "")
    full = _hf_inputs(_prompt(context, lang) + head, wav)
    if threshold is None:
        return None, head + _decode_ids(_hf_generate(full))
    probe = _hf_inputs(_probe_prompt(context), wav)
    n = probe["input_ids"].shape[1]
    shared = (full["input_ids"].shape[1] >= n
              and torch.equal(full["input_ids"][0, :n], probe["input_ids"][0]))
    p_none, cache = _hf_prefill(probe, n)
    if p_none is not None and p_none >= threshold:
        return p_none, None
    if not shared:
        LOG.warning("probe prompt is no token prefix of the transcription prompt: two passes")
        cache = None
    else:
        cache.crop(n - 1)   # generate needs at least one uncached prompt token
    return p_none, head + _decode_ids(_hf_generate(full, cache))


# --- vLLM backend -----------------------------------------------------------

def _vllm_run(prompt: str, wav: np.ndarray, params) -> object:
    out = _model.model.generate([{"prompt": prompt, "multi_modal_data": {"audio": [wav]}}],
                                sampling_params=params, use_tqdm=False)
    return out[0].outputs[0]


def _vllm_probe(wav: np.ndarray, context: str) -> Optional[float]:
    from vllm import SamplingParams

    if _none_id is None:
        return None
    params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=VLLM_PROBE_LOGPROBS)
    top = _vllm_run(_probe_prompt(context), wav, params).logprobs[0]
    return math.exp(top[_none_id].logprob) if _none_id in top else 0.0


def _vllm_probe_and_decode(wav: np.ndarray, context: str, lang: Optional[str], prefix: str,
                           threshold: Optional[float]) -> tuple[Optional[float], Optional[str]]:
    p_none = None
    if threshold is not None:
        p_none = _vllm_probe(wav, context)
        if p_none is not None and p_none >= threshold:
            return p_none, None
    # Same audio and prompt start as the probe: vLLM's prefix cache has them.
    out = _vllm_run(_prompt(context, lang) + prefix, wav, _model.sampling_params)
    return p_none, prefix + out.text


# --- backend-independent ------------------------------------------------------

def _probe(wav: np.ndarray, context: str) -> Optional[float]:
    """P("None") as the language (None = unknown, counts as speech)."""
    return (_vllm_probe if BACKEND == "vllm" else _hf_probe)(wav, context)


def _probe_and_decode(wav: np.ndarray, context: str, lang: Optional[str], prefix: str,
                      threshold: Optional[float]) -> tuple[Optional[float], Optional[str]]:
    """Speech probe (if `threshold` is set) and decode of all of `wav`.

    Returns (P("None"), raw output incl. `prefix`); raw is None for no speech."""
    fn = _vllm_probe_and_decode if BACKEND == "vllm" else _hf_probe_and_decode
    return fn(wav, context, lang, prefix, threshold)


def _rollback_prefix(raw: str) -> str:
    """Previous raw output minus its last tokens (streaming prefix)."""
    if not raw:
        return ""
    tok = _model.processor.tokenizer
    ids = tok.encode(raw, add_special_tokens=False)
    k = STREAM_ROLLBACK_TOKENS
    while True:
        end = max(0, len(ids) - k)
        prefix = tok.decode(ids[:end]) if end else ""
        if "\ufffd" not in prefix:     # don't cut inside a multi-byte character
            return prefix
        k += 1


def _parse(raw: str, lang: Optional[str]) -> tuple[str, str]:
    from qwen_asr.inference.utils import parse_asr_output
    detected, text = parse_asr_output(raw, user_language=lang)
    return detected, (text or "").strip()


def _after_gpu_work() -> None:
    if BACKEND == "transformers" and DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()


def _decode_audio(src: Path) -> np.ndarray:
    """Decode anything ffmpeg understands to 16 kHz mono float32."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail=f"ffmpeg could not decode audio: {proc.stderr.decode(errors='replace')[:300]}",
        )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    detect_speech: bool = Form(False),
    no_speech_threshold: float = Form(NO_SPEECH_THRESHOLD),
) -> JSONResponse:
    """Transcribe an uploaded audio file.

    Response is `{"text": ...}` plus `language` / `language_fallback` info
    fields (ignored by stt.py), and `no_speech` / `no_speech_prob` with
    `detect_speech`. `model` and `temperature` are accepted for
    compatibility only (greedy decoding).
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if response_format and response_format != "json":
        raise HTTPException(status_code=400, detail=f"unsupported response_format: {response_format}")

    lang, fell_back = _resolve_language(language)

    tmp_src: Optional[Path] = None
    try:
        suffix = Path(file.filename or "audio").suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_src = Path(tmp.name)
        wav = await asyncio.to_thread(_decode_audio, tmp_src)
        duration = len(wav) / SAMPLE_RATE
        empty = {"text": "", "language": "", "language_fallback": fell_back}
        if duration < 0.1:
            return JSONResponse({**empty, **({"no_speech": True, "no_speech_prob": None} if detect_speech else {})})

        def _run():
            with torch.inference_mode():
                return _model.transcribe(
                    audio=(wav, SAMPLE_RATE),
                    context=prompt or "",
                    language=lang,
                )[0]

        speech_info: dict = {}
        async with _model_lock:
            try:
                if detect_speech:
                    p_none = await asyncio.to_thread(_probe, wav, prompt or "")
                    no_speech = p_none is not None and p_none >= no_speech_threshold
                    speech_info = {"no_speech": no_speech, "no_speech_prob": p_none}
                    if no_speech:
                        LOG.info(f"No speech: p_none={p_none:.3f} duration={duration:.1f}s")
                        return JSONResponse({**empty, **speech_info})
                result = await asyncio.to_thread(_run)
            finally:
                _after_gpu_work()

        text = (result.text or "").strip()
        LOG.info(
            f"Transcription done: requested={language!r} forced={lang} detected={result.language!r} "
            f"fell_back={fell_back} duration={duration:.1f}s text_len={len(text)} {speech_info}"
        )
        return JSONResponse({
            "text": text,
            "language": result.language,
            "language_fallback": fell_back,
            **speech_info,
        })
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception(f"Transcription failed: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        if tmp_src and tmp_src.exists():
            try:
                tmp_src.unlink()
            except OSError:
                pass


class _Utterance:
    """Streaming state of one utterance on a WebSocket."""

    def __init__(self, start: dict):
        self.lang, self.fell_back = _resolve_language(start.get("language"))
        self.context = str(start.get("prompt") or "")
        chunk_ms = int(start.get("chunk_ms") or 1000)
        self.chunk_samples = max(SAMPLE_RATE // 5, chunk_ms * SAMPLE_RATE // 1000)
        self.detect_speech = bool(start.get("detect_speech", False))
        self.threshold = float(start.get("no_speech_threshold", NO_SPEECH_THRESHOLD))
        self.partials = bool(start.get("partials", True))
        self.flush = False              # decode what's there as soon as possible
        self.pcm = bytearray()          # s16le, as received
        self.decoded_samples = 0        # audio covered by `raw` / `p_none`
        self.raw = ""
        self.p_none: Optional[float] = None
        self.prefixed = False           # `raw` came from a prefixed decode
        self.step = 0
        self.decode_ms = 0              # of the last step
        self.task: Optional[asyncio.Task] = None

    @property
    def samples(self) -> int:
        return len(self.pcm) // 2

    def wav(self) -> np.ndarray:
        n = self.samples
        return np.frombuffer(bytes(self.pcm[:n * 2]), dtype=np.int16).astype(np.float32) / 32768.0


@app.websocket("/v1/audio/transcriptions/stream")
async def stream_transcriptions(ws: WebSocket) -> None:
    """Streaming transcription, any number of utterances per connection.

    client -> {"type": "start", "language": "de"|null, "prompt": "",
               "chunk_ms": 1000, "detect_speech": false,
               "no_speech_threshold": 0.5, "partials": true}
    client -> binary frames: 16 kHz mono s16le PCM
    server -> {"type": "partial", "text", "no_speech", "no_speech_prob",
              "audio_ms"} whenever another chunk_ms of audio was decoded
              (steps coalesce under load; text "" for no speech), and
              after each flush
    client -> {"type": "partials", "enabled": bool}: decode every chunk_ms
              or only on flush/finish (from the next step on)
    client -> {"type": "flush"}: the caller went quiet - decode all audio so
              far now (after a running step), like a final decode, so that
              a finish within tail_silence_ms of it needs no decode of its own
    client -> {"type": "finish", "tail_silence_ms": 0} - how much of the
              audio's end is known silence (the client's end-of-speech
              window): a step covering the rest makes a new decode unneeded
    server -> {"type": "final", "text", "language", "language_fallback",
               "no_speech", "no_speech_prob", "audio_ms", "finish_ms"}
    client -> {"type": "cancel"} drops the utterance (no reply)
    server -> {"type": "error", "detail"} on protocol errors (stays open)
    Audio beyond QWEN_ASR_STREAM_MAX_SECONDS is ignored.
    """
    await ws.accept()
    send_lock = asyncio.Lock()
    utt: Optional[_Utterance] = None
    max_bytes = int(STREAM_MAX_SECONDS * SAMPLE_RATE) * 2

    async def send(msg: dict) -> None:
        async with send_lock:
            await ws.send_text(json.dumps(msg))

    async def decode(u: _Utterance, wav: np.ndarray, final: bool = False) -> bool:
        """Speech probe (if asked) + decode of `wav`; False = no speech.
        A final decode never uses a prefix (same result as the POST endpoint)."""
        prefixed = not final and len(wav) > STREAM_PREFIX_AFTER_SECONDS * SAMPLE_RATE
        prefix = _rollback_prefix(u.raw) if prefixed else ""
        t0 = time.monotonic()
        async with _model_lock:
            try:
                p_none, raw = await asyncio.to_thread(
                    _probe_and_decode, wav, u.context, u.lang, prefix,
                    u.threshold if u.detect_speech else None)
            finally:
                _after_gpu_work()
        u.decode_ms = int((time.monotonic() - t0) * 1000)
        u.step += 1
        u.decoded_samples = len(wav)
        u.prefixed = prefixed
        u.p_none = p_none
        u.raw = raw or ""
        return raw is not None

    async def decode_step(u: _Utterance, final: bool = False) -> None:
        wav = u.wav()
        speech = await decode(u, wav, final)
        _, text = _parse(u.raw, u.lang)
        await send({"type": "partial", "text": text if speech else "", "no_speech": not speech,
                    "no_speech_prob": u.p_none, "audio_ms": len(wav) * 1000 // SAMPLE_RATE})

    def maybe_step(u: _Utterance) -> None:
        """Start a decode step if none is running and one is due (called for
        every audio frame and control message)."""
        if u.task is not None and not u.task.done():
            return
        new = u.samples - u.decoded_samples
        if u.flush and new > 0:
            u.flush = False
            u.task = asyncio.create_task(decode_step(u, final=True))
        elif u.partials and new >= u.chunk_samples:
            u.task = asyncio.create_task(decode_step(u))

    async def finish(u: _Utterance, tail_silence_ms: int) -> None:
        t0 = time.monotonic()
        if u.task is not None:
            try:
                await u.task
            except Exception:
                LOG.exception("stream: partial decode failed")
        wav = u.wav()
        audio_ms = len(wav) * 1000 // SAMPLE_RATE
        # Only silence after the last step: its result is final.
        covered = (u.step > 0 and not u.prefixed and
                   len(wav) - u.decoded_samples <= tail_silence_ms * SAMPLE_RATE // 1000)
        if len(wav) < SAMPLE_RATE // 10:
            speech = False
        elif covered:
            speech = not (u.p_none is not None and u.p_none >= u.threshold)
        else:
            speech = await decode(u, wav, final=True)
        detected, text = _parse(u.raw, u.lang) if speech else ("", "")
        finish_ms = int((time.monotonic() - t0) * 1000)
        LOG.info(f"stream final: audio={audio_ms}ms steps={u.step} reused={covered} "
                 f"finish={finish_ms}ms last_decode={u.decode_ms}ms p_none={u.p_none} "
                 f"text_len={len(text)}")
        await send({"type": "final", "text": text, "language": detected,
                    "language_fallback": u.fell_back, "no_speech": not speech,
                    "no_speech_prob": u.p_none, "audio_ms": audio_ms, "finish_ms": finish_ms})

    def drop(u: Optional[_Utterance]) -> None:
        if u is not None and u.task is not None and not u.task.done():
            u.task.cancel()     # the to_thread call finishes, its result is discarded

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if utt is None:
                    await send({"type": "error", "detail": "audio before start"})
                    continue
                room = max_bytes - len(utt.pcm)
                if room > 0:
                    utt.pcm.extend(msg["bytes"][:room])
                maybe_step(utt)
                continue
            try:
                ctrl = json.loads(msg.get("text") or "")
                kind = ctrl.get("type")
            except (ValueError, AttributeError):
                await send({"type": "error", "detail": "expected a JSON object"})
                continue
            if _model is None:
                await send({"type": "error", "detail": "model not loaded"})
            elif kind == "start":
                drop(utt)
                utt = _Utterance(ctrl)
            elif kind == "flush":
                if utt is not None:
                    utt.flush = True
                    maybe_step(utt)
            elif kind == "partials":
                if utt is not None:
                    utt.partials = bool(ctrl.get("enabled"))
                    maybe_step(utt)
            elif kind == "finish" and utt is not None:
                u, utt = utt, None
                await finish(u, max(0, int(ctrl.get("tail_silence_ms") or 0)))
            elif kind == "cancel":
                drop(utt)
                utt = None
            else:
                await send({"type": "error", "detail": f"unexpected message {kind!r}"})
    except WebSocketDisconnect:
        pass
    except Exception:
        LOG.exception("stream: connection failed")
    finally:
        drop(utt)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8001")))
