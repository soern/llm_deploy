"""Canary-1b-v2 NVFP8 STT server (DGX Spark target).

OpenAI-shaped /v1/audio/transcriptions endpoint backed by NVIDIA's
Canary-1b-v2 multitask ASR model (FastConformer encoder + Transformer
decoder, 978M params, 25 European languages), with the Linear layers
quantized to NVFP8 (FP8 e4m3 weights + per-1x128 block scales) to fit
comfortably in ~1.5-2.5 GB VRAM.

Mirrors the request/response contract of deploy/dgx-spark/faster-whisper and
deploy/cpu/canary-1b-flash so nixclaw's stt.py client works unchanged - just
repoint STT_BASE_URL.

Language handling (no auto-detect on this model - it needs an explicit
language prompt):
  - no/empty `language` param  -> CANARY_DEFAULT_LANG (de)
  - unsupported `language`     -> CANARY_FALLBACK_LANG (en), flagged in
    the response via "language_fallback": true
"""

import logging
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

LOG = logging.getLogger("canary-nvfp8")

MODEL_ID = os.environ.get("CANARY_MODEL_ID", "nvidia/canary-1b-v2")
DEFAULT_LANG = os.environ.get("CANARY_DEFAULT_LANG", "de").lower()
FALLBACK_LANG = os.environ.get("CANARY_FALLBACK_LANG", "en").lower()
QUANT_ENABLED = os.environ.get("CANARY_QUANT", "1") != "0"

# 25 languages of canary-1b-v2 (README "Supported Languages").
SUPPORTED_LANGS = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
    "sl", "es", "sv", "ru", "uk",
}

# Canary-1b-v2 ships as transformers 5.x architecture; load in BF16.
_DTYPE = torch.bfloat16

_model = None
_processor = None
_model_lock = None  # asyncio.Lock


def _detect_device() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


DEVICE = os.environ.get("DEVICE") or _detect_device()


class NVFP8Linear(torch.nn.Module):
    """Linear layer backed by FP8 e4m3 weights + per-1x128 FP32 scales.

    Dequantizes the weight to the input dtype on each forward pass
    (deterministic, version-robust; a torch._scaled_mm FP8 GEMM fast path
    can be added later as an optimization).
    """

    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor, bias, block: int, dtype):
        super().__init__()
        out_f, in_f = qweight.shape
        self.in_features = in_f
        self.out_features = out_f
        self.block = block
        self.dtype = dtype
        # registered buffers follow the module to its device on .to()/.cuda()
        self.register_buffer("qweight", qweight)      # uint8 (view of e4m3)
        self.register_buffer("scale", scale)          # fp32 [out, in//block]
        self.register_buffer("bias", bias)

    def _dequant(self, dtype) -> torch.Tensor:
        q = self.qweight.view(torch.float8_e4m3fn).view(
            self.out_features, self.in_features // self.block, self.block
        )
        return (q.to(dtype) * self.scale.unsqueeze(-1)).view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        # dequantize in FP32 for accuracy, then cast weight back to input dtype
        w = self._dequant(torch.float32).to(x.dtype)
        y = F.linear(x2, w, self.bias)
        return y.reshape(*shape[:-1], self.out_features)


def _apply_quantized(model, quant: dict, block: int) -> None:
    """Replace all Linear layers of ``model`` with NVFP8-backed ones.

    Runs on CPU; the whole model is moved to the target device afterwards.
    """
    import quantize
    for fqn, mod in quantize._iter_linear_modules(model):
        qw = quant[fqn + ".qweight"].to(torch.uint8)
        sc = quant[fqn + ".scale"].to(torch.float32)
        bias_key = fqn + ".bias"
        bias = quant[bias_key].to(_DTYPE) if bias_key in quant else None
        new = NVFP8Linear(qw, sc, bias, block, _DTYPE)
        parent = model
        parts = fqn.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new)
    LOG.info("Applied NVFP8 quantized Linear layers to model")


def _load_model() -> None:
    global _model, _model_lock, _processor
    import asyncio
    import quantize
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    LOG.info(f"Loading {MODEL_ID} (device={DEVICE}, quant={QUANT_ENABLED})")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype=_DTYPE)

    if QUANT_ENABLED:
        cache_dir = Path(os.environ.get("QUANT_CACHE_DIR", "/models/canary-1b-v2-nvfp8"))
        safe_id = MODEL_ID.replace("/", "__")
        cache = cache_dir / f"{safe_id}.nvfp8.pt"
        if cache.exists():
            LOG.info(f"Reusing NVFP8 cache: {cache}")
            ckpt = quantize.load_quantized(cache)
        else:
            LOG.info("No NVFP8 cache found - quantizing now (one-time, ~1-2 min)")
            quant = quantize.build_quantized_state(model)
            quantize.save_quantized(quant, MODEL_ID, str(cache_dir))
            ckpt = {"block": quantize.BLOCK, "quant": quant}
        _apply_quantized(model, ckpt["quant"], ckpt["block"])
        del ckpt  # free RAM, weights now live inside the modules

    model.to(DEVICE)
    model.eval()
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = model
    _model_lock = asyncio.Lock()
    LOG.info("Model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="canary-1b-v2-nvfp8-server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_id": MODEL_ID, "device": DEVICE, "quant": QUANT_ENABLED}


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


def _resolve_language(requested: Optional[str]) -> tuple[str, bool]:
    """Default -> de; unsupported request -> fallback (en), flagged."""
    lang = (requested or "").strip().lower()
    if not lang:
        return DEFAULT_LANG, False
    if lang in SUPPORTED_LANGS:
        return lang, False
    LOG.warning(f"Unsupported language {requested!r} - falling back to {FALLBACK_LANG!r}")
    return FALLBACK_LANG, True


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

    Mirrors the OpenAI `POST /v1/audio/transcriptions` shape (same as
    faster-whisper-server / canary-1b-flash). Response is `{"text": ...}`
    plus `language` / `language_fallback` info fields (ignored by stt.py).
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if response_format and response_format != "json":
        raise HTTPException(status_code=400, detail=f"unsupported response_format: {response_format}")

    lang, fell_back = _resolve_language(language)

    tmp_src: Optional[Path] = None
    tmp_wav: Optional[Path] = None
    try:
        suffix = Path(file.filename or "audio").suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_src = Path(tmp.name)
        tmp_wav = tmp_src.with_suffix(".16k.wav")
        _convert_to_wav(tmp_src, tmp_wav)

        import asyncio
        async with _model_lock:
            # run blocking generate on a thread so the event loop stays free
            def _gen():
                inputs = _processor.apply_transcription_request(
                    audio=_load_wav(tmp_wav),
                    source_language=lang,
                    target_language=lang,
                ).to(DEVICE)
                with torch.inference_mode():
                    ids = _model.generate(**inputs, max_new_tokens=512)
                return _processor.decode(ids, skip_special_tokens=True)

            result = await asyncio.to_thread(_gen)
            text = result[0].strip() if isinstance(result, list) else str(result).strip()

        LOG.info(f"Transcription done: lang={lang} fell_back={fell_back} text_len={len(text)}")
        return JSONResponse({
            "text": text,
            "language": lang,
            "language_fallback": fell_back,
        })
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


def _load_wav(path: Path) -> list:
    """Load 16kHz mono WAV into a list of float32 numpy arrays (one sample)."""
    import numpy as np
    import wave

    with wave.open(str(path), "rb") as wf:
        assert wf.getsampwidth() == 2, f"expected 16-bit PCM, got {wf.getsampwidth()}"
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return [audio]


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8006")))
