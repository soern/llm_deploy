# faster-whisper-server

OpenAI-shaped audio transcription HTTP service backed by [faster-whisper]
(CTranslate2). Used by nixclaw to turn inbound Signal voice messages into
text before they reach the LLM turn.

## Endpoints

- `GET  /health`                  — liveness + model/dtype info
- `GET  /v1/models`               — OpenAI-compatible model list (one entry)
- `POST /v1/audio/transcriptions` — multipart upload, returns `{"text": "..."}`

The `POST` body is multipart/form-data:
- `file` — the audio file (WAV / MP3 / M4A / FLAC / OGG / WEBM; anything ffmpeg decodes)
- `model` — informational (echoed back in logs); the server has a single baked model
- `language` — optional ISO-639-1 code (`"de"`, `"en"`, ...); omit for auto-detect
- `response_format` — must be `"json"` (other formats rejected)
- `temperature` — decoding temperature (default 0.0)

## Build & run

```bash
# Production (GPU host):
WHISPER_MODEL_ID=Systran/faster-whisper-large-v3 \
COMPUTE_TYPE=float16 \
docker compose -f deploy/dgx-spark/faster-whisper/docker-compose.yml up -d --build

# CPU testing (no GPU):
WHISPER_MODEL_ID=Systran/faster-whisper-base \
COMPUTE_TYPE=int8 \
FW_PORT=8001 \
docker compose -f deploy/dgx-spark/faster-whisper/docker-compose.yml up -d --build
```

The image uses `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` so it works
on both GPU and CPU hosts. On startup it autodetects CUDA via
`ctranslate2.get_device_count("cuda")`; if zero, it falls back to CPU +
int8. To force CPU on a GPU host, set `COMPUTE_TYPE=int8`.

## Verify

```bash
# Health:
curl -s localhost:8001/health
# Models:
curl -s localhost:8001/v1/models
# Transcription (any short wav/mp3):
curl -s -F file=@sample.wav -F language=de \
    localhost:8001/v1/audio/transcriptions | jq
```

## Model assets

On first start, if `MODEL_DIR` (default `/models/<basename>`) is empty, the
server downloads `WHISPER_MODEL_ID` from HuggingFace via `snapshot_download`
and stores it in the `model-cache` volume. Subsequent restarts reuse the
cached weights.

To pre-bake weights into the image instead, drop a CTranslate2 export at
`deploy/dgx-spark/faster-whisper/<basename>/` (with `model.bin`, `config.json`,
`tokenizer.json`, `vocabulary.json`, `preprocessor_config.json`) and add a
`COPY <basename>/ $MODEL_DIR` line to the Dockerfile. No HF token needed
at runtime in that mode.

## Memory / VRAM

| Model | CPU RAM (int8) | GPU VRAM (float16) |
|---|---|---|
| `tiny`  | ~0.4 GB | ~0.4 GB |
| `base`  | ~0.5 GB | ~0.5 GB |
| `small` | ~1.0 GB | ~1.0 GB |
| `large-v3` | ~2.5 GB | ~2.8 GB |

For local CPU testing on 16 GB+ machines, `base` is the sweet spot
(transcribes ~20 s of German audio in ~5–10 s). `large-v3` on CPU is
~10–30× real-time and is only practical on a GPU host.

## Notes

- `chunk_length` is the same 30 s hard limit as upstream Whisper. The
  nixclaw client chunks longer clips server-side before sending — see
  `src/nixclaw_agent/stt.py`.
- A single `asyncio.Lock` around `model.transcribe()` serializes requests
  so two simultaneous uploads don't fight for memory at once.
- VAD filtering (`vad_filter=True`) is intentionally left off; we want
  predictable behaviour on short clips and noisy crowd audio.