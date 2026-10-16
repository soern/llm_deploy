# canary-1b-flash

OpenAI-shaped audio transcription HTTP service backed by NVIDIA NeMo's
[Canary-1b-flash](https://huggingface.co/nvidia/canary-1b-flash) (FastConformer
encoder + Transformer decoder, 883M params). Originally evaluated as a
replacement for `deploy/dgx-spark/faster-whisper/`, but benchmarking against
`tests/fixtures/*.aac` showed it's **not** a safe drop-in: it silently
returns an empty transcript (no error, `score=0.0`) on noisy or
code-switched audio, and it has no language auto-detection at all (`source_lang`/
`target_lang` must always be supplied). It's occasionally *better* than
whisper on clean narration, though - see git history on this README for the
full writeup.

Given that mixed track record, this backend is wired into nixclaw only as
an on-demand second opinion (`retranscribe_audio` tool), not as the
automatic voice-message pipeline - see "Wiring into nixclaw" below.

This server mirrors `deploy/dgx-spark/faster-whisper/server.py`'s exact request/
response contract, so `src/nixclaw_agent/stt.py` (which only speaks that
generic OpenAI shape) works against either backend unchanged.

## Endpoints

- `GET  /health`                  — liveness + model/device info
- `GET  /v1/models`               — OpenAI-compatible model list (one entry)
- `POST /v1/audio/transcriptions` — multipart upload, returns `{"text": "..."}`

The `POST` body is multipart/form-data:
- `file` — the audio file (anything ffmpeg decodes; resampled to 16kHz mono internally)
- `model` — informational only (echoed back in logs)
- `language` — one of `en`/`de`/`es`/`fr` (Canary-1b-flash's supported set); defaults to `CANARY_DEFAULT_LANG`
- `response_format` — must be `"json"` (other formats rejected)
- `temperature` — accepted for contract parity, currently unused (Canary decodes greedily via its baked decoding config)

Unlike faster-whisper, Canary is *not* a drop-in for arbitrary languages -
requesting a `language` outside `en/de/es/fr` returns HTTP 400.

## Build & run (CPU test)

```bash
docker compose -f deploy/cpu/canary-1b-flash/docker-compose.yml up -d --build
# first start downloads ~3.5 GB of weights into the model-cache volume; watch:
docker logs -f canary-1b-flash
```

## Verify

```bash
curl -s localhost:8006/health
curl -s localhost:8006/v1/models

curl -s -F file=@tests/fixtures/voice_test.aac -F language=de \
    localhost:8006/v1/audio/transcriptions | jq
```

## Memory / compute

NVIDIA doesn't publish an inference VRAM spec for Canary-1b-flash. At 883M
params it's smaller than whisper-large-v3 (~1.5B); expect roughly **2-3 GB
VRAM** in fp16 on a GPU, comparable to or less than the `large-v3` row in
`deploy/dgx-spark/faster-whisper/README.md`'s table. Take this as an estimate,
not a spec-sheet number - re-measure with `nvidia-smi` once running on the
target GPU.

CPU inference (this image's default) works via NeMo's `map_location="cpu"`
fallback but is meant for correctness testing only, not production latency -
expect it to be considerably slower than whisper-large-v3 on CPU, since
Canary's encoder-decoder wasn't optimized for CPU throughput the way
CTranslate2's int8 kernels are.

## GPU deployment (not wired up here)

This Dockerfile is CPU-only (`python:3.11-slim` + CPU torch wheel). For GPU:
swap the base image for an `nvidia/cuda:*-runtime` image (as
`faster-whisper-server`'s Dockerfile does), install the CUDA build of torch
instead of the `whl/cpu` index, and uncomment the `deploy.resources` block in
`docker-compose.yml`.

## Notes

- Canary-1b-flash also supports translation (e.g. `de`→`en`) via its
  `source_lang`/`target_lang` task args; this server always sets them equal
  (pure ASR), matching what `stt.py` needs.
- No request-level batching: `_model_lock` serializes one transcription at a
  time, same rationale as `faster-whisper-server`.

## Wiring into nixclaw

This service is reached only through the `retranscribe_audio` tool
(`src/nixclaw_agent/tools/stt_ops.py`), never through the automatic
voice-message pipeline. To enable it, set in `config/agent.yaml`:

```yaml
stt:
  alternate:
    enabled: true
    base_url: "http://<host-running-this-service>:8006"
    language: null   # or force a default, e.g. "de" - no auto-detect on this backend
```

(or the `STT_ALT_ENABLED` / `STT_ALT_BASE_URL` / `STT_ALT_MODEL` /
`STT_ALT_LANGUAGE` / `STT_ALT_TIMEOUT` env vars). The agent calls this
tool itself when it thinks the primary (`faster-whisper-large-v3`)
transcript looks garbled - see the "★ VOICE REPLIES" note in `llm.py`'s
system prompt.
