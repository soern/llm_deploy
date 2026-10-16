# kokoro-tts-english

English text-to-speech HTTP service backed by [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (PyTorch model with 54 voices across 8 languages, here configured for **American & British English**).

Used by nixclaw's `text_to_speech` tool to turn agent replies into Signal voice-note attachments. Ships **CPU-capable by default**, like `deploy/cpu/kokoro-tts-german/` — an 82M-parameter model runs comfortably on CPU. This server is a natural pair to the German Martin voice: while the German model is ONNX and German-only, the English model is PyTorch and supports 20 English voices.

## Voices

**American English** (11 female, 9 male):
- **Top quality**: `af_heart` (A), `af_bella` (A-), `af_nicole` (B-)
- `af_alloy`, `af_aoede`, `af_jessica`, `af_kore`, `af_nova`, `af_river`, `af_sarah`, `af_sky`
- `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`

**British English** (4 female, 4 male):
- Female: `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`
- Male: `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

Default voice is `af_heart` (warm, emotional American female). See [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) on HuggingFace for quality assessments and samples.

## Model recon findings

These were **verified live** against the real HuggingFace repo on 2026-01-27:

| Question | Finding |
|---|---|
| Model file | `kokoro-v1_0.pth` — single PyTorch weight file |
| Voices | Directory `voices/` with individual `.pt` files per voice (one tensor per voice) |
| Voice names | `af_heart`, `af_bella`, `am_michael`, `bf_emma`, `bm_daniel`, etc. (20 English total) |
| Supported languages | 8: American English, British English, Japanese, Mandarin Chinese, Spanish, French, Hindi, Italian, Brazilian Portuguese (no German) |
| Phonemizer | `kokoro-onnx` library uses `phonemizer-fork` + `espeakng-loader`, respects lang='en' for American English phonemes (espeak-ng en-us fallback) |
| Sample rate | **24000 Hz mono float32** — identical to German Kokoro |
| Model source | `hexgrad/Kokoro-82M` (January 2025 release, v1.0) |

**What this means for `server.py`:** Load the PyTorch model with `torch.load(kokoro-v1_0.pth)`, load voice tensors individually from `voices/*.pt` files, and use the `kokoro-onnx` library's `Kokoro.create()` method to synthesize (which works with both ONNX and PyTorch backends). Phonemizer language is plain `"en"` (espeak-ng will handle American vs. British via the voice's accent training).

## Endpoints

- `GET  /health` — liveness + model id / available voices / device / sample rate
- `GET  /v1/voices` — `{"voices": [...], "default_voice": "af_heart", "default_language": "en"}`
- `POST /v1/audio/speech` — OpenAI-shaped JSON body:
  ```json
  {"input": "Hello, how are you?", "voice": "af_heart", "language": "en",
   "response_format": "aac", "speed": 1.0}
  ```
  → `200`, binary audio body, `Content-Type` per `response_format`
  (`audio/aac` for `aac`, `audio/mp4` for `m4a`, `audio/mpeg` for `mp3`,
  `audio/ogg` for `ogg`, `audio/wav` for `wav`). Default `response_format` is
  `aac` (raw ADTS AAC, matching Signal voice notes).

Server-side guards: `input` longer than `MAX_INPUT_CHARS` (default **20000**) is rejected with `400`; longer inputs are chunked on sentence boundaries and concatenated (Kokoro degrades on very long inputs). `language` must be `en`, `en-us`, or `en-gb` (`400` otherwise — normalized to `en` for the phonemizer). `speed` must be between `0.5` and `2.0` (kokoro-onnx's asserted range).

ffmpeg transcoding runs via `asyncio.create_subprocess_exec` (not blocking), after the `asyncio.to_thread`-wrapped PyTorch inference has finished — so it never stacks a second blocking call on top of that thread.

## CPU vs GPU

Ships **CPU-only by default** (`ORT_PROVIDER=cpu` in the Dockerfile, `python:3.11-slim` base). An 82M model runs comfortably on CPU — this keeps the DGX Spark's GPU free for vLLM + FLUX.2 if deployed there.

`ORT_PROVIDER=cuda` is present as a documented opt-in (inherited from the German server pattern, though this PyTorch model doesn't use ONNX Runtime). For real CUDA support, you'd need to:
1. Build with a CUDA-capable base image
2. Swap `torch==2.5.1` for `torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118` (or your CUDA version)

`start.sh` can be extended to add `--gpus all` if desired, but defaults to CPU.

## Build & run

```bash
# Plain docker-compose (mirrors deploy/cpu/kokoro-tts-german/start.sh):
cd deploy/cpu/kokoro-tts-english
./start.sh          # builds the image, downloads the model, waits for readiness
```

`start.sh` builds the image from the local `Dockerfile`, streams container logs to `.kokoro-tts-en.log`, and polls `http://127.0.0.1:8005/health` until ready. The model download is ~300 MB (PyTorch model + 20 voice .pt files), so this should take 30-60 seconds on a typical network.

```bash
./stop.sh
```

## Verify

```bash
curl -fsS http://localhost:8005/health | jq
curl -fsS http://localhost:8005/v1/voices | jq

curl -fsS -X POST http://localhost:8005/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello world.","language":"en","response_format":"aac"}' \
  -o /tmp/t.aac
```

(Port `8005` is the default for this English server, distinct from 8003 German and 8002 image gen.)

## Model assets

On first start, if `MODEL_DIR` (default `/models/kokoro-en`) doesn't contain `kokoro-v1_0.pth`, the server downloads `TTS_MODEL_ID` from HuggingFace via `snapshot_download` (restricted to `*.pth`/`voices/*.pt` files) and stores it in the `model-cache` volume (compose) or `.cache/models/` next to `start.sh` (plain docker). Subsequent restarts reuse the cached weights.

## Known limitations

- No English text normalization (numbers, currencies, abbreviations) — spell out things like "100 dollars" or "Dr. Smith" in the input text for natural reading.
- Sentence-boundary chunking (`_split_sentences` in `server.py`) is a plain regex split on `. ! ?`, not aware of abbreviations (e.g., "Mr.", "Inc."). A sentence ending in a dotted abbreviation may get split early.
- Single in-process worker (one `asyncio.Lock` around inference) — no process-pool parallelism. Fine for CPU or a single-GPU setup; revisit if concurrent TTS requests become a bottleneck.
- PyTorch model requires more RAM than ONNX (~600 MB for model + voice tensors vs. ~80 MB for ONNX German). CPU inference is slower than ONNX (real-time or slower, depending on text length and CPU).

## Deployment notes

### Local CPU testing

This server is designed for **local CPU inference** first. Typical latency on a modern CPU (e.g., Ryzen 5000 or Intel 12th-gen):
- **Short text** (20 tokens): 2-4 seconds
- **Medium text** (100 tokens): 5-10 seconds
- **Long text** (500+ tokens): 15-30+ seconds

If this is too slow for your use case, enable GPU (set `ORT_PROVIDER=cuda` and build with CUDA torch).

### Production / DGX deployment

If deploying to the DGX Spark:
1. Use the same `docker compose` commands from the local machine.
2. Ensure the machine running this has network access to the DGX (if remote).
3. Update `config/agent.yaml` to point `tts.engines.en.base_url` to the DGX IP/port (e.g., `http://gx10-811e.lan:8005`).
4. Consider enabling GPU (CUDA) if concurrent requests exceed CPU capacity.

## Configuration

Environment variables (overridable at runtime):

| Variable | Default | Purpose |
|----------|---------|---------|
| `TTS_MODEL_ID` | `hexgrad/Kokoro-82M` | HuggingFace model repo |
| `MODEL_DIR` | `/models/kokoro-en` | Cache directory for model + voices |
| `DEFAULT_LANG` | `en` | Default language for synthesis |
| `DEFAULT_VOICE` | `af_heart` | Default voice (must exist in voices/) |
| `PORT` | `8000` | Internal port (mapped to 8005 by compose) |
| `ORT_PROVIDER` | `cpu` | Execution provider (cpu\|cuda) |
| `HF_TOKEN` | (none) | Optional HuggingFace API token for faster downloads |

## Notes

- A single `asyncio.Lock` around `Kokoro.create()` serializes requests, same pattern as the German server and faster-whisper.
- `/health` reports the currently loaded voices and device so a misconfiguration is visible without digging through logs.
- **Contrast with German server**: German is ONNX (80 MB, fast CPU inference, 1 voice), English is PyTorch (300 MB, slower CPU inference, 20 voices). They pair well: deploy both for a multilingual agent.
