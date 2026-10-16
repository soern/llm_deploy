# kokoro-tts-german

German text-to-speech HTTP service backed by
[Kokoro-82M-ONNX-German-Martin](https://huggingface.co/Godelaune/Kokoro-82M-ONNX-German-Martin)
(single male voice, "Martin"). Name drops the `-dgx-spark` suffix on
purpose: this one is CPU-capable and not Spark-specific — an 82M-parameter
ONNX model runs comfortably on CPU. Used by nixclaw's planned
`text_to_speech` tool to turn agent replies into a Signal voice-note
attachment.

**German only.** This model has one voice ("Martin") and one language
("de"). There is no way to make this specific container serve another
language — to add e.g. English, deploy a second container with a different
model and point a second `tts.engines.en.base_url` at it (see
`.tmp-reports/PLAN_TTS.md` step 4).

## Model recon findings

These were **verified live** against the real HuggingFace repo on
2026-08-25 (network access to `huggingface.co` was available in the sandbox
that authored this deploy — not everything below had to be assumed):

```bash
python3 -c "from huggingface_hub import list_repo_files; \
  print(list_repo_files('Godelaune/Kokoro-82M-ONNX-German-Martin'))"
# ['.gitattributes', 'README.md', 'docker-compose.yml',
#  'german_text_rules.py', 'kokoro-martin.onnx',
#  'martin-onnx-beispiel-0.25pause.mp3', 'martin-onnx-beispiel-v1.1.mp3',
#  'martin-onnx-beispiel.mp3', 'onnx-docker/Dockerfile',
#  'onnx-docker/README.md', 'onnx-docker/main.py',
#  'onnx-docker/tts_normalizer.py', 'voices-martin.npz',
#  'wyoming_openai_german_separator/Dockerfile',
#  'wyoming_openai_german_separator/README.md',
#  'wyoming_openai_german_separator/apply_german_separator_patch.py']
```

| Question | Finding |
|---|---|
| ONNX filename | `kokoro-martin.onnx` — single file, no separate fp32/int8 quantized variants in this repo |
| Voices file | `voices-martin.npz` — numpy `.npz`, one voice: `"martin"` |
| Voice name(s) | `"martin"` (the repo's only voice; README calls it "Martin Harbecke") |
| Phonemizer | The upstream author's own reference service (`onnx-docker/main.py` + `onnx-docker/Dockerfile`, also downloaded and inspected) uses the [`kokoro-onnx`](https://pypi.org/project/kokoro-onnx/) PyPI package (pinned `==0.5.0` there). Inspecting that package's wheel: `kokoro_onnx/tokenizer.py` phonemizes via `phonemizer-fork` + `espeakng-loader` (which bundles its own espeak-ng shared library and data), falling back to a system-wide espeak-ng library if the bundled one fails to load. The language code is passed straight through to `phonemizer.phonemize(text, lang, ...)` — the upstream reference service sets `KOKORO_ONNX_LANG=de` directly, confirming plain `"de"` is the right code (no `de-de`/`de-DE` variant needed). |
| Sample rate | **24000 Hz mono float32** — confirmed in both `kokoro_onnx/config.py` (`SAMPLE_RATE = 24000`) and the upstream reference service's own `SAMPLE_RATE = 24000` / `wav_bytes()` helper. |
| Model loading API | `kokoro_onnx.Kokoro.from_session(onnxruntime_session, voices_path)` — lets the caller build its own `onnxruntime.InferenceSession` (with whatever providers/session options it wants) and hand it to the library, rather than letting the library manage the session itself. This server uses that path so `ORT_PROVIDER` controls the provider list directly. |
| Voice indexing | Kokoro-82M voices are per-input-length style vectors: `voices[name]` is an array indexed by *token count* (`voice[len(tokens)]`), same as upstream `hexgrad/Kokoro-82M`'s voice-pack format — not something this deploy needs to special-case, `kokoro-onnx` handles it internally. |

**What this means for `server.py`:** the loader hardcodes
`ONNX_FILENAME=kokoro-martin.onnx` / `VOICES_FILENAME=voices-martin.npz` as
defaults (overridable via env var, in case a future model swap uses
different names), builds its own `onnxruntime.InferenceSession`, and passes
it into `Kokoro.from_session()`. `DEFAULT_LANG=de` is sent unmodified as the
`lang=` argument to `Kokoro.create()`.

**Not ported from the upstream author's reference service** (kept out of
scope for this deploy, see "Known limitations" below): the German text
normalization layer (`german_text_rules.py` / `onnx-docker/tts_normalizer.py`
— expands dates, decimal numbers, units, abbreviations, e.g. "2,5 kWh" →
"zwei komma fünf Kilowattstunden") and the multi-worker process-pool
parallelism (`KOKORO_WORKERS`). This server does a single in-process model
behind one `asyncio.Lock`, matching the simpler pattern already used by
`deploy/dgx-spark/faster-whisper/server.py` and
`deploy/dgx-spark/flux2-klein/server.py`.

## Endpoints

- `GET  /health` — liveness + model id / available voices / ORT provider / sample rate
- `GET  /v1/voices` — `{"voices": [...], "default_voice": "martin", "default_language": "de"}`
- `POST /v1/audio/speech` — OpenAI-shaped JSON body:
  ```json
  {"input": "Guten Morgen.", "voice": "martin", "language": "de",
   "response_format": "aac", "speed": 1.0}
  ```
  → `200`, binary audio body, `Content-Type` per `response_format`
  (`audio/aac` for `aac`, `audio/mp4` for `m4a`, `audio/mpeg` for `mp3`,
  `audio/ogg` for `ogg`, `audio/wav` for `wav`). Default `response_format` is
  `aac` — raw ADTS AAC (no MP4 box), matching what real Signal voice notes
  actually are (see `src/nixclaw_agent/stt.py`'s `_AUDIO_TYPES`). `m4a`
  (MP4-boxed AAC) is still supported but is no longer the default: a
  fragmented-MP4-boxed AAC file with no video track is ambiguous enough for
  magic-byte sniffing that the Signal client can misrender it as a video
  instead of a voice note.

Server-side guards: `input` longer than `MAX_INPUT_CHARS` (default
**20000**, matching `TTSEngineConfig.max_input_chars` on the agent side so
the client-side check and this server-side rejection agree) is rejected
with `400`; longer inputs are chunked on rough sentence boundaries and
concatenated (Kokoro degrades on very long single inputs). `language` must
be exactly `"de"` (`400` otherwise — this is a German-only model, so
silently mis-synthesizing under a wrong phonemizer language code would be
worse than a clear rejection). `speed` must be between `0.5` and `2.0`
(`kokoro-onnx`'s own asserted range).

ffmpeg transcoding runs via `asyncio.create_subprocess_exec` (not blocking
`subprocess.run`), after the `asyncio.to_thread`-wrapped ORT inference has
already finished — so it never stacks a second blocking call on top of that
thread. A nonzero ffmpeg exit returns `500` with the captured stderr
(truncated to 2000 chars) as the body.

## CPU vs GPU

Ships **CPU-only** by default (`ORT_PROVIDER=cpu`, `python:3.11-slim`
base). An 82M ONNX model runs comfortably on CPU — this keeps the DGX
Spark's GPU free for vLLM + FLUX.2 and removes an operational risk for what
is a genuinely small model.

`ORT_PROVIDER=cuda` is present as a documented opt-in switch, but **using
it for real requires rebuilding the image**: swap `onnxruntime` for
`onnxruntime-gpu` in the `Dockerfile`'s pip install and use a CUDA-capable
base image (this directory's `Dockerfile` does not do that — it's
CPU-only, intentionally, per the plan). If `ORT_PROVIDER=cuda` is set but
`CUDAExecutionProvider` isn't actually available (e.g. because you didn't
rebuild for GPU), `server.py` logs a warning and falls back to CPU rather
than crashing — see `_build_providers()`.

`start.sh` only adds `--gpus all` to the `docker run` when
`ORT_PROVIDER=cuda` is set in the environment.

## Build & run

```bash
# Plain docker (mirrors deploy/dgx-spark/flux2-klein/start.sh):
cd deploy/cpu/kokoro-tts-german
./start.sh          # builds the image, downloads the model, waits for readiness
```

```bash
# Or docker compose:
docker compose -f deploy/cpu/kokoro-tts-german/docker-compose.yml up -d --build
```

`start.sh` builds the image from the local `Dockerfile`, streams container
logs to `.kokoro-tts.log`, and polls `http://127.0.0.1:8003/health` until
ready. The model download is small (~80 MB total for the onnx + npz files),
so this should be quick even on a first run — nothing like the tens of
gigabytes the whisper/flux2 sidecars pull.

```bash
./stop.sh
```

## Verify

```bash
curl -fsS http://localhost:8003/health | jq
curl -fsS http://localhost:8003/v1/voices | jq

curl -fsS -X POST http://localhost:8003/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Guten Morgen.","language":"de","response_format":"aac"}' \
  -o /tmp/t.aac
```

(Port `8003` matches `TTS_PORT`'s default in `docker-compose.yml`/`start.sh`
and this repo's port allocation: 8001 STT, 8002 image gen, 8003 TTS.)

## Model assets

On first start, if `MODEL_DIR` (default `/models/kokoro-de`) doesn't
already contain `kokoro-martin.onnx`, the server downloads `TTS_MODEL_ID`
from HuggingFace via `snapshot_download` (restricted to `*.onnx`/`*.npz`
files — the upstream repo also ships mp3 audio samples and a second,
unrelated Wyoming-bridge service's source that we don't need) and stores it
in the `model-cache` volume (compose) or `.cache/models/` next to
`start.sh` (plain docker). Subsequent restarts reuse the cached weights.

## Known limitations

- No German text normalization (numbers, dates, units, abbreviations) —
  the upstream author ships one (`german_text_rules.py`), not ported here.
  Spell out things like "2,5 kWh" or "14.05.2026" in the input text if you
  want them read naturally; the raw model will otherwise read digits/units
  literally.
- Sentence-boundary chunking for long inputs (`_split_sentences` in
  `server.py`) is a plain regex split on `. ! ?`, not aware of German
  abbreviations (`Prof.`, `Min.`, `ltr.`, ...) the way the upstream author's
  Wyoming-bridge patch is. A sentence ending in a dotted abbreviation may
  get split early, which the model tolerates but may add an unnatural
  pause.
- Single in-process worker (one `asyncio.Lock` around inference) — no
  process-pool parallelism like the upstream reference's `KOKORO_WORKERS`.
  Fine for the CPU/latency budget of a personal Signal bot; revisit if
  concurrent TTS requests become a bottleneck.

## Notes

- A single `asyncio.Lock` around `Kokoro.create()` serializes requests, same
  pattern as `deploy/dgx-spark/faster-whisper/server.py` and
  `deploy/dgx-spark/flux2-klein/server.py`.
- `/health` reports the currently loaded voices/provider so a bad
  `ORT_PROVIDER` fallback (see "CPU vs GPU" above) is visible without
  digging through logs.
