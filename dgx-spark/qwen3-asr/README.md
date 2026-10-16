# qwen3-asr-dgx-spark

Speech-to-text server backed by [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
(Apache 2.0), running on the DGX Spark GPU. It covers 30 languages plus 22
Chinese dialects and identifies the language on its own. It's a drop-in
replacement for `deploy/dgx-spark/faster-whisper/`: it serves the same endpoints
on the **same port (8001)**, so nixclaw's `stt.py` and `stt.base_url` need no
changes.

## Run

```bash
# faster-whisper and this server both use port 8001. Stop faster-whisper first:
docker compose -f deploy/dgx-spark/faster-whisper/docker-compose.yml down

deploy/dgx-spark/qwen3-asr/start.sh     # build + run + wait for /health
deploy/dgx-spark/qwen3-asr/stop.sh
```

`start.sh` follows `deploy/dgx-spark/qwen3-tts/start.sh`. It runs a named
container (`qwen3-asr`) with host networking and keeps the HF cache in
`./.cache/huggingface`. The log goes to `./.qwen3-asr.log`. The script won't
start if port 8001 is already taken. The first start downloads about 1.9 GB.

| env (start.sh) | default | meaning |
|---|---|---|
| `QWEN_ASR_PORT` | `8001` | listen port (host network) |
| `QWEN_ASR_CONTAINER` | `qwen3-asr` | container name |
| `QWEN_ASR_MODEL` | `Qwen/Qwen3-ASR-0.6B` | HF repo (`Qwen/Qwen3-ASR-1.7B` works too) |
| `QWEN_ASR_DTYPE` | `bf16` on GPU, `float32` on CPU | `bf16` / `float16` / `float32` |
| `QWEN_ASR_MAX_NEW_TOKENS` | `512` | output cap per request (nixclaw sends ≤ 30 s chunks) |
| `QWEN_ASR_NO_SPEECH_THRESHOLD` | `0.5` | default `no_speech_threshold` (see speech detection) |
| `QWEN_ASR_STREAM_MAX_SECONDS` | `60` | audio per streamed utterance; more is ignored |
| `QWEN_ASR_STREAM_PREFIX_AFTER_SECONDS` | `10` | partials of longer utterances decode with a text prefix |
| `OOM_SCORE_ADJ` | `300` | killed after TTS (+800) and qwen-image (+500), before sglang (-900) |
| `RESTART_POLICY` | `on-failure:2` | as for qwen3-tts |
| `HF_TOKEN` | — | optional |

## Endpoints

- `GET  /health`: model, device, dtype and supported languages
- `GET  /v1/models`: OpenAI-style model list with one entry
- `POST /v1/audio/transcriptions`: multipart upload, returns
  `{"text", "language", "language_fallback"}`
  - `file`: any format ffmpeg can decode. The server converts it to 16 kHz mono.
  - `language`: an ISO code (`de`, `de-DE`) or a name (`German`). Leave it
    out for auto-detect. An unsupported language falls back to auto-detect
    and sets `language_fallback: true`.
  - `prompt`: optional context passed to the model, such as names or jargon
    whose spelling it should follow
  - `model`, `response_format` (`json` only), `temperature`: accepted for
    compatibility, otherwise ignored (decoding is greedy)
  - `detect_speech` (`true`/`false`, default false) and `no_speech_threshold`:
    see speech detection below. Adds `no_speech` and `no_speech_prob` to the
    response.
- `WS /v1/audio/transcriptions/stream`: streaming transcription for voice
  calls (see below)

## Speech detection

With a forced `language` the model always outputs words, even for silence or
background noise: silence comes back as "Ich.", pink noise as "Die.". Without
one it answers "language None" for non-speech, but short words get misheard
("Nein" becomes "Nine", "Ja" becomes "Yeah"). `detect_speech=true` combines
the two. The server first asks the model only for the language and reports
the probability of "None" as `no_speech_prob`. At or above
`no_speech_threshold`, `text` is empty and `no_speech` is true. Otherwise the
normal transcription runs, with the forced language.

The question (the unforced prompt plus "language") is a token prefix of the
transcription prompt ("language German<asr_text>"). The transformers backend
therefore runs the audio and that prefix once (one forward pass, no
generation), reads P("None") from the last position, and continues the
transcription from the same KV cache. The audio is encoded once per decode
instead of twice. Measured on the CPU build, the results are identical to the
earlier two-pass version on all 24 samples (text and `no_speech_prob`, forced
and auto). Noise takes ~0.4 s instead of ~1.5 s, short words ~1 s instead of
~1.8 s. The vLLM backend sends two requests and its prefix cache reuses the
shared part.

Measured on synthetic samples (fp32 CPU build, same weights):

| sample | `no_speech_prob` |
|---|---|
| silence, pink/white noise, 100 Hz hum, click clatter, noise burst | 0.993 – 0.9998 |
| "Ja.", "Nein.", "Moment.", "Warte mal kurz." (also mixed with noise) | < 0.003 |

## Streaming (`/v1/audio/transcriptions/stream`)

A WebSocket that handles any number of utterances in sequence (nixclaw's voice
calls keep one open per call):

```
client: {"type": "start", "language": "de", "prompt": "", "chunk_ms": 500,
         "detect_speech": true, "no_speech_threshold": 0.5, "partials": true}
client: binary frames, 16 kHz mono s16le PCM, while the caller speaks
server: {"type": "partial", "text", "no_speech", "no_speech_prob", "audio_ms"}
        after each further chunk_ms of audio (steps merge under load), and
        after each flush
client: {"type": "partials", "enabled": false}   partial steps off/on
client: {"type": "flush"}    the caller went quiet: decode everything now
client: {"type": "finish", "tail_silence_ms": 660}
server: {"type": "final", "text", "language", "language_fallback",
         "no_speech", "no_speech_prob", "audio_ms", "finish_ms"}
client: {"type": "cancel"}   drops the utterance, no reply
server: {"type": "error", "detail"}   protocol error, connection stays open
```

How it works:
- The transformers backend has no incremental decoder, so each step decodes
  all the audio so far again. That's cheap for utterance-length audio.
- qwen-asr's own `streaming_transcribe()` (vLLM backend only) adds the
  previous text minus a few tokens as a prefix after two steps. That locks in
  early mishearings: "Willkommen…" became "Der Weg von Foxjump…".
- Here a prefix is used only for partials of utterances longer than
  `QWEN_ASR_STREAM_PREFIX_AFTER_SECONDS`. The final decode never uses one, so
  the final text matches the POST endpoint's.
- `tail_silence_ms` tells the server how much of the end of the audio is
  known silence (nixclaw sends its end-of-speech window). If the last step
  already covered everything before it, that step's result becomes the final
  one with no new decode.
- `flush`: nixclaw sends it after 200 ms of silence
  (`voice_calls.stt_flush_ms`). The server decodes all audio so far right
  away, after a step still running, without a prefix. It does this while
  nixclaw is still waiting for the 700 ms end of speech, so the finish
  usually reuses that result. If the caller goes on speaking, the next pause
  flushes again.
- `partials: false`: decode only on flush and finish. Partials don't make the
  final any faster, because each step decodes everything again. nixclaw only
  needs them while its own reply is playing (barge-in check) and switches them
  on and off. That keeps the GPU it shares with the LLM and TTS free. CPU
  test: 1 decode per utterance instead of 2–6.
- Each step also runs the speech check when `detect_speech` is set. Partials
  for noise therefore have empty text and don't trigger nixclaw's barge-in.
- All requests share one model lock with the POST endpoint.

Test client: `nixclaw` `src/nixclaw_agent/voice/stt_stream.py`
(`StreamingTranscriber`). To turn it on for calls, set
`voice_calls.stt_streaming: true` in nixclaw's `agent.yaml`.

## vLLM backend (optional)

```bash
deploy/dgx-spark/qwen3-asr/stop.sh
QWEN_ASR_BACKEND=vllm QWEN_ASR_MODEL=Qwen/Qwen3-ASR-1.7B deploy/dgx-spark/qwen3-asr/start.sh
```
- Builds `Dockerfile.vllm`. The base is `vllm/vllm-openai:v0.14.0`, the vLLM
  version qwen-asr 0.0.6 pins for its model plugin. Same `server.py`, same
  API. The model runs in vLLM in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`),
  with CUDA graphs and prefix caching.
- GPU memory: `QWEN_ASR_VLLM_GPU_MEMORY_UTILIZATION` (default 0.07 of the
  Spark's memory, ~8 GB: weights, KV cache, CUDA graphs). vLLM refuses to
  start if that much isn't free next to the LLM, FLUX.2 and TTS.
  `QWEN_ASR_VLLM_MAX_MODEL_LEN` (4096) covers 60 s of audio plus 512 output
  tokens. `QWEN_ASR_VLLM_ENFORCE_EAGER=1` skips CUDA graphs: less memory and
  a faster start, slower decodes.
- The speech check is a one-token request with the top-20 logprobs. If
  "None" isn't among them, `no_speech_prob` is 0.
- **Not tested yet**: the build host has no GPU. If the plugin doesn't load
  with this base image, try the vLLM image already working on the Spark
  (`--build-arg BASE_IMAGE=...` in start.sh's `docker build`). Going back:
  `stop.sh`, then `start.sh` without `QWEN_ASR_BACKEND`.

## Deploying a new version

nixclaw works with this and with older versions (flush/partials: an older
server answers them with an error, after which nixclaw stops sending them).
For the 2026-10-12 update (one-pass speech check, flush, partials,
`last_decode` in the `stream final` log line), restart with the model you use:

```bash
deploy/dgx-spark/qwen3-asr/stop.sh
QWEN_ASR_MODEL=Qwen/Qwen3-ASR-1.7B deploy/dgx-spark/qwen3-asr/start.sh
```

## Deploying the streaming/detect_speech version

The API is backward compatible: the POST endpoint behaves as before unless
`detect_speech` is sent. nixclaw works with either server version:
- `detect_speech`: nixclaw detects support from the response and falls back to
  two requests on an older server.
- Streaming: nixclaw uses HTTP if the WebSocket connect fails, and retries
  every 30 s.

```bash
deploy/dgx-spark/qwen3-asr/stop.sh
deploy/dgx-spark/qwen3-asr/start.sh     # rebuilds the image (server.py is baked in)
curl -s -F file=@tests/fixtures/voice_test.aac -F language=de -F detect_speech=true \
    localhost:8001/v1/audio/transcriptions | jq
```
Then set `voice_calls.stt_streaming: true` and restart nixclaw.

## Verify

```bash
curl -s localhost:8001/health | jq
curl -s -F file=@tests/fixtures/voice_test.aac -F language=de \
    localhost:8001/v1/audio/transcriptions | jq
curl -s -F file=@tests/fixtures/voice_test.aac \
    localhost:8001/v1/audio/transcriptions | jq    # auto-detect
```

## nixclaw config

Nothing changes if `stt.base_url` already points at `<spark>:8001`. The
`stt.model` value is only for information, so you can set it to
`"qwen3-asr-0.6b"`. `stt.language: "de"` forces German. Set it to `null` to
use the model's own language detection.

## Build notes

- **Base image:** `nvcr.io/nvidia/pytorch:25.11-py3`, the same as
  qwen3-tts and flux2-klein. It's where the ARM64 + CUDA 13 torch for GB10
  comes from. Without `nvidia-smi`, `start.sh` builds a CPU variant on
  `python:3.12-slim` instead.
- **qwen-asr is installed with `--no-deps`.** Its requirements would pull in
  gradio and flask (demo UIs), nagisa and soynlp (forced aligner) and
  sox/qwen-omni-utils (not used at inference). They could also replace
  NVCR's torch. The `Dockerfile` lists the real dependencies with qwen-asr's
  own pins: `transformers==4.57.6` and `accelerate==1.12.0`.
- **nagisa stub:** `qwen_asr/__init__` eagerly imports the forced aligner,
  and the forced aligner imports nagisa, which has no aarch64 wheel. If
  nagisa won't import, the build installs `nagisa_stub/`. The stub imports
  cleanly and raises on any use. This means there are **no timestamps**:
  the forced aligner isn't supported in this image.
- Uses the transformers backend with SDPA attention. flash-attn has no
  GB10/aarch64 wheel, and the vLLM backend would claim a fixed share of GPU
  memory on a box it shares with other services. That is also why the
  streaming endpoint is implemented here rather than with qwen-asr's
  vLLM-only `streaming_transcribe()`. Requests are serialized with a lock. After each request `torch.cuda.empty_cache()` gives the
  working memory back.
