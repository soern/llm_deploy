# kyutai-tts-en_fr-dgx-spark

French (and English) text-to-speech HTTP service backed by
[kyutai/tts-1.6b-en_fr](https://huggingface.co/kyutai/tts-1.6b-en_fr)
(Kyutai's 1.6B "delayed streams" TTS, CC-BY-4.0), running on the DGX Spark
GPU next to the LLM and FLUX.2. It's the higher-quality French engine.
`deploy/cpu/kokoro-tts-french/` is the CPU fallback with the same endpoint
contract, so nixclaw's `tts.py` works against either.

## Voices

Voices are pre-computed speaker embeddings from
[kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices). Two French
aliases are built in, both Kyutai's own recordings, licensed **CC0**:

| alias | repo path | |
|---|---|---|
| `developpeuse` (default) | `unmute-prod-website/developpeuse-3.wav` | female |
| `fabien` | `unmute-prod-website/fabieng-enhanced-v2.wav` | male |

Any other voice can be requested by its repo path, e.g.
`"voice": "cml-tts/fr/1406_1028_000009-0003_enhanced.wav"` (`cml-tts/fr/`
holds 35 French LibriVox speakers, CC-BY-4.0, each also as `_enhanced`). It is
downloaded on first use and cached. Check licenses before picking one: the
`expresso/` voices are **non-commercial** only. Paths are validated
(`[A-Za-z0-9_.-]` segments ending in `.wav`/`.mp3`, no `..`), and an unknown
path returns 400.

## How it works

- moshi's PyTorch `TTSModel` (`moshi==0.2.13`), `n_q=32` codebooks
  (`KYUTAI_N_Q`, 8-16 is faster with lower quality), CFG 2.0, temp 0.6, bf16
  on CUDA. Output is 24 kHz mono, then ffmpeg encodes it.
- French input (`language` `fr*`) first goes through `french_text.py`
  (numbers, times, ordinals, currency, abbreviations; identical copy in
  `deploy/cpu/kokoro-tts-french/`, keep the two in sync). English input is passed
  through as-is.
- Long input is split at paragraph/sentence boundaries into chunks of
  ≤ `CHUNK_CHARS` (500). The chunks are generated **as one batch** (up to
  `MAX_BATCH`=8 per `generate` call) and joined with a 0.25 s pause.
- One request at a time (asyncio lock). Parallelism comes from batching
  within a request.
- `speed` (0.5-2.0) is applied with ffmpeg `atempo` (tempo change without
  pitch shift), since the model has no speed control of its own.

## Build notes (aarch64 / GB10)

- Base: `nvcr.io/nvidia/pytorch:25.11-py3`, the same as `flux2-klein-dgx-spark`
  and the source of an ARM64 + CUDA 13 torch for sm_121.
- `moshi` is installed with `--no-deps`: it pins `torch<2.10`, and pip would
  otherwise replace NVCR's 2.10 pre-release torch with a generic wheel. Its
  runtime deps are listed explicitly in the `Dockerfile`.
- `sphn` (moshi's audio I/O) has **no aarch64 wheel**, so it is built from
  source. That needs a Rust toolchain (rustup, minimal profile) plus
  `libopus-dev` for `audiopus_sys`. The first build takes a few minutes.

## Endpoints

- `GET  /health` → `{"status":"ok","model":"kyutai/tts-1.6b-en_fr","languages":["fr","en"],"voices":["developpeuse","fabien"],"device":"cuda","n_q":32,"sample_rate":24000}`
- `GET  /v1/voices` → aliases, their repo paths, defaults
- `POST /v1/audio/speech`, same body as the Kokoro deploys:
  ```json
  {"input": "Bonjour !", "voice": "developpeuse", "language": "fr",
   "response_format": "aac", "speed": 1.0}
  ```
  `language`: `fr`, `fr-fr`, `en`, `en-us`, `en-gb`. `response_format`: `aac`
  (default), `m4a`, `mp3`, `ogg`, `wav`. `input` ≤ `MAX_INPUT_CHARS` (20000).

## Build & run (on the Spark)

```bash
cd deploy/dgx-spark/kyutai-tts-en_fr
./start.sh     # docker build + run, host network, port 8009, waits for /health
./stop.sh
```

Env for `start.sh`: `KYUTAI_PORT` (8009), `DEFAULT_VOICE`, `KYUTAI_N_Q`,
`MAX_BATCH`, `FRENCH_NORMALIZE`, `HF_TOKEN`. Weights (~4 GB) are cached in
`./.cache/huggingface` (git-ignored). `docker compose up -d --build` works
too. It uses a named volume instead.

Memory: ~4 GB of weights in bf16, plus activations per batch item. The Spark's
memory is unified and shared with the LLM server, so if that one is started
with a high `gpu_memory_utilization`, start this container first or leave
headroom.

## CPU smoke test (any x86 box, no GPU)

```bash
docker build --build-arg BASE_IMAGE=python:3.12-slim --build-arg INSTALL_CPU_TORCH=1 \
  -t kyutai-tts-en_fr:cputest deploy/dgx-spark/kyutai-tts-en_fr
docker run --rm -p 8009:8009 kyutai-tts-en_fr:cputest
```

The CPU build forces the same sphn source build as aarch64. It runs in fp32
and is ~15× slower than realtime, so it's fine for checking the plumbing and
useless for real use.

## Test results

CPU smoke test only (2026-10-07, x86, 12 cores, fp32, `python:3.12-slim`
build with sphn compiled from source). **Not yet run on the Spark GPU.**

| test | result |
|---|---|
| model load (download cached) | ~65 s, ~9 GB RAM in fp32 |
| short sentence, `developpeuse`, wav | 3.7 s of audio in 55 s |
| 267-char two-paragraph text, `fabien`, aac | 15.9 s of audio in 149 s |
| forced 2-chunk batch (`CHUNK_CHARS=60`), `fabien`, ogg, `speed` 1.25 | 7.6 s → 6.1 s after atempo, 58 s |
| bad voice path (`../etc/passwd.wav`), unknown voice, `language: de` | 400 each |

Every output was transcribed back with faster-whisper `small` (fr), word for
word, including the normalized "Monsieur Dupont / 1er décembre / 15h30 /
12,50 € / 3,5 %" and both batched chunks in order (no truncation or junk
at chunk boundaries).

## Wiring into nixclaw

```yaml
tts:
  engines:
    fr:
      base_url: "http://gx10-811e.lan:8009"
      model: "kyutai/tts-1.6b-en_fr"
      voice: "developpeuse"
      timeout: 300.0
      max_input_chars: 6000
      max_concurrent_requests: 1
```

Set `timeout` from the measured GPU speed once deployed. A Signal voice
reply of a few hundred characters should take seconds, not minutes.
