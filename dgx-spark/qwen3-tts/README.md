# qwen3-tts-dgx-spark

One text-to-speech server for **German, English, French and Spanish** (and it, pt,
ru, ja, ko, zh), backed by [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts)
(Apache 2.0), running on the DGX Spark GPU. It's meant to replace the
per-language `kokoro-tts-*` sidecars. The endpoint contract is the same, so
nixclaw's `tts.py` needs no change: every `tts.engines.<lang>.base_url` points
here, and engines sharing a `base_url` share one concurrency cap.

## Voices

| voice | kind | language | source |
|---|---|---|---|
| `aiden` | preset (CustomVoice model) | en | Qwen's "sunny American male" speaker |
| `developpeuse` | cloned (Base model) | fr | Kyutai's CC0 recording `unmute-prod-website/developpeuse-3.wav`, checked in at `voices/developpeuse.safetensors` |
| `es-villarraza` | cloned (Base model) | es | LibriVox reader Victor Villarraza, Perrault's *Cuentos de Hadas* ch. 1 (public domain, `archive.org/details/cuentosdehadas_1701_librivox`, 25 s from 1:44), checked in at `voices/es-villarraza.safetensors` |
| `de-voiceclip` | cloned (Base model) | de | your own clip, made with `deploy/dgx-spark/qwen3-tts-voice-clone/`; **not** checked in |

- **Preset voices:** `aiden ryan vivian serena uncle_fu dylan eric ono_anna sohee`.
  Every preset can speak every language, but sounds best in its native one
  (aiden/ryan: English; the rest are Chinese, Japanese or Korean). That's
  why German and French use clones.
- **Cloned voices:** `voices/<name>.safetensors`, made by
  `deploy/dgx-spark/qwen3-tts-voice-clone/` (see its README). New or changed files are
  picked up on the next request, no restart needed. A clone only works with
  the Base model it was made with (0.6B and 1.7B have different speaker
  embedding sizes). The server checks this and returns 400 on a mismatch.
- **Defaults:** a request without `voice` gets the language's default from
  `DEFAULT_VOICES` (default `de=de-voiceclip,en=aiden,fr=developpeuse,es=es-villarraza`).

## How it works

- **Models:** two models, each loaded only if a voice needs it: the
  CustomVoice model for presets and the Base model for clones. By default
  both stay in memory (`MAX_LOADED_MODELS=2`), so switching between `aiden`
  and the cloned voices costs nothing. With `MAX_LOADED_MODELS=1` a request
  for a voice of the other model unloads the current one and loads the
  other, which adds a model load to that one request. Startup preloads the
  default voices' models. After every request the generation's working
  memory is given back (`torch.cuda.empty_cache()`). Defaults: **1.7B Base**
  for the clones (`QWEN_TTS_BASE_MODEL`) and **0.6B CustomVoice** for the
  presets (`QWEN_TTS_CUSTOM_MODEL`), so one container serves all languages
  with both models loaded. Until 2026-10-09, English ran in a second
  container (`qwen3-tts-small`, port 8012) with its own CUDA context and
  torch runtime; stop and remove it (`docker rm -f qwen3-tts-small`).
  bf16 on GPU, SDPA attention.
  flash-attn would be faster but has no GB10/aarch64 wheel; the
  "flash-attn is not installed" warning at startup is expected.
- **Long input:** split at paragraph/sentence boundaries into chunks of at
  most `CHUNK_CHARS` (400). The chunks are generated in batches of up to
  `MAX_BATCH`=2 and joined with a 0.2 s pause. Peak memory grows with the
  batch size: 8 was the suspected trigger of the 2026-10-09 overnight OOM
  thrash on the Spark, so long texts now take a bit longer instead. One request runs at a time.
- **Text normalization:** none. Qwen3-TTS's text frontend is LLM-based and
  read numbers, times and currency correctly in all three languages in the
  tests below.
- **Speed:** `speed` (0.5-2.0) goes through ffmpeg `atempo`, since the model
  has no speed control of its own.
- **Sampling (optional request fields):** `temperature` (default 0.9),
  `top_k` (50), `top_p` (1.0), `repetition_penalty` (1.05),
  `subtalker_temperature` (0.9) and `seed`. Every request (and every
  chunk) is sampled on its own, so with the defaults one sentence can come
  out excited and the next calm and slow. Lower values make the delivery
  steadier. nixclaw's voice calls send `voice_calls.tts_sampling`. `seed`
  makes a request reproducible, which helps A/B listening tests.
- **Language:** `language` is the ISO code nixclaw sends (`de`, `en`, `fr`,
  ...; `de-DE` style also works).

## Build notes

- **Base image:** `nvcr.io/nvidia/pytorch:25.11-py3`, the same as
  `flux2-klein-dgx-spark` and the source of an ARM64 + CUDA 13 torch for
  GB10.
- **qwen-tts is installed with `--no-deps`.** Its requirements would pull in
  gradio (demo UI only) and a torchaudio wheel that could replace NVCR's
  torch. The real deps are listed in the `Dockerfile`, using qwen-tts's own
  pins (`transformers==4.57.3`, `accelerate==1.12.0`).
- **torchaudio stub:** qwen_tts eagerly imports its 25 Hz tokenizer module,
  which imports `torchaudio.compliance.kaldi` (and `sox`/`onnxruntime`). The
  12 Hz models never use any of them. If the base image has no working
  torchaudio, the build installs `torchaudio_stub/`, which imports fine and
  raises on any actual use.

## Endpoints

- `GET  /health`: loaded models, languages, default voices, device
- `GET  /v1/voices`: presets, plus each cloned voice's language, model and
  reference transcript
- `POST /v1/audio/speech`, same body as the Kokoro deploys:
  ```json
  {"input": "Bonjour !", "language": "fr", "voice": "developpeuse",
   "response_format": "aac", "speed": 1.0}
  ```
  `response_format`: `aac` (default), `m4a`, `mp3`, `ogg`, `wav`.
  `input` ≤ `MAX_INPUT_CHARS` (20000).

## Build & run (on the Spark)

```bash
cd deploy/dgx-spark/qwen3-tts
./start.sh     # docker build + run, host network, port 8010, waits for /health
./stop.sh
```

A second instance next to the first one (e.g. to compare model sizes)
needs its own container name and port:

```bash
QWEN_TTS_CONTAINER=qwen3-tts-small QWEN_TTS_PORT=8012 \
QWEN_TTS_CUSTOM_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice DEFAULT_VOICES=en=aiden ./start.sh
QWEN_TTS_CONTAINER=qwen3-tts-small ./stop.sh
```

`start.sh` uses the NVIDIA base image and `--gpus all` when `nvidia-smi`
exists, and otherwise builds a CPU variant (that's how it was tested; see
below). Env: `QWEN_TTS_PORT` (8010), `QWEN_TTS_BASE_MODEL`,
`QWEN_TTS_CUSTOM_MODEL`, `DEFAULT_VOICES`, `MAX_BATCH`, `MAX_LOADED_MODELS`
(1 = swap on demand, 2 = keep both loaded), `QWEN_TTS_DTYPE`
(e.g. `bfloat16` to halve RAM on CPU), `VOICES_DIR` (default `./voices`),
`HF_HOME_HOST` (default `./.cache/huggingface`, git-ignored, also used by
the voice-clone deploy), `HF_TOKEN`, `OOM_SCORE_ADJ` (800: the OOM killer
takes this container first, then qwen-image at 500, sglang at -900 last), `RESTART_POLICY` (`on-failure:2`: at
most two restarts after a crash, none after a reboot or `docker stop`, so an
OOM-killed server doesn't keep reloading into a box that is short of memory).

**Memory** (measured on the Spark, 2026-10-08, both models 1.7B): both
loaded (`MAX_LOADED_MODELS=2`) ~15.1 GB. The current default pairing with the
0.6B CustomVoice model is not measured yet; check `free -m` before and after
starting. With `MAX_LOADED_MODELS=1` it used ~5 GB with the Base
model (German/French) and ~6.6 GB with the CustomVoice model (`aiden`),
peaking at ~7.8 GB during the swap. A swap costs ~68 s the first time
(model files not yet in the OS page cache) and ~30 s after that; requests
that stay with the loaded model don't pay it (a short reply takes ~1.5-2 s). The Spark's memory is shared with the LLM
server, so leave headroom.

## Test results

### DGX Spark GPU (2026-10-08, bf16, 1.7B, aac, warm, wall-clock incl. network)

| text | de (`de-voiceclip`) | en (`aiden`) | fr (`developpeuse`) |
|---|---|---|---|
| chat reply, ~90 chars | 5.7 s audio in 4.1 s | 5.4 s in 2.9 s | 5.1 s in 3.5 s |
| info with date/time/price, 140-190 chars | 11.3 s in 8.4 s | 10.9 s in 8.6 s | 10.2 s in 7.5 s |
| story, 580-650 chars (2 chunks, one batch) | 36.4 s in 14.4 s | 36.5 s in 16.3 s | 28.3 s in 11.5 s |

That is 0.7-0.8 s of compute per second of audio for one chunk, and ~0.4 s
when the text splits into a batch of chunks. Three concurrent requests
(de/en/fr info) are served one after another: 25 s in total. A 6000-char
input is ~15 chunks (two batches), so expect roughly 2-3 minutes; the
`timeout: 300.0` above fits.

A Whisper large-v3 round trip matched the text in all nine clips except
the digit string "4711". Whisper heard it as "477" (de), "4-2000-ZEMIN-11"
(en) and "4111" (fr). For the German clip it also wrote "4,99" without
"Euro" (not yet checked by ear). Digit strings that aren't amounts may need spelling
out (the agent can write "vier sieben eins eins").

### CPU (2026-10-07)

x86, 12 cores, fp32, built and started via `start.sh` in CPU mode. These
numbers only show that the plumbing works.

| test | voice | audio | wall-clock |
|---|---|---|---|
| en, 116 chars, aac | `aiden` (preset) | 9.0 s | 69 s |
| de, 118 chars, aac | `de-voiceclip` (clone) | 7.2 s | 59 s |
| fr, 205 chars, two paragraphs, aac | `developpeuse` (clone) | 12.5 s | 95 s |
| de, 540 chars → 2 chunks in one batch, ogg | `de-voiceclip` | 29.2 s | 133 s |
| unknown language / unknown voice / `../x` voice / empty input | | 400 each | |

Startup: both 1.7B models loaded in ~80 s (cached), 16.9 GB RAM in fp32.
Every output was transcribed back with faster-whisper `large-v3-turbo`,
**word for word** in all three languages, including "Dr. Smith / December
1st / 3:30 PM / $12.50", "1. Dezember um 15:30 Uhr / 12,50 Euro", "M. Dupont
/ 1er décembre / 15h30 / 12,50 € / 3,5 %", and both batched chunks in
order. No text normalizer was needed.

## Wiring into nixclaw

```yaml
tts:
  enabled: true
  default_language: "de"
  engines:
    de: {base_url: "http://gx10-811e.lan:8010", model: "Qwen3-TTS-1.7B", voice: "de-voiceclip", timeout: 300.0, max_input_chars: 6000}
    en: {base_url: "http://gx10-811e.lan:8010", model: "Qwen3-TTS-0.6B", voice: "aiden",        timeout: 300.0, max_input_chars: 6000}
    fr: {base_url: "http://gx10-811e.lan:8010", model: "Qwen3-TTS-1.7B", voice: "developpeuse", timeout: 300.0, max_input_chars: 6000}
    es: {base_url: "http://gx10-811e.lan:8010", model: "Qwen3-TTS-1.7B", voice: "es-villarraza", timeout: 300.0, max_input_chars: 6000}
```

`max_concurrent_requests` is per `base_url`, so the three engines share one
slot pool.

**Non-verbal sounds:** the server passes text to the model verbatim.
`aiden` acts out written-out sounds ("haha", "ahem", "hmm", "phew") best,
and handles `[laughter]` and `(laughs)` too. The cloned voices speak tags
literally ("lacht"), and how lively they sound depends on how expressive
the reference clip is. The `text_to_speech` tool description therefore
tells the agent to write sounds out as words.

**Digits in Spanish:** both Spanish clones (`es-villarraza`, `es-mongope`)
misread digits the same way, so it's the clone model, not the reference.
"el 3 de octubre" lost the "3", "14:00" became "catorce cero", and "4,99"
became "cuatro nueve". `aiden` reads the same text correctly. Written-out
numbers ("el tres de octubre", "cuatro euros con noventa y nueve") are
read correctly by the clones too, so the tool description tells the agent
to write numbers out. Long clone outputs sometimes end with a few seconds
of mumbling ("y… y…") after the text: the model runs past the end of the
text.
