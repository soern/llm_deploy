# kokoro-tts-french

French text-to-speech HTTP service backed by
[hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) with its
only French voice, **`ff_siwis`** (female; hexgrad grades it B-, trained on
the SIWIS corpus). CPU-only, like `deploy/cpu/kokoro-tts-german/` and
`deploy/cpu/kokoro-tts-english/`. Used by nixclaw's `text_to_speech` tool as the
`fr` engine.

This is the cheap, robust French option. For noticeably more natural
French (and a male voice), see `deploy/dgx-spark/kyutai-tts-en_fr/`, which
needs the Spark's GPU. Same endpoint contract, so switching is a one-line
`base_url` change in `agent.yaml`.

## How it works

- `KPipeline(lang_code="f")` → misaki `EspeakG2P("fr-fr")` (espeak-ng) →
  Kokoro-82M → 24 kHz mono → ffmpeg encode.
- Input first goes through `french_text.py` (`FRENCH_NORMALIZE=1`, default),
  which spells out numbers, clock times, ordinals, currency, percentages and
  a few abbreviations: "M. Dupont, le 1er décembre à 15h30, 12,50 €" →
  "Monsieur Dupont, le premier décembre à quinze heures trente, douze euros
  cinquante". It's deliberately small; anything it doesn't know passes
  through to espeak-ng unchanged. An identical copy lives in the Kyutai
  deploy, so keep the two in sync.
- Long input: KPipeline 0.9 splits on newlines and chunks long lines at
  sentence boundaries itself.

## Version pins (read before "aligning" with the English deploy)

`kokoro==0.9.4`, `misaki[en]==0.9.4`, `torch==2.8.0` (CPU),
`transformers==5.19.0`. The English deploy's pins (`kokoro==0.7.16`,
`torch==2.5.1`) **break French**: kokoro 0.7.16 + misaki 0.9 yields ~1 s of
junk audio for any non-English input (misaki now returns a tuple), and
current transformers needs torch ≥ 2.6. Details are in the `Dockerfile`
comments.

## Endpoints

- `GET  /health` → `{"status":"ok","model":"hexgrad/Kokoro-82M","languages":["fr"],"voices":["ff_siwis"],"device":"cpu","sample_rate":24000}`
- `GET  /v1/voices`
- `POST /v1/audio/speech`, same body as the other Kokoro deploys:
  ```json
  {"input": "Bonjour !", "voice": "ff_siwis", "language": "fr",
   "response_format": "aac", "speed": 1.0}
  ```
  `language` must be `fr`/`fr-fr`, `voice` must be `ff_siwis`, `speed`
  0.5–2.0, `response_format` one of `aac` (default, raw ADTS like real Signal
  voice notes), `m4a`, `mp3`, `ogg`, `wav`. `input` capped at
  `MAX_INPUT_CHARS` (20000).

## Build & run

```bash
cd deploy/cpu/kokoro-tts-french
./start.sh     # builds, starts nixclaw-kokoro-tts-fr on host port 8008, waits for /health
./stop.sh
```

Env: `TTS_PORT` (host port, default 8008), `FRENCH_NORMALIZE` (0 to
disable), `HF_TOKEN` (optional). Model + voice (~330 MB) are cached in the
`model-cache` volume.

## Verify

```bash
curl -fsS http://127.0.0.1:8008/health
curl -sS -o /tmp/fr.aac -H 'Content-Type: application/json' \
  -d '{"input":"Bonjour ! Votre rendez-vous est confirmé pour 15h30.","language":"fr"}' \
  http://127.0.0.1:8008/v1/audio/speech && ffprobe -v error -show_entries format=duration /tmp/fr.aac
```

## Test results (2026-10-07, x86 CPU, 12 cores)

Test text: two paragraphs, 267 chars, with a title, an ordinal, a time,
currency and a percentage. Result: **15.2 s of audio in 3.2–3.5 s**
(~4.5× realtime). A faster-whisper `small` (fr) round trip of the output
transcribed every word correctly, including "Monsieur Dupont", "1er
décembre", "15h30", "12,50 €" and "3,5 %". Error paths (wrong language or
voice, empty input) return 400.

## Wiring into nixclaw

```yaml
tts:
  engines:
    fr:
      base_url: "http://<host>:8008"
      model: "hexgrad/Kokoro-82M"
      voice: "ff_siwis"
```

See `config/agent.yaml.example`.
