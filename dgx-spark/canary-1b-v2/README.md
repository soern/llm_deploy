# canary-1b-v2 (model default precision)

OpenAI-shaped Audio-Transkriptions-HTTP-Service für NVIDIA's
[Canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2)
(FastConformer-Encoder + Transformer-Decoder, 978M Parameter, 25 europäische
Sprachen) — **ohne Quantisierung**, das Checkpoint wird exakt so geladen,
wie es auf HF veröffentlicht ist (FP32, ~3.9 GB).

Komplementär zu `deploy/dgx-spark/canary-1b-v2-nvfp8/` (gleicher Endpoint-Vertrag,
dort FP8-Quantisierung). Beide spiegeln die Konventionen von
`deploy/dgx-spark/faster-whisper/server.py` wider, damit
`src/nixclaw_agent/stt.py` unverändert dagegen laufen kann.

## Port

Host-Port **8007** (das NVFP8-Deploy läuft auf 8006).

## Präzision / VRAM

| `CANARY_DTYPE` | Weights | Peak-VRAM (Batch 1, chunked) |
|---|---|---|
| `float32` (Default, Modell-Default) | ~3.9 GB | ~5–6 GB |
| `bf16` | ~1.9 GB | ~3.5–4.5 GB |
| `float16` | ~1.9 GB | ~3.5–4.5 GB |

## Sprach-Handling (kein Auto-Detect!)

- kein / leeres `language`-Formfeld → **`de`** (`CANARY_DEFAULT_LANG`)
- `language` außerhalb der 25 unterstützten Sprachen → **Fallback `en`**
  (`CANARY_FALLBACK_LANG`) + `"language_fallback": true` in der Antwort
- zusätzlich in der Antwort: `language` (tatsächlich verwandt)

Unterstützte Sprachen: `bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it,
lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk`.

## Endpunkte

- `GET  /health`                  — Liveness + Modell/Device/Dtype
- `GET  /v1/models`               — OpenAI-kompatible Modellliste (1 Eintrag)
- `POST /v1/audio/transcriptions` — Multipart-Upload, liefert `{"text": ...}`
  - `file` — Audio-Datei (alles, was ffmpeg dekodiert; intern → 16 kHz mono)
  - `language` — optional, Default `de`, Fallback `en`
  - `model`, `response_format` (`json`), `temperature` — Verträglichkeit

## Build & run (DGX Spark, GPU)

```bash
cd deploy/dgx-spark/canary-1b-v2
docker compose up -d --build
# erster Start: ~4 GB Download von HF
docker logs -f canary-1b-v2
```

### Verifikation

```bash
curl -s localhost:8007/health
curl -s localhost:8007/v1/models

curl -s -F file=@tests/fixtures/voice_test.aac \
    localhost:8007/v1/audio/transcriptions | jq

# Fallback-Test (ungestützte Sprache → en + Flag):
curl -s -F file=@tests/fixtures/voice_test.aac -F language=xx \
    localhost:8007/v1/audio/transcriptions | jq
```

## Anbindung an NixClaw

```yaml
stt:
  base_url: "http://<host>:8007"
  model: "canary-1b-v2"
  language: "de"
```

## Umgebungsvariablen

| Variable | Default | Bedeutung |
|---|---|---|
| `CANARY_MODEL_ID` | `nvidia/canary-1b-v2` | HF-Modell-Repo |
| `CANARY_DEFAULT_LANG` | `de` | Default-Sprache ohne explizite Angabe |
| `CANARY_FALLBACK_LANG` | `en` | Fallback bei ungestützter Sprache |
| `CANARY_DTYPE` | `float32` | `float32` / `bf16` / `float16` |
| `DEVICE` | Auto (CUDA→CPU) | Device-Override |
| `HF_TOKEN` | — | optional |
| `PORT` | `8007` | Container-Port |

## Chunking-Hinweis

Der Encoder ist auf ~5000 Positionen ≈ 400 s Audio pro Durchlauf begrenzt.
Lang-Audio wird wie im NVFP8-Deploy vom Client chunked
(`stt.chunk_seconds` in `config/agent.yaml`), der Server sieht nur kurze Chunks.
