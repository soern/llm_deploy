# canary-1b-v2-nvfp8

OpenAI-shaped Audio-Transkriptions-HTTP-Service für NVIDIA's
[Canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2)
(FastConformer-Encoder + Transformer-Decoder, 978M Parameter, 25 europäische
Sprachen), mit den Linear-Layern quantisiert auf **NVFP8** (FP8 e4m3 weights +
per-1×128-Blöcke FP32-Scalen), optimiert für **DGX Spark (GB10 / Blackwell)**.

Basiert auf `deploy/cpu/canary-1b-flash/` und spiegelt exakt die Request-/Response-
Konventionen von `deploy/dgx-spark/faster-whisper/server.py` wider, damit
`src/nixclaw_agent/stt.py` unverändert gegen dieses Backend laufen kann — nur
`STT_BASE_URL` umstellen.

## Sprach-Handling (kein Auto-Detect!)

Canary-1b-v2 hat **keine eingebaute LID** — der Decoder-Prompt muss explizit
die Sprache enthalten (`<|de|>` etc.). Dieses Server-Modell:

- kein / leeres `language`-Formfeld → **`de`** (Default, `CANARY_DEFAULT_LANG`)
- `language` außerhalb der 25 unterstützten Sprachen → **Fallback auf `en`**
  (`CANARY_FALLBACK_LANG`) + `"language_fallback": true` in der Antwort
- Antwort-JSON enthält zusätzlich `language` und `language_fallback`
  (ignorieren von `stt.py`, da nur `text` gelesen wird)

Unterstützte Sprachen: `bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it,
lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk`.

## Endpunkte

- `GET  /health`                  — Liveness + Modell/Device/Quant-Status
- `GET  /v1/models`               — OpenAI-kompatible Modellanfrage (1 Eintrag)
- `POST /v1/audio/transcriptions` — Multipart-Upload, liefert `{"text": ...}`
  - `file` — Audio-Datei (alles, was ffmpeg dekodieren kann; intern → 16 kHz
    mono WAV)
  - `language` — optional, Default `de`, Fallback `en`
  - `model` — rein informativ (Log)
  - `response_format` — muss `"json"` sein
  - `temperature` — akzeptiert, aktuell ungenutzt

## Build & run (DGX Spark, GPU)

```bash
cd deploy/dgx-spark/canary-1b-v2-nvfp8
docker compose up -d --build
# erster Start: ~4 GB Download von HF + Quantisierung (~1-2 min)
docker logs -f canary-1b-v2-nvfp8
```

Port-Mapping ist standardmäßig **Host-Port 8006 → Container 8006**.

### Verifikation

```bash
curl -s localhost:8006/health
curl -s localhost:8006/v1/models

# deutscher Sprachtest (Default):
curl -s -F file=@tests/fixtures/voice_test.aac \
    localhost:8006/v1/audio/transcriptions | jq

# explizite Sprache:
curl -s -F file=@tests/fixtures/voice_test.aac -F language=de \
    localhost:8006/v1/audio/transcriptions | jq

# Fallback-Test (ungenützte Sprache → en + Flag):
curl -s -F file=@tests/fixtures/voice_test.aac -F language=xx \
    localhost:8006/v1/audio/transcriptions | jq
```

## VRAM-Budget (Batch 1, chunked Audio)

| Format | Weights in VRAM | Peak total |
|---|---|---|
| FP32 (Original) | ~3.9 GB | ~6–7 GB |
| BF16 | ~1.9 GB | ~3.5–4.5 GB |
| **NVFP8 (dieses Deploy)** | **~1.0 GB** | **~1.5–2.5 GB** |

Chunking-Verhalten: 20-Min.-Audios passen NICHT in einen Durchlauf —
Encoder-Cap liegt bei ~5000 Positionen ≈ 400 s. Das Client-Chunking in
`stt.py` (`stt.chunk_seconds`, Default 28 s) übernimmt diese Arbeit bereits;
dieser Server transkribiert also pro Request ein kurzes Chunk, kein
Langzeit-Video.

## Quantisierung

`quantize.py` quantisiert alle `nn.Linear`-Layern des Modells (Encoder,
Decoder, Embedding-Nähe) auf FP8 e4m3 mit per-1×128-Block-Scalen. Conv-Layer,
Embeddings und LayerNorm bleiben in BF16. Ergebnis wird unter
`/models/canary-1b-v2-nvfp8/` (Volume `model-cache`) geparkt und beim
nächsten Start wiederverwendet — nur der allererste Start quantisiert.

Bei Inferenz werden die FP8-Weights pro Layer vor Ort (FP32-Intermediat)
dequantisiert und das GEMM in BF16 durchgezogen — deterministisch und
versionsunabhängig. Ein echter FP8-GEMM via `torch._scaled_mm` kann später
als Schnellpfad ergänzen (Blackwell/GB10 hat nativ FP8-Hardware), ist aber
hier absichtlich nicht drin, um die Versionen-Robustheit nicht zu riskieren.

`CANARY_QUANT=0` deaktiviert die Quantisierung komplett (BF16-Reinlauf).

## Anbindung an NixClaw

In `config/agent.yaml`:

```yaml
stt:
  base_url: "http://<host>:8006"
  model: "canary-1b-v2-nvfp8"   # rein informativ
  language: "de"
```

oder als `alternate` (nur via `retranscribe_audio`-Tool):

```yaml
stt:
  alternate:
    enabled: true
    base_url: "http://<host>:8006"
    language: null   # nutzt dann Default de + Fallback en des Servers
```

## Umgebungsvariablen (docker-compose.yml / Dockerfile)

| Variable | Default | Bedeutung |
|---|---|---|
| `CANARY_MODEL_ID` | `nvidia/canary-1b-v2` | HF-Modell-Repo |
| `CANARY_DEFAULT_LANG` | `de` | Default-Sprache ohne explizite Angabe |
| `CANARY_FALLBACK_LANG` | `en` | Fallback bei ungestützter Sprache |
| `CANARY_QUANT` | `1` | `0` deaktiviert NVFP8 (BF16-Reinlauf) |
| `DEVICE` | Auto (CUDA→CPU) | Device-Override |
| `HF_TOKEN` | — | optional, falls HF-Auth nötig |
| `QUANT_CACHE_DIR` | `/models/canary-1b-v2-nvfp8` | Ort des quantisierten Caches |
| `PORT` | `8006` | Container-Port |
