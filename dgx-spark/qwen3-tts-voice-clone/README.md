# qwen3-tts-voice-clone

Turns a reference clip (any format ffmpeg reads, e.g. a phone `.m4a`) into a
cloned voice for `deploy/dgx-spark/qwen3-tts/`. It's a separate service, so
the TTS server stays a plain player and cloning can run wherever is
convenient: on the Spark (GPU), or locally on CPU.

## What "the trained voice" is

Qwen3-TTS cloning is **zero-shot**: no training run. The Base model encodes
the clip into speech codes plus a speaker embedding, conditioned on the
clip's transcript (ICL mode). That `VoiceClonePromptItem` is the whole voice.
It's stored as `<name>.safetensors` (about 25 KB; tensors plus string
metadata, no pickle; format in `qwen_voice_file.py`, kept identical to the
TTS deploy's copy). The service writes, per voice:

| file | what |
|---|---|
| `<name>.safetensors` | the voice, what the TTS server loads |
| `<name>.ref.wav` | the exact (trimmed) audio the voice was built from |
| `<name>.preview.wav` | one sentence in the cloned voice, to judge it |

A voice only works with the Base model it was cloned with (1.7B by default;
0.6B embeddings have a different size). Clone with the same
`QWEN_TTS_BASE_MODEL` the TTS server runs.

## Pipeline (`POST /v1/voices`)

1. Decode the upload with ffmpeg to 24 kHz mono (≤ 25 MB, ≤ 300 s).
2. **Transcript:** without `ref_text`, faster-whisper (`large-v3-turbo`,
   CPU int8) transcribes the clip with word timestamps. The reference is then
   **trimmed** to ≤ `max_ref_seconds` (default 12) at the last word followed
   by punctuation, i.e. at a natural pause. Every TTS request re-processes
   the whole reference, so a longer one makes every reply slower without
   sounding better (Qwen says 3 s is enough; 8-12 s works well). With
   `ref_text` the clip is used untrimmed (≤ 30 s), and the text must match
   it exactly.
3. `create_voice_clone_prompt` → `<name>.safetensors` + `<name>.ref.wav`.
   The file is renamed into place last, so a TTS server sharing the directory
   never reads a half-written file.
4. Optional preview sentence → `<name>.preview.wav`. This is the only slow
   step (~45 s on CPU, seconds on GPU).

Clip tips: one speaker, no music, no room echo, natural speaking pace. The
voice copies the clip's mic and room character too.

## Companion scripts (run from anywhere, need only curl + python3)

```bash
# clone + download results (default dest ./voices; user@host:path is scp'd)
./clone_voice.sh -l de ~/clips/anna.m4a anna
./clone_voice.sh -l fr -t "exact transcript" -n clip.wav marie   # own transcript, no preview
./clone_voice.sh -f -l de -s 8 clip.m4a anna                     # overwrite, 8 s reference

./list_voices.sh                                      # what the service has made
./fetch_voice.sh -o ../qwen3-tts/voices anna            # install locally
./fetch_voice.sh -o spark:nixclaw2/deploy/dgx-spark/qwen3-tts/voices anna   # or scp
```

Point them at a remote service with `-u http://host:8011` or
`QWEN_CLONE_URL`. On the Spark itself nothing needs installing: `start.sh`
points the service at the TTS server's `voices/` dir, so a new voice is live
immediately. Delete one with `curl -X DELETE http://host:8011/v1/voices/<name>`.

## Endpoints

- `POST /v1/voices`: multipart `file`, `name` (`[a-z0-9][a-z0-9_-]*`),
  `language` (de/en/fr/es/it/pt/ru/ja/ko/zh), optional `ref_text`,
  `max_ref_seconds` (3-30), `preview` (true), `preview_text`, `overwrite`
  (false → 409 if the name exists). Returns JSON with transcript, reference
  and source lengths, the full transcript if it was trimmed, and timings.
- `GET /v1/voices`: all voices with metadata
- `GET /v1/voices/{name}/file | /preview | /reference`
- `DELETE /v1/voices/{name}`
- `GET /health`

## Build & run

```bash
cd deploy/dgx-spark/qwen3-tts-voice-clone
./start.sh   # port 8011; GPU + NVIDIA base image if nvidia-smi exists, else CPU variant
./stop.sh
```

The Dockerfile's layers up to faster-whisper are identical to the TTS deploy's, so the
images share them. The HF cache defaults to the TTS deploy's
(`../qwen3-tts/.cache/huggingface`), so the Base model is
downloaded once. Env: `QWEN_CLONE_PORT` (8011), `QWEN_TTS_BASE_MODEL`,
`WHISPER_MODEL`, `QWEN_TTS_DTYPE`, `VOICES_DIR`, `HF_HOME_HOST`, `HF_TOKEN`.
RAM: ~9 GB on CPU (1.7B Base in fp32 plus whisper).

## Test results (2026-10-07, x86 CPU, 12 cores, fp32)

Run via `start.sh` (CPU mode) and the companion scripts.

| voice | source | reference after trim | transcribe / clone / preview |
|---|---|---|---|
| `de-voiceclip` | 25.3 s German weather report (`.m4a`, phone) | 10.8 s, ends at "…zwischen 9 und 13 Grad." | 5.8 s / 1.2 s / 44.7 s |
| `developpeuse` | 13.4 s Kyutai CC0 `.wav`, French | 11.7 s, ends at "…les fables de La Fontaine," | 4.8 s / 0.5 s / 40.5 s |

2026-10-08, same setup: `es-villarraza` from 25 s of LibriVox's *Cuentos de
Hadas* (Victor Villarraza). Whisper wrote this archaic text without
punctuation, so the automatic trim fell back to a mid-phrase cut. The voice
was re-cloned from a 10.9 s cut at a sentence end, with `-t "<punctuated
transcript>"`. Preview 68 s on CPU, transcribed back word for word.

The first attempt used whisper `small` with segment-level trimming. It
gave a 5.4 s reference and transcript errors ("Horniburg", "Südtosten"), which
motivated word-level trimming at punctuation and `large-v3-turbo` as the
default. Both previews transcribe back word for word. Error paths tested:
bad name, unsupported language, `max_ref_seconds` out of range, a junk
upload (ffmpeg decode error), existing name without overwrite (409), and
unknown voice / path traversal on download (404).
