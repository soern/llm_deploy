# llm_deploy

Deploy scripts for nixclaw's model backends, organized by target hardware.
One folder per deploy, each with a `README.md` and launch files
(`start.sh`/`stop.sh` and/or `docker-compose.yml`; HTTP services add a
`Dockerfile` and `server.py`).

## Layout

```
dgx-spark/          NVIDIA DGX Spark (GB10, ARM64)
jetson-orin-nano/   NVIDIA Jetson Orin Nano (8 GB, JetPack 7)
cpu/                CPU-only
```

Folder name = `<framework>-<model>[-variant]`; the target device is the
parent folder, so it is not repeated in the name.

## Deploys

- **dgx-spark**
  - LLM: `vllm-qwen3.6`, `vllm-qwen3.8-native`, `vllm-qwen3.8-nvfp4`, `vllm-qwen3.8-flash-fp4`, `vllm-gemma4-e2b`, `vllm-qwen3.5-4b`, `sglang-qwen3.8`, `sglang-minicpm5-2b`, `lfm2.5`
  - Speech: `faster-whisper`, `canary-1b-v2`, `canary-1b-v2-nvfp8`, `qwen3-asr` (STT); `qwen3-tts`, `qwen3-tts-voice-clone`, `kyutai-tts-en_fr` (TTS)
  - Image: `flux2-klein`, `flux2-klein-uncensored`, `qwen-image-2.1`
- **jetson-orin-nano** (llama.cpp): `gemma4-e2b`, `lfm2.5-8b-a1b`, `qwen3-4b-instruct-2507`, `qwen3.5-4b`
- **cpu**: `qwen3-4b-instruct-2507` (Ollama); `kokoro-tts-german`, `kokoro-tts-english`, `kokoro-tts-french` (TTS); `canary-1b-flash` (STT)

## Running

```sh
./start.sh                     # build/start the container, wait for health
./stop.sh

# or, where a docker-compose.yml is provided:
docker compose up -d --build
```

Set `HF_TOKEN` in the environment for gated-model downloads (read from the
environment only, never hardcoded).
