# Ollama deployment for Qwen/Qwen3-4B-Instruct-2507 (CPU-only)

`start.sh` runs `Qwen/Qwen3-4B-Instruct-2507` under the official upstream
`ollama/ollama` image on a plain CPU host (no GPU/CUDA required), same
plain-`docker run` pattern as the other `deploy/` scripts in this repo.

## Model

- **HF checkpoint**: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
  — dense `Qwen3ForCausalLM`, instruction-tuned, non-thinking (the `-2507`
  suffix marks Qwen's July 2025 refresh that dropped the `<think>` blocks
  used by the plain `qwen3:4b` line). Apache-2.0, ungated.
- **Served via**: Ollama's official quantized GGUF conversions under the
  `qwen3` library tag, **not** a local GGUF build — `ollama pull` fetches
  these from Ollama's registry the first time the container starts.
  Confirmed tags (`https://ollama.com/library/qwen3/tags`):
  - `qwen3:4b-instruct-2507-q4_K_M` — **default**, ~2.6GB on disk, ~3GB RSS
    at runtime. Best fit for small CPU boxes.
  - `qwen3:4b-instruct-2507-q8_0` — ~4.5GB, higher fidelity, needs more RAM
    and is noticeably slower on CPU.
  - `qwen3:4b-instruct-2507-fp16` — ~8.5GB, full precision; only worth it
    if you have RAM to spare and care about maximum quality.
  Override with `MODEL_TAG=qwen3:4b-instruct-2507-q8_0 ./start.sh`.

## Sizing (CPU-only)

4B parameters at Q4_K_M is comfortable on modest hardware: budget ~4GB RAM
headroom (model + KV cache + Ollama runtime) and expect roughly 1
token/core/sec-ish throughput on modern x86_64/ARM64 cores (varies a lot
with core count and memory bandwidth — measure on your box). There is
nothing Jetson- or GPU-specific in this script; it works on any Linux
docker host with enough RAM, including regular VPS/cloud CPU instances.

## Usage

```bash
cd deploy/cpu/qwen3-4b-instruct-2507
./start.sh          # pulls ollama/ollama:latest, pulls the model, waits for readiness
./stop.sh           # stops the container (data + model cache in ./.ollama persist)

# Quick smoke test
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "qwen3:4b-instruct-2507-q4_K_M",
  "prompt": "Say hi in one sentence.",
  "stream": false
}'
```

Model weights and Ollama state are cached in `./.ollama` (bind-mounted,
gitignored) so restarts do not re-download anything.

## Wiring into NixClaw

`config/agent.yaml` (see `config/agent.yaml.example`):

```yaml
ollama:
  host: "127.0.0.1:11434"       # or "host.docker.internal:11434" from inside
                                 # the nixclaw container, or the compose
                                 # service name if you add this as a compose
                                 # service instead of running start.sh standalone
  model: "qwen3:4b-instruct-2507-q4_K_M"
```

Leave `config.backend` unset (or `"ollama"`, the default).

## Context window (`OLLAMA_CONTEXT_LENGTH`)

Ollama's **runtime** context window defaults to 4096 tokens regardless of a
model's native max context (this model supports up to 262144) — and there
is **no per-request way to override it** if you talk to Ollama through its
OpenAI-compatible `/v1/chat/completions` endpoint (e.g. nixclaw's `vllm`
backend pointed at this server, or any OpenAI-SDK client). A `num_ctx`/
`options` field in the request body is silently ignored on that endpoint;
only the server-wide `OLLAMA_CONTEXT_LENGTH` env var (or a Modelfile) sets
it. Once a real conversation's prompt exceeds 4096 tokens you'll get:

```json
{"error":{"code":400,"message":"request (9471 tokens) exceeds the available context size (4096 tokens), try increasing it","type":"exceed_context_size_error"}}
```

`start.sh` sets `OLLAMA_CONTEXT_LENGTH=32768` by default — override with
`OLLAMA_CONTEXT_LENGTH=65536 ./start.sh` etc. if needed, but note KV-cache
RAM usage scales with this value on top of the model weights.

## Environment overrides

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_TAG` | `qwen3:4b-instruct-2507-q4_K_M` | Any tag from `ollama.com/library/qwen3` |
| `IMAGE` | `ollama/ollama:latest` | Upstream Ollama CPU/GPU-auto image |
| `CONTAINER_NAME` | `ollama-qwen3-4b-cpu` | |
| `PORT` | `11434` | Ollama's default port |
| `OLLAMA_NUM_THREADS` | (unset, Ollama auto-detects) | Pin CPU thread count |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Keep at 1 unless you plan to serve multiple models concurrently |
| `OLLAMA_CONTEXT_LENGTH` | `32768` | Server-wide runtime context window; see above |
| `OLLAMA_KEEP_ALIVE` | `24h` | How long an idle model stays resident before Ollama unloads it (default is `5m` if unset — causes a multi-second reload stall on the next request after any idle gap). `-1` = never unload. |
