# llama.cpp deployment for Qwen/Qwen3-4B-Instruct-2507 (NVIDIA Jetson Orin Nano, 8GB)

`start.sh` runs `Qwen/Qwen3-4B-Instruct-2507` (GGUF Q4_K_M) under llama.cpp's
`llama-server` with full CUDA offload on a Jetson Orin Nano running
**JetPack 7 / L4T R39** (CUDA 13). The image is built on the Jetson from
`./Dockerfile` on first run. It serves an OpenAI-compatible `/v1` API.

## Why llama.cpp (and not Ollama / vLLM / NVFP4)

- **Ollama (previous version of this deploy)**: the only Jetson-GPU Ollama
  images (`dustynv/ollama:*-r36.4`) are built for JetPack 6 (L4T r36,
  CUDA 12.6). On an R39 board they start but can't load the GPU
  (`libcuda.so.1.1: file too short`, `no compatible GPUs were discovered`)
  and silently run on CPU. Upstream `ollama/ollama` has no Jetson CUDA
  backend either.
- **JetPack 7 changed the CUDA stack**: R39 ships the same driver/CUDA 13
  SBSA userspace as ARM servers (the host `nvidia-smi` works and reports
  CUDA 13.2). That lets a stock `nvidia/cuda:13.x-*-ubuntu24.04` arm64 base
  image talk to the Orin GPU. No L4T-matched jetson-containers image needed.
  The container runtime still auto-detects csv mode on this board, so
  containers get the GPU through `--runtime nvidia`, not `--gpus all`.
- **vLLM / SGLang**: they would work in principle, but PyTorch plus the
  Python runtime and CUDA graphs take a large part of the 8GB that CPU and
  GPU share. llama.cpp is a single C++ binary with a small footprint, and it
  reuses the same GGUF quants.
- **NVFP4 / FP8**: need Blackwell (FP4) or Ada/Hopper (FP8) tensor cores.
  The Orin GPU is Ampere (sm_87), so these formats would at best be
  unpacked to 16-bit in software, giving no speedup over INT4. The fast path on
  Ampere is 4-bit weights with fp16 compute, which is what llama.cpp's
  Q4_K_M CUDA kernels do.

## Model

- **Checkpoint**: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
  dense, non-thinking, Apache-2.0.
- **GGUF**: `unsloth/Qwen3-4B-Instruct-2507-GGUF`, file
  `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (~2.5GB). Downloaded once by
  `start.sh` into `./.models/` (resumable). Override with `MODEL_REPO` /
  `MODEL_FILE`, e.g. `MODEL_FILE=Qwen3-4B-Instruct-2507-Q8_0.gguf` (~4.3GB,
  leaves little room for KV cache on 8GB).
- Served as `qwen3-4b-instruct-2507` (`SERVED_MODEL_NAME`).

## Sizing (8GB unified RAM/VRAM)

| Part | Approx. |
|------|---------|
| Weights (Q4_K_M) | 2.4GB |
| KV cache, 32768 tokens @ q8_0 (~76KB/token: 36 layers x 8 KV heads x 128) | 2.5GB |
| CUDA context + compute buffers | ~0.5-1GB |

Measured: with the old default of 16384 tokens, 2.2GB was still free after
loading. The default is now `CTX_SIZE=32768` (+1.25GB KV), which should
leave ~0.9GB for prefill compute spikes and the OS. Check `free -h` and
`tegrastats` during a long-prompt request; if it gets tight, go back to
`CTX_SIZE=24576` or use `CACHE_TYPE=q4_0` (~40KB/token, slightly worse
quality). 40K+ at q8_0 leaves too little headroom.

`PARALLEL=1`: llama-server splits `CTX_SIZE` evenly across slots, so
`PARALLEL=2` would halve each request's window. With 1 slot, concurrent
requests queue on the server. A headless boot
(`sudo systemctl set-default multi-user.target`) frees more RAM.

## Speed

Decode is memory-bandwidth bound: every token reads all ~2.4GB of weights.

- **Power mode** (biggest lever): the Orin Nano's MAXN SUPER mode raises
  memory bandwidth from 68 to 102 GB/s. Check with `sudo nvpmodel -q`,
  switch with `sudo nvpmodel -m <MAXN SUPER id>` (ids are listed in
  `/etc/nvpmodel.conf`), then `sudo jetson_clocks`. Expect up to ~1.4x.
- **MTP: not available.** Qwen3-4B-Instruct-2507 has no MTP head. A
  separate draft model (Qwen3-0.6B) would need its own weights (~0.5GB) and
  its own KV cache (~1.9GB at 32K), i.e. it would eat the context gain.
- **n-gram speculation** (`SPEC_TYPE=ngram-mod`): drafts from text already
  in the context, no extra memory. Helps when the answer repeats the prompt
  (editing a file, quoting code, tool arguments); ~no effect on free text.
  Requires an image built after early 2026 (`REBUILD=1`).
- **Smaller quant**: `MODEL_FILE=Qwen3-4B-Instruct-2507-Q4_0.gguf` (2.38GB)
  or `IQ4_XS` (2.27GB) read 5-10% less per token; small quality cost.
- **Different model with MTP**: Qwen3.5-4B ships an MTP head
  (`unsloth/Qwen3.5-4B-MTP-GGUF`, llama.cpp `--spec-type draft-mtp` since
  May 2026), needs ~4x less KV per token (only 8 of 32 layers use full
  attention) and scores much higher on coding and tool use. See
  `../qwen3.5-4b`.

## Usage

```bash
cd deploy/jetson-orin-nano/qwen3-4b-instruct-2507
./start.sh      # first run: builds image (20-60 min), downloads GGUF, waits for /health
./stop.sh

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b-instruct-2507","messages":[{"role":"user","content":"Say hi in one sentence."}]}'
```

`start.sh` checks the container log after readiness and prints the
`offloaded N/37 layers to GPU` line, or a loud WARNING if nothing was
offloaded. `tegrastats` on the host should show `GR3D_FREQ` busy during a
request.

## Wiring into NixClaw

`config/agent.yaml`:

```yaml
backend: "vllm"            # llama-server speaks the same OpenAI /v1 API
vllm:
  host: "http://<jetson-ip>:8000"
  model: "qwen3-4b-instruct-2507"
  options:
    context_window: 32768        # match CTX_SIZE / PARALLEL
    max_tokens: 4096
    max_concurrent_requests: 1   # match PARALLEL
```

## Environment overrides

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_REPO` | `unsloth/Qwen3-4B-Instruct-2507-GGUF` | HF repo holding the GGUF |
| `MODEL_FILE` | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | |
| `SERVED_MODEL_NAME` | `qwen3-4b-instruct-2507` | `--alias`, the `model` name clients send |
| `IMAGE` | `nixclaw/llama-server-jetson-orin:local` | Local tag built from `./Dockerfile` |
| `CUDA_VERSION` | `13.0.1` | `nvidia/cuda` base tag; must be <= the host's CUDA version in `nvidia-smi` |
| `LLAMA_CPP_REF` | `master` | llama.cpp git ref; the built commit is printed at start |
| `BUILD_JOBS` | `3` | Parallel nvcc jobs; higher OOMs the 8GB board |
| `REBUILD` | `0` | `1` rebuilds the image (e.g. to pick up a newer llama.cpp) |
| `CONTAINER_NAME` | `llama-qwen3-4b-jetson` | |
| `PORT` | `8000` | |
| `CTX_SIZE` | `32768` | Total KV context (shared across slots) |
| `CACHE_TYPE` | `q8_0` | KV cache type (`f16`, `q8_0`, `q4_0`) |
| `PARALLEL` | `1` | Server slots |
| `SPEC_TYPE` | unset | e.g. `ngram-mod`; llama.cpp `--spec-type`, see Speed |
| `GPU_DOCKER_ARG` | `--runtime nvidia` | `--gpus all` fails in the runtime's csv mode (see Troubleshooting) |
| `HF_TOKEN` | unset | Only needed for gated repos |

## Migrating from the Ollama version

```bash
docker rm -f ollama-qwen3-4b-jetson
rm -rf .ollama .ollama.pid .ollama.log
```

Leave `/etc/nvidia-container-runtime/config.toml` at `mode = "auto"`. On
jetson3 (R39.2.1), auto mode resolves to `csv`, and `start.sh` is set up for
that (`--runtime nvidia`).

## Troubleshooting

- **`docker build` gets killed / OOM**: lower `BUILD_JOBS` (e.g.
  `BUILD_JOBS=1 REBUILD=1 ./start.sh`), stop other services, or add swap.
- **`invoking the NVIDIA Container Runtime Hook directly (e.g. specifying the
  docker --gpus flag) is not supported`** (after `Auto-detected mode as
  'csv'`): the container runtime is in csv mode, which only works through
  `--runtime nvidia`. That's the default now. Don't pass
  `GPU_DOCKER_ARG="--gpus all"`.
- **`ggml_cuda_init: failed to initialize CUDA` / no layers offloaded**:
  confirm the container sees the GPU:
  `docker run --rm --runtime nvidia nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi`.
  In csv mode the host driver libraries are mounted from the lists in
  `/etc/nvidia-container-runtime/host-files-for-container.d/*.csv`; check
  that `libcuda.so*` is listed there. Also check `nvidia-ctk --version`.
- **`unsatisfied condition: cuda>=13.0`** at container start: the runtime
  couldn't match the image's `NVIDIA_REQUIRE_CUDA` against the driver. If
  the host `nvidia-smi` really reports CUDA >= 13.0, bypass the check with
  `GPU_DOCKER_ARG="--runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1"`.
- **`CUDA driver version is insufficient`**: `CUDA_VERSION` is newer than
  the host driver supports; pick a lower `13.x` tag and `REBUILD=1`.
- **`no kernel image is available for execution on the device`**: the
  image wasn't built for sm_87; rebuild (`REBUILD=1`).
- **400 `exceeds the available context size`**: raise `CTX_SIZE` (see
  Sizing) and set nixclaw's `vllm.options.context_window` to match.
- **Tool calls come back as plain text**: make sure `--jinja` is still in
  `start.sh`; it makes llama-server use the GGUF's Qwen chat template,
  which supports tool calls.
