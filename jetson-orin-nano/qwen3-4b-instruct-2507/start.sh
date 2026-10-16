#!/usr/bin/env bash
set -euo pipefail

# llama.cpp `llama-server` launch script for Qwen/Qwen3-4B-Instruct-2507 on
# an NVIDIA Jetson Orin Nano (8GB unified RAM/VRAM) running JetPack 7
# (L4T R39, CUDA 13). See README.md for why llama.cpp (not Ollama/vLLM),
# quant choice, and memory sizing.
#
# The image is built locally from ./Dockerfile on first run (sm_87, CUDA 13):
# no prebuilt llama.cpp image targets arm64 + CUDA 13 + Orin. Serves an
# OpenAI-compatible /v1 API, so nixclaw uses its `vllm` backend against it.

MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3-4B-Instruct-2507-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-instruct-2507}"
IMAGE="${IMAGE:-nixclaw/llama-server-jetson-orin:local}"
CUDA_VERSION="${CUDA_VERSION:-13.0.1}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
BUILD_JOBS="${BUILD_JOBS:-3}"
REBUILD="${REBUILD:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-llama-qwen3-4b-jetson}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# KV cache for this model is ~144KB/token at f16, ~76KB/token at q8_0
# (36 layers x 8 KV heads x 128). 16384 tokens left 2.2GB free on the board;
# 32768 adds ~1.25GB KV and leaves ~0.9GB for prefill spikes and the OS.
CTX_SIZE="${CTX_SIZE:-32768}"
CACHE_TYPE="${CACHE_TYPE:-q8_0}"
# llama-server splits CTX_SIZE evenly across slots; 1 slot keeps the whole
# window for one request (extra requests queue server-side).
PARALLEL="${PARALLEL:-1}"
# Optional n-gram speculative decoding (no draft model, no extra memory):
# drafts tokens by matching the output against earlier context, so it only
# speeds up text that repeats the prompt (file edits, quoted code, tool-call
# arguments). Empty = off; e.g. SPEC_TYPE=ngram-mod. This model has no MTP head.
SPEC_TYPE="${SPEC_TYPE:-}"
# The NVIDIA container runtime auto-detects "csv" mode on this board (L4T),
# and in csv mode `--gpus all` is rejected ("invoking the NVIDIA Container
# Runtime Hook directly ... is not supported"). Use the runtime explicitly;
# the nvidia/cuda base image already sets NVIDIA_VISIBLE_DEVICES=all.
GPU_DOCKER_ARG="${GPU_DOCKER_ARG:---runtime nvidia}"

PID_FILE=".llama.pid"
LOG_FILE=".llama.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${WORK_DIR}/.models"
MODEL_PATH="${MODELS_DIR}/${MODEL_FILE}"
MODEL_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
READY_URL="http://127.0.0.1:${PORT}/health"

cd "${WORK_DIR}"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

# The LFM2.5 deploy uses the same port; with --network host the second server
# would fail to bind, and both models don't fit in 8GB anyway.
if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
  echo "Something is already serving on port ${PORT} (the LFM2.5 deploy?)."
  echo "Stop it first, e.g. ../lfm2.5-8b-a1b/stop.sh"
  exit 1
fi

if [[ "${REBUILD}" == "1" ]] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  # Pin a branch/tag to its current commit: the Dockerfile's clone step only
  # re-runs when its command line changes, so building "master" again would
  # silently reuse the cached (old) clone.
  build_ref="${LLAMA_CPP_REF}"
  if ! [[ "${build_ref}" =~ ^[0-9a-f]{40}$ ]]; then
    build_ref="$(curl -fsS "https://api.github.com/repos/ggml-org/llama.cpp/commits/${LLAMA_CPP_REF}" \
      -H 'Accept: application/vnd.github.sha' 2>/dev/null || true)"
    if ! [[ "${build_ref}" =~ ^[0-9a-f]{40}$ ]]; then
      echo "Could not resolve llama.cpp ref '${LLAMA_CPP_REF}' to a commit (GitHub API)."
      echo "Set LLAMA_CPP_REF=<40-char commit sha> and try again."
      exit 1
    fi
  fi
  echo "Building ${IMAGE} (llama.cpp ${LLAMA_CPP_REF} = ${build_ref}, CUDA ${CUDA_VERSION}, sm_87)"
  echo "This compiles CUDA kernels on the Orin -- expect 20-60 minutes on first build."
  docker build \
    --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
    --build-arg "LLAMA_CPP_REF=${build_ref}" \
    --build-arg "BUILD_JOBS=${BUILD_JOBS}" \
    -t "${IMAGE}" \
    "${WORK_DIR}"
fi

mkdir -p "${MODELS_DIR}"
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Downloading ${MODEL_URL}"
  auth_args=()
  if [[ -n "${HF_TOKEN:-}" ]]; then
    auth_args=(-H "Authorization: Bearer ${HF_TOKEN}")
  fi
  curl -fL --retry 5 -C - "${auth_args[@]}" \
    -o "${MODEL_PATH}.part" "${MODEL_URL}"
  mv "${MODEL_PATH}.part" "${MODEL_PATH}"
fi

echo "Starting llama-server container (Jetson Orin Nano, CUDA) for ${MODEL_FILE}"
echo "Image: ${IMAGE} (llama.cpp $(docker run --rm --entrypoint cat "${IMAGE}" /LLAMA_CPP_COMMIT))"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Context: ${CTX_SIZE} tokens, KV cache ${CACHE_TYPE}, ${PARALLEL} slot(s)"
echo "Speculative decoding: ${SPEC_TYPE:-off}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching llama-server container (Jetson Orin Nano, ${MODEL_FILE})
EOF

spec_args=()
if [[ -n "${SPEC_TYPE}" ]]; then
  spec_args=(--spec-type "${SPEC_TYPE}")
fi

# --jinja: use the GGUF's own chat template, required for OpenAI tool calls.
# -ngl 999: offload every layer to the GPU.
# Quantized V cache requires flash attention, hence --flash-attn on.
docker run -d \
  --name "${CONTAINER_NAME}" \
  ${GPU_DOCKER_ARG} \
  --network host \
  -v "${MODELS_DIR}:/models:ro" \
  "${IMAGE}" \
  --model "/models/${MODEL_FILE}" \
  --alias "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --n-gpu-layers 999 \
  --ctx-size "${CTX_SIZE}" \
  --parallel "${PARALLEL}" \
  --flash-attn on \
  --cache-type-k "${CACHE_TYPE}" \
  --cache-type-v "${CACHE_TYPE}" \
  --jinja \
  "${spec_args[@]}" \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

(docker logs -f "${CONTAINER_NAME}" >> "${LOG_FILE}" 2>&1) &
log_follow_pid=$!

echo "Waiting for model load at ${READY_URL}"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "llama-server container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 2
done

# The whole point of this deploy: fail loudly if it silently fell back to CPU.
if docker logs "${CONTAINER_NAME}" 2>&1 | grep -qE 'offloaded [1-9][0-9]*/[0-9]+ layers to GPU'; then
  docker logs "${CONTAINER_NAME}" 2>&1 | grep -E 'CUDA devices|Device 0|offloaded .* layers to GPU' || true
else
  echo "WARNING: no layers were offloaded to the GPU -- inference is running on CPU."
  echo "Check 'docker logs ${CONTAINER_NAME} | grep -i cuda' and README.md Troubleshooting."
fi

echo "llama-server is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Set config/agent.yaml: backend: \"vllm\", vllm.host: \"http://<jetson>:${PORT}\", vllm.model: \"${SERVED_MODEL_NAME}\""
