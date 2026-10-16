#!/usr/bin/env bash
set -euo pipefail

# llama.cpp `llama-server` launch script for LiquidAI/LFM2.5-8B-A1B on an
# NVIDIA Jetson Orin Nano (8GB unified RAM/VRAM) running JetPack 7
# (L4T R39, CUDA 13). Alternative to ../qwen3-4b-instruct-2507;
# both use the same llama.cpp image and port, so run only one at a time.
# See README.md for model choice and memory sizing.
#
# The image is built locally from ./Dockerfile on first run (sm_87, CUDA 13).
# It uses the same tag as the Qwen deploy, so whichever deploy runs first
# builds it and the other reuses it. Serves an OpenAI-compatible /v1 API, so
# nixclaw uses its `vllm` backend against it.

MODEL_REPO="${MODEL_REPO:-LiquidAI/LFM2.5-8B-A1B-GGUF}"
MODEL_FILE="${MODEL_FILE:-LFM2.5-8B-A1B-Q4_K_M.gguf}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-lfm2.5-8b-a1b}"
IMAGE="${IMAGE:-nixclaw/llama-server-jetson-orin:local}"
CUDA_VERSION="${CUDA_VERSION:-13.0.1}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
BUILD_JOBS="${BUILD_JOBS:-3}"
REBUILD="${REBUILD:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-llama-lfm2.5-8b-a1b-jetson}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# Only 6 of the 24 layers are attention layers (the rest are short conv
# blocks with a fixed-size state), 8 KV heads x 64 dims: ~12KB/token at f16,
# ~6.5KB/token at q8_0. 65536 tokens at q8_0 is ~0.4GB on top of ~4.8GB
# weights. The model's max is 128000.
CTX_SIZE="${CTX_SIZE:-65536}"
CACHE_TYPE="${CACHE_TYPE:-q8_0}"
# llama-server splits CTX_SIZE evenly across slots; 1 slot keeps the whole
# window for one request (extra requests queue server-side).
PARALLEL="${PARALLEL:-1}"
# llama.cpp sizes its prompt-processing compute buffers off the batch/ubatch
# dims at load time, not just off CTX_SIZE/KV cache - on this 8GB unified-
# memory board that spike (not steady-state KV usage) is what triggers an
# OOM-kill under a real prefill. Defaults (2048/512) are tuned for discrete
# GPUs with headroom; lower values shrink the reserved buffer at the cost of
# slower prompt processing. Raise back toward the llama.cpp defaults only if
# this board gets more headroom (e.g. after enabling swap).
BATCH_SIZE="${BATCH_SIZE:-1024}"
UBATCH_SIZE="${UBATCH_SIZE:-256}"
# Model card recommends temperature 0.2 and repetition penalty 1.05. These
# are server defaults only; a request's own sampling params override them.
TEMP="${TEMP:-0.2}"
REPEAT_PENALTY="${REPEAT_PENALTY:-1.05}"
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

# The Qwen deploy uses the same port; with --network host the second server
# would fail to bind, and both models don't fit in 8GB anyway.
if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
  echo "Something is already serving on port ${PORT} (the Qwen deploy?)."
  echo "Stop it first, e.g. ../qwen3-4b-instruct-2507/stop.sh"
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
echo "Context: ${CTX_SIZE} tokens, KV cache ${CACHE_TYPE}, ${PARALLEL} slot(s), batch ${BATCH_SIZE}/${UBATCH_SIZE}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching llama-server container (Jetson Orin Nano, ${MODEL_FILE})
EOF

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
  --batch-size "${BATCH_SIZE}" \
  --ubatch-size "${UBATCH_SIZE}" \
  --flash-attn on \
  --cache-type-k "${CACHE_TYPE}" \
  --cache-type-v "${CACHE_TYPE}" \
  --temp "${TEMP}" \
  --repeat-penalty "${REPEAT_PENALTY}" \
  --jinja \
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

# Fail loudly if it silently fell back to CPU.
if docker logs "${CONTAINER_NAME}" 2>&1 | grep -qE 'offloaded [1-9][0-9]*/[0-9]+ layers to GPU'; then
  docker logs "${CONTAINER_NAME}" 2>&1 | grep -E 'CUDA devices|Device 0|offloaded .* layers to GPU' || true
else
  echo "WARNING: no layers were offloaded to the GPU -- inference is running on CPU."
  echo "Check 'docker logs ${CONTAINER_NAME} | grep -i cuda' and README.md Troubleshooting."
fi

echo "llama-server is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Set config/agent.yaml: backend: \"vllm\", vllm.host: \"http://<jetson>:${PORT}\", vllm.model: \"${SERVED_MODEL_NAME}\", vllm.options.context_window: ${CTX_SIZE}"
