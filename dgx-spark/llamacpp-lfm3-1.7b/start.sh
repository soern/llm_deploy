#!/usr/bin/env bash
set -euo pipefail

# llama.cpp `llama-server` launch script for LiquidAI/LFM3-1.7B on an NVIDIA
# DGX Spark (GB10, arm64, CUDA 13). See README.md for why llama.cpp (neither
# vLLM nor SGLang implements the `Lfm3ForCausalLM` architecture yet, so the
# NVFP4/BF16 safetensors checkpoints are not servable here) and for sizing.
#
# Plain `docker run` against the official upstream multi-arch image
# (ghcr.io/ggml-org/llama.cpp:server-cuda13, linux/arm64 included) -- no
# custom image build, no sudo; same pattern as the sibling SGLang/vLLM
# deploys. LFM3 needs llama.cpp >= v0.8.0; the floating tag is newer.
#
# Port 8010 by default (the Spark's other LLM deploys sit on 8000/8004/8006).
# NOTE: ../qwen3-tts also defaults to 8010. start.sh refuses to start if the
# port is already bound, so run only one of them or override PORT=.

MODEL_REPO="${MODEL_REPO:-LiquidAI/LFM3-1.7B-GGUF}"
# 128GB unified memory: no need to squeeze. Q8_0 (1.85GB) is the near-lossless
# default; Q4_K_M (1.1GB) is ~equal on our llmtest suite and a bit faster.
MODEL_FILE="${MODEL_FILE:-LFM3-1.7B-Q8_0.gguf}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-lfm3-1.7b}"
IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"
CONTAINER_NAME="${CONTAINER_NAME:-llama-lfm3-1.7b-spark}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"
# 6 of 24 layers are full attention (4 KV heads x 128), 5 sliding-window
# (256 tokens), 13 conv: KV is ~12KB/token at f16. 131072 tokens ~= 1.5GB.
# llama-server splits CTX_SIZE evenly across slots (131072/4 = 32K each).
CTX_SIZE="${CTX_SIZE:-131072}"
PARALLEL="${PARALLEL:-4}"
CACHE_TYPE="${CACHE_TYPE:-f16}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
# Model-card sampling defaults (general: temp 0.1, top_k 50, rep 1.1); a
# request's own sampling params override these.
TEMP="${TEMP:-0.1}"
TOP_K="${TOP_K:-50}"
REPEAT_PENALTY="${REPEAT_PENALTY:-1.1}"
# The sibling deploys use --gpus all on this box. If the container can't see
# the GPU on a CDI-only DGX OS setup:
#   GPU_DOCKER_ARG="--device nvidia.com/gpu=all" ./start.sh
GPU_DOCKER_ARG="${GPU_DOCKER_ARG:---gpus all}"

PID_FILE=".llama.pid"
LOG_FILE=".llama.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${WORK_DIR}/.models"
MODEL_PATH="${MODELS_DIR}/${MODEL_FILE}"
MODEL_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
READY_URL="http://127.0.0.1:${PORT}/health"

cd "${WORK_DIR}"

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH"; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

# With --network host a bound port fails inside the container with a
# confusing log; catch it here. Any listener counts (not just HTTP ones) --
# e.g. ../qwen3-tts on the same default port.
if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :${PORT}" | grep -q .; then
  echo "Port ${PORT} is already in use on this host:"
  ss -H -ltnp "sport = :${PORT}" || true
  echo "Stop the other service or pick another port: PORT=<n> ./start.sh"
  exit 1
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

echo "Pulling ${IMAGE} (no-op if current)"
docker pull "${IMAGE}" >/dev/null

echo "Starting llama-server container (DGX Spark, CUDA) for ${MODEL_FILE}"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Context: ${CTX_SIZE} tokens total, KV cache ${CACHE_TYPE}, ${PARALLEL} slot(s), batch ${BATCH_SIZE}/${UBATCH_SIZE}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<LOGEOF
[$(date -Is)] launching llama-server container (DGX Spark, ${MODEL_FILE})
LOGEOF

# --jinja: use the GGUF's own chat template, required for OpenAI tool calls.
# -ngl 999: offload every layer to the GPU.
docker run -d \
  --name "${CONTAINER_NAME}" \
  ${GPU_DOCKER_ARG} \
  --network host \
  --ipc host \
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
  --top-k "${TOP_K}" \
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

# Fail loudly if it silently fell back to CPU (e.g. image without GB10 kernels).
if docker logs "${CONTAINER_NAME}" 2>&1 | grep -qE 'offloaded [1-9][0-9]*/[0-9]+ layers to GPU'; then
  docker logs "${CONTAINER_NAME}" 2>&1 | grep -E 'CUDA devices|Device 0|offloaded .* layers to GPU' || true
else
  echo "WARNING: no layers were offloaded to the GPU -- inference is running on CPU."
  echo "Check 'docker logs ${CONTAINER_NAME} | grep -i cuda' and README.md Troubleshooting."
fi

echo "llama-server is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Set config/agent.yaml: backend: \"vllm\", vllm.host: \"http://<spark>:${PORT}\", vllm.model: \"${SERVED_MODEL_NAME}\", vllm.options.context_window: $(( CTX_SIZE / PARALLEL ))"
