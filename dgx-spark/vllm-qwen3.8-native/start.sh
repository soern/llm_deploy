#!/usr/bin/env bash
set -euo pipefail

# vLLM launch script for the native (BF16, unquantized) Qwen/Qwen3.8-27B on
# NVIDIA DGX Spark (GB10, sm_121a). This is the "highest quality, slowest"
# variant -- see the NVFP4 sibling deploy (../vllm-qwen3.8-nvfp4)
# for the memory-efficient/faster alternative. See README.md for the full
# rationale and sources.
#
# Params sourced from:
# - Model card: https://huggingface.co/Qwen/Qwen3.8-27B (arch, context,
#   thinking-mode sampling defaults, MTP support)
# - Anduin Xue's benchmarked BF16-on-GB10 config for the same-size, same-arch
#   Qwen3.6-27B dense model (single GB10, --gpu-memory-utilization 0.95,
#   262144 ctx, max-num-seqs 4 all worked; ~5.5-16 tok/s depending on load):
#   https://anduin.aiursoft.com/post/2026/5/27/setup-dgx-spark-with-qwen-36-27b-dense
# - eugr/spark-vllm-docker image (widely-used community GB10/sm_121a vLLM
#   build): https://github.com/eugr/spark-vllm-docker
# - vLLM Recipes confirms MTP speculative decoding is usable with the BF16
#   checkpoint too, not just the quantized ones:
#   https://recipes.vllm.ai/Qwen/Qwen3.8-27B

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8}"
IMAGE="${IMAGE:-eugr/spark-vllm:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen3.8-native}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Conservative vs. Anduin's 0.95: this config additionally reserves headroom
# for vision buffers and MTP draft state that his bf16 benchmark didn't
# exercise. Raise toward 0.95 (matching his exact result) if you disable
# vision (LANGUAGE_ONLY=1) or MTP (ENABLE_MTP=0) and want the extra budget.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# MTP (multi-token prediction) is trained into the Qwen3.8 checkpoint itself
# -- no separate draft-model download needed (unlike the older Qwen3.6+DFlash
# deploy, which required a second ~GB-scale draft repo).
ENABLE_MTP="${ENABLE_MTP:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"

# Set to 1 to skip vision weights/buffers and save memory (text-only use).
LANGUAGE_ONLY="${LANGUAGE_ONLY:-0}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":4,\"video\":1}}"
MEDIA_IO_KWARGS="${MEDIA_IO_KWARGS:-{\"video\":{\"num_frames\":-1}}}"

PID_FILE=".vllm.pid"
LOG_FILE=".vllm.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
CACHE_DIR="${WORK_DIR}/.cache/vllm-runtime"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}" "${CACHE_DIR}"

if [[ ! -f "${WORK_DIR}/chat_template.jinja" ]]; then
  echo "Missing chat_template.jinja in ${WORK_DIR}"
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

VLLM_ARGS=(
  serve "${MODEL_ID}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size 1
  --trust-remote-code
  --dtype bfloat16
  --kv-cache-dtype bfloat16
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --enable-chunked-prefill
  --enable-prefix-caching
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --enable-auto-tool-choice
  --generation-config vllm
  --chat-template /workspace/chat_template.jinja
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
)

if [[ "${LANGUAGE_ONLY}" == "1" ]]; then
  VLLM_ARGS+=(--language-model-only)
else
  VLLM_ARGS+=(--limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" --media-io-kwargs "${MEDIA_IO_KWARGS}")
fi

if [[ "${ENABLE_MTP}" == "1" ]]; then
  VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}")
fi

echo "Starting vLLM container for ${MODEL_ID} (native BF16)"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Vision: $([ "${LANGUAGE_ONLY}" = "1" ] && echo off || echo on)  MTP: ${ENABLE_MTP} (k=${NUM_SPECULATIVE_TOKENS})"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching vLLM container (native BF16)
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus all \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e VLLM_CACHE_ROOT=/root/.cache/vllm \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${CACHE_DIR}:/root/.cache" \
  -v "${WORK_DIR}/chat_template.jinja:/workspace/chat_template.jinja" \
  -v "${WORK_DIR}:/workspace" \
  --entrypoint vllm \
  "${IMAGE}" \
  "${VLLM_ARGS[@]}" \
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

echo "Waiting for HTTP readiness at ${READY_URL}"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "vLLM container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 5
done

echo "vLLM is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
