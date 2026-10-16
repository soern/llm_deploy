#!/usr/bin/env bash
set -euo pipefail

# vLLM launch script for unsloth/Qwen3.8-27B-NVFP4 on NVIDIA DGX Spark
# (GB10, sm_121a). This is the memory-efficient/faster variant -- see the
# native sibling deploy (../vllm-qwen3.8-native) for the
# unquantized BF16 alternative. See README.md for the full rationale.
#
# Params sourced from gitcommit90/qwen38-27b-dgx-spark, a benchmarked
# start.sh for this exact checkpoint on this exact hardware (single GB10):
# https://github.com/gitcommit90/qwen38-27b-dgx-spark
# Reported: ~21.4 tok/s single-stream decode, ~66.5 tok/s aggregate at
# concurrency 4, ~1s TTFT, 81.1GiB KV cache at 8.73x concurrency headroom.
# Cross-checked against https://recipes.vllm.ai/Qwen/Qwen3.8-27B (same
# quantization/reasoning/tool-call flags, different checkpoint org) and the
# model card's own thinking-mode sampling recommendations:
# https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4
#
# A community comment on the unsloth NVFP4 discussion board argued
# --gpu-memory-utilization could go as low as ~0.39-0.42 based on a
# weights+KV-cache size estimate, disputing the source recipe's 0.90:
# https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/discussions/7
# That estimate was NOT reproduced against real throughput/OOM numbers on
# this hardware (unlike the 0.90 figure below, which comes from an actual
# benchmark run), so it's used here only as GPU_MEM_UTIL_LOW, a documented
# fallback if you hit OOM or want more headroom for other processes on the
# box -- not as the default.

MODEL_ID="${MODEL_ID:-unsloth/Qwen3.8-27B-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8}"
IMAGE="${IMAGE:-eugr/spark-vllm:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen3.8-nvfp4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Benchmarked value from gitcommit90/qwen38-27b-dgx-spark (see header). If
# you hit OOM, or want more host headroom, try GPU_MEM_UTIL=0.42 -- the
# unverified-but-plausible lower bound from the discussions/7 thread (see
# header); expect to give up some KV-cache concurrency room in exchange.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"

# MTP (multi-token prediction) is trained into the checkpoint itself -- no
# separate draft-model download needed (unlike the older Qwen3.6+DFlash
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
  --quantization compressed-tensors
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

echo "Starting vLLM container for ${MODEL_ID} (NVFP4)"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Vision: $([ "${LANGUAGE_ONLY}" = "1" ] && echo off || echo on)  MTP: ${ENABLE_MTP} (k=${NUM_SPECULATIVE_TOKENS})"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching vLLM container (NVFP4)
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus all \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
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
