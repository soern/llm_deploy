#!/usr/bin/env bash
set -euo pipefail

# FLUX.2 [klein] 9B image-generation server launch script for the DGX Spark
# (GB10, ARM64, sm_121a) box that already hosts vLLM. Builds the image from
# this directory's Dockerfile (nvcr.io/nvidia/pytorch base - no public image
# exists for this stack) and runs it alongside vLLM, sharing the GPU.
#
# Structured like deploy/dgx-spark/vllm-qwen3.6/start.sh: named container, host
# networking, host-side cache bind mounts, PID/log files next to this
# script, readiness poll with an exited-container check.

IMAGE="flux2-klein-9b:local"
CONTAINER_NAME="flux2-klein"
HOST="0.0.0.0"
PORT="${FLUX_PORT:-8002}"
PID_FILE=".flux.pid"
LOG_FILE=".flux.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
READY_URL="http://127.0.0.1:${PORT}/health"

IMAGE_MODEL_ID="${IMAGE_MODEL_ID:-black-forest-labs/FLUX.2-klein-9B}"
QUANT_MODE="${QUANT_MODE:-none}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-4}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Building ${IMAGE} from ${WORK_DIR}"
docker build -t "${IMAGE}" "${WORK_DIR}"

echo "Starting FLUX.2 [klein] container for ${IMAGE_MODEL_ID}"
echo "Quant mode: ${QUANT_MODE}, steps: ${NUM_INFERENCE_STEPS}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching flux2-klein container
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus all \
  -e IMAGE_MODEL_ID="${IMAGE_MODEL_ID}" \
  -e QUANT_MODE="${QUANT_MODE}" \
  -e NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  -e GUIDANCE_SCALE="${GUIDANCE_SCALE}" \
  -e PORT="${PORT}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME=/root/.cache/huggingface \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  "${IMAGE}" \
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
echo "(first run downloads ~40GB of weights and JIT-quantizes the transformer - this can take a while)"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "flux2-klein container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 10
done

echo "flux2-klein is ready"
echo "OpenAI-shaped base URL: http://${HOST}:${PORT}/v1"
