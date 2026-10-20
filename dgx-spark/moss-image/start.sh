#!/usr/bin/env bash
set -euo pipefail

# MOSS-Image text-to-image server for the DGX Spark (GB10, ARM64, sm_121a).
# Builds the image from this directory's Dockerfile and runs it next to
# vLLM/sglang on the shared unified-memory pool. Same layout as
# dgx-spark/qwen-image-2.1/start.sh.
#
# Default: MOSS-Image-SFT, everything bf16 (~27 GB resident).
#   TEXT_ENCODER_QUANT=fp8 ./start.sh                       -> ~19 GB
#   IMAGE_MODEL_ID=OpenMOSS-Team/MOSS-Image-Base ./start.sh -> base model
#
# Uses host port 8002 like qwen-image-2.1 / flux2-klein: stop those first.

IMAGE="moss-image:local"
CONTAINER_NAME="moss-image"
HOST="0.0.0.0"
PORT="${MOSS_IMAGE_PORT:-8002}"
PID_FILE=".moss-image.pid"
LOG_FILE=".moss-image.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
READY_URL="http://127.0.0.1:${PORT}/health"

IMAGE_MODEL_ID="${IMAGE_MODEL_ID:-OpenMOSS-Team/MOSS-Image-SFT}"
MODEL_REVISION="${MODEL_REVISION:-}"
TEXT_ENCODER_QUANT="${TEXT_ENCODER_QUANT:-none}"
DROP_VISION_TOWER="${DROP_VISION_TOWER:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
UNCOND_SKIP_LAYERS="${UNCOND_SKIP_LAYERS:-22}"
DEFAULT_SIZE="${DEFAULT_SIZE:-1024x1024}"
DENOISE_MB_PER_MP="${DENOISE_MB_PER_MP:-1500}"
TILED_VAE_MB="${TILED_VAE_MB:-1000}"
ADMISSION_MARGIN_MB="${ADMISSION_MARGIN_MB:-2000}"
ADMISSION_WAIT_SECONDS="${ADMISSION_WAIT_SECONDS:-60}"
VAE_TILING="${VAE_TILING:-auto}"
UNTILED_VAE_MB_PER_MP="${UNTILED_VAE_MB_PER_MP:-6000}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NO_CACHE="${NO_CACHE:-0}"
# Kill this before the LLM or system services when the shared pool runs out.
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
# At most two restarts: an OOM-killed server that comes straight back
# reloads ~27 GB into a box that is already short of memory.
RESTART_POLICY="${RESTART_POLICY:-on-failure:2}"

for tool in docker curl; do
  command -v "${tool}" >/dev/null 2>&1 || { echo "${tool} is not on PATH"; exit 1; }
done

mkdir -p "${HF_HOME}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

# Host networking: a second server on the same port would fail to bind
# only after the full image build and weight download.
if curl -fsS -o /dev/null --max-time 2 "${READY_URL}" 2>/dev/null; then
  echo "Port ${PORT} is already serving (qwen-image-2.1 or flux2-klein?). Stop it first, or set MOSS_IMAGE_PORT."
  exit 1
fi

echo "Building ${IMAGE} from ${WORK_DIR}"
build_args=()
if [[ "${NO_CACHE}" == "1" ]]; then
  build_args+=(--no-cache)
fi
docker build "${build_args[@]}" -t "${IMAGE}" "${WORK_DIR}"

echo "Starting MOSS-Image container for ${IMAGE_MODEL_ID}${MODEL_REVISION:+@${MODEL_REVISION}}"
echo "DiT bf16, VAE fp32, text encoder: ${TEXT_ENCODER_QUANT}; steps ${NUM_INFERENCE_STEPS}, cfg ${GUIDANCE_SCALE}, uncond skip ${UNCOND_SKIP_LAYERS}"
echo "Listening on ${HOST}:${PORT}, writing progress to ${LOG_FILE}"

echo "[$(date -Is)] launching moss-image container" >"${LOG_FILE}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus all \
  --oom-score-adj "${OOM_SCORE_ADJ}" \
  --restart "${RESTART_POLICY}" \
  -e IMAGE_MODEL_ID="${IMAGE_MODEL_ID}" \
  -e MODEL_REVISION="${MODEL_REVISION}" \
  -e TEXT_ENCODER_QUANT="${TEXT_ENCODER_QUANT}" \
  -e DROP_VISION_TOWER="${DROP_VISION_TOWER}" \
  -e NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  -e GUIDANCE_SCALE="${GUIDANCE_SCALE}" \
  -e UNCOND_SKIP_LAYERS="${UNCOND_SKIP_LAYERS}" \
  -e DEFAULT_SIZE="${DEFAULT_SIZE}" \
  -e DENOISE_MB_PER_MP="${DENOISE_MB_PER_MP}" \
  -e TILED_VAE_MB="${TILED_VAE_MB}" \
  -e ADMISSION_MARGIN_MB="${ADMISSION_MARGIN_MB}" \
  -e ADMISSION_WAIT_SECONDS="${ADMISSION_WAIT_SECONDS}" \
  -e VAE_TILING="${VAE_TILING}" \
  -e UNTILED_VAE_MB_PER_MP="${UNTILED_VAE_MB_PER_MP}" \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
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
    kill "${log_follow_pid}"
  fi
}
trap cleanup EXIT

(docker logs -f "${CONTAINER_NAME}" >> "${LOG_FILE}" 2>&1) &
log_follow_pid=$!

echo "Waiting for readiness at ${READY_URL}"
echo "(first run downloads ~27 GB of weights - this can take a while)"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "moss-image container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  if grep -q "Pipeline load failed" "${LOG_FILE}" 2>/dev/null; then
    echo "Pipeline load failed:"
    curl -sS "${READY_URL}" || true
    echo
    tail -n 60 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 10
done

echo "moss-image is ready"
echo "OpenAI-shaped base URL: http://${HOST}:${PORT}/v1"
