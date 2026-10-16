#!/usr/bin/env bash
set -euo pipefail

# Kokoro-82M ONNX German TTS ("Martin" voice) server launch script.
# CPU-first: no GPU flags unless ORT_PROVIDER=cuda. Structured like
# deploy/dgx-spark/flux2-klein/start.sh: named container, host networking,
# host-side model-cache bind mount, PID/log files next to this script,
# readiness poll with an exited-container check.

IMAGE="kokoro-tts-german:local"
CONTAINER_NAME="kokoro-tts-de"
HOST="0.0.0.0"
PORT="${TTS_PORT:-8003}"
PID_FILE=".kokoro-tts.pid"
LOG_FILE=".kokoro-tts.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READY_URL="http://127.0.0.1:${PORT}/health"

TTS_MODEL_ID="${TTS_MODEL_ID:-Godelaune/Kokoro-82M-ONNX-German-Martin}"
MODEL_DIR="${MODEL_DIR:-/models/kokoro-de}"
DEFAULT_LANG="${DEFAULT_LANG:-de}"
DEFAULT_VOICE="${DEFAULT_VOICE:-martin}"
ORT_PROVIDER="${ORT_PROVIDER:-cpu}"
MODEL_CACHE="${WORK_DIR}/.cache/models"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${MODEL_CACHE}"

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

echo "Starting Kokoro TTS (German, voice=${DEFAULT_VOICE}) container for ${TTS_MODEL_ID}"
echo "ORT provider: ${ORT_PROVIDER}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching ${CONTAINER_NAME} container
EOF

GPU_ARGS=()
if [[ "${ORT_PROVIDER}" == "cuda" ]]; then
  GPU_ARGS=(--gpus all)
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  "${GPU_ARGS[@]}" \
  -e TTS_MODEL_ID="${TTS_MODEL_ID}" \
  -e MODEL_DIR="${MODEL_DIR}" \
  -e DEFAULT_LANG="${DEFAULT_LANG}" \
  -e DEFAULT_VOICE="${DEFAULT_VOICE}" \
  -e ORT_PROVIDER="${ORT_PROVIDER}" \
  -e PORT="${PORT}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${MODEL_CACHE}:${MODEL_DIR}" \
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
echo "(first run downloads the onnx model + voice pack from HuggingFace - ~80MB, should be quick)"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "${CONTAINER_NAME} container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 5
done

echo "${CONTAINER_NAME} is ready"
echo "OpenAI-shaped base URL: http://${HOST}:${PORT}/v1"
