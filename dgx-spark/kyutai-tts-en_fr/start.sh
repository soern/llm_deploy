#!/usr/bin/env bash
set -euo pipefail

# Kyutai TTS 1.6B (en+fr) server launch script for the DGX Spark (GB10,
# ARM64, sm_121a) box that already hosts the LLM and FLUX.2. Structured like
# deploy/dgx-spark/flux2-klein/start.sh: named container, host networking,
# host-side HF cache bind mount, PID/log files next to this script, readiness
# poll with an exited-container check.

IMAGE="kyutai-tts-en_fr:local"
CONTAINER_NAME="kyutai-tts-en_fr"
HOST="0.0.0.0"
PORT="${KYUTAI_PORT:-8009}"
PID_FILE=".kyutai-tts.pid"
LOG_FILE=".kyutai-tts.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
READY_URL="http://127.0.0.1:${PORT}/health"

DEFAULT_VOICE="${DEFAULT_VOICE:-developpeuse}"
KYUTAI_N_Q="${KYUTAI_N_Q:-32}"
MAX_BATCH="${MAX_BATCH:-8}"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

cd "${WORK_DIR}"
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
echo "(first build compiles sphn from source with Rust - a few minutes)"
docker build -t "${IMAGE}" "${WORK_DIR}"

echo "Starting Kyutai TTS container (voice ${DEFAULT_VOICE}, n_q ${KYUTAI_N_Q})"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching kyutai-tts-en_fr container
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus all \
  --restart unless-stopped \
  -e PORT="${PORT}" \
  -e DEFAULT_VOICE="${DEFAULT_VOICE}" \
  -e KYUTAI_N_Q="${KYUTAI_N_Q}" \
  -e MAX_BATCH="${MAX_BATCH}" \
  -e FRENCH_NORMALIZE="${FRENCH_NORMALIZE:-1}" \
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
echo "(first run downloads ~4GB of weights)"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "kyutai-tts-en_fr container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 10
done

echo "kyutai-tts-en_fr is ready"
echo "OpenAI-shaped base URL: http://${HOST}:${PORT}/v1"
