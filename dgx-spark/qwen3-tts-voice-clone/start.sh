#!/usr/bin/env bash
set -euo pipefail

# Qwen3-TTS voice-clone service launch script. Runs on the DGX Spark (GPU) or
# on any CPU box: cloning is cheap; only the optional preview sentence is real
# generation (~a minute on CPU). Same structure as
# deploy/dgx-spark/qwen3-tts/start.sh.
#
# Voices are written to VOICES_DIR, by default the TTS server's voices dir
# (../qwen3-tts/voices), so on a shared host a new voice is live at
# once. The HF cache is shared with the TTS deploy too (same Base model).
# Companion scripts: clone_voice.sh, fetch_voice.sh, list_voices.sh.

IMAGE="qwen3-tts-voice-clone:local"
CONTAINER_NAME="qwen3-tts-voice-clone"
HOST="0.0.0.0"
PORT="${QWEN_CLONE_PORT:-8011}"
PID_FILE=".qwen3-tts-voice-clone.pid"
LOG_FILE=".qwen3-tts-voice-clone.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${HF_HOME_HOST:-${WORK_DIR}/../qwen3-tts/.cache/huggingface}"
VOICES_DIR="${VOICES_DIR:-${WORK_DIR}/../qwen3-tts/voices}"
READY_URL="http://127.0.0.1:${PORT}/health"

QWEN_TTS_BASE_MODEL="${QWEN_TTS_BASE_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3-turbo}"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

cd "${WORK_DIR}"
mkdir -p "${HF_HOME}" "${VOICES_DIR}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  build_args=()
  gpu_args=(--gpus all)
  echo "GPU host: building on the NVIDIA PyTorch base image"
else
  build_args=(--build-arg BASE_IMAGE=python:3.12-slim --build-arg INSTALL_CPU_TORCH=1)
  gpu_args=()
  echo "No nvidia-smi: building the CPU variant"
fi

echo "Building ${IMAGE} from ${WORK_DIR}"
docker build "${build_args[@]}" -t "${IMAGE}" "${WORK_DIR}"

echo "Starting voice-clone service (base ${QWEN_TTS_BASE_MODEL}, whisper ${WHISPER_MODEL})"
echo "Voices dir: ${VOICES_DIR}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching qwen3-tts-voice-clone container
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  "${gpu_args[@]}" \
  --restart unless-stopped \
  -e PORT="${PORT}" \
  -e QWEN_TTS_BASE_MODEL="${QWEN_TTS_BASE_MODEL}" \
  -e WHISPER_MODEL="${WHISPER_MODEL}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e QWEN_TTS_DTYPE="${QWEN_TTS_DTYPE:-}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e VOICES_DIR=/voices \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${VOICES_DIR}:/voices" \
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
echo "(first run downloads the ~4.5GB Base model unless the TTS deploy already did, plus ~1.6GB whisper)"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "qwen3-tts-voice-clone container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 10
done

echo "qwen3-tts-voice-clone is ready"
echo "Clone a voice: ./clone_voice.sh -l de CLIP NAME   (QWEN_CLONE_URL=http://<host>:${PORT} from elsewhere)"
