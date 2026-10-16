#!/usr/bin/env bash
set -euo pipefail

# Qwen3-TTS server launch script. Meant for the DGX Spark (GB10, ARM64), where
# it shares the GPU with the LLM and FLUX.2; on a host without nvidia-smi it
# builds the CPU variant instead (slow, for smoke tests). Structured like
# deploy/dgx-spark/flux2-klein/start.sh: named container, host networking,
# host-side HF cache bind mount, PID/log files next to this script, readiness
# poll with an exited-container check.
#
# Cloned voices live in ./voices (mounted at /voices); deploy/dgx-spark/qwen3-tts-voice-clone/
# writes there by default, and new files are picked up without a restart.
#
# A second instance (e.g. to compare model sizes) needs its own container name
# and port: QWEN_TTS_CONTAINER=qwen3-tts-small QWEN_TTS_PORT=8012 ./start.sh

IMAGE="qwen3-tts:local"
CONTAINER_NAME="${QWEN_TTS_CONTAINER:-qwen3-tts}"
HOST="0.0.0.0"
PORT="${QWEN_TTS_PORT:-8010}"
PID_FILE=".${CONTAINER_NAME}.pid"
LOG_FILE=".${CONTAINER_NAME}.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${HF_HOME_HOST:-${WORK_DIR}/.cache/huggingface}"
VOICES_DIR="${VOICES_DIR:-${WORK_DIR}/voices}"
READY_URL="http://127.0.0.1:${PORT}/health"

QWEN_TTS_BASE_MODEL="${QWEN_TTS_BASE_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
# Preset voices (en=aiden) use the 0.6B CustomVoice model so that both models
# fit side by side in one container (MAX_LOADED_MODELS=2): no ~30-68 s model
# swap between English and the cloned de/fr/es voices, and no second
# container with its own CUDA context and torch runtime (until 2026-10-09
# en ran in a separate qwen3-tts-small instance on :8012).
QWEN_TTS_CUSTOM_MODEL="${QWEN_TTS_CUSTOM_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
DEFAULT_VOICES="${DEFAULT_VOICES:-de=de-voiceclip,en=aiden,fr=developpeuse,es=es-villarraza}"
# Chunks (<= 400 chars each) generated at once. Peak memory grows with it; 8
# was the suspected trigger of the 2026-10-09 overnight OOM thrash on the
# Spark (a long English summary right as the LLM slowed to a crawl).
MAX_BATCH="${MAX_BATCH:-2}"
# 1: keep only the model in use (swap on demand); 2: keep both loaded
MAX_LOADED_MODELS="${MAX_LOADED_MODELS:-2}"
# OOM kill order on the Spark: TTS (+800) first, then qwen-image (+500), and
# sglang (-900) only when nothing else frees enough. GPU allocations don't
# count toward a process's OOM score here, so this setting decides the order.
# Restart after a crash at most twice, so an OOM-killed server doesn't keep
# reloading into a box that is short of memory; not restarted after a reboot
# or `docker stop` - start it again by hand.
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-800}"
RESTART_POLICY="${RESTART_POLICY:-on-failure:2}"

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
    echo "(for a second instance set QWEN_TTS_CONTAINER and QWEN_TTS_PORT)"
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
  echo "No nvidia-smi: building the CPU variant (smoke tests only, much slower than realtime)"
fi

echo "Building ${IMAGE} from ${WORK_DIR}"
docker build "${build_args[@]}" -t "${IMAGE}" "${WORK_DIR}"

echo "Starting Qwen3-TTS (base ${QWEN_TTS_BASE_MODEL}, custom ${QWEN_TTS_CUSTOM_MODEL})"
echo "Default voices: ${DEFAULT_VOICES}"
echo "Voices dir: ${VOICES_DIR}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching ${CONTAINER_NAME} container
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  "${gpu_args[@]}" \
  --restart "${RESTART_POLICY}" \
  --oom-score-adj "${OOM_SCORE_ADJ}" \
  -e PORT="${PORT}" \
  -e QWEN_TTS_BASE_MODEL="${QWEN_TTS_BASE_MODEL}" \
  -e QWEN_TTS_CUSTOM_MODEL="${QWEN_TTS_CUSTOM_MODEL}" \
  -e DEFAULT_VOICES="${DEFAULT_VOICES}" \
  -e MAX_BATCH="${MAX_BATCH}" \
  -e MAX_LOADED_MODELS="${MAX_LOADED_MODELS}" \
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
echo "(first run downloads ~4.5GB per model)"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "${CONTAINER_NAME} container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 10
done

echo "${CONTAINER_NAME} is ready"
echo "OpenAI-shaped base URL: http://${HOST}:${PORT}/v1"
