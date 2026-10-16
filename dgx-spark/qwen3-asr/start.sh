#!/usr/bin/env bash
set -euo pipefail

# Qwen3-ASR STT server launch script. Meant for the DGX Spark (GB10, ARM64),
# where it shares the GPU with the LLM, FLUX.2 and Qwen3-TTS; on a host
# without nvidia-smi it builds the CPU variant instead (smoke tests).
# Structured like deploy/dgx-spark/qwen3-tts/start.sh: named container, host
# networking, host-side HF cache bind mount, PID/log files next to this
# script, readiness poll with an exited-container check.
#
# Listens on 8001, the port nixclaw's stt.base_url already points at
# (faster-whisper-server's port), so it replaces that server without a config
# change. Only one of the two can run: stop faster-whisper first.

# QWEN_ASR_BACKEND=vllm: Dockerfile.vllm (GPU only), see README.
BACKEND="${QWEN_ASR_BACKEND:-transformers}"
case "${BACKEND}" in
  transformers) IMAGE="qwen3-asr:local"; DOCKERFILE="Dockerfile" ;;
  vllm) IMAGE="qwen3-asr-vllm:local"; DOCKERFILE="Dockerfile.vllm" ;;
  *) echo "QWEN_ASR_BACKEND must be transformers or vllm, got ${BACKEND}"; exit 1 ;;
esac
CONTAINER_NAME="${QWEN_ASR_CONTAINER:-qwen3-asr}"
HOST="0.0.0.0"
PORT="${QWEN_ASR_PORT:-8001}"
PID_FILE=".${CONTAINER_NAME}.pid"
LOG_FILE=".${CONTAINER_NAME}.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${HF_HOME_HOST:-${WORK_DIR}/.cache/huggingface}"
READY_URL="http://127.0.0.1:${PORT}/health"

QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-Qwen/Qwen3-ASR-0.6B}"
# OOM kill order on the Spark: TTS (+800) first, then qwen-image (+500), then
# this, and sglang (-900) last. Restart after a crash at most twice; not
# restarted after a reboot or `docker stop` - start it again by hand.
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-300}"
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
mkdir -p "${HF_HOME}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

# Port 8001 is faster-whisper's; with host networking a second server on it
# would just crash-loop on bind, so refuse up front.
if curl -fsS -o /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null \
   || { command -v ss >/dev/null 2>&1 && ss -ltnH "sport = :${PORT}" | grep -q .; }; then
  echo "Port ${PORT} is already in use (faster-whisper-server?)."
  echo "Stop it first, e.g.:"
  echo "  docker compose -f deploy/dgx-spark/faster-whisper/docker-compose.yml down"
  echo "  (or: docker rm -f faster-whisper)"
  echo "or run on another port: QWEN_ASR_PORT=8011 ./start.sh"
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  build_args=()
  gpu_args=(--gpus all)
  echo "GPU host: building on the NVIDIA PyTorch base image"
elif [[ "${BACKEND}" == "vllm" ]]; then
  echo "The vLLM backend needs a GPU (no nvidia-smi found)"
  exit 1
else
  build_args=(--build-arg BASE_IMAGE=python:3.12-slim --build-arg INSTALL_CPU_TORCH=1)
  gpu_args=()
  echo "No nvidia-smi: building the CPU variant (smoke tests only)"
fi

echo "Building ${IMAGE} from ${WORK_DIR}/${DOCKERFILE}"
docker build "${build_args[@]}" -f "${WORK_DIR}/${DOCKERFILE}" -t "${IMAGE}" "${WORK_DIR}"

echo "Starting Qwen3-ASR (${QWEN_ASR_MODEL}, ${BACKEND} backend)"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<LOG
[$(date -Is)] launching ${CONTAINER_NAME} container
LOG

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  "${gpu_args[@]}" \
  --restart "${RESTART_POLICY}" \
  --oom-score-adj "${OOM_SCORE_ADJ}" \
  -e PORT="${PORT}" \
  -e QWEN_ASR_MODEL="${QWEN_ASR_MODEL}" \
  -e QWEN_ASR_DTYPE="${QWEN_ASR_DTYPE:-}" \
  -e QWEN_ASR_MAX_NEW_TOKENS="${QWEN_ASR_MAX_NEW_TOKENS:-512}" \
  -e QWEN_ASR_BACKEND="${BACKEND}" \
  -e QWEN_ASR_VLLM_GPU_MEMORY_UTILIZATION="${QWEN_ASR_VLLM_GPU_MEMORY_UTILIZATION:-0.07}" \
  -e QWEN_ASR_VLLM_MAX_MODEL_LEN="${QWEN_ASR_VLLM_MAX_MODEL_LEN:-4096}" \
  -e QWEN_ASR_VLLM_ENFORCE_EAGER="${QWEN_ASR_VLLM_ENFORCE_EAGER:-0}" \
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
echo "(first run downloads ~1.9GB)"
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
