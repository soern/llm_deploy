#!/usr/bin/env bash
set -euo pipefail

# Ollama launch script for Qwen/Qwen3-4B-Instruct-2507 on a CPU-only host.
# See README.md for model sourcing, quant choice, and RAM/CPU sizing notes.
#
# Plain `docker run` against the official upstream Ollama image (no custom
# build), same convention as the other deploy/ scripts in this repo. Model
# weights are pulled at first start via `ollama pull` inside the running
# container and cached in a bind-mounted volume, so restarts are instant.

MODEL_TAG="${MODEL_TAG:-qwen3:4b-instruct-2507-q4_K_M}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_TAG}}"
IMAGE="${IMAGE:-ollama/ollama:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-ollama-qwen3-4b-cpu}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-11434}"

# Ollama's default CPU thread count is derived from the host at runtime; set
# OLLAMA_NUM_THREADS to pin it (e.g. to leave headroom for other services).
OLLAMA_NUM_THREADS="${OLLAMA_NUM_THREADS:-}"
OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
# Ollama's runtime context window defaults to 4096 tokens regardless of a
# model's native max context (this model supports 262144), and there is no
# per-request way to override it through the OpenAI-compatible /v1 API used
# by VLLMClient-style callers -- only this server-wide env var (or a
# Modelfile) controls it. 32768 is a reasonable CPU-friendly middle ground;
# raise it if you hit "exceeds the available context size" 400s with long
# conversations, but note KV-cache RAM cost scales with this value.
OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-32768}"
# Ollama unloads an idle model from memory after OLLAMA_KEEP_ALIVE (default
# 5m) unless a request's own "keep_alive" field overrides it -- nixclaw's
# clients don't send one, so this server default always applies. Set to
# 24h so the model stays resident and avoids a multi-second reload on the
# next request after any idle gap; use "-1" for "never unload".
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"

PID_FILE=".ollama.pid"
LOG_FILE=".ollama.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${WORK_DIR}/.ollama"
READY_URL="http://127.0.0.1:${PORT}/api/version"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${DATA_DIR}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting Ollama container (CPU-only) for ${MODEL_TAG}"
echo "Image: ${IMAGE}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching Ollama container (CPU-only, ${MODEL_TAG})
EOF

DOCKER_ENV_ARGS=(
  -e "OLLAMA_HOST=${HOST}:${PORT}"
  -e "OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS}"
  -e "OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH}"
  -e "OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}"
)
if [[ -n "${OLLAMA_NUM_THREADS}" ]]; then
  DOCKER_ENV_ARGS+=(-e "OLLAMA_NUM_THREADS=${OLLAMA_NUM_THREADS}")
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  "${DOCKER_ENV_ARGS[@]}" \
  -v "${DATA_DIR}:/root/.ollama" \
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
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Ollama container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 2
done

echo "Ollama daemon is ready, pulling ${MODEL_TAG} (first run only, may take a while)..."
if ! docker exec "${CONTAINER_NAME}" ollama pull "${MODEL_TAG}" 2>&1 | tee -a "${LOG_FILE}"; then
  echo "Model pull failed -- see ${LOG_FILE}"
  exit 1
fi

echo "Model ${MODEL_TAG} ready"
echo "OpenAI/Ollama base URL: http://${HOST}:${PORT}"
echo "Set config/agent.yaml: ollama.host: \"127.0.0.1:${PORT}\", ollama.model: \"${SERVED_MODEL_NAME}\""
