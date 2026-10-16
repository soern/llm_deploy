#!/usr/bin/env bash
set -euo pipefail

# QWEN_TTS_CONTAINER: as given to start.sh for a second instance
CONTAINER_NAME="${QWEN_TTS_CONTAINER:-qwen3-tts}"
PID_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.${CONTAINER_NAME}.pid"
LOG_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.${CONTAINER_NAME}.log"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping container ${CONTAINER_NAME}"
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
else
  echo "Container ${CONTAINER_NAME} not found"
fi

rm -f "${PID_FILE}"
echo "Stopped. Log preserved at ${LOG_FILE}"
