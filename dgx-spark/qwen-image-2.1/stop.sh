#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="qwen-image-2.1"
PID_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.qwen-image.pid"
LOG_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.qwen-image.log"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping Qwen-Image-2.1 container"
  docker stop "${CONTAINER_NAME}" 2>/dev/null || true
  docker rm "${CONTAINER_NAME}"
else
  echo "Container ${CONTAINER_NAME} not found"
fi

rm -f "${PID_FILE}"
echo "Stopped. Log preserved at ${LOG_FILE}"
