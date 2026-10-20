#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen3.8-flash-fp4}"
PID_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.vllm.pid"
LOG_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.vllm.log"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping container ${CONTAINER_NAME}"
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
else
  echo "Container ${CONTAINER_NAME} not found"
fi

rm -f "${PID_FILE}"
echo "Stopped. Log preserved at ${LOG_FILE}"
